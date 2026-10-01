"""
실시간 드론 운용 시뮬레이터 (경량) — 업체 정사영상을 "가상 세계"로 쓴다.

문갑도 정사영상(MGD.tif, 3.02 cm/px, EPSG:5186, 2.3 GB)을 통째로 올리지 않고
rasterio 윈도우 읽기(오버뷰 활용)만으로, 가상 드론이 그 위를 비행하면서
    ① 미션 계획(지그재그 커버리지) → ② 비행(시간 스텝·바람 잡음)
    → ③ 가상 카메라 프레임(고도에 따라 GSD 변화, 회전·블러·노출 잡음)
    → ④ YOLO 탐지(CPU 고정) → ⑤ 지도 투영 + 클러스터링(중복 제거)
    → ⑥ 능동 재방문(애매한 후보를 저고도로 다시 찍어 확정/기각)
    → ⑦ 업체 라벨로 재현율·정밀도
를 한 번에 돌린다.

ROS 2/Gazebo로 옮길 수 있게 역할을 클래스로 분리했다:
    VirtualWorld  : 정사영상 샘플러 (Gazebo 월드 + 카메라 플러그인 역할)
    Camera        : 화각·해상도 → GSD, 프레임 픽셀 ↔ 지도 좌표
    FlightSim     : 기체 운동 (속도·바람·상승률) → 시간 스텝마다 Pose
    render_frame  : 센서 프레임 생성   (sensor_msgs/Image)
    Detector      : YOLO 탐지          (vision_msgs/Detection2DArray)
    Mapper        : 투영 + 클러스터링 누적 지도 (후보/확정/기각 상태)
    Planner       : 커버리지 경로 + 재방문 끼워넣기

  python -m litter.sim_ortho --name demo                       # 라벨 밀집 100×100 m 자동 선택
  python -m litter.sim_ortho --name demo --model aihub --bbox 120820 508000 120920 508100
  python -m litter.sim_ortho --name demo --no-revisit         # 커버리지만

결과: runs/sim/<name>/ map.png · map.html · flight.mp4 · flight.gif · detections.csv · objects.csv · summary.json
"""
import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .seg import _imwrite
from .video_map import _thumb

ROOT = Path(__file__).resolve().parents[1]
ORTHO = ROOT / "data" / "company" / "ortho" / "MGD.tif"
LABELS = ROOT / "data" / "company" / "MGD_labels.json"
WEIGHTS = {"uavvaste": ROOT / "runs" / "seg" / "uavvaste_det_s" / "weights" / "best.pt",
           "aihub": ROOT / "runs" / "seg" / "aihub_gsd_det_s" / "weights" / "best.pt"}


# ================================================================ 세계 (정사영상)
class VirtualWorld:
    """정사영상을 윈도우 단위로만 읽는다. 축소 읽기는 GeoTIFF 오버뷰를 써서 빠르다."""

    def __init__(self, ortho=ORTHO):
        import rasterio

        self.ds = rasterio.open(str(ortho))
        self.T = self.ds.transform
        self.inv = ~self.T
        self.res = abs(self.T.a)
        self.crs = self.ds.crs

    def to_px(self, x, y):
        c, r = self.inv * (np.asarray(x, float), np.asarray(y, float))
        return np.asarray(c), np.asarray(r)

    def read_px(self, col0, row0, w, h, out_w, out_h):
        """정사영상 픽셀 창 (col0,row0,w,h) → (out_h,out_w,3) BGR. 바깥은 검정."""
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        a = self.ds.read([1, 2, 3], window=Window(col0, row0, w, h), out_shape=(3, out_h, out_w),
                         boundless=True, fill_value=0, resampling=Resampling.bilinear)
        return np.ascontiguousarray(np.transpose(a, (1, 2, 0))[:, :, ::-1])

    def read_area(self, bbox, px_w):
        """지도 bbox(xmin,ymin,xmax,ymax)를 가로 px_w 픽셀로 (개요도·배경용). 반환 (img, gsd)."""
        xmin, ymin, xmax, ymax = bbox
        c0, r0 = self.to_px(xmin, ymax)
        c1, r1 = self.to_px(xmax, ymin)
        w, h = int(c1 - c0), int(r1 - r0)
        px_h = max(1, int(round(px_w * h / w)))
        return self.read_px(int(c0), int(r0), w, h, px_w, px_h), (xmax - xmin) / px_w


# ================================================================ 카메라 · 자세
@dataclass
class Camera:
    width: int = 1024
    height: int = 768
    hfov_deg: float = 73.7   # DJI 24 mm 환산 (36 mm 폭 기준)

    def gsd(self, alt):
        return 2 * alt * math.tan(math.radians(self.hfov_deg) / 2) / self.width

    def footprint(self, alt):
        g = self.gsd(alt)
        return self.width * g, self.height * g

    def frame_to_map(self, pose, uv):
        """프레임 픽셀 (N,2) → 지도 좌표 (N,2). 수직(nadir) 카메라, 화면 위쪽 = 기수 방향."""
        uv = np.atleast_2d(np.asarray(uv, float))
        g = self.gsd(pose.alt)
        th = math.radians(pose.yaw)
        du, dv = (uv[:, 0] - self.width / 2) * g, (uv[:, 1] - self.height / 2) * g
        x = pose.x + math.cos(th) * du - math.sin(th) * dv
        y = pose.y - math.sin(th) * du - math.cos(th) * dv
        return np.stack([x, y], 1)

    def footprint_poly(self, pose):
        W, H = self.width, self.height
        return self.frame_to_map(pose, [[0, 0], [W, 0], [W, H], [0, H]])


@dataclass
class Pose:
    x: float
    y: float
    alt: float
    yaw: float      # 북 기준 시계방향 (도)
    t: float = 0.0


# ================================================================ 센서 프레임 생성
def render_frame(world, cam, pose, jitter=0.0, rng=None):
    """현재 자세의 카메라 시야를 정사영상에서 잘라 리샘플 → 가상 프레임(BGR).
    고도가 낮을수록 GSD가 작아지고(확대), 기수 방향으로 회전한다.
    jitter>0: 블러·노출·잡음을 조금 섞는다 (실제 촬영 흉내)."""
    rng = rng or np.random.default_rng(0)
    W, H = cam.width, cam.height
    g = cam.gsd(pose.alt)
    scale = g / world.res                       # 출력 1픽셀이 정사영상 몇 픽셀인지
    corners = cam.frame_to_map(pose, [[0, 0], [W, 0], [W, H], [0, H]])
    c, r = world.to_px(corners[:, 0], corners[:, 1])
    c0, r0 = int(math.floor(c.min())) - 2, int(math.floor(r.min())) - 2
    c1, r1 = int(math.ceil(c.max())) + 2, int(math.ceil(r.max())) + 2
    ow, oh = max(2, int(round((c1 - c0) / scale))), max(2, int(round((r1 - r0) / scale)))
    A = world.read_px(c0, r0, c1 - c0, r1 - r0, ow, oh)
    sx, sy = (c1 - c0) / ow, (r1 - r0) / oh    # A 픽셀 → 정사영상 픽셀 배율
    # 프레임 세 꼭짓점 → A 픽셀 → 아핀 변환
    fp = np.array([[0, 0], [W, 0], [0, H]], np.float32)
    mp = cam.frame_to_map(pose, fp)
    pc, pr = world.to_px(mp[:, 0], mp[:, 1])
    ap = np.stack([(pc - c0) / sx, (pr - r0) / sy], 1).astype(np.float32)
    M = cv2.getAffineTransform(ap, fp)
    img = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if jitter > 0:
        s = float(rng.uniform(0, 1.2 * jitter))
        if s > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), s)
        gain = float(rng.uniform(1 - 0.25 * jitter, 1 + 0.25 * jitter))
        bias = float(rng.normal(0, 10 * jitter))
        img = np.clip(img.astype(np.float32) * gain + bias + rng.normal(0, 3 * jitter, img.shape), 0, 255).astype(np.uint8)
    return img


# ================================================================ 비행 역학
class FlightSim:
    """단순 운동 모델: 목표점으로 speed로 직진, 바람(정상풍 + 돌풍 랜덤워크)이 위치를 밀고,
    고도는 climb_rate로 바뀐다. 매 dt마다 Pose를 내놓는다 (ROS에선 /odom 대신)."""

    def __init__(self, x, y, alt, speed=5.0, dt=1.0, wind=(0.0, 0.0), gust=0.0, climb_rate=3.0,
                 gps_noise=0.0, yaw_noise=0.0, seed=0):
        self.x, self.y, self.alt, self.yaw, self.t = x, y, alt, 0.0, 0.0
        self.speed, self.dt, self.climb = speed, dt, climb_rate
        self.wind, self.gust, self.gv = np.asarray(wind, float), gust, np.zeros(2)
        self.gps_noise, self.yaw_noise = gps_noise, yaw_noise
        self.rng = np.random.default_rng(seed)
        self.dist = 0.0
        self.track = []   # (t, x, y, alt, phase)

    def _pose(self):
        n = self.rng.normal(0, self.gps_noise, 2) if self.gps_noise else (0, 0)
        yj = self.rng.normal(0, self.yaw_noise) if self.yaw_noise else 0.0
        return Pose(self.x + n[0], self.y + n[1], self.alt, (self.yaw + yj) % 360, self.t)

    def _step_alt(self, alt_t):
        d = alt_t - self.alt
        mv = np.clip(d, -self.climb * self.dt, self.climb * self.dt)
        self.alt += mv
        return abs(mv)

    def fly_to(self, target, alt_t, phase="coverage"):
        """목표 (x,y)·고도까지 비행하며 매 스텝 Pose를 yield. 바람은 기체를 밀지만
        조종기(제어기)가 70%는 보정한다고 가정 → 30%만 위치에 반영."""
        tx, ty = target
        while True:
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            step = self.speed * self.dt
            if d > 1e-6:
                self.yaw = math.degrees(math.atan2(dx, dy)) % 360
            if d <= step and abs(alt_t - self.alt) < 1e-6:
                self.dist += d
                self.x, self.y = tx, ty
                self.t += self.dt * d / max(step, 1e-9)
                self.track.append((self.t, self.x, self.y, self.alt, phase))
                yield self._pose()
                return
            # 수평 이동
            if d > 1e-6:
                mv = min(step, d)
                ux, uy = dx / d, dy / d
                self.gv = 0.8 * self.gv + self.rng.normal(0, self.gust, 2) if self.gust else self.gv
                drift = (self.wind + self.gv) * self.dt * 0.3
                nx, ny = self.x + ux * mv + drift[0], self.y + uy * mv + drift[1]
                self.dist += math.hypot(nx - self.x, ny - self.y)
                self.x, self.y = nx, ny
            dz = self._step_alt(alt_t)
            self.dist += dz if d <= 1e-6 else 0.0
            self.t += self.dt
            self.track.append((self.t, self.x, self.y, self.alt, phase))
            yield self._pose()

    def hover(self, n, phase="revisit", wander_m=0.8):
        """제자리 촬영 n장 — 매번 조금씩 옮기고(wander_m) 기수를 돌려서 다른 시점으로 본다."""
        x0, y0 = self.x, self.y
        for k in range(n):
            if k:
                a = 2 * math.pi * k / n
                nx, ny = x0 + wander_m * math.cos(a), y0 + wander_m * math.sin(a)
                self.dist += math.hypot(nx - self.x, ny - self.y)
                self.x, self.y = nx, ny
                self.yaw = (self.yaw + 360.0 / n) % 360
            self.t += self.dt
            self.track.append((self.t, self.x, self.y, self.alt, phase))
            yield self._pose()
        self.x, self.y = x0, y0


# ================================================================ 탐지
class Detector:
    """YOLO 탐지 — **CPU 고정** (GPU는 COLMAP 등 다른 작업이 쓰는 중)."""

    def __init__(self, weights, conf=0.25, imgsz=1024, threads=4):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)
        self.m = YOLO(str(weights))
        self.names = self.m.names
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame):
        r = self.m.predict(frame, conf=self.conf, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        out = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append({"box": b.astype(float), "cls": self.names[int(c)], "score": float(s)})
        return out


# ================================================================ 지도화 (투영 + 클러스터링)
@dataclass
class MapObject:
    oid: int
    x: float
    y: float
    cls: str
    w_m: float
    h_m: float
    cov_max: float = 0.0       # 커버리지 프레임에서의 최고 신뢰도
    rv_max: float = -1.0       # 재방문 프레임에서의 최고 신뢰도 (-1 = 재방문 안 함)
    n_views: int = 0
    first_frame: int = -1
    first_phase: str = "coverage"
    revisited: bool = False
    thumb: str = None
    obs: list = field(default_factory=list)
    names: list = field(default_factory=list)

    @property
    def max_score(self):
        return max(self.cov_max, self.rv_max)

    def status(self, lo, hi, with_revisit=True):
        """confirmed / candidate / rejected. with_revisit=False면 커버리지 프레임 점수만 본다
        (저고도에서만 보인 물체는 None — 커버리지 전용 비교에서 제외)."""
        if not with_revisit:
            if self.cov_max >= hi:
                return "confirmed"
            return "candidate" if self.cov_max >= lo else None
        if self.max_score >= hi:
            return "confirmed"
        if self.revisited:
            return "rejected"
        return "candidate" if self.max_score >= lo else "rejected"


class Mapper:
    """프레임 탐지 → 지도 좌표 → 가까운 기존 물체에 합치기(가중 평균) 또는 새 물체.
    같은 물체가 여러 프레임에 보이면 n_views가 올라간다 (추적 역할)."""

    def __init__(self, cam, merge_m=1.0):
        self.cam, self.merge_m = cam, merge_m
        self.objects = []
        self.rows = []   # detections.csv

    def update(self, frame_id, phase, pose, frame, dets):
        g = self.cam.gsd(pose.alt)
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            cx, cy = self.cam.frame_to_map(pose, [[(x0 + x1) / 2, (y0 + y1) / 2]])[0]
            w_m, h_m = (x1 - x0) * g, (y1 - y0) * g
            sc = d["score"]
            best, bd = None, self.merge_m
            for o in self.objects:
                dd = math.hypot(o.x - cx, o.y - cy)
                if dd < bd:
                    best, bd = o, dd
            if best is None:
                best = MapObject(len(self.objects) + 1, cx, cy, d["cls"], w_m, h_m, first_frame=frame_id, first_phase=phase)
                self.objects.append(best)
            else:
                wsum = sum(s for _, _, s, _, _ in best.obs) + 1e-6
                best.x = (best.x * wsum + cx * sc) / (wsum + sc)
                best.y = (best.y * wsum + cy * sc) / (wsum + sc)
                best.w_m, best.h_m = max(best.w_m, w_m), max(best.h_m, h_m)
            best.n_views += 1
            best.names.append(d["cls"])
            best.obs.append((frame_id, phase, sc, cx, cy))
            if phase == "revisit":
                best.rv_max = max(best.rv_max, sc)
            else:
                best.cov_max = max(best.cov_max, sc)
            if best.thumb is None or sc >= best.max_score:
                best.thumb = _thumb(frame, d["box"], 140)
            best.cls = max(set(best.names), key=best.names.count)
            self.rows.append([frame_id, phase, f"{pose.t:.1f}", f"{pose.alt:.1f}", f"{g * 100:.2f}", d["cls"], f"{sc:.3f}",
                              f"{cx:.2f}", f"{cy:.2f}", f"{w_m:.2f}", f"{h_m:.2f}", best.oid])

    def candidates(self, lo, hi):
        return [o for o in self.objects if o.first_phase == "coverage" and not o.revisited and lo <= o.cov_max < hi]


# ================================================================ 계획
class Planner:
    @staticmethod
    def coverage(bbox, cam, alt, overlap_side=0.3):
        """지그재그(boustrophedon) 경로. 긴 변 방향으로 왕복, 줄 간격 = 횡방향 발자국 × (1-겹침)."""
        xmin, ymin, xmax, ymax = bbox
        fw, fh = cam.footprint(alt)
        along_x = (xmax - xmin) >= (ymax - ymin)
        across = fh if along_x else fh  # 기수가 진행방향이므로 횡방향은 항상 세로 발자국
        spacing = max(across * (1 - overlap_side), 1.0)
        if along_x:
            lo, hi, a0, a1 = ymin, ymax, xmin, xmax
        else:
            lo, hi, a0, a1 = xmin, xmax, ymin, ymax
        half = across / 2
        n = max(1, int(math.ceil((hi - lo - across) / spacing)) + 1)
        ys = [lo + half + k * spacing for k in range(n)]
        if ys[-1] + half < hi - 0.5:
            ys.append(hi - half)
        wps = []
        for k, y in enumerate(ys):
            a, b = (a0, a1) if k % 2 == 0 else (a1, a0)
            wps += [((a, y) if along_x else (y, a)), ((b, y) if along_x else (y, b))]
        return wps, {"lines": len(ys), "spacing_m": spacing, "footprint_m": [fw, fh]}

    @staticmethod
    def order_revisits(cands, start):
        """현재 위치에서 최근접 이웃 순서."""
        left, cur, out = list(cands), np.asarray(start, float), []
        while left:
            j = min(range(len(left)), key=lambda i: math.hypot(left[i].x - cur[0], left[i].y - cur[1]))
            o = left.pop(j)
            out.append(o)
            cur = np.array([o.x, o.y])
        return out


# ================================================================ 라벨 · 평가
def load_labels(world, path=LABELS):
    """업체 라벨(위경도 사각형) → 지도 좌표 폴리곤 + 중심."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    T = Transformer.from_crs(4326, world.crs, always_xy=True)
    out = []
    for f in d["features"]:
        ring = np.asarray(f["geometry"]["coordinates"][0], float)
        mx, my = T.transform(ring[:, 0], ring[:, 1])
        poly = Polygon(np.stack([mx, my], 1))
        p = f["properties"]
        out.append({"seq": p.get("detection_seq"), "material": p.get("material_code"), "area_sqm": p.get("area_sqm"),
                    "poly": poly, "x": poly.centroid.x, "y": poly.centroid.y, "ring_ll": ring.tolist()})
    return out


def densest_bbox(labels, size=100.0, step=5.0):
    """라벨이 가장 많이 들어가는 size×size 창을 찾고, 그 안 라벨 중심으로 다시 맞춘다."""
    P = np.array([[l["x"], l["y"]] for l in labels])
    best = None
    for cx in np.arange(P[:, 0].min(), P[:, 0].max() + step, step):
        for cy in np.arange(P[:, 1].min(), P[:, 1].max() + step, step):
            n = int(((abs(P[:, 0] - cx) < size / 2) & (abs(P[:, 1] - cy) < size / 2)).sum())
            if best is None or n > best[0]:
                best = (n, cx, cy)
    _, cx, cy = best
    m = (abs(P[:, 0] - cx) < size / 2) & (abs(P[:, 1] - cy) < size / 2)
    cx, cy = P[m].mean(0)
    return (float(cx - size / 2), float(cy - size / 2), float(cx + size / 2), float(cy + size / 2))


def evaluate(objects, labels, bbox, lo, hi, tol_m=1.0):
    """재현율 = 맞힌 라벨 / 영역 안 라벨. 정밀도 = 라벨과 겹친 확정 물체 / 확정 물체
    (정사영상엔 미라벨 쓰레기가 많아 정밀도는 참고용). 물체 중심이 라벨 사각형+tol 안이면 적중."""
    from shapely.geometry import Point, box

    B = box(*bbox)
    labs = [l for l in labels if B.contains(Point(l["x"], l["y"]))]
    res = {"labels_in_area": len(labs)}
    for key, with_rv in (("coverage_only", False), ("with_revisit", True)):
        conf = [o for o in objects if o.status(lo, hi, with_rv) == "confirmed"]
        cand = [o for o in objects if o.status(lo, hi, with_rv) == "candidate"]
        n_rej = sum(1 for o in objects if o.status(lo, hi, with_rv) == "rejected")
        used, tp = set(), 0
        hit_labels = set()
        for o in sorted(conf, key=lambda o: -o.max_score):
            p = Point(o.x, o.y)
            for j, l in enumerate(labs):
                if j in used:
                    continue
                if l["poly"].buffer(tol_m).contains(p):
                    used.add(j)
                    hit_labels.add(l["seq"])
                    tp += 1
                    break
        # 후보(미확정)까지 포함했을 때 (상한 참고치)
        used2 = set()
        for o in sorted(conf + cand, key=lambda o: -o.max_score):
            p = Point(o.x, o.y)
            for j, l in enumerate(labs):
                if j not in used2 and l["poly"].buffer(tol_m).contains(p):
                    used2.add(j)
                    break
        res[key] = {"confirmed": len(conf), "candidates": len(cand), "rejected": n_rej, "tp": tp, "recall": tp / max(len(labs), 1), "precision": tp / max(len(conf), 1),
                    "recall_incl_candidates": len(used2) / max(len(labs), 1),
                    "hit_label_seqs": sorted(hit_labels)}
    return res, labs


# ================================================================ 시각화
def _font(size):
    from PIL import ImageFont

    for p in (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\gulim.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _put_text(img, text, xy, size=20, color=(255, 255, 0)):
    """한글 가능한 텍스트 (PIL)."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[:, :, ::-1])
    ImageDraw.Draw(pil).text(xy, text, font=_font(size), fill=color[::-1], stroke_width=2, stroke_fill=(0, 0, 0))
    return np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])


STATUS_COLOR = {"confirmed": (0, 0, 255), "candidate": (0, 200, 255), "rejected": (160, 160, 160)}


class HUD:
    """드론 시점 프레임 + 미니맵(경로·발자국·물체·라벨)."""

    def __init__(self, world, bbox, cam, lo, hi, size=260, margin=12.0):
        self.cam, self.lo, self.hi = cam, lo, hi
        self.bbox = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
        self.base, self.gsd = world.read_area(self.bbox, size)
        self.base = cv2.convertScaleAbs(self.base, alpha=0.75)
        self.area = bbox

    def to_px(self, x, y):
        return int((x - self.bbox[0]) / self.gsd), int((self.bbox[3] - y) / self.gsd)

    def draw(self, frame, pose, dets, phase, track, objects, labels, frame_id, n_rv):
        vis = frame.copy()
        for d in dets:
            x0, y0, x1, y1 = map(int, d["box"])
            col = (0, 0, 255) if d["score"] >= self.hi else (0, 200, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.putText(vis, f"{d['score']:.2f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        g = self.cam.gsd(pose.alt) * 100
        label = {"coverage": "커버리지", "transit": "재방문 이동", "revisit": "재방문 저고도"}[phase]
        vis = _put_text(vis, f"t={pose.t:5.0f}s  고도 {pose.alt:4.1f} m  GSD {g:.1f} cm/px  기수 {pose.yaw:3.0f}°  "
                             f"[{label}]  프레임 {frame_id}  탐지 {len(dets)}", (10, 8), 20)
        # 미니맵
        mm = self.base.copy()
        x0, y0 = self.to_px(self.area[0], self.area[3])
        x1, y1 = self.to_px(self.area[2], self.area[1])
        cv2.rectangle(mm, (x0, y0), (x1, y1), (255, 255, 255), 1)
        for l in labels:
            lx0, ly0 = self.to_px(*l["poly"].bounds[0:4:3])
            lx1, ly1 = self.to_px(l["poly"].bounds[2], l["poly"].bounds[1])
            cv2.rectangle(mm, (lx0 - 1, ly0 - 1), (lx1 + 1, ly1 + 1), (255, 120, 0), 1)
        pts = [self.to_px(x, y) for _, x, y, _, _ in track]
        if len(pts) > 1:
            cv2.polylines(mm, [np.array(pts, np.int32)], False, (255, 255, 255), 1)
        fp = self.cam.footprint_poly(pose)
        cv2.polylines(mm, [np.array([self.to_px(x, y) for x, y in fp], np.int32)], True, (0, 255, 255), 1)
        for o in objects:
            st = o.status(self.lo, self.hi)
            cv2.circle(mm, self.to_px(o.x, o.y), 3, STATUS_COLOR[st], -1)
        cv2.circle(mm, self.to_px(pose.x, pose.y), 4, (0, 255, 0), -1)
        n_conf = sum(1 for o in objects if o.status(self.lo, self.hi) == "confirmed")
        n_cand = sum(1 for o in objects if o.status(self.lo, self.hi) == "candidate")
        mm = _put_text(mm, f"확정 {n_conf} · 후보 {n_cand} · 재방문 {n_rv}", (4, mm.shape[0] - 24), 15, (255, 255, 255))
        h, w = mm.shape[:2]
        H, W = vis.shape[:2]
        vis[H - h - 10:H - 10, W - w - 10:W - 10] = mm
        cv2.rectangle(vis, (W - w - 10, H - h - 10), (W - 10, H - 10), (255, 255, 255), 1)
        return vis


def write_map_png(out, world, bbox, track, objects, labels, lo, hi, title):
    from pipeline._compat import apply_korean_font
    apply_korean_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    m = 10.0
    B = (bbox[0] - m, bbox[1] - m, bbox[2] + m, bbox[3] + m)
    img, _ = world.read_area(B, 1400)
    fig, ax = plt.subplots(figsize=(10, 10 * img.shape[0] / img.shape[1]))
    ax.imshow(img[:, :, ::-1], extent=(B[0], B[2], B[1], B[3]))
    ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, ec="w", lw=1.2, ls="--"))
    T = np.array([[x, y] for _, x, y, _, _ in track])
    ph = np.array([p for *_, p in track])
    ax.plot(T[:, 0], T[:, 1], "-", c="#1d3557", lw=1.2, label="비행 경로(커버리지)")
    rv = ph != "coverage"
    if rv.any():
        ax.plot(T[rv, 0], T[rv, 1], ".", c="#ff7f0e", ms=3, label="재방문 구간")
    for l in labels:
        x0, y0, x1, y1 = l["poly"].bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1f77ff", lw=1.5))
    ax.plot([], [], "s", mfc="none", mec="#1f77ff", label=f"업체 라벨 ({len(labels)})")
    col = {"confirmed": "#e63946", "candidate": "#ffd166", "rejected": "#999999"}
    nm = {"confirmed": "확정", "candidate": "후보(미확정)", "rejected": "기각"}
    for st in ("rejected", "candidate", "confirmed"):
        P = np.array([[o.x, o.y] for o in objects if o.status(lo, hi) == st])
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], s=28, c=col[st], edgecolors="k", lw=.4, zorder=4, label=f"{nm[st]} ({len(P)})")
    ax.set_xlim(B[0], B[2])
    ax.set_ylim(B[1], B[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("E (m, EPSG:5186)")
    ax.set_ylabel("N (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_map_html(path, world, track, objects, labels, lo, hi, note):
    """Leaflet 지도 (orbit_map/video_map 방식) — 경로·라벨·물체(확정/후보/기각)·썸네일."""
    from pyproj import Transformer

    inv = Transformer.from_crs(world.crs, 4326, always_xy=True)
    tr = []
    for t, x, y, a, p in track:
        lon, lat = inv.transform(x, y)
        tr.append([round(lat, 7), round(lon, 7), round(a, 1), p])
    objs = []
    for o in objects:
        lon, lat = inv.transform(o.x, o.y)
        objs.append({"id": o.oid, "cls": o.cls, "lat": lat, "lon": lon, "n": o.n_views, "score": o.max_score,
                     "cov": o.cov_max, "rv": o.rv_max, "status": o.status(lo, hi), "thumb": o.thumb,
                     "w": round(o.w_m, 2), "h": round(o.h_m, 2)})
    labs = [{"seq": l["seq"], "mat": l["material"], "ring": [[p[1], p[0]] for p in l["ring_ll"]]} for l in labels]
    data = {"track": tr, "objs": objs, "labels": labs}
    html = """<!doctype html><meta charset="utf-8"><title>드론 시뮬레이션 탐지 지도</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}#n{position:absolute;z-index:999;left:50px;top:8px;background:#fffd;padding:6px 10px;border-radius:6px;font:13px sans-serif;max-width:520px}</style>
<div id="m"></div><div id="n">__NOTE__<br>
<span style="color:#e63946">●</span> 확정 <span style="color:#e6a700">●</span> 후보(미확정) <span style="color:#999">●</span> 기각
<span style="color:#1f77ff">▭</span> 업체 라벨 <span style="color:#1d3557">━</span> 커버리지 경로 <span style="color:#ff7f0e">━</span> 재방문</div>
<script>
const D=__DATA__;
const m=L.map('m',{maxZoom:23});
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:23,maxNativeZoom:19,attribution:'Esri'}).addTo(m);
const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:23,maxNativeZoom:19,attribution:'OSM'});
L.control.layers({'위성':sat,'지도':osm}).addTo(m);
let seg=[],segs=[];let cur=null;
D.track.forEach(t=>{const ph=t[3]==='coverage'?'c':'r';if(cur!==ph&&seg.length){segs.push([cur,seg]);seg=[seg[seg.length-1]];}cur=ph;seg.push([t[0],t[1]]);});
if(seg.length)segs.push([cur,seg]);
segs.forEach(([ph,s])=>L.polyline(s,{color:ph==='c'?'#1d3557':'#ff7f0e',weight:ph==='c'?2:3}).addTo(m));
L.circleMarker(D.track[0].slice(0,2),{radius:6,color:'#2a9d8f',fillOpacity:1}).bindPopup('출발').addTo(m);
D.labels.forEach(l=>L.polygon(l.ring,{color:'#1f77ff',weight:2,fill:false}).bindPopup(`업체 라벨 #${l.seq} (${l.mat})`).addTo(m));
const C={confirmed:'#e63946',candidate:'#e6a700',rejected:'#999999'};
const N={confirmed:'확정',candidate:'후보(미확정)',rejected:'기각'};
D.objs.forEach(o=>{const c=C[o.status];
 L.circleMarker([o.lat,o.lon],{radius:o.status==='rejected'?4:7,color:c,fillColor:c,fillOpacity:.85}).bindPopup(
  `<b>#${o.id} ${o.cls}</b> — ${N[o.status]}<br>${o.n}프레임 · 최고 신뢰도 ${o.score.toFixed(2)} (커버리지 ${o.cov.toFixed(2)}${o.rv>=0?' · 재방문 '+o.rv.toFixed(2):''})<br>`+
  `크기 ${o.w}×${o.h} m<br>${o.lat.toFixed(6)}, ${o.lon.toFixed(6)}<br>`+(o.thumb?`<img src="data:image/jpeg;base64,${o.thumb}">`:'')).addTo(m);});
const all=D.track.map(t=>[t[0],t[1]]);
m.fitBounds(L.latLngBounds(all).pad(0.2));
</script>"""
    Path(path).write_text(html.replace("__DATA__", json.dumps(data)).replace("__NOTE__", note), encoding="utf-8")


# ================================================================ 메인 루프
def run(name, model="uavvaste", bbox=None, size=100.0, alt=20.0, alt_low=8.0, speed=5.0, dt=1.0,
        overlap=0.3, conf_lo=0.25, conf_hi=0.5, merge_m=1.0, revisit=True, rv_frames=3, wind=(0.0, 0.0),
        gust=0.0, jitter=0.3, gps_noise=0.0, yaw_noise=1.0, cam=None, imgsz=1024, max_frames=1500,
        tol_m=1.0, seed=0, out_root=ROOT / "runs" / "sim", ortho=ORTHO, labels_path=LABELS, gif_every=1, gif_max=120):
    t_wall = time.time()
    out = Path(out_root) / name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cam = cam or Camera()
    world = VirtualWorld(ortho)
    labels_all = load_labels(world, labels_path)
    if bbox is None:
        bbox = densest_bbox(labels_all, size)
    bbox = tuple(float(v) for v in bbox)
    from shapely.geometry import Point, box as sbox
    B = sbox(*bbox)
    labels = [l for l in labels_all if B.contains(Point(l["x"], l["y"]))]
    weights = WEIGHTS.get(model, Path(model))
    print(f"영역 {bbox[2]-bbox[0]:.0f}×{bbox[3]-bbox[1]:.0f} m  E {bbox[0]:.1f}~{bbox[2]:.1f} N {bbox[1]:.1f}~{bbox[3]:.1f}"
          f" · 업체 라벨 {len(labels)}개 · 모델 {weights.name if weights.exists() else model} (CPU)")

    det = Detector(weights, conf=conf_lo, imgsz=imgsz)
    mapper = Mapper(cam, merge_m)
    wps, plan_info = Planner.coverage(bbox, cam, alt, overlap)
    fw, fh = cam.footprint(alt)
    print(f"계획: 고도 {alt} m · 발자국 {fw:.1f}×{fh:.1f} m · GSD {cam.gsd(alt)*100:.1f} cm/px · 줄 {plan_info['lines']}개"
          f" (간격 {plan_info['spacing_m']:.1f} m) · 재방문 고도 {alt_low} m (GSD {cam.gsd(alt_low)*100:.1f} cm/px)")
    flight = FlightSim(wps[0][0], wps[0][1], alt, speed, dt, wind, gust, gps_noise=gps_noise, yaw_noise=yaw_noise, seed=seed)
    rng = np.random.default_rng(seed + 1)
    hud = HUD(world, bbox, cam, conf_lo, conf_hi)
    vw = cv2.VideoWriter(str(out / "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (cam.width, cam.height))
    gif_frames = []
    footprints = []
    frame_id = 0
    n_rv = 0
    rv_log = []
    t_det = 0.0
    frames_meta = []

    def capture(pose, phase, do_detect=True):
        nonlocal frame_id, t_det
        frame_id += 1
        frame = render_frame(world, cam, pose, jitter, rng)
        dets = []
        if do_detect:
            t0 = time.time()
            dets = det(frame)
            t_det += time.time() - t0
            mapper.update(frame_id, phase, pose, frame, dets)
        if phase != "transit":
            footprints.append(cam.footprint_poly(pose))
        vis = hud.draw(frame, pose, dets, phase, flight.track, mapper.objects, labels, frame_id, n_rv)
        vw.write(vis)
        if phase != "transit" and frame_id % gif_every == 0:   # GIF는 탐지 프레임만 (이동 구간 제외)
            gif_frames.append(cv2.resize(vis, (480, 360), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        if frame_id % 25 == 1 or phase == "revisit":
            _imwrite(out / "frames" / f"f{frame_id:04d}_{phase}.jpg", vis, 80)
        frames_meta.append({"frame": frame_id, "phase": phase, "t": round(pose.t, 1), "x": round(pose.x, 2),
                            "y": round(pose.y, 2), "alt": round(pose.alt, 1), "yaw": round(pose.yaw, 1), "n_det": len(dets)})
        return frame, dets

    stop = False
    for k, wp in enumerate(wps):
        if stop:
            break
        for pose in flight.fly_to(wp, alt, "coverage"):
            capture(pose, "coverage")
            if frame_id >= max_frames:
                print(f"  ⚠️ 최대 프레임 {max_frames} 도달 — 중단")
                stop = True
                break
        # 줄 끝(k 홀수)마다 애매한 후보를 끼워넣어 저고도 재방문
        if revisit and k % 2 == 1 and not stop:
            cands = Planner.order_revisits(mapper.candidates(conf_lo, conf_hi), (flight.x, flight.y))
            for o in cands:
                o.revisited = True
                n_rv += 1
                for pose in flight.fly_to((o.x, o.y), alt, "transit"):
                    capture(pose, "transit", do_detect=False)
                for pose in flight.fly_to((o.x, o.y), alt_low, "transit"):   # 하강
                    capture(pose, "transit", do_detect=False)
                before = o.cov_max
                for pose in flight.hover(rv_frames, "revisit"):
                    capture(pose, "revisit")
                for pose in flight.fly_to((o.x, o.y), alt, "transit"):       # 상승
                    capture(pose, "transit", do_detect=False)
                st = o.status(conf_lo, conf_hi)
                rv_log.append({"oid": o.oid, "x": round(o.x, 2), "y": round(o.y, 2), "cov_max": round(before, 3),
                               "rv_max": round(o.rv_max, 3), "result": st})
                print(f"  재방문 #{o.oid} ({o.cls}) 커버리지 {before:.2f} → 저고도 {max(o.rv_max, 0):.2f} ⇒ {st}")
    vw.release()
    if gif_frames:
        import imageio
        step = max(1, math.ceil(len(gif_frames) / gif_max))
        imageio.mimsave(out / "flight.gif", gif_frames[::step], duration=0.3, loop=0)

    # ---- 평가·출력
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    cover = unary_union([Polygon(f) for f in footprints]).intersection(B)
    res, labs = evaluate(mapper.objects, labels_all, bbox, conf_lo, conf_hi, tol_m)
    cov_track = [t for t in flight.track if t[4] == "coverage"]
    dist_cov = float(sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(cov_track, cov_track[1:])))
    summary = {
        "name": name, "model": str(weights), "device": "cpu", "bbox_epsg5186": list(bbox),
        "area_m2": float(B.area), "covered_m2": float(cover.area), "coverage_frac": float(cover.area / B.area),
        "alt_m": alt, "alt_revisit_m": alt_low, "gsd_cm": round(cam.gsd(alt) * 100, 2),
        "gsd_revisit_cm": round(cam.gsd(alt_low) * 100, 2), "camera": {"w": cam.width, "h": cam.height, "hfov": cam.hfov_deg},
        "speed_mps": speed, "dt_s": dt, "overlap_side": overlap, "plan": plan_info, "wind": list(wind), "gust": gust,
        "jitter": jitter, "conf_lo": conf_lo, "conf_hi": conf_hi, "merge_m": merge_m, "match_tol_m": tol_m,
        "flight_time_s": round(flight.t, 1), "flight_dist_m": round(flight.dist, 1),
        "coverage_only_dist_m": round(dist_cov, 1), "revisit_extra_dist_m": round(flight.dist - dist_cov, 1),
        "frames": frame_id, "frames_detected": sum(1 for f in frames_meta if f["phase"] != "transit"),
        "detections_raw": len(mapper.rows), "objects_total": len(mapper.objects),
        "revisits": n_rv, "revisit_log": rv_log, "eval": res,
        "wall_time_s": round(time.time() - t_wall, 1), "detect_time_s": round(t_det, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "detections.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "phase", "t_s", "alt_m", "gsd_cm", "cls", "score", "x", "y", "w_m", "h_m", "object_id"])
        w.writerows(mapper.rows)
    from pyproj import Transformer
    inv = Transformer.from_crs(world.crs, 4326, always_xy=True)
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "status", "cls", "x", "y", "lat", "lon", "w_m", "h_m", "n_views", "cov_max", "rv_max", "revisited", "first_frame"])
        for o in mapper.objects:
            lon, lat = inv.transform(o.x, o.y)
            w.writerow([o.oid, o.status(conf_lo, conf_hi), o.cls, f"{o.x:.2f}", f"{o.y:.2f}", f"{lat:.7f}", f"{lon:.7f}",
                        f"{o.w_m:.2f}", f"{o.h_m:.2f}", o.n_views, f"{o.cov_max:.3f}", f"{o.rv_max:.3f}", o.revisited, o.first_frame])
    (out / "frames.json").write_text(json.dumps(frames_meta, ensure_ascii=False), encoding="utf-8")
    e0, e1 = res["coverage_only"], res["with_revisit"]
    title = (f"{name}: 고도 {alt} m · {frame_id}프레임 · 재현율 커버리지 {e0['recall']:.2f} → 재방문 {e1['recall']:.2f}"
             f" (라벨 {res['labels_in_area']})")
    write_map_png(out / "map.png", world, bbox, flight.track, mapper.objects, labs, conf_lo, conf_hi, title)
    note = (f"시뮬레이션 {name} · 모델 {weights.stem if weights.exists() else model} · 고도 {alt} m (재방문 {alt_low} m) · "
            f"비행 {flight.dist:.0f} m / {flight.t:.0f} s · 확정 {e1['confirmed']} (재현율 {e1['recall']:.2f}, 커버리지만 {e0['recall']:.2f})")
    write_map_html(out / "map.html", world, flight.track, mapper.objects, labs, conf_lo, conf_hi, note)
    print(f"비행 {flight.dist:.0f} m · {flight.t:.0f} s · 프레임 {frame_id} (탐지 {summary['frames_detected']}) · "
          f"커버 {summary['coverage_frac']*100:.0f}% · 원시 탐지 {len(mapper.rows)} · 물체 {len(mapper.objects)} · 재방문 {n_rv}")
    print(f"  커버리지만: 확정 {e0['confirmed']} · 후보 {e0['candidates']} · 재현율 {e0['recall']:.2f} · 정밀도 {e0['precision']:.2f}"
          f" (후보 포함 재현율 {e0['recall_incl_candidates']:.2f})")
    print(f"  재방문 포함: 확정 {e1['confirmed']} · 후보 {e1['candidates']} · 기각 {e1['rejected']} · 재현율 {e1['recall']:.2f}"
          f" · 정밀도 {e1['precision']:.2f} · 추가 비행 {summary['revisit_extra_dist_m']:.0f} m")
    print(f"  실행 {summary['wall_time_s']:.0f} s (탐지 {t_det:.0f} s) → {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.sim_ortho", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="runs/sim/<name>")
    ap.add_argument("--model", default="uavvaste", help="uavvaste | aihub | 가중치 경로")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="EPSG:5186 m (없으면 라벨 밀집 구간 자동)")
    ap.add_argument("--size", type=float, default=100.0, help="자동 영역 한 변 (m)")
    ap.add_argument("--alt", type=float, default=20.0, help="커버리지 고도 (m)")
    ap.add_argument("--alt_low", type=float, default=8.0, help="재방문 고도 (m)")
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=1.0, help="프레임 간격 (s)")
    ap.add_argument("--overlap", type=float, default=0.3, help="횡방향 겹침률")
    ap.add_argument("--conf_lo", type=float, default=0.25, help="후보 하한")
    ap.add_argument("--conf_hi", type=float, default=0.5, help="확정 하한")
    ap.add_argument("--merge_m", type=float, default=1.0, help="같은 물체로 합치는 반경 (m)")
    ap.add_argument("--no-revisit", action="store_true")
    ap.add_argument("--rv_frames", type=int, default=3, help="재방문 때 저고도 프레임 수")
    ap.add_argument("--wind", type=float, nargs=2, default=(0.0, 0.0), metavar=("VX", "VY"), help="정상풍 (m/s, 동·북)")
    ap.add_argument("--gust", type=float, default=0.0, help="돌풍 표준편차 (m/s)")
    ap.add_argument("--jitter", type=float, default=0.3, help="블러·노출 잡음 세기 0~1")
    ap.add_argument("--gps_noise", type=float, default=0.0, help="보고 위치 잡음 (m, 1σ)")
    ap.add_argument("--yaw_noise", type=float, default=1.0, help="기수 잡음 (도, 1σ)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--hfov", type=float, default=73.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max_frames", type=int, default=1500)
    ap.add_argument("--tol", type=float, default=1.0, help="라벨 적중 허용 거리 (m)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.name, a.model, a.bbox, a.size, a.alt, a.alt_low, a.speed, a.dt, a.overlap, a.conf_lo, a.conf_hi, a.merge_m,
        not a.no_revisit, a.rv_frames, tuple(a.wind), a.gust, a.jitter, a.gps_noise, a.yaw_noise,
        Camera(a.width, a.height, a.hfov), a.imgsz, a.max_frames, a.tol, a.seed)


if __name__ == "__main__":
    main()
