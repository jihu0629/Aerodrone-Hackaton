"""
물체별 부피 — SAM 2 마스크 + 여러 프레임 투표로 물체 점만 골라내기.

반경으로 물체를 자르면 그림자·바닥 잡음까지 들어가 부피가 크게 틀렸다 (작은 상자 5배).
여기서는:
  1) 물체 중심(지면 좌표)을 각 프레임에 다시 비춰, 그 위치를 점 프롬프트로 SAM 2가 물체 외곽선을 딴다
  2) 3D 점(조밀 복원 fused.ply, 없으면 희소 점)을 각 프레임에 비춰 '마스크 안에 들어온 비율'을 센다
  3) 여러 프레임에서 꾸준히 마스크 안에 들어온 점만 물체로 인정 (투표)
  4) 주변 바닥 높이 대비 높이지도 적분 = 부피.  +  가장 수직에 가까운 프레임의 마스크를 지면에 펴서 잰
     바닥 면적 × 윗면 높이(상자형 근사)도 같이 낸다

v2 (0010 경사뷰 교훈):
  - 바닥 = 주변 링 점으로 맞춘 지역 평면(RANSAC+최소제곱). 각 점 높이 = z − 평면(x,y). 골목 경사 대응
  - 물체 점 = 투표 통과 ∩ 평면 기준 높이 > h_min(3 cm). 경사뷰에서 상자 뒤 바닥이 마스크 안에 들어와도 걸러짐
  - 중심 재추정: 높이 10 cm 이상 점의 중앙 → SAM 프롬프트는 (평면 + h/2) 높이의 3D 점을 투영한 픽셀
  - 윗면 높이 = 높이 히스토그램의 상단 봉우리(기본) 또는 p90. 높이지도는 p98 클리핑 후 칸별 중앙값/최대 둘 다 기록
  - 뒤집힌 카메라(평면 위 0.3 m 미만, 기울기 85° 초과) 제외. 수직에 가까운(nadir ≥ 0.7) 프레임이 없으면 마스크면적은 null
  - --device cpu 로 GPU 없이 SAM 실행 가능

  python -m litter.objvol --orbit runs/orbit/0007 --srt DJI_0007.SRT --ply dense/fused.ply \
         --objects runs/map/0007_sfm/objects.csv --select "Plastic_Buoy:55,Styrofoam_Buoy:22" --out runs/orbit/0007/objvol
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .orbit3d import _center, _viewdir
from .seg import _imread, _imwrite
from .telemetry import read_srt
from .volume import fit_ground, to_metric, to_metric_auto, volume_heightmap, volume_hull


def read_ply(path):
    """COLMAP fused.ply (binary_little_endian) → xyz (N,3), rgb (N,3)."""
    with open(path, "rb") as f:
        header, props, n = [], [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}[t]))
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    rgb = np.stack([data["red"], data["green"], data["blue"]], 1) if "red" in data.dtype.names else None
    return xyz, rgb


class Scene:
    def __init__(self, orbit_dir, srt, min_track_m=10.0):
        import pycolmap

        self.dir = Path(orbit_dir)
        self.rec = pycolmap.Reconstruction(str(self.dir / "sparse" / "0"))
        self.imgs = sorted(self.rec.images.values(), key=lambda i: i.name)
        self.cam = self.rec.cameras[self.imgs[0].camera_id]
        fidx = json.loads((self.dir / "frames" / "frames.json").read_text(encoding="utf-8"))
        tel = {r["frame"]: r for r in read_srt(srt)}
        P = np.array([p.xyz for p in self.rec.points3D.values()])
        C = np.array([_center(i) for i in self.imgs])
        D = np.array([_viewdir(i) for i in self.imgs])
        alt = np.array([tel[fidx[i.name]]["alt"] for i in self.imgs])
        from .geometry import Geo
        geo = Geo(None, lon=tel[fidx[self.imgs[0].name]]["lon"])
        gps = np.array([geo.to_xy(tel[fidx[i.name]]["lat"], tel[fidx[i.name]]["lon"]) for i in self.imgs])
        self.T, self.R, self.s, self.info = to_metric_auto(P, C, D, alt, gps, min_track_m=min_track_m)
        self.p0 = self.info["p0"]
        self.sparse = P
        self.D = D
        self.Cm = self.T(C)                      # 카메라 중심 (지면 좌표, m)
        self.Dm = D @ self.R.T                   # 광축 방향 (지면 좌표)
        self.tilt_deg = np.degrees(np.arccos(np.clip(-self.Dm[:, 2], -1, 1)))  # 0 = 수직 아래

    def to_sfm(self, q):
        return (np.asarray(q, float) / self.s) @ self.R + self.p0

    def project(self, im, X):
        """SfM 좌표 (N,3) → 픽셀 (N,2), 카메라 뒤는 NaN."""
        Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Xc = X @ np.asarray(Tc.rotation.matrix()).T + np.asarray(Tc.translation)
        uv = np.full((len(X), 2), np.nan)
        ok = Xc[:, 2] > 0
        if ok.any():
            uv[ok] = np.asarray(self.cam.img_from_cam(Xc[ok], False))
        return uv

    def nadir_score(self, im):
        k = self.imgs.index(im)
        return float(-self.Dm[k, 2])  # 1 = 수직 아래


def fit_plane(Q, thr=0.02, iters=300, max_n=20000, rng=None):
    """지역 바닥 평면 z = a·x + b·y + c (RANSAC → 인라이어 최소제곱 2회). 반환 (coef, 인라이어 비율).
    링 안에 이웃 물체가 섞여 있어도(0010처럼 상자 옆에 상자) 가장 많은 점을 지나는 평면이 바닥."""
    rng = rng or np.random.default_rng(0)
    if len(Q) > max_n:
        Q = Q[rng.choice(len(Q), max_n, replace=False)]
    A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
    best, best_k = np.array([0.0, 0.0, float(np.median(Q[:, 2]))]), int((np.abs(Q[:, 2] - np.median(Q[:, 2])) < thr).sum())
    for _ in range(iters):
        i = rng.choice(len(Q), 3, replace=False)
        try:
            c = np.linalg.solve(A[i], Q[i, 2])
        except np.linalg.LinAlgError:
            continue
        if np.hypot(c[0], c[1]) > 0.6:  # 기울기 30° 넘는 평면은 바닥이 아님
            continue
        k = int((np.abs(A @ c - Q[:, 2]) < thr).sum())
        if k > best_k:
            best, best_k = c, k
    inl = np.abs(A @ best - Q[:, 2]) < thr
    for _ in range(2):
        if inl.sum() < 3:
            break
        best, *_ = np.linalg.lstsq(A[inl], Q[inl, 2], rcond=None)
        inl = np.abs(A @ best - Q[:, 2]) < thr
    return best, float(inl.mean())


def plane_z(coef, xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return xy[:, 0] * coef[0] + xy[:, 1] * coef[1] + coef[2]


def height_mode(h, bin_m=0.01, min_frac=0.2):
    """높이 히스토그램에서 '가장 높은 봉우리'(최대 봉우리의 min_frac 이상인 국소 최대 중 제일 높은 것)."""
    if len(h) < 10:
        return float("nan")
    cnt, edges = np.histogram(h, bins=np.arange(0, h.max() + 2 * bin_m, bin_m))
    if len(cnt) < 3:
        return float(np.median(h))
    sm = np.convolve(cnt, [1, 2, 1], mode="same") / 4.0
    peaks = [i for i in range(len(sm)) if sm[i] >= sm[max(i - 1, 0)] and sm[i] >= sm[min(i + 1, len(sm) - 1)]
             and sm[i] >= min_frac * sm.max() and cnt[i] > 0]
    i = max(peaks) if peaks else int(np.argmax(sm))
    return float(edges[i] + bin_m / 2)


def sam_mask(sam, img, uv, crop=260, max_frac=0.6, device=""):
    """uv(물체 중심 픽셀)을 점 프롬프트로 물체 마스크. 주변을 잘라서(crop) 빠르게.
    마스크가 잘린 영역의 max_frac 이상이면 바닥을 통째로 딴 것 → 실패로 본다."""
    H, W = img.shape[:2]
    x, y = int(uv[0]), int(uv[1])
    x0, y0 = max(x - crop, 0), max(y - crop, 0)
    x1, y1 = min(x + crop, W), min(y + crop, H)
    sub = img[y0:y1, x0:x1]
    kw = {"device": device} if device else {}
    r = sam(sub, points=[[x - x0, y - y0]], labels=[1], verbose=False, **kw)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != sub.shape[:2]:
        m = cv2.resize(m, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab = cv2.connectedComponents(m)
    if n > 1 and lab[y - y0, x - x0] > 0:  # 프롬프트 점이 속한 조각만
        m = (lab == lab[y - y0, x - x0]).astype(np.uint8)
    if m.mean() > max_frac or m.sum() < 30:
        return None
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    return full


def measure_object(sc, sam, center_xy, Pm, n_views=24, search_r=0.8, vote=0.7, h_min=0.03, height="mode", hm_agg="median",
                   min_cam_h=0.3, max_tilt=85.0, min_nadir=0.7, device="", out_dir=None, tag="obj"):
    """Pm: 지면 좌표(m)의 3D 점 (N,3). 반환 (결과 dict, 물체 점 (x, y, 평면 기준 높이))."""
    center0 = np.asarray(center_xy, float)

    def local(cxy):
        """중심 cxy 기준: 후보 점 index, 지역 바닥 평면 계수, 인라이어 비율, 후보 점 높이."""
        d = np.hypot(Pm[:, 0] - cxy[0], Pm[:, 1] - cxy[1])
        cand = np.nonzero(d < search_r)[0]
        ring = Pm[(d > search_r) & (d < search_r + 0.6)]
        if len(ring) > 50:
            coef, inl = fit_plane(ring)
        else:  # 링이 비면 후보 점 아래쪽 10%로 대충
            coef, inl = np.array([0.0, 0.0, float(np.percentile(Pm[cand, 2], 10)) if len(cand) else 0.0]), 0.0
        h = Pm[cand, 2] - plane_z(coef, Pm[cand, :2])
        return cand, coef, inl, h

    # 1) 지역 평면 + 2) 중심 재추정 (높이 0.1 m 이상 점, 없으면 h_min 이상) → 평면·후보 다시
    cand, coef, inl, h = local(center0)
    center = center0.copy()
    for _ in range(2):
        hi = h > 0.10
        if hi.sum() < 30:
            hi = h > h_min
        if hi.sum() < 30:
            break
        new = np.median(Pm[cand[hi], :2], axis=0)
        if np.linalg.norm(new - center) < 0.02:
            break
        center = new
        cand, coef, inl, h = local(center)
    hi = h > 0.10
    if hi.sum() < 30:
        hi = h > h_min
    h_pre = float(np.percentile(h[hi], 90)) if hi.sum() >= 30 else 0.0
    gz = float(plane_z(coef, center)[0])
    tilt_plane = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))

    # SAM 프롬프트: 물체 중심을 평면 위 h/2 높이에 둔 3D 점 (경사뷰에서 상자 뒤 바닥을 찍지 않도록)
    X = sc.to_sfm(Pm[cand])
    c_sfm = sc.to_sfm([center[0], center[1], gz + h_pre / 2])[None]

    # 3) 뒤집힌/비정상 카메라 제외: 지역 평면 기준 카메라 높이 < min_cam_h 또는 기울기 > max_tilt
    cam_h = sc.Cm[:, 2] - plane_z(coef, sc.Cm[:, :2])
    bad = (cam_h < min_cam_h) | (sc.tilt_deg > max_tilt)
    views = []
    for k, im in enumerate(sc.imgs):
        if bad[k]:
            continue
        uv = sc.project(im, c_sfm)[0]
        if np.isfinite(uv).all() and 150 < uv[0] < sc.cam.width - 150 and 150 < uv[1] < sc.cam.height - 150:
            views.append((np.hypot(uv[0] - sc.cam.width / 2, uv[1] - sc.cam.height / 2), im, uv, k))
    views.sort(key=lambda t: t[0])
    views = views[::max(1, len(views) // n_views)][:n_views]
    inside = np.zeros(len(X))
    seen = np.zeros(len(X))
    best_area, used, gallery, used_names, used_tilts = None, 0, [], [], []
    for _, im, uv, k in views:
        img = _imread(sc.dir / "frames" / im.name)
        m = sam_mask(sam, img, uv, device=device)
        if m is None:
            continue
        used += 1
        used_names.append(im.name)
        used_tilts.append(float(sc.tilt_deg[k]))
        p = sc.project(im, X)
        ok = np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < sc.cam.width) & (p[:, 1] >= 0) & (p[:, 1] < sc.cam.height)
        pi = p[ok].astype(int)
        seen[ok] += 1
        inside[np.nonzero(ok)[0][m[pi[:, 1], pi[:, 0]] > 0]] += 1
        # 바닥 면적: 마스크 경계를 지역 평면으로 펴서 — 수직에 가까운 프레임(nadir ≥ min_nadir)만
        ns = sc.nadir_score(im)
        if ns >= min_nadir:
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                cnt = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
                Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
                Rw = np.asarray(Tc.rotation.matrix()).T
                c0 = sc.Cm[k]
                xy = np.asarray(sc.cam.cam_from_img(cnt))
                dirs = (sc.R @ (Rw @ np.c_[xy, np.ones(len(xy))].T)).T
                # 광선 c0 + t·dir 이 평면 z = a x + b y + c 와 만나는 t
                den = dirs[:, 2] - coef[0] * dirs[:, 0] - coef[1] * dirs[:, 1]
                t = (coef[0] * c0[0] + coef[1] * c0[1] + coef[2] - c0[2]) / den
                G = c0[None, :2] + t[:, None] * dirs[:, :2]
                area = float(cv2.contourArea(G.astype(np.float32)))
                if best_area is None or ns > best_area[1]:
                    best_area = (area, ns, G)
        if len(gallery) < 4:
            v = img.copy()
            v[m > 0] = (0.5 * v[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            x, y = int(uv[0]), int(uv[1])
            cv2.circle(v, (x, y), 5, (0, 255, 0), -1)
            gallery.append(cv2.resize(v[max(y - 160, 0):y + 160, max(x - 160, 0):x + 160], (240, 240)))
    # 4) 물체 점 = 투표 통과 ∩ 평면 기준 높이 > h_min
    keep = (seen >= 3) & (inside / np.maximum(seen, 1) >= vote) & (h > h_min)
    O = np.c_[Pm[cand[keep], :2], h[keep]]
    res = {"tag": tag, "views_used": used, "views_candidates": int(len(views)), "n_views_excluded": int(bad.sum()),
           "views_tilt_deg": [round(t, 1) for t in used_tilts], "views_names": used_names,
           "points_candidate": int(len(cand)), "points_voted": int(((seen >= 3) & (inside / np.maximum(seen, 1) >= vote)).sum()),
           "points_object": int(len(O)),
           "center_xy": [float(center[0]), float(center[1])], "center_shift_m": float(np.linalg.norm(center - center0)),
           "ground_z": gz, "ground_plane_tilt_deg": tilt_plane, "ground_plane_inlier": inl,
           "height_method": height, "h_min": h_min,
           "mask_area_m2": None, "volume_mask_box_L": None, "mask_nadir_score": None}
    if len(O) >= 10:
        h_p90 = float(np.percentile(O[:, 2], 90))
        h_mode = height_mode(O[:, 2])
        h_top = h_mode if (height == "mode" and np.isfinite(h_mode)) else h_p90
        O = O[O[:, 2] <= h_top * 1.3 + 0.03]            # 윗면보다 크게 튄 점(잡음) 제거
        O[:, 2] = np.minimum(O[:, 2], np.percentile(O[:, 2], 98))  # p98 클리핑
        rect = cv2.minAreaRect(O[:, :2].astype(np.float32))
        L, W = sorted(rect[1], reverse=True)
        # 높이지도: 칸별 중앙값(잡음에 강함, 기본) / 최대(성긴 점·경사뷰용) — 둘 다 기록
        v_med, foot, _ = volume_heightmap(O, agg="median")
        v_max, _, _ = volume_heightmap(O, agg="max")
        # 윗면 점(높이 ≥ h_top/2)만의 발자국 — 상자 옆 그림자·후광(3~8 cm) 점을 뺀 크기
        top = O[O[:, 2] >= 0.5 * h_top]
        if len(top) >= 5:
            rt = cv2.minAreaRect(top[:, :2].astype(np.float32))
            Lt, Wt = sorted(rt[1], reverse=True)
            res.update(top_length_m=float(Lt), top_width_m=float(Wt))
        res.update(length_m=float(L), width_m=float(W), height_m=h_top, height_p90_m=h_p90, height_mode_m=h_mode,
                   footprint_m2=foot, hm_agg=hm_agg, volume_heightmap_L=(v_med if hm_agg == "median" else v_max) * 1000,
                   volume_heightmap_median_L=v_med * 1000, volume_heightmap_max_L=v_max * 1000,
                   volume_hull_L=volume_hull(O) * 1000)
        if best_area:
            res.update(mask_area_m2=best_area[0], volume_mask_box_L=best_area[0] * h_top * 1000,
                       mask_nadir_score=best_area[1])
    if out_dir and gallery:
        _imwrite(Path(out_dir) / f"{tag}_masks.jpg", np.hstack(gallery), 88)
    return res, O


def main(argv=None):
    from ultralytics import SAM

    ap = argparse.ArgumentParser(prog="litter.objvol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--ply", help="조밀 복원 fused.ply (없으면 희소 점)")
    ap.add_argument("--objects", required=True, help="orbit_map objects.csv")
    ap.add_argument("--select", help="'클래스:프레임수' 목록 (예: Plastic_Buoy:55,Styrofoam_Buoy:22). 없으면 쓰레기 전부")
    ap.add_argument("--truth", help="'태그:가로x세로x높이cm' 목록 (비교용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="", help="SAM 2 장치: cpu / 0 (GPU). 비우면 자동")
    ap.add_argument("--height", default="mode", choices=["mode", "p90"], help="윗면 높이: 히스토그램 상단 봉우리(기본) / 상위 10%% 분위")
    ap.add_argument("--hm-agg", default="median", choices=["median", "max"], help="높이지도 칸 집계 (기본 중앙값)")
    ap.add_argument("--n-views", type=int, default=24)
    ap.add_argument("--h-min", type=float, default=0.03, help="바닥 평면 기준 이 높이(m) 아래 점은 바닥으로 본다")
    ap.add_argument("--search-r", type=float, default=0.8, help="물체 점 탐색 반경(m). 옆에 연석·벽이 있으면 줄일 것")
    ap.add_argument("--min-track", type=float, default=10.0,
                    help="GPS 수평 이동이 이 거리(m) 이상이면 SRT 고도 대신 GPS 경로로 축척 (SRT 고도가 틀릴 때 낮춰서 강제)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scene(a.orbit, a.srt, min_track_m=a.min_track)
    pts = read_ply(a.ply)[0] if a.ply else sc.sparse
    Pm = sc.T(pts).astype(np.float32)   # 지면 좌표(m)로 한 번만 변환
    del pts
    print(f"3D 점 {len(Pm):,}개 ({'조밀' if a.ply else '희소'}) · 축척 편차 {sc.info['scale_spread_pct']:.1f}%"
          f" · 카메라 기울기 중앙값 {np.median(sc.tilt_deg):.0f}° · SAM 장치 {a.device or '자동'}")
    sam = SAM("sam2.1_b.pt")
    objs = list(csv.DictReader(open(a.objects, encoding="utf-8-sig")))
    sel = [tuple(s.split(":")) for s in a.select.split(",")] if a.select else None
    truth = dict(t.split(":") for t in a.truth.split(",")) if a.truth else {}
    results, seen_tags = [], {}
    for o in objs:
        key = (o["cls"], o["seen_frames"])
        if sel and key not in sel:
            continue
        if not sel and o["kind"] != "litter":
            continue
        base = f"{o['cls']}_{o['seen_frames']}"
        seen_tags[base] = seen_tags.get(base, 0) + 1
        tag = base if seen_tags[base] == 1 else f"{base}_{seen_tags[base]}"  # 같은 클래스·프레임수가 여럿이면 _2, _3
        r, _ = measure_object(sc, sam, (float(o["local_x_m"]), float(o["local_y_m"])), Pm, n_views=a.n_views, search_r=a.search_r,
                              h_min=a.h_min, height=a.height, hm_agg=a.hm_agg, device=a.device, out_dir=out, tag=tag)
        r["lat"], r["lon"] = o["lat"], o["lon"]
        r["csv_local_xy"] = [float(o["local_x_m"]), float(o["local_y_m"])]
        if base in truth:
            l, w, h = (float(v) / 100 for v in truth[base].split("x"))
            r["truth_L"] = l * w * h * 1000
            r["truth_cm"] = truth[base]
        results.append(r)
        msg = (f"{tag} ({o['local_x_m']},{o['local_y_m']}): 프레임 {r['views_used']}/{r['views_candidates']}장"
               f"(제외 {r['n_views_excluded']}) · 평면 기울기 {r['ground_plane_tilt_deg']:.1f}° · 중심 이동 {r['center_shift_m']:.2f} m"
               f" · 물체 점 {r['points_object']}개")
        if "volume_heightmap_L" in r:
            msg += (f" · {r['length_m']*100:.0f}×{r['width_m']*100:.0f}×{r['height_m']*100:.0f} cm"
                    f" (p90 {r['height_p90_m']*100:.1f} / 봉우리 {r['height_mode_m']*100:.1f})"
                    f" · 윗면 {r.get('top_length_m',0)*100:.0f}×{r.get('top_width_m',0)*100:.0f}"
                    f" · 높이지도 {r['volume_heightmap_median_L']:.1f} L(중앙값)/{r['volume_heightmap_max_L']:.1f} L(최대)"
                    f" · 볼록껍질 {r['volume_hull_L']:.1f} L")
            if r["volume_mask_box_L"] is not None:
                msg += f" · 마스크면적 {r['mask_area_m2']:.3f} m² × 높이 = {r['volume_mask_box_L']:.1f} L"
            else:
                msg += " · 마스크면적 없음(수직 프레임 없음)"
        if "truth_L" in r:
            msg += f"  (정답 {r['truth_cm']} = {r['truth_L']:.1f} L)"
        print(msg)
    (out / "objvol.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
