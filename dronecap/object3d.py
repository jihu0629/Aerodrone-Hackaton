"""물체 주위를 돌며 찍은 영상(orbit) → 3D 복원 → 부피 계산용 길이·넓이·높이 추정.

흐름
  1. 영상에서 프레임 추출 (일정 간격, 긴 변 max_size 로 축소)
  2. pycolmap(COLMAP 의 파이썬 패키지) 으로 SIFT → 순차 매칭 → 증분 SfM. CPU 로 동작한다 (GPU 불필요).
     결과: 카메라 자세 + 희소 점군. **단위 없음**(상대 좌표).
  3. 바닥 평면: 희소 점군에 RANSAC 평면 맞춤 (바닥이 가장 큰 평면이라는 가정).
  4. 축척: 사용자가 주는 기준 길이(두 이미지에서 같은 두 점을 찍고 실제 길이 입력) 또는 카메라 높이(대략).
  5. 물체 위치: 각 프레임의 물체 마스크 무게중심 → 광선과 바닥 평면 교점 → 중앙값.
  6. 부피: **visual hull(실루엣 교차, 복셀 카빙)**. 흰 상자처럼 특징점이 거의 없는 물체는 희소 점군만으로는
     점이 몇 개 안 잡히므로(실험: 상자 위 18점), 실루엣이 더 믿을 만하다. 여러 프레임 중 일부 마스크가 틀려도
     견디도록 '투표 비율' 로 깎는다.
     한계: 오목한 부분은 못 깎는다(상한값). 바닥에 닿은 면은 평면이라고 가정한다.
  7. 산출: 바닥 투영 면적(footprint), 방향 정렬 사각형의 길이·너비, 높이, 부피(복셀 합·높이격자·외접 상자),
     희소 점군 기반 교차 확인, PLY 점군, 투영 확인 이미지, metrics.json.

주의
  * 축척 오차는 부피에 세제곱으로 들어간다 (길이 3 % 오차 → 부피 약 9 %).
  * 카메라가 물체 **옆면까지** 보게 돌아야 높이가 잡힌다. 바로 위에서만 찍으면 높이가 과소/불안정.
  * 흰 상자용 기본 마스크(밝고 매끈한 영역)는 임시 방편이다. 물체 색이 다르면 --mask-dir 로 외부 마스크를 준다.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


# ----------------------------------------------------------------------------- 1. 프레임
def extract_orbit_frames(video: str | Path, out_dir: str | Path, target_count: int = 40, max_size: int = 1080,
                         blur_min: float = 0.0) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, n // target_count) if n else 1
    files: list[Path] = []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            if blur_min > 0:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(g, cv2.CV_64F).var() < blur_min:
                    i += 1
                    continue
            h, w = f.shape[:2]
            s = max_size / max(w, h)
            if s < 1:
                f = cv2.resize(f, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            p = out_dir / f"img{len(files):03d}.jpg"
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append(p)
        i += 1
    cap.release()
    return files


# ----------------------------------------------------------------------------- 2. SfM
def run_sfm(images_dir: str | Path, work_dir: str | Path, max_image_size: int = 1080, num_threads: int = 4,
            overlap: int = 12, exhaustive: bool = False, force: bool = False, log: Callable[[str], None] = print):
    import pycolmap
    images_dir, work_dir = Path(images_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sparse = work_dir / "sparse"
    if (sparse / "0" / "cameras.bin").exists() and not force:
        log(f"기존 SfM 결과 사용: {sparse / '0'} (--force 로 다시 계산)")
        return pycolmap.Reconstruction(sparse / "0")
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    t = time.time()
    eo = pycolmap.FeatureExtractionOptions(); eo.use_gpu = False; eo.max_image_size = max_image_size; eo.num_threads = num_threads
    pycolmap.extract_features(db, images_dir, camera_mode=pycolmap.CameraMode.SINGLE,   # 같은 카메라 → 내부 파라미터 공유
                              extraction_options=eo, device=pycolmap.Device.cpu)
    log(f"SIFT 추출 {time.time() - t:.0f}s"); t = time.time()
    mo = pycolmap.FeatureMatchingOptions(); mo.use_gpu = False; mo.num_threads = num_threads
    if exhaustive:
        pycolmap.match_exhaustive(db, matching_options=mo, device=pycolmap.Device.cpu)
    else:
        po = pycolmap.SequentialPairingOptions(); po.overlap = overlap; po.loop_detection = False
        pycolmap.match_sequential(db, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
    log(f"매칭 {time.time() - t:.0f}s"); t = time.time()
    sparse.mkdir(exist_ok=True)
    opt = pycolmap.IncrementalPipelineOptions(); opt.num_threads = num_threads
    recs = pycolmap.incremental_mapping(db, images_dir, sparse, options=opt)
    log(f"SfM {time.time() - t:.0f}s, 모델 {len(recs)} 개")
    if not recs:
        raise RuntimeError("SfM 실패: 등록된 이미지가 없습니다. 프레임 수를 늘리거나(--frames), 흐린 프레임을 빼거나, 겹침이 더 많게 천천히 돌며 찍으세요.")
    best_k = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_k]
    log(f"등록 {rec.num_reg_images()}/{len(list(images_dir.glob('*.jpg')))} 장, 3D 점 {rec.num_points3D()}")
    return rec


@dataclass
class View:
    name: str
    image_path: Path
    cam: "object"          # pycolmap.Camera
    M: np.ndarray          # 3x4 world→camera
    center: np.ndarray
    mask: Optional[np.ndarray] = None

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """세계 좌표 (N,3) → 픽셀 (N,2), 카메라 앞에 있고 화면 안에 들어오는지 (N,)"""
        Xc = X @ self.M[:, :3].T + self.M[:, 3]
        front = Xc[:, 2] > 1e-6
        uv = np.full((len(X), 2), -1.0)
        if front.any():
            uv[front] = np.asarray(self.cam.img_from_cam(Xc[front], check_cheirality=False))
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < self.cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.cam.height)
        return uv, inside

    def ray(self, uv: np.ndarray) -> np.ndarray:
        """픽셀 → 세계 좌표계 방향 벡터(단위)"""
        n = np.asarray(self.cam.cam_from_img(np.asarray(uv, dtype=np.float64).reshape(-1, 2)))
        d_cam = np.c_[n, np.ones(len(n))]
        R = self.M[:, :3]
        d = d_cam @ R          # R^T d_cam
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def views_from_reconstruction(rec, images_dir: str | Path) -> list[View]:
    views = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        views.append(View(im.name, Path(images_dir) / im.name, rec.cameras[im.camera_id], np.asarray(cfw.matrix()),
                          np.asarray(im.projection_center())))
    views.sort(key=lambda v: v.name)
    return views


def sparse_points(rec) -> tuple[np.ndarray, np.ndarray]:
    pts = list(rec.points3D.values())
    return np.array([p.xyz for p in pts]), np.array([p.color for p in pts], dtype=np.uint8)


# ----------------------------------------------------------------------------- 3. 바닥 평면
@dataclass
class Plane:
    n: np.ndarray   # 단위 법선, 카메라 쪽(위)을 향함
    d: float        # n·x + d = 0

    def height(self, X: np.ndarray) -> np.ndarray:
        return X @ self.n + self.d

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([1.0, 0, 0]) if abs(self.n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(self.n, a); u /= np.linalg.norm(u)
        v = np.cross(self.n, u)
        return u, v

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        denom = direction @ self.n
        if abs(denom) < 1e-9:
            return None
        t = -(origin @ self.n + self.d) / denom
        return origin + t * direction if t > 0 else None


def fit_plane_ransac(P: np.ndarray, cameras: np.ndarray, iters: int = 1000, thr_rel: float = 0.01,
                     seed: int = 0) -> tuple[Plane, np.ndarray]:
    """가장 많은 점을 품는 평면. thr_rel: 점군 퍼짐(표준편차 노름) 대비 허용 거리."""
    rng = np.random.default_rng(seed)
    thr = thr_rel * np.linalg.norm(P.std(0))
    best = (-1, None, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn; d = -n @ a
        inl = np.abs(P @ n + d) < thr
        k = int(inl.sum())
        if k > best[0]:
            best = (k, n, d)
    _, n, d = best
    inl = np.abs(P @ n + d) < thr
    # 내점으로 최소제곱 재추정
    Q = P[inl]; c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[-1]; n /= np.linalg.norm(n); d = -n @ c
    if np.mean(cameras @ n + d) < 0:        # 카메라가 위에 있도록
        n, d = -n, -d
    inl = np.abs(P @ n + d) < thr
    return Plane(n, float(d)), inl


# ----------------------------------------------------------------------------- 4. 축척
def triangulate_two_rays(c1, d1, c2, d2) -> np.ndarray:
    """두 광선의 최근접점 중점."""
    w = c1 - c2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        return c1
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((c1 + s * d1) + (c2 + t * d2))


def scale_from_reference(views: list[View], ref: list[tuple[str, tuple[float, float], tuple[float, float]]],
                         length_m: float) -> tuple[float, dict]:
    """ref: [(이미지이름, (u1,v1), (u2,v2)), (다른 이미지, ...)] 같은 두 점을 2장 이상에서 찍은 것.
    각 끝점을 삼각측량해 두 점 거리(SfM 단위) 를 재고 scale = length_m / 거리."""
    by = {v.name: v for v in views}
    vs = [by[name] for name, _, _ in ref if name in by]
    if len(vs) < 2:
        raise ValueError(f"기준점 이미지가 SfM 에 등록돼 있지 않습니다: {[r[0] for r in ref]}")
    ends = []
    for k in (1, 2):
        rays = [(by[name].center, by[name].ray(np.array(pt))[0]) for name, *pts in ref if name in by for pt in [pts[k - 1]]]
        X = triangulate_two_rays(*rays[0], *rays[1])
        ends.append(X)
    dist = float(np.linalg.norm(ends[0] - ends[1]))
    if dist <= 0:
        raise ValueError("기준 두 점이 같은 위치로 삼각측량됐습니다")
    # 세 번째 이상 뷰가 있으면 재투영 오차로 품질 확인
    reproj = []
    for name, p1, p2 in ref:
        if name not in by:
            continue
        uv, _ = by[name].project(np.array(ends))
        reproj.append(float(np.linalg.norm(uv[0] - p1)) + float(np.linalg.norm(uv[1] - p2)))
    return length_m / dist, {"ref_distance_sfm": dist, "reproj_err_px_sum": reproj, "endpoints_sfm": [e.tolist() for e in ends]}


# ----------------------------------------------------------------------------- 5. 마스크 · 물체 위치
def mask_bright_smooth(img: np.ndarray, bright: int = 170, std_max: float = 18.0, win: int = 15) -> np.ndarray:
    """밝고(흰 상자) 지역 표준편차가 낮은(무늬 없는) 영역. 얼룩무늬 바닥 위 흰 상자용 임시 방편."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    mean = cv2.blur(g, (win, win)); sq = cv2.blur(g * g, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    m = ((blur > bright) & (std < std_max)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m


def mask_dark_smooth(img: np.ndarray, dark: int = 80, std_max: float = 18.0, win: int = 15) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    mean = cv2.blur(g, (win, win)); sq = cv2.blur(g * g, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    m = ((blur < dark) & (std < std_max)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m


_REMBG_SESSION = None


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def mask_rembg(img: np.ndarray) -> np.ndarray:
    """rembg(U2Net, CPU, 첫 실행 때 모델 약 170 MB 다운로드): 두드러진 전경 물체 분리.
    실제 영상 시험에서 밝기 휴리스틱이 놓치던 그늘진 옆면까지 잡았다 (0.2~0.6 s/장)."""
    global _REMBG_SESSION
    from rembg import new_session, remove
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2net")
    a = np.asarray(remove(img[:, :, ::-1], session=_REMBG_SESSION, only_mask=True))   # RGB 입력
    return (a > 127).astype(np.uint8) * 255


def pick_component(mask: np.ndarray, target_uv: Optional[np.ndarray], min_area: int = 200) -> np.ndarray:
    """연결 성분 중 하나만 남긴다: target_uv 가 있으면 그 점을 포함하거나 가장 가까운 성분, 없으면 가장 큰 성분."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if target_uv is None:
        k = 1 + int(np.argmax(areas))
    else:
        u, v = int(round(target_uv[0])), int(round(target_uv[1]))
        if 0 <= v < lab.shape[0] and 0 <= u < lab.shape[1] and lab[v, u] > 0:
            k = int(lab[v, u])
        else:
            ok = np.where(areas >= min_area)[0]
            if len(ok) == 0:
                return np.zeros_like(mask)
            dist = np.linalg.norm(cent[1 + ok] - np.array(target_uv), axis=1)
            k = 1 + int(ok[np.argmin(dist)])
    return (lab == k).astype(np.uint8) * 255


def build_masks(views: list[View], plane: Plane, method: str = "auto", mask_dir: Optional[Path] = None,
                save_dir: Optional[Path] = None, log: Callable[[str], None] = print, **kw) -> np.ndarray:
    """1차: 가장 큰 성분으로 물체 위치 추정 → 2차: 그 위치를 포함하는 성분으로 확정. 반환: 바닥 위 물체 중심(세계 좌표).
    method: auto(rembg 있으면 rembg, 없으면 bright) | rembg | bright | dark | dir"""
    if method == "auto":
        method = "rembg" if rembg_available() else "bright"
        if method == "bright":
            log("[경고] rembg 가 없어 밝기 휴리스틱 마스크를 씁니다 (흰 물체 전용, 그늘진 면을 놓침). pip install rembg[cpu] 권장")
    log(f"마스크 방법: {method}")
    raw: dict[str, np.ndarray] = {}
    for v in views:
        if method == "dir":
            p = next((Path(mask_dir) / (v.image_path.stem + ext) for ext in (".png", ".jpg") if (Path(mask_dir) / (v.image_path.stem + ext)).exists()), None)
            if p is None:
                raise FileNotFoundError(f"마스크 없음: {mask_dir}/{v.image_path.stem}.png")
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            m = (m > 127).astype(np.uint8) * 255
        else:
            img = cv2.imread(str(v.image_path))
            m = {"bright": mask_bright_smooth, "dark": mask_dark_smooth, "rembg": mask_rembg}[method](img, **kw) if kw and method != "rembg" else \
                {"bright": mask_bright_smooth, "dark": mask_dark_smooth, "rembg": mask_rembg}[method](img)
        raw[v.name] = m
    # 1차 위치
    center = _object_center(views, {k: pick_component(m, None) for k, m in raw.items()}, plane)
    # 2차: 그 위치가 투영되는 성분
    for v in views:
        uv, inside = v.project(center[None])
        v.mask = pick_component(raw[v.name], uv[0] if inside[0] else None)
    center2 = _object_center(views, {v.name: v.mask for v in views}, plane)
    filter_bad_masks(views, plane, center2, log=log)
    center2 = _object_center(views, {v.name: v.mask for v in views}, plane)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for v in views:
            cv2.imwrite(str(save_dir / f"{v.image_path.stem}.png"), v.mask if v.mask is not None else np.zeros((v.cam.height, v.cam.width), np.uint8))
    used = sum(1 for v in views if v.mask is not None and v.mask.max() > 0)
    log(f"마스크 사용 {used}/{len(views)} 장, 물체 바닥 중심(SfM 좌표) {np.round(center2, 3)}")
    return center2


def filter_bad_masks(views: list[View], plane: Plane, center: np.ndarray, log: Callable[[str], None] = print,
                     area_lo: float = 0.3, area_hi: float = 3.0, dist_factor: float = 3.0) -> int:
    """엉뚱한 마스크(빈 마스크, 번쩍임 조각, 바닥 전체)를 None 으로 돌려 카빙에서 뺀다.
    실루엣 교차는 '하나라도 틀리면 깎여 나가는' 연산이라, 투표 비율로 눅이는 것보다 틀린 마스크를 빼는 쪽이 정확하다
    (투표 비율을 0.85 로 낮추면 상자 옆면이 수 cm 부풀었다 — 합성 실험)."""
    areas, dists = {}, {}
    for v in views:
        if v.mask is None or v.mask.max() == 0:
            continue
        ys, xs = np.nonzero(v.mask)
        areas[v.name] = len(xs)
        X = plane.intersect_ray(v.center, v.ray(np.array([xs.mean(), ys.mean()]))[0])
        dists[v.name] = np.linalg.norm(X - center) if X is not None else np.inf
    if not areas:
        return 0
    med_a = float(np.median(list(areas.values()))); med_d = float(np.median(list(dists.values()))) + 1e-9
    bad = 0
    for v in views:
        if v.name not in areas:
            v.mask = None; bad += 1; continue
        a, d = areas[v.name], dists[v.name]
        if a < area_lo * med_a or a > area_hi * med_a or d > dist_factor * med_d:
            v.mask = None; bad += 1
    log(f"마스크 품질 필터: {bad}/{len(views)} 장 제외 (면적 중앙값 {med_a:.0f}px, 중심 편차 중앙값 {med_d:.3g})")
    return bad


def _object_center(views: list[View], masks: dict[str, np.ndarray], plane: Plane) -> np.ndarray:
    pts = []
    for v in views:
        m = masks.get(v.name)
        if m is None or m.max() == 0:
            continue
        ys, xs = np.nonzero(m)
        uv = np.array([xs.mean(), ys.mean()])
        d = v.ray(uv)[0]
        X = plane.intersect_ray(v.center, d)
        if X is not None:
            pts.append(X)
    if not pts:
        raise RuntimeError("어느 프레임에서도 물체 마스크를 찾지 못했습니다 (--mask-method / --mask-dir 확인)")
    return np.median(np.array(pts), axis=0)


# ----------------------------------------------------------------------------- 6. visual hull
@dataclass
class HullGrid:
    origin: np.ndarray      # 격자 (0,0,0) 의 세계 좌표
    u: np.ndarray; v: np.ndarray; n: np.ndarray   # 격자 축 (바닥 평면 기준)
    voxel: float            # SfM 단위 복셀 한 변
    occ: np.ndarray         # (nx, ny, nz) bool
    votes: np.ndarray       # 투표 비율


def visual_hull(views: list[View], plane: Plane, center: np.ndarray, radius: float, zmax: float, res: int = 96,
                vote_ratio: float = 1.0, min_views: int = 5, log: Callable[[str], None] = print) -> HullGrid:
    """바닥 위 [−radius, radius]² × [0, zmax] 영역을 복셀로 나눠, 각 복셀 중심이 '화면 안에 들어온 뷰' 중
    마스크 안에 든 비율이 vote_ratio 이상이면 물체로 남긴다. 기본 1.0 = 엄밀한 실루엣 교차.
    (틀린 마스크는 filter_bad_masks 로 미리 빼는 것이 비율을 낮추는 것보다 정확하다.)"""
    u, v = plane.basis(); n = plane.n
    voxel = 2 * radius / res
    nz = max(2, int(np.ceil(zmax / voxel)))
    gx = (np.arange(res) + 0.5) * voxel - radius
    gz = (np.arange(nz) + 0.5) * voxel
    X, Y, Z = np.meshgrid(gx, gx, gz, indexing="ij")
    base = center - (center @ n + plane.d) * n          # 중심을 바닥에 투영
    P = base + X.reshape(-1, 1) * u + Y.reshape(-1, 1) * v + Z.reshape(-1, 1) * n
    inside_cnt = np.zeros(len(P), np.int32); hit_cnt = np.zeros(len(P), np.int32)
    for vw in views:
        if vw.mask is None or vw.mask.max() == 0:
            continue
        uv, inside = vw.project(P)
        if not inside.any():
            continue
        ii = np.nonzero(inside)[0]
        xs = uv[ii, 0].astype(np.int32); ys = uv[ii, 1].astype(np.int32)
        hit = vw.mask[ys, xs] > 0
        inside_cnt[ii] += 1
        hit_cnt[ii] += hit
    ratio = np.where(inside_cnt > 0, hit_cnt / np.maximum(inside_cnt, 1), 0.0)
    occ = (inside_cnt >= min_views) & (ratio >= vote_ratio - 1e-9)
    log(f"복셀 {res}x{res}x{nz} (한 변 {voxel:.4g} SfM단위), 물체 복셀 {int(occ.sum())}")
    return HullGrid(base - radius * u - radius * v, u, v, n, voxel, occ.reshape(res, res, nz), ratio.reshape(res, res, nz))


def hull_metrics(grid: HullGrid, scale: Optional[float]) -> dict:
    """scale: SfM 단위 → m. None 이면 SfM 단위 그대로(상대값)."""
    occ = grid.occ
    s = scale if scale else 1.0
    unit = "m" if scale else "sfm"
    vox = grid.voxel * s
    col = occ.any(axis=2)                       # 바닥 투영
    heights = np.where(col, (occ * np.arange(occ.shape[2])[None, None, :]).max(axis=2) + 1, 0) * vox
    # 높이별 단면적. 위에서만 찍은 orbit 은 윗면 위로 '지붕(피라미드)' 이 남아 height_max 가 과대해진다.
    # height_area90 = 단면적이 최대 단면적의 90 % 이상인 가장 높은 층 → 상자형 물체의 실제 높이에 가깝다 (경사진 물체는 과소).
    slice_area = occ.sum(axis=(0, 1)).astype(float)
    kmax = int(np.max(np.nonzero(slice_area >= 0.9 * slice_area.max())[0])) + 1 if slice_area.max() > 0 else 0
    h90 = kmax * vox
    out = {"unit": unit, "voxel": vox, "voxel_count": int(occ.sum()),
           "volume": float(occ.sum()) * vox ** 3,
           "volume_height_grid": float((heights * vox * vox).sum()),
           "volume_below_h90": float(occ[:, :, :kmax].sum()) * vox ** 3,
           "footprint_area": float(col.sum()) * vox * vox,
           "height_max": float(heights.max()), "height_p95": float(np.percentile(heights[col], 95)) if col.any() else 0.0,
           "height_area90": h90,
           "slice_area_profile": [round(float(a) * vox * vox, 6) for a in slice_area]}
    ys, xs = np.nonzero(col)
    if len(xs) >= 3:
        rect = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        (cx, cy), (w, h), ang = rect
        L, W = sorted([w * vox, h * vox], reverse=True)
        out.update({"length": L, "width": W, "bbox_volume": L * W * out["height_max"], "rect_angle_deg": ang,
                    "footprint_rect_fill": out["footprint_area"] / (L * W) if L * W > 0 else 0.0})
    return out


def sparse_object_metrics(P: np.ndarray, plane: Plane, center: np.ndarray, radius: float, scale: Optional[float],
                          h_min_rel: float = 0.1) -> dict:
    """희소 점군으로 교차 확인: 물체 반경 안에서 바닥 위로 뜬 점의 수와 최고 높이."""
    h = plane.height(P)
    rel = P - center
    r = np.linalg.norm(rel - np.outer(rel @ plane.n, plane.n), axis=1)
    sel = (r < radius) & (h > h_min_rel * radius)
    s = scale if scale else 1.0
    return {"points_above_floor": int(sel.sum()), "height_max": float(h[sel].max() * s) if sel.any() else 0.0,
            "height_p90": float(np.percentile(h[sel], 90) * s) if sel.any() else 0.0}


# ----------------------------------------------------------------------------- 7. 출력
def write_ply(path: str | Path, P: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    P = np.asarray(P, dtype=np.float32)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(P)}",
               "property float x", "property float y", "property float z"]
        if colors is not None:
            hdr += ["property uchar red", "property uchar green", "property uchar blue"]
        f.write(("\n".join(hdr) + "\nend_header\n").encode())
        if colors is not None:
            rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = P.T; rec["r"], rec["g"], rec["b"] = np.asarray(colors).T
            f.write(rec.tobytes())
        else:
            f.write(P.tobytes())


def hull_surface_points(grid: HullGrid) -> np.ndarray:
    occ = grid.occ
    pad = np.pad(occ, 1)
    interior = pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:]
    surf = occ & ~interior
    i, j, k = np.nonzero(surf)
    return grid.origin + ((i + 0.5) * grid.voxel)[:, None] * grid.u + ((j + 0.5) * grid.voxel)[:, None] * grid.v + ((k + 0.5) * grid.voxel)[:, None] * grid.n


def overlay_images(views: list[View], surf: np.ndarray, out_dir: Path, every: int = 8) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for v in views[::every]:
        img = cv2.imread(str(v.image_path))
        if img is None:
            continue
        if v.mask is not None:
            img[v.mask > 0] = (0.6 * img[v.mask > 0] + np.array([0, 0, 100])).astype(np.uint8)
        uv, inside = v.project(surf)
        for x, y in uv[inside].astype(int):
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        p = out_dir / f"overlay_{v.image_path.stem}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80]); outs.append(p)
    return outs


# ----------------------------------------------------------------------------- 파이프라인
def run_pipeline(video: Optional[str | Path], out_dir: str | Path, images_dir: Optional[str | Path] = None,
                 frames: int = 40, max_size: int = 1080, exhaustive: bool = False, force: bool = False,
                 mask_method: str = "auto", mask_dir: Optional[str | Path] = None,
                 ref: Optional[list] = None, ref_length_m: Optional[float] = None,
                 camera_height_m: Optional[float] = None, radius_factor: float = 1.6, zmax_factor: float = 1.5,
                 res: int = 96, vote_ratio: float = 1.0, num_threads: int = 4,
                 log: Callable[[str], None] = print) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if images_dir is None:
        images_dir = out / "frames"
        if not any(Path(images_dir).glob("*.jpg")) or force:
            files = extract_orbit_frames(video, images_dir, frames, max_size)
            log(f"프레임 {len(files)} 장 → {images_dir}")
    rec = run_sfm(images_dir, out / "sfm", max_size, num_threads, exhaustive=exhaustive, force=force, log=log)
    views = views_from_reconstruction(rec, images_dir)
    P, colors = sparse_points(rec)
    C = np.array([v.center for v in views])
    plane, inl = fit_plane_ransac(P, C)
    cam_h = float(np.mean(plane.height(C)))
    log(f"바닥 평면: 내점 {int(inl.sum())}/{len(P)} ({inl.mean() * 100:.0f}%), 카메라 평균 높이 {cam_h:.3f} (SfM 단위)")
    # 축척
    scale, scale_info = None, {"method": "none"}
    if ref and ref_length_m:
        scale, info = scale_from_reference(views, ref, ref_length_m)
        scale_info = {"method": "reference_length", "length_m": ref_length_m, **info}
    elif camera_height_m:
        scale = camera_height_m / cam_h
        scale_info = {"method": "camera_height(rough)", "camera_height_m": camera_height_m}
    if scale:
        log(f"축척 {scale:.5f} m/SfM단위 ({scale_info['method']}), 카메라 높이 ≈ {cam_h * scale:.2f} m")
    else:
        log("축척 정보 없음 → 결과는 SfM 상대 단위. --ref/--ref-length-m 또는 --camera-height-m 를 주면 미터로 환산")
    # 마스크 · 물체 위치
    center = build_masks(views, plane, mask_method, Path(mask_dir) if mask_dir else None, out / "masks", log=log)
    # 물체 크기 추정: 마스크 광선이 바닥에 닿는 범위
    ext = []
    for v in views:
        if v.mask is None or v.mask.max() == 0:
            continue
        ys, xs = np.nonzero(v.mask)
        sel = np.random.default_rng(0).choice(len(xs), min(200, len(xs)), replace=False)
        D = v.ray(np.c_[xs[sel], ys[sel]])
        for d in D:
            X = plane.intersect_ray(v.center, d)
            if X is not None:
                ext.append(np.linalg.norm(X - center))
    radius = float(np.percentile(ext, 95)) * radius_factor if ext else 0.2 * cam_h
    zmax = radius * zmax_factor
    # 카메라 고도각: 바닥을 얼마나 비스듬히 봤는가. 위에서만(고도각 큼) 찍으면 실루엣 교차가 위로 길쭉해져 높이를 못 잡는다
    elev = []
    for v in views:
        d = v.M[2, :3]                      # 카메라 z축(시선) 의 세계 좌표 표현
        elev.append(np.degrees(np.arcsin(abs(d @ plane.n) / np.linalg.norm(d))))
    elev = np.array(elev)
    # 희소 점군의 물체 윗면 높이로 복셀 영역 높이를 제한 (점이 충분할 때만)
    sm_raw = sparse_object_metrics(P, plane, center, radius, None)
    height_cap = None
    if sm_raw["points_above_floor"] >= 10:
        height_cap = sm_raw["height_p90"] * 1.2
        zmax = min(zmax, height_cap)
        log(f"희소 점 {sm_raw['points_above_floor']} 개로 물체 높이 상한 {height_cap:.3f} (SfM단위) 적용")
    else:
        log("물체 위 희소 점이 10개 미만 → 높이 상한 없음. 측면 뷰가 부족하면 높이가 과대해질 수 있음")
    grid = visual_hull(views, plane, center, radius, zmax, res, vote_ratio, log=log)
    hm = hull_metrics(grid, scale)
    sm = sparse_object_metrics(P, plane, center, radius, scale)
    diag = []
    if np.min(elev) > 50:
        diag.append(f"모든 프레임이 바닥을 {np.min(elev):.0f}° 이상 내려다봄 → 옆면이 거의 안 보여 높이 불확실. 다음 촬영은 카메라를 낮춰(고도각 20~40°) 한 바퀴 더 돌 것")
    if height_cap is None and hm["height_max"] > 1.2 * max(hm.get("length", 0), 1e-9):
        diag.append("높이가 길이보다 큼 + 희소 점 상한 없음 → visual hull 기둥 현상 의심")
    if sm["points_above_floor"] >= 10 and hm["height_max"] > 0:
        diag.append(f"희소 점 윗면 높이 {sm['height_p90']:.4g} vs hull 높이 {hm['height_max']:.4g} ({hm['unit']})")
    # 출력
    write_ply(out / "sparse_points.ply", P, colors)
    surf = hull_surface_points(grid)
    write_ply(out / "hull_surface.ply", surf)
    overlays = overlay_images(views, surf, out / "overlay")
    result = {
        "video": str(video) if video else None, "images": str(images_dir), "n_images_registered": len(views),
        "n_points3D": int(len(P)), "plane_inlier_ratio": float(inl.mean()), "camera_height_mean": cam_h * (scale or 1.0),
        "scale": scale, "scale_info": scale_info, "object_center_sfm": center.tolist(),
        "mask_method": mask_method if mask_method != "auto" else ("rembg" if rembg_available() else "bright"),
        "hull_region": {"radius": radius * (scale or 1.0), "zmax": zmax * (scale or 1.0), "res": res, "vote_ratio": vote_ratio,
                        "height_cap_from_sparse": (height_cap * (scale or 1.0)) if height_cap else None},
        "camera_elevation_deg": {"min": float(elev.min()), "median": float(np.median(elev)), "max": float(elev.max())},
        "hull": hm, "sparse_check": sm, "diagnosis": diag,
        "notes": [
            "visual hull 은 실루엣 교차라 오목한 부분을 못 깎는다 → 부피는 상한 쪽. 상자처럼 볼록한 물체에 적합",
            "바닥에 닿은 면은 바닥 평면과 같다고 가정 (바닥 아래/밑면 들림은 못 본다)",
            "축척 오차 ×3 ≈ 부피 오차. 기준 길이는 길수록, 두 뷰의 시차가 클수록 좋다",
            "마스크가 틀린 프레임이 vote_ratio 비율 이상이면 결과가 깎이거나 부풀 수 있다 → masks/ 와 overlay/ 로 확인",
        ],
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _write_report(out / "report.md", result, overlays)
    return result


def _fmt(x, unit):
    return f"{x:.4g} {unit}" if isinstance(x, (int, float)) else str(x)


def _write_report(path: Path, r: dict, overlays: list[Path]) -> None:
    h = r["hull"]; u = h["unit"]
    lines = [f"# 물체 3D 복원·치수 추정 결과", "",
             f"- 입력: {r['video'] or r['images']} / 등록 이미지 {r['n_images_registered']} 장, 희소 점 {r['n_points3D']}",
             f"- 바닥 평면 내점 비율 {r['plane_inlier_ratio'] * 100:.0f} %, 카메라 평균 높이 {_fmt(r['camera_height_mean'], u)}",
             f"- 축척: {r['scale_info']['method']}" + (f" (기준 길이 {r['scale_info'].get('length_m')} m, 재투영 오차 px {r['scale_info'].get('reproj_err_px_sum')})" if r['scale'] else " → **아래 값은 상대 단위**"),
             "", "## 치수 (visual hull)", "",
             "| 항목 | 값 |", "|---|---|",
             f"| 길이 × 너비 (바닥 투영 최소 외접 사각형) | {_fmt(h.get('length'), u)} × {_fmt(h.get('width'), u)} |",
             f"| 높이 (단면적 90 % 기준, 상자형에 적합) | {_fmt(h['height_area90'], u)} |",
             f"| 높이 (최대 / 95%, 위에서만 찍었으면 과대) | {_fmt(h['height_max'], u)} / {_fmt(h['height_p95'], u)} |",
             f"| 바닥 투영 면적 | {_fmt(h['footprint_area'], u + '²')} (사각형 채움률 {h.get('footprint_rect_fill', 0):.2f}) |",
             f"| 부피 (복셀 합, 단면적 90 % 높이까지) | {_fmt(h['volume_below_h90'], u + '³')} |",
             f"| 부피 (복셀 합, 전체 hull) | {_fmt(h['volume'], u + '³')} |",
             f"| 부피 (높이 격자) | {_fmt(h['volume_height_grid'], u + '³')} |",
             f"| 부피 (외접 상자 L×W×H, 상한) | {_fmt(h.get('bbox_volume'), u + '³')} |",
             f"| 복셀 한 변 | {_fmt(h['voxel'], u)} ({h['voxel_count']} 개) |",
             "", f"- 희소 점군 교차 확인: 물체 반경 안 바닥 위 점 {r['sparse_check']['points_above_floor']} 개, 최고 높이 {_fmt(r['sparse_check']['height_max'], u)}",
             f"- 카메라 고도각(바닥 기준): 최소 {r['camera_elevation_deg']['min']:.0f}° / 중앙 {r['camera_elevation_deg']['median']:.0f}° / 최대 {r['camera_elevation_deg']['max']:.0f}°",
             "", "## 진단", ""] + [f"- {d}" for d in (r["diagnosis"] or ["특이 사항 없음"])] + \
            ["", "## 확인용 이미지", ""] + [f"- {p.name}" for p in overlays] + ["", "## 주의", ""] + [f"- {n}" for n in r["notes"]]
    path.write_text("\n".join(lines), encoding="utf-8")
