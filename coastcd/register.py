"""
영상 정합(co-registration).

두 시기 영상을 "같은 픽셀이 같은 땅을 가리키도록" 맞추는 단계입니다.
비유: 투명 필름 두 장을 겹칠 때 모서리(특징점) 를 맞추는 일.

제공 기능
---------
match_features        SIFT/ORB 특징점 + 비율 검정 (기본, CPU, 학습 불필요)
match_loftr           kornia LoFTR 딥러닝 매칭 (선택. torch/kornia 설치 시. 확인 필요)
estimate_transform    MAGSAC++ 로 이상치 제거하며 similarity/affine/homography 추정
phase_shift           위상상관으로 전역 이동량만 빠르게 추정 (이종 센서 coarse 단계)
register              위 단계를 묶은 함수
synthetic_benchmark   정답을 아는 합성 변환으로 정합 정확도(px, m) 를 측정

정합 정확도를 숫자로 보여 주는 방법 (심사용)
--------------------------------------------
2시기 영상이 없어도, 1시기 영상에 알려진 이동·회전·축척·밝기 변화를 걸어
"흔들린 사본" 을 만들고 다시 원본에 맞춘 뒤 잔차를 재면 됩니다.
정답을 알기 때문에 오차를 픽셀 단위로 정확히 말할 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ------------------------------------------------------------------ 특징점 매칭

def match_features(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    method: str = "sift",
    max_features: int = 8000,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 그레이스케일 영상 사이의 대응점 (pts_a, pts_b) 를 돌려줍니다. 각 (N, 2) float32, (x, y).
    mask 가 있으면 그 영역(예: 육지) 에서만 특징점을 찾습니다. 바다는 특징점이 없고
    파도 무늬가 잘못 매칭되므로 육지 마스크를 넘기는 것을 권합니다.
    """
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    elif method == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"알 수 없는 method: {method}")

    ma = None if mask_a is None else (mask_a > 0).astype(np.uint8) * 255
    mb = None if mask_b is None else (mask_b > 0).astype(np.uint8) * 255
    kp_a, des_a = det.detectAndCompute(gray_a, ma)
    kp_b, des_b = det.detectAndCompute(gray_b, mb)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts_a, pts_b


def match_loftr(gray_a: np.ndarray, gray_b: np.ndarray, max_side: int = 1024, min_conf: float = 0.5):
    """
    kornia LoFTR(outdoor 가중치) 로 대응점을 찾습니다. torch + kornia 가 필요합니다.
        pip install torch kornia
    확인 필요: kornia 버전에 따라 API 가 다를 수 있음 (0.7 기준으로 작성).
    영상은 max_side 이하로 축소해서 넣고, 결과 좌표는 원래 크기로 되돌립니다.
    """
    import torch
    import kornia.feature as KF

    def prep(g):
        s = max(g.shape) / max_side
        s = max(s, 1.0)
        h = int(g.shape[0] / s) // 8 * 8
        w = int(g.shape[1] / s) // 8 * 8
        r = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(r).float()[None, None] / 255.0
        return t, (g.shape[1] / w, g.shape[0] / h)

    ta, sa = prep(gray_a)
    tb, sb = prep(gray_b)
    matcher = KF.LoFTR(pretrained="outdoor").eval()
    with torch.no_grad():
        out = matcher({"image0": ta, "image1": tb})
    conf = out["confidence"].numpy()
    keep = conf > min_conf
    pa = out["keypoints0"].numpy()[keep] * np.array(sa, np.float32)
    pb = out["keypoints1"].numpy()[keep] * np.array(sb, np.float32)
    return pa.astype(np.float32), pb.astype(np.float32)


# ------------------------------------------------------------------ 변환 추정

@dataclass
class Registration:
    matrix: np.ndarray                    # 2x3 (similarity/affine) 또는 3x3 (homography). B 픽셀 -> A 픽셀
    model: str
    n_matches: int
    n_inliers: int
    inlier_rmse_px: float
    notes: list[str] = field(default_factory=list)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """B 픽셀 좌표 (N,2) 를 A 픽셀 좌표로 변환."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.matrix.shape == (3, 3):
            h = np.hstack([pts, np.ones((len(pts), 1))]) @ self.matrix.T
            return h[:, :2] / h[:, 2:3]
        return pts @ self.matrix[:, :2].T + self.matrix[:, 2]

    def warp(self, img_b: np.ndarray, shape_a: tuple[int, int], nearest: bool = False) -> np.ndarray:
        """영상 B 를 A 의 픽셀 격자로 다시 그립니다."""
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        h, w = shape_a
        if self.matrix.shape == (3, 3):
            return cv2.warpPerspective(img_b, self.matrix, (w, h), flags=flags)
        return cv2.warpAffine(img_b, self.matrix, (w, h), flags=flags)


def estimate_transform(
    pts_b: np.ndarray,
    pts_a: np.ndarray,
    model: str = "similarity",
    reproj_px: float = 3.0,
) -> Registration:
    """
    pts_b -> pts_a 로 가는 변환을 MAGSAC++ 로 추정합니다.
    model: similarity(이동+회전+축척, 위성 정사영상끼리 권장) / affine / homography(드론 등 시점차 클 때)
    """
    notes = []
    n = len(pts_a)
    if n < 4:
        raise RuntimeError(f"대응점이 부족합니다 ({n}개). 마스크/특징점 수를 조정하세요.")
    if model == "similarity":
        fn = cv2.estimateAffinePartial2D
    elif model == "affine":
        fn = cv2.estimateAffine2D
    elif model == "homography":
        fn = cv2.findHomography
    else:
        raise ValueError(f"알 수 없는 model: {model}")

    # MAGSAC++ 를 우선 시도하고, 이 OpenCV 빌드가 지원하지 않으면 RANSAC 으로 내려갑니다.
    # (OpenCV 5.0 의 estimateAffine* 는 USAC 계열을 받지 않는 것으로 확인됨. findHomography 는 지원)
    methods = []
    if hasattr(cv2, "USAC_MAGSAC"):
        methods.append(("MAGSAC++", cv2.USAC_MAGSAC))
    methods.append(("RANSAC", cv2.RANSAC))
    M = inl = None
    for name, method in methods:
        try:
            M, inl = fn(pts_b, pts_a, method=method, ransacReprojThreshold=reproj_px)
            if name != "MAGSAC++":
                notes.append(f"{name} 사용 (MAGSAC++ 미지원 빌드)")
            break
        except cv2.error:
            continue
    if M is None:
        raise RuntimeError("변환 추정 실패 (이상치가 너무 많음).")

    inl = inl.ravel().astype(bool)
    reg = Registration(matrix=M, model=model, n_matches=n, n_inliers=int(inl.sum()), inlier_rmse_px=0.0, notes=notes)
    if inl.any():
        res = np.linalg.norm(reg.apply(pts_b[inl]) - pts_a[inl], axis=1)
        reg.inlier_rmse_px = float(np.sqrt(np.mean(res**2)))
    return reg


def phase_shift(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float, float]:
    """
    위상상관으로 B 가 A 대비 얼마나 밀려 있는지 (dx, dy, response) 를 돌려줍니다.
    같은 GSD 로 리샘플한 이종 센서 영상의 전역 이동량 추정에 씁니다 (coarse 단계).
    """
    a = cv2.GaussianBlur(gray_a.astype(np.float32), (0, 0), 1.0)
    b = cv2.GaussianBlur(gray_b.astype(np.float32), (0, 0), 1.0)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def register(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    model: str = "similarity",
    method: str = "sift",
    reproj_px: float = 3.0,
) -> Registration:
    """B 를 A 에 맞추는 변환(B px -> A px) 을 한 번에 구합니다."""
    if method == "loftr":
        pa, pb = match_loftr(gray_a, gray_b)
    else:
        pa, pb = match_features(gray_a, gray_b, mask_a, mask_b, method=method)
    return estimate_transform(pb, pa, model=model, reproj_px=reproj_px)


# ------------------------------------------------------------------ 합성 벤치마크

@dataclass
class BenchResult:
    trial: int
    true_dx: float
    true_dy: float
    true_rot_deg: float
    true_scale: float
    n_matches: int
    n_inliers: int
    before_rmse_px: float   # 정합 안 했을 때 오차
    after_rmse_px: float    # 정합 후 잔차
    after_max_px: float

    def row(self, pixel_m: float) -> dict:
        return {
            "trial": self.trial,
            "true_dx_px": round(self.true_dx, 2),
            "true_dy_px": round(self.true_dy, 2),
            "true_rot_deg": round(self.true_rot_deg, 3),
            "true_scale": round(self.true_scale, 4),
            "matches": self.n_matches,
            "inliers": self.n_inliers,
            "before_rmse_px": round(self.before_rmse_px, 2),
            "after_rmse_px": round(self.after_rmse_px, 3),
            "after_max_px": round(self.after_max_px, 3),
            "after_rmse_m": round(self.after_rmse_px * pixel_m, 3),
        }


def make_synthetic_pair(
    gray: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    max_shift_px: float = 30.0,
    max_rot_deg: float = 2.0,
    max_scale_dev: float = 0.02,
    gain_range: tuple[float, float] = (0.8, 1.2),
    bias_range: tuple[float, float] = (-20, 20),
    noise_sigma: float = 4.0,
    hard: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    원본(A) 에 알려진 similarity 변환 + 밝기 변화 + 잡음을 걸어 "다른 시기처럼 보이는" B 를 만듭니다.
    hard=True 면 추가로
      - 해상도 차이: 1.6배 축소 후 복원 (0.5 m 리샘플 vs 0.78 m 실제 GSD 흉내)
      - 블러: 시그마 0.5~1.5 px (초점·대기 차이)
      - 가림: 육지의 일부를 밝은 얼룩(구름·파도 거품 흉내) 으로 덮음
      - 국소 밝기 변화: 넓은 그라데이션 (태양각 차이)
    반환: (gray_b, valid_b, M_true (A px -> B px, 2x3), params)
    """
    h, w = gray.shape
    dx = rng.uniform(-max_shift_px, max_shift_px)
    dy = rng.uniform(-max_shift_px, max_shift_px)
    rot = rng.uniform(-max_rot_deg, max_rot_deg)
    scale = 1.0 + rng.uniform(-max_scale_dev, max_scale_dev)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[:, 2] += (dx, dy)

    g = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    v = cv2.warpAffine(valid.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    gain = rng.uniform(*gain_range)
    bias = rng.uniform(*bias_range)
    g = g.astype(np.float32) * gain + bias + rng.normal(0, noise_sigma, g.shape).astype(np.float32)
    if hard:
        small = cv2.resize(g, (max(1, int(w / 1.6)), max(1, int(h / 1.6))), interpolation=cv2.INTER_AREA)
        g = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(g, (0, 0), float(rng.uniform(0.5, 1.5)))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = g * (1.0 + 0.15 * (xx / w - 0.5) * rng.choice([-1, 1]) + 0.15 * (yy / h - 0.5) * rng.choice([-1, 1]))
        n_blobs = int(rng.integers(3, 8))
        for _ in range(n_blobs):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            ax, ay = int(rng.integers(w // 40, w // 12)), int(rng.integers(h // 40, h // 12))
            cv2.ellipse(g, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 235.0, -1)
    g = np.clip(g, 0, 255).astype(np.uint8)
    g[~v] = 0
    return g, v, M, dict(dx=dx, dy=dy, rot=rot, scale=scale, gain=gain, bias=bias, hard=hard)


def synthetic_benchmark(
    gray: np.ndarray,
    valid: np.ndarray,
    land_mask: np.ndarray | None = None,
    n_trials: int = 5,
    seed: int = 0,
    model: str = "similarity",
    method: str = "sift",
    grid_step: int = 50,
    **pair_kwargs,
) -> list[BenchResult]:
    """
    합성 변환 벤치마크. 각 시도마다:
      1) B = 알려진 변환(A)  2) B 를 A 에 정합  3) A 유효영역 격자점에서 잔차 계산
    land_mask 를 주면 특징점을 육지에서만 찾습니다 (바다 파도 오매칭 방지).
    """
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    ys, xs = np.mgrid[grid_step // 2 : h : grid_step, grid_step // 2 : w : grid_step]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    grid = grid[valid[grid[:, 1].astype(int), grid[:, 0].astype(int)]]

    results = []
    for t in range(n_trials):
        gb, vb, M_true, p = make_synthetic_pair(gray, valid, rng, **pair_kwargs)
        mask_b = None
        if land_mask is not None:
            mask_b = cv2.warpAffine(land_mask, M_true, (w, h), flags=cv2.INTER_NEAREST) > 0
        reg = register(gray, gb, land_mask, mask_b, model=model, method=method)

        q = grid @ M_true[:, :2].T + M_true[:, 2]           # A 격자점이 B 에서 있는 위치 (정답)
        inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        q, g = q[inside], grid[inside]
        before = np.linalg.norm(q - g, axis=1)
        after = np.linalg.norm(reg.apply(q) - g, axis=1)
        results.append(
            BenchResult(
                trial=t,
                true_dx=p["dx"], true_dy=p["dy"], true_rot_deg=p["rot"], true_scale=p["scale"],
                n_matches=reg.n_matches, n_inliers=reg.n_inliers,
                before_rmse_px=float(np.sqrt(np.mean(before**2))),
                after_rmse_px=float(np.sqrt(np.mean(after**2))),
                after_max_px=float(after.max()),
            )
        )
    return results


# ------------------------------------------------------------------ 지오레퍼런스 갱신

def georef_from_registration(reg: Registration, transform_a):
    """
    B 를 A 에 정합한 결과(B px -> A px) 로 B 의 새 geotransform 을 만듭니다.
    픽셀 값은 그대로 두고 "이 픽셀이 어디인가" 만 고칩니다 (fix_georeferencing.py 와 같은 원리).
    homography 는 affine 으로 표현이 안 되므로 지원하지 않습니다.
    """
    from affine import Affine

    M = reg.matrix
    if M.shape != (2, 3):
        raise ValueError("homography 결과는 geotransform 으로 표현할 수 없습니다. warp 를 쓰세요.")
    pix_b_to_pix_a = Affine(M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])
    from .raster_io import compose

    return compose(transform_a, pix_b_to_pix_a)
