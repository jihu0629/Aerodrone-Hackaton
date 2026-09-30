"""DSM 차분으로 물체 부피를 구한다.

V = Σ_(마스크 안 픽셀) max(DSM − 바닥, 0) × GSD²

바닥(모래 높이): 마스크 바깥 테두리(ring) 픽셀의 DSM 높이에 평면 z = a·x + b·y + c 를 최소제곱으로 맞춰
보간한다 (경사진 해변 대응). ring 픽셀이 너무 적거나 평면 맞춤이 불안정하면 중앙값으로 대체.

불확실성: ring 잔차의 표준편차 σ_z 를 DSM 노이즈로 보고, 부피 오차 ≈ σ_z × 면적 로 준다.
Kako et al. (2020) 과 같은 방식(경계선 → DSM 겹침 → 면적×높이). 속이 빈 물체도 겉 부피 전체로 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class ObjectVolume:
    obj_id: int
    class_name: str
    area_m2: float
    volume_m3: float
    volume_sigma_m3: float     # DSM 노이즈로 인한 부피 불확실성 (1σ)
    h_max_m: float
    h_mean_m: float
    ground_z_m: float          # 물체 중심에서의 추정 바닥 높이
    ring_sigma_m: float        # 링 잔차 표준편차 (= DSM 노이즈 추정)
    cx_px: float               # 마스크 중심 (픽셀, 열)
    cy_px: float               # 마스크 중심 (픽셀, 행)
    n_px: int
    n_valid_px: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _ring(mask: np.ndarray, ring_px: int, gap_px: int = 1) -> np.ndarray:
    """마스크 바깥 테두리. gap_px 만큼 띄우고(경계 흐림 회피) ring_px 두께."""
    m = mask.astype(np.uint8)
    k_gap = np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8)
    k_out = np.ones((2 * (gap_px + ring_px) + 1, 2 * (gap_px + ring_px) + 1), np.uint8)
    inner = cv2.dilate(m, k_gap)
    outer = cv2.dilate(m, k_out)
    return (outer > 0) & (inner == 0)


def fit_ground_plane(dsm: np.ndarray, ring: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, str]:
    """ring 픽셀로 바닥 평면을 맞춘다. 반환: (전체 격자 바닥 높이, 잔차 σ, 방법)."""
    ys, xs = np.nonzero(ring & valid)
    z = dsm[ys, xs].astype(np.float64)
    h, w = dsm.shape
    if len(z) < 6:
        med = float(np.median(z)) if len(z) else float(np.nanmedian(dsm[valid])) if valid.any() else 0.0
        return np.full(dsm.shape, med), 0.0, "median_fallback"
    A = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    sigma = float(res.std())
    # 강건화: 큰 잔차(다른 쓰레기·돌이 ring 에 들어온 경우) 제거 후 다시 맞춤
    if len(z) >= 12 and sigma > 0:
        keep = np.abs(res) < 2.0 * sigma
        if keep.sum() >= 6 and keep.sum() < len(z):
            coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
            res = z[keep] - A[keep] @ coef
            sigma = float(res.std())
    gy, gx = np.mgrid[0:h, 0:w]
    ground = coef[0] * gx + coef[1] * gy + coef[2]
    return ground, sigma, "plane"


def object_volume(dsm: np.ndarray, mask: np.ndarray, gsd_m: float, *, obj_id: int = 0,
                  class_name: str = "unknown", ring_px: int = 4, gap_px: int = 1,
                  nodata: float | None = None, confidence: float = 1.0) -> ObjectVolume:
    """한 물체의 부피. dsm: 2D float 배열(m), mask: 같은 크기 bool, gsd_m: DSM 픽셀 크기(m)."""
    mask = mask.astype(bool)
    valid = np.isfinite(dsm)
    if nodata is not None:
        valid &= dsm != nodata
    # 계산 범위를 물체 주변으로 잘라 속도 확보
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty mask")
    pad = ring_px + gap_px + 2
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, dsm.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, dsm.shape[1])
    d = dsm[y0:y1, x0:x1].astype(np.float64)
    m = mask[y0:y1, x0:x1]
    v = valid[y0:y1, x0:x1]
    ring = _ring(m, ring_px, gap_px)
    ground, sigma, _ = fit_ground_plane(d, ring, v)
    hgt = np.where(v, d - ground, 0.0)
    hgt[hgt < 0] = 0.0
    hm = hgt[m & v]
    n_valid = int((m & v).sum())
    area = n_valid * gsd_m ** 2
    vol = float(hm.sum()) * gsd_m ** 2
    cy, cx = ys.mean(), xs.mean()
    gz = float(ground[int(round(cy)) - y0, int(round(cx)) - x0]) if ground.size else 0.0
    return ObjectVolume(
        obj_id=obj_id, class_name=class_name, area_m2=area, volume_m3=vol,
        volume_sigma_m3=sigma * area, h_max_m=float(hm.max()) if n_valid else 0.0,
        h_mean_m=float(hm.mean()) if n_valid else 0.0, ground_z_m=gz, ring_sigma_m=sigma,
        cx_px=float(cx), cy_px=float(cy), n_px=int(m.sum()), n_valid_px=n_valid, confidence=confidence,
    )


def split_instances(class_mask: np.ndarray, min_px: int = 4) -> list[np.ndarray]:
    """의미 분할(클래스 마스크 1장) → 연결 요소 단위 인스턴스 마스크 목록."""
    n, lab = cv2.connectedComponents(class_mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        m = lab == i
        if m.sum() >= min_px:
            out.append(m)
    return out


def volumes_from_masks(dsm: np.ndarray, masks: list[tuple[str, np.ndarray, float]], gsd_m: float,
                       **kw) -> list[ObjectVolume]:
    """masks: [(class_name, bool mask, confidence), ...] → 부피 목록."""
    out = []
    for i, (cls, m, conf) in enumerate(masks):
        if not m.any():
            continue
        out.append(object_volume(dsm, m, gsd_m, obj_id=i, class_name=cls, confidence=conf, **kw))
    return out
