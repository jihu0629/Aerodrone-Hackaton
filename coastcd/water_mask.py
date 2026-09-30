"""
수륙분할(육지 vs 물) 마스크.

1) RGB 베이스라인 (SkySat 처럼 NIR 이 없는 영상)
   - (A) 밝기: Otsu 임계값보다 밝으면 육지
   - (B) 질감: 주변 픽셀 밝기 표준편차가 크면 육지 (그늘진 숲 보완)
   - (A) OR (B) -> 육지.  extract_coastline.py 의 논리를 그대로 옮기되,
     임계값을 오버뷰에서 한 번만 추정하고 모든 타일에 같은 값을 적용합니다.
     (타일마다 Otsu 를 다시 계산하면 바다만 있는 타일에서 임계값이 엉뚱하게 잡힙니다.)

2) Sentinel-2 NDWI (NIR 이 있는 영상)
   NDWI = (Green - NIR) / (Green + NIR).  물은 NIR 을 거의 반사하지 않아 값이 큽니다.

한계 (그대로 유지)
------------------
그림자, 젖은 모래, 파도 거품 오탐. 이 모듈은 베이스라인이며, 나중에
분할 모델(SAM 프롬프트 / U-Net) 로 교체할 수 있도록 입출력을 단순하게 유지합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from rasterio.windows import Window

from .raster_io import RasterChunk, core_slice, iter_tiles, open_raster, read_window


@dataclass
class Thresholds:
    brightness: float        # 0~255 그레이스케일
    texture_std: float       # 국소 표준편차 (그레이스케일 단위)
    texture_window: int = 7  # 표준편차 계산 창 (픽셀)
    blur: int = 5            # 밝기 판정 전 가우시안 블러 커널

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 베이스라인

def local_std(gray: np.ndarray, ksize: int) -> np.ndarray:
    """각 픽셀 주변 ksize x ksize 표준편차 (float32)."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (ksize, ksize))
    sq_mean = cv2.blur(g * g, (ksize, ksize))
    return np.sqrt(np.clip(sq_mean - mean * mean, 0, None))


def _otsu(values: np.ndarray) -> float:
    """1차원 uint8 배열의 Otsu 임계값."""
    values = np.ascontiguousarray(values.reshape(1, -1).astype(np.uint8))
    thr, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def estimate_thresholds(
    rgb: np.ndarray,
    valid: np.ndarray,
    texture_window: int = 7,
    blur: int = 5,
    texture_cap: float = 40.0,
) -> Thresholds:
    """
    유효 영역 픽셀만으로 밝기/질감 Otsu 임계값을 추정합니다.
    보통 오버뷰(또는 섬 크롭의 축소본) 에서 한 번 호출합니다.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    if not np.any(valid):
        raise ValueError("유효 픽셀이 없습니다 (알파 밴드가 전부 0).")
    brightness = _otsu(blurred[valid])

    std = local_std(gray, texture_window)
    std_u8 = (np.clip(std, 0, texture_cap) / texture_cap * 255).astype(np.uint8)
    texture_thr = _otsu(std_u8[valid]) / 255.0 * texture_cap
    return Thresholds(brightness=brightness, texture_std=texture_thr, texture_window=texture_window, blur=blur)


def apply_land_mask(rgb: np.ndarray, valid: np.ndarray, thr: Thresholds, clean_kernel: int = 5) -> np.ndarray:
    """정해진 임계값으로 육지 마스크(uint8, 255 = 육지) 를 만듭니다."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (thr.blur, thr.blur), 0)
    bright = blurred > thr.brightness
    textured = local_std(gray, thr.texture_window) > thr.texture_std
    land = (bright | textured) & valid
    land_u8 = land.astype(np.uint8) * 255
    if clean_kernel > 1:
        k = np.ones((clean_kernel, clean_kernel), np.uint8)
        land_u8 = cv2.morphologyEx(land_u8, cv2.MORPH_CLOSE, k)
        land_u8 = cv2.morphologyEx(land_u8, cv2.MORPH_OPEN, k)
        land_u8[~valid] = 0
    return land_u8


def clean_mask(mask: np.ndarray, min_area_px: int, fill_holes_px: int = 0) -> np.ndarray:
    """min_area_px 보다 작은 덩어리 제거, fill_holes_px 보다 작은 구멍 메우기."""
    out = _remove_small(mask, min_area_px)
    if fill_holes_px > 0:
        inv = cv2.bitwise_not(out)
        inv = _remove_small(inv, fill_holes_px)
        out = cv2.bitwise_not(inv)
    return out


def _remove_small(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return (keep[labels]).astype(np.uint8) * 255


# ------------------------------------------------------------------ 타일 단위 처리

def tiled_land_mask(
    path: str | Path,
    region: Window,
    thr: Thresholds,
    tile: int = 2048,
    overlap: int = 64,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, RasterChunk]:
    """
    region(원본 픽셀 좌표) 을 타일로 읽어 원본 해상도 육지 마스크를 만듭니다.

    반환: (land_mask uint8 (H,W), valid bool (H,W), 참조용 RasterChunk(첫 타일의 crs 와 region 의 transform))
    메모리: region 크기의 uint8 두 장만 유지 (8000 x 6000 이면 약 100 MB).
    """
    H, W = int(region.height), int(region.width)
    land = np.zeros((H, W), dtype=np.uint8)
    valid_all = np.zeros((H, W), dtype=bool)

    with open_raster(path) as src:
        region_transform = src.window_transform(region)
        crs = src.crs

    tiles = list(iter_tiles(region, tile=tile, overlap=overlap))
    for i, (read, core) in enumerate(tiles):
        chunk = read_window(path, read)
        m = apply_land_mask(chunk.rgb, chunk.valid, thr)
        rs, cs = core_slice(read, core)
        r0 = int(core.row_off - region.row_off)
        c0 = int(core.col_off - region.col_off)
        land[r0 : r0 + int(core.height), c0 : c0 + int(core.width)] = m[rs, cs]
        valid_all[r0 : r0 + int(core.height), c0 : c0 + int(core.width)] = chunk.valid[rs, cs]
        if progress:
            print(f"  타일 {i + 1}/{len(tiles)} 완료", end="\r")
    if progress:
        print()
    ref = RasterChunk(rgb=np.zeros((1, 1, 3), np.uint8), valid=valid_all, transform=region_transform, crs=crs, window=region)
    return land, valid_all, ref


# ------------------------------------------------------------------ Sentinel-2 NDWI

def ndwi_water_mask(s2_tif: str | Path, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray, object, object]:
    """
    s2_download.py 가 만든 5밴드 GeoTIFF(B02,B03,B04,B08,B11, 반사율x10000, nodata 0) 에서
    NDWI 물 마스크를 만듭니다.

    threshold=None 이면 유효 픽셀의 NDWI 에 Otsu 를 적용합니다 (보통 0 근처).
    반환: (water uint8 255=물, valid bool, transform, crs)
    """
    with open_raster(s2_tif) as src:
        desc = [d or "" for d in src.descriptions]
        try:
            g_idx = next(i for i, d in enumerate(desc) if "B03" in d) + 1
            n_idx = next(i for i, d in enumerate(desc) if "B08" in d) + 1
        except StopIteration:
            g_idx, n_idx = 2, 4  # s2_download.py 의 밴드 순서
        green = src.read(g_idx).astype(np.float32)
        nir = src.read(n_idx).astype(np.float32)
        transform, crs = src.transform, src.crs

    valid = (green > 0) & (nir > 0)
    ndwi = np.zeros_like(green)
    denom = green + nir
    np.divide(green - nir, denom, out=ndwi, where=denom > 0)

    if threshold is None:
        u8 = ((np.clip(ndwi, -1, 1) + 1) / 2 * 255).astype(np.uint8)
        t8 = _otsu(u8[valid])
        low, high = valid & (u8 <= t8), valid & (u8 > t8)
        if low.any() and high.any():
            # Otsu 가 고른 클래스 경계를 두 클래스 평균의 중간값으로 다듬음 (양자화 오차 제거)
            threshold = float((ndwi[low].mean() + ndwi[high].mean()) / 2)
        else:
            threshold = t8 / 255.0 * 2 - 1
    water = ((ndwi > threshold) & valid).astype(np.uint8) * 255
    return water, valid, transform, crs
