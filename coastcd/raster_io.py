"""
COG(Cloud-Optimized GeoTIFF) 를 메모리에 통째로 올리지 않고 읽는 도구.

세 가지 읽기 방식
-----------------
1) read_overview  : 내장 오버뷰로 축소본만 읽음 (전체 파악, 임계값 추정용)
2) read_window    : 원본 해상도로 지정 영역만 읽음
3) iter_tiles     : 큰 영역을 겹침(overlap) 있는 타일로 나눠 순회

유효 영역(nodata) 마스크
-----------------------
SkySat ortho_visual 은 4번 밴드가 알파(0 = 여백, 255 = 유효)입니다.
"검은 픽셀 = 여백" 으로 추정하던 이전 방식은 그림자를 여백으로 오인할 수
있으므로, 알파 밴드가 있으면 항상 그것을 씁니다. 알파가 없으면 GDAL 의
데이터셋 마스크(nodata 값 기반)를 씁니다.

비유: 800 MB 사진은 벽 한 면을 덮는 대형 지도입니다. 전체를 보려면
멀리서 축소해 보고(오버뷰), 자세히 보려면 돋보기로 한 구역씩(윈도우/타일) 봅니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

from .config import safe_path


def compose(a: Affine, b: Affine) -> Affine:
    """a ∘ b (b 를 먼저 적용). affine 라이브러리 버전에 따라 @ 또는 * 를 씁니다."""
    try:
        return a @ b  # affine >= 2.4
    except TypeError:
        return a * b


@dataclass
class RasterChunk:
    """읽어 온 영상 조각. rgb 는 (H, W, 3) uint8, valid 는 (H, W) bool."""

    rgb: np.ndarray
    valid: np.ndarray
    transform: Affine          # 이 조각의 픽셀 -> 좌표계 변환
    crs: object
    window: Window | None      # 원본 픽셀 기준 위치 (오버뷰면 None)
    scale: float = 1.0         # 원본 대비 축소 배율 (오버뷰 전용. 1 = 원본 해상도)

    @property
    def pixel_size_m(self) -> float:
        return abs(self.transform.a)

    def to_gray(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)


# ------------------------------------------------------------------ 열기 / 메타데이터

def open_raster(path: str | Path) -> rasterio.io.DatasetReader:
    """한글 경로를 우회해서 rasterio 로 엽니다."""
    return rasterio.open(safe_path(path))


def describe(path: str | Path) -> dict:
    """크기, 밴드, 좌표계, 오버뷰 등 기본 정보를 dict 로 돌려주고 출력합니다."""
    with open_raster(path) as src:
        info = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "bounds": tuple(src.bounds),
            "res_m": src.res,
            "overviews": src.overviews(1),
            "has_alpha": _has_alpha(src),
            "block_shapes": src.block_shapes[0],
        }
    for key, value in info.items():
        print(f"  {key:12s}: {value}")
    return info


def _has_alpha(src: rasterio.io.DatasetReader) -> bool:
    from rasterio.enums import ColorInterp

    if src.count < 4:
        return False
    try:
        return src.colorinterp[3] == ColorInterp.alpha
    except Exception:  # noqa: BLE001
        return src.count == 4


def _read_valid(src, out_shape=None, window=None) -> np.ndarray:
    """알파 밴드(우선) 또는 데이터셋 마스크로 유효 영역 bool 마스크를 읽습니다."""
    if _has_alpha(src):
        alpha = src.read(4, out_shape=out_shape, window=window, resampling=Resampling.nearest)
        return alpha > 0
    mask = src.read_masks(1, out_shape=out_shape, window=window, resampling=Resampling.nearest)
    return mask > 0


def _read_rgb(src, out_shape=None, window=None, resampling=Resampling.average) -> np.ndarray:
    bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
    shape = None if out_shape is None else (3, *out_shape)
    data = src.read(bands, out_shape=shape, window=window, resampling=resampling)
    rgb = np.transpose(data, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = _stretch_to_uint8(rgb)
    return np.ascontiguousarray(rgb)


def _stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0.0, 1.0)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


# ------------------------------------------------------------------ 읽기

def read_overview(path: str | Path, max_size: int = 2000) -> RasterChunk:
    """
    긴 변이 max_size 픽셀이 되도록 축소해서 전체를 읽습니다.
    COG 오버뷰를 쓰므로 800 MB 파일도 수 초면 됩니다.

    주의: max_size=2000 이면 SkySat 스트립(34447 px) 기준 1픽셀 = 약 8.6 m 입니다.
    해빈 폭 40 m 가 5픽셀도 안 되므로 오버뷰는 '전체 파악용' 이지 해안선 추출용이 아닙니다.
    """
    with open_raster(path) as src:
        scale = max(src.width, src.height) / float(max_size)
        scale = max(scale, 1.0)
        out_h = max(1, int(round(src.height / scale)))
        out_w = max(1, int(round(src.width / scale)))
        rgb = _read_rgb(src, out_shape=(out_h, out_w))
        valid = _read_valid(src, out_shape=(out_h, out_w))
        transform = compose(src.transform, Affine.scale(src.width / out_w, src.height / out_h))
        crs = src.crs
    return RasterChunk(rgb=rgb, valid=valid, transform=transform, crs=crs, window=None, scale=scale)


def read_window(path: str | Path, window: Window, downscale: int = 1) -> RasterChunk:
    """원본 픽셀 좌표 window 영역을 읽습니다. downscale=2 면 절반 크기로 축소해서 읽습니다."""
    with open_raster(path) as src:
        window = _clip_window(window, src.width, src.height)
        out_h = max(1, int(window.height) // downscale)
        out_w = max(1, int(window.width) // downscale)
        out_shape = None if downscale == 1 else (out_h, out_w)
        rgb = _read_rgb(src, out_shape=out_shape, window=window)
        valid = _read_valid(src, out_shape=out_shape, window=window)
        transform = src.window_transform(window)
        if downscale != 1:
            transform = compose(transform, Affine.scale(window.width / out_w, window.height / out_h))
        crs = src.crs
    return RasterChunk(rgb=rgb, valid=valid, transform=transform, crs=crs, window=window, scale=float(downscale))


def _clip_window(window: Window, width: int, height: int) -> Window:
    col0 = int(max(0, math.floor(window.col_off)))
    row0 = int(max(0, math.floor(window.row_off)))
    col1 = int(min(width, math.ceil(window.col_off + window.width)))
    row1 = int(min(height, math.ceil(window.row_off + window.height)))
    return Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))


def iter_tiles(
    region: Window,
    tile: int = 2048,
    overlap: int = 64,
) -> Iterator[tuple[Window, Window]]:
    """
    region(원본 픽셀 좌표) 을 tile x tile 크기, overlap 만큼 겹치는 타일로 나눕니다.

    yield: (read_window, core_window)
      read_window : 실제로 읽을 영역 (겹침 포함)
      core_window : 결과를 채워 넣을 영역 (겹침 제외). 경계 효과를 피하기 위해
                    타일 결과 중 core 부분만 최종 배열에 씁니다.
    """
    col_start, row_start = int(region.col_off), int(region.row_off)
    col_end = col_start + int(region.width)
    row_end = row_start + int(region.height)
    step = tile - 2 * overlap
    if step <= 0:
        raise ValueError("tile 은 overlap 의 2배보다 커야 합니다.")
    for row in range(row_start, row_end, step):
        for col in range(col_start, col_end, step):
            core = Window(col, row, min(step, col_end - col), min(step, row_end - row))
            read_col = max(col_start, col - overlap)
            read_row = max(row_start, row - overlap)
            read_col_end = min(col_end, col + step + overlap)
            read_row_end = min(row_end, row + step + overlap)
            read = Window(read_col, read_row, read_col_end - read_col, read_row_end - read_row)
            yield read, core


def core_slice(read: Window, core: Window) -> tuple[slice, slice]:
    """read_window 배열 안에서 core_window 에 해당하는 (row_slice, col_slice)."""
    r0 = int(core.row_off - read.row_off)
    c0 = int(core.col_off - read.col_off)
    return slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width))


# ------------------------------------------------------------------ 섬 영역 자동 탐색

def find_land_bbox(
    path: str | Path,
    max_size: int = 2000,
    margin_m: float = 300.0,
    all_islands: bool = False,
    min_area_m2: float = 20_000.0,
    merge_distance_m: float = 200.0,
) -> tuple[Window, RasterChunk]:
    """
    오버뷰에서 육지 덩어리를 찾아 원본 픽셀 좌표의 bbox(Window) 를 돌려줍니다.

    SkySat 스트립은 대부분 바다이고 굴업도는 전체의 5% 정도입니다.
    이 bbox 로 잘라내면 이후 모든 처리가 수십 배 빨라집니다.

    all_islands=False : 가장 큰 덩어리(굴업도) 하나만
    all_islands=True  : min_area_m2 이상 모든 덩어리를 포함하는 bbox
    merge_distance_m  : 이 거리 안에 있는 덩어리는 한 섬으로 봄. 오버뷰에서는 그늘진 숲이
                        바다로 빠져 섬이 여러 조각으로 갈라지기 때문에 필요합니다.
    """
    import cv2

    from .water_mask import estimate_thresholds, apply_land_mask

    ov = read_overview(path, max_size=max_size)
    thr = estimate_thresholds(ov.rgb, ov.valid)
    land = apply_land_mask(ov.rgb, ov.valid, thr)
    px_m = ov.pixel_size_m
    min_area_px = max(1, int(min_area_m2 / (px_m * px_m)))

    merge_px = int(merge_distance_m / px_m)
    merged = land
    if merge_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1, 2 * merge_px + 1))
        merged = cv2.dilate(land, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area_px:
            boxes.append((area, x, y, w, h))
    if not boxes:
        raise RuntimeError("육지 덩어리를 찾지 못했습니다. max_size 를 키우거나 min_area_m2 를 줄여 보세요.")
    boxes.sort(reverse=True)
    if not all_islands:
        boxes = boxes[:1]
    x0 = min(b[1] for b in boxes)
    y0 = min(b[2] for b in boxes)
    x1 = max(b[1] + b[3] for b in boxes)
    y1 = max(b[2] + b[4] for b in boxes)

    margin_px = margin_m / px_m
    x0, y0 = x0 - margin_px, y0 - margin_px
    x1, y1 = x1 + margin_px, y1 + margin_px

    s = ov.scale
    window = Window(int(x0 * s), int(y0 * s), int((x1 - x0) * s), int((y1 - y0) * s))
    with open_raster(path) as src:
        window = _clip_window(window, src.width, src.height)
    return window, ov


# ------------------------------------------------------------------ 쓰기

def write_geotiff(
    out_path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs,
    nodata=None,
    overviews: bool = True,
) -> None:
    """(H,W) 또는 (H,W,C) 배열을 타일형 GeoTIFF(deflate 압축) 로 저장합니다."""
    if array.ndim == 2:
        array = array[:, :, None]
    h, w, c = array.shape
    profile = dict(
        driver="GTiff",
        width=w,
        height=h,
        count=c,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=2 if array.dtype != np.uint8 else 1,
        nodata=nodata,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(safe_path(out_path), "w", **profile) as dst:
        for i in range(c):
            dst.write(array[:, :, i], i + 1)
        if overviews and max(h, w) > 1024:
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")


def export_crop(path: str | Path, window: Window, out_path: str | Path) -> RasterChunk:
    """원본 해상도로 window 영역을 잘라 RGBA GeoTIFF 로 저장하고, 읽은 조각을 돌려줍니다."""
    chunk = read_window(path, window)
    rgba = np.dstack([chunk.rgb, (chunk.valid.astype(np.uint8) * 255)])
    write_geotiff(out_path, rgba, chunk.transform, chunk.crs)
    with rasterio.open(safe_path(out_path), "r+") as dst:
        from rasterio.enums import ColorInterp

        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    return chunk
