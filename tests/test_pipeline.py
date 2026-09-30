"""
합성 데이터로 파이프라인 전체를 검증합니다 (실제 SkySat 파일 없이 실행 가능).

    pip install pytest
    pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.coastline import mask_to_polygons, polygons_to_coastlines, save_geojson  # noqa: E402
from coastcd.raster_io import export_crop, find_land_bbox, iter_tiles, read_overview, read_window  # noqa: E402
from coastcd.register import register, synthetic_benchmark  # noqa: E402
from coastcd.water_mask import (  # noqa: E402
    apply_land_mask, clean_mask, estimate_thresholds, fill_enclosed_water, ndwi_water_mask,
    remove_thin_objects, tiled_land_mask,
)

CRS = "EPSG:32651"


def _synthetic_island(w=3000, h=2000, seed=1):
    """바다(어둡고 매끈) + 섬(밝거나 질감) + 대각선 nodata 여백을 가진 RGB/alpha 를 만듭니다."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # 섬: 타원 두 개 합집합
    cx, cy = w * 0.55, h * 0.5
    island = ((xx - cx) / 500) ** 2 + ((yy - cy) / 300) ** 2 < 1
    island |= ((xx - cx - 350) / 250) ** 2 + ((yy - cy + 150) / 200) ** 2 < 1
    # 바다: 푸르스름, 약한 잡음
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 0] = 40 + rng.normal(0, 2, (h, w))
    rgb[..., 1] = 60 + rng.normal(0, 2, (h, w))
    rgb[..., 2] = 70 + rng.normal(0, 2, (h, w))
    # 섬: 왼쪽 절반 어두운 숲(질감 강함), 오른쪽 밝은 모래
    forest = island & (xx < cx)
    sand = island & ~forest
    tex = rng.normal(0, 25, (h, w))
    # 수관 크기(약 8 px) 의 거친 질감: 축소본에서도 살아남는 무늬
    import cv2
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 4)
    tex = tex + coarse / (coarse.std() + 1e-6) * 25
    rgb[forest] = np.stack([35 + tex, 55 + tex, 30 + tex], -1)[forest]
    rgb[sand] = np.stack([190 + tex * 0.3, 180 + tex * 0.3, 150 + tex * 0.3], -1)[sand]
    # 바위 몇 개 (특징점용)
    for _ in range(300):
        x, y = rng.integers(0, w), rng.integers(0, h)
        if island[y, x]:
            r = rng.integers(3, 12)
            rgb[max(0, y - r):y + r, max(0, x - r):x + r] = rng.uniform(60, 220)
    # 섬 안 그늘진 풀밭: 어둡고 매끈 -> 베이스라인이 물로 오인하는 구멍
    hole = ((xx - cx - 200) / 60) ** 2 + ((yy - cy - 60) / 40) ** 2 < 1
    rgb[hole] = np.array([38, 58, 68]) + rng.normal(0, 1.5, (hole.sum(), 3))
    # 배 항적: 바다 위 가늘고 긴 밝은 줄 (폭 6 px = 3 m, 길이 400 px = 200 m)
    wake = (np.abs((yy - h * 0.2) - 0.3 * (xx - w * 0.15)) < 3) & (xx > w * 0.15) & (xx < w * 0.15 + 400)
    rgb[wake] = 200
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # nodata: 왼쪽 위 삼각형
    alpha = np.where(xx + yy < 900, 0, 255).astype(np.uint8)
    rgb[alpha == 0] = 0
    return rgb, alpha, island


@pytest.fixture(scope="module")
def skysat_like(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rgb, alpha, island = _synthetic_island()
    h, w = alpha.shape
    transform = Affine(0.5, 0, 300000, 0, -0.5, 4120000)
    path = d / "fake_skysat.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=4, dtype="uint8", crs=CRS, transform=transform,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
    ) as dst:
        for i in range(3):
            dst.write(rgb[..., i], i + 1)
        dst.write(alpha, 4)
        from rasterio.enums import ColorInterp

        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
    return path, island, transform


def test_overview_and_alpha(skysat_like):
    path, island, _ = skysat_like
    ov = read_overview(path, max_size=600)
    assert ov.rgb.shape[1] == 600 and ov.rgb.dtype == np.uint8
    assert ov.valid.dtype == bool
    assert 0.9 < ov.valid.mean() < 1.0        # 여백이 일부 제외됨
    assert abs(ov.pixel_size_m - 0.5 * ov.scale) < 1e-6


def test_tiles_cover_region():
    region = Window(100, 50, 5000, 3000)
    cover = np.zeros((3000, 5000), np.int32)
    for read, core in iter_tiles(region, tile=1024, overlap=32):
        assert core.col_off >= read.col_off and core.row_off >= read.row_off
        r0, c0 = int(core.row_off - 50), int(core.col_off - 100)
        cover[r0:r0 + int(core.height), c0:c0 + int(core.width)] += 1
    assert cover.min() == 1 and cover.max() == 1  # 빠짐도 겹침도 없음


def test_find_island_and_tiled_mask(skysat_like, tmp_path):
    path, island, transform = skysat_like
    window, ov = find_land_bbox(path, max_size=600, margin_m=50)
    # bbox 가 섬을 포함해야 함
    ys, xs = np.where(island)
    assert window.col_off <= xs.min() and window.col_off + window.width >= xs.max()
    assert window.row_off <= ys.min() and window.row_off + window.height >= ys.max()

    crop = tmp_path / "crop.tif"
    chunk = export_crop(path, window, crop)
    assert chunk.rgb.shape[0] == int(window.height)

    thr = estimate_thresholds(ov.rgb, ov.valid)
    land, valid, ref = tiled_land_mask(crop, Window(0, 0, int(window.width), int(window.height)), thr, tile=1024, overlap=32, progress=False)
    land = clean_mask(land, 2000)
    land = remove_thin_objects(land, 0.5)
    land = fill_enclosed_water(land, valid)
    # 정답과 IoU 비교 (원본 픽셀 좌표로 잘라서)
    truth = island[int(window.row_off):int(window.row_off + window.height), int(window.col_off):int(window.col_off + window.width)]
    pred = land > 0
    iou = (pred & truth).sum() / (pred | truth).sum()
    assert iou > 0.95, f"IoU {iou:.3f}"

    polys = mask_to_polygons(land, ref.transform, min_area_m2=500, simplify_m=1.0)
    assert len(polys) == 1, "항적이 섬으로 남았거나 섬이 갈라짐"
    assert len(polys[0].interiors) == 0, "섬 안 구멍이 남아 있음"
    assert abs(polys[0].area - truth.sum() * 0.25) / (truth.sum() * 0.25) < 0.1
    lines = polygons_to_coastlines(polys)
    out = save_geojson(lines, ref.crs, tmp_path / "c.geojson", wgs84=True)
    assert out.exists()


def test_synthetic_registration(skysat_like):
    path, island, _ = skysat_like
    ov = read_window(path, Window(1000, 300, 1800, 1400), downscale=1)
    gray = ov.to_gray()
    thr = estimate_thresholds(ov.rgb, ov.valid)
    land = apply_land_mask(ov.rgb, ov.valid, thr)
    res = synthetic_benchmark(gray, ov.valid, land, n_trials=3, seed=0, max_shift_px=25, max_rot_deg=1.5)
    for r in res:
        assert r.n_inliers >= 20
        assert r.after_rmse_px < 1.0, r
        assert r.after_rmse_px < r.before_rmse_px / 10
    hard = synthetic_benchmark(gray, ov.valid, land, n_trials=2, seed=1, max_shift_px=25, max_rot_deg=1.5, hard=True)
    for r in hard:
        assert r.after_rmse_px < 1.5, r


def test_ndwi(tmp_path):
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w]
    water = xx > 150
    green = np.where(water, 800, 900).astype(np.uint16)
    nir = np.where(water, 200, 3000).astype(np.uint16)
    bands = np.stack([green, green, green, nir, green])
    bands[:, :5, :5] = 0  # nodata
    path = tmp_path / "s2.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=5, dtype="uint16", crs=CRS,
                       transform=Affine(10, 0, 300000, 0, -10, 4120000), nodata=0) as dst:
        dst.write(bands)
        dst.descriptions = ("B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir16")
    mask, valid, transform, crs = ndwi_water_mask(path)
    assert (mask[10:, 160:] > 0).all() and (mask[10:, :140] == 0).all()
    assert not valid[:5, :5].any()
