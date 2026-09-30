"""
4단계: Sentinel-2 NDWI 물 마스크 + SkySat 과의 전역 정합량 확인 (이종 센서 coarse 단계).

Sentinel-2 에는 NIR 이 있으므로 NDWI 로 물을 바로 구분할 수 있습니다.
이 마스크는 (a) 연도별 육지 면적 추세, (b) SkySat RGB 분할의 '약한 정답' 으로 씁니다.

--skysat 을 주면 SkySat 크롭을 10 m 격자로 재투영해 Sentinel-2 와 같은 격자에 놓고,
위상상관 + SIFT 로 두 영상 사이의 전역 이동량(m) 을 추정합니다.
이 값이 크면 두 영상 중 하나의 절대 위치가 밀려 있다는 뜻입니다 (GCR 0.17 참고).
20배 해상도 차이 때문에 원본 해상도끼리 직접 매칭하면 실패합니다. 반드시 같은 GSD 로 맞춘 뒤 매칭합니다.

사용
----
    python scripts/04_s2_ndwi_mask.py --s2 S2_GYD/S2_GYD_2025-08-12.tif
    python scripts/04_s2_ndwi_mask.py --s2 S2_GYD/S2_GYD_2025-08-12.tif --skysat outputs/island_crop.tif

출력 (outputs/)
    s2_<date>_water.tif / .png     NDWI 물 마스크
    s2_<date>_coastline_4326.geojson
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR  # noqa: E402
from coastcd.coastline import mask_to_polygons, polygons_to_coastlines, save_geojson  # noqa: E402
from coastcd.raster_io import open_raster, read_window, write_geotiff  # noqa: E402
from coastcd.register import match_features, estimate_transform, phase_shift  # noqa: E402
from coastcd.water_mask import clean_mask, ndwi_water_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2", required=True, help="s2_download.py 가 만든 5밴드 GeoTIFF")
    ap.add_argument("--skysat", default=None, help="SkySat 크롭 GeoTIFF (전역 정합량 확인용)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--threshold", type=float, default=None, help="NDWI 임계값. 없으면 Otsu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s2 = Path(args.s2)
    stem = s2.stem.replace("S2_GYD_", "s2_")

    print("[1] NDWI 물 마스크")
    water, valid, transform, crs = ndwi_water_mask(s2, args.threshold)
    px_m = abs(transform.a)
    land = ((water == 0) & valid).astype(np.uint8) * 255
    land = clean_mask(land, int(2000 / px_m**2), int(2000 / px_m**2))
    write_geotiff(out / f"{stem}_water.tif", water, transform, crs, nodata=None, overviews=False)
    import cv2

    cv2.imwrite(str(out / f"{stem}_water.png"), water)
    print(f"  물 비율 {100 * (water > 0).mean():.1f}%, 육지 면적 {(land > 0).sum() * px_m**2 / 1e6:.3f} km^2")

    polys = mask_to_polygons(land, transform, min_area_m2=2000, simplify_m=5.0)
    save_geojson(polygons_to_coastlines(polys), crs, out / f"{stem}_coastline_4326.geojson", wgs84=True)

    if not args.skysat:
        print(f"[완료] {out}")
        return

    print("[2] SkySat -> Sentinel-2 격자 재투영 (10 m)")
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window

    with open_raster(args.skysat) as src:
        sk = read_window(args.skysat, Window(0, 0, src.width, src.height), downscale=4)  # 2 m 로 먼저 축소
    sk_gray = sk.to_gray().astype(np.float32)
    sk_on_s2 = np.zeros(water.shape, np.float32)
    reproject(sk_gray, sk_on_s2, src_transform=sk.transform, src_crs=sk.crs,
              dst_transform=transform, dst_crs=crs, resampling=Resampling.average, src_nodata=0, dst_nodata=0)
    sk_valid = sk_on_s2 > 0
    if sk_valid.sum() < 100:
        sys.exit("SkySat 크롭이 Sentinel-2 범위와 겹치지 않습니다. AOI 를 확인하세요.")

    with open_raster(s2) as src:
        red = src.read(3).astype(np.float32)
    # 두 영상을 0~255 로 정규화 (밴드 특성이 달라 절대값 비교는 무의미, 구조만 비교)
    def norm(a, m):
        v = a[m]
        lo, hi = np.percentile(v, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8) * m

    a = norm(red, sk_valid)
    b = norm(sk_on_s2, sk_valid)

    print("[3] 전역 이동량 추정")
    dx, dy, resp = phase_shift(a, b)
    print(f"  위상상관: SkySat 이 S2 대비 dx {dx:+.2f} px, dy {dy:+.2f} px (= {dx * px_m:+.1f} m, {dy * px_m:+.1f} m), 응답 {resp:.2f}")
    pa, pb = match_features(a, b, sk_valid, sk_valid, method="sift", ratio=0.8)
    try:
        reg = estimate_transform(pb, pa, model="similarity", reproj_px=2.0)
        t = reg.matrix[:, 2]
        print(f"  SIFT+MAGSAC: inliers {reg.n_inliers}/{reg.n_matches}, 이동 dx {t[0]:+.2f} px dy {t[1]:+.2f} px "
              f"(= {t[0] * px_m:+.1f} m, {t[1] * px_m:+.1f} m), 잔차 {reg.inlier_rmse_px:.2f} px")
    except RuntimeError as e:
        print(f"  SIFT 매칭 실패: {e}")
    print("  해석: 두 추정치가 서로 비슷하면 신뢰. 5 m 이상이면 Sentinel-2 에 맞춰 SkySat 좌표를 보정할지 결정.")
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
