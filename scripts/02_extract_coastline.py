"""
2단계: 원본 해상도 육지 마스크 -> 해안선 GeoJSON (타일 단위 처리).

입력은 01 단계의 island_crop.tif (권장) 또는 원본 tif 아무거나 됩니다.
임계값은 축소본에서 한 번 추정하고, 원본 해상도 타일마다 같은 값을 적용합니다.

사용
----
    python scripts/02_extract_coastline.py --tif outputs/island_crop.tif --out outputs
    python scripts/02_extract_coastline.py --tif outputs/island_crop.tif --downscale 2   # 빠른 확인 (1 m/px)

출력 (outputs/)
    land_mask.tif               육지 255 / 그 외 0 (좌표계 유지)
    coastline_overlay.png       해안선 오버레이 (축소본)
    coastline_utm.geojson       폴리곤, 영상 좌표계(SkySat 은 EPSG:32652, 미터) (변화량 계산용)
    coastline_4326.geojson      해안선, 위경도 (Folium/웹지도용)
    thresholds.json             사용한 임계값 (재현용)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR  # noqa: E402
from coastcd.coastline import draw_polygons, mask_to_polygons, polygons_to_coastlines, save_geojson, total_length_m  # noqa: E402
from coastcd.raster_io import open_raster, read_overview, write_geotiff  # noqa: E402
from coastcd.water_mask import clean_mask, estimate_thresholds, fill_enclosed_water, remove_thin_objects, tiled_land_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", default=str(DEFAULT_OUT_DIR / "island_crop.tif"))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--downscale", type=int, default=1, help="1 = 원본 해상도. 2 면 절반 (빠른 확인용)")
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--texture-window", type=int, default=7, help="질감 창 (원본 px 기준)")
    ap.add_argument("--min-area-m2", type=float, default=2000.0, help="이보다 작은 육지 덩어리 제거 (배·바위 제외)")
    ap.add_argument("--keep-lakes", action="store_true", help="섬 안의 물(호수·저수지) 을 유지. 기본은 전부 육지로 채움")
    ap.add_argument("--max-wake-width-m", type=float, default=20.0, help="이보다 가늘고 긴 객체(배 항적) 제거")
    ap.add_argument("--simplify-m", type=float, default=1.0)
    args = ap.parse_args()

    tif = Path(args.tif)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not tif.exists():
        sys.exit(f"파일 없음: {tif}. 먼저 01_find_island.py 를 실행하세요.")

    from rasterio.windows import Window

    with open_raster(tif) as src:
        full = Window(0, 0, src.width, src.height)
        px_m = abs(src.transform.a)

    print("[1] 임계값 추정 (축소본)")
    ov = read_overview(tif, max_size=3000)
    # 축소본에서는 질감 창을 비례해서 줄여야 같은 지상 거리를 봄
    tw_ov = max(3, int(round(args.texture_window / ov.scale)) | 1)
    thr = estimate_thresholds(ov.rgb, ov.valid, texture_window=tw_ov)
    thr.texture_window = args.texture_window  # 원본 해상도 타일에서는 지정 창 사용
    print(f"  밝기 임계값 {thr.brightness:.1f}, 질감(표준편차) 임계값 {thr.texture_std:.2f}")
    with open(out / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thr.as_dict(), f, ensure_ascii=False, indent=2)

    print("[2] 타일 단위 육지 마스크")
    if args.downscale > 1:
        from coastcd.raster_io import read_window
        from coastcd.water_mask import apply_land_mask

        chunk = read_window(tif, full, downscale=args.downscale)
        thr.texture_window = max(3, int(round(args.texture_window / args.downscale)) | 1)
        land = apply_land_mask(chunk.rgb, chunk.valid, thr)
        transform, crs = chunk.transform, chunk.crs
        px_m *= args.downscale
    else:
        land, valid, ref = tiled_land_mask(tif, full, thr, tile=args.tile)
        transform, crs = ref.transform, ref.crs
        chunk = None

    valid_mask = chunk.valid if chunk is not None else valid
    land = clean_mask(land, int(args.min_area_m2 / px_m**2))
    land = remove_thin_objects(land, px_m, max_width_m=args.max_wake_width_m)
    if not args.keep_lakes:
        land = fill_enclosed_water(land, valid_mask)
    land = clean_mask(land, int(args.min_area_m2 / px_m**2))
    write_geotiff(out / "land_mask.tif", land, transform, crs, nodata=0)
    print(f"  육지 비율 {100 * (land > 0).mean():.2f}%  -> land_mask.tif")

    print("[3] 벡터화")
    polys = mask_to_polygons(land, transform, min_area_m2=args.min_area_m2, simplify_m=args.simplify_m)
    lines = polygons_to_coastlines(polys)
    props = [{"area_m2": round(p.area, 1), "perimeter_m": round(p.length, 1)} for p in polys]
    save_geojson(polys, crs, out / "coastline_utm.geojson", props)
    save_geojson(lines, crs, out / "coastline_4326.geojson", props, wgs84=True)
    print(f"  섬 {len(polys)}개, 해안선 총길이 {total_length_m(lines) / 1000:.2f} km, "
          f"최대 섬 면적 {polys[0].area / 1e6:.3f} km^2" if polys else "  육지 없음")

    print("[4] 오버레이")
    import cv2

    ov2 = read_overview(tif, max_size=2500)
    vis = draw_polygons(ov2.rgb, polys, ov2.transform, thickness=2)
    cv2.imwrite(str(out / "coastline_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
