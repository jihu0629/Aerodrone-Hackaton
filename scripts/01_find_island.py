"""
1단계: SkySat 원본에서 섬(굴업도) 영역을 자동으로 찾아 원본 해상도로 잘라냅니다.

왜 하나
-------
SkySat 스트립(34447 x 23973 px, 801 MB) 은 95% 가 바다입니다. 섬 주변만 잘라 두면
(약 8000 x 6000 px, RGBA 로 100~200 MB) 이후 모든 단계가 수십 배 빨라지고,
팀원 노트북에서도 돌아갑니다.

사용
----
    python scripts/01_find_island.py --tif "C:\\...\\20260817_231616_ssc1_u0002_visual.tif" --out outputs
    python scripts/01_find_island.py --all-islands      # 작은 섬까지 포함한 bbox

출력 (outputs/)
    overview.png          축소본 + 찾은 bbox (빨간 사각형)
    island_crop.tif       원본 해상도 RGBA GeoTIFF (좌표계 유지, 오버뷰 포함)
    island_crop_preview.png
    island_window.json    원본 픽셀 기준 bbox (다른 스크립트가 재사용)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402  (환경변수 설정)
from coastcd.config import DEFAULT_OUT_DIR, DEFAULT_SKYSAT_TIF  # noqa: E402
from coastcd.raster_io import describe, export_crop, find_land_bbox  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", default=str(DEFAULT_SKYSAT_TIF), help="SkySat ortho_visual GeoTIFF")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="결과 폴더")
    ap.add_argument("--overview-size", type=int, default=2000, help="탐색용 축소본 긴 변 픽셀")
    ap.add_argument("--margin-m", type=float, default=300.0, help="섬 주변 여유 (m)")
    ap.add_argument("--all-islands", action="store_true", help="가장 큰 섬 하나가 아니라 모든 섬 포함")
    ap.add_argument("--preview-size", type=int, default=2000)
    args = ap.parse_args()

    tif = Path(args.tif)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not tif.exists():
        sys.exit(f"파일 없음: {tif}\n--tif 로 경로를 지정하거나 환경변수 COASTCD_SKYSAT 를 설정하세요.")

    print("[1] 메타데이터")
    describe(tif)

    print("[2] 섬 영역 탐색 (오버뷰)")
    window, ov = find_land_bbox(tif, max_size=args.overview_size, margin_m=args.margin_m, all_islands=args.all_islands)
    print(f"  bbox (원본 px): col {int(window.col_off)}, row {int(window.row_off)}, "
          f"{int(window.width)} x {int(window.height)} px  "
          f"(= {window.width * abs(ov.transform.a) / ov.scale / 1000:.2f} x "
          f"{window.height * abs(ov.transform.e) / ov.scale / 1000:.2f} km)")

    import cv2

    s = ov.scale
    vis = cv2.cvtColor(ov.rgb, cv2.COLOR_RGB2BGR)
    x0, y0 = int(window.col_off / s), int(window.row_off / s)
    x1, y1 = int((window.col_off + window.width) / s), int((window.row_off + window.height) / s)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.imwrite(str(out / "overview.png"), vis)

    print("[3] 원본 해상도 크롭 저장 (타일 GeoTIFF)")
    crop_path = out / "island_crop.tif"
    chunk = export_crop(tif, window, crop_path)
    print(f"  저장: {crop_path}  ({chunk.rgb.shape[1]} x {chunk.rgb.shape[0]} px, {chunk.pixel_size_m} m/px)")

    scale = max(chunk.rgb.shape[:2]) / args.preview_size
    if scale > 1:
        prev = cv2.resize(chunk.rgb, (int(chunk.rgb.shape[1] / scale), int(chunk.rgb.shape[0] / scale)), interpolation=cv2.INTER_AREA)
    else:
        prev = chunk.rgb
    cv2.imwrite(str(out / "island_crop_preview.png"), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

    with open(out / "island_window.json", "w", encoding="utf-8") as f:
        json.dump(
            {"source": str(tif), "col_off": int(window.col_off), "row_off": int(window.row_off),
             "width": int(window.width), "height": int(window.height)},
            f, ensure_ascii=False, indent=2,
        )
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
