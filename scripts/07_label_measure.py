"""
7단계: 라벨 폴리곤 + DSM/DTM -> 면적·부피·무게 (쓰레기 양 추정).

라벨 만드는 법 (둘 중 하나, 섞어도 됨)
  A. 수동: QGIS 에서 outputs/3d/<project>/orthophoto.tif 를 열고, 레이어 > 레이어 생성 > 새 GeoPackage/GeoJSON
     레이어(폴리곤, 좌표계는 정사영상과 같게, 텍스트 필드 "class") 를 만들어 더미마다 폴리곤을 그리고
     class 에 mixed / plastic / styrofoam / net / wood / ... 를 적습니다.
  B. 반자동: --auto-candidates 로 DSM-DTM 이 --min-height 이상인 덩어리를 후보 폴리곤으로 뽑아
     candidates.geojson 을 만들고, QGIS 에서 잘못 잡힌 것(바위·식생) 을 지우고 class 만 채웁니다.
  C. 탐지 모델(YOLO-seg 등) 결과를 같은 형식(폴리곤 + class) GeoJSON 으로 저장해서 넣습니다.

사용
----
    python scripts/07_label_measure.py --dsm outputs/3d/beach_a/dsm.tif --dtm outputs/3d/beach_a/dtm.tif --labels labels.geojson
    python scripts/07_label_measure.py --dsm dsm.tif --labels labels.geojson --base plane          # DTM 없이
    python scripts/07_label_measure.py --dsm dsm.tif --dtm dtm.tif --auto-candidates --min-height 0.2 --aoi beach.geojson
    python scripts/07_label_measure.py ... --density-json my_density.json                            # 현장 보정 밀도

출력 (--out/)
    measurements.csv            라벨별 면적 m², 부피 m³, 평균/최대 높이, 밀도, 무게 kg
    measurements.geojson        같은 내용을 폴리곤 속성으로 (QGIS 에서 색칠·라벨 표시)
    measurements_summary.json   class 별 합계, 전체 합계
    measurements_overlay.png    정사영상 위에 폴리곤 + 무게 표시 (--ortho 를 주면)
    candidates.geojson          (--auto-candidates 일 때) 라벨 후보

정확도 메모
    부피 오차의 대부분은 (1) 바닥면 추정, (2) DSM 해상도·잡음, (3) 폴리곤 경계 입니다.
    같은 더미를 두 번 다른 비행에서 재서 차이를 보면 반복 정밀도를 알 수 있습니다.
    무게는 부피 x 겉보기 밀도이고 밀도 편차가 크므로, 현장에서 1~2 더미를 실제로 달아 --density-json 을 보정하세요.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.coastline import save_geojson  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR  # noqa: E402
from coastcd.volume import DENSITY_KG_M3, height_candidates, load_labels, measure_labels, save_measurements  # noqa: E402


def _overlay(ortho_path: Path, geoms, meas, crs, out_png: Path, max_size: int = 2500) -> None:
    import cv2
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    from coastcd.config import safe_path

    with rasterio.open(safe_path(ortho_path)) as src:
        scale = max(1.0, max(src.width, src.height) / max_size)
        out_shape = (int(src.height / scale), int(src.width / scale))
        rgb = src.read([1, 2, 3], out_shape=(3,) + out_shape, resampling=Resampling.average)
        rgb = np.moveaxis(rgb, 0, -1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb / max(1, np.percentile(rgb, 99)) * 255, 0, 255).astype(np.uint8)
        tr = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
    inv = ~tr
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    for g, m in zip(geoms, meas):
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            pts = np.array([inv * (x, y) for x, y in p.exterior.coords], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cx, cy = inv * (m.centroid_x, m.centroid_y)
        txt = f"#{m.id} {m.label} {m.volume_m3:.2f}m3" + (f" {m.weight_kg:.0f}kg" if m.weight_kg is not None else "")
        cv2.putText(img, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsm", required=True)
    ap.add_argument("--dtm", default=None)
    ap.add_argument("--labels", default=None, help="라벨 GeoJSON (폴리곤, 속성 class)")
    ap.add_argument("--ortho", default=None, help="오버레이 PNG 용 정사영상")
    ap.add_argument("--out", default=None, help="기본: DSM 이 있는 폴더")
    ap.add_argument("--base", choices=["dtm", "plane", "min"], default="dtm", help="바닥면 방식. DTM 이 없으면 자동으로 plane")
    ap.add_argument("--ring-m", type=float, default=1.0, help="plane 방식에서 바닥 평면을 맞출 테두리 폭 (m)")
    ap.add_argument("--class-field", default="class")
    ap.add_argument("--default-class", default="mixed")
    ap.add_argument("--density-json", default=None, help='{"plastic": 55, ...} kg/m³ 로 기본 밀도 덮어쓰기')
    ap.add_argument("--auto-candidates", action="store_true", help="DSM-DTM 높이로 라벨 후보 폴리곤 생성 (DTM 필요)")
    ap.add_argument("--min-height", type=float, default=0.15)
    ap.add_argument("--min-area", type=float, default=0.5)
    ap.add_argument("--max-area", type=float, default=500.0)
    ap.add_argument("--aoi", default=None, help="후보 탐색 범위 폴리곤 GeoJSON (해빈 등)")
    args = ap.parse_args()

    dsm = Path(args.dsm)
    if not dsm.exists():
        sys.exit(f"DSM 없음: {dsm}. 06 단계를 먼저 실행하세요.")
    out = Path(args.out) if args.out else dsm.parent
    out.mkdir(parents=True, exist_ok=True)

    density = dict(DENSITY_KG_M3)
    if args.density_json:
        density.update({k.lower(): float(v) for k, v in json.loads(Path(args.density_json).read_text(encoding="utf-8")).items()})

    if args.auto_candidates:
        if not args.dtm:
            sys.exit("--auto-candidates 는 --dtm 이 필요합니다.")
        aoi = None
        if args.aoi:
            import rasterio
            from shapely.ops import unary_union

            from coastcd.config import safe_path

            with rasterio.open(safe_path(dsm)) as s:
                g, _, _ = load_labels(args.aoi, target_crs=s.crs)
            aoi = unary_union(g)
        polys, props, crs = height_candidates(dsm, args.dtm, args.min_height, args.min_area, args.max_area, aoi=aoi,
                                              simplify_m=None)
        cand = save_geojson(polys, crs, out / "candidates.geojson", properties=props)
        print(f"[07] 라벨 후보 {len(polys)} 개 -> {cand}. QGIS 에서 정리 후 --labels 로 넘기세요.")
        if not args.labels:
            args.labels = str(cand)
            print("[07] --labels 가 없어 후보를 그대로 측정합니다 (전부 class=mixed).")

    if not args.labels:
        sys.exit("--labels 또는 --auto-candidates 중 하나가 필요합니다.")

    meas, geoms, crs = measure_labels(dsm, args.labels, dtm_path=args.dtm, base=args.base, ring_m=args.ring_m,
                                      class_field=args.class_field, default_label=args.default_class, density=density)
    paths = save_measurements(meas, geoms, crs, out)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    print(f"[07] 라벨 {len(meas)} 개 측정 (좌표계 {crs})")
    print(f"{'id':>4} {'class':10s} {'면적m²':>9} {'부피m³':>9} {'최대높이m':>9} {'무게kg':>9}  바닥")
    for m in meas:
        wk = f"{m.weight_kg:9.1f}" if m.weight_kg is not None else "        -"
        print(f"{m.id:4d} {m.label:10s} {m.area_m2:9.2f} {m.volume_m3:9.3f} {m.height_max_m:9.2f} {wk}  {m.base_method}")
    print(f"합계  면적 {summary['total_area_m2']:.1f} m²  부피 {summary['total_volume_m3']:.2f} m³  무게 {summary['total_weight_kg']:.0f} kg")
    for k, v in summary["by_class"].items():
        flag = "" if v["weight_known"] else " (일부 밀도 없음)"
        print(f"  {k:10s} {v['count']:3d} 개  {v['volume_m3']:8.2f} m³  {v['weight_kg']:8.0f} kg{flag}")

    if args.ortho and Path(args.ortho).exists():
        png = out / "measurements_overlay.png"
        _overlay(Path(args.ortho), geoms, meas, crs, png)
        print(f"[07] 오버레이 -> {png}")
    print(f"[07] 파일: {paths['csv']}, {paths['geojson']}, {paths['summary']}")


if __name__ == "__main__":
    main()
