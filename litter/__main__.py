"""
해안쓰레기 위치·무게 → 수거계획 파이프라인.

  python -m litter inspect <폴더|파일>          # 업체 데이터 구조 확인 (받자마자)
  python -m litter synth --out data/litter_synth  # 합성 데이터 생성
  python -m litter run --images DIR --labels FILE [--weights CSV] [--telemetry CSV]
                       [--dsm TIF] [--ortho TIF] [--epsg N] [--out DIR]
  python -m litter video --video X.MP4 --srt X.SRT --out DIR  # 동영상 → 프레임+텔레메트리

run 단계:
  1 라벨 읽기 → 2 텔레메트리 → 3 위치(픽셀 광선) → 4 중복 제거
  → 5 형상(면적·길이·DSM 부피) → 6 무게(물리+학습+구간) → 7 수거계획 → 8 결과물
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import geometry, ingest, plan, report, telemetry, weight
from .config import load_config


def cmd_run(a):
    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1] 라벨 읽기")
    anns = ingest.load(a.labels, cfg, images_dir=a.images, weights=a.weights, min_score=a.min_score)
    n_w = sum(1 for x in anns if x.get("weight_kg"))
    n_poly = sum(1 for x in anns if not x["from_bbox"])
    print(f"  라벨 {len(anns)} · 폴리곤 {n_poly} · 무게 있음 {n_w}")
    if n_poly < len(anns):
        print(f"  ⚠️ bbox만 있는 라벨 {len(anns) - n_poly}개 — 면적이 과대추정됨 (SAM으로 마스크화 권장)")

    print("[2] 텔레메트리")
    used = sorted({x["image"] for x in anns})
    tel = telemetry.collect(a.images, a.telemetry, used)
    print(f"  위치 정보 있는 이미지 {sum(1 for n in used if 'lat' in tel.get(n, {}))}/{len(used)}")

    dsm = geometry.DSM(a.dsm) if a.dsm else None
    ortho_T = None
    if a.ortho:
        import rasterio
        with rasterio.open(a.ortho) as ds:
            ortho_T, epsg = ds.transform, ds.crs.to_epsg()
        geo = geometry.Geo(epsg)
    elif any(x.get("geo_transform") for x in anns):  # 정사영상 칩 COCO
        from pyproj import CRS
        geo = geometry.Geo(CRS.from_user_input(next(x["crs"] for x in anns if x.get("crs"))).to_epsg())
    else:
        lon0 = next((t["lon"] for t in tel.values() if "lon" in t), None)
        epsg = a.epsg or (dsm.epsg if dsm else None)  # DSM이 있으면 DSM 좌표계를 따른다
        if epsg is None and lon0 is None:
            sys.exit("위치 정보가 없음: --telemetry, EXIF GPS, --ortho 중 하나가 필요")
        geo = geometry.Geo(epsg, lon=lon0)
    for x in anns:  # 수거 GPS 정답은 지도좌표로도
        if x.get("gt_lat") is not None:
            x["gt_x"], x["gt_y"] = geo.to_xy(x["gt_lat"], x["gt_lon"])

    print(f"[3] 위치 계산 (EPSG:{geo.epsg}, DSM={'있음' if dsm else '없음'})")
    anns = geometry.georef(anns, tel, geo, cfg, dsm=dsm, ortho_T=ortho_T, takeoff_z=a.takeoff_z)

    print("[4] 중복 제거")
    items = geometry.dedup(anns, cfg)
    print(f"  라벨 {len(anns)} → 물체 {len(items)}")
    if dsm and a.snap:
        n = geometry.snap_to_dsm(items, cfg, dsm)
        moved = [it["snap_m"] for it in items if "snap_m" in it]
        print(f"  DSM 스냅: {n}/{len(items)}개 이동 (중앙값 {np.median(moved) if moved else 0:.2f} m)")
    for it in items:  # 합친·스냅한 위치로 위경도 다시
        it["lat"], it["lon"] = geo.to_latlon(it["x"], it["y"])

    print("[5] 형상 측정")
    geometry.measure(items, cfg, dsm=dsm)
    if dsm:
        print(f"  3D 부피 신뢰 가능: {sum(1 for i in items if i.get('valid_3d'))}/{len(items)} · "
              f"묻힘 의심: {sum(1 for i in items if i.get('buried_suspect'))}")

    print("[6] 무게 추정")
    results = {}
    if a.eval and sum(1 for i in items if i.get("weight_kg")) >= 30:
        rows = weight.evaluate(items, cfg)
        results["weight_ablation"] = rows
        print("  방법                        물체오차중앙  총량오차  ≥23kg 탐지  오경보")
        for r in rows:
            hr = f"{r['heavy_recall_pct']:.0f}%" if r["heavy_recall_pct"] is not None else "-"
            fp = f"{r['false_2p_pct']:.1f}%" if r["false_2p_pct"] is not None else "-"
            print(f"  {r['method']:<26} {r['median_ape_pct']:6.0f}%  {r['total_err_pct']:+7.0f}%  {hr:>8}  {fp:>6}")
        report.fig_ablation(out, rows)
    items, _ = weight.estimate(items, cfg)

    print("[7] 수거 계획")
    stops, summary = plan.make_plan(items, cfg)
    print(f"  정거장 {summary['n_stops']} · {summary['kg_est']:.0f} kg (최대 {summary['kg_hi']:.0f}) · "
          f"인력 {summary['crew_counts']} · 마대 {summary['bags']} · 트럭 {summary['trucks']} · 경로 {summary['route_m']:.0f} m")

    print("[8] 결과물")
    loc = report.location_errors(items, geo)
    if loc:
        results["location"] = {k: v for k, v in loc.items() if k != "_rows"}
        print("  위치 오차 (수거 지점 기준)  중앙값 / 90%")
        for k, n in [("drone_gps", "드론 GPS"), ("frame_center", "화면 중앙"), ("ours", "픽셀 광선(제안)")]:
            if k in loc:
                print(f"    {n:<14} {loc[k]['median_m']:6.1f} m / {loc[k]['p90_m']:6.1f} m")
        if "r95_coverage_pct" in loc:
            print(f"    반경(r95) 안에 실제로 들어온 비율: {loc['r95_coverage_pct']:.0f}%")
        report.fig_location(out, loc)
    results["summary"] = {k: v for k, v in summary.items() if k != "grid"}
    report.write_tables(out, items, stops, summary)
    report.fig_map(out, items, stops, summary)
    report.work_cards(out, items, stops, summary, a.images)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    print(f"  → {out}  (work_cards.html, fig_*.png, items.csv, stops.csv, *.geojson, results.json)")
    return results


def cmd_video(a):
    tel = telemetry.extract_video_frames(a.video, a.srt, a.out, every_s=a.every, time_offset_s=a.offset)
    rows = [{"image": k, **v} for k, v in tel.items()]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k))
    with open(Path(a.out) / "telemetry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"프레임 {len(rows)}개 → {a.out} (telemetry.csv 포함)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="데이터 구조 확인")
    p.add_argument("path")

    p = sub.add_parser("synth", help="합성 데이터 생성")
    p.add_argument("--out", default="data/litter_synth")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("run", help="파이프라인 실행")
    p.add_argument("--images", help="원본 프레임 폴더 (EXIF·작업카드 사진용)")
    p.add_argument("--labels", required=True, help="COCO json / YOLO 폴더 / CSV")
    p.add_argument("--weights", help="물체별 무게 CSV (id 열로 매칭)")
    p.add_argument("--telemetry", help="텔레메트리 CSV/JSON (EXIF보다 우선)")
    p.add_argument("--dsm", help="DSM GeoTIFF (ODM 등) — 있으면 3D 부피")
    p.add_argument("--ortho", help="라벨이 정사영상 위에 달린 경우 그 GeoTIFF")
    p.add_argument("--epsg", type=int, help="지도 좌표계 (기본: UTM 자동)")
    p.add_argument("--takeoff_z", type=float, default=0.0, help="이륙 지점 높이 (DSM 기준, m)")
    p.add_argument("--min_score", type=float, default=0.25, help="모델 예측(score 있는 라벨)의 최소 신뢰도")
    p.add_argument("--snap", action="store_true",
                   help="DSM 스냅 (실험적: 물체 간격이 위치오차보다 촘촘하면 오히려 나빠짐)")
    p.add_argument("--config", help="설정 덮어쓰기 JSON")
    p.add_argument("--eval", action="store_true", help="무게 정답으로 방법별 비교(어블레이션)")
    p.add_argument("--out", default="results/litter")

    p = sub.add_parser("video", help="동영상 → 프레임 + SRT 텔레메트리")
    p.add_argument("--video", required=True)
    p.add_argument("--srt")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=float, default=1.0, help="몇 초마다 한 프레임")
    p.add_argument("--offset", type=float, default=0.0, help="영상-로그 시간 보정 (초)")

    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        ingest.inspect(a.path)
    elif a.cmd == "synth":
        from .synth import generate
        generate(a.out, n_items=a.n, seed=a.seed)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "video":
        cmd_video(a)


if __name__ == "__main__":
    main()
