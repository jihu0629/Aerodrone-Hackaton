"""DSM + 정사영상 (+ 마스크 PNG 또는 YOLO-seg 가중치) → 무게·격자·수거계획
    python scripts/13_estimate_mass.py --dsm outputs/odm/odm_dem/dsm.tif --ortho outputs/odm/odm_orthophoto/odm_orthophoto.tif --weights runs/litter3d/seg/weights/best.pt
    python scripts/13_estimate_mass.py --dsm dsm.tif --ortho ortho.tif --mask masks.png
    (둘 다 없으면 색 기반 베이스라인)
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d.pipeline import run_from_files

ap = argparse.ArgumentParser()
ap.add_argument("--dsm", required=True)
ap.add_argument("--ortho", default=None)
ap.add_argument("--mask", default=None, help="클래스 인덱스 PNG (0=배경)")
ap.add_argument("--weights", default=None, help="YOLO-seg .pt")
ap.add_argument("--out", default="outputs/mass")
ap.add_argument("--cell", type=float, default=10.0)
ap.add_argument("--wet", action="store_true", help="젖은 상태 → 최대값에 젖음 계수 적용")
a = ap.parse_args()

r = run_from_files(a.dsm, a.out, a.ortho, a.mask, a.weights, cell_m=a.cell, wet=a.wet)
lo, ty, hi = r["total_kg"]
print(f"물체 {r['n_objects']}개, 총 {ty:.1f} kg (범위 {lo:.1f}–{hi:.1f}) → {r['out_dir']}/summary.md")
