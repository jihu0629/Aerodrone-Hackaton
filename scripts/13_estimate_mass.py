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
ap.add_argument("--frames", default=None, help="원본 프레임 폴더 (리포트에 검출 갤러리 추가, 최대 6장)")
ap.add_argument("--photos", default=None, help="원본 사진 폴더 → report_photos.html 도 함께 출력 (widths.json 또는 EXIF 고도로 GSD)")
ap.add_argument("--photo-width", type=float, default=None, help="사진 폭(m) 직접 지정")
ap.add_argument("--photo-alt", type=float, default=None, help="촬영 고도(m) 직접 지정 (EXIF 없을 때)")
a = ap.parse_args()

r = run_from_files(a.dsm, a.out, a.ortho, a.mask, a.weights, frames_dir=a.frames, photos_dir=a.photos,
                   photo_width_m=a.photo_width, photo_altitude_m=a.photo_alt, cell_m=a.cell, wet=a.wet)
lo, ty, hi = r["total_kg"]
print(f"물체 {r['n_objects']}개, 총 {ty:.1f} kg (범위 {lo:.1f}–{hi:.1f}) → {r['report_html']}")
if "photo_report_html" in r:
    photos_txt = ", ".join(f"{p['name']} {p['n']}개" for p in r["photos"])
    print(f"사진 리포트 → {r['photo_report_html']}  ({photos_txt})")
