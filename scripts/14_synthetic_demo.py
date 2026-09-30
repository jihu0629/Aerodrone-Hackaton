"""합성 장면으로 전체 파이프라인 검증: 부피 오차(%) 와 W2/W3 방식 비교
    python scripts/14_synthetic_demo.py --out outputs/synthetic
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from litter3d.synthetic import make_scene, default_objects
from litter3d.reconstruct import Surface, write_geotiff
from litter3d.pipeline import run
from litter3d.volume import volumes_from_masks
from litter3d.classes import CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/synthetic")
ap.add_argument("--gsd", type=float, default=0.005)
ap.add_argument("--noise", type=float, default=0.005, help="DSM 노이즈 σ (m)")
a = ap.parse_args()

objs = default_objects()
dsm, ortho, masks = make_scene(objs, gsd_m=a.gsd, noise_m=a.noise, ripple_m=0.01)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
write_geotiff(dsm, out / "dsm.tif")
write_geotiff(ortho[..., ::-1], out / "ortho.tif")
surf = Surface(dsm=dsm, ortho=ortho, gsd_m=a.gsd, transform=None, crs=None)

print(f"{'클래스':18} {'실제(L)':>9} {'추정(L)':>9} {'오차%':>7}")
vols = volumes_from_masks(dsm, masks, a.gsd)
for o, v in zip(objs, vols):
    tv = o.true_volume_m3
    print(f"{o.cls:18} {tv*1000:9.2f} {v.volume_m3*1000:9.2f} {(v.volume_m3/tv-1)*100:7.1f}")

r = run(surf, masks, out)
lo, ty, hi = r["total_kg"]
true_kg = sum(o.true_volume_m3 * CLASSES[o.cls].rho_typ for o in objs if CLASSES[o.cls].is_litter and o.cls != "styrofoam_fragment") \
    + (CLASSES["styrofoam_fragment"].mean_item_g or 0) / 1000
n_litter = sum(1 for o in objs if CLASSES[o.cls].is_litter)
print()
print(f"3D 방식      : {ty:.2f} kg (범위 {lo:.2f}–{hi:.2f})   [기준 {true_kg:.2f} kg = 실제 부피 × ρ_typ]")
print(f"W2 개수×14 g : {n_litter*0.014:.2f} kg")
area = sum(v.area_m2 for v, o in zip(vols, objs) if CLASSES[o.cls].is_litter)
print(f"W3 면적×0.4cm×1.2 : {area*0.004*1200:.2f} kg")
print(f"결과 → {out}/summary.md, overlay.jpg, grid_kg.png, plan.json")
