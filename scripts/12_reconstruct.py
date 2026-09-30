"""프레임 → ODM 으로 DSM + 정사영상 (docker 필요)
    python scripts/12_reconstruct.py outputs/frames --project outputs/odm --gsd 0.5
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d.reconstruct import run_odm, odm_command, colmap_commands

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("--project", default="outputs/odm")
ap.add_argument("--gsd", type=float, default=0.5, help="DSM/정사영상 해상도 cm/px")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--dry", action="store_true", help="명령만 출력")
a = ap.parse_args()

if a.dry:
    print("ODM:", " ".join(odm_command(a.project, a.gsd, use_gpu=a.gpu)))
    for c in colmap_commands(a.images, Path(a.project) / "colmap"):
        print("COLMAP:", " ".join(c))
else:
    run_odm(a.project, a.images, gsd_cm=a.gsd, use_gpu=a.gpu)
    print("완료:", Path(a.project) / "odm_dem/dsm.tif", Path(a.project) / "odm_orthophoto/odm_orthophoto.tif")
