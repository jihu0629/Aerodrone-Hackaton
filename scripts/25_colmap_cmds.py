"""3단계-b: COLMAP sparse 복원 명령 출력(또는 실행).

  python scripts/25_colmap_cmds.py data/sessions/drone_xxx            # 명령만 출력 (복사해서 실행)
  python scripts/25_colmap_cmds.py data/sessions/drone_xxx --run      # colmap 이 PATH 에 있으면 순서대로 실행
  python scripts/25_colmap_cmds.py data/sessions/drone_xxx --gpu      # GPU 확인 후

결과(sparse/0/ 의 cameras/images/points3D)는 단위가 없는 상대 좌표다. 미터 축척은 별도 근거로 맞춘다 (dronecap/sfm.py 설명).
dense/mesh 는 GPU·메모리 확인 후 결정 (COLMAP patch_match_stereo 는 CUDA 필요; CPU 대안은 OpenMVS 등).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.config import load_config  # noqa: E402
from dronecap.sfm import colmap_sparse_commands  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--config", default="config/dronecap.yml")
    ap.add_argument("--images", help="이미지 폴더 (기본: 세션/sfm/images)")
    ap.add_argument("--work", help="작업 폴더 (기본: 세션/sfm/colmap)")
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--exhaustive", action="store_true")
    ap.add_argument("--colmap", default="colmap"); ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    c = load_config(a.config)["sfm"]
    s = Path(a.session)
    images = Path(a.images) if a.images else s / "sfm" / "images"
    work = Path(a.work) if a.work else s / "sfm" / "colmap"
    cmds = colmap_sparse_commands(images, work, use_gpu=(a.gpu or c["use_gpu"]), sequential=not a.exhaustive, colmap=a.colmap)
    for cmd in cmds:
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    if a.run:
        if not shutil.which(a.colmap):
            print(f"\n{a.colmap} 을 찾을 수 없습니다. https://colmap.github.io 에서 Windows 바이너리를 받아 PATH 에 추가하거나 --colmap 경로를 주세요.")
            return 1
        work.mkdir(parents=True, exist_ok=True); (work / "sparse").mkdir(exist_ok=True)
        for cmd in cmds:
            print("\n>>>", " ".join(cmd)); rc = subprocess.call(cmd)
            if rc != 0:
                print(f"실패 rc={rc}"); return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
