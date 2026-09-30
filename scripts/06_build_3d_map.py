"""
6단계: 프레임 JPEG -> 3D 지도 (OpenDroneMap, 도커).

결과: 점군(LAZ), 텍스처 메시(OBJ), DSM/DTM(GeoTIFF), 정사영상(GeoTIFF), 품질 보고서(PDF).
좌표계는 SkySat 과 같은 EPSG:32652 로 강제합니다 (--crs 로 변경 가능).

사용
----
    python scripts/06_build_3d_map.py --frames C:\\hackathon\\frames\\beach_a --project beach_a
    python scripts/06_build_3d_map.py --frames frames/beach_a --project beach_a --preset fast     # 현장 확인용
    python scripts/06_build_3d_map.py --project beach_a --preset high --rerun                       # 최종본
    python scripts/06_build_3d_map.py --frames frames/beach_a --project beach_a --dry-run           # 명령만 출력
    python scripts/06_build_3d_map.py --project beach_a --collect-only                              # WebODM 등으로 이미 돌린 결과만 모으기

폴더
    --datasets (기본 C:\\hackathon\\odm 또는 outputs/odm) 아래 <project>/images 로 프레임을 복사하고 ODM 을 돌립니다.
    끝나면 주요 결과를 outputs/3d/<project>/ 로 모아 줍니다 (dsm.tif, dtm.tif, orthophoto.tif, pointcloud.laz, mesh/).

QGIS 확인
    orthophoto.tif 와 dsm.tif 를 드래그해 올리고, 라벨 레이어(폴리곤, 속성 class) 를 새로 만들어 쓰레기 더미를 그립니다.
    -> 07_label_measure.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR, WORKING_CRS  # noqa: E402
from coastcd.odm import PRESETS, build_command, collect_outputs, docker_available, find_outputs, run_odm, stage_images  # noqa: E402


def _default_datasets() -> Path:
    if os.name == "nt":
        return Path(r"C:\hackathon\odm")
    return DEFAULT_OUT_DIR / "odm"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=None, help="05 단계 프레임 폴더. 생략하면 이미 <datasets>/<project>/images 에 있다고 가정")
    ap.add_argument("--project", required=True, help="ODM 프로젝트 이름 (영문, 공백 없이)")
    ap.add_argument("--datasets", default=str(_default_datasets()), help="ODM 작업 폴더 (한글 없는 경로 권장)")
    ap.add_argument("--out", default=None, help="결과를 모을 폴더. 기본 outputs/3d/<project>")
    ap.add_argument("--preset", choices=list(PRESETS), default="default")
    ap.add_argument("--crs", default=WORKING_CRS, help="출력 좌표계. 빈 문자열이면 ODM 자동(UTM 존 자동)")
    ap.add_argument("--gpu", action="store_true", help="opendronemap/odm:gpu 사용 (NVIDIA + nvidia-container-toolkit)")
    ap.add_argument("--rerun", action="store_true", help="이전 결과 무시하고 처음부터")
    ap.add_argument("--link", action="store_true", help="프레임을 복사하지 않고 하드링크 (같은 드라이브일 때)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=None, help="ODM 에 그대로 넘길 추가 옵션 (맨 뒤에)")
    ap.add_argument("--dry-run", action="store_true", help="도커 명령만 출력")
    ap.add_argument("--collect-only", action="store_true", help="ODM 을 돌리지 않고 결과만 outputs 로 복사")
    args = ap.parse_args()

    datasets = Path(args.datasets)
    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / "3d" / args.project

    if not args.collect_only:
        if args.frames:
            img_dir = stage_images(args.frames, datasets, args.project, link=args.link)
            n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            print(f"[06] 이미지 {n} 장 -> {img_dir}")
            if n < 20:
                print("[06] 경고: 사진이 20 장 미만입니다. 재구성이 실패하거나 구멍이 많이 납니다.")
        cmd = build_command(datasets, args.project, preset=args.preset, gpu=args.gpu,
                            crs=args.crs or None, extra=args.extra, rerun=args.rerun)
        if args.dry_run:
            print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
            return
        ok, msg = docker_available()
        if not ok:
            sys.exit(f"[06] {msg}\n대안: WebODM 또는 다른 PC 에서 돌린 뒤 --collect-only 로 결과만 모으세요.")
        code = run_odm(cmd)
        if code != 0:
            sys.exit(f"[06] ODM 종료 코드 {code}. 로그 위쪽의 오류를 확인하세요. "
                     "메모리 부족이면 --preset fast, 사진 수 감소, Docker 메모리 할당 증가.")

    found = find_outputs(datasets, args.project)
    copied = collect_outputs(datasets, args.project, out)
    (out / "odm_outputs.json").write_text(json.dumps({k: str(v) for k, v in copied.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[06] 결과:")
    for k, v in copied.items():
        print(f"    {k:11s} {v}")
    missing = [k for k, v in found.items() if v is None and k in ("dsm", "dtm", "orthophoto")]
    if missing:
        print(f"[06] 누락: {missing}. --dsm --dtm 옵션이 빠졌거나 재구성이 실패한 것입니다.")
    print(f"[06] 다음 단계: QGIS 에서 {out / 'orthophoto.tif'} 위에 라벨을 그린 뒤\n"
          f"     python scripts/07_label_measure.py --dsm {out / 'dsm.tif'} --dtm {out / 'dtm.tif'} --labels labels.geojson")


if __name__ == "__main__":
    main()
