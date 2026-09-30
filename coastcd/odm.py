"""
OpenDroneMap(ODM) 으로 사진 묶음 -> 3D 지도 (점군, 메시, DSM/DTM, 정사영상).

왜 ODM 인가
-----------
SfM(구조 복원) + MVS(밀집 매칭) + DEM/정사영상 생성을 처음부터 짜는 대신, 검증된 오픈소스
파이프라인(ODM = OpenSfM + OpenMVS + PDAL)을 도커 한 줄로 돌립니다. 결과가 전부 지리참조된
GeoTIFF/LAZ 라서 QGIS 와 이 저장소의 rasterio 코드에 바로 올라갑니다.
직접 구현이 필요한 부분(프레임 추출, GPS 주입, 라벨링, 부피 계산)만 파이썬으로 짭니다.

설치 (한 번)
-----------
  Windows: Docker Desktop 설치 -> 실행. WSL2 백엔드 권장. 메모리 8 GB 이상 할당 (Settings > Resources).
  이미지 받기:  docker pull opendronemap/odm          (CPU)
               docker pull opendronemap/odm:gpu      (NVIDIA GPU, 특징점 추출만 가속)
  GUI 가 편하면 WebODM(https://github.com/OpenDroneMap/WebODM) 을 쓰고 결과 폴더만 07 단계에 넘겨도 됩니다.

폴더 규칙 (ODM 고정)
-------------------
  <datasets>/<project>/images/*.jpg     입력 (05 단계 결과)
  <datasets>/<project>/odm_dem/dsm.tif, dtm.tif
  <datasets>/<project>/odm_orthophoto/odm_orthophoto.tif
  <datasets>/<project>/odm_georeferencing/odm_georeferenced_model.laz
  <datasets>/<project>/odm_texturing/odm_textured_model_geo.obj
  <datasets>/<project>/odm_report/report.pdf

좌표계: 기본은 사진 GPS 위치의 UTM 존을 ODM 이 자동 선택합니다. 굴업도(126°E) 는 51/52 경계라
SkySat(EPSG:32652) 과 맞추기 위해 --proj 로 UTM 52N 을 강제합니다 (config.WORKING_CRS).

품질/시간 (사진 300 장, 12 MP, CPU 8코어 기준 대략. 확인 필요)
  fast    : 15~30 분.  feature-quality medium, pc-quality low,  DEM 10 cm.  현장에서 바로 확인용
  default : 1~2 시간. feature-quality high,   pc-quality medium, DEM 5 cm
  high    : 3~6 시간. feature-quality ultra,  pc-quality high,  DEM 2 cm.  부피 측정 최종본
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import WORKING_CRS

DOCKER_IMAGE = "opendronemap/odm"
DOCKER_IMAGE_GPU = "opendronemap/odm:gpu"

PRESETS: dict[str, list[str]] = {
    "fast": ["--feature-quality", "medium", "--pc-quality", "low", "--dem-resolution", "10",
             "--orthophoto-resolution", "5", "--skip-3dmodel"],
    "default": ["--feature-quality", "high", "--pc-quality", "medium", "--dem-resolution", "5",
                "--orthophoto-resolution", "2"],
    "high": ["--feature-quality", "ultra", "--pc-quality", "high", "--dem-resolution", "2",
             "--orthophoto-resolution", "1", "--mesh-size", "300000"],
}

OUTPUTS = {
    "dsm": "odm_dem/dsm.tif",
    "dtm": "odm_dem/dtm.tif",
    "orthophoto": "odm_orthophoto/odm_orthophoto.tif",
    "pointcloud": "odm_georeferencing/odm_georeferenced_model.laz",
    "mesh": "odm_texturing/odm_textured_model_geo.obj",
    "report": "odm_report/report.pdf",
    "cameras": "cameras.json",
}


def crs_to_proj4(crs: str = WORKING_CRS) -> str:
    """EPSG 코드 -> ODM --proj 에 넘길 proj4 문자열."""
    try:
        import warnings

        from pyproj import CRS

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # proj4 문자열 변환 시 정보 손실 경고. UTM 은 문제 없음
            return CRS.from_user_input(crs).to_proj4().replace(" +type=crs", "")
    except Exception:  # noqa: BLE001
        if crs.upper() == "EPSG:32652":
            return "+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs"
        raise


def build_command(
    datasets_dir: str | Path,
    project: str,
    preset: str = "default",
    gpu: bool = False,
    crs: str | None = WORKING_CRS,
    extra: list[str] | None = None,
    rerun: bool = False,
    tty: bool | None = None,
    docker: str = "docker",
) -> list[str]:
    """docker run ... opendronemap/odm ... 명령 목록을 만듭니다 (실행은 run_odm)."""
    if preset not in PRESETS:
        raise ValueError(f"preset 은 {list(PRESETS)} 중 하나: {preset}")
    datasets_dir = Path(datasets_dir).resolve()
    if tty is None:
        tty = sys.stdin.isatty()
    cmd = [docker, "run", "--rm"]
    if tty:
        cmd.append("-ti")
    if gpu:
        cmd += ["--gpus", "all"]
    # 도커에 넘기는 호스트 경로. Windows 한글 경로도 Docker Desktop 은 처리하지만 ASCII 를 권장합니다.
    cmd += ["-v", f"{datasets_dir}:/datasets", DOCKER_IMAGE_GPU if gpu else DOCKER_IMAGE]
    cmd += ["--project-path", "/datasets", project]
    cmd += ["--dsm", "--dtm", "--auto-boundary", "--use-3dmesh"]
    cmd += PRESETS[preset]
    if crs:
        cmd += ["--proj", crs_to_proj4(crs)]
    if rerun:
        cmd += ["--rerun-all"]
    if extra:
        cmd += list(extra)
    return cmd


def stage_images(frames_dir: str | Path, datasets_dir: str | Path, project: str, link: bool = False) -> Path:
    """05 단계 프레임 폴더를 ODM 규칙(<datasets>/<project>/images) 으로 복사(또는 하드링크)합니다."""
    src = Path(frames_dir)
    dst = Path(datasets_dir) / project / "images"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        target = dst / p.name
        if target.exists():
            continue
        if link:
            try:
                os.link(p, target)
            except OSError:
                shutil.copy2(p, target)
        else:
            shutil.copy2(p, target)
        n += 1
    return dst


def docker_available(docker: str = "docker") -> tuple[bool, str]:
    """도커가 설치·실행 중인지. (가능여부, 메시지)"""
    if shutil.which(docker) is None:
        return False, "docker 명령을 찾을 수 없습니다. Docker Desktop 을 설치하고 PATH 를 확인하세요."
    try:
        r = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"docker info 실패: {e}"
    if r.returncode != 0:
        return False, "Docker 데몬이 실행 중이 아닙니다. Docker Desktop 을 켜 주세요.\n" + (r.stderr or "").strip()[:400]
    return True, "ok"


def run_odm(cmd: list[str], log=print) -> int:
    """명령을 실행하고 출력을 그대로 흘려보냅니다. 반환값은 종료 코드."""
    log("[odm] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    return proc.wait()


def find_outputs(datasets_dir: str | Path, project: str) -> dict[str, Path | None]:
    """ODM 결과 파일 위치. 없는 것은 None."""
    root = Path(datasets_dir) / project
    return {k: (root / rel if (root / rel).exists() else None) for k, rel in OUTPUTS.items()}


def collect_outputs(datasets_dir: str | Path, project: str, out_dir: str | Path, log=print) -> dict[str, Path]:
    """결과를 outputs/3d/<project>/ 로 복사하고 경로를 돌려줍니다 (정사영상·DSM·DTM·점군·메시)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = find_outputs(datasets_dir, project)
    copied: dict[str, Path] = {}
    for key, src in found.items():
        if src is None:
            log(f"[odm] 결과 없음: {key} ({OUTPUTS[key]})")
            continue
        if key == "mesh":
            # obj 는 mtl + 텍스처 png 들이 같은 폴더에 있어야 열립니다. 폴더째 복사.
            dst_dir = out_dir / "mesh"
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src.parent, dst_dir)
            copied[key] = dst_dir / src.name
        else:
            dst = out_dir / (key + src.suffix)
            shutil.copy2(src, dst)
            copied[key] = dst
    return copied
