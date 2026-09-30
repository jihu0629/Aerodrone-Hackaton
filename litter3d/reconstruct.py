"""3D 복원 (SfM-MVS) 실행 래퍼와 DSM·정사영상 읽기.

- OpenDroneMap(ODM, docker) : 사진 폴더 → DSM(GeoTIFF) + 정사영상(GeoTIFF). 메인 경로.
- COLMAP : 점군까지. DSM 래스터화는 별도 (여기서는 명령만 만들어 준다).
- VGGT/MASt3R : 빠른 경로 후보. 스케일(m) 이 없으므로 SRT 고도·기준물로 보정 필요 (미구현, 확인 필요).

ODM 출력 위치 (project/) :
  odm_dem/dsm.tif, odm_orthophoto/odm_orthophoto.tif
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def odm_command(project_dir: str | Path, gsd_cm: float = 0.5, docker_image: str = "opendronemap/odm",
                extra: list[str] | None = None, use_gpu: bool = False) -> list[str]:
    """project_dir/images/ 에 사진(+ geo.txt) 이 있어야 한다."""
    project_dir = Path(project_dir).resolve()
    img = "opendronemap/odm:gpu" if use_gpu else docker_image
    cmd = ["docker", "run", "--rm", "-v", f"{project_dir}:/datasets/code"]
    if use_gpu:
        cmd += ["--gpus", "all"]
    cmd += [img, "--project-path", "/datasets",
            "--dsm", "--dem-resolution", str(gsd_cm), "--orthophoto-resolution", str(gsd_cm),
            "--pc-quality", "high", "--feature-quality", "high", "--min-num-features", "12000",
            "--dem-gapfill-steps", "4", "--skip-3dmodel", "--fast-orthophoto"]
    if extra:
        cmd += extra
    return cmd


def run_odm(project_dir: str | Path, images_dir: str | Path | None = None, **kw) -> Path:
    """사진을 project_dir/images 로 복사(필요시) 후 ODM 실행. docker 가 없으면 명령만 출력하고 예외."""
    project_dir = Path(project_dir)
    if images_dir is not None:
        dst = project_dir / "images"
        dst.mkdir(parents=True, exist_ok=True)
        for p in Path(images_dir).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".txt"}:
                shutil.copy2(p, dst / p.name)
    cmd = odm_command(project_dir, **kw)
    if shutil.which("docker") is None:
        raise RuntimeError("docker 가 없습니다. 아래 명령을 ODM 이 있는 PC 에서 실행:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return project_dir


def colmap_commands(images_dir: str | Path, work_dir: str | Path) -> list[list[str]]:
    """COLMAP 자동 복원 명령 (dense 까지). 실행은 하지 않는다."""
    images_dir, work_dir = Path(images_dir).resolve(), Path(work_dir).resolve()
    return [["colmap", "automatic_reconstructor", "--workspace_path", str(work_dir),
             "--image_path", str(images_dir), "--dense", "1", "--quality", "high"]]


@dataclass
class Surface:
    dsm: np.ndarray            # (H, W) float32, m. nodata → nan
    ortho: np.ndarray | None   # (H, W, 3) uint8 BGR, DSM 격자에 맞춤
    gsd_m: float
    transform: object | None   # rasterio Affine
    crs: object | None

    @property
    def shape(self):
        return self.dsm.shape


def load_surface(dsm_path: str | Path, ortho_path: str | Path | None = None) -> Surface:
    """DSM 과 정사영상을 읽고 정사영상을 DSM 격자로 리샘플한다."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    with rasterio.open(dsm_path) as d:
        dsm = d.read(1).astype(np.float32)
        if d.nodata is not None:
            dsm[dsm == d.nodata] = np.nan
        tr, crs = d.transform, d.crs
        gsd = float(abs(tr.a))
        ortho = None
        if ortho_path is not None:
            with rasterio.open(ortho_path) as o:
                bands = min(3, o.count)
                src = o.read(list(range(1, bands + 1)))
                if crs is not None and o.crs is not None:
                    dst = np.zeros((bands, dsm.shape[0], dsm.shape[1]), src.dtype)
                    for b in range(bands):
                        reproject(src[b], dst[b], src_transform=o.transform, src_crs=o.crs,
                                  dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
                    rgb = np.moveaxis(dst, 0, -1)
                else:
                    # 좌표계가 없으면 같은 범위라고 보고 크기만 맞춘다
                    import cv2
                    rgb = np.moveaxis(src, 0, -1)
                    if rgb.shape[:2] != dsm.shape:
                        rgb = cv2.resize(rgb, (dsm.shape[1], dsm.shape[0]), interpolation=cv2.INTER_LINEAR)
                        if rgb.ndim == 2:
                            rgb = rgb[..., None]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb / max(float(rgb.max()), 1) * 255, 0, 255).astype(np.uint8)
                if rgb.shape[2] == 1:
                    rgb = np.repeat(rgb, 3, axis=2)
                ortho = rgb[..., ::-1].copy()   # RGB → BGR (OpenCV)
    return Surface(dsm=dsm, ortho=ortho, gsd_m=gsd, transform=tr, crs=crs)


def write_geotiff(arr: np.ndarray, path: str | Path, transform=None, crs=None, nodata=None) -> None:
    import rasterio
    from rasterio.transform import from_origin
    if transform is None:
        transform = from_origin(0, arr.shape[0], 1, 1)
    count = 1 if arr.ndim == 2 else arr.shape[2]
    data = arr[None] if arr.ndim == 2 else np.moveaxis(arr, -1, 0)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=count,
                       dtype=str(data.dtype), crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data)


def scale_check(measured_m: float, true_m: float) -> dict:
    """기준물(크기를 아는 판) 로 스케일 오차를 확인. 부피는 길이의 세제곱이므로 오차가 증폭된다."""
    s = true_m / measured_m
    return {"scale_factor": s, "length_error_pct": (measured_m / true_m - 1) * 100,
            "volume_error_pct_if_uncorrected": ((measured_m / true_m) ** 3 - 1) * 100}
