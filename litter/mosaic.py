"""
선회 영상 → 정사영상(모자이크) + 탐지 물체 표시.

3D 복원(SfM)이 프레임마다 카메라 위치·방향을 알아냈으므로, 지면 평면 위 격자의 각 점이
각 프레임 어디에 찍혔는지 계산할 수 있다. 격자 점마다 '그 점이 화면 중앙에 가장 가깝게 찍힌
프레임'의 색을 가져와 위에서 본 지도 한 장을 만든다 (가장 덜 비스듬하고 덜 왜곡된 픽셀).
격자는 UTM(동·북, m)으로 잡아 북쪽이 위, 그대로 GeoTIFF로도 저장한다.

  python -m litter.mosaic --orbit runs/orbit/0007 --srt DJI_0007.SRT --objects runs/map/0007_sfm/objects.csv \
         --out runs/map/0007_sfm --res 0.01
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Geo
from .orbit3d import _center, _viewdir
from .orbit_map import fit_rigid2d
from .seg import _imread, _imwrite
from .telemetry import read_srt
from .volume import fit_ground, to_metric


def build(orbit_dir, srt, out, objects_csv=None, res=0.01, margin=5.0):
    import pycolmap

    orbit_dir, out = Path(orbit_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(orbit_dir / "sparse" / "0"))
    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    fidx = json.loads((orbit_dir / "frames" / "frames.json").read_text(encoding="utf-8"))
    tel = {r["frame"]: r for r in read_srt(srt) if "lat" in r}
    P = np.array([p.xyz for p in rec.points3D.values()])
    C = np.array([_center(i) for i in imgs])
    D = np.array([_viewdir(i) for i in imgs])
    alt = np.array([tel[fidx[i.name]]["alt"] for i in imgs])
    T, R, s, _ = to_metric(P, C, D, alt)
    p0, _ = fit_ground(P)
    Cm = T(C)
    geo = Geo(None, lon=tel[fidx[imgs[0].name]]["lon"])
    gps = np.array([geo.to_xy(tel[fidx[i.name]]["lat"], tel[fidx[i.name]]["lon"]) for i in imgs])
    R2, t2, rms = fit_rigid2d(Cm[:, :2], gps)

    # UTM 격자 (북쪽이 위): 카메라 위치 범위 + margin
    cam_utm = Cm[:, :2] @ R2.T + t2
    e0, n0 = cam_utm.min(0) - margin
    e1, n1 = cam_utm.max(0) + margin
    W, H = int((e1 - e0) / res), int((n1 - n0) / res)
    ee, nn = np.meshgrid(e0 + (np.arange(W) + .5) * res, n1 - (np.arange(H) + .5) * res)
    utm = np.stack([ee.ravel(), nn.ravel()], 1)
    local = (utm - t2) @ R2                     # UTM → 3D 지면 좌표 (m)
    Xs = (np.c_[local, np.zeros(len(local))] / s) @ R + p0   # → SfM 좌표
    cam = rec.cameras[imgs[0].camera_id]
    cw, ch = cam.width, cam.height
    best = np.full(len(utm), np.inf, np.float32)
    mosaic = np.zeros((len(utm), 3), np.uint8)
    for k, im in enumerate(imgs):
        Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Xc = Xs @ np.asarray(Tc.rotation.matrix()).T + np.asarray(Tc.translation)
        front = Xc[:, 2] > 0
        uv = np.full((len(Xc), 2), -1.0)
        uv[front] = np.asarray(cam.img_from_cam(Xc[front], False))
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < cw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < ch - 1)
        d = np.hypot((uv[:, 0] - cw / 2) / cw, (uv[:, 1] - ch / 2) / ch)   # 화면 중앙에서의 거리
        take = inside & (d < best)
        if not take.any():
            continue
        img = _imread(orbit_dir / "frames" / im.name)
        # 쌍선형 보간으로 색 가져오기 (cv2.remap은 32767칸 제한이 있어 직접 계산)
        x, y = uv[take, 0], uv[take, 1]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
             + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        mosaic[take] = np.clip(c, 0, 255).astype(np.uint8)
        best[take] = d[take]
    mosaic = mosaic.reshape(H, W, 3)
    covered = np.isfinite(best).reshape(H, W)
    _imwrite(out / "mosaic.jpg", mosaic, 92)
    # GeoTIFF (UTM)
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(out / "mosaic.tif", "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
                       crs=f"EPSG:{geo.epsg}", transform=from_origin(e0, n1, res, res), compress="jpeg") as ds:
        ds.write(np.transpose(mosaic[:, :, ::-1], (2, 0, 1)))

    # 물체 표시
    vis = mosaic.copy()
    vis[~covered] = (40, 40, 40)
    if objects_csv:
        for o in csv.DictReader(open(objects_csv, encoding="utf-8-sig")):
            e, n = geo.to_xy(float(o["lat"]), float(o["lon"]))
            x, y = int((e - e0) / res), int((n1 - n) / res)
            if not (0 <= x < W and 0 <= y < H):
                continue
            col = (40, 40, 230) if o["kind"] == "litter" else (30, 160, 250)
            r = int(0.25 / res)
            cv2.circle(vis, (x, y), r, col, 3)
            cv2.putText(vis, f"{o['cls']} ({o['seen_frames']})", (x + r + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, col, 2, cv2.LINE_AA)
    # 축척·북쪽
    L = int(1.0 / res)
    cv2.line(vis, (30, H - 40), (30 + L, H - 40), (255, 255, 255), 6)
    cv2.putText(vis, "1 m", (30, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.arrowedLine(vis, (W - 60, 140), (W - 60, 40), (255, 255, 255), 6, tipLength=0.3)
    cv2.putText(vis, "N", (W - 75, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    _imwrite(out / "mosaic_objects.jpg", vis, 90)
    print(f"정사영상 {W}×{H}px ({res * 100:.0f} cm/px, {W * res:.1f}×{H * res:.1f} m) · 3D↔GPS 잔차 {rms:.2f} m "
          f"→ {out / 'mosaic_objects.jpg'} · mosaic.tif (GeoTIFF, EPSG:{geo.epsg})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.mosaic", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", help="orbit_map의 objects.csv")
    ap.add_argument("--res", type=float, default=0.01, help="m/px")
    a = ap.parse_args(argv)
    build(a.orbit, a.srt, a.out, a.objects, a.res)


if __name__ == "__main__":
    main()
