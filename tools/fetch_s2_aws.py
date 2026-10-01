"""AWS 공개 버킷(sentinel-cogs)에서 Sentinel-2 L2A 밴드를 STAC 없이 경로 규칙으로 창(window)만 잘라 받는다.

STAC 서버(Planetary Computer, Earth Search)가 막힌 환경에서도 s3 버킷의 HTTPS 범위 읽기(COG)는 되는 경우가 있어 만들었다.
버킷 경로:  sentinel-s2-l2a-cogs/<zone>/<band letter>/<square>/<year>/<month>/<scene>/<B02|B03|B04|B08|TCI|SCL>.tif

  # 1) 장면 목록 (타일 전체 구름률·nodata 비율; 창 안 구름은 --check 로 SCL 을 읽어 계산)
  python tools/fetch_s2_aws.py --tile 4QCK --list 2025 [--check --bounds 368000 2407000 395000 2437000]
  # 2) 받기 (bounds 는 타일 UTM 좌표 m: left bottom right top)
  python tools/fetch_s2_aws.py --tile 4QCK --bounds 368000 2407000 395000 2437000 \
      --scenes S2B_4QCK_20250502_0_L2A,S2C_4QCK_20251103_0_L2A --out s2 --prefix niihau

출력: <out>/<prefix>_<band>_<YYYYMMDD>.tif (SCL 은 20 m → 10 m 최근접 리샘플). scripts/18_priority_flight.py 의 입력 형식.
프록시 환경: HTTPS_PROXY 와 CA 번들(기본 /root/.ccr/ca-bundle.crt, 없으면 무시)을 GDAL 에 넘긴다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
CLOUD_SCL = (3, 8, 9, 10)


def _env():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy", "")
    if proxy:
        os.environ.setdefault("GDAL_HTTP_PROXY", proxy)
    ca = os.environ.get("CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    if Path(ca).exists():
        for k in ("GDAL_CURL_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ.setdefault(k, ca)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")


def _get(url: str) -> str:
    return subprocess.run(["curl", "-sS", "-m", "90", url], capture_output=True, text=True).stdout


def tile_parts(tile: str):
    m = re.fullmatch(r"(\d{1,2})([C-X])([A-Z]{2})", tile.upper())
    if not m:
        raise SystemExit("타일은 예: 4QCK, 52SBG")
    return str(int(m.group(1))), m.group(2), m.group(3)


def scene_dir(scene: str) -> str:
    zone, band, sq = tile_parts(scene.split("_")[1])
    d = scene.split("_")[2]
    return f"sentinel-s2-l2a-cogs/{zone}/{band}/{sq}/{d[:4]}/{int(d[4:6])}/{scene}/"


def list_scenes(tile: str, year: int, check_bounds=None):
    zone, band, sq = tile_parts(tile)
    scenes = []
    for m in range(1, 13):
        xml = _get(f"{BUCKET}?list-type=2&prefix=sentinel-s2-l2a-cogs/{zone}/{band}/{sq}/{year}/{m}/&delimiter=/")
        scenes += re.findall(r"<Prefix>(sentinel-s2-l2a-cogs/[^<]+/\d+/\d+/[^<]+/)</Prefix>", xml)
    rows = []
    for s in scenes:
        sid = s.rstrip("/").split("/")[-1]
        try:
            p = json.loads(_get(BUCKET + s + sid + ".json"))["properties"]
            row = {"scene": sid, "date": p.get("datetime", "")[:10], "cloud_tile": p.get("eo:cloud_cover"),
                   "nodata_tile": p.get("s2:nodata_pixel_percentage")}
        except Exception:
            row = {"scene": sid, "date": "?", "cloud_tile": None, "nodata_tile": None}
        if check_bounds is not None:
            row.update(window_cloud(sid, check_bounds))
        rows.append(row)
    rows.sort(key=lambda r: (r.get("cloud_win", r["cloud_tile"]) if r.get("cloud_win", r["cloud_tile"]) is not None else 999))
    return rows


def window_cloud(scene: str, bounds):
    """창 안 SCL 로 구름·nodata 비율 (%)."""
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    _env()
    try:
        with rasterio.open("/vsicurl/" + BUCKET + scene_dir(scene) + "SCL.tif") as src:
            arr = src.read(1, window=from_bounds(*bounds, src.transform))
        return {"cloud_win": round(float(np.isin(arr, CLOUD_SCL).mean() * 100), 2),
                "nodata_win": round(float((arr == 0).mean() * 100), 2)}
    except Exception as e:  # noqa: BLE001
        return {"cloud_win": None, "nodata_win": None, "err": str(e)[:80]}


def fetch(scene: str, bounds, out: Path, prefix: str, bands=("B03", "B08", "B04", "B02", "TCI", "SCL")):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    _env()
    out.mkdir(parents=True, exist_ok=True)
    date = scene.split("_")[2]
    L, B, R, T = bounds
    ref_shape = None
    for band in bands:
        url = "/vsicurl/" + BUCKET + scene_dir(scene) + band + ".tif"
        with rasterio.open(url) as src:
            win = from_bounds(L, B, R, T, src.transform)
            if band == "SCL" and ref_shape is not None:          # 20 m → 10 m 격자에 맞춤
                arr = src.read(window=win, out_shape=(1, *ref_shape), resampling=Resampling.nearest)
                tr = rasterio.transform.from_origin(L, T, (R - L) / ref_shape[1], (T - B) / ref_shape[0])
            else:
                arr = src.read(window=win); tr = src.window_transform(win)
                if band == "B03":
                    ref_shape = arr.shape[1:]
            prof = src.profile
            prof.update(height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], transform=tr, driver="GTiff", compress="deflate")
            dst_path = out / f"{prefix}_{band}_{date}.tif"
            with rasterio.open(dst_path, "w", **prof) as dst:
                dst.write(arr)
            print(f"  {band:4s} {arr.shape[1:]} nodata {float((arr == 0).mean()):.3f} → {dst_path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", required=True, help="MGRS 타일, 예 4QCK (니하우) · 52SBG (옹진)")
    ap.add_argument("--list", type=int, metavar="YEAR", help="그 해 장면 목록을 구름률 순으로")
    ap.add_argument("--check", action="store_true", help="--list 에서 창 안 구름률도 (SCL 읽음, 느림)")
    ap.add_argument("--bounds", type=float, nargs=4, metavar=("L", "B", "R", "T"), help="타일 UTM 좌표 m")
    ap.add_argument("--scenes", help="쉼표 구분 장면 id")
    ap.add_argument("--out", default="s2"); ap.add_argument("--prefix", default="site")
    a = ap.parse_args()
    if a.list:
        rows = list_scenes(a.tile, a.list, a.bounds if a.check else None)
        for r in rows[:30]:
            print(r)
        return
    if not (a.bounds and a.scenes):
        raise SystemExit("--bounds 와 --scenes 가 필요")
    for sc in a.scenes.split(","):
        print(sc)
        fetch(sc, a.bounds, Path(a.out), a.prefix)


if __name__ == "__main__":
    main()
