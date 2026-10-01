"""Open-Meteo Marine/Archive API 에서 경기만 격자의 표층 해류·바람을 받아 CF 규약 NetCDF 로 저장 (OpenDrift 입력).

출처
  해류  Open-Meteo Marine Weather API  https://marine-api.open-meteo.com/v1/marine
        변수 ocean_current_velocity(km/h), ocean_current_direction(° 흐르는 방향). 원자료는 Copernicus Marine Service 전지구 1/12° 해양 모형
        (Open-Meteo 문서 참조). 라이선스 CC BY 4.0 (비상업 무료, 출처 표기).
  바람  Open-Meteo Historical Weather API  https://archive-api.open-meteo.com/v1/archive  wind_speed_10m, wind_direction_10m (ERA5/ECMWF 기반)

사용
  python ocean_current_data/scripts/01_fetch_open_meteo_currents.py --start 2026-06-01 --end 2026-09-30 \
      --bbox 125.80 37.00 126.95 37.95 --step 0.0833 --out ocean_current_data/data
출력
  currents_<start>_<end>.nc  : x_sea_water_velocity, y_sea_water_velocity (m/s) [time, lat, lon]
  wind_<start>_<end>.nc      : x_wind, y_wind (m/s) [time, lat, lon]  (해류보다 성긴 0.25° 격자)
  manifest.json              : 요청 URL, 시각, 격자, 결측 비율, 출처·라이선스
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MARINE = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def get_json(url: str, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))


def fetch_grid(base, lats, lons, hourly, start, end, chunk=12, extra=""):
    """여러 지점을 chunk 개씩 한 요청으로. 반환 dict[(lat,lon)] -> {var: list}, times."""
    pts = [(la, lo) for la in lats for lo in lons]
    out, times, urls = {}, None, []
    for i in range(0, len(pts), chunk):
        sub = pts[i:i + chunk]
        q = {"latitude": ",".join(f"{p[0]:.4f}" for p in sub), "longitude": ",".join(f"{p[1]:.4f}" for p in sub),
             "hourly": ",".join(hourly), "start_date": start, "end_date": end, "timezone": "UTC"}
        url = base + "?" + urllib.parse.urlencode(q) + extra
        urls.append(url)
        js = get_json(url)
        if isinstance(js, dict):
            js = [js]
        for p, j in zip(sub, js):
            if times is None:
                times = j["hourly"]["time"]
            out[p] = {v: j["hourly"].get(v) for v in hourly}
        time.sleep(0.3)
    return out, times, urls


def to_uv(speed_kmh, direction_deg):
    """속력(km/h)·방향(°, 흐르는 쪽 / 바람은 불어오는 쪽) → m/s 동·북 성분."""
    sp = np.array(speed_kmh, dtype=float) / 3.6
    th = np.radians(np.array(direction_deg, dtype=float))
    return sp * np.sin(th), sp * np.cos(th)


def write_nc(path: Path, times, lats, lons, u, v, uname, vname, std_u, std_v, title, source):
    import netCDF4 as nc
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", len(times)); ds.createDimension("lat", len(lats)); ds.createDimension("lon", len(lons))
    t = ds.createVariable("time", "f8", ("time",)); t.units = "seconds since 1970-01-01 00:00:00"; t.standard_name = "time"; t.calendar = "standard"
    t[:] = [datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() for s in times]
    la = ds.createVariable("lat", "f4", ("lat",)); la.units = "degrees_north"; la.standard_name = "latitude"; la.axis = "Y"; la[:] = lats
    lo = ds.createVariable("lon", "f4", ("lon",)); lo.units = "degrees_east"; lo.standard_name = "longitude"; lo.axis = "X"; lo[:] = lons
    for name, std, arr in ((uname, std_u, u), (vname, std_v, v)):
        var = ds.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan), zlib=True, complevel=4)
        var.units = "m s-1"; var.standard_name = std; var[:] = arr.astype(np.float32)
    ds.title = title; ds.source = source; ds.Conventions = "CF-1.8"; ds.history = f"created {datetime.now(timezone.utc).isoformat()} by 01_fetch_open_meteo_currents.py"
    ds.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--bbox", type=float, nargs=4, default=[125.80, 37.00, 126.95, 37.95], metavar=("W", "S", "E", "N"))
    ap.add_argument("--step", type=float, default=1 / 12, help="해류 격자 간격 (°). Open-Meteo 해류 원격자가 1/12°")
    ap.add_argument("--wind-step", type=float, default=0.25); ap.add_argument("--out", default="ocean_current_data/data")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, S, E, N = a.bbox
    lats = np.round(np.arange(S, N + 1e-9, a.step), 4); lons = np.round(np.arange(W, E + 1e-9, a.step), 4)
    print(f"해류 격자 {len(lats)}×{len(lons)} = {len(lats) * len(lons)} 지점, {a.start}~{a.end}")
    cur, times, urls_c = fetch_grid(MARINE, lats, lons, ["ocean_current_velocity", "ocean_current_direction"], a.start, a.end)
    T = len(times); U = np.full((T, len(lats), len(lons)), np.nan); V = U.copy()
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            d = cur[(la, lo)]
            if d["ocean_current_velocity"] is None or all(x is None for x in d["ocean_current_velocity"]):
                continue
            sp = [np.nan if x is None else x for x in d["ocean_current_velocity"]]
            di = [np.nan if x is None else x for x in d["ocean_current_direction"]]
            U[:, i, j], V[:, i, j] = to_uv(sp, di)
    miss = float(np.isnan(U).mean())
    print(f"해류 결측(육지 포함) {miss * 100:.1f} %  | 속력 중앙값 {np.nanmedian(np.hypot(U, V)):.2f} m/s, 최대 {np.nanmax(np.hypot(U, V)):.2f} m/s")
    fc = out / f"currents_{a.start}_{a.end}.nc"
    write_nc(fc, times, lats, lons, U, V, "x_sea_water_velocity", "y_sea_water_velocity", "x_sea_water_velocity", "y_sea_water_velocity",
             "Gyeonggi Bay surface currents (Open-Meteo Marine API, Copernicus Marine 1/12 deg)", "Open-Meteo Marine Weather API; original data Copernicus Marine Service")
    # 바람
    wl = np.round(np.arange(S, N + 1e-9, a.wind_step), 4); wo = np.round(np.arange(W, E + 1e-9, a.wind_step), 4)
    wind, wtimes, urls_w = fetch_grid(ARCHIVE, wl, wo, ["wind_speed_10m", "wind_direction_10m"], a.start, a.end)
    WU = np.full((len(wtimes), len(wl), len(wo)), np.nan); WV = WU.copy()
    for i, la in enumerate(wl):
        for j, lo in enumerate(wo):
            d = wind[(la, lo)]
            sp = [np.nan if x is None else x for x in d["wind_speed_10m"]]; di = [np.nan if x is None else x for x in d["wind_direction_10m"]]
            u, v = to_uv(sp, di); WU[:, i, j], WV[:, i, j] = -u, -v         # 기상 방향은 '불어오는 쪽' → 부호 반전
    fw = out / f"wind_{a.start}_{a.end}.nc"
    write_nc(fw, wtimes, wl, wo, WU, WV, "x_wind", "y_wind", "x_wind", "y_wind", "10 m wind (Open-Meteo Historical Weather API, ERA5-based)", "Open-Meteo Historical Weather API")
    man = {"fetched_utc": datetime.now(timezone.utc).isoformat(), "period": [a.start, a.end], "bbox": a.bbox,
           "currents": {"file": fc.name, "grid": [len(lats), len(lons)], "step_deg": a.step, "n_times": T, "missing_frac": round(miss, 4), "n_requests": len(urls_c), "example_url": urls_c[0],
                        "variables": "ocean_current_velocity (km/h → m/s), ocean_current_direction (° toward)", "source": "Open-Meteo Marine Weather API (https://open-meteo.com/en/docs/marine-weather-api); upstream Copernicus Marine Service global ocean model 1/12°", "license": "CC BY 4.0 (Open-Meteo), non-commercial"},
           "wind": {"file": fw.name, "grid": [len(wl), len(wo)], "step_deg": a.wind_step, "n_times": len(wtimes), "n_requests": len(urls_w), "example_url": urls_w[0],
                    "source": "Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api), ERA5/ECMWF reanalysis", "license": "CC BY 4.0"}}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", fc, fw, out / "manifest.json")


if __name__ == "__main__":
    main()
