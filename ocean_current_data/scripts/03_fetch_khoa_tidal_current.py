"""국립해양조사원 조류예보(시계열) Open API (공공데이터포털) 에서 예보 지점의 1분 간격 유향·유속을 받아 CSV 로 저장하고,
같은 시각의 Open-Meteo(Copernicus 1/12°) 해류와 비교한다.

API  해양수산부 국립해양조사원_조류예보(시계열)  https://www.data.go.kr/data/15156024/openapi.do  (KOGL 제1유형, 출처표시)
     GET https://apis.data.go.kr/1192136/crntFcstTime/GetCrntFcstTimeApiService
         serviceKey, obsCode(예보 지점, 예 17LTC04), reqDate(YYYYMMDD), dataType(XML|JSON — 실제 응답은 XML), pageNo, numOfRows(최대 300)
     응답 item: obsvtrNm(지점명), lot, lat, predcDt("YYYY-MM-DD HH:MM", 1분 간격, 하루 1,440행), crdir(16방위 한글), crsp(유속, cm/s 로 해석)
키   환경변수 KHOA_API_KEY 또는 ocean_current_data/.secrets/khoa_api_key.txt (git 제외). 저장소에 키를 넣지 않는다.
한도 공공데이터포털 개발계정 기본 일 1,000회 안팎 → 지점·일당 5회. 필요한 지점·날짜만 받는다.

사용
  python ocean_current_data/scripts/03_fetch_khoa_tidal_current.py --stations 17LTC04,15LTC01 --start 2026-07-28 --end 2026-07-29 \
      --compare ocean_current_data/data/currents_2026-06-01_2026-09-30.nc --out ocean_current_data/data/khoa
출력  khoa/<obsCode>_<start>_<end>.csv (time_utc, time_kst, dir_deg, speed_mps, u, v), khoa/compare_open_meteo.md, khoa/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://apis.data.go.kr/1192136/crntFcstTime/GetCrntFcstTimeApiService"
DIR16 = {"북": 0, "북북동": 22.5, "북동": 45, "동북동": 67.5, "동": 90, "동남동": 112.5, "남동": 135, "남남동": 157.5,
         "남": 180, "남남서": 202.5, "남서": 225, "서남서": 247.5, "서": 270, "서북서": 292.5, "북서": 315, "북북서": 337.5}
STATIONS_FOUND = {  # 2026-10-02 탐색 (zone 15·17·18, LTC01~15). 경기만·서해 중부만 발췌
    "15LTC01": ("염하수도", 126.54300, 37.63072), "17LTC01": ("인천신항입구", 126.49138, 37.32111), "17LTC03": ("자월도남측", 126.35666, 37.32166),
    "17LTC04": ("문갑도동측", 126.14666, 37.18361), "17LTC05": ("울도", 126.06805, 36.99416), "17LTC06": ("가로림만입구", 126.30972, 36.97972),
    "18LTC01": ("난지도북측", 126.40555, 37.07416), "18LTC02": ("와도서측", 125.86972, 36.61972)}


def api_key() -> str:
    k = os.environ.get("KHOA_API_KEY")
    if k:
        return k.strip()
    p = ROOT / "ocean_current_data" / ".secrets" / "khoa_api_key.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("KHOA_API_KEY 환경변수 또는 ocean_current_data/.secrets/khoa_api_key.txt 가 필요합니다 (공공데이터포털에서 활용신청 후 발급)")


def fetch_day(key: str, obs: str, day: str, rows=300, retries=3):
    items = []
    for page in range(1, 6):
        q = {"serviceKey": key, "obsCode": obs, "reqDate": day, "dataType": "XML", "pageNo": page, "numOfRows": rows}
        url = ENDPOINT + "?" + urllib.parse.urlencode(q)
        for k in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception as e:  # noqa: BLE001
                if k == retries - 1:
                    raise
                time.sleep(5 * (k + 1))
        rc = root.findtext(".//resultCode")
        if rc != "00":
            raise RuntimeError(f"{obs} {day} p{page}: resultCode {rc} {root.findtext('.//resultMsg')}")
        got = root.findall(".//item")
        for it in got:
            items.append({c.tag: c.text for c in it})
        total = int(root.findtext(".//totalCount") or 0)
        if page * rows >= total:
            break
        time.sleep(0.3)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="17LTC04,15LTC01"); ap.add_argument("--start", default="2026-07-28"); ap.add_argument("--end", default="2026-07-29")
    ap.add_argument("--compare", default="ocean_current_data/data/currents_2026-06-01_2026-09-30.nc", help="Open-Meteo 해류 NetCDF (없으면 생략)")
    ap.add_argument("--out", default="ocean_current_data/data/khoa")
    a = ap.parse_args()
    key = api_key(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    d0 = datetime.fromisoformat(a.start); d1 = datetime.fromisoformat(a.end); days = [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]
    manifest = {"endpoint": ENDPOINT, "dataset": "https://www.data.go.kr/data/15156024/openapi.do", "license": "KOGL 제1유형 (출처표시: 해양수산부 국립해양조사원)",
                "fetched_utc": datetime.now(timezone.utc).isoformat(), "stations": {}, "n_requests": 0, "note": "crsp 는 cm/s 로 해석해 m/s 로 변환. predcDt 는 KST."}
    series = {}
    for obs in a.stations.split(","):
        rows = []
        for day in days:
            items = fetch_day(key, obs, day); manifest["n_requests"] += max(1, (len(items) + 299) // 300)
            for it in items:
                tk = datetime.strptime(it["predcDt"], "%Y-%m-%d %H:%M"); tu = tk - timedelta(hours=9)
                deg = DIR16.get(it["crdir"].strip(), np.nan); sp = float(it["crsp"]) / 100.0
                rows.append({"time_kst": tk.isoformat(), "time_utc": tu.isoformat(), "dir_deg": deg, "speed_mps": round(sp, 4),
                             "u": round(sp * np.sin(np.radians(deg)), 4), "v": round(sp * np.cos(np.radians(deg)), 4), "dir_ko": it["crdir"]})
            print(f"  {obs} {day}: {len(items)}행")
        name, lon, lat = STATIONS_FOUND.get(obs, (items[0]["obsvtrNm"] if items else obs, float(items[0]["lot"]) if items else None, float(items[0]["lat"]) if items else None))
        f = out / f"{obs}_{a.start}_{a.end}.csv"
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
        manifest["stations"][obs] = {"name": name, "lon": lon, "lat": lat, "file": f.name, "n_rows": len(rows),
                                     "speed_mps_max": round(max(r["speed_mps"] for r in rows), 2), "speed_mps_median": round(float(np.median([r["speed_mps"] for r in rows])), 2)}
        series[obs] = (name, lon, lat, rows)
    # ---- Open-Meteo 와 비교 (시간 평균)
    cmp_lines = []
    if a.compare and (ROOT / a.compare).exists():
        import netCDF4 as nc
        ds = nc.Dataset(ROOT / a.compare); la = ds["lat"][:]; lo = ds["lon"][:]
        t = nc.num2date(ds["time"][:], ds["time"].units); tt = np.array([datetime(x.year, x.month, x.day, x.hour) for x in t])
        cmp_lines = ["| 지점 | 비교 시각 수 | KHOA 유속 중앙값 | Open-Meteo 중앙값 | 유속 상관 r | 유속 RMSE | 방향차 중앙값 | 위상차(상관 최대 지연) |", "|---|---|---|---|---|---|---|---|"]
        for obs, (name, lon, lat, rows) in series.items():
            i = int(np.argmin(abs(la - lat))); j = int(np.argmin(abs(lo - lon)))
            # KHOA 1분 → 정시 평균(벡터)
            by_h = {}
            for r in rows:
                h = datetime.fromisoformat(r["time_utc"]).replace(minute=0); by_h.setdefault(h, []).append((r["u"], r["v"]))
            hs = sorted(h for h in by_h if h in set(tt))
            if not hs:
                cmp_lines.append(f"| {name} | 0 | — | — | — | — | — | — |"); continue
            ku = np.array([np.mean([p[0] for p in by_h[h]]) for h in hs]); kv = np.array([np.mean([p[1] for p in by_h[h]]) for h in hs])
            idx = [int(np.nonzero(tt == h)[0][0]) for h in hs]
            ou = np.array([float(ds["x_sea_water_velocity"][k, i, j]) for k in idx]); ov = np.array([float(ds["y_sea_water_velocity"][k, i, j]) for k in idx])
            ks = np.hypot(ku, kv); os_ = np.hypot(ou, ov); ok = np.isfinite(os_)
            if ok.sum() < 6:
                cmp_lines.append(f"| {name} | {int(ok.sum())} | — | — | — | — | — | Open-Meteo 결측 |"); continue
            r = float(np.corrcoef(ks[ok], os_[ok])[0, 1]); rmse = float(np.sqrt(np.mean((ks[ok] - os_[ok]) ** 2)))
            kd = (np.degrees(np.arctan2(ku, kv)) + 360) % 360; od = (np.degrees(np.arctan2(ou, ov)) + 360) % 360
            dd = np.abs(((kd - od + 180) % 360) - 180); med_dd = float(np.median(dd[ok]))
            # 위상: 동서 성분 교차상관 최대 지연 (시간)
            x = ku[ok] - ku[ok].mean(); y = ou[ok] - ou[ok].mean(); lags = range(-4, 5)
            cc = [np.corrcoef(x[max(0, l):len(x) + min(0, l)], y[max(0, -l):len(y) - max(0, l)])[0, 1] if len(x) - abs(l) > 5 else np.nan for l in lags]
            best = list(lags)[int(np.nanargmax(cc))]
            cmp_lines.append(f"| {name} ({obs}) | {int(ok.sum())} | {np.median(ks[ok]):.2f} m/s | {np.median(os_[ok]):.2f} m/s | {r:.2f} | {rmse:.2f} m/s | {med_dd:.0f}° | {best:+d} h |")
        (out / "compare_open_meteo.md").write_text("\n".join(cmp_lines) + "\n\n유속·방향은 KHOA 1분 예보를 정시 벡터 평균한 값과 Open-Meteo(Copernicus 1/12°, 조석 포함)의 가장 가까운 격자점. 방향차는 0~180°.\n", encoding="utf-8")
        print("\n".join(cmp_lines))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"요청 {manifest['n_requests']}회 →", out)


if __name__ == "__main__":
    main()
