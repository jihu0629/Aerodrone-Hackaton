"""국립해양조사원 조류예보(문갑도동측 17LTC04, 1분)로 Open-Meteo 1/12° 해류를 문갑도 주변에서 보정한다.

보정 방법 (지점 1곳 기준의 1차 보정)
  1. KHOA 1분 → 정시 벡터 평균, Open-Meteo 최근접 격자점과 같은 시각으로 맞춤
  2. 위상: 동·북 성분 교차상관이 최대인 지연(시간, ±6 h) → 격자 시간축을 그만큼 이동
  3. 방향: 같은 시각 벡터 각도차의 원형 중앙값 → 격자 벡터를 그 각도만큼 회전
  4. 크기: 유속 중앙값 비 → 배율
  5. 지점에서 멀어질수록 보정을 줄인다: 가중 w = exp(−d²/2σ²), σ = 15 km (문갑도 둘레는 w≈1)
  6. KHOA 59일 평균 벡터(잔차류)·조류 타원 장축 방향도 계산 → '잔차류 맞이' 점수의 방향 입력

사용
  python mungap_opendrift/scripts/02_tune_with_khoa.py --khoa ocean_current_data/data/khoa/17LTC04_2026-06-01_2026-07-29.csv \
      --currents ocean_current_data/data/currents_2026-06-01_2026-09-30.nc --out ocean_current_data/data/currents_nudged_mungap.nc
출력  보정 NetCDF, mungap_opendrift/outputs_khoa_tuned/tuning_params.json (배율·회전·지연·잔차류 방향·검증 통계)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def circ_median_deg(d):
    d = np.asarray(d, float); d = ((d + 180) % 360) - 180
    # 원형 중앙값 근사: 후보 각도 중 절대 각도차 합이 최소인 것
    cands = np.linspace(-180, 180, 721)
    cost = [np.abs(((d - c + 180) % 360) - 180).sum() for c in cands]
    return float(cands[int(np.argmin(cost))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--khoa", default="ocean_current_data/data/khoa/17LTC04_2026-06-01_2026-07-29.csv")
    ap.add_argument("--currents", default="ocean_current_data/data/currents_2026-06-01_2026-09-30.nc")
    ap.add_argument("--out", default="ocean_current_data/data/currents_nudged_mungap.nc")
    ap.add_argument("--sigma-km", type=float, default=15.0); ap.add_argument("--params-out", default="mungap_opendrift/outputs_khoa_tuned/tuning_params.json")
    a = ap.parse_args()
    import netCDF4 as nc
    rows = list(csv.DictReader(open(ROOT / a.khoa, encoding="utf-8-sig")))
    by_h = {}
    for r in rows:
        h = datetime.fromisoformat(r["time_utc"]).replace(minute=0); by_h.setdefault(h, []).append((float(r["u"]), float(r["v"])))
    hs = sorted(by_h); ku = np.array([np.mean([p[0] for p in by_h[h]]) for h in hs]); kv = np.array([np.mean([p[1] for p in by_h[h]]) for h in hs])
    # KHOA 지점 위치는 manifest 에서
    man = json.loads((ROOT / "ocean_current_data/data/khoa/manifest.json").read_text(encoding="utf-8"))
    code = Path(a.khoa).name.split("_")[0]; st = man["stations"].get(code, {"lon": 126.14666, "lat": 37.18361, "name": code})
    lon0, lat0 = float(st["lon"]), float(st["lat"])
    src = nc.Dataset(ROOT / a.currents); la = src["lat"][:]; lo = src["lon"][:]
    t = nc.num2date(src["time"][:], src["time"].units); tt = np.array([datetime(x.year, x.month, x.day, x.hour) for x in t])
    U = np.array(src["x_sea_water_velocity"][:]); V = np.array(src["y_sea_water_velocity"][:])
    i0 = int(np.argmin(abs(la - lat0))); j0 = int(np.argmin(abs(lo - lon0)))
    idx = {h: k for k, h in enumerate(tt)}; sel = [h for h in hs if h in idx]
    ku_s = np.array([ku[hs.index(h)] for h in sel]); kv_s = np.array([kv[hs.index(h)] for h in sel])
    ou = np.array([U[idx[h], i0, j0] for h in sel]); ov = np.array([V[idx[h], i0, j0] for h in sel])
    ok = np.isfinite(ou) & np.isfinite(ov)
    ku_s, kv_s, ou, ov = ku_s[ok], kv_s[ok], ou[ok], ov[ok]
    # 위상 지연
    def xcorr_lag(x, y, maxlag=6):
        x = x - x.mean(); y = y - y.mean(); best, bl = -2, 0
        for l in range(-maxlag, maxlag + 1):
            xa = x[max(0, l):len(x) + min(0, l)]; ya = y[max(0, -l):len(y) - max(0, l)]
            if len(xa) > 24:
                c = np.corrcoef(xa, ya)[0, 1]
                if c > best: best, bl = c, l
        return bl, best
    lag_u, cu = xcorr_lag(ku_s, ou); lag_v, cv = xcorr_lag(kv_s, ov); lag = int(round((lag_u + lag_v) / 2))
    # 지연 적용 후 방향·크기 — xcorr 정의와 같은 짝: ku[t+lag] ↔ ou[t]  (lag<0 이면 KHOA 가 앞선다)
    if lag > 0: ku2, kv2, ou2, ov2 = ku_s[lag:], kv_s[lag:], ou[:-lag], ov[:-lag]
    elif lag < 0: ku2, kv2, ou2, ov2 = ku_s[:lag], kv_s[:lag], ou[-lag:], ov[-lag:]
    else: ou2, ov2, ku2, kv2 = ou, ov, ku_s, kv_s
    kd = np.degrees(np.arctan2(ku2, kv2)); od = np.degrees(np.arctan2(ou2, ov2))
    strong = np.hypot(ou2, ov2) > 0.15                                       # 약한 흐름은 방향 잡음 → 제외
    rot = circ_median_deg((kd - od)[strong])
    ks = np.hypot(ku2, kv2); os_ = np.hypot(ou2, ov2); scale = float(np.median(ks) / max(np.median(os_), 1e-6))
    r_before = float(np.corrcoef(np.hypot(ku_s, kv_s), np.hypot(ou, ov))[0, 1]); dd_before = float(np.median(np.abs(((np.degrees(np.arctan2(ku_s, kv_s)) - np.degrees(np.arctan2(ou, ov)) + 180) % 360) - 180)))
    # 보정 적용 (회전 → 배율 → 시간 이동), 거리 가중
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    dkm = np.hypot((LO - lon0) * 111.32 * np.cos(np.radians(lat0)), (LA - lat0) * 111.32)
    w = np.exp(-dkm ** 2 / (2 * a.sigma_km ** 2))
    th = np.radians(rot) * w                                                  # 지점 근처만 회전
    sc = 1 + (scale - 1) * w
    Ur = (U * np.cos(th) + V * np.sin(th)) * sc; Vr = (-U * np.sin(th) + V * np.cos(th)) * sc   # 벡터를 +rot(시계) 회전: 방위각 증가
    if lag != 0:
        Ush = np.roll(Ur, lag, axis=0); Vsh = np.roll(Vr, lag, axis=0)       # ku[t] ≈ ou[t−lag] 이므로 보정값(t) = OM(t−lag) = roll(OM, lag)[t]
        Un = Ur * (1 - w) + Ush * w; Vn = Vr * (1 - w) + Vsh * w
    else:
        Un, Vn = Ur, Vr
    # 보정 후 검증
    ou3 = np.array([Un[idx[h], i0, j0] for h in np.array(sel)[ok]]); ov3 = np.array([Vn[idx[h], i0, j0] for h in np.array(sel)[ok]])
    r_after = float(np.corrcoef(np.hypot(ku_s, kv_s), np.hypot(ou3, ov3))[0, 1]); dd_after = float(np.median(np.abs(((np.degrees(np.arctan2(ku_s, kv_s)) - np.degrees(np.arctan2(ou3, ov3)) + 180) % 360) - 180)))
    rmse_b = float(np.sqrt(np.mean((np.hypot(ku_s, kv_s) - np.hypot(ou, ov)) ** 2))); rmse_a = float(np.sqrt(np.mean((np.hypot(ku_s, kv_s) - np.hypot(ou3, ov3)) ** 2)))
    vec_rmse_b = float(np.sqrt(np.mean((ku_s - ou) ** 2 + (kv_s - ov) ** 2))); vec_rmse_a = float(np.sqrt(np.mean((ku_s - ou3) ** 2 + (kv_s - ov3) ** 2)))
    # 잔차류·조류 타원
    res_dir = float((np.degrees(np.arctan2(ku.mean(), kv.mean())) + 360) % 360); res_mag = float(np.hypot(ku.mean(), kv.mean()))
    cov = np.cov(np.stack([ku - ku.mean(), kv - kv.mean()])); evals, evecs = np.linalg.eigh(cov); major = evecs[:, int(np.argmax(evals))]
    major_dir = float((np.degrees(np.arctan2(major[0], major[1])) + 360) % 180)
    # 저장
    out = ROOT / a.out
    dst = nc.Dataset(out, "w", format="NETCDF4")
    for name, dim in (("time", "time"), ("lat", "lat"), ("lon", "lon")):
        dst.createDimension(dim, len(src.dimensions[dim])); v = dst.createVariable(name, src[name].dtype, (dim,)); v.setncatts({k: src[name].getncattr(k) for k in src[name].ncattrs()}); v[:] = src[name][:]
    for name, arr in (("x_sea_water_velocity", Un), ("y_sea_water_velocity", Vn)):
        v = dst.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=np.float32(np.nan), zlib=True, complevel=4); v.units = "m s-1"; v.standard_name = name; v[:] = arr.astype(np.float32)
    dst.title = "Open-Meteo currents nudged to KHOA tidal current forecast at " + st["name"]; dst.Conventions = "CF-1.8"
    dst.history = f"02_tune_with_khoa.py: rot {rot:+.1f} deg, scale {scale:.3f}, lag {lag:+d} h, sigma {a.sigma_km} km around ({lon0},{lat0})"; dst.close()
    params = {"station": {"code": code, "name": st["name"], "lon": lon0, "lat": lat0}, "n_hours_compared": int(ok.sum()),
              "lag_h": lag, "lag_corr": {"u": round(cu, 3), "v": round(cv, 3)}, "rotation_deg": round(rot, 1), "speed_scale": round(scale, 3), "sigma_km": a.sigma_km,
              "before": {"speed_corr": round(r_before, 3), "speed_rmse_mps": round(rmse_b, 3), "dir_diff_median_deg": round(dd_before, 1), "vector_rmse_mps": round(vec_rmse_b, 3)},
              "after": {"speed_corr": round(r_after, 3), "speed_rmse_mps": round(rmse_a, 3), "dir_diff_median_deg": round(dd_after, 1), "vector_rmse_mps": round(vec_rmse_a, 3)},
              "khoa_residual": {"toward_deg": round(res_dir), "speed_mps": round(res_mag, 3)}, "khoa_tidal_ellipse_major_axis_deg": round(major_dir),
              "khoa_speed_median_mps": round(float(np.median(np.hypot(ku, kv))), 3), "khoa_speed_max_mps": round(float(np.hypot(ku, kv).max()), 3), "out": str(out.relative_to(ROOT))}
    po = ROOT / a.params_out; po.parent.mkdir(parents=True, exist_ok=True); po.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(params, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
