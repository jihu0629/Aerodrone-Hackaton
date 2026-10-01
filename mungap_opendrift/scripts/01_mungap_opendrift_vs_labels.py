"""문갑도 — 해류 데이터로 OpenDrift 표류·좌초를 돌려 "어디에 쌓이나" 를 추정하고, 기업 드론 조사 라벨(42개, 2026-07-29)과 비교.

입력
  ocean_current_data/data/currents_2026-06-01_2026-09-30.nc, wind_*.nc   (Open-Meteo: Copernicus SMOC 1/12° 조석 포함, ERA5 바람)
  input/s2_mungap/mungap_*_{20260615,20260804,20260916}.tif               (Sentinel-2 10 m 위성 해안선, 창 238790 4114800 246300 4122010)
  input/s2_mungap/labels_42.csv                                         (기업 라벨 중심 lon,lat; 저장소 밖 자료라 git 제외)
실행
  python mungap_opendrift/scripts/01_mungap_opendrift_vs_labels.py --start 2026-06-01 --survey 2026-07-29 --n-offshore 12000 --n-han 4000
출력 (mungap_opendrift/outputs/)
  stranded_particles_mungap.csv, segments_mungap.csv, summary.json, 비교표.md, 50_mungap_stranding_vs_labels_map.png, 51_mungap_capture_curves.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from litter3d import priority as P  # noqa: E402
from litter3d import strategy as S  # noqa: E402

C1, C2, C3, C4, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9b9a95"
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SCEN = {"offshore": ("덕적군도 주변 외해 균일 방출", (125.85, 37.00, 126.40, 37.36)),
        "han_river": ("한강 하구 유출 (조강·염하 합류 수역)", (126.56, 37.74, 126.66, 37.80))}


def setup_mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ["fonts/NanumGothic.ttf", r"C:\Windows\Fonts\malgun.ttf"]:
        if Path(f).exists():
            fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=T2, length=0); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def run_opendrift(cur_nc, wind_nc, t0, t_end, n_by_scen, out, seed=0, windage=0.02, diff=10.0, dt_min=20, out_h=6):
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic
    rng = np.random.default_rng(seed); rows = []
    for name, (desc, (W, S_, E, N)) in SCEN.items():
        n = n_by_scen[name]
        o = OceanDrift(loglevel=40)
        o.add_reader([reader_netCDF_CF_generic.Reader(str(cur_nc)), reader_netCDF_CF_generic.Reader(str(wind_nc))])
        o.set_config("general:coastline_action", "stranding"); o.set_config("environment:fallback:horizontal_diffusivity", diff)
        o.set_config("drift:stokes_drift", False); o.set_config("drift:advection_scheme", "runge-kutta4")
        lon = rng.uniform(W, E, n); lat = rng.uniform(S_, N, n)
        rel_end = (t_end - t0).total_seconds() - (2 if name == "offshore" else 10) * 86400
        times = [t0 + timedelta(seconds=float(s)) for s in np.sort(rng.uniform(0, rel_end, n))]
        o.seed_elements(lon=lon, lat=lat, time=times, wind_drift_factor=rng.uniform(0.5, 1.5, n) * windage)
        o.run(end_time=t_end, time_step=timedelta(minutes=dt_min), time_step_output=timedelta(hours=out_h), outfile=str(out / f"opendrift_mungap_{name}.nc"))
        ds = o.result; meanings = str(ds["status"].attrs.get("flag_meanings", "active stranded")).split()
        k_str = meanings.index("stranded") if "stranded" in meanings else 1
        st = ds["status"].values; lonh = ds["lon"].values; lath = ds["lat"].values; valid = ~np.isnan(lonh)
        for i in range(st.shape[0]):
            vi = np.nonzero(valid[i])[0]
            if len(vi) == 0: continue
            i0, i1 = vi[0], vi[-1]; final = int(st[i, i1]) if not np.isnan(st[i, i1]) else 0; stranded = final == k_str
            rows.append({"scenario": name, "id": i, "release_lon": float(lonh[i, i0]), "release_lat": float(lath[i, i0]), "release_time": times[i].isoformat(),
                         "final_status": meanings[final] if final < len(meanings) else str(final), "stranded": int(stranded), "stranded_t0": int(stranded and len(vi) <= 1),
                         "lon": float(lonh[i, i1]), "lat": float(lath[i, i1]), "drift_days": round(len(vi) * out_h / 24, 2)})
        print(f"  {name}: {n}개, 좌초 {sum(r['stranded'] for r in rows if r['scenario'] == name)} (방출 즉시 {sum(r['stranded_t0'] for r in rows if r['scenario'] == name)})")
    return rows


def mean_wind(wind_nc, lat0, lon0, t0, t1):
    import netCDF4 as nc
    w = nc.Dataset(wind_nc); wl = w["lat"][:]; wo = w["lon"][:]; tw = nc.num2date(w["time"][:], w["time"].units)
    m = np.array([t0 <= datetime(x.year, x.month, x.day, x.hour) < t1 for x in tw]); i = int(np.argmin(abs(wl - lat0))); j = int(np.argmin(abs(wo - lon0)))
    u = np.array(w["x_wind"][m, i, j]); v = np.array(w["y_wind"][m, i, j])
    return float((np.degrees(np.arctan2(-u.mean(), -v.mean())) + 360) % 360), float(np.hypot(u, v).mean())


def mean_current(cur_nc, lat0, lon0, t0, t1, r=0.1):
    import netCDF4 as nc
    d = nc.Dataset(cur_nc); la = d["lat"][:]; lo = d["lon"][:]; t = nc.num2date(d["time"][:], d["time"].units)
    m = np.array([t0 <= datetime(x.year, x.month, x.day, x.hour) < t1 for x in t])
    sel = [(i, j) for i in range(len(la)) for j in range(len(lo)) if abs(la[i] - lat0) <= r and abs(lo[j] - lon0) <= r]
    U = np.nanmean([np.array(d["x_sea_water_velocity"][m, i, j]) for i, j in sel], 0); V = np.nanmean([np.array(d["y_sea_water_velocity"][m, i, j]) for i, j in sel], 0)
    return float((np.degrees(np.arctan2(U.mean(), V.mean())) + 360) % 360), float(np.hypot(U.mean(), V.mean())), float(np.median(np.hypot(U, V)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--survey", default="2026-07-29")
    ap.add_argument("--n-offshore", type=int, default=12000); ap.add_argument("--n-han", type=int, default=4000)
    ap.add_argument("--windage", type=float, default=0.02); ap.add_argument("--diffusivity", type=float, default=10.0); ap.add_argument("--dt-min", type=int, default=20)
    ap.add_argument("--match-m", type=float, default=1500.0); ap.add_argument("--top-frac", type=float, default=0.3)
    ap.add_argument("--data", default="ocean_current_data/data"); ap.add_argument("--out", default="mungap_opendrift/outputs"); ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cur_nc = ROOT / a.data / "currents_2026-06-01_2026-09-30.nc"; wind_nc = ROOT / a.data / "wind_2026-06-01_2026-09-30.nc"
    t0 = datetime.fromisoformat(a.start); t_end = datetime.fromisoformat(a.survey)

    # ---- 1. OpenDrift
    if a.skip_run and (out / "stranded_particles_mungap.csv").exists():
        rows = list(csv.DictReader(open(out / "stranded_particles_mungap.csv", encoding="utf-8-sig")))
        for r in rows:
            for k in ("release_lon", "release_lat", "lon", "lat", "drift_days"): r[k] = float(r[k])
            for k in ("stranded", "stranded_t0", "id"): r[k] = int(r[k])
    else:
        print(f"OpenDrift: {a.start} → {a.survey} ({(t_end - t0).days}일), 외해 {a.n_offshore}·한강 {a.n_han}개, 풍압 {a.windage}, 확산 {a.diffusivity}")
        rows = run_opendrift(cur_nc, wind_nc, t0, t_end, {"offshore": a.n_offshore, "han_river": a.n_han}, out, windage=a.windage, diff=a.diffusivity, dt_min=a.dt_min)
        with open(out / "stranded_particles_mungap.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

    # ---- 2. 문갑도 위성 해안 (Sentinel-2 10 m) 과 라벨
    spec = importlib.util.spec_from_file_location("s19", ROOT / "scripts" / "19_strategy_compare.py"); m19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m19)
    wind_jul, wspd_jul = mean_wind(wind_nc, 37.17, 126.10, datetime(2026, 7, 1), t_end)
    wind_all, wspd_all = mean_wind(wind_nc, 37.17, 126.10, t0, t_end)
    cur_dir, cur_mag, cur_med = mean_current(cur_nc, 37.17, 126.10, t0, t_end)
    print(f"측정 바람: 7월 불어오는 방향 {wind_jul:.0f}° ({wspd_jul:.1f} m/s), 6~7월 {wind_all:.0f}° | 잔차류: 흐르는 방향 {cur_dir:.0f}°, {cur_mag:.3f} m/s, 조류 속력 중앙값 {cur_med:.2f} m/s")
    args = types.SimpleNamespace(s2_dir="input/s2_mungap", scenes="20260615,20260804,20260916", prefix="mungap", wind_from=wind_jul,
                                 density_grid=None, density_csv="input/s2_mungap/labels_42.csv")
    coast, transform, crs, px = m19.load_site(args)
    lab_xy, lab_w, _ = m19.load_density(args, crs)
    print(f"문갑도 위성 해안 {coast.n * px / 1000:.1f} km, 라벨 {len(lab_xy)}개")
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from scipy import ndimage as ndi
    from scipy.stats import spearmanr
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True); tree = cKDTree(coast.xy)
    segs = P.make_segments(coast, 200.0); seg_of = np.zeros(coast.n, int)
    for s in segs: seg_of[s.i0:s.i1] = s.seg_id
    L_km = np.array([s.length_m for s in segs]) / 1000
    n_str = {k: np.zeros(len(segs)) for k in SCEN}; matched = {k: 0 for k in SCEN}; stranded_pts = {k: [] for k in SCEN}
    for k in SCEN:
        pts = [r for r in rows if r["scenario"] == k and r["stranded"] and not r["stranded_t0"]]
        if not pts: continue
        x, y = tr.transform([r["lon"] for r in pts], [r["lat"] for r in pts]); xy = np.stack([x, y], 1)
        d, idx = tree.query(xy); ok = d <= a.match_m; matched[k] = int(ok.sum()); stranded_pts[k] = xy[ok]
        np.add.at(n_str[k], seg_of[idx[ok]], 1)
    tot = n_str["offshore"] + n_str["han_river"]
    print(f"문갑도 해안에 좌초(1.5 km 안 매칭): 외해 {matched['offshore']}, 한강 {matched['han_river']} → 합 {int(tot.sum())}, 좌초 있는 구간 {(tot > 0).sum()}/{len(segs)}")
    near_lab = tree.query(lab_xy)[1]; n_lab = np.zeros(len(segs)); np.add.at(n_lab, seg_of[near_lab], 1)
    rid = np.array([s.ring_id for s in segs])
    def smooth(v):
        o_ = np.zeros_like(v)
        for r_ in np.unique(rid): mk = rid == r_; o_[mk] = ndi.uniform_filter1d(v[mk], size=5, mode="wrap")
        return o_
    dens = smooth(tot / L_km)
    score_pt = np.zeros(coast.n)
    for s, sc in zip(segs, dens): score_pt[s.i0:s.i1] = sc
    coast.feats["stranding"] = score_pt
    # 잔차류 맞이: 바깥 법선이 흐름의 반대(해안으로 들어오는 흐름)일수록 +
    bearing = coast.feats["bearing"]; onshore = -np.cos(np.radians(cur_dir - bearing))
    # ---- 3. 점수별 라벨 포착 곡선
    variants = {"OpenDrift 좌초 밀도": score_pt, "만입도만": S.score_variant(coast, "bay"), f"기본(만입도+노출+띠폭, 측정 풍향 {wind_jul:.0f}°)": coast.score,
                f"노출만 (측정 7월 풍향 {wind_jul:.0f}°)": S.score_variant(coast, "exposure", wind_jul), "노출만 (315° 겨울 가정)": S.score_variant(coast, "exposure", 315.0),
                f"잔차류 맞이 (흐름 {cur_dir:.0f}°)": ndi.uniform_filter1d(onshore, 10, mode="wrap")}
    curves = {}
    for nm, sc in variants.items():
        xs, ys = P.capture_curve(coast, sc, lab_xy, lab_w); curves[nm] = [round(float(v) * 100, 1) for v in ys]
    print("라벨 포착 (상위 10/20/30/50 % 길이):"); [print(f"  {nm:40s}", [curves[nm][k] for k in (10, 20, 30, 50)]) for nm in curves]
    rho, pval = spearmanr(tot, n_lab); rho_s, p_s = spearmanr(dens, n_lab)
    order = np.argsort(-dens); cum = np.cumsum(L_km[order]) / L_km.sum(); top = np.zeros(len(segs), bool); top[order[cum <= a.top_frac]] = True
    hit = n_lab[top].sum() / n_lab.sum()
    print(f"구간 단위 Spearman: 좌초 수 vs 라벨 수 ρ={rho:.2f} (p={pval:.3f}), 평활 밀도 vs 라벨 ρ={rho_s:.2f} (p={p_s:.3f}); OpenDrift 상위 {a.top_frac * 100:.0f} % 구간에 라벨 {hit * 100:.0f} %")
    # 해안 방위별 비교 (8방위)
    oct_ = ((bearing + 22.5) // 45 % 8).astype(int); names8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    lab_oct = np.bincount(oct_[near_lab], minlength=8); str_oct = np.zeros(8)
    for k in SCEN:
        if len(stranded_pts[k]): str_oct += np.bincount(oct_[tree.query(stranded_pts[k])[1]], minlength=8)
    len_oct = np.bincount(oct_, minlength=8) * px / 1000
    by_dir = [{"dir": names8[i], "coast_km": round(float(len_oct[i]), 2), "labels": int(lab_oct[i]), "stranded": int(str_oct[i]),
               "labels_per_km": round(float(lab_oct[i] / max(len_oct[i], 1e-9)), 1), "stranded_per_km": round(float(str_oct[i] / max(len_oct[i], 1e-9)), 1)} for i in range(8)]
    print("해안 방위별 (라벨/km, 좌초/km):", [(d["dir"], d["labels_per_km"], d["stranded_per_km"]) for d in by_dir])
    # ---- 4. 경로 비용 (라벨 = 밀도)
    ops = S.Ops(alt_m=20.0); near = P.nearest_coast_index(coast, lab_xy)
    r_full = S.evaluate(coast, segs, segs, ops, lab_xy, lab_w, near)
    r_od = S.evaluate(coast, segs, [s for s, t_ in zip(segs, top) if t_], ops, lab_xy, lab_w, near)
    r_bay = S.evaluate(coast, segs, S.pick_by_score(segs, S.seg_scores(segs, variants["만입도만"]), a.top_frac), ops, lab_xy, lab_w, near)
    print(f"경로(라벨 포착): 전체 {r_full['time_h']:.1f} h·{r_full['n_sorties']}소티 | OpenDrift 상위 30 % {r_od['time_h']:.1f} h·{r_od['n_sorties']}소티·{r_od['captured_frac'] * 100:.0f} % | 만입도 상위 30 % {r_bay['time_h']:.1f} h·{r_bay['captured_frac'] * 100:.0f} %")

    # ---- 5. 저장
    ll = P.to_lonlat(np.array([[s.cx, s.cy] for s in segs]), crs)
    with open(out / "segments_mungap.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["seg_id", "lon", "lat", "length_m", "bearing_octant", "n_stranded_offshore", "n_stranded_han", "n_stranded", "stranding_density_1km", "top30_opendrift", "n_labels", "bay", "score_default"])
        for k, s in enumerate(segs):
            w.writerow([s.seg_id, round(ll[k][0], 6), round(ll[k][1], 6), int(s.length_m), names8[int(np.bincount(oct_[s.i0:s.i1]).argmax())], int(n_str["offshore"][k]), int(n_str["han_river"][k]), int(tot[k]),
                        round(dens[k], 2), int(top[k]), int(n_lab[k]), round(s.bay, 3), round(s.score, 3)])
    summ = {"period": [a.start, a.survey], "n": {"offshore": a.n_offshore, "han_river": a.n_han}, "windage": a.windage, "diffusivity": a.diffusivity, "dt_min": a.dt_min,
            "stranded_total": {k: int(sum(1 for r in rows if r["scenario"] == k and r["stranded"] and not r["stranded_t0"])) for k in SCEN}, "stranded_on_mungap": matched,
            "coast_km": round(coast.n * px / 1000, 2), "n_segments": len(segs), "segments_with_stranding": int((tot > 0).sum()), "n_labels": int(len(lab_xy)),
            "measured_wind_from_deg": {"jul": round(wind_jul), "jun_jul": round(wind_all)}, "wind_speed_mps": {"jul": round(wspd_jul, 1), "jun_jul": round(wspd_all, 1)},
            "residual_current": {"toward_deg": round(cur_dir), "speed_mps": round(cur_mag, 3), "tidal_speed_median_mps": round(cur_med, 2)},
            "curves": curves, "spearman": {"count_vs_labels": [round(float(rho), 3), round(float(pval), 4)], "density_vs_labels": [round(float(rho_s), 3), round(float(p_s), 4)]},
            "top_frac": a.top_frac, "labels_in_opendrift_top": round(float(hit), 3), "by_direction": by_dir,
            "routes": {"full": r_full, "opendrift_top": r_od, "bay_top": r_bay}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=P._jsonable), encoding="utf-8")
    lines = ["| 점수 | 상위 10 % 길이 | 20 % | 30 % | 50 % |", "|---|---|---|---|---|"]
    for nm in curves: lines.append(f"| {nm} | {curves[nm][10]:.0f} | {curves[nm][20]:.0f} | {curves[nm][30]:.0f} | {curves[nm][50]:.0f} |")
    lines += ["| 무작위 | 10 | 20 | 30 | 50 |", "", "| 해안 방위 | 해안 km | 라벨 | 라벨/km | OpenDrift 좌초 | 좌초/km |", "|---|---|---|---|---|---|"]
    for d in by_dir: lines.append(f"| {d['dir']} | {d['coast_km']} | {d['labels']} | {d['labels_per_km']} | {d['stranded']} | {d['stranded_per_km']} |")
    lines += ["", "| 전략 | 해안 km | 비행 h | 소티 | 라벨 포착 % |", "|---|---|---|---|---|"]
    for nm, r in (("전체 지그재그", r_full), ("OpenDrift 좌초 밀도 상위 30 %", r_od), ("만입도 상위 30 %", r_bay)):
        lines.append(f"| {nm} | {r['coast_km']} | {r['time_h']} | {r['n_sorties']} | {r['captured_frac'] * 100:.0f} |")
    (out / "비교표.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 6. 그림
    plt = setup_mpl()
    import rasterio
    with rasterio.open(ROOT / "input/s2_mungap/mungap_TCI_20260804.tif") as src:
        rgb = np.moveaxis(src.read(), 0, -1); H, W = rgb.shape[:2]
    ext = (transform.c, transform.c + W * transform.a, transform.f + H * transform.e, transform.f)
    fig, ax = plt.subplots(figsize=(10, 9.6), facecolor=SURF)
    ax.imshow(np.clip(rgb.astype(float) / 255 * 0.9 + 0.05, 0, 1), extent=ext, interpolation="bilinear")
    ax.scatter(coast.xy[:, 0], coast.xy[:, 1], s=1, color="white", alpha=0.5, linewidths=0)
    cxy = np.array([[s.cx, s.cy] for s in segs]); vmax = max(dens.max(), 1e-9)
    ax.scatter(cxy[:, 0], cxy[:, 1], c=dens, cmap="Blues", s=60, vmin=0, vmax=vmax, linewidths=0, alpha=0.9, label="200 m 구간 좌초 밀도 (진할수록 높음)")
    ax.scatter(cxy[top, 0], cxy[top, 1], s=160, facecolors="none", edgecolors=C2, linewidths=1.6, label=f"OpenDrift 상위 {a.top_frac * 100:.0f} % 구간")
    for k, col in (("offshore", "#ff3b30"), ("han_river", "#c77dff")):
        if len(stranded_pts[k]): ax.scatter(stranded_pts[k][:, 0], stranded_pts[k][:, 1], s=9, color=col, alpha=0.7, linewidths=0, label=f"좌초 입자 · {SCEN[k][0].split(' ')[0]} {matched[k]}개")
    ax.scatter(lab_xy[:, 0], lab_xy[:, 1], s=70, marker="*", color="#ffd60a", edgecolors="black", linewidths=0.6, label=f"기업 라벨 {len(lab_xy)}개 (2026-07-29)", zorder=5)
    pad = 800; ax.set_xlim(coast.xy[:, 0].min() - pad, coast.xy[:, 0].max() + pad); ax.set_ylim(coast.xy[:, 1].min() - pad, coast.xy[:, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9)
    ax.set_title(f"문갑도 — 해류(Copernicus 1/12°, 조석 포함) OpenDrift 좌초 vs 기업 라벨\n{a.start}~{a.survey} 방출·추적, 좌초 {int(tot.sum())}개 · 상위 30 % 구간에 라벨 {hit * 100:.0f} % · Spearman ρ={rho_s:.2f}", loc="left", fontsize=11, color=T1)
    fig.tight_layout(); fig.savefig(out / "50_mungap_stranding_vs_labels_map.png", dpi=130, facecolor=SURF); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURF); style(ax)
    ax.plot([0, 100], [0, 100], color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)), label="무작위")
    cols = [C2, C1, C3, C4, "#4a3aa7", "#e87ba4"]
    for (nm, vals), col in zip(curves.items(), cols):
        ax.plot(np.arange(101), vals, color=col, linewidth=2, label=nm)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.set_xlabel("점수 상위 해안 길이 비율 (%)", color=T2); ax.set_ylabel("기업 라벨 42개 중 포착 비율 (%)", color=T2)
    ax.set_title("문갑도 — 어떤 추정이 기업 라벨을 먼저 찾나", loc="left", fontsize=12.5, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=T1)
    fig.text(0.01, 0.01, f"OpenDrift: Open-Meteo 해류(조석 포함)+ERA5 바람, 풍압 {a.windage}, 확산 {a.diffusivity} m²/s, 좌초 즉시 고정. 라벨 42개(count 가중). 풍향·잔차류는 같은 자료에서 측정.", fontsize=8.3, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "51_mungap_capture_curves.png", dpi=170, facecolor=SURF); plt.close(fig)
    print("→", out)



if __name__ == "__main__":
    main()
