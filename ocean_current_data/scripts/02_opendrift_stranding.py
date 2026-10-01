"""OpenDrift 로 경기만 표층 표류 → 해안 좌초 시뮬 → 위성 해안 200 m 구간별 좌초 밀도 → 집적 예상 지도·위성 형상 점수와 비교 → 경로 비용.

입력
  ocean_current_data/data/currents_<start>_<end>.nc, wind_<start>_<end>.nc  (01_fetch_open_meteo_currents.py)
  input/s2_incheon/incheon_*_<scene>.tif  (위성 해안선; incheon/scripts/20 과 같은 창)
  incheon/outputs/georef.json, mask_red_core_lonlat.npy  (조석 모델 집적 예상 지도, 비교용)
실행
  python ocean_current_data/scripts/02_opendrift_stranding.py --start 2026-06-01 --end 2026-09-30 --days 90 --release-days 60 --n 3000 \
      --out ocean_current_data/outputs
출력 (ocean_current_data/outputs/)
  stranded_particles.csv            좌초 입자 (시나리오, 방출 시각, 좌초 시각·위치, 표류 일수)
  segments_stranding_density.csv    위성 해안 200 m 구간별 좌초 수·밀도·순위 (경로 최적화 입력)
  40_stranding_density_map.png, 41_opendrift_vs_tidalmap_curve.png, summary.json, 비교표.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from litter3d import priority as P  # noqa: E402
from litter3d import strategy as S  # noqa: E402

C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


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


# ──────────────────────────────── 1. OpenDrift 실행 ────────────────────────────────

SCENARIOS = {
    # 이름: (설명, 방출 다각형 lon/lat 상자 또는 점 목록)
    "han_river": ("한강 하구 유출 (조강·염하 합류부 수역)", {"box": (126.56, 37.74, 126.66, 37.80)}),
    "offshore": ("경기만 외해 균일 (해상 기인·외부 유입)", {"box": (125.90, 37.10, 126.45, 37.62)}),
}


def run_opendrift(cur_nc, wind_nc, t0, days, release_days, n, out, seed=0, windage=0.02, diff=10.0, dt_min=15, out_h=3):
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic
    rng = np.random.default_rng(seed)
    results = {}
    for name, (desc, spec) in SCENARIOS.items():
        o = OceanDrift(loglevel=40)
        o.add_reader([reader_netCDF_CF_generic.Reader(str(cur_nc)), reader_netCDF_CF_generic.Reader(str(wind_nc))])
        o.set_config("general:coastline_action", "stranding")
        o.set_config("environment:fallback:horizontal_diffusivity", diff)   # m²/s (OpenDrift 1.14 설정키)
        o.set_config("drift:stokes_drift", False)          # 파랑 자료 없음
        o.set_config("drift:advection_scheme", "runge-kutta4")
        W, S_, E, N = spec["box"]
        lon = rng.uniform(W, E, n); lat = rng.uniform(S_, N, n)
        times = [t0 + timedelta(seconds=float(s)) for s in np.sort(rng.uniform(0, release_days * 86400, n))]
        o.seed_elements(lon=lon, lat=lat, time=times, wind_drift_factor=rng.uniform(0.5, 1.5, n) * windage)
        o.run(duration=timedelta(days=days), time_step=timedelta(minutes=dt_min), time_step_output=timedelta(hours=out_h), outfile=str(out / f"opendrift_{name}.nc"))
        ds = o.result                                        # xarray (trajectory, time); status flag_meanings 'active stranded ...'
        meanings = str(ds["status"].attrs.get("flag_meanings", "active stranded")).split()
        k_str = meanings.index("stranded") if "stranded" in meanings else 1
        st = ds["status"].values; lonh = ds["lon"].values; lath = ds["lat"].values
        valid = ~np.isnan(lonh)
        rows = []
        for i in range(st.shape[0]):
            vi = np.nonzero(valid[i])[0]
            if len(vi) == 0:
                continue
            i0, i1 = vi[0], vi[-1]
            final = int(st[i, i1]) if not np.isnan(st[i, i1]) else 0
            stranded = (final == k_str)
            rows.append({"scenario": name, "id": i, "release_lon": float(lonh[i, i0]), "release_lat": float(lath[i, i0]), "release_time": times[i].isoformat(),
                         "final_status": meanings[final] if final < len(meanings) else str(final), "stranded": int(stranded), "lon": float(lonh[i, i1]), "lat": float(lath[i, i1]),
                         "drift_days": round(len(vi) * out_h / 24, 2), "stranded_t0": int(stranded and len(vi) <= 1)})
        n_str = sum(r["stranded"] for r in rows); n_t0 = sum(r["stranded_t0"] for r in rows)
        print(f"  {name}: 입자 {len(rows)}, 좌초 {n_str} (방출 즉시 좌초 {n_t0} 제외 → {n_str - n_t0}), 표류 중 {sum(1 for r in rows if r['final_status'] == 'active')}")
        results[name] = {"desc": desc, "rows": rows, "n": len(rows), "n_stranded": n_str - n_t0,
                         "median_drift_days": float(np.median([r["drift_days"] for r in rows if r["stranded"] and not r["stranded_t0"]] or [np.nan]))}
    return results


# ──────────────────────────────── 2. 위성 해안 구간에 집계 ────────────────────────────────

def load_coast(args):
    spec = importlib.util.spec_from_file_location("inch", ROOT / "incheon" / "scripts" / "20_incheon_hotspot.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    coast, land, transform, crs, px = m.load_site(ROOT / args.s2_dir, args.prefix, args.scenes.split(","), args.wind_from)
    return m, coast, land, transform, crs, px


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--days", type=int, default=90); ap.add_argument("--release-days", type=int, default=60); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--windage", type=float, default=0.02); ap.add_argument("--diffusivity", type=float, default=10.0)
    ap.add_argument("--data", default="ocean_current_data/data"); ap.add_argument("--out", default="ocean_current_data/outputs")
    ap.add_argument("--s2-dir", default="input/s2_incheon"); ap.add_argument("--prefix", default="incheon"); ap.add_argument("--scenes", default="20260501,20260531,20260615,20260916")
    ap.add_argument("--tci-scene", default="20260531"); ap.add_argument("--wind-from", type=float, default=225.0)
    ap.add_argument("--match-m", type=float, default=1500.0, help="좌초 위치(GSHHG 해안)에서 위성 해안 표본까지 허용 거리")
    ap.add_argument("--top-frac", type=float, default=0.15); ap.add_argument("--skip-run", action="store_true", help="기존 stranded_particles.csv 재사용")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cur_nc = ROOT / a.data / f"currents_{a.start}_{a.end}.nc"; wind_nc = ROOT / a.data / f"wind_{a.start}_{a.end}.nc"
    t0 = datetime.fromisoformat(a.start).replace(tzinfo=None)

    if a.skip_run and (out / "stranded_particles.csv").exists():
        rows = list(csv.DictReader(open(out / "stranded_particles.csv", encoding="utf-8-sig")))
        for r in rows:
            for k in ("release_lon", "release_lat", "lon", "lat", "drift_days"): r[k] = float(r[k])
            for k in ("stranded", "stranded_t0", "id"): r[k] = int(r[k])
        results = {nm: {"desc": SCENARIOS[nm][0], "rows": [r for r in rows if r["scenario"] == nm]} for nm in SCENARIOS}
        for nm, v in results.items():
            v["n"] = len(v["rows"]); v["n_stranded"] = sum(r["stranded"] and not r["stranded_t0"] for r in v["rows"])
            v["median_drift_days"] = float(np.median([r["drift_days"] for r in v["rows"] if r["stranded"] and not r["stranded_t0"]] or [np.nan]))
    else:
        print(f"OpenDrift 실행: {a.days}일, 방출 {a.release_days}일간 {a.n}개/시나리오, 풍압 {a.windage}, 확산 {a.diffusivity} m²/s")
        results = run_opendrift(cur_nc, wind_nc, t0, a.days, a.release_days, a.n, out, windage=a.windage, diff=a.diffusivity)
        with open(out / "stranded_particles.csv", "w", newline="", encoding="utf-8-sig") as f:
            keys = ["scenario", "id", "release_lon", "release_lat", "release_time", "final_status", "stranded", "stranded_t0", "lon", "lat", "drift_days"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for v in results.values():
                for r in v["rows"]: w.writerow(r)

    # ---- 위성 해안 구간 집계
    m, coast, land, transform, crs, px = load_coast(a)
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tree = cKDTree(coast.xy)
    segs = P.make_segments(coast, 200.0)
    seg_of = np.zeros(coast.n, int)
    for s in segs: seg_of[s.i0:s.i1] = s.seg_id
    counts = {nm: np.zeros(len(segs)) for nm in results}; unmatched = {}
    for nm, v in results.items():
        pts = [r for r in v["rows"] if r["stranded"] and not r["stranded_t0"]]
        if not pts: unmatched[nm] = 0; continue
        x, y = tr.transform([r["lon"] for r in pts], [r["lat"] for r in pts]); xy = np.stack([x, y], 1)
        d, idx = tree.query(xy); ok = d <= a.match_m
        unmatched[nm] = int((~ok).sum())
        np.add.at(counts[nm], seg_of[idx[ok]], 1)
    seg_len_km = np.array([s.length_m for s in segs]) / 1000
    total = sum(counts.values())
    dens = total / seg_len_km                                   # 좌초 입자 / km
    # 1 km 이동평균 (구간 5개) 후 점수
    from scipy import ndimage as ndi
    score_seg = np.zeros(len(segs)); rid = np.array([s.ring_id for s in segs])
    for r_ in np.unique(rid):
        mk = rid == r_; score_seg[mk] = ndi.uniform_filter1d(dens[mk], size=5, mode="wrap")
    order = np.argsort(-score_seg); rank = np.empty(len(segs), int); rank[order] = np.arange(1, len(segs) + 1)
    cum = np.cumsum(seg_len_km[order]) / seg_len_km.sum()
    top = np.zeros(len(segs), bool); top[order[cum <= a.top_frac]] = True
    print(f"좌초 입자 합 {int(total.sum())} (해안 매칭 실패 {unmatched}), 좌초가 있는 구간 {(total > 0).sum()}/{len(segs)}, 상위 {a.top_frac * 100:.0f} % 길이 = {seg_len_km[top].sum():.1f} km, 그 안 좌초 {total[top].sum() / max(total.sum(), 1) * 100:.0f} %")

    # 점수를 해안 표본점으로 (strategy/capture_curve 용)
    score_pt = np.zeros(coast.n)
    for s, sc in zip(segs, score_seg): score_pt[s.i0:s.i1] = sc
    coast.feats["stranding"] = score_pt

    # ---- 비교 1: 조석 모델 집적 예상 지도(빨간 선) 와의 겹침
    geo = json.loads((ROOT / "incheon/outputs/georef.json").read_text(encoding="utf-8"))
    red_ll = np.load(ROOT / "incheon/outputs/mask_red_core_lonlat.npy")
    rx, ry = tr.transform(red_ll[:, 0], red_ll[:, 1]); red_xy = np.stack([rx, ry], 1) + np.array([-130.0, 0.0])   # incheon 정합 보정과 동일
    d_red = cKDTree(red_xy).query(coast.xy)[0]; prior = d_red <= 500.0
    seg_prior = np.array([prior[s.i0:s.i1].mean() for s in segs]) >= 0.5
    # (a) OpenDrift 상위 15 % 구간 중 조석 지도 핫스팟인 비율, (b) 조석 지도 핫스팟 길이 중 OpenDrift 상위 15 % 에 든 비율
    overlap_a = seg_len_km[top & seg_prior].sum() / max(seg_len_km[top].sum(), 1e-9)
    overlap_b = seg_len_km[top & seg_prior].sum() / max(seg_len_km[seg_prior].sum(), 1e-9)
    # 곡선: OpenDrift 점수 순으로 길이를 늘릴 때 조석 지도 핫스팟 포착 / 반대로 위성 형상 점수 순으로 OpenDrift 좌초 포착
    dens_red_xy = coast.xy[prior]; w_red = np.ones(prior.sum())
    xs, y_od_red = P.capture_curve(coast, score_pt, dens_red_xy, w_red)
    strand_xy = coast.xy[score_pt > 0]; w_st = score_pt[score_pt > 0]
    _, y_shape_od = P.capture_curve(coast, coast.score, strand_xy, w_st)
    _, y_bay_od = P.capture_curve(coast, S.score_variant(coast, "bay"), strand_xy, w_st)
    _, y_red_od = P.capture_curve(coast, prior.astype(float), strand_xy, w_st)
    print(f"겹침: OpenDrift 상위 {a.top_frac * 100:.0f} % 길이 중 조석 지도 핫스팟 {overlap_a * 100:.0f} %; 조석 지도 핫스팟 길이 중 OpenDrift 상위에 든 비율 {overlap_b * 100:.0f} %")
    print("OpenDrift 점수 → 조석 지도 핫스팟 포착(10/20/30/50 %):", [round(float(y_od_red[k]) * 100) for k in (10, 20, 30, 50)])
    print("위성 형상 점수 → OpenDrift 좌초 포착:", [round(float(y_shape_od[k]) * 100) for k in (10, 20, 30, 50)], "| 만입도만:", [round(float(y_bay_od[k]) * 100) for k in (10, 20, 30, 50)], "| 조석 지도 → OpenDrift:", [round(float(y_red_od[k]) * 100) for k in (10, 20, 30, 50)])

    # ---- 비교 2: 경로 비용 (OpenDrift 핫스팟만 vs 전체)
    ops = S.Ops(alt_m=20.0); near = P.nearest_coast_index(coast, strand_xy)
    hot_segs = [s for s, t_ in zip(segs, top) if t_]
    r_hot = S.evaluate(coast, segs, hot_segs, ops, strand_xy, w_st, near)
    r_full = S.evaluate(coast, segs, segs, ops, strand_xy, w_st, near)
    sc_shape = S.seg_scores(segs, coast.score)
    r_shape = S.evaluate(coast, segs, S.pick_by_score(segs, sc_shape, a.top_frac), ops, strand_xy, w_st, near)
    r_prior = S.evaluate(coast, segs, [s for s, p_ in zip(segs, seg_prior) if p_], ops, strand_xy, w_st, near)
    print(f"경로: 전체 {r_full['time_h']:.0f} h·{r_full['n_sorties']}소티 | OpenDrift 핫스팟 {seg_len_km[top].sum():.0f} km → {r_hot['time_h']:.0f} h·{r_hot['n_sorties']}소티·좌초 {r_hot['captured_frac'] * 100:.0f} % | "
          f"조석 지도 핫스팟 → {r_prior['time_h']:.0f} h·좌초 {r_prior['captured_frac'] * 100:.0f} % | 위성 형상 상위 {a.top_frac * 100:.0f} % → {r_shape['time_h']:.0f} h·좌초 {r_shape['captured_frac'] * 100:.0f} %")

    # ---- 저장
    ll = P.to_lonlat(np.array([[s.cx, s.cy] for s in segs]), crs)
    with open(out / "segments_stranding_density.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["seg_id", "lon", "lat", "length_m", "n_han_river", "n_offshore", "n_total", "per_km", "score_1km", "rank", "top", "tidal_map_hotspot", "shape_score"])
        for k, s in enumerate(segs):
            w.writerow([s.seg_id, round(ll[k][0], 6), round(ll[k][1], 6), int(s.length_m), int(counts["han_river"][k]), int(counts["offshore"][k]), int(total[k]),
                        round(dens[k], 2), round(score_seg[k], 2), int(rank[k]), int(top[k]), int(seg_prior[k]), round(s.score, 3)])
    summ = {"period": [a.start, a.end], "days": a.days, "release_days": a.release_days, "n_per_scenario": a.n, "windage": a.windage, "diffusivity": a.diffusivity,
            "scenarios": {nm: {k: v[k] for k in ("desc", "n", "n_stranded", "median_drift_days")} for nm, v in results.items()}, "unmatched": unmatched,
            "coast_km": round(coast.n * px / 1000, 1), "segments_with_stranding": int((total > 0).sum()), "n_segments": len(segs),
            "top_frac": a.top_frac, "top_km": round(float(seg_len_km[top].sum()), 1), "top_capture_of_stranding": round(float(total[top].sum() / max(total.sum(), 1)), 3),
            "overlap_with_tidal_map": {"opendrift_top_that_is_tidal_hotspot": round(float(overlap_a), 3), "tidal_hotspot_covered_by_opendrift_top": round(float(overlap_b), 3)},
            "curves": {"opendrift_to_tidalmap": [round(float(v) * 100, 1) for v in y_od_red], "shape_to_opendrift": [round(float(v) * 100, 1) for v in y_shape_od],
                       "bay_to_opendrift": [round(float(v) * 100, 1) for v in y_bay_od], "tidalmap_to_opendrift": [round(float(v) * 100, 1) for v in y_red_od]},
            "routes": {"full": r_full, "opendrift_hotspot": r_hot, "tidal_map_hotspot": r_prior, "shape_top": r_shape}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=P._jsonable), encoding="utf-8")
    lines = ["| 전략 | 해안 km | 비행 h | 소티 | 프레임 | OpenDrift 좌초 포착 % |", "|---|---|---|---|---|---|"]
    for nm, r in (("전체 지그재그", r_full), (f"OpenDrift 좌초 밀도 상위 {a.top_frac * 100:.0f} %", r_hot), ("조석 모델 지도 핫스팟(빨간 선)", r_prior), (f"위성 형상 점수 상위 {a.top_frac * 100:.0f} %", r_shape)):
        lines.append(f"| {nm} | {r['coast_km']:.0f} | {r['time_h']:.1f} | {r['n_sorties']} | {r['frames']:,} | **{r['captured_frac'] * 100:.0f}** |")
    (out / "비교표.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 그림
    plt = setup_mpl()
    import rasterio
    with rasterio.open(ROOT / a.s2_dir / f"{a.prefix}_TCI_{a.tci_scene}.tif") as src:
        rgb = np.moveaxis(src.read(), 0, -1); H, W = rgb.shape[:2]
    ext = (transform.c, transform.c + W * transform.a, transform.f + H * transform.e, transform.f)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 9.2), facecolor=SURF)
    for ax in axes:
        ax.imshow(np.clip(rgb.astype(float) / 255 * 0.85 + 0.08, 0, 1), extent=ext, interpolation="bilinear")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    ax = axes[0]
    for nm, col in (("offshore", C1), ("han_river", C2)):
        pts = [r for r in results[nm]["rows"] if r["stranded"] and not r["stranded_t0"]]
        if pts:
            x, y = tr.transform([r["lon"] for r in pts], [r["lat"] for r in pts]); ax.scatter(x, y, s=4, color=col, alpha=0.6, linewidths=0, label=f"{SCENARIOS[nm][0]} 좌초 {len(pts)}개")
    ax.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9, markerscale=4)
    ax.set_title(f"OpenDrift 좌초 위치 — {a.start}~ {a.days}일, 시나리오 2개 × {a.n}개", loc="left", fontsize=10.5, color=T1)
    ax = axes[1]
    cxy = np.array([[s.cx, s.cy] for s in segs]); vmax = np.percentile(score_seg[score_seg > 0], 95) if (score_seg > 0).any() else 1
    sc = ax.scatter(cxy[:, 0], cxy[:, 1], c=np.clip(score_seg, 0, vmax), cmap="Blues", s=6, linewidths=0, vmin=0, vmax=vmax)
    ax.scatter(cxy[top, 0], cxy[top, 1], s=14, facecolors="none", edgecolors=C2, linewidths=0.8, label=f"좌초 밀도 상위 {a.top_frac * 100:.0f} % ({seg_len_km[top].sum():.0f} km)")
    ax.scatter(red_xy[::3, 0], red_xy[::3, 1], s=1.5, color="#ff3b30", alpha=0.35, linewidths=0, label="조석 모델 지도 핫스팟 (빨간 선)")
    ax.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9, markerscale=3)
    ax.set_title(f"위성 해안 200 m 구간별 좌초 밀도 (1 km 평활) — 상위 {a.top_frac * 100:.0f} % 가 좌초의 {total[top].sum() / max(total.sum(), 1) * 100:.0f} %", loc="left", fontsize=10.5, color=T1)
    fig.suptitle("해류(Open-Meteo/Copernicus 1/12°, 조석 포함) → OpenDrift 표류·좌초 → 해안 구간 밀도 — 경기만", x=0.01, ha="left", fontsize=12.5, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "40_stranding_density_map.png", dpi=120, facecolor=SURF); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=SURF); style(ax)
    ax.plot([0, 100], [0, 100], color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)), label="무작위")
    ax.plot(np.arange(101), np.array(y_od_red) * 100, color=C1, linewidth=2, label="OpenDrift 좌초 밀도 순 → 조석 모델 지도 핫스팟 포착")
    ax.plot(np.arange(101), np.array(y_red_od) * 100, color=C2, linewidth=2, label="조석 모델 지도 순 → OpenDrift 좌초 포착")
    ax.plot(np.arange(101), np.array(y_shape_od) * 100, color=C3, linewidth=2, label="위성 형상 점수 순 → OpenDrift 좌초 포착")
    ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.set_xlabel("점수 상위 해안 길이 비율 (%)", color=T2); ax.set_ylabel("포착 비율 (%)", color=T2)
    ax.set_title("세 가지 '어디에 쌓이나' 추정이 서로 얼마나 겹치나 — 경기만", loc="left", fontsize=12.5, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=T1)
    fig.text(0.01, 0.01, f"OpenDrift: Open-Meteo 해류(Copernicus 1/12°, 조석 포함)+ERA5 바람, 풍압 {a.windage}, 확산 {a.diffusivity} m²/s, 좌초 즉시 고정, {a.days}일. 모두 모형이며 실측 검증 아님.", fontsize=8.3, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "41_opendrift_vs_tidalmap_curve.png", dpi=170, facecolor=SURF); plt.close(fig)
    print("→", out)


if __name__ == "__main__":
    main()
