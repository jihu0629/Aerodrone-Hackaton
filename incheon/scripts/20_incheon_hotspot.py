"""인천·강화 집적 예상 해안(조석 잔차류 모델 지도) → 위성영상 정합 → 핫스팟 비행 vs 전체 지그재그 이점.

입력
  incheon/outputs/georef.json, mask_*_lonlat.npy    (이미지 축 눈금으로 만든 경위도 변환 + 빨간 선·분홍 수렴역 픽셀)
  input/s2_incheon/incheon_{B03,B08,B04,TCI,SCL}_<scene>.tif   (Sentinel-2 20 m, tools/fetch_s2_aws.py --res 20)
실행
  python incheon/scripts/20_incheon_hotspot.py --s2-dir input/s2_incheon --scenes 20260501,20260531,20260615,20260916 --tci-scene 20260531 \
      --wind-from 225 --out incheon/outputs
출력 (incheon/outputs/)
  30_정합확인_지도.png, 31_집적도_vs_형상점수_곡선.png, 32_시간대비포착_인천.png, 33_지도_전체_vs_집적도핫스팟.jpg
  해안별_비행계획.md/.csv, 비교표.md/.csv, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from litter3d import priority as P  # noqa: E402
from litter3d import strategy as S  # noqa: E402

C1, C2, C3, C4, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9b9a95"
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BUDGET_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]


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


def load_site(d: Path, prefix, scenes, wind):
    greens, nirs, reds, scls = [], [], [], []
    for sc in scenes:
        g, transform, crs = P.read_band(d / f"{prefix}_B03_{sc}.tif"); n, _, _ = P.read_band(d / f"{prefix}_B08_{sc}.tif")
        greens.append(g); nirs.append(n)
        p = d / f"{prefix}_B04_{sc}.tif"
        if p.exists(): reds.append(P.read_band(p)[0])
        p = d / f"{prefix}_SCL_{sc}.tif"
        if p.exists(): scls.append(P.read_band(p)[0].astype(np.uint8))
    px = abs(transform.a)
    ndwi = P.ndwi_max(greens, nirs, scls if len(scls) == len(scenes) else None)
    land = P.water_mask(ndwi, px_m=px)
    ndvi = None
    if reds:
        nir = np.max(np.stack(nirs), 0); red = np.min(np.stack(reds), 0)
        ndvi = (nir - red) / (nir + red + 1e-6)
    coast = P.coast_from_mask(land, transform, step_m=px)
    coast = P.add_features(coast, land, transform, wind_from_deg=wind, ndvi=ndvi, px_m=px)
    coast = P.score_coast(coast)
    return coast, land, transform, crs, px


def to_utm(lonlat, crs):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lonlat[:, 0], lonlat[:, 1]); return np.stack([x, y], 1)


def register(coast, red_xy, shifts_m=(-260, -195, -130, -65, 0, 65, 130, 195, 260)):
    """지도의 빨간 선 픽셀 → 위성 해안선까지 거리. 작은 평행이동으로 중앙값이 줄면 그만큼 보정 (정합 미세조정)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coast.xy)
    best = None
    for dx in shifts_m:
        for dy in shifts_m:
            d = tree.query(red_xy + np.array([dx, dy]))[0]
            med = float(np.median(d))
            if best is None or med < best[0]:
                best = (med, dx, dy, d)
    d0 = tree.query(red_xy)[0]
    return {"median_m_before": round(float(np.median(d0)), 1), "p90_m_before": round(float(np.percentile(d0, 90)), 1),
            "shift_m": [best[1], best[2]], "median_m_after": round(best[0], 1), "p90_m_after": round(float(np.percentile(best[3], 90)), 1),
            "within_500m_after": round(float((best[3] <= 500).mean()), 3)}, best[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2-dir", default="input/s2_incheon"); ap.add_argument("--prefix", default="incheon"); ap.add_argument("--scenes", required=True)
    ap.add_argument("--tci-scene"); ap.add_argument("--wind-from", type=float, default=225.0, help="여름 남서풍 225 (가정)")
    ap.add_argument("--prior-dist", type=float, default=500.0, help="빨간 선에서 이 거리(m) 안의 해안 표본 = 집적 예상 해안")
    ap.add_argument("--alt", type=float, default=20.0); ap.add_argument("--sea-m", type=float, default=20.0); ap.add_argument("--inland-m", type=float, default=100.0)
    ap.add_argument("--random-draws", type=int, default=5); ap.add_argument("--out", default="incheon/outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes.split(",")
    coast, land, transform, crs, px = load_site(Path(a.s2_dir), a.prefix, scenes, a.wind_from)
    L_km = coast.n * px / 1000
    rings = P.ring_table(coast)
    print(f"해안 표본점 {coast.n} ({L_km:.1f} km), 둘레 {len(rings)}개, 육지 {land.sum() * px * px / 1e6:.0f} km²")

    # ---- 지도 추출물 → UTM
    geo = json.loads((out / "georef.json").read_text(encoding="utf-8"))
    red_ll = np.load(out / "mask_red_core_lonlat.npy"); pink_ll = np.load(out / "mask_pink_lonlat.npy"); dash_ll = np.load(out / "mask_red_all_lonlat.npy")
    red_xy = to_utm(red_ll, crs); pink_xy = to_utm(pink_ll, crs); dash_xy = to_utm(dash_ll, crs)
    reg, red_d = register(coast, red_xy)
    print("정합:", json.dumps(reg, ensure_ascii=False))
    shift = np.array(reg["shift_m"]); red_xy = red_xy + shift; pink_xy = pink_xy + shift; dash_xy = dash_xy + shift
    from scipy.spatial import cKDTree
    d_red = cKDTree(red_xy).query(coast.xy)[0]
    d_dash = cKDTree(dash_xy).query(coast.xy)[0]
    n_pink = np.array([len(v) for v in cKDTree(pink_xy).query_ball_point(coast.xy, 1000.0)])
    coast.feats["red_d"] = d_red; coast.feats["prior"] = (d_red <= a.prior_dist).astype(float); coast.feats["conv"] = n_pink.astype(float)
    prior = coast.feats["prior"] > 0
    red_km = prior.sum() * px / 1000
    print(f"집적 예상 해안(빨간 선 {a.prior_dist:.0f} m 안): {red_km:.1f} km = 전체의 {prior.mean() * 100:.1f} %  (지도 범례 8개 합 112.4 km)")

    # 순위 해안 라벨: 가장 가까운 번호 원
    ranked = geo["ranked_coasts"]; circ_xy = to_utm(np.array([[r["lon"], r["lat"]] for r in ranked]), crs) + shift
    lab = cKDTree(circ_xy).query(coast.xy)[1]           # 0..7 → rank 1..8
    coast.feats["rank"] = np.where(prior, lab + 1, 0)
    per = []
    for i, r in enumerate(ranked):
        m = prior & (lab == i)
        per.append({"rank": r["rank"], "name": r["name"], "km_map": r["km"], "km_s2": round(m.sum() * px / 1000, 1),
                    "score_pct": round(float(np.mean([(coast.score < s).mean() for s in coast.score[m][::max(1, m.sum() // 200)]]) * 100), 0) if m.sum() else None,
                    "bay_mean": round(float(coast.feats["bay"][m].mean()), 2) if m.sum() else None,
                    "conv_mean": round(float(coast.feats["conv"][m].mean()), 1) if m.sum() else None})
    for r in per: print("  ", r)

    # ---- 위성 형상 점수가 집적 예상 해안을 얼마나 먼저 고르나 (밀도 = 빨간 선 해안 표본, w=1)
    dens_xy = coast.xy[prior]; w = np.ones(prior.sum())
    near = np.nonzero(prior)[0]
    variants = {"기본 점수 (만입도+노출+띠폭)": coast.score, "만입도만": S.score_variant(coast, "bay"),
                f"노출만 (풍향 {a.wind_from:.0f}°)": S.score_variant(coast, "exposure", a.wind_from),
                "수렴역 근접(분홍 음영 1 km 안 픽셀 수)": ndi.uniform_filter1d(coast.feats["conv"], 5, mode="nearest")}
    curves = {}
    for name, sc in variants.items():
        xs, ys = P.capture_curve(coast, sc, dens_xy, w)
        curves[name] = [round(float(v) * 100, 1) for v in ys]
    print("집적 예상 해안 포착(길이 상위 10/20/30/50 %):", {k: [v[10], v[20], v[30], v[50]] for k, v in curves.items()})

    # ---- 전략 비교 (밀도 = 집적 예상 해안, 비용 = 같은 카메라·띠 폭)
    ops = S.Ops(alt_m=a.alt, sea_m=a.sea_m, inland_m=a.inland_m); desc = ops.describe()
    segs = P.make_segments(coast, 200.0)
    seg_prior = np.array([coast.feats["prior"][s.i0:s.i1].mean() for s in segs])
    seg_rank = np.array([int(np.bincount(coast.feats["rank"][s.i0:s.i1].astype(int)).argmax()) for s in segs])
    hot_segs = [s for s, p_ in zip(segs, seg_prior) if p_ >= 0.5]
    hot_budget = sum(s.length_m for s in hot_segs) / sum(s.length_m for s in segs)
    depot_xy = circ_xy[4]  # 인천 북항 근처를 집결지로 (지리 순서 기준선용)
    rng = np.random.default_rng(0)
    rows = []
    r = S.evaluate(coast, segs, hot_segs, ops, dens_xy, w, near); rows.append({"strategy": "prior_hotspot", "budget": round(hot_budget, 3), **r})
    r = S.evaluate(coast, segs, hot_segs, ops, dens_xy, w, near, gap_fill_m=200.0); rows.append({"strategy": "prior_hotspot_gapfill", "budget": round(hot_budget, 3), **r})
    sc_def = S.seg_scores(segs, coast.score); sc_bay = S.seg_scores(segs, variants["만입도만"])
    for b in sorted(set(BUDGET_GRID + [round(hot_budget, 3)])):
        for nm, sc in (("shape_default", sc_def), ("shape_bay", sc_bay)):
            r = S.evaluate(coast, segs, S.pick_by_score(segs, sc, b), ops, dens_xy, w, near); rows.append({"strategy": nm, "budget": b, **r})
        if b < 1:
            both = []
            for d_ in (1, -1):
                rr = S.evaluate(coast, segs, S.pick_geo(coast, segs, b, depot_xy, direction=d_), ops, dens_xy, w, near); both.append(rr)
            mean = {k: round(float(np.mean([x[k] for x in both])), 3) for k in both[0] if not k.startswith("_")}
            mean["captured_min"] = min(x["captured_frac"] for x in both); mean["captured_max"] = max(x["captured_frac"] for x in both)
            rows.append({"strategy": "geo", "budget": b, **mean})
            acc = []
            for _ in range(a.random_draws):
                perm = rng.permutation(len(segs)); ch, used, budget = [], 0.0, sum(s.length_m for s in segs) * b
                for k in perm:
                    if used + segs[k].length_m > budget: continue
                    ch.append(segs[k]); used += segs[k].length_m
                acc.append(S.evaluate(coast, segs, ch, ops, dens_xy, w, near))
            mean = {k: round(float(np.mean([x[k] for x in acc])), 3) for k in acc[0] if not k.startswith("_")}
            rows.append({"strategy": "random", "budget": b, **mean})
    full = next(r for r in rows if r["strategy"] == "shape_default" and r["budget"] == 1.0); rows.append({**full, "strategy": "zigzag_all"})
    for r in rows:
        print(f"  {r['strategy']:22s} {r['budget'] * 100:5.1f} %  {r['time_h']:7.1f} h  소티 {r['n_sorties']:>6}  프레임 {r['frames']:>8,}  포착 {r['captured_frac'] * 100:5.1f} %")

    # ---- 해안별 비행 계획 (집적 예상 상위 8개)
    per_flight = []
    for i, rk in enumerate(ranked):
        ss = [s for s, p_, rr in zip(segs, seg_prior, seg_rank) if p_ >= 0.5 and rr == rk["rank"]]
        if not ss:
            per_flight.append({"rank": rk["rank"], "name": rk["name"], "km_map": rk["km"], "n_seg": 0}); continue
        r = S.evaluate(coast, segs, ss, ops, dens_xy, w, near, gap_fill_m=200.0)
        per_flight.append({"rank": rk["rank"], "name": rk["name"], "km_map": rk["km"], "km_s2": r["coast_km"], "n_seg": len(ss), "sorties": r["n_sorties"],
                           "time_h": r["time_h"], "survey_km": r["survey_km"], "transit_km": r["transit_km"], "frames": r["frames"], "relocation_km": r["relocation_km"],
                           "note": "비행 제한 가능(접경·한강하구)" if rk["rank"] in (1, 2, 4, 6) else ""})
    hot = rows[0]; hotg = rows[1]
    headline = {"coast_km": round(L_km, 1), "prior_km": round(red_km, 1), "prior_frac": round(float(prior.mean()), 3), "registration": reg,
                "full": {k: full[k] for k in ("time_h", "n_sorties", "frames", "detect_cpu_h", "data_gb", "survey_km", "transit_km")},
                "prior_hotspot": {k: hot[k] for k in ("time_h", "n_sorties", "frames", "detect_cpu_h", "data_gb", "survey_km", "transit_km", "captured_frac", "coast_km")},
                "prior_hotspot_gapfill": {k: hotg[k] for k in ("time_h", "n_sorties", "frames", "captured_frac", "coast_km")},
                "time_ratio": round(hot["time_h"] / full["time_h"], 3), "frames_ratio": round(hot["frames"] / full["frames"], 3),
                "shape_capture_at_prior_budget": {r["strategy"]: r["captured_frac"] for r in rows if abs(r["budget"] - round(hot_budget, 3)) < 1e-6 and r["strategy"] in ("shape_default", "shape_bay", "geo", "random")},
                "curves": curves, "per_coast": per, "per_coast_flight": per_flight, "ops": desc, "scenes": scenes, "wind_from": a.wind_from, "prior_dist_m": a.prior_dist}
    (out / "summary.json").write_text(json.dumps({**headline, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}, ensure_ascii=False, indent=1, default=P._jsonable), encoding="utf-8")
    print("핵심:", json.dumps({k: headline[k] for k in ("coast_km", "prior_km", "prior_frac", "time_ratio", "frames_ratio", "shape_capture_at_prior_budget")}, ensure_ascii=False))

    # ---- 표
    name = {"prior_hotspot": "집적도 핫스팟(빨간 선 구간만)", "prior_hotspot_gapfill": "집적도 핫스팟+간격 메움", "shape_default": "위성 형상 점수(기본)", "shape_bay": "위성 형상(만입도만)",
            "geo": "지리 순서(양방향 평균)", "random": "무작위(평균)", "zigzag_all": "전체 지그재그"}
    keys = ["strategy", "budget", "coast_km", "n_runs", "n_sorties", "survey_km", "transit_km", "total_km", "time_h", "relocation_km", "frames", "detect_cpu_h", "data_gb", "captured_frac"]
    with open(out / "비교표.csv", "w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wri.writeheader(); [wri.writerow(r) for r in rows]
    lines = ["| 전략 | 해안 예산 | 조사 km | 이동 km | 비행 h | 소티 | 프레임 | 탐지 CPU h | 집적 예상 해안 포착 % |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["budget"], r["strategy"])):
        cap = f"{r['captured_frac'] * 100:.0f}" + (f" ({r['captured_min'] * 100:.0f}~{r['captured_max'] * 100:.0f})" if r.get("captured_min") is not None and r["captured_min"] < r["captured_max"] else "")
        lines.append(f"| {name[r['strategy']]} | {r['budget'] * 100:.1f} % | {r['survey_km']:.0f} | {r['transit_km']:.0f} | {r['time_h']:.1f} | {r['n_sorties']} | {r['frames']:,} | {r['detect_cpu_h']:.1f} | **{cap}** |")
    (out / "비교표.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pl = ["| 순위 | 해안 | 지도 km | 위성 해안 km | 소티 | 비행 h | 조사 km | 이동 km | 프레임 | 비고 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_flight:
        if r.get("n_seg", 0) == 0:
            pl.append(f"| {r['rank']} | {r['name']} | {r['km_map']} | — | — | — | — | — | — | 위성 해안선과 매칭 실패 |"); continue
        pl.append(f"| {r['rank']} | {r['name']} | {r['km_map']} | {r['km_s2']} | {r['sorties']} | {r['time_h']} | {r['survey_km']:.0f} | {r['transit_km']:.0f} | {r['frames']:,} | {r['note']} |")
    (out / "해안별_비행계획.md").write_text("\n".join(pl) + "\n", encoding="utf-8")
    with open(out / "해안별_비행계획.csv", "w", newline="", encoding="utf-8-sig") as f:
        ks = ["rank", "name", "km_map", "km_s2", "n_seg", "sorties", "time_h", "survey_km", "transit_km", "frames", "relocation_km", "note"]
        wri = csv.DictWriter(f, fieldnames=ks, extrasaction="ignore"); wri.writeheader(); [wri.writerow(r) for r in per_flight]

    # ---- 그림
    plt = setup_mpl()
    import rasterio
    tci = Path(a.s2_dir) / f"{a.prefix}_TCI_{a.tci_scene or scenes[0]}.tif"
    with rasterio.open(tci) as src:
        rgb = np.moveaxis(src.read(), 0, -1); H, W = rgb.shape[:2]
    ext = (transform.c, transform.c + W * transform.a, transform.f + H * transform.e, transform.f)
    # 30 정합 확인
    fig, ax = plt.subplots(figsize=(9.5, 12.5), facecolor=SURF)
    ax.imshow(np.clip(rgb.astype(float) / 255 * 0.85 + 0.08, 0, 1), extent=ext, interpolation="bilinear")
    ax.scatter(pink_xy[:, 0], pink_xy[:, 1], s=1.5, color="#c77dff", alpha=0.35, linewidths=0, label="조석 잔차류 수렴역 (분홍 음영)")
    ax.scatter(coast.xy[:, 0], coast.xy[:, 1], s=0.3, color="#ffffff", alpha=0.6, linewidths=0, label="위성 해안선 (Sentinel-2 NDWI 20 m)")
    ax.scatter(red_xy[:, 0], red_xy[:, 1], s=2, color="#ff3b30", alpha=0.55, linewidths=0, label="좌초 밀도 상위 15 % (지도 빨간 선, 정합 후)")
    for (x, y), rk in zip(circ_xy, ranked):
        ax.plot(x, y, "o", markersize=15, markerfacecolor="#e34948", markeredgecolor="white", markeredgewidth=1.5)
        ax.text(x, y, str(rk["rank"]), color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=9, markerscale=6)
    ax.set_title(f"집적 예상 지도 ↔ Sentinel-2 정합 — 빨간 선과 위성 해안선 거리 중앙값 {reg['median_m_after']:.0f} m (보정 전 {reg['median_m_before']:.0f} m, 평행이동 {reg['shift_m'][0]:+.0f}/{reg['shift_m'][1]:+.0f} m)\n"
                 f"위성 해안 {L_km:.0f} km 중 집적 예상 해안 {red_km:.0f} km ({prior.mean() * 100:.0f} %) · 배경 {a.tci_scene or scenes[0]}", loc="left", fontsize=10.5, color=T1)
    fig.tight_layout(); fig.savefig(out / "30_정합확인_지도.png", dpi=130, facecolor=SURF); plt.close(fig)
    # 31 곡선
    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=SURF); style(ax)
    ax.plot([0, 100], [0, 100], color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)), label="무작위 순서")
    for (nm, vals), col in zip(curves.items(), (C2, C1, C3, C4)):
        ax.plot(np.arange(101), vals, color=col, linewidth=2, label=nm)
    y30 = curves["만입도만"][30]; ax.plot([30], [y30], "o", markersize=8, markerfacecolor=C1, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"상위 30 % 길이 → {y30:.0f} % (만입도만)", (30, y30), xytext=(10, -14), textcoords="offset points", fontsize=10, color=T1)
    ax.axvline(hot_budget * 100, color=GRAY, linewidth=1, linestyle=(0, (2, 3))); ax.text(hot_budget * 100 + 1, 3, f"집적 예상 해안 길이 = 전체의 {hot_budget * 100:.0f} %", fontsize=9, color=T2)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.set_xlabel("위성 점수 상위 해안 길이 비율 (%)", color=T2); ax.set_ylabel("집적 예상 해안(빨간 선) 포착 비율 (%)", color=T2)
    ax.set_title("위성 형상 점수가 조석 모델의 집적 예상 해안을 얼마나 먼저 고르나 — 인천·강화", loc="left", fontsize=12.5, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=T1)
    fig.text(0.01, 0.01, f"밀도 = 지도의 빨간 선 {a.prior_dist:.0f} m 안 위성 해안 표본({red_km:.0f} km). 이 지도는 관측이 아니라 조석 잔차류 모델이므로 '두 방법이 얼마나 겹치나' 로 읽는다. 풍향 {a.wind_from:.0f}° 는 가정.", fontsize=8.3, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "31_집적도_vs_형상점수_곡선.png", dpi=170, facecolor=SURF); plt.close(fig)
    # 32 시간 대비 포착
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURF); style(ax)
    for key, col, nm, ls in (("shape_default", C1, "위성 형상 점수 상위부터", "-"), ("geo", C2, "지리 순서(연속, 양방향 평균)", "-"), ("random", GRAY, "무작위 구간", (0, (4, 3)))):
        rs = sorted([r for r in rows if r["strategy"] == key], key=lambda r: r["budget"])
        if key != "shape_default": rs = rs + [full]
        xs = [0] + [r["time_h"] for r in rs]; ys = [0] + [r["captured_frac"] * 100 for r in rs]
        ax.plot(xs, ys, color=col, linewidth=2, linestyle=ls, label=nm); ax.plot(xs[1:], ys[1:], "o", markersize=6, markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=2)
    ax.plot([hot["time_h"]], [hot["captured_frac"] * 100], "D", markersize=10, markerfacecolor=C3, markeredgecolor=SURF, markeredgewidth=2, label="집적도 핫스팟(빨간 선 구간만)")
    ax.annotate(f"집적도 핫스팟 {hot['time_h']:.0f} h → {hot['captured_frac'] * 100:.0f} %  (전체 {full['time_h']:.0f} h 의 {hot['time_h'] / full['time_h'] * 100:.0f} %)", (hot["time_h"], hot["captured_frac"] * 100), xytext=(10, -16), textcoords="offset points", fontsize=9.5, color=T1)
    ax.annotate(f"전체 지그재그 {full['time_h']:.0f} h · {full['n_sorties']}소티", (full["time_h"], 100), xytext=(-4, 9), textcoords="offset points", ha="right", fontsize=9.5, color=T1)
    ax.set_xlim(0, full["time_h"] * 1.06); ax.set_ylim(0, 106); ax.set_xlabel("비행시간 (시간, 이동 포함)", color=T2); ax.set_ylabel("집적 예상 해안 포착 비율 (%)", color=T2)
    ax.set_title("인천·강화 — 전체 지그재그 vs 집적도 핫스팟 vs 위성 점수 (같은 카메라·띠 폭·배터리)", loc="left", fontsize=12, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=T1)
    fig.text(0.01, 0.01, f"고도 {desc['alt_m']:.0f} m · GSD {desc['gsd_cm']} cm · 띠 −{desc['strip_m'][0]:.0f}~+{desc['strip_m'][1]:.0f} m → 패스 {desc['passes']} · 5 m/s · 소티 25분 · 이동식 이륙. 포착률의 분모는 조석 모델이 예상한 집적 해안 길이(관측 아님).", fontsize=8.3, color=T2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out / "32_시간대비포착_인천.png", dpi=170, facecolor=SURF); plt.close(fig)
    # 33 지도: 전체 vs 집적도 핫스팟 (선택 구간)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), facecolor=SURF)
    for ax, segsel, col, title in ((axes[0], segs, C1, f"전체 지그재그 — {L_km:.0f} km 해안 · {full['time_h']:.0f} h · {full['n_sorties']}소티"),
                                   (axes[1], hot_segs, C2, f"집적도 핫스팟 — {hot['coast_km']:.0f} km · {hot['time_h']:.0f} h · {hot['n_sorties']}소티 · 포착 {hot['captured_frac'] * 100:.0f} %")):
        ax.imshow(np.clip(rgb.astype(float) / 255 * 0.85 + 0.08, 0, 1), extent=ext, interpolation="bilinear")
        cx = np.array([[s.cx, s.cy] for s in segsel]); ax.scatter(cx[:, 0], cx[:, 1], s=3, color=col, linewidths=0, alpha=0.9)
        for (x, y), rk in zip(circ_xy, ranked):
            ax.text(x, y, str(rk["rank"]), color="white", ha="center", va="center", fontsize=8, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc="#e34948", ec="white", lw=1))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=10.5, color=T1)
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(out / "33_지도_전체_vs_집적도핫스팟.jpg", dpi=110, facecolor=SURF, pil_kwargs={"quality": 85}); plt.close(fig)
    print("→", out)


if __name__ == "__main__":
    main()
