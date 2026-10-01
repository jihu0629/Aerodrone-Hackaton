"""핫스팟 코리더 비행 vs 전체 지그재그 커버리지 — 니하우(하와이 탐지 격자)로 정량 비교.

예:
  python scripts/19_strategy_compare.py --s2-dir input/s2_niihau --scenes 20250502,20251103,20250303 --tci-scene 20250502 \
      --density-grid input/s2_niihau/hawaii_niihau_ft_summary.json --depot="-160.2024,21.7869" --wind-from 60 \
      --alt 20 --sea-m 20 --inland-m 100 --out outputs/strategy_niihau

출력(outputs/<site>/): 비교표.csv · 비교표.md · summary.json ·
  22_시간대비포착_핫스팟_vs_지그재그.png · 23_한계_점수민감도_띠폭.png · 24_지도_전체커버리지_vs_핫스팟.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d import priority as P  # noqa: E402
from litter3d import strategy as S  # noqa: E402

# 팔레트: dataviz 참조 팔레트의 범주형 1~3번(파랑·주황·청록) + 기준선 회색. 표면/글자 토큰 동일.
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a95"
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def setup_mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ["fonts/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", r"C:\Windows\Fonts\malgun.ttf"]:
        if Path(f).exists():
            fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=T2, length=0); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def load_site(a):
    d = Path(a.s2_dir); scenes = a.scenes.split(",")
    greens, nirs, reds, scls = [], [], [], []
    for sc in scenes:
        g, transform, crs = P.read_band(d / f"{a.prefix}_B03_{sc}.tif"); n, _, _ = P.read_band(d / f"{a.prefix}_B08_{sc}.tif")
        greens.append(g); nirs.append(n)
        p = d / f"{a.prefix}_B04_{sc}.tif"
        if p.exists(): reds.append(P.read_band(p)[0])
        p = d / f"{a.prefix}_SCL_{sc}.tif"
        if p.exists(): scls.append(P.read_band(p)[0].astype(np.uint8))
    px = abs(transform.a)
    ndwi = P.ndwi_max(greens, nirs, scls if len(scls) == len(scenes) else None)
    land = P.water_mask(ndwi, px_m=px)
    ndvi = None
    if reds:
        nir = np.max(np.stack(nirs), 0); red = np.min(np.stack(reds), 0)
        ndvi = (nir - red) / (nir + red + 1e-6)
    coast = P.coast_from_mask(land, transform, step_m=px)
    coast = P.add_features(coast, land, transform, wind_from_deg=a.wind_from, ndvi=ndvi, px_m=px)
    coast = P.score_coast(coast)
    return coast, transform, crs, px


def load_density(a, crs):
    if a.density_grid:
        d = json.loads(Path(a.density_grid).read_text(encoding="utf-8"))
        g = d["grid"]; gm = float(d.get("grid_m", 10.0))
        xy = np.array([[c["gx"] * gm + gm / 2, c["gy"] * gm + gm / 2] for c in g], float)
        w = np.array([c["kg"] for c in g], float)
        return xy, w, f"탐지 kg 격자 {len(g)}셀 · {d.get('n_items', '?')}개 · 모델 {d.get('model', '?')}"
    rows = np.loadtxt(a.density_csv, delimiter=",", skiprows=1)
    xy, w = rows[:, :2], rows[:, 2] if rows.shape[1] > 2 else np.ones(len(rows))
    if np.abs(xy[:, 0]).max() <= 180:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tr.transform(xy[:, 0], xy[:, 1]); xy = np.stack([x, y], 1)
    return xy, w, f"CSV {len(rows)}점"


def run_all(coast, segs, ops, grid_xy, w, near, depot_xy, a, rng):
    """예산별 전략 표 + 변형(민감도)."""
    rows = []
    sc_default = S.seg_scores(segs, coast.score)
    for b in BUDGETS:
        ch = S.pick_by_score(segs, sc_default, b)
        r = S.evaluate(coast, segs, ch, ops, grid_xy, w, near, gap_fill_m=0.0, depot_xy=depot_xy)
        rows.append({"strategy": "hotspot", "budget": b, **r})
        if a.gap_fill_m > 0 and b < 1:
            r = S.evaluate(coast, segs, ch, ops, grid_xy, w, near, gap_fill_m=a.gap_fill_m, depot_xy=depot_xy)
            rows.append({"strategy": "hotspot_gapfill", "budget": b, **r})
        if b < 1:
            both = []
            for d, nm in ((1, "geo_fwd"), (-1, "geo_rev")):
                ch = S.pick_geo(coast, segs, b, depot_xy, direction=d)
                r = S.evaluate(coast, segs, ch, ops, grid_xy, w, near, depot_xy=depot_xy)
                rows.append({"strategy": nm, "budget": b, **r}); both.append(r)
            mean = {k: round(float(np.mean([x[k] for x in both])), 3) for k in both[0] if not k.startswith("_")}
            mean["frames"] = int(mean["frames"]); mean["n_sorties"] = round(mean["n_sorties"], 1)
            mean["captured_min"] = min(x["captured_frac"] for x in both); mean["captured_max"] = max(x["captured_frac"] for x in both)
            rows.append({"strategy": "geo", "budget": b, **mean})
            acc = []
            for _ in range(a.random_draws):
                perm = rng.permutation(len(segs))
                ch, used, budget = [], 0.0, sum(s.length_m for s in segs) * b
                for k in perm:
                    if used + segs[k].length_m > budget: continue
                    ch.append(segs[k]); used += segs[k].length_m
                acc.append(S.evaluate(coast, segs, ch, ops, grid_xy, w, near, depot_xy=depot_xy))
            mean = {k: round(float(np.mean([x[k] for x in acc])), 3) for k in acc[0] if not k.startswith("_")}
            mean["frames"] = int(mean["frames"]); mean["n_sorties"] = round(mean["n_sorties"], 1)
            rows.append({"strategy": "random", "budget": b, **mean})
    # 전체 커버리지 = hotspot budget 1.0 (모든 구간). 이름만 바꿔 한 줄 더.
    full = next(r for r in rows if r["strategy"] == "hotspot" and r["budget"] == 1.0)
    rows.append({**full, "strategy": "zigzag_all"})
    # 점수 변형 (예산 a.budget 고정)
    variants = []
    wrong = (a.wind_from + 180) % 360
    for name, kind in [("기본 점수 (만입도+노출+띠폭)", "default"), ("만입도만", "bay"), (f"노출만 (풍향 {a.wind_from:.0f}°)", "exposure"),
                       (f"풍향 반대 ({wrong:.0f}°)로 잘못 넣음", f"wind:{wrong:.0f}")]:
        sc = S.seg_scores(segs, S.score_variant(coast, kind, a.wind_from))
        ch = S.pick_by_score(segs, sc, a.budget)
        r = S.evaluate(coast, segs, ch, ops, grid_xy, w, near, depot_xy=depot_xy)
        variants.append({"name": name, **{k: r[k] for k in ("captured_frac", "time_h", "n_sorties", "total_km", "transit_share")}})
    geo = next(r for r in rows if r["strategy"] == "geo" and r["budget"] == a.budget)
    rnd = next(r for r in rows if r["strategy"] == "random" and r["budget"] == a.budget)
    variants.append({"name": "지리 순서 (출발지부터 연속, 양방향 평균)", **{k: geo[k] for k in ("captured_frac", "time_h", "n_sorties", "total_km", "transit_share")},
                     "captured_min": geo["captured_min"], "captured_max": geo["captured_max"]})
    variants.append({"name": f"무작위 구간 ({a.random_draws}회 평균)", **{k: rnd[k] for k in ("captured_frac", "time_h", "n_sorties", "total_km", "transit_share")}})
    return rows, variants


def fig_time_capture(path, rows, a, label, ops_desc):
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURF); style(ax)
    series = [("hotspot", C1, "핫스팟 (위성 점수 상위부터)"), ("geo", C2, "지리 순서 (출발지부터 연속, 양방향 평균)"), ("random", GRAY, "무작위 구간")]
    full = next(r for r in rows if r["strategy"] == "zigzag_all")
    for key, col, name in series:
        rs = sorted([r for r in rows if r["strategy"] == key], key=lambda r: r["budget"])
        if key != "hotspot":
            rs = rs + [full]
        xs = [0] + [r["time_h"] for r in rs]; ys = [0] + [r["captured_frac"] * 100 for r in rs]
        ls = (0, (4, 3)) if key == "random" else "-"
        if key == "geo":
            lo = [0] + [r.get("captured_min", r["captured_frac"]) * 100 for r in rs]; hi = [0] + [r.get("captured_max", r["captured_frac"]) * 100 for r in rs]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, linewidth=0, zorder=2, label="지리 순서 방향별 범위")
        ax.plot(xs, ys, color=col, linewidth=2, linestyle=ls, label=name, zorder=3)
        ax.plot(xs[1:], ys[1:], "o", markersize=7, markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=2, zorder=4)
        for r in rs:
            if r["budget"] == 0.3 and key in ("hotspot", "geo"):
                ax.annotate(f"해안 {r['budget'] * 100:.0f} % · {r['time_h']:.0f} h → {r['captured_frac'] * 100:.0f} %",
                            (r["time_h"], r["captured_frac"] * 100), xytext=(8, -14 if key == "geo" else 8), textcoords="offset points", fontsize=9.5, color=T1)
    ax.annotate(f"전체 지그재그 {full['time_h']:.0f} h → 100 %", (full["time_h"], 100), xytext=(-4, 9), textcoords="offset points", ha="right", fontsize=9.5, color=T1)
    ax.set_xlim(0, full["time_h"] * 1.06); ax.set_ylim(0, 106)
    ax.set_xlabel("비행시간 (시간, 구간 사이 이동 포함)", color=T2); ax.set_ylabel("포착한 탐지 무게 비율 (%)", color=T2)
    ax.set_title(f"같은 카메라·같은 띠 폭에서, 시간을 얼마나 쓰면 얼마나 잡나 — {label}", loc="left", fontsize=12.5, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=T1)
    note = (f"고도 {ops_desc['alt_m']:.0f} m · GSD {ops_desc['gsd_cm']} cm · 발자국 {ops_desc['footprint_m'][0]}×{ops_desc['footprint_m'][1]} m · "
            f"띠 바다 {ops_desc['strip_m'][0]:.0f} m ~ 안쪽 {ops_desc['strip_m'][1]:.0f} m → 패스 {ops_desc['passes']}개 (간격 {ops_desc['spacing_m']} m) · "
            f"조사 {ops_desc['speed_mps']:.0f} m/s · 이동 {ops_desc['transit_mps']:.0f} m/s · 소티 {ops_desc['battery_min']:.0f}분 (이동식 이륙)\n"
            "포착률 분모 = 칩이 있는 해안의 탐지 kg. 두 전략 모두 같은 탐지기를 쓰므로 모델 재현율은 비율에 영향 없음.")
    fig.text(0.01, 0.01, note, fontsize=8.3, color=T2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path, dpi=170, facecolor=SURF); plt.close(fig)


def fig_limits(path, variants, dist, w, ops, a, label):
    plt = setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2), facecolor=SURF, gridspec_kw={"width_ratios": [1.15, 1]})
    style(ax1); style(ax2); ax1.yaxis.grid(False); ax1.xaxis.grid(True, color=GRID, linewidth=0.8)
    names = [v["name"] for v in variants][::-1]; vals = [v["captured_frac"] * 100 for v in variants][::-1]
    y = np.arange(len(names))
    ax1.barh(y, vals, height=0.56, color=[C1 if "기본" in n else "#9ec5f4" for n in names], zorder=3)
    for yi, v in zip(y, vals):
        ax1.text(v + 1, yi, f"{v:.0f} %", va="center", fontsize=9.5, color=T1)
    ax1.axvline(a.budget * 100, color=GRAY, linewidth=1.2, linestyle=(0, (4, 3)))
    ax1.text(a.budget * 100 + 0.8, len(names) - 0.45, f"무작위 기대값 {a.budget * 100:.0f} %", fontsize=8.5, color=T2)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9.5, color=T1); ax1.set_xlim(0, 100)
    ax1.set_xlabel("포착한 탐지 무게 비율 (%)", color=T2)
    ax1.set_title(f"한계 1 · 점수가 틀리면 얼마나 떨어지나 (해안 {a.budget * 100:.0f} % 예산)", loc="left", fontsize=11.5, color=T1, pad=10)
    for s in ("left",): ax1.spines[s].set_visible(False)
    # 띠 폭
    order = np.argsort(dist); cum = np.cumsum(w[order]) / w.sum() * 100
    ax2.plot(dist[order], cum, color=C1, linewidth=2, zorder=3)
    for xv, txt in ((-ops.sea_m, f"바다쪽 {ops.sea_m:.0f} m"), (ops.inland_m, f"안쪽 {ops.inland_m:.0f} m")):
        ax2.axvline(xv, color=GRAY, linewidth=1.2, linestyle=(0, (4, 3)))
        ax2.text(xv + 2, 4, txt, fontsize=8.5, color=T2)
    inside = float(w[(dist >= -ops.sea_m) & (dist <= ops.inland_m)].sum() / w.sum() * 100)
    ax2.annotate(f"띠 안 kg {inside:.1f} % · 밖 {100 - inside:.1f} %", (ops.inland_m, inside), xytext=(-10, -22), textcoords="offset points", ha="right", fontsize=9.5, color=T1)
    ax2.plot([ops.inland_m], [inside], "o", markersize=7, markerfacecolor=C1, markeredgecolor=SURF, markeredgewidth=2, zorder=4)
    ax2.set_xlim(max(-60, dist.min() - 5), min(200, dist.max() + 5)); ax2.set_ylim(0, 102)
    ax2.set_xlabel("해안선 기준 거리 (m, + 안쪽)", color=T2); ax2.set_ylabel("누적 탐지 무게 비율 (%)", color=T2)
    ax2.set_title("한계 2 · 띠 폭 밖의 쓰레기 (두 전략 공통)", loc="left", fontsize=11.5, color=T1, pad=10)
    fig.suptitle(f"핫스팟 전략의 한계 — {label}", x=0.01, ha="left", fontsize=12.5, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=170, facecolor=SURF); plt.close(fig)


def fig_map(path, coast, full, hot, tci_path, transform, depot_xy, grid_xy, label, a):
    import rasterio
    plt = setup_mpl()
    with rasterio.open(tci_path) as src:
        rgb = np.moveaxis(src.read(), 0, -1)
    H, W = rgb.shape[:2]
    ext = (transform.c, transform.c + W * transform.a, transform.f + H * transform.e, transform.f)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), facecolor=SURF)
    for ax, res, col, title in ((axes[0], full, C1, f"전체 지그재그 커버리지 — {full['total_km']:.0f} km · {full['time_h']:.0f} h · {full['n_sorties']}소티 · 포착 100 %"),
                                (axes[1], hot, C2, f"핫스팟 (해안 {a.budget * 100:.0f} %) — {hot['total_km']:.0f} km · {hot['time_h']:.0f} h · {hot['n_sorties']}소티 · 포착 {hot['captured_frac'] * 100:.0f} %")):
        ax.imshow(np.clip(rgb.astype(float) / 255 * 0.8 + 0.1, 0, 1), extent=ext, interpolation="bilinear")
        ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=5, color="#1baf7a", alpha=0.5, linewidths=0, zorder=2)
        for so in res["_sorties"]:
            for st in so.stretches:
                ax.plot(st["path"][:, 0], st["path"][:, 1], color=col, linewidth=0.5, alpha=0.75, zorder=3)
            ax.plot(so.launch_xy[0], so.launch_xy[1], marker="^", markersize=5, color="#0b0b0b", markeredgecolor=SURF, markeredgewidth=0.6, linestyle="none", zorder=4)
        ax.plot(depot_xy[0], depot_xy[1], marker="*", markersize=12, color="#0b0b0b", markeredgecolor=SURF, linestyle="none", zorder=5)
        ax.set_title(title, loc="left", fontsize=10.5, color=T1); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=C1, lw=2, label="전체 지그재그 경로"), Line2D([], [], color=C2, lw=2, label="핫스팟 코리더 경로"),
                            Line2D([], [], marker="o", color="#1baf7a", lw=0, markersize=5, label="탐지 kg 격자 (검증용)"),
                            Line2D([], [], marker="^", color="#0b0b0b", lw=0, markersize=6, label="소티 이륙점 (이동식)")],
                   loc="lower left", frameon=True, facecolor=SURF, edgecolor=GRID, fontsize=8.5)
    fig.suptitle(f"같은 카메라·같은 띠 폭, 다른 전략 — {label}", x=0.01, ha="left", fontsize=12.5, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=150, facecolor=SURF); plt.close(fig)


def write_tables(out, rows, variants, ops_desc, strip, extra):
    keys = ["strategy", "budget", "coast_km", "n_runs", "n_sorties", "survey_km", "transit_km", "total_km", "transit_share", "time_h",
            "relocation_km", "frames", "detect_cpu_h", "data_gb", "captured_frac", "missed_frac", "kg_frac_per_h"]
    with open(out / "비교표.csv", "w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wri.writeheader()
        for r in rows: wri.writerow(r)
    name = {"hotspot": "핫스팟(점수순)", "hotspot_gapfill": "핫스팟+간격메움", "geo": "지리순서(양방향 평균)", "geo_fwd": "지리순서(정방향)", "geo_rev": "지리순서(역방향)", "random": "무작위(평균)", "zigzag_all": "전체 지그재그"}
    lines = ["| 전략 | 해안 예산 | 조사 km | 이동 km | 총 km | 비행 h | 소티 | 지상이동 km | 프레임 | 탐지 CPU h | 포착 kg % | 포착 %/h |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["budget"], r["strategy"])):
        lines.append(f"| {name[r['strategy']]} | {r['budget'] * 100:.0f} % | {r['survey_km']:.0f} | {r['transit_km']:.0f} | {r['total_km']:.0f} | {r['time_h']:.1f} | {r['n_sorties']} | "
                     f"{r['relocation_km']:.0f} | {r['frames']:,} | {r['detect_cpu_h']:.1f} | **{r['captured_frac'] * 100:.0f}** | {r['kg_frac_per_h'] * 100:.1f} |")
    lines += ["", "| 점수 변형 (예산 고정) | 포착 kg % | 비행 h | 소티 | 총 km | 이동 비율 |", "|---|---|---|---|---|---|"]
    for v in variants:
        lines.append(f"| {v['name']} | {v['captured_frac'] * 100:.0f} | {v['time_h']:.1f} | {v['n_sorties']} | {v['total_km']:.0f} | {v['transit_share'] * 100:.0f} % |")
    (out / "비교표.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"ops": ops_desc, "strip": strip, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
                                                  "variants": variants, **extra}, ensure_ascii=False, indent=1, default=P._jsonable), encoding="utf-8")


def wpct(v, w, p):
    """가중 백분위수."""
    o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(p / 100, cw, v[o]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2-dir", required=True); ap.add_argument("--scenes", required=True); ap.add_argument("--prefix", default="niihau")
    ap.add_argument("--label", default="니하우"); ap.add_argument("--tci-scene")
    ap.add_argument("--density-grid"); ap.add_argument("--density-csv"); ap.add_argument("--depot", required=True, help="lon,lat")
    ap.add_argument("--wind-from", type=float, default=60.0)
    ap.add_argument("--alt", type=float, default=20.0, help="조사 고도 m (기존 시뮬 20)"); ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--sea-m", type=float, default=20.0); ap.add_argument("--inland-m", type=float, default=100.0)
    ap.add_argument("--speed", type=float, default=5.0); ap.add_argument("--transit", type=float, default=10.0); ap.add_argument("--battery-min", type=float, default=25.0)
    ap.add_argument("--budget", type=float, default=0.3, help="대표 예산 (해안 길이 비율)"); ap.add_argument("--gap-fill-m", type=float, default=200.0)
    ap.add_argument("--random-draws", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/strategy")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    coast, transform, crs, px = load_site(a)
    ops = S.Ops(alt_m=a.alt, overlap_side=a.overlap, sea_m=a.sea_m, inland_m=a.inland_m, speed_mps=a.speed, transit_mps=a.transit, battery_min=a.battery_min)
    desc = ops.describe()
    print("운용 가정:", json.dumps(desc, ensure_ascii=False))
    print(f"해안 {coast.n * px / 1000:.1f} km, 둘레 {len(np.unique(coast.ring_id))}개")

    from pyproj import Transformer
    lon, lat = map(float, a.depot.split(","))
    depot_xy = np.array(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat))
    grid_xy, w, dlabel = load_density(a, crs)
    near = P.nearest_coast_index(coast, grid_xy)
    dist = S.inland_distance(coast, grid_xy, near)
    inside = float(w[(dist >= -a.sea_m) & (dist <= a.inland_m)].sum() / w.sum())
    strip = {"sea_m": a.sea_m, "inland_m": a.inland_m, "kg_inside_frac": round(inside, 4),
             "dist_pct": {str(p): round(wpct(dist, w, p), 1) for p in (5, 50, 95, 99)}}
    print(f"검증 밀도: {dlabel}; 해안 기준 거리(kg 가중) 5/50/95/99 % = {strip['dist_pct']}; 띠 안 kg {inside * 100:.1f} %")

    segs = P.make_segments(coast, 200.0)
    rows, variants = run_all(coast, segs, ops, grid_xy, w, near, depot_xy, a, rng)
    for r in rows:
        print(f"  {r['strategy']:16s} {r['budget'] * 100:4.0f} %  조사 {r['survey_km']:7.1f} km  이동 {r['transit_km']:6.1f} km  {r['time_h']:6.1f} h  소티 {r['n_sorties']:>5}  "
              f"프레임 {r['frames']:>7,}  포착 {r['captured_frac'] * 100:5.1f} %  ({r['kg_frac_per_h'] * 100:.2f} %/h)")
    for v in variants:
        print(f"  변형 {v['name']:28s} 포착 {v['captured_frac'] * 100:5.1f} %  {v['time_h']:.1f} h")

    # 대표 비교 (경로 보관) + 지도
    sc = S.seg_scores(segs, coast.score)
    hot = S.evaluate(coast, segs, S.pick_by_score(segs, sc, a.budget), ops, grid_xy, w, near, depot_xy=depot_xy, keep_paths=True)
    full = S.evaluate(coast, segs, segs, ops, grid_xy, w, near, depot_xy=depot_xy, keep_paths=True)
    geo = next(r for r in rows if r["strategy"] == "geo" and r["budget"] == a.budget)
    # 같은 포착률에 드는 시간 (보간) · 손익
    def hours_for(key, target):
        rs = sorted([r for r in rows if r["strategy"] == key], key=lambda r: r["time_h"])
        xs = [0] + [r["captured_frac"] for r in rs]; ys = [0] + [r["time_h"] for r in rs]
        return float(np.interp(target, xs, ys)) if target <= max(xs) else None
    headline = {"budget": a.budget, "hotspot": {k: hot[k] for k in hot if not k.startswith("_")}, "zigzag_all": {k: full[k] for k in full if not k.startswith("_")},
                "geo_same_budget": geo,
                "time_ratio_hot_vs_full": round(hot["time_h"] / full["time_h"], 3), "frames_ratio": round(hot["frames"] / full["frames"], 3),
                "capture_hot": hot["captured_frac"], "hours_geo_for_same_capture": hours_for("geo", hot["captured_frac"]),
                "hours_random_for_same_capture": hours_for("random", hot["captured_frac"]),
                "blend_full_every_n": {str(n): round((hot["time_h"] * (n - 1) + full["time_h"]) / n, 1) for n in (2, 3, 4, 6)}}
    print("핵심:", json.dumps(headline, ensure_ascii=False)[:900])
    write_tables(out, rows, variants, desc, strip, {"headline": headline, "density": dlabel})
    fig_time_capture(out / "22_시간대비포착_핫스팟_vs_지그재그.png", rows, a, a.label, desc)
    fig_limits(out / "23_한계_점수민감도_띠폭.png", variants, dist, w, ops, a, a.label)
    tci = Path(a.s2_dir) / f"{a.prefix}_TCI_{a.tci_scene or a.scenes.split(',')[0]}.tif"
    if tci.exists():
        fig_map(out / "24_지도_전체커버리지_vs_핫스팟.png", coast, full, hot, tci, transform, depot_xy, grid_xy, a.label, a)
    print("→", out)


if __name__ == "__main__":
    main()
