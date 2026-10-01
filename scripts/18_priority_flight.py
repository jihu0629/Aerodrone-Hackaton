"""Sentinel-2 → 해안 형상 점수 → 예산 대비 포착률 곡선 → 우선 구간 코리더 비행 경로 (Litchi CSV · DJI KMZ · GeoJSON · 지도).

예 (니하우):
  python scripts/18_priority_flight.py --s2-dir s2 --scenes 20240106,20251226 --density-grid hawaii_niihau_ft_summary.json \
      --depot -160.2024,21.7869 --wind-from 60 --budget-frac 0.3 --out outputs/niihau
  (--s2-dir 안에 niihau_B03_<scene>.tif, niihau_B08_<scene>.tif, (선택) niihau_B04_<scene>.tif, niihau_TCI_<scene>.tif)

밀도 입력:  --density-grid  coastal 브랜치 hawaii.py 요약 json (grid: gx, gy, kg / grid_m)  — 탐지 기반 kg 격자
            --density-csv   x,y,w (래스터 CRS 미터)  또는 lon,lat,w
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d import priority as P  # noqa: E402

SERIES = [("#2a78d6", "만입도만"), ("#eb6834", "만입도 + 노출 + 띠 폭 (기본 점수)"), ("#1baf7a", "노출만 (풍향 {wind:.0f}°)")]


def load_density(a, crs):
    if a.density_grid:
        d = json.loads(Path(a.density_grid).read_text(encoding="utf-8"))
        g = d["grid"]; gm = float(d.get("grid_m", 10.0))
        xy = np.array([[c["gx"] * gm + gm / 2, c["gy"] * gm + gm / 2] for c in g], float)
        w = np.array([c["kg"] for c in g], float)
        return xy, w, f"탐지 기반 kg 격자 ({len(g)}셀, {d.get('n_items', '?')}개 탐지, 모델 {d.get('model', '?')})"
    if a.density_csv:
        rows = np.loadtxt(a.density_csv, delimiter=",", skiprows=1)
        xy, w = rows[:, :2], rows[:, 2] if rows.shape[1] > 2 else np.ones(len(rows))
        if np.abs(xy[:, 0]).max() <= 180:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x, y = tr.transform(xy[:, 0], xy[:, 1]); xy = np.stack([x, y], 1)
        return xy, w, f"CSV {len(rows)}점"
    return None, None, None


def curve_png(path, coast, variants, density_xy, w, label, fp_note, wind_from=60.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ["fonts/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", r"C:\Windows\Fonts\malgun.ttf"]:
        if Path(f).exists():
            fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    S, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=S); ax.set_facecolor(S)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=T2, length=0); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.plot([0, 100], [0, 100], color="#9b9a95", linewidth=1.5, linestyle=(0, (4, 3)), label="무작위 순서 (전체 커버리지)")
    out = {}
    for (col, name), sc in zip(SERIES, variants):
        name = name.format(wind=wind_from)
        xs, ys = P.capture_curve(coast, sc, density_xy, w)
        ax.plot(xs * 100, ys * 100, color=col, linewidth=2, label=name)
        out[name] = {f"{int(x * 100)}%": round(float(ys[int(x * 100)]) * 100, 1) for x in (.1, .2, .3, .5)}
        if "기본" in name or name.startswith("만입도만"):
            y30 = ys[30] * 100
            ax.plot([30], [y30], "o", markersize=8, markerfacecolor=col, markeredgecolor=S, markeredgewidth=2)
            ax.annotate(f"상위 30 % 길이 → {y30:.0f} %", (30, y30), xytext=(10, 8 if "기본" in name else -12), textcoords="offset points", fontsize=10, color=T1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.set_xlabel("우선순위 상위 해안 길이 비율 (%)", color=T2); ax.set_ylabel("포착한 쓰레기 무게 비율 (%)", color=T2)
    ax.set_title(f"위성 형상 점수의 예산 대비 포착률 — {label}", loc="left", fontsize=13, color=T1, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=T1)
    fig.text(0.01, 0.01, fp_note, fontsize=8.5, color=T2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, dpi=170, facecolor=S); plt.close(fig)
    return out


def map_png(path, coast, segs, sorties, depot_xy, rgb, transform, density_xy, w, label):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    S, T1, T2 = "#fcfcfb", "#0b0b0b", "#52514e"
    H, W = rgb.shape[:2]
    x0, y0 = transform * (0, 0); x1, y1 = transform * (W, H)
    fig, ax = plt.subplots(figsize=(9.5, 10.5), facecolor=S)
    ax.imshow(rgb, extent=(x0, x1, y1, y0), interpolation="nearest")
    segs_xy = [coast.xy[s.i0:s.i1] for s in segs]
    sc = np.array([s.score for s in segs])
    lc = LineCollection(segs_xy, cmap="Blues", linewidths=3.2, zorder=3)
    lc.set_array(sc); lc.set_clim(np.percentile(sc, 5), np.percentile(sc, 95)); ax.add_collection(lc)
    sel = [coast.xy[s.i0:s.i1] for s in segs if s.selected]
    ax.add_collection(LineCollection(sel, colors="#eb6834", linewidths=5.5, zorder=4, alpha=0.9))
    if density_xy is not None:
        ax.scatter(density_xy[:, 0], density_xy[:, 1], s=np.clip(w / w.max() * 60, 6, 60), c="#1baf7a",
                   edgecolors=S, linewidths=0.4, zorder=5, alpha=0.9)
    for so in sorties:
        p = np.array([[x, y] for x, y, _ in so.waypoints])
        ax.plot(p[:, 0], p[:, 1], color="#ffffff", linewidth=2.2, zorder=6, alpha=0.9)
        ax.plot(p[:, 0], p[:, 1], color="#4a3aa7", linewidth=1.2, zorder=7)
        ax.plot(p[0, 0], p[0, 1], marker="^", markersize=9, color="#0b0b0b", markeredgecolor=S, zorder=9)
        ax.annotate(f"소티 {so.sortie_id + 1}", p[0], xytext=(6, 6), textcoords="offset points", fontsize=9, color=T1,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffffcc", ec="none"), zorder=8)
    ax.plot(depot_xy[0], depot_xy[1], marker="*", markersize=16, color="#0b0b0b", markeredgecolor=S, zorder=9)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off()
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01); cb.set_label("집적 점수 (높을수록 쌓이기 쉬움)", color=T2); cb.ax.tick_params(colors=T2)
    handles = [Line2D([], [], color="#eb6834", linewidth=5, label="선택 구간 (예산 안)"),
               Line2D([], [], color="#4a3aa7", linewidth=1.5, label="비행 경로 (소티별)"),
               Line2D([], [], marker="o", color="#1baf7a", linestyle="", label="탐지 kg 격자 (검증용)"),
               Line2D([], [], marker="^", color="#0b0b0b", linestyle="", markersize=9, label="소티 이륙점"),
               Line2D([], [], marker="*", color="#0b0b0b", linestyle="", markersize=12, label="집결지(출발지)")]
    ax.legend(handles=handles, loc="lower left", frameon=True, facecolor="#ffffffdd", edgecolor="none", fontsize=9.5)
    ax.set_title(f"위성 형상 점수와 우선 구간 코리더 비행 — {label}", loc="left", fontsize=13, color=T1)
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=S); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2-dir", required=True); ap.add_argument("--scenes", required=True, help="쉼표 구분 장면 날짜")
    ap.add_argument("--prefix", default="niihau"); ap.add_argument("--label", default="니하우"); ap.add_argument("--tci-scene", help="지도 배경 장면")
    ap.add_argument("--density-grid"); ap.add_argument("--density-csv")
    ap.add_argument("--depot", required=True, help="lon,lat")
    ap.add_argument("--wind-from", type=float, default=60.0, help="바람이 불어오는 방위(도). 하와이 북동 무역풍 60")
    ap.add_argument("--bay-radius", type=float, default=150.0); ap.add_argument("--seg-len", type=float, default=200.0)
    ap.add_argument("--w-bay", type=float, default=1.0); ap.add_argument("--w-exp", type=float, default=0.5); ap.add_argument("--w-strip", type=float, default=0.5)
    ap.add_argument("--budget-frac", type=float, default=0.3); ap.add_argument("--budget-km", type=float)
    ap.add_argument("--alt", type=float, default=60.0); ap.add_argument("--swath", type=float, default=86.0)
    ap.add_argument("--passes", type=int, default=1); ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--battery-min", type=float, default=25.0); ap.add_argument("--out", default="outputs/priority")
    ap.add_argument("--launch", choices=["mobile", "fixed"], default="mobile", help="mobile: 소티마다 첫 구간 근처에서 이륙(차·배 이동), fixed: 출발지 고정")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = Path(a.s2_dir); scenes = a.scenes.split(",")
    greens, nirs, reds, scls = [], [], [], []
    for sc in scenes:
        g, transform, crs = P.read_band(d / f"{a.prefix}_B03_{sc}.tif"); n, _, _ = P.read_band(d / f"{a.prefix}_B08_{sc}.tif")
        greens.append(g); nirs.append(n)
        p = d / f"{a.prefix}_B04_{sc}.tif"
        if p.exists():
            reds.append(P.read_band(p)[0])
        p = d / f"{a.prefix}_SCL_{sc}.tif"
        if p.exists():
            scls.append(P.read_band(p)[0].astype(np.uint8))
    px = abs(transform.a)
    ndwi = P.ndwi_max(greens, nirs, scls if len(scls) == len(scenes) else None)
    print(f"장면 {len(scenes)}개, SCL 구름 마스크 {'사용' if len(scls) == len(scenes) else '없음'}")
    land = P.water_mask(ndwi, px_m=px)
    ndvi = None
    if reds:
        nir = np.max(np.stack(nirs), 0); red = np.min(np.stack(reds), 0)      # 구름(밝은 적색) 최소, NIR 최대 → 맑은 쪽
        ndvi = (nir - red) / (nir + red + 1e-6)
    coast = P.coast_from_mask(land, transform, step_m=px)
    coast = P.add_features(coast, land, transform, wind_from_deg=a.wind_from, bay_radius_m=a.bay_radius, ndvi=ndvi, px_m=px)
    coast = P.score_coast(coast, a.w_bay, a.w_exp, a.w_strip)
    print(f"육지 {land.sum() * px * px / 1e6:.1f} km², 해안 표본점 {coast.n} ({coast.n * px / 1000:.1f} km), 둘레 {len(np.unique(coast.ring_id))}개")
    for r in P.ring_table(coast):
        print("  둘레", r)

    from pyproj import Transformer
    lon, lat = map(float, a.depot.split(","))
    dx, dy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    depot_xy = np.array([dx, dy])

    density_xy, w, dlabel = load_density(a, crs)
    curves = None
    if density_xy is not None:
        near = P.nearest_coast_index(coast, density_xy)
        dist = np.linalg.norm(coast.xy[near] - density_xy, axis=1)
        print(f"밀도 {dlabel}; 해안까지 거리 중앙값 {np.median(dist):.0f} m, 95 % {np.percentile(dist, 95):.0f} m")
        bay_only = P._z(coast.feats["bay"]); exp_only = P._z(coast.feats["exposure"])
        k = max(int(100 // px), 1)
        from scipy import ndimage as ndi
        sm = lambda v: ndi.uniform_filter1d(v, size=k, mode="wrap")
        variants = [sm(bay_only), coast.score, sm(exp_only)]
        note = (f"만입도 = 1 - 반경 {a.bay_radius:.0f} m 물 비율 (Sentinel-2 NDWI 10 m, 장면 {', '.join(scenes)} 최댓값 합성). "
                f"노출 = 법선·풍향({a.wind_from:.0f}°) 코사인.\n검증 밀도: {dlabel}. 칩이 있는 해안만 포함되므로 '순위' 로만 해석.")
        curves = curve_png(out / "예산대비포착률.png", coast, variants, density_xy, w, a.label, note, a.wind_from)
        print("포착률(kg 기준):", json.dumps(curves, ensure_ascii=False))
        # 만입도·노출 효과 표
        q = np.percentile(coast.feats["bay"], [33, 66])
        lab = np.digitize(coast.feats["bay"][near], q)
        tot = np.array([(np.digitize(coast.feats["bay"], q) == i).sum() * px / 1000 for i in range(3)])
        print("만입도 3분위 (곶/직선/만) 해안 km:", tot.round(1), "| kg/km:", np.array([w[lab == i].sum() for i in range(3)]).round(0) / tot)
        print(f"노출: 해안 평균 {coast.feats['exposure'].mean():+.2f}, 탐지 kg 가중 평균 {(coast.feats['exposure'][near] * w).sum() / w.sum():+.2f}")

    segs = P.make_segments(coast, a.seg_len)
    sel = P.select_budget(segs, a.budget_frac, a.budget_km)
    fp = P.FlightParams(altitude_m=a.alt, swath_m=a.swath, passes=a.passes, speed_mps=a.speed, battery_min=a.battery_min)
    sorties, unreachable = P.plan_sorties(coast, sel, depot_xy, fp, launch=a.launch)
    summ = P.summary(coast, segs, sorties, fp, unreachable, a.launch)
    for so, d_ in zip(sorties, summ["sorties"]):
        d_["launch_lonlat"] = [round(float(v), 6) for v in P.to_lonlat(np.array([so.waypoints[0][:2]]), crs)[0]]
    if density_xy is not None:
        near = P.nearest_coast_index(coast, density_xy)
        seg_of = np.zeros(coast.n, int)
        for s in segs: seg_of[s.i0:s.i1] = s.seg_id
        sel_ids = {s.seg_id for s in sel}
        summ["captured_kg_frac"] = round(float(w[[seg_of[i] in sel_ids for i in near]].sum() / w.sum()), 3)
        summ["curves"] = curves
    P.export_geojson(out / "우선구간_경로.geojson", coast, segs, sorties, crs, depot_xy)
    for so in sorties:
        P.export_litchi_csv(out / f"sortie_{so.sortie_id + 1:02d}_litchi.csv", so, crs, fp)
        P.export_wpml_kmz(out / f"sortie_{so.sortie_id + 1:02d}_dji.kmz", so, crs, fp)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=P._jsonable), encoding="utf-8")
    # 구간 표
    import csv
    with open(out / "segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        wri = csv.writer(f); wri.writerow(["seg_id", "ring", "length_m", "score", "bay", "exposure", "strip_m", "selected", "order", "sortie", "lon", "lat"])
        ll = P.to_lonlat(np.array([[s.cx, s.cy] for s in segs]), crs)
        for s, (lo, la) in zip(segs, ll):
            wri.writerow([s.seg_id, s.ring_id, int(s.length_m), round(s.score, 3), round(s.bay, 3), round(s.exposure, 3),
                          None if np.isnan(s.strip_m) else int(s.strip_m), int(s.selected), s.order, s.sortie, round(lo, 6), round(la, 6)])
    tci = d / f"{a.prefix}_TCI_{a.tci_scene or scenes[-1]}.tif"
    if tci.exists():
        import rasterio
        with rasterio.open(tci) as src:
            rgb = np.moveaxis(src.read(), 0, -1)
        map_png(out / "우선구간_비행경로_지도.png", coast, segs, sorties, depot_xy, rgb, transform, density_xy, w, a.label)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("params", "curves", "sorties")}, ensure_ascii=False))
    for so in summ["sorties"]:
        print("  소티", so)
    print("→", out)


if __name__ == "__main__":
    main()
