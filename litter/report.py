"""
결과물 — 표(CSV/GeoJSON) · 그림 · 현장용 작업 카드(HTML).

작업 카드: 정거장마다 "무엇이 · 몇 kg(최대 몇 kg) · 몇 명 · 무슨 장비 ·
위치 반경" + **원본 프레임에서 잘라낸 주변 사진**. 수거자가 도착해서
바위·구조물 같은 표지물로 눈으로 바로 찾게 하는 게 목적.
"""
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np

try:  # 기존 파이프라인의 한글 폰트 헬퍼 (Windows 맑은 고딕 / macOS AppleGothic)
    from pipeline._compat import apply_korean_font
    apply_korean_font()
except Exception:
    pass

CREW_COLOR = {"1인": "#2a9d8f", "2인": "#e9c46a", "장비": "#e76f51"}
_ITEM_COLS = ["item_id", "stop_id", "cls", "cls_raw", "lat", "lon", "x", "y", "r95_m", "n_views",
              "area_m2", "length_m", "h_p90_m", "volume_m3", "valid_3d", "buried_suspect",
              "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "weight_method", "crew", "tools",
              "weight_kg", "image"]


def location_errors(items, geo):
    """수거 GPS 정답(gt_lat/gt_lon 또는 gt_x/gt_y)이 있는 물체로 위치 오차 비교."""
    rows = []
    for it in items:
        if it.get("gt_x") is not None:
            gx, gy = it["gt_x"], it["gt_y"]
        elif it.get("gt_lat") is not None:
            gx, gy = geo.to_xy(it["gt_lat"], it["gt_lon"])
        else:
            continue
        r = {"item_id": it["item_id"], "ours": np.hypot(it["x"] - gx, it["y"] - gy), "r95": it.get("r95_m")}
        if "naive_x" in it:
            r["drone_gps"] = np.hypot(it["naive_x"] - gx, it["naive_y"] - gy)
            r["frame_center"] = np.hypot(it["center_x"] - gx, it["center_y"] - gy)
        rows.append(r)
    if not rows:
        return None
    out = {"n": len(rows)}
    for k in ("drone_gps", "frame_center", "ours"):
        v = np.array([r[k] for r in rows if k in r])
        if len(v):
            out[k] = {"median_m": float(np.median(v)), "p90_m": float(np.percentile(v, 90)),
                      "max_m": float(v.max())}
    cov = [r["ours"] <= r["r95"] for r in rows if r.get("r95") is not None and np.isfinite(r["r95"])]
    if cov:
        out["r95_coverage_pct"] = float(np.mean(cov) * 100)
    out["_rows"] = rows
    return out


def write_tables(out, items, stops, summary):
    with open(out / "items.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_ITEM_COLS, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({k: (";".join(it[k]) if isinstance(it.get(k), list) else it.get(k)) for k in _ITEM_COLS})
    scol = ["visit_order", "stop_id", "lat", "lon", "n_items", "kg_est", "kg_hi", "max_item_kg_hi", "crew",
            "tools", "bulk_m3", "r95_m", "buried_suspect", "classes"]
    with open(out / "stops.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=scol, extrasaction="ignore")
        w.writeheader()
        for s in stops:
            w.writerow({k: (";".join(s[k]) if isinstance(s.get(k), list) else s.get(k)) for k in scol})

    def fc(feats):
        return {"type": "FeatureCollection", "features": feats}

    (out / "items.geojson").write_text(json.dumps(fc([
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [it["lon"], it["lat"]]},
         "properties": {k: it.get(k) for k in _ITEM_COLS if k not in ("lat", "lon")}} for it in items
    ]), ensure_ascii=False, default=str), encoding="utf-8")
    (out / "stops.geojson").write_text(json.dumps(fc([
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
         "properties": {k: v for k, v in s.items() if k not in ("lat", "lon", "item_ids")}} for s in stops
    ]), ensure_ascii=False, default=str), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")


def fig_map(out, items, stops, summary):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    x0 = min(it["x"] for it in items)
    y0 = min(it["y"] for it in items)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
    g = summary["grid_m"]
    vmax = max((c["kg"] for c in summary["grid"]), default=1)
    cmap = plt.get_cmap("YlOrRd")
    for c in summary["grid"]:
        a2.add_patch(Rectangle((c["gx"] * g - x0, c["gy"] * g - y0), g, g, color=cmap(c["kg"] / vmax)))
        if c["kg"] > vmax * .25:
            a2.text(c["gx"] * g - x0 + g / 2, c["gy"] * g - y0 + g / 2, f"{c['kg']:.0f}", ha="center",
                    va="center", fontsize=7)
    a2.set_title(f"{g:.0f} m 격자별 추정 무게 (kg)")
    for it in items:
        a1.scatter(it["x"] - x0, it["y"] - y0, s=8 + 6 * np.sqrt(it["weight_est_kg"]),
                   c=CREW_COLOR[it["crew"]], edgecolors="k", linewidths=.3, zorder=3)
        if it["crew"] != "1인" and np.isfinite(it.get("r95_m", np.nan)):
            a1.add_patch(Circle((it["x"] - x0, it["y"] - y0), it["r95_m"], fill=False, ls="--", lw=.6,
                                color=CREW_COLOR[it["crew"]]))
    d = np.array(summary["depot"]) - [x0, y0]
    path = np.vstack([d, [[s["x"] - x0, s["y"] - y0] for s in stops], d])
    a1.plot(path[:, 0], path[:, 1], "-", color="#264653", lw=.8, alpha=.6, zorder=2)
    a1.scatter(*d, marker="s", s=80, c="#264653", zorder=4)
    a1.annotate("집하장", d, xytext=(4, -10), textcoords="offset points", fontsize=8)
    for c, col in CREW_COLOR.items():
        a1.scatter([], [], c=col, label=f"{c} ({summary['crew_counts'][c]}개)")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title(f"물체 {summary['n_items']}개 · 추정 {summary['kg_est']:.0f} kg (상한 {summary['kg_hi']:.0f}) · "
                 f"정거장 {summary['n_stops']} · 경로 {summary['route_m']:.0f} m\n"
                 f"마대 {summary['bags']}개 (무게기준 {summary['bags_by_kg']}/부피기준 {summary['bags_by_vol']}) · "
                 f"1톤 트럭 {summary['trucks']}대", fontsize=10)
    for a in (a1, a2):
        a.set_aspect("equal")
        a.set_xlabel("동쪽 (m)")
        a.set_ylabel("북쪽 (m)")
        a.autoscale()
    fig.tight_layout()
    fig.savefig(out / "fig_map.png", dpi=130)
    plt.close(fig)


def fig_location(out, loc):
    import matplotlib.pyplot as plt

    names = [("drone_gps", "드론 GPS를\n위치로 기록"), ("frame_center", "화면 중앙"), ("ours", "픽셀 광선\n(제안)")]
    names = [(k, n) for k, n in names if k in loc]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    med = [loc[k]["median_m"] for k, _ in names]
    p90 = [loc[k]["p90_m"] for k, _ in names]
    xs = np.arange(len(names))
    ax.bar(xs - .2, med, .4, label="중앙값", color="#457b9d")
    ax.bar(xs + .2, p90, .4, label="90% 분위", color="#e63946")
    for i, (m, p) in enumerate(zip(med, p90)):
        ax.text(i - .2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + .2, p, f"{p:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs, [n for _, n in names])
    ax.set_ylabel("수거 지점까지 오차 (m)")
    ax.set_title(f"위치 오차 비교 (n={loc['n']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_location_error.png", dpi=130)
    plt.close(fig)


def fig_ablation(out, rows):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [r["method"] for r in rows]
    ys = np.arange(len(rows))[::-1]
    a1.barh(ys, [r["median_ape_pct"] for r in rows], color="#457b9d")
    for y, r in zip(ys, rows):
        a1.text(r["median_ape_pct"], y, f" {r['median_ape_pct']:.0f}%", va="center", fontsize=9)
    a1.set_yticks(ys, names)
    a1.set_xlabel("물체별 무게 오차 중앙값 (%)")
    a1.set_title("물체 하나하나의 무게를 얼마나 맞히나")
    hr = [r["heavy_recall_pct"] or 0 for r in rows]
    a2.barh(ys, hr, color="#e63946")
    for y, r, v in zip(ys, rows, hr):
        a2.text(v, y, f" {v:.0f}%  (오경보 {r['false_2p_pct']:.1f}%)", va="center", fontsize=9)
    a2.set_yticks(ys, [""] * len(rows))
    a2.set_xlim(0, 130)
    a2.set_xlabel(f"23 kg 이상 물체를 '2인 이상'으로 잡은 비율 (%) — n={rows[0]['heavy_n']}")
    a2.set_title("무거운 물체를 놓치지 않나")
    fig.tight_layout()
    fig.savefig(out / "fig_weight_ablation.png", dpi=130)
    plt.close(fig)


def _crop_b64(images_dir, it, ctx=3.0, size=260):
    import cv2

    if not images_dir:
        return None
    p = Path(images_dir) / it["image"]
    from .seg import _imread  # 한글 경로 대응
    img = _imread(p) if p.exists() else None
    if img is None:
        return None
    uv = np.asarray(it["polygon"], float)
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 40) * ctx / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, img.shape[1])), int(min(cy + half, img.shape[0]))
    crop = img[b:d, a:c].copy()
    cv2.polylines(crop, [np.round(uv - [a, b]).astype(np.int32)], True, (0, 0, 255), 2)
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


def work_cards(out, items, stops, summary, images_dir, max_stops=60):
    by_id = {it["item_id"]: it for it in items}
    cards = []
    for s in stops[:max_stops]:
        its = sorted((by_id[i] for i in s["item_ids"]), key=lambda i: -i["weight_hi_kg"])
        thumbs = "".join(
            f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{html.escape(i["cls_raw"])} · '
            f'{i["weight_est_kg"]:.1f} kg (최대 {i["weight_hi_kg"]:.1f})</figcaption></figure>'
            for i in its[:3] if (b := _crop_b64(images_dir, i)))
        rows = "".join(
            f"<tr><td>{html.escape(i['cls_raw'])}</td><td>{i['weight_est_kg']:.1f}</td><td>{i['weight_hi_kg']:.1f}</td>"
            f"<td class='c-{i['crew']}'>{i['crew']}</td><td>{'묻힘 의심' if i.get('buried_suspect') else ''}</td></tr>"
            for i in its)
        tools = ", ".join(s["tools"]) or "-"
        cards.append(f"""
<section class="card">
  <header><span class="ord">{s['visit_order']}</span> <b>{s['stop_id']}</b>
    <span class="badge c-{s['crew']}">{s['crew']}</span>
    {'<span class="badge warn">묻힘 의심 · 삽</span>' if s['buried_suspect'] else ''}</header>
  <p class="big">{s['kg_est']:.1f} kg <small>(최대 {s['kg_hi']:.1f} kg · {s['n_items']}개 · 부피 {s['bulk_m3'] * 1000:.0f} L)</small></p>
  <p>📍 {s['lat']:.6f}, {s['lon']:.6f} <small>반경 {s['r95_m']:.1f} m 안</small> · 장비: {tools}</p>
  <div class="thumbs">{thumbs}</div>
  <details><summary>물체 목록</summary><table><tr><th>종류</th><th>추정 kg</th><th>최대 kg</th><th>인력</th><th></th></tr>{rows}</table></details>
</section>""")
    doc = f"""<!doctype html><meta charset="utf-8"><title>수거 작업 카드</title>
<style>
body{{font-family:'Malgun Gothic',system-ui,sans-serif;margin:16px;background:#f6f6f3;color:#222}}
.sum{{background:#fff;padding:12px 16px;border-radius:8px;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.card{{background:#fff;border-radius:8px;padding:12px;box-shadow:0 1px 3px #0002}}
.ord{{display:inline-block;width:24px;height:24px;border-radius:50%;background:#264653;color:#fff;text-align:center;line-height:24px}}
.badge{{padding:2px 8px;border-radius:10px;font-size:12px;margin-left:4px}}
.c-1인{{background:#2a9d8f33}} .c-2인{{background:#e9c46a88}} .c-장비{{background:#e76f5188}} .warn{{background:#f4a26188}}
.big{{font-size:20px;margin:6px 0}} .thumbs{{display:flex;gap:6px;flex-wrap:wrap}}
figure{{margin:0;font-size:11px}} img{{max-width:100%;border-radius:4px}} table{{font-size:12px;border-collapse:collapse}}
td,th{{padding:2px 6px;border-bottom:1px solid #eee}}
</style>
<div class="sum"><h2>수거 작업 카드</h2>
<p>물체 {summary['n_items']}개 · 추정 {summary['kg_est']:.0f} kg (최대 {summary['kg_hi']:.0f} kg) · 정거장 {summary['n_stops']}곳 · 경로 {summary['route_m']:.0f} m</p>
<p>인력: 1인 {summary['crew_counts']['1인']} · 2인 {summary['crew_counts']['2인']} · 장비 {summary['crew_counts']['장비']} ·
마대 {summary['bags']}개 · 1톤 트럭 {summary['trucks']}대 (무게 {summary['trucks_by_kg']} / 부피 {summary['trucks_by_vol']})</p>
<p><small>인력 판단은 추정 무게가 아니라 <b>최대 무게(예측구간 상한)</b> 기준 — 23 kg 이상 2인, 50 kg 이상 장비.</small></p></div>
<div class="grid">{''.join(cards)}</div>"""
    (out / "work_cards.html").write_text(doc, encoding="utf-8")
