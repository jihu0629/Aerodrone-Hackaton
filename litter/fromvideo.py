"""
영상 → 3D 부피(objvol.json) → 재질별 무게 → 수거계획 → 작업카드까지 한 번에.

  python -m litter.fromvideo run --objvol runs/orbit/0007/objvol_dense/objvol.json \\
      --material "Plastic_Buoy_55:Styrofoam,Styrofoam_Buoy_22:Styrofoam" --out runs/plan/0007
  python -m litter.fromvideo compare --objvol runs/orbit/0007/objvol_dense/objvol.json \\
      --material "..." --labels data/company/MGD_labels.json --out runs/plan/compare

run 단계:
  1 objvol.json 읽기 (물체별 크기·부피·위경도, 없으면 objects.csv에서 위경도)
  2 재질 결정: --material 로 덮어쓰기 > 탐지 클래스명(alias 매핑)
  3 무게: weight.estimate (부피 × 겉보기밀도, 구간 ×/÷2) + 젖음 계수(--wet)
  4 마대 부피: 겉 부피 ÷ 압축비(config compress)
  5 plan.make_plan → report (items/stops csv·geojson, fig_map.png, work_cards.html)
  6 업체 방식(면적 × 계수) 무게도 같이 계산해 summary에 기록

compare: 발표용 비교 그림 2장
  (a) 상자 2개: 업체 방식 vs 3D 부피×밀도 vs 참값(정답 부피×같은 밀도)
  (b) 업체 라벨 42개: 기록된 weight_kg vs 같은 면적에 현실적 두께·밀도를 줬을 때의 범위
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from matplotlib.ticker import FuncFormatter

from . import plan, report, weight
from .config import COMPANY_CODE, COMPANY_CODE_NAME, COMPANY_KG_PER_M2, load_config, map_class

# 그림 색 (dataviz 기본 팔레트: 파랑·주황·청록)
C_COMPANY, C_OURS, C_TRUTH, C_RANGE = "#eb6834", "#2a78d6", "#1baf7a", "#86b6ef"


def _font():
    import matplotlib
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _parse_kv(s):
    """'a:b,c:d' → {a: b}"""
    if not s:
        return {}
    return {k.strip(): v.strip() for k, v in (t.split(":", 1) for t in s.split(",") if ":" in t)}


def _objects_latlon(path):
    """objects.csv → {'클래스_프레임수': (lat, lon)}"""
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[f"{r['cls']}_{r['seen_frames']}"] = (float(r["lat"]), float(r["lon"]))
    return out


def load_items(objvol_path, cfg, material=None, default_material=None, objects_csv=None,
               vol_key="volume_heightmap_L", r95_m=1.2):
    """objvol.json → weight/plan이 쓰는 item dict 목록."""
    from .geometry import Geo

    objvol_path = Path(objvol_path)
    recs = json.loads(objvol_path.read_text(encoding="utf-8"))
    ll = _objects_latlon(objects_csv)
    override = _parse_kv(material)
    items, skipped = [], []
    for r in recs:
        tag = r["tag"]
        if vol_key not in r:
            skipped.append(tag)
            continue
        lat = float(r["lat"]) if r.get("lat") not in (None, "") else ll.get(tag, (None, None))[0]
        lon = float(r["lon"]) if r.get("lon") not in (None, "") else ll.get(tag, (None, None))[1]
        if lat is None:
            skipped.append(tag)
            continue
        raw = override.get(tag) or default_material or tag.rsplit("_", 1)[0]
        cls = map_class(raw, cfg)
        L, W, H = r.get("length_m", 0.0), r.get("width_m", 0.0), r.get("height_m", 0.0)
        area = r.get("mask_area_m2") or (L * W)
        vol_m3 = r[vol_key] / 1000.0
        valid_3d = H >= 2 * cfg["weight"]["dsm_noise_m"] and vol_m3 > 0
        thumb = objvol_path.parent / f"{tag}_masks.jpg"
        it = {
            "item_id": tag, "cls_raw": tag, "material_raw": raw, "cls": cls,
            "lat": lat, "lon": lon, "n_views": r.get("views_used", 0), "r95_m": r95_m,
            "area_m2": float(area), "length_m": float(L), "width_m": float(W),
            "elongation": float(L / W) if W else 1.0, "h_p90_m": float(H),
            "volume_m3": float(vol_m3), "volume_key": vol_key, "valid_3d": bool(valid_3d),
            "volume_hull_m3": (r.get("volume_hull_L") or 0) / 1000.0,
            "volume_maskbox_m3": (r.get("volume_mask_box_L") or 0) / 1000.0,
            "truth_m3": (r["truth_L"] / 1000.0) if r.get("truth_L") else None,
            "thumb": str(thumb) if thumb.exists() else None, "image": "",
            "buried_suspect": False,
        }
        items.append(it)
    if not items:
        raise SystemExit(f"objvol.json에 쓸 수 있는 물체가 없음 (건너뜀: {skipped})")
    geo = Geo(lon=items[0]["lon"])
    for it in items:
        it["x"], it["y"] = geo.to_xy(it["lat"], it["lon"])
    return items, geo, skipped


def company_weight(it):
    """업체 방식: 박스 면적(m²) × 재질 계수(kg/m²)."""
    code = COMPANY_CODE.get(it["cls"], "PLA")
    return code, it["area_m2"] * COMPANY_KG_PER_M2[code]


def add_weights(items, cfg, wet=False):
    """weight.estimate + 젖음·압축·업체방식·참값 기준."""
    items, _ = weight.estimate(items, cfg)
    for it in items:
        p = cfg["classes"][it["cls"]]
        it["rho_app"] = p["rho_app"]
        it["wet_factor"] = p.get("wet", 1.0)
        it["compress"] = p.get("compress", 1.0)
        it["weight_dry_est_kg"] = it["weight_est_kg"]
        it["weight_wet_hi_kg"] = it["weight_hi_kg"] * it["wet_factor"]
        if wet:
            for k in ("weight_est_kg", "weight_lo_kg", "weight_hi_kg"):
                it[k] *= it["wet_factor"]
            it["weight_method"] += "+wet"
        it["bulk_raw_m3"] = it["volume_m3"]
        it["bulk_m3"] = it["volume_m3"] / it["compress"]  # 마대 적재 기준 (압축 후)
        it["company_code"], it["company_kg"] = company_weight(it)
        it["truth_kg"] = it["truth_m3"] * p["rho_app"] if it.get("truth_m3") else None
    return items


def cmd_run(a):
    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("[1] objvol.json 읽기")
    items, geo, skipped = load_items(a.objvol, cfg, a.material, a.default_material, a.objects, a.vol_key, a.r95)
    print(f"  물체 {len(items)}개" + (f" · 부피 없어 건너뜀 {skipped}" if skipped else ""))
    print("[2] 재질 · [3] 무게")
    add_weights(items, cfg, wet=a.wet)
    print(f"  {'물체':<20} {'재질':<13} {'크기(cm)':<13} {'부피 L':>7} {'ρ':>4} {'무게 kg':>8} {'구간':>12} {'업체방식':>9}")
    for it in items:
        print(f"  {it['item_id']:<20} {it['cls']:<13} {it['length_m']*100:.0f}×{it['width_m']*100:.0f}×{it['h_p90_m']*100:<5.0f} "
              f"{it['volume_m3']*1000:7.1f} {it['rho_app']:4.0f} {it['weight_est_kg']:8.3f} "
              f"{it['weight_lo_kg']:5.3f}~{it['weight_hi_kg']:<5.3f} {it['company_kg']:9.4f}")
    print("[4] 수거 계획")
    stops, summary = plan.make_plan(items, cfg)
    summary["company_kg"] = float(sum(it["company_kg"] for it in items))
    summary["bulk_raw_m3"] = float(sum(it["bulk_raw_m3"] for it in items))
    summary["wet_applied"] = bool(a.wet)
    summary["kg_wet_hi"] = float(sum(it["weight_wet_hi_kg"] for it in items))
    summary["volume_key"] = a.vol_key
    summary["items"] = [{k: it.get(k) for k in (
        "item_id", "cls", "material_raw", "lat", "lon", "length_m", "width_m", "h_p90_m", "area_m2",
        "volume_m3", "truth_m3", "rho_app", "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "weight_wet_hi_kg",
        "weight_method", "company_code", "company_kg", "truth_kg", "crew", "stop_id", "compress", "bulk_m3")}
        for it in items]
    print(f"  정거장 {summary['n_stops']} · 추정 {summary['kg_est']:.2f} kg (최대 {summary['kg_hi']:.2f}, 젖으면 최대 {summary['kg_wet_hi']:.2f})"
          f" · 인력 {summary['crew_counts']} · 마대 {summary['bags']}개 (겉부피 {summary['bulk_raw_m3']*1000:.0f} L → 압축 {summary['bulk_m3']*1000:.0f} L)"
          f" · 트럭 {summary['trucks']} · 경로 {summary['route_m']:.0f} m · 업체방식 합계 {summary['company_kg']:.4f} kg")
    print("[5] 결과물")
    report.write_tables(out, items, stops, summary)
    report.fig_map(out, items, stops, summary)
    report.work_cards(out, items, stops, summary, None)
    print(f"  → {out}  (work_cards.html, fig_map.png, items.csv, stops.csv, *.geojson, summary.json)")
    return items, stops, summary


# ------------------------------------------------------------------ 비교 그림
# 업체 재질코드별 "현실적" 두께·겉보기밀도 범위 (가정 — 비교 그림 (b) 전용)
REALISTIC = {
    "STY": {"thick": (0.02, 0.35), "rho": (11, 32)},    # 스티로폼 조각(2 cm)~부표(35 cm); EPS 11–32 kg/m³
    "ROP": {"thick": (0.02, 0.05), "rho": (150, 400)},  # 로프 뭉치 (가정)
    "FIS": {"thick": (0.05, 0.30), "rho": (100, 150)},  # 어망 더미 (가정)
    "PLA": {"thick": (0.02, 0.10), "rho": (50, 80)},    # 플라스틱 조각·용기 (가정)
}


def fig_boxes(out, items):
    """(a) 상자별: 업체 방식 vs 3D 부피×밀도 vs 참값 기준."""
    import matplotlib.pyplot as plt
    _font()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    n = len(items)
    xs = np.arange(n)
    w = 0.26
    labels = [f"{it['item_id']}\n({it['length_m']*100:.0f}×{it['width_m']*100:.0f}×{it['h_p90_m']*100:.0f} cm, "
              f"{COMPANY_CODE_NAME.get(it['company_code'], it['cls'])})" for it in items]
    comp = [it["company_kg"] for it in items]
    ours = [it["weight_est_kg"] for it in items]
    lo = [it["weight_lo_kg"] for it in items]
    hi = [it["weight_hi_kg"] for it in items]
    tru = [it["truth_kg"] or np.nan for it in items]
    a1.bar(xs - w, comp, w, color=C_COMPANY, label="업체 방식 (면적 × 계수 kg/m²)")
    a1.bar(xs, ours, w, color=C_OURS, label="제안: 3D 부피 × 겉보기밀도",
           yerr=[np.array(ours) - lo, np.array(hi) - ours], capsize=3, ecolor="#444")
    a1.bar(xs + w, tru, w, color=C_TRUTH, label="참값 기준 (정답 부피 × 같은 밀도)")
    a1.set_yscale("log")
    a1.set_ylim(min(comp) * 0.4, max(hi) * 4)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 10^-2 대신 0.01 (글꼴 − 깨짐 방지)
    for i in range(n):
        ratio = f"\n(참값의 1/{tru[i]/comp[i]:.0f})" if np.isfinite(tru[i]) else ""
        a1.text(xs[i] - w, comp[i] * 1.15, f"{comp[i]:.4f}{ratio}", ha="center", va="bottom", fontsize=8, color=C_COMPANY)
        a1.text(xs[i], hi[i] * 1.15, f"{ours[i]:.2f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(tru[i]):
            a1.text(xs[i] + w, tru[i] * 1.15, f"{tru[i]:.2f}", ha="center", va="bottom", fontsize=8)
    a1.set_xticks(xs, labels, fontsize=8)
    a1.set_ylabel("무게 (kg, 로그 눈금)")
    a1.set_title("상자 2개 무게: 업체 방식 vs 3D 부피 방식", fontsize=11)
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(axis="y", alpha=.3)
    # (오른쪽) 부피 비교
    v_our = [it["volume_m3"] * 1000 for it in items]
    v_tru = [(it["truth_m3"] or np.nan) * 1000 for it in items]
    vmax = max(max(v_our), np.nanmax(v_tru))
    a2.bar(xs - w / 2, v_our, w, color=C_OURS, label="3D 측정 부피 (높이지도)")
    a2.bar(xs + w / 2, v_tru, w, color=C_TRUTH, label="정답 부피 (줄자)")
    for i in range(n):
        a2.text(xs[i] - w / 2, v_our[i], f"{v_our[i]:.1f} L", ha="center", va="bottom", fontsize=8)
        if np.isfinite(v_tru[i]):
            a2.text(xs[i] + w / 2, v_tru[i], f"{v_tru[i]:.1f} L", ha="center", va="bottom", fontsize=8)
            a2.text(xs[i], max(v_our[i], v_tru[i]) + vmax * 0.08, f"오차 {(v_our[i]/v_tru[i]-1)*100:+.0f}%", ha="center", fontsize=9)
    a2.set_ylim(0, vmax * 1.3)
    a2.set_xticks(xs, [it["item_id"] for it in items], fontsize=8)
    a2.set_ylabel("부피 (L)")
    a2.set_title("3D 부피 측정 정확도 (영상 0007, 조밀 복원)", fontsize=11)
    a2.legend(fontsize=8)
    a2.grid(axis="y", alpha=.3)
    fig.text(0.5, 0.005, "※ 무게 실측값이 없어 '정답 부피 × 같은 겉보기밀도'를 기준으로 삼음. 업체 계수: 스티로폼 0.012, 로프·어망 0.024, 플라스틱 0.020 kg/m².",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "fig_compare_boxes.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_company_labels(out, labels_path):
    """(b) 업체 라벨 42개: 기록 weight_kg vs 같은 면적의 현실적 무게 범위."""
    import matplotlib.pyplot as plt
    _font()
    d = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    rows = []
    for f in d["features"]:
        p = f["properties"]
        code = p["material_code"]
        R = REALISTIC.get(code, REALISTIC["PLA"])
        area = float(p["area_sqm"])
        rows.append({"code": code, "area": area, "kg": float(p["weight_kg"]),
                     "lo": area * R["thick"][0] * R["rho"][0], "hi": area * R["thick"][1] * R["rho"][1]})
    rows.sort(key=lambda r: r["area"])
    n = len(rows)
    xs = np.arange(n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 2.2]})
    # 왼쪽: 기록 무게 분포
    kg = np.array([r["kg"] for r in rows])
    a1.hist(kg, bins=np.linspace(0, 0.26, 14), color=C_COMPANY, edgecolor="white")
    a1.set_xlabel("업체 라벨 weight_kg (kg)")
    a1.set_ylabel("물체 수")
    a1.set_title(f"업체 라벨 {n}개의 기록 무게\n합계 {kg.sum():.2f} kg · 최대 {kg.max():.3f} kg", fontsize=10)
    a1.grid(axis="y", alpha=.3)
    # 오른쪽: 면적순 — 기록값(점) vs 현실적 범위(막대), 로그
    for i, r in enumerate(rows):
        a2.plot([i, i], [r["lo"], r["hi"]], color=C_RANGE, lw=4, solid_capstyle="butt", zorder=1)
    a2.scatter(xs, kg, color=C_COMPANY, s=22, zorder=3, label="업체 기록 (면적 × 계수)")
    a2.plot([], [], color=C_RANGE, lw=4, label="현실적 범위 (면적 × 두께 2~35 cm × 밀도)")
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_xticks(xs, [f"{r['area']:.2f}" for r in rows], rotation=90, fontsize=6.5)
    a2.set_xlabel("라벨 면적 (m², 작은 것부터)")
    a2.set_ylabel("무게 (kg, 로그 눈금)")
    codes = [r["code"] for r in rows]
    for i, r in enumerate(rows):
        if r["code"] != "STY":
            a2.text(i, r["hi"] * 1.3, COMPANY_CODE_NAME[r["code"]], ha="center", fontsize=7, color="#444")
    k = next((i for i, r in enumerate(rows) if r["code"] == "STY" and abs(r["area"] - 1.54) < 0.01), None)
    if k is not None:
        r = rows[k]
        a2.annotate(f"스티로폼 1.54 m² → 기록 {r['kg']:.3f} kg\n(현실적으론 {r['lo']:.1f}~{r['hi']:.0f} kg)",
                    (k, r["kg"]), xytext=(k - 18, r["kg"] * 4.5), fontsize=8.5, color=C_COMPANY,
                    arrowprops={"arrowstyle": "->", "color": C_COMPANY})
    lo_sum, hi_sum = sum(r["lo"] for r in rows), sum(r["hi"] for r in rows)
    a2.set_title(f"같은 면적이면 실제로는 얼마나 나갈까 — 합계 기록 {kg.sum():.2f} kg vs 현실적 {lo_sum:.0f}~{hi_sum:.0f} kg",
                 fontsize=10)
    a2.legend(fontsize=8, loc="upper left")
    a2.grid(axis="y", alpha=.3)
    fig.text(0.5, 0.005, "※ 두께·밀도 범위는 가정 (스티로폼 2~35 cm·11~32 kg/m³, 로프 2~5 cm·150~400, 어망 5~30 cm·100~150, 플라스틱 2~10 cm·50~80). "
             "업체 weight_kg는 실측이 아니라 면적×고정계수 계산값.", ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "fig_compare_company_labels.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, {"n": n, "sum_kg": float(kg.sum()), "max_kg": float(kg.max()), "realistic_lo_kg": lo_sum,
               "realistic_hi_kg": hi_sum, "codes": {c: codes.count(c) for c in set(codes)}}


def cmd_compare(a):
    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, _, _ = load_items(a.objvol, cfg, a.material, a.default_material, a.objects, a.vol_key, a.r95)
    add_weights(items, cfg)
    res = {"boxes": [{k: it.get(k) for k in ("item_id", "cls", "company_code", "area_m2", "volume_m3", "truth_m3",
                                             "rho_app", "company_kg", "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "truth_kg")}
                     for it in items]}
    p1 = fig_boxes(out, items)
    print(f"(a) {p1}")
    for b in res["boxes"]:
        tk = b["truth_kg"]
        print(f"  {b['item_id']}: 업체 {b['company_kg']:.4f} kg · 제안 {b['weight_est_kg']:.3f} ({b['weight_lo_kg']:.3f}~{b['weight_hi_kg']:.3f}) · "
              f"참값기준 {tk:.3f} kg → 업체는 참값의 1/{tk / b['company_kg']:.0f}, 제안은 {(b['weight_est_kg'] / tk - 1) * 100:+.0f}%"
              if tk else f"  {b['item_id']}: 업체 {b['company_kg']:.4f} · 제안 {b['weight_est_kg']:.3f}")
    if a.labels:
        p2, st = fig_company_labels(out, a.labels)
        res["company_labels"] = st
        print(f"(b) {p2}\n  라벨 {st['n']}개 {st['codes']} · 기록 합계 {st['sum_kg']:.2f} kg (최대 {st['max_kg']:.3f}) · "
              f"현실적 범위 합계 {st['realistic_lo_kg']:.0f}~{st['realistic_hi_kg']:.0f} kg")
    (out / "compare_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.fromvideo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--objvol", required=True, help="objvol.json (litter.objvol 결과)")
        p.add_argument("--objects", help="orbit_map objects.csv (objvol.json에 위경도 없을 때)")
        p.add_argument("--material", help="'태그:재질,...' 재질 덮어쓰기 (예: Plastic_Buoy_55:Styrofoam)")
        p.add_argument("--default_material", help="태그 매핑 안 되는 물체의 기본 재질")
        p.add_argument("--vol_key", default="volume_heightmap_L", help="volume_heightmap_L | volume_mask_box_L | volume_hull_L")
        p.add_argument("--r95", type=float, default=1.2, help="위치 95% 반경 m (0007: 3D↔GPS 잔차 1.16 m)")
        p.add_argument("--config", help="설정 덮어쓰기 JSON")
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--wet", action="store_true", help="젖음 계수를 무게에 적용")
        else:
            p.add_argument("--labels", help="업체 라벨 GeoJSON (data/company/MGD_labels.json)")
    a = ap.parse_args(argv)
    (cmd_run if a.cmd == "run" else cmd_compare)(a)


if __name__ == "__main__":
    main()
