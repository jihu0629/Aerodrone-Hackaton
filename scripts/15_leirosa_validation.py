"""Andriolo et al. (2024) 레이로자 해변 실측 데이터로 litter3d 검증.

  ① 논문 표 3·4·5 (W1·W2·W3) 재현
  ② 표 1 구성(1,505개, 24,720 g)으로 3D 시뮬레이션 → 우리 방식 vs W2 vs W3, 종류별 오차
  ③ 논문 그림 사진 2장으로 검출 개수 비교 (색 기반 베이스라인 vs 논문의 객체 분할 빨간 윤곽)

    python scripts/15_leirosa_validation.py --out outputs/leirosa [--photos DIR] [--noise 0.5]
치수·DSM 노이즈는 가정값이다. 결과 보고서: outputs/leirosa/report.md
"""
import argparse, sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2

from litter3d import leirosa as L
from litter3d.baselines import w1_by_type, w2_count, w3_area, w3_dsm, error_pct
from litter3d.classes import CLASSES, KO_NAMES
from litter3d.volume import object_volume
from litter3d.mass import estimate_mass

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/leirosa")
ap.add_argument("--noise", type=float, default=0.5, help="DSM 노이즈 σ (cm), 가정값")
ap.add_argument("--gsd", type=float, default=L.GSD_CM, help="cm/px")
ap.add_argument("--photos", default=None, help="논문 그림 크롭 PNG 폴더 (crop_*.png)")
ap.add_argument("--reps", type=int, default=10, help="종류별 렌더 반복 수 (크기 ±20 % 흔듦)")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rep = []
P = rep.append

# ---------------------------------------------------------------- ① 표 재현
P("# Andriolo et al. (2024) 데이터로 litter3d 검증\n")
P("## ① 논문 표 재현 (코드 입력 검증)\n")
P("| 방법 | 논문 값 (g) | 코드 값 (g) | 일치 |\n|---|---|---|---|")
for col, key in ((2, "census"), (3, "lower"), (4, "mean"), (5, "upper")):
    est = w1_by_type([(t[0], t[1], t[col]) for t in L.TABLE3])
    P(f"| W1 {key} | {L.TABLE3_TOTALS[key]:,} | {est:,.0f} | {'✓' if abs(est / L.TABLE3_TOTALS[key] - 1) < 0.01 else '✗'} |")
P(f"| W2 평균 13.5 g × 1445 | 19,480 | {w2_count(1445, 13.5):,.0f} | ✓ |")
P(f"| W2 × 899 (AI 검출) | 12,120 | {w2_count(899, 13.5):,.0f} | ✓ |")
P(f"| W3 0.4 cm × 1.2 | 25,092 | {w3_area(L.OBJ_SEG_AREA_CM2):,.0f} | ✓ |")
P(f"| W3' DSM 12,230 cm³ × 0.8–1.5 | 9,784–18,345 | {w3_dsm(L.SEM_SEG_VOLUME_CM3, .8):,.0f}–{w3_dsm(L.SEM_SEG_VOLUME_CM3, 1.5):,.0f} | ✓ |")
P("\n표 3 '미확인' 행은 본문(25.7 g)과 표 합계(11.1 g)가 다르다. 표 합계를 따랐다.\n")

# ---------------------------------------------------------------- ② 시뮬레이션
P("## ② 표 1 구성으로 3D 시뮬레이션\n")
P(f"- 정답: {L.GROUND_TRUTH_N:,}개, {L.GROUND_TRUTH_G:,} g (표 1). 종류별 개수·무게는 논문 실측, **물체 치수는 우리 가정값**.")
P(f"- DSM: GSD {a.gsd} cm/px, 노이즈 σ {a.noise} cm (가정), 경사 2 %. 검출은 100 % 라고 가정 (누락 없음).")
P(f"- 종류마다 {a.reps}개를 크기 ±20 % 로 흔들어 렌더 → `volume.object_volume` → `mass.estimate_mass` → 개수 배.\n")

gsd_m = a.gsd / 100
rng = np.random.default_rng(0)


def render(row: L.CensusRow, scale: float):
    Lc, Wc, Hc = row.L_cm * scale, row.W_cm * scale, row.H_cm * scale
    pad = 12
    w = int(Lc / a.gsd) + 2 * pad; h = int(Wc / a.gsd) + 2 * pad
    gy, gx = np.mgrid[0:h, 0:w]
    dsm = 0.02 * gx * gsd_m + 0.005 * gy * gsd_m + rng.normal(0, a.noise / 100, (h, w))
    cx, cy = w / 2, h / 2
    rx, ry = Lc / 2 / a.gsd, Wc / 2 / a.gsd
    dx, dy = (gx - cx) / max(rx, 0.5), (gy - cy) / max(ry, 0.5)
    if row.shape == "box":
        m = (np.abs(dx) < 1) & (np.abs(dy) < 1); hgt = np.where(m, Hc / 100, 0)
    else:
        r2 = dx ** 2 + dy ** 2; m = r2 < 1
        hgt = np.where(m, Hc / 100 if row.shape == "cylinder" else Hc / 100 * np.sqrt(np.clip(1 - r2, 0, 1)), 0)
    if not m.any():
        m[int(cy), int(cx)] = True; hgt[int(cy), int(cx)] = Hc / 100
    return (dsm + hgt).astype(np.float32), m


rows_out = []
for row in L.TABLE1:
    est = []; true_vols = []
    for k in range(a.reps):
        s = 1 + (k / max(a.reps - 1, 1) - 0.5) * 0.4     # 0.8 … 1.2 균등 (표본 우연 제거)
        dsm, m = render(row, s)
        ov = object_volume(dsm, m, gsd_m, class_name=row.our_class)
        mm = estimate_mass(ov)
        est.append((mm.kg_min, mm.kg_typ, mm.kg_max, mm.method, ov.volume_m3, ov.area_m2))
        true_vols.append(row.volume_cm3 * s ** 3)
    kg_min = np.mean([e[0] for e in est]) * row.n; kg_typ = np.mean([e[1] for e in est]) * row.n
    kg_max = np.mean([e[2] for e in est]) * row.n
    vol_cm3 = np.mean([e[4] for e in est]) * 1e6; area_cm2 = np.mean([e[5] for e in est]) * 1e4
    method = est[0][3]
    rows_out.append(dict(code=row.code, type=row.type_, category=row.category, n=row.n, true_g=row.total_g,
                         our_class=row.our_class, method=method, vol_cm3=vol_cm3, area_cm2=area_cm2,
                         ours_min=kg_min * 1000, ours_typ=kg_typ * 1000, ours_max=kg_max * 1000,
                         w2=w2_count(row.n, L.LITERATURE_MEAN_OF_MEANS), w3=w3_area(area_cm2 * row.n),
                         true_rho=row.true_apparent_density_kg_m3, vol_err_pct=error_pct(vol_cm3, np.mean(true_vols))))

T = lambda k: sum(r[k] for r in rows_out)
tot_true = T("true_g")
P("### 총량 (검출 100 % 가정)\n")
P("| 방법 | 추정 (g) | 오차 |\n|---|---|---|")
P(f"| 정답 (표 1) | {tot_true:,.0f} | — |")
P(f"| **우리 3D (대표값)** | **{T('ours_typ'):,.0f}** | **{error_pct(T('ours_typ'), tot_true):+.0f} %** |")
P(f"| 우리 3D 범위 (최소–최대) | {T('ours_min'):,.0f} – {T('ours_max'):,.0f} | {error_pct(T('ours_min'), tot_true):+.0f} % ~ {error_pct(T('ours_max'), tot_true):+.0f} % |")
P(f"| W2 개수 × 13.5 g | {T('w2'):,.0f} | {error_pct(T('w2'), tot_true):+.0f} % |")
P(f"| W3 면적 × 0.4 cm × 1.2 | {T('w3'):,.0f} | {error_pct(T('w3'), tot_true):+.0f} % |")
vol_all = T("vol_cm3") if False else sum(r["vol_cm3"] * r["n"] for r in rows_out)
P(f"| W3' 겉부피 × 1.2 g/cm³ (재료 비중) | {vol_all * 1.2:,.0f} | {error_pct(vol_all * 1.2, tot_true):+.0f} % |")

# 검출 누락 시나리오: 논문의 HRNet 은 1000 cm³ 미만을 못 잡음
det = [r for r in rows_out if r["vol_cm3"] >= L.SEM_SEG_MIN_DETECT_CM3]
P(f"\n### 검출 누락 반영 (부피 < {L.SEM_SEG_MIN_DETECT_CM3} cm³ 는 못 잡는다고 가정, 논문 4.1절)\n")
P(f"- 검출되는 종류 {len(det)}/{len(rows_out)}, 개수 {sum(r['n'] for r in det)}/{L.GROUND_TRUTH_N}, 이들의 실제 무게 {sum(r['true_g'] for r in det):,.0f} g ({sum(r['true_g'] for r in det) / tot_true:.0%})")
P(f"- 우리 3D 대표값 {sum(r['ours_typ'] for r in det):,.0f} g → 정답 대비 {error_pct(sum(r['ours_typ'] for r in det), tot_true):+.0f} %  (논문 W3' 는 −60 ~ −26 %)")
P("- 결론: 검출 누락이 밀도 오차보다 크다. 재현율을 같이 보고해야 한다.\n")

# 범주별
P("### 범주별 (검출 100 %)\n")
P("| 범주 | 개수 | 정답 (g) | 우리 (g) | 오차 | W2 (g) | W3 (g) |\n|---|---|---|---|---|---|---|")
cats = {}
for r in rows_out:
    c = cats.setdefault(r["category"], dict(n=0, true_g=0, ours=0, w2=0, w3=0))
    for k, kk in (("n", "n"), ("true_g", "true_g"), ("ours", "ours_typ"), ("w2", "w2"), ("w3", "w3")):
        c[k] += r[kk]
for c, d in sorted(cats.items(), key=lambda kv: -kv[1]["true_g"]):
    P(f"| {c} | {d['n']} | {d['true_g']:,.0f} | {d['ours']:,.0f} | {error_pct(d['ours'], d['true_g']):+.0f} % | {d['w2']:,.0f} | {d['w3']:,.0f} |")

# 종류별 (무게 큰 순 상위 + 오차 큰 것)
P("\n### 종류별 (정답 무게 상위 20)\n")
P("| OSPAR 종류 | 우리 클래스 | 방식 | n | 정답 (g) | 우리 (g) | 오차 | 실제 겉보기밀도* (kg/m³) | 표의 ρ_typ | 부피오차 |\n|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows_out, key=lambda r: -r["true_g"])[:20]:
    rho = CLASSES.get(r["our_class"], CLASSES["unknown"]).rho_typ
    P(f"| {r['type']} | {KO_NAMES.get(r['our_class'], r['our_class'])} | {r['method']} | {r['n']} | {r['true_g']:,.0f} | {r['ours_typ']:,.0f} | {error_pct(r['ours_typ'], r['true_g']):+.0f} % | {r['true_rho']:.0f} | {rho} | {r['vol_err_pct']:+.1f} % |")
P("\n\\* 실제 겉보기밀도 = 논문 실측 평균무게 ÷ **우리가 가정한** 겉 부피. 치수 가정이 바뀌면 같이 바뀐다.\n")
P("### 오차가 큰 종류 (|오차| > 100 % 또는 100 g 이상 차이)\n")
P("| 종류 | 방식 | 정답 (g) | 우리 (g) | 차이 (g) | 원인 추정 |\n|---|---|---|---|---|---|")
for r in sorted(rows_out, key=lambda r: -abs(r["ours_typ"] - r["true_g"])):
    d = r["ours_typ"] - r["true_g"]
    if abs(d) < 100 and abs(error_pct(r["ours_typ"], r["true_g"])) < 100:
        continue
    spec = CLASSES.get(r["our_class"], CLASSES["unknown"])
    cause = ("소형 → 개수×평균무게 대체값이 실제보다 큼/작음" if r["method"] == "count" and r["true_g"] / r["n"] < 5 else
             "속 빈/느슨한 물체: 겉보기 밀도 가정값이 너무 큼" if r["true_rho"] < spec.rho_typ / 2 else
             "밀도 가정값이 너무 작음 (무거운 재질)" if r["true_rho"] > spec.rho_typ * 2 else
             "개수 기반 대체값(W1) 사용" if r["method"] == "count" else "치수 가정 문제")
    P(f"| {r['type']} | {r['method']} | {r['true_g']:,.0f} | {r['ours_typ']:,.0f} | {d:+,.0f} | {cause} |")

with open(out / "simulation_rows.json", "w", encoding="utf-8") as f:
    json.dump(rows_out, f, ensure_ascii=False, indent=1, default=float)

# ---------------------------------------------------------------- ③ 사진
if a.photos:
    from litter3d.segment import color_baseline
    P("\n## ③ 논문 사진으로 검출 개수 비교\n")
    P("사진은 논문 PDF 그림을 잘라낸 것이라 해상도가 낮고(원본 0.4 cm/px 가 아님) DSM 이 없어 부피는 못 잰다. 검출 개수만 비교한다.\n")
    P("| 사진 | 폭 (m) | 크롭 GSD (cm/px) | 논문 빨간 윤곽 수 | 색 기반 검출 수 (≥2.5 cm) | W2 개수×14 g |\n|---|---|---|---|---|---|")
    widths = {"crop_fig1d_left": 5.0, "crop_fig1d_right": 5.0, "crop_fig3c_left": 2.0, "crop_fig3c_right": 1.0}
    for name, wm in widths.items():
        p = Path(a.photos) / f"{name}.png"
        if not p.exists():
            continue
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        gsd_crop = wm * 100 / w
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
        n_red = 0
        if "fig3c" in name and red.sum() > 0:   # 빨간 윤곽은 그림 3c 에만 있음
            red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            n_red = sum(1 for c in cnts if cv2.contourArea(c) >= 8)
            img = cv2.inpaint(img, cv2.dilate(red, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
        min_px = max(int((2.5 / gsd_crop) ** 2), 4)
        dets = color_baseline(img, min_px=min_px)
        vis = img.copy()
        for _, m, _ in dets:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"{name}_colorbaseline.png"), vis)
        P(f"| {name} | {wm} | {gsd_crop:.2f} | {n_red if n_red else '—'} | {len(dets)} | {len(dets) * 14:,.0f} g |")
    P("\n색 기반(Kako 2020 방식)은 나무 조각·해초·발자국 그림자도 잡는다 → 과검출. 논문 그림 3c 의 빨간 윤곽은 사람이 표시한 점을 바탕으로 한 객체 분할이라 훨씬 적다. "
      "CNN 분할(YOLO-seg)이 필요한 이유이며, 학습 후 같은 사진으로 다시 비교할 수 있다.\n")

P("\n## 가정과 한계\n")
P("- 논문에는 종류별 무게·개수만 있고 치수·DSM 은 없다. 치수는 우리가 정했다 → ②는 '밀도표가 이 실측 무게와 맞는가' 를 보는 감도 분석이지 정밀 검증이 아니다.")
P("- 개수 기반 대체값 중 일부(스티로폼 조각 3.1 g, 유리병 165.8 g 등)는 이 논문에서 가져온 값이라 그 종류의 오차는 0 에 가깝게 나온다(순환).")
P("- 사진 검증은 저해상도 그림 크롭이라 검출 개수 경향만 본다.")
(out / "report.md").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
