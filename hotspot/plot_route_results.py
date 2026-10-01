"""핫스팟 우선 경로 실험 결과 그림 2장."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

H = Path(__file__).parent
C_OURS, C_NEAR, C_RAND, C_ORAC = "#1657D0", "#e8962e", "#9aa3b5", "#3ddb84"

# ---- 그림 A: 미래 예측 검증 (MDMAP) ----
states = ["texas", "california", "oregon", "new_jersey"]
data = {s: json.loads((H / f"exp_{s}.json").read_text()) for s in states}
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle("과거 조사로 짠 경로가 '다음 조사'에서 쓰레기를 얼마나 커버했나 (NOAA MDMAP, 앞 기간으로 계획 → 뒤 기간으로 채점)",
             fontsize=12.5, fontweight="bold")

a = ax[0]
d = data["texas"]
xs = [r["budget"] * 100 for r in d["rows"]]
a.plot(xs, [r["oracle"] for r in d["rows"]], "--", color=C_ORAC, lw=2, label="상한(미래를 안다고 가정)")
a.plot(xs, [r["ours"] for r in d["rows"]], "-o", color=C_OURS, lw=2.8, label="핫스팟 우선(우리)")
a.plot(xs, [r["random"] for r in d["rows"]], "-s", color=C_RAND, lw=1.8, label="무작위 순서")
a.plot(xs, [r["nearest"] for r in d["rows"]], "-^", color=C_NEAR, lw=1.8, label="가까운 곳부터")
r20 = d["rows"][1]
a.annotate(f"예산 20%\n우리 {r20['ours']:.0f}% vs 가까운순 {r20['nearest']:.0f}%", (20, r20["ours"]), (32, 35),
           arrowprops=dict(arrowstyle="->", color=C_OURS), fontsize=11, color=C_OURS)
a.set_title(f"텍사스 해안 {len(d['points'])}곳 (전부 돌면 {d['full_km']:.0f} km)")
a.set_xlabel("이동 예산 (전부 도는 거리 대비 %)"); a.set_ylabel("다음 조사 쓰레기 중 커버한 비율 (%, 중요도 가중)")
a.legend(loc="lower right"); a.grid(alpha=0.3); a.set_ylim(0, 105)

a = ax[1]
a.axis("off")
rows = [["지역", "지점", "예산", "우리", "가까운순", "무작위", "상한"]]
picks = {"texas": [0.1, 0.2], "california": [0.3], "oregon": [0.2, 0.3], "new_jersey": [0.3, 0.4]}
names = {"texas": "텍사스", "california": "캘리포니아", "oregon": "오리건", "new_jersey": "뉴저지"}
for s in states:
    for r in data[s]["rows"]:
        if r["budget"] in picks[s]:
            rows.append([names[s], str(len(data[s]["points"])), f"{int(r['budget']*100)}%",
                         f"{r['ours']:.0f}%", f"{r['nearest']:.0f}%", f"{r['random']:.0f}%", f"{r['oracle']:.0f}%"])
t = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.9)
for (i, j), cell in t.get_celld().items():
    if i == 0:
        cell.set_facecolor("#e8eef9"); cell.set_text_props(weight="bold")
    if j == 3 and i > 0:
        cell.set_text_props(color=C_OURS, weight="bold")
a.set_title("다른 지역 반복 (예산이 빠듯한 구간)", pad=4)
a.text(0.5, 0.02, "핫스팟이 출발지에서 멀수록 차이가 크고(텍사스·뉴저지), 가까운 곳에 이미 핫스팟이 있으면\n"
       "같다(캘리포니아·오리건). 예외: 텍사스 30%에서는 가까운순이 1.7%p 높음.", ha="center", fontsize=9.5, transform=a.transAxes, color="#444")
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(H / "route_eval_mdmap.png", dpi=130)

# ---- 그림 B: 드론 규모 시연 (몰로카이) ----
d = json.loads((H / "exp_hawaii_drone.json").read_text())
cells = d["cells"]
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
fig.suptitle(f"드론 규모 시연 — 하와이 몰로카이, 배터리 1개당 {d['battery_km']:.0f} km 가정 (사전정보: 2015 항공조사)",
             fontsize=12.5, fontweight="bold")
a = ax[0]
vmax = max(c["v"] for c in cells)
a.scatter([c["lon"] for c in cells], [c["lat"] for c in cells], s=[10 + c["v"] / vmax * 600 for c in cells],
          color="#f4a261", alpha=0.55, edgecolors="#b5651d", lw=0.6, label="500 m 구간 (크기 = 중요도 가중 쓰레기)")
cols = [C_OURS, "#7b2cbf", "#d94f4f"]
for k, p in enumerate(d["plans"]["2"]):
    lx, ly = p["launch"]
    pts = [(lx, ly)] + [(cells[i]["lon"], cells[i]["lat"]) for i in p["route"]] + [(lx, ly)]
    a.plot([q[0] for q in pts], [q[1] for q in pts], "-", color=cols[k], lw=2, label=f"비행 {k+1} 경로")
    a.scatter([lx], [ly], marker="^", s=160, color=cols[k], edgecolors="white", zorder=5)
bx, by = d["base"]
a.scatter([bx], [by], marker="*", s=260, color="#333", zorder=6, label="항구(카우나카카이)")
a.set_aspect(1 / math.cos(math.radians(21.1)))
a.set_title("배터리 2개일 때: 알고리즘이 고른 이륙 지점(▲)과 경로")
a.legend(loc="lower left", fontsize=9); a.grid(alpha=0.2)
a.set_xlabel("경도"); a.set_ylabel("위도")

a = ax[1]
import numpy as np
rows = d["rows"]
x = np.arange(len(rows))
wdt = 0.27
a.bar(x - wdt, [r["harbor_sweep"] for r in rows], wdt, color="#c9ced8", label="항구에서 띄워 해안 따라")
a.bar(x, [r["random_sweep"] for r in rows], wdt, color=C_RAND, label="아무 데서 띄워 해안 따라 (30회 평균)")
a.bar(x + wdt, [r["ours"] for r in rows], wdt, color=C_OURS, label="이륙지점 선택 + 핫스팟 우선(우리)")
for i, r in enumerate(rows):
    a.text(i + wdt, r["ours"] + 1.5, f"{r['ours']:.0f}%", ha="center", color=C_OURS, fontsize=10, weight="bold")
a.set_xticks(x); a.set_xticklabels([f"배터리 {r['sorties']}개" for r in rows])
a.set_ylabel("섬 전체 쓰레기 중 본 비율 (%, 중요도 가중)"); a.set_ylim(0, 100)
a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, axis="y")
a.set_title("같은 배터리로 본 양")
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(H / "route_demo_molokai.png", dpi=130)
print("saved")
