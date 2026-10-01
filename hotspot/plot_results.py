"""하와이 공간 집중 + MDMAP 시간 반복성 결과를 한 장으로."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

H = Path(__file__).parent
chips = list(csv.DictReader(open(H / "hawaii_chips_bearing.csv")))
mdmap = json.loads((H / "mdmap_sites_summary.json").read_text())

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("쓰레기 밀집지역 검증 — 공간적으로 몰리고(하와이 2015), 시간이 지나도 같은 곳에 다시 쌓인다(NOAA MDMAP)",
             fontsize=13, fontweight="bold")

# 1) 하와이 지도
a = ax[0, 0]
lon = [float(r["lon"]) for r in chips]
lat = [float(r["lat"]) for r in chips]
n = [int(r["n_labels"]) for r in chips]
sc = a.scatter(lon, lat, c=n, s=[3 + v * 1.5 for v in n], cmap="inferno_r", alpha=0.7, edgecolors="none")
a.set_title("① 하와이 해안쓰레기 위치 (칩 1,587장, 라벨 10,703개)")
a.set_xlabel("경도"); a.set_ylabel("위도")
a.set_aspect(1 / math.cos(math.radians(20.8)))
a.annotate("카밀로 해변\n(해류 집적지)", (-155.62, 18.97), (-156.6, 19.2), arrowprops=dict(arrowstyle="->"), fontsize=9)
a.annotate("니이하우\n(전체의 53%)", (-160.1, 21.85), (-159.6, 21.2), arrowprops=dict(arrowstyle="->"), fontsize=9)
fig.colorbar(sc, ax=a, label="칩당 라벨 수")
a.grid(alpha=0.2)

# 2) 집중 곡선
a = ax[0, 1]
cells = defaultdict(int)
for r in chips:
    cells[(r["island"], round(float(r["lat"]) / 0.009), round(float(r["lon"]) / 0.0097))] += int(r["n_labels"])
vals = sorted(cells.values(), reverse=True)
tot = sum(vals)
cum, s = [], 0
for v in vals:
    s += v; cum.append(s / tot * 100)
x = [(i + 1) / len(vals) * 100 for i in range(len(vals))]
a.plot(x, cum, color="#1657D0", lw=2.5)
a.plot([0, 100], [0, 100], "--", color="#aaa", lw=1, label="고르게 퍼져 있다면")
for p, col in ((10, "#d94f4f"), (20, "#e8962e")):
    k = int(len(vals) * p / 100)
    a.scatter([p], [cum[k - 1]], color=col, zorder=5)
    a.annotate(f"상위 {p}% 칸 → {cum[k-1]:.0f}%", (p, cum[k - 1]), (p + 8, cum[k - 1] - 8), fontsize=11, color=col)
a.set_title("② 해안 1km 칸 집중도: 소수의 칸에 대부분이 몰림")
a.set_xlabel("쓰레기가 많은 순으로 칸을 골랐을 때 누적 비율 (%)"); a.set_ylabel("커버되는 쓰레기 (%)")
a.legend(loc="lower right"); a.grid(alpha=0.3); a.set_xlim(0, 100); a.set_ylim(0, 102)

# 3) 해안 방향
a = ax[1, 0]
S = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
sec = Counter()
for r in chips:
    sec[S[int(((float(r["bearing_deg"]) + 22.5) % 360) // 45)]] += int(r["n_labels"])
tot = sum(sec.values())
colors = ["#1657D0" if s in ("NE", "E") else "#9bb3e0" for s in S]
a.bar(S, [sec[s] / tot * 100 for s in S], color=colors)
a.axhline(12.5, ls="--", color="#888", lw=1)
a.text(7.4, 13.3, "고르면 12.5%", ha="right", color="#666", fontsize=9)
a.set_title("③ 해안이 바라보는 방향별 비중 (진한 색 = 무역풍 정면)")
a.set_ylabel("라벨 비중 (%)")
a.text(0.02, 0.95, "무역풍 정면 ±45°: 39.7% (기대 25%의 1.6배)\n단, 남쪽(S)도 28% — 바람만으로는 설명 안 됨",
       transform=a.transAxes, va="top", fontsize=10, bbox=dict(fc="white", ec="#ccc"))
a.set_ylim(0, 38)
a.grid(alpha=0.3, axis="y")

# 4) MDMAP 반복성
a = ax[1, 1]
f = [max(r["first"], 0.1) for r in mdmap]
s2 = [max(r["second"], 0.1) for r in mdmap]
k = round(len(mdmap) * 0.2)
top = set(sorted(range(len(mdmap)), key=lambda i: -f[i])[:k])
a.scatter([f[i] for i in range(len(f)) if i not in top], [s2[i] for i in range(len(f)) if i not in top],
          s=22, color="#9bb3e0", label="나머지 지점")
a.scatter([f[i] for i in top], [s2[i] for i in top], s=30, color="#d94f4f", label="앞 기간 상위 20%")
hi = [i for i, r in enumerate(mdmap) if r["state"] == "Hawaii"]
for i in hi:
    a.annotate(mdmap[i]["name"].split("(")[0].strip()[:14], (f[i], s2[i]), fontsize=8, xytext=(4, -10), textcoords="offset points")
lo, hi_ = 0.1, max(f + s2) * 1.5
a.plot([lo, hi_], [lo, hi_], "--", color="#aaa", lw=1)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("앞 기간 평균 (개/100m/조사)"); a.set_ylabel("뒤 기간 평균 (개/100m/조사)")
a.set_title(f"④ 같은 지점 반복조사 {len(mdmap)}곳: 많던 곳이 계속 많다")
a.text(0.02, 0.97, "순위상관 0.88\n상위 20% 유지율 74% (우연 20%)\n앞 상위 20%만 재방문 → 뒤 쓰레기 56% 커버",
       transform=a.transAxes, va="top", fontsize=10, bbox=dict(fc="white", ec="#ccc"))
a.legend(loc="lower right"); a.grid(alpha=0.3, which="both")

plt.tight_layout(rect=(0, 0, 1, 0.96))
out = H / "hotspot_results.png"
plt.savefig(out, dpi=130)
print(out)
