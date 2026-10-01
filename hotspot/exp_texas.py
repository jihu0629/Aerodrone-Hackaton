"""
실험 1 — 과거 기록으로 짠 경로가 "다음 조사"에서 실제로 더 많이 커버하는가.
(NOAA MDMAP 텍사스 반복조사 33곳, 시간 순서 기준 앞/뒤 분할 = 진짜 미래 예측)

  학습: 각 지점의 앞 절반 조사 → 기대량·편차·조사횟수
  평가: 뒤 절반 조사의 실제 양 중 경로가 방문한 지점이 차지하는 비율
  예산: 33곳 전부 도는 거리의 10~100%

비교: 핫스팟 우선(우리) / 핫스팟+탐색 / 가까운 곳부터 / 무작위 / 정답을 아는 경우(상한)
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "path_planning"))
from hotspot_route import Cell, plan_route, nearest_first, random_order, full_tour_cost, route_cost  # noqa: E402

D = Path(__file__).parent / "data" / "mdmap"
STATE = sys.argv[1] if len(sys.argv) > 1 else "Texas"

# 중요도 가중치(가정값): 무겁거나 치우기 어려워 트럭·인력 계획을 바꾸는 것일수록 높게
WEIGHT = {"rope-and-nets": 5, "buoys-and-floats": 5, "building-material": 3,
          "other-jugs-or-containers": 2, "fishing-lures-and-line": 2}
WEIGHT_MATERIAL = {("metal", "other"): 2, ("processed-lumber", "other"): 3, ("rubber", "other"): 2}


def w(d):
    return WEIGHT_MATERIAL.get((d["material_type"], d["item_type"]), WEIGHT.get(d["item_type"], 1))


def centroid(wkt):
    nums = [float(v) for v in re.findall(r"-?\d+\.\d+", wkt)]
    xs, ys = nums[0::2], nums[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def load():
    sites = {s["id"]: s for s in json.loads((D / "sites.json").read_text())}
    surveys = json.loads((D / "surveys.json").read_text())
    summ = json.loads((Path(__file__).parent / "mdmap_sites_summary.json").read_text())
    ids = [r["site"] for r in summ if r["state"] == STATE]
    dates = {s["id"]: s["date"] for s in surveys}

    lon0 = mean(centroid(sites[k]["coordinates"])[0] for k in ids)
    lat0 = mean(centroid(sites[k]["coordinates"])[1] for k in ids)
    cells, truth = [], []
    for k in ids:
        per = defaultdict(float)
        for d in json.loads((D / "debris" / f"{k}.json").read_text()):
            per[d["survey"]] += (d["count"] or 0) * w(d)
        L = sites[k].get("length") or 100
        sv = sorted((dates[s], per.get(s, 0.0) / L * 100) for s in (x["id"] for x in surveys if x["site"] == k))
        half = len(sv) // 2
        a = [v for _, v in sv[:half]]
        b = [v for _, v in sv[half:]]
        lon, lat = centroid(sites[k]["coordinates"])
        x = (lon - lon0) * 111320 * math.cos(math.radians(lat0))
        y = (lat - lat0) * 110540
        cells.append(Cell(id=str(k), x=x, y=y, mean=mean(a), std=pstdev(a), n_obs=len(a),
                          survey_cost=L, meta={"name": sites[k]["name"], "lon": lon, "lat": lat}))
        truth.append(mean(b))
    return cells, truth


def main():
    cells, truth = load()
    # 출발지: 지점들의 중심에 가장 가까운 지점 (거점 사무소라고 가정)
    cx, cy = mean(c.x for c in cells), mean(c.y for c in cells)
    hub = min(cells, key=lambda c: math.hypot(c.x - cx, c.y - cy))
    base = (hub.x, hub.y)
    full = full_tour_cost(cells, base)
    T = sum(truth)
    print(f"{STATE}: {len(cells)}곳, 전부 도는 거리 {full/1000:.0f} km, 출발지 {hub.meta['name']}")

    def cov(route):
        return sum(truth[i] for i in route) / T * 100

    rows = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        B = full * frac
        ours = plan_route(cells, base, B)
        ucb = plan_route(cells, base, B, explore=1.0)
        near = nearest_first(cells, base, B)
        rnd = mean(cov(random_order(cells, base, B, seed=s)) for s in range(30))
        orac = plan_route(cells, base, B, values=truth)
        rows.append({"budget": frac, "ours": cov(ours), "ours_explore": cov(ucb), "nearest": cov(near),
                     "random": rnd, "oracle": cov(orac), "n_ours": len(ours), "n_near": len(near)})
        r = rows[-1]
        print(f"예산 {int(frac*100):3d}%  우리 {r['ours']:5.1f}% ({r['n_ours']}곳)  탐색 {r['ours_explore']:5.1f}%  "
              f"가까운순 {r['nearest']:5.1f}% ({r['n_near']}곳)  무작위 {r['random']:5.1f}%  상한 {r['oracle']:5.1f}%")

    out = Path(__file__).parent / f"exp_{STATE.lower().replace(' ', '_')}.json"
    pts = [{"name": c.meta["name"], "lon": c.meta["lon"], "lat": c.meta["lat"], "prior": c.mean, "truth": t}
           for c, t in zip(cells, truth)]
    B = full * 0.3
    example = {"budget": 0.3, "ours": plan_route(cells, base, B), "nearest": nearest_first(cells, base, B)}
    out.write_text(json.dumps({"state": STATE, "full_km": full / 1000, "hub": hub.meta["name"],
                               "rows": rows, "points": pts, "example": example}, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
