"""
실험 2 — 드론 규모 시연: 하와이 몰로카이 섬, 출발지 카우나카카이 항구.
사전정보 = 2015 항공조사(가장 최근 조사라고 가정). MDMAP에서 '많던 곳이 계속
많다'(순위상관 0.88)를 확인했으므로 최근 조사를 다음 비행의 사전정보로 쓴다.

해안을 500 m 구간으로 나누고, 구간을 훑는 비용 = 500 m. 예산(비행거리)을 바꿔가며
중요도 가중 쓰레기를 얼마나 보는지 비교한다.
  - 핫스팟 우선(우리)
  - 가까운 곳부터
  - 해안선 따라 훑기: 출발지에서 해안을 따라 양방향으로 이어서 훑는 일반적 방식
"""
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "path_planning"))
from hotspot_route import Cell, plan_with_launch, route_cost, _two_opt  # noqa: E402
from windward_test import chip_lonlat, CENTERS, bearing  # noqa: E402

H = Path(__file__).parent
DATA = H / "data" / "imagery_and_labels"
ISLAND = "molokai"
BASE_LONLAT = (-157.0226, 21.0864)   # 카우나카카이 항구
CELL_M = 500.0
BATTERY_KM = 8.0   # 배터리 1개당 비행거리 가정값 (실측 필요)
# 중요도 가중치(가정값): 유령어구·부피 큰 것 우선
WEIGHT = {"buoy": 5, "net cloth": 5, "vessel": 5, "tire": 4, "line fragment": 3,
          "metal": 2, "processed wood": 2, "unidentified object": 1}


def main():
    lat0, lon0 = CENTERS[ISLAND]
    kx = 111320 * math.cos(math.radians(lat0))
    ky = 110540

    per_chip = defaultdict(float)
    raw = defaultdict(int)
    for f in ("training_data.csv", "evaluation_data.csv"):
        for r in csv.DictReader(open(DATA / f)):
            if r["filename"].startswith(ISLAND + "_"):
                per_chip[r["filename"]] += WEIGHT.get(r["label"], 1)
                raw[r["filename"]] += 1

    cellv = defaultdict(float)
    cellxy = defaultdict(list)
    for name, v in per_chip.items():
        lon, lat = chip_lonlat(name)
        x, y = (lon - lon0) * kx, (lat - lat0) * ky
        key = (round(x / CELL_M), round(y / CELL_M))
        cellv[key] += v
        cellxy[key].append((x, y, lon, lat))

    cells = []
    for key, v in cellv.items():
        pts = cellxy[key]
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        lon = sum(p[2] for p in pts) / len(pts)
        lat = sum(p[3] for p in pts) / len(pts)
        cells.append(Cell(id=str(key), x=x, y=y, mean=v, survey_cost=CELL_M,
                          meta={"lon": lon, "lat": lat, "bearing": bearing(lat0, lon0, lat, lon)}))
    base = ((BASE_LONLAT[0] - lon0) * kx, (BASE_LONLAT[1] - lat0) * ky)
    T = sum(c.mean for c in cells)
    print(f"{ISLAND}: 쓰레기 있는 500 m 구간 {len(cells)}개, 중요도 가중 합 {T:.0f}")


    def coast_sweep(bxy, B, exclude=()):
        # 띄운 곳에서 해안 방위 순서로 양옆을 번갈아 넓혀가며 이어서 훑기 (일반적 방식)
        bb = (math.degrees(math.atan2(bxy[0], bxy[1])) + 360) % 360
        idx = [i for i in sorted(range(len(cells)), key=lambda i: (cells[i].meta["bearing"] - bb) % 360)
               if i not in exclude]
        order, l, r = [], len(idx) - 1, 0
        while r <= l:
            order.append(idx[r]); r += 1
            if r <= l:
                order.append(idx[l]); l -= 1
        route = []
        for i in order:
            cand = _two_opt(route + [i], cells, bxy) if len(route) < 25 else route + [i]
            if route_cost(cand, cells, bxy) > B:
                break
            route = cand
        return route

    def cov(idx):
        return sum(cells[i].mean for i in idx) / T * 100

    B = BATTERY_KM * 1000
    cand = [(c.x, c.y) for c in cells]
    rng = random.Random(0)
    rows, plans_out = [], {}
    for n in (1, 2, 3, 4):
        # ① 항구에서 띄워 해안 따라 훑기 (비행마다 이어서)
        seen = set()
        for _ in range(n):
            seen |= set(coast_sweep(base, B, seen))
        harbor = cov(seen)
        # ② 아무 데서나 띄워 해안 따라 훑기 (무작위 30회 평균)
        rs = []
        for t in range(30):
            seen = set()
            for _ in range(n):
                b = rng.choice(cand)
                seen |= set(coast_sweep(b, B, seen))
            rs.append(cov(seen))
        rand_sweep = sum(rs) / len(rs)
        # ③ 위치 선택 + 핫스팟 우선 (우리)
        plans = plan_with_launch(cells, cand, B, n_sorties=n, alphas=(1.0, 2.0))
        ours = cov(set(i for p in plans for i in p["route"]))
        rows.append({"sorties": n, "harbor_sweep": harbor, "random_sweep": rand_sweep, "ours": ours})
        plans_out[n] = plans
        print(f"배터리 {n}개 ({BATTERY_KM} km씩)  항구+해안따라 {harbor:5.1f}%  아무데서+해안따라 {rand_sweep:5.1f}%  "
              f"위치선택+핫스팟(우리) {ours:5.1f}%")

    def lonlat(xy):
        return (xy[0] / kx + lon0, xy[1] / ky + lat0)

    (H / "exp_hawaii_drone.json").write_text(json.dumps({
        "island": ISLAND, "base": BASE_LONLAT, "battery_km": BATTERY_KM, "rows": rows,
        "cells": [{"lon": c.meta["lon"], "lat": c.meta["lat"], "v": c.mean} for c in cells],
        "plans": {str(k): [{"launch": lonlat(p["launch"]), "route": p["route"]} for p in v] for k, v in plans_out.items()},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
