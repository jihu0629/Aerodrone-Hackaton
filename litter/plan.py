"""
수거 계획 — 물체별 인력 판단 · 정거장 묶기 · 격자 히트맵 · 자재/차량 · 경로.

인력 판단은 **예측구간 상한**으로 한다 (평균이 아니라 최악의 경우):
    상한 < 23 kg (NIOSH)          → 1인
    23 kg ≤ 상한 < 50 kg          → 2인
    상한 ≥ 50 kg                  → 장비
정거장: stop_radius_m 안의 물체를 한 번에 수거하는 단위로 묶는다.
경로: 집하장에서 출발 → 최근접 이웃 + 2-opt (정거장 수백 개까지 충분).
"""
import math

import numpy as np


def crew(hi_kg, cfg):
    p = cfg["plan"]
    if hi_kg >= p["equip_limit_kg"]:
        return "장비"
    if hi_kg >= p["lift_limit_kg"]:
        return "2인"
    return "1인"


def _cluster(xy, r, kg):
    """무거운 물체부터 중심으로 잡고 반경 r 안의 미배정 물체를 묶는다.
    (연쇄 결합이면 쓰레기가 촘촘한 해변 전체가 한 덩어리가 된다)"""
    lab = np.full(len(xy), -1)
    for k, i in enumerate(np.argsort(-np.asarray(kg))):
        if lab[i] >= 0:
            continue
        near = (np.linalg.norm(xy - xy[i], axis=1) < r) & (lab < 0)
        lab[near] = k
    return lab.tolist()


def _route(pts, start):
    """최근접 이웃으로 시작해 2-opt로 다듬은 순회 순서 (집하장=start에서 출발·복귀)."""
    n = len(pts)
    if n == 0:
        return [], 0.0
    P = np.vstack([start, pts])
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    order, left = [0], set(range(1, n + 1))
    while left:
        j = min(left, key=lambda k: D[order[-1], k])
        order.append(j)
        left.remove(j)
    order.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(order) - 2):
            for j in range(i + 1, len(order) - 1):
                a, b, c, d = order[i - 1], order[i], order[j], order[j + 1]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    length = float(sum(D[order[k], order[k + 1]] for k in range(len(order) - 1)))
    return [k - 1 for k in order[1:-1]], length


def make_plan(items, cfg):
    p = cfg["plan"]
    for it in items:
        it["crew"] = crew(it["weight_hi_kg"], cfg)
        tools = []
        if p["tools"].get(it["cls"]):
            tools.append(p["tools"][it["cls"]])
        if it.get("buried_suspect"):
            tools.append("삽")
        it["tools"] = tools
    xy = np.array([[it["x"], it["y"]] for it in items])
    lab = _cluster(xy, p["stop_radius_m"], [it["weight_est_kg"] for it in items])
    stops = {}
    for it, l in zip(items, lab):
        stops.setdefault(l, []).append(it)
    rank = {"1인": 0, "2인": 1, "장비": 2}
    S = []
    for k, g in enumerate(stops.values()):
        gxy = np.array([[i["x"], i["y"]] for i in g])
        S.append({
            "stop_id": f"S{k + 1:03d}",
            "x": float(gxy[:, 0].mean()), "y": float(gxy[:, 1].mean()),
            "lat": float(np.mean([i["lat"] for i in g])), "lon": float(np.mean([i["lon"] for i in g])),
            "n_items": len(g),
            "kg_est": float(sum(i["weight_est_kg"] for i in g)),
            "kg_hi": float(sum(i["weight_hi_kg"] for i in g)),
            "bulk_m3": float(sum(i.get("bulk_m3") or 0 for i in g)),
            "crew": max((i["crew"] for i in g), key=rank.get),
            "tools": sorted({t for i in g for t in i["tools"]}),
            "max_item_kg_hi": float(max(i["weight_hi_kg"] for i in g)),
            "r95_m": float(max(i.get("r95_m", 0) for i in g)),
            "buried_suspect": any(i.get("buried_suspect") for i in g),
            "classes": sorted({i["cls"] for i in g}),
            "item_ids": [i["item_id"] for i in g],
        })
        for i in g:
            i["stop_id"] = S[-1]["stop_id"]
    # 경로
    pts = np.array([[s["x"], s["y"]] for s in S])
    depot = np.array(p["depot"]) if p["depot"] else pts[np.argmin(pts[:, 1])] - [0, 10]
    order, dist = _route(pts, depot)
    for rank_i, k in enumerate(order):
        S[k]["visit_order"] = rank_i + 1
    S.sort(key=lambda s: s["visit_order"])
    # 격자
    g = p["grid_m"]
    grid = {}
    for it in items:
        key = (math.floor(it["x"] / g), math.floor(it["y"] / g))
        grid[key] = grid.get(key, 0.0) + it["weight_est_kg"]
    # 자재·차량: 무게와 부피 중 먼저 차는 쪽. 총량은 추정값 합으로 한다 —
    # 물체별 오차는 서로 상쇄되므로 "상한의 합"은 총량의 상한으로 너무 크다.
    kg = sum(it["weight_est_kg"] for it in items)
    kg_hi = sum(it["weight_hi_kg"] for it in items)
    m3 = sum(it.get("bulk_m3") or 0 for it in items)
    bags_by_kg, bags_by_vol = math.ceil(kg / p["bag_kg"]), math.ceil(m3 * 1000 / p["bag_l"])
    trucks_by_kg, trucks_by_vol = math.ceil(kg / p["truck_kg"]), math.ceil(m3 / p["truck_m3"])
    summary = {
        "n_items": len(items), "n_stops": len(S),
        "kg_est": kg, "kg_hi": kg_hi, "bulk_m3": m3,
        "crew_counts": {c: sum(1 for it in items if it["crew"] == c) for c in ("1인", "2인", "장비")},
        "bags": max(bags_by_kg, bags_by_vol), "bags_by_kg": bags_by_kg, "bags_by_vol": bags_by_vol,
        "trucks": max(trucks_by_kg, trucks_by_vol, 1 if items else 0),
        "trucks_by_kg": trucks_by_kg, "trucks_by_vol": trucks_by_vol,
        "route_m": dist, "depot": depot.tolist(),
        "grid_m": g, "grid": [{"gx": k[0], "gy": k[1], "kg": v} for k, v in sorted(grid.items(), key=lambda t: -t[1])],
    }
    return S, summary
