"""수거 계획: 인력·마대·차량·경로.

⚠ 출처가 있는 값
  - 1인 들기 권장 한계 23 kg: NIOSH Revised Lifting Equation (https://stacks.cdc.gov/view/cdc/110725)
⚠ 나머지 적재량·작업 속도는 전부 가정값. 지자체·해양환경공단 기준으로 교체해야 한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .mass import ObjectMass
from .gridmap import object_xy_m

NIOSH_LIFT_LIMIT_KG = 23.0


@dataclass
class PlanParams:
    bag_kg: float = 15.0          # 가정값: 마대 1장 적재 무게
    bag_m3: float = 0.08          # 가정값: 마대 1장 부피 (80 L)
    tonbag_kg: float = 500.0      # 가정값: 톤백 1개 (안전 적재)
    tonbag_m3: float = 1.0        # 가정값
    truck_kg: float = 1000.0      # 가정값: 1 t 트럭
    truck_m3: float = 3.0         # 가정값: 적재함 부피
    worker_kg_per_hour: float = 40.0   # 가정값: 1인 시간당 수거·운반 무게
    worker_bags_per_hour: float = 6.0  # 가정값: 1인 시간당 마대 수 (부피 제한 쓰레기용)
    bulk_factor: float = 1.3      # 가정값: 담을 때 빈틈으로 부피가 늘어나는 배수
    depot_xy: tuple[float, float] | None = None   # 집결지(m). None 이면 무게 중심
    vehicle_capacity_kg: float = 1000.0            # 경로 계산용 차량 적재량 (가정값)
    n_vehicles: int = 2
    cell_m: float = 10.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_note"] = "bag/tonbag/truck/worker 값은 가정값. 23 kg 는 NIOSH."
        return d


@dataclass
class ClassPlan:
    class_name: str
    class_ko: str
    count: int
    kg_typ: float
    kg_min: float
    kg_max: float
    volume_m3: float
    bags: int
    limiting: str        # "weight" | "volume"
    heavy_items: int     # 23 kg 초과 → 2인 이상 또는 장비
    worker_hours: float


@dataclass
class CollectionPlan:
    total_kg: tuple[float, float, float]
    total_volume_m3: float
    total_bags: int
    tonbags: int
    truck_trips: int
    heavy_items: list[dict]
    worker_hours: float
    workers_for_4h: int
    by_class: list[ClassPlan]
    priority_cells: list[dict]
    routes: list[dict]
    params: dict
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_kg_min_typ_max": self.total_kg,
            "total_volume_m3": self.total_volume_m3,
            "total_bags": self.total_bags,
            "tonbags": self.tonbags,
            "truck_trips": self.truck_trips,
            "heavy_items": self.heavy_items,
            "worker_hours": self.worker_hours,
            "workers_for_4h": self.workers_for_4h,
            "by_class": [asdict(c) for c in self.by_class],
            "priority_cells": self.priority_cells,
            "routes": self.routes,
            "params": self.params,
            "assumptions": self.assumptions,
        }


def _bags_needed(kg: float, m3: float, p: PlanParams) -> tuple[int, str]:
    by_w = kg / p.bag_kg
    by_v = m3 * p.bulk_factor / p.bag_m3
    if by_v > by_w:
        return math.ceil(by_v), "volume"
    return math.ceil(by_w), "weight"


def make_plan(masses: list[ObjectMass], gsd_m: float, transform=None,
              params: PlanParams | None = None) -> CollectionPlan:
    p = params or PlanParams()
    litter = [m for m in masses if m.method != "excluded"]
    by_cls: dict[str, list[ObjectMass]] = {}
    for m in litter:
        by_cls.setdefault(m.class_name, []).append(m)

    class_plans: list[ClassPlan] = []
    total_bags = 0
    total_hours = 0.0
    heavy: list[dict] = []
    for cls, items in by_cls.items():
        kg = sum(i.kg_typ for i in items)
        m3 = sum(i.volume_m3 for i in items)
        bags, lim = _bags_needed(kg, m3, p)
        hv = [i for i in items if i.kg_typ > NIOSH_LIFT_LIMIT_KG]
        for i in hv:
            x, y = object_xy_m(i, gsd_m, transform)
            heavy.append({"obj_id": i.obj_id, "class": i.class_ko, "kg_typ": round(i.kg_typ, 1),
                          "x_m": round(x, 1), "y_m": round(y, 1),
                          "handling": "2인 이상 또는 장비 (NIOSH 23 kg 초과)"})
        hours = max(kg / p.worker_kg_per_hour, bags / p.worker_bags_per_hour)
        total_bags += bags
        total_hours += hours
        class_plans.append(ClassPlan(cls, items[0].class_ko, len(items), kg,
                                     sum(i.kg_min for i in items), sum(i.kg_max for i in items),
                                     m3, bags, lim, len(hv), hours))
    class_plans.sort(key=lambda c: -c.kg_typ)

    tk = (sum(m.kg_min for m in litter), sum(m.kg_typ for m in litter), sum(m.kg_max for m in litter))
    tv = sum(m.volume_m3 for m in litter)
    tonbags = math.ceil(max(tk[1] / p.tonbag_kg, tv * p.bulk_factor / p.tonbag_m3)) if litter else 0
    trips = math.ceil(max(tk[1] / p.truck_kg, tv * p.bulk_factor / p.truck_m3)) if litter else 0

    # 우선순위 격자
    from .gridmap import grid_kg
    g, meta = grid_kg(litter, gsd_m, p.cell_m, transform)
    cells = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] > 0:
                cells.append({"row": r, "col": c, "kg": round(float(g[r, c]), 2),
                              "x_m": meta["x0"] + (c + 0.5) * p.cell_m,
                              "y_m": meta["y0"] + (r + 0.5) * p.cell_m})
    cells.sort(key=lambda d: -d["kg"])

    routes = plan_routes(cells, p) if cells else []

    return CollectionPlan(
        total_kg=tk, total_volume_m3=tv, total_bags=total_bags, tonbags=tonbags, truck_trips=trips,
        heavy_items=heavy, worker_hours=total_hours, workers_for_4h=math.ceil(total_hours / 4) if litter else 0,
        by_class=class_plans, priority_cells=cells[:20], routes=routes, params=p.to_dict(),
        assumptions=[
            "마대·톤백·트럭 적재량, 작업 속도, 담기 부피 배수는 가정값 (현장 기준으로 교체 필요)",
            "1인 들기 한계 23 kg 는 NIOSH Revised Lifting Equation",
            "무게는 겉보기 밀도 대표값(kg_typ) 기준. 최소·최대 범위는 total_kg 참조",
            "검출 누락(재현율) 은 반영하지 않은 '최소 추정치'",
        ],
    )


def plan_routes(cells: list[dict], p: PlanParams) -> list[dict]:
    """격자 셀을 방문하는 적재량 제약 차량 경로 (CVRP).
    OR-Tools 가 있으면 사용, 없으면 최근접 이웃 + 적재량 분할 (greedy)."""
    if not cells:
        return []
    if p.depot_xy is None:
        w = sum(c["kg"] for c in cells)
        depot = (sum(c["x_m"] * c["kg"] for c in cells) / w, sum(c["y_m"] * c["kg"] for c in cells) / w)
    else:
        depot = p.depot_xy
    try:
        return _routes_ortools(cells, depot, p)
    except Exception:
        return _routes_greedy(cells, depot, p)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _routes_greedy(cells, depot, p: PlanParams) -> list[dict]:
    remaining = list(cells)
    routes = []
    while remaining:
        load, pos, route, length = 0.0, depot, [], 0.0
        while remaining:
            cand = [c for c in remaining if load + c["kg"] <= p.vehicle_capacity_kg] or ([remaining[0]] if not route else [])
            if not cand:
                break
            nxt = min(cand, key=lambda c: _dist(pos, (c["x_m"], c["y_m"])))
            length += _dist(pos, (nxt["x_m"], nxt["y_m"]))
            pos = (nxt["x_m"], nxt["y_m"])
            load += nxt["kg"]
            route.append({"row": nxt["row"], "col": nxt["col"], "kg": nxt["kg"]})
            remaining.remove(nxt)
        length += _dist(pos, depot)
        routes.append({"vehicle": len(routes) + 1, "load_kg": round(load, 1), "length_m": round(length, 1),
                       "stops": route, "solver": "greedy"})
    return routes


def _routes_ortools(cells, depot, p: PlanParams) -> list[dict]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: optional
    pts = [depot] + [(c["x_m"], c["y_m"]) for c in cells]
    demands = [0] + [int(math.ceil(c["kg"])) for c in cells]
    n_veh = max(p.n_vehicles, math.ceil(sum(demands) / p.vehicle_capacity_kg))
    mgr = pywrapcp.RoutingIndexManager(len(pts), n_veh, 0)
    rt = pywrapcp.RoutingModel(mgr)
    def dcb(i, j):
        return int(_dist(pts[mgr.IndexToNode(i)], pts[mgr.IndexToNode(j)]))
    tcb = rt.RegisterTransitCallback(dcb)
    rt.SetArcCostEvaluatorOfAllVehicles(tcb)
    dem = rt.RegisterUnaryTransitCallback(lambda i: demands[mgr.IndexToNode(i)])
    rt.AddDimensionWithVehicleCapacity(dem, 0, [int(p.vehicle_capacity_kg)] * n_veh, True, "cap")
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.time_limit.seconds = 5
    sol = rt.SolveWithParameters(sp)
    if sol is None:
        raise RuntimeError("no solution")
    routes = []
    for v in range(n_veh):
        idx = rt.Start(v)
        stops, load, length = [], 0.0, 0.0
        while not rt.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node != 0:
                c = cells[node - 1]
                stops.append({"row": c["row"], "col": c["col"], "kg": c["kg"]})
                load += c["kg"]
            nxt = sol.Value(rt.NextVar(idx))
            length += _dist(pts[node], pts[mgr.IndexToNode(nxt)])
            idx = nxt
        if stops:
            routes.append({"vehicle": len(routes) + 1, "load_kg": round(load, 1), "length_m": round(length, 1),
                           "stops": stops, "solver": "ortools"})
    return routes
