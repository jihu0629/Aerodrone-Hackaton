"""
핫스팟 우선 경로 — 예산(배터리 거리) 안에서 "많이, 중요한 것을" 보는 순서.

문제 정의: 오리엔티어링 문제(Orienteering Problem). 후보 구간마다 가치가 있고,
출발지에서 출발해 출발지로 돌아오는 경로의 총 비용이 예산 B를 넘지 않으면서
방문한 구간 가치의 합을 최대화한다. NP-hard라 휴리스틱으로 푼다
(가치/비용 비 기준 삽입 → 2-opt로 경로 단축 → 남은 예산으로 추가 삽입 반복).

구간 가치 = 기대 쓰레기량 × 중요도 + 탐색 보너스
  - 기대 쓰레기량: 과거 조사 평균. hotspot/ 분석에서 순위상관 0.88로 유지됨을 확인.
  - 중요도: 무겁거나 치우기 어려운 것(어망·로프·부표·타이어 등)에 가중치.
    개수만 많은 작은 조각보다 트럭·인력 계획을 바꾸는 물체를 우선 본다.
  - 탐색 보너스: 조사 횟수가 적은 구간에 불확실성만큼 가산(UCB 방식).
    과거에 많았던 곳만 계속 보면 새로 생긴 핫스팟을 영영 못 찾기 때문.
"""
import math
import random
from dataclasses import dataclass, field


@dataclass
class Cell:
    id: str
    x: float              # m
    y: float              # m
    mean: float           # 과거 조사 기준 중요도 가중 기대량
    std: float = 0.0      # 과거 조사 간 표준편차
    n_obs: int = 1        # 과거 조사 횟수 (0이면 미조사)
    survey_cost: float = 0.0  # 그 구간을 훑는 데 드는 비용(m), 예: 해안 구간 길이
    meta: dict = field(default_factory=dict)


def cell_value(c: Cell, explore: float = 0.0, prior_mean: float = 0.0) -> float:
    if c.n_obs == 0:
        return prior_mean + explore * prior_mean
    return c.mean + explore * c.std / math.sqrt(c.n_obs)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_cost(route, cells, base):
    pts = [base] + [(cells[i].x, cells[i].y) for i in route] + [base]
    return sum(_d(p, q) for p, q in zip(pts, pts[1:])) + sum(cells[i].survey_cost for i in route)


def _two_opt(route, cells, base):
    best = route[:]
    best_c = route_cost(best, cells, base)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = route_cost(cand, cells, base)
                if c < best_c - 1e-9:
                    best, best_c, improved = cand, c, True
    return best


def _greedy(cells, base, budget, values, alpha, route=None):
    route = list(route or [])
    remaining = set(i for i in range(len(cells)) if values[i] > 0) - set(route)
    while True:
        cur = route_cost(route, cells, base)
        best = None
        for i in remaining:
            for pos in range(len(route) + 1):
                cand = route[:pos] + [i] + route[pos:]
                extra = route_cost(cand, cells, base) - cur
                if cur + extra > budget:
                    continue
                ratio = values[i] ** alpha / max(extra, 1e-6)
                if best is None or ratio > best[0]:
                    best = (ratio, i, pos)
        if best is None:
            shorter = _two_opt(route, cells, base)
            if route_cost(shorter, cells, base) < route_cost(route, cells, base) - 1e-6:
                route = shorter
                continue
            return route
        _, i, pos = best
        route.insert(pos, i)
        remaining.discard(i)


def _total(route, values):
    return sum(values[i] for i in route)


def plan_route(cells, base, budget, values=None, explore=0.0, alphas=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """예산 안에서 가치 합을 최대화하는 방문 순서(셀 인덱스 리스트)를 반환.

    가치/비용 비 삽입을 가치 지수(alpha)를 바꿔가며 여러 번 돌리고(작으면 가까운
    곳 위주, 크면 큰 핫스팟 위주), 각각을 '하나 빼고 다시 채우기' 지역 탐색으로
    다듬은 뒤 가장 좋은 경로를 고른다.
    """
    if values is None:
        known = [c.mean for c in cells if c.n_obs > 0]
        prior = sum(known) / len(known) if known else 0.0
        values = [cell_value(c, explore, prior) for c in cells]

    best = []
    for a in alphas:
        route = _two_opt(_greedy(cells, base, budget, values, a), cells, base)
        improved = True
        while improved:
            improved = False
            for k in range(len(route)):
                trial = _greedy(cells, base, budget, values, a, route[:k] + route[k + 1:])
                if _total(trial, values) > _total(route, values) + 1e-9:
                    route, improved = _two_opt(trial, cells, base), True
                    break
        if _total(route, values) > _total(best, values):
            best = route
    return best


def plan_sorties(cells, base, budget, n_sorties, **kw):
    """배터리 여러 개: 한 번 비행 경로를 짜고 방문한 구간을 빼고 다시 짠다."""
    left = list(range(len(cells)))
    out = []
    for _ in range(n_sorties):
        sub = [cells[i] for i in left]
        r = plan_route(sub, base, budget, **kw)
        if not r:
            break
        out.append([left[i] for i in r])
        chosen = set(left[i] for i in r)
        left = [i for i in left if i not in chosen]
    return out


def plan_with_launch(cells, candidates, budget, n_sorties=1, **kw):
    """배터리 1개당 비행거리 예산이 있을 때, 띄울 위치(차량 접근 후보)와 경로를 같이 고른다.

    드론은 항속거리가 짧아서 '어디서 띄우느냐'가 '어떤 순서로 보느냐'보다 결과를
    더 크게 바꾼다. 비행마다 후보 위치 전부에 대해 경로를 짜 보고 가치가 가장 큰
    (위치, 경로)를 고른 뒤, 본 구간을 빼고 다음 비행을 짠다.
    """
    left = list(range(len(cells)))
    plans = []
    for _ in range(n_sorties):
        sub = [cells[i] for i in left]
        vals = [c.mean for c in sub]
        best = None
        for b in candidates:
            r = plan_route(sub, b, budget, values=vals, **kw)
            v = sum(vals[i] for i in r)
            if best is None or v > best[0]:
                best = (v, b, r)
        if best is None or not best[2]:
            break
        _, b, r = best
        chosen = [left[i] for i in r]
        plans.append({"launch": b, "route": chosen})
        left = [i for i in left if i not in set(chosen)]
    return plans


# ---- 비교 기준 ------------------------------------------------------------

def nearest_first(cells, base, budget):
    """가까운 곳부터 차례로 — 가치를 안 보고 거리만 본다."""
    route, pos, left = [], base, set(range(len(cells)))
    while left:
        i = min(left, key=lambda k: _d(pos, (cells[k].x, cells[k].y)))
        if route_cost(route + [i], cells, base) > budget:
            break
        route.append(i)
        left.discard(i)
        pos = (cells[i].x, cells[i].y)
    return route


def random_order(cells, base, budget, seed=0):
    rng = random.Random(seed)
    order = list(range(len(cells)))
    rng.shuffle(order)
    route = []
    for i in order:
        cand = _two_opt(route + [i], cells, base) if len(route) < 30 else route + [i]
        if route_cost(cand, cells, base) <= budget:
            route = cand
    return route


def full_tour_cost(cells, base):
    """전부 도는 데 드는 비용(최근접 순서 + 2-opt) — 예산 비율의 기준."""
    route, pos, left = [], base, set(range(len(cells)))
    while left:
        i = min(left, key=lambda k: _d(pos, (cells[k].x, cells[k].y)))
        route.append(i)
        left.discard(i)
        pos = (cells[i].x, cells[i].y)
    if len(route) <= 120:
        route = _two_opt(route, cells, base)
    return route_cost(route, cells, base)
