# 드론 조사 경로 — 핫스팟 우선(C) + 커버리지(A)

해안선 전체를 드론으로 매번 다 도는 건 배터리·인력상 현실적이지 않다. 그래서
두 단계로 나눈다.

1. **어디를 날지 고르기** (`hotspot_route.py`)
   과거 조사·위성으로 구간마다 기대 쓰레기량을 매기고, 배터리 1개(또는 여러 개)의
   비행거리 예산 안에서 가치 합이 최대가 되는 구간과 순서를 고른다
   (오리엔티어링 문제, 가치/비용 비 삽입 + 2-opt). 띄울 위치(차량 접근 후보)도
   같이 고를 수 있다(`plan_with_launch`). 과거에 많았던 곳만 계속 보면 새
   핫스팟을 못 찾으므로, 조사 횟수가 적은 구간에 탐색 보너스(UCB)를 준다.
2. **고른 구간을 빠짐없이 촬영** (`coverage.py`)
   카메라 화각·고도·오버랩으로 줄 간격을 계산한 지그재그 경로.

밀집 구간이 실제로 있고 시간이 지나도 유지된다는 근거는 `../hotspot/README.md`
(하와이 항공조사: 상위 10% 격자에 라벨 75%, NOAA MDMAP 반복조사: 순위상관 0.88).

## 사용

```python
import sys; sys.path.insert(0, "path_planning")
from hotspot_route import Cell, plan_route, full_tour_cost
route = plan_route(cells, base=(0, 0), budget=0.3 * full_tour_cost(cells, (0, 0)))
```

실험 스크립트: `../hotspot/exp_hawaii_drone.py`, `../hotspot/exp_texas.py`.

## 이전 버전

비행 중 물체를 만나면 이탈해 선회하는 경로(`orbit.py`)와 그 통합 미션은 범위에서
빼면서 삭제했다(커밋 `cffc25f`의 `archive/path_planning_v1/`). 수거 경로는 지형
최단경로를 쓰는 `../shoresweep_planner/`가 맡는다.
