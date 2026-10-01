"""수거 계획 공통 파라미터 (마대·톤백·트럭 적재량, 1인 들기 한계).

⚠ 출처가 있는 값
  - 1인 들기 권장 한계 23 kg: NIOSH Revised Lifting Equation (https://stacks.cdc.gov/view/cdc/110725)
⚠ 나머지 적재량·작업 속도는 전부 가정값. 지자체·해양환경공단 기준으로 교체해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

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
    worker_bags_per_hour: float = 6.0  # 가정값: 1인 시간당 마대 수
    bulk_factor: float = 1.3      # 가정값: 담을 때 빈틈으로 부피가 늘어나는 배수
    depot_xy: tuple[float, float] | None = None
    vehicle_capacity_kg: float = 1000.0
    n_vehicles: int = 2
    cell_m: float = 10.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_note"] = "bag/tonbag/truck/worker 값은 가정값. 23 kg 는 NIOSH."
        return d
