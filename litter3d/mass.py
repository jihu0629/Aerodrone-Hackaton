"""부피 → 무게. m = V × ρ_app(클래스), 최소/대표/최대 범위를 함께 낸다.

규칙
  1. 클래스가 is_litter=False (식생) → 제외.
  2. prefer_count 클래스(유리·금속·미확인) → 개당 평균무게(W1). 없으면 중앙값 13.4 g.
  3. 소형(면적 ≤ 5×5 cm 또는 최대 높이 ≤ 3 cm, DSM 으로 못 잼) →
     클래스에 small_item_g 가 있으면 그 값, 없으면 면적 × 0.4 cm × 1.2 g/cm³ (W3, Andriolo 2024 표 5 최적값).
     (레이로자 검증: 소형 조각 265개에 16 g 씩 주면 78 g 이 4,300 g 이 된다 → 면적 기반이 안전)
  4. 그 외 → 부피 × 겉보기 밀도. 젖음 계수(가정값) 로 최대값 확장.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .classes import (CLASSES, MEDIAN_ITEM_G, SMALL_OBJECT_MAX_AREA_M2, SMALL_OBJECT_MAX_HEIGHT_M, KO_NAMES,
                      SMALL_W3_THICKNESS_CM, SMALL_W3_SPECIFIC_WEIGHT)
from .volume import ObjectVolume


@dataclass
class ObjectMass:
    obj_id: int
    class_name: str
    class_ko: str
    method: str            # "volume" | "count" | "excluded"
    area_m2: float
    volume_m3: float
    h_max_m: float
    kg_min: float
    kg_typ: float
    kg_max: float
    cx_px: float
    cy_px: float
    confidence: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_mass(ov: ObjectVolume, wet: bool = False) -> ObjectMass:
    spec = CLASSES.get(ov.class_name, CLASSES["unknown"])
    ko = KO_NAMES.get(ov.class_name, ov.class_name)
    base = dict(obj_id=ov.obj_id, class_name=ov.class_name, class_ko=ko, area_m2=ov.area_m2,
                volume_m3=ov.volume_m3, h_max_m=ov.h_max_m, cx_px=ov.cx_px, cy_px=ov.cy_px,
                confidence=ov.confidence)
    if not spec.is_litter:
        return ObjectMass(method="excluded", kg_min=0, kg_typ=0, kg_max=0, note="쓰레기 아님", **base)

    small = ov.area_m2 <= SMALL_OBJECT_MAX_AREA_M2 or ov.h_max_m <= SMALL_OBJECT_MAX_HEIGHT_M
    if spec.prefer_count:
        g = spec.mean_item_g if spec.mean_item_g is not None else MEDIAN_ITEM_G
        # 개당 무게의 불확실성: 0.5–2배 (가정값)
        return ObjectMass(method="count", kg_min=g * 0.5 / 1000, kg_typ=g / 1000, kg_max=g * 2 / 1000,
                          note=f"속 빈/미확인 → 개수×평균무게 ({g} g)", **base)
    if small:
        if spec.small_item_g is not None:
            g = spec.small_item_g
            note = f"소형 → 개수×클래스 소형무게 ({g} g)"
        else:
            g = ov.area_m2 * 1e4 * SMALL_W3_THICKNESS_CM * SMALL_W3_SPECIFIC_WEIGHT
            note = f"소형 → 면적×{SMALL_W3_THICKNESS_CM} cm×{SMALL_W3_SPECIFIC_WEIGHT} g/cm³ (W3)"
        return ObjectMass(method="count", kg_min=g * 0.5 / 1000, kg_typ=g / 1000, kg_max=g * 2 / 1000,
                          note=note, **base)

    v_lo = max(ov.volume_m3 - ov.volume_sigma_m3, 0.0)
    v_hi = ov.volume_m3 + ov.volume_sigma_m3
    wet_max = spec.wet_factor_max if wet else 1.0
    return ObjectMass(method="volume",
                      kg_min=v_lo * spec.rho_min,
                      kg_typ=ov.volume_m3 * spec.rho_typ,
                      kg_max=v_hi * spec.rho_max * wet_max,
                      note=f"ρ {spec.rho_min}/{spec.rho_typ}/{spec.rho_max} kg/m³; {spec.source}", **base)


def estimate_masses(vols: list[ObjectVolume], wet: bool = False) -> list[ObjectMass]:
    return [estimate_mass(v, wet=wet) for v in vols]


def summarize_by_class(masses: list[ObjectMass]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in masses:
        d = out.setdefault(m.class_name, {"class_ko": m.class_ko, "count": 0, "volume_m3": 0.0,
                                          "kg_min": 0.0, "kg_typ": 0.0, "kg_max": 0.0})
        d["count"] += 1
        d["volume_m3"] += m.volume_m3
        d["kg_min"] += m.kg_min
        d["kg_typ"] += m.kg_typ
        d["kg_max"] += m.kg_max
    return out


def total_kg(masses: list[ObjectMass]) -> tuple[float, float, float]:
    return (sum(m.kg_min for m in masses), sum(m.kg_typ for m in masses), sum(m.kg_max for m in masses))
