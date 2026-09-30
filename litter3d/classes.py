"""쓰레기 클래스, 겉보기 밀도표, 개당 평균무게표.

겉보기 밀도(apparent density): 3D 로 잰 '겉 부피' 로 나눈 무게. 속이 빈 PET 병·부표는 재료 밀도(PET 1.38 g/cm³)
보다 훨씬 작다. 재료 밀도를 그대로 곱하면 수십 배 과대추정된다.

⚠ 표기 규칙
  source="가정값"  → 확정값 아님. 보정 실험(클래스별 실물 5–10개 저울 + 3D 부피)으로 교체해야 함.
  단위: kg/m³ (= g/L). 1 g/cm³ = 1000 kg/m³.
클래스 이름은 AI Hub 「해안 오염물질 데이터」 12종 + unknown(미확인) 을 기본으로 한다. 기업 데이터 클래스에 맞춰 map_class() 로 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DensitySpec:
    rho_min: float       # kg/m³
    rho_typ: float
    rho_max: float
    source: str
    hollow: bool = False           # 속이 빈 물체 → 부피 기반 추정 신뢰도 낮음
    prefer_count: bool = False     # 개수×평균무게(W1) 방식이 더 안정적인 클래스
    mean_item_g: float | None = None   # 개당 평균무게(g), W1 대체값
    item_source: str = ""
    wet_factor_max: float = 1.0    # 젖었을 때 무게 배수 상한 (가정값)
    is_litter: bool = True


CLASSES: dict[str, DensitySpec] = {
    # --- 스티로폼: 발포체가 꽉 차 있어 재료 밀도 ≈ 겉보기 밀도. 가장 신뢰할 수 있는 클래스 ---
    "styrofoam_buoy": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                  mean_item_g=44.2, item_source="Andriolo 2024, 스티로폼 >50 cm",
                                  wet_factor_max=1.5),
    "styrofoam_box": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                 wet_factor_max=1.5),
    "styrofoam_fragment": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                      mean_item_g=3.1, item_source="Andriolo 2024, 조각 2.5–5 cm",
                                      wet_factor_max=1.5),
    # --- 속이 빈 플라스틱: 겉보기 밀도는 가정값. 보정 실험 필수 ---
    "pet_bottle": DensitySpec(30, 60, 300, "가정값 (빈 500 mL PET 약 20 g / 0.55 L ≈ 36 kg/m³; 모래·물이 차면 증가)",
                              hollow=True, mean_item_g=42.3, item_source="Andriolo 2024, 페트병·용기",
                              wet_factor_max=2.0),
    "plastic_buoy": DensitySpec(40, 100, 300, "가정값 (속 빈 HDPE 부표)", hollow=True,
                                mean_item_g=19.8, item_source="Andriolo 2024, 부표"),
    "chinese_buoy": DensitySpec(40, 100, 300, "가정값 (plastic_buoy 와 동일 취급)", hollow=True,
                                mean_item_g=19.8, item_source="Andriolo 2024, 부표"),
    "other_plastic": DensitySpec(80, 200, 500, "가정값 (혼합 플라스틱 폐기물 벌크 밀도; Andriolo 2024 는 재료 비중 0.8–1.5 g/cm³ 사용)",
                                 hollow=True, mean_item_g=16.4, item_source="Andriolo 2024 전체 평균 16.4 g",
                                 wet_factor_max=1.3),
    # --- 로프·그물: 재료(PE/PP ≈ 0.9–1.0 g/cm³) 은 물보다 가볍지만 뭉치 안에 빈틈이 많음 ---
    "rope": DensitySpec(150, 300, 600, "가정값 (PP 로프 재료 ≈ 910 kg/m³ × 채움률 0.2–0.6)",
                        mean_item_g=111.0, item_source="Andriolo 2024, 굵은 로프", wet_factor_max=1.5),
    "net": DensitySpec(80, 200, 500, "가정값 (그물 뭉치 채움률 0.1–0.5)",
                       mean_item_g=353.0, item_source="Andriolo 2024, 50 cm 이상 그물", wet_factor_max=2.0),
    # --- 유리·금속: 병·캔은 속이 빔. 개당 평균무게 방식을 우선 ---
    "glass": DensitySpec(200, 400, 800, "가정값 (유리병 겉보기)", hollow=True, prefer_count=True,
                         mean_item_g=165.8, item_source="Andriolo 2024, 유리병"),
    "metal": DensitySpec(50, 150, 400, "가정값 (캔·금속 조각 겉보기)", hollow=True, prefer_count=True,
                         mean_item_g=15.0, item_source="가정값 (알루미늄 캔 약 15 g)"),
    # --- 식생: 쓰레기 아님. 무게 집계에서 제외 ---
    "vegetation": DensitySpec(0, 0, 0, "쓰레기 아님", is_litter=False),
    # --- 미확인: 개수 × 중앙값 ---
    "unknown": DensitySpec(80, 200, 500, "가정값 (other_plastic 과 동일)", prefer_count=True,
                           mean_item_g=13.4, item_source="Andriolo & Gonçalves 2024, 693개 해변 중앙값 13.4 g"),
}

# 3D 로 부피를 잴 수 없는 소형 물체 기준 (가정값). 이보다 작으면 개수 × 평균무게 로 대체.
SMALL_OBJECT_MAX_AREA_M2 = 0.0025   # 5 cm × 5 cm
SMALL_OBJECT_MAX_HEIGHT_M = 0.03    # DSM 노이즈 수준 (Kako 2020: 5 cm 이하 얇은 물체는 경사로 못 잡음)
MEDIAN_ITEM_G = 13.4                # Andriolo & Gonçalves 2024 중앙값

# 한국어 표시명
KO_NAMES = {
    "styrofoam_buoy": "스티로폼 부표", "styrofoam_box": "스티로폼 박스", "styrofoam_fragment": "스티로폼 조각",
    "pet_bottle": "PET병", "plastic_buoy": "플라스틱 부표", "chinese_buoy": "중국산 부표",
    "other_plastic": "기타 플라스틱", "rope": "로프", "net": "어망", "glass": "유리", "metal": "금속",
    "vegetation": "식생", "unknown": "미확인",
}

# 다른 데이터셋 라벨 → 우리 클래스. 기업 데이터 클래스명을 받으면 여기에 추가.
_ALIASES = {
    "스티로폼 부표": "styrofoam_buoy", "스티로폼부표": "styrofoam_buoy",
    "스티로폼 박스": "styrofoam_box", "스티로폼박스": "styrofoam_box",
    "스티로폼 조각": "styrofoam_fragment", "스티로폼조각": "styrofoam_fragment", "styrofoam": "styrofoam_fragment",
    "pet": "pet_bottle", "페트": "pet_bottle", "페트병": "pet_bottle", "bottle": "pet_bottle",
    "플라스틱 부표": "plastic_buoy", "부표": "plastic_buoy", "buoy": "plastic_buoy",
    "중국산 부표": "chinese_buoy", "중국산부표": "chinese_buoy",
    "기타 플라스틱": "other_plastic", "플라스틱": "other_plastic", "plastic": "other_plastic",
    "로프": "rope", "밧줄": "rope", "그물": "net", "어망": "net", "fishing net": "net",
    "유리": "glass", "금속": "metal", "can": "metal", "캔": "metal",
    "식생": "vegetation", "해초": "vegetation", "vegetation": "vegetation",
    "기타": "unknown", "미확인": "unknown", "rubbish": "unknown", "trash": "unknown", "litter": "unknown",
}


def map_class(name: str) -> str:
    """임의의 라벨 이름을 우리 클래스 키로 바꾼다. 못 찾으면 'unknown'."""
    if name in CLASSES:
        return name
    k = name.strip().lower()
    for alias, target in _ALIASES.items():
        if alias.lower() == k:
            return target
    for alias, target in _ALIASES.items():
        if alias.lower() in k:
            return target
    return "unknown"


def class_names() -> list[str]:
    return list(CLASSES.keys())


def density_table_markdown() -> str:
    rows = ["| 클래스 | ρ_min | ρ_typ | ρ_max (kg/m³) | 개당(g) | 출처 |", "|---|---|---|---|---|---|"]
    for k, s in CLASSES.items():
        rows.append(f"| {KO_NAMES.get(k, k)} ({k}) | {s.rho_min} | {s.rho_typ} | {s.rho_max} | "
                    f"{'' if s.mean_item_g is None else s.mean_item_g} | {s.source}"
                    f"{'; 개당: ' + s.item_source if s.item_source else ''} |")
    return "\n".join(rows)
