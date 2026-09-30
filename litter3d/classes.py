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
    mean_item_g: float | None = None   # 개당 평균무게(g), W1 대체값 (prefer_count 클래스용)
    item_source: str = ""
    small_item_g: float | None = None  # 3D 로 못 재는 소형 물체 개당 무게(g). None 이면 면적×0.4 cm×1.2 g/cm³ (W3)
    wet_factor_max: float = 1.0    # 젖었을 때 무게 배수 상한 (가정값)
    is_litter: bool = True


CLASSES: dict[str, DensitySpec] = {
    # --- 스티로폼: 발포체가 꽉 차 있어 재료 밀도 ≈ 겉보기 밀도. 가장 신뢰할 수 있는 클래스 ---
    "styrofoam_buoy": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                  mean_item_g=44.2, item_source="Andriolo 2024, 스티로폼 >50 cm",
                                  small_item_g=3.1, wet_factor_max=1.5),
    "styrofoam_box": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                 small_item_g=3.1, wet_factor_max=1.5),
    "styrofoam_fragment": DensitySpec(11, 20, 32, "EPS 11–32 kg/m³ (Wikipedia 'Polystyrene')",
                                      mean_item_g=3.1, item_source="Andriolo 2024, 조각 2.5–5 cm",
                                      small_item_g=3.1, wet_factor_max=1.5),
    # --- 속이 빈 플라스틱 ---
    # 겉보기 밀도는 가정값. 근거: Andriolo 2024 실측 평균무게(페트병 42 g, 식품용기 20 g, 봉지 17 g)를 일반적인 겉 부피
    # (누운 1 L 병 ≈ 1,200 cm³, 용기 ≈ 800 cm³, 구겨진 봉지 ≈ 400 cm³)로 나누면 25–40 kg/m³ 수준.
    # 모래·물이 차면 수 배로 늘어남 → 최대값을 넓게 둔다. 보정 실험으로 교체해야 한다.
    "pet_bottle": DensitySpec(20, 35, 150, "가정값 (Andriolo 2024 평균 42 g ÷ 누운 병 겉부피 ≈ 1.2 L → 약 35 kg/m³; 모래 차면 증가)",
                              hollow=True, mean_item_g=42.3, item_source="Andriolo 2024, 페트병·용기",
                              wet_factor_max=2.0),
    "plastic_buoy": DensitySpec(30, 70, 200, "가정값 (속 빈 HDPE 부표; Andriolo 2024 소형 부표 19.8 g)", hollow=True,
                                mean_item_g=19.8, item_source="Andriolo 2024, 부표"),
    "chinese_buoy": DensitySpec(30, 70, 200, "가정값 (plastic_buoy 와 동일 취급)", hollow=True,
                                mean_item_g=19.8, item_source="Andriolo 2024, 부표"),
    "other_plastic": DensitySpec(20, 50, 300, "가정값 (용기·봉지·시트 등 속 빈 물체 위주 25–40 kg/m³, 꽉 찬 조각은 그 이상)",
                                 hollow=True, mean_item_g=16.4, item_source="Andriolo 2024 전체 평균 16.4 g",
                                 wet_factor_max=1.3),
    # --- 로프·그물: 재료(PE/PP ≈ 0.9–1.0 g/cm³) 은 물보다 가볍고 뭉치 안에 빈틈이 대부분 ---
    # Andriolo 2024: 50 cm 이상 그물 353 g/개, 굵은 로프 111 g/개. 느슨한 뭉치(60×40×8 cm) 로 보면 30–60 kg/m³.
    "rope": DensitySpec(20, 60, 300, "가정값 (PP 로프 재료 ≈ 910 kg/m³ × 뭉치 채움률 0.02–0.3)",
                        mean_item_g=111.0, item_source="Andriolo 2024, 굵은 로프", wet_factor_max=1.5),
    "net": DensitySpec(15, 40, 200, "가정값 (그물 뭉치 채움률 0.02–0.2; Andriolo 2024 그물 353 g/개)",
                       mean_item_g=353.0, item_source="Andriolo 2024, 50 cm 이상 그물", wet_factor_max=2.0),
    # --- 유리·금속: 병·캔은 속이 빔. 개당 평균무게 방식을 우선 ---
    "glass": DensitySpec(150, 300, 800, "가정값 (유리병 겉보기)", hollow=True, prefer_count=True,
                         mean_item_g=165.8, item_source="Andriolo 2024, 유리병"),
    "metal": DensitySpec(50, 150, 400, "가정값 (캔·금속 조각 겉보기)", hollow=True, prefer_count=True,
                         mean_item_g=30.0, item_source="가정값 (Andriolo 2024 금속 13개 413 g ≈ 32 g/개)"),
    # --- 나무·섬유·고무·도자기: Andriolo 2024 에서 무게의 약 30 % (나무 17 %, 섬유 7 %) 를 차지 → 클래스 필요 ---
    "wood": DensitySpec(300, 500, 800, "가정값 (젖은 유목·가공목재 겉보기; 목재 밀도 0.4–0.7 g/cm³ 일반값)",
                        mean_item_g=708.0, item_source="Andriolo 2024, 나무 6개 4,259 g", wet_factor_max=1.3),
    "cloth": DensitySpec(30, 80, 300, "가정값 (젖은 옷·신발 뭉치 겉보기)",
                         mean_item_g=61.0, item_source="Andriolo 2024, 섬유·신발 29개 1,766 g", wet_factor_max=2.0),
    "rubber": DensitySpec(300, 700, 1200, "가정값 (고무 조각·타이어 겉보기; 고무 재료 1.1–1.2 g/cm³)",
                          mean_item_g=278.0, item_source="Andriolo 2024, 고무 1개"),
    "ceramic": DensitySpec(800, 1500, 2400, "가정값 (벽돌·도자기 조각; 재료 1.8–2.4 g/cm³)", prefer_count=False,
                           mean_item_g=68.0, item_source="Andriolo 2024, 도자기·건축자재 4개 273 g"),
    # --- 식생: 쓰레기 아님. 무게 집계에서 제외 ---
    "vegetation": DensitySpec(0, 0, 0, "쓰레기 아님", is_litter=False),
    # --- 미확인: 개수 × 중앙값 ---
    "unknown": DensitySpec(20, 50, 300, "가정값 (other_plastic 과 동일)", prefer_count=True,
                           mean_item_g=13.4, item_source="Andriolo & Gonçalves 2024, 693개 해변 중앙값 13.4 g"),
}

# 3D 로 부피를 잴 수 없는 소형 물체 기준 (가정값). 이보다 작으면 개수 × 평균무게 로 대체.
SMALL_OBJECT_MAX_AREA_M2 = 0.0025   # 5 cm × 5 cm
SMALL_OBJECT_MAX_HEIGHT_M = 0.03    # DSM 노이즈 수준 (Kako 2020: 5 cm 이하 얇은 물체는 경사로 못 잡음)
MEDIAN_ITEM_G = 13.4                # Andriolo & Gonçalves 2024 중앙값
# 소형 물체 면적 기반 대체 (W3, Andriolo 2024 표 5 최적값): 무게 = 면적 × 0.4 cm × 1.2 g/cm³
SMALL_W3_THICKNESS_CM = 0.4
SMALL_W3_SPECIFIC_WEIGHT = 1.2

# 한국어 표시명
KO_NAMES = {
    "styrofoam_buoy": "스티로폼 부표", "styrofoam_box": "스티로폼 박스", "styrofoam_fragment": "스티로폼 조각",
    "pet_bottle": "PET병", "plastic_buoy": "플라스틱 부표", "chinese_buoy": "중국산 부표",
    "other_plastic": "기타 플라스틱", "rope": "로프", "net": "어망", "glass": "유리", "metal": "금속",
    "wood": "나무", "cloth": "섬유·의류", "rubber": "고무", "ceramic": "도자기·건축자재",
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
    "나무": "wood", "목재": "wood", "유목": "wood", "wood": "wood", "timber": "wood",
    "의류": "cloth", "옷": "cloth", "섬유": "cloth", "신발": "cloth", "cloth": "cloth", "textile": "cloth", "shoe": "cloth",
    "고무": "rubber", "타이어": "rubber", "rubber": "rubber", "tire": "rubber", "tyre": "rubber",
    "도자기": "ceramic", "벽돌": "ceramic", "콘크리트": "ceramic", "ceramic": "ceramic", "brick": "ceramic",
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
