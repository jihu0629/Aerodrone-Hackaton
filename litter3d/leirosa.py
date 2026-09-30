"""Andriolo et al. (2024) Mar. Pollut. Bull. 202, 116405 — 레이로자(Leirosa) 해변 실측 데이터 (CC BY).

표 1 (OSPAR 시각조사): 1,505개, 24,720 g. 각 행 = (OSPAR 코드, 종류, 범주, 개수, 총 무게 g).
표 3 (수동 판독 W1 입력), 표 4 (문헌 16편 평균무게 W2), 표 5 (W3 입력: 면적 52,276 cm², DSM 부피 12,230 cm³).

시뮬레이션용 물체 치수(L×W×H cm, 모양)는 **논문에 없는 우리 가정값**이다. 논문은 무게와 개수만 준다.
- 속 빈 플라스틱·그물·섬유: 일반적인 물건의 겉 크기에서 정했다 → 겉보기 밀도표를 시험하는 항목.
- 나무·고무·도자기(꽉 찬 고체): 재질 밀도(목재 0.5, 고무 0.6, 벽돌 1.5 g/cm³)로 무게에서 역산했다 → 부피·분류 코드만 시험.
결과 해석 시 반드시 가정임을 밝혀야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

GROUND_TRUTH_N = 1505
GROUND_TRUTH_G = 24720
GSD_CM = 0.4                     # 논문 정사영상·DSM 해상도
OBJ_SEG_AREA_CM2 = 52276         # 객체 분할 총 면적 (표 5)
SEM_SEG_VOLUME_CM3 = 12230       # HRNet + DSM 부피 (표 5)
SEM_SEG_N = 899                  # HRNet 검출 개수
MANUAL_N_TABLE3 = 1427           # 수동 판독 (본문·표 3)
MANUAL_N_TABLE4 = 1445           # 표 4 는 1445 로 표기 (논문 내 불일치)
SEM_SEG_MIN_DETECT_CM3 = 1000    # "1000 cm³ 보다 작은 물체는 못 잡음" (본문 4.1, Hidaka 2022)

# 표 4: 문헌 16편 평균 무게 (g)
LITERATURE_MEAN_G = [5.7, 6.7, 6.8, 7.0, 8.6, 9.2, 13.5, 14.1, 14.4, 15.1, 15.6, 15.9, 18.0, 19.0, 21.0, 25.1]
LITERATURE_MEAN_OF_MEANS = 13.5
LITERATURE_MEDIAN = 14.2


@dataclass(frozen=True)
class CensusRow:
    code: str
    type_: str
    category: str
    n: int
    total_g: float
    # ---- 아래는 시뮬레이션 가정값 ----
    our_class: str        # litter3d 클래스
    shape: str            # box | cylinder | dome
    L_cm: float           # 지면 footprint 긴 변
    W_cm: float           # 짧은 변
    H_cm: float           # 높이 (DSM 이 보는 값)
    note: str = ""

    @property
    def mean_g(self) -> float:
        return self.total_g / self.n

    @property
    def footprint_cm2(self) -> float:
        import math
        if self.shape == "box":
            return self.L_cm * self.W_cm
        return math.pi * self.L_cm * self.W_cm / 4

    @property
    def volume_cm3(self) -> float:
        import math
        if self.shape == "box":
            return self.L_cm * self.W_cm * self.H_cm
        if self.shape == "cylinder":
            return math.pi * self.L_cm * self.W_cm / 4 * self.H_cm
        return 2 / 3 * math.pi * self.L_cm * self.W_cm / 4 * self.H_cm

    @property
    def true_apparent_density_kg_m3(self) -> float:
        """실측 평균무게 ÷ 가정 겉 부피 (g/cm³ × 1000)."""
        return self.mean_g / self.volume_cm3 * 1000


# (코드, 종류, 범주, n, 총g, 우리 클래스, 모양, L, W, H, 메모)
TABLE1: list[CensusRow] = [r if isinstance(r, CensusRow) else CensusRow(*r) for r in [
    ("1", "4/6-pack yokes", "Plastic", 1, 4, "other_plastic", "box", 20, 10, 0.2, "납작"),
    ("3", "Small plastic bags", "Plastic", 38, 635, "other_plastic", "dome", 20, 14, 3, "구겨진 봉지"),
    ("112", "Plastic bag ends", "Plastic", 5, 158, "other_plastic", "dome", 25, 15, 3, ""),
    ("4", "Drinks bottles/containers/drums", "Plastic", 108, 4565, "pet_bottle", "cylinder", 25, 8, 8, "누운 PET/통 (0.5–1.5 L 혼합)"),
    ("5", "Cleaner bottles", "Plastic", 4, 100, "pet_bottle", "cylinder", 22, 9, 9, ""),
    ("610a", "Food containers plastic", "Plastic", 22, 433, "other_plastic", "box", 15, 11, 5, "속 빈 용기"),
    ("620a", "Food containers foamed PS", "Plastic", 2, 26, "styrofoam_box", "box", 16, 16, 6, ""),
    ("15", "Caps/lids", "Plastic", 101, 339, "other_plastic", "cylinder", 3.5, 3.5, 1.5, "소형"),
    ("16", "Cigarette lighters", "Plastic", 11, 113, "other_plastic", "box", 8, 2.5, 1.2, ""),
    ("17", "Pens", "Plastic", 2, 14, "other_plastic", "box", 14, 1, 1, ""),
    ("18", "Combs/hair brushes", "Plastic", 265, 78, "other_plastic", "box", 4, 1.5, 0.4, "0.29 g → 빗살 조각으로 추정, 소형"),
    ("20", "Toys & party poppers", "Plastic", 39, 122, "other_plastic", "box", 5, 3, 2, ""),
    ("211a", "Cups plastic", "Plastic", 4, 24, "other_plastic", "cylinder", 10, 8, 8, "속 빔"),
    ("22", "Cutlery/trays/straws", "Plastic", 29, 23, "other_plastic", "box", 15, 1.2, 0.6, "소형·얇음"),
    ("25", "Gloves (washing up)", "Plastic", 1, 7, "other_plastic", "box", 25, 12, 1.5, ""),
    ("113", "Gloves (industrial)", "Plastic", 1, 48, "other_plastic", "box", 28, 14, 3, ""),
    ("27", "Octopus pots", "Plastic", 1, 521, "other_plastic", "cylinder", 22, 22, 25, "속 빈 항아리"),
    ("28", "Oyster nets/mussel bags", "Plastic", 1, 14, "net", "dome", 20, 12, 3, ""),
    ("31", "Rope (>1 cm)", "Plastic", 2, 222, "rope", "dome", 30, 25, 5, "느슨한 뭉치 가정"),
    ("321a", "String and cord (<1 cm)", "Plastic", 55, 228, "rope", "box", 20, 1.5, 0.8, "가늘고 얇음 → 소형"),
    ("116", "Nets and pieces of net >=50 cm", "Plastic", 9, 3180, "net", "dome", 60, 40, 8, "엉킨 그물 뭉치 가정"),
    ("331a", "Tangled nets/cord/rope", "Plastic", 7, 81, "net", "dome", 15, 10, 3, ""),
    ("36", "Light sticks", "Plastic", 9, 99, "other_plastic", "box", 15, 1.5, 1.5, ""),
    ("37", "Floats/Buoys", "Plastic", 10, 198, "plastic_buoy", "dome", 9, 9, 7, "소형 부표"),
    ("40", "Industrial packaging sheeting", "Plastic", 1, 158, "other_plastic", "dome", 80, 50, 4, "구겨진 시트"),
    ("43", "Shotgun cartridges", "Plastic", 34, 89, "other_plastic", "cylinder", 7, 2, 2, "소형"),
    ("44", "Shoes/sandals (plastic)", "Plastic", 3, 57, "other_plastic", "box", 25, 10, 5, ""),
    ("45", "Foam sponge", "Plastic", 35, 110, "other_plastic", "box", 8, 5, 3, ""),
    ("1172a", "Foamed PS fragments 0.5–2.5 cm", "Plastic", 19, 519, "styrofoam_fragment", "box", 2, 2, 1.5, "평균 27 g 은 큰 덩어리 포함 추정. 소형"),
    ("452a", "Foamed PS fragments 2.5–5 cm", "Plastic", 565, 1741, "styrofoam_fragment", "box", 5, 4, 4, "소형 경계"),
    ("472a", "Foamed PS fragments >50 cm", "Plastic", 44, 1944, "styrofoam_fragment", "box", 60, 25, 1.5, "44 g/개 → 얇은 판"),
    ("48", "Other plastic items", "Plastic", 3, 77, "other_plastic", "box", 15, 10, 4, ""),
    ("53", "Other rubber pieces", "Rubber", 1, 278, "rubber", "box", 25, 15, 1.2, "고무 ≈0.6 g/cm³ 겉보기 기준 치수"),
    ("54", "Clothing", "Cloth", 10, 645, "cloth", "dome", 40, 30, 4, "젖은 옷 뭉치"),
    ("55", "Furnishing", "Cloth", 4, 55, "cloth", "dome", 20, 15, 3, ""),
    ("57", "Shoes (leather)", "Cloth", 9, 886, "cloth", "box", 28, 11, 8, ""),
    ("59", "Other textiles", "Cloth", 6, 180, "cloth", "dome", 25, 20, 3, ""),
    ("62", "Cartons", "Paper", 3, 40, "unknown", "box", 20, 12, 3, ""),
    ("73", "Paint brushes", "Wood", 2, 74, "wood", "box", 20, 3, 1.2, "목재 ≈0.5 g/cm³ 로 치수 산정"),
    ("74", "Other wood <50 cm", "Wood", 2, 416, "wood", "box", 30, 7, 2, "가공 목재, 0.5 g/cm³ 기준 치수"),
    ("75", "Other wood >50 cm", "Wood", 2, 3769, "wood", "box", 100, 10, 3.8, "판재 1.9 kg/개, 0.5 g/cm³ 기준 치수"),
    ("75b", "Aerosol/Spray cans", "Metal", 2, 174, "metal", "cylinder", 20, 6.5, 6.5, ""),
    ("77", "Bottle caps (metal)", "Metal", 3, 5, "metal", "cylinder", 3, 3, 0.6, "소형"),
    ("79", "Electric appliances", "Metal", 1, 97, "metal", "box", 15, 10, 5, ""),
    ("81", "Foil wrappers", "Metal", 1, 21, "metal", "box", 15, 10, 0.5, ""),
    ("87", "Lobster/crab pots and tops", "Metal", 6, 116, "metal", "box", 20, 15, 4, ""),
    ("91", "Bottles (glass)", "Glass", 10, 1658, "glass", "cylinder", 25, 7, 7, "누운 병"),
    ("94", "Construction material", "Ceramics", 3, 226, "ceramic", "box", 8, 6, 1.5, "벽돌 조각 ≈1.5 g/cm³ 겉보기 기준"),
    ("96", "Other pottery/ceramic", "Ceramics", 1, 47, "ceramic", "box", 6, 4, 1.3, ""),
    ("99", "Sanitary towels", "Sanitary", 1, 54, "unknown", "box", 15, 8, 2, ""),
    ("103", "Containers/tubes (medical)", "Medical", 2, 34, "other_plastic", "cylinder", 10, 3, 3, ""),
    ("104", "Syringes", "Medical", 5, 18, "other_plastic", "box", 10, 1.5, 1.5, "소형"),
]]

assert sum(r.n for r in TABLE1) == GROUND_TRUTH_N, sum(r.n for r in TABLE1)
assert abs(sum(r.total_g for r in TABLE1) - GROUND_TRUTH_G) < 1, sum(r.total_g for r in TABLE1)

# 표 3: 수동 판독으로 찾은 종류별 개수와 4가지 무게값 (g) — (종류, n, 현장평균, Grundlehner 하한, 평균, 상한)
TABLE3 = [
    ("Bags", 22, 16.7, 10.7, 19.6, 80.5), ("Drinks", 96, 42.3, 13.8, 30.5, 65.7), ("Cleaner", 6, 25, 23.7, 38.7, 65.7),
    ("Food containers", 20, 19.7, 5.7, 20, 37.5), ("Caps/lids", 61, 3.36, 1.5, 65.1, 137.7),
    ("Octopus pot", 1, 521, 233.3, 483.3, 716.7), ("Rope/nets", 72, 4.15, 1.5, 157, 314.5),
    ("Floats/Buoys", 3, 19.8, 402.5, 3563.7, 6850), ("Foam sponge", 27, 3.14, 1.7, 15.5, 30),
    ("Plastic pieces", 37, 3.8, 2.5, 525.6, 1000), ("Other plastic items", 718, 25.7, 2.33, 25, 400),
    ("Shoes", 11, 98.4, 165, 385, 812.5), ("Wood pieces", 24, 208, 100, 200, 1000), ("Bottles (glass)", 1, 165.8, 150, 400, 900),
    ("Construction material", 3, 75.3, 10, 100, 1000), ("Other medical items", 16, 3.6, 1.5, 10, 25),
    # 논문 본문은 미확인에 25.7 g 을 줬다고 하지만 표 3 의 합계(3,399 g / 306개 = 11.1 g)와 맞지 않는다. 표의 숫자를 따른다.
    ("Undefined", 306, 3399 / 306, 2.33, 25, 400),
]
# 표 3 의 행 개수 합은 1424 (논문 총계 1427 과 3개 차이, 표에서 생략된 소수 항목으로 추정)
TABLE3_TOTALS = {"n": 1427, "census": 34657, "lower": 10407, "mean": 85877, "upper": 547380}

# 표 5: 두께 가정 (cm) × 비중 (g/cm³) → 무게 (g)
TABLE5_THICKNESS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
TABLE5_SPECIFIC_WEIGHT = [0.8, 1.2, 1.5]
TABLE5_EXPECTED = {(0.4, 1.2): 25092, (0.2, 0.8): 8364, (1.0, 1.5): 78414, (0.6, 0.8): 25092}
TABLE5_DSM_EXPECTED = {0.8: 9784, 1.2: 14676, 1.5: 18345}
