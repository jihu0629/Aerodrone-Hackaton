"""
설정값 — 클래스별 물성표 · 수거계획 기준 · 위치 불확실성.

⚠️ 여기 있는 숫자 대부분은 **가정값**이다. 업체가 준 실측 무게로
`weight.py`가 학습하면 물성표는 "사전값(prior)" 역할만 하고, 실제
예측은 데이터로 보정된다. 출처가 있는 값은 옆에 적어 뒀다.

외부 JSON(--config)으로 일부만 덮어쓸 수 있다:
    {"plan": {"bag_kg": 15}, "classes": {"net": {"rho_app": 200}}}
"""
import copy
import json
from pathlib import Path

# 클래스별 물리 모델 파라미터
#   rho_app   : 겉보기 밀도 kg/m³ = 실제 무게 ÷ 겉 부피 (속 빈 물체는 재료 밀도보다 훨씬 작음)
#   thick_m   : 2D만 있을 때 쓰는 "유효 두께" (면적 × 두께 = 부피 근사, Andriolo W3 방식)
#   kg_per_m  : 선형 물체(로프)용 단위길이당 무게 — 있으면 길이 모델도 같이 계산
#   aliases   : 업체 클래스 이름에 이 문자열이 들어 있으면 이 클래스로 매핑 (소문자 비교)
#   thick_range_m : 현실적인 두께 범위 [최소, 최대] — 업체 "면적×계수" 방식과 비교할 때 사용 (가정)
#   compress  : 마대에 담을 때 압축비 (겉 부피 ÷ 압축비 = 마대 적재 부피). EPS 5~10×, 어망 2~3×, PET 4× (가정)
#   wet       : 젖었을 때 무게 배수 (어망·로프는 물 머금어 1.5~2.5×, 나머지 ~1) (가정)
CLASSES = {
    "eps_buoy":      {"rho_app": 25,  "thick_m": 0.35, "thick_range_m": [0.15, 0.40], "compress": 5, "wet": 1.1,
                      "aliases": ["스티로폼 부표", "스티로폼부표", "eps_buoy", "styrofoam buoy", "foam buoy"]},
    "eps_box":       {"rho_app": 8,   "thick_m": 0.30, "thick_range_m": [0.20, 0.35], "compress": 8, "wet": 1.1,
                      "aliases": ["스티로폼 박스", "스티로폼박스", "eps_box", "styrofoam box", "foam box"]},
    "eps_fragment":  {"rho_app": 20,  "thick_m": 0.03, "thick_range_m": [0.02, 0.10], "compress": 10, "wet": 1.1,
                      "aliases": ["스티로폼", "styrofoam", "eps", "foam", "sty"]},  # EPS 11–32 kg/m³ (Wikipedia); 압축·젖음은 가정
    "plastic_buoy":  {"rho_app": 60,  "thick_m": 0.30, "thick_range_m": [0.15, 0.40], "compress": 1, "wet": 1.0,
                      "aliases": ["플라스틱 부표", "플라스틱부표", "plastic buoy", "buoy", "부표"]},  # 단단해서 압축 안 됨
    "pet_bottle":    {"rho_app": 33,  "thick_m": 0.06, "thick_range_m": [0.05, 0.10], "compress": 4, "wet": 1.0,
                      "aliases": ["pet", "페트", "bottle", "병"]},  # PET 4× (가정: 발로 밟아 찌그러뜨림)
    "net":           {"rho_app": 150, "thick_m": 0.12, "thick_range_m": [0.05, 0.30], "compress": 2.5, "wet": 2.0,
                      "aliases": ["어망", "그물", "net", "fis"]},  # 젖음 1.5~2.5× (가정)
    "rope":          {"rho_app": 400, "thick_m": 0.03, "thick_range_m": [0.02, 0.05], "kg_per_m": 0.25, "compress": 2, "wet": 1.8,
                      "aliases": ["로프", "밧줄", "rope", "rop"]},  # 젖음 1.5~2.5× (가정)
    "glass":         {"rho_app": 600, "thick_m": 0.07, "thick_range_m": [0.05, 0.10], "compress": 1, "wet": 1.0,
                      "aliases": ["유리", "glass"]},
    "metal":         {"rho_app": 100, "thick_m": 0.10, "thick_range_m": [0.05, 0.15], "compress": 2, "wet": 1.0,
                      "aliases": ["금속", "캔", "metal", "can"]},
    "wood":          {"rho_app": 500, "thick_m": 0.05, "thick_range_m": [0.03, 0.10], "compress": 1, "wet": 1.3,
                      "aliases": ["목재", "나무", "wood"]},
    "tire":          {"rho_app": 125, "thick_m": 0.20, "thick_range_m": [0.15, 0.25], "compress": 1, "wet": 1.0,
                      "aliases": ["타이어", "tire", "tyre"]},
    "cardboard":     {"rho_app": 30,  "thick_m": 0.10, "thick_range_m": [0.05, 0.30], "compress": 6, "wet": 2.0,
                      "aliases": ["골판지", "종이상자", "cardboard", "carton", "paper box"]},  # 빈 골판지 상자 겉보기밀도 ~30 (가정)
    "plastic_other": {"rho_app": 80,  "thick_m": 0.03, "thick_range_m": [0.02, 0.10], "compress": 3, "wet": 1.0,
                      "aliases": ["플라스틱", "plastic", "비닐", "vinyl", "pla"]},
    "other":         {"rho_app": 100, "thick_m": 0.03, "thick_range_m": [0.02, 0.10], "compress": 1.5, "wet": 1.0, "aliases": []},
}

# 업체(문갑도 라벨) 방식: 박스 면적(m²) × 고정 계수(kg/m²) = 무게. 실측이 아니라 계산값.
COMPANY_KG_PER_M2 = {"STY": 0.012, "ROP": 0.024, "FIS": 0.024, "PLA": 0.020}
# 내부 클래스 → 업체 재질 코드 (비교 그림용)
COMPANY_CODE = {"eps_buoy": "STY", "eps_box": "STY", "eps_fragment": "STY", "rope": "ROP", "net": "FIS",
                "plastic_buoy": "PLA", "pet_bottle": "PLA", "plastic_other": "PLA"}
COMPANY_CODE_NAME = {"STY": "스티로폼", "ROP": "로프", "FIS": "어망", "PLA": "플라스틱"}

DEFAULT = {
    "classes": CLASSES,
    # 선행연구 대체값: 개수 × 평균무게 (Andriolo 2024 W2, 14 g) / 리뷰 중앙값 13.4 g
    "baseline_item_kg": 0.014,
    "weight": {
        "coverage": 0.90,        # 예측구간 포함률 (conformal)
        "no_data_factor": 2.0,   # 무게 정답이 없을 때 구간 = 추정값 ×/÷ 이 값
        "dsm_noise_m": 0.02,     # DSM 높이 잡음 — 이보다 2배 이상 높아야 3D 부피를 믿음
    },
    "plan": {
        "lift_limit_kg": 23.0,   # NIOSH 이상조건 권장 한계
        "equip_limit_kg": 50.0,  # 이 이상은 2인도 무리 → 장비 (가정)
        "bag_kg": 20.0,          # 마대 1개 적재 무게 (가정)
        "bag_l": 100.0,          # 마대 1개 부피 (가정)
        "truck_kg": 1000.0,      # 1톤 트럭
        "truck_m3": 3.0,         # 1톤 트럭 적재 부피 (가정)
        "grid_m": 10.0,          # 격자 히트맵 크기 (Gonçalves 2022 격자 표준화)
        "stop_radius_m": 5.0,    # 이 거리 안의 물체는 한 번에 수거하는 "정거장"으로 묶음
        "depot": None,           # [x, y] 집하장 좌표 (None이면 가장 남쪽 정거장 근처)
        "tools": {"net": "절단기", "rope": "절단기", "tire": "운반구"},
    },
    # 텔레메트리 오차 (1σ) — 위치 불확실성 몬테카를로에 사용
    "georef": {
        "gps_sigma_m": 2.5,
        "alt_sigma_m": 1.0,
        "yaw_sigma_deg": 2.0,
        "pitch_sigma_deg": 1.0,
        "n_mc": 200,
        "dedup_radius_m": 1.5,   # 여러 프레임에 중복 라벨된 같은 물체 합치는 반경
    },
}


def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULT)
    if path:
        _deep_update(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def map_class(name, cfg):
    """업체 클래스 이름 → 내부 클래스. 가장 긴 alias가 먼저 매칭되게 한다
    ("스티로폼 부표"가 "스티로폼"보다 먼저)."""
    n = str(name).lower().replace("_", " ").strip()
    if n.replace(" ", "_") in cfg["classes"]:
        return n.replace(" ", "_")
    pairs = [(a.lower(), c) for c, p in cfg["classes"].items() for a in p.get("aliases", [])]
    for alias, cls in sorted(pairs, key=lambda t: -len(t[0])):
        if alias in n:
            return cls
    return "other"
