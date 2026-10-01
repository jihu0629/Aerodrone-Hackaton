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
CLASSES = {
    "eps_buoy":      {"rho_app": 25,  "thick_m": 0.35, "aliases": ["스티로폼 부표", "스티로폼부표", "eps_buoy", "styrofoam buoy", "foam buoy"]},
    "eps_box":       {"rho_app": 8,   "thick_m": 0.30, "aliases": ["스티로폼 박스", "스티로폼박스", "eps_box", "styrofoam box", "foam box"]},
    "eps_fragment":  {"rho_app": 20,  "thick_m": 0.03, "aliases": ["스티로폼", "styrofoam", "eps", "foam"]},  # EPS 11–32 kg/m³ (Wikipedia)
    "plastic_buoy":  {"rho_app": 60,  "thick_m": 0.30, "aliases": ["플라스틱 부표", "플라스틱부표", "plastic buoy", "buoy", "부표"]},
    "pet_bottle":    {"rho_app": 33,  "thick_m": 0.06, "aliases": ["pet", "페트", "bottle", "병"]},
    "net":           {"rho_app": 150, "thick_m": 0.12, "aliases": ["어망", "그물", "net"]},
    "rope":          {"rho_app": 400, "thick_m": 0.03, "kg_per_m": 0.25, "aliases": ["로프", "밧줄", "rope"]},
    "glass":         {"rho_app": 600, "thick_m": 0.07, "aliases": ["유리", "glass"]},
    "metal":         {"rho_app": 100, "thick_m": 0.10, "aliases": ["금속", "캔", "metal", "can"]},
    "wood":          {"rho_app": 500, "thick_m": 0.05, "aliases": ["목재", "나무", "wood"]},
    "tire":          {"rho_app": 125, "thick_m": 0.20, "aliases": ["타이어", "tire", "tyre"]},
    "plastic_other": {"rho_app": 80,  "thick_m": 0.03, "aliases": ["플라스틱", "plastic", "비닐", "vinyl"]},
    "other":         {"rho_app": 100, "thick_m": 0.03, "aliases": []},
}

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
