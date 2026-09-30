"""기존 무게 추정 방식 (Andriolo et al. 2024 의 W1·W2·W3) — 우리 3D 방식과 비교하는 기준선."""
from __future__ import annotations


def w1_by_type(items: list[tuple[str, int, float]]) -> float:
    """W1: Σ(종류별 개수 × 종류별 평균무게 g). items = [(type, n, mean_g), ...]"""
    return float(sum(n * g for _, n, g in items))


def w2_count(n_items: int, mean_g: float = 13.5) -> float:
    """W2: 전체 개수 × 일괄 평균무게 (문헌 16편 평균 13.5 g, 중앙값 14.2 g)."""
    return float(n_items * mean_g)


def w3_area(total_area_cm2: float, thickness_cm: float = 0.4, specific_weight_g_cm3: float = 1.2) -> float:
    """W3: 총 면적 × 가정 두께 × 플라스틱 비중. 논문 최적값 0.4 cm × 1.2 g/cm³ (결과를 보고 고른 값)."""
    return float(total_area_cm2 * thickness_cm * specific_weight_g_cm3)


def w3_dsm(volume_cm3: float, specific_weight_g_cm3: float) -> float:
    """W3': DSM 겉 부피 × 재료 비중 (0.8–1.5). 속 빈 물체를 꽉 찬 것으로 계산하므로 과대추정 경향."""
    return float(volume_cm3 * specific_weight_g_cm3)


def error_pct(estimate: float, truth: float) -> float:
    return (estimate / truth - 1) * 100.0
