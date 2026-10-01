"""strategy.py 검증 — 합성 원형 섬 (반지름 1.5 km, 10 m 격자) 에 점수와 밀도를 심어 기하·예산·소티·포착 규칙을 확인."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d import priority as P  # noqa: E402
from litter3d import strategy as S  # noqa: E402


@pytest.fixture(scope="module")
def island():
    from rasterio.transform import from_origin
    px, n = 10.0, 500
    tr = from_origin(0.0, n * px, px, px)
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = n / 2
    land = (xx - cx) ** 2 + (yy - cy) ** 2 <= (150) ** 2           # 반지름 1.5 km
    coast = P.coast_from_mask(land, tr, step_m=px)
    coast = P.add_features(coast, land, tr, wind_from_deg=0.0, px_m=px)
    coast = P.score_coast(coast)
    # 밀도: 북쪽(위) 해안 1/4 에 kg 집중, 나머지 균등 소량. 해안선에서 안쪽 10~30 m
    ang = np.arctan2(coast.xy[:, 1] - cy * px, coast.xy[:, 0] - cx * px)
    north = (ang > math.radians(45)) & (ang < math.radians(135))
    inland = coast.xy - coast.normal * 20.0
    w = np.where(north, 10.0, 0.5)
    return coast, inland, w, north


def test_passes_match_sim_ortho_geometry():
    ops = S.Ops(alt_m=20.0, sea_m=20.0, inland_m=100.0, overlap_side=0.3)
    n, spacing, across, offs = ops.passes()
    assert abs(across - 22.48) < 0.1 and abs(spacing - 15.74) < 0.1       # sim_uav_cov_summary.json 의 footprint·spacing
    assert n == 8 and offs[0] == pytest.approx(-20 + across / 2)
    assert offs[-1] + across / 2 >= 100.0                                  # 마지막 선이 안쪽 100 m 를 덮는다


def test_offset_line_length_is_not_inflated(island):
    coast, *_ = island
    ii = np.nonzero(coast.ring_id == 0)[0]
    L0 = S._plen(coast.xy[ii])
    R = L0 / (2 * math.pi)
    for off in (26.0, 100.0):
        pts = S.offset_line(coast.xy[ii], coast.normal[ii], off, simplify_m=4.0)
        L = S._plen(pts)
        assert L == pytest.approx(2 * math.pi * (R - off), rel=0.05)        # 안쪽 평행선은 2π(R − d)
    sea = S.offset_line(coast.xy[ii], coast.normal[ii], -20.0, simplify_m=4.0)
    assert S._plen(sea) == pytest.approx(2 * math.pi * (R + 20), rel=0.05)


def test_offset_line_removes_zigzag_on_jagged_coast():
    """울퉁불퉁한 해안(10 m 격자 실제 해안선과 비슷)에서는 법선 밀기가 멀리 갈수록 부풀고, offset_curve 는 부풀지 않는다."""
    from rasterio.transform import from_origin
    px, n = 10.0, 500
    tr = from_origin(0.0, n * px, px, px)
    yy, xx = np.mgrid[0:n, 0:n]
    th = np.arctan2(yy - 250, xx - 250)
    land = (xx - 250) ** 2 + (yy - 250) ** 2 <= (150 + 2.0 * np.sin(60 * th)) ** 2
    coast = P.coast_from_mask(land, tr, step_m=px)
    coast = P.add_features(coast, land, tr, wind_from_deg=0.0, px_m=px)
    ii = np.nonzero(coast.ring_id == 0)[0]
    L0 = S._plen(coast.xy[ii])
    raw = S._plen(coast.xy[ii] - coast.normal[ii] * 100.0)
    L = S._plen(S.offset_line(coast.xy[ii], coast.normal[ii], 100.0, simplify_m=4.0))
    assert raw > 1.3 * L0                      # 법선 밀기: 100 m 안쪽에서 30 % 넘게 부푼다
    assert L < L0                              # offset_curve: 안쪽 평행선은 둘레보다 짧다
    assert L == pytest.approx(2 * math.pi * (1500 - 100), rel=0.08)


def test_budget_capture_and_sorties(island):
    coast, grid_xy, w, north = island
    near = P.nearest_coast_index(coast, grid_xy)
    segs = P.make_segments(coast, 200.0)
    ops = S.Ops(alt_m=20.0)
    # 점수 = 북쪽 1, 나머지 0 → 상위 25 % 예산이면 북쪽 해안을 고르고 kg 의 대부분을 잡는다
    score = north.astype(float)
    sc = S.seg_scores(segs, score)
    chosen = S.pick_by_score(segs, sc, 0.25)
    r = S.evaluate(coast, segs, chosen, ops, grid_xy, w, near)
    assert 0.2 <= r["coast_frac"] <= 0.25
    assert r["captured_frac"] > 0.8
    # 전체 커버리지: 포착 100 %, 소티는 배터리 안
    full = S.evaluate(coast, segs, segs, ops, grid_xy, w, near, keep_paths=True)
    assert full["captured_frac"] == pytest.approx(1.0)
    assert full["coast_frac"] == pytest.approx(1.0)
    assert all(so.time_min(ops) <= ops.battery_min + 1e-6 for so in full["_sorties"])
    assert full["survey_km"] > r["survey_km"] and full["time_h"] > r["time_h"]
    # 조사 km ≈ 둘레 × 패스 수 보다 짧다 (안쪽 평행선) 그리고 0.6배보다는 길다
    n_pass = ops.passes()[0]
    assert 0.6 * full["coast_km"] * n_pass < full["survey_km"] < full["coast_km"] * n_pass


def test_geo_directions_and_gap_fill(island):
    coast, grid_xy, w, north = island
    near = P.nearest_coast_index(coast, grid_xy)
    segs = P.make_segments(coast, 200.0)
    ops = S.Ops(alt_m=20.0)
    depot = coast.xy[0] - coast.normal[0] * 100
    fwd = S.pick_geo(coast, segs, 0.3, depot, direction=1)
    rev = S.pick_geo(coast, segs, 0.3, depot, direction=-1)
    assert abs(sum(s.length_m for s in fwd) - sum(s.length_m for s in rev)) <= 200.0
    assert fwd[0].seg_id == rev[0].seg_id                                  # 둘 다 출발지에서 가장 가까운 구간부터
    assert {s.seg_id for s in fwd} != {s.seg_id for s in rev}
    # 간격 메우기: 한 칸씩 건너뛴 구간을 고르면 run 이 합쳐지고 조사 길이가 늘며 포착도 는다
    every_other = segs[0:20:2]
    r0 = S.evaluate(coast, segs, every_other, ops, grid_xy, w, near, gap_fill_m=0.0)
    r1 = S.evaluate(coast, segs, every_other, ops, grid_xy, w, near, gap_fill_m=200.0)
    assert r0["n_runs"] == 10 and r1["n_runs"] == 1
    assert r1["coast_km"] > r0["coast_km"] and r1["transit_km"] <= r0["transit_km"] + 1e-6


def test_inland_distance_sign(island):
    coast, grid_xy, w, _ = island
    d = S.inland_distance(coast, grid_xy)
    assert np.median(d) == pytest.approx(20.0, abs=6.0)                     # 밀도는 안쪽 20 m 에 심었다
    sea = coast.xy + coast.normal * 15.0
    assert np.median(S.inland_distance(coast, sea)) == pytest.approx(-15.0, abs=6.0)


def test_score_variants_shapes(island):
    coast, *_ = island
    for kind in ("default", "bay", "exposure", "wind:180"):
        v = S.score_variant(coast, kind, wind_from=0.0)
        assert v.shape == (coast.n,) and np.isfinite(v).all()
    e0 = S.score_variant(coast, "exposure", wind_from=0.0); e180 = S.score_variant(coast, "wind:180", wind_from=0.0)
    assert np.corrcoef(e0, e180)[0, 1] < 0                                  # 풍향을 뒤집으면 노출 점수가 뒤집힌다
