"""priority.py — 합성 섬(원판 + 동쪽 만)으로 해안선·특징·점수·예산·소티·내보내기 검증 (실제 데이터 없이 실행)."""
import json
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
from rasterio.transform import from_origin  # noqa: E402

from litter3d import priority as P  # noqa: E402

PX = 10.0
CRS = "EPSG:32652"


def synthetic_island():
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy, r = 150, 150, 100
    land = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    # 동쪽에 반지름 35 셀 만(bay) 을 파낸다
    bay = (xx - (cx + r)) ** 2 + (yy - cy) ** 2 <= 35 ** 2
    land &= ~bay
    transform = from_origin(0.0, n * PX, PX, PX)
    return land, transform


def test_coast_and_features_bay_is_detected():
    land, transform = synthetic_island()
    coast = P.coast_from_mask(land, transform, step_m=PX)
    assert coast.n > 300
    coast = P.add_features(coast, land, transform, wind_from_deg=90.0, bay_radius_m=150.0, px_m=PX)
    # 법선이 바다를 향하는지: 법선 쪽 30 m 가 물
    water = ~land
    inv = ~transform
    p = coast.xy + coast.normal * 30
    c, r = inv * (p[:, 0], p[:, 1])  # noqa: affine 곱
    r, c = np.clip(r.astype(int), 0, 299), np.clip(c.astype(int), 0, 299)
    assert water[r, c].mean() > 0.9
    # 만 안쪽(x 가 가장 큰 쪽에서 안으로 들어간 점) 의 만입도가 곶보다 높다
    # 만: 동쪽 가장자리 x=2500 에서 반지름 350 m 를 파냈으므로 만 안쪽 해안은 x 2150~2300, |y−1500| < 200
    bay_pts = (coast.xy[:, 0] > 2100) & (coast.xy[:, 0] < 2300) & (np.abs(coast.xy[:, 1] - 1500) < 200)
    assert bay_pts.sum() > 10
    assert coast.feats["bay"][bay_pts].mean() > coast.feats["bay"][~bay_pts].mean() + 0.05
    # 동풍이면 동쪽 해안이 노출 +, 서쪽이 −
    east = coast.xy[:, 0] > 2000; west = coast.xy[:, 0] < 1000
    assert coast.feats["exposure"][east].mean() > 0.5 > 0 > coast.feats["exposure"][west].mean()


def test_budget_sorties_and_exports(tmp_path):
    land, transform = synthetic_island()
    coast = P.coast_from_mask(land, transform, step_m=PX)
    coast = P.add_features(coast, land, transform, wind_from_deg=90.0, px_m=PX)
    coast = P.score_coast(coast, 1.0, 0.5, 0.0)
    segs = P.make_segments(coast, 200.0)
    sel = P.select_budget(segs, budget_frac=0.3)
    total = sum(s.length_m for s in segs)
    assert 0.2 * total < sum(s.length_m for s in sel) <= 0.3 * total + 1e-6
    assert all(s.score >= max(x.score for x in segs if not x.selected) - 1e-9 for s in sel)
    fp = P.FlightParams(battery_min=10.0, speed_mps=5.0)
    depot = np.array([1500.0, 300.0])
    sorties, unreachable = P.plan_sorties(coast, sel, depot, fp, launch="mobile")
    assert sorties and not unreachable
    assert all(so.time_min <= fp.battery_min + 1e-6 for so in sorties)
    assert sorted(x for so in sorties for x in so.segments) == sorted(s.seg_id for s in sel)
    # 고정 출발지: 멀어서 못 가는 구간은 unreachable 로
    fp2 = P.FlightParams(battery_min=3.0)
    sorties2, unreachable2 = P.plan_sorties(coast, sel, depot, fp2, launch="fixed")
    assert len(unreachable2) > 0
    # 검증 곡선: 만 안에 밀도를 두면 상위 30 % 에 전부 들어감
    dens = np.array([[2250.0, 1500.0], [2240.0, 1550.0], [2260.0, 1450.0]])
    xs, ys = P.capture_curve(coast, coast.score, dens)
    assert ys[30] == pytest.approx(1.0)
    # 내보내기
    P.export_geojson(tmp_path / "a.geojson", coast, segs, sorties, CRS, depot)
    gj = json.loads((tmp_path / "a.geojson").read_text(encoding="utf-8"))
    assert any(f["properties"].get("kind") == "sortie" for f in gj["features"])
    P.export_litchi_csv(tmp_path / "s.csv", sorties[0], CRS, fp)
    assert (tmp_path / "s.csv").read_text().count("\n") == len(sorties[0].waypoints) + 1
    P.export_wpml_kmz(tmp_path / "s.kmz", sorties[0], CRS, fp)
    with zipfile.ZipFile(tmp_path / "s.kmz") as z:
        names = z.namelist()
        assert "wpmz/template.kml" in names and "wpmz/waylines.wpml" in names
        assert z.read("wpmz/waylines.wpml").count(b"<Placemark>") == len(sorties[0].waypoints)
    summ = P.summary(coast, segs, sorties, fp, unreachable, "mobile")
    json.dumps(summ, default=P._jsonable)


def test_ndwi_max_cloud_handling():
    g1 = np.array([[100.0, 100.0]]); n1 = np.array([[50.0, 300.0]])     # 장면1: 물, 땅
    g2 = np.array([[100.0, 100.0]]); n2 = np.array([[50.0, 50.0]])      # 장면2: 물, 물(구름)
    scl1 = np.array([[6, 4]], np.uint8); scl2 = np.array([[6, 8]], np.uint8)
    v = P.ndwi_max([g1, g2], [n1, n2], [scl1, scl2])
    assert v[0, 0] > 0 and v[0, 1] < 0            # 장면2 의 구름 픽셀은 제외돼 땅으로 남는다
    v2 = P.ndwi_max([g1], [n1], [np.array([[8, 8]], np.uint8)])
    assert (v2 == 1.0).all()                      # 전부 구름이면 물로
