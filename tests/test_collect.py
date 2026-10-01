import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from litter3d.collect import (CollectParams, cluster_by_matrix, estimate_2d, load_geojson, make_collect_plan,
                              save_plan_files, summary_markdown, tour_order)
from litter3d.plan import NIOSH_LIFT_LIMIT_KG
from litter3d.terrain import BARE, OUTSIDE, VEG, WATER, Terrain, classify_rgb


def _feature(seq, code, lon, lat, w_deg=0.00001, h_deg=0.00001, area=1.0, kg=0.012):
    ring = [[lon - w_deg, lat - h_deg], [lon + w_deg, lat - h_deg], [lon + w_deg, lat + h_deg], [lon - w_deg, lat + h_deg],
            [lon - w_deg, lat - h_deg]]
    return {"type": "Feature",
            "properties": {"survey_date": "2026-07-29", "region": "TST", "material_code": code, "detection_seq": seq,
                           "center_lon": lon, "center_lat": lat, "area_sqm": area, "weight_kg": kg,
                           "image_path": f"crops/TST_{seq:04d}_{code}.jpg"},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


@pytest.fixture
def geojson(tmp_path):
    # 서쪽 3개(서로 20 m 안), 동쪽 2개 (약 1 km 떨어짐), 큰 로프 1개
    feats = [_feature(1, "STY", 126.0800, 37.1700, area=1.5), _feature(2, "STY", 126.08015, 37.1700, area=0.8),
             _feature(3, "PLA", 126.0801, 37.17015, area=2.0),
             _feature(4, "STY", 126.0910, 37.1700, area=1.0), _feature(5, "ROP", 126.0911, 37.17005, area=20.0, kg=0.48)]
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return p


def test_estimate_2d_range_and_styrofoam_density():
    kmin, ktyp, kmax, vol, formula = estimate_2d("STY", 1.54)
    assert 0 < kmin < ktyp < kmax
    assert vol == pytest.approx(1.54 * 0.5 * 0.10)
    assert ktyp == pytest.approx(vol * 20)          # EPS 대표 밀도 20 kg/m³
    assert estimate_2d("XYZ", 1.0)[1] > 0


def test_load_geojson_and_cluster(geojson):
    objs = load_geojson(geojson)
    assert len(objs) == 5
    assert objs[0].obj_id == "TST_0001_STY" and objs[0].company_kg == 0.012
    D = np.array([[np.hypot(a.x_m - b.x_m, a.y_m - b.y_m) for b in objs] for a in objs])
    lab = cluster_by_matrix(D, 150)
    assert len(set(lab)) == 2 and lab[0] == lab[1] == lab[2] and lab[3] == lab[4]


def test_tour_order_small_and_large():
    pts = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40)]
    D = np.array([[np.hypot(a[0] - b[0], a[1] - b[1]) for b in pts] for a in pts])
    assert tour_order(D, 0, [1, 2, 3, 4], round_trip=False) == [1, 2, 3, 4]
    big = [(-10, 0)] + [(i * 10.0, (i % 2) * 3.0) for i in range(14)]
    Db = np.array([[np.hypot(a[0] - b[0], a[1] - b[1]) for b in big] for a in big])
    o = tour_order(Db, 0, list(range(1, 15)), round_trip=True)
    assert sorted(o) == list(range(1, 15))


def test_make_plan_structure(geojson, tmp_path):
    objs = load_geojson(geojson)
    p = CollectParams(depot_lonlat=(126.079, 37.1695), depot_name="테스트 선착장", workers=2, hours_per_day=0.5)
    plan = make_collect_plan(objs, p, site="테스트")
    assert plan.n_objects == 5 and len(plan.zones) == 2 and not plan.terrain_used
    assert "해안" in plan.zones[0].name
    assert set(plan.zones[0].objects) == {"TST_0001_STY", "TST_0002_STY", "TST_0003_PLA"}
    assert plan.route_len_m > 2000
    assert plan.totals["bags"] >= 2 and plan.totals["kg_plan"] > plan.totals["company_kg"]
    big_rope = next(o for o in objs if o.code == "ROP")
    assert big_rope.heavy and big_rope.kg_max > NIOSH_LIFT_LIMIT_KG
    assert plan.totals["heavy"] == 1 and plan.zones[1].heavy_ids == [big_rope.obj_id]
    assert len(plan.days) == 2 and [z.day for z in plan.zones] == [1, 2]
    assert any("칼" in e for e in plan.equipment)
    md = summary_markdown(plan)
    assert "작업 순서" in md and "준비물" in md
    save_plan_files(plan, tmp_path / "out")
    d = json.loads((tmp_path / "out" / "plan.json").read_text(encoding="utf-8"))
    assert d["n_objects"] == 5 and len(d["zones"]) == 2 and d["zones"][0]["path_lonlat"]
    assert (tmp_path / "out" / "zones.csv").exists() and (tmp_path / "out" / "objects.csv").exists()


def test_filters_weight_source_and_modes(geojson):
    objs = load_geojson(geojson)
    base = dict(depot_lonlat=(126.079, 37.1695))
    plan = make_collect_plan(objs, CollectParams(weight_source="company", **base))
    assert plan.totals["kg_plan"] == pytest.approx(plan.totals["company_kg"])
    plan = make_collect_plan(objs, CollectParams(include_codes=["ROP"], **base))
    assert plan.n_objects == 1 and plan.n_skipped == 4 and len(plan.zones) == 1
    plan = make_collect_plan(objs, CollectParams(min_kg=5.0, **base))
    assert plan.n_objects == 1
    # 들고 이동: 적재량이 작으면 출발지 복귀가 생기고 거리가 늘어난다
    p0 = make_collect_plan(objs, CollectParams(carry="pile", **base))
    p1 = make_collect_plan(objs, CollectParams(carry="carry", carry_kg_per_person=1.0, carry_bags_per_person=1, workers=1, **base))
    assert p1.totals["returns"] >= 1 and p1.route_len_m > p0.route_len_m
    # 무게 우선: 큰 로프가 있는 동쪽 구역이 먼저
    pw = make_collect_plan(objs, CollectParams(objective="weight", **base))
    assert "TST_0005_ROP" in pw.zones[0].objects


def _toy_terrain(bridge=True):
    """60×60 px (1 m/px): 가운데 세로 물길이 땅을 둘로 나누고, 북쪽 끝에 다리(맨땅) 가 있다."""
    cls = np.full((60, 60), BARE, np.uint8)
    cls[:, 28:32] = WATER
    if bridge:
        cls[0:4, 28:32] = BARE      # 다리
    cls[40:60, 0:20] = VEG          # 남서쪽 숲
    return Terrain(cls, 1.0, 0.0, 60.0, cell_m=2.0)


def test_terrain_paths_walk_vs_boat():
    t = _toy_terrain()
    a, b = (10.0, 30.0), (50.0, 30.0)           # 물길 양쪽, y=30 은 가운데
    D_walk = t.matrix([a, b], "walk")
    assert np.isfinite(D_walk[0, 1]) and D_walk[0, 1] > 60      # 다리로 돌아감 (직선 40 m, 8방향 우회 ≈ 74 m)
    t2 = _toy_terrain(bridge=False)                              # 다리가 없으면 도보 불가, 보트는 물을 건넘 (승·하선 비용 포함)
    assert not np.isfinite(t2.matrix([a, b], "walk")[0, 1])
    D_boat = t2.matrix([a, b], "boat")
    assert np.isfinite(D_boat[0, 1]) and 40 < D_boat[0, 1] < 400
    path = t.path(a, b, "walk")
    assert len(path) > 5 and path[0] == a and path[-1] == b
    assert max(p[1] for p in path) > 50                          # 북쪽 다리 근처를 지남
    # 숲은 맨땅보다 비싸다
    c, d = (5.0, 5.0), (5.0, 25.0)
    assert t.matrix([c, d], "walk")[0, 1] > 20 * 1.5
    js = t.to_js()
    assert js["nr"] == 30 and js["nc"] == 30 and len(js["b64"]) > 0


def test_classify_rgb_basic():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = (180, 150, 60)       # BGR: 물 (b>r, g>r)
    img[:, 10:] = (60, 160, 70)        # 초록
    img[5:8, 12:15] = (230, 230, 230)  # 맨땅
    cls = classify_rgb(img)
    assert cls[0, 0] == WATER and cls[0, 15] == VEG and cls[6, 13] == BARE


def test_report_outputs(geojson, tmp_path):
    from litter3d.collect_report import build_collect_html, draw_static_map
    objs = load_geojson(geojson)
    t = _toy_terrain()
    plan = make_collect_plan(objs, CollectParams(depot_lonlat=(126.079, 37.1695)), site="테스트")
    png = draw_static_map(plan, tmp_path / "map.png")
    assert png.exists() and png.stat().st_size > 10_000
    html = build_collect_html(plan, tmp_path / "plan.html", static_png=png, terrain=t)
    s = html.read_text(encoding="utf-8")
    assert "window.PLAN=" in s and "작업 순서" in s and "TST_0005_ROP" in s and '"terrain": {' in s and "c-workers" in s


def test_teams_and_calibration(geojson):
    objs = load_geojson(geojson)
    base = dict(depot_lonlat=(126.079, 37.1695))
    one = make_collect_plan(objs, CollectParams(**base))
    two = make_collect_plan(objs, CollectParams(teams=2, **base))
    assert len(two.teams) == 2 and {z.team for z in two.zones} == {1, 2}
    assert two.total_min <= one.total_min                      # 동시에 일하니 가장 오래 걸리는 팀 기준 시간이 줄거나 같다
    assert sum(len(t["zones"]) for t in two.teams) == len(two.zones)
    sty_one = sum(o.plan_kg for o in one.objects if o.code == "STY")      # 같은 객체 목록을 다시 쓰므로 먼저 읽어 둔다
    cal = make_collect_plan(objs, CollectParams(calib={"STY": 2.0}, **base))
    sty_cal = sum(o.plan_kg for o in cal.objects if o.code == "STY")
    assert sty_cal == pytest.approx(sty_one * 2.0)
    assert any("보정" in a for a in cal.assumptions)
