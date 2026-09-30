import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from litter3d.drone import MINI5PRO, design_flight
from litter3d.srt import parse_srt_text, summarize, telemetry_at
from litter3d.synthetic import make_scene, default_objects, SynthObject
from litter3d.volume import object_volume, volumes_from_masks, split_instances
from litter3d.mass import estimate_masses, total_kg
from litter3d.classes import map_class, CLASSES
from litter3d.plan import make_plan, PlanParams, NIOSH_LIFT_LIMIT_KG
from litter3d.gridmap import grid_kg
from litter3d.reconstruct import Surface, write_geotiff, load_surface, scale_check
from litter3d.segment import masks_to_png, masks_from_png, merge_duplicates, coco_to_yolo_seg
from litter3d.pipeline import run


def test_gsd_monotonic_and_range():
    g20, g40 = MINI5PRO.gsd_cm(20), MINI5PRO.gsd_cm(40)
    assert abs(g40 / g20 - 2) < 1e-6
    # 84° 대각, 8192 px → 30 m 에서 약 0.5 cm/px
    assert 0.4 < MINI5PRO.gsd_cm(30) < 0.65
    alt = MINI5PRO.altitude_for_gsd(0.5)
    assert abs(MINI5PRO.gsd_cm(alt) - 0.5) < 1e-9
    fp = design_flight(0.5, 0.8, 0.7, 2.0)
    assert fp.shot_interval_m < fp.footprint_photo_m[1]


SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-09-25 16:23:55.467
[iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 30.142288] [longitude: -95.768454] [rel_alt: 0.000 abs_alt: 65.972] [ct: 4711] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
2026-09-25 16:23:55.500
[iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 30.142300] [longitude: -95.768460] [rel_alt: 25.100 abs_alt: 91.072] [ct: 4711] </font>
"""


def test_srt_parse():
    tel = parse_srt_text(SRT)
    assert len(tel) == 2
    assert tel[0].lat == pytest.approx(30.142288)
    assert tel[0].lon == pytest.approx(-95.768454)
    assert tel[1].rel_alt_m == pytest.approx(25.1)
    assert tel[1].abs_alt_m == pytest.approx(91.072)
    assert tel[0].focal_len == 24.0 and tel[0].frame == 1 and tel[0].fnum == 1.8
    assert telemetry_at(tel, 0.05).frame == 2
    s = summarize(tel)
    assert s["has_gps"] and s["track_length_m"] > 0


def test_srt_old_format():
    tel = parse_srt_text("1\n00:00:00,000 --> 00:00:01,000\nGPS (126.5, 37.4, 30) BAROMETER: 25.0\n")
    assert tel[0].lat == 37.4 and tel[0].lon == 126.5


def test_volume_boxes_on_slope():
    objs = default_objects()
    dsm, ortho, masks = make_scene(objs, gsd_m=0.005, noise_m=0.005, ripple_m=0.0)
    vols = volumes_from_masks(dsm, masks, 0.005)
    for o, v in zip(objs, vols):
        if o.true_volume_m3 > 0.01:     # 10 L 이상은 5 % 안에
            assert abs(v.volume_m3 / o.true_volume_m3 - 1) < 0.05, (o.cls, v.volume_m3, o.true_volume_m3)
        assert v.volume_sigma_m3 >= 0


def test_volume_ignores_neighbor_in_ring():
    # 바로 옆에 다른 물체가 있어도 강건 평면 맞춤으로 바닥이 크게 흔들리지 않아야 함
    objs = [SynthObject("styrofoam_box", "box", 5, 5, 0.6, 0.4, 0.3),
            SynthObject("rope", "box", 5.45, 5, 0.2, 0.4, 0.3)]
    dsm, _, masks = make_scene(objs, size_m=(10, 10), gsd_m=0.005, noise_m=0.003)
    v = object_volume(dsm, masks[0][1], 0.005)
    assert abs(v.volume_m3 / objs[0].true_volume_m3 - 1) < 0.10


def test_mass_rules():
    objs = default_objects()
    dsm, _, masks = make_scene(objs, gsd_m=0.01)
    ms = estimate_masses(volumes_from_masks(dsm, masks, 0.01))
    by = {m.class_name: m for m in ms}
    assert by["vegetation"].method == "excluded" and by["vegetation"].kg_typ == 0
    assert by["styrofoam_fragment"].method == "count"
    assert by["styrofoam_box"].method == "volume"
    assert by["styrofoam_box"].kg_min <= by["styrofoam_box"].kg_typ <= by["styrofoam_box"].kg_max
    # EPS 상자 0.072 m³ × 20 kg/m³ ≈ 1.44 kg
    assert abs(by["styrofoam_box"].kg_typ - 0.072 * 20) < 0.15
    lo, ty, hi = total_kg(ms)
    assert lo <= ty <= hi


def test_map_class():
    assert map_class("스티로폼 부표") == "styrofoam_buoy"
    assert map_class("PET") == "pet_bottle"
    assert map_class("어망") == "net"
    assert map_class("???") == "unknown"
    assert set(CLASSES) >= {"styrofoam_buoy", "net", "unknown"}


def test_plan_and_grid(tmp_path):
    objs = default_objects()
    dsm, ortho, masks = make_scene(objs, gsd_m=0.01)
    ms = estimate_masses(volumes_from_masks(dsm, masks, 0.01))
    g, meta = grid_kg(ms, 0.01, 10.0)
    assert g.shape == (2, 3)
    assert g.sum() == pytest.approx(sum(m.kg_typ for m in ms if m.method != "excluded"))
    p = make_plan(ms, 0.01, None, PlanParams(vehicle_capacity_kg=30))
    assert p.total_bags >= 1 and p.truck_trips >= 1
    assert all(h["kg_typ"] > NIOSH_LIFT_LIMIT_KG for h in p.heavy_items)
    assert p.routes and sum(len(r["stops"]) for r in p.routes) == len(p.priority_cells)
    for r in p.routes:
        assert r["load_kg"] <= 30 + 1e-6 or len(r["stops"]) == 1
    # 스티로폼은 부피가 먼저 참
    sty = [c for c in p.by_class if c.class_name == "styrofoam_buoy"][0]
    assert sty.limiting == "volume"


def test_pipeline_end_to_end(tmp_path):
    objs = default_objects()
    dsm, ortho, masks = make_scene(objs, gsd_m=0.01)
    surf = Surface(dsm=dsm, ortho=ortho, gsd_m=0.01, transform=None, crs=None)
    r = run(surf, masks, tmp_path)
    for f in ["objects.csv", "grid_kg.csv", "grid_kg.png", "plan.json", "summary.md", "overlay.jpg"]:
        assert (tmp_path / f).exists()
    assert r["n_objects"] == len(objs)


def test_geotiff_roundtrip_and_mask_png(tmp_path):
    objs = default_objects()[:3]
    dsm, ortho, masks = make_scene(objs, size_m=(25, 10), gsd_m=0.01)
    write_geotiff(dsm, tmp_path / "dsm.tif")
    write_geotiff(ortho[..., ::-1], tmp_path / "ortho.tif")
    s = load_surface(tmp_path / "dsm.tif", tmp_path / "ortho.tif")
    assert s.gsd_m == 1.0 and s.ortho.shape == ortho.shape   # write_geotiff 기본 변환은 1 px = 1 단위
    masks_to_png(masks, dsm.shape, tmp_path / "m.png")
    back = masks_from_png(tmp_path / "m.png")
    assert sorted(c for c, _, _ in back) == sorted(c for c, _, _ in masks)


def test_merge_duplicates_and_split():
    a = np.zeros((20, 20), bool); a[2:10, 2:10] = True
    b = a.copy(); b[2:10, 2:11] = True
    merged = merge_duplicates([("net", a, 0.6), ("net", b, 0.8), ("rope", a, 0.5)])
    assert len(merged) == 2
    assert len(split_instances(a | np.roll(a, 12, axis=0))) == 2


def test_scale_check():
    r = scale_check(0.9, 1.0)
    assert r["length_error_pct"] == pytest.approx(-10)
    assert r["volume_error_pct_if_uncorrected"] == pytest.approx(-27.1)


def test_coco_to_yolo(tmp_path):
    import json
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}],
            "categories": [{"id": 7, "name": "스티로폼 부표"}, {"id": 8, "name": "PET"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "segmentation": [[10, 10, 30, 10, 30, 30]]},
                            {"id": 2, "image_id": 1, "category_id": 8, "bbox": [50, 5, 10, 10]}]}
    (tmp_path / "c.json").write_text(json.dumps(coco), encoding="utf-8")
    info = coco_to_yolo_seg(tmp_path / "c.json", tmp_path, tmp_path / "yolo")
    txt = (tmp_path / "yolo/labels/a.txt").read_text().strip().split("\n")
    assert len(txt) == 2 and info["per_class"] == {"styrofoam_buoy": 1, "pet_bottle": 1}
    assert txt[0].startswith("0 0.100000 0.200000")
