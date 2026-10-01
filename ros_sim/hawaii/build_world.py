"""
하와이 항공조사 칩(2 cm/px, 지오참조)을 실제 좌표대로 바닥에 깐 Gazebo 월드를 만든다.

칩은 쓰레기가 있는 곳만 잘라 둔 학습용 표본이라 서로 떨어져 있다. 칩마다 원래
해상도 그대로 12.8 m 타일로 깔고, 칩이 없는 곳은 무늬 없는 모래색 바닥으로
채운다(그 구간에 쓰레기가 없다는 뜻은 아님 — 평가는 칩 라벨 기준으로만 한다).

출력 (generated/, git 제외):
  models/hawaii_ground/   칩 타일 모델 (텍스처 포함)
  worlds/hawaii.sdf       월드
  ground_truth.json       라벨 박스의 로컬 좌표(m, 동/북)
  mission_coverage.json   전체 커버리지 경로 (mission_runner 입력 형식)
  mission_hotspot.json    핫스팟 우선 경로 (같은 형식)
  meta.json               원점 UTM·위경도, 카메라·경로 설정

사용: python ros_sim/hawaii/build_world.py [--alt 20] [--overlap 0.6]
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
CHIPS = ROOT / "hotspot/data/imagery_and_labels/processed_image_chips"
LABEL_CSVS = [ROOT / "hotspot/data/imagery_and_labels/training_data.csv",
              ROOT / "hotspot/data/imagery_and_labels/evaluation_data.csv"]
OUT = Path(__file__).resolve().parent / "generated"
sys.path.insert(0, str(ROOT / "path_planning"))
from coverage import boustrophedon_path, path_length, swath_width  # noqa: E402
from hotspot_route import Cell, plan_route  # noqa: E402

EPSG = 26904                       # NAD83 / UTM 4N (니하우·카우아이)
BBOX = (388700, 2427900, 389100, 2428300)  # 칩 밀집 구간(200 m 격자 1위, 칩 40장)
CHIP_M = 12.8                      # 640 px × 2 cm
MARGIN = 15.0
CAM_HFOV_DEG = 84.0


def load_chips():
    chips = []
    for x in sorted(CHIPS.glob("*.aux.xml")):
        t = x.read_text()
        if f'"EPSG","{EPSG}"]]</SRS>' not in t:
            continue
        gt = [float(v) for v in re.search(r"<GeoTransform>(.*?)</GeoTransform>", t).group(1).split(",")]
        x0, y0 = gt[0], gt[3]
        if BBOX[0] <= x0 <= BBOX[2] and BBOX[1] <= y0 <= BBOX[3]:
            chips.append({"name": x.name[:-8], "x0": x0, "y0": y0, "res": gt[1]})
    return chips


def load_labels(names):
    out = []
    for p in LABEL_CSVS:
        split = "train" if "training" in p.name else "eval"
        for r in csv.DictReader(open(p)):
            if r["filename"] in names:
                out.append(dict(r, split=split))
    return out


def hotspot_mission(gt, coverage_len, alt, budget_frac, cell_m):
    """과거 조사(학습 분할 라벨)만 보고 칸별 기대량을 매겨, 예산 안에서 가치가 큰 칸을 도는 경로.

    평가 분할 라벨은 경로 계획에 쓰지 않는다 — 계획이 못 본 쓰레기를 얼마나 잡는지 따로 잰다.
    한 칸은 위를 한 번 지나가면 촬영 범위(고도 20 m에서 약 36 × 27 m) 안에 다 들어온다.
    """
    counts = {}
    for g in gt:
        if g["split"] == "train":
            k = (int(g["east"] // cell_m), int(g["north"] // cell_m))
            counts[k] = counts.get(k, 0) + 1
    cells = [Cell(f"{i}_{j}", (i + 0.5) * cell_m, (j + 0.5) * cell_m, mean=float(v))
             for (i, j), v in counts.items()]
    base = (0.0, 0.0)
    budget = budget_frac * coverage_len
    order = plan_route(cells, base, budget, values=[c.mean for c in cells])
    wps = [{"x": 0.0, "y": 0.0, "z": float(alt), "phase": "mapping", "note": "start"}]
    for i in order:
        c = cells[i]
        wps.append({"x": round(c.x, 2), "y": round(c.y, 2), "z": float(alt), "phase": "mapping",
                    "note": f"cell {c.id} prior {int(c.mean)}"})
    wps.append({"x": 0.0, "y": 0.0, "z": float(alt), "phase": "mapping", "note": "return"})
    info = {"cells_with_prior": len(cells), "cells_visited": len(order), "budget_m": round(budget, 1),
            "prior_value_share": round(sum(cells[i].mean for i in order) / sum(c.mean for c in cells), 3)}
    return wps, info


def tile_visual(i, cx, cy, tex):
    return f"""
      <visual name="tile_{i:02d}">
        <pose>{cx:.3f} {cy:.3f} 0.01 0 0 0</pose>
        <geometry><plane><normal>0 0 1</normal><size>{CHIP_M} {CHIP_M}</size></plane></geometry>
        <material>
          <diffuse>1 1 1 1</diffuse><ambient>1 1 1 1</ambient><specular>0 0 0 1</specular>
          <pbr><metal><albedo_map>materials/textures/{tex}</albedo_map><roughness>1</roughness><metalness>0</metalness></metal></pbr>
        </material>
      </visual>"""


WORLD = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="hawaii">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::Imu" filename="gz-sim-imu-system"/>
    <plugin name="gz::sim::systems::AirPressure" filename="gz-sim-air-pressure-system"/>
    <plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.9 0.9 0.9 1</ambient>
      <background>0.6 0.75 0.9 1</background>
      <shadows>false</shadows>
    </scene>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
        </collision>
        <visual name="sand">
          <geometry><plane><normal>0 0 1</normal><size>{gw:.1f} {gh:.1f}</size></plane></geometry>
          <pose>{gcx:.1f} {gcy:.1f} 0 0 0 0</pose>
          <material><ambient>0.76 0.70 0.58 1</ambient><diffuse>0.76 0.70 0.58 1</diffuse><specular>0 0 0 1</specular></material>
        </visual>
      </link>
    </model>
    <include><uri>model://hawaii_ground</uri></include>
    <light name="sun" type="directional">
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <intensity>1</intensity>
      <direction>0.1 0.1 -1</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
    </light>
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=20.0, help="맵핑 고도 (m)")
    ap.add_argument("--overlap", type=float, default=0.6, help="옆줄 겹침 비율")
    ap.add_argument("--budget", type=float, default=0.35, help="핫스팟 경로 예산 (전체 커버리지 길이 대비)")
    ap.add_argument("--cell", type=float, default=20.0, help="핫스팟 칸 크기 (m)")
    a = ap.parse_args()

    chips = load_chips()
    if not chips:
        sys.exit(f"칩이 없음: {CHIPS} (hotspot/README.md의 데이터 내려받기 먼저)")
    labels = load_labels({c["name"] for c in chips})

    ox = min(c["x0"] for c in chips) - MARGIN
    oy = min(c["y0"] for c in chips) - CHIP_M - MARGIN
    ex = max(c["x0"] for c in chips) + CHIP_M + MARGIN - ox
    ey = max(c["y0"] for c in chips) + MARGIN - oy

    if OUT.exists():
        shutil.rmtree(OUT)
    tex_dir = OUT / "models/hawaii_ground/materials/textures"
    tex_dir.mkdir(parents=True)
    (OUT / "worlds").mkdir()

    visuals, gt = [], []
    by_name = {}
    for i, c in enumerate(chips):
        tex = f"chip_{i:02d}.jpg"
        shutil.copy(CHIPS / (c["name"]), tex_dir / tex)
        cx = c["x0"] - ox + CHIP_M / 2
        cy = c["y0"] - oy - CHIP_M / 2
        visuals.append(tile_visual(i, cx, cy, tex))
        by_name[c["name"]] = c
    for r in labels:
        c = by_name[r["filename"]]
        xmin, ymin, xmax, ymax = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        e = c["x0"] - ox + (xmin + xmax) / 2 * c["res"]
        n = c["y0"] - oy - (ymin + ymax) / 2 * c["res"]
        gt.append({"chip": r["filename"], "label": r["label"], "split": r["split"],
                   "east": round(e, 3), "north": round(n, 3),
                   "w_m": round((xmax - xmin) * c["res"], 3), "h_m": round((ymax - ymin) * c["res"], 3)})

    (OUT / "models/hawaii_ground/model.config").write_text(
        '<?xml version="1.0"?>\n<model><name>hawaii_ground</name><version>1.0</version>'
        '<sdf version="1.9">model.sdf</sdf></model>\n')
    (OUT / "models/hawaii_ground/model.sdf").write_text(
        '<?xml version="1.0"?>\n<sdf version="1.9">\n  <model name="hawaii_ground">\n'
        '    <static>true</static>\n    <link name="tiles">' + "".join(visuals) +
        "\n    </link>\n  </model>\n</sdf>\n")
    (OUT / "worlds/hawaii.sdf").write_text(
        WORLD.format(gw=ex + 200, gh=ey + 200, gcx=ex / 2, gcy=ey / 2))

    # 띠가 남북으로 길어서 줄을 남북 방향으로 깐다 (회전 횟수 최소화)
    path = boustrophedon_path(0, 0, ey, ex, a.alt, fov_deg=CAM_HFOV_DEG, overlap=a.overlap)
    wps = [{"x": float(round(w.y, 2)), "y": float(round(w.x, 2)), "z": float(w.z), "phase": "mapping", "note": w.note} for w in path]
    (OUT / "mission_coverage.json").write_text(json.dumps({"mapping_orbit_path": wps}, ensure_ascii=False, indent=1))
    from coverage import Waypoint
    cov_len = path_length([Waypoint(w["x"], w["y"], w["z"], "mapping") for w in wps])
    hwps, hinfo = hotspot_mission(gt, cov_len, a.alt, a.budget, a.cell)
    (OUT / "mission_hotspot.json").write_text(json.dumps({"mapping_orbit_path": hwps}, ensure_ascii=False, indent=1))
    (OUT / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))

    lon, lat = Transformer.from_crs(EPSG, 4326, always_xy=True).transform(ox + ex / 2, oy + ey / 2)
    meta = {"epsg": EPSG, "origin_utm": [ox, oy], "extent_m": [round(ex, 2), round(ey, 2)],
            "center_latlon": [lat, lon], "chips": len(chips), "labels": len(gt),
            "alt_m": a.alt, "overlap": a.overlap, "cam_hfov_deg": CAM_HFOV_DEG, "cam_px": [1280, 960],
            "swath_m": round(swath_width(a.alt, CAM_HFOV_DEG), 2), "waypoints": len(wps),
            "spawn": [wps[0]["x"], wps[0]["y"]],
            "coverage_len_m": round(cov_len, 1), "hotspot": hinfo}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
