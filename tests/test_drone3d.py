"""
드론 3D 파이프라인(05~07) 단위 검증. 실제 DJI 영상·ODM 없이 실행 가능.

    pytest tests/test_drone3d.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.coastline import save_geojson  # noqa: E402
from coastcd.dji import (  # noqa: E402
    blur_score, extract_frames, flight_summary, haversine_m, parse_srt, read_gps_exif, spacing_for_overlap,
    telemetry_at, write_gps_exif,
)
from coastcd.odm import PRESETS, build_command, find_outputs, stage_images  # noqa: E402
from coastcd.raster_io import write_geotiff  # noqa: E402
from coastcd.volume import (  # noqa: E402
    DENSITY_KG_M3, height_candidates, load_labels, measure_labels, measure_polygon, save_measurements,
)

SRT_NEW = """1
00:00:00,000 --> 00:00:00,033
<font size="28">SrtCnt : 1, DiffTime : 33ms
2026-09-30 10:12:45.123
[iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 37.194120] [longitude: 126.011230] [rel_alt: 80.000 abs_alt: 95.200] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">SrtCnt : 2, DiffTime : 33ms
2026-09-30 10:12:45.156
[iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 37.194130] [longitude: 126.011240] [rel_alt: 80.100 abs_alt: 95.300] </font>

3
00:00:01,000 --> 00:00:01,033
<font size="28">SrtCnt : 3, DiffTime : 33ms
2026-09-30 10:12:46.123
[iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 37.194500] [longitude: 126.011800] [rel_alt: 81.000 abs_alt: 96.200] </font>
"""

SRT_OLD = """1
00:00:00,000 --> 00:00:01,000
<font size="36">FrameCnt : 1, DiffTime : 33ms
2019-08-01 10:12:45,123
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 280] [latitude : 37.19412] [longtitude : 126.01123] [altitude: 95.2] </font>

2
00:00:01,000 --> 00:00:02,000
<font size="36">FrameCnt : 2, DiffTime : 33ms
2019-08-01 10:12:46,123
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 280] [latitude : 37.19420] [longtitude : 126.01130] [altitude: 95.4] </font>
"""

SRT_ANCIENT = """1
00:00:00,000 --> 00:00:01,000
HOME(126.0110,37.1940) 2018.05.01 10:12:45
GPS(126.0112,37.1941,17) BAROMETER:80.0
ISO:100 Shutter:1000 EV: Fnum:F2.8
"""


# ------------------------------------------------------------------ SRT

def test_parse_srt_new_format(tmp_path):
    p = tmp_path / "DJI_0001.SRT"
    p.write_text(SRT_NEW, encoding="utf-8")
    recs = parse_srt(p)
    assert len(recs) == 3
    r = recs[0]
    assert r.has_gps
    assert r.lat == pytest.approx(37.194120)
    assert r.lon == pytest.approx(126.011230)
    assert r.rel_alt_m == pytest.approx(80.0)
    assert r.abs_alt_m == pytest.approx(95.2)
    assert r.focal_len == pytest.approx(24.0)
    assert r.fnum == pytest.approx(2.8)
    assert r.datetime_utc.startswith("2026-09-30 10:12:45")
    assert recs[2].t_start_s == pytest.approx(1.0)


def test_parse_srt_old_format(tmp_path):
    p = tmp_path / "DJI_0002.SRT"
    p.write_text(SRT_OLD, encoding="utf-8")
    recs = parse_srt(p)
    assert len(recs) == 2
    assert recs[0].lon == pytest.approx(126.01123)  # longtitude 오타 처리
    assert recs[0].abs_alt_m == pytest.approx(95.2)
    assert recs[0].fnum == pytest.approx(2.8)       # 280 -> 2.8
    assert recs[0].focal_len == pytest.approx(28.0)  # 280 -> 28.0


def test_parse_srt_ancient_format(tmp_path):
    p = tmp_path / "DJI_0003.SRT"
    p.write_text(SRT_ANCIENT, encoding="utf-8")
    recs = parse_srt(p)
    assert len(recs) == 1
    assert recs[0].lat == pytest.approx(37.1941)
    assert recs[0].lon == pytest.approx(126.0112)
    assert recs[0].rel_alt_m == pytest.approx(80.0)


def test_telemetry_at_and_summary(tmp_path):
    p = tmp_path / "a.SRT"
    p.write_text(SRT_NEW, encoding="utf-8")
    recs = parse_srt(p)
    assert telemetry_at(recs, 0.0).index == 1
    assert telemetry_at(recs, 0.04).index == 2
    assert telemetry_at(recs, 0.9).index == 3
    assert telemetry_at([], 1.0) is None
    s = flight_summary(recs)
    assert s["with_gps"] == 3
    assert s["rel_alt_max_m"] == pytest.approx(81.0)
    assert 40 < s["track_length_m"] < 80  # 약 0.0004° 위도 + 0.0006° 경도 이동


def test_haversine_and_spacing():
    # 위도 1도 약 111 km
    assert haversine_m(37.0, 126.0, 38.0, 126.0) == pytest.approx(111_000, rel=0.01)
    # 고도 50 m, Mini 3 근사 카메라: 지상 폭 약 72 m, 겹침 75% 면 간격 18 m
    sp = spacing_for_overlap(50.0, 0.75)
    assert 15 < sp < 22


# ------------------------------------------------------------------ EXIF + 프레임 추출

def test_gps_exif_roundtrip(tmp_path):
    import cv2

    img = np.full((64, 96, 3), 128, np.uint8)
    jpg = tmp_path / "f.jpg"
    cv2.imwrite(str(jpg), img)
    write_gps_exif(jpg, -37.194125, 126.01123, 95.2, "2026-09-30 10:12:45.123", 24.0)
    lat, lon, alt = read_gps_exif(jpg)
    assert lat == pytest.approx(-37.194125, abs=1e-6)
    assert lon == pytest.approx(126.01123, abs=1e-6)
    assert alt == pytest.approx(95.2, abs=1e-3)
    assert read_gps_exif(_plain_jpg(tmp_path)) is None


def _plain_jpg(tmp_path):
    import cv2

    p = tmp_path / "plain.jpg"
    cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
    return p


def _make_video(path: Path, n: int = 90, fps: float = 30.0, blur_from: int | None = None):
    """질감 있는 합성 영상. blur_from 이후 프레임은 흐리게."""
    import cv2

    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert vw.isOpened()
    for i in range(n):
        f = np.roll(base, i, axis=1)
        if blur_from is not None and i >= blur_from:
            f = cv2.GaussianBlur(f, (31, 31), 12)
        vw.write(f)
    vw.release()


def test_blur_score_orders():
    import cv2

    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8)
    assert blur_score(sharp) > blur_score(blurred) * 5


def test_extract_frames_every_sec_with_gps(tmp_path):
    video = tmp_path / "DJI_0001.MP4"
    _make_video(video, n=90, fps=30.0)
    (tmp_path / "DJI_0001.SRT").write_text(SRT_NEW, encoding="utf-8")
    out = tmp_path / "frames"
    frames = extract_frames(video, out, every_sec=1.0, min_blur=0.0, log=lambda *a: None)
    assert len(frames) == 3  # t = 0, 1, 2 초
    assert all(f.lat is not None for f in frames)
    assert (out / "frames.csv").exists()
    lat, lon, alt = read_gps_exif(out / frames[0].file)
    assert lat == pytest.approx(37.194120, abs=1e-5)
    assert alt == pytest.approx(95.2, abs=1e-2)


def test_extract_frames_drops_blurred_and_every_m(tmp_path):
    video = tmp_path / "DJI_0002.MP4"
    _make_video(video, n=60, fps=30.0, blur_from=30)
    (tmp_path / "DJI_0002.SRT").write_text(SRT_NEW, encoding="utf-8")
    out = tmp_path / "frames"
    # 뒤 절반은 흐림 -> 1초 간격이면 t=0 만 남아야 함 (t=1 은 흐림)
    sharp_only = extract_frames(video, out, every_sec=1.0, min_blur=100.0, log=lambda *a: None)
    assert [round(f.t_s) for f in sharp_only] == [0]
    # 거리 기준: 첫 프레임 후 50 m 이상 움직여야 다음 프레임. SRT 는 약 60 m 이동 -> 2장
    out2 = tmp_path / "frames2"
    by_dist = extract_frames(video, out2, every_m=50.0, min_blur=0.0, log=lambda *a: None)
    assert len(by_dist) == 2


def test_extract_frames_without_srt(tmp_path):
    video = tmp_path / "nosrt.MP4"
    _make_video(video, n=45)
    msgs = []
    frames = extract_frames(video, tmp_path / "f", every_sec=0.5, min_blur=0.0, log=msgs.append)
    assert len(frames) == 3
    assert all(f.lat is None for f in frames)
    assert any("SRT" in m for m in msgs)


# ------------------------------------------------------------------ ODM 명령

def test_build_command_and_stage(tmp_path):
    cmd = build_command(tmp_path / "ds", "beach", preset="fast", gpu=False, tty=False)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "-ti" not in cmd
    assert "opendronemap/odm" in cmd
    i = cmd.index("--project-path")
    assert cmd[i + 1 : i + 3] == ["/datasets", "beach"]
    assert "--dsm" in cmd and "--dtm" in cmd
    assert "--proj" in cmd and "zone=52" in cmd[cmd.index("--proj") + 1]
    for tok in PRESETS["fast"]:
        assert tok in cmd
    gpu = build_command(tmp_path / "ds", "beach", gpu=True, tty=True, extra=["--min-num-features", "12000"])
    assert "--gpus" in gpu and "opendronemap/odm:gpu" in gpu and "-ti" in gpu
    assert gpu[-2:] == ["--min-num-features", "12000"]
    with pytest.raises(ValueError):
        build_command(tmp_path, "x", preset="nope")

    frames = tmp_path / "frames"
    frames.mkdir()
    for n in ("a.jpg", "b.JPG", "frames.csv"):
        (frames / n).write_bytes(b"x")
    img_dir = stage_images(frames, tmp_path / "ds", "beach")
    assert sorted(p.name for p in img_dir.iterdir()) == ["a.jpg", "b.JPG"]
    found = find_outputs(tmp_path / "ds", "beach")
    assert all(v is None for v in found.values())


# ------------------------------------------------------------------ 부피·무게

CRS = "EPSG:32652"


def _flat_dsm_with_pile(tmp_path, px=0.05, slope=0.0, pile_h=0.6, pile_r=2.0, size=200, dtm_noise=0.0):
    """평지(또는 경사) + 원뿔 더미. 원뿔 부피 = 1/3 π r² h."""
    tr = Affine(px, 0, 300000.0, 0, -px, 4117000.0)
    yy, xx = np.mgrid[0:size, 0:size]
    x = tr.c + (xx + 0.5) * px
    y = tr.f - (yy + 0.5) * px
    ground = 10.0 + slope * (x - tr.c)
    cx, cy = tr.c + size * px / 2, tr.f - size * px / 2
    r = np.hypot(x - cx, y - cy)
    cone = np.clip(pile_h * (1 - r / pile_r), 0, None)
    dsm = (ground + cone).astype(np.float32)
    dtm = (ground + dtm_noise * np.random.default_rng(0).standard_normal(ground.shape)).astype(np.float32)
    dsm_p, dtm_p = tmp_path / "dsm.tif", tmp_path / "dtm.tif"
    write_geotiff(dsm_p, dsm, tr, CRS, nodata=-9999.0, overviews=False)
    write_geotiff(dtm_p, dtm, tr, CRS, nodata=-9999.0, overviews=False)
    true_vol = np.pi * pile_r**2 * pile_h / 3
    return dsm_p, dtm_p, tr, (cx, cy), true_vol


def test_measure_polygon_cone_all_bases(tmp_path):
    dsm_p, dtm_p, tr, (cx, cy), true_vol = _flat_dsm_with_pile(tmp_path)
    with rasterio.open(dsm_p) as s:
        dsm = s.read(1).astype(float)
    with rasterio.open(dtm_p) as s:
        dtm = s.read(1).astype(float)
    poly = Polygon([(cx - 2.5, cy - 2.5), (cx + 2.5, cy - 2.5), (cx + 2.5, cy + 2.5), (cx - 2.5, cy + 2.5)])
    for base in ("dtm", "plane", "min"):
        m = measure_polygon(poly, dsm, tr, -9999.0, dtm, base=base, label="plastic")
        assert m.area_m2 == pytest.approx(25.0)
        assert m.volume_m3 == pytest.approx(true_vol, rel=0.03), base
        assert m.height_max_m == pytest.approx(0.6, abs=0.02)
        assert m.weight_kg == pytest.approx(true_vol * DENSITY_KG_M3["plastic"], rel=0.03)
        assert m.volume_neg_m3 < 0.01
    # DTM 이 없으면 dtm -> plane 으로 자동 전환
    m = measure_polygon(poly, dsm, tr, -9999.0, None, base="dtm")
    assert m.base_method == "plane"


def test_measure_polygon_on_slope_plane_beats_min(tmp_path):
    """경사지: plane/dtm 은 맞고 min 은 과대."""
    dsm_p, dtm_p, tr, (cx, cy), true_vol = _flat_dsm_with_pile(tmp_path, slope=0.1)
    with rasterio.open(dsm_p) as s:
        dsm = s.read(1).astype(float)
    poly = Polygon([(cx - 2.5, cy - 2.5), (cx + 2.5, cy - 2.5), (cx + 2.5, cy + 2.5), (cx - 2.5, cy + 2.5)])
    plane = measure_polygon(poly, dsm, tr, -9999.0, None, base="plane")
    mn = measure_polygon(poly, dsm, tr, -9999.0, None, base="min")
    assert plane.volume_m3 == pytest.approx(true_vol, rel=0.03)
    assert mn.volume_m3 > true_vol * 2


def test_measure_labels_end_to_end_with_reprojection(tmp_path):
    dsm_p, dtm_p, tr, (cx, cy), true_vol = _flat_dsm_with_pile(tmp_path, dtm_noise=0.005)
    poly = Polygon([(cx - 2.5, cy - 2.5), (cx + 2.5, cy - 2.5), (cx + 2.5, cy + 2.5), (cx - 2.5, cy + 2.5)])
    empty = box(cx + 3.0, cy + 3.0, cx + 4.0, cy + 4.0)      # 더미 없는 곳
    outside = box(cx + 500, cy + 500, cx + 501, cy + 501)   # DSM 범위 밖
    # 라벨을 WGS84 로 저장해서 재투영 경로도 검증
    labels = save_geojson([poly, empty, outside], CRS, tmp_path / "labels.geojson",
                          properties=[{"class": "Net"}, {"class": "unknownthing"}, {"class": "wood"}], wgs84=True)
    msgs = []
    meas, geoms, crs = measure_labels(dsm_p, labels, dtm_path=dtm_p, base="dtm", log=msgs.append)
    assert crs == CRS
    assert len(meas) == 2  # 범위 밖은 건너뜀
    assert meas[0].label == "net"
    assert meas[0].volume_m3 == pytest.approx(true_vol, rel=0.05)
    assert meas[0].weight_kg == pytest.approx(true_vol * DENSITY_KG_M3["net"], rel=0.05)
    assert meas[1].volume_m3 < 0.02 and meas[1].weight_kg is None
    assert any("알 수 없는 class" in m for m in msgs)
    paths = save_measurements(meas, geoms, crs, tmp_path / "out")
    assert paths["csv"].exists() and paths["geojson"].exists()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["n_labels"] == 2
    assert summary["by_class"]["net"]["weight_known"] is True
    assert summary["by_class"]["unknownthing"]["weight_known"] is False
    assert summary["total_volume_m3"] == pytest.approx(meas[0].volume_m3 + meas[1].volume_m3)
    # 다시 읽으면 좌표계가 유지되고 속성이 남아 있어야 함
    g2, p2, c2 = load_labels(paths["geojson"])
    assert c2 == CRS and len(g2) == 2 and p2[0]["label"] == "net"


def test_height_candidates_finds_pile(tmp_path):
    dsm_p, dtm_p, tr, (cx, cy), true_vol = _flat_dsm_with_pile(tmp_path)
    polys, props, crs = height_candidates(dsm_p, dtm_p, min_height_m=0.15, min_area_m2=0.5, max_area_m2=100.0)
    assert len(polys) == 1
    # 높이 0.15 m 이상인 원의 반지름 = 2 * (1 - 0.15/0.6) = 1.5 m -> 면적 약 7.07 m²
    assert polys[0].area == pytest.approx(np.pi * 1.5**2, rel=0.1)
    assert polys[0].centroid.x == pytest.approx(cx, abs=0.1)
    assert props[0]["class"] == "mixed"
    # AOI 로 제외
    far = box(cx + 3, cy + 3, cx + 4, cy + 4)
    polys2, _, _ = height_candidates(dsm_p, dtm_p, aoi=far)
    assert polys2 == []


# ------------------------------------------------------------------ Mini 시리즈: MP4 내장 자막

def _make_video_with_embedded_srt(tmp_path: Path, n: int = 90, fps: float = 30.0) -> Path:
    """DJI Mini 처럼 자막 트랙(mov_text) 이 MP4 안에 들어 있는 영상을 ffmpeg 로 만듭니다."""
    import subprocess

    from coastcd.dji import _ffmpeg_exe

    exe = _ffmpeg_exe()
    if exe is None:
        pytest.skip("ffmpeg 없음")
    raw = tmp_path / "raw.mp4"
    _make_video(raw, n=n, fps=fps)
    srt = tmp_path / "telemetry.srt"
    srt.write_text(SRT_NEW, encoding="utf-8")
    out = tmp_path / "DJI_0005.MP4"
    subprocess.run([exe, "-y", "-v", "error", "-i", str(raw), "-i", str(srt), "-map", "0:v", "-map", "1:0",
                    "-c:v", "copy", "-c:s", "mov_text", str(out)], check=True, capture_output=True)
    assert out.exists()
    return out


def test_extract_embedded_srt_roundtrip(tmp_path):
    from coastcd.dji import extract_embedded_srt, find_or_extract_srt

    video = _make_video_with_embedded_srt(tmp_path)
    assert not video.with_suffix(".SRT").exists()  # Mini 처럼 옆에 SRT 파일이 없음
    msgs = []
    srt = find_or_extract_srt(video, log=msgs.append)
    assert srt is not None and srt.exists()
    recs = parse_srt(srt)
    assert len(recs) == 3
    assert recs[0].lat == pytest.approx(37.194120)
    assert recs[0].abs_alt_m == pytest.approx(95.2)
    # 두 번째 호출은 이미 꺼낸 파일을 재사용
    assert extract_embedded_srt(video, log=msgs.append) == srt


def test_extract_embedded_srt_no_track(tmp_path):
    from coastcd.dji import _ffmpeg_exe, extract_embedded_srt

    if _ffmpeg_exe() is None:
        pytest.skip("ffmpeg 없음")
    video = tmp_path / "plain.MP4"
    _make_video(video, n=30)
    msgs = []
    assert extract_embedded_srt(video, log=msgs.append) is None
    assert not video.with_suffix(".SRT").exists()
    assert any("영상 자막" in m for m in msgs)


def test_extract_frames_uses_embedded_srt(tmp_path):
    video = _make_video_with_embedded_srt(tmp_path)
    out = tmp_path / "frames"
    frames = extract_frames(video, out, every_sec=1.0, min_blur=0.0, log=lambda *a: None)
    assert len(frames) == 3
    assert all(f.lat is not None for f in frames)
    lat, lon, alt = read_gps_exif(out / frames[0].file)
    assert lat == pytest.approx(37.194120, abs=1e-5)
