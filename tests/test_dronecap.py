"""dronecap 자동 테스트. 실제 드론·네트워크 없이 실행 가능.

  pytest tests/test_dronecap.py -q
RTSP 경로는 여기서 시험하지 않는다 (MediaMTX 필요). docs/dronecap/README.md 의 수동 검증 절차 참고.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecap import sfm, sync  # noqa: E402
from dronecap.capture_app import CaptureApp  # noqa: E402
from dronecap.config import DEFAULTS, load_config  # noqa: E402
from dronecap.ocr.parse import convert_to, parse_number  # noqa: E402
from dronecap.ocr.roi import Roi, RoiSet  # noqa: E402
from dronecap.timeutil import parse_utc_iso, utc_iso  # noqa: E402


# ---------- 설정 ----------
def test_config_defaults_and_unknown_key(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("stream:\n  url: rtsp://x/y\n  typo_key: 1\nframes:\n  interval_s: 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg["stream"]["url"] == "rtsp://x/y"
    assert cfg["stream"]["reconnect_max_s"] == DEFAULTS["stream"]["reconnect_max_s"]
    assert cfg["frames"]["interval_s"] == 2
    assert any("typo_key" in w for w in cfg["_warnings"])


def test_repo_config_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "dronecap.yml")
    assert cfg["stream"]["url"].startswith("rtsp://127.0.0.1:8554/")
    assert not [w for w in cfg["_warnings"] if "알 수 없는" in w], cfg["_warnings"]


# ---------- 시각 ----------
def test_utc_iso_roundtrip():
    s = utc_iso()
    assert s.endswith("Z") and len(s) == 24
    assert abs(parse_utc_iso(s).timestamp() - parse_utc_iso(s).timestamp()) == 0


# ---------- OCR 숫자 해석 ----------
@pytest.mark.parametrize("raw,unit,expect_val,expect_status", [
    ("H 12.3m", "m", 12.3, "ok"),
    ("0.0 m/s", "m/s", 0.0, "ok"),            # 0 은 실패가 아니다
    ("-1.5m/s", "m/s", -1.5, "ok"),
    ("−2.0 m/s", "m/s", -2.0, "ok"),          # 유니코드 minus
    ("D 1O.5m", "m", 10.5, "ok"),             # O → 0 보정 (숫자 토큰 안에서만)
    ("12,5m", "m", 12.5, "ok"),               # 쉼표 소수점
    ("", "m", None, "empty"),
    ("H --", "m", None, "no_number"),
    ("25 km/h", "m/s", 25.0, "unit_mismatch"),  # 값은 보존, 상태로 구분
    ("5.2 ft", "m", 5.2, "unit_mismatch"),
])
def test_parse_number(raw, unit, expect_val, expect_status):
    p = parse_number(raw, unit)
    assert p.status == expect_status
    if expect_val is None:
        assert p.value is None
    else:
        assert p.value == pytest.approx(expect_val)


def test_parse_range_and_notes():
    p = parse_number("H 950m", "m", (-200, 500))
    assert p.status == "out_of_range" and p.value == 950
    p = parse_number("12.3", "m")
    assert p.status == "ok" and "unit_not_seen" in p.notes
    p = parse_number("H 1 2.3m", "m", label="H")
    assert p.status == "ambiguous" and p.value is None and any(n.startswith("candidates=") for n in p.notes)
    p = parse_number("Ho.0m", "m", label="H")           # 라벨 뒤 'o' → 0 (rapidocr 합성 화면에서 실제 관찰)
    assert p.status == "ok" and p.value == 0.0 and "label_stripped" in p.notes
    p = parse_number("H1 12.3m", "m", label="H")        # 라벨이 'H1' 로 읽히면 숫자가 둘 → 조용히 1 을 내지 않고 ambiguous
    assert p.status == "ambiguous"


def test_convert():
    assert convert_to(36.0, "km/h", "m/s")[0] == pytest.approx(10.0)
    assert convert_to(1.0, None, "m") == (None, "unknown_unit")
    assert convert_to(3.0, "m", "m")[0] == 3.0


# ---------- ROI ----------
def test_roi_save_load_and_size_check(tmp_path):
    rs = RoiSet((800, 600), "test", [Roi("H", 10, 20, 50, 30)])
    f = tmp_path / "roi.json"
    rs.save(f)
    rs2 = RoiSet.load(f)
    assert rs2.rois[0].w == 50 and tuple(rs2.reference_size) == (800, 600)
    assert rs2.check_size((600, 800, 3)) is None
    assert "다시 선택" in rs2.check_size((500, 800, 3))
    img = np.zeros((600, 800, 3), np.uint8)
    assert rs2.rois[0].crop(img).shape == (30, 50, 3)


# ---------- 1단계: 파일 입력으로 전체 흐름 ----------
def _make_video(path: Path, seconds=3, fps=20, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        img = np.full((size[1], size[0], 3), (i * 3) % 255, np.uint8)
        cv2.putText(img, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        vw.write(img)
    vw.release()
    return path


def test_capture_from_file_saves_frames_and_logs(tmp_path):
    video = _make_video(tmp_path / "v.mp4")
    cfg = load_config(None)
    cfg["stream"]["url"] = str(video)
    cfg["stream"]["file_realtime"] = True
    cfg["session"]["root"] = str(tmp_path / "sessions")
    cfg["frames"]["interval_s"] = 0.5
    cfg["record"]["backend"] = "opencv"      # ffmpeg 유무와 무관하게 테스트
    app = CaptureApp(cfg, display=False, duration_s=20, record_on_start=True)
    stats = app.run()
    assert stats["frames_received"] == 60
    assert stats["connections"] == 1
    assert 4 <= stats["frames_saved"] <= 7
    assert stats["frames_write_failed"] == 0
    s = app.session.dir
    rows = list(csv.DictReader(open(s / "frames.csv", encoding="utf-8")))
    assert len(rows) == stats["frames_saved"]
    assert all(r["saved"] == "1" and (s / r["file"]).exists() for r in rows)
    assert all(r["connection_id"] == "1" for r in rows)
    recs = list(csv.DictReader(open(s / "recordings.csv", encoding="utf-8")))
    assert len(recs) == 1 and recs[0]["backend"] == "opencv" and recs[0]["audio"] == "0" and recs[0]["reencoded"] == "1"
    assert (s / recs[0]["file"]).stat().st_size > 0
    meta = json.loads((s / "session.json").read_text(encoding="utf-8"))
    assert meta["stats"]["frames_saved"] == stats["frames_saved"]
    assert (s / "events.log").read_text(encoding="utf-8").count("EOF") >= 1


def test_capture_unopenable_source_exits_cleanly(tmp_path):
    cfg = load_config(None)
    cfg["stream"]["url"] = str(tmp_path / "missing.mp4")
    cfg["stream"]["reconnect_min_s"] = 0.2
    cfg["session"]["root"] = str(tmp_path / "sessions")
    app = CaptureApp(cfg, display=False, duration_s=1.0)
    stats = app.run()
    assert stats["frames_received"] == 0
    assert "열 수 없음" in (app.session.dir / "events.log").read_text(encoding="utf-8")


# ---------- 매칭 ----------
def test_sync_match(tmp_path):
    frames = tmp_path / "frames.csv"
    with open(frames, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "frame_id", "file", "video_receive_time_utc", "saved"])
        w.writerow(["s", 1, "frames/a.jpg", "2026-10-01T00:00:10.000Z", 1])
        w.writerow(["s", 2, "frames/b.jpg", "2026-10-01T00:00:11.000Z", 1])
        w.writerow(["s", 3, "frames/c.jpg", "2026-10-01T00:00:20.000Z", 1])   # 가까운 OCR 없음
    tele = tmp_path / "telemetry.csv"
    with open(tele, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["telemetry_receive_time_utc", "field", "value", "unit", "status"])
        w.writerow(["2026-10-01T00:00:09.900Z", "H", "12.3", "m", "ok"])
        w.writerow(["2026-10-01T00:00:09.900Z", "D", "0", "m", "ok"])          # 0 도 유효
        w.writerow(["2026-10-01T00:00:09.900Z", "HS", "36", "km/h", "ok"])     # 단위 변환
        w.writerow(["2026-10-01T00:00:10.900Z", "H", "", "", "empty"])         # 실패는 연결하지 않음
        w.writerow(["2026-10-01T00:00:10.900Z", "D", "5", "m", "ok"])
        w.writerow(["2026-10-01T00:00:10.900Z", "HS", "1.0", "m/s", "ok"])
    fmap = {"relative_altitude_m": "H", "home_distance_m": "D", "horizontal_speed_mps": "HS", "vertical_speed_mps": None}
    out = tmp_path / "matched.csv"
    st = sync.match(frames, tele, out, fmap, offset_s=0.1, tolerance_ms=300)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["telemetry_valid"] for r in rows] == ["1", "0", "0"]
    assert rows[0]["relative_altitude_m"] == "12.300" and rows[0]["home_distance_m"] == "0.000"
    assert rows[0]["horizontal_speed_mps"] == "10.000" and "converted_km/h_to_m/s" in rows[0]["unit_notes"]
    assert rows[0]["match_time_error_ms"] == "0"
    assert rows[1]["relative_altitude_m"] == "" and "H:empty" in rows[1]["fields_missing"]
    assert rows[1]["home_distance_m"] == "5.000"
    assert rows[2]["fields_matched"] == "" and "too_far" in rows[2]["fields_missing"]
    assert "vertical_speed_mps" not in rows[0]
    assert st == {"frames": 3, "valid": 1, "partial": 1, "none": 1, "fields_without_data": []}


# ---------- 프레임 선별 ----------
def test_select_frames(tmp_path):
    d = tmp_path / "frames"; d.mkdir()
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(d / "f1.jpg"), sharp)
    cv2.imwrite(str(d / "f2.jpg"), sharp)                                   # 중복
    cv2.imwrite(str(d / "f3.jpg"), cv2.GaussianBlur(sharp, (31, 31), 10))  # 흐림
    cv2.imwrite(str(d / "f4.jpg"), rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    rows = sfm.select_frames(d, tmp_path / "sel", blur_min=60, min_change=0.03)
    by = {r["file"]: r for r in rows}
    assert by["f1.jpg"]["selected"] == 1 and by["f4.jpg"]["selected"] == 1
    assert by["f2.jpg"]["reason"] == "duplicate" and by["f3.jpg"]["reason"] == "blurry"
    assert (tmp_path / "sel" / "selection.csv").exists() and (tmp_path / "sel" / "f1.jpg").exists()


def test_colmap_commands_cpu():
    cmds = sfm.colmap_sparse_commands("imgs", "work", use_gpu=False)
    assert cmds[0][1] == "feature_extractor" and "--SiftExtraction.use_gpu" in cmds[0] and cmds[0][-1] == "0"
    assert cmds[1][1] == "sequential_matcher"
    assert cmds[2][1] == "mapper"


# ---------- 기록된 비행 데이터 (SRT / XMP) ----------
SRT_SAMPLE = """1
00:00:00,000 --> 00:00:00,033
FrameCnt: 1, DiffTime: 33ms
2026-09-25 16:23:55.467
[iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 37.500000] [longitude: 127.000000] [rel_alt: 0.000 abs_alt: 65.972] [ct: 4711]

2
00:00:00,033 --> 00:00:00,066
FrameCnt: 2, DiffTime: 33ms
2026-09-25 16:23:55.500
[iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 37.500090] [longitude: 127.000000] [rel_alt: 10.000 abs_alt: 75.972] [ct: 4711]
"""


def test_srt_csv_and_colmap_ref(tmp_path):
    from dronecap import media_meta as mm
    srt = tmp_path / "DJI_0001.SRT"; srt.write_text(SRT_SAMPLE, encoding="utf-8")
    tel = mm.srt_to_csv(srt, tmp_path / "t.csv")
    assert len(tel) == 2 and tel[1].rel_alt_m == 10.0 and tel[0].lat == 37.5
    rows = list(csv.DictReader(open(tmp_path / "t.csv", encoding="utf-8")))
    assert rows[0]["timestamp"].startswith("2026-09-25") and rows[1]["abs_alt_m"] == "75.972"
    assert mm.srt_health(tel)["warning"] == ""
    frame_rows = [{"file": "a.jpg", "lat": 37.5, "lon": 127.0, "rel_alt_m": 0.0}, {"file": "b.jpg", "lat": None, "lon": 1, "rel_alt_m": 1}]
    n = mm.write_colmap_ref(frame_rows, tmp_path / "geo.txt")
    assert n == 1 and (tmp_path / "geo.txt").read_text().strip() == "a.jpg 37.50000000 127.00000000 0.000"


def test_frames_with_srt(tmp_path):
    from dronecap import media_meta as mm
    video = _make_video(tmp_path / "DJI_0001.mp4", seconds=2, fps=20)
    # 2초 영상 전체를 덮는 SRT (블록 2개를 1초씩으로)
    srt_text = SRT_SAMPLE.replace("00:00:00,000 --> 00:00:00,033", "00:00:00,000 --> 00:00:01,000") \
                         .replace("00:00:00,033 --> 00:00:00,066", "00:00:01,000 --> 00:00:02,000")
    (tmp_path / "DJI_0001.SRT").write_text(srt_text, encoding="utf-8")
    assert mm.find_srt(video) is not None
    rows = mm.frames_with_srt(video, mm.find_srt(video), tmp_path / "out", interval_s=0.5)
    assert len(rows) == 4 and rows[0]["rel_alt_m"] == 0.0 and rows[3]["rel_alt_m"] == 10.0   # t=1.5s → 두 번째 블록
    assert (tmp_path / "out" / "frames" / rows[0]["file"]).exists()
    assert (tmp_path / "out" / "frames_srt.csv").exists()


def test_enu():
    from dronecap.media_meta import latlon_to_enu
    e, n, u = latlon_to_enu(37.50009, 127.0, 10.0, 37.5, 127.0, 0.0)
    assert abs(e) < 0.01 and n == pytest.approx(10.0, abs=0.05) and u == 10.0


def test_photo_xmp_parse(tmp_path):
    from dronecap import media_meta as mm
    xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
           'drone-dji:GpsLatitude="+37.50000000" drone-dji:GpsLongitude="+127.00000000" drone-dji:AbsoluteAltitude="+65.97" '
           'drone-dji:RelativeAltitude="+28.30" drone-dji:GimbalPitchDegree="-89.90" xmp:CreateDate="2026-09-25T16:23:55"/>'
           '</rdf:RDF></x:xmpmeta>')
    d = mm.parse_dji_xmp(xmp)
    assert d["RelativeAltitude"] == "+28.30"
    # JPEG 흉내: 아무 바이트 + XMP 패킷
    img = tmp_path / "DJI_0001.JPG"
    img.write_bytes(b"\xff\xd8\xff\xe1" + b"\x00" * 50 + xmp.encode() + b"\x00" * 10)
    r = mm.read_photo_meta(img)
    assert r["source"] == "xmp" and r["RelativeAltitude"] == 28.3 and r["GpsLatitude"] == 37.5 and r["GimbalPitchDegree"] == -89.9
    assert r["DateTime"] == "2026-09-25T16:23:55"
    plain = tmp_path / "plain.jpg"
    cv2.imwrite(str(plain), np.zeros((8, 8, 3), np.uint8))
    assert mm.read_photo_meta(plain)["source"] == "none"


def test_colmap_align_commands():
    cmds = sfm.colmap_align_commands("work", "geo_gps.txt", 3.0)
    assert cmds[0][1] == "model_aligner" and "--ref_is_gps" in cmds[0] and "enu" in cmds[0]


# ---------- 수신 + 화면 OCR 동시 실행 ----------
class _FakeOcrSource:
    """화면 대신 'H 12.3m' 가 그려진 가짜 프레임을 내는 소스."""
    desc = "fake"

    def __init__(self):
        self.i = 0

    def next(self):
        img = np.full((200, 400, 3), 60, np.uint8)
        cv2.putText(img, "H 12.3m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        self.i += 1
        return img, {"source": "fake", "source_frame_index": self.i, "source_time_s": None}


class _FakeEngine:
    name = "fake"

    def read(self, img):
        from dronecap.ocr.engine import OcrResult
        return OcrResult("H 12.3m", 0.9, [("H 12.3m", 0.9)], "fake")


def test_capture_with_live_ocr(tmp_path):
    from dronecap.ocr.roi import Roi, RoiSet
    video = _make_video(tmp_path / "v.mp4", seconds=2, fps=20)
    roi_file = tmp_path / "roi.json"
    RoiSet((400, 200), "fake", [Roi("H", 10, 80, 220, 60)]).save(roi_file)
    cfg = load_config(None)
    cfg["stream"]["url"] = str(video)
    cfg["session"]["root"] = str(tmp_path / "sessions")
    cfg["ocr"]["roi_file"] = str(roi_file)
    cfg["ocr"]["interval_s"] = 0.2
    cfg["ocr"]["debug_every"] = 0
    app = CaptureApp(cfg, display=False, duration_s=10, live_ocr=True,
                     ocr_source_factory=_FakeOcrSource, ocr_engine=_FakeEngine())
    stats = app.run()
    assert stats["ocr_error"] is None and stats["ocr_samples"] >= 3
    rows = list(csv.DictReader(open(app.session.dir / "telemetry.csv", encoding="utf-8")))
    assert rows and all(r["field"] == "H" and r["value"] == "12.3" and r["status"] == "ok" for r in rows)
    assert "OCR(" in app.ocr.hud_text() and "H=12.3m[ok]" in app.ocr.hud_text()


def test_capture_live_ocr_without_roi_logs_error(tmp_path):
    video = _make_video(tmp_path / "v.mp4", seconds=1, fps=10)
    cfg = load_config(None)
    cfg["stream"]["url"] = str(video)
    cfg["session"]["root"] = str(tmp_path / "sessions")
    cfg["ocr"]["roi_file"] = str(tmp_path / "none.json")
    app = CaptureApp(cfg, display=False, duration_s=5, live_ocr=True)
    stats = app.run()
    assert stats["ocr_error"] and "ROI" in stats["ocr_error"]
    assert stats["frames_received"] == 10        # OCR 이 꺼져도 수신은 계속
