"""
DJI 드론 영상 -> 3D 재구성용 프레임 (GPS EXIF 포함).

DJI 기체(Mini/Air/Mavic/Matrice 계열)는 영상(.MP4)을 찍을 때 같은 이름의 자막 파일(.SRT)을
같이 저장합니다 (설정: 카메라 > 영상 자막/Video Caption 켜기). 이 SRT 에 프레임 단위 시각,
위도·경도·고도, 셔터/ISO/초점거리가 들어 있습니다. 3D 재구성(OpenDroneMap, COLMAP)은
사진 EXIF 의 GPS 를 초기 위치로 쓰기 때문에, 영상에서 뽑은 프레임에 SRT 의 GPS 를 EXIF 로
써 넣어 주면 별도 GCP 없이도 대략 미터 단위로 지리참조된 결과가 나옵니다 (정밀도는 기체 GNSS 에
따라 수 m. RTK 기체면 cm 급).

지원하는 SRT 형식
-----------------
1) 신형 (Mini 3/4, Air 2S/3, Mavic 3, Matrice 300/350 등):
    1
    00:00:00,000 --> 00:00:00,033
    <font size="28">SrtCnt : 1, DiffTime : 33ms
    2026-09-30 10:12:45.123
    [iso : 100] [shutter : 1/1000.0] [fnum : 2.8] [ev : 0] [focal_len : 24.00]
    [latitude: 37.194120] [longitude: 126.011230] [rel_alt: 80.000 abs_alt: 95.200] </font>

2) 구형 (Mavic 2, Phantom 4 등):
    1
    00:00:00,000 --> 00:00:01,000
    <font size="36">FrameCnt : 1, DiffTime : 33ms
    2019-08-01 10:12:45,123
    [iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default]
    [focal_len : 280] [latitude : 37.19412] [longtitude : 126.01123] [altitude: 95.2] </font>

3) 아주 오래된 형식:
    GPS(126.0112,37.1941,17) BAROMETER:80.0 ... HOME(...)

세 형식을 모두 정규식으로 파싱합니다. 필드가 없으면 None 으로 둡니다.

Mini 시리즈 주의
----------------
Mini 2/3/4/5 Pro 는 '영상 자막' 을 켜도 별도 .SRT 파일을 만들지 않고 MP4 안에 자막 트랙으로
넣습니다. `extract_embedded_srt()` 가 ffmpeg 로 꺼냅니다(`find_or_extract_srt()` 가 자동 호출).
사진은 설정과 무관하게 EXIF 에 GPS 가 항상 들어가고, DJI Fly 앱 비행 기록(Flight Record) 에도
0.1 초 단위 위치가 남습니다. 어느 쪽도 없을 때만 지상기준점(GCP) 으로 좌표를 부여합니다.

프레임 선택
-----------
영상 30 fps 를 전부 쓰면 사진이 수천 장이 되고, 인접 프레임은 거의 같은 위치라 재구성에 도움이 안 됩니다.
- 시간 간격(--every-sec) 또는 이동 거리(--every-m) 기준으로 뽑고
- 라플라시안 분산(선명도)이 낮은 흐린 프레임(모션블러, 회전 중)은 버립니다.
경험적으로 사진 간 겹침 70~80% 가 되도록 고도·속도에 따라 간격을 잡습니다.
비행 고도 h, 카메라 세로 화각 v 일 때 지상 촬영 폭 = 2 h tan(v/2). 겹침 75% 면 간격은 폭의 25%.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ SRT 파싱

_TIME_RANGE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_DATETIME = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})[.,](\d{1,3})")
_KV = re.compile(r"\[\s*([a-zA-Z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)")
_ALT_PAIR = re.compile(r"rel_alt\s*:\s*([-+]?\d+(?:\.\d+)?)\s+abs_alt\s*:\s*([-+]?\d+(?:\.\d+)?)")
_GPS_OLD = re.compile(r"GPS\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*(\d+)\s*\)")
_BARO_OLD = re.compile(r"BAROMETER\s*:\s*([-+]?\d+(?:\.\d+)?)")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class SrtRecord:
    """SRT 한 블록 = 한 프레임(또는 1초) 의 텔레메트리."""

    index: int
    t_start_s: float          # 영상 시작 기준 초
    t_end_s: float
    datetime_utc: str | None  # 문자열 그대로 (기체 시계, 보통 현지시각)
    lat: float | None
    lon: float | None
    rel_alt_m: float | None   # 이륙점 기준 고도
    abs_alt_m: float | None   # 해수면(타원체) 기준 고도. 없으면 None
    iso: float | None = None
    shutter: str | None = None
    fnum: float | None = None
    focal_len: float | None = None

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None and not (abs(self.lat) < 1e-6 and abs(self.lon) < 1e-6)


def _to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_block(block: str) -> SrtRecord | None:
    lines = [l for l in block.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    try:
        index = int(lines[0].strip())
    except ValueError:
        return None
    m = _TIME_RANGE.search(lines[1])
    if not m:
        return None
    t0 = _to_seconds(*m.groups()[:4])
    t1 = _to_seconds(*m.groups()[4:])
    body = _TAG.sub(" ", " ".join(lines[2:]))

    rec = SrtRecord(index=index, t_start_s=t0, t_end_s=t1, datetime_utc=None,
                    lat=None, lon=None, rel_alt_m=None, abs_alt_m=None)

    dm = _DATETIME.search(body)
    if dm:
        rec.datetime_utc = f"{dm.group(1)} {dm.group(2)}.{dm.group(3).ljust(3, '0')}"

    kv = {}
    for k, v in _KV.findall(body):
        kv[k.lower()] = v
    # 신형/구형 공통 키. 구형은 longtitude 오타를 씁니다.
    if "latitude" in kv:
        rec.lat = float(kv["latitude"])
    lon_key = "longitude" if "longitude" in kv else ("longtitude" if "longtitude" in kv else None)
    if lon_key:
        rec.lon = float(kv[lon_key])
    ap = _ALT_PAIR.search(body)
    if ap:
        rec.rel_alt_m = float(ap.group(1))
        rec.abs_alt_m = float(ap.group(2))
    else:
        if "rel_alt" in kv:
            rec.rel_alt_m = float(kv["rel_alt"])
        if "abs_alt" in kv:
            rec.abs_alt_m = float(kv["abs_alt"])
        if "altitude" in kv and rec.abs_alt_m is None:
            # 구형은 altitude 하나만 있고 보통 해수면 기준(기압 보정). 확인 필요
            rec.abs_alt_m = float(kv["altitude"])
    # 아주 오래된 형식: GPS(lon, lat, nsat)
    g = _GPS_OLD.search(body)
    if g and rec.lat is None:
        rec.lon = float(g.group(1))
        rec.lat = float(g.group(2))
        b = _BARO_OLD.search(body)
        if b:
            rec.rel_alt_m = float(b.group(1))

    if "iso" in kv:
        rec.iso = float(kv["iso"])
    if "shutter" in kv:
        rec.shutter = kv["shutter"]
    if "fnum" in kv:
        f = float(kv["fnum"])
        rec.fnum = f / 100.0 if f > 50 else f   # 구형은 280 = f/2.8
    if "focal_len" in kv:
        f = float(kv["focal_len"])
        rec.focal_len = f / 10.0 if f > 200 else f  # 구형은 280 = 28.0 mm (확인 필요)
    return rec


def parse_srt(path: str | Path) -> list[SrtRecord]:
    """DJI SRT 파일을 읽어 프레임 텔레메트리 목록을 돌려줍니다. 시간순 정렬."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    recs = [r for r in (_parse_block(b) for b in blocks) if r is not None]
    recs.sort(key=lambda r: r.t_start_s)
    return recs


def find_srt_for_video(video: str | Path) -> Path | None:
    """영상과 같은 이름의 .SRT/.srt 를 찾습니다 (Mavic/Air/Matrice 계열 방식)."""
    video = Path(video)
    for ext in (".SRT", ".srt"):
        cand = video.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _ffmpeg_exe() -> str | None:
    """시스템 ffmpeg 또는 pip 패키지 imageio-ffmpeg 가 내려받은 정적 바이너리."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 패키지 없음/바이너리 못 받음
        return None


def extract_embedded_srt(video: str | Path, out_path: str | Path | None = None, log=print) -> Path | None:
    """
    MP4 안에 내장된 DJI 자막 트랙을 .SRT 파일로 꺼냅니다.

    DJI Mini 시리즈(Mini 2/3/4/5 Pro)는 '영상 자막' 을 켜도 별도 .SRT 를 만들지 않고
    MP4 의 자막 스트림(mov_text) 에 텔레메트리를 넣습니다. 탐색기에서 .SRT 가 안 보여
    "GPS 를 제공하지 않는다" 고 오해하기 쉽지만, ffmpeg 로 꺼내면 같은 형식의 SRT 가 나옵니다.

        ffmpeg -i DJI_0001.MP4 -map 0:s:0 -f srt DJI_0001.SRT

    반환값: 만들어진 SRT 경로. 자막 트랙이 없거나 ffmpeg 가 없으면 None.
    """
    import subprocess

    video = Path(video)
    out_path = Path(out_path) if out_path else video.with_suffix(".SRT")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    exe = _ffmpeg_exe()
    if exe is None:
        log("[dji] ffmpeg 가 없어 내장 자막을 꺼낼 수 없습니다. `pip install imageio-ffmpeg` 또는 ffmpeg 설치 후 다시 실행하세요.")
        return None
    cmd = [exe, "-y", "-v", "error", "-i", str(video), "-map", "0:s:0", "-f", "srt", str(out_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"[dji] ffmpeg 실행 실패: {e}")
        return None
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        if out_path.exists():
            out_path.unlink()
        err = (r.stderr or "").strip().splitlines()
        hint = err[-1] if err else ""
        log(f"[dji] {video.name}: 내장 자막 트랙이 없습니다 ({hint}). 촬영 전에 DJI Fly > 카메라 > 고급 촬영 설정 > '영상 자막' 을 켜야 합니다.")
        return None
    log(f"[dji] 내장 자막 추출 -> {out_path.name}")
    return out_path


def find_or_extract_srt(video: str | Path, log=print) -> Path | None:
    """옆에 있는 .SRT 를 먼저 찾고, 없으면 MP4 내장 자막을 꺼냅니다."""
    found = find_srt_for_video(video)
    if found is not None:
        return found
    return extract_embedded_srt(video, log=log)


def telemetry_at(recs: list[SrtRecord], t_s: float) -> SrtRecord | None:
    """영상 시각 t_s 에 가장 가까운 레코드 (선형보간 없이 최근접). 레코드가 없으면 None."""
    if not recs:
        return None
    times = np.array([r.t_start_s for r in recs])
    i = int(np.clip(np.searchsorted(times, t_s), 0, len(recs) - 1))
    if i > 0 and abs(recs[i - 1].t_start_s - t_s) < abs(recs[i].t_start_s - t_s):
        i -= 1
    return recs[i]


# ------------------------------------------------------------------ 거리·간격

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def ground_footprint_m(alt_m: float, sensor_width_mm: float = 9.65, focal_mm: float = 6.7,
                       image_width_px: int = 4000) -> tuple[float, float]:
    """
    (지상 촬영 폭 m, GSD m/px). 기본값은 DJI Mini 3/4 (1/1.3", 24 mm 환산) 근사치. 확인 필요.
    Mavic 3 (4/3", 17.7x13.3 mm, 12.3 mm) 이면 sensor_width_mm=17.7, focal_mm=12.3.
    """
    width_m = alt_m * sensor_width_mm / focal_mm
    return width_m, width_m / image_width_px


def spacing_for_overlap(alt_m: float, overlap: float = 0.75, **cam) -> float:
    """앞뒤 겹침 overlap 을 만들기 위한 촬영 간격(m)."""
    width_m, _ = ground_footprint_m(alt_m, **cam)
    return width_m * (1.0 - overlap)


# ------------------------------------------------------------------ 프레임 선택

def blur_score(gray: np.ndarray) -> float:
    """라플라시안 분산. 클수록 선명. 흐린 프레임은 보통 50 미만 (해상도·장면에 따라 다름, 확인 필요)."""
    import cv2

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class FrameInfo:
    file: str
    t_s: float
    lat: float | None
    lon: float | None
    rel_alt_m: float | None
    abs_alt_m: float | None
    blur: float
    datetime: str | None


def _deg_to_dms_rational(deg: float):
    deg = abs(deg)
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = round((m_float - m) * 60 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def write_gps_exif(jpeg_path: str | Path, lat: float, lon: float, alt_m: float | None,
                   datetime_str: str | None = None, focal_mm: float | None = None) -> None:
    """JPEG 에 GPS/시각/초점거리 EXIF 를 써 넣습니다 (ODM/COLMAP 이 초기 위치로 사용)."""
    import piexif

    gps = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(lon),
    }
    if alt_m is not None:
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt_m >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(alt_m) * 1000)), 1000)
    exif = {"0th": {}, "Exif": {}, "GPS": gps}
    exif["0th"][piexif.ImageIFD.Make] = b"DJI"
    if datetime_str:
        # EXIF 형식 "YYYY:MM:DD HH:MM:SS"
        try:
            dt = datetime.strptime(datetime_str[:19], "%Y-%m-%d %H:%M:%S")
            exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt.strftime("%Y:%m:%d %H:%M:%S").encode()
        except ValueError:
            pass
    if focal_mm:
        exif["Exif"][piexif.ExifIFD.FocalLength] = (int(round(focal_mm * 100)), 100)
    piexif.insert(piexif.dump(exif), str(jpeg_path))


def read_gps_exif(jpeg_path: str | Path) -> tuple[float, float, float | None] | None:
    """(lat, lon, alt) 를 EXIF 에서 읽습니다. 없으면 None. 테스트·검증용."""
    import piexif

    ex = piexif.load(str(jpeg_path))
    g = ex.get("GPS", {})
    if piexif.GPSIFD.GPSLatitude not in g:
        return None

    def dms(v):
        return v[0][0] / v[0][1] + v[1][0] / v[1][1] / 60 + v[2][0] / v[2][1] / 3600

    lat = dms(g[piexif.GPSIFD.GPSLatitude]) * (-1 if g.get(piexif.GPSIFD.GPSLatitudeRef) == b"S" else 1)
    lon = dms(g[piexif.GPSIFD.GPSLongitude]) * (-1 if g.get(piexif.GPSIFD.GPSLongitudeRef) == b"W" else 1)
    alt = None
    if piexif.GPSIFD.GPSAltitude in g:
        a = g[piexif.GPSIFD.GPSAltitude]
        alt = a[0] / a[1] * (-1 if g.get(piexif.GPSIFD.GPSAltitudeRef) == 1 else 1)
    return lat, lon, alt


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    srt: str | Path | None = None,
    every_sec: float | None = 1.0,
    every_m: float | None = None,
    min_blur: float = 30.0,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    prefix: str | None = None,
    log=print,
) -> list[FrameInfo]:
    """
    영상에서 프레임을 뽑아 JPEG 로 저장하고 SRT 의 GPS 를 EXIF 로 넣습니다.

    every_sec : 시간 간격(초). every_m 이 주어지면 무시.
    every_m   : 이동 거리 간격(m). SRT GPS 가 필요. 정지 비행 구간에서는 프레임이 안 나옵니다.
    min_blur  : 라플라시안 분산이 이보다 작으면 흐린 프레임으로 버림.
    반환값은 저장된 프레임 목록. 같은 폴더에 frames.csv 도 씁니다.
    """
    import cv2

    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or video.stem

    if srt is None:
        srt = find_or_extract_srt(video, log=log)
    recs = parse_srt(srt) if srt else []
    if not recs:
        log(f"[dji] SRT 없음 또는 비어 있음: GPS 없이 프레임만 뽑습니다 ({video.name}). "
            "재구성은 되지만 지리참조가 안 됩니다. DJI Fly/Pilot 앱에서 '영상 자막' 을 켜고 다시 촬영하거나, "
            "비행 기록(Flight Record) 또는 사진 EXIF 를 쓰세요.")
    elif not any(r.has_gps for r in recs):
        log("[dji] SRT 에 GPS 가 없습니다 (실내 또는 GPS 미수신).")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없음: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log(f"[dji] {video.name}: {fps:.2f} fps, {n_total} frames, SRT {len(recs)} records")

    step_frames = max(1, int(round((every_sec or 1.0) * fps))) if not every_m else 1
    frames: list[FrameInfo] = []
    last_lat = last_lon = None
    idx = 0
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        if not every_m and step_frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            break
        t_s = idx / fps
        rec = telemetry_at(recs, t_s)

        take = True
        if every_m:
            if rec is None or not rec.has_gps:
                take = False
            elif last_lat is not None and haversine_m(last_lat, last_lon, rec.lat, rec.lon) < every_m:
                take = False
        if take:
            b = blur_score(bgr)
            if b < min_blur:
                take = False
        if take:
            name = f"{prefix}_{len(frames):05d}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            lat = lon = rel = ab = None
            dt = None
            if rec is not None:
                dt = rec.datetime_utc
                rel, ab = rec.rel_alt_m, rec.abs_alt_m
                if rec.has_gps:
                    lat, lon = rec.lat, rec.lon
                    write_gps_exif(path, lat, lon, ab if ab is not None else rel, dt, rec.focal_len)
                    last_lat, last_lon = lat, lon
            frames.append(FrameInfo(name, t_s, lat, lon, rel, ab, b, dt))
        idx += step_frames
        if n_total and idx >= n_total:
            break
    cap.release()

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(FrameInfo.__dataclass_fields__))
        w.writeheader()
        for fr in frames:
            w.writerow(asdict(fr))
    log(f"[dji] 저장 {len(frames)} 장 -> {out_dir}")
    return frames


def flight_summary(recs: list[SrtRecord]) -> dict:
    """비행 요약 (고도·거리·시간). 프레임 간격 결정과 README 표에 씁니다."""
    gps = [r for r in recs if r.has_gps]
    out = {"records": len(recs), "with_gps": len(gps), "duration_s": (recs[-1].t_end_s - recs[0].t_start_s) if recs else 0.0}
    if gps:
        alts = [r.rel_alt_m for r in gps if r.rel_alt_m is not None]
        out["rel_alt_min_m"] = min(alts) if alts else None
        out["rel_alt_max_m"] = max(alts) if alts else None
        dist = 0.0
        for a, b in zip(gps[:-1], gps[1:]):
            dist += haversine_m(a.lat, a.lon, b.lat, b.lon)
        out["track_length_m"] = dist
        out["lat_center"] = float(np.mean([r.lat for r in gps]))
        out["lon_center"] = float(np.mean([r.lon for r in gps]))
    return out
