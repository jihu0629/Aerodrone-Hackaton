"""촬영 파일에 **기록된** 비행 데이터 읽기 — OCR 없이, SDK 없이 얻을 수 있는 공식 경로.

  영상(.MP4) + 자막(.SRT) : DJI Fly 카메라 설정 → '영상 자막(Video Caption/Subtitles)' 을 켜면 영상과 같은 이름의 .SRT 에
                            프레임마다 위도·경도·상대고도(rel_alt, 이륙점 기준)·절대고도(abs_alt)·초점거리·노출이 기록된다.
                            → 영상 프레임과 **같은 타임라인**이라 시간 동기화 문제가 없다.
  사진(.JPG)               : XMP 'drone-dji' 네임스페이스에 GpsLatitude/GpsLongitude/AbsoluteAltitude/RelativeAltitude,
                            Gimbal*Degree/Flight*Degree 가, EXIF 에 GPS 가 들어간다.
둘 다 비행이 끝난 뒤 파일을 복사해야 얻는다(실시간 아님). 실시간이 필요한 값은 2단계 OCR 로만 가능하다.

주의
  * rel_alt / RelativeAltitude = 이륙 지점 기준 높이. 바로 아래 지면까지 거리가 아니다.
  * abs_alt / AbsoluteAltitude = 기압·GPS 기반 해발 추정. 비행마다 수 m 이상 치우칠 수 있다. 축척(상대 변화)에는 쓸 수 있지만 절대 높이로 믿지 말 것.
  * GPS 수평 정밀도는 보통 수 m. COLMAP 결과를 미터로 맞추는 '축척·방향 기준' 으로는 충분하고, 카메라 위치 그 자체로 쓰기엔 거칠다.
  * Mini 5 Pro 의 실제 .SRT/XMP 필드명은 공개 예시 1건 기준이다. 실제 파일로 확인 필요 (파서는 없는 필드를 빈칸으로 둔다).
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Optional

from litter3d.srt import Telemetry, parse_srt, telemetry_at  # 기존 파서 재사용 (가벼운 모듈)

SRT_COLUMNS = ["index", "t_start_s", "t_end_s", "frame", "timestamp", "lat", "lon", "rel_alt_m", "abs_alt_m",
               "focal_len", "iso", "shutter", "fnum"]

# ---------------------------------------------------------------- SRT

def find_srt(video: str | Path) -> Optional[Path]:
    v = Path(video)
    for ext in (".SRT", ".srt"):
        p = v.with_suffix(ext)
        if p.exists():
            return p
    return None


def srt_to_csv(srt_path: str | Path, out_csv: str | Path) -> list[Telemetry]:
    tel = parse_srt(srt_path)
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SRT_COLUMNS)
        w.writeheader()
        for t in tel:
            d = t.to_dict()
            w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in SRT_COLUMNS})
    return tel


def srt_health(tel: list[Telemetry]) -> dict:
    """실제 파일을 받았을 때 형식이 맞는지 빨리 보는 요약."""
    n = len(tel)
    gps = sum(1 for t in tel if t.lat is not None and t.lon is not None)
    rel = sum(1 for t in tel if t.rel_alt_m is not None)
    return {"blocks": n, "with_gps": gps, "with_rel_alt": rel, "with_timestamp": sum(1 for t in tel if t.timestamp),
            "duration_s": tel[-1].t_end_s if tel else 0.0,
            "warning": "" if (n and gps == n and rel == n) else
            "GPS/고도가 비어 있는 블록이 있음 → 실제 SRT 형식이 예시와 다를 수 있음. 앞 10줄을 확인해 litter3d/srt.py 정규식을 맞출 것"}


# ---------------------------------------------------------------- XMP (사진)

_XMP_START = b"<x:xmpmeta"
_XMP_END = b"</x:xmpmeta>"
_DJI_ATTR = re.compile(r'drone-dji:([A-Za-z]+)="([^"]*)"')
_DJI_ELEM = re.compile(r"<drone-dji:([A-Za-z]+)>([^<]*)</drone-dji:\1>")
DJI_KEYS = ["GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "RelativeAltitude",
            "GimbalRollDegree", "GimbalYawDegree", "GimbalPitchDegree",
            "FlightRollDegree", "FlightYawDegree", "FlightPitchDegree", "RtkFlag", "CamReverse", "GimbalReverse"]


def read_xmp_packet(path: str | Path, max_bytes: int = 4 * 1024 * 1024) -> Optional[str]:
    data = Path(path).read_bytes()[:max_bytes]   # XMP 는 보통 파일 앞쪽 APP1 세그먼트에 있다
    i = data.find(_XMP_START)
    if i < 0:
        return None
    j = data.find(_XMP_END, i)
    if j < 0:
        return None
    return data[i:j + len(_XMP_END)].decode("utf-8", errors="ignore")


def parse_dji_xmp(xmp: str) -> dict:
    d: dict[str, str] = {}
    for k, v in _DJI_ATTR.findall(xmp):
        d[k] = v
    for k, v in _DJI_ELEM.findall(xmp):
        d.setdefault(k, v.strip())
    return d


def _flt(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace("+", ""))
    except ValueError:
        return None


def _exif_gps(path: Path) -> dict:
    """XMP 가 없을 때의 대안: EXIF GPS (Pillow). 고도는 GPSAltitude(해발) 만 있다."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS
        with Image.open(path) as im:
            exif = im.getexif()
            gps = exif.get_ifd(0x8825) if exif else None
            dt = exif.get(0x0132) if exif else None
    except Exception:
        return {}
    if not gps:
        return {"DateTime": dt} if dt else {}
    g = {GPSTAGS.get(k, k): v for k, v in gps.items()}

    def dms(v, ref):
        if not v:
            return None
        deg = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -deg if ref in ("S", "W") else deg
    out = {"GpsLatitude": dms(g.get("GPSLatitude"), g.get("GPSLatitudeRef")),
           "GpsLongitude": dms(g.get("GPSLongitude"), g.get("GPSLongitudeRef"))}
    alt = g.get("GPSAltitude")
    if alt is not None:
        a = float(alt)
        if g.get("GPSAltitudeRef") in (1, b"\x01"):
            a = -a
        out["AbsoluteAltitude"] = a
    if dt:
        out["DateTime"] = dt
    return out


def read_photo_meta(path: str | Path) -> dict:
    """사진 1장 → {file, source(xmp|exif|none), GpsLatitude, GpsLongitude, AbsoluteAltitude, RelativeAltitude, Gimbal*, Flight*, DateTime}"""
    p = Path(path)
    row: dict = {"file": p.name, "source": "none"}
    xmp = read_xmp_packet(p)
    if xmp:
        d = parse_dji_xmp(xmp)
        if d:
            row["source"] = "xmp"
            for k in DJI_KEYS:
                row[k] = _flt(d.get(k)) if k not in ("RtkFlag", "CamReverse", "GimbalReverse") else d.get(k)
            m = re.search(r'xmp:CreateDate="([^"]+)"', xmp)
            if m:
                row["DateTime"] = m.group(1)
    if row["source"] == "none":
        e = _exif_gps(p)
        if e.get("GpsLatitude") is not None:
            row["source"] = "exif"
        row.update(e)
    return row


# ---------------------------------------------------------------- 좌표 · COLMAP 기준 파일

def latlon_to_enu(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    """WGS84 위경도 → 원점(lat0, lon0, alt0) 기준 동(E)·북(N)·상(U) 미터. 수 km 범위에서 충분한 근사."""
    r = 6378137.0
    dlat = math.radians(lat - lat0); dlon = math.radians(lon - lon0)
    e = dlon * r * math.cos(math.radians((lat + lat0) / 2))
    n = dlat * r
    return e, n, alt - alt0


def write_colmap_ref(rows: Iterable[dict], out_txt: str | Path, name_key: str = "file", lat_key: str = "lat",
                     lon_key: str = "lon", alt_key: str = "rel_alt_m") -> int:
    """COLMAP model_aligner 가 읽는 기준 파일: 'image_name lat lon alt' (ref_is_gps=1, alignment_type=enu).

    alt 로 rel_alt(이륙점 기준)를 쓰는 이유: ENU 정렬에서는 높이의 **상대 변화**만 축척에 영향을 주고,
    abs_alt 는 비행마다 다른 바이어스를 품어 두 비행을 합칠 때 어긋난다. 한 비행 안에서는 어느 쪽이든 축척은 같다.
    """
    n = 0
    out = Path(out_txt); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            lat, lon, alt = _flt(r.get(lat_key)), _flt(r.get(lon_key)), _flt(r.get(alt_key))
            if lat is None or lon is None or alt is None:
                continue
            f.write(f"{r[name_key]} {lat:.8f} {lon:.8f} {alt:.3f}\n")
            n += 1
    return n


def frames_with_srt(video: str | Path, srt: str | Path, out_dir: str | Path, interval_s: float = 1.0,
                    fmt: str = "jpg", jpeg_quality: int = 95, max_frames: Optional[int] = None) -> list[dict]:
    """녹화 영상에서 interval 마다 프레임을 저장하고, 같은 시각의 SRT 블록을 붙여 frames_srt.csv 로 쓴다.
    프레임 시각 = 영상 내 위치(t_s). SRT 도 같은 타임라인이므로 매칭 오차는 자막 블록 길이(1 프레임) 이내다."""
    import cv2
    tel = parse_srt(srt)
    out_dir = Path(out_dir); frames_dir = out_dir / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval_s)))
    rows: list[dict] = []
    i = 0
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if fmt.lower() in ("jpg", "jpeg") else []
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t_s = i / fps
            tt = telemetry_at(tel, t_s)
            name = f"v{len(rows) + 1:06d}.{fmt}"
            cv2.imwrite(str(frames_dir / name), frame, params)
            rows.append({"file": name, "video_frame_index": i, "t_s": round(t_s, 3),
                         "srt_index": tt.index if tt else "", "srt_timestamp": (tt.timestamp or "") if tt else "",
                         "lat": tt.lat if tt else None, "lon": tt.lon if tt else None,
                         "rel_alt_m": tt.rel_alt_m if tt else None, "abs_alt_m": tt.abs_alt_m if tt else None,
                         "focal_len": tt.focal_len if tt else None,
                         "srt_dt_ms": round((t_s - tt.t_start_s) * 1000, 1) if tt else ""})
            if max_frames and len(rows) >= max_frames:
                break
        i += 1
    cap.release()
    cols = ["file", "video_frame_index", "t_s", "srt_index", "srt_timestamp", "lat", "lon", "rel_alt_m", "abs_alt_m", "focal_len", "srt_dt_ms"]
    with open(out_dir / "frames_srt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return rows
