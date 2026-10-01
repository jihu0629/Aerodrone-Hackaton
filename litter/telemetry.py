"""
텔레메트리 읽기 — 사진/프레임마다 드론 위치·고도·짐벌 각도·화각.

지원하는 입력 (있는 것부터 자동으로):
  1) 텔레메트리 표 (CSV/JSON) — 업체가 따로 정리해 줄 경우
       열 이름 예: image, lat, lon, rel_alt, yaw, pitch, roll, f35 (또는 hfov)
  2) DJI 사진(JPG)의 EXIF + XMP (drone-dji:GimbalPitchDegree 등)
  3) DJI 동영상의 .SRT 자막 파일 (프레임별 위경도·고도·짐벌각)

결과는 {이미지 이름: {lat, lon, alt, yaw, pitch, roll, f35|hfov, width, height}}
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

# 여러 표기법 → 내부 이름
_KEYS = {
    "lat": ["lat", "latitude", "gps_lat", "위도"],
    "lon": ["lon", "lng", "longitude", "gps_lon", "경도"],
    "alt": ["rel_alt", "relative_altitude", "relativealtitude", "alt", "altitude", "height", "고도"],
    "yaw": ["gimbal_yaw", "gimbalyawdegree", "gb_yaw", "yaw", "heading", "flightyawdegree"],
    "pitch": ["gimbal_pitch", "gimbalpitchdegree", "gb_pitch", "pitch"],
    "roll": ["gimbal_roll", "gimbalrolldegree", "gb_roll", "roll"],
    "f35": ["f35", "focal35", "focallengthin35mmfilm", "focal_length_35mm"],
    "hfov": ["hfov", "fov", "hfov_deg"],
    "image": ["image", "file", "file_name", "filename", "frame", "name", "파일명"],
}


def _norm_record(rec):
    low = {str(k).lower().strip(): v for k, v in rec.items()}
    out = {}
    for key, names in _KEYS.items():
        for n in names:
            if n in low and low[n] not in ("", None):
                out[key] = low[n] if key == "image" else float(low[n])
                break
    return out


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [dict(v, image=k) for k, v in data.items()]
    else:
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        n = _norm_record(r)
        if "image" in n:
            out[Path(str(n.pop("image"))).name] = n
    return out


def _gps_to_deg(v, ref):
    d, m, s = (float(x) for x in v)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_jpg(path):
    """DJI JPG: EXIF(GPS·초점거리) + XMP(짐벌각·상대고도)."""
    from PIL import Image

    rec = {}
    with Image.open(path) as im:
        rec["width"], rec["height"] = im.size
        ex = im.getexif()
        gps = ex.get_ifd(0x8825)
        if gps.get(2) and gps.get(4):
            rec["lat"] = _gps_to_deg(gps[2], gps.get(1, "N"))
            rec["lon"] = _gps_to_deg(gps[4], gps.get(3, "E"))
        f35 = ex.get_ifd(0x8769).get(41989)
        if f35:
            rec["f35"] = float(f35)
    raw = Path(path).read_bytes()
    xmp = {k.decode().lower(): float(v) for k, v in re.findall(rb'drone-dji:(\w+)="([-+0-9.]+)"', raw)}
    rec.update(_norm_record(xmp))
    # XMP엔 기체 GPS(GpsLatitude)와 절대고도도 있다 — 짐벌각·상대고도를 명시적으로 우선
    for src, dst in [("gimbalyawdegree", "yaw"), ("gimbalpitchdegree", "pitch"),
                     ("gimbalrolldegree", "roll"), ("relativealtitude", "alt")]:
        if src in xmp:
            rec[dst] = xmp[src]
    return rec


def read_srt(path):
    """DJI SRT → [{frame, t_ms, lat, lon, alt, yaw, pitch, roll}, ...].
    기종마다 표기가 조금씩 달라서 [key: value] 쌍을 전부 긁은 뒤 이름으로 고른다."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        t_ms = (int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])) * 1000 + int(m[4]) if m else None
        kv = dict(re.findall(r"([A-Za-z_]+)\s*[:=]\s*([-+]?\d+\.?\d*)", " ".join(lines[2:])))
        rec = _norm_record(kv)
        fc = re.search(r"FrameCnt\s*:\s*(\d+)", block)
        rec["frame"] = int(fc[1]) if fc else int(lines[0]) if lines[0].strip().isdigit() else len(out) + 1
        rec["t_ms"] = t_ms
        # 프레임의 실제 시각 (DJI: "2026-09-30 14:22:31.123") — 미러링 핑 시각과 맞출 때 사용
        dt = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)", block)
        if dt:
            from datetime import datetime
            s = dt[1].replace("T", " ").replace(",", ".")
            try:
                rec["dt"] = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                pass
        out.append(rec)
    return out


def extract_video_frames(video, srt, out_dir, every_s=1.0, time_offset_s=0.0):
    """동영상에서 every_s초마다 프레임을 뽑고 SRT 텔레메트리를 붙인다.
    time_offset_s: 영상-로그 시간 어긋남 보정 (1초 = 초속 10 m 비행 시 10 m 오차)."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel = read_srt(srt) if srt else []
    t_arr = np.array([r["t_ms"] for r in tel if r.get("t_ms") is not None], float)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps * every_s)))
    res, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            name = f"{Path(video).stem}_f{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            rec = {"width": frame.shape[1], "height": frame.shape[0]}
            if len(t_arr):
                j = int(np.argmin(np.abs(t_arr - (i / fps + time_offset_s) * 1000)))
                rec.update({k: v for k, v in tel[j].items() if k not in ("frame", "t_ms")})
            res[name] = rec
        i += 1
    cap.release()
    return res


def collect(images_dir=None, table=None, image_names=()):
    """이미지 폴더 + (선택) 텔레메트리 표 → 이미지별 텔레메트리. 표가 EXIF보다 우선."""
    tel = {}
    if images_dir:
        for n in image_names:
            p = Path(images_dir) / n
            if p.suffix.lower() in (".jpg", ".jpeg") and p.exists():
                try:
                    tel[n] = read_jpg(p)
                except Exception as e:  # 손상된 EXIF 등 — 표로 채울 수 있게 넘어간다
                    print(f"  ⚠️ EXIF 읽기 실패 {n}: {e}")
    if table:
        for n, rec in read_table(table).items():
            tel.setdefault(n, {}).update(rec)
    return tel
