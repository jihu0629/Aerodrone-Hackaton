"""드론 영상 → SfM 용 프레임 추출.

- SRT 텔레메트리가 있으면 '이동 거리' 기준으로 프레임을 고른다 (전방 겹침 유지).
- 없으면 고정 시간 간격으로 추출한다.
- ODM 이 읽는 geo.txt (EPSG:4326, 파일명 경도 위도 고도) 를 같이 쓴다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .drone import MINI5PRO, DroneSpec
from .srt import Telemetry, haversine_m, parse_srt, telemetry_at


def _blur_score(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def extract_frames(video: str | Path, out_dir: str | Path, *, srt: str | Path | None = None,
                   front_overlap: float = 0.8, every_s: float = 1.0, altitude_m: float | None = None,
                   spec: DroneSpec = MINI5PRO, max_frames: int | None = None, blur_min: float = 30.0,
                   resize_w: int | None = None) -> list[dict]:
    """반환: 추출된 프레임 목록 [{file, t_s, lat, lon, rel_alt_m, abs_alt_m, blur}]."""
    video = Path(video); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tel: list[Telemetry] = []
    if srt is None and video.with_suffix(".SRT").exists():
        srt = video.with_suffix(".SRT")
    if srt is None and video.with_suffix(".srt").exists():
        srt = video.with_suffix(".srt")
    if srt is not None:
        tel = parse_srt(srt)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(fps * every_s)), 1)

    # 이동 거리 기준 간격
    interval_m = None
    if tel and any(t.lat is not None for t in tel):
        alt = altitude_m
        if alt is None:
            alts = [t.rel_alt_m for t in tel if t.rel_alt_m is not None and t.rel_alt_m > 1]
            alt = sorted(alts)[len(alts) // 2] if alts else None
        if alt:
            _, gh = spec.footprint_m(alt, "video")
            interval_m = gh * (1 - front_overlap)

    rows: list[dict] = []
    last_pos = None
    i = 0
    while True:
        ok = cap.grab()
        if not ok or i >= n and n > 0:
            break
        take = False
        t_s = i / fps
        tt = telemetry_at(tel, t_s) if tel else None
        if interval_m is not None and tt is not None and tt.lat is not None:
            if last_pos is None or haversine_m(last_pos[0], last_pos[1], tt.lat, tt.lon) >= interval_m:
                take = True
        elif i % step == 0:
            take = True
        if take:
            ok, frame = cap.retrieve()
            if not ok:
                break
            b = _blur_score(frame)
            if b < blur_min:      # 흐린 프레임 제외
                i += 1
                continue
            if resize_w and frame.shape[1] > resize_w:
                h = int(frame.shape[0] * resize_w / frame.shape[1])
                frame = cv2.resize(frame, (resize_w, h), interpolation=cv2.INTER_AREA)
            name = f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"file": name, "t_s": round(t_s, 3),
                         "lat": tt.lat if tt else None, "lon": tt.lon if tt else None,
                         "rel_alt_m": tt.rel_alt_m if tt else None, "abs_alt_m": tt.abs_alt_m if tt else None,
                         "blur": round(b, 1)})
            if tt is not None and tt.lat is not None:
                last_pos = (tt.lat, tt.lon)
            if max_frames and len(rows) >= max_frames:
                break
        i += 1
    cap.release()

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "t_s", "lat", "lon", "rel_alt_m", "abs_alt_m", "blur"])
        w.writeheader(); w.writerows(rows)
    geo = [r for r in rows if r["lat"] is not None]
    if geo:
        with open(out_dir / "geo.txt", "w", encoding="utf-8") as f:
            f.write("EPSG:4326\n")
            for r in geo:
                alt = r["abs_alt_m"] if r["abs_alt_m"] is not None else r["rel_alt_m"]
                f.write(f"{r['file']} {r['lon']} {r['lat']} {alt}\n")
    return rows
