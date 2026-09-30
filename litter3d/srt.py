"""DJI 영상 자막(.SRT) 텔레메트리 파서.

DJI Fly 에서 '영상 자막(Video Caption)' 을 켜면 영상과 같은 이름의 .SRT 파일에 프레임마다
위도·경도·상대고도(rel_alt, 이륙점 기준 m)·절대고도(abs_alt, 해발 m)·초점거리 등이 기록된다.
Mini 5 Pro 예시 (dji-drone-metadata-embedder 문서, skystamp-community issue #5):

    1
    00:00:00,000 --> 00:00:00,033
    FrameCnt: 1, DiffTime: 33ms
    2026-09-25 16:23:55.467
    [iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00]
    [latitude: 30.142288] [longitude: -95.768454] [rel_alt: 0.000 abs_alt: 65.972] [ct: 4711]

구형 포맷 "GPS (lon, lat, alt)" 도 함께 지원한다. 실제 파일에서 확인 필요.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_KV = re.compile(r"\[([a-zA-Z_]+)\s*:\s*([^\]\[]+?)\]")
_KV2 = re.compile(r"([a-zA-Z_]+)\s*:\s*(-?[\d.]+)")   # "[rel_alt: 0.000 abs_alt: 65.972]" 안의 2쌍
_FRAME = re.compile(r"(?:FrameCnt|SrtCnt)\s*:\s*(\d+)")
_GPS_OLD = re.compile(r"GPS\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")


@dataclass
class Telemetry:
    index: int
    t_start_s: float
    t_end_s: float
    frame: int | None
    timestamp: str | None
    lat: float | None
    lon: float | None
    rel_alt_m: float | None
    abs_alt_m: float | None
    focal_len: float | None
    iso: float | None = None
    shutter: str | None = None
    fnum: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_srt_text(text: str) -> list[Telemetry]:
    blocks = re.split(r"\n\s*\n", text.strip().replace("\r\n", "\n"))
    out: list[Telemetry] = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = len(out) + 1
        mt = _TIME.search(b)
        if not mt:
            continue
        t0 = _to_sec(*mt.groups()[:4])
        t1 = _to_sec(*mt.groups()[4:])
        kv: dict[str, str] = {}
        for k, v in _KV.findall(b):
            pairs = _KV2.findall(f"{k}: {v}")
            if pairs:                       # "[rel_alt: 0.000 abs_alt: 65.972]" 처럼 한 괄호에 여러 쌍
                for k2, v2 in pairs:
                    kv[k2] = v2
            else:
                kv[k] = v.strip()
        mf = _FRAME.search(b)
        md = _DATE.search(b)
        lat = _f(kv.get("latitude"))
        lon = _f(kv.get("longitude"))
        rel = _f(kv.get("rel_alt"))
        ab = _f(kv.get("abs_alt"))
        if lat is None:
            mg = _GPS_OLD.search(b)
            if mg:
                lon, lat, ab = _f(mg.group(1)), _f(mg.group(2)), _f(mg.group(3))
        out.append(Telemetry(
            index=idx, t_start_s=t0, t_end_s=t1,
            frame=int(mf.group(1)) if mf else None,
            timestamp=md.group(1) if md else None,
            lat=lat, lon=lon, rel_alt_m=rel, abs_alt_m=ab,
            focal_len=_f(kv.get("focal_len")),
            iso=_f(kv.get("iso")), shutter=kv.get("shutter"), fnum=_f(kv.get("fnum")),
        ))
    return out


def parse_srt(path: str | Path) -> list[Telemetry]:
    return parse_srt_text(Path(path).read_text(encoding="utf-8", errors="ignore"))


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def telemetry_at(tel: list[Telemetry], t_s: float) -> Telemetry | None:
    """시각 t(초)에 해당하는 자막 블록."""
    if not tel:
        return None
    lo, hi = 0, len(tel) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tel[mid].t_end_s < t_s:
            lo = mid + 1
        else:
            hi = mid
    return tel[lo]


def summarize(tel: Iterable[Telemetry]) -> dict:
    tel = list(tel)
    alts = [t.rel_alt_m for t in tel if t.rel_alt_m is not None]
    lats = [t.lat for t in tel if t.lat is not None]
    lons = [t.lon for t in tel if t.lon is not None]
    dist = 0.0
    prev = None
    for t in tel:
        if t.lat is None:
            continue
        if prev is not None:
            dist += haversine_m(prev.lat, prev.lon, t.lat, t.lon)
        prev = t
    return {
        "blocks": len(tel),
        "duration_s": tel[-1].t_end_s if tel else 0.0,
        "rel_alt_min_m": min(alts) if alts else None,
        "rel_alt_max_m": max(alts) if alts else None,
        "rel_alt_median_m": sorted(alts)[len(alts) // 2] if alts else None,
        "lat_range": (min(lats), max(lats)) if lats else None,
        "lon_range": (min(lons), max(lons)) if lons else None,
        "track_length_m": dist,
        "has_gps": bool(lats),
    }
