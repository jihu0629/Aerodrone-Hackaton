"""영상 프레임(frames.csv) ↔ OCR 텔레메트리(telemetry.csv) 시간 매칭.

원리
  * 두 기록 모두 'PC 수신 시각(UTC)' 이다. RTMP 경로와 미러링 경로의 지연이 다르므로
    같은 PC 시각이어도 같은 순간을 찍은 게 아니다. 그 차이를 offset_s 로 보정한다 (실측으로 정해야 함).
      영상 시각 ≈ OCR 시각 + offset_s
  * 프레임마다 (보정한) OCR 시각이 가장 가까운 샘플을 필드별로 고르고, |차이| 가 tolerance_ms 를 넘으면 연결하지 않는다.
  * 원본 telemetry.csv 는 건드리지 않고, 매칭 결과는 matched.csv 로 따로 쓴다.
  * telemetry_valid = 매핑된 모든 필드가 허용 오차 안에 있고 status 가 ok 일 때만 1.
    일부만 맞으면 0 이고, 맞은 필드만 값이 들어간다. 추측해서 채우지 않는다.

출력 열
  session_id, frame_id, frame_file, video_receive_time_utc, telemetry_receive_time_utc,
  relative_altitude_m, home_distance_m, horizontal_speed_mps, vertical_speed_mps,   (field_map 에 있는 것만)
  match_time_error_ms, telemetry_valid, offset_s_applied, fields_matched, fields_missing, unit_notes
"""
from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Optional

from .ocr.parse import convert_to
from .timeutil import parse_utc_iso

TARGET_UNITS = {
    "relative_altitude_m": "m", "home_distance_m": "m",
    "horizontal_speed_mps": "m/s", "vertical_speed_mps": "m/s",
}


def _load_frames(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("saved") in ("1", "True", "true")]
    for r in rows:
        r["_t"] = parse_utc_iso(r["video_receive_time_utc"]).timestamp()
    return rows


def _load_telemetry(path: Path) -> dict[str, list[tuple[float, dict]]]:
    """field → 시각순 [(epoch_s, row)]"""
    by: dict[str, list[tuple[float, dict]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = parse_utc_iso(r["telemetry_receive_time_utc"]).timestamp()
            by.setdefault(r["field"], []).append((t, r))
    for k in by:
        by[k].sort(key=lambda x: x[0])
    return by


def _nearest(series: list[tuple[float, dict]], t: float) -> Optional[tuple[float, dict]]:
    if not series:
        return None
    keys = [s[0] for s in series]
    i = bisect.bisect_left(keys, t)
    cands = []
    if i < len(series):
        cands.append(series[i])
    if i > 0:
        cands.append(series[i - 1])
    return min(cands, key=lambda s: abs(s[0] - t))


def match(frames_csv: str | Path, telemetry_csv: str | Path, out_csv: str | Path, field_map: dict,
          offset_s: float = 0.0, tolerance_ms: float = 500.0, session_id: str = "") -> dict:
    frames = _load_frames(Path(frames_csv))
    tele = _load_telemetry(Path(telemetry_csv))
    fmap = {k: v for k, v in (field_map or {}).items() if v}
    cols = ["session_id", "frame_id", "frame_file", "video_receive_time_utc", "telemetry_receive_time_utc",
            *fmap.keys(), "match_time_error_ms", "telemetry_valid", "offset_s_applied", "fields_matched",
            "fields_missing", "unit_notes"]
    stats = {"frames": len(frames), "valid": 0, "partial": 0, "none": 0, "fields_without_data": []}
    for col, fld in fmap.items():
        if fld not in tele:
            stats["fields_without_data"].append(fld)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for fr in frames:
            tv = fr["_t"]
            row = {"session_id": fr.get("session_id") or session_id, "frame_id": fr["frame_id"], "frame_file": fr["file"],
                   "video_receive_time_utc": fr["video_receive_time_utc"], "offset_s_applied": f"{offset_s:g}"}
            matched, missing, errs, notes, tele_times = [], [], [], [], []
            for col, fld in fmap.items():
                row[col] = ""
                hit = _nearest(tele.get(fld, []), tv - offset_s)   # 영상시각 - offset = OCR 시각
                if hit is None:
                    missing.append(f"{fld}:no_data"); continue
                t_ocr, r = hit
                err_ms = (t_ocr + offset_s - tv) * 1000.0
                if abs(err_ms) > tolerance_ms:
                    missing.append(f"{fld}:too_far({err_ms:+.0f}ms)"); continue
                if r.get("status") != "ok" or r.get("value", "") == "":
                    missing.append(f"{fld}:{r.get('status')}"); continue
                val = float(r["value"])
                unit = r.get("unit") or None
                target = TARGET_UNITS.get(col)
                if target:
                    if unit is None:
                        notes.append(f"{fld}:unit_not_seen_assumed_{target}")   # 화면에 단위가 안 보인 경우. 가정임을 남긴다
                    else:
                        cv, why = convert_to(val, unit, target)
                        if cv is None:
                            missing.append(f"{fld}:{why}"); continue
                        if why != "same_unit":
                            notes.append(f"{fld}:{why}")
                        val = cv
                row[col] = f"{val:.3f}"
                matched.append(fld); errs.append(err_ms); tele_times.append(r["telemetry_receive_time_utc"])
            row["telemetry_receive_time_utc"] = tele_times[0] if tele_times else ""
            row["match_time_error_ms"] = f"{max(errs, key=abs):.0f}" if errs else ""
            row["telemetry_valid"] = int(bool(fmap) and len(matched) == len(fmap))
            row["fields_matched"] = ";".join(matched)
            row["fields_missing"] = ";".join(missing)
            row["unit_notes"] = ";".join(notes)
            w.writerow(row)
            if row["telemetry_valid"]:
                stats["valid"] += 1
            elif matched:
                stats["partial"] += 1
            else:
                stats["none"] += 1
    return stats
