"""프레임 ↔ OCR 매칭 → matched.csv

  python scripts/23_sync.py data/sessions/drone_xxx                      # 세션 안의 frames.csv + telemetry.csv
  python scripts/23_sync.py data/sessions/drone_xxx --telemetry other.csv --offset 0.8 --tolerance 300
offset: OCR 시각 + offset = 영상 시각 (초). 양수면 OCR(미러링)이 영상(RTMP)보다 먼저 도착한 경우.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.config import load_config  # noqa: E402
from dronecap.sync import match  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="세션 폴더")
    ap.add_argument("--config", default="config/dronecap.yml")
    ap.add_argument("--telemetry", help="telemetry.csv 경로 (기본: 세션 폴더 안)")
    ap.add_argument("--offset", type=float, help="sync.offset_s 덮어쓰기")
    ap.add_argument("--tolerance", type=float, help="sync.tolerance_ms 덮어쓰기")
    ap.add_argument("--out", help="출력 (기본: 세션/matched.csv)")
    a = ap.parse_args()
    cfg = load_config(a.config)["sync"]
    s = Path(a.session)
    tele = Path(a.telemetry) if a.telemetry else s / "telemetry.csv"
    if not tele.exists():
        print(f"telemetry.csv 가 없습니다: {tele}"); return 1
    out = Path(a.out) if a.out else s / "matched.csv"
    st = match(s / "frames.csv", tele, out, cfg["field_map"],
               cfg["offset_s"] if a.offset is None else a.offset,
               cfg["tolerance_ms"] if a.tolerance is None else a.tolerance, session_id=s.name)
    print(f"프레임 {st['frames']}: 전체 매칭 {st['valid']}, 일부 {st['partial']}, 없음 {st['none']} → {out}")
    if st["fields_without_data"]:
        print(f"telemetry.csv 에 아예 없는 필드: {st['fields_without_data']} (화면에 없는 항목이면 config sync.field_map 에서 지우세요)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
