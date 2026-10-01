"""3단계-a: 3D 복원용 프레임 선별 (흐림·중복 제거) → <세션>/sfm/images + selection.csv

  python scripts/24_select_frames.py data/sessions/drone_xxx
  python scripts/24_select_frames.py data/sessions/drone_xxx --blur-min 40 --min-change 0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.config import load_config  # noqa: E402
from dronecap.sfm import select_frames  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session")
    ap.add_argument("--config", default="config/dronecap.yml")
    ap.add_argument("--frames", help="프레임 폴더 (기본: 세션/frames)")
    ap.add_argument("--out", help="출력 폴더 (기본: 세션/sfm/images)")
    ap.add_argument("--blur-min", type=float); ap.add_argument("--min-change", type=float); ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    c = load_config(a.config)["sfm"]
    s = Path(a.session)
    rows = select_frames(a.frames or s / "frames", a.out or s / "sfm" / "images",
                         c["blur_min"] if a.blur_min is None else a.blur_min,
                         c["min_change"] if a.min_change is None else a.min_change,
                         c["max_frames"] if a.max_frames is None else a.max_frames)
    n = sum(r["selected"] for r in rows)
    reasons = {}
    for r in rows:
        if not r["selected"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    blurs = sorted(float(r["blur"]) for r in rows if r["blur"])
    print(f"전체 {len(rows)} → 선택 {n}, 제외 {reasons}")
    if blurs:
        print(f"blur 분포: min {blurs[0]:.0f} / 중앙 {blurs[len(blurs) // 2]:.0f} / max {blurs[-1]:.0f}  (기준 조정에 참고)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
