"""2단계-a: OCR 영역(ROI) 선택. 기준 화면을 보고 필드별로 사각형을 그려 config/ocr_roi.json 에 저장.

  python scripts/21_select_roi.py --image screenshot.png          # 아이폰 스크린샷/캡처 이미지
  python scripts/21_select_roi.py --video iphone_record.mp4 --at 12.0   # 화면 녹화의 12초 프레임
  python scripts/21_select_roi.py --screen                         # 미러링 창이 보이는 현재 PC 화면
설정의 ocr.fields 순서대로 묻는다. 화면에 없는 항목은 c 로 건너뛴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.config import load_config  # noqa: E402
from dronecap.ocr.roi import RoiSet, select_rois_interactive  # noqa: E402
from dronecap.ocr.sources import ImageSource, ScreenSource, VideoSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/dronecap.yml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--video"); g.add_argument("--screen", action="store_true")
    ap.add_argument("--at", type=float, default=0.0, help="--video 일 때 사용할 시각(초)")
    ap.add_argument("--out", help="저장 경로 (기본: 설정 ocr.roi_file)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.image:
        frame, _ = ImageSource(a.image).next(); desc = f"image {a.image}"
    elif a.video:
        src = VideoSource(a.video, 1.0)
        src.cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000)
        ok, frame = src.cap.read()
        if not ok:
            print("프레임을 읽을 수 없음"); return 1
        desc = f"video {a.video} @ {a.at}s"
    else:
        sc = cfg["ocr"].get("screen", {}) or {}
        frame, _ = ScreenSource(int(sc.get("monitor", 1)), sc.get("region"), 0.0).next(); desc = "screen"
    names = list((cfg["ocr"].get("fields") or {}).keys())
    print("필드:", names)
    for n in names:
        print(f"  {n}: {cfg['ocr']['fields'][n].get('desc', '')}")
    rois = select_rois_interactive(frame, names)
    if not rois:
        print("선택된 영역이 없습니다"); return 1
    rs = RoiSet((frame.shape[1], frame.shape[0]), desc, rois)
    out = a.out or cfg["ocr"]["roi_file"]
    rs.save(out)
    print(f"저장: {out} (기준 크기 {frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
