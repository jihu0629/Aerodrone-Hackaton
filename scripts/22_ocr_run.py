"""2단계-b: OCR 실행 → telemetry.csv.

  python scripts/22_ocr_run.py --image shots/                    # 스크린샷 폴더로 모듈 시험
  python scripts/22_ocr_run.py --video iphone_record.mp4         # 아이폰 화면 녹화
  python scripts/22_ocr_run.py --screen --session data/sessions/drone_xxx   # 미러링 화면 실시간 (세션 폴더에 기록)
--session 을 주면 그 폴더의 telemetry.csv 에, 아니면 --out 경로에 쓴다. --show 로 잘라낸 영역과 결과를 창으로 확인.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.config import load_config  # noqa: E402
from dronecap.ocr.roi import RoiSet  # noqa: E402
from dronecap.ocr.runner import OcrRunner  # noqa: E402
from dronecap.ocr.sources import open_source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/dronecap.yml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--video"); g.add_argument("--screen", action="store_true")
    ap.add_argument("--roi", help="ROI 파일 (기본: 설정 ocr.roi_file)")
    ap.add_argument("--session", help="세션 폴더. telemetry.csv 와 ocr_debug/ 를 여기에 기록")
    ap.add_argument("--out", help="--session 이 없을 때 출력 CSV 경로 (기본 data/ocr_test/telemetry.csv)")
    ap.add_argument("--interval", type=float, help="처리 간격(초) 덮어쓰기")
    ap.add_argument("--show", action="store_true", help="영역·결과를 창에 표시 (q 로 종료)")
    ap.add_argument("--max", type=int, help="최대 처리 횟수 (테스트용)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.interval is not None:
        cfg["ocr"]["interval_s"] = a.interval
    roi_path = a.roi or cfg["ocr"]["roi_file"]
    if not Path(roi_path).exists():
        print(f"ROI 파일이 없습니다: {roi_path}. 먼저 scripts/21_select_roi.py 를 실행하세요."); return 1
    rois = RoiSet.load(roi_path)
    if a.session:
        out = Path(a.session) / "telemetry.csv"; dbg = Path(a.session) / "ocr_debug"
    else:
        out = Path(a.out or "data/ocr_test/telemetry.csv"); dbg = out.parent / "ocr_debug"
    kind, target = ("image", a.image) if a.image else ("video", a.video) if a.video else ("screen", None)
    src = open_source(kind, target, cfg["ocr"])
    print(f"입력: {src.desc}\nROI: {[r.name for r in rois.rois]} (기준 {rois.reference_size})\n출력: {out}")
    runner = OcrRunner(cfg["ocr"], rois, out, dbg)
    n = 0
    try:
        while True:
            item = src.next()
            if item is None:
                break
            frame, info = item
            rows = runner.process(frame, info)
            n += 1
            summary = "  ".join(f"{r['field']}={r['value'] or '-'}{r['unit']}[{r['status']}]" for r in rows)
            print(f"#{n} {info.get('source')} {summary}")
            if a.show:
                view = frame.copy()
                for roi, r in zip(rois.rois, rows):
                    cv2.rectangle(view, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2)
                    cv2.putText(view, f"{r['field']}: {r['raw_text']} -> {r['value']} [{r['status']}]",
                                (roi.x, max(14, roi.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("ocr", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if a.max and n >= a.max:
                break
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
        if a.show:
            cv2.destroyAllWindows()
    print(f"완료: {n} 회 처리, {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
