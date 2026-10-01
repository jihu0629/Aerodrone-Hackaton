"""1단계: RTSP 영상 수신·표시·녹화·프레임 저장.

예)
  python scripts/20_capture.py                               # config/dronecap.yml 사용
  python scripts/20_capture.py --url rtsp://127.0.0.1:8554/live/drone --record
  python scripts/20_capture.py --url sample.mp4 --no-display --duration 10   # 로컬 파일로 동작 확인
  python scripts/20_capture.py --record --ocr-screen                          # 영상 수신 + 미러링 화면 OCR 동시
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecap.capture_app import CaptureApp  # noqa: E402
from dronecap.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/dronecap.yml")
    ap.add_argument("--url", help="설정 파일의 stream.url 대신 사용 (rtsp://... 또는 로컬 영상 파일)")
    ap.add_argument("--session-root", help="세션 폴더 상위 경로 (기본 data/sessions)")
    ap.add_argument("--record", action="store_true", help="시작하자마자 녹화")
    ap.add_argument("--no-display", action="store_true", help="창 없이 실행 (저장만)")
    ap.add_argument("--duration", type=float, help="이 시간(초) 뒤 자동 종료 (테스트용)")
    ap.add_argument("--interval", type=float, help="프레임 저장 간격(초) 덮어쓰기")
    ap.add_argument("--transport", choices=["tcp", "udp"], help="RTSP 전송 방식 덮어쓰기")
    ap.add_argument("--ocr-screen", action="store_true",
                    help="미러링된 아이폰 화면을 같은 세션에서 OCR (config/ocr_roi.json 필요). HUD 에 최신 값 표시, telemetry.csv 기록")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.url:
        cfg["stream"]["url"] = a.url
    if a.session_root:
        cfg["session"]["root"] = a.session_root
    if a.interval is not None:
        cfg["frames"]["interval_s"] = a.interval
    if a.transport:
        cfg["stream"]["rtsp_transport"] = a.transport

    app = CaptureApp(cfg, display=(not a.no_display), duration_s=a.duration,
                     record_on_start=True if a.record else None, live_ocr=a.ocr_screen)
    stats = app.run()
    print(f"\n세션 폴더: {app.session.dir}")
    print(f"받은 프레임 {stats['frames_received']}, 저장 {stats['frames_saved']}, 녹화 구간 {stats['recording_segments']}, "
          f"재접속 {stats['reconnects']}")
    if stats["frames_received"] == 0:
        print("프레임을 하나도 받지 못했습니다. docs/dronecap/README.md 의 '문제 점검 순서' 를 보세요.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
