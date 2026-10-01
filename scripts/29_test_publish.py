"""드론 없이 시험하기: 로컬 영상 파일을 ffmpeg 로 MediaMTX 에 RTMP 송출한다 (DJI Fly 흉내).

  python scripts/29_test_publish.py sample.mp4                 # rtmp://127.0.0.1:1935/live/drone 으로 송출
  python scripts/29_test_publish.py sample.mp4 --loop          # 반복 재생
  python scripts/29_test_publish.py --synthetic 20             # 영상 파일이 없으면 20초짜리 합성 영상을 만들어 송출

그 뒤 다른 터미널에서 scripts/20_capture.py 를 실행하면 실제 스트림과 같은 경로(RTMP→MediaMTX→RTSP)로 시험할 수 있다.
송출을 Ctrl+C 로 끊으면 서버 로그에 'closed: EOF' 가 찍히고, 수신 프로그램은 재접속 대기로 들어간다 → 재접속 동작 확인용.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def make_synthetic(path: Path, seconds: int, fps: int = 30, size=(1280, 720)) -> Path:
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("합성 영상 VideoWriter 열기 실패")
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    for i in range(seconds * fps):
        img = bg.copy()
        x = int((i / (seconds * fps)) * (w - 200))
        cv2.rectangle(img, (x, 200), (x + 200, 400), (0, 180, 255), -1)
        cv2.putText(img, f"frame {i:05d}  t={i / fps:6.2f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        vw.write(img)
    vw.release()
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="송출할 영상 파일")
    ap.add_argument("--synthetic", type=int, metavar="SECONDS", help="합성 영상을 만들어 송출")
    ap.add_argument("--rtmp", default="rtmp://127.0.0.1:1935/live/drone")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--reencode", action="store_true", help="H.264 가 아닌 영상일 때 libx264 로 변환해 송출")
    a = ap.parse_args()

    ff = shutil.which(a.ffmpeg) or (a.ffmpeg if Path(a.ffmpeg).is_file() else None)
    if not ff:
        print("ffmpeg 를 찾을 수 없습니다. PATH 또는 --ffmpeg 경로 확인", file=sys.stderr)
        return 1
    if a.synthetic:
        out = Path("data") / "synthetic_test.mp4"
        out.parent.mkdir(exist_ok=True)
        make_synthetic(out, a.synthetic)
        a.video = str(out)
        a.reencode = True  # mp4v(MPEG-4 Part 2) 는 RTMP/FLV 에 못 담으므로 H.264 로 변환
    if not a.video:
        ap.error("영상 파일 또는 --synthetic 필요")
    cmd = [ff, "-hide_banner", "-re"]
    if a.loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", a.video]
    if a.reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", "30", "-an"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-f", "flv", a.rtmp]
    print(" ".join(cmd))
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
