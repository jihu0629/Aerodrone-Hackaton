"""OCR 입력 소스. 세 가지를 같은 인터페이스(next() → (frame, info) | None)로 다룬다.

  ScreenSource : mss 로 PC 화면(미러링 창) 캡처. 미러링이 준비됐을 때.
  VideoSource  : 아이폰 화면 녹화 파일 등. 프레임 시각 = 파일 내 위치(source_time_s). 벽시계 시각은 파일 시작 시각을 따로 알아야 함.
  ImageSource  : 스크린샷 이미지 1장 또는 폴더. 모듈 시험용.
info: {"source": ..., "source_frame_index": int, "source_time_s": float|None}
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ScreenSource:
    def __init__(self, monitor: int = 1, region: Optional[list[int]] = None, interval_s: float = 0.5):
        try:
            import mss
        except ImportError as e:
            raise RuntimeError("mss 가 없습니다: pip install mss") from e
        self._mss = mss.mss()
        mons = self._mss.monitors
        if monitor >= len(mons):
            raise ValueError(f"모니터 {monitor} 없음 (사용 가능: 1~{len(mons) - 1})")
        mon = mons[monitor]
        if region:
            l, t, w, h = region
            self.box = {"left": mon["left"] + l, "top": mon["top"] + t, "width": w, "height": h}
        else:
            self.box = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        self.interval = interval_s
        self.i = 0
        self._next = time.monotonic()
        self.desc = f"screen monitor={monitor} box={self.box}"

    def next(self):
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + self.interval, time.monotonic())
        shot = self._mss.grab(self.box)
        frame = np.array(shot)[:, :, :3].copy()   # BGRA → BGR
        info = {"source": "screen", "source_frame_index": self.i, "source_time_s": None}
        self.i += 1
        return frame, info


class VideoSource:
    def __init__(self, path: str | Path, interval_s: float = 0.5):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없음: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, int(round(self.fps * interval_s)))
        self.i = 0
        self.desc = f"video {self.path.name} fps={self.fps:.2f} step={self.step}"

    def next(self):
        while True:
            ok = self.cap.grab()
            if not ok:
                return None
            idx = self.i
            self.i += 1
            if idx % self.step != 0:
                continue
            ok, frame = self.cap.retrieve()
            if not ok:
                return None
            return frame, {"source": f"video:{self.path.name}", "source_frame_index": idx, "source_time_s": idx / self.fps}


class ImageSource:
    def __init__(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            self.files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        else:
            self.files = [p]
        if not self.files:
            raise FileNotFoundError(f"이미지가 없음: {p}")
        self.i = 0
        self.desc = f"images {p} ({len(self.files)})"

    def next(self):
        if self.i >= len(self.files):
            return None
        f = self.files[self.i]
        img = cv2.imread(str(f))
        info = {"source": f"image:{f.name}", "source_frame_index": self.i, "source_time_s": None}
        self.i += 1
        if img is None:
            return np.zeros((1, 1, 3), np.uint8), {**info, "source": f"image:{f.name}:unreadable"}
        return img, info


def open_source(kind: str, target: Optional[str], cfg_ocr: dict):
    if kind == "screen":
        sc = cfg_ocr.get("screen", {}) or {}
        return ScreenSource(int(sc.get("monitor", 1)), sc.get("region"), float(cfg_ocr.get("interval_s", 0.5)))
    if kind == "video":
        return VideoSource(target, float(cfg_ocr.get("interval_s", 0.5)))
    if kind == "image":
        return ImageSource(target)
    raise ValueError(kind)
