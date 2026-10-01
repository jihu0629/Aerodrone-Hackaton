"""일정 간격 프레임 저장.

수신 스레드의 콜백에서는 '저장할지' 만 판단해 큐에 넣고, 실제 디스크 쓰기는 별도 스레드가 한다.
디스크가 느려 큐가 가득 차면 그 프레임은 **버리고 frames.csv 에 saved=0 으로 남긴다** (조용히 사라지지 않게).
frame_id 는 세션 전체에서 이어지는 번호, connection_id 는 재접속 구간 번호다.
"""
from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np

from .session import Session
from .stream import FrameMeta


class FrameSaver:
    def __init__(self, session: Session, cfg_frames: dict):
        self.s = session
        self.cfg = cfg_frames
        self.enabled = bool(cfg_frames.get("enabled_on_start", True))
        self.interval = float(cfg_frames.get("interval_s", 1.0))
        self.fmt = cfg_frames.get("format", "jpg").lower().strip(".")
        self.q: queue.Queue = queue.Queue(maxsize=int(cfg_frames.get("queue_size", 30)))
        self._lock = threading.Lock()
        self.frame_id = 0
        self.saved = 0
        self.dropped = 0
        self.failed = 0
        self._last_mono: Optional[float] = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="FrameSaver", daemon=True)
        self._t.start()

    def toggle(self) -> None:
        self.enabled = not self.enabled
        self.s.log.info("프레임 자동 저장 %s", "켬" if self.enabled else "끔")

    def on_frame(self, frame: np.ndarray, meta: FrameMeta) -> None:
        if not self.enabled:
            return
        if self._last_mono is not None and (meta.receive_monotonic_s - self._last_mono) < self.interval:
            return
        self._last_mono = meta.receive_monotonic_s
        self.enqueue(frame, meta, note="interval")

    def enqueue(self, frame: np.ndarray, meta: FrameMeta, note: str = "") -> None:
        with self._lock:
            self.frame_id += 1
            fid = self.frame_id
        fname = f"f{fid:06d}_c{meta.connection_id:02d}.{self.fmt}"
        try:
            self.q.put_nowait((fid, fname, frame.copy(), meta, note))
        except queue.Full:
            self.dropped += 1
            self.s.log.warning("프레임 저장 큐가 가득 차 frame_id=%d 를 버림 (디스크 쓰기 지연)", fid)
            self._write_row(fid, fname, meta, saved=0, note=f"{note};dropped_queue_full")

    def _write_row(self, fid: int, fname: str, meta: FrameMeta, saved: int, note: str) -> None:
        self.s.frames_csv.write({
            "session_id": self.s.session_id, "frame_id": fid, "file": f"frames/{fname}" if saved else "",
            "connection_id": meta.connection_id, "conn_frame_index": meta.conn_frame_index,
            "video_receive_time_utc": meta.receive_time_utc, "receive_monotonic_s": f"{meta.receive_monotonic_s:.3f}",
            "stream_pos_ms": "" if meta.stream_pos_ms is None else f"{meta.stream_pos_ms:.1f}",
            "width": meta.width, "height": meta.height, "saved": saved, "note": note,
        })

    def _loop(self) -> None:
        params = []
        if self.fmt in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.get("jpeg_quality", 95))]
        while not self._stop.is_set() or not self.q.empty():
            try:
                fid, fname, frame, meta, note = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            path = self.s.frames_dir / fname
            try:
                ok = cv2.imwrite(str(path), frame, params)
            except Exception as e:
                ok = False
                self.s.log.error("프레임 저장 오류 %s: %r", fname, e)
            if ok:
                self.saved += 1
                self._write_row(fid, fname, meta, 1, note)
            else:
                self.failed += 1
                self.s.log.error("프레임 저장 실패: %s (폴더 권한·디스크 용량 확인)", path)
                self._write_row(fid, fname, meta, 0, f"{note};write_failed")

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=15)
        if not self.q.empty():
            self.s.log.warning("종료 시 저장되지 않은 프레임 %d 개", self.q.qsize())
