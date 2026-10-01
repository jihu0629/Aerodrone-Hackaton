"""1단계 메인 루프: 수신 스레드 + 프레임 저장 + 녹화 + 미리보기 창.

키 조작 (미리보기 창이 활성일 때)
  q / ESC : 종료            r : 녹화 시작/종료        s : 지금 프레임 1장 즉시 저장
  f       : 자동 프레임 저장 켬/끔                   h : HUD 표시 켬/끔
"""
from __future__ import annotations

import signal
import time
from typing import Optional

import cv2
import numpy as np

from . import stream as st
from .frames import FrameSaver
from .recorder import RecordingManager
from .session import Session
from .timeutil import monotonic


class CaptureApp:
    def __init__(self, cfg: dict, display: Optional[bool] = None, duration_s: Optional[float] = None,
                 record_on_start: Optional[bool] = None):
        self.cfg = cfg
        self.display = cfg["display"]["enabled"] if display is None else display
        self.duration = duration_s
        self.session = Session(cfg["session"]["root"], cfg["session"]["id_prefix"], cfg)
        self.log = self.session.log
        for w in cfg.get("_warnings", []):
            self.log.warning(w)
        self.frames = FrameSaver(self.session, cfg["frames"])
        self.rec = RecordingManager(self.session, cfg["record"], cfg["stream"])
        self.reader = st.StreamReader(cfg["stream"], self._on_frame, self._on_event)
        self.hud = bool(cfg["display"].get("hud", True))
        self._quit = False
        want_rec = cfg["record"]["enabled_on_start"] if record_on_start is None else record_on_start
        if want_rec:
            self.rec.wanted = True
        self.log.info("세션 폴더: %s", self.session.dir)
        self.log.info("스트림 주소: %s (transport=%s)", cfg["stream"]["url"], cfg["stream"]["rtsp_transport"])

    # 수신 스레드에서 호출됨 — 가볍게
    def _on_frame(self, frame: np.ndarray, meta: st.FrameMeta) -> None:
        self.frames.on_frame(frame, meta)
        self.rec.on_frame(frame, meta.connection_id)

    def _on_event(self, level: str, msg: str) -> None:
        getattr(self.log, level, self.log.info)(msg)

    # ---- 메인 루프 ----
    def run(self) -> dict:
        self.reader.start()
        t0 = monotonic()
        try:
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "_quit", True))
        except Exception:
            pass
        if self.display:
            try:
                cv2.namedWindow(self.cfg["display"]["window_name"], cv2.WINDOW_NORMAL)
            except cv2.error as e:
                self.log.error("창을 만들 수 없습니다(opencv-python-headless 가 설치돼 있으면 opencv-python 으로 교체). "
                               "--no-display 로 실행하세요. 원인: %s", e)
                self.display = False
        last_tick = 0.0
        last_shown_seq = -1
        try:
            while not self._quit:
                now = monotonic()
                if self.duration is not None and now - t0 >= self.duration:
                    self.log.info("지정한 실행 시간 %.1fs 도달 → 종료", self.duration)
                    break
                if self.reader.state == st.STATE_ENDED and (self.reader.frames_total > 0 or now - t0 > 2):
                    self.log.info("입력이 끝나 종료합니다")
                    break
                if now - last_tick >= 0.5:
                    self.rec.tick(self.reader.state == st.STATE_CONNECTED, self.reader.connection_id, self.reader.src_fps)
                    last_tick = now
                if self.display:
                    self._draw()
                    key = cv2.waitKey(15) & 0xFF
                    if key in (ord("q"), 27):
                        self._quit = True
                    elif key == ord("r"):
                        self.rec.toggle()
                    elif key == ord("s"):
                        self._snapshot()
                    elif key == ord("f"):
                        self.frames.toggle()
                    elif key == ord("h"):
                        self.hud = not self.hud
                    try:
                        if cv2.getWindowProperty(self.cfg["display"]["window_name"], cv2.WND_PROP_VISIBLE) < 1:
                            self._quit = True
                    except cv2.error:
                        pass
                else:
                    time.sleep(0.05)
        finally:
            stats = self._shutdown()
        return stats

    def _snapshot(self) -> None:
        latest = self.reader.latest()
        if latest is None:
            self.log.warning("아직 받은 프레임이 없어 저장할 수 없습니다")
            return
        frame, meta = latest
        self.frames.enqueue(frame, meta, note="manual")
        self.log.info("수동 저장 요청 (connection_id=%d, idx=%d)", meta.connection_id, meta.conn_frame_index)

    def _draw(self) -> None:
        latest = self.reader.latest()
        if latest is None:
            canvas = np.zeros((360, 640, 3), np.uint8)
        else:
            canvas = latest[0]
            mw = int(self.cfg["display"].get("max_width", 1280))
            if canvas.shape[1] > mw:
                h = int(canvas.shape[0] * mw / canvas.shape[1])
                canvas = cv2.resize(canvas, (mw, h), interpolation=cv2.INTER_AREA)
            else:
                canvas = canvas.copy()
        if self.hud:
            self._hud(canvas)
        cv2.imshow(self.cfg["display"]["window_name"], canvas)

    def _hud(self, img: np.ndarray) -> None:
        r = self.reader
        color = {st.STATE_CONNECTED: (0, 220, 0), st.STATE_CONNECTING: (0, 200, 255),
                 st.STATE_RECONNECTING: (0, 120, 255)}.get(r.state, (200, 200, 200))
        lines = [
            (f"{r.state}  conn#{r.connection_id}  reconnects={r.reconnects}  fps={r.fps:.1f}", color),
            (f"recv={r.frames_total}  preview_skipped={r.preview_skipped}  "
             f"frames saved={self.frames.saved} dropped={self.frames.dropped} auto={'on' if self.frames.enabled else 'off'}", (255, 255, 255)),
            (self.rec.status_text(), (0, 0, 255) if self.rec.wanted else (180, 180, 180)),
            (f"session={self.session.session_id}", (200, 200, 200)),
            ("q:quit r:record s:snapshot f:auto-frames h:hud", (160, 160, 160)),
        ]
        if r.state != st.STATE_CONNECTED and r.last_error:
            lines.insert(1, (r.last_error, (0, 120, 255)))
        y = 22
        for text, c in lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            y += 22

    def _shutdown(self) -> dict:
        self.log.info("종료 중...")
        self.reader.stop()
        self.reader.join(timeout=5)
        self.rec.close()
        self.frames.close()
        if self.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        stats = {
            "frames_received": self.reader.frames_total,
            "connections": self.reader.connection_id,
            "reconnects": self.reader.reconnects,
            "preview_skipped_frames": self.reader.preview_skipped,
            "frames_saved": self.frames.saved,
            "frames_dropped_queue_full": self.frames.dropped,
            "frames_write_failed": self.frames.failed,
            "recording_segments": self.rec.seg_count,
            "last_stream_error": self.reader.last_error,
        }
        self.log.info("통계: %s", stats)
        self.session.close(stats)
        return stats
