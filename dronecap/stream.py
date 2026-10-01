"""RTSP(또는 파일) 영상 수신 스레드.

설계
  * 수신은 별도 스레드에서 쉬지 않고 cap.read() 를 돈다. 받은 프레임마다 콜백(on_frame)을 바로 호출한다.
    콜백은 가볍게(프레임 복사·큐 넣기 정도) 끝내야 한다. 저장·표시는 다른 스레드가 한다.
  * 미리보기는 '가장 최근 프레임'만 보여 준다(latest_frame). 표시가 느려 중간 프레임을 건너뛰어도
    기록(프레임 저장·녹화)에는 영향이 없다. 건너뛴 개수는 통계로 남긴다.
  * 연결이 끊기면 connection_id 를 올리고 재접속한다. 저장 쪽은 connection_id 가 바뀐 것을 보고
    영상 파일·시간 기록이 잘못 이어지지 않게 새 구간을 시작한다.
  * 수신 시각은 PC 시각이다. 드론 촬영 시각이 아니다.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .timeutil import monotonic, utc_iso

STATE_CONNECTING = "CONNECTING"
STATE_CONNECTED = "CONNECTED"
STATE_RECONNECTING = "RECONNECTING"
STATE_ENDED = "ENDED"      # 파일 입력이 끝났을 때
STATE_STOPPED = "STOPPED"


@dataclass
class FrameMeta:
    connection_id: int
    conn_frame_index: int          # 이 연결에서 몇 번째 디코드 프레임인지 (0부터)
    receive_time_utc: str          # PC 수신(디코드 완료) 시각
    receive_monotonic_s: float
    stream_pos_ms: Optional[float]  # 디코더가 알려 주는 스트림 내 위치(ms). RTSP 에서는 연결마다 0 근처에서 다시 시작할 수 있음. 없으면 None
    width: int
    height: int


def is_network_url(url: str) -> bool:
    return url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "srt://"))


class StreamReader(threading.Thread):
    def __init__(self, cfg: dict, on_frame: Callable[[np.ndarray, FrameMeta], None],
                 on_event: Callable[[str, str], None] | None = None):
        super().__init__(name="StreamReader", daemon=True)
        self.cfg = cfg
        self.url: str = cfg["url"]
        self.on_frame = on_frame
        self.on_event = on_event or (lambda level, msg: None)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self.state = STATE_CONNECTING
        self.connection_id = 0
        self.frames_total = 0
        self.last_frame_mono: float | None = None
        self.fps = 0.0
        self.last_error = ""
        self.reconnects = 0
        self.src_fps: float | None = None
        self.size: tuple[int, int] | None = None
        self._latest: tuple[np.ndarray, FrameMeta] | None = None
        self._latest_seq = 0
        self._taken_seq = 0
        self.preview_skipped = 0  # 미리보기가 가져가기 전에 덮어쓴 프레임 수 (기록 손실 아님)

    # ---- 미리보기용 ----
    def latest(self) -> tuple[np.ndarray, FrameMeta] | None:
        with self._lock:
            if self._latest is None:
                return None
            if self._latest_seq > self._taken_seq + 1:
                self.preview_skipped += self._latest_seq - self._taken_seq - 1
            self._taken_seq = self._latest_seq
            return self._latest

    def stop(self) -> None:
        self._stop_evt.set()

    # ---- 내부 ----
    def _open(self) -> cv2.VideoCapture | None:
        if is_network_url(self.url):
            opts = [f"rtsp_transport;{self.cfg.get('rtsp_transport', 'tcp')}"]
            if self.cfg.get("low_latency", True):
                # nobuffer/low_delay: 디코더가 프레임을 모아 두지 않고 바로 내보냄 → 미리보기 지연 감소
                opts += ["fflags;nobuffer", "flags;low_delay"]
            to_us = int(float(self.cfg.get("open_timeout_s", 10.0)) * 1_000_000)
            opts.append(f"timeout;{to_us}")   # RTSP 소켓 대기 한도(마이크로초). FFmpeg 5+ 옵션 이름
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 지원하는 백엔드에서만 효과
        except Exception:
            pass
        return cap

    def run(self) -> None:
        backoff = float(self.cfg.get("reconnect_min_s", 1.0))
        bmax = float(self.cfg.get("reconnect_max_s", 10.0))
        stall = float(self.cfg.get("stall_timeout_s", 5.0))
        network = is_network_url(self.url)
        while not self._stop_evt.is_set():
            self.state = STATE_CONNECTING if self.connection_id == 0 else STATE_RECONNECTING
            self.on_event("info", f"연결 시도 #{self.connection_id + 1}: {self.url}")
            cap = self._open()
            if cap is None:
                self.last_error = "스트림을 열 수 없음 (MediaMTX 실행 여부, 주소, 송출 중인지 확인)"
                self.on_event("warning", f"{self.last_error} → {backoff:.1f}s 후 재시도")
                if self._stop_evt.wait(backoff):
                    break
                backoff = min(backoff * 2, bmax)
                continue
            self.connection_id += 1
            if self.connection_id > 1:
                self.reconnects += 1
            backoff = float(self.cfg.get("reconnect_min_s", 1.0))
            self.src_fps = cap.get(cv2.CAP_PROP_FPS) or None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.size = (w, h) if w and h else None
            self.state = STATE_CONNECTED
            self.on_event("info", f"연결됨 (connection_id={self.connection_id}, {w}x{h}, fps≈{self.src_fps})")
            idx = 0
            win_t0 = monotonic(); win_n = 0
            last_ok = monotonic()
            # 파일 입력을 실제 스트림처럼 원본 fps 로 천천히 읽기 (테스트용). 네트워크 입력에는 적용 안 됨
            pace = (not network) and bool(self.cfg.get("file_realtime", True)) and bool(self.src_fps)
            pace_t0 = monotonic()
            while not self._stop_evt.is_set():
                if pace:
                    due = pace_t0 + idx / float(self.src_fps)
                    wait = due - monotonic()
                    if wait > 0:
                        time.sleep(wait)
                ok, frame = cap.read()
                now_m = monotonic()
                if not ok or frame is None:
                    if not network:
                        break                      # 파일: EOF
                    if (now_m - last_ok) >= stall:
                        break                      # 네트워크: stall 시간 넘게 프레임 없음 → 재접속
                    time.sleep(0.02)               # 잠깐 뒤 다시 읽기 (깨진 패킷 등 일시 오류)
                    continue
                last_ok = now_m
                meta = FrameMeta(self.connection_id, idx, utc_iso(), now_m,
                                 self._pos_ms(cap), frame.shape[1], frame.shape[0])
                idx += 1
                self.frames_total += 1
                self.last_frame_mono = now_m
                win_n += 1
                if now_m - win_t0 >= 1.0:
                    self.fps = win_n / (now_m - win_t0)
                    win_t0, win_n = now_m, 0
                with self._lock:
                    self._latest = (frame, meta)
                    self._latest_seq += 1
                try:
                    self.on_frame(frame, meta)
                except Exception as e:  # 콜백 오류가 수신을 멈추지 않게
                    self.on_event("error", f"프레임 콜백 오류: {e!r}")
            cap.release()
            if self._stop_evt.is_set():
                break
            if not network:
                self.state = STATE_ENDED
                self.on_event("info", "파일 입력이 끝났습니다 (EOF)")
                break
            self.fps = 0.0
            self.last_error = "스트림 끊김 (EOF 또는 수신 정지)"
            self.state = STATE_RECONNECTING
            self.on_event("warning", f"{self.last_error}. connection_id={self.connection_id} 종료, {backoff:.1f}s 후 재접속")
            if self._stop_evt.wait(backoff):
                break
            backoff = min(backoff * 2, bmax)
        if self.state != STATE_ENDED:
            self.state = STATE_STOPPED

    @staticmethod
    def _pos_ms(cap: cv2.VideoCapture) -> Optional[float]:
        try:
            v = cap.get(cv2.CAP_PROP_POS_MSEC)
            return float(v) if v and v > 0 else None
        except Exception:
            return None
