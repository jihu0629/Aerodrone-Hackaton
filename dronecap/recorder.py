"""영상 녹화.

두 가지 백엔드
  ffmpeg (권장) : ffmpeg.exe 가 RTSP 를 **따로 한 번 더 열어** 패킷을 그대로(-c copy) 파일에 쓴다.
                  재인코딩 없음(화질 손실 0, CPU 거의 안 씀), 오디오(AAC) 포함.
                  미리보기 스레드와 독립이라 미리보기가 느려도 녹화는 영향 없다.
                  기본 컨테이너는 '조각 mp4'(fragmented) 라서 프로그램이 비정상 종료돼도 그 전까지는 재생된다.
  opencv        : ffmpeg 가 없을 때. 미리보기가 디코드한 프레임을 다시 인코딩해서 저장.
                  화질 손실·CPU 사용, **오디오 없음**, 수신 fps 가 흔들리면 재생 속도가 실제와 달라질 수 있다.

연결이 끊기면 구간(segment)을 닫고, 다시 연결되면 **새 파일**로 시작한다.
한 파일 안에서 시간이 건너뛰어 이어 붙는 일을 막기 위해서다. recordings.csv 에 구간별 시작·종료 시각이 남는다.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .session import Session
from .timeutil import monotonic, utc_iso


def find_ffmpeg(path_or_name: str) -> Optional[str]:
    """ffmpeg 실행 파일 경로를 찾는다. 없으면 None."""
    p = Path(path_or_name)
    if p.is_file():
        return str(p)
    return shutil.which(path_or_name)


class _Segment:
    def __init__(self, segment_id: int, connection_id: int, file: Path, backend: str, reencoded: bool, audio: bool):
        self.segment_id = segment_id
        self.connection_id = connection_id
        self.file = file
        self.backend = backend
        self.reencoded = reencoded
        self.audio = audio
        self.start_utc = utc_iso()      # 녹화 프로세스를 시작한 시각
        self.start_mono = monotonic()
        self.first_data_utc: str | None = None   # 파일에 데이터가 처음 쓰인 시각 (스트림이 실제로 들어온 때). 이 값이 영상 시작 기준


class FfmpegRecorder:
    def __init__(self, ffmpeg: str, url: str, transport: str, container: str, log_dir: Path):
        self.ffmpeg = ffmpeg
        self.url = url
        self.transport = transport
        self.container = container
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self._log_f = None

    def start(self, out: Path, segment_id: int) -> None:
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
        if self.url.lower().startswith("rtsp://"):
            cmd += ["-rtsp_transport", self.transport]
        elif not self.url.lower().startswith(("rtmp://", "http", "udp://", "srt://")):
            cmd += ["-re"]  # 파일 입력일 때는 실시간 속도로 읽어 실제 스트림처럼 동작시킨다(테스트용)
        cmd += ["-i", self.url, "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy"]
        if self.container == "mp4":
            # 조각 mp4: 키프레임마다 조각을 닫아 중간에 죽어도 그때까지 재생 가능. 일반 mp4 는 끝에 moov 를 써야만 열린다.
            cmd += ["-f", "mp4", "-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
        elif self.container == "mkv":
            cmd += ["-f", "matroska"]
        elif self.container == "ts":
            cmd += ["-f", "mpegts"]
        cmd += [str(out)]
        self._log_f = open(self.log_dir / f"ffmpeg_seg{segment_id:03d}.log", "w", encoding="utf-8")
        self._log_f.write(" ".join(cmd) + "\n"); self._log_f.flush()
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._log_f, **kw)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 8.0) -> int | None:
        """ffmpeg 에 'q' 를 보내 파일을 정상적으로 닫게 한다. 안 끝나면 강제 종료."""
        if self.proc is None:
            return None
        rc = self.proc.poll()
        if rc is None:
            try:
                self.proc.stdin.write(b"q\n"); self.proc.stdin.flush()
            except Exception:
                pass
            try:
                rc = self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    rc = self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); rc = self.proc.wait()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if self._log_f:
            self._log_f.close(); self._log_f = None
        self.proc = None
        return rc


class OpenCVRecorder:
    """디코드된 프레임을 다시 인코딩해 저장(별도 쓰기 스레드). 오디오 없음."""

    def __init__(self, fourcc: str, fps: float, queue_size: int = 60):
        self.fourcc = fourcc
        self.fps = fps
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.written = 0
        self._w: cv2.VideoWriter | None = None
        self._t: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None

    def start(self, out: Path, size: tuple[int, int]) -> None:
        self._w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"VideoWriter 를 열 수 없음: {out} (fourcc={self.fourcc}, fps={self.fps}, size={size})")
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, name="OpenCVRecorder", daemon=True)
        self._t.start()

    def push(self, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        while not self._stop.is_set() or not self.q.empty():
            try:
                f = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._w.write(f); self.written += 1
            except Exception as e:
                self.error = repr(e)

    def alive(self) -> bool:
        return self._t is not None and self._t.is_alive()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=10)
        if self._w:
            self._w.release(); self._w = None


class RecordingManager:
    """녹화 on/off 와 구간 관리. 메인 루프에서 tick() 을 주기적으로 불러 준다."""

    def __init__(self, session: Session, cfg_record: dict, cfg_stream: dict):
        self.s = session
        self.cfg = cfg_record
        self.url = cfg_stream["url"]
        self.transport = cfg_stream.get("rtsp_transport", "tcp")
        self.stall = float(cfg_stream.get("stall_timeout_s", 5.0))
        self.wanted = False
        self.backend = cfg_record.get("backend", "ffmpeg")
        self.ffmpeg_path: str | None = None
        self.seg_count = 0
        self.seg: _Segment | None = None
        self._ff: FfmpegRecorder | None = None
        self._cv: OpenCVRecorder | None = None
        self._cv_conn: int | None = None
        self._last_size = -1
        self._last_grow_mono = 0.0
        self.last_error: str | None = None
        if self.backend == "ffmpeg":
            self.ffmpeg_path = find_ffmpeg(cfg_record.get("ffmpeg_path", "ffmpeg"))
            if self.ffmpeg_path is None:
                self.s.log.error("ffmpeg 를 찾지 못했습니다. record.ffmpeg_path 에 전체 경로를 적거나 PATH 에 추가하세요. "
                                 "임시로 opencv 백엔드(재인코딩, 오디오 없음)로 전환합니다.")
                self.backend = "opencv"

    # ---- 사용자 조작 ----
    def toggle(self) -> None:
        self.wanted = not self.wanted
        self.s.log.info("녹화 %s 요청", "시작" if self.wanted else "종료")
        if not self.wanted:
            self._close_segment("user_stop")

    def status_text(self) -> str:
        if not self.wanted:
            return "REC: off"
        if self.seg is None:
            return "REC: waiting for stream"
        return f"REC: {self.backend} seg{self.seg.segment_id:03d} {self.seg.file.name}"

    # ---- 프레임 콜백(opencv 백엔드만 사용) ----
    def on_frame(self, frame: np.ndarray, connection_id: int) -> None:
        if self.backend != "opencv" or not self.wanted:
            return
        if self.seg is not None and self._cv_conn != connection_id:
            self._close_segment("connection_changed")
        if self.seg is None:
            self._open_opencv(frame, connection_id)
        if self._cv:
            self._cv.push(frame)

    # ---- 주기 호출 ----
    def tick(self, stream_connected: bool, connection_id: int, src_fps: float | None) -> None:
        if self.backend == "ffmpeg":
            if self.seg is not None and self._ff is not None:
                if not self._ff.alive():
                    self._close_segment("ffmpeg_exited(stream ended?)")
                else:
                    self._check_growth()
            if self.wanted and self.seg is None and stream_connected:
                self._open_ffmpeg(connection_id)
        else:
            if self.seg is not None and not stream_connected:
                self._close_segment("connection_lost")

    def _check_growth(self) -> None:
        """ffmpeg 가 살아 있어도 파일이 안 커지면(스트림 정지) 구간을 닫아 다음에 새로 열게 한다."""
        try:
            size = self.seg.file.stat().st_size
        except FileNotFoundError:
            size = -1
        now = monotonic()
        if size > 0 and self.seg.first_data_utc is None:
            self.seg.first_data_utc = utc_iso()
        if size != self._last_size:
            self._last_size = size; self._last_grow_mono = now
        elif now - self._last_grow_mono > self.stall * 2:
            self.s.log.warning("녹화 파일이 %.0fs 동안 커지지 않음 → 구간 종료 후 재시작", self.stall * 2)
            self._close_segment("stalled")

    def _new_segment(self, connection_id: int, ext: str, backend: str, reencoded: bool, audio: bool) -> _Segment:
        self.seg_count += 1
        f = self.s.video_dir / f"{self.s.session_id}_seg{self.seg_count:03d}.{ext}"
        return _Segment(self.seg_count, connection_id, f, backend, reencoded, audio)

    def _open_ffmpeg(self, connection_id: int) -> None:
        seg = self._new_segment(connection_id, self.cfg.get("container", "mp4"), "ffmpeg", False, True)
        try:
            self._ff = FfmpegRecorder(self.ffmpeg_path, self.url, self.transport, self.cfg.get("container", "mp4"), self.s.video_dir)
            self._ff.start(seg.file, seg.segment_id)
        except Exception as e:
            self.last_error = f"ffmpeg 시작 실패: {e!r}"
            self.s.log.error(self.last_error)
            self.wanted = False
            return
        self.seg = seg
        self._last_size = -1; self._last_grow_mono = monotonic()
        self.s.log.info("녹화 구간 시작: %s (ffmpeg copy, 오디오 포함)", seg.file.name)

    def _open_opencv(self, frame: np.ndarray, connection_id: int) -> None:
        seg = self._new_segment(connection_id, "mp4", "opencv", True, False)
        fps = float(self.cfg.get("opencv_fps_fallback", 30.0))
        try:
            self._cv = OpenCVRecorder(self.cfg.get("opencv_fourcc", "mp4v"), fps)
            self._cv.start(seg.file, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            self.last_error = f"OpenCV 녹화 시작 실패: {e!r}"
            self.s.log.error(self.last_error)
            self.wanted = False
            self._cv = None
            return
        seg.first_data_utc = utc_iso()
        self.seg = seg; self._cv_conn = connection_id
        self.s.log.info("녹화 구간 시작: %s (opencv 재인코딩, 오디오 없음, fps=%.1f 고정)", seg.file.name, fps)

    def _close_segment(self, reason: str) -> None:
        seg = self.seg
        if seg is None:
            return
        self.seg = None
        if self._ff is not None:
            rc = self._ff.stop(); self._ff = None
            self.s.log.info("녹화 구간 종료: %s (reason=%s, ffmpeg rc=%s)", seg.file.name, reason, rc)
        if self._cv is not None:
            self._cv.stop()
            if self._cv.dropped:
                self.s.log.warning("opencv 녹화 중 큐가 꽉 차 %d 프레임을 버렸습니다", self._cv.dropped)
            self.s.log.info("녹화 구간 종료: %s (reason=%s, written=%d)", seg.file.name, reason, self._cv.written)
            self._cv = None
        try:
            size = seg.file.stat().st_size
        except FileNotFoundError:
            size = 0
        self.s.recordings_csv.write({
            "session_id": self.s.session_id, "segment_id": seg.segment_id, "connection_id": seg.connection_id,
            "backend": seg.backend, "file": str(seg.file.relative_to(self.s.dir)).replace("\\", "/"),
            "reencoded": int(seg.reencoded), "audio": int(seg.audio),
            "start_time_utc": seg.start_utc, "first_data_time_utc": seg.first_data_utc or "",
            "start_monotonic_s": f"{seg.start_mono:.3f}",
            "stop_time_utc": utc_iso(), "stop_monotonic_s": f"{monotonic():.3f}", "stop_reason": reason, "bytes": size,
        })

    def close(self) -> None:
        self.wanted = False
        self._close_segment("app_exit")
