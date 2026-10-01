"""세션 폴더와 공통 기록 파일.

세션 하나 = 프로그램을 한 번 실행한 것. 폴더 구조:
  data/sessions/drone_20261001_110102_ab12/
    session.json      설정 스냅샷, 시작·종료 시각, 통계
    events.log        연결·재접속·오류·프레임 버림 등 사람이 읽는 로그
    frames.csv        저장한 프레임 목록 (+ 버린 프레임도 saved=0 으로 기록)
    recordings.csv    녹화 파일 구간 (연결이 끊기면 파일을 새로 시작하므로 여러 행)
    frames/           프레임 이미지
    video/            녹화 영상, ffmpeg 로그
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from .timeutil import monotonic, session_stamp, utc_iso

FRAME_COLUMNS = [
    "session_id", "frame_id", "file", "connection_id", "conn_frame_index",
    "video_receive_time_utc", "receive_monotonic_s", "stream_pos_ms",
    "width", "height", "saved", "note",
]
RECORDING_COLUMNS = [
    "session_id", "segment_id", "connection_id", "backend", "file", "reencoded", "audio",
    "start_time_utc", "first_data_time_utc", "start_monotonic_s", "stop_time_utc", "stop_monotonic_s", "stop_reason", "bytes",
]


class CsvLog:
    """스레드 안전한 append 전용 CSV. 행마다 flush 해서 비정상 종료 때도 남는다."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns
        self._lock = threading.Lock()
        new = not self.path.exists()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=columns, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in self.columns})
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._f.close()
            except Exception:
                pass


class Session:
    def __init__(self, root: str | Path, id_prefix: str = "drone", config: dict | None = None):
        self.session_id = f"{id_prefix}_{session_stamp()}_{uuid.uuid4().hex[:4]}"
        self.dir = Path(root) / self.session_id
        self.frames_dir = self.dir / "frames"
        self.video_dir = self.dir / "video"
        for d in (self.dir, self.frames_dir, self.video_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.start_utc = utc_iso()
        self.start_mono = monotonic()
        self.meta: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time_utc": self.start_utc,
            "start_monotonic_s": self.start_mono,
            "time_note": "모든 *_utc 는 PC 수신 시각. 촬영·측정 시각이 아님. monotonic 은 이 세션 안에서만 비교 가능.",
            "config": {k: v for k, v in (config or {}).items() if not k.startswith("_")},
        }
        self.frames_csv = CsvLog(self.dir / "frames.csv", FRAME_COLUMNS)
        self.recordings_csv = CsvLog(self.dir / "recordings.csv", RECORDING_COLUMNS)
        self.log = self._make_logger()
        self.save_meta()

    def _make_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"dronecap.{self.session_id}")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(self.dir / "events.log", encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(sh)
        return lg

    def save_meta(self, **extra: Any) -> None:
        self.meta.update(extra)
        tmp = self.dir / "session.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(self.dir / "session.json")

    def close(self, stats: dict | None = None) -> None:
        self.save_meta(stop_time_utc=utc_iso(), stop_monotonic_s=monotonic(), stats=stats or {})
        self.frames_csv.close()
        self.recordings_csv.close()
        for h in list(self.log.handlers):
            h.flush()
            h.close()
            self.log.removeHandler(h)
