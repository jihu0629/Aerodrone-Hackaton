"""수신 프로그램(20_capture) 안에서 화면 OCR 을 같이 돌리는 스레드.

별도 터미널에서 22_ocr_run.py --screen 을 돌리는 것과 같은 일을 하되,
  * 같은 세션 폴더의 telemetry.csv 에 쓰고
  * 최신 값을 HUD 에 보여 준다 (값 옆에 몇 초 전 값인지 표시 → 오래된 값을 현재 값으로 착각하지 않게).
OCR 은 미러링된 화면을 읽으므로 PC 수신 시각 기준이며, 영상과의 오프셋은 23_sync.py 에서 따로 보정한다.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..timeutil import monotonic
from .roi import RoiSet
from .runner import OcrRunner


class LiveOcr(threading.Thread):
    def __init__(self, cfg_ocr: dict, session_dir: Path, log, source_factory: Optional[Callable] = None,
                 engine=None):
        super().__init__(name="LiveOcr", daemon=True)
        self.cfg = cfg_ocr
        self.dir = Path(session_dir)
        self.log = log
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self.latest: dict[str, dict] = {}     # field → row (value, unit, status, confidence)
        self.latest_mono: Optional[float] = None
        self.count = 0
        self.error: Optional[str] = None
        self.enabled = False
        roi_path = Path(cfg_ocr.get("roi_file", "config/ocr_roi.json"))
        if not roi_path.exists():
            self.error = f"ROI 파일이 없어 OCR 을 끕니다: {roi_path} (scripts/21_select_roi.py 로 먼저 선택)"
            return
        try:
            self.rois = RoiSet.load(roi_path)
            if source_factory is None:
                from .sources import open_source
                source_factory = lambda: open_source("screen", None, cfg_ocr)  # noqa: E731
            self.source = source_factory()
            self.runner = OcrRunner(cfg_ocr, self.rois, self.dir / "telemetry.csv", self.dir / "ocr_debug", engine=engine)
            self.enabled = True
        except Exception as e:
            self.error = f"OCR 초기화 실패: {e!r}"

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        if not self.enabled:
            return
        interval = float(self.cfg.get("interval_s", 0.5))
        while not self._stop_evt.is_set():
            t0 = monotonic()
            try:
                item = self.source.next()
                if item is None:
                    self.log.info("OCR 입력이 끝났습니다")
                    break
                frame, info = item
                rows = self.runner.process(frame, info)
                with self._lock:
                    self.latest = {r["field"]: r for r in rows}
                    self.latest_mono = monotonic()
                    self.count += 1
            except Exception as e:
                self.error = f"OCR 오류: {e!r}"
                self.log.error(self.error)
                time.sleep(1.0)
            # ScreenSource 는 스스로 간격을 맞추지만, 다른 소스를 쓸 때를 위해 최소 간격 보장
            rest = interval - (monotonic() - t0)
            if rest > 0:
                self._stop_evt.wait(rest)
        try:
            self.runner.close()
        except Exception:
            pass

    def hud_text(self) -> str:
        if not self.enabled:
            return f"OCR: off ({self.error or 'disabled'})"
        with self._lock:
            if not self.latest:
                return "OCR: waiting..."
            age = monotonic() - (self.latest_mono or 0)
            parts = []
            for f, r in self.latest.items():
                v = r.get("value") or "-"
                parts.append(f"{f}={v}{r.get('unit', '')}[{r.get('status')}]")
            return f"OCR({self.count}, {age:.1f}s ago): " + "  ".join(parts)
