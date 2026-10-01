"""ROI 마다 OCR 을 돌려 telemetry.csv 에 기록.

telemetry.csv (long form, 필드마다 1행)
  telemetry_receive_time_utc : PC 가 이 화면을 캡처한 시각 (화면 녹화 파일 입력이면 '처리한' 시각이라 의미가 다름 → source_time_s 참고)
  receive_monotonic_s, source, source_frame_index, source_time_s
  field, raw_text, confidence(엔진 제공, 없으면 빈칸), value, unit, status, notes, engine
값이 없거나 의심스러우면 value 는 비우거나 status 로 표시한다. 이전 값을 채워 넣지 않는다.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..timeutil import monotonic, utc_iso
from .engine import OcrResult, make_engine, preprocess
from .parse import parse_number
from .roi import RoiSet

TELEMETRY_COLUMNS = [
    "telemetry_receive_time_utc", "receive_monotonic_s", "source", "source_frame_index", "source_time_s",
    "field", "raw_text", "confidence", "value", "unit", "status", "notes", "engine",
]


class OcrRunner:
    def __init__(self, cfg_ocr: dict, rois: RoiSet, out_csv: str | Path, debug_dir: Optional[str | Path] = None,
                 engine=None):
        self.cfg = cfg_ocr
        self.rois = rois
        self.fields: dict = cfg_ocr.get("fields", {}) or {}
        self.engine = engine or make_engine(cfg_ocr)
        self.out = Path(out_csv)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        new = not self.out.exists()
        self._f = open(self.out, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=TELEMETRY_COLUMNS)
        if new:
            self._w.writeheader()
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_every = int(cfg_ocr.get("debug_every", 0) or 0)
        self.n = 0
        self.size_warned = False

    def process(self, frame: np.ndarray, info: dict) -> list[dict]:
        warn = self.rois.check_size(frame.shape)
        if warn:
            if not self.size_warned:
                print("[경고]", warn)
                self.size_warned = True
            # 크기가 다르면 좌표가 틀어졌을 가능성이 커서 결과를 size_mismatch 로 기록한다
        t_utc, t_mono = utc_iso(), monotonic()
        rows: list[dict] = []
        crops: list[tuple[str, np.ndarray, str]] = []
        for roi in self.rois.rois:
            spec = self.fields.get(roi.name, {})
            crop = roi.crop(frame)
            if crop.size == 0:
                res = OcrResult("", None, [], self.engine.name)
            else:
                pre = preprocess(crop, float(self.cfg.get("upscale", 3.0)), bool(self.cfg.get("invert", False)),
                                 bool(self.cfg.get("threshold", False)))
                res = self.engine.read(pre)
            rng = spec.get("range")
            parsed = parse_number(res.text, spec.get("unit"), tuple(rng) if rng else None, label=spec.get("label"))
            status = "size_mismatch" if warn else parsed.status
            row = {
                "telemetry_receive_time_utc": t_utc, "receive_monotonic_s": f"{t_mono:.3f}",
                "source": info.get("source"), "source_frame_index": info.get("source_frame_index"),
                "source_time_s": "" if info.get("source_time_s") is None else f"{info['source_time_s']:.3f}",
                "field": roi.name, "raw_text": res.text,
                "confidence": "" if res.confidence is None else f"{res.confidence:.3f}",
                "value": "" if parsed.value is None else f"{parsed.value:g}",
                "unit": parsed.unit or "", "status": status, "notes": ";".join(parsed.notes), "engine": res.engine,
            }
            rows.append(row)
            crops.append((roi.name, crop, f"{res.text!r} -> {row['value']} {row['unit']} [{status}]"))
            self._w.writerow(row)
        self._f.flush()
        self.n += 1
        if self.debug_dir and self.debug_every and self.n % self.debug_every == 1:
            self._save_debug(frame, crops, info)
        return rows

    def _save_debug(self, frame: np.ndarray, crops, info: dict) -> None:
        """잘라낸 영역과 인식 결과를 한 장에 모아 저장 — 사람이 보고 ROI·전처리를 조정하는 용도."""
        tiles = []
        for name, crop, label in crops:
            if crop.size == 0:
                crop = np.zeros((20, 60, 3), np.uint8)
            c = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            pad = np.full((c.shape[0] + 30, max(c.shape[1], 600), 3), 40, np.uint8)
            pad[30:30 + c.shape[0], :c.shape[1]] = c
            cv2.putText(pad, f"{name}: {label}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append(pad)
        if not tiles:
            return
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        montage = np.vstack(tiles)
        idx = info.get("source_frame_index", self.n)
        cv2.imwrite(str(self.debug_dir / f"ocr_{self.n:05d}_src{idx}.png"), montage)
        overview = frame.copy()
        for roi in self.rois.rois:
            cv2.rectangle(overview, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2)
        cv2.imwrite(str(self.debug_dir / f"ocr_{self.n:05d}_rois.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def close(self) -> None:
        self._f.close()
