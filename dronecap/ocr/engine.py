"""OCR 엔진 래퍼. 엔진이 주는 점수만 confidence 로 보고하고, 없으면 None 으로 둔다 (임의로 만들지 않음).

  rapidocr  : rapidocr_onnxruntime. CPU, pip 설치만으로 동작, 텍스트 조각마다 인식 점수(0~1) 제공.
  tesseract : pytesseract + 별도 설치한 tesseract.exe. image_to_data 의 conf(0~100) 를 0~1 로 나눠 보고.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class OcrResult:
    text: str                      # 영역 안에서 읽은 글자 전체 (조각은 공백으로 이어 붙임)
    confidence: Optional[float]    # 엔진 제공 점수. 조각이 여럿이면 최솟값(가장 약한 조각). 없으면 None
    pieces: list[tuple[str, Optional[float]]]
    engine: str


def preprocess(crop: np.ndarray, upscale: float = 3.0, invert: bool = False, threshold: bool = False) -> np.ndarray:
    img = crop
    if upscale and upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if invert:
        gray = 255 - gray
    if threshold:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # 여백을 둬야 글자가 가장자리에 붙었을 때 인식률이 덜 떨어진다
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=int(np.median(gray)))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class RapidOcrEngine:
    """detect=False(기본): ROI 전체를 '한 줄' 로 보고 인식 모델만 돌린다.
    숫자 하나만 들어 있는 좁은 ROI 에서는 검출 단계가 '12.3' 을 '1 2.3' 처럼 쪼개는 일이 있어 끄는 쪽이 정확했다 (합성 화면 시험).
    ROI 가 넓어 글자 덩어리가 여럿이면 detect=True 로."""
    name = "rapidocr"

    def __init__(self, detect: bool = False):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError("rapidocr_onnxruntime 이 없습니다: pip install rapidocr_onnxruntime") from e
        self._ocr = RapidOCR()
        self.detect = detect

    def read(self, img: np.ndarray) -> OcrResult:
        pieces: list[tuple[str, Optional[float]]] = []
        if self.detect:
            result, _elapse = self._ocr(img)
            if result:
                # 각 항목: [box(4점), text, score]. 왼쪽→오른쪽 순으로 정렬
                for box, text, score in sorted(result, key=lambda r: min(p[0] for p in r[0])):
                    pieces.append((str(text), float(score)))
        else:
            result, _elapse = self._ocr(img, use_det=False, use_cls=False)
            if result:
                for text, score in result:
                    pieces.append((str(text), float(score)))
        text = " ".join(t for t, _ in pieces)
        confs = [c for _, c in pieces if c is not None]
        return OcrResult(text, min(confs) if confs else None, pieces, self.name)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cmd: str = "tesseract"):
        try:
            import pytesseract
        except ImportError as e:
            raise RuntimeError("pytesseract 가 없습니다: pip install pytesseract (tesseract.exe 도 따로 설치)") from e
        self._pt = pytesseract
        if cmd and cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = cmd
        # psm 7: 한 줄 텍스트. 숫자·소수점·부호·단위 글자만 허용
        self._cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.-+mskhft/HD "

    def read(self, img: np.ndarray) -> OcrResult:
        d = self._pt.image_to_data(img, config=self._cfg, output_type=self._pt.Output.DICT)
        pieces: list[tuple[str, Optional[float]]] = []
        for t, c in zip(d["text"], d["conf"]):
            t = (t or "").strip()
            if not t:
                continue
            try:
                cf = float(c)
                conf = cf / 100.0 if cf >= 0 else None
            except (TypeError, ValueError):
                conf = None
            pieces.append((t, conf))
        text = " ".join(t for t, _ in pieces)
        confs = [c for _, c in pieces if c is not None]
        return OcrResult(text, min(confs) if confs else None, pieces, self.name)


def make_engine(cfg_ocr: dict):
    name = cfg_ocr.get("engine", "rapidocr")
    if name == "rapidocr":
        return RapidOcrEngine(detect=bool(cfg_ocr.get("rapidocr_detect", False)))
    if name == "tesseract":
        return TesseractEngine(cfg_ocr.get("tesseract_cmd", "tesseract"))
    raise ValueError(f"알 수 없는 OCR 엔진: {name}")
