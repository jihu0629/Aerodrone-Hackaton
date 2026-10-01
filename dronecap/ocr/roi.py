"""OCR 영역(ROI) 선택·저장·검증.

ROI 는 캡처 프레임의 픽셀 좌표로 저장한다. 미러링 창 크기·배율·모니터가 바뀌면 좌표가 틀어지므로
기준 프레임 크기(reference_size)를 함께 저장하고, 실행 시 크기가 다르면 재설정을 요구한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..timeutil import utc_iso


@dataclass
class Roi:
    name: str
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(W, self.x + self.w), min(H, self.y + self.h)
        return img[y0:y1, x0:x1]


@dataclass
class RoiSet:
    reference_size: tuple[int, int]      # (width, height) 선택 당시 프레임 크기
    source_desc: str
    rois: list[Roi]
    created_utc: str = ""

    def save(self, path: str | Path) -> None:
        d = {"reference_size": list(self.reference_size), "source_desc": self.source_desc,
             "created_utc": self.created_utc or utc_iso(), "rois": [asdict(r) for r in self.rois],
             "note": "좌표는 reference_size 기준 픽셀. 미러링 창 크기·배율이 바뀌면 21_select_roi.py 로 다시 선택"}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "RoiSet":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(tuple(d["reference_size"]), d.get("source_desc", ""),
                   [Roi(**r) for r in d["rois"]], d.get("created_utc", ""))

    def check_size(self, frame_shape) -> str | None:
        """프레임 크기가 기준과 다르면 안내 문구를 돌려준다 (None 이면 정상)."""
        h, w = frame_shape[:2]
        if (w, h) != tuple(self.reference_size):
            return (f"입력 크기 {w}x{h} 가 ROI 를 선택할 때의 {self.reference_size[0]}x{self.reference_size[1]} 와 다릅니다. "
                    f"미러링 창 크기·배율·모니터 설정이 바뀐 것 같습니다. scripts/21_select_roi.py 로 ROI 를 다시 선택하세요.")
        return None


def select_rois_interactive(frame: np.ndarray, field_names: list[str], window: str = "select ROI") -> list[Roi]:
    """필드마다 마우스로 사각형을 그린다. Enter/Space 로 확정, c 로 그 필드 건너뛰기."""
    rois: list[Roi] = []
    base = frame.copy()
    for name in field_names:
        shown = base.copy()
        for r in rois:
            cv2.rectangle(shown, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 200, 0), 2)
            cv2.putText(shown, r.name, (r.x, max(12, r.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        msg = f"[{name}] 영역을 드래그 → Enter/Space 확정, c = 이 항목 건너뛰기"
        cv2.putText(shown, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(shown, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        x, y, w, h = cv2.selectROI(window, shown, showCrosshair=True, fromCenter=False)
        if w == 0 or h == 0:
            print(f"  {name}: 건너뜀")
            continue
        rois.append(Roi(name, int(x), int(y), int(w), int(h)))
        print(f"  {name}: x={x} y={y} w={w} h={h}")
    cv2.destroyWindow(window)
    return rois
