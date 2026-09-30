"""검증용 합성 데이터: 경사진 모래 위에 크기를 아는 상자·원기둥을 놓은 DSM + 정사영상.

실제 촬영 전에 부피 코드의 오차를 확인하는 데 쓴다. 실물 검증(기준물 촬영) 을 대체하지는 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segment import MaskList


@dataclass
class SynthObject:
    cls: str
    shape: str          # "box" | "cylinder" | "dome"
    x_m: float
    y_m: float
    w_m: float
    l_m: float
    h_m: float

    @property
    def true_volume_m3(self) -> float:
        if self.shape == "box":
            return self.w_m * self.l_m * self.h_m
        if self.shape == "cylinder":
            return np.pi * (self.w_m / 2) * (self.l_m / 2) * self.h_m
        if self.shape == "dome":   # 반타원체
            return 2 / 3 * np.pi * (self.w_m / 2) * (self.l_m / 2) * self.h_m
        raise ValueError(self.shape)


def make_scene(objects: list[SynthObject], size_m: tuple[float, float] = (30.0, 20.0), gsd_m: float = 0.005,
               slope: tuple[float, float] = (0.03, 0.01), noise_m: float = 0.005, seed: int = 0,
               ripple_m: float = 0.0) -> tuple[np.ndarray, np.ndarray, MaskList]:
    """반환: (dsm, ortho_bgr, 정답 마스크 목록)."""
    rng = np.random.default_rng(seed)
    W, H = int(size_m[0] / gsd_m), int(size_m[1] / gsd_m)
    gy, gx = np.mgrid[0:H, 0:W]
    xm, ym = gx * gsd_m, gy * gsd_m
    dsm = slope[0] * xm + slope[1] * ym + 5.0
    if ripple_m > 0:
        dsm += ripple_m * np.sin(xm / 0.8 * 2 * np.pi) * np.cos(ym / 1.1 * 2 * np.pi)
    dsm += rng.normal(0, noise_m, dsm.shape)
    ortho = np.full((H, W, 3), (150, 190, 215), np.uint8)   # 모래색 (BGR)
    ortho = np.clip(ortho.astype(np.int16) + rng.integers(-12, 12, ortho.shape), 0, 255).astype(np.uint8)
    masks: MaskList = []
    for o in objects:
        m = np.zeros((H, W), bool)
        cx, cy = o.x_m / gsd_m, o.y_m / gsd_m
        rx, ry = o.w_m / 2 / gsd_m, o.l_m / 2 / gsd_m
        dx, dy = (gx - cx) / rx, (gy - cy) / ry
        if o.shape == "box":
            m = (np.abs(dx) <= 1) & (np.abs(dy) <= 1)
            hgt = np.where(m, o.h_m, 0.0)
        else:
            r2 = dx ** 2 + dy ** 2
            m = r2 <= 1
            hgt = np.where(m, o.h_m if o.shape == "cylinder" else o.h_m * np.sqrt(np.clip(1 - r2, 0, 1)), 0.0)
        dsm = dsm + hgt
        ortho[m] = (60, 60, 200) if "styro" not in o.cls else (240, 240, 240)
        masks.append((o.cls, m, 1.0))
    return dsm.astype(np.float32), ortho, masks


def default_objects() -> list[SynthObject]:
    return [
        SynthObject("styrofoam_box", "box", 5, 5, 0.6, 0.4, 0.3),        # 0.072 m³ → EPS 20 kg/m³ ≈ 1.4 kg
        SynthObject("styrofoam_buoy", "cylinder", 12, 8, 0.5, 0.5, 0.5),  # 0.098 m³
        SynthObject("pet_bottle", "cylinder", 20, 4, 0.07, 0.25, 0.07),   # 작은 병
        SynthObject("net", "dome", 25, 14, 1.5, 1.0, 0.4),               # 그물 뭉치 0.314 m³
        SynthObject("rope", "box", 8, 15, 0.8, 0.2, 0.1),
        SynthObject("other_plastic", "box", 16, 16, 0.3, 0.3, 0.2),
        SynthObject("styrofoam_fragment", "box", 3, 17, 0.04, 0.04, 0.02),  # 소형 → 개수 기반
        SynthObject("vegetation", "dome", 27, 3, 1.0, 0.8, 0.15),        # 제외
    ]
