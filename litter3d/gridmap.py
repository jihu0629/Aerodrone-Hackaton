"""격자별 kg 히트맵 (Gonçalves et al. 2022 의 격자 표준화 제안, Andriolo 2024 의 2 m 격자 무게 지도).

물체 위치는 DSM 픽셀 좌표 (cx, cy) → GeoTIFF 변환행렬(affine) 로 지도 좌표(m) 로 바꾼다.
변환행렬이 없으면 픽셀 × GSD 로 로컬 좌표(m) 를 쓴다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .mass import ObjectMass


def object_xy_m(m: ObjectMass, gsd_m: float, transform=None) -> tuple[float, float]:
    if transform is not None:
        x, y = transform * (m.cx_px, m.cy_px)
        return float(x), float(y)
    return m.cx_px * gsd_m, m.cy_px * gsd_m


def grid_kg(masses: list[ObjectMass], gsd_m: float, cell_m: float = 10.0, transform=None,
            field: str = "kg_typ") -> tuple[np.ndarray, dict]:
    """반환: (격자 kg 2D 배열 [row, col], 메타 {x0, y0, cell_m, ncol, nrow, y_up})."""
    pts = [(object_xy_m(m, gsd_m, transform), getattr(m, field)) for m in masses if m.method != "excluded"]
    if not pts:
        return np.zeros((1, 1)), {"x0": 0, "y0": 0, "cell_m": cell_m, "ncol": 1, "nrow": 1}
    xs = np.array([p[0][0] for p in pts]); ys = np.array([p[0][1] for p in pts])
    x0, y0 = np.floor(xs.min() / cell_m) * cell_m, np.floor(ys.min() / cell_m) * cell_m
    ncol = int((xs.max() - x0) // cell_m) + 1
    nrow = int((ys.max() - y0) // cell_m) + 1
    g = np.zeros((nrow, ncol))
    for (x, y), kg in pts:
        g[int((y - y0) // cell_m), int((x - x0) // cell_m)] += kg
    return g, {"x0": float(x0), "y0": float(y0), "cell_m": cell_m, "ncol": ncol, "nrow": nrow}


def grid_to_csv(g: np.ndarray, meta: dict, path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "x_min_m", "y_min_m", "kg"])
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if g[r, c] > 0:
                    w.writerow([r, c, meta["x0"] + c * meta["cell_m"], meta["y0"] + r * meta["cell_m"],
                                round(float(g[r, c]), 3)])


def grid_to_png(g: np.ndarray, meta: dict, path: str | Path, scale: int = 40) -> None:
    """간단한 히트맵 PNG (matplotlib 없이 OpenCV 로)."""
    import cv2
    if g.max() <= 0:
        img = np.zeros((g.shape[0] * scale, g.shape[1] * scale, 3), np.uint8)
    else:
        norm = (g / g.max() * 255).astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        img[g == 0] = (40, 40, 40)
        img = cv2.resize(img, (g.shape[1] * scale, g.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] > 0:
                cv2.putText(img, f"{g[r, c]:.1f}", (c * scale + 2, r * scale + scale // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def grid_to_geotiff(g: np.ndarray, meta: dict, path: str | Path, crs=None) -> None:
    import rasterio
    from rasterio.transform import from_origin
    cell = meta["cell_m"]
    # 행이 y 증가 방향 → 북쪽이 위인 GeoTIFF 로 뒤집어서 저장
    tr = from_origin(meta["x0"], meta["y0"] + g.shape[0] * cell, cell, cell)
    with rasterio.open(path, "w", driver="GTiff", height=g.shape[0], width=g.shape[1], count=1,
                       dtype="float32", crs=crs, transform=tr) as dst:
        dst.write(g[::-1].astype(np.float32), 1)


def save_meta(meta: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
