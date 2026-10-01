"""지형 비용 격자와 최단 경로 (정사영상 축소본 색으로 물·숲·맨땅을 나눠 걷는 비용을 매긴다).

분류 (가정값, RGB/HSV 규칙)
  0 outside   정사영상 밖 (이미지 테두리에 닿은 검은 영역)          → 통행 불가
  1 water     청록·파랑 (b > r, g > r)                               → 도보 불가, 보트 모드에선 빠름
  2 veg       초록·어두운 숲 + 섬 안쪽의 촬영 안 된 영역              → 걷기 느림 (×3)
  3 bare      모래·바위·길·건물 (밝거나 회색)                         → 걷기 기준 (×1)
해안선 정리: 테두리에 닿은 물만 바다, 본섬과 떨어진 작은 조각(포말·암초)은 바다, 섬 안 웅덩이는 맨땅.
격자 셀(기본 10 m) 은 맨땅 비율 15 % 이상이면 맨땅, 아니면 물/숲 중 많은 쪽 (해안 띠는 남기고 바다 쪽 오분류는 제거).

비용 = 셀 길이(m) × 평균 통행 배수. 물↔땅 전환에는 승·하선 비용(가정값 100 m 상당)을 더한다.
결과 거리는 '걷기 기준 환산 m' 이고, 시간 = 환산 m ÷ 걷기 속도.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

import cv2
import numpy as np

OUTSIDE, WATER, VEG, BARE = 0, 1, 2, 3
CLASS_KO = {OUTSIDE: "영상 밖", WATER: "물", VEG: "숲·수풀", BARE: "모래·바위·길"}
# 통행 배수 (가정값). None = 통행 불가
COST = {
    "walk": {OUTSIDE: None, WATER: None, VEG: 3.0, BARE: 1.0},
    "boat": {OUTSIDE: None, WATER: 0.5, VEG: 3.0, BARE: 1.0},
}
TRANSITION_M = 100.0     # 가정값: 물↔땅 전환(승·하선) 비용, 걷기 환산 m
SQRT2 = math.sqrt(2)


def classify_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """정사영상 축소본 BGR → 클래스 지도 (0..3)."""
    b, g, r = [img_bgr[:, :, i].astype(int) for i in range(3)]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    nodata = (r + g + b) < 24
    water = (~nodata) & (b > r + 8) & (g > r + 4) & (s > 25)
    veg = (~nodata) & (~water) & (g > r + 6) & (g >= b)
    dark = (~nodata) & (~water) & (~veg) & (v < 70)
    cls = np.full(img_bgr.shape[:2], BARE, np.uint8)
    cls[water] = WATER
    cls[veg | dark] = VEG
    # nodata: 테두리에 닿은 것은 영상 밖, 섬 안쪽에 갇힌 것은 숲으로
    nd = nodata.astype(np.uint8)
    n, lab = cv2.connectedComponents(nd, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    outside = np.isin(lab, list(border - {0})) & nodata
    cls[nodata] = VEG
    cls[outside] = OUTSIDE
    # 바다 위 하얀 포말(맨땅으로 잡힘) 정리: 주변이 거의 물이면 물로
    bare = (cls == BARE).astype(np.uint8)
    water_n = cv2.blur((cls == WATER).astype(np.float32), (9, 9))
    veg_n = cv2.blur((cls == VEG).astype(np.float32), (9, 9))
    cls[(bare == 1) & (water_n > 0.55) & (veg_n < 0.03)] = WATER
    return refine_coast(cls)


def refine_coast(cls: np.ndarray, min_land_px: int = 3000) -> np.ndarray:
    """해안선 정리.
    ① 바다 = 영상 테두리(또는 영상 밖)에 닿은 물 덩어리. 섬 안쪽에 갇힌 '물'(웅덩이·그늘진 바위)은 맨땅으로.
    ② 본섬(큰 땅 덩어리)과 떨어진 작은 땅 조각(포말·암초·얕은 물 바닥)은 바다로.
    ③ 바다에 바로 붙은 맨땅 1 px 띠는 그대로 둔다 (해안 걷기 경로)."""
    cls = cls.copy()
    H, W = cls.shape
    water = (cls == WATER).astype(np.uint8)
    outside = (cls == OUTSIDE)
    # ① 영상 밖과 물을 합쳐 테두리에 닿는 덩어리 = 바다
    sea_cand = ((water == 1) | outside).astype(np.uint8)
    n, lab = cv2.connectedComponents(sea_cand, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sea = np.isin(lab, list(border - {0})) & (water == 1)
    cls[(water == 1) & ~sea] = BARE                       # 내륙 물 → 맨땅(젖은 모래·바위)
    # ② 땅 덩어리 중 작은 것 → 바다
    land = ((cls == VEG) | (cls == BARE)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(land, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = {i + 1 for i, a in enumerate(areas) if a >= min_land_px}
        if not keep:
            keep = {int(np.argmax(areas)) + 1}
        small = (lab > 0) & ~np.isin(lab, list(keep))
        cls[small] = WATER
    return cls


class Terrain:
    """격자 + 8방향 최단 경로. nodes 는 (x, y) m 목록 (crs_m), 인덱스 0 이 출발지라는 가정은 없다."""

    BARE_MIN_FRAC = 0.15   # 블록에서 맨땅이 이 비율 이상이면 맨땅 (가정값)

    def __init__(self, cls_img: np.ndarray, m_per_px: float, x0: float, y0: float, cell_m: float = 10.0):
        """cls_img: 분류 지도 (행=남쪽으로 증가), (x0, y0) 는 이미지 왼쪽 위 모서리 좌표(m), y 는 북쪽이 큼."""
        k = max(int(round(cell_m / m_per_px)), 1)
        self.cell = k * m_per_px
        H, W = cls_img.shape
        nr, nc = H // k, W // k
        sub = cls_img[:nr * k, :nc * k].reshape(nr, k, nc, k)
        data_frac = (sub != OUTSIDE).mean(axis=(1, 3))
        bare_frac = (sub == BARE).mean(axis=(1, 3))
        water_frac = (sub == WATER).mean(axis=(1, 3))
        veg_frac = (sub == VEG).mean(axis=(1, 3))
        # 블록 분류 (비율 기준): 맨땅 15 % 이상이면 맨땅 (좁은 해안 띠 보존), 아니면 물/숲 중 많은 쪽
        grid = np.full((nr, nc), OUTSIDE, np.uint8)
        has = data_frac >= 0.2
        grid[has & (water_frac >= veg_frac)] = WATER
        grid[has & (veg_frac > water_frac)] = VEG
        grid[has & (bare_frac >= self.BARE_MIN_FRAC)] = BARE
        self.grid = grid
        self.nr, self.nc = nr, nc
        self.x0, self.y0 = x0, y0          # 격자 왼쪽 위 모서리
        self._graph: dict[str, object] = {}
        self._runs: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_basemap(cls, basemap, cell_m: float = 10.0) -> "Terrain":
        w = basemap.world
        m_per_px = w["px"] / basemap.sx
        return cls(classify_rgb(basemap.img), m_per_px, w["x0"], w["y0"], cell_m)

    # ── 좌표 ──
    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.x0) // self.cell); r = int((self.y0 - y) // self.cell)
        return min(max(r, 0), self.nr - 1), min(max(c, 0), self.nc - 1)

    def xy_of(self, r: int, c: int) -> tuple[float, float]:
        return self.x0 + (c + 0.5) * self.cell, self.y0 - (r + 0.5) * self.cell

    def force_passable(self, nodes_xy: list[tuple[float, float]]) -> None:
        """물체·출발지가 물/영상 밖 셀에 떨어지면 그 셀을 맨땅으로 (젖은 모래 등)."""
        for x, y in nodes_xy:
            r, c = self.cell_of(x, y)
            if self.grid[r, c] in (OUTSIDE, WATER):
                self.grid[r, c] = BARE
        self._graph.clear(); self._runs.clear()

    # ── 그래프 ──
    def _edges(self, mode: str):
        if mode in self._graph:
            return self._graph[mode]
        from scipy.sparse import coo_matrix
        cost = COST[mode]
        mult = np.full(self.grid.shape, np.nan)
        for k, v in cost.items():
            if v is not None:
                mult[self.grid == k] = v
        water = self.grid == WATER
        N = self.nr * self.nc
        idx = np.arange(N).reshape(self.nr, self.nc)
        rows, cols, vals = [], [], []
        for dr, dc, dl in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
            r0, r1 = (0, self.nr - dr)
            c0, c1 = (max(0, -dc), self.nc - max(0, dc))
            a = idx[r0:r1, c0:c1]; b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ma = mult[r0:r1, c0:c1]; mb = mult[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            w = self.cell * dl * (ma + mb) / 2
            trans = water[r0:r1, c0:c1] != water[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            w = w + np.where(trans, TRANSITION_M, 0.0)
            ok = np.isfinite(w)
            rows.append(a[ok]); cols.append(b[ok]); vals.append(w[ok])
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        g = coo_matrix((np.concatenate([vals, vals]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                       shape=(N, N)).tocsr()
        self._graph[mode] = g
        return g

    def run(self, mode: str, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        key = (mode, r, c)
        if key not in self._runs:
            from scipy.sparse.csgraph import dijkstra
            d, p = dijkstra(self._edges(mode), directed=False, indices=r * self.nc + c, return_predecessors=True)
            self._runs[key] = (d, p)
        return self._runs[key]

    def matrix(self, nodes_xy: list[tuple[float, float]], mode: str = "walk") -> np.ndarray:
        """노드 사이 걷기 환산 거리 행렬 (도달 불가 = inf)."""
        cells = [self.cell_of(x, y) for x, y in nodes_xy]
        n = len(cells)
        D = np.zeros((n, n))
        for i, (r, c) in enumerate(cells):
            d, _ = self.run(mode, r, c)
            for j, (r2, c2) in enumerate(cells):
                D[i, j] = d[r2 * self.nc + c2]
        np.fill_diagonal(D, 0.0)
        return D

    def path(self, a_xy: tuple[float, float], b_xy: tuple[float, float], mode: str = "walk") -> list[tuple[float, float]]:
        """a → b 최단 경로 좌표 (m). 도달 불가면 직선."""
        ra, ca = self.cell_of(*a_xy); rb, cb = self.cell_of(*b_xy)
        d, p = self.run(mode, ra, ca)
        j = rb * self.nc + cb
        if not np.isfinite(d[j]):
            return [a_xy, b_xy]
        cells = []
        while j >= 0 and j != ra * self.nc + ca:
            cells.append(j); j = p[j]
        cells.append(ra * self.nc + ca)
        pts = [self.xy_of(k // self.nc, k % self.nc) for k in reversed(cells)]
        return [a_xy] + pts[1:-1] + [b_xy]

    # ── 내보내기 ──
    def to_js(self) -> dict:
        return {"b64": base64.b64encode(self.grid.tobytes()).decode(), "nr": self.nr, "nc": self.nc,
                "x0": self.x0, "y0": self.y0, "cell": self.cell,
                "cost": {m: {str(k): v for k, v in cst.items()} for m, cst in COST.items()},
                "transition_m": TRANSITION_M, "class_ko": {str(k): v for k, v in CLASS_KO.items()}}

    def class_image(self, scale: int = 2) -> np.ndarray:
        pal = np.array([[0, 0, 0], [200, 120, 20], [40, 140, 40], [230, 230, 230]], np.uint8)
        img = pal[self.grid]
        return cv2.resize(img, (self.nc * scale, self.nr * scale), interpolation=cv2.INTER_NEAREST)

    def summary(self) -> dict:
        return {CLASS_KO[k]: int((self.grid == k).sum()) for k in (OUTSIDE, WATER, VEG, BARE)}


def save_class_png(t: Terrain, path: str | Path) -> None:
    ok, buf = cv2.imencode(".png", t.class_image(3))      # 한글 경로도 되도록 imencode + tofile
    if ok:
        buf.tofile(str(path))
