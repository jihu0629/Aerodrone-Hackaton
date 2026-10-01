"""위성(Sentinel-2) 해안 형상 → 구간별 쓰레기 집적 점수 → 예산 안의 우선 구간 → 코리더 비행 경로.

위성 픽셀(10 m)로는 쓰레기가 안 보인다. 대신 해안선의 **형상**을 재서 "쌓이기 쉬운 구간" 을 고르고,
드론 정밀 촬영은 그 구간에만 쓴다. (문갑도 42개 라벨에서 만입도 상위 30 % 길이가 라벨 64 % 를 포착)

입력   Sentinel-2 B03(녹색)·B08(근적외) GeoTIFF (같은 격자). 장면이 여러 개면 픽셀별 NDWI 최댓값 → 구름 제거
       (선택) B04(적색) → NDVI 로 해빈 띠 폭
       (선택) 밀도 점 (x, y, 가중치) → 예산 대비 포착률 곡선으로 점수 검증
출력   해안선 10 m 표본점 + 특징 + 점수, 200 m 구간, 선택 구간, 소티별 웨이포인트 (GeoJSON · Litchi CSV · DJI WPML KMZ)

특징 (모두 형상 기반 → 다른 섬·계절로 옮겨도 계산 방법이 같다)
  bay       만입도 = 1 − (반경 R 안 물 비율).  높을수록 오목한 만
  exposure  노출  = cos(해안 바깥 법선 방위 − 바람이 불어오는 방위).  +1 이면 바람맞이
  strip     띠 폭 = 해안선에서 안쪽으로 물도 식생(NDVI ≥ veg_ndvi)도 아닌 픽셀이 이어지는 길이 (m)
  score     = w_bay·z(bay) + w_exp·z(exposure) + w_strip·z(log strip)      (가중치는 가정값, 기본 1 / 0.5 / 0.5)

경로 (가정값: 고도·촬영폭·속도·배터리)
  선택 구간마다 해안선을 안쪽으로 offset 한 코리더 선 → 웨이포인트 (Douglas-Peucker 로 수 줄임)
  구간 순서 = 출발지 기준 최근접 이웃(양방향 고려) + 2-opt,  배터리 시간을 넘으면 출발지로 복귀해 다음 소티
"""
from __future__ import annotations

import csv
import json
import math
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# ──────────────────────────────── 1. 래스터 입력 ────────────────────────────────

def read_band(path: str | Path):
    """GeoTIFF 1밴드 → (float 배열, affine transform, crs)."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform, src.crs


CLOUD_SCL = (0, 1, 3, 8, 9, 10, 11)   # Sentinel-2 SCL: nodata, 포화, 구름 그림자, 구름(중·고), 권운, 눈


def ndwi_max(greens: list[np.ndarray], nirs: list[np.ndarray], scls: list[np.ndarray] | None = None,
             all_cloud: str = "water") -> np.ndarray:
    """장면별 NDWI = (G - NIR)/(G + NIR) 를 합성. SCL(장면 분류)이 있으면 구름·그림자 픽셀은 빼고 맑은 장면끼리 최댓값.
    모든 장면이 구름인 픽셀은 all_cloud="water" 면 물(해안선 판단 불가 → 바다 위 구름이 섬으로 잡히는 것을 막음),
    "raw" 면 그냥 최댓값 (구름은 NDWI 를 낮추므로 최댓값이 맑은 쪽을 고르는 경향)."""
    raw, clean = None, None
    for i, (g, n) in enumerate(zip(greens, nirs)):
        v = (g - n) / (g + n + 1e-6)
        v[(g <= 0) & (n <= 0)] = -1.0          # nodata
        raw = v if raw is None else np.maximum(raw, v)
        if scls is not None:
            vv = np.where(np.isin(scls[i], CLOUD_SCL), -np.inf, v)
            clean = vv if clean is None else np.maximum(clean, vv)
    if clean is None:
        return raw
    return np.where(np.isfinite(clean), clean, raw if all_cloud == "raw" else 1.0)


def water_mask(ndwi: np.ndarray, thr: float = 0.0, min_land_km2: float = 0.5, px_m: float = 10.0):
    """NDWI > thr 를 물로. 육지는 큰 덩어리만 남긴다 (바다 위 구름·파도 거품 제거). 섬 안 호수는 육지로 채운다."""
    water = ndwi > thr
    land = ~water
    lab, n = ndi.label(land)
    if n == 0:
        raise ValueError("육지가 없음 (NDWI 임계값 확인)")
    sizes = ndi.sum(land, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s * px_m * px_m >= min_land_km2 * 1e6])
    land = ndi.binary_fill_holes(keep)       # 섬 안 호수·웅덩이 → 육지 (해안선은 바깥 둘레만)
    return land


# ──────────────────────────────── 2. 해안선 → 표본점 ────────────────────────────────

@dataclass
class Coast:
    xy: np.ndarray          # (N, 2) 해안선 표본점 (m, 래스터 CRS)
    tangent: np.ndarray     # (N, 2)
    normal: np.ndarray      # (N, 2) 바깥(바다) 방향 단위벡터
    s: np.ndarray           # (N,) 누적 길이 m
    ring_id: np.ndarray     # (N,) 어느 섬(폴리곤) 둘레인지
    step_m: float
    feats: dict = field(default_factory=dict)   # 이름 → (N,) 배열
    score: np.ndarray | None = None

    @property
    def n(self):
        return len(self.xy)


def coast_from_mask(land: np.ndarray, transform, step_m: float = 10.0, simplify_m: float = 15.0,
                    min_ring_m: float = 500.0) -> Coast:
    """육지 마스크 → 폴리곤 바깥 둘레 → step_m 간격 표본점 + 접선·바깥 법선."""
    from rasterio import features
    from shapely.geometry import shape
    from shapely.geometry.polygon import orient

    polys = []
    for geom, val in features.shapes(land.astype(np.uint8), mask=land, transform=transform, connectivity=8):
        if val == 1:
            p = shape(geom)
            if not p.is_valid:
                p = p.buffer(0)
            if p.geom_type == "MultiPolygon":
                polys.extend(p.geoms)
            else:
                polys.append(p)
    polys.sort(key=lambda p: p.area, reverse=True)
    xs, ts, ns, ss, rid = [], [], [], [], []
    for k, poly in enumerate(polys):
        ring = orient(poly, sign=1.0).exterior            # 반시계 → 육지가 진행 방향 왼쪽, 바다가 오른쪽
        if simplify_m > 0:
            ring = ring.simplify(simplify_m, preserve_topology=True)
        L = ring.length
        if L < min_ring_m:
            continue
        m = int(L // step_m)
        d = np.arange(m) * step_m
        pts = np.array([ring.interpolate(float(v)).coords[0] for v in d])
        nxt = np.array([ring.interpolate(float((v + step_m) % L)).coords[0] for v in d])
        prv = np.array([ring.interpolate(float((v - step_m) % L)).coords[0] for v in d])
        t = nxt - prv
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([t[:, 1], -t[:, 0]], 1)            # 오른손 법선 = 바다 쪽
        xs.append(pts); ts.append(t); ns.append(nrm); ss.append(d); rid.append(np.full(m, k))
    if not xs:
        raise ValueError("해안선이 없음")
    return Coast(np.vstack(xs), np.vstack(ts), np.vstack(ns), np.concatenate(ss), np.concatenate(rid), step_m)


def ring_table(coast: Coast) -> list[dict]:
    out = []
    for rid in np.unique(coast.ring_id):
        m = coast.ring_id == rid
        out.append({"ring": int(rid), "km": round(float(m.sum() * coast.step_m / 1000), 2),
                    "cx": round(float(coast.xy[m, 0].mean())), "cy": round(float(coast.xy[m, 1].mean()))})
    return out


def _rc(xy: np.ndarray, transform):
    """(x, y) m → (row, col) 정수 인덱스."""
    inv = ~transform
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    return np.floor(rows).astype(int), np.floor(cols).astype(int)


def _sample(arr: np.ndarray, xy: np.ndarray, transform, fill=np.nan):
    r, c = _rc(xy, transform)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full(len(xy), fill, dtype=float)
    out[ok] = arr[r[ok], c[ok]]
    return out


# ──────────────────────────────── 3. 특징 ────────────────────────────────

def add_features(coast: Coast, land: np.ndarray, transform, wind_from_deg: float = 60.0,
                 bay_radius_m: float = 150.0, ndvi: np.ndarray | None = None,
                 strip_max_m: float = 300.0, veg_ndvi: float = 0.30, px_m: float = 10.0) -> Coast:
    """만입도·노출·띠 폭을 표본점마다 계산. (바깥 법선 방향은 물 비율로 한 번 더 확인해 뒤집힌 곳을 고친다)"""
    water = ~land
    k = int(round(bay_radius_m / px_m))
    wfrac = ndi.uniform_filter(water.astype(np.float32), size=2 * k + 1, mode="constant")   # 정사각 창 근사
    # 법선 방향 검증: +30 m 쪽이 −30 m 쪽보다 물이 적으면 뒤집는다
    wp = _sample(water.astype(float), coast.xy + coast.normal * 3 * px_m, transform, 0.0)
    wm = _sample(water.astype(float), coast.xy - coast.normal * 3 * px_m, transform, 0.0)
    flip = wm > wp
    coast.normal[flip] *= -1
    bay = 1.0 - _sample(wfrac, coast.xy, transform, 0.5)
    bearing = (np.degrees(np.arctan2(coast.normal[:, 0], coast.normal[:, 1])) + 360) % 360   # 0=북, 90=동
    exposure = np.cos(np.radians(bearing - wind_from_deg))
    coast.feats.update({"bay": bay, "bearing": bearing, "exposure": exposure})
    if ndvi is not None:
        n_steps = int(strip_max_m // px_m)
        strip = np.zeros(coast.n)
        alive = np.ones(coast.n, bool)
        for i in range(1, n_steps + 1):
            p = coast.xy - coast.normal * (i * px_m)          # 안쪽으로
            w = _sample(water.astype(float), p, transform, 1.0) > 0.5
            v = _sample(ndvi, p, transform, 1.0) >= veg_ndvi
            alive &= ~(w | v)
            strip[alive] = i * px_m
        coast.feats["strip_m"] = strip
    return coast


def _z(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(v, nan=np.nanmean(v))
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def score_coast(coast: Coast, w_bay: float = 1.0, w_exp: float = 0.5, w_strip: float = 0.5,
                smooth_m: float = 100.0) -> Coast:
    """점수 = 가중 z 합. 해안선 방향으로 smooth_m 이동평균(표본점 단위)해서 들쭉날쭉한 픽셀 효과를 줄인다."""
    s = w_bay * _z(coast.feats["bay"]) + w_exp * _z(coast.feats["exposure"])
    if w_strip and "strip_m" in coast.feats:
        s = s + w_strip * _z(np.log1p(coast.feats["strip_m"]))
    k = max(int(smooth_m // coast.step_m), 1)
    out = np.empty_like(s)
    for rid in np.unique(coast.ring_id):
        m = coast.ring_id == rid
        out[m] = ndi.uniform_filter1d(s[m], size=k, mode="wrap")
    coast.score = out
    return coast


# ──────────────────────────────── 4. 구간·예산·검증 ────────────────────────────────

@dataclass
class Segment:
    seg_id: int
    ring_id: int
    i0: int                 # coast.xy 인덱스 범위 [i0, i1)
    i1: int
    length_m: float
    score: float
    bay: float
    exposure: float
    strip_m: float
    cx: float
    cy: float
    selected: bool = False
    order: int = -1
    sortie: int = -1


def make_segments(coast: Coast, seg_len_m: float = 200.0) -> list[Segment]:
    segs, k = [], max(int(seg_len_m // coast.step_m), 1)
    sid = 0
    for rid in np.unique(coast.ring_id):
        idx = np.nonzero(coast.ring_id == rid)[0]
        for a in range(0, len(idx), k):
            ii = idx[a:a + k]
            if len(ii) < max(k // 2, 1) and segs and segs[-1].ring_id == rid:
                # 자투리는 앞 구간에 붙임
                segs[-1].i1 = int(ii[-1] + 1)
                segs[-1].length_m = (segs[-1].i1 - segs[-1].i0) * coast.step_m
                continue
            f = coast.feats
            segs.append(Segment(sid, int(rid), int(ii[0]), int(ii[-1] + 1), len(ii) * coast.step_m,
                                float(coast.score[ii].mean()), float(f["bay"][ii].mean()),
                                float(f["exposure"][ii].mean()),
                                float(f["strip_m"][ii].mean()) if "strip_m" in f else float("nan"),
                                float(coast.xy[ii, 0].mean()), float(coast.xy[ii, 1].mean())))
            sid += 1
    return segs


def select_budget(segs: list[Segment], budget_frac: float | None = None, budget_km: float | None = None) -> list[Segment]:
    """점수 순으로 해안 길이 예산을 채운다."""
    total = sum(s.length_m for s in segs)
    budget = budget_km * 1000 if budget_km else total * (budget_frac if budget_frac is not None else 0.3)
    used = 0.0
    for s in sorted(segs, key=lambda s: -s.score):
        if used + s.length_m > budget:
            continue
        s.selected = True
        used += s.length_m
    return [s for s in segs if s.selected]


def nearest_coast_index(coast: Coast, xy: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(coast.xy).query(xy)[1]


def capture_curve(coast: Coast, score: np.ndarray, density_xy: np.ndarray, weights: np.ndarray | None = None,
                  xs: np.ndarray | None = None):
    """해안 표본점을 점수 순으로 정렬해 상위 x 길이 비율 안에 밀도 가중치의 몇 % 가 들어가는지."""
    near = nearest_coast_index(coast, density_xy)
    order = np.argsort(-score)
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    frac = rank[near] / len(order)
    w = np.ones(len(density_xy)) if weights is None else np.asarray(weights, float)
    xs = np.linspace(0, 1, 101) if xs is None else xs
    ys = np.array([w[frac <= x].sum() / w.sum() for x in xs])
    return xs, ys


# ──────────────────────────────── 5. 비행 경로 ────────────────────────────────

@dataclass
class FlightParams:
    altitude_m: float = 60.0        # 1차 전역 조사 고도 (Mini 5 Pro 약 1.05 cm/px, 촬영폭 86 m)
    swath_m: float = 86.0           # 고도에서의 가로 촬영폭 (가정: 화각 84°)
    inland_offset_frac: float = 0.3 # 코리더 중심을 해안선에서 안쪽으로 swath×frac 만큼 (파도선 ~ 배후까지)
    passes: int = 1                 # 띠가 넓으면 2 이상 (평행선 간격 = swath × (1 − side_overlap))
    side_overlap: float = 0.3
    speed_mps: float = 5.0          # 촬영 비행 속도
    transit_mps: float = 10.0       # 구간 사이 이동 속도
    battery_min: float = 25.0       # 소티당 쓸 수 있는 비행시간 (가정: Mini 5 Pro 실사용)
    wp_step_m: float = 50.0         # 웨이포인트 간격 (단순화 전)
    simplify_m: float = 4.0         # Douglas-Peucker 허용 오차
    gimbal_pitch: float = -90.0


@dataclass
class Sortie:
    sortie_id: int
    waypoints: list            # [(x, y, seg_id or -1)]
    segments: list[int]
    length_m: float
    time_min: float


def _corridor(coast: Coast, seg: Segment, fp: FlightParams, reverse: bool) -> list[np.ndarray]:
    """구간의 코리더 선들 (passes 개). 각 선은 (M, 2)."""
    from shapely.geometry import LineString
    ii = np.arange(seg.i0, seg.i1)
    lines = []
    for p in range(fp.passes):
        off = fp.swath_m * fp.inland_offset_frac + p * fp.swath_m * (1 - fp.side_overlap)
        pts = coast.xy[ii] - coast.normal[ii] * off
        ln = LineString(pts)
        if fp.simplify_m > 0:
            ln = ln.simplify(fp.simplify_m, preserve_topology=False)
        pts = np.array(ln.coords)
        # 왕복: 홀수 패스는 반대 방향
        if (p % 2 == 1) != reverse:
            pts = pts[::-1]
        lines.append(pts)
    return lines


def _seg_path(coast: Coast, seg: Segment, fp: FlightParams, reverse: bool) -> np.ndarray:
    return np.vstack(_corridor(coast, seg, fp, reverse))


def _plen(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def order_segments(coast: Coast, segs: list[Segment], depot: np.ndarray, fp: FlightParams):
    """최근접 이웃(양방향 끝점 고려) → 2-opt. 반환: [(seg, reverse)]"""
    ends = {s.seg_id: (_seg_path(coast, s, fp, False)[0], _seg_path(coast, s, fp, False)[-1]) for s in segs}
    left = {s.seg_id: s for s in segs}
    cur = depot.copy(); order = []
    while left:
        best = None
        for sid, s in left.items():
            a, b = ends[sid]
            for rev, start in ((False, a), (True, b)):
                d = np.linalg.norm(start - cur)
                if best is None or d < best[0]:
                    best = (d, s, rev)
        _, s, rev = best
        order.append((s, rev)); cur = ends[s.seg_id][0 if rev else 1]
        del left[s.seg_id]

    def tour_len(seq):
        p, tot = depot, 0.0
        for s, rev in seq:
            a, b = ends[s.seg_id]
            start, end = (b, a) if rev else (a, b)
            tot += np.linalg.norm(start - p); p = end
        return tot + np.linalg.norm(depot - p)

    improved, n = True, len(order)
    while improved and n > 2:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + [(s, not r) for s, r in reversed(order[i:j + 1])] + order[j + 1:]
                if tour_len(cand) + 1e-6 < tour_len(order):
                    order, improved = cand, True
    return order


def plan_sorties(coast: Coast, segs: list[Segment], depot_xy: np.ndarray, fp: FlightParams,
                 launch: str = "mobile", max_launch_transit_m: float = 2000.0):
    """순서대로 구간을 돌되 배터리 시간을 넘기면 복귀하고 새 소티.

    launch="fixed"  : 모든 소티가 depot 에서 이착륙. 한 소티 안에 못 들어가는(왕복만으로 배터리 초과) 구간은 건너뛰고 unreachable 에 기록.
    launch="mobile" : 조종자가 차·배로 이동하며 소티마다 첫 구간 근처에서 이륙 (이륙점 = 첫 구간 시작점에서 안쪽 50 m).
                      섬 둘레가 배터리 왕복 거리보다 길 때 현실적인 운용. 반환 (sorties, unreachable_seg_ids)
    """
    order = order_segments(coast, segs, depot_xy, fp)
    sorties, unreachable = [], []
    cur_wps, cur_segs, cur_len, cur_t = [], [], 0.0, 0.0
    base = depot_xy.copy(); pos = base.copy()

    def close():
        nonlocal cur_wps, cur_segs, cur_len, cur_t
        back = float(np.linalg.norm(base - pos))
        sorties.append(Sortie(len(sorties), cur_wps + [(float(base[0]), float(base[1]), -1)], cur_segs,
                              cur_len + back, cur_t + back / fp.transit_mps / 60))
        cur_wps, cur_segs, cur_len, cur_t = [], [], 0.0, 0.0

    for k, (s, rev) in enumerate(order):
        path = _seg_path(coast, s, fp, rev)
        t_fly = _plen(path) / fp.speed_mps / 60
        if launch == "mobile" and not cur_segs:
            # 새 소티: 첫 구간 시작점 근처(안쪽 50 m)를 이륙점으로
            i = s.i1 - 1 if rev else s.i0
            base = coast.xy[i] - coast.normal[i] * 50.0
            pos = base.copy()
        t_in = float(np.linalg.norm(path[0] - pos)) / fp.transit_mps / 60
        t_back = float(np.linalg.norm(base - path[-1])) / fp.transit_mps / 60
        if cur_segs and cur_t + t_in + t_fly + t_back > fp.battery_min:
            close()
            if launch == "mobile":
                i = s.i1 - 1 if rev else s.i0
                base = coast.xy[i] - coast.normal[i] * 50.0
            pos = base.copy()
            t_in = float(np.linalg.norm(path[0] - pos)) / fp.transit_mps / 60
            t_back = float(np.linalg.norm(base - path[-1])) / fp.transit_mps / 60
        if t_in + t_fly + t_back > fp.battery_min:
            unreachable.append(s.seg_id); s.order, s.sortie = -1, -1
            continue
        if not cur_wps:
            cur_wps.append((float(base[0]), float(base[1]), -1))
        cur_wps += [(float(x), float(y), s.seg_id) for x, y in path]
        cur_len += float(np.linalg.norm(path[0] - pos)) + _plen(path)
        cur_t += t_in + t_fly
        s.order, s.sortie = k, len(sorties)
        cur_segs.append(s.seg_id); pos = path[-1]
    if cur_segs:
        close()
    return sorties, unreachable


# ──────────────────────────────── 6. 내보내기 ────────────────────────────────

def to_lonlat(xy: np.ndarray, crs) -> np.ndarray:
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xy[:, 0], xy[:, 1])
    return np.stack([lon, lat], 1)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def export_geojson(path: Path, coast: Coast, segs: list[Segment], sorties: list[Sortie], crs, depot_xy):
    feats = []
    for s in segs:
        ll = to_lonlat(coast.xy[s.i0:s.i1], crs)
        props = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in asdict(s).items()}
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": ll.round(6).tolist()}, "properties": props})
    for so in sorties:
        ll = to_lonlat(np.array([[x, y] for x, y, _ in so.waypoints]), crs)
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": ll.round(6).tolist()},
                      "properties": {"kind": "sortie", "sortie_id": so.sortie_id, "segments": so.segments,
                                     "length_m": round(so.length_m), "time_min": round(so.time_min, 1)}})
    d = to_lonlat(np.array([depot_xy]), crs)[0]
    feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(d[0], 6), round(d[1], 6)]}, "properties": {"kind": "depot"}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, default=_jsonable), encoding="utf-8")


def export_litchi_csv(path: Path, sortie: Sortie, crs, fp: FlightParams):
    """Litchi Mission Hub CSV (웨이포인트 99개 제한은 Litchi 쪽 규칙)."""
    ll = to_lonlat(np.array([[x, y] for x, y, _ in sortie.waypoints]), crs)
    cols = ["latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)", "rotationdir", "gimbalmode",
            "gimbalpitchangle", "actiontype1", "actionparam1", "altitudemode", "speed(m/s)", "poi_latitude",
            "poi_longitude", "poi_altitude(m)", "poi_altitudemode", "photo_timeinterval", "photo_distinterval"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i, (lon, lat) in enumerate(ll):
            w.writerow([f"{lat:.7f}", f"{lon:.7f}", fp.altitude_m, 0, 0, 0, 2, fp.gimbal_pitch, -1, 0, 0,
                        fp.speed_mps, 0, 0, 0, 0, -1, 20])


def export_wpml_kmz(path: Path, sortie: Sortie, crs, fp: FlightParams, drone_enum: int = 68):
    """DJI WPML (wpmz/template.kml + wpmz/waylines.wpml) 최소 구성. 실기체에서 임포트 확인 필요."""
    ll = to_lonlat(np.array([[x, y] for x, y, _ in sortie.waypoints]), crs)
    now = datetime.now(timezone.utc)
    ts = int(now.timestamp() * 1000)
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:wpml="http://www.dji.com/wpmz/1.0.6"><Document>')
    cfg = (f"<wpml:author>ShoreSweep</wpml:author><wpml:createTime>{ts}</wpml:createTime><wpml:updateTime>{ts}</wpml:updateTime>"
           "<wpml:missionConfig><wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode><wpml:finishAction>goHome</wpml:finishAction>"
           "<wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost><wpml:executeRCLostAction>goBack</wpml:executeRCLostAction>"
           f"<wpml:takeOffSecurityHeight>20</wpml:takeOffSecurityHeight><wpml:globalTransitionalSpeed>{fp.transit_mps}</wpml:globalTransitionalSpeed>"
           f"<wpml:droneInfo><wpml:droneEnumValue>{drone_enum}</wpml:droneEnumValue><wpml:droneSubEnumValue>0</wpml:droneSubEnumValue></wpml:droneInfo>"
           "</wpml:missionConfig>")

    def placemarks(execute: bool):
        out = []
        for i, (lon, lat) in enumerate(ll):
            h = (f"<wpml:executeHeight>{fp.altitude_m}</wpml:executeHeight><wpml:waypointSpeed>{fp.speed_mps}</wpml:waypointSpeed>"
                 if execute else
                 f"<wpml:ellipsoidHeight>{fp.altitude_m}</wpml:ellipsoidHeight><wpml:height>{fp.altitude_m}</wpml:height>"
                 "<wpml:useGlobalHeight>1</wpml:useGlobalHeight><wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>")
            out.append(f"<Placemark><Point><coordinates>{lon:.7f},{lat:.7f}</coordinates></Point><wpml:index>{i}</wpml:index>{h}"
                       "<wpml:waypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:waypointHeadingParam>"
                       "<wpml:waypointTurnParam><wpml:waypointTurnMode>toPointAndPassWithContinuityCurvature</wpml:waypointTurnMode>"
                       "<wpml:waypointTurnDampingDist>5</wpml:waypointTurnDampingDist></wpml:waypointTurnParam>"
                       "<wpml:useStraightLine>0</wpml:useStraightLine>"
                       + ("<wpml:waypointGimbalHeadingParam><wpml:waypointGimbalPitchAngle>"
                          f"{fp.gimbal_pitch}</wpml:waypointGimbalPitchAngle></wpml:waypointGimbalHeadingParam>" if i == 0 else "")
                       + "</Placemark>")
        return "".join(out)

    template = (head + cfg + "<Folder><wpml:templateType>waypoint</wpml:templateType><wpml:templateId>0</wpml:templateId>"
                "<wpml:waylineCoordinateSysParam><wpml:coordinateMode>WGS84</wpml:coordinateMode>"
                "<wpml:heightMode>relativeToStartPoint</wpml:heightMode></wpml:waylineCoordinateSysParam>"
                f"<wpml:autoFlightSpeed>{fp.speed_mps}</wpml:autoFlightSpeed><wpml:globalHeight>{fp.altitude_m}</wpml:globalHeight>"
                "<wpml:caliFlightEnable>0</wpml:caliFlightEnable><wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>"
                "<wpml:globalWaypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:globalWaypointHeadingParam>"
                "<wpml:globalWaypointTurnMode>toPointAndPassWithContinuityCurvature</wpml:globalWaypointTurnMode>"
                "<wpml:globalUseStraightLine>0</wpml:globalUseStraightLine>" + placemarks(False) + "</Folder></Document></kml>")
    waylines = (head + cfg + "<Folder><wpml:templateId>0</wpml:templateId><wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>"
                f"<wpml:waylineId>0</wpml:waylineId><wpml:autoFlightSpeed>{fp.speed_mps}</wpml:autoFlightSpeed>"
                + placemarks(True) + "</Folder></Document></kml>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template)
        z.writestr("wpmz/waylines.wpml", waylines)


def summary(coast: Coast, segs: list[Segment], sorties: list[Sortie], fp: FlightParams, unreachable=None, launch="mobile") -> dict:
    sel = [s for s in segs if s.selected]
    total = sum(s.length_m for s in segs)
    return {
        "coast_km": round(total / 1000, 2), "n_segments": len(segs), "n_selected": len(sel),
        "selected_km": round(sum(s.length_m for s in sel) / 1000, 2),
        "selected_frac": round(sum(s.length_m for s in sel) / total, 3) if total else 0,
        "n_sorties": len(sorties), "launch": launch,
        "unreachable_segments": list(unreachable or []),
        "flight_km_total": round(float(sum(so.length_m for so in sorties)) / 1000, 2),
        "flight_min_total": round(float(sum(so.time_min for so in sorties)), 1),
        "sorties": [{"id": so.sortie_id, "segments": [int(x) for x in so.segments], "km": round(float(so.length_m) / 1000, 2),
                     "min": round(float(so.time_min), 1), "waypoints": len(so.waypoints),
                     "launch_lonlat": None} for so in sorties],
        "params": asdict(fp),
        "_note": "고도·촬영폭·속도·배터리·코리더 offset·점수 가중치는 가정값. 만입도 창은 정사각 근사.",
    }
