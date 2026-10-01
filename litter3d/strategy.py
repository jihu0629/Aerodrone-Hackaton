"""핫스팟 코리더 비행 vs 전체 지그재그 커버리지 — 같은 해안·같은 카메라·같은 띠 폭에서 정량 비교.

질문: 위성 형상 점수로 고른 "핫스팟 구간만" 나는 경로가, 해안 인접 영역 전체를 지그재그로 훑는 기존 방식보다
      비행거리·시간·소티·프레임(처리량)에서 얼마나 유리하고, 그 대가로 무엇을 놓치는가.

같게 두는 것 (전략 말고는 다 같다)
  해안선      Sentinel-2 NDWI 로 뽑은 10 m 표본점 (priority.coast_from_mask)
  카메라/고도 sim_ortho.py 의 Camera (1024×768, HFOV 73.7°) · 고도 alt → 발자국·GSD
  띠 폭       해안선 기준 바다쪽 sea_m ~ 안쪽 inland_m. 탐지 격자의 해안 거리 분포로 고른다
  패스 수     띠 폭을 횡방향 발자국 × (1 − 측면 겹침) 간격의 평행선으로 덮는 수 (sim_ortho Planner.coverage 와 같은 식)
  속도/배터리 조사 5 m/s, 이동 10 m/s, 소티 25 분, 이동식 이륙(소티마다 첫 구간 근처에서)
  검증 밀도   탐지 kg 격자 (칩이 있는 해안만 → "비율" 로만 해석)

전략
  zigzag_all   전체 커버리지: 모든 둘레를 연속으로 (기존 방식). 비행 = 둘레 × 패스 수 + 선회
  hotspot      점수 상위 X % 길이의 구간만 코리더(같은 패스 수)로, 구간 사이는 이동 속도로 건너뜀
  geo          같은 X % 길이를 출발지에서 해안 따라 연속으로 (위성 정보 없이 예산만 줄인 경우)
  random       같은 X % 길이를 무작위 구간으로 (여러 번 뽑아 평균)

출력 지표   조사 km · 이동 km · 총 km · 비행시간 h · 소티 수 · 이륙점 사이 지상 이동 km · 프레임 수(dt 간격) ·
            탐지 CPU 시간 · 데이터량 · 포착 kg 비율 · kg/비행시간 · 설계상 미포착 · 띠 폭 밖 kg · 점수 오류 민감도
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from . import priority as P


# ──────────────────────────────── 카메라 · 띠 · 운용 ────────────────────────────────

@dataclass
class Camera:
    """sim_ortho.Camera 와 같은 정의. 기수가 진행 방향이므로 횡방향 발자국 = 세로(height) 쪽."""
    width: int = 1024
    height: int = 768
    hfov_deg: float = 73.7

    def gsd(self, alt_m: float) -> float:
        return 2 * alt_m * math.tan(math.radians(self.hfov_deg) / 2) / self.width

    def footprint(self, alt_m: float):
        g = self.gsd(alt_m)
        return self.width * g, self.height * g          # (진행 방향, 횡방향)


@dataclass
class Ops:
    alt_m: float = 20.0           # 기존 시뮬(sim_ortho) 기본 고도
    overlap_side: float = 0.3
    sea_m: float = 20.0           # 띠: 해안선에서 바다쪽
    inland_m: float = 100.0       # 띠: 해안선에서 안쪽
    speed_mps: float = 5.0
    transit_mps: float = 10.0
    battery_min: float = 25.0
    launch_inland_m: float = 50.0
    dt_s: float = 1.0             # 프레임 간격 (sim_ortho dt)
    detect_s_per_frame: float = 0.22   # sim_ortho CPU: 30.9 s / 141 프레임
    mb_per_frame: float = 0.4
    simplify_m: float = 4.0
    cam: Camera = field(default_factory=Camera)

    def passes(self):
        """띠를 덮는 평행선 수와 각 선의 안쪽 오프셋(m, 양수 = 안쪽)."""
        _, across = self.cam.footprint(self.alt_m)
        spacing = max(across * (1 - self.overlap_side), 1.0)
        W = self.sea_m + self.inland_m
        n = max(1, int(math.ceil((W - across) / spacing)) + 1)
        offs = [-self.sea_m + across / 2 + k * spacing for k in range(n)]
        return n, spacing, across, offs

    def describe(self):
        n, spacing, across, offs = self.passes()
        along, _ = self.cam.footprint(self.alt_m)
        return {"alt_m": self.alt_m, "gsd_cm": round(self.cam.gsd(self.alt_m) * 100, 2), "footprint_m": [round(along, 1), round(across, 1)],
                "strip_m": [self.sea_m, self.inland_m], "passes": n, "spacing_m": round(spacing, 1),
                "speed_mps": self.speed_mps, "transit_mps": self.transit_mps, "battery_min": self.battery_min, "dt_s": self.dt_s}


# ──────────────────────────────── 경로 기하 ────────────────────────────────

def _plen(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def offset_line(xy: np.ndarray, normal: np.ndarray, off_m: float, simplify_m: float = 4.0) -> np.ndarray:
    """해안 표본점 폴리라인을 off_m 만큼 안쪽(양수)/바다쪽(음수)으로 평행 이동한 선.

    표본점마다 법선을 곱해 미는 방식은 10 m 격자 해안선의 들쭉날쭉한 법선 때문에 멀리 밀수록 지그재그로 길이가 부푼다
    (니하우 100 m 안쪽: 둘레 90 km → 201 km). shapely offset_curve(둥근 모서리)는 겹침·고리를 정리해 실제 평행선 길이를 준다 (→ 80 km)."""
    from shapely.geometry import LineString
    if len(xy) < 2:
        return xy.copy()
    if abs(off_m) < 1e-6:
        ln = LineString(xy)
    else:
        t = np.diff(xy, axis=0); t = np.vstack([t, t[-1:]])
        z = np.median(t[:, 0] * normal[:, 1] - t[:, 1] * normal[:, 0])     # 바다쪽 법선이 진행 방향 오른쪽이면 음수
        inland_sign = 1.0 if z < 0 else -1.0                                # offset_curve: 양수 = 진행 방향 왼쪽
        geom = LineString(xy).offset_curve(inland_sign * off_m, join_style="round")
        if geom.is_empty:
            return xy - normal * off_m
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
            # 진행 순서대로 이어 붙임 (시작점이 원래 선의 시작에 가까운 순)
            parts.sort(key=lambda g: float(np.linalg.norm(np.array(g.coords[0]) - xy[0])))
            pts = np.vstack([np.array(g.coords) for g in parts])
            ln = LineString(pts)
        else:
            ln = geom
    if simplify_m > 0 and len(ln.coords) > 2:
        ln = ln.simplify(simplify_m, preserve_topology=False)
    return np.array(ln.coords)


def stretch_path(coast: P.Coast, i0: int, i1: int, offs, simplify_m=4.0):
    """해안 표본점 [i0, i1) 위에 평행선 len(offs)개를 왕복(boustrophedon)으로 잇는다. 반환 (points, 길이 m)."""
    ii = np.arange(i0, i1)
    lines = []
    for p, off in enumerate(offs):
        pts = offset_line(coast.xy[ii], coast.normal[ii], off, simplify_m)
        if p % 2 == 1:
            pts = pts[::-1]
        lines.append(pts)
    path = np.vstack(lines)
    return path, _plen(path)


def runs_from_ranges(ranges, step_m, gap_fill_m=0.0):
    """[(ring, i0, i1)] 를 정렬해 인접(또는 gap_fill_m 이하 간격)한 것끼리 합친다 → 연속 비행 구간(run)."""
    rs = sorted(ranges)
    out = []
    gap_idx = int(gap_fill_m // step_m)
    for r, a, b in rs:
        if out and out[-1][0] == r and a - out[-1][2] <= gap_idx:
            out[-1] = (r, out[-1][1], max(b, out[-1][2]))
        else:
            out.append((r, a, b))
    return out


def split_runs(coast, runs, ops: Ops):
    """배터리 한 번에 들어가도록 run 을 잘라 스트레치로. 각 스트레치 = 한 번에 왕복 패스를 다 도는 단위."""
    n, spacing, _, offs = ops.passes()
    step = coast.step_m
    per_idx_s = n * step / ops.speed_mps + step / ops.transit_mps        # 조사 + 복귀 이동
    fixed_s = (n - 1) * spacing / ops.speed_mps + 2 * ops.launch_inland_m / ops.transit_mps + 30
    k_max = max(int((ops.battery_min * 60 - fixed_s) // per_idx_s), 2)
    stretches = []
    for r, a, b in runs:
        m = b - a
        parts = max(1, int(math.ceil(m / k_max)))
        edges = np.linspace(a, b, parts + 1).round().astype(int)
        for u, v in zip(edges[:-1], edges[1:]):
            if v - u < 2:
                continue
            path, L = stretch_path(coast, int(u), int(v), offs, ops.simplify_m)
            stretches.append({"ring": int(r), "i0": int(u), "i1": int(v), "path": path, "survey_m": L})
    return stretches


@dataclass
class Sortie:
    launch_xy: np.ndarray
    stretches: list = field(default_factory=list)
    survey_m: float = 0.0
    transit_m: float = 0.0

    def time_min(self, ops: Ops):
        return self.survey_m / ops.speed_mps / 60 + self.transit_m / ops.transit_mps / 60


def plan_sorties(coast, stretches, ops: Ops, launch="mobile", depot_xy=None):
    """스트레치를 순서대로 돌되 배터리를 넘기면 복귀하고 새 소티. mobile: 소티마다 첫 스트레치 시작점 근처에서 이륙."""
    sorties, cur, pos, base = [], None, None, None

    def launch_point(st):
        i = st["i0"]
        return coast.xy[i] - coast.normal[i] * ops.launch_inland_m

    def close():
        nonlocal cur
        cur.transit_m += float(np.linalg.norm(base - pos))
        sorties.append(cur); cur = None

    for st in stretches:
        if cur is None:
            base = launch_point(st) if launch == "mobile" else np.asarray(depot_xy, float)
            pos = base.copy(); cur = Sortie(base.copy())
        t_in = float(np.linalg.norm(st["path"][0] - pos)) / ops.transit_mps / 60
        t_fly = st["survey_m"] / ops.speed_mps / 60
        t_back = float(np.linalg.norm(base - st["path"][-1])) / ops.transit_mps / 60
        if cur.stretches and cur.time_min(ops) + t_in + t_fly + t_back > ops.battery_min:
            close()
            base = launch_point(st) if launch == "mobile" else np.asarray(depot_xy, float)
            pos = base.copy(); cur = Sortie(base.copy())
        cur.transit_m += float(np.linalg.norm(st["path"][0] - pos))
        cur.survey_m += st["survey_m"]
        cur.stretches.append(st); pos = st["path"][-1]
    if cur is not None and cur.stretches:
        close()
    return sorties


# ──────────────────────────────── 선택 · 포착 ────────────────────────────────

def seg_scores(segs, score):
    return np.array([float(score[s.i0:s.i1].mean()) for s in segs])


def pick_by_score(segs, sc, budget_frac):
    """점수 내림차순으로 길이 예산을 채운다 (priority.select_budget 와 같은 규칙, 객체는 건드리지 않음)."""
    total = sum(s.length_m for s in segs)
    budget = total * budget_frac
    used, chosen = 0.0, []
    for k in np.argsort(-sc):
        s = segs[k]
        if used + s.length_m > budget:
            continue
        chosen.append(s); used += s.length_m
    return chosen


def pick_geo(coast, segs, budget_frac, depot_xy, direction=1):
    """출발지에서 가장 가까운 구간부터 해안을 따라 한 방향(direction=+1 인덱스 순, −1 역순)으로 연속해서 예산만큼."""
    total = sum(s.length_m for s in segs)
    budget = total * budget_frac
    cxy = np.array([[s.cx, s.cy] for s in segs])
    k0 = int(np.argmin(np.linalg.norm(cxy - depot_xy, axis=1)))
    ring0 = segs[k0].ring_id
    ring = [s for s in segs if s.ring_id == ring0]
    others = [s for s in segs if s.ring_id != ring0]
    if direction < 0:
        ring = ring[::-1]
    j0 = next(j for j, s in enumerate(ring) if s.seg_id == segs[k0].seg_id)
    order = ring[j0:] + ring[:j0] + others
    used, chosen = 0.0, []
    for s in order:
        if used + s.length_m > budget:
            break
        chosen.append(s); used += s.length_m
    return chosen


def covered_mask(coast, runs):
    m = np.zeros(coast.n, bool)
    for _, a, b in runs:
        m[a:b] = True
    return m


def capture(coast, covered, grid_xy, w, near=None):
    near = P.nearest_coast_index(coast, grid_xy) if near is None else near
    return float(w[covered[near]].sum() / w.sum())


def inland_distance(coast, grid_xy, near=None):
    """격자 셀의 해안선 기준 부호 거리 (m, 양수 = 안쪽)."""
    near = P.nearest_coast_index(coast, grid_xy) if near is None else near
    d = grid_xy - coast.xy[near]
    return -(d * coast.normal[near]).sum(1)


# ──────────────────────────────── 전략 평가 ────────────────────────────────

def evaluate(coast, segs, chosen, ops: Ops, grid_xy, w, near, gap_fill_m=0.0, launch="mobile", depot_xy=None, keep_paths=False):
    runs = runs_from_ranges([(s.ring_id, s.i0, s.i1) for s in chosen], coast.step_m, gap_fill_m)
    stretches = split_runs(coast, runs, ops)
    sorties = plan_sorties(coast, stretches, ops, launch, depot_xy)
    cov = covered_mask(coast, runs)
    survey = sum(s.survey_m for s in sorties); transit = sum(s.transit_m for s in sorties)
    tmin = sum(s.time_min(ops) for s in sorties)
    launches = np.array([s.launch_xy for s in sorties]) if sorties else np.zeros((0, 2))
    reloc = float(np.linalg.norm(np.diff(launches, axis=0), axis=1).sum()) if len(launches) > 1 else 0.0
    frames = survey / ops.speed_mps / ops.dt_s
    cap = capture(coast, cov, grid_xy, w, near)
    out = {"coast_km": round(cov.sum() * coast.step_m / 1000, 2), "coast_frac": round(float(cov.mean()), 3),
           "n_runs": len(runs), "n_stretches": len(stretches), "n_sorties": len(sorties),
           "survey_km": round(survey / 1000, 2), "transit_km": round(transit / 1000, 2), "total_km": round((survey + transit) / 1000, 2),
           "transit_share": round(transit / max(survey + transit, 1e-9), 3),
           "time_h": round(tmin / 60, 2), "relocation_km": round(reloc / 1000, 2),
           "frames": int(frames), "detect_cpu_h": round(frames * ops.detect_s_per_frame / 3600, 2), "data_gb": round(frames * ops.mb_per_frame / 1024, 2),
           "captured_frac": round(cap, 3), "missed_frac": round(1 - cap, 3),
           "kg_frac_per_h": round(cap / max(tmin / 60, 1e-9), 4)}
    if keep_paths:
        out["_sorties"] = sorties
    return out


def score_variant(coast, kind="default", wind_from=60.0, w_bay=1.0, w_exp=0.5, w_strip=0.5, smooth_m=100.0):
    """점수 변형: default(기본 가중합) / bay / exposure / wind:<deg>(풍향만 바꾼 기본 점수)."""
    f = coast.feats
    k = max(int(smooth_m // coast.step_m), 1)

    def sm(v):
        out = np.empty_like(v)
        for rid in np.unique(coast.ring_id):
            m = coast.ring_id == rid
            out[m] = ndi.uniform_filter1d(v[m], size=k, mode="wrap")
        return out

    if kind == "bay":
        return sm(P._z(f["bay"]))
    if kind == "exposure":
        return sm(P._z(np.cos(np.radians(f["bearing"] - wind_from))))
    if kind.startswith("wind:"):
        wf = float(kind.split(":")[1])
        s = w_bay * P._z(f["bay"]) + w_exp * P._z(np.cos(np.radians(f["bearing"] - wf)))
        if "strip_m" in f:
            s = s + w_strip * P._z(np.log1p(f["strip_m"]))
        return sm(s)
    return coast.score
