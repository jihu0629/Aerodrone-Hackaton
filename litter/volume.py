"""
선회 3D → 실제 크기(m) → 쓰레기 부피.

SfM(COLMAP) 결과는 모양만 맞고 단위가 없다. 여기서:
  1) 바닥면 찾기     — 점구름에서 가장 큰 평면 (RANSAC). 법선은 카메라 쪽이 위.
  2) 실제 크기 맞추기 — 카메라가 바닥에서 떨어진 높이(SfM 단위) ↔ 드론 고도(m, SRT 상대고도
                        또는 직접 입력). 프레임마다 비율의 중앙값 = 축척.
                        (이륙 지점과 물체 바닥 높이가 같다는 가정 — 평평한 해변)
  3) 물체 찾기       — 모든 카메라가 바라보는 지점(시선 광선들의 최소제곱 교점) = 선회 중심.
                        그 주변에서 바닥보다 솟은 점 중 중심에 이어진 덩어리.
  4) 부피            — (a) 높이지도 적분: 바닥 격자마다 최고 높이 × 칸 넓이 (빈칸은 보간)
                        (b) 볼록 껍질: 물체 점 + 바닥 투영점의 볼록 껍질 부피
                       (a)는 점이 성기면 작게, (b)는 오목한 모양이면 크게 나온다 → 둘 다 보고.
"""
import numpy as np


def fit_ground(P, iters=500, thr=None, rng=None):
    """가장 많은 점을 지나는 평면 (p0, n). thr 기본 = 장면 크기의 0.5%."""
    rng = rng or np.random.default_rng(0)
    scale = np.median(np.linalg.norm(P - np.median(P, 0), axis=1))
    thr = thr or 0.005 * scale * 2
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < thr).sum())
        if k > best_n:
            best, best_n = (a, n), k
    a, n = best
    inl = P[np.abs((P - a) @ n) < thr]
    c = inl.mean(0)  # 인라이어로 다시 맞춤 (최소제곱 평면)
    _, _, vt = np.linalg.svd(inl - c)
    return c, vt[2]


def fit_ground_alt(P, C, alt, iters=3000, thr=None, rng=None, top=40):
    """바닥면 후보 여러 개 중 '카메라 높이가 드론 고도 기록과 가장 잘 맞는' 평면을 고른다.
    비스듬히 찍으면 벽·차량 면도 큰 평면이라 '가장 큰 평면'만으로는 바닥을 잘못 잡는다."""
    rng = rng or np.random.default_rng(0)
    scale = np.median(np.linalg.norm(P - np.median(P, 0), axis=1))
    thr = thr or 0.005 * scale * 2
    alt = np.broadcast_to(np.asarray(alt, float), (len(C),))
    cands = []
    for _ in range(iters):
        a, b_, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b_ - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        cands.append((int((np.abs((P - a) @ n) < thr).sum()), a, n))
    cands.sort(key=lambda t: -t[0])
    best, best_sc = None, np.inf
    kmax = cands[0][0]
    seen = []
    for k, a, n in cands:
        if k < 0.15 * kmax or len(seen) >= top:
            break
        if any(abs(n @ m) > 0.995 and abs((a - q) @ m) < 3 * thr for q, m in seen):
            continue  # 이미 본 평면과 같은 평면
        seen.append((a, n))
        h = (C - a) @ n
        if np.median(h) < 0:
            h, n = -h, -n
        ok = h > 1e-9
        if ok.sum() < 0.8 * len(C):  # 카메라가 대부분 평면 위쪽에 있어야 바닥
            continue
        r = alt[ok] / h[ok]
        cv = float(np.std(r) / np.mean(r))
        sc = cv - 0.05 * k / kmax  # 고도 비율이 일정할수록 + 점이 많을수록 좋음
        if sc < best_sc:
            best_sc, best = sc, (a, n)
    if best is None:
        return fit_ground(P)
    a, n = best
    inl = P[np.abs((P - a) @ n) < thr]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n2 = vt[2] if vt[2] @ n > 0 else -vt[2]
    return c, n2


def rays_center(C, D):
    """여러 시선 광선 (C_i + t D_i)에 가장 가까운 점."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(C, D):
        d = d / np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ c
    return np.linalg.solve(A, b)


def to_metric(P, C, D, alt_m):
    """바닥 기준 좌표계(z=위, m)로 변환. alt_m: 프레임별 고도 배열 또는 하나의 값."""
    p0, n = fit_ground_alt(P, C, alt_m)
    if np.median((C - p0) @ n) < 0:  # 카메라가 있는 쪽이 위
        n = -n
    h = (C - p0) @ n
    alt = np.broadcast_to(np.asarray(alt_m, float), h.shape)
    ok = h > 1e-9
    ratios = alt[ok] / h[ok]
    s = float(np.median(ratios))
    # 바닥 좌표축
    x = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n])

    def T(Q):
        return ((Q - p0) @ R.T) * s

    return T, R, s, {"scale_m_per_unit": s, "scale_spread_pct": float(np.std(ratios) / s * 100),
                     "n_frames_for_scale": int(ok.sum()), "p0": p0}


def segment_object(Pm, center_xy, radius=None, orbit_r=None, h_min=0.03, link=None):
    """선회 중심 주변에서 바닥보다 h_min 이상 솟은 점 중 중심에 이어진 덩어리."""
    from scipy.spatial import cKDTree

    radius = radius or (0.6 * orbit_r if orbit_r else 2.0)
    d = np.linalg.norm(Pm[:, :2] - center_xy, axis=1)
    cand = np.nonzero((Pm[:, 2] > h_min) & (d < radius))[0]
    if len(cand) < 5:
        return cand
    Q = Pm[cand]
    link = link or max(0.05, radius / 20)
    tree = cKDTree(Q)
    pairs = tree.query_pairs(link, output_type="ndarray")
    parent = np.arange(len(Q))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(len(Q))])
    # 중심에 가장 가까운 점이 속한 덩어리 (그 덩어리가 너무 작으면 가장 큰 덩어리)
    r0 = roots[np.argmin(np.linalg.norm(Q[:, :2] - center_xy, axis=1))]
    labels, counts = np.unique(roots, return_counts=True)
    if counts[labels == r0][0] < 0.2 * counts.max():
        r0 = labels[np.argmax(counts)]
    return cand[roots == r0]


def volume_heightmap(O, cell=None, agg="max"):
    """바닥 격자마다 최고 높이 → 빈칸은 발자국 안에서 보간 → Σ 높이 × 칸 넓이."""
    from matplotlib.path import Path as MPath
    from scipy.interpolate import griddata
    from scipy.spatial import ConvexHull

    xy = O[:, :2]
    span = np.ptp(xy, axis=0).max()
    try:
        hull = ConvexHull(xy)
        area = hull.volume  # 2D에서 volume = 넓이
        poly = MPath(xy[hull.vertices])
    except Exception:
        area, poly = span * span, None
    # 칸 크기: 칸마다 점이 평균 3개쯤 들어가게 (점이 성기면 칸을 키운다)
    cell = cell or float(np.clip(np.sqrt(area / max(len(O) / 3, 1)), span / 80, span / 6))
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(int)
    W, H = ij.max(0) + 1
    hm = np.full((W, H), np.nan)
    if agg == "median":  # 칸마다 중앙값 — 모서리의 튀는 점에 덜 민감 (조밀 점구름용)
        from collections import defaultdict
        cells = defaultdict(list)
        for (i, j), z in zip(ij, O[:, 2]):
            cells[(i, j)].append(z)
        for (i, j), zs in cells.items():
            hm[i, j] = float(np.median(zs))
    else:
        for (i, j), z in zip(ij, O[:, 2]):
            if np.isnan(hm[i, j]) or z > hm[i, j]:
                hm[i, j] = z
    # 발자국: 물체 점들의 2D 볼록 껍질 안쪽 칸 (쓰레기 덩어리는 대체로 볼록)
    gi, gj = np.mgrid[:W, :H]
    centers = np.c_[(gi.ravel() + .5) * cell + lo[0], (gj.ravel() + .5) * cell + lo[1]]
    foot = poly.contains_points(centers).reshape(W, H) if poly is not None else ~np.isnan(hm)
    foot |= ~np.isnan(hm)
    known = ~np.isnan(hm)
    if known.sum() >= 4 and (foot & ~known).any():
        ki, kj = np.nonzero(known)
        ui, uj = np.nonzero(foot & ~known)
        hm[ui, uj] = griddata(np.c_[ki, kj], hm[ki, kj], np.c_[ui, uj], method="linear")
        miss = np.isnan(hm) & foot
        if miss.any():
            mi, mj = np.nonzero(miss)
            hm[mi, mj] = griddata(np.c_[ki, kj], hm[ki, kj], np.c_[mi, mj], method="nearest")
    hm[~foot] = 0
    hm = np.nan_to_num(np.clip(hm, 0, None))
    return float(hm.sum() * cell * cell), float(foot.sum() * cell * cell), cell


def volume_hull(O):
    from scipy.spatial import ConvexHull

    base = O.copy()
    base[:, 2] = 0
    try:
        return float(ConvexHull(np.vstack([O, base])).volume)
    except Exception:
        return float("nan")


def measure(P, C, D, alt_m, radius=None, h_min=0.03):
    """P: SfM 점 (N,3), C: 카메라 중심 (M,3), D: 카메라 시선 방향 (M,3), alt_m: 고도(m)."""
    T, R, s, info = to_metric(P, C, D, alt_m)
    Pm, Cm = T(P), T(C)
    Dm = D @ R.T
    ctr = rays_center(Cm, Dm)
    orbit_r = float(np.median(np.linalg.norm(Cm[:, :2] - ctr[:2], axis=1)))
    idx = segment_object(Pm, ctr[:2], radius, orbit_r, h_min)
    out = dict(info, orbit_radius_m=orbit_r, orbit_height_m=float(np.median(Cm[:, 2])),
               center_xy=ctr[:2].tolist(), n_object_points=int(len(idx)))
    if len(idx) < 10:
        out["error"] = "물체 점이 너무 적음 — 선회 중심에 물체가 없거나 점구름이 성김"
        return out, Pm, idx
    O = Pm[idx]
    v_hm, foot, cell = volume_heightmap(O)
    v_hull = volume_hull(O)
    # 크기: 바닥 투영의 최소 외접 직사각형
    import cv2
    rect = cv2.minAreaRect(O[:, :2].astype(np.float32))
    L, Wd = sorted(rect[1], reverse=True)
    out.update(volume_heightmap_m3=v_hm, volume_hull_m3=v_hull, footprint_m2=foot, grid_cell_m=cell,
               length_m=float(L), width_m=float(Wd), height_m=float(np.percentile(O[:, 2], 98)))
    return out, Pm, idx


def to_metric_auto(P, C, D, alt_m, gps_xy=None, min_track_m=10.0):
    """실제 크기 맞추기를 상황에 맞게 고른다.
    - 기본: 카메라 높이 ↔ SRT 상대고도 (제자리 선회처럼 수평 이동이 작을 때)
    - GPS 수평 이동이 min_track_m 이상이면: 3D 카메라 경로 ↔ GPS 경로로 축척을 다시 맞춘다
      (SRT 고도는 '이륙 지점 기준'이라, 이륙 지점이 바닥보다 높으면 크기가 통째로 틀어진다)"""
    T, R, s, info = to_metric(P, C, D, alt_m)
    info["scale_source"] = "고도(SRT)"
    if gps_xy is None:
        return T, R, s, info
    G = np.asarray(gps_xy, float)
    if np.ptp(G, axis=0).max() < min_track_m:
        return T, R, s, info
    A = T(C)[:, :2]
    ca, cb = A.mean(0), G.mean(0)
    U, S, Vt = np.linalg.svd((G - cb).T @ (A - ca))
    d = np.sign(np.linalg.det(U @ Vt))
    k = float((S * np.array([1, d])).sum() / ((A - ca) ** 2).sum())
    p0 = info["p0"]
    s2 = s * k

    def T2(Q):
        return ((Q - p0) @ R.T) * s2

    h = T2(C)[:, 2]
    info.update(scale_m_per_unit=s2, scale_source="GPS 경로", gps_scale_factor=k,
                camera_height_m=float(np.median(h)))
    return T2, R, s2, info
