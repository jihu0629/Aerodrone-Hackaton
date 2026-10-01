"""
위치(georef) · 중복 제거 · 형상 측정(면적·길이·높이·부피).

① 위치: 라벨 폴리곤의 픽셀마다 광선을 쏴서 지면 좌표로 옮긴다.
   비교용으로 "기존 방식"(드론 GPS = 쓰레기 위치)과 "화면 중앙"
   위치도 같이 기록해서, 수거 GPS 정답이 있으면 오차를 비교한다.
② 중복 제거: 동영상/연속사진은 같은 쓰레기가 여러 프레임에 라벨된다.
   item_id가 있으면 그걸로, 없으면 같은 클래스 + 가까운 거리로 합친다.
③ 형상: 지면 폴리곤의 면적·뼈대 길이, DSM이 있으면 주변 모래 높이를
   평면으로 맞춘 바닥면 대비 높이·부피 (Σ(DSM−바닥)×픽셀면적).
"""
import numpy as np
from shapely.geometry import Polygon

from .camera import Intrinsics, Pose, georef_uncertainty, pixel_to_ground


class Geo:
    """위경도 ↔ 미터 지도좌표 (UTM). epsg를 안 주면 첫 좌표로 존 결정."""

    def __init__(self, epsg=None, lon=None):
        from pyproj import Transformer

        if epsg is None:
            epsg = 32600 + int((lon + 180) // 6) + 1
        self.epsg = epsg
        self._fwd = Transformer.from_crs(4326, epsg, always_xy=True)
        self._inv = Transformer.from_crs(epsg, 4326, always_xy=True)

    def to_xy(self, lat, lon):
        return self._fwd.transform(lon, lat)

    def to_latlon(self, x, y):
        lon, lat = self._inv.transform(x, y)
        return lat, lon


class DSM:
    """GeoTIFF DSM을 통째로 메모리에 올려서 z(x, y) 샘플링 + 창(window) 읽기."""

    def __init__(self, path):
        import rasterio

        with rasterio.open(path) as ds:
            self.z = ds.read(1).astype(np.float32)
            self.T = ds.transform
            self.epsg = ds.crs.to_epsg() if ds.crs else None
            nod = ds.nodata
        if nod is not None:
            self.z[self.z == nod] = np.nan
        self.res = abs(self.T.a)

    def rc(self, x, y):
        inv = ~self.T
        c, r = inv * (np.asarray(x, float), np.asarray(y, float))
        return r, c

    def __call__(self, x, y):
        r, c = self.rc(x, y)
        r, c = np.floor(r).astype(int), np.floor(c).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < self.z.shape[0]) & (c < self.z.shape[1])
        out = np.full(r.shape, np.nan, np.float32)
        out[ok] = self.z[r[ok], c[ok]]
        return out


def _intrinsics(t, ann):
    W = ann.get("width") or t.get("width")
    H = ann.get("height") or t.get("height")
    if "f35" in t:
        return Intrinsics.from_f35(W, H, t["f35"])
    return Intrinsics.from_hfov(W, H, t.get("hfov", 84.0))  # 84° = DJI 광각 기본


def georef(anns, tel, geo, cfg, dsm=None, ortho_T=None, takeoff_z=0.0):
    """annotation마다 지면 폴리곤·중심·불확실성 반경을 붙인다.
    ortho_T: 라벨이 정사영상 위에 달린 경우 그 GeoTIFF의 affine transform.
    takeoff_z: 이륙 지점 높이 (DSM과 같은 기준, DSM 없으면 0 = 평평한 해변 가정)."""
    rng = np.random.default_rng(0)
    skipped = 0
    for a in anns:
        uv = np.asarray(a["polygon"], float)
        T = ortho_T
        if a.get("geo_transform"):
            from affine import Affine
            T = Affine(*a["geo_transform"])
        if T is not None:
            xs, ys = T * (uv[:, 0], uv[:, 1])
            a["ground_poly"] = np.stack([xs, ys], 1).tolist()
            a["x"], a["y"] = Polygon(a["ground_poly"]).buffer(0).centroid.coords[0]
            a["r95_m"] = abs(T.a) * 2  # 정사영상 자체의 절대위치 오차는 별도 (GCP 정확도)
            continue
        t = tel.get(a["image"])
        if not t or not {"lat", "lon", "alt", "pitch"} <= set(t):
            a["georef_ok"] = False
            skipped += 1
            continue
        dx, dy = geo.to_xy(t["lat"], t["lon"])
        # DJI 상대고도는 "이륙 지점" 기준 — 그 지점의 DSM 높이가 takeoff_z
        pose = Pose(dx, dy, t["alt"], t.get("yaw", 0.0), t["pitch"], t.get("roll", 0.0), ground_z=takeoff_z)
        K = _intrinsics(t, a)
        P = pixel_to_ground(K, pose, uv, dsm=dsm)
        if not np.isfinite(P).all():
            a["georef_ok"] = False
            skipped += 1
            continue
        c_uv = Polygon(uv).centroid.coords[0]
        c = pixel_to_ground(K, pose, [c_uv], dsm=dsm)[0]
        _, r95 = georef_uncertainty(K, pose, [c_uv], cfg["georef"], rng=rng)
        ctr = pixel_to_ground(K, pose, [[K.cx, K.cy]])[0]
        a.update(georef_ok=True, ground_poly=P[:, :2].tolist(), x=float(c[0]), y=float(c[1]),
                 r95_m=r95, naive_x=dx, naive_y=dy, center_x=float(ctr[0]), center_y=float(ctr[1]),
                 pitch=t["pitch"], alt=t["alt"])
    if skipped:
        print(f"  ⚠️ 텔레메트리 부족으로 위치 계산 못 한 라벨: {skipped}/{len(anns)}")
    for a in anns:
        if a.get("georef_ok", True) and "x" in a:
            a["lat"], a["lon"] = geo.to_latlon(a["x"], a["y"])
    return [a for a in anns if "x" in a]


def dedup(anns, cfg):
    """여러 프레임의 같은 물체를 하나로. 대표 관측 = 위치 불확실성이 가장 작은 것."""
    groups = {}
    if all(a.get("item_id") for a in anns):
        for a in anns:
            groups.setdefault(a["item_id"], []).append(a)
    else:
        R = cfg["georef"]["dedup_radius_m"]
        parent = list(range(len(anns)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        xy = np.array([[a["x"], a["y"]] for a in anns])
        for i in range(len(anns)):
            d = np.linalg.norm(xy[i + 1:] - xy[i], axis=1)
            for j in np.nonzero(d < R)[0] + i + 1:
                if anns[i]["cls"] == anns[j]["cls"] and anns[i]["image"] != anns[j]["image"]:
                    parent[find(j)] = find(i)
        for i, a in enumerate(anns):
            groups.setdefault(find(i), []).append(a)
    items = []
    for gid, g in groups.items():
        best = min(g, key=lambda a: a.get("r95_m", 0))
        r = np.clip([a.get("r95_m", 1.0) for a in g], 0.1, 1e3)
        w = 1 / r ** 2   # 위치가 확실한 관측(수직·가까운 프레임)에 더 큰 가중치
        it = dict(best)
        it["item_id"] = str(best.get("item_id") or gid)
        it["x"] = float(np.average([a["x"] for a in g], weights=w))
        it["y"] = float(np.average([a["y"] for a in g], weights=w))
        it["r95_m"] = float(1 / np.sqrt(w.sum())) if len(g) > 1 else best.get("r95_m", np.nan)
        it["n_views"] = len(g)
        it["ann_ids"] = [a["ann_id"] for a in g]
        ws = [a["weight_kg"] for a in g if a.get("weight_kg") is not None]
        if ws:
            it["weight_kg"] = float(np.mean(ws))
        items.append(it)
    return items


def snap_to_dsm(items, cfg, dsm, ground_win_m=4.0):
    """텔레메트리로 옮긴 폴리곤을 DSM 위의 '솟은 덩어리'에 맞춰 붙인다.

    드론 GPS·짐벌각 오차 때문에 지면 폴리곤이 1–3 m 밀려 있으면 DSM 부피를
    엉뚱한 모래에서 재게 된다. 위치 불확실성 반경(r95) 안에서 폴리곤 모양을
    움직여 보며 "폴리곤 안 평균 높이(주변 지면 대비)"가 최대인 곳으로 옮긴다.
    ODM이 내준 보정된 카메라 자세를 쓰면 이 단계는 거의 0으로 움직여야 정상.
    """
    import cv2

    Z = np.nan_to_num(dsm.z, nan=float(np.nanmin(dsm.z)))
    k = max(3, int(round(ground_win_m / dsm.res)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    ground = cv2.GaussianBlur(cv2.morphologyEx(Z, cv2.MORPH_OPEN, ker), (0, 0), k / 6)
    relief = (Z - ground).astype(np.float32)
    noise = cfg["weight"]["dsm_noise_m"]
    k_ring = max(1, int(round(0.3 / dsm.res)))
    n_moved = 0
    for it in items:
        P = np.asarray(it["ground_poly"])
        r95 = float(it.get("r95_m", 2.0))
        if not np.isfinite(r95):
            continue
        R = float(np.clip(1.2 * r95, 0.5, 6.0))
        sigma = max(r95 / 2.45, 0.2)  # 2D 정규분포에서 r95 ≈ 2.45σ
        rr, cc = dsm.rc(P[:, 0], P[:, 1])
        pad = k_ring + 1
        pr0, pc0 = int(np.floor(rr.min())) - pad, int(np.floor(cc.min())) - pad
        tpl = np.zeros((int(np.ceil(rr.max())) - pr0 + 1 + pad, int(np.ceil(cc.max())) - pc0 + 1 + pad), np.uint8)
        cv2.fillPoly(tpl, [np.stack([cc - pc0, rr - pr0], 1).round().astype(np.int32)], 1)
        if tpl.sum() == 0 or tpl.size > 4e6:
            continue
        ring = cv2.dilate(tpl, np.ones((2 * k_ring + 1,) * 2, np.uint8)) - tpl
        rp = int(np.ceil(R / dsm.res))
        r0, c0 = pr0 - rp, pc0 - rp
        r1, c1 = r0 + tpl.shape[0] + 2 * rp, c0 + tpl.shape[1] + 2 * rp
        if r0 < 0 or c0 < 0 or r1 > Z.shape[0] or c1 > Z.shape[1]:
            continue
        win = relief[r0:r1, c0:c1]
        # 대비 = 폴리곤 안 평균 높이 − 바로 바깥 링 평균 높이
        #  (안쪽만 보면 옆의 더 큰 덩어리 일부에 걸쳐도 점수가 높아진다)
        contrast = (cv2.matchTemplate(win, tpl.astype(np.float32), cv2.TM_CCORR) / tpl.sum()
                    - cv2.matchTemplate(win, ring.astype(np.float32), cv2.TM_CCORR) / max(ring.sum(), 1))
        yy, xx = np.mgrid[:contrast.shape[0], :contrast.shape[1]]
        d2 = ((yy - rp) ** 2 + (xx - rp) ** 2) * dsm.res ** 2
        # 위치 사전분포: 텔레메트리가 말한 곳에서 멀수록 감점
        score = np.where(d2 <= R ** 2, contrast * np.exp(-d2 / (2 * sigma ** 2)), -np.inf)
        dy, dx = np.unravel_index(np.argmax(score), score.shape)
        if contrast[dy, dx] < 3 * noise:
            continue  # 뚜렷한 덩어리가 없음 (얇은 물체 등) — 제자리 유지
        # 최종 위치: 옮긴 폴리곤(+링) 안에서 높이 가중 중심 — 비스듬히 찍혀 부푼 외곽선 보정
        sub = win[dy:dy + tpl.shape[0], dx:dx + tpl.shape[1]]
        m = (tpl + ring).astype(bool) & (sub > 2 * noise)
        if m.sum() >= 3:
            ys, xs = np.nonzero(m)
            wts = sub[m]
            cy, cx = (ys * wts).sum() / wts.sum(), (xs * wts).sum() / wts.sum()
            ty, tx = np.nonzero(tpl)
            dyf, dxf = dy + cy - ty.mean(), dx + cx - tx.mean()
        else:
            dyf, dxf = dy, dx
        sx, sy = (dxf - rp) * dsm.res, -(dyf - rp) * dsm.res  # 행이 아래로 갈수록 y 감소
        it["ground_poly"] = (P + [sx, sy]).tolist()
        it["x"], it["y"] = it["x"] + sx, it["y"] + sy
        it["snap_m"] = float(np.hypot(sx, sy))
        n_moved += 1
    return n_moved


def _skeleton_length(poly, res):
    """폴리곤을 res 격자에 그려서 뼈대(skeleton) 길이를 잰다 — 로프·어망처럼 긴 물체용."""
    import cv2
    from skimage.morphology import skeletonize

    P = np.asarray(poly)
    lo = P.min(0) - res * 2
    pix = np.round((P - lo) / res).astype(np.int32)
    H, W = pix[:, 1].max() + 3, pix[:, 0].max() + 3
    if H * W > 4e7:  # 너무 크면 해상도를 낮춘다
        return _skeleton_length(poly, res * 2)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pix], 1)
    sk = skeletonize(m.astype(bool))
    n = int(sk.sum())
    if n < 2:
        return 0.0
    # 대각 이웃은 √2 — 이웃 쌍의 거리 합의 절반 ≈ 선 길이
    ys, xs = np.nonzero(sk)
    s = set(zip(ys.tolist(), xs.tolist()))
    tot = 0.0
    for y, x in s:
        for dy, dx, dd in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
            if (y + dy, x + dx) in s:
                tot += dd
    return tot * res


def measure(items, cfg, dsm=None):
    import cv2

    noise = cfg["weight"]["dsm_noise_m"]
    for it in items:
        poly = Polygon(it["ground_poly"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        area = float(poly.area)
        it["area_m2"] = area
        mrr = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
        sides = np.linalg.norm(np.diff(mrr, axis=0), axis=1)[:2]
        res = max(0.01, np.sqrt(max(area, 1e-4)) / 60)
        it["length_m"] = float(max(sides.max(), _skeleton_length(it["ground_poly"], res)))
        it["elongation"] = float(it["length_m"] ** 2 / max(area, 1e-6))
        if dsm is None:
            continue
        # DSM 창: 폴리곤 bbox + 1 m
        x0, y0, x1, y1 = poly.bounds
        r0, c0 = dsm.rc(x0 - 1, y1 + 1)
        r1, c1 = dsm.rc(x1 + 1, y0 - 1)
        r0, c0 = max(0, int(r0)), max(0, int(c0))
        r1, c1 = min(dsm.z.shape[0], int(r1) + 1), min(dsm.z.shape[1], int(c1) + 1)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        Z = dsm.z[r0:r1, c0:c1]
        # 꼬인 로프 폴리곤은 buffer(0) 후 여러 조각이 될 수 있어서 조각마다 그린다
        m = np.zeros(Z.shape, np.uint8)
        for part in getattr(poly, "geoms", [poly]):
            rr, cc = dsm.rc(*np.asarray(part.exterior.coords).T)
            cv2.fillPoly(m, [np.stack([cc - c0, rr - r0], 1).round().astype(np.int32)], 1)
        k_in = max(1, int(round(0.15 / dsm.res)))
        k_out = max(k_in + 1, int(round(0.6 / dsm.res)))
        ring = (cv2.dilate(m, np.ones((2 * k_out + 1,) * 2, np.uint8)) -
                cv2.dilate(m, np.ones((2 * k_in + 1,) * 2, np.uint8))).astype(bool)
        ring &= np.isfinite(Z)
        inside = m.astype(bool) & np.isfinite(Z)
        if ring.sum() < 10 or inside.sum() < 1:
            continue
        # 바닥면: 링의 모래 높이에 평면을 맞춘다 (잔차 큰 점 한 번 제거 — 옆 쓰레기 등)
        ys, xs = np.nonzero(ring)
        A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(float)
        zr = Z[ring]
        coef, *_ = np.linalg.lstsq(A, zr, rcond=None)
        keep = np.abs(A @ coef - zr) < 3 * max(noise, np.median(np.abs(A @ coef - zr)) * 1.5)
        if keep.sum() > 5:
            coef, *_ = np.linalg.lstsq(A[keep], zr[keep], rcond=None)
        yi, xi = np.nonzero(inside)
        base = np.stack([xi, yi, np.ones_like(xi)], 1) @ coef
        h = np.clip(Z[inside] - base, 0, None)
        it["h_p90_m"] = float(np.percentile(h, 90))
        it["h_med_m"] = float(np.median(h))
        it["volume_m3"] = float(h.sum() * dsm.res ** 2)
        it["valid_3d"] = bool(it["h_p90_m"] > 2 * noise)
        # 묻힘 의심: 넓게 퍼졌는데 높이가 거의 없음 (어망·로프)
        it["buried_suspect"] = bool(it["cls"] in ("net", "rope") and area > 0.3 and it["h_p90_m"] < 0.05)
    return items
