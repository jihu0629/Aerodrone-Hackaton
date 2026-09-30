"""
라벨 폴리곤 + DSM -> 면적·부피·무게 추정.

3D 지도(06 단계)의 결과물 중 두 장을 씁니다.
  DSM (Digital Surface Model)  : 표면 높이. 쓰레기 더미, 건물, 나무 꼭대기가 포함된 높이.
  DTM (Digital Terrain Model)  : 지면 높이. ODM 이 점군에서 지면점만 골라 만든 것.

부피 = Σ (DSM - 바닥면) x 픽셀 면적,  라벨 폴리곤 안에서만, 양수 부분만.

바닥면(base) 은 세 가지 중 하나입니다.
  dtm    : DTM 을 그대로 바닥으로. 지면이 평평하지 않은 곳(경사진 해빈)에서 가장 정확. 기본.
  plane  : 폴리곤 바깥 테두리(ring) 픽셀의 DSM 에 평면을 맞춰 바닥으로. DTM 이 없거나 DTM 이
           더미를 지면으로 잘못 분류했을 때. 테두리가 깨끗한 모래면 잘 맞습니다.
  min    : 폴리곤 안 DSM 최솟값을 바닥으로. 가장 단순, 경사지에서 부피가 과대.

무게 = 부피 x 겉보기 밀도(kg/m³). 겉보기 밀도는 쌓인 상태의 밀도이며 재질 밀도가 아닙니다
(비닐 더미는 공기가 대부분). 아래 DENSITY_KG_M3 는 문헌 근사값이고 현장 표본으로 보정해야
합니다 (확인 필요). --density-json 으로 덮어쓸 수 있습니다.

라벨은 GeoJSON 폴리곤이며 속성 "class" 에 종류를 씁니다. QGIS 에서 정사영상 위에 그리거나,
탐지 모델 결과를 같은 형식으로 저장해서 넣습니다. 좌표계는 DSM 과 같아야 합니다(다르면 재투영).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shp_transform

from .config import safe_path

# 겉보기(쌓인 상태) 밀도 kg/m³. 확인 필요: 현장에서 1 m³ 를 저울로 재서 보정하세요.
DENSITY_KG_M3: dict[str, float] = {
    "mixed": 120.0,        # 혼합 해안쓰레기 (플라스틱+어구+목재)
    "plastic": 60.0,       # 페트병·비닐·용기 (압축 안 됨)
    "styrofoam": 15.0,     # 스티로폼 부표·박스
    "net": 250.0,          # 폐어망·로프
    "rope": 250.0,
    "wood": 350.0,         # 유목·목재
    "glass": 300.0,        # 병
    "metal": 200.0,        # 캔·철재
    "tire": 400.0,
    "seaweed": 500.0,      # 해조류 (습윤)
    "sand": 1600.0,        # 모래 (침식·퇴적량 계산용)
    "soil": 1500.0,
    "gravel": 1700.0,
}


@dataclass
class Measurement:
    id: int
    label: str
    area_m2: float
    volume_m3: float          # 바닥면 위 양수 부피
    volume_neg_m3: float      # 바닥면 아래 (침식·구덩이) 부피. 절대값
    height_mean_m: float
    height_max_m: float
    base_method: str
    density_kg_m3: float | None
    weight_kg: float | None
    n_pixels: int
    n_nodata: int
    centroid_x: float
    centroid_y: float


def load_labels(path: str | Path, target_crs=None) -> tuple[list[Polygon | MultiPolygon], list[dict], str | None]:
    """라벨 GeoJSON 을 읽어 (도형, 속성, 좌표계이름) 을 돌려줍니다. target_crs 가 있으면 재투영."""
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    crs_name = None
    if isinstance(fc.get("crs"), dict):
        crs_name = fc["crs"].get("properties", {}).get("name")
    geoms, props = [], []
    for feat in fc.get("features", []):
        g = shape(feat["geometry"])
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        geoms.append(g)
        props.append(dict(feat.get("properties") or {}))
    # GeoJSON 표준(RFC 7946) 은 crs 필드가 없으면 WGS84
    src_crs = crs_name or "EPSG:4326"
    if str(src_crs).upper() == "URN:OGC:DEF:CRS:OGC:1.3:CRS84":
        src_crs = "EPSG:4326"
    if target_crs is not None:
        from pyproj import CRS, Transformer

        if not CRS.from_user_input(src_crs).equals(CRS.from_user_input(target_crs)):
            tr = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            geoms = [shp_transform(tr.transform, g) for g in geoms]
    return geoms, props, src_crs


def _fit_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[float, float, float]:
    """z = a x + b y + c 최소제곱. 점이 3개 미만이면 평균 높이 평면."""
    if len(zs) < 3:
        return 0.0, 0.0, float(np.mean(zs)) if len(zs) else 0.0
    # 수치 안정성을 위해 중심 이동
    x0, y0 = xs.mean(), ys.mean()
    A = np.column_stack([xs - x0, ys - y0, np.ones_like(xs)])
    sol, *_ = np.linalg.lstsq(A, zs, rcond=None)
    a, b, c0 = sol
    c = c0 - a * x0 - b * y0
    return float(a), float(b), float(c)


def _robust_plane(xs, ys, zs, iters: int = 3, k: float = 2.0):
    """잔차가 큰 점(테두리에 걸린 쓰레기 조각)을 반복 제거하며 평면 맞춤."""
    mask = np.ones(len(zs), dtype=bool)
    plane = _fit_plane(xs, ys, zs)
    for _ in range(iters):
        a, b, c = plane
        res = zs - (a * xs + b * ys + c)
        s = np.std(res[mask]) if mask.sum() > 3 else 0.0
        if s == 0:
            break
        new_mask = np.abs(res) < k * s
        if new_mask.sum() < 3 or new_mask.sum() == mask.sum():
            break
        mask = new_mask
        plane = _fit_plane(xs[mask], ys[mask], zs[mask])
    return plane


def measure_polygon(
    poly: Polygon | MultiPolygon,
    dsm: np.ndarray,
    transform: Affine,
    nodata: float | None,
    dtm: np.ndarray | None = None,
    base: str = "dtm",
    ring_m: float = 1.0,
    label: str = "mixed",
    density: dict[str, float] | None = None,
    pid: int = 0,
) -> Measurement:
    """폴리곤 하나의 면적·부피·무게를 계산합니다. dsm/dtm 은 같은 격자의 2D 배열(전체 또는 창)."""
    density = density or DENSITY_KG_M3
    px_w, px_h = abs(transform.a), abs(transform.e)
    px_area = px_w * px_h
    h, w = dsm.shape

    inside = features.rasterize([(mapping(poly), 1)], out_shape=(h, w), transform=transform,
                                fill=0, all_touched=False, dtype="uint8").astype(bool)
    valid = np.isfinite(dsm)
    if nodata is not None:
        valid &= dsm != nodata
    n_nodata = int((inside & ~valid).sum())
    sel = inside & valid

    if base == "dtm" and dtm is None:
        base = "plane"
    if base == "dtm":
        dvalid = np.isfinite(dtm)
        if nodata is not None:
            dvalid &= dtm != nodata
        sel &= dvalid
        base_z = np.where(sel, dtm, np.nan)
        method = "dtm"
    elif base == "plane":
        ring = poly.buffer(ring_m).difference(poly)
        ring_mask = features.rasterize([(mapping(ring), 1)], out_shape=(h, w), transform=transform,
                                       fill=0, all_touched=True, dtype="uint8").astype(bool) & valid
        rows, cols = np.nonzero(ring_mask)
        if len(rows) >= 3:
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
            xs, ys = np.asarray(xs), np.asarray(ys)
            a, b, c = _robust_plane(xs, ys, dsm[rows, cols].astype(float))
            rr, cc = np.nonzero(sel)
            px, py = rasterio.transform.xy(transform, rr, cc, offset="center")
            base_z = np.full(dsm.shape, np.nan)
            base_z[rr, cc] = a * np.asarray(px) + b * np.asarray(py) + c
            method = "plane"
        else:
            base_z = np.full(dsm.shape, np.nanmin(np.where(sel, dsm, np.nan)) if sel.any() else np.nan)
            method = "min(plane-fallback)"
    elif base == "min":
        base_z = np.full(dsm.shape, np.nanmin(np.where(sel, dsm, np.nan)) if sel.any() else np.nan)
        method = "min"
    else:
        raise ValueError(f"base 는 dtm/plane/min 중 하나: {base}")

    diff = np.where(sel, dsm - base_z, np.nan)
    pos = np.nan_to_num(np.clip(diff, 0, None), nan=0.0)
    neg = np.nan_to_num(np.clip(-diff, 0, None), nan=0.0)
    vol = float(pos.sum() * px_area)
    vol_neg = float(neg.sum() * px_area)
    heights = diff[sel]
    n = int(sel.sum())
    rho = density.get(label)
    c = poly.centroid
    return Measurement(
        id=pid, label=label,
        area_m2=float(poly.area),
        volume_m3=vol, volume_neg_m3=vol_neg,
        height_mean_m=float(np.nanmean(heights)) if n else 0.0,
        height_max_m=float(np.nanmax(heights)) if n else 0.0,
        base_method=method,
        density_kg_m3=rho,
        weight_kg=(vol * rho) if rho is not None else None,
        n_pixels=n, n_nodata=n_nodata,
        centroid_x=float(c.x), centroid_y=float(c.y),
    )


def _read_window_for(src, poly, pad_m: float):
    """폴리곤(+여유) 만 덮는 창을 읽습니다. 큰 DSM 을 통째로 올리지 않기 위해."""
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = poly.buffer(pad_m).bounds
    win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
    # 래스터 범위로 자르기
    col0 = max(0, int(win.col_off)); row0 = max(0, int(win.row_off))
    col1 = min(src.width, int(win.col_off + win.width)); row1 = min(src.height, int(win.row_off + win.height))
    if col1 <= col0 or row1 <= row0:
        return None, None
    from rasterio.windows import Window

    w = Window(col0, row0, col1 - col0, row1 - row0)
    arr = src.read(1, window=w).astype(float)
    return arr, src.window_transform(w)


def measure_labels(
    dsm_path: str | Path,
    labels_path: str | Path,
    dtm_path: str | Path | None = None,
    base: str = "dtm",
    ring_m: float = 1.0,
    class_field: str = "class",
    default_label: str = "mixed",
    density: dict[str, float] | None = None,
    log=print,
) -> tuple[list[Measurement], list, str]:
    """라벨 GeoJSON 의 모든 폴리곤을 측정합니다. (측정 목록, DSM 좌표계의 도형 목록, crs) 반환."""
    density = density or DENSITY_KG_M3
    with rasterio.open(safe_path(dsm_path)) as dsm_src:
        crs = dsm_src.crs
        nodata = dsm_src.nodata
        if crs is None:
            raise ValueError("DSM 에 좌표계가 없습니다. ODM 결과(odm_dem/dsm.tif) 를 쓰거나 EPSG 를 지정하세요.")
        geoms, props, _ = load_labels(labels_path, target_crs=crs)
        dtm_src = rasterio.open(safe_path(dtm_path)) if dtm_path else None
        if dtm_src is not None and (dtm_src.transform != dsm_src.transform or dtm_src.shape != dsm_src.shape):
            log("[volume] DTM 격자가 DSM 과 다릅니다. DSM 격자로 재샘플링합니다.")
        out: list[Measurement] = []
        for i, (g, p) in enumerate(zip(geoms, props)):
            label = str(p.get(class_field) or default_label).strip().lower()
            if label not in density:
                log(f"[volume] 알 수 없는 class '{label}' (id {i}): 밀도 없음, 무게는 비움. DENSITY_KG_M3 또는 --density-json 에 추가하세요.")
            dsm_arr, tr = _read_window_for(dsm_src, g, pad_m=ring_m + 2 * abs(dsm_src.transform.a))
            if dsm_arr is None:
                log(f"[volume] 라벨 {i} 가 DSM 범위 밖입니다. 건너뜀.")
                continue
            dtm_arr = None
            if dtm_src is not None:
                from rasterio.warp import Resampling, reproject

                dtm_arr = np.full(dsm_arr.shape, np.nan)
                reproject(rasterio.band(dtm_src, 1), dtm_arr, dst_transform=tr, dst_crs=crs,
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
            m = measure_polygon(g, dsm_arr, tr, nodata, dtm_arr, base=base, ring_m=ring_m,
                                label=label, density=density, pid=int(p.get("id", i)))
            out.append(m)
        if dtm_src is not None:
            dtm_src.close()
    return out, geoms, str(crs)


def save_measurements(meas: list[Measurement], geoms, crs: str, out_dir: str | Path, stem: str = "measurements") -> dict[str, Path]:
    """CSV + GeoJSON(속성에 측정값) + 요약 JSON 저장."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Measurement.__dataclass_fields__))
        w.writeheader()
        for m in meas:
            w.writerow(asdict(m))

    feats = []
    for m, g in zip(meas, geoms):
        feats.append({"type": "Feature", "properties": asdict(m), "geometry": mapping(g)})
    gj_path = out_dir / f"{stem}.geojson"
    gj_path.write_text(json.dumps({
        "type": "FeatureCollection", "name": stem,
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": feats,
    }, ensure_ascii=False), encoding="utf-8")

    summary: dict = {"n_labels": len(meas), "by_class": {}}
    for m in meas:
        d = summary["by_class"].setdefault(m.label, {"count": 0, "area_m2": 0.0, "volume_m3": 0.0, "weight_kg": 0.0, "weight_known": True})
        d["count"] += 1
        d["area_m2"] += m.area_m2
        d["volume_m3"] += m.volume_m3
        if m.weight_kg is None:
            d["weight_known"] = False
        else:
            d["weight_kg"] += m.weight_kg
    summary["total_area_m2"] = sum(m.area_m2 for m in meas)
    summary["total_volume_m3"] = sum(m.volume_m3 for m in meas)
    summary["total_weight_kg"] = sum(m.weight_kg or 0.0 for m in meas)
    sum_path = out_dir / f"{stem}_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "geojson": gj_path, "summary": sum_path}


def height_candidates(
    dsm_path: str | Path,
    dtm_path: str | Path,
    min_height_m: float = 0.15,
    min_area_m2: float = 0.5,
    max_area_m2: float = 500.0,
    aoi: Polygon | None = None,
    simplify_m: float | None = None,
) -> tuple[list[Polygon], list[dict], str]:
    """
    DSM - DTM 이 min_height_m 이상인 덩어리를 라벨 후보 폴리곤으로 뽑습니다.
    사람이 QGIS 에서 class 만 채우면 되도록 하는 반자동 라벨링 보조. 나무·바위·건물도 같이 잡히므로
    aoi(해빈 폴리곤 등) 로 범위를 제한하고, 큰 것은 max_area_m2 로 걸러냅니다.
    """
    with rasterio.open(safe_path(dsm_path)) as ds, rasterio.open(safe_path(dtm_path)) as dt:
        dsm = ds.read(1).astype(float)
        tr, crs, nd = ds.transform, ds.crs, ds.nodata
        from rasterio.warp import Resampling, reproject

        dtm = np.full(dsm.shape, np.nan)
        reproject(rasterio.band(dt, 1), dtm, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    valid = np.isfinite(dsm) & np.isfinite(dtm)
    if nd is not None:
        valid &= dsm != nd
    high = valid & ((dsm - dtm) >= min_height_m)
    if aoi is not None:
        aoi_mask = features.rasterize([(mapping(aoi), 1)], out_shape=dsm.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        high &= aoi_mask
    polys, props = [], []
    for geom, val in features.shapes(high.astype(np.uint8), mask=high, transform=tr):
        if val != 1:
            continue
        p = shape(geom)
        if not (min_area_m2 <= p.area <= max_area_m2):
            continue
        if simplify_m:
            p = p.simplify(simplify_m, preserve_topology=True)
        polys.append(p)
        props.append({"class": "mixed", "auto": True})
    return polys, props, str(crs)
