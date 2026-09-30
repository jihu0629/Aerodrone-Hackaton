"""
육지 마스크 -> 폴리곤 -> 해안선(GeoJSON).

- 마스크의 픽셀 경계를 그대로 벡터화하면 계단 모양이 되므로 simplify_m 만큼 단순화합니다.
  (0.5 m 픽셀이면 simplify 1.0 m 정도가 무난. 확인 필요: 해안선 정확도 목표에 따라 조정)
- 결과는 작업 좌표계(EPSG:32651, 미터) 와 웹지도용 EPSG:4326 두 벌로 저장합니다.
  변화량 계산은 반드시 미터 좌표계에서 하세요.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import LineString, MultiLineString, Polygon, mapping, shape
from shapely.ops import transform as shp_transform


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    min_area_m2: float = 500.0,
    simplify_m: float = 1.0,
) -> list[Polygon]:
    """육지 마스크(255 = 육지) 를 지리좌표 폴리곤 목록으로 바꿉니다 (큰 순서)."""
    binary = (mask > 0).astype(np.uint8)
    polys: list[Polygon] = []
    for geom, value in features.shapes(binary, mask=binary.astype(bool), transform=transform, connectivity=8):
        if value != 1:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area_m2:
            continue
        if simplify_m > 0:
            poly = poly.simplify(simplify_m, preserve_topology=True)
        if poly.geom_type == "MultiPolygon":
            polys.extend(p for p in poly.geoms if p.area >= min_area_m2)
        else:
            polys.append(poly)
    polys.sort(key=lambda p: p.area, reverse=True)
    return polys


def polygons_to_coastlines(polys: list[Polygon]) -> list[LineString]:
    """폴리곤 외곽선(해안선) 만 추출. 내부 구멍(호수 등) 은 제외."""
    return [LineString(p.exterior.coords) for p in polys]


def _to_wgs84(geoms, crs):
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return [shp_transform(tr.transform, g) for g in geoms]


def save_geojson(geoms, crs, out_path: str | Path, properties: list[dict] | None = None, wgs84: bool = False) -> Path:
    """
    shapely 도형 목록을 GeoJSON 으로 저장합니다.
    wgs84=True 면 EPSG:4326 으로 변환해서 저장 (Folium/웹지도용).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if wgs84:
        geoms = _to_wgs84(geoms, crs)
        crs_name = "EPSG:4326"
    else:
        crs_name = str(crs)
    feats = []
    for i, g in enumerate(geoms):
        props = {"id": i}
        if properties and i < len(properties):
            props.update(properties[i])
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
    fc = {
        "type": "FeatureCollection",
        "name": out_path.stem,
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": feats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return out_path


def draw_polygons(rgb: np.ndarray, polys, transform: Affine, color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    """지리좌표 폴리곤을 픽셀로 되돌려 rgb 위에 그립니다 (오버레이 PNG 용)."""
    import cv2

    inv = ~transform
    out = rgb.copy()
    for p in polys:
        rings = [p.exterior] + list(p.interiors)
        for ring in rings:
            pts = np.array([inv * (x, y) for x, y in ring.coords], dtype=np.float32)
            cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)], True, color, thickness)
    return out


def total_length_m(lines) -> float:
    return float(sum(l.length for l in lines))
