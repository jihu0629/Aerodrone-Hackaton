"""
좌표 보정(Geo-referencing 보정) 자동화 스크립트
=================================================

배경
----
이 위성사진(SkySatCollect)은 STAC 메타데이터에 ground_control_ratio(지상
기준점 비율)가 0.17로 나와 있어요. 이게 낮다는 건, 위성사진에 박혀있는
좌표 정보(geotransform)가 절대 위치 기준으로 어느 정도 어긋나 있을 수
있다는 뜻입니다. 그래서 사진 속 특정 지점(예: 바위 끝)의 픽셀 좌표를
경위도로 변환해보면, 구글 지도에서 같은 지점을 짚었을 때의 실제 좌표와
차이가 납니다.

이 스크립트가 하는 일
----------------------
사용자가 "사진에서는 이 픽셀인데, 실제로는 이 위경도다"라고 알려주는
기준점(GCP)을 몇 개(최소 2개, 권장 3개 이상) 입력하면:

  1) 그 지점들의 "현재(잘못된) 좌표"와 "진짜(구글 지도 기준) 좌표"를 비교해서
     둘 사이의 변환 관계(이동/회전/축척 오차)를 자동으로 계산합니다.
  2) 그 변환을 사진 전체의 좌표 정보(geotransform)에 적용해서, 보정된
     새 GeoTIFF를 만듭니다. (원본 픽셀 값은 그대로 두고, "이 픽셀이 실제로
     어디를 가리키는가"에 대한 정보만 고칩니다. 그래서 매우 빠릅니다.)
  3) 이미 뽑아둔 coastline.geojson 같은 벡터 좌표 파일이 있다면, 그것도
     같은 방식으로 한 번에 보정해줍니다.

기준점(GCP)을 구하는 방법
-------------------------
1) preview_thumbnail.png 나 crop_full_res.png 를 그림판/포토샵 등에서
   열어서, 사진에서 뚜렷하게 알아볼 수 있는 지점(바위 끝, 해변 모서리,
   건물 등)의 픽셀 좌표(왼쪽 위 기준 가로, 세로)를 확인하세요.
   - 그림판이면 마우스를 올렸을 때 하단에 좌표가 표시됩니다.
   - preview_thumbnail.png로 찾은 좌표는 축소된 이미지 기준이므로,
     아래 GCP_LIST에서 그 지점을 찾은 이미지 경로(SOURCE_IMAGE)와
     함께 정확히 맞춰서 입력해야 합니다.
2) 구글 지도에서 같은 지점을 찾아 우클릭하면 위도, 경도 숫자가 바로 뜹니다.
   (예: "37.192345, 125.987654")
3) 아래 GCP_LIST에 (픽셀 col, 픽셀 row, 실제 경도, 실제 위도) 형태로 입력하세요.

사용 전 준비
------------
    pip install rasterio numpy pyproj

사용법
------
    python fix_georeferencing.py
"""

import json
from pathlib import Path

import numpy as np

# ============================================================
# 설정
# ============================================================

# 보정할 원본 GeoTIFF 경로
TIF_PATH = r"C:\Users\jihuc\OneDrive - 인하대학교\바탕 화면\프로젝트\항공드론 해커톤\기업 배포 자료\SkySatCollect\20260817_231616_ssc1_u0002_visual.tif"

# 이 GCP를 찾을 때 기준으로 삼은 이미지의 크기 (col/row 좌표가 이 크기 기준입니다)
# - 원본 tif에서 직접 좌표를 읽었다면 원본 크기(예: 34447 x 23973)를 넣으세요.
# - preview_thumbnail.png(축소본)에서 좌표를 읽었다면 그 이미지의 실제 크기를 넣으세요.
SOURCE_IMAGE_WIDTH = 34447
SOURCE_IMAGE_HEIGHT = 23973

# 기준점(GCP) 목록: (사진 속 픽셀 col, 픽셀 row, 실제 경도(lon), 실제 위도(lat))
# 최소 2개면 이동+축척+회전(similarity) 보정이 되고, 3개 이상이면 더 안정적입니다.
# 아래는 예시이니 실제 확인한 값으로 바꿔서 쓰세요.
GCP_LIST = [
    # (col,       row,      lon,          lat)
    (12000,     8000,     125.987000,   37.192000),
    (18000,     14000,    126.010000,   37.175000),
    (9000,      16000,    125.960000,   37.165000),
]

# 보정된 결과를 저장할 경로
OUT_TIF_PATH = str(Path(TIF_PATH).with_name("20260817_231616_ssc1_u0002_visual_corrected.tif"))

# 이미 뽑아둔 해안선 GeoJSON도 같이 보정하고 싶으면 경로를 적어주세요 (없으면 None)
COASTLINE_GEOJSON_PATH = str(Path(TIF_PATH).with_name("coastline.geojson"))
OUT_GEOJSON_PATH = str(Path(TIF_PATH).with_name("coastline_corrected.geojson"))


def fit_similarity_transform(src_points: np.ndarray, dst_points: np.ndarray):
    """
    src_points(현재의 잘못된 좌표) -> dst_points(진짜 좌표)로 가는
    변환(이동 + 회전 + 확대축소, "similarity transform")을 최소제곱법으로 구합니다.

    2개 이상의 점이 있으면 계산됩니다. 점이 많을수록(3개 이상 권장) 더
    안정적인 결과가 나옵니다 (점 하나하나의 클릭 오차가 평균으로 상쇄되기 때문).

    반환값: 2x3 행렬 M, 다음처럼 사용합니다.
        [x_dst, y_dst] = M @ [x_src, y_src, 1]
    """
    n = src_points.shape[0]
    if n < 2:
        raise ValueError("기준점(GCP)이 최소 2개는 있어야 합니다.")

    # similarity transform: x' = a*x - b*y + c,  y' = b*x + a*y + d
    # 미지수 a, b, c, d 를 최소제곱으로 구합니다.
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = src_points[i]
        xp, yp = dst_points[i]
        A[2 * i]     = [x, -y, 1, 0]
        A[2 * i + 1] = [y,  x, 0, 1]
        b[2 * i]     = xp
        b[2 * i + 1] = yp

    solution, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c, d = solution

    M = np.array([[a, -bb, c], [bb, a, d]])

    # 보정 품질 확인: 각 기준점이 변환 후 dst와 얼마나 차이나는지(RMSE, 미터 단위)
    src_h = np.hstack([src_points, np.ones((n, 1))])
    predicted = (M @ src_h.T).T
    errors = np.linalg.norm(predicted - dst_points, axis=1)
    print("기준점별 보정 후 잔차(작을수록 좋음, 단위: 원본 좌표계 기준 - 보통 m):")
    for i, err in enumerate(errors):
        print(f"  GCP {i + 1}: {err:.2f}")
    print(f"  RMSE 전체: {np.sqrt((errors ** 2).mean()):.2f}")

    return M


def build_gcp_arrays(gcp_list, transform, crs):
    """
    GCP_LIST(픽셀 좌표 + 실제 위경도)를 받아서,
    - src_points: 현재 tif가 그 픽셀을 "어디"라고 잘못 알고 있는지(원본 CRS, 보통 UTM/미터)
    - dst_points: 실제 위경도를 같은 CRS로 변환한 값
    두 배열을 만들어 반환합니다. (같은 좌표계에서 비교해야 거리 계산이 정확합니다.)
    """
    import rasterio
    from pyproj import Transformer

    lonlat_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    src_points = []
    dst_points = []
    for col, row, lon, lat in gcp_list:
        # 1) 현재 tif가 이 픽셀을 뭐라고 알고 있는지 (잘못된 좌표)
        x_wrong, y_wrong = rasterio.transform.xy(transform, row, col)
        src_points.append([x_wrong, y_wrong])

        # 2) 구글 지도에서 확인한 진짜 위경도를 같은 좌표계(CRS)로 변환
        x_true, y_true = lonlat_to_crs.transform(lon, lat)
        dst_points.append([x_true, y_true])

    return np.array(src_points), np.array(dst_points)


def apply_correction_to_tif(tif_path: str, out_path: str, M: np.ndarray) -> None:
    """
    보정 행렬 M을 원본 tif의 geotransform에 합성해서, 보정된 새 GeoTIFF를 만듭니다.
    픽셀 값(사진 내용)은 전혀 건드리지 않고 좌표 정보만 고치기 때문에 매우 빠릅니다.
    """
    import rasterio
    from affine import Affine

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        old_transform = src.transform

        # old_transform: 픽셀 -> 원본(잘못된) 좌표
        # M: 원본(잘못된) 좌표 -> 진짜 좌표
        # 합성하면: 픽셀 -> 진짜 좌표
        correction_affine = Affine(M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])
        new_transform = correction_affine * old_transform

        profile.update(transform=new_transform)

        print(f"보정된 GeoTIFF 쓰는 중... -> {out_path}")
        with rasterio.open(out_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)

    print(f"완료. 원본 픽셀은 그대로, 좌표 정보만 보정되었습니다 -> {out_path}")


def apply_correction_to_geojson(geojson_path: str, out_path: str, M: np.ndarray, crs) -> None:
    """
    coastline.geojson처럼 이미 (위경도로) 뽑아둔 벡터 좌표를 같은 보정 행렬로 고칩니다.
    GeoJSON은 위경도로 저장돼 있으므로, 위경도 -> 원본 CRS -> 보정 -> 다시 위경도
    순서로 왕복 변환합니다.
    """
    from pyproj import Transformer

    if not Path(geojson_path).exists():
        print(f"{geojson_path} 가 없어서 벡터 보정은 건너뜁니다.")
        return

    lonlat_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    crs_to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    for feature in geojson.get("features", []):
        geom = feature["geometry"]
        if geom["type"] != "Polygon":
            continue
        new_rings = []
        for ring in geom["coordinates"]:
            new_ring = []
            for lon, lat in ring:
                x, y = lonlat_to_crs.transform(lon, lat)
                point = M @ np.array([x, y, 1])
                new_lon, new_lat = crs_to_lonlat.transform(point[0], point[1])
                new_ring.append([new_lon, new_lat])
            new_rings.append(new_ring)
        geom["coordinates"] = new_rings

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"보정된 GeoJSON 저장 -> {out_path}")


if __name__ == "__main__":
    import rasterio

    if not Path(TIF_PATH).exists():
        print(f"파일을 찾을 수 없습니다: {TIF_PATH}")
    else:
        with rasterio.open(TIF_PATH) as src:
            transform = src.transform
            crs = src.crs
            actual_w, actual_h = src.width, src.height

        # SOURCE_IMAGE_WIDTH/HEIGHT가 원본과 다르면, 축소된 이미지에서 좌표를 읽은
        # 것이므로 원본 픽셀 좌표로 환산해줍니다.
        scale_x = actual_w / SOURCE_IMAGE_WIDTH
        scale_y = actual_h / SOURCE_IMAGE_HEIGHT
        scaled_gcps = [
            (col * scale_x, row * scale_y, lon, lat) for col, row, lon, lat in GCP_LIST
        ]

        src_points, dst_points = build_gcp_arrays(scaled_gcps, transform, crs)
        M = fit_similarity_transform(src_points, dst_points)

        apply_correction_to_tif(TIF_PATH, OUT_TIF_PATH, M)

        if COASTLINE_GEOJSON_PATH:
            apply_correction_to_geojson(COASTLINE_GEOJSON_PATH, OUT_GEOJSON_PATH, M, crs)
