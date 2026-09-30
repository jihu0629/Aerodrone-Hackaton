"""
해안선(육지 vs 바다) 자동 추출 - 1단계 베이스라인 알고리즘
=========================================================

배경
----
이 위성사진은 RGB 3채널짜리라서(제약: NIR 밴드 없음), 흔히 쓰는 위성 해안선
추출 공식인 NDWI(Normalized Difference Water Index)를 못 씁니다. NDWI는
적외선(NIR)과 초록(Green) 밴드의 반사율 차이로 물을 구분하는 방식이라
NIR이 없으면 계산 자체가 안 되거든요.

그래서 이 스크립트는 RGB만으로 되는 "밝기/색 기반 이진분류(thresholding)"
방식을 씁니다. 원리는 단순합니다.

    바다/물은 대체로 어둡고 파란 쪽에 가깝다
    육지(모래, 바위, 풀)는 대체로 더 밝고 색이 다양하다

    -> 이 밝기 차이를 자동으로 딱 갈라주는 경계값(threshold)을 컴퓨터가
       스스로 찾게 하고(Otsu's method), 그 경계값보다 어두운 픽셀은 "물",
       밝은 픽셀은 "육지"로 분류합니다.
    -> 분류한 다음, 물 덩어리와 육지 덩어리의 "경계선"을 따라가면 그게
       곧 해안선(coastline)이 됩니다.

한계 (꼭 알아두세요)
--------------------
- 이 방법은 "빠르게 돌아가는 베이스라인"이지 최종 정답이 아닙니다.
- 그림자(저태양고도), 젖은 모래(물처럼 어둡게 보임), 파도 거품(육지처럼
  밝게 보임) 때문에 오탐이 생길 수 있습니다. 배포자료 분석에서도 이미
  "저태양고도 27도라 그림자 위험 있음"이라고 나와 있었죠.
- 나중에 시간이 되면 이 자리를 CAID 데이터셋으로 사전학습된 딥러닝
  분할 모델(U-Net 계열)로 교체하면 훨씬 정확해집니다. 지금은 데모/베이스라인
  용도로 먼저 "파이프라인이 끝까지 돌아가는 것"을 확인하는 게 목적입니다.

중요 - nodata(검은 여백) 처리
------------------------------
SkySat 영상은 위성이 비스듬한 방향으로 촬영한 "띠(strip)" 모양이라, 그걸
사각형 파일에 담으면 스트립 바깥 모서리 부분이 새까만 여백(nodata)으로
채워집니다. 이 검은 여백도 "어둡다"는 점에서 진짜 바다랑 똑같이 생겼기
때문에, 아무 처리 없이 밝기 기준으로만 나누면 검은 여백 + 바다가 한 덩어리로
묶여버리고, 그 덩어리의 "바깥 테두리"(=사진 자체의 테두리)가 마치 해안선인
것처럼 잘못 감지됩니다. 그래서 이 코드는 두 가지 조치를 취합니다.

  1) 검은 여백(nodata) 픽셀을 먼저 찾아서 분석 대상에서 제외합니다.
     (Otsu 임계값도 이 여백을 뺀 "진짜 사진 부분"만 갖고 계산합니다.)
  2) "물의 바깥 경계선"이 아니라 "육지(섬) 덩어리의 경계선"을 찾습니다.
     바다는 사진 테두리에 붙어있어서 테두리 오탐에 계속 휘말리지만,
     섬은 사방이 바다로 완전히 둘러싸인 독립된 덩어리라 테두리 문제에서
     자유롭기 때문입니다.

사용 전 준비
------------
    pip install opencv-python numpy pillow rasterio shapely

사용법
------
1) INPUT_IMAGE 에 분석할 이미지 경로를 넣으세요.
   - preview_thumbnail.png (축소 미리보기) 로 먼저 테스트해보고,
   - 잘 되면 원본 tif나 crop_full_res.png로 정밀하게 다시 돌리세요.
2) python extract_coastline.py 실행
3) 결과물:
   - coastline_overlay.png : 원본 위에 빨간 선으로 감지된 해안선을 표시
   - coastline_mask.png    : 육지(흰색) / 물+검은 여백(검정) 이진 마스크
   - coastline.geojson     : (원본이 좌표계를 가진 GeoTIFF인 경우만) 실제
                             위경도 좌표로 변환된 해안선 벡터
"""

from pathlib import Path

import numpy as np
import cv2

# ============================================================
# 설정
# ============================================================

# 분석할 이미지: 지금까지 만든 preview_thumbnail.png / crop_full_res.png /
# 또는 원본 tif 경로를 그대로 넣어도 됩니다.
INPUT_IMAGE = r"C:\Users\jihuc\OneDrive - 인하대학교\바탕 화면\프로젝트\항공드론 해커톤\기업 배포 자료\SkySatCollect\preview_thumbnail.png"

# 너무 작은 노이즈 덩어리는 무시하기 위한 최소 면적(픽셀 개수)
MIN_CONTOUR_AREA = 500

# 이 값보다 세 채널(R+G+B) 합이 작으면 "검은 여백(nodata)"으로 간주합니다.
# 완전히 0,0,0이 아니어도(압축 과정에서 살짝 값이 생길 수 있음) 걸러내도록
# 여유를 좀 둡니다. 너무 크게 잡으면 진짜 어두운 바다까지 nodata로
# 오인할 수 있으니, 결과가 이상하면 이 값을 조정해보세요.
NODATA_SUM_THRESHOLD = 15

# nodata 경계 바로 안쪽은 압축으로 인한 애매한 색(안티앨리어싱)이 섞여있을
# 수 있어서, 여백 마스크를 이만큼(픽셀) 더 두껍게 만들어 안전하게 제외합니다.
NODATA_MARGIN_PX = 5


def load_image_as_rgb(path: str):
    """
    일반 사진 파일(png/jpg)이면 OpenCV/PIL로,
    GeoTIFF(.tif)면 rasterio로 읽습니다. GeoTIFF일 경우 좌표계 정보(transform,
    crs)도 같이 반환해서 나중에 픽셀 좌표 -> 실제 위경도 변환에 씁니다.
    """
    ext = Path(path).suffix.lower()

    if ext in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(path) as src:
            band_indexes = [1, 2, 3] if src.count >= 3 else [1]
            data = src.read(band_indexes)
            transform = src.transform
            crs = src.crs
        rgb = np.transpose(data, (1, 2, 0))
        return rgb, transform, crs
    else:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return np.array(img), None, None


def compute_valid_mask(rgb: np.ndarray) -> np.ndarray:
    """
    '진짜 사진이 있는 영역(1)' vs '검은 여백/nodata(0)'을 구분하는 마스크입니다.
    R+G+B 합이 아주 작은(거의 새까만) 픽셀을 여백으로 간주하고,
    그 여백 주변까지 약간 더 넓게 깎아내서(erode) 경계 부분의 애매한
    색까지 안전하게 제외합니다.
    """
    channel_sum = rgb.astype(np.int32).sum(axis=2)
    valid = (channel_sum > NODATA_SUM_THRESHOLD).astype(np.uint8) * 255

    if NODATA_MARGIN_PX > 0:
        kernel = np.ones((NODATA_MARGIN_PX * 2 + 1, NODATA_MARGIN_PX * 2 + 1), np.uint8)
        valid = cv2.erode(valid, kernel)

    valid_ratio = valid.mean() / 255
    print(f"유효 영역(검은 여백 제외) 비율: {valid_ratio * 100:.1f}%")
    return valid


def compute_land_water_masks(rgb: np.ndarray, valid_mask: np.ndarray):
    """
    RGB 이미지에서 '육지(1) vs 물(0)' 이진 마스크를 계산합니다.
    valid_mask로 표시된 "진짜 사진 영역"의 픽셀만 갖고 밝기 경계값을
    계산해서, 검은 여백 때문에 임계값이 엉뚱하게 계산되는 걸 막습니다.

    단계:
      1) 그레이스케일(밝기)로 변환
      2) 노이즈를 줄이기 위해 살짝 블러 처리
      3) 유효 영역 픽셀만 뽑아서 Otsu's method로 임계값 자동 계산
      4) 그 값보다 밝은 픽셀 = 육지 (단, 유효 영역 안에서만)
      5) 작은 노이즈 덩어리 제거 (morphology: opening/closing)
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    valid_pixels = blurred[valid_mask > 0]
    if valid_pixels.size == 0:
        raise ValueError("유효한(검은 여백이 아닌) 픽셀이 하나도 없습니다. NODATA_SUM_THRESHOLD를 낮춰보세요.")

    # Otsu 임계값을 "유효 영역 픽셀들의 밝기 분포"만으로 계산
    # (cv2.threshold는 이미지 전체가 필요하므로, 1차원 배열을 1행짜리 이미지로 넣어줍니다)
    threshold_value, _ = cv2.threshold(
        valid_pixels.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"자동 계산된 밝기 임계값(Otsu, 여백 제외): {threshold_value:.1f}")

    land_mask = (blurred > threshold_value).astype(np.uint8) * 255
    land_mask = cv2.bitwise_and(land_mask, valid_mask)  # 여백 영역은 강제로 "육지 아님" 처리

    water_mask = (blurred <= threshold_value).astype(np.uint8) * 255
    water_mask = cv2.bitwise_and(water_mask, valid_mask)  # 시각화/참고용 (여백은 제외)

    # 노이즈 제거: 작은 구멍 메우기 + 작은 점 없애기
    kernel = np.ones((5, 5), np.uint8)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, kernel)

    return land_mask, water_mask


def extract_coastline_contours(land_mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA):
    """
    육지(섬) 덩어리의 윤곽선을 찾습니다. 이게 곧 해안선입니다.
    (물 덩어리는 사진 테두리에 붙어있어서 테두리를 해안선으로 착각하기
    쉽지만, 육지는 사방이 물로 둘러싸인 독립된 덩어리라 이 문제가 없습니다.)
    """
    contours, _ = cv2.findContours(
        land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 너무 작은(파도 거품 등 노이즈성) 윤곽선은 제거
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    print(f"감지된 해안선(섬) 개수: {len(contours)}개 (면적 {min_area}px 이상만)")
    return contours


def save_overlay(rgb: np.ndarray, contours, out_path: str) -> None:
    """원본 사진 위에 감지된 해안선을 빨간 선으로 그려서 저장합니다."""
    overlay = rgb.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)
    from PIL import Image

    Image.fromarray(overlay).save(out_path)
    print(f"오버레이 이미지 저장 -> {out_path}")


def save_mask(land_mask: np.ndarray, out_path: str) -> None:
    """육지(흰색) / 물+여백(검정) 이진 마스크를 저장합니다."""
    from PIL import Image

    Image.fromarray(land_mask).save(out_path)
    print(f"이진 마스크 저장 -> {out_path}")


def save_geojson(contours, transform, crs, out_path: str) -> None:
    """
    원본이 좌표계 정보를 가진 GeoTIFF일 때만 실행됩니다.
    픽셀 좌표(행/열)를 실제 지리좌표(위경도)로 변환해서 GeoJSON으로 저장합니다.
    이렇게 저장해두면 나중에 QGIS에서 열어보거나, 다른 시기 해안선과
    겹쳐서 변화량을 계산할 때 그대로 쓸 수 있습니다.
    """
    if transform is None:
        print("좌표계 정보가 없는 이미지(png/jpg)라 GeoJSON은 건너뜁니다.")
        return

    import json
    import rasterio
    from pyproj import Transformer

    # 원본 좌표계(보통 UTM, 미터 단위) -> 위경도(EPSG:4326)로 변환 준비
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    features = []
    for contour in contours:
        coords = []
        for point in contour.reshape(-1, 2):  # (x=col, y=row)
            col, row = point
            x, y = rasterio.transform.xy(transform, row, col)  # 픽셀 -> 원본 좌표계
            lon, lat = to_lonlat.transform(x, y)  # 원본 좌표계 -> 위경도
            coords.append([lon, lat])
        coords.append(coords[0])  # 폴리곤을 닫아줌

        features.append({
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"GeoJSON 저장 -> {out_path} (실제 위경도 좌표 {len(features)}개 폴리곤)")


if __name__ == "__main__":
    if not Path(INPUT_IMAGE).exists():
        print(f"파일을 찾을 수 없습니다: {INPUT_IMAGE}")
        print("INPUT_IMAGE 변수를 실제 파일 경로로 수정해주세요.")
    else:
        out_dir = Path(INPUT_IMAGE).parent

        rgb, transform, crs = load_image_as_rgb(INPUT_IMAGE)
        print(f"이미지 로드 완료: {rgb.shape[1]} x {rgb.shape[0]} px")

        valid_mask = compute_valid_mask(rgb)
        land_mask, water_mask = compute_land_water_masks(rgb, valid_mask)
        contours = extract_coastline_contours(land_mask)

        save_mask(land_mask, str(out_dir / "coastline_mask.png"))
        save_overlay(rgb, contours, str(out_dir / "coastline_overlay.png"))

        if transform is not None:
            try:
                save_geojson(contours, transform, crs, str(out_dir / "coastline.geojson"))
            except ImportError:
                print("pyproj가 없어 GeoJSON 변환을 건너뜁니다. 'pip install pyproj'로 설치 가능합니다.")
