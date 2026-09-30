"""
SkySat 위성영상(GeoTIFF) 읽기 / 미리보기 생성 스크립트
====================================================

기업 배포 자료 폴더의 20260817_231616_ssc1_u0002_visual.tif 파일을 읽기 위한 코드입니다.

이 tif는 24000 x 34000 픽셀, 4밴드(Red, Green, Blue, Alpha)짜리
Cloud-Optimized GeoTIFF(COG)라서 용량이 800MB에 가깝습니다.
그냥 PIL 등으로 통째로 열면 메모리를 엄청 먹거나 느려지기 때문에,
GeoTIFF 전용 라이브러리인 rasterio로 "다운샘플링된(축소된) 미리보기"만
효율적으로 읽어오는 방식을 씁니다. COG는 내부에 저해상도 오버뷰가 이미
들어있어서, rasterio가 필요한 해상도만 골라 읽기 때문에 훨씬 빠릅니다.

사용 전 준비 (터미널/명령 프롬프트에서 한 번만 실행)
----------------------------------------------------
    pip install rasterio numpy pillow matplotlib

rasterio 설치가 잘 안 되면(윈도우에서 가끔 까다로움) 아래 대안도 참고하세요.
    conda install -c conda-forge rasterio
"""

from pathlib import Path

import numpy as np

# ============================================================
# 1) 여기 경로만 본인 환경에 맞게 수정하세요
# ============================================================
TIF_PATH = r"C:\Users\jihuc\OneDrive - 인하대학교\바탕 화면\프로젝트\항공드론 해커톤\기업 배포 자료\SkySatCollect\20260817_231616_ssc1_u0002_visual.tif"

# 미리보기 이미지를 저장할 위치 (원본과 같은 폴더에 저장됨)
PREVIEW_OUT = str(Path(TIF_PATH).with_name("preview_thumbnail.png"))

# 미리보기 이미지의 최대 가로/세로 픽셀 크기 (너무 크게 잡으면 느려집니다)
MAX_PREVIEW_SIZE = 2000

# 실행 모드: "preview"(축소 미리보기, 기본) / "full"(원본 해상도 전체) / "crop"(원본 해상도로 일부만 자르기)
MODE = "preview"

# MODE = "crop" 일 때 사용할 설정: 잘라낼 영역을 픽셀 좌표로 지정
# (원본 이미지 기준 왼쪽 위 (col_off, row_off) 부터 가로 width x 세로 height 만큼)
# 정확한 좌표를 모르면 일단 아무 값이나 넣고 실행해본 뒤, 미리보기 사진을 보면서 조정하세요.
CROP_WINDOW = dict(col_off=10000, row_off=10000, width=3000, height=3000)


def print_metadata(path: str) -> None:
    """이미지의 기본 정보(크기, 좌표계, 범위, 밴드 수 등)를 출력합니다."""
    import rasterio

    with rasterio.open(path) as src:
        print("=== GeoTIFF 메타데이터 ===")
        print(f"파일 경로     : {path}")
        print(f"이미지 크기   : 가로 {src.width} x 세로 {src.height} px")
        print(f"밴드 개수     : {src.count}  (dtype: {src.dtypes[0]})")
        print(f"좌표계(CRS)   : {src.crs}")
        print(f"지리적 범위   : {src.bounds}")
        print(f"픽셀 해상도   : {src.res} (m/px, 투영좌표계 기준)")
        print(f"오버뷰 레벨   : {[src.overviews(i) for i in range(1, src.count + 1)]}")


def make_preview(path: str, out_path: str, max_size: int = 2000) -> None:
    """
    전체 해상도를 다 읽지 않고, COG 오버뷰를 이용해 축소된 미리보기 PNG를 만듭니다.
    Red/Green/Blue 3개 밴드만 골라 일반 사진처럼 보이게 합니다.
    """
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_size
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))

        # 밴드 순서: 1=Red, 2=Green, 3=Blue, 4=Alpha (STAC 메타데이터 기준)
        band_indexes = [1, 2, 3] if src.count >= 3 else [1]

        data = src.read(
            band_indexes,
            out_shape=(len(band_indexes), out_height, out_width),
            resampling=Resampling.average,  # 축소할 때 평균값으로 부드럽게
        )

    # (bands, H, W) -> (H, W, bands) 로 변환해서 이미지로 저장
    img_array = np.transpose(data, (1, 2, 0))

    if img_array.dtype != np.uint8:
        # 혹시 8bit가 아니면 0~255 범위로 정규화
        img_array = img_array.astype(np.float32)
        img_array -= img_array.min()
        max_val = img_array.max() or 1
        img_array = (img_array / max_val * 255).astype(np.uint8)

    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array[:, :, 0], mode="L")
    else:
        img = Image.fromarray(img_array, mode="RGB")

    img.save(out_path)
    print(f"미리보기 저장 완료 -> {out_path}  (크기: {img.size[0]}x{img.size[1]})")


def export_full_resolution(path: str, out_path: str, quality: int = 90) -> None:
    """
    원본 해상도 그대로(예: 24000x34000) 사진 파일로 저장합니다.

    주의:
    - RGB 3채널 기준 압축 전 데이터만 약 2~3GB로, 메모리를 많이 씁니다.
      RAM 여유가 8GB 이상은 되어야 안전합니다.
    - 시간이 꽤 걸릴 수 있습니다 (컴퓨터 사양에 따라 수십 초~수 분).
    - JPEG로 저장해야 그나마 용량이 줄어듭니다(PNG는 훨씬 커짐).
      그래도 결과 파일이 수백 MB가 될 수 있습니다.
    - PIL은 기본적으로 너무 큰 이미지를 "폭탄"으로 간주해 막아두는데,
      우리 데이터는 정상 데이터이므로 아래에서 그 제한을 해제합니다.
    """
    import rasterio
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # 초대형 이미지 경고/차단 해제 (우리 데이터는 안전함)

    with rasterio.open(path) as src:
        print(f"원본 해상도로 읽는 중... ({src.width} x {src.height} px, 시간이 걸릴 수 있습니다)")
        band_indexes = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(band_indexes)  # 전체 해상도로 읽기 (다운샘플링 없음)

    img_array = np.transpose(data, (1, 2, 0))
    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array[:, :, 0], mode="L")
    else:
        img = Image.fromarray(img_array, mode="RGB")

    save_kwargs = {}
    if out_path.lower().endswith((".jpg", ".jpeg")):
        save_kwargs = dict(quality=quality)

    img.save(out_path, **save_kwargs)
    print(f"원본 해상도 저장 완료 -> {out_path}  (크기: {img.size[0]}x{img.size[1]})")


def export_crop_full_res(path: str, out_path: str, col_off: int, row_off: int,
                          width: int, height: int) -> None:
    """
    이미지 전체가 아니라 지정한 좌표(col_off, row_off)부터
    width x height 픽셀만큼만 '원본 해상도 그대로' 잘라서 저장합니다.

    전체를 다 읽지 않고 필요한 부분만 읽기 때문에 export_full_resolution보다
    훨씬 빠르고 메모리도 적게 씁니다. 특정 지점(예: 해변)을 자세히 보고 싶을 때 추천합니다.
    """
    import rasterio
    from rasterio.windows import Window
    from PIL import Image

    with rasterio.open(path) as src:
        window = Window(col_off, row_off, width, height)
        band_indexes = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(band_indexes, window=window)

    img_array = np.transpose(data, (1, 2, 0))
    if img_array.shape[2] == 1:
        img = Image.fromarray(img_array[:, :, 0], mode="L")
    else:
        img = Image.fromarray(img_array, mode="RGB")

    img.save(out_path)
    print(f"부분 크롭(원본 해상도) 저장 완료 -> {out_path}  (크기: {img.size[0]}x{img.size[1]})")


def make_preview_no_rasterio(path: str, out_path: str, max_size: int = 1500) -> None:
    """
    rasterio 설치가 안 될 때 쓰는 대안 (tifffile 사용).
    COG 오버뷰를 활용하지 못해 rasterio 방식보다 느리고 메모리를 더 씁니다.
    큰 파일에서는 시간이 꽤 걸릴 수 있습니다.
    """
    import tifffile
    from PIL import Image

    with tifffile.TiffFile(path) as tif:
        # 가장 작은 서브 이미지(오버뷰)가 있으면 그것부터 사용 시도
        page = min(tif.pages, key=lambda p: p.shape[0] * p.shape[1])
        arr = page.asarray()

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]

    img = Image.fromarray(arr)
    img.thumbnail((max_size, max_size))
    img.save(out_path)
    print(f"미리보기 저장 완료(대안 방식) -> {out_path}")


if __name__ == "__main__":
    if not Path(TIF_PATH).exists():
        print(f"파일을 찾을 수 없습니다: {TIF_PATH}")
        print("TIF_PATH 변수를 실제 파일 경로로 수정해주세요.")
    else:
        try:
            print_metadata(TIF_PATH)

            if MODE == "preview":
                make_preview(TIF_PATH, PREVIEW_OUT, MAX_PREVIEW_SIZE)

            elif MODE == "full":
                full_out = str(Path(TIF_PATH).with_name("full_resolution.jpg"))
                export_full_resolution(TIF_PATH, full_out)

            elif MODE == "crop":
                crop_out = str(Path(TIF_PATH).with_name("crop_full_res.png"))
                export_crop_full_res(TIF_PATH, crop_out, **CROP_WINDOW)

            else:
                print(f"알 수 없는 MODE 값입니다: {MODE} (preview / full / crop 중 하나로 설정하세요)")

        except ImportError:
            print("rasterio가 설치되어 있지 않아 대안 방식으로 시도합니다...")
            print("(정확도/속도를 위해서는 'pip install rasterio' 권장)")
            make_preview_no_rasterio(TIF_PATH, PREVIEW_OUT)
