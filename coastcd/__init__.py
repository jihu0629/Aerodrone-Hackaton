"""
coastcd — 시계열 위성·드론 영상 해안선 변화탐지 파이프라인 (드론대장 붕붕이).

모듈
----
config      한글 경로 우회, GDAL 환경변수, 기본 경로
raster_io   COG 오버뷰·윈도우·타일 읽기, 알파(유효영역) 마스크, 섬 영역 자동 탐색
water_mask  RGB 밝기+질감 베이스라인 수륙분할 (타일 단위), Sentinel-2 NDWI 마스크
coastline   마스크 -> 폴리곤/해안선 GeoJSON (EPSG:32651, EPSG:4326)
register    특징점 정합(SIFT/ORB + MAGSAC++), 위상상관 전역 이동, 합성 변환 벤치마크

`import coastcd` 만 해도 환경변수(GDAL_FILENAME_IS_UTF8 등)가 설정됩니다.
"""

from .config import setup_env as _setup_env

_setup_env()

__all__ = ["config", "raster_io", "water_mask", "coastline", "register"]
__version__ = "0.1.0"
