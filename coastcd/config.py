"""
환경 설정: 한글 경로 우회, GDAL 환경변수, 기본 경로.

왜 필요한가
-----------
rasterio(GDAL)는 Windows에서 '인하대학교' 같은 한글 경로를 열 때
UnicodeDecodeError 를 내는 경우가 있습니다. 원인은 보통 둘 중 하나입니다.

  1) 콘솔/파이썬 기본 인코딩이 cp949 라서 GDAL 이 경로·오류 메시지를 UTF-8 로
     주고받다가 깨짐  -> PYTHONUTF8=1, GDAL_FILENAME_IS_UTF8=YES 로 해결되는 경우가 많음
  2) OneDrive 폴더의 "온라인 전용(파일 주문형)" 파일 -> 열 때 파일이 없다고 나옴.
     이 경우는 파일을 우클릭해 "항상 이 장치에 유지" 로 바꿔야 합니다.

가장 확실한 우회는 Windows 8.3 짧은 경로(예: C:\\Users\\jihuc\\ONEDRI~1\\...)를 쓰는 것입니다.
짧은 경로는 ASCII 만 들어가서 인코딩 문제가 원천적으로 사라집니다.
단, 볼륨에서 8.3 이름 생성이 꺼져 있으면 짧은 경로가 없을 수 있습니다 (확인 필요).
그 경우 기존 방식(C:\\hackathon 으로 복사)이 최후의 수단입니다.

이 모듈은 `import coastcd` 시점에 자동으로 `setup_env()` 를 호출합니다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ------------------------------------------------------------------ 환경변수

_GDAL_ENV_DEFAULTS = {
    # 한글 경로를 UTF-8 로 해석
    "GDAL_FILENAME_IS_UTF8": "YES",
    # COG 를 열 때 같은 폴더의 모든 파일을 훑지 않음 (OneDrive 폴더에서 특히 느림)
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # 블록 캐시 (MB). 800 MB tif 를 타일 단위로 읽을 때 재읽기를 줄임
    "GDAL_CACHEMAX": "512",
    # 파이썬 자체 인코딩
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def setup_env() -> None:
    """GDAL/파이썬 환경변수를 (아직 없을 때만) 설정합니다. rasterio import 전에 불러야 합니다."""
    for key, value in _GDAL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    # Windows 콘솔 출력이 cp949 로 깨지는 것을 방지
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


# ------------------------------------------------------------------ 경로

def _windows_short_path(path: str) -> str | None:
    """Windows 8.3 짧은 경로를 돌려줍니다. 실패하면 None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf_len = get_short(path, None, 0)
        if buf_len == 0:
            return None
        buf = ctypes.create_unicode_buffer(buf_len)
        if get_short(path, buf, buf_len) == 0:
            return None
        return buf.value
    except Exception:  # noqa: BLE001 - 어떤 이유로든 실패하면 None
        return None


def safe_path(path: str | os.PathLike) -> str:
    """
    rasterio/GDAL 에 넘길 수 있는 안전한 경로 문자열을 돌려줍니다.

    - 경로에 비ASCII 문자(한글 등)가 없으면 그대로 반환
    - Windows 에서 한글이 있으면 8.3 짧은 경로로 변환 시도
    - 변환이 안 되면 원래 경로를 반환 (이때는 GDAL_FILENAME_IS_UTF8 에 기대야 함)
    """
    text = os.fspath(path)
    if text.isascii():
        return text
    short = _windows_short_path(text)
    if short and short.isascii():
        return short
    return text


# ------------------------------------------------------------------ 기본 경로

REPO_ROOT = Path(__file__).resolve().parent.parent

# 기업 배포 SkySat 원본. 환경변수 COASTCD_SKYSAT 로 덮어쓸 수 있습니다.
DEFAULT_SKYSAT_TIF = Path(
    os.environ.get(
        "COASTCD_SKYSAT",
        REPO_ROOT
        / "항공드론 해커톤"
        / "기업 배포 자료"
        / "SkySatCollect"
        / "20260817_231616_ssc1_u0002_visual.tif",
    )
)

# 결과물 저장 폴더 (git 에서 제외됨)
DEFAULT_OUT_DIR = Path(os.environ.get("COASTCD_OUT", REPO_ROOT / "outputs"))

# SkySat 메타데이터에서 확인한 값들
SKYSAT_PIXEL_M = 0.5        # 리샘플 픽셀 크기 (m)
SKYSAT_NATIVE_GSD_M = 0.78  # 실제 GSD (m). 정확도 주장은 이 값을 기준으로
SENTINEL2_PIXEL_M = 10.0
WORKING_CRS = "EPSG:32651"  # UTM 51N. SkySat, Sentinel-2 모두 이 좌표계로 옴
