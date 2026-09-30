# 드론대장 붕붕이 — 시계열 위성·드론 영상 해안선 변화탐지

2026 항공·드론 산업 수요 기반 해커톤 과제4. 대상지 굴업도(인천 옹진군), 기업 배포 SkySat 0.5 m 영상 + Sentinel-2 10 m 시계열.

## 폴더 구조

```
coastcd/                 파이썬 패키지 (영문 경로, import coastcd 만 하면 GDAL 환경변수 자동 설정)
  config.py              한글 경로 우회(safe_path), 기본 경로, 상수
  raster_io.py           COG 오버뷰·윈도우·타일 읽기, 알파 마스크, 섬 영역 자동 탐색, GeoTIFF 쓰기
  water_mask.py          RGB 밝기+질감 수륙분할(타일 단위), Sentinel-2 NDWI
  coastline.py           마스크 -> 폴리곤 -> 해안선 GeoJSON (EPSG:32651 / 4326)
  register.py            SIFT/ORB + MAGSAC++ 정합, 위상상관, 합성 변환 벤치마크, LoFTR(선택)
scripts/
  01_find_island.py      원본 tif 에서 섬 bbox 자동 탐색 -> island_crop.tif (원본 해상도, 좌표계 유지)
  02_extract_coastline.py 원본 해상도 육지 마스크 + 해안선 GeoJSON (타일 처리)
  03_bench_register.py   정합 정확도 벤치마크 (2시기 없이 픽셀 단위 오차 증명)
  04_s2_ndwi_mask.py     Sentinel-2 NDWI 물 마스크 + SkySat 과의 전역 이동량(m) 추정
tests/test_pipeline.py   합성 섬 영상으로 전 과정 자동 검증 (실제 tif 없이 실행 가능)
항공드론 해커톤/         제안서·선행조사·기업 배포 메타데이터·초기 스크립트 (그대로 보존)
outputs/                 결과물 (git 제외)
```

## 빠른 시작 (Windows, VS Code)

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

set COASTCD_SKYSAT=C:\hackathon\20260817_231616_ssc1_u0002_visual.tif
python scripts\01_find_island.py
python scripts\02_extract_coastline.py
python scripts\03_bench_register.py --trials 10
python scripts\04_s2_ndwi_mask.py --s2 S2_GYD\S2_GYD_2025-08-12.tif --skysat outputs\island_crop.tif
pytest tests -q
```

`--tif`, `--out` 옵션으로 경로를 직접 줄 수도 있습니다. 결과는 `outputs/` 에 쌓입니다.

### 한글 경로 문제

- `coastcd.config.safe_path` 가 Windows 에서 한글 경로를 8.3 짧은 경로로 바꿔 GDAL 에 넘깁니다. 볼륨에서 8.3 이름이 꺼져 있으면 변환이 안 됩니다 (확인 필요). 그때는 기존처럼 `C:\hackathon` 으로 복사하세요.
- `import coastcd` 시 `GDAL_FILENAME_IS_UTF8=YES`, `PYTHONUTF8=1` 을 자동 설정합니다.
- OneDrive 폴더의 파일은 "온라인 전용" 이면 열리지 않습니다. 원본 tif 는 OneDrive 밖(`C:\hackathon`)에 두는 것을 권합니다.

## 각 단계가 하는 일 (고등학생용 설명)

1. **섬 찾기(01)**: 800 MB 사진은 95% 가 바다입니다. 축소본에서 섬을 찾아 그 주변만 원본 화질로 잘라 둡니다. 이후 모든 작업이 수십 배 빨라집니다.
2. **해안선 뽑기(02)**: "밝으면 육지, 울퉁불퉁하면 육지, 둘 다 아니면 물" 규칙을 원본 화질에 타일 단위로 적용합니다. 규칙의 기준값은 축소본에서 한 번만 정해 모든 타일에 똑같이 씁니다. 여백은 사진의 4번째 채널(알파)로 정확히 걸러냅니다.
3. **정합 검증(03)**: 사진 한 장을 일부러 밀고 돌리고 어둡게 만든 "가짜 다른 날 사진" 을 만들어 다시 맞춰 봅니다. 정답을 알기 때문에 "우리 정합 오차는 0.1 픽셀" 처럼 숫자로 말할 수 있습니다.
4. **Sentinel-2(04)**: 적외선이 있는 Sentinel-2 로 물을 바로 구분하고, SkySat 과 얼마나 어긋나 있는지 미터 단위로 잽니다. 20배 해상도 차이 때문에 반드시 같은 격자(10 m)로 맞춘 뒤 비교합니다.

## 합성 데이터 검증 결과 (tests, 6000x4000 px 가짜 섬)

| 항목 | 값 |
|---|---|
| 육지 마스크 IoU | > 0.9 |
| 정합 전 오차 | 약 26 px |
| 정합 후 RMSE | 0.07 px (최악 0.13 px) |
| 30 m 를 밀어 둔 Sentinel-2 에 대한 위상상관 추정 | -28.7 m |

실제 SkySat 영상에서는 그림자·젖은 모래·파도 거품 때문에 마스크 정확도가 이보다 떨어집니다. 이 숫자는 "코드가 맞게 동작한다" 는 확인이지 실제 성능이 아닙니다.

## 기존 스크립트(항공드론 해커톤/) 검토 메모

- `read_skysat_tif.py`: 미리보기 2000 px 는 약 17 m/px 로 Sentinel-2 보다 거칩니다. 베이스라인 해안선이 이 축소본에서 나온 것이므로 해빈(폭 40 m) 이 2 px 입니다. `full` 모드는 2.5 GB 를 메모리에 올리니 쓰지 마세요. 01 단계의 크롭으로 대체됩니다.
- `extract_coastline.py`: 검은 픽셀을 여백으로 추정하는 방식은 그림자를 여백으로 오인할 수 있어 알파 밴드로 바꿨습니다. 임계값 논리(밝기 OR 질감)는 그대로 `water_mask.py` 에 옮겼습니다.
- `fix_georeferencing.py`: 구글지도 좌표는 국내에서 수 m 오프셋이 있을 수 있습니다 (확인 필요). 800 MB 를 다시 쓰는 대신 `rasterio.open(path, "r+")` 로 transform 만 고치면 수 초에 끝납니다. 변화탐지에서 중요한 것은 절대 위치가 아니라 시기 간 상대 정합이며, 그 부분은 `register.py` 가 담당합니다.
- `s2_download.py`: 구조 좋습니다. 결과가 나오면 04 단계에 바로 넣을 수 있습니다.
- 원본 크기는 34447(가로) x 23973(세로) 입니다. 메모의 "24000 x 34000" 은 가로세로가 바뀐 표기입니다.

## 다음 단계

- 실제 SkySat 으로 01~03 실행 후 `outputs/coastline_overlay.png` 확인, 임계값·질감 창 조정
- 두 시기 마스크 비교(침식/퇴적 폴리곤, transect 변위) 모듈 `coastcd/change.py`
- 조위 무관 해안선 지표(습윤선·식생선) 로 전환 여부 결정
- Streamlit 대시보드에서 `coastline_4326.geojson` Swipe 표시
