# 드론대장 붕붕이 — 시계열 위성·드론 영상 해안선 변화탐지

2026 항공·드론 산업 수요 기반 해커톤 과제4. 대상지 굴업도(인천 옹진군), 기업 배포 SkySat 0.5 m 영상 + Sentinel-2 10 m 시계열.

## 폴더 구조

```
coastcd/                 파이썬 패키지 (영문 경로, import coastcd 만 하면 GDAL 환경변수 자동 설정)
  config.py              한글 경로 우회(safe_path), 기본 경로, 상수
  raster_io.py           COG 오버뷰·윈도우·타일 읽기, 알파 마스크, 섬 영역 자동 탐색, GeoTIFF 쓰기
  water_mask.py          RGB 밝기+질감 수륙분할(타일 단위), Sentinel-2 NDWI
  coastline.py           마스크 -> 폴리곤 -> 해안선 GeoJSON (영상 UTM / 4326)
  register.py            SIFT/ORB + MAGSAC++ 정합, 위상상관, 합성 변환 벤치마크, LoFTR(선택)
  dji.py                 DJI 영상(.MP4+.SRT) 파싱, 프레임 추출(간격·블러 필터), GPS EXIF 쓰기
  odm.py                 OpenDroneMap 도커 명령 생성·실행, 결과(DSM/DTM/정사영상/점군/메시) 수집
  volume.py              라벨 폴리곤 + DSM/DTM -> 면적·부피·무게, 높이 기반 라벨 후보 생성
scripts/
  01_find_island.py      원본 tif 에서 섬 bbox 자동 탐색 -> island_crop.tif (원본 해상도, 좌표계 유지)
  02_extract_coastline.py 원본 해상도 육지 마스크 + 해안선 GeoJSON (타일 처리)
  03_bench_register.py   정합 정확도 벤치마크 (2시기 없이 픽셀 단위 오차 증명)
  04_s2_ndwi_mask.py     Sentinel-2 NDWI 물 마스크 + SkySat 과의 전역 이동량(m) 추정
  05_dji_video_to_frames.py  DJI 영상 -> GPS EXIF 붙은 프레임 JPEG
  06_build_3d_map.py     프레임 -> 3D 지도 (ODM 도커: 점군, 메시, DSM/DTM, 정사영상, EPSG:32652)
  07_label_measure.py    라벨 폴리곤(QGIS) + DSM/DTM -> 쓰레기 면적·부피·무게 표/GeoJSON/오버레이
tests/test_pipeline.py   합성 섬 영상으로 전 과정 자동 검증 (실제 tif 없이 실행 가능)
tests/test_drone3d.py    합성 영상·SRT·DSM 으로 05~07 검증 (DJI 영상·도커 없이 실행 가능)
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

## 각 단계가 하는 일

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

## 실제 SkySat 영상 1차 결과 (2026-09-30)

| 항목 | 값 |
|---|---|
| 좌표계 | EPSG:32652 (UTM 52N). 제안서의 32651 가정은 틀렸음. Sentinel-2 도 32652 로 받도록 수정 |
| 오버뷰 | 3, 9, 27, 81 배 |
| 섬 크롭 | 8847 x 7004 px (4.42 x 3.50 km) |
| 밝기 / 질감 임계값 | 90 / 5.49 |
| 최대 섬 면적 | 1.627 km² (위키 기준 굴업도 1.71 km², 확인 필요) |
| 정합 벤치마크 (easy) | 정합 후 RMSE 0.013 px, inlier RMSE 0.32 px |

1차 결과에서 고친 것: 섬 안 그늘진 숲·풀밭이 물로 빠지는 구멍 -> 바다와 안 이어진 물은 육지로 채움(`fill_enclosed_water`),
바다 위 배 항적이 섬으로 잡힘 -> 가늘고 긴 객체 제거(`remove_thin_objects`), 벤치마크에 `--hard` 모드 추가.

## 드론 영상 -> 3D 지도 -> 쓰레기 양 측정 (05~07)

DJI 기체로 찍은 영상 한 편이 입력입니다. DJI Fly/Pilot 앱에서 **카메라 설정 > 고급 촬영 설정 > 영상 자막(Video Caption)** 을 켜면
프레임별 위도·경도·고도 자막이 기록됩니다. Mavic/Air/Matrice 는 영상 옆에 같은 이름의 `.SRT` 파일로, **Mini 시리즈(Mini 5 Pro 포함)는
MP4 안의 자막 트랙으로** 저장되므로 탐색기에서는 안 보입니다. 05 단계가 둘 다 자동으로 읽습니다(내장 자막은 ffmpeg 로 추출).
이 값을 프레임 EXIF 에 넣어 OpenDroneMap(ODM) 에 주면 별도 지상기준점 없이 미터 단위로 지리참조된 3D 결과가 나옵니다
(정확도는 기체 GNSS 수준, 수 m). 자막이 없으면 사진 EXIF(항상 GPS 포함), DJI Fly 비행 기록(Flight Record), 지상기준점 순으로 대체합니다.

```bat
:: 0) 한 번만: Docker Desktop 설치 후  docker pull opendronemap/odm
:: 1) 영상 -> 프레임 (1초 간격, 흐린 프레임 제거, GPS EXIF)
python scripts\05_dji_video_to_frames.py --video D:\DJI\DJI_0012.MP4 --out C:\hackathon\frames\beach_a
::    또는 이동거리 기준:  --every-m 6      고도에서 자동:  --auto-spacing --overlap 0.75
:: 2) 프레임 -> 3D 지도 (30분~수시간). 현장 확인은 --preset fast, 최종은 --preset high
python scripts\06_build_3d_map.py --frames C:\hackathon\frames\beach_a --project beach_a --preset fast
::    결과: outputs\3d\beach_a\{orthophoto.tif, dsm.tif, dtm.tif, pointcloud.laz, mesh\, report.pdf}
:: 3) QGIS 에서 orthophoto.tif 위에 폴리곤 레이어(텍스트 필드 class) 로 더미를 그려 labels.geojson 저장
::    반자동: --auto-candidates 로 높이 0.15 m 이상 덩어리를 후보로 뽑아 QGIS 에서 정리
python scripts\07_label_measure.py --dsm outputs\3d\beach_a\dsm.tif --dtm outputs\3d\beach_a\dtm.tif --labels labels.geojson --ortho outputs\3d\beach_a\orthophoto.tif
```

07 의 출력은 `measurements.csv`(라벨별 면적 m², 부피 m³, 최대 높이, 밀도, 무게 kg), 같은 내용을 담은
`measurements.geojson`(QGIS 색칠용), class 별 합계 `measurements_summary.json`, 정사영상 오버레이 PNG 입니다.

- **부피** = 폴리곤 안 (DSM - 바닥면) 의 양수 부분 x 픽셀 면적. 바닥면은 `--base dtm`(기본, ODM 지면모델) /
  `plane`(폴리곤 테두리에 평면 맞춤, DTM 없을 때) / `min`(최솟값, 경사지에서 과대) 중 선택.
- **무게** = 부피 x 겉보기 밀도. 기본 밀도표(`coastcd/volume.py` `DENSITY_KG_M3`: mixed 120, plastic 60, styrofoam 15,
  net 250, wood 350 kg/m³ 등)는 문헌 근사값이라 **현장에서 더미 1~2개를 실제로 달아 `--density-json` 으로 보정**해야 합니다.
- class 이름은 소문자로 비교합니다. 표에 없는 class 는 부피만 나오고 무게는 비웁니다.
- 탐지 모델(YOLO-seg 등) 을 붙일 때는 결과를 같은 형식(폴리곤 + `class`) GeoJSON 으로 저장해 `--labels` 에 넣으면 됩니다.

촬영 권장: 고도 30~60 m, 속도 3~5 m/s, 짐벌 -70~-90도, 격자 비행 + 대상 주변 궤도 비행 한 바퀴. 사진 20 장 미만이면 재구성이 실패합니다.
도커를 못 쓰는 PC 에서는 WebODM 이나 다른 PC 에서 돌린 뒤 `06 --collect-only` 로 결과만 모읍니다.

합성 검증(tests/test_drone3d.py): 반지름 2 m, 높이 0.6 m 원뿔 더미(참값 2.513 m³) 를 5 cm DSM 에서 dtm/plane/min 세 방식 모두 3% 이내로 복원,
경사 10% 지형에서 plane 은 3% 이내, min 은 2배 이상 과대. 실제 ODM DSM 은 잡음과 구멍이 있어 이보다 오차가 큽니다.

## 기존 스크립트(항공드론 해커톤/) 검토 메모

- `read_skysat_tif.py`: 미리보기 2000 px 는 약 17 m/px 로 Sentinel-2 보다 거칩니다. 베이스라인 해안선이 이 축소본에서 나온 것이므로 해빈(폭 40 m) 이 2 px 입니다. `full` 모드는 2.5 GB 를 메모리에 올리니 쓰지 마세요. 01 단계의 크롭으로 대체됩니다.
- `extract_coastline.py`: 검은 픽셀을 여백으로 추정하는 방식은 그림자를 여백으로 오인할 수 있어 알파 밴드로 바꿨습니다. 임계값 논리(밝기 OR 질감)는 그대로 `water_mask.py` 에 옮겼습니다.
- `fix_georeferencing.py`: 구글지도 좌표는 국내에서 수 m 오프셋이 있을 수 있습니다 (확인 필요). 800 MB 를 다시 쓰는 대신 `rasterio.open(path, "r+")` 로 transform 만 고치면 수 초에 끝납니다. 변화탐지에서 중요한 것은 절대 위치가 아니라 시기 간 상대 정합이며, 그 부분은 `register.py` 가 담당합니다.
- `s2_download.py`: 구조 좋습니다. 좌표계만 SkySat 과 같은 EPSG:32652 로 바꿨습니다. 이미 32651 로 받은 파일이 있어도 04 단계가 재투영하므로 그대로 쓸 수 있습니다.
- 원본 크기는 34447(가로) x 23973(세로) 입니다. 메모의 "24000 x 34000" 은 가로세로가 바뀐 표기입니다.

## 다음 단계

- 실제 SkySat 으로 01~03 실행 후 `outputs/coastline_overlay.png` 확인, 임계값·질감 창 조정
- 두 시기 마스크 비교(침식/퇴적 폴리곤, transect 변위) 모듈 `coastcd/change.py`
- 조위 무관 해안선 지표(습윤선·식생선) 로 전환 여부 결정
- Streamlit 대시보드에서 `coastline_4326.geojson` Swipe 표시
