# 인수인계 프롬프트 — 위성 우선순위 + 코리더 비행 경로 (2026-10-01 세션 → 새 채팅)

아래 블록을 그대로 새 채팅의 첫 메시지로 붙여 넣는다. (로컬 PC 에서 Claude Code 를 열고 저장소 루트에서 시작)

---

너는 2026 항공·드론 산업 수요 기반 해커톤 팀 "드론대장 붕붕이" 의 기술 작업을 이어받는다. 마감은 10-02 아침이다. 아래는 이전 세션(클라우드 Claude Code)이 한 일과 결정, 숫자, 남은 일이다. 모든 답은 한국어로, 결과는 숫자와 파일 경로로, 안 된 것은 안 됐다고 말한다. 앞서 나가지 말고 내가 지금 보고 싶은 것부터 한다.

## 1. 프로젝트와 저장소

- 주제: 드론·위성으로 해안쓰레기를 찾아 **작업자용 수거 계획**(무게·마대·인원·경로)을 만든다. 기업 라벨의 무게가 "면적 × 고정계수" 라 실제의 1/100 수준인 문제를 3D 부피·겉보기 밀도로 고친다. 과제 기업 3dlabs 는 위성 플랫폼(지상국·K-ARD·활용세부 공급) 회사라, **위성을 쓰는 이유는 기술이 아니라 사업 포지셔닝**이다("3dlabs 위성 플랫폼의 새 활용세부: 해안쓰레기").
- 저장소 1: `jihu0629/Aerodrone-Hackaton` — `main` 은 ShoreSweep Planner(기업 라벨 42개 + 정사영상 → 수거계획 HTML, `litter3d/collect.py`, `terrain.py`, `collect_report.py`, `scripts/17_collection_plan.py`). 이전 세션이 만든 브랜치 `claude/upbeat-dijkstra-hdho2d` 에 커밋 2개가 있는데 **푸시가 403 으로 막혔다**(세션 GitHub 계정 HSR2M 에 쓰기 권한 없음). 내용은 패치 파일 `위성우선순위_전체.patch` 로 받았다. 적용: `git checkout -b claude/upbeat-dijkstra-hdho2d origin/main && git am 위성우선순위_전체.patch`.
- 저장소 2: `dohun415/aerodrone_hackathon` — `main` 은 원래 과제4(위성-드론 정합) 검증 파이프라인(Sentinel-2 52SBG 2019·2026 4밴드가 `data/raw/` 에 있음, `pipeline/00_fetch_data.py` Planetary Computer, `03_stable_mask.py` NDWI vs RGB-only IoU 0.33). `feature/coastal-litter-pipeline` 은 탐지(YOLO11)→3D(COLMAP)→위치→부피(SAM2 투표)→무게→수거계획→작업카드, `sim_ortho.py`(정사영상 가상비행: 커버리지 0.71 → 재방문 0.86), `hawaii.py`(Zenodo 8381113 Winans 2023 칩 1,588장·라벨 10,703 → 니하우 5,476개 탐지·7.9~33.6 t), 발표 12장 스토리라인, 인수인계 문서. `feature/shoresweep-planner` 는 main + shoresweep_planner.
- 하와이 데이터의 실체: Moy 2018 점 데이터가 아니라 **Winans 2023 칩**(640 px, 2 cm/px, 지오참조 .aux.xml, 쓰레기 있는 곳 위주로 잘려 "없는 해안" 정보가 없음 → 밀도 "순위" 만 검증 가능).

## 2. 결정한 방향 (노션 정리와 합쳐서)

```
[위성] 어디를 날지 고르기 → [드론 1차] 자동 커버리지 → [AI] 탐지·위치 → [드론 2차] 불확실한 물체 재방문 → [3D] 부피 → 무게 → [계획] 대시보드
```
- 위성 픽셀(10 m)로는 쓰레기가 안 보인다. 그래서 위성은 **해안선 형상**(만입도·풍향 노출·해빈 띠 폭)을 재서 "쌓이기 쉬운 구간" 점수를 내고, 드론 정밀 촬영은 점수 상위 구간에만 쓴다.
- 발표 문장: "위성으로는 해안 형상을 재고, 형상 점수 상위 30 % 구간만 정밀 촬영해도 니하우 탐지 무게의 65 % 를 잡는다. 어디를 날지는 위성이, 무엇이 얼마나 무거운지는 드론 3D 가 정한다."

## 3. 이전 세션이 만든 코드 (패치 안에 전부 있음)

| 파일 | 내용 |
|---|---|
| `litter3d/priority.py` | `read_band`, `ndwi_max`(SCL 로 구름 제외, 모든 장면 구름이면 물), `water_mask`(0.5 km² 이상 덩어리, 호수 채움), `coast_from_mask`(둘레 벡터화 → 10 m 표본점·접선·바깥 법선, simplify 15 m), `add_features`(bay=1−반경 150 m 물 비율, exposure=cos(법선 방위−풍향), strip_m=NDVI 기반 띠 폭), `score_coast`(z 가중합 1/0.5/0.5, 100 m 이동평균), `make_segments`(200 m), `select_budget`(길이 예산), `capture_curve`, `FlightParams`(고도 60 m·촬영폭 86 m·offset 0.3·5 m/s·이동 10 m/s·배터리 25분), `order_segments`(최근접 이웃 양방향 + 2-opt), `plan_sorties`(launch="mobile" 소티마다 첫 구간 근처 이륙 / "fixed" 고정 출발지·도달불가 기록), `export_geojson`, `export_litchi_csv`, `export_wpml_kmz`(DJI WPML 1.0.6 최소 구성), `summary` |
| `scripts/18_priority_flight.py` | CLI. `--s2-dir --scenes --tci-scene --density-grid|--density-csv --depot lon,lat --wind-from --budget-frac|--budget-km --alt --swath --passes --speed --battery-min --launch --out`. 출력: `예산대비포착률.png`, `우선구간_비행경로_지도.png`, `segments.csv`, `우선구간_경로.geojson`, `sortie_NN_litchi.csv`, `sortie_NN_dji.kmz`, `summary.json` |
| `tools/fetch_s2_aws.py` | STAC 없이 AWS `sentinel-cogs` 버킷에서 창만 받기. `--tile 4QCK --list 2025 --check --bounds ...` 로 장면 목록(창 안 구름률), `--scenes ... --bounds L B R T --out s2 --prefix niihau` 로 B02/B03/B04/B08/TCI/SCL |
| `tests/test_priority.py` | 합성 섬 3개 테스트 (전체 12개 통과) |
| `docs/위성우선순위_니하우.md`, `docs/figures/20_*.png`, `21_*.jpg`, `docs/examples/niihau/` | 결과·재현 명령·한계 |
| `input/s2_niihau/hawaii_niihau_ft_summary.json` | 검증용 탐지 kg 격자(dohun415 coastal 브랜치 `docs/figures/` 사본) |

필요 패키지: `rasterio shapely pyproj scipy numpy matplotlib opencv-python-headless pillow pytest`. 한글 폰트는 `fonts/NanumGothic.ttf` 또는 `C:\Windows\Fonts\malgun.ttf` 를 스크립트가 찾는다.

## 4. 데이터

- Sentinel-2 니하우: MGRS 타일 **4QCK**, 창 `368000 2407000 395000 2437000` (EPSG:32604). 맑은 장면 `S2B_4QCK_20250502_0_L2A`, `S2C_4QCK_20251103_0_L2A`, `S2B_4QCK_20250303_0_L2A` (창 안 구름 1 % 미만). `20240106`·`20251226` 은 바다 위 구름이 있어 뺐다. 버킷 경로: `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/4/Q/CK/2025/5/S2B_4QCK_20250502_0_L2A/B03.tif`.
- 문갑도(한국): 타일 52SBG, dohun415 `main` 의 `data/raw/s2_2026-09-16_52SBG_B0{2,3,4,8}.tif` 를 바로 쓸 수 있다. 라벨 42개와 10 m 지형 격자는 jihu0629 저장소 `plan.json`(ccr 브랜치 `항공드론 해커톤/수거계획/src/plan.json`) 에 있다.
- 하와이 라벨 원본: Zenodo 8381113 (클라우드에서는 차단됐고 로컬 `data/external/hawaii_debris/` 에 있음). `hawaii.py geo` 가 만드는 `runs/hawaii/chips.csv`(filename, island, lat, lon, n_labels) 를 `--density-csv lon,lat,n_labels` 로 넣으면 **탐지가 아닌 라벨**로 곡선을 다시 그릴 수 있다.

## 5. 결과 숫자

니하우 (해안 98.8 km, 검증 = 탐지 kg 격자 1,052셀·5,476개·13.7 t):

| 점수 | 상위 10 % 길이 | 20 % | 30 % | 50 % |
|---|---|---|---|---|
| 만입도만 | 20 % | 42 % | 53 % | 76 % |
| 만입도+노출+띠폭(기본) | 33 % | 53 % | **65 %** | 85 % |
| 노출만(북동 무역풍 60°) | 13 % | 37 % | 57 % | 83 % |
| 무작위 | 10 % | 20 % | 30 % | 50 % |

- 만입도 3분위 밀도 곶 40 → 직선 128 → 만 244 kg/km. 노출 kg 가중 평균 +0.39 (바람맞이, Moy 2018 과 같은 방향).
- 상위 30 % = 200 m 구간 148개(29.6 km) → 탐지 무게 67 % 포착. 이동식 이륙 **12소티, 132 km, 273분**. 고정 출발지(섬 남단)로는 대부분 구간이 배터리 왕복 불가.

문갑도 (라벨 42개, 조사 2026-07-29, 10 m 지형 격자 기준):
- 만입도만: 상위 30 % 길이 → 라벨 64 %, 50 % → 93 %. 만 4.7 개/km vs 곶 0.2 개/km.
- 노출은 **계절**을 따른다: 7월 라벨은 남쪽(195°) 해안에 몰렸고 겨울 북서풍(315°) 가정은 무작위보다 못했다(30 % → 31 %). 서해는 조사 직전 몇 달 풍향으로 계산할 것.

## 6. 한계 (발표에서 먼저 말할 것)

- 점수 가중치·고도·촬영폭·속도·배터리·offset 은 가정값. 만입도 창은 정사각 근사.
- 니하우 검증은 라벨이 아니라 탐지(미세조정 모델 재현율 0.58)이고 칩 있는 해안만 포함 → "순위가 맞는다" 까지만.
- 10 m 격자 계단 때문에 둘레 길이가 실제보다 길다(니하우 91.6 km). 비율 지표엔 영향 작음.
- 구간 사이 이동·복귀는 직선. WPML KMZ 는 실기체 임포트 미확인(Mini 5 Pro + RC2 막힌 사례 있음). Litchi 는 99 웨이포인트 제한.

## 7. 지금 할 일 (우선순위)

1. 패치 적용 → `python -m pytest tests -q` 로 12개 통과 확인 → 니하우 재현(`docs/위성우선순위_니하우.md` 의 명령).
2. **문갑도에 적용**: 52SBG Sentinel-2 로 같은 점수를 내고, 42개 라벨(`--density-csv`)로 곡선. 풍향은 여름 남풍 계열(약 195°)로.
3. 하와이 **라벨**로 재검증: `chips.csv` → `--density-csv`. 섬별(니하우·몰로카이·오아후) 곡선이 비슷하면 "형상 점수는 섬·계절을 넘어 통한다".
4. `sim_ortho` 에 선택 구간 bbox 만 넣어 "전체 커버리지 vs 상위 30 %" 비행거리·재현율 (노션 2장의 곡선).
5. 수거계획 대시보드(`collect_report.py`)에 구간 점수 레이어 추가.
6. 발표 한 장: 그림 20(곡선)·21(지도) + 위 문장. 노션 스토리 9장(하와이) 뒤에 넣는다.

---
