# 세션 인수인계 — 해안쓰레기 드론 탐지·부피·수거계획 (2026-10-01 기준)

새 세션은 이 문서부터 읽고 시작한다. 프로젝트 폴더: `C:\Users\user\Desktop\드론\aerodrone_hackathon`

## 0. 사용자 선호 (반드시 지킬 것)

- **모든 답변·작업 중 안내를 한국어로만.** 영어로 쓰면 강하게 불만을 표시함 (여러 번 지적받음).
- **앞서나가지 말 것.** 사용자가 지금 보고 싶은 것(예: "실시간 탐지만 먼저")을 먼저 하고, 다음 단계는 제안만.
- 결과는 **숫자 + 그림 파일 경로**로 보여주고, 안 된 건 안 됐다고 솔직하게.
- 큰 다운로드·설치 전엔 한 줄로 알리고 진행 (사용자는 대체로 "ㄱㄱ").

## 1. 프로젝트 방향 (최종 확정)

- 해커톤 과제: 원래 "위성-드론 정합 + 변화탐지"였으나 **해안쓰레기 위치·부피·무게 추정 → 수거계획**으로 전환.
- 업체 의도: 업체는 지금 **"박스 면적 × 고정 계수 = 무게"** 로 처리 중 → 이걸 고쳐달라는 것으로 해석.
- **실시간 자동 비행은 포기.** (DJI Mini 5 Pro는 SDK 미지원, 드론 추가 구매 불가)
  → 구조: "비행은 사람, 나머지는 AI" — 영상 촬영 → 탐지 → 3D → 부피 → 무게 → 작업카드/지도.
- 발표 우선순위: ① 3D→부피→무게(차별점) ② 쓰레기 탐지 성능 ③ 대시보드·경로(기초 있음) ④ 실시간·자동비행은 "확장 계획"으로만.
- 방향 문서: `docs/해안쓰레기_수거계획_방향정리.md`

## 2. 데이터 위치

| 데이터 | 경로 | 비고 |
|---|---|---|
| 업체 라벨 (문갑도) | `..\드론 쓰레기 데이터\MGD_쓰레기.json` (복사본 `data/company/MGD_labels.json`) | 42개, 위경도 사각형. **weight_kg = 면적 × 계수(스티로폼 0.012, 로프·어망 0.024, 플라스틱 0.020 kg/m²)로 계산된 값 — 실측 아님** |
| 업체 정사영상 | `data/company/ortho/MGD.tif` (ECW를 QGIS GDAL로 변환, 2.3 GB, 3.02 cm/px, EPSG:5186) | 라벨이 일부만 있음(보이는 쓰레기 대부분 미라벨) |
| 업체 칩 | `data/company/chips/` (2048px 칩 48장 + labels_coco.json) | 평가용 |
| AI Hub 해안 오염물질 | 원본 `..\AI hub 데이터\Training\` (tar/.crdownload, 라벨 zip) | 꺼낸 이미지: `data/aihub/images_bbox` (4만장), `images_poly` (1.5만장, 어망·로프) |
| AI Hub COCO | `data/aihub/coco_bbox.json` | 11클래스, 21만 객체, 고도 메타 있음 |
| 거리별 학습셋 | `data/seg_aihub_gsd/` | 1.2만장을 2~4 cm/px로 축소해 1024 판에 붙임 |
| UAVVaste (공개) | `data/external/uavvaste/` | 폴란드 도시 쓰레기, 772장 |
| 드론 시험 영상 | `..\DJI_20261001151509_0007_D.MP4` + `.SRT` | 송도(37.3843, 126.6571), 고도 ~7 m, 선회. 상자 2개: **큰 60×45×12 cm(32.4 L), 작은 23×17×8 cm(3.1 L)** |
| | `..\DJI_20261001145314_0006_D.MP4` | SRT 없음 → 사용 불가 |

## 3. 코드 (`litter/` 패키지, 실행: `.venv\Scripts\python.exe -m litter.<모듈>`)

| 모듈 | 하는 일 |
|---|---|
| `aihub.py` | AI Hub tar(받는 중이어도)에서 이미지 꺼내기 `extract`, LabelMe→COCO `convert`, 거리별 학습셋 `pack` |
| `seg.py` | YOLO 타일 학습/추론/평가 (`prepare/train/predict/eval`), 한글경로 안전 `_imread/_imwrite` |
| `ortho.py` | 정사영상+GeoJSON → 칩 COCO |
| `orbit3d.py` | 선회 영상 → 프레임 추출(`--start/--end`) → pycolmap 3D (CPU) → viewer.html, `--srt`로 실제 크기 |
| `volume.py` | 바닥면 RANSAC, 고도로 축척, 높이지도/볼록껍질 부피 |
| `orbit_map.py` | 3D 카메라 방향으로 탐지 위치 계산 → GPS 맞춤 → `map.html`, `sheet.jpg`, `detections/`, `objects.csv` |
| `mosaic.py` | 3D로 정사영상(모자이크) 만들고 물체 표시 → `mosaic_objects.jpg`, `mosaic.tif` |
| `objvol.py` | **SAM 2 마스크 + 여러 프레임 투표**로 물체 점만 골라 부피 (`--ply`로 조밀 점 사용) |
| `video_map.py` | SRT만으로 지도 (방향 정보 없어 부정확 — orbit_map 사용 권장) |
| `weight.py`, `plan.py`, `report.py`, `geometry.py`, `camera.py` | 무게(물리+학습+구간), 수거계획·작업카드, 위치 계산 (합성 데이터로 검증됨) |
| `live.py` | 실시간 미러링 탐지(scrcpy+adb) — **실시간은 접었으므로 참고용** |
| `sim_ortho.py` | **실시간 드론 운용 시뮬레이터** — 정사영상 MGD.tif를 가상 세계로 (윈도우 읽기만), 지그재그 커버리지 → 가상 카메라 프레임 → YOLO(CPU) → 지도 클러스터링 → 애매한 후보 저고도 재방문 → 업체 라벨 재현율. `--model aihub|uavvaste`, `--no-revisit`. 결과 `runs/sim/<name>/` (map.png, map.html, flight.mp4/gif, detections.csv, objects.csv, summary.json). 100×100 m 라벨 밀집 구간 기준 3분 |
| `synth.py` | 합성 해변 데이터 생성 (파이프라인 검증용) |

## 4. 지금까지 결과 (숫자)

**쓰레기 탐지**
- UAVVaste 모델 (`runs/seg/uavvaste_det_s/weights/best.pt`): UAVVaste 시험 AP50 0.77 / **문갑도 재현율 0.28** (현장 차이 큼)
- AI Hub 거리별 모델 (`runs/seg/aihub_gsd_det_s/weights/best.pt`): **12 에폭 중 5까지만 학습**, 검증 mAP50 0.31 (아직 오르는 중). GPU 메모리 부족으로 두 번 중단됨 — 학습 중엔 다른 GPU 작업 금지.
  - 이 모델은 골판지 상자를 "Styrofoam_Buoy/Plastic_Buoy"로, 텐트 고정 추를 "Styrofoam_Buoy"로 잡음 (종이상자 클래스 없음)

**영상 0007 (선회 시험)**
- 3D: 100/100 프레임 등록, 점 28,580, 재투영 0.42 px, 축척 편차 2.8%, 3D↔GPS 잔차 1.16 m
- 지도 핀: `runs/map/0007_sfm/map.html`, 정사영상 `mosaic_objects.jpg` (17.5×17 m, 1 cm/px) — **두 상자 위치 정확히 일치**
  - 작은 상자 37.3842486, 126.6570694 / 큰 상자 37.3842818, 126.6570618
- 부피 (`runs/orbit/0007/objvol_sparse/`, 희소 점 + SAM 마스크):

| 상자 | 정답 | 반경 방식(이전) | SAM 마스크 방식 |
|---|---|---|---|
| 작은 | 23×17×8, 3.1 L | 15 L | 21×14×12 cm, 1.8~2.0 L (점 11개뿐, 불안정) |
| 큰 | 60×45×12, 32.4 L | 26~55 L | 64×41×14 cm, 27.9~30.8 L (−14 ~ −5%) |

- **조밀 복원 완료** (COLMAP GPU, 점 243만 개 → `runs/orbit/0007/fused_dense.ply`, 결과 `runs/orbit/0007/objvol_dense/`)
  + 높이를 '최댓값'이 아닌 **중앙값 기준**으로 바꿈 (모서리 잡음 제거):

| 상자 | 정답 | 조밀+마스크+중앙값 (높이지도 / 마스크면적×높이) |
|---|---|---|
| 작은 | 23×17×8 cm, 3.1 L | 27×19×7 cm, **3.0 L (−3%) / 3.4 L (+10%)** |
| 큰 | 60×45×12 cm, 32.4 L | 62×49×13 cm, **38.7 L (+19%) / 39.2 L (+21%)** |

- 남은 오차: 큰 상자 바닥면이 약 10% 크게(가장자리 그림자 경계), 높이 +1 cm. 볼록껍질 부피는 +60%로 과대 → 높이지도 방식 사용 권장
- 조밀 복원은 위 '진행 중' 1번 명령으로 재현 가능 (100프레임 약 30분)

**시뮬레이터 (`sim_ortho`, 라벨 밀집 100×100 m, 고도 20 m / 재방문 8 m, CPU, 라벨 7개)**

| 실행 | 확정 물체 | 재현율 | 비행 |
|---|---|---|---|
| `runs/sim/aihub_cov` (커버리지만) | 8 | 0.71 (5/7) | 679 m · 136 s |
| `runs/sim/aihub_rv` (재방문 25회) | 14 | **0.86 (6/7)** | 1718 m · 498 s (+697 m) |
| `runs/sim/uav_cov` / `uav_rv` (UAVVaste) | 2 / 15 | 0.14 (1/7) | — |

- 재방문은 애매한 후보(0.25~0.5) 25개 중 22개 기각·3개 확정 — 한 프레임에서만 잡힌 불안정 탐지 거르기 효과. 정밀도는 미라벨 쓰레기 때문에 참고용.

## 5-0-0. 2026-10-02 02:45 최신 상태 (리눅스 재부팅 전 기록) — **먼저 읽을 것**

- **탐지 모델 확정 수치** (`docs/figures/21_다중현장_통합모델_비교.png`, `cross_eval_multi.json`, `multi_site_data_table.md`):
  - 문갑도(우리 현장): **AI Hub 5에폭 모델이 최고** 재현율 0.74 / AP50 0.54 (`runs/seg/aihub_gsd_det_s`)
  - 하와이: 8클래스·1024 미세조정 AP50 0.62 (`runs/seg/hawaii_ft8`), 통합 모델 0.52
  - 튀니지: 통합 모델 AP50 0.69 (`runs/seg/multi_site`, AI Hub+하와이+튀니지 5,405장 12에폭)
  - Colab 30에폭 AI Hub 모델은 현장 전이 악화(문갑도 0.34) → 데이터 다양성이 핵심이라는 근거
  - 발표 프레이밍: "해안쓰레기 데이터 부족 → 여러 지역 합쳐 학습. 국내 유지, 해외 2~3배"
- **최종 학습(YOLO11-m 1280, 통합 데이터)**: 노트북에서 돌리다 리눅스 재부팅으로 중단 → **Colab**으로 이관. 패키지 `C:\work\colab\{seg_multi.zip(727 MB), multi_train_colab.ipynb, evalpack.zip, eval_models.py}` (저장소 `docs/colab/`에도 있음). 결과 `Drive/aerodrone/out/multi_m1280_best.pt` + `cross_eval.md`가 생기면 `runs/seg/multi_m1280/weights/best.pt`로 받아 `crosseval`·`hawaii eval`로 재측정.
- **0015 상자 30개 일괄 3D**: 3D 25/25, 위치 26/30, 부피 11/30(정답 42×32×39, 높이는 ±6 cm 4개, 발자국 흔들림). `runs/orbit/0015_all/`.
- **GitHub**: 결과물 1,920개·가중치 7개·코드·문서 전부 `feature/coastal-litter-pipeline`에 푸시됨. SAM2(155 MB)만 제외(자동 다운로드).
- **다음**: ROS 2/Gazebo 시뮬(리눅스, 4절 조사 참고 — 6시간 안엔 PX4 SITL+gz 기본 월드+카메라 토픽+YOLO 정도가 현실적, 안 되면 `sim_ortho` 영상으로 대체), 발표 슬라이드(`docs/발표_스토리라인.md` 12장).

## 5-0. 2026-10-01 밤 (대회 D-1, 마감 10-02 아침) 현재 상태

전체 결과·수치는 **`docs/파이프라인_결과정리.md`** 에 정리돼 있음(발표용). 아래는 밤 작업 요약과 돌고 있는 것.

- **부피 코드 v2** (`objvol.py`): 지역 바닥 평면(RANSAC)·바닥점 제거·뒤집힌 카메라 제외·중심 재추정. 크기 아는 상자 3개 오차 **−10 ~ +20 %** (0010 키 큰 상자 47.2 L/정답 52.4, 0007 큰 38.8/32.4, 작은 3.7/3.1). 옵션 `--min-track`(GPS 경로 축척 강제, SRT 고도가 틀릴 때 4 정도로), `--search-r`(연석·벽 옆이면 0.45), `--device cpu|0`.
- **빠른 조밀 복원 설정 검증**: 1200 px·소스 8·반복 3·cache 8 GB → 60프레임 **3분 24초**(기존 45분). 배치 예: `C:\work\dense0015_b24\run_fast.bat`. **cache_size 기본 32 GB가 노트북 꺼짐 원인 후보** — 꼭 8로.
- **YOLO-World**(`yolov8s-worldv2.pt`, 클래스 글로 지정): `video_map`·`orbit_map`에 `--classes` 추가. 야간 0015 영상에서 AI Hub 모델 4개 → **상자 30개**.
- **영상 0015** (`..\DJI_20261001205404_0015_D.MP4`, 야간 10분 531 m, 상자 ~28개): 탐지·지도 `runs/map/0015_world2/` 완료. 3D 뼈대는 됨(#24 구간 60/60) · 조밀도 됨 · **부피는 실패** (#24: 옆 연석 때문에 바닥 평면 30° 틀어짐, #19: 어두운 아스팔트라 37/60 등록·점 희박). 단방향 경사 통과 촬영의 한계 → 더 시도하지 말고 0007·0010 숫자 사용. SRT 고도가 실제의 1/5로 기록돼 `--alt_fix 2.0`, `--min-track 4` 필요.
- **0010 흰 송장 상자**(`Styrofoam_Buoy_4`)는 39 cm 상자가 아니라 60×50×12.5 cm 납작 상자로 측정됨 — 정답 모름.
- **공개 해안 데이터** `data/external/hawaii_debris`(칩 1,587장 2 cm/px, 지오참조, 라벨 10,703), `tunisia_litter`(3,676장 세그 라벨). 겹침 원본(3D 가능) 공개 세트는 없음 → 해안은 탐지·지도·수거계획까지, 3D는 송도 영상으로.
- **돌고 있던 에이전트 작업**(결과는 `runs/hawaii/`, `runs/eval/tunisia/`, `docs/figures/17~19_*`): ① 하와이 칩 탐지→위경도 지도→수거계획 데모(`litter/hawaii.py`) ② 튀니지+문갑도 교차평가 표(`litter/crosseval.py`). 없으면 중단된 것.
- 팀원 수거계획 대시보드: https://jihu0629.github.io/aerodrone_hackathon/ (업체 라벨 기반, 면적×가정 두께). 우리 3D 결과를 `window.PLAN.objects` 형식으로 넘기면 연결 가능 — 아직 안 함.
- git: `feature/coastal-litter-pipeline` 브랜치에 푸시 중 (공개 저장소 → 업체 정사영상·사람 찍힌 이미지는 올리지 않음).

## 5. 진행 중이던 작업 (새 세션에서 이어서)

1. **조밀 복원 (COLMAP GPU)** — ⚠️ 아래는 옛 기본 설정(45분). **5-0의 빠른 설정을 쓸 것.** — 이전 세션 임시폴더에서 돌던 것은 세션 종료 시 사라질 수 있음. 다시 돌리려면 **영문 경로**에서:
   ```bash
   W=C:/work/dense0007; P=C:/Users/user/Desktop/드론/aerodrone_hackathon; CM=$P/tools/colmap/bin/colmap.exe
   mkdir -p $W && cp -r $P/runs/orbit/0007/frames $W/images && cp -r $P/runs/orbit/0007/sparse/0 $W/sparse && cd $W
   $CM image_undistorter --image_path images --input_path sparse --output_path dense --output_type COLMAP
   $CM patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
   $CM stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric --output_path dense/fused.ply
   ```
   100프레임 기준 30분 이상 걸림. 끝나면:
   ```bash
   .venv/Scripts/python.exe -m litter.objvol --orbit runs/orbit/0007 --srt ../DJI_20261001151509_0007_D.SRT \
     --ply C:/work/dense0007/dense/fused.ply --objects runs/map/0007_sfm/objects.csv \
     --select "Plastic_Buoy:55,Styrofoam_Buoy:22" --truth "Plastic_Buoy_55:23x17x8,Styrofoam_Buoy_22:60x45x12" \
     --out runs/orbit/0007/objvol_dense
   ```
   → 희소 결과(위 표)와 비교해 사용자에게 보고.
2. **AI Hub 모델 학습 이어하기** (5/12 에폭):
   `.venv/Scripts/python.exe -c "from ultralytics import YOLO; YOLO('runs/seg/aihub_gsd_det_s/weights/last.pt').train(resume=True)"`
   끝나면 문갑도 칩 재현율 재측정 (`seg.predict` → `seg.eval --iou 0.3`, 이전 0.28과 비교).

## 5-1. 영상 0010 (2026-10-01 17:17 촬영, 골목 왕복 비행) — 세션과 별개로 돌고 있음

- 영상: `..\DJI_20261001171735_0010_D.MP4` + `.SRT` (40.7초, 실제 높이 약 2 m, 카메라 앞쪽 약 30° 아래, 25 m 왕복)
- 3D: 순차 매칭은 왕복 때문에 3조각으로 쪼개짐 → `--exhaustive`로 100/100 한 덩어리 (`runs/orbit/0010x`, 점 91,282)
- **SRT 고도(3.7 m)가 실제 바닥 높이와 안 맞음**(이륙 지점이 높았던 듯) → `volume.to_metric_auto`가 수평 이동 10 m 이상이면
  **GPS 경로로 축척**을 맞추도록 수정 (3D↔GPS 잔차 8.07 m → 0.51 m). 0007은 고도 기준 그대로(편차 2.8%)
- 지도·정사영상: `runs/map/0010/` (map.html, mosaic_objects.jpg 24×33 m, sheet.jpg, objects.csv)
- **조밀 복원 + 부피**: `C:\work\dense0010\run_all.bat`을 별도 프로세스로 실행해 둠 (세션 종료와 무관)
  - 진행 확인: `C:\work\dense0010\status.txt` (START → STEREO DONE → FUSION DONE → VOLUME DONE)
  - 세부 진행: `C:\work\dense0010\log_pm.txt`의 마지막 "Processing view N / 100" (1차·2차 두 번 돎)
  - 결과: `runs/orbit/0010x/objvol_dense/objvol.json` (쓰레기 후보 전부의 크기·부피), 마스크 그림 `*_masks.jpg`
  - 실패 시 다시: `cmd /c C:\work\dense0010\run_all.bat`
  - `C:\work\drone`은 `드론` 폴더로 가는 연결(junction) — 배치 파일에서 한글 경로를 피하려고 만든 것
- 상자 실제 크기 받음: 키 큰 상자 32×42×39 cm (= `Styrofoam_Buoy_3_2`, v2로 −10 %), 47×47×39 cm 상자는 영상에서 어느 것인지 미확인. 결과 `runs/orbit/0010x/objvol_v2/`

## 6. 다음 할 일 후보 (사용자에게 제안만, 먼저 묻기)

- 부피 → **재질별 무게** 연결 (`weight.py`) → 작업카드(`plan.py`, `report.py`)로 영상 0007 끝까지 한 번에
- 다음 촬영 가이드: 고도 3~4 m, 반경 3 m, 카메라 45° 기울여 한 바퀴 + 수직 1회, SRT 켜기, 크기 아는 기준물체 함께
- 탐지: 추적(tracking)으로 중복 제거 개선, 위성사진 특징점 정합(LoFTR)으로 절대위치 개선
- 업체 문의: weight_kg 실측 여부, 전체 라벨, 수거 지점 GPS

## 7. 환경 메모

- Python venv: `aerodrone_hackathon\.venv` (Python 3.12, torch 2.8 cu128, ultralytics 8.4, pycolmap 4.2.1 CPU, scikit-learn, mss)
- GPU: RTX 4060 Laptop 8 GB / RAM 16 GB (여유 적음 — 학습과 3D를 동시에 돌리면 죽음)
- 설치된 도구: QGIS 3.44 LTR(ECW 변환용), COLMAP 4.2.1 CUDA (`tools/colmap`), scrcpy 4.1(adb 포함), SAM 2 (`sam2.1_b.pt`)
- **한글 경로 주의**: cv2.imread/COLMAP CLI/GDAL은 한글 경로에서 실패 가능 → `litter.seg._imread`, 영문 작업 폴더 사용
- 태블릿(갤럭시 탭 S10 Lite) 무선 adb: 삼성 "보안 위험 자동 차단"을 꺼야 연결 유지. 실시간은 접었으므로 다시 켜두라고 안내할 것
- 디스크 여유 약 85 GB
