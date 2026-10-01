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

- 남은 오차 원인: 높이 2~4 cm 과대(수직 촬영), 작은 물체 점 부족

## 5. 진행 중이던 작업 (새 세션에서 이어서)

1. **조밀 복원 (COLMAP GPU)** — 이전 세션 임시폴더에서 돌던 것은 세션 종료 시 사라질 수 있음. 다시 돌리려면 **영문 경로**에서:
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
