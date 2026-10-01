# 드론대장 붕붕이 — 드론 영상 기반 해안쓰레기 무게 추정·수거 계획

2026 항공·드론 산업 수요 기반 해커톤. DJI Mini 5 Pro 로 찍은 해안 영상에서 해양쓰레기를 종류별로 분할하고,
3D 복원(DSM)으로 개별 부피를 재서 **겉보기 밀도**로 무게를 구한 뒤, 격자 kg 지도와 수거 계획까지 만든다.

```
DJI Mini 5 Pro 영상 → 쓰레기 검출·분할(종류별) → 3D 복원(DSM) → 개별 부피
→ 종류별 겉보기 밀도로 무게 → 10 m 격자 kg 지도 → 수거 계획(인력·마대·차량·경로)
```

- 대상: 해안(해변) 쓰레기. 물 위 부유쓰레기는 제외 (수면이 움직여 3D 복원 불가).
- 차별점: 기존 연구는 개수만 세거나 두께를 가정해 무게를 추정 (Andriolo et al. 2024). 우리는 3D 로 실제 높이를 재서 개별 부피 → 무게 → 수거 계획까지 연결.

## 폴더 구조

```
dronecap/                실시간 수집 패키지 (자세한 설명: docs/dronecap/README.md)
  config.py · timeutil.py · session.py        설정, UTC/monotonic 시각 규칙, 세션 폴더·CSV
  stream.py · recorder.py · frames.py         RTSP 수신 스레드(재접속), ffmpeg copy 녹화, 간격 프레임 저장
  capture_app.py                              1단계 메인 루프(HUD·키 조작)
  ocr/ (parse, roi, engine, sources, runner, live)  화면 숫자 OCR (RapidOCR CPU / Tesseract), ROI 선택, 입력 소스, 수신과 동시 실행
  media_meta.py                               녹화 .SRT / 사진 XMP 의 GPS·고도 읽기, COLMAP GPS 기준 파일 (비행 데이터 주 경로)
  sync.py · sfm.py                            프레임↔OCR 시간 매칭, 프레임 선별·COLMAP 명령·GPS 미터 정렬
  object3d.py                                 orbit 영상 → pycolmap SfM(CPU) → 바닥 평면 → visual hull → 길이·넓이·높이·부피
config/dronecap.yml      dronecap 설정 (RTSP 주소, 재접속, 녹화, OCR 필드, 동기화 오프셋)
scripts/20~30_*.py       dronecap 실행 스크립트 (20 수신, 21 ROI, 22 OCR, 23 매칭, 24 선별, 25 COLMAP, 26 SRT, 27 사진 메타, 29 테스트 송출, 30 물체 부피)
tests/test_dronecap.py   dronecap 자동 테스트 (드론 없이 실행 가능)
tests/test_object3d.py   합성 상자로 visual hull·평면·축척 기하 검증

litter3d/                파이썬 패키지 (자세한 설명: litter3d/README.md)
  drone.py               DJI Mini 5 Pro 스펙, 고도↔GSD, 촬영 설계
  srt.py                 DJI .SRT 자막 텔레메트리 파서
  frames.py              영상 → 겹침 기준 프레임 추출 + ODM geo.txt
  classes.py             쓰레기 13 클래스, 겉보기 밀도표(출처/가정값 표시), 개당 평균무게
  segment.py             COCO→YOLO-seg 변환, YOLO 학습·추론, 색 기반 베이스라인
  reconstruct.py         ODM/COLMAP 실행 래퍼, DSM·정사영상 읽기
  volume.py              DSM 차분 부피 (링 평면 바닥 추정)
  mass.py                부피 × 겉보기 밀도 → 무게 (최소/대표/최대)
  gridmap.py             격자 kg 히트맵
  plan.py                수거 계획 (마대·톤백·트럭·23 kg 규칙·CVRP 경로)
  report_html.py         결과 HTML 리포트 (검출 오버레이·차트·히트맵·수거 계획, 오프라인 동작)
  synthetic.py           검증용 합성 장면
  dataset.py             라벨 형식 판별·YOLO-seg 변환·train/val 분할
scripts/
  10_flight_design.py    고도별 GSD 표, 목표 GSD 촬영 계획
  11_extract_frames.py   영상(+SRT) → 프레임 + geo.txt
  12_reconstruct.py      ODM 으로 DSM + 정사영상
  13_estimate_mass.py    DSM + 정사영상 (+마스크/모델) → 무게·격자·수거계획
  14_synthetic_demo.py   합성 장면으로 전체 파이프라인 검증
  15_leirosa_validation.py  Andriolo 2024 실측 데이터로 검증 (표 재현·시뮬레이션·사진 검출 비교)
  16_photo_demo.py       실제 해변 사진(논문 크롭)으로 2D 리포트 (DSM 없음 모드)
docs/validation_leirosa.md  검증 결과와 해석
data/andriolo2024/        논문 그림 크롭 (CC BY)
notebooks/
  colab_train_yoloseg.ipynb  Google Colab 에서 YOLO-seg 학습 (라벨 형식 자동 판별·변환·분할·학습·평가)
tests/test_litter3d.py   자동 테스트 (실제 데이터 없이 실행 가능)
outputs/                 결과물 (git 제외)
```

## 빠른 시작 (Windows, VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\activate

python scripts\10_flight_design.py --gsd 0.5
python scripts\14_synthetic_demo.py --out outputs\synthetic
pytest tests -q
```

실시간 드론 영상 수집(RTMP→MediaMTX→RTSP)·화면 OCR·3D 복원 준비는 [docs/dronecap/README.md](docs/dronecap/README.md) 참고
(`python scripts\20_capture.py`).

실제 데이터 흐름과 각 숫자의 신뢰도(출처 있음 / 가정값 / 확인 필요)는 [litter3d/README.md](litter3d/README.md) 참고.

## 핵심 근거

- 국내 해양쓰레기 수거량 5년(2020–2024) 64만 9,749톤 중 해안쓰레기 50만 1,517톤(약 77 %) — 해양수산부 국회 제출자료, 뉴시스 2025-09-24
- 수거 현장은 무게(톤) 단위로 움직이지만 드론 연구는 개수·면적만 보고 — Andriolo & Gonçalves (2024)
- 드론 영상 부피 계산 선행연구 — Kako et al. (2020); 무게 추정 3방법 비교, DSM 부피가 개선 방향 — Andriolo et al. (2024)
- 적정 GSD 0.5–1.25 cm/px — Andriolo et al. (2023)
