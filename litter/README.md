# litter — 해안쓰레기 위치·무게 → 수거계획 파이프라인

방향 정리는 [`docs/해안쓰레기_수거계획_방향정리.md`](../docs/해안쓰레기_수거계획_방향정리.md) 참고.
현장 문제 두 가지를 푼다:

1. **위치가 50 m 어긋남** → 라벨 픽셀마다 광선을 쏴서 지면 좌표 계산 (+ 위치 반경 표시)
2. **생각보다 크고 무거움** → 물체별 무게 + **최대 무게(예측구간 상한)** 로 인력 판단

## 단계

| # | 단계 | 모듈 | 데이터 없을 때 |
|---|---|---|---|
| 1 | 라벨 읽기 (COCO / YOLO / CSV 자동 판별, 무게 CSV 병합) | `ingest.py` | — |
| 2 | 텔레메트리 (텔레메트리 표 / DJI EXIF·XMP / SRT) | `telemetry.py` | — |
| 3 | 위치: 픽셀 광선 → 지면 (DSM 있으면 DSM과 교차), 몬테카를로 위치 반경 | `camera.py`, `geometry.georef` | 정사영상 라벨이면 `--ortho` |
| 4 | 중복 제거 (item_id, 없으면 같은 클래스 + 1.5 m) | `geometry.dedup` | — |
| 5 | 형상: 면적 · 뼈대 길이 · DSM 높이/부피 · 묻힘 의심 | `geometry.measure` | DSM 없으면 2D만 |
| 6 | 무게: 물리 사전값 → GBM 학습 보정 → conformal 90% 구간 | `weight.py` | 무게 정답 < 30개면 물리값 ×/÷2 |
| 7 | 계획: 인력(상한 기준) · 정거장 · 격자 · 마대/트럭 · 경로 | `plan.py` | — |
| 8 | 결과: 작업 카드 HTML · 지도 · CSV/GeoJSON · 위치오차/어블레이션 그림 | `report.py` | — |

설정값(클래스별 겉보기 밀도, 마대·트럭 용량 등)은 `config.py`에 있다. 대부분 **가정값**이라 업체 기준으로 바꾸려면 `--config my.json`으로 덮어쓰면 된다.

## 사용법

```powershell
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"

# 0) 데이터 받자마자 — 구조 확인
& $py -m litter inspect D:\업체데이터

# 1) 합성 데이터로 전체 검증 (업체 데이터 오기 전)
& $py -m litter synth --out data/litter_synth
& $py -m litter run --images data/litter_synth/images --labels data/litter_synth/labels_coco.json `
    --weights data/litter_synth/weights.csv --telemetry data/litter_synth/telemetry.csv `
    --dsm data/litter_synth/dsm.tif --eval --out results/litter_synth

# 2) 업체 데이터 (예시 — 포맷 확인 후 인자 조정)
& $py -m litter run --images D:\업체데이터\images --labels D:\업체데이터\labels.json `
    --weights D:\업체데이터\weights.csv --eval --out results/litter_company

# 동영상이면: 프레임 + SRT 텔레메트리 먼저 뽑기
& $py -m litter video --video DJI_0001.MP4 --srt DJI_0001.SRT --out data/frames --every 1
```

## 쓰레기 분할 베이스라인 — `seg.py`

파이프라인은 라벨(폴리곤)을 입력으로 받는다. 새 드론 영상에서는 사람이 라벨을 못 다니
모델이 대신 찾아야 한다 → YOLO-seg 베이스라인. 드론 원본(4000×3000)은 쓰레기가 너무
작아서 **1024 타일(겹침 20%)로 학습·추론**하고 원본 좌표로 합친다.

```powershell
# 1) COCO → 타일 데이터셋 (이미지 단위로 train/val/test 70/15/15 분할)
& $py -m litter.seg prepare --coco <labels.json> --images <원본폴더> --out data/seg_company
# 2) 학습 (RTX 4060 8GB: yolo11s-seg, batch 8, 1024)
& $py -m litter.seg train --data data/seg_company/data.yaml --model yolo11s-seg.pt --epochs 60
# 3) test 원본 이미지 타일 추론 → COCO 예측
& $py -m litter.seg predict --weights runs/seg/seg_company/weights/best.pt --images <원본폴더> `
    --coco <labels.json> --split data/seg_company/split.json --out results/seg_company/preds.json
# 4) 원본 이미지 단위 평가 (정밀도·재현율·AP50·클래스별 재현율·면적 비율)
& $py -m litter.seg eval --gt <labels.json> --pred results/seg_company/preds.json --split data/seg_company/split.json
# 5) 예측을 그대로 파이프라인에 (score < 0.25는 버림)
& $py -m litter run --labels results/seg_company/preds.json --images <원본폴더> --out results/litter_pred
```

평가 지표에서 볼 것:
- **재현율** — 놓친 쓰레기는 무게 0으로 계산됨 (선행연구 Winans 2023은 40%)
- **면적 비율** (예측 마스크 ÷ 정답) — 무게 모델의 입력이 면적이라 1.0에서 벗어나면 무게가 그만큼 치우침
- 클래스 무시 평가가 기본 (`--by_class`로 클래스까지 맞히기)

## 실시간 탐지 + 핑 — `live.py` (DJI Mini 5 Pro, 휴대폰 미러링)

Mini 5 Pro는 DJI SDK 미지원 → **scrcpy로 폰(DJI Fly) 화면을 노트북에 미러링**해서 탐지하고,
사람이 확인하면 adb로 DJI Fly 화면을 터치(선회 대상 지정 등)하고 핑을 기록한다.
GPS는 화면에 없으므로 **비행 후 SD카드 영상의 SRT와 시각을 맞춰** 계산한다.

```powershell
# 폰: 개발자 옵션 → USB 디버깅 켜기 / DJI Fly: 동영상 자막(SRT) 켜기 / 짐벌 -90° 권장
scrcpy --window-title=DRONE --max-fps=30 --stay-awake
& $py -m litter.live screen --window DRONE --weights runs/seg/uavvaste_det_s/weights/best.pt
#   [스페이스] 가장 확실한 탐지 터치+핑 · [1-9] 번호 선택 · [클릭] 박스 선택 · [p] 핑만 · [q] 종료
#   터치는 초록 안전영역 안의 탐지만 (가장자리 이륙·착륙·귀환 버튼 보호)
# 비행 후: 핑 ↔ SRT → 위경도 (+ 원본 4K 확대 사진)
& $py -m litter.live match --pings runs/live/<날짜>/pings.csv --srt DJI_0001.SRT --video DJI_0001.MP4 --out runs/live/<날짜>/geo
# 드론 없이 시험: 동영상 파일로
& $py -m litter.live stream --source test.mp4 --weights best.pt
```

주의:
- **시계 맞추기**: 핑 시각(노트북)과 SRT 시각(드론)이 다르면 위치가 틀린다. 비행 전 노트북 시계 화면을
  드론으로 몇 초 찍어 두고 차이를 `--offset`으로 준다 (초속 5 m 비행 시 1초 = 5 m).
- **방향(yaw)**: Mini 계열 SRT에는 방향이 없을 수 있다 → 앞으로 비행 중이면 GPS 이동방향으로 추정,
  호버링 중이면 모름. 탐지 비행은 짐벌 -90° + 직진 비행이 가장 정확.

## 업체 데이터 받으면 확인할 것 → 필요한 인자

| 확인 | 있으면 | 없으면 |
|---|---|---|
| 라벨이 **폴리곤**인가 | 그대로 | bbox만 → 면적 과대추정 경고. SAM으로 마스크화 추가 예정 |
| 무게가 **물체별 id**로 있나 | `--weights` (id 열: ann_id / id / item_id / object_id) | 합계만 → 물체별 학습 불가, 구역 단위로 |
| 무게 CSV 열 이름 | weight_kg / weight / 무게 / weight_g 자동 인식 | 다르면 `ingest._WEIGHT_KEYS`에 추가 |
| **수거 지점 GPS** (gt_lat, gt_lon) | 위치 오차 비교 그림 자동 생성 | 발표용 "50 m → ○ m" 숫자를 못 냄 — 꼭 요청 |
| 사진 EXIF/XMP에 짐벌각·상대고도 | 자동 | 텔레메트리 표 요청 또는 `video` 명령으로 SRT |
| 겹쳐 찍힌 연속 사진 | ODM으로 DSM 만들고 `--dsm` | 2D 경로로만 (무게는 면적 기반) |
| 라벨이 정사영상 위에 | `--ortho ortho.tif` | — |
| 클래스 이름 | `config.CLASSES[*].aliases`에 부분일치로 매핑 | 매핑 안 되면 other → aliases 추가 |
