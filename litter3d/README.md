# litter3d — 드론 영상 → 해안쓰레기 부피·무게 → 수거 계획

DJI Mini 5 Pro 영상(또는 사진)에서 해안쓰레기를 종류별로 분할하고, 3D 복원(DSM)으로 개별 부피를 재고,
종류별 **겉보기 밀도**로 무게를 구한 뒤 10 m 격자 지도와 수거 계획(마대·인력·차량·경로)까지 만든다.

```
영상 + .SRT ──11_extract_frames──▶ 프레임 + geo.txt ──12_reconstruct(ODM)──▶ DSM + 정사영상
                                                                                  │
      기업 라벨 데이터 ──segment.coco_to_yolo_seg──▶ YOLO-seg 학습 ─────────────────┤
                                                                                  ▼
                       13_estimate_mass ──▶ objects.csv · grid_kg.png · plan.json · summary.md
```

## 빠른 확인 (데이터 없이)

```bash
pip install -r requirements.txt
python scripts/10_flight_design.py --gsd 0.5     # Mini 5 Pro 고도별 GSD, 촬영 계획
python scripts/14_synthetic_demo.py              # 합성 장면으로 부피 오차·무게·수거계획 확인
pytest tests/test_litter3d.py -q
```

## 실제 데이터 흐름

| 단계 | 명령 | 필요한 것 |
|---|---|---|
| 1. 프레임 추출 | `python scripts/11_extract_frames.py DJI_0001.MP4 --out outputs/frames` | 영상 + 같은 이름의 `.SRT` (DJI Fly '영상 자막' 켜기) |
| 2. 3D 복원 | `python scripts/12_reconstruct.py outputs/frames --project outputs/odm --gsd 0.5` | docker + OpenDroneMap (`--dry` 로 명령만 출력) |
| 3. 분할 학습 | **Colab**: `notebooks/colab_train_yoloseg.ipynb` (형식 점검→변환→학습→평가→best.pt 저장) | Google Drive 에 라벨 데이터, T4 GPU |
| 4. 무게·계획 | `python scripts/13_estimate_mass.py --dsm ... --ortho ... --weights best.pt` → `outputs/mass/report.html` 을 브라우저로 열기 | 2·3 결과 |

마스크 PNG(클래스 인덱스)가 이미 있으면 `--mask masks.png`, 둘 다 없으면 색 기반 베이스라인(Kako 2020 방식)으로 돌아간다.

## 모듈

| 모듈 | 내용 |
|---|---|
| `drone.py` | Mini 5 Pro 스펙(8192×6144, 대각 84°, 24 mm 환산), 고도↔GSD, 촬영 간격·비행선 간격 |
| `srt.py` | DJI `.SRT` 자막 파서 (위도·경도·rel_alt·abs_alt·focal_len). Mini 5 Pro 형식 + 구형 `GPS(...)` 형식 |
| `frames.py` | 이동 거리 기준 프레임 추출(전방 겹침 유지), 흐림 제거, ODM `geo.txt` |
| `classes.py` | 17 클래스(AI Hub 12종 + 나무·섬유·고무·도자기 + unknown), 겉보기 밀도표 ρ_min/typ/max, 개당 평균무게(W1), 소형 대체 규칙, 라벨 별칭 매핑 |
| `segment.py` | COCO→YOLO-seg 변환, YOLO 학습/추론(타일링+중복 병합), 색 기반 베이스라인, 마스크 PNG 입출력 |
| `reconstruct.py` | ODM docker 명령/실행, COLMAP 명령, DSM·정사영상 읽기, 스케일 검증 |
| `volume.py` | 링 픽셀 강건 평면 맞춤으로 바닥 추정 → Σ max(DSM−바닥,0)·GSD², 불확실성 σ_z·면적 |
| `mass.py` | m = V·ρ_app (최소/대표/최대). 유리·금속·미확인은 개수×평균무게, 소형은 면적×0.4 cm×1.2 g/cm³(W3) 또는 클래스 소형무게, 식생 제외, 젖음 계수 |
| `leirosa.py` · `baselines.py` | Andriolo 2024 실측 표(1,505개·24,720 g)와 기존 방식 W1/W2/W3 구현 → `scripts/15_leirosa_validation.py` 로 검증 |
| `gridmap.py` | 격자 kg (CSV·PNG·GeoTIFF) |
| `report_html.py` | 결과를 한 장의 HTML 리포트로 (총 무게·타일·검출 오버레이·종류별 막대·격자 히트맵·기존 방식 비교·수거 계획·물체 표). 외부 라이브러리·인터넷 불필요, 다크 모드·툴팁·표 보기 지원 |
| `plan.py` | 마대·톤백·트럭(무게 vs 부피 중 먼저 차는 쪽), NIOSH 23 kg 초과 물체, 인·시간, CVRP 경로(OR-Tools 있으면 사용, 없으면 greedy) |
| `synthetic.py` | 경사 모래 + 크기를 아는 상자/원기둥/돔 합성 장면 |
| `dataset.py` | 라벨 형식 자동 판별(COCO·YOLO·LabelMe·AI Hub식 JSON·VOC) → YOLO-seg 변환 → train/val 분할 |

## 숫자의 신뢰도

- **출처 있음**: EPS 밀도 11–32 kg/m³(Wikipedia Polystyrene), 개당 평균무게(Andriolo 2024 OSPAR, Andriolo & Gonçalves 2024 중앙값 13.4 g), NIOSH 23 kg, Mini 5 Pro 스펙(cined·dpreview·drdrone 스펙표).
- **가정값**: PET·부표·로프·그물·유리·금속의 겉보기 밀도, 젖음 계수, 마대·톤백·트럭 적재량, 작업 속도, 소형 기준(5×5 cm, 높이 3 cm). `classes.py`·`plan.py` 에 "가정값" 으로 표시돼 있고, 보정 실험(실물 5–10개 저울 + 3D 부피) 결과로 교체해야 한다.
- 합성 데모의 "기준 kg" 는 실제 부피 × 같은 ρ_typ 이라 밀도 오차는 검증하지 않는다. 부피 코드 오차만 본다.
- 결과는 검출 누락을 반영하지 않은 **최소 추정치**다. 재현율을 같이 보고한다.

## 검증 결과 (docs/validation_leirosa.md)

Andriolo et al. (2024) 의 실측 구성으로 시뮬레이션: 우리 3D 방식 +30 % (개선 전 +171 %), W2 −18 %, 재료 비중을 곱하는 W3' +2,700 %.
검출 누락(1 L 미만 미검출)을 넣으면 −10 %. 부피 코드 오차는 1 % 안이고 남은 오차는 전부 밀도표(가정값)에서 온다. 치수 가정·순환 항목은 문서에 명시.

## 확인 필요

- Mini 5 Pro 실제 초점거리(mm)·센서 크기(mm), 4K 영상 크롭 배율 (GSD 계산은 화각 84° 만 사용)
- 실제 `.SRT` 파일 형식 (테스트는 공개된 예시 1건 기준)
- 기업 데이터의 라벨 포맷·클래스명 → `classes._ALIASES` 에 추가
