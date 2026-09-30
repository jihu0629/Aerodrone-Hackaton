# litter3d 코드 구조와 사용법

저장소: https://github.com/jihu0629/Aerodrone-Hakaton (main) · 테스트 17개 통과

## 설치 (Windows, VS Code 터미널)
```
git clone https://github.com/jihu0629/Aerodrone-Hakaton.git
cd Aerodrone-Hakaton
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests -q
```
선택: `pip install ultralytics` (YOLO-seg), `pip install ortools` (경로 최적화, 없으면 greedy)

## 모듈
| 모듈 | 역할 |
|---|---|
| drone.py | Mini 5 Pro 스펙, 고도↔GSD, 촬영 간격·비행선 간격 |
| srt.py | DJI .SRT 자막 파서 (Mini 5 Pro 형식 + 구형 GPS(...) 형식) |
| frames.py | 영상 → 이동거리 기준 프레임 추출(전방 겹침 유지), 흐림 제거, ODM geo.txt |
| dataset.py | 라벨 형식 자동 판별(COCO·YOLO·LabelMe·AI Hub식 JSON·VOC) → YOLO-seg 변환 → train/val 분할 |
| classes.py | 17 클래스, 겉보기 밀도표(출처/가정값), 개당 평균무게, 소형 규칙, 라벨 별칭 매핑 |
| segment.py | COCO→YOLO-seg, YOLO 학습·타일 추론·중복 병합, 색 기반 베이스라인(Kako 2020 방식), 마스크 PNG |
| reconstruct.py | ODM docker 명령/실행, COLMAP 명령, DSM·정사영상 읽기, 스케일 검증 |
| volume.py | 링 픽셀 강건 평면으로 바닥 추정 → Σ max(DSM−바닥,0)·GSD², 불확실성 σ_z·면적 |
| mass.py | V × ρ_app (최소/대표/최대); 유리·금속·미확인은 개수×평균무게; 소형은 면적×0.4 cm×1.2 g/cm³; 식생 제외 |
| gridmap.py | 격자 kg (CSV·PNG·GeoTIFF) |
| plan.py | 마대·톤백·트럭(무게/부피 중 먼저 차는 쪽), NIOSH 23 kg 초과 물체, 인·시간, CVRP 경로 |
| report_html.py | 결과 HTML 리포트 (오프라인, 다크모드, 툴팁, 표 보기) |
| pipeline.py | 전체 실행 → objects.csv, grid_kg.*, plan.json, summary.md, overlay.jpg, report.html |
| synthetic.py / leirosa.py / baselines.py | 합성 검증 장면 / Andriolo 2024 실측 표 / 기존 방식 W1·W2·W3 |

## 스크립트
| 스크립트 | 용도 |
|---|---|
| 10_flight_design.py | 고도별 GSD 표, 목표 GSD 촬영 계획 |
| 11_extract_frames.py | 영상(+SRT) → 프레임 + geo.txt |
| 12_reconstruct.py | ODM 으로 DSM + 정사영상 (docker 필요, `--dry` 로 명령만) |
| 13_estimate_mass.py | DSM + 정사영상 (+마스크 PNG / YOLO 가중치) → 무게·격자·수거계획·리포트 |
| 14_synthetic_demo.py | 합성 장면으로 전체 파이프라인 검증 |
| 15_leirosa_validation.py | Andriolo 2024 실측 데이터로 검증 |

## 실제 데이터 흐름
1. `python scripts/11_extract_frames.py DJI_0001.MP4 --out outputs/frames` (같은 이름의 .SRT 자동 인식)
2. `python scripts/12_reconstruct.py outputs/frames --project outputs/odm --gsd 0.5`
3. Colab 노트북으로 YOLO-seg 학습 → best.pt
4. `python scripts/13_estimate_mass.py --dsm outputs/odm/odm_dem/dsm.tif --ortho outputs/odm/odm_orthophoto/odm_orthophoto.tif --weights best.pt`
5. `outputs/mass/report.html` 을 브라우저로 열기

## 부피 계산 원리
- 정사영상 폴리곤을 DSM 에 그대로 겹친다 (좌표가 같다).
- 바닥 = 마스크 바깥 링(간격 1 px, 두께 4 px) 픽셀에 평면 z = ax + by + c 를 최소제곱으로 맞춤. 잔차 2σ 밖은 제거(옆 물체·돌 대응).
- 부피 = Σ max(DSM − 바닥, 0) × GSD². 속 빈 물체도 겉 부피 (Kako 2020 과 동일, 청소 계획에 실용적).
- 불확실성 = 링 잔차 σ × 면적.
- 검증: 합성 장면(노이즈 0.5 cm)에서 10 L 이상 오차 1 % 안, 소형 5 % 안.
