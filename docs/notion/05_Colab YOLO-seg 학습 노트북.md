# Colab YOLO-seg 학습 노트북

파일: `notebooks/colab_train_yoloseg.ipynb` (GitHub main)

## 여는 법
1. Google Colab → 파일 → 노트 업로드 (또는 GitHub 탭에서 저장소 주소 입력)
2. 런타임 → 런타임 유형 변경 → **T4 GPU**
3. 위에서부터 Shift+Enter. `# ✏️` 표시 줄만 상황에 맞게 수정

## 단계
| 단계 | 내용 |
|---|---|
| ① GPU 확인 | nvidia-smi, torch CUDA |
| ② 코드 받기 | git clone + pip install ultralytics |
| ③ Drive 연결 | DATA_DIR(라벨 데이터), WORK_DIR(가중치 저장) 경로 지정. zip 이면 /content 로 해제 |
| ④ 라벨 형식 자동 점검 | coco / yolo / labelme / per_image_json / voc 판별, 라벨 파일 앞부분 출력 |
| ⑤ 변환·분할 | 우리 17 클래스로 자동 매핑(CLASS_MAP 으로 수정 가능), YOLO-seg 형식, train/val 8:2, 라벨 오버레이 확인 |
| ⑥ 학습 | yolo11s-seg, imgsz 1024, 회전·상하반전 증강, 조기종료, **RESUME=True** 로 이어서 학습 |
| ⑦ 평가 | 클래스별 정밀도·**재현율**·mAP50 (무게 추정에는 재현율이 핵심) |
| ⑧ 예측 예시 | val 3장 |
| ⑨ 저장 | best.pt → Drive |
| 부록 A | 큰 사진(8192×6144) 1024 타일 분할 + 라벨 자르기 |
| 부록 B | 색 기반 베이스라인(Kako 2020) 과 픽셀 IoU 비교 |

## 주의
- 실제 GPU·데이터로 전체를 돌려보지 않았다. 학습 셀에서 오류가 나면 메시지를 그대로 공유.
- ④ 가 unknown 이거나 ⑤ 클래스 개수가 이상하면 라벨 파일 앞부분을 공유 → 변환 코드 수정.
- 데이터가 적으면 epochs 100+, 메모리 부족이면 batch 4.
