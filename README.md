# 해안쓰레기 드론 탐지 · 3D 부피 · 수거계획 (aerodrone hackathon)

드론 영상 **하나**(+비행기록 SRT)로 쓰레기를 찾고, 3D로 부피를 재서 무게를 구하고, 인력·마대·트럭·경로가 담긴 **작업카드**까지 자동으로 만든다.
업체의 기존 방식("라벨 면적 × 고정 계수 = 무게")이 실제보다 수십~수백 배 작게 나오는 문제를 **3D 부피 × 재질 밀도**로 고친다.

```
영상(.MP4)+SRT → ①탐지(YOLO11) → ②3D(COLMAP) → ③위치(광선-지면) → ④부피(SAM2 투표+높이지도) → ⑤무게 → ⑥수거계획 → ⑦작업카드
```

## 문서 (여기부터)
| 문서 | 내용 |
|---|---|
| [`docs/파이프라인_결과정리.md`](docs/파이프라인_결과정리.md) | **결과 수치·그림** (상자 부피 오차 −10~+20 %, 업체 방식 비교, 야간 10분 영상, 하와이 적용, 시뮬) |
| [`docs/기술정리_전체.md`](docs/기술정리_전체.md) | 단계별 기술, 다른 접근과 다른 점, 사용/미사용 모델, 할 일 |
| [`docs/발표_스토리라인.md`](docs/발표_스토리라인.md) | 발표 12장 구성·데모 순서·예상 질문 |
| [`docs/HANDOFF_세션인수인계.md`](docs/HANDOFF_세션인수인계.md) | 데이터 위치, 재현 명령, 환경 주의사항 |
| [`docs/해안쓰레기_수거계획_방향정리.md`](docs/해안쓰레기_수거계획_방향정리.md) | 과제 방향 전환 배경 |
| `docs/figures/` | 발표용 그림·수치 json |

## 코드 (`litter/` 패키지)
실행: `.venv\Scripts\python.exe -m litter.<모듈>` (Python 3.12, `requirements.txt`, COLMAP CUDA는 `tools/colmap`에 별도)

| 모듈 | 하는 일 |
|---|---|
| `seg.py` | YOLO 타일 학습/추론/평가 (한글 경로 안전 입출력) |
| `aihub.py`, `ortho.py` | AI Hub 해안쓰레기 데이터 추출·GSD 축소 학습셋 / 업체 정사영상 칩 |
| `orbit3d.py` | 영상 → 프레임 → pycolmap SfM → 뷰어, SRT로 실제 크기 |
| `volume.py` | 바닥면 RANSAC, 축척 자동 선택(SRT 고도 / GPS 경로), 높이지도 부피 |
| `orbit_map.py`, `mosaic.py`, `video_map.py` | 3D 기반 위치·지도·정사영상 / SRT만으로 경량 지도 (`--classes`로 YOLO-World) |
| `objvol.py` | SAM 2 마스크 투표 + 지역 바닥 평면 + 높이지도 → 물체 부피 (v2) |
| `weight.py`, `plan.py`, `report.py`, `config.py` | 무게(밀도·젖음·압축·23 kg 규칙), 수거계획(정거장·2-opt 경로), 작업카드 HTML, 재질표 |
| `fromvideo.py` | ④→⑦ 한 번에 + 업체 방식 비교 그림 |
| `batch3d.py` | 긴 영상의 물체 구간별 3D·조밀·부피 일괄 처리 |
| `sim_ortho.py` | 정사영상 위 가상비행 시뮬 (커버리지·실시간 탐지·능동 재방문) |
| `hawaii.py`, `crosseval.py` | 하와이 공개 해안 데이터 적용 데모 / 현장 교차평가 |
| `synth.py`, `geometry.py`, `camera.py`, `telemetry.py` | 합성 검증 데이터, 광선-지면 기하, 카메라 모델, SRT 파서 |

## 빠른 실행 예 (영상 0007)
```bash
python -m litter.orbit3d  --video ../0007.MP4 --srt ../0007.SRT --n 100 --out runs/orbit/0007
# 조밀 복원: docs/파이프라인_결과정리.md 5절의 COLMAP 빠른 설정 (3분)
python -m litter.orbit_map --orbit runs/orbit/0007 --srt ../0007.SRT --litter runs/seg/aihub_gsd_det_s/weights/best.pt --out runs/map/0007
python -m litter.objvol    --orbit runs/orbit/0007 --srt ../0007.SRT --ply runs/orbit/0007/fused_dense.ply --objects runs/map/0007/objects.csv --out runs/orbit/0007/objvol
python -m litter.fromvideo run --objvol runs/orbit/0007/objvol/objvol.json --objects runs/map/0007/objects.csv --out runs/plan/0007
```

## 저장소에 없는 것
데이터(`data/`, 약 30 GB)·학습 가중치·3D 결과(`runs/`)·외부 도구(`tools/`)는 용량·저작권 때문에 제외. 위치와 재생성 방법은 인수인계 문서 참고.
원래 과제(위성-드론 정합·변화탐지) 코드는 방향 전환 후 저장소에서 제외(로컬 보관).
