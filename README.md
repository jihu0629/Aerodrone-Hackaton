# ② 드론 조사 비행 — 핫스팟 우선 경로 + 커버리지 + PX4 실비행 검증

> **파이프라인 ② 단계 브랜치** — 전체 흐름(① 경로 최적화 → ② 비행 → ③ 탐지·3D·무게 → ④ 수거계획)은
> [main 브랜치 README](https://github.com/jihu0629/aerodrone_hackathon) 참고.
> 이 브랜치는 ①에서 고른 우선 구간을 **실제로 어떤 순서·패턴으로 날지** 정하고, 그 경로가 실제 비행 컨트롤러로 날 수 있는지 검증한 개발 이력이다.

## 무엇을 하나

| 폴더 | 역할 |
|---|---|
| [`hotspot/`](hotspot/README.md) | 밀집 구간이 **존재하고 시간이 지나도 유지되는지** 공개 데이터로 검증 (하와이 2015 항공조사, NOAA MDMAP 반복조사) |
| [`path_planning/`](path_planning/README.md) | `hotspot_route.py` 배터리 예산 안에서 가치 합이 최대인 구간·순서 선택(오리엔티어링, 가치/비용 삽입 + 2-opt, UCB 탐색 보너스) · `coverage.py` 화각·고도·오버랩으로 줄 간격을 정한 지그재그 촬영 경로 |
| [`ros_sim/`](ros_sim/README.md) | PX4 SITL + Gazebo + ROS2/MAVROS 로 경로를 실제 비행 다이나믹스 위에서 비행. 하와이 항공사진을 깐 월드에서 하향 카메라로 촬영 |

## 핵심 결과

| 확인한 것 | 결과 |
|---|---|
| 하와이 1 km 격자 집중도 | 쓰레기 있는 칸 상위 10 % 에 라벨 **75 %** |
| NOAA MDMAP 134곳 앞/뒤 기간 | 밀도 순위상관 **0.88**, 앞 기간 상위 20 % 가 뒤 기간 쓰레기의 **56 %** 커버 |
| PX4 실비행 (니하우 해안 칩 40장, 라벨 889개) | 웨이포인트 20/20 완주, 라벨 100 % 가 3장 이상 촬영 |
| 전체 커버리지 vs 핫스팟 우선 (같은 월드) | 비행거리 3,524 m → **732 m (−79 %)**, 라벨 3장 이상 촬영 둘 다 100 %, 1 km 당 확보 라벨 **4.8배** |

> 하와이 월드는 칩이 있는 곳에만 쓰레기가 있어 "핫스팟 밖 누락"은 이 시뮬레이션으로 잴 수 없다(구조상 0).
> 실제 누락 위험은 MDMAP 수치(상위 20 % 재방문으로 56 %)를 근거로 본다.

## 실행

```bash
# 핫스팟 근거 실험
cd hotspot && python3 windward_test.py && python3 mdmap_persistence.py && python3 plot_results.py

# 경로 계획 (파이썬에서)
python -c "import sys; sys.path.insert(0,'path_planning'); import hotspot_route, coverage"

# PX4 실비행 (Docker)
docker build -t aerodrone-px4-sim ros_sim
python ros_sim/hawaii/build_world.py
ros_sim/run_hawaii.sh coverage && python ros_sim/hawaii/evaluate.py --route coverage
ros_sim/run_hawaii.sh hotspot  && python ros_sim/hawaii/evaluate.py --route hotspot
python ros_sim/hawaii/compare_routes.py
```

발표용 정리: [`docs/핫스팟_우선경로_발표정리.md`](docs/핫스팟_우선경로_발표정리.md)
