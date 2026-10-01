# 실제 시뮬레이션 엔진 연동 (PX4 SITL + Gazebo + ROS2/MAVROS)

> 현재 검증 기록은 선회 경로가 포함된 이전 미션(커밋 `cffc25f`의 `archive/path_planning_v1/`, 지금은 삭제됨)으로
> 비행한 결과다. 지금 방향(핫스팟 우선 + 커버리지)의 경로로 다시 돌리려면
> `mission_runner/mission.json`의 `mapping_orbit_path`를 새 경로로 바꾸면 된다
> (형식: x·y·z 미터, phase, note 리스트).

[path_planning](../path_planning)에서 만든 경로를 Three.js 애니메이션이
아니라 **실제 비행 다이나믹스를 흉내 내는 물리엔진** 위에서 그대로
실행해 검증한다. PX4는 실제 비행 컨트롤러 펌웨어이고, Gazebo는 중력·
관성·모터 반응 지연까지 흉내 내는 물리 시뮬레이터라서, "이 경로가 실제
드론에서도 말이 되는가"를 Three.js 재생보다 훨씬 신뢰할 수 있게
확인해준다.

맥(Apple Silicon)은 ROS/Gazebo 네이티브 설치가 불안정해서 전부 Docker
컨테이너 안에서 돌린다.

## 구성

- **PX4 SITL** (v1.14.3) — 실제 비행 컨트롤러 펌웨어를 가상 환경에서
  그대로 실행. 소스에서 빌드.
- **Gazebo Garden (gz-sim7)** — 물리엔진(중력, 관성, 모터 반응).
  기본은 GUI 없이 headless로 실행(검증 자체엔 화면이 필요 없어서),
  필요하면 `run_gui.sh`로 noVNC를 통해 브라우저에서 실시간 3D 화면도
  볼 수 있다 (아래 "실시간 3D 화면 보기" 참고).
- **ROS2 Iron + MAVROS** — PX4의 MAVLink 텔레메트리/명령을 ROS2
  토픽/서비스로 변환. (Humble은 arm64용 MAVROS 바이너리가 없어서 Iron 사용.)
- **mission_runner** (직접 작성한 ROS2 노드) — 이전 버전 미션 생성기
  (이전 `path_planning/mission.py`)가 만든 `mission.json`의 `mapping_orbit_path`(맵핑+궤도 촬영 구간)를
  `/mavros/setpoint_position/local`에 순서대로 흘려보내고, 실제로
  날아간 위치를 `flight_log.csv`로 기록.

  `collection_path`(수거 트럭의 지상 이동 경로)는 드론이 아니라
  트럭의 경로이므로 여기서는 비행시키지 않는다 — 이미 TSP 알고리즘
  자체로 검증된 부분이라, 물리엔진 검증이 필요한 건 드론이 실제로
  나는 맵핑+궤도 구간뿐이다.

## 좌표계

`path_planning`에서 만든 (x, y, z)는 각각 (동쪽, 북쪽, 고도)로
설계되어 있어서 MAVROS의 local ENU 프레임(x=East, y=North, z=Up)과
1:1로 그대로 맞는다 — 별도 좌표 변환 없이 바로 세트포인트로 흘려보낼
수 있다.

## 사용법

```bash
# 1) 이미지 빌드 (PX4를 소스에서 빌드하므로 처음 한 번은 오래 걸림, 10~20분)
docker build -t aerodrone-px4-sim .

# 2) 미션 실행 (끝나면 ros_sim/out/flight_log.csv 로 비행 로그 저장)
./run_mission.sh

# 디버깅용: PX4+MAVROS만 띄우고 쉘 진입
./run_shell.sh
# (쉘 안에서) ros2 topic echo /mavros/state
#             ros2 run mission_runner fly_mission
```

### 실시간 3D 화면 보기 (noVNC)

호스트에 아무것도 설치하지 않고, 컨테이너 안에 가상 디스플레이
(Xvfb) + VNC 서버(x11vnc) + 웹소켓 브리지(noVNC)를 띄워서 브라우저
하나로 Gazebo의 실제 3D 화면을 본다.

```bash
./run_gui.sh            # GUI만 (미션은 나중에 docker exec로 수동 실행)
./run_gui.sh mission    # GUI + 미션 바로 실행
```

뜨면 브라우저에서 **http://localhost:6080/vnc.html** 열기.

신형 Gazebo(gz-sim)는 서버/GUI가 분리된 구조라서, PX4가 띄우는
물리엔진 서버(`gz sim -s`, 창 없음)에 별도로 GUI 클라이언트
(`gz sim -g`)를 붙이는 방식으로 구현했다 — PX4 쪽 `HEADLESS` 환경변수는
이 신형 타겟(`gz_x500`)에는 적용되지 않는다는 걸 소스를 뒤져서 확인하고
알게 된 부분.

## 검증 방법

`flight_log.csv`에는 매 틱마다 `target_x/y/z`(계획된 좌표)와
`actual_x/y/z`(PX4가 실제로 보고한 위치), 그리고 둘 사이 거리가
기록된다. 이 거리가 허용 오차(`WAYPOINT_TOLERANCE_M = 1.3m`) 안에서
계속 좁혀지며 모든 웨이포인트를 통과하면 "이 경로는 실제 비행
컨트롤러로도 문제없이 날 수 있다"는 뜻이고, 특정 구간(특히 궤도
촬영처럼 급격한 방향 전환이 많은 구간)에서 반복적으로 타임아웃이
나면 그 구간의 경로가 드론이 실제로 따라가기엔 너무 촘촘하거나
급하다는 신호 — 이전 `orbit.py`의 반지름/뷰 개수를 조정해야
한다는 뜻이다.

## 검증 결과 (실제로 돌려본 기록)

110개 웨이포인트(맵핑 + 7개 물체 궤도 촬영 전부) 완주, **타임아웃 0건**
— 즉 모든 웨이포인트를 실제로 허용오차(1.5m) 안까지 도달했다. 시뮬
비행 시간 약 7.3분(434.9초). 로그: `ros_sim/out/flight_log.csv`
(매 틱 계획 좌표 vs 실제 좌표 vs 거리).

### 실제로 겪은 버그 두 가지

1. **모터가 추력을 전혀 안 냄** — gz-sim7(Gazebo Garden)에 설치된
   `MulticopterMotorModel` 플러그인이 `robotNamespace`가 없으면
   "Please specify a robotNamespace" 경고만 남기고 모터 속도 명령
   토픽 구독에 조용히 실패한다. PX4는 정상적인 모터 속도 명령(초당
   600~1000 rad/s, 이론상 호버링에 충분한 값)을 계속 내보내고
   있었는데도 기체는 그냥 땅에 파묻힌 채 꿈쩍도 안 했다 — 겉보기엔
   "명령은 나가는데 안 움직이는" 가장 헷갈리는 종류의 버그였다.
   `gz topic -e`로 모터 명령 토픽을 직접 찍어보고서야 "명령은 정상,
   물리엔진에 적용이 안 됨"이라는 걸 구분해낼 수 있었다. 해결:
   x500 계열 모델 SDF의 로터 플러그인마다 `<robotNamespace>`를
   명시적으로 추가(Dockerfile에 영구 반영).

2. **뜨긴 뜨는데 통제불능으로 날아감** — 위 버그를 고친 직후엔
   날긴 날았지만, 세트포인트를 목표 좌표로 그대로 "순간이동"시키는
   방식이라 다음 웨이포인트까지 거리가 크면(예: 이륙 직후 30m 상공)
   PX4 위치제어기가 큰 오차를 단번에 따라잡으려다 자세가 무너지고,
   그 와중에 25초 타임아웃으로 또 다음 목표로 넘어가버리면서 오차가
   누적돼 피치 40도 이상 기울어진 채 경로를 완전히 이탈했다 (목표는
   x=0인데 기체는 x=80까지 가속도가 붙은 채 날아감). 해결: 세트포인트를
   목표로 바로 꽂지 않고 초당 최대 속도(맵핑 4m/s, 궤도 촬영 1.5m/s)
   제한을 두고 목표 쪽으로 서서히 이동시키는 방식으로 변경 — PX4가
   한 번에 받는 위치 오차가 항상 작게 유지되도록 하는, OFFBOARD
   포지션 컨트롤의 표준적인 완화 기법.

## 한계

- PX4 v1.14 + Gazebo Garden(gz-sim7) 조합을 썼다 — 이 버전의
  `ubuntu.sh`가 기본으로 설치하는 조합이며, 구형 Gazebo Classic
  타겟(`gazebo-classic_iris`)은 이 환경에서 빌드되지 않는다.
- 바람, GPS 노이즈 등 환경 교란은 기본 SITL 설정 그대로이며 추가로
  켜지 않았다 — 필요하면 PX4 파라미터로 추가 가능.
- 객체 탐지 트리거는 아직 시뮬레이션에 없다 — `mission.json`은
  미리 계산된 고정 경로이고, 실시간으로 물체를 "발견"해서 경로를
  바꾸는 것은 Module 2(객체탐지)가 실제 카메라 피드와 연결된 뒤에
  이 노드에 붙일 수 있다.
