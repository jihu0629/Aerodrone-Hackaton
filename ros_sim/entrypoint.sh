#!/bin/bash
set -e
source /opt/ros/iron/setup.bash
source /workspace/install/setup.bash

MODE="${1:-default}"
GUI_MODES="gui gui-mission hawaii"
HEADLESS_FLAG="HEADLESS=1"

if [[ " $GUI_MODES " == *" $MODE "* ]]; then
  echo "[entrypoint] 가상 디스플레이(Xvfb+fluxbox) + noVNC 웹서버 기동..."
  export DISPLAY=:99
  Xvfb :99 -screen 0 1280x800x24 > /tmp/xvfb.log 2>&1 &
  sleep 1
  fluxbox > /tmp/fluxbox.log 2>&1 &
  x11vnc -display :99 -forever -shared -nopw -rfbport 5900 > /tmp/x11vnc.log 2>&1 &
  websockify --web=/usr/share/novnc 6080 localhost:5900 > /tmp/novnc.log 2>&1 &
  echo "[entrypoint] 브라우저에서 http://localhost:6080/vnc.html 로 접속하면 실시간 3D 화면이 보임"
  HEADLESS_FLAG="HEADLESS=0"
fi

if [ "$MODE" == "hawaii" ]; then
  # 하와이 칩 월드(/workspace/sim, build_world.py 출력)를 먼저 띄우면 PX4가
  # "이미 떠 있는 월드"에 붙어서 PX4_GZ_MODEL 기체를 생성한다 (PX4 원본 수정 없음).
  # 카메라 렌더링에 GL이 필요해서 위의 Xvfb(DISPLAY=:99)를 그대로 쓴다.
  export GZ_SIM_RESOURCE_PATH=/workspace/sim/models:/workspace/hawaii_models:/opt/PX4-Autopilot/Tools/simulation/gz/models
  mkdir -p /workspace/out/frames
  echo "[entrypoint] 하와이 월드 기동..."
  gz sim -r -s /workspace/sim/worlds/hawaii.sdf > /tmp/gz_server.log 2>&1 &
  for i in $(seq 1 60); do gz topic -l 2>/dev/null | grep -q "/world/hawaii/clock" && break; sleep 1; done
  SPAWN=$(python3 -c "import json;m=json.load(open('/workspace/sim/meta.json'));print(f\"{m['spawn'][0]},{m['spawn'][1]},0.3,0,0,0\")")
  LATLON=($(python3 -c "import json;m=json.load(open('/workspace/sim/meta.json'));print(*m['center_latlon'])"))
  export PX4_GZ_MODEL=x500_down_cam PX4_GZ_MODEL_POSE="$SPAWN" PX4_HOME_LAT=${LATLON[0]} PX4_HOME_LON=${LATLON[1]} PX4_HOME_ALT=2
  export MISSION_JSON=/workspace/sim/${MISSION_FILE:-mission_coverage.json}
fi

echo "[entrypoint] PX4 SITL(gz_x500${DISPLAY:+, GUI on $DISPLAY}) 시작..."
cd /opt/PX4-Autopilot
# stdin을 열어두면 PX4의 nsh 콘솔(pxh>)이 EOF를 반복해서 받으며 프롬프트를
# 무한 재출력하는 바쁜 루프에 빠져 CPU를 독점하고(거의 100%), 그 때문에
# 실제 비행 제어 스레드가 굶주려서 OFFBOARD 세트포인트를 받고도 기체가
# 움직이지 않는 문제가 있었다. stdin을 /dev/null로 명시적으로 닫아야 함.
env $HEADLESS_FLAG make px4_sitl gz_x500 < /dev/null > /tmp/px4_sitl.log 2>&1 &
PX4_PID=$!

echo "[entrypoint] PX4가 MAVLink 포트를 열 때까지 대기..."
for i in $(seq 1 60); do
  if grep -q "INFO  \[mavlink\] partner IP" /tmp/px4_sitl.log 2>/dev/null || \
     (echo > /dev/udp/127.0.0.1/14540) 2>/dev/null; then
    break
  fi
  sleep 1
done
sleep 3

if [[ " $GUI_MODES " == *" $MODE "* ]]; then
  # gz_x500 빌드 타겟은 항상 `gz sim -s`(서버만, 창 없음)로 물리엔진을
  # 띄운다 — PX4 쪽에 HEADLESS=0을 줘도 이 동작은 안 바뀐다(실제로
  # PX4의 gz_bridge 모듈은 gz sim을 직접 실행하지 않고 그냥 연결만
  # 함; 서버를 누가 어떻게 띄우는지는 별개). 그래서 GUI를 보려면 이미
  # 떠 있는 서버에 별도로 `gz sim -g`(클라이언트만)를 붙여야 한다 —
  # 보통 멀티플레이어 느낌으로 서버/클라이언트가 분리된 신형 Gazebo의
  # 구조를 그대로 활용한 것.
  echo "[entrypoint] 이미 떠 있는 gz sim 서버에 GUI 클라이언트 연결..."
  gz sim -g > /tmp/gz_gui.log 2>&1 &
fi
if [ "$MODE" == "hawaii" ]; then
  # gz 카메라는 구독자가 없으면 렌더링을 안 해서 <save> 저장도 멈춘다 → 구독자를 붙여 둔다
  gz topic -e -t /down_cam/image > /dev/null 2>&1 &
fi

echo "[entrypoint] MAVROS 시작..."
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=udp://:14540@127.0.0.1:14557 \
  -p target_system_id:=1 \
  -p target_component_id:=1 \
  < /dev/null > /tmp/mavros.log 2>&1 &
MAVROS_PID=$!

echo "[entrypoint] MAVROS가 FCU에 연결될 때까지 대기..."
for i in $(seq 1 60); do
  STATE=$(ros2 topic echo /mavros/state --once 2>/dev/null | grep "connected: true" || true)
  if [ -n "$STATE" ]; then
    echo "[entrypoint] FCU 연결됨"
    break
  fi
  sleep 1
done

case "$MODE" in
  shell)
    echo "[entrypoint] 쉘 모드 — PX4/MAVROS는 백그라운드에서 계속 실행됨"
    exec /bin/bash
    ;;
  mission)
    echo "[entrypoint] 미션 실행..."
    exec ros2 run mission_runner fly_mission
    ;;
  gui)
    echo "[entrypoint] GUI 모드 — http://localhost:6080/vnc.html 에서 확인, 미션은 'docker exec'로 수동 실행"
    wait $PX4_PID $MAVROS_PID
    ;;
  gui-mission|hawaii)
    echo "[entrypoint] GUI + 미션 실행..."
    exec ros2 run mission_runner fly_mission
    ;;
  *)
    wait $PX4_PID $MAVROS_PID
    ;;
esac
