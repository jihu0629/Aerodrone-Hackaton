#!/bin/bash
# PX4 SITL + Gazebo GUI를 가상 디스플레이로 띄우고 noVNC로 웹에 공개한다.
# 뜨면 브라우저에서 http://localhost:6080/vnc.html 로 실시간 3D 화면을 볼 수 있다.
# 미션까지 바로 돌리려면: ./run_gui.sh mission
set -e
cd "$(dirname "$0")"
MODE="gui"
[ "$1" == "mission" ] && MODE="gui-mission"

mkdir -p out
docker run --rm -it \
  -p 6080:6080 \
  -v "$(pwd)/out:/workspace/out" \
  --name aerodrone-px4-gui \
  aerodrone-px4-sim "$MODE"
