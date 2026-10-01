#!/bin/bash
# PX4 SITL + MAVROS를 백그라운드로 띄운 채 디버깅용 쉘을 연다.
# 컨테이너 안에서 `ros2 topic echo /mavros/state` 등으로 연결 상태 확인 가능.
set -e
cd "$(dirname "$0")"
docker run --rm -it \
  --name aerodrone-px4-shell \
  aerodrone-px4-sim shell
