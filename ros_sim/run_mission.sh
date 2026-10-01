#!/bin/bash
# PX4 SITL 컨테이너를 띄우고 미션을 실행한 뒤, 비행 로그(flight_log.csv)를
# 호스트의 ./out 폴더로 가져온다.
set -e
cd "$(dirname "$0")"
mkdir -p out

docker run --rm -it \
  -v "$(pwd)/out:/workspace/out" \
  --name aerodrone-px4-mission \
  aerodrone-px4-sim mission

echo "비행 로그: ros_sim/out/flight_log.csv"
