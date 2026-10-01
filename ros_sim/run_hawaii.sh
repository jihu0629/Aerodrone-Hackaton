#!/bin/bash
# 하와이 칩 월드 위를 하향 카메라 드론이 커버리지 경로로 비행하며 촬영한다.
# 먼저: python ros_sim/hawaii/build_world.py  (generated/ 생성)
# 실시간 화면: http://localhost:6080/vnc.html
# 결과: ros_sim/out_hawaii/ (flight_log.csv, frames/*.png)
set -e
cd "$(dirname "$0")"
[ -f hawaii/generated/meta.json ] || { echo "hawaii/generated 없음 — build_world.py 먼저 실행"; exit 1; }
docker rm -f aerodrone-px4-hawaii >/dev/null 2>&1 || true
rm -rf out_hawaii && mkdir -p out_hawaii
docker run -d \
  -p 6080:6080 \
  -e CRUISE_SPEED_MPS="${CRUISE_SPEED_MPS:-5}" \
  -v "$(pwd)/hawaii/generated:/workspace/sim:ro" \
  -v "$(pwd)/hawaii/models:/workspace/hawaii_models:ro" \
  -v "$(pwd)/out_hawaii:/workspace/out" \
  --name aerodrone-px4-hawaii \
  aerodrone-px4-sim hawaii
echo "실행 중: docker logs -f aerodrone-px4-hawaii"
