# PX4 SITL(가제보 클래식) + ROS2 Humble + MAVROS
# 맥(Apple Silicon)은 ROS/Gazebo 네이티브 설치가 불안정해서 컨테이너로 돌린다.
# GUI 없이 물리엔진만 띄우는 headless 모드로 실행 — 우리가 검증하려는 건
# "경로를 실제 비행 다이나믹스로 따라가는지"이지, 렌더링 품질이 아니기 때문.
FROM ros:iron-ros-base
# humble은 arm64용 mavros 바이너리 패키지가 없어서(소스만 존재) iron으로 선택.

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl lsb-release sudo gnupg \
    python3-pip python3-colcon-common-extensions \
    ros-iron-mavros ros-iron-mavros-extras \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \
    && bash install_geographiclib_datasets.sh && rm install_geographiclib_datasets.sh

# PX4 소스 (v1.14 계열). ubuntu.sh가 이 버전 기준 신형 Gazebo(gz, 구 Ignition)를
# 설치하므로 구형 gazebo-classic 타겟 대신 gz_x500(쿼드콥터) 타겟을 사용한다.
RUN git clone --branch v1.14.3 --recursive --depth 1 \
    https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot

RUN bash /opt/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx

# gz-sim7(Garden)에 설치된 MulticopterMotorModel 플러그인은 robotNamespace가
# 없으면 "Please specify a robotNamespace" 경고만 찍고 모터 속도 명령
# 구독에 실패해 추력이 전혀 발생하지 않는다(기체가 이륙하지 못하고 그냥
# 땅에 파묻힌 채 있었음). 모든 x500 계열 모델의 로터 플러그인에
# robotNamespace를 명시적으로 추가해 고친다.
RUN for f in /opt/PX4-Autopilot/Tools/simulation/gz/models/x500*/model.sdf; do \
      sed -i 's#<jointName>rotor_\([0-9]\)_joint</jointName>#<robotNamespace>x500_0</robotNamespace>\n      <jointName>rotor_\1_joint</jointName>#' "$f"; \
    done

# gz_x500 타겟은 DONT_RUN=1을 줘도 시뮬레이터를 실제로 띄워버려 빌드 단계가
# 끝나지 않는 문제가 있었다. 대신 시뮬레이터 인자 없이 기본 빌드만 수행 —
# cmake가 설정 단계에서 이미 gz-transport/gz-math 등을 감지하므로 gz_bridge
# 모듈도 이 기본 빌드에 포함된다. 실제 실행은 컨테이너 기동 시 entrypoint에서.
RUN cd /opt/PX4-Autopilot && timeout 1800 make px4_sitl_default

# 웹페이지로 실시간 3D(Gazebo GUI) 화면을 보기 위한 가상 디스플레이 체인:
# Xvfb(가상 화면) -> fluxbox(창 관리자, GUI 앱이 창 없이는 종종 깨짐) ->
# x11vnc(그 화면을 VNC로 송출) -> noVNC/websockify(VNC를 웹소켓으로 변환해
# 브라우저에서 /vnc.html 하나로 열람 가능하게 함). 호스트에 아무것도 설치할
# 필요 없이 컨테이너의 6080 포트만 열면 끝.
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc novnc websockify fluxbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY mission_runner /workspace/src/mission_runner
RUN source /opt/ros/iron/setup.bash \
    && cd /workspace && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
