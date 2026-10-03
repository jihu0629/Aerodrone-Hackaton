"""
path_planning/mission.py가 만든 맵핑+궤도 촬영 경로를 PX4 SITL(가제보
물리엔진)에 OFFBOARD 포지션 세트포인트로 그대로 흘려보내 실제 비행
다이나믹스로 따라가는지 검증하는 노드.

주의: mission.json의 collection_path(수거 트럭 경로, 고도 0)는 드론이
아니라 트럭의 지상 이동 경로를 나타내므로 여기서는 날리지 않는다.
드론이 실제로 비행하는 구간은 mapping_orbit_path뿐이다.

좌표계: 우리 미션의 (x, y, z)는 각각 (동쪽, 북쪽, 고도)로 설계했고,
MAVROS의 local ENU 프레임(x=East, y=North, z=Up)과 1:1로 그대로
대응되므로 별도 좌표 변환이 필요 없다.
"""
import csv
import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from ament_index_python.packages import get_package_share_directory

WAYPOINT_TOLERANCE_M = 1.5
MAX_SECONDS_PER_WAYPOINT = 90.0  # 정상적으로는 거의 안 걸림 — 교착 상태 탈출용 안전장치
SETPOINT_RATE_HZ = 20.0
OFFBOARD_WARMUP_SETPOINTS = 60  # PX4가 OFFBOARD를 받아들이려면 스트리밍이 먼저 있어야 함
CRUISE_SPEED_MPS = float(os.environ.get('CRUISE_SPEED_MPS', 4.0))   # 맵핑/이동 구간 세트포인트 전진 속도
ORBIT_SPEED_MPS = 1.5    # 궤도 촬영 구간은 더 느리게(사진 품질)


class MissionRunner(Node):
    def __init__(self):
        super().__init__('mission_runner')

        # MISSION_JSON 환경변수로 다른 경로 파일을 줄 수 있다 (예: 하와이 월드)
        mission_path = Path(os.environ.get('MISSION_JSON') or
                            Path(get_package_share_directory('mission_runner')) / 'mission.json')
        mission = json.loads(mission_path.read_text())
        self.waypoints = mission['mapping_orbit_path']
        self.get_logger().info(f'미션 로드: {len(self.waypoints)}개 웨이포인트 (맵핑+궤도만, 수거 경로는 드론 비행 대상 아님)')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

        self.state = State()
        self.pose = None
        self.create_subscription(State, '/mavros/state', self._on_state, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._on_pose, qos)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos)

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        log_dir = Path('/workspace/out')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'flight_log.csv'
        self.log_file = open(log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(['t', 'wall', 'qw', 'qx', 'qy', 'qz', 'target_idx', 'phase', 'note',
                                   'target_x', 'target_y', 'target_z',
                                   'actual_x', 'actual_y', 'actual_z', 'dist_to_target'])
        self.t0 = time.time()

        self.target_idx = 0
        self.setpoints_sent = 0
        self.mode_set = False
        self.armed_requested = False

    def _on_state(self, msg: State):
        self.state = msg

    def _on_pose(self, msg: PoseStamped):
        self.pose = msg.pose.position
        self.orient = msg.pose.orientation

    def _current_target(self):
        w = self.waypoints[self.target_idx]
        return w['x'], w['y'], w['z'], w['phase'], w.get('note', '')

    def _publish_setpoint(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)
        self.setpoints_sent += 1

    def _try_enable_offboard_and_arm(self):
        if not self.mode_set and self.set_mode_client.service_is_ready():
            req = SetMode.Request(custom_mode='OFFBOARD')
            self.set_mode_client.call_async(req)
            self.mode_set = True
            self.get_logger().info('OFFBOARD 모드 요청 전송')
        if not self.armed_requested and self.arming_client.service_is_ready():
            req = CommandBool.Request(value=True)
            self.arming_client.call_async(req)
            self.armed_requested = True
            self.get_logger().info('무장(arm) 요청 전송')

    def run(self):
        rate_dt = 1.0 / SETPOINT_RATE_HZ
        x0, y0, z0, _, _ = self._current_target()

        # 1) OFFBOARD 진입 전 워밍업 스트리밍 (PX4 요구사항)
        for _ in range(OFFBOARD_WARMUP_SETPOINTS):
            self._publish_setpoint(x0, y0, z0)
            rclpy.spin_once(self, timeout_sec=rate_dt)
            time.sleep(rate_dt)

        self._try_enable_offboard_and_arm()

        # 세트포인트를 목표로 "순간이동"시키지 않고, 속도 상한을 둔 채
        # 목표 쪽으로 매 틱 조금씩 이동시킨다. 웨이포인트 사이 거리가 큰데
        # (예: 처음 (0,0,30)까지 30m) 목표를 그대로 꽂아버리면 PX4 위치
        # 제어기가 큰 오차를 한번에 따라잡으려다 자세가 무너지고, 다음
        # 웨이포인트로 넘어가도 따라잡지 못한 채 계속 밀리면서 가속이
        # 누적돼 통제 불능 상태(피치 40도 이상, 경로 이탈)에 빠지는 것을
        # 실제로 겪었다. 이 방식이 실질적인 "최소 변경"으로 고친 고정.
        cur = [x0, y0, z0]

        wp_start_time = time.time()
        while rclpy.ok() and self.target_idx < len(self.waypoints):
            x, y, z, phase, note = self._current_target()
            speed = ORBIT_SPEED_MPS if phase == 'orbit' else CRUISE_SPEED_MPS

            dx, dy, dz = x - cur[0], y - cur[1], z - cur[2]
            remaining = math.dist((0, 0, 0), (dx, dy, dz))
            step = speed * rate_dt
            if remaining > 1e-6:
                f = min(1.0, step / remaining)
                cur[0] += dx * f
                cur[1] += dy * f
                cur[2] += dz * f

            self._publish_setpoint(*cur)
            rclpy.spin_once(self, timeout_sec=rate_dt)

            if not self.state.armed or self.state.mode != 'OFFBOARD':
                self._try_enable_offboard_and_arm()

            if self.pose is not None:
                dist = math.dist((self.pose.x, self.pose.y, self.pose.z), (x, y, z))
                t = time.time() - self.t0
                o = self.orient
                self.log_writer.writerow([f'{t:.2f}', f'{time.time():.3f}', f'{o.w:.5f}', f'{o.x:.5f}', f'{o.y:.5f}', f'{o.z:.5f}',
                                           self.target_idx, phase, note,
                                           f'{x:.2f}', f'{y:.2f}', f'{z:.2f}',
                                           f'{self.pose.x:.2f}', f'{self.pose.y:.2f}', f'{self.pose.z:.2f}',
                                           f'{dist:.2f}'])

                reached = dist < WAYPOINT_TOLERANCE_M
                timed_out = (time.time() - wp_start_time) > MAX_SECONDS_PER_WAYPOINT
                if reached or timed_out:
                    if timed_out and not reached:
                        self.get_logger().warn(
                            f'웨이포인트 {self.target_idx} 타임아웃(거리 {dist:.1f}m) — 다음으로 진행')
                    self.target_idx += 1
                    wp_start_time = time.time()
                    if self.target_idx % 10 == 0:
                        self.get_logger().info(
                            f'진행률 {self.target_idx}/{len(self.waypoints)} ({phase})')

            time.sleep(rate_dt)

        self.get_logger().info('미션 완료 — 착륙')
        self._land()
        self.log_file.close()

    def _land(self):
        if self.set_mode_client.service_is_ready():
            self.set_mode_client.call_async(SetMode.Request(custom_mode='AUTO.LAND'))
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)


def main():
    rclpy.init()
    node = MissionRunner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
