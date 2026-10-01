"""
1단계: 맵핑용 커버리지 경로 (boustrophedon / 왕복 경로).

드론이 일정 고도에서 일정 폭(swath)으로 사진을 찍으며 지그재그로
지역 전체를 훑는, 항공측량에서 표준으로 쓰이는 패턴. 카메라 화각과
고도로부터 한 줄에 찍을 수 있는 폭을 계산하고, 줄 간격은 오버랩을
고려해 그보다 좁게 잡는다.
"""
import math
from dataclasses import dataclass


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    phase: str          # "mapping" | "orbit" | "transit" | "collection"
    note: str = ""


def swath_width(altitude_m: float, fov_deg: float = 84.0) -> float:
    """카메라 화각(FOV)과 고도로부터 한 줄에 찍히는 지상 폭(m) 계산."""
    return 2 * altitude_m * math.tan(math.radians(fov_deg) / 2)


def boustrophedon_path(x0, y0, x1, y1, altitude_m: float,
                        fov_deg: float = 84.0, overlap: float = 0.3) -> list[Waypoint]:
    """직사각형 영역 (x0,y0)-(x1,y1)을 왕복(지그재그)으로 덮는 경로 생성.

    줄은 x축 방향으로 날고, y방향으로 줄 간격만큼씩 이동한다.
    overlap은 인접 줄 사이 중복 촬영 비율 (정합 품질을 위해 통상 20~40%).
    """
    width = swath_width(altitude_m, fov_deg)
    line_spacing = width * (1 - overlap)
    if line_spacing <= 0:
        raise ValueError("overlap이 너무 커서 줄 간격이 0 이하입니다")

    n_lines = max(1, math.ceil((y1 - y0) / line_spacing) + 1)
    waypoints: list[Waypoint] = []
    for i in range(n_lines):
        y = min(y0 + i * line_spacing, y1)
        if i % 2 == 0:
            xs = (x0, x1)
        else:
            xs = (x1, x0)
        waypoints.append(Waypoint(xs[0], y, altitude_m, "mapping", f"line {i} start"))
        waypoints.append(Waypoint(xs[1], y, altitude_m, "mapping", f"line {i} end"))
    return waypoints


def path_length(waypoints: list[Waypoint]) -> float:
    total = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        total += math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
    return total
