"""
카메라 모델 — 드론 사진의 픽셀 ↔ 실제 지면 좌표.

"위치가 50 m 어긋난다" 문제의 핵심이 여기다. 흔한 방식은
**드론의 GPS를 쓰레기 위치로 기록**하는 것인데, 고도 40 m에서
비스듬히(피치 -45°) 찍으면 화면 중앙만 해도 드론 바로 아래에서
40 m 앞이다. 이 모듈은 짐벌 각도·방향·고도로 **픽셀마다 광선을 쏴서
지면과 만나는 점**을 계산한다 (DSM이 있으면 DSM과, 없으면 평면과).

좌표계: 지도 좌표 (x=동, y=북, z=위, 단위 m — UTM 등 미터 CRS)
각도 규약: DJI와 같음
    yaw   : 북쪽 기준 시계방향 (도). 동쪽 = 90
    pitch : 수평 0, 아래로 수직 = -90
    roll  : 오른쪽으로 기울면 +
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    x: float            # 드론 위치 (지도 좌표, m)
    y: float
    alt: float          # 지면 기준 고도 (m) — DJI RelativeAltitude(이륙지점 기준)면 해변에선 거의 같음
    yaw: float
    pitch: float
    roll: float = 0.0
    ground_z: float = 0.0   # alt의 기준이 되는 지면 높이 (DSM과 같은 기준)


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_f35(cls, width, height, f35_mm):
        """35mm 환산 초점거리로 (EXIF FocalLengthIn35mmFilm). 36 mm 폭 기준."""
        fx = f35_mm / 36.0 * width
        return cls(width, height, fx, fx, width / 2, height / 2)

    @classmethod
    def from_hfov(cls, width, height, hfov_deg):
        fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return cls(width, height, fx, fx, width / 2, height / 2)


def rotation(pose):
    """world_from_cam 회전행렬. 카메라 좌표: x=오른쪽, y=아래, z=앞."""
    yaw, pitch = np.radians(pose.yaw), np.radians(pose.pitch)
    f = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    d = np.cross(f, r)
    if pose.roll:
        a = np.radians(pose.roll)
        r, d = r * np.cos(a) + d * np.sin(a), -r * np.sin(a) + d * np.cos(a)
    return np.stack([r, d, f], axis=1)


def camera_center(pose):
    return np.array([pose.x, pose.y, pose.ground_z + pose.alt])


def pixel_rays(K, pose, uv):
    uv = np.atleast_2d(np.asarray(uv, float))
    cam = np.stack([(uv[:, 0] - K.cx) / K.fx, (uv[:, 1] - K.cy) / K.fy, np.ones(len(uv))], axis=1)
    return cam @ rotation(pose).T


def pixel_to_ground(K, pose, uv, dsm=None, n_iter=6):
    """픽셀 → 지면 (N,3). dsm: 호출 가능한 z(x, y) 샘플러(없으면 평면 ground_z).
    광선이 하늘을 향하면 NaN."""
    C = camera_center(pose)
    rays = pixel_rays(K, pose, uv)
    z = np.full(len(rays), pose.ground_z, float)
    for _ in range(n_iter if dsm is not None else 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - C[2]) / rays[:, 2]
        t[(t <= 0) | ~np.isfinite(t)] = np.nan
        P = C + rays * t[:, None]
        if dsm is None:
            break
        z_new = dsm(P[:, 0], P[:, 1])
        z = np.where(np.isfinite(z_new), z_new, z)
    return P


def ground_to_pixel(K, pose, P):
    """지면 (N,3) → 픽셀 (N,2), 카메라 뒤는 NaN. 합성 데이터 생성용."""
    P = np.atleast_2d(P)
    cam = (P - camera_center(pose)) @ rotation(pose)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * cam[:, 0] / cam[:, 2] + K.cx
        v = K.fy * cam[:, 1] / cam[:, 2] + K.cy
    bad = cam[:, 2] <= 0
    u[bad] = np.nan
    v[bad] = np.nan
    return np.stack([u, v], axis=1)


def georef_uncertainty(K, pose, uv, sig, dsm=None, rng=None):
    """텔레메트리 오차를 몬테카를로로 흘려서 위치 불확실성 반경(95%)을 구한다.
    sig: config["georef"] (gps/alt/yaw/pitch 1σ). 반환: (평균 xy, r95_m)."""
    rng = rng or np.random.default_rng(0)
    n = sig["n_mc"]
    # 표본 n개를 한 번에 (rotation()과 같은 식을 벡터로)
    yaw = np.radians(pose.yaw + rng.normal(0, sig["yaw_sigma_deg"], n))
    pit = np.radians(pose.pitch + rng.normal(0, sig["pitch_sigma_deg"], n))
    alt = np.maximum(1.0, pose.alt + rng.normal(0, sig["alt_sigma_m"], n))
    f = np.stack([np.sin(yaw) * np.cos(pit), np.cos(yaw) * np.cos(pit), np.sin(pit)], 1)
    r = np.stack([np.cos(yaw), -np.sin(yaw), np.zeros(n)], 1)
    d = np.cross(f, r)
    if pose.roll:
        a = np.radians(pose.roll)
        r, d = r * np.cos(a) + d * np.sin(a), -r * np.sin(a) + d * np.cos(a)
    u, v = np.asarray(uv, float)[0]
    ray = r * (u - K.cx) / K.fx + d * (v - K.cy) / K.fy + f
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -alt / ray[:, 2]
    t[t <= 0] = np.nan
    pts = np.stack([pose.x + rng.normal(0, sig["gps_sigma_m"], n) + ray[:, 0] * t,
                    pose.y + rng.normal(0, sig["gps_sigma_m"], n) + ray[:, 1] * t], 1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < n // 2:
        return np.array([np.nan, np.nan]), np.inf
    c = pts.mean(axis=0)
    return c, float(np.percentile(np.linalg.norm(pts - c, axis=1), 95))
