"""DJI Mini 5 Pro 스펙과 촬영 설계 계산.

GSD(Ground Sampling Distance, 지상표본거리): 사진 1픽셀이 땅에서 차지하는 길이.
부피는 GSD² × 높이 합이므로, GSD 오차 10 % → 부피 오차 약 21 % (GSD² 기준) 이상.

스펙 출처 (2026-09-30 확인)
  - 센서 1인치형 50 MP, 24 mm 환산, 화각 84°, f/1.8 고정: cined.com, dpreview.com, diyphotography.net
  - 최대 사진 8192×6144: drdrone.ca, techpoint.africa
  - 호버링 정확도 비전 수직 ±0.1 m / 수평 ±0.3 m, GNSS ±0.5 m: bhphotovideo.com 스펙표
  - GNSS: GPS + BeiDou + Galileo, L1/L5 이중대역: drdrone.ca
  - 실제 초점거리(mm)·센서 물리 크기(mm)는 공식 스펙에서 직접 확인 필요 → 여기서는 화각으로만 계산.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DroneSpec:
    name: str
    image_w: int          # 사진 가로 픽셀
    image_h: int          # 사진 세로 픽셀
    video_w: int          # 영상 가로 픽셀
    video_h: int
    fov_diag_deg: float   # 대각 화각(도)
    focal_equiv_mm: float # 35 mm 환산 초점거리
    hover_acc_gnss_m: float
    hover_acc_vision_v_m: float
    note: str = ""

    # ---------- 화각 ----------
    def hfov_deg(self, w: int, h: int) -> float:
        """가로 화각. 대각 화각과 종횡비로 계산 (렌즈 왜곡 무시)."""
        d = math.hypot(w, h)
        return 2 * math.degrees(math.atan(math.tan(math.radians(self.fov_diag_deg) / 2) * w / d))

    def vfov_deg(self, w: int, h: int) -> float:
        d = math.hypot(w, h)
        return 2 * math.degrees(math.atan(math.tan(math.radians(self.fov_diag_deg) / 2) * h / d))

    # ---------- 촬영 폭·GSD ----------
    def footprint_m(self, altitude_m: float, mode: str = "photo") -> tuple[float, float]:
        """고도에서 한 장이 덮는 땅 (가로 m, 세로 m). 짐벌 -90°(수직 하향) 기준."""
        w, h = self._wh(mode)
        gw = 2 * altitude_m * math.tan(math.radians(self.hfov_deg(w, h)) / 2)
        gh = 2 * altitude_m * math.tan(math.radians(self.vfov_deg(w, h)) / 2)
        return gw, gh

    def gsd_cm(self, altitude_m: float, mode: str = "photo") -> float:
        """고도 → GSD (cm/px)."""
        w, _ = self._wh(mode)
        gw, _ = self.footprint_m(altitude_m, mode)
        return gw / w * 100.0

    def altitude_for_gsd(self, gsd_cm: float, mode: str = "photo") -> float:
        """목표 GSD (cm/px) → 필요한 고도 (m)."""
        return altitude_m_for(self, gsd_cm, mode)

    def _wh(self, mode: str) -> tuple[int, int]:
        if mode == "photo":
            return self.image_w, self.image_h
        if mode == "video":
            # 가정: 4K 영상이 센서 가로폭 전체를 사용 (16:9 크롭). 실제 크롭 배율은 확인 필요.
            return self.video_w, self.video_h
        raise ValueError(mode)


def altitude_m_for(spec: DroneSpec, gsd_cm: float, mode: str = "photo") -> float:
    w, h = spec._wh(mode)
    hf = math.radians(spec.hfov_deg(w, h))
    return gsd_cm / 100.0 * w / (2 * math.tan(hf / 2))


MINI5PRO = DroneSpec(
    name="DJI Mini 5 Pro",
    image_w=8192, image_h=6144,
    video_w=3840, video_h=2160,
    fov_diag_deg=84.0,
    focal_equiv_mm=24.0,
    hover_acc_gnss_m=0.5,
    hover_acc_vision_v_m=0.1,
    note="RTK 없음. 절대좌표는 ±0.5 m 수준이므로 기준물(크기를 아는 판)로 스케일 검증 필요.",
)


@dataclass
class FlightPlan:
    altitude_m: float
    gsd_photo_cm: float
    gsd_video_cm: float
    footprint_photo_m: tuple[float, float]
    line_spacing_m: float     # 비행선 간격 (측면 겹침)
    shot_interval_m: float    # 촬영 간격 (전방 겹침)
    shot_interval_s: float    # 속도 기준 촬영 시간 간격
    speed_mps: float
    front_overlap: float
    side_overlap: float


def design_flight(target_gsd_cm: float = 0.5, front_overlap: float = 0.8, side_overlap: float = 0.7,
                  speed_mps: float = 2.0, spec: DroneSpec = MINI5PRO) -> FlightPlan:
    """목표 GSD와 겹침으로 촬영 계획을 만든다.

    Andriolo et al. (2023): 해안쓰레기 적정 GSD 0.5–1.25 cm/px, 최소 탐지 2.5–5 cm.
    Kako et al. (2020): 겹침 90 %/60 %, 속도 1 m/s, 고도 17 m(5 mm/px, Phantom 4).
    SfM(사진에서 3D 복원)은 전방 겹침 ≥ 70–80 % 를 권장.
    """
    alt = spec.altitude_for_gsd(target_gsd_cm, "photo")
    gw, gh = spec.footprint_m(alt, "photo")
    # 사진의 짧은 변(세로)이 진행 방향이라고 가정 (드론이 세로 방향으로 진행)
    shot_interval_m = gh * (1 - front_overlap)
    line_spacing_m = gw * (1 - side_overlap)
    return FlightPlan(
        altitude_m=alt,
        gsd_photo_cm=spec.gsd_cm(alt, "photo"),
        gsd_video_cm=spec.gsd_cm(alt, "video"),
        footprint_photo_m=(gw, gh),
        line_spacing_m=line_spacing_m,
        shot_interval_m=shot_interval_m,
        shot_interval_s=shot_interval_m / speed_mps,
        speed_mps=speed_mps,
        front_overlap=front_overlap,
        side_overlap=side_overlap,
    )


def gsd_table(altitudes=(10, 15, 20, 25, 30, 40, 50, 60), spec: DroneSpec = MINI5PRO) -> list[dict]:
    rows = []
    for a in altitudes:
        gw, gh = spec.footprint_m(a, "photo")
        rows.append({
            "altitude_m": a,
            "gsd_photo_cm": round(spec.gsd_cm(a, "photo"), 3),
            "gsd_video4k_cm": round(spec.gsd_cm(a, "video"), 3),
            "footprint_photo_m": f"{gw:.1f} x {gh:.1f}",
        })
    return rows


if __name__ == "__main__":
    print(f"{MINI5PRO.name}: HFOV(photo) {MINI5PRO.hfov_deg(8192, 6144):.1f}°, VFOV {MINI5PRO.vfov_deg(8192, 6144):.1f}°")
    for r in gsd_table():
        print(r)
    fp = design_flight(0.5)
    print(fp)
