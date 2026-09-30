"""DJI Mini 5 Pro 고도별 GSD 표와 목표 GSD 촬영 계획 출력.
    python scripts/10_flight_design.py --gsd 0.5 --overlap 0.8 --side 0.7 --speed 2
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d.drone import MINI5PRO, design_flight, gsd_table

ap = argparse.ArgumentParser()
ap.add_argument("--gsd", type=float, default=0.5, help="목표 GSD cm/px (Andriolo 2023: 0.5–1.25)")
ap.add_argument("--overlap", type=float, default=0.8)
ap.add_argument("--side", type=float, default=0.7)
ap.add_argument("--speed", type=float, default=2.0, help="m/s")
a = ap.parse_args()

print(f"{MINI5PRO.name} | 사진 {MINI5PRO.image_w}x{MINI5PRO.image_h}, 대각 화각 {MINI5PRO.fov_diag_deg}°")
print(f"가로 화각(사진) {MINI5PRO.hfov_deg(MINI5PRO.image_w, MINI5PRO.image_h):.1f}°, "
      f"세로 {MINI5PRO.vfov_deg(MINI5PRO.image_w, MINI5PRO.image_h):.1f}°")
print()
print(f"{'고도(m)':>7} {'사진 GSD(cm)':>12} {'4K영상 GSD(cm)':>14} {'사진 촬영폭(m)':>16}")
for r in gsd_table():
    print(f"{r['altitude_m']:>7} {r['gsd_photo_cm']:>12.2f} {r['gsd_video4k_cm']:>14.2f} {r['footprint_photo_m']:>16}")
fp = design_flight(a.gsd, a.overlap, a.side, a.speed)
print()
print(f"목표 GSD {a.gsd} cm/px → 고도 {fp.altitude_m:.1f} m, 촬영폭 {fp.footprint_photo_m[0]:.1f} x {fp.footprint_photo_m[1]:.1f} m")
print(f"  비행선 간격 {fp.line_spacing_m:.1f} m (측면 겹침 {a.side:.0%}), 촬영 간격 {fp.shot_interval_m:.1f} m = {fp.shot_interval_s:.1f} s @ {a.speed} m/s")
print(f"  같은 고도 4K 영상 GSD {fp.gsd_video_cm:.2f} cm/px")
print("  ※ 4K 영상 화각은 사진과 같다고 가정. 실제 크롭은 확인 필요.")
