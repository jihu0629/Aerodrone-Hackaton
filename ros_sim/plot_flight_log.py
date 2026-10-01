"""
flight_log.csv(PX4 SITL 실비행 기록)를 계획 경로와 비교하는 그래프로
그린다: 위에서 본 궤적(계획 vs 실제), 시간에 따른 고도, 시간에 따른
추적 오차(목표와의 거리).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 지원 폰트
plt.rcParams['axes.unicode_minus'] = False

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('out/flight_log.csv')
rows = list(csv.DictReader(open(log_path)))

t = [float(r['t']) for r in rows]
tx, ty, tz = [float(r['target_x']) for r in rows], [float(r['target_y']) for r in rows], [float(r['target_z']) for r in rows]
ax, ay, az = [float(r['actual_x']) for r in rows], [float(r['actual_y']) for r in rows], [float(r['actual_z']) for r in rows]
dist = [float(r['dist_to_target']) for r in rows]
phase = [r['phase'] for r in rows]

PHASE_COLOR = {'mapping': '#4aa8ff', 'orbit': '#ffb648', 'transit': '#8a94a8'}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('PX4 SITL 실비행 검증 — 계획 경로 vs 실제 비행', fontsize=14, fontweight='bold')

# 1) 위에서 본 궤적
ax1 = axes[0, 0]
ax1.plot(tx, ty, color='#cccccc', linewidth=1.5, linestyle='--', label='계획 경로', zorder=1)
ax1.scatter(ax, ay, c=[PHASE_COLOR.get(p, '#333') for p in phase], s=3, label='실제 비행', zorder=2)
ax1.set_xlabel('x (m, 동쪽)')
ax1.set_ylabel('y (m, 북쪽)')
ax1.set_title('위에서 본 궤적 (회색 점선=계획, 색점=실제)')
ax1.set_aspect('equal', adjustable='box')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(alpha=0.3)

# 2) 고도 프로파일
ax2 = axes[0, 1]
ax2.plot(t, tz, color='#cccccc', linewidth=1.5, linestyle='--', label='목표 고도')
ax2.plot(t, az, color='#1657D0', linewidth=1, label='실제 고도')
ax2.set_xlabel('시간 (s)')
ax2.set_ylabel('고도 (m)')
ax2.set_title('고도: 목표 vs 실제')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 3) 추적 오차(목표까지 거리)
ax3 = axes[1, 0]
ax3.plot(t, dist, color='#d94f4f', linewidth=0.8)
ax3.axhline(1.5, color='#3ddb84', linestyle=':', linewidth=1, label='허용오차 1.5m')
ax3.set_xlabel('시간 (s)')
ax3.set_ylabel('목표까지 거리 (m)')
ax3.set_title('추적 오차 (웨이포인트 전환마다 순간적으로 커졌다 수렴)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 4) 단계별 시간 분포
ax4 = axes[1, 1]
from collections import Counter
counts = Counter(phase)
labels = list(counts.keys())
ax4.pie([counts[l] for l in labels], labels=labels,
        colors=[PHASE_COLOR.get(l, '#999') for l in labels],
        autopct='%1.0f%%', textprops={'fontsize': 9})
ax4.set_title('비행 시간 중 단계별 비중')

plt.tight_layout()
out_path = log_path.parent / 'flight_log_plot.png'
plt.savefig(out_path, dpi=140)
print(f'저장됨: {out_path}')
print(f'총 기록 {len(rows)}틱, 총 비행시간 {t[-1]:.1f}s, 평균 오차 {sum(dist)/len(dist):.2f}m, 최대 오차 {max(dist):.1f}m')
