# DJI Mini 5 Pro 스펙과 촬영 설계

## 확인된 스펙 (2026-09-30)
| 항목 | 값 | 출처 |
|---|---|---|
| 무게 | 약 250 g (249.9 g) | cined, dpreview |
| 센서 | 1인치형 CMOS 50 MP (쿼드 베이어, 12 MP 비닝 가능) | cined, dpreview, pix-pro |
| 사진 최대 | 8192 × 6144 px (4:3) | drdrone.ca, techpoint |
| 렌즈 | 24 mm 환산, 대각 화각 84°, f/1.8 고정 | cined, diyphotography |
| 영상 | 4K 최대 120 fps (4K/60 HDR) | cined |
| 짐벌 | 회전 짐벌 (세로 촬영) | cined |
| 호버링 정확도 | 비전 수직 ±0.1 m / 수평 ±0.3 m, GNSS ±0.5 m | B&H 스펙표 |
| GNSS | GPS + BeiDou + Galileo, L1/L5 이중대역, **RTK 없음** | drdrone.ca |
| 장애물 감지 | 전방 LiDAR + 어안 비전 + 하방 3D 적외선 | dronexl, heliguy |
| 웨이포인트 비행 | DJI Fly 지원 (WaypointMap·Pixpro Waypoints 로 격자 미션 가능) | dronexl, waypointmap |
| 라이브 스트리밍 | DJI RC 2 + DJI Fly V1.15.8 이상에서 RTMP 지원 | DJI 지원 문서 |
| .SRT 자막 | 프레임마다 위도·경도·rel_alt·abs_alt·focal_len·iso·shutter 기록 | dji-drone-metadata-embedder, skystamp issue #5 |
| **MSDK V5** | **Mini 5 Pro 미지원** (DJI: 지원 계획 없음) | GitHub dji-sdk issue #810 |

**확인 필요**: 실제 초점거리(mm)·센서 물리 크기(mm), 4K 영상 크롭 배율. GSD 계산은 화각 84° 만 사용하므로 이 값 없이도 동작.

## SRT 예시 (Mini 5 Pro)
```
1
00:00:00,000 --> 00:00:00,033
FrameCnt: 1, DiffTime: 33ms
2026-09-25 16:23:55.467
[iso: 200] [shutter: 1/2500.0] [fnum: 1.8] [ev: 0] [color_md: default] [focal_len: 24.00] [latitude: 30.142288] [longitude: -95.768454] [rel_alt: 0.000 abs_alt: 65.972] [ct: 4711]
```
DJI Fly 에서 **영상 자막(Video Caption)** 을 켜야 생성된다. `litter3d/srt.py` 가 파싱한다.

## 고도별 GSD (화각 84°, 짐벌 −90° 기준 계산값)
가로 화각(사진) 71.5°, 세로 56.8°

| 고도 (m) | 사진 GSD (cm/px) | 4K 영상 GSD (cm/px)* | 사진 촬영폭 (m) |
|---|---|---|---|
| 10 | 0.18 | 0.41 | 14.4 x 10.8 |
| 15 | 0.26 | 0.61 | 21.6 x 16.2 |
| 20 | 0.35 | 0.82 | 28.8 x 21.6 |
| 25 | 0.44 | 1.02 | 36.0 x 27.0 |
| 30 | 0.53 | 1.23 | 43.2 x 32.4 |
| 40 | 0.70 | 1.64 | 57.6 x 43.2 |
| 50 | 0.88 | 2.04 | 72.0 x 54.0 |
| 60 | 1.05 | 2.45 | 86.4 x 64.8 |

\* 4K 영상이 센서 가로폭 전체를 쓴다고 가정. 실제 크롭 확인 필요.

## 권장 촬영 계획 (목표 GSD 0.5 cm/px, Andriolo 2023 의 0.5–1.25 범위)
- 고도 **28 m**, 촬영폭 41 × 31 m
- 전방 겹침 80 %, 측면 겹침 70 % → 비행선 간격 12.3 m, 촬영 간격 6.1 m (2 m/s 에서 3.1 s)
- 50 MP 사진으로 3D 복원·부피, 4K 영상은 실시간 검출·개수용
- 기준물(크기를 아는 판, 예: 1 m 자·A1 보드)을 반드시 함께 촬영 → 스케일 검증 (길이 10 % 오차 = 부피 33 %)
- Kako 2020: GCP 없이 수평 30 cm·높이 90 cm 오차 → 기준점 권장
- 명령: `python scripts/10_flight_design.py --gsd 0.5`

## "실시간" 설계 (제안, 미확정)
1. 실시간: DJI Fly RTMP → 노트북 로컬 RTMP 서버 → 프레임마다 YOLO-seg → 개수·종류, 무게는 종류별 평균무게로 대략
2. 준실시간: 겹치는 프레임이 쌓이면 VGGT 등으로 빠른 3D → 부피 → 겉보기 밀도로 갱신
3. 비행 후: 4K·50 MP 원본으로 ODM/COLMAP 정밀 복원 → 최종 무게·수거 계획
확인 필요: Mini 5 Pro RTMP 화질·지연, 로컬 서버 수신 방법.
