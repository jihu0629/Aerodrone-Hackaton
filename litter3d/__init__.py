"""litter3d (ShoreSweep Planner) — 기업 라벨(지도 위 쓰레기 + 추정 무게) + 정사영상 → 작업자용 수거 계획.

모듈 구성
  classes         쓰레기 클래스, 겉보기 밀도표, 개당 평균무게표 (출처·가정값 표기)
  plan            마대·톤백·트럭 적재량, NIOSH 23 kg
  terrain         정사영상 색 → 물·숲·맨땅 격자 → 최단경로 (걷기 환산 거리, 보트 모드)
  collect         무게 추정 → 구역 묶기 → 순회 최적화 → 운반 방식 → 일차 분할
  collect_report  인터랙티브 HTML(브라우저 안에서 재계산) + 인쇄용 PNG 지도
"""
__version__ = "0.2.0"
