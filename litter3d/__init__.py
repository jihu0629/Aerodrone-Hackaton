"""litter3d — 드론 영상 → 쓰레기 분할 → 3D 부피 → 겉보기 밀도로 무게 → 격자 지도 → 수거 계획.

모듈 구성
  drone       DJI Mini 5 Pro 스펙, GSD(지상표본거리)·촬영 설계 계산
  srt         DJI 영상 자막(.SRT) 텔레메트리 파서 (위도·경도·고도)
  frames      영상 → 겹침 기준 프레임 추출 + ODM 용 geo.txt
  classes     쓰레기 클래스, 겉보기 밀도표, 개당 평균무게표 (출처·가정값 표기)
  segment     YOLO-seg 학습/추론 래퍼, COCO→YOLO 변환, 색 기반 베이스라인(Kako 2020 방식)
  reconstruct ODM/COLMAP 실행 래퍼, DSM·정사영상 읽기
  volume      DSM 차분 부피 (바닥 = 마스크 바깥 링 평면 보간)
  mass        부피 × 겉보기 밀도 → 무게 (최소/대표/최대)
  gridmap     10 m 격자 kg 히트맵
  plan        수거 계획 (인력·마대·차량·경로)
  pipeline    전체 실행
"""
__version__ = "0.1.0"
