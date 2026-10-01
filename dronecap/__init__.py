"""dronecap — DJI 드론 RTSP 영상 수집 · 화면 OCR 텔레메트리 · 오프라인 3D 복원 준비.

단계별 모듈
  1단계  stream / recorder / frames / capture_app   : RTSP 수신·표시·저장·프레임 추출
  2단계  ocr/*                                      : 미러링 화면·녹화·이미지에서 숫자 OCR
  2.5    sync                                       : 프레임 ↔ OCR 값 시간 매칭 (허용 오차·오프셋)
  3단계  sfm                                        : 프레임 선별, COLMAP 명령 생성

litter3d 패키지(해안쓰레기 무게 추정)와는 독립적으로 동작한다.
"""
__version__ = "0.1.0"
