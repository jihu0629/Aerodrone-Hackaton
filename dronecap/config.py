"""YAML 설정 로드.

설정 파일(config/dronecap.yml) 의 모든 키는 DEFAULTS 에 기본값이 있다.
파일에 없는 키는 기본값으로 채우고, 모르는 키가 있으면 오타 가능성이 있으니 경고를 출력한다.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "stream": {
        # MediaMTX 가 같은 PC 에서 돌 때 읽는 주소. DJI Fly 가 보내는 RTMP 주소와 다르다.
        "url": "rtsp://127.0.0.1:8554/live/drone",
        # tcp: Wi-Fi/루프백에서 패킷 손실·순서 뒤바뀜이 없어 깨진 프레임이 줄어든다. udp 는 지연은 조금 낮지만 깨짐이 생길 수 있다.
        "rtsp_transport": "tcp",
        # 미리보기 지연을 줄이기 위한 FFmpeg 옵션(OpenCV 가 FFmpeg 백엔드로 열 때 적용). 기록 품질에는 영향 없음.
        "low_latency": True,
        # 연결 끊김 뒤 재접속 대기. 처음 reconnect_min_s 에서 시작해 2배씩 늘려 reconnect_max_s 까지.
        "reconnect_min_s": 1.0,
        "reconnect_max_s": 10.0,
        # 이 시간 동안 프레임이 하나도 안 오면 '끊김'으로 보고 다시 연결한다.
        "stall_timeout_s": 5.0,
        # 열기(핸드셰이크) 제한 시간.
        "open_timeout_s": 10.0,
        # 입력이 로컬 파일일 때 원본 fps 속도로 읽어 실제 스트림처럼 시험한다. 네트워크 입력에는 영향 없음.
        "file_realtime": True,
    },
    "session": {
        "root": "data/sessions",   # 세션 폴더가 생기는 위치 (git 제외)
        "id_prefix": "drone",
    },
    "record": {
        "enabled_on_start": False,  # True 면 첫 연결 성공 시 자동으로 녹화 시작
        # ffmpeg: 재인코딩 없이(-c copy) 원본 H.264 + AAC 오디오를 그대로 저장 → 화질 손실 0, CPU 거의 안 씀. 권장.
        # opencv: ffmpeg.exe 가 없을 때의 대안. 디코드된 프레임을 다시 인코딩(화질 손실, CPU 사용), 오디오 없음.
        "backend": "ffmpeg",
        "ffmpeg_path": "ffmpeg",      # PATH 에 없으면 전체 경로 지정 (예: C:\\ffmpeg\\bin\\ffmpeg.exe)
        "container": "mp4",           # mp4(조각 mp4: 중간에 끊겨도 재생 가능) | mkv | ts
        "opencv_fourcc": "mp4v",
        "opencv_fps_fallback": 30.0,  # 스트림이 fps 를 안 알려줄 때
    },
    "frames": {
        "enabled_on_start": True,
        "interval_s": 1.0,            # 프레임 저장 간격 (monotonic 기준)
        "format": "jpg",              # jpg | png (png 는 무손실이지만 크고 느림)
        "jpeg_quality": 95,
        "queue_size": 30,             # 디스크 쓰기가 밀리면 여기까지 대기, 넘치면 버리고 기록
    },
    "display": {
        "enabled": True,
        "window_name": "dronecap",
        "max_width": 1280,            # 미리보기 창 가로 최대 (원본은 그대로 저장)
        "hud": True,
    },
    "ocr": {
        "engine": "rapidocr",         # rapidocr(CPU, pip 만으로 설치) | tesseract(별도 tesseract.exe 필요)
        "tesseract_cmd": "tesseract",
        "rapidocr_detect": False,     # False: ROI 를 한 줄로 보고 인식만(좁은 숫자 ROI 에 정확). True: 글자 검출 후 인식(넓은 ROI)
        "roi_file": "config/ocr_roi.json",
        "interval_s": 0.5,
        "upscale": 3.0,               # 작은 글자를 키워서 OCR (2~4 권장)
        "invert": False,              # 흰 글자/어두운 배경이면 True 로 바꿔 시험
        "threshold": False,           # 이진화. 배경이 복잡하면 켜서 비교
        # 필드 정의. 화면을 실제로 보고 확인한 뒤 수정할 것. 여기 적힌 이름이 CSV field 열이 된다.
        # unit: 기대 단위. OCR 결과에 다른 단위가 보이면 unit_mismatch 로 기록.
        # range: 그럴듯한 값 범위. 밖이면 out_of_range (값은 보존하되 status 로 구분).
        # label: 숫자 앞에 붙어 함께 잘릴 수 있는 라벨 글자. 파서가 맨 앞의 이 글자만 떼어 낸다.
        "fields": {
            "H": {"desc": "이륙 지점 기준 상대 고도. 바로 아래 지면까지 거리가 아님", "label": "H", "unit": "m", "range": [-200, 1000]},
            "D": {"desc": "홈포인트까지 수평 거리. 전방 물체까지 거리가 아님", "label": "D", "unit": "m", "range": [0, 30000]},
            "HS": {"desc": "수평 속도 (아이콘·단위 확인 필요)", "label": None, "unit": "m/s", "range": [0, 40]},
            "VS": {"desc": "수직 속도 (부호·아이콘 확인 필요)", "label": None, "unit": "m/s", "range": [-20, 20]},
        },
        "screen": {"monitor": 1, "region": None},   # region: [left, top, width, height] 화면 좌표. None 이면 모니터 전체
        "debug_every": 20,            # N 회마다 잘라낸 영역 + 인식 결과 이미지를 저장 (0 이면 끔)
    },
    "sync": {
        # 텔레메트리(OCR) 시각 + offset_s = 영상 시각 으로 간주.  양수: OCR 쪽이 영상보다 먼저 도착함.
        # 미러링과 RTMP 의 전송 지연 차이를 보정하는 값. 실측(예: 화면과 영상에 같이 보이는 사건) 으로 정해야 한다.
        "offset_s": 0.0,
        "tolerance_ms": 500,          # 이보다 먼 OCR 값은 프레임에 연결하지 않음
        # OCR field 이름 → 매칭 CSV 열 이름. 화면에 없는 항목은 지우거나 null.
        "field_map": {
            "relative_altitude_m": "H",
            "home_distance_m": "D",
            "horizontal_speed_mps": "HS",
            "vertical_speed_mps": "VS",
        },
    },
    "sfm": {
        "blur_min": 60.0,             # Laplacian 분산 기준. 이보다 흐리면 제외 (영상·해상도별로 조정)
        "min_change": 0.03,           # 이전 선택 프레임과의 차이(0~1). 이보다 작으면 중복으로 제외
        "max_frames": 400,
        "use_gpu": False,             # GPU·VRAM 확인 전까지 CPU 가정
    },
}


def _merge(base: dict, over: dict, path: str, warnings: list[str]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            warnings.append(f"알 수 없는 설정 키: {path}{k} (오타인지 확인)")
            out[k] = v
            continue
        if isinstance(base[k], dict) and isinstance(v, dict) and k not in ("fields", "field_map"):
            out[k] = _merge(base[k], v, f"{path}{k}.", warnings)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict:
    """설정 파일을 읽어 기본값과 합친다. path 가 None 이거나 없으면 기본값만 사용(경고 출력)."""
    warnings: list[str] = []
    data: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            warnings.append(f"설정 파일이 없어 기본값을 사용합니다: {p}")
    cfg = _merge(DEFAULTS, data, "", warnings)
    cfg["_warnings"] = warnings
    cfg["_path"] = str(path) if path else None
    return cfg
