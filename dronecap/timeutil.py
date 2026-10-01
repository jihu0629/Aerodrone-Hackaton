"""시각 기록 규칙.

두 종류의 시각을 항상 같이 기록한다.
  * utc      : time.time() 기반 벽시계. 사람이 읽고, 다른 장치(아이폰 등)와 맞출 때 쓴다.
               PC 시계가 NTP 로 보정되면 뒤로 튈 수도 있어 '간격' 계산엔 쓰지 않는다.
  * monotonic: time.monotonic() 기반. 프로그램 안에서 간격·지연 측정 전용. 다른 PC 와 비교 불가.

주의: 여기서 기록하는 시각은 모두 **PC 가 데이터를 받은 시각**이다.
드론이 실제로 촬영한 시각, 센서가 측정한 시각이 아니다 (RTMP 인코딩·전송 지연이 더해져 있다).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime | None = None) -> str:
    """ISO 8601, 밀리초, 'Z' 접미. 예: 2026-10-01T02:03:04.567Z"""
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_utc_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def monotonic() -> float:
    return time.monotonic()


def session_stamp(dt: datetime | None = None) -> str:
    """폴더 이름용 로컬 시각. 예: 20261001_110102"""
    dt = (dt or utc_now()).astimezone()
    return dt.strftime("%Y%m%d_%H%M%S")


class Clock:
    """utc 와 monotonic 을 한 번에 읽어 쌍으로 돌려준다 (두 값이 서로 다른 순간을 가리키지 않게)."""

    def now(self) -> tuple[str, float]:
        m = time.monotonic()
        return utc_iso(), m
