"""OCR 원문 → 숫자·단위·상태.

상태(status)
  ok             숫자를 읽었고 기대 단위와 맞음 (값이 0 이어도 ok)
  empty          OCR 이 글자를 하나도 못 읽음
  no_number      글자는 있는데 숫자 형태가 없음
  ambiguous      한 영역에서 숫자가 둘 이상 읽힘 (ROI 가 넓거나 라벨 숫자가 섞임). 어느 것이 맞는지 알 수 없어 value 는 비움
  unit_mismatch  숫자는 있으나 단위가 기대와 다름 (값은 보존, 변환은 sync 단계에서 명시적으로)
  out_of_range   숫자는 있으나 설정한 범위 밖 (오인식 의심. 값은 보존)
'이전 값 재사용' 은 여기서 절대 하지 않는다. 실패는 실패로 남긴다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# 음수 부호로 자주 나오는 문자들: hyphen, minus sign, en dash, em dash
_MINUS = "-−–—"
_NUM_RE = re.compile(rf"([{_MINUS}])?\s*(\d+(?:[.,]\d+)?|[.,]\d+)")
# 단위 사전: 표기 변형 → 정규화 단위
_UNITS = [
    (re.compile(r"km\s*/\s*h|kph|km/h", re.I), "km/h"),
    (re.compile(r"m\s*/\s*s|mps|m/s", re.I), "m/s"),
    (re.compile(r"mph", re.I), "mph"),
    (re.compile(r"ft\s*/\s*s", re.I), "ft/s"),
    (re.compile(r"(?<![a-z])ft(?![a-z])", re.I), "ft"),
    (re.compile(r"(?<![a-z])m(?![a-z/])", re.I), "m"),
]
# 숫자 안에서 흔한 오인식 (문맥상 숫자 토큰일 때만 바꾼다)
_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})


@dataclass
class Parsed:
    raw: str
    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = "empty"
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def detect_unit(text: str) -> Optional[str]:
    for rx, u in _UNITS:
        if rx.search(text):
            return u
    return None


def _strip_unit(text: str, unit: Optional[str]) -> str:
    if unit is None:
        return text
    for rx, u in _UNITS:
        if u == unit:
            return rx.sub(" ", text)
    return text


def parse_number(raw: str, expect_unit: Optional[str] = None, value_range: Optional[tuple[float, float]] = None,
                 fix_confusions: bool = True, label: Optional[str] = None) -> Parsed:
    """label: 화면에서 숫자 앞에 붙는 라벨(예: 'H', 'D'). 맨 앞의 그 글자 하나만 떼어 낸다 (오인식 문자 'o' 등은 보호)."""
    p = Parsed(raw=raw or "")
    text = (raw or "").strip()
    if not text:
        return p
    unit = detect_unit(text)
    p.unit = unit
    body = _strip_unit(text, unit).strip()
    if label:
        stripped = re.sub(rf"^{re.escape(label)}\s*", "", body, count=1)
        if stripped != body:
            p.notes.append("label_stripped")
        body = stripped
    cand = body
    if fix_confusions:
        # 숫자·점·부호 사이에 끼어 있는 글자만 치환 (예: "1O.5" → "10.5"). 전체가 글자인 경우는 그대로 둔다
        def _fix(m: re.Match) -> str:
            return m.group(0).translate(_CONFUSIONS)
        cand = re.sub(rf"(?<=[\d.,{_MINUS}])[OolI|SB](?=[\d.,])|(?<=[\d.,])[OolI|SB]\b|\b[OolI|SB](?=[\d.,])", _fix, cand)
        if cand != body:
            p.notes.append("confusion_fixed")
    m = _NUM_RE.search(cand)
    if not m:
        p.status = "no_number"
        return p
    sign, num = m.group(1), m.group(2)
    num = num.replace(",", ".")
    if num.startswith("."):
        num = "0" + num
        p.notes.append("leading_dot")
    try:
        val = float(num)
    except ValueError:
        p.status = "no_number"
        return p
    if sign:
        val = -val
    p.value = val
    # 같은 영역에 숫자가 둘 이상 보이면 어느 것이 맞는지 알 수 없다 → 값을 비우고 ambiguous
    allnums = _NUM_RE.findall(cand)
    if len(allnums) > 1:
        p.notes.append("candidates=" + "/".join((sg or "") + n for sg, n in allnums))
        p.value = None
        p.status = "ambiguous"
        return p
    if expect_unit and unit and unit != expect_unit:
        p.status = "unit_mismatch"
        return p
    if expect_unit and unit is None:
        p.notes.append("unit_not_seen")   # 단위가 안 보여도 값은 ok 로 두되 기록
    if value_range is not None and not (value_range[0] <= val <= value_range[1]):
        p.status = "out_of_range"
        return p
    p.status = "ok"
    return p


def convert_to(value: float, unit: Optional[str], target: str) -> tuple[Optional[float], str]:
    """단위 변환. 변환 근거를 두 번째 값으로 돌려준다. 단위를 모르면 (None, 'unknown_unit')."""
    if unit is None:
        return None, "unknown_unit"
    if unit == target:
        return value, "same_unit"
    table = {
        ("km/h", "m/s"): 1000 / 3600, ("mph", "m/s"): 0.44704, ("ft/s", "m/s"): 0.3048,
        ("ft", "m"): 0.3048,
    }
    k = table.get((unit, target))
    if k is None:
        return None, f"no_conversion_{unit}_to_{target}"
    return value * k, f"converted_{unit}_to_{target}"
