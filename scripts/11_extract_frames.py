"""드론 영상 → SfM 용 프레임 + geo.txt
    python scripts/11_extract_frames.py DJI_0001.MP4 --out outputs/frames --overlap 0.8
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litter3d.frames import extract_frames
from litter3d.srt import parse_srt, summarize

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--srt", default=None, help="기본: 영상과 같은 이름의 .SRT")
ap.add_argument("--out", default="outputs/frames")
ap.add_argument("--overlap", type=float, default=0.8)
ap.add_argument("--every", type=float, default=1.0, help="SRT 없을 때 추출 간격(초)")
ap.add_argument("--alt", type=float, default=None, help="비행 고도 m (SRT 없을 때)")
ap.add_argument("--max", type=int, default=None)
a = ap.parse_args()

srt = a.srt
if srt is None:
    for ext in (".SRT", ".srt"):
        p = Path(a.video).with_suffix(ext)
        if p.exists():
            srt = str(p)
if srt:
    print("SRT 요약:", summarize(parse_srt(srt)))
else:
    print("SRT 없음 → 고정 시간 간격 추출. DJI Fly 에서 '영상 자막' 을 켜세요.")
rows = extract_frames(a.video, a.out, srt=srt, front_overlap=a.overlap, every_s=a.every,
                      altitude_m=a.alt, max_frames=a.max)
print(f"프레임 {len(rows)}장 → {a.out} (frames.csv, geo.txt)")
