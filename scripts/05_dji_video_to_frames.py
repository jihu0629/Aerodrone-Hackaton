"""
5단계: DJI 드론 영상(.MP4 + .SRT) -> 3D 재구성용 프레임 JPEG (GPS EXIF 포함).

DJI Fly/Pilot 앱에서 카메라 설정 > '영상 자막(Video Caption/Subtitles)' 을 켜고 촬영하면
영상 옆에 같은 이름의 .SRT 파일이 생깁니다. 이 파일의 위도·경도·고도를 각 프레임의 EXIF 에 넣습니다.
SRT 가 없으면 프레임만 뽑습니다 (3D 는 되지만 축척·위치가 임의라 부피 계산이 틀어집니다).

사용
----
    python scripts/05_dji_video_to_frames.py --video D:\\DJI\\DJI_0012.MP4 --out C:\\hackathon\\frames\\beach_a
    python scripts/05_dji_video_to_frames.py --video DJI_0012.MP4 DJI_0013.MP4 --every-m 6 --out frames/beach_a
    python scripts/05_dji_video_to_frames.py --video DJI_0012.MP4 --auto-spacing --overlap 0.75

    --every-sec 1.0   1초마다 한 장 (기본). 속도 5 m/s 고도 50 m 면 겹침 약 85%.
    --every-m 6       기체가 6 m 움직일 때마다 한 장 (SRT GPS 필요). 정지 구간 중복 제거에 좋음.
    --auto-spacing    SRT 의 평균 고도에서 --overlap 이 되는 간격을 계산해 --every-m 으로 사용.
    --min-blur 30     흐린 프레임 제거 기준 (라플라시안 분산). 낮추면 더 많이 남김.

출력 (--out/)
    <영상이름>_00000.jpg ...   GPS EXIF 포함 프레임
    frames.csv                 프레임별 시각·위경도·고도·선명도
    flight_summary.json        비행 요약 (고도, 거리, 시간, 중심 좌표)

촬영 권장 (부피 측정 목적)
    고도 30~60 m, 속도 3~5 m/s, 김벌 -70~-90도(수직에 가깝게), 격자(그리드) 비행 + 대상 주변 한 바퀴 궤도 비행.
    같은 곳을 두 방향(가로/세로) 으로 지나면 DSM 품질이 크게 좋아집니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR  # noqa: E402
from coastcd.dji import extract_frames, find_srt_for_video, flight_summary, parse_srt, spacing_for_overlap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, help="DJI MP4 (여러 개 가능). 같은 이름의 SRT 를 자동으로 찾음")
    ap.add_argument("--srt", nargs="*", default=None, help="SRT 를 직접 지정 (영상 순서와 같게)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR / "frames"))
    ap.add_argument("--every-sec", type=float, default=1.0)
    ap.add_argument("--every-m", type=float, default=None)
    ap.add_argument("--auto-spacing", action="store_true")
    ap.add_argument("--overlap", type=float, default=0.75, help="--auto-spacing 목표 앞뒤 겹침 (0~1)")
    ap.add_argument("--min-blur", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None, help="영상당 최대 프레임 수 (빠른 시험용)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_frames = []
    summaries = {}
    for i, v in enumerate(args.video):
        video = Path(v)
        if not video.exists():
            sys.exit(f"영상 없음: {video}")
        srt = Path(args.srt[i]) if args.srt and i < len(args.srt) else find_srt_for_video(video)
        recs = parse_srt(srt) if srt else []
        summ = flight_summary(recs)
        summaries[video.name] = summ
        every_m = args.every_m
        if args.auto_spacing:
            alt = summ.get("rel_alt_max_m")
            if alt:
                every_m = spacing_for_overlap(alt, args.overlap)
                print(f"[05] {video.name}: 최대 고도 {alt:.1f} m -> 겹침 {args.overlap:.0%} 간격 {every_m:.1f} m")
            else:
                print(f"[05] {video.name}: SRT 고도 없음, --every-sec {args.every_sec} 사용")
        frames = extract_frames(video, out, srt=srt, every_sec=args.every_sec, every_m=every_m,
                                min_blur=args.min_blur, max_frames=args.max_frames, jpeg_quality=args.quality)
        all_frames += frames

    n_gps = sum(1 for f in all_frames if f.lat is not None)
    print(f"[05] 총 {len(all_frames)} 장, GPS 있는 프레임 {n_gps} 장")
    if all_frames and n_gps == 0:
        print("[05] 경고: GPS 가 하나도 없습니다. ODM 은 돌아가지만 결과 축척·위치가 임의입니다.")
    (out / "flight_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[05] 다음 단계: python scripts/06_build_3d_map.py --frames {out} --project <이름>")


if __name__ == "__main__":
    main()
