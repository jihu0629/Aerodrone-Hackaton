"""녹화 영상(.MP4) + DJI 자막(.SRT) → 기록된 비행 데이터 CSV, 프레임, COLMAP GPS 기준 파일.

  python scripts/26_srt_telemetry.py DJI_0001.MP4                      # 같은 이름의 .SRT 자동 인식
  python scripts/26_srt_telemetry.py DJI_0001.MP4 --srt other.SRT --out data/flights/f1 --interval 1.0
  python scripts/26_srt_telemetry.py DJI_0001.SRT --only-csv           # 영상 없이 SRT 만 표로

출력 <out>/
  telemetry_srt.csv   SRT 블록마다 1행 (t_start_s, timestamp, lat, lon, rel_alt_m, abs_alt_m, focal_len, iso, ...)
  frames/ + frames_srt.csv   interval 마다 프레임 + 그 시각의 SRT 값 (같은 타임라인 → 동기화 오차 1프레임 이내)
  geo_gps.txt         COLMAP model_aligner 기준: "파일명 lat lon rel_alt"  → 25_colmap_cmds.py --ref 로 넘김
DJI Fly 에서 카메라 설정 → '영상 자막' 을 켜야 .SRT 가 생긴다. 실시간 데이터가 아니라 비행 뒤 파일을 복사해 쓰는 경로다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.media_meta import find_srt, frames_with_srt, srt_health, srt_to_csv, write_colmap_ref  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="영상(.MP4) 또는 자막(.SRT) 경로")
    ap.add_argument("--srt", help="자막 경로 (기본: 영상과 같은 이름의 .SRT)")
    ap.add_argument("--out", help="출력 폴더 (기본 data/flights/<영상이름>)")
    ap.add_argument("--interval", type=float, default=1.0, help="프레임 저장 간격(초)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--only-csv", action="store_true", help="프레임 추출 없이 SRT → CSV 만")
    a = ap.parse_args()
    video = Path(a.video)
    srt = Path(a.srt) if a.srt else (video if video.suffix.lower() == ".srt" else find_srt(video))
    if srt is None or not srt.exists():
        print(f"SRT 를 찾지 못했습니다: {video.with_suffix('.SRT')}. DJI Fly 카메라 설정의 '영상 자막' 이 켜져 있었는지 확인하세요.")
        return 1
    out = Path(a.out) if a.out else Path("data/flights") / video.stem
    out.mkdir(parents=True, exist_ok=True)
    tel = srt_to_csv(srt, out / "telemetry_srt.csv")
    h = srt_health(tel)
    print(f"SRT 블록 {h['blocks']}, GPS 있음 {h['with_gps']}, rel_alt 있음 {h['with_rel_alt']}, 길이 {h['duration_s']:.1f}s")
    if h["warning"]:
        print("[경고]", h["warning"])
    if tel:
        t0 = tel[0]
        print(f"첫 블록: {t0.timestamp} lat={t0.lat} lon={t0.lon} rel_alt={t0.rel_alt_m} abs_alt={t0.abs_alt_m} focal={t0.focal_len}")
    if a.only_csv or video.suffix.lower() == ".srt":
        print(f"→ {out / 'telemetry_srt.csv'}")
        return 0
    rows = frames_with_srt(video, srt, out, a.interval, max_frames=a.max_frames)
    n = write_colmap_ref(rows, out / "geo_gps.txt")
    print(f"프레임 {len(rows)} 장 → {out / 'frames'}, GPS 기준 {n} 장 → {out / 'geo_gps.txt'}")
    print("다음: python scripts/24_select_frames.py <세션> --frames", out / "frames", "--out", out / "sfm/images")
    print("      python scripts/25_colmap_cmds.py <세션> --images", out / "sfm/images", "--work", out / "sfm/colmap", "--ref", out / "geo_gps.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
