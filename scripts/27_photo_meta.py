"""DJI 사진 폴더 → XMP/EXIF 비행 메타데이터 CSV + COLMAP GPS 기준 파일.

  python scripts/27_photo_meta.py D:\\DCIM\\100MEDIA --out data/flights/photos1

출력: photos_meta.csv (file, source, GpsLatitude, GpsLongitude, AbsoluteAltitude, RelativeAltitude, Gimbal*/Flight* Degree, DateTime)
      geo_gps.txt    (file lat lon RelativeAltitude) → 25_colmap_cmds.py --ref
사진은 영상 프레임보다 해상도·선예도가 좋고 장마다 GPS·짐벌 각이 붙어 있어 3D 복원 입력으로 더 좋다.
RelativeAltitude 는 이륙점 기준 높이이며 지면까지 거리가 아니다. Mini 5 Pro 가 어떤 필드를 쓰는지는 실제 사진 1장으로 확인한다.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dronecap.media_meta import DJI_KEYS, read_photo_meta, write_colmap_ref  # noqa: E402

COLS = ["file", "source", *DJI_KEYS, "DateTime"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", help="사진 폴더 또는 파일")
    ap.add_argument("--out", help="출력 폴더 (기본 data/flights/<폴더이름>)")
    a = ap.parse_args()
    p = Path(a.photos)
    files = sorted(f for f in (p.iterdir() if p.is_dir() else [p]) if f.suffix.lower() in (".jpg", ".jpeg", ".dng"))
    if not files:
        print("사진이 없습니다"); return 1
    out = Path(a.out) if a.out else Path("data/flights") / (p.name if p.is_dir() else p.stem)
    out.mkdir(parents=True, exist_ok=True)
    rows = [read_photo_meta(f) for f in files]
    with open(out / "photos_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    n = write_colmap_ref(rows, out / "geo_gps.txt", lat_key="GpsLatitude", lon_key="GpsLongitude", alt_key="RelativeAltitude")
    src = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print(f"사진 {len(rows)} 장, 메타 출처 {src}, GPS 기준 {n} 장 → {out}")
    first = next((r for r in rows if r["source"] != "none"), None)
    if first:
        print("예:", {k: first.get(k) for k in ("file", "GpsLatitude", "GpsLongitude", "RelativeAltitude", "AbsoluteAltitude", "GimbalPitchDegree")})
    else:
        print("[경고] XMP drone-dji / EXIF GPS 를 하나도 못 읽었습니다. 사진이 DJI 원본인지(편집·전송 중 메타 제거 여부) 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
