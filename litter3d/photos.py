"""원본 사진 묶음 → 사진 리포트 (report_photos.html). 메인 파이프라인과 함께 출력된다.

GSD 결정 순서: widths.json 의 사진 폭(m) → --photo-width → EXIF/XMP 고도(DJI RelativeAltitude, GPSAltitude) + 기체 화각 → 기본값.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .drone import MINI5PRO, DroneSpec
from .segment import MaskList, color_baseline

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_altitude_m(path: str | Path) -> float | None:
    """DJI JPEG 의 XMP RelativeAltitude(이륙점 기준 m) → 없으면 EXIF GPSAltitude → None."""
    p = Path(path)
    try:
        head = p.read_bytes()[:400_000]
        m = re.search(rb'RelativeAltitude="?\s*([+\-]?\d+(?:\.\d+)?)', head)
        if m:
            return float(m.group(1))
    except OSError:
        return None
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS
        exif = Image.open(p).getexif()
        gps = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
        if gps:
            alt = {GPSTAGS.get(k, k): v for k, v in gps.items()}.get("GPSAltitude")
            if alt is not None:
                return float(alt)
    except Exception:
        pass
    return None


def photo_gsd_m(path: str | Path, img_w: int, *, width_m: float | None = None, altitude_m: float | None = None,
                spec: DroneSpec = MINI5PRO, default_gsd_cm: float = 0.5) -> tuple[float, str]:
    """반환 (gsd_m, 근거)."""
    if width_m:
        return width_m / img_w, f"사진 폭 {width_m} m"
    alt = altitude_m if altitude_m is not None else read_altitude_m(path)
    if alt and alt > 1:
        mode = "photo" if img_w > 4500 else "video"
        gsd = spec.gsd_cm(alt, mode) / 100 * (spec.image_w if mode == "photo" else spec.video_w) / img_w
        return gsd, f"고도 {alt:.1f} m + {spec.name} 화각 ({'사진' if mode == 'photo' else '영상'} 기준)"
    return default_gsd_cm / 100, f"기본값 {default_gsd_cm} cm/px (고도·폭 정보 없음 → 확인 필요)"


def load_photos(photos_dir: str | Path, *, width_m: float | None = None, altitude_m: float | None = None,
                max_photos: int = 8, max_w: int = 2000) -> list[dict]:
    """[{name, img, gsd_m, gsd_note}] — widths.json {"파일명": 폭m} 이 있으면 우선."""
    d = Path(photos_dir)
    widths = {}
    if (d / "widths.json").exists():
        widths = json.loads((d / "widths.json").read_text(encoding="utf-8"))
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    step = max(len(paths) // max_photos, 1)
    out = []
    for p in paths[::step][:max_photos]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gsd, note = photo_gsd_m(p, img.shape[1], width_m=widths.get(p.name, width_m), altitude_m=altitude_m)
        if img.shape[1] > max_w:              # 리포트용으로 줄이고 GSD 도 같이 보정
            s = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            gsd = gsd / s
        out.append({"name": p.name, "img": img, "gsd_m": gsd, "gsd_note": note})
    return out


def detect(img: np.ndarray, gsd_m: float, segmenter=None) -> MaskList:
    if segmenter is not None:
        return segmenter.predict(img)
    return color_baseline(img, min_px=max(int((0.025 / gsd_m) ** 2), 4))


def run_photo_report(photos_dir: str | Path, out_dir: str | Path, *, weights: str | Path | None = None,
                     width_m: float | None = None, altitude_m: float | None = None, title: str = "붕붕이 사진 리포트",
                     site: str = "", report_name: str = "report_photos.html", max_photos: int = 8,
                     related: list[tuple[str, str]] | None = None) -> dict | None:
    """사진 폴더 → 검출 → 2D 추정 → report_photos.html. 첫 사진이 메인, 나머지는 갤러리."""
    from .pipeline import run_image_only
    photos = load_photos(photos_dir, width_m=width_m, altitude_m=altitude_m, max_photos=max_photos)
    if not photos:
        return None
    seg = None
    if weights:
        from .segment import YoloSegmenter
        seg = YoloSegmenter(weights)
    dets = [(ph, detect(ph["img"], ph["gsd_m"], seg)) for ph in photos]
    src = "YOLO-seg" if seg else "색 기반 베이스라인(오검출 포함)"
    main, main_masks = dets[0]
    frames = [(f"{ph['name']} · GSD {ph['gsd_m'] * 100:.2f} cm/px ({ph['gsd_note']}) · {src}", ph["img"], m) for ph, m in dets[1:]]
    r = run_image_only(main["img"], main_masks, main["gsd_m"], out_dir, frame_detections=frames, title=title,
                       site=f"{site + ' · ' if site else ''}{main['name']} ({main['gsd_note']}, {src})",
                       report_name=report_name, write_side_files=False, related=related)
    r["photos"] = [{"name": ph["name"], "n": len(m), "gsd_cm": round(ph["gsd_m"] * 100, 3), "gsd_note": ph["gsd_note"]} for ph, m in dets]
    return r
