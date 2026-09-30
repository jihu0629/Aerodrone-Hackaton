"""실제 해변 사진(Andriolo et al. 2024 그림 크롭, CC BY)으로 시각 리포트 만들기 — DSM 없이 2D 추정.

  - 그림 3c 왼쪽(2 m 폭)·오른쪽(1 m 폭): 논문 저자가 그린 빨간 윤곽 = 실제 라벨 → 그대로 마스크로 사용
  - 그림 1d 두 장(5 m 폭): 라벨이 없어 색 기반 베이스라인(Kako 2020 방식)으로 검출 → 오검출 포함
    python scripts/16_photo_demo.py --out outputs/photo_demo
학습된 YOLO-seg 가 있으면 --weights best.pt 로 그림 1d 검출을 모델로 바꾼다.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from litter3d.pipeline import run_image_only
from litter3d.segment import color_baseline
from litter3d.baselines import w2_count

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="data/andriolo2024")
ap.add_argument("--out", default="outputs/photo_demo")
ap.add_argument("--weights", default=None)
a = ap.parse_args()
P = Path(a.photos)
WIDTH_M = {"crop_fig3c_left": 2.0, "crop_fig3c_right": 1.0, "crop_fig1d_left": 5.0, "crop_fig1d_right": 5.0}


def paper_label_masks(img):
    """빨간 윤곽(논문의 객체 분할) → 채운 인스턴스 마스크. 윤곽은 이미지에서 지운다(inpaint)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = []
    for c in cnts:
        if cv2.contourArea(c) < 8:
            continue
        m = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(m, [c], 1)
        masks.append(("unknown", m.astype(bool), 1.0))
    clean = cv2.inpaint(img, cv2.dilate(red, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    return clean, masks


def model_or_color(img, gsd_cm):
    if a.weights:
        from litter3d.segment import YoloSegmenter
        return YoloSegmenter(a.weights).predict(img)
    return color_baseline(img, min_px=max(int((2.5 / gsd_cm) ** 2), 4))


sets = {}
for name, wm in WIDTH_M.items():
    img = cv2.imread(str(P / f"{name}.png"))
    gsd_cm = wm * 100 / img.shape[1]
    if "fig3c" in name:
        img, masks = paper_label_masks(img); src = "논문 라벨(빨간 윤곽)"
    else:
        masks = model_or_color(img, gsd_cm); src = "YOLO-seg" if a.weights else "색 기반 베이스라인(오검출 포함)"
    sets[name] = (img, masks, gsd_cm, wm, src)
    print(f"{name}: {wm} m 폭, GSD {gsd_cm:.2f} cm/px, {len(masks)}개 ({src}), W2 = {w2_count(len(masks), 13.4):.0f} g")

# 메인: 그림 3c 왼쪽 (논문 라벨). 나머지 3장은 갤러리
main = "crop_fig3c_left"
img, masks, gsd_cm, wm, src = sets[main]
frames = [(f"{n} · {v[3]} m 폭 · {v[4]}", v[0], v[1]) for n, v in sets.items() if n != main]
r = run_image_only(img, masks, gsd_cm / 100, a.out, frame_detections=frames, report_name="report_photos.html",
                   title="붕붕이 사진 리포트", site=f"Leirosa 해변 (Andriolo et al. 2024 그림 3c, {wm} m 폭, {src})")
lo, ty, hi = r["total_kg"]
print(f"메인 사진: {r['n_objects']}개, {ty * 1000:.0f} g (범위 {lo * 1000:.0f}–{hi * 1000:.0f}) → {r['report_html']}")
