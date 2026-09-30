"""쓰레기 검출·분할.

1) YOLO-seg (ultralytics): 기업 라벨 데이터로 학습 → 추론 → 클래스별 인스턴스 마스크
2) COCO(폴리곤) → YOLO-seg 라벨 변환
3) 색 기반 베이스라인 (Kako 2020 방식: HSV 로 모래/쓰레기 구분) — 모델이 없을 때 데모·비교용

ultralytics 가 설치돼 있지 않으면 1) 은 ImportError 를 낸다. `pip install ultralytics`.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .classes import class_names, map_class

MaskList = list[tuple[str, np.ndarray, float]]   # (class_name, bool mask, confidence)


# ---------------- COCO → YOLO-seg ----------------
def coco_to_yolo_seg(coco_json: str | Path, images_dir: str | Path, out_dir: str | Path,
                     class_map: dict[str, str] | None = None) -> dict:
    """COCO 인스턴스 분할 JSON → YOLO-seg 텍스트 라벨. 클래스는 map_class() 로 우리 13종에 맞춘다.
    반환: {"names": [...], "n_images": .., "per_class": {...}}"""
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    names = class_names()
    idx = {n: i for i, n in enumerate(names)}
    cats = {c["id"]: (class_map or {}).get(c["name"], map_class(c["name"])) for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    out_dir = Path(out_dir); (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    per_img: dict[int, list[str]] = {}
    per_class: dict[str, int] = {}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        cls = cats[a["category_id"]]
        seg = a.get("segmentation")
        polys: list[list[float]] = []
        if isinstance(seg, list) and seg:
            polys = [s for s in seg if len(s) >= 6]
        elif "bbox" in a:      # 폴리곤이 없으면 박스를 사각 폴리곤으로 (SAM 으로 정밀화 필요)
            x, y, bw, bh = a["bbox"]
            polys = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
        for p in polys:
            norm = " ".join(f"{p[i] / (w if i % 2 == 0 else h):.6f}" for i in range(len(p)))
            per_img.setdefault(a["image_id"], []).append(f"{idx[cls]} {norm}")
            per_class[cls] = per_class.get(cls, 0) + 1
    for iid, lines in per_img.items():
        stem = Path(imgs[iid]["file_name"]).stem
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml = out_dir / "data.yaml"
    yaml.write_text("path: " + str(out_dir.resolve()) + "\ntrain: images/train\nval: images/val\n"
                    + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return {"names": names, "n_images": len(per_img), "per_class": per_class, "data_yaml": str(yaml)}


# ---------------- YOLO-seg ----------------
def train_yolo_seg(data_yaml: str | Path, model: str = "yolo11n-seg.pt", epochs: int = 50, imgsz: int = 1024,
                   project: str = "runs/litter3d", name: str = "seg", **kw):
    from ultralytics import YOLO
    m = YOLO(model)
    return m.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=project, name=name, **kw)


class YoloSegmenter:
    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 1024):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def predict(self, image: np.ndarray) -> MaskList:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out: MaskList = []
        if res.masks is None:
            return out
        h, w = image.shape[:2]
        for m, c, p in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            out.append((map_class(str(self.names[int(c)])), mm, float(p)))
        return out

    def predict_tiled(self, image: np.ndarray, tile: int = 2048, overlap: int = 256) -> MaskList:
        """큰 정사영상을 타일로 나눠 추론하고 마스크를 원본 좌표로 합친다."""
        h, w = image.shape[:2]
        out: MaskList = []
        step = tile - overlap
        for y in range(0, max(h - overlap, 1), step):
            for x in range(0, max(w - overlap, 1), step):
                sub = image[y:y + tile, x:x + tile]
                for cls, m, conf in self.predict(sub):
                    full = np.zeros((h, w), bool)
                    full[y:y + m.shape[0], x:x + m.shape[1]] = m
                    out.append((cls, full, conf))
        return merge_duplicates(out)


def merge_duplicates(masks: MaskList, iou_thr: float = 0.5) -> MaskList:
    """타일 겹침 부분에서 중복 검출된 마스크를 합친다 (같은 클래스, IoU ≥ thr)."""
    kept: MaskList = []
    for cls, m, conf in sorted(masks, key=lambda t: -t[2]):
        dup = False
        for i, (kc, km, kconf) in enumerate(kept):
            if kc != cls:
                continue
            inter = np.logical_and(m, km).sum()
            if inter == 0:
                continue
            union = np.logical_or(m, km).sum()
            if inter / union >= iou_thr:
                kept[i] = (kc, np.logical_or(m, km), max(conf, kconf))
                dup = True
                break
        if not dup:
            kept.append((cls, m, conf))
    return kept


# ---------------- 색 기반 베이스라인 (Kako 2020 방식 단순화) ----------------
def color_baseline(image_bgr: np.ndarray, sand_hsv_lo=(5, 0, 80), sand_hsv_hi=(35, 90, 255),
                   min_px: int = 20, open_px: int = 3) -> MaskList:
    """모래색(HSV 범위) 이 아닌 픽셀을 쓰레기 후보로 본다. 클래스는 전부 'unknown'.
    Kako et al. (2020) 의 색 기반 3층 신경망과 같은 한계(그림자·반사·자연물 오검출)를 공유한다.
    YOLO-seg 결과와 비교하는 기준선으로만 쓴다."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sand = cv2.inRange(hsv, np.array(sand_hsv_lo), np.array(sand_hsv_hi))
    cand = (sand == 0).astype(np.uint8)
    k = np.ones((open_px, open_px), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    out: MaskList = []
    for i in range(1, n):
        m = lab == i
        if m.sum() >= min_px:
            out.append(("unknown", m, 0.5))
    return out


def masks_to_png(masks: MaskList, shape: tuple[int, int], path: str | Path) -> None:
    """클래스 인덱스 PNG (0 = 배경) 로 저장."""
    names = class_names()
    lab = np.zeros(shape, np.uint8)
    for cls, m, _ in masks:
        lab[m] = names.index(cls) + 1
    cv2.imwrite(str(path), lab)


def masks_from_png(path: str | Path) -> MaskList:
    from .volume import split_instances
    names = class_names()
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    out: MaskList = []
    for i, n in enumerate(names):
        cm = lab == i + 1
        for inst in split_instances(cm):
            out.append((n, inst, 1.0))
    return out
