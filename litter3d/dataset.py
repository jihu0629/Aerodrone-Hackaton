"""라벨 데이터 점검·변환·분할 (기업 데이터 형식이 확정되기 전이라 여러 형식을 자동 판별한다).

지원 형식
  yolo            images/*.jpg + labels/*.txt (class x1 y1 x2 y2 ... 정규화 폴리곤 또는 박스)
  coco            하나의 *.json 에 images / annotations / categories
  labelme         이미지마다 *.json, "shapes":[{"label","points"}]
  per_image_json  이미지마다 *.json (AI Hub 류). "annotations"/"objects"/"shapes" 아래에 폴리곤·박스·클래스가 있음.
                  키 이름이 제각각이라 휴리스틱으로 찾고, 안 되면 KeyHints 로 지정한다.
  voc             이미지마다 *.xml (박스만)
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .classes import class_names, map_class

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------- 점검
def list_images(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix in IMG_EXT)


def detect_format(root: str | Path) -> str:
    root = Path(root)
    jsons = list(root.rglob("*.json"))
    txts = [p for p in root.rglob("*.txt") if p.name not in {"classes.txt", "geo.txt", "README.txt"}]
    xmls = list(root.rglob("*.xml"))
    imgs = list_images(root)
    for j in jsons[:20]:
        try:
            d = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            return "coco"
        if isinstance(d, dict) and "shapes" in d and "imagePath" in d:
            return "labelme"
    if jsons and imgs and len(jsons) >= 0.5 * len(imgs):
        return "per_image_json"
    if txts and imgs and len(txts) >= 0.5 * len(imgs):
        return "yolo"
    if xmls and imgs:
        return "voc"
    if jsons:
        return "per_image_json"
    return "unknown"


def inspect_dataset(root: str | Path, n_sample: int = 2) -> dict:
    """폴더 내용 요약 + 라벨 파일 앞부분 출력용 정보."""
    root = Path(root)
    ext = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    fmt = detect_format(root)
    imgs = list_images(root)
    samples = []
    for pat in ("*.json", "*.txt", "*.xml"):
        for p in list(root.rglob(pat))[:n_sample]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            samples.append({"file": str(p.relative_to(root)), "head": txt[:800]})
    info = {"root": str(root), "format": fmt, "n_images": len(imgs), "extensions": dict(ext.most_common(15)),
            "samples": samples}
    if imgs:
        try:
            import cv2
            im = cv2.imread(str(imgs[0]))
            info["image_shape"] = None if im is None else im.shape
        except Exception:
            pass
    return info


# ---------------------------------------------------------------- 변환
@dataclass
class KeyHints:
    """per_image_json 에서 키 이름을 직접 지정할 때 사용. None 이면 휴리스틱."""
    objects: str | None = None      # 객체 목록 키 (예: "annotations", "objects", "shapes")
    label: str | None = None        # 클래스 키 (예: "class", "label", "category", "name")
    polygon: str | None = None      # 폴리곤 키 (예: "polygon", "segmentation", "points")
    bbox: str | None = None         # 박스 키 (예: "bbox", "box")
    image_file: str | None = None   # 이미지 파일명 키 (예: "image_name", "filename")
    width: str | None = None
    height: str | None = None
    class_map: dict[str, str] = field(default_factory=dict)   # 원본 클래스명 → 우리 클래스


_OBJ_KEYS = ("annotations", "objects", "shapes", "labels", "regions", "instances", "annotation")
_LABEL_KEYS = ("class", "label", "category", "category_name", "name", "class_name", "cls", "type", "kind")
_POLY_KEYS = ("polygon", "segmentation", "points", "seg", "poly", "coordinates")
_BBOX_KEYS = ("bbox", "box", "rect", "bounding_box")
_FILE_KEYS = ("image_name", "filename", "file_name", "imagePath", "image", "image_id", "name")


def _first(d: dict, keys, hint=None):
    if hint and hint in d:
        return d[hint]
    for k in keys:
        if k in d:
            return d[k]
    return None


def _flatten_points(pts) -> list[float] | None:
    """[[x,y],[x,y]] / [x,y,x,y] / [{"x":..,"y":..}] → [x,y,x,y,...]"""
    if pts is None:
        return None
    if isinstance(pts, dict):
        if {"x", "y"} <= set(pts) and isinstance(pts["x"], list):
            return [v for xy in zip(pts["x"], pts["y"]) for v in xy]
        return None
    if not isinstance(pts, list) or not pts:
        return None
    if isinstance(pts[0], (int, float)):
        return [float(v) for v in pts]
    if isinstance(pts[0], dict):
        return [float(v) for p in pts for v in (p.get("x"), p.get("y"))]
    if isinstance(pts[0], list):
        if pts[0] and isinstance(pts[0][0], (int, float)):
            return [float(v) for p in pts for v in p[:2]]
        return _flatten_points(pts[0])   # COCO 식 [[...]]
    return None


def _bbox_to_poly(b) -> list[float] | None:
    b = _flatten_points(b)
    if not b or len(b) < 4:
        return None
    x, y, w, h = b[:4]
    if w < x or h < y:            # x1,y1,x2,y2 형식으로 추정 (x2>x1 인데 "w"<x 이면)
        x1, y1, x2, y2 = x, y, w, h
    else:
        x1, y1, x2, y2 = x, y, x + w, y + h
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def per_image_json_to_coco(root: str | Path, hints: KeyHints | None = None) -> dict:
    """이미지마다 JSON 이 있는 데이터셋 → COCO dict. 이미지 파일은 같은 이름(확장자만 다름) 또는 JSON 안의 파일명 키로 찾는다."""
    hints = hints or KeyHints()
    root = Path(root)
    imgs = {p.stem: p for p in list_images(root)}
    by_name = {p.name: p for p in imgs.values()}
    names = class_names()
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    skipped = Counter()
    aid = 0
    for j in sorted(root.rglob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            skipped["bad_json"] += 1
            continue
        if isinstance(d, list):
            d = {"annotations": d}
        img_path = imgs.get(j.stem)
        if img_path is None:
            fn = _first(d, _FILE_KEYS, hints.image_file)
            if isinstance(fn, dict):
                fn = _first(fn, _FILE_KEYS)
            if fn is not None and Path(str(fn)).name in by_name:
                img_path = by_name[Path(str(fn)).name]
        if img_path is None:
            skipped["no_image"] += 1
            continue
        w = _first(d, ("width", "imageWidth", "image_width"), hints.width)
        h = _first(d, ("height", "imageHeight", "image_height"), hints.height)
        if isinstance(d.get("images"), dict):
            w = w or d["images"].get("width"); h = h or d["images"].get("height")
        if w is None or h is None:
            import cv2
            im = cv2.imread(str(img_path))
            if im is None:
                skipped["unreadable_image"] += 1
                continue
            h, w = im.shape[:2]
        iid = len(coco["images"]) + 1
        coco["images"].append({"id": iid, "file_name": str(img_path.relative_to(root)), "width": int(w), "height": int(h)})
        objs = _first(d, _OBJ_KEYS, hints.objects)
        if objs is None:
            skipped["no_objects_key"] += 1
            continue
        if isinstance(objs, dict):
            objs = list(objs.values())
        for o in objs:
            if not isinstance(o, dict):
                continue
            lab = _first(o, _LABEL_KEYS, hints.label)
            if isinstance(lab, dict):
                lab = _first(lab, _LABEL_KEYS)
            lab = str(lab) if lab is not None else "unknown"
            cls = hints.class_map.get(lab) or map_class(lab)
            poly = _flatten_points(_first(o, _POLY_KEYS, hints.polygon))
            if poly is None or len(poly) < 6:
                poly = _bbox_to_poly(_first(o, _BBOX_KEYS, hints.bbox))
                if poly is None:
                    skipped["no_geometry"] += 1
                    continue
                skipped["bbox_only"] += 1
            xs, ys = poly[0::2], poly[1::2]
            # 정규화(0~1) 좌표면 픽셀로
            if max(xs + ys) <= 1.0:
                xs = [x * w for x in xs]; ys = [y * h for y in ys]
                poly = [v for xy in zip(xs, ys) for v in xy]
            aid += 1
            coco["annotations"].append({"id": aid, "image_id": iid, "category_id": names.index(cls) + 1,
                                        "segmentation": [poly], "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                                        "area": (max(xs) - min(xs)) * (max(ys) - min(ys)), "iscrowd": 0,
                                        "orig_label": lab})
    coco["_skipped"] = dict(skipped)
    coco["_label_counts"] = dict(Counter(a["orig_label"] for a in coco["annotations"]))
    return coco


def labelme_to_coco(root: str | Path, class_map: dict[str, str] | None = None) -> dict:
    return per_image_json_to_coco(root, KeyHints(objects="shapes", label="label", polygon="points",
                                                 image_file="imagePath", width="imageWidth", height="imageHeight",
                                                 class_map=class_map or {}))


def voc_to_coco(root: str | Path, class_map: dict[str, str] | None = None) -> dict:
    root = Path(root)
    imgs = {p.stem: p for p in list_images(root)}
    names = class_names()
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    aid = 0
    for x in sorted(root.rglob("*.xml")):
        img = imgs.get(x.stem)
        if img is None:
            continue
        t = ET.parse(x).getroot()
        w = int(t.findtext("size/width")); h = int(t.findtext("size/height"))
        iid = len(coco["images"]) + 1
        coco["images"].append({"id": iid, "file_name": str(img.relative_to(root)), "width": w, "height": h})
        for o in t.findall("object"):
            lab = o.findtext("name") or "unknown"
            cls = (class_map or {}).get(lab) or map_class(lab)
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            aid += 1
            coco["annotations"].append({"id": aid, "image_id": iid, "category_id": names.index(cls) + 1,
                                        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                        "orig_label": lab})
    coco["_skipped"] = {"bbox_only": aid}
    coco["_label_counts"] = dict(Counter(a["orig_label"] for a in coco["annotations"]))
    return coco


def coco_to_yolo_labels(coco: dict, out_labels: str | Path) -> dict[str, int]:
    """COCO dict(카테고리가 우리 13종 순서) → YOLO-seg 텍스트. 파일명 stem 기준."""
    out = Path(out_labels); out.mkdir(parents=True, exist_ok=True)
    names = class_names()
    cat2idx = {}
    for c in coco["categories"]:
        cls = c["name"] if c["name"] in names else map_class(c["name"])
        cat2idx[c["id"]] = names.index(cls)
    imgs = {im["id"]: im for im in coco["images"]}
    lines: dict[str, list[str]] = {}
    per_class: Counter = Counter()
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        idx = cat2idx[a["category_id"]]
        seg = a.get("segmentation")
        polys = [s for s in seg if len(s) >= 6] if isinstance(seg, list) else []
        if not polys and "bbox" in a:
            x, y, bw, bh = a["bbox"]
            polys = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
        for p in polys:
            norm = " ".join(f"{min(max(p[i] / (w if i % 2 == 0 else h), 0), 1):.6f}" for i in range(len(p)))
            lines.setdefault(Path(im["file_name"]).stem, []).append(f"{idx} {norm}")
            per_class[names[idx]] += 1
    for stem, ls in lines.items():
        (out / f"{stem}.txt").write_text("\n".join(ls) + "\n", encoding="utf-8")
    return dict(per_class)


def yolo_class_histogram(labels_dir: str | Path) -> dict[str, int]:
    names = class_names()
    c: Counter = Counter()
    for p in Path(labels_dir).rglob("*.txt"):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts:
                try:
                    c[names[int(parts[0])] if int(parts[0]) < len(names) else f"id{parts[0]}"] += 1
                except ValueError:
                    pass
    return dict(c)


# ---------------------------------------------------------------- 분할
def split_yolo_dataset(images: list[Path], labels_dir: str | Path, out_dir: str | Path, val_ratio: float = 0.2,
                       seed: int = 0, copy: bool = True, include_unlabeled: bool = False) -> dict:
    """images + labels/*.txt → out_dir/images/{train,val}, labels/{train,val}, data.yaml"""
    labels_dir, out = Path(labels_dir), Path(out_dir)
    rng = random.Random(seed)
    pairs = []
    for im in images:
        lab = labels_dir / f"{im.stem}.txt"
        if lab.exists() or include_unlabeled:
            pairs.append((im, lab if lab.exists() else None))
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    for s, items in splits.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for im, lab in items:
            dst = out / "images" / s / im.name
            if copy:
                shutil.copy2(im, dst)
            else:
                if not dst.exists():
                    dst.symlink_to(im.resolve())
            if lab is not None:
                shutil.copy2(lab, out / "labels" / s / lab.name)
            else:
                (out / "labels" / s / f"{im.stem}.txt").write_text("", encoding="utf-8")
    names = class_names()
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return {"train": len(splits["train"]), "val": len(splits["val"]), "data_yaml": str(yaml),
            "train_hist": yolo_class_histogram(out / "labels" / "train"),
            "val_hist": yolo_class_histogram(out / "labels" / "val")}


def prepare_dataset(root: str | Path, out_dir: str | Path, fmt: str | None = None, hints: KeyHints | None = None,
                    val_ratio: float = 0.2, copy: bool = True, coco_json: str | Path | None = None) -> dict:
    """형식 자동 판별 → COCO → YOLO-seg → train/val 분할. 반환에 data_yaml 경로와 클래스 히스토그램."""
    root, out = Path(root), Path(out_dir)
    fmt = fmt or detect_format(root)
    tmp_labels = out / "_labels_all"
    report: dict = {"format": fmt}
    if fmt == "yolo":
        txts = [p for p in root.rglob("*.txt") if p.name not in {"classes.txt", "geo.txt"}]
        tmp_labels.mkdir(parents=True, exist_ok=True)
        for t in txts:
            shutil.copy2(t, tmp_labels / t.name)
        report["note"] = "YOLO 라벨을 그대로 사용. 클래스 인덱스가 우리 13종 순서와 같은지 확인 필요 (classes.txt)."
    else:
        if fmt == "coco":
            jpath = Path(coco_json) if coco_json else next(p for p in root.rglob("*.json")
                                                          if {"images", "annotations"} <= set(json.loads(p.read_text(encoding="utf-8", errors="ignore")) or {}))
            coco = json.loads(jpath.read_text(encoding="utf-8"))
            coco["images"] = [dict(im, file_name=Path(im["file_name"]).name) for im in coco["images"]]
        elif fmt == "labelme":
            coco = labelme_to_coco(root, (hints or KeyHints()).class_map)
        elif fmt == "voc":
            coco = voc_to_coco(root, (hints or KeyHints()).class_map)
        else:
            coco = per_image_json_to_coco(root, hints)
        report["skipped"] = coco.get("_skipped", {})
        report["orig_label_counts"] = coco.get("_label_counts", {})
        report["per_class"] = coco_to_yolo_labels(coco, tmp_labels)
        (out / "converted_coco.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "converted_coco.json").write_text(json.dumps({k: v for k, v in coco.items() if not k.startswith("_")},
                                                           ensure_ascii=False), encoding="utf-8")
    report.update(split_yolo_dataset(list_images(root), tmp_labels, out, val_ratio=val_ratio, copy=copy))
    return report
