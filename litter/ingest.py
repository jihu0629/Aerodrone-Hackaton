"""
업체 라벨 데이터 → 내부 표준 형식.

업체 포맷을 아직 모르니 흔한 3가지를 자동 판별한다:
  - COCO JSON  (images / annotations / categories)
  - YOLO txt   (폴더, bbox 또는 seg 폴리곤, 정규화 좌표)
  - 평탄 CSV   (image, class, bbox 또는 polygon, weight ...)
처음 보는 포맷이면 `python -m litter inspect <경로>`로 구조를 찍어
보고 어댑터를 하나 추가하면 된다.

무게·수거 위치 같은 정답은 라벨 안에 있어도 되고(annotation의 추가
필드), 별도 CSV(--weights)여도 된다. 별도 CSV는 id 열로 붙인다.

표준 형식 (annotation 하나 = dict):
  ann_id, image, width, height, cls_raw, cls, polygon[[u,v],...] (픽셀),
  from_bbox(bool), item_id, weight_kg, gt_lat, gt_lon, gt_x, gt_y
"""
import csv
import json
from pathlib import Path

from .config import map_class

_WEIGHT_KEYS = {"weight_kg": 1.0, "weight": 1.0, "kg": 1.0, "무게": 1.0, "무게_kg": 1.0,
                "weight_g": 0.001, "무게_g": 0.001}
_ID_KEYS = ["ann_id", "annotation_id", "id", "object_id", "item_id", "객체id"]
_ITEM_KEYS = ["item_id", "object_id", "track_id", "instance_id", "객체id"]
_GT_KEYS = {"gt_lat": ["gt_lat", "collect_lat", "수거_위도"], "gt_lon": ["gt_lon", "collect_lon", "수거_경도"],
            "gt_x": ["gt_x"], "gt_y": ["gt_y"]}


def _pick_extra(rec):
    """annotation/CSV 행에서 무게·item_id·수거위치 정답을 뽑는다."""
    low = {str(k).lower().strip(): v for k, v in rec.items()}
    out = {}
    for k, scale in _WEIGHT_KEYS.items():
        if low.get(k) not in (None, ""):
            try:
                out["weight_kg"] = float(low[k]) * scale
                break
            except ValueError:
                pass
    for k in _ITEM_KEYS:
        if low.get(k) not in (None, ""):
            out["item_id"] = str(low[k])
            break
    for dst, names in _GT_KEYS.items():
        for n in names:
            if low.get(n) not in (None, ""):
                out[dst] = float(low[n])
                break
    return out


def _bbox_poly(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def load_coco(path, cfg, min_score=0.0):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    out = []
    for a in d["annotations"]:
        if a.get("score", 1.0) < min_score:  # 모델 예측 결과일 때만 score가 있다
            continue
        im = imgs[a["image_id"]]
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg and len(seg[0]) >= 6:
            s = max(seg, key=len)  # 여러 조각이면 가장 큰 것
            poly, from_bbox = [[s[i], s[i + 1]] for i in range(0, len(s), 2)], False
        else:  # RLE이거나 bbox만 — 마스크는 나중에 SAM으로 다듬을 수 있다
            poly, from_bbox = _bbox_poly(*a["bbox"]), True
        name = cats.get(a["category_id"], str(a["category_id"]))
        rec = {"ann_id": str(a["id"]), "image": Path(im["file_name"]).name,
               "width": im.get("width"), "height": im.get("height"),
               "cls_raw": name, "cls": map_class(name, cfg), "polygon": poly, "from_bbox": from_bbox}
        if im.get("geo_transform"):  # 정사영상 칩 (litter.ortho) — 픽셀→지도 변환을 칩마다 가짐
            rec["geo_transform"], rec["crs"] = im["geo_transform"], im.get("crs")
        if "score" in a:
            rec["score"] = a["score"]
        rec.update(_pick_extra({**a, **a.get("attributes", {})}))
        out.append(rec)
    return out


def load_yolo(label_dir, images_dir, class_names, cfg):
    from PIL import Image

    out = []
    for txt in sorted(Path(label_dir).glob("*.txt")):
        img = next((p for p in Path(images_dir).glob(txt.stem + ".*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")), None)
        if img is None:
            continue
        W, H = Image.open(img).size
        for k, line in enumerate(txt.read_text().split("\n")):
            v = line.split()
            if len(v) < 5:
                continue
            c, nums = int(v[0]), [float(t) for t in v[1:]]
            if len(nums) == 4:
                cx, cy, w, h = nums
                poly, fb = _bbox_poly((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H), True
            else:
                poly, fb = [[nums[i] * W, nums[i + 1] * H] for i in range(0, len(nums) - 1, 2)], False
            name = class_names[c] if c < len(class_names) else str(c)
            out.append({"ann_id": f"{txt.stem}_{k}", "image": img.name, "width": W, "height": H,
                        "cls_raw": name, "cls": map_class(name, cfg), "polygon": poly, "from_bbox": fb})
    return out


def load_csv(path, cfg):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for k, r in enumerate(csv.DictReader(f)):
            low = {s.lower(): v for s, v in r.items()}
            if low.get("polygon"):
                pts = json.loads(low["polygon"])
                poly, fb = [list(p) for p in pts], False
            else:
                x1, y1, x2, y2 = (float(low[s]) for s in ("x1", "y1", "x2", "y2"))
                poly, fb = _bbox_poly(x1, y1, x2 - x1, y2 - y1), True
            name = low.get("class") or low.get("category") or low.get("label") or "other"
            rec = {"ann_id": low.get("ann_id") or low.get("id") or str(k), "image": Path(low["image"]).name,
                   "width": None, "height": None, "cls_raw": name, "cls": map_class(name, cfg),
                   "polygon": poly, "from_bbox": fb}
            rec.update(_pick_extra(r))
            out.append(rec)
    return out


def merge_weights(anns, path):
    """별도 무게 CSV를 ann_id(또는 item_id)로 붙인다. 붙은 개수를 반환."""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    key = next((k for k in rows[0] if k.lower() in _ID_KEYS), None)
    if key is None:
        raise ValueError(f"무게 CSV에 id 열이 없음 (후보: {_ID_KEYS}), 실제 열: {list(rows[0])}")
    table = {str(r[key]): _pick_extra(r) for r in rows}
    n = 0
    for a in anns:
        extra = table.get(a["ann_id"]) or table.get(a.get("item_id", "\0"))
        if extra:
            a.update(extra)
            n += 1
    return n


def load(labels, cfg, images_dir=None, class_names=None, weights=None, min_score=0.0):
    p = Path(labels)
    if p.is_dir():
        names = class_names or []
        cn = p / "classes.txt"
        if not names and cn.exists():
            names = cn.read_text(encoding="utf-8").split()
        anns = load_yolo(p, images_dir or p, names, cfg)
    elif p.suffix.lower() == ".json":
        anns = load_coco(p, cfg, min_score)
    elif p.suffix.lower() == ".csv":
        anns = load_csv(p, cfg)
    else:
        raise ValueError(f"모르는 라벨 포맷: {p}")
    if weights:
        n = merge_weights(anns, weights)
        print(f"  무게 CSV 매칭: {n}/{len(anns)}")
    return anns


def inspect(path, max_items=3):
    """처음 보는 데이터 구조 요약 — 내일 데이터 받자마자 이것부터 돌린다."""
    p = Path(path)
    if p.is_dir():
        exts = {}
        for f in p.rglob("*"):
            if f.is_file():
                exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        print(f"[폴더] {p}  파일 확장자별 개수: {dict(sorted(exts.items(), key=lambda t: -t[1]))}")
        for ext in (".json", ".csv", ".srt", ".txt"):
            for f in list(p.rglob("*" + ext))[:2]:
                print(f"\n--- 예시 {f.relative_to(p)}")
                inspect(f, max_items)
        imgs = [f for f in p.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg")][:2]
        for f in imgs:
            from .telemetry import read_jpg
            print(f"\n--- EXIF/XMP {f.name}: {read_jpg(f)}")
        vids = [f for f in p.rglob("*") if f.suffix.lower() in (".mp4", ".mov")]
        if vids:
            print(f"\n동영상 {len(vids)}개 (예: {vids[0].name}) — 같은 이름 .SRT가 있으면 텔레메트리 사용 가능")
        return
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))

        def show(o, depth=0, key="root"):
            pad = "  " * depth
            if isinstance(o, dict):
                print(f"{pad}{key}: dict({len(o)}) keys={list(o)[:15]}")
                for k in list(o)[:15]:
                    if isinstance(o[k], (dict, list)) and depth < 3:
                        show(o[k], depth + 1, k)
            elif isinstance(o, list):
                print(f"{pad}{key}: list({len(o)})")
                if o and depth < 3:
                    show(o[0], depth + 1, key + "[0]")
                    if not isinstance(o[0], (dict, list)):
                        print(f"{pad}  예: {o[:6]}")
        show(d)
        if isinstance(d, dict) and "annotations" in d:
            from collections import Counter
            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            cnt = Counter(cats.get(a["category_id"], a["category_id"]) for a in d["annotations"])
            seg = sum(1 for a in d["annotations"] if isinstance(a.get("segmentation"), list) and a["segmentation"])
            keys = set().union(*(a.keys() for a in d["annotations"]))
            print(f"\nCOCO: 이미지 {len(d['images'])} · 라벨 {len(d['annotations'])} · 폴리곤 {seg}")
            print(f"annotation 필드: {sorted(keys)}")
            print(f"클래스 분포: {dict(cnt.most_common())}")
    elif p.suffix.lower() in (".csv", ".txt", ".srt"):
        lines = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        print(f"[{p.name}] {len(lines)}줄")
        for ln in lines[: max_items + 8]:
            print("  " + ln[:200])
