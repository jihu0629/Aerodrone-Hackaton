"""
쓰레기 분할 베이스라인 — YOLO-seg + 타일링.

왜 타일링인가: 드론 원본은 4000×3000 이상인데 쓰레기는 수십 픽셀이다.
통째로 1024로 줄이면 PET병이 2–3 픽셀이 돼서 사라진다. 그래서 원본을
겹치는 타일(기본 1024, 겹침 20%)로 잘라 학습하고, 추론도 타일로 한 뒤
원본 좌표로 합친다 (SAHI와 같은 방식).

  python -m litter.seg prepare --coco labels.json --images DIR --out data/seg_xxx
  python -m litter.seg train   --data data/seg_xxx/data.yaml --model yolo11s-seg.pt --epochs 80
  python -m litter.seg predict --weights best.pt --images DIR [--coco labels.json] --out preds.json
  python -m litter.seg eval    --gt labels.json --pred preds.json

predict 출력은 COCO 형식이라 그대로 `python -m litter run --labels preds.json`에 넣는다.
eval은 원본 이미지 단위로 잰다 (타일 단위 mAP는 겹침 때문에 부풀려짐):
  - 정밀도 / 재현율 / F1 (박스 IoU 0.5), AP50
  - 클래스별 재현율 — 놓친 쓰레기는 무게 과소추정으로 직결 (Winans 2023: 재현율 40%)
  - **면적 비율** (예측 마스크 면적 ÷ 정답 면적) — 면적이 무게 추정의 입력이라 중요
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box


def _imread(p):
    """한글 경로에서도 읽히게 (Windows cv2.imread는 비ASCII 경로에서 None)."""
    try:
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _imwrite(p, img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(str(p))


# ---------------------------------------------------------------- 공통
def _tiles(W, H, tile, overlap):
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y, min(tile, W), min(tile, H)) for y in ys for x in xs]


def _poly_of(a):
    seg = a.get("segmentation")
    if isinstance(seg, list) and seg and len(seg[0]) >= 6:
        s = max(seg, key=len)
        return Polygon(np.asarray(s, float).reshape(-1, 2)).buffer(0)
    x, y, w, h = a["bbox"]
    return box(x, y, x + w, y + h)


def _load_coco(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    return d, cats, by_img


def _find_image(images_dir, file_name):
    p = Path(images_dir) / file_name
    if p.exists():
        return p
    q = next(Path(images_dir).rglob(Path(file_name).name), None)
    return q


# ---------------------------------------------------------------- prepare
def prepare(coco, images_dir, out, tile=1024, overlap=0.2, split=(0.7, 0.15, 0.15), seed=0,
            min_visible=0.4, neg_keep=0.15, single_class=False, split_file=None):
    """COCO → 타일 YOLO-seg 데이터셋. 분할은 '이미지' 단위 (같은 사진의 타일이
    train/val에 섞이면 점수가 부풀려진다). test 이미지 목록은 split.json에 저장."""
    d, cats, by_img = _load_coco(coco)
    out = Path(out)
    names = ["litter"] if single_class else [cats[k] for k in sorted(cats)]
    cid = {k: (0 if single_class else i) for i, k in enumerate(sorted(cats))}
    imgs = list(d["images"])
    if split_file:  # 데이터셋 공식 분할 ({"train": [파일명...], "val": [...], "test": [...]})
        sf = json.loads(Path(split_file).read_text(encoding="utf-8"))
        by_name = {Path(im["file_name"]).name: im for im in imgs}
        parts = {k: [by_name[n] for n in sf.get(k, []) if n in by_name] for k in ("train", "val", "test")}
    else:
        random.Random(seed).shuffle(imgs)
        n = len(imgs)
        n_tr, n_va = int(n * split[0]), int(n * split[1])
        parts = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}
    rng = random.Random(seed)
    stats = {}
    for part, ims in parts.items():
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        n_tiles = n_obj = 0
        for im in ims:
            p = _find_image(images_dir, im["file_name"])
            img = _imread(p) if p else None
            if img is None:
                print(f"  ⚠️ 이미지 없음: {im['file_name']}")
                continue
            H, W = img.shape[:2]
            polys = [(cid[a["category_id"]], _poly_of(a)) for a in by_img.get(im["id"], [])]
            for (x, y, w, h) in _tiles(W, H, tile, overlap):
                tb = box(x, y, x + w, y + h)
                lines = []
                for c, pg in polys:
                    if pg.is_empty or not pg.intersects(tb):
                        continue
                    inter = pg.intersection(tb)
                    if inter.area < min_visible * pg.area:
                        continue
                    for g in getattr(inter, "geoms", [inter]):
                        if g.geom_type != "Polygon" or g.area < 4:
                            continue
                        xy = np.asarray(g.exterior.coords)[:-1] - [x, y]
                        xy /= [w, h]
                        lines.append(f"{c} " + " ".join(f"{v:.5f}" for v in xy.clip(0, 1).ravel()))
                if not lines and rng.random() > neg_keep:
                    continue
                stem = f"{Path(im['file_name']).stem}_{x}_{y}"
                _imwrite(out / "images" / part / f"{stem}.jpg", img[y:y + h, x:x + w])
                (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines))
                n_tiles += 1
                n_obj += len(lines)
        stats[part] = {"images": len(ims), "tiles": n_tiles, "objects": n_obj}
    yaml = (f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {json.dumps(nm, ensure_ascii=False)}\n" for i, nm in enumerate(names)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    (out / "split.json").write_text(json.dumps({k: [i["file_name"] for i in v] for k, v in parts.items()},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"coco": str(coco), "tile": tile, "overlap": overlap, "single_class": single_class,
            "names": names, "stats": stats}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"데이터셋 → {out}  {stats}")
    return out


# ---------------------------------------------------------------- train
def train(data, model="yolo11s-seg.pt", epochs=80, imgsz=1024, batch=8, project="runs/seg", name=None,
          workers=3, **kw):
    """model이 -seg면 분할, 아니면(yolo11s.pt) 검출 — 같은 데이터셋(폴리곤은 박스로 변환됨)."""
    from ultralytics import YOLO

    m = YOLO(model)
    r = m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, project=str(Path(project).resolve()),
                name=name or Path(data).parent.name, workers=workers, patience=20, close_mosaic=10,
                # 드론 시점: 위아래 뒤집기·90° 회전이 자연스러움, 색은 약하게
                flipud=0.5, fliplr=0.5, degrees=90, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, **kw)
    best = Path(r.save_dir) / "weights" / "best.pt"
    print(f"best → {best}")
    return best


# ---------------------------------------------------------------- predict
def _ios(a, b):
    """두 박스의 교집합 ÷ 작은 박스 넓이 — 타일 경계에서 잘린 조각 중복 제거용."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(min(sa, sb), 1e-6)


def predict(weights, images_dir, out, coco=None, files=None, tile=1024, overlap=0.2, conf=0.25,
            imgsz=None, merge_ios=0.6, batch=8, scale=1.0):
    """원본 이미지마다 타일 추론 → 원본 좌표로 합치기 → COCO JSON.
    coco를 주면 그 images 목록·id·카테고리 이름을 그대로 쓴다 (평가용)."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    names = m.names
    if coco:
        d = json.loads(Path(coco).read_text(encoding="utf-8"))
        images = d["images"]
        if files:
            keep = set(files)
            images = [im for im in images if im["file_name"] in keep]
        name_to_cat = {c["name"]: c["id"] for c in d.get("categories", [])}
    else:
        paths = sorted(p for p in Path(images_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        images = [{"id": i, "file_name": p.relative_to(images_dir).as_posix()} for i, p in enumerate(paths)]
        name_to_cat = {}
    cats = {}
    for k, nm in names.items():
        cats[k] = name_to_cat.get(nm, max(list(name_to_cat.values()) + [0]) + 1 + k if name_to_cat else k + 1)
    anns, aid = [], 0
    for im in images:
        p = _find_image(images_dir, im["file_name"])
        img = _imread(p) if p else None
        if img is None:
            continue
        im.setdefault("width", img.shape[1])
        im.setdefault("height", img.shape[0])
        if scale != 1.0:  # 확대해서 추론하고 좌표는 원래 크기로 되돌린다
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]
        tl = _tiles(W, H, tile, overlap)
        dets = []
        for s in range(0, len(tl), batch):
            chunk = tl[s:s + batch]
            res = m.predict([img[y:y + h, x:x + w] for x, y, w, h in chunk], conf=conf, imgsz=imgsz or tile,
                            verbose=False, retina_masks=True)
            for (x, y, w, h), r in zip(chunk, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                bxs = r.boxes.xyxy.cpu().numpy()
                # 검출 모델(마스크 없음)이면 박스를 사각형 폴리곤으로
                polys = r.masks.xy if r.masks is not None else [
                    np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32) for b in bxs]
                for xy, b, c, sc in zip(polys, bxs, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if len(xy) < 3:
                        continue
                    # 타일 가장자리에 붙은 검출은 잘렸을 가능성 → 합칠 때 점수를 조금 깎는다
                    edge = (b[0] < 2 and x > 0) or (b[1] < 2 and y > 0) or (b[2] > w - 2 and x + w < W) \
                        or (b[3] > h - 2 and y + h < H)
                    dets.append({"poly": xy + [x, y], "box": b + [x, y, x, y], "cls": int(c),
                                 "score": float(sc), "rank": float(sc) * (0.8 if edge else 1.0)})
        dets.sort(key=lambda t: -t["rank"])
        kept = []
        for dt in dets:
            if all(dt["cls"] != k["cls"] or _ios(dt["box"], k["box"]) < merge_ios for k in kept):
                kept.append(dt)
        for dt in kept:
            dt["poly"] = dt["poly"] / scale
            dt["box"] = dt["box"] / scale
            aid += 1
            x0, y0, x1, y1 = dt["box"]
            anns.append({"id": aid, "image_id": im["id"], "category_id": cats[dt["cls"]],
                         "segmentation": [dt["poly"].round(1).ravel().tolist()],
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "area": float(Polygon(dt["poly"]).buffer(0).area), "score": dt["score"], "iscrowd": 0})
    out_d = {"images": images, "annotations": anns,
             "categories": [{"id": cats[k], "name": nm} for k, nm in names.items()]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_d, ensure_ascii=False), encoding="utf-8")
    print(f"예측 {len(anns)}개 · 이미지 {len(images)}장 → {out}")
    return out


# ---------------------------------------------------------------- eval
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-9)


def evaluate(gt, pred, iou_thr=0.5, conf=0.25, class_agnostic=True, files=None):
    """원본 이미지 단위 평가. class_agnostic: 클래스 무시하고 '쓰레기냐 아니냐'만
    (위치·무게 계산엔 검출 자체가 먼저 중요). 클래스별 재현율은 정답 클래스 기준."""
    G = json.loads(Path(gt).read_text(encoding="utf-8"))
    P = json.loads(Path(pred).read_text(encoding="utf-8"))
    gcat = {c["id"]: c["name"] for c in G.get("categories", [])}
    pcat = {c["id"]: c["name"] for c in P.get("categories", [])}
    fname = {im["id"]: im["file_name"] for im in G["images"]}
    pid = {im["file_name"]: im["id"] for im in P["images"]}
    use = set(files) if files else {im["file_name"] for im in P["images"]}
    gts, prs = {}, {}
    for a in G["annotations"]:
        f = fname[a["image_id"]]
        if f in use:
            gts.setdefault(f, []).append(a)
    inv_pid = {v: k for k, v in pid.items()}
    for a in P["annotations"]:
        f = inv_pid.get(a["image_id"])
        if f in use:
            prs.setdefault(f, []).append(a)

    def xyxy(a):
        x, y, w, h = a["bbox"]
        return (x, y, x + w, y + h)

    all_scores, all_tp, n_gt = [], [], 0
    tp_c = fp_c = 0
    per_cls = {}
    area_ratio, mask_iou = [], []
    for f in use:
        g = gts.get(f, [])
        p = sorted(prs.get(f, []), key=lambda a: -a.get("score", 1))
        n_gt += len(g)
        used = set()
        for a in p:
            best, bj = iou_thr, -1
            for j, b in enumerate(g):
                if j in used:
                    continue
                if not class_agnostic and pcat.get(a["category_id"]) != gcat.get(b["category_id"]):
                    continue
                v = _iou(xyxy(a), xyxy(b))
                if v >= best:
                    best, bj = v, j
            hit = bj >= 0
            all_scores.append(a.get("score", 1))
            all_tp.append(hit)
            if hit:
                used.add(bj)
            if a.get("score", 1) >= conf:
                if hit:
                    tp_c += 1
                    pa, ga = _poly_of(a), _poly_of(g[bj])
                    if ga.area > 0:
                        area_ratio.append(pa.area / ga.area)
                        u = pa.union(ga).area
                        mask_iou.append(pa.intersection(ga).area / u if u else 0)
                else:
                    fp_c += 1
        # 클래스별 재현율 (conf 이상에서 맞힌 정답)
        hit_conf = set()
        for j, b in enumerate(g):
            nm = gcat.get(b["category_id"], "?")
            per_cls.setdefault(nm, [0, 0])
            per_cls[nm][1] += 1
        for a in p:
            if a.get("score", 1) < conf:
                continue
            for j, b in enumerate(g):
                if j in hit_conf:
                    continue
                if _iou(xyxy(a), xyxy(b)) >= iou_thr and (class_agnostic or pcat.get(a["category_id"]) == gcat.get(b["category_id"])):
                    hit_conf.add(j)
                    per_cls[gcat.get(b["category_id"], "?")][0] += 1
                    break
    # AP50 (전체 이미지 기준, 101점 보간)
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)])) if len(tp) else 0.0
    P_ = tp_c / max(tp_c + fp_c, 1)
    R_ = tp_c / max(n_gt, 1)
    res = {
        "images": len(use), "gt": n_gt, "pred@conf": tp_c + fp_c, "conf": conf, "iou": iou_thr,
        "precision": P_, "recall": R_, "f1": 2 * P_ * R_ / max(P_ + R_, 1e-9), "ap50": ap,
        "count_ratio": (tp_c + fp_c) / max(n_gt, 1),
        "area_ratio_median": float(np.median(area_ratio)) if area_ratio else None,
        "mask_iou_median": float(np.median(mask_iou)) if mask_iou else None,
        "recall_by_class": {k: {"recall": v[0] / v[1], "n": v[1]} for k, v in sorted(per_cls.items(), key=lambda t: -t[1][1])},
    }
    return res


def print_eval(r):
    print(f"원본 이미지 {r['images']}장 · 정답 {r['gt']} · 예측(conf≥{r['conf']}) {r['pred@conf']}")
    print(f"  정밀도 {r['precision']:.3f} · 재현율 {r['recall']:.3f} · F1 {r['f1']:.3f} · AP50 {r['ap50']:.3f}")
    print(f"  개수 비율(예측/정답) {r['count_ratio']:.2f}")
    if r["area_ratio_median"] is not None:
        print(f"  맞힌 물체의 면적 비율 중앙값 {r['area_ratio_median']:.2f} · 마스크 IoU 중앙값 {r['mask_iou_median']:.2f}")
    print("  클래스별 재현율:")
    for k, v in r["recall_by_class"].items():
        print(f"    {k:<20} {v['recall']:.2f}  (n={v['n']})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.seg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--coco", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--single_class", action="store_true", help="클래스 무시하고 '쓰레기' 하나로")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_file", help="공식 분할 JSON (예: UAVVaste train_val_test_distribution_file.json)")
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo11s-seg.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name")
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coco", help="정답 COCO (이미지 목록·카테고리 이름 맞추기)")
    p.add_argument("--split", help="split.json — test 이미지만 추론")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.05, help="낮게 두고 eval에서 임계값 적용 (AP 계산용)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="추론 전 확대 배율 — 학습 데이터보다 해상도가 낮은 영상(예: 3 cm/px 정사영상)용")
    p = sub.add_parser("eval")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--split", help="split.json — test 이미지만 평가")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--by_class", action="store_true", help="클래스까지 맞아야 정답")
    p.add_argument("--iou", type=float, default=0.5, help="정답 인정 박스 IoU (업체 라벨처럼 박스가 느슨하면 0.3)")
    p.add_argument("--out", help="결과 JSON 저장")
    a = ap.parse_args(argv)
    files = json.loads(Path(a.split).read_text(encoding="utf-8"))["test"] if getattr(a, "split", None) else None
    if a.cmd == "prepare":
        prepare(a.coco, a.images, a.out, tile=a.tile, overlap=a.overlap, seed=a.seed, single_class=a.single_class,
                split_file=a.split_file)
    elif a.cmd == "train":
        train(a.data, a.model, a.epochs, a.imgsz, a.batch, name=a.name)
    elif a.cmd == "predict":
        predict(a.weights, a.images, a.out, coco=a.coco, files=files, tile=a.tile, conf=a.conf, scale=a.scale)
    elif a.cmd == "eval":
        r = evaluate(a.gt, a.pred, iou_thr=a.iou, conf=a.conf, class_agnostic=not a.by_class, files=files)
        print_eval(r)
        if a.out:
            Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
