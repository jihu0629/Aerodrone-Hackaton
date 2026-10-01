"""
현장 일반화 교차평가 — 공개 해변쓰레기 드론 데이터(튀니지 TUN-MarineLitter) + 문갑도 업체 칩에
우리 탐지 모델(UAVVaste / AI Hub / YOLO-World 개방형)을 재학습 없이 그대로 적용해 본다.

  python -m litter.crosseval --out runs/eval/tunisia            # 전부 (CPU, 약 10분)
  python -m litter.crosseval --only tunisia --max_images 300
  python -m litter.crosseval --only mgd

- 모든 추론은 CPU (device='cpu'). GPU는 다른 작업이 쓰고 있다.
- 평가는 seg.evaluate 재사용: 클래스 무시, 박스 IoU 0.3, conf 0.25. 세그 폴리곤은 바운딩박스로.
- 튀니지(640×640)는 통째로 추론, 문갑도 칩(2048)은 seg.predict와 같은 1024 타일 추론.
- 결과: <out>/{gt,pred}_*.json, cross_eval.json, cross_eval.md, 19_튀니지_교차평가.jpg
"""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # GPU 사용 금지
import cv2
import numpy as np

from .seg import _imread, _imwrite, _ios, _tiles, evaluate, print_eval

ROOT = Path(__file__).resolve().parents[1]
TUN = ROOT / "data/external/tunisia_litter/Data"
CHIPS = ROOT / "data/company/chips"

MODELS = {
    "uavvaste": {"label": "UAVVaste 모델", "weights": ROOT / "runs/seg/uavvaste_det_s/weights/best.pt"},
    "aihub": {"label": "AI Hub 모델", "weights": ROOT / "runs/seg/aihub_gsd_det_s/weights/best.pt"},
    "world": {"label": "YOLO-World (개방형)", "weights": ROOT / "yolov8s-worldv2.pt",
              "classes": {"tunisia": ["plastic bottle", "plastic bag", "cardboard", "glass bottle", "metal can",
                                      "fabric", "wood debris", "litter"],
                          "mgd": ["styrofoam buoy", "fishing net", "rope", "plastic debris"]}},
}


# ---------------------------------------------------------------- 정답
def tunisia_gt(split="test", max_images=300, seed=0):
    """YOLO-seg 라벨(정규화 폴리곤) → COCO(박스 + 폴리곤). 클래스 이름은 data.yaml 순서."""
    import yaml

    names = yaml.safe_load((TUN / "data.yaml").read_text(encoding="utf-8"))["names"]
    fix = {"Cardboar": "Cardboard", "Fabrics": "Fabric"}
    names = [fix.get(n, n) for n in names]
    img_dir = TUN / split / "images"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(paths) > max_images:
        paths = sorted(random.Random(seed).sample(paths, max_images))
    images, anns = [], []
    for i, p in enumerate(paths):
        im = _imread(p)
        H, W = im.shape[:2]
        images.append({"id": i, "file_name": p.name, "width": W, "height": H})
        lp = TUN / split / "labels" / (p.stem + ".txt")
        if not lp.exists():
            continue
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0])
            xy = np.asarray(v[1:], float).reshape(-1, 2) * [W, H]
            if len(xy) == 2:  # 박스 형식(cx cy w h)
                cx, cy, w, h = xy.ravel()
                x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                poly = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            else:
                x0, y0 = xy.min(0)
                x1, y1 = xy.max(0)
                poly = [xy.round(1).ravel().tolist()]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c + 1, "segmentation": poly,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0})
    return {"images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}, img_dir


def mgd_gt():
    d = json.loads((CHIPS / "labels_coco.json").read_text(encoding="utf-8"))
    return d, CHIPS / "images"


# ---------------------------------------------------------------- 추론 (CPU)
def _load(key, dataset):
    from ultralytics import YOLO

    spec = MODELS[key]
    m = YOLO(str(spec["weights"]))
    if "classes" in spec:
        m.set_classes(spec["classes"][dataset])
    return m


def _run_tiles(m, img, tile, overlap, conf, merge_ios=0.6, batch=4):
    """seg.predict와 같은 타일 추론·병합이되 device='cpu'. 반환: [(x0,y0,x1,y1,cls,score)]"""
    H, W = img.shape[:2]
    tl = _tiles(W, H, tile, overlap) if (W > tile or H > tile) else [(0, 0, W, H)]
    imgsz = tile if (W > tile or H > tile) else int(np.ceil(max(W, H) / 32) * 32)
    dets = []
    for s in range(0, len(tl), batch):
        chunk = tl[s:s + batch]
        res = m.predict([img[y:y + h, x:x + w] for x, y, w, h in chunk], conf=conf, imgsz=imgsz,
                        device="cpu", verbose=False)
        for (x, y, w, h), r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b, c, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                edge = (b[0] < 2 and x > 0) or (b[1] < 2 and y > 0) or (b[2] > w - 2 and x + w < W) \
                    or (b[3] > h - 2 and y + h < H)
                dets.append({"box": b + [x, y, x, y], "cls": int(c), "score": float(sc),
                             "rank": float(sc) * (0.8 if edge else 1.0)})
    dets.sort(key=lambda t: -t["rank"])
    kept = []
    for dt in dets:
        if all(dt["cls"] != k["cls"] or _ios(dt["box"], k["box"]) < merge_ios for k in kept):
            kept.append(dt)
    return kept


def predict_coco(key, dataset, gt, img_dir, tile=1024, overlap=0.2, conf=0.25):
    m = _load(key, dataset)
    names = m.names
    anns = []
    t0 = time.time()
    for im in gt["images"]:
        img = _imread(img_dir / im["file_name"])
        if img is None:
            continue
        for dt in _run_tiles(m, img, tile, overlap, conf):
            x0, y0, x1, y1 = [float(v) for v in dt["box"]]
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": dt["cls"] + 1,
                         "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                         "score": dt["score"], "iscrowd": 0})
    sec = time.time() - t0
    out = {"images": gt["images"], "annotations": anns,
           "categories": [{"id": k + 1, "name": v} for k, v in names.items()]}
    del m
    gc.collect()
    return out, sec


# ---------------------------------------------------------------- 그림
def _put_korean(img, text, xy, size=22, color=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    dr.text(xy, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def figure(gt, img_dir, preds, out_path, n=6, cell=320, seed=0, conf=0.25):
    """열 = 샘플 n장, 행 = 모델. 각 셀: 정답(초록) + 해당 모델 탐지(빨강)."""
    by_img = {}
    for a in gt["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    ids = [im["id"] for im in gt["images"] if len(by_img.get(im["id"], [])) >= 1]
    rng = random.Random(seed)
    # 다양한 클래스가 보이도록: 객체 2개 이상인 이미지를 우선
    rich = [i for i in ids if len(by_img[i]) >= 2]
    pick = rng.sample(rich, min(n, len(rich)))
    pick += rng.sample([i for i in ids if i not in pick], n - len(pick))
    imgs = {im["id"]: im for im in gt["images"]}
    rows = []
    for key, P in preds.items():
        pb = {}
        for a in P["annotations"]:
            if a.get("score", 1) >= conf:
                pb.setdefault(a["image_id"], []).append(a)
        cells = []
        for iid in pick:
            im = _imread(img_dir / imgs[iid]["file_name"])
            s = cell / max(im.shape[:2])
            im = cv2.resize(im, None, fx=s, fy=s)
            canvas = np.zeros((cell, cell, 3), np.uint8)
            canvas[:im.shape[0], :im.shape[1]] = im
            for a in by_img.get(iid, []):
                x, y, w, h = [v * s for v in a["bbox"]]
                cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)
            for a in pb.get(iid, []):
                x, y, w, h = [v * s for v in a["bbox"]]
                cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cells.append(canvas)
        row = np.concatenate(cells, 1)
        label = np.zeros((34, row.shape[1], 3), np.uint8)
        label = _put_korean(label, f"{MODELS[key]['label']}  —  초록: 정답  빨강: 탐지(conf≥{conf})", (8, 5))
        rows.append(np.concatenate([label, row], 0))
    fig = np.concatenate(rows, 0)
    _imwrite(out_path, fig, 90)
    return out_path


# ---------------------------------------------------------------- 보고
def to_markdown(results):
    L = ["| 데이터 | 모델 | 이미지 | 정답 | 예측 | 정밀도 | 재현율 | F1 | AP50 | CPU 초 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, R in results.items():
        for key, r in R.items():
            e = r["eval"]
            L.append(f"| {r['dataset_label']} | {MODELS[key]['label']} | {e['images']} | {e['gt']} | {e['pred@conf']} | "
                     f"{e['precision']:.2f} | **{e['recall']:.2f}** | {e['f1']:.2f} | {e['ap50']:.2f} | {r['sec']:.0f} |")
    L.append("")
    for ds, R in results.items():
        classes = []
        for r in R.values():
            for k in r["eval"]["recall_by_class"]:
                if k not in classes:
                    classes.append(k)
        if not classes:
            continue
        lab = next(iter(R.values()))["dataset_label"]
        L.append(f"**{lab} 클래스별 재현율 (정답 클래스 기준, n=정답 수)**\n")
        L.append("| 클래스 | n | " + " | ".join(MODELS[k]["label"] for k in R) + " |")
        L.append("|---|---|" + "---|" * len(R))
        n_of = {k: v["n"] for r in R.values() for k, v in r["eval"]["recall_by_class"].items()}
        for c in classes:
            vals = [f"{r['eval']['recall_by_class'].get(c, {'recall': float('nan')})['recall']:.2f}" for r in R.values()]
            L.append(f"| {c} | {n_of[c]} | " + " | ".join(vals) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.crosseval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/eval/tunisia")
    ap.add_argument("--only", choices=["tunisia", "mgd"], help="한 데이터만")
    ap.add_argument("--models", default="uavvaste,aihub,world")
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--fig", default="docs/figures/19_튀니지_교차평가.jpg")
    ap.add_argument("--fig_json", default="docs/figures/cross_eval.json")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    keys = a.models.split(",")
    datasets = [a.only] if a.only else ["tunisia", "mgd"]

    results = {}
    for ds in datasets:
        if ds == "tunisia":
            gt, img_dir = tunisia_gt(a.split, a.max_images)
            lab = f"튀니지 TUN-MarineLitter {a.split}"
            tile = 1024
        else:
            gt, img_dir = mgd_gt()
            lab = "문갑도 업체 칩"
            tile = 1024
        gt_path = out / f"gt_{ds}.json"
        gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
        print(f"\n== {lab}: 이미지 {len(gt['images'])} · 정답 {len(gt['annotations'])}")
        results[ds] = {}
        preds = {}
        for key in keys:
            print(f"-- {MODELS[key]['label']} (CPU)")
            P, sec = predict_coco(key, ds, gt, img_dir, tile=tile, conf=0.05)  # 낮게 받아 AP도 계산
            pp = out / f"pred_{ds}_{key}.json"
            pp.write_text(json.dumps(P, ensure_ascii=False), encoding="utf-8")
            e = evaluate(gt_path, pp, iou_thr=a.iou, conf=a.conf, class_agnostic=True)
            print_eval(e)
            print(f"  추론 {sec:.0f}초")
            results[ds][key] = {"dataset_label": lab, "eval": e, "sec": sec, "pred": str(pp)}
            preds[key] = P
        if ds == "tunisia":
            fig = ROOT / a.fig
            fig.parent.mkdir(parents=True, exist_ok=True)
            figure(gt, img_dir, preds, fig, conf=a.conf)
            print(f"그림 → {fig}")

    md = to_markdown(results)
    (out / "cross_eval.md").write_text(md, encoding="utf-8")
    meta = {"conf": a.conf, "iou": a.iou, "device": "cpu", "class_agnostic": True,
            "note": "세그 폴리곤→바운딩박스. 튀니지 640 통째 추론, 문갑도 2048 칩은 1024 타일.",
            "results": results}
    (out / "cross_eval.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    fj = ROOT / a.fig_json
    if not a.only or not fj.exists():
        fj.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    else:  # 일부만 돌렸으면 기존 json에 합친다
        old = json.loads(fj.read_text(encoding="utf-8"))
        old.setdefault("results", {}).update(results)
        fj.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
