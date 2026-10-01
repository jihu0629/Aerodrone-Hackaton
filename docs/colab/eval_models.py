"""Colab용 독립 평가 스크립트 — 가중치 여러 개를 세 데이터셋(문갑도 업체 칩 · 하와이 해안 · 튀니지 해변)에서
클래스 무시 · IoU 0.3 기준 재현율/정밀도로 비교한다. litter 패키지 의존 없음.

사용: python eval_models.py --pack /content/evalpack --weights a.pt b.pt --out /content/eval_out
     YOLO-World 는 --world "yolov8s-worldv2.pt" (클래스는 데이터셋별로 자동 지정)
"""
import argparse, csv, json, os, glob
from pathlib import Path

import numpy as np
import cv2

IOU = 0.3
CONF = 0.25


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(gt, det):
    """gt, det: [N,4] → tp, fp, fn (탐욕 매칭, 클래스 무시)"""
    if len(gt) == 0:
        return 0, len(det), 0
    if len(det) == 0:
        return 0, 0, len(gt)
    M = iou_matrix(gt, det)
    used_g, used_d, tp = set(), set(), 0
    pairs = sorted(((M[i, j], i, j) for i in range(len(gt)) for j in range(len(det))), reverse=True)
    for v, gi, di in pairs:
        if v < IOU:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di); tp += 1
    return tp, len(det) - tp, len(gt) - tp


# ---------- 데이터셋 로더: {이미지경로: [[x1,y1,x2,y2], ...]} ----------
def load_company(pack):
    d = Path(pack) / "company_chips"
    coco = json.load(open(d / "labels_coco.json", encoding="utf-8"))
    byid = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {str(d / "images" / fn): [] for fn in byid.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[str(d / "images" / byid[a["image_id"]])].append([x, y, x + w, y + h])
    return gt


def load_hawaii(pack):
    d = Path(pack) / "hawaii"
    gt = {}
    for r in csv.DictReader(open(d / "evaluation_data.csv")):
        p = str(d / "chips" / r["filename"])
        if os.path.exists(p):
            gt.setdefault(p, []).append([float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])])
    return gt


def load_tunisia(pack):
    d = Path(pack) / "tunisia" / "test"
    gt = {}
    for p in sorted(glob.glob(str(d / "images" / "*.jpg"))):
        lab = d / "labels" / (Path(p).stem + ".txt")
        H, W = cv2.imread(p).shape[:2]
        boxes = []
        if lab.exists():
            for line in open(lab):
                v = line.split()
                if len(v) < 5:
                    continue
                if len(v) == 5:  # cls cx cy w h
                    cx, cy, w, h = map(float, v[1:5])
                    boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
                else:  # cls x1 y1 x2 y2 ... (폴리곤) → 바운딩박스
                    xy = np.array(list(map(float, v[1:]))).reshape(-1, 2) * [W, H]
                    boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
        gt[p] = boxes
    return gt


WORLD_CLASSES = {
    "company": ["styrofoam buoy", "fishing net", "rope", "plastic debris", "styrofoam"],
    "hawaii": ["buoy", "fishing net", "rope", "tire", "wood debris", "plastic debris"],
    "tunisia": ["plastic bottle", "plastic bag", "cardboard", "glass bottle", "metal can", "fabric", "wood debris", "litter"],
}


def evaluate(model, gt, imgsz, batch=16, device=""):
    paths = list(gt)
    TP = FP = FN = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = model.predict(chunk, conf=CONF, imgsz=imgsz, verbose=False, device=device)
        for p, r in zip(chunk, res):
            det = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            tp, fp, fn = match(gt[p], det)
            TP += tp; FP += fp; FN += fn
    rec = TP / (TP + FN) if TP + FN else 0.0
    prec = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return dict(images=len(paths), gt=TP + FN, tp=TP, fp=FP, fn=FN, recall=round(rec, 3), precision=round(prec, 3), f1=round(f1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--weights", nargs="*", default=[], help="학습된 가중치들 (이름:경로 또는 경로)")
    ap.add_argument("--world", help="YOLO-World 가중치 (예: yolov8s-worldv2.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=400, help="데이터셋당 최대 이미지 수")
    ap.add_argument("--device", default="", help="cpu / 0")
    a = ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(a.out, exist_ok=True)
    sets = {"company": (load_company(a.pack), 2048), "hawaii": (load_hawaii(a.pack), 640), "tunisia": (load_tunisia(a.pack), 640)}
    for k in sets:
        gt, sz = sets[k]
        if len(gt) > a.max:
            keys = sorted(gt, key=lambda p: (len(gt[p]) == 0, p))[: a.max]  # 정답 있는 이미지 우선
            gt = {p: gt[p] for p in keys}
        sets[k] = (gt, sz)
    rows = []
    for w in a.weights:
        name, path = (w.split(":", 1) if ":" in w and not os.path.exists(w) else (Path(w).stem, w))
        m = YOLO(path)
        for k, (gt, sz) in sets.items():
            r = evaluate(m, gt, sz, device=a.device); r.update(model=name, dataset=k); rows.append(r); print(r)
    if a.world:
        m = YOLO(a.world)
        for k, (gt, sz) in sets.items():
            m.set_classes(WORLD_CLASSES[k])
            r = evaluate(m, gt, sz, device=a.device); r.update(model="YOLO-World(" + ",".join(WORLD_CLASSES[k][:3]) + "…)", dataset=k); rows.append(r); print(r)
    json.dump(rows, open(Path(a.out) / "cross_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(Path(a.out) / "cross_eval.md", "w", encoding="utf-8") as f:
        f.write("| 모델 | 데이터 | 이미지 | 정답 | 재현율 | 정밀도 | F1 |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['dataset']} | {r['images']} | {r['gt']} | {r['recall']} | {r['precision']} | {r['f1']} |\n")
    print(open(Path(a.out) / "cross_eval.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
