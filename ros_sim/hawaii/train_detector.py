"""
하와이 항공 칩으로 쓰레기 탐지 모델(YOLO)을 학습한다.

공정한 평가를 위해 시뮬레이션 월드에 깐 구역(build_world.BBOX)의 칩은 학습·검증에서
모두 뺀다. 학습/검증 분할은 데이터셋이 준 training/evaluation CSV를 그대로 따른다.

출력: runs/hawaii_det/ (git 제외) — dataset/, train/weights/best.pt
사용: python ros_sim/hawaii/train_detector.py [--model yolov8s.pt --epochs 30]
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

from ultralytics import YOLO

import build_world as BW

ROOT = BW.ROOT
SRC = BW.CHIPS
OUT = ROOT / "runs/hawaii_det"
CLASSES = ["buoy", "unidentified object", "net cloth", "line fragment",
           "metal", "tire", "processed wood", "vessel"]  # CSV class 1..8 순서


def in_sim_area(name):
    t = (SRC / f"{name}.aux.xml").read_text()
    if f'"EPSG","{BW.EPSG}"]]</SRS>' not in t:
        return False
    gt = [float(v) for v in re.search(r"<GeoTransform>(.*?)</GeoTransform>", t).group(1).split(",")]
    b = BW.BBOX
    return b[0] - 50 <= gt[0] <= b[2] + 50 and b[1] - 50 <= gt[3] <= b[3] + 50


def build(split, csv_path):
    rows = {}
    for r in csv.DictReader(open(csv_path)):
        rows.setdefault(r["filename"], []).append(r)
    img_dir = OUT / "dataset/images" / split
    lab_dir = OUT / "dataset/labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for name, rs in rows.items():
        if in_sim_area(name):
            skipped += 1
            continue
        shutil.copy(SRC / name, img_dir / name)
        with open(lab_dir / (Path(name).stem + ".txt"), "w") as f:
            for r in rs:
                x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                f.write(f"{int(r['class']) - 1} {(x0 + x1) / 1280:.6f} {(y0 + y1) / 1280:.6f} "
                        f"{(x1 - x0) / 640:.6f} {(y1 - y0) / 640:.6f}\n")
        kept += 1
    return kept, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    if (OUT / "dataset").exists():
        shutil.rmtree(OUT / "dataset")
    data_dir = SRC.parent
    tr = build("train", data_dir / "training_data.csv")
    va = build("val", data_dir / "evaluation_data.csv")
    print(f"학습 {tr[0]}장 (시뮬 구역 제외 {tr[1]}) · 검증 {va[0]}장 (제외 {va[1]})")
    yaml = OUT / "dataset/data.yaml"
    yaml.write_text(f"path: {OUT / 'dataset'}\ntrain: images/train\nval: images/val\n"
                    f"names: {CLASSES}\n")
    YOLO(a.model).train(data=str(yaml), epochs=a.epochs, imgsz=640, batch=16, device=a.device,
                        project=str(OUT), name="train", exist_ok=True, plots=True, verbose=False)


if __name__ == "__main__":
    main()
