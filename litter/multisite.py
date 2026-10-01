"""
다중 현장 통합 학습 — "해안쓰레기 학습 데이터가 부족해 여러 지역 데이터를 합쳐 학습한다".
AI Hub(국내 사진, 거리별 판) + 하와이(항공 정사영상 2 cm/px) + 튀니지(해변 드론 근접) + 문갑도(업체 정사영상 자동 라벨)
→ 통합 클래스 13개로 맞춰 한 데이터셋(data/seg_multi)으로 묶고, AI Hub 5에폭 가중치에서 이어서 학습한다.

  python -m litter.multisite pseudo            # 문갑도 정사영상 1024 타일 → AI Hub 모델(CPU) 의사라벨 (평가 칩 영역 제외)
  python -m litter.multisite build             # data/seg_multi 조립 (train 95 % / val 5 %)
  python -m litter.multisite train --epochs 12 # GPU가 빌 때까지 기다렸다가 학습 → runs/seg/multi_site
  python -m litter.multisite eval              # 문갑도·튀니지(crosseval) + 하와이(hawaii) 전/후 표
  python -m litter.multisite report            # 표(md/json) + 그림 21 + figures/cross_eval_multi.json

평가 데이터(문갑도 칩 48, 하와이 평가 420칩, 튀니지 test 179)는 학습·검증에 절대 넣지 않는다.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/seg_multi"
PSEUDO = OUT / "mgd_pseudo"
AIHUB = ROOT / "data/seg_aihub_gsd"
HAWAII = ROOT / "data/hawaii_yolo8"
TUN = ROOT / "data/external/tunisia_litter/Data"
ORTHO = ROOT / "data/company/ortho/MGD.tif"
CHIPS = ROOT / "data/company/chips"
BASE = ROOT / "runs/seg/aihub_gsd_det_s/weights/best.pt"
EVAL = ROOT / "runs/eval/multi_site"

# ---------------------------------------------------------------- 통합 클래스 (config.map_class 로 재질표에 연결됨)
CLASSES = ["styrofoam", "styrofoam_buoy", "buoy", "net", "rope", "plastic", "bottle", "metal", "glass", "wood",
           "tire", "cardboard", "other"]
CID = {n: i for i, n in enumerate(CLASSES)}
AIHUB_MAP = {"Metal": "metal", "Plastic_Buoy": "buoy", "PET_Bottle": "bottle", "Styrofoam_Piece": "styrofoam",
             "Plastic_ETC": "plastic", "Glass": "glass", "Rope": "rope", "Styrofoam_Buoy": "styrofoam_buoy",
             "Styrofoam_Box": "styrofoam", "Net": "net", "Plastic_Buoy_China": "buoy"}
HAWAII_MAP = {"buoy": "buoy", "unidentified object": "plastic", "net cloth": "net", "line fragment": "rope",
              "metal": "metal", "tire": "tire", "processed wood": "wood", "vessel": "other"}
TUN_MAP = {"Cardboar": "cardboard", "Fabrics": "other", "Glass": "glass", "Metal": "metal", "Other": "other",
           "Plastic": "plastic", "Wood": "wood"}


def _yaml_names(p):
    import yaml
    n = yaml.safe_load(Path(p).read_text(encoding="utf-8"))["names"]
    return [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)


def _link(src, dst):
    """같은 드라이브면 하드링크(디스크 절약), 아니면 복사."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _read_labels(p):
    if not Path(p).exists():
        return []
    rows = []
    for line in Path(p).read_text().splitlines():
        v = line.split()
        if len(v) >= 5:
            rows.append((int(v[0]), [float(x) for x in v[1:]]))
    return rows


def _to_box(vals):
    """YOLO 박스(cx cy w h) 또는 폴리곤(x y ...) → cx cy w h (정규화)."""
    if len(vals) == 4:
        return vals
    xy = np.asarray(vals).reshape(-1, 2)
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


# ---------------------------------------------------------------- 1. 문갑도 의사라벨
def _chip_rects():
    d = json.loads((CHIPS / "labels_coco.json").read_text(encoding="utf-8"))
    rects = []
    for im in d["images"]:
        m = re.match(r"(pos|neg)_(\d+)_(\d+)", im["file_name"])
        c, r = int(m[2]), int(m[3])
        rects.append((c, r, c + im["width"], r + im["height"], m[1]))
    return rects


def _overlaps(x0, y0, x1, y1, rects, margin=0):
    for a, b, c, d, _ in rects:
        if x0 < c + margin and x1 > a - margin and y0 < d + margin and y1 > b - margin:
            return True
    return False


def cmd_pseudo(a):
    """정사영상에서 1024 타일을 뽑아 AI Hub 모델로 의사라벨. 평가 칩(48장, 2048px) 영역과 겹치는 타일은 제외."""
    import rasterio
    from rasterio.windows import Window
    from ultralytics import YOLO

    from .seg import _imwrite

    rects = _chip_rects()
    tile = a.tile
    cand = set()
    for (c, r, _, _, kind) in rects:
        if kind != "pos":
            continue
        for dx in range(-a.reach, a.reach + 2):
            for dy in range(-a.reach, a.reach + 2):
                x0, y0 = c + dx * tile, r + dy * tile
                if x0 < 0 or y0 < 0:
                    continue
                if not _overlaps(x0, y0, x0 + tile, y0 + tile, rects, margin=a.margin):
                    cand.add((x0, y0))
    cand = sorted(cand)
    random.Random(a.seed).shuffle(cand)
    print(f"평가 칩 {len(rects)}장 제외 · 후보 타일 {len(cand)} (최대 {a.max_cand} 검사, {a.max_keep} 저장)")
    (PSEUDO / "images").mkdir(parents=True, exist_ok=True)
    (PSEUDO / "labels").mkdir(parents=True, exist_ok=True)
    m = YOLO(str(BASE))
    names = m.names
    ds = rasterio.open(ORTHO)
    W, H = ds.width, ds.height
    kept, meta, t0 = 0, [], time.time()
    n_checked = n_valid = 0
    stats = Counter()
    for i, (x0, y0) in enumerate(cand[:a.max_cand]):
        if kept >= a.max_keep:
            break
        if x0 + tile > W or y0 + tile > H:
            continue
        n_checked += 1
        arr = ds.read([1, 2, 3], window=Window(x0, y0, tile, tile))
        img = np.transpose(arr, (1, 2, 0))[:, :, ::-1].copy()
        if (img.max(axis=2) > 0).mean() < 0.97:
            continue
        n_valid += 1
        r = m.predict(img, conf=a.conf_low, imgsz=tile, device=a.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        sc = r.boxes.conf.cpu().numpy()
        hi = sc >= a.conf
        n_hi, n_lo = int(hi.sum()), int((~hi).sum())
        if n_hi == 0 or n_lo > a.lo_ratio * n_hi:  # 확실한 게 없거나 애매한 게 훨씬 많으면 버림 (미라벨→배경 오염 방지)
            continue
        lines = []
        for b, c, s in zip(r.boxes.xywhn.cpu().numpy()[hi], r.boxes.cls.cpu().numpy()[hi], sc[hi]):
            u = AIHUB_MAP[names[int(c)]]
            stats[u] += 1
            lines.append(f"{CID[u]} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f}")
        name = f"mgd_{x0}_{y0}"
        _imwrite(PSEUDO / "images" / f"{name}.jpg", img, 92)
        (PSEUDO / "labels" / f"{name}.txt").write_text("\n".join(lines))
        meta.append({"name": name, "col": x0, "row": y0, "n": n_hi, "n_low_dropped": n_lo})
        kept += 1
        if kept % 25 == 0:
            print(f"  검사 {n_checked} · 유효 {n_valid} · 저장 {kept} · 라벨 {sum(stats.values())} · {time.time() - t0:.0f}s")
    info = {"tile": tile, "conf": a.conf, "conf_low": a.conf_low, "lo_ratio": a.lo_ratio, "device": str(a.device), "checked": n_checked, "valid": n_valid, "kept": kept,
            "labels": sum(stats.values()), "by_class": dict(stats), "excluded_chip_rects": [list(r) for r in rects],
            "exclude_margin_px": a.margin, "tiles": meta, "sec": time.time() - t0}
    (PSEUDO / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {kept}타일 · 라벨 {info['labels']} {dict(stats)} · {info['sec']:.0f}s → {PSEUDO}")


# ---------------------------------------------------------------- 2. 조립
def _aihub_pick(n, seed=0):
    """클래스 균형: 드문 클래스가 든 판부터 클래스당 n/11 장씩, 나머지는 무작위."""
    names = _yaml_names(AIHUB / "data.yaml")
    lab_dir = AIHUB / "labels/train"
    per = {}
    for p in sorted(lab_dir.glob("*.txt")):
        per[p.stem] = Counter(c for c, _ in _read_labels(p))
    freq = Counter()
    for cnt in per.values():
        freq.update(cnt)
    rng = random.Random(seed)
    chosen = []
    cs = set()
    k = max(1, n // len(names))
    for c, _ in sorted(freq.items(), key=lambda t: t[1]):  # 드문 클래스부터
        pool = [s for s, cnt in per.items() if cnt[c] > 0 and s not in cs]
        rng.shuffle(pool)
        for s in pool[:k]:
            chosen.append(s)
            cs.add(s)
    rest = [s for s in per if s not in cs]
    rng.shuffle(rest)
    chosen += rest[:max(0, n - len(chosen))]
    return chosen[:n], names


def cmd_build(a):
    rng = random.Random(a.seed)
    for part in ("train", "val"):
        (OUT / "images" / part).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / part).mkdir(parents=True, exist_ok=True)
    items = []  # (source, img_path, lines)

    # AI Hub 일부
    stems, names = _aihub_pick(a.aihub, a.seed)
    for s in stems:
        lines = [f"{CID[AIHUB_MAP[names[c]]]} " + " ".join(f"{v:.5f}" for v in _to_box(v)) for c, v in _read_labels(AIHUB / "labels/train" / f"{s}.txt")]
        items.append(("aihub", AIHUB / "images/train" / f"{s}.jpg", lines))
    # 하와이 training 칩
    hn = _yaml_names(HAWAII / "data.yaml")
    for p in sorted((HAWAII / "images/train").glob("*.jpg")):
        lines = [f"{CID[HAWAII_MAP[hn[c]]]} " + " ".join(f"{v:.5f}" for v in _to_box(v)) for c, v in _read_labels(HAWAII / "labels/train" / f"{p.stem}.txt")]
        items.append(("hawaii", p, lines))
    # 튀니지 train (+valid) — test는 절대 넣지 않음
    tn = _yaml_names(TUN / "data.yaml")
    tun_items = []
    for split in (["train", "valid"] if a.tun_valid else ["train"]):
        for p in sorted((TUN / split / "images").iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lines = [f"{CID[TUN_MAP[tn[c]]]} " + " ".join(f"{v:.5f}" for v in _to_box(v)) for c, v in _read_labels(TUN / split / "labels" / f"{p.stem}.txt")]
            tun_items.append(("tunisia", p, lines))
    if a.tun_max and len(tun_items) > a.tun_max:  # 학습 시간 때문에 상한 (무작위)
        tun_items = random.Random(a.seed).sample(tun_items, a.tun_max)
    items += tun_items
    # 문갑도 의사라벨 (있으면)
    if not a.no_mgd and (PSEUDO / "images").exists():
        for p in sorted((PSEUDO / "images").glob("*.jpg")):
            lines = (PSEUDO / "labels" / f"{p.stem}.txt").read_text().splitlines()
            items.append(("mgd", p, lines))

    # 기존 내용 비우기 (이미지 단위 분할, 출처별 5 % 검증)
    for part in ("train", "val"):
        for sub in ("images", "labels"):
            for f in (OUT / sub / part).iterdir():
                f.unlink()
    comp = {}
    for src in sorted({t[0] for t in items}):
        grp = [t for t in items if t[0] == src]
        rng.shuffle(grp)
        n_val = max(1, int(round(len(grp) * a.val_frac)))
        for i, (_, p, lines) in enumerate(grp):
            part = "val" if i < n_val else "train"
            name = f"{src}_{p.stem}"
            _link(p, OUT / "images" / part / f"{name}{p.suffix.lower()}")
            (OUT / "labels" / part / f"{name}.txt").write_text("\n".join(lines))
            c = comp.setdefault(src, {"train_images": 0, "val_images": 0, "train_labels": 0, "val_labels": 0, "by_class": Counter()})
            c[f"{part}_images"] += 1
            c[f"{part}_labels"] += len(lines)
            c["by_class"].update(CLASSES[int(l.split()[0])] for l in lines)
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (OUT / "data.yaml").write_text(f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names_yaml}\n", encoding="utf-8")
    for c in comp.values():
        c["by_class"] = dict(c["by_class"])
    meta = {"classes": CLASSES, "maps": {"aihub": AIHUB_MAP, "hawaii": HAWAII_MAP, "tunisia": TUN_MAP}, "val_frac": a.val_frac,
            "composition": comp, "total_train": sum(c["train_images"] for c in comp.values()),
            "total_val": sum(c["val_images"] for c in comp.values())}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}  train {meta['total_train']} / val {meta['total_val']}")
    for src, c in comp.items():
        print(f"  {src:<8} train {c['train_images']:5d}장 {c['train_labels']:6d}라벨 · val {c['val_images']:4d}장 {c['val_labels']:5d}라벨")


# ---------------------------------------------------------------- 3. 학습
def gpu_used_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return 0


def wait_gpu(flag=None, max_mb=1500, poll=30):
    while True:
        ok_flag = (flag is None) or Path(flag).exists()
        mb = gpu_used_mb()
        if ok_flag and mb < max_mb:
            return
        print(f"  GPU 대기 (사용 {mb} MB, 플래그 {'있음' if ok_flag else '없음'}) → {poll}s")
        time.sleep(poll)


def cmd_train(a):
    from ultralytics import YOLO

    if not a.no_wait:
        wait_gpu(a.wait_flag)
    t0 = time.time()
    m = YOLO(str(a.base))
    r = m.train(data=str(OUT / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0,
                project=str(ROOT / "runs/seg"), name=a.name, exist_ok=True, patience=a.patience, cos_lr=True,
                close_mosaic=a.close_mosaic, workers=a.workers, flipud=0.5, fliplr=0.5, degrees=90, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
                plots=False)
    print(f"best → {Path(r.save_dir) / 'weights' / 'best.pt'} · {(time.time() - t0) / 60:.1f}분")


# ---------------------------------------------------------------- 4. 평가 (전/후)
def cmd_eval(a):
    """crosseval(문갑도·튀니지) + hawaii(×1·×2). 모델 키: aihub(5에폭), hawaii_ft8, multi."""
    from . import crosseval, hawaii
    from .seg import evaluate

    EVAL.mkdir(parents=True, exist_ok=True)
    keys = a.models.split(",")
    res = {"conf": a.conf, "iou": a.iou, "models": {k: crosseval.MODELS[k]["label"] for k in keys}, "results": {}}
    crosseval.DEVICE = a.device
    for ds in ("mgd", "tunisia"):
        gt, img_dir = (crosseval.mgd_gt() if ds == "mgd" else crosseval.tunisia_gt("test", 300))
        gt_path = EVAL / f"gt_{ds}.json"
        gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
        res["results"][ds] = {}
        for key in keys:
            for aug in ([False, True] if (a.tta and key in a.tta_models.split(",")) else [False]):
                tag = key + ("_tta" if aug else "")
                print(f"-- {ds} · {tag}")
                P, sec = crosseval.predict_coco(key, ds, gt, img_dir, tile=1024, conf=0.05, augment=aug)
                pp = EVAL / f"pred_{ds}_{tag}.json"
                pp.write_text(json.dumps(P, ensure_ascii=False), encoding="utf-8")
                e = evaluate(gt_path, pp, iou_thr=a.iou, conf=a.conf, class_agnostic=True)
                print(f"   P {e['precision']:.2f} R {e['recall']:.2f} F1 {e['f1']:.2f} AP50 {e['ap50']:.2f} ({sec:.0f}s)")
                res["results"][ds][tag] = {"eval": e, "sec": sec}
    # 하와이 평가 420칩
    files = [r["filename"] for r in hawaii.load_chips() if r["split"] == "eval"]
    res["results"]["hawaii"] = {}
    for key in keys:
        for scale in (1.0, 2.0):
            for aug in ([False, True] if (a.tta and key in a.tta_models.split(",")) else [False]):
                tag = f"{key}_x{scale:g}" + ("_tta" if aug else "")
                out = hawaii.OUT / f"pred_{key}_eval_x{scale:g}{'_tta' if aug else ''}.json"
                if not out.exists() or a.redo:
                    print(f"-- hawaii · {tag}")
                    hawaii.predict(key, files, out, conf=0.05, batch=4, device=a.device, wait=False, scale=scale, augment=aug)
                e = evaluate(hawaii.OUT / "labels_all.json", out, iou_thr=a.iou, conf=a.conf, class_agnostic=True, files=files)
                e2 = evaluate(hawaii.OUT / "labels_all.json", out, iou_thr=a.iou, conf=0.1, class_agnostic=True, files=files)
                print(f"   P {e['precision']:.2f} R {e['recall']:.2f} F1 {e['f1']:.2f} AP50 {e['ap50']:.2f} (conf0.1: R {e2['recall']:.2f})")
                res["results"]["hawaii"][tag] = {"eval": e, "eval_conf0.1": e2}
    (EVAL / "eval_multi.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {EVAL / 'eval_multi.json'}")


# ---------------------------------------------------------------- 5. 보고
DS_KO = {"mgd": "문갑도 업체 칩 48장", "tunisia": "튀니지 test 179장", "hawaii": "하와이 평가 420칩"}


SRC_KO = {"aihub": "AI Hub 해안 오염물질 (국내 사진→2~4 cm/px 거리별 판, 1024)", "hawaii": "하와이 항공 정사영상 칩 (2 cm/px, 640, Zenodo 8381113)",
          "tunisia": "튀니지 TUN-MarineLitter 해변 드론 근접 (640, Roboflow CC-BY)", "mgd": "문갑도 업체 정사영상 1024 타일 (3 cm/px, AI Hub 모델 의사라벨)"}


def data_table_md(meta):
    """학습 데이터 구성 표 (출처별 이미지·라벨 수 + 클래스 매핑) → docs/figures/multi_site_data_table.md"""
    comp = meta["composition"]
    L = ["# 다중 지역 통합 학습 데이터 구성 (`data/seg_multi`)", "",
         "| 출처 | 설명 | 학습 이미지 | 학습 라벨 | 검증 이미지 | 검증 라벨 |", "|---|---|---|---|---|---|"]
    for src, c in comp.items():
        L.append(f"| {src} | {SRC_KO.get(src, src)} | {c['train_images']:,} | {c['train_labels']:,} | {c['val_images']:,} | {c['val_labels']:,} |")
    L.append(f"| **합계** | | **{meta['total_train']:,}** | **{sum(c['train_labels'] for c in comp.values()):,}** | **{meta['total_val']:,}** | **{sum(c['val_labels'] for c in comp.values()):,}** |")
    L += ["", f"검증은 출처별 {meta['val_frac'] * 100:.0f} % 무작위. 평가용(문갑도 칩 48 · 하와이 평가 420칩 · 튀니지 test 179)은 학습·검증 어디에도 없음.", "",
          "## 통합 클래스별 라벨 수 (학습+검증)", "", "| 클래스 | " + " | ".join(comp) + " | 합계 |", "|---|" + "---|" * (len(comp) + 1)]
    for cls in meta["classes"]:
        vals = [c["by_class"].get(cls, 0) for c in comp.values()]
        if sum(vals):
            L.append(f"| {cls} | " + " | ".join(f"{v:,}" for v in vals) + f" | {sum(vals):,} |")
    L += ["", "## 원본 클래스 → 통합 클래스 매핑", ""]
    for src, mp in meta["maps"].items():
        L.append(f"- **{src}**: " + ", ".join(f"{k}→{v}" for k, v in mp.items()))
    L.append("- **mgd**: AI Hub 모델 예측 클래스를 aihub 매핑으로 변환 (conf≥0.4만 라벨)")
    return chr(10).join(L)


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    res = json.loads((EVAL / "eval_multi.json").read_text(encoding="utf-8"))
    labels = res["models"]
    L = [f"# 다중 현장 통합 모델 전/후 비교 (클래스 무시, IoU≥{res['iou']}, conf≥{res['conf']})", "",
         "| 데이터 | 모델 | 정답 | 예측 | 정밀도 | 재현율 | F1 | AP50 |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for ds, R in res["results"].items():
        for tag, r in R.items():
            e = r["eval"]
            key = tag.split("_x")[0].replace("_tta", "")
            nm = labels.get(key, key) + (" ×2" if "_x2" in tag else "") + (" +TTA" if tag.endswith("_tta") else "")
            L.append(f"| {DS_KO[ds]} | {nm} | {e['gt']} | {e['pred@conf']} | {e['precision']:.2f} | **{e['recall']:.2f}** | {e['f1']:.2f} | {e['ap50']:.2f} |")
            rows.append((ds, tag, nm, e))
    meta_p = OUT / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        L += ["", "## 학습 데이터 구성", "", "| 출처 | 학습 이미지 | 학습 라벨 | 검증 이미지 | 검증 라벨 |", "|---|---|---|---|---|"]
        for src, c in meta["composition"].items():
            L.append(f"| {src} | {c['train_images']} | {c['train_labels']} | {c['val_images']} | {c['val_labels']} |")
        L.append(f"| 합계 | {meta['total_train']} | {sum(c['train_labels'] for c in meta['composition'].values())} | {meta['total_val']} | {sum(c['val_labels'] for c in meta['composition'].values())} |")
        (ROOT / "docs/figures/multi_site_data_table.md").write_text(data_table_md(meta), encoding="utf-8")
    (EVAL / "compare.md").write_text("\n".join(L), encoding="utf-8")
    fig_json = ROOT / "docs/figures/cross_eval_multi.json"
    fig_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # 그림: 데이터 3개 × (재현율·정밀도·F1·AP50) 묶음 막대, 모델별 (TTA 제외, 하와이는 ×2)
    keys = [k for k in labels]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    cols = ["#9aa5b1", "#f4a261", "#2a9d8f", "#e63946"]
    for ax, ds in zip(axes, ("mgd", "tunisia", "hawaii")):
        R = res["results"].get(ds, {})
        vals = []
        for k in keys:
            tag = k if ds != "hawaii" else f"{k}_x2"
            e = R.get(tag, {}).get("eval")
            vals.append((e["recall"], e["precision"], e["f1"], e["ap50"]) if e else (0, 0, 0, 0))
        x = np.arange(4)
        w = 0.8 / max(len(keys), 1)
        for i, (k, v) in enumerate(zip(keys, vals)):
            b = ax.bar(x + (i - (len(keys) - 1) / 2) * w, v, w, label=labels[k], color=cols[i % len(cols)])
            for rect, val in zip(b, v):
                ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(["재현율", "정밀도", "F1", "AP50"])
        ax.set_ylim(0, 1.0)
        ax.set_title(DS_KO[ds] + (" (×2 확대 추론)" if ds == "hawaii" else ""), fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"다중 현장 통합 학습 전/후 (클래스 무시, IoU≥{res['iou']}, conf≥{res['conf']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = ROOT / "docs/figures/21_다중현장_통합모델_비교.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("\n".join(L))
    print(f"\n그림 → {p}\njson → {fig_json}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.multisite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pseudo")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.5, help="이 이상만 라벨로")
    p.add_argument("--conf_low", type=float, default=0.25, help="이 사이(애매)가 확실한 것보다 많으면 타일 버림")
    p.add_argument("--reach", type=int, default=6, help="양성 칩 주변 ±reach 타일 범위에서 후보")
    p.add_argument("--margin", type=int, default=256, help="평가 칩 둘레 여유 px (겹침 판정)")
    p.add_argument("--max_cand", type=int, default=900)
    p.add_argument("--max_keep", type=int, default=350)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="추론 장치 (GPU가 비면 0)")
    p.add_argument("--lo_ratio", type=float, default=1.0, help="애매한 탐지 수가 확실한 탐지 수의 이 배수를 넘으면 타일 버림")
    p = sub.add_parser("build")
    p.add_argument("--aihub", type=int, default=2000)
    p.add_argument("--tun_valid", action="store_true", help="튀니지 valid 분할도 학습에")
    p.add_argument("--tun_max", type=int, default=0, help="튀니지 이미지 상한 (0=전부)")
    p.add_argument("--no_mgd", action="store_true")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--close_mosaic", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--base", default=str(BASE))
    p.add_argument("--name", default="multi_site")
    p.add_argument("--wait_flag", default=str(ROOT / "runs/hawaii/ft8_eval_log.txt"), help="이 파일이 생기고 GPU가 비면 시작")
    p.add_argument("--no_wait", action="store_true")
    p = sub.add_parser("eval")
    p.add_argument("--models", default="aihub,hawaii_ft8,multi")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--device", default="0")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta_models", default="multi", help="TTA(augment=True)도 돌릴 모델 키")
    p.add_argument("--redo", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"pseudo": cmd_pseudo, "build": cmd_build, "train": cmd_train, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
