"""
정사영상 + GeoJSON 라벨 → 칩(chip) 이미지 + COCO.

업체 정사영상은 섬 전체(약 2.5 × 2.2 km, 3 cm/px → 수만 × 수만 픽셀)라서
한 장으로는 못 다룬다. 라벨이 있는 곳 주변을 정사각 칩으로 잘라내고,
라벨 없는 칩도 일부 넣어서(음성 표본 — 바위·풀·파도를 쓰레기로 착각하지
않게) 학습/평가용 COCO를 만든다.

칩마다 지도좌표 변환(geo_transform)과 좌표계를 COCO images에 기록해 두므로,
모델 예측도 그대로 지도좌표로 돌아간다 (`python -m litter run`이 이를 읽음).

  python -m litter.ortho chips --ortho 문갑도.tif --geojson MGD_쓰레기.json --out data/company_chips
  python -m litter.ortho grid  --ortho 문갑도.tif --out data/company_grid   # 섬 전체 추론용 격자
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

MATERIAL = {"STY": "스티로폼", "ROP": "로프", "FIS": "어망", "PLA": "플라스틱"}


def _open(path):
    import rasterio

    return rasterio.open(path)


def _read_chip(ds, col, row, size):
    from rasterio.windows import Window

    w = Window(col, row, size, size)
    arr = ds.read([1, 2, 3], window=w, boundless=True, fill_value=0)
    return np.transpose(arr, (1, 2, 0))[:, :, ::-1].copy(), ds.window_transform(w)  # RGB→BGR (cv2)


def _features_px(ds, geojson):
    """GeoJSON(위경도) → 정사영상 픽셀 좌표 폴리곤."""
    from pyproj import Transformer

    d = json.loads(Path(geojson).read_text(encoding="utf-8"))
    T = Transformer.from_crs(4326, ds.crs, always_xy=True)
    inv = ~ds.transform
    out = []
    for f in d["features"]:
        ring = np.asarray(f["geometry"]["coordinates"][0], float)
        mx, my = T.transform(ring[:, 0], ring[:, 1])
        c, r = inv * (np.asarray(mx), np.asarray(my))
        out.append({"poly": np.stack([c, r], 1), "props": f["properties"]})
    return out


def _is_valid(img, min_frac=0.5):
    """정사영상 바깥(검은색 nodata)이 대부분인 칩은 버린다."""
    return (img.max(axis=2) > 0).mean() >= min_frac


def chips(ortho, geojson, out, size=2048, neg_per_pos=1.0, seed=0, quality=92):
    from .seg import _imwrite

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = _open(ortho)
    feats = _features_px(ds, geojson)
    W, H = ds.width, ds.height
    # 라벨을 포함하는 격자 칸 (size 간격) — 같은 칸 라벨은 한 칩에
    cells = {}
    for k, f in enumerate(feats):
        cx, cy = f["poly"].mean(0)
        if not (0 <= cx < W and 0 <= cy < H):
            print(f"  ⚠️ 정사영상 밖 라벨: seq {f['props'].get('detection_seq')}")
            continue
        # 라벨이 칩 가장자리에 걸리지 않게 칩 원점을 라벨 쪽으로 맞춘다
        cells.setdefault((int(cx // size), int(cy // size)), []).append(k)
    images, anns, cats = [], [], {}
    aid = 0

    def add(img, T, name, members):
        nonlocal aid
        iid = len(images)
        _imwrite(out / "images" / name, img, quality)
        images.append({"id": iid, "file_name": name, "width": img.shape[1], "height": img.shape[0],
                       "geo_transform": list(T)[:6], "crs": ds.crs.to_string()})
        for k, (col, row) in members:
            f = feats[k]
            p = f["poly"] - [col, row]
            mat = f["props"].get("material_code", "UNK")
            name_ = MATERIAL.get(mat, mat)
            cats.setdefault(name_, len(cats) + 1)
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": cats[name_],
                         "segmentation": [p[:-1].round(1).ravel().tolist()],
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
                         "attributes": {"item_id": str(f["props"].get("detection_seq")),
                                        **{k2: v for k2, v in f["props"].items()
                                           if k2 in ("weight_kg", "area_sqm", "material_code")}}})

    pos = set()
    for (gx, gy), ks in sorted(cells.items()):
        # 칸 안 라벨들의 중심에 칩을 놓는다 (가장자리 잘림 방지)
        c = np.mean([feats[k]["poly"].mean(0) for k in ks], axis=0)
        col, row = int(c[0] - size / 2), int(c[1] - size / 2)
        img, T = _read_chip(ds, col, row, size)
        members = [(k, (col, row)) for k in range(len(feats))
                   if (feats[k]["poly"] >= [col, row]).all() and (feats[k]["poly"] < [col + size, row + size]).all()]
        add(img, T, f"pos_{col}_{row}.jpg", members)
        pos.add((gx, gy))
    # 음성 칩: 라벨 근처(같은 해안) 위주로 — 라벨 칸에서 1~3칸 떨어진 곳
    rng = random.Random(seed)
    n_neg = int(round(len(pos) * neg_per_pos))
    cand = {(gx + dx, gy + dy) for gx, gy in pos for dx in range(-3, 4) for dy in range(-3, 4)} - pos
    cand = sorted(c for c in cand if 0 <= c[0] * size < W and 0 <= c[1] * size < H)
    rng.shuffle(cand)
    n = 0
    for gx, gy in cand:
        if n >= n_neg:
            break
        img, T = _read_chip(ds, gx * size, gy * size, size)
        if not _is_valid(img):
            continue
        add(img, T, f"neg_{gx * size}_{gy * size}.jpg", [])
        n += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": nm} for nm, i in cats.items()]}
    (out / "labels_coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"칩 {len(images)}장 (라벨 있음 {len(pos)} · 음성 {n}) · 라벨 {len(anns)}/{len(feats)} → {out}")
    return out


def grid(ortho, out, size=2048, quality=90, skip_empty=0.5):
    """섬 전체를 칩으로 — 학습한 모델로 전체 해안을 훑을 때."""
    from .seg import _imwrite

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = _open(ortho)
    images = []
    for row in range(0, ds.height, size):
        for col in range(0, ds.width, size):
            img, T = _read_chip(ds, col, row, size)
            if not _is_valid(img, skip_empty):
                continue
            name = f"g_{col}_{row}.jpg"
            _imwrite(out / "images" / name, img, quality)
            images.append({"id": len(images), "file_name": name, "width": size, "height": size,
                           "geo_transform": list(T)[:6], "crs": ds.crs.to_string()})
    coco = {"images": images, "annotations": [], "categories": []}
    (out / "images.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"격자 칩 {len(images)}장 → {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.ortho", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chips")
    p.add_argument("--ortho", required=True)
    p.add_argument("--geojson", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--neg", type=float, default=1.0, help="라벨 칩 1장당 음성 칩 수")
    p = sub.add_parser("grid")
    p.add_argument("--ortho", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=2048)
    a = ap.parse_args(argv)
    if a.cmd == "chips":
        chips(a.ortho, a.geojson, a.out, size=a.size, neg_per_pos=a.neg)
    else:
        grid(a.ortho, a.out, size=a.size)


if __name__ == "__main__":
    main()
