"""
AI Hub 「해안 오염물질 데이터」 → 학습용 COCO.

AI Hub 웹 다운로드는 tar 안에 zip을 1 GB씩 쪼갠 조각(part0, part1073741824, ...)으로 담아 준다.
zip은 끝에 목차가 있어서 다 받기 전엔 보통 못 연다. 하지만 각 파일 앞에 붙은 로컬 헤더에
크기가 적혀 있으므로, **받은 앞부분만 순서대로 읽어도 이미지를 하나씩 꺼낼 수 있다**
(.crdownload처럼 받는 중인 파일도 됨).

  # 1) 받은 만큼 이미지 꺼내기 (라벨 조건에 맞는 것만, 최대 N장)
  python -m litter.aihub extract --src "미확인 55771.crdownload" --labels "TL_해안쓰레기(Bbox).zip.part0" \
         --out data/aihub/images --max 40000
  # 2) 라벨(LabelMe JSON) → COCO (꺼낸 이미지만)
  python -m litter.aihub convert --labels "TL_해안쓰레기(Bbox).zip.part0" --images data/aihub/images \
         --out data/aihub/coco_bbox.json

라벨 특징 (표본 분석):
  - 클래스 11종: PET_Bottle, Metal, Plastic_ETC, Styrofoam_Piece, Rope, Glass, Styrofoam_Buoy,
    Plastic_Buoy, Net, Plastic_Buoy_China, Styrofoam_Box
  - 메타: device(드론/폰/CCTV), altitude(m), gtype(sand/gravel/mud) — 이미지는 1200×800으로 축소됨
  - 고도 5 m 53%, 10 m 36%, 20~30 m 6% → 해상도(cm/px)를 계산해 '거리별 학습'에 쓴다
"""
import argparse
import json
import re
import struct
import tarfile
import zlib
from pathlib import Path

import numpy as np

# 기종별 수평 화각(도) — 1200폭 이미지의 해상도(cm/px) 계산용 (대각 화각에서 3:2 기준 환산, 근사)
HFOV = {"Phantom4Pro": 73.7, "Phantom4ProRTK": 73.7, "Phantom4Advanced": 73.7, "Phantom4": 73.7,
        "Mavic2Pro": 65.5, "MavicAir2S": 75.0, "MavicAir2": 72.0, "Mini2": 71.0}
DRONE_PREFIX = ("Phantom", "Mavic", "Mini", "Inspire", "Matrice", "Air")


def gsd_cm(meta):
    """1200폭 이미지 한 픽셀이 지면에서 몇 cm인지 (드론·고도 있을 때만)."""
    alt, dev, W = meta.get("altitude"), meta.get("device") or "", meta.get("imageWidth") or 1200
    if not alt or alt <= 0 or not dev.startswith(DRONE_PREFIX):
        return None
    hf = HFOV.get(dev, 72.0)
    return float(2 * alt * np.tan(np.radians(hf / 2)) / W * 100)


# ------------------------------------------------------------------ 라벨 읽기
def _loads(b):
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        try:  # 끝에 쉼표가 붙은 깨진 JSON 일부 복구
            return json.loads(re.sub(rb",\s*([\]}])", rb"\1", b).decode("utf-8"))
        except Exception:
            return None


def load_labels(label_zip):
    """{이미지 파일명: LabelMe dict} — imageData는 버린다."""
    import zipfile

    z = zipfile.ZipFile(label_zip)
    out, bad = {}, 0
    for n in z.namelist():
        if not n.endswith(".json"):
            continue
        j = _loads(z.read(n))
        if j is None:
            bad += 1
            continue
        j.pop("imageData", None)
        out[Path(j.get("imagePath") or Path(n).with_suffix(".jpg").name).name] = j
    print(f"라벨 {len(out):,}개 읽음 (깨진 JSON {bad}개 제외) ← {Path(label_zip).name}")
    return out


def select(labels, drone_only=True, max_n=None, seed=0):
    """학습에 쓸 이미지 고르기: 드론만, 높은 고도는 전부 + 낮은 고도는 표본 (고도 분포 균형)."""
    rng = np.random.default_rng(seed)
    keep = {}
    for k, j in labels.items():
        if not j.get("shapes"):
            continue
        if drone_only and gsd_cm(j) is None:
            continue
        keep[k] = j
    if max_n and len(keep) > max_n:
        alt = np.array([keep[k].get("altitude") or 0 for k in keep])
        names = np.array(list(keep))
        hi = names[alt >= 15]
        lo = names[alt < 15]
        n_lo = max(0, max_n - len(hi))
        lo = rng.choice(lo, min(n_lo, len(lo)), replace=False)
        keep = {k: keep[k] for k in list(hi[:max_n]) + list(lo)}
    return keep


# ------------------------------------------------------------------ 받는 중인 tar/zip에서 꺼내기
def _tar_stream(path):
    """tar(받는 중이어도 됨) 안의 zip 조각들을 순서대로 이어 붙인 바이트 스트림."""
    fh = open(path, "rb")
    t = tarfile.open(fileobj=fh, mode="r|")
    for m in t:
        if not m.isfile():
            continue
        f = t.extractfile(m)
        while True:
            try:
                b = f.read(8 << 20)
            except Exception:  # 받다 만 마지막 조각
                return
            if not b:
                break
            yield b


class _Buf:
    def __init__(self, it):
        self.it, self.buf, self.eof = it, bytearray(), False

    def need(self, n):
        while len(self.buf) < n and not self.eof:
            try:
                self.buf += next(self.it)
            except StopIteration:
                self.eof = True
        return len(self.buf) >= n

    def take(self, n):
        b = bytes(self.buf[:n])
        del self.buf[:n]
        return b


def extract(src, out, wanted=None, max_n=None):
    """src: AI Hub tar(.crdownload 포함) 또는 zip 조각을 이어 붙인 파일. wanted: 꺼낼 파일명 집합."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = _Buf(_tar_stream(src))
    n_seen = n_out = n_skip = 0
    while s.need(30):
        sig = s.buf[:4]
        if sig != b"PK\x03\x04":  # 중앙 목차(PK\x01\x02) 등 → 끝
            break
        hdr = s.take(30)
        _, flag, meth, _, _, _, cs, us, nl, el = struct.unpack("<HHHHHIIIHH", hdr[4:30])
        if not s.need(nl + el):
            break
        name = s.take(nl).decode("utf-8", "replace")
        extra = s.take(el)
        if cs == 0xFFFFFFFF and el >= 20:  # ZIP64
            us, cs = struct.unpack("<QQ", extra[4:20])
        if not s.need(cs):
            print(f"  받은 데이터 끝 (마지막 파일 {Path(name).name} 일부만 있음)")
            break
        data = s.take(cs)
        if name.endswith("/"):
            continue
        n_seen += 1
        base = Path(name).name
        if wanted is not None and base not in wanted:
            continue
        dst = out / base
        if dst.exists():
            n_skip += 1
        else:
            raw = zlib.decompressobj(-15).decompress(data) if meth == 8 else data
            dst.write_bytes(raw)
            n_out += 1
        if n_seen % 5000 == 0:
            print(f"  … 압축 안 이미지 {n_seen:,}개 지남 · 꺼냄 {n_out:,}")
        if max_n and n_out + n_skip >= max_n:
            break
    print(f"꺼냄 {n_out:,}장 (이미 있음 {n_skip:,}) · 압축 안에서 본 파일 {n_seen:,}개 → {out}")
    return n_out


# ------------------------------------------------------------------ COCO 변환
def convert(labels, images_dir, out, drone_only=True):
    images_dir = Path(images_dir)
    have = {p.name for p in images_dir.glob("*.jpg")}
    cats = {}
    imgs, anns = [], []
    for name, j in sorted(labels.items()):
        if name not in have:
            continue
        if drone_only and gsd_cm(j) is None:
            continue
        iid = len(imgs)
        imgs.append({"id": iid, "file_name": name, "width": j.get("imageWidth"), "height": j.get("imageHeight"),
                     "altitude": j.get("altitude"), "device": j.get("device"), "gtype": j.get("gtype"),
                     "gsd_cm": gsd_cm(j)})
        for s in j.get("shapes", []):
            pts = np.asarray(s.get("points") or [], float)
            if len(pts) < 2:
                continue
            lab = s.get("label") or "unknown"
            cats.setdefault(lab, len(cats) + 1)
            if s.get("shape_type") == "rectangle" or len(pts) == 2:
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                seg = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            else:
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                seg = [pts.ravel().round(1).tolist()]
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cats[lab], "segmentation": seg,
                         "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                         "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
                         "attributes": {"origin": s.get("origin")}})
    coco = {"images": imgs, "annotations": anns, "categories": [{"id": i, "name": n} for n, i in cats.items()]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    cc = Counter(a["category_id"] for a in anns)
    inv = {i: n for n, i in cats.items()}
    print(f"COCO: 이미지 {len(imgs):,} · 객체 {len(anns):,} → {out}")
    print("  클래스:", {inv[k]: v for k, v in cc.most_common()})
    g = [im["gsd_cm"] for im in imgs if im["gsd_cm"]]
    if g:
        print(f"  해상도(cm/px): 중앙 {np.median(g):.2f} · 범위 {min(g):.2f}~{max(g):.2f}")
    return out


def pack(coco, images_dir, out, gsd_range=(1.0, 4.0), canvas=1024, val_frac=0.1, max_images=None,
         min_obj_px=6, seed=0):
    """'거리별 학습' 데이터셋: 사진마다 해상도(cm/px)를 gsd_range 안의 임의 값으로 맞춰 축소하고,
    축소한 사진 여러 장을 canvas×canvas 판에 붙인다 (선반 배치).

    왜 판에 붙이나: YOLO는 입력을 imgsz로 키우므로 작게 줄인 사진을 그냥 넣으면 다시 커진다.
    판에 모아 붙이면 줄인 크기(= 높은 고도에서 본 크기)가 그대로 학습된다.
    고도 정보가 없는 사진은 쓰지 않는다. 축소 후 min_obj_px보다 작아진 물체는 라벨에서 뺀다.
    """
    import cv2

    from .seg import _imread, _imwrite

    rng = np.random.default_rng(seed)
    d = json.loads(Path(coco).read_text(encoding="utf-8"))
    names = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    cid = {c["id"]: i for i, c in enumerate(sorted(d["categories"], key=lambda c: c["id"]))}
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    ims = [im for im in d["images"] if im.get("gsd_cm") and im["id"] in by]
    rng.shuffle(ims)
    if max_images:
        ims = ims[:max_images]
    n_val = int(len(ims) * val_frac)
    parts = {"val": ims[:n_val], "train": ims[n_val:]}
    out = Path(out)
    stats = {}
    for part, lst in parts.items():
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        board = np.full((canvas, canvas, 3), 114, np.uint8)
        labels, x, y, row_h, k, n_obj, n_drop = [], 0, 0, 0, 0, 0, 0

        def flush():
            nonlocal board, labels, x, y, row_h, k
            if labels or x or y:
                _imwrite(out / "images" / part / f"{part}_{k:05d}.jpg", board, 90)
                (out / "labels" / part / f"{part}_{k:05d}.txt").write_text("\n".join(labels))
                k += 1
            board = np.full((canvas, canvas, 3), 114, np.uint8)
            labels, x, y, row_h = [], 0, 0, 0

        for im in lst:
            img = _imread(Path(images_dir) / im["file_name"])
            if img is None:
                continue
            target = rng.uniform(*gsd_range)
            f = im["gsd_cm"] / target  # <1이면 축소 (높은 고도처럼)
            f = min(f, 1.0)  # 원본보다 키우지는 않는다 (흐려진 가짜 디테일 방지)
            sm = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
            h, w = sm.shape[:2]
            if w > canvas or h > canvas:  # 판보다 크면 판 크기로 잘라 쓴다
                sm = sm[:canvas, :canvas]
                h, w = sm.shape[:2]
            if x + w > canvas:  # 다음 줄
                x, y, row_h = 0, y + row_h, 0
            if y + h > canvas:  # 판이 찼음
                flush()
            board[y:y + h, x:x + w] = sm
            for a in by[im["id"]]:
                bx, by_, bw, bh = (np.array(a["bbox"]) * f)
                bx0, by0 = max(bx, 0), max(by_, 0)
                bx1, by1 = min(bx + bw, w), min(by_ + bh, h)
                if bx1 - bx0 < min_obj_px or by1 - by0 < min_obj_px:
                    n_drop += 1
                    continue
                cx, cy = (x + (bx0 + bx1) / 2) / canvas, (y + (by0 + by1) / 2) / canvas
                labels.append(f"{cid[a['category_id']]} {cx:.6f} {cy:.6f} {(bx1 - bx0) / canvas:.6f} {(by1 - by0) / canvas:.6f}")
                n_obj += 1
            x += w
            row_h = max(row_h, h)
        flush()
        stats[part] = {"images": len(lst), "boards": k, "objects": n_obj, "dropped_tiny": n_drop}
    yaml = (f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"gsd_range": gsd_range, "canvas": canvas, "names": names,
                                               "stats": stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"거리별 데이터셋 → {out}  {stats}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.aihub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--src", required=True, help="AI Hub tar (.crdownload 받는 중이어도 됨)")
    p.add_argument("--labels", required=True, help="TL_*.zip.part0 (라벨 zip) — 라벨 있는 이미지만 꺼냄")
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, help="최대 장수")
    p.add_argument("--all_devices", action="store_true", help="폰·CCTV 사진도 포함")
    p = sub.add_parser("pack", help="거리별(해상도별) 학습 데이터셋 만들기")
    p.add_argument("--coco", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gsd", default="1.0,4.0", help="맞출 해상도 범위 cm/px (문갑도 정사영상 3.0)")
    p.add_argument("--max", type=int, help="쓸 최대 사진 수")
    p = sub.add_parser("convert")
    p.add_argument("--labels", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--all_devices", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        pack(a.coco, a.images, a.out, tuple(map(float, a.gsd.split(","))), max_images=a.max)
        return
    labels = load_labels(a.labels)
    if a.cmd == "extract":
        sel = select(labels, drone_only=not a.all_devices, max_n=a.max)
        print(f"꺼낼 후보 {len(sel):,}장 (드론{'' if not a.all_devices else '+기타'}, 고도 높은 사진 우선)")
        extract(a.src, a.out, wanted=set(sel), max_n=a.max)
    else:
        convert(labels, a.images, a.out, drone_only=not a.all_devices)


if __name__ == "__main__":
    main()
