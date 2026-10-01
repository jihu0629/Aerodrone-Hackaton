"""
하와이 해안쓰레기 공개 데이터(Zenodo 8381113, CC-BY 4.0)에 우리 파이프라인을 그대로 적용하는 데모.
  탐지(YOLO-World / AI Hub 모델) → 칩 지오참조로 위경도 → map.html/map.png/objects.csv
  → 2D 면적 × 재질별 두께범위 → 부피·무게 구간 → plan.py 수거계획 → 작업카드

데이터: data/external/hawaii_debris/imagery_and_labels/
  processed_image_chips/*.jpg (640×640, 2 cm/px) + *.jpg.aux.xml (GDAL PAM: SRS + GeoTransform, NAD83 UTM 4N/5N)
  training_data.csv / evaluation_data.csv : filename,xmin,ymin,xmax,ymax,class,label (픽셀 박스)

  python -m litter.hawaii geo                                   # chips.csv(섬·위경도) + COCO 라벨
  python -m litter.hawaii predict --model world --split eval    # 평가 칩 탐지 (world|aihub)
  python -m litter.hawaii eval    --pred runs/hawaii/pred_world_eval.json
  python -m litter.hawaii demo    --island molokai --model world   # 지도 + 수거계획 + 작업카드
  python -m litter.hawaii figs    --island molokai                 # 발표 그림 → docs/figures/17_, 18_
"""
import argparse
import base64
import csv
import json
import math
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import plan, report
from .config import load_config, map_class
from .geometry import Geo
from .seg import _imread, _ios, evaluate, print_eval

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/hawaii_debris/imagery_and_labels"
CHIPS = DATA / "processed_image_chips"
OUT = ROOT / "runs/hawaii"

# 섬 판별용 위경도 상자 (lat0, lat1, lon0, lon1)
ISLANDS = {
    "niihau": (21.70, 22.05, -160.35, -159.95), "kauai": (21.80, 22.30, -159.85, -159.25),
    "oahu": (21.20, 21.75, -158.35, -157.60), "molokai": (21.00, 21.30, -157.40, -156.65),
    "lanai": (20.70, 20.95, -157.10, -156.80), "maui": (20.55, 21.05, -156.75, -155.95),
    "kahoolawe": (20.48, 20.62, -156.75, -156.50), "hawaii": (18.85, 20.35, -156.15, -154.75),
}
# 라벨 8클래스 → 우리 재질표(config.CLASSES)
LABEL_TO_CLS = {"buoy": "plastic_buoy", "unidentified object": "plastic_other", "net cloth": "net",
                "line fragment": "rope", "metal": "metal", "tire": "tire", "processed wood": "wood", "vessel": "other"}
WORLD_CLASSES = ["buoy", "fishing net", "rope", "tire", "wood debris", "plastic debris"]
MODELS = {"world": ROOT / "yolov8s-worldv2.pt", "aihub": ROOT / "runs/seg/aihub_gsd_det_s/weights/best.pt",
          "hawaii_ft": ROOT / "runs/seg/hawaii_ft/weights/best.pt"}  # 하와이 학습분할로 미세조정 (ft 명령)
CLS_COLOR = {"plastic_buoy": "#e63946", "eps_buoy": "#e63946", "net": "#f4a261", "rope": "#e9c46a", "tire": "#6d597a",
             "wood": "#8d6e63", "metal": "#577590", "plastic_other": "#2a9d8f", "eps_fragment": "#2a9d8f",
             "pet_bottle": "#2a9d8f", "eps_box": "#e63946", "glass": "#577590", "other": "#999999"}
CLS_KO = {"plastic_buoy": "부표", "eps_buoy": "스티로폼 부표", "eps_box": "스티로폼 박스", "eps_fragment": "스티로폼 조각",
          "net": "어망", "rope": "로프", "tire": "타이어", "wood": "목재", "metal": "금속", "plastic_other": "플라스틱",
          "pet_bottle": "페트병", "glass": "유리", "other": "기타"}


def _font():
    import matplotlib
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 1. 지오참조
def read_aux(jpg):
    """칩 .aux.xml → (geotransform 6개, EPSG). GDAL 식: X = gt0 + col*gt1 + row*gt2, Y = gt3 + col*gt4 + row*gt5"""
    t = ET.parse(str(jpg) + ".aux.xml").getroot()
    gt = [float(v) for v in t.find("GeoTransform").text.split(",")]
    srs = t.find("SRS").text
    epsg = int(srs.rstrip("]").rsplit('"', 2)[-2])  # 마지막 AUTHORITY["EPSG","26904"]
    return gt, epsg


class ChipGeo:
    """칩 하나의 픽셀 ↔ 위경도."""
    _tf = {}

    def __init__(self, jpg):
        from pyproj import Transformer
        self.gt, self.epsg = read_aux(jpg)
        if self.epsg not in self._tf:
            self._tf[self.epsg] = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        self.tf = self._tf[self.epsg]
        self.px_m = abs(self.gt[1])  # m/px

    def to_utm(self, col, row):
        g = self.gt
        return g[0] + col * g[1] + row * g[2], g[3] + col * g[4] + row * g[5]

    def to_latlon(self, col, row):
        x, y = self.to_utm(np.asarray(col, float), np.asarray(row, float))
        lon, lat = self.tf.transform(x, y)
        return lat, lon


def island_of(lat, lon):
    for k, (a, b, c, d) in ISLANDS.items():
        if a <= lat <= b and c <= lon <= d:
            return k
    return "unknown"


def _read_labels():
    rows = []
    for name, split in (("training_data.csv", "train"), ("evaluation_data.csv", "eval")):
        with open(DATA / name, newline="") as f:
            for r in csv.DictReader(f):
                r["split"] = split
                rows.append(r)
    return rows


def _coco(images, rows, path):
    cats = {}
    anns = []
    iid = {im["file_name"]: im["id"] for im in images}
    for r in rows:
        if r["filename"] not in iid:
            continue
        cats.setdefault(r["label"], len(cats) + 1)
        x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        anns.append({"id": len(anns) + 1, "image_id": iid[r["filename"]], "category_id": cats[r["label"]],
                     "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})
    path.write_text(json.dumps({"images": images, "annotations": anns,
                                "categories": [{"id": i, "name": n} for n, i in cats.items()]}, ensure_ascii=False),
                    encoding="utf-8")
    return len(anns)


def cmd_geo(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _read_labels()
    n_lab = Counter(r["filename"] for r in rows)
    split_of = {r["filename"]: r["split"] for r in rows}
    chips = sorted(CHIPS.glob("*.jpg"))
    images, recs = [], []
    for i, p in enumerate(chips):
        g = ChipGeo(p)
        lat, lon = g.to_latlon(320, 320)
        isl = island_of(float(lat), float(lon))
        recs.append({"filename": p.name, "island": isl, "lat": f"{float(lat):.7f}", "lon": f"{float(lon):.7f}",
                     "epsg": g.epsg, "px_m": g.px_m, "split": split_of.get(p.name, "none"), "n_labels": n_lab.get(p.name, 0)})
        images.append({"id": i, "file_name": p.name, "width": 640, "height": 640, "geo_transform": g.gt,
                       "crs": f"EPSG:{g.epsg}", "island": isl, "split": recs[-1]["split"]})
    with open(OUT / "chips.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    n_all = _coco(images, rows, OUT / "labels_all.json")
    n_ev = _coco([im for im in images if im["split"] == "eval"], [r for r in rows if r["split"] == "eval"], OUT / "labels_eval.json")
    print(f"칩 {len(chips)}장 · 라벨 전체 {n_all} (평가 {n_ev}) → {OUT}/chips.csv, labels_all.json, labels_eval.json")
    print("섬별 칩 수 / 라벨 수 (전체 → 평가):")
    for isl in ISLANDS:
        c = [r for r in recs if r["island"] == isl]
        if c:
            print(f"  {isl:<10} 칩 {len(c):4d} ({sum(1 for r in c if r['split']=='eval'):3d}) · 라벨 {sum(r['n_labels'] for r in c):5d}"
                  f" ({sum(r['n_labels'] for r in c if r['split']=='eval'):4d})")
    unk = [r for r in recs if r["island"] == "unknown"]
    if unk:
        print(f"  unknown {len(unk)}장 (예: {unk[0]['filename']} {unk[0]['lat']},{unk[0]['lon']})")


def load_chips():
    with open(OUT / "chips.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ 2. 탐지
def wait_gpu(poll=30):
    while True:
        try:
            r = (subprocess.run(["tasklist"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout or "").lower()
        except (OSError, ValueError):
            r = ""
        if "colmap" not in r:
            return
        print("  colmap 실행 중 → 30초 대기")
        time.sleep(poll)


def _select(a):
    chips = load_chips()
    if a.split == "eval":
        sel = [r for r in chips if r["split"] == "eval"]
    elif a.split == "all":
        sel = chips
    else:  # island:<name>
        isl = a.split.split(":", 1)[1]
        sel = [r for r in chips if r["island"] == isl]
    return [r["filename"] for r in sel]


def predict(model_key, files, out, conf=0.05, batch=8, device=0, wait=True, scale=1.0):
    """칩(640)은 타일 없이 그대로. seg.predict와 같은 COCO 형식으로 저장 (eval·지도에 공용)."""
    from ultralytics import YOLO

    if wait:
        wait_gpu()
    m = YOLO(str(MODELS[model_key]))
    if model_key == "world":
        m.set_classes(WORLD_CLASSES)
    names = m.names
    G = json.loads((OUT / "labels_all.json").read_text(encoding="utf-8"))
    by_name = {im["file_name"]: im for im in G["images"]}
    images = [by_name[f] for f in files]
    cats = {k: k + 1 for k in names}
    anns = []
    dev = device
    t0 = time.time()
    for s in range(0, len(images), batch):
        chunk = images[s:s + batch]
        imgs = [_imread(CHIPS / im["file_name"]) for im in chunk]
        if scale != 1.0:  # 작은 물체(중앙값 30 px) → 확대해서 추론, 좌표는 되돌림
            imgs = [cv2.resize(im_, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) for im_ in imgs]
        sz = int(640 * scale)
        try:
            res = m.predict(imgs, conf=conf, imgsz=sz, verbose=False, device=dev)
        except RuntimeError as e:  # CUDA OOM → CPU
            if "out of memory" not in str(e).lower() or dev == "cpu":
                raise
            print("  GPU 메모리 부족 → CPU로 전환")
            import torch
            torch.cuda.empty_cache()
            dev = "cpu"
            res = m.predict(imgs, conf=conf, imgsz=sz, verbose=False, device=dev)
        for im, r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            dets = [(b / scale, int(c), float(sc)) for b, c, sc in
                    zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            dets.sort(key=lambda t: -t[2])
            kept = []
            for b, c, sc in dets:  # 클래스 달라도 같은 자리면 하나로 (개방형 모델은 중복이 많음)
                if all(_ios(b, k[0]) < 0.6 for k in kept):
                    kept.append((b, c, sc))
            for b, c, sc in kept:
                x0, y0, x1, y1 = (float(v) for v in b)
                anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": cats[c],
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0), "score": sc, "iscrowd": 0})
        if (s // batch) % 20 == 0:
            print(f"  {s + len(chunk)}/{len(images)} · {time.time() - t0:.0f}s · 탐지 {len(anns)}")
    d = {"images": images, "annotations": anns, "categories": [{"id": cats[k], "name": n} for k, n in names.items()],
         "model": model_key, "device": str(dev), "conf": conf, "scale": scale}
    Path(out).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"예측 {len(anns)}개 · 칩 {len(images)}장 · {time.time() - t0:.0f}s ({dev}) → {out}")
    return out


def cmd_predict(a):
    files = _select(a)
    out = a.out or OUT / f"pred_{a.model}_{a.split.replace(':', '_')}.json"
    predict(a.model, files, out, conf=a.conf, batch=a.batch, device=a.device, wait=not a.no_wait, scale=a.scale)


def cmd_eval(a):
    P = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    files = [im["file_name"] for im in P["images"]]
    rows = []
    for conf in a.conf:
        r = evaluate(OUT / "labels_all.json", a.pred, iou_thr=a.iou, conf=conf, class_agnostic=True, files=files)
        print(f"\n[{Path(a.pred).name} · conf≥{conf} · IoU≥{a.iou}]")
        print_eval(r)
        rows.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


# ------------------------------------------------------------------ 3. 위경도 → 지도
def _thumb(img, b, size=140, pad=1.0):
    x0, y0, x1, y1 = map(int, b)
    cx, cy, h = (x0 + x1) // 2, (y0 + y1) // 2, int(max(x1 - x0, y1 - y0) * (0.5 + pad)) + 8
    crop = img[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h].copy()
    if crop.size == 0:
        return None
    ox, oy = max(cx - h, 0), max(cy - h, 0)
    cv2.rectangle(crop, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (0, 0, 255), 1)
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode() if ok else None


def detections_to_objects(pred_json, island, conf, thumbs=True, max_thumbs=600, material_pred=None):
    """예측 COCO → 섬 하나의 물체 목록 (위경도·면적 m²·재질).
    material_pred: 단일 클래스(litter) 모델일 때 재질을 빌려올 다른 예측 COCO (겹치는 AI Hub 박스의 클래스, IoU≥0.3)"""
    from .seg import _iou
    P = json.loads(Path(pred_json).read_text(encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in P["categories"]}
    mat = {}
    if material_pred:
        M = json.loads(Path(material_pred).read_text(encoding="utf-8"))
        mcat = {c["id"]: c["name"] for c in M["categories"]}
        mname = {im["id"]: im["file_name"] for im in M["images"]}
        for an in M["annotations"]:
            x, y, w, h = an["bbox"]
            mat.setdefault(mname[an["image_id"]], []).append(((x, y, x + w, y + h), mcat[an["category_id"]], an["score"]))
    ims = {im["id"]: im for im in P["images"] if im.get("island") == island}
    by_im = {}
    for an in P["annotations"]:
        if an["image_id"] in ims and an["score"] >= conf:
            by_im.setdefault(an["image_id"], []).append(an)
    objs = []
    cfg = load_config()
    for iid, anns in by_im.items():
        im = ims[iid]
        g = ChipGeo(CHIPS / im["file_name"])
        img = _imread(CHIPS / im["file_name"]) if thumbs else None
        for an in sorted(anns, key=lambda t: -t["score"]):
            x, y, w, h = an["bbox"]
            lat, lon = g.to_latlon(x + w / 2, y + h / 2)
            corners = [g.to_latlon(cx, cy) for cx, cy in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
            raw = cat[an["category_id"]]
            if raw in ("litter", "item") and mat:  # 재질: 겹치는 AI Hub 박스 중 점수 높은 것, 없으면 플라스틱
                cand = [(sc, nm) for bb, nm, sc in mat.get(im["file_name"], []) if _iou((x, y, x + w, y + h), bb) >= 0.3]
                raw = max(cand)[1] if cand else "plastic debris"
            o = {"item_id": f"{Path(im['file_name']).stem.split('_', 1)[1]}_{an['id']}", "image": im["file_name"],
                 "cls_raw": raw, "cls": LABEL_TO_CLS.get(raw) or map_class(raw, cfg), "score": float(an["score"]),
                 "lat": float(lat), "lon": float(lon), "corners": [[float(a), float(b)] for a, b in corners],
                 "polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                 "length_m": float(max(w, h) * g.px_m), "width_m": float(min(w, h) * g.px_m),
                 "area_m2": float(w * h * g.px_m * g.px_m), "px_m": g.px_m,
                 "thumb_b64": _thumb(img, (x, y, x + w, y + h)) if img is not None and len(objs) < max_thumbs else None}
            objs.append(o)
    return objs


def write_objects(out, objs):
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "image", "cls_raw", "cls", "score", "lat", "lon", "length_m", "width_m", "area_m2",
                    "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "crew", "stop_id"])
        for o in objs:
            w.writerow([o["item_id"], o["image"], o["cls_raw"], o["cls"], f"{o['score']:.3f}", f"{o['lat']:.7f}", f"{o['lon']:.7f}",
                        f"{o['length_m']:.2f}", f"{o['width_m']:.2f}", f"{o['area_m2']:.3f}",
                        f"{o.get('weight_est_kg', 0):.2f}", f"{o.get('weight_lo_kg', 0):.2f}", f"{o.get('weight_hi_kg', 0):.2f}",
                        o.get("crew", ""), o.get("stop_id", "")])


def map_html(path, objs, stops, island, model, summary):
    data = {"objs": [{k: o.get(k) for k in ("item_id", "cls", "cls_raw", "score", "lat", "lon", "corners", "area_m2",
                                             "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "crew", "stop_id", "thumb_b64")}
                     for o in objs],
            "stops": [{k: s.get(k) for k in ("stop_id", "visit_order", "lat", "lon", "n_items", "kg_est", "kg_hi", "crew")} for s in stops],
            "depot": summary.get("depot_latlon"), "colors": CLS_COLOR, "ko": CLS_KO}
    legend = "".join(f'<span style="color:{CLS_COLOR[c]}">■</span> {CLS_KO[c]} ' for c in sorted({o["cls"] for o in objs}))
    note = (f"하와이 {island} · 2015 항공 정사영상 2 cm/px (Zenodo 8381113, CC-BY 4.0) · 모델 {model}<br>"
            f"물체 {summary['n_items']}개 · 면적 합 {summary['area_m2_sum']:.1f} m² · 무게 {summary['kg_lo']:.0f}~{summary['kg_hi']:.0f} kg (추정 {summary['kg_est']:.0f})"
            f" · 정거장 {summary['n_stops']} · 마대 {summary['bags']}개 · 경로 {summary['route_m'] / 1000:.1f} km<br>{legend}"
            "<br><span style='color:#264653'>━</span> 수거 경로(정거장 순서) · 점 크기 = 추정 무게")
    html = """<!doctype html><meta charset="utf-8"><title>하와이 해안쓰레기 지도</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}#n{position:absolute;z-index:999;left:50px;top:8px;background:#fffd;padding:6px 10px;border-radius:6px;font:13px 'Malgun Gothic',sans-serif;max-width:560px}
.leaflet-popup-content{font:12px 'Malgun Gothic',sans-serif}</style>
<div id="m"></div><div id="n">__NOTE__</div>
<script>
const D=__DATA__;
const m=L.map('m',{maxZoom:23});
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:23,maxNativeZoom:19,attribution:'Esri'}).addTo(m);
const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:23,maxNativeZoom:19,attribution:'OSM'});
const gObj=L.layerGroup().addTo(m), gStop=L.layerGroup().addTo(m), gRoute=L.layerGroup().addTo(m);
L.control.layers({'위성':sat,'지도':osm},{'물체':gObj,'정거장':gStop,'경로':gRoute}).addTo(m);
D.objs.forEach(o=>{const c=D.colors[o.cls]||'#999';
 const r=Math.max(4,Math.min(16,3+2*Math.sqrt(o.weight_est_kg||0)));
 const mk=L.circleMarker([o.lat,o.lon],{radius:r,color:c,fillColor:c,fillOpacity:.75,weight:1});
 mk.bindPopup(`<b>${D.ko[o.cls]||o.cls}</b> <small>(${o.cls_raw}, 신뢰도 ${o.score.toFixed(2)})</small><br>`+
  `면적 ${o.area_m2.toFixed(2)} m² · 무게 ${o.weight_lo_kg.toFixed(1)}~${o.weight_hi_kg.toFixed(1)} kg (추정 ${o.weight_est_kg.toFixed(1)}) · ${o.crew}<br>`+
  `${o.lat.toFixed(6)}, ${o.lon.toFixed(6)} · ${o.stop_id}<br>`+(o.thumb_b64?`<img src="data:image/jpeg;base64,${o.thumb_b64}">`:''));
 if(o.corners) L.polygon(o.corners,{color:c,weight:1,fill:false}).addTo(gObj);
 mk.addTo(gObj);});
D.stops.forEach(s=>{L.marker([s.lat,s.lon],{icon:L.divIcon({className:'',html:`<div style="background:#264653;color:#fff;border-radius:10px;padding:1px 5px;font:11px sans-serif;white-space:nowrap">${s.visit_order}</div>`})})
 .bindPopup(`<b>${s.stop_id}</b> (${s.visit_order}번째)<br>${s.n_items}개 · ${s.kg_est.toFixed(1)} kg (최대 ${s.kg_hi.toFixed(1)}) · ${s.crew}`).addTo(gStop);});
const rt=D.stops.slice().sort((a,b)=>a.visit_order-b.visit_order).map(s=>[s.lat,s.lon]);
if(D.depot){rt.unshift(D.depot);rt.push(D.depot);L.marker(D.depot).bindPopup('집하장(가정)').addTo(gRoute);}
L.polyline(rt,{color:'#264653',weight:2,opacity:.7,dashArray:'4 4'}).addTo(gRoute);
m.fitBounds(L.latLngBounds(D.objs.map(o=>[o.lat,o.lon])).pad(0.1));
</script>"""
    path.write_text(html.replace("__DATA__", json.dumps(data)).replace("__NOTE__", note), encoding="utf-8")


def map_png(path, objs, stops, island, summary, geo, hot_m=120):
    """(왼쪽) 섬 전체 물체 분포 + 경로, (오른쪽) 가장 밀집한 hot_m×hot_m 구간을 칩 영상 위에 박스로."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    _font()
    xy = np.array([[o["x"], o["y"]] for o in objs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
    for c in sorted({o["cls"] for o in objs}):
        sel = np.array([o["cls"] == c for o in objs])
        a1.scatter(xy[sel, 0] / 1000, xy[sel, 1] / 1000, s=8 + 3 * np.sqrt([o["weight_est_kg"] for o in objs if o["cls"] == c]),
                   c=CLS_COLOR[c], label=f"{CLS_KO[c]} ({sel.sum()})", alpha=.8, edgecolors="k", linewidths=.2)
    rt = np.array([[s["x"], s["y"]] for s in sorted(stops, key=lambda s: s["visit_order"])])
    d = np.array(summary["depot"])
    rt = np.vstack([d, rt, d])
    a1.plot(rt[:, 0] / 1000, rt[:, 1] / 1000, "-", color="#264653", lw=.6, alpha=.5, label="수거 경로")
    a1.set_aspect("equal")
    a1.set_xlabel("동쪽 (km, UTM)")
    a1.set_ylabel("북쪽 (km, UTM)")
    a1.legend(fontsize=8, loc="best")
    a1.set_title(f"하와이 {island} · 탐지 물체 {summary['n_items']}개 · 추정 {summary['kg_lo']:.0f}~{summary['kg_hi']:.0f} kg\n"
                 f"정거장 {summary['n_stops']} · 인력 1인 {summary['crew_counts']['1인']}/2인 {summary['crew_counts']['2인']}/장비 {summary['crew_counts']['장비']}"
                 f" · 마대 {summary['bags']}개 · 경로 {summary['route_m'] / 1000:.1f} km", fontsize=10)
    # 핫스팟
    best, bc = 0, xy[0]
    for p in xy[:: max(1, len(xy) // 400)]:
        n = int(((np.abs(xy - p) < hot_m / 2).all(1)).sum())
        if n > best:
            best, bc = n, p
    x0, y0 = bc - hot_m / 2
    shown = set()
    for o in objs:
        if abs(o["x"] - bc[0]) < hot_m / 2 + 7 and abs(o["y"] - bc[1]) < hot_m / 2 + 7 and o["image"] not in shown:
            shown.add(o["image"])
            img = _imread(CHIPS / o["image"])
            g = ChipGeo(CHIPS / o["image"])
            (ex0, ey0), (ex1, ey1) = g.to_utm(0, 640), g.to_utm(640, 0)
            lat0, lon0 = g.to_latlon(0, 320)
            lat1, lon1 = g.to_latlon(640, 320)
            # 칩 UTM(NAD83) → 우리 Geo(WGS84 UTM): 수 cm 차이라 위경도 경유로 모서리만 변환
            c0 = geo.to_xy(*g.to_latlon(0, 640))
            c1 = geo.to_xy(*g.to_latlon(640, 0))
            a2.imshow(img[:, :, ::-1], extent=(c0[0] - x0, c1[0] - x0, c0[1] - y0, c1[1] - y0), zorder=1)
    for o in objs:
        if abs(o["x"] - bc[0]) < hot_m / 2 + 7 and abs(o["y"] - bc[1]) < hot_m / 2 + 7:
            pts = np.array([geo.to_xy(la, lo) for la, lo in o["corners"]]) - [x0, y0]
            a2.add_patch(MPoly(pts, fill=False, ec=CLS_COLOR[o["cls"]], lw=1.2, zorder=3))
    a2.set_xlim(0, hot_m)
    a2.set_ylim(0, hot_m)
    a2.set_aspect("equal")
    a2.set_facecolor("#dfe8ef")
    a2.set_xlabel("동쪽 (m)")
    a2.set_ylabel("북쪽 (m)")
    la, lo = geo.to_latlon(*bc)
    a2.set_title(f"가장 밀집한 {hot_m}×{hot_m} m 구간 ({la:.4f}, {lo:.4f}) · 물체 {best}개 · 칩 {len(shown)}장\n"
                 "박스 = 탐지, 색 = 재질 (2 cm/px 정사영상 위)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"lat": float(la), "lon": float(lo), "n": best}


# ------------------------------------------------------------------ 4. 2D 무게·수거계획
def add_weights_2d(objs, cfg):
    """높이 정보 없음 → 면적 × 재질별 두께 [최소, 유효, 최대] × 겉보기밀도 = 무게 [하한, 추정, 상한].
    (fromvideo.REALISTIC/config.thick_range_m 와 같은 사고방식. 업체 방식 kg/m² 계수와 비교 가능)"""
    from .fromvideo import company_weight
    for o in objs:
        p = cfg["classes"].get(o["cls"], cfg["classes"]["other"])
        t_lo, t_hi = p.get("thick_range_m", [p["thick_m"] / 2, p["thick_m"] * 2])
        t = min(max(p["thick_m"], t_lo), t_hi)
        A = o["area_m2"]
        o.update(volume_lo_m3=A * t_lo, volume_m3=A * t, volume_hi_m3=A * t_hi, h_p90_m=0.0, valid_3d=False,
                 rho_app=p["rho_app"], thick_m=t, thick_range_m=[t_lo, t_hi], compress=p.get("compress", 1.0),
                 weight_lo_kg=A * t_lo * p["rho_app"], weight_est_kg=A * t * p["rho_app"], weight_hi_kg=A * t_hi * p["rho_app"],
                 weight_method="2d:area×thick_range×rho", bulk_m3=A * t / p.get("compress", 1.0),
                 n_views=1, buried_suspect=False, elongation=o["length_m"] / max(o["width_m"], 1e-3))
        if p.get("kg_per_m"):  # 로프: 길이 × kg/m 가 더 크면 그쪽
            o["weight_est_kg"] = max(o["weight_est_kg"], o["length_m"] * p["kg_per_m"])
            o["weight_hi_kg"] = max(o["weight_hi_kg"], o["weight_est_kg"])
        o["company_code"], o["company_kg"] = company_weight(o)
    return objs


def cmd_demo(a):
    cfg = load_config(a.config)
    cfg["plan"]["stop_radius_m"] = a.stop_radius
    out = OUT / (a.out_name or a.island)
    out.mkdir(parents=True, exist_ok=True)
    pred = a.pred or OUT / f"pred_{a.model}_island_{a.island}_x{a.scale:g}.json"
    if not Path(pred).exists():
        print(f"[0] 섬 {a.island} 칩 탐지 ({a.model}, 확대 {a.scale:g}배)")
        files = [r["filename"] for r in load_chips() if r["island"] == a.island]
        predict(a.model, files, pred, conf=0.05, batch=a.batch, device=a.device, wait=not a.no_wait, scale=a.scale)
    print(f"[1] 탐지 → 위경도 (conf≥{a.conf})")
    objs = detections_to_objects(pred, a.island, a.conf, material_pred=a.material_pred, max_thumbs=a.max_thumbs)
    if not objs:
        raise SystemExit("물체 없음")
    geo = Geo(lon=objs[0]["lon"])
    for o in objs:
        o["x"], o["y"] = geo.to_xy(o["lat"], o["lon"])
        o["r95_m"] = a.r95
    print(f"  물체 {len(objs)}개 · " + ", ".join(f"{CLS_KO[k]} {v}" for k, v in Counter(o["cls"] for o in objs).most_common()))
    print("[2] 2D 무게 구간 (면적 × 두께범위 × 밀도)")
    add_weights_2d(objs, cfg)
    print("[3] 수거계획 (plan.make_plan)")
    stops, summary = plan.make_plan(objs, cfg)
    dl = geo.to_latlon(*summary["depot"])
    summary.update({
        "island": a.island, "model": a.model, "scale": a.scale, "conf": a.conf, "source": "Zenodo 8381113 (2015 하와이 항공 정사영상 2 cm/px, CC-BY 4.0)",
        "n_chips": len({o["image"] for o in objs}), "area_m2_sum": float(sum(o["area_m2"] for o in objs)),
        "kg_lo": float(sum(o["weight_lo_kg"] for o in objs)),
        "volume_lo_m3": float(sum(o["volume_lo_m3"] for o in objs)), "volume_est_m3": float(sum(o["volume_m3"] for o in objs)),
        "volume_hi_m3": float(sum(o["volume_hi_m3"] for o in objs)),
        "company_kg": float(sum(o["company_kg"] for o in objs)), "depot_latlon": [float(dl[0]), float(dl[1])],
        "by_class": {CLS_KO[k]: {"n": v, "area_m2": float(sum(o["area_m2"] for o in objs if o["cls"] == k)),
                                 "kg_lo": float(sum(o["weight_lo_kg"] for o in objs if o["cls"] == k)),
                                 "kg_hi": float(sum(o["weight_hi_kg"] for o in objs if o["cls"] == k))}
                     for k, v in Counter(o["cls"] for o in objs).most_common()},
        "weight_method": "면적 × 재질별 두께범위(config thick_range_m) × 겉보기밀도 — 높이 정보 없는 2D 추정, 값은 가정",
    })
    print(f"  정거장 {summary['n_stops']} · 무게 {summary['kg_lo']:.0f}~{summary['kg_hi']:.0f} kg (추정 {summary['kg_est']:.0f}, 업체방식 {summary['company_kg']:.1f})"
          f" · 인력 {summary['crew_counts']} · 마대 {summary['bags']} · 트럭 {summary['trucks']} · 경로 {summary['route_m'] / 1000:.1f} km")
    print("[4] 결과물")
    write_objects(out, objs)
    (out / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[lo, la] for la, lo in o["corners"]] + [[o["corners"][0][1], o["corners"][0][0]]]]},
         "properties": {k: o[k] for k in ("item_id", "cls", "cls_raw", "score", "area_m2", "weight_est_kg", "weight_lo_kg", "weight_hi_kg", "crew", "stop_id")}}
        for o in objs]}, ensure_ascii=False), encoding="utf-8")
    map_html(out / "map.html", objs, stops, a.island, a.model, summary)
    summary["hotspot"] = map_png(out / "map.png", objs, stops, a.island, summary, geo)
    report.write_tables(out, objs, stops, summary)  # items.csv, stops.csv, *.geojson, summary.json
    report.fig_map(out, objs, stops, summary)
    report.work_cards(out, objs, stops, summary, CHIPS, max_stops=a.max_cards)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  → {out}  (map.html, map.png, objects.csv/geojson, work_cards.html, fig_map.png, items.csv, stops.csv, summary.json)")


# ------------------------------------------------------------------ 4-1. 미세조정 (하와이 학습분할 → 단일 'litter' 클래스)
def cmd_ft(a):
    """training_data.csv 칩(1,167장) → YOLO txt(단일 클래스) → AI Hub 가중치에서 이어서 학습. 검증은 평가 420칩."""
    import shutil
    from ultralytics import YOLO
    ds = ROOT / "data/hawaii_yolo"
    rows = _read_labels()
    by_f = {}
    for r in rows:
        by_f.setdefault(r["filename"], []).append(r)
    chips = load_chips()
    parts = {"train": [c["filename"] for c in chips if c["split"] == "train"],
             "val": [c["filename"] for c in chips if c["split"] == "eval"]}
    for part, files in parts.items():
        (ds / "images" / part).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / part).mkdir(parents=True, exist_ok=True)
        for f in files:
            if not (ds / "images" / part / f).exists():
                shutil.copy(CHIPS / f, ds / "images" / part / f)
            lines = []
            for r in by_f.get(f, []):
                x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                lines.append(f"0 {(x0 + x1) / 2 / 640:.5f} {(y0 + y1) / 2 / 640:.5f} {(x1 - x0) / 640:.5f} {(y1 - y0) / 640:.5f}")
            (ds / "labels" / part / (Path(f).stem + ".txt")).write_text("\n".join(lines))
    (ds / "data.yaml").write_text(f"path: {ds.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: litter\n",
                                  encoding="utf-8")
    print(f"데이터셋 train {len(parts['train'])} / val {len(parts['val'])} → {ds}")
    wait_gpu()
    m = YOLO(str(MODELS["aihub"]))
    r = m.train(data=str(ds / "data.yaml"), epochs=a.epochs, imgsz=640, batch=a.batch, device=0, project=str(ROOT / "runs/seg"),
                name="hawaii_ft", exist_ok=True, patience=5, workers=2, single_cls=True,
                flipud=0.5, fliplr=0.5, degrees=90, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, plots=False)
    print(f"best → {Path(r.save_dir) / 'weights' / 'best.pt'}")


# ------------------------------------------------------------------ 5. 발표 그림
def _boxes_of(coco, conf=None):
    d = json.loads(Path(coco).read_text(encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in d["categories"]}
    name = {im["id"]: im["file_name"] for im in d["images"]}
    out = {}
    for an in d["annotations"]:
        if conf is None or an.get("score", 1) >= conf:
            out.setdefault(name[an["image_id"]], []).append((an["bbox"], cat[an["category_id"]], an.get("score")))
    return out


def cmd_figs(a):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import shutil
    _font()
    figs = ROOT / "docs/figures"
    gt = _boxes_of(OUT / "labels_all.json")
    preds = {k: _boxes_of(OUT / f"pred_{k}_eval{a.suffix}.json", a.conf) for k in ("world", "aihub")}
    ft_path = OUT / "pred_hawaii_ft_eval_x1.json"  # 하와이 미세조정 모델(원 해상도, conf 0.3)
    if ft_path.exists():
        preds["ft"] = _boxes_of(ft_path, 0.3)
    chips = [r for r in load_chips() if r["split"] == "eval" and int(r["n_labels"]) >= 2]
    chips.sort(key=lambda r: -(len(preds["world"].get(r["filename"], [])) + len(preds["aihub"].get(r["filename"], []))))
    # 섬 골고루: 라벨 많은 칩 중 섬별 최대 2장
    pick, seen = [], Counter()
    for r in chips:
        if seen[r["island"]] < 2 and len(pick) < a.n:
            pick.append(r)
            seen[r["island"]] += 1
    n = len(pick)
    rows = [("라벨 (정답)", gt, "#2ec27e"), (f"YOLO-World (텍스트 클래스, conf≥{a.conf})", preds["world"], "#e63946"),
            (f"AI Hub 학습 모델 (conf≥{a.conf})", preds["aihub"], "#2a78d6")]
    if "ft" in preds:
        rows.append(("AI Hub → 하와이 10분 미세조정 (conf≥0.3)", preds["ft"], "#f4a261"))
    fig, axes = plt.subplots(len(rows), n, figsize=(2.9 * n, 3.1 * len(rows) + 0.4))
    for i, r in enumerate(pick):
        img = _imread(CHIPS / r["filename"])[:, :, ::-1]
        for j, (title, src, col) in enumerate(rows):
            ax = axes[j, i]
            ax.imshow(img)
            for (x, y, w, h), nm, sc in src.get(r["filename"], []):
                ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=1.3))
                if j > 0:
                    ax.text(x, max(y - 3, 8), nm.replace("_", " ")[:14], color=col, fontsize=5.5, va="bottom",
                            bbox={"fc": "white", "ec": "none", "alpha": .6, "pad": .5})
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(title, fontsize=9, color=col)
            if j == 0:
                ax.set_title(f"{r['island']} · 라벨 {r['n_labels']}", fontsize=8)
    fig.suptitle("하와이 공개 항공영상(2 cm/px) 칩: 라벨 vs 탐지 — 재학습 없음(0.24) → 현지 사진 10분 미세조정(0.58)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = figs / "17_하와이_칩_라벨vs탐지.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"(a) {p}")
    src_dir = OUT / (a.out_name or a.island)
    if (src_dir / "map.png").exists():
        q = figs / f"18_하와이_{a.island}_지도_수거계획.png"
        shutil.copy(src_dir / "map.png", q)
        print(f"(b) {q}")
        shutil.copy(src_dir / "summary.json", figs / f"hawaii_{a.out_name or a.island}_summary.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.hawaii", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("geo")
    p = sub.add_parser("predict")
    p.add_argument("--model", choices=list(MODELS), default="world")
    p.add_argument("--split", default="eval", help="eval | all | island:<이름>")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--no_wait", action="store_true", help="colmap 종료를 기다리지 않음")
    p.add_argument("--scale", type=float, default=1.0, help="추론 전 확대 배율 (2면 1280)")
    p.add_argument("--out")
    p = sub.add_parser("eval")
    p.add_argument("--pred", required=True)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    p.add_argument("--out")
    p = sub.add_parser("demo")
    p.add_argument("--island", required=True)
    p.add_argument("--model", choices=list(MODELS), default="world")
    p.add_argument("--pred", help="예측 COCO (없으면 섬 칩 전부 탐지)")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=2.0, help="추론 확대 배율 (평가에서 2배가 최적)")
    p.add_argument("--stop_radius", type=float, default=20.0, help="정거장 반경 m (항공 조사 규모라 기본 20)")
    p.add_argument("--r95", type=float, default=3.0, help="위치 95%% 반경 m (항공 정사영상 지오참조 오차 가정)")
    p.add_argument("--max_cards", type=int, default=60)
    p.add_argument("--out_name", help="runs/hawaii/<이름> (기본: 섬 이름)")
    p.add_argument("--material_pred", help="단일클래스 모델용: 재질을 빌려올 AI Hub 예측 COCO")
    p.add_argument("--max_thumbs", type=int, default=600, help="지도 팝업 썸네일을 넣을 물체 수 (전부 넣으면 HTML 커짐)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--no_wait", action="store_true")
    p.add_argument("--config")
    p = sub.add_parser("ft")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=16)
    p = sub.add_parser("figs")
    p.add_argument("--island", default="niihau")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--suffix", default="_x2", help="예측 파일 접미사 (pred_<model>_eval<suffix>.json)")
    p.add_argument("--out_name", help="지도 그림을 가져올 runs/hawaii/<이름> (기본: 섬 이름)")
    p.add_argument("--n", type=int, default=6)
    a = ap.parse_args(argv)
    {"geo": cmd_geo, "predict": cmd_predict, "eval": cmd_eval, "demo": cmd_demo, "ft": cmd_ft, "figs": cmd_figs}[a.cmd](a)


if __name__ == "__main__":
    main()
