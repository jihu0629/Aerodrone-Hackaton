"""
드론 영상(MP4) + SRT → 비행 경로 + 탐지 물체 위치를 지도에 표시.

  python -m litter.video_map --video DJI_0007.MP4 --srt DJI_0007.SRT --out runs/map/0007 \
         --weights yolo11s.pt runs/seg/uavvaste_det_s/weights/best.pt

단계:
  1) SRT → 프레임별 위경도·상대고도 → 비행 경로
  2) 일정 간격 프레임에서 탐지 (여러 모델 가능: COCO=사람·차 등 장애물, 쓰레기 모델)
  3) 탐지 박스 중심 픽셀 → 지면 좌표 (드론 위치·고도·카메라 각도로 광선 계산)
  4) 여러 프레임에서 본 같은 물체를 하나로 묶기 (같은 종류 + 가까운 위치)
  5) 지도 HTML (위성사진 배경) + CSV/GeoJSON

한계: Mini 계열 SRT엔 기체 방향(yaw)·짐벌 각도가 없다.
  - 짐벌은 --pitch (기본 -90, 수직 아래)
  - 방향은 이동 중이면 GPS 이동 방향으로 추정, 정지(호버링) 중이면 --yaw 값 (기본 0 = 북쪽)
  → 방향이 틀리면 물체 위치가 드론 위치를 중심으로 회전한다 (고도 7 m면 최대 약 5 m).
"""
import argparse
import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics, Pose, pixel_to_ground
from .geometry import Geo
from .telemetry import read_srt

OBSTACLE = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "boat", "bench", "chair", "umbrella",
            "dog", "potted plant", "fire hydrant", "stop sign", "traffic light", "bird"}


def _thumb(frame, b, size=160):
    x0, y0, x1, y1 = map(int, b)
    pad = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    crop = frame[max(cy - pad, 0):cy + pad, max(cx - pad, 0):cx + pad].copy()
    if crop.size == 0:
        return None
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode() if ok else None


def run(video, srt, out, weights, every_s=0.5, conf=0.35, pitch=-90.0, yaw=None, f35=24.0,
        min_alt=2.0, merge_m=1.5, tiles=2):
    from ultralytics import YOLO

    from .seg import _ios

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tel = [r for r in read_srt(srt) if "lat" in r and "lon" in r]
    if not tel:
        raise SystemExit("SRT에 위경도가 없음")
    geo = Geo(None, lon=tel[0]["lon"])
    xy = np.array([geo.to_xy(r["lat"], r["lon"]) for r in tel])
    by_frame = {r["frame"]: k for k, r in enumerate(tel)}
    models = [YOLO(str(w)) for w in weights]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    K = Intrinsics.from_f35(W, H, f35)
    step = max(1, int(round(fps * every_s)))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dets = []
    yaw_src = set()
    for fi in range(0, n_frames, step):
        k = by_frame.get(fi + 1)
        if k is None or tel[k].get("alt", 0) < min_alt:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        r = tel[k]
        # 방향: 이동 방향(최근 2초 이동 > 1.5 m) 또는 지정값
        k0, k1 = max(k - int(fps), 0), min(k + int(fps), len(tel) - 1)
        mv = xy[k1] - xy[k0]
        if yaw is None and np.hypot(*mv) > 1.5:
            yw = float(np.degrees(np.arctan2(mv[0], mv[1]))) % 360
            yaw_src.add("이동방향")
        else:
            yw = yaw if yaw is not None else 0.0
            yaw_src.add("지정값" if yaw is not None else "기본 0°(북)")
        pose = Pose(xy[k, 0], xy[k, 1], r["alt"], yw, pitch)
        # 타일 나눠 탐지 (4K에서 작은 물체)
        tw, th = W // tiles, H // tiles
        tl = [(x, y) for y in range(0, H, th) for x in range(0, W, tw)]
        found = []
        for m in models:
            res = m.predict([frame[y:y + th, x:x + tw] for x, y in tl], conf=conf, imgsz=1024, verbose=False)
            for (x, y), rr in zip(tl, res):
                for b, c, s in zip(rr.boxes.xyxy.cpu().numpy(), rr.boxes.cls.cpu().numpy(), rr.boxes.conf.cpu().numpy()):
                    found.append((b + [x, y, x, y], m.names[int(c)], float(s)))
        found.sort(key=lambda d: -d[2])
        kept = []
        for d in found:
            if all(d[1] != q[1] or _ios(d[0], q[0]) < 0.5 for q in kept):
                kept.append(d)
        for b, name, s in kept:
            u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            P = pixel_to_ground(K, pose, [[u, v]])[0]
            if not np.isfinite(P).all():
                continue
            dets.append({"frame": fi + 1, "time": r.get("dt"), "cls": name, "score": s, "x": float(P[0]),
                         "y": float(P[1]), "alt": r["alt"], "box": [float(t) for t in b],
                         "thumb": _thumb(frame, b)})
    cap.release()
    # 같은 물체 묶기
    objs = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        for o in objs:
            if o["cls"] == d["cls"] and np.hypot(o["x"] - d["x"], o["y"] - d["y"]) < merge_m:
                o["n"] += 1
                o["xs"].append(d["x"])
                o["ys"].append(d["y"])
                break
        else:
            objs.append({"cls": d["cls"], "score": d["score"], "n": 1, "xs": [d["x"]], "ys": [d["y"]],
                         "x": d["x"], "y": d["y"], "thumb": d["thumb"], "frame": d["frame"]})
    for o in objs:
        o["x"], o["y"] = float(np.median(o.pop("xs"))), float(np.median(o.pop("ys")))
        o["lat"], o["lon"] = geo.to_latlon(o["x"], o["y"])
        o["kind"] = "obstacle" if o["cls"] in OBSTACLE else "litter"
    objs = [o for o in objs if o["n"] >= 2]  # 한 번만 보인 건 오탐일 가능성 → 제외
    track = [[r["lat"], r["lon"], r.get("alt", 0)] for r in tel[::max(1, int(fps / 4))]]
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "cls", "lat", "lon", "seen_frames", "best_score", "first_frame"])
        for o in objs:
            w.writerow([o["kind"], o["cls"], f"{o['lat']:.7f}", f"{o['lon']:.7f}", o["n"], f"{o['score']:.2f}", o["frame"]])
    (out / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
         "properties": {k: o[k] for k in ("kind", "cls", "n", "score", "frame")}} for o in objs]},
        ensure_ascii=False), encoding="utf-8")
    _map_html(out / "map.html", track, objs, yaw_src, pitch)
    alts = [r.get("alt", 0) for r in tel]
    dist = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    print(f"비행: {len(tel)}프레임 · 최고 {max(alts):.1f} m · 이동거리 {dist:.1f} m (GPS 흔들림 포함)")
    from collections import Counter
    print(f"탐지 {len(dets)}건 → 물체 {len(objs)}개 (2프레임 이상): " +
          ", ".join(f"{k} {v}" for k, v in Counter(o['cls'] for o in objs).most_common()))
    print(f"방향(yaw) 출처: {sorted(yaw_src)} · 짐벌 {pitch}°")
    print(f"→ {out / 'map.html'} (브라우저로 열기) · objects.csv · objects.geojson")
    return objs


def _map_html(path, track, objs, yaw_src, pitch):
    data = {"track": track, "objs": [{k: o[k] for k in ("kind", "cls", "lat", "lon", "n", "score", "thumb", "frame")}
                                     for o in objs]}
    note = f"방향: {', '.join(sorted(yaw_src))} · 짐벌 {pitch}° (SRT에 방향 정보 없음 → 물체 위치는 수 m 오차 가능)"
    html = """<!doctype html><meta charset="utf-8"><title>드론 탐지 지도</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}#n{position:absolute;z-index:999;left:50px;top:8px;background:#fffd;padding:6px 10px;border-radius:6px;font:13px sans-serif}</style>
<div id="m"></div><div id="n">__NOTE__<br><span style="color:#e63946">●</span> 쓰레기 후보 <span style="color:#f4a261">●</span> 장애물(사람·의자 등) <span style="color:#1d3557">━</span> 비행 경로</div>
<script>
const D=__DATA__;
const m=L.map('m',{maxZoom:22});
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:22,maxNativeZoom:19,attribution:'Esri'}).addTo(m);
const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:22,maxNativeZoom:19,attribution:'OSM'});
L.control.layers({'위성':sat,'지도':osm}).addTo(m);
const line=L.polyline(D.track.map(t=>[t[0],t[1]]),{color:'#1d3557',weight:3}).addTo(m);
L.circleMarker(D.track[0].slice(0,2),{radius:6,color:'#2a9d8f',fillOpacity:1}).bindPopup('이륙').addTo(m);
D.objs.forEach(o=>{const c=o.kind==='litter'?'#e63946':'#f4a261';
 L.circleMarker([o.lat,o.lon],{radius:7,color:c,fillColor:c,fillOpacity:.8}).bindPopup(
  `<b>${o.cls}</b> (${o.kind==='litter'?'쓰레기 후보':'장애물'})<br>${o.n}프레임에서 탐지 · 신뢰도 ${o.score.toFixed(2)}<br>`+
  `${o.lat.toFixed(6)}, ${o.lon.toFixed(6)}<br>`+(o.thumb?`<img src="data:image/jpeg;base64,${o.thumb}">`:'')).addTo(m);});
const all=D.track.map(t=>[t[0],t[1]]).concat(D.objs.map(o=>[o.lat,o.lon]));
m.fitBounds(L.latLngBounds(all).pad(0.6));
</script>"""
    path.write_text(html.replace("__DATA__", json.dumps(data)).replace("__NOTE__", note), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.video_map", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", default=["yolo11s.pt"], help="여러 모델 가능 (예: COCO + 쓰레기 모델)")
    ap.add_argument("--every", type=float, default=0.5, help="몇 초마다 한 프레임")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--pitch", type=float, default=-90.0, help="짐벌 각도 (수직 아래 -90)")
    ap.add_argument("--yaw", type=float, help="기체 방향 (도, 북=0, 동=90). 없으면 이동방향 추정/북쪽 가정")
    ap.add_argument("--f35", type=float, default=24.0)
    ap.add_argument("--min_alt", type=float, default=2.0, help="이 고도 이상일 때만 (이착륙 장면 제외)")
    a = ap.parse_args(argv)
    run(a.video, a.srt, a.out, a.weights, a.every, a.conf, a.pitch, a.yaw, a.f35, a.min_alt)


if __name__ == "__main__":
    main()
