"""
선회 영상 3D 결과로 탐지 물체를 **정확한 방향**으로 지도에 찍기.

video_map.py는 SRT에 기체 방향(yaw)이 없어서 물체 위치가 수 m씩 틀렸다.
여기서는 3D 복원(SfM)이 알아낸 프레임별 카메라 방향을 쓰고,
3D 좌표계를 GPS에 맞춰(카메라 위치들 ↔ SRT 위경도, 2D 회전+이동) 지도 좌표로 옮긴다.

  python -m litter.orbit_map --orbit runs/orbit/0007 --srt DJI_0007.SRT --out runs/map/0007_sfm \
         --litter runs/seg/aihub_gsd_det_s/weights/best.pt --obstacle yolo11s.pt

결과: detections/*.jpg (탐지 결과 그림), sheet.jpg (모음), map.html (위성지도 핑), overview.png, objects.csv
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Geo
from .orbit3d import _center, _viewdir
from .seg import _imread, _imwrite
from .telemetry import read_srt
from .video_map import OBSTACLE, _map_html, _thumb
from .volume import to_metric, to_metric_auto


def fit_rigid2d(A, B):
    """A(N,2) → B(N,2) 회전+이동 (Kabsch). 반환 (Rot, t, 잔차 RMS)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    Rt = (U @ Vt).T
    if np.linalg.det(Rt) < 0:  # 반사 방지
        Vt[-1] *= -1
        Rt = (U @ Vt).T
    t = cb - ca @ Rt.T
    res = np.linalg.norm(A @ Rt.T + t - B, axis=1)
    return Rt, t, float(np.sqrt(np.mean(res ** 2)))


def run(orbit_dir, srt, out, litter_w, obstacle_w=None, conf=0.3, merge_m=0.5, min_hits=3):
    import pycolmap
    from ultralytics import YOLO

    orbit_dir, out = Path(orbit_dir), Path(out)
    (out / "detections").mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(orbit_dir / "sparse" / "0"))
    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    fidx = json.loads((orbit_dir / "frames" / "frames.json").read_text(encoding="utf-8"))
    tel = {r["frame"]: r for r in read_srt(srt) if "lat" in r}
    P = np.array([p.xyz for p in rec.points3D.values()])
    C = np.array([_center(i) for i in imgs])
    D = np.array([_viewdir(i) for i in imgs])
    alt = np.array([tel[fidx[i.name]]["alt"] for i in imgs])
    geo = Geo(None, lon=tel[fidx[imgs[0].name]]["lon"])
    gps = np.array([geo.to_xy(tel[fidx[i.name]]["lat"], tel[fidx[i.name]]["lon"]) for i in imgs])
    T, Rg, s, info = to_metric_auto(P, C, D, alt, gps)
    Cm = T(C)
    R2, t2, rms = fit_rigid2d(Cm[:, :2], gps)
    to_map = lambda q: np.asarray(q)[..., :2] @ R2.T + t2
    print(f"3D ↔ GPS 맞춤: 카메라 {len(imgs)}대 · 잔차 {rms:.2f} m · 크기 기준 {info.get('scale_source')} · 고도 축척 편차 {info['scale_spread_pct']:.1f}%")

    cam = rec.cameras[imgs[0].camera_id]
    models = [("litter", YOLO(str(litter_w)))] + ([("obstacle", YOLO(str(obstacle_w)))] if obstacle_w else [])
    hits = []
    sheet = []
    for k, im in enumerate(imgs):
        img = _imread(orbit_dir / "frames" / im.name)
        Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rw = np.asarray(Tc.rotation.matrix()).T
        c0 = T(_center(im)[None])[0]
        vis = img.copy()
        n_here = 0
        for kind, m in models:
            r = m.predict(img, conf=conf, imgsz=1600, verbose=False)[0]
            for b, c, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                name = m.names[int(c)]
                if kind == "obstacle" and name not in OBSTACLE:
                    continue
                xy = np.asarray(cam.cam_from_img(np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]], float))).reshape(-1)[:2]
                dm = Rg @ (Rw @ np.array([xy[0], xy[1], 1.0]))
                if dm[2] >= 0:
                    continue
                g = c0 + (-c0[2] / dm[2]) * dm
                hits.append({"kind": kind, "cls": name, "score": float(sc), "g": g[:2], "frame": im.name,
                             "thumb": _thumb(img, b)})
                col = (0, 0, 255) if kind == "litter" else (0, 165, 255)
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                cv2.putText(vis, f"{name} {sc:.2f}", (int(b[0]), max(int(b[1]) - 5, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                n_here += 1
        t_s = fidx[im.name] / 59.94
        cv2.putText(vis, f"{t_s:.1f}s  alt {alt[k]:.1f}m  {n_here} det", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        _imwrite(out / "detections" / im.name, vis, 85)
        if k % 12 == 0:
            sheet.append(cv2.resize(vis, (640, 360)))
    while len(sheet) % 3:
        sheet.append(np.zeros_like(sheet[0]))
    _imwrite(out / "sheet.jpg", np.vstack([np.hstack(sheet[i:i + 3]) for i in range(0, len(sheet), 3)]), 85)

    # 같은 물체 묶기 (같은 종류, merge_m 안)
    objs = []
    for h in sorted(hits, key=lambda h: -h["score"]):
        for o in objs:
            if o["kind"] == h["kind"] and np.linalg.norm(o["g"] - h["g"]) < merge_m:
                o["gs"].append(h["g"])
                o["n"] += 1
                o["names"].append(h["cls"])
                break
        else:
            objs.append({"kind": h["kind"], "g": h["g"], "gs": [h["g"]], "n": 1, "score": h["score"],
                         "names": [h["cls"]], "thumb": h["thumb"], "frame": h["frame"]})
    objs = [o for o in objs if o["n"] >= min_hits]
    for o in objs:
        g = np.median(np.array(o.pop("gs")), axis=0)
        o["cls"] = max(set(o["names"]), key=o.pop("names").count)
        o["local"] = g
        mx, my = to_map(g)
        o["lat"], o["lon"] = geo.to_latlon(mx, my)
    track = [[tel[f]["lat"], tel[f]["lon"], tel[f]["alt"]] for f in sorted(tel)[::15]]
    _map_html(out / "map.html", track, objs, {"3D 복원 카메라 방향"}, "3D 추정")
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "cls", "lat", "lon", "seen_frames", "best_score", "local_x_m", "local_y_m"])
        for o in objs:
            w.writerow([o["kind"], o["cls"], f"{o['lat']:.7f}", f"{o['lon']:.7f}", o["n"], f"{o['score']:.2f}",
                        f"{o['local'][0]:.2f}", f"{o['local'][1]:.2f}"])
    # 개요 그림 (지도 타일 없이도 보이게)
    from pipeline._compat import apply_korean_font
    apply_korean_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(Cm[:, 0], Cm[:, 1], "-", c="#1d3557", lw=1.5, label="드론 경로 (3D 복원)")
    for o in objs:
        c = "#e63946" if o["kind"] == "litter" else "#f4a261"
        ax.scatter(*o["local"], s=40 + 6 * o["n"], c=c, edgecolors="k", lw=.5, zorder=3)
        ax.annotate(f"{o['cls']} ({o['n']})", o["local"], xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.scatter([], [], c="#e63946", label="쓰레기 후보")
    ax.scatter([], [], c="#f4a261", label="장애물")
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("탐지 물체 위치 (3D 기준, 단위 m)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=120)
    plt.close(fig)
    print(f"탐지 {len(hits)}건 → 물체 {len(objs)}개 ({min_hits}프레임 이상): " +
          ", ".join(f"{o['cls']}({o['n']})" for o in sorted(objs, key=lambda o: -o['n'])))
    print(f"→ {out}  (map.html · overview.png · sheet.jpg · detections/ · objects.csv)")
    return objs, to_map, geo


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.orbit_map", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit", required=True, help="orbit3d 결과 폴더 (sparse/, frames/)")
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--litter", required=True, help="쓰레기 탐지 모델")
    ap.add_argument("--obstacle", help="장애물(사람 등) 탐지 모델, 예: yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args(argv)
    run(a.orbit, a.srt, a.out, a.litter, a.obstacle, a.conf)


if __name__ == "__main__":
    main()
