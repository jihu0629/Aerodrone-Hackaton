"""
물체별 부피 — SAM 2 마스크 + 여러 프레임 투표로 물체 점만 골라내기.

반경으로 물체를 자르면 그림자·바닥 잡음까지 들어가 부피가 크게 틀렸다 (작은 상자 5배).
여기서는:
  1) 물체 중심(지면 좌표)을 각 프레임에 다시 비춰, 그 위치를 점 프롬프트로 SAM 2가 물체 외곽선을 딴다
  2) 3D 점(조밀 복원 fused.ply, 없으면 희소 점)을 각 프레임에 비춰 '마스크 안에 들어온 비율'을 센다
  3) 여러 프레임에서 꾸준히 마스크 안에 들어온 점만 물체로 인정 (투표)
  4) 주변 바닥 높이 대비 높이지도 적분 = 부피.  +  가장 수직에 가까운 프레임의 마스크를 지면에 펴서 잰
     바닥 면적 × 윗면 높이(상자형 근사)도 같이 낸다

  python -m litter.objvol --orbit runs/orbit/0007 --srt DJI_0007.SRT --ply dense/fused.ply \
         --objects runs/map/0007_sfm/objects.csv --select "Plastic_Buoy:55,Styrofoam_Buoy:22" --out runs/orbit/0007/objvol
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .orbit3d import _center, _viewdir
from .seg import _imread, _imwrite
from .telemetry import read_srt
from .volume import fit_ground, to_metric, to_metric_auto, volume_heightmap, volume_hull


def read_ply(path):
    """COLMAP fused.ply (binary_little_endian) → xyz (N,3), rgb (N,3)."""
    with open(path, "rb") as f:
        header, props, n = [], [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}[t]))
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    rgb = np.stack([data["red"], data["green"], data["blue"]], 1) if "red" in data.dtype.names else None
    return xyz, rgb


class Scene:
    def __init__(self, orbit_dir, srt):
        import pycolmap

        self.dir = Path(orbit_dir)
        self.rec = pycolmap.Reconstruction(str(self.dir / "sparse" / "0"))
        self.imgs = sorted(self.rec.images.values(), key=lambda i: i.name)
        self.cam = self.rec.cameras[self.imgs[0].camera_id]
        fidx = json.loads((self.dir / "frames" / "frames.json").read_text(encoding="utf-8"))
        tel = {r["frame"]: r for r in read_srt(srt)}
        P = np.array([p.xyz for p in self.rec.points3D.values()])
        C = np.array([_center(i) for i in self.imgs])
        D = np.array([_viewdir(i) for i in self.imgs])
        alt = np.array([tel[fidx[i.name]]["alt"] for i in self.imgs])
        from .geometry import Geo
        geo = Geo(None, lon=tel[fidx[self.imgs[0].name]]["lon"])
        gps = np.array([geo.to_xy(tel[fidx[i.name]]["lat"], tel[fidx[i.name]]["lon"]) for i in self.imgs])
        self.T, self.R, self.s, self.info = to_metric_auto(P, C, D, alt, gps)
        self.p0 = self.info["p0"]
        self.sparse = P
        self.D = D

    def to_sfm(self, q):
        return (np.asarray(q, float) / self.s) @ self.R + self.p0

    def project(self, im, X):
        """SfM 좌표 (N,3) → 픽셀 (N,2), 카메라 뒤는 NaN."""
        Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Xc = X @ np.asarray(Tc.rotation.matrix()).T + np.asarray(Tc.translation)
        uv = np.full((len(X), 2), np.nan)
        ok = Xc[:, 2] > 0
        if ok.any():
            uv[ok] = np.asarray(self.cam.img_from_cam(Xc[ok], False))
        return uv

    def nadir_score(self, im):
        k = self.imgs.index(im)
        return float(-(self.D[k] @ self.R.T)[2])  # 1 = 수직 아래


def sam_mask(sam, img, uv, crop=260, max_frac=0.6):
    """uv(물체 중심 픽셀)을 점 프롬프트로 물체 마스크. 주변을 잘라서(crop) 빠르게.
    마스크가 잘린 영역의 max_frac 이상이면 바닥을 통째로 딴 것 → 실패로 본다."""
    H, W = img.shape[:2]
    x, y = int(uv[0]), int(uv[1])
    x0, y0 = max(x - crop, 0), max(y - crop, 0)
    x1, y1 = min(x + crop, W), min(y + crop, H)
    sub = img[y0:y1, x0:x1]
    r = sam(sub, points=[[x - x0, y - y0]], labels=[1], verbose=False)[0]
    if r.masks is None:
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != sub.shape[:2]:
        m = cv2.resize(m, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab = cv2.connectedComponents(m)
    if n > 1 and lab[y - y0, x - x0] > 0:  # 프롬프트 점이 속한 조각만
        m = (lab == lab[y - y0, x - x0]).astype(np.uint8)
    if m.mean() > max_frac or m.sum() < 30:
        return None
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    return full


def measure_object(sc, sam, center_xy, pts_sfm, n_views=24, search_r=0.8, vote=0.7, out_dir=None, tag="obj"):
    # 후보 점: 지면 좌표에서 중심 search_r 안
    Pm = sc.T(pts_sfm)
    d = np.hypot(Pm[:, 0] - center_xy[0], Pm[:, 1] - center_xy[1])
    cand = np.nonzero(d < search_r)[0]
    ring = Pm[(d > search_r) & (d < search_r + 0.6)]
    g = float(np.median(ring[:, 2])) if len(ring) > 20 else 0.0
    X = pts_sfm[cand]
    c_sfm = sc.to_sfm([center_xy[0], center_xy[1], 0.0])[None]
    # 물체가 화면 안쪽에 잘 보이는 프레임 고르기
    views = []
    for im in sc.imgs:
        uv = sc.project(im, c_sfm)[0]
        if np.isfinite(uv).all() and 150 < uv[0] < sc.cam.width - 150 and 150 < uv[1] < sc.cam.height - 150:
            views.append((np.hypot(uv[0] - sc.cam.width / 2, uv[1] - sc.cam.height / 2), im, uv))
    views.sort(key=lambda t: t[0])
    views = views[::max(1, len(views) // n_views)][:n_views]
    inside = np.zeros(len(X))
    seen = np.zeros(len(X))
    best_area, used, gallery = None, 0, []
    for _, im, uv in views:
        img = _imread(sc.dir / "frames" / im.name)
        m = sam_mask(sam, img, uv)
        if m is None:
            continue
        used += 1
        p = sc.project(im, X)
        ok = np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < sc.cam.width) & (p[:, 1] >= 0) & (p[:, 1] < sc.cam.height)
        pi = p[ok].astype(int)
        seen[ok] += 1
        inside[np.nonzero(ok)[0][m[pi[:, 1], pi[:, 0]] > 0]] += 1
        # 바닥 면적: 마스크 경계를 지면 평면으로 펴서 (가장 수직에 가까운 프레임 기준)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            cnt = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
            Tc = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            Rw = np.asarray(Tc.rotation.matrix()).T
            c0 = sc.T(_center(im)[None])[0]
            xy = np.asarray(sc.cam.cam_from_img(cnt))
            dirs = (sc.R @ (Rw @ np.c_[xy, np.ones(len(xy))].T)).T
            t = (g - c0[2]) / dirs[:, 2]
            G = c0[None, :2] + t[:, None] * dirs[:, :2]
            area = float(cv2.contourArea(G.astype(np.float32)))
            ns = sc.nadir_score(im)
            if best_area is None or ns > best_area[1]:
                best_area = (area, ns, G)
        if len(gallery) < 4:
            v = img.copy()
            v[m > 0] = (0.5 * v[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            x, y = int(uv[0]), int(uv[1])
            gallery.append(cv2.resize(v[max(y - 160, 0):y + 160, max(x - 160, 0):x + 160], (240, 240)))
    keep = (seen >= 3) & (inside / np.maximum(seen, 1) >= vote)
    O = Pm[cand[keep]].copy()
    O[:, 2] -= g
    O = O[O[:, 2] > 0.01]
    res = {"tag": tag, "views_used": used, "points_candidate": int(len(cand)), "points_object": int(len(O)),
           "ground_z": g}
    if len(O) >= 10:
        # 윗면 높이: 발자국 안쪽(중심에서 가까운 절반)의 중앙값 — 모서리 잡음 영향 제거
        c = np.median(O[:, :2], axis=0)
        r = np.linalg.norm(O[:, :2] - c, axis=1)
        h_top = float(np.median(O[r <= np.percentile(r, 50), 2]))
        O = O[O[:, 2] <= h_top * 1.3 + 0.02]   # 윗면보다 크게 튄 점 제거
        rect = cv2.minAreaRect(O[:, :2].astype(np.float32))
        L, W = sorted(rect[1], reverse=True)
        dense = len(O) > 300
        v_hm, foot, _ = volume_heightmap(O, agg="median" if dense else "max")
        res.update(length_m=float(L), width_m=float(W), height_m=h_top, volume_heightmap_L=v_hm * 1000,
                   volume_hull_L=volume_hull(O) * 1000)
        if best_area:
            res.update(mask_area_m2=best_area[0], volume_mask_box_L=best_area[0] * h_top * 1000)
    if out_dir and gallery:
        _imwrite(Path(out_dir) / f"{tag}_masks.jpg", np.hstack(gallery), 88)
    return res, O


def main(argv=None):
    from ultralytics import SAM

    ap = argparse.ArgumentParser(prog="litter.objvol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orbit", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--ply", help="조밀 복원 fused.ply (없으면 희소 점)")
    ap.add_argument("--objects", required=True, help="orbit_map objects.csv")
    ap.add_argument("--select", help="'클래스:프레임수' 목록 (예: Plastic_Buoy:55,Styrofoam_Buoy:22). 없으면 쓰레기 전부")
    ap.add_argument("--truth", help="'태그:가로x세로x높이cm' 목록 (비교용)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = Scene(a.orbit, a.srt)
    pts = read_ply(a.ply)[0] if a.ply else sc.sparse
    print(f"3D 점 {len(pts):,}개 ({'조밀' if a.ply else '희소'}) · 축척 편차 {sc.info['scale_spread_pct']:.1f}%")
    sam = SAM("sam2.1_b.pt")
    objs = list(csv.DictReader(open(a.objects, encoding="utf-8-sig")))
    sel = [tuple(s.split(":")) for s in a.select.split(",")] if a.select else None
    truth = dict(t.split(":") for t in a.truth.split(",")) if a.truth else {}
    results = []
    for o in objs:
        key = (o["cls"], o["seen_frames"])
        if sel and key not in sel:
            continue
        if not sel and o["kind"] != "litter":
            continue
        tag = f"{o['cls']}_{o['seen_frames']}"
        r, _ = measure_object(sc, sam, (float(o["local_x_m"]), float(o["local_y_m"])), pts, out_dir=out, tag=tag)
        r["lat"], r["lon"] = o["lat"], o["lon"]
        if tag in truth:
            l, w, h = (float(v) / 100 for v in truth[tag].split("x"))
            r["truth_L"] = l * w * h * 1000
        results.append(r)
        msg = f"{tag}: 프레임 {r['views_used']}장 마스크 · 물체 점 {r['points_object']}개"
        if "volume_heightmap_L" in r:
            msg += (f" · {r['length_m']*100:.0f}×{r['width_m']*100:.0f}×{r['height_m']*100:.0f} cm"
                    f" · 높이지도 {r['volume_heightmap_L']:.1f} L · 볼록껍질 {r['volume_hull_L']:.1f} L")
            if "volume_mask_box_L" in r:
                msg += f" · 마스크면적 {r['mask_area_m2']:.3f} m² × 높이 = {r['volume_mask_box_L']:.1f} L"
        if "truth_L" in r:
            msg += f"  (정답 {r['truth_L']:.1f} L)"
        print(msg)
    (out / "objvol.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
