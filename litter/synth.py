"""
합성 데이터 — 업체 데이터가 오기 전에 파이프라인 전체를 끝까지 검증하기 위한 가짜 해변.

정답을 전부 아는 상태로, **업체가 줄 법한 형태 그대로** 만든다:
  images/*.jpg        드론 프레임 (비스듬히 찍은 것 포함, 렌더링은 단순)
  labels_coco.json    폴리곤 라벨 (같은 물체가 여러 프레임에 라벨됨, item_id 포함)
  weights.csv         물체별 실측 무게 + 수거 지점 GPS (업체 "무게 기록" 흉내)
  telemetry.csv       프레임별 GPS·상대고도·짐벌각 (잡음 포함 — 실제 기록처럼)
  dsm.tif             SfM으로 만들었을 법한 DSM (높이 잡음 2 cm)
  truth.json          물체별 실제 위치·무게·형상 (평가용, 파이프라인은 안 씀)

일부러 넣은 현실적인 함정:
  - 비스듬한 촬영 (피치 -50° ~ -90°) → 드론 GPS를 위치로 쓰면 수십 m 어긋남
  - 텔레메트리 잡음 (GPS 2 m, 방향 2°, 피치 1°, 고도 0.5 m)
  - 속 빈 물체 (PET·박스·부표), 젖은 어망·로프 (무게 ×1–2.5)
  - 모래에 묻힌 어망 — 보이는 높이는 낮은데 실제로는 훨씬 무거움
  - 물 찬 PET병 — 겉보기는 같은데 10배 무거움
  - 로프는 폭 2–4 cm → DSM(5 cm 격자)에선 거의 안 보임
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics, Pose, ground_to_pixel
from .geometry import Geo

ORIGIN_LATLON = (37.1905, 125.9870)  # 굴업도 인근 해변 (대략)
EPSG = 32652
BEACH = (160.0, 60.0)                 # 로컬 x(동) × y(북) m
DSM_RES = 0.05

CLASS_MIX = {"eps_fragment": .24, "eps_buoy": .12, "eps_box": .05, "pet_bottle": .15, "plastic_buoy": .06,
             "net": .09, "rope": .09, "glass": .05, "metal": .05, "wood": .06, "tire": .04}
CLASS_NAME_KO = {"eps_fragment": "스티로폼 조각", "eps_buoy": "스티로폼 부표", "eps_box": "스티로폼 박스",
                 "pet_bottle": "PET병", "plastic_buoy": "플라스틱 부표", "net": "어망", "rope": "로프",
                 "glass": "유리병", "metal": "금속", "wood": "목재", "tire": "타이어"}
COLORS = {"eps_fragment": (235, 235, 235), "eps_buoy": (250, 250, 250), "eps_box": (240, 240, 230),
          "pet_bottle": (180, 220, 200), "plastic_buoy": (40, 120, 230), "net": (60, 150, 60),
          "rope": (30, 180, 220), "glass": (90, 160, 90), "metal": (150, 150, 160), "wood": (60, 100, 140),
          "tire": (30, 30, 30)}


def ground_z(x, y):
    return 0.03 * y + 0.3 * np.sin(x / 25.0) + 0.2 * np.cos(y / 15.0)


def _ellipse(cx, cy, L, W, th, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([L / 2 * np.cos(t), W / 2 * np.sin(t)], 1)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return p @ R.T + [cx, cy]


def _rect(cx, cy, L, W, th):
    p = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return p @ R.T + [cx, cy]


def _blob(cx, cy, L, W, th, rng, n=28):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.25 * np.sin(3 * t + rng.uniform(0, 6)) + 0.15 * np.sin(5 * t + rng.uniform(0, 6))
    p = np.stack([L / 2 * r * np.cos(t), W / 2 * r * np.sin(t)], 1)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return p @ R.T + [cx, cy]


def _rope(cx, cy, length, width, rng):
    """구불구불한 선을 폭 width로 두껍게 만든 폴리곤."""
    n = max(8, int(length / 0.5))
    ang = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.25, n))
    step = length / n
    pts = np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * step, 0)
    pts += [cx, cy] - pts.mean(0)
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return np.vstack([pts + nrm * width / 2, (pts - nrm * width / 2)[::-1]])


def make_item(i, cls, rng):
    cx, cy = rng.uniform(4, BEACH[0] - 4), rng.uniform(4, BEACH[1] - 4)
    th = rng.uniform(0, np.pi)
    u = rng.uniform
    wet = 1.0
    buried = False
    profile = "dome"
    if cls == "eps_fragment":
        L = u(.1, .6); W = L * u(.4, 1); H = u(.02, .08); poly = _blob(cx, cy, L, W, th, rng)
        vol = np.pi / 4 * L * W * H * .8; w = vol * rng.normal(18, 4)
    elif cls == "eps_buoy":
        L = u(.6, 1.2); W = L * u(.5, .7); H = W * .9; poly = _ellipse(cx, cy, L, W, th)
        vol = np.pi / 6 * L * W * H * 1.3; w = vol * rng.normal(22, 4)
    elif cls == "eps_box":
        L = u(.4, .8); W = u(.3, .5); H = u(.2, .35); poly = _rect(cx, cy, L, W, th); profile = "flat"
        vol = L * W * H; w = vol * rng.normal(6, 1.5)
    elif cls == "pet_bottle":
        L = u(.18, .32); W = u(.06, .09); H = W; poly = _rect(cx, cy, L, W, th)
        vol = L * W * H * .8; w = rng.lognormal(np.log(.03), .3)
        if rng.random() < .1:
            w += vol * 700  # 물·모래가 찬 병
    elif cls == "plastic_buoy":
        L = u(.3, .6); W = L; H = L * .9; poly = _ellipse(cx, cy, L, W, th)
        vol = np.pi / 6 * L * W * H; w = vol * rng.normal(50, 12)
    elif cls == "net":
        L = u(.6, 4); W = L * u(.3, .7); H = u(.06, .3); poly = _blob(cx, cy, L, W, th, rng); profile = "noisy"
        wet = u(1, 2.5)
        vol = np.pi / 4 * L * W * H * .6; w = vol * max(rng.normal(60, 12), 20) * wet
        if rng.random() < .25:  # 묻힌 어망: 보이는 높이는 낮고 실제는 훨씬 무거움
            buried = True; H = u(.015, .04); w *= u(2, 4)
    elif cls == "rope":
        L = u(2, 18); W = u(.02, .04); H = W; poly = _rope(cx, cy, L, W, rng)
        wet = u(1, 2); vol = L * np.pi / 4 * W * W; w = L * (W / .02) ** 2 * u(.1, .15) * wet
    elif cls == "glass":
        L = u(.2, .3); W = u(.06, .09); H = W; poly = _rect(cx, cy, L, W, th)
        vol = L * W * H * .8; w = u(.2, .5)
    elif cls == "metal":
        if rng.random() < .15:
            L = u(.5, 1.0); W = L * u(.5, 1); H = u(.2, .5); poly = _rect(cx, cy, L, W, th); profile = "flat"
            vol = L * W * H; w = u(5, 30)
        else:
            L = u(.1, .15); W = u(.06, .08); H = W; poly = _rect(cx, cy, L, W, th)
            vol = L * W * H * .8; w = u(.015, .05)
    elif cls == "wood":
        L = u(.3, 2.5); W = u(.05, .25); H = W * .7; poly = _rect(cx, cy, L, W, th); profile = "flat"
        wet = u(1, 1.5); vol = L * W * H; w = vol * rng.normal(450, 60) * wet
    elif cls == "tire":
        L = u(.55, .75); W = L; H = u(.18, .24); poly = _ellipse(cx, cy, L, W, th)
        vol = np.pi / 4 * L * W * H; w = u(7, 11) * u(1, 1.8)
    w = float(max(w, .005))
    return {"item_id": f"T{i:04d}", "cls": cls, "cls_raw": CLASS_NAME_KO[cls], "cx": float(cx), "cy": float(cy),
            "poly": poly.tolist(), "H": float(H), "profile": profile, "true_vol_m3": float(vol),
            "weight_kg": w, "wet": float(wet), "buried": buried}


def _burn_dsm(dsm, it, rng):
    P = np.asarray(it["poly"])
    pix = np.stack([P[:, 0] / DSM_RES, (BEACH[1] - P[:, 1]) / DSM_RES], 1)
    x0, y0 = np.floor(pix.min(0)).astype(int) - 1
    x1, y1 = np.ceil(pix.max(0)).astype(int) + 2
    x0, y0 = max(x0, 0), max(y0, 0)
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(sub, [np.round(pix - [x0, y0]).astype(np.int32)], 1)
    if sub.sum() == 0:
        return
    if it["profile"] == "flat":
        h = np.full(sub.shape, it["H"], np.float32)
    else:
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
        h = it["H"] * np.sqrt(np.clip(dist / max(dist.max(), 1e-6), 0, 1)).astype(np.float32)
        if it["profile"] == "noisy":
            h *= rng.uniform(.4, 1.0, h.shape).astype(np.float32)
    win = dsm[y0:y1, x0:x1]
    hh = h[: win.shape[0], : win.shape[1]]
    m = sub[: win.shape[0], : win.shape[1]].astype(bool)
    win[m] = np.maximum(win[m], win[m] + hh[m])


def _render(K, pose, items, rng, sand):
    img = sand.copy()
    for it in items:
        P = np.asarray(it["poly"])
        z = ground_z(P[:, 0], P[:, 1]) + it["H"] * .5
        uv = ground_to_pixel(K, pose, np.column_stack([P, z]))
        if not np.isfinite(uv).all():
            continue
        c = np.array(COLORS[it["cls"]], float) * rng.uniform(.85, 1.05)
        cv2.fillPoly(img, [np.round(uv).astype(np.int32)], tuple(int(v) for v in np.clip(c, 0, 255)), cv2.LINE_AA)
    return img


def generate(out, n_items=500, seed=7, frame_px=(4000, 3000), f35=24.0, alt=40.0,
             views_per_item=2, render=True):
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    geo = Geo(EPSG)
    ox, oy = geo.to_xy(*ORIGIN_LATLON)

    cls_list = list(CLASS_MIX)
    probs = np.array(list(CLASS_MIX.values()))
    items = [make_item(i, rng.choice(cls_list, p=probs / probs.sum()), rng) for i in range(n_items)]

    # DSM (로컬 → UTM transform), 지형 + 물체 + SfM 잡음
    Wd, Hd = int(BEACH[0] / DSM_RES), int(BEACH[1] / DSM_RES)
    xs = (np.arange(Wd) + .5) * DSM_RES
    ys = BEACH[1] - (np.arange(Hd) + .5) * DSM_RES
    dsm = ground_z(xs[None, :], ys[:, None]).astype(np.float32)
    for it in items:
        _burn_dsm(dsm, it, rng)
    dsm += cv2.GaussianBlur(rng.normal(0, .02, dsm.shape).astype(np.float32), (0, 0), 1.0)
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(out / "dsm.tif", "w", driver="GTiff", width=Wd, height=Hd, count=1, dtype="float32",
                       crs=f"EPSG:{EPSG}", transform=from_origin(ox, oy + BEACH[1], DSM_RES, DSM_RES),
                       nodata=-9999, compress="deflate") as ds:
        ds.write(dsm, 1)

    # 비행: 동-서 왕복 라인, 라인마다 짐벌 피치 다르게 (동영상 촬영 흉내)
    Wp, Hp = frame_px
    K = Intrinsics.from_f35(Wp, Hp, f35)
    sand = np.full((Hp, Wp, 3), (150, 190, 215), np.uint8)
    noise = cv2.resize(rng.normal(0, 12, (Hp // 50, Wp // 50, 3)).astype(np.float32), (Wp, Hp))
    sand = np.clip(sand + noise, 0, 255).astype(np.uint8)
    # (라인 y, 짐벌 피치, 카메라 방향): 수직 라인은 진행방향(동/서), 비스듬한 라인은
    # 바다 쪽(남)에서 해변(북)을 바라봄 — 드론은 해변 밖에 있고 쓰레기는 수십 m 앞
    lines = [(-30, -50, 0.0), (-8, -65, 0.0), (15, -90, 90.0), (38, -75, 0.0), (45, -90, 270.0)]
    frames = []
    for ly, pitch, yaw in lines:
        for fx in np.arange(-10, BEACH[0] + 10, 14.0):
            frames.append(Pose(float(fx), float(ly), alt, yaw, pitch))

    tel_rows, coco_imgs, coco_anns = [], [], []
    views = {it["item_id"]: [] for it in items}
    for fi, pose in enumerate(frames):
        name = f"DJI_{fi:04d}.jpg"
        visible = []
        for it in items:
            P = np.asarray(it["poly"])
            uv = ground_to_pixel(K, pose, np.column_stack([P, ground_z(P[:, 0], P[:, 1]) + it["H"] * .5]))
            if np.isfinite(uv).all() and (uv[:, 0] > 20).all() and (uv[:, 0] < Wp - 20).all() \
                    and (uv[:, 1] > 20).all() and (uv[:, 1] < Hp - 20).all():
                visible.append(it)
                views[it["item_id"]].append((fi, uv))
        if render:
            cv2.imwrite(str(out / "images" / name), _render(K, pose, visible, rng, sand),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        # 기록된 텔레메트리 = 진짜 + 잡음 (위경도로 저장, 실제 로그처럼)
        lat, lon = geo.to_latlon(ox + pose.x + rng.normal(0, 2.0), oy + pose.y + rng.normal(0, 2.0))
        tel_rows.append({"image": name, "lat": f"{lat:.8f}", "lon": f"{lon:.8f}",
                         "rel_alt": f"{pose.alt + rng.normal(0, .5):.2f}",
                         "gimbal_yaw": f"{pose.yaw + rng.normal(0, 2):.2f}",
                         "gimbal_pitch": f"{pose.pitch + rng.normal(0, 1):.2f}", "gimbal_roll": "0",
                         "f35": f35})
        coco_imgs.append({"id": fi, "file_name": name, "width": Wp, "height": Hp})

    cats = {c: k + 1 for k, c in enumerate(CLASS_MIX)}
    aid = 0
    labeled = 0
    for it in items:
        v = views[it["item_id"]]
        if not v:
            continue
        labeled += 1
        pick = rng.choice(len(v), size=min(views_per_item, len(v)), replace=False)
        for k in pick:
            fi, uv = v[k]
            aid += 1
            x0, y0 = uv.min(0)
            x1, y1 = uv.max(0)
            coco_anns.append({"id": aid, "image_id": fi, "category_id": cats[it["cls"]],
                              "segmentation": [uv.round(1).ravel().tolist()],
                              "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                              "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
                              "attributes": {"item_id": it["item_id"]}})
    coco = {"images": coco_imgs, "annotations": coco_anns,
            "categories": [{"id": i, "name": CLASS_NAME_KO[c]} for c, i in cats.items()]}
    (out / "labels_coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    with open(out / "telemetry.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tel_rows[0]))
        w.writeheader()
        w.writerows(tel_rows)
    with open(out / "weights.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "weight_kg", "gt_lat", "gt_lon"])
        for it in items:
            if views[it["item_id"]]:
                lat, lon = geo.to_latlon(ox + it["cx"] + rng.normal(0, .5), oy + it["cy"] + rng.normal(0, .5))
                w.writerow([it["item_id"], f"{it['weight_kg']:.4f}", f"{lat:.8f}", f"{lon:.8f}"])
    truth = [{**{k: v for k, v in it.items() if k != "poly"}, "x": ox + it["cx"], "y": oy + it["cy"],
              "labeled": bool(views[it["item_id"]])} for it in items]
    (out / "truth.json").write_text(json.dumps({"epsg": EPSG, "items": truth}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"합성 데이터: 물체 {n_items} (라벨됨 {labeled}) · 프레임 {len(frames)} · 라벨 {len(coco_anns)} → {out}")
    return out
