"""
하와이 비행 결과 평가: 정답 라벨이 실제로 몇 장의 사진에 찍혔는가.

- 프레임 위치: 저장 파일 시각(mtime) ↔ flight_log.csv의 wall 시각으로 가장 가까운 자세를 붙인다.
- 촬영 범위: 하향(수직) 카메라 가정. 이미지 위쪽 = 기체 전방, 오른쪽 = 기체 오른쪽.
- 지상 해상도(GSD) = 2·고도·tan(화각/2) / 1280.

출력 (out_hawaii/eval/):
  summary.json        라벨 촬영 비율, 시점 수 분포, GSD, 라벨 크기(px)
  label_views.csv     라벨별 찍힌 횟수
  overlay_*.jpg       프레임에 정답 박스를 투영해 그린 확인용 그림 (방향 검증)
  coverage.png        비행 궤적 + 촬영 범위 + 라벨 위치

사용: python ros_sim/hawaii/evaluate.py
"""
import csv
import json
import math
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
OUT = HERE.parent / "out_hawaii_coverage"  # main()에서 --route로 바꿈
EDGE = 0.05  # 프레임 가장자리 5%는 왜곡·잘림으로 보고 제외
# 파일 저장 시각이 실제 촬영보다 늦게 찍힌다. 칩 원본과의 상관을 시간 이동별로 재서
# +0.2 s에서 최대(0.42→0.49)였음 — 5 m/s 비행 기준 약 1 m 위치 차이.
TIME_OFFSET_S = 0.2
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def yaw_of(r):
    w, x, y, z = (float(r[k]) for k in ("qw", "qx", "qy", "qz"))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def project(px_e, px_n, pose, meta):
    """지면 점(동,북) → 이미지 픽셀(u,v). 범위 밖이면 None."""
    W, H = meta["cam_px"]
    alt = max(pose["z"] - 0.01, 0.5)
    gsd = 2 * alt * math.tan(math.radians(meta["cam_hfov_deg"]) / 2) / W
    de, dn = px_e - pose["x"], px_n - pose["y"]
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    fwd = de * c + dn * s
    right = de * s - dn * c
    return W / 2 + right / gsd, H / 2 - fwd / gsd, gsd


def flown(log):
    """실제 비행거리(m)와 시간(s) — 위치 로그를 그대로 적분."""
    pts = [(float(r["actual_x"]), float(r["actual_y"]), float(r["actual_z"])) for r in log]
    dist = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return dist, float(log[-1]["t"]) - float(log[0]["t"])


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="coverage", choices=["coverage", "hotspot"])
    OUT = HERE.parent / f"out_hawaii_{ap.parse_args().route}"
    meta = json.loads((GEN / "meta.json").read_text())
    gt = json.loads((GEN / "ground_truth.json").read_text())
    log = list(csv.DictReader(open(OUT / "flight_log.csv")))
    walls = [float(r["wall"]) for r in log]
    W, H = meta["cam_px"]

    frames = []
    for f in sorted((OUT / "frames").glob("*.png"), key=os.path.getmtime):
        t = os.path.getmtime(f) + TIME_OFFSET_S
        i = min(range(len(walls)), key=lambda k: abs(walls[k] - t))
        if abs(walls[i] - t) > 0.5:
            continue
        r = log[i]
        pose = {"x": float(r["actual_x"]), "y": float(r["actual_y"]), "z": float(r["actual_z"]), "yaw": yaw_of(r)}
        if pose["z"] < meta["alt_m"] * 0.8:  # 이착륙 중 프레임 제외
            continue
        frames.append((f, pose))

    views = [0] * len(gt)
    sizes, gsds = [], []
    for f, pose in frames:
        for j, g in enumerate(gt):
            u, v, gsd = project(g["east"], g["north"], pose, meta)
            if EDGE * W <= u <= (1 - EDGE) * W and EDGE * H <= v <= (1 - EDGE) * H:
                views[j] += 1
                sizes.append(max(g["w_m"], g["h_m"]) / gsd)
        gsds.append(project(0, 0, pose, meta)[2])

    n = len(gt)
    dist, dur = flown(log)
    split = {sp: [v for g, v in zip(gt, views) if g.get("split") == sp] for sp in ("train", "eval")}
    summary = {
        "route": OUT.name.replace("out_hawaii_", ""),
        "flown_m": round(dist, 1), "flight_s": round(dur, 1),
        "imaged_ge3_prior_labels": round(sum(v >= 3 for v in split["train"]) / max(len(split["train"]), 1), 3),
        "imaged_ge3_unseen_labels": round(sum(v >= 3 for v in split["eval"]) / max(len(split["eval"]), 1), 3),
        "frames_used": len(frames), "labels": n,
        "imaged_ge1": round(sum(v >= 1 for v in views) / n, 3),
        "imaged_ge3": round(sum(v >= 3 for v in views) / n, 3),
        "views_median": sorted(views)[n // 2],
        "gsd_cm_median": round(100 * sorted(gsds)[len(gsds) // 2], 2) if gsds else None,
        "label_px_median": round(sorted(sizes)[len(sizes) // 2], 1) if sizes else None,
        "texture_cm": 2.0,
    }
    ev = OUT / "eval"
    ev.mkdir(exist_ok=True)
    (ev / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    with open(ev / "label_views.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["chip", "label", "east", "north", "views"])
        for g, v in zip(gt, views):
            w.writerow([g["chip"], g["label"], g["east"], g["north"], v])

    # 정답 박스가 가장 많이 들어간 프레임 3장에 박스를 그려 방향 검증
    scored = []
    for f, pose in frames:
        inside = [g for g in gt if EDGE * W <= project(g["east"], g["north"], pose, meta)[0] <= (1 - EDGE) * W
                  and EDGE * H <= project(g["east"], g["north"], pose, meta)[1] <= (1 - EDGE) * H]
        scored.append((len(inside), f, pose, inside))
    scored.sort(key=lambda s: -s[0])
    for k, (cnt, f, pose, inside) in enumerate(scored[:3]):
        img = cv2.imread(str(f))
        for g in inside:
            u, v, gsd = project(g["east"], g["north"], pose, meta)
            hw, hh = g["w_m"] / gsd / 2, g["h_m"] / gsd / 2
            cv2.rectangle(img, (int(u - hw), int(v - hh)), (int(u + hw), int(v + hh)), (0, 0, 255), 2)
        cv2.imwrite(str(ev / f"overlay_{k}.jpg"), img)

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.plot([float(r["actual_x"]) for r in log], [float(r["actual_y"]) for r in log], lw=0.8, label="실제 비행")
    sc = ax.scatter([g["east"] for g in gt], [g["north"] for g in gt], c=views, s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="찍힌 사진 수")
    ax.set_aspect("equal")
    ax.set_xlabel("동 (m)")
    ax.set_ylabel("북 (m)")
    ax.set_title(f"라벨 {n}개 중 1장 이상 {summary['imaged_ge1']:.0%}, 3장 이상 {summary['imaged_ge3']:.0%}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ev / "coverage.png", dpi=130)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
