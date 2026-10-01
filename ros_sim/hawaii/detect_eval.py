"""
시뮬레이션 사진에 탐지 모델을 돌려 "찍힌 쓰레기를 실제로 찾았는가"를 잰다.

1. 프레임마다 YOLO 탐지 (모델은 시뮬 구역을 뺀 하와이 칩으로 학습 — train_detector.py)
2. 탐지 중심 픽셀 → 지면 좌표(동, 북)로 역투영 (evaluate.py와 같은 하향 카메라 가정)
3. 여러 프레임의 탐지를 반경 R 안에서 하나로 묶음(물체 후보), 2장 이상에서 잡힌 것만 확정
4. 정답 라벨과 거리 기준으로 짝지어 재현율·정밀도 계산

정밀도는 참고용이다: 칩에는 라벨 안 된 쓰레기도 많아서, 맞게 찾아도 오탐으로 셀 수 있다.

출력: ros_sim/out_hawaii/eval/detection_summary.json, detections_map.png, det_overlay.jpg
사용: python ros_sim/hawaii/detect_eval.py [--weights runs/hawaii_det/train/weights/best.pt]
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

import evaluate as E

MERGE_R = 0.6   # m, 같은 물체로 묶는 반경
MATCH_R = 0.8   # m, 정답과 짝짓는 반경 (시각 보정 후 남는 위치 오차 ~수십 cm 고려)


def load_frames(meta):
    log = list(csv.DictReader(open(E.OUT / "flight_log.csv")))
    walls = [float(r["wall"]) for r in log]
    out = []
    for f in sorted((E.OUT / "frames").glob("*.png"), key=os.path.getmtime):
        t = os.path.getmtime(f) + E.TIME_OFFSET_S
        i = min(range(len(walls)), key=lambda k: abs(walls[k] - t))
        if abs(walls[i] - t) > 0.5:
            continue
        r = log[i]
        pose = {"x": float(r["actual_x"]), "y": float(r["actual_y"]), "z": float(r["actual_z"]), "yaw": E.yaw_of(r)}
        if pose["z"] >= meta["alt_m"] * 0.8:
            out.append((f, pose))
    return out


def unproject(u, v, pose, meta):
    W, H = meta["cam_px"]
    gsd = E.project(pose["x"], pose["y"], pose, meta)[2]
    fwd, right = (H / 2 - v) * gsd, (u - W / 2) * gsd
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return pose["x"] + fwd * c + right * s, pose["y"] + fwd * s - right * c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(E.HERE.parents[1] / "runs/hawaii_det/train/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    meta = json.loads((E.GEN / "meta.json").read_text())
    gt = json.loads((E.GEN / "ground_truth.json").read_text())
    W, H = meta["cam_px"]
    model = YOLO(a.weights)
    frames = load_frames(meta)

    dets = []          # (동, 북, 프레임번호, 클래스, 점수)
    per_frame = []     # 단일 사진 재현율
    best_frame = None
    for k, (f, pose) in enumerate(frames):
        res = model.predict(str(f), imgsz=1280, conf=a.conf, verbose=False, device="mps")[0]
        fd = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            if not (E.EDGE * W <= u <= (1 - E.EDGE) * W and E.EDGE * H <= v <= (1 - E.EDGE) * H):
                continue
            e, n = unproject(u, v, pose, meta)
            fd.append((e, n))
            dets.append((e, n, k, int(b.cls), float(b.conf)))
        vis = [g for g in gt if E.EDGE * W <= E.project(g["east"], g["north"], pose, meta)[0] <= (1 - E.EDGE) * W
               and E.EDGE * H <= E.project(g["east"], g["north"], pose, meta)[1] <= (1 - E.EDGE) * H]
        if vis:
            hit = sum(any(math.hypot(g["east"] - e, g["north"] - n) <= MATCH_R for e, n in fd) for g in vis)
            per_frame.append(hit / len(vis))
            if best_frame is None or len(vis) > best_frame[0]:
                best_frame = (len(vis), res)

    clusters = []      # [동합, 북합, 개수, 프레임집합]
    for e, n, k, c, s in dets:
        for cl in clusters:
            if math.hypot(cl[0] / cl[2] - e, cl[1] / cl[2] - n) <= MERGE_R:
                cl[0] += e; cl[1] += n; cl[2] += 1; cl[3].add(k)
                break
        else:
            clusters.append([e, n, 1, {k}])
    objs = [(cl[0] / cl[2], cl[1] / cl[2], len(cl[3])) for cl in clusters]
    confirmed = [o for o in objs if o[2] >= 2]

    def recall(cands):
        used, hit = set(), 0
        for g in gt:
            best = None
            for i, (e, n, _) in enumerate(cands):
                d = math.hypot(g["east"] - e, g["north"] - n)
                if i not in used and d <= MATCH_R and (best is None or d < best[0]):
                    best = (d, i)
            if best:
                used.add(best[1]); hit += 1
        return hit / len(gt), len(used)

    r_all, m_all = recall(objs)
    r_conf, m_conf = recall(confirmed)
    summary = {
        "frames": len(frames), "detections": len(dets), "gt_labels": len(gt),
        "single_frame_recall_mean": round(sum(per_frame) / len(per_frame), 3) if per_frame else None,
        "candidates": len(objs), "recall_candidates": round(r_all, 3),
        "precision_candidates_ref": round(m_all / len(objs), 3) if objs else None,
        "confirmed_ge2_frames": len(confirmed), "recall_confirmed": round(r_conf, 3),
        "precision_confirmed_ref": round(m_conf / len(confirmed), 3) if confirmed else None,
        "merge_r_m": MERGE_R, "match_r_m": MATCH_R, "conf": a.conf,
    }
    ev = E.OUT / "eval"
    ev.mkdir(exist_ok=True)
    (ev / "detection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    if best_frame:
        cv2.imwrite(str(ev / "det_overlay.jpg"), best_frame[1].plot(labels=False, line_width=2))

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.scatter([g["east"] for g in gt], [g["north"] for g in gt], s=10, c="#bbbbbb", label="정답 라벨")
    ax.scatter([o[0] for o in confirmed], [o[1] for o in confirmed], s=6, c="#d6336c", label="확정 탐지(2장+)")
    ax.set_aspect("equal")
    ax.set_xlabel("동 (m)")
    ax.set_ylabel("북 (m)")
    ax.set_title(f"재현율 {summary['recall_confirmed']:.0%} (확정) · 단일 사진 {summary['single_frame_recall_mean']:.0%}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ev / "detections_map.png", dpi=130)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
