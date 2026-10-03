"""
전체 커버리지 vs 핫스팟 우선 경로 — 같은 하와이 월드에서 실제로 날린 결과 비교.

먼저 두 경로를 비행하고(run_hawaii.sh coverage / hotspot) 각각 evaluate.py를 돌린 뒤 실행.
출력: ros_sim/out_hawaii_compare/ (compare.json, compare.png)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
NAMES = {"coverage": "전체 커버리지", "hotspot": "핫스팟 우선"}


def main():
    gt = json.loads((GEN / "ground_truth.json").read_text())
    res, logs, views = {}, {}, {}
    for r in ("coverage", "hotspot"):
        out = HERE.parent / f"out_hawaii_{r}"
        res[r] = json.loads((out / "eval/summary.json").read_text())
        logs[r] = list(csv.DictReader(open(out / "flight_log.csv")))
        views[r] = [int(x["views"]) for x in csv.DictReader(open(out / "eval/label_views.csv"))]

    c, h = res["coverage"], res["hotspot"]
    table = {
        "비행거리 (m)": [c["flown_m"], h["flown_m"]],
        "비행시간 (s)": [c["flight_s"], h["flight_s"]],
        "사진 수": [c["frames_used"], h["frames_used"]],
        "라벨 3장 이상 촬영 (전체 889)": [c["imaged_ge3"], h["imaged_ge3"]],
        "  └ 경로 계획에 쓴 라벨": [c["imaged_ge3_prior_labels"], h["imaged_ge3_prior_labels"]],
        "  └ 경로 계획에 안 쓴 라벨": [c["imaged_ge3_unseen_labels"], h["imaged_ge3_unseen_labels"]],
        "라벨당 사진 수 (중앙값)": [c["views_median"], h["views_median"]],
    }
    eff = {r: res[r]["imaged_ge3"] * len(gt) / (res[r]["flown_m"] / 1000) for r in res}
    table["비행 1 km당 확보 라벨 수"] = [round(eff["coverage"], 1), round(eff["hotspot"], 1)]

    out = HERE.parent / "out_hawaii_compare"
    out.mkdir(exist_ok=True)
    (out / "compare.json").write_text(json.dumps(
        {"columns": [NAMES["coverage"], NAMES["hotspot"]], "rows": table,
         "hotspot_plan": json.loads((GEN / "meta.json").read_text())["hotspot"]}, ensure_ascii=False, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 12), sharey=True)
    for ax, r in zip(axes, ("coverage", "hotspot")):
        ax.plot([float(x["actual_x"]) for x in logs[r]], [float(x["actual_y"]) for x in logs[r]],
                lw=0.9, c="#1657D0")
        missed = [g for g, v in zip(gt, views[r]) if v < 3]
        got = [g for g, v in zip(gt, views[r]) if v >= 3]
        ax.scatter([g["east"] for g in got], [g["north"] for g in got], s=6, c="#2f9e44", label="3장 이상 촬영")
        ax.scatter([g["east"] for g in missed], [g["north"] for g in missed], s=12, c="#e03131", label="놓침")
        ax.set_aspect("equal")
        ax.set_xlabel("동 (m)")
        ax.set_title(f"{NAMES[r]}\n{res[r]['flown_m']:.0f} m · {res[r]['flight_s'] / 60:.1f}분 · "
                     f"라벨 {res[r]['imaged_ge3']:.0%}")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("북 (m)")
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=130)
    for k, (a, b) in table.items():
        print(f"{k:28s} {a!s:>10} {b!s:>10}")


if __name__ == "__main__":
    main()
