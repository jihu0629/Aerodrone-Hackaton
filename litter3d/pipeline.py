"""전체 실행: DSM + 정사영상 (+ 분할 모델 또는 마스크 PNG) → objects.csv, grid, plan.json, summary.md"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .classes import KO_NAMES, density_table_markdown
from .gridmap import grid_kg, grid_to_csv, grid_to_png, grid_to_geotiff, save_meta
from .mass import estimate_masses, summarize_by_class, total_kg, ObjectMass
from .plan import PlanParams, make_plan
from .reconstruct import Surface, load_surface
from .segment import MaskList, color_baseline, masks_from_png
from .volume import volumes_from_masks


def run(surface: Surface, masks: MaskList, out_dir: str | Path, *, wet: bool = False,
        plan_params: PlanParams | None = None, cell_m: float = 10.0, min_conf: float = 0.0) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    masks = [m for m in masks if m[2] >= min_conf]
    vols = volumes_from_masks(surface.dsm, masks, surface.gsd_m)
    masses = estimate_masses(vols, wet=wet)

    # objects.csv
    with open(out / "objects.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(masses[0].to_dict().keys()) if masses else ["obj_id"]
        w = csv.DictWriter(f, fieldnames=fields + ["x_m", "y_m"])
        w.writeheader()
        for m in masses:
            d = m.to_dict()
            if surface.transform is not None:
                x, y = surface.transform * (m.cx_px, m.cy_px)
            else:
                x, y = m.cx_px * surface.gsd_m, m.cy_px * surface.gsd_m
            d.update({"x_m": round(float(x), 3), "y_m": round(float(y), 3)})
            for k, v in d.items():
                if isinstance(v, float):
                    d[k] = round(v, 5)
            w.writerow(d)

    # 격자
    pp = plan_params or PlanParams(cell_m=cell_m)
    g, meta = grid_kg(masses, surface.gsd_m, pp.cell_m, surface.transform)
    grid_to_csv(g, meta, out / "grid_kg.csv")
    grid_to_png(g, meta, out / "grid_kg.png")
    save_meta(meta, out / "grid_meta.json")
    try:
        grid_to_geotiff(g, meta, out / "grid_kg.tif", crs=surface.crs)
    except Exception:
        pass

    # 수거 계획
    plan = make_plan(masses, surface.gsd_m, surface.transform, pp)
    (out / "plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # 오버레이
    if surface.ortho is not None:
        draw_overlay(surface.ortho, masks, masses, out / "overlay.jpg")

    # 요약
    tk = total_kg(masses)
    by = summarize_by_class(masses)
    lines = ["# 해안쓰레기 무게 추정 결과", "",
             f"- 검출 물체: {len(masses)}개 (식생 제외 {sum(1 for m in masses if m.method != 'excluded')}개)",
             f"- 총 부피: {sum(m.volume_m3 for m in masses):.3f} m³",
             f"- 총 무게: **{tk[1]:.1f} kg** (범위 {tk[0]:.1f} – {tk[2]:.1f} kg)",
             f"- 부피 기반 {sum(1 for m in masses if m.method == 'volume')}개, 개수 기반 {sum(1 for m in masses if m.method == 'count')}개",
             f"- GSD {surface.gsd_m * 100:.2f} cm/px, 격자 {pp.cell_m:.0f} m",
             "", "## 클래스별", "", "| 클래스 | 개수 | 부피(m³) | kg (최소/대표/최대) |", "|---|---|---|---|"]
    for k, d in sorted(by.items(), key=lambda kv: -kv[1]["kg_typ"]):
        lines.append(f"| {d['class_ko']} | {d['count']} | {d['volume_m3']:.3f} | "
                     f"{d['kg_min']:.1f} / {d['kg_typ']:.1f} / {d['kg_max']:.1f} |")
    lines += ["", "## 수거 계획 (적재량·작업속도는 가정값)", "",
              f"- 마대 {plan.total_bags}장, 톤백 {plan.tonbags}개, 1 t 트럭 {plan.truck_trips}회",
              f"- 작업 {plan.worker_hours:.1f} 인·시간 → 4시간 기준 {plan.workers_for_4h}명",
              f"- 23 kg 초과(2인 이상/장비) 물체 {len(plan.heavy_items)}개",
              f"- 우선 수거 격자 상위 5: " + ", ".join(f"({c['row']},{c['col']}) {c['kg']:.1f} kg" for c in plan.priority_cells[:5]),
              f"- 차량 경로 {len(plan.routes)}대 ({plan.routes[0]['solver'] if plan.routes else '-'})",
              "", "## 가정", ""] + [f"- {a}" for a in plan.assumptions] + ["", "## 겉보기 밀도표", "", density_table_markdown()]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"n_objects": len(masses), "total_kg": tk, "by_class": by, "plan": plan.to_dict(), "out_dir": str(out)}


def draw_overlay(ortho_bgr: np.ndarray, masks: MaskList, masses: list[ObjectMass], path: str | Path) -> None:
    img = ortho_bgr.copy()
    rng = np.random.default_rng(0)
    colors = {k: tuple(int(c) for c in rng.integers(60, 255, 3)) for k in KO_NAMES}
    for (cls, m, _), mm in zip(masks, masses):
        col = colors.get(cls, (0, 255, 255))
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 1)
        cv2.putText(img, f"{mm.kg_typ * 1000:.0f}g", (int(mm.cx_px), int(mm.cy_px)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def run_from_files(dsm_path: str | Path, out_dir: str | Path, ortho_path: str | Path | None = None,
                   mask_png: str | Path | None = None, weights: str | Path | None = None, **kw) -> dict:
    surf = load_surface(dsm_path, ortho_path)
    if mask_png is not None:
        masks = masks_from_png(mask_png)
    elif weights is not None:
        from .segment import YoloSegmenter
        if surf.ortho is None:
            raise ValueError("YOLO 추론에는 정사영상이 필요합니다")
        masks = YoloSegmenter(weights).predict_tiled(surf.ortho)
    else:
        if surf.ortho is None:
            raise ValueError("마스크 PNG, 모델 가중치, 정사영상 중 하나는 있어야 합니다")
        masks = color_baseline(surf.ortho)
    return run(surf, masks, out_dir, **kw)
