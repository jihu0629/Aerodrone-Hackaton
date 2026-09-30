"""검출·부피·무게·수거계획 결과를 한 장의 HTML 리포트로 만든다 (외부 라이브러리·인터넷 불필요).

구성: 총 무게(히어로) → 핵심 수치 타일 → 검출 오버레이(물체별 툴팁) → 클래스별 kg 막대 →
격자 kg 히트맵 → 기존 방식(W2·W3) 비교 → 수거 계획 표 → 물체 표 → 가정.
색 규칙: 크기(무게)는 파랑 한 색의 진하기, 클래스 구분은 고정 순서 8색 (상위 7 클래스 + 기타).
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES, KO_NAMES
from .gridmap import grid_kg, object_xy_m
from .mass import ObjectMass, summarize_by_class, total_kg
from .plan import CollectionPlan
from .segment import MaskList

# 고정 순서 범주색 (dataviz 기준 팔레트, 밝은/어두운 모드 검증 완료)
CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_DARK = ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
OTHER = "other"


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt_kg(kg: float) -> str:
    if kg >= 100:
        return f"{kg:,.0f} kg"
    if kg >= 1:
        return f"{kg:,.1f} kg"
    return f"{kg * 1000:,.0f} g"


def _img_b64(bgr: np.ndarray, max_w: int = 1400, quality: int = 82) -> tuple[str, float]:
    h, w = bgr.shape[:2]
    s = min(1.0, max_w / w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), s


def _contours(mask: np.ndarray, scale: float, eps_px: float = 1.5) -> list[str]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, eps_px, True)
        pts = " ".join(f"{p[0][0] * scale:.1f},{p[0][1] * scale:.1f}" for p in c)
        out.append(pts)
    return out


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= v:
            return float(m * e)
    return float(10 * e)


def _crop_box(mask: np.ndarray, pad_frac: float = 0.6, min_px: int = 60) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ph = max(int((y1 - y0) * pad_frac), (min_px - (y1 - y0)) // 2, 6)
    pw = max(int((x1 - x0) * pad_frac), (min_px - (x1 - x0)) // 2, 6)
    H, W = mask.shape
    return max(y0 - ph, 0), min(y1 + ph, H), max(x0 - pw, 0), min(x1 + pw, W)


def _height_image(dsm_crop: np.ndarray, ground_z: float, h_max: float) -> np.ndarray:
    """DSM − 바닥 을 파랑 한 색 진하기 이미지로 (0 = 연함, h_max = 진함)."""
    h = np.nan_to_num(dsm_crop - ground_z, nan=0.0)
    t = np.clip(h / max(h_max, 1e-3), 0, 1)
    lut = np.array([[int(c[i:i + 2], 16) for i in (5, 3, 1)] for c in SEQ_LIGHT], np.float32)   # BGR
    idx = t * (len(lut) - 1)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, len(lut) - 1); f = (idx - lo)[..., None]
    img = lut[lo] * (1 - f) + lut[hi] * f
    return img.astype(np.uint8)


def _overlay_block(img_bgr: np.ndarray, items: list[tuple[np.ndarray, str, str, str | None]], max_w: int = 900,
                   stroke: int = 2) -> str:
    """이미지 + SVG 폴리곤 오버레이. items = [(mask, color css, tooltip, label or None)]"""
    src, s = _img_b64(img_bgr, max_w=max_w)
    H, W = img_bgr.shape[:2]
    parts = []
    for m, col, tip, label in items:
        polys = _contours(m, s)
        for pts in polys:
            parts.append(f'<polygon points="{pts}" style="stroke:{col};stroke-width:{stroke}" data-tip="{_esc(tip)}" tabindex="0"/>')
        if label and polys:
            ys, xs = np.nonzero(m)
            lx, ly = xs.min() * s, max(ys.min() * s - 4, 10)
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="ov-lab" style="fill:{col}">{_esc(label)}</text>')
    return (f'<div class="ov-wrap"><img src="{src}" alt="" width="{int(W * s)}" height="{int(H * s)}">'
            f'<svg viewBox="0 0 {int(W * s)} {int(H * s)}" preserveAspectRatio="none">{"".join(parts)}</svg></div>')


def build_report(surface, masks: MaskList, masses: list[ObjectMass], plan: CollectionPlan, out_path: str | Path,
                 *, title: str = "붕붕이 무게 리포트", site: str = "", truth_kg: float | None = None,
                 cell_m: float | None = None, vols=None, frames: list[tuple[str, np.ndarray, MaskList]] | None = None,
                 n_evidence: int = 12) -> Path:
    litter = [m for m in masses if m.method != "excluded"]
    tk = total_kg(litter)
    by = summarize_by_class(litter)
    order = sorted(by, key=lambda k: -by[k]["kg_typ"])
    top = order[:7]
    slot = {k: i for i, k in enumerate(top)}
    cls_color = lambda k: f"var(--c{slot[k]})" if k in slot else "var(--other)"
    gsd = surface.gsd_m
    cell = cell_m or (plan.params.get("cell_m", 10.0) if isinstance(plan.params, dict) else 10.0)
    n_vol = sum(1 for m in litter if m.method == "volume")
    n_cnt = sum(1 for m in litter if m.method == "count")
    vol = sum(m.volume_m3 for m in litter)
    area = sum(m.area_m2 for m in litter)
    w2 = len(litter) * 0.014
    w3 = area * 1e4 * 0.4 * 1.2 / 1000
    w3d = vol * 1e6 * 1.2 / 1000

    # ---------------- 오버레이 ----------------
    overlay_html = ""
    if surface.ortho is not None:
        src, s = _img_b64(surface.ortho)
        H, W = surface.ortho.shape[:2]
        polys = []
        for (cls, m, conf), mm in zip(masks, masses):
            if mm.method == "excluded":
                col = "var(--muted)"
            else:
                col = cls_color(cls)
            tip = (f"{mm.class_ko} · {_fmt_kg(mm.kg_typ)} (범위 {_fmt_kg(mm.kg_min)}–{_fmt_kg(mm.kg_max)}) · "
                   f"부피 {mm.volume_m3 * 1000:.1f} L · 면적 {mm.area_m2 * 1e4:.0f} cm² · 높이 {mm.h_max_m * 100:.1f} cm · "
                   f"{'부피×밀도' if mm.method == 'volume' else '개수/면적 기반' if mm.method == 'count' else '제외'}")
            for pts in _contours(m, s):
                polys.append(f'<polygon points="{pts}" style="stroke:{col}" data-tip="{_esc(tip)}" tabindex="0"/>')
            # 작은 물체는 24px 원형 히트 영역
            if mm.area_m2 / (gsd * gsd) * s * s < 300:
                polys.append(f'<circle cx="{mm.cx_px * s:.1f}" cy="{mm.cy_px * s:.1f}" r="12" class="hit" data-tip="{_esc(tip)}" tabindex="0"/>')
        legend = "".join(
            f'<span class="key"><i style="background:{cls_color(k)}"></i>{_esc(KO_NAMES.get(k, k))} <b>{by[k]["count"]}</b></span>'
            for k in top) + (f'<span class="key"><i style="background:var(--other)"></i>기타 <b>{sum(by[k]["count"] for k in order[7:])}</b></span>' if len(order) > 7 else "")
        overlay_html = f"""
<figure class="card overlay">
  <figcaption><h2>검출 결과</h2><p>정사영상 위 물체 윤곽. 마우스를 올리면 종류·무게·부피가 보입니다.</p></figcaption>
  <div class="ov-wrap"><img src="{src}" alt="정사영상 {W}×{H}px, GSD {gsd * 100:.2f} cm/px" width="{int(W * s)}" height="{int(H * s)}">
  <svg viewBox="0 0 {int(W * s)} {int(H * s)}" preserveAspectRatio="none">{''.join(polys)}</svg></div>
  <div class="legend">{legend}</div>
</figure>"""

    # ---------------- 추정 근거 카드 (물체별) ----------------
    evidence_html = ""
    if surface.ortho is not None:
        vol_by_id = {v.obj_id: v for v in (vols or [])}
        order_obj = sorted(range(len(masses)), key=lambda i: -masses[i].kg_typ)
        cards = []
        for rank, i in enumerate(order_obj):
            mm = masses[i]
            if mm.method == "excluded":
                continue
            cls, m, conf = masks[i]
            y0, y1, x0, x1 = _crop_box(m)
            photo = surface.ortho[y0:y1, x0:x1]
            sub = m[y0:y1, x0:x1]
            col_css = cls_color(cls)
            hexcol = CAT_LIGHT[slot[cls]] if cls in slot else "#9a9890"
            photo_html = _overlay_block(photo, [(sub, col_css, f"{mm.class_ko} 마스크 (신뢰도 {conf:.2f})", None)], max_w=260, stroke=2)
            v = vol_by_id.get(mm.obj_id)
            gz = v.ground_z_m if v else float(np.nanmedian(surface.dsm[y0:y1, x0:x1]))
            hmap = _height_image(surface.dsm[y0:y1, x0:x1].astype(np.float32), gz, max(mm.h_max_m, 0.01))
            height_html = _overlay_block(hmap, [(sub, "#0b0b0b", f"DSM − 바닥, 최대 {mm.h_max_m * 100:.1f} cm", None)], max_w=260, stroke=1)
            spec = CLASSES.get(cls, CLASSES["unknown"])
            if mm.method == "volume":
                formula = (f"V = Σ(DSM − 바닥)·GSD² = <b>{mm.volume_m3 * 1000:.2f} L</b> &nbsp;→&nbsp; "
                           f"m = V × ρ<sub>겉보기</sub>({spec.rho_min}/{spec.rho_typ}/{spec.rho_max} kg/m³) = "
                           f"<b>{_fmt_kg(mm.kg_typ)}</b> ({_fmt_kg(mm.kg_min)}–{_fmt_kg(mm.kg_max)})")
                why = "부피 × 종류별 겉보기 밀도"
            else:
                formula = f"{_esc(mm.note)} → <b>{_fmt_kg(mm.kg_typ)}</b> ({_fmt_kg(mm.kg_min)}–{_fmt_kg(mm.kg_max)})"
                why = "면적 ≤ 25 cm² 또는 높이 ≤ 3 cm → DSM 으로 못 잼" if mm.area_m2 <= 0.0025 or mm.h_max_m <= 0.03 else "속 빈 물체 → 개당 평균무게"
            cards.append(f"""<article class="ev{' more' if rank >= n_evidence else ''}">
  <header><span class="swatch" style="background:{col_css}"></span><b>#{mm.obj_id} {_esc(mm.class_ko)}</b><span class="muted"> 신뢰도 {conf:.2f} · {_esc(why)}</span></header>
  <div class="ev-imgs"><figure>{photo_html}<figcaption>정사영상 + 분할 폴리곤</figcaption></figure>
  <figure>{height_html}<figcaption>DSM 높이 (진할수록 높음, 최대 {mm.h_max_m * 100:.1f} cm)</figcaption></figure></div>
  <dl><dt>면적</dt><dd>{mm.area_m2 * 1e4:,.0f} cm²</dd><dt>최대 높이</dt><dd>{mm.h_max_m * 100:.1f} cm</dd><dt>부피</dt><dd>{mm.volume_m3 * 1000:.2f} L</dd><dt>무게</dt><dd><b>{_esc(_fmt_kg(mm.kg_typ))}</b></dd></dl>
  <p class="formula">{formula}</p>
</article>""")
        n_more = sum(1 for c in cards if 'class="ev more"' in c)
        evidence_html = f"""
<section class="card">
  <h2>추정 근거 (물체별)</h2>
  <p class="sub">각 물체를 어떤 사진 조각과 높이 정보로 판단했고, 어떤 식으로 무게가 나왔는지. 무게 순 상위 {min(n_evidence, len(cards))}개{f', 나머지 {n_more}개는 아래 펼치기' if n_more else ''}.</p>
  <ol class="flow"><li>정사영상</li><li>분할 폴리곤 (YOLO-seg)</li><li>DSM − 바닥 = 높이</li><li>Σ 높이 × GSD² = 부피</li><li>× 종류별 겉보기 밀도</li><li>무게 범위</li></ol>
  <div class="ev-grid">{''.join(cards)}</div>
  {f'<button type="button" class="more-btn" id="ev-more">나머지 {n_more}개 펼치기</button>' if n_more else ''}
</section>"""

    # ---------------- 원본 프레임 검출 갤러리 ----------------
    frames_html = ""
    if frames:
        figs = []
        for name, img, fmasks in frames:
            items = []
            counts = {}
            for cls, m, conf in fmasks:
                ko = KO_NAMES.get(cls, cls)
                counts[ko] = counts.get(ko, 0) + 1
                items.append((m, cls_color(cls), f"{ko} · 신뢰도 {conf:.2f}", f"{ko} {conf:.2f}"))
            summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "검출 없음"
            figs.append(f'<figure class="frame">{_overlay_block(img, items, max_w=900)}<figcaption><b>{_esc(name)}</b> · {len(fmasks)}개 검출 ({_esc(summary)})</figcaption></figure>')
        frames_html = f"""
<section class="card">
  <h2>원본 프레임 검출</h2>
  <p class="sub">드론 영상 프레임에 분할 모델을 돌린 결과. 폴리곤 색은 위 검출 결과와 같은 종류색, 라벨은 종류와 신뢰도. 마우스를 올리면 자세히 보입니다.</p>
  <div class="frames">{''.join(figs)}</div>
</section>"""

    # ---------------- 클래스별 막대 ----------------
    bar_w, row_h, lab_w = 640, 30, 150
    vmax = _nice_max(max((by[k]["kg_max"] for k in order), default=1))
    rows_svg, table_rows = [], []
    for i, k in enumerate(order):
        d = by[k]
        y = i * row_h + 6
        x_typ = lab_w + d["kg_typ"] / vmax * (bar_w - lab_w - 90)
        x_min = lab_w + d["kg_min"] / vmax * (bar_w - lab_w - 90)
        x_max = lab_w + d["kg_max"] / vmax * (bar_w - lab_w - 90)
        tip = f"{KO_NAMES.get(k, k)} · {_fmt_kg(d['kg_typ'])} (범위 {_fmt_kg(d['kg_min'])}–{_fmt_kg(d['kg_max'])}) · {d['count']}개 · {d['volume_m3'] * 1000:.1f} L"
        rows_svg.append(f"""<g class="bar" data-tip="{_esc(tip)}" tabindex="0">
<rect x="{lab_w}" y="{y - 2}" width="{bar_w - lab_w}" height="{row_h - 4}" fill="transparent"/>
<text x="{lab_w - 10}" y="{y + 15}" text-anchor="end" class="lab">{_esc(KO_NAMES.get(k, k))}</text>
<rect x="{lab_w}" y="{y}" width="{max(x_typ - lab_w, 0):.1f}" height="20" rx="0" class="fill"/>
<rect x="{max(x_typ - 4, lab_w):.1f}" y="{y}" width="{4 if x_typ - lab_w >= 4 else 0}" height="20" rx="4" class="fill"/>
<line x1="{x_min:.1f}" x2="{x_max:.1f}" y1="{y + 10}" y2="{y + 10}" class="range"/>
<text x="{x_max + 8:.1f}" y="{y + 15}" class="val">{_esc(_fmt_kg(d['kg_typ']))}</text></g>""")
        table_rows.append(f"<tr><td>{_esc(KO_NAMES.get(k, k))}</td><td>{d['count']}</td><td>{d['volume_m3'] * 1000:.1f}</td>"
                          f"<td>{d['kg_min']:.2f}</td><td><b>{d['kg_typ']:.2f}</b></td><td>{d['kg_max']:.2f}</td></tr>")
    bar_h = len(order) * row_h + 12
    ticks = "".join(f'<g><line x1="{lab_w + t / vmax * (bar_w - lab_w - 90):.1f}" x2="{lab_w + t / vmax * (bar_w - lab_w - 90):.1f}" y1="0" y2="{bar_h}" class="grid"/>'
                    f'<text x="{lab_w + t / vmax * (bar_w - lab_w - 90):.1f}" y="{bar_h + 14}" text-anchor="middle" class="tick">{t:g}</text></g>'
                    for t in np.linspace(0, vmax, 5))
    bars_html = f"""
<figure class="card">
  <figcaption><h2>종류별 무게</h2><p>막대는 대표값, 가는 선은 최소–최대 범위(kg). 밀도표의 가정값 폭이 그대로 범위가 됩니다.</p></figcaption>
  <svg class="bars" viewBox="0 0 {bar_w} {bar_h + 20}" width="100%" height="auto">{ticks}{''.join(rows_svg)}</svg>
  <details><summary>표로 보기</summary><table><thead><tr><th>종류</th><th>개수</th><th>부피 (L)</th><th>최소 (kg)</th><th>대표 (kg)</th><th>최대 (kg)</th></tr></thead>
  <tbody>{''.join(table_rows)}</tbody></table></details>
</figure>"""

    # ---------------- 격자 히트맵 ----------------
    g, meta = grid_kg(litter, gsd, cell, surface.transform)
    gmax = float(g.max()) if g.size else 0.0
    nrow, ncol = g.shape
    cw = max(min(900 // max(ncol, 1), 90), 36)
    prio = {(c["row"], c["col"]): i + 1 for i, c in enumerate(plan.priority_cells[:5])}
    cells, grid_rows = [], []
    for r in range(nrow):
        for c in range(ncol):
            v = float(g[r, c])
            step = 0 if gmax <= 0 or v <= 0 else min(int(v / gmax * 6.999), 6)
            fill = f"var(--s{step})" if v > 0 else "var(--grid)"
            y = (nrow - 1 - r) * cw   # 북쪽(큰 y)이 위
            tip = f"격자 ({r},{c}) · {_fmt_kg(v)} · x {meta['x0'] + c * cell:.0f}–{meta['x0'] + (c + 1) * cell:.0f} m, y {meta['y0'] + r * cell:.0f}–{meta['y0'] + (r + 1) * cell:.0f} m"
            rank = prio.get((r, c))
            ink = "var(--ink-on-fill)" if step >= 3 else "var(--ink)"
            label = f'<text x="{c * cw + cw / 2}" y="{y + cw / 2 + 4}" text-anchor="middle" class="cell-lab" style="fill:{ink}">{_fmt_kg(v).replace(" kg", "").replace(" g", "g")}</text>' if v > 0 and cw >= 44 else ""
            badge = f'<circle cx="{c * cw + cw - 10}" cy="{y + 10}" r="8" class="badge"/><text x="{c * cw + cw - 10}" y="{y + 13}" text-anchor="middle" class="badge-t">{rank}</text>' if rank else ""
            cells.append(f'<g class="cell" data-tip="{_esc(tip)}" tabindex="0"><rect x="{c * cw + 1}" y="{y + 1}" width="{cw - 2}" height="{cw - 2}" fill="{fill}" rx="2"/>{label}{badge}</g>')
            if v > 0:
                grid_rows.append(f"<tr><td>{r}</td><td>{c}</td><td>{v:.2f}</td><td>{rank or ''}</td></tr>")
    scale_html = "".join(f'<i style="background:var(--s{i})"></i>' for i in range(7))
    heat_html = f"""
<figure class="card">
  <figcaption><h2>격자별 무게 지도 ({cell:g} m 격자)</h2><p>진할수록 무겁습니다. 숫자 배지는 우선 수거 순위 1–5. 위쪽이 북쪽(y 증가 방향)입니다.</p></figcaption>
  <div class="scroll"><svg viewBox="0 0 {ncol * cw} {nrow * cw}" width="{ncol * cw}" class="heat">{''.join(cells)}</svg></div>
  <div class="scale"><span>0</span>{scale_html}<span>{_fmt_kg(gmax)}</span></div>
  <details><summary>표로 보기</summary><table><thead><tr><th>행</th><th>열</th><th>kg</th><th>순위</th></tr></thead><tbody>{''.join(grid_rows)}</tbody></table></details>
</figure>"""

    # ---------------- 방식 비교 ----------------
    comp = [("우리 3D · 종류별 겉보기 밀도", tk[1], True), ("W2 · 개수 × 14 g", w2, False),
            ("W3 · 면적 × 0.4 cm × 1.2 g/cm³", w3, False), ("W3′ · 겉부피 × 재료 비중 1.2", w3d, False)]
    cmax = _nice_max(max([c[1] for c in comp] + ([truth_kg] if truth_kg else [])))
    cw_, cl_ = 640, 260
    comp_svg = []
    for i, (name, v, ours) in enumerate(comp):
        y = i * 34 + 6
        x = cl_ + v / cmax * (cw_ - cl_ - 90)
        err = f" ({(v / truth_kg - 1) * 100:+.0f} %)" if truth_kg else ""
        comp_svg.append(f"""<g class="bar" data-tip="{_esc(name + ' · ' + _fmt_kg(v) + err)}" tabindex="0">
<rect x="{cl_}" y="{y - 2}" width="{cw_ - cl_}" height="30" fill="transparent"/>
<text x="{cl_ - 10}" y="{y + 15}" text-anchor="end" class="lab">{_esc(name)}</text>
<rect x="{cl_}" y="{y}" width="{max(x - cl_, 0):.1f}" height="20" class="{'fill' if ours else 'fill-gray'}"/>
<rect x="{max(x - 4, cl_):.1f}" y="{y}" width="{4 if x - cl_ >= 4 else 0}" height="20" rx="4" class="{'fill' if ours else 'fill-gray'}"/>
<text x="{x + 8:.1f}" y="{y + 15}" class="val">{_esc(_fmt_kg(v) + err)}</text></g>""")
    ch = len(comp) * 34 + 12
    truth_line = ""
    if truth_kg:
        xt = cl_ + truth_kg / cmax * (cw_ - cl_ - 90)
        truth_line = f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="0" y2="{ch}" class="truth"/><text x="{xt + 4:.1f}" y="{ch + 14}" class="tick">실측 {_fmt_kg(truth_kg)}</text>'
    comp_html = f"""
<figure class="card">
  <figcaption><h2>기존 방식과 비교</h2><p>같은 검출 결과에 세 가지 기존 공식을 적용한 값. {'세로선이 실측 무게입니다.' if truth_kg else '실측 무게가 있으면 세로선으로 표시됩니다.'} W3′는 속 빈 물체를 꽉 찬 것으로 계산해 과대추정됩니다.</p></figcaption>
  <svg viewBox="0 0 {cw_} {ch + 20}" width="100%" height="auto" class="bars">{''.join(comp_svg)}{truth_line}</svg>
</figure>"""

    # ---------------- 수거 계획 ----------------
    p = plan
    cls_rows = "".join(f"<tr><td>{_esc(c.class_ko)}</td><td>{c.count}</td><td>{c.kg_typ:.1f}</td><td>{c.volume_m3 * 1000:.0f}</td>"
                       f"<td>{c.bags}</td><td>{'부피' if c.limiting == 'volume' else '무게'}</td><td>{c.heavy_items}</td><td>{c.worker_hours:.1f}</td></tr>"
                       for c in p.by_class)
    heavy_rows = "".join(f'<li><span class="chip serious">⚠ {_esc(h["class"])} {h["kg_typ"]:.1f} kg</span> 위치 x {h["x_m"]:.0f} m, y {h["y_m"]:.0f} m · {_esc(h["handling"])}</li>'
                         for h in p.heavy_items) or "<li>23 kg 초과 물체 없음</li>"
    route_items = []
    for r in p.routes:
        stops = " → ".join(f"({st['row']},{st['col']})" for st in r["stops"][:8]) + (" …" if len(r["stops"]) > 8 else "")
        route_items.append(f"<li>차량 {r['vehicle']}: 격자 {len(r['stops'])}곳, 적재 {r['load_kg']:.0f} kg, 이동 {r['length_m']:.0f} m "
                           f"<span class='muted'>({_esc(stops)})</span></li>")
    route_rows = "".join(route_items) or "<li>경로 없음</li>"
    plan_html = f"""
<section class="card">
  <h2>수거 계획</h2><p class="sub">적재량·작업속도는 가정값입니다 (현장 기준으로 교체). 1인 들기 한계 23 kg 는 NIOSH 기준.</p>
  <div class="scroll"><table><thead><tr><th>종류</th><th>개수</th><th>kg</th><th>부피 (L)</th><th>마대</th><th>먼저 차는 쪽</th><th>23 kg 초과</th><th>인·시간</th></tr></thead><tbody>{cls_rows}</tbody></table></div>
  <h3>무거운 물체 (2인 이상 또는 장비)</h3><ul class="list">{heavy_rows}</ul>
  <h3>차량 경로 ({p.routes[0]['solver'] if p.routes else '-'})</h3><ul class="list">{route_rows}</ul>
</section>"""

    # ---------------- 물체 표 ----------------
    obj_rows = []
    for m in sorted(litter, key=lambda m: -m.kg_typ):
        x, y = object_xy_m(m, gsd, surface.transform)
        obj_rows.append(f"<tr><td>{m.obj_id}</td><td>{_esc(m.class_ko)}</td><td>{'부피' if m.method == 'volume' else '개수/면적'}</td>"
                        f"<td>{m.volume_m3 * 1000:.2f}</td><td>{m.area_m2 * 1e4:.0f}</td><td>{m.h_max_m * 100:.1f}</td>"
                        f"<td>{m.kg_min:.3f}</td><td><b>{m.kg_typ:.3f}</b></td><td>{m.kg_max:.3f}</td><td>{x:.1f}, {y:.1f}</td></tr>")
    objects_html = f"""
<section class="card">
  <details><summary><h2 class="inline">물체 목록 ({len(litter)}개)</h2></summary>
  <div class="scroll"><table><thead><tr><th>ID</th><th>종류</th><th>방식</th><th>부피 (L)</th><th>면적 (cm²)</th><th>높이 (cm)</th><th>최소 (kg)</th><th>대표 (kg)</th><th>최대 (kg)</th><th>위치 (m)</th></tr></thead>
  <tbody>{''.join(obj_rows)}</tbody></table></div></details>
</section>"""

    assumptions = "".join(f"<li>{_esc(a)}</li>" for a in p.assumptions)
    truth_tile = f'<div class="tile"><span class="lab">실측 대비</span><span class="val">{(tk[1] / truth_kg - 1) * 100:+.0f} %</span><span class="sub">실측 {_fmt_kg(truth_kg)}</span></div>' if truth_kg else ""

    css = """
<style>
/* 레이아웃: 한 열, 요약 → 검출 → 무게 → 지도 → 계획. 색: 파랑 한 색의 진하기 = 무게, 8색 고정 = 종류 */
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--other:#9a9890;--serious:#ec835a;--ink-on-fill:#ffffff;
--c0:#2a78d6;--c1:#eb6834;--c2:#1baf7a;--c3:#eda100;--c4:#e87ba4;--c5:#008300;--c6:#4a3aa7;--c7:#e34948;
--s0:#cde2fb;--s1:#9ec5f4;--s2:#6da7ec;--s3:#3987e5;--s4:#256abf;--s5:#184f95;--s6:#0d366b;
--font:"Noto Sans KR",system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--accent:#3987e5;--other:#6f6e69;--ink-on-fill:#ffffff;
--c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#c98500;--c4:#d55181;--c5:#008300;--c6:#9085e9;--c7:#e66767;
--s0:#104281;--s1:#184f95;--s2:#1c5cab;--s3:#256abf;--s4:#3987e5;--s5:#6da7ec;--s6:#9ec5f4;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--accent:#3987e5;--other:#6f6e69;--ink-on-fill:#ffffff;
--c0:#3987e5;--c1:#d95926;--c2:#199e70;--c3:#c98500;--c4:#d55181;--c5:#008300;--c6:#9085e9;--c7:#e66767;
--s0:#104281;--s1:#184f95;--s2:#1c5cab;--s3:#256abf;--s4:#3987e5;--s5:#6da7ec;--s6:#9ec5f4;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:14px;line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding-inline:16px;padding-block:24px 48px;display:grid;gap:20px}
header h1{font-size:22px;margin:0 0 4px;font-weight:600;text-wrap:balance}header p{margin:0;color:var(--ink2)}
.hero{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,2fr);gap:20px}
@media (max-width:720px){.hero{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;min-width:0}
figure.card{margin:0}figcaption h2,.card h2{font-size:16px;margin:0 0 2px;font-weight:600}figcaption p,.sub{margin:0 0 12px;color:var(--ink2);font-size:13px}
h3{font-size:14px;margin:16px 0 6px;font-weight:600}h2.inline{display:inline}
.big .lab{display:block;color:var(--ink2);font-size:13px}.big .num{font-size:52px;font-weight:600;line-height:1.05;letter-spacing:-.01em}
.big .range{color:var(--ink2);margin-top:6px}.big .foot{color:var(--muted);font-size:12px;margin-top:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;display:grid;gap:2px;min-width:0}
.tile .lab{color:var(--ink2);font-size:12px}.tile .val{font-size:24px;font-weight:600;line-height:1.1}.tile .sub{font-size:12px;color:var(--muted);margin:0}
.ov-wrap{position:relative;display:inline-block;max-width:100%;vertical-align:top;border-radius:6px;overflow:hidden}.ov-wrap img{display:block;max-width:100%;height:auto}
.ov-wrap svg{position:absolute;inset:0;width:100%;height:100%}
.ov-wrap polygon{fill:rgba(255,255,255,.001);stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer;outline:none}
.ov-wrap polygon:hover,.ov-wrap polygon:focus{fill:rgba(255,255,255,.25);stroke-width:3}
.ov-wrap .hit{fill:transparent;stroke:none;cursor:pointer}
.legend,.scale{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:10px;font-size:13px;color:var(--ink2);align-items:center}
.key i,.scale i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}.scale i{margin:0;width:22px;border-radius:2px}
.key b{color:var(--ink);font-weight:600}
.scroll{overflow-x:auto;max-width:100%}
svg.bars{display:block;width:100%;max-width:640px;height:auto}svg.bars text,svg.heat text{font-family:var(--font)}svg .lab{font-size:13px;fill:var(--ink)}svg .val{font-size:12px;fill:var(--ink2);font-variant-numeric:tabular-nums}
svg .tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}svg .grid{stroke:var(--grid);stroke-width:1}
svg .fill{fill:var(--accent)}svg .fill-gray{fill:var(--other)}svg .range{stroke:var(--ink2);stroke-width:1.5}svg .truth{stroke:var(--ink);stroke-width:1.5}
svg .bar{cursor:pointer;outline:none}svg .bar:hover .fill,svg .bar:focus .fill{opacity:.8}svg .bar:hover .fill-gray,svg .bar:focus .fill-gray{opacity:.8}
svg.heat .cell{cursor:pointer;outline:none}svg.heat .cell:hover rect,svg.heat .cell:focus rect{stroke:var(--ink);stroke-width:1.5}
svg.heat .cell-lab{font-size:11px;font-variant-numeric:tabular-nums;pointer-events:none}
svg.heat .badge{fill:var(--serious)}svg.heat .badge-t{font-size:10px;font-weight:700;fill:#fff;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
details summary{cursor:pointer;color:var(--ink2);margin-top:8px}details[open] summary{margin-bottom:8px}
.list{margin:0;padding-left:18px}.list li{margin:4px 0}.muted{color:var(--muted)}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--border)}
.chip.serious{background:var(--serious);color:#1a1a19}
.ov-lab{font-size:12px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:3px;pointer-events:none}
.flow{display:flex;flex-wrap:wrap;gap:6px;list-style:none;padding:0;margin:0 0 14px;font-size:12px;color:var(--ink2)}
.flow li{background:var(--bg);border:1px solid var(--border);border-radius:999px;padding:2px 10px}.flow li+li::before{content:"→ ";color:var(--muted)}
.ev-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.ev{border:1px solid var(--border);border-radius:8px;padding:12px;min-width:0;display:grid;gap:8px;align-content:start}
.ev.more{display:none}.ev-grid.open .ev.more{display:grid}
.ev header{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:13px}.swatch{display:inline-block;width:12px;height:12px;border-radius:3px}
.ev-imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px}.ev-imgs figure{margin:0;min-width:0}.ev-imgs figcaption{font-size:11px;color:var(--muted);margin-top:4px}
.ev-imgs .ov-wrap{display:block}.ev-imgs .ov-wrap img{width:100%;height:auto}
.ev dl{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:2px 8px;margin:0;font-size:12px;font-variant-numeric:tabular-nums}.ev dt{color:var(--muted)}.ev dd{margin:0}
.formula{margin:0;font-size:12px;color:var(--ink2);line-height:1.6}
.more-btn{margin-top:12px;background:var(--surface);color:var(--ink);border:1px solid var(--axis);border-radius:6px;padding:6px 14px;cursor:pointer;font:inherit}
.frames{display:grid;gap:14px}.frame{margin:0}.frame figcaption{font-size:13px;color:var(--ink2);margin-top:6px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 10px;border-radius:6px;font-size:12px;max-width:320px;z-index:9;display:none;box-shadow:0 2px 8px rgba(0,0,0,.25)}
footer{color:var(--muted);font-size:12px}footer ul{margin:4px 0 0;padding-left:18px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:no-preference){svg .fill,svg .fill-gray{transition:opacity .15s}}
</style>"""
    js = """
<script>
(function(){var t=document.getElementById('tip');
function show(e){var el=e.currentTarget;t.textContent=el.getAttribute('data-tip');t.style.display='block';move(e);}
function move(e){var x=(e.clientX||0)+14,y=(e.clientY||0)+14;if(x+t.offsetWidth>innerWidth-8)x=innerWidth-t.offsetWidth-8;if(y+t.offsetHeight>innerHeight-8)y=y-t.offsetHeight-28;t.style.left=x+'px';t.style.top=y+'px';}
function hide(){t.style.display='none';}
document.querySelectorAll('[data-tip]').forEach(function(el){el.addEventListener('pointerenter',show);el.addEventListener('pointermove',move);el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(e){var r=el.getBoundingClientRect();t.textContent=el.getAttribute('data-tip');t.style.display='block';t.style.left=(r.left+8)+'px';t.style.top=(r.bottom+6)+'px';});el.addEventListener('blur',hide);});
var b=document.getElementById('ev-more');if(b){b.addEventListener('click',function(){document.querySelector('.ev-grid').classList.add('open');b.hidden=true;});}})();
</script>"""
    page = f"""<title>{_esc(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600&display=swap">
{css}
<div class="wrap">
<header><h1>{_esc(title)}</h1><p>{_esc(site) + ' · ' if site else ''}드론 영상 → 종류별 분할 → 3D 부피 → 겉보기 밀도 → 수거 계획 · GSD {gsd * 100:.2f} cm/px</p></header>
<div class="hero">
  <div class="card big"><span class="lab">추정 총 무게 (대표값)</span><span class="num">{_esc(_fmt_kg(tk[1]))}</span>
    <div class="range">범위 {_esc(_fmt_kg(tk[0]))} – {_esc(_fmt_kg(tk[2]))}</div>
    <div class="foot">검출 누락은 반영하지 않은 최소 추정치 · 부피 기반 {n_vol}개, 개수/면적 기반 {n_cnt}개</div></div>
  <div class="tiles">
    <div class="tile"><span class="lab">검출 물체</span><span class="val">{len(litter)}</span><span class="sub">식생 제외 {len(masses) - len(litter)}개</span></div>
    <div class="tile"><span class="lab">총 겉 부피 (L)</span><span class="val">{vol * 1000:,.0f}</span><span class="sub">{vol:.3f} m³</span></div>
    <div class="tile"><span class="lab">마대</span><span class="val">{p.total_bags}</span><span class="sub">톤백 {p.tonbags} · 1 t 트럭 {p.truck_trips}회</span></div>
    <div class="tile"><span class="lab">작업량 (인·시간)</span><span class="val">{p.worker_hours:.1f}</span><span class="sub">4시간 기준 {p.workers_for_4h}명</span></div>
    <div class="tile"><span class="lab">23 kg 초과</span><span class="val">{len(p.heavy_items)}</span><span class="sub">2인 이상 또는 장비</span></div>
    {truth_tile}
  </div>
</div>
{overlay_html}
{frames_html}
{evidence_html}
{bars_html}
{comp_html}
{heat_html}
{plan_html}
{objects_html}
<footer class="card"><b>가정과 한계</b><ul>{assumptions}<li>겉보기 밀도표에서 출처가 있는 값은 스티로폼(EPS 11–32 kg/m³)·개당 평균무게(Andriolo 2024)·NIOSH 23 kg 뿐이며 나머지는 가정값</li></ul></footer>
</div>
<div id="tip" role="tooltip"></div>
{js}
"""
    out_path = Path(out_path)
    out_path.write_text(page, encoding="utf-8")
    return out_path
