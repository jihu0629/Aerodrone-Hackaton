"""수거 계획 → 작업자용 지도·작업 지시서 (인터랙티브 HTML + 인쇄용 PNG).

  build_collect_html()  한 장짜리 HTML 앱. 인원·시간·종류·이동/운반 방식·무게 기준·출발지(별 끌기) 를 바꾸면
                        브라우저 안에서 지형 최단경로(다익스트라)·구역·순서·마대·시간을 즉시 다시 계산한다.
                        지도는 Leaflet(인터넷) 위성 + 드론 정사영상 오버레이, 인터넷이 없으면 정적 PNG 로 대체.
  draw_static_map()     인쇄용 PNG 지도 (matplotlib). 기본 파라미터 계획(파이썬) 기준.
  Basemap               정사영상 축소본 + 월드파일 (위경도 재투영, 지형 분류 입력)
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .collect import CollectPlan, material, materials_js

DAY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#eda100", "#e34948", "#0097a7", "#6d4c41"]


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _b64_file(path: Path, max_px: int | None = None) -> str | None:
    try:
        if max_px:
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            s = min(1.0, max_px / max(h, w))
            if s < 1:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        data = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except OSError:
        return None


# ─────────────────────────── 정사영상 축소본 ───────────────────────────
class Basemap:
    """정사영상 축소본 + 월드파일. world = {x0, y0, px, py} (왼쪽 위 모서리, px>0, py<0), full_px = 원본 (W, H)."""

    def __init__(self, image_path: str | Path, world: dict, full_px: tuple[int, int], crs_m: str = "EPSG:5186"):
        self.path = Path(image_path)
        self.img = cv2.imdecode(np.fromfile(str(self.path), np.uint8), cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(image_path)
        self.world, self.full_px, self.crs_m = world, full_px, crs_m
        self.sx = self.img.shape[1] / full_px[0]
        self.sy = self.img.shape[0] / full_px[1]

    @classmethod
    def from_array(cls, img: np.ndarray, world: dict, full_px: tuple[int, int], crs_m: str = "EPSG:5186") -> "Basemap":
        self = cls.__new__(cls)
        self.path = None; self.img = img; self.world = world; self.full_px = full_px; self.crs_m = crs_m
        self.sx = img.shape[1] / full_px[0]; self.sy = img.shape[0] / full_px[1]
        return self

    @property
    def extent_m(self) -> tuple[float, float, float, float]:
        w = self.world
        return (w["x0"], w["x0"] + self.full_px[0] * w["px"], w["y0"] + self.full_px[1] * w["py"], w["y0"])

    def nodata_mask(self, img: np.ndarray | None = None) -> np.ndarray:
        img = self.img if img is None else img
        m = (img.sum(axis=2) < 24).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)

    def to_wgs84(self, out_w: int = 2400, quality: int = 78, nodata_fn=None) -> tuple[str, tuple[float, float, float, float]]:
        """위경도 격자로 재투영 → (data URL(webp, 투명 nodata), (south, west, north, east))."""
        from pyproj import Transformer
        to_ll = Transformer.from_crs(self.crs_m, "EPSG:4326", always_xy=True)
        to_m = Transformer.from_crs("EPSG:4326", self.crs_m, always_xy=True)
        xmin, xmax, ymin, ymax = self.extent_m
        cs = [to_ll.transform(x, y) for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))]
        west, east = min(c[0] for c in cs), max(c[0] for c in cs)
        south, north = min(c[1] for c in cs), max(c[1] for c in cs)
        lat0 = math.radians((south + north) / 2)
        out_h = int(out_w * (north - south) / ((east - west) * math.cos(lat0)))
        lons = west + (np.arange(out_w) + 0.5) / out_w * (east - west)
        lats = north - (np.arange(out_h) + 0.5) / out_h * (north - south)
        LON, LAT = np.meshgrid(lons, lats)
        X, Y = to_m.transform(LON, LAT)
        w = self.world
        mapx = ((X - w["x0"]) / w["px"] * self.sx).astype(np.float32)
        mapy = ((Y - w["y0"]) / w["py"] * self.sy).astype(np.float32)
        interp = cv2.INTER_NEAREST if nodata_fn is not None else cv2.INTER_LINEAR
        warped = cv2.remap(self.img, mapx, mapy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        nd = nodata_fn(warped) if nodata_fn is not None else self.nodata_mask(warped)
        alpha = np.where(nd, 0, 255).astype(np.uint8)
        bgra = cv2.merge([warped[:, :, 0], warped[:, :, 1], warped[:, :, 2], alpha])
        ok, buf = cv2.imencode(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not ok:
            ok, buf = cv2.imencode(".png", bgra)
            return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), (south, west, north, east)
        return "data:image/webp;base64," + base64.b64encode(buf.tobytes()).decode(), (south, west, north, east)


def fit_affine(crs_m: str, lon_range: tuple[float, float], lat_range: tuple[float, float]) -> dict:
    """위경도 ↔ 투영좌표(m) 를 작은 영역에서 아핀으로 근사 (브라우저 계산용). 반환 {fwd, inv, max_err_m}."""
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", crs_m, always_xy=True)
    lons = np.linspace(lon_range[0], lon_range[1], 12); lats = np.linspace(lat_range[0], lat_range[1], 12)
    LON, LAT = np.meshgrid(lons, lats); X, Y = to_m.transform(LON, LAT)
    A = np.c_[LON.ravel(), LAT.ravel(), np.ones(LON.size)]
    cx = np.linalg.lstsq(A, X.ravel(), rcond=None)[0]
    cy = np.linalg.lstsq(A, Y.ravel(), rcond=None)[0]
    err = float(np.max(np.hypot(A @ cx - X.ravel(), A @ cy - Y.ravel())))
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]]); Mi = np.linalg.inv(M)
    return {"fwd": [float(v) for v in (cx[0], cx[1], cx[2], cy[0], cy[1], cy[2])],
            "inv": [float(v) for v in (Mi[0, 0], Mi[0, 1], Mi[1, 0], Mi[1, 1])], "max_err_m": err}


# ─────────────────────────── 정적 PNG 지도 ───────────────────────────
def _korean_font():
    import matplotlib
    from matplotlib import font_manager
    for f in [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_static_map(plan: CollectPlan, out_png: str | Path, basemap: Basemap | None = None, dpi: int = 150,
                    title: str | None = None, crs_m: str = "EPSG:5186") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from pyproj import Transformer
    _korean_font()
    to_m = Transformer.from_crs("EPSG:4326", crs_m, always_xy=True)

    objs = [o for o in plan.objects if o.included]
    xs = [o.x_m for o in plan.objects] + [plan.depot["x_m"]]; ys = [o.y_m for o in plan.objects] + [plan.depot["y_m"]]
    pad = max((max(xs) - min(xs)), (max(ys) - min(ys))) * 0.06 + 50
    xmin, xmax, ymin, ymax = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    if basemap is not None:
        e = basemap.extent_m
        xmin, xmax, ymin, ymax = min(xmin, e[0]), max(xmax, e[1]), min(ymin, e[2]), max(ymax, e[3])
    aspect = (ymax - ymin) / (xmax - xmin)
    fig = plt.figure(figsize=(17, max(9, 12.5 * aspect + 1.2)), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1], wspace=0.02, left=0.02, right=0.99, top=0.93, bottom=0.03)
    ax = fig.add_subplot(gs[0]); side = fig.add_subplot(gs[1]); side.axis("off")
    if basemap is not None:
        img = cv2.cvtColor(basemap.img, cv2.COLOR_BGR2RGB).copy()
        img[basemap.nodata_mask()] = (236, 240, 244)
        e = basemap.extent_m
        ax.imshow(img, extent=(e[0], e[1], e[2], e[3]), interpolation="bilinear", zorder=0)
    ax.set_facecolor("#eceff3"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c9ced6")

    by_id = {o.obj_id: o for o in plan.objects}
    for z in plan.zones:
        col = DAY_COLORS[(z.day - 1) % len(DAY_COLORS)]
        pts = [to_m.transform(lo, la) for lo, la in z.path_lonlat]
        if not pts:
            continue
        first = by_id[z.objects[0]]
        k = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - first.x_m, pts[i][1] - first.y_m))
        for seg, ls in ((pts[:k + 1], (0, (5, 3))), (pts[k:], "solid")):
            if len(seg) > 1:
                ax.plot([p[0] for p in seg], [p[1] for p in seg], color="white", lw=5, solid_capstyle="round", zorder=2, alpha=0.9)
                ax.plot([p[0] for p in seg], [p[1] for p in seg], color=col, lw=2.4, ls=ls, solid_capstyle="round", zorder=3)
    if plan.params.get("round_trip") and plan.zones and plan.route_lonlat:
        last = by_id[plan.zones[-1].objects[-1]]
        pts = [to_m.transform(lo, la) for lo, la in plan.route_lonlat]
        near = [i for i in range(len(pts)) if math.hypot(pts[i][0] - last.x_m, pts[i][1] - last.y_m) < 1.0]
        back = pts[max(near):] if near else []
        if len(back) > 1:
            ax.plot([p[0] for p in back], [p[1] for p in back], color="white", lw=5, zorder=2, alpha=0.9)
            ax.plot([p[0] for p in back], [p[1] for p in back], color=DAY_COLORS[(plan.zones[-1].day - 1) % len(DAY_COLORS)],
                    lw=2.4, ls=(0, (4, 3)), zorder=3)
    for o in plan.objects:
        if not o.included:
            ax.scatter(o.x_m, o.y_m, s=26, c="#bbb", edgecolors="#666", linewidths=0.6, zorder=4)
    for o in objs:
        ax.scatter(o.x_m, o.y_m, s=90 if o.heavy else 46, c=material(o.code).color, edgecolors="black", linewidths=0.8,
                   zorder=5, marker="D" if o.heavy else "o")
    for z in plan.zones:
        col = DAY_COLORS[(z.day - 1) % len(DAY_COLORS)]
        ax.scatter(z.cx_m, z.cy_m, s=560, c="white", edgecolors=col, linewidths=2.6, zorder=6)
        ax.text(z.cx_m, z.cy_m, str(z.step), ha="center", va="center", fontsize=12, fontweight="bold", color=col, zorder=7)
    ax.scatter(plan.depot["x_m"], plan.depot["y_m"], s=700, marker="*", c="#ffffff", edgecolors="#111", linewidths=1.5, zorder=8)
    ax.annotate(plan.depot["name"], (plan.depot["x_m"], plan.depot["y_m"]), xytext=(14, -4), textcoords="offset points",
                fontsize=11, fontweight="bold", zorder=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333", lw=0.8))
    span = xmax - xmin
    sb = 10 ** math.floor(math.log10(span / 5)); sb = sb * (5 if span / 5 / sb >= 5 else 2 if span / 5 / sb >= 2 else 1)
    x0s, y0s = xmin + span * 0.04, ymin + (ymax - ymin) * 0.04
    ax.plot([x0s, x0s + sb], [y0s, y0s], color="black", lw=4, zorder=9)
    ax.text(x0s + sb / 2, y0s + (ymax - ymin) * 0.012, f"{sb:g} m", ha="center", fontsize=11, fontweight="bold", zorder=9)
    ax.annotate("N", xy=(xmax - span * 0.05, ymax - (ymax - ymin) * 0.04), xytext=(xmax - span * 0.05, ymax - (ymax - ymin) * 0.11),
                ha="center", fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=2, color="black"), zorder=9)
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=material(c).color, mec="black", label=f"{d['ko']} {d['count']}개")
               for c, d in sorted(plan.by_code.items(), key=lambda kv: -kv[1]["count"])]
    handles.append(Line2D([], [], marker="D", ls="", ms=9, mfc="white", mec="black", label="2인 운반(무거움)"))
    for d in sorted({z.day for z in plan.zones}):
        handles.append(Line2D([], [], color=DAY_COLORS[(d - 1) % len(DAY_COLORS)], lw=3, label=f"{d}일차 경로"))
    handles.append(Line2D([], [], color="#555", lw=2, ls=(0, (5, 3)), label="구역 사이 이동" + (" (지형 최단경로)" if plan.terrain_used else " (직선)")))
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.95, title="범례", title_fontsize=10)
    t = plan.totals
    fig.suptitle(title or f"{plan.site} 해안쓰레기 수거 작업 지도", fontsize=20, fontweight="bold", x=0.02, ha="left", y=0.985)
    ax.set_title(f"조사일 {plan.survey_date} · 쓰레기 {plan.n_objects}개 · 구역 {t['zones']}곳 · 예상 {t['kg_plan']:.0f} kg "
                 f"(범위 {t['kg_min']:.0f}–{t['kg_max']:.0f}) · 마대 {t['bags']}장 · 이동 {plan.route_len_m / 1000:.1f} km · "
                 f"총 {plan.total_min / 60:.1f}시간({plan.params['workers']}명) · {len(plan.days)}일 · "
                 f"{'지형 반영' if plan.terrain_used else '직선×우회'}", fontsize=11, loc="left", color="#333")
    lines = [("순서", "구역", "개수", "kg", "마대", "분")]
    for z in plan.zones:
        lines.append((str(z.step), z.name + ("  [2인]" if z.heavy_ids else ""), str(z.n), f"{z.kg_plan:.1f}", str(z.bags), f"{z.walk_min + z.work_min:.0f}"))
    side.set_xlim(0, 1); side.set_ylim(0, 1)
    y = 0.98
    side.text(0, y, "작업 순서 (출발지 → 번호 순)", fontsize=13, fontweight="bold", va="top"); y -= 0.045
    colx = [0.0, 0.11, 0.62, 0.72, 0.84, 0.93]
    rowh = min(0.034, 0.9 / (len(lines) + 2))
    for i, row in enumerate(lines):
        bold = i == 0
        if i > 0:
            z = plan.zones[i - 1]; col = DAY_COLORS[(z.day - 1) % len(DAY_COLORS)]
            side.add_patch(FancyBboxPatch((0, y - rowh * 0.8), 0.085, rowh * 0.72, boxstyle="round,pad=0.002", fc=col, ec="none"))
            side.text(0.042, y - rowh * 0.44, row[0], color="white", fontsize=9.5, fontweight="bold", ha="center", va="center")
        for cx, txt in zip(colx[1:] if i > 0 else colx, row[1:] if i > 0 else row):
            side.text(cx, y - rowh * 0.44, txt, fontsize=9.5 if not bold else 10, fontweight="bold" if bold else "normal", va="center",
                      ha="left" if cx < 0.7 else "right" if cx > 0.9 else "center")
        y -= rowh
    y -= 0.02
    side.text(0, y, "준비물", fontsize=12, fontweight="bold", va="top"); y -= 0.035
    for e in plan.equipment:
        side.text(0.01, y, "[  ] " + e, fontsize=9.5, va="top", wrap=True); y -= 0.03
    y -= 0.01
    side.text(0, y, "※ 무게·시간은 라벨 면적 기반 추정(가정값 포함).\n   현장에서 라벨에 없는 쓰레기도 함께 수거.", fontsize=8.5, va="top", color="#555")
    out = Path(out_png); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white"); plt.close(fig)
    return out


# ─────────────────────────── HTML 앱 ───────────────────────────
CSS = r"""
:root{--bg:#f4f6f9;--card:#fff;--ink:#17202a;--ink2:#4a5563;--muted:#7a8494;--border:#dfe3e9;--accent:#2a78d6;--warn:#c0392b;--warnbg:#fdecea;--ok:#1baf7a;
--font:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1419;--card:#1a2027;--ink:#eef1f5;--ink2:#b8c0cb;--muted:#8691a0;--border:#2b343f;--warnbg:#3a1d1a}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:16px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
header{padding:8px 0 4px}header h1{font-size:28px;margin:0 0 4px;font-weight:800;letter-spacing:-.01em}header p{margin:0;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 16px}
.tile .lab{font-size:13px;color:var(--ink2)}.tile .val{font-size:27px;font-weight:800;line-height:1.15;letter-spacing:-.01em;white-space:nowrap}.tile .sub{font-size:12px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px;margin:14px 0}
.card h2{font-size:20px;margin:0 0 4px;font-weight:800}.card .sub{color:var(--ink2);font-size:14px;margin:0 0 10px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:12px 16px;margin:12px 0}
.layout{display:block}.side{display:block}.main{min-width:0}
.fold{font:inherit;font-size:13px;padding:4px 10px;border-radius:999px;border:1px solid var(--border);background:var(--bg);color:var(--ink2);cursor:pointer;margin-left:8px}
.side-open{display:none;font:inherit;font-size:14px;padding:8px 10px;border-radius:12px;border:1px solid var(--border);background:var(--card);color:var(--ink);cursor:pointer;writing-mode:vertical-rl;letter-spacing:.1em}
@media (min-width:1000px){.wrap{max-width:1500px}.layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:16px;align-items:start}
 .side{position:sticky;top:env(safe-area-inset-top,0px);max-height:100vh;overflow:auto;padding-bottom:12px}.side .panel{margin:12px 0 0}
 .side .ctrl{grid-template-columns:1fr 1fr}.side .codes label{padding:3px 8px;font-size:13px}.side .btns button{font-size:13px;padding:5px 9px}
 .layout.collapsed{grid-template-columns:48px minmax(0,1fr)}.layout.collapsed .panel{display:none}.layout.collapsed .side-open{display:block;margin-top:12px}}
.panel h2{font-size:17px;margin:0 0 8px;font-weight:800;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px}
.ctrl{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px 14px}
.ctrl label{display:flex;flex-direction:column;font-size:13px;color:var(--ink2);gap:3px}
.ctrl label b{color:var(--ink);font-size:14px}
.ctrl input[type=number],.ctrl select{font:inherit;font-size:16px;padding:6px 8px;border:1px solid var(--border);border-radius:9px;background:var(--bg);color:var(--ink);width:100%}
.seg{display:flex;border:1px solid var(--border);border-radius:9px;overflow:hidden}.seg button{flex:1;font:inherit;font-size:14px;padding:6px 4px;border:0;background:var(--bg);color:var(--ink2);cursor:pointer}
.seg button.on{background:var(--accent);color:#fff;font-weight:700}
.codes{display:flex;flex-wrap:wrap;gap:6px}.codes label{flex-direction:row;align-items:center;gap:5px;font-size:14px;color:var(--ink);padding:4px 10px;border:1px solid var(--border);border-radius:999px;cursor:pointer}
.codes label i{width:12px;height:12px;border-radius:50%;display:inline-block;border:1px solid #2225}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.btns button,.mapbtns button{font:inherit;font-size:14px;padding:6px 12px;border-radius:999px;border:1px solid var(--border);background:var(--card);color:var(--ink);cursor:pointer}
.btns button:hover,.mapbtns button:hover{border-color:var(--accent);color:var(--accent)}.btns button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
details.adv summary{cursor:pointer;font-size:14px;color:var(--ink2);padding:6px 0}
#status{font-size:13px;color:var(--muted);font-weight:400}
#map{height:560px;border-radius:12px;background:#dfe6ee;position:relative;overflow:hidden}
#map img.fallback{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:10px;font-size:14px;align-items:center}
.dot{display:inline-block;width:14px;height:14px;border-radius:50%;border:1.5px solid #222;vertical-align:-2px;margin-right:4px}
.dia{display:inline-block;width:12px;height:12px;border:1.5px solid #222;background:#fff;transform:rotate(45deg);margin:0 6px 0 2px}
.mapbtns{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.day{display:flex;align-items:center;gap:10px;margin:18px 0 8px}.day h3{margin:0;font-size:18px}.day .pill{font-size:13px;padding:3px 10px;border-radius:999px;color:#fff;font-weight:700}
.steps{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.step{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px;cursor:pointer;transition:box-shadow .15s;border-left:6px solid var(--accent)}
.step:hover{box-shadow:0 4px 18px rgba(0,0,0,.08)}.step.active{outline:2px solid var(--accent)}
.step .head{display:flex;align-items:center;gap:10px}.stepnum{width:38px;height:38px;border-radius:50%;background:var(--accent);color:#fff;font-weight:800;font-size:19px;display:flex;align-items:center;justify-content:center;flex:none}
.step .name{font-size:18px;font-weight:800}.step .meta{font-size:13px;color:var(--ink2)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.chip{font-size:13px;padding:2px 9px;border-radius:999px;border:1px solid #2223;background:#fff;color:#111;font-weight:600}
.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:8px 0}.fact{background:var(--bg);border-radius:10px;padding:6px 8px;text-align:center}
.fact b{display:block;font-size:19px;line-height:1.2}.fact span{font-size:11.5px;color:var(--ink2)}
.warn{background:var(--warnbg);color:var(--warn);border-radius:10px;padding:8px 10px;font-size:14px;font-weight:600;margin:6px 0}
.photos{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.photos img{width:72px;height:72px;object-fit:cover;border-radius:8px;border:1px solid var(--border)}
.photos .ph{position:relative}.photos .ph small{position:absolute;left:3px;top:3px;background:#000a;color:#fff;font-size:10px;padding:0 5px;border-radius:6px}
ul.check{list-style:none;padding:0;margin:0;font-size:17px}ul.check li{padding:6px 0;border-bottom:1px dashed var(--border)}ul.check li::before{content:"☐";margin-right:10px;font-size:20px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{background:var(--bg);font-weight:700;position:sticky;top:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tr.off td{color:var(--muted)}
details summary{cursor:pointer;font-weight:700;font-size:16px;padding:6px 0}
.note{font-size:14px;color:var(--ink2)}.note li{margin:4px 0}
.leaflet-popup-content{font-family:var(--font);font-size:14px;margin:10px 12px}.leaflet-popup-content img{max-width:220px;border-radius:8px;display:block;margin-top:6px}
.zn{display:flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:50%;background:#fff;color:#111;font-weight:800;font-size:16px;border:3px solid var(--accent);box-shadow:0 1px 6px #0006}
.dp{font-size:30px;filter:drop-shadow(0 1px 2px #000a);cursor:grab;line-height:1}
.foot{color:var(--muted);font-size:13px;margin:20px 0}
.zpanel{position:absolute;top:52px;right:10px;bottom:10px;width:360px;max-width:calc(100% - 20px);background:var(--card);color:var(--ink);border:1px solid var(--border);border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);z-index:850;overflow:auto;display:none;padding:12px 14px}
.zpanel.open{display:block}.zpanel .zh{display:flex;align-items:center;gap:8px}.zpanel .zh .stepnum{width:32px;height:32px;font-size:16px}.zpanel h3{margin:0;font-size:17px;flex:1;min-width:0}
.zpanel .x{font:inherit;font-size:18px;border:0;background:transparent;color:var(--ink2);cursor:pointer;padding:2px 6px}
.zpanel .olist{list-style:none;padding:0;margin:8px 0}.zpanel .olist li{display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px dashed var(--border);cursor:pointer}
.zpanel .olist img{width:52px;height:52px;object-fit:cover;border-radius:8px;border:1px solid var(--border);flex:none}.zpanel .olist .noimg{width:52px;height:52px;border-radius:8px;background:var(--bg);flex:none}
.zpanel .olist .ot{flex:1;min-width:0;font-size:13px;line-height:1.35}.zpanel .olist .ot b{font-size:14px}.zpanel .olist input{width:20px;height:20px;flex:none}
.zpanel .olist li.done .ot{opacity:.5;text-decoration:line-through}.zpanel .facts{grid-template-columns:repeat(4,1fr)}.zpanel .btns{margin-top:8px}
@media (max-width:600px){.zpanel{top:auto;left:10px;right:10px;bottom:10px;width:auto;height:62%}}
.busy{position:absolute;inset:0;background:#0006;color:#fff;display:none;align-items:center;justify-content:center;font-size:18px;font-weight:700;z-index:800;border-radius:12px}
.team{display:flex;align-items:center;gap:10px;margin:22px 0 4px;padding-top:12px;border-top:2px solid var(--border)}.team h3{margin:0;font-size:16px;color:var(--ink2);font-weight:600}
.team .pill{font-size:14px;padding:4px 12px;border-radius:999px;color:#fff;font-weight:800}
.donebtn{font:inherit;font-size:13px;padding:5px 10px;border-radius:999px;border:1px solid var(--ok);background:transparent;color:var(--ok);cursor:pointer;flex:none}.donebtn:hover{background:var(--ok);color:#fff}
.links{font-size:13px;margin-top:8px;color:var(--ink2)}.links a{color:var(--accent);text-decoration:none}.links a:hover{text-decoration:underline}
.bar{height:14px;background:var(--bg);border-radius:999px;overflow:hidden;border:1px solid var(--border)}.bar div{height:100%;background:var(--ok);transition:width .3s}
.cal{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px 14px;align-items:end}.cal label{display:flex;flex-direction:column;font-size:13px;color:var(--ink2);gap:3px}.cal label b{color:var(--ink);font-size:14px}
.cal input,.cal select{font:inherit;font-size:16px;padding:6px 8px;border:1px solid var(--border);border-radius:9px;background:var(--bg);color:var(--ink);width:100%}
#cal-msg{font-size:14px;color:var(--ok);margin-top:8px;font-weight:600}
@media (max-width:600px){header h1{font-size:22px}.tile .val{font-size:24px}#map{height:420px}.facts{grid-template-columns:repeat(2,1fr)}.panel{position:static}}
@media print{body{background:#fff;font-size:12px}.card,.tile,.step{break-inside:avoid;box-shadow:none}.mapbtns,.panel,.btns{display:none}#map{height:auto}#map .leaflet-container{display:none}#map img.fallback{display:block!important}
details>summary{display:none}details:not([open])>*{display:block}.steps{grid-template-columns:repeat(2,1fr)}}
"""

JS_APP = r"""
(function(){
'use strict';
const D = window.PLAN;
const $ = (s)=>document.querySelector(s);
const esc = (s)=>String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const DAYC = D.day_colors;
const TEAMC = ["#2a78d6","#eb6834","#1baf7a","#8e44ad","#eda100","#e34948"];
const TEAMN = ["A","B","C","D","E","F"];
const COMPASS = ["북","북동","동","남동","남","남서","서","북서"];
const CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳";
const NIOSH = 23;

// ───────── 저장 (이 브라우저에만): 완료 체크, 실측 보정 계수 ─────────
const LS_KEY = 'shoresweep:' + (D.site || 'site');
let ST = {done: new Set(), calib: {}};
try { const o = JSON.parse(localStorage.getItem(LS_KEY) || '{}'); ST.done = new Set(o.done || []); ST.calib = o.calib || {}; } catch (e) {}
const persist = ()=>{ try { localStorage.setItem(LS_KEY, JSON.stringify({done:[...ST.done], calib:ST.calib})); } catch (e) {} };

// ───────── 좌표: 위경도 ↔ 투영 m (아핀 근사) ─────────
const AF = D.affine.fwd, AI = D.affine.inv;
const ll2xy = (lon,lat)=>[AF[0]*lon+AF[1]*lat+AF[2], AF[3]*lon+AF[4]*lat+AF[5]];
const xy2ll = (x,y)=>{ const u=x-AF[2], v=y-AF[5]; return [AI[0]*u+AI[1]*v, AI[2]*u+AI[3]*v]; };
const toLL=(p)=>{ const ll=xy2ll(p[0],p[1]); return [ll[1],ll[0]]; };

// ───────── 지형 격자 + 다익스트라 ─────────
const T = D.terrain;
let G=null, NR=0, NC=0, CELL=10, GX0=0, GY0=0;
if (T) { const bin = atob(T.b64); G = new Uint8Array(bin.length); for (let i=0;i<bin.length;i++) G[i]=bin.charCodeAt(i); NR=T.nr; NC=T.nc; CELL=T.cell; GX0=T.x0; GY0=T.y0; }
const cellOf = (x,y)=>{ let c=Math.floor((x-GX0)/CELL), r=Math.floor((GY0-y)/CELL); r=Math.min(Math.max(r,0),NR-1); c=Math.min(Math.max(c,0),NC-1); return r*NC+c; };
const xyOfCell = (k)=>[GX0+((k%NC)+0.5)*CELL, GY0-(Math.floor(k/NC)+0.5)*CELL];
let runCache = {};
function forcePassable(xy){ if(!G)return; const k=cellOf(xy[0],xy[1]); if(G[k]===0||G[k]===1){ G[k]=3; runCache={}; } }
class Heap{ constructor(){this.k=[];this.v=[];}
  push(key,val){ const k=this.k,v=this.v; k.push(key); v.push(val); let i=k.length-1; while(i>0){ const p=(i-1)>>1; if(k[p]<=k[i])break; [k[p],k[i]]=[k[i],k[p]]; [v[p],v[i]]=[v[i],v[p]]; i=p; } }
  pop(){ const k=this.k,v=this.v; const rk=k[0], rv=v[0]; const lk=k.pop(), lv=v.pop(); if(k.length){ k[0]=lk; v[0]=lv; let i=0; const n=k.length; while(true){ const l=2*i+1,r=l+1; let m=i; if(l<n&&k[l]<k[m])m=l; if(r<n&&k[r]<k[m])m=r; if(m===i)break; [k[m],k[i]]=[k[i],k[m]]; [v[m],v[i]]=[v[i],v[m]]; i=m; } } return [rk,rv]; }
  get size(){return this.k.length;} }
function costKey(mode){ return mode+'|'+JSON.stringify(T.cost[mode]); }
function dijkstra(mode, src){
  const key = costKey(mode)+':'+src; if (runCache[key]) return runCache[key];
  const cost = T.cost[mode]; const mult = new Float32Array(4); const pass=[false,false,false,false];
  for (const k of [0,1,2,3]) { const v = cost[String(k)]; if (v!==null && v!==undefined) { mult[k]=v; pass[k]=true; } }
  const N = NR*NC; const dist = new Float64Array(N).fill(Infinity); const prev = new Int32Array(N).fill(-1);
  const done = new Uint8Array(N); const h = new Heap(); dist[src]=0; h.push(0,src);
  const dirs = [[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,Math.SQRT2],[1,-1,Math.SQRT2],[-1,1,Math.SQRT2],[-1,-1,Math.SQRT2]];
  while (h.size){ const [d,u] = h.pop(); if (done[u]) continue; done[u]=1; const gu=G[u]; if(!pass[gu]) continue;
    const r=Math.floor(u/NC), c=u%NC; const wu = gu===1;
    for (const [dr,dc,dl] of dirs){ const rr=r+dr, cc=c+dc; if(rr<0||rr>=NR||cc<0||cc>=NC) continue; const v=rr*NC+cc; const gv=G[v]; if(!pass[gv]||done[v]) continue;
      let w = CELL*dl*(mult[gu]+mult[gv])/2; if ((gv===1)!==wu) w += T.transition_m; const nd=d+w; if (nd<dist[v]){ dist[v]=nd; prev[v]=u; h.push(nd,v); } } }
  return runCache[key] = {dist, prev};
}
function pathCells(mode, aCell, bCell){ const {dist,prev} = dijkstra(mode,aCell); if(!isFinite(dist[bCell])) return null; const out=[]; let j=bCell; while(j>=0 && j!==aCell){ out.push(j); j=prev[j]; } out.push(aCell); out.reverse(); return out; }

// ───────── 파라미터 ─────────
const DEF = D.defaults;
let P = JSON.parse(JSON.stringify(DEF));
let depot = {lon:D.depot.lon, lat:D.depot.lat, name:D.depot.name};
const defaultDepot = {lon:D.depot.lon, lat:D.depot.lat, name:D.depot.name};
function readControls(){
  const num=(id,dflt)=>{ const el=$('#'+id); if(!el) return dflt; const v=parseFloat(el.value); return isFinite(v)?v:dflt; };
  P.workers=Math.max(1,Math.round(num('c-workers',2))); P.hours_per_day=Math.max(0.5,num('c-hours',4)); P.teams=Math.max(1,Math.round(num('c-teams',1)));
  P.link_m=num('c-link',150); P.walk_kmh=Math.max(0.5,num('c-walk',3));
  P.item_min=num('c-item',2); P.min_per_m2=num('c-m2',1.5); P.carry_kg_per_person=Math.max(1,num('c-ckg',15)); P.carry_bags_per_person=Math.max(1,Math.round(num('c-cbags',3)));
  P.min_kg=num('c-minkg',0); P.bag_kg=Math.max(1,num('c-bagkg',15)); P.bag_l=Math.max(5,num('c-bagl',80)); P.detour=num('c-detour',1.4); P.veg_cost=Math.max(1,num('c-veg',3)); P.boat_cost=Math.max(0.05,num('c-boat',0.5));
  P.round_trip=$('#c-round').checked;
  P.include_codes=[...document.querySelectorAll('.codes input:checked')].map(e=>e.value);
  P.exclude_done=$('#c-exdone') ? $('#c-exdone').checked : true;
  for (const c of Object.keys(D.mats)) { const el=$('#c-cal-'+c); if (el) { const v=parseFloat(el.value); ST.calib[c] = isFinite(v)&&v>0 ? v : 1; } }
  persist();
  if (T){ T.cost.walk['2']=P.veg_cost; T.cost.boat['2']=P.veg_cost; T.cost.boat['1']=P.boat_cost; }
}
function segVal(name){ const b=document.querySelector(`.seg[data-name="${name}"] button.on`); return b?b.dataset.v:null; }
function setSeg(name,v){ document.querySelectorAll(`.seg[data-name="${name}"] button`).forEach(b=>b.classList.toggle('on', b.dataset.v===v)); }
function opts(over){ return Object.assign({travel:segVal('travel'), carry:segVal('carry'), objective:segVal('objective'), wsrc:segVal('wsrc'), wstat:segVal('wstat'),
  workers:P.workers, hours:P.hours_per_day, teams:P.teams, scale:1, exclude_done:P.exclude_done}, over||{}); }

// ───────── 모델 (collect.py 와 같은 규칙) ─────────
function bagsOf(kg, m3){ const byW=kg/P.bag_kg, byV=m3*D.bag.bulk/(P.bag_l/1000); if (byV>=byW) return [Math.max(Math.ceil(byV), (m3>0||kg>0)?1:0),'volume']; return [Math.max(Math.ceil(byW),1),'weight']; }
function compassName(dx,dy){ const ang=(Math.atan2(dx,dy)*180/Math.PI+360)%360; return COMPASS[Math.floor((ang+22.5)/45)%8]; }
function tourOrder(Dm, start, nodes, roundTrip){
  const n=nodes.length; if(!n) return [];
  const L=(order)=>{ let s=0,pos=start; for(const i of order){ s+=Dm[pos][i]; pos=i; } return s+(roundTrip?Dm[pos][start]:0); };
  if (n<=8){ let best=Infinity, bo=nodes.slice(); const perm=(arr,m)=>{ if(!arr.length){ if(roundTrip&&n>1&&m[0]>m[n-1])return; const l=L(m); if(l<best){best=l;bo=m.slice();} return;} for(let i=0;i<arr.length;i++){ const rest=arr.slice(); const v=rest.splice(i,1)[0]; perm(rest, m.concat([v])); } }; perm(nodes,[]); return bo; }
  let rem=new Set(nodes), pos=start, order=[]; while(rem.size){ let best=null,bd=Infinity; for(const i of rem){ if(Dm[pos][i]<bd){bd=Dm[pos][i];best=i;} } order.push(best); rem.delete(best); pos=best; }
  let best=L(order), improved=true;
  while(improved){ improved=false;
    for(let i=0;i<n-1;i++) for(let j=i+1;j<n;j++){ const cand=order.slice(0,i).concat(order.slice(i,j+1).reverse(), order.slice(j+1)); const l=L(cand); if(l<best-1e-9){order=cand;best=l;improved=true;} }
    for(let i=0;i<n;i++) for(let j=0;j<n;j++){ if(i===j)continue; const cand=order.slice(); const v=cand.splice(i,1)[0]; cand.splice(j,0,v); const l=L(cand); if(l<best-1e-9){order=cand;best=l;improved=true;} } }
  return order;
}
function computePlan(o){
  const t0=performance.now();
  const mode = (T && o.travel==='boat') ? 'boat' : 'walk';
  const objs = D.objects.map(ob=>{ const cal = (o.wsrc==='company') ? 1 : (ST.calib[ob.code]||1);
    const kg = ((o.wsrc==='company' && ob.ckg!=null) ? ob.ckg : (o.wstat==='min'?ob.kmin: o.wstat==='max'?ob.kmax:ob.ktyp)*cal) * o.scale;
    const heavy = kg>NIOSH || (ob.area>=3 && ob.kmax>NIOSH); const done = ST.done.has(ob.id);
    const inc = P.include_codes.includes(ob.code) && kg>=P.min_kg && !(o.exclude_done && done);
    return Object.assign({}, ob, {kg, heavy, inc, done, zone:-1, order:0}); });
  const inc = objs.filter(ob=>ob.inc); const n=inc.length;
  const dxy = ll2xy(depot.lon, depot.lat);
  const nodes = [{x:dxy[0],y:dxy[1]}].concat(inc.map(ob=>({x:ob.x,y:ob.y})));
  let Dm, cells=null, unreachable=0;
  if (T && n){ nodes.forEach(nd=>forcePassable([nd.x,nd.y])); cells = nodes.map(nd=>cellOf(nd.x,nd.y));
    Dm = nodes.map((a,i)=>{ const {dist}=dijkstra(mode, cells[i]); return cells.map((c,j)=> i===j?0:dist[c]); });
    for(let i=0;i<=n;i++) for(let j=0;j<=n;j++) if(!isFinite(Dm[i][j])){ unreachable++; Dm[i][j]=Math.hypot(nodes[i].x-nodes[j].x, nodes[i].y-nodes[j].y)*P.detour*3; }
  } else { Dm = nodes.map(a=>nodes.map(b=>Math.hypot(a.x-b.x,a.y-b.y)*P.detour)); }
  const seg = (i,j)=>{ if (T && cells){ const pc = pathCells(mode, cells[i], cells[j]); if (pc){ const pts=pc.map(xyOfCell); return [[nodes[i].x,nodes[i].y]].concat(pts.slice(1,-1), [[nodes[j].x,nodes[j].y]]); } } return [[nodes[i].x,nodes[i].y],[nodes[j].x,nodes[j].y]]; };
  const parent = Array.from({length:n},(_,i)=>i); const find=(i)=>{ while(parent[i]!==i){ parent[i]=parent[parent[i]]; i=parent[i]; } return i; };
  for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) if(Dm[i+1][j+1]<=P.link_m){ const a=find(i),b=find(j); if(a!==b) parent[a]=b; }
  const remap={}; const groups={}; inc.forEach((ob,i)=>{ const r=find(i); if(!(r in remap)) remap[r]=Object.keys(remap).length; ob.zone=remap[r]; (groups[ob.zone]=groups[ob.zone]||[]).push(i+1); });
  const zoneIds = Object.keys(groups).map(Number).sort((a,b)=>a-b); const Z=zoneIds.length;
  const zkg={}, zwork={}; for(const z of zoneIds){ zkg[z]=groups[z].reduce((s,i)=>s+inc[i-1].kg,0); zwork[z]=groups[z].reduce((s,i)=>s+P.item_min+P.min_per_m2*inc[i-1].area,0)/o.workers + DEF.heavy_extra_min*groups[z].filter(i=>inc[i-1].heavy).length; }
  const ZD = Array.from({length:Z+1},()=>new Array(Z+1).fill(0));
  zoneIds.forEach((za,ai)=>{ const a=ai+1; ZD[0][a]=ZD[a][0]=Math.min(...groups[za].map(i=>Dm[0][i])); zoneIds.forEach((zb,bi)=>{ const b=bi+1; if(a!==b){ let m=Infinity; for(const i of groups[za]) for(const j of groups[zb]) if(Dm[i][j]<m)m=Dm[i][j]; ZD[a][b]=m; } }); });
  const wmpm = 60/(P.walk_kmh*1000);
  let order;
  if (o.objective==='weight' && Z){ let rem=zoneIds.map((_,i)=>i+1), pos=0; order=[]; while(rem.length){ let best=null,bv=-1; for(const a of rem){ const z=zoneIds[a-1]; const v=zkg[z]/Math.max(ZD[pos][a]*wmpm+zwork[z],1e-6); if(v>bv){bv=v;best=a;} } order.push(best); rem=rem.filter(x=>x!==best); pos=best; } }
  else order = tourOrder(ZD, 0, zoneIds.map((_,i)=>i+1), P.round_trip);
  const ordered = order.map(a=>zoneIds[a-1]);
  const cents={}; for(const z of zoneIds){ const L=groups[z].map(i=>inc[i-1]); cents[z]=[L.reduce((s,ob)=>s+ob.x,0)/L.length, L.reduce((s,ob)=>s+ob.y,0)/L.length]; }
  const dirCount={}, seen={}, names={}; for(const z of ordered){ const d=compassName(cents[z][0]-D.center.x, cents[z][1]-D.center.y); dirCount[d]=(dirCount[d]||0)+1; }
  for(const z of ordered){ const d=compassName(cents[z][0]-D.center.x, cents[z][1]-D.center.y); seen[d]=(seen[d]||0)+1; names[z]=d+'쪽 해안'+(dirCount[d]>1&&seen[d]<=CIRC.length?' '+CIRC[seen[d]-1]:''); }
  const capKg=o.workers*P.carry_kg_per_person, capBags=o.workers*P.carry_bags_per_person;
  // 한 팀이 seq 순서로 도는 함수
  function traverse(seq, startStep, team){
    const zones=[]; let pos=0, cumMin=0, cumBags=0, loadKg=0, loadBags=0;
    const speedF=()=>1-Math.min(0.4, DEF.load_slow*loadKg/o.workers);
    seq.forEach((z,si)=>{ const step=startStep+si; let members=groups[z].slice(); const inner=[]; let cur=pos;
      while(members.length){ let best=null,bd=Infinity; for(const i of members){ if(Dm[cur][i]<bd){bd=Dm[cur][i];best=i;} } inner.push(best); members=members.filter(x=>x!==best); cur=best; }
      inner.forEach((i,k)=>{ inc[i-1].order=k+1; });
      const dIn=Dm[pos][inner[0]]; const segs=[]; let walkMin=0, distTot=0, returns=0; cur=pos; let approach=null;
      for(const i of inner){ const ob=inc[i-1]; const b=bagsOf(ob.kg,ob.vol)[0];
        if (o.carry==='carry' && (loadKg>0||loadBags>0) && (loadKg+ob.kg>capKg || loadBags+b>capBags)){ const back=Dm[cur][0], out=Dm[0][i]; walkMin+=back*wmpm/speedF(); loadKg=0; loadBags=0; walkMin+=out*wmpm/speedF(); distTot+=back+out; segs.push(seg(cur,0)); segs.push(seg(0,i)); returns++; }
        else { const d=Dm[cur][i]; walkMin+=d*wmpm/speedF(); distTot+=d; segs.push(seg(cur,i)); }
        if (approach===null) approach = segs.length===1 ? segs[0] : segs[0].concat(segs[1].slice(1));
        if (o.carry==='carry'){ loadKg+=ob.kg; loadBags+=b; } cur=i; }
      pos=cur; const L=inner.map(i=>inc[i-1]); const kg=L.reduce((s,ob)=>s+ob.kg,0), m3=L.reduce((s,ob)=>s+ob.vol,0); const [bags]=bagsOf(kg,m3);
      const heavy=L.filter(ob=>ob.heavy); const workMin=zwork[z]; cumMin+=walkMin+workMin; cumBags+=bags;
      const byCode={}; for(const ob of L) byCode[ob.code]=(byCode[ob.code]||0)+1;
      const tools=[...new Set(Object.keys(byCode).map(c=>D.mats[c]&&D.mats[c].tool).filter(Boolean))].sort();
      const notes=[]; if(heavy.length) notes.push(`무거운 물체 ${heavy.length}개 → 2인 이상 또는 장비 (NIOSH 23 kg 초과 가능)`);
      const big=L.filter(ob=>ob.area>=3); if(big.length) notes.push(`면적 3 m² 이상 큰 물체 ${big.length}개 (${big.map(ob=>`${ob.ko} ${ob.area.toFixed(1)} m²`).join(', ')})`);
      if(returns) notes.push(`적재량 초과로 출발지 복귀 ${returns}회 포함`);
      const path=[]; segs.forEach((sg,si2)=>{ sg.forEach((q,k)=>{ if(!(si2>0&&k===0)) path.push(q); }); });
      const cll = xy2ll(cents[z][0],cents[z][1]);
      zones.push({zone:z, step, team, name:names[z], cx:cents[z][0], cy:cents[z][1], lon:cll[0], lat:cll[1], objects:L, n:L.length, byCode, kg, kmin:L.reduce((s,ob)=>s+ob.kmin,0), kmax:L.reduce((s,ob)=>s+ob.kmax,0), ckg:L.reduce((s,ob)=>s+(ob.ckg||0),0), m3, bags, heavy, tools, notes, path, approach:approach||[], dIn, distTot, walkMin, workMin, cumMin, cumBags, returns, day:1, firstCell: cells?cells[inner[0]]:null}); });
    let backM=0, backMin=0, backPath=[]; if (P.round_trip && zones.length){ backM=Dm[pos][0]; backMin=backM*wmpm/speedF(); backPath=seg(pos,0); }
    return {zones, backM, backMin, backPath};
  }
  // 팀 분할: 한 팀 순회를 시간 균형(전체 누적 기준)으로 연속 구간으로
  const nTeams=Math.max(1,o.teams); let segments;
  if (nTeams>1 && ordered.length>1){ const single=traverse(ordered,1,1); const tot=single.zones.reduce((s,z)=>s+z.walkMin+z.workMin,0); segments=[]; let cur=[], cum=0;
    single.zones.forEach((z,k)=>{ cur.push(ordered[k]); cum+=z.walkMin+z.workMin; const remaining=single.zones.length-k-1; const teamsLeft=nTeams-segments.length-1; if(segments.length<nTeams-1 && remaining>=teamsLeft && (cum>=(segments.length+1)*tot/nTeams || remaining===teamsLeft)){ segments.push(cur); cur=[]; } });
    if(cur.length) segments.push(cur); }
  else segments = ordered.length ? [ordered] : [];
  const zones=[], teams=[], days=[]; let routeLen=0, totalWalk=0, totalWork=0, step0=1; const dayCap=o.hours*60;
  segments.forEach((seq,ti)=>{ const t=ti+1; const r=traverse(seq, step0, t); step0+=r.zones.length;
    const walk=r.zones.reduce((s,z)=>s+z.walkMin,0)+r.backMin, work=r.zones.reduce((s,z)=>s+z.workMin,0); routeLen+=r.zones.reduce((s,z)=>s+z.distTot,0)+r.backM; totalWalk+=walk; totalWork+=work;
    let day=1, acc=0; for(const z of r.zones){ if(acc>0 && acc+z.walkMin+z.workMin>dayCap){ days.push({team:t, day, steps:r.zones.filter(q=>q.day===day).map(q=>q.step), minutes:acc}); day++; acc=0; } z.day=day; acc+=z.walkMin+z.workMin; }
    if(r.zones.length) days.push({team:t, day, steps:r.zones.filter(q=>q.day===day).map(q=>q.step), minutes:acc+r.backMin});
    teams.push({team:t, zones:r.zones.map(z=>z.step), n:r.zones.reduce((s,z)=>s+z.n,0), kg:r.zones.reduce((s,z)=>s+z.kg,0), bags:r.zones.reduce((s,z)=>s+z.bags,0), walkMin:walk, workMin:work, minutes:walk+work, days:r.zones.length?day:0, backPath:r.backPath, routeM:r.zones.reduce((s,z)=>s+z.distTot,0)+r.backM});
    zones.push(...r.zones); });
  const totalMin = teams.length ? Math.max(...teams.map(t=>t.minutes)) : 0;
  const kgPlan=inc.reduce((s,ob)=>s+ob.kg,0), vol=inc.reduce((s,ob)=>s+ob.vol,0);
  const doneObjs=objs.filter(ob=>ob.done); const allSel=objs.filter(ob=>P.include_codes.includes(ob.code) && ob.kg>=P.min_kg);
  const totals={kg:kgPlan, kmin:inc.reduce((s,ob)=>s+ob.kmin,0), kmax:inc.reduce((s,ob)=>s+ob.kmax,0), ckg:inc.reduce((s,ob)=>s+(ob.ckg||0),0), vol, bags:zones.reduce((s,z)=>s+z.bags,0), heavy:zones.reduce((s,z)=>s+z.heavy.length,0), zones:zones.length, returns:zones.reduce((s,z)=>s+z.returns,0), tonbags: inc.length?Math.ceil(Math.max(kgPlan/D.bag.tonbag_kg, vol*D.bag.bulk/D.bag.tonbag_m3)):0,
    done:doneObjs.length, doneKg:doneObjs.reduce((s,ob)=>s+ob.kg,0), all:allSel.length, allKg:allSel.reduce((s,ob)=>s+ob.kg,0)};
  const byCode={}; for(const ob of inc){ const d=byCode[ob.code]=byCode[ob.code]||{ko:ob.ko,color:ob.color,count:0,area:0,kg:0,kmin:0,kmax:0,ckg:0,vol:0}; d.count++; d.area+=ob.area; d.kg+=ob.kg; d.kmin+=ob.kmin; d.kmax+=ob.kmax; d.ckg+=ob.ckg||0; d.vol+=ob.vol; }
  const equipment=[`마대 ${Math.ceil(totals.bags*1.2)}장 (계산 ${totals.bags}장 + 여유 20 %)`, `장갑·집게 ${o.workers*nTeams}명분`];
  const tools=[...new Set(zones.flatMap(z=>z.tools))].sort(); if(tools.length) equipment.push(tools.join(' / ')+' (로프·그물 자르기)');
  if (totals.tonbags>=1 && vol*D.bag.bulk>0.5) equipment.push(`톤백 ${totals.tonbags}개 또는 집결지 적재 공간 ${(vol*D.bag.bulk).toFixed(1)} m³`);
  if (totals.heavy) equipment.push(`무거운 물체 ${totals.heavy}개 → 2인 운반 또는 손수레`);
  if (mode==='boat') equipment.push('보트·구명조끼, 승·하선 지점 사전 확인');
  if (nTeams>1) equipment.push(`팀 ${nTeams}개 → 팀별 무전기·연락 수단, 집결 시각 약속`);
  equipment.push('식수·구급약, 물때표 확인 (갯바위 구간)');
  return {o, objs, inc, zones, teams, days, totals, byCode, equipment, routeLen, totalWalk, totalWork, totalMin, mode, unreachable, ms:performance.now()-t0, skipped:allSel.length-inc.length, nTeams};
}

// ───────── 표시 ─────────
const fmtKg=(kg)=>kg>=100?kg.toLocaleString('ko',{maximumFractionDigits:0}):kg>=1?kg.toFixed(1):kg.toFixed(2);
const fmtMin=(m)=>m>=60?`${Math.floor(m/60)}시간 ${Math.round(m%60)}분`:`${Math.round(m)}분`;
const colorOf=(z,R)=> (R.nTeams>1 ? TEAMC[(z.team-1)%TEAMC.length] : DAYC[(z.day-1)%DAYC.length]);
const teamName=(t)=>TEAMN[t-1]||String(t);
const navLinks=(name,lat,lon)=>`<a href="https://map.kakao.com/link/to/${encodeURIComponent(name)},${lat.toFixed(6)},${lon.toFixed(6)}" target="_blank" rel="noopener">카카오맵 길찾기</a> · <a href="https://www.google.com/maps/dir/?api=1&destination=${lat.toFixed(6)},${lon.toFixed(6)}&travelmode=walking" target="_blank" rel="noopener">구글 지도</a>`;
let map=null, layers=null, zoneMarkers={}, depotMarker=null, meMarker=null, allBounds=null, fitted=false;
// 출발지 제한: 정사영상 범위 안 + 땅(맨땅·숲) 또는 해안 2칸(20 m) 이내의 물(부두·선착장). 벗어나면 이유 문자열 반환
function depotProblem(lon,lat){
  if (D.basemap){ const b=D.basemap.bounds; if (lat<b[0]||lat>b[2]||lon<b[1]||lon>b[3]) return '정사영상 범위 밖'; }
  if (!T) return null;
  const [x,y]=ll2xy(lon,lat); const c=Math.floor((x-GX0)/CELL), r=Math.floor((GY0-y)/CELL);
  if (r<0||r>=NR||c<0||c>=NC) return '지형 격자 밖';
  const g=G[r*NC+c]; if (g===2||g===3) return null;
  if (g===0) return '촬영되지 않은 영역';
  for (let dr=-2;dr<=2;dr++) for (let dc=-2;dc<=2;dc++){ const rr=r+dr, cc=c+dc; if(rr<0||rr>=NR||cc<0||cc>=NC) continue; const gg=G[rr*NC+cc]; if (gg===2||gg===3) return null; }
  return '해안에서 20 m 넘게 떨어진 바다';
}
function initMap(){
  const fb=$('#fallback');
  if (typeof L==='undefined'){ fb.style.display='block'; $('#maphint').textContent='인터넷 연결이 없어 인쇄용 지도(기본 설정)를 보여줍니다. 숫자·카드는 아래에서 계속 다시 계산됩니다.'; return; }
  map=L.map('map',{zoomControl:true,preferCanvas:true,scrollWheelZoom:false}); map.on('click focus',()=>map.scrollWheelZoom.enable()); map.on('mouseout',()=>map.scrollWheelZoom.disable());
  const base={}; const ov={};
  if (!D.no_tiles){
    const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:21,maxNativeZoom:19,attribution:'Esri World Imagery'});
    const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}); sat.addTo(map);
    let ok=false; sat.on('tileload',()=>{ok=true;}); setTimeout(()=>{ if(!ok&&!D.basemap){ fb.style.display='block'; } },6000);
    base['위성 지도']=sat; base['일반 지도']=osm;
  } else if (!D.basemap) { fb.style.display='block'; }
  if (D.basemap){ const o=L.imageOverlay(D.basemap.url,[[D.basemap.bounds[0],D.basemap.bounds[1]],[D.basemap.bounds[2],D.basemap.bounds[3]]],{opacity:1,zIndex:5}); o.addTo(map); ov['드론 정사영상 (조사일)']=o; }
  if (D.terrain_png){ const tb=D.terrain_png.bounds; const o=L.imageOverlay(D.terrain_png.url,[[tb[0],tb[1]],[tb[2],tb[3]]],{opacity:.45,zIndex:6}); ov['지형 분류 (파랑 물·초록 숲·흰색 맨땅)']=o; }
  layers={route:L.layerGroup().addTo(map), obj:L.layerGroup().addTo(map), zone:L.layerGroup().addTo(map)};
  ov['이동 경로']=layers.route; ov['쓰레기 위치']=layers.obj; ov['구역 번호']=layers.zone;
  L.control.layers(Object.keys(base).length?base:null,ov,{collapsed:true,position:'topright'}).addTo(map); L.control.scale({imperial:false}).addTo(map);
  if (D.no_tiles){ map.options.maxZoom=20; map.setMaxZoom(20); }
  depotMarker=L.marker([depot.lat,depot.lon],{icon:L.divIcon({className:'',html:'<div class="dp" title="끌어서 출발지 변경">★</div>',iconSize:[30,30],iconAnchor:[15,15]}),zIndexOffset:2000,draggable:true}).addTo(map);
  depotMarker.bindTooltip('출발·집결지 (끌어서 옮길 수 있음)',{direction:'top',offset:[0,-12]});
  depotMarker.on('dragend',()=>{ const ll=depotMarker.getLatLng(); const why=depotProblem(ll.lng,ll.lat);
    if (why){ depotMarker.setLatLng([depot.lat,depot.lon]); $('#status').textContent='출발지를 옮길 수 없습니다: '+why+' — 정사영상 안의 땅이나 해안 20 m 이내(부두 포함)에만 둘 수 있습니다'; $('#status').style.color='#c0392b'; setTimeout(()=>{$('#status').style.color='';},4000); return; }
    depot={lon:ll.lng,lat:ll.lat,name:'지정 출발지'}; recompute(); });
  allBounds=L.latLngBounds(D.objects.map(ob=>[ob.lat,ob.lon]).concat([[depot.lat,depot.lon]]));
  const refit=()=>{ const el=map.getContainer(); if(el.clientWidth>0&&el.clientHeight>0){ map.invalidateSize(); if(!fitted){ fitted=true; map.fitBounds(allBounds.pad(0.08),{animate:false}); } } };
  refit(); map.whenReady(refit); window.addEventListener('load',refit); setTimeout(refit,300); setTimeout(refit,1500);
  if (window.ResizeObserver) new ResizeObserver(refit).observe(map.getContainer());
  const iv=setInterval(()=>{refit(); if(fitted)clearInterval(iv);},500); setTimeout(()=>clearInterval(iv),20000);
  document.addEventListener('visibilitychange',refit); ['pointermove','touchstart','scroll','focus'].forEach(ev=>window.addEventListener(ev,refit,{passive:true}));
  window.fitAll=()=>{ map.invalidateSize(); map.fitBounds(allBounds.pad(0.08)); };
  window._map=map;
}
function renderMap(R){
  if(!map) return; layers.route.clearLayers(); layers.obj.clearLayers(); layers.zone.clearLayers(); zoneMarkers={};
  for(const z of R.zones){ const col=colorOf(z,R); const ap=z.approach.map(toLL); const rest=z.path.slice(Math.max(z.approach.length-1,0)).map(toLL);
    if(ap.length>1){ L.polyline(ap,{color:'#fff',weight:7,opacity:.85}).addTo(layers.route); L.polyline(ap,{color:col,weight:3.5,dashArray:'10 8'}).addTo(layers.route); }
    if(rest.length>1){ L.polyline(rest,{color:'#fff',weight:7,opacity:.85}).addTo(layers.route); L.polyline(rest,{color:col,weight:4}).addTo(layers.route); } }
  for(const t of R.teams){ if (t.backPath.length>1){ const bp=t.backPath.map(toLL); const last=R.zones.filter(z=>z.team===t.team).slice(-1)[0]; const col=last?colorOf(last,R):'#555'; L.polyline(bp,{color:'#fff',weight:7,opacity:.85}).addTo(layers.route); L.polyline(bp,{color:col,weight:3.5,dashArray:'4 8'}).addTo(layers.route); } }
  const stepOf={}; R.zones.forEach(z=>z.objects.forEach(ob=>stepOf[ob.id]=z.step));
  for(const ob of R.objs){ const m=L.circleMarker([ob.lat,ob.lon],{radius:ob.inc?(ob.heavy?9:6):4,color:ob.done?'#1baf7a':(ob.inc?'#111':'#777'),weight:ob.done?2:1.2,fillColor:ob.inc?ob.color:(ob.done?'#b9f0d8':'#bbb'),fillOpacity:.95}).addTo(layers.obj);
    const state = ob.done ? '✅ 완료' : (ob.inc ? `${stepOf[ob.id]}-${ob.order}` : '(제외)');
    m.bindPopup(`<b>${state} · ${esc(ob.ko)}</b><br>크기 ${ob.w.toFixed(1)}×${ob.h.toFixed(1)} m (${ob.area.toFixed(1)} m²)<br>무게 ${fmtKg(ob.kg)} kg (범위 ${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)})<br><span style="color:#888">기업 제공값 ${ob.ckg!=null?ob.ckg.toFixed(3):'-'} kg</span>${ob.heavy?'<br><b style="color:#c0392b">⚠ 2인 운반</b>':''}${ob.img?`<img src="${ob.img}" alt="">`:''}<br><a href="#" onclick="toggleDone(['${ob.id}']);return false;">${ob.done?'완료 취소':'이 물체 완료'}</a>${ob.inc?` · <a href="#" onclick="openZone(${stepOf[ob.id]},false);return false;">구역 목록 보기</a>`:''}`); }
  for(const z of R.zones){ const col=colorOf(z,R); const lab=(R.nTeams>1?teamName(z.team):'')+z.step; const ic=L.divIcon({className:'',html:`<div class="zn" style="border-color:${col}">${lab}</div>`,iconSize:[34,34],iconAnchor:[17,17]});
    const m=L.marker([z.lat,z.lon],{icon:ic,zIndexOffset:1000}).addTo(layers.zone); zoneMarkers[z.step]=m;
    const comp=Object.entries(z.byCode).sort((a,b)=>b[1]-a[1]).map(([c,n])=>`${esc(D.mats[c]?D.mats[c].ko:c)} ${n}개`).join(', ');
    m.bindPopup(`<b>${z.step}. ${esc(z.name)}</b>${R.nTeams>1?` · ${teamName(z.team)}팀`:''}<br>${comp}<br>무게 ${fmtKg(z.kg)} kg · 마대 ${z.bags}장<br>접근 ${Math.round(z.dIn)} m · 이동 ${Math.round(z.walkMin)}분 · 작업 ${Math.round(z.workMin)}분${z.heavy.length?`<br><b style="color:#c0392b">⚠ 2인 운반 물체 ${z.heavy.length}개</b>`:''}<br>${navLinks(z.name,z.lat,z.lon)}<br><a href="#" onclick="doneZone(${z.step});return false;">이 구역 완료</a>`);
    m.unbindPopup(); m.on('click',()=>{ highlight(z.step,false); openZone(z.step); }); }
  if (depotMarker) depotMarker.setLatLng([depot.lat,depot.lon]);
  if (PANEL_IDS){ const z=R.zones.find(z=>z.objects.some(ob=>PANEL_IDS.includes(ob.id))); if(z) openZone(z.step,false); else closeZone(); }
}
window.highlight=(step,fly=true)=>{ document.querySelectorAll('.step').forEach(e=>e.classList.toggle('active',+e.dataset.step===step)); if(!CUR)return; const z=CUR.zones.find(z=>z.step===step); if(!z)return; if(fly){ $('#map').scrollIntoView({behavior:'smooth',block:'center'}); openZone(step,true); } };
// ───────── 구역 상세 패널 (waypoint 를 누르면 수거할 물체 목록) ─────────
let PANEL_IDS=null;
window.closeZone=()=>{ PANEL_IDS=null; const el=$('#zpanel'); el.classList.remove('open'); el.innerHTML=''; };
window.zoomZone=(step)=>{ if(!map||!CUR)return; const z=CUR.zones.find(z=>z.step===step); if(!z)return; const b=L.latLngBounds(z.path.map(toLL)); map.flyToBounds(b.pad(0.4),{maxZoom:19, paddingBottomRight:[window.innerWidth>600?380:0, window.innerWidth>600?0:Math.round(map.getSize().y*0.6)]}); };
window.flyObj=(id)=>{ if(!map||!CUR)return; const ob=CUR.objs.find(o=>o.id===id); if(!ob)return; map.flyTo([ob.lat,ob.lon],19); };
window.openZone=(step,fly=true)=>{ if(!CUR)return; const R=CUR; const z=R.zones.find(z=>z.step===step); if(!z){ closeZone(); return; } PANEL_IDS=z.objects.map(ob=>ob.id); const col=colorOf(z,R);
  const chips=Object.entries(z.byCode).sort((a,b)=>b[1]-a[1]).map(([c,n])=>`<span class="chip" style="background:${D.mats[c]?D.mats[c].color:'#ccc'}">${esc(D.mats[c]?D.mats[c].ko:c)} ${n}</span>`).join('');
  const rows=z.objects.map((ob,i)=>`<li class="${ob.done?'done':''}" onclick="flyObj('${ob.id}')">${ob.img?`<img src="${ob.img}" alt="">`:'<div class="noimg"></div>'}<div class="ot"><b>${z.step}-${i+1} ${esc(ob.ko)}</b>${ob.heavy?' <span style="color:#c0392b;font-weight:700">⚠ 2인</span>':''}<br>${ob.w.toFixed(1)}×${ob.h.toFixed(1)} m · ${ob.area.toFixed(1)} m²<br><b>${fmtKg(ob.kg)} kg</b> <span style="color:var(--muted)">(${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)})</span></div><input type="checkbox" title="완료" ${ob.done?'checked':''} onclick="event.stopPropagation();toggleDone(['${ob.id}'])"></li>`).join('');
  const notes=z.notes.map(n=>`<div class="warn">⚠ ${esc(n)}</div>`).join(''); const tools=z.tools.length?`<p class="sub" style="margin:6px 0 0">도구: ${esc(z.tools.join(', '))}</p>`:'';
  $('#zpanel').innerHTML=`<div class="zh"><div class="stepnum" style="background:${col}">${z.step}</div><h3>${esc(z.name)}${R.nTeams>1?` · ${teamName(z.team)}팀`:''} · ${z.day}일차</h3><button class="x" onclick="closeZone()" title="닫기">✕</button></div><div class="chips">${chips}</div><div class="facts"><div class="fact"><b>${z.n}</b><span>개</span></div><div class="fact"><b>${fmtKg(z.kg)}</b><span>kg</span></div><div class="fact"><b>${z.bags}</b><span>마대</span></div><div class="fact"><b>${Math.round(z.walkMin+z.workMin)}</b><span>분</span></div></div><p class="sub" style="margin:4px 0">이전 지점에서 ${Math.round(z.dIn)} m · 이동 ${Math.round(z.walkMin)}분 · 작업 ${Math.round(z.workMin)}분</p>${notes}${tools}<p class="sub" style="margin:8px 0 2px;font-weight:700;color:var(--ink)">수거할 물체 (체크 = 완료, 누르면 지도 이동)</p><ul class="olist">${rows}</ul><div class="links">${navLinks(z.name,z.lat,z.lon)}</div><div class="btns"><button class="primary" onclick="doneZone(${z.step})">구역 전체 완료 ✓</button><button onclick="zoomZone(${z.step})">구역 확대</button></div>`;
  $('#zpanel').classList.add('open'); if(fly) zoomZone(step); };
window.showDay=(team,d)=>{ if(!map||!CUR)return; const zs=CUR.zones.filter(z=>z.team===team&&z.day===d); if(!zs.length)return; const pts=[].concat(...zs.map(z=>z.path.map(toLL))); map.flyToBounds(L.latLngBounds(pts).pad(0.15)); };
function renderTiles(R){ const t=R.totals;
  const tiles=[['수거할 쓰레기',`${R.inc.length}개`,`구역 ${t.zones}곳${R.skipped?` · 제외 ${R.skipped}개`:''}`],['예상 무게',`${fmtKg(t.kg)} kg`,`범위 ${fmtKg(t.kmin)}–${fmtKg(t.kmax)} kg`],['마대',`${t.bags}장`,`부피 약 ${(t.vol*D.bag.bulk).toFixed(1)} m³`],
    [R.nTeams>1?'가장 오래 걸리는 팀':'총 작업 시간',fmtMin(R.totalMin),R.nTeams>1?`${R.nTeams}팀 × ${R.o.workers}명 · 전 팀 합 ${fmtMin(R.totalWalk+R.totalWork)}`:`이동 ${fmtMin(R.totalWalk)} + 작업 ${fmtMin(R.totalWork)}`],
    ['일정',`${Math.max(0,...R.teams.map(x=>x.days))}일`,`팀당 ${R.o.workers}명 · 하루 ${R.o.hours}시간`],['이동 거리',`${(R.routeLen/1000).toFixed(1)} km`,(T?'지형 최단경로(걷기 환산)':'직선×우회')+(t.returns?` · 복귀 ${t.returns}회`:'')+(R.nTeams>1?' · 전 팀 합':'')],['2인 운반 물체',`${t.heavy}개`,'23 kg 넘을 수 있음'],
    ['진행',`${t.all?Math.round(t.done/t.all*100):0} %`,`완료 ${t.done}/${t.all}개 · ${fmtKg(t.doneKg)} kg`]];
  $('#tiles').innerHTML=tiles.map(([l,v,s])=>`<div class="tile"><div class="lab">${l}</div><div class="val">${v}</div><div class="sub">${s}</div></div>`).join('');
  $('#daybtns').innerHTML='<button onclick="fitAll()">전체 보기</button>'+R.days.map(d=>`<button onclick="showDay(${d.team},${d.day})" style="border-color:${R.nTeams>1?TEAMC[(d.team-1)%TEAMC.length]:DAYC[(d.day-1)%DAYC.length]}">${R.nTeams>1?teamName(d.team)+'팀 ':''}${d.day}일차</button>`).join('')+'<button onclick="locateMe()">📍 내 위치</button>';
  $('#legend').innerHTML=Object.entries(R.byCode).sort((a,b)=>b[1].count-a[1].count).map(([c,d])=>`<span><i class="dot" style="background:${d.color}"></i>${esc(d.ko)} ${d.count}개</span>`).join('')+'<span><i class="dot" style="background:#b9f0d8;border-color:#1baf7a"></i>완료</span><span><i class="dot" style="background:#bbb;border-color:#777"></i>제외</span><span><i class="dia"></i>2인 운반(무거움)</span><span>★ 출발·집결지 (끌어서 변경)</span><span>점선 = 구역 접근 / 복귀</span>'+(R.nTeams>1?R.teams.map(x=>`<span><i style="display:inline-block;width:22px;height:4px;background:${TEAMC[(x.team-1)%TEAMC.length]};vertical-align:middle;margin-right:4px"></i>${teamName(x.team)}팀 경로</span>`).join(''):R.days.map(d=>`<span><i style="display:inline-block;width:22px;height:4px;background:${DAYC[(d.day-1)%DAYC.length]};vertical-align:middle;margin-right:4px"></i>${d.day}일차 경로</span>`).join(''));
  $('#status').textContent=`${Math.round(R.ms)} ms 에 다시 계산 · ${R.mode==='boat'?'보트 지원':'도보'} · ${R.o.carry==='carry'?'들고 이동':'현장 적치'} · ${R.o.objective==='weight'?'무게 우선':'최단 이동'}${R.nTeams>1?` · ${R.nTeams}팀`:''}${R.unreachable?` · 경로 없음 ${R.unreachable}쌍(직선 대체)`:''}`;
}
function renderSteps(R){
  let h='';
  for(const tm of R.teams){ if (R.nTeams>1){ const col=TEAMC[(tm.team-1)%TEAMC.length]; h+=`<div class="team"><span class="pill" style="background:${col}">${teamName(tm.team)}팀</span><h3>${tm.zones.length}개 구역 · ${tm.n}개 · ${fmtKg(tm.kg)} kg · 마대 ${tm.bags}장 · ${fmtMin(tm.minutes)} · ${tm.days}일</h3></div>`; }
    for(const d of R.days.filter(d=>d.team===tm.team)){ const col=R.nTeams>1?TEAMC[(d.team-1)%TEAMC.length]:DAYC[(d.day-1)%DAYC.length]; h+=`<div class="day"><span class="pill" style="background:${col}">${d.day}일차</span><h3>${d.steps.length}개 구역 · 약 ${fmtMin(d.minutes)}</h3></div><div class="steps">`;
      for(const z of R.zones.filter(z=>z.team===tm.team&&z.day===d.day)){ const chips=Object.entries(z.byCode).sort((a,b)=>b[1]-a[1]).map(([c,n])=>`<span class="chip" style="background:${D.mats[c]?D.mats[c].color:'#ccc'}">${esc(D.mats[c]?D.mats[c].ko:c)} ${n}</span>`).join('');
        const warn=z.notes.map(n=>`<div class="warn">⚠ ${esc(n)}</div>`).join(''); const tools=z.tools.length?` · 도구: ${esc(z.tools.join(', '))}`:'';
        const phs=z.objects.map((ob,i)=>ob.img?`<div class="ph"><img src="${ob.img}" alt="" title="${esc(ob.id)}"><small>${z.step}-${i+1}</small></div>`:'').join('');
        h+=`<div class="step" data-step="${z.step}" style="border-left-color:${col}" onclick="highlight(${z.step})"><div class="head"><div class="stepnum" style="background:${col}">${z.step}</div><div style="flex:1;min-width:0"><div class="name">${esc(z.name)}</div><div class="meta">이전 지점에서 ${Math.round(z.dIn)} m · 이 구역 이동 ${Math.round(z.walkMin)}분${tools}</div></div><button class="donebtn" onclick="event.stopPropagation();doneZone(${z.step})">완료 ✓</button></div><div class="chips">${chips}</div><div class="facts"><div class="fact"><b>${z.n}</b><span>개</span></div><div class="fact"><b>${fmtKg(z.kg)}</b><span>kg (${fmtKg(z.kmin)}–${fmtKg(z.kmax)})</span></div><div class="fact"><b>${z.bags}</b><span>마대</span></div><div class="fact"><b>${Math.round(z.workMin)}</b><span>분 작업</span></div></div>${warn}<div class="photos">${phs}</div><div class="links" onclick="event.stopPropagation()">${navLinks(z.name,z.lat,z.lon)}</div></div>`; }
      h+='</div>'; } }
  if(!R.zones.length) h='<p class="sub">'+(R.totals.all&&R.totals.done>=R.totals.all?'선택한 쓰레기를 모두 수거했습니다 🎉':'선택한 조건에 맞는 쓰레기가 없습니다. 종류·최소 무게를 확인하세요.')+'</p>';
  $('#steps').innerHTML=h;
  $('#equip').innerHTML=R.equipment.map(e=>`<li>${esc(e)}</li>`).join('');
  $('#bycode').innerHTML=Object.entries(R.byCode).sort((a,b)=>b[1].kg-a[1].kg).map(([c,d])=>{ const m=D.mats[c]||{}; return `<tr><td><i class="dot" style="background:${d.color}"></i>${esc(d.ko)}</td><td class="num">${d.count}</td><td class="num">${d.area.toFixed(1)}</td><td class="num"><b>${fmtKg(d.kg)}</b></td><td class="num">${fmtKg(d.kmin)}–${fmtKg(d.kmax)}</td><td class="num">${d.ckg.toFixed(3)}</td><td>${esc(m.handling||'-')}${m.tool?' · 도구: '+esc(m.tool):''}</td></tr>`; }).join('');
  const stepOf={}; R.zones.forEach(z=>z.objects.forEach(ob=>stepOf[ob.id]=z.step));
  const rows=R.objs.slice().sort((a,b)=>(a.inc?0:1)-(b.inc?0:1)||(stepOf[a.id]||0)-(stepOf[b.id]||0)||a.order-b.order);
  $('#objtab').innerHTML=rows.map(ob=>`<tr class="${ob.inc?'':'off'}"><td>${ob.done?'✅':(ob.inc?`${stepOf[ob.id]}-${ob.order}`:'제외')}</td><td>${esc(ob.id)}</td><td><i class="dot" style="background:${ob.color}"></i>${esc(ob.ko)}</td><td class="num">${ob.w.toFixed(1)}×${ob.h.toFixed(1)}</td><td class="num">${ob.area.toFixed(2)}</td><td class="num"><b>${fmtKg(ob.kg)}</b></td><td class="num">${fmtKg(ob.kmin)}–${fmtKg(ob.kmax)}</td><td class="num">${ob.ckg!=null?ob.ckg:'-'}</td><td>${ob.heavy?'⚠ 2인 운반':''}</td></tr>`).join('');
  // 진행 현황
  const t=R.totals; const pct=t.all?Math.round(t.done/t.all*100):0;
  $('#progress').innerHTML=`<div class="bar"><div style="width:${pct}%"></div></div><p class="sub">완료 ${t.done}개 / 선택 ${t.all}개 (${pct} %) · 완료 무게 ${fmtKg(t.doneKg)} kg · 남은 ${R.inc.length}개 ${fmtKg(t.kg)} kg · 남은 시간 ${fmtMin(R.totalMin)}</p>`+(t.done?`<div class="btns"><button onclick="resetDone()">완료 전부 취소</button></div>`:'');
  // 실측 보정: 구역 선택 목록
  const sel=$('#cal-zone'); if(sel){ const curv=sel.value; sel.innerHTML=R.zones.map(z=>`<option value="${z.step}">${z.step}. ${esc(z.name)} (예상 ${fmtKg(z.kg)} kg)</option>`).join(''); if([...sel.options].some(o=>o.value===curv)) sel.value=curv; }
  $('#assume-params').textContent=`구역 묶기 ${P.link_m} m · 걷기 ${P.walk_kmh} km/h · 물체당 ${P.item_min}분 + ${P.min_per_m2}분/m² · 마대 ${P.bag_kg} kg / ${P.bag_l} L · 1인 운반 ${P.carry_kg_per_person} kg·${P.carry_bags_per_person}마대 · 숲 통과 ×${P.veg_cost} · 보트 ×${P.boat_cost}${T?'':' · 우회 ×'+P.detour}`+(Object.values(ST.calib).some(v=>v!==1)?` · 실측 보정 ${Object.entries(ST.calib).filter(([c,v])=>v!==1).map(([c,v])=>`${D.mats[c]?D.mats[c].ko:c} ×${v.toFixed(2)}`).join(', ')}`:'')+' (가정값)';
}
let CUR=null, timer=null;
function recompute(){ const busy=$('#busy'); busy.style.display='flex'; setTimeout(()=>{ try{ readControls(); CUR=computePlan(opts()); renderTiles(CUR); renderMap(CUR); renderSteps(CUR); if (typeof maybePush==='function') maybePush(false); } catch(e){ console.error(e); $('#status').textContent='계산 오류: '+e.message; } busy.style.display='none'; },10); }
window.recompute=recompute;
window.scheduleRecompute=()=>{ clearTimeout(timer); timer=setTimeout(recompute,150); };
window.resetAll=()=>{ for(const [id,v] of Object.entries(D.control_defaults)){ const el=document.getElementById(id); if(!el)continue; if(el.type==='checkbox') el.checked=v; else el.value=v; } document.querySelectorAll('.codes input').forEach(e=>e.checked=true); for(const [n,v] of Object.entries(D.seg_defaults)) setSeg(n,v); depot=Object.assign({},defaultDepot); recompute(); if(map) fitAll(); };
// 진행 체크
window.toggleDone=(ids)=>{ for(const id of ids){ if(ST.done.has(id)) ST.done.delete(id); else ST.done.add(id); } persist(); recompute(); };
window.doneZone=(step)=>{ if(!CUR)return; const z=CUR.zones.find(z=>z.step===step); if(!z)return; z.objects.forEach(ob=>ST.done.add(ob.id)); persist(); recompute(); };
window.resetDone=()=>{ ST.done.clear(); persist(); recompute(); };
// 내 위치
window.locateMe=()=>{ if(!navigator.geolocation){ $('#status').textContent='이 기기에서는 위치를 쓸 수 없습니다'; return; }
  navigator.geolocation.getCurrentPosition(pos=>{ const lat=pos.coords.latitude, lon=pos.coords.longitude; if(map){ if(!meMarker){ meMarker=L.circleMarker([lat,lon],{radius:9,color:'#fff',weight:3,fillColor:'#e34948',fillOpacity:1}).addTo(map); meMarker.bindTooltip('내 위치'); } else meMarker.setLatLng([lat,lon]); }
    if(!CUR||!CUR.zones.length){ $('#status').textContent='내 위치 표시'; if(map) map.flyTo([lat,lon],17); return; }
    const xy=ll2xy(lon,lat); let best=null;
    if (T){ forcePassable(xy); const {dist}=dijkstra(CUR.mode, cellOf(xy[0],xy[1])); for(const z of CUR.zones){ const d=z.firstCell!=null?dist[z.firstCell]:Infinity; if(isFinite(d)&&(best===null||d<best.d)) best={z,d}; } }
    if(!best){ for(const z of CUR.zones){ const d=Math.hypot(z.cx-xy[0],z.cy-xy[1])*P.detour; if(best===null||d<best.d) best={z,d}; } }
    const min=best.d/1000/P.walk_kmh*60; $('#status').textContent=`내 위치에서 가장 가까운 구역: ${best.z.step}. ${best.z.name} · ${Math.round(best.d)} m · 약 ${Math.round(min)}분`;
    if(map){ map.flyToBounds(L.latLngBounds([[lat,lon],[best.z.lat,best.z.lon]]).pad(0.3)); } highlight(best.z.step,false);
  }, err=>{ $('#status').textContent='위치를 가져오지 못했습니다 ('+err.message+'). HTTPS 주소에서, 위치 권한을 허용해야 합니다'; }, {enableHighAccuracy:true, timeout:10000});
};
// 실측 보정
window.applyMeasured=()=>{ if(!CUR)return; const step=parseInt($('#cal-zone').value,10); const kg=parseFloat($('#cal-kg').value); const z=CUR.zones.find(z=>z.step===step); if(!z||!isFinite(kg)||kg<=0){ $('#cal-msg').textContent='구역과 실측 무게(kg)를 입력하세요'; return; }
  const base=z.objects.reduce((s,ob)=>s+ob.kg/((ST.calib[ob.code]||1)),0); if(base<=0){ $('#cal-msg').textContent='이 구역의 예상 무게가 0 입니다'; return; }
  const f=kg/base; const codes=[...new Set(z.objects.map(ob=>ob.code))]; for(const c of codes){ ST.calib[c]=Math.round(f*100)/100; const el=$('#c-cal-'+c); if(el) el.value=ST.calib[c]; } persist();
  $('#cal-msg').textContent=`구역 ${step} 예상 ${fmtKg(base)} kg → 실측 ${fmtKg(kg)} kg : ${codes.map(c=>D.mats[c]?D.mats[c].ko:c).join('·')} 보정 계수 ×${f.toFixed(2)} 적용`; recompute(); };
window.resetCalib=()=>{ for(const c of Object.keys(ST.calib)){ ST.calib[c]=1; const el=$('#c-cal-'+c); if(el) el.value=1; } persist(); $('#cal-msg').textContent='보정 계수를 1로 되돌렸습니다'; recompute(); };
// 시나리오 비교 + 민감도
window.compareScenarios=()=>{ readControls(); const cur=opts(); const rows=[['현재 설정',cur]];
  if(T) rows.push(['도보 · 현장 적치',opts({travel:'walk',carry:'pile'})],['보트 지원 · 현장 적치',opts({travel:'boat',carry:'pile'})]);
  rows.push([`${cur.workers*2}명 (인원 2배)`,opts({workers:cur.workers*2})],['팀 2개',opts({teams:2})],['들고 이동',opts({carry:'carry'})],['무게 우선 순서',opts({objective:'weight'})],['무게 추정 최소값',opts({wsrc:'ours',wstat:'min'})],['무게 추정 최대값',opts({wsrc:'ours',wstat:'max'})],['무게 ×0.5 (민감도)',opts({scale:0.5})],['무게 ×1.5 (민감도)',opts({scale:1.5})]);
  const out=rows.map(([name,o])=>{ const R=computePlan(o); return `<tr><td>${esc(name)}</td><td class="num">${R.totals.zones}</td><td class="num">${fmtKg(R.totals.kg)}</td><td class="num">${R.totals.bags}</td><td class="num">${(R.routeLen/1000).toFixed(1)}</td><td class="num">${fmtMin(R.totalMin)}</td><td class="num">${Math.max(0,...R.teams.map(x=>x.days))}</td><td class="num">${R.totals.heavy}</td></tr>`; }).join('');
  $('#scen').innerHTML=`<table><thead><tr><th>시나리오</th><th class="num">구역</th><th class="num">예상 kg</th><th class="num">마대</th><th class="num">이동 km</th><th class="num">총 시간</th><th class="num">일수</th><th class="num">2인 운반</th></tr></thead><tbody>${out}</tbody></table><p class="sub">같은 완료·종류·고급 설정에서 조건 하나씩만 바꾼 결과. 민감도 행은 모든 무게에 배수를 곱해 가정값(채움률·두께·밀도)의 영향을 본 것.</p>`; };
window.downloadCsv=()=>{ if(!CUR)return; const rows=[['순서','팀','일차','구역','개수','구성','계획 무게(kg)','최소(kg)','최대(kg)','기업값(kg)','부피(m³)','마대','접근(m)','이동(분)','작업(분)','누적(분)','복귀','주의']];
  for(const z of CUR.zones) rows.push([z.step,teamName(z.team),z.day,z.name,z.n,Object.entries(z.byCode).map(([c,n])=>`${D.mats[c]?D.mats[c].ko:c} ${n}`).join(' '),z.kg.toFixed(2),z.kmin.toFixed(2),z.kmax.toFixed(2),z.ckg.toFixed(3),z.m3.toFixed(3),z.bags,Math.round(z.dIn),z.walkMin.toFixed(1),z.workMin.toFixed(1),z.cumMin.toFixed(1),z.returns,z.notes.join(' / ')]);
  const csv='﻿'+rows.map(r=>r.map(v=>`"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n'); const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})); a.download='수거계획_작업순서.csv'; a.click(); };
window.downloadJson=()=>{ if(!CUR)return; const out={params:P, depot, options:CUR.o, calib:ST.calib, done:[...ST.done], totals:CUR.totals, teams:CUR.teams.map(t=>({team:t.team,zones:t.zones,n:t.n,kg:t.kg,bags:t.bags,minutes:t.minutes,days:t.days})), days:CUR.days, zones:CUR.zones.map(z=>({step:z.step,team:z.team,day:z.day,name:z.name,n:z.n,kg:z.kg,kmin:z.kmin,kmax:z.kmax,bags:z.bags,dIn:z.dIn,distTot:z.distTot,walkMin:z.walkMin,workMin:z.workMin,returns:z.returns,objects:z.objects.map(ob=>ob.id),notes:z.notes}))};
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'})); a.download='수거계획_설정결과.json'; a.click(); };
document.querySelectorAll('.ctrl input, .ctrl select').forEach(el=>el.addEventListener('input',scheduleRecompute));
document.querySelectorAll('.seg button').forEach(b=>b.addEventListener('click',()=>{ setSeg(b.parentElement.dataset.name,b.dataset.v); recompute(); }));
for (const [c,v] of Object.entries(ST.calib)) { const el=$('#c-cal-'+c); if (el) el.value=v; }
window.toggleSide=()=>{ const l=$('#layout'); l.classList.toggle('collapsed'); try{ localStorage.setItem(LS_KEY+':side', l.classList.contains('collapsed')?'1':'0'); }catch(e){} setTimeout(()=>{ if(map){ map.invalidateSize(); } },250); };
try{ if(localStorage.getItem(LS_KEY+':side')==='1') $('#layout').classList.add('collapsed'); }catch(e){}

// ───────── 공유 동기화 (claude.ai 공개 링크: db 캐퍼빌리티) ─────────
// 완료 체크·실측 보정 계수·출발지를 모든 접속자가 같이 본다. 다른 사람이 바꾸면 onSnapshot 으로 바로 반영.
let SHDOC=null, SH_READONLY=false, SH_LAST='', SH_TIMER=null, SH_WRITING=null, SH_USER=null, SH_NAMES={};
const sharedState=()=>({done:[...ST.done].sort(), calib:ST.calib, depot:{lon:depot.lon,lat:depot.lat,name:depot.name}});
const sharedSig=(o)=>JSON.stringify([o.done, o.calib, o.depot&&[+o.depot.lon.toFixed(6), +o.depot.lat.toFixed(6)]]);
function syncStatus(msg, ok){ const el=$('#sync'); if(!el) return; el.textContent=msg; el.style.color = ok===false ? '#c0392b' : ''; }
// Supabase 백엔드 (GitHub Pages 등 일반 호스팅): 테이블 shared_state(site, data jsonb, updated_at, updated_by)
let SB=null, SB_ROW=null, SB_POLL=null;
const myName=()=>{ try{ return localStorage.getItem('shoresweep:name')||''; }catch(e){ return ''; } };
window.setMyName=(v)=>{ try{ localStorage.setItem('shoresweep:name', v||''); }catch(e){} };
function applyRemoteBody(d, meta){
  const sig=sharedSig({done:d.done||[], calib:d.calib||{}, depot:d.depot||depot});
  const when = d.updatedAt ? new Date(d.updatedAt).toLocaleTimeString('ko',{hour:'2-digit',minute:'2-digit'}) : '';
  syncStatus(`공유 동기화 켜짐 (${meta})${when?' · 마지막 변경 '+when:''}${d.byName?' · '+d.byName:''}`);
  if (sig===SH_LAST) return;
  SH_LAST=sig;
  ST.done=new Set(d.done||[]); ST.calib=Object.assign({}, d.calib||{});
  for (const c of Object.keys(D.mats)) { const el=$('#c-cal-'+c); if (el) el.value = ST.calib[c]||1; }
  if (d.depot && isFinite(d.depot.lon) && isFinite(d.depot.lat)) { depot={lon:d.depot.lon, lat:d.depot.lat, name:d.depot.name||'지정 출발지'}; if (depotMarker) depotMarker.setLatLng([depot.lat,depot.lon]); }
  persist(); recompute();
}
async function initSupabase(){
  const cfg=D.supabase; if(!cfg) return false;
  if (typeof supabase==='undefined' || !supabase.createClient){ syncStatus('Supabase 라이브러리를 불러오지 못했습니다 (인터넷 확인) · 이 브라우저에만 저장', false); return true; }
  try { SB=supabase.createClient(cfg.url, cfg.key); } catch(e){ syncStatus('Supabase 연결 실패: '+e.message, false); return true; }
  const site=D.site||'site';
  syncStatus('Supabase 연결 중…');
  const load=async()=>{ const {data,error}=await SB.from(cfg.table).select('data,updated_at,updated_by').eq('site',site).maybeSingle();
    if (error){ syncStatus('Supabase 읽기 실패: '+error.message+' (tools/supabase_setup.sql 실행·권한 확인)', false); return; }
    if (!data){ syncStatus('공유 동기화 켜짐 (Supabase) · 아직 저장된 변경 없음'); maybePush(true); return; }
    const body=Object.assign({}, data.data||{}, {updatedAt: data.updated_at ? Date.parse(data.updated_at) : null, byName: data.updated_by||''});
    applyRemoteBody(body, 'Supabase'); };
  await load();
  try {
    SB.channel('shared_state_'+site).on('postgres_changes', {event:'*', schema:'public', table:cfg.table, filter:'site=eq.'+site}, payload=>{ const r=payload.new; if(!r||!r.data) return; applyRemoteBody(Object.assign({}, r.data, {updatedAt: r.updated_at?Date.parse(r.updated_at):null, byName:r.updated_by||''}), 'Supabase 실시간'); }).subscribe();
  } catch(e){}
  SB_POLL=setInterval(load, 15000);          // 실시간이 꺼져 있어도 15 초마다 맞춤
  SHDOC = { set: async (body)=>{ const {error}=await SB.from(cfg.table).upsert({site, data:{done:body.done, calib:body.calib, depot:body.depot}, updated_at:new Date().toISOString(), updated_by: myName()||'이름 없음'}, {onConflict:'site'}); if (error){ const err=new Error(error.message); err.code = /permission|policy|row-level/i.test(error.message)?'invalid_argument':'unavailable'; throw err; } } };
  return true;
}
async function initShared(){
  if (D.supabase){ await initSupabase(); return; }
  if (!D.shared || !window.claude || typeof window.claude.use!=='function'){ return; }
  syncStatus('공유 저장소 연결 중…');
  let db=null, user=null;
  try { [db, user] = await Promise.all([window.claude.use('db'), window.claude.use('user')]); } catch(e){ db=null; }
  if (!db){ syncStatus('공유 저장 사용 불가 (로그인 필요) · 이 브라우저에만 저장', false); return; }
  SH_USER=user;
  try { if (user && user.can) { const w = await user.can('data.write'); if (w===false){ SH_READONLY=true; } } } catch(e){}
  SHDOC = db.doc('state/shared');
  SHDOC.onSnapshot(snap=>{
    if (!snap.exists){ syncStatus('공유 동기화 켜짐 (아직 저장된 변경 없음)'+(SH_READONLY?' · 읽기 전용':'')); if(!SH_READONLY) maybePush(true); return; }
    const d=snap.data(); const sig=sharedSig({done:d.done||[], calib:d.calib||{}, depot:d.depot||depot});
    const when = d.updatedAt ? new Date(d.updatedAt).toLocaleTimeString('ko',{hour:'2-digit',minute:'2-digit'}) : '';
    const byName = d.by ? (SH_NAMES[d.by] || '다른 사용자') : '';
    syncStatus(`공유 동기화 켜짐${when?' · 마지막 변경 '+when:''}${byName?' · '+byName:''}${SH_READONLY?' · 읽기 전용(변경은 이 브라우저에만)':''}${snap.metadata&&snap.metadata.hasPendingWrites?' · 저장 중':''}`);
    if (d.by && SH_USER && SH_USER.profiles && !SH_NAMES[d.by]) { SH_USER.profiles([d.by]).then(ps=>{ const nm=(ps&&ps[d.by]&&ps[d.by].name)||''; if(nm){ SH_NAMES[d.by]=nm; syncStatus(`공유 동기화 켜짐${when?' · 마지막 변경 '+when:''} · ${nm}${SH_READONLY?' · 읽기 전용':''}`); } }).catch(()=>{}); }
    if (sig===SH_LAST) return;                       // 내가 보낸 것 또는 이미 반영된 것
    SH_LAST=sig;
    ST.done=new Set(d.done||[]); ST.calib=Object.assign({}, d.calib||{});
    for (const c of Object.keys(D.mats)) { const el=$('#c-cal-'+c); if (el) el.value = ST.calib[c]||1; }
    if (d.depot && isFinite(d.depot.lon) && isFinite(d.depot.lat)) { depot={lon:d.depot.lon, lat:d.depot.lat, name:d.depot.name||'지정 출발지'}; if (depotMarker) depotMarker.setLatLng([depot.lat,depot.lon]); }
    persist(); recompute();
  }, err=>{ syncStatus('공유 동기화 중단 ('+(err&&err.code||'오류')+') · 이 브라우저에만 저장', false); SHDOC=null; });
}
function maybePush(force){
  if (!SHDOC || SH_READONLY) return;
  const st=sharedState(); const sig=sharedSig(st);
  if (!force && sig===SH_LAST) return;
  SH_LAST=sig; clearTimeout(SH_TIMER);
  SH_TIMER=setTimeout(async()=>{
    const body=Object.assign({}, st, {updatedAt: Date.now(), by: null});
    try { if (SH_USER && SH_USER.id) body.by = await SH_USER.id(); } catch(e){}
    const run=async()=>{ try { await SHDOC.set(body); } catch(e){ const code=e&&e.code; if (code==='invalid_argument'){ SH_READONLY=true; syncStatus('읽기 전용: 공유 변경 권한이 없어 이 브라우저에만 저장됩니다 (공유 메뉴에서 Contributor 이상으로 초대 필요)', false); } else if (code==='unavailable'){ setTimeout(()=>{ SHDOC && SHDOC.set(body).catch(()=>{}); }, 800+Math.random()*700); } else { syncStatus('공유 저장 실패: '+(code||e), false); } } };
    SH_WRITING = (SH_WRITING||Promise.resolve()).then(run, run);
  }, 400);
}
initShared();
if (D.supabase){ const ni=$('#c-name'); if(ni){ ni.style.display='block'; ni.value=myName(); } }
initMap(); recompute();
})();
"""


def _control_panel(d: dict, mats: dict, terrain: bool, present_codes: list[str], artifact: bool = False) -> str:
    esc = _esc
    def seg(name, opts, cur):
        return f'<div class="seg" data-name="{name}">' + "".join(
            f'<button data-v="{v}" class="{"on" if v == cur else ""}">{esc(t)}</button>' for v, t in opts) + "</div>"
    codes = "".join(f'<label><input type="checkbox" value="{c}" checked><i style="background:{mats[c]["color"]}"></i>{esc(mats[c]["ko"])}</label>'
                    for c in present_codes)
    cal = "".join(f'<label><b>{esc(mats[c]["ko"])} 보정 계수</b><input type="number" id="c-cal-{c}" min="0.05" max="20" step="0.05" value="1"></label>'
                  for c in present_codes)
    travel_opts = [("walk", "도보"), ("boat", "보트 지원")] if terrain else [("walk", "도보 (지형 없음)")]
    return f"""
<div class="panel"><h2><span>조건 바꾸기 → 바로 다시 계산</span><span><span id="status">…</span><span id="sync" style="display:block;font-size:12px;color:var(--ok);font-weight:600"></span><input id="c-name" placeholder="내 이름 (변경 표시용)" style="display:none;font:inherit;font-size:12px;padding:3px 8px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--ink);width:150px;margin-top:4px" oninput="setMyName(this.value)"><button class="fold" onclick="toggleSide()" title="설정 접기">◀ 접기</button></span></h2>
<div class="ctrl">
 <label><b>팀당 인원</b><input type="number" id="c-workers" min="1" max="30" step="1" value="{d['workers']}"></label>
 <label><b>팀 수 (동시 투입)</b><input type="number" id="c-teams" min="1" max="6" step="1" value="{d['teams']}"></label>
 <label><b>하루 작업 시간</b><input type="number" id="c-hours" min="0.5" max="12" step="0.5" value="{d['hours_per_day']}"></label>
 <label><b>이동 방식</b>{seg('travel', travel_opts, d['travel'])}</label>
 <label><b>운반 방식</b>{seg('carry', [('pile', '현장 적치'), ('carry', '들고 이동')], d['carry'])}</label>
 <label><b>최적화 목표</b>{seg('objective', [('distance', '최단 이동'), ('weight', '무게 우선')], d['objective'])}</label>
 <label><b>무게 기준</b>{seg('wsrc', [('ours', '우리 추정'), ('company', '기업값')], d['weight_source'])}</label>
 <label><b>추정 범위 (우리 추정일 때)</b>{seg('wstat', [('min', '최소'), ('typ', '대표'), ('max', '최대')], d['weight_stat'])}</label>
 <label><b>최소 무게(kg) 미만 건너뜀</b><input type="number" id="c-minkg" min="0" step="0.1" value="{d['min_kg']}"></label>
 <label><b>완료한 구역 제외</b><input type="checkbox" id="c-exdone" checked style="width:22px;height:22px"></label>
 <label style="grid-column:1/-1"><b>수거할 종류</b><div class="codes">{codes}</div></label>
</div>
<details class="adv"><summary>고급 설정 (가정값) · 실측 보정 계수</summary><div class="ctrl" style="margin-top:8px">
 <label><b>구역 묶기 거리 (m)</b><input type="number" id="c-link" min="20" max="2000" step="10" value="{d['link_m']}"></label>
 <label><b>걷기 속도 (km/h)</b><input type="number" id="c-walk" min="1" max="6" step="0.1" value="{d['walk_kmh']}"></label>
 <label><b>물체당 시간 (분)</b><input type="number" id="c-item" min="0" max="30" step="0.5" value="{d['item_min']}"></label>
 <label><b>면적 1 m² 당 시간 (분)</b><input type="number" id="c-m2" min="0" max="20" step="0.1" value="{d['min_per_m2']}"></label>
 <label><b>1인 운반 무게 (kg)</b><input type="number" id="c-ckg" min="1" max="50" step="1" value="{d['carry_kg_per_person']}"></label>
 <label><b>1인 운반 마대 수</b><input type="number" id="c-cbags" min="1" max="10" step="1" value="{d['carry_bags_per_person']}"></label>
 <label><b>마대 적재 무게 (kg)</b><input type="number" id="c-bagkg" min="1" max="100" step="1" value="{d['bag_kg']}"></label>
 <label><b>마대 부피 (L)</b><input type="number" id="c-bagl" min="10" max="1000" step="10" value="{d['bag_l']}"></label>
 <label><b>숲 통과 배수</b><input type="number" id="c-veg" min="1" max="20" step="0.5" value="{d['veg_cost']}"></label>
 <label><b>보트 이동 배수 (물)</b><input type="number" id="c-boat" min="0.1" max="3" step="0.1" value="{d['boat_cost']}"></label>
 <label><b>우회 배수 (지형 없을 때)</b><input type="number" id="c-detour" min="1" max="3" step="0.1" value="{d['detour']}"></label>
 <label><b>출발지 왕복</b><input type="checkbox" id="c-round" {"checked" if d['round_trip'] else ""} style="width:22px;height:22px"></label>
 {cal}
</div></details>
<div class="btns"><button class="primary" onclick="recompute()">다시 계산</button><button onclick="resetAll()">초기화</button><button onclick="compareScenarios();document.getElementById('scen').scrollIntoView({{behavior:'smooth'}})">시나리오 비교</button>{'' if artifact else '<button onclick="downloadCsv()">작업순서 CSV</button><button onclick="downloadJson()">설정·결과 JSON</button><button onclick="window.print()">인쇄</button>'}</div>
</div>"""


def build_collect_html(plan: CollectPlan, out_html: str | Path, *, photos_dir: str | Path | None = None,
                       basemap: Basemap | None = None, static_png: str | Path | None = None, terrain=None,
                       center_xy: tuple[float, float] | None = None, title: str | None = None,
                       crs_m: str = "EPSG:5186", artifact: bool = False, shared_cfg: dict | None = None) -> Path:
    """plan 은 기본 파라미터로 만든 계획 (초기값·가정 문구용). 실제 숫자는 브라우저에서 다시 계산한다.
    artifact=True: 인터넷 공개용(claude.ai 아티팩트) 변형 — 문서 뼈대 없이, Leaflet CSS 인라인, 외부 타일 없이 드론 정사영상만,
    인쇄·내려받기 버튼 없음 (공개 뷰어에서 막힘). 공유 저장은 claude.ai db 캐퍼빌리티.
    shared_cfg: {"provider": "supabase", "url": ..., "anon_key": ..., "table": "shared_state"} 이면 일반(GitHub Pages) 버전에서
    Supabase 로 완료 체크·보정·출발지를 모든 접속자에게 실시간 공유 (tools/supabase_setup.sql 로 테이블 생성)."""
    title = title or f"{plan.site} 해안쓰레기 수거 작업 계획"
    photos_dir = Path(photos_dir) if photos_dir else None
    pp = plan.params

    photo: dict[str, str | None] = {}
    for o in plan.objects:
        p = None
        if photos_dir and o.image_path:
            cand = [photos_dir / Path(o.image_path).name, photos_dir / o.image_path, photos_dir / f"{o.obj_id}.png", photos_dir / f"{o.obj_id}.jpg"]
            p = next((c for c in cand if c.exists()), None)
        photo[o.obj_id] = _b64_file(p, max_px=260) if p else None

    objs_js = [{"id": o.obj_id, "code": o.code, "ko": o.class_ko, "color": material(o.code).color, "lon": o.lon, "lat": o.lat,
                "x": o.x_m, "y": o.y_m, "area": o.area_m2, "w": o.w_m, "h": o.h_m, "ckg": o.company_kg, "kmin": o.kg_min,
                "ktyp": o.kg_typ, "kmax": o.kg_max, "vol": o.volume_m3, "img": photo[o.obj_id]} for o in plan.objects]
    present_codes = sorted({o.code for o in plan.objects}, key=lambda c: -sum(1 for o in plan.objects if o.code == c))
    mats = materials_js(present_codes)

    lons = [o.lon for o in plan.objects] + [plan.depot["lon"]]; lats = [o.lat for o in plan.objects] + [plan.depot["lat"]]
    bm = None; terrain_png = None
    if basemap is not None:
        url, bounds = basemap.to_wgs84()
        bm = {"url": url, "bounds": list(bounds)}
        lons += [bounds[1], bounds[3]]; lats += [bounds[0], bounds[2]]
    affine = fit_affine(crs_m, (min(lons) - 0.01, max(lons) + 0.01), (min(lats) - 0.01, max(lats) + 0.01))
    terrain_js = None
    if terrain is not None:
        terrain_js = terrain.to_js()
        tmp = Basemap.from_array(terrain.class_image(1), {"x0": terrain.x0, "y0": terrain.y0, "px": terrain.cell, "py": -terrain.cell},
                                 (terrain.nc, terrain.nr), crs_m)
        url, bounds = tmp.to_wgs84(out_w=1200, quality=60, nodata_fn=lambda img: img.sum(axis=2) < 10)
        terrain_png = {"url": url, "bounds": list(bounds)}
    if center_xy is None:
        center_xy = (float(np.mean([o.x_m for o in plan.objects])), float(np.mean([o.y_m for o in plan.objects])))
    defaults = {"workers": pp["workers"], "teams": pp.get("teams", 1), "hours_per_day": pp["hours_per_day"], "link_m": pp["link_m"], "walk_kmh": pp["walk_kmh"],
                "item_min": pp["item_min"], "min_per_m2": pp["min_per_m2"], "heavy_extra_min": pp["heavy_extra_min"], "load_slow": pp["load_slow"],
                "carry_kg_per_person": pp["carry_kg_per_person"], "carry_bags_per_person": pp["carry_bags_per_person"],
                "min_kg": pp["min_kg"], "bag_kg": pp["bag"]["bag_kg"], "bag_l": round(pp["bag"]["bag_m3"] * 1000), "detour": pp["detour"],
                "veg_cost": 3.0, "boat_cost": 0.5, "round_trip": pp["round_trip"], "travel": pp["travel"], "carry": pp["carry"],
                "objective": pp["objective"], "weight_source": pp["weight_source"], "weight_stat": pp["weight_stat"]}
    if terrain is not None:
        from .terrain import COST, VEG, WATER
        defaults["veg_cost"] = COST["walk"][VEG]; defaults["boat_cost"] = COST["boat"][WATER]
    control_defaults = {"c-workers": defaults["workers"], "c-teams": defaults["teams"], "c-hours": defaults["hours_per_day"], "c-link": defaults["link_m"], "c-walk": defaults["walk_kmh"],
                        "c-item": defaults["item_min"], "c-m2": defaults["min_per_m2"], "c-ckg": defaults["carry_kg_per_person"],
                        "c-cbags": defaults["carry_bags_per_person"], "c-minkg": defaults["min_kg"], "c-bagkg": defaults["bag_kg"],
                        "c-bagl": defaults["bag_l"], "c-veg": defaults["veg_cost"], "c-boat": defaults["boat_cost"], "c-detour": defaults["detour"],
                        "c-round": defaults["round_trip"]}
    seg_defaults = {"travel": defaults["travel"], "carry": defaults["carry"], "objective": defaults["objective"],
                    "wsrc": defaults["weight_source"], "wstat": defaults["weight_stat"]}
    data = {"site": plan.site, "survey": plan.survey_date, "depot": plan.depot, "objects": objs_js, "mats": mats,
            "terrain": terrain_js, "terrain_png": terrain_png, "affine": affine, "basemap": bm, "day_colors": DAY_COLORS,
            "center": {"x": center_xy[0], "y": center_xy[1]},
            "bag": {"bulk": pp["bag"]["bulk_factor"], "tonbag_kg": pp["bag"]["tonbag_kg"], "tonbag_m3": pp["bag"]["tonbag_m3"]},
            "defaults": defaults, "control_defaults": control_defaults, "seg_defaults": seg_defaults, "no_tiles": artifact, "shared": artifact,
            "supabase": ({"url": shared_cfg["url"], "key": shared_cfg["anon_key"], "table": shared_cfg.get("table", "shared_state")}
                         if (shared_cfg and not artifact and shared_cfg.get("provider") == "supabase" and shared_cfg.get("url") and shared_cfg.get("anon_key")) else None)}
    fallback = _b64_file(Path(static_png), max_px=2200) if static_png and Path(static_png).exists() else None
    leaflet_css = Path(__file__).with_name("assets") / "leaflet.css"
    if artifact:
        css_block = (f"<style>{leaflet_css.read_text(encoding='utf-8')}</style>" if leaflet_css.exists() else "")
        head = (f'<meta charset="utf-8"><title>{_esc(title)}</title>\n<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>\n'
                f'{css_block}<style>{CSS}\n#map{{background:#8fb3c9}}.leaflet-control-layers-toggle{{background-image:none!important;'
                f'width:auto!important;height:auto!important;padding:4px 8px;font-size:13px}}.leaflet-control-layers-toggle::after{{content:"레이어"}}</style>\n<div class="wrap">')
        tail = "</div>"
    else:
        head = (f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
{'<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.min.js" crossorigin=""></script>' if data.get("supabase") else ''}
<style>{CSS}</style></head><body><div class="wrap">""")
        tail = "</body></html>"

    H = [head + f"""
<header><h1>{_esc(title)}</h1><p>조사일 {_esc(plan.survey_date)} · 출발·집결지 <b>{_esc(plan.depot['name'])}</b> (지도의 ★ 를 끌어서 바꿀 수 있음) · {'지형(물·숲·맨땅) 최단경로 반영' if terrain is not None else '직선 거리 × 우회 배수'}</p></header>""",
         '<div class="layout" id="layout"><aside class="side"><button class="side-open" onclick="toggleSide()">설정 열기 ▶</button>' + _control_panel(defaults, mats, terrain is not None, present_codes, artifact) + '</aside><main class="main">',
         '<div class="tiles" id="tiles"></div>',
         '<div class="card"><h2>수거 지도</h2><p class="sub" id="maphint">번호 순서대로 이동합니다. 번호를 누르면 그 구역에서 수거할 물체 목록(사진·종류·크기·무게)이 패널로 뜨고, 점을 누르면 물체 하나의 정보가 나옵니다. ★ 출발지는 끌어서 옮기면 경로가 다시 계산됩니다 (정사영상 안의 땅·해안 20 m 이내에만 놓을 수 있음).</p>',
         '<div class="mapbtns" id="daybtns"></div>',
         f'<div id="map"><img id="fallback" class="fallback" style="display:none" src="{fallback or ""}" alt="수거 지도"><div class="busy" id="busy">계산 중…</div><div class="zpanel" id="zpanel"></div></div>',
         '<div class="legend" id="legend"></div></div>',
         '<div class="card"><h2>작업 순서</h2><p class="sub">출발지에서 번호 순서대로 돕니다. 시간은 이동 + 줍기·담기 예상값입니다. 카드를 누르면 지도가 그 구역으로 가고, 완료 ✓ 를 누르면 남은 구역만으로 다시 계산됩니다. 길찾기 링크는 휴대폰에서 지도 앱으로 열립니다.</p><div id="steps"></div></div>',
         '<div class="card"><h2>진행 현황</h2><p class="sub">구역 카드의 "완료 ✓" 를 누르면 그 구역이 완료 처리되고(이 브라우저에 저장; claude.ai 공개 링크에서는 접속한 모두에게 바로 공유), 남은 구역만으로 경로·시간이 다시 계산됩니다. 지도의 점을 눌러 물체 하나씩 완료할 수도 있습니다.</p><div id="progress"></div></div>\n<div class="card"><h2>실측 보정</h2><p class="sub">현장에서 한 구역의 마대를 저울로 재면, 예상 무게와 비교해 그 재질의 보정 계수를 자동으로 구하고 모든 구역에 적용합니다 (우리 추정 무게에만 곱함, 이 브라우저에 저장).</p>\n<div class="cal"><label><b>실측한 구역</b><select id="cal-zone"></select></label><label><b>실측 무게 (kg)</b><input type="number" id="cal-kg" min="0" step="0.1" placeholder="예: 12.5"></label><label><b>&nbsp;</b><button class="primary" style="font:inherit;padding:8px 12px;border-radius:9px;border:0;background:var(--accent);color:#fff;cursor:pointer" onclick="applyMeasured()">보정 계수 계산·적용</button></label><label><b>&nbsp;</b><button style="font:inherit;padding:8px 12px;border-radius:9px;border:1px solid var(--border);background:var(--card);color:var(--ink);cursor:pointer" onclick="resetCalib()">보정 해제</button></label></div><div id="cal-msg"></div></div>\n<div class="card"><h2>시나리오 비교 · 민감도</h2><p class="sub">버튼을 누르면 현재 설정에서 조건을 하나씩 바꾼 결과를 한 표로 보여 줍니다.</p><div class="btns" style="margin:0 0 10px"><button class="primary" onclick="compareScenarios()">시나리오 비교 계산</button></div><div id="scen" style="overflow:auto"></div></div>',
         '<div class="card"><h2>준비물 체크리스트</h2><ul class="check" id="equip"></ul></div>',
         '<div class="card"><h2>종류별 요약과 다루는 법</h2><table><thead><tr><th>종류</th><th class="num">개수</th><th class="num">면적 m²</th><th class="num">예상 kg</th><th class="num">범위 kg</th><th class="num">기업 제공 kg</th><th>다루는 법</th></tr></thead><tbody id="bycode"></tbody></table></div>',
         '<div class="card"><details><summary>전체 쓰레기 목록 (수거 순서)</summary><div style="overflow:auto;max-height:520px"><table><thead><tr><th>순서</th><th>ID</th><th>종류</th><th class="num">가로×세로 m</th><th class="num">면적 m²</th><th class="num">예상 kg</th><th class="num">범위</th><th class="num">기업 kg</th><th>비고</th></tr></thead><tbody id="objtab"></tbody></table></div></details></div>',
         '<div class="card"><h2>이 숫자는 어떻게 나왔나 (가정과 한계)</h2><ul class="note">' + "".join(f"<li>{_esc(a)}</li>" for a in plan.assumptions) +
         f'<li>무게 계산식 예: {_esc(plan.objects[0].note) if plan.objects else ""}</li><li id="assume-params"></li></ul></div>',
         '</main></div><p class="foot">드론대장 붕붕이 · litter3d collect · 파일: plan.json, zones.csv, objects.csv, 수거계획.xlsx, 수거계획_지도.png (기본 설정 기준)</p></div>',
         f'<script>window.PLAN={json.dumps(data, ensure_ascii=False)};</script><script>{JS_APP}</script>' + tail]
    out = Path(out_html); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(H), encoding="utf-8")
    return out
