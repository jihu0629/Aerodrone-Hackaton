"""
기업 배포 데이터(문갑도 MGD) 추출기 — extract_company_data.py
================================================================
VS Code에서 이 파일을 열고 오른쪽 위 ▶(Run Python File) 버튼만 누르면 됩니다.
(이 파일은 '항공드론 해커톤  찐' 폴더 바로 아래에 두세요.)

입력
  기업 배포 데이터/MGD_쓰레기.json          ← 쓰레기 라벨 42개 (GeoJSON, 위경도 사각형 폴리곤)
  기업 배포 데이터/정사영상/7_2_문갑도.ecw  ← 정사영상 (ECW, 3 cm/px, EPSG:5186)
  기업 배포 데이터/crops/*.jpg              ← 기업이 잘라 준 쓰레기 사진

출력 (기본: C:\\hackathon\\MGD_out  — 한글 없는 경로)
  objects.csv / objects.xlsx    물체별 표 (좌표·크기·면적·무게·픽셀 위치)
  labels_pixel.json             정사영상 픽셀 좌표로 바꾼 라벨
  summary.md                    요약 + 데이터에서 발견한 점 (기업 질문용)
  overview.jpg / overview_labeled.jpg   정사영상 전체 축소본 + 라벨 위치
  chips/, chips_overlay/        물체마다 원해상도 잘라낸 사진 (+윤곽선, 오른쪽에 기업 crop 나란히)
  yolo_dataset/                 YOLO-seg 학습용 1024px 타일 + 라벨 + data.yaml
  tiles_preview.jpg             타일·라벨이 맞게 들어갔는지 확인용

정사영상(ECW) 읽기
  pip 로 설치하는 rasterio/GDAL 에는 ECW 드라이버가 없습니다.
  이 스크립트는 아래 순서로 ECW 를 읽을 수 있는 GDAL 을 자동으로 찾습니다.
    1) 파이썬 osgeo(GDAL) 에 ECW 드라이버가 있으면 사용
    2) PATH 의 gdal_translate
    3) QGIS / OSGeo4W 설치 폴더의 gdal_translate (o4w_env.bat 환경 사용)
  전부 없으면 라벨 표·요약만 만들고, QGIS 설치 방법을 안내합니다.
  (QGIS 의 ECW 지원 여부는 설치 버전에 따라 다를 수 있어 실행 시 자동으로 확인합니다.)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────── 설정 (필요하면 여기만 수정) ───────────────────────────────
ROOT = Path(__file__).resolve().parents[1]          # 저장소 루트
DATA_DIR = ROOT / "input"                           # 기업 자료를 넣는 폴더
LABELS_JSON = DATA_DIR / "labels.json"              # 기업 GeoJSON 라벨
ORTHO = next(iter(sorted((DATA_DIR / "ortho").glob("*.ecw"))), DATA_DIR / "ortho" / "ortho.ecw")   # 원본 ECW
CROPS_DIR = DATA_DIR / "crops"

OUT_DIR = ROOT / "outputs" / "extract"
SRC_COPY_DIR = Path(r"C:\hackathon\MGD_src") if os.name == "nt" else OUT_DIR / "src"   # 한글 경로 문제 회피용 복사 위치
OVERVIEW_OUT = DATA_DIR / "ortho" / "overview.jpg"  # 수거 계획 프로그램이 읽는 축소본 (+ .jgw 월드파일)

CHIP_MARGIN_M = 1.5        # 물체 사진 자를 때 주변 여유 (m)
TILE = 1024                # YOLO 타일 크기 (px). 3 cm/px 이면 약 31 m
NEG_TILES = 20             # 쓰레기 없는 배경 타일 개수 (오검출 줄이기용)
OVERVIEW_PCT = 3           # 전체 축소본 크기 (원본 대비 %)
RANDOM_SEED = 0
GDAL_TRANSLATE = None      # 직접 지정하려면 예: r"C:\Program Files\QGIS 3.40.0\bin\gdal_translate.exe"

# 재질 코드 → (YOLO 클래스 이름, 한글 설명, 우리 litter3d 클래스)
CLASS_INFO = {
    "STY": ("styrofoam", "스티로폼", "styrofoam_fragment"),
    "ROP": ("rope", "로프", "rope"),
    "FIS": ("fishing_gear", "어구 (그물·통발 등, 코드 의미 확인 필요)", "net"),
    "PLA": ("plastic", "플라스틱", "other_plastic"),
}
COLORS = {"STY": (255, 255, 255), "ROP": (255, 200, 0), "FIS": (0, 220, 180), "PLA": (255, 60, 60)}


# ─────────────────────────────── 패키지 준비 ───────────────────────────────
def ensure(pkgs: dict[str, str]):
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[설치] {' '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "-q"])


ensure({"pyproj": "pyproj", "PIL": "pillow", "numpy": "numpy", "openpyxl": "openpyxl"})

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def font(size=14):
    for f in [r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


# ─────────────────────────────── 1. 좌표계 · 월드파일 ───────────────────────────────
def read_world_file(ortho: Path):
    """ECW 월드파일(.eww): A, D, B, E, C, F → (원점 x, y, 픽셀 크기)."""
    for ext in (".eww", ".ecww", ".wld"):
        p = ortho.with_suffix(ext)
        if p.exists():
            v = [float(x) for x in p.read_text().split()]
            a, d, b, e, c, f = v[:6]
            if abs(d) > 1e-12 or abs(b) > 1e-12:
                print("  ⚠ 월드파일에 회전값이 있음 — 무시하고 진행")
            # 월드파일 C,F 는 왼쪽 위 '픽셀 중심' → 모서리로 반 픽셀 이동
            return {"x0": c - a / 2, "y0": f - e / 2, "px": a, "py": e, "src": p.name}
    return None


def read_crs(ortho: Path) -> CRS:
    prj = ortho.with_suffix(".prj")
    if prj.exists():
        try:
            crs = CRS.from_wkt(prj.read_text())
            return crs
        except Exception:  # noqa: BLE001
            pass
    print("  ⚠ .prj 를 못 읽어 EPSG:5186 (중부원점) 으로 가정")
    return CRS.from_epsg(5186)


# ─────────────────────────────── 2. 라벨 읽기 ───────────────────────────────
def shoelace(pts):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def load_objects(gt, tr):
    d = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    objs = []
    for f in d["features"]:
        p = f["properties"]
        ring = f["geometry"]["coordinates"][0]
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        xy = [tr.transform(lon, lat) for lon, lat in ring]
        px = [((x - gt["x0"]) / gt["px"], (y - gt["y0"]) / gt["py"]) for x, y in xy]
        xs, ys = [q[0] for q in xy], [q[1] for q in xy]
        pxs, pys = [q[0] for q in px], [q[1] for q in px]
        cx, cy = tr.transform(p["center_lon"], p["center_lat"])
        area_poly = shoelace(xy)
        code = p.get("material_code", "UNK")
        o = {
            "id": f"MGD_{int(p['detection_seq']):04d}_{code}",
            "seq": int(p["detection_seq"]),
            "code": code,
            "class_ko": CLASS_INFO.get(code, (code, code, "unknown"))[1],
            "litter3d_class": CLASS_INFO.get(code, (code, code, "unknown"))[2],
            "survey_date": p.get("survey_date"),
            "lon": p["center_lon"], "lat": p["center_lat"],
            "x_m": round(cx, 3), "y_m": round(cy, 3),
            "width_m": round(max(xs) - min(xs), 3), "height_m": round(max(ys) - min(ys), 3),
            "area_sqm_given": p.get("area_sqm"), "area_sqm_polygon": round(area_poly, 3),
            "weight_kg_given": p.get("weight_kg"),
            "kg_per_m2": round(p["weight_kg"] / p["area_sqm"], 5) if p.get("area_sqm") else None,
            "px_xmin": int(math.floor(min(pxs))), "px_ymin": int(math.floor(min(pys))),
            "px_xmax": int(math.ceil(max(pxs))), "px_ymax": int(math.ceil(max(pys))),
            "image_path": p.get("image_path"),
            "_poly_px": px, "_props": p,
        }
        objs.append(o)
    objs.sort(key=lambda o: o["seq"])
    return objs


# ─────────────────────────────── 3. 정사영상 읽기 (GDAL 찾기) ───────────────────────────────
class Raster:
    """ECW 를 읽을 수 있는 GDAL 을 찾아 창(window) 잘라내기·축소본을 만든다."""

    def __init__(self, path: Path):
        self.path = path
        self.mode = None
        self.env = None
        self.exe = None
        self.width = self.height = None
        self.geo = None
        self._find()

    # -- 탐색 --
    def _find(self):
        # 1) python osgeo
        try:
            from osgeo import gdal  # type: ignore
            gdal.UseExceptions()
            ds = gdal.Open(str(self.path))
            if ds is not None:
                self.mode, self._gdal, self._ds = "osgeo", gdal, ds
                self.width, self.height = ds.RasterXSize, ds.RasterYSize
                g = ds.GetGeoTransform()
                self.geo = {"x0": g[0], "y0": g[3], "px": g[1], "py": g[5]}
                return
        except Exception:  # noqa: BLE001
            pass
        # 2) PATH / 3) QGIS·OSGeo4W
        for exe, env in self._cli_candidates():
            info = self._gdalinfo(exe, env)
            if info:
                self.mode, self.exe, self.env = "cli", exe, env
                self._parse_info(info)
                return

    def _cli_candidates(self):
        if GDAL_TRANSLATE:
            yield Path(GDAL_TRANSLATE), None
        p = shutil.which("gdal_translate")
        if p:
            yield Path(p), None
        if os.name == "nt":
            roots = sorted(glob.glob(r"C:\Program Files\QGIS*"), reverse=True) + \
                sorted(glob.glob(r"C:\OSGeo4W*")) + sorted(glob.glob(r"C:\Program Files\OSGeo4W*"))
            for r in roots:
                exe = Path(r) / "bin" / "gdal_translate.exe"
                bat = Path(r) / "bin" / "o4w_env.bat"
                if exe.exists():
                    yield exe, self._bat_env(bat) if bat.exists() else None

    @staticmethod
    def _bat_env(bat: Path):
        try:
            out = subprocess.run(f'cmd /c ""{bat}" >nul && set"', capture_output=True, text=True,
                                 shell=True, encoding="mbcs", errors="ignore").stdout
            env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            return env or None
        except Exception:  # noqa: BLE001
            return None

    def _gdalinfo(self, translate_exe: Path, env):
        info_exe = translate_exe.with_name("gdalinfo" + translate_exe.suffix)
        try:
            r = subprocess.run([str(info_exe), str(self.path)], capture_output=True, text=True,
                               env=env, errors="ignore", timeout=300)
            if r.returncode == 0 and "Size is" in r.stdout:
                return r.stdout
            if "not recognized" in (r.stderr + r.stdout) or "ECW" in r.stderr:
                print(f"  - {info_exe}: ECW 를 열 수 없음 ({r.stderr.strip()[:120]})")
        except Exception as e:  # noqa: BLE001
            print(f"  - {info_exe}: 실행 실패 ({e})")
        return None

    def _parse_info(self, info: str):
        m = re.search(r"Size is (\d+),\s*(\d+)", info)
        self.width, self.height = int(m.group(1)), int(m.group(2))
        o = re.search(r"Origin = \(([-\d.e+]+),([-\d.e+]+)\)", info)
        s = re.search(r"Pixel Size = \(([-\d.e+]+),([-\d.e+]+)\)", info)
        if o and s:
            self.geo = {"x0": float(o.group(1)), "y0": float(o.group(2)),
                        "px": float(s.group(1)), "py": float(s.group(2))}

    @property
    def ok(self):
        return self.mode is not None

    # -- 읽기 --
    def window(self, x, y, w, h, out: Path):
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(min(w, self.width - x)), int(min(h, self.height - y))
        if w <= 0 or h <= 0:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "osgeo":
            self._gdal.Translate(str(out), self._ds, format="PNG", srcWin=[x, y, w, h], bandList=[1, 2, 3])
        else:
            cmd = [str(self.exe), "-q", "-of", "PNG", "-b", "1", "-b", "2", "-b", "3",
                   "-srcwin", str(x), str(y), str(w), str(h), str(self.path), str(out)]
            subprocess.run(cmd, check=True, env=self.env, capture_output=True)
        for junk in out.parent.glob(out.name + ".aux.xml"):
            junk.unlink()
        return (x, y, w, h)

    def overview(self, pct: float, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "osgeo":
            self._gdal.Translate(str(out), self._ds, format="JPEG", widthPct=pct, heightPct=pct,
                                 bandList=[1, 2, 3], resampleAlg="average")
        else:
            cmd = [str(self.exe), "-q", "-of", "JPEG", "-b", "1", "-b", "2", "-b", "3",
                   "-outsize", f"{pct}%", f"{pct}%", "-r", "average", str(self.path), str(out)]
            subprocess.run(cmd, check=True, env=self.env, capture_output=True)
        for junk in out.parent.glob(out.name + ".aux.xml"):
            junk.unlink()


def ascii_copy_if_needed(src: Path) -> Path:
    """Windows 에서 GDAL 이 한글 경로를 못 읽는 경우가 있어, 한 번만 영문 경로로 복사."""
    if os.name != "nt" or str(src).isascii():
        return src
    SRC_COPY_DIR.mkdir(parents=True, exist_ok=True)
    dst = SRC_COPY_DIR / "MGD_ortho.ecw"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"  정사영상을 영문 경로로 복사 중 (처음 한 번, {src.stat().st_size / 1e6:.0f} MB) → {dst}")
        shutil.copy2(src, dst)
    for ext in (".eww", ".prj"):
        s = src.with_suffix(ext)
        if s.exists():
            shutil.copy2(s, dst.with_suffix(ext))
    return dst


# ─────────────────────────────── 4. 출력물 ───────────────────────────────
PUBLIC = ["id", "seq", "code", "class_ko", "litter3d_class", "survey_date", "lon", "lat", "x_m", "y_m",
          "width_m", "height_m", "area_sqm_given", "area_sqm_polygon", "weight_kg_given", "kg_per_m2",
          "px_xmin", "px_ymin", "px_xmax", "px_ymax", "image_path"]


def write_tables(objs, out: Path):
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC)
        w.writeheader()
        for o in objs:
            w.writerow({k: o[k] for k in PUBLIC})
    by = class_summary(objs)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        wb = Workbook()
        ws = wb.active
        ws.title = "objects"
        ws.append(PUBLIC)
        for o in objs:
            ws.append([o[k] for k in PUBLIC])
        ws2 = wb.create_sheet("by_class")
        cols = ["code", "class_ko", "count", "area_sqm_sum", "weight_kg_sum", "kg_per_m2_min", "kg_per_m2_max"]
        ws2.append(cols)
        for r in by:
            ws2.append([r[c] for c in cols])
        for s in (ws, ws2):
            for c in s[1]:
                c.font = Font(bold=True)
            s.freeze_panes = "A2"
        wb.save(out / "objects.xlsx")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ xlsx 저장 실패 ({e}) — csv 만 저장")
    pix = [{"id": o["id"], "code": o["code"], "polygon_px": [[round(x, 2), round(y, 2)] for x, y in o["_poly_px"]],
            "bbox_px": [o["px_xmin"], o["px_ymin"], o["px_xmax"], o["px_ymax"]],
            "weight_kg": o["weight_kg_given"], "area_sqm": o["area_sqm_given"]} for o in objs]
    (out / "labels_pixel.json").write_text(json.dumps(pix, ensure_ascii=False, indent=1), encoding="utf-8")


def class_summary(objs):
    g = defaultdict(list)
    for o in objs:
        g[o["code"]].append(o)
    rows = []
    for code, L in sorted(g.items(), key=lambda kv: -len(kv[1])):
        k = [o["kg_per_m2"] for o in L if o["kg_per_m2"] is not None]
        rows.append({"code": code, "class_ko": L[0]["class_ko"], "count": len(L),
                     "area_sqm_sum": round(sum(o["area_sqm_given"] or 0 for o in L), 2),
                     "weight_kg_sum": round(sum(o["weight_kg_given"] or 0 for o in L), 4),
                     "kg_per_m2_min": min(k) if k else None, "kg_per_m2_max": max(k) if k else None})
    return rows


def findings(objs, raster, gt_world):
    notes = []
    by = class_summary(objs)
    const = [r for r in by if r["kg_per_m2_min"] and r["kg_per_m2_max"] / r["kg_per_m2_min"] < 1.05]
    if const:
        s = ", ".join(f"{r['code']} {r['kg_per_m2_min']:.3f}" for r in const)
        notes.append(f"**무게가 '면적 × 재질별 고정계수'로 계산된 값으로 보임** (kg/m²: {s}). "
                     "실측 무게가 아닐 가능성 → 기업에 '무게 산정 방법'을 질문할 것. 실측이 아니면 검증 정답으로 쓰면 안 됨.")
    rects = sum(1 for o in objs if len(o["_poly_px"]) == 4)
    if rects == len(objs):
        notes.append("라벨 폴리곤이 전부 **사각형 4점** → 사실상 바운딩박스. 부피 계산용 마스크는 SAM 등으로 따로 만들어야 함.")
    diff = [abs(o["area_sqm_polygon"] - o["area_sqm_given"]) / o["area_sqm_given"]
            for o in objs if o["area_sqm_given"]]
    if diff:
        notes.append(f"제공 면적(area_sqm)과 폴리곤 면적 차이: 중앙값 {np.median(diff) * 100:.1f} %, 최대 {max(diff) * 100:.1f} % "
                     "(작으면 area_sqm 은 사각형 박스 면적)")
    seqs = [o["seq"] for o in objs]
    notes.append(f"detection_seq {min(seqs)}–{max(seqs)} 중 {len(seqs)}개만 있음 → 원래 검출 {max(seqs)}개 이상 중 일부만 배포된 것으로 보임 (나머지 존재 여부 질문).")
    notes.append("높이·부피·DSM 정보 없음 → 3D 부피는 직접 촬영/복원이 필요.")
    if raster and raster.ok and raster.geo and gt_world:
        dx = abs(raster.geo["x0"] - gt_world["x0"]) / abs(gt_world["px"])
        dy = abs(raster.geo["y0"] - gt_world["y0"]) / abs(gt_world["py"])
        if dx > 1 or dy > 1:
            notes.append(f"⚠ GDAL 원점과 .eww 원점이 {dx:.1f}, {dy:.1f} px 차이 → GDAL 값 사용")
    return notes


def write_summary(objs, raster, gt, gt_world, out: Path, stats: dict):
    by = class_summary(objs)
    L = ["# 문갑도(MGD) 기업 배포 데이터 추출 요약", ""]
    L += [f"- 라벨: {len(objs)}개, 조사일 {', '.join(sorted({o['survey_date'] for o in objs}))}",
          f"- 정사영상: {ORTHO.name}, 픽셀 {abs(gt['px']) * 100:.2f} cm, 좌표계 {stats.get('crs')}",
          f"- 정사영상 크기: {raster.width} × {raster.height} px" if raster and raster.ok else
          "- 정사영상: **ECW 를 읽을 GDAL 을 못 찾음** → 아래 '정사영상 읽기' 참고",
          f"- 총 면적 {sum(o['area_sqm_given'] or 0 for o in objs):.2f} m², 총 무게(기업 값) {sum(o['weight_kg_given'] or 0 for o in objs):.3f} kg", ""]
    L += ["## 재질별", "", "| 코드 | 의미 | 개수 | 면적 합 (m²) | 무게 합 (kg) | kg/m² |", "|---|---|---|---|---|---|"]
    for r in by:
        L.append(f"| {r['code']} | {r['class_ko']} | {r['count']} | {r['area_sqm_sum']} | {r['weight_kg_sum']} | "
                 f"{r['kg_per_m2_min']}–{r['kg_per_m2_max']} |")
    L += ["", "## 데이터에서 발견한 점 (기업 질문 후보)", ""] + [f"- {n}" for n in findings(objs, raster, gt_world)]
    L += ["", "## 만든 파일", ""] + [f"- {k}: {v}" for k, v in stats.items() if k != "crs"]
    if not (raster and raster.ok):
        L += ["", "## 정사영상 읽기 (ECW)", "",
              "1. QGIS(무료) 설치: https://qgis.org → 설치 후 이 스크립트를 다시 ▶ 실행 (자동으로 찾음)",
              "2. 그래도 안 되면 QGIS 에서 ECW 를 열어 보고, 열리면 '래스터 → 변환 → 변환(Translate)' 으로 GeoTIFF 저장 후",
              "   스크립트 맨 위 ORTHO 를 그 .tif 경로로 바꿔 실행",
              "3. QGIS 에서도 안 열리면 → 기업에 GeoTIFF(또는 COG) 형식으로 다시 요청"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────── 5. 잘라내기 · 타일 ───────────────────────────────
def make_chips(objs, R: Raster, gt, out: Path):
    m = int(CHIP_MARGIN_M / abs(gt["px"]))
    n = 0
    for o in objs:
        x0, y0 = o["px_xmin"] - m, o["px_ymin"] - m
        w, h = o["px_xmax"] - o["px_xmin"] + 2 * m, o["px_ymax"] - o["px_ymin"] + 2 * m
        raw = out / "chips" / f"{o['id']}.png"
        win = R.window(x0, y0, w, h, raw)
        if not win:
            print(f"  ⚠ {o['id']} 은 정사영상 범위 밖")
            continue
        im = Image.open(raw).convert("RGB")
        dr = ImageDraw.Draw(im)
        poly = [(x - win[0], y - win[1]) for x, y in o["_poly_px"]]
        dr.polygon(poly, outline=COLORS.get(o["code"], (255, 0, 255)), width=2)
        dr.text((3, 3), f"{o['code']} {o['weight_kg_given']} kg", fill=(255, 255, 0), font=font(12))
        (out / "chips_overlay").mkdir(exist_ok=True)
        # 기업이 준 crop 이 있으면 옆에 붙여서 위치가 맞는지 눈으로 확인
        given = DATA_DIR / (o["image_path"] or "")
        if o["image_path"] and given.exists():
            g = Image.open(given).convert("RGB")
            s = im.size[1] / g.size[1]
            g = g.resize((max(1, int(g.size[0] * s)), im.size[1]))
            both = Image.new("RGB", (im.size[0] + g.size[0] + 6, im.size[1]), (40, 40, 40))
            both.paste(im, (0, 0))
            both.paste(g, (im.size[0] + 6, 0))
            ImageDraw.Draw(both).text((im.size[0] + 9, 3), "기업 crop", fill=(0, 255, 255), font=font(12))
            im = both
        im.save(out / "chips_overlay" / f"{o['id']}.jpg", quality=92)
        n += 1
    return n


def clip_rect(poly, x0, y0, x1, y1):
    """Sutherland–Hodgman: 폴리곤을 타일 사각형으로 자르기."""
    def clip(pts, inside, inter):
        res = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if inside(p):
                if not inside(q):
                    res.append(inter(q, p))
                res.append(p)
            elif inside(q):
                res.append(inter(q, p))
        return res

    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    pts = list(poly)
    for inside, inter in [(lambda p: p[0] >= x0, lambda a, b: ix(a, b, x0)),
                          (lambda p: p[0] <= x1, lambda a, b: ix(a, b, x1)),
                          (lambda p: p[1] >= y0, lambda a, b: iy(a, b, y0)),
                          (lambda p: p[1] <= y1, lambda a, b: iy(a, b, y1))]:
        pts = clip(pts, inside, inter) if pts else pts
    return pts


def make_yolo(objs, R: Raster, out: Path):
    codes = [c for c in CLASS_INFO if any(o["code"] == c for o in objs)] + \
        sorted({o["code"] for o in objs} - set(CLASS_INFO))
    cid = {c: i for i, c in enumerate(codes)}
    root = out / "yolo_dataset"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    tiles = defaultdict(list)
    for o in objs:
        for ty in range(o["px_ymin"] // TILE, o["px_ymax"] // TILE + 1):
            for tx in range(o["px_xmin"] // TILE, o["px_xmax"] // TILE + 1):
                tiles[(tx, ty)].append(o)
    n_obj_tiles = 0
    previews = []
    for (tx, ty), L in sorted(tiles.items()):
        x0, y0 = tx * TILE, ty * TILE
        name = f"MGD_t{tx}_{ty}"
        win = R.window(x0, y0, TILE, TILE, root / "images" / f"{name}.png")
        if not win:
            continue
        lines = []
        for o in L:
            pts = clip_rect(o["_poly_px"], x0, y0, x0 + win[2], y0 + win[3])
            if len(pts) < 3 or shoelace(pts) < 4:
                continue
            coords = " ".join(f"{(x - x0) / win[2]:.6f} {(y - y0) / win[3]:.6f}" for x, y in pts)
            lines.append(f"{cid[o['code']]} {coords}")
        (root / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_obj_tiles += 1
        if len(previews) < 12:
            previews.append((root / "images" / f"{name}.png", lines))
    # 배경(음성) 타일
    rnd = random.Random(RANDOM_SEED)
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    n_neg, tries = 0, 0
    while xs and n_neg < NEG_TILES and tries < NEG_TILES * 30:
        tries += 1
        tx = rnd.randint(max(0, min(xs) - 3), max(xs) + 3)
        ty = rnd.randint(max(0, min(ys) - 3), max(ys) + 3)
        if (tx, ty) in tiles or tx * TILE >= R.width or ty * TILE >= R.height:
            continue
        p = root / "images" / f"MGD_bg_t{tx}_{ty}.png"
        if p.exists() or not R.window(tx * TILE, ty * TILE, TILE, TILE, p):
            continue
        a = np.asarray(Image.open(p).convert("RGB"))
        if (a.max(axis=2) < 8).mean() > 0.3 or a.std() < 3:  # 영상 바깥(검은색)·빈 영역 제외
            p.unlink()
            continue
        (root / "labels" / f"{p.stem}.txt").write_text("")
        n_neg += 1
    names = "\n".join(f"  {i}: {CLASS_INFO.get(c, (c,))[0]}" for c, i in cid.items())
    (root / "data.yaml").write_text(
        f"# 문갑도 기업 라벨 → YOLO-seg. 학습 전에 train/val 로 나누세요 (Colab 노트북 ⑤ 단계가 자동 분할)\n"
        f"path: {root.as_posix()}\ntrain: images\nval: images\nnames:\n{names}\n", encoding="utf-8")
    preview(previews, out / "tiles_preview.jpg", {v: k for k, v in cid.items()})
    return n_obj_tiles, n_neg, cid


def preview(items, out: Path, id2code):
    if not items:
        return
    th = 300
    cols = 4
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * th, rows * th), (30, 30, 30))
    for k, (p, lines) in enumerate(items):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for ln in lines:
            v = ln.split()
            c = id2code[int(v[0])]
            pts = [(float(v[i]) * w, float(v[i + 1]) * h) for i in range(1, len(v), 2)]
            dr.polygon(pts, outline=COLORS.get(c, (255, 0, 255)), width=4)
        im = im.resize((th, th))
        ImageDraw.Draw(im).text((4, 4), p.stem, fill=(255, 255, 0), font=font(12))
        sheet.paste(im, ((k % cols) * th, (k // cols) * th))
    sheet.save(out, quality=90)


def make_overview(objs, R: Raster, out: Path):
    ov = out / "overview.jpg"
    R.overview(OVERVIEW_PCT, ov)
    im = Image.open(ov).convert("RGB")
    # 수거 계획 프로그램용: input/ortho/overview.jpg + 월드파일(.jgw, 픽셀 중심 기준) + .prj
    if R.geo:
        OVERVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ov, OVERVIEW_OUT)
        px = R.geo["px"] * R.width / im.size[0]; py = R.geo["py"] * R.height / im.size[1]
        OVERVIEW_OUT.with_suffix(".jgw").write_text(f"{px:.9f}
0
0
{py:.9f}
{R.geo['x0'] + px / 2:.6f}
{R.geo['y0'] + py / 2:.6f}
")
        prj = ORTHO.with_suffix(".prj")
        if prj.exists():
            shutil.copy2(prj, OVERVIEW_OUT.with_suffix(".prj"))
        print(f"  → {OVERVIEW_OUT} (+ .jgw) : 수거 계획 프로그램이 이 파일을 읽습니다")
    sx, sy = im.size[0] / R.width, im.size[1] / R.height
    dr = ImageDraw.Draw(im)
    f = font(14)
    for o in objs:
        cx = (o["px_xmin"] + o["px_xmax"]) / 2 * sx
        cy = (o["px_ymin"] + o["px_ymax"]) / 2 * sy
        r = 6
        col = COLORS.get(o["code"], (255, 0, 255))
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0), fill=col, width=1)
        dr.text((cx + r + 1, cy - r), str(o["seq"]), fill=(255, 255, 0), font=f)
    y = 8
    for c, col in COLORS.items():
        dr.rectangle([8, y, 22, y + 14], fill=col, outline=(0, 0, 0))
        dr.text((28, y - 2), f"{c} {CLASS_INFO[c][1]}", fill=(255, 255, 255), font=f)
        y += 20
    im.save(out / "overview_labeled.jpg", quality=90)


# ─────────────────────────────── 실행 ───────────────────────────────
def main():
    print("=" * 60)
    print(" 문갑도(MGD) 기업 배포 데이터 추출")
    print("=" * 60)
    for p in (LABELS_JSON, ORTHO):
        print(f"  {'✓' if p.exists() else '✗'} {p}")
    if not LABELS_JSON.exists():
        sys.exit("라벨 JSON 이 없습니다. 파일 위치를 확인하세요.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_world = read_world_file(ORTHO) if ORTHO.exists() else None
    crs = read_crs(ORTHO)
    print(f"[1] 좌표계: {crs.name}  / 월드파일: {gt_world['src'] if gt_world else '없음'}")

    R = None
    if ORTHO.exists():
        print("[2] 정사영상을 읽을 GDAL 찾는 중…")
        src = ORTHO
        R = Raster(src)
        if not R.ok and os.name == "nt" and not str(ORTHO).isascii():
            print("  한글 경로 문제일 수 있어 영문 경로로 복사 후 재시도")
            R = Raster(ascii_copy_if_needed(ORTHO))
        if R.ok:
            print(f"  ✓ {R.mode} ({R.exe or 'python osgeo'})  크기 {R.width} × {R.height} px")
        else:
            print("  ✗ ECW 를 읽을 수 있는 GDAL 을 못 찾음 → 라벨 표·요약만 만듭니다 (summary.md 안내 참고)")
    gt = (R.geo if R and R.ok and R.geo else None) or gt_world
    if gt is None:
        sys.exit("정사영상 좌표 정보(.eww)가 없어 픽셀 위치를 계산할 수 없습니다.")

    tr = Transformer.from_crs(4326, crs, always_xy=True)
    objs = load_objects(gt, tr)
    print(f"[3] 라벨 {len(objs)}개: " + ", ".join(f"{c} {n}" for c, n in Counter(o['code'] for o in objs).most_common()))
    write_tables(objs, OUT_DIR)
    stats = {"crs": crs.name, "objects.csv / objects.xlsx": f"{len(objs)}행", "labels_pixel.json": "픽셀 좌표 라벨"}

    if R and R.ok:
        inside = [o for o in objs if 0 <= o["px_xmin"] < R.width and 0 <= o["px_ymin"] < R.height]
        if len(inside) < len(objs):
            print(f"  ⚠ {len(objs) - len(inside)}개 라벨이 정사영상 범위 밖")
        print("[4] 전체 축소본…")
        try:
            make_overview(objs, R, OUT_DIR)
            stats["overview_labeled.jpg"] = f"전체 {OVERVIEW_PCT}% 축소 + 라벨 위치"
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ 축소본 실패: {e}")
        print("[5] 물체별 원해상도 사진…")
        n = make_chips(inside, R, gt, OUT_DIR)
        stats["chips/, chips_overlay/"] = f"{n}장 (여유 {CHIP_MARGIN_M} m)"
        print(f"[6] YOLO-seg 타일 ({TILE}px)…")
        nt, nn, cid = make_yolo(inside, R, OUT_DIR)
        stats["yolo_dataset/"] = f"쓰레기 타일 {nt}장 + 배경 타일 {nn}장, 클래스 {cid}"
        stats["tiles_preview.jpg"] = "타일·라벨 확인용"
    write_summary(objs, R, gt, gt_world, OUT_DIR, stats)

    print("\n[완료] 결과 폴더:", OUT_DIR)
    for k, v in stats.items():
        print(f"  - {k}: {v}")
    print("\n발견한 점:")
    for n_ in findings(objs, R, gt_world):
        print("  •", re.sub(r"\*\*", "", n_))
    if os.name == "nt":
        try:
            os.startfile(OUT_DIR)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
