"""기업 GeoJSON 라벨(지도 위 쓰레기 + 추정 무게) + 정사영상 → 작업자용 최적 수거 계획.

input/ 폴더에 파일을 넣고 실행하면 된다 (input/README.md 참고):
  input/labels.json          기업 GeoJSON 라벨 (필수)
  input/crops/*.jpg          물체 사진 (선택, 카드·팝업에 표시)
  input/ortho/overview.jpg   정사영상 축소본 + 월드파일 overview.jgw (+ .prj)  (선택, 지형·드론 영상 오버레이)
                             또는 GeoTIFF (ortho/*.tif)
  input/config.json          현장 이름·출발지·좌표계 (선택)

  python run.py                                   # 저장소 루트에서 한 번에
  python scripts/17_collection_plan.py --workers 4 --hours 6 --travel boat --carry carry --objective weight
  python scripts/17_collection_plan.py --input 다른폴더 --out outputs/다른현장

출력 (--out 폴더, 기본 outputs/plan):
  수거계획.html      인터랙티브 (조건을 바꾸면 브라우저 안에서 즉시 재계산, ★ 출발지 끌기)  ← 브라우저로 열기
  수거계획_공개용.html  인터넷 링크(claude.ai 아티팩트 등) 로 올리는 변형 — 외부 지도 타일 없이 드론 영상만, 인쇄·내려받기 없음
  수거계획_지도.png  인쇄용 지도 · 수거계획.xlsx / zones.csv / objects.csv / plan.json / summary.md / 지형분류.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litter3d.collect import CollectParams, load_geojson, make_collect_plan, save_plan_files, summary_markdown  # noqa: E402
from litter3d.collect_report import Basemap, build_collect_html, draw_static_map  # noqa: E402
from litter3d.plan import PlanParams  # noqa: E402
from litter3d.terrain import Terrain, save_class_png  # noqa: E402

DEFAULT_INPUT = ROOT / "input"
IMG_EXT = (".jpg", ".jpeg", ".png")
WORLD_EXT = {".jpg": [".jgw", ".jpgw", ".wld"], ".jpeg": [".jgw", ".jpegw", ".wld"], ".png": [".pgw", ".pngw", ".wld"],
             ".tif": [".tfw", ".tifw", ".wld"], ".tiff": [".tfw", ".wld"]}


def read_world_file(path: Path) -> dict | None:
    """월드파일(픽셀 중심 기준) → 왼쪽 위 모서리 기준 {x0, y0, px, py}."""
    if not path.exists():
        return None
    v = [float(x) for x in path.read_text().split()]
    a, d, b, e, c, f = v[:6]
    return {"x0": c - a / 2, "y0": f - e / 2, "px": a, "py": e}


def read_crs(img: Path, default: str) -> str:
    prj = img.with_suffix(".prj")
    if prj.exists():
        try:
            from pyproj import CRS
            crs = CRS.from_wkt(prj.read_text())
            epsg = crs.to_epsg() or crs.to_epsg(min_confidence=25)
            if epsg:
                return f"EPSG:{epsg}"
            print(f"  .prj 에 EPSG 코드가 없어 config 의 좌표계 {default} 사용 ({crs.name})")
        except Exception:  # noqa: BLE001
            pass
    return default


def load_basemap(ortho: Path | None, crs_default: str, max_px: int = 3200) -> tuple[Basemap | None, str]:
    """input/ortho 의 GeoTIFF 또는 (jpg/png + 월드파일) → Basemap. 반환 (basemap 또는 None, crs)."""
    if ortho is None or not ortho.exists():
        return None, crs_default
    suf = ortho.suffix.lower()
    if suf in (".tif", ".tiff"):
        try:
            import numpy as np
            import rasterio
            with rasterio.open(ortho) as ds:
                s = max(1, int(max(ds.width, ds.height) / max_px))
                arr = ds.read([1, 2, 3], out_shape=(3, ds.height // s, ds.width // s))
                img = np.ascontiguousarray(arr.transpose(1, 2, 0)[:, :, ::-1])     # RGB → BGR
                t = ds.transform
                world = {"x0": t.c, "y0": t.f, "px": t.a, "py": t.e}
                crs = f"EPSG:{ds.crs.to_epsg()}" if ds.crs and ds.crs.to_epsg() else crs_default
                return Basemap.from_array(img, world, (ds.width, ds.height), crs), crs
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ GeoTIFF 읽기 실패 ({e}) → 정사영상 없이 진행")
            return None, crs_default
    world = None
    for ext in WORLD_EXT.get(suf, [".wld"]):
        world = read_world_file(ortho.with_suffix(ext))
        if world:
            break
    if world is None:
        print(f"  ⚠ {ortho.name} 의 월드파일(.jgw/.pgw/.wld) 이 없어 정사영상 없이 진행")
        return None, crs_default
    crs = read_crs(ortho, crs_default)
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.fromfile(str(ortho), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지를 읽을 수 없음")
        return Basemap.from_array(img, world, (img.shape[1], img.shape[0]), crs), crs   # 월드파일이 이 이미지 기준
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 정사영상 읽기 실패 ({e})")
        return None, crs_default


def find_ortho(ortho_dir: Path, hint: str | None) -> Path | None:
    if hint:
        p = Path(hint)
        if p.exists():
            return p
    if not ortho_dir.exists():
        return None
    for pat in ("*.tif", "*.tiff", "overview.jpg", "*.jpg", "*.jpeg", "*.png"):
        for p in sorted(ortho_dir.glob(pat)):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="입력 폴더 (labels.json, crops/, ortho/, config.json)")
    ap.add_argument("--json", default=None, help="라벨 GeoJSON (기본 input/labels.json)")
    ap.add_argument("--photos", default=None, help="사진 폴더 (기본 input/crops)")
    ap.add_argument("--ortho", default=None, help="정사영상 (GeoTIFF 또는 월드파일 있는 jpg/png; 기본 input/ortho 에서 자동)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "plan"))
    ap.add_argument("--site", default=None)
    ap.add_argument("--depot", default=None, help="출발·집결지 경도,위도 ('none' → 무게중심). 기본 config.json")
    ap.add_argument("--depot-name", default=None)
    ap.add_argument("--workers", type=int, default=2, help="한 팀 인원")
    ap.add_argument("--hours", type=float, default=4.0, help="하루 작업 시간")
    ap.add_argument("--teams", type=int, default=1, help="동시에 투입하는 팀 수 (순회를 시간 균형으로 나눔)")
    ap.add_argument("--calib", default=None, help="실측 보정 계수 (예: STY=0.8,ROP=1.5) — 우리 추정 무게에 곱함")
    ap.add_argument("--link", type=float, default=150.0, help="구역 묶기 거리 m (가정값)")
    ap.add_argument("--detour", type=float, default=1.4, help="지형 없을 때 직선 → 실제 거리 배수 (가정값)")
    ap.add_argument("--walk", type=float, default=3.0, help="걷기 속도 km/h (가정값)")
    ap.add_argument("--weight", choices=["ours", "company"], default="ours", help="계획에 쓰는 무게")
    ap.add_argument("--one-way", action="store_true", help="출발지로 돌아오지 않음")
    ap.add_argument("--travel", choices=["walk", "boat"], default="walk", help="이동 방식 (정사영상 있을 때 보트 지원 가능)")
    ap.add_argument("--carry", choices=["pile", "carry"], default="pile", help="pile 현장 적치 / carry 들고 이동(적재량 넘으면 복귀)")
    ap.add_argument("--objective", choices=["distance", "weight"], default="distance", help="최단 이동 / 무게 우선")
    ap.add_argument("--codes", default=None, help="수거할 재질 코드만 (예: STY,ROP)")
    ap.add_argument("--min-kg", type=float, default=0.0, help="이 계획 무게 미만 물체는 건너뜀")
    ap.add_argument("--no-terrain", action="store_true", help="지형 격자(물·숲·맨땅) 사용 안 함")
    ap.add_argument("--cell", type=float, default=10.0, help="지형 격자 셀 크기 m")
    ap.add_argument("--crs", default=None, help="거리 계산용 투영 좌표계 (기본 config.json 또는 EPSG:5186)")
    ap.add_argument("--no-open", action="store_true", help="끝나고 브라우저로 열지 않음")
    a = ap.parse_args()

    inp = Path(a.input)
    cfg = {}
    if (inp / "config.json").exists():
        cfg = json.loads((inp / "config.json").read_text(encoding="utf-8"))
    labels = Path(a.json) if a.json else inp / cfg.get("labels", "labels.json")
    photos = Path(a.photos) if a.photos else inp / cfg.get("photos", "crops")
    ortho = find_ortho(inp / "ortho", a.ortho or (str(inp / cfg["ortho"]) if cfg.get("ortho") else None))
    site = a.site or cfg.get("site", inp.name)
    crs_default = a.crs or cfg.get("crs", "EPSG:5186")
    if a.depot:
        depot = None if a.depot.lower() == "none" else tuple(float(v) for v in a.depot.split(","))
        depot_name = a.depot_name or "출발지"
    elif cfg.get("depot"):
        depot = (float(cfg["depot"]["lon"]), float(cfg["depot"]["lat"])); depot_name = a.depot_name or cfg["depot"].get("name", "출발지")
    else:
        depot = None; depot_name = a.depot_name or "무게중심"
    if not labels.exists():
        sys.exit(f"라벨 파일이 없습니다: {labels}\n  input/labels.json 에 기업 GeoJSON 을 넣으세요 (input/README.md 참고).")

    out = Path(a.out)
    print(f"[1] 입력 폴더: {inp}")
    print(f"    라벨: {labels.name} · 사진: {photos if photos.exists() else '없음'} · 정사영상: {ortho.name if ortho else '없음'}")
    basemap, crs = load_basemap(ortho, crs_default)
    objs = load_geojson(labels, crs_m=crs)
    print(f"    {len(objs)}개 · 재질 " + ", ".join(f"{c} {n}" for c, n in sorted(
        {o.code: sum(1 for q in objs if q.code == o.code) for o in objs}.items(), key=lambda kv: -kv[1])) + f" · 좌표계 {crs}")

    params = CollectParams(link_m=a.link, detour=a.detour, walk_kmh=a.walk, workers=a.workers, hours_per_day=a.hours,
                           round_trip=not a.one_way, weight_source=a.weight, depot_lonlat=depot, depot_name=depot_name,
                           travel=a.travel, carry=a.carry, objective=a.objective, min_kg=a.min_kg, teams=a.teams,
                           calib={k.strip(): float(v) for k, v in (kv.split("=") for kv in a.calib.split(","))} if a.calib else None,
                           include_codes=[c.strip() for c in a.codes.split(",")] if a.codes else None, bag=PlanParams())

    center = None; terrain = None
    if basemap is not None:
        e = basemap.extent_m
        center = ((e[0] + e[1]) / 2, (e[2] + e[3]) / 2)
        print(f"[2] 정사영상 축소본 {basemap.img.shape[1]}×{basemap.img.shape[0]} px, {basemap.world['px'] * basemap.full_px[0] / basemap.img.shape[1]:.2f} m/px")
        if not a.no_terrain:
            terrain = Terrain.from_basemap(basemap, a.cell)
            print(f"[2b] 지형 격자 {terrain.nr}×{terrain.nc} ({a.cell:g} m): " + ", ".join(f"{k} {v}" for k, v in terrain.summary().items()))
    else:
        print("[2] 정사영상 없음 → 위성 지도만, 거리는 직선 × 우회 배수 (input/ortho 에 overview.jpg + .jgw 또는 GeoTIFF 를 넣으면 지형 반영)")

    print("[3] 구역 묶기 · 경로 최적화 · 시간 계산")
    plan = make_collect_plan(objs, params, site=site, center_xy=center, crs_m=crs, terrain=terrain)
    t = plan.totals
    print(f"    구역 {t['zones']}곳 · 예상 {t['kg_plan']:.1f} kg (범위 {t['kg_min']:.1f}–{t['kg_max']:.1f}; 기업값 합 {t['company_kg']:.3f}) · "
          f"마대 {t['bags']}장 · 이동 {plan.route_len_m / 1000:.1f} km ({'지형' if terrain else '직선×우회'}, {a.travel}/{a.carry}) · "
          f"총 {plan.total_min / 60:.1f}시간 ({len(plan.teams)}팀 × {a.workers}명) → 최대 {max((t['days'] for t in plan.teams), default=0)}일" + (f" · 제외 {plan.n_skipped}개" if plan.n_skipped else ""))

    print(f"[4] 저장: {out}")
    save_plan_files(plan, out)
    (out / "summary.md").write_text(summary_markdown(plan), encoding="utf-8")
    if terrain is not None:
        save_class_png(terrain, out / "지형분류.png")
    png = draw_static_map(plan, out / "수거계획_지도.png", basemap=basemap, crs_m=crs)
    shared_cfg = cfg.get("shared")
    if shared_cfg:                      # "NEXT_PUBLIC_SUPABASE_URL=https://..." 처럼 이름=값 으로 붙여 넣어도 값만 쓴다
        for key in ("url", "anon_key"):
            v = str(shared_cfg.get(key, "")).strip().strip('"').strip("'")
            if "=" in v and v.split("=", 1)[0].isupper():
                v = v.split("=", 1)[1].strip().strip('"').strip("'")
            shared_cfg[key] = v.rstrip("/")
    if shared_cfg and (str(shared_cfg.get("url", "")).startswith("<") or str(shared_cfg.get("anon_key", "")).startswith("<")
                       or not str(shared_cfg.get("url", "")).startswith("https://")):
        print("    공유 저장: config.json 의 shared.url / anon_key 가 아직 자리값(<...>) 이라 끄고 진행 (Supabase 값을 넣으면 켜짐)")
        shared_cfg = None
    if shared_cfg and shared_cfg.get("provider") == "supabase" and shared_cfg.get("url") and shared_cfg.get("anon_key"):
        print(f"    공유 저장: Supabase {shared_cfg['url']} (테이블 {shared_cfg.get('table', 'shared_state')})")
    build_collect_html(plan, out / "수거계획.html", photos_dir=photos if photos.exists() else None, basemap=basemap,
                       static_png=png, terrain=terrain, center_xy=center, crs_m=crs, shared_cfg=shared_cfg)
    build_collect_html(plan, out / "수거계획_공개용.html", photos_dir=photos if photos.exists() else None, basemap=basemap,
                       static_png=png, terrain=terrain, center_xy=center, crs_m=crs, artifact=True)   # claude.ai 아티팩트 등 공개 링크용
    for z in plan.zones:
        comp = ", ".join(f"{k} {v}" for k, v in z.by_code.items())
        flag = " ⚠2인" if z.heavy_ids else ""
        print(f"    {'ABCDEF'[z.team - 1] + '팀 ' if len(plan.teams) > 1 else ''}{z.day}일차 {z.step:>2}. {z.name:<10} {z.n:>2}개 ({comp}) {z.kg_plan:6.1f} kg 마대 {z.bags:>2} 접근 {z.dist_from_prev_m:5.0f} m "
              f"이동 {z.walk_min:3.0f}분 작업 {z.work_min:3.0f}분{flag}" + (f" 복귀 {z.returns}회" if z.returns else ""))
    print("\n결과:")
    for name in ["수거계획.html", "수거계획_지도.png", "수거계획.xlsx", "zones.csv", "objects.csv", "plan.json", "summary.md"]:
        print(f"  {out / name}")
    if not a.no_open:
        try:
            webbrowser.open((out / "수거계획.html").resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
