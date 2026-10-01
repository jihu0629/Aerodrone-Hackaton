"""물체 주위를 돌며 찍은 영상 → SfM(pycolmap, CPU) → 바닥 평면 → visual hull → 길이·넓이·높이·부피.

  python scripts/30_orbit_volume.py IMG_6571.MOV --out data/objects/box1
  python scripts/30_orbit_volume.py IMG_6571.MOV --out data/objects/box1 --camera-height-m 1.2        # 대략 축척
  python scripts/30_orbit_volume.py IMG_6571.MOV --out data/objects/box1 \\
      --ref img005.jpg:412,633:588,640 --ref img020.jpg:300,700:470,712 --ref-length-m 0.60            # 기준 길이로 축척
  python scripts/30_orbit_volume.py --images photos/ --out data/objects/box2 --mask-method rembg       # 사진 폴더, rembg 마스크

--ref 형식: 이미지이름:u1,v1:u2,v2  (같은 두 점을 서로 다른 2장 이상에서). 이미지 이름은 out/frames 의 파일명.
--pick-ref 를 주면 창을 띄워 마우스로 찍는다 (2장 × 2점, 창에서 클릭 → Enter).
결과: out/metrics.json, report.md, masks/, overlay/, sparse_points.ply, hull_surface.ply (CloudCompare·MeshLab 로 열기)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_ref(items: list[str]):
    out = []
    for s in items:
        name, p1, p2 = s.split(":")
        u1, v1 = map(float, p1.split(",")); u2, v2 = map(float, p2.split(","))
        out.append((name, (u1, v1), (u2, v2)))
    return out


def pick_ref_interactive(images_dir: Path, names: list[str]):
    import cv2
    refs = []
    for name in names:
        img = cv2.imread(str(images_dir / name))
        pts = []

        def cb(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((float(x), float(y)))
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.namedWindow("ref", cv2.WINDOW_NORMAL); cv2.setMouseCallback("ref", cb)
        while True:
            cv2.imshow("ref", img)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10) and len(pts) == 2:
                break
            if k == 27:
                cv2.destroyAllWindows(); return None
        refs.append((name, pts[0], pts[1]))
        print(f"{name}: {pts[0]} → {pts[1]}")
    cv2.destroyAllWindows()
    return refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--images", help="영상 대신 이미지 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40, help="영상에서 뽑을 프레임 수")
    ap.add_argument("--max-size", type=int, default=1080, help="긴 변 픽셀 (클수록 정밀, 느림)")
    ap.add_argument("--exhaustive", action="store_true", help="순차 매칭 대신 전체 매칭 (사진 순서가 뒤섞였을 때)")
    ap.add_argument("--force", action="store_true", help="기존 프레임·SfM 결과 무시하고 다시 계산")
    ap.add_argument("--mask-method", choices=["auto", "rembg", "bright", "dark", "dir"], default="auto",
                    help="auto: rembg(U2Net) 가 설치돼 있으면 사용, 없으면 bright(흰 물체 휴리스틱)")
    ap.add_argument("--mask-dir", help="--mask-method dir 일 때 마스크 폴더 (프레임과 같은 이름의 png)")
    ap.add_argument("--ref", action="append", default=[], help="이미지:u1,v1:u2,v2 (2개 이상)")
    ap.add_argument("--ref-length-m", type=float)
    ap.add_argument("--pick-ref", nargs=2, metavar="IMG", help="두 이미지 이름. 창에서 기준 두 점을 클릭")
    ap.add_argument("--camera-height-m", type=float, help="대략적인 카메라 높이로 축척 (기준 길이가 없을 때)")
    ap.add_argument("--res", type=int, default=96, help="복셀 격자 해상도 (한 축)")
    ap.add_argument("--vote", type=float, default=1.0, help="복셀을 남기는 마스크 투표 비율 (1.0 = 엄밀한 실루엣 교차. 틀린 마스크는 자동 제외됨)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if not a.video and not a.images:
        ap.error("영상 파일 또는 --images 폴더가 필요합니다")
    from dronecap.object3d import run_pipeline
    ref = parse_ref(a.ref) if a.ref else None
    if a.pick_ref:
        if not a.ref_length_m:
            ap.error("--pick-ref 에는 --ref-length-m 이 필요합니다")
        # 프레임이 아직 없으면 먼저 추출만
        from dronecap.object3d import extract_orbit_frames
        images_dir = Path(a.images) if a.images else Path(a.out) / "frames"
        if a.video and not any(images_dir.glob("*.jpg")):
            extract_orbit_frames(a.video, images_dir, a.frames, a.max_size)
        ref = pick_ref_interactive(images_dir, a.pick_ref)
        if ref is None:
            return 1
    r = run_pipeline(a.video, a.out, a.images, a.frames, a.max_size, a.exhaustive, a.force, a.mask_method, a.mask_dir,
                     ref, a.ref_length_m, a.camera_height_m, res=a.res, vote_ratio=a.vote, num_threads=a.threads)
    h = r["hull"]; u = h["unit"]
    print("\n=== 결과 ({}) ===".format("미터" if u == "m" else "SfM 상대 단위"))
    print(f"길이×너비: {h.get('length', 0):.4g} × {h.get('width', 0):.4g} {u},  높이: {h['height_area90']:.4g} {u} (단면적 90% 기준; hull 최대 {h['height_max']:.4g})")
    print(f"바닥 면적 {h['footprint_area']:.4g} {u}²,  부피 {h['volume_below_h90']:.4g} {u}³ (90% 높이까지; 전체 hull {h['volume']:.4g}),  외접상자 {h.get('bbox_volume', 0):.4g} {u}³")
    for d in r.get("diagnosis", []):
        print("진단:", d)
    print(f"→ {Path(a.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
