"""
선회 영상 → 3D 복원 (빠른 확인용).

드론이 물체 주위를 한 바퀴 돌며 찍은 영상에서 프레임을 뽑아 COLMAP(SfM)으로
카메라 위치와 3D 점구름을 복원한다. 결과:
  points.ply         3D 점구름 (CloudCompare·MeshLab·Blender로 열기)
  viewer.html        브라우저로 바로 보는 3D 뷰어 (점 + 카메라 궤적)
  summary.json       등록된 프레임 수 · 점 개수 · 재투영 오차 · 궤적 반경

  python -m litter.orbit3d --video DJI_0001.MP4 --out runs/orbit/test1
  python -m litter.orbit3d --images 폴더 --out ...        # 사진이 이미 있으면

--alt(선회 고도 m) 또는 --srt를 주면 실제 크기(m)로 맞추고 선회 중심의 물체 부피를 잰다:
  python -m litter.orbit3d --video DJI_0001.MP4 --srt DJI_0001.SRT --out runs/orbit/box1
  (이륙 지점과 물체 바닥 높이가 같다고 가정 — 평평한 곳에서 이륙)
CPU 버전 pycolmap이라 프레임 80장 기준 수 분 걸린다.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video, out_dir, n=80, max_side=1600, blur_keep=0.7, start_s=0.0, end_s=None):
    """영상 전체에서 고르게 n장을 고르되, 구간마다 가장 선명한 프레임을 쓴다
    (흔들림 블러 프레임은 특징점 매칭을 망친다)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"영상을 못 엶: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    f0 = int(start_s * fps)
    f1 = min(total, int(end_s * fps)) if end_s else total
    edges = np.linspace(f0, f1, n + 1).astype(int)  # 선회 구간만 (이착륙 장면 제외)
    names, k, fidx = [], 0, {}
    for s, e in zip(edges[:-1], edges[1:]):
        best, best_score, best_f = None, -1, None
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        cand = range(s, e, max(1, (e - s) // 4))  # 구간에서 최대 4장만 보고 고른다
        for f in cand:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
            sc = cv2.Laplacian(g, cv2.CV_64F).var()
            if sc > best_score:
                best, best_score, best_f = img, sc, f
        if best is None:
            continue
        s_ = max_side / max(best.shape[:2])
        if s_ < 1:
            best = cv2.resize(best, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
        k += 1
        name = f"f{k:04d}.jpg"
        cv2.imwrite(str(out_dir / name), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
        fidx[name] = int(best_f) + 1  # SRT FrameCnt는 1부터
    cap.release()
    (out_dir / "frames.json").write_text(json.dumps(fidx), encoding="utf-8")
    print(f"프레임 {len(names)}장 추출 (영상 {total}프레임) → {out_dir}")
    return names


def reconstruct(image_dir, work, sequential=True):
    import pycolmap

    work = Path(work)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    sparse = work / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    # 영상 프레임은 모두 같은 카메라 → SINGLE (초점거리를 함께 추정해 안정적)
    eo = pycolmap.FeatureExtractionOptions()
    eo.num_threads = 6           # 스레드를 줄여 메모리 부족으로 죽는 것 방지 (학습과 동시 실행 시)
    eo.sift.max_num_features = 8192
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=eo)
    if sequential:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = 15
        po.loop_detection = False
        pycolmap.match_sequential(db, pairing_options=po)
    else:
        pycolmap.match_exhaustive(db)
    recs = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not recs:
        raise SystemExit("복원 실패 — 프레임 간 겹침이 부족하거나 질감이 너무 적음 (물·하늘 위주 화면)")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    return rec


def _center(img):
    pc = getattr(img, "projection_center", None)
    if callable(pc):
        return np.asarray(pc())
    T = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    R = np.asarray(T.rotation.matrix())
    return -R.T @ np.asarray(T.translation)


def _viewdir(img):
    T = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    R = np.asarray(T.rotation.matrix())
    return R.T @ np.array([0.0, 0.0, 1.0])


def export(rec, out, n_frames, highlight=None):
    out = Path(out)
    rec.export_PLY(str(out / "points.ply"))
    P = np.array([p.xyz for p in rec.points3D.values()])
    C = np.array([p.color for p in rec.points3D.values()], float)
    if highlight is not None and len(highlight):  # 물체로 잡힌 점은 빨간색
        C[highlight] = [255, 40, 40]
    cams = np.array([_center(im) for im in rec.images.values()])
    # 궤적 반경: 카메라 중심들이 중심점에서 떨어진 거리 (스케일 미정, 상대값)
    c0 = cams.mean(0)
    summary = {
        "frames_extracted": n_frames, "frames_registered": rec.num_reg_images(),
        "points3D": rec.num_points3D(), "mean_reproj_error_px": rec.compute_mean_reprojection_error(),
        "mean_track_length": rec.compute_mean_track_length(),
        "orbit_radius_rel": float(np.median(np.linalg.norm(cams - c0, axis=1))),
        "note": "스케일 미정 (상대 단위) — GPS/기준물체로 m 단위 보정은 다음 단계",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 브라우저 뷰어 (점 최대 15만 개)
    if len(P) > 150000:
        idx = np.random.default_rng(0).choice(len(P), 150000, replace=False)
        P, C = P[idx], C[idx]
    ctr = np.median(P, axis=0)
    data = {"p": (P - ctr).round(4).ravel().tolist(), "c": (C / 255).round(3).ravel().tolist(),
            "cam": (cams - ctr).round(4).ravel().tolist()}
    html = _VIEWER.replace("__DATA__", json.dumps(data))
    (out / "viewer.html").write_text(html, encoding="utf-8")
    return summary


_VIEWER = """<!doctype html><meta charset="utf-8"><title>orbit 3D</title>
<style>body{margin:0;background:#111;color:#ddd;font:13px sans-serif}#i{position:fixed;left:10px;top:8px}</style>
<div id="i">드래그: 회전 · 휠: 확대 · 오른쪽 드래그: 이동 &nbsp; <span style="color:#f55">●</span> 카메라 위치(선회 궤적)</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__; const s=new THREE.Scene(); const r=new THREE.WebGLRenderer({antialias:true});
r.setSize(innerWidth,innerHeight); document.body.appendChild(r.domElement);
const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(D.p,3));
g.setAttribute('color',new THREE.Float32BufferAttribute(D.c,3)); g.computeBoundingSphere();
const R=g.boundingSphere.radius; s.add(new THREE.Points(g,new THREE.PointsMaterial({size:R/400,vertexColors:true})));
const cg=new THREE.BufferGeometry(); cg.setAttribute('position',new THREE.Float32BufferAttribute(D.cam,3));
s.add(new THREE.Points(cg,new THREE.PointsMaterial({size:R/60,color:0xff5555}))); s.add(new THREE.Line(cg,new THREE.LineBasicMaterial({color:0xff5555})));
const c=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,R/1000,R*100); c.position.set(0,-R*1.5,R*1.2); c.up.set(0,0,1);
const o=new OrbitControls(c,r.domElement); (function a(){requestAnimationFrame(a);o.update();r.render(s,c)})();
addEventListener('resize',()=>{c.aspect=innerWidth/innerHeight;c.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});
</script>"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.orbit3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--images", help="프레임/사진 폴더 (영상 대신)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80, help="영상에서 뽑을 프레임 수")
    ap.add_argument("--max_side", type=int, default=1600)
    ap.add_argument("--start", type=float, default=0.0, help="이 시각(초)부터 (이륙 장면 제외)")
    ap.add_argument("--end", type=float, help="이 시각(초)까지")
    ap.add_argument("--exhaustive", action="store_true", help="사진 순서가 뒤섞여 있을 때")
    ap.add_argument("--alt", type=float, help="선회 고도 (m, DJI Fly 화면의 H 값) — 주면 실제 크기·부피 계산")
    ap.add_argument("--srt", help="영상의 SRT 자막 — 프레임별 상대고도로 실제 크기 계산 (--alt보다 정확)")
    ap.add_argument("--radius", type=float, help="물체를 찾을 반경 (m, 기본: 선회 반경의 60%)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.video:
        img_dir = out / "frames"
        names = extract_frames(a.video, img_dir, a.n, a.max_side, start_s=a.start, end_s=a.end)
    elif a.images:
        img_dir = Path(a.images)
        names = [p.name for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    else:
        raise SystemExit("--video 또는 --images 필요")
    rec = reconstruct(img_dir, out, sequential=not a.exhaustive)
    vol, hl = None, None
    if a.alt or a.srt:
        from .volume import measure
        imgs = list(rec.images.values())
        P = np.array([p.xyz for p in rec.points3D.values()])
        C = np.array([_center(im) for im in imgs])
        D = np.array([_viewdir(im) for im in imgs])
        if a.srt:  # 프레임마다 SRT 상대고도
            from .telemetry import read_srt
            tel = {r["frame"]: r for r in read_srt(a.srt) if "alt" in r}
            fj = img_dir / "frames.json"
            fidx = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {}
            alt = np.array([tel.get(fidx.get(im.name, -1), {}).get("alt", np.nan) for im in imgs])
            if np.isnan(alt).all():
                raise SystemExit("SRT 고도를 프레임과 못 맞춤 — --alt로 고도를 직접 주세요")
            C, D, alt = C[~np.isnan(alt)], D[~np.isnan(alt)], alt[~np.isnan(alt)]
        else:
            alt = a.alt
        vol, Pm, hl = measure(P, C, D, alt, radius=a.radius)
    s = export(rec, out, len(names), highlight=hl)
    if vol:
        s["volume"] = vol
        (out / "summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
            encoding="utf-8")
    print(f"3D 복원: 프레임 {s['frames_registered']}/{s['frames_extracted']} 등록 · 점 {s['points3D']:,}개 · "
          f"재투영 오차 {s['mean_reproj_error_px']:.2f}px")
    if vol and "error" not in vol:
        print(f"물체: {vol['length_m']:.2f} × {vol['width_m']:.2f} × {vol['height_m']:.2f} m · 바닥면적 {vol['footprint_m2']:.2f} m² · "
              f"부피 {vol['volume_heightmap_m3'] * 1000:.0f} L (높이지도) / {vol['volume_hull_m3'] * 1000:.0f} L (볼록껍질) · "
              f"선회 반경 {vol['orbit_radius_m']:.1f} m · 고도 {vol['orbit_height_m']:.1f} m · 축척 편차 {vol['scale_spread_pct']:.1f}%")
        if vol["scale_spread_pct"] > 15:
            print(f"  ⚠️ 축척 편차 {vol['scale_spread_pct']:.0f}% — 프레임마다 고도/바닥 거리 비율이 크게 다름. "
                  "선회가 아니거나 바닥이 평평하지 않거나 이륙 지점 높이가 다름 → 부피를 믿기 어려움")
    elif vol:
        print(f"⚠️ 부피 측정 실패: {vol['error']}")
    print(f"  → {out / 'viewer.html'} (브라우저로 열기, 빨간 점 = 물체) · {out / 'points.ply'}")


if __name__ == "__main__":
    main()
