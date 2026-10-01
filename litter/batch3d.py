"""
영상 한 편의 탐지 상자 전부를 구간별로 "3D 뼈대 → 빠른 조밀 복원 → 지도 위치 → 부피"까지 한 번에.

  python -m litter.batch3d run --video ../DJI_0015.MP4 --srt ../DJI_0015.SRT \
         --objects runs/map/0015_world2/objects.csv --out runs/orbit/0015_all [--only 24,19]

구간 계획: 상자별 시각 t(= first_frame/fps) → [t-4, t+4] 초. 겹치는 상자는 한 구간으로 합친다.
구간마다:
  sfm    (CPU, 2개 동시)  프레임 n장 추출(구간을 한 번에 순차 읽기) → pycolmap 뼈대 (orbit3d 재사용)
  dense  (GPU, 1개씩)     COLMAP 빠른 조밀 (undistort 1200 · 소스 8 · 반복 3 · cache 8GB) → fused_dense.ply
  map    (GPU, 1개씩)     orbit_map (YOLO-World "cardboard box", GPS 경로 축척)
  vol    (GPU, 1개씩)     objvol v2 (SAM 2, 조밀 점, search-r 0.45)
  mosaic (CPU, 선택)      정사영상
진행은 <out>/status.json 에 구간·단계별 기록, 실패해도 다음 구간 계속. 다시 실행하면 끝난 단계는 건너뛴다.
끝나면 objects3d.csv · map.html · map.png · summary.json · masks_sheet.jpg 를 <out>에 만든다.
"""
import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # aerodrone_hackathon
PY = sys.executable
STAGES = ["sfm", "dense", "map", "vol", "mosaic"]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _colmap_exe():
    """COLMAP CLI는 한글 경로를 싫어하므로 junction(C:\\work\\drone)이 있으면 그쪽으로."""
    for base in (Path(r"C:\work\drone\aerodrone_hackathon"), ROOT):
        exe = base / "tools" / "colmap" / "bin" / "colmap.exe"
        if exe.exists():
            return exe
    raise SystemExit("colmap.exe 를 못 찾음 (tools/colmap/bin)")


# ------------------------------------------------------------------ 구간 계획
def plan_segments(objects_csv, half=4.0, cls="cardboard box", fps=59.94, n_per_s=5.0, n_min=40, n_max=60):
    rows = [r for r in csv.DictReader(open(objects_csv, encoding="utf-8-sig")) if r["cls"] == cls]
    boxes = [{"t": int(r["first_frame"]) / fps, "lat": float(r["lat"]), "lon": float(r["lon"]),
              "seen": int(r["seen_frames"]), "score": float(r["best_score"])} for r in rows]
    boxes.sort(key=lambda b: b["t"])
    for k, b in enumerate(boxes):           # 상자 번호 = 영상 시간순 (#4/#5 = 48·51초, #24 = 434초)
        b["id"] = k + 1
    segs = []
    for b in boxes:
        if segs and b["t"] - half <= segs[-1]["end"]:
            segs[-1]["end"] = b["t"] + half
            segs[-1]["boxes"].append(b["id"])
        else:
            segs.append({"start": max(0.0, b["t"] - half), "end": b["t"] + half, "boxes": [b["id"]]})
    for k, s in enumerate(segs):
        s["name"] = f"seg{k + 1:02d}_b" + "-".join(f"{i:02d}" for i in s["boxes"])
        s["n"] = int(min(n_max, max(n_min, round(n_per_s * (s["end"] - s["start"])))))
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return boxes, segs


# ------------------------------------------------------------------ sfm 단계 (하위 프로세스)
def extract_frames_seq(video, out_dir, n, start_s, end_s, max_side=1600, cands=4):
    """구간을 한 번에 순차로 읽으며(seek 1회) 칸마다 가장 선명한 프레임을 고른다.
    orbit3d.extract_frames 와 같은 결과(f0001.jpg…, frames.json) — 4K 영상에서 seek 160회보다 훨씬 빠르다."""
    import cv2
    from .seg import _imwrite

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    f0, f1 = int(start_s * fps), min(total, int(end_s * fps))
    edges = np.linspace(f0, f1, n + 1).astype(int)
    want = {}
    for b, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        for f in list(range(s, e, max(1, (e - s) // cands)))[:cands]:
            want[f] = b
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    best = {}
    f = f0
    while f < f1:
        if f in want:
            ok, img = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
            sc = cv2.Laplacian(g, cv2.CV_64F).var()
            b = want[f]
            if b not in best or sc > best[b][0]:
                s_ = max_side / max(img.shape[:2])
                if s_ < 1:
                    img = cv2.resize(img, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
                best[b] = (sc, img, f)
        elif not cap.grab():
            break
        f += 1
    cap.release()
    names, fidx = [], {}
    for k, b in enumerate(sorted(best)):
        _, img, fr = best[b]
        name = f"f{k + 1:04d}.jpg"
        _imwrite(out_dir / name, img, 95)
        names.append(name)
        fidx[name] = int(fr) + 1          # SRT FrameCnt는 1부터
    (out_dir / "frames.json").write_text(json.dumps(fidx), encoding="utf-8")
    print(f"프레임 {len(names)}장 추출 ({start_s:.1f}~{end_s:.1f}s)")
    return names


def stage_sfm(video, start, end, n, out):
    from .orbit3d import export, reconstruct

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    t0 = time.time()
    names = extract_frames_seq(video, frames, n, start, end)
    t1 = time.time()
    rec = reconstruct(frames, out)
    import pycolmap

    # incremental_mapping 이 모델을 여러 개 만들면 가장 큰 것을 sparse/0 에 두어 뒷단(orbit_map·objvol)이 그걸 쓰게
    r0 = pycolmap.Reconstruction(str(out / "sparse" / "0"))
    if r0.num_reg_images() < rec.num_reg_images():
        rec.write(str(out / "sparse" / "0"))
    s = export(rec, out, len(names))
    s["time_extract_s"], s["time_sfm_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"3D 뼈대: {s['frames_registered']}/{s['frames_extracted']} 등록 · 점 {s['points3D']:,} · "
          f"재투영 {s['mean_reproj_error_px']:.2f}px · 추출 {s['time_extract_s']}s · SfM {s['time_sfm_s']}s")


# ------------------------------------------------------------------ 실행기
class Runner:
    def __init__(self, a, boxes, segs):
        self.a = a
        self.boxes = {b["id"]: b for b in boxes}
        self.segs = segs
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.work = Path(a.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.colmap = _colmap_exe()
        self.cpu = threading.Semaphore(a.cpu_jobs)
        self.gpu = threading.Semaphore(1)
        self.lock = threading.Lock()
        self.status_path = self.out / "status.json"
        self.status = {}
        if self.status_path.exists():
            try:
                self.status = json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:
                self.status = {}
        for s in segs:
            st = self.status.setdefault(s["name"], {})
            st.update(start=s["start"], end=s["end"], n=s["n"], boxes=s["boxes"])
            st.setdefault("stages", {})
        self._save()

    # ---- 상태 기록
    def _save(self):
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.status_path)

    def mark(self, seg, stage, **kw):
        with self.lock:
            st = self.status[seg]["stages"].setdefault(stage, {})
            st.update(kw)
            self._save()

    def done(self, seg, stage):
        return self.status[seg]["stages"].get(stage, {}).get("status") == "ok"

    def log(self, msg):
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        with self.lock:
            with open(self.out / "batch.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ---- 보조
    def wait_mem(self, min_gb=2.5, max_wait=300):
        try:
            import psutil
        except ImportError:
            return
        t0 = time.time()
        while psutil.virtual_memory().available / 1e9 < min_gb and time.time() - t0 < max_wait:
            time.sleep(10)

    def run_cmd(self, cmd, log_file, cwd=None, env=None, timeout=900):
        e = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if env:
            e.update(env)
        with open(log_file, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                raise RuntimeError(f"시간 초과 {timeout}s")
        if rc != 0:
            tail = Path(log_file).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
            raise RuntimeError(f"종료코드 {rc}: " + " | ".join(t.strip() for t in tail if t.strip())[-400:])
        return Path(log_file).read_text(encoding="utf-8", errors="replace")

    def _stage(self, seg, stage, fn, sem):
        """세마포어 안에서 fn() 실행, 상태 기록. 반환 True/False."""
        name = seg["name"]
        if self.done(name, stage):
            self.log(f"{name} {stage}: 이미 완료 → 건너뜀")
            return True
        with sem:
            self.wait_mem()
            self.mark(name, stage, status="running", t0=_now(), reason=None)
            self.log(f"{name} {stage}: 시작")
            t = time.time()
            try:
                info = fn() or {}
                self.mark(name, stage, status="ok", t1=_now(), sec=round(time.time() - t, 1), **info)
                self.log(f"{name} {stage}: 완료 {time.time() - t:.0f}s " + " ".join(f"{k}={v}" for k, v in info.items()))
                return True
            except Exception as ex:
                self.mark(name, stage, status="fail", t1=_now(), sec=round(time.time() - t, 1), reason=str(ex)[:500])
                self.log(f"{name} {stage}: 실패 ({ex})")
                return False

    # ---- 단계들
    def do_sfm(self, seg):
        d = self.out / seg["name"]
        d.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-m", "litter.batch3d", "sfm", "--video", self.a.video, "--start", seg["start"], "--end", seg["end"],
               "--n", seg["n"], "--out", d]
        self.run_cmd(cmd, d / "log_sfm.txt", cwd=ROOT, env={"OMP_NUM_THREADS": str(self.a.omp)}, timeout=self.a.t_sfm)
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if s["frames_registered"] < 12:
            raise RuntimeError(f"등록 프레임 부족 {s['frames_registered']}/{s['frames_extracted']}")
        return {"reg": s["frames_registered"], "ext": s["frames_extracted"], "pts": s["points3D"],
                "reproj": round(s["mean_reproj_error_px"], 2)}

    def do_dense(self, seg):
        d = self.out / seg["name"]
        w = self.work / seg["name"]
        if w.exists():
            shutil.rmtree(w)
        shutil.copytree(d / "frames", w / "images")
        shutil.copytree(d / "sparse" / "0", w / "sparse")
        env = {"PATH": f"{self.colmap.parent};{self.colmap.parent.parent / 'lib'};" + os.environ.get("PATH", "")}
        log = w / "log_dense.txt"
        cm = str(self.colmap)
        self.run_cmd([cm, "image_undistorter", "--image_path", "images", "--input_path", "sparse", "--output_path", "dense",
                      "--output_type", "COLMAP", "--max_image_size", "1200"], log, cwd=w, env=env, timeout=300)
        cfg = w / "dense" / "stereo" / "patch-match.cfg"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace("__auto__, 20", "__auto__, 8"), encoding="utf-8")
        self.run_cmd([cm, "patch_match_stereo", "--workspace_path", "dense", "--workspace_format", "COLMAP",
                      "--PatchMatchStereo.max_image_size", "1200", "--PatchMatchStereo.num_iterations", "3",
                      "--PatchMatchStereo.num_samples", "10", "--PatchMatchStereo.window_step", "2",
                      "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.cache_size", "8"],
                     log, cwd=w, env=env, timeout=self.a.t_dense)
        self.run_cmd([cm, "stereo_fusion", "--workspace_path", "dense", "--workspace_format", "COLMAP", "--input_type", "geometric",
                      "--StereoFusion.max_image_size", "1200", "--StereoFusion.cache_size", "8", "--StereoFusion.num_threads", "8",
                      "--output_path", "dense/fused.ply"], log, cwd=w, env=env, timeout=600)
        ply = w / "dense" / "fused.ply"
        if not ply.exists():
            raise RuntimeError("fused.ply 없음")
        shutil.copy2(ply, d / "fused_dense.ply")
        shutil.copy2(log, d / "log_dense.txt")
        n = 0
        with open(ply, "rb") as f:
            for _ in range(30):
                line = f.readline().decode("ascii", "ignore")
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                    break
        shutil.rmtree(w / "dense" / "stereo", ignore_errors=True)   # 깊이맵 1 GB/구간 — 디스크 절약
        if n < 1000:
            raise RuntimeError(f"조밀 점 부족 {n}")
        return {"dense_pts": n}

    def do_map(self, seg):
        d = self.out / seg["name"]
        cmd = [PY, "-m", "litter.orbit_map", "--orbit", d, "--srt", self.a.srt, "--out", d / "map",
               "--litter", self.a.weights, "--classes", self.a.classes, "--min-track", self.a.min_track, "--conf", self.a.conf]
        txt = self.run_cmd(cmd, d / "log_map.txt", cwd=ROOT, timeout=self.a.t_map)
        info = {}
        m = re.search(r"잔차 ([\d.]+) m · 크기 기준 (\S+)", txt)
        if m:
            info["gps_rms_m"], info["scale_src"] = float(m[1]), m[2]
        objs = [r for r in csv.DictReader(open(d / "map" / "objects.csv", encoding="utf-8-sig")) if r["kind"] == "litter"]
        info["n_obj"] = len(objs)
        if not objs:
            raise RuntimeError("3D 프레임에서 상자 탐지 없음 (3프레임 이상 조건)")
        return info

    def do_vol(self, seg):
        d = self.out / seg["name"]
        ply = d / "fused_dense.ply"
        def run(search_r, sub):
            cmd = [PY, "-m", "litter.objvol", "--orbit", d, "--srt", self.a.srt, "--objects", d / "map" / "objects.csv",
                   "--out", d / sub, "--min-track", self.a.min_track, "--search-r", search_r, "--device", self.a.device,
                   "--det-weights", self.a.weights, "--det-classes", self.a.classes]   # 탐지 상자를 SAM 프롬프트로
            if ply.exists():
                cmd += ["--ply", ply]
            self.run_cmd(cmd, d / "log_vol.txt", cwd=ROOT, timeout=self.a.t_vol)
            res = json.loads((d / sub / "objvol.json").read_text(encoding="utf-8"))
            for r in res:
                r["search_r"] = search_r
            return res

        res = run(self.a.search_r, "objvol")
        # 1차(좁은 반경: 연석 영향 줄임)에서 못 재면 넓은 반경으로 한 번 더 — 상자 위치 오차 0.3~0.5 m 때
        retry = [r for r in res if "volume_heightmap_L" not in r]
        n_retry = 0
        if retry and self.a.search_r2 > self.a.search_r:
            res2 = {r["tag"]: r for r in run(self.a.search_r2, "objvol_r2")}
            for i, r in enumerate(res):
                r2 = res2.get(r["tag"])
                if "volume_heightmap_L" not in r and r2 and "volume_heightmap_L" in r2:
                    res[i] = r2
                    n_retry += 1
                    m = d / "objvol_r2" / f"{r['tag']}_masks.jpg"
                    if m.exists():
                        shutil.copy2(m, d / "objvol" / m.name)
            (d / "objvol" / "objvol.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = [r for r in res if "volume_heightmap_L" in r]
        return {"n_measured": len(ok), "n_obj": len(res), "dense": ply.exists(), "n_retry_ok": n_retry}

    def do_mosaic(self, seg):
        d = self.out / seg["name"]
        cmd = [PY, "-m", "litter.mosaic", "--orbit", d, "--srt", self.a.srt, "--objects", d / "map" / "objects.csv",
               "--out", d / "map", "--res", self.a.mosaic_res, "--min-track", self.a.min_track]
        self.run_cmd(cmd, d / "log_mosaic.txt", cwd=ROOT, timeout=self.a.t_mosaic)
        return {}

    def run_segment(self, seg):
        if not self._stage(seg, "sfm", lambda: self.do_sfm(seg), self.cpu):
            return
        dense_ok = self._stage(seg, "dense", lambda: self.do_dense(seg), self.gpu)
        if not self._stage(seg, "map", lambda: self.do_map(seg), self.gpu):
            return
        self._stage(seg, "vol", lambda: self.do_vol(seg), self.gpu)
        if self.a.mosaic:
            self._stage(seg, "mosaic", lambda: self.do_mosaic(seg), self.cpu)

    def run_all(self):
        t0 = time.time()
        threads = []
        for seg in self.segs:
            th = threading.Thread(target=self.run_segment, args=(seg,), name=seg["name"], daemon=True)
            th.start()
            threads.append(th)
            time.sleep(2)
        for th in threads:
            th.join()
        self.log(f"전체 완료 {(time.time() - t0) / 60:.1f}분")
        return time.time() - t0


# ------------------------------------------------------------------ 결과 통합
def _dist_m(lat1, lon1, lat2, lon2):
    k = math.cos(math.radians(lat1))
    return math.hypot((lat2 - lat1) * 111320, (lon2 - lon1) * 111320 * k)


def _classify(stages, matched, vol):
    """상태 문자열, 실패 분류."""
    if stages.get("sfm", {}).get("status") != "ok":
        return "3D 실패", "3D 뼈대 실패(등록 부족)"
    if stages.get("map", {}).get("status") != "ok":
        why = stages.get("map", {}).get("reason") or ""
        if "SVD" in why or "nan" in why.lower():
            return "3D 탐지 없음", "3D↔GPS 정합 실패(축척 NaN)"
        if "탐지 없음" in why:
            return "3D 탐지 없음", "3D 프레임에서 탐지 안 됨"
        return "3D 탐지 없음", ("orbit_map 오류: " + why[:60]) if why else "3D 프레임에서 탐지 안 됨"
    if not matched:
        return "3D 탐지 없음", "3D 탐지 물체와 위치 매칭 실패"
    if stages.get("vol", {}).get("status") != "ok" or vol is None:
        return "부피 실패", "objvol 오류"
    if "volume_heightmap_L" in vol:
        tilt = vol.get("ground_plane_tilt_deg", 0)
        if tilt >= 15:
            return "부피 OK(바닥 기울기 주의)", f"바닥 평면 {tilt:.0f}° 기울어짐"
        if vol.get("points_object", 0) < 150:
            return "부피 OK(점 부족 주의)", f"물체 점 {vol.get('points_object', 0)}개뿐"
        return "부피 OK", ""
    if vol.get("ground_plane_tilt_deg", 0) > 20:
        return "부피 실패", f"바닥 평면 틀어짐 {vol['ground_plane_tilt_deg']:.0f}° (연석·경사, 뷰 {vol.get('n_views_excluded', 0)}개 제외)"
    if vol.get("views_used", 0) == 0:
        return "부피 실패", "마스크 실패(SAM 마스크 없음)"
    return "부피 실패", f"점 부족(물체 점 {vol.get('points_object', 0)})"


def _confidence(vol, gps_rms):
    """부피 신뢰 플래그: 물체 점 수 · 사용 뷰 수 · 바닥 평면 기울기 · 3D↔GPS 잔차."""
    if vol is None or "volume_heightmap_L" not in vol:
        return ""
    pts, views, tilt = vol.get("points_object", 0), vol.get("views_used", 0), vol.get("ground_plane_tilt_deg", 0)
    rms = float(gps_rms) if gps_rms not in ("", None) else 9.9
    why = []
    if pts < 150:
        why.append(f"점 {pts}")
    if views < 5:
        why.append(f"뷰 {views}")
    if tilt >= 15:
        why.append(f"기울기 {tilt:.0f}°")
    if rms > 1.0:
        why.append(f"GPS잔차 {rms:.1f}m")
    if why:
        return "낮음(" + ",".join(why) + ")"
    if pts >= 400 and views >= 8 and tilt < 10 and rms < 0.5:
        return "높음"
    return "중간"


def aggregate(out, boxes, segs, status, srt, total_sec=None):
    from .seg import _imread, _imwrite
    import cv2

    out = Path(out)
    rows, thumbs = [], {}
    for seg in segs:
        name = seg["name"]
        st = status.get(name, {})
        stages = st.get("stages", {})
        d = out / name
        objs, vols = [], {}
        if stages.get("map", {}).get("status") == "ok":
            objs = [r for r in csv.DictReader(open(d / "map" / "objects.csv", encoding="utf-8-sig")) if r["kind"] == "litter"]
            for o in objs:
                o["lat"], o["lon"] = float(o["lat"]), float(o["lon"])
        vj = d / "objvol" / "objvol.json"
        if vj.exists():
            for r in json.loads(vj.read_text(encoding="utf-8")):
                vols[tuple(round(v, 2) for v in r["csv_local_xy"])] = r
        used = set()
        sfm = stages.get("sfm", {})
        for bid in seg["boxes"]:
            b = boxes[bid]
            best, bd = None, 1e9
            for k, o in enumerate(objs):
                if k in used:
                    continue
                dd = _dist_m(b["lat"], b["lon"], o["lat"], o["lon"])
                if dd < bd:
                    best, bd = k, dd
            matched = best is not None and bd <= 12.0
            o = objs[best] if matched else None
            if matched:
                used.add(best)
            vol = vols.get((round(float(o["local_x_m"]), 2), round(float(o["local_y_m"]), 2))) if matched else None
            state, reason = _classify(stages, matched, vol)
            if state == "3D 실패" and sfm.get("reason"):
                reason = sfm["reason"][:80]
            r = {"box": bid, "t_s": round(b["t"], 1), "seg": name,
                 "lat": o["lat"] if matched else b["lat"], "lon": o["lon"] if matched else b["lon"],
                 "latlon_src": "3D(orbit_map)" if matched else "video_map",
                 "match_dist_m": round(bd, 1) if matched else "",
                 "frames_reg": sfm.get("reg", ""), "frames_ext": sfm.get("ext", ""),
                 "sparse_pts": sfm.get("pts", ""), "dense_pts": stages.get("dense", {}).get("dense_pts", ""),
                 "scale_src": stages.get("map", {}).get("scale_src", ""), "gps_rms_m": stages.get("map", {}).get("gps_rms_m", ""),
                 "det_frames_3d": o["seen_frames"] if matched else "",
                 "L_cm": "", "W_cm": "", "H_cm": "", "vol_med_L": "", "vol_max_L": "", "vol_mask_L": "",
                 "plane_tilt_deg": "", "views_used": "", "views_excluded": "", "points_object": "",
                 "confidence": _confidence(vol, stages.get("map", {}).get("gps_rms_m", "")),
                 "status": state, "reason": reason, "viewer": f"{name}/viewer.html", "mask_img": ""}
            if vol:
                r.update(plane_tilt_deg=round(vol.get("ground_plane_tilt_deg", 0), 1), views_used=vol.get("views_used", ""),
                         views_excluded=vol.get("n_views_excluded", ""), points_object=vol.get("points_object", ""))
                if "volume_heightmap_L" in vol:
                    r.update(L_cm=round(vol["length_m"] * 100, 1), W_cm=round(vol["width_m"] * 100, 1), H_cm=round(vol["height_m"] * 100, 1),
                             vol_med_L=round(vol["volume_heightmap_median_L"], 1), vol_max_L=round(vol["volume_heightmap_max_L"], 1),
                             vol_mask_L=round(vol["volume_mask_box_L"], 1) if vol.get("volume_mask_box_L") else "")
                mi = d / "objvol" / f"{vol['tag']}_masks.jpg"
                if mi.exists():
                    r["mask_img"] = f"{name}/objvol/{mi.name}"
                    im = _imread(mi)
                    if im is not None:
                        thumbs[bid] = im[:240, :240]
            rows.append(r)
    rows.sort(key=lambda r: r["box"])
    cols = list(rows[0].keys()) if rows else []
    with open(out / "objects3d.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 요약
    n_seg = len(segs)
    rate = {}
    for s in STAGES:
        done = sum(1 for seg in segs if status.get(seg["name"], {}).get("stages", {}).get(s, {}).get("status") == "ok")
        secs = [status[seg["name"]]["stages"][s]["sec"] for seg in segs
                if status.get(seg["name"], {}).get("stages", {}).get(s, {}).get("status") in ("ok", "fail")
                and "sec" in status[seg["name"]]["stages"][s]]
        if secs or done:
            rate[s] = {"ok": done, "of": n_seg, "mean_sec": round(float(np.mean(secs)), 1) if secs else None}
    n_vol = sum(1 for r in rows if r["status"].startswith("부피 OK"))
    reasons = {}
    for r in rows:
        if not r["status"].startswith("부피 OK"):
            key = r["reason"].split("(")[0].split(" ")[0] if r["reason"] else r["status"]
            reasons[key] = reasons.get(key, 0) + 1
    summary = {"segments": n_seg, "boxes": len(rows), "stage_success": rate,
               "boxes_located_3d": sum(1 for r in rows if r["latlon_src"].startswith("3D")),
               "boxes_volume_ok": n_vol, "failure_reasons": reasons,
               "total_sec": round(total_sec, 1) if total_sec else None, "generated": _now()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 마스크 썸네일 모음
    tiles = []
    for r in rows:
        im = thumbs.get(r["box"])
        if im is None:
            im = np.full((240, 240, 3), 40, np.uint8)
        im = cv2.resize(im, (240, 240)).copy()
        lab = f"#{r['box']} {r['t_s']:.0f}s " + (f"{r['vol_med_L']}L" if r["vol_med_L"] != "" else r["status"][:6])
        cv2.rectangle(im, (0, 0), (240, 22), (0, 0, 0), -1)
        cv2.putText(im, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros((240, 240, 3), np.uint8))
        _imwrite(out / "masks_sheet.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), 85)

    # 지도 HTML
    from .telemetry import read_srt
    tel = [r for r in read_srt(srt) if "lat" in r]
    track = [[r["lat"], r["lon"]] for r in tel[::30]]
    objs = []
    for r in rows:
        th = thumbs.get(r["box"])
        b64 = ""
        if th is not None:
            ok, buf = cv2.imencode(".jpg", cv2.resize(th, (160, 160)), [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode() if ok else ""
        objs.append({k: r[k] for k in ("box", "t_s", "lat", "lon", "status", "reason", "L_cm", "W_cm", "H_cm", "vol_med_L", "vol_max_L",
                                       "vol_mask_L", "frames_reg", "frames_ext", "dense_pts", "plane_tilt_deg", "latlon_src", "viewer", "confidence")} | {"thumb": b64})
    html = _MAP_HTML.replace("__DATA__", json.dumps({"track": track, "objs": objs}, ensure_ascii=False)) \
        .replace("__NOTE__", f"영상 0015 · 상자 {len(rows)}개 / 구간 {n_seg}개 · 부피 측정 {n_vol}개 · 3D 위치 {summary['boxes_located_3d']}개")
    (out / "map.html").write_text(html, encoding="utf-8")

    # 지도 PNG
    try:
        from pipeline._compat import apply_korean_font
        apply_korean_font()
    except Exception:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10))
    tr = np.array([[r["lon"], r["lat"]] for r in tel[::15]])
    ax.plot(tr[:, 0], tr[:, 1], "-", c="#1d3557", lw=1.2, label="비행 경로 (SRT)")
    col = {"부피 OK": "#2a9d8f", "부피 OK(바닥 기울기 주의)": "#8ab17d", "부피 OK(점 부족 주의)": "#b5c99a", "부피 실패": "#e9c46a", "3D 탐지 없음": "#f4a261", "3D 실패": "#e63946"}
    seen_lab = set()
    for r in rows:
        c = col.get(r["status"], "#999")
        lab = r["status"] if r["status"] not in seen_lab else None
        seen_lab.add(r["status"])
        ax.scatter(r["lon"], r["lat"], s=70, c=c, edgecolors="k", lw=.6, zorder=3, label=lab)
        txt = f"#{r['box']}" + (f" {r['vol_med_L']:.0f}L" if r["vol_med_L"] != "" else "")
        ax.annotate(txt, (r["lon"], r["lat"]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_aspect(1 / math.cos(math.radians(float(np.mean(tr[:, 1])))))
    ax.grid(alpha=.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"영상 0015 상자 {len(rows)}개 — 3D 위치·부피 (부피 OK {n_vol}, 3D 위치 {summary['boxes_located_3d']})")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(out / "map.png", dpi=130)
    plt.close(fig)
    return rows, summary


_MAP_HTML = """<!doctype html><meta charset="utf-8"><title>0015 상자 3D·부피 지도</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}#n{position:absolute;z-index:999;left:50px;top:8px;background:#fffd;padding:6px 10px;border-radius:6px;font:13px sans-serif;line-height:1.5}
.leaflet-popup-content{font:12px/1.4 sans-serif}.leaflet-popup-content img{display:block;margin-top:4px}</style>
<div id="m"></div><div id="n">__NOTE__<br>
<span style="color:#2a9d8f">●</span> 부피 OK <span style="color:#8ab17d">●</span> 부피 OK(바닥 기울기 주의) <span style="color:#b5c99a">●</span> 부피 OK(점 부족 주의) <span style="color:#e9c46a">●</span> 부피 실패 <span style="color:#f4a261">●</span> 3D 탐지 없음(영상 지도 위치) <span style="color:#e63946">●</span> 3D 실패 <span style="color:#1d3557">━</span> 비행 경로</div>
<script>
const D=__DATA__;
const col={"부피 OK":"#2a9d8f","부피 OK(바닥 기울기 주의)":"#8ab17d","부피 OK(점 부족 주의)":"#b5c99a","부피 실패":"#e9c46a","3D 탐지 없음":"#f4a261","3D 실패":"#e63946"};
const m=L.map('m',{maxZoom:22});
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:22,maxNativeZoom:19,attribution:'Esri'}).addTo(m);
const osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:22,maxNativeZoom:19,attribution:'OSM'});
L.control.layers({'위성':sat,'지도':osm}).addTo(m);
L.polyline(D.track,{color:'#1d3557',weight:3}).addTo(m);
D.objs.forEach(o=>{const c=col[o.status]||'#999';
 let h=`<b>상자 #${o.box}</b> (${o.t_s}초) · ${o.status}<br>${o.lat.toFixed(6)}, ${o.lon.toFixed(6)} <small>(${o.latlon_src})</small><br>`;
 if(o.L_cm!=='') h+=`크기 ${o.L_cm}×${o.W_cm}×${o.H_cm} cm<br>부피 높이지도 ${o.vol_med_L} L(중앙값) / ${o.vol_max_L} L(최대)`+(o.vol_mask_L!==''?` · 마스크 ${o.vol_mask_L} L`:'')+` · 신뢰 ${o.confidence}<br>`;
 if(o.reason) h+=`<span style="color:#b00">${o.reason}</span><br>`;
 h+=`3D 등록 ${o.frames_reg}/${o.frames_ext} · 조밀 점 ${o.dense_pts||'-'}`+(o.plane_tilt_deg!==''?` · 바닥 기울기 ${o.plane_tilt_deg}°`:'')+`<br>`;
 h+=`<a href="${o.viewer}" target="_blank">3D 뷰어 열기</a>`;
 if(o.thumb) h+=`<img src="data:image/jpeg;base64,${o.thumb}">`;
 L.circleMarker([o.lat,o.lon],{radius:8,color:'#222',weight:1,fillColor:c,fillOpacity:.9}).bindTooltip(`#${o.box}`+(o.vol_med_L!==''?` ${Math.round(o.vol_med_L)}L`:''),{permanent:true,direction:'top',offset:[0,-6],className:'lab'}).bindPopup(h,{maxWidth:320}).addTo(m);});
const all=D.track.concat(D.objs.map(o=>[o.lat,o.lon]));
m.fitBounds(L.latLngBounds(all).pad(0.15));
</script>"""


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.batch3d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sfm", help="(내부) 한 구간 프레임 추출 + 뼈대")
    s.add_argument("--video", required=True)
    s.add_argument("--start", type=float, required=True)
    s.add_argument("--end", type=float, required=True)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--out", required=True)

    r = sub.add_parser("run", help="전체 일괄 실행")
    r.add_argument("--video", required=True)
    r.add_argument("--srt", required=True)
    r.add_argument("--objects", required=True, help="video_map objects.csv (first_frame 포함)")
    r.add_argument("--out", required=True)
    r.add_argument("--work", default=r"C:\work\b15", help="COLMAP 영문 작업 폴더")
    r.add_argument("--only", help="상자 번호만 (예: 24,19) — 동작 확인용")
    r.add_argument("--half", type=float, default=4.0, help="상자 시각 앞뒤 초")
    r.add_argument("--cls", default="cardboard box")
    r.add_argument("--weights", default="yolov8s-worldv2.pt")
    r.add_argument("--classes", default="cardboard box")
    r.add_argument("--conf", type=float, default=0.3)
    r.add_argument("--min-track", dest="min_track", type=float, default=4.0)
    r.add_argument("--search-r", dest="search_r", type=float, default=0.45)
    r.add_argument("--search-r2", dest="search_r2", type=float, default=0.8, help="1차 실패 시 재시도 반경 (0이면 안 함)")
    r.add_argument("--device", default="0")
    r.add_argument("--cpu-jobs", dest="cpu_jobs", type=int, default=2)
    r.add_argument("--omp", type=int, default=6)
    r.add_argument("--mosaic", action="store_true", help="정사영상도 (CPU, 느림)")
    r.add_argument("--mosaic-res", dest="mosaic_res", type=float, default=0.02)
    r.add_argument("--t-sfm", dest="t_sfm", type=int, default=900)
    r.add_argument("--t-dense", dest="t_dense", type=int, default=900)
    r.add_argument("--t-map", dest="t_map", type=int, default=600)
    r.add_argument("--t-vol", dest="t_vol", type=int, default=600)
    r.add_argument("--t-mosaic", dest="t_mosaic", type=int, default=600)
    r.add_argument("--plan-only", dest="plan_only", action="store_true")
    r.add_argument("--aggregate-only", dest="aggregate_only", action="store_true", help="실행 없이 결과 통합만")

    a = ap.parse_args(argv)
    if a.cmd == "sfm":
        stage_sfm(a.video, a.start, a.end, a.n, a.out)
        return

    boxes, segs = plan_segments(a.objects, half=a.half, cls=a.cls)
    if a.only:
        want = {int(x) for x in a.only.split(",")}
        segs = [s for s in segs if want & set(s["boxes"])]
    print(f"상자 {len(boxes)}개 → 구간 {len(segs)}개")
    for s in segs:
        print(f"  {s['name']}: {s['start']:.1f}~{s['end']:.1f}s n={s['n']} 상자 {s['boxes']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"boxes": boxes, "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.plan_only:
        return
    runner = Runner(a, boxes, segs)
    total = None
    if not a.aggregate_only:
        total = runner.run_all()
    rows, summary = aggregate(out, runner.boxes, segs, runner.status, a.srt, total)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"#{r['box']:>2} {r['t_s']:6.1f}s {r['seg']:<16} 등록 {r['frames_reg']}/{r['frames_ext']} 조밀 {r['dense_pts'] or '-'} "
              f"{r['L_cm']}×{r['W_cm']}×{r['H_cm']} cm {r['vol_med_L']} L 신뢰 {r['confidence'] or '-'}  {r['status']} {r['reason']}")
    print(f"→ {out / 'objects3d.csv'} · map.html · map.png · summary.json · masks_sheet.jpg")


if __name__ == "__main__":
    main()
