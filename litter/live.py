"""
실시간 쓰레기 탐지 + 핑 — 휴대폰 미러링(scrcpy) 화면에서 탐지하고, 사람이 확인하면
DJI Fly 화면을 자동으로 터치하고, 핑(시각·화면 위치·스냅샷)을 기록한다.
비행 후 SD카드 영상의 SRT(프레임별 드론 GPS·고도)와 시각을 맞춰 쓰레기 GPS를 구한다.

  [드론] → [조종기 + 폰(DJI Fly)] ──USB──→ [노트북: scrcpy 창]
                ↑                                 │ ① 창 캡처 → YOLO 탐지
                └──── ③ adb 터치 ←── ② 사람이 확인(스페이스/숫자/클릭)
                                                  │ ④ pings.csv + 스냅샷
  비행 후: python -m litter.live match --pings ... --srt DJI_0001.SRT → 핑마다 위경도

왜 미러링인가: DJI Mini 5 Pro는 (2026-10 기준) DJI Mobile SDK 미지원이라 앱을 만들어
영상·텔레메트리를 직접 받을 수 없다. 미러링 화면은 볼 수 있고, adb로 터치도 된다.

준비 (한 번만):
  - 폰: 설정 → 휴대전화 정보 → 소프트웨어 정보 → 빌드번호 7번 탭 → 개발자 옵션 → USB 디버깅 켜기
  - 노트북: scrcpy 설치됨 (winget Genymobile.scrcpy, adb 포함)
  - DJI Fly: 카메라 설정 → 동영상 자막(SRT) 켜기. 탐지 비행은 짐벌 -90°(수직 아래) 권장

실행:
  scrcpy --window-title=DRONE --max-fps=30 --stay-awake
  python -m litter.live screen --window DRONE --weights best.pt --out runs/live/2026-10-01

조작 (탐지 창에서):
  스페이스  가장 확실한 탐지를 터치 + 핑      1~9   해당 번호 탐지를 터치 + 핑
  마우스 클릭  클릭한 박스를 터치 + 핑         p     터치 없이 핑만 (가장 확실한 탐지)
  q  종료
안전: 터치는 '영상 안전영역'(기본: 가장자리 버튼 줄 제외) 안의 탐지에만 한다.
      이륙·착륙·귀환 버튼을 잘못 누르지 않게 하기 위함. --safe 로 조정.

동영상 파일/RTMP로 시험:  python -m litter.live stream --source test.mp4 --weights best.pt
"""
import argparse
import csv
import ctypes
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ 입력: 화면 캡처
class WindowCapture:
    """Windows 창(scrcpy)의 클라이언트 영역을 캡처. 창을 옮기거나 크기를 바꿔도 매번 다시 찾는다."""

    def __init__(self, title):
        import mss

        try:  # 고해상도 화면 배율(125% 등)에서도 실제 픽셀 좌표를 쓰도록
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        self.title = title
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self.u32 = ctypes.windll.user32
        t0 = time.time()
        while not self._hwnd():  # 미러링이 막 시작됐거나 재연결 중이면 잠시 기다린다
            if time.time() - t0 > 60:
                raise SystemExit(f"'{title}' 창을 못 찾음 — scrcpy --window-title={title} 로 실행했는지 확인")
            time.sleep(0.5)

    def _hwnd(self):
        return self.u32.FindWindowW(None, self.title)

    def rect(self):
        class R(ctypes.Structure):
            _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]

        class P(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        h = self._hwnd()
        if not h:
            return None
        rc, p = R(), P(0, 0)
        self.u32.GetClientRect(h, ctypes.byref(rc))
        self.u32.ClientToScreen(h, ctypes.byref(p))
        return p.x, p.y, rc.r, rc.b

    def _printwindow(self, w, h):
        """창 자체의 내용을 그린다 (다른 창에 가려져도 됨). PW_CLIENTONLY|PW_RENDERFULLCONTENT."""
        g32, u32 = ctypes.windll.gdi32, self.u32
        hwnd = self._hwnd()
        hdc = u32.GetDC(hwnd)
        mdc = g32.CreateCompatibleDC(hdc)
        bmp = g32.CreateCompatibleBitmap(hdc, w, h)
        g32.SelectObject(mdc, bmp)
        ok = u32.PrintWindow(hwnd, mdc, 1 | 2)

        class BIH(ctypes.Structure):
            _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                        ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
                        ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
                        ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
                        ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32)]

        bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
        g32.DeleteObject(bmp)
        g32.DeleteDC(mdc)
        u32.ReleaseDC(hwnd, hdc)
        if not ok:
            return None
        return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    def read(self):
        r = self.rect()
        if not r or r[2] < 50 or r[3] < 50:
            return False, None
        x, y, w, h = r
        img = None
        if getattr(self, "use_pw", True):
            img = self._printwindow(w, h)
            # 처음 몇 번 새까맣게 나오면 이 렌더러는 PrintWindow가 안 되는 것 → 화면 캡처로
            if img is None or img.max() == 0:
                self.black = getattr(self, "black", 0) + 1
                if self.black >= 10:
                    self.use_pw = False
                    print("  ⚠️ 창 직접 캡처가 안 돼서 화면 캡처로 전환 — 탐지 창을 미러링 창과 겹치지 않게 두세요")
                img = None
        if img is None:
            img = np.asarray(self.sct.grab({"left": x, "top": y, "width": w, "height": h}))[:, :, :3]
        return True, np.ascontiguousarray(img)


class StreamCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"스트림을 못 엶: {source}")

    def read(self):
        return self.cap.read()


# ------------------------------------------------------------------ 출력: adb 터치
def find_adb():
    a = shutil.which("adb")
    if a:
        return a
    base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    hit = sorted(base.glob("Genymobile.scrcpy*/scrcpy-win64-*/adb.exe"))
    return str(hit[-1]) if hit else None


class Tapper:
    """캡처 창 좌표 → 폰 화면 좌표로 바꿔 adb input tap."""

    def __init__(self, adb=None, serial=None, dry=False):
        self.adb = adb or find_adb()
        self.serial = serial
        self.dry = dry or not self.adb
        self.size = None
        if not self.adb:
            print("  ⚠️ adb를 못 찾음 — 터치 없이 핑만 기록합니다")
        elif not dry:
            out = self._run("shell", "wm", "size")
            m = re.findall(r"(\d+)x(\d+)", out or "")
            if m:
                w, h = map(int, m[-1])  # Override size가 있으면 그게 마지막
                self.size = (max(w, h), min(w, h))  # DJI Fly는 가로 화면
                print(f"  폰 화면 {self.size[0]}×{self.size[1]} (가로) · adb {self.adb}")
            else:
                print(f"  ⚠️ adb 연결 안 됨 (USB 디버깅 허용 확인) — 터치 없이 핑만 기록: {out!r}")
                self.dry = True

    def _run(self, *args):
        cmd = [self.adb] + (["-s", self.serial] if self.serial else []) + list(args)
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except Exception as e:
            return f"ERR {e}"

    def tap(self, x, y, frame_wh):
        """x, y: 캡처 프레임 픽셀. scrcpy 창은 폰 화면을 비율 유지로 확대/축소한다."""
        if self.dry or not self.size:
            return None
        W, H = frame_wh
        # 가로/세로 방향은 미러링 화면 모양으로 판단 (DJI Fly는 가로, 설정 화면 등은 세로일 수 있음)
        pw, ph = (max(self.size), min(self.size)) if W >= H else (min(self.size), max(self.size))
        s = min(W / pw, H / ph)
        ox, oy = (W - pw * s) / 2, (H - ph * s) / 2  # 창을 비율과 다르게 늘렸을 때 생기는 검은 띠
        px, py = int((x - ox) / s), int((y - oy) / s)
        if not (0 <= px < pw and 0 <= py < ph):
            return None
        self._run("shell", "input", "tap", str(px), str(py))
        return px, py

    def _to_phone(self, x, y, frame_wh):
        W, H = frame_wh
        pw, ph = (max(self.size), min(self.size)) if W >= H else (min(self.size), max(self.size))
        s = min(W / pw, H / ph)
        ox, oy = (W - pw * s) / 2, (H - ph * s) / 2
        return int(np.clip((x - ox) / s, 0, pw - 1)), int(np.clip((y - oy) / s, 0, ph - 1))

    def drag_box(self, b, frame_wh, pad=0.25, min_px=120, dur_ms=600):
        """물체 둘레에 네모를 그리는 드래그 (DJI Fly FocusTrack 대상 선택 동작).
        박스를 조금 키우고(pad), 너무 작으면 최소 크기로 — 손가락 드래그처럼."""
        if self.dry or not self.size:
            return None
        x0, y0, x1, y1 = map(float, b)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        p0 = self._to_phone(x0, y0, frame_wh)
        p1 = self._to_phone(x1, y1, frame_wh)
        pc = self._to_phone(cx, cy, frame_wh)
        hw = max((p1[0] - p0[0]) * (1 + pad) / 2, min_px / 2)
        hh = max((p1[1] - p0[1]) * (1 + pad) / 2, min_px / 2)
        a = (int(pc[0] - hw), int(pc[1] - hh))
        z = (int(pc[0] + hw), int(pc[1] + hh))
        self._run("shell", "input", "swipe", str(a[0]), str(a[1]), str(z[0]), str(z[1]), str(dur_ms))
        return a, z


# ------------------------------------------------------------------ 탐지
def _tiles(W, H, n):
    if n <= 1:
        return [(0, 0, W, H)]
    tw, th = int(np.ceil(W / n * 1.15)), int(np.ceil(H / n * 1.15))  # 15% 겹침
    xs = np.linspace(0, W - tw, n).astype(int)
    ys = np.linspace(0, H - th, n).astype(int)
    return [(x, y, tw, th) for y in ys for x in xs]


def video_area(frame, thr=12):
    """DJI Fly 화면에서 실제 영상이 차지하는 영역 (좌우/상하 검은 띠 제외).
    20:9 폰에 16:9 영상이면 좌우에 띠가 생긴다. 핑 위치를 원본 4K 프레임에 옮길 때 필요."""
    g = frame.max(axis=2)
    cols = np.nonzero(np.percentile(g, 90, axis=0) > thr)[0]
    rows = np.nonzero(np.percentile(g, 90, axis=1) > thr)[0]
    if len(cols) < 10 or len(rows) < 10:
        return 0, 0, frame.shape[1], frame.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


class Detector:
    def __init__(self, weights, conf=0.3, tiles=2, imgsz=1024):
        from ultralytics import YOLO

        self.m = YOLO(str(weights))
        self.conf, self.tiles, self.imgsz = conf, tiles, imgsz

    def __call__(self, frame, area=None):
        from .seg import _ios

        x0, y0, x1, y1 = area or (0, 0, frame.shape[1], frame.shape[0])
        sub = frame[y0:y1, x0:x1]
        tl = _tiles(sub.shape[1], sub.shape[0], self.tiles)
        res = self.m.predict([sub[y:y + h, x:x + w] for x, y, w, h in tl], conf=self.conf, imgsz=self.imgsz, verbose=False)
        dets = []
        for (x, y, w, h), r in zip(tl, res):
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dets.append((b + [x + x0, y + y0, x + x0, y + y0], int(c), float(s)))
        dets.sort(key=lambda d: -d[2])
        kept = []
        for d in dets:
            if all(_ios(d[0], k[0]) < 0.6 for k in kept):
                kept.append(d)
        return kept


# ------------------------------------------------------------------ 실시간 루프
PING_COLS = ["ping_id", "wall_time", "epoch", "frame", "cls", "score", "x0", "y0", "x1", "y1",
             "u", "v", "va_x0", "va_y0", "va_x1", "va_y1", "tapped", "phone_x", "phone_y", "snapshot"]


def run_live(cap, det, out, safe=(0.12, 0.12, 0.88, 0.82), tapper=None, every=1, show=True, action="drag"):
    """action: "drag" = 물체 둘레 네모 드래그 (DJI Fly FocusTrack 대상 선택), "tap" = 한 번 탭 (초점)"""
    out = Path(out)
    (out / "snap").mkdir(parents=True, exist_ok=True)
    pf = open(out / "pings.csv", "a", newline="", encoding="utf-8")
    pw = csv.DictWriter(pf, fieldnames=PING_COLS)
    if pf.tell() == 0:
        pw.writeheader()
    df = open(out / "detections.csv", "a", newline="", encoding="utf-8")
    dw = csv.writer(df)
    if df.tell() == 0:
        dw.writerow(["epoch", "frame", "cls", "score", "x0", "y0", "x1", "y1"])
    names = det.m.names
    state = {"click": None}
    win = "litter live  [space/click] select+ping  [1-9] pick  [t] tap  [p] ping only  [q] quit"
    if show:
        # AUTOSIZE: 창 크기 = 표시 이미지 크기 → 마우스 좌표를 그대로 환산할 수 있다
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, lambda e, x, y, f, p: state.update(click=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    i, n_ping, fps, dets, area, disp_scale = 0, 0, 0.0, [], None, 1.0
    msg, msg_t = "", 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(cap, WindowCapture):
                time.sleep(0.2)
                continue
            break
        i += 1
        H, W = frame.shape[:2]
        if area is None or i % 30 == 0:  # 영상 영역은 가끔 다시 잰다 (창 크기 변경 대응)
            area = video_area(frame)
        ax0, ay0, ax1, ay1 = area
        sx0, sy0 = ax0 + safe[0] * (ax1 - ax0), ay0 + safe[1] * (ay1 - ay0)
        sx1, sy1 = ax0 + safe[2] * (ax1 - ax0), ay0 + safe[3] * (ay1 - ay0)
        if i % every == 0:
            tic = time.time()
            dets = det(frame, area)
            fps = 0.9 * fps + 0.1 / max(time.time() - tic, 1e-3)
            now = time.time()
            for b, c, s in dets:
                dw.writerow([f"{now:.3f}", i, names[c], f"{s:.3f}", *map(int, b)])

        def in_safe(b):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            return sx0 <= cx <= sx1 and sy0 <= cy <= sy1

        def ping(d, do_tap):
            nonlocal n_ping, msg, msg_t
            b, c, s = d
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            tapped = None
            if do_tap and tapper and in_safe(b):
                if action == "drag":
                    dz = tapper.drag_box(b, (W, H))
                    tapped = dz[0] if dz else None
                    msg = f"DRAG box -> tablet {dz}" if dz else "DRAG FAILED (adb not connected?)"
                else:
                    tapped = tapper.tap(cx, cy, (W, H))
                    msg = f"TAP -> tablet {tapped}" if tapped else "TAP FAILED (adb not connected?)"
            elif do_tap and not in_safe(b):
                msg = "outside safe zone: ping only (no tap)"
                print("  ⚠️ 안전영역 밖이라 터치하지 않음 (핑만 기록)")
            elif do_tap:
                msg = "no adb: ping only"
            else:
                msg = "ping only"
            msg_t = time.time()
            n_ping += 1
            snap = f"snap/ping_{n_ping:04d}.jpg"
            v = frame.copy()
            cv2.rectangle(v, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 3)
            cv2.imwrite(str(out / snap), v)
            now = time.time()
            row = {"ping_id": n_ping, "wall_time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                   "epoch": f"{now:.3f}", "frame": i, "cls": names.get(c, c) if isinstance(names, dict) else c,
                   "score": f"{s:.3f}",
                   "x0": int(b[0]), "y0": int(b[1]), "x1": int(b[2]), "y1": int(b[3]),
                   # 영상 영역 기준 정규화 좌표 (0~1) — 원본 4K 프레임의 같은 위치
                   "u": f"{(cx - ax0) / (ax1 - ax0):.4f}", "v": f"{(cy - ay0) / (ay1 - ay0):.4f}",
                   "va_x0": ax0, "va_y0": ay0, "va_x1": ax1, "va_y1": ay1,
                   "tapped": int(tapped is not None), "phone_x": tapped[0] if tapped else "",
                   "phone_y": tapped[1] if tapped else "", "snapshot": snap}
            pw.writerow(row)
            pf.flush()
            print(f"  📍 핑 {n_ping}: {names.get(c, c) if isinstance(names, dict) else c} {s:.2f} (u={row['u']}, v={row['v']}) "
                  f"{'터치함' if tapped else '터치 안 함'}")

        # 화면 표시
        vis = frame.copy()
        cv2.rectangle(vis, (int(sx0), int(sy0)), (int(sx1), int(sy1)), (0, 200, 0), 1)
        for k, (b, c, s) in enumerate(dets[:9]):
            col = (0, 0, 255) if in_safe(b) else (128, 128, 128)
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.putText(vis, f"{k + 1}:{names[c]} {s:.2f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2)
        cv2.putText(vis, f"{len(dets)} det | {fps:.1f}/s | ping {n_ping}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        if msg and time.time() - msg_t < 2.0:
            cv2.putText(vis, msg, (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2)
        if not show:
            continue
        # 표시 크기 고정 (최대 폭 1100) — 클릭 좌표 = 표시좌표 × disp_scale
        disp_scale = max(1.0, W / 1100)
        cv2.imshow(win, cv2.resize(vis, (int(W / disp_scale), int(H / disp_scale))) if disp_scale > 1 else vis)
        key = cv2.waitKey(1) & 0xFF
        safe_dets = [d for d in dets if in_safe(d[0])]
        if key == ord("q"):
            break
        if key == ord(" ") and safe_dets:
            ping(safe_dets[0], True)
        elif key == ord("t") and safe_dets and tapper:  # 탭 한 번 (초점 맞추기 등)
            b = safe_dets[0][0]
            tp = tapper.tap((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (W, H))
            msg, msg_t = f"TAP -> tablet {tp}", time.time()
        elif key == ord("p") and dets:
            ping(dets[0], False)
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(dets[:9]):
            ping(dets[key - ord("1")], True)
        if state["click"]:
            cx, cy = state["click"][0] * disp_scale, state["click"][1] * disp_scale
            state["click"] = None
            hit = [d for d in dets if d[0][0] <= cx <= d[0][2] and d[0][1] <= cy <= d[0][3]]
            if hit:  # 박스를 클릭 → 박스 중심 터치
                ping(hit[0], True)
            else:    # 빈 곳 클릭 → 그 지점 터치 (사람이 직접 고른 대상, 클래스 'manual')
                r = 12
                ping((np.array([cx - r, cy - r, cx + r, cy + r]), "manual", 1.0), True)
    pf.close()
    df.close()
    cv2.destroyAllWindows()
    print(f"핑 {n_ping}개 → {out / 'pings.csv'}")


# ------------------------------------------------------------------ 비행 후: 핑 ↔ SRT → GPS
def match(pings, srt, out, video=None, offset_s=0.0, f35=24.0, pitch=-90.0, yaw=None, frame_wh=(3840, 2160),
          epsg=None):
    """핑 시각 ↔ SRT 프레임 시각 → 그 순간 드론 위치·고도 → 핑 화면 위치를 지면 좌표로.

    offset_s: 노트북 시계 − 드론 시계 (초). 비행 전 노트북 화면 시계를 드론으로 찍어 두면 잴 수 있다.
    yaw: SRT에 짐벌/기체 방향이 없으면 → GPS 이동 방향으로 추정 (앞으로 비행할 때만 맞음) 또는 직접 지정.
    """
    from .camera import Intrinsics, Pose, pixel_to_ground
    from .geometry import Geo
    from .telemetry import read_srt

    tel = [r for r in read_srt(srt) if r.get("dt") and "lat" in r and "lon" in r]
    if not tel:
        raise SystemExit("SRT에 시각·위경도가 없음 — DJI Fly에서 '동영상 자막'을 켰는지 확인")
    t = np.array([r["dt"] for r in tel])
    geo = Geo(epsg, lon=tel[0]["lon"])
    xy = np.array([geo.to_xy(r["lat"], r["lon"]) for r in tel])
    K = Intrinsics.from_f35(frame_wh[0], frame_wh[1], f35)
    rows = list(csv.DictReader(open(pings, encoding="utf-8")))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    capv = cv2.VideoCapture(str(video)) if video else None
    fps_v = capv.get(cv2.CAP_PROP_FPS) if capv else None
    res = []
    for r in rows:
        te = float(r["epoch"]) - offset_s
        j = int(np.argmin(np.abs(t - te)))
        gap = abs(t[j] - te)
        rec = tel[j]
        if gap > 2.0:
            print(f"  ⚠️ 핑 {r['ping_id']}: SRT와 시각이 {gap:.1f}초 어긋남 — 다른 영상이거나 --offset 필요")
        # 방향: SRT 값 → 지정값 → 이동 방향 추정
        if "yaw" in rec:
            yw, how = rec["yaw"], "SRT"
        elif yaw is not None:
            yw, how = yaw, "지정"
        else:
            k0, k1 = max(j - 15, 0), min(j + 15, len(tel) - 1)
            d = xy[k1] - xy[k0]
            yw = float(np.degrees(np.arctan2(d[0], d[1]))) % 360 if np.hypot(*d) > 1.0 else 0.0
            how = "이동방향 추정" if np.hypot(*d) > 1.0 else "정지 중 — 방향 모름(0°)"
        pose = Pose(xy[j, 0], xy[j, 1], rec.get("alt", 30.0), yw, rec.get("pitch", pitch))
        u, v = float(r["u"]), float(r["v"])
        P = pixel_to_ground(K, pose, [[u * frame_wh[0], v * frame_wh[1]]])[0]
        lat, lon = geo.to_latlon(P[0], P[1])
        item = {"ping_id": r["ping_id"], "cls": r["cls"], "score": r["score"], "lat": lat, "lon": lon,
                "drone_lat": rec["lat"], "drone_lon": rec["lon"], "alt": rec.get("alt"),
                "dist_from_drone_m": float(np.hypot(P[0] - xy[j, 0], P[1] - xy[j, 1])),
                "yaw": yw, "yaw_source": how, "time_gap_s": gap, "srt_frame": rec.get("frame")}
        if capv is not None and rec.get("frame"):  # 원본 4K 프레임에서 확대 사진
            capv.set(cv2.CAP_PROP_POS_FRAMES, int(rec["frame"]) - 1)
            ok, fr = capv.read()
            if ok:
                cx, cy = int(u * fr.shape[1]), int(v * fr.shape[0])
                s = 300
                crop = fr[max(cy - s, 0):cy + s, max(cx - s, 0):cx + s]
                p = out / f"ping_{int(r['ping_id']):04d}_4k.jpg"
                cv2.imwrite(str(p), crop)
                item["crop_4k"] = p.name
        res.append(item)
    with open(out / "pings_geo.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for it in res for k in it}, key=lambda k: (k != "ping_id", k)))
        w.writeheader()
        w.writerows(res)
    import json

    (out / "pings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [it["lon"], it["lat"]]},
         "properties": {k: v for k, v in it.items() if k not in ("lat", "lon")}} for it in res]},
        ensure_ascii=False, default=float), encoding="utf-8")
    print(f"핑 {len(res)}개 위치 계산 → {out / 'pings_geo.csv'} (yaw 출처: {sorted({it['yaw_source'] for it in res})})")
    return res


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="litter.live", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--weights", required=True)
        p.add_argument("--conf", type=float, default=0.3)
        p.add_argument("--tiles", type=int, default=2, help="영상 영역을 n×n으로 나눠 탐지 (작은 쓰레기용)")
        p.add_argument("--imgsz", type=int, default=1024)
        p.add_argument("--every", type=int, default=1, help="n프레임마다 탐지 (느리면 2~3)")
        p.add_argument("--out", default=f"runs/live/{datetime.now():%Y%m%d_%H%M%S}")
        p.add_argument("--no_show", action="store_true")

    p = sub.add_parser("screen", help="scrcpy 창 캡처 + adb 터치 + 핑")
    common(p)
    p.add_argument("--window", default="DRONE", help="scrcpy --window-title 값")
    p.add_argument("--safe", default="0.12,0.12,0.88,0.82",
                   help="터치 허용 영역 (영상 영역 대비 x0,y0,x1,y1 비율) — 가장자리 버튼 줄 제외")
    p.add_argument("--no_tap", action="store_true", help="터치 없이 핑만 기록")
    p.add_argument("--action", choices=["drag", "tap"], default="drag",
                   help="선택 동작: drag = 물체 둘레 네모 드래그 (DJI Fly FocusTrack 선택), tap = 탭 (초점)")
    p.add_argument("--serial", help="adb 기기 시리얼 (폰이 여러 대일 때)")

    p = sub.add_parser("stream", help="동영상 파일/RTMP/웹캠으로 시험 (터치 없음)")
    common(p)
    p.add_argument("--source", required=True)

    p = sub.add_parser("match", help="비행 후: 핑 ↔ SRT 시각 맞춤 → 위경도")
    p.add_argument("--pings", required=True)
    p.add_argument("--srt", required=True)
    p.add_argument("--video", help="같은 이름의 MP4 (원본 4K 확대 사진 저장)")
    p.add_argument("--out", required=True)
    p.add_argument("--offset", type=float, default=0.0, help="노트북 시계 − 드론 시계 (초)")
    p.add_argument("--f35", type=float, default=24.0, help="35mm 환산 초점거리 (Mini 5 Pro 24mm)")
    p.add_argument("--pitch", type=float, default=-90.0, help="SRT에 짐벌각이 없을 때 쓸 피치")
    p.add_argument("--yaw", type=float, help="SRT에 방향이 없을 때 쓸 방향 (미지정 시 이동방향 추정)")
    p.add_argument("--frame", default="3840x2160", help="녹화 해상도")

    a = ap.parse_args(argv)
    if a.cmd == "match":
        fw, fh = map(int, a.frame.lower().split("x"))
        match(a.pings, a.srt, a.out, a.video, a.offset, a.f35, a.pitch, a.yaw, (fw, fh))
        return
    det = Detector(a.weights, a.conf, a.tiles, a.imgsz)
    if a.cmd == "screen":
        cap = WindowCapture(a.window)
        tapper = None if a.no_tap else Tapper(serial=a.serial)
        run_live(cap, det, a.out, tuple(map(float, a.safe.split(","))), tapper, a.every, not a.no_show, a.action)
    else:
        run_live(StreamCapture(a.source), det, a.out, tapper=None, every=a.every, show=not a.no_show)


if __name__ == "__main__":
    main()
