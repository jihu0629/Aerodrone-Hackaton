"""object3d 기하 부분 테스트: SfM 없이, 알려진 카메라·상자로 visual hull·평면·축척을 검증한다.
(pycolmap 의 Camera 만 사용. SfM 전체 실행은 느려서 여기서 하지 않는다 — 실제 영상 검증은 docs 참고)"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pycolmap = pytest.importorskip("pycolmap")

from dronecap import object3d as o3  # noqa: E402

BOX = np.array([0.30, 0.20, 0.12])   # 길이, 너비, 높이 (m). 바닥 z=0 에 놓임, 중심 (0,0)


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """카메라 중심 → 대상 을 보는 3x4 world→cam (z 앞, y 아래)."""
    z = target - center; z /= np.linalg.norm(z)
    up = np.array([0, 0, 1.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ center
    return np.c_[R, t]


def _views(n=24, elev_deg=35.0, dist=1.5, w=640, h=480, f=600.0, elevations=None):
    """elevations: 여러 고도각의 링을 겹칠 때 [(elev_deg, n), ...]"""
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, params=[f, w / 2, h / 2])
    views = []
    corners = np.array([[sx * BOX[0] / 2, sy * BOX[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, BOX[2])])
    rings = elevations or [(elev_deg, n)]
    specs = [(e, 2 * np.pi * i / k) for e, k in rings for i in range(k)]
    for i, (e, a) in enumerate(specs):
        c = np.array([dist * np.cos(a) * np.cos(np.radians(e)), dist * np.sin(a) * np.cos(np.radians(e)),
                      dist * np.sin(np.radians(e))])
        v = o3.View(f"img{i:03d}.jpg", Path(f"img{i:03d}.jpg"), cam, _look_at(c, np.zeros(3)), c)
        uv, inside = v.project(corners)
        assert inside.all()
        mask = np.zeros((h, w), np.uint8)
        hull = cv2.convexHull(uv.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        v.mask = mask
        views.append(v)
    return views


def test_project_and_ray_roundtrip():
    v = _views(1)[0]
    X = np.array([[0.05, -0.02, 0.1]])
    uv, inside = v.project(X)
    d = v.ray(uv)[0]
    # 광선 위에 X 가 있어야 한다
    t = (X[0] - v.center) @ d
    assert np.linalg.norm(v.center + t * d - X[0]) < 1e-6


def test_plane_fit_and_height():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000), rng.normal(0, 0.002, 2000)]
    obj = np.c_[rng.uniform(-0.1, 0.1, 100), rng.uniform(-0.1, 0.1, 100), rng.uniform(0.05, 0.12, 100)]
    P = np.vstack([floor, obj])
    cams = np.array([[0, 0, 1.5], [1, 0, 1.2]])
    plane, inl = o3.fit_plane_ransac(P, cams)
    assert plane.n[2] > 0.99 and abs(plane.d) < 0.01
    assert inl[:2000].mean() > 0.95 and inl[2000:].mean() < 0.05
    assert plane.height(np.array([[0, 0, 0.12]]))[0] == pytest.approx(0.12, abs=0.01)


def test_visual_hull_box_volume():
    views = _views(24, 35.0)
    plane = o3.Plane(np.array([0, 0, 1.0]), 0.0)
    center = np.array([0.02, -0.01, 0.0])
    grid = o3.visual_hull(views, plane, center, radius=0.3, zmax=0.25, res=120, log=lambda *_: None)
    m = o3.hull_metrics(grid, scale=1.0)
    true_v = float(np.prod(BOX))
    # 한 고도각(35°) 링만으로는 스치는 광선이 윗모서리를 지나 옆면이 1 cm 쯤 부푼다 (기하적 한계, 버그 아님)
    assert m["length"] == pytest.approx(BOX[0], rel=0.10) and m["length"] >= BOX[0]
    assert m["width"] == pytest.approx(BOX[1], rel=0.12) and m["width"] >= BOX[1]
    # 위에서만 본 실루엣 교차는 윗면 위에 '지붕' 이 남는다 → height_max 는 과대, height_area90 이 실제 높이
    assert m["height_max"] > BOX[2] * 1.2
    assert m["height_area90"] == pytest.approx(BOX[2], rel=0.08)
    assert m["volume_below_h90"] == pytest.approx(true_v, rel=0.25) and m["volume_below_h90"] >= true_v * 0.95
    assert m["volume"] >= true_v * 0.95                              # 전체 hull 은 상한


def test_visual_hull_two_rings_is_tighter():
    """높은 링 + 낮은 링(15°)을 섞으면 '지붕' 이 줄어 height_max 가 실제 높이에 가까워진다 → 촬영 권장안의 근거.
    남는 옆면 슬랙(약 1 복셀 ≈ 영상 1 px 에 해당)은 해상도 한계라 치수 허용을 8 % 로 둔다."""
    views = _views(dist=2.5, elevations=[(35.0, 24), (15.0, 24)])
    plane = o3.Plane(np.array([0, 0, 1.0]), 0.0)
    grid = o3.visual_hull(views, plane, np.zeros(3), radius=0.3, zmax=0.25, res=120, log=lambda *_: None)
    m = o3.hull_metrics(grid, 1.0)
    true_v = float(np.prod(BOX))
    assert m["length"] == pytest.approx(BOX[0], rel=0.08)
    assert m["width"] == pytest.approx(BOX[1], rel=0.08)
    assert m["height_area90"] == pytest.approx(BOX[2], rel=0.08)
    assert m["height_max"] < BOX[2] * 1.25                     # 단일 35° 링은 1.4배 이상
    assert m["volume_below_h90"] == pytest.approx(true_v, rel=0.10)


def test_visual_hull_tolerates_bad_masks():
    views = _views(24, 35.0)
    for v in views[:2]:                       # 2장은 엉뚱한 마스크 (구석의 작은 조각 → 그대로 쓰면 모든 복셀이 깎임)
        v.mask = np.zeros_like(v.mask); v.mask[0:50, 0:50] = 255
    views[2].mask = np.zeros_like(views[2].mask)   # 1장은 빈 마스크
    plane = o3.Plane(np.array([0, 0, 1.0]), 0.0)
    bad = o3.filter_bad_masks(views, plane, np.zeros(3), log=lambda *_: None)
    assert bad == 3 and all(v.mask is None for v in views[:3])
    grid = o3.visual_hull(views, plane, np.zeros(3), 0.3, 0.25, res=96, log=lambda *_: None)
    m = o3.hull_metrics(grid, 1.0)
    assert m["volume_below_h90"] == pytest.approx(float(np.prod(BOX)), rel=0.12)


def test_object_center_from_masks():
    views = _views(12, 40.0)
    plane = o3.Plane(np.array([0, 0, 1.0]), 0.0)
    c = o3._object_center(views, {v.name: v.mask for v in views}, plane)
    assert np.linalg.norm(c[:2]) < 0.03 and abs(c[2]) < 1e-6


def test_scale_from_reference():
    views = _views(8, 35.0)
    A = np.array([0.1, 0.0, 0.0]); B = np.array([-0.2, 0.05, 0.0])     # 실제 거리
    true = float(np.linalg.norm(A - B))
    ref = []
    for v in views[:3]:
        uv, _ = v.project(np.stack([A, B]))
        ref.append((v.name, tuple(uv[0]), tuple(uv[1])))
    # SfM 좌표가 실제의 1/2 축척이었다고 가정: 카메라 중심·행렬 평행이동을 반으로
    views2 = []
    for v in views[:3]:
        M = v.M.copy(); M[:, 3] *= 0.5
        views2.append(o3.View(v.name, v.image_path, v.cam, M, v.center * 0.5))
    scale, info = o3.scale_from_reference(views2, ref, length_m=true)
    assert scale == pytest.approx(2.0, rel=1e-3)
    assert max(info["reproj_err_px_sum"]) < 0.5


def test_pick_component():
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 255          # 작은 성분
    m[50:90, 50:90] = 255          # 큰 성분
    big = o3.pick_component(m, None); assert big[70, 70] == 255 and big[20, 20] == 0
    small = o3.pick_component(m, np.array([20.0, 20.0])); assert small[20, 20] == 255 and small[70, 70] == 0
    near = o3.pick_component(m, np.array([35.0, 35.0]), min_area=100); assert near[20, 20] == 255
