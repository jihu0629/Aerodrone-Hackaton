"""3단계 준비: 3D 복원용 프레임 선별과 COLMAP 명령 생성.

선별 기준
  * 흐림: 그레이스케일 Laplacian 분산. 값이 작으면 흐림. 기준(blur_min)은 영상·해상도에 따라 다르므로
    selection.csv 의 blur 열 분포를 보고 조정한다.
  * 중복: 이전에 '선택된' 프레임과의 차이(축소 그레이스케일 평균 절대차 / 255). min_change 보다 작으면 거의 같은 장면.
    호버링 중 찍힌 수백 장을 걷어 낸다. 너무 크게 하면 겹침이 부족해져 SfM 이 끊긴다.

축척(스케일)
  COLMAP 결과는 **단위가 없는** 상대 좌표다. 미터로 바꾸려면 별도 근거가 필요하다:
    - 장면 안의 알려진 길이 (줄자로 잰 벽·보도블록 등) → 가장 확실
    - 상대 고도(H) 변화량: 이륙점 기준 고도이므로 '카메라 높이 차' 로만 쓸 수 있다. 지면까지 거리가 아니다.
      또 OCR 값은 0.1 m 해상도·표시 지연이 있어 수 m 이동 구간에서만 쓸 만하다.
    - 홈포인트 거리(D)·속도: 방향 정보가 없어 단독으로는 위치를 정하지 못한다.
  H·D·속도만으로는 카메라의 위치·자세(6자유도)를 알 수 없다. 축척·검증 보조로만 쓴다.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


def blur_score(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _thumb(img: np.ndarray, w: int = 64) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = max(1, int(g.shape[0] * w / g.shape[1]))
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def select_frames(frames_dir: str | Path, out_dir: str | Path, blur_min: float = 60.0, min_change: float = 0.03,
                  max_frames: int | None = 400, copy: bool = True) -> list[dict]:
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in IMG_EXT)
    rows: list[dict] = []
    last_thumb = None
    n_sel = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            rows.append({"file": f.name, "blur": "", "change": "", "selected": 0, "reason": "unreadable"}); continue
        b = blur_score(img)
        th = _thumb(img)
        change = float(np.mean(np.abs(th - last_thumb)) / 255.0) if last_thumb is not None else 1.0
        reason = ""
        if b < blur_min:
            reason = "blurry"
        elif change < min_change:
            reason = "duplicate"
        elif max_frames and n_sel >= max_frames:
            reason = "max_frames"
        sel = int(reason == "")
        if sel:
            n_sel += 1; last_thumb = th
            if copy:
                shutil.copy2(f, out_dir / f.name)
        rows.append({"file": f.name, "blur": f"{b:.1f}", "change": f"{change:.4f}", "selected": sel, "reason": reason})
    with open(out_dir / "selection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "blur", "change", "selected", "reason"])
        w.writeheader(); w.writerows(rows)
    return rows


def colmap_sparse_commands(images_dir: str | Path, work_dir: str | Path, use_gpu: bool = False,
                           sequential: bool = True, camera_model: str = "SIMPLE_RADIAL",
                           colmap: str = "colmap") -> list[list[str]]:
    """카메라 위치 + sparse point cloud 까지의 COLMAP CLI 명령.

    - single_camera 1: 같은 카메라로 찍은 영상 프레임이므로 내부 파라미터를 공유 → 안정적.
    - sequential_matcher: 영상 프레임은 시간 순으로 이웃이 겹치므로 전부 비교(exhaustive)보다 훨씬 빠르다.
      다른 비행에서 찍은 사진을 섞으면 exhaustive 로.
    - use_gpu 0: GPU/VRAM 을 확인하기 전까지 CPU. SIFT 추출·매칭이 느리지만 동작은 한다.
    """
    images_dir, work_dir = Path(images_dir), Path(work_dir)
    db = work_dir / "database.db"
    gpu = "1" if use_gpu else "0"
    cmds = [
        [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images_dir),
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera_model,
         "--SiftExtraction.use_gpu", gpu],
    ]
    if sequential:
        cmds.append([colmap, "sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", "10",
                     "--SequentialMatching.loop_detection", "0", "--SiftMatching.use_gpu", gpu])
    else:
        cmds.append([colmap, "exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", gpu])
    cmds.append([colmap, "mapper", "--database_path", str(db), "--image_path", str(images_dir),
                 "--output_path", str(work_dir / "sparse")])
    cmds.append([colmap, "model_converter", "--input_path", str(work_dir / "sparse" / "0"),
                 "--output_path", str(work_dir / "sparse" / "0"), "--output_type", "TXT"])
    return cmds


def colmap_align_commands(work_dir: str | Path, ref_txt: str | Path, max_error_m: float = 3.0,
                          colmap: str = "colmap") -> list[list[str]]:
    """sparse 모델을 GPS 기준(image_name lat lon alt)으로 ENU 미터 좌표에 맞춘다 → 축척이 미터가 된다.

    - ref_is_gps 1 + alignment_type enu: COLMAP 이 위경도를 첫 이미지 기준 동·북·상(m) 으로 바꿔 유사변환(회전·이동·축척)을 푼다.
    - robust_alignment_max_error: GPS 오차(보통 수 m)보다 조금 큰 값. 너무 작으면 정렬 실패, 너무 크면 튄 GPS 가 축척을 망친다.
    - 결과 좌표의 **상대 거리**는 미터지만, 절대 위치는 GPS 정밀도(수 m) 안에서만 맞다.
    - 기준 이미지가 3장 이상, 그리고 한 직선 위에 있지 않아야 한다(격자 비행 권장). 고도 변화가 없으면 Z 축척은 X·Y 에서 따라온다.
    - 옵션 이름은 COLMAP 3.8~3.11 기준. 다른 버전이면 `colmap model_aligner -h` 로 확인.
    """
    work_dir = Path(work_dir)
    aligned = work_dir / "sparse_aligned"
    return [
        [colmap, "model_aligner", "--input_path", str(work_dir / "sparse" / "0"), "--output_path", str(aligned),
         "--ref_images_path", str(ref_txt), "--ref_is_gps", "1", "--alignment_type", "enu",
         "--robust_alignment", "1", "--robust_alignment_max_error", f"{max_error_m}"],
        [colmap, "model_converter", "--input_path", str(aligned), "--output_path", str(aligned), "--output_type", "TXT"],
    ]
