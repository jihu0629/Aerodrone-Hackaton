"""
3단계: 정합 정확도 합성 벤치마크 (2시기 영상 없이 정합 성능을 숫자로).

원리
----
섬 크롭(A) 에 알려진 이동·회전·축척·밝기 변화·잡음을 걸어 "가짜 2시기" B 를 만들고,
B 를 A 에 다시 정합한 뒤 정답과의 잔차를 잽니다. 정답을 알기 때문에
"우리 정합은 RMSE 0.3 px (0.15 m)" 처럼 말할 수 있습니다.

사용
----
    python scripts/03_bench_register.py --tif outputs/island_crop.tif --trials 10
    python scripts/03_bench_register.py --model affine --method orb --max-shift 80

출력 (outputs/)
    bench_register.csv         시도별 결과
    bench_register_example.png 마지막 시도: A | B(흔들림) | B 정합 후, 차분 영상
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coastcd  # noqa: E402
from coastcd.config import DEFAULT_OUT_DIR  # noqa: E402
from coastcd.raster_io import open_raster, read_window  # noqa: E402
from coastcd.register import make_synthetic_pair, register, synthetic_benchmark  # noqa: E402
from coastcd.water_mask import apply_land_mask, estimate_thresholds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", default=str(DEFAULT_OUT_DIR / "island_crop.tif"))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--downscale", type=int, default=2, help="벤치마크용 축소 배율 (2 = 1 m/px). 1 이면 원본")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--model", default="similarity", choices=["similarity", "affine", "homography"])
    ap.add_argument("--method", default="sift", choices=["sift", "orb", "loftr"])
    ap.add_argument("--max-shift", type=float, default=30.0, help="합성 이동 최대 (px)")
    ap.add_argument("--max-rot", type=float, default=2.0, help="합성 회전 최대 (도)")
    ap.add_argument("--max-scale", type=float, default=0.02, help="합성 축척 편차 최대")
    ap.add_argument("--no-land-mask", action="store_true", help="특징점 탐색에 육지 마스크를 쓰지 않음")
    ap.add_argument("--hard", action="store_true", help="해상도 차이·블러·가림·밝기 그라데이션까지 넣은 어려운 조건")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tif = Path(args.tif)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not tif.exists():
        sys.exit(f"파일 없음: {tif}")

    from rasterio.windows import Window

    with open_raster(tif) as src:
        full = Window(0, 0, src.width, src.height)
    chunk = read_window(tif, full, downscale=args.downscale)
    px_m = chunk.pixel_size_m
    gray = chunk.to_gray()
    print(f"[1] 입력 {gray.shape[1]} x {gray.shape[0]} px, {px_m} m/px")

    land = None
    if not args.no_land_mask:
        thr = estimate_thresholds(chunk.rgb, chunk.valid)
        land = apply_land_mask(chunk.rgb, chunk.valid, thr)
        import cv2

        land = cv2.dilate(land, np.ones((15, 15), np.uint8))  # 해안선 근처 특징점도 허용
        print(f"    특징점 탐색 영역(육지+완충) 비율 {100 * (land > 0).mean():.1f}%")

    print(f"[2] 벤치마크 {args.trials}회 (model={args.model}, method={args.method}, {'hard' if args.hard else 'easy'})")
    results = synthetic_benchmark(
        gray, chunk.valid, land, n_trials=args.trials, seed=args.seed, model=args.model, method=args.method,
        max_shift_px=args.max_shift, max_rot_deg=args.max_rot, max_scale_dev=args.max_scale, hard=args.hard,
    )
    rows = [r.row(px_m) for r in results]
    keys = list(rows[0].keys())
    print("    " + " | ".join(f"{k:>14s}" for k in keys))
    for r in rows:
        print("    " + " | ".join(f"{str(v):>14s}" for v in r.values()))
    after = np.array([r.after_rmse_px for r in results])
    before = np.array([r.before_rmse_px for r in results])
    print(f"\n    정합 전 RMSE 평균 {before.mean():.2f} px ({before.mean() * px_m:.2f} m)")
    print(f"    정합 후 RMSE 평균 {after.mean():.3f} px ({after.mean() * px_m:.3f} m), 최악 {after.max():.3f} px")
    print(f"    참고: SkySat 실제 GSD 0.78 m 이므로 0.5 m/px 기준 1 px 이하는 '센서 해상도 한계 이내' 로 해석")
    print(f"    주의: easy 모드는 같은 영상을 변형한 것이라 이상적입니다. 발표에는 --hard 결과와 inlier RMSE 를 함께 쓰세요.")

    with open(out / "bench_register_hard.csv" if args.hard else out / "bench_register.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("[3] 예시 그림")
    import cv2

    rng = np.random.default_rng(args.seed + 999)
    gb, vb, M_true, p = make_synthetic_pair(gray, chunk.valid, rng, max_shift_px=args.max_shift,
                                            max_rot_deg=args.max_rot, max_scale_dev=args.max_scale, hard=args.hard)
    mask_b = None if land is None else cv2.warpAffine(land, M_true, gray.shape[::-1], flags=cv2.INTER_NEAREST) > 0
    reg = register(gray, gb, land, mask_b, model=args.model, method=args.method)
    gb_reg = reg.warp(gb, gray.shape)
    diff_before = cv2.absdiff(gray, gb)
    diff_after = cv2.absdiff(gray, gb_reg)

    def small(im):
        s = max(im.shape) / 800
        return cv2.resize(im, (int(im.shape[1] / s), int(im.shape[0] / s)), interpolation=cv2.INTER_AREA) if s > 1 else im

    top = np.hstack([small(gray), small(gb), small(gb_reg)])
    bottom = np.hstack([small(diff_before), small(diff_after), np.zeros_like(small(gray))])
    panel = np.vstack([top, bottom])
    cv2.putText(panel, "A (ref) | B (shifted) | B registered  //  |A-B| before | after", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "bench_register_example.png"), panel)
    print(f"    inliers {reg.n_inliers}/{reg.n_matches}, inlier RMSE {reg.inlier_rmse_px:.3f} px")
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
