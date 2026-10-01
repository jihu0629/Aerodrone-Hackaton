"""
하와이 2015 항공조사 라벨(Zenodo 8381113)로 "지형(해안 방향)만으로 쓰레기
밀집 구간을 미리 고를 수 있는가"를 검증한다.

방법: 칩 중심의 위경도를 구하고, 섬 중심에서 그 칩까지의 방위각을 그 해안이
바라보는 방향의 근사값으로 쓴다(하와이 섬들은 대체로 둥글어서 성립). 무역풍은
북동(약 60°)에서 불어오므로, 바람을 정면으로 받는 쪽에 쓰레기가 몰리는지 본다.

주의: 이 데이터의 칩은 모두 쓰레기가 있는 칩이다(학습용 표본). 그래서 "쓰레기가
없는 해안"의 길이는 알 수 없고, 방위 구간마다 해안 길이가 같다고 가정한다.
섬별 칩 비율(니이하우 36%, 오아후 5%)이 원 조사의 전수 집계(38%, 5%)와 거의
같아서, 칩 분포가 실제 분포를 크게 왜곡하지는 않는 것으로 본다.
"""
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from pyproj import Transformer

DATA = Path(__file__).parent / "data" / "imagery_and_labels"
CHIPS = DATA / "processed_image_chips"
CHIP_PX = 512
TRADE_WIND_FROM = 60.0  # 북동 무역풍이 불어오는 방향(°)

# 섬 중심 근사 (위도, 경도)
CENTERS = {
    "niihau": (21.90, -160.16), "kauai": (22.07, -159.50), "oahu": (21.47, -157.98),
    "molokai": (21.13, -157.02), "lanai": (20.83, -156.92), "maui": (20.80, -156.33),
    "kahoolawe": (20.55, -156.61), "hawaii": (19.60, -155.52),
}
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def chip_lonlat(name):
    xml = (CHIPS / f"{name}.aux.xml").read_text()
    epsg = re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]</SRS>', xml).group(1)
    gt = [float(v) for v in re.search(r"<GeoTransform>(.*?)</GeoTransform>", xml).group(1).split(",")]
    x = gt[0] + gt[1] * CHIP_PX / 2
    y = gt[3] + gt[5] * CHIP_PX / 2
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(x, y)


def bearing(lat0, lon0, lat1, lon1):
    dy = lat1 - lat0
    dx = (lon1 - lon0) * math.cos(math.radians(lat0))
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    labels = Counter()
    for f in ("training_data.csv", "evaluation_data.csv"):
        for r in csv.DictReader(open(DATA / f)):
            labels[r["filename"]] += 1

    rows = []
    for name, n in labels.items():
        island = name.split("_")[0]
        lon, lat = chip_lonlat(name)
        b = bearing(*CENTERS[island], lat, lon)
        rows.append((island, lat, lon, b, n))

    total = sum(r[4] for r in rows)
    print(f"칩 {len(rows)}개, 라벨 {total}개\n")

    print("== 섬별 비중 (원 조사 전수집계: 니이하우 38%, 오아후 5%)")
    by_island = Counter()
    for r in rows:
        by_island[r[0]] += r[4]
    for k, v in by_island.most_common():
        print(f"  {k:10s} {v:5d}  {v/total*100:5.1f}%")

    print("\n== 해안 방향(8방위)별 라벨 비중 (방위마다 해안 길이가 같다면 기대값 12.5%)")
    sec = Counter()
    for r in rows:
        sec[SECTORS[int(((r[3] + 22.5) % 360) // 45)]] += r[4]
    for s in SECTORS:
        bar = "#" * int(sec[s] / total * 100)
        print(f"  {s:3s} {sec[s]/total*100:5.1f}%  {bar}")

    print(f"\n== 무역풍({TRADE_WIND_FROM:.0f}°)을 정면으로 받는 쪽(±45°, 해안의 25%)에 몰린 비중")
    for island in sorted(by_island, key=lambda k: -by_island[k]):
        sub = [r for r in rows if r[0] == island]
        n_all = sum(r[4] for r in sub)
        n_wind = sum(r[4] for r in sub if angdiff(r[3], TRADE_WIND_FROM) <= 45)
        print(f"  {island:10s} {n_wind/n_all*100:5.1f}%  (기대 25%)")
    n_wind = sum(r[4] for r in rows if angdiff(r[3], TRADE_WIND_FROM) <= 45)
    print(f"  {'전체':10s} {n_wind/total*100:5.1f}%  (기대 25%) → {n_wind/total/0.25:.1f}배")

    print("\n== 1 km 격자 집중도 (쓰레기가 있는 칸 기준)")
    cells = defaultdict(int)
    for r in rows:
        cells[(r[0], round(r[1] / 0.009), round(r[2] / 0.0097))] += r[4]
    vals = sorted(cells.values(), reverse=True)
    for p in (0.1, 0.2, 0.3):
        k = max(1, int(len(vals) * p))
        print(f"  상위 {int(p*100)}% 칸({k}/{len(vals)})이 라벨의 {sum(vals[:k])/total*100:.1f}%")

    out = Path(__file__).parent / "hawaii_chips_bearing.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["island", "lat", "lon", "bearing_deg", "n_labels"])
        w.writerows(rows)
    print(f"\n칩별 결과: {out}")


if __name__ == "__main__":
    main()
