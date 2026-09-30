"""
굴업도 Sentinel-2 L2A 시계열 자동 다운로드
=========================================
- 데이터: Element84 Earth Search (AWS) 의 Sentinel-2 Collection-1 L2A (무료, 로그인 불필요)
- 기업 SkySat AOI 만큼만 잘라서 받음 → 날짜당 수 MB
- 섬 위 구름을 SCL(장면분류) 밴드로 직접 계산해서 맑은 날만 저장
- 출력: 날짜별 5밴드 GeoTIFF + RGB 미리보기 PNG + 한 장짜리 썸네일 모음 + 목록 CSV

설치 (VS Code 터미널):
    pip install pystac-client odc-stac rioxarray matplotlib pandas

실행:
    python s2_download.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import odc.stac
import rioxarray  # noqa: F401  (.rio 기능 활성화용)
from pystac_client import Client

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────── 설정 (여기만 바꾸면 됨) ──────────────────────────────
BBOX = [125.868, 37.139, 126.060, 37.249]   # 기업 SkySat AOI (서, 남, 동, 북)
DATE_RANGE = "2016-01-01/2026-09-30"        # 검색 기간
MONTHS = [6, 7, 8, 9, 10]                   # 이 달만 사용 (SkySat 8월과 계절 맞추기). 전체면 None
SCENE_CLOUD_MAX = 60                        # 1차 필터: 장면 전체 구름 % (느슨하게)
AOI_CLOUD_MAX = 5.0                         # 2차 필터: AOI 안 구름+그림자 % (엄격하게)
AOI_VALID_MIN = 95.0                        # AOI 중 영상이 실제로 찍힌 비율 % (타일 가장자리 대비)
ONE_PER_YEAR = True                         # True: 해마다 가장 맑은 날 1장만 (10년 ≈ 10장)
                                            # False: 조건 맞는 날 전부 (10년이면 100장+, 1 GB 이상)
CRS = "EPSG:32651"                          # UTM 51N (굴업도). SkySat 좌표계와 맞출 것
RES = 10                                    # m
OUT_DIR = Path("S2_GYD")                    # 저장 폴더

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
BANDS = ["blue", "green", "red", "nir", "swir16"]   # B02 B03 B04 B08 B11
BAND_DESC = ["B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir16"]
# SCL: 3=구름그림자, 8=구름(중), 9=구름(고), 10=권운  /  0=nodata
SCL_BAD = [3, 8, 9, 10]
# ──────────────────────────────────────────────────────────────────────────────


def search_items():
    cat = Client.open(STAC_URL)
    search = cat.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": SCENE_CLOUD_MAX}},
    )
    items = list(search.items())
    if MONTHS:
        items = [it for it in items if it.datetime.month in MONTHS]
    print(f"[1] 검색: {len(items)}개 타일-장면 (장면 구름 < {SCENE_CLOUD_MAX}%, 월 {MONTHS})")
    return items


def aoi_cloud_table(items):
    """SCL 밴드(20 m)만 먼저 가볍게 받아서 AOI 안 구름 비율 계산."""
    scl = odc.stac.load(
        items, bands=["scl"], bbox=BBOX, crs=CRS, resolution=20,
        groupby="solar_day", chunks={},
    )["scl"].compute()

    rows = []
    for t in scl.time.values:
        a = scl.sel(time=t).values
        valid = a > 0
        n_valid = valid.sum()
        valid_pct = 100 * n_valid / a.size
        cloud_pct = 100 * np.isin(a, SCL_BAD).sum() / max(n_valid, 1)
        rows.append({"date": str(t)[:10], "valid_pct": round(valid_pct, 1),
                     "aoi_cloud_pct": round(cloud_pct, 2)})
    df = pd.DataFrame(rows)
    print(f"[2] 날짜(solar day) {len(df)}개에 대해 AOI 구름 계산 완료")
    return df


def pick_dates(df):
    ok = df[(df.aoi_cloud_pct <= AOI_CLOUD_MAX) & (df.valid_pct >= AOI_VALID_MIN)].copy()
    if ONE_PER_YEAR:
        ok["year"] = ok.date.str[:4]
        ok = ok.sort_values("aoi_cloud_pct").groupby("year").head(1).sort_values("date")
    print(f"[3] 맑은 날 선택: {len(ok)}개 (AOI 구름 ≤ {AOI_CLOUD_MAX}%, 유효 ≥ {AOI_VALID_MIN}%)")
    return ok


def item_meta(items):
    """날짜 → 촬영시각(UTC), 타일, 반사율 offset 매핑."""
    meta = {}
    for it in items:
        d = it.datetime.strftime("%Y-%m-%d")
        rb = it.assets["red"].extra_fields.get("raster:bands", [{}])[0]
        meta.setdefault(d, {
            "utc": it.datetime.strftime("%H:%M"),
            "kst": (it.datetime + pd.Timedelta(hours=9)).strftime("%H:%M"),
            "tiles": set(),
            "scale": rb.get("scale", 0.0001),
            "offset": rb.get("offset", 0.0),
        })
        meta[d]["tiles"].add(it.properties.get("s2:mgrs_tile") or it.id.split("_")[1])
    return meta


def rgb_preview(arr):
    """(band,y,x) 반사율*10000 → 2~98% 스트레치 RGB."""
    rgb = arr[[2, 1, 0]].astype("float32")  # red, green, blue
    rgb[rgb <= 0] = np.nan
    lo, hi = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(np.moveaxis(rgb, 0, -1))


def download(items, chosen, meta):
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "preview").mkdir(exist_ok=True)
    keep = set(chosen.date)
    sel_items = [it for it in items if it.datetime.strftime("%Y-%m-%d") in keep]

    ds = odc.stac.load(
        sel_items, bands=BANDS, bbox=BBOX, crs=CRS, resolution=RES,
        groupby="solar_day", chunks={}, resampling={"swir16": "bilinear", "*": "nearest"},
    )

    records, previews = [], []
    for t in ds.time.values:
        d = str(t)[:10]
        m = meta[d]
        arr = ds.sel(time=t).to_array("band").compute()      # uint16 DN

        # DN → 반사율×10000 (offset 보정: 2022-01-25 이후 영상의 -1000 처리 포함)
        dn = arr.values.astype("int32")
        refl = dn * m["scale"] + m["offset"]                   # 실제 반사율 (0~1)
        out = np.where(dn == 0, 0, np.clip(refl * 10000, 1, 65535)).astype("uint16")
        arr = arr.copy(data=out)
        arr = arr.assign_coords(band=BAND_DESC)
        arr.rio.write_nodata(0, inplace=True)

        tif = OUT_DIR / f"S2_GYD_{d}.tif"
        arr.rio.to_raster(tif, compress="deflate", tiled=True)

        prev = rgb_preview(out)
        plt.imsave(OUT_DIR / "preview" / f"{d}.png", prev)
        previews.append((d, prev))

        row = chosen[chosen.date == d].iloc[0]
        records.append({
            "date": d, "time_utc": m["utc"], "time_kst": m["kst"],
            "tiles": "+".join(sorted(m["tiles"])),
            "aoi_cloud_pct": row.aoi_cloud_pct, "valid_pct": row.valid_pct,
            "file": tif.name,
        })
        print(f"    저장: {tif.name}  (KST {m['kst']}, AOI 구름 {row.aoi_cloud_pct}%)")

    pd.DataFrame(records).to_csv(OUT_DIR / "S2_GYD_list.csv", index=False, encoding="utf-8-sig")
    contact_sheet(previews)
    print(f"[4] 완료: {len(records)}장 → {OUT_DIR.resolve()}")


def contact_sheet(previews, ncol=6):
    if not previews:
        return
    nrow = int(np.ceil(len(previews) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (d, img) in zip(np.atleast_1d(axes).ravel(), previews):
        ax.imshow(img)
        ax.set_title(d, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contact_sheet.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    items = search_items()
    if not items:
        raise SystemExit("검색 결과 없음 — DATE_RANGE / SCENE_CLOUD_MAX 를 늘려보세요.")
    OUT_DIR.mkdir(exist_ok=True)
    df = aoi_cloud_table(items)
    df.to_csv(OUT_DIR / "S2_GYD_all_dates.csv", index=False, encoding="utf-8-sig")
    chosen = pick_dates(df)
    if chosen.empty:
        raise SystemExit("맑은 날 없음 — AOI_CLOUD_MAX 를 늘려보세요.")
    download(items, chosen, item_meta(items))
