"""
NOAA MDMAP(같은 해안 구간을 반복 조사한 시민과학 데이터)로 "쓰레기가 많던 곳은
다음에도 많은가"를 검증한다.

조사 방식: 고정 구간(보통 100 m)에서 쓰레기를 세고 전부 치운 뒤, 다음 조사에서
다시 센다. 즉 매 조사 값은 "치운 뒤 다시 쌓인 양"이라 반복 집적을 보기에 맞다.

검증:
  1) 지점별 기간을 앞/뒤 절반으로 나눠, 앞 절반 밀도 순위와 뒤 절반 밀도 순위의
     상관(스피어만).
  2) 앞 절반 상위 20% 지점 중 뒤 절반에도 상위 20%에 남은 비율 (우연이면 20%).
"""
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

API = "https://portal.diver.orr.noaa.gov/pentaho/mdmap"
D = Path(__file__).parent / "data" / "mdmap"
MIN_SURVEYS = 8


def fetch(url, path):
    if path.exists():
        return json.loads(path.read_text())
    for _ in range(3):
        try:
            data = urllib.request.urlopen(url, timeout=90).read()
            path.write_bytes(data)
            return json.loads(data)
        except Exception:
            time.sleep(2)
    return None


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = k
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def main():
    sites = {s["id"]: s for s in json.loads((D / "sites.json").read_text())}
    surveys = json.loads((D / "surveys.json").read_text())
    by_site = defaultdict(list)
    for s in surveys:
        by_site[s["site"]].append(s)

    cand = [k for k, v in by_site.items()
            if len(v) >= MIN_SURVEYS and len({x["date"][:4] for x in v}) >= 2 and k in sites]
    (D / "debris").mkdir(exist_ok=True)

    rows = []
    for k in cand:
        deb = fetch(f"{API}/search/debris?site={k}", D / "debris" / f"{k}.json")
        if not deb:
            continue
        cnt = defaultdict(int)
        for d in deb:
            cnt[d["survey"]] += d["count"] or 0
        length = sites[k].get("length") or 100
        series = sorted((s["date"], cnt.get(s["id"], 0) / length * 100) for s in by_site[k])
        series = [x for x in series if x[1] is not None]
        if len(series) < MIN_SURVEYS or sum(v for _, v in series) == 0:
            continue
        half = len(series) // 2
        rows.append({
            "site": k, "name": sites[k]["name"], "state": sites[k]["location"].get("state"),
            "country": sites[k]["location"].get("country"), "aspect": sites[k].get("aspect"),
            "n": len(series), "start": series[0][0], "end": series[-1][0],
            "first": mean(v for _, v in series[:half]), "second": mean(v for _, v in series[half:]),
        })

    print(f"분석 지점 {len(rows)}곳 (조사 {MIN_SURVEYS}회 이상, 2개 연도 이상, 쓰레기 기록 있음)")
    print(f"지점당 조사 횟수 중앙값 {sorted(r['n'] for r in rows)[len(rows)//2]}회")

    a = [r["first"] for r in rows]
    b = [r["second"] for r in rows]
    print(f"\n1) 앞 절반 vs 뒤 절반 밀도 순위상관(스피어만) = {spearman(a, b):.2f}")

    k = max(1, round(len(rows) * 0.2))
    top_a = set(sorted(range(len(rows)), key=lambda i: -a[i])[:k])
    top_b = set(sorted(range(len(rows)), key=lambda i: -b[i])[:k])
    stay = len(top_a & top_b) / k
    print(f"2) 앞 절반 상위 20%({k}곳) 중 뒤 절반에도 상위 20%: {stay*100:.0f}% (우연이면 20%)")

    tot_b = sum(b)
    share = sum(b[i] for i in top_a) / tot_b
    print(f"3) 앞 절반 상위 20% 지점만 다시 갔다면 뒤 절반 쓰레기의 {share*100:.0f}%를 커버")

    hi = [r for r in rows if r["state"] == "Hawaii"]
    print(f"\n하와이 지점 {len(hi)}곳:")
    for r in sorted(hi, key=lambda r: -r["first"]):
        print(f"  {r['name'][:30]:30s} 조사 {r['n']:3d}회  앞 {r['first']:7.1f}  뒤 {r['second']:7.1f}  (개/100m)")

    (Path(__file__).parent / "mdmap_sites_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
