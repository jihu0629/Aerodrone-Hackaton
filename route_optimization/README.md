# route_optimization — 위성으로 "어디를 날지" 고르고, 핫스팟 경로가 전체 지그재그보다 이득인지 재기

> 드론대장 붕붕이 · 2026-10-01~02. 파이프라인 **① 단계**. 개발 이력은 `stage1-ocean-current-route` 브랜치에 있다 (수거계획 대시보드는 `../shoresweep_planner/`).

**인천·강화 적용 (조석 모델 집적 예상 지도 + 위성 정합)**: `incheon/README.md` — 집적 예상 해안 140 km(전체 1,108 km 의 13 %)만 날면 전체 지그재그 448 h·1,333소티 → 57 h·178소티. 위성 형상 점수는 하구 수로형 핫스팟과 겹치지 않아(무작위 수준) 인천에서는 조석 모델이 1차 선별, 위성은 해안선 정합·갱신 역할.

**해류 데이터 수집 → OpenDrift 좌초 시뮬**: `ocean_current_data/README.md` — Open-Meteo(Copernicus SMOC 1/12°, 조석 포함) 해류로 90일 표류·좌초를 돌려 위성 해안 구간 밀도로 집계. 집적 해안 88 km(8 %), 상위 15 % 비행 68 h(전체의 15 %). 조석 지도와 겹침 17~21 % → 고해상 조류·실측 필요.

**문갑도 — OpenDrift 좌초 vs 기업 라벨**: `mungap_opendrift/README.md` — 8 km 해류로는 섬 둘레 어느 200 m 인지 못 맞히지만(상위 30 % 에 라벨 21 %) 남향 면 집중은 일치. 해류 자료에서 측정한 풍향·잔차류를 형상 점수에 넣으면 52~55 %, 만입도만 62 %.

**결론 한 문장**: 쌓이는 자리는 지형과 조석이 정하므로 잘 바뀌지 않는다. 위성 해안 형상과 해류·바람 방향으로 그 자리를 골라(상위 30 % 구간에 쓰레기의 절반 이상: 니하우 65~67 %, 문갑도 62~67 %) 드론을 반복 투입하고 수거 계획을 세우며, 전체는 가끔 전수·무작위로 확인한다. 근거의 범위와 피할 표현("주기적으로 생긴다"·"확률이 높다"·"그쪽만 날린다")은 `docs/ROUTE_OPTIMIZATION.md` 8.4절.

**먼저 볼 것**
- `docs/ROUTE_OPTIMIZATION.md` — 무엇을 했고, 어떤 순서로 생각했고, 숫자가 어떻게 나왔고, 어디까지 믿을 수 있는지 (요약 표는 0절).
- `docs/route_optimization.html` — 같은 내용을 한 장짜리 인터랙티브로 (브라우저로 열기: 포착률 곡선, 예산·고도별 시간 대비 포착, 니하우 점수 지도, 한계, 판단 규칙).
- `docs/위성우선순위_니하우.md`, `docs/전략비교_핫스팟_vs_전체커버리지.md` — 세부.

**한 줄 결론 (니하우, 해안 98.8 km, 하와이 탐지 격자 5,476개·13.7 t)**
위성 10 m 로는 쓰레기가 안 보이므로 해안 형상(만입도·풍향 노출·띠 폭) 점수로 구간을 고른다. 점수 상위 30 % 길이에 탐지 무게 65~67 %.
같은 카메라(20 m, GSD 2.9 cm)·같은 띠 폭(−20~+100 m, 8패스)이면 핫스팟 30 % 는 전체 지그재그의 37 % 시간(15.4 h vs 41.3 h)·35 % 프레임으로 67 % 를 잡는다.
위성 없이 같은 시간을 쓰면 25~29 %. 대가는 설계상 못 보는 33 %, 풍향을 잘못 넣으면 38 % 로 하락, 분절 때문에 같은 길이에 시간 +24 %, 안 본 70 % 해안은 관측이 없음.

## 폴더

```
litter3d/priority.py            Sentinel-2 NDWI → 해안선 10 m 표본점 → 만입도·노출·띠 폭 점수 → 200 m 구간 → 길이 예산 → 코리더 경로·소티 → GeoJSON·Litchi CSV·DJI WPML KMZ
litter3d/strategy.py            핫스팟 코리더 vs 전체 지그재그 커버리지 비교 모델 (같은 카메라·띠 폭·배터리·소티 계획기)
scripts/18_priority_flight.py   점수·예산 대비 포착률 곡선·지도·경로 파일 (그림 20·21·25·26)
scripts/19_strategy_compare.py  예산별 전략 표·시간 대비 포착·한계·지도 (그림 22·23·24, 비교표)
tools/fetch_s2_aws.py           Sentinel-2 L2A 를 AWS 공개 버킷에서 STAC 없이 창만 받기 (B02/B03/B04/B08/TCI/SCL)
tests/                          합성 섬 테스트 (원형·울퉁불퉁·만 있는 섬) — python -m pytest tests -q
docs/figures/20~26              결과 그림, docs/examples/                 표·summary.json·경로 예시
input/s2_niihau/hawaii_niihau_ft_summary.json   검증용 탐지 kg 격자 (coastal 브랜치 litter/hawaii.py 결과)
```

## 실행 (이 폴더에서)

```bash
pip install -r requirements.txt
python -m pytest tests -q
# 니하우 Sentinel-2 (약 200 MB, git 제외)
python tools/fetch_s2_aws.py --tile 4QCK --bounds 368000 2407000 395000 2437000 \
    --scenes S2B_4QCK_20250502_0_L2A,S2C_4QCK_20251103_0_L2A,S2B_4QCK_20250303_0_L2A --out input/s2_niihau --prefix niihau
# 점수·곡선·경로
python scripts/18_priority_flight.py --s2-dir input/s2_niihau --scenes 20250502,20251103,20250303 --tci-scene 20250502 \
    --density-grid input/s2_niihau/hawaii_niihau_ft_summary.json --depot="-160.2024,21.7869" --wind-from 60 --budget-frac 0.3 --out outputs/niihau
# 전략 비교
python scripts/19_strategy_compare.py --s2-dir input/s2_niihau --scenes 20250502,20251103,20250303 --tci-scene 20250502 \
    --density-grid input/s2_niihau/hawaii_niihau_ft_summary.json --depot="-160.2024,21.7869" --wind-from 60 --alt 20 --sea-m 20 --inland-m 100 --out outputs/strategy_niihau
```
Windows 콘솔은 `PYTHONUTF8=1`. 문갑도·다른 섬은 `--tile/--bounds/--prefix/--depot/--wind-from` 과 `--density-csv lon,lat,w` 만 바꾼다 (문갑도 창과 장면은 `docs/ROUTE_OPTIMIZATION.md` 6절).

## 기존 파이프라인과의 관계

- 비교 모델의 "기존 방식" 은 `stage3-detection-3d-weight` 브랜치 의 `litter/sim_ortho.py` 커버리지(1024×768·HFOV 73.7°·고도 20 m·측면 겹침 0.3·5 m/s·1 프레임/초)와 같은 카메라·같은 식으로 둔다.
- 검증 밀도는 같은 브랜치 `litter/hawaii.py` 가 니하우 칩 553장에서 낸 탐지 격자. 라벨이 아니라 탐지(재현율 0.58)이고 칩이 있는 해안만이라 "순위와 비율" 까지만 해석한다.
- 다음: 하와이 원본 라벨(Zenodo 8381113 `chips.csv`)로 재검증, 2단계 운용(고고도 1패스 → 저고도 재방문) 전략 추가, 수거계획 대시보드에 구간 점수 레이어.
