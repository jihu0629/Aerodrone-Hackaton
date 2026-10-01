# input/ — 여기에 기업 자료를 넣으면 됩니다

```
input/
  labels.json          (필수) 기업 GeoJSON 라벨. 물체마다 Polygon(위경도) + properties:
                         material_code(STY/ROP/FIS/PLA…), area_sqm, weight_kg, center_lon/lat, detection_seq, image_path
  crops/               (선택) 물체 사진. labels.json 의 image_path 파일명과 같게 (예: MGD_0005_STY.jpg)
  ortho/               (선택) 정사영상 — 있으면 드론 영상 오버레이 + 지형(물·숲·맨땅) 최단경로가 켜짐
    overview.jpg       정사영상 축소본 (약 1 m/px, 3000 px 정도)
    overview.jgw       월드파일 (6줄: 픽셀크기 x, 0, 0, -픽셀크기 y, 왼쪽위 픽셀 중심 x, y)  ← overview.jpg 와 같은 이름
    overview.prj       좌표계 (없으면 config.json 의 crs)
    또는 *.tif         GeoTIFF (좌표 정보 포함) 를 넣어도 됨
  config.json          (선택) 현장 이름, 좌표계, 출발·집결지
```

config.json 예:
```json
{"site": "문갑도", "crs": "EPSG:5186",
 "labels": "labels.json", "photos": "crops", "ortho": "ortho/overview.jpg",
 "depot": {"lon": 126.11139948, "lat": 37.17081007, "name": "문갑도 선착장"}}
```

## 실행

```
python run.py                      # 저장소 루트. 결과: outputs/plan/수거계획.html (자동으로 브라우저가 열림)
python scripts/17_collection_plan.py --workers 4 --hours 6 --travel boat     # 옵션은 --help
```

## 기업이 ECW 정사영상(.ecw) 만 줬을 때

`ortho/` 에 `.ecw + .eww + .prj` 를 넣고 `python tools/extract_company_data.py` 를 실행하면 overview.jpg + .jgw 가 자동으로 만들어진다.
ECW 는 QGIS(무료) 를 설치해야 읽을 수 있다 (스크립트가 자동으로 찾음). GeoTIFF 를 받았다면 그냥 `ortho/` 에 넣으면 된다.

정사영상이 없어도 라벨만으로 계획은 만들어진다 (위성 지도 + 직선 거리 × 우회 배수).

## 실시간 공유 (GitHub Pages 버전, Supabase)

claude.ai 링크 버전은 자체 공유 저장소를 쓰지만, GitHub Pages 는 서버가 없어 기본은 각자 브라우저 저장이다.
Supabase(무료) 프로젝트를 하나 만들면 완료 체크·실측 보정·출발지가 접속한 모두에게 실시간으로 공유된다.

1. https://supabase.com 에서 프로젝트 생성 (무료 플랜, 리전 아무거나)
2. 대시보드 → SQL Editor → `tools/supabase_setup.sql` 내용을 붙여 넣고 Run (테이블 `shared_state`, 공개 읽기/쓰기 정책, 실시간 켜기)
3. 대시보드 → Settings → API 에서 **Project URL** 과 **anon public key** 복사
4. `input/config.json` 에 추가 (`shared_example` 키를 참고해 `shared` 로):
   ```json
   "shared": {"provider": "supabase", "url": "https://xxxx.supabase.co", "anon_key": "eyJ...", "table": "shared_state"}
   ```
5. `python run.py` 로 다시 만들고 `수거계획.html` 을 Pages 에 올리면, 설정 패널 아래 "공유 동기화 켜짐 (Supabase)" 가 뜬다.
   "내 이름" 칸에 이름을 적어 두면 누가 바꿨는지 표시된다.

- anon 키는 공개용 키이고 테이블 권한은 RLS 정책으로 제한된다. 위 SQL 은 링크를 아는 사람 모두가 읽고 쓸 수 있게 한 팀 내부용 설정이다. 더 엄격히 하려면 Supabase Auth 를 붙이고 정책을 바꾼다.
- 실시간(postgres_changes) 이 꺼져 있어도 15 초마다 다시 읽어 맞춘다.
- claude.ai 링크 버전(수거계획_공개용.html) 에는 외부 접속이 막혀 Supabase 를 넣지 않는다 (그쪽은 자체 공유 저장소 사용).
