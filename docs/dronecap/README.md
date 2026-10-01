# dronecap — DJI Fly 라이브 영상 수집 · 화면 OCR · 3D 복원 준비

DJI Mini 5 Pro → (아이폰 DJI Fly, RTMP) → MediaMTX(PC) → RTSP → **이 프로그램**

```
1단계  scripts/20_capture.py      RTSP 수신·표시·녹화(원본 복사)·프레임 저장·시각 기록      ← 지금 완성
2단계  scripts/21_select_roi.py   OCR 영역 선택 (스크린샷/화면녹화/미러링 화면)
       scripts/22_ocr_run.py      OCR → telemetry.csv (원문·점수·상태 보존)
       scripts/23_sync.py         프레임 ↔ OCR 시간 매칭 → matched.csv
3단계  scripts/24_select_frames.py 흐림·중복 프레임 제거
       scripts/25_colmap_cmds.py   COLMAP sparse 복원 명령 생성/실행
시험   scripts/29_test_publish.py  드론 없이 로컬 영상을 RTMP 로 송출해 전체 경로 시험
```

- 패키지: `dronecap/` (litter3d 와 독립). 설정: `config/dronecap.yml`. 결과: `data/sessions/<세션>/` (git 제외).
- 모든 시각은 **PC 수신 시각**이다. 드론 촬영 시각·센서 측정 시각이 아니다.

---

## 1. 설치 (Windows PowerShell)

```powershell
cd <저장소 폴더>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- `requirements.txt` 가 `requirements-dronecap.txt` 를 포함한다. 기존 `opencv-python-headless` 는 창(imshow)을 못 띄우므로 `opencv-python` 으로 바꿨다 (litter3d 도 그대로 동작). **둘을 같이 설치하지 말 것** — 이미 headless 가 있으면 `.\.venv\Scripts\python.exe -m pip uninstall opencv-python-headless` 후 다시 설치.
- `rapidocr_onnxruntime` 은 첫 실행 때 모델을 패키지 안에서 읽는다(인터넷 불필요, CPU).

### FFmpeg (녹화에 필요, 미리보기·프레임 저장에는 불필요)

녹화는 `ffmpeg.exe` 가 RTSP 를 따로 열어 **재인코딩 없이(-c copy) H.264 영상 + AAC 오디오를 그대로** 파일에 쓴다.
ffmpeg 가 없으면 자동으로 `opencv` 백엔드(재인코딩, 화질 손실, **오디오 없음**)로 바뀌고 로그에 경고가 남는다.

```powershell
winget install Gyan.FFmpeg          # 또는 https://www.gyan.dev/ffmpeg/builds/ 에서 release-essentials zip
# 새 PowerShell 창을 열고
ffmpeg -version                      # 버전이 나오면 PATH 등록 완료
where.exe ffmpeg                     # 경로 확인
```
PATH 에 안 넣겠다면 `config/dronecap.yml` 의 `record.ffmpeg_path` 에 전체 경로를 적는다 (예: `C:\ffmpeg\bin\ffmpeg.exe`).

---

## 2. 1단계 실행

### 2-1. MediaMTX 와 DJI Fly

```powershell
cd C:\mediamtx
.\mediamtx.exe
```
DJI Fly 라이브 스트리밍(RTMP) 주소: `rtmp://<서버 PC 의 실제 IPv4>:1935/live/drone`
PC IPv4 는 `ipconfig` 의 Wi-Fi/이더넷 어댑터 "IPv4 주소". 서버 로그에 찍히는 `conn 10.10.20.222` 같은 주소는 **송신 장치(아이폰)** 쪽이다.
서버 로그에 `stream is available and online, 2 tracks (H264, MPEG-4 Audio)` 가 보이면 수신 중이다. `closed: EOF` 면 송출이 끊긴 상태다.

### 2-2. 수신 프로그램

```powershell
cd <저장소 폴더>
.\.venv\Scripts\python.exe scripts\20_capture.py                 # 설정 파일의 rtsp://127.0.0.1:8554/live/drone
.\.venv\Scripts\python.exe scripts\20_capture.py --record        # 시작하자마자 녹화
```
창 키: `q`/ESC 종료, `r` 녹화 시작/종료, `s` 지금 프레임 1장 저장, `f` 자동 프레임 저장 켬/끔, `h` HUD.
HUD 첫 줄 상태: `CONNECTING`(처음 접속 중) → `CONNECTED`(수신 중, fps 표시) → `RECONNECTING`(끊김, 1→2→4→8→10초 간격 재시도).

### 2-3. 드론 없이 시험 (권장: 먼저 이걸로 PC 쪽을 확인)

터미널 1 `.\mediamtx.exe`, 터미널 2:
```powershell
.\.venv\Scripts\python.exe scripts\29_test_publish.py --synthetic 30 --loop     # 합성 영상을 rtmp://127.0.0.1:1935/live/drone 로 송출
# 또는 실제 DJI 영상 파일:  python scripts\29_test_publish.py DJI_0001.MP4 --loop
```
터미널 3: `python scripts\20_capture.py --record`. 터미널 2 를 Ctrl+C 로 끊으면 서버에 `closed: EOF` 가 찍히고 수신 창은 `RECONNECTING` 이 된다. 다시 송출하면 `conn#2` 로 이어지고 녹화는 **새 파일(seg002)** 로 시작된다.

### 2-4. 결과 폴더

```
data/sessions/drone_20261001_110102_ab12/
  session.json      설정 스냅샷, 시작·종료 시각, 통계(받은/저장/버린 프레임, 재접속 수)
  events.log        연결·재접속·오류·버림 기록
  frames.csv        frame_id, connection_id, conn_frame_index, video_receive_time_utc, receive_monotonic_s, stream_pos_ms, saved, note
  recordings.csv    구간별 파일, backend, reencoded(0/1), audio(0/1), start/first_data/stop 시각, stop_reason
  frames/           f000012_c01.jpg  (프레임 번호_연결 번호)
  video/            <세션>_seg001.mp4, ffmpeg_seg001.log
```
- `saved=0` 행 = 디스크 쓰기가 밀려 **버린** 프레임. 조용히 사라지지 않고 기록된다.
- `connection_id` 가 바뀌면 그 사이에 끊김이 있었다는 뜻. `stream_pos_ms` 는 디코더가 주는 스트림 내 위치로, 연결마다 다시 0 근처에서 시작하므로 연결 안에서만 비교한다.
- `recordings.csv` 의 `first_data_time_utc` 가 그 파일 영상이 실제로 시작된 PC 시각이다 (`start_time_utc` 는 ffmpeg 를 띄운 시각으로, 스트림이 없으면 기다린다).
- 미리보기 지연 줄이기(`low_latency`, 최신 프레임만 표시, `preview_skipped`)는 **화면 표시에만** 영향을 준다. 프레임 저장·녹화는 수신 스레드/ffmpeg 가 따로 하므로 표시가 느려도 기록은 빠지지 않는다.

---

## 3. 문제 점검 순서

| 증상 | 확인 |
|---|---|
| `스트림을 열 수 없음` 반복 | ① `.\mediamtx.exe` 가 떠 있나 ② 서버 로그에 `is publishing to path 'live/drone'` 가 **현재** 있나 (EOF 뒤면 송출 끊김) ③ 주소 오타: `rtsp://127.0.0.1:8554/live/drone` (`live/drone` 는 DJI Fly 에 넣은 경로와 같아야 함) ④ 방화벽: 같은 PC 에서 127.0.0.1 은 보통 무관. 다른 PC 면 8554 허용 |
| DJI Fly 가 연결 안 됨 | 아이폰과 PC 가 같은 Wi-Fi 인지, PC IPv4 가 맞는지(`ipconfig`), Windows 방화벽에서 TCP 1935 인바운드 허용(`mediamtx.exe` 첫 실행 때 뜨는 허용 창을 '허용'), 아이폰 핫스팟이면 PC 가 그 핫스팟에 붙어야 함 |
| 서버에 `closed: EOF` | 송신 쪽이 끊은 것. 아이폰 화면 꺼짐/앱 전환/Wi-Fi 전환/드론 전원. 수신 프로그램은 자동 재접속하므로 재송출만 하면 됨 |
| 연결은 되는데 검은 화면·회색 깨짐 | `rtsp_transport: tcp` 인지 확인(udp 에서 흔함). ffmpeg 로 직접 확인: `ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/live/drone`. 첫 키프레임 전까지 1~2초 깨질 수 있음 |
| fps 가 0 에 가깝고 `RECONNECTING` 반복 | `stall_timeout_s`(기본 5초) 안에 프레임이 없음. 송출 비트레이트/네트워크 문제. 서버 로그와 아이폰 DJI Fly 의 송출 상태 확인 |
| 녹화 파일이 0 바이트 | `video/ffmpeg_segNNN.log` 확인. 스트림이 없으면 ffmpeg 가 기다리기만 한다. 파일이 안 커지면 2×stall 후 구간을 닫고 재시작한다 |
| `ffmpeg 를 찾지 못했습니다` | 1 절 FFmpeg. 설치 후 **새 터미널**에서 실행 |
| 창이 안 뜨고 오류 | `opencv-python-headless` 가 설치돼 있음 → 제거 후 `opencv-python`. 창 없이 저장만 하려면 `--no-display` |
| 프레임 `saved=0` 많음 | 디스크가 느림. `frames.interval_s` 늘리기, `format: jpg`, SSD 로 `session.root` 변경 |

VLC 로 보려면: 미디어 → 네트워크 스트림 열기 → `rtsp://127.0.0.1:8554/live/drone`. (VLC 는 기본 캐시가 1초 이상이라 지연이 크다. 수신 프로그램 창과 비교용으로만.)

---

## 4. 2단계: 화면 OCR

### 4-1. 아이폰 화면을 PC 로 가져오는 방법 (아직 검증 안 됨)

아이폰 USB 포트는 조종기에 쓰므로 **무선**이어야 한다. Windows 에는 AirPlay 수신 기능이 **기본으로 없다** — 서드파티 수신 앱이 필요하다. (2026-10 기준 검색 결과. 가격·제한은 설치 전에 각 사이트에서 다시 확인할 것.)

| 방법 | 비용/제약 | 비고 |
|---|---|---|
| **아이폰 자체 화면 녹화** (제어 센터 → 화면 기록) → 비행 후 파일을 PC 로 복사 → `22_ocr_run.py --video` | 무료, 실시간 아님 | **가장 확실하고 먼저 권장.** DJI Fly 중에도 동작하는지, 프레임 드랍·발열은 실비행에서 확인. 파일 시작 시각은 아이폰 사진 앱의 생성 시각으로만 알 수 있어 오프셋 보정 필수 |
| AirPlay 수신 앱 (LonelyScreen, 5KPlayer, PigeonCast, AirServer, Reflector, ApowerMirror, LetsView 등) → `22_ocr_run.py --screen` | 무료판은 워터마크·시간 제한·해상도 제한이 흔함. AirServer/Reflector 는 유료 | 아이폰과 PC 가 **같은 Wi-Fi**. DJI Fly 가 RTMP 송출 중이면 아이폰 업로드 대역폭을 둘이 나눠 쓴다(둘 다 끊길 수 있음). 미러링 지연은 수백 ms~1초 이상이며 일정하지 않다 |
| DJI Fly 자체 기능 | — | Mini 5 Pro 는 2026-10 현재 DJI 공식 호환표에서 MSDK 미지원. 영상 자막(SRT)에 고도·속도가 기록되지만 **비행 후 파일**로만 얻는다. 실시간 OCR 의 대체가 아니라 사후 검증용으로 좋다 (litter3d/srt.py 참고) |

미러링이 준비되지 않아도 **스크린샷 1장**으로 ROI 선택과 OCR 을 시험할 수 있다 (`--image`).

### 4-2. 표시값의 의미 (화면을 확인한 뒤 config 의 `ocr.fields` 를 고칠 것)

DJI 지원 문서 기준 DJI Fly 왼쪽 아래 표시: `H` = 홈포인트(이륙 지점) 기준 **상대 고도**, `D` = 홈포인트까지 **수평 거리**, 그리고 수평 속도·수직 속도 두 값(아이콘·단위 확인). 기본 설정의 필드 이름 `H, D, HS, VS` 는 이 가정이며, 실제 화면과 다르면 바꾼다.
- **H 는 바로 아래 지면까지의 거리가 아니다.** 경사지·건물 위에서는 지면 거리와 크게 다르다.
- **D 는 전방 물체까지의 거리가 아니다.**
- 속도 단위는 설정(m/s, km/h, mph)에 따라 달라지며 OCR 이 읽은 단위를 `unit` 열에 그대로 남긴다. 변환은 `23_sync.py` 에서만, `unit_notes` 에 근거를 적어 한다.
- 표시 소수 자릿수(보통 0.1 m, 0.1 m/s)가 측정 해상도의 한계다.

### 4-3. 실행

```powershell
# (a) 기준 화면으로 ROI 선택: 필드마다 드래그 → Enter, 없는 항목은 c
.\.venv\Scripts\python.exe scripts\21_select_roi.py --image iphone_screenshot.png
.\.venv\Scripts\python.exe scripts\21_select_roi.py --video iphone_record.mp4 --at 30
.\.venv\Scripts\python.exe scripts\21_select_roi.py --screen        # 미러링 창이 보이는 상태에서

# (b) OCR
.\.venv\Scripts\python.exe scripts\22_ocr_run.py --image iphone_screenshot.png --show
.\.venv\Scripts\python.exe scripts\22_ocr_run.py --video iphone_record.mp4 --out data\ocr_test\telemetry.csv
.\.venv\Scripts\python.exe scripts\22_ocr_run.py --screen --session data\sessions\drone_xxx     # 20_capture 와 동시에 실행
```
- ROI 는 선택 당시 프레임 크기와 함께 `config/ocr_roi.json` 에 저장된다. 입력 크기가 다르면 모든 행이 `size_mismatch` 로 기록되고 재선택 안내가 뜬다.
- `ocr_debug/` 에 N 회마다 잘라낸 영역 + 인식 원문 + 해석 결과 몽타주가 저장된다. ROI 를 숫자에 **딱 맞게** 잡는 것이 정확도에 가장 중요하다 (라벨 글자 `H`/`D` 가 함께 들어가면 `label` 설정으로 떼어 낸다).
- `telemetry.csv` 열: `raw_text`(원문), `confidence`(엔진 제공 점수, RapidOCR 0~1 / Tesseract 0~1 환산, 없으면 빈칸), `value`, `unit`, `status`(`ok`/`empty`/`no_number`/`ambiguous`/`unit_mismatch`/`out_of_range`/`size_mismatch`), `notes`.
- `value=0` 과 `status=ok` 는 **정상 0**, 실패는 `status` 로 구분된다. 이전 값을 채우는 일은 없다.
- 처리 빈도 `ocr.interval_s`. CPU 사용량을 보고 조정.

### 4-4. 시간 동기화와 매칭

```powershell
.\.venv\Scripts\python.exe scripts\23_sync.py data\sessions\drone_xxx --offset 0.0 --tolerance 500
```
- `offset_s`: **OCR 시각 + offset = 영상 시각**. RTMP 경로(아이폰 인코딩→서버→디코드)와 미러링 경로의 지연이 다르므로 0 이 아닐 가능성이 높다. 정하는 법: 영상과 화면에 **같이 보이는 사건**(예: 이륙 순간 H 가 0→0.x 로 바뀌는 때와 영상에서 지면이 멀어지기 시작하는 프레임, 또는 PC 화면의 시계를 드론 카메라로 찍기)의 두 수신 시각 차이를 몇 번 재서 평균을 넣는다. 그 분산이 동기화 오차의 하한이다.
- 두 기록의 PC 시각이 가깝다고 해서 같은 순간을 가리키는 것이 아니다. `match_time_error_ms` 는 '보정 후 수신 시각 차이' 일 뿐이다.
- `telemetry_valid=1` 은 `field_map` 의 모든 필드가 허용 오차 안에서 `ok` 였을 때만. 일부만 맞으면 0 이고 맞은 필드만 채워진다. 화면에 없는 항목은 `sync.field_map` 에서 지운다(빈 열로 남지 않게).
- 원본 `telemetry.csv` 는 그대로 두고 `matched.csv` 만 새로 만든다.

---

## 5. 3단계: 오프라인 3D 복원 (준비 단계)

```powershell
.\.venv\Scripts\python.exe scripts\24_select_frames.py data\sessions\drone_xxx           # 흐림·중복 제거 → sfm\images, selection.csv
.\.venv\Scripts\python.exe scripts\25_colmap_cmds.py data\sessions\drone_xxx             # COLMAP 명령 출력 (CPU 기본)
.\.venv\Scripts\python.exe scripts\25_colmap_cmds.py data\sessions\drone_xxx --run       # colmap.exe 가 PATH 에 있으면 실행
```
- COLMAP Windows 바이너리: https://colmap.github.io (CUDA 버전·비CUDA 버전이 따로 있다). GPU·VRAM·RAM 을 확인한 뒤 결정. sparse(카메라 위치 + 희소 점군)는 CPU 로도 된다. dense(patch match)는 CUDA 가 필요하고, CPU 대안은 OpenMVS 등.
- **결과는 단위가 없는 상대 좌표**다. 미터로 바꾸려면 장면 안의 알려진 길이가 가장 확실하다. 상대 고도 H 의 **변화량**은 카메라 높이 차로 쓸 수 있지만(지면 거리 아님), 표시 해상도 0.1 m·표시 지연·동기화 오차가 그대로 축척 오차가 된다. D·속도는 방향이 없어 위치를 정하지 못한다. H·D·속도만으로 카메라 6자유도 자세를 알 수 없으므로 SfM 결과의 검증·축척 보조로만 쓴다.
- 1 프레임/초 저장은 넉넉한 쪽이다. 느리게 비행했으면 `min_change` 로 중복을 더 걷어 내고, 빠르게 비행했으면 `frames.interval_s` 를 줄여 다시 수집한다. 겹침이 60 % 아래면 SfM 이 끊긴다.
- Gaussian Splatting 은 시각화용이다. 측정 가능한 표면 메시(COLMAP/OpenMVS 메시)와 같은 것으로 다루지 않는다.

---

## 6. 검증 상태 (2026-10-01)

**이 환경(Linux 컨테이너, 드론 없음)에서 실제로 돌려 확인한 것**
- `pytest tests/test_dronecap.py`: 21개 통과 (설정 병합, 숫자 해석 10 케이스, ROI 저장/크기 검사, 파일 입력 전체 흐름(프레임 저장·opencv 녹화·CSV·세션 JSON), 없는 입력 종료, 매칭(0 값·단위 변환·실패 미연결·허용 오차 초과), 프레임 선별, COLMAP 명령).
- **RTSP 전체 경로 루프백**: MediaMTX v1.21.1(Linux) + `29_test_publish.py` 로 합성 영상을 RTMP 송출 → `20_capture.py` 가 `rtsp://127.0.0.1:8554/live/drone` 수신. 송출을 죽인 뒤 7초 후 재송출 → `RECONNECTING` → `conn#2` 로 재접속, 프레임 저장 1초 간격 유지, 녹화가 seg001/seg002 **두 파일**로 나뉨(둘 다 ffprobe 로 h264 재생 가능, `-c copy`). `frames.csv` 의 `connection_id` 가 1→2, `stream_pos_ms` 가 새 연결에서 다시 작은 값으로 시작하는 것 확인.
- 파일 입력 녹화: ffmpeg 백엔드 12초 영상 → 12.000초 mp4(코덱 그대로).
- OCR: DJI Fly 를 흉내 낸 **합성 화면 3장**(12개 값: 0.0, 음수, 소수 포함)에서 RapidOCR(CPU, 인식 전용) + 파서가 12/12 정확. 검출+인식 모드는 `12.3` 을 `1`/`2.3` 으로 쪼개 틀린 값을 냈기 때문에 기본을 인식 전용으로 바꾸고, 숫자가 둘 이상이면 `ambiguous` 로 비우게 했다.

**검증하지 못한 것 (사용자 환경에서 확인 필요)**
- 실제 DJI Mini 5 Pro / DJI Fly 의 RTMP 스트림 (해상도·fps·키프레임 간격·오디오). 수신 성공 로그는 있지만 이 프로그램으로 받아 본 적은 없다.
- Windows 에서의 창 표시, `winget` FFmpeg 설치, `CREATE_NO_WINDOW`, ffmpeg 에 `q` 를 보내 정상 종료되는지 (Linux 에서는 rc=0 확인).
- `OPENCV_FFMPEG_CAPTURE_OPTIONS` 의 `timeout` 옵션이 사용자 PC 의 OpenCV 빌드(번들 FFmpeg 버전)에서 받아들여지는지. 거부되면 로그에만 남고 동작엔 영향 없다.
- 실제 DJI Fly 화면 레이아웃·글꼴에서의 OCR 정확도, 미러링 앱 지연, 아이폰 화면 녹화가 DJI Fly 와 동시에 되는지.
- COLMAP 실행 (명령 생성만 테스트).

**다음에 필요한 정보**: ① DJI Fly 비행 화면 스크린샷 1장(ROI·필드 확정) ② PC 의 GPU/VRAM/RAM (`dxdiag` 또는 작업 관리자 → 성능) ③ 조종기 모델명(참고용).
