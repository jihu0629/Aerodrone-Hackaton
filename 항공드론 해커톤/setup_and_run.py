""
터미널 없이 한 번에 해결하는 스크립트
====================================
VS Code에서 이 파일을 열고 오른쪽 위 ▶(Run Python File) 버튼만 누르면 됩니다.

하는 일:
  1) 한글이 없는 경로 C:\\hackathon 폴더 생성
  2) s2_download.py 를 그 폴더로 복사
  3) C:\\hackathon\\.venv 가상환경 새로 생성 + 패키지 설치
  4) 새 환경으로 s2_download.py 실행
  5) 결과 폴더(C:\\hackathon\\S2_GYD) 자동으로 열기

이유: rasterio(GDAL)가 윈도우에서 '인하대학교' 같은 한글 경로를 처리하지 못해
      UnicodeDecodeError 가 발생함. 작업 위치만 영문 경로로 옮기면 해결됨.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(r"C:\hackathon")
VENV = WORK / ".venv"
PY = VENV / "Scripts" / "python.exe"
PKGS = ["pystac-client", "odc-stac", "rioxarray", "matplotlib", "pandas"]

# 기존 가상환경/GDAL 관련 환경변수가 새 환경에 섞이지 않도록 정리
env = os.environ.copy()
for k in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA"]:
    env.pop(k, None)


def run(cmd, **kw):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], env=env, **kw)


print(f"[1] 작업 폴더: {WORK}")
WORK.mkdir(exist_ok=True)

print("[2] s2_download.py 복사")
shutil.copy2(HERE / "s2_download.py", WORK / "s2_download.py")

if not PY.exists():
    print("[3] 가상환경 생성 (처음 한 번만, 1~2분)")
    base_python = getattr(sys, "_base_executable", sys.executable)
    run([base_python, "-m", "venv", VENV])
else:
    print("[3] 가상환경 이미 있음 → 건너뜀")

print("[3] 패키지 설치 (처음엔 몇 분 걸림)")
run([PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])
run([PY, "-m", "pip", "install", *PKGS, "-q"])

print("[4] Sentinel-2 다운로드 실행")
run([PY, "s2_download.py"], cwd=WORK)

out = WORK / "S2_GYD"
print(f"\n[5] 완료! 결과 폴더: {out}")
if out.exists():
    os.startfile(out)
