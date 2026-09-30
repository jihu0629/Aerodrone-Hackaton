"""
Windows 원클릭 실행: 가상환경 생성 -> 패키지 설치 -> 01 -> 02 -> 03 순서로 실행.

VS Code 에서 이 파일을 열고 오른쪽 위 ▶ (Run Python File) 버튼만 누르면 됩니다.
(터미널을 열어도 됩니다:  python scripts\\run_windows.py)

하는 일
  1) C:\\hackathon\\.venv 가상환경 생성 (처음 한 번, 1~2분) + requirements.txt 설치 (처음 몇 분)
  2) SkySat tif 위치 확인 (아래 TIF_PATH, 없으면 C:\\hackathon 에서 *_visual.tif 검색)
  3) scripts/01_find_island.py      섬 크롭
     scripts/02_extract_coastline.py 해안선
     scripts/03_bench_register.py    정합 벤치마크
  4) 결과 폴더 C:\\hackathon\\outputs 열기

결과물은 한글 경로를 피해서 C:\\hackathon\\outputs 에 저장합니다.
"""

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# 여기만 본인 환경에 맞게 수정
# ============================================================
# SkySat 원본 tif. OneDrive 밖(C:\hackathon)에 복사해 두는 것을 권장합니다.
TIF_PATH = r"C:\hackathon\20260817_231616_ssc1_u0002_visual.tif"

# 저장소(Aerodrone-Hakaton) 폴더. 비워 두면 자동으로 찾습니다.
# 자동 탐색이 실패하면 GitHub Desktop 의 Repository > Show in Explorer 로 확인한 경로를 넣으세요.
REPO_PATH = r""

WORK = Path(r"C:\hackathon")          # 가상환경·결과 저장 위치 (한글 없음)
OUT = WORK / "outputs"
STEPS = ["01", "02", "03"]             # 실행할 단계. 04 는 Sentinel-2 파일이 있을 때 따로 실행
BENCH_TRIALS = 5
# ============================================================



def _is_repo(p: Path) -> bool:
    return (p / "coastcd" / "__init__.py").exists() and (p / "requirements.txt").exists()


def find_repo() -> Path:
    """이 파일이 저장소 밖에 복사돼 있어도 저장소를 찾습니다."""
    here = Path(__file__).resolve()
    tried = []
    if REPO_PATH:
        tried.append(Path(REPO_PATH))
    tried += [here.parent, *here.parents]
    home = Path.home()
    tried += [
        WORK / "Aerodrone-Hakaton",
        home / "OneDrive - 인하대학교" / "바탕 화면" / "프로젝트",
        home / "OneDrive - 인하대학교" / "바탕 화면" / "프로젝트" / "Aerodrone-Hakaton",
        home / "Desktop" / "Aerodrone-Hakaton",
        home / "Documents" / "GitHub" / "Aerodrone-Hakaton",
    ]
    for p in tried:
        if _is_repo(p):
            return p
    # 사용자 홈 아래 2단계까지 검색
    for base in [home, home / "OneDrive - 인하대학교", home / "OneDrive"]:
        if base.exists():
            for p in base.glob("*/*"):
                if p.is_dir() and _is_repo(p):
                    return p
    sys.exit(
        "\n[오류] 저장소(Aerodrone-Hakaton) 폴더를 찾지 못했습니다.\n"
        "  저장소 안에는 coastcd 폴더와 requirements.txt 가 있어야 합니다.\n"
        "  1) GitHub Desktop 에서 main 브랜치로 'Pull origin' 을 했는지 확인하세요.\n"
        "  2) 그래도 안 되면 이 파일 위쪽 REPO_PATH 에 저장소 경로를 넣으세요.\n"
        "     (GitHub Desktop: Repository > Show in Explorer 로 경로 확인)\n"
        f"  확인한 위치: {here}"
    )


REPO = find_repo()
VENV = WORK / ".venv"
PY = VENV / "Scripts" / "python.exe"

env = os.environ.copy()
for k in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "GDAL_DATA", "PROJ_LIB", "PROJ_DATA"]:
    env.pop(k, None)
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"


def run(cmd, **kw):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd], env=env, **kw)


def find_tif() -> Path:
    p = Path(TIF_PATH)
    if p.exists():
        return p
    print(f"[!] TIF_PATH 에 파일이 없습니다: {p}")
    for cand in list(WORK.glob("*_visual.tif")) + list(WORK.rglob("*_visual.tif")):
        print(f"    대신 찾음: {cand}")
        return cand
    default = REPO / "항공드론 해커톤" / "기업 배포 자료" / "SkySatCollect" / "20260817_231616_ssc1_u0002_visual.tif"
    if default.exists():
        print(f"    저장소 안 원본 사용 (한글 경로 -> 8.3 짧은 경로로 우회 시도): {default}")
        return default
    sys.exit("    tif 를 찾지 못했습니다. TIF_PATH 를 수정하세요.")


def main() -> None:
    print(f"[0] 저장소: {REPO}")
    print(f"[1] 작업 폴더 {WORK}")
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    if not PY.exists():
        print("[2] 가상환경 생성 (처음 한 번만)")
        base = getattr(sys, "_base_executable", sys.executable)
        run([base, "-m", "venv", VENV])
    else:
        print("[2] 가상환경 이미 있음")
    print("[2] 패키지 설치/확인")
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    run([PY, "-m", "pip", "install", "-r", REPO / "requirements.txt", "-q"])

    tif = find_tif()
    print(f"[3] 입력: {tif}")

    scripts = {
        "01": [REPO / "scripts" / "01_find_island.py", "--tif", tif, "--out", OUT],
        "02": [REPO / "scripts" / "02_extract_coastline.py", "--tif", OUT / "island_crop.tif", "--out", OUT],
        "03": [REPO / "scripts" / "03_bench_register.py", "--tif", OUT / "island_crop.tif", "--out", OUT,
               "--trials", BENCH_TRIALS],
    }
    for step in STEPS:
        print(f"\n===== 단계 {step} =====")
        run([PY, *scripts[step]], cwd=REPO)

    print(f"\n[완료] 결과: {OUT}")
    print("  overview.png              섬 bbox 표시")
    print("  island_crop_preview.png   섬 크롭 미리보기")
    print("  coastline_overlay.png     해안선 (빨간 선)  <- 이 그림을 확인하세요")
    print("  bench_register.csv        정합 벤치마크")
    if os.name == "nt":
        os.startfile(OUT)  # type: ignore[attr-defined]


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[오류] 명령 실패 (종료코드 {e.returncode}). 위 출력의 마지막 오류 메시지를 복사해서 보내 주세요.")
        sys.exit(1)
