"""한 번에 실행: input/ 폴더의 기업 자료 → outputs/plan/수거계획.html (브라우저 자동 열림).

VS Code 에서 이 파일을 열고 ▶ 를 누르거나, 터미널에서  python run.py
옵션이 필요하면  python scripts/17_collection_plan.py --help
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "17_collection_plan.py"
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")
