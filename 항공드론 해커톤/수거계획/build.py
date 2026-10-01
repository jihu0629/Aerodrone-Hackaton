#!/usr/bin/env python3
"""src/ 의 조각을 합쳐 단일 HTML(문갑도_수거계획_v2.html)을 만든다.

    python3 build.py            # 기본 출력 파일
    python3 build.py out.html   # 다른 이름으로

편집은 src/app.js (로직), src/styles.css (모양), src/index.src.html (화면 구조),
src/plan.json (데이터·이미지) 에서 하고, 이 스크립트로 다시 합친다.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "문갑도_수거계획_v2.html"

meta = json.loads((SRC / "build_meta.json").read_text(encoding="utf-8"))
tpl = (SRC / "index.src.html").read_text(encoding="utf-8")
css = (SRC / "styles.css").read_text(encoding="utf-8")
plan_txt = (SRC / "plan.json").read_text(encoding="utf-8").rstrip("\n")
app = (SRC / "app.js").read_text(encoding="utf-8").rstrip("\n")

json.loads(plan_txt)  # 데이터가 올바른 JSON 인지 먼저 확인

si, pi, ai = meta["style_indent"], meta["plan_indent"], meta["app_indent"]
html = (
    tpl.replace("<!--@@STYLES@@-->", f"<style>\n{css}\n{si}</style>", 1)
    .replace("<!--@@PLAN@@-->", f"<script>\n{pi}    window.PLAN = {plan_txt};\n{pi}</script>", 1)
    .replace("<!--@@APP@@-->", f"<script>\n{app}\n{ai}</script>", 1)
)
for marker in ("<!--@@STYLES@@-->", "<!--@@PLAN@@-->", "<!--@@APP@@-->"):
    if marker in tpl and marker in html:
        sys.exit(f"치환 실패: {marker}")
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html.encode('utf-8')):,} bytes)")
