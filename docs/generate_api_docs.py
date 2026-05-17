# SPDX-License-Identifier: GPL-3.0-or-later
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "public" / "api"
SITE_NAV = """
        <h2>Site</h2>
        <ul>
            <li><a href="/fdm-edl/">Home</a></li>
        </ul>

"""

if OUT.exists():
    shutil.rmtree(OUT)

OUT.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pdoc",
        "fdm_edl",
        "--docformat",
        "numpy",
        "--output-directory",
        str(OUT),
    ],
    cwd=ROOT,
    check=True,
)

for html_file in OUT.rglob("*.html"):
    content = html_file.read_text(encoding="utf-8")
    if "<h2>Site</h2>" in content:
        continue
    content = content.replace(
        "            <h2>Submodules</h2>",
        SITE_NAV + "            <h2>Submodules</h2>",
        1,
    )
    html_file.write_text(content, encoding="utf-8")
