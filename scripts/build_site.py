"""Pack the Python package for the browser build: site/py/kigumi.zip (the web
worker unpacks it into Pyodide's file system).  CI runs this before deploying
site/ to GitHub Pages.      uv run python scripts/build_site.py"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "site" / "py" / "kigumi.zip"
out.parent.mkdir(parents=True, exist_ok=True)
skip = {"server.py", "render.py", "render_blender.py"}          # local-only (HTTP server, Blender)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((ROOT / "kigumi").glob("*.py")):
        if f.name not in skip:
            z.write(f, f"kigumi/{f.name}")
print(out, out.stat().st_size, "bytes")
