"""Build one design and render it.   uv run python scripts/render_one.py [preset] [view...]"""
import sys
import time
from pathlib import Path

from kigumi.build import build
from kigumi.params import PRESETS, Params, preset
from kigumi.render import render

name = sys.argv[1] if len(sys.argv) > 1 else "field-four"
views = sys.argv[2:] or ["hero"]
p = preset(name) if name in PRESETS else Params()
d = build(p)
print("notes:", d.layout.notes)
for v in views:
    t = time.time()
    out = render(d, Path(f"out/{name}-{v}.png"), view=v, samples=64, res="1400x950")
    print(v, out, f"{time.time()-t:.1f}s")
