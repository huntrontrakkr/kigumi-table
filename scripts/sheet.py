"""Render a grid of variants for side-by-side comparison.

    uv run python scripts/sheet.py out/sheet.png view '{"tiers":5}' '{"pitch":70}' ...
Each JSON overrides the default (or "preset": name) parameters.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kigumi.build import build
from kigumi.params import PRESETS, Params, preset
from kigumi.render import render

out = Path(sys.argv[1])
view = sys.argv[2]
variants = [json.loads(a) for a in sys.argv[3:]]
res = "760x520"
tmp = out.parent / (out.stem + "_parts")
tmp.mkdir(parents=True, exist_ok=True)


def one(i_v):
    i, v = i_v
    v = dict(v)
    base = preset(v.pop("preset")) if "preset" in v else Params()
    p = base.replace(**v)
    d = build(p)
    f = render(d, tmp / f"{i:02d}.png", view=view, samples=40, res=res)
    return i, f, d.layout.notes


t = time.time()
with ThreadPoolExecutor(3) as ex:
    results = sorted(ex.map(one, enumerate(variants)))
cols = min(3, len(results))
rows = (len(results) + cols - 1) // cols
W, H = (int(x) for x in res.split("x"))
sheet = Image.new("RGB", (cols * W, rows * (H + 34)), (245, 244, 240))
dr = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
for i, f, notes in results:
    r, c = divmod(i, cols)
    sheet.paste(Image.open(f), (c * W, r * (H + 34)))
    label = f"[{i}] " + json.dumps(variants[i])[:110]
    dr.text((c * W + 8, r * (H + 34) + H + 8), label, fill=(40, 40, 40), font=font)
sheet.save(out)
print("sheet", out, f"{time.time() - t:.1f}s")
