"""Search many random designs, keep the soundest distinct ones, render a sheet.
    uv run python scripts/gallery.py out/gallery.png [batches] [seed0]"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kigumi.build import build
from kigumi.params import Params
from kigumi.render import render
from kigumi.search import best_of

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/gallery.png")
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 6
seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 20261016
t = time.time()
pool = []
for b in range(batches):
    pool.extend(best_of(seed0 + 104729 * b, 12))
pool.sort(key=lambda r: r["score"], reverse=True)
picked, seen = [], set()
for r in pool:
    pr = r["params"]
    key = (pr["leg_count"] if pr["leg_layout"] == "ring" else pr["leg_layout"], pr["top_shape"], pr["edge_mode"])
    if key in seen and len(picked) < 6:
        continue
    seen.add(key)
    picked.append(r)
    if len(picked) == 12:
        break
print(f"searched {len(pool)} designs in {time.time() - t:.0f}s")
tmp = out.parent / (out.stem + "_parts")
tmp.mkdir(parents=True, exist_ok=True)
(out.parent / (out.stem + ".json")).write_text(json.dumps([dict(seed=r["seed"], score=r["score"], fails=r.get("fails"),
                                                               warns=r.get("warns"), params=r["params"]) for r in picked], indent=1))


def one(i_r):
    i, r = i_r
    d = build(Params.from_dict(r["params"]))
    f = render(d, tmp / f"{i:02d}.png", view="hero", samples=40, res="760x520")
    return i, f


with ThreadPoolExecutor(3) as ex:
    done = sorted(ex.map(one, enumerate(picked)))
W, H = 760, 520
cols = 3
rows = (len(done) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * (H + 30)), (245, 244, 240))
dr = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    font = ImageFont.load_default()
for i, f in done:
    r = picked[i]
    pr = r["params"]
    c, rr = i % cols, i // cols
    sheet.paste(Image.open(f), (c * W, rr * (H + 30)))
    legs = pr["leg_count"] if pr["leg_layout"] == "ring" else pr["leg_layout"]
    label = (f"seed {r['seed']}  legs {legs}  {pr['top_shape']}/{pr['edge_mode']}  {pr['wood_beams']}+{pr['wood_legs']}"
             f"  fails {r.get('fails')} warns {r.get('warns')}")
    dr.text((c * W + 8, rr * (H + 30) + H + 7), label, fill=(40, 40, 40), font=font)
sheet.save(out)
print("gallery", out, f"{time.time() - t:.0f}s")
