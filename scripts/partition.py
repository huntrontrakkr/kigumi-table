"""Plan figure of the kd-tree partition: leaves (grey), kept cells (tinted) and
the top strips coloured by level (dark = the lines framing the posts).

    uv run python scripts/partition.py out/partition.png '{}' '{"preset":"field-pedestal"}' ...
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MPoly

import kigumi.layout as KL
from kigumi.params import Params, preset

out = sys.argv[1]
variants = [json.loads(a) for a in sys.argv[2:]] or [{}]
cols = min(3, len(variants))
rows = (len(variants) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 6.6 * rows), squeeze=False)
captured = {}
_make = KL.make_partition


def make(*a, **k):
    captured["part"] = _make(*a, **k)
    return captured["part"]


KL.make_partition = make
cmap = plt.get_cmap("copper")
for ax, v in zip(axes.flat, variants):
    v = dict(v)
    base = preset(v.pop("preset")) if "preset" in v else Params()
    p = base.replace(**v)
    captured.clear()
    lay = KL.build_layout(p.effective())
    part = captured.get("part")
    ax.add_patch(MPoly(list(lay.outline.exterior.coords), fill=False, ec="0.45", lw=1.2))
    if part is not None and part.get("tree"):
        nA, nB = part["axes"][0]["n"], part["axes"][1]["n"]
        kept = set(part["kept"])
        for (I, J) in part["leaves"]:
            cs = [I[0] * nA + J[0] * nB, I[1] * nA + J[0] * nB, I[1] * nA + J[1] * nB, I[0] * nA + J[1] * nB]
            ax.add_patch(MPoly(cs, fill=(I, J) in kept, fc=(0.93, 0.89, 0.80), ec="0.82", lw=0.4))
    top = [b for b in lay.beams if b.top_surface]
    lmax = max([getattr(b, "level", 0) for b in top] + [1])
    for B in sorted(top, key=lambda b: -getattr(b, "level", 0)):
        lv = getattr(B, "level", 0)
        col = cmap(0.15 + 0.75 * lv / lmax)
        if B.core is not None:
            ax.add_patch(MPoly(list(KL.strip(B.origin, B.u, B.t0, B.t1, B.w).exterior.coords),
                               fc=(*col[:3], 0.25), ec="none"))
            c0, c1 = B.core
        else:
            c0, c1 = B.t0, B.t1
        ax.add_patch(MPoly(list(KL.strip(B.origin, B.u, c0, c1, B.w).exterior.coords), fc=col, ec="none"))
    for L in lay.legs:
        ax.add_patch(Circle(L.xy, 0.5 * p.post_d, fc="k", ec="none"))
    label = json.dumps(v) if v else "default"
    ax.set_title(f"{label[:60]}\nlevels 0..{lmax} (dark = framing the posts), symmetry {lay.sym_label}", fontsize=9)
    b = lay.outline.bounds
    R = 0.62 * max(b[2] - b[0], b[3] - b[1])
    cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cy - R, cy + R)
    ax.set_aspect("equal")
    ax.axis("off")
for ax in list(axes.flat)[len(variants):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=90)
print(out)
