"""Plan-view diagnostics of layouts (no Blender): beams coloured by tier.

    uv run python scripts/plan.py out/plan.png '{}' '{"grammar":"radial"}' ...
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MPoly

from kigumi.layout import build_layout
from kigumi.params import PRESETS, Params, preset

out = sys.argv[1]
variants = [json.loads(a) for a in sys.argv[2:]] or [{}]
cols = min(3, len(variants))
rows = (len(variants) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.4 * rows), squeeze=False)
cmap = plt.get_cmap("viridis")
for ax, v in zip(axes.flat, variants):
    v = dict(v)
    base = preset(v.pop("preset")) if "preset" in v else Params()
    p = base.replace(**v)
    lay = build_layout(p)
    T = len(lay.z_tiers)
    def polys(g):
        return list(g.geoms) if hasattr(g, "geoms") else [g]
    for g in polys(lay.outline):
        ax.add_patch(MPoly(list(g.exterior.coords), fill=False, ec="0.5", lw=1.2))
    for r in lay.regions:
        for g in polys(r):
            if not g.is_empty:
                ax.add_patch(MPoly(list(g.exterior.coords), fill=False, ec="0.8", lw=0.8, ls="--"))
    for B in sorted(lay.beams, key=lambda b: b.tier):
        col = cmap(B.tier / max(1, T - 1))
        ax.add_patch(MPoly(list(B.strip().exterior.coords), fc=(*col[:3], 0.55), ec=col, lw=0.8))
    for L in lay.legs:
        ax.add_patch(Circle(L.xy, L.r_cap, fc="none", ec="saddlebrown", lw=1.5))
        ax.add_patch(Circle(L.xy, 0.5 * p.post_d, fc="saddlebrown", ec="none"))
        ax.add_patch(Circle(L.foot_xy, 0.5 * p.leg_base_d, fc="none", ec="saddlebrown", lw=0.8, ls=":"))
    for b in lay.blocks:
        ax.plot(*b.xy, "s", color="crimson", ms=6)
    counts = [sum(1 for B in lay.beams if B.tier == t) for t in range(T)]
    ax.set_title(json.dumps(v)[:70] + f"\nbeams/tier {counts}  " + "; ".join(lay.notes)[:80], fontsize=8)
    b = lay.outline.bounds
    R = 0.55 * max(b[2] - b[0], b[3] - b[1])
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.axis("off")
for ax in list(axes.flat)[len(variants):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=80)
print(out)
