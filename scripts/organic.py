"""Asymmetric, organic designs: for each family, a few seeds are built and
checked, and the soundest is rendered (hero and top views).

    uv run python scripts/organic.py out/organic [seeds_per_family]
"""
import json
import sys
import time
from pathlib import Path

from kigumi.build import build
from kigumi.checks import run_checks
from kigumi.params import Params
from kigumi.render import render

FAMILIES = {
    "pebble": dict(leg_layout="spiral", leg_count=5, leg_radius=330, field_var=0.25, leg_var=0.5,
                   field_radius=220, wood_beams="white_oak", wood_legs="walnut", wood_caps="walnut",
                   leg_style="bun"),
    "archipelago": dict(leg_layout="spiral", leg_count=7, leg_radius=440, field_radius=215, field_var=0.3,
                        leg_var=0.6, glass_kind="none", design_load=400, spacing="tree",
                        wood_beams="hinoki", wood_legs="yakisugi", wood_caps="yakisugi", leg_style="spindle"),
    "river": dict(leg_layout="wave", leg_count=5, leg_radius=640, leg_aspect=0.35, leg_waves=1.0,
                  field_radius=230, field_var=0.2, leg_var=0.4, grid="hex", wood_beams="ash",
                  wood_legs="wenge", wood_caps="wenge", leg_style="gourd"),
    "drift": dict(leg_layout="scatter", leg_count=5, leg_radius=380, leg_aspect=0.75, field_radius=225,
                  field_var=0.25, leg_var=0.5, grid="hex", hex_joint="mitsukude", beam_h=40,
                  wood_beams="white_oak", wood_legs="walnut", wood_caps="walnut", leg_style="mushroom"),
    "footprints": dict(leg_layout="free", leg_points="-380,-120; -60,180; 260,-60; 420,260",
                       field_radius=245, field_var=0.15, leg_var=0.8, wood_beams="cherry",
                       wood_legs="walnut", wood_caps="cherry", leg_style="bulb"),
    "cloud": dict(leg_layout="scatter", leg_count=6, leg_radius=430, leg_aspect=0.8, field_radius=215,
                  field_var=0.3, leg_var=0.5, grid="hex", wood_beams="hinoki", wood_legs="sugi",
                  wood_caps="sugi", leg_style="taper"),
    "mushrooms": dict(leg_layout="spiral", leg_count=3, leg_radius=300, field_radius=230, field_var=0.4,
                      leg_var=0.7, leg_style="mushroom", wood_beams="maple", wood_legs="walnut",
                      wood_caps="walnut", spacing="tree"),
    "coral": dict(leg_layout="scatter", leg_count=4, leg_radius=360, leg_aspect=0.7, field_radius=230,
                  field_var=0.3, leg_var=0.6, glass_kind="none", design_load=400, spacing="tree",
                  rim_refine=1.0, wood_beams="white_oak", wood_legs="wenge", wood_caps="wenge",
                  leg_style="hourglass"),
}


def score(res):
    fails = sum(1 for r in res if r["status"] == "fail")
    warns = sum(1 for r in res if r["status"] == "warn")
    return (fails, warns)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/organic")
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, v in FAMILIES.items():
        best = None
        for seed in range(1, n_seeds + 1):
            p = Params().replace(seed=seed, **v)
            t = time.time()
            try:
                d = build(p)
                res = run_checks(d, quick=True)
            except Exception as e:                      # keep going; report it
                print(name, seed, "error", repr(e)[:120], flush=True)
                continue
            sc = score(res)
            print(name, seed, sc, f"{time.time() - t:.1f}s", flush=True)
            if best is None or sc < best[0]:
                best = (sc, seed, d, res)
        if best is None:
            continue
        sc, seed, d, res = best
        summary[name] = dict(params=dict(v, seed=seed), sym=d.layout.sym_label, notes=d.layout.notes,
                             parts=len(d.parts), beams=len(d.layout.beams),
                             checks={r["name"]: [r["status"], r["value"]] for r in res if r["status"] != "pass"})
        for view, res_ in (("hero", "1600x1100"), ("top", "1300x1100")):
            render(d, out / f"{name}-{view}.png", view=view, samples=96 if view == "hero" else 48, res=res_)
        print("rendered", name, flush=True)
    (out / "organic.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
