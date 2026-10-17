"""High-quality renders of the four showcase designs."""
import json
import time
from pathlib import Path

from kigumi.build import build
from kigumi.params import Params
from kigumi.render import render

DESIGNS = {
    "four-legs": dict(wood_beams="white_oak", wood_legs="walnut", wood_caps="walnut"),
    "three-legs": dict(leg_count=3, leg_radius=340, field_radius=215, leg_lean=70, wood_beams="white_oak",
                       wood_legs="yakisugi", wood_caps="yakisugi", cap_style="hoju", leg_rot=90),
    "pedestal": dict(leg_count=1, field_radius=250, leg_base_d=440, leg_top_d=140, leg_foot_round=0.14,
                     leg_cap_d=160, wood_beams="ash", wood_legs="wenge", wood_caps="wenge", height=440),
    "dumbbell": dict(leg_layout="rect", leg_radius=400, leg_aspect=0.3, leg_rot=0, field_radius=230,
                     field_blend=110, leg_lean=40, wood_beams="hinoki", wood_legs="walnut", wood_caps="walnut"),
}
out = Path("out/showcase")
(out / "designs.json").write_text(json.dumps(DESIGNS, indent=1))
for name, v in DESIGNS.items():
    d = build(Params().replace(**v))
    for view, res in (("hero", "1600x1100"), ("top", "1200x1200"), ("low", "1600x900")):
        t = time.time()
        render(d, out / f"{name}-{view}.png", view=view, samples=128 if view != "top" else 64, res=res)
        print(name, view, f"{time.time() - t:.0f}s", flush=True)
