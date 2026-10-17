"""Design report: summary numbers, parts, assembly order (checks are added by checks.py)."""
from __future__ import annotations

from collections import Counter

from .build import Design

DENSITY = {  # kg/m^3, air-dry
    "hinoki": 410, "sugi": 380, "white_oak": 750, "walnut": 640, "cherry": 560, "ash": 670,
    "maple": 700, "yakisugi": 380, "wenge": 870, "glass": 2500, "acrylic": 1190,
    "rubber": 1200, "silicone": 1100,
}


def summarize(d: Design) -> dict:
    lay = d.layout
    p = d.p
    parts = []
    mass = Counter()
    for q in d.parts:
        if q.solid.is_empty():
            continue
        vol = q.solid.volume()                  # mm^3
        kg = vol * 1e-9 * DENSITY.get(q.material, 600)
        mass[q.material] += kg
        info = dict(id=q.id, kind=q.kind, material=q.material, order=q.order, kg=round(kg, 3))
        if q.kind == "beam":
            info.update(tier=q.meta["tier"], length=round(q.meta["length"], 1),
                        section=f'{p.beam_w:.0f} x {lay.tier_h[q.meta["tier"]]:.0f}')
        parts.append(info)
    b = lay.outline.bounds
    joints = Counter(r.note for r in d.rels)
    return dict(
        size=dict(w=round(b[2] - b[0]), d=round(b[3] - b[1]), h=round(p.height)),
        counts=dict(Counter(q.kind for q in d.parts)),
        beams_per_tier=[sum(1 for B in lay.beams if B.tier == t) for t in range(len(lay.z_tiers))],
        joints=dict(joints),
        mass_kg={k: round(v, 2) for k, v in mass.items()},
        total_kg=round(sum(mass.values()), 1),
        notes=list(lay.notes),
        parts=parts,
        assembly=assembly_steps(d),
        timings={k: round(v, 3) for k, v in d.timings.items()},
    )


def assembly_steps(d: Design) -> list[dict]:
    """Group parts that drop in at the same stage."""
    steps = []
    def stage(q):
        if q.kind in ("leg", "post"):
            return (0, "Stand the legs (posts up) on their layout")
        if q.kind == "beam":
            t = q.meta["tier"]
            return (1 + 2 * t, f"Drop in tier {t} beams")
        if q.kind == "block":
            return (2 + 2 * int((q.order - 10) // 10), "Seat the masu blocks over the posts")
        if q.kind in ("washer", "pad"):
            return (900, "Place washers on the posts and silicone bumpers on the grid")
        if q.kind == "glass":
            return (1000, "Lower the glass over the posts")
        if q.kind == "bushing":
            return (1001, "Fit the flanged bushings into the glass holes")
        return (1002, "Screw down the caps (hand tight)")
    groups = {}
    for q in sorted(d.parts, key=lambda q: q.order):
        k, text = stage(q)
        groups.setdefault((k, text), []).append(q.id)
    for i, ((k, text), ids) in enumerate(sorted(groups.items())):
        steps.append(dict(step=i + 1, text=text, parts=ids))
    return steps
