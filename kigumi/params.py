"""Design parameters: one flat, typed table that drives every generator stage.

Each entry carries its range, group and label so the same table can feed the
command line, the web sliders and the random sampler.  Lengths are millimetres,
angles degrees.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, fields, asdict
from typing import Any


def P(default, lo=None, hi=None, step=None, group="", label="", choices=None, sample=None, help=""):
    """Declare a parameter.  `sample` is the range the random sampler draws from
    (defaults to lo..hi); it is narrower where extremes rarely look good."""
    return field(default=default, metadata=dict(lo=lo, hi=hi, step=step, group=group, label=label,
                                                choices=choices, sample=sample, help=help))


SPECIES = ["hinoki", "white_oak", "walnut", "cherry", "ash", "maple", "sugi", "yakisugi", "wenge"]
TOP_SHAPES = ["field", "circle", "squircle", "square", "rect", "polygon"]
LEG_LAYOUTS = ["ring", "square", "rect", "spiral", "scatter", "wave", "free"]
GRAMMARS = ["tokyo", "radial"]
EDGE_MODES = ["eaves", "cells", "smooth"]
LEG_MODES = ["shaft", "crossing"]
END_STYLES = ["kurikata", "taper", "scroll", "plain"]
CAP_STYLES = ["rokuyo", "dome", "hoju", "mushroom", "octagon"]
CAPITALS = ["round", "square", "octagon"]
LEG_STYLES = ["custom", "taper", "mushroom", "bun", "gourd", "spindle", "hourglass", "bulb"]
POST_MODES = ["dowel", "integral"]

# Leg styles are settings of the continuous shape controls (a style other than
# "custom" overrides those controls).
LEG_STYLE_KNOBS = {
    "taper": dict(leg_flare_mix=0.1, leg_bow=0.03, leg_belly=0.0, leg_foot_round=0.45, leg_foot_bulge=0.04),
    "mushroom": dict(leg_flare_mix=0.75, leg_flare=0.35, leg_flare_k=1.6, leg_bow=0.08, leg_belly=0.0,
                     leg_foot_round=0.4, leg_foot_bulge=0.08),
    "bun": dict(leg_flare_mix=0.1, leg_bow=0.02, leg_belly=0.0, leg_foot_round=0.75, leg_foot_bulge=0.32,
                leg_foot_bulge_w=0.9),
    "gourd": dict(leg_flare_mix=0.0, leg_bow=0.0, leg_belly=0.42, leg_belly_at=0.24, leg_belly_w=0.2,
                  leg_foot_round=0.7, leg_foot_bulge=0.0),
    "spindle": dict(leg_flare_mix=0.0, leg_bow=0.0, leg_belly=0.28, leg_belly_at=0.45, leg_belly_w=0.32,
                    leg_foot_round=0.6, leg_foot_bulge=0.0),
    "hourglass": dict(leg_flare_mix=0.0, leg_bow=0.0, leg_belly=-0.3, leg_belly_at=0.55, leg_belly_w=0.3,
                      leg_foot_round=0.5, leg_foot_bulge=0.06),
    "bulb": dict(leg_flare_mix=0.55, leg_flare=0.3, leg_flare_k=1.8, leg_bow=0.0, leg_belly=0.25,
                 leg_belly_at=0.1, leg_belly_w=0.12, leg_foot_round=0.75, leg_foot_bulge=0.0),
}


@dataclass
class Params:
    seed: int = P(0, 0, 10_000_000, 1, "Design", "Seed")

    # ---- top ----------------------------------------------------------------
    height: float = P(420.0, 330, 520, 5, "Top", "Table height (floor to glass top)", sample=(390, 460))
    top_shape: str = P("field", group="Top", label="Top shape (field = from the legs)", choices=TOP_SHAPES)
    top_w: float = P(1000.0, 600, 1500, 10, "Top", "Top width / diameter", sample=(820, 1250))
    top_d: float = P(1000.0, 500, 1500, 10, "Top", "Top depth (rect / ellipse)", sample=(600, 1000))
    top_sides: int = P(8, 3, 12, 1, "Top", "Polygon sides", sample=(5, 8))
    top_corner: float = P(70.0, 0, 300, 5, "Top", "Corner radius", sample=(20, 160))
    top_rot: float = P(0.0, -90, 90, 0.5, "Top", "Top rotation", sample=(0, 0))
    squircle_n: float = P(4.0, 2.2, 8, 0.1, "Top", "Squircle exponent", sample=(3, 5))
    glass_t: float = P(10.0, 6, 19, 1, "Top", "Glass thickness", sample=(10, 12))
    glass_kind: str = P("glass", group="Top", label="Top material", choices=["glass", "acrylic", "none"])
    pad_t: float = P(1.5, 0.5, 4.0, 0.1, "Top", "Silicone bumper thickness", sample=(1.5, 1.5))
    overhang: float = P(28.0, 0, 200, 1, "Top", "Glass overhang past the lattice", sample=(15, 45))

    # ---- radiance field ---------------------------------------------------
    field_radius: float = P(270.0, 120, 700, 5, "Field", "Leg field radius", sample=(210, 340))
    field_blend: float = P(90.0, 1, 400, 1, "Field", "Field blend (metaball softness)", sample=(40, 180))
    field_n: float = P(2.0, 1.2, 8.0, 0.1, "Field", "Field shape (2 circle, >2 squarer)", sample=(2.0, 3.0))
    field_var: float = P(0.0, 0.0, 0.4, 0.01, "Field", "Per-leg radius variation", sample=(0.0, 0.2))
    edge_mode: str = P("eaves", group="Field", label="Edge (eaves: tapering tails / whole cells / cut on the curve)",
                       choices=EDGE_MODES)
    tail_reach: float = P(30.0, -100, 150, 1, "Field", "Tail tips past the glass edge (mm)", sample=(0, 60))
    tail_max: float = P(170.0, 20, 400, 5, "Field", "Longest tail past the last crossing (mm)", sample=(120, 220))
    tail_rise: float = P(0.62, 0.0, 0.85, 0.01, "Field", "Tail tip thinness (fraction of depth)", sample=(0.45, 0.75))
    tail_curve: float = P(1.6, 0.5, 4.0, 0.05, "Field", "Tail underside curve", sample=(1.2, 2.4))
    tail_drop: float = P(4.0, 0.0, 15.0, 0.5, "Field", "Tail top falls away from the glass (mm)", sample=(3, 6))
    glass_fit: str = P("field", group="Field", label="Glass outline (the field / hug the lattice)", choices=["field", "lattice"])
    glass_cover: bool = P(True, group="Field",
                          label="Glass covers the tails (they tuck in under its edge, clear of it)")

    # ---- legs ---------------------------------------------------------------
    leg_layout: str = P("ring", group="Legs", label="Leg layout", choices=LEG_LAYOUTS)
    leg_count: int = P(4, 1, 8, 1, "Legs", "Leg count (ring; 1 = pedestal)", sample=(3, 6))
    leg_radius: float = P(275.0, 120, 600, 5, "Legs", "Leg radius from centre", sample=(210, 330))
    leg_mode: str = P("shaft", group="Legs", label="Leg joint (open shaft / through crossing)", choices=LEG_MODES)
    leg_aspect: float = P(0.6, 0.3, 1.0, 0.01, "Legs", "Depth ratio (rect, scatter ellipse, wave amplitude)")
    leg_waves: float = P(1.0, 0.25, 3.0, 0.05, "Legs", "Wave layout: meanders along its length", sample=(0.5, 1.5))
    leg_points: str = P("", group="Legs", label="Free layout points: x,y; x,y; ... (mm)")
    leg_var: float = P(0.0, 0.0, 1.0, 0.01, "Legs", "Each leg its own (lean, girth, belly; seeded)", sample=(0.0, 0.6))
    leg_rot: float = P(45.0, -180, 180, 0.5, "Legs", "Leg layout rotation", sample=(0, 45))
    leg_style: str = P("custom", group="Legs", label="Leg style (overrides the shape controls)", choices=LEG_STYLES)
    leg_base_d: float = P(140.0, 70, 520, 1, "Legs", "Foot diameter", sample=(118, 170))
    leg_top_d: float = P(76.0, 40, 140, 1, "Legs", "Stem top diameter", sample=(64, 90))
    leg_flare_mix: float = P(0.12, 0.0, 1.0, 0.01, "Legs", "Mushroom flare (0 = straight taper)", sample=(0.0, 0.3))
    leg_flare: float = P(0.38, 0.08, 1.0, 0.01, "Legs", "Flare height (fraction of stem)", sample=(0.2, 0.55))
    leg_flare_k: float = P(1.6, 1.0, 3.0, 0.05, "Legs", "Flare softness", sample=(1.2, 2.2))
    leg_bow: float = P(0.04, -0.2, 0.4, 0.01, "Legs", "Entasis bow", sample=(0.0, 0.08))
    leg_belly: float = P(0.0, -0.4, 0.6, 0.01, "Legs", "Belly (+ gourd/spindle, - waist)", sample=(0.0, 0.0))
    leg_belly_at: float = P(0.35, 0.0, 1.0, 0.01, "Legs", "Belly height (fraction of stem)", sample=(0.2, 0.6))
    leg_belly_w: float = P(0.25, 0.05, 0.8, 0.01, "Legs", "Belly width", sample=(0.15, 0.4))
    leg_foot_round: float = P(0.55, 0.0, 0.85, 0.01, "Legs", "Foot rounding (fraction of foot radius)", sample=(0.4, 0.75))
    leg_foot_bulge: float = P(0.08, 0.0, 0.5, 0.01, "Legs", "Bun foot bulge", sample=(0.0, 0.2))
    leg_foot_bulge_w: float = P(0.7, 0.2, 2.0, 0.01, "Legs", "Bun foot height", sample=(0.5, 1.0))
    leg_sweep: float = P(0.0, -80, 80, 1, "Legs", "Stem bows outward (mm)", sample=(0, 0))
    leg_swirl: float = P(0.0, -80, 80, 1, "Legs", "Stem swirls sideways (mm)", sample=(0, 0))
    leg_lean: float = P(30.0, -60, 160, 1, "Legs", "Foot lean outward (mm)", sample=(10, 60))
    leg_splay: str = P("tilt", group="Legs", label="Lean by tilting a straight turned leg / bending the stem",
                       choices=["tilt", "bend"])
    leg_capital: str = P("round", group="Legs", label="Capital", choices=CAPITALS)
    leg_cap_d: float = P(104.0, 50, 180, 1, "Legs", "Capital diameter", sample=(96, 124))
    leg_cap_flare: float = P(22.0, 0, 80, 1, "Legs", "Capital flare height", sample=(10, 34))
    leg_slot: float = P(0.72, 0.3, 1.0, 0.01, "Legs", "Capital slot depth (x beam h)", sample=(0.62, 0.85))
    shaft: float = P(34.0, 16, 120, 1, "Legs", "Open shaft around the post", sample=(30, 44))
    leg_collar: float = P(0.0, 0, 12, 0.5, "Legs", "Neck collar bead", sample=(0, 6))

    # ---- lattice ------------------------------------------------------------
    grammar: str = P("tokyo", group="Lattice", label="Lattice grammar", choices=GRAMMARS)
    grid: str = P("auto", group="Lattice", label="Grid (auto follows the legs' symmetry)",
                  choices=["auto", "square", "hex", "free"])
    tiers: int = P(4, 2, 7, 1, "Lattice", "Tiers", sample=(3, 5))
    tier_step: float = P(16.0, 0, 200, 1, "Lattice", "Corbel step per tier (field inset)", sample=(8, 40))
    beam_w: float = P(18.0, 10, 70, 1, "Lattice", "Beam width", sample=(15, 24))
    beam_h: float = P(32.0, 16, 110, 1, "Lattice", "Beam depth", sample=(26, 40))
    overlap: float = P(0.42, 0.1, 0.5, 0.01, "Lattice", "Tier overlap (x beam h)", sample=(0.3, 0.5))
    spacing: str = P("uniform", group="Lattice",
                     label="Top grid (uniform / tree: kd-tree cells by load / split: dyadic lines / field: smooth)",
                     choices=["uniform", "tree", "split", "field"])
    tier_load: bool = P(True, group="Lattice", label="Bracket tiers follow the load (a tiered system)")
    hex_joint: str = P("kagome", group="Lattice",
                       label="Hex grid joints (kagome: two strips per crossing / mitsukude: three at every node)",
                       choices=["kagome", "mitsukude"])
    hex_motif: str = P("none", group="Lattice",
                       label="Hex tree (experimental): motif in refined hexagons (needs a pitch near 190 mm)",
                       choices=["none", "star", "ring"])
    hex_base: float = P(2.6, 1.0, 4.0, 0.05, "Lattice", "Hex tree: kagome pitch = pitch x", sample=(2.4, 3.2))
    load_refine: float = P(0.8, 0.0, 1.5, 0.05, "Lattice",
                           "Denser where the top bends most (tree; 1 = fully stressed)", sample=(0.4, 1.2))
    rim_refine: float = P(0.0, 0.0, 3.0, 0.05, "Lattice", "Finer cells at the rim (tree, an effect)", sample=(0.0, 1.0))
    pitch: float = P(72.0, 30, 260, 1, "Lattice", "Beam pitch", sample=(52, 96))
    density: float = P(0.6, -0.6, 2.0, 0.01, "Lattice", "Denser toward edge (field / split spacing)", sample=(0.0, 1.2))
    angle: float = P(0.0, -90, 90, 0.5, "Lattice", "Lattice rotation", sample=(0, 0))
    orient: str = P("load", group="Lattice",
                    label="Grid direction (load: least bending, among the legs' symmetric choices / as set)",
                    choices=["load", "as set"])
    tier_turn: float = P(90.0, 30, 90, 0.5, "Lattice", "Turn between tiers (90 grid, 60 kagome)", sample=(90, 90))
    twist: float = P(0.0, -30, 30, 0.5, "Lattice", "Extra twist per tier", sample=(0, 0))
    cantilever: float = P(4.0, 0.8, 8.0, 0.05, "Lattice", "Cantilever past last support (x beam h)", sample=(3.0, 5.0))
    max_cantilever: float = P(320.0, 40, 600, 5, "Lattice", "Cantilever hard limit (mm)", sample=(320, 320))
    top_cantilever: float = P(5.5, 1.0, 12.0, 0.1, "Lattice", "Top grid cantilever (x beam h)", sample=(4.5, 6.5))
    flush_top: bool = P(True, group="Lattice", label="Top two tiers flush (kumiko grid)")
    projection: float = P(1.5, 0.6, 4.0, 0.05, "Lattice", "Beam end projection past last joint (x beam h)", sample=(1.1, 2.0))
    pitch_taper: float = P(0.9, 0.0, 2.0, 0.01, "Lattice", "Lower tiers sparser by", sample=(0.3, 1.4))
    h_taper: float = P(0.18, -0.3, 0.8, 0.01, "Lattice", "Lower tiers deeper by", sample=(0.0, 0.35))
    # radial grammar
    spin_e: float = P(0.45, 0.0, 0.95, 0.01, "Lattice", "Radial: pinwheel offset (x leg radius)", sample=(0.25, 0.7))
    spin_dir: int = P(1, -1, 1, 2, "Lattice", "Radial: spin direction")
    spokes_per_leg: int = P(2, 1, 6, 1, "Lattice", "Radial: pinwheel lines per leg", sample=(1, 3))
    ring_sides_mult: int = P(2, 1, 4, 1, "Lattice", "Radial: ring sides per leg", sample=(1, 3))
    rings_per_tier: int = P(1, 1, 3, 1, "Lattice", "Radial: rings per ring tier", sample=(1, 2))

    # ---- beam ends ------------------------------------------------------------
    end_style: str = P("kurikata", group="Ends", label="Beam end style", choices=END_STYLES)
    end_len: float = P(1.7, 0.0, 5.0, 0.05, "Ends", "End curve length (x beam h)", sample=(1.2, 2.6))
    end_rise: float = P(0.5, 0.0, 0.85, 0.01, "Ends", "End rise (x beam h)", sample=(0.35, 0.65))
    end_p: float = P(2.0, 0.5, 4.0, 0.05, "Ends", "End curve exponent", sample=(1.4, 2.6))

    # ---- joinery --------------------------------------------------------------
    clearance: float = P(0.15, 0.0, 1.0, 0.01, "Joinery", "Joint clearance per side", sample=(0.15, 0.15))
    post_d: float = P(19.0, 12, 40, 0.5, "Joinery", "Threaded post diameter", sample=(19.0, 19.0))
    post_clear: float = P(0.4, 0.1, 2.0, 0.05, "Joinery", "Post hole clearance (radial)", sample=(0.4, 0.4))
    glass_hole_clear: float = P(3.0, 1.0, 8.0, 0.5, "Joinery", "Glass hole clearance (radial)", sample=(3.0, 3.0))
    masu: float = P(1.55, 1.1, 2.4, 0.01, "Joinery", "Masu block size (x beam w)", sample=(1.4, 1.8))
    cap_style: str = P("rokuyo", group="Joinery", label="Cap style", choices=CAP_STYLES)
    cap_d: float = P(76.0, 34, 130, 1, "Joinery", "Cap diameter", sample=(66, 90))
    cap_h: float = P(24.0, 12, 70, 1, "Joinery", "Cap height", sample=(18, 32))
    thread_len: float = P(18.0, 8, 50, 1, "Joinery", "Thread engagement in cap", sample=(18, 18))
    post_mode: str = P("dowel", group="Joinery", label="Post (separate threaded dowel / turned with the leg)", choices=POST_MODES)
    dowel_engage: float = P(40.0, 15, 90, 1, "Joinery", "Dowel threaded into the leg (mm)", sample=(40, 40))
    post_wood: str = P("maple", group="Materials", label="Threaded dowel wood", choices=SPECIES)
    tool_d: float = P(6.35, 1.0, 12.7, 0.05, "Joinery", "CNC bit diameter (dogbone corners)", sample=(6.35, 6.35))

    # ---- checks -------------------------------------------------------------
    design_load: float = P(750.0, 200, 2000, 10, "Checks", "Point load on the top (N)", sample=(750, 750))
    safety: float = P(3.0, 1.5, 6.0, 0.1, "Checks", "Safety factor on wood strength", sample=(3.0, 3.0))

    # ---- materials ----------------------------------------------------------
    wood_legs: str = P("walnut", group="Materials", label="Leg wood", choices=SPECIES)
    wood_beams: str = P("hinoki", group="Materials", label="Lattice wood", choices=SPECIES)
    wood_caps: str = P("walnut", group="Materials", label="Cap wood", choices=SPECIES)

    # ---- helpers ------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        p = cls()
        for f in fields(cls):
            if f.name in d and d[f.name] is not None:
                v = d[f.name]
                t = type(getattr(p, f.name))
                try:
                    v = t(v) if t is not bool else bool(v)
                except (TypeError, ValueError):
                    continue
                setattr(p, f.name, v)
        return p

    def effective(self) -> "Params":
        """The parameters the generator uses: a named leg style sets its controls,
        and the grid setting fixes the turn between tiers (auto: hexes when the
        legs have three- or six-fold symmetry, squares otherwise)."""
        q = self
        if q.leg_style in LEG_STYLE_KNOBS:
            q = q.replace(**LEG_STYLE_KNOBS[q.leg_style])
        if q.grid != "free" and q.grammar == "tokyo":
            kind = q.grid
            angle = q.angle
            if kind == "auto":
                import math as _m
                from .layout import leg_positions
                from .symmetry import common_group, grid_symmetry, point_symmetry
                sym = point_symmetry(leg_positions(q))
                if q.top_shape != "field":
                    # a drawn outline (square, polygon...) has its own symmetry; the table has both
                    from .shapes import top_outline
                    from .symmetry import intersect_symmetry, shape_symmetry
                    sym = intersect_symmetry(sym, shape_symmetry(top_outline(q)))
                kind = "hex" if sym[0] in (3, 6, 9, 12) else "square"
                # turn the lattice so its mirror lines meet the legs' (largest common group)
                step = 30.0 if kind == "hex" else 45.0
                cands = [angle] + ([(_m.degrees(m) % step) for m in sym[1]] if sym[1] else [])
                angle = max(cands, key=lambda a: (len(common_group(sym, grid_symmetry(kind, a))[0]), -abs(a - q.angle)))
            q = q.replace(tier_turn=60.0 if kind == "hex" else 90.0, twist=0.0, angle=angle)
        return q

    def replace(self, **kw) -> "Params":
        d = self.to_dict()
        d.update(kw)
        return Params.from_dict(d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)


def schema() -> list[dict]:
    out = []
    for f in fields(Params):
        m = dict(f.metadata)
        out.append(dict(name=f.name, default=f.default, type=type(f.default).__name__, **m))
    return out


# ---------------------------------------------------------------------------
# presets: hand-picked starting points for the families the grammar covers
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    # the radiance-field family: thin strips in one flat plane, outline from the legs
    "field-four": dict(wood_beams="white_oak", wood_legs="walnut", wood_caps="walnut"),
    "field-trefoil": dict(leg_count=3, leg_radius=340, field_radius=215, leg_lean=80, leg_style="bun",
                          leg_base_d=170, wood_beams="white_oak", wood_legs="yakisugi", wood_caps="yakisugi",
                          cap_style="hoju", leg_rot=90),
    "field-pedestal": dict(leg_count=1, field_radius=250, leg_base_d=440, leg_top_d=140, leg_foot_round=0.14,
                           leg_cap_d=160, beam_h=38, wood_beams="white_oak", wood_legs="wenge", wood_caps="wenge",
                           height=440),
    "field-peanut": dict(leg_layout="rect", leg_radius=400, leg_aspect=0.34, leg_rot=0, field_radius=225,
                         field_blend=110, leg_lean=70, wood_beams="white_oak", wood_legs="walnut", wood_caps="walnut"),
    "field-five-open": dict(leg_count=5, leg_radius=300, field_radius=220, glass_kind="none", leg_lean=50,
                            beam_w=20, beam_h=44, pitch=64, design_load=400, wood_beams="white_oak",
                            wood_legs="walnut", wood_caps="walnut", cap_style="dome"),
    # hexes and triangles: a kagome lattice (three strip families at 60 degrees), each post in a hexagon
    "hex-trefoil": dict(tier_turn=60, tiers=4, leg_count=3, leg_radius=340, field_radius=210, leg_rot=90,
                        beam_w=20, beam_h=40, pitch=76, pitch_taper=0.0, glass_fit="lattice", leg_style="bun",
                        leg_lean=90, leg_base_d=180, wood_beams="white_oak", wood_legs="yakisugi",
                        wood_caps="yakisugi", cap_style="hoju"),
    "hex-six": dict(tier_turn=60, tiers=4, leg_count=6, leg_radius=330, field_radius=200, leg_rot=0,
                    beam_h=38, pitch=72, pitch_taper=0.0, glass_fit="lattice", leg_style="gourd", leg_lean=40,
                    wood_beams="ash", wood_legs="wenge", wood_caps="wenge", cap_style="rokuyo"),
    # the first, chunkier temple-bracket (tokyo) look
    "temple-round": dict(top_shape="circle", top_w=1000, leg_radius=250, leg_rot=45, leg_lean=60, tiers=6, beam_w=34, beam_h=50,
                         pitch=96, density=0.55, cantilever=3.4, projection=1.5, tier_step=40, h_taper=0.15,
                         pitch_taper=0.35, edge_mode="smooth", leg_flare_mix=0.7, leg_foot_round=0.22,
                         leg_top_d=64, leg_base_d=150, leg_bow=0.1, cap_d=60, post_d=22, overhang=45,
                         wood_beams="hinoki", wood_legs="walnut", wood_caps="walnut", cap_style="rokuyo"),
    "igeta-square": dict(top_shape="square", top_w=1000, top_corner=40, leg_layout="square", leg_radius=300,
                         leg_rot=45, tiers=4, beam_w=30, beam_h=46, pitch=110, density=0.2, tier_step=50,
                         edge_mode="smooth", overhang=45, wood_beams="white_oak", wood_legs="yakisugi",
                         wood_caps="yakisugi", cap_style="octagon"),
    "kumiko-long": dict(top_shape="rect", top_w=1300, top_d=650, top_corner=50, leg_layout="rect",
                        leg_radius=470, leg_aspect=0.42, leg_rot=0, tiers=4, pitch=64, density=0.8,
                        overhang=35, wood_beams="cherry", wood_legs="walnut", cap_style="dome"),
}


def preset(name: str) -> Params:
    return Params().replace(**PRESETS[name])


def sample(seed: int, base: Params | None = None, keep: set[str] | None = None, jitter: float = 0.0) -> Params:
    """Random design.  Every parameter with a sampling range is drawn, except
    those in `keep`, which stay at `base`.  jitter > 0 gives a gentle mutation:
    numeric parameters move at most `jitter` x their range from `base` and
    choices change with probability `jitter`."""
    rng = random.Random(seed)
    base = base or Params()
    keep = set(keep or ())
    d = base.to_dict()
    bd = base.to_dict()
    for f in fields(Params):
        if f.name in keep or f.name == "seed":
            continue
        m = f.metadata
        if m.get("choices"):
            if f.name == "glass_kind":
                continue
            if jitter > 0 and rng.random() > jitter:
                continue
            if f.name == "top_shape":
                d[f.name] = rng.choices(TOP_SHAPES, weights=[10, 2, 2, 1, 1, 1])[0]
            elif f.name == "grammar":
                d[f.name] = "tokyo"          # radial is not yet adapted to the open-shaft leg joint
            elif f.name == "edge_mode":
                d[f.name] = rng.choices(EDGE_MODES, weights=[6, 2, 1])[0]
            elif f.name in ("leg_mode", "post_mode", "leg_splay", "glass_fit"):
                continue
            elif f.name == "leg_layout":
                d[f.name] = rng.choices(LEG_LAYOUTS, weights=[5, 2, 1, 3, 3, 2, 0])[0]
            elif f.name == "end_style":
                d[f.name] = rng.choices(END_STYLES, weights=[5, 2, 2, 1])[0]
            elif f.name == "leg_capital":
                d[f.name] = rng.choices(CAPITALS, weights=[8, 1, 1])[0]
            elif f.name == "leg_style":
                d[f.name] = rng.choices(LEG_STYLES, weights=[2, 2, 3, 3, 2, 2, 1, 2])[0]
            elif f.name == "hex_motif":
                continue
            elif f.name == "spacing":
                d[f.name] = rng.choices(["uniform", "tree", "split", "field"], weights=[5, 4, 1, 1])[0]
            else:
                d[f.name] = rng.choice(m["choices"])
            continue
        if isinstance(f.default, bool):
            continue
        rngs = m.get("sample") or ((m["lo"], m["hi"]) if m.get("lo") is not None else None)
        if rngs is None or rngs[0] == rngs[1]:
            continue
        lo, hi = rngs
        if jitter > 0:
            span = (m["hi"] - m["lo"]) if m.get("hi") is not None else (hi - lo)
            v = float(bd[f.name]) + rng.uniform(-1, 1) * jitter * span
            lo_c, hi_c = (m["lo"], m["hi"]) if m.get("lo") is not None else (lo, hi)
            v = min(max(v, lo_c), hi_c)
        else:
            v = rng.uniform(lo, hi)
        if isinstance(f.default, int):
            d[f.name] = int(round(v))
        else:
            d[f.name] = round(v, 3)
    d["seed"] = seed
    if jitter > 0:
        return Params.from_dict(d)

    def free(*names):
        return all(n not in keep for n in names)

    # correlated choices that make coherent families (only where not locked)
    if d["top_shape"] == "polygon" and free("top_rot"):
        d["top_rot"] = 180.0 / d["top_sides"] if rng.random() < 0.5 else 0.0
    if d["grammar"] == "tokyo":
        if free("grid"):
            d["grid"] = "auto"
        if free("twist"):
            d["twist"] = 0.0
        hexy = Params.from_dict(d).effective().tier_turn == 60.0
        if not hexy and d.get("spacing") == "tree" and free("pitch"):
            # the partition shows best with room for a level or two between centre and rim
            d["pitch"] = round(rng.uniform(64, 124))
        if hexy:
            if free("pitch_taper"):
                d["pitch_taper"] = 0.0
            if free("glass_fit"):
                d["glass_fit"] = "lattice"
    if free("spin_dir"):
        d["spin_dir"] = rng.choice([-1, 1])
    if (d["leg_layout"] == "rect" or d["top_shape"] == "rect") and free("leg_layout", "top_shape"):
        d["leg_layout"], d["top_shape"] = "rect", "rect"
        if free("top_d"):
            d["top_d"] = round(d["top_w"] * rng.uniform(0.5, 0.75))
        if free("leg_aspect"):
            d["leg_aspect"] = round(d["top_d"] / d["top_w"] * rng.uniform(0.8, 1.0), 3)
        if free("leg_radius"):
            d["leg_radius"] = round(d["top_w"] * rng.uniform(0.28, 0.36))
        if free("leg_rot"):
            d["leg_rot"] = 0.0
        d["tier_turn"] = 90.0 if free("tier_turn") else d["tier_turn"]
    if d["leg_layout"] == "square" and free("leg_rot"):
        d["leg_rot"] = 45.0
    if free("leg_sweep") and rng.random() < 0.3:
        d["leg_sweep"] = round(rng.uniform(-35, 45), 1)
    if free("leg_swirl") and rng.random() < 0.15:
        d["leg_swirl"] = round(rng.uniform(-40, 40), 1)
    if free("wood_caps"):
        d["wood_caps"] = d["wood_legs"] if rng.random() < 0.7 else d["wood_beams"]
    if d["wood_legs"] == d["wood_beams"] and free("wood_legs") and rng.random() < 0.7:
        d["wood_legs"] = rng.choice(["walnut", "yakisugi", "wenge", "cherry"])
    return Params.from_dict(d)
