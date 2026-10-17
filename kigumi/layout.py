"""Generative layout.

The lattice grows from the legs upward, one tier at a time.  Every tier offers
candidate lines (parallel families for the `tokyo` grammar, rings and
pinwheels for `radial`) clipped to an offset of the top outline: lower tiers
are inset further, so the lattice corbels outward as it rises, like a temple
bracket set (tokyo).

Acceptance rule for a candidate beam:
  * it must rest on at least two earlier beams (a leg under it counts as two,
    because the post clamps it there), crossing each completely;
  * each of those earlier beams must be able to reach the crossing without
    running more than `cantilever` past its own outermost support;
  * it may itself run at most `cantilever` past its outermost support.

Once every tier is placed, each beam is cut to its outermost joint plus a
uniform projection (the visible bracket end), and any strips that still touch
without crossing cleanly are extended or cut back.  Notches, slots and holes
are derived later from overlaps (joinery.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .params import Params
from .field import connected_field
from .shapes import clip_line, clip_line_all, extent_along, inset, line_cross, strip, top_outline

EPS = 1e-6


@dataclass
class Leg:
    idx: int
    xy: np.ndarray          # capital centre = post axis
    foot_xy: np.ndarray     # foot centre (differs when the stem leans)
    z_top: float = 0.0      # capital top
    z_slot: float = 0.0     # channel floor = bottom of tier 0
    r_cap: float = 46.0     # capital radius (the beams it carries sit in its slots)
    knobs: dict = field(default_factory=dict)   # this leg's own stem settings (leg_var)


@dataclass
class Beam:
    id: str
    tier: int
    origin: np.ndarray
    u: np.ndarray
    t0: float
    t1: float
    z0: float
    h: float
    w: float
    family: str
    region_t: tuple = (0.0, 0.0)        # interval allowed by the tier region
    allowed: tuple = (0.0, 0.0)         # region and cantilever limits together
    legs: list = field(default_factory=list)
    ends: list = field(default_factory=lambda: [None, None])
    order: float = 0.0
    alive: bool = True
    top_surface: bool = False
    flush_ends: bool = False          # may end flush at a crossing (end lap), as in dense kagome lattices
    tail: bool = False
    core: tuple | None = None         # stretch between the outermost crossings (what carries the glass)
    rim: tuple = (True, True)         # which ends face the rim (may grow eave tails)
    level: int = 0                    # depth in the hierarchical partition (0 = a line framing a post)

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])

    def p(self, t: float) -> np.ndarray:
        return self.origin + t * self.u

    def strip(self, pad: float = 0.0) -> Polygon:
        return strip(self.origin, self.u, self.t0 - pad, self.t1 + pad, self.w + 2 * pad)

    @property
    def length(self) -> float:
        return self.t1 - self.t0

    @property
    def z1(self) -> float:
        return self.z0 + self.h

    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.u[1], self.u[0]))

    def leg_t(self, L: "Leg") -> float:
        return float((L.xy - self.origin) @ self.u)


@dataclass
class Block:
    id: str
    leg: int
    xy: np.ndarray
    u: np.ndarray            # orientation (along the beam it carries)
    size: float
    z0: float
    z1: float
    slot_into: str | None    # id of the beam received in the top slot
    order: float = 0.0


@dataclass
class Layout:
    p: Params
    outline: Polygon
    regions: list
    legs: list
    beams: list
    blocks: list
    z_tiers: list
    z_glass0: float
    z_glass1: float
    post_top: float
    cap_z0: float
    notes: list = field(default_factory=list)
    z_lat: float = 0.0
    tier_h: list = field(default_factory=list)
    pads: list = field(default_factory=list)
    sym_ops: list = field(default_factory=list)
    sym_label: str = "none"
    sym_legs: str = "none"
    field: object = None          # the legs' radiance field (named after the idea, shadows dataclasses.field below)
    top_region: object = None     # where the top grid may reach (outline minus overhang)

    def beam(self, bid: str) -> Beam:
        for b in self.beams:
            if b.id == bid:
                return b
        raise KeyError(bid)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def leg_positions(p: Params) -> list[np.ndarray]:
    R = p.leg_radius
    rot = math.radians(p.leg_rot)
    pts = []
    if p.leg_layout == "ring":
        n = max(1, int(p.leg_count))
        if n == 1:
            return [np.zeros(2)]
        for k in range(n):
            pts.append(R * _unit(rot + 2 * math.pi * k / n))
    elif p.leg_layout == "square":
        for k in range(4):
            pts.append(R * _unit(rot + math.pi / 2 * k))
    elif p.leg_layout == "rect":
        a, b = R, R * p.leg_aspect
        c, s = math.cos(rot), math.sin(rot)
        for x, y in [(a, b), (-a, b), (-a, -b), (a, -b)]:
            pts.append(np.array([c * x - s * y, s * x + c * y]))
    elif p.leg_layout == "spiral":
        # phyllotaxis: the sunflower's golden-angle spiral -- uneven, yet balanced
        n = max(1, int(p.leg_count))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for k in range(n):
            r = R * math.sqrt((k + 0.5) / n)
            pts.append(r * _unit(rot + k * golden))
    elif p.leg_layout == "scatter":
        # well-spread random points in an ellipse (Mitchell's best candidate), seeded
        n = max(1, int(p.leg_count))
        rng = np.random.default_rng(int(p.seed) + 101)
        ax_, ay_ = R, R * p.leg_aspect
        c, s = math.cos(rot), math.sin(rot)

        def draw():
            while True:
                x, y = rng.uniform(-1, 1, 2)
                if x * x + y * y <= 1.0:
                    return np.array([c * ax_ * x - s * ay_ * y, s * ax_ * x + c * ay_ * y])

        pts.append(draw())
        for _ in range(n - 1):
            cand = [draw() for _ in range(24)]
            pts.append(max(cand, key=lambda q: min(np.linalg.norm(q - r_) for r_ in pts)))
    elif p.leg_layout == "wave":
        # along a meander: a long table or bench that sways
        n = max(2, int(p.leg_count))
        c, s = math.cos(rot), math.sin(rot)
        for k in range(n):
            f = k / (n - 1)
            x = R * (2.0 * f - 1.0)
            y = R * p.leg_aspect * math.sin(2.0 * math.pi * f * p.leg_waves)
            pts.append(np.array([c * x - s * y, s * x + c * y]))
    elif p.leg_layout == "free":
        # explicit points "x,y; x,y; ..." in mm
        for chunk in str(p.leg_points).replace("\n", ";").split(";"):
            if chunk.strip():
                x, y = (float(v) for v in chunk.split(","))
                pts.append(np.array([x, y]))
        if not pts:
            pts = [np.zeros(2)]
    else:
        raise ValueError(p.leg_layout)
    if p.leg_layout in ("spiral", "scatter", "wave", "free") and len(pts) > 1:
        # the table balances over its legs: centre them
        cen = np.mean(pts, axis=0)
        pts = [q - cen for q in pts]
    # a degenerate layout (e.g. a rectangle of zero depth) can put two legs on
    # one spot: merge coincident legs
    out: list[np.ndarray] = []
    for q in pts:
        if all(np.linalg.norm(q - r) > 1.0 for r in out):
            out.append(q)
    return out


def end_margin(p: Params) -> float:
    """Wood left beyond a notch before the end of a beam (stops it shearing out)."""
    return max(0.55 * p.beam_h, 22.0)


@dataclass
class Cross:
    ta: float
    tb: float
    ha: float      # half-span of B's strip measured along A
    hb: float
    sin: float


def crossing(A: Beam, B: Beam) -> Cross | None:
    """Where the centre lines of A and B cross, and how much of each line the
    other strip covers (the notch half-length, parallelogram included)."""
    sin = abs(A.u[0] * B.u[1] - A.u[1] * B.u[0])
    if sin < 0.26:          # under ~15 degrees: treat as parallel
        return None
    r = line_cross(A.origin, A.u, B.origin, B.u)
    if r is None:
        return None
    ta, tb = r
    cos = abs(float(A.u @ B.u))
    ha = 0.5 * B.w / sin + 0.5 * A.w * cos / sin
    hb = 0.5 * A.w / sin + 0.5 * B.w * cos / sin
    return Cross(ta, tb, ha, hb, sin)


def inside(t: float, half: float, lo: float, hi: float, margin: float) -> bool:
    return lo + half + margin - 1e-6 <= t <= hi - half - margin + 1e-6


def end_room(X: "Beam", em: float) -> float:
    """Wood a beam must keep past its outermost notch.  Top-grid beams (and every
    beam of a dense kagome lattice, where crossings come every ~half pitch) may
    end flush with the far side of the crossing member -- an end lap, as at a
    frame corner; other corbel beams keep a full end margin."""
    return 0.0 if (X.top_surface or X.flush_ends) else em


def bands_overlap(A: Beam, B: Beam) -> float:
    return min(A.z1, B.z1) - max(A.z0, B.z0)


def coincident(A: "Beam", B: "Beam") -> bool:
    """Same centre line (a stacked pair)."""
    return parallel(A.u, B.u, 0.2) and abs(float((B.origin - A.origin) @ A.n)) < 0.5


def parallel(u: np.ndarray, v: np.ndarray, tol_deg: float = 2.0) -> bool:
    return abs(u[0] * v[1] - u[1] * v[0]) < math.sin(math.radians(tol_deg))


def fill_offsets(lo: float, hi: float, forced: list[float], pitch: float, density: float, min_gap: float,
                 forbidden: list[float] = (), forbid_gap: float = 0.0) -> list[float]:
    """Offsets in [lo, hi]: every forced offset (leg lines) plus lines filling
    each gap so the local spacing follows pitch(s) = pitch / (1 + density |s|/W)
    (density > 0: denser toward the edge; < 0: denser toward the centre).

    Forbidden offsets (lines of a parallel tier at the same height) are anchors
    of a second kind: the fill sits half a step away from them, so the new
    lines interleave with the old ones instead of colliding."""
    def ok(x):
        return all(abs(x - f) >= forbid_gap - 1e-9 for f in forbidden)

    forced = sorted(f for f in forced if ok(f) and (hi <= lo or lo - 1e-6 <= f <= hi + 1e-6))
    merged: list[float] = []
    for f in forced:
        if merged and abs(f - merged[-1]) < min_gap:
            merged[-1] = 0.5 * (merged[-1] + f)
        else:
            merged.append(f)
    if hi - lo <= min_gap:
        return merged
    W = max(abs(lo), abs(hi), 1.0)
    s = np.linspace(lo, hi, 4001)
    if density >= 0:
        loc = pitch / (1.0 + density * np.abs(s) / W)
    else:
        loc = pitch * (1.0 - density * np.abs(s) / W)
    C = np.concatenate([[0.0], np.cumsum(0.5 * (1 / loc[1:] + 1 / loc[:-1]) * np.diff(s))])
    anchors = {lo: "R", hi: "R"}
    for f in merged:
        anchors[f] = "R"
    for f in forbidden:
        if lo < f < hi:
            anchors[f] = "F"
    keys = sorted(anchors)
    out = set(merged)
    for x in (lo, hi):
        if ok(x):
            out.add(x)
    for a, b in zip(keys[:-1], keys[1:]):
        ca, cb = np.interp([a, b], s, C)
        D = cb - ca
        fa, fb = anchors[a] == "F", anchors[b] == "F"
        if not fa and not fb:
            m = int(round(D)) - 1
            pos = [ca + k * D / (m + 1) for k in range(1, m + 1)]
        elif fa and fb:
            m = max(1, int(round(D)))
            pos = [ca + (k - 0.5) * D / m for k in range(1, m + 1)]
        else:
            m = max(0, int(round(D - 0.5)))
            if fa:
                pos = [ca + (k - 0.5) * D / (m + 0.5) for k in range(1, m + 1)]
            else:
                pos = [cb - (k - 0.5) * D / (m + 0.5) for k in range(1, m + 1)]
        for c in pos:
            out.add(float(np.interp(c, C, s)))
    res: list[float] = []
    for x in sorted(out):
        if not ok(x):
            continue
        if res and x - res[-1] < min_gap:
            if x in merged and res[-1] not in merged:
                res[-1] = x
            continue
        res.append(x)
    return res


def reach(u: np.ndarray, band: tuple, beams: list, margin, legs: list, w: float,
          leg_half: float = 0.5) -> list[tuple]:
    """Offsets s for which the line s*n + t*u would rest on at least two earlier
    beams (each reached within its allowed extent, leaving `margin(A)` beyond
    the crossing).  Returns the feasible intervals."""
    n = np.array([-u[1], u[0]])
    ivs = []
    for A in beams:
        if not A.alive or min(A.z1, band[1]) - max(A.z0, band[0]) <= EPS:
            continue
        sin = abs(u[0] * A.u[1] - u[1] * A.u[0])
        if sin < 0.26:
            continue
        r0 = line_cross(np.zeros(2), u, A.origin, A.u)
        r1 = line_cross(n, u, A.origin, A.u)
        if r0 is None or r1 is None:
            continue
        tb0, slope = r0[1], r1[1] - r0[1]
        cos = abs(float(u @ A.u))
        hb = 0.5 * w / sin + 0.5 * A.w * cos / sin
        m = margin(A)
        lo_A, hi_A = A.allowed[0] + hb + m, A.allowed[1] - hb - m
        if lo_A > hi_A or abs(slope) < 1e-12:
            continue
        s1, s2 = sorted(((lo_A - tb0) / slope, (hi_A - tb0) / slope))
        ivs.append((s1, s2, 1))
    for L in legs:
        sl = float(L.xy @ n)
        ivs.append((sl - leg_half, sl + leg_half, 2))
    if not ivs:
        return []
    ev = sorted([(a, wt) for a, b, wt in ivs] + [(b, -wt) for a, b, wt in ivs], key=lambda e: (e[0], -e[1]))
    out, cur, start = [], 0, None
    for x, d in ev:
        prev = cur
        cur += d
        if prev < 2 <= cur:
            start = x
        elif prev >= 2 > cur and start is not None:
            out.append((start, x))
            start = None
    return out


def ring_lines(apothem: float, n_sides: int, phase: float):
    lines = []
    for k in range(n_sides):
        phi = phase + 2 * math.pi * (k + 0.5) / n_sides
        nn = _unit(phi)
        u = np.array([-nn[1], nn[0]])
        lines.append((apothem * nn, u, "ring", k))
    return lines


def flush_count(p: Params, T: int) -> int:
    """How many top tiers share the flat top band: one per grid direction
    (2 for a square grid, 3 for a 60-degree kagome grid), leaving at least
    one tier below to carry them."""
    if not p.flush_top or T < 3:
        return 1
    nf = max(2, int(round(180.0 / max(p.tier_turn, 1.0))))
    return min(nf, 3, T - 1)


# ---------------------------------------------------------------------------
# line sets: free (square) grids and the regular kagome lattice
# ---------------------------------------------------------------------------
def square_offsets(p, t, T, n, legs, w, band, beams, support_margin, leg_half, rlo, rhi, shaft_mode, frame,
                   part=None, all_lines=False):
    """Offsets of one tier of a free grid: forced leg lines plus a fill, either a
    smooth density field or (with a partition, see partition.py) the lines of
    the hierarchical split: a bracket tier holds the coarse levels plus its
    outermost reachable line (the bracket tip steps out), the top grid all."""
    u = np.array([n[1], -n[0]])
    stacked = [float(B.origin @ n) for B in beams
               if parallel(B.u, u) and min(B.z1, band[1]) - max(B.z0, band[0]) > EPS]
    if shaft_mode:
        leg_offs = [float(L.xy @ n) + sgn * 0.5 * frame for L in legs for sgn in (-1, 1)]
    else:
        leg_offs = [float(L.xy @ n) for L in legs]
    if t == 0:
        return fill_offsets(0.0, 0.0, leg_offs, 1.0, 0.0, w + 6.0)
    feas = [(max(a + 0.5, rlo), min(b - 0.5, rhi)) for a, b in reach(u, band, beams, support_margin, legs, w, leg_half)]
    feas = [(a, b) for a, b in feas if b - a > 1.0]
    if not feas:
        return []
    if part is not None:
        A = max(part["axes"], key=lambda A: abs(float(A["n"] @ n)))
        sg = 1.0 if float(A["n"] @ n) > 0 else -1.0
        lv = {sg * x: l for x, l in A["used"].items()}
        cand = sorted(x for x in lv if any(a <= x <= b for a, b in feas))
        if t >= T - flush_count(p, T) or all_lines:
            return cand
        # the coarse levels, and the bracket tips as far out as the tier can reach;
        # a tip that would run alongside a top strip too closely stacks under it
        # (or backs off clear of it)
        flo, fhi = min(a for a, b in feas), max(b for a, b in feas)
        gap = w + 14.0
        used = sorted(lv)

        def tip(x, sgn):
            near = [y for y in used if abs(y - x) < gap]
            if not near:
                return x
            ok = [y for y in near if sgn * (x - y) >= 0]          # not past the reach
            return (max(ok, key=lambda y: sgn * y) if ok else
                    (min(near) - gap if sgn > 0 else max(near) + gap))

        t_lo, t_hi = tip(flo, -1.0), tip(fhi, 1.0)
        keep = [x for x in cand if lv[x] <= t and t_lo + gap <= x <= t_hi - gap]
        return sorted(set(keep) | {t_lo, t_hi})
    pitch = p.pitch * (1.0 + p.pitch_taper * (T - 1 - t) / max(1, T - 1))
    flo = min(a for a, b in feas)
    fhi = max(b for a, b in feas)
    return fill_offsets(flo, fhi, leg_offs + [x for x in stacked if flo <= x <= fhi], pitch, p.density, w + 14.0)


def top_plate(p: Params, pts, fld, has_glass: bool):
    """The top as a plate (plate.py): the glass outline when there is glass
    (grown over the tails if it covers them), else the lattice's own region;
    held over each leg capital; the design load as a hand-sized patch."""
    from .plate import plate_for
    if fld is not None:
        lvl = cover_level(p) if has_glass else -p.overhang
        dom = fld.region(lvl)
    else:
        base = top_outline(p)
        dom = base.buffer(cover_level(p), quad_segs=16) if has_glass else inset(base, p.overhang)
    if dom.geom_type != "Polygon":
        dom = max(dom.geoms, key=lambda g: g.area)
    q = 1e-3 + (2.5e-5 * p.glass_t if has_glass else 0.0)       # 1 kPa + the glass
    sup = [(np.asarray(c, float), 0.5 * p.leg_cap_d) for c in pts]
    return plate_for(dom, sup, p.design_load, q, patch_r=50.0)


def cover_level(p: Params) -> float:
    """Where the glass edge sits (field level, mm past the outline): when the
    glass covers the tails, 10 mm past the outline, with the tails tucked in
    under it (tail_reach is then capped at the outline)."""
    if p.glass_cover and p.edge_mode == "eaves" and p.glass_kind != "none":
        return 10.0
    return 0.0


def tail_level(p: Params) -> float:
    """Field level the eave tails run out to."""
    return min(p.tail_reach, 0.0) if cover_level(p) > 0 else p.tail_reach


def best_orientation(p: Params, plate, leg_sym, gkind: str) -> float:
    """Grid angle with the least total bending demand (plate.total_demand)
    among the angles that keep the most of the legs' symmetry."""
    from .plate import total_demand
    from .symmetry import common_group, grid_symmetry
    period = 90.0 if gkind == "square" else 60.0
    fam = 2 if gkind == "square" else 3
    cands, best_n = [], -1
    for a in sorted(set([round(p.angle % period, 6)] + [round(x, 6) for x in np.arange(0.0, period, 1.5)])):
        n_ops = len(common_group(leg_sym, grid_symmetry(gkind, a))[0])
        if n_ops > best_n:
            cands, best_n = [a], n_ops
        elif n_ops == best_n:
            cands.append(a)
    J = {a: total_demand(plate, math.radians(a), fam) for a in cands}
    a_best = min(J, key=lambda a: J[a])
    # keep the angle as set unless turning saves a real amount (a round top on
    # one column bends the same every way; mesh noise must not turn it)
    a0 = round(p.angle % period, 6)
    if a0 in J and J[a0] <= 1.02 * J[a_best]:
        return float(p.angle)
    return float(a_best)


def hex_motifs(p: Params, T: int, kag: dict, legs: list, top_region, w: float, plate, sym_ops, notes: list):
    """Hex tree: the kagome hexagons that hold a motif (partition.motif_lines) --
    every post's own hexagon, and those where the plate's design moments call
    for strips closer than the kagome's pitch (s = a (m_ref / m)^load_refine,
    enveloped over the hexagon and its symmetry images).  Returns the top-band
    axes (one per family, in its tier's frame) and the motif strips."""
    from .partition import _hex_frame, motif_lines
    from .plate import directional_demand
    a, f = kag["pitch"], kag["motif_f"]
    theta = math.radians(p.angle)
    us, ns = _hex_frame(theta)
    nf = flush_count(p, T)
    axes = []
    for k in range(3):
        # the top-band tier whose strips run along family k
        t_k = next(t for t in range(T - nf, T) if abs(abs(float(_unit(math.radians(p.angle + t * p.tier_turn)) @ us[k])) - 1) < 1e-6)
        u = _unit(math.radians(p.angle + t_k * p.tier_turn))
        axes.append(dict(t=t_k, u=u, n=np.array([-u[1], u[0]])))
    # hexagon centres: half a pitch past a line in every family
    phA, phB = kag["phi"][0], kag["phi"][1]
    Mx = np.array([kag["ns"][0], kag["ns"][1]])
    x0, y0, x1, y1 = top_region.bounds
    span = max(x1 - x0, y1 - y0)
    nmax = int(span / a) + 3
    inner = top_region.buffer(-(0.5 * w + 6.0))
    Vr = 0.5 * a / math.cos(math.pi / 6)
    vdirs = [np.array([math.cos(theta + math.pi / 6 + i * math.pi / 3), math.sin(theta + math.pi / 6 + i * math.pi / 3)])
             for i in range(6)]
    demand = None
    if plate is not None and p.load_refine > 0:
        from shapely import contains_xy
        dem = directional_demand(plate.M, theta, 3)
        ins = contains_xy(top_region, plate.xs[:, 0], plate.xs[:, 1])
        m_ref = float(np.median(dem[ins].max(axis=1))) if ins.any() else 1.0
        demand = (plate.lookup(dem), max(m_ref, 1e-9))
    orbit = [(np.eye(2), [0, 1, 2])]
    for M in sym_ops:
        orbit.append((M, [int(np.argmax([abs(float(us[j] @ (M @ us[k]))) for j in range(3)])) for k in range(3)]))
    sample = [np.zeros(2)] + [0.55 * Vr * v for v in vdirs] + [0.3 * Vr * v for v in vdirs]
    chosen = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            c = np.linalg.solve(Mx, [a * (i + phA + 0.5), a * (j + phB + 0.5)])
            hexagon = Polygon([c + Vr * v for v in vdirs])
            leg_here = any(np.linalg.norm(L.xy - c) < 0.25 * a for L in legs)
            if not leg_here and not inner.contains(hexagon):
                continue
            refine = leg_here
            if not refine and demand is not None:
                at, m_ref = demand
                P = np.array([c + q for q in sample])
                m = np.zeros(3)
                for M, perm in orbit:
                    v = at(P @ M.T).max(axis=0)
                    m = np.maximum(m, np.array([v[perm[k]] for k in range(3)]))
                s_req = a * (m_ref / np.maximum(m, 1e-9)) ** p.load_refine
                refine = float(s_req.min()) < a / math.sqrt(2.0)
            if refine:
                chosen.append(c)
    segs = []
    for c in chosen:
        for k, o, lo, hi in motif_lines(c, a, f, us, ns):
            A = axes[k]
            sig = 1.0 if float(A["n"] @ ns[k]) > 0 else -1.0
            P0, P1 = o * ns[k] + lo * us[k], o * ns[k] + hi * us[k]
            t0, t1 = sorted((float(P0 @ A["u"]), float(P1 @ A["u"])))
            # each end laps flush over the hexagon side it meets
            segs.append(dict(k=k, s=sig * o, t0=t0 - 0.5 * w, t1=t1 + 0.5 * w, rank=1.0, level=1, rim=(False, False)))
    if chosen:
        notes.append(f"hex tree: kagome pitch {a:.0f} mm; {len(chosen)} hexagons hold a {p.hex_motif} "
                     f"({sum(1 for c in chosen if any(np.linalg.norm(L.xy - c) < 0.25 * a for L in legs))} round the posts)")
    return axes, segs


def load_tier(p: Params, t: int, T: int, n, u, cand: list, legs: list, frame: float, w: float, region,
              at_col, sym_ops=()) -> list:
    """Bracket tier t chosen by the load (a tiered system under a uniform top).
    The lines framing the posts are always there; of the others, tier t keeps
    the share t / (bracket tiers) that carry the most bending along their
    length (the plate's design moment for strips in this direction, at_col),
    each running only as far as that bending reaches (plus a pitch and a half;
    the tier right under the top runs out to its whole step).  So the lowest
    tiers hold the main load paths between and around the legs and every tier
    above adds the next most loaded lines.  Returns [(offset,
    extent along u or None)]."""
    nf = flush_count(p, T)
    Tc = max(1, T - nf)
    near = 0.5 * frame + 0.5 * w + 1.0
    framing = [x for x in cand if any(abs(x - float(L.xy @ n)) <= near for L in legs)]
    others = [x for x in cand if x not in framing]
    # operations that keep this direction: they map line x to sn x, run reversed if su < 0
    ops = []
    for M in sym_ops:
        Mn, Mu = M @ n, M @ u
        if abs(abs(float(n @ Mn)) - 1.0) < 1e-6:
            ops.append((1.0 if float(n @ Mn) > 0 else -1.0, 1.0 if float(u @ Mu) > 0 else -1.0))
    step = max(8.0, 0.5 * w)
    prof = {}
    for x in others:
        o = x * n
        ivs = clip_line_all(region, o, u)
        if not ivs:
            continue
        ts = np.concatenate([np.arange(a, b + 1e-9, step) for a, b in ivs])
        prof[x] = (ts, at_col(o[None, :] + ts[:, None] * u[None, :]))
    score = {x: float(m.max()) if len(m) else 0.0 for x, (ts, m) in prof.items()}

    def partner(x, sn):
        y = sn * x
        return min(score, key=lambda z: abs(z - y)) if score else None

    for x in list(score):                      # symmetric partners share a score
        for sn, su in ops:
            y = partner(x, sn)
            if y is not None and abs(y - sn * x) < 0.5:
                score[x] = score[y] = max(score[x], score[y])
    ranked = sorted(score.values(), reverse=True)
    k = int(math.ceil(len(ranked) * t / Tc))
    thr = ranked[k - 1] if k >= 1 else math.inf
    ext = {}
    full = t == Tc - 1          # the tier right under the top reaches as far as its step allows
    for x in others:
        if x in score and score[x] >= thr * (1.0 - 1e-9) and score[x] > 0:
            ts, m = prof[x]
            if full:
                ext[x] = (float(ts.min()) - 1.0, float(ts.max()) + 1.0)
                continue
            hot = ts[m >= 0.25 * score[x]]
            ext[x] = (float(hot.min()) - 1.5 * p.pitch, float(hot.max()) + 1.5 * p.pitch)
    for _ in range(2):                         # partners reach as far as each other
        for x in list(ext):
            for sn, su in ops:
                y = next((z for z in ext if abs(z - sn * x) < 0.5), None)
                if y is None:
                    continue
                a, b = sorted((su * ext[x][0], su * ext[x][1]))
                ext[y] = (min(ext[y][0], a), max(ext[y][1], b))
    return [(x, None) for x in framing] + [(x, ext[x]) for x in sorted(ext)]


def make_partition(p: Params, T: int, legs: list, top_region, w: float, shaft_mode: bool, frame: float,
                   depth_at, sym_ops=(), plate=None) -> dict:
    """Binary partitions of the two grid directions (partition.py).  In tree
    mode also the top grid's kd-tree: its kept cells (centre inside the top
    region; a post's shaft always) and their sides as strips, each with its
    rank, level and which of its ends face the rim."""
    from .partition import SQRT2, Axis, cell_edges, kd_tree
    nf = flush_count(p, T)
    tree = p.spacing in ("tree", "uniform") and nf == 2
    uniform = p.spacing == "uniform"
    axes = []
    for t in (T - 2, T - 1):
        u = _unit(math.radians(p.angle + t * p.tier_turn))
        n = np.array([-u[1], u[0]])
        # the span over the symmetry orbit: the traced outline is symmetric only to
        # within its sampling, and the partition must come out exactly symmetric
        ext = [extent_along(top_region, M.T @ n) for M in [np.eye(2)] + list(sym_ops)]
        lo, hi = min(e[0] for e in ext), max(e[1] for e in ext)
        if tree:
            # the tree's box reaches past the field, so the cells kept at the rim
            # (centre inside) close on real strips instead of the box's open sides
            m_box = p.overhang + 0.5 * w + 6.0
            lo, hi = lo - m_box, hi + m_box
        else:
            lo, hi = lo + 0.5 * w + 6.0, hi - 0.5 * w - 6.0
        if shaft_mode:
            anchors = [float(L.xy @ n) + sg * 0.5 * frame for L in legs for sg in (-1, 1)]
        else:
            anchors = [float(L.xy @ n) for L in legs]
        axes.append(dict(t=t, u=u, n=n, ax=Axis(lo, hi, anchors, p.pitch, 0.0 if tree else p.density, w + 14.0)))
    part = dict(axes=axes, tree=tree, segs=[])
    if not tree:
        for A in axes:
            if p.spacing == "uniform":
                ax_ = A["ax"]
                A["used"] = {x: 0 for x in fill_offsets(ax_.lo, ax_.hi, ax_.anchors, p.pitch, 0.0, w + 14.0)}
            else:
                A["used"] = A["ax"].full(1.0)
        return part
    ax, ay = axes[0]["ax"], axes[1]["ax"]
    nA, nB = axes[0]["n"], axes[1]["n"]
    leg_ab = [(float(L.xy @ nA), float(L.xy @ nB)) for L in legs]
    # how deep a point lies in the top region, averaged over its symmetry images:
    # a traced or polygonal outline is symmetric only to within its sampling,
    # and the cells kept at the rim must come out exactly symmetric
    ops_all = [np.eye(2)] + list(sym_ops)
    _depth_raw = depth_at

    def depth_at(q):
        q = np.asarray(q, float)
        return float(np.mean([_depth_raw(M @ q) for M in ops_all]))

    def plan(a, b):
        return a * nA + b * nB

    def fixed(I, J):
        return any(I[0] < a < I[1] and J[0] < b < J[1] for a, b in leg_ab)

    # the strips' spacing: a fully stressed grillage spaces strips inversely to
    # the bending moment they carry, s = pitch (m_ref / m)^load_refine, where m
    # is the plate's design moment for that family (Wood-Armer, enveloped over
    # the load cases) and m_ref its median over the top
    gap = w + 14.0
    demand = None
    if plate is not None and p.load_refine > 0:
        from shapely import contains_xy
        from .plate import directional_demand
        thA = math.radians(p.angle + (T - 2) * p.tier_turn)
        dem = directional_demand(plate.M, thA, 2)
        inside_top = contains_xy(top_region, plate.xs[:, 0], plate.xs[:, 1])
        m_ref = float(np.median(dem[inside_top].max(axis=1))) if inside_top.any() else 1.0
        demand = (plate.lookup(dem), max(m_ref, 1e-9))
    sample = (np.arange(4) + 0.5) / 4.0
    uA = axes[0]["u"]
    orbit = [(np.eye(2), False)] + [(M, abs(float(uA @ (M @ uA))) < 0.5) for M in sym_ops]

    def target(I, J):
        dep = depth_at(plan(0.5 * (I[0] + I[1]), 0.5 * (J[0] + J[1])))
        if dep < -0.5 * math.hypot(I[1] - I[0], J[1] - J[0]):
            return math.inf                 # wholly outside: never kept, no need to refine
        rim = 1.0 - dep / p.pitch
        boost = 1.0 + p.rim_refine * min(1.0, max(0.0, rim))
        if demand is None:
            return SQRT2 / boost
        at, m_ref = demand
        aa = I[0] + sample * (I[1] - I[0])
        bb = J[0] + sample * (J[1] - J[0])
        A_, B_ = np.meshgrid(aa, bb, indexing="ij")
        pts = A_.ravel()[:, None] * nA[None, :] + B_.ravel()[:, None] * nB[None, :]
        # the largest demand over the cell and over its symmetry images (the
        # traced outline and its plate mesh are symmetric only to within their
        # sampling; the tree must come out exactly symmetric)
        m = np.zeros(2)
        for M, swap in orbit:
            v = at(pts @ M.T).max(axis=0)
            m = np.maximum(m, v[::-1] if swap else v)
        out = []
        for mk in m:
            s_req = p.pitch * (m_ref / max(mk, 1e-9)) ** p.load_refine
            s_req = min(max(s_req, gap), 2.0 * p.pitch) / boost
            out.append(SQRT2 * s_req / p.pitch)
        return tuple(out)

    if uniform:
        # an even grid through the posts' framing lines: every gap between them
        # (and out to the box) divided into equal parts near the pitch
        for A in axes:
            ax_ = A["ax"]
            offs = fill_offsets(ax_.lo, ax_.hi, ax_.anchors, p.pitch, 0.0, w + 14.0)
            offs = [x for x in offs if ax_.lo + 1e-6 < x < ax_.hi - 1e-6]
            for x in offs:
                if x not in ax_.info:
                    ax_.info[x] = (1, 1.0)
            ends = [ax_.lo] + sorted(set(offs) | set(ax_.anchors)) + [ax_.hi]
            A["ivs"] = list(zip(ends[:-1], ends[1:]))
        leaves = [(I, J) for I in axes[0]["ivs"] for J in axes[1]["ivs"]]
    else:
        leaves = kd_tree(ax, ay, target, fixed)
    # a cell is kept when its centre is in the top region and all of it lies
    # under the glass (12 mm short of its edge): a coarse rim cell, or a diamond
    # of a turned grid along a straight edge, must not push strips out past it
    reach_out = p.overhang + cover_level(p) - 12.0

    def keep(I, J):
        if fixed(I, J):
            return True
        if depth_at(plan(0.5 * (I[0] + I[1]), 0.5 * (J[0] + J[1]))) <= 0.0:
            return False
        return all(depth_at(plan(x, y)) > -reach_out for x in I for y in J)

    kept = [(I, J) for I, J in leaves if keep(I, J)]
    # bucket the kept cells for point queries
    G = max(8.0, 0.5 * w)
    grid: dict = {}
    for c in kept:
        (xa, xb), (ya, yb) = c
        for i in range(int(math.floor(xa / G)), int(math.floor(xb / G)) + 1):
            for j in range(int(math.floor(ya / G)), int(math.floor(yb / G)) + 1):
                grid.setdefault((i, j), []).append(c)

    def kept_at(a, b):
        for (I, J) in grid.get((int(math.floor(a / G)), int(math.floor(b / G))), ()):
            if I[0] <= a <= I[1] and J[0] <= b <= J[1]:
                return True
        return False

    eps, dl = 1e-3, 1.0
    used = ({}, {})
    for (kind, s), ivs in cell_edges(ax, ay, kept).items():
        k = 0 if kind == "x" else 1
        A, other = axes[k], axes[1 - k]
        sig = float(other["n"] @ A["u"])                 # along-line t = sig * other offset
        lv, rk = A["ax"].info.get(s, (99, 0.0))
        for a, b in ivs:
            ends = []
            for e, out in ((a, -1.0), (b, 1.0)):
                virt = any(abs(e - v) < 1e-6 for v in other["ax"].virtual)
                q = e + out * dl
                beyond = (kept_at(s - eps, q) or kept_at(s + eps, q)) if k == 0 else \
                         (kept_at(q, s - eps) or kept_at(q, s + eps))
                # an end on a line laps flush with its far side
                ends.append((e + (0.0 if virt else out * 0.5 * w), virt or not beyond))
            (e0, r0), (e1, r1) = ends
            t0, t1 = sig * e0, sig * e1
            rim = (r0, r1)
            if t0 > t1:
                t0, t1, rim = t1, t0, (r1, r0)
            part["segs"].append(dict(k=k, s=s, t0=t0, t1=t1, rank=rk, level=lv, rim=rim))
            used[k][s] = lv
    for k in (0, 1):
        axes[k]["used"] = used[k]
    part["leaves"] = leaves
    part["kept"] = kept
    return part


def kagome_lattice(p: Params, legs: list, notes: list, pitch_range: tuple | None = None) -> dict:
    """A regular kagome (triangles and hexagons) for three directions 60 degrees
    apart.  Family k has lines x.n_k = pitch (j + phi_k).  With phi_C = phi_B -
    phi_A + 1/2 no point lies on three lines.  Hexagon centres sit where
    x.n_A and x.n_B are both half a pitch past a line; the pitch (within
    -20%..+25% of the asked pitch) and the phases are chosen so the legs land on
    hexagon centres, and each post rises through a hexagon of six strips."""
    from .symmetry import point_symmetry
    th = [math.radians(p.angle + 60.0 * k) for k in range(3)]
    ns = [np.array([-math.sin(a), math.cos(a)]) for a in th]
    xy = [np.asarray(getattr(L, "xy", L), float) for L in legs]
    n_sym, _ = point_symmetry(xy)
    centred = n_sym != 1            # symmetric legs: the lattice's 6-fold centre sits on theirs
    L0 = np.zeros(2) if centred else xy[0]
    others = xy if centred else xy[1:]
    best = None
    plo, phi_ = pitch_range if pitch_range is not None else (0.8 * p.pitch, 1.25 * p.pitch)
    # a strip's crossings with the other two families come one triangle side
    # (pitch / sqrt 3) apart and each notch runs sqrt 3 w along it: below
    # 3 w + 10 mm the notches from above and below overlap and starve the strip
    p_min = 3.0 * p.beam_w + 10.0
    if plo < p_min:
        if p.pitch < p_min:
            notes.append(f"kagome pitch kept at {p_min:.0f} mm or more so notches from above and below do not overlap")
        plo, phi_ = p_min, max(phi_, p_min * 1.25)
    # In lattice coordinates (u, v) = (x.n_0 / pitch - phi_0, x.n_1 / pitch - phi_1)
    # lines lie at whole u, whole v and, for the third family, whole v - u
    # (mitsukude: every node a three-way crossing) or v - u + 1/2 (kagome).
    # A post sits in a cell: a kagome hexagon centred at (1/2, 1/2), or a
    # mitsukude triangle with its centroid at (1/3, 2/3) or (2/3, 1/3).
    mitsu = p.hex_joint == "mitsukude"
    seats = [(1 / 3, 2 / 3), (2 / 3, 1 / 3)] if mitsu else [(0.5, 0.5)]
    # where the reference point sits: a seat, or (mitsukude, symmetric legs) a
    # node -- a mitsukude grid turns six ways only about its nodes, three ways
    # about a triangle's centroid, so even-fold legs need a node at the centre
    if mitsu and centred:
        origins = [(0.0, 0.0)] + ([] if (n_sym == 0 or n_sym % 2 == 0) else seats)
    else:
        origins = seats

    def frac(x):
        return x - round(x)

    def seat_err(uv):
        return min(frac(uv[0] - a) ** 2 + frac(uv[1] - b) ** 2 for a, b in seats)

    for o in origins:
        for pitch in np.linspace(plo, phi_, 181):
            cost = 0.0
            for q in others:
                d = [float((q - L0) @ ns[k]) / pitch for k in (0, 1)]
                cost += seat_err((o[0] + d[0], o[1] + d[1]))
            key = (round(cost, 6), abs(pitch - (plo / 0.85 if pitch_range is not None else p.pitch)))
            if best is None or key < best[0]:
                best = (key, pitch, o)
    pitch, o = float(best[1]), best[2]
    phi = [((float(L0 @ ns[k]) / pitch) - o[k]) % 1.0 for k in (0, 1)]
    phi.append((phi[1] - phi[0] + (0.0 if mitsu else 0.5)) % 1.0)
    off = []
    for q in xy:
        uv = [float(q @ ns[k]) / pitch - phi[k] for k in (0, 1)]
        off.append(math.sqrt(seat_err(uv)) * pitch)
    if max(off) > 4.0:
        cell = "triangle centroid" if mitsu else "hexagon centre"
        notes.append(f"{'mitsukude' if mitsu else 'kagome'} pitch {pitch:.1f} mm: legs sit up to {max(off):.0f} mm "
                     f"off a {cell}")
    return dict(pitch=pitch, phi=phi, ns=ns, joint=p.hex_joint, off=max(off),
                centre="node" if (mitsu and centred and o == (0.0, 0.0)) else ("seat" if centred else "free"))


def kagome_offsets(kag, t, T, n, legs, p, w, band, beams, support_margin, leg_half, rlo, rhi, every=None):
    """Lattice lines of tier t's direction within reach; corbel tiers take every
    m-th line (sparser lower down) plus the lines framing each leg's hexagon."""
    k = int(np.argmax([abs(float(n @ q)) for q in kag["ns"]]))
    sgn = 1.0 if float(n @ kag["ns"][k]) > 0 else -1.0
    pitch, ph = kag["pitch"], kag["phi"][k]
    u = np.array([n[1], -n[0]])
    if t == 0:
        lo, hi = -1e9, 1e9
    else:
        feas = [(max(a + 0.5, rlo), min(b - 0.5, rhi)) for a, b in reach(u, band, beams, support_margin, legs, w, leg_half)]
        feas = [(a, b) for a, b in feas if b - a > 1.0]
        if not feas:
            return []
        lo, hi = min(a for a, b in feas), max(b for a, b in feas)
    # lines adjacent to each leg (its hexagon's sides)
    framing = set()
    for L in legs:
        x = float(L.xy @ kag["ns"][k]) / pitch - ph
        framing.update((math.floor(x), math.floor(x) + 1))
    nf = flush_count(p, T)
    m = 1 if t >= T - nf else max(1, int(round(1.0 + p.pitch_taper * (T - nf - t) / max(1, T - nf))))
    if every is not None:
        m = every
    out = []
    if t == 0:
        js = sorted(framing)
    else:
        # s = sgn * pitch * (j + ph): the j covering [lo, hi], whichever way n points
        ja, jb = lo / (sgn * pitch) - ph, hi / (sgn * pitch) - ph
        js = range(int(math.floor(min(ja, jb))) - 1, int(math.ceil(max(ja, jb))) + 2)
    mf = kag.get("motif_f")
    for j in js:
        s = sgn * pitch * (j + ph)
        if t == 0:
            if j not in framing or mf is not None:
                continue
        elif not (lo <= s <= hi):
            continue
        if j in framing or j % m == 0:
            out.append(s)
    if mf is not None and t < T - nf:
        # hex tree: each post rises through its hexagon's motif; the motif's
        # lines carry it down through the bracket tiers to the capital
        for L in legs:
            for sg_ in (-1.0, 1.0):
                x = float(L.xy @ n) + sg_ * mf * pitch
                if t == 0 or lo <= x <= hi:
                    out.append(x)
    return sorted(set(round(x, 6) for x in out))


def tier_bands(p: Params, T: int, z_lat: float):
    """Bottom z and depth of every tier.  Adjacent tiers overlap by `overlap`;
    the top `flush_count` tiers share one band (a flat, fully notched grid)."""
    h = p.beam_h
    tier_h = [h * (1.0 + p.h_taper * ((T - 1 - t) / max(1, T - 1) - 0.5)) for t in range(T)]
    nf = flush_count(p, T)
    for t in range(T - nf, T):
        tier_h[t] = tier_h[T - 1]
    z = [0.0] * T
    for t in range(T - nf, T):
        z[t] = z_lat - tier_h[t]
    t_low = T - nf
    top = z[t_low] + min(p.overlap, 0.5) * min(tier_h[t_low], tier_h[t_low - 1]) if t_low > 0 else z_lat
    for t in range(t_low - 1, -1, -1):
        z[t] = top - tier_h[t]
        if t > 0:
            o = min(p.overlap, 0.5) * min(tier_h[t], tier_h[t - 1])
            top = z[t] + o
    return z, tier_h, nf > 1


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
def coverage(lay: "Layout") -> float:
    """Share of the top region within reach of the top grid (both directions)."""
    reg = lay.top_region
    if reg is None or reg.is_empty:
        return 0.0
    top = [b for b in lay.beams if b.top_surface]
    if not top:
        return 0.0
    pad = 0.5 * lay.p.pitch
    strips = []
    for b in top:
        if b.core is not None:
            strips.append(strip(b.origin, b.u, b.core[0] - pad, b.core[1] + pad, b.w + 2 * pad))
        else:
            strips.append(b.strip(pad))
    covered = unary_union(strips)
    return float(covered.intersection(reg).area / reg.area)


def build_layout(p: Params, structural: bool = True) -> Layout:
    """Lay out the table, adding bracket tiers (up to 3 more than asked) until
    the top grid reaches across the field and -- with `structural` -- the
    lattice carries the design load (a quick grillage run per candidate)."""
    best = None
    first_covered = None
    for extra in range(0, 4):
        T = int(p.tiers) + extra
        if T > 9:
            break
        lay = _build_layout(p, T)
        c = coverage(lay)
        covered = c >= 0.92
        if covered and first_covered is None:
            first_covered = T
        fails, util = 0, 0.0
        if structural and covered:
            from .checks import quick_structure
            fails, util = quick_structure(lay)
        key = (covered, -fails, c if not covered else -util)
        if best is None or key > best[0]:
            best = (key, c, fails, lay)
        if covered and fails == 0:
            break
    key, c, fails, lay = best
    T = len(lay.z_tiers)
    if T != int(p.tiers):
        why = ("the lattice carries the design load" if first_covered is not None and T > first_covered
               else "the grid reaches the field edge")
        lay.notes.append(f"{T} tiers used (asked {int(p.tiers)}) so {why}")
    if fails:
        lay.notes.append("no tier count up to 3 more than asked passes the structural checks")
    if c < 0.85:
        lay.notes.append(f"top grid covers {100 * c:.0f}% of the field")
    return lay


def _build_layout(p: Params, T: int) -> Layout:
    notes: list[str] = []
    T = max(2, int(T))
    w = p.beam_w
    em = end_margin(p)
    has_glass = p.glass_kind != "none"
    shaft_mode = p.leg_mode == "shaft"

    z_glass1 = p.height
    z_glass0 = p.height - p.glass_t if has_glass else p.height
    z_lat = z_glass0 - p.pad_t if has_glass else p.height
    z_tiers, tier_h, flush = tier_bands(p, T, z_lat)

    # ---- legs and the plan regions -----------------------------------------
    pts = leg_positions(p)
    from .symmetry import common_group, grid_symmetry, point_symmetry
    radii = None
    if p.top_shape == "field" and p.field_var > 0:
        rng_ = np.random.default_rng(int(p.seed) + 17)
        radii = p.field_radius * (1.0 + rng_.uniform(-1, 1, len(pts)) * p.field_var)
    leg_sym = point_symmetry(pts, radii)
    if p.top_shape != "field":
        from .symmetry import intersect_symmetry, shape_symmetry
        leg_sym = intersect_symmetry(leg_sym, shape_symmetry(top_outline(p)))
    gkind = "hex" if abs(p.tier_turn - 60.0) < 1e-6 else "square"
    gridded = p.grammar == "tokyo" and p.twist == 0 and abs(p.tier_turn - 90.0) * abs(p.tier_turn - 60.0) < 1e-6
    # the outline first: the grid's direction and density follow how the top bends
    fld = connected_field(p, pts, notes) if p.top_shape == "field" else None
    plate = None
    if gridded:
        plate = top_plate(p, pts, fld, has_glass)
        # (a hex grid's direction is pinned by its fit: the posts sit in hexagons)
        if p.orient == "load" and gkind == "square":
            a = best_orientation(p, plate, leg_sym, gkind)
            if abs(((a - p.angle + 45.0) % 90.0) - 45.0) > 1e-6:
                notes.append(f"grid turned to {a:g} deg: least bending demand among the legs' symmetric choices")
            p = p.replace(angle=a)
    if gridded and gkind == "hex" and p.twist == 0:
        # a hex grid has two directions that keep the legs' symmetry (30 deg apart);
        # take the one that seats the posts in cells (hexagons, or mitsukude triangles)
        fits = []
        for a in (p.angle, p.angle + 30.0):
            k_ = kagome_lattice(p.replace(angle=a), [np.asarray(q, float) for q in pts], [])
            fits.append((round(k_["off"], 3), abs(a - p.angle), a, k_))
        best_fit = min(fits, key=lambda f: f[:3])
        a_best, hex_fit = best_fit[2], best_fit[3]
        if abs(a_best - p.angle) > 1e-9:
            notes.append(f"hex grid turned {a_best - p.angle:+g} deg so the posts sit in cells")
            p = p.replace(angle=a_best)
    else:
        hex_fit = None
    if gridded:
        g_sym = grid_symmetry(gkind, p.angle)
        if hex_fit is not None and hex_fit["joint"] == "mitsukude" and hex_fit["centre"] == "seat":
            # a mitsukude grid centred on a triangle: three turns, mirrors on its altitudes
            a0 = math.radians(p.angle)
            g_sym = (3, [(a0 + math.pi / 2 + k * math.pi / 3 * 2) % math.pi for k in range(3)])
        sym_ops, sym_label = common_group(leg_sym, g_sym)
    else:
        sym_ops, sym_label = [], "none"
    n_l, m_l = leg_sym
    sym_legs = ("every rotation" if n_l == 0 else
                (f"D{n_l}" if m_l else f"C{n_l}") if (n_l > 1 or m_l) else "none")
    # the open shaft must clear the post by 4 mm a side, whatever was asked
    shaft = max(p.shaft, p.post_d + 8.0)
    frame = shaft + w                         # spacing of the two lines framing a post
    kag = None
    if p.grammar == "tokyo" and abs(p.tier_turn - 60.0) < 1e-6 and p.twist == 0:
        # kagome: fit the lattice to the legs first; each post's hexagon is one lattice cell
        motif_f = None
        if p.spacing == "tree" and p.hex_motif != "none":
            # hex tree: a coarse kagome whose hexagons can hold the motif, and the
            # posts framed by the motif in their own hexagons
            from .partition import MOTIF_F, motif_min_pitch
            motif_f = MOTIF_F[p.hex_motif]
            a_min = motif_min_pitch(motif_f, w)
            target = max(p.pitch * p.hex_base, a_min)
            # posts must sit at hexagon centres, which ties the pitch to their spacing:
            # search up to twice the smallest pitch that holds the motif
            kag = kagome_lattice(p, [np.asarray(q, float) for q in pts], notes,
                                 pitch_range=(max(a_min, 0.85 * target), max(1.25 * target, 2.0 * a_min)))
            kag["motif_f"] = motif_f
            frame = 2.0 * motif_f * kag["pitch"]
        else:
            kag = kagome_lattice(p, [np.asarray(q, float) for q in pts], notes)
            # the capital reaches the corners of the post's cell: a hexagon's at
            # pitch / sqrt 3, a mitsukude triangle's at 2/3 pitch
            frame = kag["pitch"] * (4.0 / 3.0 * math.cos(math.pi / 6) if kag["joint"] == "mitsukude" else 1.0)
    cap_r = 0.5 * p.leg_cap_d
    if shaft_mode:
        n_dir = int(round(180.0 / max(p.tier_turn, 1.0)))
        corner = 0.5 * frame / math.cos(math.pi / (2 * n_dir)) if n_dir > 2 else 0.5 * frame
        cap_r = max(cap_r, corner + 0.5 * w + 9.0)
    need = cap_r + em + 0.5 * w
    step = p.tier_step
    if fld is not None:
        outline = fld.region(0.0)
        # tier footprints are deeper level sets of the same field
        def region_at(inset_mm: float):
            return fld.region(-inset_mm)
        dist_to_edge = min(float(-fld.d(np.array([q[0]]), np.array([q[1]]))[0]) for q in pts)
    else:
        outline = top_outline(p)
        def region_at(inset_mm: float):
            return inset(outline, inset_mm)
        inside_all = all(outline.contains(Point(*q)) for q in pts)
        dist_to_edge = min(outline.exterior.distance(Point(*q)) for q in pts) if inside_all else -1.0
    avail = dist_to_edge - p.overhang - need
    if avail < 0 and fld is None:
        scale = 1.0
        qs = pts
        for _ in range(60):
            scale *= 0.97
            qs = [q * scale for q in pts]
            d = min(outline.exterior.distance(Point(*q)) for q in qs)
            if d - p.overhang - need >= 0 and all(outline.contains(Point(*q)) for q in qs):
                break
        pts = qs
        notes.append(f"legs pulled inward to {scale:.2f}x radius so tier 0 clears the glass edge")
        step = 0.0
    elif avail < 0:
        step = 0.0
        notes.append("field radius too small for the corbel: all tiers use the full footprint")
    else:
        # the corbel's total inset is set by the tiers asked for; extra tiers
        # (added to reach the field edge) make finer steps, not a smaller base
        step = p.tier_step * max(1, int(p.tiers) - 1) / max(1, T - 1)
        if (T - 1) * step > avail:
            step = avail / (T - 1)
            notes.append(f"corbel step reduced to {step:.0f} mm so tier 0 still spans the legs")
    regions = [region_at(p.overhang + (T - 1 - t) * step) for t in range(T)]
    top_region = regions[-1]
    # bracket tiers may always span between the legs (beams from column to
    # column); outside the legs' hull they step out along the field contours
    if len(pts) > 1:
        from shapely.geometry import MultiPoint
        hull = MultiPoint([tuple(q) for q in pts]).convex_hull.buffer(need, quad_segs=16)
        hull = hull.intersection(top_region)
        nf_ = flush_count(p, T)
        regions = [(r.union(hull) if t < T - nf_ else r) for t, r in enumerate(regions)]

    s_cap = p.leg_slot * tier_h[0]
    legs = []
    for i, q in enumerate(pts):
        r = float(np.hypot(*q))
        out = q / r if r > 1e-6 else np.zeros(2)
        lean, knobs = p.leg_lean, {}
        if p.leg_var > 0:
            # each leg its own: a seeded wobble of lean (amount and direction) and stem
            rg = np.random.default_rng(int(p.seed) * 7919 + 31 * i + 5)
            v = p.leg_var
            lean = p.leg_lean * (1.0 + 0.6 * v * rg.uniform(-1, 1)) + 25.0 * v * rg.uniform(0, 1)
            turn = math.radians(35.0 * v * rg.uniform(-1, 1))
            out = np.array([math.cos(turn) * out[0] - math.sin(turn) * out[1],
                            math.sin(turn) * out[0] + math.cos(turn) * out[1]])
            knobs = dict(leg_base_d=round(p.leg_base_d * (1.0 + 0.25 * v * rg.uniform(-1, 1)), 1),
                         leg_belly=round(p.leg_belly + 0.25 * v * rg.uniform(-1, 1), 3),
                         leg_belly_at=round(min(0.8, max(0.15, p.leg_belly_at + 0.2 * v * rg.uniform(-1, 1))), 3),
                         leg_flare_mix=round(min(1.0, max(0.0, p.leg_flare_mix + 0.3 * v * rg.uniform(-1, 1))), 3),
                         leg_foot_bulge=round(max(0.0, p.leg_foot_bulge + 0.15 * v * rg.uniform(-1, 1)), 3),
                         leg_sweep=round(p.leg_sweep + 30.0 * v * rg.uniform(-1, 1), 1),
                         leg_swirl=round(p.leg_swirl + 20.0 * v * rg.uniform(-1, 1), 1))
        foot = q + out * lean
        legs.append(Leg(i, np.array(q, float), np.array(foot, float), z_tiers[0] + s_cap, z_tiers[0], cap_r,
                        knobs=knobs))

    # in shaft mode every post is an obstacle: a line that would run over a post
    # is split there (its pieces stop clear of the post), wherever the leg is
    clip_regions = regions
    if shaft_mode:
        keep_out = unary_union([Point(*L.xy).buffer(0.5 * p.post_d + 0.5 * w + 2.0, quad_segs=16) for L in legs])
        clip_regions = [r.difference(keep_out) for r in regions]

    beams: list[Beam] = []
    counter = [0]
    # a beam rests on a capital when its strip lies within the capital
    leg_half = (cap_r - 0.5 * w - 2.0) if shaft_mode else 1.0

    def cant_of(B: Beam) -> float:
        # the top grid only carries glass near the rim: it may run out to the
        # outline offset (hard limit); corbel tiers keep the bracket proportion
        if abs(B.z1 - z_lat) < 1e-6:
            return min(p.max_cantilever, p.top_cantilever * B.h)
        return min(p.max_cantilever, p.cantilever * B.h)

    def support_margin(A: Beam) -> float:
        """Room a carrying beam keeps past a crossing it carries (the end margin;
        finalize lengthens ends to the bracket projection where there is room)."""
        return em

    def try_accept(t, origin, u, fam, tag, extent=None) -> list:
        made = []
        ivs = clip_line_all(clip_regions[t], np.asarray(origin, float), np.asarray(u, float))
        for j, (a, b) in enumerate(ivs):
            B = accept_piece(t, origin, u, fam, tag if len(ivs) == 1 else f"{tag}{chr(97 + j)}", a, b, extent)
            if B is not None:
                made.append(B)
        return made

    def accept_piece(t, origin, u, fam, tag, a, b, extent) -> Beam | None:
        if extent is not None:
            a, b = max(a, extent[0]), min(b, extent[1])
        if b - a < 2 * em + w:
            return None
        B = Beam(f"t{t}-{tag}", t, np.array(origin, float), np.array(u, float), a, b,
                 z_tiers[t], tier_h[t], w, fam, region_t=(a, b))
        B.order = 10 + 10 * t + 0.001 * counter[0]
        B.top_surface = abs(B.z1 - z_lat) < 1e-6
        B.flush_ends = kag is not None
        sup = []                      # (t on B, weight)
        for L in legs:
            if abs(float((L.xy - B.origin) @ B.n)) <= leg_half + 1e-6:
                tl = B.leg_t(L)
                if a + L.r_cap + em <= tl <= b - L.r_cap - em:
                    B.legs.append(L.idx)
                    sup.append((tl, 2))
        for A in beams:
            if not A.alive or bands_overlap(A, B) <= EPS:
                continue
            cr = crossing(B, A)
            if cr is None:
                continue
            if not inside(cr.ta, cr.ha, a, b, em):
                continue
            if not inside(cr.tb, cr.hb, A.allowed[0], A.allowed[1], support_margin(A)):
                continue
            sup.append((cr.ta, 1))
        if sum(wt for _, wt in sup) < 2:
            return None
        ts = [s for s, _ in sup]
        cant = cant_of(B)
        lo = max(a, min(ts) - cant)
        hi = min(b, max(ts) + cant)
        B.allowed = (lo, hi)
        B.t0, B.t1 = lo, hi
        counter[0] += 1
        beams.append(B)
        return B

    leg_angles = [math.atan2(L.xy[1], L.xy[0]) for L in legs]
    R_leg = float(np.mean([np.hypot(*L.xy) for L in legs]))

    part = None
    if p.grammar == "tokyo" and kag is None and p.spacing in ("split", "tree", "uniform") and p.twist == 0 \
            and abs(p.tier_turn - 90.0) < 1e-6:
        if fld is not None:
            def depth_at(q):
                return -float(fld.d(np.array([q[0]]), np.array([q[1]]))[0]) - p.overhang
        else:
            def depth_at(q):
                pt = Point(float(q[0]), float(q[1]))
                dd = top_region.boundary.distance(pt)
                return dd if top_region.contains(pt) else -dd
        part = make_partition(p, T, legs, top_region, w, shaft_mode, frame, depth_at, sym_ops, plate)
    tree = part is not None and part["tree"]

    tier_dem = None
    if plate is not None and p.tier_load and p.grammar == "tokyo":
        from .plate import directional_demand
        fam = 3 if kag is not None else 2
        _dem_cache: dict = {}

        def tier_dem(theta_t):
            """Demand lookup (plan points -> design moment) for strips along theta_t."""
            key = round(math.degrees(theta_t) % 180.0, 6)
            if key not in _dem_cache:
                if fam == 2:
                    col = directional_demand(plate.M, theta_t, 2)[:, 0]
                else:
                    th0 = math.radians(p.angle)
                    k_ = int(round(((math.degrees(theta_t) - p.angle) % 180.0) / 60.0)) % 3
                    col = directional_demand(plate.M, th0, 3)[:, k_]
                at = plate.lookup(col)
                _dem_cache[key] = at
            return _dem_cache[key]

    def place_tree(segs, axes=None, base_order=None):
        """The kd-tree's strips, coarsest first, each placed once it rests on two
        earlier parts (a leg counts as two).  The placed set is the closure of
        that rule, so it does not depend on the order and symmetric trees stay
        symmetric; the order only decides which strip lies on top at a lap."""
        import heapq
        axes = axes if axes is not None else part["axes"]
        base_order = base_order if base_order is not None else 10 + 10 * (T - 2)
        if shaft_mode:
            # every post is a keep-out: a strip that would run over one (another
            # leg's framing line, say) stops clear of it on both sides
            r_keep = 0.5 * p.post_d + 0.5 * w + 2.0
            cut = []
            for sg in segs:
                A = axes[sg["k"]]
                pieces = [(sg["t0"], sg["t1"], sg["rim"][0], sg["rim"][1])]
                for L in legs:
                    off = abs(float(L.xy @ A["n"]) - sg["s"])
                    if off >= r_keep:
                        continue
                    tl = float(L.xy @ A["u"])
                    half = math.sqrt(r_keep ** 2 - off ** 2)
                    nxt = []
                    for a0, a1, r0, r1 in pieces:
                        if a1 <= tl - half or a0 >= tl + half:
                            nxt.append((a0, a1, r0, r1))
                            continue
                        if tl - half - a0 > w:
                            nxt.append((a0, tl - half, r0, False))
                        if a1 - (tl + half) > w:
                            nxt.append((tl + half, a1, False, r1))
                    pieces = nxt
                for a0, a1, r0, r1 in pieces:
                    cut.append(dict(sg, t0=a0, t1=a1, rim=(r0, r1)))
            segs = cut
        cands = []
        for k_order, sg in enumerate(segs):
            A = axes[sg["k"]]
            t = A["t"]
            a, b = sg["t0"], sg["t1"]
            B = Beam(f"t{t}-k{k_order:03d}", t, sg["s"] * A["n"], A["u"].copy(), a, b,
                     z_tiers[t], tier_h[t], w, "parallel", region_t=(a, b))
            B.top_surface = True
            B.allowed = (a, b)
            B.rim = sg["rim"]
            B.level = sg["level"]
            B.rank = sg["rank"]
            for L in legs:
                if abs(float((L.xy - B.origin) @ B.n)) <= leg_half + 1e-6:
                    tl = B.leg_t(L)
                    if a + L.r_cap + em <= tl <= b - L.r_cap - em:
                        B.legs.append(L.idx)
            cands.append(B)
        n_c = len(cands)
        # supports along each candidate: (t, notch half-length, kind)
        sups = []
        for B in cands:
            ts = [(B.leg_t(legs[i]), L_half, True) for i in B.legs for L_half in (cap_r, cap_r)]
            for A in beams:
                if not A.alive or bands_overlap(A, B) <= EPS:
                    continue
                cr = crossing(B, A)
                if cr and inside(cr.ta, cr.ha, B.t0, B.t1, 0.0) \
                        and inside(cr.tb, cr.hb, A.allowed[0], A.allowed[1], support_margin(A)):
                    ts.append((cr.ta, cr.ha, True))
            sups.append(ts)
        nbrs = [[] for _ in range(n_c)]
        for i in range(n_c):
            Bi = cands[i]
            for j in range(i + 1, n_c):
                Bj = cands[j]
                if Bj.tier == Bi.tier:
                    continue
                cr = crossing(Bi, Bj)
                if cr and inside(cr.ta, cr.ha, Bi.t0, Bi.t1, 0.0) and inside(cr.tb, cr.hb, Bj.t0, Bj.t1, 0.0):
                    nbrs[i].append((j, cr.ta, cr.ha, cr.tb, cr.hb))
                    nbrs[j].append((i, cr.tb, cr.hb, cr.ta, cr.ha))
        # symmetry orbits of the strips: the bracket tiers below follow only the
        # operations that keep their direction, so the two top families get unequal
        # support; a whole orbit is placed or none of it
        def key_of(P0, P1):
            a = (round(float(P0[0]), 2), round(float(P0[1]), 2))
            b = (round(float(P1[0]), 2), round(float(P1[1]), 2))
            return (a, b) if a <= b else (b, a)

        ends = [(B.p(B.t0), B.p(B.t1)) for B in cands]
        index = {key_of(P0, P1): i for i, (P0, P1) in enumerate(ends)}
        orbit = list(range(n_c))

        def root(i):
            while orbit[i] != i:
                orbit[i] = orbit[orbit[i]]
                i = orbit[i]
            return i

        broken = set()
        for i, (P0, P1) in enumerate(ends):
            for M in sym_ops:
                j = index.get(key_of(M @ P0, M @ P1))
                if j is None:
                    broken.add(i)
                else:
                    orbit[root(j)] = root(i)
        members: dict = {}
        for i in range(n_c):
            members.setdefault(root(i), []).append(i)
        bad_roots = {root(i) for i in broken}
        placed = [False] * n_c
        seq = 0
        # coarsest first: of the orbits whose strips all rest on two placed parts,
        # those of the highest rank are placed together; the placed set is the
        # closure of that rule, so the order cannot break the symmetry
        while True:
            ready = [r for r, ms in members.items() if r not in bad_roots and not placed[ms[0]]
                     and all(len(sups[i]) >= 2 for i in ms)]
            if not ready:
                break
            top_rank = max(cands[r].rank for r in ready)
            batch = [i for r in ready if cands[r].rank >= top_rank * (1 - 1e-9) - 1e-9 for i in members[r]]
            for i in batch:
                B = cands[i]
                B.order = base_order + 0.01 * seq
                seq += 1
                placed[i] = True
                beams.append(B)
            for i in batch:
                B = cands[i]
                for j, ta, ha, tb, hb in nbrs[i]:
                    if not placed[j]:
                        sups[j].append((tb, hb, True))
        lost = [i for i in range(n_c) if not placed[i] and cands[i].length > p.pitch]
        if lost:
            notes.append(f"{len(lost)} tree strips had no two supports and were left out")

    if p.grammar == "tokyo":
        r_post = 0.5 * p.post_d
        for t in range(T):
            if tree and t >= T - 2:
                if t == T - 2:
                    # the kd-tree's strips, coarsest first: each rests on the strips it ends on
                    place_tree(sorted(part["segs"], key=lambda g: (-g["rank"], g["k"], abs(g["s"]), g["s"], g["t0"])))
                continue
            theta = math.radians(p.angle + t * p.tier_turn + t * p.twist)
            u = _unit(theta)
            n = np.array([-u[1], u[0]])
            lo_, hi_ = extent_along(regions[t], n)
            rlo, rhi = lo_ + 0.5 * w + 6.0, hi_ - 0.5 * w - 6.0
            band = (z_tiers[t], z_tiers[t] + tier_h[t])
            load_t = tier_dem is not None and 0 < t < T - flush_count(p, T)
            if kag is not None:
                offs = kagome_offsets(kag, t, T, n, legs, p, w, band, beams, support_margin, leg_half, rlo, rhi,
                                      every=1 if load_t else None)
            else:
                offs = square_offsets(p, t, T, n, legs, w, band, beams, support_margin, leg_half, rlo, rhi,
                                      shaft_mode, frame, part=part, all_lines=load_t)
            if load_t:
                picks = load_tier(p, t, T, n, u, offs, legs, frame, w, clip_regions[t], tier_dem(theta),
                                  sym_ops)
            else:
                picks = [(x, None) for x in offs]
            # inner lines first so they claim support before the outer ones
            for k, (s_, ext_) in enumerate(sorted(picks, key=lambda q: abs(q[0]))):
                if not try_accept(t, s_ * n, u, "parallel", f"{k:02d}", extent=ext_) and ext_ is not None:
                    try_accept(t, s_ * n, u, "parallel", f"{k:02d}")
        if kag is not None and kag.get("motif_f") is not None:
            hx_axes, hx_segs = hex_motifs(p, T, kag, legs, top_region, w, plate, sym_ops, notes)
            if hx_segs:
                place_tree(sorted(hx_segs, key=lambda g: (-g["rank"], g["k"], abs(g["s"]), g["s"], g["t0"])),
                           axes=hx_axes, base_order=10 + 10 * T)
    elif p.grammar == "radial":
        radial_grammar(p, T, legs, leg_angles, R_leg, regions, w, em, cap_r, try_accept, notes)
    else:
        raise ValueError(p.grammar)

    for B in beams:
        B.top_surface = abs(B.z1 - z_lat) < 1e-6
    finalize(p, beams, legs, em, cap_r, notes)
    if p.edge_mode in ("cells", "eaves") and not tree:
        cells_edge(p, beams, legs, top_region, em, cap_r, notes)
    if p.edge_mode == "eaves":
        eaves(p, beams, legs, fld, outline, em, cap_r, notes)
    if sym_ops:
        from .symmetry import symmetrize
        # first let short partners grow to their orbit's reach; if growing and the
        # joint rules keep trading places, finish with intersections (monotone)
        # (growing can trade places with the joint rules forever, so the last
        # rounds only intersect, and only those may end the loop)
        for rnd in range(24):
            grow = rnd < 6
            sig0 = [(b.alive, round(b.t0, 4), round(b.t1, 4), round(float(b.origin @ b.n), 4)) for b in beams]
            symmetrize(beams, sym_ops, grow=grow)
            resolve_conflicts(p, beams, legs, em, cap_r, notes, extend_rounds=0)
            if not grow and [(b.alive, round(b.t0, 4), round(b.t1, 4), round(float(b.origin @ b.n), 4))
                             for b in beams] == sig0:
                break
        else:
            notes.append("symmetry and joint rules did not settle together")
    beams = [b for b in beams if b.alive]
    beams.sort(key=lambda b: b.order)
    for L in legs:
        if not any(L.idx in B.legs and B.tier == 0 for B in beams):
            notes.append(f"leg {L.idx} has no tier-0 beam over it")

    for B in beams:
        assign_ends(p, B, beams, legs, em, cap_r)

    if p.top_shape == "field" and p.glass_fit == "lattice" and fld is not None:
        outline = hug_outline(beams, p, fld, outline, legs)
    if has_glass and cover_level(p) > 0:
        outline = cover_outline(p, fld, outline, beams)
    blocks = [] if shaft_mode else post_columns(p, legs, beams, z_lat, notes)
    if has_glass:
        cap_z0 = z_glass1 + 2.0                 # on the bushing flange
    else:
        cap_z0 = z_lat                          # straight onto the grid around the shaft
    post_top = cap_z0 + p.thread_len
    pads = bumpers(p, legs, beams, outline, z_lat, z_glass0, frame if shaft_mode else None, kag) if has_glass else []
    if sym_ops and pads:
        from .symmetry import symmetrize_points
        pads = symmetrize_points(pads, sym_ops)
        for k_, d_ in enumerate(pads):
            d_["id"] = f"pad{k_:02d}" if d_["kind"] == "pad" else d_["id"]
    return Layout(p, outline, regions, legs, beams, blocks, z_tiers, z_glass0, z_glass1,
                  post_top, cap_z0, notes, z_lat=z_lat, tier_h=tier_h, pads=pads,
                  field=fld, top_region=top_region, sym_ops=sym_ops, sym_label=sym_label, sym_legs=sym_legs)


def cover_outline(p: Params, fld, outline, beams: list[Beam]):
    """Glass grown over the tails, so no wood sticks out past it: the field
    contour just beyond the tail tips (or the outline grown by as much), and in
    any case everything on top plus a margin, closed into one smooth piece."""
    lvl = cover_level(p)
    if fld is not None and p.glass_fit != "lattice":
        g = fld.region(lvl)
    else:
        g = outline.buffer(lvl, quad_segs=24)
    top = [b.strip() for b in beams if b.alive and b.top_surface]
    if top:
        need = unary_union(top).buffer(10.0, quad_segs=8)
        if not g.buffer(1.0).contains(need):
            R = 1.5 * p.pitch
            g = unary_union([g, need]).buffer(R, quad_segs=24).buffer(-R, quad_segs=24)
    if g.geom_type != "Polygon":
        g = max(g.geoms, key=lambda q: q.area)
    from shapely.geometry.polygon import orient
    return orient(Polygon(g.exterior).simplify(0.3), 1.0)


def hug_outline(beams: list[Beam], p: Params, fld, fallback, legs=None):
    """Glass outline that follows the top grid: its footprint grown by the
    overhang and rounded by a morphological closing of about three cells, then
    eased with a small opening -- an organic edge that keeps the same distance
    to the lattice all round."""
    top = [(strip(b.origin, b.u, b.core[0], b.core[1], b.w) if b.core is not None else b.strip())
           for b in beams if b.alive and b.top_surface]
    if not top:
        return fallback
    R = 1.5 * p.pitch
    g = unary_union(top).buffer(p.overhang + R, quad_segs=32).buffer(-R, quad_segs=32)
    # every post hole keeps the glass-shop margin (two thicknesses) to the edge
    holes = [Point(*L.xy).buffer(0.5 * p.post_d + p.glass_hole_clear + 2.5 * p.glass_t + 10.0, quad_segs=24)
             for L in legs] if legs else []
    if holes:
        g = unary_union([g] + holes).buffer(10.0, quad_segs=16).buffer(-10.0, quad_segs=16)
    g = g.buffer(-0.5 * p.pitch, quad_segs=32).buffer(0.5 * p.pitch, quad_segs=32)
    if g.is_empty:
        return fallback
    if g.geom_type != "Polygon":
        g = max(g.geoms, key=lambda q: q.area)
    from shapely.geometry.polygon import orient
    return orient(Polygon(g.exterior).simplify(0.3), 1.0)


def eaves(p: Params, beams: list[Beam], legs: list[Leg], fld, outline, em: float, cap_r: float, notes: list):
    """Rafter tails.  Every top strip runs on past its last crossing like the
    rafters of a temple eave: the tips stop on a smooth contour (the field
    contour at +tail_reach, so with a positive reach they peek out beyond the
    glass), at most tail_max past the core.  Tails that would run into another
    strip are cut back clear of it, or cross it cleanly."""
    top = [b for b in beams if b.alive and b.top_surface]
    if not top:
        return
    if fld is not None:
        R = fld.region(float(tail_level(p)))
    else:
        R = outline.buffer(float(tail_level(p)), quad_segs=24)
    if R.is_empty:
        return
    for X in top:
        X.core = (X.t0, X.t1)
        mid = 0.5 * (X.t0 + X.t1)
        ivs = clip_line_all(R, X.origin, X.u)
        iv = next(((a, b) for a, b in ivs if a - 1.0 <= mid <= b + 1.0), None)
        if iv is None:
            continue
        rim = getattr(X, "rim", (True, True))
        lo = min(X.t0, max(iv[0], X.t0 - p.tail_max)) if rim[0] else X.t0
        hi = max(X.t1, min(iv[1], X.t1 + p.tail_max)) if rim[1] else X.t1
        X.allowed = (lo, hi)
        X.tail = lo < X.t0 - 1e-6 or hi > X.t1 + 1e-6
        X.t0, X.t1 = lo, hi
    resolve_conflicts(p, beams, legs, em, cap_r, notes, extend_rounds=0)


def cells_edge(p: Params, beams: list[Beam], legs: list[Leg], region, em: float, cap_r: float, notes: list):
    """Whole-cell edge for the top grid: a grid segment between two crossings
    stays only if its midpoint lies inside the field; each top beam then runs
    over its longest run of kept segments and stops flush with the far side of
    the last crossing (an end lap), so the outline steps cell by cell."""
    top = [b for b in beams if b.alive and b.top_surface]
    for X in top:
        cs = []
        for B in top:
            if B is X:
                continue
            cr = crossing(X, B)
            if cr is None:
                continue
            if B.t0 - 1e-6 <= cr.tb <= B.t1 + 1e-6 and X.t0 - cr.ha - 1.0 <= cr.ta <= X.t1 + cr.ha + 1.0:
                cs.append((cr.ta, cr.ha))
        cs.sort()
        if len(cs) < 2:
            X.alive = False
            continue
        keep = []
        for (ta, ha), (tb, hb) in zip(cs[:-1], cs[1:]):
            mid = X.p(0.5 * (ta + tb))
            keep.append(region.contains(Point(*mid)))
        best, cur, best_rng = 0, 0, None
        for i, k in enumerate(keep):
            cur = cur + 1 if k else 0
            if cur > best:
                best, best_rng = cur, (i - cur + 1, i)
        if best_rng is None:
            X.alive = False
            continue
        i0, i1 = best_rng
        X.t0 = cs[i0][0] - cs[i0][1]
        X.t1 = cs[i1 + 1][0] + cs[i1 + 1][1]
        X.legs = [i for i in X.legs if X.t0 <= X.leg_t(legs[i]) <= X.t1]
    # beams that lost their footing go; repeat until stable
    for _ in range(20):
        drop = []
        for B in [b for b in beams if b.alive and b.top_surface]:
            wt = 2 * sum(1 for i in B.legs if B.t0 + cap_r <= B.leg_t(legs[i]) <= B.t1 - cap_r)
            for A in beams:
                if A is B or not A.alive or A.order >= B.order or bands_overlap(A, B) <= EPS:
                    continue
                cr = crossing(B, A)
                if cr and B.t0 + cr.ha - 1e-6 <= cr.ta <= B.t1 - cr.ha + 1e-6 \
                        and A.t0 + cr.hb - 1e-6 <= cr.tb <= A.t1 - cr.hb + 1e-6:
                    wt += 1
            if wt < 2:
                drop.append(B)
        if not drop:
            break
        for B in drop:
            B.alive = False


# ---------------------------------------------------------------------------
# radial grammar: a ring through the legs, then pinwheels and rings
# ---------------------------------------------------------------------------
def radial_grammar(p, T, legs, leg_angles, R_leg, regions, w, em, cap_r, try_accept, notes):
    order = sorted(range(len(legs)), key=lambda i: leg_angles[i])
    n_leg = len(legs)
    h = p.beam_h
    corner_ext = max(cap_r + em, 0.9 * h)
    for k in range(n_leg):
        A = legs[order[k]].xy
        Bp = legs[order[(k + 1) % n_leg]].xy
        u = (Bp - A) / np.linalg.norm(Bp - A)
        L = float(np.linalg.norm(Bp - A))
        try_accept(0, A, u, "legring", f"r{k:02d}", extent=(-corner_ext - 40, L + corner_ext + 40))
    ring_r_prev = R_leg * math.cos(math.pi / n_leg)
    reach_prev = R_leg + p.cantilever * h
    for t in range(1, T):
        region = regions[t]
        lo, hi = extent_along(region, np.array([1.0, 0.0]))
        lo2, hi2 = extent_along(region, np.array([0.0, 1.0]))
        R_reg = min(-lo, hi, -lo2, hi2)
        if t % 2 == 1:
            m = (t - 1) // 2
            e = min(p.spin_e * R_leg * (1.0 + 0.4 * m), 0.92 * R_leg)
            count = n_leg * max(1, p.spokes_per_leg) * (2 ** m)
            sd = 1 if p.spin_dir >= 0 else -1
            base = [leg_angles[i] - sd * math.acos(min(1.0, e / max(np.hypot(*legs[i].xy), e + 1e-6)))
                    for i in order]
            per = max(1, count // n_leg)
            phis = []
            for k in range(n_leg):
                a0 = base[k]
                a1 = base[(k + 1) % n_leg]
                gap = (a1 - a0) % (2 * math.pi) or 2 * math.pi
                for j in range(per):
                    phis.append(a0 + gap * j / per)
            made = []
            for k, phi in enumerate(sorted(phis)):
                nn = _unit(phi)
                u = sd * np.array([-nn[1], nn[0]])
                made.extend(try_accept(t, e * nn, u, "pinwheel", f"p{k:02d}"))
            if made:
                reach_prev = max(float(np.hypot(*B.p(B.allowed[1]))) for B in made)
        else:
            count = n_leg * max(1, p.ring_sides_mult)
            r_lo = ring_r_prev + 0.9 * h + w
            r_hi = min(R_reg - 0.5 * w - 8.0, reach_prev - em - w)
            nr = max(1, p.rings_per_tier)
            if r_hi <= r_lo:
                notes.append(f"tier {t}: no room for a ring")
                continue
            radii = []
            for j in range(nr):
                x = (j + 1) / nr
                beta = 1.0 / (1.0 + max(p.density, 0)) if p.density >= 0 else 1 - p.density
                radii.append(r_lo + (r_hi - r_lo) * (x ** beta) if nr > 1 else 0.5 * (r_lo + r_hi))
            phase = leg_angles[order[0]]
            for j, rr in enumerate(radii):
                ph = phase - (math.pi / count if j % 2 == 0 else 0.0)
                half = rr * math.tan(math.pi / count)
                cx = max(0.9 * h, w + em)
                for (origin, u, fam, k) in ring_lines(rr, count, ph):
                    try_accept(t, origin, u, "ring", f"g{j}{k:02d}", extent=(-half - cx, half + cx))
            ring_r_prev = max(radii)


# ---------------------------------------------------------------------------
# ends and conflicts
# ---------------------------------------------------------------------------
def joints_of(B: Beam, beams: list[Beam], legs: list[Leg], em: float, cap_r: float, use_allowed=False):
    """Complete crossings on B: list of (t, half-span)."""
    out = []
    lo, hi = (B.allowed if use_allowed else (B.t0, B.t1))
    for L in legs:
        if L.idx in B.legs:
            out.append((B.leg_t(L), cap_r))
    for A in beams:
        if A is B or not A.alive or bands_overlap(A, B) <= EPS:
            continue
        cr = crossing(B, A)
        if cr is None:
            continue
        alo, ahi = (A.allowed if use_allowed else (A.t0, A.t1))
        if inside(cr.ta, cr.ha, lo, hi, end_room(B, em)) and inside(cr.tb, cr.hb, alo, ahi, end_room(A, em)):
            out.append((cr.ta, cr.ha))
    return out


def valid_end(X: Beam, end: int, limit: float, beams: list[Beam], em: float) -> float:
    """Nearest end position at or inside `limit` that does not stop inside a
    crossing zone (between touching another strip and clearing it by `em`)."""
    zones = []
    room = end_room(X, em)
    for B in beams:
        if B is X or bands_overlap(X, B) <= EPS:
            continue
        cr = crossing(X, B)
        if cr is None or not (B.t0 - cr.hb - 3.0 < cr.tb < B.t1 + cr.hb + 3.0):
            continue
        if end == 1:
            zones.append((cr.ta - cr.ha - 3.0, cr.ta + cr.ha + room))
        else:
            zones.append((cr.ta - cr.ha - room, cr.ta + cr.ha + 3.0))
    v = limit
    for _ in range(len(zones) + 1):
        moved = False
        for a, b in zones:
            if a < v < b:
                v = a if end == 1 else b
                moved = True
        if not moved:
            break
    return v


def finalize(p: Params, beams: list[Beam], legs: list[Leg], em: float, cap_r: float, notes: list):
    """Cut beams to last joint + projection, then clear touching strips."""
    # joints are judged against the reachable extents, so a beam is kept long
    # enough to carry everything that was placed on it
    cuts = []
    for B in beams:
        js = joints_of(B, beams, legs, em, cap_r, use_allowed=True)
        if not js:
            cuts.append((B, None))
            continue
        proj = max(p.projection * B.h, em)
        jlo = min(t - hs for t, hs in js)
        jhi = max(t + hs for t, hs in js)
        if B.top_surface:
            cuts.append((B, B.allowed))
        else:
            cuts.append((B, (max(B.allowed[0], jlo - proj), min(B.allowed[1], jhi + proj))))
    for B, c in cuts:
        if c is None:
            B.alive = False
        else:
            B.t0, B.t1 = c
    resolve_conflicts(p, beams, legs, em, cap_r, notes)
    for B in beams:
        if B.alive:
            B.legs = [i for i in B.legs if B.t0 <= B.leg_t(legs[i]) <= B.t1]


def merge_collinear(beams: list[Beam], gap: float = 12.0) -> int:
    """Pieces of one line in one tier that meet or overlap become one strip
    (a line split by a concave outline must not lap into itself)."""
    n = 0
    alive = [b for b in beams if b.alive]
    for i, A in enumerate(alive):
        if not A.alive:
            continue
        for B in alive[i + 1:]:
            if not B.alive or B.tier != A.tier or not coincident(A, B):
                continue
            # parameters of B's ends on A's line
            sb = sorted(float((B.p(t) - A.origin) @ A.u) for t in (B.t0, B.t1))
            if sb[0] <= A.t1 + gap and sb[1] >= A.t0 - gap:
                A.t0, A.t1 = min(A.t0, sb[0]), max(A.t1, sb[1])
                lo = min(A.allowed[0], sb[0])
                hi = max(A.allowed[1], sb[1])
                A.allowed = (lo, hi)
                A.legs = sorted(set(A.legs) | set(B.legs))
                if A.core is not None and B.core is not None:
                    cb = sorted(float((B.p(t) - A.origin) @ A.u) for t in B.core)
                    A.core = (min(A.core[0], cb[0]), max(A.core[1], cb[1]))
                B.alive = False
                n += 1
    return n


def _strips_touch(A: Beam, B: Beam, pad: float) -> bool:
    """Whether two strips grown by `pad` overlap (exact for parallel strips,
    shapely otherwise -- crossing() treats under ~15 degrees as parallel)."""
    if parallel(A.u, B.u, 0.01):
        if abs(float((B.origin - A.origin) @ A.n)) >= 0.5 * (A.w + B.w) + 2 * pad:
            return False
        sb = sorted(float((B.p(t) - A.origin) @ A.u) for t in (B.t0, B.t1))
        return sb[0] - pad < A.t1 + pad and sb[1] + pad > A.t0 - pad
    return A.strip(pad).intersects(B.strip(pad))


def resolve_conflicts(p: Params, beams: list[Beam], legs: list[Leg], em: float, cap_r: float, notes: list,
                      extend_rounds: int = 6):
    """Make every pair of touching strips either cross cleanly or clear each other."""
    # Conflicts are found on a snapshot and resolved all at once (trims beat
    # extensions), so the outcome does not depend on processing order and
    # symmetric parameters give symmetric tables.
    for it in range(80):
        merge_collinear(beams)
        alive = [b for b in beams if b.alive]
        trims: dict = {}        # (id(beam), end) -> most restrictive new value
        exts: dict = {}
        kill: set = set()
        for i, A in enumerate(alive):
            for B in alive[i + 1:]:
                if bands_overlap(A, B) <= EPS:
                    continue
                cr = crossing(A, B)
                if cr is None:
                    if not coincident(A, B) and _strips_touch(A, B, 1.5):
                        kill.add(id(B) if B.order > A.order else id(A))
                    continue
                touchA = A.t0 - cr.ha - 3.0 < cr.ta < A.t1 + cr.ha + 3.0
                touchB = B.t0 - cr.hb - 3.0 < cr.tb < B.t1 + cr.hb + 3.0
                if not (touchA and touchB):
                    continue
                okA = inside(cr.ta, cr.ha, A.t0, A.t1, end_room(A, em))
                okB = inside(cr.tb, cr.hb, B.t0, B.t1, end_room(B, em))
                if okA and okB:
                    continue
                for X, tx, hx, ok in ((A, cr.ta, cr.ha, okA), (B, cr.tb, cr.hb, okB)):
                    if ok:
                        continue
                    end = 0 if tx - X.t0 < X.t1 - tx else 1
                    key = (id(X), end)
                    room = end_room(X, em)
                    if end == 0:
                        ext, cut = tx - hx - room, valid_end(X, 0, tx + hx + 3.0, alive, em)
                        if it < extend_rounds and ext >= X.allowed[0] - 1e-6:
                            exts[key] = min(exts.get(key, X.t0), ext)
                        else:
                            trims[key] = max(trims.get(key, X.t0), cut)
                    else:
                        ext, cut = tx + hx + room, valid_end(X, 1, tx - hx - 3.0, alive, em)
                        if it < extend_rounds and ext <= X.allowed[1] + 1e-6:
                            exts[key] = max(exts.get(key, X.t1), ext)
                        else:
                            trims[key] = min(trims.get(key, X.t1), cut)
        changed = bool(kill)
        for X in alive:
            if id(X) in kill:
                X.alive = False
                continue
            for end in (0, 1):
                key = (id(X), end)
                if key in trims:
                    v = trims[key]
                elif key in exts:
                    v = exts[key]
                else:
                    continue
                old_v = X.t0 if end == 0 else X.t1
                if abs(v - old_v) > 1e-6:
                    changed = True
                if end == 0:
                    X.t0 = v
                else:
                    X.t1 = v
            if X.length < 2 * em + X.w:
                X.alive = False
        # every beam must still rest on two supports (legs count double)
        drop = []
        for B in [b for b in beams if b.alive]:
            wt = 0
            for L in legs:
                if L.idx in B.legs and B.t0 + cap_r <= B.leg_t(L) <= B.t1 - cap_r:
                    wt += 2
            for A in beams:
                if A is B or not A.alive or A.order >= B.order or bands_overlap(A, B) <= EPS:
                    continue
                cr = crossing(B, A)
                if cr and inside(cr.ta, cr.ha, B.t0, B.t1, end_room(B, em)) \
                        and inside(cr.tb, cr.hb, A.t0, A.t1, end_room(A, em)):
                    wt += 1
            if wt < 2:
                drop.append(B)
        for B in drop:
            B.alive = False
        if not changed and not drop:
            break
    else:
        notes.append("conflict resolution did not settle")


def assign_ends(p: Params, B: Beam, beams: list[Beam], legs: list[Leg], em: float, cap_r: float):
    """Each free end gets the chosen curve, as long as the projection allows."""
    js = joints_of(B, beams, legs, em, cap_r)
    lo = min((t - hs for t, hs in js), default=B.t0 + 0.5 * B.length)
    hi = max((t + hs for t, hs in js), default=B.t0 + 0.5 * B.length)
    Lc = p.end_len * B.h
    ends = []
    tail = getattr(B, "tail", False)
    for free in (lo - B.t0 - 6.0, B.t1 - hi - 6.0):
        if tail and free >= 0.6 * B.h:
            # a rafter tail: its underside curves up over the whole tail and its top
            # falls away from the glass, so the tail tapers off and carries nothing
            style = "taper" if p.end_style == "taper" else "kurikata"
            ends.append(dict(style=style, L=max(0.0, free), rise=p.tail_rise, p=p.tail_curve, drop=p.tail_drop))
            continue
        L = min(Lc, max(0.0, free))
        if p.end_style == "plain" or L < 0.35 * B.h:
            ends.append(dict(style="plain", L=0.0, rise=0.0, p=1.0))
        else:
            ends.append(dict(style=p.end_style, L=L, rise=p.end_rise, p=p.end_p))
    B.ends = ends


# ---------------------------------------------------------------------------
# post columns, masu blocks, bumpers
# ---------------------------------------------------------------------------
def post_columns(p: Params, legs, beams, z_lat, notes):
    blocks = []
    for L in legs:
        over = []
        for B in beams:
            d = abs((L.xy - B.origin) @ B.n)
            tl = B.leg_t(L)
            if d < 0.5 * B.w - 0.5 * p.post_d - 2 and B.t0 < tl < B.t1:
                over.append(B)
        intervals = sorted([(B.z0, B.z1, B) for B in over], key=lambda x: x[0])
        cur = L.z_slot
        gaps = []
        for z0, z1, B in intervals:
            if z0 > cur + 0.5:
                gaps.append((cur, z0, B))
            cur = max(cur, z1)
        if z_lat > cur + 0.5:
            gaps.append((cur, z_lat, None))
        for k, (g0, g1, above) in enumerate(gaps):
            ref = above or (intervals[-1][2] if intervals else None)
            u = ref.u if ref is not None else np.array([1.0, 0.0])
            slot = 0.25 * above.h if above is not None else 0.0
            below_tier = max([B.tier for z0, z1, B in intervals if z1 <= g0 + 0.5], default=-1)
            blocks.append(Block(f"blk{L.idx}-{k}", L.idx, L.xy.copy(), u, p.masu * p.beam_w, g0, g1 + slot,
                                above.id if above is not None else None,
                                order=10 + 10 * below_tier + 5 + 0.001 * L.idx))
        if not over:
            notes.append(f"leg {L.idx}: no beam passes over it")
    return blocks


def bumpers(p: Params, legs, beams, outline, z_lat, z_glass0, frame=None, kag=None):
    pads = []
    w = p.beam_w
    r_post = 0.5 * p.post_d
    k = 0
    placed: list[np.ndarray] = []
    if frame is None:
        for L in legs:
            pads.append(dict(id=f"wash{L.idx}", kind="washer", xy=L.xy.copy(), r=max(0.5 * w - 1.5, r_post + 4.0),
                             r_in=r_post + 0.3, z0=z_lat, z1=z_glass0, leg=L.idx))
    else:
        # the clamp path: cap -> glass -> pads on the crossings framing the shaft
        # (four round a square shaft, three on alternate corners of a hexagonal one)
        a = math.radians(p.angle)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        per_leg = {}
        if abs(p.tier_turn - 60.0) < 1e-6 and kag is not None and kag.get("joint") == "mitsukude":
            # the three nodes round each post (the corners of its triangle)
            Mx = np.array([kag["ns"][0], kag["ns"][1]])
            P_ = kag["pitch"]
            for L in legs:
                uv = np.array([float(L.xy @ kag["ns"][k]) / P_ - kag["phi"][k] for k in (0, 1)])
                base = np.floor(uv)
                f_ = uv - base
                corners = [(0, 0), (0, 1), (1, 1)] if f_[1] >= f_[0] else [(0, 0), (1, 0), (1, 1)]
                per_leg[L.idx] = [np.linalg.solve(Mx, P_ * (base + np.array(c) + np.array(kag["phi"][:2]))) - L.xy
                                  for c in corners]
            offs = None
        elif abs(p.tier_turn - 60.0) < 1e-6:
            rv = 0.5 * frame / math.cos(math.pi / 6)
            offs = [rv * np.array([math.cos(a + math.pi / 6 + i * 2 * math.pi / 3),
                                   math.sin(a + math.pi / 6 + i * 2 * math.pi / 3)]) for i in range(3)]
        else:
            offs = [0.5 * frame * (su * u + sv * v) for su in (-1, 1) for sv in (-1, 1)]
        for L in legs:
            for o_ in (offs if offs is not None else per_leg[L.idx]):
                q = L.xy + o_
                pads.append(dict(id=f"pad{k:02d}", kind="pad", xy=q, r=min(6.0, 0.5 * w - 1), r_in=0.0,
                                 z0=z_lat, z1=z_glass0, leg=L.idx))
                placed.append(q)
                k += 1
    for B in beams:
        if not B.top_surface:
            continue
        lo_, hi_ = B.core if B.core is not None else (B.t0, B.t1)
        a, b = lo_ + 30.0, hi_ - 30.0
        if b <= a:
            continue
        n_int = max(1, int(math.ceil((b - a) / 280.0)))
        for tt in np.linspace(a, b, n_int + 1):
            q = B.p(tt)
            if min(np.linalg.norm(q - L.xy) for L in legs) < 70.0:
                continue
            if placed and min(np.linalg.norm(q - r) for r in placed) < 120.0:
                continue
            if not outline.buffer(-15.0).contains(Point(*q)):
                continue
            pads.append(dict(id=f"pad{k:02d}", kind="pad", xy=q, r=5.0, r_in=0.0, z0=z_lat, z1=z_glass0,
                             leg=None))
            placed.append(q)
            k += 1
    return pads
