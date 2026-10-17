"""Stance: where the feet stand so the table does not tip.

The tipping check (checks.check_stability) presses down on the rim.  The table
tips over an edge of its feet's support polygon when the push's moment about
that edge beats the weight's: P d_out > W d_com, with d_out how far the rim
point reaches past the edge and d_com how far the centre of mass sits inside
it.  support_tip() finds the smallest such push; the check and the splay both
use it.

With auto splay (Params.auto_splay) the feet lean out just far enough that the
design push, TIP_LOAD, cannot tip the table:

* the weight and its centre come from the layout (glass, strips, blocks, legs,
  posts, caps) before any solid is made (weight());
* the weakest edge is pushed outward by what it lacks, (P d_out - W d_com) /
  (W + P): its two feet move along the edge's outward normal;
* that move is averaged over the layout's symmetry group, so every image of
  the edge moves too and a symmetric table stays symmetric (a leg on a mirror
  line only moves along it);
* each foot is then projected back into what it may reach: MAX_SPLAY_DEG from
  vertical, its footprint under the glass outline.  A foot at a limit slides
  along it, so the edge can still gain.

That repeats until the weakest edge holds.  The splay is the mechanism; the
check, measuring the real solids, is the certificate."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import Point

TIP_LOAD = 600.0          # N pressed down on the rim: the tipping check's pass mark
MAX_SPLAY_DEG = 18.0      # auto splay leans a leg at most this far from vertical
MARGIN = 1.03             # the layout's weight is an estimate: aim a little above TIP_LOAD


def support_tip(feet, radii, ring, W: float, com):
    """The smallest downward push on the rim that tips the table.

    feet: foot centres (n x 2), radii: the flat contact circle of each foot,
    ring: rim points (m x 2), W: weight (N), com: centre of mass (x, y).
    Returns dict(P, legs=(i, j), d_out, d_com, normal, at) for the weakest
    edge of the support polygon (P = inf when the rim never reaches past it),
    or None when the feet enclose no area."""
    from scipy.spatial import ConvexHull, QhullError
    pts, owner = [], []
    for k, (f, r) in enumerate(zip(np.asarray(feet, float), radii)):
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            pts.append(f + r * np.array([math.cos(a), math.sin(a)]))
            owner.append(k)
    pts = np.array(pts)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    vs = hull.vertices                          # counter-clockwise in 2D
    ring = np.asarray(ring, float)
    com = np.asarray(com, float)
    best = dict(P=math.inf, legs=None, d_out=0.0, d_com=0.0, normal=None, at=None)
    for k in range(len(vs)):
        a, b = pts[vs[k]], pts[vs[(k + 1) % len(vs)]]
        e = b - a
        L = float(np.hypot(*e))
        if L < 1e-9:
            continue
        n = np.array([e[1], -e[0]]) / L                 # outward for a ccw hull
        d_com = -float((com - a) @ n)
        reach = (ring - a) @ n
        j = int(np.argmax(reach))
        d_out = float(reach[j])
        if d_out <= 1.0:
            continue
        P = W * max(d_com, 0.0) / d_out
        if P < best["P"]:
            best = dict(P=P, legs=(owner[vs[k]], owner[vs[(k + 1) % len(vs)]]), d_out=d_out, d_com=d_com,
                        normal=n, at=ring[j])
    return best


def weight(lay, feet=None) -> tuple[float, np.ndarray]:
    """Weight (N) and centre of mass (x, y) of the table from its layout: the
    glass, the strips and blocks as plain prisms (notches ignored), the legs
    from their turned profiles along their spines, posts and caps."""
    from .model import leg_spine
    from .profiles import leg_stations
    from .report import DENSITY
    p = lay.p
    m, mxy = 0.0, np.zeros(2)

    def add(kg, xy):
        nonlocal m, mxy
        m += kg
        mxy = mxy + kg * np.asarray(xy, float)

    if p.glass_kind != "none":
        r_hole = 0.5 * p.post_d + p.glass_hole_clear
        area = lay.outline.area - len(lay.legs) * math.pi * r_hole ** 2
        c = lay.outline.centroid
        add(area * (lay.z_glass1 - lay.z_glass0) * 1e-9 * DENSITY.get(p.glass_kind, 2500), (c.x, c.y))
    rho_b = DENSITY.get(p.wood_beams, 600) * 1e-9
    strips = [B.strip() for B in lay.beams]
    for B, s in zip(lay.beams, strips):
        add(s.area * (B.z1 - B.z0) * rho_b, (s.centroid.x, s.centroid.y))
    # where two strips cross, the joint leaves the shared block of wood once
    # (half-laps, cogged laps): take the double count off
    from shapely import STRtree
    tree = STRtree(strips)
    for i, j in zip(*tree.query(strips, predicate="intersects")):
        if i >= j:
            continue
        Bi, Bj = lay.beams[i], lay.beams[j]
        dz = min(Bi.z1, Bj.z1) - max(Bi.z0, Bj.z0)
        if dz <= 0:
            continue
        x = strips[i].intersection(strips[j])
        if x.area > 0:
            add(-x.area * dz * rho_b, (x.centroid.x, x.centroid.y))
    for k in getattr(lay, "blocks", []) or []:
        add(k.size ** 2 * (k.z1 - k.z0) * rho_b, k.xy)
    rho_l = DENSITY.get(p.wood_legs, 600) * 1e-9
    rho_c = DENSITY.get(p.wood_caps, 600) * 1e-9
    r_post = 0.5 * p.post_d
    for i, L in enumerate(lay.legs):
        pL = p.replace(**L.knobs) if getattr(L, "knobs", None) else p
        st = leg_stations(pL, L.z_top, L.z_slot, L.r_cap)
        Lf = L if feet is None else replace(L, foot_xy=np.asarray(feet[i], float))
        spine = leg_spine(Lf, st, pL.leg_sweep, pL.leg_swirl, pL.leg_splay)
        z, r = np.asarray(st["z"], float), np.asarray(st["r"], float)
        dz = np.diff(z)
        area = math.pi * 0.5 * (r[:-1] ** 2 + r[1:] ** 2)
        kg = area * dz * rho_l
        mid = 0.5 * (spine[:-1] + spine[1:])
        add(float(kg.sum()), (kg[:, None] * mid).sum(0) / max(float(kg.sum()), 1e-12))
        rho_p = (DENSITY.get(p.post_wood, 600) if p.post_mode == "dowel" else DENSITY.get(p.wood_legs, 600)) * 1e-9
        add(math.pi * r_post ** 2 * max(lay.post_top - L.z_slot, 0.0) * rho_p, L.xy)
        add(0.7 * math.pi * (0.5 * p.cap_d) ** 2 * p.cap_h * rho_c, L.xy)       # caps are domed: ~0.7 of a cylinder
    return 9.81 * m, (mxy / m if m > 0 else np.zeros(2))


def _group(lay):
    """The layout's symmetry as (2x2 op, leg permutation) pairs, identity
    included: op @ legs[i].xy is legs[perm[i]].xy."""
    xy = np.array([L.xy for L in lay.legs], float)
    out = [(np.eye(2), np.arange(len(xy)))]
    for M in lay.sym_ops or []:
        M = np.asarray(M, float)
        img = xy @ M.T
        perm = np.array([int(np.argmin(np.hypot(*(xy - img[i]).T))) for i in range(len(xy))])
        if all(math.hypot(*(xy[perm[i]] - img[i])) < 1.0 for i in range(len(xy))):
            out.append((M, perm))
    return out


def _symmetrize(vecs, group):
    """Average a per-leg vector field over the group (positions or moves)."""
    acc = np.zeros_like(vecs)
    for M, perm in group:
        acc[perm] += vecs @ M.T
    return acc / len(group)


def splay_feet(lay, target: float = TIP_LOAD, max_deg: float = MAX_SPLAY_DEG) -> dict:
    """Lean the feet out until a `target` N push on the rim cannot tip the
    table (see the module notes).  Moves Leg.foot_xy in place; returns what it
    did, and notes it on the layout."""
    from .profiles import leg_stations
    p = lay.p
    legs = lay.legs
    n = len(legs)
    if n < 2:
        return dict(done=False, why="one leg: no support polygon to widen")
    q = np.array([L.xy for L in legs], float)
    foot0 = np.array([L.foot_xy for L in legs], float)
    radii, lim, inner = [], [], []
    for L in legs:
        pL = p.replace(**L.knobs) if getattr(L, "knobs", None) else p
        st = leg_stations(pL, L.z_top, L.z_slot, L.r_cap)
        radii.append(float(st["r"][0]))
        # a foot keeps at least the lean it was given; the limit only caps what is added
        lim.append(max(L.z_top * math.tan(math.radians(max_deg)), float(np.hypot(*(L.foot_xy - L.xy)))))
        inner.append(lay.outline.buffer(-(float(st["r_foot"]) + 5.0)))
    group = _group(lay)
    # a foot already past the glass-edge limit (its own lean put it there) stays put
    frozen = [poly.is_empty or not poly.contains(Point(*f)) for poly, f in zip(inner, foot0)]

    def reach(i, f):
        """The point nearest f that foot i may stand on (alternating projections
        onto its lean disc and the glass outline shrunk by its footprint)."""
        if frozen[i]:
            return foot0[i].copy()
        g = np.array(f, float)
        for _ in range(8):
            v = g - q[i]
            r = float(np.hypot(*v))
            if r > lim[i]:
                g = q[i] + v * (lim[i] / r)
            poly = inner[i]
            if not poly.contains(Point(*g)):
                ext = poly.exterior if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda x: x.area).exterior
                c = ext.interpolate(ext.project(Point(*g)))
                g = np.array([c.x, c.y])
            if float(np.hypot(*(g - q[i]))) <= lim[i] + 1e-6:
                break
        return g

    ring = np.asarray(lay.outline.exterior.coords)[::2]
    feet = foot0.copy()
    best, last_P, stall = None, -1.0, 0
    for _ in range(80):
        W, com = weight(lay, feet)
        best = support_tip(feet, radii, ring, W, com)
        if best is None or best["P"] >= target * MARGIN:
            break
        stall = stall + 1 if best["P"] <= last_P + 0.5 else 0
        if stall >= 3:
            break
        last_P = max(last_P, best["P"])
        need = (target * MARGIN * best["d_out"] - W * best["d_com"]) / (W + target * MARGIN) + 0.5
        D = np.zeros((n, 2))
        for i in set(best["legs"]):
            D[i] += best["normal"]
        D = _symmetrize(D, group)
        gain = min(float(D[i] @ best["normal"]) for i in set(best["legs"]))
        if gain < 1e-3:
            break
        D *= need / gain
        new = np.array([reach(i, feet[i] + D[i]) for i in range(n)])
        new = _symmetrize(new, group)
        if float(np.abs(new - feet).max()) < 0.05:
            break
        feet = new
    for L, f in zip(legs, feet):
        L.foot_xy = np.array(f, float)
    moved = np.hypot(*(feet - foot0).T)
    out = dict(done=best is not None and best["P"] >= target, P=None if best is None else best["P"],
               moved_mm=float(moved.max()),
               max_deg=max(math.degrees(math.atan2(float(np.hypot(*(f - L.xy))), L.z_top)) for L, f in zip(legs, feet)))
    if out["moved_mm"] > 0.5:
        if out["done"]:
            lay.notes.append(f"auto splay: feet moved out up to {out['moved_mm']:.0f} mm (legs at most "
                             f"{out['max_deg']:.1f} deg from vertical) so a {target:.0f} N push on the rim cannot tip it")
        else:
            lay.notes.append(f"auto splay: feet out to their limits ({max_deg:.0f} deg, the glass edge); "
                             f"a {out['P']:.0f} N push on the rim still tips it")
    elif not out["done"] and best is not None:
        lay.notes.append(f"auto splay could not widen the stance (feet at the glass edge or the splay limit); "
                         f"a {out['P']:.0f} N push on the rim tips it")
    return out
