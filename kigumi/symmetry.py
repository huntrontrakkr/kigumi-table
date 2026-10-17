"""Symmetry: find the legs' symmetry group, keep the part a grid can honour,
and make the lattice follow it.

The legs (with their field radii) are a weighted point set about the origin.
Its symmetry group is some dihedral or cyclic group: rotations by 2*pi*k/n and
mirror lines through the origin.  A square grid can only follow rotations by
90 degrees and mirrors every 45 degrees; a kagome (hex) grid rotations by 60
degrees and mirrors every 30 degrees.  The group the table follows is the
common part.  The lattice is then made exactly invariant: a strip survives
only if every image of it is also a strip, and its extent becomes the
intersection of its images' extents (in plan; the bracket tiers below may sit
at different heights, so they are matched in plan only).
"""
from __future__ import annotations

import math
from math import gcd

import numpy as np


def rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def refl(phi: float) -> np.ndarray:
    """Reflection across the line through the origin at angle phi."""
    c, s = math.cos(2 * phi), math.sin(2 * phi)
    return np.array([[c, s], [s, -c]])


def _maps_onto(M: np.ndarray, pts: np.ndarray, w: np.ndarray, tol: float) -> bool:
    for p, wp in zip(pts, w):
        q = M @ p
        d = np.linalg.norm(pts - q, axis=1)
        j = int(np.argmin(d))
        if d[j] > tol or abs(w[j] - wp) > tol:
            return False
    return True


def point_symmetry(pts, weights=None, tol: float = 1.0, max_n: int = 12):
    """(n, mirror angles) of a weighted point set about the origin.
    n = 0 means every rotation (a single point at the origin)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    w = np.zeros(len(pts)) if weights is None else np.asarray(weights, float)
    if len(pts) == 1 and np.linalg.norm(pts[0]) < tol:
        return 0, None
    n = 1
    for k in range(max_n, 1, -1):
        if _maps_onto(rot(2 * math.pi / k), pts, w, tol):
            n = k
            break
    # mirror candidates: lines through the points and through midpoints of pairs
    cands = set()
    for i, p in enumerate(pts):
        if np.linalg.norm(p) > tol:
            cands.add(round(math.atan2(p[1], p[0]) % math.pi, 6))
        for q in pts[i + 1:]:
            m = 0.5 * (p + q)
            if np.linalg.norm(m) > tol:
                cands.add(round(math.atan2(m[1], m[0]) % math.pi, 6))
            dpq = q - p
            if np.linalg.norm(dpq) > tol:
                cands.add(round((math.atan2(dpq[1], dpq[0]) + math.pi / 2) % math.pi, 6))
    mirrors = sorted(a for a in cands if _maps_onto(refl(a), pts, w, tol))
    return n, mirrors


def grid_symmetry(kind: str, angle_deg: float):
    """Rotation order and mirror angles a lattice can follow about the origin."""
    a = math.radians(angle_deg)
    if kind == "hex":
        return 6, [(a + k * math.pi / 6) % math.pi for k in range(6)]
    return 4, [(a + k * math.pi / 4) % math.pi for k in range(4)]


def common_group(leg_sym, grid_sym, tol_deg: float = 0.05):
    """The operations (2x2 matrices, identity excluded) both can follow, and a label."""
    n_legs, m_legs = leg_sym
    n_grid, m_grid = grid_sym
    n = n_grid if n_legs == 0 else gcd(n_legs, n_grid)
    if m_legs is None:
        mirrors = list(m_grid)
    else:
        mirrors = [g for g in m_grid
                   if any(abs(((g - l + math.pi / 2) % math.pi) - math.pi / 2) < math.radians(tol_deg) for l in m_legs)]
    # keep the mirror set closed under the rotations
    ops = [rot(2 * math.pi * k / n) for k in range(1, n)]
    ops += [refl(phi) for phi in mirrors]
    label = (f"D{n}" if mirrors else f"C{n}") if (n > 1 or mirrors) else "none"
    if mirrors and n == 1:
        label = "D1 (one mirror)"
    return ops, label


def _class(B):
    return "top" if B.top_surface else "corbel"


def symmetrize(beams, ops, tol: float = 0.75, grow: bool = True) -> bool:
    """Make the beam set invariant under `ops` (in plan).  Partners' extents are
    combined: with grow=True each gets the union of the orbit's extents, capped
    by what every partner may reach (its allowed range); with grow=False the
    intersection (always converges).  Returns True if anything changed."""
    if not ops:
        return False
    alive = [b for b in beams if b.alive]
    plan = []
    for B in alive:
        P0, P1 = B.p(B.t0), B.p(B.t1)
        C0, C1 = (B.p(B.core[0]), B.p(B.core[1])) if B.core is not None else (None, None)
        plan.append((B, P0, P1, C0, C1))
    new = {}
    shift = {}
    kill = set()
    for B, P0, P1, C0, C1 in plan:
        t0, t1 = B.t0, B.t1
        c0, c1 = B.core if B.core is not None else (None, None)
        u0, u1 = B.t0, B.t1             # union of the orbit's extents
        a0, a1 = B.allowed              # what every partner may reach
        offs = []                       # partners' offsets mapped back onto B's normal
        for M in ops:
            u2 = M @ B.u
            o2 = M @ B.origin
            # a bracket tier holds one direction at one height: it can only follow
            # the operations that map that direction onto itself (its own tier
            # must hold the image); the flat top band holds every direction
            if not B.top_surface and abs(B.u[0] * u2[1] - B.u[1] * u2[0]) > 1e-3:
                continue
            Q0, Q1 = M @ P0, M @ P1
            best, best_ov = None, 0.0
            for C, *_ in plan:
                if _class(C) != _class(B) or abs(C.z0 - B.z0) > 1e-6:
                    continue
                if not B.top_surface and C.tier != B.tier:
                    continue
                if abs(C.u[0] * u2[1] - C.u[1] * u2[0]) > 1e-3:
                    continue
                if abs(float((o2 - C.origin) @ C.n)) > tol:
                    continue
                tq = sorted((float((Q0 - C.origin) @ C.u), float((Q1 - C.origin) @ C.u)))
                ov = min(C.t1, tq[1]) - max(C.t0, tq[0])
                if ov > best_ov:
                    best, best_ov = C, ov
            if best is None:
                kill.add(id(B))
                break
            Minv = np.linalg.inv(M)
            offs.append(float((Minv @ best.origin - B.origin) @ B.n))
            back = sorted(float((Minv @ best.p(t) - B.origin) @ B.u) for t in (best.t0, best.t1))
            t0, t1 = max(t0, back[0]), min(t1, back[1])
            u0, u1 = min(u0, back[0]), max(u1, back[1])
            ab = sorted(float((Minv @ best.p(t) - B.origin) @ B.u) for t in best.allowed)
            a0, a1 = max(a0, ab[0]), min(a1, ab[1])
            if c0 is not None and best.core is not None:
                bc = sorted(float((Minv @ best.p(t) - B.origin) @ B.u) for t in best.core)
                c0, c1 = max(c0, bc[0]), min(c1, bc[1])
        if grow:
            # the orbit's union, as far as every member can reach (else the intersection)
            g0, g1 = max(u0, a0), min(u1, a1)
            if g1 - g0 >= t1 - t0 and g0 <= t0 + 1e-6 and g1 >= t1 - 1e-6:
                t0, t1 = g0, g1
        new[id(B)] = (t0, t1, c0, c1)
        # the orbit's lines are made to coincide exactly: average position
        shift[id(B)] = float(np.mean(offs + [0.0])) if offs else 0.0
    changed = False
    for B, *_ in plan:
        if id(B) in kill:
            B.alive = False
            changed = True
            continue
        if abs(shift.get(id(B), 0.0)) > 1e-6:
            B.origin = B.origin + shift[id(B)] * B.n
            changed = True
        t0, t1, c0, c1 = new[id(B)]
        if t1 - t0 < 1.0:
            B.alive = False
            changed = True
            continue
        if abs(t0 - B.t0) > 1e-6 or abs(t1 - B.t1) > 1e-6:
            changed = True
        B.t0, B.t1 = t0, t1
        if c0 is not None:
            B.core = (max(c0, t0), min(c1, t1))
    return changed


def symmetrize_points(items, ops, key=lambda d: d["xy"], min_gap: float = 20.0):
    """Union of the orbits of point-like items (pads), without near-duplicates."""
    out = []
    for d in items:
        for M in [np.eye(2)] + list(ops):
            q = M @ np.asarray(key(d), float)
            if all(np.linalg.norm(q - np.asarray(key(e), float)) > min_gap for e in out):
                e = dict(d)
                e["xy"] = q
                out.append(e)
    return out


def _mismatch(polys, ops) -> float:
    from shapely.affinity import affine_transform
    from shapely.ops import unary_union
    if not polys or not ops:
        return 0.0
    U = unary_union(polys)
    worst = 0.0
    for M in ops:
        V = affine_transform(U, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], 0.0, 0.0])
        worst = max(worst, U.symmetric_difference(V).area)
    return worst


def plan_mismatch(beams, ops) -> dict:
    """Certificate: the largest area (mm^2) by which a layer's plan differs from
    its image -- the flat top band under every operation, each bracket tier
    under the operations that map its direction onto itself."""
    from .shapes import strip
    alive = [b for b in beams if b.alive]
    top = [strip(b.origin, b.u, b.t0, b.t1, b.w) for b in alive if b.top_surface]
    out = dict(top=_mismatch(top, ops), tiers=0.0)
    for t in sorted(set(b.tier for b in alive if not b.top_surface)):
        tb = [b for b in alive if b.tier == t and not b.top_surface]
        u = tb[0].u
        stab = [M for M in ops if abs(u[0] * (M @ u)[1] - u[1] * (M @ u)[0]) < 1e-3]
        out["tiers"] = max(out["tiers"], _mismatch([strip(b.origin, b.u, b.t0, b.t1, b.w) for b in tb], stab))
    return out


def shape_symmetry(poly, max_n: int = 12, tol: float = 0.5):
    """(n, mirror angles) of a convex-ish outline about the origin, from its
    support function h(phi) = max over the outline of x.(cos phi, sin phi)."""
    V = np.asarray(poly.exterior.coords)[:-1]
    phis = np.linspace(0.0, 2 * math.pi, 1440, endpoint=False)
    h = (V @ np.stack([np.cos(phis), np.sin(phis)])).max(axis=0)

    def at(ph):
        return np.interp(np.mod(ph, 2 * math.pi), phis, h, period=2 * math.pi)

    if h.max() - h.min() < tol:
        return 0, None                      # round: every rotation and mirror
    n = 1
    for k in range(max_n, 1, -1):
        if np.abs(at(phis + 2 * math.pi / k) - h).max() < tol:
            n = k
            break
    mirrors = []
    for a in np.arange(0.0, math.pi, math.pi / 720):
        if np.abs(at(2 * a - phis) - h).max() < tol:
            if not mirrors or a - mirrors[-1] > math.radians(1.0):
                mirrors.append(float(a))
    return n, mirrors


def intersect_symmetry(s1, s2, tol_deg: float = 0.5):
    """The symmetry two figures share: (n, mirrors) of each -> of both."""
    n1, m1 = s1
    n2, m2 = s2
    n = n2 if n1 == 0 else (n1 if n2 == 0 else gcd(n1, n2))
    if m1 is None:
        mirrors = m2
    elif m2 is None:
        mirrors = m1
    else:
        mirrors = [a for a in m1
                   if any(abs(((a - b + math.pi / 2) % math.pi) - math.pi / 2) < math.radians(tol_deg) for b in m2)]
    return n, mirrors
