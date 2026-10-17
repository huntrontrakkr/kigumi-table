"""Hierarchical partitions of the plan (kd-tree / Haar-wavelet style).

Each grid direction gets a 1D binary partition.  The lines framing the posts
are its roots; every interval is split at its midpoint in *mass* coordinates,
where the mass density is 1/pitch(s), so with density > 0 the splits crowd
toward the rim.  A line's level is its depth in that tree and its rank is the
mass (in pitches) of the interval it split; roots rank infinite.

* split mode (1D): every line spans the field; a bracket tier holds the coarse
  levels only and the flat top grid holds them all.
* tree mode (2D): the top grid is a kd-tree of cells.  A cell is split across
  its longest side (across both when nearly square) while it is larger than the
  local target, and the target shrinks near the rim (wavelet detail lives at
  edges).  A split line spans only its own cell and ends on the cell's sides.
  Those sides always rank higher than the new line (a cell's aspect stays
  under 2), so ordering the strips by rank makes every strip rest on the
  strips it ends on.

Mirror- and rotation-symmetric inputs give exactly symmetric partitions:
midpoints, the longest-side rule and the rim test all commute with the
symmetries of the legs.
"""
from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)


class Axis:
    """Binary partition of the offsets [lo, hi] along one grid normal."""

    def __init__(self, lo: float, hi: float, anchors, pitch: float, density: float, min_gap: float):
        self.lo, self.hi = float(lo), float(hi)
        self.pitch, self.dens = float(pitch), float(density)
        self.W = max(abs(self.lo), abs(self.hi), 1.0)
        self.min_gap = float(min_gap)
        a: list[float] = []
        for x in sorted(float(v) for v in anchors):
            if self.lo + 0.5 * min_gap <= x <= self.hi - 0.5 * min_gap and (not a or x - a[-1] > 1e-6):
                a.append(x)
        self.anchors = a
        ends = [self.lo] + a + [self.hi]
        self.roots = [(ends[i], ends[i + 1]) for i in range(len(ends) - 1)]
        self.virtual = (self.lo, self.hi)           # span ends: not lines
        self.depth = {r: 0 for r in self.roots}     # interval -> depth
        self.info = {x: (0, math.inf) for x in a}   # offset -> (level, rank)
        self._split: dict = {}

    # mass coordinate F(x) = integral of 1/pitch(s) ds from 0 (odd in x)
    def F(self, x: float) -> float:
        X, d, W, P = abs(x), self.dens, self.W, self.pitch
        if d > 1e-9:
            m = (X + d * X * X / (2.0 * W)) / P
        elif d < -1e-9:
            m = W / (P * -d) * math.log1p(-d * X / W)
        else:
            m = X / P
        return math.copysign(m, x)

    def Finv(self, y: float) -> float:
        Y, d, W, P = abs(y), self.dens, self.W, self.pitch
        if d > 1e-9:
            X = (math.sqrt(1.0 + 2.0 * d * Y * P / W) - 1.0) * W / d
        elif d < -1e-9:
            X = math.expm1(Y * P * -d / W) * W / -d
        else:
            X = Y * P
        return math.copysign(X, y)

    def mass(self, a: float, b: float) -> float:
        return self.F(b) - self.F(a)

    def split(self, a: float, b: float):
        """Mass midpoint of [a, b] and the two halves, or None when a half
        would be narrower than the minimum gap."""
        key = (a, b)
        if key not in self._split:
            m = self.Finv(0.5 * (self.F(a) + self.F(b)))
            if abs(m) < 1e-9:
                m = 0.0
            res = None
            if min(m - a, b - m) >= self.min_gap:
                dep = self.depth.get(key, 0)
                lv = dep + 1
                if m not in self.info or self.info[m][0] > lv:
                    self.info[m] = (lv, self.mass(a, b))
                self.depth[(a, m)] = dep + 1
                self.depth[(m, b)] = dep + 1
                res = (m, (a, m), (m, b))
            self._split[key] = res
        return self._split[key]

    def full(self, thr=1.0) -> dict:
        """1D partition: every interval split while its mass exceeds thr(x)
        (thr may be a function of the interval's midpoint).  {offset: level}."""
        out = {x: 0 for x in self.anchors}
        stack = list(self.roots)
        while stack:
            a, b = stack.pop()
            t = thr(0.5 * (a + b)) if callable(thr) else thr
            if self.mass(a, b) <= t:
                continue
            s = self.split(a, b)
            if s is None:
                continue
            m, I0, I1 = s
            out[m] = self.info[m][0]
            stack += [I0, I1]
        return out


def kd_tree(ax: Axis, ay: Axis, target, fixed=None, max_cells: int = 20000):
    """Leaves of the 2D kd-tree over the root cells of ax x ay.

    target(I, J) -> mass threshold, or one per axis (tx, ty): a side is split
    while its mass exceeds its threshold.
    fixed(I, J) -> True for cells that must stay whole (a post's shaft).
    Cells are (I, J) = ((xa, xb), (ya, yb)) in axis offsets.  The longest side
    (in mass) is split, both when the cell is nearly square; a side that is
    too narrow to split leaves the other free to split (it never becomes
    splittable later, so cells stay under 2:1 wherever lines are made)."""
    leaves = []
    stack = [(I, J) for I in ax.roots for J in ay.roots]
    while stack:
        if len(leaves) + len(stack) > max_cells:
            leaves.extend(stack)
            break
        I, J = stack.pop()
        if fixed is not None and fixed(I, J):
            leaves.append((I, J))
            continue
        thr = target(I, J)
        tx, ty = thr if isinstance(thr, tuple) else (thr, thr)
        mx, my = ax.mass(*I), ay.mass(*J)
        sx = ax.split(*I) if mx > tx else None
        sy = ay.split(*J) if my > ty else None
        if sx is not None and sy is not None:
            # the side that overshoots its target most goes first; both when close
            r = (mx / tx) / (my / ty)
            if r > 1.25:
                sy = None
            elif r < 0.8:
                sx = None
        if sx is None and sy is None:
            leaves.append((I, J))
        elif sx is not None and sy is not None:
            stack += [(I2, J2) for I2 in sx[1:] for J2 in sy[1:]]
        elif sx is not None:
            stack += [(I2, J) for I2 in sx[1:]]
        else:
            stack += [(I, J2) for J2 in sy[1:]]
    return leaves


def _merge(ivs, tol=1e-6):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cell_edges(ax: Axis, ay: Axis, kept):
    """Strips of the kept leaves' sides.  Returns {('x' | 'y', offset): [(a, b), ...]}:
    for an 'x' line (constant x offset) the intervals are in y, and vice versa.
    The span ends are not lines (a leaf there is open on that side)."""
    lines: dict = {}
    for (I, J) in kept:
        for x in I:
            if x not in ax.virtual:
                lines.setdefault(("x", x), []).append(J)
        for y in J:
            if y not in ay.virtual:
                lines.setdefault(("y", y), []).append(I)
    return {k: _merge(v) for k, v in lines.items()}


def in_kept(kept_index, ax: Axis, ay: Axis, x: float, y: float) -> bool:
    """Whether the point (x, y) (axis offsets) lies in a kept leaf."""
    for (I, J) in kept_index:
        if I[0] - 1e-6 <= x <= I[1] + 1e-6 and J[0] - 1e-6 <= y <= J[1] + 1e-6:
            return True
    return False


# ---------------------------------------------------------------------------
# hex tree: kagome hexagons refined with a motif of their own
# ---------------------------------------------------------------------------
# A kagome (three families of lines 60 degrees apart, no three through one
# point) cannot be halved: every nested kagome or triangular refinement puts
# three strips through one point.  So a hexagon of the kagome is refined in
# place: two lines per family at +-f a from its centre, run out to the
# hexagon's sides (T-joints on them).  f ~ 0.15 draws a hexagram (a star of
# David with arms), f ~ 0.33 a hexagon nested in the hexagon with its sides
# run out.  Every crossing stays a two-strip lap.

MOTIF_F = {"star": 0.15, "ring": 0.33}


def _hex_frame(theta: float):
    us = [np.array([math.cos(theta + k * math.pi / 3), math.sin(theta + k * math.pi / 3)]) for k in range(3)]
    ns = [np.array([-u[1], u[0]]) for u in us]
    return us, ns


def motif_lines(c, a: float, f: float, us, ns):
    """Motif in the kagome hexagon centred at c (apothem a/2): for each family k
    the two lines x.n_k = c.n_k +- f a, clipped to the hexagon.
    Returns [(k, offset along n_k, t0, t1 along u_k)]."""
    R = 0.5 * a
    out = []
    for k in range(3):
        for sgn in (-1.0, 1.0):
            o = float(c @ ns[k]) + sgn * f * a
            lo, hi = -math.inf, math.inf
            for j in range(3):
                if j == k:
                    continue
                aj = o * float(ns[k] @ ns[j])
                bj = float(us[k] @ ns[j])
                cj = float(c @ ns[j])
                t1, t2 = sorted(((cj - R - aj) / bj, (cj + R - aj) / bj))
                lo, hi = max(lo, t1), min(hi, t2)
            out.append((k, o, lo, hi))
    return out


def motif_clearance(a: float, f: float, w: float) -> float:
    """The tightest clearance (mm) between strips of the motif and the hexagon
    around it: ends on one side apart, ends to the hexagon's corners (its
    crossings), crossings apart, and parallel strips apart -- each less the
    room its strips need.  Positive: the motif fits strips w wide."""
    us, ns = _hex_frame(0.0)
    c = np.zeros(2)
    L = motif_lines(c, a, f, us, ns)
    ends = []
    for k, o, t0, t1 in L:
        ends += [o * ns[k] + t0 * us[k], o * ns[k] + t1 * us[k]]
    ends = np.array(ends)
    D = np.linalg.norm(ends[:, None] - ends[None], axis=2) + np.eye(len(ends)) * 1e9
    Vr = 0.5 * a / math.cos(math.pi / 6)
    verts = np.array([Vr * np.array([math.cos(math.pi / 6 + i * math.pi / 3), math.sin(math.pi / 6 + i * math.pi / 3)])
                      for i in range(6)])
    dv = np.linalg.norm(ends[:, None] - verts[None], axis=2).min()
    pts = []
    for i, (k1, o1, a1, b1) in enumerate(L):
        for (k2, o2, a2, b2) in L[i + 1:]:
            if k1 == k2:
                continue
            A = np.array([us[k1], -us[k2]]).T
            tt = np.linalg.solve(A, o2 * ns[k2] - o1 * ns[k1])
            if a1 <= tt[0] <= b1 and a2 <= tt[1] <= b2:
                pts.append(o1 * ns[k1] + tt[0] * us[k1])
    pts = np.array(pts) if pts else np.zeros((0, 2))
    dp = (np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 1e9).min() if len(pts) > 1 else 1e9
    par = min(2 * f * a, (0.5 - f) * a)            # strips of one family: each other, the hexagon's side
    gap = w + 14.0
    return float(min(D.min() - gap, dv - (0.5 * w + 9.0), dp - gap, par - gap))


def motif_min_pitch(f: float, w: float) -> float:
    """Smallest kagome pitch whose hexagons hold the motif with strips w wide."""
    lo, hi = 20.0, 2000.0
    if motif_clearance(hi, f, w) < 0:
        return math.inf
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if motif_clearance(mid, f, w) >= 0:
            hi = mid
        else:
            lo = mid
    return hi
