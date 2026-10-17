"""Engineering checks for a built design.  Every verdict carries numbers.

  interference   no two finished parts overlap (manifold boolean volume)
  assembly       each part can drop into place, in order, without hitting an
                 earlier one (vertical rays through both solids)
  sections       the wood left at every notch (vertical rays along each beam)
  structure      grillage finite elements: every beam an Euler-Bernoulli beam,
                 crossings as compression-only contacts (they may open), legs
                 as supports; self weight + glass + a 1000 N point load moved
                 over the top, and a uniform 1 kPa load
  glass          hole size and distance to the edge
  stability      the edge load that would tip the table
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import MultiPoint, Point

from .build import Design
from .layout import bands_overlap, crossing, end_room, end_margin, inside
from .shapes import overlap
from .report import DENSITY

# modulus of elasticity (MPa) and modulus of rupture (MPa), clear wood at ~12% MC
WOOD = {
    "hinoki": (9000, 75), "sugi": (7500, 65), "white_oak": (12300, 105), "walnut": (11600, 101),
    "cherry": (10300, 85), "ash": (12000, 103), "maple": (12600, 109), "yakisugi": (7000, 60),
    "wenge": (17600, 150),
}
SAFETY = 3.0          # default; the design's own `safety` parameter is used
G = 9.81e-3           # N per kg, in N/mm units: mass_kg * 9.81 N


def status(u: float, warn: float = 1.0, fail: float = 1.5) -> str:
    return "pass" if u <= warn else ("warn" if u <= fail else "fail")


def item(name, st, value, limit, detail):
    return dict(name=name, status=st, value=value, limit=limit, detail=detail)


# ---------------------------------------------------------------------------
# geometry checks
# ---------------------------------------------------------------------------
def check_interference(d: Design) -> dict:
    parts = d.parts
    worst, pair, n = 0.0, None, 0
    bb = {q.id: q.solid.bounding_box() for q in parts}
    for i, P in enumerate(parts):
        a = bb[P.id]
        for Q in parts[i + 1:]:
            if P.group == Q.group:
                continue
            b = bb[Q.id]
            if a[3] <= b[0] or b[3] <= a[0] or a[4] <= b[1] or b[4] <= a[1] or a[5] <= b[2] or b[5] <= a[2]:
                continue
            n += 1
            v = (P.solid ^ Q.solid).volume()
            if v > worst:
                worst, pair = v, (P.id, Q.id)
    st = "pass" if worst < 1.0 else ("warn" if worst < 50 else "fail")
    det = f"{n} touching pairs checked; " + (f"worst overlap {worst:.1f} mm3 between {pair[0]} and {pair[1]}"
                                               if pair else "no overlap")
    return item("No parts overlap", st, round(worst, 2), 1.0, det)


def _column(solid, x, y, z_lo=-10.0, z_hi=2000.0):
    """(min z, max z) of a solid along a vertical line, or None."""
    hits = solid.ray_cast((x, y, z_lo), (x, y, z_hi))
    if not hits:
        return None
    zs = [h.position[2] for h in hits]
    return min(zs), max(zs)


def check_assembly(d: Design, step: float = 7.0) -> dict:
    """Drop order: wherever an earlier part and a later part share plan area,
    the earlier one must lie entirely below the later one."""
    parts = sorted(d.parts, key=lambda q: q.order)
    worst, where, pairs = 0.0, None, 0
    rng = np.random.default_rng(1)
    for i, P in enumerate(parts):
        for Q in parts[i + 1:]:
            if P.group == Q.group or not P.footprint.intersects(Q.footprint):
                continue
            inter = overlap(P.footprint, Q.footprint)
            if inter.area < 1.0:
                continue
            pairs += 1
            x0, y0, x1, y1 = inter.bounds
            nx = max(2, int((x1 - x0) / step))
            ny = max(2, int((y1 - y0) / step))
            xs = np.linspace(x0, x1, nx) + rng.uniform(-0.3, 0.3)
            ys = np.linspace(y0, y1, ny) + rng.uniform(-0.3, 0.3)
            for x in xs:
                for y in ys:
                    if not inter.contains(Point(x, y)):
                        continue
                    a = _column(P.solid, x, y)
                    b = _column(Q.solid, x, y)
                    if a is None or b is None:
                        continue
                    bad = a[1] - b[0]
                    if bad > worst:
                        worst, where = bad, (P.id, Q.id, round(x), round(y))
    st = "pass" if worst < 0.05 else ("warn" if worst < 0.5 else "fail")
    det = f"{pairs} overlapping footprints probed; " + (
        f"{where[0]} sits {worst:.2f} mm above {where[1]} at ({where[2]}, {where[3]})" if where
        else "every part drops straight into place")
    return item("Parts drop in, in order", st, round(worst, 3), 0.05, det)


def beam_depths(d: Design, B, n: int = 160):
    """Remaining wood depth along a beam (min over three rays across its width)."""
    q = d.part(B.id)
    ts = np.linspace(B.t0 + 0.5, B.t1 - 0.5, n)
    out = []
    for t in ts:
        c = B.p(t)
        best = None
        for off in (-0.35, 0.0, 0.35):
            xy = c + off * B.w * B.n
            col = q.solid.ray_cast((xy[0], xy[1], B.z0 - 5), (xy[0], xy[1], B.z1 + 5))
            if len(col) < 2:
                dep = 0.0
            else:
                zs = sorted(h.position[2] for h in col)
                dep = sum(zs[k + 1] - zs[k] for k in range(0, len(zs) - 1, 2))
            best = dep if best is None else min(best, dep)
        out.append(best)
    return ts, np.array(out)


def check_sections(d: Design) -> dict:
    worst, where = 1.0, None
    for B in d.layout.beams:
        ts, dep = beam_depths(d, B, n=90)
        # ignore the curved free ends: only the stretch between the outermost joints
        mid = (ts > B.t0 + 0.12 * B.length) & (ts < B.t1 - 0.12 * B.length)
        if not mid.any():
            continue
        r = float(dep[mid].min() / B.h)
        if r < worst:
            worst, where = r, (B.id, float(ts[mid][np.argmin(dep[mid])]))
    st = "pass" if worst >= 0.40 else ("warn" if worst >= 0.3 else "fail")
    det = (f"thinnest point keeps {100 * worst:.0f}% of the depth ({where[0]} at t={where[1]:.0f} mm)"
           if where else "no beams")
    return item("Wood left at notches", st, round(worst, 3), 0.40, det)


# ---------------------------------------------------------------------------
# grillage finite elements
# ---------------------------------------------------------------------------
class GlassStrip:
    """A strip of the glass plate lying over one top beam (same line and nodes)."""
    def __init__(self, gid, base, width, thick):
        self.id = gid
        self.base = base
        self.origin, self.u = base.origin, base.u
        self.t0, self.t1 = base.t0, base.t1
        self.w, self.h = width, thick
        self.top_surface = False

    @property
    def n(self):
        return self.base.n

    def p(self, t):
        return self.base.p(t)


class Grillage:
    def __init__(self, d: Design):
        lay = d.layout
        p = d.p
        self.d = d
        self.beams = lay.beams
        self.legs = lay.legs
        em = end_margin(p)
        E, mor = WOOD.get(p.wood_beams, WOOD["hinoki"])
        self.E, self.allow, self.mor = E, mor / p.safety, mor
        rho = DENSITY.get(p.wood_beams, 500) * 1e-9 * 9.81      # N / mm^3
        # node positions along each beam
        pos = {B.id: {B.t0, B.t1} for B in self.beams}
        self.contacts = []          # (lower, t_lower, upper, t_upper, bilateral)
        self.supports = []          # (beam, t)
        for L in self.legs:
            for B in self.beams:
                if L.idx in B.legs:
                    # the beam bears on the capital over the chord the capital
                    # cuts on its line and is clamped there by the post: two
                    # supports at the chord ends give it a fixed (moment) seat
                    t = B.leg_t(L)
                    off = abs(float((L.xy - B.origin) @ B.n))
                    half = math.sqrt(max(L.r_cap ** 2 - off ** 2, 1.0)) * 0.8
                    for tt in (t - half, t + half):
                        pos[B.id].add(tt)
                        self.supports.append((B, tt))
        for i, A in enumerate(self.beams):
            for Bm in self.beams[i + 1:]:
                if bands_overlap(A, Bm) <= 1e-6:
                    continue
                cr = crossing(A, Bm)
                if cr is None:
                    continue
                if not (inside(cr.ta, cr.ha, A.t0, A.t1, end_room(A, em)) and
                        inside(cr.tb, cr.hb, Bm.t0, Bm.t1, end_room(Bm, em))):
                    continue
                lower, tl, upper, tu = (A, cr.ta, Bm, cr.tb) if A.order < Bm.order else (Bm, cr.tb, A, cr.ta)
                xy = A.p(cr.ta)
                near_leg = any(np.linalg.norm(xy - L.xy) < L.r_cap + 0.75 * p.shaft for L in self.legs)
                self.contacts.append((lower, tl, upper, tu, near_leg))
                pos[A.id].add(cr.ta)
                pos[Bm.id].add(cr.tb)
        # the glass rests on silicone pads on top strips (and is clamped at each
        # post by its cap): find the strip under every pad
        self.pad_links = []
        if p.glass_kind != "none":
            tops = [B for B in self.beams if B.top_surface]
            for pad in getattr(lay, "pads", []) or []:
                q = np.asarray(pad["xy"], float)
                best = None
                for B in tops:
                    off = abs(float((q - B.origin) @ B.n))
                    t = float((q - B.origin) @ B.u)
                    lo_c, hi_c = B.core if getattr(B, "core", None) is not None else (B.t0, B.t1)
                    if off <= 0.5 * B.w + 1.0 and lo_c - 1.0 <= t <= hi_c + 1.0 and (best is None or off < best[0]):
                        best = (off, B, t)
                if best is not None:
                    self.pad_links.append((best[1], best[2], q))
                    pos[best[1].id].add(best[2])
        # merge nearly coincident positions
        self.nodes = {}
        for B in self.beams:
            ts = sorted(pos[B.id])
            merged = []
            for t in ts:
                if merged and t - merged[-1] < 1.0:
                    continue
                merged.append(t)
            self.nodes[B.id] = np.array(merged)
        for B in self.beams:
            B_E = E
        self.glass = []             # (the plate below replaces the old glass strips)
        self.Eg = 70000.0 if p.glass_kind == "glass" else 3200.0
        self.glass_allow = 50.0 if p.glass_kind == "glass" else 10.0
        self.members = list(self.beams) + self.glass
        dirs = []
        for B in self.beams:
            if B.top_surface and not any(abs(B.u[0] * d_[1] - B.u[1] * d_[0]) < 0.05 for d_ in dirs):
                dirs.append(B.u)
        self.n_dirs = len(dirs)
        # DOF numbering: (w, theta) per member node
        self.offset = {}
        n = 0
        for B in self.members:
            self.offset[B.id] = n
            n += 2 * len(self.nodes[B.id])
        self.ndof = n
        # element stiffness and self weight
        rows, cols, vals = [], [], []
        self.f_self = np.zeros(n)
        self.elems = []
        rho_g = DENSITY.get(p.glass_kind, 2500) * 1e-9 * 9.81
        for B in self.members:
            I = B.w * B.h ** 3 / 12.0
            # n crossing families of glass strips each carry the plate: share its stiffness
            EI = (self.Eg * 2.0 / max(2, self.n_dirs) if isinstance(B, GlassStrip) else E) * I
            ts = self.nodes[B.id]
            o = self.offset[B.id]
            for k in range(len(ts) - 1):
                Le = ts[k + 1] - ts[k]
                if Le <= 1e-9:
                    continue
                ke = EI / Le ** 3 * np.array([[12, 6 * Le, -12, 6 * Le],
                                              [6 * Le, 4 * Le * Le, -6 * Le, 2 * Le * Le],
                                              [-12, -6 * Le, 12, -6 * Le],
                                              [6 * Le, 2 * Le * Le, -6 * Le, 4 * Le * Le]])
                dofs = [o + 2 * k, o + 2 * k + 1, o + 2 * k + 2, o + 2 * k + 3]
                for a in range(4):
                    for b in range(4):
                        rows.append(dofs[a]); cols.append(dofs[b]); vals.append(ke[a, b])
                # two crossing families of glass strips each cover the plate: half weight each
                wgt = (rho_g / max(2, self.n_dirs) if isinstance(B, GlassStrip) else rho) * B.w * B.h * Le
                self.f_self[dofs[0]] -= 0.5 * wgt
                self.f_self[dofs[2]] -= 0.5 * wgt
                self.elems.append((B, k, dofs, ke))
        # the glass: a Kirchhoff plate over its own outline, coupled to the pads
        self.plate = None
        self.extra_rows = []        # (dofs, coefficients, initial gap, bilateral)
        if p.glass_kind != "none":
            from .plate import PlateMesh
            nu_g = 0.22 if p.glass_kind == "glass" else 0.37
            Dg = self.Eg * p.glass_t ** 3 / (12.0 * (1.0 - nu_g ** 2))
            pm = PlateMesh(lay.outline, Dg, offset=n, nu=nu_g)
            self.plate = pm
            r_, c_, v_ = pm.triplets()
            rows.extend(r_.tolist()); cols.extend(c_.tolist()); vals.extend(v_.tolist())
            n += pm.ndof
            self.ndof = n
            self.f_self = np.concatenate([self.f_self, np.zeros(pm.ndof)])
            rho_g = DENSITY.get(p.glass_kind, 2500) * 1e-9 * 9.81
            pm.pressure(self.f_self, np.full(pm.ne, -rho_g * p.glass_t))
            # pads: the glass presses on the strip below (compression only)
            for B, t, q in self.pad_links:
                gd, gc = pm.w_row(q)
                self.extra_rows.append((np.concatenate([gd, [self.wdof(B, t)]]), np.concatenate([gc, [-1.0]]),
                                        0.0, False))
            # posts: the cap clamps the glass onto the pads round the shaft, so the
            # glass is held there both ways (four points on the cap's bearing ring)
            r_cap = 0.5 * p.cap_d
            for L in self.legs:
                for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
                    q = L.xy + r_cap * np.array([math.cos(a), math.sin(a)])
                    gd, gc = pm.w_row(q)
                    self.extra_rows.append((gd, gc, 0.0, True))
        self.K0 = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        self.kc = 1e6        # contact / support penalty, N/mm
        # remaining depth at each node (for stresses)
        self.depth = {}
        for B in self.members:
            self.depth[B.id] = np.full(len(self.nodes[B.id]), B.h)
        split = {}
        for r in d.rels:
            if r.kind == "split":
                split[(r.a, r.b)] = r.z
        cuts_top = {B.id: {} for B in self.beams}
        cuts_bot = {B.id: {} for B in self.beams}
        for lower, tl, upper, tu, _ in self.contacts:
            if isinstance(lower, GlassStrip) or isinstance(upper, GlassStrip):
                continue
            zs = split.get((lower.id, upper.id), 0.5 * (max(lower.z0, upper.z0) + min(lower.z1, upper.z1)))
            kl = self.node_index(lower, tl)
            ku = self.node_index(upper, tu)
            cuts_top[lower.id][kl] = min(cuts_top[lower.id].get(kl, lower.z1), zs)
            cuts_bot[upper.id][ku] = max(cuts_bot[upper.id].get(ku, upper.z0), zs)
        for B in self.beams:
            for k in range(len(self.nodes[B.id])):
                top = cuts_top[B.id].get(k, B.z1)
                bot = cuts_bot[B.id].get(k, B.z0)
                self.depth[B.id][k] = max(top - bot, 0.5)
        # load nodes when there is no glass: the top beams
        self.top_nodes = []
        for B in ([] if self.plate is not None else [b for b in self.beams if b.top_surface]):
            for k, t in enumerate(self.nodes[B.id]):
                self.top_nodes.append((B, k, B.p(t)))
        self.top_xy = np.array([q for _, _, q in self.top_nodes]) if self.top_nodes else np.zeros((0, 2))

    def node_index(self, B, t) -> int:
        return int(np.argmin(np.abs(self.nodes[B.id] - t)))

    def wdof(self, B, t) -> int:
        return self.offset[B.id] + 2 * self.node_index(B, t)

    def spread(self, xy, P: float, radius: float) -> np.ndarray:
        """Point load P (N, downward) over a disc of `radius` on the glass, or
        shared by the top beam nodes within it when there is no glass."""
        f = np.zeros(self.ndof)
        if self.plate is not None:
            self.plate.patch(f, xy, P, radius)
            return f
        if len(self.top_xy) == 0:
            return f
        r = np.linalg.norm(self.top_xy - xy, axis=1)
        w = np.clip(1 - r / radius, 0, None) ** 2
        if w.sum() < 1e-9:
            idx = np.argsort(r)[:3]
            w = np.zeros_like(r)
            w[idx] = 1.0 / (r[idx] + 1.0)
        w /= w.sum()
        for (B, k, _), wi in zip(self.top_nodes, w):
            if wi > 0:
                f[self.offset[B.id] + 2 * k] -= P * wi
        return f

    def _prep(self):
        if hasattr(self, "_C"):
            return
        ri, cj, cv, g0, bil = [], [], [], [], []
        r = 0
        for lo, tl, up, tu, b in self.contacts:            # gap = w_upper - w_lower
            ri += [r, r]
            cj += [self.wdof(up, tu), self.wdof(lo, tl)]
            cv += [1.0, -1.0]
            g0.append(0.0)
            bil.append(b)
            r += 1
        for dofs, coef, gap0, b in self.extra_rows:
            ri += [r] * len(dofs)
            cj += [int(x) for x in dofs]
            cv += [float(x) for x in coef]
            g0.append(gap0)
            bil.append(b)
            r += 1
        self._C = coo_matrix((cv, (ri, cj)), shape=(r, self.ndof)).tocsr()
        self._g0 = np.array(g0, dtype=float)
        self._bil = np.array(bil, dtype=bool).reshape(r)
        sup = np.array([self.wdof(B, t) for B, t in self.supports], dtype=np.int64)
        nb = self.plate.offset if self.plate is not None else self.ndof
        d = np.arange(nb)
        diag = np.where(d % 2 == 0, 1e-3, 1e-1)          # a whisper of ground stiffness
        if self.plate is not None:
            dp = np.arange(nb, self.ndof)
            d = np.concatenate([d, dp])
            diag = np.concatenate([diag, np.full(len(dp), 1e-9 * float(self.plate.Ke.diagonal().max()))])
        rows = np.concatenate([sup, d])
        vals = np.concatenate([np.full(len(sup), self.kc), diag])
        self._Kbase = self.K0 + coo_matrix((vals, (rows, rows)), shape=(self.ndof, self.ndof)).tocsr()
        self._lu = {}

    def _factor(self, active: np.ndarray):
        key = active.tobytes()
        if key in self._lu:
            return self._lu[key]
        on = active | self._bil
        Ca = self._C[on]
        K = self._Kbase + self.kc * (Ca.T @ Ca)
        # symmetric positive definite: a symmetric ordering without pivoting
        # factors 2-3x faster than the default (and with less fill)
        from .linalg import factor
        lu = factor(K)
        if len(self._lu) > 8:              # each factor holds ~10 MB: keep the WebAssembly heap small
            self._lu.clear()
        self._lu[key] = lu
        return lu

    def _open_solve(self, rhs: np.ndarray, open_rows: np.ndarray):
        """Solve with the contact rows `open_rows` opened.  Every load case
        starts from all contacts closed and opens only a few laps and pads, so
        instead of a factorization per contact state this updates the one
        all-closed factorization F (Woodbury):

            (K - kc C_o^T C_o)^-1 r = y + G (I/kc - C_o G)^-1 C_o y,
            y = F^-1 r,  G = F^-1 C_o^T  (columns cached per row).

        The small matrix is the opened contacts' flexibility; when it is too
        close to singular to trust (an opened part with almost nothing
        holding it) the state gets its own factorization instead."""
        F = self._factor(np.ones(len(self._g0), dtype=bool))
        y = F.solve(rhs)
        if not len(open_rows):
            return y
        if not hasattr(self, "_G"):
            self._G = {}
        new = [int(j) for j in open_rows if int(j) not in self._G]
        if new:
            X = F.solve(np.asarray(self._C[new].T.todense()))
            for k, j in enumerate(new):
                self._G[j] = X[:, k]
        G = np.stack([self._G[int(j)] for j in open_rows], axis=1)
        Co = self._C[open_rows]
        M = np.eye(len(open_rows)) / self.kc - Co @ G
        try:
            if np.linalg.cond(M) < 1e12:
                return y + G @ np.linalg.solve(M, Co @ y)
        except np.linalg.LinAlgError:
            pass
        active = np.ones(len(self._g0), dtype=bool)
        active[open_rows] = False
        return self._factor(active).solve(rhs)

    def solve(self, f: np.ndarray, iters: int = 12):
        """Linear solve with compression-only contacts (opening ones dropped)."""
        self._prep()
        active = np.ones(len(self._g0), dtype=bool)
        u = None
        for it in range(iters):
            on = active | self._bil
            rhs = f - self.kc * (self._C[on].T @ self._g0[on]) if self._g0.any() else f
            u = self._open_solve(rhs, np.flatnonzero(~on))
            gap = self._C @ u + self._g0                 # > 0: the upper part lifts off
            want = (gap <= 1e-4) | self._bil
            if np.array_equal(want, active):
                break
            active = want
        return u, list(active)

    def release(self):
        """Free the factorizations (they are large)."""
        self._lu = {}
        self._G = {}

    def top_deflection(self, u) -> np.ndarray:
        """Downward deflections of the surface one touches: the glass, or the top beams."""
        if self.plate is not None:
            return -self.plate.node_w(u)
        if not self.top_nodes:
            return np.zeros(1)
        return -np.array([u[self.offset[B.id] + 2 * k] for (B, k, _) in self.top_nodes])

    def glass_stress(self, u):
        """Peak bending stress in the glass (MPa) and where: 6 M / t^2 from the
        larger principal moment at each element centre."""
        if self.plate is None:
            return 0.0, None
        M = self.plate.moments(u)
        c = 0.5 * (M[:, 0] + M[:, 1])
        r = np.sqrt((0.5 * (M[:, 0] - M[:, 1])) ** 2 + M[:, 2] ** 2)
        m = np.maximum(np.abs(c + r), np.abs(c - r))
        e = int(np.argmax(m))
        t = self.d.p.glass_t
        return float(6.0 * m[e] / t ** 2), self.plate.xs[e]

    def stresses(self, u, glass: bool = False):
        """Max bending stress (MPa): wood at the notched depth (x1.3 for the
        notch), or the glass strips when glass=True."""
        worst = (0.0, None, None)
        if glass:
            return worst
        for B, k, dofs, ke in self.elems:
            fe = ke @ u[dofs]
            for end, M in ((k, fe[1]), (k + 1, fe[3])):
                dep = max(self.depth[B.id][end], 1.0)
                notched = dep < B.h - 1e-6
                S = B.w * dep ** 2 / 6.0
                sig = abs(M) / S * (1.3 if notched else 1.0)
                if sig > worst[0]:
                    worst = (sig, B.id, float(self.nodes[B.id][end]))
        return worst


def f_dead_probe(g: "Grillage", xy, P, radius):
    return g.spread(xy, P, radius)


def check_structure(d: Design, n_points: int = 28) -> list[dict]:
    lay = d.layout
    p = d.p
    if not lay.beams:
        return [item("Structure", "fail", 0, 0, "no beams")]
    g = Grillage(d)
    # dead load: self weight of wood and glass strips
    f_dead = g.f_self.copy()
    spread = 50.0          # a hand or knee; the glass plate (if any) spreads it further itself
    # candidate spots for a person's weight: the rim (most cantilever) and between
    # legs -- on the glass if there is glass, else on the grid that carries load
    if p.glass_kind != "none":
        reg = lay.outline
    else:
        from shapely.ops import unary_union
        from .shapes import strip as _strip
        core = [(_strip(b.origin, b.u, b.core[0], b.core[1], b.w) if getattr(b, "core", None) else b.strip())
                for b in lay.beams if b.top_surface]
        reg = unary_union(core).convex_hull if core else lay.outline
    ring = reg.buffer(-25.0)
    if ring.is_empty or ring.geom_type != "Polygon":
        ring = reg
    cand = [np.array(ring.exterior.interpolate(s, normalized=True).coords[0])
            for s in np.linspace(0, 1, n_points, endpoint=False)]
    legs_xy = np.array([L.xy for L in lay.legs])
    c0 = np.array(reg.centroid.coords[0])
    cand.append(c0)
    for i in range(len(legs_xy)):
        for j in range(i + 1, len(legs_xy)):
            cand.append(0.5 * (legs_xy[i] + legs_xy[j]))
    P = float(p.design_load)
    worst = dict(util=0.0, defl=0.0, where=None, beam=None, unstable=False, opened=0)
    for q in cand:
        f = f_dead + g.spread(q, P, spread)
        u, active = g.solve(f)
        defl = float(g.top_deflection(u).max())
        sig, bid, t = g.stresses(u)
        util = sig / g.allow
        if g.plate is not None:
            sg, _ = g.glass_stress(u)
            if sg > worst.get("glass_sig", 0.0):
                worst.update(glass_sig=sg, glass_where=q)
        if defl > 60.0:
            worst.update(unstable=True, where=q, beam=bid)
        if util > worst["util"]:
            worst.update(util=util, where_s=q, beam=bid, t=t, sig=sig)
        if defl > worst["defl"]:
            worst.update(defl=defl, where_d=q)
        worst["opened"] = max(worst["opened"], sum(1 for a in active if not a))
    # uniform 1 kPa over the top
    if g.plate is not None or g.top_nodes:
        f = f_dead.copy()
        if g.plate is not None:
            g.plate.pressure(f, np.full(g.plate.ne, -1e-3))
        else:
            q_N = 1e-3 * reg.area
            for (B, k, _) in g.top_nodes:
                f[g.offset[B.id] + 2 * k] -= q_N / len(g.top_nodes)
        u, _ = g.solve(f)
        sig_u, bid_u, _ = g.stresses(u)
    else:
        sig_u, bid_u = 0.0, None
    g.release()
    out = []
    if worst["unstable"]:
        out.append(item("Load path", "fail", 0, 0,
                        f"a {P:.0f} N load near ({worst['where'][0]:.0f}, {worst['where'][1]:.0f}) lifts {worst['beam']} "
                        "off its supports (the lattice is a mechanism there)"))
    else:
        out.append(item("Load path", "pass", worst["opened"], None,
                        f"stable for every probe; at most {worst['opened']} lap contacts open (they may, they are not glued)"))
    u = worst["util"]
    if worst.get("where_s") is not None:
        ws = worst["where_s"]
        out.append(item(f"Beam bending ({P:.0f} N anywhere)", status(u), round(worst["sig"], 1), round(g.allow, 1),
                        f"peak {worst['sig']:.1f} MPa in {worst['beam']} (notches included) with the load at "
                        f"({ws[0]:.0f}, {ws[1]:.0f}); allowable {g.allow:.1f} MPa = MOR {g.mor:g}/{p.safety:g} for {p.wood_beams}"))
    out.append(item("Beam bending (1 kPa uniform)", status(sig_u / g.allow), round(sig_u, 1), round(g.allow, 1),
                    f"peak {sig_u:.1f} MPa in {bid_u}"))
    dd = worst["defl"]
    st = "pass" if dd <= 3.0 else ("warn" if dd <= 6.0 else "fail")
    wd = worst.get("where_d", c0)
    out.append(item(f"Deflection under {P:.0f} N", st, round(dd, 2), 3.0,
                    f"worst {dd:.2f} mm with the load at ({wd[0]:.0f}, {wd[1]:.0f})"))
    if g.plate is not None:
        sg = worst.get("glass_sig", 0.0)
        gw = worst.get("glass_where", c0)
        out.append(item(f"Glass bending ({P:.0f} N)", status(sg / g.glass_allow), round(sg, 1), g.glass_allow,
                        f"peak {sg:.1f} MPa in the {p.glass_t:.0f} mm {p.glass_kind} with the load at "
                        f"({gw[0]:.0f}, {gw[1]:.0f}); design strength {g.glass_allow:.0f} MPa"))
    return out


# ---------------------------------------------------------------------------
# glass and stability
# ---------------------------------------------------------------------------
def check_glass(d: Design) -> list[dict]:
    p = d.p
    lay = d.layout
    if p.glass_kind == "none":
        return []
    t = p.glass_t
    out = []
    hole_d = p.post_d + 2 * p.glass_hole_clear
    ok_d = hole_d >= t
    out.append(item("Glass hole diameter", "pass" if ok_d else "fail", round(hole_d, 1), t,
                    f"holes {hole_d:.1f} mm in {t:.0f} mm glass (must be at least the thickness)"))
    edge = min(lay.outline.exterior.distance(Point(*L.xy)) - 0.5 * hole_d for L in lay.legs)
    need = 2.0 * t
    out.append(item("Glass hole to edge", "pass" if edge >= need else ("warn" if edge >= 1.5 * t else "fail"),
                    round(edge, 1), need, f"closest hole edge is {edge:.0f} mm from the glass edge (rule: >= 2 x thickness)"))
    if len(lay.legs) > 1:
        dmin = min(np.linalg.norm(a.xy - b.xy) for i, a in enumerate(lay.legs) for b in lay.legs[i + 1:]) - hole_d
        out.append(item("Glass hole spacing", "pass" if dmin >= 2 * t else "fail", round(dmin, 1), 2 * t,
                        f"closest holes {dmin:.0f} mm apart edge to edge"))
    # overhang past the lattice
    from .shapes import strip as _strip
    top = [(_strip(b.origin, b.u, b.core[0], b.core[1], b.w) if getattr(b, "core", None) else b.strip())
           for b in lay.beams if b.top_surface]
    if top:
        from shapely.ops import unary_union
        lat = unary_union(top)
        far = max(lat.distance(Point(*c)) for c in np.asarray(lay.outline.exterior.coords)[::4])
        # a rule of thumb; the glass bending check above is the real test of the overhang
        lim = 10 * t
        out.append(item("Glass overhang", "pass" if far <= lim else ("warn" if far <= 20 * t else "fail"),
                        round(far, 1), lim, f"glass reaches at most {far:.0f} mm past the supporting grid "
                                            "(the glass bending check carries the load out there)"))
    return out


def mass_centre(solid) -> tuple[float, np.ndarray]:
    """Volume (mm^3) and centre of a closed solid, from its triangle mesh
    (signed tetrahedra to the origin)."""
    m = solid.to_mesh()
    V = np.asarray(m.vert_properties, float)[:, :3]
    T = np.asarray(m.tri_verts, np.int64)
    if not len(T):
        return 0.0, np.zeros(3)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    tot = float(vol.sum())
    if abs(tot) < 1e-9:
        return 0.0, np.zeros(3)
    return tot, (vol[:, None] * (a + b + c)).sum(0) / (4.0 * tot)


def check_stability(d: Design) -> list[dict]:
    """Tipping: the smallest push down on the rim that tips the table over an
    edge of its feet (stance.support_tip), from the real solids' weight and
    centre of mass."""
    from .stance import TIP_LOAD, support_tip
    lay = d.layout
    W = 0.0
    m_xy = np.zeros(2)
    for q in d.parts:
        if q.solid.is_empty():
            continue
        vol, c = mass_centre(q.solid)
        kg = vol * 1e-9 * DENSITY.get(q.material, 600)
        W += kg * 9.81
        m_xy += kg * 9.81 * c[:2]
    com = m_xy / max(W, 1e-9)
    # floor contact: the flat circle under each rounded foot
    feet, radii = [], []
    for q in d.parts:
        if q.kind == "leg":
            L = lay.legs[q.meta["leg"]]
            feet.append(L.foot_xy)
            radii.append(float(q.meta["stations"]["r"][0]))
    ring = np.asarray(lay.outline.exterior.coords)[::2]
    t = support_tip(feet, radii, ring, W, com) if len(feet) >= 2 else None
    if t is None:
        return [item("Tipping", "fail", 0, TIP_LOAD, "legs do not enclose an area")]
    if not math.isfinite(t["P"]):
        return [item("Tipping", "pass", None, TIP_LOAD, "the top never reaches past the feet")]
    worst, pt = t["P"], t["at"]
    st = "pass" if worst >= TIP_LOAD else ("warn" if worst >= 350 else "fail")
    return [item("Tipping", st, round(worst), TIP_LOAD,
                 f"{worst:.0f} N pressing down at ({pt[0]:.0f}, {pt[1]:.0f}) on the rim tips it "
                 f"(table weighs {W / 9.81:.1f} kg; feet touch the floor on {float(st_r0(d)):.0f} mm circles)")]


def st_r0(d: Design) -> float:
    for q in d.parts:
        if q.kind == "leg":
            return 2 * float(q.meta["stations"]["r"][0])
    return 0.0


def check_symmetry(d: Design) -> dict:
    """Certificate: map the finished plan onto itself under every operation of
    the group it follows; report the worst mismatch."""
    from .symmetry import plan_mismatch
    lay = d.layout
    if not lay.sym_ops:
        why = "the legs have no symmetry a grid can follow" if lay.sym_legs == "none" else \
            "a twisted or free grid is not held to a symmetry"
        return item("Symmetry", "pass", 0.0, None, f"legs: {lay.sym_legs}; plan not held to a symmetry ({why})")
    mm = plan_mismatch(lay.beams, lay.sym_ops)
    worst = max(mm["top"], mm["tiers"])
    st = "pass" if worst < 50.0 else ("warn" if worst < 500.0 else "fail")
    return item("Symmetry", st, round(worst, 1), 50.0,
                f"legs: {lay.sym_legs}; the top grid follows {lay.sym_label} ({len(lay.sym_ops) + 1} operations), "
                f"each bracket tier the part of it that keeps its direction; worst mismatch "
                f"{mm['top']:.1f} mm2 (top), {mm['tiers']:.1f} mm2 (tiers)")


def quick_structure(lay, n_points: int = 8) -> tuple[int, float]:
    """(failing structural checks, worst utilisation) of a bare layout, with
    the joints derived but no solids cut: picks how many bracket tiers a design
    needs to carry its design load."""
    from .joinery import relations
    from .model import make_parts
    parts = make_parts(lay)
    rels = relations(parts, lay.p)
    res = check_structure(Design(lay.p, lay, parts, rels), n_points=n_points)
    fails = sum(1 for r in res if r["status"] == "fail")
    utils = [float(r["value"]) / float(r["limit"]) for r in res
             if r.get("limit") and r.get("value") is not None and float(r["limit"]) > 0]
    return fails, max(utils, default=0.0)


def run_checks(d: Design, full: bool = False, quick: bool = False) -> list[dict]:
    """quick: for ranking candidates in a search (no overlap test -- it holds by
    construction and is re-checked on the winner -- and fewer load probes)."""
    out = []
    if not quick:
        out.append(check_interference(d))
    if full:
        out.append(check_assembly(d))
    out.append(check_sections(d))
    out.extend(check_structure(d, n_points=36 if full else (10 if quick else 20)))
    out.extend(check_glass(d))
    # the glass overhang rule of thumb defers to the plate model: when the glass
    # itself passes under the design load at its edge, a long overhang only warns
    glass_ok = all(r["status"] == "pass" for r in out if r["name"].startswith("Glass bending"))
    for r in out:
        if r["name"] == "Glass overhang" and r["status"] == "fail" and glass_ok:
            r["status"] = "warn"
    out.extend(check_stability(d))
    out.append(check_symmetry(d))
    return out
