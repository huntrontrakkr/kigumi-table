"""The legs' radiance field.

Every leg radiates a disc (or a rounded square) of influence.  The fields are
combined with an exponential smooth minimum of signed distances,

    d(p) = -k * ln( sum_i exp( -d_i(p) / k ) ),   d_i = |p - L_i|_n - R_i,

which is exactly the leg's own circle when it stands alone and fuses
neighbouring circles into one blob when they come close -- the 2D version of
metaballs / smooth union in an SDF shader.  The zero level set is the outline of
the top; deeper level sets (d = -inset) are the footprints of lower tiers, so
low tiers sit in separate discs around each leg and merge as they rise.
"""
from __future__ import annotations

import math

import contourpy
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union


class Field:
    def __init__(self, centres, radii, blend: float, n: float = 2.0, angle_deg: float = 0.0):
        self.c = np.asarray(centres, float).reshape(-1, 2)
        self.r = np.broadcast_to(np.asarray(radii, float), (len(self.c),)).copy()
        self.k = max(1e-3, float(blend))
        self.n = max(1.0, float(n))
        a = math.radians(angle_deg)
        self.rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])

    def d_each(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.stack([x, y], axis=-1)
        out = []
        for c, r in zip(self.c, self.r):
            v = (pts - c) @ self.rot.T
            if self.n == 2.0:
                dist = np.hypot(v[..., 0], v[..., 1])
            else:
                dist = (np.abs(v[..., 0]) ** self.n + np.abs(v[..., 1]) ** self.n) ** (1.0 / self.n)
            out.append(dist - r)
        return np.stack(out)

    def d(self, x, y) -> np.ndarray:
        di = self.d_each(np.asarray(x, float), np.asarray(y, float))
        m = di.min(axis=0)
        return m - self.k * np.log(np.exp(-(di - m) / self.k).sum(axis=0))

    def bounds(self, pad: float = 0.0):
        R = self.r.max() + self.k * math.log(len(self.c) + 1) + pad
        lo = self.c.min(axis=0) - R
        hi = self.c.max(axis=0) + R
        return lo, hi

    def region(self, level: float = 0.0, step: float = 3.0, fill_holes: bool = True):
        """Polygon {d <= level} (largest piece), via marching squares (remembered:
        a layout asks for the same level sets again for every tier count)."""
        key = (round(float(level), 6), float(step), bool(fill_holes), self.k, tuple(np.round(self.r, 6)))
        memo = self.__dict__.setdefault("_regions", {})
        if key not in memo:
            memo[key] = self._region(level, step, fill_holes)
        return memo[key]

    def _region(self, level: float, step: float, fill_holes: bool):
        lo, hi = self.bounds(pad=20.0)
        nx = int(math.ceil((hi[0] - lo[0]) / step)) + 1
        ny = int(math.ceil((hi[1] - lo[1]) / step)) + 1
        xs = np.linspace(lo[0], hi[0], nx)
        ys = np.linspace(lo[1], hi[1], ny)
        X, Y = np.meshgrid(xs, ys)
        Z = self.d(X, Y)
        gen = contourpy.contour_generator(xs, ys, Z)
        lines = [LineString(seg) for seg in gen.lines(level) if len(seg) >= 4]
        if not lines:
            return Polygon()
        faces = [f for f in polygonize(unary_union(lines))]
        keep = []
        for f in faces:
            q = f.representative_point()
            if self.d(np.array([q.x]), np.array([q.y]))[0] <= level:
                keep.append(f)
        if not keep:
            return Polygon()
        reg = unary_union(keep)
        if isinstance(reg, MultiPolygon):
            pieces = sorted(reg.geoms, key=lambda g: g.area, reverse=True)
        else:
            pieces = [reg]
        out = []
        for g in pieces:
            if fill_holes:
                g = Polygon(g.exterior)
            out.append(orient(g.simplify(0.4), 1.0))
        return out[0] if len(out) == 1 else MultiPolygon(out)


def leg_field(p, legs_xy) -> Field:
    """Field for a parameter set: one radius per leg (with optional seeded variation)."""
    rng = np.random.default_rng(int(p.seed) + 17)
    var = rng.uniform(-1, 1, len(legs_xy)) * p.field_var
    radii = p.field_radius * (1.0 + var)
    return Field(legs_xy, radii, p.field_blend, p.field_n, p.angle)


_FIELDS: dict = {}


def connected_field(p, legs_xy, notes: list) -> Field:
    """Grow the leg radius until the field is one piece where the lattice lives
    (inset by the overhang and half a cell), so every leg's patch joins the
    others through a neck at least a cell wide (remembered per legs and field
    parameters)."""
    key = (tuple(np.round(np.asarray(legs_xy, float).ravel(), 6)), p.field_radius, p.field_blend, p.field_n,
           p.field_var, p.seed, p.angle, p.overhang, p.pitch)
    if key in _FIELDS:
        f, msgs = _FIELDS[key]
        notes.extend(msgs)
        return f
    msgs: list = []
    f = _connected_field(p, legs_xy, msgs)
    if len(_FIELDS) > 32:
        _FIELDS.clear()
    _FIELDS[key] = (f, msgs)
    notes.extend(msgs)
    return f


def _connected_field(p, legs_xy, notes: list) -> Field:
    f = leg_field(p, legs_xy)
    level = -(p.overhang + 0.5 * p.pitch)
    k0, r0 = f.k, f.r.copy()

    def joined():
        reg = f.region(level, step=6.0)
        return isinstance(reg, Polygon) and not reg.is_empty

    # first soften the blend (the neck between blobs fills in while the outer
    # edge barely moves), then, if that is not enough, grow the radius
    for i in range(12):
        if joined():
            if i:
                notes.append(f"field blend softened to {f.k:.0f} mm so the legs' fields join into one top")
            return f
        f.k = k0 * 1.15 ** (i + 1)
        if f.k > max(3.0 * k0, 0.6 * float(r0.mean())):
            break
    f.k = k0
    for i in range(40):
        if joined():
            if i:
                notes.append(f"field radius grown to {f.r.mean():.0f} mm so the legs' fields join into one top")
            return f
        f.r *= 1.04
    return f
