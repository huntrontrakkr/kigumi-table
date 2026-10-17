"""Plan-view geometry: the top outline, its tier offsets, and line clipping."""
from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .params import Params


def _rounded(poly: Polygon, r: float) -> Polygon:
    if r <= 0:
        return poly
    return poly.buffer(-r, join_style="mitre").buffer(r, quad_segs=24)


def top_outline(p: Params) -> Polygon:
    """Glass outline in plan, centred on the origin."""
    a, b = p.top_w / 2, p.top_d / 2
    s = p.top_shape
    if s == "circle":
        poly = Point(0, 0).buffer(1.0, quad_segs=64)
        poly = affinity.scale(poly, a, a)
    elif s == "squircle":
        t = np.linspace(0, 2 * math.pi, 256, endpoint=False)
        n = p.squircle_n
        c, sn = np.cos(t), np.sin(t)
        x = a * np.sign(c) * np.abs(c) ** (2 / n)
        y = a * np.sign(sn) * np.abs(sn) ** (2 / n)
        poly = Polygon(np.c_[x, y])
    elif s == "square":
        poly = _rounded(box(-a, -a, a, a), min(p.top_corner, a * 0.9))
    elif s == "rect":
        poly = _rounded(box(-a, -b, a, b), min(p.top_corner, min(a, b) * 0.9))
    elif s == "polygon":
        n = max(3, int(p.top_sides))
        # top_w is the across-flats width for even n, circumdiameter otherwise
        R = a / math.cos(math.pi / n) if n % 2 == 0 else a
        t = np.arange(n) * 2 * math.pi / n + math.pi / 2
        poly = Polygon(np.c_[R * np.cos(t), R * np.sin(t)])
        poly = _rounded(poly, min(p.top_corner, a * 0.5))
    else:
        raise ValueError(s)
    if p.top_rot:
        poly = affinity.rotate(poly, p.top_rot, origin=(0, 0))
    return orient(poly, 1.0)


def inset(poly: Polygon, d: float) -> Polygon:
    if d <= 0:
        return poly
    q = poly.buffer(-d, join_style="round", quad_segs=24)
    if q.is_empty:
        return Polygon()
    if q.geom_type != "Polygon":
        q = max(q.geoms, key=lambda g: g.area)
    return orient(q, 1.0)


def extent_along(poly, n: np.ndarray) -> tuple[float, float]:
    """min/max of x.n over the polygon (or every polygon of a multipolygon)."""
    geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
    xy = np.vstack([np.asarray(g.exterior.coords) for g in geoms])
    v = xy @ n
    return float(v.min()), float(v.max())


def clip_line(poly: Polygon, origin: np.ndarray, u: np.ndarray, big: float = 1e4):
    """Parameter interval [t0, t1] of the line origin + t u inside poly (longest piece)."""
    ls = LineString([origin - big * u, origin + big * u])
    inter = ls.intersection(poly)
    if inter.is_empty:
        return None
    if inter.geom_type == "LineString":
        pieces = [inter]
    else:
        pieces = [g for g in getattr(inter, "geoms", []) if g.geom_type == "LineString"]
        if not pieces:
            return None
    best = max(pieces, key=lambda g: g.length)
    c = np.asarray(best.coords)
    ts = (c - origin) @ u
    return float(ts.min()), float(ts.max())


def strip(origin: np.ndarray, u: np.ndarray, t0: float, t1: float, w: float) -> Polygon:
    n = np.array([-u[1], u[0]])
    a = origin + t0 * u
    b = origin + t1 * u
    h = 0.5 * w * n
    return Polygon([a - h, b - h, b + h, a + h])


def line_cross(o1, u1, o2, u2):
    """Parameters (t1, t2) of the intersection of two lines, or None if parallel."""
    # Cramer's rule on [u1 -u2] t = o2 - o1
    a, b, c, d = float(u1[0]), -float(u2[0]), float(u1[1]), -float(u2[1])
    det = a * d - b * c
    if abs(det) < 1e-9:
        return None
    rx, ry = float(o2[0] - o1[0]), float(o2[1] - o1[1])
    return (rx * d - b * ry) / det, (a * ry - rx * c) / det


def warp_offsets(W: float, pitch: float, density: float) -> list[float]:
    """Symmetric offsets in [-W, W] whose spacing shrinks toward the edge
    (density > 0) or toward the centre (density < 0)."""
    if W <= 0:
        return [0.0]
    K = max(1, int(round(W / pitch)))
    beta = 1.0 / (1.0 + density) if density >= 0 else 1.0 - density
    out = []
    for k in range(-K, K + 1):
        x = k / K
        out.append(W * math.copysign(abs(x) ** beta, x))
    return out


def clip_line_all(region, origin: np.ndarray, u: np.ndarray, big: float = 1e4) -> list[tuple]:
    """All parameter intervals of the line origin + t u inside region (any shape)."""
    if region is None or region.is_empty:
        return []
    ls = LineString([origin - big * u, origin + big * u])
    inter = ls.intersection(region)
    if inter.is_empty:
        return []
    pieces = [inter] if inter.geom_type == "LineString" else \
        [g for g in getattr(inter, "geoms", []) if g.geom_type == "LineString"]
    out = []
    for g in pieces:
        c = np.asarray(g.coords)
        ts = (c - origin) @ u
        out.append((float(ts.min()), float(ts.max())))
    return sorted(out)


def _areal(g):
    """Only the polygonal parts of g (an overlap can carry stray lines or points)."""
    if g.is_empty or g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [q for q in getattr(g, "geoms", []) if q.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else Polygon()


def overlap(a, b):
    """a & b on a 0.001 mm grid: two strips on one line have nearly collinear
    edges, and the floating-point overlay can then return nothing at all.
    The result keeps only its polygonal parts."""
    a, b = _areal(a), _areal(b)
    if a.is_empty or b.is_empty:
        return Polygon()
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:        # apart: skip the overlay
        return Polygon()
    return _areal(a.intersection(b, grid_size=1e-3))
