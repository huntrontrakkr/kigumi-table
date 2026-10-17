"""The top as a plate: where the bending moments are, and which way they run.

A Kirchhoff plate over the top's outline, held at the leg capitals, is solved
for a uniform load and for the design load (as a hand-sized patch) at many
places.  Elements are Bogner-Fox-Schmit rectangles (bicubic Hermite, 16
degrees of freedom: w, w_x, w_y, w_xy at each corner) on a regular grid kept
where the element centre lies in the outline; free edges need no conditions.

A lattice of straight strips carries the plate's moments in its own
directions.  For two families at angle theta the design moments per unit
width are Wood-Armer's

    m_u = |M_uu| + |M_uv|,   m_v = |M_vv| + |M_uv|

(the twisting moment has to be carried by bending in both), enveloped over
the load cases.  A fully stressed grillage spaces its strips inversely to the
moment they carry: spacing = capacity / m.  The total demand, the integral of
m_u + m_v over the top, measures how much strip a direction pair needs, and
the grid angle that minimises it is the best orientation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


def _hermite(x: float, h: float):
    """1D Hermite cubics on [0, h] at x: values and first/second derivatives of
    (value at 0, slope at 0, value at h, slope at h)."""
    s = x / h
    N = np.array([1 - 3 * s * s + 2 * s ** 3, h * (s - 2 * s * s + s ** 3), 3 * s * s - 2 * s ** 3, h * (-s * s + s ** 3)])
    dN = np.array([(-6 * s + 6 * s * s) / h, 1 - 4 * s + 3 * s * s, (6 * s - 6 * s * s) / h, -2 * s + 3 * s * s])
    d2N = np.array([(-6 + 12 * s) / h ** 2, (-4 + 6 * s) / h, (6 - 12 * s) / h ** 2, (-2 + 6 * s) / h])
    return N, dN, d2N


# local node order (0,0), (h,0), (h,h), (0,h); per node dofs w, w_x, w_y, w_xy
_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _shape(x: float, y: float, h: float):
    """Rows: N, N_xx, N_yy, N_xy for the 16 element dofs at (x, y)."""
    Nx, dNx, d2Nx = _hermite(x, h)
    Ny, dNy, d2Ny = _hermite(y, h)
    out = np.zeros((4, 16))
    for c, (i, j) in enumerate(_CORNERS):
        # 1D function indices: value / slope at end i (x) and j (y)
        vx, sx = 2 * i, 2 * i + 1
        vy, sy = 2 * j, 2 * j + 1
        for d, (fx, fy) in enumerate(((vx, vy), (sx, vy), (vx, sy), (sx, sy))):
            k = 4 * c + d
            out[0, k] = Nx[fx] * Ny[fy]
            out[1, k] = d2Nx[fx] * Ny[fy]
            out[2, k] = Nx[fx] * d2Ny[fy]
            out[3, k] = dNx[fx] * dNy[fy]
    return out


def element_matrices(h: float, nu: float = 0.22):
    """Unit-rigidity (D = 1) stiffness, unit-pressure load vector and the
    centre curvature operator of a square BFS element of side h."""
    g, wts = np.polynomial.legendre.leggauss(4)
    pts = 0.5 * h * (g + 1.0)
    wq = 0.5 * h * wts
    C = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])
    K = np.zeros((16, 16))
    F = np.zeros(16)
    for x, wx in zip(pts, wq):
        for y, wy in zip(pts, wq):
            S = _shape(x, y, h)
            B = np.vstack([S[1], S[2], 2.0 * S[3]])
            K += wx * wy * B.T @ C @ B
            F += wx * wy * S[0]
    S = _shape(0.5 * h, 0.5 * h, h)
    Bc = np.vstack([S[1], S[2], S[3]])                  # w_xx, w_yy, w_xy
    Mc = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 1.0 - nu]]) @ Bc
    return K, F, Mc                                      # moments per unit D: (Mxx, Myy, Mxy)


@dataclass
class PlateField:
    xs: np.ndarray            # element centres (n, 2)
    M: np.ndarray             # moments per case (cases, n, 3): Mxx, Myy, Mxy (N mm / mm)
    h: float
    area: float
    origin: tuple = (0.0, 0.0)
    shape: tuple = (0, 0)
    ij: np.ndarray | None = None      # grid indices of the elements (n, 2)

    def lookup(self, values: np.ndarray):
        """Callable giving `values` (one row per element) at plan points: the
        element under each point, zero off the plate."""
        nx, ny = self.shape
        grid = np.zeros((nx, ny) + values.shape[1:])
        grid[self.ij[:, 0], self.ij[:, 1]] = values
        ox, oy = self.origin

        def at(pts):
            pts = np.asarray(pts, float).reshape(-1, 2)
            i = np.floor((pts[:, 0] - ox) / self.h).astype(int)
            j = np.floor((pts[:, 1] - oy) / self.h).astype(int)
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            out = np.zeros((len(pts),) + values.shape[1:])
            out[ok] = grid[i[ok], j[ok]]
            return out
        return at


_CACHE: dict = {}


def plate_for(region, supports, load: float, q: float, patch_r: float = 50.0) -> PlateField:
    """solve_plate, remembered (a layout is built several times while bracket
    tiers are added; the plate does not change)."""
    key = (region.wkb, tuple((round(float(c[0]), 3), round(float(c[1]), 3), round(float(r), 3)) for c, r in supports),
           round(float(load), 3), round(float(q), 9), float(patch_r))
    if key not in _CACHE:
        if len(_CACHE) > 16:
            _CACHE.clear()
        _CACHE[key] = solve_plate(region, supports, load, q, patch_r)
    return _CACHE[key]


def solve_plate(region, supports: list[tuple[np.ndarray, float]], load: float, q: float,
                patch_r: float = 50.0, h: float | None = None, probe_step: float | None = None,
                nu: float = 0.22) -> PlateField:
    """Moments of a uniform plate over `region` (shapely polygon), held over the
    discs `supports` [(centre, radius)], under a uniform pressure q (N/mm^2)
    and the patch load `load` (N, a disc of radius patch_r) placed in turn on
    a grid of points and around the rim."""
    from shapely import contains_xy
    x0, y0, x1, y1 = region.bounds
    span = max(x1 - x0, y1 - y0)
    if h is None:
        h = max(15.0, span / 34.0)
    nx = int(math.ceil((x1 - x0) / h)) + 1
    ny = int(math.ceil((y1 - y0) / h)) + 1
    # centre the grid on the region's box so symmetric outlines mesh symmetrically
    ox = 0.5 * (x0 + x1) - 0.5 * nx * h
    oy = 0.5 * (y0 + y1) - 0.5 * ny * h
    ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx = ox + (ci + 0.5) * h
    cy = oy + (cj + 0.5) * h
    act = contains_xy(region, cx, cy)
    ei, ej = ci[act], cj[act]
    xs = np.stack([cx[act], cy[act]], axis=1)
    ne = len(ei)
    Ke, Fe, Mc = element_matrices(h, nu)
    # node numbering over the (nx+1) x (ny+1) corner grid, 4 dofs each
    nid = lambda i, j: i * (ny + 1) + j
    corners = np.stack([nid(ei, ej), nid(ei + 1, ej), nid(ei + 1, ej + 1), nid(ei, ej + 1)], axis=1)
    used = np.unique(corners)
    remap = -np.ones((nx + 1) * (ny + 1), dtype=np.int64)
    remap[used] = np.arange(len(used))
    cn = remap[corners]
    dofs = (4 * cn[:, :, None] + np.arange(4)[None, None, :]).reshape(ne, 16)
    ndof = 4 * len(used)
    rows = np.repeat(dofs, 16, axis=1).ravel()
    cols = np.tile(dofs, (1, 16)).ravel()
    vals = np.tile(Ke.ravel(), ne)
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    # supports: every node over a capital disc is held (w and slopes): a column head
    ui = used // (ny + 1)
    uj = used % (ny + 1)
    px, py = ox + ui * h, oy + uj * h
    held = np.zeros(len(used), dtype=bool)
    for c, r in supports:
        held |= (px - c[0]) ** 2 + (py - c[1]) ** 2 <= r * r
    if not held.any():
        # no node inside a disc: hold the nearest node to each support
        for c, r in supports:
            held[int(np.argmin((px - c[0]) ** 2 + (py - c[1]) ** 2))] = True
    hd = (4 * np.nonzero(held)[0][:, None] + np.arange(3)[None, :]).ravel()
    pen = 1e6 * float(Ke.diagonal().max())
    K = K + coo_matrix((np.full(len(hd), pen), (hd, hd)), shape=(ndof, ndof)).tocsr()

    def load_vec(weights):
        f = np.zeros(ndof)
        np.add.at(f, dofs.ravel(), (weights[:, None] * Fe[None, :]).ravel())
        return f

    cases = [load_vec(np.full(ne, q))]
    # patch load positions: a grid over the top and a ring just inside the rim
    step = probe_step or max(2.5 * h, span / 10.0)
    gx = np.arange(x0 + 0.5 * step, x1, step)
    gy = np.arange(y0 + 0.5 * step, y1, step)
    G = np.array([(a, b) for a in gx for b in gy])
    probes = [g for g in G if contains_xy(region, g[0], g[1])]
    ring = region.buffer(-0.6 * patch_r)
    if not ring.is_empty and ring.geom_type == "Polygon":
        L = ring.exterior.length
        for s in np.arange(0.0, L, step):
            pt = ring.exterior.interpolate(s)
            probes.append(np.array([pt.x, pt.y]))
    area_e = h * h
    for c in probes:
        w = ((xs[:, 0] - c[0]) ** 2 + (xs[:, 1] - c[1]) ** 2 <= patch_r ** 2).astype(float)
        if w.sum() == 0:
            w[int(np.argmin((xs[:, 0] - c[0]) ** 2 + (xs[:, 1] - c[1]) ** 2))] = 1.0
        cases.append(load_vec(w * (load / (w.sum() * area_e))))
    F = np.stack(cases, axis=1)
    from .linalg import solve_many
    U = solve_many(K, F)                              # (ndof, cases); D = 1 cancels for moments
    Ue = U[dofs, :]                                   # (ne, 16, cases)
    M = np.einsum("rk,ekc->cer", Mc, Ue)              # (cases, ne, 3)
    return PlateField(xs=xs, M=M, h=h, area=float(ne * area_e), origin=(ox, oy), shape=(nx, ny),
                      ij=np.stack([ei, ej], axis=1))


def directional_demand(M: np.ndarray, theta: float, family: int = 2) -> np.ndarray:
    """Design moments per unit width (enveloped over load cases) carried by
    each strip family of a grid at angle theta: (n_elements, family).

    Two families (u, v = u + 90 deg): Wood-Armer m = |M_uu| + |M_uv|.
    Three families 60 deg apart: the moment tensor is split exactly,
    M = sum_k m_k u_k u_k^T, and each family carries |m_k|."""
    Mxx, Myy, Mxy = M[..., 0], M[..., 1], M[..., 2]
    if family == 2:
        c, s = math.cos(theta), math.sin(theta)
        Muu = Mxx * c * c + Myy * s * s + 2 * Mxy * s * c
        Mvv = Mxx * s * s + Myy * c * c - 2 * Mxy * s * c
        Muv = (Myy - Mxx) * s * c + Mxy * (c * c - s * s)
        d = np.stack([np.abs(Muu) + np.abs(Muv), np.abs(Mvv) + np.abs(Muv)], axis=-1)
    else:
        us = [(math.cos(theta + k * math.pi / 3), math.sin(theta + k * math.pi / 3)) for k in range(3)]
        A = np.array([[ux * ux, uy * uy, ux * uy] for ux, uy in us]).T     # columns: families
        Ainv = np.linalg.inv(A)
        m = np.einsum("kr,...r->...k", Ainv, np.stack([Mxx, Myy, Mxy], axis=-1))
        d = np.abs(m)
    return d.max(axis=0)


def total_demand(field: PlateField, theta: float, family: int = 2) -> float:
    """Integral of the families' design moments: how much strip the top needs
    with the grid at angle theta."""
    return float(directional_demand(field.M, theta, family).sum() * field.h * field.h)


class PlateMesh:
    """A plate on a regular grid of BFS elements over a polygon, for coupling
    into a larger stiffness system: element matrices, point interpolation,
    pressure loads and moment recovery.  Degrees of freedom are numbered from
    `offset` (4 per node: w, w_x, w_y, w_xy)."""

    def __init__(self, region, D: float, h: float | None = None, offset: int = 0, nu: float = 0.22):
        from shapely import contains_xy
        x0, y0, x1, y1 = region.bounds
        span = max(x1 - x0, y1 - y0)
        self.h = h = float(h or max(15.0, span / 30.0))
        nx = int(math.ceil((x1 - x0) / h)) + 1
        ny = int(math.ceil((y1 - y0) / h)) + 1
        self.ox = 0.5 * (x0 + x1) - 0.5 * nx * h
        self.oy = 0.5 * (y0 + y1) - 0.5 * ny * h
        self.nx, self.ny = nx, ny
        ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cx = self.ox + (ci + 0.5) * h
        cy = self.oy + (cj + 0.5) * h
        act = contains_xy(region, cx, cy)
        self.ei, self.ej = ci[act], cj[act]
        self.xs = np.stack([cx[act], cy[act]], axis=1)
        self.ne = len(self.ei)
        self.elem_of = -np.ones((nx, ny), dtype=np.int64)
        self.elem_of[self.ei, self.ej] = np.arange(self.ne)
        Ke, Fe, Mc = element_matrices(h, nu)
        self.D = D
        self.Ke, self.Fe, self.Mc = D * Ke, Fe, D * Mc
        nid = lambda i, j: i * (ny + 1) + j
        corners = np.stack([nid(self.ei, self.ej), nid(self.ei + 1, self.ej),
                            nid(self.ei + 1, self.ej + 1), nid(self.ei, self.ej + 1)], axis=1)
        used = np.unique(corners)
        remap = -np.ones((nx + 1) * (ny + 1), dtype=np.int64)
        remap[used] = np.arange(len(used))
        cn = remap[corners]
        self.nn = len(used)
        self.node_xy = np.stack([self.ox + (used // (ny + 1)) * h, self.oy + (used % (ny + 1)) * h], axis=1)
        self.offset = offset
        self.dofs = offset + (4 * cn[:, :, None] + np.arange(4)[None, None, :]).reshape(self.ne, 16)
        self.ndof = 4 * self.nn

    def triplets(self):
        rows = np.repeat(self.dofs, 16, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 16)).ravel()
        vals = np.tile(self.Ke.ravel(), self.ne)
        return rows, cols, vals

    def element_at(self, xy):
        i = int(math.floor((xy[0] - self.ox) / self.h))
        j = int(math.floor((xy[1] - self.oy) / self.h))
        if 0 <= i < self.nx and 0 <= j < self.ny and self.elem_of[i, j] >= 0:
            return int(self.elem_of[i, j])
        # off the mesh (a hole or the stepped edge): the nearest element
        return int(np.argmin(((self.xs - np.asarray(xy)) ** 2).sum(axis=1)))

    def w_row(self, xy):
        """(dofs, coefficients) giving the deflection w at a plan point."""
        e = self.element_at(xy)
        xl = min(max(xy[0] - (self.ox + self.ei[e] * self.h), 0.0), self.h)
        yl = min(max(xy[1] - (self.oy + self.ej[e] * self.h), 0.0), self.h)
        return self.dofs[e], _shape(xl, yl, self.h)[0]

    def pressure(self, f: np.ndarray, weights: np.ndarray):
        """Add the consistent load of a pressure (N/mm^2) per element to f."""
        np.add.at(f, self.dofs.ravel(), (weights[:, None] * self.Fe[None, :]).ravel())

    def patch(self, f: np.ndarray, xy, P: float, radius: float):
        """A downward load P (N) spread evenly over a disc."""
        r2 = ((self.xs - np.asarray(xy)) ** 2).sum(axis=1)
        w = (r2 <= radius * radius).astype(float)
        if w.sum() == 0:
            w[int(np.argmin(r2))] = 1.0
        self.pressure(f, -w * (P / (w.sum() * self.h * self.h)))

    def moments(self, u: np.ndarray) -> np.ndarray:
        """(ne, 3): Mxx, Myy, Mxy (N mm / mm) at the element centres."""
        return np.einsum("rk,ek->er", self.Mc, u[self.dofs])

    def node_w(self, u: np.ndarray) -> np.ndarray:
        return u[self.offset + 4 * np.arange(self.nn)]
