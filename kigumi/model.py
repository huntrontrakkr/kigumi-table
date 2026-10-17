"""Parts: nominal solids (before joinery), plan footprints, frames and materials."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .m3d import m3d
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.polygon import orient

from .layout import Beam, Block, Layout, Leg
from .profiles import (beam_profile, cap_profile, leg_stations, masu_stations, rosette,
                       section_radius)

SEG = 96


@dataclass
class Part:
    id: str
    kind: str
    group: str
    material: str
    order: float
    footprint: Polygon
    zmin: float
    zmax: float
    frame: np.ndarray
    make: Callable[[], m3d.Manifold]
    meta: dict = field(default_factory=dict)
    nominal: m3d.Manifold | None = None
    solid: m3d.Manifold | None = None


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------
def loft(rings: list[np.ndarray], zs: list[float]) -> m3d.Manifold:
    """Closed solid through horizontal rings (each (N, 2), counter-clockwise)."""
    N = len(rings[0])
    nz = len(rings)
    V = np.zeros((nz * N + 2, 3))
    for i, (r, z) in enumerate(zip(rings, zs)):
        V[i * N:(i + 1) * N, :2] = r
        V[i * N:(i + 1) * N, 2] = z
    b = nz * N
    t = nz * N + 1
    V[b] = [*rings[0].mean(axis=0), zs[0]]
    V[t] = [*rings[-1].mean(axis=0), zs[-1]]
    tris = []
    j = np.arange(N)
    jn = (j + 1) % N
    for i in range(nz - 1):
        a0 = i * N + j
        a1 = i * N + jn
        b0 = (i + 1) * N + j
        b1 = (i + 1) * N + jn
        tris.append(np.c_[a0, a1, b1])
        tris.append(np.c_[a0, b1, b0])
    tris.append(np.c_[np.full(N, b), jn, j])
    top = (nz - 1) * N
    tris.append(np.c_[np.full(N, t), top + j, top + jn])
    T = np.vstack(tris).astype(np.uint32)
    mesh = m3d.Mesh(vert_properties=V.astype(np.float32), tri_verts=T)
    return m3d.Manifold(mesh)


def cs_from_polygon(poly) -> m3d.CrossSection:
    if poly.geom_type == "MultiPolygon":
        polys = list(poly.geoms)
    else:
        polys = [poly]
    contours = []
    for g in polys:
        g = orient(g, 1.0)
        contours.append(np.asarray(g.exterior.coords)[:-1])
        for hole in g.interiors:
            contours.append(np.asarray(hole.coords)[:-1])
    return m3d.CrossSection(contours, m3d.FillRule.Positive)


def prism(poly, z0: float, z1: float) -> m3d.Manifold:
    return cs_from_polygon(poly).extrude(z1 - z0).translate((0, 0, z0))


def frame_matrix(xaxis2d: np.ndarray, origin3d) -> np.ndarray:
    u = np.array([xaxis2d[0], xaxis2d[1], 0.0])
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0], 0.0])
    F = np.eye(4)
    F[:3, 0] = u
    F[:3, 1] = n
    F[:3, 2] = [0, 0, 1]
    F[:3, 3] = origin3d
    return F


def vertical_frame(xy, z0: float) -> np.ndarray:
    """Local X (grain) up the world Z axis."""
    F = np.eye(4)
    F[:3, 0] = [0, 0, 1]
    F[:3, 1] = [0, 1, 0]
    F[:3, 2] = [-1, 0, 0]
    F[:3, 3] = [xy[0], xy[1], z0]
    return F


# ---------------------------------------------------------------------------
# nominal solids
# ---------------------------------------------------------------------------
def beam_solid(B: Beam) -> m3d.Manifold:
    prof = beam_profile(B.t0, B.t1, B.h, B.ends)
    s = m3d.CrossSection([prof]).extrude(B.w)
    u3 = np.array([B.u[0], B.u[1], 0.0])
    n3 = np.array([B.n[0], B.n[1], 0.0])
    M = np.zeros((3, 4))
    M[:, 0] = u3
    M[:, 1] = [0, 0, 1]
    M[:, 2] = -n3
    M[:, 3] = np.array([B.origin[0], B.origin[1], B.z0]) + 0.5 * B.w * n3
    return s.transform(M)


def leg_spine(L: Leg, st: dict, sweep: float, swirl: float, splay: str = "tilt"):
    """Centre of the leg's section at each station.  'tilt': a straight axis
    from the foot to the capital (a turned leg set at a splay angle); 'bend':
    the stem curves from the foot into a vertical neck.  Sweep and swirl bow
    the stem outward and sideways between the foot and the neck."""
    r = float(np.hypot(*L.xy))
    out = L.xy / r if r > 1e-6 else np.array([1.0, 0.0])
    side = np.array([-out[1], out[0]])
    z_top = float(st["z"].max())
    cs = []
    for z, le, bw in zip(st["z"], st["lean"], st["bow"]):
        w = (1.0 - z / z_top) if splay == "tilt" else le
        cs.append(L.xy + w * (L.foot_xy - L.xy) + bw * (sweep * out + swirl * side))
    return np.array(cs)


def leg_solid(L: Leg, st: dict, kind: str, phase: float, sweep: float = 0.0, swirl: float = 0.0,
              splay: str = "tilt") -> m3d.Manifold:
    th = np.linspace(0, 2 * np.pi, SEG, endpoint=False)
    rings = []
    spine = leg_spine(L, st, sweep, swirl, splay)
    for c, z, r, sh in zip(spine, st["z"], st["r"], st["shape"]):
        rr = r * section_radius(th, kind, sh, phase)
        rings.append(np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)])
    return loft(rings, list(st["z"]))


def block_solid(b: Block, slot: float) -> m3d.Manifold:
    zs, half = masu_stations(b.z0, b.z1, b.size, slot)
    u = b.u / np.linalg.norm(b.u)
    n = np.array([-u[1], u[0]])
    rings = []
    for hw in half:
        corners = [b.xy + hw * (u + n), b.xy + hw * (-u + n), b.xy + hw * (-u - n), b.xy + hw * (u - n)]
        rings.append(np.array(corners))
    return loft(rings, list(zs))


def post_solid(xy, z0: float, z1: float, r: float) -> m3d.Manifold:
    ch = min(1.5, 0.3 * r)
    body = m3d.Manifold.cylinder(z1 - z0 - ch, r, r, 48).translate((0, 0, z0))
    tip = m3d.Manifold.cylinder(ch, r, r - ch, 48).translate((0, 0, z1 - ch))
    return (body + tip).translate((float(xy[0]), float(xy[1]), 0))


def cap_solid(style: str, xy, z0: float, R: float, H: float) -> m3d.Manifold:
    if style in ("dome", "mushroom", "hoju"):
        prof = cap_profile(style, R, H)
        s = m3d.Manifold.revolve(m3d.CrossSection([prof]), SEG)
    elif style == "rokuyo":
        ros = rosette(R)
        plate_h = 0.36 * H
        rings = [ros * 0.965, ros, ros, ros * 0.93]
        zs = [0.0, 1.2, plate_h - 1.6, plate_h]
        plate = loft(rings, zs)
        boss = m3d.Manifold.revolve(m3d.CrossSection([cap_profile("dome", 0.52 * R, H)]), SEG)
        s = plate + boss
    elif style == "octagon":
        th = np.arange(8) * np.pi / 4 + np.pi / 8
        k = 1.0 / np.cos(np.pi / 8)
        def octo(r):
            return np.c_[r * k * np.cos(th), r * k * np.sin(th)]
        body_h = 0.42 * H
        rings = [octo(0.96 * R), octo(R), octo(R), octo(1.12 * R), octo(1.12 * R)]
        zs = [0.0, 1.0, body_h, body_h, body_h + 2.5]
        roof_top = 0.9 * H
        for s_ in np.linspace(0, 1, 9)[1:]:
            rings.append(octo(max(1.12 * R * (1 - s_) ** 1.7, 0.6)))
            zs.append(body_h + 2.5 + (roof_top - body_h - 2.5) * s_)
        s = loft(rings, zs)
        ball = m3d.Manifold.sphere(0.13 * R, 32).translate((0, 0, roof_top + 0.07 * R))
        s = s + ball
    else:
        raise ValueError(style)
    return s.translate((float(xy[0]), float(xy[1]), z0))


# ---------------------------------------------------------------------------
# parts from a layout
# ---------------------------------------------------------------------------
def make_parts(lay: Layout) -> list[Part]:
    p = lay.p
    parts: list[Part] = []
    r_post = 0.5 * p.post_d

    # legs (+ integral posts)
    for L in lay.legs:
        pL = p.replace(**L.knobs) if getattr(L, "knobs", None) else p
        st = leg_stations(pL, L.z_top, L.z_slot, L.r_cap)
        phase = 0.0
        t0 = [B for B in lay.beams if B.tier == 0 and L.idx in B.legs]
        if t0:
            phase = math.atan2(t0[0].u[1], t0[0].u[0])
        foot = Point(*L.foot_xy).buffer(st["r_foot"] * 1.02, quad_segs=16)
        cap = Point(*L.xy).buffer(st["r_cap"] * (1.42 if p.leg_capital != "round" else 1.02), quad_segs=16)
        fp = MultiPoint(list(foot.exterior.coords) + list(cap.exterior.coords)).convex_hull
        g = f"leg{L.idx}"
        parts.append(Part(
            id=g, kind="leg", group=g, material=p.wood_legs, order=0.0 + 0.001 * L.idx,
            footprint=fp, zmin=0.0, zmax=L.z_top,
            frame=vertical_frame(L.xy, 0.0),
            make=(lambda L=L, st=st, phase=phase, pL=pL: leg_solid(L, st, pL.leg_capital, phase, pL.leg_sweep,
                                                                   pL.leg_swirl, pL.leg_splay)),
            meta=dict(stations=st, leg=L.idx, phase=phase)))
        dowel = p.post_mode == "dowel"
        z0 = L.z_slot - (p.dowel_engage if dowel else 14.0)
        parts.append(Part(
            id=f"post{L.idx}", kind="post", group=(f"post{L.idx}" if dowel else g),
            material=(p.post_wood if dowel else p.wood_legs), order=0.0005 + 0.001 * L.idx,
            footprint=Point(*L.xy).buffer(r_post, quad_segs=16), zmin=z0, zmax=lay.post_top,
            frame=vertical_frame(L.xy, z0),
            make=(lambda L=L, z0=z0: post_solid(L.xy, z0, lay.post_top, r_post)),
            meta=dict(leg=L.idx)))

    # beams
    for B in lay.beams:
        mid = B.origin + 0.5 * (B.t0 + B.t1) * B.u
        parts.append(Part(
            id=B.id, kind="beam", group=B.id, material=p.wood_beams, order=B.order,
            footprint=B.strip(), zmin=B.z0, zmax=B.z1,
            frame=frame_matrix(B.u, [mid[0], mid[1], B.z0]),
            make=(lambda B=B: beam_solid(B)),
            meta=dict(tier=B.tier, family=B.family, length=B.length, legs=list(B.legs),
                      angle=B.angle_deg(), ends=B.ends)))

    # masu blocks
    for b in lay.blocks:
        slot = 0.25 * p.beam_h if b.slot_into else 0.0
        u = b.u / np.linalg.norm(b.u)
        n = np.array([-u[1], u[0]])
        hw = 0.5 * b.size
        fp = Polygon([b.xy + hw * (u + n), b.xy + hw * (-u + n), b.xy + hw * (-u - n), b.xy + hw * (u - n)])
        parts.append(Part(
            id=b.id, kind="block", group=b.id, material=p.wood_beams, order=b.order,
            footprint=fp, zmin=b.z0, zmax=b.z1,
            frame=frame_matrix(u, [b.xy[0], b.xy[1], b.z0]),
            make=(lambda b=b, slot=slot: block_solid(b, slot)),
            meta=dict(leg=b.leg, slot_into=b.slot_into)))

    # bumpers and washers between the lattice and the glass
    for d in lay.pads:
        xy = d["xy"]
        fp = Point(*xy).buffer(d["r"], quad_segs=12)
        if d["kind"] == "washer":
            make = (lambda d=d: (m3d.Manifold.cylinder(d["z1"] - d["z0"], d["r"], d["r"], 48)
                                 .translate((float(d["xy"][0]), float(d["xy"][1]), d["z0"]))))
            mat = "rubber"
        else:
            make = (lambda d=d: (m3d.Manifold.cylinder(d["z1"] - d["z0"], d["r"], d["r"] - 0.6, 24)
                                 .translate((float(d["xy"][0]), float(d["xy"][1]), d["z0"]))))
            mat = "silicone"
        parts.append(Part(
            id=d["id"], kind=d["kind"], group=d["id"], material=mat, order=900.0,
            footprint=fp, zmin=d["z0"], zmax=d["z1"],
            frame=frame_matrix(np.array([1.0, 0.0]), [xy[0], xy[1], d["z0"]]),
            make=make, meta=dict(leg=d.get("leg"))))

    if p.glass_kind == "none":
        R = 0.5 * p.cap_d
        for L in lay.legs:
            g = f"cap{L.idx}"
            Rfp = R * (1.13 * 1.0824 if p.cap_style == "octagon" else 1.0)
            parts.append(Part(
                id=g, kind="cap", group=g, material=p.wood_caps, order=1002.0 + 0.001 * L.idx,
                footprint=Point(*L.xy).buffer(Rfp, quad_segs=16), zmin=lay.cap_z0, zmax=lay.cap_z0 + p.cap_h,
                frame=vertical_frame(L.xy, lay.cap_z0),
                make=(lambda L=L: cap_solid(p.cap_style, L.xy, lay.cap_z0, R, p.cap_h)),
                meta=dict(leg=L.idx)))
        return parts

    # glass (its footprint already excludes the post holes, so bushings can sit in them)
    holes = [Point(*L.xy).buffer(r_post + p.glass_hole_clear, quad_segs=24) for L in lay.legs]
    glass_fp = lay.outline
    for hl in holes:
        glass_fp = glass_fp.difference(hl)
    parts.append(Part(
        id="top", kind="glass", group="top", material=p.glass_kind, order=1000.0,
        footprint=glass_fp, zmin=lay.z_glass0, zmax=lay.z_glass1,
        frame=frame_matrix(np.array([1.0, 0.0]), [0, 0, lay.z_glass0]),
        make=(lambda glass_fp=glass_fp: prism(glass_fp, lay.z_glass0, lay.z_glass1))))

    # bushings and caps
    R = 0.5 * p.cap_d
    for L in lay.legs:
        r_sl = r_post + p.glass_hole_clear - 0.3
        r_fl = max(R - 2.5, r_sl + 3)
        g = f"bush{L.idx}"
        parts.append(Part(
            id=g, kind="bushing", group=g, material="rubber", order=1001.0 + 0.001 * L.idx,
            footprint=Point(*L.xy).buffer(r_fl, quad_segs=16), zmin=lay.z_glass0 + 1.0,
            zmax=lay.cap_z0,
            frame=frame_matrix(np.array([1.0, 0.0]), [L.xy[0], L.xy[1], lay.z_glass1]),
            make=(lambda L=L, r_sl=r_sl, r_fl=r_fl: (
                m3d.Manifold.cylinder(lay.cap_z0 - lay.z_glass1, r_fl, r_fl, 64).translate((0, 0, lay.z_glass1))
                + m3d.Manifold.cylinder(lay.z_glass1 - lay.z_glass0 - 1.0, r_sl, r_sl, 64)
                .translate((0, 0, lay.z_glass0 + 1.0))).translate((float(L.xy[0]), float(L.xy[1]), 0)))))
        g = f"cap{L.idx}"
        Rfp = R * (1.13 * 1.0824 if p.cap_style == "octagon" else 1.0)
        parts.append(Part(
            id=g, kind="cap", group=g, material=p.wood_caps, order=1002.0 + 0.001 * L.idx,
            footprint=Point(*L.xy).buffer(Rfp, quad_segs=16), zmin=lay.cap_z0,
            zmax=lay.cap_z0 + p.cap_h,
            frame=vertical_frame(L.xy, lay.cap_z0),
            make=(lambda L=L: cap_solid(p.cap_style, L.xy, lay.cap_z0, R, p.cap_h)),
            meta=dict(leg=L.idx)))
    return parts
