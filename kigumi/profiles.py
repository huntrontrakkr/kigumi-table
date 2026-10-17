"""Profiles: mushroom-stem legs, beam elevations with curved ends, masu blocks, caps."""
from __future__ import annotations

import math

import numpy as np

from .params import Params


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------
def leg_stations(p: Params, z_top: float, z_slot: float, r_cap: float | None = None) -> dict:
    """Radius and section shape of the leg along its height.

    Stem: a straight taper from the foot to the neck, blended with an optional
    mushroom flare (leg_flare_mix), a movable belly bump (leg_belly > 0 gives
    gourds and spindles, < 0 a waist), a very slight sine bow and an optional
    bun bulge at the foot.  The foot edge is rounded by a rolling-ball fillet:
    the circle of radius leg_foot_round * r_foot that touches both the floor
    and the stem curve, whatever its shape.  The capital flares out (trumpet
    curve) into a straight band that carries the slots for the beams.
    """
    r_b = 0.5 * p.leg_base_d
    r_t = 0.5 * p.leg_top_d
    r_c = r_cap if r_cap is not None else 0.5 * p.leg_cap_d
    band = (z_top - z_slot) + 10.0
    z_band0 = z_top - band
    flare_h = max(0.0, p.leg_cap_flare)
    z_neck = z_band0 - flare_h
    if z_neck < 0.35 * z_top:
        z_neck = 0.35 * z_top
        flare_h = max(0.0, z_band0 - z_neck)
    lam = max(1.0, p.leg_flare * z_neck)
    k = p.leg_flare_k
    mix = min(max(p.leg_flare_mix, 0.0), 1.0)

    def stem(z):
        z = np.asarray(z, float)
        u = np.clip(z / z_neck, 0.0, 1.0)
        lin = r_b + (r_t - r_b) * u
        flare = r_t + (r_b - r_t) * np.exp(-np.power(np.maximum(z, 0) / lam, k))
        r = (1 - mix) * lin + mix * flare
        r = r + p.leg_bow * r_t * np.sin(np.pi * u)
        if p.leg_belly:
            r = r + p.leg_belly * r_b * np.exp(-((u - p.leg_belly_at) / max(p.leg_belly_w, 1e-3)) ** 2)
        if p.leg_foot_bulge > 0:
            zc = 0.5 * p.leg_foot_bulge_w * r_b
            wz = max(0.6 * p.leg_foot_bulge_w * r_b, 1.0)
            r = r + p.leg_foot_bulge * r_b * np.exp(-((z - zc) / wz) ** 2)
        return np.maximum(r, 4.0)

    # stem samples, dense near the foot where the fillet lands
    zs_stem = np.unique(np.concatenate([np.linspace(0.0, 0.5 * z_neck, 90), np.linspace(0.5 * z_neck, z_neck, 40)]))
    rs_stem = stem(zs_stem)
    r_max_low = float(rs_stem[zs_stem < 0.6 * z_neck].max())
    pts = np.c_[rs_stem, zs_stem]

    def fillet(rho):
        """Rolling ball of radius rho touching the floor and the stem curve:
        returns the ball centre's radius c and the index of the touch point."""
        def dist(c):
            return float(np.min(np.hypot(pts[:, 0] - c, pts[:, 1] - rho)))
        lo_c, hi_c = -10.0 * r_b, float(rs_stem.max())
        for _ in range(60):
            mid = 0.5 * (lo_c + hi_c)
            if dist(mid) > rho:
                lo_c = mid
            else:
                hi_c = mid
        c = 0.5 * (lo_c + hi_c)
        return c, int(np.argmin(np.hypot(pts[:, 0] - c, pts[:, 1] - rho)))

    rho = min(p.leg_foot_round * r_max_low, 0.4 * z_neck)
    # the foot must keep a flat at least 35% of its radius wide, or it rocks
    flat_min = 0.35 * r_b
    arc_r = arc_z = None
    z_join = 0.0
    if rho > 0.5:
        c, j = fillet(rho)
        if c < flat_min:
            lo_r, hi_r = 0.0, rho
            for _ in range(40):
                mid = 0.5 * (lo_r + hi_r)
                if fillet(mid)[0] < flat_min:
                    hi_r = mid
                else:
                    lo_r = mid
            rho = lo_r
            c, j = fillet(rho)
    if rho > 0.5:
        tx, tz = pts[j]
        phi_t = math.atan2(tz - rho, tx - c)
        phis = np.linspace(-math.pi / 2, phi_t, 32)
        arc_r = c + rho * np.cos(phis)
        arc_z = rho + rho * np.sin(phis)
        z_join = float(tz)
    keep = zs_stem > z_join + 1e-6
    zu = list(zs_stem[keep])
    if flare_h > 0.5:
        zu += list(np.linspace(z_neck, z_band0, 22)[1:])
    zu += list(np.linspace(z_band0, z_top - 1.5, 6)[1:]) + [z_top]
    zu = np.array(zu)
    ru = stem(np.minimum(zu, z_neck))
    r_neck = float(stem(z_neck))
    m = (zu > z_neck) & (zu <= z_band0)
    if flare_h > 0.5:
        s_ = (zu[m] - z_neck) / flare_h
        ru[m] = r_neck + (r_c - r_neck) * s_ ** 2.2
    ru[zu > z_band0] = r_c
    ru[-1] = r_c - 1.5            # top chamfer
    if arc_r is not None:
        z = np.concatenate([arc_z, zu])
        r = np.concatenate([arc_r, ru])
    else:
        z = np.concatenate([[0.0], zu])
        r = np.concatenate([[float(stem(0.0))], ru])
    order = np.argsort(z, kind="stable")
    z, r = z[order], r[order]
    keep = np.concatenate([[True], np.diff(z) > 1e-6])
    z, r = z[keep], r[keep]
    if p.leg_collar > 0:
        zc = z_neck - 1.2 * p.leg_collar
        cc = p.leg_collar
        mm = np.abs(z - zc) < cc
        r[mm] = np.maximum(r[mm], stem(zc) + cc * np.sqrt(1 - ((z[mm] - zc) / cc) ** 2))
    shape = np.zeros_like(z)
    if p.leg_capital != "round":
        if flare_h > 0.5:
            mm = (z > z_neck) & (z <= z_band0)
            shape[mm] = ((z[mm] - z_neck) / flare_h) ** 1.5
        shape[z > z_band0] = 1.0
    u = np.clip(z / z_neck, 0, 1)
    lean = 1 - (3 * u ** 2 - 2 * u ** 3)      # 1 at the foot, 0 at the neck
    bow = np.sin(np.pi * u)                   # 0 at both ends: the stem bows between foot and neck
    return dict(z=z, r=r, shape=shape, lean=lean, bow=bow, z_neck=z_neck, z_band0=z_band0,
                r_neck=r_neck, r_cap=r_c, r_foot=float(r.max() if len(r) else r_b),
                r_contact=float(r[0]))


def section_radius(theta: np.ndarray, kind: str, blend: float, phase: float = 0.0) -> np.ndarray:
    """Radial distance of a unit-inradius section (circle, square or octagon)."""
    if kind == "round" or blend <= 0:
        return np.ones_like(theta)
    th = theta - phase
    if kind == "square":
        poly = 1.0 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    else:  # octagon
        c = np.max(np.abs(np.stack([np.cos(th - i * np.pi / 4) for i in range(4)])), axis=0)
        poly = 1.0 / c
    return (1 - blend) + blend * poly


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------
def _end_points(t_end: float, sign: int, h: float, e: dict, n: int = 18):
    """One free end of a beam.  Returns (bottom, nose, top):
    bottom -- bottom-edge points from the curve start outward to the tip,
    nose   -- end-face points from bottom to top,
    top    -- top-edge points from the tip inward (a dropped tail top slopes)."""
    style, L, rise, pe = e["style"], e["L"], e["rise"], e["p"]
    drop = float(e.get("drop", 0.0))
    if style == "plain" or L <= 0:
        return [(t_end, 0.0)], [], [(t_end, h)]
    z_tip = rise * h
    rn = 0.55 * 0.5 * (h - z_tip) if style == "scroll" else 0.0
    s = np.linspace(0.0, 1.0, n)
    dz = z_tip * (s if style == "taper" else s ** pe)
    x = t_end - sign * (L - s * (L - rn))
    bottom = list(zip(x, dz))
    if style == "scroll":
        r = 0.5 * (h - z_tip)
        zc = z_tip + r
        a = np.linspace(-np.pi / 2, np.pi / 2, 15)
        nose = [(t_end - sign * rn + sign * rn * np.cos(ai), zc + r * np.sin(ai)) for ai in a]
        return bottom, nose, [(t_end - sign * rn, h)]
    if drop > 0:
        # a rafter tail: the top falls away from the glass toward the tip
        top_tip = min(h - drop, h - 0.5)
        top_tip = max(top_tip, z_tip + 1.5)
        top = [(t_end, top_tip), (t_end - sign * L, h)]
        return bottom, [(t_end, z_tip), (t_end, top_tip)], top
    return bottom, [(t_end, z_tip)], [(t_end, h)]


def beam_profile(t0: float, t1: float, h: float, ends: list) -> np.ndarray:
    """Side elevation polygon (u, z), z in [0, h], counter-clockwise:
    bottom left->right, right end up, top right->left, left end down."""
    lb, ln, ltop = _end_points(t0, -1, h, ends[0])
    rb, rn, rtop = _end_points(t1, +1, h, ends[1])
    # top edge right to left: right tip inward, then left inward point outward to its tip
    pts = list(reversed(lb)) + rb + rn + rtop + list(reversed(ltop)) + list(reversed(ln))
    out = []
    for q in pts:
        q = (float(q[0]), float(q[1]))
        if not out or abs(q[0] - out[-1][0]) > 1e-6 or abs(q[1] - out[-1][1]) > 1e-6:
            out.append(q)
    if abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return np.array(out)


# ---------------------------------------------------------------------------
# masu blocks
# ---------------------------------------------------------------------------
def masu_stations(z0: float, z1: float, size: float, slot: float):
    """Half-width of a masu (bearing block) by height: straight top band that
    holds the slot, concave taper below to 78% at the foot."""
    hgt = z1 - z0
    band = min(hgt, slot + 0.3 * max(hgt - slot, 0))
    zt = z1 - band
    zs = [z0]
    if zt - z0 > 1.0:
        zs += list(np.linspace(z0, zt, 10)[1:])
    zs += [z1 - 0.8, z1]
    zs = np.array(sorted(set(np.round(zs, 6))))
    half = np.empty_like(zs)
    for i, z in enumerate(zs):
        if z >= zt or zt - z0 <= 1.0:
            half[i] = 0.5 * size
        else:
            s = (z - z0) / (zt - z0)
            half[i] = 0.5 * size * (0.78 + 0.22 * max(s, 0.0) ** 0.55)
    half[-1] -= 0.8
    return zs, half


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------
def cap_profile(style: str, R: float, H: float) -> np.ndarray:
    """Revolve profile (r, z) for the round part of a cap, from the axis at the
    bottom, out and up, back to the axis at the top."""
    if style in ("dome", "rokuyo"):
        band = 0.42 * H
        pts = [(0, 0), (R - 1.2, 0), (R, 1.2), (R, band)]
        a = np.linspace(0, np.pi / 2, 20)[1:]
        pts += [(R * np.cos(ai), band + (H - band) * np.sin(ai)) for ai in a]
        pts[-1] = (0, H)
        return np.array(pts)
    if style == "mushroom":
        stem_r, stem_h = 0.55 * R, 0.30 * H
        pts = [(0, 0), (stem_r, 0), (stem_r, stem_h - 2), (R - 2, stem_h), (R, stem_h + 2)]
        a = np.linspace(0, np.pi / 2, 24)[1:]
        pts += [(R * np.cos(ai) ** 0.9, stem_h + 2 + (H - stem_h - 2) * np.sin(ai) ** 1.2) for ai in a]
        pts[-1] = (0, H)
        return np.array(pts)
    if style == "hoju":
        base_h = 0.22 * H
        pts = [(0, 0), (R, 0), (R, base_h * 0.6), (0.82 * R, base_h * 0.6), (0.82 * R, base_h)]
        # onion jewel with a pointed tip
        s = np.linspace(0, 1, 30)[1:]
        jr = 0.78 * R
        zz = base_h + (H - base_h) * s
        rr = jr * np.sin(np.pi * np.minimum(s, 0.999) ** 0.75) * (1 - s ** 6) ** 0.5
        rr = np.maximum(rr, 0)
        pts += list(zip(rr, zz))
        pts[-1] = (0, H)
        return np.array(pts)
    raise ValueError(style)


def rosette(R: float, lobes: int = 6, n: int = 192) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R * (0.80 + 0.20 * np.abs(np.cos(lobes * th / 2)) ** 0.7)
    return np.c_[r * np.cos(th), r * np.sin(th)]
