"""Joinery compiler.

Every part has a plan footprint, a height range and an assembly order (all
parts drop in vertically).  For each pair whose footprints and height ranges
overlap, one relation is chosen from the part kinds:

  split(z)  -- in the shared plan area the earlier part keeps what lies below
               z and the later part keeps what lies above.  Laps between beams
               split at the middle of the overlap; legs and masu blocks yield
               to the beams they carry or receive; caps sit on the glass.
  pin       -- the later part gets a vertical hole for the earlier one
               (posts through beams, blocks, glass, bushings).
  blind     -- like split at the top of the pin: the threaded blind hole in a cap.

Because the earlier part is always below the later one wherever they share
plan area, dropping parts in order can never collide, and no two finished
parts overlap -- both are checked again on the final solids.
"""
from __future__ import annotations

from dataclasses import dataclass

from .m3d import m3d

from .model import Part, prism
from .params import Params
from .shapes import overlap

BIG = 5000.0


@dataclass
class Relation:
    a: str            # earlier part
    b: str            # later part
    kind: str         # split | pin | blind
    z: float | None
    note: str
    area: float
    defect: str = ""


def decide(P: Part, Q: Part, p: Params) -> Relation | None:
    if P.group == Q.group:
        return None
    if not P.footprint.intersects(Q.footprint):
        return None
    zlo, zhi = max(P.zmin, Q.zmin), min(P.zmax, Q.zmax)
    if zhi - zlo <= 1e-6:
        return None
    inter = overlap(P.footprint, Q.footprint)
    area = inter.area
    if area < 0.5:
        return None
    k = (P.kind, Q.kind)
    if P.kind == "post":
        if Q.kind == "cap":
            return Relation(P.id, Q.id, "blind", P.zmax + 1.0, "threaded blind hole", area)
        rel = Relation(P.id, Q.id, "pin", None, "post hole", area)
        if Q.kind in ("beam", "block"):
            inner = Q.footprint.buffer(-3.0)
            if not inner.contains(P.footprint):
                rel.defect = "post grazes the edge of " + Q.id
        return rel
    if k == ("beam", "beam"):
        z = 0.5 * (zlo + zhi)
        note = "flush half-lap" if abs(P.zmin - Q.zmin) < 1e-6 else "cogged lap"
        return Relation(P.id, Q.id, "split", z, note, area)
    if P.kind == "leg" and Q.kind == "post":
        return Relation(P.id, Q.id, "split", Q.zmin, "threaded socket", area)
    if P.kind == "leg":
        return Relation(P.id, Q.id, "split", Q.zmin, "capital slot", area)
    if k == ("beam", "block"):
        return Relation(P.id, Q.id, "split", P.zmax, "masu seat", area)
    if k == ("block", "beam"):
        return Relation(P.id, Q.id, "split", Q.zmin, "masu slot", area)
    if Q.kind in ("glass", "bushing", "cap") or P.kind in ("glass", "bushing"):
        return Relation(P.id, Q.id, "split", P.zmax, "seat", area)
    return Relation(P.id, Q.id, "split", 0.5 * (zlo + zhi), "split", area)


def relations(parts: list[Part], p: Params) -> list[Relation]:
    order = sorted(parts, key=lambda q: q.order)
    bounds = {q.id: q.footprint.bounds for q in parts}
    out = []
    for i, P in enumerate(order):
        bp = bounds[P.id]
        for Q in order[i + 1:]:
            bq = bounds[Q.id]
            if bp[2] < bq[0] or bq[2] < bp[0] or bp[3] < bq[1] or bq[3] < bp[1]:
                continue
            r = decide(P, Q, p)
            if r is not None:
                out.append(r)
    share_nodes(out, parts)
    balance(out, parts)
    return out


def share_nodes(rels: list[Relation], parts: list[Part]) -> int:
    """Where k > 2 strips of one band cross at one spot, a half-lap would cut
    the middle strip away: give each strip an equal share of the depth there
    (split heights at i/k by assembly order).  Returns the number of nodes."""
    by_id = {q.id: q for q in parts}
    flush = [r for r in rels if r.kind == "split" and r.note.startswith("flush")]
    beams = [q for q in parts if q.kind == "beam"]
    n = 0
    for r in flush:
        P, Q = by_id[r.a], by_id[r.b]
        spot = overlap(P.footprint, Q.footprint)
        if spot.area < 1.0:
            continue
        group = [P, Q] + [R for R in beams if R is not P and R is not Q and abs(R.zmin - P.zmin) < 1e-6
                          and overlap(R.footprint, spot).area > 0.25 * spot.area]
        if len(group) <= 2:
            continue
        group.sort(key=lambda q: q.order)
        k = len(group)
        rank = group.index(P)
        h = P.zmax - P.zmin
        r.z = P.zmin + h * (rank + 1) / k
        r.note = f"flush {k}-way lap"
        n += 1
    return n


def balance(rels: list[Relation], parts: list[Part], target: float = 0.5, max_iter: int = 400) -> int:
    """Spread notch depth so no beam is starved at one spot.

    A *spot* is a place where a beam carries a top notch (it is the earlier
    part of one lap) and a bottom notch (the later part of another) over the
    same plan area; the wood left there is the gap between the two split
    heights.  Repeatedly take the thinnest spot in the whole lattice and try
    every allowed shift of the laps that make it -- a cogged lap may put its
    whole overlap into either partner, a flush half-lap may move 15% of the
    depth either way -- keeping a shift only if it raises the thinnest of all
    spots it touches.  Stops when every spot keeps `target` of its depth or no
    shift helps.  Returns the number of laps moved."""
    by_id = {q.id: q for q in parts}
    beam_rels = [r for r in rels if r.kind == "split" and by_id[r.a].kind == "beam" and by_id[r.b].kind == "beam"]
    if not beam_rels:
        return 0
    span = {}
    for r in beam_rels:
        P, Q = by_id[r.a], by_id[r.b]
        lo, hi = max(P.zmin, Q.zmin), min(P.zmax, Q.zmax)
        span[id(r)] = (lo, hi, 0.5 * (lo + hi))
    spots = []          # (beam, top relation or None, bottom relation or None)
    for X in [q for q in parts if q.kind == "beam"]:
        tops = [r for r in beam_rels if r.a == X.id]
        bots = [r for r in beam_rels if r.b == X.id]
        for rt in tops:
            spots.append((X, rt, None))
        for rb in bots:
            spots.append((X, None, rb))
        for rt in tops:
            reg_t = overlap(X.footprint, by_id[rt.b].footprint)
            for rb in bots:
                reg_b = overlap(X.footprint, by_id[rb.a].footprint)
                if overlap(reg_t, reg_b).area >= 1.0:
                    spots.append((X, rt, rb))
    if not spots:
        return 0
    involved: dict = {}
    for s in spots:
        for r in s[1:]:
            if r is not None:
                involved.setdefault(id(r), []).append(s)

    def depth(s):
        X, rt, rb = s
        top = rt.z if rt is not None else X.zmax
        bot = rb.z if rb is not None else X.zmin
        return (top - bot) / (X.zmax - X.zmin)

    def options(r):
        lo, hi, mid = span[id(r)]
        if r.note.startswith("flush"):
            h = hi - lo
            return [mid - 0.15 * h, mid, mid + 0.15 * h]
        return [lo, mid, hi]

    moved = set()
    stuck = set()
    for _ in range(max_iter):
        live = [s for s in spots if id(s) not in stuck]
        if not live:
            break
        worst = min(live, key=depth)
        d0 = depth(worst)
        if d0 >= target:
            break
        best = (d0, None, None)
        for r in (worst[1], worst[2]):
            if r is None:
                continue
            old = r.z
            for z in options(r):
                if abs(z - old) < 1e-9:
                    continue
                r.z = z
                m = min(depth(s) for s in involved[id(r)])
                if m > best[0] + 1e-9:
                    best = (m, r, z)
            r.z = old
        if best[1] is None:
            stuck.add(id(worst))
            continue
        best[1].z = best[2]
        moved.add(id(best[1]))
    for r in beam_rels:
        if id(r) in moved and not r.note.endswith("(balanced)"):
            r.note = r.note + " (balanced)"
    return len(moved)


def pin_clearance(kind: str, p: Params) -> float:
    if kind == "glass":
        return p.glass_hole_clear
    if kind == "bushing":
        return 0.25
    return p.post_clear


def apply(parts: list[Part], rels: list[Relation], p: Params) -> None:
    """Build nominal solids and subtract the cutters each relation implies."""
    by_id = {q.id: q for q in parts}
    cuts: dict[str, list[m3d.Manifold]] = {q.id: [] for q in parts}
    c = p.clearance
    for r in rels:
        P, Q = by_id[r.a], by_id[r.b]
        if r.kind == "pin":
            cuts[Q.id].append(prism(P.footprint.buffer(pin_clearance(Q.kind, p), quad_segs=16), -BIG, BIG))
        elif r.kind == "blind":
            cuts[Q.id].append(prism(P.footprint.buffer(0.3, quad_segs=16), -BIG, r.z))
        else:
            cuts[P.id].append(prism(Q.footprint.buffer(c, join_style="mitre"), r.z, BIG))
            cuts[Q.id].append(prism(P.footprint.buffer(c, join_style="mitre"), -BIG, r.z))
    for q in parts:
        q.nominal = q.make()
        if cuts[q.id]:
            cutter = m3d.Manifold.batch_boolean(cuts[q.id], m3d.OpType.Add)
            q.solid = q.nominal - cutter
        else:
            q.solid = q.nominal
