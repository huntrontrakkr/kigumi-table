"""Fabrication outputs for one design.

    export_all(design, folder) writes:
      table.3dm       Rhino 8 file: every part a named mesh on a layer (legs, tier N, ...)
      table.step      B-rep solids rebuilt with OpenCascade (same cuts as the meshes)
      table.glb/.stl  meshes
      glass.dxf       glass outline and holes, full size, for the glass shop
      beams.dxf       every beam's side profile (notches, curved ends), labelled
      cutlist.csv     stock list with quantities (identical beams grouped)
      drawings.pdf    overview, tier plans, beam elevations, full-size leg and cap profiles
      assembly.md     the drop-in order, step by step
      report.json     parameters, summary and the full engineering checks
"""
from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import ezdxf
from .m3d import m3d
import numpy as np

from .build import Design
from .export import PALETTE, export_glb, export_stl, part_mesh
from .profiles import beam_profile, cap_profile, leg_stations, masu_stations, rosette


# ---------------------------------------------------------------------------
# Rhino
# ---------------------------------------------------------------------------
def layer_name(q) -> str:
    if q.kind in ("leg", "post"):
        return "legs"
    if q.kind == "beam":
        return f"lattice::tier {q.meta['tier']}"
    if q.kind == "block":
        return "lattice::masu"
    if q.kind in ("glass",):
        return "glass"
    if q.kind in ("cap", "bushing", "washer", "pad"):
        return "caps and hardware"
    return "other"


def export_3dm(d: Design, path: Path) -> Path:
    import rhino3dm as r3
    f = r3.File3dm()
    f.Settings.ModelUnitSystem = r3.UnitSystem.Millimeters
    layers = {}

    def layer(name, rgb):
        if name in layers:
            return layers[name]
        parent = None
        if "::" in name:
            pname = name.split("::")[0]
            parent = layer(pname, rgb)
        L = r3.Layer()
        L.Name = name.split("::")[-1]
        L.Color = tuple(int(255 * c) for c in rgb) + (255,)
        if parent is not None:
            L.ParentLayerId = f.Layers[parent].Id
        layers[name] = f.Layers.Add(L)
        return layers[name]

    for q in d.parts:
        V, F = part_mesh(q, local=False)
        mesh = r3.Mesh()
        for x, y, z in V:
            mesh.Vertices.Add(float(x), float(y), float(z))
        for a, b, c in F:
            mesh.Faces.AddFace(int(a), int(b), int(c))
        mesh.Normals.ComputeNormals()
        mesh.Compact()
        rgb = PALETTE.get(q.material, (0.7, 0.7, 0.7))
        attr = r3.ObjectAttributes()
        attr.LayerIndex = layer(layer_name(q), rgb)
        attr.Name = q.id
        attr.SetUserString("kind", q.kind)
        attr.SetUserString("material", q.material)
        attr.SetUserString("assembly_order", f"{q.order:.3f}")
        if q.kind == "beam":
            attr.SetUserString("tier", str(q.meta["tier"]))
            attr.SetUserString("length_mm", f"{q.meta['length']:.1f}")
        attr.ObjectColor = tuple(int(255 * c) for c in rgb) + (255,)
        attr.ColorSource = r3.ObjectColorSource.ColorFromObject
        f.Objects.AddMesh(mesh, attr)
    f.Write(str(path), 8)
    return path


# ---------------------------------------------------------------------------
# STEP (B-rep) -- same nominal solids and cutters as the mesh pipeline
# ---------------------------------------------------------------------------
def _bd_poly_face(bd, pts2d, z: float = 0.0):
    pts = [(float(x), float(y), z) for x, y in pts2d]
    return bd.Face(bd.Wire.make_polygon(pts, close=True))


def _bd_prism(bd, poly, z0: float, z1: float):
    from shapely.geometry.polygon import orient
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    out = None
    for g in polys:
        g = orient(g, 1.0)
        outer = bd.Wire.make_polygon([(x, y, z0) for x, y in np.asarray(g.exterior.coords)[:-1]], close=True)
        inner = [bd.Wire.make_polygon([(x, y, z0) for x, y in np.asarray(h.coords)[:-1]], close=True)
                 for h in g.interiors]
        face = bd.Face(outer, inner) if inner else bd.Face(outer)
        s = bd.extrude(face, amount=z1 - z0)
        out = s if out is None else out + s
    return out


def _bd_revolve(bd, rz, origin_xy, z0: float = 0.0):
    pts = [(0.0, float(rz[0][1]))] + [(float(r), float(z)) for r, z in rz] + [(0.0, float(rz[-1][1]))]
    clean = []
    for q in pts:
        if not clean or abs(q[0] - clean[-1][0]) > 1e-6 or abs(q[1] - clean[-1][1]) > 1e-6:
            clean.append(q)
    face = _bd_poly_face(bd, clean)
    s = bd.revolve(face, axis=bd.Axis.Y)
    loc = bd.Location(bd.Plane(origin=(float(origin_xy[0]), float(origin_xy[1]), z0),
                               x_dir=(1, 0, 0), z_dir=(0, -1, 0)))
    return s.moved(loc)


def _bd_nominal(bd, q, d: Design):
    lay = d.layout
    p = d.p
    if q.kind == "beam":
        B = lay.beam(q.id)
        prof = beam_profile(B.t0, B.t1, B.h, B.ends)
        s = bd.extrude(_bd_poly_face(bd, prof), amount=B.w)
        u = (float(B.u[0]), float(B.u[1]), 0.0)
        n = np.array([B.n[0], B.n[1], 0.0])
        org = np.array([B.origin[0], B.origin[1], B.z0]) + 0.5 * B.w * n
        loc = bd.Location(bd.Plane(origin=tuple(org), x_dir=u, z_dir=tuple(-n)))
        return s.moved(loc)
    if q.kind == "leg":
        st = q.meta["stations"]
        L = lay.legs[q.meta["leg"]]
        upright = float(np.hypot(*(L.foot_xy - L.xy))) < 1e-6
        if upright and p.leg_capital == "round" and p.leg_sweep == 0 and p.leg_swirl == 0:
            return _bd_revolve(bd, list(zip(st["r"], st["z"])), L.xy)
        # lofted sections for leaning or polygonal stems
        from .profiles import section_radius
        idx = np.unique(np.linspace(0, len(st["z"]) - 1, 28).astype(int))
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        wires = []
        from .model import leg_spine
        spine = leg_spine(L, st, p.leg_sweep, p.leg_swirl, p.leg_splay)
        for i in idx:
            c = spine[i]
            rr = st["r"][i] * section_radius(th, p.leg_capital, st["shape"][i], q.meta.get("phase", 0.0))
            pts = [(c[0] + r_ * math.cos(a), c[1] + r_ * math.sin(a), float(st["z"][i])) for r_, a in zip(rr, th)]
            wires.append(bd.Wire.make_polygon(pts, close=True))
        return bd.Solid.make_loft(wires, ruled=False)
    if q.kind == "post":
        L = lay.legs[q.meta["leg"]]
        r = 0.5 * p.post_d
        return bd.Solid.make_cylinder(r, q.zmax - q.zmin, bd.Plane(origin=(L.xy[0], L.xy[1], q.zmin)))
    if q.kind == "glass":
        return _bd_prism(bd, q.footprint, q.zmin, q.zmax)
    if q.kind in ("pad", "washer", "bushing"):
        c = q.footprint.centroid
        r = math.sqrt(q.footprint.area / math.pi)
        return bd.Solid.make_cylinder(r, q.zmax - q.zmin, bd.Plane(origin=(c.x, c.y, q.zmin)))
    if q.kind == "cap":
        L = lay.legs[q.meta["leg"]]
        R = 0.5 * p.cap_d
        if p.cap_style in ("dome", "mushroom", "hoju"):
            prof = cap_profile(p.cap_style, R, p.cap_h)
            return _bd_revolve(bd, [(r, z) for r, z in prof[1:-1]], L.xy, q.zmin)
        if p.cap_style == "rokuyo":
            ros = rosette(R, n=96) + L.xy
            plate = bd.extrude(_bd_poly_face(bd, ros, q.zmin), amount=0.36 * p.cap_h)
            prof = cap_profile("dome", 0.52 * R, p.cap_h)
            return plate + _bd_revolve(bd, [(r, z) for r, z in prof[1:-1]], L.xy, q.zmin)
        th = np.arange(8) * np.pi / 4 + np.pi / 8
        k = 1.0 / np.cos(np.pi / 8)
        pts = np.c_[R * k * np.cos(th), R * k * np.sin(th)] + L.xy
        body = bd.extrude(_bd_poly_face(bd, pts, q.zmin), amount=0.42 * p.cap_h)
        apex = (float(L.xy[0]), float(L.xy[1]), q.zmin + 0.9 * p.cap_h)
        roof = bd.Solid.make_loft([bd.Wire.make_polygon([(x, y, q.zmin + 0.42 * p.cap_h) for x, y in pts], close=True),
                                   bd.Wire.make_polygon([(apex[0] + 0.6 * math.cos(a), apex[1] + 0.6 * math.sin(a), apex[2])
                                                         for a in th], close=True)])
        return body + roof
    if q.kind == "block":
        b = [x for x in lay.blocks if x.id == q.id][0]
        zs, half = masu_stations(b.z0, b.z1, b.size, 0.25 * p.beam_h if b.slot_into else 0.0)
        u = b.u / np.linalg.norm(b.u)
        n = np.array([-u[1], u[0]])
        wires = []
        for z, hw in zip(zs, half):
            cs = [b.xy + hw * (u + n), b.xy + hw * (-u + n), b.xy + hw * (-u - n), b.xy + hw * (u - n)]
            wires.append(bd.Wire.make_polygon([(c[0], c[1], float(z)) for c in cs], close=True))
        return bd.Solid.make_loft(wires, ruled=True)
    return None


def export_step(d: Design, path: Path) -> Path:
    import build123d as bd
    from .joinery import BIG, pin_clearance
    by_id = {q.id: q for q in d.parts}
    cuts = {q.id: [] for q in d.parts}
    c = d.p.clearance
    for r in d.rels:
        P, Q = by_id[r.a], by_id[r.b]
        if r.kind == "pin":
            cuts[Q.id].append((P.footprint.buffer(pin_clearance(Q.kind, d.p), quad_segs=16), -BIG, BIG))
        elif r.kind == "blind":
            cuts[Q.id].append((P.footprint.buffer(0.3, quad_segs=16), -BIG, r.z))
        else:
            cuts[P.id].append((Q.footprint.buffer(c, join_style="mitre"), r.z, BIG))
            cuts[Q.id].append((P.footprint.buffer(c, join_style="mitre"), -BIG, r.z))
    solids = []
    groups: dict = OrderedDict()
    for q in d.parts:
        s = _bd_nominal(bd, q, d)
        if s is None:
            continue
        zlo, zhi = q.zmin - 5, q.zmax + 5
        for poly, z0, z1 in cuts[q.id]:
            z0, z1 = max(z0, zlo), min(z1, zhi)
            if z1 <= z0:
                continue
            s = s - _bd_prism(bd, poly, z0, z1)
        s.label = q.id
        groups.setdefault(q.group, []).append(s)
    for g, ss in groups.items():
        s = ss[0]
        for o in ss[1:]:
            s = s + o
        s.label = g
        solids.append(s)
    comp = bd.Compound(children=solids)
    bd.export_step(comp, str(path))
    return path


# ---------------------------------------------------------------------------
# profiles from the finished solids
# ---------------------------------------------------------------------------
def beam_elevation(q):
    """Side profile of a finished beam: slice through its mid-width.
    Returns polygons in (u along the beam from its left end, z up from its bottom)."""
    Finv = np.linalg.inv(q.frame)
    s = q.solid.transform(Finv[:3, :])                    # local: x along, y across, z up
    R = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=float)   # y -> z
    s = s.transform(R)
    cs = s.slice(0.0)
    polys = [np.asarray(c) for c in cs.to_polygons()]
    out = []
    for P in polys:
        P = np.c_[P[:, 0], -P[:, 1]]                      # back to (x, z)
        out.append(P)
    if not out:
        return out
    x0 = min(P[:, 0].min() for P in out)
    z0 = min(P[:, 1].min() for P in out)
    return [P - [x0, z0] for P in out]


def notch_table(d: Design, B):
    """Notches along a beam: (from-left position of the notch centre, width, depth, top/bottom, partner)."""
    rows = []
    split = {(r.a, r.b): r for r in d.rels if r.kind == "split"}
    from .layout import crossing
    for r in d.rels:
        if r.kind != "split" or B.id not in (r.a, r.b):
            continue
        other = r.b if r.a == B.id else r.a
        try:
            O = d.layout.beam(other)
        except KeyError:
            continue
        cr = crossing(B, O)
        if cr is None:
            continue
        t = cr.ta - B.t0
        if r.a == B.id:      # B is the earlier part: it loses wood above the split
            depth = B.z1 - r.z
            side = "top"
        else:
            depth = r.z - B.z0
            side = "bottom"
        if depth <= 0.05:
            continue
        rows.append((round(t, 1), round(2 * cr.ha + 2 * d.p.clearance, 1), round(depth, 1), side, other))
    rows.sort()
    return rows


# ---------------------------------------------------------------------------
# CNC: strip profiles with dogbones, nested on boards
# ---------------------------------------------------------------------------
def dogboned(poly: np.ndarray, r: float):
    """A router bit of radius r cannot cut a square inside corner, so the
    crossing strip would not seat.  Clear every inside (reflex) corner of the
    profile with a circle of radius r centred on the corner: a dogbone."""
    from shapely.geometry import Point as SPoint, Polygon as SPoly
    from shapely.geometry.polygon import orient
    g = orient(SPoly(poly).buffer(0), 1.0)
    P = np.asarray(g.exterior.coords)[:-1]
    n = len(P)
    holes = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = b - a, c - b
        if np.linalg.norm(e1) < 1e-6 or np.linalg.norm(e2) < 1e-6:
            continue
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cosang = float(e1 @ e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        # reflex corner of a CCW outline (turns right) by more than ~45 degrees
        if cross < 0 and cosang < 0.7:
            holes.append(SPoint(*b).buffer(r, quad_segs=12))
    if holes:
        from shapely.ops import unary_union
        g = g.difference(unary_union(holes))
    if g.geom_type != "Polygon":
        g = max(g.geoms, key=lambda q: q.area)
    return np.asarray(orient(g, 1.0).exterior.coords), len(holes)


def nest_strips(d: Design, board_w: float = 140.0, board_l: float = 2400.0, margin: float = 15.0):
    """Shelf-nest every strip profile on boards (grain along the length)."""
    p = d.p
    gap = p.tool_d + 4.0
    r = 0.5 * p.tool_d
    items = []
    for B in d.layout.beams:
        polys = beam_elevation(d.part(B.id))
        if not polys:
            continue
        P = max(polys, key=lambda q: len(q))
        prof, n_db = dogboned(P, r)
        L = float(prof[:, 0].max() - prof[:, 0].min())
        H = float(prof[:, 1].max() - prof[:, 1].min())
        items.append(dict(id=B.id, prof=prof - prof.min(axis=0), L=L, H=H, dogbones=n_db))
    items.sort(key=lambda it: (-it["H"], -it["L"]))
    boards = []          # list of rows; each row: dict(y, h, x, parts)
    for it in items:
        placed = False
        for bd in boards:
            for row in bd["rows"]:
                if it["H"] <= row["h"] + 1e-6 and row["x"] + it["L"] <= board_l - margin:
                    row["parts"].append((row["x"], row["y"], it))
                    row["x"] += it["L"] + gap
                    placed = True
                    break
            if placed:
                break
            y = bd["rows"][-1]["y"] + bd["rows"][-1]["h"] + gap if bd["rows"] else margin
            if y + it["H"] <= board_w - margin:
                row = dict(y=y, h=it["H"], x=margin, parts=[])
                row["parts"].append((row["x"], row["y"], it))
                row["x"] += it["L"] + gap
                bd["rows"].append(row)
                placed = True
                break
        if not placed:
            row = dict(y=margin, h=it["H"], x=margin, parts=[])
            row["parts"].append((row["x"], row["y"], it))
            row["x"] += it["L"] + gap
            boards.append(dict(rows=[row]))
    return boards, dict(board_w=board_w, board_l=board_l, thick=p.beam_w, gap=gap)


def export_cnc(d: Design, folder: Path) -> dict:
    boards, info = nest_strips(d)
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    for name, col in (("BOARD", 8), ("CUT", 1), ("LABEL", 3), ("INFO", 7)):
        doc.layers.add(name, color=col)
    y_off = 0.0
    n_parts = n_db = 0
    cut_len = 0.0
    for k, bd in enumerate(boards):
        W, L = info["board_w"], info["board_l"]
        msp.add_lwpolyline([(0, y_off), (L, y_off), (L, y_off + W), (0, y_off + W)], close=True,
                           dxfattribs={"layer": "BOARD"})
        msp.add_text(f"board {k + 1}/{len(boards)}: {info['thick']:.0f} thick x {W:.0f} x {L:.0f} {d.p.wood_beams}, "
                     f"grain along the length", height=10, dxfattribs={"layer": "INFO"}).set_placement((0, y_off + W + 8))
        for row in bd["rows"]:
            for x, y, it in row["parts"]:
                P = it["prof"] + [x, y + y_off]
                msp.add_lwpolyline(P, close=True, dxfattribs={"layer": "CUT"})
                cut_len += float(np.sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)))
                msp.add_text(it["id"], height=6, dxfattribs={"layer": "LABEL"}).set_placement(
                    (x + 0.5 * it["L"], y + y_off + 0.5 * it["H"]), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
                n_parts += 1
                n_db += it["dogbones"]
        y_off += info["board_w"] + 60
    path = folder / "cnc_strips.dxf"
    doc.saveas(path)
    # preview
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly, Rectangle
    fig, axes = plt.subplots(len(boards), 1, figsize=(16.54, 1.2 + 1.6 * len(boards)), squeeze=False)
    for k, (ax, bd) in enumerate(zip(axes[:, 0], boards)):
        ax.add_patch(Rectangle((0, 0), info["board_l"], info["board_w"], fc="#efe2c8", ec="0.4", lw=0.6))
        for row in bd["rows"]:
            for x, y, it in row["parts"]:
                ax.add_patch(MPoly(it["prof"] + [x, y], fc="#d2b48c", ec="#3a2a1a", lw=0.4))
                ax.text(x + 0.5 * it["L"], y + 0.5 * it["H"], it["id"], fontsize=5, ha="center", va="center")
        ax.set_xlim(-10, info["board_l"] + 10)
        ax.set_ylim(-10, info["board_w"] + 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"board {k + 1}/{len(boards)}  {info['thick']:.0f} x {info['board_w']:.0f} x {info['board_l']:.0f} mm",
                     fontsize=7, loc="left")
    fig.suptitle(f"CNC cut sheets: {n_parts} strips on {len(boards)} boards, {n_db} dogbone corners for a "
                 f"{d.p.tool_d:.2f} mm bit", x=0.02, ha="left", fontsize=10)
    fig.savefig(folder / "cnc_strips.pdf")
    plt.close(fig)
    return dict(boards=len(boards), strips=n_parts, dogbones=n_db, cut_length_m=cut_len / 1000.0,
                board=f"{info['thick']:.0f} x {info['board_w']:.0f} x {info['board_l']:.0f} mm")


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------
def export_glass_dxf(d: Design, path: Path) -> Path | None:
    if d.p.glass_kind == "none":
        return None
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.add("CUT", color=1)
    doc.layers.add("HOLES", color=5)
    doc.layers.add("NOTES", color=7)
    g = d.part("top").footprint
    msp.add_lwpolyline(np.asarray(g.exterior.coords)[:, :2], close=True, dxfattribs={"layer": "CUT"})
    for h in g.interiors:
        c = np.asarray(h.coords)
        ctr = c.mean(axis=0)
        r = float(np.linalg.norm(c - ctr, axis=1).mean())
        msp.add_circle(ctr, r, dxfattribs={"layer": "HOLES"})
    b = g.bounds
    note = (f"{d.p.glass_kind} {d.p.glass_t:.0f} mm, tempered after drilling; polished edge; "
            f"{len(g.interiors)} holes dia {2 * (0.5 * d.p.post_d + d.p.glass_hole_clear):.1f} mm; "
            f"overall {b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} mm")
    msp.add_text(note, height=12, dxfattribs={"layer": "NOTES"}).set_placement((b[0], b[1] - 40))
    doc.saveas(path)
    return path


def export_beams_dxf(d: Design, path: Path) -> Path:
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.add("PROFILE", color=3)
    doc.layers.add("LABEL", color=7)
    y = 0.0
    for B in d.layout.beams:
        q = d.part(B.id)
        polys = beam_elevation(q)
        for P in polys:
            msp.add_lwpolyline(P + [0, y], close=True, dxfattribs={"layer": "PROFILE"})
        msp.add_text(f"{B.id}  L={B.length:.0f}  {B.w:.0f}x{B.h:.0f}  {d.p.wood_beams}", height=8,
                     dxfattribs={"layer": "LABEL"}).set_placement((0, y + B.h + 6))
        y += B.h + 36
    doc.saveas(path)
    return path


# ---------------------------------------------------------------------------
# cut list
# ---------------------------------------------------------------------------
def cutlist_rows(d: Design):
    p = d.p
    rows = []
    sig: dict = OrderedDict()
    for B in d.layout.beams:
        nt = tuple((t, w, dp, s) for t, w, dp, s, _ in notch_table(d, B))
        key = (round(B.length), round(B.h, 1), nt, tuple(e["style"] for e in B.ends))
        sig.setdefault(key, []).append(B.id)
    for (L, h, nt, ends), ids in sig.items():
        rows.append(dict(part="beam", ids=" ".join(ids), qty=len(ids), material=p.wood_beams,
                         stock=f"{L + 20:.0f} x {p.beam_w:.0f} x {h:.0f}",
                         finished=f"{L:.0f} x {p.beam_w:.0f} x {h:.0f}",
                         notes=f"{len(nt)} notches; ends {ends[0]}/{ends[1]}"))
    legs = [q for q in d.parts if q.kind == "leg"]
    if legs:
        q = legs[0]
        st = q.meta["stations"]
        dmax = 2 * float(np.max(st["r"]))
        post_len = d.layout.post_top - (d.layout.legs[0].z_slot - 14)
        integral = p.post_mode != "dowel"
        rows.append(dict(part="leg + integral post" if integral else "leg", ids=" ".join(x.id for x in legs),
                         qty=len(legs), material=p.wood_legs,
                         stock=f"dia {dmax + 10:.0f} x {q.zmax + (post_len if integral else 0) + 30:.0f} turning blank",
                         finished=(f"leg {q.zmax:.0f} tall, max dia {dmax:.0f}; post dia {p.post_d:.1f} x {post_len:.0f}"
                                   if integral else f"leg {q.zmax:.0f} tall, max dia {dmax:.0f}; threaded socket dia "
                                                    f"{p.post_d:.1f} x {p.dowel_engage:.0f}"),
                         notes=("lathe" if legs_turnable(p, d) else "not lathe-turnable (lean, sweep, swirl or polygonal capital)")
                               + "; capital slots differ per leg (see drawings)"))
    posts = [q for q in d.parts if q.kind == "post"]
    if posts and p.post_mode == "dowel":
        q = posts[0]
        rows.append(dict(part="threaded dowel", ids=" ".join(x.id for x in posts), qty=len(posts), material=p.post_wood,
                         stock=f"dia {p.post_d + 3:.0f} x {q.zmax - q.zmin + 20:.0f}",
                         finished=f"dia {p.post_d:.1f} x {q.zmax - q.zmin:.0f}",
                         notes=f"wood thread both ends: {p.dowel_engage:.0f} mm into the leg socket, "
                               f"{p.thread_len:.0f} mm into the cap"))
    caps = [q for q in d.parts if q.kind == "cap"]
    if caps:
        rows.append(dict(part="cap", ids=" ".join(q.id for q in caps), qty=len(caps), material=p.wood_caps,
                         stock=f"dia {p.cap_d * 1.15 + 10:.0f} x {p.cap_h + 15:.0f}",
                         finished=f"{p.cap_style} dia {p.cap_d:.0f} x {p.cap_h:.0f}",
                         notes=f"blind threaded hole dia {p.post_d:.1f} x {p.thread_len + 1:.0f} deep"))
    blocks = [q for q in d.parts if q.kind == "block"]
    if blocks:
        rows.append(dict(part="masu block", ids=" ".join(q.id for q in blocks), qty=len(blocks),
                         material=p.wood_beams, stock="", finished=f"{p.masu * p.beam_w:.0f} square", notes=""))
    if p.glass_kind != "none":
        g = d.part("top").footprint
        b = g.bounds
        rows.append(dict(part="top", ids="top", qty=1, material=f"{p.glass_kind} (tempered)",
                         stock=f"{b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} x {p.glass_t:.0f}",
                         finished="cut to glass.dxf", notes=f"{len(g.interiors)} holes"))
        rows.append(dict(part="flanged bushing", ids="", qty=len(d.layout.legs), material="nylon or rubber",
                         stock="", finished=f"bore {p.post_d + 0.6:.1f}, sleeve {p.post_d + 2 * p.glass_hole_clear - 0.6:.1f}",
                         notes="sits in the glass hole"))
        n_pads = sum(1 for q in d.parts if q.kind == "pad")
        rows.append(dict(part="clear silicone bumper", ids="", qty=n_pads, material="silicone", stock="",
                         finished=f"dia 10 x {p.pad_t:.1f}", notes="self-adhesive"))
    return rows


def export_cutlist(d: Design, path: Path) -> Path:
    rows = cutlist_rows(d)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "qty", "material", "stock", "finished", "notes", "ids"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------------
# drawings
# ---------------------------------------------------------------------------
def export_drawings(d: Design, path: Path, checks: list | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon as MPoly, Circle

    lay = d.layout
    p = d.p
    T = len(lay.z_tiers)
    cmap = plt.get_cmap("viridis")
    ink = "#1d1a17"
    with PdfPages(path) as pdf:
        # ---- page 1: overview -------------------------------------------------
        fig = plt.figure(figsize=(16.54, 11.69))       # A3 landscape
        fig.suptitle(f"kigumi table  -  seed {p.seed}  -  {p.grammar} lattice, {T} tiers, "
                     f"{len(lay.legs)} legs", x=0.04, ha="left", fontsize=14, color=ink)
        ax = fig.add_axes([0.03, 0.08, 0.5, 0.82])
        g = lay.outline
        ax.add_patch(MPoly(np.asarray(g.exterior.coords), fill=False, ec="#6a8fa0", lw=1.2))
        for B in sorted(lay.beams, key=lambda b: b.tier):
            col = cmap(B.tier / max(1, T - 1))
            ax.add_patch(MPoly(np.asarray(B.strip().exterior.coords), fc=(*col[:3], 0.45), ec=col, lw=0.5))
        for L in lay.legs:
            ax.add_patch(Circle(L.xy, L.r_cap, fill=False, ec="#7a4a2a", lw=1.2))
            ax.add_patch(Circle(L.xy, 0.5 * p.post_d, fc="#7a4a2a"))
        b = g.bounds
        ax.set_xlim(b[0] - 40, b[2] + 40)
        ax.set_ylim(b[1] - 40, b[3] + 40)
        ax.set_aspect("equal")
        ax.set_title("plan (lattice coloured by tier, glass outline in blue)", fontsize=9, color=ink)
        ax.tick_params(labelsize=7)
        # text block
        tx = fig.add_axes([0.56, 0.08, 0.41, 0.82])
        tx.axis("off")
        lines = [f"size  {b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} x {p.height:.0f} mm (glass top)",
                 f"lattice  {p.beam_w:.0f} wide, {min(lay.tier_h):.0f}-{max(lay.tier_h):.0f} deep, {p.wood_beams}",
                 f"legs  {p.wood_legs}, foot dia {p.leg_base_d:.0f}, stem top {p.leg_top_d:.0f}, capital {2 * lay.legs[0].r_cap:.0f}",
                 f"posts  dia {p.post_d:.1f}, wood thread {p.thread_len:.0f} mm into {p.cap_style} caps ({p.wood_caps})",
                 f"top  {p.glass_kind} {p.glass_t:.0f} mm" if p.glass_kind != "none" else "top  open lattice (no glass)",
                 f"joints  clearance {p.clearance:.2f} mm per side; tier overlap {p.overlap:.2f} x depth",
                 ""]
        if checks:
            lines.append("engineering checks")
            for c in checks:
                lines.append(f"  [{c['status']}] {c['name']}: {c['detail']}")
        y = 1.0
        for ln in lines:
            for chunk in _wrap(ln, 105):
                tx.text(0, y, chunk, fontsize=8, family="monospace", color=ink, va="top")
                y -= 0.022
        pdf.savefig(fig)
        plt.close(fig)

        # ---- tier plans ----------------------------------------------------------
        cols = 3
        rows = (T + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16.54, 11.69), squeeze=False)
        for t in range(T):
            ax = axes.flat[t]
            ax.add_patch(MPoly(np.asarray(g.exterior.coords), fill=False, ec="0.8", lw=0.6))
            for B in lay.beams:
                if B.tier == t:
                    ax.add_patch(MPoly(np.asarray(B.strip().exterior.coords), fc="#d9c19a", ec=ink, lw=0.4))
                    c = B.p(0.5 * (B.t0 + B.t1))
                    ax.text(c[0], c[1], B.id.split("-", 1)[1], fontsize=4.5, ha="center", va="center",
                            rotation=B.angle_deg() if abs(B.angle_deg()) <= 90 else B.angle_deg() - 180)
            for L in lay.legs:
                ax.add_patch(Circle(L.xy, L.r_cap, fill=False, ec="#7a4a2a", lw=0.8))
            ax.set_xlim(b[0] - 20, b[2] + 20)
            ax.set_ylim(b[1] - 20, b[3] + 20)
            ax.set_aspect("equal")
            ax.set_title(f"tier {t}: z {lay.z_tiers[t]:.1f} to {lay.z_tiers[t] + lay.tier_h[t]:.1f} mm, "
                         f"{sum(1 for B in lay.beams if B.tier == t)} beams", fontsize=8)
            ax.axis("off")
        for ax in list(axes.flat)[T:]:
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- beam elevations, 1:2 on A3 ----------------------------------------
        beams = list(lay.beams)
        per_page = 9
        for k in range(0, len(beams), per_page):
            fig = plt.figure(figsize=(16.54, 11.69))
            fig.suptitle("beam elevations (about 1:3 on A3; exact profiles in beams.dxf)   "
                         "notch table: centre from left end / width / depth / top or bottom",
                         x=0.03, ha="left", fontsize=10)
            for i, B in enumerate(beams[k:k + per_page]):
                ax = fig.add_axes([0.03, 0.86 - i * 0.1, 0.62, 0.075])
                for P in beam_elevation(d.part(B.id)):
                    ax.add_patch(MPoly(P, fc="#ead7b5", ec=ink, lw=0.6))
                ax.set_xlim(-5, 820)
                ax.set_ylim(-5, 60)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.text(0, 52, f"{B.id}   L {B.length:.0f}   {B.w:.0f} x {B.h:.0f}", fontsize=7)
                nt = notch_table(d, B)
                s = "  ".join(f"{t:.0f}/{w:.0f}/{dp:.0f}{side[0]}" for t, w, dp, side, _ in nt)
                tx = fig.add_axes([0.66, 0.86 - i * 0.1, 0.33, 0.075])
                tx.axis("off")
                for j, chunk in enumerate(_wrap(s, 70)[:5]):
                    tx.text(0, 0.9 - j * 0.2, chunk, fontsize=6, family="monospace", va="top")
            pdf.savefig(fig)
            plt.close(fig)

        # ---- leg turning profile, full size ------------------------------------
        q = [x for x in d.parts if x.kind == "leg"][0]
        st = q.meta["stations"]
        H = float(st["z"].max())
        fig = plt.figure(figsize=(11.69, 16.54))       # A3 portrait, 1:1 if it fits
        scale = 1.0 if H < 380 else 380.0 / H
        w_mm, h_mm = 297, 420
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, w_mm)
        ax.set_ylim(0, h_mm)
        ax.axis("off")
        x0, y0 = 0.5 * w_mm, 20
        r = st["r"] * scale
        z = st["z"] * scale
        ax.plot(x0 + r, y0 + z, color=ink, lw=0.8)
        ax.plot(x0 - r, y0 + z, color=ink, lw=0.8)
        ax.plot([x0, x0], [y0 - 5, y0 + z.max() + 5], color="0.6", lw=0.4, ls="-.")
        for zz in np.arange(0, H + 1, 20):
            rr = float(np.interp(zz, st["z"], st["r"]))
            ax.plot([x0 - rr * scale, x0 + rr * scale], [y0 + zz * scale] * 2, color="0.8", lw=0.3)
            ax.text(x0 + rr * scale + 6, y0 + zz * scale, f"z {zz:.0f}: dia {2 * rr:.1f}", fontsize=5, va="center")
        ax.text(8, h_mm - 10, f"leg turning template ({'full size' if scale == 1 else f'scale {scale:.2f}'})  -  "
                f"{p.wood_legs}, {len(lay.legs)} off; capital slots and post per plan", fontsize=7)
        ax.plot([8, 108], [8, 8], color=ink, lw=1)
        ax.text(8, 10, "100 mm (check print scale)", fontsize=6)
        pdf.savefig(fig)
        plt.close(fig)
        capital_page(d, pdf)
    return path


def capital_slots(d: Design, leg_idx: int):
    """Slots cut into one leg's capital: direction, offset from the axis,
    width and depth below the capital top (from the joinery relations)."""
    lay = d.layout
    L = lay.legs[leg_idx]
    rows = []
    for r in d.rels:
        if r.note != "capital slot" or r.a != f"leg{leg_idx}":
            continue
        try:
            B = lay.beam(r.b)
        except KeyError:
            continue
        off = float((L.xy - B.origin) @ B.n)
        ang = B.angle_deg() % 180.0
        rows.append(dict(beam=B.id, angle=round(ang, 1), offset=round(-off, 1),
                         width=round(B.w + 2 * d.p.clearance, 2), depth=round(L.z_top - r.z, 1)))
    rows.sort(key=lambda x: (x["angle"], x["offset"]))
    return rows


def capital_page(d: Design, pdf):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    lay = d.layout
    n = len(lay.legs)
    fig, axes = plt.subplots(1, n, figsize=(16.54, 11.69), squeeze=False)
    fig.suptitle("leg capitals seen from above (slots at the top, deeper slots darker; depth below the capital top in the table)",
                 x=0.02, ha="left", fontsize=10)
    for i, ax in enumerate(axes[0]):
        q = d.part(f"leg{i}")
        L = lay.legs[i]
        for z, col in ((L.z_top - 0.5, "#d2b48c"), (L.z_slot + 0.5, "#a07850")):
            cs = q.solid.slice(z)
            for P in cs.to_polygons():
                P = np.asarray(P) - L.xy
                ax.add_patch(MPoly(P, fc=col, ec="#2a1d12", lw=0.5, alpha=0.9))
        R = L.r_cap * 1.15
        ax.set_xlim(-R, R)
        ax.set_ylim(-R - 40, R)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([-R, R], [0, 0], color="0.7", lw=0.3)
        ax.plot([0, 0], [-R, R], color="0.7", lw=0.3)
        rows = capital_slots(d, i)
        txt = "\n".join(f"{r['beam']}: {r['angle']:.0f} deg, offset {r['offset']:+.1f}, width {r['width']:.2f}, "
                        f"depth {r['depth']:.1f}" for r in rows)
        ax.set_title(f"leg{i}  capital dia {2 * L.r_cap:.0f}, top z {L.z_top:.1f}", fontsize=8)
        ax.text(-R, -R - 5, txt, fontsize=5.5, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _wrap(s: str, n: int):
    out = []
    while len(s) > n:
        cut = s.rfind(" ", 0, n)
        cut = n if cut <= 0 else cut
        out.append(s[:cut])
        s = "    " + s[cut:].lstrip()
    out.append(s)
    return out


# ---------------------------------------------------------------------------
# assembly notes
# ---------------------------------------------------------------------------
def export_assembly(d: Design, path: Path, steps: list) -> Path:
    p = d.p
    lines = ["# Assembly", "",
             "Everything drops in from above, in this order; no glue, no nails. Dry-fit each tier before the next.",
             f"Joint clearance is {p.clearance:.2f} mm per side - snug hand pressure, a soft mallet at most.", ""]
    for s in steps:
        ids = s["parts"]
        lines.append(f"{s['step']}. {s['text']} ({len(ids)} part{'s' if len(ids) != 1 else ''})")
        lines.append(f"   {', '.join(ids[:40])}{' ...' if len(ids) > 40 else ''}")
    lines += ["",
              "Caps: wax the thread, screw down hand-tight. The caps clamp the glass (or the grid) to the leg "
              "capitals, which is what holds the stack together.",
              "Seasonal movement: wood moves across the grain; re-snug the caps in dry months."]
    path.write_text("\n".join(lines) + "\n")
    return path


# rough 2026 US prices for a one-off; every estimate is printed with its inputs
PRICE_BDFT = {"hinoki": 28, "sugi": 14, "white_oak": 11, "walnut": 15, "cherry": 10, "ash": 7, "maple": 8,
              "yakisugi": 12, "wenge": 38}
RATE_CNC, RATE_TURN = 120.0, 75.0          # $/h
GLASS_SQFT, GLASS_HOLE, GLASS_EDGE_FT = 38.0, 12.0, 4.0


def leg_splay_deg(d: Design) -> float:
    """The largest angle a leg's axis leans from vertical (its foot off its capital)."""
    return max((math.degrees(math.atan2(float(np.hypot(*(L.foot_xy - L.xy))), max(L.z_top, 1.0)))
                for L in d.layout.legs), default=0.0)


def legs_turnable(p, d: Design | None = None) -> bool:
    """A leg is lathe work if its axis is straight (it may be tilted to splay)."""
    leans = (leg_splay_deg(d) > 0.01) if d is not None else p.leg_lean != 0
    straight = not leans or p.leg_splay == "tilt"
    return straight and p.leg_sweep == 0 and p.leg_swirl == 0 and p.leg_capital == "round"


def fabrication_md(d: Design, cnc: dict, checks: list, cad: bool = True) -> str:
    p = d.p
    lay = d.layout
    bdft = lambda mm3: mm3 / 2359737.0          # board feet
    strip_vol = cnc["boards"] * p.beam_w * 140 * 2400 * 1.15      # boards, with planing waste
    legs = [q for q in d.parts if q.kind == "leg"]
    st = legs[0].meta["stations"]
    dmax = 2 * float(np.max(st["r"]))
    leg_blank = math.pi / 4 * (dmax + 12) ** 2 * (legs[0].zmax + 40) * len(legs)
    cap_blank = math.pi / 4 * (p.cap_d * 1.15 + 12) ** 2 * (p.cap_h + 20) * len(legs)
    wood_cost = (bdft(strip_vol) * PRICE_BDFT.get(p.wood_beams, 12) + bdft(leg_blank) * PRICE_BDFT.get(p.wood_legs, 12)
                 + bdft(cap_blank) * PRICE_BDFT.get(p.wood_caps, 12))
    cnc_h = 0.75 + cnc["cut_length_m"] * 3 / (2.5 * 60) + cnc["dogbones"] * 3 / 3600
    turn_h = (len(legs) * (2.0 if legs_turnable(p, d) else 5.0) + len(legs) * 0.75
              + (len(legs) * 0.5 if p.post_mode == "dowel" else 0))
    glass = ""
    glass_cost = 0.0
    if p.glass_kind != "none":
        g = d.part("top").footprint
        sqft = g.area / 92903.0
        edge_ft = g.exterior.length / 304.8
        glass_cost = sqft * GLASS_SQFT * (p.glass_t / 10) + len(g.interiors) * GLASS_HOLE + edge_ft * GLASS_EDGE_FT
        glass = f"{sqft:.1f} sq ft, {edge_ft:.1f} ft of polished freeform edge, {len(g.interiors)} holes"
    cost_cnc, cost_turn = cnc_h * RATE_CNC, turn_h * RATE_TURN
    total = wood_cost + cost_cnc + cost_turn + glass_cost + 30
    b = lay.outline.bounds
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    L = []
    L += [f"# Fabrication package - seed {p.seed}", "",
          f"Table {b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} mm, {p.height:.0f} mm high. Lattice {p.wood_beams}, legs {p.wood_legs}, "
          f"caps {p.wood_caps}" + (f", {p.glass_t:.0f} mm {p.glass_kind} top." if p.glass_kind != "none" else ", open lattice top."),
          "", "## Who makes what", "",
          "| part | qty | how | files |", "|---|---|---|---|",
          (f"| lattice strips {p.beam_w:.0f} x {min(lay.tier_h):.0f}-{max(lay.tier_h):.0f} mm | {cnc['strips']} | CNC router: 2D profiles cut "
           f"through {p.beam_w:.0f} mm boards ({cnc['boards']} boards {cnc['board']}), {cnc['dogbones']} dogbone corners for a "
           f"{p.tool_d:.2f} mm bit. A woodworker can also cut them on a table saw with a dado stack (identical strips are grouped "
           "in the cut list). | cnc_strips.dxf / .pdf, beams.dxf, cutlist.csv |" if abs(p.tier_turn - 90.0) < 1e-6 else
           f"| lattice strips {p.beam_w:.0f} x {min(lay.tier_h):.0f}-{max(lay.tier_h):.0f} mm | {cnc['strips']} | the strips cross at "
           f"{p.tier_turn:.0f} degrees, so the notch walls are angled: a flat through-cut cannot make them. Either a kumiko-style "
           "woodworker with an angled half-lap jig, or a CNC with each strip clamped on edge and the notches pocketed from the top "
           "and bottom edges (two setups per strip; table.step has the exact solids). The outlines in cnc_strips.dxf give "
           "lengths and end curves only. | table.step, beams.dxf, cutlist.csv |"),
          f"| legs | {len(legs)} | " + ("woodturner: turn to the full-size template, rout the capital slots with a jig, drill and "
                                       "tap the centre socket" + (f"; the legs splay up to {leg_splay_deg(d):.1f} deg, "
                                       "so trim the foot and capital faces level after turning" if leg_splay_deg(d) > 0.05 else "")
                                       if legs_turnable(p, d) else
                                       "CNC carving (4/5-axis) or hand carving: these stems lean, bow or swirl, so they are "
                                       "not lathe work") + " | drawings.pdf (turning template, capital slots), table.step |",
          (f"| threaded dowels dia {p.post_d:.0f} | {len(legs)} | {p.post_wood} dowel, wood thread both ends "
           f"({p.dowel_engage:.0f} mm into the leg, {p.thread_len:.0f} mm into the cap) with a matching thread box and tap "
           "| cutlist.csv |" if p.post_mode == "dowel" else
           f"| posts | {len(legs)} | turned with the legs as one piece | drawings.pdf |"),
          f"| caps ({p.cap_style}) | {len(legs)} | woodturner, tapped to match the dowel thread | table.step |"]
    if p.glass_kind != "none":
        L.append("| glass top | 1 | glass shop: CNC / waterjet cut to the DXF, holes drilled before tempering, "
                 "edges polished | glass.dxf |")
        L.append(f"| flanged bushings, clear bumpers | {len(legs)} / {sum(1 for q in d.parts if q.kind == 'pad')} | "
                 "off the shelf | cutlist.csv |")
    L += ["", "## Fit and tolerances", "",
          f"* Notches are cut {p.clearance:.2f} mm wider than the strip on each side (a snug, hand-pressed fit). CNC "
          "routers hold about +/-0.1 mm in wood, but wood moves with humidity: **cut one test crossing first** and change "
          "`clearance` if needed.",
          f"* Dogbones assume a {p.tool_d:.2f} mm bit. If the shop uses another bit, regenerate with `tool_d` set to it.",
          "* Use quartersawn or rift-sawn stock for the strips (it moves least), acclimatise the boards for a week, and "
          "finish (oil) before assembly. No glue: the caps clamp the stack together.",
          "* Every strip is labelled (LABEL layer); keep the labels on until assembly.",
          "", "## Engineering checks", ""]
    for c in checks:
        L.append(f"* **{c['status']}** - {c['name']}: {c['detail']}")
    L += ["", "## Rough cost (one-off, 2026 US prices, +/- 40%)", "",
          "| item | estimate | basis |", "|---|---|---|",
          f"| wood | ${wood_cost:,.0f} | boards {bdft(strip_vol):.0f} bd ft {p.wood_beams}; leg blanks "
          f"{bdft(leg_blank):.0f} bd ft {p.wood_legs}; caps |",
          f"| CNC routing | ${cost_cnc:,.0f} | {cnc_h:.1f} h at ${RATE_CNC:.0f}/h ({cnc['cut_length_m']:.0f} m of profile, 3 passes) |",
          f"| turning, slots, threads | ${cost_turn:,.0f} | {turn_h:.1f} h at ${RATE_TURN:.0f}/h |"]
    if p.glass_kind != "none":
        L.append(f"| glass | ${glass_cost:,.0f} | {glass} |")
    L += ["| hardware | $30 | bushings, bumpers |", f"| **total** | **${total:,.0f}** | |", ""]
    if fails or warns:
        L += ["## Before you order", ""]
        for c in fails:
            L.append(f"* fix: {c['name']} - {c['detail']}")
        for c in warns:
            L.append(f"* consider: {c['name']} - {c['detail']}")
    md = "\n".join(L) + "\n"
    if not cad:         # made in the browser: the exact solids come from the Python package
        md = md.replace("table.step", "table.step (Python package)")
        md = md.replace("\n## Who makes what", "\nMade in the browser. The exact solids (table.step, table.3dm) come from the "
                        "Python package: `uv run python -m kigumi.server`, then Export.\n\n## Who makes what", 1)
    return md


def export_all(d: Design, folder: Path, step: bool = True, cad: bool = True) -> Path:
    """The fabrication package in `folder`.  cad=False leaves out the Rhino and
    STEP solids (they need native libraries: the browser build has neither)."""
    from .checks import run_checks
    from .report import summarize
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(d.p.to_json())
    rep = summarize(d)
    checks = run_checks(d, full=True)
    rep["checks"] = checks
    (folder / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    export_glb(d, folder / "table.glb")
    export_stl(d, folder / "table.stl")
    if cad:
        export_3dm(d, folder / "table.3dm")
    export_glass_dxf(d, folder / "glass.dxf")
    export_beams_dxf(d, folder / "beams.dxf")
    export_cutlist(d, folder / "cutlist.csv")
    export_drawings(d, folder / "drawings.pdf", checks)
    export_assembly(d, folder / "assembly.md", rep["assembly"])
    cnc = export_cnc(d, folder)
    (folder / "FABRICATION.md").write_text(fabrication_md(d, cnc, checks, cad=cad))
    if step and cad:
        try:
            export_step(d, folder / "table.step")
        except Exception as e:  # noqa: keep the rest even if OCC trips
            (folder / "table.step.error.txt").write_text(repr(e))
    return folder
