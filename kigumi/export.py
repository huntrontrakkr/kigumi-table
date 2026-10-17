"""Mesh exports: GLB (web viewer), NPZ (Blender renderer), STL, OBJ."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .build import Design

PALETTE = {
    "hinoki": (0.86, 0.74, 0.55),
    "white_oak": (0.70, 0.53, 0.34),
    "walnut": (0.30, 0.19, 0.12),
    "cherry": (0.50, 0.28, 0.18),
    "ash": (0.84, 0.74, 0.58),
    "maple": (0.90, 0.80, 0.63),
    "sugi": (0.70, 0.43, 0.31),
    "yakisugi": (0.05, 0.045, 0.04),
    "wenge": (0.19, 0.13, 0.09),
    "glass": (0.86, 0.95, 0.93),
    "acrylic": (0.95, 0.97, 0.98),
    "rubber": (0.04, 0.04, 0.04),
    "silicone": (0.9, 0.92, 0.92),
}

ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])


def part_mesh(q, local: bool = True):
    mesh = q.solid.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    if local:
        Finv = np.linalg.inv(q.frame)
        V = V @ Finv[:3, :3].T + Finv[:3, 3]
    return V, F


def export_glb(d: Design, path: Path, explode: float = 0.0) -> Path:
    scene = trimesh.Scene()
    S = np.diag([1e-3, 1e-3, 1e-3, 1.0])           # mm -> m
    ranks = _explode_offsets(d, explode)
    for q in d.parts:
        V, F = part_mesh(q, local=True)
        tm = trimesh.Trimesh(V * 1e-3, F, process=False)
        col = PALETTE.get(q.material, (0.7, 0.7, 0.7))
        alpha = 0.35 if q.kind == "glass" else 1.0
        tm.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=q.material, baseColorFactor=[*col, alpha], metallicFactor=0.0,
            roughnessFactor=0.05 if q.kind == "glass" else 0.6,
            alphaMode="BLEND" if q.kind == "glass" else "OPAQUE"))
        F4 = q.frame.copy()
        F4[2, 3] += ranks.get(q.id, 0.0)
        Fm = F4.copy()
        Fm[:3, 3] *= 1e-3
        T = ZUP_TO_YUP @ Fm
        scene.add_geometry(tm, node_name=q.id, geom_name=q.id, transform=T,
                           metadata=dict(kind=q.kind, material=q.material, order=q.order))
    path = Path(path)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def _explode_offsets(d: Design, explode: float) -> dict:
    if explode <= 0:
        return {}
    kinds = {"leg": 0, "post": 0, "beam": 1, "block": 1, "glass": 2, "bushing": 3, "cap": 4}
    out = {}
    for q in d.parts:
        if q.kind in ("leg", "post"):
            out[q.id] = 0.0
        elif q.kind in ("beam", "block"):
            out[q.id] = explode * (1 + q.meta.get("tier", 0) if q.kind == "beam" else 1.5 + q.order / 10 - 1)
        else:
            out[q.id] = explode * (2 + d.p.tiers + kinds[q.kind])
    return out


def export_npz(d: Design, path: Path, explode: float = 0.0) -> Path:
    arrays = {}
    meta = []
    ranks = _explode_offsets(d, explode)
    for i, q in enumerate(d.parts):
        V, F = part_mesh(q, local=True)
        arrays[f"v{i}"] = V.astype(np.float32)
        arrays[f"f{i}"] = F.astype(np.int32)
        F4 = q.frame.copy()
        F4[2, 3] += ranks.get(q.id, 0.0)
        arrays[f"m{i}"] = F4
        meta.append(dict(id=q.id, kind=q.kind, material=q.material, order=q.order))
    arrays["meta"] = np.frombuffer(json.dumps(dict(parts=meta, params=d.p.to_dict())).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return Path(path)


def export_stl(d: Design, path: Path) -> Path:
    meshes = []
    for q in d.parts:
        V, F = part_mesh(q, local=False)
        meshes.append(trimesh.Trimesh(V, F, process=False))
    trimesh.util.concatenate(meshes).export(path)
    return Path(path)
