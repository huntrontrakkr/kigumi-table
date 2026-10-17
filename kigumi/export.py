"""Mesh exports: GLB (web viewer), NPZ (Blender renderer), STL, OBJ."""
from __future__ import annotations

import json
from pathlib import Path

import struct

import numpy as np

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


def glb_bytes(d: Design, explode: float = 0.0) -> bytes:
    """Binary glTF of the design: one node per part (named by its id, with
    kind/material/order in the node's extras), its mesh in the part's own
    frame, a plain PBR material per species; metres, Y up."""
    ranks = _explode_offsets(d, explode)
    blobs: list[bytes] = []
    offset = 0
    views, accessors, meshes, nodes, materials = [], [], [], [], []
    mat_index: dict = {}

    def add_view(data: bytes, target: int) -> int:
        nonlocal offset
        pad = (-len(data)) % 4
        blobs.append(data + b"\0" * pad)
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        offset += len(data) + pad
        return len(views) - 1

    for q in d.parts:
        V, F = part_mesh(q, local=True)
        if len(F) == 0:
            continue
        Vm = (V * 1e-3).astype(np.float32)
        Fi = F.astype(np.uint32).ravel()
        iv = add_view(Fi.tobytes(), 34963)
        accessors.append(dict(bufferView=iv, componentType=5125, count=int(Fi.size), type="SCALAR",
                              min=[int(Fi.min())], max=[int(Fi.max())]))
        ia = len(accessors) - 1
        vv = add_view(Vm.tobytes(), 34962)
        accessors.append(dict(bufferView=vv, componentType=5126, count=int(len(Vm)), type="VEC3",
                              min=Vm.min(axis=0).tolist(), max=Vm.max(axis=0).tolist()))
        va = len(accessors) - 1
        key = (q.material, q.kind == "glass")
        if key not in mat_index:
            col = PALETTE.get(q.material, (0.7, 0.7, 0.7))
            glass = q.kind == "glass"
            materials.append(dict(name=q.material, doubleSided=False, alphaMode="BLEND" if glass else "OPAQUE",
                                  pbrMetallicRoughness=dict(baseColorFactor=[*col, 0.35 if glass else 1.0],
                                                            metallicFactor=0.0, roughnessFactor=0.05 if glass else 0.6)))
            mat_index[key] = len(materials) - 1
        meshes.append(dict(name=q.id, primitives=[dict(attributes=dict(POSITION=va), indices=ia, mode=4,
                                                       material=mat_index[key])]))
        F4 = q.frame.copy()
        F4[2, 3] += ranks.get(q.id, 0.0)
        Fm = F4.copy()
        Fm[:3, 3] *= 1e-3
        T = ZUP_TO_YUP @ Fm
        nodes.append(dict(name=q.id, mesh=len(meshes) - 1, matrix=T.T.ravel().tolist(),
                          extras=dict(kind=q.kind, material=q.material, order=float(q.order))))
    gltf = dict(asset=dict(version="2.0", generator="kigumi"), scene=0,
                scenes=[dict(nodes=list(range(len(nodes))))], nodes=nodes, meshes=meshes, materials=materials,
                accessors=accessors, bufferViews=views, buffers=[dict(byteLength=offset)])
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binary = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def export_glb(d: Design, path: Path, explode: float = 0.0) -> Path:
    path = Path(path)
    path.write_bytes(glb_bytes(d, explode))
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


def stl_bytes(d: Design) -> bytes:
    """Binary STL of every part in place (millimetres, Z up)."""
    tris = []
    for q in d.parts:
        V, F = part_mesh(q, local=False)
        if len(F):
            tris.append(V[F])
    T = np.concatenate(tris).astype(np.float32) if tris else np.zeros((0, 3, 3), np.float32)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = T
    return b"kigumi table".ljust(80, b" ") + struct.pack("<I", len(T)) + rec.tobytes()


def export_stl(d: Design, path: Path) -> Path:
    path = Path(path)
    path.write_bytes(stl_bytes(d))
    return path
