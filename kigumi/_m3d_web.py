"""The subset of the `manifold3d` Python API that kigumi uses, on top of
Manifold's WebAssembly build (site/engine/manifold-bridge.js), for Pyodide.

The JS bridge is registered as the module `kigumi_manifold_js` by the web
worker before kigumi is imported.  Objects wrap Embind handles and free them
when Python lets go of them."""
from __future__ import annotations

import numpy as np
from pyodide.ffi import to_js

import kigumi_manifold_js as _js       # registered by the worker


class FillRule:
    Positive = "Positive"
    Negative = "Negative"
    NonZero = "NonZero"
    EvenOdd = "EvenOdd"


class OpType:
    Add = "add"
    Subtract = "subtract"
    Intersect = "intersect"


def _f32(a):
    return to_js(np.ascontiguousarray(a, dtype=np.float32).ravel())


def _u32(a):
    return to_js(np.ascontiguousarray(a, dtype=np.uint32).ravel())


def _f64(a):
    return to_js(np.ascontiguousarray(a, dtype=np.float64).ravel())


def _i32(a):
    return to_js(np.ascontiguousarray(a, dtype=np.int32).ravel())


class Mesh:
    def __init__(self, vert_properties=None, tri_verts=None, **_):
        self.vert_properties = np.asarray(vert_properties, dtype=np.float32)
        self.tri_verts = np.asarray(tri_verts, dtype=np.uint32)


class _Handle:
    __slots__ = ("_o",)

    def __init__(self, o):
        self._o = o

    def __del__(self):
        o = getattr(self, "_o", None)
        if o is not None:
            try:
                _js.free(o)
            except Exception:
                pass
            self._o = None


class RayHit:
    __slots__ = ("distance", "position")

    def __init__(self, distance, position):
        self.distance = distance
        self.position = position


class CrossSection(_Handle):
    def __init__(self, contours=None, fill_rule=FillRule.Positive, _o=None):
        if _o is not None:
            super().__init__(_o)
            return
        rings = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in contours]
        flat = np.concatenate(rings) if rings else np.zeros((0, 2))
        super().__init__(_js.crossSection(_f64(flat), _i32([len(r) for r in rings]), fill_rule))

    def extrude(self, height):
        return Manifold(_o=_js.extrude(self._o, float(height)))

    def revolve(self, circular_segments=0, revolve_degrees=360.0):
        return Manifold(_o=_js.revolve(self._o, int(circular_segments), float(revolve_degrees)))

    def area(self):
        return float(_js.csArea(self._o))

    def to_polygons(self):
        flat = np.asarray(_js.csPolygons(self._o).to_py())
        out, i = [], 0
        while i < len(flat):
            n = int(flat[i])
            out.append(flat[i + 1:i + 1 + 2 * n].reshape(n, 2))
            i += 1 + 2 * n
        return out


class Manifold(_Handle):
    def __init__(self, mesh: Mesh | None = None, _o=None):
        if _o is None:
            _o = _js.mesh(_f32(mesh.vert_properties[:, :3]), _u32(mesh.tri_verts))
        super().__init__(_o)

    # constructors ---------------------------------------------------------
    @staticmethod
    def cylinder(height, radius_low, radius_high=-1.0, circular_segments=0, center=False):
        if radius_high is None or radius_high < 0:
            radius_high = radius_low
        return Manifold(_o=_js.cylinder(float(height), float(radius_low), float(radius_high),
                                        int(circular_segments), bool(center)))

    @staticmethod
    def sphere(radius, circular_segments=0):
        return Manifold(_o=_js.sphere(float(radius), int(circular_segments)))

    @staticmethod
    def revolve(cross_section, circular_segments=0, revolve_degrees=360.0):
        return Manifold(_o=_js.revolve(cross_section._o, int(circular_segments), float(revolve_degrees)))

    @staticmethod
    def batch_boolean(manifolds, op=OpType.Add):
        ms = list(manifolds)
        if op != OpType.Add:
            acc = ms[0]
            for m in ms[1:]:
                acc = acc - m if op == OpType.Subtract else acc ^ m
            return acc
        from pyodide.ffi import to_js as _tj
        return Manifold(_o=_js.union(_tj([m._o for m in ms])))

    # transforms -----------------------------------------------------------
    def translate(self, v):
        return Manifold(_o=_js.translate(self._o, float(v[0]), float(v[1]), float(v[2])))

    def transform(self, M):
        M = np.asarray(M, dtype=np.float64)
        col = np.zeros(16)
        col[0:3] = M[0:3, 0]
        col[4:7] = M[0:3, 1]
        col[8:11] = M[0:3, 2]
        col[12:15] = M[0:3, 3]
        col[15] = 1.0
        return Manifold(_o=_js.transform(self._o, _f64(col)))

    # booleans -------------------------------------------------------------
    def __add__(self, other):
        return Manifold(_o=_js.add(self._o, other._o))

    def __sub__(self, other):
        return Manifold(_o=_js.subtract(self._o, other._o))

    def __xor__(self, other):
        return Manifold(_o=_js.intersect(self._o, other._o))

    # queries --------------------------------------------------------------
    def volume(self):
        return float(_js.volume(self._o))

    def is_empty(self):
        return bool(_js.isEmpty(self._o))

    def bounding_box(self):
        return tuple(float(x) for x in _js.box(self._o).to_py())

    def to_mesh(self):
        g = _js.getMesh(self._o)
        v = np.asarray(g.vert.to_py(), dtype=np.float32).reshape(-1, 3)
        t = np.asarray(g.tri.to_py(), dtype=np.uint32).reshape(-1, 3)
        return Mesh(v, t)

    def ray_cast(self, origin, endpoint):
        flat = np.asarray(_js.rayCast(self._o, *[float(x) for x in origin], *[float(x) for x in endpoint]).to_py())
        return [RayHit(float(flat[4 * i]), tuple(flat[4 * i + 1:4 * i + 4])) for i in range(len(flat) // 4)]

    def slice(self, height):
        return CrossSection(_o=_js.slice(self._o, float(height)))
