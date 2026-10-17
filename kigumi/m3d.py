"""Manifold, natively (the `manifold3d` wheel) or, in the browser (Pyodide),
through its WebAssembly build (see _m3d_web.py)."""
try:
    import manifold3d as m3d
except ImportError:                      # Pyodide: no native wheel
    from . import _m3d_web as m3d        # noqa: F401
