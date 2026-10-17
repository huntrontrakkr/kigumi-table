"""The WebGPU solver as kigumi's linear-algebra accelerator (browser only).

The web worker registers `kigumi_gpu_js.solve(n, k, indptr, indices, data,
rhs)` (site/engine/gpu/sparse.js: mixed-precision conjugate gradients on the
GPU, f64-accurate) and calls install().  A Python solve then waits on the GPU
through JS Promise Integration; anything the GPU does not converge on falls
back to SuperLU (see linalg.solve_many)."""
from __future__ import annotations

import numpy as np

MIN_N = 400                 # small systems are quicker on the CPU


def install() -> None:
    import kigumi_gpu_js as gpu                     # registered by the worker
    from pyodide.ffi import can_run_sync, run_sync, to_js

    from .linalg import set_accelerator

    def accel(K, B):
        n = K.shape[0]
        if n < MIN_N or not can_run_sync():
            return None
        B2 = np.asarray(B, dtype=np.float64).reshape(n, -1)
        k = B2.shape[1]
        K = K.tocsr()
        K.sum_duplicates()
        rhs = np.ascontiguousarray(B2.T).ravel()               # column-major: one RHS after another
        x = run_sync(gpu.solve(n, k, to_js(K.indptr.astype(np.int32)), to_js(K.indices.astype(np.int32)),
                               to_js(K.data.astype(np.float64)), to_js(rhs)))
        if x is None:
            return None
        X = np.asarray(x.to_py(), dtype=np.float64).reshape(k, n).T
        return X.reshape(np.shape(B))

    set_accelerator(accel)
