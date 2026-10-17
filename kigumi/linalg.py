"""Sparse symmetric positive definite solves.

SuperLU (scipy) everywhere -- natively and, in the browser, in Pyodide's
WebAssembly build -- with a symmetric ordering and no pivoting, which suits
the stiffness matrices here.  An accelerator can be plugged in (the web worker
plugs in the WebGPU conjugate-gradient solver): it gets the matrix and the
right-hand sides and returns the solutions, or None to fall back."""
from __future__ import annotations

import numpy as np

_ACCEL = None          # callable(K_csr, B (n x k)) -> X | None
STATS = dict(accel_calls=0, accel_fallbacks=0, cpu_factor=0)


def set_accelerator(fn) -> None:
    global _ACCEL
    _ACCEL = fn


def factor(K):
    """SuperLU factor of a symmetric positive definite sparse matrix."""
    from scipy.sparse.linalg import splu
    STATS["cpu_factor"] += 1
    A = K.tocsc()
    # growing SuperLU's work arrays mid-factorization can fail inside
    # WebAssembly ("Can't expand MemType"): then retry with the general
    # column ordering and partial pivoting, which fills differently
    try:
        return splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    except (RuntimeError, MemoryError):
        STATS["cpu_retry"] = STATS.get("cpu_retry", 0) + 1
        return splu(A, permc_spec="COLAMD")


def solve_many(K, B, accelerate: bool = True) -> np.ndarray:
    """X with K X = B (B is n or n x k)."""
    if accelerate and _ACCEL is not None:
        STATS["accel_calls"] += 1
        try:
            X = _ACCEL(K.tocsr(), np.asarray(B, dtype=np.float64))
        except Exception:
            X = None
        if X is not None:
            return X
        STATS["accel_fallbacks"] += 1
    return factor(K).solve(B)
