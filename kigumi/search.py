"""Generative search: sample many designs, evaluate them, keep the best.

Each candidate is built and checked in a worker process; the score prefers
designs with no failing checks, then fewer warnings, then a lattice that fills
its field.  This is the 'Galapagos' step of the Grasshopper analogy."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from .params import Params, sample

_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        # forkserver: workers start clean, so they never inherit the web server's
        # listening socket (forked workers kept the port after the server was stopped)
        import multiprocessing as mp
        _POOL = ProcessPoolExecutor(max_workers=max(2, min(12, (os.cpu_count() or 4) - 2)),
                                    mp_context=mp.get_context("forkserver"))
    return _POOL


def evaluate(args) -> dict:
    seed, base, keep, jitter = args
    from .build import build
    from .checks import run_checks
    from .layout import coverage
    p = sample(seed, Params.from_dict(base) if base else None, set(keep or []), jitter)
    try:
        d = build(p)
        checks = run_checks(d, quick=True)
        cov = coverage(d.layout)
    except Exception as e:  # noqa
        return dict(seed=seed, params=p.to_dict(), score=-1e9, error=repr(e))
    fails = sum(1 for c in checks if c["status"] == "fail")
    warns = sum(1 for c in checks if c["status"] == "warn")
    n_beams = len(d.layout.beams)
    score = -100.0 * fails - 10.0 * warns + 20.0 * cov + (0 if n_beams >= 12 else -50)
    return dict(seed=seed, params=p.to_dict(), score=score, fails=fails, warns=warns, coverage=cov,
                checks=checks)


def best_of(seed0: int, n: int = 12, base: dict | None = None, keep: list | None = None,
            jitter: float = 0.0) -> list[dict]:
    jobs = [(seed0 + 7919 * i, base, keep, jitter) for i in range(n)]
    try:
        res = list(_pool().map(evaluate, jobs))
    except (ImportError, NotImplementedError, OSError, RuntimeError):
        res = [evaluate(j) for j in jobs]          # no processes (e.g. the browser): one by one
    res.sort(key=lambda r: r["score"], reverse=True)
    return res
