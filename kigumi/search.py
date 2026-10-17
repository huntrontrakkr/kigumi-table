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


def evaluate(args, keep_design: bool = False) -> dict:
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
    out = dict(seed=seed, params=p.to_dict(), score=score, fails=fails, warns=warns, coverage=cov,
               checks=checks)
    if keep_design:
        out["design"] = d
    return out


def best_of(seed0: int, n: int = 12, base: dict | None = None, keep: list | None = None,
            jitter: float = 0.0, budget_s: float | None = None) -> list[dict]:
    """n candidates, best first.  Without worker processes (the browser) they
    run one by one, and with a time budget the search stops once it is spent
    and some candidate built; the winner then keeps its built design
    (["design"]) so it need not be built again."""
    jobs = [(seed0 + 7919 * i, base, keep, jitter) for i in range(n)]
    try:
        res = list(_pool().map(evaluate, jobs))
    except (ImportError, NotImplementedError, OSError, RuntimeError):
        import time
        t0, res = time.monotonic(), []
        for j in jobs:
            r = evaluate(j, keep_design=True)
            if res and r["score"] <= max(q["score"] for q in res):
                r.pop("design", None)                   # only the best so far keeps its design
            else:
                for q in res:
                    q.pop("design", None)
            res.append(r)
            if budget_s is not None and time.monotonic() - t0 > budget_s and any(q["score"] > -1e9 for q in res):
                break
    res.sort(key=lambda r: r["score"], reverse=True)
    return res
