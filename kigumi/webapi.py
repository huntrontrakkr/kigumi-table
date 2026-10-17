"""The configurator's API for the browser: the same requests the local
server answers (see server.py), as plain functions for the web worker, which
runs this package in Pyodide.  Builds are cached by their parameters."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from collections import OrderedDict

from .build import build as _build
from .export import glb_bytes, stl_bytes
from .params import PRESETS, Params, preset, schema as _schema
from .report import summarize

_cache: "OrderedDict[str, dict]" = OrderedDict()


def _key(p: Params) -> str:
    return hashlib.sha1(json.dumps(p.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def schema() -> str:
    return json.dumps(dict(params=_schema(), presets={k: preset(k).to_dict() for k in PRESETS}), default=float)


def _entry(p: Params) -> dict:
    k = _key(p)
    if k in _cache:
        _cache.move_to_end(k)
        return _cache[k]
    t = time.perf_counter()
    d = _build(p)
    rep = summarize(d)
    rep["checks"] = []
    rep["checks_url"] = f"/api/checks/{k}"
    rep["build_s"] = round(time.perf_counter() - t, 3)
    e = dict(id=k, params=p.to_dict(), report=rep, design=d, glb=None, checks=None)
    _cache[k] = e
    while len(_cache) > 8:
        _cache.popitem(last=False)
    return e


def build(params_json: str) -> tuple[str, bytes]:
    """-> (JSON like the server's /api/build, GLB bytes)."""
    body = json.loads(params_json)
    p = Params.from_dict(body.get("params", body))
    e = _entry(p)
    if e["glb"] is None:
        e["glb"] = glb_bytes(e["design"])
    pub = {k: v for k, v in e.items() if k not in ("design", "glb", "checks")}
    return json.dumps(pub, default=float), e["glb"]


def checks(k: str) -> str:
    from .checks import run_checks
    e = _cache.get(k)
    if e is None:
        return json.dumps(dict(error="unknown id"))
    if e["checks"] is None:
        try:
            e["checks"] = run_checks(e["design"])
        except Exception as ex:  # noqa
            e["checks"] = [dict(name="checks", status="fail", value=None, limit=None, detail=repr(ex))]
    return json.dumps(dict(id=k, checks=e["checks"]), default=float)


def random(body_json: str) -> tuple[str, bytes]:
    """Best of a few candidates (one at a time in the browser)."""
    import random as _r
    from .search import best_of
    body = json.loads(body_json)
    seed = int(body.get("seed", _r.randrange(10_000_000)))
    n = int(body.get("candidates", 4))
    ranked = best_of(seed, n, body.get("base"), body.get("keep", []), float(body.get("jitter", 0.0)))
    top = ranked[0]
    pub, glb = build(json.dumps(dict(params=top["params"])))
    out = json.loads(pub)
    out["search"] = [dict(seed=r["seed"], score=round(r["score"], 1), fails=r.get("fails"), warns=r.get("warns"))
                     for r in ranked]
    return json.dumps(out, default=float), glb


def export(params_json: str) -> bytes:
    """The fabrication package as a zip (what the local version writes to a folder)."""
    import tempfile
    from pathlib import Path
    from .fab import export_all
    body = json.loads(params_json)
    p = Params.from_dict(body.get("params", body))
    e = _entry(p)
    with tempfile.TemporaryDirectory() as td:
        folder = export_all(e["design"], Path(td) / e["id"])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(folder).rglob("*")):
                if f.is_file():
                    z.write(f, f"kigumi-{e['id']}/{f.relative_to(folder)}")
            z.writestr(f"kigumi-{e['id']}/table.stl", stl_bytes(e["design"]))
        return buf.getvalue()
