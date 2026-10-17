"""Local web server for the configurator.

    uv run python -m kigumi.server [--port 8765]

GET  /                 web/index.html
GET  /api/schema       parameter table (groups, ranges, choices), presets
POST /api/build        {params} -> {id, glb, report}
POST /api/random       {seed?, keep?: [names], base?: {params}, jitter?: 0..1, candidates?: n}
                       -> best of n candidates (jitter > 0: gentle mutation of base)
GET  /api/glb/<id>     binary GLB of a built design (explode via ?explode=mm)
GET  /api/checks/<id>  engineering checks for a build (computed in the background)
POST /api/render       {params, view} -> {png}  (Blender Cycles, slow)
GET  /files/<path>     files under out/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import random
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .build import build
from .export import export_glb
from .params import PRESETS, Params, preset, sample, schema
from .report import summarize

ROOT = Path(__file__).resolve().parent.parent
WEB = (ROOT / "web").resolve()
OUT = ROOT / "out" / "server"
OUT.mkdir(parents=True, exist_ok=True)

_cache: "OrderedDict[str, dict]" = OrderedDict()
_checks: "OrderedDict[str, list]" = OrderedDict()      # outlives the build cache
_lock = threading.Lock()


def _key(p: Params) -> str:
    return hashlib.sha1(json.dumps(p.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def build_entry(p: Params) -> dict:
    k = _key(p)
    with _lock:
        if k in _cache:
            _cache.move_to_end(k)
            return _cache[k]
    t = time.perf_counter()
    d = build(p)
    rep = summarize(d)
    rep["checks"] = []
    rep["checks_url"] = f"/api/checks/{k}"
    glb = OUT / f"{k}.glb"
    export_glb(d, glb)
    rep["build_s"] = round(time.perf_counter() - t, 3)
    entry = dict(id=k, glb=f"/api/glb/{k}", params=p.to_dict(), report=rep, design=d)
    # engineering checks take ~0.5-1 s: run them in the background
    ev = threading.Event()
    entry["checks_done"] = ev

    def work():
        try:
            from .checks import run_checks
            entry["checks"] = run_checks(d)
        except Exception as e:  # noqa
            entry["checks"] = [dict(name="checks", status="fail", value=None, limit=None, detail=repr(e))]
        with _lock:
            _checks[k] = entry["checks"]
            while len(_checks) > 500:
                _checks.popitem(last=False)
        ev.set()
    threading.Thread(target=work, daemon=True).start()
    with _lock:
        _cache[k] = entry
        while len(_cache) > 24:
            _cache.popitem(last=False)
    return entry


def public(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in ("design", "checks_done", "checks")}


class Handler(BaseHTTPRequestHandler):
    server_version = "kigumi/0.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=float).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if path == "/" or path == "/index.html":
                return self._file(WEB / "index.html")
            if path == "/api/schema":
                return self._json(dict(params=schema(), presets={k: preset(k).to_dict() for k in PRESETS}))
            if path.startswith("/api/checks/"):
                k = path.rsplit("/", 1)[1]
                e = _cache.get(k)
                if e is None:
                    if k in _checks:
                        return self._json(dict(id=k, checks=_checks[k]))
                    return self._json(dict(error="unknown id"), 404)
                e["checks_done"].wait(timeout=60)
                return self._json(dict(id=k, checks=e.get("checks", [])))
            if path.startswith("/api/glb/"):
                k = path.rsplit("/", 1)[1].split(".")[0]
                q = parse_qs(url.query)
                explode = float(q.get("explode", ["0"])[0])
                if explode > 0 and k in _cache:
                    f = OUT / f"{k}-x{int(explode)}.glb"
                    if not f.exists():
                        export_glb(_cache[k]["design"], f, explode=explode)
                    return self._file(f)
                return self._file(OUT / f"{k}.glb")
            if path.startswith("/files/"):
                f = (ROOT / "out" / path[len("/files/"):]).resolve()
                if (ROOT / "out").resolve() in f.parents:
                    return self._file(f)
                return self._json(dict(error="forbidden"), 403)
            f = (WEB / path.lstrip("/")).resolve()
            if WEB in f.parents and f.is_file():
                return self._file(f)
            self._json(dict(error="not found"), 404)
        except Exception as e:  # noqa
            self._json(dict(error=str(e), trace=traceback.format_exc()), 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/build":
                p = Params.from_dict(body.get("params", body))
                return self._json(public(build_entry(p)))
            if path == "/api/random":
                # generative search: evaluate a dozen candidates in parallel and
                # return the soundest (fewest failing checks, then warnings)
                from .search import best_of
                seed = int(body.get("seed", random.randrange(10_000_000)))
                n = int(body.get("candidates", 12))
                ranked = best_of(seed, n, body.get("base"), body.get("keep", []), float(body.get("jitter", 0.0)))
                top = ranked[0]
                e = build_entry(Params.from_dict(top["params"]))
                out = public(e)
                out["search"] = [dict(seed=r["seed"], score=round(r["score"], 1), fails=r.get("fails"),
                                      warns=r.get("warns")) for r in ranked]
                return self._json(out)
            if path == "/api/render":
                from .render import render
                p = Params.from_dict(body.get("params", {}))
                view = body.get("view", "hero")
                e = build_entry(p)
                samples = int(body.get("samples", 96))
                res = str(body.get("res", "1600x1100"))
                f = OUT / f"{e['id']}-{view}-{samples}-{res}.png"
                if not f.exists():
                    render(e["design"], f, view=view, samples=samples, res=res)
                return self._json(dict(png=f"/files/server/{f.name}"))
            if path == "/api/export":
                from .fab import export_all
                p = Params.from_dict(body.get("params", {}))
                e = build_entry(p)
                folder = export_all(e["design"], ROOT / "out" / "export" / e["id"])
                files = sorted(str(x.relative_to(ROOT / "out")) for x in folder.rglob("*") if x.is_file())
                return self._json(dict(folder=str(folder), files=[f"/files/{x}" for x in files]))
            self._json(dict(error="not found"), 404)
        except Exception as e:  # noqa
            self._json(dict(error=str(e), trace=traceback.format_exc()), 500)

    def _file(self, f: Path):
        if not f.exists():
            return self._json(dict(error=f"missing {f.name}"), 404)
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        if f.suffix == ".glb":
            ctype = "model/gltf-binary"
        self._send(200, f.read_bytes(), ctype, cache="/vendor/" in f.as_posix())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"kigumi configurator on http://{a.host}:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
