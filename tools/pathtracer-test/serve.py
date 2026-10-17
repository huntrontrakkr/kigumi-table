#!/usr/bin/env python3
"""Test helper for the path tracer: a no-cache static server with CORS and a
POST sink, so a page served from site/ (port 8790) can import the test scripts
and the render module fresh on every run and upload its images.

    python3 tools/pathtracer-test/serve.py [port]      # default 8791

    GET  /test/<file>     tools/pathtracer-test/<file>
    GET  /render/<file>   site/render/<file>   (fresh copy, never cached)
    POST /save?name=x.png body -> tools/pathtracer-test/out/x.png
    POST /log             body appended to tools/pathtracer-test/out/log.txt
"""
import os
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
MOUNTS = {'/test/': HERE, '/render/': os.path.join(ROOT, 'site', 'render')}


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = urllib.parse.urlsplit(path).path
        for prefix, base in MOUNTS.items():
            if path.startswith(prefix):
                rel = os.path.normpath(urllib.parse.unquote(path[len(prefix):])).lstrip('/')
                if rel.startswith('..'):
                    break
                return os.path.join(base, rel)
        return os.path.join(HERE, '__nothing__')

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        n = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(n)
        os.makedirs(OUT, exist_ok=True)
        if u.path == '/save':
            name = os.path.basename(urllib.parse.parse_qs(u.query).get('name', ['upload.bin'])[0])
            with open(os.path.join(OUT, name), 'wb') as f:
                f.write(body)
        elif u.path == '/log':
            with open(os.path.join(OUT, 'log.txt'), 'ab') as f:
                f.write(body + b'\n')
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ok')

    def guess_type(self, path):
        if path.endswith('.js') or path.endswith('.mjs'):
            return 'text/javascript'
        return super().guess_type(path)

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
