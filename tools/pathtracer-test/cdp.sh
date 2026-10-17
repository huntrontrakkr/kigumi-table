#!/usr/bin/env bash
# Run a test module in the kigumi app on the Windows host's headless Chrome (real GPU).
#   tools/pathtracer-test/cdp.sh <script.js> '<json args>' [page query] [timeout s]
# Opens http://localhost:8790/<query>, waits for the engine to boot and settle, imports
# http://localhost:8791/test/<script.js> (serve.py) and awaits run(args).  Prints the
# RESULT line; the screenshot lands in /mnt/c/Temp/pt-shot.png.  Needs the Windows Chrome from
# tools/win-chrome/start.ps1 (an isolated profile with DevTools on 127.0.0.1:9223).
set -euo pipefail
script=${1:?script}
args=${2:-{\}}
query=${3:-}
timeout=${4:-600}
expr="(async () => { for (let i = 0; i < 600 && !window.kigumi; i++) await new Promise((r) => setTimeout(r, 50)); await window.kigumi.ready; await window.kigumi.settled(); const m = await import('http://localhost:8791/test/${script}?t=' + Date.now()); try { return await m.run(${args}); } catch (e) { return { error: String(e && e.stack || e) }; } })()"
if [[ "$expr" == *'"'* ]]; then echo "expression must not contain double quotes" >&2; exit 2; fi
out=$(mktemp)
driver=$(wslpath -w "$(dirname "$0")/../win-chrome/cdp.ps1")
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$driver" \
  -Url "http://localhost:8790/${query}" -Expr "$expr" -Shot 'C:\Temp\pt-shot.png' -TimeoutS "$timeout" > "$out" 2>&1 || true
cut -c1-6000 "$out"
rm -f "$out"
