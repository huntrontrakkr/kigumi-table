// Test run of the WebGPU sparse SPD solver (site/engine/gpu/sparse.js): serves
// the repository over http://localhost (WebGPU needs a secure context), opens
// tools/gpu-test/index.html in a browser, collects the results the page posts
// back, prints a summary and writes out/results.json.
//
//   node tools/gpu-test/run.mjs [--browser chromium|win-chrome|win-edge] [--cases id,group,...]
//                               [--compare] [--mode worker|main] [--bigk 8|all] [--eta 1e-4]
//                               [--precond fsai] [--timeout 3600] [--out results.json] [--headed]
//
// --browser chromium (default): Playwright's headless Chromium from tools/ui-test/node_modules.
//   On this machine it needs the libraries in ~/.local/lib/chromedeps and its WebGPU adapter is
//   SwiftShader (CPU): correctness and iteration counts are real, timings are not.
// --browser win-chrome | win-edge: (WSL2) the Windows host's Chrome or Edge, headless, on the
//   real GPU, started through PowerShell and killed by PID at the end.  It reaches this server
//   through WSL's localhost forwarding.
// Exit code 0 when every case with acceptance criteria passes.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');

const argv = process.argv.slice(2);
const flags = {};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (!a.startsWith('--')) continue;
  const key = a.slice(2);
  const next = argv[i + 1];
  if (next === undefined || next.startsWith('--')) flags[key] = '1';
  else { flags[key] = next; i++; }
}
const headed = !!flags.headed;
const timeoutMs = (+flags.timeout || 3600) * 1000;
const outFile = path.resolve(HERE, 'out', flags.out || 'results.json');
const browserKind = flags.browser || 'chromium';
for (const k of ['headed', 'timeout', 'out', 'browser']) delete flags[k];

const WIN_BROWSERS = {
  'win-chrome': 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'win-edge': 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
};
if (browserKind !== 'chromium' && !WIN_BROWSERS[browserKind]) {
  console.error(`unknown --browser ${browserKind}`);
  process.exit(2);
}

// test data (converted from tools/gpu-test/matrices by convert.py)
if (!fs.existsSync(path.join(HERE, 'data/manifest.json'))) {
  console.log('data/manifest.json missing: running convert.py');
  const r = spawnSync('uv', ['run', 'python', 'tools/gpu-test/convert.py'], { cwd: ROOT, stdio: 'inherit' });
  if (r.status !== 0) console.warn('convert.py failed: only the synthetic cases will run');
}

// ---- a static file server for the repository, plus /__log and /__results for the page
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.css': 'text/css', '.wasm': 'application/wasm' };
let resolveResults;
const resultsPosted = new Promise((r) => { resolveResults = r; });
const server = http.createServer((req, res) => {
  if (req.method === 'POST') {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      res.end('ok');
      if (req.url === '/__log') console.log(body);
      else if (req.url === '/__results') resolveResults(JSON.parse(body));
    });
    return;
  }
  const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.resolve(ROOT, '.' + rel);
  if (!file.startsWith(ROOT + path.sep)) { res.statusCode = 403; res.end(); return; }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.statusCode = 404; res.end(); return; }
    res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
    res.setHeader('Content-Length', st.size);
    res.setHeader('Cache-Control', 'no-store');
    fs.createReadStream(file).pipe(res);
  });
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

let browser = null;      // Playwright browser
let winPid = null;       // Windows browser process id
const cleanup = async () => {
  if (browser) { try { await browser.close(); } catch { /* gone */ } browser = null; }
  if (winPid) { spawnSync('taskkill.exe', ['/PID', String(winPid), '/T', '/F'], { stdio: 'ignore' }); winPid = null; }
  server.close();
};
process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

let exitCode = 1;
try {
  const query = new URLSearchParams({ ...flags, post: '1' }).toString();
  const url = `http://localhost:${port}/tools/gpu-test/index.html?${query}`;
  if (browserKind === 'chromium') {
    let chromium;
    try {
      ({ chromium } = await import(pathToFileURL(path.join(ROOT, 'tools/ui-test/node_modules/playwright/index.mjs')).href));
    } catch (e) {
      throw new Error('Playwright not found in tools/ui-test/node_modules (cd tools/ui-test && npm install): ' + e.message);
    }
    const env = { ...process.env };
    const deps = path.join(os.homedir(), '.local/lib/chromedeps');
    if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
    browser = await chromium.launch({
      headless: !headed,
      args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
      env,
    });
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page ${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => console.log('[pageerror]', e.message));
    console.log(`open ${url}  (Playwright Chromium ${browser.version()})`);
    await page.goto(url);
  } else {
    const exe = WIN_BROWSERS[browserKind];
    const ps = [
      `$p = Start-Process -PassThru -FilePath '${exe}' -ArgumentList @(`,
      `'--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-extensions',`,
      `"--user-data-dir=$env:TEMP\\kigumi-gpu-test-${browserKind}", '${url}');`,
      '$p.Id',
    ].join(' ');
    const r = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { encoding: 'utf8' });
    winPid = parseInt(String(r.stdout).trim(), 10);
    if (!(winPid > 0)) throw new Error(`could not start ${exe}: ${r.stderr || r.stdout}`);
    console.log(`open ${url}  (${browserKind}, Windows pid ${winPid})`);
  }
  const out = await Promise.race([
    resultsPosted,
    new Promise((_, rej) => setTimeout(() => rej(new Error(`no results after ${timeoutMs / 1000} s`)), timeoutMs)),
  ]);
  out.browser = browserKind;

  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, JSON.stringify(out, null, 1));
  const e = (v) => (v == null ? '' : Number(v).toExponential(1));
  const rows = [['case', 'n', 'k', 'precond', 'iters', 'refs', 'relres', 'error', 'floor', 'setup ms', 'total ms', 'ms/iter', 'result']];
  for (const r of out.results) {
    rows.push(r.expectedError ? [r.name, '', '', '', '', '', '', '', '', '', '', '', 'pass (threw)']
      : r.error ? [r.id, '', '', '', '', '', '', '', '', '', '', '', 'ERROR']
      : [r.name, r.n, r.k, r.precond, r.iters, r.refinements, e(r.relresMax), e(r.errMax), e(r.floorMax),
        Math.round(r.setupMs || 0), Math.round(r.ms), r.msPerIter == null ? '' : r.msPerIter.toFixed(2),
        r.pass === true ? 'pass' : r.pass === false ? 'FAIL' : 'info']);
  }
  const w = rows[0].map((_, c) => Math.max(...rows.map((r) => String(r[c]).length)));
  console.log('');
  for (const r of rows) console.log(r.map((v, c) => (c === 0 ? String(v).padEnd(w[c]) : String(v).padStart(w[c]))).join('  '));
  for (const r of out.results) if (r.error) console.log(`\n${r.id}: ${r.error}`);
  console.log(`\nadapter: ${out.gpu ? JSON.stringify(out.gpu) : 'none'}${out.error ? '\nerror: ' + out.error : ''}`);
  console.log(`${out.ok ? 'PASS' : 'FAIL'} (${browserKind}, ${out.mode} mode, ${(out.wallMs / 1000).toFixed(1)} s); details in ${path.relative(ROOT, outFile)}`);
  exitCode = out.ok ? 0 : 1;
} catch (err) {
  console.error(err);
} finally {
  await cleanup();
}
process.exit(exitCode);
