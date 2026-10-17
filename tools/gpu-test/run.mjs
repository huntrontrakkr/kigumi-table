// Headless test run of the WebGPU sparse SPD solver (site/engine/gpu/sparse.js):
// serves the repository over http://localhost (WebGPU needs a secure context),
// opens tools/gpu-test/index.html in Playwright's Chromium with WebGPU enabled,
// waits for window.__results, prints a summary and writes out/results.json.
//
//   node tools/gpu-test/run.mjs [--cases id,group,...] [--compare] [--mode worker|main]
//                               [--bigk 16] [--eta 1e-4] [--precond fsai] [--timeout 3600] [--headed]
//
// Exit code 0 when every case with acceptance criteria passes.
// Reuses Playwright from tools/ui-test/node_modules.  On this machine Chromium needs
// the libraries in ~/.local/lib/chromedeps; the adapter is SwiftShader (CPU), so
// timings are not representative of a GPU -- correctness and iteration counts are.

import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
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
delete flags.headed;
delete flags.timeout;

let chromium;
try {
  ({ chromium } = await import(pathToFileURL(path.join(ROOT, 'tools/ui-test/node_modules/playwright/index.mjs')).href));
} catch (e) {
  console.error('Playwright not found in tools/ui-test/node_modules (cd tools/ui-test && npm install):', e.message);
  process.exit(2);
}

// test data (converted from tools/gpu-test/matrices by convert.py)
if (!fs.existsSync(path.join(HERE, 'data/manifest.json'))) {
  console.log('data/manifest.json missing: running convert.py');
  const r = spawnSync('uv', ['run', 'python', 'tools/gpu-test/convert.py'], { cwd: ROOT, stdio: 'inherit' });
  if (r.status !== 0) console.warn('convert.py failed: only the synthetic cases will run');
}

const freePort = () => new Promise((resolve, reject) => {
  const s = net.createServer();
  s.unref();
  s.on('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});

const port = await freePort();
const server = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd: ROOT, stdio: 'ignore' });
let browser = null;
const cleanup = async () => {
  if (browser) { try { await browser.close(); } catch { /* already gone */ } browser = null; }
  if (server.exitCode === null && !server.killed) server.kill('SIGTERM');   // by PID
};
process.on('SIGINT', async () => { await cleanup(); process.exit(130); });

let exitCode = 1;
try {
  const base = `http://localhost:${port}`;
  for (let i = 0; ; i++) {
    try { if ((await fetch(`${base}/tools/gpu-test/index.html`)).ok) break; } catch { /* not up yet */ }
    if (i > 100) throw new Error('http.server did not start');
    await new Promise((r) => setTimeout(r, 100));
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
  page.on('console', (m) => {
    const t = m.text();
    if (m.type() === 'error' || m.type() === 'warning') console.log(`[page ${m.type()}] ${t}`);
    else if (t.startsWith('[test] ')) console.log(t.slice(7));
  });
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  const query = new URLSearchParams(flags).toString();
  const url = `${base}/tools/gpu-test/index.html${query ? '?' + query : ''}`;
  console.log(`open ${url}  (Chromium ${browser.version()})`);
  await page.goto(url);
  await page.waitForFunction(() => window.__results, null, { timeout: timeoutMs, polling: 1000 });
  const out = await page.evaluate(() => window.__results);

  fs.mkdirSync(path.join(HERE, 'out'), { recursive: true });
  fs.writeFileSync(path.join(HERE, 'out/results.json'), JSON.stringify(out, null, 1));
  const e = (v) => (v == null ? '' : Number(v).toExponential(1));
  const rows = [['case', 'n', 'k', 'precond', 'iters', 'refs', 'relres', 'error', 'floor', 'setup ms', 'total ms', 'result']];
  for (const r of out.results) {
    rows.push(r.error ? [r.id, '', '', '', '', '', '', '', '', '', '', 'ERROR']
      : [r.name, r.n, r.k, r.precond, r.iters, r.refinements, e(r.relresMax), e(r.errMax), e(r.floorMax),
        Math.round(r.setupMs || 0), Math.round(r.ms), r.pass === true ? 'pass' : r.pass === false ? 'FAIL' : 'info']);
  }
  const w = rows[0].map((_, c) => Math.max(...rows.map((r) => String(r[c]).length)));
  console.log('');
  for (const r of rows) console.log(r.map((v, c) => (c === 0 ? String(v).padEnd(w[c]) : String(v).padStart(w[c]))).join('  '));
  for (const r of out.results) if (r.error) console.log(`\n${r.id}: ${r.error}`);
  console.log(`\nadapter: ${out.gpu ? JSON.stringify(out.gpu) : 'none'}${out.error ? '\nerror: ' + out.error : ''}`);
  console.log(`${out.ok ? 'PASS' : 'FAIL'} (${out.mode} mode, ${(out.wallMs / 1000).toFixed(1)} s); details in tools/gpu-test/out/results.json`);
  exitCode = out.ok ? 0 : 1;
} catch (err) {
  console.error(err);
} finally {
  await cleanup();
}
process.exit(exitCode);
