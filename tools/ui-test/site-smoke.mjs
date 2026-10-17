// Smoke test of the browser build (site/): boots the engine (Pyodide from the
// CDN), waits for the first design and its checks, takes a screenshot.
//   node site-smoke.mjs --url http://localhost:8790/
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg('url', 'http://localhost:8790/');
const OUT = arg('out', 'shots/site');
fs.mkdirSync(OUT, { recursive: true });
const env = { ...process.env };
const deps = path.join(process.env.HOME || '', '.local/lib/chromedeps');
if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
// headless Linux Chromium has only SwiftShader's WebGPU, which three.js cannot
// draw with (old texture-view swizzle); by default the viewer falls back to
// WebGL2 here.  --webgpu 1 turns SwiftShader's WebGPU on anyway.
const gpuArgs = arg('webgpu', '') ? ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] : [];
const browser = await chromium.launch({ headless: true, env,
  args: [...gpuArgs, '--enable-features=WebAssemblyJSPromiseIntegration'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.setDefaultTimeout(600_000);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') log('console.' + m.type(), m.text().slice(0, 300)); });
page.on('pageerror', (e) => log('pageerror', e.message.slice(0, 400)));
const t0 = Date.now();
await page.goto(URL_);
await page.waitForFunction(() => window.kigumi && window.kigumi.engine);
let last = '';
const poll = setInterval(async () => {
  try {
    const t = await page.$eval('#boot-text', (e) => e.textContent).catch(() => 'boot gone');
    if (t !== last) { last = t; log('boot:', t); }
  } catch {}
}, 500);
await page.evaluate(() => window.kigumi.ready);
log(`engine ready + first design in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await page.evaluate(() => window.kigumi.settled());
clearInterval(poll);
const backend = await page.evaluate(() => window.kigumi.backend);
log('backend', backend);
if (backend === 'page') log('engine status', JSON.stringify(await page.evaluate(() => window.kigumi.engine.status)));
log('viewer backend', await page.evaluate(() => window.kigumi.viewer.backend));
const rep = await page.evaluate(() => window.kigumi.report);
log('report keys', Object.keys(rep || {}).slice(0, 12).join(','), 'checks', (rep && rep.checks || []).length);
await page.screenshot({ path: path.join(OUT, '01-first-design.png') });
log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await browser.close();
