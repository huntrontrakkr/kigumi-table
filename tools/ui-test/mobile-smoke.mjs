// Phone smoke test of the browser build: an emulated phone (WebKit as an iPhone,
// or Chromium as a Pixel with a slowed CPU) boots the engine, shows the first
// design and its checks, then is driven by touch: open the parameters, change one,
// read the report.  Screenshots go to shots/mobile-<browser>/.
//   node mobile-smoke.mjs --browser webkit   --url http://127.0.0.1:8790/
//   node mobile-smoke.mjs --browser chromium --device "Pixel 7" --cpu 4
// WebKit on Linux needs its libraries (see README: PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS).
import { chromium, webkit, devices } from 'playwright';
import fs from 'fs';
import path from 'path';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg('url', 'http://127.0.0.1:8790/');
const which = arg('browser', 'webkit');
const deviceName = arg('device', which === 'webkit' ? 'iPhone 15' : 'Pixel 7');
const cpu = Number(arg('cpu', '1'));
const landscape = !!arg('landscape', '');
const OUT = arg('out', `shots/mobile-${which}${landscape ? '-land' : ''}`);
fs.mkdirSync(OUT, { recursive: true });

const env = { ...process.env };
const libs = ['webkitdeps', 'chromedeps'].map((d) => path.join(process.env.HOME || '', '.local/lib', d))
  .filter((d) => fs.existsSync(d));
if (libs.length) env.LD_LIBRARY_PATH = libs.join(':') + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
const gio = path.join(process.env.HOME || '', '.local/lib/webkitdeps/gio');      // TLS for https (glib-networking)
if (fs.existsSync(gio)) env.GIO_EXTRA_MODULES = gio;

const bt = which === 'webkit' ? webkit : chromium;
const browser = await bt.launch({ headless: true, env });
const dev = devices[landscape ? deviceName + ' landscape' : deviceName];
if (!dev) throw new Error('unknown device ' + deviceName);
const ctx = await browser.newContext({ ...dev });
const page = await ctx.newPage();
page.setDefaultTimeout(900_000);
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') log('console.' + m.type(), m.text().slice(0, 300)); });
page.on('pageerror', (e) => log('pageerror', e.message.slice(0, 400)));
if (which === 'chromium' && cpu > 1) {
  const cdp = await ctx.newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: cpu });
  log(`CPU slowed ${cpu}x`);
}
const shot = async (name) => page.screenshot({ path: path.join(OUT, name + '.png') });

const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(1) + ' s';
await page.goto(URL_);
await page.waitForFunction(() => window.kigumi && window.kigumi.engine);
let last = '';
const poll = setInterval(async () => {
  const t = await page.$eval('#boot-text', (e) => e.textContent).catch(() => 'boot gone');
  if (t !== last) { last = t; log('boot:', t); }
}, 500);
await shot('00-boot');
await page.evaluate(() => window.kigumi.ready);
log('first design in', secs(), JSON.stringify(await page.evaluate(() => window.kigumi.stats)));
await page.evaluate(() => window.kigumi.settled());
clearInterval(poll);
log('settled in', secs());
const info = await page.evaluate(async () => ({
  backend: window.kigumi.backend,
  viewer: window.kigumi.viewer.backend,
  status: window.kigumi.engine.status,
  stats: window.kigumi.backend === 'page' ? await window.kigumi.engine.api('/api/stats') : null,
  viewport: [innerWidth, innerHeight, devicePixelRatio],
  pixelRatio: window.kigumi.viewer.renderer?.getPixelRatio?.(),
  checks: (window.kigumi.report.checks || []).map((c) => c.status).join(' '),
}));
log('info', JSON.stringify(info));
await shot('01-first-design');

// tap a part: the inspector tip shows (touch has no hover)
{
  const r = await page.$eval('#gl-host canvas', (c) => { const b = c.getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; });
  let shown = false;
  for (const [fx, fy] of [[0.5, 0.45], [0.5, 0.55], [0.4, 0.5], [0.6, 0.5], [0.5, 0.65]]) {
    await page.touchscreen.tap(r[0] + fx * r[2], r[1] + fy * r[3]);
    await page.waitForTimeout(350);
    shown = await page.$eval('#tip', (t) => t.classList.contains('show'));
    if (shown) break;
  }
  log('tap inspects a part:', shown, shown ? await page.$eval('#tip b', (b) => b.textContent) : '');
  await shot('01b-tap');
}

// the parameter sheet: open it, move one slider by touch, wait for the rebuild
const tab = await page.$('[data-tab="design"]');
if (tab) {
  await tab.tap();
  await page.waitForTimeout(400);
  await shot('02-design-sheet');
  const before = await page.evaluate(() => JSON.stringify(window.kigumi.appliedParams));
  const tb = Date.now();
  await page.evaluate(() => window.kigumi.setParams({ leg_count: 3 }));
  await page.waitForFunction((b) => JSON.stringify(window.kigumi.appliedParams) !== b, before);
  await page.evaluate(() => window.kigumi.settled());
  log('rebuild (leg_count 3) in', ((Date.now() - tb) / 1000).toFixed(1), 's;', JSON.stringify(await page.evaluate(() => window.kigumi.stats)));
  await shot('03-three-legs');
  const rtab = await page.$('[data-tab="report"]');
  if (rtab) { await rtab.tap(); await page.waitForTimeout(400); await shot('04-report-sheet'); }
} else {
  log('no mobile tabs on this page');
}
// the fabrication package from the Export tab (a zip made in the page)
if (arg('export', '')) {
  const te = Date.now();
  await page.tap('#tab-export');
  await page.waitForFunction(() => !document.querySelector('#lightbox .lb-wait') && document.querySelector('#lightbox .lb-msg'),
    null, { timeout: 600_000 });
  const files = await page.$$eval('#lightbox .files a', (as) => as.map((a) => a.textContent));
  if (!files.length) log('EXPORT FAILED:', await page.$eval('#lightbox .lb-msg', (m) => m.textContent.replace(/\s+/g, ' ').slice(0, 400)));
  else log('export in', ((Date.now() - te) / 1000).toFixed(1), 's:', files.join(', '));
  await shot('05-export');
  await page.tap('#lb-close');
}
// orbit with a one-finger drag on the canvas (touch events)
const box = await page.$eval('#gl-host canvas', (c) => { const r = c.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; });
log('canvas box', box.map((v) => Math.round(v)).join(' '));
const stats = await page.evaluate(async () => (window.kigumi.backend === 'page' ? window.kigumi.engine.api('/api/stats') : null));
log('engine stats', JSON.stringify(stats));
log('done in', secs());
await browser.close();
