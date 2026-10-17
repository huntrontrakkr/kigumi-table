// Profile one build of the in-page engine in a given browser (cProfile output).
//   node engine-profile.mjs --browser webkit --params '{"leg_count":5}'
import { chromium, webkit, devices } from 'playwright';
import fs from 'fs';
import path from 'path';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const which = arg('browser', 'webkit');
const env = { ...process.env };
const libs = ['webkitdeps', 'chromedeps'].map((d) => path.join(process.env.HOME || '', '.local/lib', d)).filter((d) => fs.existsSync(d));
if (libs.length) env.LD_LIBRARY_PATH = libs.join(':');
const gio = path.join(process.env.HOME || '', '.local/lib/webkitdeps/gio');
if (fs.existsSync(gio)) env.GIO_EXTRA_MODULES = gio;
const browser = await (which === 'webkit' ? webkit : chromium).launch({ headless: true, env });
const page = await (await browser.newContext({ ...devices[which === 'webkit' ? 'iPhone 15' : 'Pixel 7'] })).newPage();
page.setDefaultTimeout(900_000);
await page.goto(arg('url', 'http://127.0.0.1:8790/'));
await page.waitForFunction(() => window.kigumi && window.kigumi.engine);
await page.evaluate(() => window.kigumi.ready);
await page.evaluate(() => window.kigumi.settled());
const params = JSON.parse(arg('params', '{"leg_count":5}'));
const t = Date.now();
const out = await page.evaluate((p) => window.kigumi.engine.api('/api/profile', { params: { ...window.kigumi.params, ...p } }), params);
console.log(`${which}: profiled build in ${((Date.now() - t) / 1000).toFixed(1)} s`);
console.log(out.split('\n').filter((l) => /kigumi|gstrf|function calls|_m3d|shapely|contour/.test(l)).slice(0, 34).join('\n'));
await browser.close();
