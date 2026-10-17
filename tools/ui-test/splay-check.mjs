// The auto splay toggle in the page: the checkbox exists, and turning it on
// leans the feet out until the tipping check passes (where it can).
//   node splay-check.mjs [--url http://127.0.0.1:8790/]
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const env = { ...process.env };
const deps = path.join(process.env.HOME || '', '.local/lib/chromedeps');
if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps;
const browser = await chromium.launch({ headless: true, env });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.setDefaultTimeout(600_000);
await page.goto(arg('url', 'http://127.0.0.1:8790/') + '#preset=field-four');
await page.waitForFunction(() => window.kigumi && window.kigumi.engine);
await page.evaluate(() => window.kigumi.ready);
await page.evaluate(() => window.kigumi.settled());
const row = await page.$$eval('.row.bool', (rs) => rs.map((r) => r.textContent.trim()).filter((t) => /splay/i.test(t)));
console.log('checkbox:', row);
const tip = () => page.evaluate(() => (window.kigumi.report.checks || []).find((c) => c.name === 'Tipping'));
console.log('off:', JSON.stringify(await tip()));
const before = await page.evaluate(() => JSON.stringify(window.kigumi.appliedParams));
await page.evaluate(() => window.kigumi.setParams({ auto_splay: true }));
await page.waitForFunction((b) => JSON.stringify(window.kigumi.appliedParams) !== b, before);
await page.evaluate(() => window.kigumi.settled());
console.log('on:', JSON.stringify(await tip()));
console.log('notes:', JSON.stringify(await page.evaluate(() => window.kigumi.report.notes)));
await page.screenshot({ path: 'shots/splay-on.png' });
await browser.close();
