// Live-logging page load of the browser build, for debugging.
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
const URL_ = process.argv[2] || 'http://localhost:8790/';
const WAIT = Number(process.argv[3] || 120);
const env = { ...process.env };
const deps = path.join(process.env.HOME || '', '.local/lib/chromedeps');
if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
const exe = process.env.CHROME || path.join(process.env.HOME, '.cache/ms-playwright/chromium-1243/chrome-linux64/chrome');
const browser = await chromium.launch({ headless: true, env: { ...env, PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS: '1' },
  executablePath: fs.existsSync(exe) ? exe : undefined,
  args: ['--headless=new', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const t0 = Date.now();
const log = (...a) => console.log(((Date.now() - t0) / 1000).toFixed(1).padStart(6), ...a);
page.on('console', (m) => { const t = m.text(); if (!/GPU stall|GL Driver/.test(t)) log('console.' + m.type(), t.slice(0, 400)); });
page.on('pageerror', (e) => log('pageerror', e.message.slice(0, 600)));
page.on('requestfailed', (r) => log('requestfailed', r.url().slice(0, 120), r.failure()?.errorText));
await page.goto(URL_);
for (let s = 0; s < WAIT; s += 5) {
  await page.waitForTimeout(5000);
  const st = await page.evaluate(() => ({
    boot: document.querySelector('#boot-text')?.textContent || 'gone',
    backend: window.kigumi?.viewer?.backend, report: !!window.kigumi?.report,
    err: document.documentElement.dataset.error || '',
  })).catch((e) => ({ evalError: String(e).slice(0, 200) }));
  log('state', JSON.stringify(st));
  if (st.report) break;
}
await page.screenshot({ path: 'shots/site/debug.png' });
await browser.close();
