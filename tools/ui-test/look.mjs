// Dev helper: close-up screenshots of the 3D stage for judging materials.
//
//   node look.mjs [--url http://127.0.0.1:8766/] [--hash "wood_beams=walnut"] [--query "tm=aces"]
//                 [--out shots/look] [--glass 0|1] [--size 1440x900]
//
// Shots: the framed hero view, plus close-ups at a leg, a lattice corner and a cap.

import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i > 0 ? process.argv[i + 1] : d; };
const base = arg('url', 'http://127.0.0.1:8766/');
const hash = arg('hash', '');
const query = arg('query', '');
const OUT = path.resolve(arg('out', 'shots/look'));
const glass = arg('glass', '1') !== '0';
const preset = arg('preset', '');
const [W, H] = arg('size', '1440x900').split('x').map(Number);
fs.mkdirSync(OUT, { recursive: true });

const env = { ...process.env };
const deps = path.join(process.env.HOME || '', '.local/lib/chromedeps');
if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
const browser = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'], env });
const page = await browser.newPage({ viewport: { width: W, height: H } });
page.on('console', (m) => { if (m.type() === 'error') console.log('console.error', m.text()); });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto(base + (query ? '?' + query : '') + (hash ? '#' + hash : ''));
await page.waitForFunction(() => document.documentElement.dataset.ready || document.documentElement.dataset.error, null, { timeout: 120000 });
await page.evaluate(() => window.kigumi.settled());
if (preset) { await page.selectOption('#preset', preset); await page.waitForTimeout(50); await page.evaluate(() => window.kigumi.settled()); }
if (!glass) { await page.keyboard.press('g'); await page.evaluate(() => window.kigumi.settled()); }

const stage = page.locator('#stage');
await stage.screenshot({ path: path.join(OUT, 'a-hero.png') });

// close-ups chosen from the model itself
const targets = await page.evaluate(() => {
  const v = window.kigumi.viewer;
  const out = {};
  const pick = (pred) => [...v.parts.values()].find(pred);
  const wc = (p) => {
    const e = p.meshes[0].geometry.boundingBox;
    const c = [(e.min.x + e.max.x) / 2, (e.min.y + e.max.y) / 2, (e.min.z + e.max.z) / 2];
    const m = p.meshes[0].matrixWorld.elements;
    return [m[0] * c[0] + m[4] * c[1] + m[8] * c[2] + m[12], m[1] * c[0] + m[5] * c[1] + m[9] * c[2] + m[13], m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14]];
  };
  const leg = pick((p) => p.kind === 'leg');
  const beam = [...v.parts.values()].filter((p) => p.kind === 'beam').sort((a, b) => b.step - a.step)[0];
  const cap = pick((p) => p.kind === 'cap');
  if (leg) out.leg = wc(leg);
  if (beam) out.beam = wc(beam);
  if (cap) out.cap = wc(cap);
  return out;
});
const views = [
  ['b-leg', targets.leg, [0.35, 0.10, 0.45]],
  ['c-lattice', targets.beam, [0.30, 0.28, 0.42]],
  ['d-cap', targets.cap, [0.16, 0.12, 0.2]],
];
for (const [name, t, off] of views) {
  if (!t) continue;
  await page.evaluate(([t, off]) => {
    const v = window.kigumi.viewer;
    v.controls.target.set(t[0], t[1], t[2]);
    v.camera.position.set(t[0] + off[0], t[1] + off[1], t[2] + off[2]);
    v.controls.update();
    v.requestRender();
  }, [t, off]);
  await page.evaluate(() => window.kigumi.settled());
  await stage.screenshot({ path: path.join(OUT, name + '.png') });
}
console.log('wrote', OUT);
await browser.close();
