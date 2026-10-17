// Headless smoke test for the kigumi configurator.
//
//   node ui-test.mjs [--url http://127.0.0.1:8766/] [--out shots] [--size 1440x900] [--headed]
//
// Opens the page, waits for window.kigumi.ready, drives the UI (sliders,
// Randomize, Mutate, lock, explode, assembly, glass, preset, export dialog) and
// writes screenshots.  Exits non-zero on console errors, page errors or failed
// requests.  --quick skips the slow export (~25 s).

import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const arg = (name, dflt) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : dflt;
};
const URL_ = arg('url', 'http://127.0.0.1:8766/');
const OUT = path.resolve(arg('out', 'shots'));
const [W, H] = arg('size', '1440x900').split('x').map(Number);
const HEADED = process.argv.includes('--headed');
const QUICK = process.argv.includes('--quick');
fs.mkdirSync(OUT, { recursive: true });

const problems = [];
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

// WSL boxes without system Chrome deps keep them in ~/.local/lib/chromedeps
const env = { ...process.env };
const deps = path.join(process.env.HOME || '', '.local/lib/chromedeps');
if (fs.existsSync(deps)) env.LD_LIBRARY_PATH = deps + (env.LD_LIBRARY_PATH ? ':' + env.LD_LIBRARY_PATH : '');
const browser = await chromium.launch({
  headless: !HEADED,
  args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  env,
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.setDefaultTimeout(120_000);

page.on('console', (msg) => {
  const t = msg.type();
  const text = msg.text();
  if (t === 'error') {
    problems.push(`console.${t}: ${text} ${msg.location()?.url || ''}`);
  } else if (t === 'warning') {
    log('warn:', text.slice(0, 200));
  }
});
page.on('pageerror', (err) => problems.push('pageerror: ' + (err.stack || err.message)));
page.on('requestfailed', (req) => problems.push(`requestfailed: ${req.method()} ${req.url()} ${req.failure()?.errorText}`));
page.on('response', (res) => {
  if (res.status() >= 400) problems.push(`HTTP ${res.status()} ${res.url()}`);
});

const settled = () => page.evaluate(() => window.kigumi.settled());
const shot = async (name) => {
  const f = path.join(OUT, name);
  await page.screenshot({ path: f });
  log('screenshot', f);
};
const params = () => page.evaluate(() => window.kigumi.params);

try {
  const t0 = Date.now();
  await page.goto(URL_, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.kigumi && (document.documentElement.dataset.ready || document.documentElement.dataset.error), null, { timeout: 120_000 });
  const err = await page.evaluate(() => document.documentElement.dataset.error);
  if (err) throw new Error('page reported error: ' + err);
  await page.evaluate(() => window.kigumi.ready);
  await settled();
  log(`ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  await shot('01-initial.png');

  // schema-driven sanity: groups and controls exist
  const nGroups = await page.locator('#params .group').count();
  const nRows = await page.locator('#params .row').count();
  log(`panel: ${nGroups} groups, ${nRows} rows`);
  if (nGroups < 2 || nRows < 10) problems.push(`parameter panel looks empty (${nGroups} groups, ${nRows} rows)`);
  const dims = await page.locator('#report .dims').innerText();
  log('report size:', dims.replace(/\s+/g, ' '));
  const nChecks = await page.locator('#checks-sec .checks li').count();
  const checkHead = (await page.locator('#checks-sec h3').innerText()).replace(/\s+/g, ' ');
  log(`checks: ${nChecks} (${checkHead})`);
  if (!nChecks) problems.push('engineering checks never arrived in the report panel');
  const stats = await page.evaluate(() => window.kigumi.stats);
  if (stats) log('load stats:', JSON.stringify(stats));

  // --- change three sliders (preferring names we know, else the first sliders)
  const want = ['tiers', 'pitch', 'leg_count'];
  const sliders = [];
  for (const n of want) if (await page.locator(`.row[data-name="${n}"] input.slider`).count()) sliders.push(n);
  if (sliders.length < 3) {
    const names = await page.$$eval('.row.num', (rows) => rows.filter((r) => r.querySelector('input.slider')).map((r) => r.dataset.name));
    for (const n of names) if (sliders.length < 3 && !sliders.includes(n)) sliders.push(n);
  }
  for (const n of sliders) {
    const sl = page.locator(`.row[data-name="${n}"] input.slider`);
    const [lo, hi, v, st] = await sl.evaluate((e) => [Number(e.min), Number(e.max), Number(e.value), Number(e.step) || 1]);
    let target = v + (hi - lo) * 0.18 <= hi ? v + (hi - lo) * 0.18 : v - (hi - lo) * 0.18;
    target = Math.min(hi, Math.max(lo, lo + Math.round((target - lo) / st) * st));
    await sl.fill(String(+target.toFixed(6)));
    log(`slider ${n}: ${v} -> ${await sl.inputValue()}`);
  }
  await settled();
  const p1 = await params();
  for (const n of sliders) log(`  ${n} = ${p1[n]}`);
  await shot('02-sliders.png');

  // --- rapid slider changes: only the newest value may end up applied
  {
    const n = sliders[0];
    const sl = page.locator(`.row[data-name="${n}"] input.slider`);
    const [lo, hi, st] = await sl.evaluate((e) => [Number(e.min), Number(e.max), Number(e.step) || 1]);
    const vals = [0.2, 0.8, 0.4, 0.9, 0.55].map((f) => +(lo + Math.round((hi - lo) * f / st) * st).toFixed(6));
    for (const v of vals) { await sl.fill(String(v)); await page.waitForTimeout(90); }
    await settled();
    const applied = await page.evaluate(() => window.kigumi.appliedParams);
    const cur = await params();
    log(`rapid ${n}: sent ${vals.join(' → ')}; applied ${applied?.[n]}, panel ${cur[n]}`);
    if (!applied || Number(applied[n]) !== vals[vals.length - 1] || Number(cur[n]) !== vals[vals.length - 1]) {
      problems.push(`rapid slider: expected ${vals[vals.length - 1]}, applied ${applied?.[n]}, panel ${cur[n]}`);
    }
  }

  // --- lock a group, randomize, verify the locked group is kept
  const lockGroup = await page.evaluate(() => {
    const g = [...document.querySelectorAll('#params .group')].find((s) => s.dataset.group === 'Materials') || document.querySelectorAll('#params .group')[1];
    return g.dataset.group;
  });
  const before = await params();
  await page.locator(`.group[data-group="${lockGroup}"] .lock`).click();
  await page.locator('#btn-random').click();
  await page.waitForTimeout(50);
  await settled();
  const after = await params();
  const lockedNames = await page.$$eval(`.group[data-group="${lockGroup}"] .row`, (rows) => rows.map((r) => r.dataset.name));
  const brokeLock = lockedNames.filter((n) => n !== 'seed' && JSON.stringify(before[n]) !== JSON.stringify(after[n]));
  const changedCount = Object.keys(after).filter((k) => JSON.stringify(before[k]) !== JSON.stringify(after[k])).length;
  log(`randomize with "${lockGroup}" locked: ${changedCount} params changed; lock violations: ${brokeLock.join(', ') || 'none'}`);
  if (brokeLock.length) problems.push(`locked group ${lockGroup} changed on randomize: ${brokeLock.join(', ')}`);
  if (changedCount < 3) problems.push('randomize changed fewer than 3 parameters');
  await shot('03-randomized.png');
  await page.locator(`.group[data-group="${lockGroup}"] .lock`).click();   // unlock again

  // --- undo brings back the design from before Randomize, redo returns
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(50);
  await settled();
  const undone = await params();
  const undoOk = Object.keys(before).every((k) => k === 'seed' || JSON.stringify(before[k]) === JSON.stringify(undone[k]));
  log(`undo restores pre-randomize design: ${undoOk}`);
  if (!undoOk) problems.push('Ctrl+Z did not restore the design from before Randomize');
  await page.keyboard.press('Control+Shift+z');
  await page.waitForTimeout(50);
  await settled();
  const redone = await params();
  const redoOk = Object.keys(after).every((k) => JSON.stringify(after[k]) === JSON.stringify(redone[k]));
  log(`redo returns to the randomized design: ${redoOk}`);
  if (!redoOk) problems.push('Ctrl+Shift+Z did not return to the randomized design');

  // --- parameter filter
  await page.keyboard.press('/');
  await page.keyboard.type('glass');
  const shown = await page.$$eval('#params .row', (rows) => rows.filter((r) => !r.hidden).map((r) => r.dataset.name));
  log(`filter "glass": ${shown.length} rows (${shown.slice(0, 6).join(', ')}${shown.length > 6 ? ', …' : ''})`);
  if (!shown.length || shown.length > 20) problems.push(`filter "glass" shows ${shown.length} rows`);
  await shot('03b-filter.png');
  await page.keyboard.press('Escape');

  // --- mutate via keyboard: a gentle nudge -- numbers move at most ~12% of their range
  await page.locator('#stage').click({ position: { x: 30, y: 60 } });
  const bm = await params();
  await page.keyboard.press('m');
  await page.waitForTimeout(50);
  await settled();
  const am = await params();
  const mutated = Object.keys(am).filter((k) => k !== 'seed' && JSON.stringify(bm[k]) !== JSON.stringify(am[k]));
  log(`mutate changed ${mutated.length} params`);
  if (!mutated.length) problems.push('mutate changed nothing');
  const schema = await page.evaluate(() => fetch('/api/schema').then((r) => r.json()).then((s) => s.params));
  const big = [];
  for (const p of schema) {
    if (!mutated.includes(p.name) || typeof bm[p.name] !== 'number' || p.lo == null || p.hi == null || p.hi <= p.lo) continue;
    const delta = Math.abs(am[p.name] - bm[p.name]);
    if (p.type === 'int' && delta <= 1) continue;   // integers may move one whole step
    const frac = delta / (p.hi - p.lo);
    if (frac > 0.13) big.push(`${p.name} ${bm[p.name]} -> ${am[p.name]}`);
  }
  if (big.length) problems.push(`mutate jumped too far: ${big.join(', ')}`);

  // --- explode to max
  await page.locator('#explode').fill('1');
  await settled();
  await shot('04-explode-max.png');

  // --- assembly step 3 (explode back to 0 first)
  await page.locator('#explode').fill('0');
  await page.locator('#step').fill('3');
  await page.waitForTimeout(50);
  await settled();
  log('assembly caption:', (await page.locator('.stepcap').innerText()).replace(/\s+/g, ' '));
  await shot('05-assembly-step3.png');
  const nSteps = await page.locator('#step').evaluate((e) => e.max);
  await page.locator('#step').fill(String(nSteps));
  await settled();

  // --- E toggles the explode, Play runs the assembly to the end
  await page.keyboard.press('e');
  await settled();
  const exOn = await page.evaluate(() => window.kigumi.viewer.explode);
  await page.keyboard.press('e');
  await settled();
  const exOff = await page.evaluate(() => window.kigumi.viewer.explode);
  log(`E toggle: explode ${exOn.toFixed(2)} -> ${exOff.toFixed(2)}`);
  if (!(exOn > 0.1) || exOff !== 0) problems.push(`E toggle did not explode and reassemble (${exOn}, ${exOff})`);
  await page.locator('#step').fill('1');
  await settled();
  const t2 = Date.now();
  await page.locator('#play').click();
  await page.waitForTimeout(50);
  await settled();
  const [k, N] = await page.evaluate(() => [window.kigumi.viewer.stepK, window.kigumi.viewer.steps.length]);
  log(`play: reached step ${k}/${N} in ${((Date.now() - t2) / 1000).toFixed(1)} s`);
  if (k !== N) problems.push(`play stopped at step ${k} of ${N}`);

  // --- glass toggle (G) and back
  await page.keyboard.press('g');
  await settled();
  const glassOff = await page.evaluate(() => !window.kigumi.viewer.glassVisible);
  await shot('06-glass-off.png');
  await page.keyboard.press('g');
  await settled();
  if (!glassOff) problems.push('G did not hide the glass');

  // --- preset (last one in the list) frames the camera
  const presetNames = await page.$$eval('#preset option', (o) => o.map((x) => x.value).filter(Boolean));
  if (presetNames.length) {
    const pick = presetNames[Math.min(presetNames.length - 1, 1)];
    await page.selectOption('#preset', pick);
    await page.waitForTimeout(50);
    await settled();
    const sel = await page.locator('#preset').inputValue();
    log(`preset ${pick}: select shows "${sel}"`);
    await shot('07-preset.png');
  }

  // --- hover tooltip over the centre of the view
  const box = await page.locator('#stage').boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.45);
  await page.waitForTimeout(300);
  await settled();
  const tip = await page.locator('#tip').innerText().catch(() => '');
  log('hover tip:', tip.replace(/\s+/g, ' ').slice(0, 120) || '(none)');
  await shot('08-hover.png');
  await page.mouse.move(box.x + 5, box.y + 5);

  // --- no glass: the glass chip disables, the model has no glass part
  const kindRow = await page.$$eval('.row', (rows) => {
    const r = rows.find((x) => [...x.querySelectorAll('[data-v], option')].some((o) => (o.dataset?.v ?? o.value) === 'none'));
    return r ? r.dataset.name : null;
  });
  if (kindRow) {
    const before = await page.evaluate((n) => window.kigumi.params[n], kindRow);
    const seg = page.locator(`.row[data-name="${kindRow}"] button[data-v="none"]`);
    if (await seg.count()) await seg.click(); else await page.selectOption(`.row[data-name="${kindRow}"] select`, 'none');
    await page.waitForTimeout(50);
    await settled();
    const glassDisabled = await page.locator('#btn-glass').isDisabled();
    const hasGlass = await page.evaluate(() => window.kigumi.viewer.hasGlass);
    log(`${kindRow}=none: glass chip disabled=${glassDisabled}, model has glass=${hasGlass}`);
    if (hasGlass || !glassDisabled) problems.push(`${kindRow}=none still shows glass`);
    await shot('09-no-glass.png');
    await page.evaluate(([n, v]) => window.kigumi.setParams({ [n]: v }), [kindRow, before]);
    await page.waitForTimeout(50);
    await settled();
  }

  // --- render photo (Blender Cycles on the server)
  if (!QUICK) {
    const t1 = Date.now();
    await page.locator('#btn-render').click();
    await page.waitForFunction(() => document.querySelector('#lb-body img') || document.querySelector('#lb-body .lb-msg'), null, { timeout: 240_000 });
    const img = page.locator('#lb-body img');
    if (await img.count()) {
      await page.waitForFunction(() => { const i = document.querySelector('#lb-body img'); return i && i.complete && i.naturalWidth > 0; }, null, { timeout: 60_000 });
      const [w, h] = await img.evaluate((i) => [i.naturalWidth, i.naturalHeight]);
      log(`render photo: ${((Date.now() - t1) / 1000).toFixed(1)} s, ${w}x${h}`);
    } else {
      problems.push('render photo failed: ' + (await page.locator('#lb-body').innerText()).slice(0, 300));
    }
    await shot('12-photo.png');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(100);
  }

  // --- export (writes CAD, drawings, cut list; ~25 s)
  if (!QUICK) {
    const t1 = Date.now();
    await page.locator('#btn-export').click();
    await page.waitForFunction(() => !document.querySelector('#lb-body .spinner'), null, { timeout: 240_000 });
    const text = await page.locator('#lb-body').innerText();
    const nFiles = await page.locator('#lb-body .files a').count();
    log(`export: ${((Date.now() - t1) / 1000).toFixed(1)} s, ${nFiles} files — ${text.split('\n')[0]}`);
    if (!nFiles) problems.push('export listed no files: ' + text.slice(0, 300));
    else {
      const href = await page.locator('#lb-body .files a').first().getAttribute('href');
      const r = await page.request.get(new URL(href, URL_).toString());
      log(`first export file ${href}: HTTP ${r.status()}`);
      if (!r.ok()) problems.push(`export file not downloadable: ${href}`);
    }
    await shot('11-export.png');
    await page.keyboard.press('Escape');
    await page.waitForTimeout(100);
  }

  // --- narrow layout
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.waitForTimeout(200);
  await settled();
  await shot('10-1280x800.png');
} catch (e) {
  problems.push('test error: ' + (e.stack || e.message));
  await shot('zz-failure.png').catch(() => {});
} finally {
  await browser.close();
}

if (problems.length) {
  console.error('\nFAIL');
  for (const p of problems) console.error(' -', p);
  process.exit(1);
}
log('PASS');
