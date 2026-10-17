// kigumi table — browser configurator.  Everything about parameters comes from
// /api/schema; this file only knows the API shapes.  In this build the API is
// answered by the engine running in the page (engine/: Python in WebAssembly,
// Manifold, WebGPU) instead of a server.

import { Viewer } from './viewer.js';
import { swatch } from './materials.js';
import { engine } from './engine/client.js';

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pretty = (s) => String(s ?? '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
const cap = (s) => { s = pretty(s); return s.charAt(0).toUpperCase() + s.slice(1); };
const store = {
  get(k, d) { try { const v = localStorage.getItem('kigumi.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('kigumi.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ---------------------------------------------------------------- state
const state = {
  schema: [],
  byName: new Map(),
  groups: [],                       // [{name, params: [...]}]
  presets: {},
  defaults: {},
  params: {},
  locked: new Set(store.get('locked', [])),
  collapsed: new Set(store.get('collapsed', [])),
  report: null,
  buildId: null,
  appliedJSON: '',
  playing: false,
};
const controls = new Map();         // name -> {set(v), row}

let viewer;
let resolveReady, rejectReady;
const ready = new Promise((res, rej) => { resolveReady = res; rejectReady = rej; });
ready.catch(() => {});

// ---------------------------------------------------------------- API
async function api(path, body) {
  if (path === '/api/render') return renderLocally(body);
  try {
    return await engine.api(path, body);
  } catch (e) {
    if (!e.trace) e.trace = e.stack;
    throw e;
  }
}

// Photo: a high-resolution render of the live scene on the GPU (see viewer.photo)
async function renderLocally(body) {
  const blob = await viewer.photo({ view: (body && body.view) || 'hero' });
  return { png: URL.createObjectURL(blob) };
}

// ---------------------------------------------------------------- param helpers
function isNumeric(p) { return (p.type === 'int' || p.type === 'float') && !p.choices; }
function decimals(p) {
  if (p.type === 'int') return 0;
  const st = Number(p.step);
  if (!st || !isFinite(st)) return 2;
  return Math.max(0, Math.min(4, -Math.floor(Math.log10(st) + 1e-9)));
}
function coerce(p, v) {
  if (p.choices && p.choices.length) {
    const hit = p.choices.find((c) => String(c) === String(v));
    return hit === undefined ? p.default : hit;
  }
  switch (p.type) {
    case 'int': {
      let n = Math.round(Number(v));
      if (!isFinite(n)) return p.default;
      if (p.lo != null) n = Math.max(p.lo, n);
      if (p.hi != null) n = Math.min(p.hi, n);
      return n;
    }
    case 'float': {
      let n = Number(v);
      if (!isFinite(n)) return p.default;
      if (p.lo != null) n = Math.max(p.lo, n);
      if (p.hi != null) n = Math.min(p.hi, n);
      return n;
    }
    case 'bool': return v === true || v === 'true' || v === 1 || v === '1';
    default: return v;
  }
}
function fmtValue(p, v) {
  if (isNumeric(p)) return Number(v).toFixed(decimals(p));
  return String(v);
}
function sameValue(p, a, b) {
  if (isNumeric(p)) return Math.abs(Number(a) - Number(b)) < 1e-9;
  return String(a) === String(b);
}
// huge ranges (seeds, ids) get a number field but no slider and are never mutated
function isIdLike(p) {
  return isNumeric(p) && p.lo != null && p.hi != null && (p.hi - p.lo) / (Number(p.step) || 1) > 1e5;
}
function lockedNames() {
  return state.schema.filter((p) => state.locked.has(p.group || 'Other')).map((p) => p.name);
}

// ---------------------------------------------------------------- icons
const ICON = {
  chevron: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M3 2 L7 5 L3 8" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  lockOpen: '<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6.5" width="9" height="6" rx="1" fill="none" stroke="currentColor"/><path d="M4.5 6.5 V4.4 a2.5 2.5 0 0 1 4.9-.7" fill="none" stroke="currentColor"/></svg>',
  lockShut: '<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6.5" width="9" height="6" rx="1" fill="currentColor" stroke="currentColor"/><path d="M4.5 6.5 V4.4 a2.5 2.5 0 0 1 5 0 V6.5" fill="none" stroke="currentColor"/></svg>',
  play: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3.5 2.2 L9.6 6 L3.5 9.8 Z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="3" y="2.4" width="2" height="7.2" fill="currentColor"/><rect x="7" y="2.4" width="2" height="7.2" fill="currentColor"/></svg>',
};

// ---------------------------------------------------------------- parameter panel
function buildPanel() {
  const panel = $('#params');
  panel.innerHTML = '';
  for (const g of state.groups) {
    const sec = document.createElement('section');
    sec.className = 'group';
    sec.dataset.group = g.name;
    if (state.collapsed.has(g.name)) sec.classList.add('collapsed');
    if (state.locked.has(g.name)) sec.classList.add('locked');
    sec.innerHTML = `
      <header class="group-head">
        <button type="button" class="group-toggle" aria-expanded="${!state.collapsed.has(g.name)}">
          <span class="chev">${ICON.chevron}</span><span class="group-name">${esc(g.name)}</span>
          <span class="group-count">${g.params.length}</span>
        </button>
        <button type="button" class="lock" aria-pressed="${state.locked.has(g.name)}"
          title="Lock ${esc(g.name)} — kept when randomizing or mutating">${state.locked.has(g.name) ? ICON.lockShut : ICON.lockOpen}</button>
      </header>
      <div class="group-body"></div>`;
    const body = $('.group-body', sec);
    for (const p of g.params) body.appendChild(makeRow(p));
    $('.group-toggle', sec).addEventListener('click', () => {
      const c = !sec.classList.contains('collapsed');
      sec.classList.toggle('collapsed', c);
      $('.group-toggle', sec).setAttribute('aria-expanded', String(!c));
      if (c) state.collapsed.add(g.name); else state.collapsed.delete(g.name);
      store.set('collapsed', [...state.collapsed]);
    });
    const lockBtn = $('.lock', sec);
    lockBtn.addEventListener('click', () => {
      const on = !state.locked.has(g.name);
      if (on) state.locked.add(g.name); else state.locked.delete(g.name);
      sec.classList.toggle('locked', on);
      lockBtn.setAttribute('aria-pressed', String(on));
      lockBtn.innerHTML = on ? ICON.lockShut : ICON.lockOpen;
      store.set('locked', [...state.locked]);
    });
    panel.appendChild(sec);
  }
}

function rowTitle(p) {
  const bits = [];
  if (p.help) bits.push(p.help);
  if (isNumeric(p) && p.lo != null && p.hi != null) bits.push(`range ${fmtValue(p, p.lo)} – ${fmtValue(p, p.hi)}`);
  bits.push(`default ${fmtValue(p, p.default)}`);
  bits.push(`${p.name} · double-click the label to reset`);
  return bits.join('\n');
}

function makeRow(p) {
  const row = document.createElement('div');
  row.className = 'row';
  row.dataset.name = p.name;
  row.title = rowTitle(p);
  const id = 'p-' + p.name;
  const label = p.label || cap(p.name);
  let set;

  if (p.choices && p.choices.length) {
    row.classList.add('choice');
    row.innerHTML = `<label class="lbl" for="${id}">${esc(label)}</label><select id="${id}">${
      p.choices.map((c) => `<option value="${esc(c)}">${esc(pretty(c))}</option>`).join('')}</select>`;
    const sel = $('select', row);
    sel.addEventListener('change', () => userSet(p, sel.value));
    set = (v) => { sel.value = String(v); };
  } else if (p.type === 'bool') {
    row.classList.add('bool');
    row.innerHTML = `<label class="lbl check" for="${id}"><input type="checkbox" id="${id}"><span>${esc(label)}</span></label>`;
    const cb = $('input', row);
    cb.addEventListener('change', () => userSet(p, cb.checked));
    set = (v) => { cb.checked = !!v; };
  } else if (isNumeric(p)) {
    row.classList.add('num');
    const dec = decimals(p);
    const step = p.step || (p.type === 'int' ? 1 : 'any');
    const hasRange = p.lo != null && p.hi != null && p.hi > p.lo && !isIdLike(p);
    row.innerHTML = `<label class="lbl" for="${id}">${esc(label)}</label>
      <input class="val" id="${id}" type="number" inputmode="decimal" step="${step}"
        ${p.lo != null ? `min="${p.lo}"` : ''} ${p.hi != null ? `max="${p.hi}"` : ''}>
      ${hasRange ? `<input class="slider" type="range" min="${p.lo}" max="${p.hi}" step="${step}" aria-label="${esc(label)}" tabindex="-1">` : ''}`;
    const val = $('.val', row);
    const sl = $('.slider', row);
    const fill = (v) => { if (sl) sl.style.setProperty('--fill', ((v - p.lo) / (p.hi - p.lo) * 100).toFixed(2) + '%'); };
    set = (v) => {
      if (document.activeElement !== val) val.value = Number(v).toFixed(dec);
      if (sl) { sl.value = v; fill(v); }
    };
    if (sl) {
      sl.addEventListener('input', () => {
        const v = coerce(p, sl.value);
        val.value = Number(v).toFixed(dec);
        fill(v);
        userSet(p, v, { fromSlider: true });
      });
      sl.addEventListener('pointerdown', () => row.classList.add('active'));
      const up = () => row.classList.remove('active');
      sl.addEventListener('pointerup', up);
      sl.addEventListener('blur', up);
    }
    val.addEventListener('change', () => {
      const v = coerce(p, val.value);
      val.value = Number(v).toFixed(dec);
      userSet(p, v);
    });
    val.addEventListener('keydown', (e) => { if (e.key === 'Enter') val.blur(); });
  } else {
    row.classList.add('text');
    row.innerHTML = `<label class="lbl" for="${id}">${esc(label)}</label><input class="txt" id="${id}" type="text">`;
    const t = $('input', row);
    t.addEventListener('change', () => userSet(p, t.value));
    set = (v) => { t.value = typeof v === 'string' ? v : JSON.stringify(v); };
  }
  $('.lbl', row)?.addEventListener('dblclick', (e) => { e.preventDefault(); userSet(p, p.default); });
  controls.set(p.name, { set: (v) => { set(v); row.classList.toggle('changed', !sameValue(p, v, p.default)); }, row });
  return row;
}

function refreshControls() {
  for (const p of state.schema) controls.get(p.name)?.set(state.params[p.name]);
  updatePresetSelect();
  updateSeed();
}

function userSet(p, raw) {
  const v = coerce(p, raw);
  state.params[p.name] = v;
  controls.get(p.name)?.set(v);
  updatePresetSelect();
  updateSeed();
  scheduleBuild();
}

// ---------------------------------------------------------------- presets, seed
function presetMatch() {
  for (const [name, pp] of Object.entries(state.presets)) {
    let ok = true;
    for (const p of state.schema) {
      if (p.name in pp && !sameValue(p, pp[p.name], state.params[p.name])) { ok = false; break; }
    }
    if (ok) return name;
  }
  return '';
}
function updatePresetSelect() {
  const sel = $('#preset');
  const m = presetMatch();
  sel.value = m;
  sel.classList.toggle('custom', !m);
}
function seedParam() {
  return state.schema.find((p) => p.name === 'seed') || state.schema.find(isIdLike);
}
function updateSeed() {
  const sp = seedParam();
  const el = $('#seed');
  if (!sp) { $('.seedbox').hidden = true; return; }
  if (document.activeElement !== el) el.value = state.params[sp.name] ?? '';
}

// ---------------------------------------------------------------- build pipeline
// One request at a time; the newest intent wins.  Slider builds are debounced
// (with a max wait so long drags still update live); random/preset jobs replace
// any pending build and make an in-flight result stale.
const jobs = { running: null, pending: null, debounce: null, maxWait: null, seq: 0, applied: 0 };

function scheduleBuild() {
  clearTimeout(jobs.debounce);
  jobs.debounce = setTimeout(fireBuild, 200);
  if (!jobs.maxWait) jobs.maxWait = setTimeout(fireBuild, 650);
  writeHashSoon();
  updateBusy();
}
function fireBuild() {
  clearTimeout(jobs.debounce); jobs.debounce = null;
  clearTimeout(jobs.maxWait); jobs.maxWait = null;
  submit({ kind: 'build', url: '/api/build', body: () => ({ params: { ...state.params } }) });
}
function submit(job) {
  if (job.kind !== 'build') {
    clearTimeout(jobs.debounce); jobs.debounce = null;
    clearTimeout(jobs.maxWait); jobs.maxWait = null;
    jobs.pending = job;
  } else if (!jobs.pending || jobs.pending.kind === 'build') {
    jobs.pending = job;
  }
  pump();
}
async function pump() {
  updateBusy();
  if (jobs.running || !jobs.pending) return;
  const job = jobs.pending;
  jobs.pending = null;
  const body = job.body();
  if (job.kind === 'build') {
    job.sent = JSON.stringify(body.params);
    if (job.sent === state.appliedJSON) { updateBusy(); return pump(); }
  }
  jobs.running = job;
  const seq = ++jobs.seq;
  updateBusy();
  const stale = () => seq < jobs.applied || (jobs.pending && jobs.pending.kind !== 'build');
  try {
    const t0 = performance.now();
    let res = await api(job.url, body);
    if (job.after) res = await job.after(res);
    const t1 = performance.now();
    if (!stale()) {
      const r = await fetch(res.glb);
      if (!r.ok) throw new Error(`model download failed (HTTP ${r.status})`);
      const buf = await r.arrayBuffer();
      const t2 = performance.now();
      if (!stale()) {
        const first = !state.report;
        const st = await viewer.loadModel(buf, res.report, { frame: first ? true : job.frame ? 'animate' : false });
        window.kigumi.stats = { kind: job.kind, request_ms: Math.round(t1 - t0), glb_ms: Math.round(t2 - t1),
          glb_kb: Math.round(buf.byteLength / 1024), ...st, total_ms: Math.round(performance.now() - t0) };
        jobs.applied = seq;
        applyResult(res, job);
        if (first) {
          delete document.documentElement.dataset.error;
          document.documentElement.dataset.ready = '1';
          resolveReady();
        }
      }
    }
    hideToast('error');
  } catch (e) {
    console.warn('kigumi:', e);
    if (!state.report) {
      document.documentElement.dataset.error = e.message;
      rejectReady(e);
      showToast(`Couldn't build the first design: ${e.message}`, 'error', e.trace);
    } else {
      showToast(job.kind === 'build' ? `This combination didn't build: ${e.message}` : `${cap(job.kind)} failed: ${e.message}`, 'error', e.trace);
    }
  } finally {
    jobs.running = null;
    pump();
  }
}

function applyResult(res, job) {
  if (res.params && (job.kind !== 'build')) {
    for (const p of state.schema) {
      if (p.name in res.params) state.params[p.name] = coerce(p, res.params[p.name]);
    }
    refreshControls();
    writeHashSoon(0);
  }
  state.appliedJSON = job.kind === 'build' ? job.sent : JSON.stringify(state.params);
  if (job.kind !== 'history' && !jobs.pending && !jobs.debounce) recordHistory();
  state.report = res.report;
  state.buildId = res.id;
  renderReport(res.report, res.id);
  loadChecks(res.report, res.id);
  showTip(null);
  updateDock();
  updateSeed();
  $('#btn-glass').disabled = !viewer.hasGlass;
  $('#btn-glass').setAttribute('aria-pressed', String(viewer.hasGlass && viewer.glassVisible));
}

function updateBusy() {
  const busy = !!(jobs.running || jobs.pending || jobs.debounce);
  $('#stage').classList.toggle('busy', busy);
  const label = jobs.running ? ({ random: 'sampling', mutate: 'mutating' }[jobs.running.kind] || 'building') : '';
  $('#busy-label').textContent = label;
}

// ---------------------------------------------------------------- undo / redo of whole designs
// A design enters the history once it has been built and nothing newer is queued,
// so a slider drag is one step, not dozens.
const hist = { stack: [], index: -1 };
function recordHistory() {
  const j = JSON.stringify(state.params);
  if (hist.stack[hist.index] === j) return;
  hist.stack = hist.stack.slice(0, hist.index + 1);
  hist.stack.push(j);
  if (hist.stack.length > 200) hist.stack.shift();
  hist.index = hist.stack.length - 1;
  updateHistButtons();
}
function updateHistButtons() {
  $('#btn-undo').disabled = hist.index <= 0;
  $('#btn-redo').disabled = hist.index >= hist.stack.length - 1;
}
function stepHistory(d) {
  const i = hist.index + d;
  if (i < 0 || i >= hist.stack.length) return;
  hist.index = i;
  updateHistButtons();
  const params = JSON.parse(hist.stack[i]);
  for (const p of state.schema) if (p.name in params) state.params[p.name] = coerce(p, params[p.name]);
  refreshControls();
  writeHashSoon(0);
  submit({ kind: 'history', url: '/api/build', body: () => ({ params: { ...state.params } }), frame: false });
}

// ---------------------------------------------------------------- parameter filter
function applyFilter(q) {
  q = q.trim().toLowerCase();
  let total = 0;
  for (const sec of $$('#params .group')) {
    let any = 0;
    for (const row of $$('.row', sec)) {
      const p = state.byName.get(row.dataset.name);
      const hay = `${p.label || ''} ${p.name} ${p.help || ''} ${sec.dataset.group}`.toLowerCase();
      const hit = !q || hay.includes(q);
      row.hidden = !hit;
      if (hit) any++;
    }
    sec.hidden = !any;
    sec.classList.toggle('filtering', !!q);
    total += any;
  }
  let empty = $('#params .empty-filter');
  if (!total && q) {
    if (!empty) { empty = document.createElement('p'); empty.className = 'empty-filter'; $('#params').appendChild(empty); }
    empty.textContent = `Nothing matches “${q}”.`;
  } else if (empty) empty.remove();
}

// ---------------------------------------------------------------- randomize / mutate / presets
function enforceLocks(keep) {
  // the sampler may still adjust correlated parameters; put locked ones back
  return async (res) => {
    const fixed = { ...res.params };
    let changed = false;
    for (const name of keep) {
      const p = state.byName.get(name);
      if (!p || name === seedParam()?.name) continue;
      if (name in fixed && !sameValue(p, fixed[name], state.params[name])) { fixed[name] = state.params[name]; changed = true; }
    }
    if (!changed) return res;
    return api('/api/build', { params: fixed });
  };
}
function randomize(seed) {
  const keep = lockedNames();
  const body = { base: { ...state.params }, keep };
  if (seed !== undefined && seed !== null && seed !== '') body.seed = Math.max(0, Math.round(Number(seed)) || 0);
  submit({ kind: 'random', url: '/api/random', body: () => body, frame: true, after: enforceLocks(keep) });
}
function mutate() {
  // gentle mutation on the server: every unlocked number moves a little (jitter is a
  // fraction of its range), choices flip rarely, and the soundest of a few candidates wins
  const keep = lockedNames();
  const unlocked = state.schema.filter((p) => !keep.includes(p.name) && !isIdLike(p));
  if (!unlocked.length) { showToast('Everything is locked — unlock a group to mutate.', 'info'); return; }
  const body = { base: { ...state.params }, keep, jitter: 0.12, candidates: 8 };
  showToast('Nudging the unlocked parameters', 'info');
  submit({ kind: 'mutate', url: '/api/random', body: () => body, frame: false, after: enforceLocks(keep) });
}
function applyPreset(name) {
  const pp = state.presets[name];
  if (!pp) return;
  const params = { ...state.defaults };
  for (const p of state.schema) if (p.name in pp) params[p.name] = coerce(p, pp[p.name]);
  submit({ kind: 'preset', url: '/api/build', body: () => ({ params }), frame: true });
}

// ---------------------------------------------------------------- URL hash (non-default params)
let hashTimer = null;
function writeHashSoon(delay = 400) {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const q = new URLSearchParams();
    for (const p of state.schema) {
      const v = state.params[p.name];
      if (!sameValue(p, v, p.default)) q.set(p.name, p.type === 'bool' ? (v ? '1' : '0') : String(v));
    }
    const s = q.toString();
    history.replaceState(null, '', s ? '#' + s : location.pathname + location.search);
  }, delay);
}
function readHash() {
  const out = {};
  const h = location.hash.replace(/^#/, '');
  if (!h) return out;
  const q = new URLSearchParams(h);
  for (const [k, v] of q) {
    const p = state.byName.get(k);
    if (p) out[k] = coerce(p, v);
  }
  return out;
}

// ---------------------------------------------------------------- report panel
const KIND_ORDER = ['leg', 'post', 'beam', 'block', 'washer', 'pad', 'glass', 'bushing', 'cap'];
const plural = (k, n) => (n === 1 ? k : k === 'glass' ? 'glass' : k.endsWith('sh') ? k + 'es' : k + 's');
const num = (v, d = 0) => (v === null || v === undefined || v === '' || isNaN(Number(v))) ? esc(v) : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const mm = (v) => (v === null || v === undefined || isNaN(Number(v))) ? esc(v) : String(Math.round(Number(v)));
const kgText = (kg) => kg >= 1 ? `${num(kg, 2)} kg` : kg >= 0.001 ? `${num(kg * 1000, 0)} g` : '&lt; 1 g';
function fmtAny(v) {
  if (typeof v === 'number') return Math.abs(v) >= 100 ? num(v, 0) : Math.abs(v) >= 1 ? num(v, 2).replace(/\.?0+$/, '') : num(v, 3).replace(/\.?0+$/, '');
  if (Array.isArray(v)) return v.map(fmtAny).join(' – ');
  if (v === null || v === undefined) return '—';
  return esc(typeof v === 'object' ? JSON.stringify(v) : v);
}

function checksHtml(checks, failure, waiting) {
  if (waiting) return `<h3>Checks <span class="checking"><i class="mini-spin" aria-hidden="true"></i>checking</span></h3><p class="faint">Running the engineering checks…</p>`;
  if (failure) return `<h3>Checks</h3><p class="faint">Checks unavailable: ${esc(failure.message)}</p>`;
  if (!checks.length) return '<h3>Checks</h3><p class="faint">No checks reported.</p>';
  const nFail = checks.filter((c) => c.status === 'fail').length, nWarn = checks.filter((c) => c.status === 'warn').length;
  const pill = `<span class="pill ${nFail ? 'fail' : nWarn ? 'warn' : 'pass'}">${nFail ? nFail + ' fail' : nWarn ? nWarn + ' warn' : 'all pass'}</span>`;
  const order = { fail: 0, warn: 1, pass: 2 };
  const sorted = checks.map((c, i) => ({ c, i })).sort((a, b) => ((order[a.c.status] ?? 3) - (order[b.c.status] ?? 3)) || a.i - b.i).map((x) => x.c);
  // details stay visible for warnings and failures; passing checks keep them in a tooltip
  return `<h3>Checks ${pill}</h3><ul class="checks">${sorted.map((c) => `
      <li class="${esc(c.status)}"${c.status === 'pass' && c.detail ? ` title="${esc(c.detail)}"` : ''}>
        <span class="dot" title="${esc(c.status)}"></span>
        <span class="cname">${esc(c.name)}</span>
        <span class="cval">${fmtAny(c.value)}${c.limit !== undefined && c.limit !== null && c.limit !== '' ? `<span class="lim"> / ${fmtAny(c.limit)}</span>` : ''}</span>
        ${c.detail && c.status !== 'pass' ? `<span class="cdetail">${esc(c.detail)}</span>` : ''}
      </li>`).join('')}</ul>`;
}
function renderChecks(checks, failure) {
  const sec = $('#checks-sec');
  if (sec) sec.innerHTML = checksHtml(checks || [], failure, false);
}

function renderReport(rep, id) {
  const el = $('#report');
  if (!rep) { el.innerHTML = ''; return; }
  const out = [];
  const size = rep.size || {};
  out.push(`<section class="rsec headline">
    <div class="dims"><span>${mm(size.w)}</span><i>×</i><span>${mm(size.d)}</span><i>×</i><span>${mm(size.h)}</span><em>mm</em></div>
    <div class="sub">${rep.total_kg != null ? `<b>${num(rep.total_kg, 1)}</b> kg total` : ''}</div>
  </section>`);

  out.push(`<section class="rsec" id="checks-sec">${checksHtml(rep.checks || [], null, !!rep.checks_url && !(rep.checks || []).length)}</section>`);

  // materials only on parts the model leaves out (e.g. bushings without glass) are not listed
  const present = new Set([...viewer.parts.values()].filter((p) => !p.absent).map((p) => p.material));
  const mass = Object.entries(rep.mass_kg || {})
    .filter(([m, kg]) => m !== 'none' && (kg > 0 || present.has(m) || !viewer.parts.size))
    .sort((a, b) => b[1] - a[1]);
  if (mass.length) {
    out.push(`<section class="rsec"><h3>Mass</h3><ul class="kv">${mass.map(([m, kg]) => `
      <li><span class="k"><i class="sw" style="background:${swatch(m)}"></i>${esc(pretty(m))}</span><span class="v">${kgText(kg)}</span></li>`).join('')}</ul></section>`);
  }

  // parts the model leaves out (no geometry, material "none") are not counted
  const absent = {};
  for (const p of viewer.parts.values()) if (p.absent) absent[p.kind] = (absent[p.kind] || 0) + 1;
  const counts = Object.entries(rep.counts || {}).map(([k, n]) => [k, n - (absent[k] || 0)]).filter(([, n]) => n > 0).sort((a, b) => {
    const ia = KIND_ORDER.indexOf(a[0]), ib = KIND_ORDER.indexOf(b[0]);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  if (counts.length) {
    const total = counts.reduce((s, [, n]) => s + n, 0);
    out.push(`<section class="rsec"><h3>Parts <span class="faint">${total}</span></h3><ul class="counts">${counts.map(([k, n]) => `
      <li><b>${num(n)}</b> ${esc(plural(pretty(k), n))}</li>`).join('')}</ul></section>`);
  }

  const tiers = rep.beams_per_tier || [];
  if (tiers.length) {
    const mx = Math.max(1, ...tiers);
    const rows = tiers.map((n, t) => ({ n, t })).reverse();
    out.push(`<section class="rsec"><h3>Beams per tier</h3><div class="tiers">${rows.map(({ n, t }) => `
      <div class="tier"><span class="tn">${t}</span><span class="bar"><i style="width:${(n / mx * 100).toFixed(1)}%"></i></span><span class="tc">${n}</span></div>`).join('')}
      </div><p class="faint small">top tier first</p></section>`);
  }

  const joints = Object.entries(rep.joints || {}).sort((a, b) => b[1] - a[1]);
  if (joints.length) {
    const jt = joints.reduce((s, [, n]) => s + n, 0);
    out.push(`<section class="rsec"><h3>Joints <span class="faint">${jt}</span></h3><ul class="kv">${joints.map(([k, n]) => `
      <li><span class="k">${esc(pretty(k))}</span><span class="v">${num(n)}</span></li>`).join('')}</ul></section>`);
  }

  const notes = rep.notes || [];
  out.push(`<section class="rsec"><h3>Notes</h3>${notes.length ? `<ul class="notes">${notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '<p class="faint">Nothing to note.</p>'}</section>`);

  const beams = (rep.parts || []).filter((p) => p.length !== undefined && p.length !== null);
  if (beams.length) {
    out.push(`<section class="rsec"><details class="cut"><summary><h3>Cut list <span class="faint">${beams.length}</span></h3></summary>
      <table><thead><tr><th>part</th><th>tier</th><th class="r">length</th><th>section</th></tr></thead><tbody>${
      beams.map((b) => `<tr data-part="${esc(b.id)}"><td>${esc(b.id)}</td><td>${esc(b.tier ?? '')}</td><td class="r">${mm(b.length)}</td><td>${esc(b.section ?? '')}</td></tr>`).join('')
    }</tbody></table></details></section>`);
  }

  const tm = rep.timings ? Object.entries(rep.timings).map(([k, v]) => `${esc(k)} ${num(v * 1000, 0)}`).join(' · ') : '';
  out.push(`<section class="rsec build"><h3>Build</h3><p>${rep.build_s != null ? `<b>${num(rep.build_s * 1000, 0)}</b> ms` : ''}${tm ? ` <span class="faint">(${tm})</span>` : ''}</p>
    <p class="faint small">id ${esc(id || '')}</p></section>`);

  const wasOpen = $('details.cut', el)?.open;
  el.innerHTML = out.join('');
  if (wasOpen) $('details.cut', el).open = true;
  $$('tr[data-part]', el).forEach((tr) => {
    tr.addEventListener('mouseenter', () => viewer.highlight(tr.dataset.part));
    tr.addEventListener('mouseleave', () => viewer.highlight(null));
  });
}

// engineering checks arrive separately (the server computes them in the background)
const checksState = { pending: null };
async function loadChecks(rep, id) {
  if (!rep || !rep.checks_url || (rep.checks && rep.checks.length)) { checksState.pending = null; return; }
  const token = {};
  checksState.pending = token;
  let checks = null, failure = null;
  try {
    const res = await api(rep.checks_url);
    checks = res.checks || [];
  } catch (e) {
    failure = e;
  }
  if (checksState.pending !== token) return;          // a newer build took over
  checksState.pending = null;
  if (state.buildId !== id) return;
  if (checks) state.report.checks = checks;
  renderChecks(checks, failure);
}

// ---------------------------------------------------------------- dock: explode + assembly
function updateDock() {
  const N = viewer.steps.length;
  const sl = $('#step');
  sl.max = String(Math.max(1, N));
  sl.value = String(viewer.stepK || N);
  sl.disabled = N <= 1;
  setStepCaption();
  fillRange(sl);
  fillRange($('#explode'));
}
function setStepCaption() {
  const N = viewer.steps.length;
  const k = viewer.stepK || N;
  const s = viewer.steps[k - 1];
  const complete = k >= N;
  $('#step-num').textContent = N ? `${k} / ${N}` : '–';
  $('#step-text').textContent = s ? s.text : '';
  $('#assembly').classList.toggle('partial', !complete);
}
function fillRange(sl) {
  const lo = Number(sl.min || 0), hi = Number(sl.max || 1), v = Number(sl.value);
  sl.style.setProperty('--fill', (hi > lo ? (v - lo) / (hi - lo) * 100 : 100).toFixed(2) + '%');
}
let playTimer = null;
function setPlaying(on) {
  state.playing = on;
  clearTimeout(playTimer);
  $('#play').innerHTML = on ? ICON.pause : ICON.play;
  $('#play').setAttribute('aria-pressed', String(on));
  $('#play').title = on ? 'Pause' : 'Play the assembly';
  if (!on) return;
  const N = viewer.steps.length;
  let k = viewer.stepK >= N ? 1 : viewer.stepK + 1;
  const tick = () => {
    if (!state.playing) return;
    viewer.setStep(k, true);
    $('#step').value = String(k);
    fillRange($('#step'));
    setStepCaption();
    if (k >= viewer.steps.length) { setPlaying(false); return; }
    k += 1;
    playTimer = setTimeout(tick, 1150);
  };
  tick();
}
let lastExplode = 0.55;
function toggleExplode() {
  const sl = $('#explode');
  const target = viewer.explode > 0.01 ? 0 : lastExplode;
  if (target > 0) lastExplode = target;
  viewer.setExplode(target, true);
  sl.value = String(target);
  fillRange(sl);
  $('#explode-val').textContent = explodeLabel(target);
}
const explodeLabel = (v) => v > 0 ? `${Math.round(v * 75)} mm / step` : 'assembled';
function toggleGlass(force) {
  if (!viewer.hasGlass) return;
  const on = force === undefined ? !viewer.glassVisible : force;
  viewer.setGlassVisible(on);
  $('#btn-glass').setAttribute('aria-pressed', String(on));
}

// ---------------------------------------------------------------- toast / lightbox / tooltip
let toastTimer = null;
function showToast(msg, kind = 'info', trace) {
  const t = $('#toast');
  t.className = 'toast show ' + kind;
  t.innerHTML = `<span class="msg">${esc(msg)}</span>${trace ? `<details><summary>details</summary><pre>${esc(trace)}</pre></details>` : ''}<button type="button" class="x" aria-label="Dismiss">×</button>`;
  $('.x', t).addEventListener('click', () => hideToast());
  clearTimeout(toastTimer);
  if (kind !== 'error') toastTimer = setTimeout(() => hideToast(), 4200);
}
function hideToast(onlyKind) {
  const t = $('#toast');
  if (onlyKind && !t.classList.contains(onlyKind)) return;
  t.classList.remove('show');
}

function openLightbox(title, html, mode = '') {
  const lb = $('#lightbox');
  lightboxMode = mode;
  $('#lb-title').textContent = title;
  $('#lb-body').innerHTML = html;
  lb.hidden = false;
  requestAnimationFrame(() => lb.classList.add('open'));
  $('#lb-close').focus({ preventScroll: true });
  return $('#lb-body');
}
function closeLightbox() {
  const lb = $('#lightbox');
  lb.classList.remove('open');
  lb.hidden = true;
}

function showTip(id, ptr) {
  const tip = $('#tip');
  if (!id || !ptr || !state.report) { tip.classList.remove('show'); return; }
  const part = (state.report.parts || []).find((p) => p.id === id) || { id };
  const st = viewer.parts.get(id);
  const step = st && viewer.steps[st.step];
  const bits = [];
  if (part.length != null) bits.push(`${mm(part.length)} mm`);
  if (part.section) bits.push(esc(part.section));
  if (part.kg != null) bits.push(kgText(part.kg));
  tip.innerHTML = `<b>${esc(id)}</b> <span>${esc(pretty(part.kind || ''))}${part.tier != null ? ' · tier ' + esc(part.tier) : ''} · ${esc(pretty(part.material || ''))}</span>
    ${bits.length ? `<span>${bits.join(' · ')}</span>` : ''}
    ${step ? `<span class="faint">step ${esc(step.step)} — ${esc(step.text)}</span>` : ''}`;
  const stage = $('#stage').getBoundingClientRect();
  let x = ptr.cx - stage.left + 14, y = ptr.cy - stage.top + 16;
  tip.classList.add('show');
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  if (x + tw > stage.width - 8) x = ptr.cx - stage.left - tw - 12;
  if (y + th > stage.height - 70) y = ptr.cy - stage.top - th - 12;
  tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
}

// ---------------------------------------------------------------- render photo / export
const VIEWS = ['hero', 'top', 'low', 'detail', 'side'];
const photo = { busy: false, view: 'hero', url: null, error: null, trace: null, t0: 0, key: '' };
let lightboxMode = '';
let photoTimer = null;
const paramsKey = () => JSON.stringify(state.params);

async function renderPhoto(view = 'hero') {
  if (photo.busy) { showPhotoBox(); return; }
  Object.assign(photo, { busy: true, view, url: null, error: null, trace: null, t0: performance.now(), key: paramsKey() });
  $('#btn-render').classList.add('working');
  showPhotoBox();
  clearInterval(photoTimer);
  photoTimer = setInterval(() => {
    const el = $('#lb-elapsed');
    if (el) el.textContent = `${Math.round((performance.now() - photo.t0) / 1000)} s`;
  }, 500);
  try {
    const res = await api('/api/render', { params: { ...state.params }, view });
    photo.url = res.png;
  } catch (e) {
    photo.error = e.message;
    photo.trace = e.trace;
  } finally {
    photo.busy = false;
    clearInterval(photoTimer);
    $('#btn-render').classList.remove('working');
    if (lightboxMode === 'photo' && !$('#lightbox').hidden) showPhotoBox();
    else if (photo.url) showToast('Photo ready — press Render photo to see it.', 'info');
    else showToast('Render failed: ' + photo.error, 'error', photo.trace);
  }
}
function showPhotoBox() {
  const chips = `<div class="views" role="group" aria-label="Camera">${VIEWS.map((v) =>
    `<button type="button" class="chip" data-view="${v}" aria-pressed="${v === photo.view}" ${photo.busy ? 'disabled' : ''}>${v}</button>`).join('')}</div>`;
  let content;
  if (photo.busy) {
    content = `<div class="lb-wait"><div class="spinner" aria-hidden="true"></div>
      <p>Rendering on your GPU <span class="faint">· a few seconds ·</span> <span id="lb-elapsed" class="faint">${Math.round((performance.now() - photo.t0) / 1000)} s</span></p></div>
      <div class="lb-foot">${chips}</div>`;
  } else if (photo.url) {
    const stale = photo.key !== paramsKey();
    content = `<figure class="photo"><img src="${esc(photo.url)}" alt="Rendered ${esc(photo.view)} view"></figure>
      <div class="lb-foot">${chips}<span class="links">${stale ? '<span class="faint">design changed since — pick a view to re-render</span>' : ''}
      <a href="${esc(photo.url)}" target="_blank" rel="noopener">Open full size</a><a href="${esc(photo.url)}" download>Download</a></span></div>`;
  } else {
    content = `<div class="lb-msg"><p>The photo couldn't be rendered.</p><p class="faint">${esc(photo.error || 'unknown error')}</p>
      ${photo.trace ? `<details><summary>details</summary><pre>${esc(photo.trace)}</pre></details>` : ''}</div>
      <div class="lb-foot">${chips}</div>`;
  }
  const body = openLightbox(`Photo · ${photo.view}`, content, 'photo');
  $$('.views .chip', body).forEach((b) => b.addEventListener('click', () => renderPhoto(b.dataset.view)));
}
function onRenderButton() {
  if (photo.busy || (photo.url && photo.key === paramsKey())) showPhotoBox();
  else renderPhoto(photo.url || photo.error ? photo.view : 'hero');
}

const exporting = { busy: false, t0: 0, html: '' };
async function exportFiles() {
  if (exporting.busy) { showExportBox(); return; }
  exporting.busy = true;
  exporting.ok = false;
  exporting.key = paramsKey();
  exporting.t0 = performance.now();
  $('#btn-export').classList.add('working');
  showExportBox();
  const timer = setInterval(() => {
    const el = $('#lb-elapsed');
    if (el && lightboxMode === 'export') el.textContent = `${Math.round((performance.now() - exporting.t0) / 1000)} s`;
  }, 500);
  try {
    const res = await api('/api/export', { params: { ...state.params } });
    const files = res.files || [];
    const url = (f) => (typeof f === 'string' ? f : f.url);
    const name = (f) => (typeof f === 'string' ? decodeURIComponent(f.split('/').pop()) : f.name);
    const ext = (f) => (name(f).match(/\.([a-z0-9]+)$/i) || [, ''])[1].toLowerCase();
    exporting.ok = true;
    exporting.html = `<div class="lb-msg">
      <p>The fabrication package is ready: DXF cut sheets for the strips and the glass, drawings, a cut list,
      <code>FABRICATION.md</code>, and the model as GLB and STL. <span class="faint">(STEP and Rhino .3dm come from the
      Python package on your own machine.)</span></p>
      <ul class="files">${files.map((f) => `<li><span class="ext">${esc(ext(f))}</span><a href="${esc(url(f))}" download="${esc(name(f))}">${esc(name(f))}</a></li>`).join('')}</ul></div>`;
  } catch (e) {
    exporting.html = `<div class="lb-msg">
      <p>The export didn't finish.</p>
      <p class="faint">The engine said: ${esc(e.message)}</p>
      ${e.trace ? `<details><summary>details</summary><pre>${esc(e.trace)}</pre></details>` : ''}</div>`;
  } finally {
    clearInterval(timer);
    exporting.busy = false;
    $('#btn-export').classList.remove('working');
    if (lightboxMode === 'export' && !$('#lightbox').hidden) showExportBox();
    else if (exporting.ok) showToast('Export finished — press Export to see the files.', 'info');
    else showToast("The export didn't finish — press Export for details.", 'error');
  }
}
function showExportBox() {
  const html = exporting.busy
    ? `<div class="lb-wait"><div class="spinner" aria-hidden="true"></div>
        <p>Writing fabrication files <span class="faint">· cut sheets, drawings, cut list · about a minute in the browser ·</span> <span id="lb-elapsed" class="faint">${Math.round((performance.now() - exporting.t0) / 1000)} s</span></p></div>`
    : exporting.html;
  openLightbox('Export', html, 'export');
}

// ---------------------------------------------------------------- settled()
const settleWaiters = [];
function isIdle() {
  return !jobs.running && !jobs.pending && !jobs.debounce && !checksState.pending &&
    !viewer.isAnimating() && !viewer.needsRender && !state.playing;
}
function settled() {
  return new Promise((resolve) => {
    settleWaiters.push(resolve);
    viewer.requestRender();
  });
}
function checkSettled(rendered) {
  if (!settleWaiters.length) return;
  if (!isIdle()) return;
  if (!rendered) { viewer.requestRender(); return; }
  settleWaiters.splice(0).forEach((r) => r());
}

// ---------------------------------------------------------------- wiring
function wire() {
  $('#preset').addEventListener('change', (e) => { if (e.target.value) applyPreset(e.target.value); });
  $('#btn-random').addEventListener('click', () => randomize());
  $('#btn-mutate').addEventListener('click', () => mutate());
  $('#btn-render').addEventListener('click', onRenderButton);
  $('#btn-export').addEventListener('click', () => {
    if (!exporting.busy && exporting.ok && exporting.key === paramsKey()) showExportBox();
    else exportFiles();
  });
  $('#seed').addEventListener('change', (e) => randomize(e.target.value));
  $('#seed').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });
  $('#btn-glass').addEventListener('click', () => toggleGlass());
  $('#btn-undo').addEventListener('click', () => stepHistory(-1));
  $('#btn-redo').addEventListener('click', () => stepHistory(1));
  const filter = $('#filter');
  filter.addEventListener('input', () => applyFilter(filter.value));
  filter.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); filter.value = ''; applyFilter(''); filter.blur(); }
  });
  $('#btn-frame').addEventListener('click', () => viewer.frame(true));
  $('#lb-close').addEventListener('click', closeLightbox);
  $('#lightbox').addEventListener('click', (e) => { if (e.target.id === 'lightbox') closeLightbox(); });

  const ex = $('#explode');
  ex.addEventListener('input', () => {
    const v = Number(ex.value);
    if (v > 0.01) lastExplode = v;
    viewer.setExplode(v, false);
    fillRange(ex);
    $('#explode-val').textContent = explodeLabel(v);
  });
  const st = $('#step');
  st.addEventListener('input', () => {
    setPlaying(false);
    viewer.setStep(Number(st.value), true);
    fillRange(st);
    setStepCaption();
  });
  $('#play').addEventListener('click', () => setPlaying(!state.playing));

  document.addEventListener('keydown', (e) => {
    const t = e.target;
    const typing = t && (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' ||
      (t.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button'].includes(t.type)));
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !typing && $('#lightbox').hidden) {
      const k = e.key.toLowerCase();
      if (k === 'z') { e.preventDefault(); stepHistory(e.shiftKey ? 1 : -1); }
      else if (k === 'y') { e.preventDefault(); stepHistory(1); }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape') { closeLightbox(); hideToast(); return; }
    if (typing || !$('#lightbox').hidden) return;
    const k = e.key.toLowerCase();
    if (k === 'r') { e.preventDefault(); randomize(); }
    else if (k === 'm') { e.preventDefault(); mutate(); }
    else if (k === 'e') { e.preventDefault(); toggleExplode(); }
    else if (k === 'g') { e.preventDefault(); toggleGlass(); }
    else if (k === 'f') { e.preventDefault(); viewer.frame(true); }
    else if (e.key === '/') { e.preventDefault(); $('#filter').focus(); }
  });
  window.addEventListener('hashchange', () => {
    const h = readHash();
    const next = { ...state.defaults, ...h };
    if (JSON.stringify(next) === JSON.stringify(state.params)) return;
    state.params = next;
    refreshControls();
    scheduleBuild();
  });
}

async function main() {
  const stage = $('#stage');
  try {
    const q = new URLSearchParams(location.search);     // dev knobs: ?tm=aces|agx|neutral&exposure=1.1
    viewer = new Viewer($('#gl-host'), {
      background: getComputedStyle(document.documentElement).getPropertyValue('--stage').trim() || '#EAE4D8',
      toneMapping: q.get('tm') || undefined,
      exposure: q.has('exposure') ? Number(q.get('exposure')) : undefined,
    });
  } catch (e) {
    document.documentElement.dataset.error = 'WebGL unavailable: ' + e.message;
    rejectReady(e);
    showToast('This browser could not start WebGL: ' + e.message, 'error');
    return;
  }
  viewer.onRendered = () => checkSettled(true);
  viewer.onIdleFrame = () => checkSettled(false);
  viewer.onHover = (id, ptr) => {
    showTip(id, ptr);
    stage.classList.toggle('hovering', !!id);
  };
  viewer.onError = (m) => showToast(m, 'error');
  wire();

  const bootText = $('#boot-text');
  const bootBar = $('#boot-bar');
  const unsub = engine.onProgress((m) => {
    if (bootText) bootText.textContent = m.text;
    if (bootBar) bootBar.style.width = `${Math.round(100 * (m.frac || 0))}%`;
  });
  let schema;
  try {
    await engine.start();
    unsub();
    $('#boot')?.classList.add('done');
    setTimeout(() => $('#boot')?.remove(), 600);
    schema = await api('/api/schema');
  } catch (e) {
    document.documentElement.dataset.error = e.message;
    rejectReady(e);
    showToast('Could not load the parameter schema: ' + e.message, 'error', e.trace);
    return;
  }
  state.schema = (schema.params || []).filter((p) => p && p.name);
  state.byName = new Map(state.schema.map((p) => [p.name, p]));
  const groups = new Map();
  for (const p of state.schema) {
    const g = p.group || 'Other';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(p);
  }
  state.groups = [...groups].map(([name, params]) => ({ name, params }));
  state.presets = schema.presets || {};
  state.defaults = Object.fromEntries(state.schema.map((p) => [p.name, p.default]));
  state.params = { ...state.defaults, ...readHash() };

  const sel = $('#preset');
  sel.innerHTML = `<option value="">custom</option>` +
    Object.keys(state.presets).map((n) => `<option value="${esc(n)}">${esc(cap(n))}</option>`).join('');
  buildPanel();
  refreshControls();
  submit({ kind: 'build', url: '/api/build', body: () => ({ params: { ...state.params } }), frame: true });
}

window.kigumi = {
  ready,
  engine,
  settled,
  get params() { return { ...state.params }; },
  get report() { return state.report; },
  get appliedParams() { return state.appliedJSON ? JSON.parse(state.appliedJSON) : null; },
  setParams(obj) {
    for (const [k, v] of Object.entries(obj || {})) {
      const p = state.byName.get(k);
      if (p) { state.params[k] = coerce(p, v); controls.get(k)?.set(state.params[k]); }
    }
    updatePresetSelect();
    scheduleBuild();
  },
  randomize, mutate,
  setExplode(v) { const ex = $('#explode'); ex.value = String(v); ex.dispatchEvent(new Event('input')); },
  setStep(k) { const st = $('#step'); st.value = String(k); st.dispatchEvent(new Event('input')); },
  get viewer() { return viewer; },
};

main();
