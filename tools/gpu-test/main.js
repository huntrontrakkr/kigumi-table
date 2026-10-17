// Test page driver: runs the plan in a worker (default, ?mode=worker) or on
// the main thread (?mode=main), shows a table and leaves the JSON summary in
// window.__results for run.mjs.

const params = Object.fromEntries(new URLSearchParams(location.search));
const mode = params.mode || 'worker';
const rows = document.getElementById('rows');
const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const t0 = performance.now();

const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '' : typeof v === 'number' ? (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? v.toExponential(d) : v.toFixed(v % 1 ? 1 : 0)) : String(v));

function log(text) {
  logEl.textContent += text + '\n';
  console.log('[test] ' + text);     // run.mjs relays these
}

function show(r) {
  const tr = document.createElement('tr');
  const cell = (v, cls) => { const td = document.createElement('td'); td.textContent = v; if (cls) td.className = cls; tr.appendChild(td); };
  cell(r.name || r.id);
  if (r.error) {
    cell(''); cell(''); cell(''); cell(''); cell(''); cell(''); cell(''); cell(''); cell(''); cell(''); cell('');
    cell('error', 'fail');
    log(`${r.id}: ${r.error}`);
  } else {
    cell(r.n); cell(r.nnz); cell(r.k); cell(r.precond); cell(r.iters); cell(r.refinements);
    cell(fmt(r.relresMax)); cell(fmt(r.errMax)); cell(fmt(r.floorMax)); cell(fmt(r.setupMs)); cell(fmt(r.ms));
    cell(r.pass === true ? 'pass' : r.pass === false ? 'FAIL' : 'info', r.pass === true ? 'pass' : r.pass === false ? 'fail' : 'info');
    log(`${r.id}: iters ${r.iters}, refinements ${r.refinements}, relres ${fmt(r.relresMax, 2)}, err ${fmt(r.errMax, 2)}, ` +
        `per refinement ${r.perRefinement.map((h) => `${h.iters}it->${fmt(h.relres, 1)}`).join(' ')}`);
  }
  rows.appendChild(tr);
}

function finish(out) {
  out.mode = mode;
  out.wallMs = performance.now() - t0;
  statusEl.textContent = out.error ? `error: ${out.error}` :
    `${out.ok ? 'all acceptance checks passed' : 'SOME CHECKS FAILED'} — ${out.results.length} cases, ` +
    `${(out.wallMs / 1000).toFixed(1)} s, ${mode} mode, adapter: ${out.gpu ? out.gpu.description : 'none'}`;
  window.__results = out;
}

if (mode === 'main') {
  const { runAll } = await import('./cases.js');
  runAll(params, show, log).then(finish, (e) => finish({ ok: false, error: String(e.stack || e), results: [] }));
} else {
  const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  w.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'case') show(m.result);
    else if (m.type === 'log') log(m.text);
    else if (m.type === 'done') { finish(m.out); w.terminate(); }
  };
  w.onerror = (e) => finish({ ok: false, error: 'worker error: ' + (e.message || e), results: [] });
  w.postMessage({ params });
}
