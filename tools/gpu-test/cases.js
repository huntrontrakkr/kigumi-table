// Test plan for the WebGPU sparse SPD solver: the converted model systems
// (data/, made by convert.py), synthetic systems with known solutions, and the
// scoring.  Runs in a dedicated worker (as in the app) or on the main thread.
//
// URL parameters (forwarded by index.html):
//   cases=a,b,...   subset of case ids (default: all of the default plan)
//   compare=1       also run plate_four / grillage_four with the jacobi and fsai2 preconditioners
//   bigk=K          right-hand sides of plate_big to solve (default: all 90)
//   eta, precond, checkEvery, maxIter, tol    forwarded to solveSPD

import { laplace2d, beam1d } from './synthetic.js';

const BASE = new URL('./', import.meta.url);

async function bin(name, Type) {
  const r = await fetch(new URL('data/' + name, BASE));
  if (!r.ok) throw new Error(`fetch data/${name}: ${r.status} (run convert.py?)`);
  return new Type(await r.arrayBuffer());
}

async function loadExported(meta) {
  const f = meta.files;
  const [indptr, indices, data, rhs, xref] = await Promise.all([
    bin(f.indptr, Int32Array), bin(f.indices, Int32Array), bin(f.data, Float64Array),
    bin(f.rhs, Float64Array), bin(f.xref, Float64Array),
  ]);
  return { name: meta.name, n: meta.n, k: meta.k, indptr, indices, data, rhs, xref, note: meta.note };
}

/** Columns [j0, j0+k) of a case (column-major blocks are contiguous). */
function columns(c, j0, k, name) {
  return { ...c, name: name || `${c.name}[${j0}:${j0 + k}]`, k,
    rhs: c.rhs.subarray(j0 * c.n, (j0 + k) * c.n), xref: c.xref.subarray(j0 * c.n, (j0 + k) * c.n) };
}

/** Relative error per column vs the reference, and the scored result. */
function score(item, c, r) {
  const { n, k } = c;
  let errMax = 0;
  for (let j = 0; j < k; j++) {
    let e = 0, s = 0;
    for (let i = 0; i < n; i++) {
      const d = r.x[j * n + i] - c.xref[j * n + i];
      e += d * d;
      s += c.xref[j * n + i] * c.xref[j * n + i];
    }
    const rel = s > 0 ? Math.sqrt(e / s) : Math.sqrt(e);
    if (!(rel <= errMax)) errMax = rel;
  }
  let relresMax = 0, floorMax = 0;
  for (let j = 0; j < k; j++) if (!(r.relres[j] <= relresMax)) relresMax = r.relres[j];
  if (r.floor) for (let j = 0; j < k; j++) floorMax = Math.max(floorMax, r.floor[j]);
  const acc = item.accept;
  const pass = acc ? relresMax <= acc.relres && errMax <= acc.err : null;
  const hist = Array.isArray(r.history?.[0]) ? r.history.flat() : (r.history || []);
  return {
    id: item.id, name: c.name, n, nnz: c.indptr[n], k, precond: r.precond,
    iters: r.iters, refinements: r.refinements, relresMax, errMax, floorMax,
    converged: r.converged, ms: r.ms, setupMs: r.setupMs, gpuMs: r.gpuMs, cpuMs: r.cpuMs,
    perRefinement: hist.map((h) => ({ relres: h.relres, iters: h.iters, innerRel: h.innerRel, stagnated: h.stagnated })),
    accept: acc || null, pass, note: item.note || c.note,
  };
}

const ACCEPT = { relres: 1e-8, err: 1e-6 };

function plan(params, manifest) {
  const meta = Object.fromEntries((manifest?.cases || []).map((m) => [m.name, m]));
  const cache = new Map();
  const exported = (name) => {
    if (!meta[name]) return null;
    if (!cache.has(name)) cache.set(name, loadExported(meta[name]));
    return cache.get(name);
  };
  const bigk = params.bigk ? +params.bigk : null;
  const items = [];
  // layouts: lane widths 1/2/4, padded and non-power-of-two column groups
  for (const k of [1, 2, 3, 5, 33]) {
    items.push({ id: `layout_k${k}`, group: 'layouts', accept: ACCEPT, make: async () => laplace2d(30, 30, k) });
  }
  items.push({ id: 'laplace2d', group: 'synthetic', accept: ACCEPT, make: async () => laplace2d(80, 80, 4) });
  items.push({ id: 'beam1d_100', group: 'synthetic', accept: ACCEPT, make: async () => beam1d(100) });
  items.push({ id: 'beam1d_200', group: 'synthetic', accept: ACCEPT, make: async () => beam1d(200) });
  items.push({ id: 'beam1d_400', group: 'synthetic', accept: null,
    note: 'stress case beyond float32 reach (cond(G As G^T) ~ 1e8): informational', make: async () => beam1d(400) });
  if (meta.plate_four) {
    items.push({ id: 'plate_four', group: 'model', accept: ACCEPT, make: () => exported('plate_four') });
    items.push({ id: 'plate_four_again', group: 'model', accept: ACCEPT,
      note: 'same matrix again, 3 of the right-hand sides: cached FSAI set-up, a different batch width',
      make: async () => columns(await exported('plate_four'), 2, 3, 'plate_four[2:5] (cached)') });
    items.push({ id: 'plate_four_x0', group: 'model', accept: ACCEPT,
      note: 'warm start: x0 = reference + 1e-4 relative perturbation',
      x0: (c) => { const x = Float64Array.from(c.xref); for (let i = 0; i < x.length; i++) x[i] *= 1 + 1e-4 * Math.sin(i); return x; },
      make: () => exported('plate_four') });
  }
  if (meta.grillage_four) items.push({ id: 'grillage_four', group: 'model', accept: ACCEPT, make: () => exported('grillage_four') });
  if (meta.plate_big) {
    items.push({ id: 'plate_big', group: 'model', accept: ACCEPT,
      make: async () => { const c = await exported('plate_big'); return bigk && bigk < c.k ? columns(c, 0, bigk) : c; } });
  }
  if (params.compare) {
    for (const name of ['plate_four', 'grillage_four']) {
      if (!meta[name]) continue;
      for (const precond of ['jacobi', 'fsai2']) {
        items.push({ id: `${name}_${precond}`, group: 'compare', accept: ACCEPT, precond, make: () => exported(name) });
      }
    }
  }
  const only = params.cases ? new Set(params.cases.split(',')) : null;
  return only ? items.filter((it) => only.has(it.id) || only.has(it.group)) : items;
}

export async function runAll(params, onCase = () => {}, onLog = () => {}) {
  const sparse = await import('../../site/engine/gpu/sparse.js');
  const gpu = await sparse.getGpu();
  if (!gpu) return { ok: false, error: 'WebGPU unavailable (getGpu() returned null)', results: [] };
  onLog(`GPU: ${JSON.stringify(gpu.info)}`);
  let manifest = null;
  try {
    const r = await fetch(new URL('data/manifest.json', BASE));
    if (r.ok) manifest = await r.json();
  } catch { manifest = null; }
  if (!manifest) onLog('data/manifest.json missing: only synthetic cases (run convert.py)');
  const opts = {};
  for (const key of ['eta', 'checkEvery', 'maxIter', 'tol', 'maxRefine']) if (params[key] != null) opts[key] = +params[key];
  if (params.precond) opts.precond = params.precond;
  const results = [];
  for (const item of plan(params, manifest)) {
    let res;
    try {
      const c = await item.make();
      onLog(`${item.id}: n=${c.n} nnz=${c.indptr[c.n]} k=${c.k} ...`);
      const r = await sparse.solveSPD(gpu, {
        n: c.n, k: c.k, indptr: c.indptr, indices: c.indices, data: c.data, rhs: c.rhs,
        x0: item.x0 ? item.x0(c) : undefined, ...opts, ...(item.precond ? { precond: item.precond } : {}),
      });
      res = score(item, c, r);
    } catch (e) {
      res = { id: item.id, name: item.id, error: String((e && e.stack) || e), pass: item.accept ? false : null };
    }
    results.push(res);
    onCase(res);
  }
  const ok = results.every((r) => r.pass !== false);
  return { ok, gpu: gpu.info, params, results };
}
