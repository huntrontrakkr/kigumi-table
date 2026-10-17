// Test plan for the WebGPU sparse SPD solver: the converted model systems
// (data/, made by convert.py), synthetic systems with known solutions, and the
// scoring.  Runs in a dedicated worker (as in the app) or on the main thread.
//
// URL parameters (forwarded by index.html):
//   cases=a,b,...   subset of case ids (default: all of the default plan)
//   compare=1       also run plate_four / grillage_four with the jacobi and fsai (level 1) preconditioners
//   bigk=K|all      right-hand sides of plate_big to solve (default 8 of 90)
//   eta, precond, checkEvery, maxIter, tol    forwarded to solveSPD

import { laplace2d, beam1d } from './synthetic.js';

const BASE = new URL('./', import.meta.url);

// SuperLU in Pyodide (WebAssembly), the app's CPU path, for comparison: the version the
// site's worker loads (site/engine/worker.js), from the CDN
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
let pyodidePromise = null;
function pyodide() {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      const { loadPyodide } = await import(PYODIDE + 'pyodide.mjs');
      const py = await loadPyodide({ indexURL: PYODIDE });
      await py.loadPackage(['numpy', 'scipy'], { messageCallback: () => {} });
      return py;
    })();
  }
  return pyodidePromise;
}

/** kigumi.linalg.factor(K).solve(B) under Pyodide, timed (best of `reps`). */
async function superluCase(c, reps = 3) {
  const py = await pyodide();
  const put = (name, a) => py.FS.writeFile('/tmp/' + name, new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  put('indptr', Int32Array.from(c.indptr, Number));
  put('indices', Int32Array.from(c.indices, Number));
  put('data', c.data);
  put('rhs', c.rhs);
  put('xref', c.xref);
  const t = performance.now();
  const res = JSON.parse(py.runPython(`
import json, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
n, k = ${c.n}, ${c.k}
K = csr_matrix((np.fromfile('/tmp/data', dtype=np.float64), np.fromfile('/tmp/indices', dtype=np.int32),
                np.fromfile('/tmp/indptr', dtype=np.int32)), shape=(n, n))
B = np.fromfile('/tmp/rhs', dtype=np.float64).reshape(k, n).T.copy()
Xref = np.fromfile('/tmp/xref', dtype=np.float64).reshape(k, n).T
best = None
for rep in range(${reps}):
    t0 = time.perf_counter()
    lu = splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    t1 = time.perf_counter()
    X = lu.solve(B)
    t2 = time.perf_counter()
    if best is None or t2 - t0 < sum(best):
        best = (t1 - t0, t2 - t1)
rel = np.linalg.norm(B - K @ X, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
err = np.linalg.norm(X - Xref, axis=0) / np.maximum(np.linalg.norm(Xref, axis=0), 1e-300)
json.dumps(dict(factor_ms=best[0] * 1e3, solve_ms=best[1] * 1e3, relres=float(rel.max()), err=float(err.max())))
`));
  return {
    name: `SuperLU (Pyodide) ${c.name}`, n: c.n, nnz: Number(c.indptr[c.n]), k: c.k, precond: 'superlu',
    iters: 0, refinements: 0, relresMax: res.relres, errMax: res.err, floorMax: null,
    ms: res.factor_ms + res.solve_ms, setupMs: res.factor_ms, gpuMs: 0, cpuMs: res.factor_ms + res.solve_ms,
    note: `factor ${res.factor_ms.toFixed(1)} ms + solve ${res.solve_ms.toFixed(1)} ms (best of ${reps}); ` +
      `wall ${(performance.now() - t).toFixed(0)} ms incl. data transfer`,
    pass: null,
  };
}

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
  if (item.bench) errMax = null;
  const hist = Array.isArray(r.history?.[0]) ? r.history.flat() : (r.history || []);
  return {
    id: item.id, name: c.name, n, nnz: c.indptr[n], k, precond: r.precond,
    iters: r.iters, refinements: r.refinements, relresMax, errMax, floorMax,
    converged: r.converged, ms: r.ms, setupMs: r.setupMs, gpuMs: r.gpuMs, cpuMs: r.cpuMs,
    perRefinement: hist.map((h) => ({ relres: h.relres, iters: h.iters, innerRel: h.innerRel, stagnated: h.stagnated, cols: h.cols })),
    stopped: r.stopped && r.stopped.length <= 16 ? r.stopped : undefined, tolMet: r.tolMet,
    accept: acc || null, pass, note: item.note || c.note,
    msPerIter: r.iters ? r.gpuMs / r.iters : null,
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
  // plate_big: 8 of its 90 right-hand sides by default (SwiftShader is slow); bigk=all for every one
  const bigk = params.bigk === 'all' ? null : +(params.bigk || 8);
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
      make: async () => ({ ...(await exported('plate_four')), name: 'plate_four (warm x0)' }) });
  }
  if (meta.grillage_four) items.push({ id: 'grillage_four', group: 'model', accept: ACCEPT, make: () => exported('grillage_four') });
  if (meta.plate_model) {
    items.push({ id: 'plate_model', group: 'model', accept: ACCEPT,
      note: "the app's own call: the 'field-four' plate with all 123 load cases", make: () => exported('plate_model') });
    items.push({ id: 'plate_model_warm', group: 'model', accept: ACCEPT,
      note: 'the same call again: prepared matrix, pipelines and buffers cached',
      make: async () => ({ ...(await exported('plate_model')), name: 'plate_model (warm)' }) });
  }
  // superlu: SuperLU in Pyodide on the same systems (needs the network for Pyodide)
  for (const name of ['plate_model', 'plate_four', 'grillage_four', 'plate_big']) {
    if (!meta[name]) continue;
    items.push({ id: `superlu_${name}`, group: 'superlu', accept: null, custom: superluCase,
      make: async () => { const c = await exported(name); return name === 'plate_big' && bigk && bigk < c.k ? columns(c, 0, bigk) : c; } });
  }
  if (meta.plate_big) {
    items.push({ id: 'plate_big', group: 'model', accept: ACCEPT,
      make: async () => { const c = await exported('plate_big'); return bigk && bigk < c.k ? columns(c, 0, bigk) : c; } });
  }
  // robustness: chunked batches, zero and exact columns, Jacobi, 64-bit indices, bad input
  items.push({ id: 'chunks', group: 'robust', accept: ACCEPT, opts: { maxCols: 16 },
    note: 'k = 40 split into 3 GPU batches (maxCols 16)', make: async () => ({ ...laplace2d(24, 24, 40), name: 'laplace2d k=40 in 3 batches' }) });
  items.push({ id: 'zero_rhs', group: 'robust', accept: ACCEPT, note: 'a zero right-hand side column (x = 0 exactly)',
    make: async () => { const c = laplace2d(24, 24, 3); c.rhs.fill(0, c.n, 2 * c.n); c.xref.fill(0, c.n, 2 * c.n); return { ...c, name: 'laplace2d, zero column' }; } });
  items.push({ id: 'x0_exact', group: 'robust', accept: ACCEPT, note: 'x0 = the exact solution: no GPU work',
    x0: (c) => Float64Array.from(c.xref), expect: (r) => r.iters === 0 && r.refinements === 0,
    make: async () => ({ ...laplace2d(24, 24, 2), name: 'laplace2d, exact x0' }) });
  items.push({ id: 'jacobi', group: 'robust', accept: ACCEPT, precond: 'jacobi', make: async () => laplace2d(30, 30, 4) });
  items.push({ id: 'int64_indices', group: 'robust', accept: ACCEPT, note: 'BigInt64Array indptr/indices (int64 from Python)',
    make: async () => { const c = laplace2d(24, 24, 2);
      return { ...c, name: 'laplace2d, int64 indices', indptr: BigInt64Array.from(c.indptr, BigInt), indices: BigInt64Array.from(c.indices, BigInt) }; } });
  items.push({ id: 'device_lost', group: 'robust', accept: ACCEPT, note: 'device destroyed before the call: solveSPD takes a fresh one',
    before: async (gpu) => { gpu.device.destroy(); await gpu.device.lost; await new Promise((r) => setTimeout(r, 0)); },
    make: async () => ({ ...laplace2d(24, 24, 2), name: 'laplace2d after device loss' }) });
  if (meta.plate_four) {
    items.push({ id: 'too_big', group: 'robust', accept: null, opts: { memBudget: 2 ** 20 },
      note: 'memory budget (1 MiB) below the matrix: converged = false with a reason, at once',
      expect: (r) => !r.converged && /too large/.test(r.reason || '') && r.iters === 0 && r.ms < 200,
      make: async () => ({ ...(await exported('plate_four')), name: 'plate_four, 1 MiB budget' }) });
    items.push({ id: 'precond_stepdown', group: 'robust', accept: ACCEPT, opts: { memBudget: 3.5 * 2 ** 20 },
      note: "budget fits FSAI level 1 but not level 2: steps down to 'fsai'",
      expect: (r) => r.precond === 'fsai' && r.converged,
      make: async () => columns(await exported('plate_four'), 0, 1, 'plate_four[0:1], 3.5 MiB budget') });
  }
  items.push({ id: 'short_submits', group: 'robust', accept: ACCEPT, opts: { submitMs: 1 },
    note: 'submitMs = 1: about one iteration per queue submission',
    expect: (r) => r.converged && r.submits >= r.iters / 4,
    make: async () => ({ ...laplace2d(24, 24, 2), name: 'laplace2d, 1 ms submissions' }) });
  items.push({ id: 'not_spd', group: 'robust', expectError: /not SPD/, note: 'negative diagonal entry: must throw',
    make: async () => { const c = laplace2d(10, 10, 1); const d = Float64Array.from(c.data);
      for (let p = c.indptr[5]; p < c.indptr[6]; p++) if (c.indices[p] === 5) d[p] = -1;
      return { ...c, data: d, name: 'negative diagonal' }; } });
  items.push({ id: 'bad_csr', group: 'robust', expectError: /indptr/, note: 'indptr of the wrong length: must throw',
    make: async () => { const c = laplace2d(10, 10, 1); return { ...c, indptr: c.indptr.subarray(0, c.n), name: 'malformed CSR' }; } });
  // bench: fixed iteration counts (eta ~ 0, one refinement) for per-iteration GPU cost; open
  // index.html?cases=bench in a browser with a real GPU to measure it there (informational)
  if (meta.plate_four) {
    for (const k of [1, 8, 64, 256]) {
      items.push({ id: `bench_k${k}`, group: 'bench', accept: null, bench: true,
        opts: { eta: 1e-30, maxRefine: 1, maxIter: +(params.benchIters || 200), checkEvery: 32 },
        note: 'per-iteration cost: plate_four, fixed iterations',
        make: async () => {
          const c = await exported('plate_four');
          const rhs = new Float64Array(c.n * k);
          for (let j = 0; j < k; j++) rhs.set(c.rhs.subarray((j % c.k) * c.n, (j % c.k + 1) * c.n), j * c.n);
          return { ...c, k, rhs, xref: new Float64Array(c.n * k), name: `bench plate_four k=${k}` };
        } });
    }
  }
  if (params.compare) {
    for (const name of ['plate_four', 'grillage_four']) {
      if (!meta[name]) continue;
      for (const precond of ['jacobi', 'fsai']) {
        items.push({ id: `${name}_${precond}`, group: 'compare', accept: ACCEPT, precond, make: () => exported(name) });
      }
    }
  }
  const only = params.cases ? new Set(params.cases.split(',')) : null;
  return only ? items.filter((it) => only.has(it.id) || only.has(it.group))
    : items.filter((it) => !it.bench && it.group !== 'superlu');
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
    let c = null;
    try {
      c = await item.make();
      if (item.before) await item.before(gpu);
      onLog(`${item.id}: n=${c.n} nnz=${Number(c.indptr[c.indptr.length - 1])} k=${c.k} ...`);
      if (item.custom) {
        res = { id: item.id, ...(await item.custom(c)) };
        results.push(res);
        onCase(res);
        continue;
      }
      const r = await sparse.solveSPD(gpu, {
        n: c.n, k: c.k, indptr: c.indptr, indices: c.indices, data: c.data, rhs: c.rhs,
        x0: item.x0 ? item.x0(c) : undefined, ...opts, ...(item.opts || {}), ...(item.precond ? { precond: item.precond } : {}),
      });
      if (item.expectError) {
        res = { id: item.id, name: c.name, error: 'expected an error, got a result', pass: false };
      } else {
        res = score(item, { ...c, indptr: Int32Array.from(c.indptr, Number) }, r);
        if (item.expect) {
          if (!item.expect(r)) { res.pass = false; res.note = 'unexpected: ' + (item.note || ''); }
          else if (res.pass === null) res.pass = true;
        }
        if (r.reason) res.reason = r.reason;
        res.submits = r.submits;
        res.gpuBytes = r.gpuBytes;
      }
    } catch (e) {
      const msg = String((e && e.message) || e);
      if (item.expectError && item.expectError.test(msg)) {
        res = { id: item.id, name: (c && c.name) || item.id, expectedError: msg, pass: true, note: item.note };
      } else {
        res = { id: item.id, name: item.id, error: String((e && e.stack) || e), pass: item.accept || item.expectError ? false : null };
      }
    }
    results.push(res);
    onCase(res);
  }
  const ok = results.every((r) => r.pass !== false);
  return { ok, gpu: gpu.info, params, results };
}
