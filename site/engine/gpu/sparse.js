// WebGPU solver for sparse symmetric positive definite systems A X = B with
// many right-hand sides, float64-accurate although WGSL only has float32.
//
//   const gpu = await getGpu();                 // { device, info } or null
//   const r = await solveSPD(gpu, { n, indptr, indices, data, rhs, k, tol: 1e-9 });
//   // r = { x: Float64Array(n*k), iters, refinements, relres: Float64Array(k), ms, converged, ... }
//
// A is full symmetric CSR (both triangles), rhs and x are column-major
// (column j at [j*n, (j+1)*n)).  Works on the main thread and in workers.
//
// Method (mixed-precision iterative refinement):
//   * symmetric Jacobi scaling  As = S A S,  S = diag(1/sqrt(a_ii));
//   * an FSAI preconditioner G (lower triangular, G As G^T ~ I), set up once
//     per matrix on the CPU in float64 (cheap: one small dense Cholesky per row);
//   * outer loop on the CPU in float64:  r = b - A x;  d = inner(S r);  x += S d;
//     until ||b - A x|| / ||b|| <= tol for every column, or no more progress;
//   * inner solve on the GPU in float32: batched preconditioned CG on As over
//     all k columns at once (CSR SpMM), per-column alpha/beta and convergence
//     flags computed on the GPU, read back only every few dozen iterations,
//     run to a modest relative residual (eta, default 1e-4).
// Each refinement gains roughly eta (limited by cond(G As G^T) * 2^-24), so
// 1e-9 typically takes 3-4 refinements.  See tools/gpu-test/NOTES.md.

import { WG, NF, F, MAX_GRID, layoutFor, rowKernel, reduceKernel, axpyKernel, xpbyKernel } from './kernels.js';
import {
  asInt32, asFloat64, checkCSR, jacobiScale, fsai, transposeCSR, residualRowMajor,
  residualFloorRowMajor, colNormsRowMajor, colToRow, rowToCol, sameArray,
} from './csr.js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export const DEFAULTS = Object.freeze({
  eta: 1e-4,          // inner float32 PCG: relative residual target per refinement
  maxRefine: 15,      // cap on outer (float64) refinements
  precond: 'fsai',    // 'fsai' (pattern of tril(A)), 'fsai2' (tril(A^2): fewer, dearer iterations), 'jacobi'
  checkEvery: 32,     // longest stretch of GPU iterations between read-backs
  minGain: 0.5,       // a refinement must cut a column's residual by this factor ...
  maxSlow: 2,         // ... or, after this many slow ones in a row, the column stops
  maxCols: 256,       // right-hand sides per GPU batch (more are solved in chunks)
  memBudget: 512 * 1024 * 1024,   // bytes of GPU vectors per batch
});

// ---------------------------------------------------------------- device

let gpuPromise = null;

/**
 * The WebGPU device (shared, created once), or null if WebGPU is unavailable.
 * Works in windows and dedicated workers (navigator.gpu exists in both).
 * info.fallback is true for software adapters (SwiftShader), where the
 * solver works but is slow.
 */
export async function getGpu() {
  if (gpuPromise) {
    const g = await gpuPromise;
    if (!g || !g.lost) return g;
    gpuPromise = null;
  }
  gpuPromise = (async () => {
    const nav = typeof navigator !== 'undefined' ? navigator : null;
    if (!nav || !nav.gpu) return null;
    let adapter = null;
    try {
      adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch {
      adapter = null;
    }
    if (!adapter) return null;
    const al = adapter.limits;
    const want = {};
    for (const name of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage']) {
      if (typeof al[name] === 'number') want[name] = al[name];
    }
    let device = null;
    try {
      device = await adapter.requestDevice({ requiredLimits: want });
    } catch {
      try { device = await adapter.requestDevice(); } catch { device = null; }
    }
    if (!device) return null;
    const dl = device.limits;
    if (dl.maxStorageBuffersPerShaderStage < 7 || dl.maxComputeInvocationsPerWorkgroup < WG ||
        dl.maxComputeWorkgroupSizeX < WG || dl.maxComputeWorkgroupStorageSize < 2 * WG * 16) {
      device.destroy();
      return null;
    }
    let ai = adapter.info;
    if (!ai && typeof adapter.requestAdapterInfo === 'function') {
      try { ai = await adapter.requestAdapterInfo(); } catch { ai = null; }
    }
    ai = ai || {};
    const vendor = ai.vendor || '';
    const architecture = ai.architecture || '';
    const fallback = !!(ai.isFallbackAdapter || adapter.isFallbackAdapter || /swiftshader/i.test(architecture + ' ' + (ai.description || '')));
    const gpu = {
      device,
      info: {
        vendor,
        architecture,
        device: ai.device || '',
        description: ai.description || [vendor, architecture].filter(Boolean).join(' ') || 'WebGPU',
        fallback,
        maxStorageBufferBindingSize: dl.maxStorageBufferBindingSize,
        maxBufferSize: dl.maxBufferSize,
      },
      lost: false,
      _pipes: new Map(),     // pipelines by layout + variant
      _states: [],           // buffer sets, most recently used first
    };
    device.lost.then((e) => {
      gpu.lost = true;
      if (gpuPromise) gpuPromise = null;
      if (typeof console !== 'undefined') console.warn('WebGPU device lost:', e && e.message);
    });
    return gpu;
  })();
  return gpuPromise;
}

// ------------------------------------------------- per-matrix preparation

const prepCache = [];      // last few prepared matrices (matched by content)
let prepSerial = 0;

function prepare(n, indptr, indices, data, precond) {
  for (let i = 0; i < prepCache.length; i++) {
    const e = prepCache[i];
    if (e.n === n && e.precond === precond && sameArray(e.indptr, indptr) &&
        sameArray(e.indices, indices) && sameArray(e.data, data)) {
      prepCache.splice(i, 1);
      prepCache.unshift(e);
      return { prep: e, fresh: false };
    }
  }
  const t = now();
  const { s, scaled } = jacobiScale(n, indptr, indices, data);
  const e = {
    id: ++prepSerial, n, precond,
    indptr: indptr.slice(), indices: indices.slice(), data: data.slice(),
    s, A32: Float32Array.from(scaled), G: null, GT: null, fsaiFails: 0, setupMs: 0,
  };
  if (precond === 'fsai' || precond === 'fsai2') {
    const g = fsai(n, indptr, indices, scaled, precond === 'fsai2' ? 2 : 1);
    // transpose the float32 values so that G^T G is exactly symmetric on the GPU
    e.G = { ptr: g.ptr, idx: g.idx, val: Float32Array.from(g.val) };
    e.GT = transposeCSR(n, e.G.ptr, e.G.idx, e.G.val);
    e.fsaiFails = g.fails;
  } else if (precond !== 'jacobi') {
    throw new RangeError(`unknown preconditioner '${precond}'`);
  }
  e.setupMs = now() - t;
  prepCache.unshift(e);
  if (prepCache.length > 2) prepCache.pop();
  return { prep: e, fresh: true };
}

// ------------------------------------------------ GPU buffers and kernels

async function pipeline(gpu, key, code) {
  let p = gpu._pipes.get(key);
  if (!p) {
    p = (async () => {
      const { device } = gpu;
      const module = device.createShaderModule({ code, label: key });
      if (module.getCompilationInfo) {
        const ci = await module.getCompilationInfo();
        const errs = ci.messages.filter((m) => m.type === 'error');
        if (errs.length) throw new Error(`WGSL ${key}: ` + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
      }
      return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: key });
    })();
    gpu._pipes.set(key, p);
    p.catch(() => gpu._pipes.delete(key));
  }
  return p;
}

async function pipelinesFor(gpu, lay, fs) {
  const tag = `${lay.W}x${lay.C}`;
  const want = {
    spmm1: () => rowKernel(lay, { dots: 1 }),
    redAlpha: () => reduceKernel(lay, 'alpha'),
    redBeta: () => reduceKernel(lay, 'beta'),
    redInit: () => reduceKernel(lay, 'init'),
    axpy: () => axpyKernel(lay),
    xpby: () => xpbyKernel(lay),
  };
  if (fs) {
    want.spmm0 = () => rowKernel(lay, { dots: 0 });
    want.spmm2 = () => rowKernel(lay, { dots: 2 });
  } else {
    want.dotR = () => rowKernel(lay, { spmv: false, dots: 2 });
  }
  const names = Object.keys(want);
  const ps = await Promise.all(names.map((nm) => pipeline(gpu, `${nm}/${tag}`, want[nm]())));
  return Object.fromEntries(names.map((nm, i) => [nm, ps[i]]));
}

const align16 = (b) => Math.max(16, Math.ceil(b / 16) * 16);

/** Largest batch of right-hand sides that fits the device limits and the memory budget. */
function maxColumns(gpu, n, want, budget) {
  const lim = gpu.device.limits;
  const cap = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  let k = Math.max(1, Math.min(want, 4 * WG));
  while (k > 1) {
    const kp = layoutFor(k).kp;
    if (n * kp * 4 <= cap && 7 * n * kp * 4 <= budget) break;
    k = k > 4 ? Math.max(4, (k >> 1) & ~3) : k - 1;
  }
  return k;
}

async function getState(gpu, prep, lay) {
  const fs = !!prep.G;
  const key = [prep.n, prep.A32.length, fs ? prep.G.idx.length : 0, lay.kp].join('/');
  let st = gpu._states.find((s) => s.key === key);
  if (st) {
    gpu._states.splice(gpu._states.indexOf(st), 1);
  } else {
    st = await createState(gpu, prep, lay, key);
  }
  gpu._states.unshift(st);
  while (gpu._states.length > 2) gpu._states.pop().destroy();
  if (st.prepId !== prep.id) {
    const q = gpu.device.queue;
    const u32 = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
    q.writeBuffer(st.aPtr, 0, u32(prep.indptr));
    q.writeBuffer(st.aIdx, 0, u32(prep.indices));
    q.writeBuffer(st.aVal, 0, prep.A32);
    if (fs) {
      q.writeBuffer(st.gPtr, 0, u32(prep.G.ptr));
      q.writeBuffer(st.gIdx, 0, u32(prep.G.idx));
      q.writeBuffer(st.gVal, 0, prep.G.val);
      q.writeBuffer(st.tPtr, 0, u32(prep.GT.ptr));
      q.writeBuffer(st.tIdx, 0, u32(prep.GT.idx));
      q.writeBuffer(st.tVal, 0, prep.GT.val);
    }
    st.prepId = prep.id;
  }
  return st;
}

async function createState(gpu, prep, lay, key) {
  const { device } = gpu;
  const n = prep.n;
  const { kp, C, RPW } = lay;
  const fs = !!prep.G;
  const pipes = await pipelinesFor(gpu, lay, fs);
  const nwg = Math.ceil(n / RPW);
  const gx = Math.min(nwg, MAX_GRID);
  const gy = Math.ceil(nwg / gx);
  const ne = n * C;
  const newg = Math.ceil(ne / WG);
  const egx = Math.min(newg, MAX_GRID);
  const egy = Math.ceil(newg / egx);
  const lim = device.limits;
  const vecBytes = n * kp * 4;
  const partBytes = 2 * nwg * kp * 4;
  const biggest = Math.max(vecBytes, partBytes, prep.A32.length * 4, fs ? prep.G.val.length * 4 : 0);
  if (biggest > Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize)) {
    throw new RangeError(`system too large for this GPU (${biggest} byte buffer)`);
  }
  const STO = GPUBufferUsage.STORAGE, DST = GPUBufferUsage.COPY_DST, SRC = GPUBufferUsage.COPY_SRC;
  const all = [];
  const buf = (size, usage, label) => {
    const b = device.createBuffer({ size: align16(size), usage, label });
    all.push(b);
    return b;
  };
  const st = { key, n, lay, fs, pipes, nwg, prepId: 0, vecBytes, scalBytes: NF * kp * 4 };
  st.uni = buf(32, GPUBufferUsage.UNIFORM | DST, 'params');
  st.aPtr = buf((n + 1) * 4, STO | DST, 'A.ptr');
  st.aIdx = buf(prep.A32.length * 4, STO | DST, 'A.idx');
  st.aVal = buf(prep.A32.length * 4, STO | DST, 'A.val');
  if (fs) {
    st.gPtr = buf((n + 1) * 4, STO | DST, 'G.ptr');
    st.gIdx = buf(prep.G.idx.length * 4, STO | DST, 'G.idx');
    st.gVal = buf(prep.G.val.length * 4, STO | DST, 'G.val');
    st.tPtr = buf((n + 1) * 4, STO | DST, 'GT.ptr');
    st.tIdx = buf(prep.GT.idx.length * 4, STO | DST, 'GT.idx');
    st.tVal = buf(prep.GT.val.length * 4, STO | DST, 'GT.val');
  }
  st.X = buf(vecBytes, STO | SRC | DST, 'X');
  st.R = buf(vecBytes, STO | DST, 'R');
  st.P = buf(vecBytes, STO | DST, 'P');
  st.Q = buf(vecBytes, STO, 'Q');
  if (fs) {
    st.Z = buf(vecBytes, STO, 'Z');
    st.T = buf(vecBytes, STO, 'T');
  }
  st.part = buf(partBytes, STO, 'partials');
  st.scal = buf(st.scalBytes, STO | SRC | DST, 'scalars');
  st.stageScal = buf(st.scalBytes, GPUBufferUsage.MAP_READ | DST, 'scalars.read');
  st.stageX = buf(vecBytes, GPUBufferUsage.MAP_READ | DST, 'X.read');
  st.destroy = () => { for (const b of all) b.destroy(); };

  const params = new ArrayBuffer(32);
  const pu = new Uint32Array(params);
  pu[0] = n; pu[1] = nwg; pu[2] = gx; pu[3] = ne; pu[4] = egx;
  st.setEta = (eta) => {
    new Float32Array(params)[5] = eta * eta;
    device.queue.writeBuffer(st.uni, 0, params);
  };

  const bind = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: Object.entries(entries).map(([b, buffer]) => ({ binding: +b, resource: { buffer } })),
  });
  const bg = {};
  bg.A = bind(pipes.spmm1, { 0: st.uni, 1: st.aPtr, 2: st.aIdx, 3: st.aVal, 4: st.P, 5: st.Q, 6: st.part });
  bg.redAlpha = bind(pipes.redAlpha, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.redBeta = bind(pipes.redBeta, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.redInit = bind(pipes.redInit, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.axpy = bind(pipes.axpy, { 0: st.uni, 1: st.scal, 2: st.X, 3: st.R, 4: st.P, 5: st.Q });
  if (fs) {
    bg.G = bind(pipes.spmm0, { 0: st.uni, 1: st.gPtr, 2: st.gIdx, 3: st.gVal, 4: st.R, 5: st.T });
    bg.GT = bind(pipes.spmm2, { 0: st.uni, 1: st.tPtr, 2: st.tIdx, 3: st.tVal, 4: st.T, 5: st.Z, 6: st.part, 7: st.R });
    bg.xpby = bind(pipes.xpby, { 0: st.uni, 1: st.scal, 2: st.Z, 3: st.P });
  } else {
    bg.dotR = bind(pipes.dotR, { 0: st.uni, 6: st.part, 7: st.R });
    bg.xpby = bind(pipes.xpby, { 0: st.uni, 1: st.scal, 2: st.R, 3: st.P });
  }

  const rows = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(gx, gy); };
  const elems = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(egx, egy); };
  const red = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(C); };
  const precondition = (pass) => {
    // z = G^T (G r), with partial sums of r.z and r.r (Jacobi: z = r)
    if (fs) {
      rows(pass, pipes.spmm0, bg.G);
      rows(pass, pipes.spmm2, bg.GT);
    } else {
      rows(pass, pipes.dotR, bg.dotR);
    }
  };
  st.encodeInit = (pass) => {
    precondition(pass);
    red(pass, pipes.redInit, bg.redInit);       // rz, rr0, tol2, active flags; beta = 0
    elems(pass, pipes.xpby, bg.xpby);           // p = z
  };
  st.encodeIter = (pass) => {
    rows(pass, pipes.spmm1, bg.A);              // q = As p, partial p.q
    red(pass, pipes.redAlpha, bg.redAlpha);     // alpha = rz / p.q
    elems(pass, pipes.axpy, bg.axpy);           // x += alpha p, r -= alpha q
    precondition(pass);                         // z = M^-1 r, partial r.z, r.r
    red(pass, pipes.redBeta, bg.redBeta);       // beta = rz'/rz, convergence flags
    elems(pass, pipes.xpby, bg.xpby);           // p = z + beta p
  };
  return st;
}

// ------------------------------------------------------------ inner solve

/**
 * Batched float32 PCG on the scaled system for the (row-major, kp-padded)
 * right-hand sides rs, from x = 0, each column to a relative residual eta
 * (or stagnation, or maxIter).  Returns the float32 solutions (row-major).
 */
async function innerSolve(gpu, st, rs, opt) {
  const { device } = gpu;
  const kp = st.lay.kp;
  const maxIter = opt.maxIter;
  device.queue.writeBuffer(st.R, 0, rs);
  st.setEta(opt.eta);
  const best = new Float64Array(kp).fill(Infinity);
  const bestAt = new Float64Array(kp);
  const prevRR = new Float64Array(kp);
  let its = 0;
  let m = Math.min(opt.checkEvery, maxIter);
  let first = true;
  let stagnated = false;
  let sc = null;
  for (;;) {
    const enc = device.createCommandEncoder();
    if (first) enc.clearBuffer(st.X);
    const pass = enc.beginComputePass();
    if (first) st.encodeInit(pass);
    for (let t = 0; t < m; t++) st.encodeIter(pass);
    pass.end();
    enc.copyBufferToBuffer(st.scal, 0, st.stageScal, 0, st.scalBytes);
    enc.copyBufferToBuffer(st.X, 0, st.stageX, 0, st.vecBytes);
    device.queue.submit([enc.finish()]);
    its += m;
    await st.stageScal.mapAsync(GPUMapMode.READ);
    sc = new Float32Array(st.stageScal.getMappedRange().slice(0));
    st.stageScal.unmap();
    // poll: active columns, convergence rate, stagnation
    let active = 0;
    let need = 0;
    let stalled = true;
    for (let j = 0; j < kp; j++) {
      if (!(sc[F.ACT * kp + j] > 0)) continue;
      active++;
      const rr = sc[F.RR * kp + j];
      const tol2 = sc[F.TOL2 * kp + j];
      if (rr < 0.25 * best[j]) { best[j] = rr; bestAt[j] = its; }
      if (its - bestAt[j] < Math.max(256, its / 2)) stalled = false;
      // iterations still needed at the current rate (log-linear model)
      const rate = first || !(prevRR[j] > 0) ? 0 : Math.log(rr / prevRR[j]) / m;
      const est = rate < 0 && tol2 > 0 ? Math.log(tol2 / rr) / rate : opt.checkEvery;
      if (est > need) need = est;
      prevRR[j] = rr;
    }
    first = false;
    if (!active || its >= maxIter) break;
    if (stalled) { stagnated = true; break; }
    m = Math.max(4, Math.min(opt.checkEvery, Math.ceil(need) + 1, maxIter - its));
  }
  await st.stageX.mapAsync(GPUMapMode.READ);
  const X = new Float32Array(st.stageX.getMappedRange().slice(0));
  st.stageX.unmap();
  const colIts = sc.slice(F.ITS * kp, F.ITS * kp + kp);
  const colRel = new Float64Array(kp);
  for (let j = 0; j < kp; j++) {
    const rr0 = sc[F.RR0 * kp + j];
    colRel[j] = rr0 > 0 ? Math.sqrt(Math.max(0, sc[F.RR * kp + j]) / rr0) : 0;
  }
  return { X, its, colIts, colRel, stagnated };
}

// ----------------------------------------------------------- outer solve

function relOf(rn, bn) {
  const out = new Float64Array(rn.length);
  for (let j = 0; j < rn.length; j++) out[j] = bn[j] > 0 ? rn[j] / bn[j] : rn[j];
  return out;
}

const maxOf = (a) => { let m = 0; for (let i = 0; i < a.length; i++) if (!(a[i] <= m)) m = a[i]; return m; };

/** Mixed-precision refinement for columns [j0, j0+m) of the problem. */
async function solveBlock(gpu, prep, rhs, x0, j0, m, tol, opt) {
  const n = prep.n;
  const lay = layoutFor(m);
  const kp = lay.kp;
  const st = await getState(gpu, prep, lay);
  const { s, indptr, indices, data } = prep;
  const b = colToRow(n, m, rhs, j0);
  const x = x0 ? colToRow(n, m, x0, j0) : new Float64Array(n * m);
  const r = new Float64Array(n * m);
  let tc = now();
  residualRowMajor(n, m, indptr, indices, data, x, b, r);
  const bn = colNormsRowMajor(n, m, b);
  const rel = relOf(colNormsRowMajor(n, m, r), bn);
  let cpuMs = now() - tc;
  let gpuMs = 0;
  const stalled = new Uint8Array(m);
  const slow = new Uint8Array(m);
  const history = [{ relres: maxOf(rel), iters: 0 }];
  const rs = new Float32Array(n * kp);
  const nu = new Float64Array(m);
  const xNew = new Float64Array(n * m);
  const rNew = new Float64Array(n * m);
  let iters = 0;
  let refinements = 0;
  for (let ref = 0; ref < opt.maxRefine; ref++) {
    const act = [];
    for (let j = 0; j < m; j++) if (rel[j] > tol && !stalled[j]) act.push(j);
    if (!act.length) break;
    // scaled residuals, each column normalised to unit length (float32-safe)
    tc = now();
    rs.fill(0);
    for (const j of act) {
      let ss = 0;
      for (let i = 0; i < n; i++) { const t = s[i] * r[i * m + j]; ss += t * t; }
      nu[j] = Math.sqrt(ss);
    }
    for (let i = 0; i < n; i++) {
      const si = s[i];
      for (const j of act) if (nu[j] > 0) rs[i * kp + j] = (si * r[i * m + j]) / nu[j];
    }
    cpuMs += now() - tc;
    const tg = now();
    const inner = await innerSolve(gpu, st, rs, opt);
    gpuMs += now() - tg;
    iters += inner.its;
    refinements++;
    // candidate x + S d, accepted per column only if it lowers the residual
    tc = now();
    xNew.set(x);
    for (let i = 0; i < n; i++) {
      const si = s[i];
      for (const j of act) xNew[i * m + j] += nu[j] * si * inner.X[i * kp + j];
    }
    residualRowMajor(n, m, indptr, indices, data, xNew, b, rNew);
    const relNew = relOf(colNormsRowMajor(n, m, rNew), bn);
    for (const j of act) {
      if (relNew[j] < rel[j]) {
        for (let i = 0; i < n; i++) { x[i * m + j] = xNew[i * m + j]; r[i * m + j] = rNew[i * m + j]; }
        slow[j] = relNew[j] > opt.minGain * rel[j] ? slow[j] + 1 : 0;
        if (slow[j] >= opt.maxSlow) stalled[j] = 1;
        rel[j] = relNew[j];
      } else {
        stalled[j] = 1;           // no progress: keep the previous iterate
      }
    }
    cpuMs += now() - tc;
    let innerRel = 0;
    for (const j of act) innerRel = Math.max(innerRel, inner.colRel[j]);
    history.push({ relres: maxOf(rel), iters: inner.its, innerRel, stagnated: inner.stagnated, active: act.length });
  }
  tc = now();
  const floor = residualFloorRowMajor(n, m, indptr, indices, data, x, b);
  cpuMs += now() - tc;
  return { x, relres: rel, floor, iters, refinements, history, gpuMs, cpuMs, kp };
}

// -------------------------------------------------------------------- API

let queue = Promise.resolve();

/**
 * Solve A X = B for SPD A (full symmetric CSR) and k column-major right-hand sides.
 *
 * Options: n, indptr (Int32Array n+1), indices (Int32Array), data (Float64Array),
 *   rhs (Float64Array n*k, column-major), k (default 1), tol (default 1e-9, on
 *   ||b - A x||_2 / ||b||_2 per column), maxIter (inner PCG iterations per
 *   refinement, default max(2000, 2n)), x0 (initial guess, column-major), and
 *   tuning knobs from DEFAULTS (eta, maxRefine, precond, checkEvery, ...).
 * Returns { x: Float64Array(n*k), iters, refinements, relres: Float64Array(k),
 *   ms, converged } plus diagnostics (setupMs, gpuMs, cpuMs, precond, history).
 * Calls are serialised (the GPU buffers are shared).  Throws on malformed input
 * or a non-positive diagonal; returns converged = false if the tolerance is not
 * reached (relres then says how far it got).
 */
export function solveSPD(gpu, opts) {
  const run = queue.then(() => solveImpl(gpu, opts));
  queue = run.then(() => {}, () => {});
  return run;
}

async function solveImpl(gpu, o) {
  const t0 = now();
  if (!gpu || !gpu.device) throw new Error('solveSPD: no WebGPU device (getGpu() returned null?)');
  if (gpu.lost) throw new Error('solveSPD: the WebGPU device was lost');
  const n = o.n;
  const k = o.k == null ? 1 : o.k;
  if (!(k >= 0) || (k | 0) !== k) throw new RangeError(`k must be a non-negative integer, got ${k}`);
  const indptr = asInt32(o.indptr, 'indptr');
  const indices = asInt32(o.indices, 'indices');
  const data = asFloat64(o.data, 'data');
  checkCSR(n, indptr, indices, data);
  const rhs = asFloat64(o.rhs, 'rhs');
  if (rhs.length < n * k) throw new RangeError(`rhs has ${rhs.length} entries, expected n*k = ${n * k}`);
  const x0 = o.x0 ? asFloat64(o.x0, 'x0') : null;
  if (x0 && x0.length < n * k) throw new RangeError(`x0 has ${x0.length} entries, expected n*k = ${n * k}`);
  const tol = o.tol == null ? 1e-9 : o.tol;
  const opt = { ...DEFAULTS };
  for (const key of Object.keys(DEFAULTS)) if (o[key] != null) opt[key] = o[key];
  opt.maxIter = o.maxIter == null ? Math.max(2000, 2 * n) : o.maxIter;

  const x = new Float64Array(n * k);
  const relres = new Float64Array(k);
  const floor = new Float64Array(k);
  if (n === 0 || k === 0) {
    return { x, iters: 0, refinements: 0, relres, floor, ms: now() - t0, converged: true };
  }
  const nnz = indptr[n];
  const { prep, fresh } = prepare(n, indptr.subarray(0, n + 1), indices.subarray(0, nnz), data.subarray(0, nnz), opt.precond);
  const kmax = maxColumns(gpu, n, opt.maxCols, opt.memBudget);
  const nchunks = Math.ceil(k / kmax);
  const kc = Math.ceil(k / nchunks);
  let iters = 0;
  let refinements = 0;
  let gpuMs = 0;
  let cpuMs = 0;
  const history = [];
  for (let j0 = 0; j0 < k; j0 += kc) {
    const m = Math.min(kc, k - j0);
    const r = await solveBlock(gpu, prep, rhs, x0, j0, m, tol, opt);
    rowToCol(n, m, r.x, x, j0);
    relres.set(r.relres, j0);
    floor.set(r.floor, j0);
    iters += r.iters;
    refinements = Math.max(refinements, r.refinements);
    gpuMs += r.gpuMs;
    cpuMs += r.cpuMs;
    history.push(r.history);
  }
  let converged = true;
  for (let j = 0; j < k; j++) if (!(relres[j] <= tol)) converged = false;
  return {
    x, iters, refinements, relres, ms: now() - t0, converged, floor,
    precond: opt.precond, setupMs: fresh ? prep.setupMs : 0, gpuMs, cpuMs,
    fsaiFails: prep.fsaiFails, chunks: nchunks, history: nchunks === 1 ? history[0] : history,
  };
}
