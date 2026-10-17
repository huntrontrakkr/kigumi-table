// WebGPU solver for sparse symmetric positive definite systems A X = B with
// many right-hand sides, float64-accurate although WGSL only has float32.
//
//   const gpu = await getGpu();                 // { device, info } or null
//   const r = await solveSPD(gpu, { n, indptr, indices, data, rhs, k, tol: 1e-9 });
//   // r = { x: Float64Array(n*k), iters, refinements, relres: Float64Array(k), ms, converged, ... }
//
// A is full symmetric CSR (both triangles), rhs and x are column-major
// (column j at [j*n, (j+1)*n)).  Works on the main thread and in workers, on
// phones too: only WebGPU's default limits are requested (see LIMITS_USED in
// kernels.js), no queue submission runs much longer than DEFAULTS.submitMs,
// and a system too large for the device returns { converged: false, reason }
// at once, so that the caller can use its CPU solver.
//
// Method (mixed-precision iterative refinement):
//   * symmetric Jacobi scaling  As = S A S,  S = diag(1/sqrt(a_ii));
//   * an FSAI preconditioner G (lower triangular, G As G^T ~ I, on the pattern
//     of tril(As^2)), set up once per matrix on the CPU in float64 (one small
//     dense Cholesky per row, bordered for consecutive dofs of a node);
//   * outer loop on the CPU in float64:  r = b - A x;  d = inner(S r);  x += S d;
//     until ||b - A x|| / ||b|| <= tol for every column, or no more progress;
//   * inner solve on the GPU in float32: batched preconditioned CG on As over
//     all k columns at once (CSR SpMM), per-column alpha/beta and convergence
//     flags computed on the GPU, read back only every few dozen iterations,
//     run to a modest relative residual (eta, default 1e-4).
// Each refinement cuts the error by about min(eta, cond(As) * 2^-24): on the
// model's ~4.5k-dof systems 1e-9 takes 3 refinements of ~140-180 PCG
// iterations; on a 15k-dof plate 6 refinements of ~550.  Refinement cannot
// converge once cond(As) * 2^-24 approaches 1 (float32 cannot represent the
// operator well enough).  See tools/gpu-test/NOTES.md for measurements.

import { WG, NF, F, MAX_GRID, LIMITS_USED, layoutFor, rowKernel, reduceKernel, axpyKernel, xpbyKernel } from './kernels.js';
import {
  asInt32, asFloat64, checkCSR, jacobiScale, fsai, transposeCSR, residualRowMajor,
  colNormsRowMajor, colToRow, rowToCol, sameArray, packCSR, packedBytes,
} from './csr.js';

export { LIMITS_USED };

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export const DEFAULTS = Object.freeze({
  eta: 1e-4,          // inner float32 PCG: relative residual target per refinement
  maxRefine: 15,      // cap on outer (float64) refinements
  precond: 'fsai2',   // 'fsai2' (FSAI on the pattern of tril(A^2)), 'fsai' (tril(A): cheaper set-up,
                      //   ~2x the iterations), 'jacobi' (plain CG on the scaled system)
  checkEvery: 32,     // longest stretch of GPU iterations between read-backs
  submitMs: 25,       // GPU time per queue submission (measured, adapted): mobile watchdogs reset long ones
  minGain: 0.5,       // a column stops after `patience` refinements without cutting its
  patience: 2,        //   residual by this factor
  floorFactor: 4,     // 'converged' also accepts relres <= floorFactor * float64 floor (see below)
  maxCols: 256,       // right-hand sides per GPU batch (more are solved in chunks)
  memBudget: 256 * 1024 * 1024,   // bytes of GPU buffers per batch (more columns are chunked)
});

// ---------------------------------------------------------------- device

let gpuPromise = null;

/**
 * The WebGPU device (shared, created once), or null if WebGPU is unavailable
 * or the adapter lacks the default limits the kernels use (LIMITS_USED).
 * Requests no features and no limits beyond the defaults.  Works in windows
 * and dedicated workers (navigator.gpu exists in both).  info.fallback is true
 * for software adapters (SwiftShader), where the solver works but is slow.
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
    let device = null;
    try {
      device = await adapter.requestDevice();           // default limits, no features
    } catch {
      device = null;
    }
    if (!device) return null;
    const dl = device.limits;
    for (const [name, need] of Object.entries(LIMITS_USED)) {
      if (!(dl[name] >= need)) {
        device.destroy();
        return null;
      }
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
      gpu._states.length = 0;
      if (gpuPromise) gpuPromise = null;
      if (typeof console !== 'undefined' && e && e.reason !== 'destroyed') console.warn('WebGPU device lost:', e.message);
    });
    if (typeof device.addEventListener === 'function') {
      device.addEventListener('uncapturederror', (ev) => {
        if (typeof console !== 'undefined') console.warn('WebGPU error:', ev.error && ev.error.message);
      });
    }
    return gpu;
  })();
  return gpuPromise;
}

/** Frees the GPU buffers kept between calls (they are re-made on the next solve). */
export function releaseGpuBuffers(gpu) {
  if (!gpu || !gpu._states) return;
  for (const st of gpu._states) st.destroy();
  gpu._states.length = 0;
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
    indptr: indptr.slice(), indices: indices.slice(), data: data.slice(), s,
    mA: packCSR(n, indptr, indices, Float32Array.from(scaled)), mG: null, mGT: null,
    nnzG: 0, fsaiFails: 0, setupMs: 0,
  };
  if (precond === 'fsai' || precond === 'fsai2') {
    const g = fsai(n, indptr, indices, scaled, precond === 'fsai2' ? 2 : 1);
    // transpose the float32 values so that G^T G is exactly symmetric on the GPU
    const val = Float32Array.from(g.val);
    const gt = transposeCSR(n, g.ptr, g.idx, val);
    e.mG = packCSR(n, g.ptr, g.idx, val);
    e.mGT = packCSR(n, gt.ptr, gt.idx, gt.val);
    e.nnzG = g.idx.length;
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

/** A GPU resource problem (too large, out of memory, device lost): the caller should use its CPU solver. */
class GpuResourceError extends Error {}

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
    spmmA: () => rowKernel(lay, 'pq'),
    redAlpha: () => reduceKernel(lay, 'alpha'),
    redBeta: () => reduceKernel(lay, 'beta'),
    redInit: () => reduceKernel(lay, 'init'),
    axpy: () => axpyKernel(lay),
    xpby: () => xpbyKernel(lay),
  };
  if (fs) {
    want.spmmG = () => rowKernel(lay, 'tt_rr');
    want.spmmGT = () => rowKernel(lay, 'none');
  } else {
    want.dotR = () => rowKernel(lay, 'rr');
  }
  const names = Object.keys(want);
  const ps = await Promise.all(names.map((nm) => pipeline(gpu, `${nm}/${tag}`, want[nm]())));
  return Object.fromEntries(names.map((nm, i) => [nm, ps[i]]));
}

const align16 = (b) => Math.max(16, Math.ceil(b / 16) * 16);

/** Bytes of GPU buffers for a prepared matrix and a lane layout, and the largest single buffer. */
function stateBytes(prep, lay) {
  const n = prep.n;
  const vec = align16(n * lay.kp * 4);
  const nwg = Math.ceil(n / lay.RPW);
  const part = align16(2 * nwg * lay.kp * 4);
  const mats = [prep.mA, prep.mG, prep.mGT].filter(Boolean).map((m) => align16(m.byteLength));
  const nvec = prep.mG ? 7 : 5;                  // X R P Q (Z T) + the X read-back
  const total = nvec * vec + part + mats.reduce((a, b) => a + b, 0) + 3 * align16(NF * lay.kp * 4) + 32;
  return { total, largest: Math.max(vec, part, ...mats) };
}

/** Largest batch of right-hand sides (<= want) whose buffers fit the device and the budget, or 0. */
function maxColumns(gpu, prep, want, budget) {
  const lim = gpu.device.limits;
  const cap = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  let k = Math.max(1, Math.min(want, 4 * WG));
  for (;;) {
    const sb = stateBytes(prep, layoutFor(k));
    if (sb.largest <= cap && sb.total <= budget) return k;
    if (k === 1) return 0;
    k = k > 4 ? Math.max(4, (k >> 1) & ~3) : k - 1;
  }
}

async function getState(gpu, prep, lay) {
  const key = [prep.n, prep.mA.length, prep.nnzG, lay.kp].join('/');
  let st = gpu._states.find((s) => s.key === key);
  if (st) {
    gpu._states.splice(gpu._states.indexOf(st), 1);
  } else {
    // make room first: at most one other buffer set stays cached
    while (gpu._states.length > 1) gpu._states.pop().destroy();
    st = await createState(gpu, prep, lay, key);
  }
  gpu._states.unshift(st);
  if (st.prepId !== prep.id) {
    const q = gpu.device.queue;
    q.writeBuffer(st.mA, 0, prep.mA);
    if (st.fs) {
      q.writeBuffer(st.mG, 0, prep.mG);
      q.writeBuffer(st.mGT, 0, prep.mGT);
    }
    st.prepId = prep.id;
  }
  return st;
}

async function createState(gpu, prep, lay, key) {
  const { device } = gpu;
  const n = prep.n;
  const { kp, C, RPW } = lay;
  const fs = !!prep.mG;
  const pipes = await pipelinesFor(gpu, lay, fs);
  const nwg = Math.ceil(n / RPW);
  const gx = Math.min(nwg, MAX_GRID);
  const gy = Math.ceil(nwg / gx);
  const ne = n * C;
  const newg = Math.ceil(ne / WG);
  const egx = Math.min(newg, MAX_GRID);
  const egy = Math.ceil(newg / egx);
  const vecBytes = n * kp * 4;
  const STO = GPUBufferUsage.STORAGE, DST = GPUBufferUsage.COPY_DST, SRC = GPUBufferUsage.COPY_SRC;
  const all = [];
  let bytes = 0;
  const buf = (size, usage, label) => {
    const b = device.createBuffer({ size: align16(size), usage, label });
    all.push(b);
    bytes += align16(size);
    return b;
  };
  const st = { key, n, lay, fs, pipes, nwg, prepId: 0, vecBytes, scalBytes: NF * kp * 4, msPerIter: 0 };
  // out-of-memory and validation errors surface here instead of as a lost device later
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  st.uni = buf(32, GPUBufferUsage.UNIFORM | DST, 'params');
  st.mA = buf(prep.mA.byteLength, STO | DST, 'A');
  if (fs) {
    st.mG = buf(prep.mG.byteLength, STO | DST, 'G');
    st.mGT = buf(prep.mGT.byteLength, STO | DST, 'GT');
  }
  st.X = buf(vecBytes, STO | SRC | DST, 'X');
  st.R = buf(vecBytes, STO | DST, 'R');
  st.P = buf(vecBytes, STO, 'P');
  st.Q = buf(vecBytes, STO, 'Q');
  if (fs) {
    st.Z = buf(vecBytes, STO, 'Z');
    st.T = buf(vecBytes, STO, 'T');
  }
  st.part = buf(2 * nwg * kp * 4, STO, 'partials');
  st.scal = buf(st.scalBytes, STO | SRC, 'scalars');
  st.stageScal = buf(st.scalBytes, GPUBufferUsage.MAP_READ | DST, 'scalars.read');
  st.stageX = buf(vecBytes, GPUBufferUsage.MAP_READ | DST, 'X.read');
  st.bytes = bytes;
  st.destroy = () => { for (const b of all) b.destroy(); };
  const errV = await device.popErrorScope();
  const errM = await device.popErrorScope();
  if (errV || errM) {
    st.destroy();
    throw new GpuResourceError(`GPU buffers (${(bytes / 2 ** 20).toFixed(1)} MiB): ${(errM || errV).message}`);
  }

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
  bg.A = bind(pipes.spmmA, { 0: st.uni, 1: st.mA, 2: st.P, 3: st.Q, 4: st.part });
  bg.redAlpha = bind(pipes.redAlpha, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.redBeta = bind(pipes.redBeta, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.redInit = bind(pipes.redInit, { 0: st.uni, 1: st.part, 2: st.scal });
  bg.axpy = bind(pipes.axpy, { 0: st.uni, 1: st.scal, 2: st.R, 3: st.Q });
  if (fs) {
    bg.G = bind(pipes.spmmG, { 0: st.uni, 1: st.mG, 2: st.R, 3: st.T, 4: st.part });
    bg.GT = bind(pipes.spmmGT, { 0: st.uni, 1: st.mGT, 2: st.T, 3: st.Z });
    bg.xpby = bind(pipes.xpby, { 0: st.uni, 1: st.scal, 2: st.X, 3: st.Z, 4: st.P });
  } else {
    bg.dotR = bind(pipes.dotR, { 0: st.uni, 2: st.R, 4: st.part });
    bg.xpby = bind(pipes.xpby, { 0: st.uni, 1: st.scal, 2: st.X, 3: st.R, 4: st.P });
  }

  const rows = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(gx, gy); };
  const elems = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(egx, egy); };
  const red = (pass, p, g) => { pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(C); };
  const precondition = (pass) => {
    // z = G^T (G r), with partial sums of r.z = |G r|^2 and r.r (Jacobi: z = r)
    if (fs) {
      rows(pass, pipes.spmmG, bg.G);
      rows(pass, pipes.spmmGT, bg.GT);
    } else {
      rows(pass, pipes.dotR, bg.dotR);
    }
  };
  st.encodeInit = (pass) => {
    precondition(pass);
    red(pass, pipes.redInit, bg.redInit);       // rz, rr0, tol2, active flags; alpha = beta = 0
    elems(pass, pipes.xpby, bg.xpby);           // p = z
  };
  st.encodeIter = (pass) => {
    rows(pass, pipes.spmmA, bg.A);              // q = As p, partial p.q
    red(pass, pipes.redAlpha, bg.redAlpha);     // alpha = rz / p.q
    elems(pass, pipes.axpy, bg.axpy);           // r -= alpha q
    precondition(pass);                         // z = M^-1 r, partial r.z, r.r
    red(pass, pipes.redBeta, bg.redBeta);       // beta = rz'/rz, convergence flags
    elems(pass, pipes.xpby, bg.xpby);           // x += alpha p, p = z + beta p
  };
  return st;
}

// ------------------------------------------------------------ inner solve

/**
 * Batched float32 PCG on the scaled system for the (row-major, kp-padded)
 * right-hand sides rs, from x = 0, each column to a relative residual eta
 * (or stagnation, or maxIter).  Returns the float32 solutions (row-major).
 *
 * The iterations between two read-backs are split into queue submissions of
 * about opt.submitMs of GPU time each (the time per iteration is measured on
 * the first few iterations and kept per buffer set), so that a slow mobile GPU
 * never sees one long-running submission.
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
  let m = Math.min(st.msPerIter ? opt.checkEvery : 4, maxIter);   // unknown speed: a short first stretch
  let started = false;
  let havePrev = false;
  let stagnated = false;
  let submits = 0;
  let maxPer = 0;
  let sc = null;
  for (;;) {
    const per = st.msPerIter ? Math.max(1, Math.min(m, Math.floor(opt.submitMs / st.msPerIter))) : 1;
    if (per > maxPer) maxPer = per;
    const t0 = now();
    for (let left = m; left > 0;) {
      const s = Math.min(per, left);
      left -= s;
      const enc = device.createCommandEncoder();
      if (!started) enc.clearBuffer(st.X);
      const pass = enc.beginComputePass();
      if (!started) st.encodeInit(pass);
      started = true;
      for (let t = 0; t < s; t++) st.encodeIter(pass);
      pass.end();
      if (left === 0) {
        enc.copyBufferToBuffer(st.scal, 0, st.stageScal, 0, st.scalBytes);
        enc.copyBufferToBuffer(st.X, 0, st.stageX, 0, st.vecBytes);
      }
      device.queue.submit([enc.finish()]);
      submits++;
    }
    its += m;
    await st.stageScal.mapAsync(GPUMapMode.READ);
    sc = new Float32Array(st.stageScal.getMappedRange().slice(0));
    st.stageScal.unmap();
    const msIt = (now() - t0) / m;
    st.msPerIter = st.msPerIter ? 0.5 * (st.msPerIter + msIt) : msIt;
    // poll: active columns, convergence rate, stagnation
    let active = 0;
    let need = 0;
    let stalled = true;
    for (let j = 0; j < kp; j++) {
      if (!(sc[F.ACT * kp + j] > 0)) continue;
      active++;
      const rr = sc[F.RR * kp + j];
      const tol2 = sc[F.TOL2 * kp + j];
      // stagnation: rr has not dropped 4x below its best for the last half of the run
      // (and >= 1000 iterations: the PCG residual 2-norm can rise for hundreds of
      // iterations on stiff 4th-order problems before it falls)
      if (rr < 0.25 * best[j]) { best[j] = rr; bestAt[j] = its; }
      if (its - bestAt[j] < Math.max(1000, its / 2)) stalled = false;
      // iterations still needed at the current rate (log-linear model)
      const rate = !havePrev || !(prevRR[j] > 0) ? 0 : Math.log(rr / prevRR[j]) / m;
      const est = rate < 0 && tol2 > 0 ? Math.log(tol2 / rr) / rate : opt.checkEvery;
      if (est > need) need = est;
      prevRR[j] = rr;
    }
    havePrev = true;
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
  return { X, its, colIts, colRel, stagnated, submits, maxPer };
}

// ----------------------------------------------------------- outer solve

function relOf(rn, bn) {
  const out = new Float64Array(rn.length);
  for (let j = 0; j < rn.length; j++) out[j] = bn[j] > 0 ? rn[j] / bn[j] : rn[j];
  return out;
}

const maxOf = (a) => { let m = 0; for (let i = 0; i < a.length; i++) if (!(a[i] <= m)) m = a[i]; return m; };

/**
 * Mixed-precision refinement for columns [j0, j0+m) of the problem.
 *
 * Every refinement step x += S d is taken (classical iterative refinement):
 * the float32 inner solve reduces the *error* by roughly eta (or by what
 * float32 can resolve, ~ cond(As) * 2^-24), but on stiff mixed-unit systems the
 * first steps can raise the unscaled residual norm while the error drops by
 * orders of magnitude (beam tests), so steps are not judged by the residual.
 * The iterate with the lowest residual is kept per column and returned.  A
 * column stops when relres <= tol, when relres reaches its float64 floor, or
 * after `patience` refinements without halving its residual.
 */
async function solveBlock(gpu, prep, rhs, x0, j0, m, tol, opt) {
  const n = prep.n;
  const lay = layoutFor(m);
  const kp = lay.kp;
  let st = null;                                // GPU buffers, made when first needed
  const { s, indptr, indices, data } = prep;
  const b = colToRow(n, m, rhs, j0);
  const x = x0 ? colToRow(n, m, x0, j0) : new Float64Array(n * m);
  const xBest = Float64Array.from(x);
  const r = new Float64Array(n * m);
  const floor = new Float64Array(m);
  let tc = now();
  residualRowMajor(n, m, indptr, indices, data, x, b, r, x0 ? floor : null);
  const bn = colNormsRowMajor(n, m, b);
  const rel = relOf(colNormsRowMajor(n, m, r), bn);
  let cpuMs = now() - tc;
  let gpuMs = 0;
  let submits = 0;
  let maxPer = 0;
  const best = Float64Array.from(rel);          // lowest relres so far (iterate in xBest)
  const mark = Float64Array.from(rel);          // relres at the last halving ...
  const markAt = new Int32Array(m).fill(-1);    // ... and when
  const done = new Uint8Array(m);
  const why = new Array(m).fill('');
  const history = [{ relres: maxOf(rel), iters: 0 }];
  const rs = new Float32Array(n * kp);
  const nu = new Float64Array(m);
  const okCol = new Uint8Array(m);
  let iters = 0;
  let refinements = 0;
  let failure = null;
  const stop = (j, ref) => {
    if (rel[j] <= tol) return 'tol';
    if (floor[j] > 0 && rel[j] <= floor[j]) return 'floor';
    if (ref - markAt[j] >= opt.patience) return 'stalled';
    return '';
  };
  for (let j = 0; j < m; j++) if ((why[j] = rel[j] <= tol ? 'tol' : '')) done[j] = 1;
  const ac = new Int32Array(m);                 // active columns (row-major loops below)
  const inv = new Float64Array(m);
  const imp = new Uint8Array(m);
  for (let ref = 0; ref < opt.maxRefine; ref++) {
    let na = 0;
    for (let j = 0; j < m; j++) if (!done[j]) ac[na++] = j;
    if (!na) break;
    const act = Array.from(ac.subarray(0, na));
    // scaled residuals, each column normalised to unit length (float32-safe)
    tc = now();
    rs.fill(0);
    nu.fill(0);
    for (let i = 0; i < n; i++) {
      const si = s[i], o = i * m;
      for (let a = 0; a < na; a++) { const j = ac[a]; const t = si * r[o + j]; nu[j] += t * t; }
    }
    for (let a = 0; a < na; a++) { const j = ac[a]; nu[j] = Math.sqrt(nu[j]); inv[j] = nu[j] > 0 ? 1 / nu[j] : 0; }
    for (let i = 0; i < n; i++) {
      const si = s[i], o = i * m, q = i * kp;
      for (let a = 0; a < na; a++) { const j = ac[a]; rs[q + j] = si * r[o + j] * inv[j]; }
    }
    cpuMs += now() - tc;
    const tg = now();
    let inner;
    try {
      if (!st) st = await getState(gpu, prep, lay);
      inner = await innerSolve(gpu, st, rs, opt);
    } catch (e) {
      // GPU trouble (out of memory, lost device, ...): keep the best iterate so far
      failure = e;
      for (const j of act) why[j] = 'gpu';
      break;
    }
    gpuMs += now() - tg;
    iters += inner.its;
    submits += inner.submits;
    if (inner.maxPer > maxPer) maxPer = inner.maxPer;
    refinements++;
    // x += S d (columns with a non-finite correction are left alone and stop)
    tc = now();
    const X = inner.X;
    for (let a = 0; a < na; a++) { const j = ac[a]; okCol[j] = nu[j] > 0 ? 1 : 0; }
    for (let i = 0; i < n; i++) {
      const q = i * kp;
      for (let a = 0; a < na; a++) { const j = ac[a]; if (!Number.isFinite(X[q + j])) okCol[j] = 0; }
    }
    for (let a = 0; a < na; a++) { const j = ac[a]; inv[j] = okCol[j] ? nu[j] : 0; }   // scale, 0 = skip
    for (let i = 0; i < n; i++) {
      const si = s[i], o = i * m, q = i * kp;
      for (let a = 0; a < na; a++) { const j = ac[a]; x[o + j] += inv[j] * si * X[q + j]; }
    }
    // the float64 floor depends on |x| only: evaluate it with the first refined x and at the end
    residualRowMajor(n, m, indptr, indices, data, x, b, r, ref === 0 ? floor : null);
    const relNew = relOf(colNormsRowMajor(n, m, r), bn);
    let anyImp = false;
    for (let a = 0; a < na; a++) {
      const j = ac[a];
      rel[j] = relNew[j];
      imp[j] = rel[j] < best[j] ? 1 : 0;
      if (imp[j]) { best[j] = rel[j]; anyImp = true; }
      if (rel[j] <= opt.minGain * mark[j]) { mark[j] = rel[j]; markAt[j] = ref; }
      why[j] = okCol[j] ? stop(j, ref) : 'non-finite';
      if (why[j]) done[j] = 1;
    }
    if (anyImp) {
      for (let i = 0; i < n; i++) {
        const o = i * m;
        for (let a = 0; a < na; a++) { const j = ac[a]; if (imp[j]) xBest[o + j] = x[o + j]; }
      }
    }
    cpuMs += now() - tc;
    let innerRel = 0;
    for (const j of act) innerRel = Math.max(innerRel, inner.colRel[j]);
    history.push({ relres: maxOf(rel), iters: inner.its, innerRel, stagnated: inner.stagnated, active: act.length,
      ...(m <= 16 ? { cols: Array.from(rel), innerCols: act.map((j) => inner.colRel[j]) } : {}) });
  }
  // return the best iterate of each column (and the float64 floor at it)
  tc = now();
  let restore = false;
  for (let j = 0; j < m; j++) { imp[j] = rel[j] <= best[j] ? 0 : 1; if (imp[j]) restore = true; }
  if (restore) {
    for (let i = 0; i < n; i++) {
      const o = i * m;
      for (let j = 0; j < m; j++) if (imp[j]) x[o + j] = xBest[o + j];
    }
  }
  residualRowMajor(n, m, indptr, indices, data, x, b, r, floor);
  const relFinal = relOf(colNormsRowMajor(n, m, r), bn);
  cpuMs += now() - tc;
  for (let j = 0; j < m; j++) if (!why[j]) why[j] = 'maxRefine';
  return { x, relres: relFinal, floor, iters, refinements, history, gpuMs, cpuMs, kp, why, failure,
    gpuBytes: st ? st.bytes : 0, submits, maxPer, msPerIter: st ? st.msPerIter : 0 };
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
 *   tuning knobs from DEFAULTS (eta, maxRefine, precond, checkEvery, submitMs, ...).
 * Returns { x: Float64Array(n*k), iters, refinements, relres: Float64Array(k),
 *   ms, converged } plus diagnostics: floor (Float64Array(k), the float64
 *   attainable-residual estimate u||A||x|+|b||/||b|| of each column), tolMet
 *   (every relres <= tol), stopped (per column: 'tol' | 'floor' | 'stalled' |
 *   'maxRefine' | 'non-finite' | 'gpu' | 'unsolved'), reason (why not
 *   converged, when it is a resource problem), setupMs, gpuMs, cpuMs, gpuBytes
 *   (GPU buffers of the solve), submits / maxItersPerSubmit / msPerIter (queue
 *   submissions, the most PCG iterations in one, measured ms per iteration),
 *   precond, history.
 * converged: every column has relres <= tol, or relres <= floorFactor * floor
 *   (solved to float64 working precision: no float64 method -- SuperLU
 *   included -- certifies a smaller residual for that column).  Use tolMet for
 *   the strict test.
 * A system too large for the device's buffers or the memory budget, an
 *   out-of-memory error or a lost device give converged = false with `reason`
 *   (quickly: the size checks come before any set-up).  Malformed input and a
 *   non-positive diagonal throw.  Calls are serialised (the GPU buffers are
 *   shared).  If the tolerance is not reached, relres says how far it got and
 *   x is the best iterate found.
 */
export function solveSPD(gpu, opts) {
  const run = queue.then(() => solveImpl(gpu, opts));
  queue = run.then(() => {}, () => {});
  return run;
}

/** The answer for a system the GPU will not take: x0 (or 0), with its residual. */
function unsolved(n, k, indptr, indices, data, rhs, x0, reason, t0) {
  const x = x0 ? Float64Array.from(x0.subarray(0, n * k)) : new Float64Array(n * k);
  const relres = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    let rr = 0, bb = 0;
    const o = j * n;
    for (let i = 0; i < n; i++) {
      let ax = 0;
      if (x0) for (let p = indptr[i]; p < indptr[i + 1]; p++) ax += data[p] * x[o + indices[p]];
      const ri = rhs[o + i] - ax;
      rr += ri * ri;
      bb += rhs[o + i] * rhs[o + i];
    }
    relres[j] = bb > 0 ? Math.sqrt(rr / bb) : Math.sqrt(rr);
  }
  return {
    x, iters: 0, refinements: 0, relres, ms: now() - t0, converged: false, tolMet: false,
    floor: new Float64Array(k), stopped: new Array(k).fill('unsolved'), reason, precond: 'none',
    gpuMs: 0, cpuMs: 0, setupMs: 0, gpuBytes: 0, submits: 0,
  };
}

async function solveImpl(gpu, o) {
  const t0 = now();
  if (!gpu || !gpu.device) throw new Error('solveSPD: no WebGPU device (getGpu() returned null?)');
  if (gpu.lost) {
    // the device was lost (driver reset, GPU process crash): use a fresh one
    const fresh = await getGpu();
    if (!fresh) throw new Error('solveSPD: the WebGPU device was lost and none is available');
    gpu = fresh;
  }
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
    return { x, iters: 0, refinements: 0, relres, floor, ms: now() - t0, converged: true, tolMet: true, stopped: [] };
  }
  const nnz = indptr[n];
  // too large for this device?  Decide before any set-up work.
  const lim = gpu.device.limits;
  const cap = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  const minBytes = 5 * align16(n * 4) + align16(packedBytes(n, nnz));
  if (packedBytes(n, nnz) > cap || minBytes > opt.memBudget) {
    return unsolved(n, k, indptr, indices, data, rhs, x0,
      `system too large for this GPU (matrix ${(packedBytes(n, nnz) / 2 ** 20).toFixed(1)} MiB, binding limit ` +
      `${(cap / 2 ** 20).toFixed(0)} MiB, budget ${(opt.memBudget / 2 ** 20).toFixed(0)} MiB)`, t0);
  }
  // the preconditioner, stepping down (fsai2 -> fsai -> jacobi) if its factor does not fit
  const ladder = ['fsai2', 'fsai', 'jacobi'];
  let precond = opt.precond;
  let { prep, fresh } = prepare(n, indptr.subarray(0, n + 1), indices.subarray(0, nnz), data.subarray(0, nnz), precond);
  let kmax = maxColumns(gpu, prep, opt.maxCols, opt.memBudget);
  while (!kmax && ladder.indexOf(precond) >= 0 && ladder.indexOf(precond) < ladder.length - 1) {
    precond = ladder[ladder.indexOf(precond) + 1];
    ({ prep, fresh } = prepare(n, indptr.subarray(0, n + 1), indices.subarray(0, nnz), data.subarray(0, nnz), precond));
    kmax = maxColumns(gpu, prep, opt.maxCols, opt.memBudget);
  }
  if (!kmax) {
    return unsolved(n, k, indptr, indices, data, rhs, x0, 'system too large for this GPU (vectors exceed the limits)', t0);
  }
  const nchunks = Math.ceil(k / kmax);
  const kc = Math.ceil(k / nchunks);
  let iters = 0;
  let refinements = 0;
  let gpuMs = 0;
  let cpuMs = 0;
  let gpuBytes = 0;
  let submits = 0;
  let maxItersPerSubmit = 0;
  let msPerIter = 0;
  let reason;
  const history = [];
  const stopped = new Array(k);
  for (let j0 = 0; j0 < k; j0 += kc) {
    const m = Math.min(kc, k - j0);
    const r = await solveBlock(gpu, prep, rhs, x0, j0, m, tol, opt);
    rowToCol(n, m, r.x, x, j0);
    relres.set(r.relres, j0);
    floor.set(r.floor, j0);
    for (let j = 0; j < m; j++) stopped[j0 + j] = r.why[j];
    iters += r.iters;
    refinements = Math.max(refinements, r.refinements);
    gpuMs += r.gpuMs;
    cpuMs += r.cpuMs;
    gpuBytes = Math.max(gpuBytes, r.gpuBytes);
    submits += r.submits;
    maxItersPerSubmit = Math.max(maxItersPerSubmit, r.maxPer);
    msPerIter = r.msPerIter || msPerIter;
    history.push(r.history);
    if (r.failure) {
      // free what we hold: the next call starts from fresh buffers (or a fresh device)
      releaseGpuBuffers(gpu);
      reason = r.failure instanceof GpuResourceError ? r.failure.message : `GPU error: ${r.failure.message || r.failure}`;
      for (let j = j0 + m; j < k; j++) stopped[j] = 'unsolved';
      if (j0 + m < k) {
        const rest = unsolved(n, k, indptr, indices, data, rhs, x0, reason, t0);
        for (let j = j0 + m; j < k; j++) {
          relres[j] = rest.relres[j];
          x.set(rest.x.subarray(j * n, (j + 1) * n), j * n);
        }
      }
      break;
    }
  }
  let converged = !reason;
  let tolMet = !reason;
  for (let j = 0; j < k; j++) {
    if (!(relres[j] <= tol)) tolMet = false;
    if (!(relres[j] <= Math.max(tol, opt.floorFactor * floor[j]))) converged = false;
  }
  return {
    x, iters, refinements, relres, ms: now() - t0, converged, tolMet, floor, stopped,
    ...(reason ? { reason } : {}),
    precond, setupMs: fresh ? prep.setupMs : 0, gpuMs, cpuMs, gpuBytes, submits, maxItersPerSubmit, msPerIter,
    fsaiFails: prep.fsaiFails, chunks: nchunks, history: nchunks === 1 ? history[0] : history,
  };
}
