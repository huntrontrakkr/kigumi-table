// Progressive GPU path tracer for "Render photo" (WebGPU compute).
//
//   import { renderPhoto } from './render/pathtracer.js';
//   const blob = await renderPhoto({ viewer, view: 'hero', onProgress: (frac, spp) => {} });
//   if (!blob) { /* no WebGPU, scene over the caps, or the GPU was lost: raster photo */ }
//
// Takes the visible meshes of viewer.model, builds a BVH on the CPU (bvh.js), and
// path-traces them on the GPU (kernel.js) in a studio: a warm key light from
// viewer.keyDir, a cool fill, an environment of the background colour and an
// infinite matte paper floor whose colour is solved so it fades into the backdrop
// like the live view.  Khronos PBR Neutral tone mapping, sRGB, PNG out.
//
// Phones: no optional features; storage sizes only up to the adapter's own limits;
// at most 3 storage buffers per stage; work split into dispatches of about 30 ms;
// a time budget (budgetMs) after which it stops and delivers the samples so far;
// triangle and memory caps beyond which it resolves to null.

import { buildScene } from './build.js';
import { extractScene, countTriangles, MAT_VEC4, XF_VEC4 } from './scene.js';
import { traceWGSL, resolveWGSL, PARAMS_BYTES } from './kernel.js';

export const POSES = {
  hero: { azimuth: 38, elevation: 24 },
  top: { azimuth: 0, elevation: 89 },
  low: { azimuth: 25, elevation: 6 },
  side: { azimuth: 90, elevation: 2 },
  front: { azimuth: 0, elevation: 14 },
};

// Studio, in the units of the live view (a white surface square to the key light
// at 1 gets radiance 1/pi).  Irradiances at normal incidence; colours have unit
// luminance and are scaled by the levels.  The key is a distant disc light: its
// angular size sets the softness of the shadows (at 1.5 degrees the lattice still
// draws its pattern on the floor, with penumbrae widening away from the parts;
// 8-10 degrees is a big softbox nearby and washes the pattern out).  The floor's
// colour is solved from these levels, so they change the table, not the paper.
export const STUDIO = {
  key: { E: 2.8, color: [1.0, 0.95, 0.87], halfAngle: 1.5 },
  fill: { E: 0.5, color: [0.86, 0.94, 1.08], halfAngle: 24, elevation: 26 },
  env: { horizon: 0.7, zenith: 0.95 },        // radiance, tinted by the background colour
  clamp: 4,                                    // max radiance of an indirect sample
  bump: 1,
};

export const MOBILE = isMobile();

// Defaults: desktop renders 1600 x 1100 at 256 samples (about 4 s on an RTX 3090);
// phones 1200 x 825.  Both stop when the budget is spent, with what they have.
export const DEFAULTS = {
  width: MOBILE ? 1200 : 1600,
  height: MOBILE ? 825 : 1100,
  samples: 256,
  budgetMs: 15000,                 // whole call: scene, BVH, tracing and the PNG
  maxTriangles: MOBILE ? 500000 : 3000000,
  maxPixels: MOBILE ? 2.5e6 : 16.8e6,
};

const DISPATCH_MS = 40;           // GPU time per dispatch: far below mobile watchdogs (2 s TDR)
const MAX_EVENTS = 32;            // path vertices including glass interfaces: light entering a
                                  // sheet's edge runs along it by total internal reflection; a
                                  // longer chain costs one thread more time than a dispatch has

export let lastStats = null;

let gpuPromise = null;
let chain = Promise.resolve();

// --------------------------------------------------------------------------- API
// Resolves to a PNG Blob, or null when WebGPU is unavailable, the scene or image is
// over the caps, the GPU device was lost or anything else failed (lastStats.reason
// says which; the caller falls back on the raster photo).  Rejects only with an
// AbortError when `signal` aborts.
export function renderPhoto(opts = {}) {
  const run = chain.then(() => render(opts)).catch((e) => {
    if (e && e.name === 'AbortError') throw e;
    console.warn('path tracer:', e);
    lastStats = { reason: 'error', message: String((e && e.message) || e) };
    return null;
  });
  chain = run.catch(() => {});                        // one render at a time
  return run;
}

// Optional: create the device and compile the pipelines ahead of the first render
// (shader compilation can take seconds on phones).  Resolves true when WebGPU is
// usable.  `viewer` is only used to borrow three.js's device if ours fails.
export async function preparePathTracer(viewer) {
  const g = await getGPU(viewer);
  if (!g) return false;
  try { await g.pipelines; getWorker(); return true; } catch (e) { return false; }
}

// Release the GPU device and pipelines (they are kept between renders).
export async function disposePathTracer() {
  const g = await (gpuPromise || Promise.resolve(null));
  gpuPromise = null;
  if (g && g.own) g.device.destroy();
}

async function render({ viewer, view = 'hero', width = DEFAULTS.width, height = DEFAULTS.height, samples = DEFAULTS.samples,
  budgetMs = DEFAULTS.budgetMs, maxTriangles = DEFAULTS.maxTriangles, onProgress, signal, studio = STUDIO, _wgsl, _dispatchMs } = {}) {
  if (!viewer || !viewer.model) throw new Error('renderPhoto: the viewer has no model');
  const T0 = performance.now();
  const fail = (reason, extra = {}) => { lastStats = { reason, ...extra }; return null; };
  const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
  width = Math.max(8, Math.round(num(width, DEFAULTS.width)));
  height = Math.max(8, Math.round(num(height, DEFAULTS.height)));
  samples = Math.max(1, Math.round(num(samples, DEFAULTS.samples)));
  budgetMs = Math.max(0, num(budgetMs, DEFAULTS.budgetMs));
  maxTriangles = num(maxTriangles, DEFAULTS.maxTriangles);
  const pixels = width * height;
  if (pixels > DEFAULTS.maxPixels) return fail('image too large', { pixels });

  // the scene's size first: nothing big is allocated for a scene over the cap
  const triCount = countTriangles(viewer.model);
  if (triCount > maxTriangles) return fail('too many triangles', { triangles: triCount, maxTriangles });

  const gpu = await getGPU(viewer);
  if (!gpu) return fail('no WebGPU');
  const { device } = gpu;
  const lim = device.limits;
  const maxBinding = Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize);
  if (pixels * 16 > maxBinding) return fail('image too large for this GPU', { pixels });

  const cam = cameraFor(viewer, view, width / height);
  const lights = lighting(viewer, studio);
  const sc = extractScene(viewer.model);
  const N = sc.count, nMeshes = sc.nMeshes, nMats = sc.nMats, extractMs = sc.ms;
  // the GPU buffers' sizes are known before building: refuse what will not fit
  const geoBytes = 16 * (4 * Math.max(1, N) + nMats * MAT_VEC4 + Math.max(1, nMeshes) * XF_VEC4);
  if (geoBytes > maxBinding || 64 * Math.max(1, N) > maxBinding) {
    return fail('scene too large for this GPU', { triangles: N, bytes: geoBytes, maxBinding });
  }
  // BVH and packing off the main thread (a second or more for big scenes on phones)
  const built = await buildOffThread(sc);
  const { nodes, geo, offShade, offMats, offXf } = built;
  const T1 = performance.now();
  if (signal?.aborted) throw new DOMException('Render cancelled', 'AbortError');

  const bufs = [];
  const lost = { flag: false };
  const lostWatch = device.lost.then(() => { lost.flag = true; });
  const make = (size, usage) => { const b = device.createBuffer({ size: Math.max(16, align(size, 16)), usage }); bufs.push(b); return b; };
  const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST;
  try {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    const bNodes = make(nodes.byteLength, S | D);
    device.queue.writeBuffer(bNodes, 0, nodes.buffer, nodes.byteOffset, nodes.byteLength);
    const bGeo = make(geo.byteLength, S | D);
    device.queue.writeBuffer(bGeo, 0, geo.buffer, 0, geo.byteLength);
    const bAccum = make(pixels * 16, S | D);          // WebGPU buffers start zeroed
    const bParams = make(PARAMS_BYTES, GPUBufferUsage.UNIFORM | D);
    const bOut = make(pixels * 4, S | GPUBufferUsage.COPY_SRC);
    const bRead = make(pixels * 4, GPUBufferUsage.MAP_READ | D);
    const bRes = make(32, GPUBufferUsage.UNIFORM | D);

    // _wgsl: a patched kernel source (tests only), compiled for this render
    const { trace, resolve } = await (_wgsl ? compile(device, _wgsl) : gpu.pipelines);
    const bgTrace = device.createBindGroup({
      layout: trace.getBindGroupLayout(0),
      entries: [bParams, bAccum, bNodes, bGeo].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const bgResolve = device.createBindGroup({
      layout: resolve.getBindGroupLayout(0),
      entries: [bRes, bAccum, bOut].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const vErr = await device.popErrorScope();
    const mErr = await device.popErrorScope();
    if (mErr) return fail('out of GPU memory', { message: mErr.message });
    if (vErr) throw new Error('path tracer: ' + vErr.message);
    const T2 = performance.now();

    // ---- progressive passes
    const params = new ArrayBuffer(PARAMS_BYTES);
    const pu = new Uint32Array(params), pf = new Float32Array(params);
    const setV = (i, v, w = 0) => { pf[4 * i] = v[0]; pf[4 * i + 1] = v[1]; pf[4 * i + 2] = v[2]; pf[4 * i + 3] = w; };
    pu[0] = width; pu[1] = height;
    pu[6] = MAX_EVENTS; pu[7] = 0x2545f491;
    setV(2, cam.pos, 2 * cam.tanV / height);       // pixel angle: the ray cone's spread
    setV(3, cam.right.map((x) => x * cam.tanH));
    setV(4, cam.up.map((x) => x * cam.tanV));
    setV(5, cam.fwd);
    setV(6, lights.keyDir, lights.keyCos);
    setV(7, lights.keyL, lights.keyProb);
    setV(8, lights.fillDir, lights.fillCos);
    setV(9, lights.fillL);
    setV(10, lights.envH);
    setV(11, lights.envZ);
    setV(12, lights.backdrop, studio.clamp ?? STUDIO.clamp);
    setV(13, lights.floor, 0);
    setV(14, [studio.bump ?? 1, 0, 0]);
    pu[60] = offShade; pu[61] = offMats; pu[62] = offXf;

    // Dispatches: a pass visits the 8-row blocks in a strided order (see the
    // kernel), each dispatch taking the next n blocks, spread over the image, so
    // they all hold a similar mix of floor, wood and glass and their time is close
    // to proportional to n.  n follows the measured time towards the target (at
    // most 2x per step, from 2 blocks); once a dispatch holds the whole frame,
    // the number of samples per dispatch follows it the same way.
    const dispatchMs = _dispatchMs || DISPATCH_MS;
    const aim = 0.75 * dispatchMs;
    const reserveMs = 100 + pixels * (MOBILE ? 6e-4 : 2e-4);    // resolve, read back, PNG
    const nb = Math.ceil(height / 8);
    let stride = Math.max(1, Math.round(nb * 0.618));
    while (gcd(stride, nb) !== 1) stride++;
    pu[63] = stride;
    let n = Math.min(nb, 2);     // blocks per dispatch
    let k = 1;                   // samples per dispatch, once n covers the frame
    let perBlock = 0;            // ms per block-sample (moving average, for the budget)
    let spp = 0;                 // complete passes (every pixel has at least this many samples)
    let traceMs = 0, dispatches = 0, maxDispatchMs = 0, stopped = '';
    const slowest = [];          // the five longest dispatches [ms, spp, first block, blocks, samples]
    // progress: samples done of those asked, or time spent of the budget, whichever
    // is further along (the budget usually decides on slow GPUs)
    let reported = 0;
    const report = (sppDone) => {
      const f = Math.min(0.99, Math.max(sppDone / samples, (performance.now() - T0) / budgetMs));
      if (f > reported) { reported = f; onProgress?.(f, Math.floor(sppDone)); }
    };
    passes: while (spp < samples) {
      const whole = n >= nb;
      const kk = whole ? Math.min(k, samples - spp) : 1;
      for (let s0 = 0; s0 < nb;) {
        if (signal?.aborted) throw new DOMException('Render cancelled', 'AbortError');
        const m = Math.min(whole ? nb : n, nb - s0);
        // the budget: stop between dispatches once the next one would overrun it
        // (sample counts are kept per pixel, so a partial pass is fine)
        if (spp > 0 && performance.now() - T0 + perBlock * m * kk + reserveMs > budgetMs) { stopped = 'budget'; break passes; }
        pu[2] = spp; pu[3] = kk; pu[4] = s0; pu[5] = m;
        device.queue.writeBuffer(bParams, 0, params);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(trace);
        pass.setBindGroup(0, bgTrace);
        pass.dispatchWorkgroups(Math.ceil(width / 8), m);
        pass.end();
        const t = performance.now();
        device.queue.submit([enc.finish()]);
        await Promise.race([device.queue.onSubmittedWorkDone(), lostWatch]);
        if (lost.flag) return fail('GPU device lost', { dispatches, spp });
        const dt = performance.now() - t;
        traceMs += dt;
        dispatches++;
        maxDispatchMs = Math.max(maxDispatchMs, dt);
        if (slowest.length < 5 || dt > slowest[4][0]) {
          slowest.push([Math.round(dt), spp, s0, m, kk]);
          slowest.sort((a, b) => b[0] - a[0]);
          slowest.length = Math.min(slowest.length, 5);
        }
        perBlock = perBlock ? 0.7 * perBlock + 0.3 * dt / (m * kk) : dt / (m * kk);
        const f = Math.min(2, Math.max(0.5, aim / Math.max(dt, 0.1)));
        if (whole) {
          if (dt > dispatchMs && kk === 1) n = Math.max(1, Math.floor(nb * aim / dt));   // back to parts
          else k = Math.max(1, Math.min(16, Math.floor(kk * f)));
        } else {
          n = Math.max(1, Math.min(nb, Math.round(m * f)));
        }
        s0 += m;
        report(spp + kk * s0 / nb);
      }
      spp += kk;
    }
    const T3 = performance.now();

    // ---- resolve, read back, encode
    const ru = new Uint32Array(8), rf = new Float32Array(ru.buffer);
    ru[0] = width; ru[1] = height;
    rf[5] = lights.exposure; rf[6] = 1;
    device.queue.writeBuffer(bRes, 0, ru);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(resolve);
    pass.setBindGroup(0, bgResolve);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    enc.copyBufferToBuffer(bOut, 0, bRead, 0, pixels * 4);
    device.queue.submit([enc.finish()]);
    try {
      await bRead.mapAsync(GPUMapMode.READ);
    } catch (e) {
      if (lost.flag) return fail('GPU device lost', { dispatches, spp });
      throw e;
    }
    const rgba = new Uint8ClampedArray(bRead.getMappedRange().slice(0));
    bRead.unmap();
    const blob = await encodePNG(rgba, width, height);
    const T4 = performance.now();
    lastStats = {
      triangles: N, meshes: nMeshes, materials: nMats, bvhNodes: built.nodeCount, bvhDepth: built.depth, worker: built.worker,
      extractMs: Math.round(extractMs), bvhMs: Math.round(built.bvhMs), packMs: Math.round(built.packMs), buildMs: Math.round(T1 - T0),
      setupMs: Math.round(T2 - T1), traceMs: Math.round(traceMs),
      outputMs: Math.round(T4 - T3), totalMs: Math.round(T4 - T0), samples: spp, requested: samples, stopped,
      width, height, dispatches, maxDispatchMs: Math.round(maxDispatchMs), slowest, msPerSample: +(traceMs / Math.max(spp, 1)).toFixed(2),
      gpuBytes: nodes.byteLength + geo.byteLength + pixels * 24, adapter: gpu.info, limits: gpu.requested, mobile: MOBILE,
    };
    onProgress?.(1, spp);
    return blob;
  } finally {
    for (const b of bufs) b.destroy();
  }
}

// --------------------------------------------------------------------------- GPU
// Our own device (the viewer's when that fails): no optional features; the two
// storage size limits raised only as far as the adapter allows (big scenes on
// desktop GPUs; retried with no limits if refused); everything else at the WebGPU
// defaults.  Used: 1 bind group, 3 storage buffers and 1 uniform per stage (the
// resolve pass 2 + 1), 8x8 workgroups (64 invocations), no workgroup memory, at
// most ceil(width / 8) x ceil(height / 8) workgroups per dispatch.  All within
// the compatibility-mode minimums (4 storage buffers, 128 invocations).
function getGPU(viewer) {
  if (gpuPromise) return gpuPromise;
  const p = (async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    let device = null, own = true, info = '', requested = {};
    let adapter = null;
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    } catch (e) { adapter = null; }
    if (adapter) {
      const al = adapter.limits;
      const want = {};
      for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize']) {
        const v = Math.min(al[k] || 0, 1 << 30);
        if (v > 0) want[k] = v;
      }
      for (const req of [want, {}]) {
        try { device = await adapter.requestDevice({ requiredLimits: req }); requested = req; break; } catch (e) { device = null; }
      }
      const ai = adapter.info || {};
      info = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(' ');
    }
    if (!device) {
      // fall back on the viewer's own device when three.js runs on WebGPU
      device = viewer?.renderer?.backend?.device || null;
      own = false;
    }
    if (!device) return null;
    device.lost.then(() => { if (gpuPromise === p) gpuPromise = null; });
    const pipelines = compile(device);
    pipelines.catch(() => { if (gpuPromise === p) gpuPromise = null; });
    return { device, own, info, pipelines, requested };
  })().catch(() => null);
  gpuPromise = p;
  return p;
}

async function compile(device, code = traceWGSL) {
  const mod = (c, label) => device.createShaderModule({ code: c, label });
  const tm = mod(code, 'kigumi-pathtracer');
  const rm = mod(resolveWGSL, 'kigumi-resolve');
  for (const m of [tm, rm]) {
    const info = await m.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === 'error');
    if (errs.length) throw new Error('path tracer shader: ' + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
  }
  const [trace, resolve] = await Promise.all([
    device.createComputePipelineAsync({ layout: 'auto', compute: { module: tm, entryPoint: 'main' } }),
    device.createComputePipelineAsync({ layout: 'auto', compute: { module: rm, entryPoint: 'main' } }),
  ]);
  return { trace, resolve };
}

function isMobile() {
  if (typeof navigator === 'undefined') return false;
  if (navigator.userAgentData && typeof navigator.userAgentData.mobile === 'boolean') return navigator.userAgentData.mobile;
  const ua = navigator.userAgent || '';
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

// --------------------------------------------------------------------------- worker
// A module worker made from a blob (same origin as the page wherever this module
// is served from) that imports build.js; checked with a ping before any data is
// transferred to it, so a failure falls back on building here.
let workerPromise = null;
function getWorker() {
  if (workerPromise) return workerPromise;
  workerPromise = new Promise((resolve) => {
    try {
      const src = `import { buildScene } from ${JSON.stringify(new URL('./build.js', import.meta.url).href)};
self.onmessage = (e) => {
  if (e.data === 'ping') { self.postMessage('pong'); return; }
  try { const r = buildScene(e.data); self.postMessage({ ok: true, r }, [r.nodes.buffer, r.geo.buffer]); }
  catch (err) { self.postMessage({ ok: false, error: String((err && err.stack) || err) }); }
};`;
      const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
      const w = new Worker(url, { type: 'module' });
      const timer = setTimeout(() => { w.terminate(); resolve(null); }, 10000);
      w.onmessage = (e) => { if (e.data === 'pong') { clearTimeout(timer); w.onmessage = null; w.onerror = null; resolve(w); } };
      w.onerror = (e) => { e.preventDefault?.(); clearTimeout(timer); w.terminate(); resolve(null); };
      w.postMessage('ping');
    } catch (e) {
      resolve(null);
    }
  });
  return workerPromise;
}

async function buildOffThread(sc) {
  const w = await getWorker();
  if (!w) return { ...buildScene(sc), worker: false };
  return new Promise((resolve, reject) => {
    w.onmessage = (e) => { w.onmessage = w.onerror = null; if (e.data.ok) resolve({ ...e.data.r, worker: true }); else reject(new Error(e.data.error)); };
    w.onerror = (e) => { e.preventDefault?.(); workerPromise = null; w.terminate(); reject(new Error('path tracer: BVH worker failed')); };
    const msg = { pos: sc.pos, nrm: sc.nrm, ids: sc.ids, mats: sc.mats, xforms: sc.xforms, nMats: sc.nMats, nMeshes: sc.nMeshes };
    w.postMessage(msg, [sc.pos.buffer, sc.nrm.buffer, sc.ids.buffer]);
  });
}

// --------------------------------------------------------------------------- camera
function cameraFor(viewer, view, aspect) {
  const cam = viewer.camera;
  const ctl = viewer.controls;
  const keep = { aspect: cam.aspect, pos: cam.position.clone(), quat: cam.quaternion.clone(), target: ctl ? ctl.target.clone() : null };
  try {
    cam.aspect = aspect;
    cam.updateProjectionMatrix();
    if (view !== 'current' && typeof viewer.frame === 'function') viewer.frame(false, POSES[view] || POSES.hero);
    cam.updateMatrixWorld(true);
    const e = cam.matrixWorld.elements;
    const n = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((x) => x / l); };
    const tanV = Math.tan(cam.fov * Math.PI / 360) / (cam.zoom || 1);
    return {
      pos: [e[12], e[13], e[14]],
      right: n([e[0], e[1], e[2]]),
      up: n([e[4], e[5], e[6]]),
      fwd: n([-e[8], -e[9], -e[10]]),
      tanV, tanH: tanV * aspect,
    };
  } finally {
    cam.aspect = keep.aspect;
    cam.updateProjectionMatrix();
    cam.position.copy(keep.pos);
    cam.quaternion.copy(keep.quat);
    if (ctl && keep.target) { ctl.target.copy(keep.target); ctl.update?.(); }
    cam.updateMatrixWorld(true);
    viewer.requestRender?.();
  }
}

// --------------------------------------------------------------------------- studio
function lighting(viewer, st) {
  const exposure = viewer.renderer?.toneMappingExposure ?? 1;
  const n3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const kd = n3(viewer.keyDir ? [viewer.keyDir.x, viewer.keyDir.y, viewer.keyDir.z] : [-0.55, 1.0, 0.42]);
  // fill: opposite the key in azimuth, lower
  const az = Math.atan2(kd[0], kd[2]) + Math.PI;
  const fe = (st.fill.elevation ?? 26) * Math.PI / 180;
  const fd = [Math.sin(az) * Math.cos(fe), Math.sin(fe), Math.cos(az) * Math.cos(fe)];
  // background: linear working colour of the page paper
  const bgc = viewer.scene?.background?.isColor ? viewer.scene.background : null;
  const bg = bgc ? [bgc.r, bgc.g, bgc.b] : cssToLinear(viewer.bgCss || '#EAE4D8');
  const backdrop = inverseNeutral(bg, exposure);
  const unitLum = (c) => { const l = lum(c) || 1; return c.map((x) => x / l); };
  const tintBg = unitLum(bg);
  const keyC = unitLum(st.key.color), fillC = unitLum(st.fill.color);
  const cone = (deg) => Math.cos(deg * Math.PI / 180);
  const solid = (deg) => Math.PI * Math.sin(deg * Math.PI / 180) ** 2;    // irradiance per unit radiance
  const keyL = keyC.map((c) => c * st.key.E / solid(st.key.halfAngle));
  const fillL = fillC.map((c) => c * st.fill.E / solid(st.fill.halfAngle));
  const envH = tintBg.map((c) => c * st.env.horizon);
  const envZ = tintBg.map((c) => c * st.env.zenith);
  // floor irradiance with nothing in the way, and the albedo that makes the open
  // floor exactly as bright as the backdrop (so the horizon disappears)
  const E = [0, 1, 2].map((i) => st.key.E * keyC[i] * Math.max(kd[1], 0) + st.fill.E * fillC[i] * Math.max(fd[1], 0)
    + Math.PI * envH[i] + (envZ[i] - envH[i]) * 2 * Math.PI / 3);
  const floor = [0, 1, 2].map((i) => Math.min(0.98, Math.PI * backdrop[i] / E[i]));
  return {
    exposure, keyDir: kd, keyCos: cone(st.key.halfAngle), keyL, keyProb: st.key.E / (st.key.E + st.fill.E),
    fillDir: fd, fillCos: cone(st.fill.halfAngle), fillL, envH, envZ, backdrop, floor,
  };
}

function lum(c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }

function cssToLinear(css) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(css).trim());
  const v = m ? parseInt(m[1], 16) : 0xeae4d8;
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255].map((x) => {
    const c = x / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
}

// Khronos PBR Neutral and its inverse (fixed point), as in viewer.js
function neutral(v, exposure) {
  let c = v.map((x) => x * exposure);
  const x = Math.min(c[0], c[1], c[2]);
  const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  c = c.map((y) => y - offset);
  const peak = Math.max(c[0], c[1], c[2]);
  const start = 0.8 - 0.04;
  if (peak < start) return c;
  const d = 1 - start;
  const newPeak = 1 - d * d / (peak + d - start);
  c = c.map((y) => y * newPeak / peak);
  const g = 1 - 1 / (0.15 * (peak - newPeak) + 1);
  return c.map((y) => y * (1 - g) + newPeak * g);
}
function inverseNeutral(target, exposure) {
  let x = target.slice();
  for (let i = 0; i < 80; i++) {
    const y = neutral(x, exposure);
    x = x.map((xi, k) => Math.min(64, Math.max(1e-5, xi * Math.pow(target[k] / Math.max(y[k], 1e-6), 0.8))));
  }
  return x;
}

// --------------------------------------------------------------------------- helpers
function align(n, a) { return Math.ceil(n / a) * a; }
function gcd(a, b) { while (b) { [a, b] = [b, a % b]; } return a; }

async function encodePNG(rgba, w, h) {
  const img = new ImageData(rgba, w, h);
  if (typeof OffscreenCanvas !== 'undefined') {
    const c = new OffscreenCanvas(w, h);
    c.getContext('2d').putImageData(img, 0, 0);
    return c.convertToBlob({ type: 'image/png' });
  }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').putImageData(img, 0, 0);
  return new Promise((res) => c.toBlob(res, 'image/png'));
}
