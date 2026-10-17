// Progressive GPU path tracer for "Render photo" (WebGPU compute).
//
//   import { renderPhoto } from './render/pathtracer.js';
//   const blob = await renderPhoto({ viewer, view: 'hero', width: 1600, height: 1100,
//                                    samples: 256, onProgress: (frac, spp) => {} });
//
// Takes the visible meshes of viewer.model, builds a BVH on the CPU (bvh.js), and
// path-traces them on the GPU (kernel.js) in a soft studio: a large warm key light
// from viewer.keyDir, a cool fill, an environment of the background colour and an
// infinite matte paper floor whose colour is solved so it fades into the backdrop
// like the live view.  Khronos PBR Neutral tone mapping, sRGB, PNG out.
// Resolves to null when WebGPU is unavailable.

import { buildBVH } from './bvh.js';
import { extractScene } from './scene.js';
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

const TARGET_MS = 110;            // GPU time per submission (keeps the browser responsive)
const MAX_EVENTS = 128;           // path vertices including glass interfaces (light entering a
                                  // sheet's edge runs along it by total internal reflection)

export let lastStats = null;

let gpuPromise = null;
let chain = Promise.resolve();

// --------------------------------------------------------------------------- API
export function renderPhoto(opts = {}) {
  // one render at a time
  const run = chain.then(() => render(opts));
  chain = run.catch(() => {});
  return run;
}

// Release the GPU device and pipelines (they are kept between renders).
export async function disposePathTracer() {
  const g = await (gpuPromise || Promise.resolve(null));
  gpuPromise = null;
  if (g && g.own) g.device.destroy();
}

async function render({ viewer, view = 'hero', width = 1600, height = 1100, samples = 256, onProgress, signal,
  studio = STUDIO } = {}) {
  if (!viewer || !viewer.model) throw new Error('renderPhoto: the viewer has no model');
  const T0 = performance.now();
  const gpu = await getGPU(viewer);
  if (!gpu) return null;
  const { device } = gpu;
  width = Math.max(8, Math.round(width));
  height = Math.max(8, Math.round(height));
  samples = Math.max(1, Math.round(samples));

  const cam = cameraFor(viewer, view, width / height);
  const sc = extractScene(viewer.model);
  const bvh = buildBVH(sc.pos);
  const T1 = performance.now();
  const lights = lighting(viewer, studio);

  // ---- GPU buffers
  const N = sc.count;
  const tri = new Float32Array(Math.max(1, N) * 12);
  const triU = new Uint32Array(tri.buffer);
  const shade = new Uint32Array(Math.max(1, N) * 4);
  const ord = bvh.order, P = sc.pos, Nr = sc.nrm, ids = sc.ids;
  for (let k = 0; k < N; k++) {
    const s = ord[k], o = 9 * s, b = 12 * k;
    tri[b] = P[o]; tri[b + 1] = P[o + 1]; tri[b + 2] = P[o + 2];
    tri[b + 4] = P[o + 3] - P[o]; tri[b + 5] = P[o + 4] - P[o + 1]; tri[b + 6] = P[o + 5] - P[o + 2];
    tri[b + 8] = P[o + 6] - P[o]; tri[b + 9] = P[o + 7] - P[o + 1]; tri[b + 10] = P[o + 8] - P[o + 2];
    triU[b + 3] = ids[s] & 0xffff;
    shade[4 * k] = oct(Nr[o], Nr[o + 1], Nr[o + 2]);
    shade[4 * k + 1] = oct(Nr[o + 3], Nr[o + 4], Nr[o + 5]);
    shade[4 * k + 2] = oct(Nr[o + 6], Nr[o + 7], Nr[o + 8]);
    shade[4 * k + 3] = ids[s];
  }
  const nodes = N ? bvh.nodes : emptyNodes();
  const bufs = [];
  const mk = (data, usage) => {
    const b = device.createBuffer({ size: Math.max(16, align(data.byteLength, 16)), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    bufs.push(b);
    return b;
  };
  const S = GPUBufferUsage.STORAGE;
  const bNodes = mk(nodes, S);
  const bTris = mk(tri, S);
  const bShade = mk(shade, S);
  const bMats = mk(sc.mats, S);
  const bXf = mk(sc.xforms, S);
  const pixels = width * height;
  const bAccum = device.createBuffer({ size: pixels * 16, usage: S | GPUBufferUsage.COPY_DST });
  bufs.push(bAccum);
  device.queue.writeBuffer(bAccum, 0, new Float32Array(pixels * 4));
  const bParams = device.createBuffer({ size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  bufs.push(bParams);
  const bOut = device.createBuffer({ size: pixels * 4, usage: S | GPUBufferUsage.COPY_SRC });
  const bRead = device.createBuffer({ size: pixels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bRes = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  bufs.push(bOut, bRead, bRes);

  try {
    const { trace, resolve } = await gpu.pipelines;
    const T2 = performance.now();
    device.pushErrorScope('validation');
    const bgTrace = device.createBindGroup({
      layout: trace.getBindGroupLayout(0),
      entries: [bParams, bAccum, bNodes, bTris, bShade, bMats, bXf].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

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

    let spp = 0;
    let k = 1, rows = Math.min(height, 128);
    let traceMs = 0;
    let chunks = 0;
    while (spp < samples) {
      const kk = Math.min(k, samples - spp);
      let passMs = 0;
      for (let y0 = 0; y0 < height; y0 += rows) {
        if (signal?.aborted) throw new DOMException('Render cancelled', 'AbortError');
        const r = Math.min(rows, height - y0);
        pu[2] = spp; pu[3] = kk; pu[4] = y0; pu[5] = r;
        device.queue.writeBuffer(bParams, 0, params);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(trace);
        pass.setBindGroup(0, bgTrace);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(r / 8));
        pass.end();
        const t = performance.now();
        device.queue.submit([enc.finish()]);
        if (chunks === 0) {
          const err = await device.popErrorScope();
          if (err) throw new Error('path tracer: ' + err.message);
        }
        await device.queue.onSubmittedWorkDone();
        const dt = performance.now() - t;
        passMs += dt;
        chunks++;
        onProgress?.(Math.min(1, (spp + kk * (y0 + r) / height) / samples), spp + (y0 + r >= height ? kk : 0));
      }
      spp += kk;
      traceMs += passMs;
      // next pass: aim for TARGET_MS per submission
      const perPixelSample = passMs / (kk * pixels);
      const budget = TARGET_MS / Math.max(perPixelSample, 1e-9);     // pixel-samples per submission
      if (budget >= pixels) { rows = height; k = Math.max(1, Math.min(32, Math.floor(budget / pixels))); }
      else { k = 1; rows = Math.max(8, Math.min(height, Math.floor(budget / width / 8) * 8)); }
    }
    const T3 = performance.now();

    // ---- resolve, read back, encode
    const ru = new Uint32Array(8), rf = new Float32Array(ru.buffer);
    ru[0] = width; ru[1] = height;
    rf[4] = 1 / samples; rf[5] = lights.exposure; rf[6] = 1;
    device.queue.writeBuffer(bRes, 0, ru);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(resolve);
    pass.setBindGroup(0, device.createBindGroup({
      layout: resolve.getBindGroupLayout(0),
      entries: [bRes, bAccum, bOut].map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(Math.ceil(pixels / 256));
    pass.end();
    enc.copyBufferToBuffer(bOut, 0, bRead, 0, pixels * 4);
    device.queue.submit([enc.finish()]);
    await bRead.mapAsync(GPUMapMode.READ);
    const rgba = new Uint8ClampedArray(bRead.getMappedRange().slice(0));
    bRead.unmap();
    const blob = await encodePNG(rgba, width, height);
    const T4 = performance.now();
    lastStats = {
      triangles: N, meshes: sc.nMeshes, materials: sc.nMats, bvhNodes: bvh.nodeCount, bvhDepth: bvh.depth,
      extractMs: Math.round(sc.ms), bvhMs: Math.round(bvh.ms), setupMs: Math.round(T2 - T1), traceMs: Math.round(traceMs),
      outputMs: Math.round(T4 - T3), totalMs: Math.round(T4 - T0), samples, width, height, chunks,
      msPerSample: +(traceMs / samples).toFixed(2),
      adapter: gpu.info,
    };
    return blob;
  } finally {
    for (const b of bufs) b.destroy();
  }
}

// --------------------------------------------------------------------------- GPU
function getGPU(viewer) {
  if (gpuPromise) return gpuPromise;
  const p = (async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    let device = null, own = true, info = '';
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        const lim = adapter.limits;
        device = await adapter.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 1 << 30),
            maxBufferSize: Math.min(lim.maxBufferSize, 1 << 30),
            maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 8),
          },
        });
        const ai = adapter.info || {};
        info = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(' ');
      }
    } catch (e) {
      device = null;
    }
    if (!device) {
      // fall back on the viewer's own device when three.js runs on WebGPU
      device = viewer?.renderer?.backend?.device || null;
      own = false;
    }
    if (!device) return null;
    if (own) device.lost.then(() => { if (gpuPromise === p) gpuPromise = null; });
    const pipelines = compile(device);
    pipelines.catch(() => { if (gpuPromise === p) gpuPromise = null; });
    return { device, own, info, pipelines };
  })().catch(() => null);
  gpuPromise = p;
  return p;
}

async function compile(device) {
  const mod = (code, label) => device.createShaderModule({ code, label });
  const tm = mod(traceWGSL, 'kigumi-pathtracer');
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

function emptyNodes() {
  const f = new Float32Array(16);
  f.set([Infinity, Infinity, Infinity], 0); f.set([-Infinity, -Infinity, -Infinity], 4);
  f.set([Infinity, Infinity, Infinity], 8); f.set([-Infinity, -Infinity, -Infinity], 12);
  return f;
}

// octahedral unit vector as two snorm16
function oct(x, y, z) {
  const l = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  let u = x / l, v = y / l;
  if (z < 0) {
    const ou = u;
    u = (1 - Math.abs(v)) * (ou >= 0 ? 1 : -1);
    v = (1 - Math.abs(ou)) * (v >= 0 ? 1 : -1);
  }
  const q = (a) => Math.round(Math.max(-1, Math.min(1, a)) * 32767) & 0xffff;
  return (q(u) | (q(v) << 16)) >>> 0;
}

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
