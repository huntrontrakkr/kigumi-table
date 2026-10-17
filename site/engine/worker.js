// The kigumi engine in a module worker: CPython (Pyodide, WebAssembly) running
// the kigumi package, Manifold's WebAssembly build for the solids, and -- when
// the browser has WebGPU and JS Promise Integration -- the WebGPU solver for
// the large linear systems.  Messages: {id, op, args} -> {id, ok, result} | {id, ok:false, error}
// plus {type:'progress', text, frac} while loading.

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
const PACKAGES = ['numpy', 'scipy', 'shapely', 'contourpy'];

let py = null;
let api = null;
const status = { gpu: null, jspi: false, packages: PACKAGES, pyodide: PYODIDE };

const progress = (text, frac) => postMessage({ type: 'progress', text, frac });

async function init(base) {
  progress('Loading Python (Pyodide, WebAssembly)…', 0.05);
  const { loadPyodide } = await import(PYODIDE + 'pyodide.mjs');
  py = await loadPyodide({ indexURL: PYODIDE });
  progress('Loading numpy, scipy, shapely…', 0.3);
  await py.loadPackage(PACKAGES, { messageCallback: () => {} });

  progress('Loading Manifold (WebAssembly)…', 0.7);
  const { initManifold } = await import(new URL('./manifold-bridge.js', import.meta.url).href);
  const bridge = await initManifold(new URL('../vendor/manifold/manifold.js', import.meta.url).href);
  py.registerJsModule('kigumi_manifold_js', bridge);

  progress('Loading the kigumi package…', 0.8);
  const zip = await (await fetch(new URL('py/kigumi.zip', base))).arrayBuffer();
  py.unpackArchive(zip, 'zip', { extractDir: '/home/pyodide' });
  await py.runPythonAsync('import sys; sys.path.insert(0, "/home/pyodide"); import kigumi.webapi');
  api = py.pyimport('kigumi.webapi');

  progress('Waking the GPU…', 0.9);
  await initGpu();
  progress('Ready', 1.0);
  return status;
}

async function initGpu() {
  // WebGPU for the solver, used from Python through JS Promise Integration
  // (a synchronous Python call waits on the GPU's promise).  Without either,
  // the solves stay on the CPU (SuperLU in WebAssembly).
  try {
    status.jspi = !!py.ffi?.can_run_sync?.() || (typeof WebAssembly.Suspending === 'function');
    const gpuMod = await import(new URL('./gpu/sparse.js', import.meta.url).href).catch(() => null);
    if (!gpuMod) { status.gpu = 'solver module missing'; return; }
    const gpu = await gpuMod.getGpu();
    if (!gpu) { status.gpu = 'no WebGPU adapter'; return; }
    status.gpu = gpu.info?.description || gpu.info?.vendor || 'WebGPU';
    if (!status.jspi) { status.gpu += ' (no JSPI: solves stay on the CPU)'; return; }
    const solve = async (n, k, indptr, indices, data, rhs) => {
      const r = await gpuMod.solveSPD(gpu, { n, k, indptr, indices, data, rhs, tol: 1e-9 });
      return r.converged ? r.x : null;
    };
    py.registerJsModule('kigumi_gpu_js', { solve });
    await py.runPythonAsync('from kigumi.gpu_web import install; install()');
    status.gpuSolver = true;
  } catch (e) {
    status.gpu = 'unavailable: ' + (e.message || e);
  }
}

const ops = {
  async init({ base }) { return init(base); },
  async schema() { return api.schema(); },
  async build({ body }) {
    const r = await api.build.callPromising(JSON.stringify(body));
    const [json, glb] = r.toJs();
    r.destroy?.();
    const bytes = glb.toJs ? glb.toJs() : glb;
    return { json, glb: bytes.buffer ? bytes : new Uint8Array(bytes) };
  },
  async checks({ id }) { return api.checks.callPromising(id); },
  async random({ body }) {
    const r = await api.random.callPromising(JSON.stringify(body));
    const [json, glb] = r.toJs();
    r.destroy?.();
    const bytes = glb.toJs ? glb.toJs() : glb;
    return { json, glb: bytes.buffer ? bytes : new Uint8Array(bytes) };
  },
  async export({ body }) {
    await py.loadPackage(['micropip'], { messageCallback: () => {} });
    await py.runPythonAsync(`
import micropip
try:
    import ezdxf  # noqa
except ImportError:
    await micropip.install("ezdxf")
`);
    const z = await api.export.callPromising(JSON.stringify(body));
    const bytes = z.toJs ? z.toJs() : z;
    return { zip: bytes };
  },
  async stats() {
    return py.runPython('import json; from kigumi.linalg import STATS; json.dumps(STATS)');
  },
};

onmessage = async (ev) => {
  const { id, op, args } = ev.data || {};
  if (!ops[op]) { postMessage({ id, ok: false, error: 'unknown op ' + op }); return; }
  try {
    const result = await ops[op](args || {});
    const transfer = [];
    for (const k of ['glb', 'zip']) if (result && result[k]?.buffer) transfer.push(result[k].buffer);
    postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    postMessage({ id, ok: false, error: String(e.message || e).slice(-4000) });
  }
};
