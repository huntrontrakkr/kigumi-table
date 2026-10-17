// The kigumi engine in a module worker: CPython (Pyodide, WebAssembly) running
// the kigumi package, Manifold's WebAssembly build for the solids, and -- opt-in
// with ?gpu-solver, when the browser has WebGPU and JS Promise Integration --
// the WebGPU solver for the large linear systems.  Messages: {id, op, args} -> {id, ok, result} | {id, ok:false, error}
// plus {type:'progress', text, frac} while loading.

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/';
const PACKAGES = ['numpy', 'scipy', 'shapely', 'contourpy'];

let py = null;
let api = null;
// JS Promise Integration lets synchronous Python wait on the GPU (Chrome, Edge);
// without it (Safari, iOS) Python is called directly and solves stay on the CPU
const JSPI = 'Suspending' in WebAssembly || 'Suspender' in WebAssembly;
const status = { gpu: null, jspi: JSPI, packages: PACKAGES, pyodide: PYODIDE };
const callPy = (fn, ...args) => (JSPI ? fn.callPromising(...args) : fn(...args));

const progress = (text, frac) => postMessage({ type: 'progress', text, frac });

async function init(base, gpuSolver = false) {
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

  if (gpuSolver) {
    progress('Waking the GPU…', 0.9);
    await initGpu();
  } else {
    status.gpu = 'solver off (SuperLU in WebAssembly is faster for these systems; ?gpu-solver turns it on)';
  }
  progress('Ready', 1.0);
  return status;
}

async function initGpu() {
  // WebGPU for the solver, used from Python through JS Promise Integration
  // (a synchronous Python call waits on the GPU's promise).  Without either,
  // the solves stay on the CPU (SuperLU in WebAssembly).
  try {
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
  async init({ base, gpuSolver }) { return init(base, gpuSolver); },
  async schema() { return api.schema(); },
  async build({ body }) {
    const r = await callPy(api.build, JSON.stringify(body));
    const [json, glb] = r.toJs();
    r.destroy?.();
    const bytes = glb.toJs ? glb.toJs() : glb;
    return { json, glb: bytes.buffer ? bytes : new Uint8Array(bytes) };
  },
  async checks({ id }) { return callPy(api.checks, id); },
  async random({ body }) {
    const r = await callPy(api.random, JSON.stringify(body));
    const [json, glb] = r.toJs();
    r.destroy?.();
    const bytes = glb.toJs ? glb.toJs() : glb;
    return { json, glb: bytes.buffer ? bytes : new Uint8Array(bytes) };
  },
  async export({ body }) {
    // DXF (ezdxf, pure Python, from PyPI) and the PDF sheets (matplotlib)
    await py.loadPackage(['micropip', 'matplotlib'], { messageCallback: () => {} });
    await py.runPythonAsync(`
import micropip
try:
    import ezdxf  # noqa
except ImportError:
    await micropip.install("ezdxf")
`);
    const z = await callPy(api.export, JSON.stringify(body));
    const bytes = z.toJs ? z.toJs() : z;
    return { zip: bytes };
  },
  // where a build spends its time (cProfile, top entries by cumulative time)
  async profile({ body }) {
    py.globals.set('_kg_body', JSON.stringify(body || {}));
    return py.runPython(`
import cProfile, io, pstats
from kigumi import webapi
_pr = cProfile.Profile()
_pr.enable()
webapi.build(_kg_body)
_pr.disable()
_s = io.StringIO()
pstats.Stats(_pr, stream=_s).sort_stats("cumulative").print_stats(40)
_s.getvalue()
`);
  },
  async stats() {
    const s = JSON.parse(py.runPython('import json; from kigumi.linalg import STATS; json.dumps(STATS)'));
    s.wasm_mb = Math.round((py._module?.HEAP8?.length || 0) / 2 ** 20);      // the Python heap
    return JSON.stringify(s);
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
