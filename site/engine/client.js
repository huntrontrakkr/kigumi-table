// Main-thread side of the engine: one module worker (engine/worker.js) running
// the kigumi package in Pyodide.  api(path, body) answers the same requests the
// local Python server does, so the configurator code stays as it was.

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let seq = 0;
const pending = new Map();
const listeners = new Set();
let lastProgress = { text: 'Starting…', frac: 0 };

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m && m.type === 'progress') {
    lastProgress = m;
    for (const f of listeners) f(m);
    return;
  }
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.ok) p.resolve(m.result);
  else p.reject(Object.assign(new Error(shortError(m.error)), { trace: m.error }));
};
worker.onerror = (e) => {
  for (const [, p] of pending) p.reject(new Error('The engine stopped: ' + (e.message || 'worker error')));
  pending.clear();
};

function shortError(s) {
  const lines = String(s || '').trim().split('\n');
  return lines[lines.length - 1] || 'engine error';
}

function call(op, args) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
}

// one engine call at a time: Python is single-threaded, and the configurator
// cancels stale builds itself
let chain = Promise.resolve();
function queued(op, args) {
  const run = chain.then(() => call(op, args));
  chain = run.catch(() => {});
  return run;
}

export const engine = {
  status: null,
  ready: null,
  onProgress(f) { listeners.add(f); f(lastProgress); return () => listeners.delete(f); },
  start() {
    if (!this.ready) {
      const base = new URL('../', import.meta.url).href;
      this.ready = call('init', { base }).then((s) => (this.status = s));
    }
    return this.ready;
  },
  async api(path, body) {
    await this.start();
    if (path === '/api/schema') return JSON.parse(await queued('schema'));
    if (path === '/api/build' || path === '/api/random') {
      const r = await queued(path === '/api/build' ? 'build' : 'random', { body });
      const res = JSON.parse(r.json);
      res.glb = URL.createObjectURL(new Blob([r.glb], { type: 'model/gltf-binary' }));
      return res;
    }
    if (path.startsWith('/api/checks/')) return JSON.parse(await queued('checks', { id: path.split('/').pop() }));
    if (path === '/api/export') {
      const r = await queued('export', { body });
      const name = 'kigumi-table.zip';
      return { files: [{ url: URL.createObjectURL(new Blob([r.zip], { type: 'application/zip' })), name }] };
    }
    if (path === '/api/stats') return JSON.parse(await queued('stats'));
    throw new Error('unknown request ' + path);
  },
};
