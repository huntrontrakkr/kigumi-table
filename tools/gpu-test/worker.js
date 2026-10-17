// Dedicated module worker running the test plan (the app calls the solver from a worker too).
import { runAll } from './cases.js';

onmessage = async (ev) => {
  try {
    const out = await runAll(ev.data.params || {},
      (result) => postMessage({ type: 'case', result }),
      (text) => postMessage({ type: 'log', text }));
    postMessage({ type: 'done', out });
  } catch (e) {
    postMessage({ type: 'done', out: { ok: false, error: String((e && e.stack) || e), results: [] } });
  }
};
