// Edge cases: caps (null + reason), abort, device loss mid-render, recovery.
export async function run() {
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${Date.now()}`);
  const viewer = window.kigumi.viewer;
  const out = {};
  const small = { viewer, width: 400, height: 275, samples: 16 };

  let b = await m.renderPhoto({ ...small, maxTriangles: 1000 });
  out.triCap = { blob: !!b, stats: m.lastStats };

  b = await m.renderPhoto({ ...small, width: 6000, height: 4000 });
  out.pixelCap = { blob: !!b, stats: m.lastStats };

  const ac = new AbortController();
  setTimeout(() => ac.abort(), 1500);
  try {
    b = await m.renderPhoto({ viewer, width: 1600, height: 1100, samples: 4096, budgetMs: 60000, signal: ac.signal });
    out.abort = { blob: !!b, note: 'not aborted?' };
  } catch (e) {
    out.abort = { error: e.name };
  }

  // device loss: destroy the device while a long render runs
  const p = m.renderPhoto({ viewer, width: 1600, height: 1100, samples: 4096, budgetMs: 60000 });
  setTimeout(() => m.disposePathTracer(), 1500);
  const t0 = performance.now();
  try {
    b = await p;
    out.lost = { blob: !!b, ms: Math.round(performance.now() - t0), stats: m.lastStats };
  } catch (e) {
    out.lost = { error: String(e), ms: Math.round(performance.now() - t0) };
  }

  // recovery: a new device
  b = await m.renderPhoto(small);
  out.after = { blob: !!b, samples: m.lastStats?.samples, reason: m.lastStats?.reason };

  // budget: a short one
  b = await m.renderPhoto({ viewer, width: 1600, height: 1100, samples: 4096, budgetMs: 3000 });
  out.budget = { blob: !!b, totalMs: m.lastStats?.totalMs, samples: m.lastStats?.samples, stopped: m.lastStats?.stopped };
  return out;
}
