// Emulate a slower GPU: a patched kernel that traces `slow` paths per sample (and
// averages them), so a sample costs `slow` times as much.  Checks the time budget,
// the dispatch sizes and the progress reports.
const LOG = (o) => { try { fetch('http://localhost:8791/log', { method: 'POST', body: JSON.stringify(o) }); } catch (e) { /* ignore */ } };
let lastLog = 0;
const logProgress = (f, s) => { const t = performance.now(); if (t - lastLog > 500) { lastLog = t; LOG({ f: +f.toFixed(3), s, t: Math.round(t) }); } };
export async function run({ mode = 'spread', slow = 10, width, height, samples = 256, budgetMs, name = 'slow.png', view = 'hero', dispatchMs } = {}) {
  const t = Date.now();
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${t}`);
  const { traceWGSL } = await import(`http://localhost:8791/render/kernel.js?t=${t}`);
  const a = 'let c = radiance(px, py, P.size.z + k, pix);';
  if (!traceWGSL.includes(a)) return { error: 'patch failed' };
  // mode 'spread' (default): each thread also traces slow-1 other pixels spread over
  // the frame and folds them in at 1e-30 (as if the GPU had 1/slow of the cores);
  // mode 'same': the same pixel slow times (tail latency of the slowest pixels x slow)
  const code = mode === 'same'
    ? traceWGSL.replace(a, `var c = vec3f(0.0);
    for (var rr = 0u; rr < ${slow}u; rr++) { c += radiance(px, py, (P.size.z + k) * ${slow}u + rr, pix); }
    c *= ${(1 / slow).toFixed(8)};`)
    : traceWGSL.replace(a, `var c = radiance(px, py, P.size.z + k, pix);
    for (var rr = 1u; rr < ${slow}u; rr++) {
      let qy = (py + rr * (P.size.y / ${slow}u) + 7u * rr) % P.size.y;
      let qx = (px + 131u * rr) % P.size.x;
      c += 1e-30 * radiance(qx, qy, P.size.z + k, hash1(pix + rr));
    }`);
  const marks = [];
  const t0 = performance.now();
  const opts = { viewer: window.kigumi.viewer, view, samples, _wgsl: code, onProgress: (f, s) => { logProgress(f, s); marks.push([+f.toFixed(3), s, Math.round(performance.now() - t0)]); } };
  if (width) { opts.width = width; opts.height = height; }
  if (budgetMs) opts.budgetMs = budgetMs;
  if (dispatchMs) opts._dispatchMs = dispatchMs;
  const blob = await m.renderPhoto(opts);
  if (blob) await fetch(`http://localhost:8791/save?name=${name}`, { method: 'POST', body: blob });
  return { ms: Math.round(performance.now() - t0), stats: m.lastStats, progress: marks.filter((_, i) => i % Math.ceil(marks.length / 14) === 0 || i === marks.length - 1) };
}
