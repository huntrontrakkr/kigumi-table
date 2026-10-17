// Render photos with the path tracer in the running app and upload them to out/.
// Run through cdp.sh; the module is loaded fresh from serve.py (never cached).
//   run({ view, width, height, samples, name, studio, params, glass, explode })
//   run({ batch: [ {...}, {...} ] })        several renders in one page session
export async function run(cfg = {}) {
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${Date.now()}`);
  if (cfg.batch) {
    const out = [];
    for (const c of cfg.batch) out.push(await one(m, { ...cfg, batch: undefined, ...c }));
    return out;
  }
  return one(m, cfg);
}

async function one(m, { view = 'hero', width = 800, height = 550, samples = 16, name, studio, params, glass, explode, progress } = {}) {
  const k = window.kigumi;
  if (params) { k.setParams(params); await k.settled(); }
  if (glass != null && k.viewer.glassVisible !== glass) k.viewer.setGlassVisible(glass);
  if (explode != null) { k.setExplode(explode); await k.settled(); }
  const opts = { viewer: k.viewer, view, width, height, samples };
  if (studio) {
    opts.studio = { ...m.STUDIO };
    for (const [key, v] of Object.entries(studio)) opts.studio[key] = typeof v === 'object' && !Array.isArray(v) ? { ...m.STUDIO[key], ...v } : v;
  }
  const marks = [];
  const t0 = performance.now();
  opts.onProgress = (f, s) => { if (marks.length < 400) marks.push([+f.toFixed(3), s, Math.round(performance.now() - t0)]); };
  const blob = await m.renderPhoto(opts);
  const ms = Math.round(performance.now() - t0);
  if (!blob) return { error: 'no WebGPU' };
  const file = name || `pt-${view}-${width}x${height}-${samples}.png`;
  await fetch(`http://localhost:8791/save?name=${encodeURIComponent(file)}`, { method: 'POST', body: blob });
  const res = { file, ms, bytes: blob.size, stats: m.lastStats };
  if (progress) res.progress = marks.filter((_, i) => i % Math.ceil(marks.length / 12) === 0 || i === marks.length - 1);
  return res;
}
