// Render a photo with the path tracer in the running app and upload it to out/.
// Run through cdp.sh; the module is loaded fresh from serve.py (never cached).
export async function run({ view = 'hero', width = 800, height = 550, samples = 16, name, studio, params, glass, explode } = {}) {
  const k = window.kigumi;
  if (params) { k.setParams(params); await k.settled(); }
  if (glass === false && k.viewer.glassVisible) k.viewer.setGlassVisible(false);
  if (explode != null) { k.setExplode(explode); await k.settled(); }
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${Date.now()}`);
  const opts = { viewer: k.viewer, view, width, height, samples };
  if (studio) opts.studio = { ...m.STUDIO, ...studio };
  const marks = [];
  const t0 = performance.now();
  opts.onProgress = (f, s) => { if (marks.length < 400) marks.push([+f.toFixed(3), s, Math.round(performance.now() - t0)]); };
  const blob = await m.renderPhoto(opts);
  const ms = Math.round(performance.now() - t0);
  if (!blob) return { error: 'no WebGPU' };
  const file = name || `pt-${view}-${width}x${height}-${samples}.png`;
  await fetch(`http://localhost:8791/save?name=${encodeURIComponent(file)}`, { method: 'POST', body: blob });
  return { ms, bytes: blob.size, file, stats: m.lastStats, progress: marks.filter((_, i) => i % Math.ceil(marks.length / 12) === 0 || i === marks.length - 1) };
}
