// Traversal cost heatmap: BVH node visits per path (closest-hit + shadow rays),
// written as the pixel colour (via a patched kernel), plus a histogram summary.
//   run({ scale: 2000 })  -> out/heat.png: grey ramp, 1.0 = `scale` visits
export async function run({ scale = 2000, width = 800, height = 550, samples = 4, view = 'hero', name = 'heat.png', events } = {}) {
  const t = Date.now();
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${t}`);
  const { traceWGSL } = await import(`http://localhost:8791/render/kernel.js?t=${t}`);
  let code = traceWGSL;
  const sub = (a, b) => { if (!code.includes(a)) throw new Error('patch: ' + a); code = code.split(a).join(b); };
  sub('fn closest(o: vec3f, d: vec3f, tmax: f32) -> Hit {', 'var<private> gVisits: u32;\nfn closest(o: vec3f, d: vec3f, tmax: f32) -> Hit {');
  sub('for (var guard = 0u; guard < 200000u; guard++) {', 'for (var guard = 0u; guard < 200000u; guard++) {\n    gVisits++;');
  sub('let c = radiance(px, py, P.size.z + k, pix);', 'gVisits = 0u;\n    let c0 = radiance(px, py, P.size.z + k, pix);\n    let c = vec3f(f32(gVisits) / ' + scale.toFixed(1) + ') + c0 * 0.0;');
  // no tone mapping surprises: the resolve applies Neutral; fine for a ramp
  const blob = await m.renderPhoto({ viewer: window.kigumi.viewer, view, width, height, samples, _wgsl: code });
  await fetch(`http://localhost:8791/save?name=${name}`, { method: 'POST', body: blob });
  return { stats: m.lastStats };
}
