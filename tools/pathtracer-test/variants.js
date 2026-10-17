// Kernel experiments: render the same frame with patched WGSL and compare trace times.
//   run({ variants: ['base', 'pcg', 'base'], samples: 128 })
const PATCHES = {
  base: [],
  // plain hashed random numbers instead of Owen-scrambled Sobol points
  pcg: [[/fn rand2\(si: u32, seed: u32\) -> vec2f \{[^}]*\}/, `fn rand2(si: u32, seed: u32) -> vec2f {
  let h = hash1(seed ^ hash1(si + 0x632be5abu));
  let h2 = hash1(h ^ 0x85ebca6bu);
  return vec2f(vec2u(h, h2) >> vec2u(8u)) * (1.0 / 16777216.0);
}`], [/fn rand1\(si: u32, seed: u32\) -> f32 \{[^}]*\}/, `fn rand1(si: u32, seed: u32) -> f32 {
  return f32(hash1(seed ^ hash1(si + 0x632be5abu)) >> 8u) * (1.0 / 16777216.0);
}`]],
};

export async function run({ variants = ['base', 'pcg', 'base'], samples = 128, width = 1600, height = 1100, view = 'hero', save = false } = {}) {
  const t = Date.now();
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${t}`);
  const { traceWGSL } = await import(`http://localhost:8791/render/kernel.js?t=${t}`);
  const out = [];
  for (const name of variants) {
    let code = traceWGSL;
    for (const [a, b] of PATCHES[name]) {
      const before = code;
      code = code.replace(a, b);
      if (code === before) return { error: `patch ${name} did not apply: ${a}` };
    }
    const blob = await m.renderPhoto({ viewer: window.kigumi.viewer, view, width, height, samples, _wgsl: code });
    if (save) await fetch(`http://localhost:8791/save?name=var-${name}-${samples}.png`, { method: 'POST', body: blob });
    out.push({ name, traceMs: m.lastStats.traceMs, msPerSample: m.lastStats.msPerSample });
  }
  return out;
}
