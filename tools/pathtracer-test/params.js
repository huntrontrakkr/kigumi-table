// List the design parameters that take a choice (materials, layouts...).
export async function run() {
  const k = window.kigumi;
  const s = await k.engine.api('/api/schema');
  return (s.params || []).filter((p) => p.choices).map((p) => ({ name: p.name, def: p.default, choices: p.choices }))
    .concat([{ presets: Object.keys(s.presets || {}) }]);
}
