// Stress test: add a high-poly torus knot (about `tris` triangles) to the model for
// one render, then remove it.  Reports the path tracer's stats.
export async function run({ tris = 1000000, width = 1600, height = 1100, samples = 256, name = 'stress.png', view = 'hero' } = {}) {
  const THREE = await import('three/webgpu');
  const k = window.kigumi, v = k.viewer;
  const radial = 250, tubular = Math.round(tris / (2 * radial));
  const geo = new THREE.TorusKnotGeometry(0.16, 0.035, tubular, radial, 2, 3);
  const mat = v.parts.values().next().value.meshes[0].material;      // a wood material
  const mesh = new THREE.Mesh(geo, Array.isArray(mat) ? mat[0] : mat);
  mesh.position.set(0, 0.68, 0);
  mesh.rotation.set(Math.PI / 2, 0, 0);
  v.model.add(mesh);
  v.model.updateMatrixWorld(true);
  try {
    const m = await import(`http://localhost:8791/render/pathtracer.js?t=${Date.now()}`);
    const t0 = performance.now();
    const blob = await m.renderPhoto({ viewer: v, view, width, height, samples });
    const ms = Math.round(performance.now() - t0);
    await fetch(`http://localhost:8791/save?name=${name}`, { method: 'POST', body: blob });
    return { ms, knotTris: geo.index.count / 3, stats: m.lastStats };
  } finally {
    v.model.remove(mesh);
    geo.dispose();
  }
}
