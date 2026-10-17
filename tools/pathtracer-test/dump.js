// Dump the extracted world-space triangles (Float32Array, 9 floats each) of the current
// model, optionally with the stress-test torus knot, to out/<name> for Node benchmarks.
export async function run({ knot = 0, name = 'scene.f32' } = {}) {
  const THREE = await import('three/webgpu');
  const v = window.kigumi.viewer;
  let mesh = null;
  if (knot) {
    const geo = new THREE.TorusKnotGeometry(0.16, 0.035, Math.round(knot / 500), 250, 2, 3);
    mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial());
    mesh.position.set(0, 0.68, 0);
    mesh.rotation.set(Math.PI / 2, 0, 0);
    v.model.add(mesh);
  }
  try {
    const { extractScene } = await import(`http://localhost:8791/render/scene.js?t=${Date.now()}`);
    const sc = extractScene(v.model);
    await fetch(`http://localhost:8791/save?name=${name}`, { method: 'POST', body: sc.pos });
    return { triangles: sc.count, ms: Math.round(sc.ms) };
  } finally {
    if (mesh) { v.model.remove(mesh); mesh.geometry.dispose(); }
  }
}
