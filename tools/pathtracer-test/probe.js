// Probe: what the viewer's scene and the GPU look like (run through cdp.sh).
export async function run() {
  const v = window.kigumi.viewer;
  const out = { backend: v.backend, bg: v.bgCss, keyDir: v.keyDir.toArray(), fov: v.camera.fov, aspect: v.camera.aspect };
  const ad = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (ad) {
    out.adapter = { ...(ad.info ? { vendor: ad.info.vendor, arch: ad.info.architecture, desc: ad.info.description } : {}) };
    out.limits = {};
    for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBuffersPerShaderStage']) out.limits[k] = ad.limits[k];
    out.features = [...ad.features];
  }
  const meshes = [];
  let tris = 0;
  const mats = {};
  v.model.traverse((m) => {
    if (!m.isMesh) return;
    const g = m.geometry;
    const n = g.index ? g.index.count / 3 : g.attributes.position.count / 3;
    tris += n;
    const ml = Array.isArray(m.material) ? m.material : [m.material];
    ml.forEach((x) => { mats[x.name || x.type] = (mats[x.name || x.type] || 0) + 1; });
    let vis = true;
    for (let o = m; o; o = o.parent) if (!o.visible) vis = false;
    if (meshes.length < 12) meshes.push({ name: m.name, part: m.userData.partId, tris: n, groups: g.groups.length, idx: g.index ? g.index.array.constructor.name : null,
      det: m.matrixWorld.determinant().toFixed(3), mats: ml.map((x) => x.name), vis, ud: ml[0].userData?.kigumi ? Object.keys(ml[0].userData.kigumi.uniforms) : null });
  });
  out.tris = tris;
  out.nMeshes = 0;
  v.model.traverse((m) => { if (m.isMesh) out.nMeshes++; });
  out.mats = mats;
  out.meshes = meshes;
  out.stats = window.kigumi.stats;
  out.restBox = [v.restBox.min.toArray(), v.restBox.max.toArray()];
  return out;
}
