// Close-up comparison: the live view and the path tracer from the same camera, aimed
// at the leg nearest the hero camera (or a given target).  Uploads <name>-ref.png
// and <name>-pt.png.
export async function run({ name = 'close', width = 1200, height = 800, samples = 64, dist = 0.55, az = 38, el = 18,
  aim = 0.75, params, glass, studio } = {}) {
  const k = window.kigumi;
  if (params) { k.setParams(params); await k.settled(); }
  if (glass != null && k.viewer.glassVisible !== glass) k.viewer.setGlassVisible(glass);
  const v = k.viewer;
  // the leg nearest the camera direction
  const a = az * Math.PI / 180, e = el * Math.PI / 180;
  const dir = [Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)];
  let best = null, bestS = -Infinity;
  for (const p of v.parts.values()) {
    if (p.kind !== 'leg' || !p.meshes.length) continue;
    const g = p.meshes[0].geometry;
    g.computeBoundingBox();
    const b = g.boundingBox.clone().applyMatrix4(p.meshes[0].matrixWorld);
    const c = [(b.min.x + b.max.x) / 2, b.min.y + (b.max.y - b.min.y) * aim, (b.min.z + b.max.z) / 2];
    const s = c[0] * dir[0] + c[2] * dir[2];
    if (s > bestS) { bestS = s; best = c; }
  }
  const target = best || [0, 0.3, 0];
  const cam = v.camera, r = v.renderer;
  const keep = { pos: cam.position.clone(), target: v.controls.target.clone(), ratio: r.getPixelRatio() };
  cam.position.set(target[0] + dir[0] * dist, target[1] + dir[1] * dist, target[2] + dir[2] * dist);
  v.controls.target.set(...target);
  v.controls.update();
  // raster reference at the photo size
  r.setPixelRatio(1);
  r.setSize(width, height, false);
  cam.aspect = width / height;
  cam.updateProjectionMatrix();
  v.key.shadow.needsUpdate = true;
  await r.renderAsync(v.scene, cam);
  const refBlob = await new Promise((res) => r.domElement.toBlob(res, 'image/png'));
  await fetch(`http://localhost:8791/save?name=${name}-ref.png`, { method: 'POST', body: refBlob });
  // path traced from the same camera
  const m = await import(`http://localhost:8791/render/pathtracer.js?t=${Date.now()}`);
  const opts = { viewer: v, view: 'current', width, height, samples };
  if (studio) opts.studio = { ...m.STUDIO, ...studio };
  const blob = await m.renderPhoto(opts);
  await fetch(`http://localhost:8791/save?name=${name}-pt.png`, { method: 'POST', body: blob });
  r.setPixelRatio(keep.ratio);
  cam.position.copy(keep.pos);
  v.controls.target.copy(keep.target);
  v.controls.update();
  v._resize();
  return { target, stats: m.lastStats };
}
