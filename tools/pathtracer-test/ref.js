// Reference: the live (raster) view rendered at the photo's size and framing,
// uploaded to out/ref-<view>.png, for side-by-side comparison with the path tracer.
export async function run({ view = 'hero', width = 1600, height = 1100, params, glass, name } = {}) {
  const k = window.kigumi;
  if (params) { k.setParams(params); await k.settled(); }
  if (glass === false && k.viewer.glassVisible) k.viewer.setGlassVisible(false);
  const v = k.viewer;
  const poses = { hero: { azimuth: 38, elevation: 24 }, top: { azimuth: 0, elevation: 89 },
    low: { azimuth: 25, elevation: 6 }, side: { azimuth: 90, elevation: 2 }, front: { azimuth: 0, elevation: 14 } };
  const r = v.renderer, cam = v.camera;
  const keep = { pos: cam.position.clone(), target: v.controls.target.clone(), ratio: r.getPixelRatio() };
  r.setPixelRatio(1);
  r.setSize(width, height, false);
  cam.aspect = width / height;
  cam.updateProjectionMatrix();
  v.frame(false, poses[view] || poses.hero);
  v.key.shadow.needsUpdate = true;
  await r.renderAsync(v.scene, cam);
  const blob = await new Promise((res) => r.domElement.toBlob(res, 'image/png'));
  r.setPixelRatio(keep.ratio);
  cam.position.copy(keep.pos);
  v.controls.target.copy(keep.target);
  v.controls.update();
  v._resize();
  const file = name || `ref-${view}.png`;
  await fetch(`http://localhost:8791/save?name=${file}`, { method: 'POST', body: blob });
  return { size: blob.size, file };
}
