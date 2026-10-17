// Reference: the live (raster) view's photo for a view, uploaded to out/ref-<view>.png.
export async function run({ view = 'hero' } = {}) {
  const v = window.kigumi.viewer;
  const blob = await v.photo({ view, scale: 2 });
  await fetch(`http://localhost:8791/save?name=ref-${view}.png`, { method: 'POST', body: blob });
  return { size: blob.size };
}
