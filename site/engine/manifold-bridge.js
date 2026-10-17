// Manifold (the WebAssembly build of github.com/elalish/manifold) for the Python
// shim kigumi/_m3d_web.py, which runs in Pyodide.  Python hands over typed
// arrays; every function here takes and returns plain values or Manifold /
// CrossSection objects, which Python wraps and frees (delete()) when done.

let wasm = null;

export async function initManifold(moduleUrl) {
  if (wasm) return bridge;
  const Module = (await import(moduleUrl)).default;
  wasm = await Module();
  wasm.setup();
  return bridge;
}

function ringsFrom(flat, lengths) {
  // flat Float64Array [x0, y0, x1, y1, ...], lengths Int32Array of point counts
  const rings = [];
  let o = 0;
  for (let i = 0; i < lengths.length; i++) {
    const n = lengths[i];
    const r = new Array(n);
    for (let j = 0; j < n; j++) r[j] = [flat[2 * (o + j)], flat[2 * (o + j) + 1]];
    rings.push(r);
    o += n;
  }
  return rings;
}

export const bridge = {
  mesh(vertProperties, triVerts) {
    const m = new wasm.Mesh({ numProp: 3, vertProperties, triVerts });
    m.merge();
    return new wasm.Manifold(m);
  },
  crossSection(flat, lengths, fillRule) {
    return new wasm.CrossSection(ringsFrom(flat, lengths), fillRule || 'Positive');
  },
  extrude(cs, height) { return cs.extrude(height); },
  revolve(cs, segments, degrees) { return wasm.Manifold.revolve(cs, segments, degrees ?? 360); },
  cylinder(h, r0, r1, segments, center) { return wasm.Manifold.cylinder(h, r0, r1, segments, !!center); },
  sphere(r, segments) { return wasm.Manifold.sphere(r, segments); },
  translate(m, x, y, z) { return m.translate([x, y, z]); },
  transform(m, mat16) { return m.transform(Array.from(mat16)); },
  add(a, b) { return a.add(b); },
  subtract(a, b) { return a.subtract(b); },
  intersect(a, b) { return a.intersect(b); },
  union(list) { return wasm.Manifold.union(list); },
  volume(m) { return m.volume(); },
  isEmpty(m) { return m.isEmpty(); },
  box(m) {
    const b = m.boundingBox();
    return new Float64Array([b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]]);
  },
  getMesh(m) {
    const g = m.getMesh();
    const np = g.numProp;
    let v = g.vertProperties;
    if (np !== 3) {                             // keep positions only
      const n = v.length / np;
      const w = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) { w[3 * i] = v[np * i]; w[3 * i + 1] = v[np * i + 1]; w[3 * i + 2] = v[np * i + 2]; }
      v = w;
    }
    return { vert: new Float32Array(v), tri: new Uint32Array(g.triVerts) };
  },
  rayCast(m, ox, oy, oz, ex, ey, ez) {
    // [distance, x, y, z] per hit, flattened
    const hits = m.rayCast([ox, oy, oz], [ex, ey, ez]);
    const out = new Float64Array(4 * hits.length);
    hits.forEach((h, i) => { out[4 * i] = h.distance; out[4 * i + 1] = h.position[0]; out[4 * i + 2] = h.position[1]; out[4 * i + 3] = h.position[2]; });
    return out;
  },
  slice(m, z) { return m.slice(z); },
  csArea(cs) { return cs.area(); },
  csPolygons(cs) {
    // flattened rings: [count, x0, y0, ..., count, ...]
    const polys = cs.toPolygons();
    const out = [];
    for (const r of polys) { out.push(r.length); for (const p of r) out.push(p[0], p[1]); }
    return new Float64Array(out);
  },
  free(o) { try { o.delete(); } catch (e) { /* already freed */ } },
};
