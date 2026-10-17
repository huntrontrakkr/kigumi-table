// Scene build for the path tracer: BVH (bvh.js) and the GPU buffers, from the
// arrays extractScene (scene.js) makes.  No DOM, no three.js: it runs in a worker
// (see pathtracer.js), or on the main thread when workers are unavailable.
//
//   buildScene({ pos, nrm, ids, mats, xforms, nMats, nMeshes })
//     -> { nodes: Float32Array, geo: Uint32Array, offShade, offMats, offXf,
//          nodeCount, depth, bvhMs, packMs }
//
// geo layout (vec4 units, see kernel.js): triangles [v0, e1, e2] (v0.w: material
// id), from offShade one vec4u per triangle [3 octahedral normals, ids], from
// offMats the material records, from offXf the mesh world -> local rows.

import { buildBVH, emptySlot } from './bvh.js';

const MAT_VEC4 = 8;
const XF_VEC4 = 3;

export function buildScene({ pos, nrm, ids, mats, xforms, nMats, nMeshes }) {
  const bvh = buildBVH(pos);
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const N = ids.length;
  const nT = Math.max(1, N);
  const offShade = 3 * nT, offMats = 4 * nT, offXf = offMats + nMats * MAT_VEC4;
  const geoVec4 = offXf + Math.max(1, nMeshes) * XF_VEC4;
  const geo = new Uint32Array(geoVec4 * 4);
  const geoF = new Float32Array(geo.buffer);
  const ord = bvh.order;
  for (let k = 0; k < N; k++) {
    const s = ord[k], o = 9 * s, b = 12 * k, h = 4 * (offShade + k);
    geoF[b] = pos[o]; geoF[b + 1] = pos[o + 1]; geoF[b + 2] = pos[o + 2];
    geoF[b + 4] = pos[o + 3] - pos[o]; geoF[b + 5] = pos[o + 4] - pos[o + 1]; geoF[b + 6] = pos[o + 5] - pos[o + 2];
    geoF[b + 8] = pos[o + 6] - pos[o]; geoF[b + 9] = pos[o + 7] - pos[o + 1]; geoF[b + 10] = pos[o + 8] - pos[o + 2];
    geo[b + 3] = ids[s] & 0xffff;
    geo[h] = oct(nrm[o], nrm[o + 1], nrm[o + 2]);
    geo[h + 1] = oct(nrm[o + 3], nrm[o + 4], nrm[o + 5]);
    geo[h + 2] = oct(nrm[o + 6], nrm[o + 7], nrm[o + 8]);
    geo[h + 3] = ids[s];
  }
  geoF.set(mats.subarray(0, nMats * MAT_VEC4 * 4), 4 * offMats);
  geoF.set(xforms.subarray(0, nMeshes * XF_VEC4 * 4), 4 * offXf);
  let nodes = bvh.nodes;
  if (!N) {
    nodes = new Float32Array(16);
    const u = new Uint32Array(nodes.buffer);
    emptySlot(nodes, u, 0);
    emptySlot(nodes, u, 8);
  } else {
    nodes = nodes.slice();                      // its own buffer, to transfer
  }
  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return { nodes, geo, offShade, offMats, offXf, nodeCount: bvh.nodeCount, depth: bvh.depth, bvhMs: bvh.ms, packMs: t1 - t0 };
}

// octahedral unit vector as two snorm16
export function oct(x, y, z) {
  const l = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  let u = x / l, v = y / l;
  if (z < 0) {
    const ou = u;
    u = (1 - Math.abs(v)) * (ou >= 0 ? 1 : -1);
    v = (1 - Math.abs(ou)) * (v >= 0 ? 1 : -1);
  }
  const q = (a) => Math.round(Math.max(-1, Math.min(1, a)) * 32767) & 0xffff;
  return (q(u) | (q(v) << 16)) >>> 0;
}
