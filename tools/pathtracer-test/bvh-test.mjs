// BVH builder test under Node: closest hits against brute force, build time at scale.
//   node tools/pathtracer-test/bvh-test.mjs [triangles]
import { buildBVH, traceCPU } from '../../site/render/bvh.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// a lattice-like test scene: long thin boxes (strips) plus a cloud of small triangles
function scene(nTarget) {
  const r = rng(7);
  const tris = [];
  const box = (x0, y0, z0, x1, y1, z1) => {
    const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    const f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]];
    for (const [a, b, c] of f) tris.push(...v[a], ...v[b], ...v[c]);
  };
  for (let i = 0; i < 40; i++) {
    const x = -0.5 + i / 39;
    box(x - 0.006, 0.4, -0.5, x + 0.006, 0.44, 0.5);
    box(-0.5, 0.4, x - 0.006, 0.5, 0.44, x + 0.006);
  }
  while (tris.length / 9 < nTarget) {
    const cx = r() - 0.5, cy = r() * 0.4, cz = r() - 0.5, s = 0.004;
    tris.push(cx, cy, cz, cx + s * (r() - 0.5), cy + s * r(), cz + s * (r() - 0.5), cx + s * (r() - 0.5), cy + s * (r() - 0.5), cz + s * r());
  }
  return new Float32Array(tris);
}

function pack(pos, order) {
  const T = new Float32Array(order.length * 12);
  for (let k = 0; k < order.length; k++) {
    const o = 9 * order[k], b = 12 * k;
    T[b] = pos[o]; T[b + 1] = pos[o + 1]; T[b + 2] = pos[o + 2];
    for (let a = 0; a < 3; a++) { T[b + 4 + a] = pos[o + 3 + a] - pos[o + a]; T[b + 8 + a] = pos[o + 6 + a] - pos[o + a]; }
  }
  return T;
}

function brute(T, n, o, d) {
  let best = Infinity, tri = -1;
  for (let k = 0; k < n; k++) {
    const b = 12 * k;
    const e1 = [T[b + 4], T[b + 5], T[b + 6]], e2 = [T[b + 8], T[b + 9], T[b + 10]];
    const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
    const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (Math.abs(det) < 1e-20) continue;
    const id = 1 / det;
    const s = [o[0] - T[b], o[1] - T[b + 1], o[2] - T[b + 2]];
    const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * id;
    if (u < 0 || u > 1) continue;
    const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
    const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * id;
    if (v < 0 || u + v > 1) continue;
    const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * id;
    if (t > 0 && t < best) { best = t; tri = k; }
  }
  return { t: best, tri };
}

const big = Number(process.argv[2] || 1000000);
// correctness on a small scene
{
  const pos = scene(20000);
  const bvh = buildBVH(pos);
  const T = pack(pos, bvh.order);
  const r = rng(3);
  let bad = 0, hits = 0;
  for (let i = 0; i < 2000; i++) {
    const o = [r() * 2 - 1, r() * 0.8, r() * 2 - 1];
    let d = [r() - 0.5, r() - 0.5, r() - 0.5];
    const l = Math.hypot(...d); d = d.map((x) => x / l);
    const a = traceCPU(bvh, T, o, d), b = brute(T, pos.length / 9, o, d);
    if (b.tri >= 0) hits++;
    if (Math.abs(a.t - b.t) > 1e-6 * Math.max(1, b.t) && !(a.t === Infinity && b.t === Infinity)) bad++;
  }
  console.log(`small: ${pos.length / 9} tris, ${bvh.nodeCount} nodes, ${bvh.leafCount} leaves, depth ${bvh.depth}, ${bvh.ms.toFixed(0)} ms; rays ${hits} hits, ${bad} mismatches`);
  if (bad) process.exitCode = 1;
}
// build time at scale
{
  const pos = scene(big);
  const bvh = buildBVH(pos);
  console.log(`big: ${pos.length / 9} tris, ${bvh.nodeCount} nodes, ${bvh.leafCount} leaves, depth ${bvh.depth}, build ${bvh.ms.toFixed(0)} ms`);
}
