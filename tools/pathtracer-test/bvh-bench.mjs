// BVH build benchmark on real scenes dumped by dump.js:
//   node tools/pathtracer-test/bvh-bench.mjs out/scene-knot.f32 [runs]
// Prints build time, node count, depth, SAH cost and CPU ray-cast throughput.
import { readFileSync } from 'node:fs';
import { buildBVH, traceCPU } from '../../site/render/bvh.js';

const file = process.argv[2] || new URL('./out/scene-table.f32', import.meta.url).pathname;
const runs = Number(process.argv[3] || 3);
const buf = readFileSync(file);
const pos = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

function sahCost(bvh) {
  const f = bvh.nodes, u = new Uint32Array(f.buffer, f.byteOffset, f.length);
  const area = (o) => { const dx = f[o + 4] - f[o], dy = f[o + 5] - f[o + 1], dz = f[o + 6] - f[o + 2]; return dx < 0 ? 0 : dx * dy + dy * dz + dz * dx; };
  const b = bvh.bounds;
  const root = (b[3] - b[0]) * (b[4] - b[1]) + (b[4] - b[1]) * (b[5] - b[2]) + (b[5] - b[2]) * (b[3] - b[0]);
  let c = 1.2;
  for (let n = 0; n < bvh.nodeCount; n++) {
    for (let s = 0; s < 2; s++) {
      const o = 16 * n + 8 * s;
      const cnt = u[o + 7];
      c += (area(o) / root) * (cnt ? cnt : 1.2);
    }
  }
  return c;
}

function pack(order) {
  const T = new Float32Array(order.length * 12);
  for (let k = 0; k < order.length; k++) {
    const o = 9 * order[k], b = 12 * k;
    T[b] = pos[o]; T[b + 1] = pos[o + 1]; T[b + 2] = pos[o + 2];
    for (let a = 0; a < 3; a++) { T[b + 4 + a] = pos[o + 3 + a] - pos[o + a]; T[b + 8 + a] = pos[o + 6 + a] - pos[o + a]; }
  }
  return T;
}

let bvh;
const times = [];
for (let r = 0; r < runs; r++) {
  bvh = buildBVH(pos);
  times.push(bvh.ms);
}
const T = pack(bvh.order);
// rays from a hero-like camera at the scene's bounding box
const b = bvh.bounds;
const c = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
const R = Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]);
const eye = [c[0] + R * 0.9, c[1] + R * 0.6, c[2] + R * 1.1];
let hits = 0;
const t0 = performance.now();
const n = 20000;
let seed = 1;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
for (let i = 0; i < n; i++) {
  const p = [c[0] + (rnd() - 0.5) * R * 0.8, c[1] + (rnd() - 0.5) * R * 0.5, c[2] + (rnd() - 0.5) * R * 0.8];
  let d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
  const l = Math.hypot(...d); d = d.map((x) => x / l);
  if (traceCPU(bvh, T, eye, d).tri >= 0) hits++;
}
const ms = performance.now() - t0;
console.log(`${file.split('/').pop()}: ${pos.length / 9} tris, build ${times.map((t) => t.toFixed(0)).join('/')} ms, ` +
  `${bvh.nodeCount} nodes, depth ${bvh.depth}, SAH ${sahCost(bvh).toFixed(1)}, CPU ${(n / ms * 1000 / 1e3).toFixed(1)} krays/s (${hits} hits)`);
