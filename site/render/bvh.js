// Bounding volume hierarchy for the path tracer: a binned-SAH binary BVH built on
// the CPU from world-space triangles, flattened for stackful GPU traversal.
//
// GPU layout ("child boxes in the parent", Aila & Laine 2009): one interior node
// is four vec4s, the boxes of both children and how to reach them:
//   [c0.min.xyz, c0.ref] [c0.max.xyz, c0.count] [c1.min.xyz, c1.ref] [c1.max.xyz, c1.count]
// count == 0: ref is an interior node index; count > 0: ref is the first triangle
// of a leaf in the reordered triangle list.  The root is node 0; nodes are in
// depth-first order.  Visiting a node tests both child boxes, one 64-byte fetch.
//
// Pure JavaScript (no three.js), so it also runs under Node for testing.

const BINS = 16;
const MAX_LEAF = 8;          // SAH may stop splitting at this many triangles or fewer
const MAX_DEPTH = 60;        // the GPU traversal stack is 64 deep
const C_TRAV = 1.2;          // cost of a node visit relative to one triangle test

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function area(x0, y0, z0, x1, y1, z1) {
  const dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
  return dx < 0 ? 0 : dx * dy + dy * dz + dz * dx;
}

// pos: Float32Array, 9 floats per triangle (world space).
// Returns { nodes: Float32Array (16 per interior node), order: Uint32Array (BVH order
// -> original triangle index), nodeCount, leafCount, depth, bounds, ms }.
export function buildBVH(pos) {
  const t0 = now();
  const N = Math.floor(pos.length / 9);
  // per-triangle boxes and centroids (box centres)
  const bb = new Float32Array(N * 6);
  const cen = new Float32Array(N * 3);
  let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity, rx1 = -Infinity, ry1 = -Infinity, rz1 = -Infinity;
  for (let i = 0; i < N; i++) {
    const o = 9 * i;
    const ax = pos[o], ay = pos[o + 1], az = pos[o + 2];
    const bx = pos[o + 3], by = pos[o + 4], bz = pos[o + 5];
    const cx = pos[o + 6], cy = pos[o + 7], cz = pos[o + 8];
    const x0 = Math.min(ax, bx, cx), y0 = Math.min(ay, by, cy), z0 = Math.min(az, bz, cz);
    const x1 = Math.max(ax, bx, cx), y1 = Math.max(ay, by, cy), z1 = Math.max(az, bz, cz);
    const b = 6 * i;
    bb[b] = x0; bb[b + 1] = y0; bb[b + 2] = z0; bb[b + 3] = x1; bb[b + 4] = y1; bb[b + 5] = z1;
    cen[3 * i] = 0.5 * (x0 + x1); cen[3 * i + 1] = 0.5 * (y0 + y1); cen[3 * i + 2] = 0.5 * (z0 + z1);
    if (x0 < rx0) rx0 = x0; if (y0 < ry0) ry0 = y0; if (z0 < rz0) rz0 = z0;
    if (x1 > rx1) rx1 = x1; if (y1 > ry1) ry1 = y1; if (z1 > rz1) rz1 = z1;
  }
  const idx = new Uint32Array(N);
  for (let i = 0; i < N; i++) idx[i] = i;

  // build nodes: box, triangle range [start, end), first child (children are adjacent)
  const cap = Math.max(1, 2 * N);
  const nBox = new Float32Array(cap * 6);
  const nStart = new Uint32Array(cap);
  const nEnd = new Uint32Array(cap);
  const nLeft = new Int32Array(cap).fill(-1);      // -1: leaf
  const nDepth = new Uint8Array(cap);
  let count = 1;
  nBox[0] = rx0; nBox[1] = ry0; nBox[2] = rz0; nBox[3] = rx1; nBox[4] = ry1; nBox[5] = rz1;
  nStart[0] = 0; nEnd[0] = N;

  // scratch
  const binCnt = new Uint32Array(3 * BINS);
  const binBox = new Float64Array(3 * BINS * 6);
  const rArea = new Float64Array(BINS);
  const rCnt = new Uint32Array(BINS);
  const rBox = new Float64Array(BINS * 6);
  const lb = new Float32Array(6), rb = new Float32Array(6);
  const ext = new Float64Array(3), cmin = new Float64Array(3), sc = new Float64Array(3);
  const stack = [0];
  let maxDepth = 0, leaves = 0;

  while (stack.length) {
    const ni = stack.pop();
    const s = nStart[ni], e = nEnd[ni], n = e - s;
    const depth = nDepth[ni];
    if (depth > maxDepth) maxDepth = depth;
    if (n <= 1 || depth >= MAX_DEPTH) { leaves++; continue; }
    // centroid bounds
    let c0x = Infinity, c0y = Infinity, c0z = Infinity, c1x = -Infinity, c1y = -Infinity, c1z = -Infinity;
    for (let k = s; k < e; k++) {
      const i = 3 * idx[k];
      const x = cen[i], y = cen[i + 1], z = cen[i + 2];
      if (x < c0x) c0x = x; if (x > c1x) c1x = x;
      if (y < c0y) c0y = y; if (y > c1y) c1y = y;
      if (z < c0z) c0z = z; if (z > c1z) c1z = z;
    }
    ext[0] = c1x - c0x; ext[1] = c1y - c0y; ext[2] = c1z - c0z;
    cmin[0] = c0x; cmin[1] = c0y; cmin[2] = c0z;
    const pArea = area(nBox[6 * ni], nBox[6 * ni + 1], nBox[6 * ni + 2], nBox[6 * ni + 3], nBox[6 * ni + 4], nBox[6 * ni + 5]);

    let bestAxis = -1, bestBin = 0, bestCost = Infinity;
    // small nodes get fewer bins: resetting and sweeping bins is per-node work
    const nb = n < BINS ? Math.max(4, n) : BINS;
    if (Math.max(ext[0], ext[1], ext[2]) > 1e-12) {
      for (let a = 0; a < 3; a++) {
        for (let j = 0; j < nb; j++) {
          const b = a * BINS + j, o = 6 * b;
          binCnt[b] = 0;
          binBox[o] = binBox[o + 1] = binBox[o + 2] = Infinity;
          binBox[o + 3] = binBox[o + 4] = binBox[o + 5] = -Infinity;
        }
      }
      for (let a = 0; a < 3; a++) sc[a] = ext[a] > 1e-12 ? (nb * (1 - 1e-6)) / ext[a] : 0;
      for (let k = s; k < e; k++) {
        const t = idx[k];
        const ci = 3 * t, bi = 6 * t;
        const x0 = bb[bi], y0 = bb[bi + 1], z0 = bb[bi + 2], x1 = bb[bi + 3], y1 = bb[bi + 4], z1 = bb[bi + 5];
        for (let a = 0; a < 3; a++) {
          if (sc[a] === 0) continue;
          let q = ((cen[ci + a] - cmin[a]) * sc[a]) | 0;
          if (q > nb - 1) q = nb - 1;
          const b = a * BINS + q;
          binCnt[b]++;
          const o = 6 * b;
          if (x0 < binBox[o]) binBox[o] = x0;
          if (y0 < binBox[o + 1]) binBox[o + 1] = y0;
          if (z0 < binBox[o + 2]) binBox[o + 2] = z0;
          if (x1 > binBox[o + 3]) binBox[o + 3] = x1;
          if (y1 > binBox[o + 4]) binBox[o + 4] = y1;
          if (z1 > binBox[o + 5]) binBox[o + 5] = z1;
        }
      }
      for (let a = 0; a < 3; a++) {
        if (sc[a] === 0) continue;
        // sweep from the right: area and count of bins [j, BINS)
        let X0 = Infinity, Y0 = Infinity, Z0 = Infinity, X1 = -Infinity, Y1 = -Infinity, Z1 = -Infinity, cnt = 0;
        for (let j = nb - 1; j > 0; j--) {
          const o = 6 * (a * BINS + j);
          cnt += binCnt[a * BINS + j];
          if (binBox[o] < X0) X0 = binBox[o];
          if (binBox[o + 1] < Y0) Y0 = binBox[o + 1];
          if (binBox[o + 2] < Z0) Z0 = binBox[o + 2];
          if (binBox[o + 3] > X1) X1 = binBox[o + 3];
          if (binBox[o + 4] > Y1) Y1 = binBox[o + 4];
          if (binBox[o + 5] > Z1) Z1 = binBox[o + 5];
          rArea[j] = area(X0, Y0, Z0, X1, Y1, Z1);
          rCnt[j] = cnt;
          const r = 6 * j;
          rBox[r] = X0; rBox[r + 1] = Y0; rBox[r + 2] = Z0; rBox[r + 3] = X1; rBox[r + 4] = Y1; rBox[r + 5] = Z1;
        }
        // sweep from the left and evaluate each split plane
        X0 = Infinity; Y0 = Infinity; Z0 = Infinity; X1 = -Infinity; Y1 = -Infinity; Z1 = -Infinity; cnt = 0;
        for (let j = 1; j < nb; j++) {
          const o = 6 * (a * BINS + j - 1);
          cnt += binCnt[a * BINS + j - 1];
          if (binBox[o] < X0) X0 = binBox[o];
          if (binBox[o + 1] < Y0) Y0 = binBox[o + 1];
          if (binBox[o + 2] < Z0) Z0 = binBox[o + 2];
          if (binBox[o + 3] > X1) X1 = binBox[o + 3];
          if (binBox[o + 4] > Y1) Y1 = binBox[o + 4];
          if (binBox[o + 5] > Z1) Z1 = binBox[o + 5];
          if (cnt === 0 || rCnt[j] === 0) continue;
          const cost = area(X0, Y0, Z0, X1, Y1, Z1) * cnt + rArea[j] * rCnt[j];
          if (cost < bestCost) {
            bestCost = cost; bestAxis = a; bestBin = j;
            lb[0] = X0; lb[1] = Y0; lb[2] = Z0; lb[3] = X1; lb[4] = Y1; lb[5] = Z1;
            const r = 6 * j;
            rb[0] = rBox[r]; rb[1] = rBox[r + 1]; rb[2] = rBox[r + 2]; rb[3] = rBox[r + 3]; rb[4] = rBox[r + 4]; rb[5] = rBox[r + 5];
          }
        }
      }
    }

    // leaf or split?
    let mid = -1;
    if (bestAxis >= 0) {
      const splitCost = C_TRAV + bestCost / Math.max(pArea, 1e-30);
      if (n <= MAX_LEAF && splitCost >= n) { leaves++; continue; }
      const a = bestAxis, cm = cmin[a], sa = sc[a];
      let i = s, j = e - 1;
      while (i <= j) {
        let q = ((cen[3 * idx[i] + a] - cm) * sa) | 0;
        if (q > nb - 1) q = nb - 1;
        if (q < bestBin) i++;
        else { const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; j--; }
      }
      mid = i;
    }
    if (mid <= s || mid >= e) {
      // every centroid in one spot: split the list in half (or stop if small)
      if (n <= MAX_LEAF) { leaves++; continue; }
      mid = (s + e) >>> 1;
      boxOf(idx, bb, s, mid, lb);
      boxOf(idx, bb, mid, e, rb);
    }
    const L = count++, R = count++;
    nLeft[ni] = L;
    nStart[L] = s; nEnd[L] = mid; nStart[R] = mid; nEnd[R] = e;
    nDepth[L] = nDepth[R] = depth + 1;
    nBox.set(lb, 6 * L);
    nBox.set(rb, 6 * R);
    stack.push(R, L);
  }

  // flatten: interior nodes in depth-first pre-order, child boxes in the parent
  const interior = Math.max(1, count - leaves);
  const out = new Float32Array(interior * 16);
  const outU = new Uint32Array(out.buffer);
  const slot = (dst, ci) => {
    out[dst] = nBox[6 * ci]; out[dst + 1] = nBox[6 * ci + 1]; out[dst + 2] = nBox[6 * ci + 2];
    out[dst + 4] = nBox[6 * ci + 3]; out[dst + 5] = nBox[6 * ci + 4]; out[dst + 6] = nBox[6 * ci + 5];
    if (nLeft[ci] < 0) { outU[dst + 3] = nStart[ci]; outU[dst + 7] = nEnd[ci] - nStart[ci]; }
  };
  let next = 0;
  if (nLeft[0] < 0) {
    // the root is a leaf: one interior node holding it next to an empty slot
    slot(0, 0);
    emptySlot(out, outU, 8);
    next = 1;
  } else {
    const st = [0, -1];               // pairs: build node, parent slot to patch
    while (st.length) {
      const dstRef = st.pop(), ni = st.pop();
      const oi = next++;
      if (dstRef >= 0) { outU[dstRef + 3] = oi; outU[dstRef + 7] = 0; }
      const L = nLeft[ni], R = L + 1;
      slot(16 * oi, L);
      slot(16 * oi + 8, R);
      if (nLeft[R] >= 0) st.push(R, 16 * oi + 8);
      if (nLeft[L] >= 0) st.push(L, 16 * oi);      // popped first: left subtree follows its parent
    }
  }
  return { nodes: out.subarray(0, next * 16), order: idx, nodeCount: next, leafCount: leaves, depth: maxDepth,
    bounds: [rx0, ry0, rz0, rx1, ry1, rz1], ms: now() - t0 };
}

// A child slot no ray reaches: a point box 1.7e30 away, beyond any ray's range (an
// inverted box would pass the min/max slab test), flagged as a one-triangle leaf so
// that even a hit could not recurse.
export function emptySlot(f32, u32, dst) {
  f32[dst] = f32[dst + 1] = f32[dst + 2] = -1e30;
  f32[dst + 4] = f32[dst + 5] = f32[dst + 6] = -1e30;
  u32[dst + 3] = 0;
  u32[dst + 7] = 1;
}

function boxOf(idx, bb, s, e, outBox) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let k = s; k < e; k++) {
    const i = 6 * idx[k];
    if (bb[i] < x0) x0 = bb[i];
    if (bb[i + 1] < y0) y0 = bb[i + 1];
    if (bb[i + 2] < z0) z0 = bb[i + 2];
    if (bb[i + 3] > x1) x1 = bb[i + 3];
    if (bb[i + 4] > y1) y1 = bb[i + 4];
    if (bb[i + 5] > z1) z1 = bb[i + 5];
  }
  outBox[0] = x0; outBox[1] = y0; outBox[2] = z0; outBox[3] = x1; outBox[4] = y1; outBox[5] = z1;
}

// CPU reference traversal (closest hit) for tests: returns {t, tri} in BVH order.
export function traceCPU(bvh, tris, o, d, tmax = Infinity) {
  const nodes = bvh.nodes, nu = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
  const inv = [1 / d[0], 1 / d[1], 1 / d[2]];
  let best = tmax, bestTri = -1;
  const stack = [0];
  const box = (o4) => {
    let t0 = 0, t1 = best;
    for (let a = 0; a < 3; a++) {
      let ta = (nodes[o4 + a] - o[a]) * inv[a], tb = (nodes[o4 + 4 + a] - o[a]) * inv[a];
      if (ta > tb) { const x = ta; ta = tb; tb = x; }
      t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    }
    return t0 <= t1;
  };
  while (stack.length) {
    const ni = stack.pop();
    for (let c = 0; c < 2; c++) {
      const b = 16 * ni + 8 * c;
      if (!box(b)) continue;
      const ref = nu[b + 3], cnt = nu[b + 7];
      if (cnt === 0) { stack.push(ref); continue; }
      for (let k = ref; k < ref + cnt; k++) {
        const t = rayTri(tris, k, o, d);
        if (t > 0 && t < best) { best = t; bestTri = k; }
      }
    }
  }
  return { t: best, tri: bestTri };
}

// Möller–Trumbore against the packed triangle list (v0, e1, e2 as vec4s).
function rayTri(T, k, o, d) {
  const b = 12 * k;
  const v0 = [T[b], T[b + 1], T[b + 2]], e1 = [T[b + 4], T[b + 5], T[b + 6]], e2 = [T[b + 8], T[b + 9], T[b + 10]];
  const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
  const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (Math.abs(det) < 1e-20) return -1;
  const id = 1 / det;
  const s = [o[0] - v0[0], o[1] - v0[1], o[2] - v0[2]];
  const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * id;
  if (u < 0 || u > 1) return -1;
  const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
  const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * id;
  if (v < 0 || u + v > 1) return -1;
  return (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * id;
}
