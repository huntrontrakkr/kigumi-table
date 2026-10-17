// WGSL for the batched conjugate-gradient solver: CSR SpMM over k columns
// with fused dot products, the per-column CG scalars (alpha, beta, the
// convergence flags) computed on the GPU, and the vector updates.
//
// Layout.  A block of k vectors is stored row-major and padded to kp columns
// (element (i, j) at i*kp + j), read as array<V> where V is f32, vec2<f32> or
// vec4<f32> (W = 1, 2, 4 lanes): C = kp / W "column groups" per row.  A row
// kernel runs one thread per (row, column group); the threads of a row read
// the same CSR entries and neighbouring (coalesced) vector entries.  A
// workgroup of WG threads covers RPW = floor(WG / C) rows, and reduces its dot
// products over those rows in workgroup memory; a second, tiny pass (one
// workgroup per column group) finishes the sums and updates the scalars.
//
// Per-column scalars live in one storage buffer `scal` of NF fields of kp
// floats (field f, column j at f*kp + j): no CPU read-back per iteration; the
// host only polls the ACT flags every few dozen iterations.

export const WG = 256;          // threads per workgroup, row and element kernels
export const RT = 64;           // threads per workgroup, final reductions
export const MAX_GRID = 65535;  // workgroups per dispatch dimension (WebGPU default limit)

// scalar fields
export const F = { RZ: 0, RR: 1, RR0: 2, TOL2: 3, ALPHA: 4, BETA: 5, ACT: 6, ITS: 7, PQ: 8 };
export const NF = 9;

/** Lane width, padded column count and row tiling for k right-hand sides. */
export function layoutFor(k) {
  const W = k === 1 ? 1 : k === 2 ? 2 : 4;
  const kp = Math.ceil(k / W) * W;
  const C = kp / W;
  if (C > WG) throw new RangeError(`too many columns in one batch (${k})`);
  const RPW = Math.floor(WG / C);
  let p2 = 1;
  while (p2 < RPW) p2 *= 2;
  const SHALF = RPW > 1 ? p2 / 2 : 0;
  return { W, kp, C, RPW, SHALF };
}

function header({ W, C, RPW, SHALF }) {
  const V = W === 1 ? 'f32' : `vec${W}<f32>`;
  return /* wgsl */ `
alias V = ${V};
const C : u32 = ${C}u;
const RPW : u32 = ${RPW}u;
const SHALF : u32 = ${SHALF}u;
const WG : u32 = ${WG}u;
const RT : u32 = ${RT}u;
${Object.entries(F).map(([k, v]) => `const ${k} : u32 = ${v}u;`).join('\n')}
struct Params {
  n : u32,      // rows
  nwg : u32,    // workgroups of the row kernels
  gx : u32,     // their grid width (2-D dispatch beyond ${MAX_GRID} workgroups)
  ne : u32,     // n * C vector elements
  egx : u32,    // grid width of the element-wise kernels
  eta2 : f32,   // inner relative tolerance, squared
  pad0 : u32,
  pad1 : u32,
}
@group(0) @binding(0) var<uniform> prm : Params;
`;
}

/**
 * Row kernel: y = M x for a CSR matrix M over all column groups, with
 *   dots = 0: nothing else;
 *   dots = 1: partial sums of x.y          (p.q for alpha);
 *   dots = 2: partial sums of r.y and r.r  (r.z and r.r for beta / convergence).
 * spmv = false (with dots = 2): no product, partial sums of r.r twice (Jacobi: z = r).
 */
export function rowKernel(lay, { spmv = true, dots = 0 } = {}) {
  const b = [];
  if (spmv) {
    b.push('@group(0) @binding(1) var<storage, read> rowptr : array<u32>;');
    b.push('@group(0) @binding(2) var<storage, read> colidx : array<u32>;');
    b.push('@group(0) @binding(3) var<storage, read> vals : array<f32>;');
    b.push('@group(0) @binding(4) var<storage, read> xin : array<V>;');
    b.push('@group(0) @binding(5) var<storage, read_write> yout : array<V>;');
  }
  if (dots >= 1) {
    b.push('@group(0) @binding(6) var<storage, read_write> partial : array<V>;');
    b.push('var<workgroup> sh0 : array<V, WG>;');
  }
  if (dots === 2) {
    b.push('@group(0) @binding(7) var<storage, read> rvec : array<V>;');
    b.push('var<workgroup> sh1 : array<V, WG>;');
  }
  let body = '';
  if (spmv) {
    body += `
    var acc = V(0.0);
    let pend = rowptr[row + 1u];
    for (var p = rowptr[row]; p < pend; p++) {
      acc += vals[p] * xin[colidx[p] * C + c];
    }
    yout[e] = acc;`;
    if (dots === 1) body += '\n    d0 = acc * xin[e];';
    if (dots === 2) body += '\n    let r = rvec[e];\n    d0 = acc * r;\n    d1 = r * r;';
  } else {
    body += '\n    let r = rvec[e];\n    d0 = r * r;\n    d1 = d0;';
  }
  let red = '';
  if (dots >= 1) {
    red = `
  sh0[li] = d0;${dots === 2 ? '\n  sh1[li] = d1;' : ''}
  workgroupBarrier();
  for (var s = SHALF; s > 0u; s >>= 1u) {
    if (slot < s && slot + s < RPW) {
      sh0[li] += sh0[li + s * C];${dots === 2 ? '\n      sh1[li] += sh1[li + s * C];' : ''}
    }
    workgroupBarrier();
  }
  if (li < C && wg < prm.nwg) {
    partial[wg * C + li] = sh0[li];${dots === 2 ? '\n    partial[(prm.nwg + wg) * C + li] = sh1[li];' : ''}
  }`;
  }
  return header(lay) + b.join('\n') + `
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
  let wg = wid.x + wid.y * prm.gx;
  let slot = li / C;
  let c = li - slot * C;
  let row = wg * RPW + slot;
  var d0 = V(0.0);
  var d1 = V(0.0);
  if (wg < prm.nwg && slot < RPW && row < prm.n) {
    let e = row * C + c;${body}
  }${red}
}
`;
}

/**
 * Final reduction of the row-kernel partial sums (one workgroup per column
 * group) and the CG scalar update:
 *   'alpha': pq = sum(p.q);  alpha = rz / pq  (0 for inactive columns, or on breakdown pq <= 0)
 *   'beta' : rz' = sum(r.z), rr = sum(r.r);  beta = rz'/rz;  a column stays active while rr > tol2
 *   'init' : rz = sum(r.z), rr0 = sum(r.r);  tol2 = eta^2 rr0;  active iff rr0 > 0
 */
export function reduceKernel(lay, mode) {
  const two = mode !== 'alpha';
  let fin;
  if (mode === 'alpha') {
    fin = `
    let pq = sh0[0];
    let ok = (scal[ACT * C + c] > V(0.0)) & (pq > V(0.0));
    scal[ALPHA * C + c] = select(V(0.0), scal[RZ * C + c] / pq, ok);
    scal[ACT * C + c] = select(V(0.0), V(1.0), ok);
    scal[PQ * C + c] = pq;`;
  } else if (mode === 'beta') {
    fin = `
    let rzn = sh0[0];
    let rr = sh1[0];
    let act = scal[ACT * C + c] > V(0.0);
    let keep = act & (rr > scal[TOL2 * C + c]) & (rzn > V(0.0));
    scal[BETA * C + c] = select(V(0.0), rzn / scal[RZ * C + c], keep);
    scal[ITS * C + c] = scal[ITS * C + c] + select(V(0.0), V(1.0), act);
    scal[ACT * C + c] = select(V(0.0), V(1.0), keep);
    scal[RZ * C + c] = rzn;
    scal[RR * C + c] = rr;`;
  } else if (mode === 'init') {
    fin = `
    let rz = sh0[0];
    let rr = sh1[0];
    let ok = (rr > V(0.0)) & (rz > V(0.0));
    scal[RZ * C + c] = rz;
    scal[RR * C + c] = rr;
    scal[RR0 * C + c] = rr;
    scal[TOL2 * C + c] = prm.eta2 * rr;
    scal[ALPHA * C + c] = V(0.0);
    scal[BETA * C + c] = V(0.0);
    scal[ITS * C + c] = V(0.0);
    scal[PQ * C + c] = V(0.0);
    scal[ACT * C + c] = select(V(0.0), V(1.0), ok);`;
  } else throw new Error('unknown reduce mode ' + mode);
  return header(lay) + `
@group(0) @binding(1) var<storage, read> partial : array<V>;
@group(0) @binding(2) var<storage, read_write> scal : array<V>;
var<workgroup> sh0 : array<V, RT>;
${two ? 'var<workgroup> sh1 : array<V, RT>;' : ''}
@compute @workgroup_size(RT)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
  let c = wid.x;
  var s0 = V(0.0);
  ${two ? 'var s1 = V(0.0);' : ''}
  for (var w = li; w < prm.nwg; w += RT) {
    s0 += partial[w * C + c];
    ${two ? 's1 += partial[(prm.nwg + w) * C + c];' : ''}
  }
  sh0[li] = s0;
  ${two ? 'sh1[li] = s1;' : ''}
  workgroupBarrier();
  for (var s = RT / 2u; s > 0u; s >>= 1u) {
    if (li < s) {
      sh0[li] += sh0[li + s];
      ${two ? 'sh1[li] += sh1[li + s];' : ''}
    }
    workgroupBarrier();
  }
  if (li == 0u) {${fin}
  }
}
`;
}

/** x += alpha p; r -= alpha q (per column alpha). */
export function axpyKernel(lay) {
  return header(lay) + `
@group(0) @binding(1) var<storage, read> scal : array<V>;
@group(0) @binding(2) var<storage, read_write> xv : array<V>;
@group(0) @binding(3) var<storage, read_write> rv : array<V>;
@group(0) @binding(4) var<storage, read> pv : array<V>;
@group(0) @binding(5) var<storage, read> qv : array<V>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
  let e = (wid.x + wid.y * prm.egx) * WG + li;
  if (e >= prm.ne) { return; }
  let a = scal[ALPHA * C + e % C];
  xv[e] += a * pv[e];
  rv[e] -= a * qv[e];
}
`;
}

/** p = z + beta p (p = z where beta = 0, so stale or non-finite p never leaks in). */
export function xpbyKernel(lay) {
  return header(lay) + `
@group(0) @binding(1) var<storage, read> scal : array<V>;
@group(0) @binding(2) var<storage, read> zv : array<V>;
@group(0) @binding(3) var<storage, read_write> pv : array<V>;
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
  let e = (wid.x + wid.y * prm.egx) * WG + li;
  if (e >= prm.ne) { return; }
  let b = scal[BETA * C + e % C];
  let z = zv[e];
  pv[e] = select(z + b * pv[e], z, b == V(0.0));
}
`;
}
