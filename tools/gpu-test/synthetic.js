// Synthetic SPD test systems with known solutions, built in float64 JS:
//   laplace2d -- 5-point Laplacian on an nx x ny grid (Dirichlet), manufactured
//                solutions (smooth and random columns): x* exact, b = A x*;
//   beam1d    -- Euler-Bernoulli beam, Hermite cubic elements (w, theta per
//                node), penalty supports like the kigumi models, realistic
//                loads; reference x from a banded Cholesky in float64.
// Each returns { name, n, k, indptr, indices, data, rhs, xref, note } with
// column-major rhs/xref.

/** CSR from a map of rows: rows[i] = Map(col -> value). */
function toCSR(n, rows) {
  const indptr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) indptr[i + 1] = indptr[i] + rows[i].size;
  const indices = new Int32Array(indptr[n]);
  const data = new Float64Array(indptr[n]);
  for (let i = 0; i < n; i++) {
    const cols = [...rows[i].keys()].sort((a, b) => a - b);
    let o = indptr[i];
    for (const c of cols) { indices[o] = c; data[o] = rows[i].get(c); o++; }
  }
  return { indptr, indices, data };
}

function matvecCol(n, A, x, xo, y, yo) {
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let p = A.indptr[i]; p < A.indptr[i + 1]; p++) s += A.data[p] * x[xo + A.indices[p]];
    y[yo + i] = s;
  }
}

/** Deterministic pseudo-random numbers in [-1, 1). */
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return (s / 4294967296) * 2 - 1;
  };
}

export function laplace2d(nx, ny, k = 4, seed = 7) {
  const n = nx * ny;
  const rows = Array.from({ length: n }, () => new Map());
  const id = (i, j) => j * nx + i;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const r = rows[id(i, j)];
      r.set(id(i, j), 4);
      if (i > 0) r.set(id(i - 1, j), -1);
      if (i < nx - 1) r.set(id(i + 1, j), -1);
      if (j > 0) r.set(id(i, j - 1), -1);
      if (j < ny - 1) r.set(id(i, j + 1), -1);
    }
  }
  const A = toCSR(n, rows);
  const xref = new Float64Array(n * k);
  const rnd = rng(seed);
  for (let c = 0; c < k; c++) {
    // even columns smooth (a few low modes), odd columns random
    const a = 1 + (c % 3), b = 1 + ((c >> 1) % 2);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x = (i + 1) / (nx + 1), y = (j + 1) / (ny + 1);
        xref[c * n + id(i, j)] = c % 2 === 0
          ? Math.sin(Math.PI * a * x) * Math.sin(Math.PI * b * y) + 0.3 * x * y * (1 - x) * (1 - y)
          : rnd();
      }
    }
  }
  const rhs = new Float64Array(n * k);
  for (let c = 0; c < k; c++) matvecCol(n, A, xref, c * n, rhs, c * n);
  return { name: `laplace2d_${nx}x${ny}_k${k}`, n, k, ...A, rhs, xref,
    note: '5-point Laplacian, manufactured x* (smooth + random columns), b = A x*' };
}

/** Solve a symmetric banded system (half bandwidth bw) by Cholesky in float64; B column-major n x k. */
export function bandedSolve(n, A, bw, B, k) {
  // dense band storage: L[i][d] = L(i, i-d), d = 0..bw
  const L = new Float64Array(n * (bw + 1));
  for (let i = 0; i < n; i++) {
    for (let p = A.indptr[i]; p < A.indptr[i + 1]; p++) {
      const j = A.indices[p];
      if (j <= i) L[i * (bw + 1) + (i - j)] += A.data[p];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let d = Math.min(bw, i); d >= 0; d--) {
      const j = i - d;
      let s = L[i * (bw + 1) + d];
      for (let t = 1; t <= bw - d && j - t >= 0; t++) {
        // L(i, j-t) * L(j, j-t)
        if (d + t <= bw) s -= L[i * (bw + 1) + d + t] * L[j * (bw + 1) + t];
      }
      if (d === 0) {
        if (!(s > 0)) throw new Error('bandedSolve: matrix not SPD');
        L[i * (bw + 1)] = Math.sqrt(s);
      } else {
        L[i * (bw + 1) + d] = s / L[j * (bw + 1)];
      }
    }
  }
  const X = Float64Array.from(B);
  for (let c = 0; c < k; c++) {
    const o = c * n;
    for (let i = 0; i < n; i++) {           // L y = b
      let s = X[o + i];
      for (let d = 1; d <= Math.min(bw, i); d++) s -= L[i * (bw + 1) + d] * X[o + i - d];
      X[o + i] = s / L[i * (bw + 1)];
    }
    for (let i = n - 1; i >= 0; i--) {      // L^T x = y
      let s = X[o + i];
      for (let d = 1; d <= bw && i + d < n; d++) s -= L[(i + d) * (bw + 1) + d] * X[o + i + d];
      X[o + i] = s / L[i * (bw + 1)];
    }
  }
  return X;
}

/**
 * Beam of length L (mm) and stiffness EI (N mm^2), ne Hermite elements; w held
 * by penalty springs (1e6 x the largest diagonal) at the ends and mid-span,
 * theta clamped at the left end.  Loads: uniform q, a point load at 1/3 span,
 * a moment at the right end.
 */
export function beam1d(ne, { L = 1000, EI = 2e11, refine = 2 } = {}) {
  const n = 2 * (ne + 1);
  const h = L / ne;
  const c = EI / (h * h * h);
  const ke = [
    [12, 6 * h, -12, 6 * h],
    [6 * h, 4 * h * h, -6 * h, 2 * h * h],
    [-12, -6 * h, 12, -6 * h],
    [6 * h, 2 * h * h, -6 * h, 4 * h * h],
  ];
  const rows = Array.from({ length: n }, () => new Map());
  const add = (i, j, v) => rows[i].set(j, (rows[i].get(j) || 0) + v);
  for (let e = 0; e < ne; e++) {
    const d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3];
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) add(d[a], d[b], c * ke[a][b]);
  }
  let dmax = 0;
  for (let i = 0; i < n; i++) dmax = Math.max(dmax, rows[i].get(i));
  const pen = 1e6 * dmax;
  for (const node of [0, Math.round(ne / 2), ne]) add(2 * node, 2 * node, pen);
  add(1, 1, pen);
  const A = toCSR(n, rows);
  const k = 3;
  const rhs = new Float64Array(n * k);
  const q = 0.5;                                     // N/mm
  for (let e = 0; e < ne; e++) {
    const f = [q * h / 2, q * h * h / 12, q * h / 2, -q * h * h / 12];
    for (let a = 0; a < 4; a++) rhs[2 * e + a] += f[a];
  }
  rhs[n + 2 * Math.round(ne / 3)] = -800;            // point load, N
  rhs[2 * n + 2 * ne + 1] = 2e5;                     // end moment, N mm
  const X = bandedSolve(n, A, 3, rhs, k);
  // float64 iterative refinement of the reference (the penalties cost digits)
  const r = new Float64Array(n);
  for (let it = 0; it < refine; it++) {
    for (let col = 0; col < k; col++) {
      matvecCol(n, A, X, col * n, r, 0);
      for (let i = 0; i < n; i++) r[i] = rhs[col * n + i] - r[i];
      const d = bandedSolve(n, A, 3, r, 1);
      for (let i = 0; i < n; i++) X[col * n + i] += d[i];
    }
  }
  return { name: `beam1d_ne${ne}`, n, k, ...A, rhs, xref: X,
    note: `Hermite beam, ${ne} elements, penalty supports; reference: banded Cholesky (f64) + refinement` };
}
