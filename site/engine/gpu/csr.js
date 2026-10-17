// CPU-side (float64) sparse helpers for the WebGPU solver: input coercion and
// checks, symmetric Jacobi scaling, the FSAI preconditioner set-up, CSR
// transposes, residuals and layout conversions.  Plain JS, no dependencies.
//
// Conventions: CSR with 0-based Int32 indices; dense blocks of k vectors are
// either column-major (v[j*n + i], the public API) or row-major (v[i*k + j],
// used internally because it matches the GPU layout and keeps SpMM loads
// contiguous).

/** Coerce an integer array (Int32Array, Uint32Array, BigInt64Array, plain array, ...) to Int32Array. */
export function asInt32(a, name) {
  if (a instanceof Int32Array) return a;
  if (a instanceof Uint32Array) return new Int32Array(a.buffer, a.byteOffset, a.length);
  if (typeof BigInt64Array !== 'undefined' && (a instanceof BigInt64Array || a instanceof BigUint64Array)) {
    const out = new Int32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = Number(a[i]);
    return out;
  }
  if (a && typeof a.length === 'number') return Int32Array.from(a);
  throw new TypeError(`${name}: expected an integer array`);
}

/** Coerce a numeric array to Float64Array (no copy if it already is one). */
export function asFloat64(a, name) {
  if (a instanceof Float64Array) return a;
  if (a && typeof a.length === 'number') return Float64Array.from(a);
  throw new TypeError(`${name}: expected a numeric array`);
}

/** Validate CSR structure (throws on malformed input). */
export function checkCSR(n, indptr, indices, data) {
  if (!(n >= 0) || (n | 0) !== n) throw new RangeError(`n must be a non-negative integer, got ${n}`);
  if (indptr.length !== n + 1) throw new RangeError(`indptr has length ${indptr.length}, expected n+1 = ${n + 1}`);
  if (indptr[0] !== 0) throw new RangeError('indptr[0] must be 0');
  const nnz = indptr[n];
  if (indices.length < nnz || data.length < nnz) {
    throw new RangeError(`indices/data shorter than nnz = ${nnz} (${indices.length}, ${data.length})`);
  }
  for (let i = 0; i < n; i++) {
    if (indptr[i + 1] < indptr[i]) throw new RangeError(`indptr decreases at row ${i}`);
  }
  for (let p = 0; p < nnz; p++) {
    const c = indices[p];
    if (c < 0 || c >= n) throw new RangeError(`column index ${c} out of range at position ${p}`);
    if (!Number.isFinite(data[p])) throw new RangeError(`non-finite matrix entry at position ${p}`);
  }
  return nnz;
}

/**
 * Symmetric Jacobi scaling S = diag(1/sqrt(a_ii)): returns s and the values of
 * S A S (same pattern as A).  Throws if a diagonal entry is not positive (the
 * matrix cannot be SPD then).
 */
export function jacobiScale(n, indptr, indices, data) {
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let p = indptr[i]; p < indptr[i + 1]; p++) if (indices[p] === i) d[i] += data[p];
  }
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (!(d[i] > 0) || !Number.isFinite(d[i])) {
      throw new RangeError(`matrix is not SPD: diagonal entry ${d[i]} at row ${i}`);
    }
    s[i] = 1 / Math.sqrt(d[i]);
  }
  const nnz = indptr[n];
  const scaled = new Float64Array(nnz);
  for (let i = 0; i < n; i++) {
    const si = s[i];
    for (let p = indptr[i]; p < indptr[i + 1]; p++) scaled[p] = si * data[p] * s[indices[p]];
  }
  return { s, scaled };
}

/**
 * Factorized sparse approximate inverse (FSAI, Kolotilina & Yeremin): a lower
 * triangular G with G A G^T ~ I and diag(G A G^T) = 1 exactly, so that
 * M^{-1} = G^T G is an SPD preconditioner applied with two sparse
 * mat-vecs -- no triangular solves, which is what makes it GPU-friendly.
 *
 * Row i of G lives on a pattern P_i of columns j <= i:
 *   level 1: the lower triangle of A's own pattern (row i of tril(A));
 *   level 2: the lower triangle of the pattern of A^2 (neighbours of neighbours).
 * For each row, with M = A[P_i, P_i] = L L^T (dense Cholesky), the row is
 * L^{-T} e_last, i.e. the solution of A[P,P] g = e_last scaled so that
 * g^T A[P,P] g = 1.  Uses the lower triangle of A only (so a slightly
 * unsymmetric input is read as its lower triangle).
 *
 * Returns { ptr, idx, val (Float64Array), fails } -- rows whose local
 * Cholesky broke down (not SPD in floating point) fall back to a Jacobi row.
 */
export function fsai(n, indptr, indices, vals, level = 1) {
  const pos = new Int32Array(n).fill(-1);      // column -> local index, per row
  const mark = new Int32Array(n).fill(-1);     // pattern de-duplication
  // pass 1: patterns (so that the output can be allocated exactly)
  const rowPat = new Array(n);
  let total = 0;
  let mmax = 1;
  const tmp = [];
  for (let i = 0; i < n; i++) {
    tmp.length = 0;
    const add = (j) => { if (j <= i && mark[j] !== i) { mark[j] = i; tmp.push(j); } };
    add(i);
    for (let p = indptr[i]; p < indptr[i + 1]; p++) {
      const j = indices[p];
      add(j);
      if (level >= 2) for (let q = indptr[j]; q < indptr[j + 1]; q++) add(indices[q]);
    }
    const P = Int32Array.from(tmp).sort();
    rowPat[i] = P;
    total += P.length;
    if (P.length > mmax) mmax = P.length;
  }
  const ptr = new Int32Array(n + 1);
  const idx = new Int32Array(total);
  const val = new Float64Array(total);
  const L = new Float64Array(mmax * mmax);
  const h = new Float64Array(mmax);
  let fails = 0;
  let o = 0;
  for (let i = 0; i < n; i++) {
    const P = rowPat[i];
    const m = P.length;                          // P is sorted, so P[m-1] === i
    for (let a = 0; a < m; a++) pos[P[a]] = a;
    // dense lower triangle of A[P, P] (row-major m x m in L)
    for (let a = 0; a < m; a++) {
      const ra = a * m;
      for (let b = 0; b <= a; b++) L[ra + b] = 0;
      const r = P[a];
      for (let p = indptr[r]; p < indptr[r + 1]; p++) {
        const b = pos[indices[p]];
        if (b >= 0 && b <= a) L[ra + b] += vals[p];
      }
    }
    for (let a = 0; a < m; a++) pos[P[a]] = -1;
    // in-place Cholesky, row by row
    let ok = true;
    for (let a = 0; a < m && ok; a++) {
      const ra = a * m;
      for (let b = 0; b <= a; b++) {
        const rb = b * m;
        let sum = L[ra + b];
        for (let c = 0; c < b; c++) sum -= L[ra + c] * L[rb + c];
        if (a === b) {
          if (!(sum > 0)) { ok = false; break; }
          L[ra + a] = Math.sqrt(sum);
        } else {
          L[ra + b] = sum / L[rb + b];
        }
      }
    }
    ptr[i] = o;
    if (ok) {
      // h = L^{-T} e_last  (back substitution with L^T)
      h[m - 1] = 1 / L[(m - 1) * m + (m - 1)];
      for (let a = m - 2; a >= 0; a--) {
        let sum = 0;
        for (let b = a + 1; b < m; b++) sum += L[b * m + a] * h[b];
        h[a] = -sum / L[a * m + a];
      }
      for (let a = 0; a < m; a++) { idx[o] = P[a]; val[o] = h[a]; o++; }
    } else {
      fails++;
      let dii = 0;
      for (let p = indptr[i]; p < indptr[i + 1]; p++) if (indices[p] === i) dii += vals[p];
      idx[o] = i; val[o] = dii > 0 ? 1 / Math.sqrt(dii) : 1; o++;
    }
  }
  ptr[n] = o;
  return { ptr, idx: o === total ? idx : idx.slice(0, o), val: o === total ? val : val.slice(0, o), fails };
}

/** Transpose of a CSR matrix (n x n). */
export function transposeCSR(n, ptr, idx, val) {
  const nnz = ptr[n];
  const tptr = new Int32Array(n + 1);
  for (let p = 0; p < nnz; p++) tptr[idx[p] + 1]++;
  for (let i = 0; i < n; i++) tptr[i + 1] += tptr[i];
  const next = tptr.slice(0, n);
  const tidx = new Int32Array(nnz);
  const tval = new val.constructor(nnz);
  for (let i = 0; i < n; i++) {
    for (let p = ptr[i]; p < ptr[i + 1]; p++) {
      const q = next[idx[p]]++;
      tidx[q] = i;
      tval[q] = val[p];
    }
  }
  return { ptr: tptr, idx: tidx, val: tval };
}

/**
 * r = b - A x in float64, all blocks row-major n x k.  With `floor` (a
 * Float64Array(k)), also the float64 attainable-residual estimate per column,
 *   floor_j = u || |A||x_j| + |b_j| || / ||b_j||      (u = 2^-53),
 * in the same pass: below about this, any residual evaluated in float64 is
 * rounding noise, whatever the solver (direct solvers land at 0.5-3x it).
 */
export function residualRowMajor(n, k, indptr, indices, data, x, b, r, floor = null) {
  if (floor) return residualAndFloor(n, k, indptr, indices, data, x, b, r, floor);
  if (k === 1) {
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let p = indptr[i]; p < indptr[i + 1]; p++) acc += data[p] * x[indices[p]];
      r[i] = b[i] - acc;
    }
    return r;
  }
  const acc = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    acc.fill(0);
    for (let p = indptr[i]; p < indptr[i + 1]; p++) {
      const a = data[p];
      const base = indices[p] * k;
      for (let j = 0; j < k; j++) acc[j] += a * x[base + j];
    }
    const o = i * k;
    for (let j = 0; j < k; j++) r[o + j] = b[o + j] - acc[j];
  }
  return r;
}

function residualAndFloor(n, k, indptr, indices, data, x, b, r, floor) {
  const u = 2 ** -53;
  const acc = new Float64Array(k);
  const abs = new Float64Array(k);
  const num = new Float64Array(k);
  const den = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    acc.fill(0);
    abs.fill(0);
    for (let p = indptr[i]; p < indptr[i + 1]; p++) {
      const a = data[p];
      const aa = Math.abs(a);
      const base = indices[p] * k;
      for (let j = 0; j < k; j++) {
        const xv = x[base + j];
        acc[j] += a * xv;
        abs[j] += aa * Math.abs(xv);
      }
    }
    const o = i * k;
    for (let j = 0; j < k; j++) {
      const bv = b[o + j];
      r[o + j] = bv - acc[j];
      const t = abs[j] + Math.abs(bv);
      num[j] += t * t;
      den[j] += bv * bv;
    }
  }
  for (let j = 0; j < k; j++) floor[j] = u * Math.sqrt(num[j]) / (den[j] > 0 ? Math.sqrt(den[j]) : 1);
  return r;
}

/** Euclidean norms of the k columns of a row-major n x k block. */
export function colNormsRowMajor(n, k, v) {
  const s = new Float64Array(k);
  for (let i = 0; i < n; i++) {
    const o = i * k;
    for (let j = 0; j < k; j++) { const t = v[o + j]; s[j] += t * t; }
  }
  for (let j = 0; j < k; j++) s[j] = Math.sqrt(s[j]);
  return s;
}

/** Column-major (v[j*n+i]) columns [j0, j0+k) of an n-row block -> row-major n x k. */
export function colToRow(n, k, col, j0 = 0, out = new Float64Array(n * k)) {
  for (let j = 0; j < k; j++) {
    const o = (j0 + j) * n;
    for (let i = 0; i < n; i++) out[i * k + j] = col[o + i];
  }
  return out;
}

/** Row-major n x k -> columns [j0, j0+k) of a column-major block `col`. */
export function rowToCol(n, k, row, col, j0 = 0) {
  for (let j = 0; j < k; j++) {
    const o = (j0 + j) * n;
    for (let i = 0; i < n; i++) col[o + i] = row[i * k + j];
  }
  return col;
}

/** Element-wise equality of two typed arrays (same length). */
export function sameArray(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
