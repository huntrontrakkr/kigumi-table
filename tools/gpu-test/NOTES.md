# WebGPU sparse SPD solver: notes and results

`site/engine/gpu/sparse.js` solves `A X = B` for sparse symmetric positive definite `A`
(full symmetric CSR) with many right-hand sides on the GPU through WebGPU, to float64
accuracy although WGSL only has float32. This folder holds its tests.

**Bottom line (2026-10-17).** The solver is correct and float64-accurate on every model
system: relres ≤ 1e-9 and error against scipy ≤ 7e-11. It is robust on phones by
design: default limits only, short submissions, and a quick `converged: false` when a
system is too big. **But it is slower than SuperLU in Pyodide on all of the app's
systems.** On an RTX 3090 with a fast desktop CPU, the app's plate call (n = 4352,
k = 123) takes 273-287 ms on the GPU, against 62 ms for `kigumi.linalg.factor(K).solve(B)`
in Pyodide in the same browser. The 15k-dof plate takes 2.3 s against 0.24 s. These
fourth-order 2D systems need hundreds to thousands of CG iterations, and the iteration
count grows like n (κ ~ h⁻⁴), while a sparse Cholesky of a 2D mesh is cheap. So bigger
plates make the gap *wider*, not narrower (4.4x at 4.4k dof, 9.3x at 15k dof). Phones
were not available to test: phone GPUs have higher dispatch latency and phone CPUs run
WebAssembly at a fraction of desktop speed, so the ratio there is unknown but not
obviously better. Keep SuperLU as the default, unless measurements on target phones say
otherwise (`?cases=superlu,model` on the test page measures both).

## Running the tests

```sh
cd ~/code/kigumi-table
node tools/gpu-test/run.mjs                                  # default plan, Playwright Chromium (SwiftShader here)
node tools/gpu-test/run.mjs --browser win-chrome             # the same on the Windows host's Chrome: RTX 3090
node tools/gpu-test/run.mjs --browser win-chrome --cases model,superlu --bigk all   # GPU vs SuperLU in Pyodide
node tools/gpu-test/run.mjs --cases robust                   # groups: layouts synthetic model robust compare superlu bench
node tools/gpu-test/run.mjs --compare --cases compare        # Jacobi / FSAI level 1 vs the default
node tools/gpu-test/run.mjs --cases bench --benchIters 400   # ms per PCG iteration at k = 1, 8, 64, 256
```

`run.mjs` runs `uv run python tools/gpu-test/convert.py` if `data/` is missing. It
serves the repository itself on `http://localhost:<free port>`: WebGPU needs a secure
context, and the page POSTs its log and results back. It opens
`tools/gpu-test/index.html`, prints a table, writes `out/results.json` (or `--out`) and
exits 0 when every checked case passes. The tests run in a dedicated worker, as in the
app; `--mode main` runs them on the main thread.

| `--browser` | browser | WebGPU adapter |
|---|---|---|
| `chromium` (default) | Playwright's headless Chromium from `tools/ui-test/node_modules`, with `LD_LIBRARY_PATH=~/.local/lib/chromedeps` and `--enable-unsafe-webgpu --enable-unsafe-swiftshader --ignore-gpu-blocklist` | SwiftShader, a CPU implementation of Vulkan: correctness and iteration counts are real, times are emulation |
| `win-chrome`, `win-edge` | the Windows host's browser, headless, started through PowerShell and killed by PID (`taskkill /T /F`); it reaches the WSL server through WSL's localhost forwarding | the real GPU. WSL2 itself has no hardware Vulkan driver (no Mesa dzn ICD), so Linux Chromium only gets SwiftShader |

Other options: `--eta`, `--precond`, `--timeout` (seconds), `--out`, `--headed`. The
`superlu` group loads Pyodide 314.0.7 from the CDN (the build `site/engine/worker.js`
uses) and needs the network. The page also works by hand: serve the repository root
and open `/tools/gpu-test/index.html?cases=...`; results land in `window.__results`.

Test data:

| file | content |
|---|---|
| `matrices/*.mtx`, `*_rhs.npy`, `*_x.npy` | systems exported from the Python model, with a scipy reference. Not in git: when missing, `convert.py` re-exports them from the `field-four` preset, with identical matrices but the model's own load cases (the original export used other right-hand sides) |
| `data/` | the systems as flat little-endian binaries (`convert.py`; gitignored) |
| `data/plate_model.*` | the app's own call: the `field-four` top plate with all 123 load cases, exactly what `solve_many` hands the GPU |
| `data/plate_big.*` | a 15k-dof plate built with `kigumi.plate.solve_plate` (h = 15 mm, 90 load cases), with a SuperLU reference refined in float64 |
| `synthetic.js` | 2D Laplacian (manufactured smooth and random solutions); Hermite beam (fourth order, mixed w/θ units, 1e6x penalty supports, realistic loads) with a banded-Cholesky float64 reference |

## API

```js
import { getGpu, solveSPD, releaseGpuBuffers, LIMITS_USED } from './gpu/sparse.js';  // from site/engine/
const gpu = await getGpu();      // { device, info } or null; works in a DedicatedWorker
const r = await solveSPD(gpu, { n, indptr, indices, data, rhs, k, tol: 1e-9 });
// r.x: Float64Array(n*k), column-major like rhs (column j at [j*n, (j+1)*n))
```

`indptr` and `indices` may be Int32Array, Uint32Array or BigInt64Array; `data`, `rhs`
and `x0` are Float64Array.

| result field | meaning |
|---|---|
| `x` | the solution, column-major |
| `iters` | lock-step PCG iterations executed on the GPU, all refinements and batches |
| `refinements` | float64 refinement steps |
| `relres[j]` | `‖b_j − A x_j‖₂/‖b_j‖₂`, evaluated in float64 |
| `converged` | see below |
| `tolMet` | every `relres[j] ≤ tol` (the strict test) |
| `floor[j]` | the float64 floor `u‖|A||x_j| + |b_j|‖/‖b_j‖` (u = 2⁻⁵³) |
| `stopped[j]` | `'tol'`, `'floor'`, `'stalled'`, `'maxRefine'`, `'non-finite'`, `'gpu'` or `'unsolved'` |
| `reason` | set when the GPU could not take the system |
| `gpuBytes` | GPU buffers of the solve |
| `submits`, `maxItersPerSubmit`, `msPerIter` | queue submissions, the most PCG iterations in one, the measured time per iteration |
| `ms`, `setupMs`, `gpuMs`, `cpuMs`, `precond`, `history` | timings, the preconditioner actually used, per-refinement trace |

`converged` means every column has `relres ≤ tol` **or** `relres ≤ 4·floor`, that is,
it is solved to float64 working precision. Below its floor no float64 evaluation, SuperLU
included, can certify a column's residual: SuperLU's own answer to the 15k plate's
uniform-load case has relres 8.3e-10 against a floor of 7.7e-10. `tolMet` is the strict
test.

Options:

| option | default | meaning |
|---|---|---|
| `maxIter` | max(2000, 2n) | PCG iterations per refinement |
| `x0` | — | initial guess |
| `precond` | `'fsai2'` | `'fsai2'`, `'fsai'` or `'jacobi'` |
| `eta` | 1e-4 | inner tolerance |
| `maxRefine` | 15 | cap on refinements |
| `submitMs` | 25 | target GPU time per queue submission |
| `maxCols` | 256 | right-hand sides per GPU batch; more are solved in sequential chunks |
| `memBudget` | 256 MiB | GPU buffers per batch |

Calls are serialised. A call with a lost device takes a fresh one. Malformed CSR and a
non-positive diagonal throw. `releaseGpuBuffers(gpu)` frees the cached buffer sets.

## Phones: limits, submissions, big systems, memory

- **Limits.** `getGpu` calls `requestAdapter({powerPreference: 'high-performance'})` and
  then `requestDevice()` with **no features and no limits**, so it gets WebGPU's
  defaults. It returns `null` if the device lacks what the kernels use
  (`LIMITS_USED`):

  | limit | used | WebGPU default |
  |---|---|---|
  | `maxStorageBuffersPerShaderStage` | 4 | 8 |
  | `maxUniformBuffersPerShaderStage` | 1 | 12 |
  | `maxComputeInvocationsPerWorkgroup` | 256 | 256 |
  | `maxComputeWorkgroupSizeX` | 256 | 256 |
  | `maxComputeWorkgroupStorageSize` | 8 KiB | 16 KiB |

  - Each CSR matrix is one packed u32 buffer, which keeps every kernel at 4 storage
    buffers.
  - Buffer sizes are checked against `maxStorageBufferBindingSize` (128 MiB default)
    and `maxBufferSize` (256 MiB) before any work.
  - Dispatches never exceed 65535 workgroups per dimension (2-D grids beyond).
  - No dynamic offsets, textures or timestamp queries are used.
- **Submission length.** The iterations between two read-backs (at most 32) are split
  into queue submissions of about `submitMs` = 25 ms of *measured* GPU time each. A
  new buffer set starts with one iteration per submission until it is timed. On the
  RTX 3090 an iteration of the app's plate call takes ~0.25 ms, so a submission holds
  ~8 ms of work. A GPU 20x slower gets ~5 iterations per submission. Each submission
  is ≥ 7 dispatches.
- **Big systems.** Size checks run before any set-up. If the packed matrix exceeds
  the binding limit, or the minimal buffer set exceeds the budget, `solveSPD` returns
  at once with `{converged: false, reason: 'system too large for this GPU (...)'}`,
  `x = x0` (or 0) and its residual. The `too_big` test takes 1-2 ms.
  - The preconditioner steps down (fsai2 → fsai → jacobi) when its factor does not fit
    (`precond_stepdown`).
  - Many right-hand sides are chunked to fit.
  - Buffers are created inside `out-of-memory` and `validation` error scopes, and any
    GPU failure mid-solve frees the buffers and returns the best iterate with
    `converged: false`. A lost device is replaced on the next call (`device_lost`).
- **Memory.** One buffer set holds:
  - 7 vectors of n·kp float32 (x, r, p, q, z, t, and the x read-back);
  - the packed A, G and Gᵀ;
  - the partial sums.

  **The app's plate call (n = 4352, k = 123 → kp = 124) uses 19.2 MiB** (20,160,224 B):
  vectors 14.4 MiB, A 1.1 MiB, G and Gᵀ 3.2 MiB, partials 0.5 MiB. At most two buffer
  sets stay cached, and the older ones are freed *before* a new set is allocated. The
  peak with one other set cached (the grillage's, 3.8 MiB) is therefore about 23 MiB.
  The 15k plate with k = 90 uses 53 MiB. `releaseGpuBuffers(gpu)` frees everything.

## Method

1. **Symmetric Jacobi scaling.** `As = S A S` with `S = diag(1/√a_ii)`. The model's
   diagonals span 1e-2..4e9: displacements, rotations and 1e6x penalty springs.
2. **FSAI preconditioner** (factorized sparse approximate inverse). A lower-triangular
   `G` with `G As Gᵀ ≈ I` and unit diagonal, set up on the CPU in float64. Each row of
   G is `L⁻ᵀe_last` for the dense Cholesky `L` of `As[P,P]`. The default pattern `P` is
   the lower triangle of the pattern of `As²` (`'fsai2'`). When consecutive dofs of a
   node have nested patterns, `L` is extended by one row instead of being refactored,
   2.5x faster. Applying it needs two SpMVs (`z = Gᵀ(G r)`) and no triangular solves.
3. **Inner solve (GPU, float32).** Batched PCG on `As` for all k columns at once:
   - The k vectors are stored row-major, padded to kp, and read as
     `f32`/`vec2`/`vec4` lanes.
   - One CSR SpMM thread per (row, column group).
   - Dot products are fused into the SpMMs with workgroup tree reductions. `r·z` is
     computed as ‖G r‖².
   - A second, tiny pass computes the per-column α, β and active flags **on the GPU**.
     Nothing is read back per iteration.
   - Seven dispatches per iteration. The host reads the scalars back only between
     stretches of ≤ 32 iterations, whose length is predicted from the observed
     convergence rate.
   - Columns that reach `eta` freeze (α = 0). Breakdowns (`p·Ap ≤ 0`) and zero columns
     freeze too.
4. **Outer loop (CPU, float64).** Iterative refinement `r = b − A x`, `d ≈ As⁻¹(S r/ν)`
   (normalised per column), `x += ν S d`.
   - Every step is taken. The best iterate per column is kept and returned.
   - A column stops at `tol`, at its float64 floor, or after 2 refinements that don't
     halve its residual.
5. **Caching.**
   - Pipelines are cached per lane layout.
   - GPU buffers are cached per (n, nnz, nnz(G), kp), two sets.
   - The prepared matrix (scaling, packed A, G, Gᵀ) is cached for the last 2 matrices,
     matched by content. Solving the same K again skips set-up and upload.

## Results

### RTX 3090 (Windows Chrome 155, headless; `--browser win-chrome --bigk all --compare`)

Times are wall-clock ms inside the worker. `plate_model` reuses the prepared matrix of
`plate_four` (the same K), so its set-up shows 0; a cold call adds ~50 ms.

| case | n | k | precond | PCG its | refs | max relres | max error | f64 floor | set-up ms | total ms | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace2d_30x30_k1 | 900 | 1 | fsai2 | 100 | 3 | 3.6e-13 | 2.6e-14 | 4.3e-14 | 6 | 41 | pass |
| laplace2d_30x30_k2 | 900 | 2 | fsai2 | 100 | 3 | 5.1e-13 | 1.8e-11 | 4.3e-14 | 0 | 30 | pass |
| laplace2d_30x30_k3 | 900 | 3 | fsai2 | 100 | 3 | 5.1e-13 | 1.8e-11 | 4.3e-14 | 0 | 29 | pass |
| laplace2d_30x30_k5 | 900 | 5 | fsai2 | 100 | 3 | 4.6e-13 | 8.9e-12 | 4.3e-14 | 0 | 26 | pass |
| laplace2d_30x30_k33 | 900 | 33 | fsai2 | 100 | 3 | 7.3e-13 | 3.6e-11 | 4.3e-14 | 0 | 33 | pass |
| laplace2d_80x80_k4 | 6400 | 4 | fsai2 | 164 | 3 | 1.4e-12 | 7.0e-11 | 3.0e-13 | 9 | 57 | pass |
| beam1d_ne100 | 202 | 3 | fsai2 | 1304 | 5 | 9.4e-10 | 1.9e-10 | 3.0e-10 | 0 | 276 | pass |
| beam1d_ne200 | 402 | 3 | fsai2 | 4970 | 5 | 1.6e-09 | 5.6e-11 | 2.8e-09 | 0 | 986 | pass |
| beam1d_ne400 | 802 | 3 | fsai2 | 7552 | 4 | 1.0e+00 | 1.0e+00 | 4.8e-11 | 1 | 1273 | info |
| plate_four | 4352 | 8 | fsai2 | 426 | 3 | 4.4e-10 | 2.0e-11 | 4.3e-12 | 51 | 121 | pass |
| plate_four[2:5] (cached) | 4352 | 3 | fsai2 | 423 | 3 | 2.4e-10 | 8.6e-12 | 3.8e-12 | 0 | 61 | pass |
| plate_four (warm x0) | 4352 | 8 | fsai2 | 337 | 3 | 6.0e-12 | 4.9e-12 | 4.3e-12 | 0 | 64 | pass |
| grillage_four | 4720 | 1 | fsai2 | 524 | 3 | 5.0e-11 | 3.1e-12 | 8.9e-13 | 40 | 99 | pass |
| plate_model | 4352 | 123 | fsai2 | 585 | 4 | 7.3e-10 | 2.3e-11 | 3.8e-11 | 0 | 273 | pass |
| plate_model (warm) | 4352 | 123 | fsai2 | 585 | 4 | 7.3e-10 | 2.3e-11 | 3.8e-11 | 0 | 287 | pass |
| plate_big | 14992 | 90 | fsai2 | 3444 | 6 | 9.9e-10 | 6.6e-11 | 7.7e-10 | 144 | 2257 | pass |
| laplace2d k=40 in 3 batches | 576 | 40 | fsai2 | 296 | 3 | 7.1e-13 | 2.1e-11 | 2.8e-14 | 0 | 48 | pass |
| laplace2d, zero column | 576 | 3 | fsai2 | 100 | 3 | 5.1e-13 | 6.1e-14 | 2.8e-14 | 0 | 14 | pass |
| laplace2d, exact x0 | 576 | 2 | fsai2 | 0 | 0 | 0.0e+00 | 0.0e+00 | 2.8e-14 | 0 | 0 | pass |
| laplace2d_30x30_k4 | 900 | 4 | jacobi | 172 | 3 | 7.8e-13 | 3.1e-11 | 4.3e-14 | 0 | 28 | pass |
| laplace2d, int64 indices | 576 | 2 | fsai2 | 100 | 3 | 5.1e-13 | 4.8e-12 | 2.8e-14 | 0 | 15 | pass |
| laplace2d after device loss | 576 | 2 | fsai2 | 100 | 3 | 5.1e-13 | 4.8e-12 | 2.8e-14 | 0 | 37 | pass |
| plate_four, 1 MiB budget | 4352 | 8 | none | 0 | 0 | 1.0e+00 | 1.0e+00 | 0.0e+00 | 0 | 2 | pass |
| plate_four[0:1], 3.5 MiB budget | 4352 | 1 | fsai | 883 | 3 | 4.8e-10 | 1.3e-11 | 3.9e-12 | 12 | 155 | pass |
| laplace2d, 1 ms submissions | 576 | 2 | fsai2 | 96 | 3 | 5.1e-13 | 4.8e-12 | 2.8e-14 | 0 | 13 | pass |
| negative diagonal |  |  |  |  |  |  |  |  |  |  | pass (threw) |
| malformed CSR |  |  |  |  |  |  |  |  |  |  | pass (threw) |

### SwiftShader (Playwright Chromium 140, the default run): correctness

Same plan, with `plate_big` limited to 8 of its 90 load cases (all 90 pass too: 3496
iterations, relres 9.7e-10, error 6.2e-11, 24 minutes of CPU emulation). Iteration
counts differ slightly from the RTX 3090 because float32 rounding differs between
implementations.

| case | k | precond | PCG its | refs | max relres | max error | result |
|---|---|---|---|---|---|---|---|
| laplace2d_30x30_k1 | 1 | fsai2 | 100 | 3 | 3.5e-13 | 2.6e-14 | pass |
| laplace2d_30x30_k2 | 2 | fsai2 | 100 | 3 | 3.6e-13 | 5.4e-12 | pass |
| laplace2d_30x30_k3 | 3 | fsai2 | 100 | 3 | 3.6e-13 | 5.4e-12 | pass |
| laplace2d_30x30_k5 | 5 | fsai2 | 100 | 3 | 4.2e-13 | 1.7e-11 | pass |
| laplace2d_30x30_k33 | 33 | fsai2 | 100 | 3 | 7.5e-13 | 3.6e-11 | pass |
| laplace2d_80x80_k4 | 4 | fsai2 | 168 | 3 | 1.4e-12 | 5.4e-11 | pass |
| beam1d_ne100 | 3 | fsai2 | 1000 | 4 | 1.9e-10 | 4.5e-11 | pass |
| beam1d_ne200 | 3 | fsai2 | 5159 | 5 | 2.2e-09 | 6.9e-11 | pass |
| beam1d_ne400 | 3 | fsai2 | 19856 | 14 | 1.0e+00 | 1.0e+00 | info |
| plate_four | 8 | fsai2 | 428 | 3 | 4.4e-10 | 2.0e-11 | pass |
| plate_four[2:5] (cached) | 3 | fsai2 | 420 | 3 | 2.0e-10 | 9.8e-12 | pass |
| plate_four (warm x0) | 8 | fsai2 | 341 | 3 | 6.5e-12 | 4.8e-12 | pass |
| grillage_four | 1 | fsai2 | 519 | 3 | 8.8e-11 | 6.4e-12 | pass |
| plate_model | 123 | fsai2 | 589 | 4 | 7.2e-10 | 2.5e-11 | pass |
| plate_model (warm) | 123 | fsai2 | 589 | 4 | 7.2e-10 | 2.5e-11 | pass |
| plate_big[0:8] | 8 | fsai2 | 3427 | 6 | 6.0e-10 | 2.2e-11 | pass |
| laplace2d k=40 in 3 batches | 40 | fsai2 | 296 | 3 | 7.7e-13 | 3.6e-11 | pass |
| laplace2d, zero column | 3 | fsai2 | 100 | 3 | 5.0e-13 | 6.1e-14 | pass |
| laplace2d, exact x0 | 2 | fsai2 | 0 | 0 | 0.0e+00 | 0.0e+00 | pass |
| laplace2d_30x30_k4 | 4 | jacobi | 172 | 3 | 7.7e-13 | 4.7e-11 | pass |
| laplace2d, int64 indices | 2 | fsai2 | 100 | 3 | 5.0e-13 | 4.8e-12 | pass |
| laplace2d after device loss | 2 | fsai2 | 100 | 3 | 5.0e-13 | 4.8e-12 | pass |
| plate_four, 1 MiB budget | 8 | none | 0 | 0 | 1.0e+00 | 1.0e+00 | pass |
| plate_four[0:1], 3.5 MiB budget | 1 | fsai | 863 | 3 | 5.2e-10 | 1.3e-11 | pass |
| laplace2d, 1 ms submissions | 2 | fsai2 | 96 | 3 | 5.0e-13 | 4.8e-12 | pass |
| negative diagonal |  |  |  |  |  |  | pass (threw) |
| malformed CSR |  |  |  |  |  |  | pass (threw) |

### Against SuperLU in Pyodide (RTX 3090 machine, same browser and worker)

`kigumi.linalg.factor(K).solve(B)` under Pyodide 314.0.7 (best of 3) on exactly the
same systems:

| system | GPU solver total ms (set-up / GPU / CPU f64) | PCG its | SuperLU in Pyodide ms (factor + solve) | GPU ÷ SuperLU |
|---|---|---|---|---|
| the app's plate call: field-four, n = 4352, k = 123 | 273 (0 / 128 / 140) | 585 | 62 (23 + 40) | 4.4x |
| same matrix, k = 8 | 121 (51 / 55 / 13) | 426 | 25 (22 + 3) | 4.9x |
| grillage, n = 4720, k = 1 | 99 (40 / 56 / 2) | 524 | 29 (29 + 1) | 3.4x |
| 15k-dof plate, k = 90 | 2257 (144 / 1622 / 478) | 3444 | 242 (121 + 122) | 9.3x |

The float64 refinement on the CPU (residuals and updates for all k columns) is half of
the GPU path's time at k = 123. The GPU part alone (128 ms) is still 2x SuperLU's total.
Per-iteration GPU cost on the RTX 3090 (`bench`, plate_four, FSAI level 2):

| k | 1 | 8 | 64 | 256 |
|---|---|---|---|---|
| ms per iteration | 0.10 | 0.12 | 0.15 | 0.35 |

That is latency-bound up to k ≈ 64: seven dispatches at ~15 µs each.

## What works for which matrix

Condition numbers of the scaled operators (scipy, `eigsh`):

| matrix | n | cond(As) (Jacobi) | cond·2⁻²⁴ | cond(G₁AsG₁ᵀ) | cond(G₂AsG₂ᵀ) |
|---|---|---|---|---|---|
| plate_four / plate_model | 4 352 | 8.7e4 | 5.2e-3 | 5.9e3 | 1.1e3 |
| grillage_four | 4 720 | 7.6e4 | 4.5e-3 | 8.1e3 | 1.7e3 |
| plate_big (h = 15 mm) | 14 992 | 3.7e6 | 0.22 | 3.0e5 | 5.3e4 |
| beam, 100 elements | 202 | 2.3e6 | 0.14 | 5.3e5 | — |
| beam, 200 elements | 402 | 3.6e7 | 2.2 | 8.4e6 | — |
| beam, 400 elements | 802 | 5.8e8 | 35 | 1.4e8 | — |

PCG iterations and refinements to relres ≤ 1e-9 on the RTX 3090 (η = 1e-4):

| matrix | Jacobi (plain f32 CG) | FSAI level 1 | **FSAI level 2 (default)** |
|---|---|---|---|
| plate_four (k = 8) | 3987 its / 4 refs, 388 ms | 932 / 3, 115 ms | **426 / 3, 121 ms** (51 ms set-up) |
| grillage_four (k = 1) | 3028 / 3, 284 ms | 1036 / 3, 123 ms | **524 / 3, 99 ms** (40 ms set-up) |
| plate_big (k = 8, SwiftShader) | — | 7566 / 6 | **3427 / 6** |

Findings:

- **Plain Jacobi-PCG in float32 does not stagnate on either model matrix.** With float64
  refinement it reaches 1e-9; it just needs ~3000-4000 iterations. Once scaled, the
  "hard" grillage is no harder than the plate.
- **FSAI is the improvement that pays.** Level 1 (pattern of tril(A), nnz(G) = 0.5·nnz(A))
  cuts iterations 3-4x. Level 2 (tril(A²), nnz(G) = 1.4·nnz(A)) cuts them another 2x for
  about the same total SpMV work. Set-up in JS (V8):

  | | plate_four | grillage_four | plate_big |
  |---|---|---|---|
  | level 1 | 27 ms | 10 ms | 40 ms |
  | level 2 | 58 ms | 40 ms | 154 ms |

  (the browser runs measured 40-87 ms at 4.4-4.7k dof). For one cold solve with few
  right-hand sides, level 1 is as fast; for many right-hand sides or a reused matrix,
  level 2 is.
- **Block-Jacobi** scaling with node blocks (supervariables found from the pattern) barely
  helps: cond 8.7e4 → 6.1e4, 10% fewer iterations. Not used.
- **Inner tolerance η = 1e-4** is best for the 4.5k-dof systems: 1e-3 needs a fourth
  refinement, and 1e-2 needs ~25% more iterations. For the 15k plate, η = 1e-3 saves 7%.
- **What a refinement gains** is about `min(η, cond(As)·2⁻²⁴/20)` in the residual:
  - the 4.5k systems gain 1e-3..1e-4 per refinement, so 3-4 refinements reach 1e-9;
  - the 15k plate (cond·2⁻²⁴ = 0.22) gains only 1e-2, so it needs 6.

  The float32 *operator* limits this, not the preconditioner. f32 PCG's recursive
  residual keeps falling, while the true residual of its iterate stalls at about
  cond(As)·2⁻²⁴.
- **Beams** are the stiff end. With 200 elements (cond·2⁻²⁴ ≈ 2) refinement still
  converges: 5 refinements, error 6e-11, stopping at its float64 floor of 2.8e-9. The
  first f32 correction *raises* the unscaled residual 18x while cutting the error to
  1e-3, because the 2-norm of a mixed force/moment residual is dominated by the stiff
  rotation rows. Hence refinement steps are not judged by the residual norm. An early
  version rejected such steps and stalled. With 400 elements (cond·2⁻²⁴ ≈ 35) the f32
  operator's rounding exceeds its smallest eigenvalue: two of the three load cases
  diverge and the solver returns `converged: false` (best iterate x0 = 0). It is kept
  in the plan as an informational case.

## Tried and not adopted

- **Rejecting refinement steps that raise ‖r‖₂.** This is wrong for stiff mixed-unit
  systems, as described above.
- **Minimal-residual / GCR outer steps** (α minimising ‖r‖₂ over the new direction and
  the last m ones). Far worse on the beams: the 2-norm is dominated by the stiff rows,
  so α ≈ 0.
- **Flexible CG in float64 around the f32 solve** (FCG(m), energy-norm steps with the
  GPU solve as a variable preconditioner):
  - no gain on the model systems;
  - 3x *slower* on the 200-element beam;
  - but it converges on the 400-element beam (error 3e-10 after 30 outer steps with
    m = 1) where plain refinement crawls.

  It is the remedy if stiffer systems ever appear (cond(As)·2⁻²⁴ ≳ 1), as a fallback
  when refinement stalls. Not implemented.
- **Reliable updates** (QUDA-style mixed-precision CG: one f32 CG run, residual replaced
  by the float64 one every 10x, keeping the search direction):
  - 10-20% fewer iterations than restarted refinement on the 4.5k systems;
  - catastrophic once the f32 residual gap is large: 30000 iterations to reach 6e-7 on
    the 15k plate (restarted refinement: 7400 to reach 6e-10), and stuck on the beams.
- **Compensated (two-float) arithmetic** in WGSL was not needed for accuracy. It would
  extend the f32 reach at 2-4x the cost per iteration, and WGSL does not guarantee
  unfused, unreordered float operations on every backend.

## Limitations and notes for the integration

- **Speed.** SuperLU in Pyodide wins on the app's systems (see above). What could close
  part of the gap:
  - fusing the reductions into neighbouring kernels when the grid is small (7 → 5
    dispatches);
  - moving the float64 residuals to fewer, larger passes;
  - a multilevel preconditioner, the only thing that would stop the iteration count
    growing with the mesh.

  None of these changes the scaling argument for 2D fourth-order plates.
- **Float32 reach.** Plain mixed-precision refinement needs cond(As)·2⁻²⁴ below about 1.
  kigumi meshes its plates at h = max(15 mm, span/34), about 34 elements across, so
  cond(As) ≈ 1e5 and the margin is large. plate_big (h = 15 mm) is the stress end, at
  0.22. A plate twice as fine would not converge, nor would long, finely divided beams.
- **The float64 floor.** For the 15k plate's uniform load the floor is 7.7e-10, so
  `tol = 1e-9` is at the edge of what any float64 solver can show. `converged` accepts
  `relres ≤ 4·floor`; `tolMet` is the strict flag.
- **Set-up per matrix.** The FSAI set-up runs on the CPU, about 40-90 ms at 4.5k dof in
  the browser. The Python side currently routes only `solve_many` (the plate) through
  the accelerator; the grillage's contact loop calls `linalg.factor` directly.
- **`iters` counts lock-step GPU iterations,** including those in which some columns
  were already frozen.
