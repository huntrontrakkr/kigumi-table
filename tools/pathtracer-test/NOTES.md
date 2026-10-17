# Path tracer for "Render photo": notes and measurements

`site/render/` replaces the offline Blender Cycles photo with a progressive path tracer
in WebGPU compute, running on the visitor's GPU.

| file | what |
|---|---|
| `pathtracer.js` | API, device, camera, studio lights, dispatch scheduling, budget, PNG |
| `scene.js` | visible meshes of `viewer.model` -> world triangles, normals, material table |
| `bvh.js` | binned-SAH BVH (adaptive bin counts), flattened with child boxes in the parent |
| `build.js` | BVH + GPU buffer packing (runs in a module worker made from a blob) |
| `kernel.js` | WGSL: path tracing megakernel, resolve pass (Neutral tone map, sRGB, dither) |

## API

```js
import { renderPhoto, preparePathTracer, lastStats } from './render/pathtracer.js';

preparePathTracer(viewer);              // optional, early: device + pipeline compile
const blob = await renderPhoto({
  viewer,                               // window.kigumi.viewer
  view: 'hero',                         // hero | top | low | side | front | current
  width, height,                        // default 1600 x 1100 (phones 1200 x 825)
  samples: 256,                         // per pixel, the most it will take
  budgetMs: 15000,                      // whole call; stops early with what it has
  maxTriangles,                         // default 3M (phones 500k): over it -> null
  onProgress: (frac, spp) => {},        // frac: max(samples done, budget spent)
  onPreview: (bitmap, spp) => {},       // optional: the converging image as an ImageBitmap
  previewMs: 1000, previewWidth: 800,   //   about once a second, about 800 px wide
  signal,                               // AbortSignal: rejects with AbortError
  studio,                               // optional, see STUDIO
});
if (!blob) { /* lastStats.reason: no WebGPU, too many triangles, image too large,
                scene too large for this GPU, out of GPU memory, GPU device lost,
                error -> use the raster viewer.photo() */ }
```

It never rejects except on abort.  Renders are queued (one at a time).  Phones (the UA's
mobile flag or name, iPadOS, or a coarse pointer on a screen of 900 px or less, as the
viewer's low-power test) get the smaller defaults; `MOBILE` and `DEFAULTS` are exported.  The viewer's camera
is placed with `viewer.frame(false, pose)` at the photo's aspect and restored afterwards.
`lastStats` holds the timings of the last render.  `disposePathTracer()` frees the device.

## What it renders

- Scene: visible meshes of `viewer.model` (groups / array materials, indexed or not,
  instanced meshes, negative scales), per-mesh world -> local transforms for the wood.
- Wood: the TSL patterns of `site/materials.js` ported to WGSL with the same hash, noise,
  fbm and cells, the same constants (rings, latewood, figure, fibres; yakisugi crackle).
  `fwidth()` becomes the footprint of a ray cone on the surface (anisotropic along the
  slant), so detail fades where the live view fades it.  Ring and char bumps are analytic.
  BSDF: Lambert + one GGX lobe (F0 = 0.04 x specularIntensity + clear coat), VNDF sampling.
- Glass / acrylic: smooth dielectric, exact Fresnel, Beer absorption from attenuationColor /
  attenuationDistance, the colour as a tint per interface; a sheet's edges share its
  medium, so the green edge comes from light running along the sheet (total internal
  reflection), as in real float glass.  Shadow rays pass straight through glass with its
  transmittance (exact for flat sheets).
- Studio: a warm key light (distant disc, 1.5 degrees: the lattice still draws its shadow,
  with penumbrae widening away from the parts) from `viewer.keyDir`, a cool fill opposite
  it, an environment of the background colour (brighter overhead), and an infinite paper
  floor whose albedo is solved so the open floor equals the backdrop: the horizon
  disappears, as in the live view.  The camera sees the backdrop straight through glass;
  reflections and indirect light see the studio.  Khronos PBR Neutral, sRGB, dither.
- Integrator: next-event estimation with MIS (power heuristic), 6 rough bounces with
  Russian roulette after 3, up to 32 path events in glass, indirect clamp 4, no reflective
  caustics off glass (faint, all fireflies).  Owen-scrambled Sobol points (Burley 2020)
  with the second dimension computed by a zeta transform: RMSE 2.18 vs 3.62 for hashed
  random numbers at 64 spp against a 1024 spp reference (8-bit units), same speed.

## WebGPU limits and features

No optional features.  Requested: `maxStorageBufferBindingSize` and `maxBufferSize`, each
min(adapter limit, 1 GiB), only for big scenes on desktop GPUs; if the device request
fails it retries with no required limits (defaults), then falls back on three.js's device.
Used: 1 bind group; 3 storage buffers + 1 uniform in the trace pass (accumulation, BVH
nodes, one geometry buffer), 2 + 1 in the resolve pass; 8x8 workgroups (64 invocations);
no workgroup memory; at most ceil(w/8) x ceil(h/8) workgroups per dispatch.  All within
the compatibility-mode minimums (4 storage buffers, 128 invocations, 128 MiB bindings).
Buffers: accumulation w*h*16 B, output and readback w*h*4 B each, geometry 64 B per
triangle, BVH about 33 B per triangle; all are checked against the device limits first.

## Scheduling (phones, watchdogs)

Each pass visits the 8-row blocks of the image in a strided order (stride coprime with the
block count), and each dispatch takes the next n blocks.  Every dispatch holds the same mix
of floor, wood and glass, so its time is predictable; a feedback loop sizes it at about 30
ms (target 40), from 2 blocks up to whole frames with several samples per dispatch.  The
per-pixel sample count lives in the accumulation buffer's w, so the budget can stop the
render between any two dispatches.  The longest glass paths set every dispatch's latency
floor, hence the 32-event cap (paths cut there end on the studio radiance: no visible
change).  The BVH build runs in a worker, overlapping pipeline compilation.

## Measurements (RTX 3090, Chrome 155 on Windows, headless, real GPU)

Default design: 172,692 triangles, 105 meshes; 1600 x 1100, 256 spp.

| view | total | tracing | ms / sample | dispatches (max ms) |
|---|---|---|---|---|
| hero | 3.9 s | 3.5 s | 13.7 | 142 (37) |
| top | 4.9 s | 4.5 s | 17.5 | 270 (30) |
| low | 2.8 s | 2.3 s | 9.1 | 106 (35) |
| side | 2.7 s | 2.3 s | 8.9 | 107 (34) |
| front | 3.5 s | 3.2 s | 12.7 | 151 (35) |

Previews (`onPreview`, 800 x 550 every second) cost about 0.1 s over a whole render.

Hero breakdown: extraction 34 ms, BVH 214 ms (worker), packing 14 ms, GPU setup 24 ms,
tracing 3.5 s (about 130 M camera paths/s), resolve + readback + PNG 74 ms.  800 x 550,
256 spp: 1.5 s.  The first render in a page also compiles the pipelines (0.5-0.7 s,
overlapped with the BVH build; `preparePathTracer` can do it earlier).

Stress: the table plus a 1,000,000-triangle torus knot (1,172,692 triangles): 6.9 s total,
extraction 187 ms, BVH 1.84 s (worker, 611,632 nodes, depth 24), packing 87 ms, tracing
4.4 s (17.3 ms / sample), 156 MB of GPU buffers.

BVH build under Node 24 (`bvh-bench.mjs`, dumps of the real scenes): 172,692 triangles
240-300 ms, 1,172,692 triangles 1.8-2.0 s.

Slow GPUs: an emulated GPU with a tenth of the cores (every thread also traces 9 other
pixels spread over the image; `slow.js`) reaches 185 spp at 1200 x 825 (the phone
default) and 102 spp at 1600 x 1100 within the 15 s budget (77 and 122 ms / sample);
dispatches stay at 45-48 ms for the cold first one and 35 ms or less after.  Projected on
a real 10x slower GPU: 256 spp at 1600 x 1100 would take about 35-40 s; with the default
budget it stops at 15 s with about 100 spp (1600 x 1100) or about 180 spp (1200 x 825).

## Testing

- `serve.py` (port 8791): no-cache server for `/test/` (this folder) and `/render/`
  (`site/render/`), with CORS and a POST sink into `out/`.
- `cdp.sh <script.js> '<args>'`: opens the app (port 8790) in the Windows host's headless
  Chrome over CDP, waits for the engine, runs the script's `run(args)`.
- Scripts: `render.js` (renders, batches), `ref.js` (raster reference at the same size and
  framing), `closeup.js` (raster vs path traced from one camera), `edge.js` (caps, abort,
  device loss, recovery, budget), `slow.js` (slow-GPU emulation), `stress.js` (1M
  triangles), `variants.js` (kernel experiments), `heatmap.js` (BVH visits per pixel),
  `dump.js` + `bvh-bench.mjs` (BVH timing on real scenes), `bvh-test.mjs` (correctness).
- `crop.py`: crops, region stats, diffs, contact sheets (project .venv PIL).
- `out/sample-hero.png`: 800 x 550, 256 spp (committed); everything else in `out/` is ignored.

## Limitations, not done

- Not run on actual phones (Android Chrome, iOS Safari 26); limits and scheduling were
  checked against the specs and an emulated slow GPU only.  Phone detection by UA.
- Glass is smooth (the edge material's roughness 0.2 is ignored); silicone is opaque.
- No fibre bump (it fades at photo resolution); one merged GGX lobe for specular and clear
  coat; no multiple-scattering energy compensation.
- No denoiser: budget-limited renders on slow GPUs keep some grain.
- Framing is `viewer.frame()`'s (rest pose): exploded views are cropped like the raster photo.
- The BVH build is single-threaded (1.8 s for 1.2M triangles, in a worker).
- The hover highlight (emissive) is ignored.
