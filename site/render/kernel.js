// WGSL for the path tracer: one megakernel (camera ray -> path with next-event
// estimation, MIS, Russian roulette) accumulating into a float buffer, and a
// resolve pass (average, Khronos PBR Neutral tone mapping, sRGB, dither, RGBA8).
//
// Wood patterns are ports of the TSL node graphs in ../materials.js (same hash,
// value noise, fbm and cellular functions, same constants); the screen-space
// fwidth() of the live view becomes the footprint of a ray cone on the surface.

export const PARAMS_BYTES = 16 * 16;

export const traceWGSL = /* wgsl */`
const PI = 3.14159265358979;
const INV_PI = 0.318309886183791;
const NONE = 0xffffffffu;
const BIG = 1e30;
const KIND_PLASTIC = 0u;
const KIND_WOOD = 1u;
const KIND_CHAR = 2u;
const KIND_GLASS = 3u;
const MAT = 8u;        // vec4s per material
const MAX_DIFFUSE = 6u;

struct Params {
  size: vec4u,        // width, height, first sample index, samples this dispatch
  band: vec4u,        // first row, rows, max path events, seed
  camPos: vec4f,      // xyz, pixel angle (rad)
  camRight: vec4f,    // right * tan(half horizontal fov)
  camUp: vec4f,       // up * tan(half vertical fov)
  camFwd: vec4f,
  keyDir: vec4f,      // direction to the key light, w: cos(cone half-angle)
  keyL: vec4f,        // key radiance, w: probability of sampling the key (else fill)
  fillDir: vec4f,
  fillL: vec4f,
  envH: vec4f,        // environment radiance at the horizon
  envZ: vec4f,        // ... at the zenith
  backdrop: vec4f,    // radiance seen by the camera past everything; w: indirect clamp
  floorA: vec4f,      // floor albedo, w: floor height
  misc: vec4f,        // x: bump strength
  bounds: vec4f,      // unused
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(3) var<storage, read> tris: array<vec4f>;
@group(0) @binding(4) var<storage, read> shade: array<vec4u>;
@group(0) @binding(5) var<storage, read> mats: array<vec4f>;
@group(0) @binding(6) var<storage, read> xf: array<vec4f>;

// ---------------------------------------------------------------- sampling
// Owen-scrambled, shuffled Sobol points (Burley 2020, "Practical Hash-based Owen
// Scrambling"): every dimension pair gets its own index shuffle and scramble.
fn lk(x: u32, seed: u32) -> u32 {
  var v = x;
  v ^= v * 0x3d20adeau;
  v += seed;
  v *= (seed >> 16u) | 1u;
  v ^= v * 0x05526c56u;
  v ^= v * 0x53a22864u;
  return v;
}
fn nus(x: u32, seed: u32) -> u32 { return reverseBits(lk(reverseBits(x), seed)); }
fn hash1(x: u32) -> u32 {
  var v = x;
  v ^= v >> 16u; v *= 0x7feb352du;
  v ^= v >> 15u; v *= 0x846ca68bu;
  v ^= v >> 16u;
  return v;
}
fn sobol1(i: u32) -> u32 {          // second Sobol dimension (x + 1)
  var r = 0u;
  var v = 0x80000000u;
  var j = i;
  loop {
    if (j == 0u) { break; }
    if ((j & 1u) != 0u) { r ^= v; }
    j >>= 1u;
    v ^= v >> 1u;
  }
  return r;
}
fn rand2(si: u32, seed: u32) -> vec2f {
  let i = nus(si, seed);
  let a = nus(reverseBits(i), hash1(seed ^ 0xa511e9b3u));
  let b = nus(sobol1(i), hash1(seed ^ 0x63d83595u));
  return vec2f(vec2u(a, b) >> vec2u(8u)) * (1.0 / 16777216.0);
}
fn rand1(si: u32, seed: u32) -> f32 {
  let i = nus(si, seed);
  let a = nus(reverseBits(i), hash1(seed ^ 0xa511e9b3u));
  return f32(a >> 8u) * (1.0 / 16777216.0);
}
// seed of dimension set k at path event 'depth'
fn dimSeed(pix: u32, depth: u32, k: u32) -> u32 { return hash1(pix ^ ((depth * 8u + k + 1u) * 0x9e3779b9u)); }

fn tent(u: f32) -> f32 {
  let x = 2.0 * u;
  return select(1.0 - sqrt(max(2.0 - x, 0.0)), sqrt(x) - 1.0, x < 1.0);
}

// orthonormal basis around n (Duff et al. 2017)
fn basisT(n: vec3f) -> vec3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  return vec3f(1.0 + s * n.x * n.x * a, s * n.x * n.y * a, -s * n.x);
}
fn basisB(n: vec3f) -> vec3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  return vec3f(n.x * n.y * a, s + n.y * n.y * a, -n.y);
}

fn maxc(v: vec3f) -> f32 { return max(v.x, max(v.y, v.z)); }
fn lum(v: vec3f) -> f32 { return dot(v, vec3f(0.2126, 0.7152, 0.0722)); }

// Ray Tracing Gems ch. 6: offset a ray origin off a surface along its normal
fn offsetRay(p: vec3f, n: vec3f) -> vec3f {
  let oi = vec3i(256.0 * n);
  let pi = vec3f(
    bitcast<f32>(bitcast<i32>(p.x) + select(oi.x, -oi.x, p.x < 0.0)),
    bitcast<f32>(bitcast<i32>(p.y) + select(oi.y, -oi.y, p.y < 0.0)),
    bitcast<f32>(bitcast<i32>(p.z) + select(oi.z, -oi.z, p.z < 0.0)));
  return select(pi, p + n * (1.0 / 65536.0), abs(p) < vec3f(1.0 / 32.0));
}

fn octDecode(u: u32) -> vec3f {
  let e = unpack2x16snorm(u);
  var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

// ---------------------------------------------------------------- traversal
struct Hit { t: f32, u: f32, v: f32, tri: u32 }

fn slab(lo: vec3f, hi: vec3f, inv: vec3f, oi: vec3f, tmax: f32) -> f32 {
  let t0 = lo * inv - oi;
  let t1 = hi * inv - oi;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  let a = max(max(tn.x, tn.y), max(tn.z, 0.0));
  let b = min(min(tf.x, tf.y), min(tf.z, tmax));
  return select(BIG, a, a <= b);
}

fn safeInv(d: vec3f) -> vec3f {
  return 1.0 / select(d, vec3f(1e-20), abs(d) < vec3f(1e-20));
}

fn closest(o: vec3f, d: vec3f, tmax: f32) -> Hit {
  var h = Hit(tmax, 0.0, 0.0, NONE);
  let inv = safeInv(d);
  let oi = o * inv;
  var stack: array<u32, 64>;
  var sp = 0u;
  var ni = 0u;
  for (var guard = 0u; guard < 200000u; guard++) {
    let b = 4u * ni;
    let a0 = nodes[b];
    let a1 = nodes[b + 1u];
    let c0 = nodes[b + 2u];
    let c1 = nodes[b + 3u];
    let t0 = slab(a0.xyz, a1.xyz, inv, oi, h.t);
    let t1 = slab(c0.xyz, c1.xyz, inv, oi, h.t);
    let r0 = bitcast<u32>(a0.w);
    let n0 = bitcast<u32>(a1.w);
    let r1 = bitcast<u32>(c0.w);
    let n1 = bitcast<u32>(c1.w);
    var go0 = t0 < BIG;
    var go1 = t1 < BIG;
    if (go0 && n0 > 0u) {
      for (var k = r0; k < r0 + n0; k++) { hitTri(k, o, d, &h); }
      go0 = false;
    }
    if (go1 && n1 > 0u) {
      for (var k = r1; k < r1 + n1; k++) { hitTri(k, o, d, &h); }
      go1 = false;
    }
    if (go0 && go1) {
      let near1 = t1 < t0;
      stack[sp] = select(r1, r0, near1);
      sp++;
      ni = select(r0, r1, near1);
    } else if (go0) {
      ni = r0;
    } else if (go1) {
      ni = r1;
    } else {
      if (sp == 0u) { break; }
      sp--;
      ni = stack[sp];
    }
  }
  return h;
}

fn hitTri(k: u32, o: vec3f, d: vec3f, h: ptr<function, Hit>) {
  let v0 = tris[3u * k].xyz;
  let e1 = tris[3u * k + 1u].xyz;
  let e2 = tris[3u * k + 2u].xyz;
  let pv = cross(d, e2);
  let id = 1.0 / dot(e1, pv);
  let s = o - v0;
  let u = dot(s, pv) * id;
  let q = cross(s, e1);
  let v = dot(d, q) * id;
  let t = dot(e2, q) * id;
  if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0 && t < (*h).t) {
    (*h).t = t; (*h).u = u; (*h).v = v; (*h).tri = k;
  }
}

fn fresnelDielectric(cosi: f32, eta: f32) -> f32 {
  let s2 = eta * eta * (1.0 - cosi * cosi);
  if (s2 >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - s2);
  let rs = (eta * cosi - ct) / (eta * cosi + ct);
  let rp = (cosi - eta * ct) / (cosi + eta * ct);
  return 0.5 * (rs * rs + rp * rp);
}

// Shadow ray: transmittance towards a distant light.  Opaque hits block; glass
// sheets let light through straight (exact for a flat sheet, which shifts a ray
// sideways without turning it), losing two Fresnel reflections, tint and absorption.
fn transmit(o: vec3f, d: vec3f) -> vec3f {
  var T = vec3f(1.0);
  let inv = safeInv(d);
  let oi = o * inv;
  var stack: array<u32, 64>;
  var sp = 0u;
  var ni = 0u;
  for (var guard = 0u; guard < 200000u; guard++) {
    let b = 4u * ni;
    let a0 = nodes[b];
    let a1 = nodes[b + 1u];
    let c0 = nodes[b + 2u];
    let c1 = nodes[b + 3u];
    let t0 = slab(a0.xyz, a1.xyz, inv, oi, BIG);
    let t1 = slab(c0.xyz, c1.xyz, inv, oi, BIG);
    let r0 = bitcast<u32>(a0.w);
    let n0 = bitcast<u32>(a1.w);
    let r1 = bitcast<u32>(c0.w);
    let n1 = bitcast<u32>(c1.w);
    var go0 = t0 < BIG;
    var go1 = t1 < BIG;
    for (var c = 0u; c < 2u; c++) {
      let goc = select(go1, go0, c == 0u);
      let nc = select(n1, n0, c == 0u);
      if (!goc || nc == 0u) { continue; }
      let rc = select(r1, r0, c == 0u);
      for (var k = rc; k < rc + nc; k++) {
        var h = Hit(BIG, 0.0, 0.0, NONE);
        hitTri(k, o, d, &h);
        if (h.tri == NONE) { continue; }
        let m = bitcast<u32>(tris[3u * k].w);
        let kind = u32(mats[MAT * m].x);
        if (kind != KIND_GLASS) { return vec3f(0.0); }
        let ng = normalize(cross(tris[3u * k + 1u].xyz, tris[3u * k + 2u].xyz));
        let ci = abs(dot(ng, d));
        let m1 = mats[MAT * m + 1u];
        let m2 = mats[MAT * m + 2u];
        let eta = 1.0 / m1.w;
        let ct = sqrt(max(1.0 - eta * eta * (1.0 - ci * ci), 1e-4));
        T *= (1.0 - fresnelDielectric(ci, eta)) * sqrt(m1.xyz) * exp(-m2.xyz * (0.5 * m2.w / ct));
        if (maxc(T) < 1e-4) { return vec3f(0.0); }
      }
      if (c == 0u) { go0 = false; } else { go1 = false; }
    }
    if (go0 && go1) {
      stack[sp] = r1;
      sp++;
      ni = r0;
    } else if (go0) {
      ni = r0;
    } else if (go1) {
      ni = r1;
    } else {
      if (sp == 0u) { break; }
      sp--;
      ni = stack[sp];
    }
  }
  return T;
}

// ---------------------------------------------------------------- wood (materials.js)
fn kgHash(p0: vec3f) -> f32 {
  let p = fract(p0 * 0.3183099 + 0.1) * 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
fn kgNoise(x: vec3f) -> f32 {
  let i = floor(x);
  let f0 = fract(x);
  let f = f0 * f0 * (f0 * -2.0 + 3.0);
  return mix(
    mix(mix(kgHash(i), kgHash(i + vec3f(1.0, 0.0, 0.0)), f.x),
        mix(kgHash(i + vec3f(0.0, 1.0, 0.0)), kgHash(i + vec3f(1.0, 1.0, 0.0)), f.x), f.y),
    mix(mix(kgHash(i + vec3f(0.0, 0.0, 1.0)), kgHash(i + vec3f(1.0, 0.0, 1.0)), f.x),
        mix(kgHash(i + vec3f(0.0, 1.0, 1.0)), kgHash(i + vec3f(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
fn kgFbm(p: vec3f) -> f32 {
  return kgNoise(p) * 0.5333 + kgNoise(p * 2.03 + 3.1) * 0.2667
       + kgNoise(p * 4.07 + 7.7) * 0.1333 + kgNoise(p * 8.11 + 1.3) * 0.0667;
}

struct Cells { e: f32, cv: f32, g: vec3f }   // (F2 - F1) / 2, cell value, gradient of e (cell units)
fn kgCells(x: vec3f) -> Cells {
  let n = floor(x);
  let f = fract(x);
  var d1 = 8.0;
  var d2 = 8.0;
  var cv = 0.0;
  var r1 = vec3f(1.0);
  var r2 = vec3f(1.0);
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let g = vec3f(f32(i), f32(j), f32(k));
        let c = n + g;
        let r = g + vec3f(kgHash(c), kgHash(c + 17.31), kgHash(c + 41.97)) - f;
        let d = dot(r, r);
        if (d < d1) {
          d2 = d1; r2 = r1;
          cv = kgHash(c + 7.77);
          d1 = d; r1 = r;
        } else if (d < d2) {
          d2 = d; r2 = r;
        }
      }
    }
  }
  let f1 = sqrt(d1);
  let f2 = sqrt(d2);
  return Cells(0.5 * (f2 - f1), cv, 0.5 * (r1 / max(f1, 1e-6) - r2 / max(f2, 1e-6)));
}

fn sstep(a: f32, b: f32, x: f32) -> f32 { return smoothstep(a, b, x); }
fn dsstep(a: f32, b: f32, x: f32) -> f32 {
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return 6.0 * t * (1.0 - t) / (b - a);
}

struct Wood { col: vec3f, dr: f32, gh: vec3f }   // albedo, roughness offset, bump gradient (local)

// fwidth() of a function with local gradient g over the footprint (fa, fb)
fn fwv(g: vec3f, fa: vec3f, fb: vec3f) -> f32 { return abs(dot(g, fa)) + abs(dot(g, fb)); }

fn woodEval(p: vec3f, fa: vec3f, fb: vec3f, m: u32, isChar: bool) -> Wood {
  let early = mats[MAT * m + 3u];
  let late = mats[MAT * m + 4u].xyz;
  let grain = mats[MAT * m + 5u];
  let seed = mats[MAT * m + 6u];
  let tint = mats[MAT * m + 7u].xyz;
  var w: Wood;
  if (isChar) {
    // yakisugi: charred scales elongated along the grain, dark cracks
    let sc = vec3f(52.0, 118.0, 118.0);
    let c = p * sc + seed.xyz * 40.0;
    let fwc = fwv(vec3f(sc.x, 0.0, 0.0), fa, fb) + fwv(vec3f(0.0, sc.y, 0.0), fa, fb) + fwv(vec3f(0.0, 0.0, sc.z), fa, fb);
    let detail = 1.0 - sstep(0.35, 1.1, fwc);
    let soot = kgFbm(p * vec3f(9.0, 34.0, 34.0) + seed.w * 9.0);
    var e = 0.5;
    var cv = 0.5;
    var ge = vec3f(0.0);
    if (detail > 0.0) {
      let v = kgCells(c);
      e = v.e; cv = v.cv; ge = v.g * sc;
    }
    let crack = (1.0 - sstep(0.018, fwv(ge, fa, fb) + 0.05, e)) * detail;
    let scale = mix(1.0, cv * 0.55 + 0.8, detail);
    w.col = mix(early.xyz * scale * (soot * 0.4 + 0.8), early.xyz * 0.1, crack) * tint;
    w.dr = mix(detail * -0.12, 0.16, crack);
    w.gh = ge * (detail * dsstep(0.0, 0.14, e) * 0.00016);
    return w;
  }
  // the pith runs roughly along +X, outside the board and slightly tilted
  let tilt = vec2f(seed.z, seed.z * -0.55);
  let yz = p.yz + seed.xy + tilt * p.x;
  let ryz = length(yz);
  let dyz = yz / max(ryz, 1e-6);
  let gR = grain.x * vec3f(dot(dyz, tilt), dyz.x, dyz.y);     // gradient of the ring count
  let fw = fwv(gR, fa, fb);
  let aa = 1.0 - sstep(0.12, 0.5, fw);
  let figFade = 1.0 - sstep(2.0, 6.0, fw);
  var lw = 0.27;
  var fig = 0.0;
  var gh = vec3f(0.0);
  if (aa > 0.0 || figFade > 0.0) {
    let wob = kgFbm(vec3f(p.x * 3.0, p.y * 15.0, p.z * 15.0) + seed.w * 37.0) - 0.5;
    let r = ryz + wob * 0.010;
    let rings0 = r * grain.x;
    // uneven growth: ring widths wander over a few rings
    let rings = rings0 + (kgNoise(vec3f(rings0 * 0.13, seed.w * 7.0, 0.5)) - 0.5) * 1.7;
    if (aa > 0.0) {
      let f = fract(rings);
      let edge = max(fw, 0.025);
      let s1 = sstep(0.50, 0.90, f);
      let s2 = 1.0 - sstep(1.0 - edge, 1.0, f);
      let rv = kgHash(vec3f(floor(rings), seed.w * 13.0, 1.7));
      let k = rv * 0.8 + 0.6;
      lw = mix(0.27, s1 * s2 * k, aa);
      // bump: latewood stands proud by 12 um (the live view's h = lw * 1.2e-5 * aa)
      let dlw = (dsstep(0.50, 0.90, f) * s2 - s1 * dsstep(1.0 - edge, 1.0, f)) * k * aa;
      gh = gR * (dlw * 1.2e-5 * aa);
    }
    fig = (kgNoise(vec3f(rings * 0.21, p.x * 0.9, seed.w * 9.0)) - 0.5) * figFade;
  }
  let fwF = fwv(vec3f(0.0, 1100.0, 0.0), fa, fb) + fwv(vec3f(0.0, 0.0, 1100.0), fa, fb);
  let fibFade = 1.0 - sstep(0.35, 1.3, fwF);
  var fib = 0.0;
  if (fibFade > 0.0) {
    let fq = vec3f(p.x * 7.0, p.y * 1100.0, p.z * 1100.0);
    fib = (kgNoise(fq) * 0.62 + kgNoise(fq * vec3f(1.7, 2.3, 2.3) + 5.1) * 0.38 - 0.5) * fibFade;
  }
  w.col = mix(early.xyz, late, clamp(lw * grain.y, 0.0, 1.0))
    * (grain.w * 2.0 * fig + 1.0) * (grain.z * 2.0 * fib + 1.0) * tint;
  w.dr = (lw - 0.27) * early.w;
  w.gh = gh;
  return w;
}

// ---------------------------------------------------------------- BSDF
struct Surf {
  p: vec3f,
  ng: vec3f,        // geometric normal, on the side the ray came from
  ns: vec3f,        // shading normal, same side
  kd: vec3f,        // diffuse albedo (already weighted)
  f0: vec3f,        // specular reflectance at normal incidence
  alpha: f32,       // GGX roughness (roughness^2)
  ps: f32,          // probability of sampling the specular lobe
}

fn schlick(f0: vec3f, c: f32) -> vec3f {
  let m = 1.0 - clamp(c, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * (m2 * m2 * m);
}
fn ggxD(nh: f32, a2: f32) -> f32 {
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn ggxLambda(c: f32, a2: f32) -> f32 {
  let c2 = max(c * c, 1e-7);
  return 0.5 * (sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0);
}

struct Eval { f: vec3f, pdf: f32 }      // f * cos(wi), pdf of the lobe mixture

fn bsdfEval(s: Surf, wo: vec3f, wi: vec3f) -> Eval {
  let co = dot(s.ns, wo);
  let ci = dot(s.ns, wi);
  if (co <= 0.0 || ci <= 0.0) { return Eval(vec3f(0.0), 0.0); }
  var f = s.kd * (ci * INV_PI);
  var pdf = (1.0 - s.ps) * ci * INV_PI;
  if (s.ps > 0.0) {
    let h = normalize(wo + wi);
    let nh = max(dot(s.ns, h), 0.0);
    let a2 = max(s.alpha * s.alpha, 1e-7);
    let D = ggxD(nh, a2);
    let lo = ggxLambda(co, a2);
    let li = ggxLambda(ci, a2);
    f += schlick(s.f0, dot(wo, h)) * (D / (4.0 * co * (1.0 + lo + li)));
    pdf += s.ps * D / (4.0 * co * (1.0 + lo));
  }
  return Eval(f, pdf);
}

// visible-normal sampling of GGX (Dupuy & Benyoub 2023, spherical caps), local frame
fn sampleVNDF(wo: vec3f, alpha: f32, u: vec2f) -> vec3f {
  let wi = normalize(vec3f(wo.xy * alpha, wo.z));
  let phi = 2.0 * PI * u.x;
  let z = (1.0 - u.y) * (1.0 + wi.z) - wi.z;
  let st = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let h = vec3f(st * cos(phi), st * sin(phi), z) + wi;
  return normalize(vec3f(h.xy * alpha, max(h.z, 1e-6)));
}

struct Sample { wi: vec3f, w: vec3f, pdf: f32, spec: bool }

fn bsdfSample(s: Surf, wo: vec3f, u: vec2f, ul: f32) -> Sample {
  let t = basisT(s.ns);
  let b = basisB(s.ns);
  var wi: vec3f;
  let spec = ul < s.ps;
  if (spec) {
    let wl = vec3f(dot(wo, t), dot(wo, b), dot(wo, s.ns));
    let h = sampleVNDF(wl, max(s.alpha, 1e-4), u);
    let hw = t * h.x + b * h.y + s.ns * h.z;
    wi = reflect(-wo, hw);
  } else {
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    wi = t * (r * cos(phi)) + b * (r * sin(phi)) + s.ns * sqrt(max(0.0, 1.0 - u.x));
  }
  let e = bsdfEval(s, wo, wi);
  if (e.pdf <= 0.0) { return Sample(wi, vec3f(0.0), 0.0, spec); }
  return Sample(wi, e.f / e.pdf, e.pdf, spec);
}

// ---------------------------------------------------------------- lights
fn envL(d: vec3f) -> vec3f { return mix(P.envH.xyz, P.envZ.xyz, clamp(d.y, 0.0, 1.0)); }
fn conePdf(cosMax: f32) -> f32 { return 1.0 / (2.0 * PI * (1.0 - cosMax)); }

// a distant disc light (cone of directions) hit by a BSDF-sampled ray
fn coneLe(d: vec3f, axis: vec4f, L: vec3f, sel: f32, bsdfPdf: f32) -> vec3f {
  if (dot(d, axis.xyz) < axis.w || sel <= 0.0) { return vec3f(0.0); }
  if (bsdfPdf <= 0.0) { return L; }
  let pl = sel * conePdf(axis.w);
  return L * (bsdfPdf * bsdfPdf / (bsdfPdf * bsdfPdf + pl * pl));
}

fn sampleCone(axis: vec3f, cosMax: f32, u: vec2f) -> vec3f {
  let ct = 1.0 - u.x * (1.0 - cosMax);
  let st = sqrt(max(0.0, 1.0 - ct * ct));
  let phi = 2.0 * PI * u.y;
  return normalize(basisT(axis) * (st * cos(phi)) + basisB(axis) * (st * sin(phi)) + axis * ct);
}

// next-event estimation: one of the two lights, power heuristic against the BSDF
fn directLight(s: Surf, wo: vec3f, u: vec2f, ul: f32) -> vec3f {
  let useKey = ul < P.keyL.w;
  let axis = select(P.fillDir, P.keyDir, useKey);
  let L = select(P.fillL.xyz, P.keyL.xyz, useKey);
  let sel = select(1.0 - P.keyL.w, P.keyL.w, useKey);
  if (sel <= 0.0) { return vec3f(0.0); }
  let wi = sampleCone(axis.xyz, axis.w, u);
  if (dot(wi, s.ng) <= 0.0) { return vec3f(0.0); }
  let e = bsdfEval(s, wo, wi);
  if (e.pdf <= 0.0) { return vec3f(0.0); }
  let T = transmit(offsetRay(s.p, s.ng), wi);
  if (maxc(T) <= 0.0) { return vec3f(0.0); }
  let pl = sel * conePdf(axis.w);
  return e.f * L * T * (pl / (pl * pl + e.pdf * e.pdf));
}

// ---------------------------------------------------------------- integrator
fn clampInd(c: vec3f, indirect: bool) -> vec3f {
  let m = maxc(c);
  if (indirect && m > P.backdrop.w) { return c * (P.backdrop.w / m); }
  return c;
}

fn radiance(px: u32, py: u32, si: u32, pix: u32) -> vec3f {
  let j = rand2(si, hash1(pix ^ 0x5bd1e995u));
  let sx = (f32(px) + 0.5 + tent(j.x)) / f32(P.size.x) * 2.0 - 1.0;
  let sy = 1.0 - (f32(py) + 0.5 + tent(j.y)) / f32(P.size.y) * 2.0;
  var d = normalize(P.camFwd.xyz + P.camRight.xyz * sx + P.camUp.xyz * sy);
  var o = P.camPos.xyz;
  var L = vec3f(0.0);
  var beta = vec3f(1.0);
  var specPath = true;          // only mirror / refraction events since the camera
  var mirrored = false;         // ... and one of them a reflection
  var lastPdf = 0.0;            // BSDF pdf at the last rough vertex; 0: delta event or camera
  var inGlass = false;
  var sigma = vec3f(0.0);
  var coneW = 0.0;              // ray cone: width and spread, for texture footprints
  var coneS = P.camPos.w;
  var bounces = 0u;
  for (var depth = 0u; depth < P.band.z; depth++) {
    let h = closest(o, d, BIG);
    var tFloor = BIG;
    if (d.y < 0.0) { tFloor = (P.floorA.w - o.y) / d.y; }
    let onTri = h.tri != NONE && h.t < tFloor;
    if (!onTri && tFloor >= BIG) {
      // straight through glass from the camera: the backdrop paper; reflections
      // and everything lit indirectly: the studio around it
      var Le = select(envL(d), P.backdrop.xyz, specPath && !mirrored);
      // the lights: seen in the glass from the camera, or reached from a rough
      // surface (MIS with next-event estimation); not after rough -> mirror
      // (reflective caustics off the glass: faint, and all fireflies)
      if (depth > 0u && (specPath || lastPdf > 0.0)) {
        Le += coneLe(d, P.keyDir, P.keyL.xyz, P.keyL.w, lastPdf);
        Le += coneLe(d, P.fillDir, P.fillL.xyz, 1.0 - P.keyL.w, lastPdf);
      }
      L += clampInd(beta * Le, bounces > 0u);
      break;
    }
    let t = select(tFloor, h.t, onTri);
    if (inGlass) { beta *= exp(-sigma * t); }
    let width = coneW + coneS * t;

    var s: Surf;
    if (onTri) {
      let k = h.tri;
      let v0 = tris[3u * k];
      let e1 = tris[3u * k + 1u].xyz;
      let e2 = tris[3u * k + 2u].xyz;
      let p = v0.xyz + e1 * h.u + e2 * h.v;
      let ngo = normalize(cross(e1, e2));             // outward
      let sh = shade[k];
      var ns = normalize(octDecode(sh.x) * (1.0 - h.u - h.v) + octDecode(sh.y) * h.u + octDecode(sh.z) * h.v);
      let m = sh.w & 0xffffu;
      let mesh = sh.w >> 16u;
      let m0 = mats[MAT * m];
      let kind = u32(m0.x);
      let entering = dot(d, ngo) < 0.0;
      let ng = select(-ngo, ngo, entering);           // faces the incoming ray
      if (dot(ns, ng) < 0.0) { ns = -ns; }

      if (kind == KIND_GLASS) {
        // smooth dielectric: reflect or refract by Fresnel; absorption inside
        let m1 = mats[MAT * m + 1u];
        let eta = select(m1.w, 1.0 / m1.w, entering);
        var n = ns;
        if (dot(n, d) >= 0.0) { n = ng; }
        let ci = -dot(d, n);
        let F = fresnelDielectric(ci, eta);
        let u = rand1(si, dimSeed(pix, depth, 3u));
        if (u < F) {
          d = reflect(d, n);
          if (dot(d, ng) <= 0.0) { d = reflect(d, ng); }
          o = offsetRay(p, ng);
          lastPdf = 0.0;
          mirrored = true;
        } else {
          d = refract(d, n, eta);
          if (dot(d, ng) >= 0.0) { break; }
          o = offsetRay(p, -ng);
          beta *= sqrt(m1.xyz);
          inGlass = entering;
          sigma = mats[MAT * m + 2u].xyz;
        }
        coneW = width;
        continue;
      }

      // opaque material
      var rough = m0.y;
      var albedo = mats[MAT * m + 1u].xyz;
      if (kind == KIND_WOOD || kind == KIND_CHAR) {
        let r0 = xf[3u * mesh];
        let r1 = xf[3u * mesh + 1u];
        let r2 = xf[3u * mesh + 2u];
        let pl = vec3f(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
        // footprint of the ray cone on the surface, stretched along the ray's slant
        let cv = max(abs(dot(d, ng)), 0.05);
        var td = d - ng * dot(d, ng);
        let tl = length(td);
        td = select(basisT(ng), td / tl, tl > 1e-6);
        let fa = td * (width / cv);
        let fb = cross(ng, td) * width;
        let fal = vec3f(dot(r0.xyz, fa), dot(r1.xyz, fa), dot(r2.xyz, fa));
        let fbl = vec3f(dot(r0.xyz, fb), dot(r1.xyz, fb), dot(r2.xyz, fb));
        let w = woodEval(pl, fal, fbl, m, kind == KIND_CHAR);
        albedo = w.col;
        rough = clamp(rough + w.dr, 0.05, 1.0);
        // bump: tilt the shading normal against the height gradient (world space)
        let gw = (r0.xyz * w.gh.x + r1.xyz * w.gh.y + r2.xyz * w.gh.z) * P.misc.x;
        let nb = normalize(ns - (gw - ns * dot(gw, ns)));
        if (dot(nb, ng) > 0.05) { ns = nb; }
      }
      let metal = m0.w;
      s.p = p;
      s.ng = ng;
      s.ns = ns;
      s.alpha = rough * rough;
      s.f0 = mix(vec3f(m0.z), albedo, metal);
      let fo = schlick(s.f0, dot(ns, -d));
      s.kd = albedo * (1.0 - metal) * (1.0 - maxc(fo));
      let es = lum(fo);
      s.ps = clamp(es / max(es + lum(s.kd), 1e-6), 0.08, 0.9);
    } else {
      // the paper floor: Lambertian, infinite
      s.p = vec3f(o.x + d.x * t, P.floorA.w, o.z + d.z * t);
      s.ng = vec3f(0.0, 1.0, 0.0);
      s.ns = s.ng;
      s.kd = P.floorA.xyz;
      s.f0 = vec3f(0.0);
      s.alpha = 1.0;
      s.ps = 0.0;
    }

    let wo = -d;
    // next-event estimation
    let ul = rand2(si, dimSeed(pix, depth, 0u));
    let uls = rand1(si, dimSeed(pix, depth, 2u));
    L += clampInd(beta * directLight(s, wo, ul, uls), bounces > 0u);

    // continue the path
    let ub = rand2(si, dimSeed(pix, depth, 1u));
    let ulobe = rand1(si, dimSeed(pix, depth, 3u));
    let smp = bsdfSample(s, wo, ub, ulobe);
    if (smp.pdf <= 0.0 || dot(smp.wi, s.ng) <= 0.0) { break; }
    beta *= smp.w;
    lastPdf = smp.pdf;
    specPath = false;
    coneW = width;
    coneS += select(1.0, 2.0 * s.alpha, smp.spec);
    o = offsetRay(s.p, s.ng);
    d = smp.wi;
    bounces++;
    if (bounces >= MAX_DIFFUSE) { break; }
    if (bounces >= 3u) {
      let q = min(0.95, maxc(beta));
      if (rand1(si, dimSeed(pix, depth, 4u)) >= q) { break; }
      beta /= q;
    }
  }
  return L;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y + P.band.x;
  if (px >= P.size.x || gid.y >= P.band.y || py >= P.size.y) { return; }
  let idx = py * P.size.x + px;
  let pix = hash1(idx ^ hash1(P.band.w));
  var sum = vec3f(0.0);
  for (var k = 0u; k < P.size.w; k++) {
    let c = radiance(px, py, P.size.z + k, pix);
    // drop NaN / inf samples (they would poison the pixel for good)
    let e = (bitcast<vec3u>(c) >> vec3u(23u)) & vec3u(0xffu);
    if (all(e != vec3u(0xffu))) { sum += c; }
  }
  accum[idx] += vec4f(sum, 0.0);
}
`;

export const resolveWGSL = /* wgsl */`
struct R { size: vec4u, k: vec4f }     // width, height, -, -; 1 / samples, exposure, dither, -
@group(0) @binding(0) var<uniform> U: R;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outPx: array<u32>;

// Khronos PBR Neutral (three.js NeutralToneMapping)
fn neutral(c0: vec3f) -> vec3f {
  var c = c0;
  let x = min(c.r, min(c.g, c.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  c -= offset;
  let peak = max(c.r, max(c.g, c.b));
  if (peak < 0.76) { return c; }
  let d = 0.24;
  let newPeak = 1.0 - d * d / (peak + d - 0.76);
  c *= newPeak / peak;
  let g = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);
  return mix(c, vec3f(newPeak), g);
}
fn srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}
fn hash1(x: u32) -> u32 {
  var v = x;
  v ^= v >> 16u; v *= 0x7feb352du;
  v ^= v >> 15u; v *= 0x846ca68bu;
  v ^= v >> 16u;
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.size.x * U.size.y) { return; }
  let c = neutral(max(accum[i].xyz * (U.k.x * U.k.y), vec3f(0.0)));
  let s = srgb(clamp(c, vec3f(0.0), vec3f(1.0))) * 255.0;
  // triangular dither of one step: no banding in the soft floor gradients
  let h = hash1(i * 3u + 1u);
  let dz = (f32(h & 0xffffu) + f32(h >> 16u)) * (1.0 / 65536.0) - 1.0;
  let v = clamp(floor(s + 0.5 + dz * U.k.z), vec3f(0.0), vec3f(255.0));
  outPx[i] = u32(v.x) | (u32(v.y) << 8u) | (u32(v.z) << 16u) | (255u << 24u);
}
`;
