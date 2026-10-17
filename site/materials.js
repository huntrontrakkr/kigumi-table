// Materials for kigumi parts: procedural wood grain (rings around the part's local
// +X axis, which is the grain direction), charred cedar crackle, glass and the
// small rubber / silicone bits.  One material per part so each board gets its own
// pith offset and tint; the patterns are TSL node graphs (three.js shading
// language), compiled to WGSL for WebGPU or GLSL for the WebGL2 fallback, and
// every board of a species shares one pipeline (only its uniforms differ).

import * as THREE from 'three/webgpu';
import {
  Fn, uniform, float, vec2, vec3, fract, floor, mix, smoothstep, length, dot, fwidth, dFdx, dFdy,
  normalize, cross, sign, abs, clamp, sqrt, min, positionLocal, positionView, normalView,
} from 'three/tsl';

// Mean sRGB colours (approximate) and figure parameters per species.
//   rings  : growth rings per metre (fine woods have more)
//   late   : latewood darkening (0 = invisible rings, 1 = strong)
//   fibre  : fine streaks along the grain
//   figure : coarse banding that follows the rings
//   rough  : base roughness; lateTint: latewood colour relative to earlywood
export const SPECIES = {
  hinoki:    { hex: '#DBBD8C', rings: 360, late: 0.36, fibre: 0.05, figure: 0.05, rough: 0.50, lateTint: [0.80, 0.70, 0.55] },
  white_oak: { hex: '#B38757', rings: 240, late: 0.55, fibre: 0.11, figure: 0.07, rough: 0.55, lateTint: [0.62, 0.50, 0.38] },
  walnut:    { hex: '#4D301F', rings: 160, late: 0.50, fibre: 0.09, figure: 0.16, rough: 0.56, lateTint: [0.56, 0.46, 0.44] },
  cherry:    { hex: '#80472E', rings: 300, late: 0.38, fibre: 0.06, figure: 0.09, rough: 0.45, lateTint: [0.68, 0.52, 0.46] },
  ash:       { hex: '#D6BD94', rings: 190, late: 0.62, fibre: 0.09, figure: 0.05, rough: 0.55, lateTint: [0.66, 0.54, 0.40] },
  maple:     { hex: '#E6CCA1', rings: 360, late: 0.20, fibre: 0.04, figure: 0.05, rough: 0.44, lateTint: [0.84, 0.76, 0.64] },
  sugi:      { hex: '#B36E4F', rings: 170, late: 0.62, fibre: 0.06, figure: 0.11, rough: 0.58, lateTint: [0.60, 0.43, 0.36] },
  yakisugi:  { hex: '#0D0B0A', char: true, rough: 0.82 },
  wenge:     { hex: '#30211A', rings: 280, late: 0.80, fibre: 0.22, figure: 0.08, rough: 0.58, lateTint: [0.34, 0.30, 0.29] },
};

export const OTHER = {
  glass:    { hex: '#D8ECE6' },
  acrylic:  { hex: '#F4F7F8' },
  rubber:   { hex: '#151413' },
  silicone: { hex: '#E8ECEA' },
};

// Colour swatch for the UI (any material name).
export function swatch(name) {
  const s = SPECIES[name] || OTHER[name];
  return s ? s.hex : '#9a9083';
}

const WOOD_KINDS = new Set(['leg', 'post', 'beam', 'block', 'cap']);

// ---------------------------------------------------------------------------
// deterministic per-part randomness
// ---------------------------------------------------------------------------
function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
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

// ---------------------------------------------------------------------------
// patterns (TSL)
// ---------------------------------------------------------------------------
const kgHash = Fn(([p0]) => {
  const p = fract(p0.mul(0.3183099).add(0.1)).mul(17.0);
  return fract(p.x.mul(p.y).mul(p.z).mul(p.x.add(p.y).add(p.z)));
});

const kgNoise = Fn(([x]) => {
  const i = floor(x);
  const f0 = fract(x);
  const f = f0.mul(f0).mul(f0.mul(-2.0).add(3.0));
  const h = (a, b, c) => kgHash(i.add(vec3(a, b, c)));
  return mix(
    mix(mix(h(0, 0, 0), h(1, 0, 0), f.x), mix(h(0, 1, 0), h(1, 1, 0), f.x), f.y),
    mix(mix(h(0, 0, 1), h(1, 0, 1), f.x), mix(h(0, 1, 1), h(1, 1, 1), f.x), f.y), f.z);
});

const kgFbm = Fn(([p]) => kgNoise(p).mul(0.5333)
  .add(kgNoise(p.mul(2.03).add(3.1)).mul(0.2667))
  .add(kgNoise(p.mul(4.07).add(7.7)).mul(0.1333))
  .add(kgNoise(p.mul(8.11).add(1.3)).mul(0.0667)));

// cellular pattern for charred cedar: F2 - F1 (distance to the nearest crack)
// and a per-cell random value, unrolled over the 27 neighbour cells
const kgCells = Fn(([x]) => {
  const n = floor(x);
  const f = fract(x);
  const d1 = float(8.0).toVar();
  const d2 = float(8.0).toVar();
  const cv = float(0.0).toVar();
  for (let k = -1; k <= 1; k++) {
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const g = vec3(i, j, k);
        const c = n.add(g);
        const r = g.add(vec3(kgHash(c), kgHash(c.add(17.31)), kgHash(c.add(41.97)))).sub(f);
        const d = dot(r, r);
        const closer = d.lessThan(d1);
        d2.assign(closer.select(d1, min(d2, d)));
        cv.assign(closer.select(kgHash(c.add(7.77)), cv));
        d1.assign(min(d1, d));
      }
    }
  }
  return vec3(sqrt(d2).sub(sqrt(d1)), cv, sqrt(d1));
});

// Node graphs for one board: albedo (linear), bump height (m), roughness offset.
// p is the local position in metres, +X along the grain.
function woodNodes(u, char) {
  const p = positionLocal;
  if (char) {
    // yakisugi: charred "alligator" scales, elongated along the grain, dark cracks
    const c = p.mul(vec3(52.0, 118.0, 118.0)).add(u.seed.xyz.mul(40.0));
    const v = kgCells(c);
    const e = v.x.mul(0.5);
    const detail = float(1.0).sub(smoothstep(0.35, 1.1, fwidth(c.x).add(fwidth(c.y)).add(fwidth(c.z))));
    const crack = float(1.0).sub(smoothstep(0.018, fwidth(e).add(0.05), e)).mul(detail);
    const soot = kgFbm(p.mul(vec3(9.0, 34.0, 34.0)).add(u.seed.w.mul(9.0)));
    const scale = mix(float(1.0), v.y.mul(0.55).add(0.8), detail);
    const col = mix(u.early.mul(scale).mul(soot.mul(0.4).add(0.8)), u.early.mul(0.1), crack).mul(u.tint);
    const dr = mix(detail.mul(-0.12), float(0.16), crack);
    const h = detail.mul(smoothstep(0.0, 0.14, e)).mul(0.00016);
    return { col, h, dr };
  }
  // the pith runs roughly along +X, outside the board and slightly tilted
  const yz = p.yz.add(u.seed.xy).add(vec2(u.seed.z, u.seed.z.mul(-0.55)).mul(p.x));
  const wob = kgFbm(vec3(p.x.mul(3.0), p.y.mul(15.0), p.z.mul(15.0)).add(u.seed.w.mul(37.0))).sub(0.5);
  const r = length(yz).add(wob.mul(0.010));
  const rings0 = r.mul(u.grain.x);
  // uneven growth: ring widths wander over a few rings
  const rings = rings0.add(kgNoise(vec3(rings0.mul(0.13), u.seed.w.mul(7.0), 0.5)).sub(0.5).mul(1.7));
  const fw = fwidth(rings);
  const aa = float(1.0).sub(smoothstep(0.12, 0.5, fw));
  const f = fract(rings);
  const edge = fw.max(0.025);
  const lw0 = smoothstep(0.50, 0.90, f).mul(float(1.0).sub(smoothstep(float(1.0).sub(edge), 1.0, f)));
  const rv = kgHash(vec3(floor(rings), u.seed.w.mul(13.0), 1.7));
  const lw = mix(float(0.27), lw0.mul(rv.mul(0.8).add(0.6)), aa);
  const fig = kgNoise(vec3(rings.mul(0.21), p.x.mul(0.9), u.seed.w.mul(9.0))).sub(0.5)
    .mul(float(1.0).sub(smoothstep(2.0, 6.0, fw)));
  const fq = vec3(p.x.mul(7.0), p.y.mul(1100.0), p.z.mul(1100.0));
  const fwF = fwidth(fq.y).add(fwidth(fq.z));
  const fib = kgNoise(fq).mul(0.62).add(kgNoise(fq.mul(vec3(1.7, 2.3, 2.3)).add(5.1)).mul(0.38)).sub(0.5)
    .mul(float(1.0).sub(smoothstep(0.35, 1.3, fwF)));
  const col = mix(u.early, u.late, clamp(lw.mul(u.grain.y), 0.0, 1.0))
    .mul(u.grain.w.mul(2.0).mul(fig).add(1.0))
    .mul(u.grain.z.mul(2.0).mul(fib).add(1.0))
    .mul(u.tint);
  const dr = lw.sub(0.27).mul(u.lateRough);
  const h = lw.mul(0.000012).add(fib.mul(0.000006)).mul(aa);
  return { col, h, dr };
}

// bump from a procedural height (surface gradient, in physical units)
const kgBump = Fn(([hgt]) => {
  const pos = positionView;
  const n = normalView;
  const sx = dFdx(pos);
  const sy = dFdy(pos);
  const r1 = cross(sy, n);
  const r2 = cross(n, sx);
  const det = dot(sx, r1);
  const g = sign(det).mul(r1.mul(dFdx(hgt)).add(r2.mul(dFdy(hgt))));
  return normalize(abs(det).mul(n).sub(g));
});

const _c = new THREE.Color();

export function makeWoodMaterial(species, partId, fallbackHex) {
  const sp = SPECIES[species] || { hex: fallbackHex || '#B89A78', rings: 240, late: 0.4, fibre: 0.07, figure: 0.08, rough: 0.52, lateTint: [0.66, 0.55, 0.45] };
  const rand = rng(hashString(partId + ':' + species));
  const mean = _c.set(sp.hex).clone();                 // linear (ColorManagement)
  let early = mean.clone();
  let late = mean.clone();
  if (!sp.char) {
    const lt = sp.lateTint;
    late = new THREE.Color(mean.r * lt[0], mean.g * lt[1], mean.b * lt[2]);
    // keep the average close to the species colour: earlywood slightly lighter
    const m = 0.27 * sp.late;
    early = new THREE.Color(
      mean.r / (1 - m + m * lt[0]), mean.g / (1 - m + m * lt[1]), mean.b / (1 - m + m * lt[2]));
    late = new THREE.Color(early.r * lt[0], early.g * lt[1], early.b * lt[2]);
  }
  // per-board variation: brightness and a slight warm/cool shift
  const b = 0.9 + 0.2 * rand();
  const warm = (rand() - 0.5) * 0.06;
  const tint = new THREE.Vector3(b * (1 + warm), b, b * (1 - warm));
  const ang = rand() * Math.PI * 2;
  const dist = 0.18 + 0.35 * rand();                  // pith 18..53 cm away
  const seed = new THREE.Vector4(Math.cos(ang) * dist, Math.sin(ang) * dist,
    (rand() - 0.5) * 0.08, rand());
  const u = {
    early: uniform(early),
    late: uniform(late),
    grain: uniform(new THREE.Vector4(sp.rings || 0, sp.late || 0, sp.fibre || 0, sp.figure || 0)),
    seed: uniform(seed),
    tint: uniform(tint),
    lateRough: uniform(sp.char ? 0 : 0.10),
  };
  const mat = new THREE.MeshPhysicalNodeMaterial({
    color: 0xffffff,
    roughness: sp.rough,
    metalness: 0,
    clearcoat: sp.char ? 0.03 : 0.07,
    clearcoatRoughness: 0.5,
    specularIntensity: sp.char ? 0.45 : 0.5,
  });
  mat.name = 'wood:' + species;
  const w = woodNodes(u, !!sp.char);
  mat.colorNode = w.col;
  mat.roughnessNode = clamp(float(sp.rough).add(w.dr), 0.05, 1.0);
  mat.normalNode = kgBump(w.h);
  const uniforms = u;
  mat.userData.kigumi = { species, uniforms };
  return mat;
}

export function makeGlassMaterial(kind, thicknessM) {
  const acrylic = kind === 'acrylic';
  const mat = new THREE.MeshPhysicalNodeMaterial({
    color: acrylic ? 0xffffff : 0xf3faf8,
    metalness: 0,
    roughness: 0,
    transmission: 1,
    thickness: thicknessM || 0.01,
    ior: acrylic ? 1.49 : 1.52,
    attenuationColor: new THREE.Color(acrylic ? '#f6f8f8' : '#cfe6dc'),
    attenuationDistance: acrylic ? 4 : 0.45,
    specularIntensity: 1,
    envMapIntensity: 0.24,
  });
  mat.name = kind;
  return mat;
}

// Float glass edges look green: a long path through slightly iron-rich glass.
export function makeGlassEdgeMaterial(thicknessM) {
  const mat = new THREE.MeshPhysicalNodeMaterial({
    color: 0xe8f4ef,
    metalness: 0,
    roughness: 0.2,
    transmission: 1,
    thickness: 0.04,
    ior: 1.52,
    attenuationColor: new THREE.Color('#9fd0bc'),
    attenuationDistance: 0.18,
    specularIntensity: 1,
    envMapIntensity: 0.5,
  });
  mat.name = 'glass-edge';
  return mat;
}

// Material for any part; `kind`/`material` come from the GLB node extras or the
// report.  Unknown materials fall back to the GLB colour so new ones still render.
export function makePartMaterial({ id, kind, material, fallbackHex, glassThickness, underHex }) {
  if (material === 'glass' || material === 'acrylic' || kind === 'glass') {
    return makeGlassMaterial(material === 'acrylic' ? 'acrylic' : 'glass', glassThickness);
  }
  if (material === 'rubber') {
    return new THREE.MeshPhysicalNodeMaterial({ color: OTHER.rubber.hex, roughness: 0.78, metalness: 0, specularIntensity: 0.5 });
  }
  if (material === 'silicone') {
    // thin frosted bumper: mostly the wood below, a little milky
    const c = new THREE.Color(underHex || OTHER.silicone.hex).lerp(new THREE.Color(OTHER.silicone.hex), 0.45);
    return new THREE.MeshPhysicalNodeMaterial({ color: c, roughness: 0.3, metalness: 0, specularIntensity: 0.8 });
  }
  if (SPECIES[material] || WOOD_KINDS.has(kind)) {
    return makeWoodMaterial(material, id, fallbackHex);
  }
  return new THREE.MeshPhysicalNodeMaterial({ color: fallbackHex || '#9a9083', roughness: 0.6, metalness: 0 });
}
