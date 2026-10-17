// Materials for kigumi parts: procedural wood grain (rings around the part's local
// +X axis, which is the grain direction), charred cedar crackle, glass and the
// small rubber / silicone bits.  One material per part so each board gets its own
// pith offset and tint; all wood materials share a single compiled program.

import * as THREE from 'three';

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
// shader code
// ---------------------------------------------------------------------------
const WOOD_PARS = /* glsl */`
varying vec3 vKgLocal;
uniform vec3 uKgEarly;
uniform vec3 uKgLate;
uniform vec4 uKgGrain;   // rings/m, latewood contrast, fibre contrast, figure contrast
uniform vec4 uKgSeed;    // pith offset y z (m), pith tilt, random 0..1
uniform vec3 uKgTint;
uniform float uKgChar;
uniform float uKgLateRough;

float kgHash(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float kgNoise(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(kgHash(i + vec3(0.0, 0.0, 0.0)), kgHash(i + vec3(1.0, 0.0, 0.0)), f.x),
                 mix(kgHash(i + vec3(0.0, 1.0, 0.0)), kgHash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
             mix(mix(kgHash(i + vec3(0.0, 0.0, 1.0)), kgHash(i + vec3(1.0, 0.0, 1.0)), f.x),
                 mix(kgHash(i + vec3(0.0, 1.0, 1.0)), kgHash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float kgFbm(vec3 p) {
  return 0.5333 * kgNoise(p) + 0.2667 * kgNoise(p * 2.03 + 3.1)
       + 0.1333 * kgNoise(p * 4.07 + 7.7) + 0.0667 * kgNoise(p * 8.11 + 1.3);
}
// 3D Voronoi: distance to the nearest cell border (two passes, so it has no
// creases inside a cell), a per-cell random value and the distance to the seed
vec3 kgVoronoi(vec3 x) {
  vec3 n = floor(x);
  vec3 f = fract(x);
  vec3 mg = vec3(0.0);
  vec3 mr = vec3(0.0);
  float md = 8.0;
  for (int k = -1; k <= 1; k++)
  for (int j = -1; j <= 1; j++)
  for (int i = -1; i <= 1; i++) {
    vec3 g = vec3(float(i), float(j), float(k));
    vec3 c = n + g;
    vec3 r = g + vec3(kgHash(c), kgHash(c + 17.31), kgHash(c + 41.97)) - f;
    float d = dot(r, r);
    if (d < md) { md = d; mr = r; mg = g; }
  }
  float be = 8.0;
  for (int k = -1; k <= 1; k++)
  for (int j = -1; j <= 1; j++)
  for (int i = -1; i <= 1; i++) {
    vec3 g = mg + vec3(float(i), float(j), float(k));
    vec3 c = n + g;
    vec3 r = g + vec3(kgHash(c), kgHash(c + 17.31), kgHash(c + 41.97)) - f;
    vec3 dr = r - mr;
    if (dot(dr, dr) > 1e-5) be = min(be, dot(0.5 * (mr + r), normalize(dr)));
  }
  return vec3(be, kgHash(n + mg + 7.77), sqrt(md));
}

// p: local position in metres, +X along the grain.  Returns linear albedo and
// writes a bump height (m) and a roughness offset.
vec3 kgWood(vec3 p, out float h, out float dr) {
  h = 0.0;
  dr = 0.0;
  if (uKgChar > 0.5) {
    // yakisugi: charred "alligator" scales, elongated along the grain, dark cracks
    vec3 c = p * vec3(52.0, 118.0, 118.0) + uKgSeed.xyz * 40.0;     // scales ~19 x 8 mm
    vec3 v = kgVoronoi(c);
    float e = v.x;                                                   // distance to a crack
    float detail = 1.0 - smoothstep(0.35, 1.1, fwidth(c.x) + fwidth(c.y) + fwidth(c.z));
    float crack = (1.0 - smoothstep(0.018, 0.05 + fwidth(e), e)) * detail;
    float soot = kgFbm(p * vec3(9.0, 34.0, 34.0) + uKgSeed.w * 9.0);
    float scale = mix(1.0, 0.8 + 0.55 * v.y, detail);
    vec3 col = uKgEarly * scale * (0.8 + 0.4 * soot);
    col = mix(col, uKgEarly * 0.1, crack);
    dr = mix(-0.12 * detail, 0.16, crack);
    h = detail * smoothstep(0.0, 0.14, e) * 0.00016;                 // scale edges round into the cracks
    return col * uKgTint;
  }
  // pith runs roughly along +X, offset outside the board and slightly tilted
  vec2 yz = p.yz + uKgSeed.xy + p.x * vec2(uKgSeed.z, -0.55 * uKgSeed.z);
  float wob = kgFbm(vec3(p.x * 3.0, p.y * 15.0, p.z * 15.0) + uKgSeed.w * 37.0) - 0.5;
  float r = length(yz) + wob * 0.010;
  float rings = r * uKgGrain.x;
  // uneven growth: ring widths wander over a few rings
  rings += 1.7 * (kgNoise(vec3(rings * 0.13, uKgSeed.w * 7.0, 0.5)) - 0.5);
  float fw = fwidth(rings);
  float aa = 1.0 - smoothstep(0.12, 0.5, fw);
  float f = fract(rings);
  float edge = max(0.025, fw);
  float lw = smoothstep(0.50, 0.90, f) * (1.0 - smoothstep(1.0 - edge, 1.0, f));
  float rv = kgHash(vec3(floor(rings), uKgSeed.w * 13.0, 1.7));
  lw *= 0.6 + 0.8 * rv;
  lw = mix(0.27, lw, aa);
  float fig = kgNoise(vec3(rings * 0.21, p.x * 0.9, uKgSeed.w * 9.0)) - 0.5;
  fig *= 1.0 - smoothstep(2.0, 6.0, fw);
  vec3 fq = vec3(p.x * 7.0, p.y * 1100.0, p.z * 1100.0);
  float fib = 0.62 * kgNoise(fq) + 0.38 * kgNoise(fq * vec3(1.7, 2.3, 2.3) + 5.1) - 0.5;
  float fwF = fwidth(fq.y) + fwidth(fq.z);
  fib *= 1.0 - smoothstep(0.35, 1.3, fwF);
  vec3 col = mix(uKgEarly, uKgLate, clamp(lw * uKgGrain.y, 0.0, 1.0));
  col *= 1.0 + 2.0 * uKgGrain.w * fig;
  col *= 1.0 + 2.0 * uKgGrain.z * fib;
  dr = (lw - 0.27) * uKgLateRough;
  h = (lw * 0.000012 + fib * 0.000006) * aa;
  return col * uKgTint;
}

// bump from a procedural height (Mikkelsen surface gradient, unnormalised so the
// height is in physical units)
vec3 kgBump(vec3 surfPos, vec3 n, float hgt, float faceDir) {
  vec3 sx = dFdx(surfPos);
  vec3 sy = dFdy(surfPos);
  vec3 r1 = cross(sy, n);
  vec3 r2 = cross(n, sx);
  float det = dot(sx, r1) * faceDir;
  vec2 dh = vec2(dFdx(hgt), dFdy(hgt));
  vec3 g = sign(det) * (dh.x * r1 + dh.y * r2);
  return normalize(abs(det) * n - g);
}
`;

function patchWood(shader, uniforms) {
  Object.assign(shader.uniforms, uniforms);
  shader.vertexShader = shader.vertexShader
    .replace('#include <common>', '#include <common>\nvarying vec3 vKgLocal;')
    .replace('#include <begin_vertex>', '#include <begin_vertex>\nvKgLocal = position;');
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', '#include <common>\n' + WOOD_PARS)
    .replace('#include <color_fragment>',
      '#include <color_fragment>\nfloat kgH; float kgDR;\ndiffuseColor.rgb = kgWood(vKgLocal, kgH, kgDR);')
    .replace('#include <roughnessmap_fragment>',
      '#include <roughnessmap_fragment>\nroughnessFactor = clamp(roughnessFactor + kgDR, 0.05, 1.0);')
    .replace('#include <normal_fragment_maps>',
      '#include <normal_fragment_maps>\nnormal = kgBump(-vViewPosition, normal, kgH, faceDirection);');
}

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
  const uniforms = {
    uKgEarly: { value: early },
    uKgLate: { value: late },
    uKgGrain: { value: new THREE.Vector4(sp.rings || 0, sp.late || 0, sp.fibre || 0, sp.figure || 0) },
    uKgSeed: { value: seed },
    uKgTint: { value: tint },
    uKgChar: { value: sp.char ? 1 : 0 },
    uKgLateRough: { value: sp.char ? 0 : 0.10 },
  };
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: sp.rough,
    metalness: 0,
    clearcoat: sp.char ? 0.03 : 0.07,
    clearcoatRoughness: 0.5,
    specularIntensity: sp.char ? 0.45 : 0.5,
  });
  mat.name = 'wood:' + species;
  mat.onBeforeCompile = (shader) => patchWood(shader, uniforms);
  mat.customProgramCacheKey = () => 'kigumi-wood-1';
  mat.userData.kigumi = { species, uniforms };
  return mat;
}

export function makeGlassMaterial(kind, thicknessM) {
  const acrylic = kind === 'acrylic';
  const mat = new THREE.MeshPhysicalMaterial({
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
  const mat = new THREE.MeshPhysicalMaterial({
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
    return new THREE.MeshPhysicalMaterial({ color: OTHER.rubber.hex, roughness: 0.78, metalness: 0, specularIntensity: 0.5 });
  }
  if (material === 'silicone') {
    // thin frosted bumper: mostly the wood below, a little milky
    const c = new THREE.Color(underHex || OTHER.silicone.hex).lerp(new THREE.Color(OTHER.silicone.hex), 0.45);
    return new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.3, metalness: 0, specularIntensity: 0.8 });
  }
  if (SPECIES[material] || WOOD_KINDS.has(kind)) {
    return makeWoodMaterial(material, id, fallbackHex);
  }
  return new THREE.MeshPhysicalMaterial({ color: fallbackHex || '#9a9083', roughness: 0.6, metalness: 0 });
}
