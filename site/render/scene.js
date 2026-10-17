// Scene extraction for the path tracer: world-space triangles, vertex normals and a
// material table from a three.js object tree (the viewer's model).  Reads the
// materials made by ../materials.js: wood boards carry their procedural pattern
// uniforms in material.userData.kigumi; glass is a transmissive physical material;
// everything else is read as a dielectric "plastic" (colour, roughness, specular).
//
// No three.js import: only the objects' own fields and methods are used.

export const KIND = { PLASTIC: 0, WOOD: 1, CHAR: 2, GLASS: 3 };
export const MAT_VEC4 = 8;          // vec4s per material record
export const XF_VEC4 = 3;           // vec4s per mesh transform (world -> local rows)

// Packed material record (8 vec4, see the WGSL in kernel.js):
//   0: kind, roughness, F0 (specular reflectance at normal incidence), metalness
//   1: colour rgb (albedo; glass: transmission tint), ior
//   2: absorption sigma rgb (1/m), thickness (m)
//   3: wood early rgb, latewood roughness offset
//   4: wood late rgb, 0
//   5: grain: rings per metre, latewood contrast, fibre contrast, figure contrast
//   6: seed: pith offset y, z (m), pith tilt, random
//   7: tint rgb, 0
function convertMaterial(mat) {
  const r = new Float32Array(MAT_VEC4 * 4);
  const col = mat && mat.color ? [mat.color.r, mat.color.g, mat.color.b] : [0.6, 0.58, 0.55];
  const rough = num(mat?.roughness, 0.6);
  const metal = num(mat?.metalness, 0);
  const ior = num(mat?.ior, 1.5);
  const specI = num(mat?.specularIntensity, 1);
  const sc = mat?.specularColor ? (mat.specularColor.r + mat.specularColor.g + mat.specularColor.b) / 3 : 1;
  const f0d = Math.min(1, ((ior - 1) / (ior + 1)) ** 2 * sc) * specI;
  const cc = num(mat?.clearcoat, 0);
  const name = mat?.name || '';
  const transmissive = num(mat?.transmission, 0) > 0.5 || name === 'glass' || name === 'acrylic' || name === 'glass-edge';
  r[1] = rough; r[3] = metal;
  r.set(col, 4); r[7] = ior;
  if (transmissive) {
    r[0] = KIND.GLASS;
    r[2] = f0d;
    const ad = num(mat?.attenuationDistance, Infinity);
    const ac = mat?.attenuationColor;
    if (ac && isFinite(ad) && ad > 0) {
      r[8] = -Math.log(Math.max(ac.r, 1e-4)) / ad;
      r[9] = -Math.log(Math.max(ac.g, 1e-4)) / ad;
      r[10] = -Math.log(Math.max(ac.b, 1e-4)) / ad;
    }
    r[11] = num(mat?.thickness, 0.01);
    return r;
  }
  // a clear coat adds its own reflection on top (F0 0.04 for ior 1.5)
  r[2] = f0d + 0.04 * cc;
  const k = mat?.userData?.kigumi;
  const u = k?.uniforms;
  if (u && u.early && u.grain) {
    const g = u.grain.value, s = u.seed.value, e = u.early.value, l = u.late.value, t = u.tint.value;
    r[0] = g.x > 0 ? KIND.WOOD : KIND.CHAR;          // charred cedar has no rings
    r.set([e.r, e.g, e.b, num(u.lateRough?.value, 0)], 12);
    r.set([l.r, l.g, l.b, cc], 16);
    r.set([g.x, g.y, g.z, g.w], 20);
    r.set([s.x, s.y, s.z, s.w], 24);
    r.set([t.x, t.y, t.z, 0], 28);
    r.set([e.r, e.g, e.b], 4);
    return r;
  }
  r[0] = KIND.PLASTIC;
  return r;
}

function num(v, d) { return typeof v === 'number' && isFinite(v) ? v : d; }

// root: THREE.Object3D (the viewer's model group).  Visible meshes only.
// Returns { count, pos: Float32Array(9n), nrm: Float32Array(9n), ids: Uint32Array(n)
// ((mesh << 16) | material), mats: Float32Array, xforms: Float32Array, nMats, nMeshes, ms }.
export function extractScene(root) {
  const t0 = performance.now();
  root.updateMatrixWorld(true);
  const items = [];                 // { mesh, matrix (16, column-major) }
  root.traverseVisible((o) => {
    if (!o.isMesh || !o.geometry?.attributes?.position) return;
    if (o.isInstancedMesh) {
      const im = o.instanceMatrix.array;
      for (let i = 0; i < o.count; i++) items.push({ mesh: o, matrix: mul4(o.matrixWorld.elements, im.subarray(16 * i, 16 * i + 16)) });
    } else {
      items.push({ mesh: o, matrix: o.matrixWorld.elements });
    }
  });

  // count triangles first
  let total = 0;
  for (const it of items) total += triRanges(it.mesh).reduce((a, r) => a + r.count, 0);
  const pos = new Float32Array(total * 9);
  const nrm = new Float32Array(total * 9);
  const ids = new Uint32Array(total);
  const matIndex = new Map();
  const matRecs = [];
  const xforms = new Float32Array(Math.max(1, items.length) * XF_VEC4 * 4);
  let n = 0;
  const matId = (m) => {
    if (!matIndex.has(m)) { matIndex.set(m, matRecs.length); matRecs.push(convertMaterial(m)); }
    return matIndex.get(m);
  };
  if (items.length > 65535) throw new Error('too many meshes for the path tracer');

  items.forEach((it, mi) => {
    const g = it.mesh.geometry;
    const M = it.matrix;
    // world -> local (rows of the inverse 3x4)
    const inv = invert4(M);
    for (let r = 0; r < 3; r++) {
      xforms[12 * mi + 4 * r] = inv[r]; xforms[12 * mi + 4 * r + 1] = inv[4 + r];
      xforms[12 * mi + 4 * r + 2] = inv[8 + r]; xforms[12 * mi + 4 * r + 3] = inv[12 + r];
    }
    // normal matrix: inverse transpose of the upper 3x3; row r of it is column r of
    // the inverse, which is inv[4r .. 4r + 2] in column-major order
    const N3 = [inv[0], inv[1], inv[2], inv[4], inv[5], inv[6], inv[8], inv[9], inv[10]];   // rows
    const flip = det3(M) < 0;
    const P = g.attributes.position, NA = g.attributes.normal;
    const I = g.index;
    const mats = Array.isArray(it.mesh.material) ? it.mesh.material : [it.mesh.material];
    const vert = new Float64Array(9), nv = new Float64Array(9);
    for (const range of triRanges(it.mesh)) {
      const mid = matId(mats[range.material] || mats[0]);
      if (mid > 65535) throw new Error('too many materials for the path tracer');
      for (let f = range.start; f < range.start + range.count; f++) {
        for (let c = 0; c < 3; c++) {
          const cc = flip ? (c === 0 ? 0 : 3 - c) : c;
          const vi = I ? I.getX(3 * f + c) : 3 * f + c;
          const x = P.getX(vi), y = P.getY(vi), z = P.getZ(vi);
          vert[3 * cc] = M[0] * x + M[4] * y + M[8] * z + M[12];
          vert[3 * cc + 1] = M[1] * x + M[5] * y + M[9] * z + M[13];
          vert[3 * cc + 2] = M[2] * x + M[6] * y + M[10] * z + M[14];
          if (NA) {
            const a = NA.getX(vi), b = NA.getY(vi), d = NA.getZ(vi);
            nv[3 * cc] = N3[0] * a + N3[1] * b + N3[2] * d;
            nv[3 * cc + 1] = N3[3] * a + N3[4] * b + N3[5] * d;
            nv[3 * cc + 2] = N3[6] * a + N3[7] * b + N3[8] * d;
          }
        }
        // face normal (also the fallback shading normal); drop degenerate triangles
        const e1x = vert[3] - vert[0], e1y = vert[4] - vert[1], e1z = vert[5] - vert[2];
        const e2x = vert[6] - vert[0], e2y = vert[7] - vert[1], e2z = vert[8] - vert[2];
        let fx = e1y * e2z - e1z * e2y, fy = e1z * e2x - e1x * e2z, fz = e1x * e2y - e1y * e2x;
        const fl = Math.hypot(fx, fy, fz);
        if (!(fl > 1e-14)) continue;
        fx /= fl; fy /= fl; fz /= fl;
        const o = 9 * n;
        for (let k = 0; k < 9; k++) pos[o + k] = vert[k];
        for (let c = 0; c < 3; c++) {
          let a = NA ? nv[3 * c] : fx, b = NA ? nv[3 * c + 1] : fy, d = NA ? nv[3 * c + 2] : fz;
          let l = Math.hypot(a, b, d);
          if (!(l > 1e-12)) { a = fx; b = fy; d = fz; l = 1; }
          nrm[o + 3 * c] = a / l; nrm[o + 3 * c + 1] = b / l; nrm[o + 3 * c + 2] = d / l;
        }
        ids[n] = (mi << 16) | mid;
        n++;
      }
    }
  });
  const mats = new Float32Array(Math.max(1, matRecs.length) * MAT_VEC4 * 4);
  matRecs.forEach((r, i) => mats.set(r, i * MAT_VEC4 * 4));
  return {
    count: n, pos: pos.subarray(0, 9 * n), nrm: nrm.subarray(0, 9 * n), ids: ids.subarray(0, n),
    mats, xforms, nMats: matRecs.length, nMeshes: items.length, ms: performance.now() - t0,
  };
}

// triangle ranges of a mesh: [{start, count, material}] in triangles (groups, draw range)
function triRanges(mesh) {
  const g = mesh.geometry;
  const nIdx = g.index ? g.index.count : g.attributes.position.count;
  const dr0 = Math.max(0, g.drawRange?.start || 0);
  const dr1 = Math.min(nIdx, dr0 + (isFinite(g.drawRange?.count) ? g.drawRange.count : nIdx));
  const out = [];
  const add = (s, e, m) => {
    s = Math.max(s, dr0); e = Math.min(e, dr1);
    const fs = Math.ceil(s / 3), fe = Math.floor(e / 3);
    if (fe > fs) out.push({ start: fs, count: fe - fs, material: m });
  };
  if (Array.isArray(mesh.material) && g.groups && g.groups.length) {
    for (const gr of g.groups) add(gr.start, gr.start + (isFinite(gr.count) ? gr.count : nIdx), gr.materialIndex || 0);
  } else {
    add(0, nIdx, 0);
  }
  return out;
}

function mul4(a, b) {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[4 * c + r] = a[r] * b[4 * c] + a[4 + r] * b[4 * c + 1] + a[8 + r] * b[4 * c + 2] + a[12 + r] * b[4 * c + 3];
  }
  return o;
}

function det3(m) {
  return m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
}

// inverse of an affine 4x4 (column-major)
function invert4(m) {
  const a00 = m[0], a01 = m[4], a02 = m[8], a10 = m[1], a11 = m[5], a12 = m[9], a20 = m[2], a21 = m[6], a22 = m[10];
  const c00 = a11 * a22 - a12 * a21, c01 = a02 * a21 - a01 * a22, c02 = a01 * a12 - a02 * a11;
  const c10 = a12 * a20 - a10 * a22, c11 = a00 * a22 - a02 * a20, c12 = a02 * a10 - a00 * a12;
  const c20 = a10 * a21 - a11 * a20, c21 = a01 * a20 - a00 * a21, c22 = a00 * a11 - a01 * a10;
  const det = a00 * c00 + a01 * c10 + a02 * c20;
  const id = det !== 0 ? 1 / det : 0;
  const r = [c00 * id, c01 * id, c02 * id, c10 * id, c11 * id, c12 * id, c20 * id, c21 * id, c22 * id];   // rows
  const tx = m[12], ty = m[13], tz = m[14];
  const o = new Float64Array(16);
  o[0] = r[0]; o[4] = r[1]; o[8] = r[2];
  o[1] = r[3]; o[5] = r[4]; o[9] = r[5];
  o[2] = r[6]; o[6] = r[7]; o[10] = r[8];
  o[12] = -(r[0] * tx + r[1] * ty + r[2] * tz);
  o[13] = -(r[3] * tx + r[4] * ty + r[5] * tz);
  o[14] = -(r[6] * tx + r[7] * ty + r[8] * tz);
  o[15] = 1;
  return o;
}
