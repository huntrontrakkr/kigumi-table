// Three.js stage: lighting, shadow-catching paper floor, GLB loading with
// per-part materials, explode / assembly animation, picking and camera framing.

import * as THREE from 'three/webgpu';
import { Fn, uniform, uniformArray, vec3, vec4, float, int, positionWorld, length, max, min, smoothstep, mix, abs, clamp, Loop, If, Break } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { makePartMaterial, makeGlassEdgeMaterial, swatch } from './materials.js';

const EXPLODE_PER_STEP = 0.075;    // metres of lift per assembly step at explode = 1
const DROP_HEIGHT = 0.22;          // metres a part falls from in the assembly scrubber
const DROP_MS = 650;
const MAX_FEET = 16;

// ---------------------------------------------------------------------------
// tone-mapping helpers: find the linear colour that displays as a given sRGB
// colour after tone mapping, so floor and background match the page paper.
// ---------------------------------------------------------------------------
function mulCols(c0, c1, c2, v) {
  return [c0[0] * v[0] + c1[0] * v[1] + c2[0] * v[2],
          c0[1] * v[0] + c1[1] * v[1] + c2[1] * v[2],
          c0[2] * v[0] + c1[2] * v[1] + c2[2] * v[2]];
}
function agx(v, exposure) {
  const S2R = [[0.6274, 0.0691, 0.0164], [0.3293, 0.9195, 0.0880], [0.0433, 0.0113, 0.8956]];
  const R2S = [[1.6605, -0.1246, -0.0182], [-0.5876, 1.1329, -0.1006], [-0.0728, -0.0083, 1.1187]];
  const INSET = [[0.856627153315983, 0.137318972929847, 0.11189821299995],
                 [0.0951212405381588, 0.761241990602591, 0.0767994186031903],
                 [0.0482516061458583, 0.101439036467562, 0.811302368396859]];
  const OUTSET = [[1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
                  [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
                  [-0.016493938717834573, -0.016493938717834257, 1.2519364065950405]];
  const minEv = -12.47393, maxEv = 4.026069;
  let c = v.map((x) => x * exposure);
  c = mulCols(...S2R, c);
  c = mulCols(...INSET, c);
  c = c.map((x) => {
    let t = (Math.log2(Math.max(x, 1e-10)) - minEv) / (maxEv - minEv);
    t = Math.min(1, Math.max(0, t));
    const t2 = t * t, t4 = t2 * t2;
    return 15.5 * t4 * t2 - 40.14 * t4 * t + 31.96 * t4 - 6.868 * t2 * t + 0.4298 * t2 + 0.1191 * t - 0.00232;
  });
  c = mulCols(...OUTSET, c);
  c = c.map((x) => Math.pow(Math.max(0, x), 2.2));
  c = mulCols(...R2S, c);
  return c.map((x) => Math.min(1, Math.max(0, x)));
}
function aces(v, exposure) {
  const IN = [[0.59719, 0.07600, 0.02840], [0.35458, 0.90834, 0.13383], [0.04823, 0.01566, 0.83777]];
  const OUT = [[1.60475, -0.10208, -0.00327], [-0.53108, 1.10813, -0.07276], [-0.07367, -0.00605, 1.07602]];
  let c = v.map((x) => x * exposure / 0.6);
  c = mulCols(...IN, c);
  c = c.map((x) => (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510) + 0.238081));
  c = mulCols(...OUT, c);
  return c.map((x) => Math.min(1, Math.max(0, x)));
}
function neutral(v, exposure) {
  let c = v.map((x) => x * exposure);
  const x = Math.min(c[0], c[1], c[2]);
  const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  c = c.map((y) => y - offset);
  const peak = Math.max(c[0], c[1], c[2]);
  const start = 0.8 - 0.04;
  if (peak < start) return c;
  const d = 1 - start;
  const newPeak = 1 - d * d / (peak + d - start);
  c = c.map((y) => y * newPeak / peak);
  const g = 1 - 1 / (0.15 * (peak - newPeak) + 1);
  return c.map((y) => y * (1 - g) + newPeak * g);
}
function inverseToneMap(targetLinear, mapping, exposure) {
  const f = mapping === THREE.ACESFilmicToneMapping ? aces : mapping === THREE.AgXToneMapping ? agx
    : mapping === THREE.NeutralToneMapping ? neutral : null;
  if (!f) return targetLinear.slice();
  let x = targetLinear.slice();
  for (let i = 0; i < 60; i++) {
    const y = f(x, exposure);
    x = x.map((xi, k) => Math.min(64, Math.max(1e-5, xi * Math.pow(targetLinear[k] / Math.max(y[k], 1e-6), 0.8))));
  }
  return x;
}

// ---------------------------------------------------------------------------
// geometry: crease-angle normals using the mesh topology (indexed input)
// ---------------------------------------------------------------------------
export function creased(geom, creaseDeg = 35) {
  const pa = geom.attributes.position;
  const nV = pa.count;
  let P;
  if (pa.isBufferAttribute && pa.itemSize === 3 && pa.array instanceof Float32Array && !pa.normalized) {
    P = pa.array;
  } else {
    P = new Float32Array(nV * 3);
    for (let i = 0; i < nV; i++) { P[3 * i] = pa.getX(i); P[3 * i + 1] = pa.getY(i); P[3 * i + 2] = pa.getZ(i); }
  }
  let I = geom.index ? geom.index.array : null;
  if (!I) { I = new Uint32Array(nV); for (let i = 0; i < nV; i++) I[i] = i; }
  const nF = Math.floor(I.length / 3);
  const fn = new Float32Array(nF * 3);
  const ang = new Float32Array(nF * 3);
  for (let f = 0; f < nF; f++) {
    const a = 3 * I[3 * f], b = 3 * I[3 * f + 1], c = 3 * I[3 * f + 2];
    const abx = P[b] - P[a], aby = P[b + 1] - P[a + 1], abz = P[b + 2] - P[a + 2];
    const acx = P[c] - P[a], acy = P[c + 1] - P[a + 1], acz = P[c + 2] - P[a + 2];
    const bcx = P[c] - P[b], bcy = P[c + 1] - P[b + 1], bcz = P[c + 2] - P[b + 2];
    let nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    const nl = Math.hypot(nx, ny, nz);
    if (nl > 0) { nx /= nl; ny /= nl; nz /= nl; }
    fn[3 * f] = nx; fn[3 * f + 1] = ny; fn[3 * f + 2] = nz;
    const lab = Math.hypot(abx, aby, abz), lac = Math.hypot(acx, acy, acz), lbc = Math.hypot(bcx, bcy, bcz);
    const angle = (ux, uy, uz, lu, vx, vy, vz, lv) => {
      if (lu === 0 || lv === 0) return 0;
      const d = (ux * vx + uy * vy + uz * vz) / (lu * lv);
      return Math.acos(Math.min(1, Math.max(-1, d)));
    };
    ang[3 * f] = angle(abx, aby, abz, lab, acx, acy, acz, lac);
    ang[3 * f + 1] = angle(-abx, -aby, -abz, lab, bcx, bcy, bcz, lbc);
    ang[3 * f + 2] = Math.max(0, Math.PI - ang[3 * f] - ang[3 * f + 1]);
  }
  // vertex -> incident corners (CSR)
  const start = new Uint32Array(nV + 1);
  for (let k = 0; k < nF * 3; k++) start[I[k] + 1]++;
  for (let v = 0; v < nV; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nV);
  const corners = new Uint32Array(nF * 3);
  for (let k = 0; k < nF * 3; k++) corners[fill[I[k]]++] = k;

  const cosC = Math.cos(creaseDeg * Math.PI / 180);
  const outP = new Float32Array(nF * 9);
  const outN = new Float32Array(nF * 9);
  const outI = new Uint32Array(nF * 3);
  const head = new Int32Array(nV).fill(-1);
  const next = new Int32Array(nF * 3);
  let count = 0;
  for (let f = 0; f < nF; f++) {
    const fx = fn[3 * f], fy = fn[3 * f + 1], fz = fn[3 * f + 2];
    for (let c = 0; c < 3; c++) {
      const v = I[3 * f + c];
      let sx = 0, sy = 0, sz = 0;
      for (let j = start[v]; j < start[v + 1]; j++) {
        const k = corners[j];
        const g = (k / 3) | 0;
        const gx = fn[3 * g], gy = fn[3 * g + 1], gz = fn[3 * g + 2];
        if (fx * gx + fy * gy + fz * gz >= cosC) {
          const w = ang[k];
          sx += gx * w; sy += gy * w; sz += gz * w;
        }
      }
      let sl = Math.hypot(sx, sy, sz);
      if (sl < 1e-12) { sx = fx; sy = fy; sz = fz; sl = Math.hypot(sx, sy, sz) || 1; }
      sx /= sl; sy /= sl; sz /= sl;
      let e = head[v];
      while (e >= 0) {
        if (outN[3 * e] * sx + outN[3 * e + 1] * sy + outN[3 * e + 2] * sz > 0.99995) break;
        e = next[e];
      }
      if (e < 0) {
        e = count++;
        outP[3 * e] = P[3 * v]; outP[3 * e + 1] = P[3 * v + 1]; outP[3 * e + 2] = P[3 * v + 2];
        outN[3 * e] = sx; outN[3 * e + 1] = sy; outN[3 * e + 2] = sz;
        next[e] = head[v];
        head[v] = e;
      }
      outI[3 * f + c] = e;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(outP.slice(0, count * 3), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(outN.slice(0, count * 3), 3));
  g.setIndex(new THREE.BufferAttribute(count > 65535 ? outI : Uint16Array.from(outI), 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

// Put faces that look up/down first (group 0) and the rim faces after (group 1).
function splitByFacing(geo, mesh) {
  const idx = geo.index && geo.index.array;
  const nrm = geo.attributes.normal;
  if (!idx || !nrm) return false;
  const nm = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
  const caps = [], rim = [];
  const n = new THREE.Vector3(), t = new THREE.Vector3();
  for (let f = 0; f < idx.length; f += 3) {
    n.set(0, 0, 0);
    for (let c = 0; c < 3; c++) n.add(t.fromBufferAttribute(nrm, idx[f + c]));
    n.applyMatrix3(nm).normalize();
    (Math.abs(n.y) > 0.7 ? caps : rim).push(idx[f], idx[f + 1], idx[f + 2]);
  }
  if (!rim.length || !caps.length) return false;
  const all = caps.concat(rim);
  geo.setIndex(new THREE.BufferAttribute(idx instanceof Uint16Array ? Uint16Array.from(all) : Uint32Array.from(all), 1));
  geo.clearGroups();
  geo.addGroup(0, caps.length, 0);
  geo.addGroup(caps.length, rim.length, 1);
  return true;
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

// ---------------------------------------------------------------------------
export class Viewer {
  // Tone mapping: Khronos PBR Neutral keeps species colours true (AgX greys out pale
  // woods like hinoki and maple); ?tm=agx or ?tm=aces switch for comparison.
  constructor(container, { background = '#EAE4D8', toneMapping = 'neutral', exposure = 1.0 } = {}) {
    this.container = container;
    // WebGPU where the browser has it, WebGL2 otherwise (three.js picks the backend)
    this.renderer = new THREE.WebGPURenderer({ antialias: true, powerPreference: 'high-performance' });
    const r = this.renderer;
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = { aces: THREE.ACESFilmicToneMapping, neutral: THREE.NeutralToneMapping, agx: THREE.AgXToneMapping }[toneMapping] ?? THREE.NeutralToneMapping;
    r.toneMappingExposure = exposure;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this._shadowDirty = true;
    container.appendChild(r.domElement);
    r.domElement.classList.add('gl');

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.02, 200);
    this.camera.position.set(1.6, 1.1, 2.2);
    this.scene.environmentIntensity = 0.62;
    this.scene.environmentRotation.set(0, Math.PI * 0.32, 0);

    // the renderer starts asynchronously (WebGPU adapter and device); then the
    // environment: a soft studio room, prefiltered on the GPU
    this.ready = r.init().then(async () => {
      this.backend = r.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
      const pmrem = new THREE.PMREMGenerator(r);
      const rt = pmrem.fromScene(new RoomEnvironment(), 0.04);
      this.scene.environment = rt.texture;
      this.requestRender();
      return this.backend;
    });

    // key light with soft shadows
    this.key = new THREE.DirectionalLight(0xfff6ec, 2.4);
    this.keyDir = new THREE.Vector3(-0.55, 1.0, 0.42).normalize();
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.radius = 4;
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.0032;
    this.scene.add(this.key, this.key.target);

    // paper floor: unlit, the background colour, darkened by contact occlusion
    // round the feet and under the table (node material), with the key light's
    // shadow laid over it by a shadow-only sheet
    this.setBackground(background);
    this.floorUniforms = {
      uFloor: uniform(new THREE.Color()),
      uShadowTint: uniform(new THREE.Color(0.40, 0.36, 0.33)),
      uFeet: uniformArray(Array.from({ length: MAX_FEET }, () => new THREE.Vector4(0, 0, 0, 0)), 'vec4'),
      uFeetN: uniform(0, 'int'),
      uBox: uniform(new THREE.Vector4(0, 0, 0, 0)),
      uBoxAO: uniform(0.0),
    };
    this._applyFloorColour();
    const fu = this.floorUniforms;
    const floorAO = Fn(() => {
      const p = positionWorld.xz;
      const ao = float(0.0).toVar();
      Loop(MAX_FEET, ({ i }) => {
        If(int(i).greaterThanEqual(fu.uFeetN), () => { Break(); });
        const f = fu.uFeet.element(i);
        const d = length(p.sub(f.xy)).div(max(f.z, 1e-3));
        ao.addAssign(f.w.mul(float(1.0).sub(smoothstep(0.55, 1.9, d))).mul(float(1.0).sub(smoothstep(0.0, 1.0, d)).mul(0.45).add(0.55)));
      });
      const q = abs(p.sub(fu.uBox.xy)).sub(fu.uBox.zw);
      const sd = length(max(q, 0.0)).add(min(max(q.x, q.y), 0.0));
      ao.addAssign(fu.uBoxAO.mul(float(1.0).sub(smoothstep(-0.25, 0.35, sd))));
      return clamp(ao, 0.0, 0.9);
    });
    const floorMat = new THREE.MeshBasicNodeMaterial();
    floorMat.colorNode = fu.uFloor.mul(mix(vec3(1.0), fu.uShadowTint, floorAO()));
    floorMat.toneMapped = true;
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), floorMat);
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.renderOrder = -2;
    this.scene.add(this.floor);
    const shadowMat = new THREE.ShadowNodeMaterial({ color: new THREE.Color(0.20, 0.17, 0.15), opacity: 0.42 });
    this.shadowSheet = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), shadowMat);
    this.shadowSheet.rotation.x = -Math.PI / 2;
    this.shadowSheet.position.y = 0.0004;
    this.shadowSheet.receiveShadow = true;
    this.shadowSheet.renderOrder = -1;
    this.scene.add(this.shadowSheet);

    // controls
    this.controls = new OrbitControls(this.camera, r.domElement);
    const c = this.controls;
    c.enableDamping = true;
    c.dampingFactor = 0.09;
    c.rotateSpeed = 0.75;
    c.zoomSpeed = 0.9;
    c.panSpeed = 0.8;
    c.screenSpacePanning = true;
    c.maxPolarAngle = Math.PI / 2 - 0.035;
    c.minDistance = 0.2;
    c.maxDistance = 14;
    c.target.set(0, 0.25, 0);
    c.addEventListener('change', () => this.requestRender());
    c.addEventListener('start', () => { this.userMoved = true; this._cancelCameraTween(); });

    // model state
    this.model = null;           // THREE.Group
    this.parts = new Map();      // id -> {obj, meshes, step, base, kind, material, info}
    this.steps = [];
    this.explode = 0;
    this.stepK = 0;              // 1..N; N = complete
    this.glassVisible = true;
    this.hasGlass = false;
    this.restBox = new THREE.Box3();
    this.tweens = [];
    this.needsRender = true;
    this.highlighted = null;
    this.onRendered = null;
    this.onHover = null;

    // picking
    this.raycaster = new THREE.Raycaster();
    this.pointer = null;
    this._pickPending = false;
    r.domElement.addEventListener('pointermove', (e) => {
      if (e.buttons) { this.pointer = null; return; }
      const rect = r.domElement.getBoundingClientRect();
      this.pointer = { x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width, h: rect.height, cx: e.clientX, cy: e.clientY };
      this._pickPending = true;
    });
    r.domElement.addEventListener('pointerleave', () => { this.pointer = null; this._pickPending = true; });
    r.domElement.addEventListener('pointerdown', () => { this.pointer = null; this._pickPending = true; });

    this.loader = new GLTFLoader();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(container);
    r.domElement.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.onError?.('WebGL context lost — reload the page'); });
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ---------------------------------------------------------------- basics
  setBackground(css) {
    this.bgCss = css;
    this.scene.background = new THREE.Color(css);
    if (this.floorUniforms) this._applyFloorColour();
    this.requestRender();
  }
  _applyFloorColour() {
    const lin = new THREE.Color(this.bgCss);   // linear working colour of the target
    const inv = inverseToneMap([lin.r, lin.g, lin.b], this.renderer.toneMapping, this.renderer.toneMappingExposure);
    this.floorUniforms.uFloor.value.setRGB(inv[0], inv[1], inv[2], THREE.LinearSRGBColorSpace);
  }
  requestRender() { this.needsRender = true; }
  _resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + 'px';
    this.renderer.domElement.style.height = h + 'px';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }
  isAnimating() { return this.tweens.length > 0 || this._damping; }

  _loop(now) {
    requestAnimationFrame(this._loop);
    let changed = false;
    if (this.tweens.length) {
      const keep = [];
      for (const tw of this.tweens) {
        const t = Math.min(1, (now - tw.t0) / tw.dur);
        tw.fn(t);
        if (t < 1) keep.push(tw); else tw.done?.();
      }
      this.tweens = keep;
      changed = true;
    }
    this._damping = this.controls.update();
    if (this._damping) changed = true;
    if (this._pickPending) { this._pickPending = false; this._doPick(); }
    if (changed || this.needsRender) {
      this.needsRender = false;
      if (!this.backend) return;               // the renderer is still starting
      if (this._shadowDirty) { this.key.shadow.needsUpdate = true; this._shadowDirty = false; }
      this.renderer.render(this.scene, this.camera);
      this.onRendered?.();
    } else {
      this.onIdleFrame?.();
    }
  }
  tween(dur, fn, done) {
    const tw = { t0: performance.now(), dur, fn, done };
    this.tweens.push(tw);
    this.requestRender();
    return tw;
  }

  // ---------------------------------------------------------------- model
  async loadModel(buffer, report, { frame = false } = {}) {
    const T0 = performance.now();
    const gltf = await this.loader.parseAsync(buffer, '');
    const T1 = performance.now();
    const assoc = gltf.parser.associations;
    const jsonNodes = gltf.parser.json.nodes || [];
    const repParts = new Map((report?.parts || []).map((p) => [p.id, p]));
    const rawName = (obj) => {
      const a = assoc.get(obj);
      if (a && a.nodes !== undefined && jsonNodes[a.nodes]) return jsonNodes[a.nodes].name || obj.name;
      return obj.name;
    };
    gltf.scene.updateMatrixWorld(true);

    // which objects are parts: named after a report part, else top-level nodes
    let partObjs = [];
    gltf.scene.traverse((o) => { if (o !== gltf.scene && repParts.has(rawName(o))) partObjs.push(o); });
    partObjs = partObjs.filter((o) => { for (let p = o.parent; p; p = p.parent) if (partObjs.includes(p)) return false; return true; });
    if (!partObjs.length) partObjs = gltf.scene.children.slice();

    // assembly step for each part
    const steps = (report?.assembly || []).map((s, i) => ({ index: i, step: s.step ?? i + 1, text: s.text || '', parts: s.parts || [] }));
    const stepOf = new Map();
    steps.forEach((s, i) => s.parts.forEach((id) => stepOf.set(id, i)));
    const lastStep = Math.max(0, steps.length - 1);

    // the colour silicone bumpers sit on: the most common beam wood
    const woodCount = {};
    for (const p of repParts.values()) if (p.kind === 'beam') woodCount[p.material] = (woodCount[p.material] || 0) + 1;
    const beamWood = Object.entries(woodCount).sort((a, b) => b[1] - a[1])[0]?.[0];
    const underHex = beamWood ? swatch(beamWood) : undefined;

    const group = new THREE.Group();
    group.name = 'kigumi-model';
    const parts = new Map();
    for (const obj of partObjs) {
      const id = rawName(obj);
      const rp = repParts.get(id) || {};
      const ud = obj.userData || {};
      const kind = rp.kind || ud.kind || 'part';
      const material = rp.material || ud.material || 'unknown';
      group.attach(obj);
      let meshes = [];
      obj.traverse((m) => { if (m.isMesh) meshes.push(m); });
      // material "none" (e.g. no glass top) or empty geometry: the part is absent
      const absent = material === 'none' || !meshes.some((m) => (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position?.count || 0) >= 3);
      if (absent) {
        meshes.forEach((m) => { m.geometry.dispose(); m.removeFromParent(); });
        meshes = [];
      }
      for (const m of meshes) {
        const old = m.material;
        let fallbackHex;
        // the exporter writes its sRGB palette straight into baseColorFactor
        if (old && old.color) fallbackHex = '#' + old.color.getHexString(THREE.LinearSRGBColorSpace);
        const geo = creased(m.geometry, kind === 'beam' || kind === 'block' ? 28 : 38);
        m.geometry.dispose();
        m.geometry = geo;
        let glassThickness;                     // from its world-space height (flat sheet)
        if (kind === 'glass') {
          const bb = geo.boundingBox.clone().applyMatrix4(m.matrixWorld);
          glassThickness = Math.max(0.003, Math.min(0.03, bb.max.y - bb.min.y));
        }
        m.material = makePartMaterial({ id, kind, material, fallbackHex, glassThickness, underHex });
        if (kind === 'glass' && material !== 'acrylic' && splitByFacing(geo, m)) {
          m.material = [m.material, makeGlassEdgeMaterial(glassThickness)];
        }
        if (old) (Array.isArray(old) ? old : [old]).forEach((x) => x.dispose());
        m.castShadow = kind !== 'glass';
        m.receiveShadow = true;
        m.userData.partId = id;
      }
      parts.set(id, {
        id, obj, meshes, kind, material, info: rp, absent,
        step: stepOf.has(id) ? stepOf.get(id) : lastStep,
        base: obj.position.clone(),
        drop: 0,
      });
    }
    group.updateMatrixWorld(true);

    const T2 = performance.now();
    // rest bounds, a thinned point cloud for camera framing, contact points on the floor
    const restBox = new THREE.Box3().setFromObject(group);
    const framePts = [];
    for (const p of parts.values()) {
      for (const m of p.meshes) {
        const pos = m.geometry.attributes.position;
        const stride = Math.max(1, Math.floor(pos.count / 160));
        for (let i = 0; i < pos.count; i += stride) framePts.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld));
      }
    }
    const feet = [];
    const v = new THREE.Vector3();
    for (const p of parts.values()) {
      if (p.absent) continue;
      const pb = new THREE.Box3().setFromObject(p.obj);
      if (pb.min.y > restBox.min.y + 0.004 || p.kind === 'glass') continue;
      for (const m of p.meshes) {
        const pos = m.geometry.attributes.position;
        let sx = 0, sz = 0, n = 0;
        const pts = [];
        for (let i = 0; i < pos.count; i++) {
          v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
          if (v.y < pb.min.y + 0.004) { sx += v.x; sz += v.z; n++; pts.push(v.x, v.z); }
        }
        if (!n) continue;
        const cx = sx / n, cz = sz / n;
        let rr = 0;
        for (let i = 0; i < pts.length; i += 2) rr = Math.max(rr, Math.hypot(pts[i] - cx, pts[i + 1] - cz));
        feet.push(new THREE.Vector4(cx, cz, Math.max(0.02, rr), 0.55));
      }
    }

    const T3 = performance.now();
    // compile off-screen before swapping so the old model stays until the new one is ready
    group.visible = true;
    try {
      await this.ready;
      await this.renderer.compileAsync(group, this.camera, this.scene);
    } catch (e) { /* compile lazily on first render */ }

    // swap
    const old = this.model;
    if (old) {
      this.scene.remove(old);
      old.traverse((m) => {
        if (m.isMesh) {
          m.geometry.dispose();
          (Array.isArray(m.material) ? m.material : [m.material]).forEach((x) => x.dispose());
        }
      });
    }
    this.model = group;
    this.parts = parts;
    this.steps = steps;
    this.restBox.copy(restBox);
    this.framePts = framePts;
    this.hasGlass = [...parts.values()].some((p) => p.kind === 'glass' && !p.absent);
    this.highlighted = null;
    this.scene.add(group);

    // floor occlusion
    const fu = this.floorUniforms;
    fu.uFeetN.value = Math.min(MAX_FEET, feet.length);
    feet.slice(0, MAX_FEET).forEach((f, i) => fu.uFeet.array[i].copy(f));
    const c = restBox.getCenter(new THREE.Vector3());
    const s = restBox.getSize(new THREE.Vector3());
    fu.uBox.value.set(c.x, c.z, Math.max(0.01, s.x * 0.5 - 0.06), Math.max(0.01, s.z * 0.5 - 0.06));
    fu.uBoxAO.value = 0.16;

    // keep the scrubber where it was (complete stays complete)
    const N = steps.length;
    if (!this.stepK || this.stepK >= this._lastN || this.stepK > N) this.stepK = N;
    this._lastN = N;
    this._applyPose();
    this._fitShadow();
    if (frame) this.frame(frame === 'animate');
    this.requestRender();
    const T4 = performance.now();
    let tris = 0;
    for (const p of parts.values()) for (const m of p.meshes) tris += m.geometry.index ? m.geometry.index.count / 3 : 0;
    return { parts: parts.size, steps: N, triangles: tris, parse_ms: Math.round(T1 - T0), process_ms: Math.round(T2 - T1),
      bounds_ms: Math.round(T3 - T2), compile_swap_ms: Math.round(T4 - T3) };
  }

  // ---------------------------------------------------------------- pose
  _applyPose() {
    const k = this.stepK || this.steps.length;
    for (const p of this.parts.values()) {
      const visibleByStep = this.steps.length === 0 || p.step <= k - 1;
      const glassOk = p.kind !== 'glass' || this.glassVisible;
      p.obj.visible = visibleByStep && glassOk;
      p.obj.position.copy(p.base);
      p.obj.position.y += p.step * this.explode * EXPLODE_PER_STEP + p.drop;
    }
    this._shadowDirty = true;
    this.requestRender();
  }
  setExplode(v, animate = false) {
    v = Math.max(0, Math.min(1, v));
    if (this._explodeTween) { this.tweens = this.tweens.filter((t) => t !== this._explodeTween); this._explodeTween = null; }
    if (!animate) { this.explode = v; this._applyPose(); this._fitShadow(); return; }
    const from = this.explode;
    this._explodeTween = this.tween(520, (t) => {
      this.explode = from + (v - from) * easeInOut(t);
      this._applyPose();
      this._fitShadow();
    }, () => { this._explodeTween = null; });
  }
  setStep(k, animate = true) {
    const N = this.steps.length;
    k = Math.max(1, Math.min(N || 1, Math.round(k)));
    this.stepK = k;
    for (const p of this.parts.values()) p.drop = 0;
    if (this._dropTween) { this.tweens = this.tweens.filter((t) => t !== this._dropTween); this._dropTween = null; }
    if (animate && N) {
      const active = [...this.parts.values()].filter((p) => p.step === k - 1);
      const drop = DROP_HEIGHT + 0.25 * this.explode * EXPLODE_PER_STEP;
      active.forEach((p) => { p.drop = drop; });
      this._applyPose();
      this._dropTween = this.tween(DROP_MS, (t) => {
        const e = easeOutCubic(t);
        active.forEach((p) => { p.drop = drop * (1 - e); });
        this._applyPose();
      }, () => { this._dropTween = null; });
    } else {
      this._applyPose();
    }
    this._fitShadow();
  }
  setGlassVisible(on) {
    this.glassVisible = on;
    this._applyPose();
  }

  // ---------------------------------------------------------------- shadows / camera
  _fitShadow() {
    if (!this.model) return;
    const box = new THREE.Box3();
    for (const p of this.parts.values()) {
      if (!p.obj.visible && p.drop === 0) continue;
      box.expandByObject(p.obj);
    }
    if (box.isEmpty()) box.copy(this.restBox);
    box.min.y = Math.min(box.min.y, 0);
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() * 0.5 + 0.05;
    const L = this.key;
    L.position.copy(center).addScaledVector(this.keyDir, radius * 3);
    L.target.position.copy(center);
    L.updateMatrixWorld();
    L.target.updateMatrixWorld();
    // tight ortho bounds in light space
    const cam = L.shadow.camera;
    const view = new THREE.Matrix4().lookAt(L.position, center, new THREE.Vector3(0, 1, 0));
    view.setPosition(L.position);
    const inv = view.clone().invert();
    const pts = [];
    for (let i = 0; i < 8; i++) {
      pts.push(new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).applyMatrix4(inv));
    }
    const lb = new THREE.Box3().setFromPoints(pts);
    const pad = 0.05;
    cam.left = lb.min.x - pad; cam.right = lb.max.x + pad;
    cam.bottom = lb.min.y - pad; cam.top = lb.max.y + pad;
    cam.near = Math.max(0.01, -lb.max.z - 0.5);
    cam.far = -lb.min.z + 0.5;
    cam.updateProjectionMatrix();
    L.shadow.needsUpdate = true;
    this._shadowDirty = true;
    this.requestRender();
  }

  _cancelCameraTween() {
    if (this._camTween) { this.tweens = this.tweens.filter((t) => t !== this._camTween); this._camTween = null; }
  }

  // Frame the model: 3/4 view from the front-right, slightly elevated.
  frame(animate = false, { azimuth = 38, elevation = 24 } = {}) {
    const box = this.restBox.clone();
    if (box.isEmpty()) return;
    const cam = this.camera;
    const center = box.getCenter(new THREE.Vector3());
    center.y = box.min.y + (box.max.y - box.min.y) * 0.42;
    const az = azimuth * Math.PI / 180, el = elevation * Math.PI / 180;
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
    const fwd = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    const tanV = Math.tan(cam.fov * Math.PI / 360);
    const tanH = tanV * cam.aspect;
    const mx = 0.80, my = 0.70;     // leave room for the dock, the tools and some air
    let pts = this.framePts;
    if (!pts || !pts.length) {
      pts = [];
      for (let i = 0; i < 8; i++) pts.push(new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
    }
    // centre the projected extent, then find the distance that fits every point
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const q of pts) {
      const p = q.clone().sub(center);
      const x = p.dot(right), y = p.dot(up);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    center.addScaledVector(right, (x0 + x1) / 2).addScaledVector(up, (y0 + y1) / 2);
    let dist = 0;
    for (const q of pts) {
      const p = q.clone().sub(center);
      const x = p.dot(right), y = p.dot(up), z = p.dot(fwd);
      dist = Math.max(dist, z + Math.abs(x) / (tanH * mx), z + Math.abs(y) / (tanV * my));
    }
    // nudge the table up a little: the dock covers the bottom of the view
    const lift = 0.035 * 2 * dist * tanV;
    center.addScaledVector(up, -lift);
    const pos = center.clone().addScaledVector(dir, dist);
    this._cancelCameraTween();
    if (!animate) {
      cam.position.copy(pos);
      this.controls.target.copy(center);
      this.controls.update();
      this.requestRender();
      return;
    }
    const p0 = cam.position.clone(), t0 = this.controls.target.clone();
    // interpolate in spherical coordinates around a moving target for a calm arc
    const s0 = new THREE.Spherical().setFromVector3(p0.clone().sub(t0));
    const s1 = new THREE.Spherical().setFromVector3(pos.clone().sub(center));
    let dTheta = s1.theta - s0.theta;
    if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    this._camTween = this.tween(750, (t) => {
      const e = easeInOut(t);
      const tg = t0.clone().lerp(center, e);
      const s = new THREE.Spherical(s0.radius + (s1.radius - s0.radius) * e, s0.phi + (s1.phi - s0.phi) * e, s0.theta + dTheta * e);
      cam.position.copy(tg).add(new THREE.Vector3().setFromSpherical(s));
      this.controls.target.copy(tg);
    }, () => { this._camTween = null; });
  }

  // ---------------------------------------------------------------- photo
  // A still at twice the screen resolution from one of the standard views,
  // rendered on the GPU; returns a PNG blob.
  async photo({ view = 'hero', scale = 2 } = {}) {
    const poses = { hero: { azimuth: 38, elevation: 24 }, top: { azimuth: 0, elevation: 89 },
                    low: { azimuth: 25, elevation: 6 }, side: { azimuth: 90, elevation: 2 }, front: { azimuth: 0, elevation: 14 } };
    const r = this.renderer;
    const cam = this.camera;
    const keep = { pos: cam.position.clone(), target: this.controls.target.clone(), ratio: r.getPixelRatio() };
    this.frame(false, poses[view] || poses.hero);
    r.setPixelRatio(keep.ratio * scale);
    this._resize();
    await this.ready;
    this.key.shadow.needsUpdate = true;
    await this.renderer.renderAsync(this.scene, cam);
    const blob = await new Promise((res) => r.domElement.toBlob(res, 'image/png'));
    r.setPixelRatio(keep.ratio);
    cam.position.copy(keep.pos);
    this.controls.target.copy(keep.target);
    this.controls.update();
    this._resize();
    return blob;
  }

  // ---------------------------------------------------------------- picking
  _doPick() {
    let id = null;
    if (this.pointer && this.model) {
      const p = this.pointer;
      const ndc = new THREE.Vector2(p.x / p.w * 2 - 1, -(p.y / p.h) * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.camera);
      const meshes = [];
      for (const part of this.parts.values()) {
        if (!part.obj.visible || part.kind === 'glass') continue;
        meshes.push(...part.meshes);
      }
      const hits = this.raycaster.intersectObjects(meshes, false);
      if (hits.length) id = hits[0].object.userData.partId;
    }
    if (id !== this.highlighted) {
      this.highlight(id);
    }
    this.onHover?.(id, this.pointer);
  }
  highlight(id) {
    const prev = this.highlighted && this.parts.get(this.highlighted);
    if (prev) prev.meshes.forEach((m) => { if (m.material.emissive) m.material.emissive.setRGB(0, 0, 0); });
    this.highlighted = id || null;
    const cur = id && this.parts.get(id);
    if (cur) cur.meshes.forEach((m) => { if (m.material.emissive) m.material.emissive.setRGB(0.16, 0.035, 0.015); });
    this.requestRender();
  }
}
