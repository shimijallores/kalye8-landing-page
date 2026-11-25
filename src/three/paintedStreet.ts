import * as THREE from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export const OUTLINE_LAYER = 1;

const DRACO_PATH = 'draco/';
const BASIS_PATH = 'basis/';
const STREET_TEXTURE_DIR = 'textures/street/';

const STREET_SURFACES: Record<string, { colour: boolean }> = {
  plaster: { colour: false },
  peel: { colour: false },
  block: { colour: false },
  planks: { colour: false },
  yero: { colour: true },
  yero_worn: { colour: true },
  shutter: { colour: false },
  road: { colour: false },
  walk: { colour: false },
  concrete: { colour: false },
  paintmetal: { colour: false },
  sign: { colour: true },
};

const DETAIL_GAIN = 1.25;
const HANGING = new Set(['cable']);
const NO_INK = new Set(['cable', 'lit_window', 'lit_bulb', 'lit_sign', 'sign', 'glass', 'road', 'walk']);
const FLAT_SHARED = new Set(['trim', 'metal', 'plastic', 'cloth', 'plant']);
const BOLD_INK = new Set(['plaster', 'peel', 'block', 'planks', 'yero', 'yero_worn', 'shutter', 'concrete']);
const STREET_INK_SHADOW = new THREE.Color('#4a3a30');

const STREET_GLOW: Record<string, { color: string; strength: number }> = {
  MAT_lit_window: { color: '#ffc66b', strength: 0.9 },
  MAT_lit_bulb: { color: '#fff0b8', strength: 1.0 },
  MAT_lit_sign: { color: '#ffd23f', strength: 0.7 },
};

const LAMP_LIGHT_POSITIONS: readonly [number, number, number][] = [-12, 0, 12].flatMap((z) => [
  [-17.15, 6.45, z] as [number, number, number],
  [17.15, 6.45, z] as [number, number, number],
]);

const DEFAULT_COURT = {
  width: 28,
  lineZ: [16, 6, -4, -14],
  startBaseZ: 22,
  farEndZoneZ: -21,
  jailBox: { x: -15.8, z: 1.0, width: 3.0, length: 8.0 },
} as const;

const SKY_RADIUS = 350;
const CLOUD_FIELD = 340;
const MAX_TOON_LAMPS = 6;

const OUTLINE_VERT = /* glsl */ `
#include <common>
#include <fog_pars_vertex>
attribute vec3 outlineNormal;
uniform float uWidthPx;
uniform float uWidthMul;
uniform float uRefDist;
uniform float uViewportH;
varying vec3 vOutWorld;
void main() {
  vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
  vec3 viewNormal = normalize( normalMatrix * outlineNormal );
  float depth = max( -mvPosition.z, 0.05 );
  float px = uWidthPx * uWidthMul * clamp( uRefDist / depth, 0.4, 1.8 );
  float metresPerPx = 2.0 * depth / ( uViewportH * projectionMatrix[ 1 ][ 1 ] );
  mvPosition.xyz += viewNormal * px * metresPerPx;
  gl_Position = projectionMatrix * mvPosition;
  vOutWorld = ( modelMatrix * vec4( position, 1.0 ) ).xyz;
  #include <fog_vertex>
}
`;

const OUTLINE_FRAG = /* glsl */ `
uniform vec3 uInk;
uniform vec3 uShadowTint;
uniform float uBright;
#include <fog_pars_fragment>
uniform vec3 uHazeColor;
uniform float uHazeAmount;
uniform vec2 uHazeHalf;
uniform vec2 uHazeRange;
varying vec3 vOutWorld;
void main() {
  gl_FragColor = vec4( uInk * uShadowTint * uBright, 1.0 );
  if ( uHazeAmount > 0.0 ) {
    vec2 hzOut = max( abs( vOutWorld.xz ) - uHazeHalf, vec2( 0.0 ) );
    float hz = smoothstep( uHazeRange.x, uHazeRange.y, length( hzOut ) ) * uHazeAmount;
    gl_FragColor.rgb = mix( gl_FragColor.rgb, uHazeColor, min( 1.0, hz * 1.15 ) );
  }
  #include <colorspace_fragment>
  #include <fog_fragment>
}
`;

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize( position );
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

const SKY_FRAG = /* glsl */ `
uniform vec3 topColor;
uniform vec3 horizonColor;
uniform vec3 sunTint;
uniform float glow;
varying vec3 vDir;
void main() {
  vec3 d = normalize( vDir );
  vec3 mid = mix( horizonColor, topColor, 0.45 ) + sunTint * 0.06 * glow;
  vec3 col = mix( horizonColor, mid, smoothstep( -0.02, 0.22, d.y ) );
  col = mix( col, topColor, smoothstep( 0.14, 0.75, d.y ) );
  col += ( fract( sin( dot( gl_FragCoord.xy, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ) - 0.5 ) / 255.0;
  gl_FragColor = vec4( col, 1.0 );
  #include <colorspace_fragment>
}
`;

function kindOf(materialName: string): string {
  return materialName.replace(/^MAT_/, '');
}

function ktx2Name(file: string): string {
  return file.replace(/\.(jpe?g|png)$/i, '') + '.ktx2';
}

function flagOf(mesh: THREE.Object3D, key: string): boolean {
  for (let o: THREE.Object3D | null = mesh; o; o = o.parent) {
    if (o.userData?.[key]) return true;
  }
  return false;
}

function wearFileOf(mesh: THREE.Object3D): string | undefined {
  for (let o: THREE.Object3D | null = mesh; o; o = o.parent) {
    const w = o.userData?.wear;
    if (typeof w === 'string') return w;
  }
  return undefined;
}

function sunDirection(azDeg: number, elDeg: number): THREE.Vector3 {
  const az = THREE.MathUtils.degToRad(azDeg);
  const el = THREE.MathUtils.degToRad(elDeg);
  return new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).normalize();
}

function inkColorOf(shadow: THREE.Color): THREE.Color {
  const hsl = { h: 0, s: 0, l: 0 };
  shadow.getHSL(hsl);
  return new THREE.Color().setHSL(hsl.h, Math.min(1, hsl.s * 1.1), hsl.l * 0.32);
}

function ensureOutlineNormals(geo: THREE.BufferGeometry): void {
  if (geo.getAttribute('outlineNormal')) return;
  const pos = geo.getAttribute('position') as THREE.BufferAttribute | undefined;
  if (!pos) return;
  const count = pos.count;
  geo.computeBoundingBox();
  const box = geo.boundingBox!;
  const size = box.getSize(new THREE.Vector3());
  const q = Math.max(size.x, size.y, size.z, 1e-6) / 65535;
  const keys = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const ix = Math.round((pos.getX(i) - box.min.x) / q);
    const iy = Math.round((pos.getY(i) - box.min.y) / q);
    const iz = Math.round((pos.getZ(i) - box.min.z) / q);
    keys[i] = (ix * 65536 + iy) * 65536 + iz;
  }

  const sums = new Map<number, [number, number, number]>();
  const index = geo.index;
  const triCount = (index ? index.count : count) / 3;
  const p0 = new THREE.Vector3();
  const p1 = new THREE.Vector3();
  const p2 = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const faceN = new THREE.Vector3();

  const acc = (i: number, from: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3) => {
    e1.subVectors(a, from).normalize();
    e2.subVectors(b, from).normalize();
    const angle = Math.acos(Math.min(1, Math.max(-1, e1.dot(e2))));
    let s = sums.get(keys[i]);
    if (!s) sums.set(keys[i], (s = [0, 0, 0]));
    s[0] += faceN.x * angle;
    s[1] += faceN.y * angle;
    s[2] += faceN.z * angle;
  };

  for (let t = 0; t < triCount; t++) {
    const i0 = index ? index.getX(t * 3) : t * 3;
    const i1 = index ? index.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = index ? index.getX(t * 3 + 2) : t * 3 + 2;
    p0.fromBufferAttribute(pos, i0);
    p1.fromBufferAttribute(pos, i1);
    p2.fromBufferAttribute(pos, i2);
    faceN.crossVectors(e1.subVectors(p1, p0), e2.subVectors(p2, p0));
    if (faceN.lengthSq() < 1e-18) continue;
    faceN.normalize();
    acc(i0, p0, p1, p2);
    acc(i1, p1, p2, p0);
    acc(i2, p2, p0, p1);
  }

  const out = new Float32Array(count * 3);
  const normalAttr = geo.getAttribute('normal');
  for (let i = 0; i < count; i++) {
    const s = sums.get(keys[i]);
    const len = s ? Math.hypot(s[0], s[1], s[2]) : 0;
    if (s && len > 1e-9) {
      out[i * 3] = s[0] / len;
      out[i * 3 + 1] = s[1] / len;
      out[i * 3 + 2] = s[2] / len;
    } else {
      out[i * 3] = normalAttr ? normalAttr.getX(i) : 0;
      out[i * 3 + 1] = normalAttr ? normalAttr.getY(i) : 1;
      out[i * 3 + 2] = normalAttr ? normalAttr.getZ(i) : 0;
    }
  }
  geo.setAttribute('outlineNormal', new THREE.BufferAttribute(out, 3));
}

type P2 = readonly [number, number];

function hull(points: P2[]): P2[] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: P2, a: P2, b: P2): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: P2[]): P2[] => {
    const out: P2[] = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...half(pts), ...half([...pts].reverse())];
}

function hullsOverlap(a: P2[], b: P2[]): boolean {
  for (const poly of [a, b]) {
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const q = poly[(i + 1) % poly.length];
      const ax = -(q[1] - p[1]);
      const az = q[0] - p[0];
      const range = (h: P2[]): [number, number] => {
        let lo = Infinity;
        let hi = -Infinity;
        for (const v of h) {
          const d = v[0] * ax + v[1] * az;
          lo = Math.min(lo, d);
          hi = Math.max(hi, d);
        }
        return [lo, hi];
      };
      const [aLo, aHi] = range(a);
      const [bLo, bHi] = range(b);
      if (aHi <= bLo || bHi <= aLo) return false;
    }
  }
  return true;
}

function separateRoadLayers(geometry: THREE.BufferGeometry): void {
  const pos = geometry.getAttribute('position');
  const index = geometry.getIndex();
  if (!pos || !index) return;
  const onLayer = (v: number): boolean => {
    const y = pos.getY(v);
    return y > 0.0015 && y < 0.0028;
  };
  const parent = new Map<number, number>();
  const find = (a: number): number => {
    let r = a;
    while (parent.get(r) !== r) r = parent.get(r)!;
    parent.set(a, r);
    return r;
  };
  const add = (a: number): void => {
    if (!parent.has(a)) parent.set(a, a);
  };
  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i);
    const b = index.getX(i + 1);
    const c = index.getX(i + 2);
    if (!onLayer(a) || !onLayer(b) || !onLayer(c)) continue;
    add(a);
    add(b);
    add(c);
    parent.set(find(b), find(a));
    parent.set(find(c), find(a));
  }
  const groups = new Map<number, number[]>();
  for (const v of parent.keys()) {
    const r = find(v);
    (groups.get(r) ?? groups.set(r, []).get(r)!).push(v);
  }
  const kept: { patch: boolean; outline: P2[] }[] = [];
  const droppedVerts = new Set<number>();
  for (const verts of groups.values()) {
    const patch = verts.length <= 6;
    const outline = hull(verts.map((v): P2 => [pos.getX(v), pos.getZ(v)]));
    if (kept.some((k) => k.patch === patch && hullsOverlap(k.outline, outline))) {
      for (const v of verts) droppedVerts.add(v);
      continue;
    }
    kept.push({ patch, outline });
    for (const v of verts) pos.setY(v, patch ? 0.001 : 0.002);
  }
  if (droppedVerts.size > 0) {
    const out: number[] = [];
    for (let i = 0; i < index.count; i += 3) {
      if (droppedVerts.has(index.getX(i))) continue;
      out.push(index.getX(i), index.getX(i + 1), index.getX(i + 2));
    }
    geometry.setIndex(out);
  }
  pos.needsUpdate = true;
}

function createChalkTexture(): THREE.CanvasTexture {
  const w = 64;
  const h = 32;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  const inset = Math.round(h * 0.06);
  ctx.fillRect(0, inset, w, h - 2 * inset);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.anisotropy = 8;
  return tex;
}

function createChalkLine(
  texture: THREE.CanvasTexture,
  color: string,
  length: number,
  width: number,
  y: number,
  renderOrder = 3,
): THREE.Mesh {
  const tex = texture.clone();
  tex.repeat.set(Math.max(1, length / 2.5), 1);
  tex.needsUpdate = true;
  const mat = new THREE.MeshBasicMaterial({
    color,
    map: tex,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: 0,
    polygonOffsetUnits: -1,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(length, width), mat);
  mesh.rotation.order = 'YXZ';
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.renderOrder = renderOrder;
  return mesh;
}

function buildCourtMarkings(): THREE.Group {
  const group = new THREE.Group();
  const chalkTex = createChalkTexture();
  const baseZoneMat = new THREE.MeshBasicMaterial({
    color: '#ef4444',
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  baseZoneMat.forceSinglePass = true;

  for (const z of DEFAULT_COURT.lineZ) {
    const line = createChalkLine(chalkTex, '#f8fafc', DEFAULT_COURT.width, 0.3, 0.02);
    line.position.set(0, 0.02, z);
    group.add(line);
  }

  const spineLength = Math.abs(DEFAULT_COURT.lineZ[0] - DEFAULT_COURT.lineZ[2]);
  const spineCenterZ = (DEFAULT_COURT.lineZ[0] + DEFAULT_COURT.lineZ[2]) / 2;
  const spine = createChalkLine(chalkTex, '#fde047', spineLength, 0.34, 0.025, 4);
  spine.rotation.y = Math.PI / 2;
  spine.position.set(0, 0.025, spineCenterZ);
  group.add(spine);

  const zoneDepth = 4;
  const outlineNorth = DEFAULT_COURT.farEndZoneZ - zoneDepth / 2;
  const outlineSouth = DEFAULT_COURT.startBaseZ + zoneDepth / 2;
  const sideLength = outlineSouth - outlineNorth;
  const sideCenterZ = (outlineSouth + outlineNorth) / 2;
  for (const xSign of [-1, 1]) {
    const side = createChalkLine(chalkTex, '#f8fafc', sideLength, 0.26, 0.02);
    side.rotation.y = Math.PI / 2;
    side.position.set(xSign * (DEFAULT_COURT.width / 2), 0.02, sideCenterZ);
    group.add(side);
  }
  for (const z of [outlineNorth, outlineSouth]) {
    const end = createChalkLine(chalkTex, '#f8fafc', DEFAULT_COURT.width + 0.26, 0.26, 0.02);
    end.position.set(0, 0.02, z);
    group.add(end);
  }

  for (const z of [DEFAULT_COURT.startBaseZ, DEFAULT_COURT.farEndZoneZ]) {
    const zoneMesh = new THREE.Mesh(new THREE.PlaneGeometry(DEFAULT_COURT.width, zoneDepth), baseZoneMat);
    zoneMesh.rotation.x = -Math.PI / 2;
    zoneMesh.position.set(0, 0.015, z);
    zoneMesh.renderOrder = 2;
    group.add(zoneMesh);
  }

  const jailMat = new THREE.MeshBasicMaterial({
    color: '#dc2626',
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  const jailMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(DEFAULT_COURT.jailBox.width, DEFAULT_COURT.jailBox.length),
    jailMat,
  );
  jailMesh.rotation.x = -Math.PI / 2;
  jailMesh.position.set(DEFAULT_COURT.jailBox.x, 0.02, DEFAULT_COURT.jailBox.z);
  jailMesh.renderOrder = 2;
  group.add(jailMesh);

  return group;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildCloudGeometry(rand: () => number): THREE.BufferGeometry {
  const puffs = 5 + Math.floor(rand() * 3);
  const parts: THREE.BufferGeometry[] = [];
  for (let i = 0; i < puffs; i++) {
    const g = new THREE.IcosahedronGeometry(1, 1);
    const centre = 1 - Math.abs(i - (puffs - 1) / 2) / (puffs / 2);
    const r = 0.55 + 0.55 * centre + rand() * 0.25;
    g.scale(r * 1.25, r * 0.8, r);
    g.translate((i - (puffs - 1) / 2) * 0.95 + (rand() - 0.5) * 0.3, r * 0.35 + rand() * 0.15, (rand() - 0.5) * 0.7);
    parts.push(g);
  }
  const merged = mergeGeometries(parts, false)!;
  parts.forEach((p) => p.dispose());

  const pos = merged.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) if (pos.getY(i) < 0) pos.setY(i, 0);
  merged.computeVertexNormals();

  const nrm = merged.getAttribute('normal') as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const shade = new THREE.Color('#6f5c7a');
  const light = new THREE.Color('#d8b4a0');
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    c.copy(shade).lerp(light, nrm.getY(i) > 0.12 ? 1 : 0);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  merged.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  merged.deleteAttribute('uv');
  return merged;
}

export interface RealStreetWorld {
  skyMesh: THREE.Mesh;
  clouds: { mesh: THREE.Mesh; speed: number }[];
  updateViewport(bufH: number): void;
  updateClouds(dt: number): void;
}

export interface RealStreetOptions {
  camPos: THREE.Vector3;
  withShadows?: boolean;
  withClouds?: boolean;
  programKeyPrefix: string;
  isDisposed: () => boolean;
}

export async function loadRealStreetWorld(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  opts: RealStreetOptions,
): Promise<RealStreetWorld> {
  const sunDir = sunDirection(26, 11);
  const streetLight = {
    sunDir: { value: sunDir },
    litTint: { value: new THREE.Color(1, 0.98, 0.95) },
    shadowTint: { value: new THREE.Color(0.88, 0.85, 0.96) },
    brightness: { value: 0.98 },
  };
  const streetHaze = {
    color: { value: new THREE.Color('#9d5b42') },
    amount: { value: 0.78 },
    half: { value: new THREE.Vector2(14, 25) },
    range: { value: new THREE.Vector2(4, 54) },
  };
  const streetShadowFade = {
    depthRange: { value: 199 },
    fade: { value: new THREE.Vector3(2.8, 5.5, 0.55) },
  };
  const streetLamps = {
    pos: { value: LAMP_LIGHT_POSITIONS.map(([x, y, z]) => new THREE.Vector3(x, y, z)) },
    color: { value: new THREE.Color('#ffb85c') },
    on: { value: 0.72 },
    radius: { value: 9.5 },
  };
  const outlineShared = {
    uWidthPx: { value: 2.0 },
    uRefDist: { value: 16 },
    uViewportH: { value: 1080 },
  };

  scene.background = new THREE.Color('#09090b');
  scene.fog = new THREE.FogExp2('#9d5b42', 0.0075);

  const hemi = new THREE.HemisphereLight('#f5a623', '#241d38', 1.05);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight('#f5a623', 2.4);
  sun.position.copy(sunDir).multiplyScalar(80);
  if (opts.withShadows) {
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -24;
    sc.right = 24;
    sc.top = 24;
    sc.bottom = -24;
    sc.near = 1;
    sc.far = 200;
    streetShadowFade.depthRange.value = sc.far - sc.near;
    sun.shadow.bias = -0.0022;
    sun.shadow.normalBias = 0.09;
  }
  scene.add(sun, sun.target);

  const skyMat = new THREE.ShaderMaterial({
    uniforms: {
      topColor: { value: new THREE.Color('#3a2d6e') },
      horizonColor: { value: new THREE.Color('#9d5b42') },
      sunTint: { value: new THREE.Color('#f5a623') },
      glow: { value: 0.75 },
    },
    vertexShader: SKY_VERT,
    fragmentShader: SKY_FRAG,
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
  });
  const skyMesh = new THREE.Mesh(new THREE.SphereGeometry(SKY_RADIUS, 28, 16), skyMat);
  skyMesh.position.copy(opts.camPos);
  skyMesh.renderOrder = -10;
  skyMesh.frustumCulled = false;
  scene.add(skyMesh);

  const clouds: { mesh: THREE.Mesh; speed: number }[] = [];
  if (opts.withClouds) {
    const rand = mulberry32(7);
    const cloudGeos = [0, 1, 2].map(() => buildCloudGeometry(rand));
    for (let i = 0; i < 14; i++) {
      const mat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.52,
        depthWrite: false,
        fog: false,
      });
      const mesh = new THREE.Mesh(cloudGeos[i % cloudGeos.length], mat);
      const s = 8 + rand() * 9;
      mesh.scale.set(s, s * 0.55, s * 0.8);
      mesh.rotation.y = rand() * Math.PI * 2;
      const ang = rand() * Math.PI * 2;
      const rad = 160 + rand() * 140;
      const el = THREE.MathUtils.degToRad(9 + rand() * 18);
      mesh.position.set(Math.cos(ang) * rad, Math.tan(el) * rad, Math.sin(ang) * rad);
      mesh.renderOrder = -4;
      scene.add(mesh);
      clouds.push({ mesh, speed: 1.0 + rand() * 1.4 });
    }
  }

  scene.add(buildCourtMarkings());

  const toonMatCache = new Map<string, THREE.MeshToonMaterial>();
  const outlineMatCache = new Map<string, THREE.ShaderMaterial>();

  const createPaintedToonMaterial = (
    name: string,
    mOpts: { glowColor?: string; glow?: number; detailGain: number; cacheTag: string },
  ): THREE.MeshToonMaterial => {
    const key = `${name}|${mOpts.glowColor ?? ''}|${mOpts.glow ?? 0}|${mOpts.detailGain}|${mOpts.cacheTag}`;
    const hit = toonMatCache.get(key);
    if (hit) return hit;

    const mat = new THREE.MeshToonMaterial({
      name,
      color: '#ffffff',
      side: THREE.DoubleSide,
      vertexColors: true,
    });
    mat.toneMapped = false;

    const uGlow = { value: mOpts.glow ?? 0 };
    const uGlowColor = { value: new THREE.Color(mOpts.glowColor ?? '#ffffff') };
    const uDetailGain = { value: mOpts.detailGain };

    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uToonGlow = uGlow;
      shader.uniforms.uToonGlowColor = uGlowColor;
      shader.uniforms.uToonDetailGain = uDetailGain;
      shader.uniforms.uToonSun = streetLight.sunDir;
      shader.uniforms.uToonLitTint = streetLight.litTint;
      shader.uniforms.uToonShadowTint = streetLight.shadowTint;
      shader.uniforms.uToonBright = streetLight.brightness;
      shader.uniforms.uHazeColor = streetHaze.color;
      shader.uniforms.uHazeAmount = streetHaze.amount;
      shader.uniforms.uHazeHalf = streetHaze.half;
      shader.uniforms.uHazeRange = streetHaze.range;
      shader.uniforms.uShadowDepthRange = streetShadowFade.depthRange;
      shader.uniforms.uShadowFade = streetShadowFade.fade;
      shader.uniforms.uLampPos = streetLamps.pos;
      shader.uniforms.uLampColor = streetLamps.color;
      shader.uniforms.uLampOn = streetLamps.on;
      shader.uniforms.uLampRadius = streetLamps.radius;

      shader.vertexShader = shader.vertexShader
        .replace('void main() {', 'varying vec3 vToonWorld;\nvoid main() {')
        .replace(
          '#include <project_vertex>',
          '#include <project_vertex>\n  vToonWorld = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;',
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          'void main() {',
          `uniform float uToonGlow;
uniform vec3 uToonGlowColor;
uniform float uToonDetailGain;
uniform vec3 uToonSun;
uniform vec3 uToonLitTint;
uniform vec3 uToonShadowTint;
uniform float uToonBright;
uniform vec3 uHazeColor;
uniform float uHazeAmount;
uniform vec2 uHazeHalf;
uniform vec2 uHazeRange;
uniform float uShadowDepthRange;
uniform vec3 uShadowFade;
uniform vec3 uLampPos[ ${MAX_TOON_LAMPS} ];
uniform vec3 uLampColor;
uniform float uLampOn;
uniform float uLampRadius;
varying vec3 vToonWorld;
void main() {`,
        )
        .replace(
          '#include <opaque_fragment>',
          `{
            vec3 toonN = normalize( normal );
            vec3 toonL = normalize( ( viewMatrix * vec4( uToonSun, 0.0 ) ).xyz );
            float toonNdl = dot( toonN, toonL ) * 0.5 + 0.5;
            float toonW = clamp( fwidth( toonNdl ), 0.0005, 0.08 );
            float toonLit = smoothstep( 0.45 - toonW, 0.45 + toonW, toonNdl );
            #if defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 )
              DirectionalLightShadow toonS = directionalLightShadows[ 0 ];
              float toonSh = getShadow( directionalShadowMap[ 0 ], toonS.shadowMapSize, 1.0, toonS.shadowBias, toonS.shadowRadius, vDirectionalShadowCoord[ 0 ] );
              float toonShW = clamp( fwidth( toonSh ) * 0.75, 0.02, 0.25 );
              if ( receiveShadow ) {
                float toonInShadow = 1.0 - smoothstep( 0.5 - toonShW, 0.5 + toonShW, toonSh );
                vec4 toonSc = vDirectionalShadowCoord[ 0 ];
                toonSc.xyz /= toonSc.w;
                float toonOcc = unpackRGBAToDepth( texture2D( directionalShadowMap[ 0 ], toonSc.xy ) );
                float toonCasterH = max( toonSc.z + toonS.shadowBias - toonOcc, 0.0 ) * uShadowDepthRange * max( uToonSun.y, 0.05 );
                float toonTall = smoothstep( uShadowFade.x, uShadowFade.y, toonCasterH );
                toonLit *= 1.0 - toonInShadow * ( 1.0 - toonTall * uShadowFade.z );
              }
            #endif
            vec3 toonBase = vColor.rgb;
            vec3 toonShade = toonBase * vec3( 0.66, 0.58, 0.55 ) + vec3( 0.008, 0.003, 0.0 );
            vec3 toonCol = mix( toonShade * uToonShadowTint, toonBase * uToonLitTint, toonLit ) * uToonBright;
            if ( uLampOn > 0.001 ) {
              vec3 toonLampSum = vec3( 0.0 );
              for ( int li = 0; li < ${MAX_TOON_LAMPS}; li++ ) {
                vec3 toonLd = uLampPos[ li ] - vToonWorld;
                float toonLdist = length( toonLd );
                float toonAtt = clamp( 1.0 - toonLdist / uLampRadius, 0.0, 1.0 );
                toonAtt *= toonAtt;
                vec3 toonLv = normalize( ( viewMatrix * vec4( toonLd / max( toonLdist, 0.001 ), 0.0 ) ).xyz );
                float toonFace = clamp( dot( toonN, toonLv ) * 0.65 + 0.35, 0.0, 1.0 );
                toonLampSum += toonAtt * toonFace;
              }
              toonCol += toonBase * uLampColor * toonLampSum * uLampOn * 1.15;
            }
            #ifdef USE_MAP
              toonCol *= sampledDiffuseColor.rgb * uToonDetailGain;
            #endif
            #ifdef USE_LIGHTMAP
              toonCol *= texture2D( lightMap, vLightMapUv ).rgb;
            #endif
            outgoingLight = mix( toonCol, uToonGlowColor, uToonGlow );
            if ( uHazeAmount > 0.0 ) {
              vec2 hzOut = max( abs( vToonWorld.xz ) - uHazeHalf, vec2( 0.0 ) );
              float hzD = length( hzOut );
              float hz = smoothstep( uHazeRange.x, uHazeRange.y, hzD ) * uHazeAmount;
              float hzLum = dot( outgoingLight, vec3( 0.299, 0.587, 0.114 ) );
              outgoingLight = mix( mix( outgoingLight, vec3( hzLum ), hz * 0.5 ), uHazeColor, hz );
            }
          }
          #include <opaque_fragment>`,
        );
    };

    mat.customProgramCacheKey = () => `${opts.programKeyPrefix}-toon-v1`;
    toonMatCache.set(key, mat);
    return mat;
  };

  const getOutlineMaterial = (shadow: THREE.Color, widthMul: number): THREE.ShaderMaterial => {
    const ink = inkColorOf(shadow);
    const key = `${ink.getHexString()}|${widthMul}`;
    const hit = outlineMatCache.get(key);
    if (hit) return hit;

    const mat = new THREE.ShaderMaterial({
      name: `${opts.programKeyPrefix}-outline`,
      vertexShader: OUTLINE_VERT,
      fragmentShader: OUTLINE_FRAG,
      side: THREE.BackSide,
      fog: true,
      uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.fog,
        { uInk: { value: ink }, uWidthMul: { value: widthMul } },
      ]),
    });
    mat.uniforms.uWidthPx = outlineShared.uWidthPx;
    mat.uniforms.uRefDist = outlineShared.uRefDist;
    mat.uniforms.uViewportH = outlineShared.uViewportH;
    mat.uniforms.uShadowTint = streetLight.shadowTint;
    mat.uniforms.uBright = streetLight.brightness;
    mat.uniforms.uHazeColor = streetHaze.color;
    mat.uniforms.uHazeAmount = streetHaze.amount;
    mat.uniforms.uHazeHalf = streetHaze.half;
    mat.uniforms.uHazeRange = streetHaze.range;
    mat.customProgramCacheKey = () => `${opts.programKeyPrefix}-outline-v1`;
    outlineMatCache.set(key, mat);
    return mat;
  };

  const attachStreetOutline = (mesh: THREE.Mesh, widthMul: number): void => {
    const geo = mesh.geometry;
    if (!geo.getAttribute('position') || !geo.getAttribute('normal')) return;
    ensureOutlineNormals(geo);
    const hullMesh = new THREE.Mesh(geo, getOutlineMaterial(STREET_INK_SHADOW, widthMul));
    hullMesh.name = 'OUTLINE';
    hullMesh.userData.isOutline = true;
    hullMesh.layers.set(OUTLINE_LAYER);
    hullMesh.castShadow = false;
    hullMesh.receiveShadow = false;
    hullMesh.frustumCulled = mesh.frustumCulled;
    hullMesh.matrixAutoUpdate = false;
    hullMesh.raycast = () => {};
    mesh.add(hullMesh);
  };

  const draco = new DRACOLoader();
  draco.setDecoderPath(DRACO_PATH);
  const gltfLoader = new GLTFLoader();
  gltfLoader.setDRACOLoader(draco);
  const ktx2Loader = new KTX2Loader().setTranscoderPath(BASIS_PATH).detectSupport(renderer);

  try {
    const detailTextures = new Map<string, THREE.Texture>();
    const wearCache = new Map<string, Promise<THREE.Texture>>();

    const [streetGltf, beyondGltf] = await Promise.all([
      gltfLoader.loadAsync('models/kalye_street.glb'),
      gltfLoader.loadAsync('models/kalye_beyond.glb'),
      Promise.all(
        Object.entries(STREET_SURFACES).map(async ([kind, { colour }]) => {
          const t = await ktx2Loader.loadAsync(`${STREET_TEXTURE_DIR}${kind}.ktx2`);
          t.wrapS = t.wrapT = THREE.RepeatWrapping;
          t.flipY = false;
          t.colorSpace = colour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
          t.anisotropy = 8;
          t.generateMipmaps = false;
          t.minFilter = THREE.LinearMipmapLinearFilter;
          t.magFilter = THREE.LinearFilter;
          detailTextures.set(kind, t);
        }),
      ),
    ]);

    if (!opts.isDisposed()) {
      const getWear = (file: string): Promise<THREE.Texture> => {
        let p = wearCache.get(file);
        if (!p) {
          p = ktx2Loader.loadAsync(`${STREET_TEXTURE_DIR}${ktx2Name(file)}`).then((t) => {
            t.colorSpace = THREE.NoColorSpace;
            t.channel = 1;
            t.flipY = false;
            t.generateMipmaps = false;
            return t;
          });
          wearCache.set(file, p);
        }
        return p;
      };

      const paintGroup = async (root: THREE.Object3D): Promise<void> => {
        const meshes: THREE.Mesh[] = [];
        root.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (mesh.isMesh) meshes.push(mesh);
        });
        const wearFiles = [...new Set(meshes.map(wearFileOf).filter((w): w is string => !!w))];
        const wears = new Map(await Promise.all(wearFiles.map(async (f) => [f, await getWear(f)] as const)));

        for (const mesh of meshes) {
          const src = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.Material;
          const kind = kindOf(src.name);
          if (kind === 'road') separateRoadLayers(mesh.geometry);
          const name = FLAT_SHARED.has(kind) ? 'MAT_flat' : src.name;
          const wearFile = wearFileOf(mesh);
          const glow = STREET_GLOW[name];
          const surface = STREET_SURFACES[kind];

          const mat = createPaintedToonMaterial(name, {
            glowColor: glow?.color,
            glow: glow ? glow.strength * 0.78 : 0,
            detailGain: surface && !surface.colour ? DETAIL_GAIN : 1,
            cacheTag: wearFile ?? 'none',
          });
          if (surface) mat.map = detailTextures.get(kind) ?? null;
          if (wearFile) mat.lightMap = wears.get(wearFile) ?? null;
          mat.needsUpdate = true;
          mesh.material = mat;

          const noShadow = flagOf(mesh, 'noShadow');
          mesh.receiveShadow = !!opts.withShadows && !HANGING.has(kind) && !noShadow;
          mesh.castShadow =
            !!opts.withShadows && !HANGING.has(kind) && !name.startsWith('MAT_lit_') && kind !== 'road' && !noShadow;
          mesh.userData.inkWidth =
            !NO_INK.has(kind) && !glow && !flagOf(mesh, 'noInk') ? (BOLD_INK.has(kind) ? 1.8 : 1.25) : 0;
        }

        root.updateMatrixWorld(true);
        const toRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
        const groups = new Map<string, THREE.Mesh[]>();
        root.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh || mesh.userData.isOutline === true || Array.isArray(mesh.material)) return;
          const key = `${(mesh.material as THREE.Material).uuid}|${mesh.castShadow ? 1 : 0}${mesh.receiveShadow ? 1 : 0}|${mesh.userData.inkWidth ?? 0}`;
          const list = groups.get(key);
          if (list) list.push(mesh);
          else groups.set(key, [mesh]);
        });

        const mat4 = new THREE.Matrix4();
        for (const groupMeshes of groups.values()) {
          let merged = groupMeshes[0];
          if (groupMeshes.length > 1) {
            const attrs = Object.keys(groupMeshes[0].geometry.attributes).filter((n) =>
              groupMeshes.every((x) => x.geometry.getAttribute(n)),
            );
            const geos = groupMeshes.map((x) => {
              const g = new THREE.BufferGeometry();
              for (const n of attrs) g.setAttribute(n, x.geometry.getAttribute(n).clone());
              if (x.geometry.index) g.setIndex(x.geometry.index.clone());
              g.applyMatrix4(mat4.multiplyMatrices(toRoot, x.matrixWorld));
              return g;
            });
            const geo =
              geos.every((g) => !!g.index) || geos.every((g) => !g.index) ? mergeGeometries(geos, false) : null;
            for (const g of geos) g.dispose();
            if (geo) {
              const first = groupMeshes[0];
              merged = new THREE.Mesh(geo, first.material);
              merged.castShadow = first.castShadow;
              merged.receiveShadow = first.receiveShadow;
              merged.userData.inkWidth = first.userData.inkWidth;
              for (const x of groupMeshes) {
                x.removeFromParent();
                x.geometry.dispose();
              }
              root.add(merged);
            }
          }
          if (merged.userData.inkWidth) attachStreetOutline(merged, merged.userData.inkWidth);
        }
      };

      await Promise.all([paintGroup(streetGltf.scene), paintGroup(beyondGltf.scene)]);
      if (!opts.isDisposed()) {
        scene.add(streetGltf.scene, beyondGltf.scene);
      }
    }
  } finally {
    draco.dispose();
    ktx2Loader.dispose();
  }

  return {
    skyMesh,
    clouds,
    updateViewport(bufH: number) {
      outlineShared.uViewportH.value = Math.max(1, bufH);
      outlineShared.uWidthPx.value = Math.min(3.5, Math.max(1.2, (2.0 * bufH) / 1080));
    },
    updateClouds(dt: number) {
      for (const cloud of clouds) {
        cloud.mesh.position.x += cloud.speed * dt;
        if (cloud.mesh.position.x > CLOUD_FIELD) cloud.mesh.position.x = -CLOUD_FIELD;
      }
    },
  };
}

