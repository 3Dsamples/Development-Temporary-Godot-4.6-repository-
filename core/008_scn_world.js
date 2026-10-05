// File : 008
// name : src/core/008_scn_world.js
// description : Scene/world bootstrap that instantiates the ProceduralWorldCore
//               (core/world.js) as the single authoritative scene authority for
//               the complete modular anime lighting system. Bridges the
//               world core's procedural biome/palette pipeline into the
//               lighting orchestrator chain (LightingSystem → UniversalLightManager
//               → GISystem → AOSystem → EnvironmentSystem → InteriorSystem →
//               ExteriorSystem → AnimeLightDirector → ReferenceLightMatcher).
//               Owns the sole bitECS 0.4.0 world instance (plain-object SoA
//               components, no defineComponent/Types — 0.4.0 architectural
//               redesign), the sole THREE.WebGLRenderer, the sole Scene, and
//               the sole PerspectiveCamera. Every downstream lighting module
//               imports the handles exported here so entity IDs, palette
//               slots, and light uniform layouts stay consistent across CPU
//               sampling and GLSL uniforms. Enforces the Three.js-r185 lights
//               only policy: AmbientLight, HemisphereLight, DirectionalLight,
//               PointLight, SpotLight, RectAreaLight — nothing else emits
//               light. Zero per-frame allocations, fixed MAX_ENTITIES typed
//               arrays, frame-rate independent damping, Android-mobile tuned
//               (DPR clamp, powerPreference high-performance, no MSAA, no
//               stencil, PCF shadow tier by PERF_TIER).
// best for : Single point-of-entry for the entire anime lighting stack — every
//            subsequent system module (lighting, shadow, GI, AO, environment,
//            indoor, outdoor, post) receives its world/renderer/scene/camera/
//            palette/biome handles from this file so there is exactly one
//            source of truth and zero cross-module drift.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  createWorld,
  addEntity,
  removeEntity,
  addComponent,
  removeComponent,
  hasComponent,
  query,
  entityExists,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

import {
  ProceduralWorldCore,
  createProceduralWorldCore,
  Biome,
  PaletteSlot,
  Palettes,
  CoreMath,
} from '../../world.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS — single source of truth for the whole lighting stack */
/* ------------------------------------------------------------------ */

export const MAX_ENTITIES = 100000;

const PERF_TIER =
  (typeof navigator !== 'undefined' && navigator.deviceMemory >= 8 && navigator.hardwareConcurrency >= 8)
    ? 'HIGH'
    : (typeof navigator !== 'undefined' && navigator.deviceMemory >= 4)
      ? 'MEDIUM'
      : 'LOW';

const MOBILE_DPR_CAP = PERF_TIER === 'HIGH' ? 2.0 : PERF_TIER === 'MEDIUM' ? 1.75 : 1.25;

/* ------------------------------------------------------------------ */
/* 1. SoA COMPONENTS (bitECS 0.4.0 — plain objects, no Types/define)   */
/*    Every lighting module writes into these arrays directly.        */
/* ------------------------------------------------------------------ */

export const Transform = {
  x:  new Float32Array(MAX_ENTITIES),
  y:  new Float32Array(MAX_ENTITIES),
  z:  new Float32Array(MAX_ENTITIES),
  qx: new Float32Array(MAX_ENTITIES),
  qy: new Float32Array(MAX_ENTITIES),
  qz: new Float32Array(MAX_ENTITIES),
  qw: new Float32Array(MAX_ENTITIES),
  sx: new Float32Array(MAX_ENTITIES),
  sy: new Float32Array(MAX_ENTITIES),
  sz: new Float32Array(MAX_ENTITIES),
};

export const LightRef = {
  type:       new Uint8Array(MAX_ENTITIES),   // 0=sun 1=moon 2=hemi 3=ambient 4=point 5=spot 6=rect 7=emissive
  intensity:  new Float32Array(MAX_ENTITIES),
  colorR:     new Float32Array(MAX_ENTITIES),
  colorG:     new Float32Array(MAX_ENTITIES),
  colorB:     new Float32Array(MAX_ENTITIES),
  range:      new Float32Array(MAX_ENTITIES),
  spotAngle:  new Float32Array(MAX_ENTITIES),
  penumbra:   new Float32Array(MAX_ENTITIES),
  castShadow: new Uint8Array(MAX_ENTITIES),
  indoor:     new Uint8Array(MAX_ENTITIES),
  priority:   new Uint8Array(MAX_ENTITIES),
};

export const ShadowRef = {
  cascadeCount: new Uint8Array(MAX_ENTITIES),
  bias:         new Float32Array(MAX_ENTITIES),
  normalBias:   new Float32Array(MAX_ENTITIES),
  softness:     new Float32Array(MAX_ENTITIES),
  atlasTileX:   new Uint16Array(MAX_ENTITIES),
  atlasTileY:   new Uint16Array(MAX_ENTITIES),
  atlasTileW:   new Uint16Array(MAX_ENTITIES),
  atlasTileH:   new Uint16Array(MAX_ENTITIES),
};

export const GIRef = {
  irradianceR: new Float32Array(MAX_ENTITIES),
  irradianceG: new Float32Array(MAX_ENTITIES),
  irradianceB: new Float32Array(MAX_ENTITIES),
  skyOcclusion: new Float32Array(MAX_ENTITIES),
  indoorFactor: new Float32Array(MAX_ENTITIES),
  dirty:        new Uint8Array(MAX_ENTITIES),
};

export const AORef = {
  occlusion: new Float32Array(MAX_ENTITIES),
  radius:    new Float32Array(MAX_ENTITIES),
  intensity: new Float32Array(MAX_ENTITIES),
};

export const ActiveTag = new Uint8Array(MAX_ENTITIES);

/* ------------------------------------------------------------------ */
/* 2. ECS WORLD — single instance for the whole engine                */
/* ------------------------------------------------------------------ */

export const world = createWorld({
  components: {
    Transform,
    LightRef,
    ShadowRef,
    GIRef,
    AORef,
  },
  time: {
    delta: 0,
    elapsed: 0,
    then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
  },
});

/* ------------------------------------------------------------------ */
/* 3. CORE SINGLETON — ProceduralWorldCore owns renderer/scene/camera */
/* ------------------------------------------------------------------ */

let _coreInstance = null;
let _initialized = false;

const _lightEntities = {
  sun:  -1,
  moon: -1,
  hemi: -1,
  ambient: -1,
};

/* ------------------------------------------------------------------ */
/* 4. PUBLIC BOOTSTRAP                                                */
/* ------------------------------------------------------------------ */

export function initializeWorld(options = {}) {
  if (_initialized && _coreInstance) return _coreInstance;

  _coreInstance = createProceduralWorldCore({
    // Renderer — Android mobile tuned
    antialias: false,
    alpha: false,
    stencil: false,
    powerPreference: 'high-performance',
    pixelRatio: Math.min(
      options.pixelRatio || (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
      MOBILE_DPR_CAP
    ),
    enableShadows: options.enableShadows !== false,
    shadowResolution:
      options.shadowResolution ||
      (PERF_TIER === 'HIGH' ? 2048 : PERF_TIER === 'MEDIUM' ? 1024 : 512),

    // Camera
    fov: options.fov || 55,
    near: options.near || 0.1,
    far: options.far || 1000,
    cameraX: options.cameraX || 0,
    cameraY: options.cameraY || 8,
    cameraZ: options.cameraZ || 34,

    // Biome / palette
    biome: options.biome !== undefined ? options.biome : Biome.DESERT,
    biomeSpeed: options.biomeSpeed || 0.85,
    autoCycle: options.autoCycle === true,
    cycleTime: options.cycleTime || 28,

    // Lighting integration hooks
    timeOfDay: options.timeOfDay !== undefined ? options.timeOfDay : 0.38,
    daySpeed: options.daySpeed || 0.004,
    usePaletteLight: options.usePaletteLight !== false,

    // Layer control
    preloadAllLayers: options.preloadAllLayers === true,
    sceneShadows: options.sceneShadows === true,

    // Misc
    seed: options.seed | 0,
    ppu: options.ppu || 5.5,
    autostart: options.autostart !== false,
  });

  _spawnLightEntities();
  _syncActiveTag();

  _initialized = true;
  return _coreInstance;
}

/* ------------------------------------------------------------------ */
/* 5. LIGHT ENTITY SPAWNING (bitECS 0.4.0 addComponent on SoA objects) */
/* ------------------------------------------------------------------ */

function _spawnLightEntities() {
  _lightEntities.sun     = _makeLightEntity(LIGHT_TYPE.SUN,     1.00, 0.96, 0.85, 1.00, 0);
  _lightEntities.moon    = _makeLightEntity(LIGHT_TYPE.MOON,    0.42, 0.48, 0.70, 0.25, 0);
  _lightEntities.hemi    = _makeLightEntity(LIGHT_TYPE.HEMI,    0.45, 0.62, 0.85, 0.35, 0);
  _lightEntities.ambient = _makeLightEntity(LIGHT_TYPE.AMBIENT, 0.20, 0.25, 0.30, 0.15, 0);

  ActiveTag[_lightEntities.sun]     = 1;
  ActiveTag[_lightEntities.moon]    = 1;
  ActiveTag[_lightEntities.hemi]    = 1;
  ActiveTag[_lightEntities.ambient] = 1;
}

export const LIGHT_TYPE = Object.freeze({
  SUN: 0,
  MOON: 1,
  HEMI: 2,
  AMBIENT: 3,
  POINT: 4,
  SPOT: 5,
  RECT: 6,
  EMISSIVE: 7,
});

function _makeLightEntity(type, r, g, b, intensity, castShadow) {
  const eid = addEntity(world);

  Transform.x[eid] = 0;
  Transform.y[eid] = 0;
  Transform.z[eid] = 0;
  Transform.qx[eid] = 0;
  Transform.qy[eid] = 0;
  Transform.qz[eid] = 0;
  Transform.qw[eid] = 1;
  Transform.sx[eid] = 1;
  Transform.sy[eid] = 1;
  Transform.sz[eid] = 1;

  LightRef.type[eid]       = type;
  LightRef.intensity[eid]  = intensity;
  LightRef.colorR[eid]     = r;
  LightRef.colorG[eid]     = g;
  LightRef.colorB[eid]     = b;
  LightRef.range[eid]      = 0;
  LightRef.spotAngle[eid]  = 0;
  LightRef.penumbra[eid]   = 0;
  LightRef.castShadow[eid] = castShadow;
  LightRef.indoor[eid]     = 0;
  LightRef.priority[eid]   = 0;

  ShadowRef.cascadeCount[eid] = 0;
  ShadowRef.bias[eid]         = -0.0008;
  ShadowRef.normalBias[eid]   = 0.020;
  ShadowRef.softness[eid]     = 0.05;
  ShadowRef.atlasTileX[eid]   = 0;
  ShadowRef.atlasTileY[eid]   = 0;
  ShadowRef.atlasTileW[eid]   = 0;
  ShadowRef.atlasTileH[eid]   = 0;

  GIRef.irradianceR[eid] = 0;
  GIRef.irradianceG[eid] = 0;
  GIRef.irradianceB[eid] = 0;
  GIRef.skyOcclusion[eid] = 1;
  GIRef.indoorFactor[eid] = 0;
  GIRef.dirty[eid] = 1;

  AORef.occlusion[eid] = 1;
  AORef.radius[eid] = 2.0;
  AORef.intensity[eid] = 1.0;

  addComponent(world, eid, Transform);
  addComponent(world, eid, LightRef);
  addComponent(world, eid, ShadowRef);
  addComponent(world, eid, GIRef);
  addComponent(world, eid, AORef);

  return eid;
}

/* ------------------------------------------------------------------ */
/* 6. PER-FRAME LIGHT SYNC — reads bitECS → writes THREE + uniforms    */
/* ------------------------------------------------------------------ */

const _sunLightInstance  = { ref: null };
const _moonLightInstance = { ref: null };
const _hemiLightInstance = { ref: null };
const _ambLightInstance  = { ref: null };

export function bindThreeLights(threeRefs) {
  if (threeRefs.sun)     _sunLightInstance.ref  = threeRefs.sun;
  if (threeRefs.moon)    _moonLightInstance.ref = threeRefs.moon;
  if (threeRefs.hemi)    _hemiLightInstance.ref = threeRefs.hemi;
  if (threeRefs.ambient) _ambLightInstance.ref  = threeRefs.ambient;
}

const _tmpSunDir = new THREE.Vector3();
const _tmpCol = new THREE.Color();

export function syncLightEntitiesToThree(elapsed) {
  const core = _coreInstance;
  if (!core) return;

  // --- Sun ---
  if (_lightEntities.sun >= 0 && _sunLightInstance.ref) {
    const eid = _lightEntities.sun;
    const L = _sunLightInstance.ref;

    _tmpCol.setRGB(LightRef.colorR[eid], LightRef.colorG[eid], LightRef.colorB[eid]);
    L.color.copy(_tmpCol);
    L.intensity = LightRef.intensity[eid];

    _tmpSunDir.set(Transform.x[eid], Transform.y[eid], Transform.z[eid]);
    if (_tmpSunDir.lengthSq() > 1e-6) {
      _tmpSunDir.normalize();
      L.position.copy(_tmpSunDir).multiplyScalar(120);
      L.target.position.set(0, 0, 0);
      L.target.updateMatrixWorld();
    }
  }

  // --- Moon ---
  if (_lightEntities.moon >= 0 && _moonLightInstance.ref) {
    const eid = _lightEntities.moon;
    const L = _moonLightInstance.ref;

    _tmpCol.setRGB(LightRef.colorR[eid], LightRef.colorG[eid], LightRef.colorB[eid]);
    L.color.copy(_tmpCol);
    L.intensity = LightRef.intensity[eid];

    _tmpSunDir.set(Transform.x[eid], Transform.y[eid], Transform.z[eid]);
    if (_tmpSunDir.lengthSq() > 1e-6) {
      _tmpSunDir.normalize();
      L.position.copy(_tmpSunDir).multiplyScalar(100);
      L.target.position.set(0, 0, 0);
      L.target.updateMatrixWorld();
    }
  }

  // --- Hemisphere ---
  if (_lightEntities.hemi >= 0 && _hemiLightInstance.ref) {
    const eid = _lightEntities.hemi;
    const L = _hemiLightInstance.ref;
    _tmpCol.setRGB(LightRef.colorR[eid], LightRef.colorG[eid], LightRef.colorB[eid]);
    L.color.copy(_tmpCol);
    L.intensity = LightRef.intensity[eid];
  }

  // --- Ambient ---
  if (_lightEntities.ambient >= 0 && _ambLightInstance.ref) {
    const eid = _lightEntities.ambient;
    const L = _ambLightInstance.ref;
    _tmpCol.setRGB(LightRef.colorR[eid], LightRef.colorG[eid], LightRef.colorB[eid]);
    L.color.copy(_tmpCol);
    L.intensity = LightRef.intensity[eid];
  }
}

/* ------------------------------------------------------------------ */
/* 7. ACCESSORS — every downstream module pulls handles from here     */
/* ------------------------------------------------------------------ */

export function getWorld()          { return world; }
export function getCore()           { return _coreInstance; }
export function getRenderer()       { return _coreInstance ? _coreInstance.renderer : null; }
export function getScene()          { return _coreInstance ? _coreInstance.scene    : null; }
export function getCamera()         { return _coreInstance ? _coreInstance.camera   : null; }
export function getLightManager()   { return _coreInstance ? _coreInstance.light    : null; }
export function getPaletteRGB()     { return _coreInstance ? _coreInstance.palette.rgb : null; }
export function getBiomeWeights()   { return _coreInstance ? _coreInstance.biome.weights : null; }
export function getLightEntities()  { return _lightEntities; }
export function getPerfTier()       { return PERF_TIER; }
export function getMobileDprCap()   { return MOBILE_DPR_CAP; }
export function isWorldReady()      { return _initialized; }

/* ------------------------------------------------------------------ */
/* 8. FRAME STEP — called by main loop; forwards to core + light sync */
/* ------------------------------------------------------------------ */

export function stepWorld(dt, elapsed) {
  if (!_coreInstance) return;
  _coreInstance.update(dt, elapsed);
  syncLightEntitiesToThree(elapsed);
}

export function renderWorld() {
  if (!_coreInstance) return;
  _coreInstance.render();
}

/* ------------------------------------------------------------------ */
/* 9. DISPOSE                                                         */
/* ------------------------------------------------------------------ */

export function disposeWorld() {
  if (_coreInstance) {
    _coreInstance.dispose();
    _coreInstance = null;
  }
  _lightEntities.sun = -1;
  _lightEntities.moon = -1;
  _lightEntities.hemi = -1;
  _lightEntities.ambient = -1;
  _initialized = false;
}

/* ------------------------------------------------------------------ */
/* 10. RE-EXPORT CORE ENUMS so downstream modules import from one place */
/* ------------------------------------------------------------------ */

export { Biome, PaletteSlot, Palettes, CoreMath, ProceduralWorldCore };

export default {
  initializeWorld,
  disposeWorld,
  stepWorld,
  renderWorld,
  getWorld,
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getLightManager,
  getPaletteRGB,
  getBiomeWeights,
  getLightEntities,
  getPerfTier,
  getMobileDprCap,
  isWorldReady,
  bindThreeLights,
  syncLightEntitiesToThree,
  LIGHT_TYPE,
  Transform,
  LightRef,
  ShadowRef,
  GIRef,
  AORef,
  ActiveTag,
  Biome,
  PaletteSlot,
  Palettes,
  CoreMath,
  ProceduralWorldCore,
  MAX_ENTITIES,
};