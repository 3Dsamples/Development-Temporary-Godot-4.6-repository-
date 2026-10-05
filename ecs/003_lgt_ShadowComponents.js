// File : 003
// name : src/ecs/003_lgt_ShadowComponents.js
// description : bitECS 0.4.0 SoA component definitions for the shadow subsystem
//               of the anime lighting stack on Android mobile. Every piece of
//               state the shadow pipeline needs — cascade splits, shadow atlas
//               tiles, bias / normal bias, filter selection, softness ramps,
//               contact-shadow volumes, VSM/PCSS/ESM controller parameters,
//               shadow casters/receivers registries, culling state — is
//               declared here as a plain-object component whose fields are
//               pre-sized TypedArrays. This is the bitECS 0.4.0 architecture:
//               NO `defineComponent`, NO `Types` export, NO separate store
//               registry. Components are plain JS objects passed directly to
//               `createWorld({ components })` and attached via
//               `addComponent(world, eid, ComponentObject)`.
//
//               Components declared:
//                 • ShadowCasterRef     — caster entity → shadow light binding
//                 • ShadowReceiverRef   — receiver entity → shadow light binding
//                 • ShadowAtlas         — atlas tile allocation (x,y,w,h, gen)
//                 • ShadowCascade       — cascade split depths + matrices
//                 • ShadowBias          — per-light bias / normal bias
//                 • ShadowFilter        — filter mode (basic / pcf / pcss / vsm / esm)
//                 • ShadowSoftness      — penumbra / soft edge
//                 • ShadowFrustum       — shadow camera frustum state
//                 • ShadowCache         — per-frame cache validity
//                 • ShadowBudget        — per-frame cost + LOD
//                 • ShadowContact       — contact shadow (SSAO-backed) params
//                 • ShadowVolume        — shadow volume (stencil) params
//                 • ShadowDirLight      — directional light shadow state
//                 • ShadowPointLight    — point light shadow state (cube map)
//                 • ShadowSpotLight     — spot light shadow state
//                 • ShadowAtlasMap      — atlas texture residency
//                 • ShadowTint          — anime shadow tint color
//                 • ShadowEdge          — anime posterized edge parameters
//                 • ShadowTile          — per-tile allocation record
//                 • ShadowUpdatePolicy  — update frequency per light
//
//               Also exports:
//                 • MAX_ENTITIES (100000) — same as light components
//                 • MAX_SHADOW_CASCADES = 4
//                 • MAX_SHADOW_TILES = 64
//                 • SHADOW_FILTER / SHADOW_STATE / SHADOW_MODE / SHADOW_EDGE /
//                   SHADOW_TINT enums
//                 • SHADOW_COMPONENTS bundle for createWorld()
//                 • spawnShadowCaster / spawnShadowReceiver /
//                   spawnShadowAtlas / spawnShadowCascadeSet /
//                   spawnContactShadowVolume factories
//                 • Utility helpers: allocateShadowTile,
//                   releaseShadowTile, markShadowDirty,
//                   clearShadowDirty, setShadowFilter,
//                   setShadowBias, setShadowSoftness
//
//               Strictly Three.js r185 lights only (DirectionalLightShadow,
//               PointLightShadow, SpotLightShadow); strictly bitECS 0.4.0 API
//               only; every typed array sized once to MAX_ENTITIES = 100000.
// best for : Single source of truth for every shadow-related ECS component
//            in the anime lighting stack. Every downstream shadow system
//            (074_lgt_ShadowManager.js through 118_lgt_ShadowLightTagger.js)
//            imports its component definitions from here so entity IDs,
//            atlas tile layouts, and filter enums stay consistent across
//            CPU side prep, GPU atlas packing, and worker-side shadow
//            updates.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

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
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  MAX_ENTITIES,
  LIGHT_TYPE,
  LIGHT_TYPE_NAME,
} from './002_lgt_LightComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of cascades per directional light shadow. Matches
 * THREE.js r185's CSM conventions.
 */
export const MAX_SHADOW_CASCADES = 4;

/**
 * Maximum number of shadow atlas tiles. Each tile can be assigned to one
 * shadow-casting light (or one cascade of one light).
 */
export const MAX_SHADOW_TILES =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 32 :
                                 16;

/**
 * Maximum number of cascades across all lights simultaneously.
 */
export const MAX_TOTAL_CASCADES = MAX_SHADOW_TILES;

/**
 * Shadow map resolutions supported by the atlas packer.
 */
export const SHADOW_MAP_SIZE = Object.freeze({
  SIZE_256:  256,
  SIZE_512:  512,
  SIZE_1024: 1024,
  SIZE_2048: 2048,
  SIZE_4096: 4096,
});

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * Shadow filter modes. Maps 1:1 to Three.js r185 shadow types and the
 * anime-specific edge posterization.
 */
export const SHADOW_FILTER = Object.freeze({
  BASIC:    0,   // PCFShadowMap with 1 tap
  PCF:      1,   // PCFShadowMap with 3×3 taps
  PCF_SOFT: 2,   // PCFSoftShadowMap
  PCSS:     3,   // Percentage-Closer Soft Shadows (blocker search)
  VSM:      4,   // Variance Shadow Map
  ESM:      5,   // Exponential Shadow Map
  CONTACT:  6,   // Screen-space contact shadows
  ANIME:    7,   // Anime posterized hard-edge
  COUNT:    8,
});

export const SHADOW_FILTER_NAME = Object.freeze([
  'basic',
  'pcf',
  'pcf_soft',
  'pcss',
  'vsm',
  'esm',
  'contact',
  'anime',
]);

/**
 * Shadow pipeline state — per-light shadow update state machine.
 */
export const SHADOW_STATE = Object.freeze({
  IDLE:       0,   // no shadow
  DIRTY:      1,   // needs full re-render
  UPDATING:   2,   // rendering this frame
  CACHED:     3,   // reusing last frame's map
  AWAITING:   4,   // async update in flight
  FAILED:     5,   // update failed — skip this frame
  COUNT:      6,
});

export const SHADOW_STATE_NAME = Object.freeze([
  'idle',
  'dirty',
  'updating',
  'cached',
  'awaiting',
  'failed',
]);

/**
 * Shadow update modes (frequency policy).
 */
export const SHADOW_UPDATE_MODE = Object.freeze({
  EVERY_FRAME:   0,   // update every frame
  EVERY_OTHER:   1,   // update every 2 frames
  ON_DIRTY:      2,   // update only when marked dirty
  ON_CAMERA:     3,   // update when camera moves significantly
  STATIC:        4,   // never update after first bake
  COUNT:         5,
});

export const SHADOW_UPDATE_MODE_NAME = Object.freeze([
  'every_frame',
  'every_other',
  'on_dirty',
  'on_camera',
  'static',
]);

/**
 * Shadow atlas modes.
 */
export const SHADOW_ATLAS_MODE = Object.freeze({
  PER_LIGHT:    0,   // one texture per shadow-casting light (Three.js default)
  ATLAS_PACKED: 1,   // single atlas texture, tile-per-light
  CASCADED:     2,   // CSM with per-cascade texture
  HYBRID:       3,   // CSM + point lights in one atlas
  COUNT:        4,
});

export const SHADOW_ATLAS_MODE_NAME = Object.freeze([
  'per_light',
  'atlas_packed',
  'cascaded',
  'hybrid',
]);

/**
 * Anime shadow edge style — matching the reference image set.
 */
export const SHADOW_EDGE = Object.freeze({
  SOFT:      0,   // soft PCF edges
  CRISP:     1,   // hard cel edge
  POSTERIZED:2,   // 2-3 band posterized edge (anime)
  RIM:       3,   // rim-lit shadow with warm/cool tint
  GLOW:      4,   // glowing shadow (magic, image 6 + 8)
  DITHERED:  5,   // ordered dither edge (retro anime)
  COUNT:     6,
});

export const SHADOW_EDGE_NAME = Object.freeze([
  'soft',
  'crisp',
  'posterized',
  'rim',
  'glow',
  'dithered',
]);

/**
 * Shadow update source — how the shadow map is being produced.
 */
export const SHADOW_SOURCE = Object.freeze({
  REAL_TIME:    0,   // live render each frame
  BAKED_ONCE:   1,   // single bake at spawn
  STREAMED:     2,   // updated by worker
  IMPOSTOR:     3,   // 2D shadow impostor
  ANALYTIC:     4,   // SDF-based analytic shadow
  COUNT:        5,
});

export const SHADOW_SOURCE_NAME = Object.freeze([
  'real_time',
  'baked_once',
  'streamed',
  'impostor',
  'analytic',
]);

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * ShadowCasterRef — links a caster entity to a shadow-casting light.
 * One caster can be bound to multiple lights (via multiple entities), and
 * one light can have many casters.
 */
export const ShadowCasterRef = {
  lightEid:      new Int32Array(MAX_ENTITIES),   // -1 if unbound
  casterKind:    new Uint8Array(MAX_ENTITIES),   // 0=mesh 1=proxy 2=impostor 3=sdf
  castStrength:  new Float32Array(MAX_ENTITIES), // [0..1] contribution
  casterRadius:  new Float32Array(MAX_ENTITIES), // bounding sphere radius
  casterCenterX: new Float32Array(MAX_ENTITIES),
  casterCenterY: new Float32Array(MAX_ENTITIES),
  casterCenterZ: new Float32Array(MAX_ENTITIES),
  enabled:       new Uint8Array(MAX_ENTITIES),
  frameActive:   new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowReceiverRef — links a receiver entity to a shadow-casting light.
 * Used for the receiver-side filter policy (e.g. higher-resolution PCF on
 * hero meshes, cheaper on background).
 */
export const ShadowReceiverRef = {
  lightEid:       new Int32Array(MAX_ENTITIES),
  receiverKind:   new Uint8Array(MAX_ENTITIES),   // 0=mesh 1=terrain 2=impostor
  sampleQuality:  new Uint8Array(MAX_ENTITIES),   // 0=low 1=med 2=high
  selfShadowBias: new Float32Array(MAX_ENTITIES),
  enabled:        new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowAtlas — one record per shadow-casting light, describing its
 * allocation in the atlas texture.
 */
export const ShadowAtlas = {
  mode:        new Uint8Array(MAX_ENTITIES),   // SHADOW_ATLAS_MODE
  tileIndex:   new Int16Array(MAX_ENTITIES),   // -1 if unallocated
  tileX:       new Uint16Array(MAX_ENTITIES),
  tileY:       new Uint16Array(MAX_ENTITIES),
  tileW:       new Uint16Array(MAX_ENTITIES),
  tileH:       new Uint16Array(MAX_ENTITIES),
  atlasWidth:  new Uint16Array(MAX_ENTITIES),
  atlasHeight: new Uint16Array(MAX_ENTITIES),
  valid:       new Uint8Array(MAX_ENTITIES),
  generation:  new Uint32Array(MAX_ENTITIES),
};

/**
 * ShadowCascade — per-cascade data for CSM (directional light shadow).
 * Cascades are stored per-light via an index base (lightEid * MAX_SHADOW_CASCADES).
 */
export const ShadowCascade = {
  splitNear:    new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  splitFar:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  bias:         new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  normalBias:   new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  softness:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  mapSize:      new Uint16Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  // Matrix stored as 16 floats per cascade (world → light clip).
  matrix0:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix1:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix2:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix3:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix4:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix5:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix6:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix7:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix8:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix9:      new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix10:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix11:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix12:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix13:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix14:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  matrix15:     new Float32Array(MAX_ENTITIES * MAX_SHADOW_CASCADES),
  // Cascade counts + enabled flags.
  cascadeCount: new Uint8Array(MAX_ENTITIES),
  enabled:      new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowBias — per-light bias parameters (copied from LightShadow on
 * shadow system update, kept here as the runtime working set).
 */
export const ShadowBias = {
  bias:          new Float32Array(MAX_ENTITIES),
  normalBias:    new Float32Array(MAX_ENTITIES),
  depthBias:     new Float32Array(MAX_ENTITIES),
  slopeBias:     new Float32Array(MAX_ENTITIES),
  panCakeFix:    new Float32Array(MAX_ENTITIES),
  adaptiveBias:  new Uint8Array(MAX_ENTITIES),
  lastBiasFrame: new Uint32Array(MAX_ENTITIES),
};

/**
 * ShadowFilter — per-light filter selection + kernel size.
 */
export const ShadowFilter = {
  mode:         new Uint8Array(MAX_ENTITIES),    // SHADOW_FILTER
  kernelSize:   new Uint8Array(MAX_ENTITIES),    // 1, 3, 5, 7
  blockerSearch:new Uint8Array(MAX_ENTITIES),    // PCSS blocker search on/off
  penumbraSize: new Float32Array(MAX_ENTITIES),  // PCSS penumbra in shadow UV
  samples:      new Uint8Array(MAX_ENTITIES),    // sample count for VSM/PCSS
  lightBleed:   new Float32Array(MAX_ENTITIES),  // VSM light-bleed reduction
};

/**
 * ShadowSoftness — anime-soft edge control.
 */
export const ShadowSoftness = {
  softness:        new Float32Array(MAX_ENTITIES),
  penumbra:        new Float32Array(MAX_ENTITIES),
  edgeRounding:    new Float32Array(MAX_ENTITIES),
  bandCount:       new Uint8Array(MAX_ENTITIES),
  edgeStyle:       new Uint8Array(MAX_ENTITIES),  // SHADOW_EDGE
  gradientStrength:new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowFrustum — per-light orthographic/perspective shadow frustum state.
 */
export const ShadowFrustum = {
  near:       new Float32Array(MAX_ENTITIES),
  far:        new Float32Array(MAX_ENTITIES),
  left:       new Float32Array(MAX_ENTITIES),
  right:      new Float32Array(MAX_ENTITIES),
  top:        new Float32Array(MAX_ENTITIES),
  bottom:     new Float32Array(MAX_ENTITIES),
  fov:        new Float32Array(MAX_ENTITIES),
  aspect:     new Float32Array(MAX_ENTITIES),
  isOrtho:    new Uint8Array(MAX_ENTITIES),
  valid:      new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowCache — per-frame cache validity + last-update timestamps.
 */
export const ShadowCache = {
  lastRenderFrame:  new Uint32Array(MAX_ENTITIES),
  lastCommitFrame:  new Uint32Array(MAX_ENTITIES),
  dirty:            new Uint8Array(MAX_ENTITIES),
  valid:            new Uint8Array(MAX_ENTITIES),
  updateMode:       new Uint8Array(MAX_ENTITIES),   // SHADOW_UPDATE_MODE
  updateInterval:   new Uint16Array(MAX_ENTITIES),  // frames between updates
  source:           new Uint8Array(MAX_ENTITIES),   // SHADOW_SOURCE
};

/**
 * ShadowBudget — per-light shadow cost + LOD.
 */
export const ShadowBudget = {
  cost:          new Float32Array(MAX_ENTITIES),
  costEma:       new Float32Array(MAX_ENTITIES),
  lastCostMs:    new Float32Array(MAX_ENTITIES),
  lod:           new Uint8Array(MAX_ENTITIES),     // 0=full 1=half 2=quarter 3=off
  lodTarget:     new Uint8Array(MAX_ENTITIES),
  lastLodFrame:  new Uint32Array(MAX_ENTITIES),
  allocated:     new Uint8Array(MAX_ENTITIES),
  allocatedTile: new Int16Array(MAX_ENTITIES),     // -1 if none
};

/**
 * ShadowContact — contact-shadow (SSAO-backed) parameters.
 */
export const ShadowContact = {
  enabled:      new Uint8Array(MAX_ENTITIES),
  radius:       new Float32Array(MAX_ENTITIES),
  thickness:    new Float32Array(MAX_ENTITIES),
  strength:     new Float32Array(MAX_ENTITIES),
  bias:         new Float32Array(MAX_ENTITIES),
  rayCount:     new Uint8Array(MAX_ENTITIES),
  jitterStrength: new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowVolume — stencil shadow volume (opto-in; used for stylized shadow
 * on very bright anime scenes).
 */
export const ShadowVolume = {
  enabled:    new Uint8Array(MAX_ENTITIES),
  extrude:    new Float32Array(MAX_ENTITIES),
  zFail:      new Uint8Array(MAX_ENTITIES),   // 0=z-pass 1=z-fail
  alpha:      new Float32Array(MAX_ENTITIES),
  tintR:      new Float32Array(MAX_ENTITIES),
  tintG:      new Float32Array(MAX_ENTITIES),
  tintB:      new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowDirLight — directional light shadow specifics.
 */
export const ShadowDirLight = {
  cascades:       new Uint8Array(MAX_ENTITIES),
  cascadeBlend:   new Uint8Array(MAX_ENTITIES),
  cascadeSplitLambda: new Float32Array(MAX_ENTITIES),
  stabilize:      new Uint8Array(MAX_ENTITIES),
  texelSnap:      new Uint8Array(MAX_ENTITIES),
  maxDistance:    new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowPointLight — point light shadow specifics (cube map).
 */
export const ShadowPointLight = {
  cubeMapSize:    new Uint16Array(MAX_ENTITIES),
  farPlane:       new Float32Array(MAX_ENTITIES),
  nearPlane:      new Float32Array(MAX_ENTITIES),
  faceResolution: new Uint16Array(MAX_ENTITIES),
  atlasFaceX:     new Int16Array(MAX_ENTITIES),
  atlasFaceY:     new Int16Array(MAX_ENTITIES),
};

/**
 * ShadowSpotLight — spot light shadow specifics.
 */
export const ShadowSpotLight = {
  fov:          new Float32Array(MAX_ENTITIES),
  aspect:       new Float32Array(MAX_ENTITIES),
  mapSize:      new Uint16Array(MAX_ENTITIES),
  focus:        new Float32Array(MAX_ENTITIES),
  penumbra:     new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowAtlasMap — shadow atlas texture residency record.
 */
export const ShadowAtlasMap = {
  textureWidth:    new Uint16Array(MAX_ENTITIES),
  textureHeight:   new Uint16Array(MAX_ENTITIES),
  tileCount:       new Uint16Array(MAX_ENTITIES),
  usedTiles:       new Uint16Array(MAX_ENTITIES),
  generation:      new Uint32Array(MAX_ENTITIES),
  boundLightEid:   new Int32Array(MAX_ENTITIES),
  valid:           new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowTint — per-light anime shadow tint (matches reference images).
 * Shadow tint converts the pure-black shadow into a colored anime shadow
 * (e.g. warm purple in sunset scene, cold blue in snow scene).
 */
export const ShadowTint = {
  tintR:          new Float32Array(MAX_ENTITIES),
  tintG:          new Float32Array(MAX_ENTITIES),
  tintB:          new Float32Array(MAX_ENTITIES),
  tintStrength:   new Float32Array(MAX_ENTITIES),
  rimStrength:    new Float32Array(MAX_ENTITIES),
  rimR:           new Float32Array(MAX_ENTITIES),
  rimG:           new Float32Array(MAX_ENTITIES),
  rimB:           new Float32Array(MAX_ENTITIES),
  enabled:        new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowEdge — anime posterized edge parameters.
 */
export const ShadowEdge = {
  posterizeBands:  new Uint8Array(MAX_ENTITIES),
  posterizeSharpness: new Float32Array(MAX_ENTITIES),
  ditherStrength:  new Float32Array(MAX_ENTITIES),
  outlineWidth:    new Float32Array(MAX_ENTITIES),
  outlineStrength: new Float32Array(MAX_ENTITIES),
  edgeJitter:      new Float32Array(MAX_ENTITIES),
  edgeJitterHz:    new Float32Array(MAX_ENTITIES),
};

/**
 * ShadowTile — per-tile allocation record for the atlas packer.
 * One entry per potential tile slot.
 */
export const ShadowTile = {
  ownerEid:      new Int32Array(MAX_TOTAL_CASCADES),
  x:             new Uint16Array(MAX_TOTAL_CASCADES),
  y:             new Uint16Array(MAX_TOTAL_CASCADES),
  w:             new Uint16Array(MAX_TOTAL_CASCADES),
  h:             new Uint16Array(MAX_TOTAL_CASCADES),
  cascadeIndex:  new Int8Array(MAX_TOTAL_CASCADES),  // -1 if not a cascade
  lightType:     new Uint8Array(MAX_TOTAL_CASCADES), // LIGHT_TYPE
  inUse:         new Uint8Array(MAX_TOTAL_CASCADES),
  generation:    new Uint32Array(MAX_TOTAL_CASCADES),
  lastUpdateFrame: new Uint32Array(MAX_TOTAL_CASCADES),
};

/**
 * ShadowUpdatePolicy — per-light update frequency policy.
 */
export const ShadowUpdatePolicy = {
  mode:            new Uint8Array(MAX_ENTITIES),    // SHADOW_UPDATE_MODE
  interval:        new Uint16Array(MAX_ENTITIES),
  lastUpdateFrame: new Uint32Array(MAX_ENTITIES),
  frameCounter:    new Uint16Array(MAX_ENTITIES),
  cameraDeltaThreshold: new Float32Array(MAX_ENTITIES),
  lastCameraX:     new Float32Array(MAX_ENTITIES),
  lastCameraY:     new Float32Array(MAX_ENTITIES),
  lastCameraZ:     new Float32Array(MAX_ENTITIES),
  forceUpdate:     new Uint8Array(MAX_ENTITIES),
};

/**
 * ShadowState — per-light shadow state machine.
 */
export const ShadowState = {
  state:          new Uint8Array(MAX_ENTITIES),     // SHADOW_STATE
  prevState:      new Uint8Array(MAX_ENTITIES),
  lastStateFrame: new Uint32Array(MAX_ENTITIES),
  failureCount:   new Uint8Array(MAX_ENTITIES),
  lastError:      new Int32Array(MAX_ENTITIES),     // error code, -1 = none
};

/* ------------------------------------------------------------------ */
/* 3. ECS WORLD COMPONENT BUNDLE                                      */
/* ------------------------------------------------------------------ */

export const SHADOW_COMPONENTS = Object.freeze({
  ShadowCasterRef,
  ShadowReceiverRef,
  ShadowAtlas,
  ShadowCascade,
  ShadowBias,
  ShadowFilter,
  ShadowSoftness,
  ShadowFrustum,
  ShadowCache,
  ShadowBudget,
  ShadowContact,
  ShadowVolume,
  ShadowDirLight,
  ShadowPointLight,
  ShadowSpotLight,
  ShadowAtlasMap,
  ShadowTint,
  ShadowEdge,
  ShadowTile,
  ShadowUpdatePolicy,
  ShadowState,
});

/* ------------------------------------------------------------------ */
/* 4. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

let _shadowWorld = null;

export function getShadowWorld() {
  if (_shadowWorld) return _shadowWorld;
  try {
    _shadowWorld = createWorld({
      components: SHADOW_COMPONENTS,
      time: {
        delta: 0,
        elapsed: 0,
        then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      },
    });
  } catch (e) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.SHADOWS, `[003_lgt_ShadowComponents] failed to create shadow world: ${e && e.message}`);
    _shadowWorld = null;
  }
  return _shadowWorld;
}

/* ------------------------------------------------------------------ */
/* 5. SPAWN HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Attach the base shadow components (bias/filter/softness/frustum/cache/
 * budget/state/policy/tint/edge) to the given entity.
 */
function _attachShadowBase(world, eid, lightEid, lightType) {
  // Bias
  ShadowBias.bias[eid] = -0.0008;
  ShadowBias.normalBias[eid] = 0.020;
  ShadowBias.depthBias[eid] = 0.0;
  ShadowBias.slopeBias[eid] = 1.0;
  ShadowBias.panCakeFix[eid] = 0.0;
  ShadowBias.adaptiveBias[eid] = 0;
  ShadowBias.lastBiasFrame[eid] = 0;

  // Filter
  ShadowFilter.mode[eid] = SHADOW_FILTER.PCF;
  ShadowFilter.kernelSize[eid] = 3;
  ShadowFilter.blockerSearch[eid] = 0;
  ShadowFilter.penumbraSize[eid] = 0.05;
  ShadowFilter.samples[eid] = 9;
  ShadowFilter.lightBleed[eid] = 0.30;

  // Softness
  ShadowSoftness.softness[eid] = 0.05;
  ShadowSoftness.penumbra[eid] = 0.10;
  ShadowSoftness.edgeRounding[eid] = 0.0;
  ShadowSoftness.bandCount[eid] = 4;
  ShadowSoftness.edgeStyle[eid] = SHADOW_EDGE.POSTERIZED;
  ShadowSoftness.gradientStrength[eid] = 1.0;

  // Frustum
  ShadowFrustum.near[eid] = 0.5;
  ShadowFrustum.far[eid] = 250.0;
  ShadowFrustum.left[eid] = -80.0;
  ShadowFrustum.right[eid] = 80.0;
  ShadowFrustum.top[eid] = 80.0;
  ShadowFrustum.bottom[eid] = -80.0;
  ShadowFrustum.fov[eid] = 90.0;
  ShadowFrustum.aspect[eid] = 1.0;
  ShadowFrustum.isOrtho[eid] = 1;
  ShadowFrustum.valid[eid] = 0;

  // Cache
  ShadowCache.lastRenderFrame[eid] = 0;
  ShadowCache.lastCommitFrame[eid] = 0;
  ShadowCache.dirty[eid] = 1;
  ShadowCache.valid[eid] = 0;
  ShadowCache.updateMode[eid] = SHADOW_UPDATE_MODE.EVERY_OTHER;
  ShadowCache.updateInterval[eid] = 2;
  ShadowCache.source[eid] = SHADOW_SOURCE.REAL_TIME;

  // Budget
  ShadowBudget.cost[eid] = 1.0;
  ShadowBudget.costEma[eid] = 1.0;
  ShadowBudget.lastCostMs[eid] = 0;
  ShadowBudget.lod[eid] = 0;
  ShadowBudget.lodTarget[eid] = 0;
  ShadowBudget.lastLodFrame[eid] = 0;
  ShadowBudget.allocated[eid] = 0;
  ShadowBudget.allocatedTile[eid] = -1;

  // Contact
  ShadowContact.enabled[eid] = 0;
  ShadowContact.radius[eid] = 0.5;
  ShadowContact.thickness[eid] = 0.1;
  ShadowContact.strength[eid] = 0.5;
  ShadowContact.bias[eid] = 0.02;
  ShadowContact.rayCount[eid] = 8;
  ShadowContact.jitterStrength[eid] = 1.0;

  // Volume
  ShadowVolume.enabled[eid] = 0;
  ShadowVolume.extrude[eid] = 100.0;
  ShadowVolume.zFail[eid] = 1;
  ShadowVolume.alpha[eid] = 0.5;
  ShadowVolume.tintR[eid] = 0.1;
  ShadowVolume.tintG[eid] = 0.1;
  ShadowVolume.tintB[eid] = 0.15;

  // Atlas
  ShadowAtlas.mode[eid] = SHADOW_ATLAS_MODE.ATLAS_PACKED;
  ShadowAtlas.tileIndex[eid] = -1;
  ShadowAtlas.tileX[eid] = 0;
  ShadowAtlas.tileY[eid] = 0;
  ShadowAtlas.tileW[eid] = 0;
  ShadowAtlas.tileH[eid] = 0;
  ShadowAtlas.atlasWidth[eid] = 2048;
  ShadowAtlas.atlasHeight[eid] = 2048;
  ShadowAtlas.valid[eid] = 0;
  ShadowAtlas.generation[eid] = 0;

  // Cascade (per light)
  ShadowCascade.cascadeCount[eid] = 1;
  ShadowCascade.enabled[eid] = 0;
  for (let c = 0; c < MAX_SHADOW_CASCADES; c++) {
    const off = eid * MAX_SHADOW_CASCADES + c;
    ShadowCascade.splitNear[off] = 0;
    ShadowCascade.splitFar[off] = 0;
    ShadowCascade.bias[off] = -0.0008;
    ShadowCascade.normalBias[off] = 0.020;
    ShadowCascade.softness[off] = 0.05;
    ShadowCascade.mapSize[off] = 1024;
    ShadowCascade.matrix0[off] = 1;
    ShadowCascade.matrix1[off] = 0;
    ShadowCascade.matrix2[off] = 0;
    ShadowCascade.matrix3[off] = 0;
    ShadowCascade.matrix4[off] = 0;
    ShadowCascade.matrix5[off] = 1;
    ShadowCascade.matrix6[off] = 0;
    ShadowCascade.matrix7[off] = 0;
    ShadowCascade.matrix8[off] = 0;
    ShadowCascade.matrix9[off] = 0;
    ShadowCascade.matrix10[off] = 1;
    ShadowCascade.matrix11[off] = 0;
    ShadowCascade.matrix12[off] = 0;
    ShadowCascade.matrix13[off] = 0;
    ShadowCascade.matrix14[off] = 0;
    ShadowCascade.matrix15[off] = 1;
  }

  // Dir light
  ShadowDirLight.cascades[eid] = 1;
  ShadowDirLight.cascadeBlend[eid] = 1;
  ShadowDirLight.cascadeSplitLambda[eid] = 0.5;
  ShadowDirLight.stabilize[eid] = 1;
  ShadowDirLight.texelSnap[eid] = 1;
  ShadowDirLight.maxDistance[eid] = 140.0;

  // Point light
  ShadowPointLight.cubeMapSize[eid] = 512;
  ShadowPointLight.farPlane[eid] = 100.0;
  ShadowPointLight.nearPlane[eid] = 0.5;
  ShadowPointLight.faceResolution[eid] = 512;
  ShadowPointLight.atlasFaceX[eid] = -1;
  ShadowPointLight.atlasFaceY[eid] = -1;

  // Spot light
  ShadowSpotLight.fov[eid] = Math.PI / 3;
  ShadowSpotLight.aspect[eid] = 1.0;
  ShadowSpotLight.mapSize[eid] = 512;
  ShadowSpotLight.focus[eid] = 1.0;
  ShadowSpotLight.penumbra[eid] = 0.1;

  // Atlas map
  ShadowAtlasMap.textureWidth[eid] = 2048;
  ShadowAtlasMap.textureHeight[eid] = 2048;
  ShadowAtlasMap.tileCount[eid] = 0;
  ShadowAtlasMap.usedTiles[eid] = 0;
  ShadowAtlasMap.generation[eid] = 0;
  ShadowAtlasMap.boundLightEid[eid] = lightEid !== undefined ? lightEid : -1;
  ShadowAtlasMap.valid[eid] = 0;

  // Tint (anime style — cool neutral default)
  ShadowTint.tintR[eid] = 0.12;
  ShadowTint.tintG[eid] = 0.18;
  ShadowTint.tintB[eid] = 0.30;
  ShadowTint.tintStrength[eid] = 0.65;
  ShadowTint.rimStrength[eid] = 0.20;
  ShadowTint.rimR[eid] = 0.90;
  ShadowTint.rimG[eid] = 0.95;
  ShadowTint.rimB[eid] = 1.00;
  ShadowTint.enabled[eid] = 1;

  // Edge (anime posterize)
  ShadowEdge.posterizeBands[eid] = 4;
  ShadowEdge.posterizeSharpness[eid] = 1.0;
  ShadowEdge.ditherStrength[eid] = 1.0 / 255.0;
  ShadowEdge.outlineWidth[eid] = 0.005;
  ShadowEdge.outlineStrength[eid] = 0.30;
  ShadowEdge.edgeJitter[eid] = 0.0;
  ShadowEdge.edgeJitterHz[eid] = 0.5;

  // Update policy
  ShadowUpdatePolicy.mode[eid] = SHADOW_UPDATE_MODE.EVERY_OTHER;
  ShadowUpdatePolicy.interval[eid] = 2;
  ShadowUpdatePolicy.lastUpdateFrame[eid] = 0;
  ShadowUpdatePolicy.frameCounter[eid] = 0;
  ShadowUpdatePolicy.cameraDeltaThreshold[eid] = 0.05;
  ShadowUpdatePolicy.lastCameraX[eid] = 0;
  ShadowUpdatePolicy.lastCameraY[eid] = 0;
  ShadowUpdatePolicy.lastCameraZ[eid] = 0;
  ShadowUpdatePolicy.forceUpdate[eid] = 0;

  // State
  ShadowState.state[eid] = SHADOW_STATE.DIRTY;
  ShadowState.prevState[eid] = SHADOW_STATE.IDLE;
  ShadowState.lastStateFrame[eid] = 0;
  ShadowState.failureCount[eid] = 0;
  ShadowState.lastError[eid] = -1;

  // Attach all components.
  addComponent(world, eid, ShadowBias);
  addComponent(world, eid, ShadowFilter);
  addComponent(world, eid, ShadowSoftness);
  addComponent(world, eid, ShadowFrustum);
  addComponent(world, eid, ShadowCache);
  addComponent(world, eid, ShadowBudget);
  addComponent(world, eid, ShadowContact);
  addComponent(world, eid, ShadowVolume);
  addComponent(world, eid, ShadowAtlas);
  addComponent(world, eid, ShadowCascade);
  addComponent(world, eid, ShadowDirLight);
  addComponent(world, eid, ShadowPointLight);
  addComponent(world, eid, ShadowSpotLight);
  addComponent(world, eid, ShadowAtlasMap);
  addComponent(world, eid, ShadowTint);
  addComponent(world, eid, ShadowEdge);
  addComponent(world, eid, ShadowUpdatePolicy);
  addComponent(world, eid, ShadowState);
}

/* ------------------------------------------------------------------ */
/* 6. PUBLIC SPAWN FUNCTIONS                                          */
/* ------------------------------------------------------------------ */

/**
 * Spawns a shadow-caster reference linking a caster entity to a
 * shadow-casting light.
 */
export function spawnShadowCaster(world, spec = {}) {
  const w = world || getShadowWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  ShadowCasterRef.lightEid[eid]      = spec.lightEid !== undefined ? spec.lightEid : -1;
  ShadowCasterRef.casterKind[eid]    = spec.casterKind !== undefined ? spec.casterKind : 0;
  ShadowCasterRef.castStrength[eid]  = spec.castStrength !== undefined ? spec.castStrength : 1.0;
  ShadowCasterRef.casterRadius[eid]  = spec.casterRadius !== undefined ? spec.casterRadius : 1.0;
  ShadowCasterRef.casterCenterX[eid] = spec.center ? spec.center[0] : 0;
  ShadowCasterRef.casterCenterY[eid] = spec.center ? spec.center[1] : 0;
  ShadowCasterRef.casterCenterZ[eid] = spec.center ? spec.center[2] : 0;
  ShadowCasterRef.enabled[eid]       = spec.enabled !== false ? 1 : 0;
  ShadowCasterRef.frameActive[eid]   = 1;

  addComponent(w, eid, ShadowCasterRef);
  return eid;
}

/**
 * Spawns a shadow-receiver reference.
 */
export function spawnShadowReceiver(world, spec = {}) {
  const w = world || getShadowWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  ShadowReceiverRef.lightEid[eid]       = spec.lightEid !== undefined ? spec.lightEid : -1;
  ShadowReceiverRef.receiverKind[eid]   = spec.receiverKind !== undefined ? spec.receiverKind : 0;
  ShadowReceiverRef.sampleQuality[eid]  = spec.sampleQuality !== undefined ? spec.sampleQuality : 2;
  ShadowReceiverRef.selfShadowBias[eid] = spec.selfShadowBias !== undefined ? spec.selfShadowBias : 0.02;
  ShadowReceiverRef.enabled[eid]        = spec.enabled !== false ? 1 : 0;

  addComponent(w, eid, ShadowReceiverRef);
  return eid;
}

/**
 * Spawns a shadow-casting light binding on the given light entity. Attaches
 * the base shadow component set to the light entity itself.
 */
export function spawnShadowAtlas(world, lightEid, spec = {}) {
  const w = world || getShadowWorld();
  if (!w) return -1;
  if (typeof lightEid !== 'number' || lightEid < 0) return -1;

  _attachShadowBase(w, lightEid, lightEid, spec.lightType !== undefined ? spec.lightType : LIGHT_TYPE.DIRECTIONAL);

  if (spec.filterMode !== undefined) ShadowFilter.mode[lightEid] = spec.filterMode;
  if (spec.mapSize !== undefined) {
    ShadowCascade.mapSize[lightEid * MAX_SHADOW_CASCADES + 0] = spec.mapSize;
    ShadowAtlas.tileW[lightEid] = spec.mapSize;
    ShadowAtlas.tileH[lightEid] = spec.mapSize;
  }
  if (spec.cascades !== undefined) {
    ShadowCascade.cascadeCount[lightEid] = spec.cascades;
    ShadowDirLight.cascades[lightEid] = spec.cascades;
    ShadowCascade.enabled[lightEid] = 1;
  }
  if (spec.bias !== undefined) ShadowBias.bias[lightEid] = spec.bias;
  if (spec.normalBias !== undefined) ShadowBias.normalBias[lightEid] = spec.normalBias;
  if (spec.softness !== undefined) ShadowSoftness.softness[lightEid] = spec.softness;
  if (spec.tint) {
    ShadowTint.tintR[lightEid] = spec.tint[0];
    ShadowTint.tintG[lightEid] = spec.tint[1];
    ShadowTint.tintB[lightEid] = spec.tint[2];
    ShadowTint.enabled[lightEid] = 1;
  }
  if (spec.edgeStyle !== undefined) ShadowSoftness.edgeStyle[lightEid] = spec.edgeStyle;
  if (spec.updateMode !== undefined) ShadowUpdatePolicy.mode[lightEid] = spec.updateMode;
  if (spec.updateInterval !== undefined) ShadowUpdatePolicy.interval[lightEid] = spec.updateInterval;

  ShadowCache.dirty[lightEid] = 1;
  ShadowState.state[lightEid] = SHADOW_STATE.DIRTY;

  return lightEid;
}

/**
 * Spawns a standalone shadow-cascade set entity. Use when a shadow system
 * wants cascade data decoupled from the light entity.
 */
export function spawnShadowCascadeSet(world, spec = {}) {
  const w = world || getShadowWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  const cascadeCount = spec.cascades !== undefined ? spec.cascades : 4;
  ShadowCascade.cascadeCount[eid] = Math.min(cascadeCount, MAX_SHADOW_CASCADES);
  ShadowCascade.enabled[eid] = 1;

  const lambda = spec.splitLambda !== undefined ? spec.splitLambda : 0.5;
  const near = spec.near !== undefined ? spec.near : 0.5;
  const far  = spec.far  !== undefined ? spec.far  : 140.0;

  for (let c = 0; c < cascadeCount && c < MAX_SHADOW_CASCADES; c++) {
    const off = eid * MAX_SHADOW_CASCADES + c;
    const t0 = c / cascadeCount;
    const t1 = (c + 1) / cascadeCount;
    const s0 = near + Math.pow(t0, lambda) * (far - near);
    const s1 = near + Math.pow(t1, lambda) * (far - near);
    ShadowCascade.splitNear[off] = s0;
    ShadowCascade.splitFar[off]  = s1;
    ShadowCascade.mapSize[off]   = spec.mapSize !== undefined ? spec.mapSize : 1024;
    ShadowCascade.bias[off]      = spec.bias !== undefined ? spec.bias : -0.0008;
    ShadowCascade.normalBias[off]= spec.normalBias !== undefined ? spec.normalBias : 0.020;
    ShadowCascade.softness[off]  = spec.softness !== undefined ? spec.softness : 0.05;
  }

  addComponent(w, eid, ShadowCascade);
  return eid;
}

/**
 * Spawns a contact-shadow volume on the given entity.
 */
export function spawnContactShadowVolume(world, spec = {}) {
  const w = world || getShadowWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  ShadowContact.enabled[eid] = 1;
  if (spec.radius !== undefined) ShadowContact.radius[eid] = spec.radius;
  if (spec.thickness !== undefined) ShadowContact.thickness[eid] = spec.thickness;
  if (spec.strength !== undefined) ShadowContact.strength[eid] = spec.strength;
  if (spec.bias !== undefined) ShadowContact.bias[eid] = spec.bias;
  if (spec.rayCount !== undefined) ShadowContact.rayCount[eid] = spec.rayCount;

  addComponent(w, eid, ShadowContact);
  return eid;
}

/* ------------------------------------------------------------------ */
/* 7. UTILITY HELPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Allocates a shadow atlas tile for the given light entity.
 * Returns the tile index, or -1 if the atlas is full.
 */
export function allocateShadowTile(lightEid, width, height, cascadeIndex = -1) {
  if (typeof lightEid !== 'number' || lightEid < 0) return -1;
  if (typeof width !== 'number' || typeof height !== 'number') return -1;
  if (width <= 0 || height <= 0) return -1;

  for (let i = 0; i < MAX_TOTAL_CASCADES; i++) {
    if (ShadowTile.inUse[i] === 0) {
      ShadowTile.ownerEid[i]      = lightEid;
      ShadowTile.x[i]             = 0;
      ShadowTile.y[i]             = 0;
      ShadowTile.w[i]             = width;
      ShadowTile.h[i]             = height;
      ShadowTile.cascadeIndex[i]  = cascadeIndex;
      ShadowTile.inUse[i]         = 1;
      ShadowTile.generation[i]    = (ShadowTile.generation[i] + 1) >>> 0;

      ShadowBudget.allocated[lightEid]     = 1;
      ShadowBudget.allocatedTile[lightEid] = i;
      ShadowAtlas.tileIndex[lightEid]      = i;

      return i;
    }
  }
  return -1;
}

/**
 * Releases the shadow atlas tile assigned to the given light entity.
 */
export function releaseShadowTile(lightEid) {
  const tileIdx = ShadowBudget.allocatedTile[lightEid];
  if (tileIdx < 0 || tileIdx >= MAX_TOTAL_CASCADES) return false;
  if (ShadowTile.inUse[tileIdx] === 0) return false;

  ShadowTile.ownerEid[tileIdx]      = -1;
  ShadowTile.cascadeIndex[tileIdx]  = -1;
  ShadowTile.inUse[tileIdx]         = 0;
  ShadowTile.generation[tileIdx]    = (ShadowTile.generation[tileIdx] + 1) >>> 0;

  ShadowBudget.allocated[lightEid]     = 0;
  ShadowBudget.allocatedTile[lightEid] = -1;
  ShadowAtlas.tileIndex[lightEid]      = -1;
  return true;
}

/**
 * Marks a shadow-casting light as needing a full shadow map re-render.
 */
export function markShadowDirty(lightEid) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  ShadowCache.dirty[lightEid] = 1;
  ShadowState.prevState[lightEid] = ShadowState.state[lightEid];
  ShadowState.state[lightEid] = SHADOW_STATE.DIRTY;
  return true;
}

/**
 * Clears the dirty flag after a successful shadow update.
 */
export function clearShadowDirty(lightEid) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  ShadowCache.dirty[lightEid] = 0;
  ShadowCache.valid[lightEid] = 1;
  ShadowState.prevState[lightEid] = ShadowState.state[lightEid];
  ShadowState.state[lightEid] = SHADOW_STATE.CACHED;
  return true;
}

/**
 * Sets the shadow filter mode on a light.
 */
export function setShadowFilter(lightEid, filterMode) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (filterMode < 0 || filterMode >= SHADOW_FILTER.COUNT) return false;
  ShadowFilter.mode[lightEid] = filterMode;

  // Update kernel size + samples to sensible defaults for the mode.
  switch (filterMode) {
    case SHADOW_FILTER.BASIC:    ShadowFilter.kernelSize[lightEid] = 1; ShadowFilter.samples[lightEid] = 1; break;
    case SHADOW_FILTER.PCF:      ShadowFilter.kernelSize[lightEid] = 3; ShadowFilter.samples[lightEid] = 9; break;
    case SHADOW_FILTER.PCF_SOFT: ShadowFilter.kernelSize[lightEid] = 5; ShadowFilter.samples[lightEid] = 25; break;
    case SHADOW_FILTER.PCSS:     ShadowFilter.kernelSize[lightEid] = 5; ShadowFilter.samples[lightEid] = 16; ShadowFilter.blockerSearch[lightEid] = 1; break;
    case SHADOW_FILTER.VSM:      ShadowFilter.kernelSize[lightEid] = 5; ShadowFilter.samples[lightEid] = 9; break;
    case SHADOW_FILTER.ESM:      ShadowFilter.kernelSize[lightEid] = 3; ShadowFilter.samples[lightEid] = 4; break;
    case SHADOW_FILTER.CONTACT:  ShadowFilter.kernelSize[lightEid] = 1; ShadowFilter.samples[lightEid] = 8; break;
    case SHADOW_FILTER.ANIME:    ShadowFilter.kernelSize[lightEid] = 1; ShadowFilter.samples[lightEid] = 1; break;
    default: break;
  }
  return true;
}

/**
 * Sets the shadow bias + normal bias on a light.
 */
export function setShadowBias(lightEid, bias, normalBias) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (typeof bias === 'number') ShadowBias.bias[lightEid] = bias;
  if (typeof normalBias === 'number') ShadowBias.normalBias[lightEid] = normalBias;
  return true;
}

/**
 * Sets the shadow softness + edge style on a light.
 */
export function setShadowSoftness(lightEid, softness, edgeStyle) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (typeof softness === 'number') ShadowSoftness.softness[lightEid] = softness;
  if (typeof edgeStyle === 'number') ShadowSoftness.edgeStyle[lightEid] = edgeStyle;
  return true;
}

/**
 * Sets the shadow atlas mode on a light.
 */
export function setShadowAtlasMode(lightEid, mode) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (mode < 0 || mode >= SHADOW_ATLAS_MODE.COUNT) return false;
  ShadowAtlas.mode[lightEid] = mode;
  return true;
}

/**
 * Sets the shadow update policy (mode + interval).
 */
export function setShadowUpdatePolicy(lightEid, mode, interval) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (mode < 0 || mode >= SHADOW_UPDATE_MODE.COUNT) return false;
  ShadowUpdatePolicy.mode[lightEid] = mode;
  if (typeof interval === 'number') ShadowUpdatePolicy.interval[lightEid] = Math.max(1, interval | 0);
  return true;
}

/**
 * Enables or disables anime shadow tint on a light.
 */
export function setShadowTint(lightEid, tintRGB, strength, rimStrength) {
  if (typeof lightEid !== 'number' || lightEid < 0) return false;
  if (Array.isArray(tintRGB) && tintRGB.length >= 3) {
    ShadowTint.tintR[lightEid] = tintRGB[0];
    ShadowTint.tintG[lightEid] = tintRGB[1];
    ShadowTint.tintB[lightEid] = tintRGB[2];
  }
  if (typeof strength === 'number') ShadowTint.tintStrength[lightEid] = strength;
  if (typeof rimStrength === 'number') ShadowTint.rimStrength[lightEid] = rimStrength;
  ShadowTint.enabled[lightEid] = 1;
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createShadowWorld() {
  return createWorld({
    components: SHADOW_COMPONENTS,
    time: {
      delta: 0,
      elapsed: 0,
      then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
    },
  });
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_SHADOW_CASCADES,
  MAX_SHADOW_TILES,
  MAX_TOTAL_CASCADES,
  SHADOW_MAP_SIZE,

  // Components
  ShadowCasterRef,
  ShadowReceiverRef,
  ShadowAtlas,
  ShadowCascade,
  ShadowBias,
  ShadowFilter,
  ShadowSoftness,
  ShadowFrustum,
  ShadowCache,
  ShadowBudget,
  ShadowContact,
  ShadowVolume,
  ShadowDirLight,
  ShadowPointLight,
  ShadowSpotLight,
  ShadowAtlasMap,
  ShadowTint,
  ShadowEdge,
  ShadowTile,
  ShadowUpdatePolicy,
  ShadowState,
  SHADOW_COMPONENTS,

  // Enums
  SHADOW_FILTER,
  SHADOW_FILTER_NAME,
  SHADOW_STATE,
  SHADOW_STATE_NAME,
  SHADOW_UPDATE_MODE,
  SHADOW_UPDATE_MODE_NAME,
  SHADOW_ATLAS_MODE,
  SHADOW_ATLAS_MODE_NAME,
  SHADOW_EDGE,
  SHADOW_EDGE_NAME,
  SHADOW_SOURCE,
  SHADOW_SOURCE_NAME,

  // World
  getShadowWorld,
  createShadowWorld,

  // Spawn
  spawnShadowCaster,
  spawnShadowReceiver,
  spawnShadowAtlas,
  spawnShadowCascadeSet,
  spawnContactShadowVolume,

  // Utilities
  allocateShadowTile,
  releaseShadowTile,
  markShadowDirty,
  clearShadowDirty,
  setShadowFilter,
  setShadowBias,
  setShadowSoftness,
  setShadowAtlasMode,
  setShadowUpdatePolicy,
  setShadowTint,
};

export default _defaultExport;