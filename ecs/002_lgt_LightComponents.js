// File : 002
// name : src/ecs/002_lgt_LightComponents.js
// description : bitECS 0.4.0 SoA component definitions for the anime lighting
//               stack on Android mobile. Every light-related piece of state the
//               lighting pipeline needs is declared here as a plain-object
//               component whose fields are pre-sized TypedArrays. This is the
//               bitECS 0.4.0 architecture — there is NO `defineComponent`,
//               NO `Types` export, NO separate store registry. Components are
//               plain JS objects passed directly to `createWorld({ components })`
//               and attached via `addComponent(world, eid, ComponentObject)`.
//
//               Components declared:
//                 • Transform        — position / rotation / scale SoA
//                 • LightRef         — canonical light descriptor (type, color,
//                                      intensity, range, angle, penumbra, decay)
//                 • LightState       — runtime state (enabled, dirty, visible,
//                                      castShadow, receivesShadow)
//                 • LightShadow      — shadow parameters (bias, normalBias,
//                                      mapSize, cascades, filter, softness)
//                 • LightCluster     — cluster grid membership + cell id
//                 • LightBudget      — per-light cost budget + LOD state
//                 • LightPriority    — sort key + priority bucket
//                 • LightBehavior    — attached behavior slots (SoA over
//                                      MAX_BEHAVIORS_PER_LIGHT)
//                 • LightComposite   — composite kind + member mask
//                 • LightIndoor      — indoor / outdoor / transition weights
//                 • LightIES         — IES profile handle + ballast factor
//                 • LightEmissive    — emissive proxy parameters
//                 • LightFlicker     — flicker / pulse parameters
//                 • LightDayCycle    — day-cycle binding
//                 • LightTag         — bitmask tag for scene filtering
//                 • CameraTag        — marker component for the active camera
//                 • GIProbeRef       — GI probe entity reference
//                 • AOVolumeRef      — AO volume entity reference
//
//               Also exports:
//                 • MAX_ENTITIES constant (100000)
//                 • All component objects (frozen, plain, SoA)
//                 • LIGHT_TYPE / LIGHT_KIND / LIGHT_BEHAVIOR / LIGHT_PRIORITY /
//                   LIGHT_TAG enums
//                 • ECS-world components bundle (ready for createWorld)
//                 • Entity factories for every sanctioned light kind:
//                     spawnAmbientLight, spawnHemisphereLight,
//                     spawnDirectionalLight (sun/moon), spawnPointLight,
//                     spawnSpotLight, spawnRectAreaLight,
//                     spawnFireLight, spawnNeonLight, spawnMagicGlow,
//                     spawnInteriorLamp, spawnWindowShaft, spawnCaustic,
//                     spawnAurora, spawnCamera, spawnGIProbe, spawnAOVolume
//                 • Utility helpers: ensureCapacity, markLightDirty,
//                   clearLightDirty, setLightActive, isLightActive
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only (createWorld / addEntity / addComponent / query /
//               hasComponent / removeComponent / entityExists); every typed
//               array is sized once to MAX_ENTITIES = 100000 so no resizing
//               ever happens during gameplay on Android.
// best for : Single source of truth for every light-related ECS component
//            in the anime lighting stack. Every downstream system
//            (003_lgt_ShadowComponents.js through 380_lgt_lights.js) imports
//            its component definitions from here so entity IDs, array
//            layouts, and light type enums stay consistent across CPU
//            sampling, GPU uploads, and worker communication.
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

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Fixed maximum number of ECS entities for the entire anime lighting stack.
 * Sized once at module load; never resized. All typed arrays below are
 * allocated to this length.
 */
export const MAX_ENTITIES = 100000;

/**
 * Per-tier budgets that control how many of each light type can exist
 * simultaneously. Feeds directly into the pooling/culling systems.
 */
export const MAX_LIGHTS_BY_TIER = Object.freeze({
  HIGH:   Object.freeze({ total: 512, shadowCasting: 8,  clusterCells: 128 }),
  MEDIUM: Object.freeze({ total: 256, shadowCasting: 4,  clusterCells:  64 }),
  LOW:    Object.freeze({ total: 128, shadowCasting: 2,  clusterCells:  32 }),
});

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * The six sanctioned Three.js r185 light types.
 * Every light entity MUST have one of these values in its LightRef.type.
 */
export const LIGHT_TYPE = Object.freeze({
  AMBIENT:     0,
  HEMISPHERE:  1,
  DIRECTIONAL: 2,
  POINT:       3,
  SPOT:        4,
  RECT_AREA:   5,
  COUNT:       6,
});

export const LIGHT_TYPE_NAME = Object.freeze([
  'ambient',
  'hemisphere',
  'directional',
  'point',
  'spot',
  'rect_area',
]);

/**
 * Composite / logical light kinds. Every light entity also carries a
 * LIGHT_KIND value so downstream systems can filter (e.g. sun_cycle vs
 * interior_lamp) without inspecting the underlying THREE light class.
 */
export const LIGHT_KIND = Object.freeze({
  GENERIC:        0,
  SUN:            1,
  MOON:           2,
  FIRE:           3,
  NEON:           4,
  MAGIC:          5,
  INTERIOR_LAMP:  6,
  WINDOW_SHAFT:   7,
  CAUSTIC:        8,
  AURORA:         9,
  BIO_LUMINESCENT:10,
  REFLECTION:     11,
  COUNT:          12,
});

export const LIGHT_KIND_NAME = Object.freeze([
  'generic',
  'sun',
  'moon',
  'fire',
  'neon',
  'magic',
  'interior_lamp',
  'window_shaft',
  'caustic',
  'aurora',
  'bio_luminescent',
  'reflection',
]);

/**
 * Behavior slot ids — match the behaviors registered in
 * 001_lgt_ThreeLightsOnlyPolicy.js.
 */
export const LIGHT_BEHAVIOR = Object.freeze({
  NONE:               0,
  FLICKER:            1,
  PULSE:              2,
  DAY_CYCLE:          3,
  IES_PROFILE:        4,
  TEMPERATURE_DRIFT:  5,
  RIM_BOOST:          6,
  BIOME_BLEND:        7,
  INTERIOR_CROSSFADE: 8,
  COUNT:              9,
});

export const LIGHT_BEHAVIOR_NAME = Object.freeze([
  'none',
  'flicker',
  'pulse',
  'day_cycle',
  'ies_profile',
  'temperature_drift',
  'rim_boost',
  'biome_blend',
  'interior_crossfade',
]);

export const MAX_BEHAVIORS_PER_LIGHT = 4;

/**
 * Light priority buckets for sorted GPU upload.
 */
export const LIGHT_PRIORITY = Object.freeze({
  CRITICAL: 0,
  HIGH:     1,
  NORMAL:   2,
  LOW:      3,
  IDLE:     4,
  COUNT:    5,
});

export const LIGHT_PRIORITY_NAME = Object.freeze([
  'critical',
  'high',
  'normal',
  'low',
  'idle',
]);

/**
 * Bitmask tags for scene / biome filtering. Multiple tags can be set on
 * one entity.
 */
export const LIGHT_TAG = Object.freeze({
  NONE:       0,
  INDOOR:     1 << 0,
  EXTERIOR:   1 << 1,
  DESERT:     1 << 2,
  SNOW:       1 << 3,
  SEA:        1 << 4,
  HOUSE:      1 << 5,
  CANYON:     1 << 6,
  FOREST:     1 << 7,
  NIGHT:      1 << 8,
  DAY:        1 << 9,
  CRITICAL:   1 << 10,
  OPTIONAL:   1 << 11,
  DEBUG:      1 << 12,
});

/**
 * Per-light runtime state flags (bitmask, stored in Uint8Array).
 */
export const LIGHT_STATE_FLAG = Object.freeze({
  NONE:             0,
  ACTIVE:           1 << 0,
  VISIBLE:          1 << 1,
  DIRTY:            1 << 2,
  CAST_SHADOW:      1 << 3,
  RECEIVES_SHADOW:  1 << 4,
  IN_CLUSTER:       1 << 5,
  IN_FRUSTUM:       1 << 6,
  INDOOR:           1 << 7,
});

/**
 * Indoor / outdoor / transition state.
 */
export const LIGHT_ENVIRONMENT = Object.freeze({
  OUTDOOR:    0,
  TRANSITION: 1,
  INDOOR:     2,
  COUNT:      3,
});

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * Transform — position / quaternion / scale for every light entity.
 */
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

/**
 * Target — for DirectionalLight and SpotLight target positions.
 */
export const Target = {
  x: new Float32Array(MAX_ENTITIES),
  y: new Float32Array(MAX_ENTITIES),
  z: new Float32Array(MAX_ENTITIES),
  active: new Uint8Array(MAX_ENTITIES),   // 1 if this entity has an active target
};

/**
 * LightRef — the canonical light descriptor. Every light entity MUST have
 * this component. Fields mirror the union of every sanctioned light type's
 * parameters.
 */
export const LightRef = {
  type:      new Uint8Array(MAX_ENTITIES),     // LIGHT_TYPE
  kind:      new Uint8Array(MAX_ENTITIES),     // LIGHT_KIND
  colorR:    new Float32Array(MAX_ENTITIES),
  colorG:    new Float32Array(MAX_ENTITIES),
  colorB:    new Float32Array(MAX_ENTITIES),
  intensity: new Float32Array(MAX_ENTITIES),
  range:     new Float32Array(MAX_ENTITIES),   // distance for Point / Spot
  decay:     new Float32Array(MAX_ENTITIES),   // 2.0 typical
  angle:     new Float32Array(MAX_ENTITIES),   // Spot cone half-angle (radians)
  penumbra:  new Float32Array(MAX_ENTITIES),   // Spot soft edge [0,1]
  width:     new Float32Array(MAX_ENTITIES),   // RectAreaLight width
  height:    new Float32Array(MAX_ENTITIES),   // RectAreaLight height
  groundColorR: new Float32Array(MAX_ENTITIES),// HemisphereLight ground color
  groundColorG: new Float32Array(MAX_ENTITIES),
  groundColorB: new Float32Array(MAX_ENTITIES),
};

/**
 * LightState — runtime state flags + dirty / active tracking.
 */
export const LightState = {
  flags:        new Uint16Array(MAX_ENTITIES),    // LIGHT_STATE_FLAG bitmask
  env:          new Uint8Array(MAX_ENTITIES),     // LIGHT_ENVIRONMENT
  envBlend:     new Float32Array(MAX_ENTITIES),   // [0..1] indoor weight
  lastUpdateFrame: new Uint32Array(MAX_ENTITIES),
  lastDirtyFrame:  new Uint32Array(MAX_ENTITIES),
  frameActive:  new Uint8Array(MAX_ENTITIES),
};

/**
 * LightShadow — shadow mapping parameters for shadow-casting lights.
 */
export const LightShadow = {
  enabled:       new Uint8Array(MAX_ENTITIES),
  mapSize:       new Uint16Array(MAX_ENTITIES),   // 512 / 1024 / 2048 / 4096
  cascadeCount:  new Uint8Array(MAX_ENTITIES),    // 1..4
  filter:        new Uint8Array(MAX_ENTITIES),    // 0=basic 1=pcf 2=pcfsoft 3=pcss 4=vsm
  bias:          new Float32Array(MAX_ENTITIES),
  normalBias:    new Float32Array(MAX_ENTITIES),
  softness:      new Float32Array(MAX_ENTITIES),
  distance:      new Float32Array(MAX_ENTITIES),
  atlasTileX:    new Uint16Array(MAX_ENTITIES),
  atlasTileY:    new Uint16Array(MAX_ENTITIES),
  atlasTileW:    new Uint16Array(MAX_ENTITIES),
  atlasTileH:    new Uint16Array(MAX_ENTITIES),
  atlasValid:    new Uint8Array(MAX_ENTITIES),
};

/**
 * LightCluster — per-light cluster grid membership.
 * A light may be referenced by multiple cells, but the primary cell is
 * stored here for the fast path.
 */
export const LightCluster = {
  cellX:    new Int16Array(MAX_ENTITIES),
  cellY:    new Int16Array(MAX_ENTITIES),
  cellZ:    new Int16Array(MAX_ENTITIES),
  cellCount:new Uint16Array(MAX_ENTITIES),   // number of cells this light touches
  clusterDirty: new Uint8Array(MAX_ENTITIES),
};

/**
 * LightBudget — per-light cost accounting + LOD state.
 */
export const LightBudget = {
  cost:        new Float32Array(MAX_ENTITIES),    // estimated GPU cost
  costEma:     new Float32Array(MAX_ENTITIES),
  lod:         new Uint8Array(MAX_ENTITIES),      // 0 = full, 1 = half, 2 = quarter
  lodTarget:   new Uint8Array(MAX_ENTITIES),
  lastLodFrame:new Uint32Array(MAX_ENTITIES),
  shadowAllocated: new Uint8Array(MAX_ENTITIES),
};

/**
 * LightPriority — sorted upload priority.
 */
export const LightPriority = {
  bucket:     new Uint8Array(MAX_ENTITIES),    // LIGHT_PRIORITY
  sortKey:    new Float32Array(MAX_ENTITIES),  // larger = more important
  frameRank:  new Uint16Array(MAX_ENTITIES),   // stable rank within frame
};

/**
 * LightBehavior — attached behavior slots.
 * Each light can carry up to MAX_BEHAVIORS_PER_LIGHT behaviors, stored as
 * a fixed SoA layout: behaviorIds[eid * MAX_BEHAVIORS_PER_LIGHT + slot].
 */
export const LightBehavior = {
  behaviorIds:   new Uint8Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT),
  behaviorCount: new Uint8Array(MAX_ENTITIES),
  behaviorCtx0:  new Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT), // generic float ctx A
  behaviorCtx1:  new Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT), // generic float ctx B
  behaviorCtx2:  new Float32Array(MAX_ENTITIES * MAX_BEHAVIORS_PER_LIGHT), // generic float ctx C
};

/**
 * LightComposite — composite / group membership.
 */
export const LightComposite = {
  compositeKind: new Uint8Array(MAX_ENTITIES),   // LIGHT_KIND
  anchorEid:     new Int32Array(MAX_ENTITIES),   // -1 if this IS the anchor
  memberCount:   new Uint8Array(MAX_ENTITIES),
  memberMask:    new Uint16Array(MAX_ENTITIES),  // bitmask of member slots
};

/**
 * LightIndoor — indoor / outdoor crossfade weights.
 */
export const LightIndoor = {
  indoorWeight:    new Float32Array(MAX_ENTITIES),
  outdoorWeight:   new Float32Array(MAX_ENTITIES),
  transitionRate:  new Float32Array(MAX_ENTITIES),
  portalVisible:   new Uint8Array(MAX_ENTITIES),
  occludedByWalls: new Uint8Array(MAX_ENTITIES),
};

/**
 * LightIES — IES profile handle + ballast.
 */
export const LightIES = {
  hasProfile:    new Uint8Array(MAX_ENTITIES),
  profileId:     new Uint16Array(MAX_ENTITIES),
  ballastFactor: new Float32Array(MAX_ENTITIES),
  candelaScale:  new Float32Array(MAX_ENTITIES),
  luminousFlux:  new Float32Array(MAX_ENTITIES),
};

/**
 * LightEmissive — emissive proxy parameters (for the glowing core).
 */
export const LightEmissive = {
  emissiveR:     new Float32Array(MAX_ENTITIES),
  emissiveG:     new Float32Array(MAX_ENTITIES),
  emissiveB:     new Float32Array(MAX_ENTITIES),
  emissiveScale: new Float32Array(MAX_ENTITIES),
  proxyVisible:  new Uint8Array(MAX_ENTITIES),
};

/**
 * LightFlicker — flicker / pulse behavior parameters.
 */
export const LightFlicker = {
  baseIntensity: new Float32Array(MAX_ENTITIES),
  amplitude:     new Float32Array(MAX_ENTITIES),
  hz:            new Float32Array(MAX_ENTITIES),
  phase:         new Float32Array(MAX_ENTITIES),
  enabled:       new Uint8Array(MAX_ENTITIES),
};

/**
 * LightDayCycle — day cycle binding.
 */
export const LightDayCycle = {
  dayCycle:      new Float32Array(MAX_ENTITIES),
  daySpeed:      new Float32Array(MAX_ENTITIES),
  enabled:       new Uint8Array(MAX_ENTITIES),
  phaseOffset:   new Float32Array(MAX_ENTITIES),
  maxElevation:  new Float32Array(MAX_ENTITIES),
};

/**
 * LightTag — bitmask tags for scene / biome filtering.
 */
export const LightTag = {
  mask: new Uint16Array(MAX_ENTITIES),
};

/**
 * CameraTag — marker for the active camera entity.
 */
export const CameraTag = {
  active: new Uint8Array(MAX_ENTITIES),
  fov:    new Float32Array(MAX_ENTITIES),
  near:   new Float32Array(MAX_ENTITIES),
  far:    new Float32Array(MAX_ENTITIES),
  aspect: new Float32Array(MAX_ENTITIES),
};

/**
 * GIProbeRef — GI probe entity reference.
 */
export const GIProbeRef = {
  probeX:       new Float32Array(MAX_ENTITIES),
  probeY:       new Float32Array(MAX_ENTITIES),
  probeZ:       new Float32Array(MAX_ENTITIES),
  irradianceR:  new Float32Array(MAX_ENTITIES),
  irradianceG:  new Float32Array(MAX_ENTITIES),
  irradianceB:  new Float32Array(MAX_ENTITIES),
  skyOcclusion: new Float32Array(MAX_ENTITIES),
  indoorFactor: new Float32Array(MAX_ENTITIES),
  dirty:        new Uint8Array(MAX_ENTITIES),
};

/**
 * AOVolumeRef — AO volume entity reference.
 */
export const AOVolumeRef = {
  radius:      new Float32Array(MAX_ENTITIES),
  intensity:   new Float32Array(MAX_ENTITIES),
  sampleCount: new Uint8Array(MAX_ENTITIES),
  halfRes:     new Uint8Array(MAX_ENTITIES),
  temporal:    new Uint8Array(MAX_ENTITIES),
};

/* ------------------------------------------------------------------ */
/* 3. ECS WORLD COMPONENT BUNDLE                                      */
/* ------------------------------------------------------------------ */

/**
 * Ready-to-use component bundle for `createWorld({ components })`.
 * Import this bundle when bootstrapping a dedicated lighting world.
 */
export const LIGHTING_COMPONENTS = Object.freeze({
  Transform,
  Target,
  LightRef,
  LightState,
  LightShadow,
  LightCluster,
  LightBudget,
  LightPriority,
  LightBehavior,
  LightComposite,
  LightIndoor,
  LightIES,
  LightEmissive,
  LightFlicker,
  LightDayCycle,
  LightTag,
  CameraTag,
  GIProbeRef,
  AOVolumeRef,
});

/* ------------------------------------------------------------------ */
/* 4. ENTITY COUNTER + SAFE ID HELPER                                 */
/* ------------------------------------------------------------------ */

let _nextEntityHint = 0;

/**
 * The bitECS 0.4.0 world we register our components against. Downstream
 * systems should call `getLightWorld()` to obtain the shared handle, or
 * pass their own world to the spawn functions.
 */
let _lightWorld = null;

/**
 * Lazy-create the light world with all components pre-registered.
 */
export function getLightWorld() {
  if (_lightWorld) return _lightWorld;
  try {
    _lightWorld = createWorld({
      components: LIGHTING_COMPONENTS,
      time: {
        delta: 0,
        elapsed: 0,
        then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      },
    });
  } catch (e) {
    const log = getDefaultLogger();
    if (log) log.error(LOG_CHANNEL.LIGHTS, `[002_lgt_LightComponents] failed to create light world: ${e && e.message}`);
    _lightWorld = null;
  }
  return _lightWorld;
}

/**
 * Validate that an entity id is within our fixed MAX_ENTITIES range.
 */
export function isValidEntityId(eid) {
  return typeof eid === 'number' && eid >= 0 && eid < MAX_ENTITIES;
}

/* ------------------------------------------------------------------ */
/* 5. SPAWN HELPERS — SANCTIONED LIGHT TYPES                          */
/* ------------------------------------------------------------------ */

/**
 * Attach the shared light components to an entity and initialize the
 * default values. Called by every spawn function below.
 */
function _attachLightBase(world, eid, type, kind) {
  // Transform
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

  // Target
  Target.x[eid] = 0;
  Target.y[eid] = 0;
  Target.z[eid] = 0;
  Target.active[eid] = 0;

  // LightRef
  LightRef.type[eid]      = type;
  LightRef.kind[eid]      = kind;
  LightRef.colorR[eid]    = 1.0;
  LightRef.colorG[eid]    = 1.0;
  LightRef.colorB[eid]    = 1.0;
  LightRef.intensity[eid] = 1.0;
  LightRef.range[eid]     = 0.0;
  LightRef.decay[eid]     = 2.0;
  LightRef.angle[eid]     = Math.PI / 4;
  LightRef.penumbra[eid]  = 0.1;
  LightRef.width[eid]     = 1.0;
  LightRef.height[eid]    = 1.0;
  LightRef.groundColorR[eid] = 0.2;
  LightRef.groundColorG[eid] = 0.2;
  LightRef.groundColorB[eid] = 0.2;

  // LightState
  LightState.flags[eid] = LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE;
  LightState.env[eid] = LIGHT_ENVIRONMENT.OUTDOOR;
  LightState.envBlend[eid] = 0.0;
  LightState.lastUpdateFrame[eid] = 0;
  LightState.lastDirtyFrame[eid] = 0;
  LightState.frameActive[eid] = 1;

  // LightShadow
  LightShadow.enabled[eid] = 0;
  LightShadow.mapSize[eid] = 1024;
  LightShadow.cascadeCount[eid] = 1;
  LightShadow.filter[eid] = 1;
  LightShadow.bias[eid] = -0.0008;
  LightShadow.normalBias[eid] = 0.020;
  LightShadow.softness[eid] = 0.05;
  LightShadow.distance[eid] = 100.0;
  LightShadow.atlasTileX[eid] = 0;
  LightShadow.atlasTileY[eid] = 0;
  LightShadow.atlasTileW[eid] = 0;
  LightShadow.atlasTileH[eid] = 0;
  LightShadow.atlasValid[eid] = 0;

  // LightCluster
  LightCluster.cellX[eid] = -1;
  LightCluster.cellY[eid] = -1;
  LightCluster.cellZ[eid] = -1;
  LightCluster.cellCount[eid] = 0;
  LightCluster.clusterDirty[eid] = 1;

  // LightBudget
  LightBudget.cost[eid] = 1.0;
  LightBudget.costEma[eid] = 1.0;
  LightBudget.lod[eid] = 0;
  LightBudget.lodTarget[eid] = 0;
  LightBudget.lastLodFrame[eid] = 0;
  LightBudget.shadowAllocated[eid] = 0;

  // LightPriority
  LightPriority.bucket[eid] = LIGHT_PRIORITY.NORMAL;
  LightPriority.sortKey[eid] = 1.0;
  LightPriority.frameRank[eid] = 0;

  // LightBehavior (reset slots)
  for (let b = 0; b < MAX_BEHAVIORS_PER_LIGHT; b++) {
    const off = eid * MAX_BEHAVIORS_PER_LIGHT + b;
    LightBehavior.behaviorIds[off] = LIGHT_BEHAVIOR.NONE;
    LightBehavior.behaviorCtx0[off] = 0;
    LightBehavior.behaviorCtx1[off] = 0;
    LightBehavior.behaviorCtx2[off] = 0;
  }
  LightBehavior.behaviorCount[eid] = 0;

  // LightComposite
  LightComposite.compositeKind[eid] = kind;
  LightComposite.anchorEid[eid] = -1;
  LightComposite.memberCount[eid] = 1;
  LightComposite.memberMask[eid] = 0;

  // LightIndoor
  LightIndoor.indoorWeight[eid] = 0.0;
  LightIndoor.outdoorWeight[eid] = 1.0;
  LightIndoor.transitionRate[eid] = 1.0;
  LightIndoor.portalVisible[eid] = 0;
  LightIndoor.occludedByWalls[eid] = 0;

  // LightIES
  LightIES.hasProfile[eid] = 0;
  LightIES.profileId[eid] = 0;
  LightIES.ballastFactor[eid] = 1.0;
  LightIES.candelaScale[eid] = 1.0;
  LightIES.luminousFlux[eid] = 0.0;

  // LightEmissive
  LightEmissive.emissiveR[eid] = 0.0;
  LightEmissive.emissiveG[eid] = 0.0;
  LightEmissive.emissiveB[eid] = 0.0;
  LightEmissive.emissiveScale[eid] = 1.0;
  LightEmissive.proxyVisible[eid] = 0;

  // LightFlicker
  LightFlicker.baseIntensity[eid] = 1.0;
  LightFlicker.amplitude[eid] = 0.0;
  LightFlicker.hz[eid] = 8.0;
  LightFlicker.phase[eid] = 0.0;
  LightFlicker.enabled[eid] = 0;

  // LightDayCycle
  LightDayCycle.dayCycle[eid] = 0.5;
  LightDayCycle.daySpeed[eid] = 0.004;
  LightDayCycle.enabled[eid] = 0;
  LightDayCycle.phaseOffset[eid] = 0.0;
  LightDayCycle.maxElevation[eid] = 75.0;

  // LightTag
  LightTag.mask[eid] = LIGHT_TAG.NONE;

  // Attach all light-related components to the entity.
  addComponent(world, eid, Transform);
  addComponent(world, eid, Target);
  addComponent(world, eid, LightRef);
  addComponent(world, eid, LightState);
  addComponent(world, eid, LightShadow);
  addComponent(world, eid, LightCluster);
  addComponent(world, eid, LightBudget);
  addComponent(world, eid, LightPriority);
  addComponent(world, eid, LightBehavior);
  addComponent(world, eid, LightComposite);
  addComponent(world, eid, LightIndoor);
  addComponent(world, eid, LightIES);
  addComponent(world, eid, LightEmissive);
  addComponent(world, eid, LightFlicker);
  addComponent(world, eid, LightDayCycle);
  addComponent(world, eid, LightTag);
}

/* ------------------------------------------------------------------ */
/* 6. SPAWN FUNCTIONS — ONE PER SANCTIONED LIGHT TYPE                 */
/* ------------------------------------------------------------------ */

export function spawnAmbientLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.AMBIENT, spec.kind !== undefined ? spec.kind : LIGHT_KIND.GENERIC);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  if (spec.ownerId !== undefined) LightState.lastUpdateFrame[eid] = spec.ownerId;
  return eid;
}

export function spawnHemisphereLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.HEMISPHERE, spec.kind !== undefined ? spec.kind : LIGHT_KIND.GENERIC);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.groundColor) { LightRef.groundColorR[eid] = spec.groundColor[0]; LightRef.groundColorG[eid] = spec.groundColor[1]; LightRef.groundColorB[eid] = spec.groundColor[2]; }
  if (spec.position) { Transform.x[eid] = spec.position[0]; Transform.y[eid] = spec.position[1]; Transform.z[eid] = spec.position[2]; }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  return eid;
}

export function spawnDirectionalLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.DIRECTIONAL, spec.kind !== undefined ? spec.kind : LIGHT_KIND.SUN);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.position) { Transform.x[eid] = spec.position[0]; Transform.y[eid] = spec.position[1]; Transform.z[eid] = spec.position[2]; }
  if (spec.target) { Target.x[eid] = spec.target[0]; Target.y[eid] = spec.target[1]; Target.z[eid] = spec.target[2]; Target.active[eid] = 1; }
  if (spec.castShadow) {
    LightShadow.enabled[eid] = 1;
    LightState.flags[eid] |= LIGHT_STATE_FLAG.CAST_SHADOW;
    if (spec.shadowMapSize !== undefined) LightShadow.mapSize[eid] = spec.shadowMapSize;
    if (spec.shadowCascades !== undefined) LightShadow.cascadeCount[eid] = spec.shadowCascades;
    if (spec.shadowBias !== undefined) LightShadow.bias[eid] = spec.shadowBias;
    if (spec.shadowNormalBias !== undefined) LightShadow.normalBias[eid] = spec.shadowNormalBias;
  }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  if (spec.dayCycle !== undefined) {
    LightDayCycle.enabled[eid] = 1;
    LightDayCycle.dayCycle[eid] = spec.dayCycle;
    if (spec.daySpeed !== undefined) LightDayCycle.daySpeed[eid] = spec.daySpeed;
  }
  return eid;
}

export function spawnPointLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.POINT, spec.kind !== undefined ? spec.kind : LIGHT_KIND.GENERIC);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.distance !== undefined) LightRef.range[eid] = spec.distance;
  if (spec.decay !== undefined) LightRef.decay[eid] = spec.decay;
  if (spec.position) { Transform.x[eid] = spec.position[0]; Transform.y[eid] = spec.position[1]; Transform.z[eid] = spec.position[2]; }
  if (spec.castShadow) {
    LightShadow.enabled[eid] = 1;
    LightState.flags[eid] |= LIGHT_STATE_FLAG.CAST_SHADOW;
  }
  if (spec.flicker) {
    LightFlicker.enabled[eid] = 1;
    LightFlicker.baseIntensity[eid] = spec.intensity !== undefined ? spec.intensity : 1.0;
    if (spec.flickerAmplitude !== undefined) LightFlicker.amplitude[eid] = spec.flickerAmplitude;
    if (spec.flickerHz !== undefined) LightFlicker.hz[eid] = spec.flickerHz;
    LightBehavior.behaviorIds[eid * MAX_BEHAVIORS_PER_LIGHT + 0] = LIGHT_BEHAVIOR.FLICKER;
    LightBehavior.behaviorCount[eid] = 1;
  }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  return eid;
}

export function spawnSpotLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.SPOT, spec.kind !== undefined ? spec.kind : LIGHT_KIND.GENERIC);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.distance !== undefined) LightRef.range[eid] = spec.distance;
  if (spec.decay !== undefined) LightRef.decay[eid] = spec.decay;
  if (spec.angle !== undefined) LightRef.angle[eid] = spec.angle;
  if (spec.penumbra !== undefined) LightRef.penumbra[eid] = spec.penumbra;
  if (spec.position) { Transform.x[eid] = spec.position[0]; Transform.y[eid] = spec.position[1]; Transform.z[eid] = spec.position[2]; }
  if (spec.target) { Target.x[eid] = spec.target[0]; Target.y[eid] = spec.target[1]; Target.z[eid] = spec.target[2]; Target.active[eid] = 1; }
  if (spec.castShadow) {
    LightShadow.enabled[eid] = 1;
    LightState.flags[eid] |= LIGHT_STATE_FLAG.CAST_SHADOW;
  }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  return eid;
}

export function spawnRectAreaLight(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachLightBase(w, eid, LIGHT_TYPE.RECT_AREA, spec.kind !== undefined ? spec.kind : LIGHT_KIND.GENERIC);
  if (spec.intensity !== undefined) LightRef.intensity[eid] = spec.intensity;
  if (spec.color) { LightRef.colorR[eid] = spec.color[0]; LightRef.colorG[eid] = spec.color[1]; LightRef.colorB[eid] = spec.color[2]; }
  if (spec.width !== undefined) LightRef.width[eid] = spec.width;
  if (spec.height !== undefined) LightRef.height[eid] = spec.height;
  if (spec.position) { Transform.x[eid] = spec.position[0]; Transform.y[eid] = spec.position[1]; Transform.z[eid] = spec.position[2]; }
  if (spec.tags !== undefined) LightTag.mask[eid] = spec.tags;
  return eid;
}

/* ------------------------------------------------------------------ */
/* 7. SPAWN FUNCTIONS — COMPOSITE LIGHT KINDS                         */
/* ------------------------------------------------------------------ */

export function spawnFireLight(world, spec = {}) {
  const eid = spawnPointLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.FIRE,
    color: spec.color || [1.0, 0.55, 0.15],
    intensity: spec.intensity !== undefined ? spec.intensity : 2.5,
    distance: spec.distance !== undefined ? spec.distance : 12,
    decay: 2.0,
    castShadow: spec.castShadow !== false,
    flicker: true,
    flickerAmplitude: 0.18,
    flickerHz: 9.0,
  }));
  if (eid >= 0) {
    LightEmissive.emissiveR[eid] = 1.0;
    LightEmissive.emissiveG[eid] = 0.55;
    LightEmissive.emissiveB[eid] = 0.15;
    LightEmissive.proxyVisible[eid] = 1;
    LightPriority.bucket[eid] = LIGHT_PRIORITY.HIGH;
    LightPriority.sortKey[eid] = 2.0;
  }
  return eid;
}

export function spawnNeonLight(world, spec = {}) {
  return spawnRectAreaLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.NEON,
    color: spec.color || [0.9, 0.4, 0.7],
    intensity: spec.intensity !== undefined ? spec.intensity : 3.0,
    width: spec.width !== undefined ? spec.width : 1.2,
    height: spec.height !== undefined ? spec.height : 0.2,
  }));
}

export function spawnMagicGlow(world, spec = {}) {
  const eid = spawnPointLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.MAGIC,
    color: spec.color || [1.0, 0.85, 0.45],
    intensity: spec.intensity !== undefined ? spec.intensity : 3.5,
    distance: spec.distance !== undefined ? spec.distance : 8,
    decay: 2.0,
    flicker: true,
    flickerAmplitude: 0.25,
    flickerHz: 5.0,
  }));
  if (eid >= 0) {
    LightEmissive.emissiveR[eid] = 1.0;
    LightEmissive.emissiveG[eid] = 0.85;
    LightEmissive.emissiveB[eid] = 0.45;
    LightEmissive.proxyVisible[eid] = 1;
    LightPriority.bucket[eid] = LIGHT_PRIORITY.CRITICAL;
    LightPriority.sortKey[eid] = 3.0;
  }
  return eid;
}

export function spawnInteriorLamp(world, spec = {}) {
  const eid = spawnPointLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.INTERIOR_LAMP,
    color: spec.color || [1.0, 0.85, 0.65],
    intensity: spec.intensity !== undefined ? spec.intensity : 1.8,
    distance: spec.distance !== undefined ? spec.distance : 6,
    decay: 2.0,
    castShadow: spec.castShadow !== false,
  }));
  if (eid >= 0) {
    LightIndoor.indoorWeight[eid] = 1.0;
    LightIndoor.outdoorWeight[eid] = 0.0;
    LightTag.mask[eid] |= LIGHT_TAG.INDOOR;
  }
  return eid;
}

export function spawnWindowShaft(world, spec = {}) {
  return spawnDirectionalLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.WINDOW_SHAFT,
    color: spec.color || [1.0, 0.94, 0.82],
    intensity: spec.intensity !== undefined ? spec.intensity : 1.2,
    castShadow: true,
  }));
}

export function spawnCaustic(world, spec = {}) {
  const eid = spawnPointLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.CAUSTIC,
    color: spec.color || [0.6, 0.95, 1.0],
    intensity: spec.intensity !== undefined ? spec.intensity : 0.8,
    distance: spec.distance !== undefined ? spec.distance : 4,
    decay: 2.5,
  }));
  if (eid >= 0) {
    LightBehavior.behaviorIds[eid * MAX_BEHAVIORS_PER_LIGHT + 0] = LIGHT_BEHAVIOR.PULSE;
    LightBehavior.behaviorCount[eid] = 1;
  }
  return eid;
}

export function spawnAurora(world, spec = {}) {
  return spawnHemisphereLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.AURORA,
    color: spec.color || [0.35, 0.95, 0.75],
    intensity: spec.intensity !== undefined ? spec.intensity : 0.45,
    groundColor: spec.groundColor || [0.05, 0.10, 0.20],
  }));
}

export function spawnSun(world, spec = {}) {
  return spawnDirectionalLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.SUN,
    color: spec.color || [1.0, 0.96, 0.85],
    intensity: spec.intensity !== undefined ? spec.intensity : 1.25,
    position: spec.position || [50, 80, 30],
    target: spec.target || [0, 0, 0],
    castShadow: true,
    shadowMapSize: spec.shadowMapSize !== undefined ? spec.shadowMapSize : 2048,
    shadowCascades: spec.shadowCascades !== undefined ? spec.shadowCascades : 4,
    dayCycle: spec.dayCycle !== undefined ? spec.dayCycle : 0.38,
    daySpeed: spec.daySpeed !== undefined ? spec.daySpeed : 0.004,
  }));
}

export function spawnMoon(world, spec = {}) {
  return spawnDirectionalLight(world, Object.assign({}, spec, {
    kind: LIGHT_KIND.MOON,
    color: spec.color || [0.42, 0.48, 0.70],
    intensity: spec.intensity !== undefined ? spec.intensity : 0.35,
    position: spec.position || [-40, 60, -30],
    target: spec.target || [0, 0, 0],
    castShadow: spec.castShadow === true,
    dayCycle: spec.dayCycle !== undefined ? spec.dayCycle : 0.0,
    daySpeed: spec.daySpeed !== undefined ? spec.daySpeed : 0.0008,
  }));
}

/* ------------------------------------------------------------------ */
/* 8. SPAWN FUNCTIONS — SUPPORT ENTITIES                              */
/* ------------------------------------------------------------------ */

export function spawnCamera(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  Transform.x[eid] = spec.position ? spec.position[0] : 0;
  Transform.y[eid] = spec.position ? spec.position[1] : 8;
  Transform.z[eid] = spec.position ? spec.position[2] : 34;
  Transform.qx[eid] = 0;
  Transform.qy[eid] = 0;
  Transform.qz[eid] = 0;
  Transform.qw[eid] = 1;
  Transform.sx[eid] = 1;
  Transform.sy[eid] = 1;
  Transform.sz[eid] = 1;

  CameraTag.active[eid] = 1;
  CameraTag.fov[eid]    = spec.fov !== undefined ? spec.fov : 55;
  CameraTag.near[eid]   = spec.near !== undefined ? spec.near : 0.1;
  CameraTag.far[eid]    = spec.far !== undefined ? spec.far : 1000;
  CameraTag.aspect[eid] = spec.aspect !== undefined ? spec.aspect : 0.56;

  addComponent(w, eid, Transform);
  addComponent(w, eid, CameraTag);
  return eid;
}

export function spawnGIProbe(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  GIProbeRef.probeX[eid]       = spec.x !== undefined ? spec.x : 0;
  GIProbeRef.probeY[eid]       = spec.y !== undefined ? spec.y : 3.5;
  GIProbeRef.probeZ[eid]       = spec.z !== undefined ? spec.z : 0;
  GIProbeRef.irradianceR[eid]  = 0.15;
  GIProbeRef.irradianceG[eid]  = 0.18;
  GIProbeRef.irradianceB[eid]  = 0.22;
  GIProbeRef.skyOcclusion[eid] = 1.0;
  GIProbeRef.indoorFactor[eid] = 0.0;
  GIProbeRef.dirty[eid]        = 1;

  addComponent(w, eid, GIProbeRef);
  return eid;
}

export function spawnAOVolume(world, spec = {}) {
  const w = world || getLightWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOVolumeRef.radius[eid]      = spec.radius !== undefined ? spec.radius : 2.0;
  AOVolumeRef.intensity[eid]   = spec.intensity !== undefined ? spec.intensity : 1.0;
  AOVolumeRef.sampleCount[eid] = spec.sampleCount !== undefined ? spec.sampleCount : 8;
  AOVolumeRef.halfRes[eid]     = spec.halfRes !== false ? 1 : 0;
  AOVolumeRef.temporal[eid]    = spec.temporal !== false ? 1 : 0;

  addComponent(w, eid, AOVolumeRef);
  return eid;
}

/* ------------------------------------------------------------------ */
/* 9. UTILITY HELPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Marks a light as dirty so downstream systems re-upload its uniforms.
 */
export function markLightDirty(eid) {
  if (!isValidEntityId(eid)) return false;
  LightState.flags[eid] |= LIGHT_STATE_FLAG.DIRTY;
  LightState.lastDirtyFrame[eid] = LightState.lastUpdateFrame[eid];
  return true;
}

/**
 * Clears the dirty flag after the light has been uploaded.
 */
export function clearLightDirty(eid) {
  if (!isValidEntityId(eid)) return false;
  LightState.flags[eid] &= ~LIGHT_STATE_FLAG.DIRTY;
  return true;
}

/**
 * Activates or deactivates a light.
 */
export function setLightActive(eid, active) {
  if (!isValidEntityId(eid)) return false;
  if (active) LightState.flags[eid] |= LIGHT_STATE_FLAG.ACTIVE;
  else        LightState.flags[eid] &= ~LIGHT_STATE_FLAG.ACTIVE;
  LightState.frameActive[eid] = active ? 1 : 0;
  markLightDirty(eid);
  return true;
}

/**
 * Checks whether a light is active.
 */
export function isLightActive(eid) {
  if (!isValidEntityId(eid)) return false;
  return (LightState.flags[eid] & LIGHT_STATE_FLAG.ACTIVE) !== 0;
}

/**
 * Checks whether a light is visible (in frustum + not culled).
 */
export function isLightVisible(eid) {
  if (!isValidEntityId(eid)) return false;
  return (LightState.flags[eid] & LIGHT_STATE_FLAG.VISIBLE) !== 0;
}

/**
 * Sets a light's type / kind explicitly. Should be called only during
 * spawn or when intentionally switching a light's behavior.
 */
export function setLightType(eid, type, kind) {
  if (!isValidEntityId(eid)) return false;
  if (type < 0 || type >= LIGHT_TYPE.COUNT) return false;
  LightRef.type[eid] = type;
  if (kind !== undefined) LightRef.kind[eid] = kind;
  markLightDirty(eid);
  return true;
}

/**
 * Attaches a behavior to a light's ECS record. The behavior's params are
 * stored in the LightBehavior ctx arrays.
 */
export function attachECSBehavior(eid, behaviorId, ctx0, ctx1, ctx2) {
  if (!isValidEntityId(eid)) return false;
  if (behaviorId < 0 || behaviorId >= LIGHT_BEHAVIOR.COUNT) return false;

  const count = LightBehavior.behaviorCount[eid];
  if (count >= MAX_BEHAVIORS_PER_LIGHT) return false;

  const off = eid * MAX_BEHAVIORS_PER_LIGHT + count;
  LightBehavior.behaviorIds[off] = behaviorId;
  LightBehavior.behaviorCtx0[off] = ctx0 !== undefined ? ctx0 : 0;
  LightBehavior.behaviorCtx1[off] = ctx1 !== undefined ? ctx1 : 0;
  LightBehavior.behaviorCtx2[off] = ctx2 !== undefined ? ctx2 : 0;
  LightBehavior.behaviorCount[eid] = count + 1;

  markLightDirty(eid);
  return true;
}

/**
 * Detaches the behavior at a given slot index.
 */
export function detachECSBehavior(eid, slot) {
  if (!isValidEntityId(eid)) return false;
  const count = LightBehavior.behaviorCount[eid];
  if (slot < 0 || slot >= count) return false;

  // Shift remaining behaviors down.
  for (let b = slot; b < count - 1; b++) {
    const dst = eid * MAX_BEHAVIORS_PER_LIGHT + b;
    const src = eid * MAX_BEHAVIORS_PER_LIGHT + (b + 1);
    LightBehavior.behaviorIds[dst] = LightBehavior.behaviorIds[src];
    LightBehavior.behaviorCtx0[dst] = LightBehavior.behaviorCtx0[src];
    LightBehavior.behaviorCtx1[dst] = LightBehavior.behaviorCtx1[src];
    LightBehavior.behaviorCtx2[dst] = LightBehavior.behaviorCtx2[src];
  }
  const lastOff = eid * MAX_BEHAVIORS_PER_LIGHT + (count - 1);
  LightBehavior.behaviorIds[lastOff] = LIGHT_BEHAVIOR.NONE;
  LightBehavior.behaviorCtx0[lastOff] = 0;
  LightBehavior.behaviorCtx1[lastOff] = 0;
  LightBehavior.behaviorCtx2[lastOff] = 0;
  LightBehavior.behaviorCount[eid] = count - 1;

  markLightDirty(eid);
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. FACTORY — dedicated light world                                */
/* ------------------------------------------------------------------ */

/**
 * Creates a new dedicated ECS world for the lighting stack. Use this if
 * you want a lighting world separate from the main scene world (e.g. for
 * testing, or for a subscene).
 */
export function createLightWorld() {
  return createWorld({
    components: LIGHTING_COMPONENTS,
    time: {
      delta: 0,
      elapsed: 0,
      then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
    },
  });
}

/* ------------------------------------------------------------------ */
/* 11. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_ENTITIES,
  MAX_BEHAVIORS_PER_LIGHT,
  MAX_LIGHTS_BY_TIER,

  // Components
  Transform,
  Target,
  LightRef,
  LightState,
  LightShadow,
  LightCluster,
  LightBudget,
  LightPriority,
  LightBehavior,
  LightComposite,
  LightIndoor,
  LightIES,
  LightEmissive,
  LightFlicker,
  LightDayCycle,
  LightTag,
  CameraTag,
  GIProbeRef,
  AOVolumeRef,
  LIGHTING_COMPONENTS,

  // Enums
  LIGHT_TYPE,
  LIGHT_TYPE_NAME,
  LIGHT_KIND,
  LIGHT_KIND_NAME,
  LIGHT_BEHAVIOR,
  LIGHT_BEHAVIOR_NAME,
  LIGHT_PRIORITY,
  LIGHT_PRIORITY_NAME,
  LIGHT_TAG,
  LIGHT_STATE_FLAG,
  LIGHT_ENVIRONMENT,

  // World
  getLightWorld,
  createLightWorld,

  // Utilities
  isValidEntityId,
  markLightDirty,
  clearLightDirty,
  setLightActive,
  isLightActive,
  isLightVisible,
  setLightType,
  attachECSBehavior,
  detachECSBehavior,

  // Spawn — sanctioned types
  spawnAmbientLight,
  spawnHemisphereLight,
  spawnDirectionalLight,
  spawnPointLight,
  spawnSpotLight,
  spawnRectAreaLight,

  // Spawn — composite kinds
  spawnFireLight,
  spawnNeonLight,
  spawnMagicGlow,
  spawnInteriorLamp,
  spawnWindowShaft,
  spawnCaustic,
  spawnAurora,
  spawnSun,
  spawnMoon,

  // Spawn — support entities
  spawnCamera,
  spawnGIProbe,
  spawnAOVolume,
};

export default _defaultExport;