// File : 004
// name : src/ecs/004_lgt_GIComponents.js
// description : bitECS 0.4.0 SoA component definitions for the global
//               illumination subsystem of the anime lighting stack on Android
//               mobile. Every piece of state the GI pipeline needs — probe
//               grids, spherical harmonics coefficients, bounce paths, voxel
//               occupancy, distance fields, sky occlusion, portals, radiance
//               caches, reflection probes, lightfields, indoor/outdoor
//               volumes, and the anime-specific cel-shaded ambient — is
//               declared here as a plain-object component whose fields are
//               pre-sized TypedArrays.
//
//               This is the bitECS 0.4.0 architecture: NO `defineComponent`,
//               NO `Types` export, NO separate store registry. Components
//               are plain JS objects passed directly to
//               `createWorld({ components })` and attached via
//               `addComponent(world, eid, ComponentObject)`.
//
//               Components declared:
//                 • GIProbeRef          — probe entity → GI grid binding
//                 • GIIrradiance        — 3-channel irradiance per probe
//                 • GISH                — 9-coefficient (L2) spherical harmonics per probe
//                 • GISHHigh            — 16-coefficient (L3) SH for hero probes
//                 • GIBouncePath        — bounce path (source → bounce → receiver)
//                 • GIVoxel             — voxel occupancy + albedo + emission
//                 • GIDistanceField     — SDF sample per voxel
//                 • GIOcclusion         — sky occlusion / obstacle occlusion
//                 • GIPortal            — portal volume (indoor/outdoor handoff)
//                 • GIBudget            — per-probe cost + LOD
//                 • GIState             — per-probe state machine
//                 • GIUpdateQueue       — frame-scoped update queue
//                 • GIRadianceCache     — radiance cache entry per probe
//                 • GIReflectionProbe   — reflection probe capture state
//                 • GILightfield        — lightfield sample (4D position+dir)
//                 • GIVolume            — GI volume bounds (blend region)
//                 • GIIndoor            — indoor GI specifics
//                 • GIOutdoor           — outdoor GI specifics
//                 • GICelBands          — anime cel-shaded ambient banding
//                 • GIPalette           — palette-driven GI (matches 033 composer)
//                 • GIAsync             — async update state (worker in flight)
//                 • GILeak              — leak detection / correction
//                 • GITemporal          — temporal accumulation history
//
//               Also exports:
//                 • MAX_ENTITIES (100000)
//                 • MAX_SH_COEFFICIENTS = 9, MAX_SH_COEFFICIENTS_HIGH = 16
//                 • MAX_BOUNCE_PATHS = 8
//                 • MAX_VOXEL_RESOLUTION = 64
//                 • GI_STATE / GI_QUALITY / GI_MODE / GI_BOUNCE_TYPE /
//                   GI_PORTAL_TYPE / GI_PROBE_TYPE enums
//                 • GI_COMPONENTS bundle for createWorld()
//                 • spawnGIProbeSet / spawnGIVoxelGrid / spawnGIPortalVolume /
//                   spawnGIReflectionProbe / spawnGILightfield /
//                   spawnGICelBandController / spawnGIIndoorVolume /
//                   spawnGIOutdoorVolume / spawnGIRadianceCache factories
//                 • Utility helpers: markGIDirty, clearGIDirty,
//                   setGIQuality, setGIMode, allocateGIPortal,
//                   releaseGIPortal, enqueueGIUpdate, drainGIUpdateQueue
//
//               Strictly Three.js r185 lights only (indirect contribution of
//               AmbientLight / HemisphereLight / DirectionalLight / PointLight
//               / SpotLight / RectAreaLight); strictly bitECS 0.4.0 API only;
//               every typed array sized once to MAX_ENTITIES = 100000.
// best for : Single source of truth for every GI-related ECS component in
//            the anime lighting stack. Every downstream GI system
//            (119_lgt_GIManager.js through 167_lgt_ReSTIRBuffer.js) imports
//            its component definitions from here so probe IDs, SH layouts,
//            and quality enums stay consistent across CPU sampling, GPU
//            uploads, and async worker bakes.
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
  LIGHT_TAG,
} from './002_lgt_LightComponents.js';

import {
  SHADOW_FILTER,
} from './003_lgt_ShadowComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum SH coefficients for L2 (9-coeff) representation.
 */
export const MAX_SH_COEFFICIENTS = 9;

/**
 * Maximum SH coefficients for L3 (16-coeff) hero probe representation.
 */
export const MAX_SH_COEFFICIENTS_HIGH = 16;

/**
 * Maximum simultaneous bounce paths tracked per probe.
 */
export const MAX_BOUNCE_PATHS = 8;

/**
 * Maximum voxel resolution per axis for the voxel GI grid.
 */
export const MAX_VOXEL_RESOLUTION = PERF_TIER_LOCAL === 'HIGH' ? 64 : PERF_TIER_LOCAL === 'MEDIUM' ? 48 : 32;

/**
 * Maximum probe capacity per tier (matches the probe grid configuration
 * declared in 019_rnd_AndroidProfile.js).
 */
export const MAX_GI_PROBES =
  PERF_TIER_LOCAL === 'HIGH'   ? 4096 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2048 :
                                 1024;

/**
 * Maximum lightfield sample count per tier.
 */
export const MAX_LIGHTFIELD_SAMPLES =
  PERF_TIER_LOCAL === 'HIGH'   ? 2048 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 1024 :
                                 512;

/**
 * Maximum reflection probes per tier.
 */
export const MAX_REFLECTION_PROBES =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 16 :
                                 8;

/**
 * Maximum GI portals per tier (indoor ↔ outdoor handoff).
 */
export const MAX_GI_PORTALS =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 16 :
                                 8;

/**
 * Maximum radiance cache entries per tier.
 */
export const MAX_RADIANCE_CACHE =
  PERF_TIER_LOCAL === 'HIGH'   ? 4096 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2048 :
                                 1024;

/**
 * Maximum simultaneous async GI updates per tier.
 */
export const MAX_ASYNC_UPDATES =
  PERF_TIER_LOCAL === 'HIGH'   ? 8 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 4 :
                                 2;

/**
 * Maximum frame-scoped GI update queue length.
 */
export const MAX_GI_UPDATE_QUEUE =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 256 :
                                 128;

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * Per-probe / per-volume GI state machine.
 */
export const GI_STATE = Object.freeze({
  IDLE:       0,
  DIRTY:      1,
  BAKING:     2,
  READY:      3,
  STALE:      4,
  FAILED:     5,
  DISABLED:   6,
  COUNT:      7,
});

export const GI_STATE_NAME = Object.freeze([
  'idle',
  'dirty',
  'baking',
  'ready',
  'stale',
  'failed',
  'disabled',
]);

/**
 * GI quality tiers (per-probe).
 */
export const GI_QUALITY = Object.freeze({
  OFF:       0,
  MINIMAL:   1,
  LOW:       2,
  MEDIUM:    3,
  HIGH:      4,
  ULTRA:     5,
  COUNT:     6,
});

export const GI_QUALITY_NAME = Object.freeze([
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'ultra',
]);

/**
 * GI evaluation modes — how indirect light is computed.
 */
export const GI_MODE = Object.freeze({
  NONE:             0,   // disabled
  AMBIENT_FLAT:     1,   // single ambient color
  HEMISPHERE_FLAT:  2,   // hemisphere sky/ground
  IRRADIANCE_VOLUME:3,   // probe grid
  SH_L1:            4,   // 4-coeff spherical harmonics
  SH_L2:            5,   // 9-coeff spherical harmonics
  SH_L3:            6,   // 16-coeff spherical harmonics
  VOXEL_CONE:       7,   // voxel cone tracing
  SCREEN_SPACE:     8,   // SSR / SSGI
  RADIANCE_CACHE:   9,   // world-space radiance cache
  LIGHTFIELD:      10,   // 4D lightfield
  REFLECTION_PROBE:11,   // cube-map reflection probes
  RESTIR:          12,   // ReSTIR GI
  HYBRID:          13,   // auto-selected combination
  COUNT:           14,
});

export const GI_MODE_NAME = Object.freeze([
  'none',
  'ambient_flat',
  'hemisphere_flat',
  'irradiance_volume',
  'sh_l1',
  'sh_l2',
  'sh_l3',
  'voxel_cone',
  'screen_space',
  'radiance_cache',
  'lightfield',
  'reflection_probe',
  'restir',
  'hybrid',
]);

/**
 * Bounce types — how indirect light bounces.
 */
export const GI_BOUNCE_TYPE = Object.freeze({
  DIFFUSE:     0,
  SPECULAR:    1,
  GLOSSY:      2,
  SUBSURFACE:  3,
  TRANSMISSION:4,
  EMISSION:    5,
  COUNT:       6,
});

export const GI_BOUNCE_TYPE_NAME = Object.freeze([
  'diffuse',
  'specular',
  'glossy',
  'subsurface',
  'transmission',
  'emission',
]);

/**
 * Portal types — indoor/outdoor light transport handoff.
 */
export const GI_PORTAL_TYPE = Object.freeze({
  NONE:      0,
  DOORWAY:   1,
  WINDOW:    2,
  SKYLIGHT:  3,
  CAVE_MOUTH:4,
  CANOPY:    5,
  MIRROR:    6,
  COUNT:     7,
});

export const GI_PORTAL_TYPE_NAME = Object.freeze([
  'none',
  'doorway',
  'window',
  'skylight',
  'cave_mouth',
  'canopy',
  'mirror',
]);

/**
 * GI probe types — where the probe draws its contribution.
 */
export const GI_PROBE_TYPE = Object.freeze({
  OUTDOOR:    0,
  INDOOR:     1,
  TRANSITION: 2,
  HERO:       3,   // high-resolution probe
  BULK:       4,   // low-resolution grid
  REFLECTION: 5,
  COUNT:      6,
});

export const GI_PROBE_TYPE_NAME = Object.freeze([
  'outdoor',
  'indoor',
  'transition',
  'hero',
  'bulk',
  'reflection',
]);

/**
 * GI bounce path kinds — used by the path tracer.
 */
export const GI_PATH_KIND = Object.freeze({
  DIRECT:     0,
  ONE_BOUNCE: 1,
  TWO_BOUNCE: 2,
  MULTI:      3,
  AMBIENT:    4,
  EMISSIVE:   5,
  COUNT:      6,
});

export const GI_PATH_KIND_NAME = Object.freeze([
  'direct',
  'one_bounce',
  'two_bounce',
  'multi',
  'ambient',
  'emissive',
]);

/**
 * Leak correction modes.
 */
export const GI_LEAK_MODE = Object.freeze({
  NONE:        0,
  CLAMP:       1,
  SMOOTH:      2,
  RAY_REJECT:  3,
  PORTAL_GATE: 4,
  COUNT:       5,
});

export const GI_LEAK_MODE_NAME = Object.freeze([
  'none',
  'clamp',
  'smooth',
  'ray_reject',
  'portal_gate',
]);

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * GIProbeRef — links a probe entity to its grid position and metadata.
 */
export const GIProbeRef = {
  type:           new Uint8Array(MAX_ENTITIES),      // GI_PROBE_TYPE
  quality:        new Uint8Array(MAX_ENTITIES),      // GI_QUALITY
  mode:           new Uint8Array(MAX_ENTITIES),      // GI_MODE
  gridX:          new Int16Array(MAX_ENTITIES),
  gridY:          new Int16Array(MAX_ENTITIES),
  gridZ:          new Int16Array(MAX_ENTITIES),
  worldX:         new Float32Array(MAX_ENTITIES),
  worldY:         new Float32Array(MAX_ENTITIES),
  worldZ:         new Float32Array(MAX_ENTITIES),
  radius:         new Float32Array(MAX_ENTITIES),    // influence radius
  gridCellId:     new Int32Array(MAX_ENTITIES),      // -1 if unbound
  enabled:        new Uint8Array(MAX_ENTITIES),
  indoorFactor:   new Float32Array(MAX_ENTITIES),    // [0,1]
};

/**
 * GIIrradiance — 3-channel irradiance per probe.
 */
export const GIIrradiance = {
  r:           new Float32Array(MAX_ENTITIES),
  g:           new Float32Array(MAX_ENTITIES),
  b:           new Float32Array(MAX_ENTITIES),
  // Previous frame values for temporal blending.
  rPrev:       new Float32Array(MAX_ENTITIES),
  gPrev:       new Float32Array(MAX_ENTITIES),
  bPrev:       new Float32Array(MAX_ENTITIES),
  // Confidence in [0,1] — temporal accumulator weight.
  confidence:  new Float32Array(MAX_ENTITIES),
  // Luminance (cached for cheap sort / leak checks).
  luminance:   new Float32Array(MAX_ENTITIES),
};

/**
 * GISH — 9-coefficient L2 spherical harmonics per probe.
 * Stored as a flat array: sh[eid * 9 + coeffIndex].
 */
export const GISH = {
  coefficients: new Float32Array(MAX_ENTITIES * MAX_SH_COEFFICIENTS),
  coefficientsPrev: new Float32Array(MAX_ENTITIES * MAX_SH_COEFFICIENTS),
  valid:        new Uint8Array(MAX_ENTITIES),
  // Directional intensity for the top 3 lobes (fast lighting).
  lobe0R:       new Float32Array(MAX_ENTITIES),
  lobe0G:       new Float32Array(MAX_ENTITIES),
  lobe0B:       new Float32Array(MAX_ENTITIES),
  lobe1R:       new Float32Array(MAX_ENTITIES),
  lobe1G:       new Float32Array(MAX_ENTITIES),
  lobe1B:       new Float32Array(MAX_ENTITIES),
  lobe2R:       new Float32Array(MAX_ENTITIES),
  lobe2G:       new Float32Array(MAX_ENTITIES),
  lobe2B:       new Float32Array(MAX_ENTITIES),
};

/**
 * GISHHigh — 16-coefficient L3 spherical harmonics for hero probes.
 */
export const GISHHigh = {
  coefficients: new Float32Array(MAX_ENTITIES * MAX_SH_COEFFICIENTS_HIGH),
  valid:        new Uint8Array(MAX_ENTITIES),
  enabled:      new Uint8Array(MAX_ENTITIES),
};

/**
 * GIBouncePath — per-probe bounce path record.
 * Stored as a flat array: paths[eid * MAX_BOUNCE_PATHS + slot].
 */
export const GIBouncePath = {
  kind:         new Uint8Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),   // GI_PATH_KIND
  bounceType:   new Uint8Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),   // GI_BOUNCE_TYPE
  // Source direction (normalized).
  dirX:         new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  dirY:         new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  dirZ:         new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  // Contributed radiance.
  radianceR:    new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  radianceG:    new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  radianceB:    new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  distance:     new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  weight:       new Float32Array(MAX_ENTITIES * MAX_BOUNCE_PATHS),
  pathCount:    new Uint8Array(MAX_ENTITIES),
};

/**
 * GIVoxel — voxel grid occupancy + surface properties.
 * Voxel indices: voxelIdx = z * MAX_VOXEL_RESOLUTION² + y * MAX_VOXEL_RESOLUTION + x.
 */
export const GIVoxel = {
  occupancy:    new Uint8Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  albedoR:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  albedoG:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  albedoB:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  emissionR:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  emissionG:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  emissionB:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  normalX:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  normalY:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  normalZ:      new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  // Grid metadata (stored once, on entity 0).
  gridMinX:     new Float32Array(MAX_ENTITIES),
  gridMinY:     new Float32Array(MAX_ENTITIES),
  gridMinZ:     new Float32Array(MAX_ENTITIES),
  gridMaxX:     new Float32Array(MAX_ENTITIES),
  gridMaxY:     new Float32Array(MAX_ENTITIES),
  gridMaxZ:     new Float32Array(MAX_ENTITIES),
  resolution:   new Uint8Array(MAX_ENTITIES),
  generation:   new Uint32Array(MAX_ENTITIES),
  valid:        new Uint8Array(MAX_ENTITIES),
};

/**
 * GIDistanceField — signed distance field samples per voxel.
 */
export const GIDistanceField = {
  distance:     new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  gradientX:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  gradientY:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  gradientZ:    new Float32Array(MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION),
  valid:        new Uint8Array(MAX_ENTITIES),
  generation:   new Uint32Array(MAX_ENTITIES),
};

/**
 * GIOcclusion — sky and obstacle occlusion per probe.
 */
export const GIOcclusion = {
  skyOcclusion:      new Float32Array(MAX_ENTITIES),
  obstacleOcclusion: new Float32Array(MAX_ENTITIES),
  portalOcclusion:   new Float32Array(MAX_ENTITIES),
  sampleCount:       new Uint8Array(MAX_ENTITIES),
  lastUpdateFrame:   new Uint32Array(MAX_ENTITIES),
  valid:             new Uint8Array(MAX_ENTITIES),
};

/**
 * GIPortal — indoor/outdoor light-transport portals.
 */
export const GIPortal = {
  type:            new Uint8Array(MAX_ENTITIES),  // GI_PORTAL_TYPE
  enabled:         new Uint8Array(MAX_ENTITIES),
  positionX:       new Float32Array(MAX_ENTITIES),
  positionY:       new Float32Array(MAX_ENTITIES),
  positionZ:       new Float32Array(MAX_ENTITIES),
  normalX:         new Float32Array(MAX_ENTITIES),
  normalY:         new Float32Array(MAX_ENTITIES),
  normalZ:         new Float32Array(MAX_ENTITIES),
  width:           new Float32Array(MAX_ENTITIES),
  height:          new Float32Array(MAX_ENTITIES),
  // Room / volume references (entity ids).
  indoorRoomEid:   new Int32Array(MAX_ENTITIES),
  outdoorVolumeEid:new Int32Array(MAX_ENTITIES),
  // Transport weights.
  transmission:    new Float32Array(MAX_ENTITIES),
  tintR:           new Float32Array(MAX_ENTITIES),
  tintG:           new Float32Array(MAX_ENTITIES),
  tintB:           new Float32Array(MAX_ENTITIES),
  // Current throughput (updated per frame).
  fluxR:           new Float32Array(MAX_ENTITIES),
  fluxG:           new Float32Array(MAX_ENTITIES),
  fluxB:           new Float32Array(MAX_ENTITIES),
  visible:         new Uint8Array(MAX_ENTITIES),
};

/**
 * GIBudget — per-probe cost + LOD + async state.
 */
export const GIBudget = {
  cost:            new Float32Array(MAX_ENTITIES),
  costEma:         new Float32Array(MAX_ENTITIES),
  lastCostMs:      new Float32Array(MAX_ENTITIES),
  lod:             new Uint8Array(MAX_ENTITIES),      // 0=full 1=half 2=quarter 3=off
  lodTarget:       new Uint8Array(MAX_ENTITIES),
  lastLodFrame:    new Uint32Array(MAX_ENTITIES),
  priority:        new Uint8Array(MAX_ENTITIES),
  // Async state (worker in flight).
  asyncPending:    new Uint8Array(MAX_ENTITIES),
  asyncWorkerId:   new Int16Array(MAX_ENTITIES),
  asyncStartFrame: new Uint32Array(MAX_ENTITIES),
  asyncTimeout:    new Uint16Array(MAX_ENTITIES),
};

/**
 * GIState — per-probe / per-volume state machine.
 */
export const GIState = {
  state:           new Uint8Array(MAX_ENTITIES),      // GI_STATE
  prevState:       new Uint8Array(MAX_ENTITIES),
  lastStateFrame:  new Uint32Array(MAX_ENTITIES),
  lastUpdateFrame: new Uint32Array(MAX_ENTITIES),
  lastBakeFrame:   new Uint32Array(MAX_ENTITIES),
  failureCount:    new Uint8Array(MAX_ENTITIES),
  lastError:       new Int32Array(MAX_ENTITIES),      // -1 = none
  // Leak flag — set when the leak detector fires.
  leakFlag:        new Uint8Array(MAX_ENTITIES),
};

/**
 * GIUpdateQueue — frame-scoped FIFO of probes needing update.
 */
export const GIUpdateQueue = {
  queue:           new Int32Array(MAX_GI_UPDATE_QUEUE),
  head:            new Uint16Array(1),
  tail:            new Uint16Array(1),
  count:           new Uint16Array(1),
  droppedCount:    new Uint32Array(1),
  totalEnqueued:   new Uint32Array(1),
  totalProcessed:  new Uint32Array(1),
  queueValid:      new Uint8Array(1),
  // Frame counter for starvation detection.
  currentFrame:    new Uint32Array(1),
};

/**
 * GIRadianceCache — world-space radiance cache.
 * Stored as a ring buffer of entries with a spatial hash index.
 */
export const GIRadianceCache = {
  positionX:    new Float32Array(MAX_RADIANCE_CACHE),
  positionY:    new Float32Array(MAX_RADIANCE_CACHE),
  positionZ:    new Float32Array(MAX_RADIANCE_CACHE),
  normalX:      new Float32Array(MAX_RADIANCE_CACHE),
  normalY:      new Float32Array(MAX_RADIANCE_CACHE),
  normalZ:      new Float32Array(MAX_RADIANCE_CACHE),
  radianceR:    new Float32Array(MAX_RADIANCE_CACHE),
  radianceG:    new Float32Array(MAX_RADIANCE_CACHE),
  radianceB:    new Float32Array(MAX_RADIANCE_CACHE),
  age:          new Uint16Array(MAX_RADIANCE_CACHE),
  sampleCount:  new Uint16Array(MAX_RADIANCE_CACHE),
  confidence:   new Float32Array(MAX_RADIANCE_CACHE),
  valid:        new Uint8Array(MAX_RADIANCE_CACHE),
  // Spatial hash buckets.
  hashTable:    new Int32Array(2048),
  hashNext:     new Int32Array(MAX_RADIANCE_CACHE),
  generation:   new Uint32Array(1),
  entryCount:   new Uint32Array(1),
  writeCursor:  new Uint32Array(1),
};

/**
 * GIReflectionProbe — cube-map reflection probe capture state.
 */
export const GIReflectionProbe = {
  positionX:     new Float32Array(MAX_ENTITIES),
  positionY:     new Float32Array(MAX_ENTITIES),
  positionZ:     new Float32Array(MAX_ENTITIES),
  radius:        new Float32Array(MAX_ENTITIES),
  resolution:    new Uint16Array(MAX_ENTITIES),
  captureFace:   new Uint8Array(MAX_ENTITIES),   // 0..5, or 6 if complete
  captureFaceFrame: new Uint32Array(MAX_ENTITIES),
  lastCaptureFrame: new Uint32Array(MAX_ENTITIES),
  updateInterval: new Uint16Array(MAX_ENTITIES),
  hdrBias:       new Float32Array(MAX_ENTITIES),
  parallaxCorrect: new Uint8Array(MAX_ENTITIES),
  // Box parallax (OBB corners).
  boxMinX:       new Float32Array(MAX_ENTITIES),
  boxMinY:       new Float32Array(MAX_ENTITIES),
  boxMinZ:       new Float32Array(MAX_ENTITIES),
  boxMaxX:       new Float32Array(MAX_ENTITIES),
  boxMaxY:       new Float32Array(MAX_ENTITIES),
  boxMaxZ:       new Float32Array(MAX_ENTITIES),
  valid:         new Uint8Array(MAX_ENTITIES),
  enabled:       new Uint8Array(MAX_ENTITIES),
};

/**
 * GILightfield — 4D lightfield sample (position + direction).
 */
export const GILightfield = {
  positionX:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  positionY:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  positionZ:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  dirX:          new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  dirY:          new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  dirZ:          new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  radianceR:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  radianceG:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  radianceB:     new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  depth:         new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  confidence:    new Float32Array(MAX_LIGHTFIELD_SAMPLES),
  valid:         new Uint8Array(MAX_LIGHTFIELD_SAMPLES),
  sampleCount:   new Uint32Array(1),
  capacity:      new Uint32Array(1),
  // Per-entity binding (which entity owns this lightfield).
  ownerEid:      new Int32Array(1),
  generation:    new Uint32Array(1),
};

/**
 * GIVolume — GI volume bounds for blended regional GI.
 * A volume may be indoor (a room) or outdoor (a biome chunk).
 */
export const GIVolume = {
  kind:           new Uint8Array(MAX_ENTITIES),    // 0=indoor 1=outdoor 2=transition
  enabled:        new Uint8Array(MAX_ENTITIES),
  // Bounds.
  minX:           new Float32Array(MAX_ENTITIES),
  minY:           new Float32Array(MAX_ENTITIES),
  minZ:           new Float32Array(MAX_ENTITIES),
  maxX:           new Float32Array(MAX_ENTITIES),
  maxY:           new Float32Array(MAX_ENTITIES),
  maxZ:           new Float32Array(MAX_ENTITIES),
  // Interior fill color (fallback when no probe is inside).
  fillR:          new Float32Array(MAX_ENTITIES),
  fillG:          new Float32Array(MAX_ENTITIES),
  fillB:          new Float32Array(MAX_ENTITIES),
  // Blend region (soft transition with neighbors).
  blendDistance:  new Float32Array(MAX_ENTITIES),
  // Spatial hash cell references.
  cellCount:      new Uint16Array(MAX_ENTITIES),
  // Probe density factor (higher density inside rooms).
  probeDensity:   new Float32Array(MAX_ENTITIES),
  // Ambient "leak gate" — 0 lets no outdoor light in (fully sealed room).
  leakGate:       new Float32Array(MAX_ENTITIES),
  generation:     new Uint32Array(MAX_ENTITIES),
};

/**
 * GIIndoor — indoor-specific GI state.
 */
export const GIIndoor = {
  volumeEid:         new Int32Array(MAX_ENTITIES),
  portalEidList:     new Int32Array(MAX_ENTITIES * 4),
  portalCount:       new Uint8Array(MAX_ENTITIES),
  // Interior-specific ambient parameters.
  ceilingBounceR:    new Float32Array(MAX_ENTITIES),
  ceilingBounceG:    new Float32Array(MAX_ENTITIES),
  ceilingBounceB:    new Float32Array(MAX_ENTITIES),
  floorBounceR:      new Float32Array(MAX_ENTITIES),
  floorBounceG:      new Float32Array(MAX_ENTITIES),
  floorBounceB:      new Float32Array(MAX_ENTITIES),
  // Fireplace, lamp, chandelier contributions get folded here.
  emissiveFillR:     new Float32Array(MAX_ENTITIES),
  emissiveFillG:     new Float32Array(MAX_ENTITIES),
  emissiveFillB:     new Float32Array(MAX_ENTITIES),
  // Wall occlusion — how much outdoor light bleeds in.
  wallOcclusion:     new Float32Array(MAX_ENTITIES),
  // Curtain filter (image 5 — mossy wall with warm light).
  curtainTransmission:new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * GIOutdoor — outdoor-specific GI state.
 */
export const GIOutdoor = {
  volumeEid:         new Int32Array(MAX_ENTITIES),
  // Sky irradiance ramp.
  skyZenithR:        new Float32Array(MAX_ENTITIES),
  skyZenithG:        new Float32Array(MAX_ENTITIES),
  skyZenithB:        new Float32Array(MAX_ENTITIES),
  skyHorizonR:       new Float32Array(MAX_ENTITIES),
  skyHorizonG:       new Float32Array(MAX_ENTITIES),
  skyHorizonB:       new Float32Array(MAX_ENTITIES),
  // Ground bounce.
  groundAlbedoR:     new Float32Array(MAX_ENTITIES),
  groundAlbedoG:     new Float32Array(MAX_ENTITIES),
  groundAlbedoB:     new Float32Array(MAX_ENTITIES),
  // Fog / haze contribution.
  hazeR:             new Float32Array(MAX_ENTITIES),
  hazeG:             new Float32Array(MAX_ENTITIES),
  hazeB:             new Float32Array(MAX_ENTITIES),
  // Sun contribution (broadcast from active sun).
  sunDirR:           new Float32Array(MAX_ENTITIES),
  sunDirG:           new Float32Array(MAX_ENTITIES),
  sunDirB:           new Float32Array(MAX_ENTITIES),
  // Biome weight (matches 033 composer).
  biomeWeight:       new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * GICelBands — anime cel-shaded ambient banding.
 * GI is quantized to cel bands to match the reference image style.
 */
export const GICelBands = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  bandCount:         new Uint8Array(MAX_ENTITIES),   // 2..8
  bandSoftness:      new Float32Array(MAX_ENTITIES), // 0 = hard, 1 = smooth
  bandBias:          new Float32Array(MAX_ENTITIES),
  // Per-band palette (matches 033 composer bands).
  band0R:            new Float32Array(MAX_ENTITIES),
  band0G:            new Float32Array(MAX_ENTITIES),
  band0B:            new Float32Array(MAX_ENTITIES),
  band1R:            new Float32Array(MAX_ENTITIES),
  band1G:            new Float32Array(MAX_ENTITIES),
  band1B:            new Float32Array(MAX_ENTITIES),
  band2R:            new Float32Array(MAX_ENTITIES),
  band2G:            new Float32Array(MAX_ENTITIES),
  band2B:            new Float32Array(MAX_ENTITIES),
  band3R:            new Float32Array(MAX_ENTITIES),
  band3G:            new Float32Array(MAX_ENTITIES),
  band3B:            new Float32Array(MAX_ENTITIES),
  // Post-band dither (matches the anime dither chunk).
  ditherStrength:    new Float32Array(MAX_ENTITIES),
  ditherScale:       new Float32Array(MAX_ENTITIES),
};

/**
 * GIPalette — palette-driven GI. Indirect light is composed through the
 * same palette machinery as 033_rnd_ProceduralColorComposer.js.
 */
export const GIPalette = {
  styleId:           new Uint8Array(MAX_ENTITIES),   // 0..N — STYLE_ID index
  satBias:           new Float32Array(MAX_ENTITIES),
  hueBias:           new Float32Array(MAX_ENTITIES),
  ambientColorR:     new Float32Array(MAX_ENTITIES),
  ambientColorG:     new Float32Array(MAX_ENTITIES),
  ambientColorB:     new Float32Array(MAX_ENTITIES),
  shadowTintR:       new Float32Array(MAX_ENTITIES),
  shadowTintG:       new Float32Array(MAX_ENTITIES),
  shadowTintB:       new Float32Array(MAX_ENTITIES),
  bounceWarmth:      new Float32Array(MAX_ENTITIES),
  bounceCoolness:    new Float32Array(MAX_ENTITIES),
  lerpRate:          new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * GIAsync — async update state (worker bake in flight).
 */
export const GIAsync = {
  pendingBatches:     new Uint32Array(1),
  completedBatches:   new Uint32Array(1),
  failedBatches:      new Uint32Array(1),
  cancelledBatches:   new Uint32Array(1),
  workerCount:        new Uint8Array(1),
  workerBusy:         new Uint8Array(8),
  workerLastJobMs:    new Float32Array(8),
  workerTotalMs:      new Float32Array(8),
  workerJobsDone:     new Uint32Array(8),
  totalBakeTimeMs:    new Float32Array(1),
  lastBakeStartFrame: new Uint32Array(1),
  lastBakeEndFrame:   new Uint32Array(1),
};

/**
 * GILeak — leak detection / correction state per probe.
 */
export const GILeak = {
  mode:              new Uint8Array(MAX_ENTITIES),   // GI_LEAK_MODE
  leakMagnitude:     new Float32Array(MAX_ENTITIES), // [0,1]
  threshold:         new Float32Array(MAX_ENTITIES),
  correctionFactor:  new Float32Array(MAX_ENTITIES),
  portalGateStrength:new Float32Array(MAX_ENTITIES),
  detectionCount:    new Uint32Array(MAX_ENTITIES),
  lastDetectionFrame:new Uint32Array(MAX_ENTITIES),
  valid:             new Uint8Array(MAX_ENTITIES),
};

/**
 * GITemporal — temporal accumulation state per probe.
 */
export const GITemporal = {
  historyLength:      new Uint8Array(MAX_ENTITIES),  // frames of history
  historyWeight:      new Float32Array(MAX_ENTITIES),
  reprojectionBias:   new Float32Array(MAX_ENTITIES),
  rejectionThreshold: new Float32Array(MAX_ENTITIES),
  lastHistoryFrame:   new Uint32Array(MAX_ENTITIES),
  // Reprojection offsets (screen-space).
  reprojX:            new Float32Array(MAX_ENTITIES),
  reprojY:            new Float32Array(MAX_ENTITIES),
  // Variance estimate.
  variance:           new Float32Array(MAX_ENTITIES),
  valid:              new Uint8Array(MAX_ENTITIES),
};

/**
 * GIVolumeBlend — regional blend weights for GI volumes overlapping.
 */
export const GIVolumeBlend = {
  primaryVolumeEid:   new Int32Array(MAX_ENTITIES),
  secondaryVolumeEid: new Int32Array(MAX_ENTITIES),
  blendWeight:        new Float32Array(MAX_ENTITIES),
  blendDirty:         new Uint8Array(MAX_ENTITIES),
};

/* ------------------------------------------------------------------ */
/* 3. ECS WORLD COMPONENT BUNDLE                                      */
/* ------------------------------------------------------------------ */

export const GI_COMPONENTS = Object.freeze({
  GIProbeRef,
  GIIrradiance,
  GISH,
  GISHHigh,
  GIBouncePath,
  GIVoxel,
  GIDistanceField,
  GIOcclusion,
  GIPortal,
  GIBudget,
  GIState,
  GIUpdateQueue,
  GIRadianceCache,
  GIReflectionProbe,
  GILightfield,
  GIVolume,
  GIIndoor,
  GIOutdoor,
  GICelBands,
  GIPalette,
  GIAsync,
  GILeak,
  GITemporal,
  GIVolumeBlend,
});

/* ------------------------------------------------------------------ */
/* 4. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

let _giWorld = null;

export function getGIWorld() {
  if (_giWorld) return _giWorld;
  try {
    _giWorld = createWorld({
      components: GI_COMPONENTS,
      time: {
        delta: 0,
        elapsed: 0,
        then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      },
    });
  } catch (e) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.GI, `[004_lgt_GIComponents] failed to create GI world: ${e && e.message}`);
    _giWorld = null;
  }
  return _giWorld;
}

/* ------------------------------------------------------------------ */
/* 5. SPAWN HELPERS                                                   */
/* ------------------------------------------------------------------ */

function _attachGIBase(world, eid, spec) {
  // GIProbeRef
  GIProbeRef.type[eid]        = spec.type !== undefined ? spec.type : GI_PROBE_TYPE.OUTDOOR;
  GIProbeRef.quality[eid]     = spec.quality !== undefined ? spec.quality : GI_QUALITY.MEDIUM;
  GIProbeRef.mode[eid]        = spec.mode !== undefined ? spec.mode : GI_MODE.SH_L2;
  GIProbeRef.gridX[eid]       = spec.gridX !== undefined ? spec.gridX : 0;
  GIProbeRef.gridY[eid]       = spec.gridY !== undefined ? spec.gridY : 0;
  GIProbeRef.gridZ[eid]       = spec.gridZ !== undefined ? spec.gridZ : 0;
  GIProbeRef.worldX[eid]      = spec.worldX !== undefined ? spec.worldX : 0;
  GIProbeRef.worldY[eid]      = spec.worldY !== undefined ? spec.worldY : 3.5;
  GIProbeRef.worldZ[eid]      = spec.worldZ !== undefined ? spec.worldZ : 0;
  GIProbeRef.radius[eid]      = spec.radius !== undefined ? spec.radius : 2.0;
  GIProbeRef.gridCellId[eid]  = -1;
  GIProbeRef.enabled[eid]     = spec.enabled !== false ? 1 : 0;
  GIProbeRef.indoorFactor[eid]= spec.indoorFactor !== undefined ? spec.indoorFactor : 0.0;

  // GIIrradiance
  GIIrradiance.r[eid] = 0.15;
  GIIrradiance.g[eid] = 0.18;
  GIIrradiance.b[eid] = 0.22;
  GIIrradiance.rPrev[eid] = 0.15;
  GIIrradiance.gPrev[eid] = 0.18;
  GIIrradiance.bPrev[eid] = 0.22;
  GIIrradiance.confidence[eid] = 0.0;
  GIIrradiance.luminance[eid] = 0.18;

  // GISH — 9 coefficients
  for (let i = 0; i < MAX_SH_COEFFICIENTS; i++) {
    GISH.coefficients[eid * MAX_SH_COEFFICIENTS + i] = 0;
    GISH.coefficientsPrev[eid * MAX_SH_COEFFICIENTS + i] = 0;
  }
  GISH.valid[eid] = 0;
  GISH.lobe0R[eid] = 0.20;
  GISH.lobe0G[eid] = 0.22;
  GISH.lobe0B[eid] = 0.26;
  GISH.lobe1R[eid] = 0.10;
  GISH.lobe1G[eid] = 0.12;
  GISH.lobe1B[eid] = 0.15;
  GISH.lobe2R[eid] = 0.05;
  GISH.lobe2G[eid] = 0.06;
  GISH.lobe2B[eid] = 0.08;

  // GISHHigh
  for (let i = 0; i < MAX_SH_COEFFICIENTS_HIGH; i++) {
    GISHHigh.coefficients[eid * MAX_SH_COEFFICIENTS_HIGH + i] = 0;
  }
  GISHHigh.valid[eid] = 0;
  GISHHigh.enabled[eid] = spec.heroProbe ? 1 : 0;

  // GIBouncePath — reset all slots
  for (let i = 0; i < MAX_BOUNCE_PATHS; i++) {
    const off = eid * MAX_BOUNCE_PATHS + i;
    GIBouncePath.kind[off] = GI_PATH_KIND.DIRECT;
    GIBouncePath.bounceType[off] = GI_BOUNCE_TYPE.DIFFUSE;
    GIBouncePath.dirX[off] = 0;
    GIBouncePath.dirY[off] = 1;
    GIBouncePath.dirZ[off] = 0;
    GIBouncePath.radianceR[off] = 0;
    GIBouncePath.radianceG[off] = 0;
    GIBouncePath.radianceB[off] = 0;
    GIBouncePath.distance[off] = 0;
    GIBouncePath.weight[off] = 0;
  }
  GIBouncePath.pathCount[eid] = 0;

  // GIOcclusion
  GIOcclusion.skyOcclusion[eid] = 1.0;
  GIOcclusion.obstacleOcclusion[eid] = 0.0;
  GIOcclusion.portalOcclusion[eid] = 0.0;
  GIOcclusion.sampleCount[eid] = 0;
  GIOcclusion.lastUpdateFrame[eid] = 0;
  GIOcclusion.valid[eid] = 0;

  // GIBudget
  GIBudget.cost[eid] = 1.0;
  GIBudget.costEma[eid] = 1.0;
  GIBudget.lastCostMs[eid] = 0;
  GIBudget.lod[eid] = 0;
  GIBudget.lodTarget[eid] = 0;
  GIBudget.lastLodFrame[eid] = 0;
  GIBudget.priority[eid] = 100;
  GIBudget.asyncPending[eid] = 0;
  GIBudget.asyncWorkerId[eid] = -1;
  GIBudget.asyncStartFrame[eid] = 0;
  GIBudget.asyncTimeout[eid] = 0;

  // GIState
  GIState.state[eid] = GI_STATE.IDLE;
  GIState.prevState[eid] = GI_STATE.IDLE;
  GIState.lastStateFrame[eid] = 0;
  GIState.lastUpdateFrame[eid] = 0;
  GIState.lastBakeFrame[eid] = 0;
  GIState.failureCount[eid] = 0;
  GIState.lastError[eid] = -1;
  GIState.leakFlag[eid] = 0;

  // GICelBands — anime cel ambient
  GICelBands.enabled[eid] = 0;
  GICelBands.bandCount[eid] = 4;
  GICelBands.bandSoftness[eid] = 0.10;
  GICelBands.bandBias[eid] = 0.0;
  GICelBands.ditherStrength[eid] = 1.0 / 255.0;
  GICelBands.ditherScale[eid] = 1.0;

  // GIPalette
  GIPalette.styleId[eid] = 0;
  GIPalette.satBias[eid] = 1.0;
  GIPalette.hueBias[eid] = 0.0;
  GIPalette.ambientColorR[eid] = 0.45;
  GIPalette.ambientColorG[eid] = 0.55;
  GIPalette.ambientColorB[eid] = 0.70;
  GIPalette.shadowTintR[eid] = 0.15;
  GIPalette.shadowTintG[eid] = 0.20;
  GIPalette.shadowTintB[eid] = 0.35;
  GIPalette.bounceWarmth[eid] = 0.15;
  GIPalette.bounceCoolness[eid] = 0.10;
  GIPalette.lerpRate[eid] = 3.2;
  GIPalette.enabled[eid] = 0;

  // GILeak
  GILeak.mode[eid] = GI_LEAK_MODE.SMOOTH;
  GILeak.leakMagnitude[eid] = 0.0;
  GILeak.threshold[eid] = 0.35;
  GILeak.correctionFactor[eid] = 1.0;
  GILeak.portalGateStrength[eid] = 0.85;
  GILeak.detectionCount[eid] = 0;
  GILeak.lastDetectionFrame[eid] = 0;
  GILeak.valid[eid] = 0;

  // GITemporal
  GITemporal.historyLength[eid] = 0;
  GITemporal.historyWeight[eid] = 0.90;
  GITemporal.reprojectionBias[eid] = 0.02;
  GITemporal.rejectionThreshold[eid] = 0.15;
  GITemporal.lastHistoryFrame[eid] = 0;
  GITemporal.reprojX[eid] = 0;
  GITemporal.reprojY[eid] = 0;
  GITemporal.variance[eid] = 0;
  GITemporal.valid[eid] = 0;

  // GIVolumeBlend
  GIVolumeBlend.primaryVolumeEid[eid] = -1;
  GIVolumeBlend.secondaryVolumeEid[eid] = -1;
  GIVolumeBlend.blendWeight[eid] = 1.0;
  GIVolumeBlend.blendDirty[eid] = 0;

  // Attach all components.
  addComponent(world, eid, GIProbeRef);
  addComponent(world, eid, GIIrradiance);
  addComponent(world, eid, GISH);
  addComponent(world, eid, GISHHigh);
  addComponent(world, eid, GIBouncePath);
  addComponent(world, eid, GIOcclusion);
  addComponent(world, eid, GIBudget);
  addComponent(world, eid, GIState);
  addComponent(world, eid, GICelBands);
  addComponent(world, eid, GIPalette);
  addComponent(world, eid, GILeak);
  addComponent(world, eid, GITemporal);
  addComponent(world, eid, GIVolumeBlend);
}

/* ------------------------------------------------------------------ */
/* 6. PUBLIC SPAWN FUNCTIONS                                          */
/* ------------------------------------------------------------------ */

/**
 * Spawns a single GI probe entity.
 */
export function spawnGIProbe(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachGIBase(w, eid, spec);
  return eid;
}

/**
 * Spawns a GI probe grid. Returns an array of entity ids (or an object
 * with the ids + the anchor entity that owns the grid metadata).
 *
 *   const grid = spawnGIProbeSet(world, {
 *     resolution: 16,
 *     spacing: 4.0,
 *     centerX: 0, centerY: 3.5, centerZ: 0,
 *     quality: GI_QUALITY.MEDIUM,
 *     mode: GI_MODE.SH_L2,
 *   });
 */
export function spawnGIProbeSet(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return null;

  const resolution = spec.resolution !== undefined ? spec.resolution : 16;
  const spacing = spec.spacing !== undefined ? spec.spacing : 4.0;
  const centerX = spec.centerX !== undefined ? spec.centerX : 0;
  const centerY = spec.centerY !== undefined ? spec.centerY : 3.5;
  const centerZ = spec.centerZ !== undefined ? spec.centerZ : 0;
  const quality = spec.quality !== undefined ? spec.quality : GI_QUALITY.MEDIUM;
  const mode = spec.mode !== undefined ? spec.mode : GI_MODE.SH_L2;
  const probeType = spec.type !== undefined ? spec.type : GI_PROBE_TYPE.OUTDOOR;

  const half = (resolution - 1) * 0.5;
  const eids = new Array(resolution * resolution * resolution);

  let idx = 0;
  for (let z = 0; z < resolution; z++) {
    for (let y = 0; y < resolution; y++) {
      for (let x = 0; x < resolution; x++) {
        const gridX = x;
        const gridY = y;
        const gridZ = z;
        const worldX = centerX + (x - half) * spacing;
        const worldY = centerY + (y - half) * spacing * 0.5;
        const worldZ = centerZ + (z - half) * spacing;

        const eid = spawnGIProbe(w, {
          type: probeType,
          quality,
          mode,
          gridX, gridY, gridZ,
          worldX, worldY, worldZ,
          radius: spacing * 1.5,
        });
        eids[idx++] = eid;
      }
    }
  }

  // Anchor entity carries grid metadata.
  const anchorEid = addEntity(w);
  GIVoxel.gridMinX[anchorEid] = centerX - half * spacing;
  GIVoxel.gridMinY[anchorEid] = centerY - half * spacing * 0.5;
  GIVoxel.gridMinZ[anchorEid] = centerZ - half * spacing;
  GIVoxel.gridMaxX[anchorEid] = centerX + half * spacing;
  GIVoxel.gridMaxY[anchorEid] = centerY + half * spacing * 0.5;
  GIVoxel.gridMaxZ[anchorEid] = centerZ + half * spacing;
  GIVoxel.resolution[anchorEid] = resolution;
  GIVoxel.generation[anchorEid] = 1;
  GIVoxel.valid[anchorEid] = 1;
  addComponent(w, anchorEid, GIVoxel);

  return {
    anchorEid,
    eids,
    resolution,
    spacing,
    centerX, centerY, centerZ,
  };
}

/**
 * Spawns a GI volume (indoor room or outdoor chunk).
 */
export function spawnGIVolume(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  GIVolume.kind[eid] = spec.kind !== undefined ? spec.kind : 0;
  GIVolume.enabled[eid] = 1;
  GIVolume.minX[eid] = spec.minX !== undefined ? spec.minX : -10;
  GIVolume.minY[eid] = spec.minY !== undefined ? spec.minY : 0;
  GIVolume.minZ[eid] = spec.minZ !== undefined ? spec.minZ : -10;
  GIVolume.maxX[eid] = spec.maxX !== undefined ? spec.maxX : 10;
  GIVolume.maxY[eid] = spec.maxY !== undefined ? spec.maxY : 6;
  GIVolume.maxZ[eid] = spec.maxZ !== undefined ? spec.maxZ : 10;
  GIVolume.fillR[eid] = spec.fillR !== undefined ? spec.fillR : 0.20;
  GIVolume.fillG[eid] = spec.fillG !== undefined ? spec.fillG : 0.22;
  GIVolume.fillB[eid] = spec.fillB !== undefined ? spec.fillB : 0.28;
  GIVolume.blendDistance[eid] = spec.blendDistance !== undefined ? spec.blendDistance : 1.5;
  GIVolume.cellCount[eid] = 0;
  GIVolume.probeDensity[eid] = spec.probeDensity !== undefined ? spec.probeDensity : 1.0;
  GIVolume.leakGate[eid] = spec.leakGate !== undefined ? spec.leakGate : 0.85;
  GIVolume.generation[eid] = 1;

  addComponent(w, eid, GIVolume);
  return eid;
}

/**
 * Spawns an indoor GI volume with interior-specific parameters.
 */
export function spawnGIIndoorVolume(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;

  const volumeEid = spawnGIVolume(w, Object.assign({}, spec, { kind: 0 }));

  GIIndoor.volumeEid[volumeEid] = volumeEid;
  GIIndoor.portalCount[volumeEid] = 0;
  GIIndoor.ceilingBounceR[volumeEid] = spec.ceilingBounce ? spec.ceilingBounce[0] : 0.30;
  GIIndoor.ceilingBounceG[volumeEid] = spec.ceilingBounce ? spec.ceilingBounce[1] : 0.28;
  GIIndoor.ceilingBounceB[volumeEid] = spec.ceilingBounce ? spec.ceilingBounce[2] : 0.26;
  GIIndoor.floorBounceR[volumeEid] = spec.floorBounce ? spec.floorBounce[0] : 0.28;
  GIIndoor.floorBounceG[volumeEid] = spec.floorBounce ? spec.floorBounce[1] : 0.24;
  GIIndoor.floorBounceB[volumeEid] = spec.floorBounce ? spec.floorBounce[2] : 0.20;
  GIIndoor.emissiveFillR[volumeEid] = 0.0;
  GIIndoor.emissiveFillG[volumeEid] = 0.0;
  GIIndoor.emissiveFillB[volumeEid] = 0.0;
  GIIndoor.wallOcclusion[volumeEid] = spec.wallOcclusion !== undefined ? spec.wallOcclusion : 0.90;
  GIIndoor.curtainTransmission[volumeEid] = spec.curtainTransmission !== undefined ? spec.curtainTransmission : 0.20;
  GIIndoor.enabled[volumeEid] = 1;

  // Portal entity list placeholders.
  for (let i = 0; i < 4; i++) {
    GIIndoor.portalEidList[volumeEid * 4 + i] = -1;
  }

  addComponent(w, volumeEid, GIIndoor);
  return volumeEid;
}

/**
 * Spawns an outdoor GI volume with exterior-specific parameters.
 */
export function spawnGIOutdoorVolume(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;

  const volumeEid = spawnGIVolume(w, Object.assign({}, spec, { kind: 1 }));

  GIOutdoor.volumeEid[volumeEid] = volumeEid;
  GIOutdoor.skyZenithR[volumeEid] = spec.skyZenith ? spec.skyZenith[0] : 0.29;
  GIOutdoor.skyZenithG[volumeEid] = spec.skyZenith ? spec.skyZenith[1] : 0.64;
  GIOutdoor.skyZenithB[volumeEid] = spec.skyZenith ? spec.skyZenith[2] : 0.91;
  GIOutdoor.skyHorizonR[volumeEid] = spec.skyHorizon ? spec.skyHorizon[0] : 0.75;
  GIOutdoor.skyHorizonG[volumeEid] = spec.skyHorizon ? spec.skyHorizon[1] : 0.88;
  GIOutdoor.skyHorizonB[volumeEid] = spec.skyHorizon ? spec.skyHorizon[2] : 0.96;
  GIOutdoor.groundAlbedoR[volumeEid] = spec.groundAlbedo ? spec.groundAlbedo[0] : 0.35;
  GIOutdoor.groundAlbedoG[volumeEid] = spec.groundAlbedo ? spec.groundAlbedo[1] : 0.30;
  GIOutdoor.groundAlbedoB[volumeEid] = spec.groundAlbedo ? spec.groundAlbedo[2] : 0.25;
  GIOutdoor.hazeR[volumeEid] = 0.0;
  GIOutdoor.hazeG[volumeEid] = 0.0;
  GIOutdoor.hazeB[volumeEid] = 0.0;
  GIOutdoor.sunDirR[volumeEid] = 0.5;
  GIOutdoor.sunDirG[volumeEid] = 0.8;
  GIOutdoor.sunDirB[volumeEid] = 0.3;
  GIOutdoor.biomeWeight[volumeEid] = 1.0;
  GIOutdoor.enabled[volumeEid] = 1;

  addComponent(w, volumeEid, GIOutdoor);
  return volumeEid;
}

/**
 * Spawns a GI portal between an indoor and an outdoor volume.
 */
export function spawnGIPortalVolume(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  GIPortal.type[eid] = spec.type !== undefined ? spec.type : GI_PORTAL_TYPE.DOORWAY;
  GIPortal.enabled[eid] = 1;
  GIPortal.positionX[eid] = spec.position ? spec.position[0] : 0;
  GIPortal.positionY[eid] = spec.position ? spec.position[1] : 1.0;
  GIPortal.positionZ[eid] = spec.position ? spec.position[2] : 0;
  GIPortal.normalX[eid] = spec.normal ? spec.normal[0] : 0;
  GIPortal.normalY[eid] = spec.normal ? spec.normal[1] : 0;
  GIPortal.normalZ[eid] = spec.normal ? spec.normal[2] : 1;
  GIPortal.width[eid] = spec.width !== undefined ? spec.width : 1.0;
  GIPortal.height[eid] = spec.height !== undefined ? spec.height : 2.0;
  GIPortal.indoorRoomEid[eid] = spec.indoorRoomEid !== undefined ? spec.indoorRoomEid : -1;
  GIPortal.outdoorVolumeEid[eid] = spec.outdoorVolumeEid !== undefined ? spec.outdoorVolumeEid : -1;
  GIPortal.transmission[eid] = spec.transmission !== undefined ? spec.transmission : 1.0;
  GIPortal.tintR[eid] = spec.tint ? spec.tint[0] : 1.0;
  GIPortal.tintG[eid] = spec.tint ? spec.tint[1] : 1.0;
  GIPortal.tintB[eid] = spec.tint ? spec.tint[2] : 1.0;
  GIPortal.fluxR[eid] = 0.0;
  GIPortal.fluxG[eid] = 0.0;
  GIPortal.fluxB[eid] = 0.0;
  GIPortal.visible[eid] = 0;

  addComponent(w, eid, GIPortal);

  // Register the portal with the indoor volume's portal list.
  if (GIPortal.indoorRoomEid[eid] >= 0) {
    const roomEid = GIPortal.indoorRoomEid[eid];
    const count = GIIndoor.portalCount[roomEid];
    if (count < 4) {
      GIIndoor.portalEidList[roomEid * 4 + count] = eid;
      GIIndoor.portalCount[roomEid] = count + 1;
    }
  }

  return eid;
}

/**
 * Spawns a reflection probe.
 */
export function spawnGIReflectionProbe(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  GIReflectionProbe.positionX[eid] = spec.position ? spec.position[0] : 0;
  GIReflectionProbe.positionY[eid] = spec.position ? spec.position[1] : 3.0;
  GIReflectionProbe.positionZ[eid] = spec.position ? spec.position[2] : 0;
  GIReflectionProbe.radius[eid] = spec.radius !== undefined ? spec.radius : 20.0;
  GIReflectionProbe.resolution[eid] = spec.resolution !== undefined ? spec.resolution : 128;
  GIReflectionProbe.captureFace[eid] = 6;
  GIReflectionProbe.captureFaceFrame[eid] = 0;
  GIReflectionProbe.lastCaptureFrame[eid] = 0;
  GIReflectionProbe.updateInterval[eid] = spec.updateInterval !== undefined ? spec.updateInterval : 120;
  GIReflectionProbe.hdrBias[eid] = spec.hdrBias !== undefined ? spec.hdrBias : 0.0;
  GIReflectionProbe.parallaxCorrect[eid] = spec.parallaxCorrect !== false ? 1 : 0;
  GIReflectionProbe.boxMinX[eid] = spec.boxMin ? spec.boxMin[0] : -20;
  GIReflectionProbe.boxMinY[eid] = spec.boxMin ? spec.boxMin[1] : -2;
  GIReflectionProbe.boxMinZ[eid] = spec.boxMin ? spec.boxMin[2] : -20;
  GIReflectionProbe.boxMaxX[eid] = spec.boxMax ? spec.boxMax[0] : 20;
  GIReflectionProbe.boxMaxY[eid] = spec.boxMax ? spec.boxMax[1] : 20;
  GIReflectionProbe.boxMaxZ[eid] = spec.boxMax ? spec.boxMax[2] : 20;
  GIReflectionProbe.valid[eid] = 0;
  GIReflectionProbe.enabled[eid] = 1;

  addComponent(w, eid, GIReflectionProbe);
  return eid;
}

/**
 * Spawns a lightfield owner entity.
 */
export function spawnGILightfield(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  const sampleCount = spec.sampleCount !== undefined ? spec.sampleCount : 256;
  const start = spec.startOffset !== undefined ? spec.startOffset : 0;
  const count = Math.min(sampleCount, MAX_LIGHTFIELD_SAMPLES - start);

  for (let i = 0; i < count; i++) {
    const off = start + i;
    GILightfield.positionX[off] = 0;
    GILightfield.positionY[off] = 0;
    GILightfield.positionZ[off] = 0;
    GILightfield.dirX[off] = 0;
    GILightfield.dirY[off] = 0;
    GILightfield.dirZ[off] = 1;
    GILightfield.radianceR[off] = 0;
    GILightfield.radianceG[off] = 0;
    GILightfield.radianceB[off] = 0;
    GILightfield.depth[off] = 1e6;
    GILightfield.confidence[off] = 0;
    GILightfield.valid[off] = 0;
  }

  GILightfield.sampleCount[0] = count;
  GILightfield.capacity[0] = MAX_LIGHTFIELD_SAMPLES;
  GILightfield.ownerEid[0] = eid;
  GILightfield.generation[0] = 1;

  addComponent(w, eid, GILightfield);
  return eid;
}

/**
 * Spawns an anime cel-band GI controller (attaches to a GI probe or volume).
 */
export function spawnGICelBandController(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  GICelBands.enabled[eid] = 1;
  GICelBands.bandCount[eid] = spec.bandCount !== undefined ? spec.bandCount : 4;
  GICelBands.bandSoftness[eid] = spec.bandSoftness !== undefined ? spec.bandSoftness : 0.10;
  GICelBands.bandBias[eid] = spec.bandBias !== undefined ? spec.bandBias : 0.0;

  if (spec.band0) { GICelBands.band0R[eid] = spec.band0[0]; GICelBands.band0G[eid] = spec.band0[1]; GICelBands.band0B[eid] = spec.band0[2]; }
  if (spec.band1) { GICelBands.band1R[eid] = spec.band1[0]; GICelBands.band1G[eid] = spec.band1[1]; GICelBands.band1B[eid] = spec.band1[2]; }
  if (spec.band2) { GICelBands.band2R[eid] = spec.band2[0]; GICelBands.band2G[eid] = spec.band2[1]; GICelBands.band2B[eid] = spec.band2[2]; }
  if (spec.band3) { GICelBands.band3R[eid] = spec.band3[0]; GICelBands.band3G[eid] = spec.band3[1]; GICelBands.band3B[eid] = spec.band3[2]; }

  GICelBands.ditherStrength[eid] = spec.ditherStrength !== undefined ? spec.ditherStrength : 1.0 / 255.0;
  GICelBands.ditherScale[eid] = spec.ditherScale !== undefined ? spec.ditherScale : 1.0;

  addComponent(w, eid, GICelBands);
  return eid;
}

/**
 * Spawns a radiance cache entity (one per lighting world typically).
 */
export function spawnGIRadianceCache(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  // Reset radiance cache to defaults.
  for (let i = 0; i < MAX_RADIANCE_CACHE; i++) {
    GIRadianceCache.positionX[i] = 0;
    GIRadianceCache.positionY[i] = 0;
    GIRadianceCache.positionZ[i] = 0;
    GIRadianceCache.normalX[i] = 0;
    GIRadianceCache.normalY[i] = 1;
    GIRadianceCache.normalZ[i] = 0;
    GIRadianceCache.radianceR[i] = 0;
    GIRadianceCache.radianceG[i] = 0;
    GIRadianceCache.radianceB[i] = 0;
    GIRadianceCache.age[i] = 0;
    GIRadianceCache.sampleCount[i] = 0;
    GIRadianceCache.confidence[i] = 0;
    GIRadianceCache.valid[i] = 0;
    GIRadianceCache.hashNext[i] = -1;
  }
  for (let i = 0; i < 2048; i++) GIRadianceCache.hashTable[i] = -1;

  GIRadianceCache.generation[0] = 1;
  GIRadianceCache.entryCount[0] = 0;
  GIRadianceCache.writeCursor[0] = 0;

  addComponent(w, eid, GIRadianceCache);
  return eid;
}

/**
 * Spawns a voxel grid entity. Occupancy + albedo arrays are reset.
 */
export function spawnGIVoxelGrid(world, spec = {}) {
  const w = world || getGIWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  const resolution = spec.resolution !== undefined ? spec.resolution : MAX_VOXEL_RESOLUTION;
  const total = MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION * MAX_VOXEL_RESOLUTION;

  for (let i = 0; i < total; i++) {
    GIVoxel.occupancy[i] = 0;
    GIVoxel.albedoR[i] = 0.5;
    GIVoxel.albedoG[i] = 0.5;
    GIVoxel.albedoB[i] = 0.5;
    GIVoxel.emissionR[i] = 0;
    GIVoxel.emissionG[i] = 0;
    GIVoxel.emissionB[i] = 0;
    GIVoxel.normalX[i] = 0;
    GIVoxel.normalY[i] = 1;
    GIVoxel.normalZ[i] = 0;
    GIDistanceField.distance[i] = 1.0;
    GIDistanceField.gradientX[i] = 0;
    GIDistanceField.gradientY[i] = 1;
    GIDistanceField.gradientZ[i] = 0;
  }

  GIVoxel.gridMinX[eid] = spec.minX !== undefined ? spec.minX : -32;
  GIVoxel.gridMinY[eid] = spec.minY !== undefined ? spec.minY : -8;
  GIVoxel.gridMinZ[eid] = spec.minZ !== undefined ? spec.minZ : -32;
  GIVoxel.gridMaxX[eid] = spec.maxX !== undefined ? spec.maxX : 32;
  GIVoxel.gridMaxY[eid] = spec.maxY !== undefined ? spec.maxY : 32;
  GIVoxel.gridMaxZ[eid] = spec.maxZ !== undefined ? spec.maxZ : 32;
  GIVoxel.resolution[eid] = resolution;
  GIVoxel.generation[eid] = 1;
  GIVoxel.valid[eid] = 1;

  GIDistanceField.valid[eid] = 0;
  GIDistanceField.generation[eid] = 1;

  addComponent(w, eid, GIVoxel);
  addComponent(w, eid, GIDistanceField);
  return eid;
}

/* ------------------------------------------------------------------ */
/* 7. UTILITY HELPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Marks a GI entity as dirty so downstream systems will re-bake it.
 */
export function markGIDirty(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  GIState.prevState[eid] = GIState.state[eid];
  GIState.state[eid] = GI_STATE.DIRTY;
  GIState.lastStateFrame[eid] = GIQueueFrame();
  return true;
}

/**
 * Clears the dirty flag after a successful bake.
 */
export function clearGIDirty(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  GIState.prevState[eid] = GIState.state[eid];
  GIState.state[eid] = GI_STATE.READY;
  GIState.lastBakeFrame[eid] = GIQueueFrame();
  GISH.valid[eid] = 1;
  return true;
}

/**
 * Sets GI quality on a probe.
 */
export function setGIQuality(eid, quality) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (quality < 0 || quality >= GI_QUALITY.COUNT) return false;
  GIProbeRef.quality[eid] = quality;
  markGIDirty(eid);
  return true;
}

/**
 * Sets GI mode on a probe.
 */
export function setGIMode(eid, mode) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (mode < 0 || mode >= GI_MODE.COUNT) return false;
  GIProbeRef.mode[eid] = mode;
  markGIDirty(eid);
  return true;
}

/**
 * Enables anime cel-band GI on the given entity.
 */
export function enableGICelBands(eid, bandCount, softness) {
  if (typeof eid !== 'number' || eid < 0) return false;
  GICelBands.enabled[eid] = 1;
  if (typeof bandCount === 'number') GICelBands.bandCount[eid] = Math.max(2, Math.min(8, bandCount));
  if (typeof softness === 'number') GICelBands.bandSoftness[eid] = Math.max(0, Math.min(1, softness));
  return true;
}

/**
 * Enables palette-driven GI on the given entity.
 */
export function enableGIPalette(eid, styleId, satBias, hueBias) {
  if (typeof eid !== 'number' || eid < 0) return false;
  GIPalette.enabled[eid] = 1;
  if (typeof styleId === 'number') GIPalette.styleId[eid] = styleId;
  if (typeof satBias === 'number') GIPalette.satBias[eid] = satBias;
  if (typeof hueBias === 'number') GIPalette.hueBias[eid] = hueBias;
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. GI UPDATE QUEUE (frame-scoped FIFO)                             */
/* ------------------------------------------------------------------ */

/**
 * Returns the current update queue frame counter.
 */
export function GIQueueFrame() {
  return GIUpdateQueue.currentFrame[0];
}

/**
 * Advances the queue frame counter. Called once per frame by the loop.
 */
export function advanceGIQueueFrame(frameNumber) {
  if (typeof frameNumber === 'number') {
    GIUpdateQueue.currentFrame[0] = frameNumber;
  } else {
    GIUpdateQueue.currentFrame[0]++;
  }
  // Reset head/tail/count.
  GIUpdateQueue.head[0] = 0;
  GIUpdateQueue.tail[0] = 0;
  GIUpdateQueue.count[0] = 0;
  GIUpdateQueue.queueValid[0] = 1;
}

/**
 * Enqueues an entity for GI update. Returns true if accepted, false if
 * the queue is full (oldest dropped).
 */
export function enqueueGIUpdate(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (GIUpdateQueue.queueValid[0] === 0) advanceGIQueueFrame();

  const tail = GIUpdateQueue.tail[0];
  const head = GIUpdateQueue.head[0];
  const count = GIUpdateQueue.count[0];

  if (count >= MAX_GI_UPDATE_QUEUE) {
    // Drop oldest.
    GIUpdateQueue.head[0] = (head + 1) % MAX_GI_UPDATE_QUEUE;
    GIUpdateQueue.count[0] = count - 1;
    GIUpdateQueue.droppedCount[0]++;
  }

  GIUpdateQueue.queue[tail] = eid;
  GIUpdateQueue.tail[0] = (tail + 1) % MAX_GI_UPDATE_QUEUE;
  GIUpdateQueue.count[0]++;
  GIUpdateQueue.totalEnqueued[0]++;
  return true;
}

/**
 * Drains up to `maxItems` from the queue into the caller-provided array.
 * Returns the number drained.
 */
export function drainGIUpdateQueue(maxItems, outArray) {
  const limit = maxItems !== undefined ? maxItems : GIUpdateQueue.count[0];
  let drained = 0;
  while (drained < limit && GIUpdateQueue.count[0] > 0) {
    const head = GIUpdateQueue.head[0];
    const eid = GIUpdateQueue.queue[head];
    GIUpdateQueue.head[0] = (head + 1) % MAX_GI_UPDATE_QUEUE;
    GIUpdateQueue.count[0]--;
    GIUpdateQueue.totalProcessed[0]++;

    if (outArray) outArray[drained] = eid;
    drained++;
  }
  return drained;
}

/* ------------------------------------------------------------------ */
/* 9. PORTAL ALLOCATION HELPERS                                       */
/* ------------------------------------------------------------------ */

/**
 * Allocates a portal slot from the global pool (in addition to the entity
 * spawn). Returns the entity id of the newly created portal.
 */
export function allocateGIPortal(world, spec) {
  return spawnGIPortalVolume(world, spec);
}

/**
 * Releases a portal entity.
 */
export function releaseGIPortal(world, portalEid) {
  if (typeof portalEid !== 'number' || portalEid < 0) return false;
  GIPortal.enabled[portalEid] = 0;
  GIPortal.visible[portalEid] = 0;

  // Remove from indoor room's portal list.
  const roomEid = GIPortal.indoorRoomEid[portalEid];
  if (roomEid >= 0) {
    const count = GIIndoor.portalCount[roomEid];
    for (let i = 0; i < count; i++) {
      if (GIIndoor.portalEidList[roomEid * 4 + i] === portalEid) {
        for (let j = i; j < count - 1; j++) {
          GIIndoor.portalEidList[roomEid * 4 + j] = GIIndoor.portalEidList[roomEid * 4 + j + 1];
        }
        GIIndoor.portalEidList[roomEid * 4 + (count - 1)] = -1;
        GIIndoor.portalCount[roomEid] = count - 1;
        break;
      }
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createGIWorld() {
  return createWorld({
    components: GI_COMPONENTS,
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
  MAX_SH_COEFFICIENTS,
  MAX_SH_COEFFICIENTS_HIGH,
  MAX_BOUNCE_PATHS,
  MAX_VOXEL_RESOLUTION,
  MAX_GI_PROBES,
  MAX_LIGHTFIELD_SAMPLES,
  MAX_REFLECTION_PROBES,
  MAX_GI_PORTALS,
  MAX_RADIANCE_CACHE,
  MAX_ASYNC_UPDATES,
  MAX_GI_UPDATE_QUEUE,

  // Components
  GIProbeRef,
  GIIrradiance,
  GISH,
  GISHHigh,
  GIBouncePath,
  GIVoxel,
  GIDistanceField,
  GIOcclusion,
  GIPortal,
  GIBudget,
  GIState,
  GIUpdateQueue,
  GIRadianceCache,
  GIReflectionProbe,
  GILightfield,
  GIVolume,
  GIIndoor,
  GIOutdoor,
  GICelBands,
  GIPalette,
  GIAsync,
  GILeak,
  GITemporal,
  GIVolumeBlend,
  GI_COMPONENTS,

  // Enums
  GI_STATE,
  GI_STATE_NAME,
  GI_QUALITY,
  GI_QUALITY_NAME,
  GI_MODE,
  GI_MODE_NAME,
  GI_BOUNCE_TYPE,
  GI_BOUNCE_TYPE_NAME,
  GI_PORTAL_TYPE,
  GI_PORTAL_TYPE_NAME,
  GI_PROBE_TYPE,
  GI_PROBE_TYPE_NAME,
  GI_PATH_KIND,
  GI_PATH_KIND_NAME,
  GI_LEAK_MODE,
  GI_LEAK_MODE_NAME,

  // World
  getGIWorld,
  createGIWorld,

  // Spawn
  spawnGIProbe,
  spawnGIProbeSet,
  spawnGIVolume,
  spawnGIIndoorVolume,
  spawnGIOutdoorVolume,
  spawnGIPortalVolume,
  spawnGIReflectionProbe,
  spawnGILightfield,
  spawnGICelBandController,
  spawnGIRadianceCache,
  spawnGIVoxelGrid,

  // Utilities
  markGIDirty,
  clearGIDirty,
  setGIQuality,
  setGIMode,
  enableGICelBands,
  enableGIPalette,

  // Queue
  GIQueueFrame,
  advanceGIQueueFrame,
  enqueueGIUpdate,
  drainGIUpdateQueue,

  // Portals
  allocateGIPortal,
  releaseGIPortal,
};

export default _defaultExport;