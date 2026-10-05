// File : 005
// name : src/ecs/005_lgt_AOComponents.js
// description : bitECS 0.4.0 SoA component definitions for the ambient
//               occlusion subsystem of the anime lighting stack on Android
//               mobile. Every piece of state the AO pipeline needs — sampling
//               kernels, blur ping-pong buffers, temporal accumulators,
//               screen-space AO input/output, contact-shadow volumes,
//               distance-field AO, indoor AO volumes, per-band cel-shading AO,
//               anime ink AO, and the corresponding budget/state machines — is
//               declared here as a plain-object component whose fields are
//               pre-sized TypedArrays.
//
//               This is the bitECS 0.4.0 architecture: NO `defineComponent`,
//               NO `Types` export, NO separate store registry. Components are
//               plain JS objects passed directly to
//               `createWorld({ components })` and attached via
//               `addComponent(world, eid, ComponentObject)`.
//
//               Components declared:
//                 • AOVolumeRef            — AO volume entity reference
//                 • AOSampling             — per-volume sampling parameters
//                 • AOKernel               — sample kernel (directions + weights)
//                 • AOHistory              — temporal history buffers
//                 • AOBlur                 — blur ping-pong state
//                 • AOQuality              — per-volume quality tier
//                 • AOState                — per-volume state machine
//                 • AOBudget               — per-volume cost + LOD
//                 • AOScreenSpace          — SS AO input/output params
//                 • AOContactShadow        — contact-shadow (ray-marched)
//                 • AODistanceField        — distance-field AO (per-volume)
//                 • AOIndoorVolume         — indoor-specific AO
//                 • AOOutdoorVolume        — outdoor-specific AO
//                 • AOTemporalAccumulator  — frame-to-frame AO accumulator
//                 • AODither               — ordered-dither state for AO
//                 • AOCelBands             — anime cel AO banding
//                 • AOInkOutline           — anime ink-like dark outline AO
//                 • AOEdgeFade             — screen edge fade (vignette AO)
//                 • AOBilateral            — bilateral blur state
//                 • AODenoiser             — denoiser state
//                 • AOAsync                — async update state (worker)
//                 • AOResidency            — residency tracking (mobile memory)
//                 • AOLeak                 — AO leak detection / correction
//                 • AOStyle                — anime AO style descriptor
//
//               Also exports:
//                 • MAX_ENTITIES (100000)
//                 • MAX_AO_KERNEL_SAMPLES = 32
//                 • MAX_AO_HISTORY_FRAMES = 4
//                 • MAX_AO_ASYNC_UPDATES = 4
//                 • AO_METHOD / AO_QUALITY / AO_STATE / AO_EDGE / AO_STYLE /
//                   AO_BLUR_MODE / AO_TEMPORAL_MODE / AO_DITHER enums
//                 • AO_COMPONENTS bundle for createWorld()
//                 • spawnAOVolume / spawnAOScreenSpace / spawnAOContactShadow /
//                   spawnAODistanceField / spawnAOIndoorVolume /
//                   spawnAOOutdoorVolume / spawnAOCelBandController /
//                   spawnAOInkOutline / spawnAODenoiser / spawnAOBilateral
//                   factories
//                 • Utility helpers: markAODirty, clearAODirty, setAOQuality,
//                   setAOMethod, setAOKernelSampleCount, enableAOCelBands,
//                   enableAOInkOutline, setAOBlurMode, setAOTemporalMode
//
//               Strictly Three.js r185 lights only (AO reads indirect
//               contribution of AmbientLight / HemisphereLight /
//               DirectionalLight / PointLight / SpotLight / RectAreaLight);
//               strictly bitECS 0.4.0 API only; every typed array sized once
//               to MAX_ENTITIES = 100000.
// best for : Single source of truth for every AO-related ECS component in
//            the anime lighting stack. Every downstream AO system
//            (168_lgt_AOManager.js through 185_lgt_AOMobileScaler.js)
//            imports its component definitions from here so kernel layouts,
//            quality enums, temporal accumulators, and cel-band parameters
//            stay consistent across CPU sampling, GPU uploads, and async
//            worker denoising.
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
} from './002_lgt_LightComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of samples in the AO kernel (per pixel ray count).
 */
export const MAX_AO_KERNEL_SAMPLES =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 16 :
                                 8;

/**
 * Maximum number of temporal history frames kept for AO.
 */
export const MAX_AO_HISTORY_FRAMES = 4;

/**
 * Maximum simultaneous async AO denoise jobs per tier.
 */
export const MAX_AO_ASYNC_UPDATES =
  PERF_TIER_LOCAL === 'HIGH'   ? 4 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2 :
                                 1;

/**
 * Maximum screen-space AO half-res buffer dimensions per tier. These are
 * used to size pre-computed offsets, not to allocate textures.
 */
export const MAX_AO_SCREEN_WIDTH =
  PERF_TIER_LOCAL === 'HIGH'   ? 1920 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 1280 :
                                 960;

export const MAX_AO_SCREEN_HEIGHT =
  PERF_TIER_LOCAL === 'HIGH'   ? 1080 :
  PERF_TIER_LOCAL === 'MEDIUM' ?  720 :
                                 540;

/**
 * Maximum number of cel bands for anime AO quantization.
 */
export const MAX_AO_CEL_BANDS = 8;

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * AO methods supported by the pipeline. Each one maps to a specific
 * fragment shader path in the AO composer.
 */
export const AO_METHOD = Object.freeze({
  OFF:              0,
  FLAT:             1,   // flat grey constant
  SSAO:             2,   // classic screen-space AO
  HBAO:             3,   // horizon-based AO (NVIDIA)
  GTAO:             4,   // ground truth AO
  VXAO:             5,   // voxel AO
  DF_AO:            6,   // distance-field AO
  CONTACT:          7,   // contact-shadow AO
  RAY_MARCHED:      8,   // ray-marched AO
  SDF_RAY:          9,   // SDF ray-marched AO
  MULTI_BOUNCE:    10,   // multi-bounce AO
  ANIME_CEL:       11,   // anime cel-banded AO
  INK_OUTLINE:     12,   // anime ink-like outline AO
  HYBRID:          13,   // auto-selected combination
  COUNT:           14,
});

export const AO_METHOD_NAME = Object.freeze([
  'off',
  'flat',
  'ssao',
  'hbao',
  'gtao',
  'vxao',
  'df_ao',
  'contact',
  'ray_marched',
  'sdf_ray',
  'multi_bounce',
  'anime_cel',
  'ink_outline',
  'hybrid',
]);

/**
 * Per-volume AO quality tiers.
 */
export const AO_QUALITY = Object.freeze({
  OFF:      0,
  MINIMAL:  1,
  LOW:      2,
  MEDIUM:   3,
  HIGH:     4,
  ULTRA:    5,
  COUNT:    6,
});

export const AO_QUALITY_NAME = Object.freeze([
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'ultra',
]);

/**
 * Per-volume AO state machine.
 */
export const AO_STATE = Object.freeze({
  IDLE:       0,
  DIRTY:      1,
  BAKING:     2,
  READY:      3,
  STALE:      4,
  FAILED:     5,
  DISABLED:   6,
  COUNT:      7,
});

export const AO_STATE_NAME = Object.freeze([
  'idle',
  'dirty',
  'baking',
  'ready',
  'stale',
  'failed',
  'disabled',
]);

/**
 * Blur modes.
 */
export const AO_BLUR_MODE = Object.freeze({
  NONE:         0,
  BOX:          1,
  GAUSSIAN:     2,
  BILATERAL:    3,
  DEPTH_AWARE:  4,
  NORMAL_AWARE: 5,
  ANIME_SOFT:   6,   // anime-styled soft edge
  COUNT:        7,
});

export const AO_BLUR_MODE_NAME = Object.freeze([
  'none',
  'box',
  'gaussian',
  'bilateral',
  'depth_aware',
  'normal_aware',
  'anime_soft',
]);

/**
 * Temporal accumulation modes.
 */
export const AO_TEMPORAL_MODE = Object.freeze({
  NONE:            0,
  LINEAR:          1,   // uniform rolling average
  EXPONENTIAL:     2,   // EMA
  VARIANCE_GUIDED: 3,   // variance-clipped accumulation
  ANIME_PERSIST:   4,   // high persistence for anime stable look
  COUNT:           5,
});

export const AO_TEMPORAL_MODE_NAME = Object.freeze([
  'none',
  'linear',
  'exponential',
  'variance_guided',
  'anime_persist',
]);

/**
 * Dither modes.
 */
export const AO_DITHER = Object.freeze({
  NONE:    0,
  BAYER4:  1,
  BAYER8:  2,
  BLUE:    3,
  HASH:    4,
  COUNT:   5,
});

export const AO_DITHER_NAME = Object.freeze([
  'none',
  'bayer4',
  'bayer8',
  'blue',
  'hash',
]);

/**
 * Anime AO style descriptors matching the reference image set.
 */
export const AO_STYLE = Object.freeze({
  REALISTIC:   0,   // soft realistic AO
  CEL_HARD:    1,   // 2-band anime cel AO
  CEL_SOFT:    2,   // 3-band anime cel AO with soft transitions
  INK_LINE:    3,   // anime ink-outline-like dark edges
  POSTERIZED:  4,   // posterized AO with hard bands
  PASTEL:      5,   // very soft pastel AO (image 7)
  DENSE_LINE:  6,   // denser ink edges (image 8)
  WISP:        7,   // wispy soft AO (image 2 river)
  COUNT:       8,
});

export const AO_STYLE_NAME = Object.freeze([
  'realistic',
  'cel_hard',
  'cel_soft',
  'ink_line',
  'posterized',
  'pastel',
  'dense_line',
  'wisp',
]);

/**
 * Residency / memory state.
 */
export const AO_RESIDENCY = Object.freeze({
  UNLOADED:  0,
  RESIDENT:  1,
  EVICTING:  2,
  RELOADING: 3,
  COUNT:     4,
});

export const AO_RESIDENCY_NAME = Object.freeze([
  'unloaded',
  'resident',
  'evicting',
  'reloading',
]);

/**
 * AO edge fade styles (vignette-like AO reduction at frame edges).
 */
export const AO_EDGE = Object.freeze({
  NONE:      0,
  LINEAR:    1,
  RADIAL:    2,
  QUADRATIC: 3,
  SMOOTH:    4,
  COUNT:     5,
});

export const AO_EDGE_NAME = Object.freeze([
  'none',
  'linear',
  'radial',
  'quadratic',
  'smooth',
]);

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * AOVolumeRef — binds an AO volume entity to a world-space box and a
 * method/quality pair.
 */
export const AOVolumeRef = {
  method:         new Uint8Array(MAX_ENTITIES),    // AO_METHOD
  quality:        new Uint8Array(MAX_ENTITIES),    // AO_QUALITY
  style:          new Uint8Array(MAX_ENTITIES),    // AO_STYLE
  enabled:        new Uint8Array(MAX_ENTITIES),
  // World bounds.
  minX:           new Float32Array(MAX_ENTITIES),
  minY:           new Float32Array(MAX_ENTITIES),
  minZ:           new Float32Array(MAX_ENTITIES),
  maxX:           new Float32Array(MAX_ENTITIES),
  maxY:           new Float32Array(MAX_ENTITIES),
  maxZ:           new Float32Array(MAX_ENTITIES),
  // Global AO scale factors.
  intensity:      new Float32Array(MAX_ENTITIES),
  radius:         new Float32Array(MAX_ENTITIES),
  bias:           new Float32Array(MAX_ENTITIES),
  maxDistance:    new Float32Array(MAX_ENTITIES),
  // Indoor/outdoor mix.
  indoorFactor:   new Float32Array(MAX_ENTITIES),
};

/**
 * AOSampling — per-volume sampling parameters.
 */
export const AOSampling = {
  sampleCount:       new Uint8Array(MAX_ENTITIES),
  stepCount:         new Uint8Array(MAX_ENTITIES),
  stepScale:         new Float32Array(MAX_ENTITIES),
  jitterAmount:      new Float32Array(MAX_ENTITIES),
  jitterHz:          new Float32Array(MAX_ENTITIES),
  hemisphereBias:    new Float32Array(MAX_ENTITIES),
  useNoise:          new Uint8Array(MAX_ENTITIES),
  noiseScale:        new Float32Array(MAX_ENTITIES),
  // Adaptive sampling parameters.
  adaptiveEnabled:   new Uint8Array(MAX_ENTITIES),
  adaptiveThreshold: new Float32Array(MAX_ENTITIES),
  adaptiveMinSteps:  new Uint8Array(MAX_ENTITIES),
  adaptiveMaxSteps:  new Uint8Array(MAX_ENTITIES),
};

/**
 * AOKernel — AO sample kernel (dirs + weights). Stored as a fixed
 * MAX_AO_KERNEL_SAMPLES × 4 flat array (x, y, z, weight) shared across
 * all AO volumes.
 */
export const AOKernel = {
  dirX:      new Float32Array(MAX_AO_KERNEL_SAMPLES),
  dirY:      new Float32Array(MAX_AO_KERNEL_SAMPLES),
  dirZ:      new Float32Array(MAX_AO_KERNEL_SAMPLES),
  weight:    new Float32Array(MAX_AO_KERNEL_SAMPLES),
  count:     new Uint8Array(1),
  seed:      new Uint32Array(1),
  generation:new Uint32Array(1),
  // Per-volume override coefficients.
  kernelScale:new Float32Array(MAX_ENTITIES),
  kernelBias: new Float32Array(MAX_ENTITIES),
};

/**
 * AOHistory — temporal history buffers for AO. Indexed by volume × history
 * frame index.
 */
export const AOHistory = {
  historyWidth:      new Uint16Array(MAX_ENTITIES),
  historyHeight:     new Uint16Array(MAX_ENTITIES),
  historyValid:      new Uint8Array(MAX_ENTITIES),
  frameIndex:        new Uint8Array(MAX_ENTITIES),
  historyWeight:     new Float32Array(MAX_ENTITIES),
  historyTintR:      new Float32Array(MAX_ENTITIES),
  historyTintG:      new Float32Array(MAX_ENTITIES),
  historyTintB:      new Float32Array(MAX_ENTITIES),
  lastUpdateFrame:   new Uint32Array(MAX_ENTITIES),
  reprojectX:        new Float32Array(MAX_ENTITIES),
  reprojectY:        new Float32Array(MAX_ENTITIES),
  varianceEstimate:  new Float32Array(MAX_ENTITIES),
  rejectionThreshold:new Float32Array(MAX_ENTITIES),
};

/**
 * AOBlur — blur pass parameters (ping-pong).
 */
export const AOBlur = {
  mode:              new Uint8Array(MAX_ENTITIES),    // AO_BLUR_MODE
  radius:            new Float32Array(MAX_ENTITIES),
  kernelSize:        new Uint8Array(MAX_ENTITIES),    // 3, 5, 7, 9
  depthThreshold:    new Float32Array(MAX_ENTITIES),
  normalThreshold:   new Float32Array(MAX_ENTITIES),
  sharpness:         new Float32Array(MAX_ENTITIES),
  pingPongPhase:     new Uint8Array(MAX_ENTITIES),    // 0 or 1
  passes:            new Uint8Array(MAX_ENTITIES),    // number of blur passes
  lastBlurMs:        new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * AOQuality — per-volume quality tier + LOD + scale.
 */
export const AOQuality = {
  tier:              new Uint8Array(MAX_ENTITIES),    // AO_QUALITY
  tierTarget:        new Uint8Array(MAX_ENTITIES),
  resolutionScale:   new Float32Array(MAX_ENTITIES),  // 0.25..1.0
  halfRes:           new Uint8Array(MAX_ENTITIES),
  quarterRes:        new Uint8Array(MAX_ENTITIES),
  downscaleFactor:   new Float32Array(MAX_ENTITIES),
  upsampleMode:      new Uint8Array(MAX_ENTITIES),    // 0=linear 1=bilateral 2=anime
  lastQualityChange: new Uint32Array(MAX_ENTITIES),
};

/**
 * AOState — per-volume state machine.
 */
export const AOState = {
  state:             new Uint8Array(MAX_ENTITIES),    // AO_STATE
  prevState:         new Uint8Array(MAX_ENTITIES),
  lastStateFrame:    new Uint32Array(MAX_ENTITIES),
  lastUpdateFrame:   new Uint32Array(MAX_ENTITIES),
  lastBakeFrame:     new Uint32Array(MAX_ENTITIES),
  failureCount:      new Uint8Array(MAX_ENTITIES),
  lastError:         new Int32Array(MAX_ENTITIES),
  leakFlag:          new Uint8Array(MAX_ENTITIES),
};

/**
 * AOBudget — per-volume cost + LOD + async.
 */
export const AOBudget = {
  cost:              new Float32Array(MAX_ENTITIES),
  costEma:           new Float32Array(MAX_ENTITIES),
  lastCostMs:        new Float32Array(MAX_ENTITIES),
  lod:               new Uint8Array(MAX_ENTITIES),
  lodTarget:         new Uint8Array(MAX_ENTITIES),
  lastLodFrame:      new Uint32Array(MAX_ENTITIES),
  priority:          new Uint8Array(MAX_ENTITIES),
  asyncPending:      new Uint8Array(MAX_ENTITIES),
  asyncWorkerId:     new Int16Array(MAX_ENTITIES),
  asyncStartFrame:   new Uint32Array(MAX_ENTITIES),
  asyncTimeout:      new Uint16Array(MAX_ENTITIES),
};

/**
 * AOScreenSpace — screen-space AO input/output dimensions + parameters.
 * Attached to a single controller entity per frame.
 */
export const AOScreenSpace = {
  screenWidth:       new Uint16Array(MAX_ENTITIES),
  screenHeight:      new Uint16Array(MAX_ENTITIES),
  aoWidth:           new Uint16Array(MAX_ENTITIES),
  aoHeight:          new Uint16Array(MAX_ENTITIES),
  depthWidth:        new Uint16Array(MAX_ENTITIES),
  depthHeight:       new Uint16Array(MAX_ENTITIES),
  tanHalfFov:        new Float32Array(MAX_ENTITIES),
  aspect:            new Float32Array(MAX_ENTITIES),
  nearPlane:         new Float32Array(MAX_ENTITIES),
  farPlane:          new Float32Array(MAX_ENTITIES),
  projection00:      new Float32Array(MAX_ENTITIES),
  projection11:      new Float32Array(MAX_ENTITIES),
  inverseProjection00:new Float32Array(MAX_ENTITIES),
  inverseProjection11:new Float32Array(MAX_ENTITIES),
  // Dynamic offset scaling.
  offsetScale:       new Float32Array(MAX_ENTITIES),
  offsetBias:        new Float32Array(MAX_ENTITIES),
  screenSpaceValid:  new Uint8Array(MAX_ENTITIES),
};

/**
 * AOContactShadow — contact-shadow (ray-marched) parameters per volume.
 */
export const AOContactShadow = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  rayCount:          new Uint8Array(MAX_ENTITIES),
  maxDistance:       new Float32Array(MAX_ENTITIES),
  stepCount:         new Uint8Array(MAX_ENTITIES),
  thickness:         new Float32Array(MAX_ENTITIES),
  bias:              new Float32Array(MAX_ENTITIES),
  jitter:            new Float32Array(MAX_ENTITIES),
  fadeNear:          new Float32Array(MAX_ENTITIES),
  fadeFar:           new Float32Array(MAX_ENTITIES),
  strength:          new Float32Array(MAX_ENTITIES),
  tintR:             new Float32Array(MAX_ENTITIES),
  tintG:             new Float32Array(MAX_ENTITIES),
  tintB:             new Float32Array(MAX_ENTITIES),
};

/**
 * AODistanceField — per-volume distance-field AO state.
 */
export const AODistanceField = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  fieldEid:          new Int32Array(MAX_ENTITIES),     // voxel grid entity
  maxDistance:       new Float32Array(MAX_ENTITIES),
  stepScale:         new Float32Array(MAX_ENTITIES),
  bias:              new Float32Array(MAX_ENTITIES),
  useGradientBias:   new Uint8Array(MAX_ENTITIES),
  gradientScale:     new Float32Array(MAX_ENTITIES),
  coneAngle:         new Float32Array(MAX_ENTITIES),
  coneSteps:         new Uint8Array(MAX_ENTITIES),
  fadeEdge:          new Float32Array(MAX_ENTITIES),
};

/**
 * AOIndoorVolume — indoor-specific AO.
 */
export const AOIndoorVolume = {
  volumeEid:         new Int32Array(MAX_ENTITIES),
  ceilingAO:         new Float32Array(MAX_ENTITIES),
  floorAO:           new Float32Array(MAX_ENTITIES),
  wallAO:            new Float32Array(MAX_ENTITIES),
  cornerBoost:       new Float32Array(MAX_ENTITIES),
  contactAO:         new Float32Array(MAX_ENTITIES),
  curtainAO:         new Float32Array(MAX_ENTITIES),
  ceilingAOColorR:   new Float32Array(MAX_ENTITIES),
  ceilingAOColorG:   new Float32Array(MAX_ENTITIES),
  ceilingAOColorB:   new Float32Array(MAX_ENTITIES),
  floorAOColorR:     new Float32Array(MAX_ENTITIES),
  floorAOColorG:     new Float32Array(MAX_ENTITIES),
  floorAOColorB:     new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * AOOutdoorVolume — outdoor-specific AO.
 */
export const AOOutdoorVolume = {
  volumeEid:         new Int32Array(MAX_ENTITIES),
  groundAO:          new Float32Array(MAX_ENTITIES),
  skyAO:             new Float32Array(MAX_ENTITIES),
  horizonAO:         new Float32Array(MAX_ENTITIES),
  canopyAO:          new Float32Array(MAX_ENTITIES),
  waterAO:           new Float32Array(MAX_ENTITIES),
  biomeDensity:      new Float32Array(MAX_ENTITIES),
  groundAOColorR:    new Float32Array(MAX_ENTITIES),
  groundAOColorG:    new Float32Array(MAX_ENTITIES),
  groundAOColorB:    new Float32Array(MAX_ENTITIES),
  skyAOColorR:       new Float32Array(MAX_ENTITIES),
  skyAOColorG:       new Float32Array(MAX_ENTITIES),
  skyAOColorB:       new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * AOTemporalAccumulator — frame-to-frame AO accumulator state.
 */
export const AOTemporalAccumulator = {
  mode:               new Uint8Array(MAX_ENTITIES),   // AO_TEMPORAL_MODE
  frameCount:         new Uint16Array(MAX_ENTITIES),
  blendFactor:        new Float32Array(MAX_ENTITIES),
  historyLength:      new Uint8Array(MAX_ENTITIES),
  resetOnDisocclusion:new Uint8Array(MAX_ENTITIES),
  disocclusionThreshold:new Float32Array(MAX_ENTITIES),
  jitterPhaseX:       new Float32Array(MAX_ENTITIES),
  jitterPhaseY:       new Float32Array(MAX_ENTITIES),
  lastAccumulateFrame:new Uint32Array(MAX_ENTITIES),
  // Persistence for anime stable look.
  persistence:        new Float32Array(MAX_ENTITIES),
  enabled:            new Uint8Array(MAX_ENTITIES),
};

/**
 * AODither — ordered-dither state for AO quantization.
 */
export const AODither = {
  mode:              new Uint8Array(MAX_ENTITIES),    // AO_DITHER
  strength:          new Float32Array(MAX_ENTITIES),
  scale:             new Float32Array(MAX_ENTITIES),
  animated:          new Uint8Array(MAX_ENTITIES),
  animationHz:       new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * AOCelBands — anime cel AO banding.
 */
export const AOCelBands = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  bandCount:         new Uint8Array(MAX_ENTITIES),
  bandSoftness:      new Float32Array(MAX_ENTITIES),
  bandBias:          new Float32Array(MAX_ENTITIES),
  // Per-band colors (matches 033 composer palette slots).
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
  // Anime-specific extras.
  rimBoost:          new Float32Array(MAX_ENTITIES),
  shadowTintR:       new Float32Array(MAX_ENTITIES),
  shadowTintG:       new Float32Array(MAX_ENTITIES),
  shadowTintB:       new Float32Array(MAX_ENTITIES),
};

/**
 * AOInkOutline — anime ink-like dark outline AO.
 */
export const AOInkOutline = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  thickness:         new Float32Array(MAX_ENTITIES),
  strength:          new Float32Array(MAX_ENTITIES),
  softness:          new Float32Array(MAX_ENTITIES),
  // Detection thresholds.
  depthThreshold:    new Float32Array(MAX_ENTITIES),
  normalThreshold:   new Float32Array(MAX_ENTITIES),
  // Line style.
  colorR:            new Float32Array(MAX_ENTITIES),
  colorG:            new Float32Array(MAX_ENTITIES),
  colorB:            new Float32Array(MAX_ENTITIES),
  // Variable line width based on luminance.
  widthLumaBias:     new Float32Array(MAX_ENTITIES),
  widthDistanceBias: new Float32Array(MAX_ENTITIES),
};

/**
 * AOEdgeFade — screen-edge AO fade (approximate vignette).
 */
export const AOEdgeFade = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  style:             new Uint8Array(MAX_ENTITIES),    // AO_EDGE
  startRadius:       new Float32Array(MAX_ENTITIES),
  endRadius:         new Float32Array(MAX_ENTITIES),
  strength:          new Float32Array(MAX_ENTITIES),
  aspectCompensate:  new Uint8Array(MAX_ENTITIES),
};

/**
 * AOBilateral — bilateral-blur specific state.
 */
export const AOBilateral = {
  depthSigma:        new Float32Array(MAX_ENTITIES),
  normalSigma:       new Float32Array(MAX_ENTITIES),
  lumaSigma:         new Float32Array(MAX_ENTITIES),
  spatialSigma:      new Float32Array(MAX_ENTITIES),
  kernelRadius:      new Uint8Array(MAX_ENTITIES),
  passes:            new Uint8Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/**
 * AODenoiser — denoiser state (spatial + temporal).
 */
export const AODenoiser = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  spatialPasses:     new Uint8Array(MAX_ENTITIES),
  temporalPasses:    new Uint8Array(MAX_ENTITIES),
  blendStrength:     new Float32Array(MAX_ENTITIES),
  // Noise-aware thresholds.
  noiseThreshold:    new Float32Array(MAX_ENTITIES),
  minVariance:       new Float32Array(MAX_ENTITIES),
  // Anime-specific preserve edges.
  preserveEdges:     new Uint8Array(MAX_ENTITIES),
  edgeSharpness:     new Float32Array(MAX_ENTITIES),
  lastDenoiseMs:     new Float32Array(MAX_ENTITIES),
};

/**
 * AOAsync — async AO update state (worker denoise in flight).
 */
export const AOAsync = {
  pendingJobs:       new Uint32Array(1),
  completedJobs:     new Uint32Array(1),
  failedJobs:        new Uint32Array(1),
  cancelledJobs:     new Uint32Array(1),
  workerCount:       new Uint8Array(1),
  workerBusy:        new Uint8Array(8),
  workerLastJobMs:   new Float32Array(8),
  workerTotalMs:     new Float32Array(8),
  workerJobsDone:    new Uint32Array(8),
  totalDenoiseMs:    new Float32Array(1),
  lastJobStartFrame: new Uint32Array(1),
  lastJobEndFrame:   new Uint32Array(1),
};

/**
 * AOResidency — residency tracking for mobile memory budgets.
 */
export const AOResidency = {
  state:             new Uint8Array(MAX_ENTITIES),    // AO_RESIDENCY
  bytesAllocated:    new Uint32Array(MAX_ENTITIES),
  bytesPeak:         new Uint32Array(MAX_ENTITIES),
  lastResidentFrame: new Uint32Array(MAX_ENTITIES),
  lastEvictFrame:    new Uint32Array(MAX_ENTITIES),
  evictAfterFrames:  new Uint16Array(MAX_ENTITIES),
  pinned:            new Uint8Array(MAX_ENTITIES),
};

/**
 * AOLeak — AO leak detection / correction state.
 */
export const AOLeak = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  magnitude:         new Float32Array(MAX_ENTITIES),
  threshold:         new Float32Array(MAX_ENTITIES),
  correctionFactor:  new Float32Array(MAX_ENTITIES),
  detectionCount:    new Uint32Array(MAX_ENTITIES),
  lastDetectionFrame:new Uint32Array(MAX_ENTITIES),
  // Smooth leak removal.
  smoothing:         new Float32Array(MAX_ENTITIES),
  valid:             new Uint8Array(MAX_ENTITIES),
};

/**
 * AOStyle — full anime AO style descriptor (matches the reference image set).
 */
export const AOStyle = {
  style:             new Uint8Array(MAX_ENTITIES),    // AO_STYLE
  celBandCount:      new Uint8Array(MAX_ENTITIES),
  celSoftness:       new Float32Array(MAX_ENTITIES),
  inkEnabled:        new Uint8Array(MAX_ENTITIES),
  inkThickness:      new Float32Array(MAX_ENTITIES),
  inkStrength:       new Float32Array(MAX_ENTITIES),
  inkColorR:         new Float32Array(MAX_ENTITIES),
  inkColorG:         new Float32Array(MAX_ENTITIES),
  inkColorB:         new Float32Array(MAX_ENTITIES),
  overallIntensity:  new Float32Array(MAX_ENTITIES),
  ambientBoost:      new Float32Array(MAX_ENTITIES),
  tintR:             new Float32Array(MAX_ENTITIES),
  tintG:             new Float32Array(MAX_ENTITIES),
  tintB:             new Float32Array(MAX_ENTITIES),
  enabled:           new Uint8Array(MAX_ENTITIES),
};

/* ------------------------------------------------------------------ */
/* 3. ECS WORLD COMPONENT BUNDLE                                      */
/* ------------------------------------------------------------------ */

export const AO_COMPONENTS = Object.freeze({
  AOVolumeRef,
  AOSampling,
  AOKernel,
  AOHistory,
  AOBlur,
  AOQuality,
  AOState,
  AOBudget,
  AOScreenSpace,
  AOContactShadow,
  AODistanceField,
  AOIndoorVolume,
  AOOutdoorVolume,
  AOTemporalAccumulator,
  AODither,
  AOCelBands,
  AOInkOutline,
  AOEdgeFade,
  AOBilateral,
  AODenoiser,
  AOAsync,
  AOResidency,
  AOLeak,
  AOStyle,
});

/* ------------------------------------------------------------------ */
/* 4. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

let _aoWorld = null;

export function getAOWorld() {
  if (_aoWorld) return _aoWorld;
  try {
    _aoWorld = createWorld({
      components: AO_COMPONENTS,
      time: {
        delta: 0,
        elapsed: 0,
        then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      },
    });
  } catch (e) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.AO, `[005_lgt_AOComponents] failed to create AO world: ${e && e.message}`);
    _aoWorld = null;
  }
  return _aoWorld;
}

/* ------------------------------------------------------------------ */
/* 5. KERNEL GENERATOR                                                */
/* ------------------------------------------------------------------ */

/**
 * Generates the standard cosine-weighted hemisphere AO kernel. Called once
 * at boot; the result is stored in the shared AOKernel arrays.
 */
export function generateAOKernel(sampleCount, seed) {
  const count = Math.max(1, Math.min(sampleCount, MAX_AO_KERNEL_SAMPLES));
  const s = seed !== undefined ? seed : 1337;

  // Deterministic LCG.
  let state = s >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };

  for (let i = 0; i < count; i++) {
    // Cosine-weighted hemisphere sample.
    const u1 = next();
    const u2 = next();
    const r = Math.sqrt(u1);
    const theta = 2 * Math.PI * u2;
    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    const z = Math.sqrt(Math.max(0, 1 - u1));

    AOKernel.dirX[i] = x;
    AOKernel.dirY[i] = z;   // Y is the hemisphere pole
    AOKernel.dirZ[i] = y;

    // Weight — scale so near samples are stronger.
    const weight = 0.3 + 0.7 * (i / count);
    AOKernel.weight[i] = weight;
  }

  AOKernel.count[0] = count;
  AOKernel.seed[0] = s;
  AOKernel.generation[0]++;

  return count;
}

/* ------------------------------------------------------------------ */
/* 6. SPAWN HELPERS                                                   */
/* ------------------------------------------------------------------ */

function _attachAOBase(world, eid, spec) {
  // AOVolumeRef
  AOVolumeRef.method[eid]        = spec.method !== undefined ? spec.method : AO_METHOD.HBAO;
  AOVolumeRef.quality[eid]       = spec.quality !== undefined ? spec.quality : AO_QUALITY.MEDIUM;
  AOVolumeRef.style[eid]         = spec.style !== undefined ? spec.style : AO_STYLE.CEL_SOFT;
  AOVolumeRef.enabled[eid]       = 1;
  AOVolumeRef.minX[eid]          = spec.minX !== undefined ? spec.minX : -32;
  AOVolumeRef.minY[eid]          = spec.minY !== undefined ? spec.minY : -4;
  AOVolumeRef.minZ[eid]          = spec.minZ !== undefined ? spec.minZ : -32;
  AOVolumeRef.maxX[eid]          = spec.maxX !== undefined ? spec.maxX : 32;
  AOVolumeRef.maxY[eid]          = spec.maxY !== undefined ? spec.maxY : 32;
  AOVolumeRef.maxZ[eid]          = spec.maxZ !== undefined ? spec.maxZ : 32;
  AOVolumeRef.intensity[eid]     = spec.intensity !== undefined ? spec.intensity : 1.0;
  AOVolumeRef.radius[eid]        = spec.radius !== undefined ? spec.radius : 2.0;
  AOVolumeRef.bias[eid]          = spec.bias !== undefined ? spec.bias : 0.025;
  AOVolumeRef.maxDistance[eid]   = spec.maxDistance !== undefined ? spec.maxDistance : 12.0;
  AOVolumeRef.indoorFactor[eid]  = spec.indoorFactor !== undefined ? spec.indoorFactor : 0.0;

  // AOSampling
  AOSampling.sampleCount[eid]     = Math.min(spec.sampleCount !== undefined ? spec.sampleCount : 8, MAX_AO_KERNEL_SAMPLES);
  AOSampling.stepCount[eid]       = spec.stepCount !== undefined ? spec.stepCount : 8;
  AOSampling.stepScale[eid]       = 1.0;
  AOSampling.jitterAmount[eid]    = 0.5;
  AOSampling.jitterHz[eid]        = 8.0;
  AOSampling.hemisphereBias[eid]  = 0.05;
  AOSampling.useNoise[eid]        = 1;
  AOSampling.noiseScale[eid]      = 4.0;
  AOSampling.adaptiveEnabled[eid] = 0;
  AOSampling.adaptiveThreshold[eid] = 0.15;
  AOSampling.adaptiveMinSteps[eid]  = 4;
  AOSampling.adaptiveMaxSteps[eid]  = 16;

  // Kernel scale
  AOKernel.kernelScale[eid]  = 1.0;
  AOKernel.kernelBias[eid]   = 0.0;

  // AOHistory
  AOHistory.historyWidth[eid]       = MAX_AO_SCREEN_WIDTH;
  AOHistory.historyHeight[eid]      = MAX_AO_SCREEN_HEIGHT;
  AOHistory.historyValid[eid]       = 0;
  AOHistory.frameIndex[eid]         = 0;
  AOHistory.historyWeight[eid]      = 0.90;
  AOHistory.historyTintR[eid]       = 1.0;
  AOHistory.historyTintG[eid]       = 1.0;
  AOHistory.historyTintB[eid]       = 1.0;
  AOHistory.lastUpdateFrame[eid]    = 0;
  AOHistory.reprojectX[eid]         = 0;
  AOHistory.reprojectY[eid]         = 0;
  AOHistory.varianceEstimate[eid]   = 0;
  AOHistory.rejectionThreshold[eid] = 0.15;

  // AOBlur
  AOBlur.mode[eid]            = AO_BLUR_MODE.ANIME_SOFT;
  AOBlur.radius[eid]          = 4.0;
  AOBlur.kernelSize[eid]      = 5;
  AOBlur.depthThreshold[eid]  = 0.05;
  AOBlur.normalThreshold[eid] = 0.15;
  AOBlur.sharpness[eid]       = 0.5;
  AOBlur.pingPongPhase[eid]   = 0;
  AOBlur.passes[eid]          = 1;
  AOBlur.lastBlurMs[eid]      = 0;
  AOBlur.enabled[eid]         = 1;

  // AOQuality
  AOQuality.tier[eid]              = spec.quality !== undefined ? spec.quality : AO_QUALITY.MEDIUM;
  AOQuality.tierTarget[eid]        = AOQuality.tier[eid];
  AOQuality.resolutionScale[eid]   = 0.5;
  AOQuality.halfRes[eid]           = 1;
  AOQuality.quarterRes[eid]        = 0;
  AOQuality.downscaleFactor[eid]   = 2.0;
  AOQuality.upsampleMode[eid]      = 2;
  AOQuality.lastQualityChange[eid] = 0;

  // AOState
  AOState.state[eid]              = AO_STATE.IDLE;
  AOState.prevState[eid]          = AO_STATE.IDLE;
  AOState.lastStateFrame[eid]     = 0;
  AOState.lastUpdateFrame[eid]    = 0;
  AOState.lastBakeFrame[eid]      = 0;
  AOState.failureCount[eid]       = 0;
  AOState.lastError[eid]          = -1;
  AOState.leakFlag[eid]           = 0;

  // AOBudget
  AOBudget.cost[eid]              = 1.0;
  AOBudget.costEma[eid]           = 1.0;
  AOBudget.lastCostMs[eid]        = 0;
  AOBudget.lod[eid]               = 0;
  AOBudget.lodTarget[eid]         = 0;
  AOBudget.lastLodFrame[eid]      = 0;
  AOBudget.priority[eid]          = 100;
  AOBudget.asyncPending[eid]      = 0;
  AOBudget.asyncWorkerId[eid]     = -1;
  AOBudget.asyncStartFrame[eid]   = 0;
  AOBudget.asyncTimeout[eid]      = 0;

  // AOTemporalAccumulator
  AOTemporalAccumulator.mode[eid]                     = AO_TEMPORAL_MODE.EXPONENTIAL;
  AOTemporalAccumulator.frameCount[eid]               = 0;
  AOTemporalAccumulator.blendFactor[eid]              = 0.10;
  AOTemporalAccumulator.historyLength[eid]            = 0;
  AOTemporalAccumulator.resetOnDisocclusion[eid]      = 1;
  AOTemporalAccumulator.disocclusionThreshold[eid]    = 0.15;
  AOTemporalAccumulator.jitterPhaseX[eid]             = 0;
  AOTemporalAccumulator.jitterPhaseY[eid]             = 0;
  AOTemporalAccumulator.lastAccumulateFrame[eid]      = 0;
  AOTemporalAccumulator.persistence[eid]              = 0.92;
  AOTemporalAccumulator.enabled[eid]                  = 1;

  // AODither
  AODither.mode[eid]        = AO_DITHER.BAYER4;
  AODither.strength[eid]    = 1.0 / 255.0;
  AODither.scale[eid]       = 1.0;
  AODither.animated[eid]    = 0;
  AODither.animationHz[eid] = 8.0;
  AODither.enabled[eid]     = 1;

  // AOCelBands
  AOCelBands.enabled[eid]     = 0;
  AOCelBands.bandCount[eid]   = 4;
  AOCelBands.bandSoftness[eid]= 0.10;
  AOCelBands.bandBias[eid]    = 0.0;
  AOCelBands.rimBoost[eid]    = 0.0;
  AOCelBands.shadowTintR[eid] = 0.20;
  AOCelBands.shadowTintG[eid] = 0.22;
  AOCelBands.shadowTintB[eid] = 0.28;

  // AOInkOutline
  AOInkOutline.enabled[eid]         = 0;
  AOInkOutline.thickness[eid]       = 0.003;
  AOInkOutline.strength[eid]        = 0.35;
  AOInkOutline.softness[eid]        = 0.15;
  AOInkOutline.depthThreshold[eid]  = 0.03;
  AOInkOutline.normalThreshold[eid] = 0.20;
  AOInkOutline.colorR[eid]          = 0.05;
  AOInkOutline.colorG[eid]          = 0.05;
  AOInkOutline.colorB[eid]          = 0.08;
  AOInkOutline.widthLumaBias[eid]   = 0.5;
  AOInkOutline.widthDistanceBias[eid] = 1.0;

  // AOEdgeFade
  AOEdgeFade.enabled[eid]          = 0;
  AOEdgeFade.style[eid]            = AO_EDGE.RADIAL;
  AOEdgeFade.startRadius[eid]      = 0.80;
  AOEdgeFade.endRadius[eid]        = 1.20;
  AOEdgeFade.strength[eid]         = 0.25;
  AOEdgeFade.aspectCompensate[eid] = 1;

  // AOBilateral
  AOBilateral.depthSigma[eid]   = 0.02;
  AOBilateral.normalSigma[eid]  = 0.10;
  AOBilateral.lumaSigma[eid]    = 0.05;
  AOBilateral.spatialSigma[eid] = 2.0;
  AOBilateral.kernelRadius[eid] = 4;
  AOBilateral.passes[eid]       = 1;
  AOBilateral.enabled[eid]      = 0;

  // AODenoiser
  AODenoiser.enabled[eid]         = 1;
  AODenoiser.spatialPasses[eid]   = 2;
  AODenoiser.temporalPasses[eid]  = 1;
  AODenoiser.blendStrength[eid]   = 0.15;
  AODenoiser.noiseThreshold[eid]  = 0.10;
  AODenoiser.minVariance[eid]     = 0.001;
  AODenoiser.preserveEdges[eid]   = 1;
  AODenoiser.edgeSharpness[eid]   = 0.75;
  AODenoiser.lastDenoiseMs[eid]   = 0;

  // AOResidency
  AOResidency.state[eid]             = AO_RESIDENCY.UNLOADED;
  AOResidency.bytesAllocated[eid]    = 0;
  AOResidency.bytesPeak[eid]         = 0;
  AOResidency.lastResidentFrame[eid] = 0;
  AOResidency.lastEvictFrame[eid]    = 0;
  AOResidency.evictAfterFrames[eid]  = 240;
  AOResidency.pinned[eid]            = 0;

  // AOLeak
  AOLeak.enabled[eid]            = 1;
  AOLeak.magnitude[eid]          = 0.0;
  AOLeak.threshold[eid]          = 0.35;
  AOLeak.correctionFactor[eid]   = 1.0;
  AOLeak.detectionCount[eid]     = 0;
  AOLeak.lastDetectionFrame[eid] = 0;
  AOLeak.smoothing[eid]          = 0.85;
  AOLeak.valid[eid]              = 0;

  // AOStyle
  AOStyle.style[eid]            = spec.style !== undefined ? spec.style : AO_STYLE.CEL_SOFT;
  AOStyle.celBandCount[eid]     = 4;
  AOStyle.celSoftness[eid]      = 0.10;
  AOStyle.inkEnabled[eid]       = 0;
  AOStyle.inkThickness[eid]     = 0.003;
  AOStyle.inkStrength[eid]      = 0.35;
  AOStyle.inkColorR[eid]        = 0.05;
  AOStyle.inkColorG[eid]        = 0.05;
  AOStyle.inkColorB[eid]        = 0.08;
  AOStyle.overallIntensity[eid] = 1.0;
  AOStyle.ambientBoost[eid]     = 0.0;
  AOStyle.tintR[eid]            = 0.20;
  AOStyle.tintG[eid]            = 0.22;
  AOStyle.tintB[eid]            = 0.28;
  AOStyle.enabled[eid]          = 1;

  // Attach all components.
  addComponent(world, eid, AOVolumeRef);
  addComponent(world, eid, AOSampling);
  addComponent(world, eid, AOKernel);
  addComponent(world, eid, AOHistory);
  addComponent(world, eid, AOBlur);
  addComponent(world, eid, AOQuality);
  addComponent(world, eid, AOState);
  addComponent(world, eid, AOBudget);
  addComponent(world, eid, AOTemporalAccumulator);
  addComponent(world, eid, AODither);
  addComponent(world, eid, AOCelBands);
  addComponent(world, eid, AOInkOutline);
  addComponent(world, eid, AOEdgeFade);
  addComponent(world, eid, AOBilateral);
  addComponent(world, eid, AODenoiser);
  addComponent(world, eid, AOResidency);
  addComponent(world, eid, AOLeak);
  addComponent(world, eid, AOStyle);
}

/* ------------------------------------------------------------------ */
/* 7. PUBLIC SPAWN FUNCTIONS                                          */
/* ------------------------------------------------------------------ */

/**
 * Spawns an AO volume with base components.
 */
export function spawnAOVolume(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);
  _attachAOBase(w, eid, spec);
  return eid;
}

/**
 * Spawns a screen-space AO controller (attached to one entity per frame).
 */
export function spawnAOScreenSpace(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOScreenSpace.screenWidth[eid]  = spec.screenWidth !== undefined ? spec.screenWidth : MAX_AO_SCREEN_WIDTH;
  AOScreenSpace.screenHeight[eid] = spec.screenHeight !== undefined ? spec.screenHeight : MAX_AO_SCREEN_HEIGHT;
  AOScreenSpace.aoWidth[eid]      = spec.aoWidth !== undefined ? spec.aoWidth : Math.round((spec.screenWidth || MAX_AO_SCREEN_WIDTH) * 0.5);
  AOScreenSpace.aoHeight[eid]     = spec.aoHeight !== undefined ? spec.aoHeight : Math.round((spec.screenHeight || MAX_AO_SCREEN_HEIGHT) * 0.5);
  AOScreenSpace.depthWidth[eid]   = spec.depthWidth !== undefined ? spec.depthWidth : (spec.screenWidth || MAX_AO_SCREEN_WIDTH);
  AOScreenSpace.depthHeight[eid]  = spec.depthHeight !== undefined ? spec.depthHeight : (spec.screenHeight || MAX_AO_SCREEN_HEIGHT);
  AOScreenSpace.tanHalfFov[eid]   = spec.tanHalfFov !== undefined ? spec.tanHalfFov : 0.577;
  AOScreenSpace.aspect[eid]       = spec.aspect !== undefined ? spec.aspect : 1.0;
  AOScreenSpace.nearPlane[eid]    = spec.nearPlane !== undefined ? spec.nearPlane : 0.1;
  AOScreenSpace.farPlane[eid]     = spec.farPlane !== undefined ? spec.farPlane : 1000.0;
  AOScreenSpace.projection00[eid] = spec.projection00 !== undefined ? spec.projection00 : 1.0;
  AOScreenSpace.projection11[eid] = spec.projection11 !== undefined ? spec.projection11 : 1.0;
  AOScreenSpace.inverseProjection00[eid] = spec.inverseProjection00 !== undefined ? spec.inverseProjection00 : 1.0;
  AOScreenSpace.inverseProjection11[eid] = spec.inverseProjection11 !== undefined ? spec.inverseProjection11 : 1.0;
  AOScreenSpace.offsetScale[eid]  = spec.offsetScale !== undefined ? spec.offsetScale : 1.0;
  AOScreenSpace.offsetBias[eid]   = spec.offsetBias !== undefined ? spec.offsetBias : 0.02;
  AOScreenSpace.screenSpaceValid[eid] = 1;

  addComponent(w, eid, AOScreenSpace);
  return eid;
}

/**
 * Spawns a contact-shadow AO entity.
 */
export function spawnAOContactShadow(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOContactShadow.enabled[eid]     = 1;
  AOContactShadow.rayCount[eid]    = spec.rayCount !== undefined ? spec.rayCount : 8;
  AOContactShadow.maxDistance[eid] = spec.maxDistance !== undefined ? spec.maxDistance : 2.0;
  AOContactShadow.stepCount[eid]   = spec.stepCount !== undefined ? spec.stepCount : 12;
  AOContactShadow.thickness[eid]   = spec.thickness !== undefined ? spec.thickness : 0.1;
  AOContactShadow.bias[eid]        = spec.bias !== undefined ? spec.bias : 0.02;
  AOContactShadow.jitter[eid]      = spec.jitter !== undefined ? spec.jitter : 0.5;
  AOContactShadow.fadeNear[eid]    = spec.fadeNear !== undefined ? spec.fadeNear : 0.5;
  AOContactShadow.fadeFar[eid]     = spec.fadeFar !== undefined ? spec.fadeFar : 8.0;
  AOContactShadow.strength[eid]    = spec.strength !== undefined ? spec.strength : 0.6;
  AOContactShadow.tintR[eid]       = spec.tint ? spec.tint[0] : 0.10;
  AOContactShadow.tintG[eid]       = spec.tint ? spec.tint[1] : 0.10;
  AOContactShadow.tintB[eid]       = spec.tint ? spec.tint[2] : 0.15;

  addComponent(w, eid, AOContactShadow);
  return eid;
}

/**
 * Spawns a distance-field AO entity binding an SDF grid to a volume.
 */
export function spawnAODistanceField(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AODistanceField.enabled[eid]         = 1;
  AODistanceField.fieldEid[eid]        = spec.fieldEid !== undefined ? spec.fieldEid : -1;
  AODistanceField.maxDistance[eid]     = spec.maxDistance !== undefined ? spec.maxDistance : 8.0;
  AODistanceField.stepScale[eid]       = spec.stepScale !== undefined ? spec.stepScale : 1.0;
  AODistanceField.bias[eid]            = spec.bias !== undefined ? spec.bias : 0.02;
  AODistanceField.useGradientBias[eid] = spec.useGradientBias !== false ? 1 : 0;
  AODistanceField.gradientScale[eid]   = spec.gradientScale !== undefined ? spec.gradientScale : 1.0;
  AODistanceField.coneAngle[eid]       = spec.coneAngle !== undefined ? spec.coneAngle : Math.PI / 6;
  AODistanceField.coneSteps[eid]       = spec.coneSteps !== undefined ? spec.coneSteps : 12;
  AODistanceField.fadeEdge[eid]        = spec.fadeEdge !== undefined ? spec.fadeEdge : 0.15;

  addComponent(w, eid, AODistanceField);
  return eid;
}

/**
 * Spawns an indoor AO volume with interior-specific parameters.
 */
export function spawnAOIndoorVolume(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;

  const volumeEid = spawnAOVolume(w, Object.assign({}, spec, { indoorFactor: 1.0 }));

  AOIndoorVolume.volumeEid[volumeEid]      = volumeEid;
  AOIndoorVolume.ceilingAO[volumeEid]      = spec.ceilingAO !== undefined ? spec.ceilingAO : 0.35;
  AOIndoorVolume.floorAO[volumeEid]        = spec.floorAO !== undefined ? spec.floorAO : 0.55;
  AOIndoorVolume.wallAO[volumeEid]         = spec.wallAO !== undefined ? spec.wallAO : 0.75;
  AOIndoorVolume.cornerBoost[volumeEid]    = spec.cornerBoost !== undefined ? spec.cornerBoost : 0.35;
  AOIndoorVolume.contactAO[volumeEid]      = spec.contactAO !== undefined ? spec.contactAO : 0.50;
  AOIndoorVolume.curtainAO[volumeEid]      = spec.curtainAO !== undefined ? spec.curtainAO : 0.30;
  AOIndoorVolume.ceilingAOColorR[volumeEid] = spec.ceilingColor ? spec.ceilingColor[0] : 0.20;
  AOIndoorVolume.ceilingAOColorG[volumeEid] = spec.ceilingColor ? spec.ceilingColor[1] : 0.20;
  AOIndoorVolume.ceilingAOColorB[volumeEid] = spec.ceilingColor ? spec.ceilingColor[2] : 0.22;
  AOIndoorVolume.floorAOColorR[volumeEid]   = spec.floorColor ? spec.floorColor[0] : 0.15;
  AOIndoorVolume.floorAOColorG[volumeEid]   = spec.floorColor ? spec.floorColor[1] : 0.14;
  AOIndoorVolume.floorAOColorB[volumeEid]   = spec.floorColor ? spec.floorColor[2] : 0.12;
  AOIndoorVolume.enabled[volumeEid]         = 1;

  addComponent(w, volumeEid, AOIndoorVolume);
  return volumeEid;
}

/**
 * Spawns an outdoor AO volume with exterior-specific parameters.
 */
export function spawnAOOutdoorVolume(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;

  const volumeEid = spawnAOVolume(w, Object.assign({}, spec, { indoorFactor: 0.0 }));

  AOOutdoorVolume.volumeEid[volumeEid]     = volumeEid;
  AOOutdoorVolume.groundAO[volumeEid]      = spec.groundAO !== undefined ? spec.groundAO : 0.65;
  AOOutdoorVolume.skyAO[volumeEid]         = spec.skyAO !== undefined ? spec.skyAO : 0.20;
  AOOutdoorVolume.horizonAO[volumeEid]     = spec.horizonAO !== undefined ? spec.horizonAO : 0.35;
  AOOutdoorVolume.canopyAO[volumeEid]      = spec.canopyAO !== undefined ? spec.canopyAO : 0.55;
  AOOutdoorVolume.waterAO[volumeEid]       = spec.waterAO !== undefined ? spec.waterAO : 0.40;
  AOOutdoorVolume.biomeDensity[volumeEid]  = spec.biomeDensity !== undefined ? spec.biomeDensity : 1.0;
  AOOutdoorVolume.groundAOColorR[volumeEid]= spec.groundColor ? spec.groundColor[0] : 0.15;
  AOOutdoorVolume.groundAOColorG[volumeEid]= spec.groundColor ? spec.groundColor[1] : 0.14;
  AOOutdoorVolume.groundAOColorB[volumeEid]= spec.groundColor ? spec.groundColor[2] : 0.12;
  AOOutdoorVolume.skyAOColorR[volumeEid]   = spec.skyColor ? spec.skyColor[0] : 0.20;
  AOOutdoorVolume.skyAOColorG[volumeEid]   = spec.skyColor ? spec.skyColor[1] : 0.22;
  AOOutdoorVolume.skyAOColorB[volumeEid]   = spec.skyColor ? spec.skyColor[2] : 0.28;
  AOOutdoorVolume.enabled[volumeEid]       = 1;

  addComponent(w, volumeEid, AOOutdoorVolume);
  return volumeEid;
}

/**
 * Spawns an anime cel-band AO controller.
 */
export function spawnAOCelBandController(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOCelBands.enabled[eid]      = 1;
  AOCelBands.bandCount[eid]    = spec.bandCount !== undefined ? spec.bandCount : 4;
  AOCelBands.bandSoftness[eid] = spec.bandSoftness !== undefined ? spec.bandSoftness : 0.10;
  AOCelBands.bandBias[eid]     = spec.bandBias !== undefined ? spec.bandBias : 0.0;
  AOCelBands.rimBoost[eid]     = spec.rimBoost !== undefined ? spec.rimBoost : 0.0;

  if (spec.band0) { AOCelBands.band0R[eid] = spec.band0[0]; AOCelBands.band0G[eid] = spec.band0[1]; AOCelBands.band0B[eid] = spec.band0[2]; }
  if (spec.band1) { AOCelBands.band1R[eid] = spec.band1[0]; AOCelBands.band1G[eid] = spec.band1[1]; AOCelBands.band1B[eid] = spec.band1[2]; }
  if (spec.band2) { AOCelBands.band2R[eid] = spec.band2[0]; AOCelBands.band2G[eid] = spec.band2[1]; AOCelBands.band2B[eid] = spec.band2[2]; }
  if (spec.band3) { AOCelBands.band3R[eid] = spec.band3[0]; AOCelBands.band3G[eid] = spec.band3[1]; AOCelBands.band3B[eid] = spec.band3[2]; }

  if (spec.shadowTint) {
    AOCelBands.shadowTintR[eid] = spec.shadowTint[0];
    AOCelBands.shadowTintG[eid] = spec.shadowTint[1];
    AOCelBands.shadowTintB[eid] = spec.shadowTint[2];
  }

  addComponent(w, eid, AOCelBands);
  return eid;
}

/**
 * Spawns an anime ink-outline AO controller.
 */
export function spawnAOInkOutline(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOInkOutline.enabled[eid]         = 1;
  AOInkOutline.thickness[eid]       = spec.thickness !== undefined ? spec.thickness : 0.003;
  AOInkOutline.strength[eid]        = spec.strength !== undefined ? spec.strength : 0.35;
  AOInkOutline.softness[eid]        = spec.softness !== undefined ? spec.softness : 0.15;
  AOInkOutline.depthThreshold[eid]  = spec.depthThreshold !== undefined ? spec.depthThreshold : 0.03;
  AOInkOutline.normalThreshold[eid] = spec.normalThreshold !== undefined ? spec.normalThreshold : 0.20;
  if (spec.color) {
    AOInkOutline.colorR[eid] = spec.color[0];
    AOInkOutline.colorG[eid] = spec.color[1];
    AOInkOutline.colorB[eid] = spec.color[2];
  }
  AOInkOutline.widthLumaBias[eid]     = spec.widthLumaBias !== undefined ? spec.widthLumaBias : 0.5;
  AOInkOutline.widthDistanceBias[eid] = spec.widthDistanceBias !== undefined ? spec.widthDistanceBias : 1.0;

  addComponent(w, eid, AOInkOutline);
  return eid;
}

/**
 * Spawns a bilateral blur controller.
 */
export function spawnAOBilateral(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AOBilateral.enabled[eid]        = 1;
  AOBilateral.depthSigma[eid]     = spec.depthSigma !== undefined ? spec.depthSigma : 0.02;
  AOBilateral.normalSigma[eid]    = spec.normalSigma !== undefined ? spec.normalSigma : 0.10;
  AOBilateral.lumaSigma[eid]      = spec.lumaSigma !== undefined ? spec.lumaSigma : 0.05;
  AOBilateral.spatialSigma[eid]   = spec.spatialSigma !== undefined ? spec.spatialSigma : 2.0;
  AOBilateral.kernelRadius[eid]   = spec.kernelRadius !== undefined ? spec.kernelRadius : 4;
  AOBilateral.passes[eid]         = spec.passes !== undefined ? spec.passes : 1;

  addComponent(w, eid, AOBilateral);
  return eid;
}

/**
 * Spawns a spatial+temporal denoiser controller.
 */
export function spawnAODenoiser(world, spec = {}) {
  const w = world || getAOWorld();
  if (!w) return -1;
  const eid = addEntity(w);

  AODenoiser.enabled[eid]         = 1;
  AODenoiser.spatialPasses[eid]   = spec.spatialPasses !== undefined ? spec.spatialPasses : 2;
  AODenoiser.temporalPasses[eid]  = spec.temporalPasses !== undefined ? spec.temporalPasses : 1;
  AODenoiser.blendStrength[eid]   = spec.blendStrength !== undefined ? spec.blendStrength : 0.15;
  AODenoiser.noiseThreshold[eid]  = spec.noiseThreshold !== undefined ? spec.noiseThreshold : 0.10;
  AODenoiser.minVariance[eid]     = spec.minVariance !== undefined ? spec.minVariance : 0.001;
  AODenoiser.preserveEdges[eid]   = spec.preserveEdges !== false ? 1 : 0;
  AODenoiser.edgeSharpness[eid]   = spec.edgeSharpness !== undefined ? spec.edgeSharpness : 0.75;

  addComponent(w, eid, AODenoiser);
  return eid;
}

/* ------------------------------------------------------------------ */
/* 8. UTILITY HELPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Marks an AO volume as dirty so downstream systems re-bake it.
 */
export function markAODirty(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  AOState.prevState[eid] = AOState.state[eid];
  AOState.state[eid] = AO_STATE.DIRTY;
  AOState.lastStateFrame[eid] = 0;
  return true;
}

/**
 * Clears the dirty flag after a successful bake.
 */
export function clearAODirty(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  AOState.prevState[eid] = AOState.state[eid];
  AOState.state[eid] = AO_STATE.READY;
  AOState.lastBakeFrame[eid] = 0;
  AOHistory.historyValid[eid] = 1;
  return true;
}

/**
 * Sets AO quality on a volume.
 */
export function setAOQuality(eid, quality) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (quality < 0 || quality >= AO_QUALITY.COUNT) return false;
  AOQuality.tier[eid] = quality;
  AOVolumeRef.quality[eid] = quality;
  markAODirty(eid);
  return true;
}

/**
 * Sets AO method on a volume.
 */
export function setAOMethod(eid, method) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (method < 0 || method >= AO_METHOD.COUNT) return false;
  AOVolumeRef.method[eid] = method;
  markAODirty(eid);
  return true;
}

/**
 * Sets the AO kernel sample count on a volume (clamped to the shared kernel).
 */
export function setAOKernelSampleCount(eid, count) {
  if (typeof eid !== 'number' || eid < 0) return false;
  AOSampling.sampleCount[eid] = Math.max(1, Math.min(count, MAX_AO_KERNEL_SAMPLES));
  markAODirty(eid);
  return true;
}

/**
 * Enables anime cel-band AO on a volume.
 */
export function enableAOCelBands(eid, bandCount, softness) {
  if (typeof eid !== 'number' || eid < 0) return false;
  AOCelBands.enabled[eid] = 1;
  if (typeof bandCount === 'number') AOCelBands.bandCount[eid] = Math.max(2, Math.min(MAX_AO_CEL_BANDS, bandCount));
  if (typeof softness === 'number') AOCelBands.bandSoftness[eid] = Math.max(0, Math.min(1, softness));
  return true;
}

/**
 * Enables anime ink-outline AO on a volume.
 */
export function enableAOInkOutline(eid, thickness, strength) {
  if (typeof eid !== 'number' || eid < 0) return false;
  AOInkOutline.enabled[eid] = 1;
  if (typeof thickness === 'number') AOInkOutline.thickness[eid] = thickness;
  if (typeof strength === 'number') AOInkOutline.strength[eid] = strength;
  return true;
}

/**
 * Sets the AO blur mode on a volume.
 */
export function setAOBlurMode(eid, mode) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (mode < 0 || mode >= AO_BLUR_MODE.COUNT) return false;
  AOBlur.mode[eid] = mode;
  return true;
}

/**
 * Sets the AO temporal mode on a volume.
 */
export function setAOTemporalMode(eid, mode, blendFactor) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (mode < 0 || mode >= AO_TEMPORAL_MODE.COUNT) return false;
  AOTemporalAccumulator.mode[eid] = mode;
  if (typeof blendFactor === 'number') {
    AOTemporalAccumulator.blendFactor[eid] = Math.max(0.01, Math.min(0.99, blendFactor));
  }
  return true;
}

/**
 * Sets the anime AO style on a volume and applies the corresponding preset
 * to cel bands + ink outline.
 */
export function setAOStyle(eid, style) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (style < 0 || style >= AO_STYLE.COUNT) return false;
  AOVolumeRef.style[eid] = style;
  AOStyle.style[eid] = style;

  switch (style) {
    case AO_STYLE.REALISTIC:
      AOCelBands.enabled[eid] = 0;
      AOInkOutline.enabled[eid] = 0;
      AOStyle.celSoftness[eid] = 1.0;
      break;
    case AO_STYLE.CEL_HARD:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 2;
      AOCelBands.bandSoftness[eid] = 0.02;
      AOInkOutline.enabled[eid] = 0;
      break;
    case AO_STYLE.CEL_SOFT:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 3;
      AOCelBands.bandSoftness[eid] = 0.15;
      AOInkOutline.enabled[eid] = 0;
      break;
    case AO_STYLE.INK_LINE:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 2;
      AOCelBands.bandSoftness[eid] = 0.05;
      AOInkOutline.enabled[eid] = 1;
      AOInkOutline.thickness[eid] = 0.0025;
      AOInkOutline.strength[eid] = 0.40;
      break;
    case AO_STYLE.POSTERIZED:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 4;
      AOCelBands.bandSoftness[eid] = 0.02;
      AOInkOutline.enabled[eid] = 0;
      break;
    case AO_STYLE.PASTEL:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 3;
      AOCelBands.bandSoftness[eid] = 0.40;
      AOInkOutline.enabled[eid] = 0;
      break;
    case AO_STYLE.DENSE_LINE:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 3;
      AOCelBands.bandSoftness[eid] = 0.05;
      AOInkOutline.enabled[eid] = 1;
      AOInkOutline.thickness[eid] = 0.004;
      AOInkOutline.strength[eid] = 0.55;
      break;
    case AO_STYLE.WISP:
      AOCelBands.enabled[eid] = 1;
      AOCelBands.bandCount[eid] = 5;
      AOCelBands.bandSoftness[eid] = 0.30;
      AOInkOutline.enabled[eid] = 0;
      break;
    default:
      break;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createAOWorld() {
  return createWorld({
    components: AO_COMPONENTS,
    time: {
      delta: 0,
      elapsed: 0,
      then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
    },
  });
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_AO_KERNEL_SAMPLES,
  MAX_AO_HISTORY_FRAMES,
  MAX_AO_ASYNC_UPDATES,
  MAX_AO_SCREEN_WIDTH,
  MAX_AO_SCREEN_HEIGHT,
  MAX_AO_CEL_BANDS,

  // Components
  AOVolumeRef,
  AOSampling,
  AOKernel,
  AOHistory,
  AOBlur,
  AOQuality,
  AOState,
  AOBudget,
  AOScreenSpace,
  AOContactShadow,
  AODistanceField,
  AOIndoorVolume,
  AOOutdoorVolume,
  AOTemporalAccumulator,
  AODither,
  AOCelBands,
  AOInkOutline,
  AOEdgeFade,
  AOBilateral,
  AODenoiser,
  AOAsync,
  AOResidency,
  AOLeak,
  AOStyle,
  AO_COMPONENTS,

  // Enums
  AO_METHOD,
  AO_METHOD_NAME,
  AO_QUALITY,
  AO_QUALITY_NAME,
  AO_STATE,
  AO_STATE_NAME,
  AO_BLUR_MODE,
  AO_BLUR_MODE_NAME,
  AO_TEMPORAL_MODE,
  AO_TEMPORAL_MODE_NAME,
  AO_DITHER,
  AO_DITHER_NAME,
  AO_STYLE,
  AO_STYLE_NAME,
  AO_RESIDENCY,
  AO_RESIDENCY_NAME,
  AO_EDGE,
  AO_EDGE_NAME,

  // World
  getAOWorld,
  createAOWorld,

  // Kernel
  generateAOKernel,

  // Spawn
  spawnAOVolume,
  spawnAOScreenSpace,
  spawnAOContactShadow,
  spawnAODistanceField,
  spawnAOIndoorVolume,
  spawnAOOutdoorVolume,
  spawnAOCelBandController,
  spawnAOInkOutline,
  spawnAOBilateral,
  spawnAODenoiser,

  // Utilities
  markAODirty,
  clearAODirty,
  setAOQuality,
  setAOMethod,
  setAOKernelSampleCount,
  enableAOCelBands,
  enableAOInkOutline,
  setAOBlurMode,
  setAOTemporalMode,
  setAOStyle,
};

export default _defaultExport;