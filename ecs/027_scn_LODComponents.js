// File : 027
// name : src/ecs/027_scn_LODComponents.js
// description : LOD (level-of-detail) SoA component module for the scene ECS
//               world of the anime lighting stack on Android mobile. Declares
//               every LOD level, distance band, screen-size threshold,
//               hysteresis guard, group binding, transition state, and
//               budget-aware override that the lighting stack needs — as
//               fixed-capacity typed arrays sized once to
//               MAX_ENTITIES = 100000.
//
//               Provides the fast helpers that every LOD-consuming system
//               runs on the hot path:
//                 • assignLODLevel       — set the current LOD level for one
//                                          entity without hysteresis
//                 • computeScreenSize    — estimate screen-space size from
//                                          distance + FOV + bounding radius
//                 • evaluateLOD          — pick the target LOD level from
//                                          distance bands and hysteresis
//                 • applyHysteresis      — debounce LOD transitions to prevent
//                                          flicker
//                 • selectLODGeometry    — resolve the geometry id for the
//                                          current LOD level
//                 • switchLODLevel       — commit a level change with the
//                                          transition state machine
//                 • interpolateLODTransition — smooth cross-fade between two
//                                          LOD levels (impostor blending)
//                 • getLODBias           — read the global/per-entity LOD bias
//                 • setLODBias           — write the global/per-entity LOD bias
//                 • evaluateGroupLOD     — evaluate every member of an LOD
//                                          group as one unit
//                 • updateLODTransitions — advance every active transition
//                                          once per frame
//                 • computeDistanceBand  — classify a distance into one of
//                                          the entity's LOD distance bands
//                 • refreshLODForLight   — convenience for lights
//                 • refreshLODForShadowCaster — convenience for casters
//                 • refreshLODForGIProbe — convenience for GI probes
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no runtime resizes.
//                 • Up to MAX_LOD_LEVELS (8) discrete LOD levels per entity,
//                   each with its own distance band, screen-size threshold,
//                   geometry id, and cost estimate.
//                 • Hysteresis bands guard every transition: a level changes
//                   only when the entity crosses the band edge by more than
//                   the hysteresis fraction, preventing edge flicker.
//                 • Cross-fade transitions: the impostor/proxy for the
//                   previous LOD level is retained for LOD_TRANSITION_FRAMES
//                   frames while the new level fades in.
//                 • Global LOD bias scales every distance band by a scalar
//                   so quality controllers can nudge the whole scene's LOD
//                   distribution without re-authoring per-entity bands.
//                 • Per-entity bias overrides the global bias for hero
//                   entities (character, sun, near camera).
//                 • Group evaluation: an LOD group shares a single level
//                   across all members — used for shadow caster clusters,
//                   GI probe grids, vegetation patches.
//                 • Budget-aware: LODBudget tracks the aggregate cost of
//                   visible levels so the quality controller can force a
//                   downgrade if the budget is exceeded.
//                 • Zero allocations on the hot path — every helper works
//                   directly on the SoA arrays; scratch vectors are
//                   module-level and reused.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — LightRef, LightBudget
//                 • 003_lgt_ShadowComponents.js     — ShadowCasterRef,
//                                                     ShadowBudget
//                 • 004_lgt_GIComponents.js         — GIProbeRef, GIBudget
//                 • 005_lgt_AOComponents.js         — AOVolumeRef, AOBudget
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — visibility tags
//                 • 015_scn_Relations.js            — group edges
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime records
//                 • 025_scn_SpatialComponents.js    — Sphere / AABB radius
//                 • 026_scn_TransformComponents.js  — world position / scale
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every light, shadow caster, GI probe, AO
//            volume, and mesh in the anime lighting stack picks its LOD
//            level deterministically with anti-flicker hysteresis, smooth
//            transitions, budget-aware override, and zero allocations on
//            the hot path — so the anime look is preserved at every distance
//            band and on every Android GPU tier.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getDefaultQualityController,
} from '../core/017_rnd_QualityConfig.js';

import {
  MAX_ENTITIES,
  Transform,
  LightRef,
  LightBudget as LightBudgetComponent,
} from './002_lgt_LightComponents.js';

import {
  ShadowCasterRef,
  ShadowBudget,
} from './003_lgt_ShadowComponents.js';

import {
  GIProbeRef,
  GIBudget,
} from './004_lgt_GIComponents.js';

import {
  AOVolumeRef,
  AOBudget,
} from './005_lgt_AOComponents.js';

import {
  getECSWorld,
  entityAlive,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  COMPONENT_TYPE_ID,
} from './013_scn_ComponentTypes.js';

import {
  TAG,
  TAG2,
  FDIRTY,
  markFrameDirty,
  clearFrameDirty,
  isFrameDirty,
  EntityTag,
} from './014_scn_Tags.js';

import {
  Parent,
  Children,
  MAX_CHILDREN_PER_ENTITY,
  NULL_ENTITY,
  bindLODGroup,
  forEachReferenceOfKind,
  REF,
} from './015_scn_Relations.js';

import {
  Sphere,
  AABB,
  VisibilityState,
} from './025_scn_SpatialComponents.js';

import {
  TransformWorld,
  getWorldMaxScale,
} from './026_scn_TransformComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of discrete LOD levels per entity.
 */
export const MAX_LOD_LEVELS = 8;

/**
 * Maximum number of members in one LOD group.
 */
export const MAX_LOD_GROUP_MEMBERS =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 48 :
                                 32;

/**
 * Number of frames a cross-fade transition between two LOD levels runs.
 */
export const LOD_TRANSITION_FRAMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 12 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 10 :
                                 8;

/**
 * Default hysteresis fraction. A level changes only when the entity
 * crosses the band edge by more than this fraction of the band width.
 */
export const DEFAULT_HYSTERESIS = 0.10;

/**
 * LOD level index sentinel — "no LOD" or "auto".
 */
export const LOD_AUTO = -1;
export const LOD_NONE = -2;

/**
 * LOD state machine.
 */
export const LOD_STATE = Object.freeze({
  NONE:        0,
  STABLE:      1,
  TRANSITION:  2,
  FORCED:      3,
  SUSPENDED:   4,
  DISABLED:    5,
  COUNT:       6,
});

export const LOD_STATE_NAME = Object.freeze([
  'none',
  'stable',
  'transition',
  'forced',
  'suspended',
  'disabled',
]);

/**
 * LOD geometry kinds — used by the geometry resolver to pick the right
 * pool.
 */
export const LOD_GEOMETRY_KIND = Object.freeze({
  NONE:        0,
  FULL:        1,
  HALF:        2,
  QUARTER:     3,
  IMPOSTOR:    4,
  PROXY:       5,
  POINT:       6,
  COUNT:       7,
});

export const LOD_GEOMETRY_KIND_NAME = Object.freeze([
  'none',
  'full',
  'half',
  'quarter',
  'impostor',
  'proxy',
  'point',
]);

/**
 * LOD flag bitmask.
 */
export const LOD_FLAG = Object.freeze({
  NONE:            0,
  ENABLED:         1 << 0,
  GROUP_ANCHOR:    1 << 1,
  GROUP_MEMBER:    1 << 2,
  HAS_IMPOSTOR:    1 << 3,
  HAS_PROXY:       1 << 4,
  BUDGET_AWARE:    1 << 5,
  HERO_ENTITY:     1 << 6,
  SUSPENDED:       1 << 7,
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const LODState = {
  frame:               0,
  globalBias:          1.0,
  totalEvaluations:    0,
  totalLevelSwitches:  0,
  totalTransitions:    0,
  totalTransitionAdv:  0,
  totalHysteresisHolds:0,
  totalBudgetDowngrades:0,
  totalGroupEvaluations:0,
  activeTransitions:   0,
  peakActiveTransitions:0,
  lastEvalMs:          0,
  avgEvalMs:           0,
  lastSwitchMs:        0,
  avgSwitchMs:         0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.lod', {
        tag: BOUNDARY_TAG.GENERIC,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * LODLevel — per-entity current level + target level + resolution state.
 */
export const LODLevel = {
  currentLevel:    new Int8Array(MAX_ENTITIES).fill(-1),
  targetLevel:     new Int8Array(MAX_ENTITIES).fill(-1),
  previousLevel:   new Int8Array(MAX_ENTITIES).fill(-1),
  levelCount:      new Uint8Array(MAX_ENTITIES),
  state:           new Uint8Array(MAX_ENTITIES),
  flags:           new Uint16Array(MAX_ENTITIES),
  lastSwitchFrame: new Uint32Array(MAX_ENTITIES),
  lastEvalFrame:   new Uint32Array(MAX_ENTITIES),
  switchCount:     new Uint16Array(MAX_ENTITIES),
};

/**
 * LODDistance — per-entity distance band table.
 * Distances are stored per (entity, level) as a flat array.
 * `distance[i * MAX_LOD_LEVELS + level]` is the distance at which the
 * entity switches FROM `level` TO `level + 1`.
 */
export const LODDistance = {
  distances:        new Float32Array(MAX_ENTITIES * MAX_LOD_LEVELS),
  screenThresholds: new Float32Array(MAX_ENTITIES * MAX_LOD_LEVELS),
  hysteresis:       new Float32Array(MAX_ENTITIES),
  currentDistance:  new Float32Array(MAX_ENTITIES),
  currentDistanceSq:new Float32Array(MAX_ENTITIES),
  currentScreenSize:new Float32Array(MAX_ENTITIES),
  distanceBand:     new Int8Array(MAX_ENTITIES),
};

/**
 * LODGeometry — per-entity geometry id per LOD level.
 * `geometryIds[i * MAX_LOD_LEVELS + level]` is the pool geometry id.
 */
export const LODGeometry = {
  geometryIds:  new Int32Array(MAX_ENTITIES * MAX_LOD_LEVELS).fill(-1),
  geometryKind: new Uint8Array(MAX_ENTITIES * MAX_LOD_LEVELS),
  triangleCount:new Uint32Array(MAX_ENTITIES * MAX_LOD_LEVELS),
  vertexCount:  new Uint32Array(MAX_ENTITIES * MAX_LOD_LEVELS),
  materialId:   new Int32Array(MAX_ENTITIES * MAX_LOD_LEVELS).fill(-1),
};

/**
 * LODBias — per-entity bias + global-aware scaling.
 */
export const LODBias = {
  entityBias:  new Float32Array(MAX_ENTITIES).fill(1.0),
  useGlobal:   new Uint8Array(MAX_ENTITIES).fill(1),
  minLevel:    new Int8Array(MAX_ENTITIES),
  maxLevel:    new Int8Array(MAX_ENTITIES),
  biasDirty:   new Uint8Array(MAX_ENTITIES),
};

/**
 * LODHysteresis — hysteresis state.
 * `previousDistance` and `previousScreenSize` are the values that were
 * used for the last level decision. They are compared against the
 * current values to enforce the hysteresis band.
 */
export const LODHysteresis = {
  previousDistance:   new Float32Array(MAX_ENTITIES),
  previousScreenSize: new Float32Array(MAX_ENTITIES),
  holdFrames:         new Uint8Array(MAX_ENTITIES),
  holdThreshold:      new Uint8Array(MAX_ENTITIES),
};

/**
 * LODGroup — group anchor + membership state.
 */
export const LODGroup = {
  anchorEid:      new Int32Array(MAX_ENTITIES).fill(-1),
  memberCount:    new Uint8Array(MAX_ENTITIES),
  groupLevel:     new Int8Array(MAX_ENTITIES),
  groupDistance:  new Float32Array(MAX_ENTITIES),
  groupRadius:    new Float32Array(MAX_ENTITIES),
  groupDirty:     new Uint8Array(MAX_ENTITIES),
  groupLevelCount:new Uint8Array(MAX_ENTITIES),
};

/**
 * LODMember — a member's index within its group.
 */
export const LODMember = {
  memberIndex:   new Uint8Array(MAX_ENTITIES),
  groupEid:      new Int32Array(MAX_ENTITIES).fill(-1),
  localWeight:   new Float32Array(MAX_ENTITIES).fill(1.0),
};

/**
 * LODTransition — the cross-fade state between two LOD levels.
 */
export const LODTransition = {
  fromLevel:     new Int8Array(MAX_ENTITIES),
  toLevel:       new Int8Array(MAX_ENTITIES),
  frameCount:    new Uint8Array(MAX_ENTITIES),
  maxFrames:     new Uint8Array(MAX_ENTITIES),
  alpha:         new Float32Array(MAX_ENTITIES),
  active:        new Uint8Array(MAX_ENTITIES),
  fadeMode:      new Uint8Array(MAX_ENTITIES),   // 0=linear 1=smooth 2=hard
};

/**
 * LODBudget — the aggregate cost and per-entity LOD cost estimate.
 */
export const LODBudget = {
  entityCost:      new Float32Array(MAX_ENTITIES),
  entityCostEma:   new Float32Array(MAX_ENTITIES),
  totalVisibleCost:new Float32Array(1),
  budgetCap:       new Float32Array(1),
  budgetExceeded:  new Uint8Array(1),
  forcedDowngrades:new Uint32Array(1),
};

/**
 * LODScreenSize — cached screen-space size per entity + thresholds.
 */
export const LODScreenSize = {
  screenSize:         new Float32Array(MAX_ENTITIES),
  screenSizePixels:   new Float32Array(MAX_ENTITIES),
  screenSizePercent:  new Float32Array(MAX_ENTITIES),
  pixelThreshold:     new Float32Array(MAX_ENTITIES),
  valid:              new Uint8Array(MAX_ENTITIES),
};

/**
 * LODImpostor — impostor binding per entity.
 */
export const LODImpostor = {
  impostorEid:     new Int32Array(MAX_ENTITIES).fill(-1),
  impostorRadius:  new Float32Array(MAX_ENTITIES),
  impostorHeight:  new Float32Array(MAX_ENTITIES),
  billboardMode:   new Uint8Array(MAX_ENTITIES),
  valid:           new Uint8Array(MAX_ENTITIES),
};

/**
 * LODProxy — proxy geometry binding per entity.
 */
export const LODProxy = {
  proxyEid:       new Int32Array(MAX_ENTITIES).fill(-1),
  proxyRadius:    new Float32Array(MAX_ENTITIES),
  proxyKind:      new Uint8Array(MAX_ENTITIES),
  valid:          new Uint8Array(MAX_ENTITIES),
};

/**
 * LODDebug — debug metadata for the HUD.
 */
export const LODDebug = {
  lastDecisionFrame: new Uint32Array(MAX_ENTITIES),
  lastDecisionLevel: new Int8Array(MAX_ENTITIES),
  lastDecisionReason:new Uint8Array(MAX_ENTITIES),   // 0=distance 1=screen 2=budget 3=forced
  lastDecisionBias:  new Float32Array(MAX_ENTITIES),
};

/**
 * LODStats — aggregate per-frame LOD statistics.
 */
export const LODStats = {
  visibleCount:    new Uint32Array(1),
  levelHistogram:  new Uint32Array(MAX_LOD_LEVELS),
  transitionsActive: new Uint32Array(1),
  evaluationsThisFrame: new Uint32Array(1),
  switchesThisFrame: new Uint32Array(1),
  hysteresisHolds:  new Uint32Array(1),
};

/**
 * LOD component bundle for bitECS createWorld.
 */
export const LOD_COMPONENTS = Object.freeze({
  LODLevel,
  LODDistance,
  LODGeometry,
  LODBias,
  LODHysteresis,
  LODGroup,
  LODMember,
  LODTransition,
  LODBudget,
  LODScreenSize,
  LODImpostor,
  LODProxy,
  LODDebug,
  LODStats,
});

/* ------------------------------------------------------------------ */
/* 3. DEFAULT LEVEL TABLES                                            */
/* ------------------------------------------------------------------ */

/**
 * Default distance bands (in world units) for a 4-level LOD:
 *   LOD 0 (full)    : 0   .. 20
 *   LOD 1 (half)    : 20  .. 60
 *   LOD 2 (quarter) : 60  .. 120
 *   LOD 3 (impostor): 120 .. ∞
 */
export const DEFAULT_DISTANCE_BANDS = Object.freeze([
  0.0,
  20.0,
  60.0,
  120.0,
]);

/**
 * Default screen-size thresholds (percentage of screen height) for the
 * same 4-level LOD.
 */
export const DEFAULT_SCREEN_THRESHOLDS = Object.freeze([
  0.50,   // LOD 0 when > 50 % screen
  0.20,   // LOD 1 when > 20 % screen
  0.05,   // LOD 2 when > 5 % screen
  0.00,   // LOD 3 when > 0 % screen
]);

/**
 * Default per-level cost (relative units). LOD 0 is 1.0, LOD 3 is 0.05.
 */
export const DEFAULT_LEVEL_COSTS = Object.freeze([
  1.00,
  0.40,
  0.15,
  0.05,
  0.02,
  0.01,
  0.005,
  0.001,
]);

/* ------------------------------------------------------------------ */
/* 4. REGISTRATION OF LEVEL TABLES                                    */
/* ------------------------------------------------------------------ */

/**
 * Assigns a distance band table to an entity. The number of bands
 * defines the number of LOD levels.
 *
 *   setDistanceBands(eid, [0, 20, 60, 120])
 */
export function setDistanceBands(eid, bands) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!Array.isArray(bands) || bands.length === 0) return false;

  const count = Math.min(bands.length, MAX_LOD_LEVELS);
  const base = eid * MAX_LOD_LEVELS;

  for (let i = 0; i < count; i++) {
    LODDistance.distances[base + i] = Math.max(0, Number(bands[i]) || 0);
  }
  for (let i = count; i < MAX_LOD_LEVELS; i++) {
    LODDistance.distances[base + i] = Infinity;
  }

  LODLevel.levelCount[eid] = count;
  LODLevel.flags[eid] |= LOD_FLAG.ENABLED;
  return true;
}

/**
 * Assigns a screen-size threshold table to an entity.
 */
export function setScreenThresholds(eid, thresholds) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!Array.isArray(thresholds) || thresholds.length === 0) return false;

  const count = Math.min(thresholds.length, MAX_LOD_LEVELS);
  const base = eid * MAX_LOD_LEVELS;

  for (let i = 0; i < count; i++) {
    LODDistance.screenThresholds[base + i] = Math.max(0, Number(thresholds[i]) || 0);
  }
  for (let i = count; i < MAX_LOD_LEVELS; i++) {
    LODDistance.screenThresholds[base + i] = 0;
  }
  return true;
}

/**
 * Assigns a geometry id per LOD level.
 */
export function setLODGeometries(eid, geometryIds, kinds, triangleCounts) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!Array.isArray(geometryIds) || geometryIds.length === 0) return false;

  const count = Math.min(geometryIds.length, MAX_LOD_LEVELS);
  const base = eid * MAX_LOD_LEVELS;

  for (let i = 0; i < count; i++) {
    LODGeometry.geometryIds[base + i] = geometryIds[i] | 0;
    if (kinds && kinds[i] !== undefined) LODGeometry.geometryKind[base + i] = kinds[i] | 0;
    if (triangleCounts && triangleCounts[i] !== undefined) {
      LODGeometry.triangleCount[base + i] = (triangleCounts[i] | 0) >>> 0;
    }
  }
  for (let i = count; i < MAX_LOD_LEVELS; i++) {
    LODGeometry.geometryIds[base + i] = -1;
    LODGeometry.geometryKind[base + i] = LOD_GEOMETRY_KIND.NONE;
  }
  return true;
}

/**
 * Sets a single geometry id for one LOD level.
 */
export function setLODGeometry(eid, level, geometryId, kind, triangleCount) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (level < 0 || level >= MAX_LOD_LEVELS) return false;

  const base = eid * MAX_LOD_LEVELS;
  LODGeometry.geometryIds[base + level] = geometryId | 0;
  if (kind !== undefined) LODGeometry.geometryKind[base + level] = kind | 0;
  if (triangleCount !== undefined) LODGeometry.triangleCount[base + level] = (triangleCount | 0) >>> 0;
  return true;
}

/**
 * Sets a per-entity hysteresis fraction.
 */
export function setHysteresis(eid, fraction) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const f = Math.max(0, Math.min(0.5, Number(fraction) || 0));
  LODDistance.hysteresis[eid] = f;
  return true;
}

/**
 * Sets a per-entity LOD bias. `useGlobal` controls whether the global
 * bias is also applied.
 */
export function setLODBias(eid, bias, useGlobal) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  LODBias.entityBias[eid] = Math.max(0.1, Math.min(4.0, Number(bias) || 1.0));
  if (useGlobal !== undefined) LODBias.useGlobal[eid] = useGlobal ? 1 : 0;
  LODBias.biasDirty[eid] = 1;
  return true;
}

/**
 * Returns the effective bias for an entity (entity × global if enabled).
 */
export function getLODBias(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 1.0;
  const entity = LODBias.entityBias[eid];
  if (LODBias.useGlobal[eid] === 1) {
    return entity * LODState.globalBias;
  }
  return entity;
}

/**
 * Sets the global LOD bias. Called by the quality controller.
 */
export function setGlobalLODBias(bias) {
  const b = Math.max(0.1, Math.min(4.0, Number(bias) || 1.0));
  LODState.globalBias = b;
  return true;
}

export function getGlobalLODBias() {
  return LODState.globalBias;
}

/* ------------------------------------------------------------------ */
/* 5. SCREEN-SIZE COMPUTATION                                         */
/* ------------------------------------------------------------------ */

/**
 * Computes the screen-space size of an entity given its current world
 * position and its bounding radius. Writes into LODScreenSize and
 * LODDistance.currentScreenSize.
 *
 *   screenSize = (2 * radius) / (distance * 2 * tan(fovHalf))
 *              = radius / (distance * tan(fovHalf))
 *
 * Returns the screen size in [0, 1] (fraction of screen height).
 */
export function computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;

  const wx = TransformWorld.x[eid];
  const wy = TransformWorld.y[eid];
  const wz = TransformWorld.z[eid];

  const dx = wx - cameraX;
  const dy = wy - cameraY;
  const dz = wz - cameraZ;
  const distSq = dx * dx + dy * dy + dz * dz;

  LODDistance.currentDistanceSq[eid] = distSq;
  const dist = Math.sqrt(distSq);
  LODDistance.currentDistance[eid] = dist;

  // Bounding radius: prefer sphere, fall back to AABB-derived, fall
  // back to world max scale × 0.5.
  let radius = 0;
  if (Sphere.valid[eid] === 1) {
    radius = Sphere.radius[eid];
  } else if (AABB.valid[eid] === 1) {
    const hx = (AABB.maxX[eid] - AABB.minX[eid]) * 0.5;
    const hy = (AABB.maxY[eid] - AABB.minY[eid]) * 0.5;
    const hz = (AABB.maxZ[eid] - AABB.minZ[eid]) * 0.5;
    radius = Math.sqrt(hx * hx + hy * hy + hz * hz);
  } else {
    radius = (TransformWorld.maxScale[eid] || 1) * 0.5;
  }

  // Screen size as fraction of screen height.
  const denom = Math.max(0.001, dist * tanFovHalf);
  let screenSize = radius / denom;
  if (screenSize > 1) screenSize = 1;
  if (screenSize < 0) screenSize = 0;

  LODScreenSize.screenSize[eid] = screenSize;
  LODScreenSize.valid[eid] = 1;
  LODDistance.currentScreenSize[eid] = screenSize;

  return screenSize;
}

/**
 * Computes the screen size in pixels given the viewport height.
 */
export function computeScreenSizePixels(eid, viewportHeight) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  const size = LODScreenSize.screenSize[eid];
  const pixels = size * viewportHeight;
  LODScreenSize.screenSizePixels[eid] = pixels;
  return pixels;
}

/**
 * Computes the screen size as a percentage of the viewport.
 */
export function computeScreenSizePercent(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  const pct = LODScreenSize.screenSize[eid] * 100;
  LODScreenSize.screenSizePercent[eid] = pct;
  return pct;
}

/* ------------------------------------------------------------------ */
/* 6. DISTANCE BAND CLASSIFICATION                                    */
/* ------------------------------------------------------------------ */

/**
 * Classifies a distance into one of the entity's LOD distance bands.
 * Returns the band index (0..levelCount-1).
 */
export function computeDistanceBand(eid, distance) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;

  const levelCount = LODLevel.levelCount[eid];
  if (levelCount === 0) return 0;

  const bias = getLODBias(eid);
  const d = distance * bias;   // larger bias → higher effective distance
  const base = eid * MAX_LOD_LEVELS;

  for (let i = 0; i < levelCount; i++) {
    const band = LODDistance.distances[base + i];
    if (d < band) return i;
  }
  return levelCount - 1;
}

/**
 * Classifies a screen size into one of the entity's LOD screen bands.
 * Returns the band index (0..levelCount-1).
 */
export function computeScreenBand(eid, screenSize) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;

  const levelCount = LODLevel.levelCount[eid];
  if (levelCount === 0) return 0;

  const base = eid * MAX_LOD_LEVELS;

  // Screen bands work in reverse: larger screen size → lower LOD index.
  for (let i = 0; i < levelCount; i++) {
    const threshold = LODDistance.screenThresholds[base + i];
    if (screenSize >= threshold) return i;
  }
  return levelCount - 1;
}

/* ------------------------------------------------------------------ */
/* 7. HYSTERESIS                                                      */
/* ------------------------------------------------------------------ */

/**
 * Applies hysteresis to a candidate LOD level. If the entity has just
 * crossed a band edge and the change is smaller than the hysteresis
 * fraction, the previous level is retained.
 *
 * Returns the debounced level.
 */
export function applyHysteresis(eid, candidateLevel) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return candidateLevel;

  const currentLevel = LODLevel.currentLevel[eid];
  if (currentLevel < 0) return candidateLevel;   // no previous decision
  if (candidateLevel === currentLevel) return candidateLevel;

  const hysteresis = LODDistance.hysteresis[eid];
  if (hysteresis <= 0) return candidateLevel;

  // Only guard 1-level transitions; multi-level jumps bypass hysteresis.
  const delta = Math.abs(candidateLevel - currentLevel);
  if (delta > 1) return candidateLevel;

  const prevDistance = LODHysteresis.previousDistance[eid];
  const curDistance = LODDistance.currentDistance[eid];
  const prevScreen = LODHysteresis.previousScreenSize[eid];
  const curScreen = LODDistance.currentScreenSize[eid];

  // Compute both candidate bands.
  const prevBand = computeDistanceBand(eid, prevDistance);
  const curBand = computeDistanceBand(eid, curDistance);
  const prevScreenBand = computeScreenBand(eid, prevScreen);
  const curScreenBand = computeScreenBand(eid, curScreen);

  // If distance band has not actually changed, hold.
  if (prevBand === curBand && prevScreenBand === curScreenBand) {
    LODState.totalHysteresisHolds++;
    return currentLevel;
  }

  // If we are moving toward the current level's band, hold.
  if (Math.abs(curBand - currentLevel) <= Math.abs(curBand - candidateLevel) &&
      Math.abs(curScreenBand - currentLevel) <= Math.abs(curScreenBand - candidateLevel)) {
    LODState.totalHysteresisHolds++;
    return currentLevel;
  }

  return candidateLevel;
}

/**
 * Updates the hysteresis history for the entity. Called after a
 * successful level decision.
 */
export function updateHysteresisHistory(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  LODHysteresis.previousDistance[eid] = LODDistance.currentDistance[eid];
  LODHysteresis.previousScreenSize[eid] = LODDistance.currentScreenSize[eid];
  LODHysteresis.holdFrames[eid] = 0;
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. LOD EVALUATION                                                  */
/* ------------------------------------------------------------------ */

/**
 * Evaluates the target LOD level for an entity. Uses distance bands
 * first, then screen-size bands as a fallback (or as a supplement when
 * the entity has screen thresholds configured).
 *
 * Returns the target LOD level (does NOT commit it).
 */
export function evaluateLOD(eid, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_NONE;

  const flags = LODLevel.flags[eid];
  if ((flags & LOD_FLAG.ENABLED) === 0) return LOD_NONE;
  if ((flags & LOD_FLAG.SUSPENDED) !== 0) return LODLevel.currentLevel[eid];

  LODState.totalEvaluations++;
  LODStats.evaluationsThisFrame[0]++;

  const levelCount = LODLevel.levelCount[eid];
  if (levelCount === 0) return LOD_NONE;

  const useScreenSize = options && options.useScreenSize === true;

  // Distance-based decision.
  const distBand = computeDistanceBand(eid, LODDistance.currentDistance[eid]);

  // Screen-size decision (optional).
  let screenBand = -1;
  if (useScreenSize) {
    screenBand = computeScreenBand(eid, LODDistance.currentScreenSize[eid]);
  }

  // Combine: take the coarser (higher index) of the two.
  let candidate = distBand;
  if (screenBand >= 0 && screenBand > candidate) {
    candidate = screenBand;
  }

  // Clamp to configured min/max levels.
  const minLevel = LODBias.minLevel[eid];
  const maxLevel = LODBias.maxLevel[eid];
  if (minLevel >= 0 && candidate < minLevel) candidate = minLevel;
  if (maxLevel >= 0 && candidate > maxLevel) candidate = maxLevel;

  // Apply hysteresis.
  const debounced = applyHysteresis(eid, candidate);

  // Budget-aware downgrade.
  if ((flags & LOD_FLAG.BUDGET_AWARE) !== 0 && LODBudget.budgetExceeded[0] === 1) {
    if (debounced < levelCount - 1) {
      const downgraded = debounced + 1;
      LODDebug.lastDecisionReason[eid] = 2;   // budget
      LODBudget.forcedDowngrades[0]++;
      LODState.totalBudgetDowngrades++;
      return downgraded;
    }
  }

  // Record debug reason.
  if (debounced !== LODLevel.currentLevel[eid]) {
    LODDebug.lastDecisionReason[eid] = useScreenSize ? 1 : 0;
  }

  LODDebug.lastDecisionFrame[eid] = LODState.frame;
  LODDebug.lastDecisionLevel[eid] = debounced;
  LODDebug.lastDecisionBias[eid] = getLODBias(eid);

  return debounced;
}

/**
 * Evaluates the LOD level and commits it if it differs from the current
 * level. Returns the new level.
 */
export function refreshLOD(eid, options) {
  const target = evaluateLOD(eid, options);
  if (target === LOD_NONE) return LOD_NONE;

  const current = LODLevel.currentLevel[eid];
  if (target === current) return current;

  return switchLODLevel(eid, target) ? target : current;
}

/* ------------------------------------------------------------------ */
/* 9. LEVEL SWITCHING                                                 */
/* ------------------------------------------------------------------ */

/**
 * Commits a LOD level change. Starts a cross-fade transition if the
 * entity has an active transition state and the delta is small.
 *
 * Returns true on success.
 */
export function switchLODLevel(eid, newLevel) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const levelCount = LODLevel.levelCount[eid];
  if (newLevel < 0 || newLevel >= levelCount) return false;

  const current = LODLevel.currentLevel[eid];
  if (current === newLevel) return false;

  const t0 = _now();

  LODLevel.previousLevel[eid] = current;
  LODLevel.currentLevel[eid] = newLevel;
  LODLevel.targetLevel[eid] = newLevel;
  LODLevel.lastSwitchFrame[eid] = LODState.frame;
  if (LODLevel.switchCount[eid] < 0xFFFF) LODLevel.switchCount[eid]++;

  // Start cross-fade transition if enabled.
  if (current >= 0) {
    const delta = Math.abs(newLevel - current);
    if (delta === 1) {
      LODTransition.fromLevel[eid] = current;
      LODTransition.toLevel[eid] = newLevel;
      LODTransition.frameCount[eid] = 0;
      LODTransition.maxFrames[eid] = LOD_TRANSITION_FRAMES;
      LODTransition.alpha[eid] = 0;
      LODTransition.active[eid] = 1;
      LODTransition.fadeMode[eid] = 1;   // smooth
      LODState.activeTransitions++;
      if (LODState.activeTransitions > LODState.peakActiveTransitions) {
        LODState.peakActiveTransitions = LODState.activeTransitions;
      }
      LODState.totalTransitions++;
      LODLevel.state[eid] = LOD_STATE.TRANSITION;
    } else {
      LODLevel.state[eid] = LOD_STATE.STABLE;
    }
  } else {
    // First assignment — no transition.
    LODLevel.state[eid] = LOD_STATE.STABLE;
  }

  // Update budget cost.
  const base = eid * MAX_LOD_LEVELS;
  const cost = DEFAULT_LEVEL_COSTS[newLevel] || 1.0;
  LODBudget.entityCost[eid] = cost;
  LODBudget.entityCostEma[eid] += (cost - LODBudget.entityCostEma[eid]) * 0.15;

  LODState.totalLevelSwitches++;
  LODStats.switchesThisFrame[0]++;

  const t1 = _now();
  const switchCost = t1 - t0;
  LODState.lastSwitchMs = switchCost;
  LODState.avgSwitchMs += (switchCost - LODState.avgSwitchMs) * 0.15;

  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);

  return true;
}

/**
 * Forces a specific LOD level, ignoring evaluation. Sets the FORCED state
 * so subsequent evaluations do not override it until `clearForcedLOD`.
 */
export function forceLODLevel(eid, level) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (level < 0 || level >= MAX_LOD_LEVELS) return false;

  const current = LODLevel.currentLevel[eid];
  LODLevel.previousLevel[eid] = current;
  LODLevel.currentLevel[eid] = level;
  LODLevel.targetLevel[eid] = level;
  LODLevel.state[eid] = LOD_STATE.FORCED;
  LODLevel.lastSwitchFrame[eid] = LODState.frame;

  LODStats.switchesThisFrame[0]++;
  LODState.totalLevelSwitches++;

  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);
  return true;
}

/**
 * Clears the FORCED state so evaluation resumes.
 */
export function clearForcedLOD(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (LODLevel.state[eid] === LOD_STATE.FORCED) {
    LODLevel.state[eid] = LOD_STATE.STABLE;
  }
  return true;
}

/**
 * Returns true if the entity's LOD level is currently forced.
 */
export function isLODForced(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return LODLevel.state[eid] === LOD_STATE.FORCED;
}

/* ------------------------------------------------------------------ */
/* 10. LOD TRANSITION                                                 */
/* ------------------------------------------------------------------ */

/**
 * Advances every active cross-fade transition by one frame. Called once
 * per frame by the engine loop.
 *
 * Returns the number of transitions that completed this frame.
 */
export function updateLODTransitions() {
  let completed = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (LODTransition.active[eid] === 0) continue;

    const frameCount = LODTransition.frameCount[eid];
    const maxFrames = LODTransition.maxFrames[eid];

    if (maxFrames === 0) {
      LODTransition.active[eid] = 0;
      LODTransition.alpha[eid] = 1.0;
      LODLevel.state[eid] = LOD_STATE.STABLE;
      LODState.activeTransitions--;
      completed++;
      continue;
    }

    const t = Math.min(1.0, (frameCount + 1) / maxFrames);
    LODTransition.alpha[eid] = t;
    LODTransition.frameCount[eid] = frameCount + 1;

    LODState.totalTransitionAdv++;

    if (frameCount + 1 >= maxFrames) {
      LODTransition.active[eid] = 0;
      LODTransition.alpha[eid] = 1.0;
      LODLevel.state[eid] = LOD_STATE.STABLE;
      LODState.activeTransitions--;
      completed++;
    }
  }

  LODStats.transitionsActive[0] = LODState.activeTransitions;
  return completed;
}

/**
 * Interpolates between the entity's from/to LOD level given the current
 * transition alpha. Returns a blend factor in [0, 1]:
 *   0 = fully from level
 *   1 = fully to level
 *  -1 = no active transition (entity is at its current level)
 */
export function interpolateLODTransition(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;
  if (LODTransition.active[eid] === 0) return -1;
  return LODTransition.alpha[eid];
}

/**
 * Starts a manual cross-fade transition between two levels.
 */
export function startLODTransition(eid, fromLevel, toLevel, frames) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const maxFrames = frames !== undefined ? Math.max(1, frames | 0) : LOD_TRANSITION_FRAMES;
  LODTransition.fromLevel[eid] = fromLevel;
  LODTransition.toLevel[eid] = toLevel;
  LODTransition.frameCount[eid] = 0;
  LODTransition.maxFrames[eid] = maxFrames;
  LODTransition.alpha[eid] = 0;
  LODTransition.active[eid] = 1;
  LODTransition.fadeMode[eid] = 1;

  LODLevel.state[eid] = LOD_STATE.TRANSITION;

  if (LODTransition.active[eid] === 1) {
    LODState.activeTransitions++;
    LODState.totalTransitions++;
  }

  return true;
}

/**
 * Returns true if the entity is currently cross-fading.
 */
export function isLODTransitioning(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return LODTransition.active[eid] === 1;
}

/* ------------------------------------------------------------------ */
/* 11. GEOMETRY SELECTION                                             */
/* ------------------------------------------------------------------ */

/**
 * Returns the geometry id for the entity's current LOD level, or -1.
 */
export function selectLODGeometry(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;
  const level = LODLevel.currentLevel[eid];
  if (level < 0 || level >= MAX_LOD_LEVELS) return -1;
  return LODGeometry.geometryIds[eid * MAX_LOD_LEVELS + level];
}

/**
 * Returns the geometry kind for the entity's current LOD level.
 */
export function selectLODGeometryKind(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_GEOMETRY_KIND.NONE;
  const level = LODLevel.currentLevel[eid];
  if (level < 0 || level >= MAX_LOD_LEVELS) return LOD_GEOMETRY_KIND.NONE;
  return LODGeometry.geometryKind[eid * MAX_LOD_LEVELS + level];
}

/**
 * Returns the triangle count for the entity's current LOD level.
 */
export function selectLODTriangleCount(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  const level = LODLevel.currentLevel[eid];
  if (level < 0 || level >= MAX_LOD_LEVELS) return 0;
  return LODGeometry.triangleCount[eid * MAX_LOD_LEVELS + level];
}

/**
 * Returns the material id for the entity's current LOD level.
 */
export function selectLODMaterialId(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;
  const level = LODLevel.currentLevel[eid];
  if (level < 0 || level >= MAX_LOD_LEVELS) return -1;
  return LODGeometry.materialId[eid * MAX_LOD_LEVELS + level];
}

/* ------------------------------------------------------------------ */
/* 12. IMPOSTOR & PROXY BINDING                                       */
/* ------------------------------------------------------------------ */

/**
 * Binds an impostor entity as the visual proxy for the entity's lowest
 * LOD level.
 */
export function bindLODImpostor(eid, impostorEid, radius, height, billboardMode) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  LODImpostor.impostorEid[eid] = impostorEid | 0;
  LODImpostor.impostorRadius[eid] = Number.isFinite(radius) ? radius : 1.0;
  LODImpostor.impostorHeight[eid] = Number.isFinite(height) ? height : 2.0;
  LODImpostor.billboardMode[eid] = (billboardMode | 0) & 0xFF;
  LODImpostor.valid[eid] = 1;
  LODLevel.flags[eid] |= LOD_FLAG.HAS_IMPOSTOR;
  return true;
}

/**
 * Binds a proxy geometry as the visual proxy for the entity's mid LOD
 * levels.
 */
export function bindLODProxy(eid, proxyEid, radius, kind) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  LODProxy.proxyEid[eid] = proxyEid | 0;
  LODProxy.proxyRadius[eid] = Number.isFinite(radius) ? radius : 1.0;
  LODProxy.proxyKind[eid] = (kind | 0) & 0xFF;
  LODProxy.valid[eid] = 1;
  LODLevel.flags[eid] |= LOD_FLAG.HAS_PROXY;
  return true;
}

/* ------------------------------------------------------------------ */
/* 13. LOD GROUP SUPPORT                                              */
/* ------------------------------------------------------------------ */

/**
 * Registers an entity as the anchor of an LOD group.
 */
export function registerLODGroupAnchor(anchorEid) {
  if (typeof anchorEid !== 'number' || anchorEid < 0 || anchorEid >= MAX_ENTITIES) return false;
  LODGroup.anchorEid[anchorEid] = anchorEid;
  LODGroup.memberCount[anchorEid] = 0;
  LODGroup.groupDirty[anchorEid] = 1;
  LODLevel.flags[anchorEid] |= LOD_FLAG.GROUP_ANCHOR;
  return true;
}

/**
 * Attaches a member entity to an LOD group anchor.
 */
export function attachLODGroupMember(groupAnchorEid, memberEid, localWeight) {
  if (typeof groupAnchorEid !== 'number' || groupAnchorEid < 0 || groupAnchorEid >= MAX_ENTITIES) return false;
  if (typeof memberEid !== 'number' || memberEid < 0 || memberEid >= MAX_ENTITIES) return false;

  if (LODGroup.memberCount[groupAnchorEid] >= MAX_LOD_GROUP_MEMBERS) return false;

  LODMember.groupEid[memberEid] = groupAnchorEid;
  LODMember.memberIndex[memberEid] = LODGroup.memberCount[groupAnchorEid];
  LODMember.localWeight[memberEid] = Number.isFinite(localWeight) ? localWeight : 1.0;
  LODGroup.memberCount[groupAnchorEid]++;
  LODLevel.flags[memberEid] |= LOD_FLAG.GROUP_MEMBER;

  // Register the relation edge so the group survives serialization.
  bindLODGroup(memberEid, groupAnchorEid);

  return true;
}

/**
 * Detaches a member entity from its LOD group.
 */
export function detachLODGroupMember(memberEid) {
  if (typeof memberEid !== 'number' || memberEid < 0 || memberEid >= MAX_ENTITIES) return false;
  const groupEid = LODMember.groupEid[memberEid];
  if (groupEid < 0) return false;

  if (LODGroup.memberCount[groupEid] > 0) {
    LODGroup.memberCount[groupEid]--;
  }
  LODMember.groupEid[memberEid] = -1;
  LODLevel.flags[memberEid] &= ~LOD_FLAG.GROUP_MEMBER;
  return true;
}

/**
 * Evaluates an entire LOD group as one unit. All members receive the
 * same LOD level derived from the group's aggregate distance and
 * bounding radius.
 *
 * Returns the group's LOD level.
 */
export function evaluateGroupLOD(groupAnchorEid, cameraX, cameraY, cameraZ) {
  if (typeof groupAnchorEid !== 'number' || groupAnchorEid < 0 || groupAnchorEid >= MAX_ENTITIES) return LOD_NONE;

  LODState.totalGroupEvaluations++;

  const wx = TransformWorld.x[groupAnchorEid];
  const wy = TransformWorld.y[groupAnchorEid];
  const wz = TransformWorld.z[groupAnchorEid];

  const dx = wx - cameraX;
  const dy = wy - cameraY;
  const dz = wz - cameraZ;
  const groupDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  LODGroup.groupDistance[groupAnchorEid] = groupDistance;

  // Aggregate group radius from members.
  let maxRadius = 0;
  const levelCount = LODLevel.levelCount[groupAnchorEid];
  const base = groupAnchorEid * MAX_LOD_LEVELS;
  const bias = getLODBias(groupAnchorEid);
  const biasedDistance = groupDistance * bias;

  let candidate = 0;
  for (let i = 0; i < levelCount; i++) {
    if (biasedDistance >= LODDistance.distances[base + i]) candidate = i;
  }

  LODGroup.groupLevel[groupAnchorEid] = candidate;
  LODGroup.groupDirty[groupAnchorEid] = 0;

  // Propagate to every member.
  const memberCount = LODGroup.memberCount[groupAnchorEid];
  if (memberCount > 0) {
    for (let eid = 0; eid < MAX_ENTITIES; eid++) {
      if (LODMember.groupEid[eid] !== groupAnchorEid) continue;
      switchLODLevel(eid, candidate);
    }
  }

  return candidate;
}

/* ------------------------------------------------------------------ */
/* 14. BUDGET-AWARE EVALUATION                                        */
/* ------------------------------------------------------------------ */

/**
 * Sets the LOD budget cap in relative cost units.
 */
export function setLODBudgetCap(cap) {
  LODBudget.budgetCap[0] = Math.max(0, Number(cap) || 0);
  return true;
}

/**
 * Returns the current LOD budget cap.
 */
export function getLODBudgetCap() {
  return LODBudget.budgetCap[0];
}

/**
 * Recomputes the total visible cost and updates the budget-exceeded
 * flag. Called once per frame after all LOD evaluations.
 */
export function updateLODBudget() {
  let total = 0;
  let visible = 0;

  const histogram = LODStats.levelHistogram;
  histogram.fill(0);

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const level = LODLevel.currentLevel[eid];
    if (level < 0) continue;

    const flags = EntityTag[eid];
    if ((flags & TAG.VISIBLE) === 0) continue;

    visible++;
    total += LODBudget.entityCostEma[eid];

    if (level < MAX_LOD_LEVELS) histogram[level]++;
  }

  LODBudget.totalVisibleCost[0] = total;
  LODStats.visibleCount[0] = visible;

  const cap = LODBudget.budgetCap[0];
  if (cap > 0 && total > cap) {
    LODBudget.budgetExceeded[0] = 1;
  } else {
    LODBudget.budgetExceeded[0] = 0;
  }

  return total;
}

/**
 * Returns the current total visible LOD cost.
 */
export function getLODVisibleCost() {
  return LODBudget.totalVisibleCost[0];
}

/**
 * Returns true if the LOD budget is currently exceeded.
 */
export function isLODBudgetExceeded() {
  return LODBudget.budgetExceeded[0] === 1;
}

/**
 * Applies a global budget-aware downgrade: nudges the global bias higher
 * so every entity picks a coarser LOD.
 */
export function applyBudgetDowngrade(factor) {
  const f = Math.max(1.0, Math.min(2.0, Number(factor) || 1.1));
  const newBias = Math.min(4.0, LODState.globalBias * f);
  LODState.globalBias = newBias;
  LODBudget.forcedDowngrades[0]++;
  return newBias;
}

/**
 * Clears the budget downgrade by resetting the global bias to 1.0.
 */
export function clearBudgetDowngrade() {
  LODState.globalBias = 1.0;
  LODBudget.budgetExceeded[0] = 0;
  return true;
}

/* ------------------------------------------------------------------ */
/* 15. LIGHT / SHADOW / GI / AO CONVENIENCE                           */
/* ------------------------------------------------------------------ */

/**
 * Refreshes LOD for a light entity. Uses the light's LightBudget
 * component's cost estimate as the per-level cost.
 */
export function refreshLODForLight(eid, cameraX, cameraY, cameraZ, tanFovHalf, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_NONE;
  if ((EntityTag[eid] & TAG.LIGHT) === 0) return LOD_NONE;

  computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf);
  return refreshLOD(eid, options);
}

/**
 * Refreshes LOD for a shadow caster entity.
 */
export function refreshLODForShadowCaster(eid, cameraX, cameraY, cameraZ, tanFovHalf, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_NONE;
  if ((EntityTag[eid] & TAG.SHADOW_CASTER) === 0) return LOD_NONE;

  computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf);
  return refreshLOD(eid, options);
}

/**
 * Refreshes LOD for a GI probe entity.
 */
export function refreshLODForGIProbe(eid, cameraX, cameraY, cameraZ, tanFovHalf, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_NONE;
  if ((EntityTag[eid] & TAG.GI_PROBE) === 0) return LOD_NONE;

  computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf);
  return refreshLOD(eid, options);
}

/**
 * Refreshes LOD for an AO volume entity.
 */
export function refreshLODForAOVolume(eid, cameraX, cameraY, cameraZ, tanFovHalf, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return LOD_NONE;
  if ((EntityTag[eid] & TAG.AO_VOLUME) === 0) return LOD_NONE;

  computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf);
  return refreshLOD(eid, options);
}

/* ------------------------------------------------------------------ */
/* 16. BULK EVALUATION                                                */
/* ------------------------------------------------------------------ */

/**
 * Evaluates LOD for every tagged entity that has the LOD ENABLED flag.
 * Called once per frame by the LOD system.
 *
 * Returns the number of entities evaluated.
 */
export function evaluateAllLOD(cameraX, cameraY, cameraZ, tanFovHalf, options) {
  const t0 = _now();
  let evaluated = 0;

  LODStats.evaluationsThisFrame[0] = 0;
  LODStats.switchesThisFrame[0] = 0;
  LODStats.hysteresisHolds[0] = 0;

  const useScreenSize = options && options.useScreenSize === true;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const flags = LODLevel.flags[eid];
    if ((flags & LOD_FLAG.ENABLED) === 0) continue;
    if (LODLevel.state[eid] === LOD_STATE.FORCED) continue;
    if (LODLevel.state[eid] === LOD_STATE.SUSPENDED) continue;
    if (LODLevel.state[eid] === LOD_STATE.DISABLED) continue;

    // Skip group members — the group anchor evaluates them.
    if ((flags & LOD_FLAG.GROUP_MEMBER) !== 0) continue;

    // Update screen size.
    if (useScreenSize) {
      computeScreenSize(eid, cameraX, cameraY, cameraZ, tanFovHalf);
    } else {
      // Still compute distance for the distance-band decision.
      const wx = TransformWorld.x[eid];
      const wy = TransformWorld.y[eid];
      const wz = TransformWorld.z[eid];
      const dx = wx - cameraX;
      const dy = wy - cameraY;
      const dz = wz - cameraZ;
      const dSq = dx * dx + dy * dy + dz * dz;
      LODDistance.currentDistanceSq[eid] = dSq;
      LODDistance.currentDistance[eid] = Math.sqrt(dSq);
    }

    // Evaluate.
    const target = evaluateLOD(eid, options);
    if (target !== LOD_NONE) {
      const current = LODLevel.currentLevel[eid];
      if (target !== current) {
        switchLODLevel(eid, target);
        updateHysteresisHistory(eid);
      } else {
        updateHysteresisHistory(eid);
      }
    }

    // Update screen-size caches.
    if (useScreenSize) {
      LODScreenSize.screenSize[eid] = LODDistance.currentScreenSize[eid];
    }

    LODLevel.lastEvalFrame[eid] = LODState.frame;
    evaluated++;
  }

  // Group anchors.
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((LODLevel.flags[eid] & LOD_FLAG.GROUP_ANCHOR) === 0) continue;
    evaluateGroupLOD(eid, cameraX, cameraY, cameraZ);
  }

  const t1 = _now();
  const cost = t1 - t0;
  LODState.lastEvalMs = cost;
  LODState.avgEvalMs += (cost - LODState.avgEvalMs) * 0.15;

  return evaluated;
}

/* ------------------------------------------------------------------ */
/* 17. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the LOD system frame counter. Called once per frame.
 */
export function tickLOD(frameNumber) {
  if (typeof frameNumber === 'number') LODState.frame = frameNumber;
  else LODState.frame++;
}

/**
 * Full per-frame LOD pipeline:
 *   1. tickLOD(frame)
 *   2. evaluateAllLOD(camera, fov, options)
 *   3. updateLODTransitions()
 *   4. updateLODBudget()
 *
 * Returns a summary object.
 */
export function tickLODSystem(frameNumber, cameraX, cameraY, cameraZ, tanFovHalf, options) {
  tickLOD(frameNumber);
  const evaluated = evaluateAllLOD(cameraX, cameraY, cameraZ, tanFovHalf, options);
  const completed = updateLODTransitions();
  const totalCost = updateLODBudget();

  return {
    frame: LODState.frame,
    evaluated,
    completed,
    totalCost,
    activeTransitions: LODState.activeTransitions,
    budgetExceeded: LODBudget.budgetExceeded[0] === 1,
  };
}

/* ------------------------------------------------------------------ */
/* 18. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerLODComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'LODLevel',       component: LODLevel,       category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODDistance',    component: LODDistance,    category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODGeometry',    component: LODGeometry,    category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODBias',        component: LODBias,        category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODHysteresis',  component: LODHysteresis,  category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODGroup',       component: LODGroup,       category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODMember',      component: LODMember,      category: 16, subsystem: 1, dependencies: ['LODGroup'] },
    { name: 'LODTransition',  component: LODTransition,  category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODBudget',      component: LODBudget,      category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODScreenSize',  component: LODScreenSize,  category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODImpostor',    component: LODImpostor,    category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODProxy',       component: LODProxy,       category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODDebug',       component: LODDebug,       category: 16, subsystem: 1, dependencies: [] },
    { name: 'LODStats',       component: LODStats,       category: 16, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 19. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getLODStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  const base = eid * MAX_LOD_LEVELS;
  return {
    entity:          eid,
    currentLevel:    LODLevel.currentLevel[eid],
    targetLevel:     LODLevel.targetLevel[eid],
    previousLevel:   LODLevel.previousLevel[eid],
    levelCount:      LODLevel.levelCount[eid],
    state:           LOD_STATE_NAME[LODLevel.state[eid]] || 'none',
    flags:           LODLevel.flags[eid],
    distance:        LODDistance.currentDistance[eid],
    screenSize:      LODDistance.currentScreenSize[eid],
    hysteresis:      LODDistance.hysteresis[eid],
    bias:            getLODBias(eid),
    switchCount:     LODLevel.switchCount[eid],
    lastSwitchFrame: LODLevel.lastSwitchFrame[eid],
    geometryId:      selectLODGeometry(eid),
    geometryKind:    LOD_GEOMETRY_KIND_NAME[selectLODGeometryKind(eid)] || 'none',
    triangleCount:   selectLODTriangleCount(eid),
    transitionActive:LODTransition.active[eid] === 1,
    transitionAlpha: LODTransition.alpha[eid],
    budgetCost:      LODBudget.entityCostEma[eid],
    hasImpostor:     LODImpostor.valid[eid] === 1,
    hasProxy:        LODProxy.valid[eid] === 1,
    isGroupAnchor:   (LODLevel.flags[eid] & LOD_FLAG.GROUP_ANCHOR) !== 0,
    isGroupMember:   (LODLevel.flags[eid] & LOD_FLAG.GROUP_MEMBER) !== 0,
    groupEid:        LODMember.groupEid[eid],
  };
}

export function getLODSystemReport() {
  const histogram = [];
  for (let i = 0; i < MAX_LOD_LEVELS; i++) {
    histogram.push(LODStats.levelHistogram[i]);
  }

  return {
    frame:                LODState.frame,
    globalBias:           LODState.globalBias,
    totalEvaluations:     LODState.totalEvaluations,
    totalLevelSwitches:   LODState.totalLevelSwitches,
    totalTransitions:     LODState.totalTransitions,
    totalTransitionAdv:   LODState.totalTransitionAdv,
    totalHysteresisHolds: LODState.totalHysteresisHolds,
    totalBudgetDowngrades:LODState.totalBudgetDowngrades,
    totalGroupEvaluations:LODState.totalGroupEvaluations,
    activeTransitions:    LODState.activeTransitions,
    peakActiveTransitions:LODState.peakActiveTransitions,
    lastEvalMs:           LODState.lastEvalMs,
    avgEvalMs:            LODState.avgEvalMs,
    lastSwitchMs:         LODState.lastSwitchMs,
    avgSwitchMs:          LODState.avgSwitchMs,
    visibleCount:         LODStats.visibleCount[0],
    levelHistogram:       histogram,
    totalVisibleCost:     LODBudget.totalVisibleCost[0],
    budgetCap:            LODBudget.budgetCap[0],
    budgetExceeded:       LODBudget.budgetExceeded[0] === 1,
    forcedDowngrades:     LODBudget.forcedDowngrades[0],
    perfTier:             PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 20. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every LOD structure and resets counters. The global bias is
 * preserved.
 */
export function resetLODState() {
  LODLevel.currentLevel.fill(-1);
  LODLevel.targetLevel.fill(-1);
  LODLevel.previousLevel.fill(-1);
  LODLevel.levelCount.fill(0);
  LODLevel.state.fill(0);
  LODLevel.flags.fill(0);
  LODLevel.lastSwitchFrame.fill(0);
  LODLevel.lastEvalFrame.fill(0);
  LODLevel.switchCount.fill(0);

  LODDistance.distances.fill(0);
  LODDistance.screenThresholds.fill(0);
  LODDistance.hysteresis.fill(DEFAULT_HYSTERESIS);
  LODDistance.currentDistance.fill(0);
  LODDistance.currentDistanceSq.fill(0);
  LODDistance.currentScreenSize.fill(0);
  LODDistance.distanceBand.fill(0);

  LODGeometry.geometryIds.fill(-1);
  LODGeometry.geometryKind.fill(LOD_GEOMETRY_KIND.NONE);
  LODGeometry.triangleCount.fill(0);
  LODGeometry.vertexCount.fill(0);
  LODGeometry.materialId.fill(-1);

  LODBias.entityBias.fill(1.0);
  LODBias.useGlobal.fill(1);
  LODBias.minLevel.fill(-1);
  LODBias.maxLevel.fill(-1);
  LODBias.biasDirty.fill(0);

  LODHysteresis.previousDistance.fill(0);
  LODHysteresis.previousScreenSize.fill(0);
  LODHysteresis.holdFrames.fill(0);
  LODHysteresis.holdThreshold.fill(0);

  LODGroup.anchorEid.fill(-1);
  LODGroup.memberCount.fill(0);
  LODGroup.groupLevel.fill(-1);
  LODGroup.groupDistance.fill(0);
  LODGroup.groupRadius.fill(0);
  LODGroup.groupDirty.fill(0);
  LODGroup.groupLevelCount.fill(0);

  LODMember.memberIndex.fill(0);
  LODMember.groupEid.fill(-1);
  LODMember.localWeight.fill(1.0);

  LODTransition.fromLevel.fill(-1);
  LODTransition.toLevel.fill(-1);
  LODTransition.frameCount.fill(0);
  LODTransition.maxFrames.fill(0);
  LODTransition.alpha.fill(0);
  LODTransition.active.fill(0);
  LODTransition.fadeMode.fill(0);

  LODBudget.entityCost.fill(0);
  LODBudget.entityCostEma.fill(0);
  LODBudget.totalVisibleCost[0] = 0;
  LODBudget.budgetExceeded[0] = 0;
  LODBudget.forcedDowngrades[0] = 0;

  LODScreenSize.screenSize.fill(0);
  LODScreenSize.screenSizePixels.fill(0);
  LODScreenSize.screenSizePercent.fill(0);
  LODScreenSize.pixelThreshold.fill(0);
  LODScreenSize.valid.fill(0);

  LODImpostor.impostorEid.fill(-1);
  LODImpostor.impostorRadius.fill(0);
  LODImpostor.impostorHeight.fill(0);
  LODImpostor.billboardMode.fill(0);
  LODImpostor.valid.fill(0);

  LODProxy.proxyEid.fill(-1);
  LODProxy.proxyRadius.fill(0);
  LODProxy.proxyKind.fill(0);
  LODProxy.valid.fill(0);

  LODDebug.lastDecisionFrame.fill(0);
  LODDebug.lastDecisionLevel.fill(-1);
  LODDebug.lastDecisionReason.fill(0);
  LODDebug.lastDecisionBias.fill(1.0);

  LODStats.visibleCount[0] = 0;
  LODStats.levelHistogram.fill(0);
  LODStats.transitionsActive[0] = 0;
  LODStats.evaluationsThisFrame[0] = 0;
  LODStats.switchesThisFrame[0] = 0;
  LODStats.hysteresisHolds[0] = 0;

  LODState.frame = 0;
  LODState.totalEvaluations = 0;
  LODState.totalLevelSwitches = 0;
  LODState.totalTransitions = 0;
  LODState.totalTransitionAdv = 0;
  LODState.totalHysteresisHolds = 0;
  LODState.totalBudgetDowngrades = 0;
  LODState.totalGroupEvaluations = 0;
  LODState.activeTransitions = 0;
  LODState.peakActiveTransitions = 0;
  LODState.lastEvalMs = 0;
  LODState.avgEvalMs = 0;
  LODState.lastSwitchMs = 0;
  LODState.avgSwitchMs = 0;
}

/* ------------------------------------------------------------------ */
/* 21. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_LOD_LEVELS,
  MAX_LOD_GROUP_MEMBERS,
  LOD_TRANSITION_FRAMES,
  DEFAULT_HYSTERESIS,
  LOD_AUTO,
  LOD_NONE,
  DEFAULT_DISTANCE_BANDS,
  DEFAULT_SCREEN_THRESHOLDS,
  DEFAULT_LEVEL_COSTS,

  // Enums
  LOD_STATE,
  LOD_STATE_NAME,
  LOD_GEOMETRY_KIND,
  LOD_GEOMETRY_KIND_NAME,
  LOD_FLAG,

  // Components
  LODLevel,
  LODDistance,
  LODGeometry,
  LODBias,
  LODHysteresis,
  LODGroup,
  LODMember,
  LODTransition,
  LODBudget,
  LODScreenSize,
  LODImpostor,
  LODProxy,
  LODDebug,
  LODStats,
  LOD_COMPONENTS,

  // Module state
  LODState,

  // Level table registration
  setDistanceBands,
  setScreenThresholds,
  setLODGeometries,
  setLODGeometry,
  setHysteresis,
  setLODBias,
  getLODBias,
  setGlobalLODBias,
  getGlobalLODBias,

  // Screen-size
  computeScreenSize,
  computeScreenSizePixels,
  computeScreenSizePercent,

  // Band classification
  computeDistanceBand,
  computeScreenBand,

  // Hysteresis
  applyHysteresis,
  updateHysteresisHistory,

  // Evaluation
  evaluateLOD,
  refreshLOD,
  switchLODLevel,
  forceLODLevel,
  clearForcedLOD,
  isLODForced,

  // Transition
  updateLODTransitions,
  interpolateLODTransition,
  startLODTransition,
  isLODTransitioning,

  // Geometry
  selectLODGeometry,
  selectLODGeometryKind,
  selectLODTriangleCount,
  selectLODMaterialId,

  // Impostor & proxy
  bindLODImpostor,
  bindLODProxy,

  // Group
  registerLODGroupAnchor,
  attachLODGroupMember,
  detachLODGroupMember,
  evaluateGroupLOD,

  // Budget
  setLODBudgetCap,
  getLODBudgetCap,
  updateLODBudget,
  getLODVisibleCost,
  isLODBudgetExceeded,
  applyBudgetDowngrade,
  clearBudgetDowngrade,

  // Convenience
  refreshLODForLight,
  refreshLODForShadowCaster,
  refreshLODForGIProbe,
  refreshLODForAOVolume,

  // Bulk
  evaluateAllLOD,

  // Frame
  tickLOD,
  tickLODSystem,

  // Diagnostics
  getLODStats,
  getLODSystemReport,

  // Registration
  registerLODComponents,

  // Reset
  resetLODState,
};

export default _defaultExport;