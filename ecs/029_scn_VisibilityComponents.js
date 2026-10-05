// File : 029
// name : src/ecs/029_scn_VisibilityComponents.js
// description : Visibility and culling SoA component module for the scene ECS
//               world of the anime lighting stack on Android mobile. Declares
//               every per-entity visibility state, frustum plane, frustum
//               sphere, occlusion query, Hi-Z pyramid record, visibility
//               group, and culling budget the lighting stack needs — as
//               fixed-capacity typed arrays sized once to
//               MAX_ENTITIES = 100000.
//
//               Provides the fast helpers that every culling system runs on
//               the hot path:
//                 • computeVisibility         — full pipeline for one entity
//                 • testSpherePlane           — sphere vs single frustum plane
//                 • testAABBPlane             — AABB vs single frustum plane
//                 • testSphereFrustum         — sphere vs all 6 planes
//                 • testAABBFrustum           — AABB vs all 6 planes
//                 • beginOcclusionQuery       — start a hardware occlusion query
//                 • endOcclusionQuery         — resolve the previous query
//                 • isOccluded                — read the query result
//                 • pushHiZLevel              — record a Hi-Z pyramid level
//                 • popHiZLevel               — pop the last pyramid level
//                 • evaluateVisibilityGroup   — cull a whole group as one unit
//                 • updateVisibilityBudget    — recompute the aggregate cost
//                 • tickVisibility            — one-shot per-frame pipeline
//                 • setFrustumFromCamera      — populate the 6 planes from a
//                                               camera + projection matrix
//                 • testPointFrustum          — fast point-in-frustum test
//                 • computeTemporalCoherence  — reproject last-frame visibility
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no runtime resizes.
//                 • Frustum planes stored as flat Float32Array[6][4] with
//                   plane-distance early-exit in the culling loop.
//                 • Two-level occlusion culling: coarse hardware query on
//                   candidate objects, fine Hi-Z pyramid test on resolved
//                   geometry.
//                 • Temporal coherence: previous-frame visibility cached
//                   per entity and reused when the camera moved less than a
//                   threshold.
//                 • Hysteresis band: entities at the frustum edge retain
//                   their visible state for a few frames to prevent
//                   flicker.
//                 • Group culling: entities registered in a visibility group
//                   share a single group-level frustum/occlusion decision.
//                 • Budget-aware: VisibilityBudget caps the number of
//                   visible entities per frame so adaptive quality can
//                   throttle without ever dropping an entity mid-frame.
//                 • Zero allocations on the hot path — every helper works
//                   directly on the SoA arrays; scratch vectors and frustum
//                   planes are module-level and reused.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — lights culled by visibility
//                 • 003_lgt_ShadowComponents.js     — casters culled by visibility
//                 • 004_lgt_GIComponents.js         — GI probes culled by visibility
//                 • 005_lgt_AOComponents.js         — AO volumes culled by visibility
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — visibility tags
//                 • 015_scn_Relations.js            — group relations
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime records
//                 • 025_scn_SpatialComponents.js    — AABB / sphere bounds
//                 • 026_scn_TransformComponents.js  — world transforms
//                 • 027_scn_LODComponents.js        — LOD-aware culling
//                 • 028_scn_StreamingComponents.js  — chunk visibility
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every light, shadow caster, GI probe, AO
//            volume, and mesh in the anime lighting stack is culled
//            deterministically with frustum + occlusion + temporal
//            coherence + group awareness, all behind a bounded per-frame
//            culling budget — with zero allocations on the hot path and
//            no flicker at frustum edges.
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
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

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
  EntityTag,
  EntityTag2,
  markFrameDirty,
  clearFrameDirty,
  isFrameDirty,
} from './014_scn_Tags.js';

import {
  Parent,
  Children,
  MAX_CHILDREN_PER_ENTITY,
  NULL_ENTITY,
  REF,
  setReference,
  resolveReference,
} from './015_scn_Relations.js';

import {
  AABB,
  Sphere,
  FrustumPlanes,
  FRUSTUM_PLANE,
  FRUSTUM_RESULT,
  SPATIAL_FLAG,
  VisibilityState as SpatialVisibilityState,
  testFrustumSphere,
  testFrustumAABB,
} from './025_scn_SpatialComponents.js';

import {
  TransformWorld,
} from './026_scn_TransformComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of Hi-Z pyramid levels. Log2 of the max viewport
 * dimension. At 4K that is log2(3840) ≈ 12 levels; we allocate 16.
 */
export const MAX_HIZ_LEVELS = 16;

/**
 * Maximum number of members in one visibility group.
 */
export const MAX_VISIBILITY_GROUP_MEMBERS =
  PERF_TIER_LOCAL === 'HIGH'   ? 128 :
  PERF_TIER_LOCAL === 'MEDIUM' ?  96 :
                                  64;

/**
 * Number of frames an entity at the frustum edge retains its visibility
 * state (hysteresis).
 */
export const FRUSTUM_HYSTERESIS_FRAMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 3 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2 :
                                 1;

/**
 * Number of frames an entity retains its visibility state when the camera
 * moves less than TEMPORAL_COHERENCE_DISTANCE.
 */
export const TEMPORAL_COHERENCE_FRAMES = 2;

/**
 * Camera movement threshold (world units) below which temporal coherence
 * is used.
 */
export const TEMPORAL_COHERENCE_DISTANCE = 0.25;

/**
 * Maximum number of occlusion queries in flight per frame.
 */
export const MAX_OCCLUSION_QUERIES_IN_FLIGHT =
  PERF_TIER_LOCAL === 'HIGH'   ? 8 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 4 :
                                 2;

/**
 * Default per-frame visibility budget cap (in entities).
 */
export const DEFAULT_VISIBILITY_BUDGET =
  PERF_TIER_LOCAL === 'HIGH'   ? 4096 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2048 :
                                 1024;

/**
 * Visibility flags.
 */
export const VIS_FLAG = Object.freeze({
  NONE:            0,
  VISIBLE:         1 << 0,
  FRUSTUM_TESTED:  1 << 1,
  IN_FRUSTUM:      1 << 2,
  OCCLUSION_TESTED:1 << 3,
  OCCLUDED:        1 << 4,
  TEMPORAL_HOLD:   1 << 5,
  HYSTERESIS_HOLD: 1 << 6,
  GROUP_ANCHOR:    1 << 7,
  GROUP_MEMBER:    1 << 8,
  SKIP_CULLING:    1 << 9,
  BUDGET_CULLED:   1 << 10,
  FORCED_VISIBLE:  1 << 11,
  FORCED_HIDDEN:   1 << 12,
});

/**
 * Occlusion query state.
 */
export const OCCLUSION_STATE = Object.freeze({
  NONE:       0,
  PENDING:    1,
  RESOLVED:   2,
  TIMEOUT:    3,
  NOT_READY:  4,
  UNAVAILABLE:5,
  COUNT:      6,
});

export const OCCLUSION_STATE_NAME = Object.freeze([
  'none',
  'pending',
  'resolved',
  'timeout',
  'not_ready',
  'unavailable',
]);

/**
 * Culling result reasons (debug HUD).
 */
export const CULL_REASON = Object.freeze({
  NONE:          0,
  VISIBLE:       1,
  FRUSTUM_OUT:   2,
  OCCLUDED:      3,
  BUDGET:        4,
  FORCED_HIDDEN: 5,
  SMALL_SCREEN:  6,
  BEHIND_CAMERA: 7,
});

export const CULL_REASON_NAME = Object.freeze([
  'none',
  'visible',
  'frustum_out',
  'occluded',
  'budget',
  'forced_hidden',
  'small_screen',
  'behind_camera',
]);

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const VisibilityState = {
  frame:                    0,
  budgetCap:                DEFAULT_VISIBILITY_BUDGET,
  budgetUsed:               0,
  budgetExceeded:           0,
  totalEvaluations:         0,
  totalFrustumTests:        0,
  totalOcclusionTests:      0,
  totalTemporalHolds:       0,
  totalHysteresisHolds:     0,
  totalBudgetCulls:         0,
  totalGroupEvaluations:    0,
  visibleCount:             0,
  peakVisibleCount:         0,
  occludedCount:            0,
  lastEvalMs:               0,
  avgEvalMs:                0,
  lastFrustumMs:            0,
  avgFrustumMs:             0,
  lastOcclusionMs:          0,
  avgOcclusionMs:           0,
  lastCameraX:              0,
  lastCameraY:              0,
  lastCameraZ:              0,
  cameraStationary:         0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.visibility', {
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
 * VisibilityState — the canonical per-entity visibility record.
 */
export const VisibilityEntityState = {
  flags:              new Uint32Array(MAX_ENTITIES),
  prevFlags:          new Uint32Array(MAX_ENTITIES),
  frustumResult:      new Uint8Array(MAX_ENTITIES),
  occlusionState:     new Uint8Array(MAX_ENTITIES),
  cullReason:         new Uint8Array(MAX_ENTITIES),

  visibleFrames:      new Uint16Array(MAX_ENTITIES),
  hiddenFrames:       new Uint16Array(MAX_ENTITIES),
  hysteresisCounter:  new Uint8Array(MAX_ENTITIES),

  lastFrustumFrame:   new Uint32Array(MAX_ENTITIES),
  lastOcclusionFrame: new Uint32Array(MAX_ENTITIES),
  lastVisibleFrame:   new Uint32Array(MAX_ENTITIES),
  lastHiddenFrame:    new Uint32Array(MAX_ENTITIES),

  screenSize:         new Float32Array(MAX_ENTITIES),
  distanceToCam:      new Float32Array(MAX_ENTITIES),
  distanceSq:         new Float32Array(MAX_ENTITIES),
  cameraDelta:        new Float32Array(MAX_ENTITIES),
};

/**
 * FrustumPlane — the 6 planes of the current frustum stored as a flat
 * Float32Array[6 * 4]. Each plane is (nx, ny, nz, d) with d being the
 * plane offset.
 */
export const FrustumPlane = {
  planes:      new Float32Array(FRUSTUM_PLANE.COUNT * 4),
  cameraX:     new Float32Array(1),
  cameraY:     new Float32Array(1),
  cameraZ:     new Float32Array(1),
  cameraDirX:  new Float32Array(1),
  cameraDirY:  new Float32Array(1),
  cameraDirZ:  new Float32Array(1),
  nearPlane:   new Float32Array(1),
  farPlane:    new Float32Array(1),
  aspect:      new Float32Array(1),
  tanFovHalf:  new Float32Array(1),
  valid:       new Uint8Array(1),
  generation:  new Uint32Array(1),
};

/**
 * FrustumSphere — an optional bounding sphere representation used for
 * fast coarse culling passes.
 */
export const FrustumSphere = {
  centerX:     new Float32Array(1),
  centerY:     new Float32Array(1),
  centerZ:     new Float32Array(1),
  radius:      new Float32Array(1),
  radiusSq:    new Float32Array(1),
  valid:       new Uint8Array(1),
};

/**
 * OcclusionQuery — per-entity hardware occlusion query state.
 */
export const OcclusionQuery = {
  queryId:          new Uint32Array(MAX_ENTITIES),
  queryGeneration:  new Uint32Array(MAX_ENTITIES),
  startFrame:       new Uint32Array(MAX_ENTITIES),
  resolveFrame:     new Uint32Array(MAX_ENTITIES),
  sampleCount:      new Uint32Array(MAX_ENTITIES),
  passedSamples:    new Uint32Array(MAX_ENTITIES),
  visibilityRatio:  new Float32Array(MAX_ENTITIES),
  state:            new Uint8Array(MAX_ENTITIES),
  inFlight:         new Uint8Array(MAX_ENTITIES),
  timeout:          new Uint16Array(MAX_ENTITIES),
  enabled:          new Uint8Array(MAX_ENTITIES),
};

/**
 * HiZPyramid — Hi-Z (hierarchical Z) pyramid level records per entity.
 * Each entity can have up to MAX_HIZ_LEVELS pyramid entries.
 */
export const HiZPyramid = {
  levelWidth:       new Uint16Array(MAX_ENTITIES * MAX_HIZ_LEVELS),
  levelHeight:      new Uint16Array(MAX_ENTITIES * MAX_HIZ_LEVELS),
  minDepth:         new Float32Array(MAX_ENTITIES * MAX_HIZ_LEVELS),
  maxDepth:         new Float32Array(MAX_ENTITIES * MAX_HIZ_LEVELS),
  levelCount:       new Uint8Array(MAX_ENTITIES),
  baseWidth:        new Uint16Array(MAX_ENTITIES),
  baseHeight:       new Uint16Array(MAX_ENTITIES),
  generation:       new Uint32Array(MAX_ENTITIES),
  valid:            new Uint8Array(MAX_ENTITIES),
};

/**
 * VisibilityGroup — group anchor and aggregate bounding volume.
 */
export const VisibilityGroup = {
  anchorEid:      new Int32Array(MAX_ENTITIES).fill(-1),
  memberCount:    new Uint16Array(MAX_ENTITIES),
  groupRadius:    new Float32Array(MAX_ENTITIES),
  groupCenterX:   new Float32Array(MAX_ENTITIES),
  groupCenterY:   new Float32Array(MAX_ENTITIES),
  groupCenterZ:   new Float32Array(MAX_ENTITIES),
  groupFlags:     new Uint16Array(MAX_ENTITIES),
  groupDirty:     new Uint8Array(MAX_ENTITIES),
  groupVisible:   new Uint8Array(MAX_ENTITIES),
};

/**
 * VisibilityGroupMember — a member's index within its visibility group.
 */
export const VisibilityGroupMember = {
  groupEid:      new Int32Array(MAX_ENTITIES).fill(-1),
  memberIndex:   new Uint8Array(MAX_ENTITIES),
  localWeight:   new Float32Array(MAX_ENTITIES).fill(1.0),
};

/**
 * VisibilityBudget — per-frame culling budget and aggregate cost.
 */
export const VisibilityBudget = {
  entityCost:      new Float32Array(MAX_ENTITIES),
  entityCostEma:   new Float32Array(MAX_ENTITIES),
  totalCost:       new Float32Array(1),
  budgetCap:       new Float32Array(1),
  budgetExceeded:  new Uint8Array(1),
  budgetUsed:      new Uint32Array(1),
  budgetCulls:     new Uint32Array(1),
};

/**
 * CullingStats — aggregate per-frame statistics.
 */
export const CullingStats = {
  visible:            new Uint32Array(1),
  hidden:             new Uint32Array(1),
  frustumCulled:      new Uint32Array(1),
  occlusionCulled:    new Uint32Array(1),
  budgetCulled:       new Uint32Array(1),
  temporalHolds:      new Uint32Array(1),
  hysteresisHolds:    new Uint32Array(1),
  groupCulled:        new Uint32Array(1),
  totalEvaluated:     new Uint32Array(1),
  histogram:          new Uint32Array(CULL_REASON_NAME.length),
};

/**
 * CullingScratch — reusable scratch buffers for culling passes.
 * Pre-allocated once; never resized.
 */
export const CullingScratch = {
  // Buffer of entity ids considered for culling this frame.
  candidateIds:  new Int32Array(MAX_ENTITIES),
  candidateCount:new Uint32Array(1),
  // Buffer of entities that passed frustum culling.
  frustumPassed: new Int32Array(MAX_ENTITIES),
  frustumPassedCount: new Uint32Array(1),
  // Buffer of entities that passed occlusion culling.
  occlusionPassed: new Int32Array(MAX_ENTITIES),
  occlusionPassedCount: new Uint32Array(1),
  // Sort key buffer.
  sortKeys:      new Float32Array(MAX_ENTITIES),
};

/**
 * Visibility component bundle for bitECS createWorld.
 */
export const VISIBILITY_COMPONENTS = Object.freeze({
  VisibilityEntityState,
  FrustumPlane,
  FrustumSphere,
  OcclusionQuery,
  HiZPyramid,
  VisibilityGroup,
  VisibilityGroupMember,
  VisibilityBudget,
  CullingStats,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Module-level scratch vectors. Reused on every call.
 */
const _scratchVec3 = new Float32Array(3);
const _scratchVec3B = new Float32Array(3);

/**
 * Frustum plane base offsets (into FrustumPlane.planes).
 */
const _planeOffsetLeft    = FRUSTUM_PLANE.LEFT    * 4;
const _planeOffsetRight   = FRUSTUM_PLANE.RIGHT   * 4;
const _planeOffsetBottom  = FRUSTUM_PLANE.BOTTOM  * 4;
const _planeOffsetTop     = FRUSTUM_PLANE.TOP     * 4;
const _planeOffsetNear    = FRUSTUM_PLANE.NEAR    * 4;
const _planeOffsetFar     = FRUSTUM_PLANE.FAR     * 4;

/* ------------------------------------------------------------------ */
/* 4. FRUSTUM MANAGEMENT                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets a frustum plane by index.
 */
export function setFrustumPlane(planeIndex, nx, ny, nz, d) {
  if (planeIndex < 0 || planeIndex >= FRUSTUM_PLANE.COUNT) return false;
  const base = planeIndex * 4;
  FrustumPlane.planes[base + 0] = nx;
  FrustumPlane.planes[base + 1] = ny;
  FrustumPlane.planes[base + 2] = nz;
  FrustumPlane.planes[base + 3] = d;
  FrustumPlane.valid[0] = 1;
  return true;
}

/**
 * Sets the frustum camera metadata.
 */
export function setFrustumCamera(cameraX, cameraY, cameraZ, dirX, dirY, dirZ, nearPlane, farPlane, aspect, tanFovHalf) {
  FrustumPlane.cameraX[0] = cameraX;
  FrustumPlane.cameraY[0] = cameraY;
  FrustumPlane.cameraZ[0] = cameraZ;
  FrustumPlane.cameraDirX[0] = dirX;
  FrustumPlane.cameraDirY[0] = dirY;
  FrustumPlane.cameraDirZ[0] = dirZ;
  FrustumPlane.nearPlane[0] = nearPlane;
  FrustumPlane.farPlane[0] = farPlane;
  FrustumPlane.aspect[0] = aspect;
  FrustumPlane.tanFovHalf[0] = tanFovHalf;
  FrustumPlane.valid[0] = 1;
  FrustumPlane.generation[0] = (FrustumPlane.generation[0] + 1) >>> 0;
  return true;
}

/**
 * Builds the 6 frustum planes from a perspective camera's parameters.
 * Uses the standard Gribb-Hartmann extraction method.
 */
export function buildPerspectiveFrustum(cameraX, cameraY, cameraZ, dirX, dirY, dirZ, upX, upY, upZ, fovY, aspect, nearPlane, farPlane) {
  const tanHalfFovY = Math.tan(fovY * 0.5);
  const tanHalfFovX = tanHalfFovY * aspect;

  // Compute right vector = normalize(dir × up).
  let rX = dirY * upZ - dirZ * upY;
  let rY = dirZ * upX - dirX * upZ;
  let rZ = dirX * upY - dirY * upX;
  const rLen = Math.sqrt(rX * rX + rY * rY + rZ * rZ) || 1;
  rX /= rLen; rY /= rLen; rZ /= rLen;

  // Recomputed up = right × dir.
  let uX = rY * dirZ - rZ * dirY;
  let uY = rZ * dirX - rX * dirZ;
  let uZ = rX * dirY - rY * dirX;

  // Left plane: normal = normalize(dir * tanHalfFovX + right), d = -normal · camPos - ... (see derivation)
  // Simpler derivation: plane through camera axis:
  //   left  : N = normalize(dir + right / tanHalfFovX) — approximate sign
  //   right : N = normalize(dir - right / tanHalfFovX)
  //   bottom: N = normalize(dir + up / tanHalfFovY)
  //   top   : N = normalize(dir - up / tanHalfFovY)
  //   near  : N = dir,  d = -(camPos · dir) + nearPlane
  //   far   : N = -dir, d = (camPos · dir) - farPlane  => N = -dir, d = (camPos · dir) + farPlane

  const camLen = cameraX * dirX + cameraY * dirY + cameraZ * dirZ;

  // Left plane (positive side = inside).
  let lx = dirX + rX / Math.max(1e-6, tanHalfFovX);
  let ly = dirY + rY / Math.max(1e-6, tanHalfFovX);
  let lz = dirZ + rZ / Math.max(1e-6, tanHalfFovX);
  let lLen = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
  lx /= lLen; ly /= lLen; lz /= lLen;
  const ld = -(lx * cameraX + ly * cameraY + lz * cameraZ);
  setFrustumPlane(FRUSTUM_PLANE.LEFT, lx, ly, lz, ld);

  // Right plane.
  let rx = dirX - rX / Math.max(1e-6, tanHalfFovX);
  let ry = dirY - rY / Math.max(1e-6, tanHalfFovX);
  let rz = dirZ - rZ / Math.max(1e-6, tanHalfFovX);
  let rLen2 = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
  rx /= rLen2; ry /= rLen2; rz /= rLen2;
  const rd = -(rx * cameraX + ry * cameraY + rz * cameraZ);
  setFrustumPlane(FRUSTUM_PLANE.RIGHT, rx, ry, rz, rd);

  // Bottom plane.
  let bx = dirX + uX / Math.max(1e-6, tanHalfFovY);
  let by = dirY + uY / Math.max(1e-6, tanHalfFovY);
  let bz = dirZ + uZ / Math.max(1e-6, tanHalfFovY);
  let bLen = Math.sqrt(bx * bx + by * by + bz * bz) || 1;
  bx /= bLen; by /= bLen; bz /= bLen;
  const bd = -(bx * cameraX + by * cameraY + bz * cameraZ);
  setFrustumPlane(FRUSTUM_PLANE.BOTTOM, bx, by, bz, bd);

  // Top plane.
  let tx = dirX - uX / Math.max(1e-6, tanHalfFovY);
  let ty = dirY - uY / Math.max(1e-6, tanHalfFovY);
  let tz = dirZ - uZ / Math.max(1e-6, tanHalfFovY);
  let tLen = Math.sqrt(tx * tx + ty * ty + tz * tz) || 1;
  tx /= tLen; ty /= tLen; tz /= tLen;
  const td = -(tx * cameraX + ty * cameraY + tz * cameraZ);
  setFrustumPlane(FRUSTUM_PLANE.TOP, tx, ty, tz, td);

  // Near plane: N = dir, d = -(camPos · dir) + nearPlane.
  setFrustumPlane(FRUSTUM_PLANE.NEAR, dirX, dirY, dirZ, -camLen + nearPlane);

  // Far plane: N = -dir, d = camLen + farPlane.
  setFrustumPlane(FRUSTUM_PLANE.FAR, -dirX, -dirY, -dirZ, camLen + farPlane);

  // Update the frustum camera metadata.
  setFrustumCamera(
    cameraX, cameraY, cameraZ,
    dirX, dirY, dirZ,
    nearPlane, farPlane,
    aspect, tanHalfFovY
  );

  return true;
}

/* ------------------------------------------------------------------ */
/* 5. FRUSTUM PLANE TESTS                                             */
/* ------------------------------------------------------------------ */

/**
 * Tests a sphere against a single frustum plane.
 * Returns:
 *   > 0 : fully inside
 *   = 0 : intersecting
 *   < 0 : fully outside
 */
export function testSpherePlane(cx, cy, cz, radius, planeIndex) {
  if (planeIndex < 0 || planeIndex >= FRUSTUM_PLANE.COUNT) return 0;
  const base = planeIndex * 4;
  const nx = FrustumPlane.planes[base + 0];
  const ny = FrustumPlane.planes[base + 1];
  const nz = FrustumPlane.planes[base + 2];
  const d  = FrustumPlane.planes[base + 3];

  const dist = nx * cx + ny * cy + nz * cz + d;
  if (dist < -radius) return -1;
  if (dist > radius) return 1;
  return 0;
}

/**
 * Tests an AABB against a single frustum plane.
 */
export function testAABBPlane(minX, minY, minZ, maxX, maxY, maxZ, planeIndex) {
  if (planeIndex < 0 || planeIndex >= FRUSTUM_PLANE.COUNT) return 0;
  const base = planeIndex * 4;
  const nx = FrustumPlane.planes[base + 0];
  const ny = FrustumPlane.planes[base + 1];
  const nz = FrustumPlane.planes[base + 2];
  const d  = FrustumPlane.planes[base + 3];

  // Most-positive corner.
  const px = nx >= 0 ? maxX : minX;
  const py = ny >= 0 ? maxY : minY;
  const pz = nz >= 0 ? maxZ : minZ;
  const dMax = nx * px + ny * py + nz * pz + d;
  if (dMax < 0) return -1;

  // Most-negative corner.
  const qx = nx >= 0 ? minX : maxX;
  const qy = ny >= 0 ? minY : maxY;
  const qz = nz >= 0 ? minZ : maxZ;
  const dMin = nx * qx + ny * qy + nz * qz + d;
  if (dMin < 0) return 0;

  return 1;
}

/**
 * Tests a sphere against every frustum plane. Returns a FRUSTUM_RESULT.
 */
export function testSphereFrustum(cx, cy, cz, radius) {
  if (FrustumPlane.valid[0] === 0) return FRUSTUM_RESULT.INSIDE;

  let result = FRUSTUM_RESULT.INSIDE;
  for (let p = 0; p < FRUSTUM_PLANE.COUNT; p++) {
    const r = testSpherePlane(cx, cy, cz, radius, p);
    if (r < 0) return FRUSTUM_RESULT.OUTSIDE;
    if (r === 0) result = FRUSTUM_RESULT.INTERSECTING;
  }
  return result;
}

/**
 * Tests an AABB against every frustum plane.
 */
export function testAABBFrustum(minX, minY, minZ, maxX, maxY, maxZ) {
  if (FrustumPlane.valid[0] === 0) return FRUSTUM_RESULT.INSIDE;

  let result = FRUSTUM_RESULT.INSIDE;
  for (let p = 0; p < FRUSTUM_PLANE.COUNT; p++) {
    const r = testAABBPlane(minX, minY, minZ, maxX, maxY, maxZ, p);
    if (r < 0) return FRUSTUM_RESULT.OUTSIDE;
    if (r === 0) result = FRUSTUM_RESULT.INTERSECTING;
  }
  return result;
}

/**
 * Fast point-in-frustum test.
 */
export function testPointFrustum(px, py, pz) {
  if (FrustumPlane.valid[0] === 0) return true;
  for (let p = 0; p < FRUSTUM_PLANE.COUNT; p++) {
    const base = p * 4;
    const nx = FrustumPlane.planes[base + 0];
    const ny = FrustumPlane.planes[base + 1];
    const nz = FrustumPlane.planes[base + 2];
    const d  = FrustumPlane.planes[base + 3];
    if (nx * px + ny * py + nz * pz + d < 0) return false;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. OCCLUSION QUERY                                                 */
/* ------------------------------------------------------------------ */

/**
 * Enables hardware occlusion queries for an entity.
 */
export function enableOcclusionQuery(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  OcclusionQuery.enabled[eid] = 1;
  OcclusionQuery.state[eid] = OCCLUSION_STATE.NONE;
  return true;
}

/**
 * Disables hardware occlusion queries for an entity.
 */
export function disableOcclusionQuery(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  OcclusionQuery.enabled[eid] = 0;
  OcclusionQuery.inFlight[eid] = 0;
  OcclusionQuery.state[eid] = OCCLUSION_STATE.UNAVAILABLE;
  return true;
}

/**
 * Begins an occlusion query for an entity. Returns true on success.
 * The caller is expected to issue the draw call and then call
 * `endOcclusionQuery(eid)` when it has been submitted.
 */
export function beginOcclusionQuery(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (OcclusionQuery.enabled[eid] === 0) return false;
  if (OcclusionQuery.inFlight[eid] === 1) return false;

  OcclusionQuery.queryId[eid] = (OcclusionQuery.queryId[eid] + 1) >>> 0;
  OcclusionQuery.queryGeneration[eid] = (OcclusionQuery.queryGeneration[eid] + 1) >>> 0;
  OcclusionQuery.startFrame[eid] = VisibilityState.frame;
  OcclusionQuery.state[eid] = OCCLUSION_STATE.PENDING;
  OcclusionQuery.inFlight[eid] = 1;
  OcclusionQuery.passedSamples[eid] = 0;
  OcclusionQuery.sampleCount[eid] = 0;
  OcclusionQuery.visibilityRatio[eid] = 0;
  OcclusionQuery.timeout[eid] = 0;

  VisibilityState.totalOcclusionTests++;
  return true;
}

/**
 * Ends an occlusion query for an entity and marks it ready for resolve.
 */
export function endOcclusionQuery(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (OcclusionQuery.inFlight[eid] === 0) return false;
  // The query is submitted; the actual result is resolved later.
  return true;
}

/**
 * Resolves an occlusion query with the sample count returned by the GPU.
 * Typically called from `getQueryParameter` when the result becomes
 * available.
 */
export function resolveOcclusionQuery(eid, passedSamples, sampleCount) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (OcclusionQuery.inFlight[eid] === 0) return false;

  OcclusionQuery.passedSamples[eid] = (passedSamples | 0) >>> 0;
  OcclusionQuery.sampleCount[eid] = (sampleCount | 0) >>> 0;
  OcclusionQuery.visibilityRatio[eid] = sampleCount > 0 ? passedSamples / sampleCount : 0;
  OcclusionQuery.resolveFrame[eid] = VisibilityState.frame;
  OcclusionQuery.state[eid] = OCCLUSION_STATE.RESOLVED;
  OcclusionQuery.inFlight[eid] = 0;

  VisibilityEntityState.occlusionState[eid] = OCCLUSION_STATE.RESOLVED;
  VisibilityEntityState.lastOcclusionFrame[eid] = VisibilityState.frame;

  return true;
}

/**
 * Marks the occlusion query as timed out.
 */
export function timeoutOcclusionQuery(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  OcclusionQuery.inFlight[eid] = 0;
  OcclusionQuery.state[eid] = OCCLUSION_STATE.TIMEOUT;
  return true;
}

/**
 * Returns true if the entity is currently considered occluded based on
 * the last resolved query.
 */
export function isOccluded(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (OcclusionQuery.state[eid] !== OCCLUSION_STATE.RESOLVED) return false;
  // Fully occluded when the visibility ratio is below a threshold.
  return OcclusionQuery.visibilityRatio[eid] < 0.5;
}

/**
 * Returns the entity's last visibility ratio [0, 1].
 */
export function getOcclusionVisibility(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 1.0;
  return OcclusionQuery.visibilityRatio[eid];
}

/* ------------------------------------------------------------------ */
/* 7. Hi-Z PYRAMID                                                    */
/* ------------------------------------------------------------------ */

/**
 * Initializes the Hi-Z pyramid for an entity. `baseWidth` and `baseHeight`
 * are the base texture dimensions; the pyramid descends by successive
 * halving until a 1×1 level.
 */
export function initHiZPyramid(eid, baseWidth, baseHeight) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  HiZPyramid.baseWidth[eid] = (baseWidth | 0) & 0xFFFF;
  HiZPyramid.baseHeight[eid] = (baseHeight | 0) & 0xFFFF;

  let level = 0;
  let w = baseWidth;
  let h = baseHeight;

  while (level < MAX_HIZ_LEVELS && (w >= 1 || h >= 1)) {
    HiZPyramid.levelWidth[eid * MAX_HIZ_LEVELS + level] = Math.max(1, w) & 0xFFFF;
    HiZPyramid.levelHeight[eid * MAX_HIZ_LEVELS + level] = Math.max(1, h) & 0xFFFF;
    HiZPyramid.minDepth[eid * MAX_HIZ_LEVELS + level] = 1.0;
    HiZPyramid.maxDepth[eid * MAX_HIZ_LEVELS + level] = 0.0;
    w = w >> 1;
    h = h >> 1;
    level++;
    if (w === 0 && h === 0) break;
  }

  HiZPyramid.levelCount[eid] = level;
  HiZPyramid.generation[eid] = (HiZPyramid.generation[eid] + 1) >>> 0;
  HiZPyramid.valid[eid] = 1;
  return true;
}

/**
 * Records the min/max depth for a Hi-Z pyramid level.
 */
export function pushHiZLevel(eid, level, minDepth, maxDepth) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (level < 0 || level >= MAX_HIZ_LEVELS) return false;
  const off = eid * MAX_HIZ_LEVELS + level;
  HiZPyramid.minDepth[off] = minDepth;
  HiZPyramid.maxDepth[off] = maxDepth;
  return true;
}

/**
 * Reads the min/max depth for a Hi-Z pyramid level into out[0]/out[1].
 */
export function popHiZLevel(eid, level, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (level < 0 || level >= MAX_HIZ_LEVELS) return false;
  const off = eid * MAX_HIZ_LEVELS + level;
  if (out) {
    out[0] = HiZPyramid.minDepth[off];
    out[1] = HiZPyramid.maxDepth[off];
  }
  return true;
}

/**
 * Returns true if the entity has a valid Hi-Z pyramid.
 */
export function hasHiZPyramid(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return HiZPyramid.valid[eid] === 1;
}

/**
 * Returns the number of levels in the entity's Hi-Z pyramid.
 */
export function getHiZLevelCount(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  return HiZPyramid.levelCount[eid];
}

/* ------------------------------------------------------------------ */
/* 8. VISIBILITY GROUPS                                               */
/* ------------------------------------------------------------------ */

/**
 * Registers an entity as the anchor of a visibility group.
 */
export function registerVisibilityGroupAnchor(anchorEid) {
  if (typeof anchorEid !== 'number' || anchorEid < 0 || anchorEid >= MAX_ENTITIES) return false;
  VisibilityGroup.anchorEid[anchorEid] = anchorEid;
  VisibilityGroup.memberCount[anchorEid] = 0;
  VisibilityGroup.groupRadius[anchorEid] = 0;
  VisibilityGroup.groupDirty[anchorEid] = 1;
  VisibilityEntityState.flags[anchorEid] |= VIS_FLAG.GROUP_ANCHOR;
  return true;
}

/**
 * Attaches a member entity to a visibility group.
 */
export function attachVisibilityGroupMember(groupAnchorEid, memberEid, localWeight) {
  if (typeof groupAnchorEid !== 'number' || groupAnchorEid < 0 || groupAnchorEid >= MAX_ENTITIES) return false;
  if (typeof memberEid !== 'number' || memberEid < 0 || memberEid >= MAX_ENTITIES) return false;

  if (VisibilityGroup.memberCount[groupAnchorEid] >= MAX_VISIBILITY_GROUP_MEMBERS) return false;

  VisibilityGroupMember.groupEid[memberEid] = groupAnchorEid;
  VisibilityGroupMember.memberIndex[memberEid] = VisibilityGroup.memberCount[groupAnchorEid];
  VisibilityGroupMember.localWeight[memberEid] = Number.isFinite(localWeight) ? localWeight : 1.0;
  VisibilityGroup.memberCount[groupAnchorEid]++;
  VisibilityEntityState.flags[memberEid] |= VIS_FLAG.GROUP_MEMBER;

  // Relation edge so the group survives serialization.
  setReference(memberEid, groupAnchorEid, REF.GROUP_MEMBER, 1.0);

  return true;
}

/**
 * Detaches a member from its visibility group.
 */
export function detachVisibilityGroupMember(memberEid) {
  if (typeof memberEid !== 'number' || memberEid < 0 || memberEid >= MAX_ENTITIES) return false;
  const groupEid = VisibilityGroupMember.groupEid[memberEid];
  if (groupEid < 0) return false;

  if (VisibilityGroup.memberCount[groupEid] > 0) {
    VisibilityGroup.memberCount[groupEid]--;
  }
  VisibilityGroupMember.groupEid[memberEid] = -1;
  VisibilityEntityState.flags[memberEid] &= ~VIS_FLAG.GROUP_MEMBER;
  return true;
}

/**
 * Evaluates a visibility group as a single unit. All members share the
 * group's visibility decision.
 *
 * Returns the group's visibility result (FRUSTUM_RESULT).
 */
export function evaluateVisibilityGroup(groupAnchorEid) {
  if (typeof groupAnchorEid !== 'number' || groupAnchorEid < 0 || groupAnchorEid >= MAX_ENTITIES) {
    return FRUSTUM_RESULT.OUTSIDE;
  }

  VisibilityState.totalGroupEvaluations++;

  const cx = VisibilityGroup.groupCenterX[groupAnchorEid];
  const cy = VisibilityGroup.groupCenterY[groupAnchorEid];
  const cz = VisibilityGroup.groupCenterZ[groupAnchorEid];
  const radius = VisibilityGroup.groupRadius[groupAnchorEid];

  const result = testSphereFrustum(cx, cy, cz, radius);

  const visible = result !== FRUSTUM_RESULT.OUTSIDE;
  VisibilityGroup.groupVisible[groupAnchorEid] = visible ? 1 : 0;

  // Propagate to every member.
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (VisibilityGroupMember.groupEid[eid] !== groupAnchorEid) continue;

    if (visible) {
      VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE | VIS_FLAG.IN_FRUSTUM | VIS_FLAG.FRUSTUM_TESTED;
      VisibilityEntityState.flags[eid] &= ~VIS_FLAG.OCCLUDED;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
    } else {
      VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.FRUSTUM_OUT;
      VisibilityState.totalFrustumTests++;
      CullingStats.frustumCulled[0]++;
    }
  }

  CullingStats.groupCulled[0] += visible ? 0 : 1;

  return result;
}

/* ------------------------------------------------------------------ */
/* 9. TEMPORAL COHERENCE                                              */
/* ------------------------------------------------------------------ */

/**
 * Tests whether the camera has moved less than the temporal coherence
 * threshold.
 */
export function isCameraStationary(cameraX, cameraY, cameraZ) {
  const dx = cameraX - VisibilityState.lastCameraX;
  const dy = cameraY - VisibilityState.lastCameraY;
  const dz = cameraZ - VisibilityState.lastCameraZ;
  const dSq = dx * dx + dy * dy + dz * dz;
  return dSq < TEMPORAL_COHERENCE_DISTANCE * TEMPORAL_COHERENCE_DISTANCE;
}

/**
 * Updates the camera position cache.
 */
export function updateCameraCache(cameraX, cameraY, cameraZ) {
  VisibilityState.lastCameraX = cameraX;
  VisibilityState.lastCameraY = cameraY;
  VisibilityState.lastCameraZ = cameraZ;
  VisibilityState.cameraStationary = isCameraStationary(cameraX, cameraY, cameraZ) ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* 10. VISIBILITY PIPELINE                                            */
/* ------------------------------------------------------------------ */

/**
 * Evaluates the visibility of a single entity. Runs frustum culling,
 * occlusion culling, temporal coherence, and hysteresis in one call.
 *
 * The entity's AABB/sphere must already be populated (via
 * `computeAABB` / `computeSphere` from 025).
 *
 * Returns the final visibility decision (VIS_FLAG bitmask).
 */
export function computeVisibility(eid, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return VIS_FLAG.NONE;

  VisibilityState.totalEvaluations++;
  CullingStats.totalEvaluated[0]++;

  const state = VisibilityEntityState.flags[eid];

  // Forced overrides.
  if ((state & VIS_FLAG.FORCED_VISIBLE) !== 0) {
    VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE;
    VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
    VisibilityState.visibleCount++;
    return VisibilityEntityState.flags[eid];
  }
  if ((state & VIS_FLAG.FORCED_HIDDEN) !== 0) {
    VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
    VisibilityEntityState.cullReason[eid] = CULL_REASON.FORCED_HIDDEN;
    return VisibilityEntityState.flags[eid];
  }
  if ((state & VIS_FLAG.SKIP_CULLING) !== 0) {
    VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE;
    VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
    VisibilityState.visibleCount++;
    return VisibilityEntityState.flags[eid];
  }

  // Group members defer to their group anchor.
  if ((state & VIS_FLAG.GROUP_MEMBER) !== 0) {
    const groupEid = VisibilityGroupMember.groupEid[eid];
    if (groupEid >= 0 && VisibilityGroup.groupVisible[groupEid] === 1) {
      VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
      VisibilityState.visibleCount++;
      return VisibilityEntityState.flags[eid];
    }
  }

  // ------ 1. Frustum culling ------
  let frustumResult = FRUSTUM_RESULT.INSIDE;
  if (FrustumPlane.valid[0] === 1) {
    if (Sphere.valid[eid] === 1) {
      frustumResult = testSphereFrustum(Sphere.cx[eid], Sphere.cy[eid], Sphere.cz[eid], Sphere.radius[eid]);
    } else if (AABB.valid[eid] === 1) {
      frustumResult = testAABBFrustum(
        AABB.minX[eid], AABB.minY[eid], AABB.minZ[eid],
        AABB.maxX[eid], AABB.maxY[eid], AABB.maxZ[eid]
      );
    }
  }

  VisibilityEntityState.frustumResult[eid] = frustumResult;
  VisibilityEntityState.lastFrustumFrame[eid] = VisibilityState.frame;
  VisibilityEntityState.flags[eid] |= VIS_FLAG.FRUSTUM_TESTED;
  VisibilityState.totalFrustumTests++;

  if (frustumResult === FRUSTUM_RESULT.OUTSIDE) {
    VisibilityEntityState.flags[eid] &= ~VIS_FLAG.IN_FRUSTUM;

    // Hysteresis: hold the previous decision for a few frames if we
    // were previously visible.
    const prevFlags = VisibilityEntityState.prevFlags[eid];
    if ((prevFlags & VIS_FLAG.VISIBLE) !== 0) {
      const counter = VisibilityEntityState.hysteresisCounter[eid];
      if (counter < FRUSTUM_HYSTERESIS_FRAMES) {
        VisibilityEntityState.hysteresisCounter[eid] = counter + 1;
        VisibilityEntityState.flags[eid] |= VIS_FLAG.HYSTERESIS_HOLD;
        if ((prevFlags & VIS_FLAG.VISIBLE) !== 0) {
          VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE;
          VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
          VisibilityState.visibleCount++;
          VisibilityState.totalHysteresisHolds++;
          return VisibilityEntityState.flags[eid];
        }
      } else {
        VisibilityEntityState.hysteresisCounter[eid] = 0;
      }
    }

    VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
    VisibilityEntityState.cullReason[eid] = CULL_REASON.FRUSTUM_OUT;
    VisibilityEntityState.hiddenFrames[eid]++;
    VisibilityEntityState.lastHiddenFrame[eid] = VisibilityState.frame;
    CullingStats.frustumCulled[0]++;
    return VisibilityEntityState.flags[eid];
  }

  VisibilityEntityState.hysteresisCounter[eid] = 0;
  VisibilityEntityState.flags[eid] |= VIS_FLAG.IN_FRUSTUM;

  // ------ 2. Temporal coherence ------
  if (VisibilityState.cameraStationary === 1) {
    const prevFlags = VisibilityEntityState.prevFlags[eid];
    if ((prevFlags & VIS_FLAG.VISIBLE) !== 0) {
      VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE | VIS_FLAG.TEMPORAL_HOLD;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
      VisibilityState.visibleCount++;
      VisibilityState.totalTemporalHolds++;
      VisibilityEntityState.lastVisibleFrame[eid] = VisibilityState.frame;
      VisibilityEntityState.visibleFrames[eid]++;
      return VisibilityEntityState.flags[eid];
    }
  }

  // ------ 3. Occlusion culling ------
  if (OcclusionQuery.enabled[eid] === 1) {
    if (OcclusionQuery.state[eid] === OCCLUSION_STATE.RESOLVED) {
      VisibilityEntityState.flags[eid] |= VIS_FLAG.OCCLUSION_TESTED;
      if (isOccluded(eid)) {
        VisibilityEntityState.flags[eid] |= VIS_FLAG.OCCLUDED;
        VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
        VisibilityEntityState.cullReason[eid] = CULL_REASON.OCCLUDED;
        VisibilityEntityState.hiddenFrames[eid]++;
        CullingStats.occlusionCulled[0]++;
        VisibilityState.occludedCount++;
        return VisibilityEntityState.flags[eid];
      } else {
        VisibilityEntityState.flags[eid] &= ~VIS_FLAG.OCCLUDED;
      }
    }
  }

  // ------ 4. Screen-size cull ------
  const minScreenSize = options && options.minScreenSize !== undefined
    ? options.minScreenSize
    : 0.0;
  if (minScreenSize > 0) {
    if (VisibilityEntityState.screenSize[eid] < minScreenSize) {
      VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.SMALL_SCREEN;
      CullingStats.histogram[CULL_REASON.SMALL_SCREEN]++;
      return VisibilityEntityState.flags[eid];
    }
  }

  // Passed all tests.
  VisibilityEntityState.flags[eid] |= VIS_FLAG.VISIBLE;
  VisibilityEntityState.flags[eid] &= ~(VIS_FLAG.OCCLUDED | VIS_FLAG.TEMPORAL_HOLD | VIS_FLAG.HYSTERESIS_HOLD);
  VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
  VisibilityEntityState.lastVisibleFrame[eid] = VisibilityState.frame;
  VisibilityEntityState.visibleFrames[eid]++;
  VisibilityState.visibleCount++;

  return VisibilityEntityState.flags[eid];
}

/**
 * Runs the visibility pipeline for every entity that has a valid AABB
 * or sphere. Uses the current frustum and camera cache.
 *
 * Returns the number of entities evaluated.
 */
export function computeAllVisibility(options) {
  const t0 = _now();
  VisibilityState.visibleCount = 0;
  VisibilityState.occludedCount = 0;

  // Reset per-frame histogram.
  CullingStats.histogram.fill(0);
  CullingStats.frustumCulled[0] = 0;
  CullingStats.occlusionCulled[0] = 0;
  CullingStats.budgetCulled[0] = 0;
  CullingStats.temporalHolds[0] = 0;
  CullingStats.hysteresisHolds[0] = 0;
  CullingStats.groupCulled[0] = 0;

  const budgetCap = VisibilityBudget.budgetCap[0] || VisibilityState.budgetCap;
  let budgetUsed = 0;
  let budgetExceeded = 0;

  const adapter = getAdapter();
  let evaluated = 0;

  // Pass 1: groups (anchors first).
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.GROUP_ANCHOR) === 0) continue;
    if (!adapter.entityAlive(eid)) continue;
    evaluateVisibilityGroup(eid);
  }

  // Pass 2: individual entities.
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (Sphere.valid[eid] === 0 && AABB.valid[eid] === 0) continue;

    // Skip group members — they were handled above.
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.GROUP_MEMBER) !== 0) continue;

    // Budget check.
    if (budgetCap > 0 && budgetUsed >= budgetCap) {
      VisibilityEntityState.flags[eid] &= ~VIS_FLAG.VISIBLE;
      VisibilityEntityState.flags[eid] |= VIS_FLAG.BUDGET_CULLED;
      VisibilityEntityState.cullReason[eid] = CULL_REASON.BUDGET;
      budgetExceeded = 1;
      CullingStats.budgetCulled[0]++;
      continue;
    }

    // Snapshot the previous flags.
    VisibilityEntityState.prevFlags[eid] = VisibilityEntityState.flags[eid];

    computeVisibility(eid, options);
    evaluated++;

    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) !== 0) {
      budgetUsed++;
    }
  }

  VisibilityBudget.totalCost[0] = budgetUsed;
  VisibilityBudget.budgetUsed[0] = budgetUsed;
  VisibilityBudget.budgetExceeded[0] = budgetExceeded;
  VisibilityState.budgetUsed = budgetUsed;
  VisibilityState.budgetExceeded = budgetExceeded;

  CullingStats.visible[0] = VisibilityState.visibleCount;
  CullingStats.hidden[0] = evaluated - VisibilityState.visibleCount;
  CullingStats.temporalHolds[0] = VisibilityState.totalTemporalHolds;
  CullingStats.hysteresisHolds[0] = VisibilityState.totalHysteresisHolds;

  if (VisibilityState.visibleCount > VisibilityState.peakVisibleCount) {
    VisibilityState.peakVisibleCount = VisibilityState.visibleCount;
  }

  const t1 = _now();
  const cost = t1 - t0;
  VisibilityState.lastEvalMs = cost;
  VisibilityState.avgEvalMs += (cost - VisibilityState.avgEvalMs) * 0.15;

  return evaluated;
}

/* ------------------------------------------------------------------ */
/* 11. MANUAL VISIBILITY CONTROL                                      */
/* ------------------------------------------------------------------ */

/**
 * Forces an entity to be visible regardless of frustum / occlusion.
 */
export function forceVisible(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  VisibilityEntityState.flags[eid] |= VIS_FLAG.FORCED_VISIBLE | VIS_FLAG.VISIBLE;
  VisibilityEntityState.flags[eid] &= ~VIS_FLAG.FORCED_HIDDEN;
  VisibilityEntityState.cullReason[eid] = CULL_REASON.VISIBLE;
  return true;
}

/**
 * Forces an entity to be hidden.
 */
export function forceHidden(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  VisibilityEntityState.flags[eid] |= VIS_FLAG.FORCED_HIDDEN;
  VisibilityEntityState.flags[eid] &= ~(VIS_FLAG.FORCED_VISIBLE | VIS_FLAG.VISIBLE);
  VisibilityEntityState.cullReason[eid] = CULL_REASON.FORCED_HIDDEN;
  return true;
}

/**
 * Clears the forced visibility override.
 */
export function clearForcedVisibility(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  VisibilityEntityState.flags[eid] &= ~(VIS_FLAG.FORCED_VISIBLE | VIS_FLAG.FORCED_HIDDEN);
  return true;
}

/**
 * Skips the culling pipeline for an entity (always visible).
 */
export function skipCulling(eid, skip) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (skip) VisibilityEntityState.flags[eid] |= VIS_FLAG.SKIP_CULLING;
  else VisibilityEntityState.flags[eid] &= ~VIS_FLAG.SKIP_CULLING;
  return true;
}

/**
 * Returns true if the entity is currently marked visible.
 */
export function isEntityVisible(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return (VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) !== 0;
}

/**
 * Returns the entity's last cull reason.
 */
export function getCullReason(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return CULL_REASON.NONE;
  return VisibilityEntityState.cullReason[eid];
}

/**
 * Returns the entity's last frustum result.
 */
export function getFrustumResult(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return FRUSTUM_RESULT.OUTSIDE;
  return VisibilityEntityState.frustumResult[eid];
}

/* ------------------------------------------------------------------ */
/* 12. BUDGET MANAGEMENT                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets the visibility budget cap.
 */
export function setVisibilityBudgetCap(cap) {
  VisibilityBudget.budgetCap[0] = Math.max(0, Number(cap) || 0);
  VisibilityState.budgetCap = VisibilityBudget.budgetCap[0];
  return true;
}

export function getVisibilityBudgetCap() {
  return VisibilityBudget.budgetCap[0];
}

/**
 * Updates the aggregate visibility budget from the current state.
 */
export function updateVisibilityBudget() {
  let total = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) === 0) continue;
    total += VisibilityBudget.entityCostEma[eid];
  }

  VisibilityBudget.totalCost[0] = total;
  VisibilityState.budgetUsed = total;

  const cap = VisibilityBudget.budgetCap[0];
  if (cap > 0 && total > cap) {
    VisibilityBudget.budgetExceeded[0] = 1;
    VisibilityState.budgetExceeded = 1;
  } else {
    VisibilityBudget.budgetExceeded[0] = 0;
    VisibilityState.budgetExceeded = 0;
  }

  return total;
}

/**
 * Returns true if the visibility budget is exceeded.
 */
export function isVisibilityBudgetExceeded() {
  return VisibilityBudget.budgetExceeded[0] === 1;
}

/* ------------------------------------------------------------------ */
/* 13. QUERY HELPERS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity currently marked visible. Allocation-free.
 */
export function forEachVisibleEntity(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) === 0) continue;
    fn.call(ctx, eid);
    n++;
  }
  return n;
}

/**
 * Iterates every entity currently marked occluded. Allocation-free.
 */
export function forEachOccludedEntity(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.OCCLUDED) === 0) continue;
    fn.call(ctx, eid);
    n++;
  }
  return n;
}

/**
 * Collects every visible entity into a caller-provided array.
 */
export function collectVisibleEntities(outArray, outOffset) {
  if (!outArray) return 0;
  const offset = outOffset !== undefined ? outOffset : 0;
  const cap = outArray.length - offset;
  let write = 0;
  for (let eid = 0; eid < MAX_ENTITIES && write < cap; eid++) {
    if ((VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) === 0) continue;
    outArray[offset + write++] = eid;
  }
  return write;
}

/**
 * Returns the current visible entity count.
 */
export function getVisibleCount() {
  return VisibilityState.visibleCount;
}

/**
 * Returns the current occluded entity count.
 */
export function getOccludedCount() {
  return VisibilityState.occludedCount;
}

/* ------------------------------------------------------------------ */
/* 14. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the visibility system frame counter.
 */
export function tickVisibility(frameNumber) {
  if (typeof frameNumber === 'number') VisibilityState.frame = frameNumber;
  else VisibilityState.frame++;

  // Roll the "previous flags" forward for hysteresis.
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const cur = VisibilityEntityState.flags[eid];
    const prev = VisibilityEntityState.prevFlags[eid];
    VisibilityEntityState.prevFlags[eid] = cur;
    VisibilityEntityState.flags[eid] = cur & ~(
      VIS_FLAG.FRUSTUM_TESTED |
      VIS_FLAG.OCCLUSION_TESTED |
      VIS_FLAG.TEMPORAL_HOLD |
      VIS_FLAG.HYSTERESIS_HOLD |
      VIS_FLAG.BUDGET_CULLED
    );
  }
}

/**
 * Full per-frame visibility pipeline:
 *   1. tickVisibility(frame)
 *   2. updateCameraCache(camera)
 *   3. computeAllVisibility(options)
 *   4. updateVisibilityBudget()
 */
export function tickVisibilitySystem(frameNumber, cameraX, cameraY, cameraZ, options) {
  tickVisibility(frameNumber);
  updateCameraCache(cameraX, cameraY, cameraZ);
  const evaluated = computeAllVisibility(options);
  updateVisibilityBudget();
  return {
    frame: VisibilityState.frame,
    evaluated,
    visible: VisibilityState.visibleCount,
    occluded: VisibilityState.occludedCount,
    budgetExceeded: VisibilityState.budgetExceeded === 1,
  };
}

/* ------------------------------------------------------------------ */
/* 15. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerVisibilityComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'VisibilityEntityState',    component: VisibilityEntityState,    category: 17, subsystem: 1, dependencies: [] },
    { name: 'FrustumPlane',             component: FrustumPlane,             category: 17, subsystem: 1, dependencies: [] },
    { name: 'FrustumSphere',            component: FrustumSphere,            category: 17, subsystem: 1, dependencies: [] },
    { name: 'OcclusionQuery',           component: OcclusionQuery,           category: 17, subsystem: 1, dependencies: [] },
    { name: 'HiZPyramid',               component: HiZPyramid,               category: 17, subsystem: 1, dependencies: [] },
    { name: 'VisibilityGroup',          component: VisibilityGroup,          category: 17, subsystem: 1, dependencies: [] },
    { name: 'VisibilityGroupMember',    component: VisibilityGroupMember,    category: 17, subsystem: 1, dependencies: ['VisibilityGroup'] },
    { name: 'VisibilityBudget',         component: VisibilityBudget,         category: 17, subsystem: 1, dependencies: [] },
    { name: 'CullingStats',             component: CullingStats,             category: 17, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 16. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getVisibilityEntityStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  return {
    entity:             eid,
    flags:              VisibilityEntityState.flags[eid],
    prevFlags:          VisibilityEntityState.prevFlags[eid],
    frustumResult:      VisibilityEntityState.frustumResult[eid],
    occlusionState:     OCCLUSION_STATE_NAME[VisibilityEntityState.occlusionState[eid]] || 'none',
    cullReason:         CULL_REASON_NAME[VisibilityEntityState.cullReason[eid]] || 'none',
    visible:            (VisibilityEntityState.flags[eid] & VIS_FLAG.VISIBLE) !== 0,
    occluded:           (VisibilityEntityState.flags[eid] & VIS_FLAG.OCCLUDED) !== 0,
    hysteresisHold:     (VisibilityEntityState.flags[eid] & VIS_FLAG.HYSTERESIS_HOLD) !== 0,
    temporalHold:       (VisibilityEntityState.flags[eid] & VIS_FLAG.TEMPORAL_HOLD) !== 0,
    visibleFrames:      VisibilityEntityState.visibleFrames[eid],
    hiddenFrames:       VisibilityEntityState.hiddenFrames[eid],
    lastVisibleFrame:   VisibilityEntityState.lastVisibleFrame[eid],
    lastHiddenFrame:    VisibilityState.lastHiddenFrame[eid],
    screenSize:         VisibilityEntityState.screenSize[eid],
    distance:           VisibilityEntityState.distanceToCam[eid],
    occlusionRatio:     OcclusionQuery.visibilityRatio[eid],
    isGroupAnchor:      (VisibilityEntityState.flags[eid] & VIS_FLAG.GROUP_ANCHOR) !== 0,
    isGroupMember:      (VisibilityEntityState.flags[eid] & VIS_FLAG.GROUP_MEMBER) !== 0,
    groupEid:           VisibilityGroupMember.groupEid[eid],
    hizLevels:          HiZPyramid.levelCount[eid],
  };
}

export function getVisibilitySystemReport() {
  const histogram = [];
  for (let i = 0; i < CULL_REASON_NAME.length; i++) {
    histogram.push({ reason: CULL_REASON_NAME[i], count: CullingStats.histogram[i] });
  }

  return {
    frame:                    VisibilityState.frame,
    budgetCap:                VisibilityBudget.budgetCap[0],
    budgetUsed:               VisibilityState.budgetUsed,
    budgetExceeded:           VisibilityState.budgetExceeded === 1,
    totalEvaluations:         VisibilityState.totalEvaluations,
    totalFrustumTests:        VisibilityState.totalFrustumTests,
    totalOcclusionTests:      VisibilityState.totalOcclusionTests,
    totalTemporalHolds:       VisibilityState.totalTemporalHolds,
    totalHysteresisHolds:     VisibilityState.totalHysteresisHolds,
    totalBudgetCulls:         VisibilityState.totalBudgetCulls,
    totalGroupEvaluations:    VisibilityState.totalGroupEvaluations,
    visibleCount:             VisibilityState.visibleCount,
    peakVisibleCount:         VisibilityState.peakVisibleCount,
    occludedCount:            VisibilityState.occludedCount,
    lastEvalMs:               VisibilityState.lastEvalMs,
    avgEvalMs:                VisibilityState.avgEvalMs,
    lastFrustumMs:            VisibilityState.lastFrustumMs,
    avgFrustumMs:             VisibilityState.avgFrustumMs,
    cameraStationary:         VisibilityState.cameraStationary === 1,
    frustumValid:             FrustumPlane.valid[0] === 1,
    frustumGeneration:        FrustumPlane.generation[0],
    histogram,
    perfTier:                 PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 17. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every visibility structure and resets counters.
 */
export function resetVisibilityState() {
  VisibilityEntityState.flags.fill(0);
  VisibilityEntityState.prevFlags.fill(0);
  VisibilityEntityState.frustumResult.fill(0);
  VisibilityEntityState.occlusionState.fill(0);
  VisibilityEntityState.cullReason.fill(0);
  VisibilityEntityState.visibleFrames.fill(0);
  VisibilityEntityState.hiddenFrames.fill(0);
  VisibilityEntityState.hysteresisCounter.fill(0);
  VisibilityEntityState.lastFrustumFrame.fill(0);
  VisibilityEntityState.lastOcclusionFrame.fill(0);
  VisibilityEntityState.lastVisibleFrame.fill(0);
  VisibilityEntityState.lastHiddenFrame.fill(0);
  VisibilityEntityState.screenSize.fill(0);
  VisibilityEntityState.distanceToCam.fill(0);
  VisibilityEntityState.distanceSq.fill(0);
  VisibilityEntityState.cameraDelta.fill(0);

  FrustumPlane.planes.fill(0);
  FrustumPlane.cameraX[0] = 0;
  FrustumPlane.cameraY[0] = 0;
  FrustumPlane.cameraZ[0] = 0;
  FrustumPlane.cameraDirX[0] = 0;
  FrustumPlane.cameraDirY[0] = 0;
  FrustumPlane.cameraDirZ[0] = 0;
  FrustumPlane.nearPlane[0] = 0;
  FrustumPlane.farPlane[0] = 0;
  FrustumPlane.aspect[0] = 0;
  FrustumPlane.tanFovHalf[0] = 0;
  FrustumPlane.valid[0] = 0;
  FrustumPlane.generation[0] = 0;

  FrustumSphere.centerX[0] = 0;
  FrustumSphere.centerY[0] = 0;
  FrustumSphere.centerZ[0] = 0;
  FrustumSphere.radius[0] = 0;
  FrustumSphere.radiusSq[0] = 0;
  FrustumSphere.valid[0] = 0;

  OcclusionQuery.queryId.fill(0);
  OcclusionQuery.queryGeneration.fill(0);
  OcclusionQuery.startFrame.fill(0);
  OcclusionQuery.resolveFrame.fill(0);
  OcclusionQuery.sampleCount.fill(0);
  OcclusionQuery.passedSamples.fill(0);
  OcclusionQuery.visibilityRatio.fill(0);
  OcclusionQuery.state.fill(0);
  OcclusionQuery.inFlight.fill(0);
  OcclusionQuery.timeout.fill(0);
  OcclusionQuery.enabled.fill(0);

  HiZPyramid.levelWidth.fill(0);
  HiZPyramid.levelHeight.fill(0);
  HiZPyramid.minDepth.fill(0);
  HiZPyramid.maxDepth.fill(0);
  HiZPyramid.levelCount.fill(0);
  HiZPyramid.baseWidth.fill(0);
  HiZPyramid.baseHeight.fill(0);
  HiZPyramid.generation.fill(0);
  HiZPyramid.valid.fill(0);

  VisibilityGroup.anchorEid.fill(-1);
  VisibilityGroup.memberCount.fill(0);
  VisibilityGroup.groupRadius.fill(0);
  VisibilityGroup.groupCenterX.fill(0);
  VisibilityGroup.groupCenterY.fill(0);
  VisibilityGroup.groupCenterZ.fill(0);
  VisibilityGroup.groupFlags.fill(0);
  VisibilityGroup.groupDirty.fill(0);
  VisibilityGroup.groupVisible.fill(0);

  VisibilityGroupMember.groupEid.fill(-1);
  VisibilityGroupMember.memberIndex.fill(0);
  VisibilityGroupMember.localWeight.fill(1.0);

  VisibilityBudget.entityCost.fill(0);
  VisibilityBudget.entityCostEma.fill(0);
  VisibilityBudget.totalCost[0] = 0;
  VisibilityBudget.budgetCap[0] = DEFAULT_VISIBILITY_BUDGET;
  VisibilityBudget.budgetExceeded[0] = 0;
  VisibilityBudget.budgetUsed[0] = 0;
  VisibilityBudget.budgetCulls[0] = 0;

  CullingStats.visible[0] = 0;
  CullingStats.hidden[0] = 0;
  CullingStats.frustumCulled[0] = 0;
  CullingStats.occlusionCulled[0] = 0;
  CullingStats.budgetCulled[0] = 0;
  CullingStats.temporalHolds[0] = 0;
  CullingStats.hysteresisHolds[0] = 0;
  CullingStats.groupCulled[0] = 0;
  CullingStats.totalEvaluated[0] = 0;
  CullingStats.histogram.fill(0);

  CullingScratch.candidateIds.fill(0);
  CullingScratch.candidateCount[0] = 0;
  CullingScratch.frustumPassed.fill(0);
  CullingScratch.frustumPassedCount[0] = 0;
  CullingScratch.occlusionPassed.fill(0);
  CullingScratch.occlusionPassedCount[0] = 0;
  CullingScratch.sortKeys.fill(0);

  VisibilityState.frame = 0;
  VisibilityState.budgetCap = DEFAULT_VISIBILITY_BUDGET;
  VisibilityState.budgetUsed = 0;
  VisibilityState.budgetExceeded = 0;
  VisibilityState.totalEvaluations = 0;
  VisibilityState.totalFrustumTests = 0;
  VisibilityState.totalOcclusionTests = 0;
  VisibilityState.totalTemporalHolds = 0;
  VisibilityState.totalHysteresisHolds = 0;
  VisibilityState.totalBudgetCulls = 0;
  VisibilityState.totalGroupEvaluations = 0;
  VisibilityState.visibleCount = 0;
  VisibilityState.peakVisibleCount = 0;
  VisibilityState.occludedCount = 0;
  VisibilityState.lastEvalMs = 0;
  VisibilityState.avgEvalMs = 0;
  VisibilityState.lastFrustumMs = 0;
  VisibilityState.avgFrustumMs = 0;
  VisibilityState.lastOcclusionMs = 0;
  VisibilityState.avgOcclusionMs = 0;
  VisibilityState.lastCameraX = 0;
  VisibilityState.lastCameraY = 0;
  VisibilityState.lastCameraZ = 0;
  VisibilityState.cameraStationary = 0;
}

/* ------------------------------------------------------------------ */
/* 18. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_HIZ_LEVELS,
  MAX_VISIBILITY_GROUP_MEMBERS,
  FRUSTUM_HYSTERESIS_FRAMES,
  TEMPORAL_COHERENCE_FRAMES,
  TEMPORAL_COHERENCE_DISTANCE,
  MAX_OCCLUSION_QUERIES_IN_FLIGHT,
  DEFAULT_VISIBILITY_BUDGET,

  // Enums
  VIS_FLAG,
  OCCLUSION_STATE,
  OCCLUSION_STATE_NAME,
  CULL_REASON,
  CULL_REASON_NAME,
  FRUSTUM_PLANE,
  FRUSTUM_RESULT,

  // Components
  VisibilityEntityState,
  FrustumPlane,
  FrustumSphere,
  OcclusionQuery,
  HiZPyramid,
  VisibilityGroup,
  VisibilityGroupMember,
  VisibilityBudget,
  CullingStats,
  CullingScratch,
  VISIBILITY_COMPONENTS,

  // Module state
  VisibilityState,

  // Frustum
  setFrustumPlane,
  setFrustumCamera,
  buildPerspectiveFrustum,

  // Plane tests
  testSpherePlane,
  testAABBPlane,
  testSphereFrustum,
  testAABBFrustum,
  testPointFrustum,

  // Occlusion
  enableOcclusionQuery,
  disableOcclusionQuery,
  beginOcclusionQuery,
  endOcclusionQuery,
  resolveOcclusionQuery,
  timeoutOcclusionQuery,
  isOccluded,
  getOcclusionVisibility,

  // Hi-Z
  initHiZPyramid,
  pushHiZLevel,
  popHiZLevel,
  hasHiZPyramid,
  getHiZLevelCount,

  // Groups
  registerVisibilityGroupAnchor,
  attachVisibilityGroupMember,
  detachVisibilityGroupMember,
  evaluateVisibilityGroup,

  // Temporal
  isCameraStationary,
  updateCameraCache,

  // Pipeline
  computeVisibility,
  computeAllVisibility,

  // Manual control
  forceVisible,
  forceHidden,
  clearForcedVisibility,
  skipCulling,
  isEntityVisible,
  getCullReason,
  getFrustumResult,

  // Budget
  setVisibilityBudgetCap,
  getVisibilityBudgetCap,
  updateVisibilityBudget,
  isVisibilityBudgetExceeded,

  // Query
  forEachVisibleEntity,
  forEachOccludedEntity,
  collectVisibleEntities,
  getVisibleCount,
  getOccludedCount,

  // Frame
  tickVisibility,
  tickVisibilitySystem,

  // Diagnostics
  getVisibilityEntityStats,
  getVisibilitySystemReport,

  // Registration
  registerVisibilityComponents,

  // Reset
  resetVisibilityState,
};

export default _defaultExport;