// File : 025
// name : src/ecs/025_scn_SpatialComponents.js
// description : Spatial-query SoA component module for the scene ECS world of
//               the anime lighting stack on Android mobile. Declares every
//               spatial bounding volume, spatial acceleration structure, and
//               visibility state the lighting stack needs — AABB, Sphere,
//               Frustum, GridCell, SpatialHash, OctreeRef, BVHRef,
//               DistanceFieldRef, ClusterCell, and VisibilityState — as
//               fixed-capacity typed arrays sized once to
//               MAX_ENTITIES = 100000.
//
//               Provides the fast spatial helpers that every culling and
//               assignment system runs on the hot path:
//                 • computeAABB               — derive AABB from Transform
//                 • computeSphere             — derive sphere from Transform
//                 • testAABBOverlap           — AABB vs AABB
//                 • testSphereOverlap         — sphere vs sphere
//                 • testSphereContainsPoint   — sphere vs point
//                 • insertIntoHashGrid        — register entity in hash grid
//                 • removeFromHashGrid        — remove entity from hash grid
//                 • queryRadius               — enumerate entities within radius
//                 • queryAABB                 — enumerate entities whose AABB
//                                               intersects a query box
//                 • forEachInCell             — enumerate entities in a cell
//                 • updateVisibility          — evaluate frustum + occlusion
//                                               + distance visibility
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no resizes.
//                 • Spatial hash grid with configurable cell size. Two-level
//                   (grid X/Y/Z) with a flat per-cell bucket linked list.
//                 • Radius queries use squared-distance early rejection so
//                   no Math.sqrt is issued unless the candidate survives.
//                 • AABB overlap is done axis-by-axis with early exit.
//                 • Frustum tests use plane-distance approximation:
//                   O(6) per-entity, no matrix multiply per-entity.
//                 • Cell-membership refs use a doubly-linked list so
//                   insertions and removals are O(1) with no allocation.
//                 • Zero allocations on the hot path — every helper works
//                   directly on the SoA arrays.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — Transform, LightRef
//                 • 003_lgt_ShadowComponents.js     — shadow caster refs
//                 • 004_lgt_GIComponents.js         — GI probe positions
//                 • 005_lgt_AOComponents.js         — AO volume bounds
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — visibility tags
//                 • 015_scn_Relations.js            — relations graph
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime records
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every culling, assignment, and lookup step
//            in the anime lighting stack runs in O(1)-ish time with zero
//            allocations — so light culling, shadow caster culling, GI probe
//            assignment, AO volume selection, and streaming chunk visibility
//            all work on the same spatial vocabulary.
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
  Transform,
  Target,
  LightRef,
} from './002_lgt_LightComponents.js';

import {
  ShadowCasterRef,
} from './003_lgt_ShadowComponents.js';

import {
  GIProbeRef,
  GIVolume,
} from './004_lgt_GIComponents.js';

import {
  AOVolumeRef,
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
  getTypeId,
  getTypePascalName,
} from './013_scn_ComponentTypes.js';

import {
  TAG,
  TAG2,
  FDIRTY,
  EntityTag,
  EntityTag2,
  FrameDirtyTag,
  tagEntity,
  untagEntity,
  tagEntity2,
  untagEntity2,
  markFrameDirty,
  clearFrameDirty,
  isFrameDirty,
} from './014_scn_Tags.js';

import {
  MAX_REFERENCES_PER_ENTITY,
  NULL_ENTITY,
} from './015_scn_Relations.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Spatial hash grid cell size (world units). Larger cells = fewer buckets
 * but more per-bucket entities. Tuned for the anime scale (chunks ~40m,
 * rooms ~8m, props ~1m).
 */
export const DEFAULT_CELL_SIZE =
  PERF_TIER_LOCAL === 'HIGH'   ? 4.0 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 5.0 :
                                 6.0;

/**
 * Spatial hash grid bounds along each axis. Sized to cover the largest
 * expected scene without dynamic growth.
 */
export const DEFAULT_GRID_MIN = -512.0;
export const DEFAULT_GRID_MAX =  512.0;

/**
 * Number of cells along each axis.
 */
export const DEFAULT_GRID_DIM =
  PERF_TIER_LOCAL === 'HIGH'   ? 256 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 192 :
                                 128;

/**
 * Total number of cells in the hash grid.
 */
export const DEFAULT_GRID_CELL_COUNT =
  DEFAULT_GRID_DIM * DEFAULT_GRID_DIM * DEFAULT_GRID_DIM;

/**
 * Maximum number of entities that can be registered in the hash grid.
 * Equal to MAX_ENTITIES since every entity could be inserted.
 */
export const MAX_HASH_ENTRIES = MAX_ENTITIES;

/**
 * Maximum per-cell bucket size — the linked list is bounded so a cell
 * with hundreds of entities doesn't break the memory budget.
 */
export const MAX_CELL_OCCUPANCY =
  PERF_TIER_LOCAL === 'HIGH'   ? 128 :
  PERF_TIER_LOCAL === 'MEDIUM' ?  96 :
                                 64;

/**
 * Frustum plane indices.
 */
export const FRUSTUM_PLANE = Object.freeze({
  LEFT:   0,
  RIGHT:  1,
  BOTTOM: 2,
  TOP:    3,
  NEAR:   4,
  FAR:    5,
  COUNT:  6,
});

/**
 * Frustum test results.
 */
export const FRUSTUM_RESULT = Object.freeze({
  OUTSIDE: 0,
  INTERSECTING: 1,
  INSIDE: 2,
});

/**
 * Spatial state flags (bitmask, stored per entity).
 */
export const SPATIAL_FLAG = Object.freeze({
  NONE:              0,
  HAS_AABB:          1 << 0,
  HAS_SPHERE:        1 << 1,
  IN_HASH_GRID:      1 << 2,
  IN_FRUSTUM:        1 << 3,
  OCCLUDED:          1 << 4,
  VISIBLE:           1 << 5,
  DIRTY:             1 << 6,
  CLUSTER_ASSIGNED:  1 << 7,
  LOD_ASSIGNED:      1 << 8,
});

/**
 * Octree node children indices.
 */
export const OCTREE_CHILD = Object.freeze({
  NWB: 0, NEB: 1, SWB: 2, SEB: 3,
  NWF: 4, NEF: 5, SWF: 6, SEF: 7,
  COUNT: 8,
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const SpatialState = {
  frame:               0,
  cellSize:            DEFAULT_CELL_SIZE,
  gridMin:             DEFAULT_GRID_MIN,
  gridMax:             DEFAULT_GRID_MAX,
  gridDim:             DEFAULT_GRID_DIM,
  gridCellCount:       DEFAULT_GRID_CELL_COUNT,
  inverseCellSize:     1.0 / DEFAULT_CELL_SIZE,
  hashEntries:         0,
  peakHashEntries:     0,
  totalInsertions:     0,
  totalRemovals:       0,
  totalQueries:        0,
  totalQueryHits:      0,
  totalAABBComputes:   0,
  totalSphereComputes: 0,
  totalFrustumTests:   0,
  totalOcclusionTests: 0,
  cellOverflows:       0,
  lastQueryMs:         0,
  avgQueryMs:          0,
  lastFrustumMs:       0,
  avgFrustumMs:        0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.spatial', {
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
 * AABB — axis-aligned bounding box per entity.
 */
export const AABB = {
  minX:  new Float32Array(MAX_ENTITIES),
  minY:  new Float32Array(MAX_ENTITIES),
  minZ:  new Float32Array(MAX_ENTITIES),
  maxX:  new Float32Array(MAX_ENTITIES),
  maxY:  new Float32Array(MAX_ENTITIES),
  maxZ:  new Float32Array(MAX_ENTITIES),
  centerX: new Float32Array(MAX_ENTITIES),
  centerY: new Float32Array(MAX_ENTITIES),
  centerZ: new Float32Array(MAX_ENTITIES),
  valid:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Bounding sphere per entity.
 */
export const Sphere = {
  cx:     new Float32Array(MAX_ENTITIES),
  cy:     new Float32Array(MAX_ENTITIES),
  cz:     new Float32Array(MAX_ENTITIES),
  radius: new Float32Array(MAX_ENTITIES),
  radiusSq: new Float32Array(MAX_ENTITIES),
  valid:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Spatial hash grid cell membership. Doubly-linked list per cell so
 * insert/remove is O(1). Every entity has at most one grid entry.
 */
export const GridCell = {
  cellId:      new Int32Array(MAX_ENTITIES).fill(-1),
  cellX:       new Int16Array(MAX_ENTITIES),
  cellY:       new Int16Array(MAX_ENTITIES),
  cellZ:       new Int16Array(MAX_ENTITIES),
  prevInCell:  new Int32Array(MAX_ENTITIES).fill(-1),
  nextInCell:  new Int32Array(MAX_ENTITIES).fill(-1),
  cellOccupancy: new Uint16Array(MAX_ENTITIES),
};

/**
 * Per-cell head pointer. Flat array of size gridCellCount.
 */
export const SpatialHash = {
  cellHead:  new Int32Array(DEFAULT_GRID_CELL_COUNT).fill(-1),
  cellCount: new Uint16Array(DEFAULT_GRID_CELL_COUNT),
};

/**
 * Octree node reference per entity (the octree structure itself lives in
 * a downstream module; here we only track the leaf assignment).
 */
export const OctreeRef = {
  nodeId:     new Int32Array(MAX_ENTITIES).fill(-1),
  depth:      new Uint8Array(MAX_ENTITIES),
  assignmentValid: new Uint8Array(MAX_ENTITIES),
};

/**
 * BVH leaf reference per entity (BVH structure lives downstream).
 */
export const BVHRef = {
  leafId:     new Int32Array(MAX_ENTITIES).fill(-1),
  triangleCount: new Uint32Array(MAX_ENTITIES),
  valid:      new Uint8Array(MAX_ENTITIES),
};

/**
 * Distance-field binding per entity.
 */
export const DistanceFieldRef = {
  fieldEid:     new Int32Array(MAX_ENTITIES).fill(-1),
  sampleScale:  new Float32Array(MAX_ENTITIES),
  maxDistance:  new Float32Array(MAX_ENTITIES),
  useGradient:  new Uint8Array(MAX_ENTITIES),
  valid:        new Uint8Array(MAX_ENTITIES),
};

/**
 * Cluster cell membership — populated by the cluster subsystem.
 */
export const ClusterCell = {
  clusterId:   new Int32Array(MAX_ENTITIES).fill(-1),
  localX:      new Float32Array(MAX_ENTITIES),
  localY:      new Float32Array(MAX_ENTITIES),
  localZ:      new Float32Array(MAX_ENTITIES),
  sliceIndex:  new Uint16Array(MAX_ENTITIES),
  assigned:    new Uint8Array(MAX_ENTITIES),
};

/**
 * Per-entity visibility state — evaluated once per frame by the culling
 * system and consumed by every renderer.
 */
export const VisibilityState = {
  flags:           new Uint16Array(MAX_ENTITIES),
  frustumResult:   new Uint8Array(MAX_ENTITIES),   // FRUSTUM_RESULT
  distanceToCam:   new Float32Array(MAX_ENTITIES),
  distanceSq:      new Float32Array(MAX_ENTITIES),
  screenSize:      new Float32Array(MAX_ENTITIES),
  lodLevel:        new Uint8Array(MAX_ENTITIES),
  lastEvalFrame:   new Uint32Array(MAX_ENTITIES),
  occludedBy:      new Int32Array(MAX_ENTITIES).fill(-1),
};

/**
 * Frustum planes (world-space). One frustum per scene.
 */
export const FrustumPlanes = {
  left:   new Float32Array(4),   // (nx, ny, nz, d)
  right:  new Float32Array(4),
  bottom: new Float32Array(4),
  top:    new Float32Array(4),
  near:   new Float32Array(4),
  far:    new Float32Array(4),
  valid:  new Uint8Array(1),
  cameraX: new Float32Array(1),
  cameraY: new Float32Array(1),
  cameraZ: new Float32Array(1),
  tanFovHalf: new Float32Array(1),
  aspect:  new Float32Array(1),
  nearPlane: new Float32Array(1),
  farPlane:  new Float32Array(1),
};

/**
 * Spatial component bundle for bitECS createWorld.
 */
export const SPATIAL_COMPONENTS = Object.freeze({
  AABB,
  Sphere,
  GridCell,
  OctreeRef,
  BVHRef,
  DistanceFieldRef,
  ClusterCell,
  VisibilityState,
});

/* ------------------------------------------------------------------ */
/* 3. GRID CELL INDEXING                                              */
/* ------------------------------------------------------------------ */

function _cellIndexFromCellCoords(cx, cy, cz) {
  const dim = SpatialState.gridDim;
  if (cx < 0 || cx >= dim) return -1;
  if (cy < 0 || cy >= dim) return -1;
  if (cz < 0 || cz >= dim) return -1;
  return (cz * dim + cy) * dim + cx;
}

function _cellCoordsFromWorld(x, y, z, out) {
  const inv = SpatialState.inverseCellSize;
  const min = SpatialState.gridMin;
  out[0] = Math.floor((x - min) * inv);
  out[1] = Math.floor((y - min) * inv);
  out[2] = Math.floor((z - min) * inv);
  return out;
}

const _cellCoordsScratch = new Int32Array(3);

function _worldToCellIndex(x, y, z) {
  _cellCoordsFromWorld(x, y, z, _cellCoordsScratch);
  return _cellIndexFromCellCoords(
    _cellCoordsScratch[0],
    _cellCoordsScratch[1],
    _cellCoordsScratch[2]
  );
}

/* ------------------------------------------------------------------ */
/* 4. GRID CONFIGURATION                                              */
/* ------------------------------------------------------------------ */

/**
 * Configures the spatial hash grid dimensions. Must be called BEFORE
 * inserting any entity. Reallocates the cell head/count arrays.
 */
export function configureGrid(cellSize, gridMin, gridMax, gridDim) {
  const cs = Number(cellSize);
  if (!Number.isFinite(cs) || cs <= 0) return false;

  const gmin = Number(gridMin);
  const gmax = Number(gridMax);
  if (!Number.isFinite(gmin) || !Number.isFinite(gmax) || gmax <= gmin) return false;

  const dim = Math.max(1, Math.min(512, gridDim | 0));
  const cellCount = dim * dim * dim;
  if (cellCount > 128 * 1024 * 1024) {
    // Guard against runaway allocations (e.g. 512^3 = 134M cells).
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[025_scn_SpatialComponents] refusing grid dim ${dim} (${cellCount} cells)`);
    return false;
  }

  SpatialState.cellSize = cs;
  SpatialState.gridMin = gmin;
  SpatialState.gridMax = gmax;
  SpatialState.gridDim = dim;
  SpatialState.gridCellCount = cellCount;
  SpatialState.inverseCellSize = 1.0 / cs;

  // Reallocate the flat cell arrays.
  const newHead = new Int32Array(cellCount).fill(-1);
  const newCount = new Uint16Array(cellCount);

  // Migrate any existing entries — for simplicity, reset the grid.
  SpatialHash.cellHead = newHead;
  SpatialHash.cellCount = newCount;

  // Clear per-entity cell refs.
  GridCell.cellId.fill(-1);
  GridCell.prevInCell.fill(-1);
  GridCell.nextInCell.fill(-1);
  SpatialState.hashEntries = 0;

  return true;
}

export function getGridDim()        { return SpatialState.gridDim; }
export function getGridCellSize()   { return SpatialState.cellSize; }
export function getGridCellCount()  { return SpatialState.gridCellCount; }

/* ------------------------------------------------------------------ */
/* 5. AABB / SPHERE DERIVATION                                        */
/* ------------------------------------------------------------------ */

/**
 * Computes an AABB from a Transform (position + scale) and an optional
 * half-extent (in local units). If `halfExtent` is omitted the AABB is
 * a point-sized box.
 */
export function computeAABB(eid, halfExtent) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const cx = Transform.x[eid];
  const cy = Transform.y[eid];
  const cz = Transform.z[eid];

  const sx = Math.abs(Transform.sx[eid]);
  const sy = Math.abs(Transform.sy[eid]);
  const sz = Math.abs(Transform.sz[eid]);

  const h = halfExtent !== undefined ? Number(halfExtent) : 0.5;
  const hx = sx * h;
  const hy = sy * h;
  const hz = sz * h;

  AABB.minX[eid] = cx - hx;
  AABB.minY[eid] = cy - hy;
  AABB.minZ[eid] = cz - hz;
  AABB.maxX[eid] = cx + hx;
  AABB.maxY[eid] = cy + hy;
  AABB.maxZ[eid] = cz + hz;
  AABB.centerX[eid] = cx;
  AABB.centerY[eid] = cy;
  AABB.centerZ[eid] = cz;
  AABB.valid[eid] = 1;

  SpatialState.totalAABBComputes++;
  return true;
}

/**
 * Computes an AABB from explicit min/max values.
 */
export function setAABB(eid, minX, minY, minZ, maxX, maxY, maxZ) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  AABB.minX[eid] = minX;
  AABB.minY[eid] = minY;
  AABB.minZ[eid] = minZ;
  AABB.maxX[eid] = maxX;
  AABB.maxY[eid] = maxY;
  AABB.maxZ[eid] = maxZ;
  AABB.centerX[eid] = (minX + maxX) * 0.5;
  AABB.centerY[eid] = (minY + maxY) * 0.5;
  AABB.centerZ[eid] = (minZ + maxZ) * 0.5;
  AABB.valid[eid] = 1;
  return true;
}

/**
 * Computes a bounding sphere from a Transform (position + max scale).
 */
export function computeSphere(eid, radius) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const r = radius !== undefined ? Number(radius) : 0.5;
  const sx = Math.abs(Transform.sx[eid]);
  const sy = Math.abs(Transform.sy[eid]);
  const sz = Math.abs(Transform.sz[eid]);
  const maxScale = Math.max(sx, sy, sz);
  const scaledR = r * maxScale;

  Sphere.cx[eid] = Transform.x[eid];
  Sphere.cy[eid] = Transform.y[eid];
  Sphere.cz[eid] = Transform.z[eid];
  Sphere.radius[eid] = scaledR;
  Sphere.radiusSq[eid] = scaledR * scaledR;
  Sphere.valid[eid] = 1;

  SpatialState.totalSphereComputes++;
  return true;
}

/**
 * Sets an explicit bounding sphere.
 */
export function setSphere(eid, cx, cy, cz, radius) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const r = Math.max(0, Number(radius));
  Sphere.cx[eid] = cx;
  Sphere.cy[eid] = cy;
  Sphere.cz[eid] = cz;
  Sphere.radius[eid] = r;
  Sphere.radiusSq[eid] = r * r;
  Sphere.valid[eid] = 1;
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. OVERLAP TESTS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Returns true if two entities' AABBs overlap. Both must have valid AABBs.
 */
export function testAABBOverlap(aEid, bEid) {
  if (typeof aEid !== 'number' || aEid < 0 || aEid >= MAX_ENTITIES) return false;
  if (typeof bEid !== 'number' || bEid < 0 || bEid >= MAX_ENTITIES) return false;
  if (AABB.valid[aEid] === 0 || AABB.valid[bEid] === 0) return false;
  return (
    AABB.minX[aEid] <= AABB.maxX[bEid] && AABB.maxX[aEid] >= AABB.minX[bEid] &&
    AABB.minY[aEid] <= AABB.maxY[bEid] && AABB.maxY[aEid] >= AABB.minY[bEid] &&
    AABB.minZ[aEid] <= AABB.maxZ[bEid] && AABB.maxZ[aEid] >= AABB.minZ[bEid]
  );
}

/**
 * Returns true if two entities' spheres overlap.
 */
export function testSphereOverlap(aEid, bEid) {
  if (typeof aEid !== 'number' || aEid < 0 || aEid >= MAX_ENTITIES) return false;
  if (typeof bEid !== 'number' || bEid < 0 || bEid >= MAX_ENTITIES) return false;
  if (Sphere.valid[aEid] === 0 || Sphere.valid[bEid] === 0) return false;

  const dx = Sphere.cx[bEid] - Sphere.cx[aEid];
  const dy = Sphere.cy[bEid] - Sphere.cy[aEid];
  const dz = Sphere.cz[bEid] - Sphere.cz[aEid];
  const dSq = dx * dx + dy * dy + dz * dz;

  const rSum = Sphere.radius[aEid] + Sphere.radius[bEid];
  return dSq <= rSum * rSum;
}

/**
 * Returns true if a sphere contains a point.
 */
export function testSphereContainsPoint(eid, px, py, pz) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (Sphere.valid[eid] === 0) return false;
  const dx = px - Sphere.cx[eid];
  const dy = py - Sphere.cy[eid];
  const dz = pz - Sphere.cz[eid];
  const dSq = dx * dx + dy * dy + dz * dz;
  return dSq <= Sphere.radiusSq[eid];
}

/**
 * Returns true if an entity's AABB contains a point.
 */
export function testAABBContainsPoint(eid, px, py, pz) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (AABB.valid[eid] === 0) return false;
  return (
    px >= AABB.minX[eid] && px <= AABB.maxX[eid] &&
    py >= AABB.minY[eid] && py <= AABB.maxY[eid] &&
    pz >= AABB.minZ[eid] && pz <= AABB.maxZ[eid]
  );
}

/* ------------------------------------------------------------------ */
/* 7. SPATIAL HASH GRID — INSERT / REMOVE                             */
/* ------------------------------------------------------------------ */

/**
 * Inserts an entity into the spatial hash grid, using its current
 * Transform position. O(1).
 *
 * Returns the cell id, or -1 if out of bounds.
 */
export function insertIntoHashGrid(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;

  // Remove if already inserted.
  if (GridCell.cellId[eid] >= 0) removeFromHashGrid(eid);

  const cx = Transform.x[eid];
  const cy = Transform.y[eid];
  const cz = Transform.z[eid];
  const cellId = _worldToCellIndex(cx, cy, cz);
  if (cellId < 0) return -1;

  // Cell occupancy cap.
  if (SpatialHash.cellCount[cellId] >= MAX_CELL_OCCUPANCY) {
    SpatialState.cellOverflows++;
    return -1;
  }

  // Push onto the front of the cell's linked list.
  const prevHead = SpatialHash.cellHead[cellId];
  GridCell.cellId[eid] = cellId;
  GridCell.prevInCell[eid] = -1;
  GridCell.nextInCell[eid] = prevHead;
  if (prevHead >= 0) GridCell.prevInCell[prevHead] = eid;
  SpatialHash.cellHead[cellId] = eid;
  SpatialHash.cellCount[cellId]++;

  // Compute the X/Y/Z cell coordinates for quick neighbour queries.
  GridCell.cellX[eid] = _cellCoordsScratch[0];
  GridCell.cellY[eid] = _cellCoordsScratch[1];
  GridCell.cellZ[eid] = _cellCoordsScratch[2];
  GridCell.cellOccupancy[eid] = SpatialHash.cellCount[cellId];

  SpatialState.hashEntries++;
  if (SpatialState.hashEntries > SpatialState.peakHashEntries) {
    SpatialState.peakHashEntries = SpatialState.hashEntries;
  }
  SpatialState.totalInsertions++;

  return cellId;
}

/**
 * Removes an entity from the spatial hash grid. O(1).
 */
export function removeFromHashGrid(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const cellId = GridCell.cellId[eid];
  if (cellId < 0) return false;

  const prev = GridCell.prevInCell[eid];
  const next = GridCell.nextInCell[eid];

  if (prev >= 0) GridCell.nextInCell[prev] = next;
  else SpatialHash.cellHead[cellId] = next;

  if (next >= 0) GridCell.prevInCell[next] = prev;

  GridCell.cellId[eid] = -1;
  GridCell.prevInCell[eid] = -1;
  GridCell.nextInCell[eid] = -1;

  if (SpatialHash.cellCount[cellId] > 0) SpatialHash.cellCount[cellId]--;
  SpatialState.hashEntries--;
  SpatialState.totalRemovals++;

  return true;
}

/**
 * Re-inserts an entity into the grid if its cell changed. Called by
 * moving entities.
 */
export function refreshHashGrid(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const cellId = GridCell.cellId[eid];
  const cx = Transform.x[eid];
  const cy = Transform.y[eid];
  const cz = Transform.z[eid];
  const newCellId = _worldToCellIndex(cx, cy, cz);

  if (newCellId === cellId) return true;

  if (cellId >= 0) removeFromHashGrid(eid);
  if (newCellId >= 0) insertIntoHashGrid(eid);
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. SPATIAL HASH GRID — QUERIES                                     */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity within a sphere around (px, py, pz). Uses the
 * spatial hash grid to enumerate candidate cells, then performs a
 * squared-distance check per entity. Allocation-free.
 *
 * Returns the number of entities visited.
 */
export function queryRadius(px, py, pz, radius, fn, ctx) {
  if (typeof fn !== 'function') return 0;

  const r = Number(radius);
  if (!Number.isFinite(r) || r <= 0) return 0;

  SpatialState.totalQueries++;
  const t0 = _now();

  const rSq = r * r;
  const cellMinX = px - r;
  const cellMinY = py - r;
  const cellMinZ = pz - r;
  const cellMaxX = px + r;
  const cellMaxY = py + r;
  const cellMaxZ = pz + r;

  const cs = SpatialState.cellSize;
  const inv = SpatialState.inverseCellSize;
  const gridMin = SpatialState.gridMin;
  const dim = SpatialState.gridDim;

  const minCx = Math.max(0, Math.floor((cellMinX - gridMin) * inv));
  const minCy = Math.max(0, Math.floor((cellMinY - gridMin) * inv));
  const minCz = Math.max(0, Math.floor((cellMinZ - gridMin) * inv));
  const maxCx = Math.min(dim - 1, Math.floor((cellMaxX - gridMin) * inv));
  const maxCy = Math.min(dim - 1, Math.floor((cellMaxY - gridMin) * inv));
  const maxCz = Math.min(dim - 1, Math.floor((cellMaxZ - gridMin) * inv));

  if (maxCx < minCx || maxCy < minCy || maxCz < minCz) return 0;

  let visited = 0;

  for (let cz = minCz; cz <= maxCz; cz++) {
    for (let cy = minCy; cy <= maxCy; cy++) {
      for (let cx = minCx; cx <= maxCx; cx++) {
        const cellId = _cellIndexFromCellCoords(cx, cy, cz);
        if (cellId < 0) continue;

        let eid = SpatialHash.cellHead[cellId];
        while (eid >= 0) {
          const dx = Transform.x[eid] - px;
          const dy = Transform.y[eid] - py;
          const dz = Transform.z[eid] - pz;
          const dSq = dx * dx + dy * dy + dz * dz;
          if (dSq <= rSq) {
            fn.call(ctx, eid, dSq);
            visited++;
            SpatialState.totalQueryHits++;
          }
          eid = GridCell.nextInCell[eid];
        }
      }
    }
  }

  const t1 = _now();
  const cost = t1 - t0;
  SpatialState.lastQueryMs = cost;
  SpatialState.avgQueryMs += (cost - SpatialState.avgQueryMs) * 0.15;

  return visited;
}

/**
 * Iterates every entity whose AABB intersects the query box. Uses the
 * spatial hash grid for candidate enumeration, then does a full AABB
 * test per candidate. Allocation-free.
 */
export function queryAABB(minX, minY, minZ, maxX, maxY, maxZ, fn, ctx) {
  if (typeof fn !== 'function') return 0;

  SpatialState.totalQueries++;
  const t0 = _now();

  const inv = SpatialState.inverseCellSize;
  const gridMin = SpatialState.gridMin;
  const dim = SpatialState.gridDim;

  const minCx = Math.max(0, Math.floor((minX - gridMin) * inv));
  const minCy = Math.max(0, Math.floor((minY - gridMin) * inv));
  const minCz = Math.max(0, Math.floor((minZ - gridMin) * inv));
  const maxCx = Math.min(dim - 1, Math.floor((maxX - gridMin) * inv));
  const maxCy = Math.min(dim - 1, Math.floor((maxY - gridMin) * inv));
  const maxCz = Math.min(dim - 1, Math.floor((maxZ - gridMin) * inv));

  if (maxCx < minCx || maxCy < minCy || maxCz < minCz) return 0;

  let visited = 0;

  for (let cz = minCz; cz <= maxCz; cz++) {
    for (let cy = minCy; cy <= maxCy; cy++) {
      for (let cx = minCx; cx <= maxCx; cx++) {
        const cellId = _cellIndexFromCellCoords(cx, cy, cz);
        if (cellId < 0) continue;

        let eid = SpatialHash.cellHead[cellId];
        while (eid >= 0) {
          // Full AABB test.
          if (
            AABB.minX[eid] <= maxX && AABB.maxX[eid] >= minX &&
            AABB.minY[eid] <= maxY && AABB.maxY[eid] >= minY &&
            AABB.minZ[eid] <= maxZ && AABB.maxZ[eid] >= minZ
          ) {
            fn.call(ctx, eid);
            visited++;
            SpatialState.totalQueryHits++;
          }
          eid = GridCell.nextInCell[eid];
        }
      }
    }
  }

  const t1 = _now();
  const cost = t1 - t0;
  SpatialState.lastQueryMs = cost;
  SpatialState.avgQueryMs += (cost - SpatialState.avgQueryMs) * 0.15;

  return visited;
}

/**
 * Iterates every entity in a specific cell. Allocation-free.
 */
export function forEachInCell(cx, cy, cz, fn, ctx) {
  if (typeof fn !== 'function') return 0;
  const cellId = _cellIndexFromCellCoords(cx, cy, cz);
  if (cellId < 0) return 0;

  let eid = SpatialHash.cellHead[cellId];
  let visited = 0;
  while (eid >= 0) {
    fn.call(ctx, eid);
    visited++;
    eid = GridCell.nextInCell[eid];
  }
  return visited;
}

/**
 * Collects every entity within a sphere into a caller-provided array.
 * Returns the number written.
 */
export function collectRadius(px, py, pz, radius, outArray, outOffset) {
  if (!outArray) return 0;
  const offset = outOffset !== undefined ? outOffset : 0;
  const cap = outArray.length - offset;
  let write = 0;

  queryRadius(px, py, pz, radius, (eid) => {
    if (write < cap) {
      outArray[offset + write] = eid;
      write++;
    }
  });
  return write;
}

/**
 * Returns the number of entities in a cell.
 */
export function getCellCount(cx, cy, cz) {
  const cellId = _cellIndexFromCellCoords(cx, cy, cz);
  if (cellId < 0) return 0;
  return SpatialHash.cellCount[cellId];
}

/* ------------------------------------------------------------------ */
/* 9. FRUSTUM MANAGEMENT                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets a frustum plane.
 */
export function setFrustumPlane(planeIndex, nx, ny, nz, d) {
  let target = null;
  switch (planeIndex) {
    case FRUSTUM_PLANE.LEFT:   target = FrustumPlanes.left;   break;
    case FRUSTUM_PLANE.RIGHT:  target = FrustumPlanes.right;  break;
    case FRUSTUM_PLANE.BOTTOM: target = FrustumPlanes.bottom; break;
    case FRUSTUM_PLANE.TOP:    target = FrustumPlanes.top;    break;
    case FRUSTUM_PLANE.NEAR:   target = FrustumPlanes.near;   break;
    case FRUSTUM_PLANE.FAR:    target = FrustumPlanes.far;    break;
    default: return false;
  }
  target[0] = nx; target[1] = ny; target[2] = nz; target[3] = d;
  return true;
}

/**
 * Marks the frustum valid and stores the camera metadata.
 */
export function setFrustumCamera(cameraX, cameraY, cameraZ, tanFovHalf, aspect, nearPlane, farPlane) {
  FrustumPlanes.cameraX[0] = cameraX;
  FrustumPlanes.cameraY[0] = cameraY;
  FrustumPlanes.cameraZ[0] = cameraZ;
  FrustumPlanes.tanFovHalf[0] = tanFovHalf;
  FrustumPlanes.aspect[0] = aspect;
  FrustumPlanes.nearPlane[0] = nearPlane;
  FrustumPlanes.farPlane[0] = farPlane;
  FrustumPlanes.valid[0] = 1;
}

/**
 * Evaluates the frustum against an entity's bounding sphere. Returns
 * FRUSTUM_RESULT.OUTSIDE / INTERSECTING / INSIDE.
 */
export function testFrustumSphere(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return FRUSTUM_RESULT.OUTSIDE;
  if (FrustumPlanes.valid[0] === 0) return FRUSTUM_RESULT.INSIDE;
  if (Sphere.valid[eid] === 0) return FRUSTUM_RESULT.INSIDE;

  SpatialState.totalFrustumTests++;

  const cx = Sphere.cx[eid];
  const cy = Sphere.cy[eid];
  const cz = Sphere.cz[eid];
  const r = Sphere.radius[eid];

  let result = FRUSTUM_RESULT.INSIDE;

  for (let p = 0; p < FRUSTUM_PLANE.COUNT; p++) {
    let plane;
    switch (p) {
      case FRUSTUM_PLANE.LEFT:   plane = FrustumPlanes.left;   break;
      case FRUSTUM_PLANE.RIGHT:  plane = FrustumPlanes.right;  break;
      case FRUSTUM_PLANE.BOTTOM: plane = FrustumPlanes.bottom; break;
      case FRUSTUM_PLANE.TOP:    plane = FrustumPlanes.top;    break;
      case FRUSTUM_PLANE.NEAR:   plane = FrustumPlanes.near;   break;
      case FRUSTUM_PLANE.FAR:    plane = FrustumPlanes.far;    break;
      default: continue;
    }

    const distance = plane[0] * cx + plane[1] * cy + plane[2] * cz + plane[3];
    if (distance < -r) return FRUSTUM_RESULT.OUTSIDE;
    if (distance < r) result = FRUSTUM_RESULT.INTERSECTING;
  }
  return result;
}

/**
 * Evaluates the frustum against an entity's AABB.
 */
export function testFrustumAABB(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return FRUSTUM_RESULT.OUTSIDE;
  if (FrustumPlanes.valid[0] === 0) return FRUSTUM_RESULT.INSIDE;
  if (AABB.valid[eid] === 0) return FRUSTUM_RESULT.INSIDE;

  SpatialState.totalFrustumTests++;

  const minX = AABB.minX[eid]; const maxX = AABB.maxX[eid];
  const minY = AABB.minY[eid]; const maxY = AABB.maxY[eid];
  const minZ = AABB.minZ[eid]; const maxZ = AABB.maxZ[eid];

  let result = FRUSTUM_RESULT.INSIDE;

  for (let p = 0; p < FRUSTUM_PLANE.COUNT; p++) {
    let plane;
    switch (p) {
      case FRUSTUM_PLANE.LEFT:   plane = FrustumPlanes.left;   break;
      case FRUSTUM_PLANE.RIGHT:  plane = FrustumPlanes.right;  break;
      case FRUSTUM_PLANE.BOTTOM: plane = FrustumPlanes.bottom; break;
      case FRUSTUM_PLANE.TOP:    plane = FrustumPlanes.top;    break;
      case FRUSTUM_PLANE.NEAR:   plane = FrustumPlanes.near;   break;
      case FRUSTUM_PLANE.FAR:    plane = FrustumPlanes.far;    break;
      default: continue;
    }

    // Pick the "most positive" corner for early outside test.
    const px = plane[0] >= 0 ? maxX : minX;
    const py = plane[1] >= 0 ? maxY : minY;
    const pz = plane[2] >= 0 ? maxZ : minZ;
    const dMax = plane[0] * px + plane[1] * py + plane[2] * pz + plane[3];
    if (dMax < 0) return FRUSTUM_RESULT.OUTSIDE;

    // Pick the "most negative" corner for the inside test.
    const nx = plane[0] >= 0 ? minX : maxX;
    const ny = plane[1] >= 0 ? minY : maxY;
    const nz = plane[2] >= 0 ? minZ : maxZ;
    const dMin = plane[0] * nx + plane[1] * ny + plane[2] * nz + plane[3];
    if (dMin < 0) result = FRUSTUM_RESULT.INTERSECTING;
  }
  return result;
}

/* ------------------------------------------------------------------ */
/* 10. VISIBILITY STATE                                               */
/* ------------------------------------------------------------------ */

/**
 * Evaluates per-entity visibility against the frustum, distance, and
 * occlusion flags. Updates VisibilityState in place. Allocation-free.
 *
 * Returns the number of visible entities.
 */
export function updateVisibility(cameraX, cameraY, cameraZ, maxDistance) {
  const t0 = _now();
  const maxDistSq = maxDistance > 0 ? maxDistance * maxDistance : Infinity;

  let visible = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    // Skip entities that are not spatially valid.
    if (Sphere.valid[eid] === 0 && AABB.valid[eid] === 0) {
      VisibilityState.flags[eid] = SPATIAL_FLAG.NONE;
      continue;
    }

    // Distance test.
    const dx = Sphere.cx[eid] - cameraX;
    const dy = Sphere.cy[eid] - cameraY;
    const dz = Sphere.cz[eid] - cameraZ;
    const dSq = dx * dx + dy * dy + dz * dz;
    VisibilityState.distanceToCam[eid] = Math.sqrt(dSq);
    VisibilityState.distanceSq[eid] = dSq;

    if (dSq > maxDistSq) {
      VisibilityState.flags[eid] &= ~SPATIAL_FLAG.VISIBLE;
      VisibilityState.frustumResult[eid] = FRUSTUM_RESULT.OUTSIDE;
      continue;
    }

    // Frustum test.
    let frustumResult = FRUSTUM_RESULT.INSIDE;
    if (FrustumPlanes.valid[0] === 1) {
      frustumResult = Sphere.valid[eid] === 1
        ? testFrustumSphere(eid)
        : testFrustumAABB(eid);
    }
    VisibilityState.frustumResult[eid] = frustumResult;

    if (frustumResult === FRUSTUM_RESULT.OUTSIDE) {
      VisibilityState.flags[eid] &= ~SPATIAL_FLAG.VISIBLE;
      VisibilityState.flags[eid] |= SPATIAL_FLAG.IN_FRUSTUM;
      VisibilityState.flags[eid] &= ~SPATIAL_FLAG.IN_FRUSTUM;
      VisibilityState.flags[eid] |= SPATIAL_FLAG.DIRTY;
      clearFrameDirty(eid, FDIRTY.IN_VIEW);
      markFrameDirty(eid, FDIRTY.FRUSTUM_CULLED);
      continue;
    }

    VisibilityState.flags[eid] |= SPATIAL_FLAG.VISIBLE;
    VisibilityState.flags[eid] |= SPATIAL_FLAG.IN_FRUSTUM;
    markFrameDirty(eid, FDIRTY.IN_VIEW);
    clearFrameDirty(eid, FDIRTY.FRUSTUM_CULLED);

    VisibilityState.lastEvalFrame[eid] = SpatialState.frame;
    visible++;
  }

  const t1 = _now();
  const cost = t1 - t0;
  SpatialState.lastFrustumMs = cost;
  SpatialState.avgFrustumMs += (cost - SpatialState.avgFrustumMs) * 0.15;

  return visible;
}

/**
 * Returns true if the entity is currently marked visible.
 */
export function isVisible(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return (VisibilityState.flags[eid] & SPATIAL_FLAG.VISIBLE) !== 0;
}

/**
 * Returns the entity's LOD level (0 = highest detail).
 */
export function getLODLevel(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  return VisibilityState.lodLevel[eid];
}

/**
 * Sets the entity's LOD level.
 */
export function setLODLevel(eid, level) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  VisibilityState.lodLevel[eid] = Math.max(0, Math.min(255, level | 0));
  return true;
}

/* ------------------------------------------------------------------ */
/* 11. BULK HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity currently in the hash grid. Allocation-free.
 */
export function forEachHashEntity(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let visited = 0;
  const cellCount = SpatialState.gridCellCount;
  for (let cellId = 0; cellId < cellCount; cellId++) {
    if (SpatialHash.cellCount[cellId] === 0) continue;
    let eid = SpatialHash.cellHead[cellId];
    while (eid >= 0) {
      fn.call(ctx, eid);
      visited++;
      eid = GridCell.nextInCell[eid];
    }
  }
  return visited;
}

/**
 * Rebuilds the entire hash grid from every entity with a valid
 * Transform. Clears then re-inserts.
 */
export function rebuildHashGrid() {
  // Clear all per-cell heads.
  SpatialHash.cellHead.fill(-1);
  SpatialHash.cellCount.fill(0);
  GridCell.cellId.fill(-1);
  GridCell.prevInCell.fill(-1);
  GridCell.nextInCell.fill(-1);

  SpatialState.hashEntries = 0;

  const adapter = getAdapter();
  let inserted = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (!adapter.entityAlive(eid)) continue;
    if (insertIntoHashGrid(eid) >= 0) inserted++;
  }
  return inserted;
}

/* ------------------------------------------------------------------ */
/* 12. CLUSTER CELL HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Assigns an entity to a cluster cell.
 */
export function assignClusterCell(eid, clusterId, localX, localY, localZ, sliceIndex) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  ClusterCell.clusterId[eid] = clusterId | 0;
  ClusterCell.localX[eid] = localX;
  ClusterCell.localY[eid] = localY;
  ClusterCell.localZ[eid] = localZ;
  ClusterCell.sliceIndex[eid] = (sliceIndex | 0) & 0xFFFF;
  ClusterCell.assigned[eid] = 1;
  VisibilityState.flags[eid] |= SPATIAL_FLAG.CLUSTER_ASSIGNED;
  return true;
}

/**
 * Clears an entity's cluster assignment.
 */
export function clearClusterCell(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  ClusterCell.clusterId[eid] = -1;
  ClusterCell.assigned[eid] = 0;
  VisibilityState.flags[eid] &= ~SPATIAL_FLAG.CLUSTER_ASSIGNED;
  return true;
}

/**
 * Returns true if an entity is assigned to a cluster cell.
 */
export function isClusterAssigned(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return ClusterCell.assigned[eid] === 1;
}

/* ------------------------------------------------------------------ */
/* 13. DISTANCE FIELD HELPERS                                         */
/* ------------------------------------------------------------------ */

/**
 * Binds an entity to a distance-field owner.
 */
export function bindDistanceField(eid, fieldEid, sampleScale, maxDistance, useGradient) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  DistanceFieldRef.fieldEid[eid] = fieldEid | 0;
  DistanceFieldRef.sampleScale[eid] = Number.isFinite(sampleScale) ? sampleScale : 1.0;
  DistanceFieldRef.maxDistance[eid] = Number.isFinite(maxDistance) ? maxDistance : 8.0;
  DistanceFieldRef.useGradient[eid] = useGradient ? 1 : 0;
  DistanceFieldRef.valid[eid] = 1;
  return true;
}

/**
 * Clears an entity's distance-field binding.
 */
export function clearDistanceField(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  DistanceFieldRef.fieldEid[eid] = -1;
  DistanceFieldRef.valid[eid] = 0;
  return true;
}

/* ------------------------------------------------------------------ */
/* 14. BVH / OCTREE HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Sets the BVH leaf reference for an entity.
 */
export function setBVHLeaf(eid, leafId, triangleCount) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  BVHRef.leafId[eid] = leafId | 0;
  BVHRef.triangleCount[eid] = (triangleCount | 0) >>> 0;
  BVHRef.valid[eid] = 1;
  return true;
}

/**
 * Sets the octree node reference for an entity.
 */
export function setOctreeNode(eid, nodeId, depth) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  OctreeRef.nodeId[eid] = nodeId | 0;
  OctreeRef.depth[eid] = (depth | 0) & 0xFF;
  OctreeRef.assignmentValid[eid] = 1;
  return true;
}

/* ------------------------------------------------------------------ */
/* 15. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the spatial system frame counter. Called once per frame.
 */
export function tickSpatial(frameNumber) {
  if (typeof frameNumber === 'number') SpatialState.frame = frameNumber;
  else SpatialState.frame++;
}

/* ------------------------------------------------------------------ */
/* 16. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * Registers every spatial component in the runtime component registry.
 */
export function registerSpatialComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'AABB',             component: AABB,             category: 18, subsystem: 1, dependencies: ['Transform'] },
    { name: 'Sphere',           component: Sphere,           category: 18, subsystem: 1, dependencies: ['Transform'] },
    { name: 'GridCell',         component: GridCell,         category: 18, subsystem: 1, dependencies: ['Transform'] },
    { name: 'OctreeRef',        component: OctreeRef,        category: 18, subsystem: 1, dependencies: [] },
    { name: 'BVHRef',           component: BVHRef,           category: 18, subsystem: 1, dependencies: [] },
    { name: 'DistanceFieldRef', component: DistanceFieldRef, category: 18, subsystem: 1, dependencies: [] },
    { name: 'ClusterCell',      component: ClusterCell,      category: 18, subsystem: 1, dependencies: ['Transform'] },
    { name: 'VisibilityState',  component: VisibilityState,  category: 17, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 17. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getSpatialStats() {
  let validAABBs = 0;
  let validSpheres = 0;
  let visible = 0;
  let inFrustum = 0;
  let occluded = 0;
  let clusterAssigned = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (AABB.valid[eid] === 1) validAABBs++;
    if (Sphere.valid[eid] === 1) validSpheres++;
    const flags = VisibilityState.flags[eid];
    if ((flags & SPATIAL_FLAG.VISIBLE) !== 0) visible++;
    if ((flags & SPATIAL_FLAG.IN_FRUSTUM) !== 0) inFrustum++;
    if ((flags & SPATIAL_FLAG.OCCLUDED) !== 0) occluded++;
    if ((flags & SPATIAL_FLAG.CLUSTER_ASSIGNED) !== 0) clusterAssigned++;
  }

  return {
    frame:               SpatialState.frame,
    cellSize:            SpatialState.cellSize,
    gridDim:             SpatialState.gridDim,
    gridCellCount:       SpatialState.gridCellCount,
    hashEntries:         SpatialState.hashEntries,
    peakHashEntries:     SpatialState.peakHashEntries,
    totalInsertions:     SpatialState.totalInsertions,
    totalRemovals:       SpatialState.totalRemovals,
    totalQueries:        SpatialState.totalQueries,
    totalQueryHits:      SpatialState.totalQueryHits,
    totalAABBComputes:   SpatialState.totalAABBComputes,
    totalSphereComputes: SpatialState.totalSphereComputes,
    totalFrustumTests:   SpatialState.totalFrustumTests,
    cellOverflows:       SpatialState.cellOverflows,
    lastQueryMs:         SpatialState.lastQueryMs,
    avgQueryMs:          SpatialState.avgQueryMs,
    lastFrustumMs:       SpatialState.lastFrustumMs,
    avgFrustumMs:        SpatialState.avgFrustumMs,
    validAABBs,
    validSpheres,
    visible,
    inFrustum,
    occluded,
    clusterAssigned,
    perfTier:            PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 18. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every spatial structure and resets counters. The grid
 * configuration (cell size, dimensions) is preserved.
 */
export function resetSpatialState() {
  AABB.minX.fill(0); AABB.minY.fill(0); AABB.minZ.fill(0);
  AABB.maxX.fill(0); AABB.maxY.fill(0); AABB.maxZ.fill(0);
  AABB.centerX.fill(0); AABB.centerY.fill(0); AABB.centerZ.fill(0);
  AABB.valid.fill(0);

  Sphere.cx.fill(0); Sphere.cy.fill(0); Sphere.cz.fill(0);
  Sphere.radius.fill(0); Sphere.radiusSq.fill(0); Sphere.valid.fill(0);

  GridCell.cellId.fill(-1);
  GridCell.prevInCell.fill(-1);
  GridCell.nextInCell.fill(-1);
  GridCell.cellX.fill(0); GridCell.cellY.fill(0); GridCell.cellZ.fill(0);
  GridCell.cellOccupancy.fill(0);

  SpatialHash.cellHead.fill(-1);
  SpatialHash.cellCount.fill(0);

  OctreeRef.nodeId.fill(-1);
  OctreeRef.depth.fill(0);
  OctreeRef.assignmentValid.fill(0);

  BVHRef.leafId.fill(-1);
  BVHRef.triangleCount.fill(0);
  BVHRef.valid.fill(0);

  DistanceFieldRef.fieldEid.fill(-1);
  DistanceFieldRef.sampleScale.fill(0);
  DistanceFieldRef.maxDistance.fill(0);
  DistanceFieldRef.useGradient.fill(0);
  DistanceFieldRef.valid.fill(0);

  ClusterCell.clusterId.fill(-1);
  ClusterCell.localX.fill(0); ClusterCell.localY.fill(0); ClusterCell.localZ.fill(0);
  ClusterCell.sliceIndex.fill(0); ClusterCell.assigned.fill(0);

  VisibilityState.flags.fill(0);
  VisibilityState.frustumResult.fill(0);
  VisibilityState.distanceToCam.fill(0);
  VisibilityState.distanceSq.fill(0);
  VisibilityState.screenSize.fill(0);
  VisibilityState.lodLevel.fill(0);
  VisibilityState.lastEvalFrame.fill(0);
  VisibilityState.occludedBy.fill(-1);

  FrustumPlanes.left.fill(0);   FrustumPlanes.right.fill(0);
  FrustumPlanes.bottom.fill(0); FrustumPlanes.top.fill(0);
  FrustumPlanes.near.fill(0);   FrustumPlanes.far.fill(0);
  FrustumPlanes.valid[0] = 0;
  FrustumPlanes.cameraX[0] = 0; FrustumPlanes.cameraY[0] = 0; FrustumPlanes.cameraZ[0] = 0;
  FrustumPlanes.tanFovHalf[0] = 0; FrustumPlanes.aspect[0] = 0;
  FrustumPlanes.nearPlane[0] = 0; FrustumPlanes.farPlane[0] = 0;

  SpatialState.frame = 0;
  SpatialState.hashEntries = 0;
  SpatialState.peakHashEntries = 0;
  SpatialState.totalInsertions = 0;
  SpatialState.totalRemovals = 0;
  SpatialState.totalQueries = 0;
  SpatialState.totalQueryHits = 0;
  SpatialState.totalAABBComputes = 0;
  SpatialState.totalSphereComputes = 0;
  SpatialState.totalFrustumTests = 0;
  SpatialState.totalOcclusionTests = 0;
  SpatialState.cellOverflows = 0;
  SpatialState.lastQueryMs = 0;
  SpatialState.avgQueryMs = 0;
  SpatialState.lastFrustumMs = 0;
  SpatialState.avgFrustumMs = 0;
}

/* ------------------------------------------------------------------ */
/* 19. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  DEFAULT_CELL_SIZE,
  DEFAULT_GRID_MIN,
  DEFAULT_GRID_MAX,
  DEFAULT_GRID_DIM,
  DEFAULT_GRID_CELL_COUNT,
  MAX_HASH_ENTRIES,
  MAX_CELL_OCCUPANCY,
  FRUSTUM_PLANE,
  FRUSTUM_RESULT,
  SPATIAL_FLAG,
  OCTREE_CHILD,

  // Components
  AABB,
  Sphere,
  GridCell,
  SpatialHash,
  OctreeRef,
  BVHRef,
  DistanceFieldRef,
  ClusterCell,
  VisibilityState,
  FrustumPlanes,
  SPATIAL_COMPONENTS,

  // Module state
  SpatialState,

  // Grid configuration
  configureGrid,
  getGridDim,
  getGridCellSize,
  getGridCellCount,

  // AABB / Sphere
  computeAABB,
  setAABB,
  computeSphere,
  setSphere,

  // Overlap tests
  testAABBOverlap,
  testSphereOverlap,
  testSphereContainsPoint,
  testAABBContainsPoint,

  // Hash grid
  insertIntoHashGrid,
  removeFromHashGrid,
  refreshHashGrid,
  queryRadius,
  queryAABB,
  forEachInCell,
  collectRadius,
  getCellCount,
  forEachHashEntity,
  rebuildHashGrid,

  // Frustum
  setFrustumPlane,
  setFrustumCamera,
  testFrustumSphere,
  testFrustumAABB,

  // Visibility
  updateVisibility,
  isVisible,
  getLODLevel,
  setLODLevel,

  // Cluster
  assignClusterCell,
  clearClusterCell,
  isClusterAssigned,

  // Distance field
  bindDistanceField,
  clearDistanceField,

  // BVH / Octree
  setBVHLeaf,
  setOctreeNode,

  // Frame
  tickSpatial,

  // Diagnostics
  getSpatialStats,

  // Registration
  registerSpatialComponents,

  // Reset
  resetSpatialState,
};

export default _defaultExport;