// File : 028
// name : src/ecs/028_scn_StreamingComponents.js
// description : Streaming SoA component module for the scene ECS world of the
//               anime lighting stack on Android mobile. Declares every
//               streaming-chunk state, load-state machine, budget counter,
//               priority ranking, residency record, async worker binding,
//               mesh / texture residency, and streaming statistics the
//               lighting stack needs — as fixed-capacity typed arrays sized
//               once to MAX_ENTITIES = 100000.
//
//               Provides the fast helpers that the world streamer runs on
//               the hot path:
//                 • registerChunk             — declare a chunk entity
//                 • requestChunkLoad          — enqueue a chunk for load
//                 • cancelChunkLoad           — abort a pending load
//                 • markChunkLoading          — transition to LOADING
//                 • markChunkResident         — transition to RESIDENT
//                 • markChunkEvict            — transition to EVICTING
//                 • markChunkFailed           — transition to FAILED
//                 • evaluateChunkPriority     — compute the streaming priority
//                                               from distance + camera direction
//                 • updateStreamingBudget     — recompute the aggregate cost
//                 • advanceChunkLoads         — advance every pending load
//                                               state machine one frame
//                 • collectEvictableChunks    — enumerate chunks eligible for
//                                               eviction
//                 • tickStreaming             — one-shot per-frame pipeline
//                 • refreshChunkDistance      — recompute camera distance
//                 • setChunkBudget            — per-chunk cost budget
//                 • setChunkMeshResident      — mesh residency flag
//                 • setChunkTextureResident   — texture residency flag
//                 • setChunkLightsResident    — light residency flag
//                 • setChunkGIResident        — GI residency flag
//                 • setChunkWorkerRef         — bind a worker slot
//                 • clearChunkWorkerRef       — unbind the worker slot
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no runtime resizes.
//                 • State machine: IDLE → REQUESTED → LOADING → RESIDENT →
//                   EVICTING → IDLE (or → FAILED → IDLE via retry).
//                 • Priority is a signed float: higher value = higher
//                   priority. Computed from inverse distance, camera
//                   facing dot product, and a per-chunk bias.
//                 • Budget-aware: the streamer can load up to
//                   `budgetCap` cost units per frame. `updateStreamingBudget`
//                   recomputes the aggregate cost and marks the budget
//                   exceeded flag.
//                 • Residency tracking per chunk: mesh, texture, lights,
//                   GI probes. A chunk is fully resident when all four
//                   components are resident.
//                 • Async worker binding: each chunk can be assigned to one
//                   of the parallel worker slots so async geometry and
//                   texture generation is deduplicated.
//                 • Zero allocations on the hot path — every helper works
//                   directly on the SoA arrays; scratch vectors are
//                   module-level and reused.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — lights spawned by chunks
//                 • 003_lgt_ShadowComponents.js     — shadow casters in chunks
//                 • 004_lgt_GIComponents.js         — GI probes in chunks
//                 • 005_lgt_AOComponents.js         — AO volumes in chunks
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — streaming tags
//                 • 015_scn_Relations.js            — chunk relations
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime records
//                 • 022_scn_SpawnPrefabs.js         — prefab spawn
//                 • 025_scn_SpatialComponents.js    — chunk bounds
//                 • 026_scn_TransformComponents.js  — world transforms
//                 • 027_scn_LODComponents.js        — chunk LOD levels
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that the anime lighting stack streams procedural
//            chunks, lights, GI probes, and AO volumes in and out of memory
//            deterministically — with priority ranking, budget-aware load
//            admission, residency tracking, and async worker binding, all
//            in one fixed-capacity SoA record per chunk.
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
  markFrameDirty,
  clearFrameDirty,
} from './014_scn_Tags.js';

import {
  Parent,
  Children,
  MAX_CHILDREN_PER_ENTITY,
  NULL_ENTITY,
  bindStreamingChunk,
  detachAllRelations,
} from './015_scn_Relations.js';

import {
  POOL,
  isInUse,
} from './016_scn_EntityPool.js';

import {
  LIFETIME_STATE,
  EntityLifetime,
} from './017_scn_EntityLifetime.js';

import {
  TransformWorld,
} from './026_scn_TransformComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of chunks that can be in the streaming pipeline at once.
 * Bounded by MAX_ENTITIES.
 */
export const MAX_STREAMING_CHUNKS = MAX_ENTITIES;

/**
 * Maximum number of chunks that can be resident simultaneously. Sized by
 * tier because each resident chunk holds real GPU memory.
 */
export const MAX_RESIDENT_CHUNKS =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 256 :
                                 128;

/**
 * Maximum number of chunk load requests admitted per frame.
 */
export const MAX_LOADS_PER_FRAME =
  PERF_TIER_LOCAL === 'HIGH'   ? 4 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2 :
                                 1;

/**
 * Maximum number of chunk evictions admitted per frame.
 */
export const MAX_EVICTIONS_PER_FRAME =
  PERF_TIER_LOCAL === 'HIGH'   ? 4 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2 :
                                 1;

/**
 * Number of frames a chunk load is allowed to stay in LOADING before it
 * is marked FAILED and can be retried.
 */
export const LOAD_TIMEOUT_FRAMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 300 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 240 :
                                 180;

/**
 * Default streaming budget cap in relative cost units per frame.
 */
export const DEFAULT_BUDGET_CAP =
  PERF_TIER_LOCAL === 'HIGH'   ? 8.0 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 4.0 :
                                 2.0;

/**
 * Default chunk radius (world units) used for priority computation.
 */
export const DEFAULT_CHUNK_RADIUS = 40.0;

/**
 * Chunk load state machine.
 */
export const STREAM_STATE = Object.freeze({
  IDLE:       0,   // not in the pipeline
  REQUESTED:  1,   // queued for load
  LOADING:    2,   // worker / generator is producing the chunk
  RESIDENT:   3,   // fully loaded and visible
  EVICTING:   4,   // queued for unload
  UNLOADING:  5,   // actively releasing resources
  FAILED:     6,   // load failed — eligible for retry
  COUNT:      7,
});

export const STREAM_STATE_NAME = Object.freeze([
  'idle',
  'requested',
  'loading',
  'resident',
  'evicting',
  'unloading',
  'failed',
]);

/**
 * Per-chunk residency flags. A chunk is fully resident when all four
 * flags are set.
 */
export const CHUNK_RESIDENCY = Object.freeze({
  NONE:     0,
  MESH:     1 << 0,
  TEXTURE:  1 << 1,
  LIGHTS:   1 << 2,
  GI:       1 << 3,
  AO:       1 << 4,
  SHADOW:   1 << 5,
  AUDIO:    1 << 6,
  FULL:     0xFF,
});

/**
 * Per-chunk streaming flags.
 */
export const STREAM_FLAG = Object.freeze({
  NONE:           0,
  ENABLED:        1 << 0,
  HIGH_PRIORITY:  1 << 1,
  PINNED:         1 << 2,   // never evict
  STREAMING_MESH: 1 << 3,
  STREAMING_TEX:  1 << 4,
  STREAMING_GI:   1 << 5,
  IN_FRUSTUM:     1 << 6,
  BEHIND_CAMERA:  1 << 7,
  SPAWNED_LIGHTS: 1 << 8,
  SPAWNED_GI:     1 << 9,
  ASYNC_IN_FLIGHT:1 << 10,
  DIRTY:          1 << 11,
});

/**
 * Chunk LOD level index (independent of entity LOD — chunks have their
 * own coarser granularity).
 */
export const CHUNK_LOD = Object.freeze({
  FULL:     0,
  HALF:     1,
  QUARTER:  2,
  PROXY:    3,
  COUNT:    4,
});

/**
 * Reason codes for debug HUD.
 */
export const STREAM_REASON = Object.freeze({
  NONE:           0,
  DISTANCE:       1,
  FRUSTUM:        2,
  BUDGET:         3,
  PRIORITY:       4,
  FAILED:         5,
  PINNED:         6,
  MANUAL:         7,
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const StreamingState = {
  frame:                    0,
  budgetCap:                DEFAULT_BUDGET_CAP,
  budgetUsed:               0,
  budgetExceeded:           0,
  residentCount:            0,
  peakResidentCount:        0,
  totalRegistrations:       0,
  totalLoadRequests:        0,
  totalLoadCompletions:     0,
  totalLoadFailures:        0,
  totalEvictions:           0,
  totalLoadsPerFrame:       0,
  totalEvictionsPerFrame:   0,
  totalAdmissionRejects:    0,
  totalTimeoutFails:        0,
  lastTickMs:               0,
  avgTickMs:                0,
  lastLoadMs:               0,
  avgLoadMs:                0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.streaming', {
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
 * StreamingChunk — the canonical streaming chunk record.
 */
export const StreamingChunk = {
  state:              new Uint8Array(MAX_ENTITIES),
  prevState:          new Uint8Array(MAX_ENTITIES),
  flags:              new Uint16Array(MAX_ENTITIES),
  lod:                new Uint8Array(MAX_ENTITIES),
  lodTarget:          new Uint8Array(MAX_ENTITIES),

  gridX:              new Int16Array(MAX_ENTITIES),
  gridZ:              new Int16Array(MAX_ENTITIES),
  chunkSize:          new Float32Array(MAX_ENTITIES),
  centerX:            new Float32Array(MAX_ENTITIES),
  centerY:            new Float32Array(MAX_ENTITIES),
  centerZ:            new Float32Array(MAX_ENTITIES),
  radius:             new Float32Array(MAX_ENTITIES),

  distance:           new Float32Array(MAX_ENTITIES),
  distanceSq:         new Float32Array(MAX_ENTITIES),
  priority:           new Float32Array(MAX_ENTITIES),
  bias:               new Float32Array(MAX_ENTITIES),

  lastStateFrame:     new Uint32Array(MAX_ENTITIES),
  lastAccessFrame:    new Uint32Array(MAX_ENTITIES),
  requestedAtFrame:   new Uint32Array(MAX_ENTITIES),
  residentAtFrame:    new Uint32Array(MAX_ENTITIES),
  evictedAtFrame:     new Uint32Array(MAX_ENTITIES),

  loadFrameCount:     new Uint16Array(MAX_ENTITIES),
  loadTimeoutFrames:  new Uint16Array(MAX_ENTITIES),
  retryCount:         new Uint8Array(MAX_ENTITIES),
  maxRetries:         new Uint8Array(MAX_ENTITIES),

  seed:               new Uint32Array(MAX_ENTITIES),
  biomeId:            new Uint8Array(MAX_ENTITIES),
  variant:            new Uint8Array(MAX_ENTITIES),

  reasonCode:         new Uint8Array(MAX_ENTITIES),
  lastFrameLoaded:    new Uint32Array(MAX_ENTITIES),
};

/**
 * StreamingResidency — per-chunk sub-resource residency flags.
 */
export const StreamingResidency = {
  mesh:      new Uint8Array(MAX_ENTITIES),
  texture:   new Uint8Array(MAX_ENTITIES),
  lights:    new Uint8Array(MAX_ENTITIES),
  gi:        new Uint8Array(MAX_ENTITIES),
  ao:        new Uint8Array(MAX_ENTITIES),
  shadow:    new Uint8Array(MAX_ENTITIES),
  audio:     new Uint8Array(MAX_ENTITIES),
  mask:      new Uint8Array(MAX_ENTITIES),   // CHUNK_RESIDENCY bitmask
};

/**
 * StreamingBudget — per-chunk cost and aggregate accounting.
 */
export const StreamingBudget = {
  entityCost:        new Float32Array(MAX_ENTITIES),
  entityCostEma:     new Float32Array(MAX_ENTITIES),
  loadCostEstimate:  new Float32Array(MAX_ENTITIES),
  evictCostEstimate: new Float32Array(MAX_ENTITIES),
  bytesEstimate:     new Uint32Array(MAX_ENTITIES),
  totalResidentCost: new Float32Array(1),
  totalResidentBytes:new Uint32Array(1),
  budgetCap:         new Float32Array(1),
  budgetExceeded:    new Uint8Array(1),
};

/**
 * StreamingPriority — sorted priority metadata used by the load queue.
 */
export const StreamingPriority = {
  bucket:      new Uint8Array(MAX_ENTITIES),   // 0=critical 1=high 2=normal 3=low 4=idle
  sortKey:     new Float32Array(MAX_ENTITIES),
  frameRank:   new Uint16Array(MAX_ENTITIES),
  dirty:       new Uint8Array(MAX_ENTITIES),
};

/**
 * StreamingMeshState — mesh residency details for the chunk.
 */
export const StreamingMeshState = {
  geometryId:     new Int32Array(MAX_ENTITIES).fill(-1),
  materialId:     new Int32Array(MAX_ENTITIES).fill(-1),
  proxyGeometryId:new Int32Array(MAX_ENTITIES).fill(-1),
  triangleCount:  new Uint32Array(MAX_ENTITIES),
  vertexCount:    new Uint32Array(MAX_ENTITIES),
  instanceCount:  new Uint16Array(MAX_ENTITIES),
};

/**
 * StreamingTextureState — texture residency details for the chunk.
 */
export const StreamingTextureState = {
  albedoId:       new Int32Array(MAX_ENTITIES).fill(-1),
  normalId:       new Int32Array(MAX_ENTITIES).fill(-1),
  roughnessId:    new Int32Array(MAX_ENTITIES).fill(-1),
  emissionId:     new Int32Array(MAX_ENTITIES).fill(-1),
  textureBytes:   new Uint32Array(MAX_ENTITIES),
};

/**
 * StreamingLightState — lights spawned within a chunk.
 */
export const StreamingLightState = {
  lightCount:     new Uint16Array(MAX_ENTITIES),
  shadowCasters:  new Uint16Array(MAX_ENTITIES),
  giProbeCount:   new Uint16Array(MAX_ENTITIES),
  aoVolumeCount:  new Uint16Array(MAX_ENTITIES),
  firstLightEid:  new Int32Array(MAX_ENTITIES).fill(-1),
  firstGIEid:     new Int32Array(MAX_ENTITIES).fill(-1),
  firstAOEid:     new Int32Array(MAX_ENTITIES).fill(-1),
};

/**
 * StreamingAsync — async worker binding per chunk.
 */
export const StreamingAsync = {
  workerSlot:       new Int8Array(MAX_ENTITIES).fill(-1),
  asyncStartFrame:  new Uint32Array(MAX_ENTITIES),
  asyncStage:       new Uint8Array(MAX_ENTITIES),   // 0=none 1=geometry 2=texture 3=finalize
  asyncProgress:    new Float32Array(MAX_ENTITIES),
  asyncCancelFlag:  new Uint8Array(MAX_ENTITIES),
  asyncErrorCode:   new Uint16Array(MAX_ENTITIES),
};

/**
 * StreamingWorkerRef — per-worker slot state (fixed 8 workers).
 */
export const StreamingWorkerRef = {
  workerBusy:      new Uint8Array(8),
  workerChunkEid:  new Int32Array(8).fill(-1),
  workerStartFrame:new Uint32Array(8),
  workerJobCount:  new Uint32Array(8),
  workerFailCount: new Uint32Array(8),
  workerTotalMs:   new Float32Array(8),
  workerLastMs:    new Float32Array(8),
};

/**
 * StreamingStats — aggregate per-frame statistics.
 */
export const StreamingStats = {
  residentCount:       new Uint32Array(1),
  loadingCount:        new Uint32Array(1),
  requestedCount:      new Uint32Array(1),
  evictingCount:       new Uint32Array(1),
  failedCount:         new Uint32Array(1),
  stateHistogram:      new Uint32Array(STREAM_STATE.COUNT),
  loadsThisFrame:      new Uint32Array(1),
  evictionsThisFrame:  new Uint32Array(1),
  budgetExceeded:      new Uint8Array(1),
};

/**
 * Streaming component bundle for bitECS createWorld.
 */
export const STREAMING_COMPONENTS = Object.freeze({
  StreamingChunk,
  StreamingResidency,
  StreamingBudget,
  StreamingPriority,
  StreamingMeshState,
  StreamingTextureState,
  StreamingLightState,
  StreamingAsync,
  StreamingWorkerRef,
  StreamingStats,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Scratch index arrays for sorting by priority. Sized once.
 */
const _priorityCandidates = new Int32Array(MAX_LOADS_PER_FRAME * 8);
const _evictCandidates = new Int32Array(MAX_EVICTIONS_PER_FRAME * 8);
const _chunkScratch = new Int32Array(MAX_ENTITIES);

/* ------------------------------------------------------------------ */
/* 4. CHUNK REGISTRATION                                              */
/* ------------------------------------------------------------------ */

/**
 * Registers an entity as a streaming chunk. Called once per chunk when
 * the chunk entity is created.
 */
export function registerChunk(eid, spec) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!spec || typeof spec !== 'object') return false;

  StreamingChunk.state[eid] = STREAM_STATE.IDLE;
  StreamingChunk.prevState[eid] = STREAM_STATE.IDLE;
  StreamingChunk.flags[eid] = STREAM_FLAG.ENABLED;
  StreamingChunk.lod[eid] = CHUNK_LOD.FULL;
  StreamingChunk.lodTarget[eid] = CHUNK_LOD.FULL;

  StreamingChunk.gridX[eid] = (spec.gridX | 0) & 0x7FFF;
  StreamingChunk.gridZ[eid] = (spec.gridZ | 0) & 0x7FFF;
  StreamingChunk.chunkSize[eid] = spec.chunkSize !== undefined ? Number(spec.chunkSize) : DEFAULT_CHUNK_RADIUS * 2;
  StreamingChunk.centerX[eid] = spec.centerX !== undefined ? Number(spec.centerX) : 0;
  StreamingChunk.centerY[eid] = spec.centerY !== undefined ? Number(spec.centerY) : 0;
  StreamingChunk.centerZ[eid] = spec.centerZ !== undefined ? Number(spec.centerZ) : 0;
  StreamingChunk.radius[eid] = spec.radius !== undefined ? Number(spec.radius) : DEFAULT_CHUNK_RADIUS;

  StreamingChunk.distance[eid] = Infinity;
  StreamingChunk.distanceSq[eid] = Infinity;
  StreamingChunk.priority[eid] = 0;
  StreamingChunk.bias[eid] = spec.bias !== undefined ? Number(spec.bias) : 1.0;

  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.lastAccessFrame[eid] = StreamingState.frame;
  StreamingChunk.requestedAtFrame[eid] = 0;
  StreamingChunk.residentAtFrame[eid] = 0;
  StreamingChunk.evictedAtFrame[eid] = 0;

  StreamingChunk.loadFrameCount[eid] = 0;
  StreamingChunk.loadTimeoutFrames[eid] = spec.loadTimeoutFrames !== undefined
    ? (spec.loadTimeoutFrames | 0)
    : LOAD_TIMEOUT_FRAMES;
  StreamingChunk.retryCount[eid] = 0;
  StreamingChunk.maxRetries[eid] = spec.maxRetries !== undefined ? (spec.maxRetries | 0) : 3;

  StreamingChunk.seed[eid] = (spec.seed !== undefined ? spec.seed : 0) >>> 0;
  StreamingChunk.biomeId[eid] = (spec.biomeId | 0) & 0xFF;
  StreamingChunk.variant[eid] = (spec.variant | 0) & 0xFF;

  StreamingChunk.reasonCode[eid] = STREAM_REASON.NONE;
  StreamingChunk.lastFrameLoaded[eid] = 0;

  // Residency.
  StreamingResidency.mesh[eid] = 0;
  StreamingResidency.texture[eid] = 0;
  StreamingResidency.lights[eid] = 0;
  StreamingResidency.gi[eid] = 0;
  StreamingResidency.ao[eid] = 0;
  StreamingResidency.shadow[eid] = 0;
  StreamingResidency.audio[eid] = 0;
  StreamingResidency.mask[eid] = CHUNK_RESIDENCY.NONE;

  // Budget.
  StreamingBudget.entityCost[eid] = 1.0;
  StreamingBudget.entityCostEma[eid] = 1.0;
  StreamingBudget.loadCostEstimate[eid] = spec.loadCost !== undefined ? Number(spec.loadCost) : 1.0;
  StreamingBudget.evictCostEstimate[eid] = spec.evictCost !== undefined ? Number(spec.evictCost) : 0.5;
  StreamingBudget.bytesEstimate[eid] = (spec.bytesEstimate !== undefined ? (spec.bytesEstimate | 0) : 0) >>> 0;

  // Priority.
  StreamingPriority.bucket[eid] = 2;   // normal
  StreamingPriority.sortKey[eid] = 0;
  StreamingPriority.frameRank[eid] = 0;
  StreamingPriority.dirty[eid] = 1;

  // Mesh / texture / light state.
  StreamingMeshState.geometryId[eid] = -1;
  StreamingMeshState.materialId[eid] = -1;
  StreamingMeshState.proxyGeometryId[eid] = -1;
  StreamingMeshState.triangleCount[eid] = 0;
  StreamingMeshState.vertexCount[eid] = 0;
  StreamingMeshState.instanceCount[eid] = 0;

  StreamingTextureState.albedoId[eid] = -1;
  StreamingTextureState.normalId[eid] = -1;
  StreamingTextureState.roughnessId[eid] = -1;
  StreamingTextureState.emissionId[eid] = -1;
  StreamingTextureState.textureBytes[eid] = 0;

  StreamingLightState.lightCount[eid] = 0;
  StreamingLightState.shadowCasters[eid] = 0;
  StreamingLightState.giProbeCount[eid] = 0;
  StreamingLightState.aoVolumeCount[eid] = 0;
  StreamingLightState.firstLightEid[eid] = -1;
  StreamingLightState.firstGIEid[eid] = -1;
  StreamingLightState.firstAOEid[eid] = -1;

  // Async.
  StreamingAsync.workerSlot[eid] = -1;
  StreamingAsync.asyncStartFrame[eid] = 0;
  StreamingAsync.asyncStage[eid] = 0;
  StreamingAsync.asyncProgress[eid] = 0;
  StreamingAsync.asyncCancelFlag[eid] = 0;
  StreamingAsync.asyncErrorCode[eid] = 0;

  // Tag the entity as a streaming chunk.
  EntityTag[eid] |= TAG.ACTIVE;
  EntityTag2[eid] |= TAG2.STREAMING_CHUNK;

  StreamingState.totalRegistrations++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 5. CHUNK LOAD REQUESTS                                             */
/* ------------------------------------------------------------------ */

/**
 * Requests that a chunk be loaded. Transitions IDLE → REQUESTED.
 * Returns true on success.
 */
export function requestChunkLoad(eid, reasonCode) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const state = StreamingChunk.state[eid];
  if (state === STREAM_STATE.REQUESTED ||
      state === STREAM_STATE.LOADING ||
      state === STREAM_STATE.RESIDENT) {
    return false;
  }
  if (state === STREAM_STATE.UNLOADING) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.REQUESTED;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.requestedAtFrame[eid] = StreamingState.frame;
  StreamingChunk.loadFrameCount[eid] = 0;
  StreamingChunk.reasonCode[eid] = reasonCode !== undefined ? reasonCode : STREAM_REASON.DISTANCE;

  StreamingState.totalLoadRequests++;
  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);

  return true;
}

/**
 * Cancels a pending load. REQUESTED → IDLE.
 */
export function cancelChunkLoad(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.REQUESTED) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.IDLE;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  return true;
}

/**
 * Transitions a chunk REQUESTED → LOADING. Called by the streamer when
 * a load slot is available.
 */
export function markChunkLoading(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.REQUESTED) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.LOADING;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.loadFrameCount[eid] = 0;
  StreamingChunk.flags[eid] |= STREAM_FLAG.ASYNC_IN_FLIGHT;
  return true;
}

/**
 * Transitions a chunk LOADING → RESIDENT. Called when all sub-resources
 * have been generated and attached.
 */
export function markChunkResident(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.LOADING && state !== STREAM_STATE.REQUESTED) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.RESIDENT;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.residentAtFrame[eid] = StreamingState.frame;
  StreamingChunk.lastFrameLoaded[eid] = StreamingState.frame;
  StreamingChunk.lastAccessFrame[eid] = StreamingState.frame;
  StreamingChunk.flags[eid] &= ~STREAM_FLAG.ASYNC_IN_FLIGHT;

  StreamingState.residentCount++;
  if (StreamingState.residentCount > StreamingState.peakResidentCount) {
    StreamingState.peakResidentCount = StreamingState.residentCount;
  }
  StreamingState.totalLoadCompletions++;
  StreamingState.lastLoadMs = StreamingChunk.loadFrameCount[eid] * 16.67;
  StreamingState.avgLoadMs += (StreamingState.lastLoadMs - StreamingState.avgLoadMs) * 0.15;

  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);
  return true;
}

/**
 * Transitions a chunk RESIDENT → EVICTING. Called by the eviction policy.
 */
export function markChunkEvict(eid, reasonCode) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.RESIDENT) return false;
  if ((StreamingChunk.flags[eid] & STREAM_FLAG.PINNED) !== 0) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.EVICTING;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.reasonCode[eid] = reasonCode !== undefined ? reasonCode : STREAM_REASON.DISTANCE;
  return true;
}

/**
 * Transitions a chunk EVICTING → UNLOADING → IDLE. Called when the
 * eviction pass starts.
 */
export function markChunkUnloading(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.EVICTING) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.UNLOADING;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  return true;
}

/**
 * Completes the unload. UNLOADING → IDLE. Clears residency flags and
 * decrements the resident counter.
 */
export function markChunkEvicted(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state !== STREAM_STATE.UNLOADING && state !== STREAM_STATE.RESIDENT) return false;

  const wasResident = state === STREAM_STATE.RESIDENT;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.IDLE;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.evictedAtFrame[eid] = StreamingState.frame;

  // Clear residency flags.
  StreamingResidency.mesh[eid] = 0;
  StreamingResidency.texture[eid] = 0;
  StreamingResidency.lights[eid] = 0;
  StreamingResidency.gi[eid] = 0;
  StreamingResidency.ao[eid] = 0;
  StreamingResidency.shadow[eid] = 0;
  StreamingResidency.audio[eid] = 0;
  StreamingResidency.mask[eid] = CHUNK_RESIDENCY.NONE;

  // Clear mesh / texture ids.
  StreamingMeshState.geometryId[eid] = -1;
  StreamingMeshState.materialId[eid] = -1;
  StreamingMeshState.proxyGeometryId[eid] = -1;
  StreamingTextureState.albedoId[eid] = -1;
  StreamingTextureState.normalId[eid] = -1;
  StreamingTextureState.roughnessId[eid] = -1;
  StreamingTextureState.emissionId[eid] = -1;

  // Clear light state.
  StreamingLightState.lightCount[eid] = 0;
  StreamingLightState.shadowCasters[eid] = 0;
  StreamingLightState.giProbeCount[eid] = 0;
  StreamingLightState.aoVolumeCount[eid] = 0;
  StreamingLightState.firstLightEid[eid] = -1;
  StreamingLightState.firstGIEid[eid] = -1;
  StreamingLightState.firstAOEid[eid] = -1;

  // Unbind worker if still bound.
  if (StreamingAsync.workerSlot[eid] >= 0) {
    const slot = StreamingAsync.workerSlot[eid];
    if (slot < 8) {
      StreamingWorkerRef.workerBusy[slot] = 0;
      StreamingWorkerRef.workerChunkEid[slot] = -1;
    }
    StreamingAsync.workerSlot[eid] = -1;
  }

  if (wasResident) {
    if (StreamingState.residentCount > 0) StreamingState.residentCount--;
  }

  StreamingState.totalEvictions++;
  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);
  return true;
}

/**
 * Transitions a chunk to FAILED. Called when a load errors out or times
 * out.
 */
export function markChunkFailed(eid, errorCode) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const state = StreamingChunk.state[eid];
  if (state === STREAM_STATE.RESIDENT) return false;

  StreamingChunk.prevState[eid] = state;
  StreamingChunk.state[eid] = STREAM_STATE.FAILED;
  StreamingChunk.lastStateFrame[eid] = StreamingState.frame;
  StreamingChunk.flags[eid] &= ~STREAM_FLAG.ASYNC_IN_FLIGHT;
  StreamingChunk.reasonCode[eid] = STREAM_REASON.FAILED;

  // Bump retry count.
  if (StreamingChunk.retryCount[eid] < 0xFF) {
    StreamingChunk.retryCount[eid]++;
  }

  // Unbind worker.
  if (StreamingAsync.workerSlot[eid] >= 0) {
    const slot = StreamingAsync.workerSlot[eid];
    if (slot < 8) {
      StreamingWorkerRef.workerBusy[slot] = 0;
      StreamingWorkerRef.workerChunkEid[slot] = -1;
      StreamingWorkerRef.workerFailCount[slot]++;
    }
    StreamingAsync.workerSlot[eid] = -1;
  }

  StreamingAsync.asyncErrorCode[eid] = errorCode !== undefined ? (errorCode | 0) & 0xFFFF : 0;

  StreamingState.totalLoadFailures++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. CHUNK RESIDENCY HELPERS                                         */
/* ------------------------------------------------------------------ */

export function setChunkMeshResident(eid, geometryId, materialId, triangleCount, vertexCount) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.mesh[eid] = 1;
  StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.MESH;
  StreamingMeshState.geometryId[eid] = geometryId | 0;
  StreamingMeshState.materialId[eid] = materialId | 0;
  if (triangleCount !== undefined) StreamingMeshState.triangleCount[eid] = (triangleCount | 0) >>> 0;
  if (vertexCount !== undefined) StreamingMeshState.vertexCount[eid] = (vertexCount | 0) >>> 0;
  return true;
}

export function setChunkProxyResident(eid, proxyGeometryId) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingMeshState.proxyGeometryId[eid] = proxyGeometryId | 0;
  return true;
}

export function setChunkTextureResident(eid, albedoId, normalId, roughnessId, emissionId, textureBytes) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.texture[eid] = 1;
  StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.TEXTURE;
  if (albedoId !== undefined) StreamingTextureState.albedoId[eid] = albedoId | 0;
  if (normalId !== undefined) StreamingTextureState.normalId[eid] = normalId | 0;
  if (roughnessId !== undefined) StreamingTextureState.roughnessId[eid] = roughnessId | 0;
  if (emissionId !== undefined) StreamingTextureState.emissionId[eid] = emissionId | 0;
  if (textureBytes !== undefined) StreamingTextureState.textureBytes[eid] = (textureBytes | 0) >>> 0;
  return true;
}

export function setChunkLightsResident(eid, lightCount, shadowCasters, firstLightEid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.lights[eid] = 1;
  StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.LIGHTS;
  StreamingLightState.lightCount[eid] = (lightCount | 0) & 0xFFFF;
  StreamingLightState.shadowCasters[eid] = (shadowCasters | 0) & 0xFFFF;
  if (firstLightEid !== undefined) StreamingLightState.firstLightEid[eid] = firstLightEid | 0;
  return true;
}

export function setChunkGIResident(eid, probeCount, firstGIEid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.gi[eid] = 1;
  StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.GI;
  StreamingLightState.giProbeCount[eid] = (probeCount | 0) & 0xFFFF;
  if (firstGIEid !== undefined) StreamingLightState.firstGIEid[eid] = firstGIEid | 0;
  return true;
}

export function setChunkAOResident(eid, volumeCount, firstAOEid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.ao[eid] = 1;
  StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.AO;
  StreamingLightState.aoVolumeCount[eid] = (volumeCount | 0) & 0xFFFF;
  if (firstAOEid !== undefined) StreamingLightState.firstAOEid[eid] = firstAOEid | 0;
  return true;
}

export function setChunkShadowResident(eid, resident) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.shadow[eid] = resident ? 1 : 0;
  if (resident) StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.SHADOW;
  else StreamingResidency.mask[eid] &= ~CHUNK_RESIDENCY.SHADOW;
  return true;
}

export function setChunkAudioResident(eid, resident) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingResidency.audio[eid] = resident ? 1 : 0;
  if (resident) StreamingResidency.mask[eid] |= CHUNK_RESIDENCY.AUDIO;
  else StreamingResidency.mask[eid] &= ~CHUNK_RESIDENCY.AUDIO;
  return true;
}

/**
 * Returns true if the chunk has all required sub-resources resident.
 */
export function isChunkFullyResident(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const mask = StreamingResidency.mask[eid];
  return (mask & (CHUNK_RESIDENCY.MESH | CHUNK_RESIDENCY.LIGHTS | CHUNK_RESIDENCY.GI)) ===
         (CHUNK_RESIDENCY.MESH | CHUNK_RESIDENCY.LIGHTS | CHUNK_RESIDENCY.GI);
}

/* ------------------------------------------------------------------ */
/* 7. PRIORITY EVALUATION                                             */
/* ------------------------------------------------------------------ */

/**
 * Recomputes the chunk's distance to the camera and its priority.
 *
 * Priority formula:
 *   priority = bias * (radius / max(dist, epsilon))
 *            * (1 + facing_bonus)
 *            * pinned_bonus
 *
 * Larger priority = more urgent to load.
 */
export function evaluateChunkPriority(eid, cameraX, cameraY, cameraZ, cameraDirX, cameraDirY, cameraDirZ) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;

  const cx = StreamingChunk.centerX[eid];
  const cy = StreamingChunk.centerY[eid];
  const cz = StreamingChunk.centerZ[eid];

  const dx = cx - cameraX;
  const dy = cy - cameraY;
  const dz = cz - cameraZ;
  const dSq = dx * dx + dy * dy + dz * dz;
  const dist = Math.sqrt(dSq);

  StreamingChunk.distance[eid] = dist;
  StreamingChunk.distanceSq[eid] = dSq;

  const radius = StreamingChunk.radius[eid] || DEFAULT_CHUNK_RADIUS;
  const invDist = 1.0 / Math.max(dist, 1.0);

  // Facing bonus — if the chunk is in front of the camera, boost.
  let facingBonus = 0;
  if (dist > 1e-3) {
    const nx = dx * invDist;
    const ny = dy * invDist;
    const nz = dz * invDist;
    const facing = nx * cameraDirX + ny * cameraDirY + nz * cameraDirZ;
    facingBonus = Math.max(0, facing) * 0.5;
    if (facing < -0.25) {
      StreamingChunk.flags[eid] |= STREAM_FLAG.BEHIND_CAMERA;
    } else {
      StreamingChunk.flags[eid] &= ~STREAM_FLAG.BEHIND_CAMERA;
    }
  }

  const pinnedBonus = (StreamingChunk.flags[eid] & STREAM_FLAG.PINNED) !== 0 ? 10.0 : 1.0;
  const bias = StreamingChunk.bias[eid] || 1.0;

  const priority = bias * radius * invDist * (1 + facingBonus) * pinnedBonus;

  StreamingChunk.priority[eid] = priority;
  StreamingPriority.sortKey[eid] = priority;
  StreamingPriority.dirty[eid] = 0;

  // Classify the priority bucket.
  if (priority > 8.0)      StreamingPriority.bucket[eid] = 0;   // critical
  else if (priority > 4.0) StreamingPriority.bucket[eid] = 1;   // high
  else if (priority > 1.0) StreamingPriority.bucket[eid] = 2;   // normal
  else if (priority > 0.25)StreamingPriority.bucket[eid] = 3;   // low
  else                     StreamingPriority.bucket[eid] = 4;   // idle

  return priority;
}

/**
 * Returns the chunk's current priority.
 */
export function getChunkPriority(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  return StreamingChunk.priority[eid];
}

/**
 * Sets a per-chunk priority bias.
 */
export function setChunkBias(eid, bias) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingChunk.bias[eid] = Math.max(0, Math.min(10, Number(bias) || 1.0));
  StreamingPriority.dirty[eid] = 1;
  return true;
}

/**
 * Pins or unpins a chunk so it is never evicted.
 */
export function setChunkPinned(eid, pinned) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (pinned) StreamingChunk.flags[eid] |= STREAM_FLAG.PINNED;
  else StreamingChunk.flags[eid] &= ~STREAM_FLAG.PINNED;
  return true;
}

export function isChunkPinned(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return (StreamingChunk.flags[eid] & STREAM_FLAG.PINNED) !== 0;
}

/* ------------------------------------------------------------------ */
/* 8. CHUNK LOD LEVELS                                                */
/* ------------------------------------------------------------------ */

/**
 * Sets the chunk's LOD level from distance and radius.
 */
export function setChunkLOD(eid, lod) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (lod < 0 || lod >= CHUNK_LOD.COUNT) return false;
  StreamingChunk.lodTarget[eid] = lod;
  return true;
}

/**
 * Evaluates the chunk's LOD level from its current distance and radius.
 */
export function evaluateChunkLOD(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return CHUNK_LOD.FULL;

  const dist = StreamingChunk.distance[eid];
  const radius = StreamingChunk.radius[eid] || DEFAULT_CHUNK_RADIUS;

  // LOD 0 (full):    dist < radius * 2
  // LOD 1 (half):    dist < radius * 4
  // LOD 2 (quarter): dist < radius * 8
  // LOD 3 (proxy):   otherwise
  let lod;
  if (dist < radius * 2) lod = CHUNK_LOD.FULL;
  else if (dist < radius * 4) lod = CHUNK_LOD.HALF;
  else if (dist < radius * 8) lod = CHUNK_LOD.QUARTER;
  else lod = CHUNK_LOD.PROXY;

  StreamingChunk.lodTarget[eid] = lod;

  // Commit only if the delta is one level (hysteresis).
  const current = StreamingChunk.lod[eid];
  if (Math.abs(lod - current) <= 1) {
    StreamingChunk.lod[eid] = lod;
  }

  return StreamingChunk.lod[eid];
}

/* ------------------------------------------------------------------ */
/* 9. ASYNC WORKER BINDING                                            */
/* ------------------------------------------------------------------ */

/**
 * Binds a worker slot to a chunk. Returns true on success.
 */
export function setChunkWorkerRef(eid, workerSlot, stage) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (workerSlot < 0 || workerSlot >= 8) return false;

  // Release the previous binding.
  if (StreamingAsync.workerSlot[eid] >= 0) {
    const prev = StreamingAsync.workerSlot[eid];
    if (prev < 8) {
      StreamingWorkerRef.workerBusy[prev] = 0;
      StreamingWorkerRef.workerChunkEid[prev] = -1;
    }
  }

  // Bind the new slot.
  StreamingAsync.workerSlot[eid] = workerSlot;
  StreamingAsync.asyncStage[eid] = stage !== undefined ? (stage | 0) & 0xFF : 1;
  StreamingAsync.asyncStartFrame[eid] = StreamingState.frame;
  StreamingAsync.asyncCancelFlag[eid] = 0;

  StreamingWorkerRef.workerBusy[workerSlot] = 1;
  StreamingWorkerRef.workerChunkEid[workerSlot] = eid;
  StreamingWorkerRef.workerStartFrame[workerSlot] = StreamingState.frame;
  StreamingWorkerRef.workerJobCount[workerSlot]++;

  StreamingChunk.flags[eid] |= STREAM_FLAG.ASYNC_IN_FLIGHT;

  return true;
}

/**
 * Unbinds the worker slot from a chunk.
 */
export function clearChunkWorkerRef(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const slot = StreamingAsync.workerSlot[eid];
  if (slot < 0 || slot >= 8) return false;

  StreamingWorkerRef.workerBusy[slot] = 0;
  StreamingWorkerRef.workerChunkEid[slot] = -1;
  StreamingAsync.workerSlot[eid] = -1;
  StreamingChunk.flags[eid] &= ~STREAM_FLAG.ASYNC_IN_FLIGHT;
  return true;
}

/**
 * Requests cancellation of an in-flight async load.
 */
export function cancelChunkAsync(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (StreamingAsync.workerSlot[eid] < 0) return false;
  StreamingAsync.asyncCancelFlag[eid] = 1;
  return true;
}

/**
 * Updates the async progress of a chunk.
 */
export function setChunkAsyncProgress(eid, progress) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  StreamingAsync.asyncProgress[eid] = Math.max(0, Math.min(1, Number(progress) || 0));
  return true;
}

/**
 * Finds a free worker slot. Returns -1 if none available.
 */
export function findFreeWorkerSlot() {
  for (let i = 0; i < 8; i++) {
    if (StreamingWorkerRef.workerBusy[i] === 0) return i;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/* 10. BUDGET ACCOUNTING                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets the global streaming budget cap.
 */
export function setStreamingBudgetCap(cap) {
  StreamingBudget.budgetCap[0] = Math.max(0, Number(cap) || 0);
  StreamingState.budgetCap = StreamingBudget.budgetCap[0];
  return true;
}

export function getStreamingBudgetCap() {
  return StreamingBudget.budgetCap[0];
}

/**
 * Updates the aggregate resident cost and bytes. Called once per frame.
 */
export function updateStreamingBudget() {
  let totalCost = 0;
  let totalBytes = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (StreamingChunk.state[eid] !== STREAM_STATE.RESIDENT) continue;
    totalCost += StreamingBudget.entityCostEma[eid];
    totalBytes += StreamingBudget.bytesEstimate[eid];
  }

  StreamingBudget.totalResidentCost[0] = totalCost;
  StreamingBudget.totalResidentBytes[0] = totalBytes >>> 0;
  StreamingState.budgetUsed = totalCost;

  const cap = StreamingBudget.budgetCap[0];
  if (cap > 0 && totalCost > cap) {
    StreamingBudget.budgetExceeded[0] = 1;
    StreamingState.budgetExceeded = 1;
  } else {
    StreamingBudget.budgetExceeded[0] = 0;
    StreamingState.budgetExceeded = 0;
  }

  return totalCost;
}

/**
 * Returns true if the streaming budget is currently exceeded.
 */
export function isStreamingBudgetExceeded() {
  return StreamingBudget.budgetExceeded[0] === 1;
}

/* ------------------------------------------------------------------ */
/* 11. LOAD / EVICTION ADMISSION                                      */
/* ------------------------------------------------------------------ */

/**
 * Admits up to `maxLoads` chunk loads from the REQUESTED queue, ordered
 * by priority. Transitions them to LOADING.
 *
 * Returns the number of loads admitted.
 */
export function admitChunkLoads(maxLoads) {
  const limit = maxLoads !== undefined ? (maxLoads | 0) : MAX_LOADS_PER_FRAME;
  if (limit <= 0) return 0;

  // Collect candidates: chunks in REQUESTED state.
  let candidateCount = 0;
  const cap = _priorityCandidates.length;
  for (let eid = 0; eid < MAX_ENTITIES && candidateCount < cap; eid++) {
    if (StreamingChunk.state[eid] !== STREAM_STATE.REQUESTED) continue;
    _priorityCandidates[candidateCount++] = eid;
  }
  if (candidateCount === 0) return 0;

  // Sort candidates by priority descending (in-place bubble to keep
  // allocations zero; candidateCount is small).
  for (let i = 1; i < candidateCount; i++) {
    const eid = _priorityCandidates[i];
    const priority = StreamingChunk.priority[eid];
    let j = i - 1;
    while (j >= 0 && StreamingChunk.priority[_priorityCandidates[j]] < priority) {
      _priorityCandidates[j + 1] = _priorityCandidates[j];
      j--;
    }
    _priorityCandidates[j + 1] = eid;
  }

  // Admit the top `limit` candidates.
  let admitted = 0;
  let cost = 0;
  const budgetCap = StreamingBudget.budgetCap[0];

  for (let i = 0; i < candidateCount && admitted < limit; i++) {
    const eid = _priorityCandidates[i];
    const loadCost = StreamingBudget.loadCostEstimate[eid];

    // Budget check — if a budget cap is set, stop admitting when it is
    // about to be exceeded.
    if (budgetCap > 0 && cost + loadCost > budgetCap) {
      StreamingState.totalAdmissionRejects++;
      break;
    }

    markChunkLoading(eid);
    cost += loadCost;
    admitted++;
  }

  StreamingState.totalLoadsPerFrame = admitted;
  StreamingStats.loadsThisFrame[0] = admitted;

  return admitted;
}

/**
 * Admits up to `maxEvictions` chunk evictions from the RESIDENT pool,
 * ordered by lowest priority (highest distance). Transitions them to
 * EVICTING.
 *
 * Returns the number of evictions admitted.
 */
export function admitChunkEvictions(maxEvictions) {
  const limit = maxEvictions !== undefined ? (maxEvictions | 0) : MAX_EVICTIONS_PER_FRAME;
  if (limit <= 0) return 0;

  // Collect candidates: chunks in RESIDENT state that are not pinned
  // and are not fully resident for the current camera.
  let candidateCount = 0;
  const cap = _evictCandidates.length;
  for (let eid = 0; eid < MAX_ENTITIES && candidateCount < cap; eid++) {
    if (StreamingChunk.state[eid] !== STREAM_STATE.RESIDENT) continue;
    if ((StreamingChunk.flags[eid] & STREAM_FLAG.PINNED) !== 0) continue;
    _evictCandidates[candidateCount++] = eid;
  }
  if (candidateCount === 0) return 0;

  // Sort candidates by priority ascending (lowest priority first).
  for (let i = 1; i < candidateCount; i++) {
    const eid = _evictCandidates[i];
    const priority = StreamingChunk.priority[eid];
    let j = i - 1;
    while (j >= 0 && StreamingChunk.priority[_evictCandidates[j]] > priority) {
      _evictCandidates[j + 1] = _evictCandidates[j];
      j--;
    }
    _evictCandidates[j + 1] = eid;
  }

  let admitted = 0;
  for (let i = 0; i < candidateCount && admitted < limit; i++) {
    const eid = _evictCandidates[i];
    if (markChunkEvict(eid, STREAM_REASON.PRIORITY)) {
      admitted++;
    }
  }

  StreamingState.totalEvictionsPerFrame = admitted;
  StreamingStats.evictionsThisFrame[0] = admitted;

  return admitted;
}

/* ------------------------------------------------------------------ */
/* 12. LOAD PROGRESS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Advances every LOADING chunk's internal frame counter and checks for
 * timeouts. Called once per frame.
 *
 * Returns the number of chunks that timed out.
 */
export function advanceChunkLoads() {
  let timedOut = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (StreamingChunk.state[eid] !== STREAM_STATE.LOADING) continue;

    StreamingChunk.loadFrameCount[eid]++;

    if (StreamingChunk.loadFrameCount[eid] >= StreamingChunk.loadTimeoutFrames[eid]) {
      // Timeout.
      if (StreamingChunk.retryCount[eid] < StreamingChunk.maxRetries[eid]) {
        // Retry.
        StreamingChunk.prevState[eid] = STREAM_STATE.LOADING;
        StreamingChunk.state[eid] = STREAM_STATE.REQUESTED;
        StreamingChunk.loadFrameCount[eid] = 0;
        StreamingChunk.flags[eid] &= ~STREAM_FLAG.ASYNC_IN_FLIGHT;
        StreamingState.totalTimeoutFails++;
        // Unbind worker.
        if (StreamingAsync.workerSlot[eid] >= 0) {
          clearChunkWorkerRef(eid);
        }
      } else {
        markChunkFailed(eid, 1);
        timedOut++;
      }
    }
  }

  return timedOut;
}

/**
 * Advances every EVICTING chunk to UNLOADING and then IDLE. Called once
 * per frame.
 *
 * Returns the number of chunks fully evicted.
 */
export function advanceChunkEvictions() {
  let completed = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const state = StreamingChunk.state[eid];
    if (state === STREAM_STATE.EVICTING) {
      markChunkUnloading(eid);
    } else if (state === STREAM_STATE.UNLOADING) {
      if (markChunkEvicted(eid)) completed++;
    }
  }

  return completed;
}

/* ------------------------------------------------------------------ */
/* 13. QUERY HELPERS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Iterates every chunk in the given state. Allocation-free.
 */
export function forEachChunkInState(state, fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (StreamingChunk.state[eid] !== state) continue;
    fn.call(ctx, eid);
    n++;
  }
  return n;
}

/**
 * Iterates every chunk regardless of state. Allocation-free.
 */
export function forEachChunk(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if ((StreamingChunk.flags[eid] & STREAM_FLAG.ENABLED) === 0) continue;
    fn.call(ctx, eid);
    n++;
  }
  return n;
}

/**
 * Collects every chunk in the given state into a caller-provided array.
 */
export function collectChunksInState(state, outArray, outOffset) {
  if (!outArray) return 0;
  const offset = outOffset !== undefined ? outOffset : 0;
  const cap = outArray.length - offset;
  let write = 0;
  for (let eid = 0; eid < MAX_ENTITIES && write < cap; eid++) {
    if (StreamingChunk.state[eid] !== state) continue;
    outArray[offset + write++] = eid;
  }
  return write;
}

/**
 * Returns the number of chunks currently resident.
 */
export function getResidentChunkCount() {
  return StreamingState.residentCount;
}

/**
 * Returns the number of chunks currently loading.
 */
export function getLoadingChunkCount() {
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (StreamingChunk.state[eid] === STREAM_STATE.LOADING) n++;
  }
  return n;
}

/**
 * Returns the number of chunks currently requested.
 */
export function getRequestedChunkCount() {
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (StreamingChunk.state[eid] === STREAM_STATE.REQUESTED) n++;
  }
  return n;
}

/**
 * Returns the number of chunks currently evicting or unloading.
 */
export function getEvictingChunkCount() {
  let n = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const s = StreamingChunk.state[eid];
    if (s === STREAM_STATE.EVICTING || s === STREAM_STATE.UNLOADING) n++;
  }
  return n;
}

/**
 * Returns true if the given entity is a streaming chunk.
 */
export function isStreamingChunk(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return (StreamingChunk.flags[eid] & STREAM_FLAG.ENABLED) !== 0;
}

/**
 * Returns the chunk's current state.
 */
export function getChunkState(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return STREAM_STATE.IDLE;
  return StreamingChunk.state[eid];
}

/* ------------------------------------------------------------------ */
/* 14. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the streaming system frame counter.
 */
export function tickStreaming(frameNumber) {
  if (typeof frameNumber === 'number') StreamingState.frame = frameNumber;
  else StreamingState.frame++;
}

/**
 * Full per-frame streaming pipeline:
 *   1. tickStreaming(frame)
 *   2. update priorities (caller must supply camera)
 *   3. admitChunkLoads
 *   4. admitChunkEvictions
 *   5. advanceChunkLoads
 *   6. advanceChunkEvictions
 *   7. updateStreamingBudget
 *
 * Returns a summary object.
 */
export function tickStreamingSystem(frameNumber, cameraX, cameraY, cameraZ, cameraDirX, cameraDirY, cameraDirZ, options) {
  const t0 = _now();

  tickStreaming(frameNumber);

  // Update priorities for every registered chunk.
  const updatePriorities = !options || options.updatePriorities !== false;
  if (updatePriorities) {
    for (let eid = 0; eid < MAX_ENTITIES; eid++) {
      if ((StreamingChunk.flags[eid] & STREAM_FLAG.ENABLED) === 0) continue;
      evaluateChunkPriority(eid, cameraX, cameraY, cameraZ, cameraDirX, cameraDirY, cameraDirZ);
      evaluateChunkLOD(eid);
    }
  }

  const loadsAdmitted = admitChunkLoads(options && options.maxLoads);
  const evictionsAdmitted = admitChunkEvictions(options && options.maxEvictions);
  const loadTimeouts = advanceChunkLoads();
  const evictionsCompleted = advanceChunkEvictions();
  const totalCost = updateStreamingBudget();

  const t1 = _now();
  const cost = t1 - t0;
  StreamingState.lastTickMs = cost;
  StreamingState.avgTickMs += (cost - StreamingState.avgTickMs) * 0.15;

  // Update stats.
  StreamingStats.residentCount[0] = StreamingState.residentCount;
  StreamingStats.loadingCount[0] = getLoadingChunkCount();
  StreamingStats.requestedCount[0] = getRequestedChunkCount();
  StreamingStats.evictingCount[0] = getEvictingChunkCount();
  StreamingStats.failedCount[0] = 0;
  StreamingStats.budgetExceeded[0] = StreamingState.budgetExceeded;

  return {
    frame: StreamingState.frame,
    loadsAdmitted,
    evictionsAdmitted,
    loadTimeouts,
    evictionsCompleted,
    totalCost,
    residentCount: StreamingState.residentCount,
    budgetExceeded: StreamingState.budgetExceeded === 1,
    cost,
  };
}

/* ------------------------------------------------------------------ */
/* 15. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerStreamingComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'StreamingChunk',        component: StreamingChunk,        category: 7, subsystem: 10, dependencies: [] },
    { name: 'StreamingResidency',    component: StreamingResidency,    category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingBudget',       component: StreamingBudget,       category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingPriority',     component: StreamingPriority,     category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingMeshState',    component: StreamingMeshState,    category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingTextureState', component: StreamingTextureState, category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingLightState',   component: StreamingLightState,   category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingAsync',        component: StreamingAsync,        category: 7, subsystem: 10, dependencies: ['StreamingChunk'] },
    { name: 'StreamingWorkerRef',    component: StreamingWorkerRef,    category: 7, subsystem: 10, dependencies: [] },
    { name: 'StreamingStats',        component: StreamingStats,        category: 7, subsystem: 10, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 16. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getChunkStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  return {
    entity:           eid,
    state:            STREAM_STATE_NAME[StreamingChunk.state[eid]] || 'idle',
    prevState:        STREAM_STATE_NAME[StreamingChunk.prevState[eid]] || 'idle',
    flags:            StreamingChunk.flags[eid],
    lod:              StreamingChunk.lod[eid],
    lodTarget:        StreamingChunk.lodTarget[eid],
    gridX:            StreamingChunk.gridX[eid],
    gridZ:            StreamingChunk.gridZ[eid],
    center:           [StreamingChunk.centerX[eid], StreamingChunk.centerY[eid], StreamingChunk.centerZ[eid]],
    radius:           StreamingChunk.radius[eid],
    distance:         StreamingChunk.distance[eid],
    priority:         StreamingChunk.priority[eid],
    bias:             StreamingChunk.bias[eid],
    bucket:           StreamingPriority.bucket[eid],
    lastStateFrame:   StreamingChunk.lastStateFrame[eid],
    requestedAtFrame: StreamingChunk.requestedAtFrame[eid],
    residentAtFrame:  StreamingChunk.residentAtFrame[eid],
    loadFrameCount:   StreamingChunk.loadFrameCount[eid],
    loadTimeout:      StreamingChunk.loadTimeoutFrames[eid],
    retryCount:       StreamingChunk.retryCount[eid],
    maxRetries:       StreamingChunk.maxRetries[eid],
    residencyMask:    StreamingResidency.mask[eid],
    isFullyResident:  isChunkFullyResident(eid),
    isPinned:         isChunkPinned(eid),
    mesh:             StreamingMeshState.geometryId[eid],
    texture:          StreamingTextureState.albedoId[eid],
    lightCount:       StreamingLightState.lightCount[eid],
    giProbeCount:     StreamingLightState.giProbeCount[eid],
    aoVolumeCount:    StreamingLightState.aoVolumeCount[eid],
    workerSlot:       StreamingAsync.workerSlot[eid],
    asyncProgress:    StreamingAsync.asyncProgress[eid],
    budgetCost:       StreamingBudget.entityCostEma[eid],
    bytesEstimate:    StreamingBudget.bytesEstimate[eid],
  };
}

export function getStreamingSystemReport() {
  const histogram = [];
  for (let i = 0; i < STREAM_STATE.COUNT; i++) {
    let count = 0;
    for (let eid = 0; eid < MAX_ENTITIES; eid++) {
      if (StreamingChunk.state[eid] === i) count++;
    }
    histogram.push({ state: STREAM_STATE_NAME[i], count });
  }

  return {
    frame:                  StreamingState.frame,
    budgetCap:              StreamingBudget.budgetCap[0],
    budgetUsed:             StreamingState.budgetUsed,
    budgetExceeded:         StreamingState.budgetExceeded === 1,
    residentCount:          StreamingState.residentCount,
    peakResidentCount:      StreamingState.peakResidentCount,
    totalRegistrations:     StreamingState.totalRegistrations,
    totalLoadRequests:      StreamingState.totalLoadRequests,
    totalLoadCompletions:   StreamingState.totalLoadCompletions,
    totalLoadFailures:      StreamingState.totalLoadFailures,
    totalEvictions:         StreamingState.totalEvictions,
    totalAdmissionRejects:  StreamingState.totalAdmissionRejects,
    totalTimeoutFails:      StreamingState.totalTimeoutFails,
    loadsThisFrame:         StreamingState.totalLoadsPerFrame,
    evictionsThisFrame:     StreamingState.totalEvictionsPerFrame,
    lastTickMs:             StreamingState.lastTickMs,
    avgTickMs:              StreamingState.avgTickMs,
    lastLoadMs:             StreamingState.lastLoadMs,
    avgLoadMs:              StreamingState.avgLoadMs,
    totalResidentCost:      StreamingBudget.totalResidentCost[0],
    totalResidentBytes:     StreamingBudget.totalResidentBytes[0],
    histogram,
    perfTier:               PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 17. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every streaming structure and resets counters.
 */
export function resetStreamingState() {
  StreamingChunk.state.fill(STREAM_STATE.IDLE);
  StreamingChunk.prevState.fill(STREAM_STATE.IDLE);
  StreamingChunk.flags.fill(0);
  StreamingChunk.lod.fill(CHUNK_LOD.FULL);
  StreamingChunk.lodTarget.fill(CHUNK_LOD.FULL);
  StreamingChunk.gridX.fill(0); StreamingChunk.gridZ.fill(0);
  StreamingChunk.chunkSize.fill(0);
  StreamingChunk.centerX.fill(0); StreamingChunk.centerY.fill(0); StreamingChunk.centerZ.fill(0);
  StreamingChunk.radius.fill(0);
  StreamingChunk.distance.fill(Infinity);
  StreamingChunk.distanceSq.fill(Infinity);
  StreamingChunk.priority.fill(0);
  StreamingChunk.bias.fill(1.0);
  StreamingChunk.lastStateFrame.fill(0);
  StreamingChunk.lastAccessFrame.fill(0);
  StreamingChunk.requestedAtFrame.fill(0);
  StreamingChunk.residentAtFrame.fill(0);
  StreamingChunk.evictedAtFrame.fill(0);
  StreamingChunk.loadFrameCount.fill(0);
  StreamingChunk.loadTimeoutFrames.fill(LOAD_TIMEOUT_FRAMES);
  StreamingChunk.retryCount.fill(0);
  StreamingChunk.maxRetries.fill(3);
  StreamingChunk.seed.fill(0);
  StreamingChunk.biomeId.fill(0);
  StreamingChunk.variant.fill(0);
  StreamingChunk.reasonCode.fill(STREAM_REASON.NONE);
  StreamingChunk.lastFrameLoaded.fill(0);

  StreamingResidency.mesh.fill(0);
  StreamingResidency.texture.fill(0);
  StreamingResidency.lights.fill(0);
  StreamingResidency.gi.fill(0);
  StreamingResidency.ao.fill(0);
  StreamingResidency.shadow.fill(0);
  StreamingResidency.audio.fill(0);
  StreamingResidency.mask.fill(0);

  StreamingBudget.entityCost.fill(0);
  StreamingBudget.entityCostEma.fill(0);
  StreamingBudget.loadCostEstimate.fill(0);
  StreamingBudget.evictCostEstimate.fill(0);
  StreamingBudget.bytesEstimate.fill(0);
  StreamingBudget.totalResidentCost[0] = 0;
  StreamingBudget.totalResidentBytes[0] = 0;
  StreamingBudget.budgetCap[0] = DEFAULT_BUDGET_CAP;
  StreamingBudget.budgetExceeded[0] = 0;

  StreamingPriority.bucket.fill(2);
  StreamingPriority.sortKey.fill(0);
  StreamingPriority.frameRank.fill(0);
  StreamingPriority.dirty.fill(0);

  StreamingMeshState.geometryId.fill(-1);
  StreamingMeshState.materialId.fill(-1);
  StreamingMeshState.proxyGeometryId.fill(-1);
  StreamingMeshState.triangleCount.fill(0);
  StreamingMeshState.vertexCount.fill(0);
  StreamingMeshState.instanceCount.fill(0);

  StreamingTextureState.albedoId.fill(-1);
  StreamingTextureState.normalId.fill(-1);
  StreamingTextureState.roughnessId.fill(-1);
  StreamingTextureState.emissionId.fill(-1);
  StreamingTextureState.textureBytes.fill(0);

  StreamingLightState.lightCount.fill(0);
  StreamingLightState.shadowCasters.fill(0);
  StreamingLightState.giProbeCount.fill(0);
  StreamingLightState.aoVolumeCount.fill(0);
  StreamingLightState.firstLightEid.fill(-1);
  StreamingLightState.firstGIEid.fill(-1);
  StreamingLightState.firstAOEid.fill(-1);

  StreamingAsync.workerSlot.fill(-1);
  StreamingAsync.asyncStartFrame.fill(0);
  StreamingAsync.asyncStage.fill(0);
  StreamingAsync.asyncProgress.fill(0);
  StreamingAsync.asyncCancelFlag.fill(0);
  StreamingAsync.asyncErrorCode.fill(0);

  StreamingWorkerRef.workerBusy.fill(0);
  StreamingWorkerRef.workerChunkEid.fill(-1);
  StreamingWorkerRef.workerStartFrame.fill(0);
  StreamingWorkerRef.workerJobCount.fill(0);
  StreamingWorkerRef.workerFailCount.fill(0);
  StreamingWorkerRef.workerTotalMs.fill(0);
  StreamingWorkerRef.workerLastMs.fill(0);

  StreamingStats.residentCount[0] = 0;
  StreamingStats.loadingCount[0] = 0;
  StreamingStats.requestedCount[0] = 0;
  StreamingStats.evictingCount[0] = 0;
  StreamingStats.failedCount[0] = 0;
  StreamingStats.stateHistogram.fill(0);
  StreamingStats.loadsThisFrame[0] = 0;
  StreamingStats.evictionsThisFrame[0] = 0;
  StreamingStats.budgetExceeded[0] = 0;

  StreamingState.frame = 0;
  StreamingState.budgetCap = DEFAULT_BUDGET_CAP;
  StreamingState.budgetUsed = 0;
  StreamingState.budgetExceeded = 0;
  StreamingState.residentCount = 0;
  StreamingState.peakResidentCount = 0;
  StreamingState.totalRegistrations = 0;
  StreamingState.totalLoadRequests = 0;
  StreamingState.totalLoadCompletions = 0;
  StreamingState.totalLoadFailures = 0;
  StreamingState.totalEvictions = 0;
  StreamingState.totalLoadsPerFrame = 0;
  StreamingState.totalEvictionsPerFrame = 0;
  StreamingState.totalAdmissionRejects = 0;
  StreamingState.totalTimeoutFails = 0;
  StreamingState.lastTickMs = 0;
  StreamingState.avgTickMs = 0;
  StreamingState.lastLoadMs = 0;
  StreamingState.avgLoadMs = 0;
}

/* ------------------------------------------------------------------ */
/* 18. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_STREAMING_CHUNKS,
  MAX_RESIDENT_CHUNKS,
  MAX_LOADS_PER_FRAME,
  MAX_EVICTIONS_PER_FRAME,
  LOAD_TIMEOUT_FRAMES,
  DEFAULT_BUDGET_CAP,
  DEFAULT_CHUNK_RADIUS,

  // Enums
  STREAM_STATE,
  STREAM_STATE_NAME,
  CHUNK_RESIDENCY,
  STREAM_FLAG,
  CHUNK_LOD,
  STREAM_REASON,

  // Components
  StreamingChunk,
  StreamingResidency,
  StreamingBudget,
  StreamingPriority,
  StreamingMeshState,
  StreamingTextureState,
  StreamingLightState,
  StreamingAsync,
  StreamingWorkerRef,
  StreamingStats,
  STREAMING_COMPONENTS,

  // Module state
  StreamingState,

  // Registration
  registerChunk,

  // Load transitions
  requestChunkLoad,
  cancelChunkLoad,
  markChunkLoading,
  markChunkResident,
  markChunkEvict,
  markChunkUnloading,
  markChunkEvicted,
  markChunkFailed,

  // Residency
  setChunkMeshResident,
  setChunkProxyResident,
  setChunkTextureResident,
  setChunkLightsResident,
  setChunkGIResident,
  setChunkAOResident,
  setChunkShadowResident,
  setChunkAudioResident,
  isChunkFullyResident,

  // Priority
  evaluateChunkPriority,
  getChunkPriority,
  setChunkBias,
  setChunkPinned,
  isChunkPinned,

  // Chunk LOD
  setChunkLOD,
  evaluateChunkLOD,

  // Async
  setChunkWorkerRef,
  clearChunkWorkerRef,
  cancelChunkAsync,
  setChunkAsyncProgress,
  findFreeWorkerSlot,

  // Budget
  setStreamingBudgetCap,
  getStreamingBudgetCap,
  updateStreamingBudget,
  isStreamingBudgetExceeded,

  // Admission
  admitChunkLoads,
  admitChunkEvictions,

  // Progress
  advanceChunkLoads,
  advanceChunkEvictions,

  // Query
  forEachChunkInState,
  forEachChunk,
  collectChunksInState,
  getResidentChunkCount,
  getLoadingChunkCount,
  getRequestedChunkCount,
  getEvictingChunkCount,
  isStreamingChunk,
  getChunkState,

  // Frame
  tickStreaming,
  tickStreamingSystem,

  // Diagnostics
  getChunkStats,
  getStreamingSystemReport,

  // Registration
  registerStreamingComponents,

  // Reset
  resetStreamingState,
};

export default _defaultExport;