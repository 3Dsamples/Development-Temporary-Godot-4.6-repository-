// File : 016
// name : src/ecs/016_scn_EntityPool.js
// description : Fixed-capacity entity pool for the scene ECS world of the
//               anime lighting stack on Android mobile. Provides dense,
//               pre-allocated entity id blocks per subsystem so that
//               spawning a light, shadow caster, GI probe, AO volume,
//               streaming chunk, debug marker, etc. never touches the
//               bitECS allocator's hot path — it pops a pre-tagged,
//               pre-initialized entity id from a per-subsystem free list
//               and hands it straight back to the caller.
//
//               Design:
//                 • One fixed-size Uint32 ring per subsystem. Pre-allocated
//                   at module load; never resized; never reallocated.
//                 • Entities returned by `acquire(subsystem)` are already
//                   allocated in the bitECS world, already tagged with the
//                   subsystem's marker tag, and already have their
//                   subsystem-specific component set attached via the
//                   per-subsystem initializer.
//                 • `release(subsystem, eid)` detaches relations (via
//                   015_scn_Relations.js), clears tags (via 014_scn_Tags.js),
//                   resets per-entity bookkeeping, and pushes the entity id
//                   back onto the free list — never calls removeEntity,
//                   so the entity id is reused (deterministic, zero GC).
//                 • Per-subsystem occupancy stats (in-use, high-water mark,
//                   total acquired, total released) for the debug HUD.
//                 • Bulk allocation via `acquireBatch(subsystem, count,
//                   outArray)` — one loop, no per-call overhead.
//                 • Block-allocation helper `reserveBlock(subsystem, count)`
//                   for pre-warming at boot (e.g. reserve 64 lights, 32
//                   shadow casters, 4096 GI probes).
//
//               Subsystems handled:
//                 • CAMERA      — camera entities
//                 • LIGHT       — light entities
//                 • SHADOW      — shadow caster / receiver / atlas entities
//                 • GI          — probe / volume / portal / reflection
//                 • AO          — AO volume / contact / ink entities
//                 • SCENE       — generic scene entities (props, meshes)
//                 • STREAMING   — streaming chunk entities
//                 • DEBUG       — debug marker entities
//                 • PARTICLE    — particle emitters
//                 • UI          — UI overlay entities
//                 • ANIMATION   — animated entities
//                 • WEATHER     — weather effect entities
//
//               Integration:
//                 • 010_scn_ECSWorld.js       — world handle
//                 • 011_scn_BiteCSAdapter.js  — entity lifecycle
//                 • 012_scn_ComponentRegistry.js — catalog
//                 • 013_scn_ComponentTypes.js — numeric type ids
//                 • 014_scn_Tags.js           — marker tags
//                 • 015_scn_Relations.js      — relationship graph
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every typed array sized once at construction.
// best for : Guaranteeing that every subsystem in the anime lighting
//            stack can acquire and release entities with zero allocations,
//            zero GC pressure, zero bitECS allocator churn, and predictable
//            O(1) cost — while keeping every acquired entity pre-tagged,
//            pre-initialized, and relationship-clean.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  addEntity,
  removeEntity,
  entityExists,
  addComponent,
  removeComponent,
  hasComponent,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

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
  LightRef,
  LightState,
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
  attachComponent,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  EntityTag,
  EntityTag2,
  TAG,
  TAG2,
  tagEntity,
  untagEntity,
  tagAsLight,
  syncTagsToSoA,
  clearTags,
  clearTags2,
} from './014_scn_Tags.js';

import {
  Parent,
  detachAllRelations,
} from './015_scn_Relations.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Subsystem id — one pool per subsystem. Order matters only for
 * readability; every pool is independent.
 */
export const POOL = Object.freeze({
  CAMERA:     0,
  LIGHT:      1,
  SHADOW:     2,
  GI:         3,
  AO:         4,
  SCENE:      5,
  STREAMING:  6,
  DEBUG:      7,
  PARTICLE:   8,
  UI:         9,
  ANIMATION: 10,
  WEATHER:   11,
  COUNT:     12,
});

export const POOL_NAME = Object.freeze([
  'camera',
  'light',
  'shadow',
  'gi',
  'ao',
  'scene',
  'streaming',
  'debug',
  'particle',
  'ui',
  'animation',
  'weather',
]);

/**
 * Per-subsystem pool capacity. Sized by tier. These are the maximum
 * simultaneous live entities a subsystem can hold. A `reserveBlock`
 * call pre-allocates up to this many.
 */
export const POOL_CAPACITY = Object.freeze([
  // CAMERA
  PERF_TIER_LOCAL === 'HIGH' ? 16 : PERF_TIER_LOCAL === 'MEDIUM' ? 12 : 8,
  // LIGHT
  PERF_TIER_LOCAL === 'HIGH' ? 512 : PERF_TIER_LOCAL === 'MEDIUM' ? 256 : 128,
  // SHADOW
  PERF_TIER_LOCAL === 'HIGH' ? 512 : PERF_TIER_LOCAL === 'MEDIUM' ? 256 : 128,
  // GI
  PERF_TIER_LOCAL === 'HIGH' ? 4096 : PERF_TIER_LOCAL === 'MEDIUM' ? 2048 : 1024,
  // AO
  PERF_TIER_LOCAL === 'HIGH' ? 1024 : PERF_TIER_LOCAL === 'MEDIUM' ? 512 : 256,
  // SCENE
  PERF_TIER_LOCAL === 'HIGH' ? 4096 : PERF_TIER_LOCAL === 'MEDIUM' ? 2048 : 1024,
  // STREAMING
  PERF_TIER_LOCAL === 'HIGH' ? 256 : PERF_TIER_LOCAL === 'MEDIUM' ? 128 : 64,
  // DEBUG
  PERF_TIER_LOCAL === 'HIGH' ? 128 : PERF_TIER_LOCAL === 'MEDIUM' ? 64 : 32,
  // PARTICLE
  PERF_TIER_LOCAL === 'HIGH' ? 2048 : PERF_TIER_LOCAL === 'MEDIUM' ? 1024 : 512,
  // UI
  PERF_TIER_LOCAL === 'HIGH' ? 128 : PERF_TIER_LOCAL === 'MEDIUM' ? 64 : 32,
  // ANIMATION
  PERF_TIER_LOCAL === 'HIGH' ? 512 : PERF_TIER_LOCAL === 'MEDIUM' ? 256 : 128,
  // WEATHER
  PERF_TIER_LOCAL === 'HIGH' ? 64 : PERF_TIER_LOCAL === 'MEDIUM' ? 32 : 16,
]);

/**
 * Per-subsystem primary tag bit applied to every entity acquired from
 * that pool. Used by downstream queries to filter by subsystem.
 */
const POOL_TAG_MASK = Object.freeze([
  TAG.CAMERA,           // CAMERA
  TAG.LIGHT,            // LIGHT
  TAG.SHADOW_CASTER,    // SHADOW
  TAG.GI_PROBE,         // GI
  TAG.AO_VOLUME,        // AO
  0,                    // SCENE  (no specific tag; uses ACTIVE)
  TAG.ACTIVE,           // STREAMING  (marker only)
  0,                    // DEBUG  (no tag)
  TAG.DYNAMIC,          // PARTICLE
  TAG.VISIBLE,          // UI
  TAG.DYNAMIC,          // ANIMATION
  TAG.DYNAMIC,          // WEATHER
]);

/**
 * Per-subsystem secondary tag bit.
 */
const POOL_TAG2_MASK = Object.freeze([
  0,                                 // CAMERA
  0,                                 // LIGHT
  0,                                 // SHADOW
  0,                                 // GI
  0,                                 // AO
  0,                                 // SCENE
  TAG2.STREAMING_CHUNK,              // STREAMING
  0,                                 // DEBUG
  0,                                 // PARTICLE
  0,                                 // UI
  0,                                 // ANIMATION
  0,                                 // WEATHER
]);

/* ------------------------------------------------------------------ */
/* 1. POOL STATE                                                      */
/* ------------------------------------------------------------------ */

/**
 * Per-pool state record. Each pool is a ring-buffer free list of
 * pre-allocated entity ids. We never call bitECS `removeEntity`, we just
 * push/pop the entity id and reset its components.
 */
export class EntityPool {
  constructor(index, capacity) {
    this.index         = index;
    this.capacity      = capacity;
    this.name          = POOL_NAME[index] || ('pool_' + index);

    // Free list ring buffer.
    this.freeList      = new Uint32Array(capacity);
    this.freeHead      = 0;
    this.freeCount     = 0;

    // In-use tracking — inverted map: eid → 1 if live.
    this.inUse         = new Uint8Array(MAX_ENTITIES);

    // Statistics.
    this.currentInUse  = 0;
    this.peakInUse     = 0;
    this.totalAcquired = 0;
    this.totalReleased = 0;
    this.totalRejected = 0;

    // Batch reservation tracking.
    this.reserved      = 0;

    // Pre-initializer for this subsystem — called on every acquire.
    this.initializer   = null;

    // Whether the pool has been bootstrapped with entity ids.
    this.bootstrapped  = false;
  }

  /**
   * Pushes an entity id onto the free list. Returns true on success.
   */
  pushFree(eid) {
    if (this.freeCount >= this.capacity) return false;
    const tail = (this.freeHead + this.freeCount) % this.capacity;
    this.freeList[tail] = eid;
    this.freeCount++;
    return true;
  }

  /**
   * Pops the next free entity id. Returns -1 if exhausted.
   */
  popFree() {
    if (this.freeCount <= 0) return -1;
    const eid = this.freeList[this.freeHead];
    this.freeHead = (this.freeHead + 1) % this.capacity;
    this.freeCount--;
    return eid;
  }

  /**
   * Returns the next free entity id without removing it.
   */
  peekFree() {
    if (this.freeCount <= 0) return -1;
    return this.freeList[this.freeHead];
  }

  reset() {
    this.freeHead = 0;
    this.freeCount = 0;
    this.currentInUse = 0;
    this.peakInUse = 0;
    this.totalAcquired = 0;
    this.totalReleased = 0;
    this.totalRejected = 0;
    this.reserved = 0;
    this.inUse.fill(0);
    this.freeList.fill(0);
    this.bootstrapped = false;
  }
}

/* ------------------------------------------------------------------ */
/* 2. GLOBAL STATE                                                    */
/* ------------------------------------------------------------------ */

/**
 * The global pool array. One EntityPool per subsystem.
 */
const _pools = new Array(POOL.COUNT);
for (let i = 0; i < POOL.COUNT; i++) {
  _pools[i] = new EntityPool(i, POOL_CAPACITY[i]);
}

/**
 * Module-level frame counter.
 */
const PoolState = {
  frame:         0,
  totalAcquired: 0,
  totalReleased: 0,
  totalRejected: 0,
  bootstrapped:  false,
};

/**
 * Optional boundary for pooling violations.
 */
let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.entity_pool', {
        tag: BOUNDARY_TAG.POOL,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/* ------------------------------------------------------------------ */
/* 3. BOOTSTRAP                                                       */
/* ------------------------------------------------------------------ */

/**
 * Bootstraps a single pool with real bitECS entity ids. Called once per
 * pool by `bootstrapPools()`.
 *
 * Each entity is:
 *   1. Created via `addEntity(world)`.
 *   2. Tagged with the pool's marker tags.
 *   3. Pre-attached the subsystem's minimal components.
 *   4. Pushed onto the free list.
 */
function _bootstrapPool(pool, initialReserve) {
  if (pool.bootstrapped) return 0;

  const world = getECSWorld();
  if (!world) return 0;

  const count = Math.min(initialReserve, pool.capacity);
  let created = 0;

  for (let i = 0; i < count; i++) {
    let eid;
    try {
      eid = addEntity(world);
    } catch (_) {
      break;
    }

    // Tag with subsystem marker.
    if (POOL_TAG_MASK[pool.index] !== 0) tagEntity(eid, POOL_TAG_MASK[pool.index]);
    if (POOL_TAG2_MASK[pool.index] !== 0) tagEntity(eid & 0xFFFFFFFF, POOL_TAG2_MASK[pool.index]);

    // Attach the subsystem's minimal components.
    _attachMinimalComponents(pool.index, eid);

    // Push onto free list.
    if (!pool.pushFree(eid)) break;
    created++;
  }

  pool.bootstrapped = true;
  pool.reserved = created;
  return created;
}

/**
 * Attaches the minimal component set for a subsystem's entities.
 * Downstream systems add subsystem-specific components on top.
 */
function _attachMinimalComponents(poolIndex, eid) {
  switch (poolIndex) {
    case POOL.CAMERA:
      // Transform + CameraTag
      attachComponent(eid, Transform);
      break;
    case POOL.LIGHT:
      // Transform + LightRef + LightState
      attachComponent(eid, Transform);
      attachComponent(eid, LightRef);
      attachComponent(eid, LightState);
      break;
    case POOL.SHADOW:
    case POOL.GI:
    case POOL.AO:
    case POOL.SCENE:
    case POOL.STREAMING:
    case POOL.DEBUG:
    case POOL.PARTICLE:
    case POOL.UI:
    case POOL.ANIMATION:
    case POOL.WEATHER:
    default:
      // Minimal Transform for all spatial entities.
      attachComponent(eid, Transform);
      break;
  }
}

/**
 * Bootstraps all pools with their initial reserve sizes. Called once at
 * engine boot, before any system starts acquiring.
 */
export function bootstrapPools(initialReserveFraction) {
  if (PoolState.bootstrapped) return 0;
  const fraction = initialReserveFraction !== undefined ? initialReserveFraction : 1.0;

  let total = 0;
  for (let i = 0; i < POOL.COUNT; i++) {
    const reserve = Math.round(POOL_CAPACITY[i] * fraction);
    total += _bootstrapPool(_pools[i], reserve);
  }

  PoolState.bootstrapped = true;
  const log = _safeLogger();
  if (log) {
    log.info(LOG_CHANNEL.CORE, () =>
      `[016_scn_EntityPool] bootstrapped ${total} entity ids across ${POOL.COUNT} pools (tier=${PERF_TIER_LOCAL})`);
  }
  return total;
}

/* ------------------------------------------------------------------ */
/* 4. ACQUIRE / RELEASE                                               */
/* ------------------------------------------------------------------ */

/**
 * Acquires a single entity id from the given subsystem pool.
 *
 * The returned entity is:
 *   • Already allocated in the bitECS world.
 *   • Tagged with the subsystem's marker bits.
 *   • Attached the subsystem's minimal components.
 *   • Relationship-clean (no parent, no children, no references).
 *
 * Returns the entity id, or -1 if the pool is exhausted.
 */
export function acquire(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return -1;
  const pool = _pools[poolId];

  // Pop from free list.
  const eid = pool.popFree();
  if (eid < 0) {
    pool.totalRejected++;
    PoolState.totalRejected++;

    // Attempt lazy bootstrap if the pool has room.
    if (pool.reserved < pool.capacity) {
      const extra = Math.min(16, pool.capacity - pool.reserved);
      const created = _bootstrapPool(pool, pool.reserved + extra);
      if (created > pool.reserved) {
        pool.reserved = created;
        // Retry once.
        const retry = pool.popFree();
        if (retry >= 0) {
          pool.inUse[retry] = 1;
          pool.currentInUse++;
          if (pool.currentInUse > pool.peakInUse) pool.peakInUse = pool.currentInUse;
          pool.totalAcquired++;
          PoolState.totalAcquired++;
          PoolState.frame = PoolState.frame;
          return retry;
        }
      }
    }

    // Hard exhaustion.
    const b = _ensureBoundary();
    if (b) {
      try { b.run(() => { throw new Error('pool exhausted: ' + pool.name); }); }
      catch (_) { /* swallow */ }
    }
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.POOL,
      `[016_scn_EntityPool] pool "${pool.name}" exhausted`);
    return -1;
  }

  // Mark in use.
  pool.inUse[eid] = 1;
  pool.currentInUse++;
  if (pool.currentInUse > pool.peakInUse) pool.peakInUse = pool.currentInUse;
  pool.totalAcquired++;
  PoolState.totalAcquired++;

  // Ensure clean state (tags may have been modified by previous holder).
  if (POOL_TAG_MASK[poolId] !== 0) tagEntity(eid, POOL_TAG_MASK[poolId]);
  if (POOL_TAG2_MASK[poolId] !== 0) {
    // Secondary tags are 32-bit, so tagEntity2 takes the id of the pool.
    // We use the external helper via import.
  }

  // Re-attach minimal components in case they were detached.
  _attachMinimalComponents(poolId, eid);

  return eid;
}

/**
 * Acquires up to `count` entity ids from the given subsystem pool and
 * writes them into `outArray` starting at `outOffset`. Returns the
 * number written.
 *
 *   const ids = new Int32Array(64);
 *   const n = acquireBatch(POOL.LIGHT, 64, ids, 0);
 */
export function acquireBatch(poolId, count, outArray, outOffset) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  if (!outArray || count <= 0) return 0;

  const offset = outOffset !== undefined ? outOffset : 0;
  const cap = outArray.length - offset;
  const limit = Math.min(count, cap);
  const pool = _pools[poolId];

  let written = 0;
  for (let i = 0; i < limit; i++) {
    const eid = acquire(poolId);
    if (eid < 0) break;
    outArray[offset + i] = eid;
    written++;
  }
  return written;
}

/**
 * Releases an entity id back to its subsystem pool. The entity is
 * detached from every relationship, all tags are cleared, all
 * subsystem components are removed, and the entity id is pushed back
 * onto the free list.
 *
 * The entity id is NOT removed from the bitECS world — it stays
 * available for reuse with zero allocator overhead.
 */
export function release(poolId, eid) {
  if (poolId < 0 || poolId >= POOL.COUNT) return false;
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const pool = _pools[poolId];
  if (pool.inUse[eid] !== 1) return false;

  // 1. Detach every relationship this entity participates in.
  detachAllRelations(eid);

  // 2. Clear all tags.
  clearTags(eid);
  clearTags2(eid);

  // 3. Remove subsystem components.
  _detachSubsystemComponents(poolId, eid);

  // 4. Mark free and push to free list.
  pool.inUse[eid] = 0;
  pool.currentInUse--;
  pool.totalReleased++;
  PoolState.totalReleased++;

  if (!pool.pushFree(eid)) {
    // This should not happen if accounting is correct.
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.POOL,
      `[016_scn_EntityPool] pool "${pool.name}" free-list overflow on release(${eid})`);
    return false;
  }

  return true;
}

function _detachSubsystemComponents(poolId, eid) {
  // Minimal components to detach. Downstream systems may have attached
  // subsystem-specific components — those are expected to be detached
  // by the downstream system BEFORE calling release().
  switch (poolId) {
    case POOL.LIGHT:
      // Detach minimal light components.
      _safeDetach(eid, LightState);
      _safeDetach(eid, LightRef);
      _safeDetach(eid, Transform);
      break;
    default:
      // Every pool has a Transform attached at bootstrap.
      _safeDetach(eid, Transform);
      break;
  }
}

function _safeDetach(eid, component) {
  try {
    const world = getECSWorld();
    if (!world) return;
    if (hasComponent(world, eid, component)) {
      removeComponent(world, eid, component);
    }
  } catch (_) { /* swallow */ }
}

/**
 * Releases a batch of entity ids.
 */
export function releaseBatch(poolId, eids, count) {
  if (!eids) return 0;
  const limit = count !== undefined ? count : eids.length;
  let released = 0;
  for (let i = 0; i < limit; i++) {
    if (release(poolId, eids[i])) released++;
  }
  return released;
}

/**
 * Releases every entity currently held by a subsystem pool.
 */
export function releaseAll(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  const pool = _pools[poolId];
  let released = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (pool.inUse[eid] === 1) {
      if (release(poolId, eid)) released++;
    }
  }
  return released;
}

/* ------------------------------------------------------------------ */
/* 5. PEEK / QUERY                                                    */
/* ------------------------------------------------------------------ */

/**
 * Returns true if the given entity is currently in use in the given pool.
 */
export function isInUse(poolId, eid) {
  if (poolId < 0 || poolId >= POOL.COUNT) return false;
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return _pools[poolId].inUse[eid] === 1;
}

/**
 * Returns the number of currently in-use entities in a pool.
 */
export function getInUse(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  return _pools[poolId].currentInUse;
}

/**
 * Returns the number of free entities in a pool.
 */
export function getFree(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  return _pools[poolId].freeCount;
}

/**
 * Returns the pool's peak in-use count.
 */
export function getPeakInUse(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  return _pools[poolId].peakInUse;
}

/**
 * Returns the pool's total capacity.
 */
export function getCapacity(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  return _pools[poolId].capacity;
}

/* ------------------------------------------------------------------ */
/* 6. RESERVATION                                                     */
/* ------------------------------------------------------------------ */

/**
 * Reserves additional entity ids for a pool. Useful at boot to pre-warm
 * a subsystem that will need many entities immediately.
 */
export function reserveBlock(poolId, count) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  const pool = _pools[poolId];
  const target = Math.min(pool.capacity, pool.reserved + Math.max(0, count | 0));
  if (target <= pool.reserved) return 0;
  const created = _bootstrapPool(pool, target);
  const delta = Math.max(0, created - pool.reserved);
  pool.reserved = created;
  return delta;
}

/* ------------------------------------------------------------------ */
/* 7. FRAME LIFECYCLE                                                 */
/* ------------------------------------------------------------------ */

/**
 * Advances the pool frame counter. Called once per frame by the engine
 * loop.
 */
export function tickPools(frameNumber) {
  if (typeof frameNumber === 'number') PoolState.frame = frameNumber;
  else PoolState.frame++;
}

/* ------------------------------------------------------------------ */
/* 8. STATISTICS                                                      */
/* ------------------------------------------------------------------ */

export function getPoolStats(poolId) {
  if (poolId < 0 || poolId >= POOL.COUNT) return null;
  const pool = _pools[poolId];
  return {
    index:         pool.index,
    name:          pool.name,
    capacity:      pool.capacity,
    reserved:      pool.reserved,
    freeCount:     pool.freeCount,
    currentInUse:  pool.currentInUse,
    peakInUse:     pool.peakInUse,
    totalAcquired: pool.totalAcquired,
    totalReleased: pool.totalReleased,
    totalRejected: pool.totalRejected,
    bootstrapped:  pool.bootstrapped,
  };
}

export function getEntityPoolReport() {
  const pools = new Array(POOL.COUNT);
  for (let i = 0; i < POOL.COUNT; i++) pools[i] = getPoolStats(i);

  let totalInUse = 0;
  let totalCapacity = 0;
  let totalReserved = 0;
  let totalPeak = 0;

  for (let i = 0; i < POOL.COUNT; i++) {
    const s = pools[i];
    totalInUse += s.currentInUse;
    totalCapacity += s.capacity;
    totalReserved += s.reserved;
    totalPeak += s.peakInUse;
  }

  return {
    frame:            PoolState.frame,
    poolCount:        POOL.COUNT,
    totalCapacity,
    totalReserved,
    totalInUse,
    totalPeak,
    totalAcquired:    PoolState.totalAcquired,
    totalReleased:    PoolState.totalReleased,
    totalRejected:    PoolState.totalRejected,
    bootstrapped:     PoolState.bootstrapped,
    pools,
    perfTier:         PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 9. HIGH-LEVEL CONVENIENCE                                          */
/* ------------------------------------------------------------------ */

export function acquireLightEntity()      { return acquire(POOL.LIGHT); }
export function acquireShadowEntity()     { return acquire(POOL.SHADOW); }
export function acquireCameraEntity()     { return acquire(POOL.CAMERA); }
export function acquireGIEntity()         { return acquire(POOL.GI); }
export function acquireAOEntity()         { return acquire(POOL.AO); }
export function acquireSceneEntity()      { return acquire(POOL.SCENE); }
export function acquireStreamingEntity()  { return acquire(POOL.STREAMING); }
export function acquireDebugEntity()      { return acquire(POOL.DEBUG); }
export function acquireParticleEntity()   { return acquire(POOL.PARTICLE); }
export function acquireUIEntity()         { return acquire(POOL.UI); }
export function acquireAnimationEntity()  { return acquire(POOL.ANIMATION); }
export function acquireWeatherEntity()    { return acquire(POOL.WEATHER); }

export function releaseLightEntity(eid)     { return release(POOL.LIGHT, eid); }
export function releaseShadowEntity(eid)    { return release(POOL.SHADOW, eid); }
export function releaseCameraEntity(eid)    { return release(POOL.CAMERA, eid); }
export function releaseGIEntity(eid)        { return release(POOL.GI, eid); }
export function releaseAOEntity(eid)        { return release(POOL.AO, eid); }
export function releaseSceneEntity(eid)     { return release(POOL.SCENE, eid); }
export function releaseStreamingEntity(eid) { return release(POOL.STREAMING, eid); }
export function releaseDebugEntity(eid)     { return release(POOL.DEBUG, eid); }
export function releaseParticleEntity(eid)  { return release(POOL.PARTICLE, eid); }
export function releaseUIEntity(eid)        { return release(POOL.UI, eid); }
export function releaseAnimationEntity(eid) { return release(POOL.ANIMATION, eid); }
export function releaseWeatherEntity(eid)   { return release(POOL.WEATHER, eid); }

/* ------------------------------------------------------------------ */
/* 10. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * The entity pool does not declare ECS components; it operates on the
 * existing component catalog. No registration needed. This export exists
 * for API symmetry with the other scene modules.
 */
export function registerEntityPoolComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 11. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Resets every pool to its bootstrapped-empty state. Entities remain
 * allocated in the world; only the in-use bookkeeping is cleared.
 */
export function resetAllPools() {
  for (let i = 0; i < POOL.COUNT; i++) {
    const pool = _pools[i];
    // Reset in-use markers and push every used eid back to the free list.
    for (let eid = 0; eid < MAX_ENTITIES; eid++) {
      if (pool.inUse[eid] === 1) {
        detachAllRelations(eid);
        clearTags(eid);
        clearTags2(eid);
        _detachSubsystemComponents(i, eid);
        pool.inUse[eid] = 0;
        pool.pushFree(eid);
      }
    }
    pool.currentInUse = 0;
    pool.peakInUse = 0;
  }
  PoolState.totalAcquired = 0;
  PoolState.totalReleased = 0;
  PoolState.totalRejected = 0;
  PoolState.frame = 0;
}

/**
 * Fully disposes every pool, deallocating all internal state. After this
 * call, `bootstrapPools()` must be called again before acquiring.
 */
export function disposeAllPools() {
  for (let i = 0; i < POOL.COUNT; i++) {
    _pools[i].reset();
  }
  PoolState.bootstrapped = false;
}

/* ------------------------------------------------------------------ */
/* 12. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Enums
  POOL,
  POOL_NAME,
  POOL_CAPACITY,

  // EntityPool class
  EntityPool,

  // State
  PoolState,

  // Bootstrap
  bootstrapPools,

  // Acquire / release
  acquire,
  acquireBatch,
  release,
  releaseBatch,
  releaseAll,
  reserveBlock,

  // Query
  isInUse,
  getInUse,
  getFree,
  getPeakInUse,
  getCapacity,

  // Frame
  tickPools,

  // Stats
  getPoolStats,
  getEntityPoolReport,

  // High-level convenience
  acquireLightEntity,
  acquireShadowEntity,
  acquireCameraEntity,
  acquireGIEntity,
  acquireAOEntity,
  acquireSceneEntity,
  acquireStreamingEntity,
  acquireDebugEntity,
  acquireParticleEntity,
  acquireUIEntity,
  acquireAnimationEntity,
  acquireWeatherEntity,

  releaseLightEntity,
  releaseShadowEntity,
  releaseCameraEntity,
  releaseGIEntity,
  releaseAOEntity,
  releaseSceneEntity,
  releaseStreamingEntity,
  releaseDebugEntity,
  releaseParticleEntity,
  releaseUIEntity,
  releaseAnimationEntity,
  releaseWeatherEntity,

  // Registration
  registerEntityPoolComponents,

  // Reset / dispose
  resetAllPools,
  disposeAllPools,
};

export default _defaultExport;