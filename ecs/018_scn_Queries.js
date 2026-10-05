// File : 018
// name : src/ecs/018_scn_Queries.js
// description : Query helper and compilation layer for the scene ECS world
//               of the anime lighting stack on Android mobile. Provides
//               cached, stable query handles so that downstream systems
//               declare their queries once at boot and never rebuild them
//               inside update loops.
//
//               The bitECS 0.4.0 `query(world, [Comp1, Comp2, ...])`
//               primitive is the raw form — it walks the world's internal
//               component masks and returns the current matching entity
//               list. This module wraps that primitive with:
//
//                 • Named query registry — `defineQuery([A, B, C])` returns
//                   a stable query descriptor with a numeric id that is
//                   stable across frames. Downstream systems store the
//                   descriptor once and call `runQuery(desc)` every frame.
//
//                 • Allocation-free iteration — `forEachEntity(desc, fn)`
//                   walks the current result set without allocating
//                   intermediate arrays. `collectEntities(desc, out)` writes
//                   results into a caller-provided typed array.
//
//                 • Component-name resolution — `defineQueryByName(['LightRef',
//                   'Transform'])` resolves names through the component
//                   registry (012) so callers never import component objects
//                   directly.
//
//                 • Type-id resolution — `defineQueryByTypeId([TID_LIGHT_REF,
//                   TID_TRANSFORM])` resolves through the type table (013).
//
//                 • Tag-aware queries — `queryByTag(mask)`,
//                   `queryTagged(mask)`, `queryTaggedAny(mask)` bridge to
//                   the tag bitmask helpers in 014 without requiring an
//                   underlying bitECS query.
//
//                 • Subsystem-aware queries — `queryPool(poolId)` and
//                   `queryAlive()` bridge to the entity pool (016) and
//                   lifetime state (017).
//
//                 • Rebuild policy — some queries depend on components that
//                   can be detached (e.g. a light that loses its shadow
//                   component); `setRebuildPolicy(desc, 'perFrame' | 'lazy'
//                   | 'manual')` lets a system choose when the cached
//                   result set is refreshed.
//
//                 • Per-query statistics — match count, peak match count,
//                   total runs, average cost, last cost — exposed for the
//                   debug HUD (024) and stats collector (025).
//
//               Integration:
//                 • 010_scn_ECSWorld.js       — world handle
//                 • 011_scn_BiteCSAdapter.js  — bitECS façade
//                 • 012_scn_ComponentRegistry.js — component-by-name
//                 • 013_scn_ComponentTypes.js — component-by-type-id
//                 • 014_scn_Tags.js           — tag bitmask
//                 • 016_scn_EntityPool.js     — subsystem pool binding
//                 • 017_scn_EntityLifetime.js — alive/dying state
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every subsystem in the anime lighting
//            stack runs its queries against a stable, cached handle —
//            zero allocations per frame, zero duplicated query logic
//            across modules, and one place to instrument query cost.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  query as bitecsQuery,
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
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getDefaultProfiler,
} from '../core/024_rnd_Profiler.js';

import {
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
  getComponent as getWorldComponent,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  getTypeId,
  getTypeName,
  getTypePascalName,
  TYPE_ID_INVALID,
} from './013_scn_ComponentTypes.js';

import {
  EntityTag,
  EntityTag2,
  forEachTagged,
  queryTagged,
  queryTaggedAny,
  collectTagged,
  countTagged,
} from './014_scn_Tags.js';

import {
  isInUse,
  getInUse,
  POOL,
  POOL_NAME,
} from './016_scn_EntityPool.js';

import {
  LIFETIME_STATE,
  EntityLifetime,
  isAlive as isLifetimeAlive,
  getState as getLifetimeState,
} from './017_scn_EntityLifetime.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of registered queries. Sized once.
 */
export const MAX_QUERIES =
  PERF_TIER_LOCAL === 'HIGH'   ? 256 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 192 :
                                 128;

/**
 * Maximum number of components in one query definition.
 */
export const MAX_QUERY_COMPONENTS = 8;

/**
 * Rebuild policies for cached query result sets.
 */
export const REBUILD_POLICY = Object.freeze({
  PER_FRAME:  0,   // refresh result set on every runQuery()
  LAZY:       1,   // refresh result set on first runQuery() of a frame
  MANUAL:     2,   // refresh only when refreshQuery() is called
  ONCE:       3,   // compute result set on defineQuery() and never again
  COUNT:      4,
});

export const REBUILD_POLICY_NAME = Object.freeze([
  'per_frame',
  'lazy',
  'manual',
  'once',
]);

/**
 * Query result — the read-only handle a downstream system sees when it
 * runs a query.
 */
export const QUERY_RESULT_OK          = 0;
export const QUERY_RESULT_EMPTY       = 1;
export const QUERY_RESULT_NOT_READY   = 2;
export const QUERY_RESULT_INVALID     = 3;

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const QueryState = {
  frame:              0,
  totalDefinitions:   0,
  totalRuns:          0,
  totalRefreshes:     0,
  totalAllocations:   0,
  totalRejected:      0,
  peakResultSize:     0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.queries', {
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
/* 2. QUERY DESCRIPTOR                                                */
/* ------------------------------------------------------------------ */

/**
 * One registered query. Holds the resolved component objects, a cached
 * entity list, and the metadata needed to decide when to refresh.
 */
export class QueryDescriptor {
  constructor(id) {
    this.id              = id;
    this.name            = null;

    // Resolved component objects (bitECS 0.4.0 plain SoA objects).
    this.components      = new Array(MAX_QUERY_COMPONENTS).fill(null);
    this.componentIds    = new Int32Array(MAX_QUERY_COMPONENTS).fill(-1);
    this.componentNames  = new Array(MAX_QUERY_COMPONENTS).fill(null);
    this.componentCount  = 0;

    // Cached entity list.
    this.entities        = new Int32Array(MAX_ENTITIES);
    this.entityCount     = 0;
    this.peakCount       = 0;

    // Rebuild policy.
    this.rebuildPolicy   = REBUILD_POLICY.LAZY;
    this.lastRefreshFrame= -1;
    this.dirty           = 1;
    this.enabled         = 1;

    // Stats.
    this.totalRuns       = 0;
    this.totalRefreshes  = 0;
    this.lastCostMs      = 0;
    this.avgCostMs       = 0;
    this.peakCostMs      = 0;
  }

  reset() {
    this.name            = null;
    for (let i = 0; i < this.componentCount; i++) {
      this.components[i] = null;
      this.componentIds[i] = -1;
      this.componentNames[i] = null;
    }
    this.componentCount  = 0;
    this.entityCount     = 0;
    this.peakCount       = 0;
    this.rebuildPolicy   = REBUILD_POLICY.LAZY;
    this.lastRefreshFrame= -1;
    this.dirty           = 1;
    this.enabled         = 1;
    this.totalRuns       = 0;
    this.totalRefreshes  = 0;
    this.lastCostMs      = 0;
    this.avgCostMs       = 0;
    this.peakCostMs      = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. GLOBAL REGISTRY                                                 */
/* ------------------------------------------------------------------ */

const _queries = new Array(MAX_QUERIES);
for (let i = 0; i < MAX_QUERIES; i++) _queries[i] = new QueryDescriptor(i);
let _queryCount = 0;
const _queryByName = new Map();

/* ------------------------------------------------------------------ */
/* 4. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _resolveComponentByName(name) {
  // Try the world-level component map first.
  let comp = getWorldComponent(name);
  if (comp) return comp;

  // Fall back to the registry (which uses the same naming convention).
  const reg = getDefaultComponentRegistry();
  if (reg) {
    const entry = reg.get(name);
    if (entry && entry.ref) return entry.ref;
  }
  return null;
}

function _resolveComponentByTypeId(tid) {
  const name = getTypePascalName(tid);
  if (!name) return null;
  return _resolveComponentByName(name);
}

/* ------------------------------------------------------------------ */
/* 5. QUERY DEFINITION                                                */
/* ------------------------------------------------------------------ */

/**
 * Defines a query from an array of resolved component objects. Returns
 * the descriptor or null.
 *
 *   const lightQuery = defineQuery([LightRef, Transform], 'lightMain');
 */
export function defineQuery(components, name) {
  if (_queryCount >= MAX_QUERIES) {
    QueryState.totalRejected++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[018_scn_Queries] query registry full (${MAX_QUERIES})`);
    return null;
  }
  if (!Array.isArray(components) || components.length === 0) {
    QueryState.totalRejected++;
    return null;
  }
  if (components.length > MAX_QUERY_COMPONENTS) {
    QueryState.totalRejected++;
    return null;
  }

  // Optional name-based dedup.
  if (name && _queryByName.has(name)) {
    return _queries[_queryByName.get(name)];
  }

  const id = _queryCount++;
  const desc = _queries[id];
  desc.reset();
  desc.name = name || null;

  for (let i = 0; i < components.length; i++) {
    const c = components[i];
    if (!c || typeof c !== 'object') {
      QueryState.totalRejected++;
      return null;
    }
    desc.components[i] = c;
    desc.componentCount++;
  }

  // Register the name → id mapping.
  if (name) _queryByName.set(name, id);

  QueryState.totalDefinitions++;
  return desc;
}

/**
 * Defines a query from component names. Names are resolved via the world
 * component map and the component registry.
 */
export function defineQueryByName(names, name) {
  if (!Array.isArray(names) || names.length === 0) return null;
  const components = new Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const c = _resolveComponentByName(names[i]);
    if (!c) {
      const log = _safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE,
        `[018_scn_Queries] unknown component "${names[i]}" for query "${name || 'unnamed'}"`);
      return null;
    }
    components[i] = c;
  }
  const desc = defineQuery(components, name);
  if (desc) {
    for (let i = 0; i < names.length; i++) {
      desc.componentNames[i] = names[i];
      desc.componentIds[i] = getTypeId(names[i]);
    }
  }
  return desc;
}

/**
 * Defines a query from component numeric type ids.
 */
export function defineQueryByTypeId(tids, name) {
  if (!Array.isArray(tids) || tids.length === 0) return null;
  const components = new Array(tids.length);
  const resolvedNames = new Array(tids.length);
  for (let i = 0; i < tids.length; i++) {
    const c = _resolveComponentByTypeId(tids[i]);
    if (!c) {
      const log = _safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE,
        `[018_scn_Queries] unknown type id ${tids[i]} for query "${name || 'unnamed'}"`);
      return null;
    }
    components[i] = c;
    resolvedNames[i] = getTypePascalName(tids[i]);
  }
  const desc = defineQuery(components, name);
  if (desc) {
    for (let i = 0; i < tids.length; i++) {
      desc.componentNames[i] = resolvedNames[i];
      desc.componentIds[i] = tids[i];
    }
  }
  return desc;
}

/**
 * Removes a query definition by name. Frees its slot.
 */
export function undefineQuery(name) {
  const id = _queryByName.get(name);
  if (id === undefined) return false;

  const last = _queryCount - 1;
  if (id !== last) {
    // Swap the last query into the freed slot.
    const moved = _queries[last];
    _queries[id] = moved;
    _queries[id].id = id;
    _queries[last] = new QueryDescriptor(last);
    if (moved.name) _queryByName.set(moved.name, id);
  } else {
    _queries[last] = new QueryDescriptor(last);
  }
  _queryCount--;
  _queryByName.delete(name);
  return true;
}

/**
 * Returns a query descriptor by name, or null.
 */
export function getQuery(name) {
  const id = _queryByName.get(name);
  return id === undefined ? null : _queries[id];
}

/**
 * Sets the rebuild policy of a query.
 */
export function setRebuildPolicy(queryOrName, policy) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return false;
  if (policy < 0 || policy >= REBUILD_POLICY.COUNT) return false;
  desc.rebuildPolicy = policy;
  return true;
}

function _resolveDescriptor(queryOrName) {
  if (!queryOrName) return null;
  if (queryOrName instanceof QueryDescriptor) return queryOrName;
  if (typeof queryOrName === 'string') return getQuery(queryOrName);
  if (typeof queryOrName === 'number') return _queries[queryOrName] || null;
  return null;
}

/* ------------------------------------------------------------------ */
/* 6. QUERY EXECUTION                                                 */
/* ------------------------------------------------------------------ */

/**
 * Ensures the query's cached result set is current. Returns the entity
 * count. Called internally by `runQuery` / `forEachEntity` / etc.
 */
export function refreshQuery(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return 0;
  if (desc.enabled === 0) return 0;
  if (desc.componentCount === 0) return 0;

  // Skip refresh when policy is ONCE and we've already refreshed.
  if (desc.rebuildPolicy === REBUILD_POLICY.ONCE && desc.lastRefreshFrame >= 0) {
    return desc.entityCount;
  }

  // Skip refresh when policy is LAZY and we've already refreshed this frame.
  if (desc.rebuildPolicy === REBUILD_POLICY.LAZY &&
      desc.lastRefreshFrame === QueryState.frame &&
      desc.dirty === 0) {
    return desc.entityCount;
  }

  // Skip refresh when policy is MANUAL and nothing marked it dirty.
  if (desc.rebuildPolicy === REBUILD_POLICY.MANUAL && desc.dirty === 0) {
    return desc.entityCount;
  }

  const t0 = _now();

  const world = getECSWorld();
  if (!world) return 0;

  // Build the component array slice for bitECS.
  const comps = desc.components;
  const n = desc.componentCount;
  let result;
  try {
    // bitECS 0.4.0 query accepts a plain array of component objects.
    const queryComponents = new Array(n);
    for (let i = 0; i < n; i++) queryComponents[i] = comps[i];
    result = bitecsQuery(world, queryComponents);
  } catch (e) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[018_scn_Queries] query "${desc.name || desc.id}" failed: ${e && e.message}`);
    return 0;
  }

  const count = result.length;
  const cap = desc.entities.length;
  const limit = count < cap ? count : cap;

  // Copy the result into the query's cached entity list.
  for (let i = 0; i < limit; i++) {
    desc.entities[i] = result[i];
  }

  desc.entityCount = limit;
  if (limit > desc.peakCount) desc.peakCount = limit;
  if (limit > QueryState.peakResultSize) QueryState.peakResultSize = limit;

  desc.lastRefreshFrame = QueryState.frame;
  desc.dirty = 0;
  desc.totalRefreshes++;
  QueryState.totalRefreshes++;

  const t1 = _now();
  const cost = t1 - t0;
  desc.lastCostMs = cost;
  const alpha = 0.15;
  desc.avgCostMs += (cost - desc.avgCostMs) * alpha;
  if (cost > desc.peakCostMs) desc.peakCostMs = cost;

  return limit;
}

/**
 * Marks a query as dirty, forcing a refresh on the next run.
 */
export function markQueryDirty(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return false;
  desc.dirty = 1;
  return true;
}

/**
 * Enables / disables a query. A disabled query always returns 0.
 */
export function setQueryEnabled(queryOrName, enabled) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return false;
  desc.enabled = enabled ? 1 : 0;
  return true;
}

/**
 * Runs the query and returns the cached entity list. The list is owned
 * by the query — do NOT mutate it and do NOT cache its reference beyond
 * the current frame if the query uses PER_FRAME rebuild policy.
 *
 * Returns { entities, count }.
 */
export function runQuery(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return _emptyResult();

  desc.totalRuns++;
  QueryState.totalRuns++;

  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) {
    desc.dirty = 1;
  }

  const count = refreshQuery(desc);
  _currentResult.entities = desc.entities;
  _currentResult.count = count;
  _currentResult.query = desc;
  return _currentResult;
}

const _currentResult = {
  entities: null,
  count:    0,
  query:    null,
};

const _emptyResult = Object.freeze({
  entities: null,
  count:    0,
  query:    null,
});

/* ------------------------------------------------------------------ */
/* 7. ITERATION HELPERS                                               */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity matching a query. Allocation-free.
 *
 *   forEachEntity(lightQuery, (eid) => { ... });
 */
export function forEachEntity(queryOrName, fn, ctx) {
  if (typeof fn !== 'function') return 0;
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return 0;

  desc.totalRuns++;
  QueryState.totalRuns++;

  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) desc.dirty = 1;
  refreshQuery(desc);

  const entities = desc.entities;
  const n = desc.entityCount;
  for (let i = 0; i < n; i++) {
    fn.call(ctx, entities[i]);
  }
  return n;
}

/**
 * Iterates every entity matching a query and stops early if `fn` returns
 * `true`. Returns the number visited.
 */
export function forEachEntityUntil(queryOrName, fn, ctx) {
  if (typeof fn !== 'function') return 0;
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return 0;

  desc.totalRuns++;
  QueryState.totalRuns++;
  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) desc.dirty = 1;
  refreshQuery(desc);

  const entities = desc.entities;
  const n = desc.entityCount;
  for (let i = 0; i < n; i++) {
    const stop = fn.call(ctx, entities[i]);
    if (stop === true) return i + 1;
  }
  return n;
}

/**
 * Copies the query's current entity list into a caller-provided
 * Int32Array. Returns the number copied.
 */
export function collectEntities(queryOrName, outArray, outOffset) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc || !outArray) return 0;

  desc.totalRuns++;
  QueryState.totalRuns++;
  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) desc.dirty = 1;
  refreshQuery(desc);

  const offset = outOffset !== undefined ? outOffset : 0;
  const cap = outArray.length - offset;
  const n = Math.min(desc.entityCount, cap);

  for (let i = 0; i < n; i++) {
    outArray[offset + i] = desc.entities[i];
  }
  return n;
}

/**
 * Returns the count of entities matching the query.
 */
export function countEntities(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return 0;
  desc.totalRuns++;
  QueryState.totalRuns++;
  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) desc.dirty = 1;
  return refreshQuery(desc);
}

/**
 * Returns the first entity matching the query, or -1.
 */
export function queryFirst(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return -1;
  desc.totalRuns++;
  QueryState.totalRuns++;
  if (desc.rebuildPolicy === REBUILD_POLICY.PER_FRAME) desc.dirty = 1;
  refreshQuery(desc);
  return desc.entityCount > 0 ? desc.entities[0] : -1;
}

/* ------------------------------------------------------------------ */
/* 8. TAG-AWARE QUERIES                                               */
/* ------------------------------------------------------------------ */

/**
 * Returns the number of entities matching ALL bits in `tagMask` (bridges
 * to 014_scn_Tags.js).
 */
export function queryByTag(tagMask) {
  return countTagged(tagMask);
}

/**
 * Iterates every entity matching ALL bits in `tagMask`. Allocation-free.
 */
export function forEachTaggedEntity(tagMask, fn, ctx) {
  return forEachTagged(tagMask, fn, ctx);
}

/**
 * Copies every entity matching ALL bits in `tagMask` into `outArray`.
 */
export function collectTaggedEntities(tagMask, outArray, outOffset) {
  if (!outArray) return 0;
  const offset = outOffset !== undefined ? outOffset : 0;
  const sub = outArray.subarray ? outArray.subarray(offset) : outArray;
  return collectTagged(tagMask, sub);
}

/**
 * Returns every entity matching ANY bits in `tagMask`. Allocates.
 */
export function queryByTagAny(tagMask) {
  return queryTaggedAny(tagMask);
}

/* ------------------------------------------------------------------ */
/* 9. SUBSYSTEM-AWARE QUERIES                                         */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity that is currently in use by the given subsystem
 * pool (bridges to 016_scn_EntityPool.js). Allocation-free.
 */
export function forEachPoolEntity(poolId, fn, ctx) {
  if (poolId < 0 || poolId >= POOL.COUNT) return 0;
  if (typeof fn !== 'function') return 0;
  let count = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (isInUse(poolId, eid)) {
      fn.call(ctx, eid);
      count++;
    }
  }
  return count;
}

/**
 * Returns the count of entities currently in use by a pool.
 */
export function countPoolEntities(poolId) {
  return getInUse(poolId);
}

/**
 * Iterates every entity that is currently in the ALIVE lifecycle state
 * (bridges to 017_scn_EntityLifetime.js). Allocation-free.
 */
export function forEachAliveEntity(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let count = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (EntityLifetime.state[eid] === LIFETIME_STATE.ALIVE) {
      fn.call(ctx, eid);
      count++;
    }
  }
  return count;
}

/**
 * Iterates every entity that is in the DYING lifecycle state.
 */
export function forEachDyingEntity(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let count = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (EntityLifetime.state[eid] === LIFETIME_STATE.DYING) {
      fn.call(ctx, eid);
      count++;
    }
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* 10. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the query system frame counter. Called once per frame by the
 * engine loop, before any system runs its queries.
 */
export function tickQueries(frameNumber) {
  if (typeof frameNumber === 'number') QueryState.frame = frameNumber;
  else QueryState.frame++;
}

/* ------------------------------------------------------------------ */
/* 11. BUILT-IN LIGHTING QUERIES                                      */
/* ------------------------------------------------------------------ */

let _lightQuery = null;
let _shadowCasterQuery = null;
let _shadowReceiverQuery = null;
let _giProbeQuery = null;
let _aoVolumeQuery = null;
let _cameraQuery = null;
let _lightShadowQuery = null;
let _compositeQuery = null;

/**
 * Defines the standard set of lighting queries used by the engine. Called
 * once at boot. Safe to call multiple times — subsequent calls are no-ops.
 */
export function defineBuiltinQueries() {
  if (!_lightQuery) {
    _lightQuery = defineQueryByName(
      ['Transform', 'LightRef', 'LightState'],
      'builtin.lights');
    if (_lightQuery) _lightQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_shadowCasterQuery) {
    _shadowCasterQuery = defineQueryByName(
      ['Transform', 'ShadowCasterRef'],
      'builtin.shadowCasters');
    if (_shadowCasterQuery) _shadowCasterQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_shadowReceiverQuery) {
    _shadowReceiverQuery = defineQueryByName(
      ['Transform', 'ShadowReceiverRef'],
      'builtin.shadowReceivers');
    if (_shadowReceiverQuery) _shadowReceiverQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_giProbeQuery) {
    _giProbeQuery = defineQueryByName(
      ['GIProbeRef', 'GIIrradiance'],
      'builtin.giProbes');
    if (_giProbeQuery) _giProbeQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_aoVolumeQuery) {
    _aoVolumeQuery = defineQueryByName(
      ['AOVolumeRef'],
      'builtin.aoVolumes');
    if (_aoVolumeQuery) _aoVolumeQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_cameraQuery) {
    _cameraQuery = defineQueryByName(
      ['Transform', 'CameraTag'],
      'builtin.cameras');
    if (_cameraQuery) _cameraQuery.rebuildPolicy = REBUILD_POLICY.ONCE;
  }
  if (!_lightShadowQuery) {
    _lightShadowQuery = defineQueryByName(
      ['LightRef', 'LightShadow'],
      'builtin.lightShadows');
    if (_lightShadowQuery) _lightShadowQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }
  if (!_compositeQuery) {
    _compositeQuery = defineQueryByName(
      ['LightComposite'],
      'builtin.lightComposites');
    if (_compositeQuery) _compositeQuery.rebuildPolicy = REBUILD_POLICY.LAZY;
  }

  return {
    lightQuery: _lightQuery,
    shadowCasterQuery: _shadowCasterQuery,
    shadowReceiverQuery: _shadowReceiverQuery,
    giProbeQuery: _giProbeQuery,
    aoVolumeQuery: _aoVolumeQuery,
    cameraQuery: _cameraQuery,
    lightShadowQuery: _lightShadowQuery,
    compositeQuery: _compositeQuery,
  };
}

export function getLightQuery()           { return _lightQuery; }
export function getShadowCasterQuery()    { return _shadowCasterQuery; }
export function getShadowReceiverQuery()  { return _shadowReceiverQuery; }
export function getGIProbeQuery()         { return _giProbeQuery; }
export function getAOVolumeQuery()        { return _aoVolumeQuery; }
export function getCameraQuery()          { return _cameraQuery; }
export function getLightShadowQuery()     { return _lightShadowQuery; }
export function getCompositeQuery()       { return _compositeQuery; }

/* ------------------------------------------------------------------ */
/* 12. STATISTICS                                                     */
/* ------------------------------------------------------------------ */

export function getQueryStats(queryOrName) {
  const desc = _resolveDescriptor(queryOrName);
  if (!desc) return null;
  return {
    id:               desc.id,
    name:             desc.name,
    componentCount:   desc.componentCount,
    componentNames:   desc.componentNames.slice(0, desc.componentCount),
    entityCount:      desc.entityCount,
    peakCount:        desc.peakCount,
    totalRuns:        desc.totalRuns,
    totalRefreshes:   desc.totalRefreshes,
    lastCostMs:       desc.lastCostMs,
    avgCostMs:        desc.avgCostMs,
    peakCostMs:       desc.peakCostMs,
    rebuildPolicy:    REBUILD_POLICY_NAME[desc.rebuildPolicy] || 'lazy',
    enabled:          desc.enabled === 1,
  };
}

export function getAllQueryStats() {
  const out = new Array(_queryCount);
  for (let i = 0; i < _queryCount; i++) {
    out[i] = getQueryStats(_queries[i]);
  }
  return out;
}

export function getQuerySystemReport() {
  return {
    frame:             QueryState.frame,
    registeredQueries: _queryCount,
    capacity:          MAX_QUERIES,
    totalDefinitions:  QueryState.totalDefinitions,
    totalRuns:         QueryState.totalRuns,
    totalRefreshes:    QueryState.totalRefreshes,
    totalRejected:     QueryState.totalRejected,
    peakResultSize:    QueryState.peakResultSize,
    queries:           getAllQueryStats(),
    perfTier:          PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 13. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every query definition and result cache.
 */
export function resetAllQueries() {
  for (let i = 0; i < MAX_QUERIES; i++) _queries[i].reset();
  _queryCount = 0;
  _queryByName.clear();

  _lightQuery = null;
  _shadowCasterQuery = null;
  _shadowReceiverQuery = null;
  _giProbeQuery = null;
  _aoVolumeQuery = null;
  _cameraQuery = null;
  _lightShadowQuery = null;
  _compositeQuery = null;

  QueryState.frame = 0;
  QueryState.totalDefinitions = 0;
  QueryState.totalRuns = 0;
  QueryState.totalRefreshes = 0;
  QueryState.totalAllocations = 0;
  QueryState.totalRejected = 0;
  QueryState.peakResultSize = 0;
}

/* ------------------------------------------------------------------ */
/* 14. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * The query system does not declare its own ECS components. No-op,
 * present for API symmetry with the other scene modules.
 */
export function registerQueryComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 15. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_QUERIES,
  MAX_QUERY_COMPONENTS,
  REBUILD_POLICY,
  REBUILD_POLICY_NAME,
  QUERY_RESULT_OK,
  QUERY_RESULT_EMPTY,
  QUERY_RESULT_NOT_READY,
  QUERY_RESULT_INVALID,

  // Descriptor class
  QueryDescriptor,

  // Module state
  QueryState,

  // Definition
  defineQuery,
  defineQueryByName,
  defineQueryByTypeId,
  undefineQuery,
  getQuery,
  setRebuildPolicy,

  // Execution
  runQuery,
  refreshQuery,
  markQueryDirty,
  setQueryEnabled,

  // Iteration
  forEachEntity,
  forEachEntityUntil,
  collectEntities,
  countEntities,
  queryFirst,

  // Tag queries
  queryByTag,
  queryByTagAny,
  forEachTaggedEntity,
  collectTaggedEntities,

  // Subsystem queries
  forEachPoolEntity,
  countPoolEntities,
  forEachAliveEntity,
  forEachDyingEntity,

  // Frame
  tickQueries,

  // Built-in queries
  defineBuiltinQueries,
  getLightQuery,
  getShadowCasterQuery,
  getShadowReceiverQuery,
  getGIProbeQuery,
  getAOVolumeQuery,
  getCameraQuery,
  getLightShadowQuery,
  getCompositeQuery,

  // Stats
  getQueryStats,
  getAllQueryStats,
  getQuerySystemReport,

  // Registration
  registerQueryComponents,

  // Reset
  resetAllQueries,
};

export default _defaultExport;