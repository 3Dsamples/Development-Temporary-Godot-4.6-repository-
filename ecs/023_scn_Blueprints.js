// File : 023
// name : src/ecs/023_scn_Blueprints.js
// description : Blueprint-composition module for the scene ECS world of the
//               anime lighting stack on Android mobile. A blueprint is a
//               NAMED collection of prefab entities plus the relationship
//               edges that bind them (parent/child hierarchy, typed
//               references such as caster→light, probe→volume, portal↔portal,
//               composite member↔anchor). Where 022_scn_SpawnPrefabs.js
//               spawns ONE entity at a time, this module spawns a WHOLE
//               scene region — an outdoor biome with sun + sky light + GI
//               probe grid + AO outdoor volume; an indoor room with lamp +
//               window shaft + GI indoor volume + AO indoor volume; a magic
//               scene with glow + point flicker + caustic; a forest scene,
//               a snow tundra, a desert canyon, a coastal bay, an orbital
//               space scene, a pastel portrait scene, a flower field, a
//               neon alley — as a single O(members + edges) call.
//
//               Design:
//                 • Fixed-capacity blueprint registry — MAX_BLUEPRINTS slots,
//                   declared once at module load and never resized.
//                 • Fixed-capacity instance registry — MAX_INSTANCES slots,
//                   each holding up to MAX_MEMBERS_PER_BLUEPRINT live
//                   entity ids plus metadata.
//                 • Zero-allocation instantiation on the hot path — every
//                   member spawn, every edge, every tag write goes straight
//                   into typed arrays; no dynamic prop bags, no closures,
//                   no intermediate objects.
//                 • Auto-lifetime — every blueprint declares a lifetime
//                   policy (permanent, scene, transient). When an instance
//                   is disposed, every member entity is marked dying with
//                   the blueprint's grace period so the lifetime system
//                   (017) can collect them deterministically.
//                 • Auto-relations — the blueprint's edge list is applied
//                   through 015_scn_Relations.js so hierarchy + typed
//                   references are constructed in one pass.
//                 • Auto-tags — every member is tagged with the blueprint's
//                   tag mask plus its own per-member tag mask, so a
//                   downstream system can query "every entity in the snow
//                   biome blueprint" with one integer compare.
//                 • Deterministic — instance ids are monotonic; member
//                   order is preserved; relations are applied in declaration
//                   order. Two identical blueprint instantiations always
//                   produce identical entity layouts.
//                 • Bulk dispose — `disposeBlueprintInstance(instanceId)`
//                   walks every live member and marks it dying; the lifetime
//                   system does the actual pool return. `disposeAllInstances
//                   (blueprintId)` and `disposeAll()` are provided for
//                   teardown.
//                 • Snapshot / restore — `getInstanceSnapshot(instanceId)`
//                   returns an immutable view of the member entity ids so a
//                   debug HUD (179) or regression tool can capture the
//                   blueprint's live state.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — light components
//                 • 003_lgt_ShadowComponents.js     — shadow components
//                 • 004_lgt_GIComponents.js         — GI components
//                 • 005_lgt_AOComponents.js         — AO components
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component catalog
//                 • 014_scn_Tags.js                 — tag bits
//                 • 015_scn_Relations.js            — relations graph
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime states
//                 • 022_scn_SpawnPrefabs.js         — prefab spawn
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every scene region in the anime lighting
//            stack is spawned, tagged, related, and disposed as one
//            deterministic unit — so a biome transition, a room load, a
//            scene switch, or a full engine teardown is a single call
//            with predictable cost and predictable entity layout.
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
  getECSWorld,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  TAG,
  TAG2,
  tagEntity,
  untagEntity,
  tagEntity2,
  untagEntity2,
  syncTagsToSoA,
} from './014_scn_Tags.js';

import {
  REF,
  attachChild,
  detachChild,
  setReference,
  clearReference,
  detachAllRelations,
} from './015_scn_Relations.js';

import {
  POOL,
  release,
  isInUse,
} from './016_scn_EntityPool.js';

import {
  LIFETIME_STATE,
  LIFETIME_POOL,
  EntityLifetime,
  markDying,
  markSpawning,
  setPersistent,
  setTransient,
  setTTL,
  DEFAULT_DYING_GRACE_FRAMES,
} from './017_scn_EntityLifetime.js';

import {
  PREFAB,
  PREFAB_NAME,
  PREFAB_LIFETIME,
  getPrefab,
  getPrefabByName,
  spawnPrefab,
  releasePrefab,
} from './022_scn_SpawnPrefabs.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of blueprint definitions. Sized once.
 */
export const MAX_BLUEPRINTS =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 48 :
                                 32;

/**
 * Maximum number of simultaneous blueprint instances.
 */
export const MAX_INSTANCES =
  PERF_TIER_LOCAL === 'HIGH'   ? 256 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 128 :
                                 64;

/**
 * Maximum number of members in one blueprint.
 */
export const MAX_MEMBERS_PER_BLUEPRINT =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 48 :
                                 32;

/**
 * Maximum number of relation edges in one blueprint.
 */
export const MAX_EDGES_PER_BLUEPRINT =
  PERF_TIER_LOCAL === 'HIGH'   ? 128 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 96 :
                                 64;

/**
 * Maximum number of parent-child hierarchy entries in one blueprint.
 */
export const MAX_HIERARCHY_PER_BLUEPRINT = 32;

/**
 * Blueprint lifecycle states.
 */
export const BLUEPRINT_STATE = Object.freeze({
  NONE:          0,
  REGISTERED:    1,
  INSTANTIATING: 2,
  LIVE:          3,
  DISPOSING:     4,
  DISPOSED:      5,
  FAILED:        6,
  COUNT:         7,
});

export const BLUEPRINT_STATE_NAME = Object.freeze([
  'none',
  'registered',
  'instantiating',
  'live',
  'disposing',
  'disposed',
  'failed',
]);

/**
 * Canonical blueprint ids for the anime lighting stack. Every reference
 * image style has a matching blueprint.
 */
export const BLUEPRINT = Object.freeze({
  /* ---------------- Base scene blueprints ---------------- */
  OUTDOOR_SCENE:          0,
  INTERIOR_ROOM:          1,
  TRANSITION_ZONE:        2,

  /* ---------------- Reference image biomes ---------------- */
  FOREST_VALLEY:          3,   // image 2 top-left/top-middle
  SNOW_TUNDRA:            4,   // image 3 top-right
  DESERT_CANYON:          5,   // image 1
  COASTAL_BAY:            6,   // image 5
  ORBITAL_SPACE:          7,   // image 4
  SUNSET_TOMBSTONE:       8,   // image 6
  PASTEL_PORTRAIT:        9,   // image 7
  MAGIC_CASTER:          10,   // image 8
  FLOWER_FIELD:          11,   // image 9

  /* ---------------- Small scenery blueprints ---------------- */
  FIRE_CAMP:             12,   // fire light + caustic
  NEON_ALLEY:            13,   // neon light + interior ambient
  WINDOW_ROOM:           14,   // window shaft + interior lamp
  CAVE_INTERIOR:         15,   // cave + GI portal cave mouth + AO indoor

  /* ---------------- Test / debug ---------------- */
  DEBUG_LIGHT_PROBE:     30,
  DEBUG_GI_GRID:         31,

  COUNT:                 32,
});

export const BLUEPRINT_NAME = Object.freeze([
  'outdoor_scene', 'interior_room', 'transition_zone',
  'forest_valley', 'snow_tundra', 'desert_canyon', 'coastal_bay',
  'orbital_space', 'sunset_tombstone', 'pastel_portrait', 'magic_caster',
  'flower_field',
  'fire_camp', 'neon_alley', 'window_room', 'cave_interior',
  'unused_16', 'unused_17', 'unused_18', 'unused_19',
  'unused_20', 'unused_21', 'unused_22', 'unused_23',
  'unused_24', 'unused_25', 'unused_26', 'unused_27',
  'unused_28', 'unused_29', 'debug_light_probe', 'debug_gi_grid',
]);

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const BlueprintState = {
  frame:              0,
  totalDefinitions:   0,
  totalInstantiations:0,
  totalDisposals:     0,
  totalRejected:      0,
  totalMembersSpawned:0,
  totalEdgesBuilt:    0,
  totalHierarchyBuilt:0,
  peakLiveInstances:  0,
  liveInstanceCount:  0,
  lastInstantiateMs:  0,
  lastDisposeMs:      0,
  avgInstantiateMs:   0,
  avgDisposeMs:       0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.blueprints', {
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
/* 2. BLUEPRINT DESCRIPTOR                                            */
/* ------------------------------------------------------------------ */

/**
 * One member of a blueprint. Frozen after definition.
 *
 *   {
 *     prefabId,             // PREFAB.*
 *     memberName,           // optional symbolic name for edges
 *     overrides,            // optional prefab override object
 *     tagMask,              // additional tag bits
 *     tag2Mask,             // additional secondary tag bits
 *     parentMember,         // optional index of parent member (-1 = none)
 *     parentRef,            // optional REF kind for the parent edge
 *     persistent,           // optional override of lifetime persistence
 *     transient,            // optional override of lifetime transient flag
 *     ttlFrames,            // optional TTL override
 *   }
 */
export class BlueprintMember {
  constructor(index, spec) {
    this.index        = index;
    this.prefabId     = spec.prefabId !== undefined ? spec.prefabId : PREFAB.POINT_LIGHT;
    this.memberName   = spec.memberName || null;
    this.overrides    = spec.overrides ? Object.freeze(_freezeOverrides(spec.overrides)) : null;
    this.tagMask      = (spec.tagMask  !== undefined ? spec.tagMask  : 0) >>> 0;
    this.tag2Mask     = (spec.tag2Mask !== undefined ? spec.tag2Mask : 0) >>> 0;
    this.parentMember = spec.parentMember !== undefined ? (spec.parentMember | 0) : -1;
    this.parentRef    = spec.parentRef !== undefined ? (spec.parentRef | 0) : REF.NONE;
    this.persistent   = spec.persistent === true ? 1 : 0;
    this.transient    = spec.transient  === true ? 1 : 0;
    this.ttlFrames    = spec.ttlFrames  !== undefined ? (spec.ttlFrames | 0) : 0;
    Object.freeze(this);
  }
}

function _freezeOverrides(o) {
  const out = {};
  for (const compName in o) {
    const fields = o[compName];
    if (!fields || typeof fields !== 'object') continue;
    const frozen = {};
    for (const fieldName in fields) {
      const v = fields[fieldName];
      frozen[fieldName] = Array.isArray(v) ? Object.freeze(v.slice()) : v;
    }
    out[compName] = Object.freeze(frozen);
  }
  return out;
}

/**
 * One relation edge of a blueprint. Frozen after definition.
 *
 *   {
 *     fromMember,       // index of source member
 *     toMember,         // index of target member (or -1 = instance root)
 *     kind,             // REF kind
 *     weight,           // optional float weight
 *   }
 */
export class BlueprintEdge {
  constructor(index, spec) {
    this.index      = index;
    this.fromMember = spec.fromMember !== undefined ? (spec.fromMember | 0) : 0;
    this.toMember   = spec.toMember   !== undefined ? (spec.toMember   | 0) : -1;
    this.kind       = spec.kind !== undefined ? (spec.kind | 0) : REF.NONE;
    this.weight     = spec.weight !== undefined ? Number(spec.weight) : 1.0;
    Object.freeze(this);
  }
}

/**
 * A blueprint definition. Frozen after registration.
 */
export class BlueprintDescriptor {
  constructor(id, spec) {
    this.id            = id;
    this.name          = spec.name || BLUEPRINT_NAME[id] || ('blueprint_' + id);
    this.description   = spec.description || null;
    this.lifetime      = spec.lifetime !== undefined ? spec.lifetime : PREFAB_LIFETIME.SCENE;
    this.ttlFrames     = spec.ttlFrames !== undefined ? (spec.ttlFrames | 0) : 0;
    this.graceFrames   = spec.graceFrames !== undefined ? (spec.graceFrames | 0) : DEFAULT_DYING_GRACE_FRAMES;

    this.tagMask       = (spec.tagMask  !== undefined ? spec.tagMask  : 0) >>> 0;
    this.tag2Mask      = (spec.tag2Mask !== undefined ? spec.tag2Mask : 0) >>> 0;

    this.members       = Object.freeze((spec.members || []).map((m, i) => new BlueprintMember(i, m)));
    this.edges         = Object.freeze((spec.edges   || []).map((e, i) => new BlueprintEdge(i, e)));

    this.memberCount   = this.members.length;
    this.edgeCount     = this.edges.length;

    // Sanity — cap member + edge count.
    if (this.memberCount > MAX_MEMBERS_PER_BLUEPRINT) {
      throw new Error(`[023_scn_Blueprints] blueprint "${this.name}" has ${this.memberCount} members > ${MAX_MEMBERS_PER_BLUEPRINT}`);
    }
    if (this.edgeCount > MAX_EDGES_PER_BLUEPRINT) {
      throw new Error(`[023_scn_Blueprints] blueprint "${this.name}" has ${this.edgeCount} edges > ${MAX_EDGES_PER_BLUEPRINT}`);
    }

    Object.freeze(this);
  }
}

/* ------------------------------------------------------------------ */
/* 3. BLUEPRINT REGISTRY                                              */
/* ------------------------------------------------------------------ */

const _blueprints = new Array(MAX_BLUEPRINTS).fill(null);
const _blueprintByName = new Map();

/**
 * Registers a blueprint definition. Returns true on success.
 */
export function defineBlueprint(spec) {
  if (!spec || typeof spec !== 'object') return false;
  const id = spec.id !== undefined ? (spec.id | 0) : -1;
  if (id < 0 || id >= MAX_BLUEPRINTS) {
    BlueprintState.totalRejected++;
    return false;
  }
  if (_blueprints[id]) {
    // Already registered — replace only if the name matches, otherwise fail.
    if (_blueprints[id].name === spec.name) return true;
    BlueprintState.totalRejected++;
    return false;
  }

  try {
    const descriptor = new BlueprintDescriptor(id, spec);
    _blueprints[id] = descriptor;
    _blueprintByName.set(descriptor.name, id);
    BlueprintState.totalDefinitions++;
    return true;
  } catch (e) {
    BlueprintState.totalRejected++;
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[023_scn_Blueprints] failed to define blueprint "${spec.name}": ${e && e.message}`);
    return false;
  }
}

/**
 * Returns a blueprint descriptor by id, or null.
 */
export function getBlueprint(id) {
  if (id < 0 || id >= MAX_BLUEPRINTS) return null;
  return _blueprints[id];
}

/**
 * Returns a blueprint descriptor by name, or null.
 */
export function getBlueprintByName(name) {
  const id = _blueprintByName.get(name);
  return id === undefined ? null : _blueprints[id];
}

/**
 * Returns the number of registered blueprints.
 */
export function getBlueprintCount() {
  let n = 0;
  for (let i = 0; i < MAX_BLUEPRINTS; i++) if (_blueprints[i]) n++;
  return n;
}

/* ------------------------------------------------------------------ */
/* 4. BLUEPRINT INSTANCE                                              */
/* ------------------------------------------------------------------ */

/**
 * One live blueprint instance. Holds the entity ids of every member in
 * declaration order, plus metadata.
 */
export class BlueprintInstance {
  constructor(index) {
    this.index        = index;
    this.instanceId   = 0;    // monotonic public id
    this.blueprintId  = -1;
    this.blueprint    = null;
    this.state        = BLUEPRINT_STATE.NONE;

    // Live member entity ids in declaration order.
    this.memberEids   = new Int32Array(MAX_MEMBERS_PER_BLUEPRINT);
    this.memberCount  = 0;

    // Optional root entity (member 0 by convention, or -1).
    this.rootEid      = -1;

    // Instance-level tag bits (applied to every member).
    this.instanceTagMask  = 0;
    this.instanceTag2Mask = 0;

    // Timestamps.
    this.instantiatedAtMs   = 0;
    this.instantiatedAtFrame= 0;
    this.disposedAtMs       = 0;
    this.disposedAtFrame    = 0;

    // Counters.
    this.relationsApplied   = 0;
    this.hierarchyApplied   = 0;

    // Failure state.
    this.failureCount       = 0;
  }

  reset() {
    this.instanceId         = 0;
    this.blueprintId        = -1;
    this.blueprint          = null;
    this.state              = BLUEPRINT_STATE.NONE;
    for (let i = 0; i < this.memberCount; i++) this.memberEids[i] = -1;
    this.memberCount        = 0;
    this.rootEid            = -1;
    this.instanceTagMask    = 0;
    this.instanceTag2Mask   = 0;
    this.instantiatedAtMs   = 0;
    this.instantiatedAtFrame= 0;
    this.disposedAtMs       = 0;
    this.disposedAtFrame    = 0;
    this.relationsApplied   = 0;
    this.hierarchyApplied   = 0;
    this.failureCount       = 0;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. INSTANCE REGISTRY                                               */
/* ------------------------------------------------------------------ */

const _instances = new Array(MAX_INSTANCES);
for (let i = 0; i < MAX_INSTANCES; i++) _instances[i] = new BlueprintInstance(i);
let _instanceCount = 0;
let _instanceIdCounter = 0;
const _instanceById = new Map();

/* ------------------------------------------------------------------ */
/* 6. INSTANTIATION                                                   */
/* ------------------------------------------------------------------ */

/**
 * Instantiates a blueprint. Returns a public instance id (>= 1) or -1.
 *
 *   const instId = instantiateBlueprint(world, BLUEPRINT.SNOW_TUNDRA);
 */
export function instantiateBlueprint(world, blueprintId, options) {
  const t0 = _now();

  if (blueprintId < 0 || blueprintId >= MAX_BLUEPRINTS) {
    BlueprintState.totalRejected++;
    return -1;
  }
  const blueprint = _blueprints[blueprintId];
  if (!blueprint) {
    BlueprintState.totalRejected++;
    return -1;
  }

  if (_instanceCount >= MAX_INSTANCES) {
    BlueprintState.totalRejected++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[023_scn_Blueprints] instance capacity reached (${MAX_INSTANCES})`);
    return -1;
  }

  // Allocate a slot.
  const slot = _instanceCount++;
  const instance = _instances[slot];
  instance.reset();

  const publicId = ++_instanceIdCounter;
  instance.instanceId       = publicId;
  instance.blueprintId      = blueprintId;
  instance.blueprint        = blueprint;
  instance.state            = BLUEPRINT_STATE.INSTANTIATING;
  instance.instantiatedAtMs = t0;
  instance.instantiatedAtFrame = BlueprintState.frame;

  // Apply instance-level tag masks from options.
  if (options && typeof options === 'object') {
    if (typeof options.tagMask === 'number')  instance.instanceTagMask  = options.tagMask >>> 0;
    if (typeof options.tag2Mask === 'number') instance.instanceTag2Mask = options.tag2Mask >>> 0;
  }

  // 1. Spawn every member.
  let spawned = 0;
  for (let m = 0; m < blueprint.memberCount; m++) {
    const member = blueprint.members[m];
    const mergedOverrides = _mergeOverrides(member.overrides, options && options.memberOverrides ? options.memberOverrides[m] : null);

    const eid = spawnPrefab(world, member.prefabId, mergedOverrides || undefined);
    if (eid < 0) {
      instance.failureCount++;
      // Roll back already spawned members.
      _rollbackMembers(instance, spawned);
      instance.state = BLUEPRINT_STATE.FAILED;
      BlueprintState.totalRejected++;
      return -1;
    }

    instance.memberEids[m] = eid;
    instance.memberCount++;

    // Apply per-member tag bits on top of the prefab defaults.
    if (member.tagMask !== 0)  tagEntity(eid, member.tagMask);
    if (member.tag2Mask !== 0) tagEntity2(eid, member.tag2Mask);

    // Apply instance-level tags.
    if (instance.instanceTagMask !== 0)  tagEntity(eid, instance.instanceTagMask);
    if (instance.instanceTag2Mask !== 0) tagEntity2(eid, instance.instanceTag2Mask);

    // Apply per-member lifetime overrides.
    if (member.persistent === 1) setPersistent(eid, true);
    if (member.transient === 1)  setTransient(eid, true);
    if (member.ttlFrames > 0)    setTTL(eid, member.ttlFrames);

    syncTagsToSoA(eid);
    spawned++;
    BlueprintState.totalMembersSpawned++;
  }

  // Set the root entity (member 0 by convention).
  instance.rootEid = blueprint.memberCount > 0 ? instance.memberEids[0] : -1;

  // 2. Apply hierarchy edges (parent → child).
  let hierarchyApplied = 0;
  for (let m = 0; m < blueprint.memberCount; m++) {
    const member = blueprint.members[m];
    if (member.parentMember < 0 || member.parentMember >= blueprint.memberCount) continue;
    if (m === member.parentMember) continue;   // refuse self-parent

    const parentEid = instance.memberEids[member.parentMember];
    const childEid  = instance.memberEids[m];
    if (parentEid < 0 || childEid < 0) continue;

    if (attachChild(parentEid, childEid, member.parentRef)) {
      hierarchyApplied++;
      BlueprintState.totalHierarchyBuilt++;
    }
  }
  instance.hierarchyApplied = hierarchyApplied;

  // 3. Apply typed reference edges.
  let relationsApplied = 0;
  for (let e = 0; e < blueprint.edgeCount; e++) {
    const edge = blueprint.edges[e];
    if (edge.fromMember < 0 || edge.fromMember >= blueprint.memberCount) continue;
    const fromEid = instance.memberEids[edge.fromMember];
    if (fromEid < 0) continue;

    let toEid = -1;
    if (edge.toMember === -1) {
      // Edge to instance root.
      toEid = instance.rootEid;
    } else if (edge.toMember >= 0 && edge.toMember < blueprint.memberCount) {
      toEid = instance.memberEids[edge.toMember];
    }
    if (toEid < 0) continue;

    if (setReference(fromEid, toEid, edge.kind, edge.weight) >= 0) {
      relationsApplied++;
      BlueprintState.totalEdgesBuilt++;
    }
  }
  instance.relationsApplied = relationsApplied;

  // Mark live.
  instance.state = BLUEPRINT_STATE.LIVE;
  BlueprintState.totalInstantiations++;
  BlueprintState.liveInstanceCount++;
  if (BlueprintState.liveInstanceCount > BlueprintState.peakLiveInstances) {
    BlueprintState.peakLiveInstances = BlueprintState.liveInstanceCount;
  }

  _instanceById.set(publicId, instance);

  const t1 = _now();
  const cost = t1 - t0;
  BlueprintState.lastInstantiateMs = cost;
  BlueprintState.avgInstantiateMs += (cost - BlueprintState.avgInstantiateMs) * 0.15;

  return publicId;
}

function _mergeOverrides(base, extra) {
  if (!base && !extra) return null;
  if (!base) return extra;
  if (!extra) return base;

  // Shallow merge per component; deeper fields copied by reference.
  const out = {};
  for (const compName in base) out[compName] = base[compName];
  for (const compName in extra) {
    if (out[compName]) {
      const merged = {};
      for (const f in out[compName]) merged[f] = out[compName][f];
      for (const f in extra[compName]) merged[f] = extra[compName][f];
      out[compName] = merged;
    } else {
      out[compName] = extra[compName];
    }
  }
  return out;
}

function _rollbackMembers(instance, count) {
  for (let i = 0; i < count; i++) {
    const eid = instance.memberEids[i];
    if (eid >= 0) {
      // Detach relations first.
      try { detachAllRelations(eid); } catch (_) {}
      markDying(eid, 0);   // immediate release via lifetime system
      instance.memberEids[i] = -1;
    }
  }
  instance.memberCount = 0;
}

/* ------------------------------------------------------------------ */
/* 7. DISPOSAL                                                        */
/* ------------------------------------------------------------------ */

/**
 * Disposes a live blueprint instance. Marks every member entity as dying
 * with the blueprint's grace period; the lifetime system (017) will
 * release them to their pools.
 *
 * Returns true on success.
 */
export function disposeBlueprintInstance(instanceId) {
  const t0 = _now();

  const instance = _instanceById.get(instanceId);
  if (!instance) return false;
  if (instance.state === BLUEPRINT_STATE.DISPOSED) return false;

  instance.state = BLUEPRINT_STATE.DISPOSING;

  const grace = instance.blueprint ? instance.blueprint.graceFrames : DEFAULT_DYING_GRACE_FRAMES;

  let disposed = 0;
  for (let i = 0; i < instance.memberCount; i++) {
    const eid = instance.memberEids[i];
    if (eid < 0) continue;

    // Detach relations before marking dying so the graph stays clean
    // while the entity fades out.
    try { detachAllRelations(eid); } catch (_) {}

    if (markDying(eid, grace)) {
      disposed++;
    }
  }

  instance.state = BLUEPRINT_STATE.DISPOSED;
  instance.disposedAtMs = _now();
  instance.disposedAtFrame = BlueprintState.frame;

  // Remove from the public index.
  _instanceById.delete(instanceId);

  BlueprintState.totalDisposals++;
  if (BlueprintState.liveInstanceCount > 0) BlueprintState.liveInstanceCount--;

  const t1 = _now();
  const cost = t1 - t0;
  BlueprintState.lastDisposeMs = cost;
  BlueprintState.avgDisposeMs += (cost - BlueprintState.avgDisposeMs) * 0.15;

  return true;
}

/**
 * Disposes every live instance of the given blueprint. Returns the count
 * disposed.
 */
export function disposeAllInstancesOfBlueprint(blueprintId) {
  let count = 0;
  const ids = [];
  for (const [publicId, instance] of _instanceById) {
    if (instance.blueprintId === blueprintId) ids.push(publicId);
  }
  for (let i = 0; i < ids.length; i++) {
    if (disposeBlueprintInstance(ids[i])) count++;
  }
  return count;
}

/**
 * Disposes every live instance. Returns the count disposed.
 */
export function disposeAllInstances() {
  let count = 0;
  const ids = [];
  for (const [publicId] of _instanceById) ids.push(publicId);
  for (let i = 0; i < ids.length; i++) {
    if (disposeBlueprintInstance(ids[i])) count++;
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* 8. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Returns the instance record for a public instance id, or null.
 */
export function getInstance(instanceId) {
  return _instanceById.get(instanceId) || null;
}

/**
 * Returns the entity id of a member in a live instance, or -1.
 */
export function getInstanceMemberEid(instanceId, memberIndex) {
  const instance = _instanceById.get(instanceId);
  if (!instance) return -1;
  if (memberIndex < 0 || memberIndex >= instance.memberCount) return -1;
  return instance.memberEids[memberIndex];
}

/**
 * Returns the entity id of the member with the given symbolic name, or -1.
 */
export function getInstanceMemberByName(instanceId, memberName) {
  const instance = _instanceById.get(instanceId);
  if (!instance || !instance.blueprint) return -1;
  const blueprint = instance.blueprint;
  for (let i = 0; i < blueprint.memberCount; i++) {
    if (blueprint.members[i].memberName === memberName) {
      return instance.memberEids[i];
    }
  }
  return -1;
}

/**
 * Returns the root entity id of a live instance, or -1.
 */
export function getInstanceRootEid(instanceId) {
  const instance = _instanceById.get(instanceId);
  return instance ? instance.rootEid : -1;
}

/**
 * Iterates every live instance. Allocation-free.
 */
export function forEachInstance(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  let n = 0;
  for (const instance of _instanceById.values()) {
    if (instance.state === BLUEPRINT_STATE.LIVE) {
      fn.call(ctx, instance, instance.instanceId);
      n++;
    }
  }
  return n;
}

/**
 * Iterates every live member of an instance. Allocation-free.
 */
export function forEachInstanceMember(instanceId, fn, ctx) {
  if (typeof fn !== 'function') return 0;
  const instance = _instanceById.get(instanceId);
  if (!instance) return 0;
  let n = 0;
  for (let i = 0; i < instance.memberCount; i++) {
    const eid = instance.memberEids[i];
    if (eid < 0) continue;
    fn.call(ctx, eid, i);
    n++;
  }
  return n;
}

/**
 * Returns a snapshot of an instance's live entity layout. Allocates a
 * fresh object — use only off the hot path.
 */
export function getInstanceSnapshot(instanceId) {
  const instance = _instanceById.get(instanceId);
  if (!instance) return null;
  const members = [];
  for (let i = 0; i < instance.memberCount; i++) {
    const bpMember = instance.blueprint ? instance.blueprint.members[i] : null;
    members.push({
      index:      i,
      eid:        instance.memberEids[i],
      prefabId:   bpMember ? bpMember.prefabId : -1,
      prefabName: bpMember ? (PREFAB_NAME[bpMember.prefabId] || null) : null,
      memberName: bpMember ? bpMember.memberName : null,
      alive:      instance.memberEids[i] >= 0,
    });
  }
  return {
    instanceId:           instance.instanceId,
    blueprintId:          instance.blueprintId,
    blueprintName:        instance.blueprint ? instance.blueprint.name : null,
    state:                BLUEPRINT_STATE_NAME[instance.state] || 'unknown',
    rootEid:              instance.rootEid,
    memberCount:          instance.memberCount,
    relationsApplied:     instance.relationsApplied,
    hierarchyApplied:     instance.hierarchyApplied,
    instantiatedAtFrame:  instance.instantiatedAtFrame,
    disposedAtFrame:      instance.disposedAtFrame,
    members,
  };
}

/* ------------------------------------------------------------------ */
/* 9. FRAME LIFECYCLE                                                 */
/* ------------------------------------------------------------------ */

/**
 * Advances the blueprint system frame counter. Called once per frame.
 */
export function tickBlueprints(frameNumber) {
  if (typeof frameNumber === 'number') BlueprintState.frame = frameNumber;
  else BlueprintState.frame++;
}

/* ------------------------------------------------------------------ */
/* 10. BUILT-IN BLUEPRINTS                                            */
/* ------------------------------------------------------------------ */

(function _declareBuiltinBlueprints() {

  /* ---------------- BASE OUTDOOR SCENE ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.OUTDOOR_SCENE,
    name: 'outdoor_scene',
    description: 'Base outdoor lighting: sun, sky light, ambient, GI probe grid, AO outdoor volume.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    members: [
      { prefabId: PREFAB.SUN,         memberName: 'sun' },
      { prefabId: PREFAB.SKY_LIGHT,   memberName: 'sky_light' },
      { prefabId: PREFAB.AMBIENT,     memberName: 'ambient' },
      { prefabId: PREFAB.GI_PROBE,    memberName: 'gi_probe_center' },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME, memberName: 'ao_outdoor' },
      { prefabId: PREFAB.CAMERA,      memberName: 'camera' },
    ],
    edges: [
      { fromMember: 1, toMember: 0, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 4, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- INTERIOR ROOM ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.INTERIOR_ROOM,
    name: 'interior_room',
    description: 'Interior room: lamp, window shaft, indoor GI volume, indoor AO volume.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.INDOOR_LIGHT,
    members: [
      { prefabId: PREFAB.INTERIOR_LAMP,       memberName: 'lamp' },
      { prefabId: PREFAB.WINDOW_SHAFT,        memberName: 'window_shaft' },
      { prefabId: PREFAB.GI_VOLUME_INDOOR,    memberName: 'gi_indoor' },
      { prefabId: PREFAB.AO_INDOOR_VOLUME,    memberName: 'ao_indoor' },
      { prefabId: PREFAB.AO_INK_CONTROLLER,   memberName: 'ink' },
    ],
    edges: [
      { fromMember: 0, toMember: 4, kind: REF.AO_VOLUME },
      { fromMember: 1, toMember: 4, kind: REF.AO_VOLUME },
    ],
  });

  /* ---------------- TRANSITION ZONE ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.TRANSITION_ZONE,
    name: 'transition_zone',
    description: 'Indoor↔outdoor portal: GI portal doorway, small indoor volume, small outdoor volume.',
    lifetime: PREFAB_LIFETIME.SCENE,
    members: [
      { prefabId: PREFAB.GI_PORTAL_DOORWAY,   memberName: 'portal' },
      { prefabId: PREFAB.GI_VOLUME_INDOOR,    memberName: 'gi_indoor' },
      { prefabId: PREFAB.GI_VOLUME_OUTDOOR,   memberName: 'gi_outdoor' },
    ],
    edges: [
      { fromMember: 0, toMember: 1, kind: REF.GI_VOLUME },
      { fromMember: 0, toMember: 2, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- FOREST VALLEY (image 2) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.FOREST_VALLEY,
    name: 'forest_valley',
    description: 'Forest valley biome with bright canopy lighting, GI probes, AO outdoor volume.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.FOREST,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.0, colorG: 0.95, colorB: 0.75, intensity: 1.35 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.55, colorG: 0.80, colorB: 0.65, intensity: 0.55 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.25, colorG: 0.35, colorB: 0.22, intensity: 0.20 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -8, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 8, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
      { prefabId: PREFAB.BIO_LUMINESCENT,     memberName: 'bio_glow',
        overrides: { Transform: { x: -6, y: 1.5, z: 2 } } },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 6, kind: REF.GI_VOLUME },
      { fromMember: 4, toMember: 6, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 6, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- SNOW TUNDRA (image 3) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.SNOW_TUNDRA,
    name: 'snow_tundra',
    description: 'Snow tundra biome with cool sun, aurora, snow AO, snow GI probes.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.SNOW,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 0.92, colorG: 0.96, colorB: 1.00, intensity: 1.15 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.65, colorG: 0.78, colorB: 0.92, intensity: 0.55 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.18, colorG: 0.22, colorB: 0.32, intensity: 0.20 } } },
      { prefabId: PREFAB.AURORA,              memberName: 'aurora' },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -10, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 10, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
    ],
    edges: [
      { fromMember: 3, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 4, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 6, toMember: 7, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- DESERT CANYON (image 1 + image 5) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.DESERT_CANYON,
    name: 'desert_canyon',
    description: 'Desert canyon biome with warm sun, canyon bounce, turquoise water caustics.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.DESERT | TAG2.CANYON,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.86, colorB: 0.62, intensity: 1.30 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.42, colorG: 0.72, colorB: 0.94, intensity: 0.45 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.35, colorG: 0.28, colorB: 0.18, intensity: 0.18 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -12, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 12, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_water',
        overrides: { GIProbeRef: { worldX: 0, worldY: 0.5, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
      { prefabId: PREFAB.CAUSTIC,             memberName: 'caustic',
        overrides: { Transform: { x: 0, y: 0.1, z: 0 } } },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 4, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 6, toMember: 7, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- COASTAL BAY (image 5) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.COASTAL_BAY,
    name: 'coastal_bay',
    description: 'Coastal bay with warm sun, cool sky light, water caustics, coastal GI.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.SEA | TAG2.COASTAL,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.94, colorB: 0.82, intensity: 1.30 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.20, colorG: 0.56, colorB: 0.90, intensity: 0.50 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.20, colorG: 0.30, colorB: 0.40, intensity: 0.18 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -10, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 10, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_water',
        overrides: { GIProbeRef: { worldX: 0, worldY: 0.5, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
      { prefabId: PREFAB.CAUSTIC,             memberName: 'caustic',
        overrides: { Transform: { x: 0, y: 0.1, z: 0 } } },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 4, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 6, toMember: 7, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- ORBITAL SPACE (image 4) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.ORBITAL_SPACE,
    name: 'orbital_space',
    description: 'Orbital space scene with low ambient, sharp directional sun, star-tinted sky.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.NIGHT,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.95, colorB: 0.85, intensity: 2.20 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.20, colorG: 0.30, colorB: 0.55, intensity: 0.25 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.05, colorG: 0.07, colorB: 0.12, intensity: 0.10 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 4, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- SUNSET TOMBSTONE (image 6) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.SUNSET_TOMBSTONE,
    name: 'sunset_tombstone',
    description: 'Sunset tombstone scene with warm directional sun, mint-green magic glow, GI probes.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.55, colorB: 0.35, intensity: 1.40 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.55, colorG: 0.40, colorB: 0.55, intensity: 0.45 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.30, colorG: 0.20, colorB: 0.30, intensity: 0.18 } } },
      { prefabId: PREFAB.MAGIC_GLOW,          memberName: 'glow',
        overrides: { Transform: { x: 0, y: 2, z: 0 },
                     LightRef: { colorR: 0.40, colorG: 1.00, colorB: 0.60, intensity: 2.5 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero' },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -8, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 8, worldY: 4, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 4, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 7, kind: REF.GI_VOLUME },
      { fromMember: 6, toMember: 7, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- PASTEL PORTRAIT (image 7) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.PASTEL_PORTRAIT,
    name: 'pastel_portrait',
    description: 'Pastel portrait scene with soft pink/lavender ambient, gentle sun, soft rim.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.92, colorB: 0.98, intensity: 0.85 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.85, colorG: 0.75, colorB: 0.92, intensity: 0.55 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.60, colorG: 0.50, colorB: 0.70, intensity: 0.30 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero',
        overrides: { GIPalette: { styleId: 6, satBias: 0.75, hueBias: 0.0,
                                  ambientColorR: 0.85, ambientColorG: 0.75, ambientColorB: 0.92,
                                  lerpRate: 3.2, enabled: 1 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor',
        overrides: { AOVolumeRef: { style: 5, intensity: 0.7, radius: 1.6 } } },
      { prefabId: PREFAB.AO_CEL_CONTROLLER,   memberName: 'cel',
        overrides: { AOCelBands: { enabled: 1, bandCount: 3, bandSoftness: 0.40 } } },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 4, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- MAGIC CASTER (image 8) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.MAGIC_CASTER,
    name: 'magic_caster',
    description: 'Magic caster scene with dark base, golden glow core, strong rim, cel AO.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.NIGHT,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 0.90, colorG: 0.80, colorB: 0.55, intensity: 0.70 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.15, colorG: 0.20, colorB: 0.12, intensity: 0.30 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.10, colorG: 0.12, colorB: 0.08, intensity: 0.15 } } },
      { prefabId: PREFAB.MAGIC_GLOW,          memberName: 'glow_core',
        overrides: { Transform: { x: 0, y: 1.2, z: 0 },
                     LightRef: { colorR: 1.00, colorG: 0.85, colorB: 0.45, intensity: 4.0, range: 10 } } },
      { prefabId: PREFAB.CAUSTIC,             memberName: 'caustic_ring',
        overrides: { Transform: { x: 0, y: 0.05, z: 0 },
                     LightRef: { colorR: 1.00, colorG: 0.75, colorB: 0.35, intensity: 1.6, range: 5 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero',
        overrides: { GIPalette: { styleId: 7, satBias: 1.25, hueBias: 0.0,
                                  ambientColorR: 0.20, ambientColorG: 0.22, ambientColorB: 0.18,
                                  lerpRate: 3.2, enabled: 1 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor',
        overrides: { AOVolumeRef: { style: 3, intensity: 1.1, radius: 1.4 } } },
      { prefabId: PREFAB.AO_INK_CONTROLLER,   memberName: 'ink' },
      { prefabId: PREFAB.AO_CEL_CONTROLLER,   memberName: 'cel',
        overrides: { AOCelBands: { enabled: 1, bandCount: 4, bandSoftness: 0.05 } } },
    ],
    edges: [
      { fromMember: 3, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 4, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 5, toMember: 6, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- FLOWER FIELD (image 9) ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.FLOWER_FIELD,
    name: 'flower_field',
    description: 'Flower field with bright magenta/violet accents, deep blue sky light, backlit rim.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    members: [
      { prefabId: PREFAB.SUN,                 memberName: 'sun',
        overrides: { LightRef: { colorR: 1.00, colorG: 0.90, colorB: 0.75, intensity: 1.20 } } },
      { prefabId: PREFAB.SKY_LIGHT,           memberName: 'sky_light',
        overrides: { LightRef: { colorR: 0.30, colorG: 0.55, colorB: 0.90, intensity: 0.55 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.45, colorG: 0.30, colorB: 0.50, intensity: 0.22 } } },
      { prefabId: PREFAB.GI_HERO_PROBE,       memberName: 'gi_hero',
        overrides: { GIPalette: { styleId: 8, satBias: 1.20, hueBias: 0.0,
                                  ambientColorR: 0.85, ambientColorG: 0.55, ambientColorB: 0.75,
                                  lerpRate: 3.2, enabled: 1 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_a',
        overrides: { GIProbeRef: { worldX: -10, worldY: 3, worldZ: 0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'gi_probe_b',
        overrides: { GIProbeRef: { worldX: 10, worldY: 3, worldZ: 0 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor' },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
      { fromMember: 3, toMember: 6, kind: REF.GI_VOLUME },
      { fromMember: 4, toMember: 6, kind: REF.GI_VOLUME },
      { fromMember: 5, toMember: 6, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- FIRE CAMP ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.FIRE_CAMP,
    name: 'fire_camp',
    description: 'Small campfire scene with fire light, warm ambient, small AO volume.',
    lifetime: PREFAB_LIFETIME.SCENE,
    members: [
      { prefabId: PREFAB.FIRE_LIGHT,          memberName: 'fire',
        overrides: { Transform: { x: 0, y: 0.8, z: 0 },
                     LightRef: { colorR: 1.0, colorG: 0.55, colorB: 0.15, intensity: 2.8, range: 14 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.15, colorG: 0.10, colorB: 0.05, intensity: 0.18 } } },
      { prefabId: PREFAB.AO_OUTDOOR_VOLUME,   memberName: 'ao_outdoor',
        overrides: { AOVolumeRef: { intensity: 0.85, radius: 1.2 } } },
      { prefabId: PREFAB.CONTACT_SHADOW,      memberName: 'contact' },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.LIGHT_SOURCE },
    ],
  });

  /* ---------------- NEON ALLEY ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.NEON_ALLEY,
    name: 'neon_alley',
    description: 'Neon alley with magenta neon panel, cool ambient, AO ink outline, strong rim.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.INDOOR_LIGHT,
    members: [
      { prefabId: PREFAB.NEON_LIGHT,          memberName: 'neon_main',
        overrides: { Transform: { x: 0, y: 2.5, z: 0 },
                     LightRef: { colorR: 0.9, colorG: 0.4, colorB: 0.7, intensity: 3.5 } } },
      { prefabId: PREFAB.NEON_LIGHT,          memberName: 'neon_secondary',
        overrides: { Transform: { x: 3, y: 1.5, z: 0 },
                     LightRef: { colorR: 0.4, colorG: 0.7, colorB: 1.0, intensity: 2.0,
                                 width: 0.8, height: 0.15 } } },
      { prefabId: PREFAB.AMBIENT,             memberName: 'ambient',
        overrides: { LightRef: { colorR: 0.15, colorG: 0.10, colorB: 0.20, intensity: 0.25 } } },
      { prefabId: PREFAB.AO_INDOOR_VOLUME,    memberName: 'ao_indoor',
        overrides: { AOVolumeRef: { style: 3, intensity: 1.0, radius: 1.2 } } },
      { prefabId: PREFAB.AO_INK_CONTROLLER,   memberName: 'ink',
        overrides: { AOInkOutline: { enabled: 1, thickness: 0.004, strength: 0.55 } } },
      { prefabId: PREFAB.REFLECTION_PROBE,    memberName: 'refl',
        overrides: { GIReflectionProbe: { positionY: 2.0, radius: 12.0, resolution: 128, updateInterval: 120 } } },
    ],
    edges: [
      { fromMember: 3, toMember: 2, kind: REF.GI_VOLUME },
    ],
  });

  /* ---------------- WINDOW ROOM ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.WINDOW_ROOM,
    name: 'window_room',
    description: 'Interior room with warm window shaft, cool indoor lamp, soft AO.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.INDOOR_LIGHT,
    members: [
      { prefabId: PREFAB.WINDOW_SHAFT,        memberName: 'shaft' },
      { prefabId: PREFAB.INTERIOR_LAMP,       memberName: 'lamp',
        overrides: { Transform: { x: 3, y: 2.2, z: -2 } } },
      { prefabId: PREFAB.GI_VOLUME_INDOOR,    memberName: 'gi_indoor' },
      { prefabId: PREFAB.AO_INDOOR_VOLUME,    memberName: 'ao_indoor' },
      { prefabId: PREFAB.AO_INK_CONTROLLER,   memberName: 'ink' },
    ],
    edges: [
      { fromMember: 0, toMember: 3, kind: REF.AO_VOLUME },
      { fromMember: 1, toMember: 3, kind: REF.AO_VOLUME },
    ],
  });

  /* ---------------- CAVE INTERIOR ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.CAVE_INTERIOR,
    name: 'cave_interior',
    description: 'Cave interior with a single portal, dim indoor light, strong ink AO.',
    lifetime: PREFAB_LIFETIME.SCENE,
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.INDOOR_LIGHT,
    members: [
      { prefabId: PREFAB.GI_PORTAL_CAVE,      memberName: 'portal' },
      { prefabId: PREFAB.INTERIOR_LAMP,       memberName: 'lamp',
        overrides: { Transform: { x: 0, y: 2.0, z: -3 },
                     LightRef: { colorR: 0.7, colorG: 0.9, colorB: 0.6, intensity: 1.2, range: 8 } } },
      { prefabId: PREFAB.GI_VOLUME_INDOOR,    memberName: 'gi_indoor',
        overrides: { GIIndoor: { wallOcclusion: 0.95, curtainTransmission: 0.05 } } },
      { prefabId: PREFAB.AO_INDOOR_VOLUME,    memberName: 'ao_indoor',
        overrides: { AOVolumeRef: { intensity: 1.2, radius: 1.2 } } },
      { prefabId: PREFAB.AO_INK_CONTROLLER,   memberName: 'ink',
        overrides: { AOInkOutline: { enabled: 1, thickness: 0.005, strength: 0.6 } } },
    ],
    edges: [
      { fromMember: 0, toMember: 2, kind: REF.GI_VOLUME },
      { fromMember: 1, toMember: 3, kind: REF.AO_VOLUME },
    ],
  });

  /* ---------------- DEBUG LIGHT PROBE ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.DEBUG_LIGHT_PROBE,
    name: 'debug_light_probe',
    description: 'Debug scene with a single point light and a marker.',
    lifetime: PREFAB_LIFETIME.TRANSIENT,
    ttlFrames: 60,
    members: [
      { prefabId: PREFAB.POINT_LIGHT,         memberName: 'light',
        overrides: { LightRef: { colorR: 1.0, colorG: 0.8, colorB: 0.4, intensity: 3.0, range: 6 } } },
      { prefabId: PREFAB.DEBUG_MARKER,        memberName: 'marker' },
    ],
    edges: [],
  });

  /* ---------------- DEBUG GI GRID ---------------- */
  defineBlueprint({
    id:   BLUEPRINT.DEBUG_GI_GRID,
    name: 'debug_gi_grid',
    description: 'Debug scene with a small GI probe grid.',
    lifetime: PREFAB_LIFETIME.TRANSIENT,
    ttlFrames: 120,
    members: [
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_0',
        overrides: { GIProbeRef: { worldX: -4, worldY: 2, worldZ: -4 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_1',
        overrides: { GIProbeRef: { worldX:  0, worldY: 2, worldZ: -4 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_2',
        overrides: { GIProbeRef: { worldX:  4, worldY: 2, worldZ: -4 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_3',
        overrides: { GIProbeRef: { worldX: -4, worldY: 2, worldZ:  0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_4',
        overrides: { GIProbeRef: { worldX:  0, worldY: 2, worldZ:  0 } } },
      { prefabId: PREFAB.GI_PROBE,            memberName: 'probe_5',
        overrides: { GIProbeRef: { worldX:  4, worldY: 2, worldZ:  0 } } },
    ],
    edges: [],
  });
})();

/* ------------------------------------------------------------------ */
/* 11. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getBlueprintReport() {
  const list = [];
  for (let i = 0; i < MAX_BLUEPRINTS; i++) {
    const bp = _blueprints[i];
    if (!bp) continue;
    list.push({
      id:          bp.id,
      name:        bp.name,
      description: bp.description,
      lifetime:    bp.lifetime,
      memberCount: bp.memberCount,
      edgeCount:   bp.edgeCount,
      tagMask:     bp.tagMask,
      tag2Mask:    bp.tag2Mask,
    });
  }

  return {
    frame:                 BlueprintState.frame,
    totalDefinitions:      BlueprintState.totalDefinitions,
    totalInstantiations:   BlueprintState.totalInstantiations,
    totalDisposals:        BlueprintState.totalDisposals,
    totalRejected:         BlueprintState.totalRejected,
    totalMembersSpawned:   BlueprintState.totalMembersSpawned,
    totalEdgesBuilt:       BlueprintState.totalEdgesBuilt,
    totalHierarchyBuilt:   BlueprintState.totalHierarchyBuilt,
    liveInstanceCount:     BlueprintState.liveInstanceCount,
    peakLiveInstances:     BlueprintState.peakLiveInstances,
    avgInstantiateMs:      BlueprintState.avgInstantiateMs,
    avgDisposeMs:          BlueprintState.avgDisposeMs,
    lastInstantiateMs:     BlueprintState.lastInstantiateMs,
    lastDisposeMs:         BlueprintState.lastDisposeMs,
    registeredBlueprints:  getBlueprintCount(),
    blueprintCapacity:     MAX_BLUEPRINTS,
    instanceCapacity:      MAX_INSTANCES,
    blueprints:            list,
    perfTier:              PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 12. REGISTRATION                                                   */
/* ------------------------------------------------------------------ */

/**
 * The blueprint module composes existing components; it does not declare
 * its own. No-op, present for API symmetry.
 */
export function registerBlueprintComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 13. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Disposes every live instance and resets counters. The blueprint
 * DEFINITIONS remain registered and are not affected.
 */
export function resetBlueprintState() {
  disposeAllInstances();

  for (let i = 0; i < MAX_INSTANCES; i++) _instances[i].reset();
  _instanceCount = 0;
  _instanceIdCounter = 0;
  _instanceById.clear();

  BlueprintState.frame = 0;
  BlueprintState.totalDefinitions = 0;
  BlueprintState.totalInstantiations = 0;
  BlueprintState.totalDisposals = 0;
  BlueprintState.totalRejected = 0;
  BlueprintState.totalMembersSpawned = 0;
  BlueprintState.totalEdgesBuilt = 0;
  BlueprintState.totalHierarchyBuilt = 0;
  BlueprintState.peakLiveInstances = 0;
  BlueprintState.liveInstanceCount = 0;
  BlueprintState.lastInstantiateMs = 0;
  BlueprintState.lastDisposeMs = 0;
  BlueprintState.avgInstantiateMs = 0;
  BlueprintState.avgDisposeMs = 0;
}

/* ------------------------------------------------------------------ */
/* 14. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Enums
  BLUEPRINT,
  BLUEPRINT_NAME,
  BLUEPRINT_STATE,
  BLUEPRINT_STATE_NAME,

  // Constants
  MAX_BLUEPRINTS,
  MAX_INSTANCES,
  MAX_MEMBERS_PER_BLUEPRINT,
  MAX_EDGES_PER_BLUEPRINT,
  MAX_HIERARCHY_PER_BLUEPRINT,

  // Descriptor + instance classes
  BlueprintDescriptor,
  BlueprintMember,
  BlueprintEdge,
  BlueprintInstance,

  // Module state
  BlueprintState,

  // Definition
  defineBlueprint,
  getBlueprint,
  getBlueprintByName,
  getBlueprintCount,

  // Instantiation
  instantiateBlueprint,
  disposeBlueprintInstance,
  disposeAllInstancesOfBlueprint,
  disposeAllInstances,

  // Query
  getInstance,
  getInstanceMemberEid,
  getInstanceMemberByName,
  getInstanceRootEid,
  forEachInstance,
  forEachInstanceMember,
  getInstanceSnapshot,

  // Frame lifecycle
  tickBlueprints,

  // Diagnostics
  getBlueprintReport,

  // Registration
  registerBlueprintComponents,

  // Reset
  resetBlueprintState,
};

export default _defaultExport;