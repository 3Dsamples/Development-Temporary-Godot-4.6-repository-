// File : 015
// name : src/ecs/015_scn_Relations.js
// description : Canonical relationship-component module for the scene ECS
//               world of the anime lighting stack on Android mobile. Provides
//               the entity relationship graph primitives that every subsystem
//               uses to express "entity A is attached to entity B", "light L
//               drives shadow caster C", "GI probe P belongs to volume V",
//               "composite light K owns member lights M1..Mn", and every
//               other directed or bidirectional edge the engine needs.
//
//               Two complementary relationship forms are provided:
//
//                 1. HIERARCHY (parent/child)
//                    A single-parent tree with a fixed-capacity child edge
//                    list per parent. Supports fast upward walk
//                    (`getParent`, `getRoot`) and downward iteration
//                    (`forEachChild`, `walkSubtree`). Parents can be
//                    reparented at runtime without allocation.
//
//                 2. TYPED REFERENCES (unidirectional / bidirectional edges)
//                    A per-entity slot table where each entity can hold up
//                    to MAX_REFERENCES_PER_ENTITY typed edges. Each edge is
//                    (kind, targetEid, weight). Reference kinds are declared
//                    in the REF enum (LIGHT_SOURCE, SHADOW_LIGHT,
//                    GI_VOLUME, AO_VOLUME, CAMERA_FOLLOW, CASTER_OF,
//                    COMPOSITE_MEMBER, COMPOSITE_ANCHOR, PORTAL_PEER,
//                    PROBE_GRID, LOD_GROUP, STREAMING_CHUNK, BIOME_PARENT,
//                    MATERIAL_OVERRIDE, DEBUG_TARGET, etc.).
//
//               Both forms are stored as fixed-capacity SoA typed arrays
//               sized once to MAX_ENTITIES = 100000. No Map/Set on the hot
//               path. All traversal helpers are allocation-free and
//               depth-safe with cycle detection.
//
//               Integration:
//                 • 002_lgt_LightComponents.js — LightComposite, LightShadow
//                 • 003_lgt_ShadowComponents.js — ShadowCasterRef, ShadowReceiverRef
//                 • 004_lgt_GIComponents.js — GIPortal, GIVolume
//                 • 005_lgt_AOComponents.js — AOVolumeRef
//                 • 010_scn_ECSWorld.js — world handle
//                 • 011_scn_BiteCSAdapter.js — entity lifecycle
//                 • 012_scn_ComponentRegistry.js — catalog
//                 • 013_scn_ComponentTypes.js — numeric ids
//                 • 014_scn_Tags.js — tag helpers
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every typed array sized once at construction.
// best for : Guaranteeing that every subsystem in the anime lighting stack
//            can express and traverse entity relationships with a single
//            typed-array operation — no allocations, no maps, no drift —
//            while keeping the graph depth-safe and cycle-safe.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
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

import {
  getECSWorld,
} from './010_scn_ECSWorld.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of children per parent entity.
 */
export const MAX_CHILDREN_PER_ENTITY =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 32 :
                                 16;

/**
 * Maximum number of typed references per entity.
 */
export const MAX_REFERENCES_PER_ENTITY =
  PERF_TIER_LOCAL === 'HIGH'   ? 16 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 12 :
                                 8;

/**
 * Maximum depth of the hierarchy before cycle detection kicks in.
 */
export const MAX_HIERARCHY_DEPTH = 256;

/**
 * Reserved "no entity" sentinel. Use instead of -1 in any field that
 * stores an entity id so downstream comparisons stay type-stable.
 */
export const NULL_ENTITY = -1;

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * Relationship kinds. Each typed reference edge carries one of these
 * kinds so downstream systems can filter edges without extra lookups.
 */
export const REF = Object.freeze({
  NONE:                0,

  // Lighting
  LIGHT_SOURCE:        1,    // shadow/emissive → light
  SHADOW_LIGHT:        2,    // shadow atlas → light
  CLUSTER_LIGHT:       3,    // cluster cell → light

  // Shadow
  CASTER_OF:           4,    // caster → light (which light it casts for)
  RECEIVER_OF:         5,    // receiver → light
  SHADOW_PROXY:        6,    // mesh → shadow proxy
  CASCADE_OWNER:       7,    // cascade → light

  // GI
  GI_VOLUME:           8,    // probe → volume
  GI_PORTAL_PEER:      9,    // portal → peer portal
  PROBE_GRID:         10,    // probe → grid anchor
  REFLECTION_PROBE:   11,    // entity → reflection probe
  LIGHTFIELD_OWNER:   12,    // sample → lightfield owner

  // AO
  AO_VOLUME:          13,    // entity → AO volume
  AO_CONTACT:         14,    // mesh → contact AO volume

  // Camera
  CAMERA_FOLLOW:      15,    // camera → follow target
  CAMERA_LOOK_AT:     16,    // camera → look-at target
  CAMERA_DOF:         17,    // camera → DOF target

  // Composite / grouping
  COMPOSITE_MEMBER:   18,    // member light → composite anchor
  COMPOSITE_ANCHOR:   19,    // anchor → member light
  GROUP_MEMBER:       20,    // generic group member → group anchor

  // Streaming / LOD
  STREAMING_CHUNK:    21,    // entity → chunk owner
  LOD_GROUP:          22,    // entity → LOD group anchor
  LOD_PARENT:         23,    // entity → LOD parent
  IMPOSTOR_OF:        24,    // impostor → source

  // Biome / environment
  BIOME_PARENT:       25,    // entity → biome anchor
  ENV_OWNER:          26,    // entity → environment owner

  // Material
  MATERIAL_OVERRIDE:  27,    // mesh → material override entity
  DECAL_OWNER:        28,    // decal → host mesh

  // Debug
  DEBUG_TARGET:       29,    // debug view → target
  DEBUG_INSPECT:      30,    // HUD → inspected entity

  // Generic
  USER:               31,
  COUNT:              32,
});

export const REF_NAME = Object.freeze([
  'none',
  'light_source',
  'shadow_light',
  'cluster_light',
  'caster_of',
  'receiver_of',
  'shadow_proxy',
  'cascade_owner',
  'gi_volume',
  'gi_portal_peer',
  'probe_grid',
  'reflection_probe',
  'lightfield_owner',
  'ao_volume',
  'ao_contact',
  'camera_follow',
  'camera_look_at',
  'camera_dof',
  'composite_member',
  'composite_anchor',
  'group_member',
  'streaming_chunk',
  'lod_group',
  'lod_parent',
  'impostor_of',
  'biome_parent',
  'env_owner',
  'material_override',
  'decal_owner',
  'debug_target',
  'debug_inspect',
  'user',
]);

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * Parent — single-parent pointer per entity. -1 if no parent.
 */
export const Parent = {
  eid: new Int32Array(MAX_ENTITIES).fill(NULL_ENTITY),
  ref: new Uint8Array(MAX_ENTITIES),   // REF kind of the parent edge
};

/**
 * Hierarchy state — per-entity metadata about its position in the tree.
 */
export const HierarchyState = {
  depth:          new Uint16Array(MAX_ENTITIES),
  childCount:     new Uint16Array(MAX_ENTITIES),
  subtreeSize:    new Uint32Array(MAX_ENTITIES),
  dirty:          new Uint8Array(MAX_ENTITIES),
  lastTouchFrame: new Uint32Array(MAX_ENTITIES),
};

/**
 * Child edge list — flat fixed-capacity storage.
 * childEid[parent * MAX_CHILDREN_PER_ENTITY + slot] = child entity id, or -1.
 */
export const Children = {
  childEid: new Int32Array(MAX_ENTITIES * MAX_CHILDREN_PER_ENTITY).fill(NULL_ENTITY),
  childRef: new Uint8Array(MAX_ENTITIES * MAX_CHILDREN_PER_ENTITY),
};

/**
 * Typed reference edge list — flat fixed-capacity storage.
 * refKind[eid * MAX_REFERENCES_PER_ENTITY + slot] = REF kind
 * refTarget[eid * MAX_REFERENCES_PER_ENTITY + slot] = target entity id
 * refWeight[eid * MAX_REFERENCES_PER_ENTITY + slot] = float weight
 */
export const References = {
  refKind:   new Uint8Array(MAX_ENTITIES * MAX_REFERENCES_PER_ENTITY),
  refTarget: new Int32Array(MAX_ENTITIES * MAX_REFERENCES_PER_ENTITY).fill(NULL_ENTITY),
  refWeight: new Float32Array(MAX_ENTITIES * MAX_REFERENCES_PER_ENTITY),
  refCount:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Reverse reference list — keeps incoming references for fast "who points
 * to me?" queries. Stored per-entity as a bounded ring of source ids.
 * Full reverse graph is not always needed; MAX_REVERSE_REFERENCES caps
 * the memory cost.
 */
export const MAX_REVERSE_REFERENCES =
  PERF_TIER_LOCAL === 'HIGH'   ? 16 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 12 :
                                 8;

export const ReverseReferences = {
  srcEid:   new Int32Array(MAX_ENTITIES * MAX_REVERSE_REFERENCES).fill(NULL_ENTITY),
  srcKind:  new Uint8Array(MAX_ENTITIES * MAX_REVERSE_REFERENCES),
  count:    new Uint8Array(MAX_ENTITIES),
};

/* ------------------------------------------------------------------ */
/* 3. ECS COMPONENT BUNDLE                                            */
/* ------------------------------------------------------------------ */

export const RELATION_COMPONENTS = Object.freeze({
  Parent,
  HierarchyState,
  Children,
  References,
  ReverseReferences,
});

/* ------------------------------------------------------------------ */
/* 4. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _isValidEntity(eid) {
  return typeof eid === 'number' && eid >= 0 && eid < MAX_ENTITIES;
}

function _nextFrame() {
  // Frame counter bumped externally by tickRelations; return current.
  return RelationsState.frame;
}

/**
 * Shared state for the relationship module (frame counter, stats).
 */
export const RelationsState = {
  frame:           0,
  totalAttach:     0,
  totalDetach:     0,
  totalReparents:  0,
  totalReferences: 0,
  totalReverse:    0,
  cycleDetections: 0,
  depthOverflows:  0,
};

/* ------------------------------------------------------------------ */
/* 5. HIERARCHY OPERATIONS                                            */
/* ------------------------------------------------------------------ */

/**
 * Attaches `childEid` under `parentEid`. Detaches the child from any
 * previous parent first. Returns true on success.
 *
 * Cycle-safe: refuses to attach if `parentEid` is a descendant of
 * `childEid`.
 */
export function attachChild(parentEid, childEid, refKind = REF.NONE) {
  if (!_isValidEntity(parentEid) || !_isValidEntity(childEid)) return false;
  if (parentEid === childEid) return false;

  // Cycle check — walk up from parent to root; if we meet childEid,
  // attaching would create a loop.
  let cursor = parentEid;
  let depth = 0;
  while (cursor !== NULL_ENTITY && depth < MAX_HIERARCHY_DEPTH) {
    if (cursor === childEid) {
      RelationsState.cycleDetections++;
      const log = _safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE,
        `[015_scn_Relations] cycle prevented: attach ${childEid} under ${parentEid}`);
      return false;
    }
    cursor = Parent.eid[cursor];
    depth++;
  }
  if (depth >= MAX_HIERARCHY_DEPTH) {
    RelationsState.depthOverflows++;
    return false;
  }

  // Detach from previous parent first.
  const prevParent = Parent.eid[childEid];
  if (prevParent !== NULL_ENTITY) {
    _removeChildSlot(prevParent, childEid);
  }

  // Find a free slot in the parent's children array.
  const base = parentEid * MAX_CHILDREN_PER_ENTITY;
  let slot = -1;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    if (Children.childEid[base + i] === NULL_ENTITY) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[015_scn_Relations] parent ${parentEid} children cap reached`);
    return false;
  }

  // Write the edge.
  Children.childEid[base + slot]  = childEid;
  Children.childRef[base + slot]  = refKind >>> 0;
  Parent.eid[childEid]            = parentEid;
  Parent.ref[childEid]            = refKind >>> 0;
  HierarchyState.childCount[parentEid]++;
  HierarchyState.lastTouchFrame[parentEid] = RelationsState.frame;

  // Update depth of child subtree.
  _recomputeDepth(childEid);

  RelationsState.totalAttach++;
  return true;
}

/**
 * Detaches `childEid` from its current parent. Returns true if it was
 * attached.
 */
export function detachChild(childEid) {
  if (!_isValidEntity(childEid)) return false;
  const parentEid = Parent.eid[childEid];
  if (parentEid === NULL_ENTITY) return false;
  _removeChildSlot(parentEid, childEid);
  Parent.eid[childEid] = NULL_ENTITY;
  Parent.ref[childEid] = REF.NONE;
  RelationsState.totalDetach++;
  return true;
}

/**
 * Reparents a child under a new parent. Returns true on success.
 */
export function reparentChild(newParentEid, childEid, refKind = REF.NONE) {
  if (!_isValidEntity(newParentEid) || !_isValidEntity(childEid)) return false;
  RelationsState.totalReparents++;
  return attachChild(newParentEid, childEid, refKind);
}

function _removeChildSlot(parentEid, childEid) {
  const base = parentEid * MAX_CHILDREN_PER_ENTITY;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    if (Children.childEid[base + i] === childEid) {
      Children.childEid[base + i] = NULL_ENTITY;
      Children.childRef[base + i] = REF.NONE;
      if (HierarchyState.childCount[parentEid] > 0) {
        HierarchyState.childCount[parentEid]--;
      }
      HierarchyState.lastTouchFrame[parentEid] = RelationsState.frame;
      return true;
    }
  }
  return false;
}

function _recomputeDepth(eid) {
  let depth = 0;
  let cursor = eid;
  while (cursor !== NULL_ENTITY && depth < MAX_HIERARCHY_DEPTH) {
    cursor = Parent.eid[cursor];
    depth++;
  }
  HierarchyState.depth[eid] = depth;
  HierarchyState.dirty[eid] = 1;

  // Recompute depths of children (iterative, bounded).
  const base = eid * MAX_CHILDREN_PER_ENTITY;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    const child = Children.childEid[base + i];
    if (child === NULL_ENTITY) continue;
    HierarchyState.depth[child] = depth + 1;
    HierarchyState.dirty[child] = 1;
  }
}

/**
 * Returns the parent entity id, or NULL_ENTITY.
 */
export function getParent(eid) {
  if (!_isValidEntity(eid)) return NULL_ENTITY;
  return Parent.eid[eid];
}

/**
 * Returns the root ancestor of the given entity.
 */
export function getRoot(eid) {
  if (!_isValidEntity(eid)) return NULL_ENTITY;
  let cursor = eid;
  let depth = 0;
  while (depth < MAX_HIERARCHY_DEPTH) {
    const p = Parent.eid[cursor];
    if (p === NULL_ENTITY) return cursor;
    cursor = p;
    depth++;
  }
  return NULL_ENTITY;
}

/**
 * Returns the depth of the given entity in the hierarchy (0 = root).
 */
export function getDepth(eid) {
  if (!_isValidEntity(eid)) return 0;
  return HierarchyState.depth[eid];
}

/**
 * Returns the number of direct children.
 */
export function getChildCount(eid) {
  if (!_isValidEntity(eid)) return 0;
  return HierarchyState.childCount[eid];
}

/**
 * Iterates every direct child of an entity. Allocation-free.
 */
export function forEachChild(parentEid, fn, ctx) {
  if (!_isValidEntity(parentEid) || typeof fn !== 'function') return 0;
  const base = parentEid * MAX_CHILDREN_PER_ENTITY;
  let count = 0;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    const child = Children.childEid[base + i];
    if (child === NULL_ENTITY) continue;
    fn.call(ctx, child, Children.childRef[base + i]);
    count++;
  }
  return count;
}

/**
 * Returns an array of direct children (allocates).
 */
export function getChildren(parentEid) {
  if (!_isValidEntity(parentEid)) return [];
  const out = [];
  const base = parentEid * MAX_CHILDREN_PER_ENTITY;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    const child = Children.childEid[base + i];
    if (child !== NULL_ENTITY) out.push(child);
  }
  return out;
}

/**
 * Walks the entire subtree rooted at `rootEid` in depth-first order.
 * Visits each entity exactly once, with cycle-safe depth cap.
 */
export function walkSubtree(rootEid, fn, ctx) {
  if (!_isValidEntity(rootEid) || typeof fn !== 'function') return 0;
  let visited = 0;
  const stack = _getWalkStack();
  let sp = 0;
  stack[sp++] = rootEid;
  const seen = _getWalkSeen();

  while (sp > 0) {
    const eid = stack[--sp];
    if (seen[eid] === RelationsState.frame + 1) continue;
    seen[eid] = RelationsState.frame + 1;
    fn.call(ctx, eid);
    visited++;

    const base = eid * MAX_CHILDREN_PER_ENTITY;
    for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
      const child = Children.childEid[base + i];
      if (child === NULL_ENTITY) continue;
      if (sp >= stack.length) break;
      stack[sp++] = child;
    }
  }
  return visited;
}

/* Pre-allocated traversal scratch — sized once, never resized. */
const _walkStack = new Int32Array(MAX_HIERARCHY_DEPTH * 2);
const _walkSeen  = new Uint32Array(MAX_ENTITIES);
function _getWalkStack() { return _walkStack; }
function _getWalkSeen()  { return _walkSeen; }

/**
 * Counts the entities in the subtree rooted at `rootEid`.
 */
export function countSubtree(rootEid) {
  let count = 0;
  walkSubtree(rootEid, () => { count++; });
  return count;
}

/**
 * Returns an array of every descendant (depth-first).
 */
export function getDescendants(rootEid) {
  const out = [];
  walkSubtree(rootEid, (eid) => { out.push(eid); });
  return out;
}

/* ------------------------------------------------------------------ */
/* 6. TYPED REFERENCE OPERATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * Adds a typed reference from `srcEid` to `dstEid`. Returns the slot
 * index (0..MAX-1), or -1 on failure.
 *
 * The reverse reference is also recorded on `dstEid`.
 */
export function setReference(srcEid, dstEid, kind, weight) {
  if (!_isValidEntity(srcEid) || !_isValidEntity(dstEid)) return -1;
  if (typeof kind !== 'number' || kind <= 0 || kind >= REF.COUNT) return -1;

  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];

  // Check for existing edge of same kind+target and update it.
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] === kind &&
        References.refTarget[base + i] === dstEid) {
      References.refWeight[base + i] = weight !== undefined ? weight : 1.0;
      return i;
    }
  }

  // Append a new edge.
  if (count >= MAX_REFERENCES_PER_ENTITY) {
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[015_scn_Relations] entity ${srcEid} reference cap reached`);
    return -1;
  }

  References.refKind[base + count]   = kind >>> 0;
  References.refTarget[base + count] = dstEid;
  References.refWeight[base + count] = weight !== undefined ? weight : 1.0;
  References.refCount[srcEid] = count + 1;

  // Record reverse reference.
  _addReverseReference(dstEid, srcEid, kind);

  RelationsState.totalReferences++;
  return count;
}

/**
 * Removes a typed reference edge.
 */
export function clearReference(srcEid, dstEid, kind) {
  if (!_isValidEntity(srcEid) || !_isValidEntity(dstEid)) return false;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] === kind &&
        References.refTarget[base + i] === dstEid) {
      // Shift remaining edges down.
      for (let j = i; j < count - 1; j++) {
        References.refKind[base + j]   = References.refKind[base + j + 1];
        References.refTarget[base + j] = References.refTarget[base + j + 1];
        References.refWeight[base + j] = References.refWeight[base + j + 1];
      }
      const last = base + count - 1;
      References.refKind[last]   = REF.NONE;
      References.refTarget[last] = NULL_ENTITY;
      References.refWeight[last] = 0;
      References.refCount[srcEid] = count - 1;

      // Remove reverse reference.
      _removeReverseReference(dstEid, srcEid, kind);
      return true;
    }
  }
  return false;
}

/**
 * Removes every typed reference from `srcEid`.
 */
export function clearAllReferences(srcEid) {
  if (!_isValidEntity(srcEid)) return false;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    const dst = References.refTarget[base + i];
    const kind = References.refKind[base + i];
    if (dst !== NULL_ENTITY) _removeReverseReference(dst, srcEid, kind);
  }
  for (let i = 0; i < MAX_REFERENCES_PER_ENTITY; i++) {
    References.refKind[base + i]   = REF.NONE;
    References.refTarget[base + i] = NULL_ENTITY;
    References.refWeight[base + i] = 0;
  }
  References.refCount[srcEid] = 0;
  return true;
}

/**
 * Resolves the first reference of the given kind, or NULL_ENTITY.
 */
export function resolveReference(srcEid, kind) {
  if (!_isValidEntity(srcEid)) return NULL_ENTITY;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] === kind) {
      return References.refTarget[base + i];
    }
  }
  return NULL_ENTITY;
}

/**
 * Returns true if srcEid has a reference of the given kind to dstEid.
 */
export function hasReference(srcEid, dstEid, kind) {
  if (!_isValidEntity(srcEid) || !_isValidEntity(dstEid)) return false;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] === kind &&
        References.refTarget[base + i] === dstEid) {
      return true;
    }
  }
  return false;
}

/**
 * Returns the weight of the reference, or 0 if not present.
 */
export function getReferenceWeight(srcEid, dstEid, kind) {
  if (!_isValidEntity(srcEid) || !_isValidEntity(dstEid)) return 0;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] === kind &&
        References.refTarget[base + i] === dstEid) {
      return References.refWeight[base + i];
    }
  }
  return 0;
}

/**
 * Iterates every reference from `srcEid`. Allocation-free.
 */
export function forEachReference(srcEid, fn, ctx) {
  if (!_isValidEntity(srcEid) || typeof fn !== 'function') return 0;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  for (let i = 0; i < count; i++) {
    const kind = References.refKind[base + i];
    const dst  = References.refTarget[base + i];
    const w    = References.refWeight[base + i];
    if (kind === REF.NONE || dst === NULL_ENTITY) continue;
    fn.call(ctx, kind, dst, w);
  }
  return count;
}

/**
 * Iterates every reference of a specific kind from `srcEid`.
 */
export function forEachReferenceOfKind(srcEid, kind, fn, ctx) {
  if (!_isValidEntity(srcEid) || typeof fn !== 'function') return 0;
  const base = srcEid * MAX_REFERENCES_PER_ENTITY;
  const count = References.refCount[srcEid];
  let hits = 0;
  for (let i = 0; i < count; i++) {
    if (References.refKind[base + i] !== kind) continue;
    const dst = References.refTarget[base + i];
    if (dst === NULL_ENTITY) continue;
    fn.call(ctx, dst, References.refWeight[base + i]);
    hits++;
  }
  return hits;
}

/* ------------------------------------------------------------------ */
/* 7. REVERSE REFERENCE OPERATIONS                                    */
/* ------------------------------------------------------------------ */

function _addReverseReference(targetEid, srcEid, kind) {
  const base = targetEid * MAX_REVERSE_REFERENCES;
  const count = ReverseReferences.count[targetEid];
  if (count >= MAX_REVERSE_REFERENCES) return false;
  ReverseReferences.srcEid[base + count] = srcEid;
  ReverseReferences.srcKind[base + count] = kind >>> 0;
  ReverseReferences.count[targetEid] = count + 1;
  RelationsState.totalReverse++;
  return true;
}

function _removeReverseReference(targetEid, srcEid, kind) {
  const base = targetEid * MAX_REVERSE_REFERENCES;
  const count = ReverseReferences.count[targetEid];
  for (let i = 0; i < count; i++) {
    if (ReverseReferences.srcEid[base + i] === srcEid &&
        ReverseReferences.srcKind[base + i] === kind) {
      for (let j = i; j < count - 1; j++) {
        ReverseReferences.srcEid[base + j] = ReverseReferences.srcEid[base + j + 1];
        ReverseReferences.srcKind[base + j] = ReverseReferences.srcKind[base + j + 1];
      }
      const last = base + count - 1;
      ReverseReferences.srcEid[last] = NULL_ENTITY;
      ReverseReferences.srcKind[last] = REF.NONE;
      ReverseReferences.count[targetEid] = count - 1;
      return true;
    }
  }
  return false;
}

/**
 * Iterates every entity that references `targetEid`. Allocation-free.
 */
export function forEachReverseReference(targetEid, fn, ctx) {
  if (!_isValidEntity(targetEid) || typeof fn !== 'function') return 0;
  const base = targetEid * MAX_REVERSE_REFERENCES;
  const count = ReverseReferences.count[targetEid];
  for (let i = 0; i < count; i++) {
    const src = ReverseReferences.srcEid[base + i];
    if (src === NULL_ENTITY) continue;
    fn.call(ctx, src, ReverseReferences.srcKind[base + i]);
  }
  return count;
}

/**
 * Returns the number of reverse references to `targetEid`.
 */
export function getReverseReferenceCount(targetEid) {
  if (!_isValidEntity(targetEid)) return 0;
  return ReverseReferences.count[targetEid];
}

/* ------------------------------------------------------------------ */
/* 8. CLEANUP ON ENTITY DESTRUCTION                                   */
/* ------------------------------------------------------------------ */

/**
 * Detaches every relationship associated with an entity. Called by the
 * engine loop before destroying an entity so the graph stays clean.
 */
export function detachAllRelations(eid) {
  if (!_isValidEntity(eid)) return false;

  // Detach from parent.
  detachChild(eid);

  // Detach every child.
  const base = eid * MAX_CHILDREN_PER_ENTITY;
  for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
    const child = Children.childEid[base + i];
    if (child === NULL_ENTITY) continue;
    Parent.eid[child] = NULL_ENTITY;
    Parent.ref[child] = REF.NONE;
    Children.childEid[base + i] = NULL_ENTITY;
    Children.childRef[base + i] = REF.NONE;
  }
  HierarchyState.childCount[eid] = 0;

  // Clear outbound references.
  clearAllReferences(eid);

  // Clear inbound reverse references by walking every entity that has
  // us in its References list. Since we don't have a full reverse graph,
  // we also handle the reverse list directly.
  const rBase = eid * MAX_REVERSE_REFERENCES;
  const rCount = ReverseReferences.count[eid];
  for (let i = 0; i < rCount; i++) {
    const src = ReverseReferences.srcEid[rBase + i];
    const kind = ReverseReferences.srcKind[rBase + i];
    if (src !== NULL_ENTITY) {
      clearReference(src, eid, kind);
    }
  }
  ReverseReferences.count[eid] = 0;

  return true;
}

/* ------------------------------------------------------------------ */
/* 9. LIGHTING-SPECIFIC CONVENIENCE                                   */
/* ------------------------------------------------------------------ */

/**
 * Binds a shadow caster to a light. Both directions are recorded so
 * "which lights have casters?" and "which lights does this caster serve?"
 * are O(1).
 */
export function bindCasterToLight(casterEid, lightEid) {
  return setReference(casterEid, lightEid, REF.CASTER_OF, 1.0) >= 0;
}

/**
 * Binds a shadow receiver to a light.
 */
export function bindReceiverToLight(receiverEid, lightEid) {
  return setReference(receiverEid, lightEid, REF.RECEIVER_OF, 1.0) >= 0;
}

/**
 * Binds a GI probe to its owning volume.
 */
export function bindProbeToVolume(probeEid, volumeEid) {
  return setReference(probeEid, volumeEid, REF.GI_VOLUME, 1.0) >= 0;
}

/**
 * Binds a portal to its peer portal (bidirectional).
 */
export function bindPortalPeer(portalA, portalB) {
  const a = setReference(portalA, portalB, REF.GI_PORTAL_PEER, 1.0) >= 0;
  const b = setReference(portalB, portalA, REF.GI_PORTAL_PEER, 1.0) >= 0;
  return a && b;
}

/**
 * Binds an AO volume to an entity (e.g. indoor room).
 */
export function bindAOVolume(entityEid, aoVolumeEid) {
  return setReference(entityEid, aoVolumeEid, REF.AO_VOLUME, 1.0) >= 0;
}

/**
 * Binds a camera to its follow target.
 */
export function bindCameraFollow(cameraEid, targetEid) {
  return setReference(cameraEid, targetEid, REF.CAMERA_FOLLOW, 1.0) >= 0;
}

/**
 * Registers a light as a member of a composite light group.
 */
export function bindCompositeMember(memberEid, anchorEid) {
  const a = setReference(memberEid, anchorEid, REF.COMPOSITE_MEMBER, 1.0) >= 0;
  const b = setReference(anchorEid, memberEid, REF.COMPOSITE_ANCHOR, 1.0) >= 0;
  return a && b;
}

/**
 * Returns the anchor of a composite light, given a member.
 */
export function getCompositeAnchor(memberEid) {
  return resolveReference(memberEid, REF.COMPOSITE_MEMBER);
}

/**
 * Iterates every member light of a composite, given the anchor.
 */
export function forEachCompositeMember(anchorEid, fn, ctx) {
  return forEachReferenceOfKind(anchorEid, REF.COMPOSITE_ANCHOR, fn, ctx);
}

/* ------------------------------------------------------------------ */
/* 10. LOD / STREAMING CONVENIENCE                                    */
/* ------------------------------------------------------------------ */

/**
 * Registers an entity as a member of an LOD group.
 */
export function bindLODGroup(memberEid, groupAnchorEid) {
  return setReference(memberEid, groupAnchorEid, REF.LOD_GROUP, 1.0) >= 0;
}

/**
 * Registers an entity as belonging to a streaming chunk.
 */
export function bindStreamingChunk(entityEid, chunkEid) {
  return setReference(entityEid, chunkEid, REF.STREAMING_CHUNK, 1.0) >= 0;
}

/**
 * Registers an impostor as representing a source mesh.
 */
export function bindImpostor(impostorEid, sourceEid) {
  return setReference(impostorEid, sourceEid, REF.IMPOSTOR_OF, 1.0) >= 0;
}

/* ------------------------------------------------------------------ */
/* 11. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the relationship system frame counter. Called once per frame.
 */
export function tickRelations(frameNumber) {
  if (typeof frameNumber === 'number') RelationsState.frame = frameNumber;
  else RelationsState.frame++;
}

/* ------------------------------------------------------------------ */
/* 12. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * Registers every relationship component in the runtime component
 * registry so downstream systems can discover them by name.
 */
export function registerRelationComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'Parent',              component: Parent,              category: 19, subsystem: 1, dependencies: [] },
    { name: 'HierarchyState',      component: HierarchyState,      category: 19, subsystem: 1, dependencies: [] },
    { name: 'Children',            component: Children,            category: 19, subsystem: 1, dependencies: [] },
    { name: 'References',          component: References,          category: 19, subsystem: 1, dependencies: [] },
    { name: 'ReverseReferences',   component: ReverseReferences,   category: 19, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 13. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getRelationStats() {
  let rootCount = 0;
  let totalChildEdges = 0;
  let totalRefEdges = 0;
  let totalReverseEdges = 0;

  for (let i = 0; i < MAX_ENTITIES; i++) {
    if (Parent.eid[i] === NULL_ENTITY) {
      // Potentially a root (only if it has children).
      if (HierarchyState.childCount[i] > 0) rootCount++;
    }
    totalChildEdges += HierarchyState.childCount[i];
    totalRefEdges += References.refCount[i];
    totalReverseEdges += ReverseReferences.count[i];
  }

  return {
    frame:                RelationsState.frame,
    maxEntities:          MAX_ENTITIES,
    maxChildren:          MAX_CHILDREN_PER_ENTITY,
    maxReferences:        MAX_REFERENCES_PER_ENTITY,
    maxReverse:           MAX_REVERSE_REFERENCES,
    rootCount,
    totalChildEdges,
    totalRefEdges,
    totalReverseEdges,
    totalAttach:          RelationsState.totalAttach,
    totalDetach:          RelationsState.totalDetach,
    totalReparents:       RelationsState.totalReparents,
    totalReferences:      RelationsState.totalReferences,
    totalReverse:         RelationsState.totalReverse,
    cycleDetections:      RelationsState.cycleDetections,
    depthOverflows:       RelationsState.depthOverflows,
    perfTier:             PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 14. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every relationship in the world. Use for full teardown / restart.
 */
export function resetAllRelations() {
  Parent.eid.fill(NULL_ENTITY);
  Parent.ref.fill(0);
  HierarchyState.depth.fill(0);
  HierarchyState.childCount.fill(0);
  HierarchyState.subtreeSize.fill(0);
  HierarchyState.dirty.fill(0);
  HierarchyState.lastTouchFrame.fill(0);
  Children.childEid.fill(NULL_ENTITY);
  Children.childRef.fill(0);
  References.refKind.fill(0);
  References.refTarget.fill(NULL_ENTITY);
  References.refWeight.fill(0);
  References.refCount.fill(0);
  ReverseReferences.srcEid.fill(NULL_ENTITY);
  ReverseReferences.srcKind.fill(0);
  ReverseReferences.count.fill(0);
  _walkSeen.fill(0);

  RelationsState.frame = 0;
  RelationsState.totalAttach = 0;
  RelationsState.totalDetach = 0;
  RelationsState.totalReparents = 0;
  RelationsState.totalReferences = 0;
  RelationsState.totalReverse = 0;
  RelationsState.cycleDetections = 0;
  RelationsState.depthOverflows = 0;
}

/* ------------------------------------------------------------------ */
/* 15. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_CHILDREN_PER_ENTITY,
  MAX_REFERENCES_PER_ENTITY,
  MAX_REVERSE_REFERENCES,
  MAX_HIERARCHY_DEPTH,
  NULL_ENTITY,

  // Components
  Parent,
  HierarchyState,
  Children,
  References,
  ReverseReferences,
  RELATION_COMPONENTS,

  // Enums
  REF,
  REF_NAME,

  // State
  RelationsState,

  // Hierarchy ops
  attachChild,
  detachChild,
  reparentChild,
  getParent,
  getRoot,
  getDepth,
  getChildCount,
  forEachChild,
  getChildren,
  walkSubtree,
  countSubtree,
  getDescendants,

  // Typed references
  setReference,
  clearReference,
  clearAllReferences,
  resolveReference,
  hasReference,
  getReferenceWeight,
  forEachReference,
  forEachReferenceOfKind,

  // Reverse references
  forEachReverseReference,
  getReverseReferenceCount,

  // Cleanup
  detachAllRelations,

  // Lighting convenience
  bindCasterToLight,
  bindReceiverToLight,
  bindProbeToVolume,
  bindPortalPeer,
  bindAOVolume,
  bindCameraFollow,
  bindCompositeMember,
  getCompositeAnchor,
  forEachCompositeMember,

  // LOD / streaming convenience
  bindLODGroup,
  bindStreamingChunk,
  bindImpostor,

  // Frame lifecycle
  tickRelations,

  // Registration
  registerRelationComponents,

  // Diagnostics
  getRelationStats,

  // Reset
  resetAllRelations,
};

export default _defaultExport;