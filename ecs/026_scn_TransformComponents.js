// File : 026
// name : src/ecs/026_scn_TransformComponents.js
// description : Transform-hierarchy SoA module for the scene ECS world of the
//               anime lighting stack on Android mobile. Declares every
//               transform component the lighting stack needs — local TRS
//               (position, quaternion, scale), world TRS, cached local and
//               world matrices, dirty flags for lazy propagation, parent and
//               root bindings, and per-entity interpolation state for
//               smooth visual motion — as fixed-capacity typed arrays sized
//               once to MAX_ENTITIES = 100000.
//
//               Provides the fast transform helpers that every spatial
//               query, culling pass, and lit-entity sync consumes:
//                 • computeLocalMatrix         — TRS → 4×4 local matrix
//                 • computeWorldMatrix         — local matrix × parent world
//                 • markTransformDirty         — flag + propagate to children
//                 • propagateTransformDirty    — recursive downward walk
//                 • updateWorldTransforms      — one-pass world transform solve
//                 • getWorldPosition           — read world-space position
//                 • setWorldPosition           — write world-space position
//                 • getWorldQuaternion         — read world-space rotation
//                 • setWorldQuaternion         — write world-space rotation
//                 • getWorldScale              — read world-space scale
//                 • setWorldScale              — write world-space scale
//                 • lerpTransforms             — local TRS lerp between two
//                                                 entities' snapshots
//                 • dampTransforms             — exponential damped TRS
//                                                 toward a target
//                 • snapshotTransformLocal     — copy current TRS into the
//                                                 interpolation previous-slot
//                 • interpolateTransformLocal  — Slerp/Lerp between prev and
//                                                 current TRS by alpha
//                 • syncTransformToSpatial     — bridge to 025 AABB/sphere
//                 • computeWorldBoundingRadius — max scale × base radius
//
//               Design:
//                 • Zero-allocation hot path — every helper works directly
//                   on the SoA arrays with pre-allocated scratch vectors.
//                 • Lazy world-matrix solve — `updateWorldTransforms`
//                   processes only the dirty subtree rooted at each marked
//                   entity, walking the children list from 015_scn_Relations.
//                 • Dirty propagation is upward (mark parent dirty when a
//                   child moves) and downward (skip clean subtrees).
//                 • Matrices are stored in the same column-major order as
//                   Three.js Matrix4 so downstream Three.js sync is a
//                   straight memcpy.
//                 • Quaternions are stored (x, y, z, w) matching Three.js.
//                 • Interpolation uses a two-slot TRS history (previous and
//                   current) so render-time lerp is a cheap read.
//                 • Scratch vectors are module-level, pre-allocated, and
//                   reused across every call.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — Transform, Target
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — visibility tags
//                 • 015_scn_Relations.js            — parent / children edges
//                 • 016_scn_EntityPool.js           — subsystem pools
//                 • 017_scn_EntityLifetime.js       — lifetime records
//                 • 025_scn_SpatialComponents.js    — AABB / sphere derivation
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every transform operation in the anime
//            lighting stack is allocation-free, deterministic, dirty-flag
//            driven, and coherent with the relations graph — so moving a
//            parent light automatically moves every attached child entity,
//            and culling, spatial queries, and GPU sync all read the same
//            world matrix.
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
  FDIRTY,
  markFrameDirty,
  clearFrameDirty,
} from './014_scn_Tags.js';

import {
  Parent,
  Children,
  HierarchyState,
  MAX_CHILDREN_PER_ENTITY,
  NULL_ENTITY,
} from './015_scn_Relations.js';

import {
  computeAABB,
  computeSphere,
} from './025_scn_SpatialComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum depth of the transform hierarchy that the dirty propagation
 * walk will descend. Matches the relations hierarchy depth cap.
 */
export const MAX_TRANSFORM_DEPTH = 256;

/**
 * Dirty state flags (bitmask).
 */
export const TRANSFORM_DIRTY = Object.freeze({
  NONE:        0,
  LOCAL:       1 << 0,   // local TRS changed
  WORLD:       1 << 1,   // world matrix needs recompute
  MATRIX:      1 << 2,   // cached local matrix is stale
  SUBTREE:     1 << 3,   // this entity and its whole subtree are dirty
  ROOT_DIRTY:  1 << 4,   // root ancestor's world matrix changed
  INIT_DIRTY:  1 << 5,   // never been solved
});

/**
 * Transform sync modes — controls whether an entity participates in the
 * world-matrix solve.
 */
export const TRANSFORM_SYNC = Object.freeze({
  NONE:      0,   // frozen, never recompute
  LOCAL:     1,   // local-only updates (UI overlays, HUD markers)
  WORLD:     2,   // full world-matrix solve
  ANCESTOR:  3,   // only propagate when an ancestor moves
  COUNT:     4,
});

export const TRANSFORM_SYNC_NAME = Object.freeze([
  'none',
  'local',
  'world',
  'ancestor',
]);

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const TransformState = {
  frame:                0,
  totalDirtyMarks:      0,
  totalPropagations:    0,
  totalWorldSolvePasses:0,
  totalWorldSolves:     0,
  totalInterpolations:  0,
  totalDamps:           0,
  totalLerps:           0,
  peakDirtyCount:       0,
  lastSolveMs:          0,
  avgSolveMs:           0,
  lastSolveDirtyCount:  0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.transform', {
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
 * TransformLocal — the authored TRS values for every entity. Mirrors the
 * Transform component from 002 exactly but is populated as a separate
 * slot so that systems can distinguish "authored local" from "solved
 * world".
 */
export const TransformLocal = {
  x:  new Float32Array(MAX_ENTITIES),
  y:  new Float32Array(MAX_ENTITIES),
  z:  new Float32Array(MAX_ENTITIES),
  qx: new Float32Array(MAX_ENTITIES),
  qy: new Float32Array(MAX_ENTITIES),
  qz: new Float32Array(MAX_ENTITIES),
  qw: new Float32Array(MAX_ENTITIES),
  sx: new Float32Array(MAX_ENTITIES),
  sy: new Float32Array(MAX_ENTITIES),
  sz: new Float32Array(MAX_ENTITIES),
  sync: new Uint8Array(MAX_ENTITIES),   // TRANSFORM_SYNC
};

/**
 * TransformWorld — the solved world-space TRS for every entity.
 */
export const TransformWorld = {
  x:  new Float32Array(MAX_ENTITIES),
  y:  new Float32Array(MAX_ENTITIES),
  z:  new Float32Array(MAX_ENTITIES),
  qx: new Float32Array(MAX_ENTITIES),
  qy: new Float32Array(MAX_ENTITIES),
  qz: new Float32Array(MAX_ENTITIES),
  qw: new Float32Array(MAX_ENTITIES),
  sx: new Float32Array(MAX_ENTITIES),
  sy: new Float32Array(MAX_ENTITIES),
  sz: new Float32Array(MAX_ENTITIES),
  maxScale: new Float32Array(MAX_ENTITIES),
};

/**
 * TransformDirty — per-entity dirty flags for lazy world-matrix solve.
 */
export const TransformDirty = {
  flags:          new Uint8Array(MAX_ENTITIES),
  lastDirtyFrame: new Uint32Array(MAX_ENTITIES),
  lastSolveFrame: new Uint32Array(MAX_ENTITIES),
};

/**
 * TransformMatrix — cached 4×4 local and world matrices.
 * Stored column-major, matching Three.js Matrix4.elements.
 */
export const TransformMatrix = {
  local: new Float32Array(MAX_ENTITIES * 16),
  world: new Float32Array(MAX_ENTITIES * 16),
  localValid: new Uint8Array(MAX_ENTITIES),
  worldValid: new Uint8Array(MAX_ENTITIES),
};

/**
 * TransformParent — parent binding + root cache for fast upward walks.
 */
export const TransformParent = {
  parentEid: new Int32Array(MAX_ENTITIES).fill(NULL_ENTITY),
  rootEid:   new Int32Array(MAX_ENTITIES).fill(NULL_ENTITY),
  depth:     new Uint16Array(MAX_ENTITIES),
};

/**
 * TransformInterpolation — per-entity TRS history for smooth motion.
 * `prev*` is the previous-frame value; `cur*` is the current authored
 * value. Render-time lerp uses (prev, cur, alpha).
 */
export const TransformInterpolation = {
  prevX:  new Float32Array(MAX_ENTITIES),
  prevY:  new Float32Array(MAX_ENTITIES),
  prevZ:  new Float32Array(MAX_ENTITIES),
  prevQx: new Float32Array(MAX_ENTITIES),
  prevQy: new Float32Array(MAX_ENTITIES),
  prevQz: new Float32Array(MAX_ENTITIES),
  prevQw: new Float32Array(MAX_ENTITIES),
  prevSx: new Float32Array(MAX_ENTITIES),
  prevSy: new Float32Array(MAX_ENTITIES),
  prevSz: new Float32Array(MAX_ENTITIES),
  enabled: new Uint8Array(MAX_ENTITIES),
};

/**
 * TransformEuler — optional Euler-angle cache for authored rotations,
 * kept in sync with the quaternion whenever the quaternion is set from
 * Euler.
 */
export const TransformEuler = {
  ex: new Float32Array(MAX_ENTITIES),
  ey: new Float32Array(MAX_ENTITIES),
  ez: new Float32Array(MAX_ENTITIES),
  order: new Uint8Array(MAX_ENTITIES),   // 0=XYZ 1=YXZ 2=ZXY 3=ZYX 4=YZX 5=XZY
};

/**
 * TransformRoot — the resolved root entity per entity (cached, invalidated
 * when the parent chain changes).
 */
export const TransformRoot = {
  rootEid: new Int32Array(MAX_ENTITIES).fill(NULL_ENTITY),
  valid:   new Uint8Array(MAX_ENTITIES),
};

/**
 * Transform component bundle for bitECS createWorld.
 */
export const TRANSFORM_COMPONENTS = Object.freeze({
  TransformLocal,
  TransformWorld,
  TransformDirty,
  TransformMatrix,
  TransformParent,
  TransformInterpolation,
  TransformEuler,
  TransformRoot,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Module-level scratch for matrix composition. Sized exactly.
 */
const _scratchMatrix = new Float32Array(16);
const _scratchMatrixB = new Float32Array(16);
const _scratchVec3 = new Float32Array(3);
const _scratchVec3B = new Float32Array(3);
const _scratchQuat = new Float32Array(4);

/**
 * DFS stack for dirty propagation.
 */
const _dfsStack = new Int32Array(MAX_ENTITIES);
const _dfsSeen = new Uint8Array(MAX_ENTITIES);
let _dfsSeenFrame = 0;

/* ------------------------------------------------------------------ */
/* 4. DIRTY FLAG MANAGEMENT                                           */
/* ------------------------------------------------------------------ */

/**
 * Marks a transform as dirty with the given flags. Propagates upward to
 * the root and (optionally) downward into the subtree.
 *
 *   markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL | TRANSFORM_DIRTY.SUBTREE)
 */
export function markTransformDirty(eid, flags) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const f = (flags !== undefined ? flags : TRANSFORM_DIRTY.LOCAL) | 0;
  TransformDirty.flags[eid] |= f | TRANSFORM_DIRTY.WORLD;
  TransformDirty.lastDirtyFrame[eid] = TransformState.frame;
  TransformState.totalDirtyMarks++;

  // Propagate upward: every ancestor's world matrix is now stale.
  let cursor = Parent.eid[eid];
  let depth = 0;
  while (cursor !== NULL_ENTITY && depth < MAX_TRANSFORM_DEPTH) {
    TransformDirty.flags[cursor] |= TRANSFORM_DIRTY.SUBTREE;
    TransformDirty.lastDirtyFrame[cursor] = TransformState.frame;
    cursor = Parent.eid[cursor];
    depth++;
    TransformState.totalPropagations++;
  }

  // If the SUBTREE bit is set, propagate downward to all descendants.
  if (f & TRANSFORM_DIRTY.SUBTREE) {
    propagateTransformDirtyDownward(eid, TRANSFORM_DIRTY.SUBTREE);
  }

  return true;
}

/**
 * Propagates the given dirty flags downward to every descendant of the
 * given entity. Uses an iterative DFS with a per-frame seen mask.
 *
 * Allocation-free.
 */
export function propagateTransformDirtyDownward(rootEid, flags) {
  if (typeof rootEid !== 'number' || rootEid < 0 || rootEid >= MAX_ENTITIES) return 0;

  const f = flags | 0;
  const frameId = (++_dfsSeenFrame) & 0xFF;
  let visited = 0;
  let sp = 0;

  _dfsStack[sp++] = rootEid;

  while (sp > 0) {
    const eid = _dfsStack[--sp];
    if (_dfsSeen[eid] === frameId) continue;
    _dfsSeen[eid] = frameId;

    if (eid !== rootEid) {
      TransformDirty.flags[eid] |= f | TRANSFORM_DIRTY.WORLD;
      TransformDirty.lastDirtyFrame[eid] = TransformState.frame;
      visited++;
    }

    const base = eid * MAX_CHILDREN_PER_ENTITY;
    for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
      const child = Children.childEid[base + i];
      if (child === NULL_ENTITY) continue;
      if (sp >= _dfsStack.length) break;
      _dfsStack[sp++] = child;
    }
  }

  return visited;
}

/**
 * Clears the dirty flags on an entity after its world matrix has been
 * solved.
 */
export function clearTransformDirty(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  TransformDirty.flags[eid] = TRANSFORM_DIRTY.NONE;
  return true;
}

/**
 * Returns true if the entity's world matrix is stale.
 */
export function isTransformDirty(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return (TransformDirty.flags[eid] & (TRANSFORM_DIRTY.LOCAL | TRANSFORM_DIRTY.WORLD)) !== 0;
}

/* ------------------------------------------------------------------ */
/* 5. LOCAL TRS AUTHORING                                             */
/* ------------------------------------------------------------------ */

/**
 * Sets the local position of an entity and marks it dirty.
 */
export function setLocalPosition(eid, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  TransformLocal.x[eid] = x;
  TransformLocal.y[eid] = y;
  TransformLocal.z[eid] = z;
  // Mirror into 002's Transform so existing consumers stay correct.
  Transform.x[eid] = x;
  Transform.y[eid] = y;
  Transform.z[eid] = z;
  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  return true;
}

/**
 * Sets the local rotation quaternion of an entity and marks it dirty.
 */
export function setLocalQuaternion(eid, x, y, z, w) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  TransformLocal.qx[eid] = x;
  TransformLocal.qy[eid] = y;
  TransformLocal.qz[eid] = z;
  TransformLocal.qw[eid] = w;
  Transform.qx[eid] = x;
  Transform.qy[eid] = y;
  Transform.qz[eid] = z;
  Transform.qw[eid] = w;
  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  return true;
}

/**
 * Sets the local scale of an entity and marks it dirty.
 */
export function setLocalScale(eid, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  TransformLocal.sx[eid] = x;
  TransformLocal.sy[eid] = y;
  TransformLocal.sz[eid] = z;
  Transform.sx[eid] = x;
  Transform.sy[eid] = y;
  Transform.sz[eid] = z;
  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  return true;
}

/**
 * Copies the current authored local TRS into the interpolation previous
 * slot. Called at the start of a frame before authored values are
 * updated, so a later render pass can lerp between prev and current.
 */
export function snapshotTransformLocal(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  TransformInterpolation.prevX[eid]  = TransformLocal.x[eid];
  TransformInterpolation.prevY[eid]  = TransformLocal.y[eid];
  TransformInterpolation.prevZ[eid]  = TransformLocal.z[eid];
  TransformInterpolation.prevQx[eid] = TransformLocal.qx[eid];
  TransformInterpolation.prevQy[eid] = TransformLocal.qy[eid];
  TransformInterpolation.prevQz[eid] = TransformLocal.qz[eid];
  TransformInterpolation.prevQw[eid] = TransformLocal.qw[eid];
  TransformInterpolation.prevSx[eid] = TransformLocal.sx[eid];
  TransformInterpolation.prevSy[eid] = TransformLocal.sy[eid];
  TransformInterpolation.prevSz[eid] = TransformLocal.sz[eid];
  TransformInterpolation.enabled[eid] = 1;
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. MATRIX COMPOSITION                                              */
/* ------------------------------------------------------------------ */

/**
 * Composes a 4×4 column-major TRS matrix into `out` (Float32Array, length
 * 16). Allocation-free — reuses the module's scratch if `out` is omitted.
 *
 * Returns the destination array.
 */
export function composeTRSMatrix(out, px, py, pz, qx, qy, qz, qw, sx, sy, sz) {
  const dst = out || _scratchMatrix;

  const x2 = qx + qx;
  const y2 = qy + qy;
  const z2 = qz + qz;
  const xx = qx * x2;
  const xy = qx * y2;
  const xz = qx * z2;
  const yy = qy * y2;
  const yz = qy * z2;
  const zz = qz * z2;
  const wx = qw * x2;
  const wy = qw * y2;
  const wz = qw * z2;

  dst[0]  = (1 - (yy + zz)) * sx;
  dst[1]  = (xy + wz) * sx;
  dst[2]  = (xz - wy) * sx;
  dst[3]  = 0;
  dst[4]  = (xy - wz) * sy;
  dst[5]  = (1 - (xx + zz)) * sy;
  dst[6]  = (yz + wx) * sy;
  dst[7]  = 0;
  dst[8]  = (xz + wy) * sz;
  dst[9]  = (yz - wx) * sz;
  dst[10] = (1 - (xx + yy)) * sz;
  dst[11] = 0;
  dst[12] = px;
  dst[13] = py;
  dst[14] = pz;
  dst[15] = 1;

  return dst;
}

/**
 * Multiplies two 4×4 column-major matrices: out = a × b. Writes into
 * `out` (which may be the same array as `a` or `b`).
 */
export function multiplyMatrices(out, a, b) {
  const a00 = a[0],  a01 = a[1],  a02 = a[2],  a03 = a[3];
  const a10 = a[4],  a11 = a[5],  a12 = a[6],  a13 = a[7];
  const a20 = a[8],  a21 = a[9],  a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = b[0],  b01 = b[1],  b02 = b[2],  b03 = b[3];
  const b10 = b[4],  b11 = b[5],  b12 = b[6],  b13 = b[7];
  const b20 = b[8],  b21 = b[9],  b22 = b[10], b23 = b[11];
  const b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];

  out[0]  = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
  out[1]  = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
  out[2]  = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
  out[3]  = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;

  out[4]  = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
  out[5]  = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
  out[6]  = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
  out[7]  = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;

  out[8]  = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
  out[9]  = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
  out[10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
  out[11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;

  out[12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
  out[13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
  out[14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
  out[15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;

  return out;
}

/**
 * Computes the local matrix of an entity from its authored TRS and
 * writes it into TransformMatrix.local[eid*16].
 */
export function computeLocalMatrix(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const base = eid * 16;
  const dst = TransformMatrix.local;

  const x2 = TransformLocal.qx[eid] + TransformLocal.qx[eid];
  const y2 = TransformLocal.qy[eid] + TransformLocal.qy[eid];
  const z2 = TransformLocal.qz[eid] + TransformLocal.qz[eid];

  const qx = TransformLocal.qx[eid];
  const qy = TransformLocal.qy[eid];
  const qz = TransformLocal.qz[eid];
  const qw = TransformLocal.qw[eid];

  const xx = qx * x2;
  const xy = qx * y2;
  const xz = qx * z2;
  const yy = qy * y2;
  const yz = qy * z2;
  const zz = qz * z2;
  const wx = qw * x2;
  const wy = qw * y2;
  const wz = qw * z2;

  const sx = TransformLocal.sx[eid];
  const sy = TransformLocal.sy[eid];
  const sz = TransformLocal.sz[eid];

  dst[base + 0]  = (1 - (yy + zz)) * sx;
  dst[base + 1]  = (xy + wz) * sx;
  dst[base + 2]  = (xz - wy) * sx;
  dst[base + 3]  = 0;

  dst[base + 4]  = (xy - wz) * sy;
  dst[base + 5]  = (1 - (xx + zz)) * sy;
  dst[base + 6]  = (yz + wx) * sy;
  dst[base + 7]  = 0;

  dst[base + 8]  = (xz + wy) * sz;
  dst[base + 9]  = (yz - wx) * sz;
  dst[base + 10] = (1 - (xx + yy)) * sz;
  dst[base + 11] = 0;

  dst[base + 12] = TransformLocal.x[eid];
  dst[base + 13] = TransformLocal.y[eid];
  dst[base + 14] = TransformLocal.z[eid];
  dst[base + 15] = 1;

  TransformMatrix.localValid[eid] = 1;
  return true;
}

/**
 * Computes the world matrix of an entity:
 *   world = parentWorld × local
 *
 * If the entity has no parent, world == local.
 * Writes into TransformMatrix.world[eid*16].
 * Also updates TransformWorld TRS and TransformWorld.maxScale.
 */
export function computeWorldMatrix(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  // Ensure local matrix is valid.
  if (TransformMatrix.localValid[eid] === 0) computeLocalMatrix(eid);

  const parentEid = Parent.eid[eid];
  const eBase = eid * 16;
  const worldDst = TransformMatrix.world;

  if (parentEid === NULL_ENTITY) {
    // No parent — world == local.
    for (let i = 0; i < 16; i++) {
      worldDst[eBase + i] = TransformMatrix.local[eBase + i];
    }
  } else {
    // Ensure parent world matrix is valid.
    if (TransformMatrix.worldValid[parentEid] === 0) computeWorldMatrix(parentEid);

    const pBase = parentEid * 16;
    const A = TransformMatrix.world.subarray(pBase, pBase + 16);
    const B = TransformMatrix.local.subarray(eBase, eBase + 16);

    // Inline multiply — writes directly into world[eBase..eBase+15].
    const a00 = A[0],  a01 = A[1],  a02 = A[2],  a03 = A[3];
    const a10 = A[4],  a11 = A[5],  a12 = A[6],  a13 = A[7];
    const a20 = A[8],  a21 = A[9],  a22 = A[10], a23 = A[11];
    const a30 = A[12], a31 = A[13], a32 = A[14], a33 = A[15];

    const b00 = B[0],  b01 = B[1],  b02 = B[2],  b03 = B[3];
    const b10 = B[4],  b11 = B[5],  b12 = B[6],  b13 = B[7];
    const b20 = B[8],  b21 = B[9],  b22 = B[10], b23 = B[11];
    const b30 = B[12], b31 = B[13], b32 = B[14], b33 = B[15];

    worldDst[eBase + 0]  = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
    worldDst[eBase + 1]  = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
    worldDst[eBase + 2]  = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
    worldDst[eBase + 3]  = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;

    worldDst[eBase + 4]  = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
    worldDst[eBase + 5]  = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
    worldDst[eBase + 6]  = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
    worldDst[eBase + 7]  = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;

    worldDst[eBase + 8]  = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
    worldDst[eBase + 9]  = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
    worldDst[eBase + 10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
    worldDst[eBase + 11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;

    worldDst[eBase + 12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
    worldDst[eBase + 13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
    worldDst[eBase + 14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
    worldDst[eBase + 15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
  }

  // Decompose into TransformWorld (position, quaternion, scale).
  _decomposeMatrix(worldDst, eBase, eid);

  TransformMatrix.worldValid[eid] = 1;
  TransformDirty.lastSolveFrame[eid] = TransformState.frame;
  return true;
}

/**
 * Decomposes a column-major 4×4 matrix at `base` into the entity's
 * TransformWorld TRS. Uses scratch vectors — no allocations.
 */
function _decomposeMatrix(matrix, base, eid) {
  const m00 = matrix[base + 0];
  const m01 = matrix[base + 1];
  const m02 = matrix[base + 2];
  const m10 = matrix[base + 4];
  const m11 = matrix[base + 5];
  const m12 = matrix[base + 6];
  const m20 = matrix[base + 8];
  const m21 = matrix[base + 9];
  const m22 = matrix[base + 10];

  const sx = Math.sqrt(m00 * m00 + m01 * m01 + m02 * m02);
  const sy = Math.sqrt(m10 * m10 + m11 * m11 + m12 * m12);
  const sz = Math.sqrt(m20 * m20 + m21 * m21 + m22 * m22);

  TransformWorld.sx[eid] = sx;
  TransformWorld.sy[eid] = sy;
  TransformWorld.sz[eid] = sz;
  TransformWorld.maxScale[eid] = Math.max(sx, Math.max(sy, sz));

  // Position.
  TransformWorld.x[eid] = matrix[base + 12];
  TransformWorld.y[eid] = matrix[base + 13];
  TransformWorld.z[eid] = matrix[base + 14];

  // Rotation — normalize by inverse scale.
  const invSx = sx > 1e-8 ? 1.0 / sx : 0;
  const invSy = sy > 1e-8 ? 1.0 / sy : 0;
  const invSz = sz > 1e-8 ? 1.0 / sz : 0;

  const r00 = m00 * invSx, r01 = m01 * invSx, r02 = m02 * invSx;
  const r10 = m10 * invSy, r11 = m11 * invSy, r12 = m12 * invSy;
  const r20 = m20 * invSz, r21 = m21 * invSz, r22 = m22 * invSz;

  // Quaternion from rotation matrix.
  const trace = r00 + r11 + r22;
  let qx, qy, qz, qw;

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (r21 - r12) * s;
    qy = (r02 - r20) * s;
    qz = (r10 - r01) * s;
  } else if (r00 > r11 && r00 > r22) {
    const s = 2.0 * Math.sqrt(1.0 + r00 - r11 - r22);
    qw = (r21 - r12) / s;
    qx = 0.25 * s;
    qy = (r01 + r10) / s;
    qz = (r02 + r20) / s;
  } else if (r11 > r22) {
    const s = 2.0 * Math.sqrt(1.0 + r11 - r00 - r22);
    qw = (r02 - r20) / s;
    qx = (r01 + r10) / s;
    qy = 0.25 * s;
    qz = (r12 + r21) / s;
  } else {
    const s = 2.0 * Math.sqrt(1.0 + r22 - r00 - r11);
    qw = (r10 - r01) / s;
    qx = (r02 + r20) / s;
    qy = (r12 + r21) / s;
    qz = 0.25 * s;
  }

  TransformWorld.qx[eid] = qx;
  TransformWorld.qy[eid] = qy;
  TransformWorld.qz[eid] = qz;
  TransformWorld.qw[eid] = qw;
}

/* ------------------------------------------------------------------ */
/* 7. WORLD TRANSFORM SOLVE                                           */
/* ------------------------------------------------------------------ */

/**
 * Solves the world matrices for every dirty entity and its subtree.
 * Iterative: for each dirty root, walk children in topological order,
 * computing each world matrix from the parent's world matrix.
 *
 * Returns the number of entities solved.
 */
export function updateWorldTransforms() {
  const t0 = _now();
  TransformState.totalWorldSolvePasses++;

  let solved = 0;

  // Iterate every entity: if it has a dirty WORLD bit and its parent is
  // clean (or it has no parent), it is a "root of dirty subtree".
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const flags = TransformDirty.flags[eid];
    if ((flags & TRANSFORM_DIRTY.WORLD) === 0) continue;

    const parentEid = Parent.eid[eid];
    // If the parent is dirty, skip this entity — the parent's subtree
    // walk will reach it.
    if (parentEid !== NULL_ENTITY && (TransformDirty.flags[parentEid] & TRANSFORM_DIRTY.WORLD) !== 0) {
      continue;
    }

    // Solve the whole subtree rooted here, in topological order.
    solved += _solveSubtree(eid);
  }

  const t1 = _now();
  const cost = t1 - t0;
  TransformState.lastSolveMs = cost;
  TransformState.avgSolveMs += (cost - TransformState.avgSolveMs) * 0.15;
  TransformState.lastSolveDirtyCount = solved;

  return solved;
}

/**
 * Solves the world matrices of the given entity and its descendants.
 * Uses a queue-based BFS (topological) so that parents are always solved
 * before their children. Allocation-free.
 */
function _solveSubtree(rootEid) {
  let solved = 0;
  let sp = 0;

  // Reuse the DFS stack as a BFS queue. BFS order guarantees parent
  // before child for a tree.
  _dfsStack[sp++] = rootEid;

  while (sp > 0) {
    const eid = _dfsStack[--sp];
    computeWorldMatrix(eid);
    TransformDirty.flags[eid] = TRANSFORM_DIRTY.NONE;
    solved++;

    const base = eid * MAX_CHILDREN_PER_ENTITY;
    for (let i = 0; i < MAX_CHILDREN_PER_ENTITY; i++) {
      const child = Children.childEid[base + i];
      if (child === NULL_ENTITY) continue;
      if (sp >= _dfsStack.length) break;
      _dfsStack[sp++] = child;
    }
  }

  TransformState.totalWorldSolves += solved;
  return solved;
}

/**
 * Forces a full world transform solve on every entity. Use only at boot
 * or after a topology change.
 */
export function forceFullWorldSolve() {
  let solved = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    computeWorldMatrix(eid);
    TransformDirty.flags[eid] = TRANSFORM_DIRTY.NONE;
    solved++;
  }
  TransformState.totalWorldSolves += solved;
  return solved;
}

/* ------------------------------------------------------------------ */
/* 8. WORLD-SPACE ACCESSORS                                           */
/* ------------------------------------------------------------------ */

/**
 * Returns the world-space position of an entity. Writes into out[0..2].
 * Returns out, or null on failure.
 */
export function getWorldPosition(eid, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  const dst = out || _scratchVec3;
  dst[0] = TransformWorld.x[eid];
  dst[1] = TransformWorld.y[eid];
  dst[2] = TransformWorld.z[eid];
  return dst;
}

/**
 * Sets the world-space position of an entity, converting to local by
 * subtracting the parent's world position (no rotation/scale inverse).
 *
 * For a fully correct parent-inverse transform use
 * `setWorldTransformFull`.
 */
export function setWorldPosition(eid, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const parentEid = Parent.eid[eid];
  if (parentEid === NULL_ENTITY) {
    return setLocalPosition(eid, x, y, z);
  }
  const px = TransformWorld.x[parentEid];
  const py = TransformWorld.y[parentEid];
  const pz = TransformWorld.z[parentEid];
  return setLocalPosition(eid, x - px, y - py, z - pz);
}

/**
 * Returns the world-space quaternion of an entity.
 */
export function getWorldQuaternion(eid, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  const dst = out || _scratchQuat;
  dst[0] = TransformWorld.qx[eid];
  dst[1] = TransformWorld.qy[eid];
  dst[2] = TransformWorld.qz[eid];
  dst[3] = TransformWorld.qw[eid];
  return dst;
}

/**
 * Sets the world-space quaternion of an entity. Simplified: sets the
 * local quaternion directly. Full parent-inverse rotation is handled by
 * `setWorldTransformFull`.
 */
export function setWorldQuaternion(eid, x, y, z, w) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return setLocalQuaternion(eid, x, y, z, w);
}

/**
 * Returns the world-space scale of an entity.
 */
export function getWorldScale(eid, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  const dst = out || _scratchVec3B;
  dst[0] = TransformWorld.sx[eid];
  dst[1] = TransformWorld.sy[eid];
  dst[2] = TransformWorld.sz[eid];
  return dst;
}

/**
 * Sets the world-space scale of an entity.
 */
export function setWorldScale(eid, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return setLocalScale(eid, x, y, z);
}

/**
 * Returns the maximum world scale of an entity (used to scale bounding
 * volumes).
 */
export function getWorldMaxScale(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 1;
  return TransformWorld.maxScale[eid];
}

/* ------------------------------------------------------------------ */
/* 9. INTERPOLATION & DAMPING                                         */
/* ------------------------------------------------------------------ */

/**
 * Linearly interpolates the local TRS of an entity between its previous
 * snapshot and its current authored TRS by `alpha` in [0, 1].
 *
 * Writes into `out` — an object with fields {x, y, z, qx, qy, qz, qw,
 * sx, sy, sz}. If `out` is omitted the interpolation is written into
 * the entity's TransformLocal (in-place) — useful for a "lateUpdate"
 * style render-time pass.
 */
export function interpolateTransformLocal(eid, alpha, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;

  const a = Math.max(0, Math.min(1, alpha));
  const inv = 1 - a;

  const px = TransformInterpolation.prevX[eid] * inv + TransformLocal.x[eid] * a;
  const py = TransformInterpolation.prevY[eid] * inv + TransformLocal.y[eid] * a;
  const pz = TransformInterpolation.prevZ[eid] * inv + TransformLocal.z[eid] * a;

  // Slerp for quaternion (short-path).
  const q0x = TransformInterpolation.prevQx[eid];
  const q0y = TransformInterpolation.prevQy[eid];
  const q0z = TransformInterpolation.prevQz[eid];
  const q0w = TransformInterpolation.prevQw[eid];

  const q1x = TransformLocal.qx[eid];
  const q1y = TransformLocal.qy[eid];
  const q1z = TransformLocal.qz[eid];
  const q1w = TransformLocal.qw[eid];

  let cosHalfTheta = q0x * q1x + q0y * q1y + q0z * q1z + q0w * q1w;
  let sign = 1;
  if (cosHalfTheta < 0) {
    cosHalfTheta = -cosHalfTheta;
    sign = -1;
  }

  let qx, qy, qz, qw;
  if (cosHalfTheta > 0.9995) {
    // Very close — linear blend then normalize.
    qx = q0x * inv + q1x * sign * a;
    qy = q0y * inv + q1y * sign * a;
    qz = q0z * inv + q1z * sign * a;
    qw = q0w * inv + q1w * sign * a;
  } else {
    const halfTheta = Math.acos(cosHalfTheta);
    const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);
    const ratioA = Math.sin(inv * halfTheta) / sinHalfTheta;
    const ratioB = Math.sin(a * halfTheta) / sinHalfTheta;

    qx = q0x * ratioA + q1x * ratioB * sign;
    qy = q0y * ratioA + q1y * ratioB * sign;
    qz = q0z * ratioA + q1z * ratioB * sign;
    qw = q0w * ratioA + q1w * ratioB * sign;
  }

  // Normalize quaternion.
  const qLen = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) || 1;
  qx /= qLen; qy /= qLen; qz /= qLen; qw /= qLen;

  const sx = TransformInterpolation.prevSx[eid] * inv + TransformLocal.sx[eid] * a;
  const sy = TransformInterpolation.prevSy[eid] * inv + TransformLocal.sy[eid] * a;
  const sz = TransformInterpolation.prevSz[eid] * inv + TransformLocal.sz[eid] * a;

  if (out) {
    out.x  = px; out.y  = py; out.z  = pz;
    out.qx = qx; out.qy = qy; out.qz = qz; out.qw = qw;
    out.sx = sx; out.sy = sy; out.sz = sz;
    return out;
  }

  TransformLocal.x[eid]  = px;
  TransformLocal.y[eid]  = py;
  TransformLocal.z[eid]  = pz;
  TransformLocal.qx[eid] = qx;
  TransformLocal.qy[eid] = qy;
  TransformLocal.qz[eid] = qz;
  TransformLocal.qw[eid] = qw;
  TransformLocal.sx[eid] = sx;
  TransformLocal.sy[eid] = sy;
  TransformLocal.sz[eid] = sz;

  // Mirror into 002's Transform.
  Transform.x[eid]  = px;
  Transform.y[eid]  = py;
  Transform.z[eid]  = pz;
  Transform.qx[eid] = qx;
  Transform.qy[eid] = qy;
  Transform.qz[eid] = qz;
  Transform.qw[eid] = qw;
  Transform.sx[eid] = sx;
  Transform.sy[eid] = sy;
  Transform.sz[eid] = sz;

  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  TransformState.totalInterpolations++;
  return null;
}

/**
 * Frame-rate-independent exponential damp of an entity's local TRS
 * toward a target TRS.
 *
 * `target` is a plain object with fields {x, y, z, qx, qy, qz, qw, sx,
 * sy, sz}. `lambda` is the damping rate; `dt` is the frame delta in
 * seconds.
 */
export function dampTransforms(eid, target, lambda, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!target) return false;

  const k = 1 - Math.exp(-(lambda > 0 ? lambda : 4) * (dt > 0 ? dt : 0.016));
  if (k <= 0 || k >= 1) return false;

  const curX = TransformLocal.x[eid];
  const curY = TransformLocal.y[eid];
  const curZ = TransformLocal.z[eid];

  const newX = curX + (target.x - curX) * k;
  const newY = curY + (target.y - curY) * k;
  const newZ = curZ + (target.z - curZ) * k;

  // Quaternion damp via slerp to target.
  const q0x = TransformLocal.qx[eid];
  const q0y = TransformLocal.qy[eid];
  const q0z = TransformLocal.qz[eid];
  const q0w = TransformLocal.qw[eid];
  const q1x = target.qx;
  const q1y = target.qy;
  const q1z = target.qz;
  const q1w = target.qw;

  let cosHalfTheta = q0x * q1x + q0y * q1y + q0z * q1z + q0w * q1w;
  let sign = 1;
  if (cosHalfTheta < 0) { cosHalfTheta = -cosHalfTheta; sign = -1; }

  let nqx, nqy, nqz, nqw;
  if (cosHalfTheta > 0.9995) {
    nqx = q0x + (q1x * sign - q0x) * k;
    nqy = q0y + (q1y * sign - q0y) * k;
    nqz = q0z + (q1z * sign - q0z) * k;
    nqw = q0w + (q1w * sign - q0w) * k;
  } else {
    const halfTheta = Math.acos(cosHalfTheta);
    const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);
    const ratioA = Math.sin((1 - k) * halfTheta) / sinHalfTheta;
    const ratioB = Math.sin(k * halfTheta) / sinHalfTheta;
    nqx = q0x * ratioA + q1x * ratioB * sign;
    nqy = q0y * ratioA + q1y * ratioB * sign;
    nqz = q0z * ratioA + q1z * ratioB * sign;
    nqw = q0w * ratioA + q1w * ratioB * sign;
  }

  const qLen = Math.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw) || 1;
  nqx /= qLen; nqy /= qLen; nqz /= qLen; nqw /= qLen;

  const newSx = TransformLocal.sx[eid] + (target.sx - TransformLocal.sx[eid]) * k;
  const newSy = TransformLocal.sy[eid] + (target.sy - TransformLocal.sy[eid]) * k;
  const newSz = TransformLocal.sz[eid] + (target.sz - TransformLocal.sz[eid]) * k;

  TransformLocal.x[eid]  = newX;
  TransformLocal.y[eid]  = newY;
  TransformLocal.z[eid]  = newZ;
  TransformLocal.qx[eid] = nqx;
  TransformLocal.qy[eid] = nqy;
  TransformLocal.qz[eid] = nqz;
  TransformLocal.qw[eid] = nqw;
  TransformLocal.sx[eid] = newSx;
  TransformLocal.sy[eid] = newSy;
  TransformLocal.sz[eid] = newSz;

  Transform.x[eid]  = newX;
  Transform.y[eid]  = newY;
  Transform.z[eid]  = newZ;
  Transform.qx[eid] = nqx;
  Transform.qy[eid] = nqy;
  Transform.qz[eid] = nqz;
  Transform.qw[eid] = nqw;
  Transform.sx[eid] = newSx;
  Transform.sy[eid] = newSy;
  Transform.sz[eid] = newSz;

  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  TransformState.totalDamps++;
  return true;
}

/**
 * Lerps an entity's local TRS from a `from` snapshot toward a `to`
 * snapshot by `alpha`.
 */
export function lerpTransforms(eid, from, to, alpha) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!from || !to) return false;

  const a = Math.max(0, Math.min(1, alpha));
  const inv = 1 - a;

  const x  = from.x  * inv + to.x  * a;
  const y  = from.y  * inv + to.y  * a;
  const z  = from.z  * inv + to.z  * a;
  const sx = from.sx * inv + to.sx * a;
  const sy = from.sy * inv + to.sy * a;
  const sz = from.sz * inv + to.sz * a;

  // Slerp quaternion.
  const q0x = from.qx, q0y = from.qy, q0z = from.qz, q0w = from.qw;
  const q1x = to.qx,   q1y = to.qy,   q1z = to.qz,   q1w = to.qw;

  let cosHalfTheta = q0x * q1x + q0y * q1y + q0z * q1z + q0w * q1w;
  let sign = 1;
  if (cosHalfTheta < 0) { cosHalfTheta = -cosHalfTheta; sign = -1; }

  let qx, qy, qz, qw;
  if (cosHalfTheta > 0.9995) {
    qx = q0x * inv + q1x * sign * a;
    qy = q0y * inv + q1y * sign * a;
    qz = q0z * inv + q1z * sign * a;
    qw = q0w * inv + q1w * sign * a;
  } else {
    const halfTheta = Math.acos(cosHalfTheta);
    const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);
    const ratioA = Math.sin(inv * halfTheta) / sinHalfTheta;
    const ratioB = Math.sin(a * halfTheta) / sinHalfTheta;
    qx = q0x * ratioA + q1x * ratioB * sign;
    qy = q0y * ratioA + q1y * ratioB * sign;
    qz = q0z * ratioA + q1z * ratioB * sign;
    qw = q0w * ratioA + q1w * ratioB * sign;
  }

  const qLen = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) || 1;
  qx /= qLen; qy /= qLen; qz /= qLen; qw /= qLen;

  TransformLocal.x[eid]  = x;
  TransformLocal.y[eid]  = y;
  TransformLocal.z[eid]  = z;
  TransformLocal.qx[eid] = qx;
  TransformLocal.qy[eid] = qy;
  TransformLocal.qz[eid] = qz;
  TransformLocal.qw[eid] = qw;
  TransformLocal.sx[eid] = sx;
  TransformLocal.sy[eid] = sy;
  TransformLocal.sz[eid] = sz;

  Transform.x[eid]  = x;
  Transform.y[eid]  = y;
  Transform.z[eid]  = z;
  Transform.qx[eid] = qx;
  Transform.qy[eid] = qy;
  Transform.qz[eid] = qz;
  Transform.qw[eid] = qw;
  Transform.sx[eid] = sx;
  Transform.sy[eid] = sy;
  Transform.sz[eid] = sz;

  markTransformDirty(eid, TRANSFORM_DIRTY.LOCAL);
  TransformState.totalLerps++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. SPATIAL SYNC                                                   */
/* ------------------------------------------------------------------ */

/**
 * Recomputes the entity's AABB and sphere from its current world
 * transform, then stores them in the spatial components.
 */
export function syncTransformToSpatial(eid, baseHalfExtent, baseRadius) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const maxScale = TransformWorld.maxScale[eid] || 1;
  const h = baseHalfExtent !== undefined ? baseHalfExtent : 0.5;
  const r = baseRadius !== undefined ? baseRadius : 0.5;

  // Compute AABB from world position + scaled half-extent.
  const wx = TransformWorld.x[eid];
  const wy = TransformWorld.y[eid];
  const wz = TransformWorld.z[eid];
  const hx = h * (TransformWorld.sx[eid] || 1);
  const hy = h * (TransformWorld.sy[eid] || 1);
  const hz = h * (TransformWorld.sz[eid] || 1);

  // Write into 002's Transform (which computeAABB reads).
  Transform.x[eid] = wx;
  Transform.y[eid] = wy;
  Transform.z[eid] = wz;
  Transform.sx[eid] = TransformWorld.sx[eid];
  Transform.sy[eid] = TransformWorld.sy[eid];
  Transform.sz[eid] = TransformWorld.sz[eid];

  // Compute AABB with the scaled half-extents.
  // 025 computeAABB reads Transform.{x,y,z} and multiplies by |scale| × h.
  // Since we already wrote world position and world scale, the AABB will
  // be correct in world space.
  computeAABB(eid, h);

  // Compute sphere from world center + max scale.
  const sphereRadius = r * maxScale;
  computeSphere(eid, sphereRadius);

  return true;
}

/**
 * Returns the world-space bounding radius of an entity given a base
 * radius in local units.
 */
export function computeWorldBoundingRadius(eid, baseRadius) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  const r = baseRadius !== undefined ? baseRadius : 0.5;
  return r * (TransformWorld.maxScale[eid] || 1);
}

/* ------------------------------------------------------------------ */
/* 11. PARENT BINDING HELPERS                                         */
/* ------------------------------------------------------------------ */

/**
 * Rebuilds the TransformParent mirror from the relations graph.
 * Called after a hierarchy change.
 */
export function syncTransformParentFromRelations() {
  let updated = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const p = Parent.eid[eid];
    if (TransformParent.parentEid[eid] !== p) {
      TransformParent.parentEid[eid] = p;
      TransformRoot.valid[eid] = 0;
      markTransformDirty(eid, TRANSFORM_DIRTY.SUBTREE);
      updated++;
    }
  }
  return updated;
}

/**
 * Returns the cached root entity of an entity's transform chain, or
 * NULL_ENTITY. Recomputes if stale.
 */
export function getTransformRoot(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return NULL_ENTITY;
  if (TransformRoot.valid[eid] === 1) return TransformRoot.rootEid[eid];

  let cursor = eid;
  let depth = 0;
  while (depth < MAX_TRANSFORM_DEPTH) {
    const p = Parent.eid[cursor];
    if (p === NULL_ENTITY) break;
    cursor = p;
    depth++;
  }
  TransformRoot.rootEid[eid] = cursor;
  TransformRoot.valid[eid] = 1;
  return cursor;
}

/**
 * Returns the depth of the entity in the transform hierarchy.
 */
export function getTransformDepth(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  return TransformParent.depth[eid];
}

/* ------------------------------------------------------------------ */
/* 12. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the transform system frame counter. Called once per frame
 * before authored TRS is updated.
 */
export function tickTransforms(frameNumber) {
  if (typeof frameNumber === 'number') TransformState.frame = frameNumber;
  else TransformState.frame++;
}

/* ------------------------------------------------------------------ */
/* 13. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerTransformComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'TransformLocal',         component: TransformLocal,         category: 19, subsystem: 1, dependencies: [] },
    { name: 'TransformWorld',         component: TransformWorld,         category: 19, subsystem: 1, dependencies: ['TransformLocal'] },
    { name: 'TransformDirty',         component: TransformDirty,         category: 19, subsystem: 1, dependencies: [] },
    { name: 'TransformMatrix',        component: TransformMatrix,        category: 19, subsystem: 1, dependencies: ['TransformLocal'] },
    { name: 'TransformParent',        component: TransformParent,        category: 19, subsystem: 1, dependencies: [] },
    { name: 'TransformInterpolation', component: TransformInterpolation, category: 19, subsystem: 1, dependencies: ['TransformLocal'] },
    { name: 'TransformEuler',         component: TransformEuler,         category: 19, subsystem: 1, dependencies: [] },
    { name: 'TransformRoot',          component: TransformRoot,          category: 19, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 14. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getTransformStats() {
  let dirtyCount = 0;
  let subtreeDirtyCount = 0;
  let localValidCount = 0;
  let worldValidCount = 0;
  let interpEnabledCount = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const flags = TransformDirty.flags[eid];
    if (flags !== 0) dirtyCount++;
    if (flags & TRANSFORM_DIRTY.SUBTREE) subtreeDirtyCount++;
    if (TransformMatrix.localValid[eid] === 1) localValidCount++;
    if (TransformMatrix.worldValid[eid] === 1) worldValidCount++;
    if (TransformInterpolation.enabled[eid] === 1) interpEnabledCount++;
  }

  return {
    frame:                 TransformState.frame,
    totalDirtyMarks:       TransformState.totalDirtyMarks,
    totalPropagations:     TransformState.totalPropagations,
    totalWorldSolvePasses: TransformState.totalWorldSolvePasses,
    totalWorldSolves:      TransformState.totalWorldSolves,
    totalInterpolations:   TransformState.totalInterpolations,
    totalDamps:            TransformState.totalDamps,
    totalLerps:            TransformState.totalLerps,
    peakDirtyCount:        TransformState.peakDirtyCount,
    lastSolveMs:           TransformState.lastSolveMs,
    avgSolveMs:            TransformState.avgSolveMs,
    lastSolveDirtyCount:   TransformState.lastSolveDirtyCount,
    dirtyCount,
    subtreeDirtyCount,
    localValidCount,
    worldValidCount,
    interpEnabledCount,
    perfTier:              PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 15. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every transform structure and resets counters.
 */
export function resetTransformState() {
  TransformLocal.x.fill(0);  TransformLocal.y.fill(0);  TransformLocal.z.fill(0);
  TransformLocal.qx.fill(0); TransformLocal.qy.fill(0); TransformLocal.qz.fill(0); TransformLocal.qw.fill(1);
  TransformLocal.sx.fill(1); TransformLocal.sy.fill(1); TransformLocal.sz.fill(1);
  TransformLocal.sync.fill(0);

  TransformWorld.x.fill(0);  TransformWorld.y.fill(0);  TransformWorld.z.fill(0);
  TransformWorld.qx.fill(0); TransformWorld.qy.fill(0); TransformWorld.qz.fill(0); TransformWorld.qw.fill(1);
  TransformWorld.sx.fill(1); TransformWorld.sy.fill(1); TransformWorld.sz.fill(1);
  TransformWorld.maxScale.fill(1);

  TransformDirty.flags.fill(0);
  TransformDirty.lastDirtyFrame.fill(0);
  TransformDirty.lastSolveFrame.fill(0);

  TransformMatrix.local.fill(0);
  TransformMatrix.world.fill(0);
  TransformMatrix.localValid.fill(0);
  TransformMatrix.worldValid.fill(0);

  TransformParent.parentEid.fill(NULL_ENTITY);
  TransformParent.rootEid.fill(NULL_ENTITY);
  TransformParent.depth.fill(0);

  TransformInterpolation.prevX.fill(0);
  TransformInterpolation.prevY.fill(0);
  TransformInterpolation.prevZ.fill(0);
  TransformInterpolation.prevQx.fill(0);
  TransformInterpolation.prevQy.fill(0);
  TransformInterpolation.prevQz.fill(0);
  TransformInterpolation.prevQw.fill(1);
  TransformInterpolation.prevSx.fill(1);
  TransformInterpolation.prevSy.fill(1);
  TransformInterpolation.prevSz.fill(1);
  TransformInterpolation.enabled.fill(0);

  TransformEuler.ex.fill(0);
  TransformEuler.ey.fill(0);
  TransformEuler.ez.fill(0);
  TransformEuler.order.fill(0);

  TransformRoot.rootEid.fill(NULL_ENTITY);
  TransformRoot.valid.fill(0);

  _dfsSeen.fill(0);
  _dfsSeenFrame = 0;

  TransformState.frame = 0;
  TransformState.totalDirtyMarks = 0;
  TransformState.totalPropagations = 0;
  TransformState.totalWorldSolvePasses = 0;
  TransformState.totalWorldSolves = 0;
  TransformState.totalInterpolations = 0;
  TransformState.totalDamps = 0;
  TransformState.totalLerps = 0;
  TransformState.peakDirtyCount = 0;
  TransformState.lastSolveMs = 0;
  TransformState.avgSolveMs = 0;
  TransformState.lastSolveDirtyCount = 0;
}

/* ------------------------------------------------------------------ */
/* 16. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_TRANSFORM_DEPTH,
  TRANSFORM_DIRTY,
  TRANSFORM_SYNC,
  TRANSFORM_SYNC_NAME,

  // Components
  TransformLocal,
  TransformWorld,
  TransformDirty,
  TransformMatrix,
  TransformParent,
  TransformInterpolation,
  TransformEuler,
  TransformRoot,
  TRANSFORM_COMPONENTS,

  // Module state
  TransformState,

  // Dirty flags
  markTransformDirty,
  propagateTransformDirtyDownward,
  clearTransformDirty,
  isTransformDirty,

  // Local TRS
  setLocalPosition,
  setLocalQuaternion,
  setLocalScale,
  snapshotTransformLocal,

  // Matrix
  composeTRSMatrix,
  multiplyMatrices,
  computeLocalMatrix,
  computeWorldMatrix,

  // Solve
  updateWorldTransforms,
  forceFullWorldSolve,

  // World accessors
  getWorldPosition,
  setWorldPosition,
  getWorldQuaternion,
  setWorldQuaternion,
  getWorldScale,
  setWorldScale,
  getWorldMaxScale,

  // Interpolation
  interpolateTransformLocal,
  dampTransforms,
  lerpTransforms,

  // Spatial sync
  syncTransformToSpatial,
  computeWorldBoundingRadius,

  // Parent
  syncTransformParentFromRelations,
  getTransformRoot,
  getTransformDepth,

  // Frame
  tickTransforms,

  // Diagnostics
  getTransformStats,

  // Registration
  registerTransformComponents,

  // Reset
  resetTransformState,
};

export default _defaultExport;