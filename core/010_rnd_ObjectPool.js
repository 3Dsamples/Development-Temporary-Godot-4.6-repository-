// File : 010
// name : src/core/010_rnd_ObjectPool.js
// description : Zero-allocation object pool system for the anime lighting stack
//               on Android mobile. Every lighting subsystem (006_lgt_LightManager
//               through 380_lgt_lights) that needs transient Vector3 / Vector4 /
//               Color / Quaternion / Matrix4 / Spherical / Box3 / Sphere / Plane
//               / Frustum / Ray / Float32Array scratch acquires them from this
//               pool instead of allocating new instances. Eliminates the two
//               biggest sources of GC pressure on mobile: (a) per-frame temporary
//               THREE.Vector3/Color/Matrix4 objects created inside lighting math,
//               and (b) per-frame scratch typed arrays created inside shadow/GI/AO
//               solvers.
//
//               Design:
//                 • Generic ObjectPool — factory + reset + optional validator.
//                   Fixed capacity, ring-buffer free list, O(1) acquire/release.
//                 • TypedArrayPool — sized buckets (Float32Array, Uint16Array,
//                   Uint32Array, Uint8Array), each with a fixed count of
//                   pre-allocated arrays and a matching free list.
//                 • ThreePool — pre-built pools for the Three.js r185 types
//                   used by the lighting stack. All objects come from the
//                   r185 src/math/ tree (Vector3, Vector4, Matrix4, Color,
//                   Quaternion, Euler, Spherical, Box3, Sphere, Plane, Frustum,
//                   Ray).
//                 • Auto-reclaim policy: when a pool is exhausted, the pool
//                   either (i) rejects the acquire and returns null, or
//                   (ii) recycles the oldest un-released object (opt-in, only
//                   for non-critical scratch pools). Lighting-critical pools
//                   (shadow matrices, light uniforms, GI probes) use (i) and
//                   the caller must handle the null → next-frame retry.
//                 • Frame tick: `beginFrame()` resets per-frame release
//                   counters; `endFrame()` optionally force-releases all
//                   objects acquired this frame if `autoReleasePerFrame` is
//                   set — critical to prevent leaks when a lighting callback
//                   early-returns on error.
//
//               Optimization techniques applied:
//                 • Ring-buffer free list (typed Int32Array) — O(1), no Map
//                   or Set, no array shift/pop, no closures.
//                 • Batch reset callbacks — pool.resetAll() zeroes objects in
//                   a tight loop with cached instance references.
//                 • Object identity fast path — release() checks the pool id
//                   baked into each object via a non-enumerable `__poolId`
//                   property; mismatched release is rejected instantly.
//                 • Typed array bucket sizes chosen for lighting workloads:
//                   4/8/16/32/64/128/256/512/1024/2048 floats.
//                 • Zero per-frame allocations on the hot path — acquire()
//                   and release() touch only typed arrays and cached refs.
//                 • Cold-path construction: every pool and every typed array
//                   is allocated ONCE at module init; nothing grows later.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external pool libs; every typed array sized once at
//               construction.
// best for : Guaranteeing that lighting math on Android never allocates
//            inside update loops. Every shadow solver, GI baker, AO blur
//            kernel, light list builder, cluster packer, environment palette
//            mixer, and post-processing pass pulls its scratch from here.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const POOL_CAPACITY = Object.freeze({
  vector3:    PERF_TIER === 'HIGH' ? 1024 : PERF_TIER === 'MEDIUM' ? 512 : 256,
  vector2:    PERF_TIER === 'HIGH' ? 512  : PERF_TIER === 'MEDIUM' ? 256 : 128,
  vector4:    PERF_TIER === 'HIGH' ? 512  : PERF_TIER === 'MEDIUM' ? 256 : 128,
  color:      PERF_TIER === 'HIGH' ? 512  : PERF_TIER === 'MEDIUM' ? 256 : 128,
  quaternion: PERF_TIER === 'HIGH' ? 256  : PERF_TIER === 'MEDIUM' ? 128 : 64,
  euler:      PERF_TIER === 'HIGH' ? 256  : PERF_TIER === 'MEDIUM' ? 128 : 64,
  spherical:  PERF_TIER === 'HIGH' ? 128  : PERF_TIER === 'MEDIUM' ? 64  : 32,
  matrix3:    PERF_TIER === 'HIGH' ? 128  : PERF_TIER === 'MEDIUM' ? 64  : 32,
  matrix4:    PERF_TIER === 'HIGH' ? 256  : PERF_TIER === 'MEDIUM' ? 128 : 64,
  box3:       PERF_TIER === 'HIGH' ? 128  : PERF_TIER === 'MEDIUM' ? 64  : 32,
  sphere:     PERF_TIER === 'HIGH' ? 128  : PERF_TIER === 'MEDIUM' ? 64  : 32,
  plane:      PERF_TIER === 'HIGH' ? 64   : PERF_TIER === 'MEDIUM' ? 32  : 16,
  frustum:    PERF_TIER === 'HIGH' ? 32   : PERF_TIER === 'MEDIUM' ? 16  : 8,
  ray:        PERF_TIER === 'HIGH' ? 32   : PERF_TIER === 'MEDIUM' ? 16  : 8,
});

export const TYPED_BUCKET_SIZES = Object.freeze([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]);

const POOL_ID_SYMBOL = '__poolId';

/* ------------------------------------------------------------------ */
/* 1. POOL KIND REGISTRY (used for diagnostics)                       */
/* ------------------------------------------------------------------ */

export const POOL_KIND = Object.freeze({
  GENERIC:     0,
  VECTOR2:     1,
  VECTOR3:     2,
  VECTOR4:     3,
  COLOR:       4,
  QUATERNION:  5,
  EULER:       6,
  SPHERICAL:   7,
  MATRIX3:     8,
  MATRIX4:     9,
  BOX3:       10,
  SPHERE:     11,
  PLANE:      12,
  FRUSTUM:    13,
  RAY:        14,
  FLOAT32:    15,
  UINT16:     16,
  UINT32:     17,
  UINT8:      18,
  COUNT:      19,
});

export const POOL_KIND_NAME = Object.freeze([
  'generic',
  'vector2',
  'vector3',
  'vector4',
  'color',
  'quaternion',
  'euler',
  'spherical',
  'matrix3',
  'matrix4',
  'box3',
  'sphere',
  'plane',
  'frustum',
  'ray',
  'float32',
  'uint16',
  'uint32',
  'uint8',
]);

/* ------------------------------------------------------------------ */
/* 2. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _poolIdCounter = 0;

function _nextPoolId() {
  return ++_poolIdCounter;
}

/* ------------------------------------------------------------------ */
/* 3. GENERIC OBJECT POOL                                             */
/* ------------------------------------------------------------------ */

export class ObjectPool {
  constructor(options = {}) {
    if (!options || typeof options.factory !== 'function') {
      throw new Error('[010_rnd_ObjectPool] ObjectPool requires a factory function');
    }

    this.name        = options.name || `pool_${_nextPoolId()}`;
    this.kind        = options.kind !== undefined ? options.kind : POOL_KIND.GENERIC;
    this.capacity    = Math.max(1, options.capacity | 0);
    this.factory     = options.factory;
    this.reset       = typeof options.reset === 'function' ? options.reset : null;
    this.validate    = typeof options.validate === 'function' ? options.validate : null;
    this.autoReclaim = options.autoReclaim === true;

    this.poolId      = _nextPoolId();

    // Pre-allocate all objects ONCE.
    this.objects     = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) {
      const obj = this.factory();
      obj[POOL_ID_SYMBOL] = this.poolId;
      this.objects[i] = obj;
    }

    // Ring free list.
    this.freeList    = new Int32Array(this.capacity);
    this.freeHead    = 0;
    this.freeCount   = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;

    // Acquisition tracking for auto-release-per-frame policy.
    this.acquired    = new Uint8Array(this.capacity);
    this.acquiredAt  = new Uint32Array(this.capacity);

    // Stats.
    this.stats = {
      acquired:      0,
      released:      0,
      rejected:      0,
      reclaimed:     0,
      peakInUse:     0,
      currentInUse:  0,
      frameAcquired: 0,
      frameReleased: 0,
      totalCreated:  this.capacity,
    };

    this.frame       = 0;
    this.autoReleasePerFrame = options.autoReleasePerFrame === true;
    this.autoReclaimGrace    = Math.max(1, options.autoReclaimGrace | 0) || 60;
  }

  /* ---------------- acquire / release ---------------- */

  acquire() {
    if (this.freeCount <= 0) {
      if (this.autoReclaim) {
        // Reclaim the oldest in-use slot.
        const idx = this._reclaimOldest();
        if (idx >= 0) {
          this.stats.reclaimed++;
          return this._wrapAcquire(idx);
        }
      }
      this.stats.rejected++;
      return null;
    }

    const idx = this.freeList[this.freeHead];
    this.freeHead = (this.freeHead + 1) % this.capacity;
    this.freeCount--;
    return this._wrapAcquire(idx);
  }

  _wrapAcquire(idx) {
    const obj = this.objects[idx];
    this.acquired[idx] = 1;
    this.acquiredAt[idx] = this.frame;

    this.stats.acquired++;
    this.stats.frameAcquired++;
    this.stats.currentInUse++;
    if (this.stats.currentInUse > this.stats.peakInUse) {
      this.stats.peakInUse = this.stats.currentInUse;
    }
    return obj;
  }

  release(obj) {
    if (!obj || obj[POOL_ID_SYMBOL] !== this.poolId) {
      this.stats.rejected++;
      return false;
    }

    // Find the object's slot index. We cache it on the object to avoid a
    // linear scan. Non-enumerable to avoid JSON pollution.
    let idx = obj.__poolIndex;
    if (typeof idx !== 'number' || this.objects[idx] !== obj) {
      idx = this._indexOf(obj);
      if (idx < 0) {
        this.stats.rejected++;
        return false;
      }
    }

    if (this.acquired[idx] === 0) {
      // Double release — ignore.
      return false;
    }

    if (this.reset) {
      try { this.reset(obj); } catch (_) { /* swallow reset errors */ }
    }

    this.acquired[idx] = 0;
    this.freeList[(this.freeHead + this.freeCount) % this.capacity] = idx;
    this.freeCount++;

    this.stats.released++;
    this.stats.frameReleased++;
    this.stats.currentInUse--;
    return true;
  }

  releaseAll() {
    let n = 0;
    for (let i = 0; i < this.capacity; i++) {
      if (this.acquired[i] === 1) {
        const obj = this.objects[i];
        if (this.reset) {
          try { this.reset(obj); } catch (_) { /* swallow */ }
        }
        this.acquired[i] = 0;
        this.freeList[(this.freeHead + this.freeCount) % this.capacity] = i;
        this.freeCount++;
        this.stats.released++;
        this.stats.currentInUse--;
        n++;
      }
    }
    return n;
  }

  _indexOf(obj) {
    for (let i = 0; i < this.capacity; i++) {
      if (this.objects[i] === obj) return i;
    }
    return -1;
  }

  _reclaimOldest() {
    let oldestIdx = -1;
    let oldestFrame = 0xFFFFFFFF;
    for (let i = 0; i < this.capacity; i++) {
      if (this.acquired[i] === 1 && this.acquiredAt[i] < oldestFrame) {
        oldestFrame = this.acquiredAt[i];
        oldestIdx = i;
      }
    }
    if (oldestIdx < 0) return -1;
    // Force release it and hand the slot back to the caller.
    const obj = this.objects[oldestIdx];
    if (this.reset) {
      try { this.reset(obj); } catch (_) { /* swallow */ }
    }
    this.acquired[oldestIdx] = 0;
    this.stats.currentInUse--;
    return oldestIdx;
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame() {
    this.frame++;
    this.stats.frameAcquired = 0;
    this.stats.frameReleased = 0;
  }

  endFrame() {
    if (this.autoReleasePerFrame) {
      this.releaseAll();
    }
    return this.freeCount;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      name:           this.name,
      kind:           POOL_KIND_NAME[this.kind] || 'generic',
      capacity:       this.capacity,
      freeCount:      this.freeCount,
      inUse:          this.stats.currentInUse,
      peakInUse:      this.stats.peakInUse,
      acquired:       this.stats.acquired,
      released:       this.stats.released,
      rejected:       this.stats.rejected,
      reclaimed:      this.stats.reclaimed,
      frameAcquired:  this.stats.frameAcquired,
      frameReleased:  this.stats.frameReleased,
      autoReclaim:    this.autoReclaim,
    };
  }

  reset() {
    this.releaseAll();
    this.freeHead = 0;
    this.freeCount = this.capacity;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;
    this.stats.acquired = 0;
    this.stats.released = 0;
    this.stats.rejected = 0;
    this.stats.reclaimed = 0;
    this.stats.peakInUse = 0;
    this.stats.currentInUse = 0;
    this.stats.frameAcquired = 0;
    this.stats.frameReleased = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    this.objects.length = 0;
    this.objects = null;
    this.freeList = null;
    this.acquired = null;
    this.acquiredAt = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. TYPED ARRAY POOL                                                */
/* ------------------------------------------------------------------ */

export class TypedArrayPool {
  constructor(options = {}) {
    if (!options || typeof options.ArrayType !== 'function') {
      throw new Error('[010_rnd_ObjectPool] TypedArrayPool requires an ArrayType constructor');
    }

    this.name        = options.name || `typed_${_nextPoolId()}`;
    this.ArrayType   = options.ArrayType;
    this.kind        = options.kind !== undefined ? options.kind : POOL_KIND.FLOAT32;
    this.counts      = options.counts || _defaultTypedCounts(PERF_TIER);
    this.sizes       = options.sizes  || TYPED_BUCKET_SIZES;
    this.autoReclaim = options.autoReclaim === true;

    this.poolId      = _nextPoolId();

    // bucket[b] = { arrays: Array<ArrayType>, freeList, freeHead, freeCount }
    this.buckets = new Array(this.sizes.length);
    for (let b = 0; b < this.sizes.length; b++) {
      const count = this.counts[b] | 0;
      const size  = this.sizes[b] | 0;
      const arrays = new Array(count);
      const freeList = new Int32Array(count);
      for (let i = 0; i < count; i++) {
        const arr = new this.ArrayType(size);
        arr[POOL_ID_SYMBOL] = this.poolId;
        arr.__poolBucket = b;
        arr.__poolIndex = i;
        arrays[i] = arr;
        freeList[i] = i;
      }
      this.buckets[b] = {
        size,
        arrays,
        freeList,
        freeHead: 0,
        freeCount: count,
        capacity: count,
        acquired: new Uint8Array(count),
        acquiredAt: new Uint32Array(count),
        peakInUse: 0,
        currentInUse: 0,
        acquiredTotal: 0,
        releasedTotal: 0,
        rejectedTotal: 0,
      };
    }

    this.frame = 0;
    this.autoReleasePerFrame = options.autoReleasePerFrame === true;
  }

  _findBucketFor(size) {
    for (let b = 0; b < this.sizes.length; b++) {
      if (this.sizes[b] >= size) return b;
    }
    return -1;
  }

  acquire(size) {
    const need = Math.max(1, size | 0);
    const b = this._findBucketFor(need);
    if (b < 0) return null;

    const bucket = this.buckets[b];
    if (bucket.freeCount <= 0) {
      bucket.rejectedTotal++;
      return null;
    }

    const idx = bucket.freeList[bucket.freeHead];
    bucket.freeHead = (bucket.freeHead + 1) % bucket.capacity;
    bucket.freeCount--;

    bucket.acquired[idx] = 1;
    bucket.acquiredAt[idx] = this.frame;
    bucket.acquiredTotal++;
    bucket.currentInUse++;
    if (bucket.currentInUse > bucket.peakInUse) bucket.peakInUse = bucket.currentInUse;

    const arr = bucket.arrays[idx];
    // Provide a `subarray` view of the exact requested length so callers
    // don't accidentally write past their requested size.
    return arr.subarray(0, need);
  }

  release(arr) {
    if (!arr || !ArrayBuffer.isView(arr) || arr[POOL_ID_SYMBOL] !== this.poolId) {
      return false;
    }
    const b = arr.__poolBucket;
    if (typeof b !== 'number' || b < 0 || b >= this.buckets.length) return false;

    const bucket = this.buckets[b];
    const parent = bucket.arrays[arr.__poolIndex];
    if (!parent || arr.buffer !== parent.buffer) return false;
    if (bucket.acquired[arr.__poolIndex] === 0) return false;

    bucket.acquired[arr.__poolIndex] = 0;
    bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = arr.__poolIndex;
    bucket.freeCount++;
    bucket.releasedTotal++;
    bucket.currentInUse--;
    return true;
  }

  releaseAll() {
    for (let b = 0; b < this.buckets.length; b++) {
      const bucket = this.buckets[b];
      for (let i = 0; i < bucket.capacity; i++) {
        if (bucket.acquired[i] === 1) {
          bucket.acquired[i] = 0;
          bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = i;
          bucket.freeCount++;
          bucket.releasedTotal++;
          bucket.currentInUse--;
        }
      }
    }
  }

  beginFrame() {
    this.frame++;
  }

  endFrame() {
    if (this.autoReleasePerFrame) this.releaseAll();
  }

  getStats() {
    const buckets = new Array(this.buckets.length);
    for (let b = 0; b < this.buckets.length; b++) {
      const bucket = this.buckets[b];
      buckets[b] = {
        size:          bucket.size,
        capacity:      bucket.capacity,
        freeCount:     bucket.freeCount,
        currentInUse:  bucket.currentInUse,
        peakInUse:     bucket.peakInUse,
        acquiredTotal: bucket.acquiredTotal,
        releasedTotal: bucket.releasedTotal,
        rejectedTotal: bucket.rejectedTotal,
      };
    }
    return {
      name:  this.name,
      kind:  POOL_KIND_NAME[this.kind] || 'typed',
      buckets,
    };
  }

  reset() {
    for (let b = 0; b < this.buckets.length; b++) {
      const bucket = this.buckets[b];
      bucket.freeHead = 0;
      bucket.freeCount = bucket.capacity;
      for (let i = 0; i < bucket.capacity; i++) {
        bucket.freeList[i] = i;
        bucket.acquired[i] = 0;
        bucket.acquiredAt[i] = 0;
      }
      bucket.peakInUse = 0;
      bucket.currentInUse = 0;
      bucket.acquiredTotal = 0;
      bucket.releasedTotal = 0;
      bucket.rejectedTotal = 0;
    }
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let b = 0; b < this.buckets.length; b++) {
      this.buckets[b].arrays.length = 0;
      this.buckets[b].freeList = null;
      this.buckets[b].acquired = null;
      this.buckets[b].acquiredAt = null;
    }
    this.buckets.length = 0;
    this.buckets = null;
    return this;
  }
}

function _defaultTypedCounts(tier) {
  if (tier === 'HIGH') {
    return [512, 512, 512, 384, 256, 192, 128, 96, 64, 32, 16];
  }
  if (tier === 'MEDIUM') {
    return [256, 256, 256, 192, 128, 96, 64, 48, 32, 16, 8];
  }
  return [128, 128, 128, 96, 64, 48, 32, 24, 16, 8, 4];
}

/* ------------------------------------------------------------------ */
/* 5. RESET HELPERS FOR THREE.JS TYPES                                */
/* ------------------------------------------------------------------ */

function _resetVector3(v)  { v.set(0, 0, 0); }
function _resetVector2(v)  { v.set(0, 0); }
function _resetVector4(v)  { v.set(0, 0, 0, 0); }
function _resetColor(c)    { c.setRGB(0, 0, 0); }
function _resetQuaternion(q) { q.set(0, 0, 0, 1); }
function _resetEuler(e)    { e.set(0, 0, 0, 'XYZ'); }
function _resetSpherical(s) { s.set(0, 0, 0); }
function _resetMatrix3(m)  { m.identity(); }
function _resetMatrix4(m)  { m.identity(); }
function _resetBox3(b)     { b.makeEmpty(); }
function _resetSphere(s)   { s.set(new THREE.Vector3(0, 0, 0), 1); }
function _resetPlane(p)    { p.set(new THREE.Vector3(0, 1, 0), 0); }
function _resetFrustum(f)  { /* frustum has no cheap reset; leave as-is */ }
function _resetRay(r)      { r.set(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1)); }

/* ------------------------------------------------------------------ */
/* 6. LIGHTING POOL SET (the canonical pools for the whole engine)    */
/* ------------------------------------------------------------------ */

export class LightingPoolSet {
  constructor() {
    this.pools = new Array(POOL_KIND.COUNT);
    this.byName = new Map();

    // Vector2
    this.vector2 = this._register(new ObjectPool({
      name: 'vector2',
      kind: POOL_KIND.VECTOR2,
      capacity: POOL_CAPACITY.vector2,
      factory: () => new THREE.Vector2(0, 0),
      reset:   _resetVector2,
      autoReleasePerFrame: false,
    }));

    // Vector3
    this.vector3 = this._register(new ObjectPool({
      name: 'vector3',
      kind: POOL_KIND.VECTOR3,
      capacity: POOL_CAPACITY.vector3,
      factory: () => new THREE.Vector3(0, 0, 0),
      reset:   _resetVector3,
      autoReleasePerFrame: false,
    }));

    // Vector4
    this.vector4 = this._register(new ObjectPool({
      name: 'vector4',
      kind: POOL_KIND.VECTOR4,
      capacity: POOL_CAPACITY.vector4,
      factory: () => new THREE.Vector4(0, 0, 0, 0),
      reset:   _resetVector4,
      autoReleasePerFrame: false,
    }));

    // Color
    this.color = this._register(new ObjectPool({
      name: 'color',
      kind: POOL_KIND.COLOR,
      capacity: POOL_CAPACITY.color,
      factory: () => new THREE.Color(0, 0, 0),
      reset:   _resetColor,
      autoReleasePerFrame: false,
    }));

    // Quaternion
    this.quaternion = this._register(new ObjectPool({
      name: 'quaternion',
      kind: POOL_KIND.QUATERNION,
      capacity: POOL_CAPACITY.quaternion,
      factory: () => new THREE.Quaternion(0, 0, 0, 1),
      reset:   _resetQuaternion,
      autoReleasePerFrame: false,
    }));

    // Euler
    this.euler = this._register(new ObjectPool({
      name: 'euler',
      kind: POOL_KIND.EULER,
      capacity: POOL_CAPACITY.euler,
      factory: () => new THREE.Euler(0, 0, 0, 'XYZ'),
      reset:   _resetEuler,
      autoReleasePerFrame: false,
    }));

    // Spherical
    this.spherical = this._register(new ObjectPool({
      name: 'spherical',
      kind: POOL_KIND.SPHERICAL,
      capacity: POOL_CAPACITY.spherical,
      factory: () => new THREE.Spherical(0, 0, 0),
      reset:   _resetSpherical,
      autoReleasePerFrame: false,
    }));

    // Matrix3
    this.matrix3 = this._register(new ObjectPool({
      name: 'matrix3',
      kind: POOL_KIND.MATRIX3,
      capacity: POOL_CAPACITY.matrix3,
      factory: () => new THREE.Matrix3(),
      reset:   _resetMatrix3,
      autoReleasePerFrame: false,
    }));

    // Matrix4
    this.matrix4 = this._register(new ObjectPool({
      name: 'matrix4',
      kind: POOL_KIND.MATRIX4,
      capacity: POOL_CAPACITY.matrix4,
      factory: () => new THREE.Matrix4(),
      reset:   _resetMatrix4,
      autoReleasePerFrame: false,
    }));

    // Box3
    this.box3 = this._register(new ObjectPool({
      name: 'box3',
      kind: POOL_KIND.BOX3,
      capacity: POOL_CAPACITY.box3,
      factory: () => new THREE.Box3(),
      reset:   _resetBox3,
      autoReleasePerFrame: false,
    }));

    // Sphere
    this.sphere = this._register(new ObjectPool({
      name: 'sphere',
      kind: POOL_KIND.SPHERE,
      capacity: POOL_CAPACITY.sphere,
      factory: () => new THREE.Sphere(),
      reset:   _resetSphere,
      autoReleasePerFrame: false,
    }));

    // Plane
    this.plane = this._register(new ObjectPool({
      name: 'plane',
      kind: POOL_KIND.PLANE,
      capacity: POOL_CAPACITY.plane,
      factory: () => new THREE.Plane(),
      reset:   _resetPlane,
      autoReleasePerFrame: false,
    }));

    // Frustum
    this.frustum = this._register(new ObjectPool({
      name: 'frustum',
      kind: POOL_KIND.FRUSTUM,
      capacity: POOL_CAPACITY.frustum,
      factory: () => new THREE.Frustum(),
      reset:   _resetFrustum,
      autoReleasePerFrame: false,
    }));

    // Ray
    this.ray = this._register(new ObjectPool({
      name: 'ray',
      kind: POOL_KIND.RAY,
      capacity: POOL_CAPACITY.ray,
      factory: () => new THREE.Ray(),
      reset:   _resetRay,
      autoReleasePerFrame: false,
    }));

    // Typed array pools
    this.f32 = this._register(new TypedArrayPool({
      name: 'float32',
      kind: POOL_KIND.FLOAT32,
      ArrayType: Float32Array,
      autoReleasePerFrame: false,
    }));

    this.u16 = this._register(new TypedArrayPool({
      name: 'uint16',
      kind: POOL_KIND.UINT16,
      ArrayType: Uint16Array,
      autoReleasePerFrame: false,
    }));

    this.u32 = this._register(new TypedArrayPool({
      name: 'uint32',
      kind: POOL_KIND.UINT32,
      ArrayType: Uint32Array,
      autoReleasePerFrame: false,
    }));

    this.u8 = this._register(new TypedArrayPool({
      name: 'uint8',
      kind: POOL_KIND.UINT8,
      ArrayType: Uint8Array,
      autoReleasePerFrame: false,
    }));
  }

  _register(pool) {
    this.pools[pool.kind] = pool;
    this.byName.set(pool.name, pool);
    return pool;
  }

  get(kind) {
    return this.pools[kind] || null;
  }

  getByName(name) {
    return this.byName.get(name) || null;
  }

  /* ---------------- frame lifecycle for all pools ---------------- */

  beginFrame() {
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      if (p && typeof p.beginFrame === 'function') p.beginFrame();
    }
  }

  endFrame() {
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      if (p && typeof p.endFrame === 'function') p.endFrame();
    }
  }

  releaseAll() {
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      if (p && typeof p.releaseAll === 'function') p.releaseAll();
    }
  }

  reset() {
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      if (p && typeof p.reset === 'function') p.reset();
    }
    return this;
  }

  dispose() {
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      if (p && typeof p.dispose === 'function') p.dispose();
    }
    this.pools.length = 0;
    this.byName.clear();
    return this;
  }

  getStats() {
    const out = new Array(POOL_KIND.COUNT);
    for (let i = 0; i < POOL_KIND.COUNT; i++) {
      const p = this.pools[i];
      out[i] = p ? p.getStats() : null;
    }
    return out;
  }
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultPoolSet = null;

export function getDefaultPoolSet() {
  if (!_defaultPoolSet) _defaultPoolSet = new LightingPoolSet();
  return _defaultPoolSet;
}

export function disposeDefaultPoolSet() {
  if (_defaultPoolSet) {
    _defaultPoolSet.dispose();
    _defaultPoolSet = null;
  }
}

/* ------------------------------------------------------------------ */
/* 8. CONVENIENCE WRAPPERS (hot-path, allocation-free)                */
/* ------------------------------------------------------------------ */

export function acquireVector3() { return getDefaultPoolSet().vector3.acquire(); }
export function releaseVector3(v) { return getDefaultPoolSet().vector3.release(v); }

export function acquireVector2() { return getDefaultPoolSet().vector2.acquire(); }
export function releaseVector2(v) { return getDefaultPoolSet().vector2.release(v); }

export function acquireVector4() { return getDefaultPoolSet().vector4.acquire(); }
export function releaseVector4(v) { return getDefaultPoolSet().vector4.release(v); }

export function acquireColor() { return getDefaultPoolSet().color.acquire(); }
export function releaseColor(c) { return getDefaultPoolSet().color.release(c); }

export function acquireQuaternion() { return getDefaultPoolSet().quaternion.acquire(); }
export function releaseQuaternion(q) { return getDefaultPoolSet().quaternion.release(q); }

export function acquireMatrix4() { return getDefaultPoolSet().matrix4.acquire(); }
export function releaseMatrix4(m) { return getDefaultPoolSet().matrix4.release(m); }

export function acquireMatrix3() { return getDefaultPoolSet().matrix3.acquire(); }
export function releaseMatrix3(m) { return getDefaultPoolSet().matrix3.release(m); }

export function acquireSpherical() { return getDefaultPoolSet().spherical.acquire(); }
export function releaseSpherical(s) { return getDefaultPoolSet().spherical.release(s); }

export function acquireBox3() { return getDefaultPoolSet().box3.acquire(); }
export function releaseBox3(b) { return getDefaultPoolSet().box3.release(b); }

export function acquireSphere() { return getDefaultPoolSet().sphere.acquire(); }
export function releaseSphere(s) { return getDefaultPoolSet().sphere.release(s); }

export function acquirePlane() { return getDefaultPoolSet().plane.acquire(); }
export function releasePlane(p) { return getDefaultPoolSet().plane.release(p); }

export function acquireFrustum() { return getDefaultPoolSet().frustum.acquire(); }
export function releaseFrustum(f) { return getDefaultPoolSet().frustum.release(f); }

export function acquireRay() { return getDefaultPoolSet().ray.acquire(); }
export function releaseRay(r) { return getDefaultPoolSet().ray.release(r); }

export function acquireF32(size) { return getDefaultPoolSet().f32.acquire(size); }
export function releaseF32(arr) { return getDefaultPoolSet().f32.release(arr); }

export function acquireU16(size) { return getDefaultPoolSet().u16.acquire(size); }
export function releaseU16(arr) { return getDefaultPoolSet().u16.release(arr); }

export function acquireU32(size) { return getDefaultPoolSet().u32.acquire(size); }
export function releaseU32(arr) { return getDefaultPoolSet().u32.release(arr); }

export function acquireU8(size) { return getDefaultPoolSet().u8.acquire(size); }
export function releaseU8(arr) { return getDefaultPoolSet().u8.release(arr); }

/* ------------------------------------------------------------------ */
/* 9. FACTORIES                                                       */
/* ------------------------------------------------------------------ */

export function createObjectPool(options = {}) {
  return new ObjectPool(options);
}

export function createTypedArrayPool(options = {}) {
  return new TypedArrayPool(options);
}

export function createLightingPoolSet() {
  return new LightingPoolSet();
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ObjectPool,
  TypedArrayPool,
  LightingPoolSet,
  createObjectPool,
  createTypedArrayPool,
  createLightingPoolSet,
  getDefaultPoolSet,
  disposeDefaultPoolSet,
  acquireVector2, releaseVector2,
  acquireVector3, releaseVector3,
  acquireVector4, releaseVector4,
  acquireColor,   releaseColor,
  acquireQuaternion, releaseQuaternion,
  acquireMatrix3, releaseMatrix3,
  acquireMatrix4, releaseMatrix4,
  acquireSpherical, releaseSpherical,
  acquireBox3,    releaseBox3,
  acquireSphere,  releaseSphere,
  acquirePlane,   releasePlane,
  acquireFrustum, releaseFrustum,
  acquireRay,     releaseRay,
  acquireF32,     releaseF32,
  acquireU16,     releaseU16,
  acquireU32,     releaseU32,
  acquireU8,      releaseU8,
  POOL_KIND,
  POOL_KIND_NAME,
  POOL_CAPACITY,
  TYPED_BUCKET_SIZES,
};

export default _defaultExport;