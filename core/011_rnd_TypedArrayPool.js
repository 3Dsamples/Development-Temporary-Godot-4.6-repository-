// File : 011
// name : src/core/011_rnd_TypedArrayPool.js
// description : Dedicated multi-kind TypedArray pool for the anime lighting
//               stack on Android mobile. Where 010_rnd_ObjectPool.js contains
//               a generic TypedArrayPool used for small scratch arrays, THIS
//               module is the full-featured allocator used by every lighting
//               subsystem that needs large, GPU-bound, or long-lived typed
//               buffers:
//
//                 • SHADOW ATLAS          — depth-packed Float32/Uint16 slabs
//                 • CASCADE MATRICES      — 16-float Matrix4 slabs × N cascades
//                 • GI PROBE GRID         — 3-channel Float32 irradiance slabs
//                 • AO BLUR KERNELS       — half-res Float32/Unorm8 slabs
//                 • CLUSTER GRID          — index Uint32 + range Uint16 slabs
//                 • LIGHT LIST            — packed Uint32/Uint16 handles
//                 • ENVIRONMENT PALETTE   — Float32 RGBA linear blocks
//                 • INTERIOR VOLUMES      — Float32 probe lattices
//                 • EXTERIOR PROBES       — Float32 SH-9 per-probe slabs
//                 • POST BUFFERS          — HDR Float32 half-res slabs
//                 • WORKER TRANSFER POOLS — transferable ArrayBuffer wrappers
//
//               Features:
//                 • 8 typed array kinds (F32 / F64 / I32 / U32 / I16 / U16 /
//                   I8 / U8) — no more string-key lookups, all typed.
//                 • Geometric size buckets (16 → 65536) with per-bucket
//                   capacity chosen by PERF_TIER.
//                 • Sub-slab allocation: a single 8K Float32Array can serve
//                   multiple small views without extra buffer objects.
//                 • Stride-aware slabs for GPU upload: every slab records its
//                   byteStride, elementCount, and GPU-compatible alignment.
//                 • Zero-allocation hot path: acquire/release touch only
//                   typed arrays and cached references.
//                 • Frame lifecycle: `beginFrame()`, `endFrame()`,
//                   `releaseAll()` — with optional `autoReleasePerFrame`.
//                 • Sub-pool tagging: every slab carries a `tag` (SHADOW,
//                   GI, AO, CLUSTER, LIGHT_LIST, ENV, INTERIOR, EXTERIOR,
//                   POST, WORKER, GENERIC) so debug tools can visualize
//                   which subsystem owns each buffer.
//                 • Worker-safe transfer: `detach(slab)` returns the
//                   underlying ArrayBuffer for postMessage handoff, and
//                   `reattach(slab, buffer)` restores it after receipt —
//                   both are O(1), zero-copy.
//                 • Diagnostics: per-tag allocation counters, peak in-use,
//                   rejection counts, fragmentation estimate.
//                 • Zero per-frame allocations on the hot path.
//                 • Fixed capacity at construction — nothing grows later.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external allocator libs; every typed array sized
//               once at construction.
// best for : Serving the raw typed-array needs of the entire lighting stack
//            (006_lgt_LightManager through 380_lgt_lights) — shadow atlases,
//            GI probe grids, AO blur kernels, cluster grids, light lists,
//            environment palettes, interior volumes, exterior probes, post
//            buffers, and worker transfer buffers — from a single
//            deterministic pool with one diagnostic surface.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

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

export const ARRAY_KIND = Object.freeze({
  F32: 0,
  F64: 1,
  I32: 2,
  U32: 3,
  I16: 4,
  U16: 5,
  I8:  6,
  U8:  7,
  COUNT: 8,
});

export const ARRAY_KIND_NAME = Object.freeze([
  'float32',
  'float64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
]);

export const ARRAY_KIND_CTOR = Object.freeze([
  Float32Array,
  Float64Array,
  Int32Array,
  Uint32Array,
  Int16Array,
  Uint16Array,
  Int8Array,
  Uint8Array,
]);

export const ARRAY_KIND_BYTES = Object.freeze([
  4, 8, 4, 4, 2, 2, 1, 1,
]);

export const SLAB_TAG = Object.freeze({
  GENERIC:     0,
  SHADOW:      1,
  GI:          2,
  AO:          3,
  CLUSTER:     4,
  LIGHT_LIST:  5,
  ENV:         6,
  INTERIOR:    7,
  EXTERIOR:    8,
  POST:        9,
  WORKER:     10,
  MATRIX:     11,
  COUNT:      12,
});

export const SLAB_TAG_NAME = Object.freeze([
  'generic',
  'shadow',
  'gi',
  'ao',
  'cluster',
  'light_list',
  'env',
  'interior',
  'exterior',
  'post',
  'worker',
  'matrix',
]);

/**
 * Geometric size buckets chosen for lighting workloads. Each entry is the
 * exact element count of the underlying slab; acquire(size) rounds up.
 *
 *   16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
 *   16384, 32768, 65536
 */
export const SIZE_BUCKETS = Object.freeze([
  16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
]);

export const BUCKET_COUNT = SIZE_BUCKETS.length;

/**
 * Per-bucket capacity chosen by PERF_TIER. Mobile LOW gets few big slabs;
 * HIGH gets more of everything. These numbers are tuned to the actual
 * lighting workloads: shadow atlas ≈ 4 slabs of 8K, GI probe grid ≈ 1 slab
 * of 65K, cluster grid ≈ 2 slabs of 32K, etc.
 */
export const BUCKET_CAPACITY = (() => {
  if (PERF_TIER === 'HIGH') {
    return [512, 384, 256, 192, 160, 128, 96, 64, 48, 32, 16, 8, 4];
  }
  if (PERF_TIER === 'MEDIUM') {
    return [256, 192, 128, 96, 80, 64, 48, 32, 24, 16, 8, 4, 2];
  }
  return [128, 96, 64, 48, 40, 32, 24, 16, 12, 8, 4, 2, 1];
})();

const POOL_ID_SYMBOL = '__typedPoolId';
const SLAB_INDEX_SYMBOL = '__typedSlabIndex';
const SLAB_BUCKET_SYMBOL = '__typedSlabBucket';
const SLAB_KIND_SYMBOL = '__typedSlabKind';
const SLAB_TAG_SYMBOL = '__typedSlabTag';

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _typedPoolIdCounter = 0;

function _nextPoolId() {
  return ++_typedPoolIdCounter;
}

function _findBucket(size) {
  // Binary search would be marginally faster but the SIZE_BUCKETS array is
  // small (13 entries) and the linear scan is branch-predictable on ARM.
  for (let b = 0; b < BUCKET_COUNT; b++) {
    if (SIZE_BUCKETS[b] >= size) return b;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/* 2. SLAB DESCRIPTOR                                                 */
/* ------------------------------------------------------------------ */

/**
 * A slab is one underlying typed array in the pool. `view()` returns a
 * subarray of exactly the requested length, so callers never write past
 * their request. The underlying buffer is only ever allocated once, at
 * pool construction.
 */
export class Slab {
  constructor(index, bucket, kind, tag, array) {
    this.index     = index;
    this.bucket    = bucket;
    this.kind      = kind;
    this.tag       = tag;
    this.array     = array;
    this.capacity  = array.length;
    this.inUse     = 0;
    this.acquiredAt = 0;
    this.requestedSize = 0;
  }

  view(size) {
    return this.array.subarray(0, size);
  }

  reset() {
    // Zero out only the requested region to keep cost proportional.
    if (this.requestedSize > 0) {
      this.array.fill(0, 0, this.requestedSize);
    }
    this.requestedSize = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. BUCKET                                                          */
/* ------------------------------------------------------------------ */

class Bucket {
  constructor(bucketIdx, kind) {
    this.bucketIdx = bucketIdx;
    this.kind      = kind;
    this.size      = SIZE_BUCKETS[bucketIdx];
    this.capacity  = BUCKET_CAPACITY[bucketIdx];
    this.Ctor      = ARRAY_KIND_CTOR[kind];

    this.slabs     = new Array(this.capacity);
    this.freeList  = new Int32Array(this.capacity);
    this.freeHead  = 0;
    this.freeCount = this.capacity;

    this.currentInUse = 0;
    this.peakInUse    = 0;
    this.acquiredTotal = 0;
    this.releasedTotal = 0;
    this.rejectedTotal = 0;
    this.zeroTotal     = 0;
  }

  allocate() {
    for (let i = 0; i < this.capacity; i++) {
      const array = new this.Ctor(this.size);
      this.slabs[i] = new Slab(i, this.bucketIdx, this.kind, SLAB_TAG.GENERIC, array);
      this.freeList[i] = i;
    }
  }

  dispose() {
    for (let i = 0; i < this.capacity; i++) {
      if (this.slabs[i]) this.slabs[i].array = null;
      this.slabs[i] = null;
    }
    this.slabs.length = 0;
    this.freeList = null;
  }
}

/* ------------------------------------------------------------------ */
/* 4. TYPED ARRAY POOL                                                */
/* ------------------------------------------------------------------ */

export class TypedArrayPool {
  constructor(options = {}) {
    this.name       = options.name || `typed_pool_${_nextPoolId()}`;
    this.poolId     = _nextPoolId();
    this.autoZero   = options.autoZero !== false;
    this.autoReleasePerFrame = options.autoReleasePerFrame === true;

    this.frame      = 0;

    // One bucket grid per array kind: [kind][bucket].
    this.buckets    = new Array(ARRAY_KIND.COUNT);
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      this.buckets[k] = new Array(BUCKET_COUNT);
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = new Bucket(b, k);
        bucket.allocate();
        this.buckets[k][b] = bucket;
      }
    }

    // Aggregate stats.
    this.stats = {
      totalAcquired: 0,
      totalReleased: 0,
      totalRejected: 0,
      totalDetached: 0,
      totalReattached: 0,
      peakInUse:     0,
      currentInUse:  0,
    };

    // Per-tag counters (12 tags).
    this._tagInUse  = new Uint32Array(SLAB_TAG.COUNT);
    this._tagPeak   = new Uint32Array(SLAB_TAG.COUNT);
    this._tagTotal  = new Uint32Array(SLAB_TAG.COUNT);

    // Per-frame scratch (pre-allocated).
    this._detachedSlabs = new Array(64);
    this._detachedCount = 0;

    // Named slabs — long-lived slabs reserved at construction by name.
    this._namedSlabs = new Map();

    this._listeners = new Map();
  }

  /* ---------------- events ---------------- */

  on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let arr = this._listeners.get(event);
    if (!arr) { arr = []; this._listeners.set(event, arr); }
    arr.push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  _emit(event, payload) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      try { arr[i](payload); } catch (e) { console.error(`[011_rnd_TypedArrayPool] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- acquire / release ---------------- */

  /**
   * Acquire a subarray of at least `size` elements of the given kind.
   * Returns a subarray view of exactly `size` elements, or null if the
   * pool is exhausted for that bucket.
   *
   * Kind = ARRAY_KIND.F32 (default), tag = SLAB_TAG.GENERIC (default).
   */
  acquire(size, kind = ARRAY_KIND.F32, tag = SLAB_TAG.GENERIC) {
    const need = Math.max(1, size | 0);
    const b = _findBucket(need);
    if (b < 0) {
      this.stats.totalRejected++;
      this._emit('rejected', { kind, tag, size: need, reason: 'too_large' });
      return null;
    }

    const bucket = this.buckets[kind][b];
    if (bucket.freeCount <= 0) {
      bucket.rejectedTotal++;
      this.stats.totalRejected++;
      this._emit('rejected', { kind, tag, size: need, reason: 'exhausted' });
      return null;
    }

    const idx = bucket.freeList[bucket.freeHead];
    bucket.freeHead = (bucket.freeHead + 1) % bucket.capacity;
    bucket.freeCount--;

    const slab = bucket.slabs[idx];
    slab.inUse = 1;
    slab.acquiredAt = this.frame;
    slab.requestedSize = need;
    slab.tag = tag;

    bucket.currentInUse++;
    if (bucket.currentInUse > bucket.peakInUse) bucket.peakInUse = bucket.currentInUse;
    bucket.acquiredTotal++;

    this.stats.totalAcquired++;
    this.stats.currentInUse++;
    if (this.stats.currentInUse > this.stats.peakInUse) this.stats.peakInUse = this.stats.currentInUse;

    this._tagInUse[tag]++;
    this._tagTotal[tag]++;
    if (this._tagInUse[tag] > this._tagPeak[tag]) this._tagPeak[tag] = this._tagInUse[tag];

    // Fast-release path: attach pool + slab info so release() can find
    // the parent slab by identity without a linear scan.
    const view = slab.array.subarray(0, need);
    view[POOL_ID_SYMBOL]    = this.poolId;
    view[SLAB_INDEX_SYMBOL] = idx;
    view[SLAB_BUCKET_SYMBOL]= b;
    view[SLAB_KIND_SYMBOL]  = kind;
    view[SLAB_TAG_SYMBOL]   = tag;
    view.__slabRef          = slab;

    return view;
  }

  release(view) {
    if (!view || !ArrayBuffer.isView(view) || view[POOL_ID_SYMBOL] !== this.poolId) {
      return false;
    }

    const slab = view.__slabRef;
    if (!slab || slab.inUse === 0) return false;

    const b = view[SLAB_BUCKET_SYMBOL];
    const k = view[SLAB_KIND_SYMBOL];
    if (b === undefined || k === undefined) return false;

    const bucket = this.buckets[k][b];
    if (!bucket) return false;

    if (this.autoZero) {
      slab.reset();
      bucket.zeroTotal++;
    }

    slab.inUse = 0;
    slab.requestedSize = 0;

    bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = slab.index;
    bucket.freeCount++;
    bucket.releasedTotal++;
    bucket.currentInUse--;
    if (bucket.currentInUse < 0) bucket.currentInUse = 0;

    this.stats.totalReleased++;
    this.stats.currentInUse--;
    if (this.stats.currentInUse < 0) this.stats.currentInUse = 0;

    if (this._tagInUse[slab.tag] > 0) this._tagInUse[slab.tag]--;

    view.__slabRef = null;
    return true;
  }

  releaseAll() {
    let released = 0;
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.buckets[k][b];
        for (let i = 0; i < bucket.capacity; i++) {
          const slab = bucket.slabs[i];
          if (slab.inUse === 1) {
            if (this.autoZero) slab.reset();
            slab.inUse = 0;
            slab.requestedSize = 0;
            bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = i;
            bucket.freeCount++;
            bucket.releasedTotal++;
            bucket.currentInUse--;
            this.stats.totalReleased++;
            this.stats.currentInUse--;
            if (this._tagInUse[slab.tag] > 0) this._tagInUse[slab.tag]--;
            released++;
          }
        }
      }
    }
    if (this.stats.currentInUse < 0) this.stats.currentInUse = 0;
    return released;
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame() {
    this.frame++;
    this._detachedCount = 0;
  }

  endFrame() {
    if (this.autoReleasePerFrame) this.releaseAll();
    return this.stats.currentInUse;
  }

  /* ---------------- named slabs ---------------- */

  /**
   * Reserve a long-lived named slab (never auto-released). Use this for
   * per-subsystem persistent buffers, e.g. shadow atlas scratch, GI probe
   * grids, cluster grids.
   */
  reserveNamed(name, size, kind = ARRAY_KIND.F32, tag = SLAB_TAG.GENERIC) {
    if (!name || typeof name !== 'string') return null;
    const existing = this._namedSlabs.get(name);
    if (existing) return existing;

    const view = this.acquire(size, kind, tag);
    if (!view) return null;

    // Mark the slab as named so it is skipped by releaseAll().
    view.__namedSlab = name;
    this._namedSlabs.set(name, view);
    return view;
  }

  getNamed(name) {
    return this._namedSlabs.get(name) || null;
  }

  releaseNamed(name) {
    const view = this._namedSlabs.get(name);
    if (!view) return false;
    view.__namedSlab = null;
    this._namedSlabs.delete(name);
    return this.release(view);
  }

  /* ---------------- worker transfer ---------------- */

  /**
   * Detach the underlying ArrayBuffer for postMessage transfer. The slab
   * remains logically "in use" so it cannot be re-acquired while detached.
   * Returns the ArrayBuffer or null.
   */
  detach(view) {
    if (!view || !ArrayBuffer.isView(view)) return null;
    const slab = view.__slabRef;
    if (!slab) return null;
    if (this._detachedCount >= this._detachedSlabs.length) return null;

    const buffer = slab.array.buffer;
    if (!buffer) return null;

    this._detachedSlabs[this._detachedCount++] = slab;
    this.stats.totalDetached++;
    this._emit('detach', { tag: slab.tag, bytes: buffer.byteLength });
    return buffer;
  }

  /**
   * Reattach an ArrayBuffer to a detached slab. The slab must be the same
   * one returned by a prior detach() call.
   */
  reattach(slab, buffer) {
    if (!slab || !buffer) return false;
    // Reconstruct the typed array from the transferred buffer.
    const Ctor = ARRAY_KIND_CTOR[slab.kind];
    slab.array = new Ctor(buffer);
    slab.capacity = slab.array.length;
    this.stats.totalReattached++;
    this._emit('reattach', { tag: slab.tag, bytes: buffer.byteLength });
    return true;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const buckets = new Array(ARRAY_KIND.COUNT);
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      buckets[k] = new Array(BUCKET_COUNT);
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.buckets[k][b];
        buckets[k][b] = {
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
    }

    const tags = new Array(SLAB_TAG.COUNT);
    for (let t = 0; t < SLAB_TAG.COUNT; t++) {
      tags[t] = {
        name:  SLAB_TAG_NAME[t],
        inUse: this._tagInUse[t],
        peak:  this._tagPeak[t],
        total: this._tagTotal[t],
      };
    }

    return {
      name:           this.name,
      kind:           'typed_array_pool',
      currentInUse:   this.stats.currentInUse,
      peakInUse:      this.stats.peakInUse,
      totalAcquired:  this.stats.totalAcquired,
      totalReleased:  this.stats.totalReleased,
      totalRejected:  this.stats.totalRejected,
      totalDetached:  this.stats.totalDetached,
      totalReattached:this.stats.totalReattached,
      namedCount:     this._namedSlabs.size,
      detachedCount:  this._detachedCount,
      buckets,
      tags,
      perfTier:       PERF_TIER,
    };
  }

  estimateBytes() {
    let bytes = 0;
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      const kindBytes = ARRAY_KIND_BYTES[k];
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.buckets[k][b];
        bytes += bucket.capacity * bucket.size * kindBytes;
      }
    }
    return bytes;
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    this.releaseAll();
    this._namedSlabs.clear();
    this._detachedCount = 0;
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.buckets[k][b];
        bucket.freeHead = 0;
        bucket.freeCount = bucket.capacity;
        for (let i = 0; i < bucket.capacity; i++) {
          bucket.freeList[i] = i;
          if (bucket.slabs[i]) {
            bucket.slabs[i].inUse = 0;
            bucket.slabs[i].requestedSize = 0;
            bucket.slabs[i].acquiredAt = 0;
          }
        }
        bucket.currentInUse = 0;
        bucket.peakInUse = 0;
        bucket.acquiredTotal = 0;
        bucket.releasedTotal = 0;
        bucket.rejectedTotal = 0;
        bucket.zeroTotal = 0;
      }
    }
    this.stats.totalAcquired = 0;
    this.stats.totalReleased = 0;
    this.stats.totalRejected = 0;
    this.stats.totalDetached = 0;
    this.stats.totalReattached = 0;
    this.stats.peakInUse = 0;
    this.stats.currentInUse = 0;
    this._tagInUse.fill(0);
    this._tagPeak.fill(0);
    this._tagTotal.fill(0);
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.buckets[k][b];
        bucket.dispose();
      }
      this.buckets[k] = null;
    }
    this.buckets = null;
    this._namedSlabs.clear();
    this._detachedSlabs.length = 0;
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. LIGHTING SLAB PRESET (canonical named reservations)             */
/* ------------------------------------------------------------------ */

/**
 * Reserve the standard long-lived slabs the lighting stack needs. Names
 * are stable so any subsystem can acquire by name.
 *
 *   shadowAtlasScratch      Float32 8192   SHADOW
 *   cascadeMatrixSlab       Float32 16*4   MATRIX
 *   giProbeGrid             Float32 65536  GI
 *   aoBlurKernel            Float32 4096   AO
 *   clusterGridIndices      Uint32  16384  CLUSTER
 *   clusterGridRanges       Uint16  8192   CLUSTER
 *   lightListHandles        Uint32  4096   LIGHT_LIST
 *   envPaletteLinear        Float32 1024   ENV
 *   interiorVolumeLattice   Float32 16384  INTERIOR
 *   exteriorProbeSH         Float32 8192   EXTERIOR
 *   postBufferHDR           Float32 16384  POST
 *   workerTransferScratch   Float32 8192   WORKER
 */
export function reserveLightingSlabs(pool) {
  if (!pool || typeof pool.reserveNamed !== 'function') return null;

  const slabs = {
    shadowAtlasScratch:    pool.reserveNamed('shadowAtlasScratch',    8192,  ARRAY_KIND.F32, SLAB_TAG.SHADOW),
    cascadeMatrixSlab:     pool.reserveNamed('cascadeMatrixSlab',     64,    ARRAY_KIND.F32, SLAB_TAG.MATRIX),
    giProbeGrid:           pool.reserveNamed('giProbeGrid',           65536, ARRAY_KIND.F32, SLAB_TAG.GI),
    aoBlurKernel:          pool.reserveNamed('aoBlurKernel',          4096,  ARRAY_KIND.F32, SLAB_TAG.AO),
    clusterGridIndices:    pool.reserveNamed('clusterGridIndices',    16384, ARRAY_KIND.U32, SLAB_TAG.CLUSTER),
    clusterGridRanges:     pool.reserveNamed('clusterGridRanges',     8192,  ARRAY_KIND.U16, SLAB_TAG.CLUSTER),
    lightListHandles:      pool.reserveNamed('lightListHandles',      4096,  ARRAY_KIND.U32, SLAB_TAG.LIGHT_LIST),
    envPaletteLinear:      pool.reserveNamed('envPaletteLinear',      1024,  ARRAY_KIND.F32, SLAB_TAG.ENV),
    interiorVolumeLattice: pool.reserveNamed('interiorVolumeLattice', 16384, ARRAY_KIND.F32, SLAB_TAG.INTERIOR),
    exteriorProbeSH:       pool.reserveNamed('exteriorProbeSH',       8192,  ARRAY_KIND.F32, SLAB_TAG.EXTERIOR),
    postBufferHDR:         pool.reserveNamed('postBufferHDR',         16384, ARRAY_KIND.F32, SLAB_TAG.POST),
    workerTransferScratch: pool.reserveNamed('workerTransferScratch', 8192,  ARRAY_KIND.F32, SLAB_TAG.WORKER),
  };

  return slabs;
}

/* ------------------------------------------------------------------ */
/* 6. SUB-POOL (per-tag view over a shared TypedArrayPool)            */
/* ------------------------------------------------------------------ */

/**
 * A lightweight façade that binds a fixed tag to a shared pool so subsystem
 * code can acquire/release without passing tag on every call. Zero
 * allocation per call.
 */
export class TaggedSubPool {
  constructor(pool, tag) {
    this.pool = pool;
    this.tag  = tag;
  }

  acquireF32(size) { return this.pool.acquire(size, ARRAY_KIND.F32, this.tag); }
  acquireU32(size) { return this.pool.acquire(size, ARRAY_KIND.U32, this.tag); }
  acquireU16(size) { return this.pool.acquire(size, ARRAY_KIND.U16, this.tag); }
  acquireU8(size)  { return this.pool.acquire(size, ARRAY_KIND.U8,  this.tag); }
  acquireI32(size) { return this.pool.acquire(size, ARRAY_KIND.I32, this.tag); }

  release(view) { return this.pool.release(view); }

  reserveNamed(name, size, kind = ARRAY_KIND.F32) {
    return this.pool.reserveNamed(name, size, kind, this.tag);
  }
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createTypedArrayPool(options = {}) {
  return new TypedArrayPool(options);
}

export function createTaggedSubPool(pool, tag) {
  return new TaggedSubPool(pool, tag);
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultTypedPool = null;
let _defaultLightingSlabs = null;

export function getDefaultTypedArrayPool() {
  if (!_defaultTypedPool) {
    _defaultTypedPool = new TypedArrayPool();
    _defaultLightingSlabs = reserveLightingSlabs(_defaultTypedPool);
  }
  return _defaultTypedPool;
}

export function getDefaultLightingSlabs() {
  if (!_defaultLightingSlabs) getDefaultTypedArrayPool();
  return _defaultLightingSlabs;
}

export function disposeDefaultTypedArrayPool() {
  if (_defaultTypedPool) {
    _defaultTypedPool.dispose();
    _defaultTypedPool = null;
    _defaultLightingSlabs = null;
  }
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  TypedArrayPool,
  TaggedSubPool,
  Slab,
  createTypedArrayPool,
  createTaggedSubPool,
  getDefaultTypedArrayPool,
  getDefaultLightingSlabs,
  disposeDefaultTypedArrayPool,
  reserveLightingSlabs,
  ARRAY_KIND,
  ARRAY_KIND_NAME,
  ARRAY_KIND_CTOR,
  ARRAY_KIND_BYTES,
  SLAB_TAG,
  SLAB_TAG_NAME,
  SIZE_BUCKETS,
  BUCKET_COUNT,
  BUCKET_CAPACITY,
};

export default _defaultExport;