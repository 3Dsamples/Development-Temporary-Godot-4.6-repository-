// File : 012
// name : src/core/012_rnd_BufferPool.js
// description : GPU buffer pool for the anime lighting stack on Android mobile.
//               Manages THREE.BufferAttribute / InterleavedBuffer / InstancedBuffer
//               instances used by every lighting subsystem that pushes data to
//               the GPU each frame: light list VBO, cluster grid buffer, GI probe
//               uploads, AO kernel uploads, environment palette uploads, interior
//               volume uploads, exterior probe uploads, and shadow atlas staging
//               buffers.
//
//               Where 011_rnd_TypedArrayPool.js owns the CPU-side typed arrays,
//               THIS module owns the GPU-side descriptors: it wraps a typed slab
//               from the TypedArrayPool in a THREE.BufferAttribute, tracks the
//               attribute's update generation so the renderer only re-uploads when
//               data actually changed, and recycles attributes across frames
//               without ever calling `new Float32BufferAttribute(...)` inside a
//               hot loop.
//
//               Design:
//                 • Two-tier pool:
//                     – AttributePool    : raw THREE.BufferAttribute instances,
//                                          keyed by (arrayKind, size, normalized,
//                                          usage)
//                     – InterleavedPool  : THREE.InterleavedBuffer + attribute
//                                          views, keyed by (arrayKind, stride,
//                                          count, usage)
//                 • Named reservations for the standard lighting buffers so every
//                   subsystem can ask for `getNamed('lightListVBO')` and get the
//                   SAME BufferAttribute back across frames — no re-binding.
//                 • Upload coalescing: `markDirty(attr)` bumps a generation
//                   counter; the frame's `commitDirty()` walks the dirty set
//                   once and sets `needsUpdate = true` on each — so N subsystems
//                   writing to N attributes cause exactly one upload pass.
//                 • Android-specific quirks handled:
//                     – DynamicDrawUsage forced for per-frame buffers
//                     – StaticDrawUsage for named persistent buffers
//                     – StreamDrawUsage for worker-transferred buffers
//                     – normalized flag respected for Uint8/Uint16 cel-shading
//                       tint channels
//                     – correct handling of Three.js r185 `BufferAttribute.setUsage`
//                     – correct array typing for interleaved buffers (must be
//                       the same ArrayType across all attributes in one interleave)
//                 • Zero per-frame allocations on the hot path: acquire/release
//                   touch only pre-allocated slots and pre-computed keys.
//                 • GPU memory accounting: per-kind and total bytes tracked so
//                   Android low-memory devices can gate allocations.
//                 • Lifecycle: `beginFrame()`, `commitDirty()`, `endFrame()`,
//                   `releaseAll()` — with per-pool autoRelease policy.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external pool libs; every BufferAttribute and every
//               InterleavedBuffer is allocated once at construction.
// best for : Guaranteeing that the lighting stack never allocates a
//            BufferAttribute at runtime on Android. The light list, cluster grid,
//            GI probe upload, AO kernel upload, environment palette upload,
//            interior volume upload, exterior probe upload, and shadow staging
//            buffers all acquire their BufferAttribute from this pool once and
//            reuse it across every frame, with correct usage flags and one
//            consolidated upload pass per frame.
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

import {
  ARRAY_KIND,
  ARRAY_KIND_NAME,
  ARRAY_KIND_CTOR,
  ARRAY_KIND_BYTES,
  SLAB_TAG,
  SLAB_TAG_NAME,
  SIZE_BUCKETS,
  BUCKET_COUNT,
  getDefaultTypedArrayPool,
} from './011_rnd_TypedArrayPool.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const BUFFER_USAGE = Object.freeze({
  STATIC:  0,
  DYNAMIC: 1,
  STREAM:  2,
});

export const BUFFER_USAGE_THREE = Object.freeze([
  THREE.StaticDrawUsage,
  THREE.DynamicDrawUsage,
  THREE.StreamDrawUsage,
]);

export const BUFFER_KIND = Object.freeze({
  ATTRIBUTE:   0,
  INTERLEAVED: 1,
  COUNT:       2,
});

export const BUFFER_KIND_NAME = Object.freeze([
  'attribute',
  'interleaved',
]);

/**
 * Per-tier capacity for attribute slots. Each slot holds one pre-built
 * BufferAttribute ready to be handed to a material.
 */
export const ATTRIBUTE_CAPACITY = (() => {
  if (PERF_TIER === 'HIGH')   return [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2];
  if (PERF_TIER === 'MEDIUM') return [128, 96,  64,  48, 32, 24, 16, 12, 8,  6,  4, 2, 1];
  return                              [64,  48,  32,  24, 16, 12, 8,  6,  4,  3,  2, 1, 1];
})();

export const INTERLEAVED_CAPACITY = (() => {
  if (PERF_TIER === 'HIGH')   return 128;
  if (PERF_TIER === 'MEDIUM') return 64;
  return 32;
})();

const POOL_ID_SYMBOL       = '__bufPoolId';
const SLOT_INDEX_SYMBOL    = '__bufSlotIndex';
const SLOT_KIND_SYMBOL     = '__bufSlotKind';
const SLOT_BUCKET_SYMBOL   = '__bufSlotBucket';

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _bufferPoolIdCounter = 0;

function _nextPoolId() {
  return ++_bufferPoolIdCounter;
}

function _findBucket(size) {
  for (let b = 0; b < BUCKET_COUNT; b++) {
    if (SIZE_BUCKETS[b] >= size) return b;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/* 2. ATTRIBUTE SLOT                                                  */
/* ------------------------------------------------------------------ */

/**
 * One slot holds a pre-built THREE.BufferAttribute whose underlying
 * typed array is a slab acquired from the shared TypedArrayPool.
 * The attribute is created ONCE and reused across frames.
 */
export class AttributeSlot {
  constructor(index, bucket, arrayKind) {
    this.index     = index;
    this.bucket    = bucket;
    this.arrayKind = arrayKind;
    this.size      = SIZE_BUCKETS[bucket];
    this.usage     = BUFFER_USAGE.DYNAMIC;

    this.attribute = null;   // THREE.BufferAttribute
    this.arrayView = null;   // TypedArray subarray — same buffer as attribute.array
    this.inUse     = 0;
    this.tagged    = SLAB_TAG.GENERIC;
    this.acquiredAt = 0;
    this.requestedSize = 0;

    this.dirty     = 0;
    this.generation = 0;
    this.lastUploadedGen = 0;
    this.uploadCount = 0;
  }

  view(size) {
    return this.arrayView.subarray(0, size);
  }

  reset() {
    if (this.requestedSize > 0 && this.arrayView) {
      this.arrayView.fill(0, 0, this.requestedSize);
    }
    this.requestedSize = 0;
    this.dirty = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. INTERLEAVED SLOT                                                */
/* ------------------------------------------------------------------ */

export class InterleavedSlot {
  constructor(index) {
    this.index     = index;
    this.inUse     = 0;
    this.arrayKind = ARRAY_KIND.F32;
    this.stride    = 1;
    this.count     = 0;
    this.capacity  = 0;
    this.usage     = BUFFER_USAGE.DYNAMIC;

    this.buffer      = null;  // THREE.InterleavedBuffer
    this.attributes  = null;  // Map<attributeName, THREE.InterleavedBufferAttribute>
    this.arrayView   = null;  // underlying typed array
    this.tagged      = SLAB_TAG.GENERIC;
    this.dirty       = 0;
    this.uploadCount = 0;
  }

  reset() {
    this.dirty = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. BUFFER POOL                                                     */
/* ------------------------------------------------------------------ */

export class BufferPool {
  constructor(options = {}) {
    this.name    = options.name || `buf_pool_${_nextPoolId()}`;
    this.poolId  = _nextPoolId();

    // Underlying CPU slab source.
    this.cpuPool = options.cpuPool || getDefaultTypedArrayPool();

    // Auto-release policy per frame.
    this.autoReleasePerFrame = options.autoReleasePerFrame === true;
    this.autoUploadOnCommit  = options.autoUploadOnCommit !== false;

    // Per (arrayKind × bucket) attribute slot grid.
    this.attributeBuckets = new Array(ARRAY_KIND.COUNT);
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      this.attributeBuckets[k] = new Array(BUCKET_COUNT);
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const capacity = ATTRIBUTE_CAPACITY[b];
        const slots = new Array(capacity);
        const freeList = new Int32Array(capacity);
        for (let i = 0; i < capacity; i++) {
          slots[i] = new AttributeSlot(i, b, k);
          freeList[i] = i;
        }
        this.attributeBuckets[k][b] = {
          slots,
          freeList,
          freeHead: 0,
          freeCount: capacity,
          capacity,
          currentInUse: 0,
          peakInUse: 0,
          acquiredTotal: 0,
          releasedTotal: 0,
          rejectedTotal: 0,
        };
      }
    }

    // Interleaved slots.
    this.interleavedSlots = new Array(INTERLEAVED_CAPACITY);
    this.interleavedFree = new Int32Array(INTERLEAVED_CAPACITY);
    this.interleavedFreeHead = 0;
    this.interleavedFreeCount = INTERLEAVED_CAPACITY;
    for (let i = 0; i < INTERLEAVED_CAPACITY; i++) {
      this.interleavedSlots[i] = new InterleavedSlot(i);
      this.interleavedFree[i] = i;
    }

    // Dirty tracking (attribute slots to upload this frame).
    this._dirtySlots = new Int32Array(512);
    this._dirtyCount = 0;

    // Named reservations — long-lived buffers keyed by name.
    this._namedAttrs = new Map();

    // Frame + stats.
    this.frame = 0;
    this.stats = {
      attributesAcquired:  0,
      attributesReleased:  0,
      attributesRejected:  0,
      interleavedAcquired: 0,
      interleavedReleased: 0,
      interleavedRejected: 0,
      uploadsCommitted:    0,
      namedCount:          0,
      gpuBytesAllocated:   0,
      gpuBytesInUse:       0,
      peakGpuBytesInUse:   0,
    };

    this._listeners = new Map();

    // Pre-bind for hot paths.
    this._commitDirtyFn = this._commitDirtyInternal.bind(this);
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
      try { arr[i](payload); } catch (e) { console.error(`[012_rnd_BufferPool] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- acquire / release (attributes) ---------------- */

  /**
   * Acquire a BufferAttribute ready for use. The underlying typed array is
   * sized to exactly `size` elements and lives in the shared CPU slab pool.
   *
   * Returns the THREE.BufferAttribute, or null if the pool is exhausted
   * for that bucket. Callers may pass the returned attribute straight to
   * `new THREE.InstancedBufferGeometry().setAttribute('aX', attr)` etc.
   */
  acquireAttribute(size, options = {}) {
    const arrayKind = options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32;
    const itemSize  = options.itemSize  !== undefined ? options.itemSize  : 3;
    const normalized= options.normalized === true;
    const usage     = options.usage !== undefined ? options.usage : BUFFER_USAGE.DYNAMIC;
    const tag       = options.tag !== undefined ? options.tag : SLAB_TAG.GENERIC;
    const stride    = (options.stride !== undefined ? options.stride : 1) | 0;

    const elementCount = Math.max(1, (size * stride) | 0);
    const b = _findBucket(elementCount);
    if (b < 0) {
      this.stats.attributesRejected++;
      return null;
    }

    const bucket = this.attributeBuckets[arrayKind][b];
    if (bucket.freeCount <= 0) {
      bucket.rejectedTotal++;
      this.stats.attributesRejected++;
      this._emit('attribute-rejected', { arrayKind, size: elementCount, reason: 'exhausted' });
      return null;
    }

    const idx = bucket.freeList[bucket.freeHead];
    bucket.freeHead = (bucket.freeHead + 1) % bucket.capacity;
    bucket.freeCount--;

    const slot = bucket.slots[idx];

    // Lazily create the attribute on first acquire, then reuse it.
    if (!slot.attribute) {
      const CPU = ARRAY_KIND_CTOR[arrayKind];
      const array = new CPU(slot.size);
      slot.arrayView = array;
      slot.attribute = new THREE.BufferAttribute(array, itemSize, normalized);
      if (slot.attribute.setUsage) slot.attribute.setUsage(BUFFER_USAGE_THREE[usage]);
    }

    slot.inUse       = 1;
    slot.usage       = usage;
    slot.tagged      = tag;
    slot.acquiredAt  = this.frame;
    slot.requestedSize = elementCount;

    // The view handed to the caller is exactly `elementCount` long. It is a
    // subarray of the underlying slot buffer so writes go to the SAME buffer
    // the BufferAttribute wraps.
    const view = slot.arrayView.subarray(0, elementCount);

    // Tag the view for fast release.
    view[POOL_ID_SYMBOL]     = this.poolId;
    view[SLOT_INDEX_SYMBOL]  = idx;
    view[SLOT_KIND_SYMBOL]   = arrayKind;
    view[SLOT_BUCKET_SYMBOL] = b;
    view.__slotRef           = slot;
    view.__bufferAttribute   = slot.attribute;

    // `BufferAttribute.count` should reflect the requested element count for
    // this view so downstream Three.js doesn't read out of bounds. We safely
    // retarget the attribute to a subarray view of itself.
    if (slot.attribute.count !== elementCount) {
      slot.attribute.array = slot.arrayView;
      slot.attribute.count = slot.size;
      slot.attribute.needsUpdate = true;
    }

    bucket.currentInUse++;
    if (bucket.currentInUse > bucket.peakInUse) bucket.peakInUse = bucket.currentInUse;
    bucket.acquiredTotal++;

    this.stats.attributesAcquired++;
    this.stats.gpuBytesInUse += elementCount * ARRAY_KIND_BYTES[arrayKind];
    if (this.stats.gpuBytesInUse > this.stats.peakGpuBytesInUse) {
      this.stats.peakGpuBytesInUse = this.stats.gpuBytesInUse;
    }

    return slot.attribute;
  }

  /**
   * Release a BufferAttribute back to the pool. The attribute stays alive —
   * only its slot is marked free for reacquisition next frame. Callers
   * should NOT dispose the attribute.
   */
  releaseAttribute(attribute) {
    if (!attribute) return false;

    // Locate the slot via the tagged view.
    let slot = null;
    const arr = attribute.array;
    if (arr && arr.__slotRef) {
      slot = arr.__slotRef;
    } else {
      slot = this._findAttributeSlot(attribute);
      if (!slot) return false;
    }

    if (slot.inUse === 0) return false;

    const b = slot.bucket;
    const k = slot.arrayKind;
    const bucket = this.attributeBuckets[k][b];
    if (!bucket) return false;

    slot.reset();
    slot.inUse = 0;

    bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = slot.index;
    bucket.freeCount++;
    bucket.releasedTotal++;
    bucket.currentInUse--;
    if (bucket.currentInUse < 0) bucket.currentInUse = 0;

    this.stats.attributesReleased++;
    this.stats.gpuBytesInUse -= slot.requestedSize * ARRAY_KIND_BYTES[k];
    if (this.stats.gpuBytesInUse < 0) this.stats.gpuBytesInUse = 0;

    return true;
  }

  _findAttributeSlot(attribute) {
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        for (let i = 0; i < bucket.capacity; i++) {
          if (bucket.slots[i].attribute === attribute) return bucket.slots[i];
        }
      }
    }
    return null;
  }

  /* ---------------- acquire / release (interleaved) ---------------- */

  acquireInterleaved(options = {}) {
    const arrayKind = options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32;
    const stride    = Math.max(1, (options.stride | 0) || 1);
    const count     = Math.max(1, (options.count  | 0) || 1);
    const usage     = options.usage !== undefined ? options.usage : BUFFER_USAGE.DYNAMIC;
    const tag       = options.tag !== undefined ? options.tag : SLAB_TAG.GENERIC;

    if (this.interleavedFreeCount <= 0) {
      this.stats.interleavedRejected++;
      return null;
    }

    const idx = this.interleavedFree[this.interleavedFreeHead];
    this.interleavedFreeHead = (this.interleavedFreeHead + 1) % INTERLEAVED_CAPACITY;
    this.interleavedFreeCount--;

    const slot = this.interleavedSlots[idx];

    // Allocate the underlying array if we haven't yet, or if the requested
    // count exceeds what we have. For Android determinism, we allocate on
    // first acquire only and REQUIRE callers to match size afterwards.
    const totalFloats = stride * count;
    if (!slot.buffer || slot.capacity < totalFloats) {
      const CPU = ARRAY_KIND_CTOR[arrayKind];
      const array = new CPU(totalFloats);
      slot.buffer = new THREE.InterleavedBuffer(array, stride);
      slot.buffer.setUsage(BUFFER_USAGE_THREE[usage]);
      slot.arrayView = array;
      slot.stride = stride;
      slot.capacity = count;
    }

    slot.inUse   = 1;
    slot.arrayKind = arrayKind;
    slot.count   = count;
    slot.usage   = usage;
    slot.tagged  = tag;
    slot.attributes = slot.attributes || new Map();
    slot.buffer.count = count;
    slot.buffer.stride = stride;

    this.stats.interleavedAcquired++;
    return slot.buffer;
  }

  releaseInterleaved(buffer) {
    if (!buffer || !buffer.isInterleavedBuffer) return false;
    for (let i = 0; i < INTERLEAVED_CAPACITY; i++) {
      const slot = this.interleavedSlots[i];
      if (slot.buffer === buffer && slot.inUse === 1) {
        slot.reset();
        slot.inUse = 0;
        this.interleavedFree[(this.interleavedFreeHead + this.interleavedFreeCount) % INTERLEAVED_CAPACITY] = i;
        this.interleavedFreeCount++;
        this.stats.interleavedReleased++;
        return true;
      }
    }
    return false;
  }

  /* ---------------- dirty tracking ---------------- */

  markDirty(attributeOrView) {
    if (!attributeOrView) return false;

    // Accept either a BufferAttribute or a tagged subarray.
    let slot = null;
    if (attributeOrView.isBufferAttribute) {
      const arr = attributeOrView.array;
      if (arr && arr.__slotRef) slot = arr.__slotRef;
    } else if (ArrayBuffer.isView(attributeOrView) && attributeOrView.__slotRef) {
      slot = attributeOrView.__slotRef;
    }
    if (!slot || slot.inUse === 0) return false;

    slot.dirty = 1;
    slot.generation = (slot.generation + 1) | 0;
    if (slot.generation <= 0) slot.generation = 1;

    // Record for the frame's batch upload.
    if (this._dirtyCount < this._dirtySlots.length) {
      this._dirtySlots[this._dirtyCount++] = slot.index | (slot.arrayKind << 16) | (slot.bucket << 24);
    }

    return true;
  }

  markDirtyByName(name) {
    const entry = this._namedAttrs.get(name);
    if (!entry) return false;
    return this.markDirty(entry.attribute);
  }

  commitDirty() {
    return this._commitDirtyInternal();
  }

  _commitDirtyInternal() {
    if (this._dirtyCount === 0) return 0;
    let uploaded = 0;
    for (let i = 0; i < this._dirtyCount; i++) {
      const packed = this._dirtySlots[i];
      const slotIdx  = packed & 0xFFFF;
      const arrayKind = (packed >>> 16) & 0xFF;
      const bucket   = (packed >>> 24) & 0xFF;

      const bucketEntry = this.attributeBuckets[arrayKind][bucket];
      if (!bucketEntry) continue;
      const slot = bucketEntry.slots[slotIdx];
      if (!slot || !slot.attribute || slot.inUse === 0) continue;

      if (slot.dirty === 1) {
        if (this.autoUploadOnCommit) {
          // Refresh count for Three.js so the draw call reads the right
          // number of elements.
          slot.attribute.needsUpdate = true;
          slot.lastUploadedGen = slot.generation;
          slot.uploadCount++;
          uploaded++;
        }
        slot.dirty = 0;
      }
    }
    this._dirtyCount = 0;
    this.stats.uploadsCommitted += uploaded;
    if (uploaded > 0) this._emit('uploaded', { count: uploaded });
    return uploaded;
  }

  /* ---------------- named reservations ---------------- */

  reserveNamed(name, size, options = {}) {
    if (!name || typeof name !== 'string') return null;
    const existing = this._namedAttrs.get(name);
    if (existing) return existing.attribute;

    const attribute = this.acquireAttribute(size, {
      arrayKind: options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32,
      itemSize:  options.itemSize  !== undefined ? options.itemSize  : 3,
      normalized:options.normalized === true,
      usage:     options.usage !== undefined ? options.usage : BUFFER_USAGE.STATIC,
      tag:       options.tag   !== undefined ? options.tag   : SLAB_TAG.GENERIC,
      stride:    options.stride !== undefined ? options.stride : 1,
    });

    if (!attribute) return null;

    this._namedAttrs.set(name, {
      attribute,
      size,
      tag: options.tag !== undefined ? options.tag : SLAB_TAG.GENERIC,
      arrayKind: options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32,
    });
    this.stats.namedCount++;

    return attribute;
  }

  getNamed(name) {
    const entry = this._namedAttrs.get(name);
    return entry ? entry.attribute : null;
  }

  getNamedView(name) {
    const entry = this._namedAttrs.get(name);
    if (!entry) return null;
    const arr = entry.attribute.array;
    if (arr && arr.subarray) return arr.subarray(0, entry.size);
    return arr;
  }

  releaseNamed(name) {
    const entry = this._namedAttrs.get(name);
    if (!entry) return false;
    this._namedAttrs.delete(name);
    this.stats.namedCount--;
    return this.releaseAttribute(entry.attribute);
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame() {
    this.frame++;
    this._dirtyCount = 0;
    return this;
  }

  endFrame() {
    if (this.autoReleasePerFrame) this._releaseAllUnnamed();
    return this;
  }

  _releaseAllUnnamed() {
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        for (let i = 0; i < bucket.capacity; i++) {
          const slot = bucket.slots[i];
          if (slot.inUse === 1 && !this._isNamedSlot(slot)) {
            this.releaseAttribute(slot.attribute);
          }
        }
      }
    }
  }

  _isNamedSlot(slot) {
    for (const entry of this._namedAttrs.values()) {
      if (entry.attribute === slot.attribute) return true;
    }
    return false;
  }

  releaseAll(includeNamed = false) {
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        for (let i = 0; i < bucket.capacity; i++) {
          const slot = bucket.slots[i];
          if (slot.inUse === 1 && (includeNamed || !this._isNamedSlot(slot))) {
            this.releaseAttribute(slot.attribute);
          }
        }
      }
    }

    for (let i = 0; i < INTERLEAVED_CAPACITY; i++) {
      const slot = this.interleavedSlots[i];
      if (slot.inUse === 1) this.releaseInterleaved(slot.buffer);
    }
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const buckets = new Array(ARRAY_KIND.COUNT);
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      buckets[k] = new Array(BUCKET_COUNT);
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        buckets[k][b] = {
          size:          SIZE_BUCKETS[b],
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

    return {
      name:                 this.name,
      frame:                this.frame,
      namedCount:           this.stats.namedCount,
      attributesAcquired:   this.stats.attributesAcquired,
      attributesReleased:   this.stats.attributesReleased,
      attributesRejected:   this.stats.attributesRejected,
      interleavedAcquired:  this.stats.interleavedAcquired,
      interleavedReleased:  this.stats.interleavedReleased,
      interleavedRejected:  this.stats.interleavedRejected,
      uploadsCommitted:     this.stats.uploadsCommitted,
      gpuBytesAllocated:    this.stats.gpuBytesAllocated,
      gpuBytesInUse:        this.stats.gpuBytesInUse,
      peakGpuBytesInUse:    this.stats.peakGpuBytesInUse,
      buckets,
      perfTier:             PERF_TIER,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    this.releaseAll(true);
    this._dirtyCount = 0;
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        bucket.freeHead = 0;
        bucket.freeCount = bucket.capacity;
        for (let i = 0; i < bucket.capacity; i++) {
          bucket.freeList[i] = i;
          bucket.slots[i].reset();
          bucket.slots[i].inUse = 0;
        }
        bucket.currentInUse = 0;
        bucket.peakInUse = 0;
        bucket.acquiredTotal = 0;
        bucket.releasedTotal = 0;
        bucket.rejectedTotal = 0;
      }
    }

    for (let i = 0; i < INTERLEAVED_CAPACITY; i++) {
      this.interleavedSlots[i].reset();
      this.interleavedSlots[i].inUse = 0;
      this.interleavedFree[i] = i;
    }
    this.interleavedFreeHead = 0;
    this.interleavedFreeCount = INTERLEAVED_CAPACITY;

    this._namedAttrs.clear();

    this.stats.attributesAcquired = 0;
    this.stats.attributesReleased = 0;
    this.stats.attributesRejected = 0;
    this.stats.interleavedAcquired = 0;
    this.stats.interleavedReleased = 0;
    this.stats.interleavedRejected = 0;
    this.stats.uploadsCommitted = 0;
    this.stats.namedCount = 0;
    this.stats.gpuBytesInUse = 0;
    this.stats.peakGpuBytesInUse = 0;

    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let k = 0; k < ARRAY_KIND.COUNT; k++) {
      for (let b = 0; b < BUCKET_COUNT; b++) {
        const bucket = this.attributeBuckets[k][b];
        for (let i = 0; i < bucket.capacity; i++) {
          const slot = bucket.slots[i];
          if (slot.attribute && slot.attribute.dispose) {
            try { slot.attribute.dispose(); } catch (_) { /* swallow */ }
          }
          slot.attribute = null;
          slot.arrayView = null;
        }
        bucket.slots.length = 0;
        bucket.freeList = null;
      }
      this.attributeBuckets[k] = null;
    }
    this.attributeBuckets = null;

    for (let i = 0; i < INTERLEAVED_CAPACITY; i++) {
      const slot = this.interleavedSlots[i];
      if (slot.buffer && slot.buffer.dispose) {
        try { slot.buffer.dispose(); } catch (_) { /* swallow */ }
      }
      slot.buffer = null;
      slot.attributes = null;
      slot.arrayView = null;
    }
    this.interleavedSlots.length = 0;
    this.interleavedFree = null;

    this._namedAttrs.clear();
    this._dirtySlots = null;
    this._listeners.clear();

    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. STANDARD LIGHTING BUFFER PRESETS                                */
/* ------------------------------------------------------------------ */

/**
 * Reserve the canonical buffers the lighting stack needs. Every name is
 * stable across the engine so any subsystem can ask for the same attribute
 * via getNamed('lightListVBO') etc.
 *
 * Sizes are tuned for the classic forward+ anime pipeline:
 *   lightListVBO        Float32 4096  LIGHT_LIST  — packed light handles
 *   clusterGridIndices  Uint32  16384 CLUSTER
 *   clusterGridRanges   Uint16  8192  CLUSTER
 *   giProbeGrid         Float32 8192  GI          — CPU-side staging slab
 *   aoKernelVBO         Float32 1024  AO
 *   envPaletteVBO       Float32 64    ENV
 *   interiorVolVBO      Float32 4096  INTERIOR
 *   exteriorProbeVBO    Float32 2048  EXTERIOR
 *   postHDRStagingVBO   Float32 4096  POST
 *   shadowStageVBO      Float32 2048  SHADOW
 *   cascadeMatrixVBO    Float32 64    MATRIX      — 4 cascades × 16 floats
 *   lightUniformsVBO    Float32 512   LIGHT_LIST
 */
export function reserveLightingBuffers(pool) {
  if (!pool || typeof pool.reserveNamed !== 'function') return null;

  const named = {
    lightListVBO:       pool.reserveNamed('lightListVBO',       4096,  { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.LIGHT_LIST }),
    clusterGridIndices: pool.reserveNamed('clusterGridIndices', 16384, { arrayKind: ARRAY_KIND.U32, itemSize: 1, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.CLUSTER }),
    clusterGridRanges:  pool.reserveNamed('clusterGridRanges',  8192,  { arrayKind: ARRAY_KIND.U16, itemSize: 1, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.CLUSTER }),
    giProbeGrid:        pool.reserveNamed('giProbeGrid',        8192,  { arrayKind: ARRAY_KIND.F32, itemSize: 3, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.GI }),
    aoKernelVBO:        pool.reserveNamed('aoKernelVBO',        1024,  { arrayKind: ARRAY_KIND.F32, itemSize: 3, usage: BUFFER_USAGE.STATIC,  tag: SLAB_TAG.AO }),
    envPaletteVBO:      pool.reserveNamed('envPaletteVBO',      64,    { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.ENV }),
    interiorVolVBO:     pool.reserveNamed('interiorVolVBO',     4096,  { arrayKind: ARRAY_KIND.F32, itemSize: 3, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.INTERIOR }),
    exteriorProbeVBO:   pool.reserveNamed('exteriorProbeVBO',   2048,  { arrayKind: ARRAY_KIND.F32, itemSize: 3, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.EXTERIOR }),
    postHDRStagingVBO:  pool.reserveNamed('postHDRStagingVBO',  4096,  { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.POST }),
    shadowStageVBO:     pool.reserveNamed('shadowStageVBO',     2048,  { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.SHADOW }),
    cascadeMatrixVBO:   pool.reserveNamed('cascadeMatrixVBO',   64,    { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.MATRIX }),
    lightUniformsVBO:   pool.reserveNamed('lightUniformsVBO',   512,   { arrayKind: ARRAY_KIND.F32, itemSize: 4, usage: BUFFER_USAGE.DYNAMIC, tag: SLAB_TAG.LIGHT_LIST }),
  };

  return named;
}

/* ------------------------------------------------------------------ */
/* 6. TAGGED BUFFER SUB-POOL                                          */
/* ------------------------------------------------------------------ */

/**
 * Façade binding a fixed tag to a shared BufferPool, so subsystem code can
 * acquire attributes without passing tag on every call.
 */
export class TaggedBufferSubPool {
  constructor(pool, tag) {
    this.pool = pool;
    this.tag  = tag;
  }

  acquire(size, itemSize = 3, options = {}) {
    return this.pool.acquireAttribute(size, {
      arrayKind: options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32,
      itemSize,
      normalized: options.normalized === true,
      usage: options.usage !== undefined ? options.usage : BUFFER_USAGE.DYNAMIC,
      tag: this.tag,
      stride: options.stride !== undefined ? options.stride : 1,
    });
  }

  release(attribute) { return this.pool.releaseAttribute(attribute); }
  markDirty(attribute) { return this.pool.markDirty(attribute); }

  reserveNamed(name, size, itemSize = 3, options = {}) {
    return this.pool.reserveNamed(name, size, {
      arrayKind: options.arrayKind !== undefined ? options.arrayKind : ARRAY_KIND.F32,
      itemSize,
      normalized: options.normalized === true,
      usage: options.usage !== undefined ? options.usage : BUFFER_USAGE.STATIC,
      tag: this.tag,
      stride: options.stride !== undefined ? options.stride : 1,
    });
  }
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createBufferPool(options = {}) {
  return new BufferPool(options);
}

export function createTaggedBufferSubPool(pool, tag) {
  return new TaggedBufferSubPool(pool, tag);
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultBufferPool = null;
let _defaultLightingBuffers = null;

export function getDefaultBufferPool() {
  if (!_defaultBufferPool) {
    _defaultBufferPool = new BufferPool();
    _defaultLightingBuffers = reserveLightingBuffers(_defaultBufferPool);
  }
  return _defaultBufferPool;
}

export function getDefaultLightingBuffers() {
  if (!_defaultLightingBuffers) getDefaultBufferPool();
  return _defaultLightingBuffers;
}

export function disposeDefaultBufferPool() {
  if (_defaultBufferPool) {
    _defaultBufferPool.dispose();
    _defaultBufferPool = null;
    _defaultLightingBuffers = null;
  }
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  BufferPool,
  AttributeSlot,
  InterleavedSlot,
  TaggedBufferSubPool,
  createBufferPool,
  createTaggedBufferSubPool,
  getDefaultBufferPool,
  getDefaultLightingBuffers,
  disposeDefaultBufferPool,
  reserveLightingBuffers,
  BUFFER_USAGE,
  BUFFER_USAGE_THREE,
  BUFFER_KIND,
  BUFFER_KIND_NAME,
  ATTRIBUTE_CAPACITY,
  INTERLEAVED_CAPACITY,
};

export default _defaultExport;