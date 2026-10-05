API Documentation — src/core/012_rnd_BufferPool.js

File Purpose

This file is the GPU buffer pool for the anime lighting stack on Android mobile. Where 011_rnd_TypedArrayPool.js owns the CPU-side typed arrays, this module owns the GPU-side descriptors. It wraps a typed slab from the typed-array pool in a THREE.BufferAttribute, tracks the attribute's update generation so the renderer only re-uploads when data actually changed, and recycles attributes across frames without ever calling new Float32BufferAttribute(...) inside a hot loop.

The workloads it serves:

· Light list VBO — per-frame upload of the active light list to the GPU
· Cluster grid buffer — index and range buffers for clustered forward+
· GI probe uploads — irradiance slabs written each GI update tick
· AO kernel uploads — blur kernels and depth-resolve coefficients
· Environment palette uploads — palette LUT slabs
· Interior volume uploads — per-room probe lattices
· Exterior probe uploads — SH-9 probe slabs
· Shadow atlas staging buffers — intermediate buffers for the shadow packer

The module also owns interleaved buffers via THREE.InterleavedBuffer and THREE.InterleavedBufferAttribute, so multi-field vertex data (e.g. position + color + AO weight) can be packed into one buffer object.

Two-tier design:

· AttributePool — raw THREE.BufferAttribute instances keyed by array kind, size bucket, item size, normalized flag, and usage hint.
· InterleavedPool — THREE.InterleavedBuffer plus attribute views, keyed by array kind, stride, count, and usage hint.

Three critical Android-specific features:

1. Upload coalescing — markDirty(attr) bumps a generation counter; the frame's commitDirty() walks the dirty set once and sets needsUpdate = true on each attribute. N subsystems writing to N attributes cause exactly one upload pass, not N.
2. Correct usage hints — DynamicDrawUsage for per-frame buffers, StaticDrawUsage for named persistent buffers, StreamDrawUsage for worker-transferred buffers. This matters on mobile because a mis-set usage hint costs 2–4 ms per upload on Mali and PowerVR.
3. Named reservations — long-lived buffers for the canonical lighting VBOs (lightListVBO, clusterGridIndices, giProbeGrid, etc.) so every subsystem asks for the same attribute by name and gets the same GPU object back across frames.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

BUFFER_USAGE

Type: frozen enum

Values:

· STATIC = 0
· DYNAMIC = 1
· STREAM = 2

BUFFER_USAGE_THREE

Type: frozen array

Maps the enum to the corresponding Three.js constants: [StaticDrawUsage, DynamicDrawUsage, StreamDrawUsage].

BUFFER_KIND

Type: frozen enum

Values:

· ATTRIBUTE = 0
· INTERLEAVED = 1
· COUNT = 2

BUFFER_KIND_NAME

Type: frozen array

Values: ['attribute', 'interleaved'].

ATTRIBUTE_CAPACITY

Type: frozen array of 13 numbers

Per-bucket capacity (number of attribute slots) as a function of PERF_TIER.

On HIGH: [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2].

On MEDIUM: [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1].

On LOW: [64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1, 1].

The bucket sizes are the same as SIZE_BUCKETS in 011 — 16, 32, 64, …, 65536. The capacities are tuned so small attributes are abundant and large attributes are scarce.

INTERLEAVED_CAPACITY

Type: number

Value: 128 on HIGH, 64 on MEDIUM, 32 on LOW.

The number of interleaved buffer slots. Smaller than the attribute capacity because interleaved buffers are used less frequently and each one is typically larger.

POOL_ID_SYMBOL, SLOT_INDEX_SYMBOL, SLOT_KIND_SYMBOL, SLOT_BUCKET_SYMBOL

Internal string constants used to tag the array views and attributes with their owning pool, slot, kind, and bucket. These tags let releaseAttribute() find the parent slot in O(1).

---

Module-Level State (Not Exported Directly)

_bufferPoolIdCounter

Type: number

Monotonic counter for pool ids.

_defaultBufferPool

Type: BufferPool | null

The module-level singleton.

_defaultLightingBuffers

Type: object | null

The named reservations for the standard lighting buffers.

---

Exported Class — AttributeSlot

One slot holds a pre-built THREE.BufferAttribute whose underlying typed array is a slab acquired from the shared TypedArrayPool (from 011). The attribute is created ONCE and reused across frames.

Constructor

```
new AttributeSlot(index, bucket, arrayKind)
```

Parameters:

· index — the slot's index within its bucket.
· bucket — the bucket index.
· arrayKind — one of ARRAY_KIND.

Instance Properties

· index — the slot's index within its bucket.
· bucket — the bucket index.
· arrayKind — the array kind.
· size — the element capacity (SIZE_BUCKETS[bucket]).
· usage — one of BUFFER_USAGE.
· attribute — the THREE.BufferAttribute, or null before first acquire.
· arrayView — the typed array that the attribute wraps.
· inUse — 1 if acquired, 0 if free.
· tagged — the slab tag.
· acquiredAt — the frame number when acquired.
· requestedSize — the element count requested on the last acquire.
· dirty — 1 if the slot's data needs to be re-uploaded.
· generation — a monotonic counter incremented on each markDirty.
· lastUploadedGen — the generation of the last actual upload.
· uploadCount — how many times this slot's attribute has been uploaded.

Instance Methods

view(size)

Parameters: size — the requested element count.

Returns: a subarray of the underlying typed array.

reset()

Returns: nothing. Zeroes the requested region and clears the dirty flag.

---

Exported Class — InterleavedSlot

One slot holds a THREE.InterleavedBuffer and a map of attribute views on that buffer.

Constructor

```
new InterleavedSlot(index)
```

Instance Properties

· index — the slot's index within the interleaved pool.
· inUse — 1 if acquired, 0 if free.
· arrayKind — one of ARRAY_KIND.
· stride — the number of elements between consecutive vertices.
· count — the number of vertices.
· capacity — the maximum number of vertices this slot can hold.
· usage — one of BUFFER_USAGE.
· buffer — the THREE.InterleavedBuffer, or null before first acquire.
· attributes — a Map from attribute name to THREE.InterleavedBufferAttribute.
· arrayView — the underlying typed array.
· tagged — the slab tag.
· dirty — 1 if the buffer needs re-upload.
· uploadCount — how many times the buffer has been uploaded.

Instance Methods

reset()

Returns: nothing. Clears the dirty flag.

---

Exported Class — BufferPool

The main pool.

Constructor

```
new BufferPool(options = {})
```

Parameters:

· name — the pool's diagnostic name. Default buf_pool_<id>.
· cpuPool — the TypedArrayPool to source slabs from. Default the singleton from 011.
· autoReleasePerFrame — if true, endFrame() releases every in-use attribute. Default false.
· autoUploadOnCommit — if true (default), commitDirty() sets needsUpdate on each dirty attribute.

Constructor work:

1. Stores the CPU pool reference.
2. Allocates attributeBuckets — an 8 by 13 grid of bucket objects.
3. Each bucket has slots, freeList, freeHead, freeCount, capacity, and per-bucket stats.
4. Allocates interleavedSlots — a INTERLEAVED_CAPACITY-length array of InterleavedSlot instances plus a free list.
5. Allocates _dirtySlots — an Int32Array(512) recording packed slot coordinates for the frame's batch upload.
6. Allocates _namedAttrs — a Map from name to { attribute, size, tag, arrayKind }.
7. Initializes the stats object.
8. Allocates _listeners (Map).
9. Pre-binds _commitDirtyInternal to this._commitDirtyFn.

Instance Properties

· name — the pool name.
· poolId — the unique pool id.
· cpuPool — the CPU typed-array pool.
· autoReleasePerFrame — the frame-release flag.
· autoUploadOnCommit — the auto-upload flag.
· attributeBuckets — the 8-by-13 attribute slot grid.
· interleavedSlots — the interleaved slot array.
· interleavedFree — the free list for interleaved slots.
· frame — the frame counter.
· stats — the aggregate stats object.

The stats object has: attributesAcquired, attributesReleased, attributesRejected, interleavedAcquired, interleavedReleased, interleavedRejected, uploadsCommitted, namedCount, gpuBytesAllocated, gpuBytesInUse, peakGpuBytesInUse.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'attribute-rejected', 'uploaded'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

acquireAttribute(size, options = {})

Parameters:

· size — the requested element count (not vertex count).
· options.arrayKind — one of ARRAY_KIND. Default F32.
· options.itemSize — elements per vertex. Default 3.
· options.normalized — whether to normalize integer arrays. Default false.
· options.usage — one of BUFFER_USAGE. Default DYNAMIC.
· options.tag — one of SLAB_TAG. Default GENERIC.
· options.stride — element stride between vertices. Default 1.

Returns: a THREE.BufferAttribute, or null if the pool is exhausted.

Flow:

1. Computes elementCount = size * stride.
2. Rounds up to the next SIZE_BUCKETS entry.
3. Pops a slot from the bucket's free list. If empty, rejects with reason exhausted.
4. Lazily creates the attribute on first acquire. The underlying array is a fresh typed array of slot.size elements, wrapped in a THREE.BufferAttribute, with usage applied.
5. Tags the subarray view with the pool id, slot index, kind, and bucket, plus __slotRef and __bufferAttribute.
6. Updates slot.attribute.array and slot.attribute.count so Three.js reads the correct range.
7. Increments in-use counters, updates GPU byte accounting, and returns the attribute.

Important: the attribute is the SAME object across acquires. This is the key to zero-allocation. Callers should not hold references across frames without checking inUse.

releaseAttribute(attribute)

Parameters: attribute — a THREE.BufferAttribute returned by acquireAttribute.

Returns: boolean.

Purpose: locates the parent slot via the tags on attribute.array, resets the slot, and returns it to the free list.

acquireInterleaved(options = {})

Parameters:

· options.arrayKind — one of ARRAY_KIND. Default F32.
· options.stride — elements per vertex. Default 1.
· options.count — number of vertices. Default 1.
· options.usage — one of BUFFER_USAGE. Default DYNAMIC.
· options.tag — one of SLAB_TAG. Default GENERIC.

Returns: a THREE.InterleavedBuffer, or null if the pool is exhausted.

Flow:

1. Pops a slot from the interleaved free list.
2. If the slot's existing buffer is smaller than needed, allocates a fresh typed array and a fresh InterleavedBuffer.
3. Updates count and stride on the buffer.
4. Returns the buffer.

Callers then create THREE.InterleavedBufferAttribute objects as needed and register them on the slot's attributes map.

releaseInterleaved(buffer)

Parameters: buffer — a THREE.InterleavedBuffer.

Returns: boolean.

markDirty(attributeOrView)

Parameters: attributeOrView — either a BufferAttribute or a tagged subarray view.

Returns: boolean.

Purpose: marks the slot's data as needing re-upload. Increments the slot's generation counter, sets the dirty flag, and records the slot's packed coordinates in _dirtySlots for the frame's batch upload.

This is the CRITICAL method for upload coalescing. N subsystems each call markDirty once; a single commitDirty call performs N uploads.

markDirtyByName(name)

Parameters: name — the named reservation.

Returns: boolean.

Purpose: convenience wrapper for markDirty on a named attribute.

commitDirty()

Returns: the number of attributes actually uploaded.

Purpose: walks _dirtySlots. For each dirty slot, if autoUploadOnCommit is on, sets attribute.needsUpdate = true and records the generation. Emits uploaded with the count.

reserveNamed(name, size, options = {})

Parameters:

· name — a unique name.
· size — the element count.
· options.arrayKind — one of ARRAY_KIND.
· options.itemSize — elements per vertex.
· options.normalized — boolean.
· options.usage — one of BUFFER_USAGE. Default STATIC.
· options.tag — one of SLAB_TAG.
· options.stride — element stride.

Returns: the reserved THREE.BufferAttribute, or null.

Purpose: creates a long-lived attribute that is skipped by releaseAll() unless includeNamed=true. Records the entry in _namedAttrs.

getNamed(name)

Parameters: name — the reservation name.

Returns: the attribute, or null.

getNamedView(name)

Parameters: name — the reservation name.

Returns: a subarray view of the named attribute's array, or null.

releaseNamed(name)

Parameters: name — the reservation name.

Returns: boolean.

beginFrame()

Returns: this. Increments frame and clears the dirty set.

endFrame()

Returns: this. If autoReleasePerFrame is on, releases every unnamed attribute.

_releaseAllUnnamed()

Internal. Iterates every slot and releases unnamed attributes.

_isNamedSlot(slot)

Internal. Returns true if the slot's attribute is currently a named reservation.

releaseAll(includeNamed = false)

Parameters: includeNamed — if true, releases named attributes too.

Returns: nothing.

getStats()

Returns: an object with name, frame, namedCount, the nine aggregate counters, a two-dimensional buckets array, and perfTier.

reset()

Returns: this. Releases everything and zeroes every counter.

dispose()

Returns: this. Resets and disposes every BufferAttribute and InterleavedBuffer it created.

---

Exported Function — reserveLightingBuffers(pool)

Parameters: pool — a BufferPool instance.

Returns: an object mapping each reservation name to its acquired BufferAttribute.

Purpose: reserves the canonical lighting VBOs:

· lightListVBO — Float32 4096 elements, item size 4, DYNAMIC, tag LIGHT_LIST.
· clusterGridIndices — Uint32 16384 elements, item size 1, DYNAMIC, tag CLUSTER.
· clusterGridRanges — Uint16 8192 elements, item size 1, DYNAMIC, tag CLUSTER.
· giProbeGrid — Float32 8192 elements, item size 3, DYNAMIC, tag GI.
· aoKernelVBO — Float32 1024 elements, item size 3, STATIC, tag AO.
· envPaletteVBO — Float32 64 elements, item size 4, DYNAMIC, tag ENV.
· interiorVolVBO — Float32 4096 elements, item size 3, DYNAMIC, tag INTERIOR.
· exteriorProbeVBO — Float32 2048 elements, item size 3, DYNAMIC, tag EXTERIOR.
· postHDRStagingVBO — Float32 4096 elements, item size 4, DYNAMIC, tag POST.
· shadowStageVBO — Float32 2048 elements, item size 4, DYNAMIC, tag SHADOW.
· cascadeMatrixVBO — Float32 64 elements, item size 4, DYNAMIC, tag MATRIX (four cascades times sixteen floats).
· lightUniformsVBO — Float32 512 elements, item size 4, DYNAMIC, tag LIGHT_LIST.

Every buffer is created once and reused across the engine's lifetime.

---

Exported Class — TaggedBufferSubPool

A façade binding a fixed tag to a shared BufferPool so subsystem code can acquire and release without passing the tag on every call.

Constructor

```
new TaggedBufferSubPool(pool, tag)
```

Instance Methods

· acquire(size, itemSize = 3, options = {}) — delegates to pool.acquireAttribute with tag pre-filled.
· release(attribute) — delegates.
· markDirty(attribute) — delegates.
· reserveNamed(name, size, itemSize = 3, options = {}) — delegates.

---

Exported Functions

getDefaultBufferPool()

Returns: the module-level singleton BufferPool, creating it on first call and immediately calling reserveLightingBuffers().

getDefaultLightingBuffers()

Returns: the named-reservation object from the default pool.

disposeDefaultBufferPool()

Returns: nothing.

createBufferPool(options = {})

Returns: a new BufferPool.

createTaggedBufferSubPool(pool, tag)

Returns: a new TaggedBufferSubPool.

_findBucket(size)

Internal. Linear scan over the size buckets.

_nextPoolId()

Internal. Returns the next monotonic pool id.

---

Default Export

The default export bundles: BufferPool, AttributeSlot, InterleavedSlot, TaggedBufferSubPool, createBufferPool, createTaggedBufferSubPool, getDefaultBufferPool, getDefaultLightingBuffers, disposeDefaultBufferPool, reserveLightingBuffers, BUFFER_USAGE, BUFFER_USAGE_THREE, BUFFER_KIND, BUFFER_KIND_NAME, ATTRIBUTE_CAPACITY, INTERLEAVED_CAPACITY.

---

Usage Pattern

A subsystem that wants a tagged VBO sub-pool:

```
import {
  getDefaultBufferPool,
  createTaggedBufferSubPool,
  BUFFER_USAGE,
  SLAB_TAG,
} from './src/core/012_rnd_BufferPool.js';

const pool = getDefaultBufferPool();
const lightPool = createTaggedBufferSubPool(pool, SLAB_TAG.LIGHT_LIST);

// Per frame:
const attr = lightPool.acquire(1024, 4, { usage: BUFFER_USAGE.DYNAMIC });
const arr = attr.array;
// ... write light data into arr ...
lightPool.markDirty(attr);
// At the end of the frame, the EngineLoop calls:
pool.commitDirty();
```

A subsystem that wants a persistent VBO:

```
const lightListAttr = pool.reserveNamed('lightListVBO', 4096, {
  arrayKind: ARRAY_KIND.F32,
  itemSize: 4,
  usage: BUFFER_USAGE.DYNAMIC,
  tag: SLAB_TAG.LIGHT_LIST,
});

// Each frame, write into the array and mark dirty:
const arr = lightListAttr.array;
for (let i = 0; i < lightCount * 4; i++) arr[i] = ...;
pool.markDirty(lightListAttr);
```

The pool guarantees that after the first frame no new GPU buffer objects are created. Every acquire and release is O(1). commitDirty performs a single coalesced upload pass per frame regardless of how many subsystems marked dirty. This is essential on Android where each bufferSubData call has 0.5–2 ms of overhead on Mali and PowerVR.
