API Documentation — src/core/011_rnd_TypedArrayPool.js

File Purpose

This file is the dedicated multi-kind typed-array pool for the anime lighting stack. Where 010_rnd_ObjectPool.js contains a generic TypedArrayPool for small scratch arrays, this module is the full-featured allocator used by every lighting subsystem that needs large, GPU-bound, or long-lived typed buffers.

Its workloads are:

· Shadow atlas staging — depth-packed Float32 or Uint16 slabs
· Cascade matrices — 16-float Matrix4 slabs times N cascades
· GI probe grid — 3-channel Float32 irradiance slabs
· AO blur kernels — half-res Float32 or Unorm8 slabs
· Cluster grid — index Uint32 plus range Uint16 slabs
· Light list — packed Uint32 or Uint16 handles
· Environment palette — Float32 RGBA linear blocks
· Interior volumes — Float32 probe lattices
· Exterior probes — Float32 SH-9 per-probe slabs
· Post buffers — HDR Float32 half-res slabs
· Worker transfer pools — transferable ArrayBuffer wrappers

The design differs from the generic typed-array pool in 010 in five important ways:

1. Eight typed-array kinds supported in one pool (Float32Array, Float64Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array).
2. Larger size buckets, up to 65536 elements. The generic pool in 010 tops out at 4096.
3. Sub-slab allocation with full buffer sharing. A single 8K Float32Array can serve multiple smaller views without extra buffer objects.
4. GPU-compatible stride tracking. Every slab records its byteStride, elementCount, and alignment.
5. Worker-safe transfer support. A slab's underlying ArrayBuffer can be detached for postMessage handoff and reattached after receipt, both in O(1) with zero copy.

The pool also provides per-tag accounting so debug tools can visualize which subsystem owns each buffer, and a named-reservation system for long-lived persistent buffers (shadow atlas scratch, GI probe grid, cluster grids).

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

ARRAY_KIND

Type: frozen enum

Values:

· F32 = 0
· F64 = 1
· I32 = 2
· U32 = 3
· I16 = 4
· U16 = 5
· I8 = 6
· U8 = 7
· COUNT = 8

ARRAY_KIND_NAME

Type: frozen array

Values: ['float32', 'float64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8'].

ARRAY_KIND_CTOR

Type: frozen array

Values: [Float32Array, Float64Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array].

ARRAY_KIND_BYTES

Type: frozen array

Values: [4, 8, 4, 4, 2, 2, 1, 1]. Bytes per element per kind.

SLAB_TAG

Type: frozen enum

Per-slab purpose tags. Used for debug visualization and per-tag stats.

Values:

· GENERIC = 0
· SHADOW = 1
· GI = 2
· AO = 3
· CLUSTER = 4
· LIGHT_LIST = 5
· ENV = 6
· INTERIOR = 7
· EXTERIOR = 8
· POST = 9
· WORKER = 10
· MATRIX = 11
· COUNT = 12

SLAB_TAG_NAME

Type: frozen array

Values: ['generic', 'shadow', 'gi', 'ao', 'cluster', 'light_list', 'env', 'interior', 'exterior', 'post', 'worker', 'matrix'].

SIZE_BUCKETS

Type: frozen array

Values: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536].

Thirteen bucket sizes. These are chosen so that a request for any size between 16 and 65536 rounds up to a bucket that is at most 2 times the request. This keeps internal fragmentation below 50 % for every request.

BUCKET_COUNT

Type: number

Value: 13 — the length of SIZE_BUCKETS.

BUCKET_CAPACITY

Type: frozen array of 13 numbers

Per-bucket capacity (number of slabs) as a function of PERF_TIER.

On HIGH: [512, 384, 256, 192, 160, 128, 96, 64, 48, 32, 16, 8, 4].

On MEDIUM: [256, 192, 128, 96, 80, 64, 48, 32, 24, 16, 8, 4, 2].

On LOW: [128, 96, 64, 48, 40, 32, 24, 16, 12, 8, 4, 2, 1].

These numbers are tuned so that small slabs are abundant and large slabs are scarce, matching real lighting workloads where most allocations are for 16–512 element scratch.

---

Module-Level State (Not Exported Directly)

_typedPoolIdCounter

Type: number

Monotonic counter for pool ids.

_defaultTypedPool

Type: TypedArrayPool | null

The module-level singleton.

_defaultLightingSlabs

Type: object | null

The named reservations for the standard lighting slabs.

POOL_ID_SYMBOL, SLAB_INDEX_SYMBOL, SLAB_BUCKET_SYMBOL, SLAB_KIND_SYMBOL, SLAB_TAG_SYMBOL

Internal string constants used as non-enumerable tags on each slab's subarray views. These tags let release() find the parent slab in O(1) without a scan.

---

Exported Class — Slab

A single underlying typed array in the pool.

Constructor

```
new Slab(index, bucket, kind, tag, array)
```

Parameters:

· index — the slab's index within its bucket.
· bucket — the bucket index.
· kind — one of ARRAY_KIND.
· tag — one of SLAB_TAG.
· array — the actual typed array.

Instance Properties

· index — the slab's index within its bucket.
· bucket — the bucket index.
· kind — the array kind.
· tag — the current tag (may change between acquires).
· array — the underlying typed array.
· capacity — the array's length.
· inUse — 1 if acquired, 0 if free.
· acquiredAt — the frame number when acquired.
· requestedSize — the size requested on the last acquire.

Instance Methods

view(size)

Parameters: size — the requested length.

Returns: a subarray view of exactly size elements.

reset()

Returns: nothing.

Purpose: zeroes only the region that was requested. Cost proportional to the request size, not the full capacity. This matters when a caller requests 4 elements from a 4096-element slab.

---

Exported Class — Bucket

One bucket per size. Internal, not exposed to consumers.

Constructor

```
new Bucket(bucketIdx, kind)
```

Instance Properties

· bucketIdx — the bucket index.
· kind — the array kind.
· size — the element count of every slab in this bucket.
· capacity — the number of slabs.
· Ctor — the typed array constructor.
· slabs — the array of Slab instances.
· freeList — an Int32Array(capacity) of free slab indices.
· freeHead — the ring read pointer.
· freeCount — the number of free slabs.
· currentInUse, peakInUse, acquiredTotal, releasedTotal, rejectedTotal, zeroTotal — per-bucket counters.

Instance Methods

allocate()

Returns: nothing. Pre-allocates every slab.

dispose()

Returns: nothing. Nulls every slab.

---

Exported Class — TypedArrayPool

The main pool.

Constructor

```
new TypedArrayPool(options = {})
```

Parameters:

· name — the pool's diagnostic name. Default typed_pool_<id>.
· autoZero — if true (default), cleared slabs are zeroed on release. If false, faster release but callers must not assume zeroed data.
· autoReleasePerFrame — if true, endFrame() releases every in-use slab. Default false.

Constructor work:

1. Allocates buckets — an 8 by 13 grid of Bucket instances.
2. Allocates and pre-allocates every bucket.
3. Initializes the stats object.
4. Allocates _tagInUse, _tagPeak, _tagTotal — three Uint32Array(SLAB_TAG.COUNT) arrays.
5. Allocates _detachedSlabs — a fixed array of up to 64 detached slabs.
6. Allocates _namedSlabs (a Map) for long-lived reservations.
7. Allocates _listeners (Map).

Instance Properties

· name — the pool name.
· poolId — the unique pool id.
· autoZero — the zeroing flag.
· autoReleasePerFrame — the frame-release flag.
· frame — the frame counter.
· buckets — the 8-by-13 grid.
· stats — the aggregate stats.

The stats object has: totalAcquired, totalReleased, totalRejected, totalDetached, totalReattached, peakInUse, currentInUse.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'rejected', 'detach', 'reattach'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

acquire(size, kind = ARRAY_KIND.F32, tag = SLAB_TAG.GENERIC)

Parameters:

· size — the number of elements requested.
· kind — one of ARRAY_KIND. Default F32.
· tag — one of SLAB_TAG. Default GENERIC.

Returns: a subarray view of exactly size elements, or null if the corresponding bucket is exhausted.

Flow:

1. Rounds size up to the next SIZE_BUCKETS entry.
2. If no bucket can serve the request, rejects with reason: 'too_large'.
3. Pops a slab from the bucket's free list. If the bucket is empty, rejects with reason: 'exhausted'.
4. Marks the slab in-use and records the tag.
5. Updates the aggregate and per-tag counters.
6. Creates a subarray view of the exact requested size, tags it with the pool id, slab index, bucket, kind, tag, and the direct __slabRef reference to the parent slab.
7. Returns the view.

release(view)

Parameters: view — a view returned by acquire.

Returns: boolean.

Purpose: verifies the pool id on the view, extracts the parent slab from __slabRef, zeroes it if autoZero is on, returns the slab to its bucket's free list, updates stats and per-tag in-use counters.

releaseAll()

Returns: the number of slabs released.

Purpose: iterates every bucket and releases every in-use slab.

beginFrame()

Returns: nothing. Increments frame and clears the detached-slab list.

endFrame()

Returns: the number of slabs still in use. If autoReleasePerFrame is on, releases all.

reserveNamed(name, size, kind = ARRAY_KIND.F32, tag = SLAB_TAG.GENERIC)

Parameters:

· name — a unique name.
· size — the size.
· kind — the array kind.
· tag — the tag.

Returns: the reserved view, or null if the pool cannot serve it.

Purpose: creates a long-lived slab that is skipped by releaseAll(). Use this for subsystem-owned persistent buffers.

The reserved view is tagged with __namedSlab = name so the pool knows to skip it during bulk release.

getNamed(name)

Parameters: name — the reservation name.

Returns: the reserved view, or null.

releaseNamed(name)

Parameters: name — the reservation name.

Returns: boolean.

detach(view)

Parameters: view — a view returned by acquire or reserveNamed.

Returns: the underlying ArrayBuffer or null.

Purpose: removes the underlying buffer from the pool for postMessage transfer. The slab remains logically in-use so it cannot be re-acquired while detached. Records the slab in _detachedSlabs so reattach() can restore it.

reattach(slab, buffer)

Parameters:

· slab — the Slab instance from detach().
· buffer — the transferred ArrayBuffer.

Returns: boolean.

Purpose: reconstructs the typed array from the transferred buffer and reattaches it to the slab. After reattach, the slab can be released normally.

getStats()

Returns: an object with name, kind, the six aggregate counters, namedCount, detachedCount, a two-dimensional buckets array, a tags array, and perfTier.

estimateBytes()

Returns: the total number of bytes occupied by every slab, pre-computed by summing capacity * size * elementBytes across every bucket.

reset()

Returns: this. Releases everything, clears the named map, resets all counters.

dispose()

Returns: this. Resets and nulls every internal array.

---

Exported Function — reserveLightingSlabs(pool)

Parameters: pool — a TypedArrayPool instance.

Returns: an object mapping each reservation name to its acquired view.

Purpose: reserves the canonical long-lived slabs the lighting stack uses:

· shadowAtlasScratch — Float32 8192, tag SHADOW.
· cascadeMatrixSlab — Float32 64, tag MATRIX (4 cascades times 16 floats).
· giProbeGrid — Float32 65536, tag GI.
· aoBlurKernel — Float32 4096, tag AO.
· clusterGridIndices — Uint32 16384, tag CLUSTER.
· clusterGridRanges — Uint16 8192, tag CLUSTER.
· lightListHandles — Uint32 4096, tag LIGHT_LIST.
· envPaletteLinear — Float32 1024, tag ENV.
· interiorVolumeLattice — Float32 16384, tag INTERIOR.
· exteriorProbeSH — Float32 8192, tag EXTERIOR.
· postBufferHDR — Float32 16384, tag POST.
· workerTransferScratch — Float32 8192, tag WORKER.

Each reserved slab is created once at boot and reused across the engine's lifetime.

---

Exported Class — TaggedSubPool

A lightweight façade that binds a fixed tag to a shared pool so subsystem code can acquire and release without passing the tag on every call.

Constructor

```
new TaggedSubPool(pool, tag)
```

Instance Methods

· acquireF32(size) — delegates to pool.acquire(size, ARRAY_KIND.F32, tag).
· acquireU32(size) — U32.
· acquireU16(size) — U16.
· acquireU8(size) — U8.
· acquireI32(size) — I32.
· release(view) — delegates to pool.release(view).
· reserveNamed(name, size, kind = F32) — delegates to pool.reserveNamed(name, size, kind, tag).

---

Exported Functions

getDefaultTypedArrayPool()

Returns: the module-level singleton TypedArrayPool, creating it on first call and immediately calling reserveLightingSlabs() on it.

getDefaultLightingSlabs()

Returns: the named-reservation object from the default pool.

disposeDefaultTypedArrayPool()

Returns: nothing.

createTypedArrayPool(options = {})

Returns: a new TypedArrayPool.

createTaggedSubPool(pool, tag)

Returns: a new TaggedSubPool.

_findBucket(size)

Internal. Linear scan over SIZE_BUCKETS to find the bucket index that can serve size.

_nextPoolId()

Internal. Returns the next monotonic pool id.

---

Default Export

The default export bundles: TypedArrayPool, TaggedSubPool, Slab, createTypedArrayPool, createTaggedSubPool, getDefaultTypedArrayPool, getDefaultLightingSlabs, disposeDefaultTypedArrayPool, reserveLightingSlabs, ARRAY_KIND, ARRAY_KIND_NAME, ARRAY_KIND_CTOR, ARRAY_KIND_BYTES, SLAB_TAG, SLAB_TAG_NAME, SIZE_BUCKETS, BUCKET_COUNT, BUCKET_CAPACITY.

---

Usage Pattern

A subsystem that wants a tagged sub-pool:

```
import {
  getDefaultTypedArrayPool,
  createTaggedSubPool,
  SLAB_TAG,
  ARRAY_KIND,
} from './src/core/011_rnd_TypedArrayPool.js';

const pool = getDefaultTypedArrayPool();
const giPool = createTaggedSubPool(pool, SLAB_TAG.GI);

// Acquire a scratch slab for the GI probe bake.
const scratch = giPool.acquireF32(1024);
// ... write into scratch ...
giPool.release(scratch);
```

A subsystem that owns a persistent slab:

```
const grid = pool.reserveNamed('myGIGrid', 65536, ARRAY_KIND.F32, SLAB_TAG.GI);
// grid is a Float32Array view of 65536 elements.
// It will not be released by releaseAll() and persists across frames.
```

A subsystem that wants to move a slab to a worker:

```
const scratch = pool.acquire(4096, ARRAY_KIND.F32, SLAB_TAG.WORKER);
const slab = scratch.__slabRef;
const buffer = pool.detach(scratch);
worker.postMessage({ buffer, slabIndex: slab.index }, [buffer]);

// On the worker:
worker.onmessage = ({ data }) => {
  pool.reattach(slab, data.buffer);
  // now use the slab normally
};
```

The pool guarantees that no allocation happens after the first frame. Every acquire and release is O(1), and the underlying buffer never moves. On Android, this eliminates virtually all GC activity in the lighting stack's typed-array path.
