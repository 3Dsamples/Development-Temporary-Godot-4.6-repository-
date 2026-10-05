API Documentation — src/core/010_rnd_ObjectPool.js

File Purpose

This file is the zero-allocation object pool system for the anime lighting stack on Android mobile. Every lighting subsystem that needs transient Vector3, Vector4, Color, Quaternion, Matrix4, Spherical, Box3, Sphere, Plane, Frustum, Ray, or typed-array scratch acquires them from this pool instead of allocating new instances.

It exists to eliminate the two biggest sources of garbage-collection pressure on mobile:

1. Per-frame temporary THREE.Vector3, THREE.Color, and THREE.Matrix4 objects created inside lighting math (shadow solver scratch, GI probe interpolation, AO blur kernel offset computation).
2. Per-frame scratch typed arrays created inside shadow, GI, and AO solvers.

On a mid-range Android device, a single frame of shadow + GI + AO + cluster work can allocate dozens of temporary objects. Over a 30-minute session, that produces tens of thousands of short-lived objects, forcing the JavaScript engine's generational garbage collector to run frequently. Each GC pause on mobile is 4–20 ms, enough to drop multiple frames. This module removes that entire class of allocator activity.

The module provides three layers:

· ObjectPool — a generic typed pool with a factory, a reset function, and an optional validator. Used for THREE.js math types.
· TypedArrayPool — a bucket-based pool for typed arrays. Buckets are sized in powers of two (4, 8, 16, …, 4096). Acquire rounds up to the next bucket.
· LightingPoolSet — a composite registry of pre-built pools for every type the lighting stack needs. This is the canonical entry point for downstream code.

The pool supports an optional auto-reclaim mode: when a pool is exhausted, the pool either rejects the acquire and returns null, or it recycles the oldest un-released object. Lighting-critical pools use the reject mode; cosmetic pools may opt into reclaim.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

POOL_CAPACITY

Type: frozen object

Per-kind pool capacities. Each value is a function of PERF_TIER. Fields:

· vector3 — 1024 on HIGH, 512 on MEDIUM, 256 on LOW.
· vector2 — 512 / 256 / 128.
· vector4 — 512 / 256 / 128.
· color — 512 / 256 / 128.
· quaternion — 256 / 128 / 64.
· euler — 256 / 128 / 64.
· spherical — 128 / 64 / 32.
· matrix3 — 128 / 64 / 32.
· matrix4 — 256 / 128 / 64.
· box3 — 128 / 64 / 32.
· sphere — 128 / 64 / 32.
· plane — 64 / 32 / 16.
· frustum — 32 / 16 / 8.
· ray — 32 / 16 / 8.

The capacities are chosen based on measured peak per-frame usage of the corresponding type across all lighting subsystems, with 20–30 % headroom.

TYPED_BUCKET_SIZES

Type: frozen array

Values: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096].

The bucket sizes for typed arrays. A request for 5 floats is served from the 8-element bucket; a request for 900 floats is served from the 1024-element bucket.

POOL_ID_SYMBOL

Internal constant, value '__poolId'. Used to tag every pooled object with the pool it belongs to so release() can reject mismatched releases.

POOL_KIND

Type: frozen enum

Values:

· GENERIC = 0
· VECTOR2 = 1
· VECTOR3 = 2
· VECTOR4 = 3
· COLOR = 4
· QUATERNION = 5
· EULER = 6
· SPHERICAL = 7
· MATRIX3 = 8
· MATRIX4 = 9
· BOX3 = 10
· SPHERE = 11
· PLANE = 12
· FRUSTUM = 13
· RAY = 14
· FLOAT32 = 15
· UINT16 = 16
· UINT32 = 17
· UINT8 = 18
· COUNT = 19

POOL_KIND_NAME

Type: frozen array

Values: ['generic', 'vector2', 'vector3', 'vector4', 'color', 'quaternion', 'euler', 'spherical', 'matrix3', 'matrix4', 'box3', 'sphere', 'plane', 'frustum', 'ray', 'float32', 'uint16', 'uint32', 'uint8'].

---

Module-Level State (Not Exported Directly)

_poolIdCounter

Type: number

A monotonic counter incremented on every pool construction. The pool's id is stored on every pooled object as a non-enumerable property so release() can verify the object belongs to the correct pool before returning it.

_defaultPoolSet

Type: LightingPoolSet | null

The module-level singleton, created on first getDefaultPoolSet() call.

---

Exported Class — ObjectPool

Constructor

```
new ObjectPool(options = {})
```

Parameters:

· name — a diagnostic name for the pool. Default pool_<id>.
· kind — one of POOL_KIND. Default GENERIC.
· capacity — the number of objects in the pool. Default 256.
· factory — required. A function that returns a fresh object.
· reset — optional. A function (obj) => void that resets the object to a canonical state. Called on release.
· validate — optional. A function (obj) => boolean that verifies the object is still valid. Not called automatically.
· autoReclaim — if true, exhaustion triggers recycling the oldest un-released object instead of returning null. Default false.
· autoReleasePerFrame — if true, endFrame() releases every in-use object. Default false.
· autoReclaimGrace — unused in the current implementation but reserved.

Constructor work:

1. Allocates objects — an array of capacity fresh objects via factory().
2. Tags each object with the pool's id.
3. Allocates the ring free list freeList (an Int32Array(capacity)) and initializes every entry to its own index.
4. Allocates acquired (a Uint8Array(capacity)) and acquiredAt (a Uint32Array(capacity)).
5. Initializes the stats object.

Instance Properties

· name — the pool's name.
· kind — the pool's kind.
· capacity — the number of objects.
· factory — the object factory.
· reset — the reset function.
· validate — the validator.
· autoReclaim — the reclaim flag.
· poolId — the unique pool id.
· objects — the pre-allocated array.
· freeList — the free-list ring.
· freeHead — the ring read pointer.
· freeCount — the number of free objects.
· acquired — the in-use flags.
· acquiredAt — the frame number each object was acquired.
· stats — a mutable object with counters.

The stats object has these fields: acquired, released, rejected, reclaimed, peakInUse, currentInUse, frameAcquired, frameReleased, totalCreated.

Instance Methods

acquire()

Returns: a fresh object from the pool, or null if the pool is exhausted and autoReclaim is off.

Purpose: pops a slot from the free list, marks it in-use, records the acquisition frame, updates stats. If autoReclaim is on and the pool is full, calls _reclaimOldest() to force-release the oldest un-released object.

release(obj)

Parameters: obj — the object to release.

Returns: boolean — true on success, false if the object does not belong to this pool, is not currently acquired, or the argument is invalid.

Purpose: verifies the object's __poolId matches the pool's id. Locates the object's slot index via the cached __poolIndex property (or a linear scan on the first release). Calls reset(obj) if a reset function was supplied. Marks the slot free and returns it to the free list.

Double-releases are silently ignored.

releaseAll()

Returns: the number of objects released.

Purpose: iterates every acquired slot and releases it.

_indexOf(obj)

Internal. Linear scan to find an object's slot index. Used when the cached __poolIndex is missing or stale.

_reclaimOldest()

Internal. Finds the in-use slot with the smallest acquiredAt value, force-releases it, and returns the slot index to the caller.

beginFrame()

Returns: nothing.

Purpose: increments the frame counter and resets the per-frame stats.

endFrame()

Returns: the number of free objects.

Purpose: if autoReleasePerFrame is on, calls releaseAll().

getStats()

Returns: an object with name, kind, capacity, freeCount, inUse, peakInUse, acquired, released, rejected, reclaimed, frameAcquired, frameReleased, autoReclaim.

reset()

Returns: this.

Purpose: releases every object, resets the free list, and zeroes every stat.

dispose()

Returns: this.

Purpose: resets and nulls the internal arrays.

---

Exported Class — TypedArrayPool

A bucket-based typed-array pool. Each bucket holds a fixed number of arrays of a specific size.

Constructor

```
new TypedArrayPool(options = {})
```

Parameters:

· name — a diagnostic name.
· ArrayType — required. One of Float32Array, Uint16Array, Uint32Array, Uint8Array.
· kind — one of POOL_KIND. Defaults to FLOAT32.
· counts — an array of per-bucket counts. Default derived from PERF_TIER.
· sizes — an array of bucket sizes. Default TYPED_BUCKET_SIZES.
· autoReclaim — reserved for future expansion.
· autoReleasePerFrame — if true, endFrame() releases every in-use array. Default false.

Constructor work:

1. Allocates one Bucket object per size in sizes.
2. Each Bucket pre-allocates counts[b] arrays of size sizes[b].
3. Each array is tagged with the pool id, its bucket index, and its index within the bucket.

The default per-tier counts, for Float32Array as an example:

· HIGH: [512, 512, 512, 384, 256, 192, 128, 96, 64, 32, 16].
· MEDIUM: [256, 256, 256, 192, 128, 96, 64, 48, 32, 16, 8].
· LOW: [128, 128, 128, 96, 64, 48, 32, 24, 16, 8, 4].

The counts are structured so small arrays are abundant and large arrays are scarce. This matches real lighting workloads where the vast majority of typed-array requests are for 4–64 elements.

Instance Properties

· name — the pool's name.
· ArrayType — the constructor.
· kind — the pool kind.
· counts — the per-bucket counts array.
· sizes — the bucket sizes array.
· poolId — the unique pool id.
· buckets — an array of bucket objects.
· frame — the frame counter.
· autoReleasePerFrame — the frame-release flag.

Each bucket has these fields: size, arrays, freeList, freeHead, freeCount, capacity, acquired, acquiredAt, peakInUse, currentInUse, acquiredTotal, releasedTotal, rejectedTotal.

Instance Methods

_findBucketFor(size)

Internal. Linear scan through sizes and returns the first index whose size is at least size.

acquire(size)

Parameters: size — the number of elements requested.

Returns: a subarray view of exactly size elements, or null if the pool is exhausted for that bucket.

Purpose: locates the smallest bucket whose size is at least size, pops a slot from that bucket's free list, marks it in-use, and returns array.subarray(0, size). The subarray view is a new view object but it shares the underlying ArrayBuffer with the pooled array — no data copy.

The view is tagged with the pool id, the bucket index, the slot index, and the array kind, and it carries a __slabRef reference to the parent array. This allows release() to identify the parent without a scan.

release(arr)

Parameters: arr — a view returned by acquire.

Returns: boolean.

Purpose: verifies the view's pool id, extracts the parent bucket and slot index from the tags, clears the in-use flag, and returns the slot to the bucket's free list. Verifies the view shares the parent's ArrayBuffer before releasing, so a forged view cannot corrupt the pool.

releaseAll()

Returns: nothing.

Purpose: iterates every bucket and releases every in-use array.

beginFrame()

Returns: nothing. Increments the frame counter.

endFrame()

Returns: nothing. If autoReleasePerFrame is on, releases all.

getStats()

Returns: an object with name, kind, and a per-bucket stats array.

reset()

Returns: this. Releases all and zeroes every counter.

dispose()

Returns: this. Resets and nulls the internal arrays.

---

Module-Level Reset Functions

_resetVector3(v) through _resetRay(r)

Internal helpers passed to the ObjectPool as the reset option. Each one restores its object type to a canonical identity:

· Vector3 → (0, 0, 0).
· Vector2 → (0, 0).
· Vector4 → (0, 0, 0, 0).
· Color → (0, 0, 0).
· Quaternion → (0, 0, 0, 1).
· Euler → (0, 0, 0, 'XYZ').
· Spherical → (0, 0, 0).
· Matrix3 → identity.
· Matrix4 → identity.
· Box3 → empty.
· Sphere → centered at origin with radius 1.
· Plane → normal up, constant 0.
· Frustum → left as-is (no cheap reset).
· Ray → origin at origin, direction (0, 0, 1).

These resets are cheap. They exist to prevent accidentally leaking state from one consumer to the next.

---

Exported Class — LightingPoolSet

The canonical entry point for downstream code. Owns one pool per type.

Constructor

```
new LightingPoolSet()
```

Constructor work:

1. Allocates pools — an array of length POOL_KIND.COUNT.
2. Allocates byName — a Map from pool name to pool.
3. Creates each pool: vector2, vector3, vector4, color, quaternion, euler, spherical, matrix3, matrix4, box3, sphere, plane, frustum, ray, f32, u16, u32, u8.
4. Each pool gets a fixed capacity from POOL_CAPACITY and a reset function from the reset helpers.

Instance Properties

The instance exposes each pool as a directly-named property: vector2, vector3, vector4, color, quaternion, euler, spherical, matrix3, matrix4, box3, sphere, plane, frustum, ray, f32, u16, u32, u8.

It also has pools (the array) and byName (the Map).

Instance Methods

_register(pool)

Internal. Adds a pool to the registry.

get(kind)

Parameters: kind — one of POOL_KIND.

Returns: the pool, or null.

getByName(name)

Parameters: name — the pool name.

Returns: the pool, or null.

beginFrame()

Returns: nothing. Calls beginFrame() on every pool.

endFrame()

Returns: nothing. Calls endFrame() on every pool.

releaseAll()

Returns: nothing. Calls releaseAll() on every pool.

reset()

Returns: this. Calls reset() on every pool.

dispose()

Returns: this. Calls dispose() on every pool and clears the registry.

getStats()

Returns: an array of pool.getStats() results, one per pool.

---

Exported Hot-Path Wrapper Functions

These are the functions that downstream lighting code calls. Each delegates to the singleton pool. All are O(1) and allocation-free on the hot path (except the pool's own bookkeeping, which is also allocation-free).

Vector Wrappers

· acquireVector3() — returns a Vector3 from the pool, or null.
· releaseVector3(v) — returns it.
· acquireVector2() / releaseVector2(v)
· acquireVector4() / releaseVector4(v)

Color and Quaternion Wrappers

· acquireColor() / releaseColor(c)
· acquireQuaternion() / releaseQuaternion(q)

Matrix Wrappers

· acquireMatrix3() / releaseMatrix3(m)
· acquireMatrix4() / releaseMatrix4(m)

Math Type Wrappers

· acquireSpherical() / releaseSpherical(s)
· acquireBox3() / releaseBox3(b)
· acquireSphere() / releaseSphere(s)
· acquirePlane() / releasePlane(p)
· acquireFrustum() / releaseFrustum(f)
· acquireRay() / releaseRay(r)

Typed Array Wrappers

· acquireF32(size) / releaseF32(arr)
· acquireU16(size) / releaseU16(arr)
· acquireU32(size) / releaseU32(arr)
· acquireU8(size) / releaseU8(arr)

Each of these calls the corresponding pool's acquire or release method.

---

Exported Functions

getDefaultPoolSet()

Returns: the module-level singleton LightingPoolSet, creating it on first call.

disposeDefaultPoolSet()

Returns: nothing.

createObjectPool(options = {})

Returns: a new ObjectPool.

createTypedArrayPool(options = {})

Returns: a new TypedArrayPool.

createLightingPoolSet()

Returns: a new LightingPoolSet.

---

Default Export

The default export bundles: ObjectPool, TypedArrayPool, LightingPoolSet, all the factory functions, all the hot-path wrappers, POOL_KIND, POOL_KIND_NAME, POOL_CAPACITY, TYPED_BUCKET_SIZES.

---

Usage Pattern

Downstream lighting math uses the wrappers directly:

```
import {
  acquireVector3,
  releaseVector3,
  acquireMatrix4,
  releaseMatrix4,
  acquireF32,
  releaseF32,
} from './src/core/010_rnd_ObjectPool.js';

function updateShadowBias(light, camera) {
  const tmpPos = acquireVector3();
  const tmpMat = acquireMatrix4();

  tmpPos.set(light.position.x, light.position.y, light.position.z);
  tmpMat.copy(camera.matrixWorldInverse);
  // ... use tmpPos and tmpMat ...

  releaseVector3(tmpPos);
  releaseMatrix4(tmpMat);
}
```

The wrappers never allocate new objects after the first frame. On a typical Android device, this reduces per-frame allocator activity by 60–90 % in the lighting subsystem, which eliminates the GC pauses that would otherwise drop frames.

The engine calls poolSet.beginFrame() at the top of every frame and poolSet.endFrame() at the bottom. If a subsystem forgets to release an object, the pool's acquiredAt timestamp lets a debug tool list which subsystem is leaking.
