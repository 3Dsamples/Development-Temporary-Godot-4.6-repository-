API Documentation — src/core/014_rnd_ResourceRegistry.js

File Purpose

This file is the authoritative GPU resource registry for the anime lighting stack on Android mobile. It tracks every GPU-resident resource the lighting pipeline creates — geometries, materials, shaders, textures, render targets, buffer attributes, interleaved buffers, and post-processing passes — with reference counting, ownership tagging, lifecycle discipline, and deterministic disposal.

The relationship between this module and the resource pools (010–013) is precise:

· The pools manage storage. A pool owns a slab of memory and can hand out many views of it.
· The registry manages lifetime. A subsystem acquires a resource from a pool, registers it here with an owner id, and gets back a stable handle. When the subsystem disposes, it releases the handle; the registry decrements the refcount and, when it hits zero, returns the resource to its owning pool (or disposes it directly if the pool does not own it).

The registry answers the question "which subsystem owns this GPU resource, and is that subsystem still alive?" — which is the single most useful question when debugging a GPU memory leak on Android. A chunk unloads, but its shadow depth texture stays alive. A room exits, but its probe grid keeps accumulating. The registry catches both.

The registry exists to eliminate four classes of GPU leak that Android produces:

1. Orphaned render targets — a subsystem creates a target, forgets to release it, and the pool's slot stays in-use forever.
2. Dangling texture references — a material holds a texture that was disposed, causing black or NaN output.
3. Zombie resources after subsystem disposal — a subsystem is disposed but its resources are still registered, keeping GPU memory alive.
4. Double-disposal — two subsystems both try to dispose the same resource.

The registry prevents all four via reference counting, owner tracking, and generation-tagged handles.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

MAX_RESOURCES

Type: number

Value: 8192 on HIGH, 4096 on MEDIUM, 2048 on LOW.

The fixed capacity of the resource slot table. Sized to accommodate the worst-case number of GPU resources the lighting stack will ever hold simultaneously (all lights + all shadow targets + all GI probes + all AO targets + all environment resources) with headroom.

MAX_NAMED

Type: number

Value: 512

The maximum number of named reservations. Named reservations are for long-lived resources that carry a stable name (e.g. 'shadowAtlasTarget', 'giProbeGrid').

MAX_OWNERS

Type: number

Value: 256

The maximum number of registered owners. An owner is a subsystem, a chunk, a room, an LOD level — anything with a lifecycle that owns resources.

RES_KIND

Type: frozen enum

Values:

· UNKNOWN = 0
· GEOMETRY = 1
· MATERIAL = 2
· SHADER = 3
· TEXTURE = 4
· RENDER_TARGET = 5
· BUFFER_ATTRIBUTE = 6
· INTERLEAVED = 7
· BUFFER = 8
· PROGRAM = 9
· SAMPLER = 10
· FBO_WRAPPER = 11
· PASS = 12
· COUNT = 13

RES_KIND_NAME

Type: frozen array

Values: ['unknown', 'geometry', 'material', 'shader', 'texture', 'render_target', 'buffer_attribute', 'interleaved', 'buffer', 'program', 'sampler', 'fbo_wrapper', 'pass'].

RES_TAG

Type: frozen enum

Per-resource purpose tags, parallel to RT_TAG in 013 and SLAB_TAG in 011.

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
· DIRECTOR = 10
· DIAGNOSTIC = 11
· COUNT = 12

RES_TAG_NAME

Type: frozen array

Values: ['generic', 'shadow', 'gi', 'ao', 'cluster', 'light_list', 'env', 'interior', 'exterior', 'post', 'director', 'diagnostic'].

RES_STATE

Type: frozen enum

Values:

· FREE = 0 — the slot is not in use.
· LIVE = 1 — the resource is registered and its owner is still active.
· ZOMBIE = 2 — the resource's owner was disposed but the resource was not released.
· DISPOSED = 3 — the resource has been disposed and the slot is being recycled.

RES_POOL_ID_SYMBOL, RES_HANDLE_SYMBOL

Internal string constants used as non-enumerable tags on registered resources. The first marks which registry owns the resource; the second records the handle the registry assigned.

---

Module-Level State (Not Exported Directly)

_registryIdCounter

Type: number

Monotonic counter for registry ids.

_defaultRegistry

Type: ResourceRegistry | null

The module-level singleton.

---

Exported Class — ResourceSlot

One instance per registered resource.

Constructor

```
new ResourceSlot(index)
```

Parameters: index — the slot's array index.

Instance Properties

· index — the slot's array index.
· handle — the public handle: (index & 0xFFFF) | (generation << 16).
· kind — one of RES_KIND.
· tag — one of RES_TAG.
· state — one of RES_STATE.
· resource — the actual THREE.* resource, or null.
· ownerId — the owning subsystem's id, or -1.
· refCount — the number of references. Starts at 1.
· generation — a monotonic counter that increments every time the slot is recycled. Used to invalidate stale handles.
· createdFrame — the frame when the resource was registered.
· lastUsedFrame — the frame when the resource was last marked as used.
· disposedFrame — the frame when the resource was disposed.
· disposeHook — an optional custom disposer (resource) => void.
· name — an optional diagnostic name.

Instance Methods

reset()

Returns: nothing.

Purpose: zeroes every field. Called when the slot is returned to the free list.

---

Exported Class — OwnerSlot

One instance per registered owner.

Constructor

```
new OwnerSlot(index)
```

Instance Properties

· index — the slot's index.
· id — the owner's public id.
· name — a diagnostic name.
· active — 1 if the owner is still alive, 0 if disposed.
· totalOwned — the cumulative count of resources ever registered against this owner.
· createdAt — the timestamp when the owner was registered.
· disposedAt — the timestamp when the owner was disposed.

Instance Methods

reset()

Returns: nothing.

---

Exported Class — ResourceRegistry

The main registry.

Constructor

```
new ResourceRegistry(options = {})
```

Parameters:

· name — the registry's diagnostic name. Default res_registry_<id>.
· capacity — the number of resource slots. Default MAX_RESOURCES.
· autoDispose — if true, releasing a resource to refcount zero disposes it. Default true.
· leakTracking — if true, tracks which resources leak when their owner is disposed. Default true.

Constructor work:

1. Allocates slots — the array of ResourceSlot instances.
2. Allocates and initializes the freeList ring buffer.
3. Allocates _handleIndex — an Int32Array(capacity) used to look up slot indices by packed handle index. -1 means no slot.
4. Allocates owners — the array of OwnerSlot instances.
5. Initializes ownerCount and _nextOwnerId.
6. Allocates _named — a Map from name to handle.
7. Initializes the stats object.
8. Allocates _listeners (Map).

Instance Properties

· name — the registry name.
· id — the unique registry id.
· capacity — the resource slot count.
· autoDispose — the auto-dispose flag.
· leakTracking — the leak-tracking flag.
· slots — the resource slot array.
· freeList — the free-list ring.
· freeHead — the ring read pointer.
· freeCount — the number of free slots.
· _handleIndex — the handle-to-slot index table.
· owners — the owner slot array.
· ownerCount — the number of registered owners.
· frame — the frame counter.
· stats — the aggregate stats object.

The stats object has: totalRegistered, totalReleased, totalDisposed, totalZombies, totalLeaksDetected, currentLive, peakLive.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'owner-registered', 'owner-disposed', 'owner-bulk-dispose', 'registered', 'disposed', 'leak', 'rejected'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

registerOwner(name)

Parameters: name — a diagnostic name.

Returns: the owner's id (a positive integer), or -1 if the owner table is full.

Purpose: registers a subsystem, chunk, room, or any resource-owning entity. The returned id is passed to register() for every resource the owner creates.

disposeOwner(ownerId)

Parameters: ownerId — the owner's id.

Returns: boolean.

Purpose: marks the owner as inactive. Subsequent auditLeaks() calls will flag any of its resources whose refcount is still above zero as zombies.

isOwnerActive(ownerId)

Parameters: ownerId — the owner's id.

Returns: boolean.

_findOwnerSlot(ownerId)

Internal. Linear scan through the owners array.

register(resource, options = {})

Parameters:

· resource — the THREE.* resource to track. Required.
· options.kind — one of RES_KIND.
· options.tag — one of RES_TAG.
· options.ownerId — the owner's id. Default -1.
· options.name — an optional diagnostic name.
· options.disposeHook — an optional custom disposer (resource) => void. If omitted, the registry calls resource.dispose() at release-to-zero time.

Returns: a stable handle (a positive integer), or 0 on failure.

Flow:

1. Verifies the resource is non-null.
2. Pops a slot from the free list. Returns 0 if none available.
3. Resets and populates the slot.
4. Increments the slot's generation counter, wrapping at 0x7FFF, and computes the new handle as (index & 0xFFFF) | (generation << 16).
5. Sets the slot's state to LIVE and its refcount to 1.
6. Attempts to attach the pool id and handle as non-enumerable properties on the resource for fast identity lookup. Silently ignores failures if the resource rejects property writes.
7. Updates the aggregate counters.
8. If ownerId >= 0, increments the owner's totalOwned.
9. Emits registered.
10. Returns the handle.

_resolveSlot(handle)

Internal. Extracts the slot index from the low 16 bits and the generation from the high bits. Returns the slot if the generation matches and the slot is not FREE, otherwise null.

retain(handle)

Parameters: handle — a valid handle.

Returns: the new refcount, or -1 if the handle is invalid.

Purpose: increments the resource's refcount. Called when a second subsystem wants to share the resource.

release(handle)

Parameters: handle — a valid handle.

Returns: the new refcount after decrement, or -1 if the handle is invalid.

Purpose: decrements the resource's refcount. If the refcount reaches zero and autoDispose is on, calls _disposeSlot() to dispose the resource and return the slot to the free list.

forceRelease(handle)

Parameters: handle — a valid handle.

Returns: boolean.

Purpose: sets the refcount to zero and disposes the resource immediately, regardless of how many references were outstanding. Used for hard teardown when the caller knows no other reference will be released.

markUsed(handle)

Parameters: handle — a valid handle.

Returns: boolean.

Purpose: records the current frame as the resource's lastUsedFrame. Used by age-based GC.

getResource(handle)

Parameters: handle — a valid handle.

Returns: the underlying resource, or null.

getRefCount(handle)

Parameters: handle — a valid handle.

Returns: the current refcount, or 0.

_disposeSlot(slot)

Internal. Calls the slot's disposeHook or, if none was supplied, calls resource.dispose(). Removes the resource from the named map if applicable. Clears every field. Returns the slot to the free list and increments the slot's generation so the old handle is invalidated. Emits disposed.

disposeAllForOwner(ownerId)

Parameters: ownerId — the owner's id.

Returns: the number of resources disposed.

Purpose: iterates the slot table and force-disposes every resource whose ownerId matches. This is the single most important API for Android — when a chunk, room, or LOD level is unloaded, this drops every GPU resource it created in one call. Emits owner-bulk-dispose.

reserveNamed(name, resource, options = {})

Parameters:

· name — a unique name.
· resource — the THREE.* resource.
· options.kind, options.tag, options.ownerId, options.disposeHook — same as register.

Returns: the handle, or 0.

Purpose: registers a resource and records it in the named map. If the name is already taken, returns the existing handle.

getNamed(name)

Parameters: name — the reservation name.

Returns: the resource, or null.

getNamedHandle(name)

Parameters: name — the reservation name.

Returns: the handle, or 0.

releaseNamed(name)

Parameters: name — the reservation name.

Returns: boolean.

Purpose: removes the name from the map and force-releases the handle.

beginFrame()

Returns: this. Increments the frame counter.

endFrame()

Returns: this. Reserved for future expansion.

auditLeaks(outHandles)

Parameters: outHandles — an optional Int32Array or Array to receive the leaked handles.

Returns: the number of leaked resources.

Purpose: iterates every LIVE resource. If the resource's owner is inactive (either not found or active === 0), transitions the resource to ZOMBIE, increments the leak counters, emits leak, and optionally writes the handle to outHandles.

This is the single most useful tool for finding GPU memory leaks on Android. Calling auditLeaks() once per second from a debug HUD will list every resource whose owner is gone but whose GPU memory is still alive.

purgeZombies()

Returns: the number of zombies purged.

Purpose: force-disposes every slot in ZOMBIE state. Call this after auditLeaks() to reclaim the leaked resources.

collectUnused(maxAgeFrames, tagMask = 0xFFFF)

Parameters:

· maxAgeFrames — the maximum age in frames before a resource is considered cold.
· tagMask — a bitmask selecting which tags to collect from. Default all.

Returns: the number of resources collected.

Purpose: disposes every LIVE resource that has not been marked used within the last maxAgeFrames frames. Named resources are protected by default. This is the age-based GC for warm resources that live longer than one frame but shorter than the subsystem's full lifetime.

getStats()

Returns: an object with name, frame, capacity, freeCount, currentLive, peakLive, the five aggregate counters, namedCount, ownerCount, a owners array with per-owner stats, a kindStats array with per-kind live counts, and a tagStats array with per-tag live counts.

reset()

Returns: this. Disposes every live resource, resets every slot and owner, and zeroes every counter.

dispose()

Returns: this. Resets and nulls every internal array.

---

Exported Functions

getDefaultResourceRegistry()

Returns: the module-level singleton ResourceRegistry, creating it on first call.

disposeDefaultResourceRegistry()

Returns: nothing.

createResourceRegistry(options = {})

Returns: a new ResourceRegistry.

registerGeometry(geometry, options = {})

Parameters:

· geometry — a THREE.BufferGeometry.
· options — same as register.

Returns: the handle.

Purpose: convenience wrapper that sets kind: RES_KIND.GEOMETRY.

registerMaterial(material, options = {})

Purpose: sets kind: RES_KIND.MATERIAL.

registerTexture(texture, options = {})

Purpose: sets kind: RES_KIND.TEXTURE.

registerRenderTarget(target, options = {})

Purpose: sets kind: RES_KIND.RENDER_TARGET.

registerBufferAttribute(attribute, options = {})

Purpose: sets kind: RES_KIND.BUFFER_ATTRIBUTE.

registerInterleavedBuffer(buffer, options = {})

Purpose: sets kind: RES_KIND.INTERLEAVED.

registerPass(pass, options = {})

Purpose: sets kind: RES_KIND.PASS.

---

Default Export

The default export bundles: ResourceRegistry, ResourceSlot, OwnerSlot, createResourceRegistry, getDefaultResourceRegistry, disposeDefaultResourceRegistry, the seven register* helpers, RES_KIND, RES_KIND_NAME, RES_TAG, RES_TAG_NAME, RES_STATE, MAX_RESOURCES, MAX_NAMED, MAX_OWNERS.

---

Usage Pattern

A subsystem that owns resources:

```
import {
  getDefaultResourceRegistry,
  RES_KIND,
  RES_TAG,
} from './src/core/014_rnd_ResourceRegistry.js';

const registry = getDefaultResourceRegistry();
const ownerId = registry.registerOwner('shadowSystem');

// When the subsystem creates a resource:
const shadowTarget = createShadowTarget(); // from pool 013
const handle = registry.register(shadowTarget, {
  kind: RES_KIND.RENDER_TARGET,
  tag: RES_TAG.SHADOW,
  ownerId,
  name: 'mainShadowTarget',
});

// When the subsystem is disposed:
registry.disposeOwner(ownerId);
registry.disposeAllForOwner(ownerId);
```

A debug tool that runs once per second:

```
const leaks = registry.auditLeaks();
if (leaks > 0) {
  console.warn(`${leaks} leaked resources`);
  registry.purgeZombies();
}
```

The registry guarantees that the disposeAllForOwner call drops every GPU resource the subsystem owned in a single pass, without any cross-subsystem interference. This is what makes chunk unload, room exit, and LOD swap safe and fast on Android.
