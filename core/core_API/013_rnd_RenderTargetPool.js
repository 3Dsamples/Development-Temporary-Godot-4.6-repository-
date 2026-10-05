API Documentation — src/core/013_rnd_RenderTargetPool.js

File Purpose

This file is the GPU render-target pool for the anime lighting stack on Android mobile. Where 012_rnd_BufferPool.js owns VBO and attribute memory, this module owns framebuffer memory. Every lighting subsystem that renders to texture acquires its render target from here — never new THREE.WebGLRenderTarget(...) inside a hot loop.

The workloads it serves:

· Shadow map targets — one depth target per shadow-casting light, or an atlas target for many lights
· GI probe render targets — the ping-pong buffers for probe bake and bounce passes
· AO blur ping-pong targets — full-res, half-res, and quarter-res variants
· Cluster debug targets — visualization of the cluster grid
· Environment palette capture targets — for offline and runtime env prefiltering
· Interior and exterior probe targets — room volume and terrain probe bakes
· Sun ray and volumetric targets — screen-space light scattering
· Post-processing intermediates — bloom, HDR composite, contact shadows, SSGI, ReSTIR reservoirs

The design is a two-dimensional grid:

1. Kind — the color format and depth configuration. Eight kinds are supported: COLOR_RGBA8, COLOR_RGB8, COLOR_RGBA16F, COLOR_RGBA32F, DEPTH, DEPTH_STENCIL, COMBINED_RGBA8, COMBINED_RGBA16F.
2. Rung — a size from a fixed ladder: 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192. Any requested size rounds up to the next rung. This guarantees no dynamic reallocation when the viewport scales between 0.25× and 1.0× — the resolution scaler just picks a different ladder rung.

The pool handles Android-specific quirks:

· Avoids generateMipmaps on HDR targets (costly on tile-based renderers)
· Forces LinearFilter on color, NearestFilter on depth
· Uses HalfFloatType for HDR when EXT_color_buffer_half_float or WebGL2 is available, falls back to UnsignedByteType with tonemapping in shader
· Respects depthBuffer and stencilBuffer flags per kind so no wasted attachments
· Detects MSAA support and uses the highest safe sample count per kind
· Forces MSAA to be fixed at creation — MSAA-bearing targets never change sample count

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

RT_KIND

Type: frozen enum

Values:

· COLOR_RGBA8 = 0
· COLOR_RGB8 = 1
· COLOR_RGBA16F = 2
· COLOR_RGBA32F = 3
· DEPTH = 4
· DEPTH_STENCIL = 5
· COMBINED_RGBA8 = 6
· COMBINED_RGBA16F = 7
· COUNT = 8

RT_KIND_NAME

Type: frozen array

Values: ['color_rgba8', 'color_rgb8', 'color_rgba16f', 'color_rgba32f', 'depth', 'depth_stencil', 'combined_rgba8', 'combined_rgba16f'].

RT_TAG

Type: frozen enum

Per-target purpose tags. Used for debug visualization and per-tag stats.

Values:

· GENERIC = 0
· SHADOW = 1
· GI = 2
· AO = 3
· CLUSTER = 4
· ENV = 5
· INTERIOR = 6
· EXTERIOR = 7
· POST = 8
· SUNRAY = 9
· VOLUMETRIC = 10
· CONTACT = 11
· SSGI = 12
· RESTIR = 13
· COUNT = 14

RT_TAG_NAME

Type: frozen array

Values: ['generic', 'shadow', 'gi', 'ao', 'cluster', 'env', 'interior', 'exterior', 'post', 'sunray', 'volumetric', 'contact', 'ssgi', 'restir'].

SIZE_LADDER

Type: frozen array

Values: [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192].

Fifteen rungs. The ratios between consecutive rungs are between 1.33 and 1.5, so the internal fragmentation when rounding up is at most 50 % and typically much less.

LADDER_COUNT

Type: number

Value: 15 — the length of SIZE_LADDER.

RT_CAPACITY

Type: frozen array of 15 numbers

Per-rung capacity (number of targets) as a function of PERF_TIER.

On HIGH: [16, 16, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4].

On MEDIUM: [8, 8, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2].

On LOW: [4, 4, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1].

The small-rung counts are higher because small targets are frequently ping-ponged. The large-rung counts are small because 8K targets are reserved for shadows and HDR post, and even on HIGH there are only a handful.

RT_POOL_ID_SYMBOL, RT_SLOT_SYMBOL

Internal string constants used as tags on WebGLRenderTarget instances so releaseTarget() can find the parent slot in O(1).

HW_CAPS

Type: frozen object

Hardware capability probe run once at module init. Fields:

· halfFloatColor — boolean, whether the device can render to HalfFloatType color textures
· floatColor — boolean, whether the device can render to FloatType color textures
· depthTexture — boolean, whether the device supports depth textures
· webgl2 — boolean, whether the context is WebGL2

The probe creates a temporary canvas, tries webgl2 then webgl, reads the relevant extensions, then calls loseContext to release the probe context. It never leaves a WebGL context alive.

---

Module-Level State (Not Exported Directly)

_rtPoolIdCounter

Type: number

Monotonic counter for pool ids.

_defaultRTPool

Type: RenderTargetPool | null

The module-level singleton.

_defaultLightingTargets

Type: object | null

The named reservations for the standard lighting targets.

---

Exported Class — RTSlot

One slot per render-target instance.

Constructor

```
new RTSlot(index, kind, rung)
```

Parameters:

· index — the slot's index within its bucket.
· kind — one of RT_KIND.
· rung — the size ladder rung.

Instance Properties

· index — the slot's index within its bucket.
· kind — the render-target kind.
· rung — the ladder rung.
· size — the rung's nominal size.
· tag — the current tag (may change between acquires).
· target — the THREE.WebGLRenderTarget, or null before first acquire.
· width, height — the actual target dimensions.
· msaa — the sample count, or 0 for no MSAA.
· inUse — 1 if acquired, 0 if free.
· acquiredAt — the frame number when acquired.
· generation — a monotonic counter incremented on each acquire.
· uploadCount — unused for now, reserved.
· named — the reservation name, or null.
· bytesPerPixel — the estimated bytes per pixel for this kind.
· estimatedBytes — width * height * bytesPerPixel.

Instance Methods

reset()

Returns: nothing. Clears inUse and generation.

---

Exported Class — RenderTargetPool

The main pool.

Constructor

```
new RenderTargetPool(options = {})
```

Parameters:

· name — the pool's diagnostic name. Default rt_pool_<id>.
· autoReleasePerFrame — if true, endFrame() releases every in-use target. Default false.

Constructor work:

1. Allocates buckets — an 8 by 15 grid of bucket objects.
2. Each bucket has slots, freeList, freeHead, freeCount, capacity, and per-bucket stats.
3. Allocates _namedTargets — a Map from name to { target, width, height, kind, tag, options }.
4. Initializes the stats object.
5. Allocates _listeners (Map).

Instance Properties

· name — the pool name.
· poolId — the unique pool id.
· autoReleasePerFrame — the frame-release flag.
· buckets — the 8-by-15 target slot grid.
· frame — the frame counter.
· stats — the aggregate stats object.

The stats object has: acquired, released, rejected, namedCount, gpuBytesInUse, peakGpuBytesInUse, totalAllocated.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'rejected'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_makeTarget(kind, rung, width, height, options)

Internal. Creates a fresh THREE.WebGLRenderTarget with the correct format, type, filters, and depth/stencil flags for the requested kind.

Kind handling:

· COLOR_RGBA8 — RGBA / UnsignedByte, no depth, Linear filter.
· COLOR_RGB8 — RGB / UnsignedByte, no depth, Linear filter.
· COLOR_RGBA16F — RGBA / HalfFloat if supported, otherwise RGBA / UnsignedByte.
· COLOR_RGBA32F — RGBA / Float if supported, RGBA / HalfFloat if not, RGBA / UnsignedByte otherwise.
· DEPTH — Depth / UnsignedInt, depthBuffer on, Nearest filter.
· DEPTH_STENCIL — DepthStencil / UnsignedInt248, depth and stencil on, Nearest filter.
· COMBINED_RGBA8 — RGBA / UnsignedByte, depth on.
· COMBINED_RGBA16F — RGBA / HalfFloat if supported, RGBA / UnsignedByte otherwise, depth on.

When depth texture support is absent, DEPTH and DEPTH_STENCIL fall back to COMBINED_RGBA8 with depth on so at least the depth information is available.

All targets have generateMipmaps: false to avoid tile-based renderer overhead.

acquireTarget(width, height, options = {})

Parameters:

· width, height — the logical dimensions in pixels.
· options.kind — one of RT_KIND. Default COLOR_RGBA8.
· options.tag — one of RT_TAG. Default GENERIC.
· options.samples — MSAA sample count, or 0 for no MSAA. Default 0.

Returns: a THREE.WebGLRenderTarget, or null if the pool is exhausted for that kind and rung.

Flow:

1. Rounds max(width, height) up to the next SIZE_LADDER rung.
2. Pops a slot from the appropriate bucket.
3. If the slot's target is missing, or its dimensions, samples, or kind do not match the request, disposes the old target and creates a new one. Records the new estimatedBytes and adds it to gpuBytesInUse.
4. Tags the target with the pool id and slot info.
5. Marks the slot in-use and returns the target.

Important: on Android, MSAA requires full reallocation when the sample count changes, so the pool tracks msaa per slot and forces reallocation on mismatch.

releaseTarget(target)

Parameters: target — a WebGLRenderTarget returned by acquireTarget.

Returns: boolean.

Purpose: locates the parent slot via the tags on the target, resets the slot, returns it to the free list, and decrements the in-use GPU byte counter.

acquirePair(width, height, options = {})

Parameters: same as acquireTarget.

Returns: { a, b } — two identical render targets from the same bucket, or null if either acquire fails.

Purpose: convenience for ping-pong passes (blur, reprojection) that swap source and destination every frame.

reserveNamed(name, width, height, options = {})

Parameters:

· name — a unique name.
· width, height — the logical dimensions.
· options.kind — one of RT_KIND.
· options.tag — one of RT_TAG.
· options.samples — MSAA sample count.

Returns: the reserved WebGLRenderTarget, or null.

Purpose: creates a long-lived target that is skipped by releaseAll() unless includeNamed=true. Records the entry in _namedTargets.

getNamed(name)

Parameters: name — the reservation name.

Returns: the target, or null.

getNamedEntry(name)

Parameters: name — the reservation name.

Returns: the full entry object { target, width, height, kind, tag, options }, or null.

resizeNamed(name, width, height, options = {})

Parameters:

· name — the reservation name.
· width, height — the new dimensions.
· options — optional overrides for kind, tag, or samples.

Returns: the resized target, or null.

Purpose: if the requested size matches the current size, returns the existing target. Otherwise releases the old target and reserves a new one. The kind, tag, and MSAA sample count are preserved unless explicitly overridden.

This is the entry point the EngineLoop calls on viewport resize.

releaseNamed(name)

Parameters: name — the reservation name.

Returns: boolean.

beginFrame()

Returns: this. Increments frame.

endFrame()

Returns: this. If autoReleasePerFrame is on, calls releaseAll(false).

releaseAll(includeNamed = false)

Parameters: includeNamed — if true, releases named targets too.

Returns: the number of targets released.

_isNamedSlot(slot)

Internal. Returns true if the slot's target is currently a named reservation.

getStats()

Returns: an object with name, frame, namedCount, the six aggregate counters, gpuBytesInUse, peakGpuBytesInUse, gpuMegabytesInUse, gpuMegabytesPeak, hwCaps, a two-dimensional buckets array, and perfTier.

reset()

Returns: this. Releases everything and zeroes every counter.

dispose()

Returns: this. Resets and calls .dispose() on every WebGLRenderTarget it created.

---

Exported Function — reserveLightingTargets(pool, baseWidth, baseHeight)

Parameters:

· pool — a RenderTargetPool instance.
· baseWidth, baseHeight — the reference viewport dimensions.

Returns: an object mapping each reservation name to its reserved WebGLRenderTarget.

Purpose: reserves the canonical render targets the lighting stack uses. The dimensions are scaled from the base viewport:

· half(w) = max(64, w * 0.5)
· quarter(w) = max(64, w * 0.25)

The reserved targets:

· shadowMapTarget — DEPTH, base size, SHADOW tag.
· shadowAtlasTarget — DEPTH_STENCIL, base size, SHADOW tag.
· giProbeRT — COLOR_RGBA16F, half size, GI tag.
· giBounceRT — COLOR_RGBA16F, half size, GI tag.
· aoFullRT — COLOR_RGBA8, base size, AO tag.
· aoHalfRT — COLOR_RGBA8, half size, AO tag.
· aoQuarterRT — COLOR_RGBA8, quarter size, AO tag.
· aoBlurRT — COLOR_RGBA8, half size, AO tag.
· clusterDebugRT — COLOR_RGBA8, half size, CLUSTER tag.
· envCaptureRT — COLOR_RGBA16F, half size, ENV tag.
· interiorProbeRT — COLOR_RGBA16F, quarter size, INTERIOR tag.
· exteriorProbeRT — COLOR_RGBA16F, half size, EXTERIOR tag.
· sunRayRT — COLOR_RGBA8, half size, SUNRAY tag.
· volumetricRT — COLOR_RGBA16F, half size, VOLUMETRIC tag.
· bloomRT — COLOR_RGBA16F, half size, POST tag.
· postCompositeRT — COMBINED_RGBA8, base size, POST tag.
· postHDRRT — COLOR_RGBA16F, base size, POST tag.
· contactShadowRT — COLOR_RGBA8, half size, CONTACT tag.
· ssgiRT — COLOR_RGBA16F, half size, SSGI tag.
· reSTIRReservoirRT — COLOR_RGBA32F, half size, RESTIR tag.

Twenty targets in total. Every one is created once at boot and reused across the engine's lifetime.

---

Exported Class — TaggedRTSubPool

A façade binding a fixed tag to a shared RenderTargetPool.

Constructor

```
new TaggedRTSubPool(pool, tag)
```

Instance Methods

· acquire(width, height, kind = COLOR_RGBA8, samples = 0) — delegates to pool.acquireTarget with the tag pre-filled.
· acquirePair(width, height, kind = COLOR_RGBA8, samples = 0) — delegates to pool.acquirePair.
· release(target) — delegates.
· reserveNamed(name, width, height, kind = COLOR_RGBA8, samples = 0) — delegates.

---

Exported Functions

getDefaultRenderTargetPool(baseWidth, baseHeight)

Parameters: baseWidth, baseHeight — the reference viewport. Only used on first call to reserve the standard targets.

Returns: the module-level singleton RenderTargetPool.

getDefaultLightingTargets()

Returns: the named-reservation object from the default pool, or null if the pool was not created with base dimensions.

resizeDefaultLightingTargets(baseWidth, baseHeight)

Parameters: baseWidth, baseHeight — the new viewport dimensions.

Returns: the updated reservation object.

Purpose: calls resizeNamed on every standard target. Each resizeNamed releases the old target and reserves a new one only if the size actually changed. Called by the EngineLoop on viewport resize.

disposeDefaultRenderTargetPool()

Returns: nothing.

createRenderTargetPool(options = {})

Returns: a new RenderTargetPool.

createTaggedRTSubPool(pool, tag)

Returns: a new TaggedRTSubPool.

_findBucket(size)

Internal. Linear scan over SIZE_LADDER.

_nextPoolId()

Internal. Returns the next monotonic pool id.

_bytesPerPixel(kind)

Internal. Returns the estimated bytes per pixel for a given kind. RGBA8 = 4, RGB8 = 3, RGBA16F = 8, RGBA32F = 16, DEPTH = 4, DEPTH_STENCIL = 4, COMBINED_RGBA8 = 8, COMBINED_RGBA16F = 12.

---

Default Export

The default export bundles: RenderTargetPool, RTSlot, TaggedRTSubPool, createRenderTargetPool, createTaggedRTSubPool, getDefaultRenderTargetPool, getDefaultLightingTargets, resizeDefaultLightingTargets, disposeDefaultRenderTargetPool, reserveLightingTargets, RT_KIND, RT_KIND_NAME, RT_TAG, RT_TAG_NAME, SIZE_LADDER, LADDER_COUNT, RT_CAPACITY, HW_CAPS.

---

Usage Pattern

A subsystem that wants a scratch render target:

```
import {
  getDefaultRenderTargetPool,
  RT_KIND,
  RT_TAG,
} from './src/core/013_rnd_RenderTargetPool.js';

const pool = getDefaultRenderTargetPool(1920, 1080);

// Per frame:
const rt = pool.acquireTarget(
  window.innerWidth * 0.5,
  window.innerHeight * 0.5,
  { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.GI }
);
renderer.setRenderTarget(rt);
// ... render into rt ...
renderer.setRenderTarget(null);
pool.releaseTarget(rt);
```

A subsystem that wants a persistent target:

```
const shadowAtlas = pool.reserveNamed('shadowAtlasTarget', 2048, 2048, {
  kind: RT_KIND.DEPTH_STENCIL,
  tag: RT_TAG.SHADOW,
});

// Each frame, render into shadowAtlas, no reallocation.
```

On viewport resize, the EngineLoop calls:

```
resizeDefaultLightingTargets(window.innerWidth, window.innerHeight);
```

Every named target is resized only if its ladder rung actually changed. Because the ladder is coarse (64, 96, 128, …), most resizes do not trigger a reallocation at all — they just continue using the same rung. This eliminates the 10–30 ms hit that a naive resize would incur on Android.
