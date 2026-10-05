API Documentation — src/core/024_rnd_Profiler.js

File Purpose

This file is the runtime profiler and performance telemetry hub for the anime lighting stack on Android mobile. Where 017_rnd_QualityConfig.js decides WHAT quality level to run at, this module provides the raw evidence: minimum, maximum, mean, p50, p95, and p99 frame time; per-domain cost profile; per-scope cost profile; hitch histogram; jank count; and memory pressure trend. The quality controller reads from here. Nothing writes back through this module.

The profiler maintains five independent telemetry surfaces:

1. Frame-time history — a fixed ring buffer of the last N frame times, sized to 240 / 480 / 960 frames depending on PERF_TIER. Percentile computation runs on the sorted slice when the caller requests it, so the cost is paid only when the data is needed.
2. Per-domain cost profile — one bucket per FrameScheduler domain (SIMULATION, LIGHTS, SHADOWS, GI, AO, ENVIRONMENT, INTERIOR, EXTERIOR, DIRECTOR, POST). Each bucket records last, EMA, peak, and total cost, plus call count.
3. Named scope timers — a registry of named scopes with beginScope('giProbeBake') and endScope(). Nesting is LIFO with a fixed depth of 32. Every scope name resolves to a fixed slot index once, so the hot path only touches typed arrays.
4. Hitch log — any frame over hitchMs (default 33.3 ms) is recorded in a ring buffer with a category tag (frame, domain, scope, memory, gpu) so the offending subsystem is identifiable.
5. Memory tracker — reads performance.memory (Chrome) for the JavaScript heap, tracks GPU bytes fed by the pools (010–013) and the registry (014), and reports heap and GPU percentages against the platform's configured caps.

Optionally, the profiler also attaches to a WebGL renderer and uses EXT_disjoint_timer_query (WebGL2) or EXT_disjoint_timer_query_webgl2 to record GPU time per frame. Most Android browsers expose the extension; when they do not, the profiler silently falls back to CPU-only telemetry.

The design constraint is zero per-frame allocations on the hot path. Every buffer is pre-allocated at construction. Every read returns cached views. Percentile computation reuses a single sort buffer. Nothing in the module allocates after the first frame.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

HISTORY_CAPACITY

Type: number

Value: 960 on HIGH, 480 on MEDIUM, 240 on LOW.

The ring buffer size for frame-time history. On HIGH, a 960-frame history covers 16 seconds at 60 FPS or 32 seconds at 30 FPS. This is long enough to capture biome transitions, thermal ramps, and chunk unload bursts.

MAX_SCOPES

Type: number

Value: 64

The maximum number of registered scope timers. Sized to accommodate every named profiling region the lighting stack creates: shadowAtlasPack, giProbeBake, aoBlur, clusterGridBuild, envPaletteSolve, interiorVolUpdate, exteriorProbeSolve, postPrepare, directorHint, plus up to 54 more for debugging.

MAX_SCOPE_DEPTH

Type: number

Value: 32

The maximum nesting depth for scopes. Deeper nesting is ignored rather than throwing.

MAX_HITCHES

Type: number

Value: 64

The ring buffer size for the hitch log.

MAX_MARKERS

Type: number

Value: 128

The ring buffer size for the marker log.

HITCH_MS_DEFAULT

Type: number

Value: 33.3

The hitch threshold in milliseconds. Matches the 30 FPS budget. Any frame longer than this is a hitch.

JANK_MS_DEFAULT

Type: number

Value: 50.0

The jank threshold. Any frame over this is a visible stall.

CRITICAL_MS_DEFAULT

Type: number

Value: 100.0

The critical threshold. Any frame over this is a dropped frame.

EMA_ALPHA_FRAME

Type: number

Value: 0.10

The frame-time EMA smoothing factor.

EMA_ALPHA_DOMAIN

Type: number

Value: 0.12

The per-domain cost EMA smoothing factor.

EMA_ALPHA_SCOPE

Type: number

Value: 0.15

The per-scope cost EMA smoothing factor.

HITCH_CATEGORY

Type: frozen enum

Values:

· NONE = 0
· FRAME = 1
· DOMAIN = 2
· SCOPE = 3
· MEMORY = 4
· GPU = 5
· COUNT = 6

HITCH_CATEGORY_NAME

Type: frozen array

Values: ['none', 'frame', 'domain', 'scope', 'memory', 'gpu'].

SAMPLE_KIND

Type: frozen enum

Values:

· FRAME = 0
· DOMAIN = 1
· SCOPE = 2
· GPU = 3
· COUNT = 4

---

Module-Level State (Not Exported Directly)

_defaultProfiler

Type: Profiler | null

The module-level singleton.

---

Internal Helper Functions (Documented)

_now()

Returns: the current high-resolution timestamp via performance.now(), or Date.now() as a fallback.

_percentileFromSorted(sorted, len, p)

Parameters:

· sorted — a sorted Float32Array slice.
· len — the number of valid entries.
· p — the percentile in [0, 1].

Returns: the value at that percentile.

Purpose: internal percentile lookup. Uses index interpolation. Handles the boundary cases (p <= 0, p >= 1) without allocating.

---

Internal Class — GpuTimer

Attaches to a WebGL renderer and records GPU time via EXT_disjoint_timer_query. Gracefully degrades when the extension is unavailable.

Constructor

```
new GpuTimer()
```

Instance Properties

· available — boolean, whether the extension was acquired.
· gl — the WebGL context, or null.
· ext — the timer query extension, or null.
· pendingQueries — the number of in-flight queries.
· lastGpuMs — the GPU time of the last recorded frame.
· samples — a Float32Array(32) of recent GPU times.
· sampleCount — the number of valid entries in samples.
· sampleHead — the ring buffer write pointer.

Instance Methods

attach(renderer)

Parameters: renderer — a THREE.WebGLRenderer.

Returns: boolean.

Purpose: tries to acquire EXT_disjoint_timer_query_webgl2 (WebGL2) then EXT_disjoint_timer_query (WebGL1). If either succeeds, sets available = true and stores the references. Never throws.

detach()

Returns: nothing. Clears the WebGL references and sets available = false.

recordGpuMs(ms)

Parameters: ms — the GPU time in milliseconds.

Returns: nothing.

Purpose: writes the sample into the ring buffer and updates lastGpuMs. Ignores non-finite or negative values.

getAverageGpuMs()

Returns: the mean of the recorded samples, or 0 if no samples have been recorded.

---

Internal Class — ScopeRegistry

A fixed-capacity registry of named scope timers with a LIFO stack for nesting.

Constructor

```
new ScopeRegistry(capacity)
```

Parameters: capacity — the maximum number of scopes.

Instance Properties

· capacity — the registry size.
· name — an array of scope names, indexed by slot.
· nameIndex — a Map from name to slot.
· lastMs — a Float32Array(capacity) of last-call costs.
· emaMs — a Float32Array(capacity) of EMA costs.
· peakMs — a Float32Array(capacity) of peak costs.
· totalMs — a Float32Array(capacity) of total accumulated costs.
· callCount — a Uint32Array(capacity) of call counts.
· count — the number of registered scopes.
· stackSlot — an Int32Array(MAX_SCOPE_DEPTH) of slot indices.
· stackStart — a Float64Array(MAX_SCOPE_DEPTH) of start timestamps.
· stackDepth — the current nesting depth.

Instance Methods

register(name)

Parameters: name — a scope name string.

Returns: the slot index, or -1 if the registry is full or the name is invalid.

Purpose: registers a scope name to a fixed slot. Idempotent — registering the same name twice returns the existing slot.

begin(slot)

Parameters: slot — the slot index from register.

Returns: boolean.

Purpose: pushes the slot and the current timestamp onto the LIFO stack. Silently ignores requests when the stack is at maximum depth.

end()

Returns: the slot index, or -1 if the stack is empty.

Purpose: pops the stack, computes the elapsed milliseconds, and updates the slot's last, EMA, peak, total, and call count.

reset()

Returns: nothing. Zeroes every counter.

---

Internal Class — HitchLog

A ring buffer of recent hitch events.

Constructor

```
new HitchLog(capacity)
```

Instance Properties

· capacity — the ring size.
· frame — a Uint32Array of frame numbers.
· costMs — a Float32Array of costs.
· category — a Uint8Array of HITCH_CATEGORY values.
· slot — an Int16Array of scope or domain indices, or -1.
· head — the ring write pointer.
· count — the number of valid entries.
· total — the cumulative number of hitches ever recorded.

Instance Methods

record(frame, costMs, category, slot)

Returns: nothing. Writes to the ring and advances the head.

clear()

Returns: nothing. Zeroes every array and resets the head.

copyRecent(max, outFrame, outCost, outCategory, outSlot)

Parameters:

· max — the maximum number of recent entries to copy.
· outFrame, outCost, outCategory, outSlot — the caller-provided arrays to write into.

Returns: the number of entries copied.

Purpose: copies the most recent entries into caller-owned buffers, in chronological order.

---

Internal Class — MarkerLog

A ring buffer of named markers. Markers are lightweight annotations that downstream tools use to correlate a hitch with the code that caused it.

Constructor

```
new MarkerLog(capacity)
```

Instance Properties

· capacity — the ring size.
· frame — a Uint32Array.
· timeMs — a Float32Array.
· labelId — an Int16Array.
· head — the ring write pointer.
· count — the number of valid entries.
· labels — an array of label strings.
· labelIndex — a Map from label to id.
· labelCount — the number of registered labels.

Instance Methods

registerLabel(label)

Parameters: label — a string.

Returns: the label id, or -1 if the label table is full.

mark(frame, labelId)

Parameters:

· frame — the current frame number.
· labelId — the id from registerLabel.

Returns: nothing.

clear()

Returns: nothing.

---

Internal Class — MemoryTracker

Reads JavaScript heap usage via performance.memory and receives GPU byte counts from the pools.

Constructor

```
new MemoryTracker()
```

Instance Properties

· available — boolean, whether performance.memory is present.
· heapUsedBytes, heapTotalBytes, heapLimitBytes, externalBytes — the last-read heap values.
· gpuBytesAllocated, gpuBytesInUse, gpuBytesPeak — aggregate GPU byte counters.
· rtBytesAllocated, rtBytesInUse, rtBytesPeak — render target byte counters.
· attrBytesAllocated, attrBytesInUse, attrBytesPeak — buffer attribute byte counters.
· registryLiveCount, registryPeakCount — resource registry counters.
· lastUpdateFrame — the last frame the heap was polled.
· updateIntervalFrames — how often to poll performance.memory (default 30 frames, because reading it has non-trivial cost on Android).

Instance Methods

refresh(frame)

Parameters: frame — the current frame number.

Returns: nothing.

Purpose: reads performance.memory.usedJSHeapSize, totalJSHeapSize, and jsHeapSizeLimit if available.

reportGpuFromPool(name, bytesInUse, bytesPeak)

Parameters:

· name — one of 'render_target', 'attribute', 'object'.
· bytesInUse — the current in-use bytes.
· bytesPeak — the peak bytes.

Returns: nothing.

Purpose: receives byte counts from the pools and the registry. Aggregates them into gpuBytesInUse and gpuBytesPeak.

getHeapMB()

Returns: the heap used in megabytes.

getGpuMB()

Returns: the GPU bytes in megabytes.

getHeapPercent()

Returns: heapUsedBytes / heapLimitBytes, clamped to [0, 1].

getGpuPercent()

Returns: gpuBytesInUse / (ANDROID_PROFILE.maxGpuMemoryMB * 1048576), clamped to [0, 1].

---

Exported Class — Profiler

The main profiler.

Constructor

```
new Profiler(options = {})
```

Parameters:

· historyCapacity — the frame-time ring buffer size. Default HISTORY_CAPACITY.
· hitchMs — the hitch threshold. Default HITCH_MS_DEFAULT.
· jankMs — the jank threshold. Default JANK_MS_DEFAULT.
· criticalMs — the critical threshold. Default CRITICAL_MS_DEFAULT.
· enableGpu — if true, attach the GPU timer when a renderer is bound. Default true.
· enableMemory — if true, poll performance.memory. Default true.
· autoLog — if true, log a summary every 60 frames. Default false.

Constructor work:

1. Allocates the frame ring buffers: frameIndex (Uint32Array), frameDtMs (Float32Array), frameDomains (Float32Array sized capacity * DOMAIN.COUNT), frameScopes (Float32Array sized capacity * MAX_SCOPES), frameGpuMs (Float32Array).
2. Initializes the ring head and count.
3. Initializes the frame and elapsed counters.
4. Allocates domainLastMs, domainEmaMs, domainPeakMs, domainTotalMs (each a Float32Array(DOMAIN.COUNT)), and domainCalls (a Uint32Array(DOMAIN.COUNT)).
5. Constructs the ScopeRegistry, HitchLog, MarkerLog, MemoryTracker, and GpuTimer sub-objects.
6. Allocates _sortBuf — a Float32Array(capacity) reused for percentile computation.
7. Allocates _frameStats — a reusable stats object.
8. Allocates _listeners (Map).

Instance Properties

· options — the merged options.
· capacity — the ring buffer size.
· frameIndex, frameDtMs, frameDomains, frameScopes, frameGpuMs — the frame ring buffers.
· head, frameCount — the ring write pointer and current count.
· frame, elapsedMs — the monotonic frame and elapsed-millisecond counters.
· dtLastMs, dtEmaMs, dtMinMs, dtMaxMs, dtSumMs — frame-time statistics.
· hitches, janks, criticals — the hitch counters.
· domainLastMs, domainEmaMs, domainPeakMs, domainTotalMs, domainCalls — per-domain arrays.
· scopes — the ScopeRegistry.
· hitchLog — the HitchLog.
· markerLog — the MarkerLog.
· memory — the MemoryTracker.
· gpu — the GpuTimer.
· lastFrameCategory — the HITCH_CATEGORY of the last frame.

Instance Methods

on(event, fn)

Parameters:

· event — currently only 'endframe' is emitted by the profiler's own helpers.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to listeners.

attachRenderer(renderer)

Parameters: renderer — a THREE.WebGLRenderer.

Returns: nothing.

Purpose: stores the renderer and, if enableGpu is on, tries to attach the GPU timer extension.

detachRenderer()

Returns: nothing. Detaches the GPU timer.

beginFrame(dtMs)

Parameters: dtMs — the last frame's delta time in milliseconds.

Returns: this.

Purpose: the per-frame entry point.

Flow:

1. Increments frame.
2. Stores dtLastMs, updates elapsedMs.
3. Updates dtEmaMs, dtMinMs, dtMaxMs, dtSumMs.
4. Categorizes the frame: critical if >= criticalMs, jank if >= jankMs, hitch if >= hitchMs, else none.
5. Records a hitch entry if the category is not none.
6. Writes into the frame ring buffers.
7. Advances the ring head and count.
8. Polls performance.memory on a throttled schedule if enableMemory.
9. Calls _autoLogFrame() on a 60-frame cadence if autoLog.

recordDomain(domain, costMs)

Parameters:

· domain — a DOMAIN enum value.
· costMs — the cost in milliseconds.

Returns: nothing.

Purpose: updates the per-domain last, EMA, peak, total, and call count. Records a domain hitch if the cost exceeds hitchMs.

registerScope(name)

Parameters: name — a scope name string.

Returns: the slot index.

Purpose: convenience wrapper around scopes.register(name).

beginScope(nameOrSlot)

Parameters: nameOrSlot — a scope name string or slot index.

Returns: boolean.

Purpose: pushes the scope onto the LIFO stack. Resolves the name to a slot if a string is given.

endScope()

Returns: the slot index of the scope that ended, or -1.

Purpose: pops the stack, updates the slot's stats, and records a scope hitch if the cost exceeds hitchMs.

registerMarker(label)

Parameters: label — a string.

Returns: the label id.

mark(nameOrLabelId)

Parameters: nameOrLabelId — a label string or id.

Returns: boolean.

Purpose: records a marker at the current frame. Markers are lightweight annotations that downstream tools use to correlate a hitch with the code that caused it.

recordGpuMs(ms)

Parameters: ms — the GPU time in milliseconds.

Returns: nothing.

Purpose: passes the value to gpu.recordGpuMs(ms) and records a GPU hitch if the cost exceeds hitchMs.

reportMemory(name, inUse, peak)

Parameters:

· name — one of 'render_target', 'attribute', 'object'.
· inUse — the current in-use bytes.
· peak — the peak bytes.

Returns: nothing.

Purpose: receives byte counts from pools and registry.

computeFrameStats()

Returns: the cached _frameStats object with frame, samples, mean, min, max, p50, p95, p99, and fps.

Purpose: iterates the frame-time ring, sorts a subarray of _sortBuf, and computes percentiles. The result is written into the reusable _frameStats object, so no allocation occurs after the first call.

_autoLogFrame()

Internal. Logs a summary line via console.log.

getFrameStats()

Returns: the result of computeFrameStats().

getDomainStats()

Returns: an array of per-domain stats objects, each with name, lastMs, emaMs, peakMs, totalMs, callCount.

getScopeStats()

Returns: an array of per-scope stats objects with the same shape.

getTopScopes(max)

Parameters: max — the number of top scopes to return.

Returns: an array of the N scopes with the highest EMA cost, sorted descending.

Purpose: a convenience for debug HUDs and CI logs.

getHitchStats()

Returns: an object with hitches, janks, criticals, totalFrames, hitchRate (the fraction of frames that were hitches, janks, or criticals), recentCount, totalLogged.

getMemoryStats()

Returns: an object with available, heapUsedMB, heapPercent, gpuMB, gpuPercent, rtBytesInUse, rtBytesPeak, attrBytesInUse, attrBytesPeak, registryLive, registryPeak.

getGpuStats()

Returns: an object with available, lastMs, averageMs, sampleCount.

getStats()

Returns: a comprehensive stats object with version, frame, elapsedMs, dtEmaMs, dtMinMs, dtMaxMs, dtSumMs, historyCapacity, historyCount, frameStats, hitchStats, domainStats, scopeStats, memoryStats, gpuStats, and a platform sub-object with perfTier, platformTier, isAndroid, isMobile, gpuVendor, webgl2, quirks.

Purpose: the canonical snapshot for the debug HUD or CI log. Do not call per frame — it allocates.

exportJSON()

Returns: a pretty-printed JSON string of getStats().

exportCompact()

Returns: a compact object with only the essential fields: f, fps, mean, p95, p99, max, h, j, c.

Purpose: designed for CI log lines that need to be short.

reset()

Returns: this. Zeroes every buffer and counter.

dispose()

Returns: nothing. Resets, detaches the renderer, and clears listeners.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton. They exist so downstream code does not need to hold a reference to the profiler.

· profilerBeginFrame(dtMs) — calls getDefaultProfiler().beginFrame(dtMs).
· profilerRecordDomain(domain, ms) — calls recordDomain.
· profilerBeginScope(nameOrSlot) — calls beginScope.
· profilerEndScope() — calls endScope.
· profilerMark(nameOrLabelId) — calls mark.
· profilerRecordGpuMs(ms) — calls recordGpuMs.
· profilerReportMemory(name, inUse, peak) — calls reportMemory.
· profilerStats() — returns getStats().

---

Exported Functions

getDefaultProfiler()

Returns: the module-level singleton Profiler, creating it on first call.

disposeDefaultProfiler()

Returns: nothing.

createProfiler(options = {})

Returns: a new Profiler.

---

Default Export

The default export bundles: Profiler, ScopeRegistry, HitchLog, MarkerLog, MemoryTracker, GpuTimer, createProfiler, getDefaultProfiler, disposeDefaultProfiler, the eight profiler* hot-path wrappers, HITCH_CATEGORY, HITCH_CATEGORY_NAME, SAMPLE_KIND, HISTORY_CAPACITY, MAX_SCOPES, MAX_SCOPE_DEPTH, MAX_HITCHES, MAX_MARKERS.

---

Usage Pattern

The EngineLoop drives the profiler once per frame:

```
import {
  getDefaultProfiler,
  profilerBeginFrame,
  profilerRecordDomain,
  profilerBeginScope,
  profilerEndScope,
  profilerStats,
} from './src/core/024_rnd_Profiler.js';

const profiler = getDefaultProfiler();
profiler.attachRenderer(renderer);

// Per frame, at the top of tick():
profilerBeginFrame(dtMs);

// Per domain, inside the FrameScheduler callback:
profilerRecordDomain(DOMAIN.SHADOWS, shadowCostMs);
profilerRecordDomain(DOMAIN.GI, giCostMs);

// Per scope, for finer granularity:
profilerBeginScope('giProbeBake');
// ... do the probe bake ...
profilerEndScope();
```

A debug HUD that displays the top five costs:

```
const stats = profilerStats();
for (const scope of stats.scopeStats.sort((a, b) => b.emaMs - a.emaMs).slice(0, 5)) {
  console.log(`${scope.name}: ${scope.emaMs.toFixed(2)} ms`);
}
```

A CI log that captures the compact summary:

```
console.log(JSON.stringify(profiler.exportCompact()));
```

The pools and registry feed byte counts once per second:

```
import { getDefaultRenderTargetPool } from './src/core/013_rnd_RenderTargetPool.js';
const pool = getDefaultRenderTargetPool();
setInterval(() => {
  const stats = pool.getStats();
  profiler.reportMemory('render_target', stats.gpuBytesInUse, stats.peakGpuBytesInUse);
}, 1000);
```

The profiler is what makes the adaptive quality controller possible. Without accurate per-frame, per-domain, and per-scope timing, the controller cannot distinguish between "the shadow atlas repack is slow" and "the GI probe bake is slow" — it would have to react to the aggregate frame time and downgrade everything uniformly. With the profiler's granular data, the controller downgrades exactly the subsystem that is over budget, preserving the visual output of every other subsystem.
