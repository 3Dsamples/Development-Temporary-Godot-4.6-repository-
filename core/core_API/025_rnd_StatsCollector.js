API Documentation — src/core/025_rnd_StatsCollector.js

File Purpose

This file is the unified statistics collector for the anime lighting stack on Android mobile. Where 024_rnd_Profiler.js owns the raw timing ring buffers and the hitch log, this module owns the aggregation surface. It pulls the current values from every subsystem — the pools (010–013), the registry (014), the manifest (015), the config (016), the quality controller (017), the platform profiles (018–020), the capability detector (021), the feature detector (022), the tier resolver (023), and the profiler (024) — and produces one flat, frozen, human-readable snapshot every N frames that a debug HUD, a stats panel, a regression snapshot, or a CI log can consume in a single read.

The relationship between this module and the profiler is precise:

· The profiler owns the numbers. It records every frame, every domain, every scope, every hitch, every memory sample.
· The stats collector owns the assembly. It queries every source once per refresh, writes their current values into a single StatsSnapshot, and lets consumers read that snapshot without touching any other module.

The collector exists because inspecting ten separate modules every time a debug HUD wants to display a value is expensive and error-prone. The HUD would need to import ten modules, know the field names of each, and handle the case where a module is missing. The collector centralizes that knowledge so downstream code reads from one place.

The collector also provides:

1. A refresh cadence — instead of collecting on every frame, it collects on a fixed interval (15 frames on HIGH, 30 on MEDIUM, 45 on LOW). This keeps the cost amortized to less than 0.5 % of frame budget.
2. A source registry — downstream subsystems can register a named callback that the collector invokes on each refresh. Each callback writes into a custom section of the snapshot. This lets subsystems contribute their own stats without modifying the collector.
3. History — a small ring buffer of the last 60 snapshot summaries. A debug HUD can render sparklines from the history without allocating per frame.
4. Exports — exportJSON for regression capture, exportCompact for CI logs, exportCSV for spreadsheet analysis.

The design constraint is that the snapshot object is a single mutable instance updated in place. No allocation happens on the hot path. Every read returns a cached reference. Every section is a plain object with primitive fields.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_SOURCES

Type: number

Value: 32

The maximum number of registered source callbacks. Sized to accommodate every subsystem that will ever want to contribute its own stats section.

HISTORY_SNAPSHOTS

Type: number

Value: 60

The size of the summary history ring buffer. At the default refresh cadence of 30 frames on MEDIUM, 60 snapshots cover about 30 seconds of runtime.

DEFAULT_REFRESH_FRAMES

Type: number

Value: 15 on HIGH, 30 on MEDIUM, 45 on LOW.

The default number of frames between collections. Higher tiers collect more often because the cost is negligible at 60 FPS.

SOURCE_INTERVAL

Type: frozen enum

Values:

· EVERY_REFRESH = 0 — run the source on every collection cycle.
· EVERY_OTHER = 1 — run on every second collection cycle.
· EVERY_QUARTER = 2 — run on every fourth collection cycle.
· MANUAL = 3 — never run automatically; the source must be triggered by calling the callback directly.

The interval gating lets expensive sources run less often without needing their own throttling logic.

---

Module-Level State (Not Exported Directly)

_defaultCollector

Type: StatsCollector | null

The module-level singleton, created on first getDefaultStatsCollector() call.

---

Exported Class — StatsSnapshot

The single-instance snapshot object. Updated in place by the collector on every refresh.

Constructor

```
new StatsSnapshot()
```

Instance Properties

Top-level:

· frame — the current frame number.
· elapsedMs — total elapsed milliseconds.
· timestamp — the wall-clock timestamp at collection.
· collectedAtMs — the high-resolution timestamp at collection.
· collectTimeMs — the cost of the collection itself.
· refreshCount — how many times the snapshot has been refreshed.
· perfTier — the cached PERF_TIER string.

platform section:

· isAndroid, isIOS, isMobile, isDesktop — booleans.
· webglVersion — 1 or 2.
· gpuFamily — the GPU family name string.
· hardwareConcurrency — navigator.hardwareConcurrency.
· deviceMemoryGB — navigator.deviceMemory.
· workerPoolSize — the recommended worker count.
· profileTier — the platform profile tier.
· precision — the recommended shader precision from the Android profile.
· dprCap — the platform DPR cap.
· quirks — the quirks array.

tier section:

· effective — the effective tier name.
· score — the weighted score from the tier resolver.
· confidence — the confidence estimate.
· override — whether a user override is active.
· thermalCap, batteryCap, extraCap — the current bias caps.

quality section:

· level — the current quality level name.
· resolutionScale — the resolved DPR scale.
· shadowMapSize — the resolved shadow map size.
· giUpdateBudgetHz — the resolved GI update rate.
· aoResolutionScale — the resolved AO scale.
· postPassBudget — the resolved post-pass count.
· drawDistance — the resolved fog far distance.
· maxClusterLights — the resolved cluster count.
· lastKnob — the last quality knob adjusted.
· lastReason — the last decision reason.

frameStats section:

· mean, min, max — frame-time statistics in milliseconds.
· p50, p95, p99 — frame-time percentiles.
· fps — the smoothed FPS.
· samples — the number of frames in the ring.

hitchStats section:

· hitches, janks, criticals — the counters.
· totalFrames — the total frame count.
· hitchRate — the fraction of frames that hitched.

domainStats section:

· An array of per-domain stats objects with name, lastMs, emaMs, peakMs, totalMs, callCount.

topScopes section:

· An array of the top ten scopes by EMA cost.

gpu section:

· available, lastMs, averageMs, sampleCount.

memory section:

· heapUsedMB, heapPercent, gpuMB, gpuPercent.
· rtBytesInUse, rtBytesPeak.
· attrBytesInUse, attrBytesPeak.
· registryLive, registryPeak.

pools section:

· object.vector3InUse, vector3Peak, vector3Free.
· object.vector4InUse, vector4Peak, vector4Free.
· object.colorInUse, colorPeak, colorFree.
· object.quaternionInUse, quaternionFree.
· object.matrix4InUse, matrix4Free.
· typed.f32InUse, f32Free, u32InUse, u32Free, u16InUse, u16Free, u8InUse, u8Free, estimatedBytes.
· buffer.attributesAcquired, attributesReleased, gpuBytesInUse, gpuBytesPeak, namedCount.
· renderTarget.acquired, released, namedCount, gpuBytesInUse, gpuBytesPeak, gpuMegabytesInUse, gpuMegabytesPeak.

registry section:

· currentLive, peakLive.
· totalRegistered, totalDisposed, totalZombies, totalLeaks.
· namedCount, ownerCount.

manifest section:

· count, registered, loaded, failed, skipped, loadMs.

lights section (fed by external sources):

· ambient, hemisphere, directional, point, spot, rectArea.
· totalActive, totalShadowCasting, clusterCells.

shadows section (fed by external sources):

· atlasAllocated, atlasUsed, cascades, mapSize, filter, lastUpdateMs.

gi section (fed by external sources):

· probeCount, probeCapacity, updateHz, lastUpdateMs, sampleCount, multiBounce.

ao section (fed by external sources):

· sampleCount, resolutionScale, temporal, lastUpdateMs.

features section:

· A reference to the FEATURES object from 022_rnd_FeatureDetector.js.

custom section:

· A Map from section name to arbitrary data. Populated by registered sources.

---

Exported Class — StatsCollector

The main collector.

Constructor

```
new StatsCollector(options = {})
```

Parameters:

· refreshEveryFrames — the collection interval in frames. Default DEFAULT_REFRESH_FRAMES.
· historyCapacity — the summary history ring size. Default HISTORY_SNAPSHOTS.
· collectCustom — reserved. Default true.
· collectFeatures — whether to include the FEATURES object in the snapshot. Default true.
· collectHistory — whether to push the summary into the history ring on each collection. Default true.

Constructor work:

1. Allocates the snapshot — a single StatsSnapshot instance.
2. Allocates a previous snapshot (used for delta comparisons by future extensions).
3. Initializes historyFrame, historyFps, historyMean, historyP95, historyQuality, historyGpuMB — all typed arrays sized to historyCapacity.
4. Initializes the history ring pointers.
5. Allocates customSections — a Map.
6. Allocates sources — an array of 32 entries.
7. Initializes the frame counter and refresh tracking.
8. Allocates _listeners (Map).

Instance Properties

· options — the merged options.
· snapshot — the mutable StatsSnapshot.
· previous — the previous snapshot.
· hasPrevious — boolean.
· historyCapacity, historyFrame, historyFps, historyMean, historyP95, historyQuality, historyGpuMB — the history ring buffers.
· historyHead, historyCount — the ring pointers.
· customSections — the custom sections Map.
· sources — the source callback array.
· sourceCount — the number of registered sources.
· frame — the frame counter.
· lastRefreshFrame — the frame of the last collection.
· refreshInterval — the current collection interval.

Instance Methods

on(event, fn)

Parameters:

· event — currently only 'collected' is emitted.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to listeners.

registerSource(name, fn, interval)

Parameters:

· name — a unique source name.
· fn — a callback (snapshot) => void that writes into the snapshot.
· interval — one of SOURCE_INTERVAL.

Returns: boolean.

Purpose: registers a named source. The callback is invoked on each collection cycle, filtered by the interval. The callback is responsible for writing its data into snapshot.custom or one of the top-level sections.

unregisterSource(name)

Parameters: name — the source name.

Returns: boolean.

Purpose: removes a registered source.

setSection(name, data)

Parameters:

· name — the section name.
· data — arbitrary data.

Returns: boolean.

Purpose: writes arbitrary data into the custom sections map. Called by source callbacks.

getSection(name)

Parameters: name — the section name.

Returns: the data, or null.

tick(dtMs)

Parameters: dtMs — the frame delta in milliseconds.

Returns: nothing.

Purpose: the per-frame entry point.

Flow:

1. Increments frame.
2. If frame - lastRefreshFrame >= refreshInterval, calls collect(dtMs) and updates lastRefreshFrame.

collect(dtMs)

Parameters: dtMs — unused, reserved.

Returns: the updated snapshot.

Purpose: the collection routine.

Flow:

1. Records the start time.
2. Preserves the previous frame number for delta readers.
3. Updates the top-level fields.
4. Calls _collectPlatform().
5. Calls _collectTier().
6. Calls _collectQuality().
7. Calls _collectProfiler().
8. Calls _collectObjectPool(), _collectTypedPool(), _collectBufferPool(), _collectRenderTargetPool().
9. Calls _collectRegistry().
10. Calls _collectManifest().
11. Calls _collectFeatures() if enabled.
12. Calls _runSources() to invoke every registered source.
13. Calls _pushHistory() if enabled.
14. Records the end time and stores the collection duration in the snapshot.
15. Emits collected.

_collectPlatform(snap)

Copies the current platform values from 018_rnd_PlatformConfig.js into snap.platform.

_collectTier(snap)

Reads the snapshot from 023_rnd_PerfTier.js and copies the tier, score, confidence, override, and caps into snap.tier.

_collectQuality(snap)

Reads the snapshot from 017_rnd_QualityConfig.js and copies every quality field into snap.quality.

_collectProfiler(snap)

Calls profilerStats() from 024_rnd_Profiler.js and copies the frame stats, hitch stats, domain stats, top scopes, memory stats, and GPU stats into the corresponding snapshot sections.

_collectObjectPool(snap)

Reads the LightingPoolSet from 010_rnd_ObjectPool.js and copies the in-use, peak, and free counts for Vector3, Vector4, Color, Quaternion, and Matrix4.

_collectTypedPool(snap)

Reads the TypedArrayPool from 011_rnd_TypedArrayPool.js. Sums the per-bucket in-use and free counts across all buckets for F32, U32, U16, and U8. Copies the total estimated bytes.

_collectBufferPool(snap)

Reads the BufferPool from 012_rnd_BufferPool.js and copies the attribute acquire/release counts, GPU byte counts, and named reservation count.

_collectRenderTargetPool(snap)

Reads the RenderTargetPool from 013_rnd_RenderTargetPool.js and copies the acquire/release counts, named count, GPU byte counts, and megabyte totals.

_collectRegistry(snap)

Reads the ResourceRegistry from 014_rnd_ResourceRegistry.js and copies the live and peak counts, the disposal totals, the leak and zombie counters, and the named and owner counts.

_collectManifest(snap)

Reads the AssetManifest from 015_rnd_AssetManifest.js and copies the count, registered, loaded, failed, skipped, and loadMs fields.

_collectFeatures(snap)

Stores a reference to the FEATURES object from 022_rnd_FeatureDetector.js.

_runSources(snap)

Iterates the registered sources. For each source, checks its interval:

· EVERY_REFRESH — always runs.
· EVERY_OTHER — runs on even refresh counts.
· EVERY_QUARTER — runs on refresh counts divisible by four.
· MANUAL — never runs automatically.

Wraps each source in try/catch and logs failures via console.error.

_pushHistory(snap)

Writes the current snapshot's summary values into the history ring: frame, FPS, mean frame time, p95, quality level index, GPU megabytes.

getSnapshot()

Returns: the current StatsSnapshot.

getSection(name)

Parameters: name — a section name.

Returns: the section contents. If name is omitted, returns the full snapshot.

getHistory()

Returns: an object with capacity, count, head, and the six history arrays.

exportJSON()

Returns: a pretty-printed JSON string of the snapshot. Maps and typed arrays are converted to plain objects and arrays via the replacer.

Purpose: the canonical regression snapshot.

exportCompact()

Returns: an object with only the essential fields: f, fps, mean, p95, p99, max, hitches, janks, crit, quality, tier, gpuMB, rtInUse, attrInUse, regLive, leaks.

Purpose: designed for CI log lines.

exportCSV()

Returns: a comma-separated-values string with one metric per line. The metrics are grouped by section: top, tier, quality, frameStats, hitchStats, pools.renderTarget, pools.buffer, registry.

Purpose: for spreadsheet-based analysis of regression runs.

reset()

Returns: this. Zeroes every counter and clears the history.

dispose()

Returns: nothing. Resets, clears the source array, clears the custom sections map, and clears the listeners.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton.

· statsTick(dtMs) — calls getDefaultStatsCollector().tick(dtMs).
· statsCollect() — calls collect().
· statsSnapshot() — returns getSnapshot().
· statsRegisterSource(name, fn, interval) — registers a source.
· statsUnregisterSource(name) — unregisters a source.

---

Exported Functions

getDefaultStatsCollector()

Returns: the module-level singleton StatsCollector, creating it on first call.

disposeDefaultStatsCollector()

Returns: nothing.

createStatsCollector(options = {})

Returns: a new StatsCollector.

---

Default Export

The default export bundles: StatsCollector, StatsSnapshot, createStatsCollector, getDefaultStatsCollector, disposeDefaultStatsCollector, the five stats* hot-path wrappers, SOURCE_INTERVAL, MAX_SOURCES, HISTORY_SNAPSHOTS, DEFAULT_REFRESH_FRAMES.

---

Usage Pattern

The EngineLoop drives the collector once per frame:

```
import {
  statsTick,
  statsSnapshot,
} from './src/core/025_rnd_StatsCollector.js';

// Per frame:
statsTick(dtMs);

// The debug HUD reads from the snapshot:
const snap = statsSnapshot();
console.log('FPS:', snap.frameStats.fps);
console.log('Quality:', snap.quality.level);
console.log('GPU:', snap.pools.renderTarget.gpuMegabytesInUse, 'MB');
```

A subsystem that wants to contribute its own stats section:

```
import { statsRegisterSource, SOURCE_INTERVAL } from './src/core/025_rnd_StatsCollector.js';

statsRegisterSource('shadowSystem', (snap) => {
  snap.shadows.atlasAllocated = shadowSystem.getAllocatedPageCount();
  snap.shadows.atlasUsed = shadowSystem.getUsedPageCount();
  snap.shadows.cascades = shadowSystem.getCascadeCount();
  snap.shadows.lastUpdateMs = shadowSystem.getLastUpdateMs();
}, SOURCE_INTERVAL.EVERY_REFRESH);
```

A CI run that captures the compact summary:

```
console.log(JSON.stringify(statsSnapshot().exportCompact()));
```

A regression tool that captures the full JSON:

```
fs.writeFileSync('snapshot.json', statsSnapshot().exportJSON());
```

The collector is what makes the debug HUD one-stop-shop. Instead of the HUD importing ten modules and knowing every field name, it reads the snapshot once and gets the entire engine's state in one object. When a subsystem adds a new metric, it registers a source and the HUD picks it up automatically.
