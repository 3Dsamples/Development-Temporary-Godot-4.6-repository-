API Documentation — src/core/022_rnd_FeatureDetector.js

File Purpose

This file is the runtime feature detector for the anime lighting stack. Where 021_rnd_Capabilities.js owns the WebGL and GPU capability surface, this module owns the browser and runtime capability surface: worker parallelism, shared memory, off-thread rendering, sensor APIs, battery and thermal APIs, event listener features, observers, and storage APIs that the parallel lighting pipeline depends on.

The distinction is important. 021 answers "does this GPU support PCSS?" — a rendering concern. 022 answers "can I spawn a worker thread?" or "can I read the battery level?" or "can I share memory between the main thread and a worker?" — a browser runtime concern.

The probe covers five broad areas:

1. Worker parallelism — the Worker, SharedWorker, and ServiceWorker APIs, module workers, SharedArrayBuffer plus Atomics, OffscreenCanvas and its 2D/WebGL/WebGL2 contexts, BroadcastChannel, MessageChannel, and transferable objects. This is the surface that makes the parallel lighting pipeline possible on Android.
2. Asynchronous scheduling — requestAnimationFrame, cancelAnimationFrame, requestIdleCallback, cancelIdleCallback, queueMicrotask, high-resolution performance timers, performance marks and measures, PerformanceObserver, and Long Tasks API.
3. Event features — passive listeners, once listeners, Pointer Events, Touch Events, Gesture Events, Wheel Events, and AbortController. This is what the input handlers on the EngineLoop depend on.
4. Observers — ResizeObserver, IntersectionObserver, MutationObserver, and ReportingObserver.
5. Device APIs relevant to lighting — Battery Status API, Network Information API including saveData flag, Compute Pressure API, Device Orientation, Device Motion, Ambient Light Sensor, and Vibration API.
6. Storage and caching — IndexedDB, caches, StorageManager, persistent storage, storage estimate, and the Origin Private File System.
7. Page-level APIs — Fullscreen, Page Visibility, Screen Orientation, Wake Lock, and Pointer Lock.
8. Advanced runtimes — WebAssembly, WebAssembly SIMD, WebAssembly threads, BigInt64Array, FinalizationRegistry, WeakRef, and Intl.

The probe runs exactly once at module load. Every field is a boolean or a cached number or string. No DOM accesses happen after the initial probe. Every hot-path check is a single boolean read.

The output has three layers, mirroring 021:

1. RAW_FEATURES — the raw probe result. Every field in its native type.
2. FEATURES — the derived boolean decisions the lighting stack actually branches on. canUseWorkers, canUseSharedMemory, canUseOffscreenRT, canUseParallelLighting, and so on.
3. FEATURE_DETECTOR — a frozen object that bundles both with a hasFeature(name) method and exposes the hardware concurrency, device memory, and recommended worker pool size.

There are also convenience helpers (hasFeature, getWorkerPoolSize, getHardwareConcurrency, getDeviceMemoryGB, getStorageEstimate, getNetworkInfo) and safety wrappers (safestWorkerPoolSize, shouldPreferSyncUpdates, shouldUseWakeLock, canUseOffscreenFor, shouldScheduleIdle) that downstream subsystems call instead of re-implementing the logic.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

HAS_WINDOW, HAS_DOCUMENT, HAS_NAVIGATOR, HAS_WORKER_G, HAS_SAB_G, HAS_ATOMICS_G

Internal booleans used to guard every DOM access. Computed once at module load.

WORKER_KIND

Type: frozen enum

Values:

· NONE = 0 — no worker support.
· CLASSIC = 1 — classic worker only.
· MODULE = 2 — module worker available.
· SHARED = 3 — shared worker available.
· SERVICE = 4 — service worker available.

RAW_FEATURES

Type: frozen object

The complete probe result. Every field is a boolean, a number, or a string.

Worker parallelism:

· worker — classic Worker API is available.
· sharedWorker — SharedWorker API is available.
· serviceWorker — navigator.serviceWorker is available.
· moduleWorker — module workers can be constructed (new Worker(url, { type: 'module' })).
· workerPoolRecommended — the recommended number of workers for the current device.
· sharedArrayBuffer — SharedArrayBuffer can be constructed and read back correctly.
· atomics — Atomics operations work on a SharedArrayBuffer.
· offscreenCanvas — OffscreenCanvas can be constructed.
· offscreenCanvas2D — a 2D context can be acquired on an OffscreenCanvas.
· offscreenCanvasWebGL — a WebGL1 context can be acquired.
· offscreenCanvasWebGL2 — a WebGL2 context can be acquired.
· broadcastChannel — BroadcastChannel is available.
· messageChannel — MessageChannel is available.
· transferable — ArrayBuffer exists so transferable posting is possible.

Async scheduling:

· requestAnimationFrame — the rAF API exists.
· cancelAnimationFrame — the cancel API exists.
· requestIdleCallback — the idle callback API exists.
· cancelIdleCallback — the cancel idle API exists.
· queueMicrotask — the microtask queue API exists.
· performanceHighRes — performance.now() exists.
· performanceMark — performance.mark() exists.
· performanceMeasure — performance.measure() exists.
· performanceObserver — PerformanceObserver is available.
· longTaskObserver — the longtask PerformanceObserver entry type is supported.

Event features:

· passiveListeners — passive listener options are honored.
· onceListeners — once listener options are honored (same as passive on modern browsers).
· pointerEvents — PointerEvent API is available.
· touchEvents — TouchEvent API is available.
· gestureEvents — GestureEvent API is available (iOS Safari).
· wheelEvents — WheelEvent API is available.
· abortController — AbortController is available.

Observers:

· resizeObserver — ResizeObserver is available.
· intersectionObserver — IntersectionObserver is available.
· mutationObserver — MutationObserver is available.
· reportingObserver — ReportingObserver is available.

Device APIs:

· battery — navigator.getBattery() is available.
· networkInformation — navigator.connection is available.
· saveData — the saveData flag on the connection is true.
· computePressure — window.computePressure is available.
· deviceOrientation — the DeviceOrientationEvent API is available.
· deviceMotion — the DeviceMotionEvent API is available.
· ambientLightSensor — the AmbientLightSensor API is available.
· vibration — navigator.vibrate() is available.

Storage:

· indexedDB — IndexedDB is available.
· caches — the Cache Storage API is available.
· storageManager — navigator.storage is available.
· persistentStorage — navigator.storage.persist() is available.
· storageEstimate — navigator.storage.estimate() is available.
· originPrivateFs — navigator.storage.getDirectory() is available.

Page APIs:

· fullscreen — the Fullscreen API is available.
· pageVisibility — the Page Visibility API is available.
· screenOrientation — screen.orientation is available.
· wakeLock — navigator.wakeLock is available.
· pointerLock — pointer lock is available.

Advanced runtimes:

· webAssembly — WebAssembly is available.
· webAssemblySIMD — WebAssembly SIMD is available.
· webAssemblyThreads — WebAssembly threads are available (requires SharedArrayBuffer).
· bigInt64Array — BigInt64Array is available.
· finalizationRegistry — FinalizationRegistry is available.
· weakRef — WeakRef is available.
· intl — Intl is available.

Cached values:

· hardwareConcurrency — navigator.hardwareConcurrency or 1.
· deviceMemoryGB — navigator.deviceMemory or 4.
· storageQuotaBytes — populated asynchronously by navigator.storage.estimate().
· storageUsageBytes — populated asynchronously.
· networkEffectiveType — the connection's effectiveType, or 'unknown'.
· networkDownlinkMbps — the connection's downlink.
· networkRttMs — the connection's rtt.
· maxTouchPoints — navigator.maxTouchPoints.

Context isolation:

· hasCoopCoep — true if the page is cross-origin isolated.
· isSecureContext — true if the page is served over HTTPS or localhost.
· isCrossOriginIsolated — true if COOP and COEP headers are set and the page is isolated.

FEATURES

Type: frozen object

The derived feature flags. The full list mirrors RAW_FEATURES but adds the higher-level composite decisions.

Raw mappings (same name as raw):

· hasWorker, hasModuleWorker, hasSharedWorker, hasServiceWorker
· hasSharedArrayBuffer, hasAtomics
· hasOffscreenCanvas, hasOffscreenCanvas2D, hasOffscreenCanvasGL, hasOffscreenCanvasGL2
· hasBroadcastChannel, hasMessageChannel, hasTransferable
· hasRAF, hasIdleCallback, hasMicrotask
· hasPerfHighRes, hasPerfMarks, hasPerfObserver, hasLongTaskObserver
· hasPassiveListeners, hasOnceListeners, hasPointerEvents, hasTouchEvents, hasGestureEvents, hasWheelEvents, hasAbortController
· hasResizeObserver, hasIntersectionObserver, hasMutationObserver, hasReportingObserver
· hasBattery, hasNetworkInformation, hasSaveData, hasComputePressure
· hasDeviceOrientation, hasDeviceMotion, hasAmbientLightSensor, hasVibration
· hasIndexedDB, hasCacheStorage, hasStorageManager, hasPersistentStorage, hasStorageEstimate, hasOPFS
· hasFullscreen, hasPageVisibility, hasScreenOrientation, hasWakeLock, hasPointerLock
· hasWASM, hasWASMSIMD, hasWASMThreads, hasBigInt64Array, hasFinalizationRegistry, hasWeakRef, hasIntl
· isSecureContext, isCrossOriginIsolated, hasCoopCoep

Composite decisions:

· canUseWorkers — a classic Worker can be constructed.
· canUseModuleWorker — a module Worker can be constructed.
· canUseWorkerPool — workers are available AND hardwareConcurrency >= 2.
· canUseSharedMemory — SharedArrayBuffer plus Atomics plus cross-origin isolation.
· canUseAtomics — SAB plus Atomics, regardless of isolation.
· canUseOffscreenRT — OffscreenCanvas plus a WebGL2 context.
· canUseOffscreen2D — OffscreenCanvas plus a 2D context.
· canUseBroadcast — BroadcastChannel is available.
· canUseMessagePort — MessageChannel is available.
· canUseRAF, canUseIdleCallback, canUseMicrotask — direct.
· canUseIdleCompile — IdleCallback plus WebGL async shader compile.
· canUsePerfMarks, canUsePerfObserver, canUseLongTaskWatch — direct.
· canUsePassiveListeners, canUsePointerEvents, canUseTouchEvents, canUseAbortController — direct.
· canUseResizeObserver, canUseIntersectionObserver, canUseMutationObserver — direct.
· canUseBatteryGuard — hasBattery.
· canUseThermalGuard — hasComputePressure.
· canUseNetworkGuard — hasNetworkInformation.
· canUseSaveDataGuard — Network Information plus saveData.
· canUseOrientation, canUseMotion, canUseAmbientLight, canUseVibration — direct.
· canUseIndexedDB, canUseCacheStorage, canEstimateStorage, canRequestPersist, canUseOPFS — direct.
· canUseFullscreen, canUsePageVisibility, canUseScreenOrient, canUseWakeLock, canUsePointerLock — direct.
· canUseWASM, canUseWASMSIMD, canUseWASMThreads, canUseBigInt64, canUseFinalization, canUseWeakRef — direct.

Top-level composite decisions used by the lighting stack:

· canUseParallelLighting — worker pool plus shared memory.
· canUseAsyncShadowAtlas — worker pool plus shared memory.
· canUseAsyncProbeBake — worker pool plus shared memory plus perf marks.
· canUseAsyncAOBlur — worker pool plus shared memory.
· canUseOffscreenShadow — OffscreenCanvas WebGL2.
· canUseOffscreenPost — OffscreenCanvas WebGL2 plus shared memory.
· canUseZeroCopyTransfer — shared memory.
· canUsePooledWorkerMessaging — MessageChannel plus Worker.
· canUseAdaptiveUnderSaveData — saveData guard.
· canUseAdaptiveUnderThermal — thermal guard.
· canUseAdaptiveUnderBattery — battery guard.
· canUseFullLightingGuards — thermal guard OR battery guard.
· canUseWakeLockDuringBake — WakeLock API.

FEATURE_DETECTOR

Type: frozen object

The canonical feature-detection surface.

Fields:

· name — always 'feature_detector'.
· platform — a frozen sub-object with isAndroid, isIOS, isMobile, isDesktop, perfTier, gpuFamily.
· raw — the RAW_FEATURES object.
· features — the FEATURES object.
· hardwareConcurrency — the cached navigator.hardwareConcurrency.
· deviceMemoryGB — the cached navigator.deviceMemory.
· workerPoolSize — the recommended worker count.
· hasFeature(name) — a method that reads FEATURES[name] === true.

---

Internal State (Not Exported Directly)

_raw

Type: mutable object

The working object the probe fills in. Once _probe() completes, its values are frozen into RAW_FEATURES.

_probeWorkers()

Probes every worker-related API.

Flow:

1. If the global Worker constructor exists, sets worker = true. Then tries to construct a module worker from a small Blob URL to test module worker support. Terminates the worker and revokes the URL immediately.
2. Checks for SharedWorker and serviceWorker constructors.
3. If SharedArrayBuffer exists, tries to construct a small buffer, wrap it in an Int32Array, and confirm Atomics.store then Atomics.load round-trip correctly. Both sharedArrayBuffer and atomics are set based on the result.
4. If OffscreenCanvas exists, tries constructing one and acquiring each of the three context types ('2d', 'webgl', 'webgl2').
5. Checks for BroadcastChannel and MessageChannel constructors.
6. Sets transferable if ArrayBuffer exists.
7. Reads navigator.hardwareConcurrency and navigator.deviceMemory.
8. Computes workerPoolRecommended based on the tier.

_probeAsync()

Probes every async scheduling API. Checks window.requestAnimationFrame, cancelAnimationFrame, requestIdleCallback, cancelIdleCallback, the global queueMicrotask, performance.now, performance.mark, performance.measure, PerformanceObserver, and the longtask entry type.

The longtask probe is done by reading PerformanceObserver.supportedEntryTypes and searching for the 'longtask' string.

_probeEvents()

Probes every event feature.

Flow:

1. Detects passive listener support by using Object.defineProperty on a getter for the passive option. Registers a dummy listener with the probe options and checks whether the getter fired.
2. Sets onceListeners to the same result as passive.
3. Checks for PointerEvent, TouchEvent, GestureEvent, and WheelEvent constructors on window.
4. Checks for AbortController.
5. Reads navigator.maxTouchPoints.

_probeObservers()

Probes every observer class. Simple typeof checks for ResizeObserver, IntersectionObserver, MutationObserver, ReportingObserver.

_probeDeviceApis()

Probes every device API.

Flow:

1. Checks navigator.getBattery.
2. Checks navigator.connection and reads effectiveType, downlink, rtt, saveData.
3. Checks window.computePressure.
4. Checks window.DeviceOrientationEvent, window.DeviceMotionEvent, window.AmbientLightSensor.
5. Checks navigator.vibrate.

_probeStorage()

Probes every storage API.

Flow:

1. Checks for window.indexedDB.
2. Checks for the global caches.
3. Checks navigator.storage and its persist, estimate, and getDirectory methods.

_probePage()

Probes page-level APIs.

Flow:

1. Checks document.fullscreenEnabled (and the WebKit and Mozilla prefixes).
2. Checks typeof document.hidden === 'boolean'.
3. Checks window.screen.orientation.
4. Checks 'wakeLock' in navigator.
5. Checks 'pointerLockElement' in document.

_probeAdvanced()

Probes advanced runtimes.

Flow:

1. Checks for WebAssembly. If present, probes 'Simd' in WebAssembly or 'simd' in WebAssembly for SIMD support. Sets webAssemblyThreads based on sharedArrayBuffer.
2. Checks for BigInt64Array.
3. Checks for FinalizationRegistry.
4. Checks for WeakRef.
5. Checks for Intl.

_probeContextIsolation()

Reads window.isSecureContext and window.crossOriginIsolated. Sets hasCoopCoep based on cross-origin isolation.

_probeAsyncStorageEstimate()

If navigator.storage.estimate is available, calls it and writes quota and usage into _raw.storageQuotaBytes and _raw.storageUsageBytes when the Promise resolves. This is the only async part of the probe, and it does not block module initialization.

_probe()

The orchestrator. Calls each of the nine probe sub-functions in sequence.

_computeFeatureFlags()

Reads RAW_FEATURES, WEBGL_FEATURES, and the platform profile. Returns the FEATURES object.

---

Exported Functions

hasFeature(name)

Parameters: name — the feature name.

Returns: FEATURES[name] === true.

Purpose: the fast query. Every subsystem that needs a feature check calls this.

getWorkerPoolSize()

Returns: RAW_FEATURES.workerPoolRecommended.

Purpose: the worker pool size for this device. On LOW tier with 4 cores, this is 2. On HIGH tier with 8 cores, this is 6.

getHardwareConcurrency()

Returns: RAW_FEATURES.hardwareConcurrency.

getDeviceMemoryGB()

Returns: RAW_FEATURES.deviceMemoryGB.

getStorageEstimate()

Returns: an object with quotaBytes, usageBytes, quotaMB, usageMB. The last two are the first two divided by 1024*1024.

getNetworkInfo()

Returns: an object with effectiveType, downlinkMbps, rttMs, saveData.

safestWorkerPoolSize()

Returns: the worker pool size if workers are available, otherwise 1.

Purpose: the safe worker count for the lighting stack.

shouldPreferSyncUpdates()

Returns: boolean.

Purpose: the single gate that parallel pipelines check before offloading. Returns true if:

· Workers are unavailable, OR
· The worker pool is unavailable, OR
· The saveData flag is set, OR
· hardwareConcurrency <= 2.

When this returns true, the lighting stack runs synchronously instead of offloading to workers. This is the fallback that keeps the engine correct on low-end devices and data-saver connections.

shouldUseWakeLock()

Returns: true if WakeLock is available and the device is mobile.

Purpose: whether long lighting bakes should request a wake lock to prevent the screen from sleeping mid-bake.

canUseOffscreenFor(purpose)

Parameters: purpose — one of 'shadow' | 'post' | 'gi' | 'ao' | 'env'.

Returns: boolean.

Purpose: reports whether the OffscreenCanvas path is safe for the given purpose.

· 'shadow' requires canUseOffscreenShadow.
· 'post' requires canUseOffscreenPost.
· 'gi' requires canUseOffscreenRT and canUseSharedMemory.
· 'ao' requires canUseOffscreenRT and canUseSharedMemory.
· 'env' requires canUseOffscreenRT.

shouldScheduleIdle(taskCostMs)

Parameters: taskCostMs — the estimated cost of the task in milliseconds.

Returns: boolean.

Purpose: reports whether the given task should be deferred via requestIdleCallback.

· Returns false if IdleCallback is unavailable.
· Returns true if workers are unavailable, since there is no other way to offload.
· Returns true if the task cost exceeds 4 ms.
· Returns false otherwise.

getFeatureReport()

Returns: an object with platform, hardwareConcurrency, deviceMemoryGB, workerPoolSize, isSecureContext, isCrossOriginIsolated, raw, features, networkInfo, storageInfo.

Purpose: the human-readable summary for the debug HUD or CI log.

---

Default Export

The default export bundles: FEATURE_DETECTOR, RAW_FEATURES, FEATURES, WORKER_KIND, hasFeature, getWorkerPoolSize, getHardwareConcurrency, getDeviceMemoryGB, getStorageEstimate, getNetworkInfo, safestWorkerPoolSize, shouldPreferSyncUpdates, shouldUseWakeLock, canUseOffscreenFor, shouldScheduleIdle, getFeatureReport.

---

Usage Pattern

A subsystem that wants to know whether to offload to a worker:

```
import {
  hasFeature,
  safestWorkerPoolSize,
  shouldPreferSyncUpdates,
} from './src/core/022_rnd_FeatureDetector.js';

if (!shouldPreferSyncUpdates() && hasFeature('canUseAsyncShadowAtlas')) {
  const workerCount = safestWorkerPoolSize();
  const pool = new WorkerPool(workerCount);
  shadowAtlasBuilder.setWorkerPool(pool);
} else {
  shadowAtlasBuilder.setSyncMode(true);
}
```

A subsystem that wants to use OffscreenCanvas for post-processing:

```
if (hasFeature('canUseOffscreenPost')) {
  const rt = new OffscreenCanvas(width, height);
  const gl = rt.getContext('webgl2');
  postProcessor.attachOffscreen(rt, gl);
} else {
  postProcessor.useMainThread();
}
```

The App's battery guard that wants to know whether it can install itself:

```
if (hasFeature('canUseBatteryGuard')) {
  navigator.getBattery().then((battery) => {
    battery.addEventListener('levelchange', onBatteryChange);
    battery.addEventListener('chargingchange', onBatteryChange);
  });
}
```

Because every subsystem reads the same FEATURES object, the engine makes a single coherent decision about whether to use workers, shared memory, OffscreenCanvas, or any other runtime feature. On a low-end Android device without cross-origin isolation, canUseSharedMemory is false and the entire parallel path is disabled — but the engine still runs correctly with the synchronous fallback.

