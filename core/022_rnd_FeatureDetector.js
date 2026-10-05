// File : 022
// name : src/core/022_rnd_FeatureDetector.js
// description : Runtime feature detector for the anime lighting stack on
//               Android mobile. Where 021_rnd_Capabilities.js owns the
//               WEBGL/GPU capability surface, THIS module owns the
//               BROWSER/RUNTIME capability surface: worker parallelism,
//               shared memory, off-thread rendering, sensor APIs, battery
//               and thermal APIs, event listener features, observers, and
//               storage APIs that the parallel lighting pipeline depends on.
//
//               Probed features:
//                 • Worker parallelism
//                     – Worker, SharedWorker, ServiceWorker
//                     – module workers (new Worker(..., { type: 'module' }))
//                     – SharedArrayBuffer + Atomics (needs COOP/COEP)
//                     – OffscreenCanvas (main-thread + worker transfer)
//                     – OffscreenCanvas 2D / WebGL / WebGL2 contexts
//                     – BroadcastChannel
//                     – MessageChannel / MessagePort transferables
//                 • Asynchronous scheduling
//                     – requestAnimationFrame, requestIdleCallback
//                     – queueMicrotask, setImmediate (Node-ish)
//                     – performance.now() high-res, performance.mark
//                     – PerformanceObserver, Long Tasks API
//                 • Event features
//                     – passive listeners, once listeners
//                     – Pointer Events (unified mouse+touch)
//                     – Touch Events, Gesture Events (iOS)
//                     – Wheel Events with passive option
//                     – AbortController / AbortSignal (listener removal)
//                 • Observers
//                     – ResizeObserver, IntersectionObserver
//                     – MutationObserver, ReportingObserver
//                 • Device APIs relevant to lighting
//                     – Battery Status API (level, charging, dischargingTime)
//                     – Network Information API (effectiveType, saveData)
//                     – Compute Pressure API (thermal state, if present)
//                     – Device Orientation / Motion (for exterior light tilt)
//                     – Ambient Light Sensor (if exposed)
//                     – Vibration API (optional haptic feedback)
//                 • Storage / caching
//                     – IndexedDB, caches, StorageManager.estimate()
//                     – persistent storage
//                     – navigator.storage.getDirectory (OPFS)
//                 • Page APIs
//                     – Fullscreen API
//                     – Page Visibility API
//                     – Screen Orientation API
//                     – Wake Lock API (prevent screen sleep during long GI bakes)
//                 • Advanced
//                     – WebAssembly (SIMD, threads)
//                     – BigInt64Array / BigUint64Array
//                     – FinalizationRegistry / WeakRef
//                     – Intl APIs
//
//               All results frozen; every query helper O(1) and allocation-
//               free. Provides `computeFeatureFlags()` that maps the raw
//               probe to the boolean decisions the lighting stack branches
//               on (canUseWorkerPool, canUseSharedMemory, canUseOffscreenRT,
//               canUseBatteryGuard, canUseThermalGuard, canUseIdleCompile,
//               canUseWakeLock, etc.).
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external detection libs; every probe runs once at
//               module load; every read on the hot path is a boolean check.
// best for : Giving every lighting subsystem a single authoritative answer
//            to "can I use worker threads?", "can I share memory with the
//            worker?", "can I render offscreen?", "can I read the battery
//            level?", "can I detect thermal state?", "can I schedule idle
//            compile work?" — without ever touching the API directly.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  QUIRK,
} from './018_rnd_PlatformConfig.js';

import {
  ANDROID_PROFILE,
  GPU_FAMILY,
  GPU_FAMILY_NAME,
} from './019_rnd_AndroidProfile.js';

import {
  CAPABILITIES,
  FEATURES as WEBGL_FEATURES,
} from './021_rnd_Capabilities.js';

import {
  getPerfTier,
} from './008_scn_world.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

const HAS_WINDOW     = (typeof window !== 'undefined');
const HAS_DOCUMENT   = (typeof document !== 'undefined');
const HAS_NAVIGATOR  = (typeof navigator !== 'undefined');
const HAS_WORKER_G   = (typeof Worker !== 'undefined');
const HAS_SAB_G      = (typeof SharedArrayBuffer !== 'undefined');
const HAS_ATOMICS_G  = (typeof Atomics !== 'undefined');

export const WORKER_KIND = Object.freeze({
  NONE:    0,
  CLASSIC: 1,
  MODULE:  2,
  SHARED:  3,
  SERVICE: 4,
});

/* ------------------------------------------------------------------ */
/* 1. RAW PROBE (one-shot at module load)                             */
/* ------------------------------------------------------------------ */

const _raw = {
  // Workers
  worker:                false,
  sharedWorker:          false,
  serviceWorker:         false,
  moduleWorker:          false,
  workerPoolRecommended: 0,
  sharedArrayBuffer:     false,
  atomics:               false,
  offscreenCanvas:       false,
  offscreenCanvas2D:     false,
  offscreenCanvasWebGL:  false,
  offscreenCanvasWebGL2: false,
  broadcastChannel:      false,
  messageChannel:        false,
  transferable:          false,

  // Async scheduling
  requestAnimationFrame: false,
  cancelAnimationFrame:  false,
  requestIdleCallback:   false,
  cancelIdleCallback:    false,
  queueMicrotask:        false,
  performanceHighRes:    false,
  performanceMark:       false,
  performanceMeasure:    false,
  performanceObserver:   false,
  longTaskObserver:      false,

  // Events
  passiveListeners:      false,
  onceListeners:         false,
  pointerEvents:         false,
  touchEvents:           false,
  gestureEvents:         false,
  wheelEvents:           false,
  abortController:       false,

  // Observers
  resizeObserver:        false,
  intersectionObserver:  false,
  mutationObserver:      false,
  reportingObserver:     false,

  // Device APIs (may resolve asynchronously)
  battery:               false,
  networkInformation:    false,
  saveData:              false,
  computePressure:       false,
  deviceOrientation:     false,
  deviceMotion:          false,
  ambientLightSensor:    false,
  vibration:             false,

  // Storage
  indexedDB:             false,
  caches:                false,
  storageManager:        false,
  persistentStorage:     false,
  storageEstimate:       false,
  originPrivateFs:       false,

  // Page APIs
  fullscreen:            false,
  pageVisibility:        false,
  screenOrientation:     false,
  wakeLock:              false,
  pointerLock:           false,

  // Advanced
  webAssembly:           false,
  webAssemblySIMD:       false,
  webAssemblyThreads:    false,
  bigInt64Array:         false,
  finalizationRegistry:  false,
  weakRef:               false,
  intl:                  false,

  // Specific capability values
  hardwareConcurrency:   1,
  deviceMemoryGB:        4,
  storageQuotaBytes:     0,
  storageUsageBytes:     0,
  networkEffectiveType:  'unknown',
  networkDownlinkMbps:   0,
  networkRttMs:          0,
  maxTouchPoints:        0,

  // Browser hints
  hasCoopCoep:           false,
  isSecureContext:       false,
  isCrossOriginIsolated: false,
};

function _probeWorkers() {
  if (HAS_WORKER_G) {
    _raw.worker = true;

    // Module worker probe (construction only; not started).
    try {
      const blob = new Blob(['export default {};'], { type: 'text/javascript' });
      const url = URL.createObjectURL(blob);
      const w = new Worker(url, { type: 'module' });
      w.terminate();
      URL.revokeObjectURL(url);
      _raw.moduleWorker = true;
    } catch (_) {
      // Some Android browsers reject module workers — silently ignore.
    }
  }

  if (typeof SharedWorker !== 'undefined') _raw.sharedWorker = true;
  if (HAS_NAVIGATOR && 'serviceWorker' in navigator) _raw.serviceWorker = true;

  // SharedArrayBuffer: only available when the page is cross-origin
  // isolated (COOP+COEP headers).
  if (HAS_SAB_G) {
    try {
      // Try to actually construct one — some environments expose the
      // constructor but throw on use.
      const sab = new SharedArrayBuffer(4);
      const view = new Int32Array(sab);
      if (HAS_ATOMICS_G) {
        Atomics.store(view, 0, 1);
        if (Atomics.load(view, 0) === 1) {
          _raw.sharedArrayBuffer = true;
          _raw.atomics = true;
        }
      } else {
        _raw.sharedArrayBuffer = true;
      }
    } catch (_) {
      _raw.sharedArrayBuffer = false;
      _raw.atomics = false;
    }
  }

  if (typeof OffscreenCanvas !== 'undefined') {
    _raw.offscreenCanvas = true;
    try {
      const c = new OffscreenCanvas(1, 1);
      if (c.getContext('2d'))          _raw.offscreenCanvas2D = true;
      if (c.getContext('webgl'))       _raw.offscreenCanvasWebGL = true;
      if (c.getContext('webgl2'))      _raw.offscreenCanvasWebGL2 = true;
    } catch (_) {
      // Leave sub-features as false.
    }
  }

  if (typeof BroadcastChannel !== 'undefined') _raw.broadcastChannel = true;
  if (typeof MessageChannel !== 'undefined')   _raw.messageChannel = true;

  // Transferable objects probe: Transferable is a TS interface, but
  // ArrayBuffer + postMessage transfer is the runtime feature. We only
  // need to confirm ArrayBuffer exists; Three.js uses transferables via
  // typed arrays which are always supported on modern browsers.
  if (typeof ArrayBuffer !== 'undefined') _raw.transferable = true;

  _raw.hardwareConcurrency = HAS_NAVIGATOR ? (navigator.hardwareConcurrency || 1) : 1;
  _raw.deviceMemoryGB = HAS_NAVIGATOR ? (navigator.deviceMemory || 4) : 4;

  // Recommended worker pool size (Android tuning from 019).
  if (_raw.worker) {
    const reserve = PERF_TIER_LOCAL === 'HIGH' ? 1 : 2;
    const cap = PERF_TIER_LOCAL === 'HIGH' ? 6 : PERF_TIER_LOCAL === 'MEDIUM' ? 4 : 2;
    _raw.workerPoolRecommended = Math.max(1, Math.min(cap, _raw.hardwareConcurrency - reserve));
  }
}

function _probeAsync() {
  if (HAS_WINDOW) {
    if (typeof window.requestAnimationFrame === 'function') _raw.requestAnimationFrame = true;
    if (typeof window.cancelAnimationFrame  === 'function') _raw.cancelAnimationFrame  = true;
    if (typeof window.requestIdleCallback   === 'function') _raw.requestIdleCallback   = true;
    if (typeof window.cancelIdleCallback    === 'function') _raw.cancelIdleCallback    = true;
  }
  if (typeof queueMicrotask === 'function') _raw.queueMicrotask = true;

  if (typeof performance !== 'undefined') {
    if (typeof performance.now === 'function')   _raw.performanceHighRes = true;
    if (typeof performance.mark === 'function')  _raw.performanceMark = true;
    if (typeof performance.measure === 'function') _raw.performanceMeasure = true;
    if (typeof PerformanceObserver !== 'undefined') {
      _raw.performanceObserver = true;
      try {
        // Try to construct a long-task observer.
        const supported = PerformanceObserver.supportedEntryTypes || [];
        if (supported.indexOf('longtask') >= 0) _raw.longTaskObserver = true;
      } catch (_) { /* swallow */ }
    }
  }
}

function _probeEvents() {
  if (!HAS_WINDOW) return;

  // Passive listener support detection.
  try {
    let supported = false;
    const opts = Object.defineProperty({}, 'passive', {
      get() { supported = true; return false; },
    });
    const noop = () => {};
    window.addEventListener('__feature_probe__', noop, opts);
    window.removeEventListener('__feature_probe__', noop, opts);
    _raw.passiveListeners = supported;
  } catch (_) {
    _raw.passiveListeners = false;
  }

  // Once listeners (supported by every browser that supports passive).
  _raw.onceListeners = _raw.passiveListeners;

  // Pointer Events (unified input).
  if ('PointerEvent' in window)    _raw.pointerEvents = true;
  if ('TouchEvent' in window)      _raw.touchEvents = true;
  if ('GestureEvent' in window)    _raw.gestureEvents = true;
  if ('WheelEvent' in window)      _raw.wheelEvents = true;
  if (typeof AbortController !== 'undefined') _raw.abortController = true;

  if (HAS_NAVIGATOR && typeof navigator.maxTouchPoints === 'number') {
    _raw.maxTouchPoints = navigator.maxTouchPoints;
  }
}

function _probeObservers() {
  if (typeof ResizeObserver !== 'undefined')       _raw.resizeObserver = true;
  if (typeof IntersectionObserver !== 'undefined') _raw.intersectionObserver = true;
  if (typeof MutationObserver !== 'undefined')     _raw.mutationObserver = true;
  if (typeof ReportingObserver !== 'undefined')    _raw.reportingObserver = true;
}

function _probeDeviceApis() {
  if (!HAS_NAVIGATOR) return;

  if (typeof navigator.getBattery === 'function') _raw.battery = true;

  if ('connection' in navigator && navigator.connection) {
    _raw.networkInformation = true;
    const c = navigator.connection;
    _raw.networkEffectiveType = c.effectiveType || 'unknown';
    _raw.networkDownlinkMbps  = c.downlink || 0;
    _raw.networkRttMs         = c.rtt || 0;
    _raw.saveData             = c.saveData === true;
  }

  if (HAS_WINDOW && typeof window.computePressure === 'function') _raw.computePressure = true;

  if (HAS_WINDOW) {
    if (typeof window.DeviceOrientationEvent !== 'undefined') _raw.deviceOrientation = true;
    if (typeof window.DeviceMotionEvent !== 'undefined')      _raw.deviceMotion = true;
    if (typeof window.AmbientLightSensor !== 'undefined')     _raw.ambientLightSensor = true;
  }

  if (HAS_NAVIGATOR && typeof navigator.vibrate === 'function') _raw.vibration = true;
}

function _probeStorage() {
  if (HAS_WINDOW && 'indexedDB' in window) _raw.indexedDB = true;
  if (typeof caches !== 'undefined')       _raw.caches = true;

  if (HAS_NAVIGATOR && navigator.storage) {
    _raw.storageManager = true;
    if (typeof navigator.storage.persist === 'function') {
      _raw.persistentStorage = true;
    }
    if (typeof navigator.storage.estimate === 'function') {
      _raw.storageEstimate = true;
    }
    if (typeof navigator.storage.getDirectory === 'function') {
      _raw.originPrivateFs = true;
    }
  }
}

function _probePage() {
  if (HAS_DOCUMENT) {
    if (document.fullscreenEnabled !== undefined ||
        document.webkitFullscreenEnabled !== undefined ||
        document.mozFullScreenEnabled !== undefined) {
      _raw.fullscreen = true;
    }
    if (typeof document.hidden === 'boolean') _raw.pageVisibility = true;
  }

  if (HAS_WINDOW && typeof window.screen !== 'undefined' && 'orientation' in window.screen) {
    _raw.screenOrientation = true;
  }

  if (HAS_NAVIGATOR && 'wakeLock' in navigator) _raw.wakeLock = true;
  if (HAS_DOCUMENT && 'pointerLockElement' in document) _raw.pointerLock = true;
}

function _probeAdvanced() {
  if (typeof WebAssembly !== 'undefined') {
    _raw.webAssembly = true;
    // SIMD probe: check for v128 ops by inspecting the compiler.
    try {
      const simdTest = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // \0asm
        0x01, 0x00, 0x00, 0x00, // version 1
      ]);
      // We can't easily compile a SIMD module without a full binary.
      // Feature detection via `WebAssembly.validate` on a minimal SIMD module
      // is more reliable but adds binary payload; we conservatively check
      // for the WebAssembly.Simd namespace, which some engines expose.
      _raw.webAssemblySIMD = 'Simd' in WebAssembly || 'simd' in WebAssembly;
    } catch (_) {
      _raw.webAssemblySIMD = false;
    }
    _raw.webAssemblyThreads = _raw.sharedArrayBuffer;
  }

  if (typeof BigInt64Array !== 'undefined') _raw.bigInt64Array = true;
  if (typeof FinalizationRegistry !== 'undefined') _raw.finalizationRegistry = true;
  if (typeof WeakRef !== 'undefined') _raw.weakRef = true;
  if (typeof Intl !== 'undefined') _raw.intl = true;
}

function _probeContextIsolation() {
  if (typeof window !== 'undefined') {
    if ('isSecureContext' in window) {
      _raw.isSecureContext = window.isSecureContext === true;
    }
    if ('crossOriginIsolated' in window) {
      _raw.isCrossOriginIsolated = window.crossOriginIsolated === true;
    }
    // COOP/COEP are required for SharedArrayBuffer + high-res timers.
    _raw.hasCoopCoep = _raw.isCrossOriginIsolated;
  }
}

function _probeAsyncStorageEstimate() {
  // Storage estimate is async; we warm the value without blocking.
  if (!_raw.storageEstimate) return;
  try {
    navigator.storage.estimate().then((est) => {
      _raw.storageQuotaBytes = est.quota || 0;
      _raw.storageUsageBytes = est.usage || 0;
    }).catch(() => { /* swallow */ });
  } catch (_) { /* swallow */ }
}

function _probe() {
  _probeWorkers();
  _probeAsync();
  _probeEvents();
  _probeObservers();
  _probeDeviceApis();
  _probeStorage();
  _probePage();
  _probeAdvanced();
  _probeContextIsolation();
  _probeAsyncStorageEstimate();
}

_probe();

/* ------------------------------------------------------------------ */
/* 2. FROZEN RAW CAPS EXPORT                                          */
/* ------------------------------------------------------------------ */

export const RAW_FEATURES = Object.freeze(Object.assign({}, _raw));

/* ------------------------------------------------------------------ */
/* 3. FEATURE FLAGS (derived decisions the lighting stack branches on) */
/* ------------------------------------------------------------------ */

function _computeFeatureFlags() {
  const r = RAW_FEATURES;

  // ---- Worker parallelism ----
  const canUseWorkers       = r.worker;
  const canUseModuleWorker  = r.worker && r.moduleWorker;
  const canUseWorkerPool    = r.worker && r.hardwareConcurrency >= 2;
  const canUseSharedMemory  = r.sharedArrayBuffer && r.atomics && r.isCrossOriginIsolated;
  const canUseAtomics       = r.sharedArrayBuffer && r.atomics;
  const canUseOffscreenRT   = r.offscreenCanvas && r.offscreenCanvasWebGL2;
  const canUseOffscreen2D   = r.offscreenCanvas && r.offscreenCanvas2D;
  const canUseBroadcast     = r.broadcastChannel;
  const canUseMessagePort   = r.messageChannel;

  // ---- Async scheduling ----
  const canUseRAF            = r.requestAnimationFrame;
  const canUseIdleCallback   = r.requestIdleCallback;
  const canUseIdleCompile    = r.requestIdleCallback && WEBGL_FEATURES.supportsAsyncCompile;
  const canUseMicrotask      = r.queueMicrotask;
  const canUsePerfMarks      = r.performanceMark && r.performanceMeasure;
  const canUsePerfObserver   = r.performanceObserver;
  const canUseLongTaskWatch  = r.longTaskObserver;

  // ---- Events ----
  const canUsePassiveListeners = r.passiveListeners;
  const canUsePointerEvents    = r.pointerEvents;
  const canUseTouchEvents      = r.touchEvents;
  const canUseAbortController  = r.abortController;

  // ---- Observers ----
  const canUseResizeObserver       = r.resizeObserver;
  const canUseIntersectionObserver = r.intersectionObserver;
  const canUseMutationObserver     = r.mutationObserver;

  // ---- Device APIs (lighting-relevant) ----
  const canUseBatteryGuard   = r.battery;
  const canUseThermalGuard   = r.computePressure;
  const canUseNetworkGuard   = r.networkInformation;
  const canUseSaveDataGuard  = r.networkInformation && r.saveData;
  const canUseOrientation    = r.deviceOrientation;
  const canUseMotion         = r.deviceMotion;
  const canUseAmbientLight   = r.ambientLightSensor;
  const canUseVibration      = r.vibration;

  // ---- Storage ----
  const canUseIndexedDB      = r.indexedDB;
  const canUseCacheStorage   = r.caches;
  const canEstimateStorage   = r.storageEstimate;
  const canRequestPersist    = r.persistentStorage;
  const canUseOPFS           = r.originPrivateFs;

  // ---- Page APIs ----
  const canUseFullscreen     = r.fullscreen;
  const canUsePageVisibility = r.pageVisibility;
  const canUseScreenOrient   = r.screenOrientation;
  const canUseWakeLock       = r.wakeLock;
  const canUsePointerLock    = r.pointerLock;

  // ---- Advanced ----
  const canUseWASM           = r.webAssembly;
  const canUseWASMSIMD       = r.webAssembly && r.webAssemblySIMD;
  const canUseWASMThreads    = r.webAssembly && r.sharedArrayBuffer && r.isCrossOriginIsolated;
  const canUseBigInt64       = r.bigInt64Array;
  const canUseFinalization   = r.finalizationRegistry;
  const canUseWeakRef        = r.weakRef;

  // ---- Composite decisions used by the lighting stack ----
  const canUseParallelLighting  = canUseWorkerPool && canUseSharedMemory;
  const canUseAsyncShadowAtlas  = canUseWorkerPool && canUseSharedMemory;
  const canUseAsyncProbeBake    = canUseWorkerPool && canUseSharedMemory && canUsePerfMarks;
  const canUseAsyncAOBlur       = canUseWorkerPool && canUseSharedMemory;
  const canUseOffscreenShadow   = canUseOffscreenRT;
  const canUseOffscreenPost     = canUseOffscreenRT && canUseSharedMemory;
  const canUseZeroCopyTransfer  = canUseSharedMemory;
  const canUsePooledWorkerMessaging = canUseMessagePort && canUseWorkers;
  const canUseAdaptiveUnderSaveData = canUseSaveDataGuard;
  const canUseAdaptiveUnderThermal  = canUseThermalGuard;
  const canUseAdaptiveUnderBattery  = canUseBatteryGuard;
  const canUseFullLightingGuards    = canUseThermalGuard || canUseBatteryGuard;
  const canUseWakeLockDuringBake    = canUseWakeLock;

  return Object.freeze({
    // ---- Raw booleans ----
    hasWorker:              r.worker,
    hasModuleWorker:        r.moduleWorker,
    hasSharedWorker:        r.sharedWorker,
    hasServiceWorker:       r.serviceWorker,
    hasSharedArrayBuffer:   r.sharedArrayBuffer,
    hasAtomics:             r.atomics,
    hasOffscreenCanvas:     r.offscreenCanvas,
    hasOffscreenCanvas2D:   r.offscreenCanvas2D,
    hasOffscreenCanvasGL:   r.offscreenCanvasWebGL,
    hasOffscreenCanvasGL2:  r.offscreenCanvasWebGL2,
    hasBroadcastChannel:    r.broadcastChannel,
    hasMessageChannel:      r.messageChannel,
    hasTransferable:        r.transferable,

    hasRAF:                 r.requestAnimationFrame,
    hasIdleCallback:        r.requestIdleCallback,
    hasMicrotask:           r.queueMicrotask,
    hasPerfHighRes:         r.performanceHighRes,
    hasPerfMarks:           r.performanceMark && r.performanceMeasure,
    hasPerfObserver:        r.performanceObserver,
    hasLongTaskObserver:    r.longTaskObserver,

    hasPassiveListeners:    r.passiveListeners,
    hasOnceListeners:       r.onceListeners,
    hasPointerEvents:       r.pointerEvents,
    hasTouchEvents:         r.touchEvents,
    hasGestureEvents:       r.gestureEvents,
    hasWheelEvents:         r.wheelEvents,
    hasAbortController:     r.abortController,

    hasResizeObserver:      r.resizeObserver,
    hasIntersectionObserver:r.intersectionObserver,
    hasMutationObserver:    r.mutationObserver,
    hasReportingObserver:   r.reportingObserver,

    hasBattery:             r.battery,
    hasNetworkInformation:  r.networkInformation,
    hasSaveData:            r.saveData,
    hasComputePressure:     r.computePressure,
    hasDeviceOrientation:   r.deviceOrientation,
    hasDeviceMotion:        r.deviceMotion,
    hasAmbientLightSensor:  r.ambientLightSensor,
    hasVibration:           r.vibration,

    hasIndexedDB:           r.indexedDB,
    hasCacheStorage:        r.caches,
    hasStorageManager:      r.storageManager,
    hasPersistentStorage:   r.persistentStorage,
    hasStorageEstimate:     r.storageEstimate,
    hasOPFS:                r.originPrivateFs,

    hasFullscreen:          r.fullscreen,
    hasPageVisibility:      r.pageVisibility,
    hasScreenOrientation:   r.screenOrientation,
    hasWakeLock:            r.wakeLock,
    hasPointerLock:         r.pointerLock,

    hasWASM:                r.webAssembly,
    hasWASMSIMD:            r.webAssemblySIMD,
    hasWASMThreads:         r.webAssemblyThreads,
    hasBigInt64Array:       r.bigInt64Array,
    hasFinalizationRegistry:r.finalizationRegistry,
    hasWeakRef:             r.weakRef,
    hasIntl:                r.intl,

    isSecureContext:        r.isSecureContext,
    isCrossOriginIsolated:  r.isCrossOriginIsolated,
    hasCoopCoep:            r.hasCoopCoep,

    // ---- High-level lighting decisions ----
    canUseWorkers,
    canUseModuleWorker,
    canUseWorkerPool,
    canUseSharedMemory,
    canUseAtomics,
    canUseOffscreenRT,
    canUseOffscreen2D,
    canUseBroadcast,
    canUseMessagePort,

    canUseRAF,
    canUseIdleCallback,
    canUseIdleCompile,
    canUseMicrotask,
    canUsePerfMarks,
    canUsePerfObserver,
    canUseLongTaskWatch,

    canUsePassiveListeners,
    canUsePointerEvents,
    canUseTouchEvents,
    canUseAbortController,

    canUseResizeObserver,
    canUseIntersectionObserver,
    canUseMutationObserver,

    canUseBatteryGuard,
    canUseThermalGuard,
    canUseNetworkGuard,
    canUseSaveDataGuard,
    canUseOrientation,
    canUseMotion,
    canUseAmbientLight,
    canUseVibration,

    canUseIndexedDB,
    canUseCacheStorage,
    canEstimateStorage,
    canRequestPersist,
    canUseOPFS,

    canUseFullscreen,
    canUsePageVisibility,
    canUseScreenOrient,
    canUseWakeLock,
    canUsePointerLock,

    canUseWASM,
    canUseWASMSIMD,
    canUseWASMThreads,
    canUseBigInt64,
    canUseFinalization,
    canUseWeakRef,

    // ---- Composite decisions ----
    canUseParallelLighting,
    canUseAsyncShadowAtlas,
    canUseAsyncProbeBake,
    canUseAsyncAOBlur,
    canUseOffscreenShadow,
    canUseOffscreenPost,
    canUseZeroCopyTransfer,
    canUsePooledWorkerMessaging,
    canUseAdaptiveUnderSaveData,
    canUseAdaptiveUnderThermal,
    canUseAdaptiveUnderBattery,
    canUseFullLightingGuards,
    canUseWakeLockDuringBake,
  });
}

export const FEATURES = _computeFeatureFlags();

/* ------------------------------------------------------------------ */
/* 4. CANONICAL FEATURE DETECTOR OBJECT (read-only surface)           */
/* ------------------------------------------------------------------ */

export const FEATURE_DETECTOR = Object.freeze({
  name:       'feature_detector',
  platform:   Object.freeze({
    isAndroid:  DEVICE.isAndroid,
    isIOS:      DEVICE.isIOS,
    isMobile:   DEVICE.isMobile,
    isDesktop:  DEVICE.isDesktop,
    perfTier:   PERF_TIER_LOCAL,
    gpuFamily:  GPU_FAMILY_NAME,
  }),

  raw:        RAW_FEATURES,
  features:   FEATURES,

  hardwareConcurrency: RAW_FEATURES.hardwareConcurrency,
  deviceMemoryGB:      RAW_FEATURES.deviceMemoryGB,
  workerPoolSize:      RAW_FEATURES.workerPoolRecommended,

  hasFeature(name) {
    return FEATURES[name] === true;
  },
});

/* ------------------------------------------------------------------ */
/* 5. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

export function hasFeature(name) {
  return FEATURES[name] === true;
}

export function getWorkerPoolSize() {
  return RAW_FEATURES.workerPoolRecommended;
}

export function getHardwareConcurrency() {
  return RAW_FEATURES.hardwareConcurrency;
}

export function getDeviceMemoryGB() {
  return RAW_FEATURES.deviceMemoryGB;
}

export function getStorageEstimate() {
  return {
    quotaBytes: RAW_FEATURES.storageQuotaBytes,
    usageBytes: RAW_FEATURES.storageUsageBytes,
    quotaMB:    RAW_FEATURES.storageQuotaBytes / (1024 * 1024),
    usageMB:    RAW_FEATURES.storageUsageBytes / (1024 * 1024),
  };
}

export function getNetworkInfo() {
  return {
    effectiveType: RAW_FEATURES.networkEffectiveType,
    downlinkMbps:  RAW_FEATURES.networkDownlinkMbps,
    rttMs:         RAW_FEATURES.networkRttMs,
    saveData:      RAW_FEATURES.saveData,
  };
}

/* ------------------------------------------------------------------ */
/* 6. SAFETY WRAPPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Returns a worker pool size appropriate for the current device. Falls
 * back to 1 (main-thread only) if workers are unavailable.
 */
export function safestWorkerPoolSize() {
  return FEATURES.canUseWorkers ? Math.max(1, RAW_FEATURES.workerPoolRecommended) : 1;
}

/**
 * Returns true when the lighting stack should prefer synchronous updates
 * over async/parallel ones. This is the single gate that parallel
 * pipelines check before offloading.
 */
export function shouldPreferSyncUpdates() {
  if (!FEATURES.canUseWorkers)         return true;
  if (!FEATURES.canUseWorkerPool)      return true;
  if (FEATURES.hasSaveData)            return true;
  if (RAW_FEATURES.hardwareConcurrency <= 2) return true;
  return false;
}

/**
 * Returns true when a wake lock can and should be used during long
 * lighting bakes (GI probe rebuild, shadow atlas full repack).
 */
export function shouldUseWakeLock() {
  return FEATURES.canUseWakeLock && DEVICE.isMobile;
}

/**
 * Returns true when the OffscreenCanvas path is safe to enable for the
 * given purpose.
 */
export function canUseOffscreenFor(purpose) {
  switch (purpose) {
    case 'shadow':    return FEATURES.canUseOffscreenShadow;
    case 'post':      return FEATURES.canUseOffscreenPost;
    case 'gi':        return FEATURES.canUseOffscreenRT && FEATURES.canUseSharedMemory;
    case 'ao':        return FEATURES.canUseOffscreenRT && FEATURES.canUseSharedMemory;
    case 'env':       return FEATURES.canUseOffscreenRT;
    default:          return FEATURES.canUseOffscreenRT;
  }
}

/**
 * Returns true when the given async task should be scheduled via
 * requestIdleCallback instead of immediately.
 */
export function shouldScheduleIdle(taskCostMs) {
  if (!FEATURES.canUseIdleCallback) return false;
  if (!FEATURES.canUseWorkers)      return true; // no worker → defer to idle
  if (taskCostMs > 4)               return true;
  return false;
}

/* ------------------------------------------------------------------ */
/* 7. DIAGNOSTICS                                                     */
/* ------------------------------------------------------------------ */

export function getFeatureReport() {
  return {
    platform: Object.freeze({
      isAndroid:  DEVICE.isAndroid,
      isIOS:      DEVICE.isIOS,
      isMobile:   DEVICE.isMobile,
      isDesktop:  DEVICE.isDesktop,
      perfTier:   PERF_TIER_LOCAL,
    }),
    hardwareConcurrency: RAW_FEATURES.hardwareConcurrency,
    deviceMemoryGB:      RAW_FEATURES.deviceMemoryGB,
    workerPoolSize:      RAW_FEATURES.workerPoolRecommended,
    isSecureContext:     RAW_FEATURES.isSecureContext,
    isCrossOriginIsolated: RAW_FEATURES.isCrossOriginIsolated,
    raw:                 RAW_FEATURES,
    features:            FEATURES,
    networkInfo:         getNetworkInfo(),
    storageInfo:         getStorageEstimate(),
  };
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  FEATURE_DETECTOR,
  RAW_FEATURES,
  FEATURES,
  WORKER_KIND,

  hasFeature,
  getWorkerPoolSize,
  getHardwareConcurrency,
  getDeviceMemoryGB,
  getStorageEstimate,
  getNetworkInfo,

  safestWorkerPoolSize,
  shouldPreferSyncUpdates,
  shouldUseWakeLock,
  canUseOffscreenFor,
  shouldScheduleIdle,

  getFeatureReport,
};

export default _defaultExport;