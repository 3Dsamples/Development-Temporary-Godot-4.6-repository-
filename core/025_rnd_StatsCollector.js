// File : 025
// name : src/core/025_rnd_StatsCollector.js
// description : Unified statistics collector for the anime lighting stack on
//               Android mobile. Where 024_rnd_Profiler.js owns the raw
//               timing ring buffers and hitch log, THIS module owns the
//               AGGREGATION surface: it pulls the current values from every
//               pool (010–013), the registry (014), the manifest (015), the
//               config (016), the quality controller (017), the platform
//               profiles (018–020), the capability detector (021), the
//               feature detector (022), the tier resolver (023), and the
//               profiler (024), then produces ONE flat, frozen, human-
//               readable snapshot every N frames that a debug HUD, a stats
//               panel, a regression snapshot, or a CI log can consume in a
//               single read.
//
//               Responsibilities:
//                 • Own the canonical StatsSnapshot object (single instance,
//                   updated in place — zero per-frame allocations).
//                 • Own the collection cadence: refresh at a caller-chosen
//                   interval (default every 15 frames on HIGH, 30 on
//                   MEDIUM, 45 on LOW) so the snapshot stays cheap on
//                   Android.
//                 • Own the subscriptions: register/unregister named
//                   "source" callbacks that the collector invokes each
//                   refresh; each source writes into a pre-allocated
//                   sub-section of the snapshot. This decouples the
//                   collector from the concrete pool/registry/profiler
//                   APIs and lets downstream systems add their own stats
//                   sections without touching this file.
//                 • Own the history: a small ring of the last N snapshots
//                   (lightweight summary only) so a debug HUD can render
//                   sparklines without allocating per frame.
//                 • Own the export: `exportJSON()` for regression capture,
//                   `exportCompact()` for CI logs, `exportCSV()` for
//                   spreadsheet analysis.
//                 • Own the API filter: `getSection('pools')`,
//                   `getSection('lights')`, `getSection('shadows')`, etc.
//                   returns a cached reference into the snapshot — no
//                   copying, no allocation.
//
//               Design:
//                 • Fixed-capacity source registry (MAX_SOURCES).
//                 • Fixed-capacity snapshot history (HISTORY_SNAPSHOTS).
//                 • Zero per-frame allocations on the hot path.
//                 • Never blocks: every source is expected to be O(1) or
//                   O(n) with n = number of items it owns. Collectors run
//                   at a throttled cadence, not per-frame.
//                 • Deterministic ordering: sources are invoked in
//                   registration order so exported snapshots are stable
//                   across runs — critical for regression diffing.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external stats libs; every internal buffer sized
//               once at construction.
// best for : Giving every debug and regression surface one canonical
//            read-only stats object. Debug HUD, stats panel, GI viewer,
//            shadow viewer, cluster viewer, probe viewer, memory profiler,
//            performance audit, and CI screenshot comparison all read the
//            same numbers from the same source.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  DOMAIN,
  DOMAIN_NAME,
} from './005_rnd_FrameScheduler.js';

import {
  getDefaultPoolSet,
  POOL_KIND,
  POOL_KIND_NAME,
} from './010_rnd_ObjectPool.js';

import {
  getDefaultTypedArrayPool,
  getDefaultLightingSlabs,
  ARRAY_KIND,
  ARRAY_KIND_NAME,
  SLAB_TAG,
  SLAB_TAG_NAME,
} from './011_rnd_TypedArrayPool.js';

import {
  getDefaultBufferPool,
  getDefaultLightingBuffers,
  BUFFER_USAGE,
} from './012_rnd_BufferPool.js';

import {
  getDefaultRenderTargetPool,
  getDefaultLightingTargets,
  RT_KIND,
  RT_KIND_NAME,
  RT_TAG,
  RT_TAG_NAME,
} from './013_rnd_RenderTargetPool.js';

import {
  getDefaultResourceRegistry,
  RES_KIND,
  RES_KIND_NAME,
  RES_TAG,
  RES_TAG_NAME,
} from './014_rnd_ResourceRegistry.js';

import {
  getDefaultManifest,
} from './015_rnd_AssetManifest.js';

import {
  getDefaultConfig,
  getResolvedConfig,
  PERF_TIER as CONFIG_PERF_TIER,
} from './016_rnd_Config.js';

import {
  getDefaultQualityController,
  getQualitySnapshot,
  QUALITY_LEVEL_NAME,
} from './017_rnd_QualityConfig.js';

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  RAW_CAPS as PLATFORM_RAW_CAPS,
} from './018_rnd_PlatformConfig.js';

import {
  ANDROID_PROFILE,
  GPU_FAMILY_NAME,
  SHADOW_TUNING,
  AO_TUNING,
} from './019_rnd_AndroidProfile.js';

import {
  DESKTOP_PROFILE,
  DESKTOP_GPU_NAME,
  IS_ANGLE,
  ANGLE_BACKEND,
} from './020_rnd_DesktopProfile.js';

import {
  CAPABILITIES,
  RAW_CAPS_DEEP,
  FEATURES as WEBGL_FEATURES,
} from './021_rnd_Capabilities.js';

import {
  FEATURE_DETECTOR,
  RAW_FEATURES,
  FEATURES as RUNTIME_FEATURES,
  getWorkerPoolSize,
  getHardwareConcurrency,
  getDeviceMemoryGB,
  getStorageEstimate,
  getNetworkInfo,
} from './022_rnd_FeatureDetector.js';

import {
  getDefaultPerfTierResolver,
  getEffectiveTierName,
  getCurrentBudget,
  PERF_TIER as TIER_ENUM,
} from './023_rnd_PerfTier.js';

import {
  getDefaultProfiler,
  profilerStats,
  HISTORY_CAPACITY as PROFILER_HISTORY_CAPACITY,
} from './024_rnd_Profiler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_SOURCES = 32;
export const HISTORY_SNAPSHOTS = 60;

export const DEFAULT_REFRESH_FRAMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 15 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 30 :
                                 45;

export const SOURCE_INTERVAL = Object.freeze({
  EVERY_REFRESH:   0,
  EVERY_OTHER:     1,
  EVERY_QUARTER:   2,
  MANUAL:          3,
});

/* ------------------------------------------------------------------ */
/* 1. SNAPSHOT STRUCTURE (single instance, in-place update)            */
/* ------------------------------------------------------------------ */

export class StatsSnapshot {
  constructor() {
    // ---- Top-level ----
    this.frame          = 0;
    this.elapsedMs      = 0;
    this.timestamp      = 0;
    this.collectedAtMs  = 0;
    this.collectTimeMs  = 0;
    this.refreshCount   = 0;
    this.perfTier       = PERF_TIER_LOCAL;

    // ---- Platform ----
    this.platform = {
      isAndroid:     DEVICE.isAndroid,
      isIOS:         DEVICE.isIOS,
      isMobile:      DEVICE.isMobile,
      isDesktop:     DEVICE.isDesktop,
      webglVersion:  DEVICE.webglVersion,
      gpuFamily:     DEVICE.isDesktop ? DESKTOP_GPU_NAME : GPU_FAMILY_NAME,
      hardwareConcurrency: getHardwareConcurrency(),
      deviceMemoryGB:      getDeviceMemoryGB(),
      workerPoolSize:      getWorkerPoolSize(),
      profileTier:         PLATFORM_CONFIG.tier,
      precision:           ANDROID_PROFILE ? ANDROID_PROFILE.precision : null,
      dprCap:              PLATFORM_CONFIG.dprCap,
      quirks:              QUIRKS,
    };

    // ---- Tier ----
    this.tier = {
      effective:      'medium',
      score:          0.5,
      confidence:     0.5,
      override:       false,
      thermalCap:     'ultra',
      batteryCap:     'ultra',
      extraCap:       'ultra',
    };

    // ---- Quality ----
    this.quality = {
      level:             'high',
      resolutionScale:   1.0,
      shadowMapSize:     1024,
      giUpdateBudgetHz:  15,
      aoResolutionScale: 0.5,
      postPassBudget:    4,
      drawDistance:      140,
      maxClusterLights:  64,
      lastKnob:          'none',
      lastReason:        'init',
    };

    // ---- Frame stats ----
    this.frameStats = {
      mean: 0, min: 0, max: 0,
      p50:  0, p95: 0, p99: 0,
      fps:  0, samples: 0,
    };

    // ---- Hitch stats ----
    this.hitchStats = {
      hitches: 0, janks: 0, criticals: 0, totalFrames: 0, hitchRate: 0,
    };

    // ---- Domain stats ----
    this.domainStats = null;   // Array<{name, lastMs, emaMs, peakMs, totalMs, callCount}>

    // ---- Top scopes ----
    this.topScopes = null;     // Array<{name, emaMs, lastMs, peakMs, callCount}>

    // ---- GPU ----
    this.gpu = {
      available: false,
      lastMs:    0,
      averageMs: 0,
      sampleCount: 0,
    };

    // ---- Memory ----
    this.memory = {
      heapUsedMB:       0,
      heapPercent:      0,
      gpuMB:            0,
      gpuPercent:       0,
      rtBytesInUse:     0,
      rtBytesPeak:      0,
      attrBytesInUse:   0,
      attrBytesPeak:    0,
      registryLive:     0,
      registryPeak:     0,
    };

    // ---- Pools ----
    this.pools = {
      object: {
        vector3InUse: 0, vector3Peak: 0, vector3Free: 0,
        vector4InUse: 0, vector4Peak: 0, vector4Free: 0,
        colorInUse:   0, colorPeak:   0, colorFree:   0,
        quaternionInUse: 0, quaternionFree: 0,
        matrix4InUse: 0, matrix4Free: 0,
      },
      typed: {
        f32InUse: 0, f32Free: 0,
        u32InUse: 0, u32Free: 0,
        u16InUse: 0, u16Free: 0,
        u8InUse:  0, u8Free:  0,
        estimatedBytes: 0,
      },
      buffer: {
        attributesAcquired: 0,
        attributesReleased: 0,
        gpuBytesInUse:      0,
        gpuBytesPeak:       0,
        namedCount:         0,
      },
      renderTarget: {
        acquired:         0,
        released:         0,
        namedCount:       0,
        gpuBytesInUse:    0,
        gpuBytesPeak:     0,
        gpuMegabytesInUse:0,
        gpuMegabytesPeak: 0,
      },
    };

    // ---- Registry ----
    this.registry = {
      currentLive: 0,
      peakLive:    0,
      totalRegistered: 0,
      totalDisposed:   0,
      totalZombies:    0,
      totalLeaks:      0,
      namedCount:      0,
      ownerCount:      0,
    };

    // ---- Manifest ----
    this.manifest = {
      count:      0,
      registered: 0,
      loaded:     0,
      failed:     0,
      skipped:    0,
      loadMs:     0,
    };

    // ---- Lights (fed by external sources) ----
    this.lights = {
      ambient:      0,
      hemisphere:   0,
      directional:  0,
      point:        0,
      spot:         0,
      rectArea:     0,
      totalActive:  0,
      totalShadowCasting: 0,
      clusterCells: 0,
    };

    // ---- Shadows (fed by external sources) ----
    this.shadows = {
      atlasAllocated:  0,
      atlasUsed:       0,
      cascades:        0,
      mapSize:         1024,
      filter:          'pcf',
      lastUpdateMs:    0,
    };

    // ---- GI / AO (fed by external sources) ----
    this.gi = {
      probeCount:      0,
      probeCapacity:   0,
      updateHz:        0,
      lastUpdateMs:    0,
      sampleCount:     0,
      multiBounce:     false,
    };

    this.ao = {
      sampleCount:     0,
      resolutionScale: 0.5,
      temporal:        false,
      lastUpdateMs:    0,
    };

    // ---- Feature flags (read-only, useful for debug) ----
    this.features = null;      // snapshot copy of RUNTIME_FEATURES

    // ---- Custom sections (downstream systems add their own) ----
    this.custom = null;        // Map<string, object> created on first use
  }
}

/* ------------------------------------------------------------------ */
/* 2. STATS COLLECTOR                                                 */
/* ------------------------------------------------------------------ */

export class StatsCollector {
  constructor(options = {}) {
    this.options = Object.assign({
      refreshEveryFrames: DEFAULT_REFRESH_FRAMES,
      historyCapacity:    HISTORY_SNAPSHOTS,
      collectCustom:      true,
      collectFeatures:    true,
      collectHistory:     true,
    }, options || {});

    // Snapshots.
    this.snapshot = new StatsSnapshot();
    this.previous = new StatsSnapshot();
    this.hasPrevious = false;

    // History ring (lightweight numeric summaries only).
    this.historyCapacity = this.options.historyCapacity;
    this.historyFrame     = new Uint32Array(this.historyCapacity);
    this.historyFps       = new Float32Array(this.historyCapacity);
    this.historyMean      = new Float32Array(this.historyCapacity);
    this.historyP95       = new Float32Array(this.historyCapacity);
    this.historyQuality   = new Uint8Array(this.historyCapacity);
    this.historyGpuMB     = new Float32Array(this.historyCapacity);
    this.historyHead      = 0;
    this.historyCount     = 0;

    // Custom sections registry.
    this.customSections = new Map();

    // Named source registry (upstream stat providers).
    this.sources = new Array(MAX_SOURCES);
    for (let i = 0; i < MAX_SOURCES; i++) this.sources[i] = null;
    this.sourceCount = 0;

    // Frame counter & timing.
    this.frame = 0;
    this.lastRefreshFrame = -9999;
    this.refreshInterval = this.options.refreshEveryFrames;

    this._listeners = new Map();

    // Pre-built scratch for the aggregate step.
    this._scratchFeatures = null;
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
      try { arr[i](payload); } catch (e) { console.error(`[025_rnd_StatsCollector] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- source registry ---------------- */

  /**
   * Register a named source callback. The callback receives the current
   * snapshot and is expected to write its section into `snapshot.custom`
   * (created lazily) or one of the top-level sections.
   *
   *   collector.registerSource('lights', (snap) => {
   *     snap.lights.point = lightManager.activePointLights;
   *     ...
   *   }, SOURCE_INTERVAL.EVERY_REFRESH);
   */
  registerSource(name, fn, interval = SOURCE_INTERVAL.EVERY_REFRESH) {
    if (typeof name !== 'string' || typeof fn !== 'function') return false;
    if (this.sourceCount >= MAX_SOURCES) return false;
    const idx = this.sourceCount++;
    this.sources[idx] = { name, fn, interval, lastRan: -1 };
    return true;
  }

  unregisterSource(name) {
    for (let i = 0; i < this.sourceCount; i++) {
      const s = this.sources[i];
      if (s && s.name === name) {
        for (let j = i; j < this.sourceCount - 1; j++) {
          this.sources[j] = this.sources[j + 1];
        }
        this.sources[this.sourceCount - 1] = null;
        this.sourceCount--;
        return true;
      }
    }
    return false;
  }

  /* ---------------- custom sections ---------------- */

  setSection(name, data) {
    if (typeof name !== 'string') return false;
    if (!this.snapshot.custom) this.snapshot.custom = new Map();
    this.snapshot.custom.set(name, data);
    return true;
  }

  getSection(name) {
    if (!this.snapshot.custom) return null;
    return this.snapshot.custom.get(name) || null;
  }

  /* ---------------- per-frame tick ---------------- */

  tick(dtMs) {
    this.frame++;
    if ((this.frame - this.lastRefreshFrame) >= this.refreshInterval) {
      this.collect(dtMs);
      this.lastRefreshFrame = this.frame;
    }
  }

  /* ---------------- collection ---------------- */

  collect(dtMs) {
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    // Preserve the previous snapshot (shallow swap for delta readers).
    if (this.hasPrevious) {
      // Copy select fields — the rest are only read once per refresh.
      this.previous.frame = this.snapshot.frame;
    }

    const snap = this.snapshot;

    // ---- Top-level ----
    snap.frame         = this.frame;
    snap.timestamp     = Date.now();
    snap.collectedAtMs = t0;
    snap.refreshCount++;

    // ---- Platform ----
    this._collectPlatform(snap);

    // ---- Tier ----
    this._collectTier(snap);

    // ---- Quality ----
    this._collectQuality(snap);

    // ---- Profiler (frame stats, hitches, domains, scopes, memory, gpu) ----
    this._collectProfiler(snap);

    // ---- Pools ----
    this._collectObjectPool(snap);
    this._collectTypedPool(snap);
    this._collectBufferPool(snap);
    this._collectRenderTargetPool(snap);

    // ---- Registry ----
    this._collectRegistry(snap);

    // ---- Manifest ----
    this._collectManifest(snap);

    // ---- Features ----
    if (this.options.collectFeatures) {
      this._collectFeatures(snap);
    }

    // ---- Named sources (downstream systems) ----
    this._runSources(snap);

    // ---- History ----
    if (this.options.collectHistory) {
      this._pushHistory(snap);
    }

    // ---- Timing ----
    const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    snap.collectTimeMs = t1 - t0;
    snap.elapsedMs = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    this.hasPrevious = true;
    this._emit('collected', { frame: snap.frame, refreshCount: snap.refreshCount, timeMs: snap.collectTimeMs });

    return snap;
  }

  /* ---------------- sub-collectors ---------------- */

  _collectPlatform(snap) {
    const p = snap.platform;
    p.isAndroid    = DEVICE.isAndroid;
    p.isIOS        = DEVICE.isIOS;
    p.isMobile     = DEVICE.isMobile;
    p.isDesktop    = DEVICE.isDesktop;
    p.webglVersion = DEVICE.webglVersion;
    p.gpuFamily    = DEVICE.isDesktop ? DESKTOP_GPU_NAME : GPU_FAMILY_NAME;
    p.hardwareConcurrency = getHardwareConcurrency();
    p.deviceMemoryGB      = getDeviceMemoryGB();
    p.workerPoolSize      = getWorkerPoolSize();
    p.profileTier         = PLATFORM_CONFIG.tier;
    p.dprCap              = PLATFORM_CONFIG.dprCap;
    p.quirks              = QUIRKS;
  }

  _collectTier(snap) {
    const resolver = getDefaultPerfTierResolver();
    const s = resolver.getSnapshot();
    snap.tier.effective  = s.effectiveTierName;
    snap.tier.score      = s.score;
    snap.tier.confidence = s.confidence;
    snap.tier.override   = s.override;
    snap.tier.thermalCap = PERF_TIER_NAME[s.thermalCap] || 'ultra';
    snap.tier.batteryCap = PERF_TIER_NAME[s.batteryCap] || 'ultra';
    snap.tier.extraCap   = PERF_TIER_NAME[s.extraCap]   || 'ultra';
  }

  _collectQuality(snap) {
    const q = getQualitySnapshot();
    snap.quality.level             = QUALITY_LEVEL_NAME[q.level] || 'high';
    snap.quality.resolutionScale   = q.resolutionScale;
    snap.quality.shadowMapSize     = q.shadowMapSize;
    snap.quality.giUpdateBudgetHz  = q.giUpdateBudgetHz;
    snap.quality.aoResolutionScale = q.aoResolutionScale;
    snap.quality.postPassBudget    = q.postPassBudget;
    snap.quality.drawDistance      = q.drawDistanceMeters;
    snap.quality.maxClusterLights  = q.maxClusterLights;
    snap.quality.lastKnob          = q.lastKnob;
    snap.quality.lastReason        = q.lastReason;
  }

  _collectProfiler(snap) {
    const p = profilerStats();

    snap.frameStats = p.frameStats || snap.frameStats;
    snap.hitchStats = p.hitchStats || snap.hitchStats;
    snap.domainStats = p.domainStats || [];
    snap.topScopes   = getDefaultProfiler().getTopScopes(10);
    snap.memory      = p.memoryStats || snap.memory;
    snap.gpu         = p.gpuStats    || snap.gpu;
  }

  _collectObjectPool(snap) {
    const pool = getDefaultPoolSet();
    if (!pool) return;
    const v3 = pool.vector3;
    const v4 = pool.vector4;
    const c  = pool.color;
    const q  = pool.quaternion;
    const m4 = pool.matrix4;

    snap.pools.object.vector3InUse  = v3 ? v3.stats.currentInUse : 0;
    snap.pools.object.vector3Peak   = v3 ? v3.stats.peakInUse    : 0;
    snap.pools.object.vector3Free   = v3 ? v3.freeCount          : 0;
    snap.pools.object.vector4InUse  = v4 ? v4.stats.currentInUse : 0;
    snap.pools.object.vector4Peak   = v4 ? v4.stats.peakInUse    : 0;
    snap.pools.object.vector4Free   = v4 ? v4.freeCount          : 0;
    snap.pools.object.colorInUse    = c  ? c.stats.currentInUse  : 0;
    snap.pools.object.colorPeak     = c  ? c.stats.peakInUse     : 0;
    snap.pools.object.colorFree     = c  ? c.freeCount           : 0;
    snap.pools.object.quaternionInUse = q ? q.stats.currentInUse : 0;
    snap.pools.object.quaternionFree  = q ? q.freeCount          : 0;
    snap.pools.object.matrix4InUse  = m4 ? m4.stats.currentInUse : 0;
    snap.pools.object.matrix4Free   = m4 ? m4.freeCount          : 0;
  }

  _collectTypedPool(snap) {
    const pool = getDefaultTypedArrayPool();
    if (!pool || !pool.buckets) return;

    // Aggregate per-kind in-use / free across all buckets.
    let f32InUse = 0, f32Free = 0;
    let u32InUse = 0, u32Free = 0;
    let u16InUse = 0, u16Free = 0;
    let u8InUse  = 0, u8Free  = 0;

    const bF32 = pool.buckets[ARRAY_KIND.F32];
    const bU32 = pool.buckets[ARRAY_KIND.U32];
    const bU16 = pool.buckets[ARRAY_KIND.U16];
    const bU8  = pool.buckets[ARRAY_KIND.U8];

    if (bF32) for (let i = 0; i < bF32.length; i++) { f32InUse += bF32[i].currentInUse; f32Free += bF32[i].freeCount; }
    if (bU32) for (let i = 0; i < bU32.length; i++) { u32InUse += bU32[i].currentInUse; u32Free += bU32[i].freeCount; }
    if (bU16) for (let i = 0; i < bU16.length; i++) { u16InUse += bU16[i].currentInUse; u16Free += bU16[i].freeCount; }
    if (bU8)  for (let i = 0; i < bU8.length;  i++) { u8InUse  += bU8[i].currentInUse;  u8Free  += bU8[i].freeCount;  }

    snap.pools.typed.f32InUse = f32InUse;
    snap.pools.typed.f32Free  = f32Free;
    snap.pools.typed.u32InUse = u32InUse;
    snap.pools.typed.u32Free  = u32Free;
    snap.pools.typed.u16InUse = u16InUse;
    snap.pools.typed.u16Free  = u16Free;
    snap.pools.typed.u8InUse  = u8InUse;
    snap.pools.typed.u8Free   = u8Free;
    snap.pools.typed.estimatedBytes = pool.estimateBytes ? pool.estimateBytes() : 0;
  }

  _collectBufferPool(snap) {
    const pool = getDefaultBufferPool();
    if (!pool || !pool.stats) return;
    snap.pools.buffer.attributesAcquired = pool.stats.attributesAcquired;
    snap.pools.buffer.attributesReleased = pool.stats.attributesReleased;
    snap.pools.buffer.gpuBytesInUse      = pool.stats.gpuBytesInUse;
    snap.pools.buffer.gpuBytesPeak       = pool.stats.peakGpuBytesInUse;
    snap.pools.buffer.namedCount         = pool.stats.namedCount;
  }

  _collectRenderTargetPool(snap) {
    const pool = getDefaultRenderTargetPool();
    if (!pool || !pool.stats) return;
    snap.pools.renderTarget.acquired          = pool.stats.acquired;
    snap.pools.renderTarget.released          = pool.stats.released;
    snap.pools.renderTarget.namedCount        = pool.stats.namedCount;
    snap.pools.renderTarget.gpuBytesInUse     = pool.stats.gpuBytesInUse;
    snap.pools.renderTarget.gpuBytesPeak      = pool.stats.peakGpuBytesInUse;
    snap.pools.renderTarget.gpuMegabytesInUse = pool.stats.gpuMegabytesInUse;
    snap.pools.renderTarget.gpuMegabytesPeak  = pool.stats.gpuMegabytesPeak;
  }

  _collectRegistry(snap) {
    const reg = getDefaultResourceRegistry();
    if (!reg || !reg.stats) return;
    snap.registry.currentLive     = reg.stats.currentLive;
    snap.registry.peakLive        = reg.stats.peakLive;
    snap.registry.totalRegistered = reg.stats.totalRegistered;
    snap.registry.totalDisposed   = reg.stats.totalDisposed;
    snap.registry.totalZombies    = reg.stats.totalZombies;
    snap.registry.totalLeaks      = reg.stats.totalLeaksDetected;
    snap.registry.namedCount      = reg._named.size;
    snap.registry.ownerCount      = reg.ownerCount;
  }

  _collectManifest(snap) {
    const m = getDefaultManifest();
    if (!m || !m.stats) return;
    snap.manifest.count      = m.count;
    snap.manifest.registered = m.stats.registered;
    snap.manifest.loaded     = m.stats.loaded;
    snap.manifest.failed     = m.stats.failed;
    snap.manifest.skipped    = m.stats.skipped;
    snap.manifest.loadMs     = m.stats.loadMs;
  }

  _collectFeatures(snap) {
    snap.features = RUNTIME_FEATURES;
  }

  _runSources(snap) {
    for (let i = 0; i < this.sourceCount; i++) {
      const s = this.sources[i];
      if (!s) continue;

      // Interval gating.
      if (s.interval === SOURCE_INTERVAL.EVERY_OTHER && (snap.refreshCount % 2) !== 0) continue;
      if (s.interval === SOURCE_INTERVAL.EVERY_QUARTER && (snap.refreshCount % 4) !== 0) continue;
      if (s.interval === SOURCE_INTERVAL.MANUAL) continue;

      try {
        s.fn(snap);
        s.lastRan = snap.refreshCount;
      } catch (e) {
        console.error(`[025_rnd_StatsCollector] source "${s.name}" failed`, e);
      }
    }
  }

  _pushHistory(snap) {
    const idx = this.historyHead;
    this.historyFrame[idx]   = snap.frame;
    this.historyFps[idx]     = snap.frameStats.fps;
    this.historyMean[idx]    = snap.frameStats.mean;
    this.historyP95[idx]     = snap.frameStats.p95;
    this.historyQuality[idx] = QUALITY_LEVEL_NAME.indexOf(snap.quality.level);
    this.historyGpuMB[idx]   = snap.pools.renderTarget.gpuMegabytesInUse;

    this.historyHead = (this.historyHead + 1) % this.historyCapacity;
    if (this.historyCount < this.historyCapacity) this.historyCount++;
  }

  /* ---------------- accessors ---------------- */

  getSnapshot() { return this.snapshot; }

  getSection(name) {
    if (!name) return this.snapshot;
    if (Object.prototype.hasOwnProperty.call(this.snapshot, name)) {
      return this.snapshot[name];
    }
    return this.getSection(name);
  }

  getHistory() {
    return {
      capacity: this.historyCapacity,
      count:    this.historyCount,
      head:     this.historyHead,
      frame:    this.historyFrame,
      fps:      this.historyFps,
      mean:     this.historyMean,
      p95:      this.historyP95,
      quality:  this.historyQuality,
      gpuMB:    this.historyGpuMB,
    };
  }

  /* ---------------- exports ---------------- */

  exportJSON() {
    // Ensure a fresh collection before export.
    this.collect(0);
    return JSON.stringify(this.snapshot, (k, v) => {
      // Typed arrays & Maps → plain arrays/objects.
      if (v instanceof Map) return Object.fromEntries(v);
      return v;
    }, 2);
  }

  exportCompact() {
    this.collect(0);
    const s = this.snapshot;
    return {
      f:        s.frame,
      fps:      Number(s.frameStats.fps.toFixed(1)),
      mean:     Number(s.frameStats.mean.toFixed(2)),
      p95:      Number(s.frameStats.p95.toFixed(2)),
      p99:      Number(s.frameStats.p99.toFixed(2)),
      max:      Number(s.frameStats.max.toFixed(2)),
      hitches:  s.hitchStats.hitches,
      janks:    s.hitchStats.janks,
      crit:     s.hitchStats.criticals,
      quality:  s.quality.level,
      tier:     s.tier.effective,
      gpuMB:    Number(s.pools.renderTarget.gpuMegabytesInUse.toFixed(1)),
      rtInUse:  s.pools.renderTarget.acquired,
      attrInUse:s.pools.buffer.attributesAcquired,
      regLive:  s.registry.currentLive,
      leaks:    s.registry.totalLeaks,
    };
  }

  exportCSV() {
    this.collect(0);
    const s = this.snapshot;
    const rows = [
      ['section', 'key', 'value'],
      ['top', 'frame', s.frame],
      ['top', 'refreshCount', s.refreshCount],
      ['top', 'collectTimeMs', s.collectTimeMs],
      ['tier', 'effective', s.tier.effective],
      ['tier', 'score', s.tier.score],
      ['quality', 'level', s.quality.level],
      ['quality', 'resolutionScale', s.quality.resolutionScale],
      ['quality', 'shadowMapSize', s.quality.shadowMapSize],
      ['frameStats', 'mean', s.frameStats.mean],
      ['frameStats', 'p95', s.frameStats.p95],
      ['frameStats', 'p99', s.frameStats.p99],
      ['frameStats', 'fps', s.frameStats.fps],
      ['hitchStats', 'hitches', s.hitchStats.hitches],
      ['hitchStats', 'janks', s.hitchStats.janks],
      ['hitchStats', 'criticals', s.hitchStats.criticals],
      ['pools.renderTarget', 'acquired', s.pools.renderTarget.acquired],
      ['pools.renderTarget', 'gpuMB', s.pools.renderTarget.gpuMegabytesInUse],
      ['pools.buffer', 'attributesAcquired', s.pools.buffer.attributesAcquired],
      ['pools.buffer', 'gpuBytesInUse', s.pools.buffer.gpuBytesInUse],
      ['registry', 'currentLive', s.registry.currentLive],
      ['registry', 'totalLeaks', s.registry.totalLeaks],
    ];
    let csv = '';
    for (let i = 0; i < rows.length; i++) {
      csv += rows[i].join(',') + '\n';
    }
    return csv;
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    this.frame = 0;
    this.lastRefreshFrame = -9999;
    this.hasPrevious = false;
    this.historyHead = 0;
    this.historyCount = 0;
    this.historyFrame.fill(0);
    this.historyFps.fill(0);
    this.historyMean.fill(0);
    this.historyP95.fill(0);
    this.historyQuality.fill(0);
    this.historyGpuMB.fill(0);
    return this;
  }

  dispose() {
    this.reset();
    for (let i = 0; i < this.sourceCount; i++) this.sources[i] = null;
    this.sourceCount = 0;
    this.customSections.clear();
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultCollector = null;

export function getDefaultStatsCollector() {
  if (!_defaultCollector) _defaultCollector = new StatsCollector();
  return _defaultCollector;
}

export function disposeDefaultStatsCollector() {
  if (_defaultCollector) {
    _defaultCollector.dispose();
    _defaultCollector = null;
  }
}

/* ------------------------------------------------------------------ */
/* 4. HOT-PATH HELPERS                                                */
/* ------------------------------------------------------------------ */

export function statsTick(dtMs) {
  getDefaultStatsCollector().tick(dtMs);
}

export function statsCollect() {
  return getDefaultStatsCollector().collect(0);
}

export function statsSnapshot() {
  return getDefaultStatsCollector().getSnapshot();
}

export function statsRegisterSource(name, fn, interval) {
  return getDefaultStatsCollector().registerSource(name, fn, interval);
}

export function statsUnregisterSource(name) {
  return getDefaultStatsCollector().unregisterSource(name);
}

/* ------------------------------------------------------------------ */
/* 5. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createStatsCollector(options = {}) {
  return new StatsCollector(options);
}

/* ------------------------------------------------------------------ */
/* 6. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  StatsCollector,
  StatsSnapshot,

  createStatsCollector,
  getDefaultStatsCollector,
  disposeDefaultStatsCollector,

  statsTick,
  statsCollect,
  statsSnapshot,
  statsRegisterSource,
  statsUnregisterSource,

  SOURCE_INTERVAL,
  MAX_SOURCES,
  HISTORY_SNAPSHOTS,
  DEFAULT_REFRESH_FRAMES,
};

export default _defaultExport;