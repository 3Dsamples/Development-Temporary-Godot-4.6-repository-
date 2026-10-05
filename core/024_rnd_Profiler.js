// File : 024
// name : src/core/024_rnd_Profiler.js
// description : Runtime profiler and performance telemetry hub for the anime
//               lighting stack on Android mobile. Owns frame-time history,
//               per-domain timing breakdown (one bucket per FrameScheduler
//               domain), named scope timers, hitch detection, GPU timing
//               hooks (EXT_disjoint_timer_query), memory accounting, and a
//               rolling ring buffer of the last N frames for regression
//               capture and adaptive-quality feedback.
//
//               Where 017_rnd_QualityConfig.js decides WHAT quality level to
//               run at, THIS module provides the raw evidence: min / max /
//               mean / p50 / p95 / p99 frame-time, per-domain cost profile,
//               per-scope cost profile, hitch histogram, jank count, and
//               memory pressure trend. The quality controller reads from
//               here; nothing writes back through this module.
//
//               Design:
//                 • Ring buffer of fixed frame capacity (PERF_TIER tuned:
//                   240 / 480 / 960 frames) — no dynamic growth.
//                 • Per-domain timing buckets aligned to DOMAIN enum from
//                   005_rnd_FrameScheduler.js so analysis matches scheduling.
//                 • Named scope registry: `beginScope('giProbeBake')` /
//                   `endScope()` — LIFO stack, up to MAX_SCOPE_DEPTH.
//                   Scope names resolve to fixed slot indices once, so the
//                   hot path only touches typed arrays.
//                 • Hitch detection: any frame > HITCH_MS (default 33.3 ms)
//                   is recorded in a hitch ring buffer with a category tag
//                   (frame / domain / scope) so the offending subsystem is
//                   identifiable.
//                 • GPU timing: optional EXT_disjoint_timer_query path with
//                   graceful fallback when unavailable (most Android
//                   browsers). Never blocks the main thread.
//                 • Memory tracking: navigator.deviceMemory for the cap,
//                   performance.memory (Chrome) for live heap when present,
//                   and a manual GPU byte counter fed by the pools.
//                 • Export: `exportJSON()` for regression snapshots and
//                   `exportCompact()` for CI logs.
//                 • Zero per-frame allocations on the hot path — every
//                   buffer is pre-allocated and reused; every read returns
//                   cached views.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external profiler libs; every internal buffer sized
//               once at construction.
// best for : Giving the entire lighting stack one authoritative performance
//            telemetry hub. Quality controller, adaptive LOD scalers,
//            debug HUD, and CI regression tests all read the same numbers
//            from the same source so there is exactly one truth.
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
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
} from './018_rnd_PlatformConfig.js';

import {
  ANDROID_PROFILE,
} from './019_rnd_AndroidProfile.js';

import {
  CAPABILITIES,
  RAW_CAPS_DEEP,
} from './021_rnd_Capabilities.js';

import {
  RAW_FEATURES,
  FEATURES as RUNTIME_FEATURES,
} from './022_rnd_FeatureDetector.js';

import {
  PERF_TIER_NAME,
  getEffectiveTierName,
} from './023_rnd_PerfTier.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const HISTORY_CAPACITY =
  PERF_TIER_LOCAL === 'HIGH'   ? 960 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 480 :
                                 240;

export const MAX_SCOPES       = 64;
export const MAX_SCOPE_DEPTH  = 32;
export const MAX_HITCHES      = 64;
export const MAX_MARKERS      = 128;

const HITCH_MS_DEFAULT        = 33.3;   // ~30 fps boundary
const JANK_MS_DEFAULT         = 50.0;   // heavy jank
const CRITICAL_MS_DEFAULT     = 100.0;  // dropped frame

const EMA_ALPHA_FRAME         = 0.10;
const EMA_ALPHA_DOMAIN        = 0.12;
const EMA_ALPHA_SCOPE         = 0.15;

export const HITCH_CATEGORY = Object.freeze({
  NONE:     0,
  FRAME:    1,
  DOMAIN:   2,
  SCOPE:    3,
  MEMORY:   4,
  GPU:      5,
  COUNT:    6,
});

export const HITCH_CATEGORY_NAME = Object.freeze([
  'none',
  'frame',
  'domain',
  'scope',
  'memory',
  'gpu',
]);

export const SAMPLE_KIND = Object.freeze({
  FRAME:   0,
  DOMAIN:  1,
  SCOPE:   2,
  GPU:     3,
  COUNT:   4,
});

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _percentileFromSorted(sorted, len, p) {
  if (len === 0) return 0;
  if (p <= 0) return sorted[0];
  if (p >= 1) return sorted[len - 1];
  const idx = Math.min(len - 1, Math.max(0, Math.floor(p * (len - 1))));
  return sorted[idx];
}

/* ------------------------------------------------------------------ */
/* 2. GPU TIMER (optional)                                            */
/* ------------------------------------------------------------------ */

class GpuTimer {
  constructor() {
    this.available = false;
    this.gl = null;
    this.ext = null;
    this.pendingQueries = 0;
    this.lastGpuMs = 0;
    this.samples = new Float32Array(32);
    this.sampleCount = 0;
    this.sampleHead = 0;
  }

  attach(renderer) {
    if (!renderer || !renderer.getContext) return false;
    try {
      const gl = renderer.getContext();
      if (!gl) return false;
      const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2')
                || gl.getExtension('EXT_disjoint_timer_query');
      if (!ext) return false;
      this.gl = gl;
      this.ext = ext;
      this.available = true;
      return true;
    } catch (_) {
      this.available = false;
      return false;
    }
  }

  detach() {
    this.available = false;
    this.gl = null;
    this.ext = null;
    this.pendingQueries = 0;
  }

  recordGpuMs(ms) {
    if (!Number.isFinite(ms) || ms < 0) return;
    this.lastGpuMs = ms;
    this.samples[this.sampleHead] = ms;
    this.sampleHead = (this.sampleHead + 1) % this.samples.length;
    if (this.sampleCount < this.samples.length) this.sampleCount++;
  }

  getAverageGpuMs() {
    if (this.sampleCount === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.sampleCount; i++) sum += this.samples[i];
    return sum / this.sampleCount;
  }
}

/* ------------------------------------------------------------------ */
/* 3. SCOPE REGISTRY                                                  */
/* ------------------------------------------------------------------ */

class ScopeRegistry {
  constructor(capacity) {
    this.capacity   = capacity;
    this.name       = new Array(capacity);
    this.nameIndex  = new Map();

    // Per-scope rolling stats.
    this.lastMs     = new Float32Array(capacity);
    this.emaMs      = new Float32Array(capacity);
    this.peakMs     = new Float32Array(capacity);
    this.totalMs    = new Float32Array(capacity);
    this.callCount  = new Uint32Array(capacity);
    this.count      = 0;

    // LIFO stack for nesting.
    this.stackSlot  = new Int32Array(MAX_SCOPE_DEPTH);
    this.stackStart = new Float64Array(MAX_SCOPE_DEPTH);
    this.stackDepth = 0;
  }

  register(name) {
    if (typeof name !== 'string' || name.length === 0) return -1;
    const existing = this.nameIndex.get(name);
    if (existing !== undefined) return existing;
    if (this.count >= this.capacity) return -1;
    const idx = this.count++;
    this.name[idx] = name;
    this.nameIndex.set(name, idx);
    return idx;
  }

  begin(slot) {
    if (slot < 0 || slot >= this.count) return false;
    if (this.stackDepth >= MAX_SCOPE_DEPTH) return false;
    this.stackSlot[this.stackDepth]  = slot;
    this.stackStart[this.stackDepth] = _now();
    this.stackDepth++;
    return true;
  }

  end() {
    if (this.stackDepth <= 0) return -1;
    this.stackDepth--;
    const slot = this.stackSlot[this.stackDepth];
    const start = this.stackStart[this.stackDepth];
    const cost = _now() - start;

    this.lastMs[slot] = cost;
    this.emaMs[slot] += (cost - this.emaMs[slot]) * EMA_ALPHA_SCOPE;
    if (cost > this.peakMs[slot]) this.peakMs[slot] = cost;
    this.totalMs[slot] += cost;
    this.callCount[slot]++;

    return slot;
  }

  reset() {
    this.lastMs.fill(0);
    this.emaMs.fill(0);
    this.peakMs.fill(0);
    this.totalMs.fill(0);
    this.callCount.fill(0);
    this.stackDepth = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. HITCH LOG                                                       */
/* ------------------------------------------------------------------ */

class HitchLog {
  constructor(capacity) {
    this.capacity = capacity;
    this.frame    = new Uint32Array(capacity);
    this.costMs   = new Float32Array(capacity);
    this.category = new Uint8Array(capacity);
    this.slot     = new Int16Array(capacity); // scope/domain index, -1 if N/A
    this.head     = 0;
    this.count    = 0;
    this.total    = 0;
  }

  record(frame, costMs, category, slot) {
    this.frame[this.head]    = frame;
    this.costMs[this.head]   = costMs;
    this.category[this.head] = category;
    this.slot[this.head]     = slot | 0;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
    this.total++;
  }

  clear() {
    this.frame.fill(0);
    this.costMs.fill(0);
    this.category.fill(0);
    this.slot.fill(-1);
    this.head = 0;
    this.count = 0;
    this.total = 0;
  }

  /**
   * Copies up to `max` recent hits into caller-provided arrays.
   * Returns the number copied.
   */
  copyRecent(max, outFrame, outCost, outCategory, outSlot) {
    const n = Math.min(max, this.count);
    for (let i = 0; i < n; i++) {
      const idx = (this.head - n + i + this.capacity) % this.capacity;
      outFrame[i]    = this.frame[idx];
      outCost[i]     = this.costMs[idx];
      outCategory[i] = this.category[idx];
      outSlot[i]     = this.slot[idx];
    }
    return n;
  }
}

/* ------------------------------------------------------------------ */
/* 5. MARKER LOG                                                      */
/* ------------------------------------------------------------------ */

class MarkerLog {
  constructor(capacity) {
    this.capacity = capacity;
    this.frame    = new Uint32Array(capacity);
    this.timeMs   = new Float32Array(capacity);
    this.labelId  = new Int16Array(capacity);
    this.head     = 0;
    this.count    = 0;
    this.labels   = new Array(32);
    this.labelIndex = new Map();
    this.labelCount = 0;
  }

  registerLabel(label) {
    if (typeof label !== 'string' || label.length === 0) return -1;
    const existing = this.labelIndex.get(label);
    if (existing !== undefined) return existing;
    if (this.labelCount >= this.labels.length) return -1;
    const idx = this.labelCount++;
    this.labels[idx] = label;
    this.labelIndex.set(label, idx);
    return idx;
  }

  mark(frame, labelId) {
    this.frame[this.head]   = frame;
    this.timeMs[this.head]  = _now();
    this.labelId[this.head] = labelId | 0;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  clear() {
    this.head = 0;
    this.count = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 6. MEMORY TRACKER                                                  */
/* ------------------------------------------------------------------ */

class MemoryTracker {
  constructor() {
    this.available           = false;
    this.heapUsedBytes       = 0;
    this.heapTotalBytes      = 0;
    this.heapLimitBytes      = 0;
    this.externalBytes       = 0;

    this.gpuBytesAllocated   = 0;
    this.gpuBytesInUse       = 0;
    this.gpuBytesPeak        = 0;

    this.rtBytesAllocated    = 0;
    this.rtBytesInUse        = 0;
    this.rtBytesPeak         = 0;

    this.attrBytesAllocated  = 0;
    this.attrBytesInUse      = 0;
    this.attrBytesPeak       = 0;

    this.registryLiveCount   = 0;
    this.registryPeakCount   = 0;

    this.lastUpdateFrame     = 0;
    this.updateIntervalFrames= 30; // poll performance.memory every 30 frames
  }

  refresh(frame) {
    if (typeof performance === 'undefined' || !performance.memory) {
      this.available = false;
      return;
    }
    this.available = true;
    const m = performance.memory;
    this.heapUsedBytes  = m.usedJSHeapSize || 0;
    this.heapTotalBytes = m.totalJSHeapSize || 0;
    this.heapLimitBytes = m.jsHeapSizeLimit || 0;
    this.lastUpdateFrame = frame;
  }

  reportGpuFromPool(name, bytesInUse, bytesPeak) {
    if (name === 'render_target') {
      this.rtBytesInUse   = bytesInUse | 0;
      if (bytesPeak > this.rtBytesPeak) this.rtBytesPeak = bytesPeak;
    } else if (name === 'attribute') {
      this.attrBytesInUse = bytesInUse | 0;
      if (bytesPeak > this.attrBytesPeak) this.attrBytesPeak = bytesPeak;
    } else if (name === 'object') {
      this.registryLiveCount = bytesInUse | 0;
      if (bytesPeak > this.registryPeakCount) this.registryPeakCount = bytesPeak;
    }
    this.gpuBytesInUse = this.rtBytesInUse + this.attrBytesInUse;
    if (this.gpuBytesInUse > this.gpuBytesPeak) this.gpuBytesPeak = this.gpuBytesInUse;
  }

  getHeapMB() { return this.heapUsedBytes / 1048576; }
  getGpuMB()  { return this.gpuBytesInUse  / 1048576; }
  getHeapPercent() {
    if (this.heapLimitBytes <= 0) return 0;
    return Math.min(1, this.heapUsedBytes / this.heapLimitBytes);
  }
  getGpuPercent() {
    const capMB = ANDROID_PROFILE ? ANDROID_PROFILE.maxGpuMemoryMB : 128;
    const capBytes = capMB * 1048576;
    if (capBytes <= 0) return 0;
    return Math.min(1, this.gpuBytesInUse / capBytes);
  }
}

/* ------------------------------------------------------------------ */
/* 7. PROFILER                                                        */
/* ------------------------------------------------------------------ */

export class Profiler {
  constructor(options = {}) {
    this.options = Object.assign({
      historyCapacity:  HISTORY_CAPACITY,
      hitchMs:          HITCH_MS_DEFAULT,
      jankMs:           JANK_MS_DEFAULT,
      criticalMs:       CRITICAL_MS_DEFAULT,
      enableGpu:        true,
      enableMemory:     true,
      autoLog:          false,
    }, options || {});

    // Frame ring buffer.
    this.capacity     = this.options.historyCapacity;
    this.frameIndex   = new Uint32Array(this.capacity);
    this.frameDtMs    = new Float32Array(this.capacity);
    this.frameDomains = new Float32Array(this.capacity * DOMAIN.COUNT);
    this.frameScopes  = new Float32Array(this.capacity * MAX_SCOPES);
    this.frameGpuMs   = new Float32Array(this.capacity);
    this.head         = 0;
    this.frameCount   = 0;

    // Aggregate stats.
    this.frame        = 0;
    this.elapsedMs    = 0;

    this.dtLastMs     = 0;
    this.dtEmaMs      = 16.67;
    this.dtMinMs      = Number.POSITIVE_INFINITY;
    this.dtMaxMs      = 0;
    this.dtSumMs      = 0;

    this.hitches      = 0;
    this.janks        = 0;
    this.criticals    = 0;

    // Per-domain stats aligned to FrameScheduler DOMAIN enum.
    this.domainLastMs = new Float32Array(DOMAIN.COUNT);
    this.domainEmaMs  = new Float32Array(DOMAIN.COUNT);
    this.domainPeakMs = new Float32Array(DOMAIN.COUNT);
    this.domainTotalMs= new Float32Array(DOMAIN.COUNT);
    this.domainCalls  = new Uint32Array(DOMAIN.COUNT);

    // Scope / hitch / marker / GPU / memory subsystems.
    this.scopes       = new ScopeRegistry(MAX_SCOPES);
    this.hitchLog     = new HitchLog(MAX_HITCHES);
    this.markerLog    = new MarkerLog(MAX_MARKERS);
    this.memory       = new MemoryTracker();
    this.gpu          = new GpuTimer();

    // Frame-type categorization (based on where the cost landed).
    this.lastFrameCategory = HITCH_CATEGORY.NONE;

    // Scratch sort buffer for percentile computation.
    this._sortBuf = new Float32Array(this.capacity);

    // Cached views returned to callers (allocation-free).
    this._frameStats = {
      frame:       0,
      samples:     0,
      mean:        0,
      min:         0,
      max:         0,
      p50:         0,
      p95:         0,
      p99:         0,
      fps:         0,
    };

    this._listeners = new Map();
    this._attachedRenderer = null;
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
      try { arr[i](payload); } catch (e) { console.error(`[024_rnd_Profiler] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- renderer / GPU hook ---------------- */

  attachRenderer(renderer) {
    this._attachedRenderer = renderer;
    if (this.options.enableGpu) {
      this.gpu.attach(renderer);
    }
  }

  detachRenderer() {
    this.gpu.detach();
    this._attachedRenderer = null;
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame(dtMs) {
    this.frame++;
    this.dtLastMs = Number.isFinite(dtMs) ? Math.max(0, Math.min(100, dtMs)) : 0;
    this.elapsedMs += this.dtLastMs;

    // Update running stats.
    this.dtEmaMs += (this.dtLastMs - this.dtEmaMs) * EMA_ALPHA_FRAME;
    if (this.dtLastMs < this.dtMinMs) this.dtMinMs = this.dtLastMs;
    if (this.dtLastMs > this.dtMaxMs) this.dtMaxMs = this.dtLastMs;
    this.dtSumMs += this.dtLastMs;

    // Frame hitches.
    let category = HITCH_CATEGORY.NONE;
    if (this.dtLastMs >= this.options.criticalMs) {
      this.criticals++;
      category = HITCH_CATEGORY.FRAME;
    } else if (this.dtLastMs >= this.options.jankMs) {
      this.janks++;
      category = HITCH_CATEGORY.FRAME;
    } else if (this.dtLastMs >= this.options.hitchMs) {
      this.hitches++;
      category = HITCH_CATEGORY.FRAME;
    }
    this.lastFrameCategory = category;
    if (category !== HITCH_CATEGORY.NONE) {
      this.hitchLog.record(this.frame, this.dtLastMs, category, -1);
    }

    // Write frame ring buffer.
    this.frameIndex[this.head] = this.frame;
    this.frameDtMs[this.head]  = this.dtLastMs;
    this.frameGpuMs[this.head] = this.gpu.lastGpuMs;
    const domainBase = this.head * DOMAIN.COUNT;
    for (let d = 0; d < DOMAIN.COUNT; d++) {
      this.frameDomains[domainBase + d] = this.domainLastMs[d];
    }
    const scopeBase = this.head * MAX_SCOPES;
    for (let s = 0; s < this.scopes.count; s++) {
      this.frameScopes[scopeBase + s] = this.scopes.lastMs[s];
    }

    this.head = (this.head + 1) % this.capacity;
    if (this.frameCount < this.capacity) this.frameCount++;

    // Memory poll (throttled).
    if (this.options.enableMemory &&
        (this.frame - this.memory.lastUpdateFrame) >= this.memory.updateIntervalFrames) {
      this.memory.refresh(this.frame);
    }

    if (this.options.autoLog && (this.frame % 60 === 0)) {
      this._autoLogFrame();
    }

    return this;
  }

  /* ---------------- domain timing ---------------- */

  recordDomain(domain, costMs) {
    if (domain < 0 || domain >= DOMAIN.COUNT) return;
    const c = Number.isFinite(costMs) ? Math.max(0, costMs) : 0;
    this.domainLastMs[domain] = c;
    this.domainEmaMs[domain] += (c - this.domainEmaMs[domain]) * EMA_ALPHA_DOMAIN;
    if (c > this.domainPeakMs[domain]) this.domainPeakMs[domain] = c;
    this.domainTotalMs[domain] += c;
    this.domainCalls[domain]++;

    if (c >= this.options.hitchMs) {
      this.hitchLog.record(this.frame, c, HITCH_CATEGORY.DOMAIN, domain);
    }
  }

  /* ---------------- scope timing ---------------- */

  registerScope(name) {
    return this.scopes.register(name);
  }

  beginScope(nameOrSlot) {
    const slot = (typeof nameOrSlot === 'number')
      ? nameOrSlot
      : this.scopes.register(nameOrSlot);
    return this.scopes.begin(slot);
  }

  endScope() {
    const slot = this.scopes.end();
    if (slot >= 0 && this.scopes.lastMs[slot] >= this.options.hitchMs) {
      this.hitchLog.record(this.frame, this.scopes.lastMs[slot], HITCH_CATEGORY.SCOPE, slot);
    }
    return slot;
  }

  /* ---------------- markers ---------------- */

  registerMarker(label) {
    return this.markerLog.registerLabel(label);
  }

  mark(nameOrLabelId) {
    const id = (typeof nameOrLabelId === 'number')
      ? nameOrLabelId
      : this.markerLog.registerLabel(nameOrLabelId);
    if (id < 0) return false;
    this.markerLog.mark(this.frame, id);
    return true;
  }

  /* ---------------- GPU timing ---------------- */

  recordGpuMs(ms) {
    this.gpu.recordGpuMs(ms);
    if (ms >= this.options.hitchMs) {
      this.hitchLog.record(this.frame, ms, HITCH_CATEGORY.GPU, -1);
    }
  }

  /* ---------------- memory reporting ---------------- */

  reportMemory(name, inUse, peak) {
    this.memory.reportGpuFromPool(name, inUse, peak);
  }

  /* ---------------- percentile computation ---------------- */

  computeFrameStats() {
    const n = this.frameCount;
    const out = this._frameStats;
    out.frame   = this.frame;
    out.samples = n;

    if (n === 0) {
      out.mean = 0; out.min = 0; out.max = 0;
      out.p50 = 0; out.p95 = 0; out.p99 = 0; out.fps = 0;
      return out;
    }

    const buf = this._sortBuf;
    let sum = 0;
    let min = Number.POSITIVE_INFINITY;
    let max = 0;
    for (let i = 0; i < n; i++) {
      const v = this.frameDtMs[i];
      buf[i] = v;
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    // Sort only the first n entries. TypedArray.sort handles numeric sort
    // natively and is very fast on Android (usually native sort).
    const sub = buf.subarray(0, n);
    sub.sort();

    out.mean = sum / n;
    out.min  = min;
    out.max  = max;
    out.p50  = _percentileFromSorted(sub, n, 0.50);
    out.p95  = _percentileFromSorted(sub, n, 0.95);
    out.p99  = _percentileFromSorted(sub, n, 0.99);
    out.fps  = out.mean > 0 ? (1000 / out.mean) : 0;

    return out;
  }

  /* ---------------- diagnostics ---------------- */

  _autoLogFrame() {
    const stats = this.computeFrameStats();
    console.log(
      `[024_rnd_Profiler] frame=${stats.frame} ` +
      `fps=${stats.fps.toFixed(1)} ` +
      `mean=${stats.mean.toFixed(2)}ms ` +
      `p95=${stats.p95.toFixed(2)}ms ` +
      `p99=${stats.p99.toFixed(2)}ms ` +
      `max=${stats.max.toFixed(2)}ms ` +
      `hitches=${this.hitches} janks=${this.janks} criticals=${this.criticals}`
    );
  }

  getFrameStats() {
    return this.computeFrameStats();
  }

  getDomainStats() {
    const out = new Array(DOMAIN.COUNT);
    for (let d = 0; d < DOMAIN.COUNT; d++) {
      out[d] = {
        name:      DOMAIN_NAME[d],
        lastMs:    this.domainLastMs[d],
        emaMs:     this.domainEmaMs[d],
        peakMs:    this.domainPeakMs[d],
        totalMs:   this.domainTotalMs[d],
        callCount: this.domainCalls[d],
      };
    }
    return out;
  }

  getScopeStats() {
    const out = new Array(this.scopes.count);
    for (let s = 0; s < this.scopes.count; s++) {
      out[s] = {
        name:      this.scopes.name[s],
        lastMs:    this.scopes.lastMs[s],
        emaMs:     this.scopes.emaMs[s],
        peakMs:    this.scopes.peakMs[s],
        totalMs:   this.scopes.totalMs[s],
        callCount: this.scopes.callCount[s],
      };
    }
    return out;
  }

  getTopScopes(max) {
    const n = Math.min(max | 0, this.scopes.count);
    if (n <= 0) return [];
    const indices = new Array(this.scopes.count);
    for (let s = 0; s < this.scopes.count; s++) indices[s] = s;
    indices.sort((a, b) => this.scopes.emaMs[b] - this.scopes.emaMs[a]);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const s = indices[i];
      out[i] = {
        name:      this.scopes.name[s],
        emaMs:     this.scopes.emaMs[s],
        lastMs:    this.scopes.lastMs[s],
        peakMs:    this.scopes.peakMs[s],
        callCount: this.scopes.callCount[s],
      };
    }
    return out;
  }

  getHitchStats() {
    return {
      hitches:     this.hitches,
      janks:       this.janks,
      criticals:   this.criticals,
      totalFrames: this.frame,
      hitchRate:   this.frame > 0 ? (this.hitches + this.janks + this.criticals) / this.frame : 0,
      recentCount: this.hitchLog.count,
      totalLogged: this.hitchLog.total,
    };
  }

  getMemoryStats() {
    const m = this.memory;
    return {
      available:        m.available,
      heapUsedMB:       m.getHeapMB(),
      heapPercent:      m.getHeapPercent(),
      gpuMB:            m.getGpuMB(),
      gpuPercent:       m.getGpuPercent(),
      rtBytesInUse:     m.rtBytesInUse,
      rtBytesPeak:      m.rtBytesPeak,
      attrBytesInUse:   m.attrBytesInUse,
      attrBytesPeak:    m.attrBytesPeak,
      registryLive:     m.registryLiveCount,
      registryPeak:     m.registryPeakCount,
    };
  }

  getGpuStats() {
    return {
      available:        this.gpu.available,
      lastMs:           this.gpu.lastGpuMs,
      averageMs:        this.gpu.getAverageGpuMs(),
      sampleCount:      this.gpu.sampleCount,
    };
  }

  getStats() {
    const frameStats = this.computeFrameStats();
    return {
      version:      'profiler_v1',
      frame:        this.frame,
      elapsedMs:    this.elapsedMs,
      dtEmaMs:      this.dtEmaMs,
      dtMinMs:      this.dtMinMs,
      dtMaxMs:      this.dtMaxMs,
      dtSumMs:      this.dtSumMs,
      historyCapacity: this.capacity,
      historyCount: this.frameCount,

      frameStats: {
        mean: frameStats.mean,
        min:  frameStats.min,
        max:  frameStats.max,
        p50:  frameStats.p50,
        p95:  frameStats.p95,
        p99:  frameStats.p99,
        fps:  frameStats.fps,
      },

      hitchStats:  this.getHitchStats(),
      domainStats: this.getDomainStats(),
      scopeStats:  this.getScopeStats(),
      memoryStats: this.getMemoryStats(),
      gpuStats:    this.getGpuStats(),

      platform: {
        perfTier:    getEffectiveTierName(),
        platformTier:PLATFORM_CONFIG.tier,
        isAndroid:   DEVICE.isAndroid,
        isMobile:    DEVICE.isMobile,
        gpuVendor:   CAPABILITIES.platform.gpuFamily,
        webgl2:      RAW_CAPS_DEEP.webgl2,
        quirks:      QUIRKS.slice(),
      },
    };
  }

  /* ---------------- export ---------------- */

  exportJSON() {
    const stats = this.getStats();
    return JSON.stringify(stats, null, 2);
  }

  exportCompact() {
    const s = this.computeFrameStats();
    return {
      f:    this.frame,
      fps:  Number(s.fps.toFixed(1)),
      mean: Number(s.mean.toFixed(2)),
      p95:  Number(s.p95.toFixed(2)),
      p99:  Number(s.p99.toFixed(2)),
      max:  Number(s.max.toFixed(2)),
      h:    this.hitches,
      j:    this.janks,
      c:    this.criticals,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    this.frameIndex.fill(0);
    this.frameDtMs.fill(0);
    this.frameDomains.fill(0);
    this.frameScopes.fill(0);
    this.frameGpuMs.fill(0);

    this.head = 0;
    this.frameCount = 0;
    this.frame = 0;
    this.elapsedMs = 0;
    this.dtLastMs = 0;
    this.dtEmaMs = 16.67;
    this.dtMinMs = Number.POSITIVE_INFINITY;
    this.dtMaxMs = 0;
    this.dtSumMs = 0;

    this.hitches = 0;
    this.janks = 0;
    this.criticals = 0;

    this.domainLastMs.fill(0);
    this.domainEmaMs.fill(0);
    this.domainPeakMs.fill(0);
    this.domainTotalMs.fill(0);
    this.domainCalls.fill(0);

    this.scopes.reset();
    this.hitchLog.clear();
    this.markerLog.clear();

    return this;
  }

  dispose() {
    this.reset();
    this.detachRenderer();
    this._listeners.clear();
    this._sortBuf = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultProfiler = null;

export function getDefaultProfiler() {
  if (!_defaultProfiler) _defaultProfiler = new Profiler();
  return _defaultProfiler;
}

export function disposeDefaultProfiler() {
  if (_defaultProfiler) {
    _defaultProfiler.dispose();
    _defaultProfiler = null;
  }
}

/* ------------------------------------------------------------------ */
/* 9. HOT-PATH HELPERS (delegate to default profiler)                 */
/* ------------------------------------------------------------------ */

export function profilerBeginFrame(dtMs) {
  getDefaultProfiler().beginFrame(dtMs);
}

export function profilerRecordDomain(domain, ms) {
  getDefaultProfiler().recordDomain(domain, ms);
}

export function profilerBeginScope(nameOrSlot) {
  return getDefaultProfiler().beginScope(nameOrSlot);
}

export function profilerEndScope() {
  return getDefaultProfiler().endScope();
}

export function profilerMark(nameOrLabelId) {
  return getDefaultProfiler().mark(nameOrLabelId);
}

export function profilerRecordGpuMs(ms) {
  getDefaultProfiler().recordGpuMs(ms);
}

export function profilerReportMemory(name, inUse, peak) {
  getDefaultProfiler().reportMemory(name, inUse, peak);
}

export function profilerStats() {
  return getDefaultProfiler().getStats();
}

/* ------------------------------------------------------------------ */
/* 10. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createProfiler(options = {}) {
  return new Profiler(options);
}

/* ------------------------------------------------------------------ */
/* 11. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Profiler,
  ScopeRegistry,
  HitchLog,
  MarkerLog,
  MemoryTracker,
  GpuTimer,

  createProfiler,
  getDefaultProfiler,
  disposeDefaultProfiler,

  profilerBeginFrame,
  profilerRecordDomain,
  profilerBeginScope,
  profilerEndScope,
  profilerMark,
  profilerRecordGpuMs,
  profilerReportMemory,
  profilerStats,

  HITCH_CATEGORY,
  HITCH_CATEGORY_NAME,
  SAMPLE_KIND,
  HISTORY_CAPACITY,
  MAX_SCOPES,
  MAX_SCOPE_DEPTH,
  MAX_HITCHES,
  MAX_MARKERS,
};

export default _defaultExport;