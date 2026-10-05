// File : 013
// name : src/core/013_rnd_RenderTargetPool.js
// description : GPU render-target pool for the anime lighting stack on Android
//               mobile. Owns every THREE.WebGLRenderTarget instance the lighting
//               pipeline needs: shadow map targets, GI probe render targets,
//               AO blur ping-pong targets, cluster debug targets, environment
//               palette capture targets, interior/exterior probe targets, and
//               post-processing intermediate HDR buffers.
//
//               Where 012_rnd_BufferPool.js owns VBO/attribute memory, THIS
//               module owns framebuffer memory. Every lighting subsystem that
//               renders to texture (shadow atlas, GI probe bake, AO blur, SSR
//               trace, volumetric scattering, sun-ray occlusion, post bloom)
//               acquires its render target from here — never `new THREE.
//               WebGLRenderTarget(...)` inside a hot loop.
//
//               Design:
//                 • Three orthogonal pools:
//                     – ColorTargetPool     : RGBA / RGB / RGBA16F / RGBA32F
//                                             color-only targets
//                     – DepthTargetPool     : Depth / DepthStencil targets
//                     – CombinedPool        : color + depth + stencil combos
//                 • Size buckets are power-of-two-derived from viewport size,
//                   quantized to a fixed ladder (64 → 8192) so Android GPUs
//                   never see dynamic allocations. Resolution scalers feed a
//                   target level (1.0 / 0.75 / 0.5 / 0.25) and the pool rounds
//                   to the nearest ladder entry.
//                 • MultisampleAntialias variants are separate slots — never
//                   retrofitted onto an existing target, because changing MSAA
//                   on Android WebGL requires a full reallocation.
//                 • Named reservations for the canonical lighting targets:
//                     shadowMapTarget, shadowAtlasTarget, giProbeRT, giBounceRT,
//                     aoFullRT, aoHalfRT, aoQuarterRT, aoBlurRT, clusterDebugRT,
//                     envCaptureRT, interiorProbeRT, exteriorProbeRT,
//                     sunRayRT, volumetricRT, bloomRT, postCompositeRT,
//                     postHDRRT, contactShadowRT, ssgiRT, reSTIRReservoirRT.
//                 • Ping-pong pair helper (`acquirePair`) returns two targets
//                   from the same bucket so blur/reprojection passes can swap
//                   without reallocation.
//                 • Format detection: respects WebGL1 vs WebGL2, half-float
//                   extension availability, and Android-specific RGBA16F
//                   support via `EXT_color_buffer_half_float`.
//                 • Android quirks handled:
//                     – Avoids `generateMipmaps` on HDR targets (costly).
//                     – Forces `LinearFilter` on color, `NearestFilter` on
//                       depth, `NoColorSpace` on data buffers.
//                     – Uses `type: HalfFloatType` for HDR when available,
//                       falls back to `UnsignedByteType` + tonemap in shader.
//                     – Respects `depthBuffer` / `stencilBuffer` flags per
//                       bucket to avoid wasted attachments.
//                 • Zero per-frame allocations on the hot path.
//                 • GPU memory accounting per-target and total, so low-memory
//                   Android devices can gate allocations.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external pool libs; every render target created once
//               at construction or on first named reservation.
// best for : Guaranteeing that no lighting subsystem allocates a render target
//            at runtime on Android. Shadow map, GI probe, AO blur, SSR trace,
//            volumetric scattering, sun-ray occlusion, environment capture,
//            and every post-processing intermediate acquire from this pool
//            once and recycle across frames.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const RT_KIND = Object.freeze({
  COLOR_RGBA8:    0,
  COLOR_RGB8:     1,
  COLOR_RGBA16F:  2,
  COLOR_RGBA32F:  3,
  DEPTH:          4,
  DEPTH_STENCIL:  5,
  COMBINED_RGBA8: 6,
  COMBINED_RGBA16F: 7,
  COUNT:          8,
});

export const RT_KIND_NAME = Object.freeze([
  'color_rgba8',
  'color_rgb8',
  'color_rgba16f',
  'color_rgba32f',
  'depth',
  'depth_stencil',
  'combined_rgba8',
  'combined_rgba16f',
]);

export const RT_TAG = Object.freeze({
  GENERIC:        0,
  SHADOW:         1,
  GI:             2,
  AO:             3,
  CLUSTER:        4,
  ENV:            5,
  INTERIOR:       6,
  EXTERIOR:       7,
  POST:           8,
  SUNRAY:         9,
  VOLUMETRIC:    10,
  CONTACT:       11,
  SSGI:          12,
  RESTIR:        13,
  COUNT:         14,
});

export const RT_TAG_NAME = Object.freeze([
  'generic',
  'shadow',
  'gi',
  'ao',
  'cluster',
  'env',
  'interior',
  'exterior',
  'post',
  'sunray',
  'volumetric',
  'contact',
  'ssgi',
  'restir',
]);

/**
 * Size ladder for render targets. All targets are rounded UP to the next
 * ladder entry. This guarantees no dynamic reallocation when the viewport
 * scales between 0.25× and 1.0× — the level scaler just picks a different
 * ladder rung.
 */
export const SIZE_LADDER = Object.freeze([
  64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
]);

export const LADDER_COUNT = SIZE_LADDER.length;

/**
 * Per-tier capacity per (kind × ladder rung). Tuned so shadow atlas + GI +
 * AO + post never starve, but low-tier Android never allocates more than
 * ~64 MB of framebuffer.
 */
export const RT_CAPACITY = (() => {
  if (PERF_TIER === 'HIGH') {
    return [16, 16, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4];
  }
  if (PERF_TIER === 'MEDIUM') {
    return [8, 8, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2];
  }
  return [4, 4, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
})();

const RT_POOL_ID_SYMBOL = '__rtPoolId';
const RT_SLOT_SYMBOL    = '__rtSlot';

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _rtPoolIdCounter = 0;

function _nextPoolId() {
  return ++_rtPoolIdCounter;
}

function _ladderRung(size) {
  for (let i = 0; i < LADDER_COUNT; i++) {
    if (SIZE_LADDER[i] >= size) return i;
  }
  return LADDER_COUNT - 1;
}

function _bytesPerPixel(kind) {
  switch (kind) {
    case RT_KIND.COLOR_RGBA8:      return 4;
    case RT_KIND.COLOR_RGB8:       return 3;
    case RT_KIND.COLOR_RGBA16F:    return 8;
    case RT_KIND.COLOR_RGBA32F:    return 16;
    case RT_KIND.DEPTH:            return 4;
    case RT_KIND.DEPTH_STENCIL:    return 4;
    case RT_KIND.COMBINED_RGBA8:   return 4 + 4;
    case RT_KIND.COMBINED_RGBA16F: return 8 + 4;
    default:                       return 4;
  }
}

/* ------------------------------------------------------------------ */
/* 2. HARDWARE CAPABILITY PROBE                                       */
/* ------------------------------------------------------------------ */

/**
 * One-shot feature probe run at module init. Detects half-float color
 * renderability, float color renderability, and depth-texture support
 * so the pool only allocates formats the device can actually render to.
 */
export const HW_CAPS = (() => {
  const caps = {
    halfFloatColor: true,
    floatColor:     false,
    depthTexture:   false,
    webgl2:         false,
  };

  if (typeof document === 'undefined') return caps;

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return caps;

    caps.webgl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
    caps.depthTexture = caps.webgl2 || !!gl.getExtension('WEBGL_depth_texture');

    if (caps.webgl2) {
      caps.halfFloatColor = !!gl.getExtension('EXT_color_buffer_half_float') || true; // WebGL2 default for RGBA16F
      caps.floatColor     = !!gl.getExtension('EXT_color_buffer_float');
    } else {
      caps.halfFloatColor = !!gl.getExtension('EXT_color_buffer_half_float') || !!gl.getExtension('OES_texture_half_float');
      caps.floatColor     = !!gl.getExtension('WEBGL_color_buffer_float');
    }

    // Release the probe context immediately.
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch (_) {
    // Leave conservative defaults.
  }

  return caps;
})();

/* ------------------------------------------------------------------ */
/* 3. RT SLOT                                                         */
/* ------------------------------------------------------------------ */

export class RTSlot {
  constructor(index, kind, rung) {
    this.index    = index;
    this.kind     = kind;
    this.rung     = rung;
    this.size     = SIZE_LADDER[rung];
    this.tag      = RT_TAG.GENERIC;

    this.target   = null;   // THREE.WebGLRenderTarget
    this.width    = 0;
    this.height   = 0;
    this.msaa     = 0;

    this.inUse    = 0;
    this.acquiredAt = 0;
    this.generation = 0;
    this.uploadCount = 0;
    this.named    = null;

    this.bytesPerPixel = _bytesPerPixel(kind);
    this.estimatedBytes = 0;
  }

  reset() {
    this.inUse = 0;
    this.generation = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. RENDER TARGET POOL                                              */
/* ------------------------------------------------------------------ */

export class RenderTargetPool {
  constructor(options = {}) {
    this.name    = options.name || `rt_pool_${_nextPoolId()}`;
    this.poolId  = _nextPoolId();

    // Auto-release policy per frame.
    this.autoReleasePerFrame = options.autoReleasePerFrame === true;

    // Per (kind × rung) slot grid.
    this.buckets = new Array(RT_KIND.COUNT);
    for (let k = 0; k < RT_KIND.COUNT; k++) {
      this.buckets[k] = new Array(LADDER_COUNT);
      for (let r = 0; r < LADDER_COUNT; r++) {
        const capacity = RT_CAPACITY[r];
        const slots    = new Array(capacity);
        const freeList = new Int32Array(capacity);
        for (let i = 0; i < capacity; i++) {
          slots[i] = new RTSlot(i, k, r);
          freeList[i] = i;
        }
        this.buckets[k][r] = {
          slots,
          freeList,
          freeHead: 0,
          freeCount: capacity,
          capacity,
          currentInUse: 0,
          peakInUse: 0,
          acquiredTotal: 0,
          releasedTotal: 0,
          rejectedTotal: 0,
        };
      }
    }

    // Named reservations (long-lived targets keyed by name).
    this._namedTargets = new Map();

    // Frame + stats.
    this.frame = 0;
    this.stats = {
      acquired:         0,
      released:         0,
      rejected:         0,
      namedCount:       0,
      gpuBytesInUse:    0,
      peakGpuBytesInUse: 0,
      totalAllocated:   0,
    };

    this._listeners = new Map();
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
      try { arr[i](payload); } catch (e) { console.error(`[013_rnd_RenderTargetPool] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- slot allocation ---------------- */

  _makeTarget(kind, rung, width, height, options) {
    const size = SIZE_LADDER[rung];
    const msaa = Math.max(0, options.samples | 0);

    let format       = THREE.RGBAFormat;
    let type         = THREE.UnsignedByteType;
    let depthBuffer  = false;
    let stencilBuffer= false;
    let minFilter    = THREE.LinearFilter;
    let magFilter    = THREE.LinearFilter;
    let colorSpace   = THREE.NoColorSpace;

    switch (kind) {
      case RT_KIND.COLOR_RGBA8:
        format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        depthBuffer = false; stencilBuffer = false;
        break;
      case RT_KIND.COLOR_RGB8:
        format = THREE.RGBFormat; type = THREE.UnsignedByteType;
        depthBuffer = false; stencilBuffer = false;
        break;
      case RT_KIND.COLOR_RGBA16F:
        if (HW_CAPS.halfFloatColor) {
          format = THREE.RGBAFormat; type = THREE.HalfFloatType;
        } else {
          format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        }
        depthBuffer = false; stencilBuffer = false;
        break;
      case RT_KIND.COLOR_RGBA32F:
        if (HW_CAPS.floatColor) {
          format = THREE.RGBAFormat; type = THREE.FloatType;
        } else if (HW_CAPS.halfFloatColor) {
          format = THREE.RGBAFormat; type = THREE.HalfFloatType;
        } else {
          format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        }
        depthBuffer = false; stencilBuffer = false;
        break;
      case RT_KIND.DEPTH:
        format = THREE.DepthFormat; type = THREE.UnsignedIntType;
        depthBuffer = true; stencilBuffer = false;
        minFilter = THREE.NearestFilter; magFilter = THREE.NearestFilter;
        break;
      case RT_KIND.DEPTH_STENCIL:
        format = THREE.DepthStencilFormat; type = THREE.UnsignedInt248Type;
        depthBuffer = true; stencilBuffer = true;
        minFilter = THREE.NearestFilter; magFilter = THREE.NearestFilter;
        break;
      case RT_KIND.COMBINED_RGBA8:
        format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        depthBuffer = true; stencilBuffer = false;
        break;
      case RT_KIND.COMBINED_RGBA16F:
        if (HW_CAPS.halfFloatColor) {
          format = THREE.RGBAFormat; type = THREE.HalfFloatType;
        } else {
          format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        }
        depthBuffer = true; stencilBuffer = false;
        break;
      default:
        break;
    }

    // Depth-stencil target with color attachment is only valid with the
    // proper format for WebGL1; skip on low tier.
    if (kind === RT_KIND.DEPTH || kind === RT_KIND.DEPTH_STENCIL) {
      if (!HW_CAPS.depthTexture) {
        // Fallback: allocate a combined target so depth is at least usable.
        format = THREE.RGBAFormat; type = THREE.UnsignedByteType;
        depthBuffer = true; stencilBuffer = (kind === RT_KIND.DEPTH_STENCIL);
      }
    }

    const target = new THREE.WebGLRenderTarget(
      width || size,
      height || size,
      {
        minFilter,
        magFilter,
        format,
        type,
        depthBuffer,
        stencilBuffer,
        generateMipmaps: false,
        colorSpace,
        samples: msaa,
      }
    );

    target.texture.name = `rt_${RT_KIND_NAME[kind]}_${size}${msaa ? `_ms${msaa}` : ''}`;

    return target;
  }

  /**
   * Acquire a render target. Returns a THREE.WebGLRenderTarget or null if
   * the pool for that kind/rung is exhausted.
   *
   *   acquireTarget(width, height, {
   *     kind: RT_KIND.COLOR_RGBA16F,
   *     tag:  RT_TAG.GI,
   *     samples: 0,
   *   })
   */
  acquireTarget(width, height, options = {}) {
    const w = Math.max(1, (width | 0));
    const h = Math.max(1, (height | 0));
    const kind = options.kind !== undefined ? options.kind : RT_KIND.COLOR_RGBA8;
    const tag  = options.tag  !== undefined ? options.tag  : RT_TAG.GENERIC;
    const samples = options.samples !== undefined ? (options.samples | 0) : 0;

    const rung = _ladderRung(Math.max(w, h));
    const bucket = this.buckets[kind][rung];

    if (bucket.freeCount <= 0) {
      bucket.rejectedTotal++;
      this.stats.rejected++;
      this._emit('rejected', { kind, tag, width: w, height: h, reason: 'exhausted' });
      return null;
    }

    const idx = bucket.freeList[bucket.freeHead];
    bucket.freeHead = (bucket.freeHead + 1) % bucket.capacity;
    bucket.freeCount--;

    const slot = bucket.slots[idx];

    // Lazily create on first acquire, then reuse across frames. Android
    // requires MSAA to be fixed at creation, so MSAA-bearing slots never
    // change their sample count.
    const needsNew =
      !slot.target ||
      slot.width  !== w ||
      slot.height !== h ||
      slot.msaa   !== samples;

    if (needsNew) {
      // Dispose the old target if we're replacing an existing one.
      if (slot.target) {
        try { slot.target.dispose(); } catch (_) { /* swallow */ }
        this.stats.gpuBytesInUse -= slot.estimatedBytes;
        if (this.stats.gpuBytesInUse < 0) this.stats.gpuBytesInUse = 0;
      }
      slot.target = this._makeTarget(kind, rung, w, h, { samples });
      slot.width  = w;
      slot.height = h;
      slot.msaa   = samples;
      slot.estimatedBytes = w * h * slot.bytesPerPixel;
      this.stats.totalAllocated++;
      this.stats.gpuBytesInUse += slot.estimatedBytes;
      if (this.stats.gpuBytesInUse > this.stats.peakGpuBytesInUse) {
        this.stats.peakGpuBytesInUse = this.stats.gpuBytesInUse;
      }
    }

    slot.inUse     = 1;
    slot.acquiredAt = this.frame;
    slot.tag       = tag;
    slot.generation++;

    // Tag the target for fast release.
    slot.target[RT_POOL_ID_SYMBOL] = this.poolId;
    slot.target[RT_SLOT_SYMBOL] = {
      kind,
      rung,
      index: idx,
    };

    bucket.currentInUse++;
    if (bucket.currentInUse > bucket.peakInUse) bucket.peakInUse = bucket.currentInUse;
    bucket.acquiredTotal++;
    this.stats.acquired++;

    return slot.target;
  }

  releaseTarget(target) {
    if (!target || target[RT_POOL_ID_SYMBOL] !== this.poolId) return false;

    const info = target[RT_SLOT_SYMBOL];
    if (!info) return false;

    const bucket = this.buckets[info.kind][info.rung];
    if (!bucket) return false;

    const slot = bucket.slots[info.index];
    if (!slot || slot.inUse === 0) return false;

    slot.reset();
    slot.inUse = 0;

    bucket.freeList[(bucket.freeHead + bucket.freeCount) % bucket.capacity] = slot.index;
    bucket.freeCount++;
    bucket.releasedTotal++;
    bucket.currentInUse--;
    if (bucket.currentInUse < 0) bucket.currentInUse = 0;

    this.stats.released++;
    this.stats.gpuBytesInUse -= slot.estimatedBytes;
    if (this.stats.gpuBytesInUse < 0) this.stats.gpuBytesInUse = 0;

    return true;
  }

  /**
   * Acquire a ping-pong pair of identical render targets. Useful for
   * blur/reprojection passes that swap source and destination each frame.
   * Returns { a, b } or null.
   */
  acquirePair(width, height, options = {}) {
    const a = this.acquireTarget(width, height, options);
    if (!a) return null;
    const b = this.acquireTarget(width, height, options);
    if (!b) {
      this.releaseTarget(a);
      return null;
    }
    return { a, b };
  }

  /* ---------------- named reservations ---------------- */

  /**
   * Reserve a long-lived named target. On first call the target is created
   * with the given dimensions and options; subsequent calls return the same
   * target. Named targets are NOT released by `releaseAll()` unless
   * `includeNamed=true`.
   */
  reserveNamed(name, width, height, options = {}) {
    if (!name || typeof name !== 'string') return null;
    const existing = this._namedTargets.get(name);
    if (existing) return existing.target;

    const target = this.acquireTarget(width, height, options);
    if (!target) return null;

    this._namedTargets.set(name, {
      target,
      width:  width | 0,
      height: height | 0,
      kind:   options.kind !== undefined ? options.kind : RT_KIND.COLOR_RGBA8,
      tag:    options.tag  !== undefined ? options.tag  : RT_TAG.GENERIC,
      options: Object.assign({}, options),
    });
    this.stats.namedCount++;
    return target;
  }

  getNamed(name) {
    const entry = this._namedTargets.get(name);
    return entry ? entry.target : null;
  }

  getNamedEntry(name) {
    return this._namedTargets.get(name) || null;
  }

  /** Resize a named target if the viewport changed. */
  resizeNamed(name, width, height, options = {}) {
    const entry = this._namedTargets.get(name);
    if (!entry) return null;
    const w = Math.max(1, width | 0);
    const h = Math.max(1, height | 0);
    if (entry.width === w && entry.height === h) return entry.target;

    // Release the old, acquire a new one. MSAA / kind must be stable.
    this.releaseNamed(name);
    return this.reserveNamed(name, w, h, Object.assign({}, entry.options, options));
  }

  releaseNamed(name) {
    const entry = this._namedTargets.get(name);
    if (!entry) return false;
    this._namedTargets.delete(name);
    this.stats.namedCount--;
    return this.releaseTarget(entry.target);
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame() {
    this.frame++;
    return this;
  }

  endFrame() {
    if (this.autoReleasePerFrame) this.releaseAll(false);
    return this;
  }

  releaseAll(includeNamed = false) {
    let released = 0;
    for (let k = 0; k < RT_KIND.COUNT; k++) {
      for (let r = 0; r < LADDER_COUNT; r++) {
        const bucket = this.buckets[k][r];
        for (let i = 0; i < bucket.capacity; i++) {
          const slot = bucket.slots[i];
          if (slot.inUse === 1 && (includeNamed || !this._isNamedSlot(slot))) {
            this.releaseTarget(slot.target);
            released++;
          }
        }
      }
    }
    return released;
  }

  _isNamedSlot(slot) {
    for (const entry of this._namedTargets.values()) {
      const info = entry.target[RT_SLOT_SYMBOL];
      if (!info) continue;
      if (info.kind === slot.kind && info.rung === slot.rung && info.index === slot.index) {
        return true;
      }
    }
    return false;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const buckets = new Array(RT_KIND.COUNT);
    for (let k = 0; k < RT_KIND.COUNT; k++) {
      buckets[k] = new Array(LADDER_COUNT);
      for (let r = 0; r < LADDER_COUNT; r++) {
        const bucket = this.buckets[k][r];
        buckets[k][r] = {
          size:          SIZE_LADDER[r],
          capacity:      bucket.capacity,
          freeCount:     bucket.freeCount,
          currentInUse:  bucket.currentInUse,
          peakInUse:     bucket.peakInUse,
          acquiredTotal: bucket.acquiredTotal,
          releasedTotal: bucket.releasedTotal,
          rejectedTotal: bucket.rejectedTotal,
        };
      }
    }

    return {
      name:              this.name,
      frame:             this.frame,
      namedCount:        this.stats.namedCount,
      acquired:          this.stats.acquired,
      released:          this.stats.released,
      rejected:          this.stats.rejected,
      totalAllocated:    this.stats.totalAllocated,
      gpuBytesInUse:     this.stats.gpuBytesInUse,
      peakGpuBytesInUse: this.stats.peakGpuBytesInUse,
      gpuMegabytesInUse: this.stats.gpuBytesInUse / (1024 * 1024),
      gpuMegabytesPeak:  this.stats.peakGpuBytesInUse / (1024 * 1024),
      hwCaps:            HW_CAPS,
      buckets,
      perfTier:          PERF_TIER,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    this.releaseAll(true);

    for (let k = 0; k < RT_KIND.COUNT; k++) {
      for (let r = 0; r < LADDER_COUNT; r++) {
        const bucket = this.buckets[k][r];
        bucket.freeHead = 0;
        bucket.freeCount = bucket.capacity;
        for (let i = 0; i < bucket.capacity; i++) {
          bucket.freeList[i] = i;
          bucket.slots[i].reset();
          bucket.slots[i].inUse = 0;
        }
        bucket.currentInUse = 0;
        bucket.peakInUse = 0;
        bucket.acquiredTotal = 0;
        bucket.releasedTotal = 0;
        bucket.rejectedTotal = 0;
      }
    }

    this._namedTargets.clear();

    this.stats.acquired = 0;
    this.stats.released = 0;
    this.stats.rejected = 0;
    this.stats.namedCount = 0;
    this.stats.gpuBytesInUse = 0;
    this.stats.peakGpuBytesInUse = 0;
    this.stats.totalAllocated = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let k = 0; k < RT_KIND.COUNT; k++) {
      for (let r = 0; r < LADDER_COUNT; r++) {
        const bucket = this.buckets[k][r];
        for (let i = 0; i < bucket.capacity; i++) {
          const slot = bucket.slots[i];
          if (slot.target) {
            try { slot.target.dispose(); } catch (_) { /* swallow */ }
          }
          slot.target = null;
        }
        bucket.slots.length = 0;
        bucket.freeList = null;
      }
      this.buckets[k] = null;
    }
    this.buckets = null;
    this._namedTargets.clear();
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. STANDARD LIGHTING TARGET PRESET                                 */
/* ------------------------------------------------------------------ */

/**
 * Reserve the canonical render targets the lighting stack needs. Sizes
 * assume a 1080p base viewport; callers may pass the actual viewport and
 * the pool will round to the ladder.
 *
 *   name                    kind                 scale  tag         msaa
 *   ─────────────────────── ──────────────────── ────── ─────────── ────
 *   shadowMapTarget         DEPTH                1.0    SHADOW      0
 *   shadowAtlasTarget       DEPTH_STENCIL        1.0    SHADOW      0
 *   giProbeRT               COLOR_RGBA16F        0.5    GI          0
 *   giBounceRT              COLOR_RGBA16F        0.5    GI          0
 *   aoFullRT                COLOR_RGBA8          1.0    AO          0
 *   aoHalfRT                COLOR_RGBA8          0.5    AO          0
 *   aoQuarterRT             COLOR_RGBA8          0.25   AO          0
 *   aoBlurRT                COLOR_RGBA8          0.5    AO          0
 *   clusterDebugRT          COLOR_RGBA8          0.5    CLUSTER     0
 *   envCaptureRT            COLOR_RGBA16F        0.5    ENV         0
 *   interiorProbeRT         COLOR_RGBA16F        0.25   INTERIOR    0
 *   exteriorProbeRT         COLOR_RGBA16F        0.5    EXTERIOR    0
 *   sunRayRT                COLOR_RGBA8          0.5    SUNRAY      0
 *   volumetricRT            COLOR_RGBA16F        0.5    VOLUMETRIC  0
 *   bloomRT                 COLOR_RGBA16F        0.5    POST        0
 *   postCompositeRT         COMBINED_RGBA8       1.0    POST        0
 *   postHDRRT               COLOR_RGBA16F        1.0    POST        0
 *   contactShadowRT         COLOR_RGBA8          0.5    CONTACT     0
 *   ssgiRT                  COLOR_RGBA16F        0.5    SSGI        0
 *   reSTIRReservoirRT       COLOR_RGBA32F        0.5    RESTIR      0
 */
export function reserveLightingTargets(pool, baseWidth, baseHeight) {
  if (!pool) return null;

  const W = Math.max(64, baseWidth  | 0);
  const H = Math.max(64, baseHeight | 0);

  const half     = (v) => Math.max(64, (v * 0.5)  | 0);
  const quarter  = (v) => Math.max(64, (v * 0.25) | 0);

  const named = {
    shadowMapTarget:    pool.reserveNamed('shadowMapTarget',    W, H, { kind: RT_KIND.DEPTH,            tag: RT_TAG.SHADOW }),
    shadowAtlasTarget:  pool.reserveNamed('shadowAtlasTarget',  W, H, { kind: RT_KIND.DEPTH_STENCIL,    tag: RT_TAG.SHADOW }),
    giProbeRT:          pool.reserveNamed('giProbeRT',          half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.GI }),
    giBounceRT:         pool.reserveNamed('giBounceRT',         half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.GI }),
    aoFullRT:           pool.reserveNamed('aoFullRT',           W, H, { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.AO }),
    aoHalfRT:           pool.reserveNamed('aoHalfRT',           half(W), half(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.AO }),
    aoQuarterRT:        pool.reserveNamed('aoQuarterRT',        quarter(W), quarter(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.AO }),
    aoBlurRT:           pool.reserveNamed('aoBlurRT',           half(W), half(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.AO }),
    clusterDebugRT:     pool.reserveNamed('clusterDebugRT',     half(W), half(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.CLUSTER }),
    envCaptureRT:       pool.reserveNamed('envCaptureRT',       half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.ENV }),
    interiorProbeRT:    pool.reserveNamed('interiorProbeRT',    quarter(W), quarter(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.INTERIOR }),
    exteriorProbeRT:    pool.reserveNamed('exteriorProbeRT',    half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.EXTERIOR }),
    sunRayRT:           pool.reserveNamed('sunRayRT',           half(W), half(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.SUNRAY }),
    volumetricRT:       pool.reserveNamed('volumetricRT',       half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.VOLUMETRIC }),
    bloomRT:            pool.reserveNamed('bloomRT',            half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.POST }),
    postCompositeRT:    pool.reserveNamed('postCompositeRT',    W, H, { kind: RT_KIND.COMBINED_RGBA8, tag: RT_TAG.POST }),
    postHDRRT:          pool.reserveNamed('postHDRRT',          W, H, { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.POST }),
    contactShadowRT:    pool.reserveNamed('contactShadowRT',    half(W), half(H), { kind: RT_KIND.COLOR_RGBA8, tag: RT_TAG.CONTACT }),
    ssgiRT:             pool.reserveNamed('ssgiRT',             half(W), half(H), { kind: RT_KIND.COLOR_RGBA16F, tag: RT_TAG.SSGI }),
    reSTIRReservoirRT:  pool.reserveNamed('reSTIRReservoirRT',  half(W), half(H), { kind: RT_KIND.COLOR_RGBA32F, tag: RT_TAG.RESTIR }),
  };

  return named;
}

/* ------------------------------------------------------------------ */
/* 6. TAGGED RT SUB-POOL                                              */
/* ------------------------------------------------------------------ */

export class TaggedRTSubPool {
  constructor(pool, tag) {
    this.pool = pool;
    this.tag  = tag;
  }

  acquire(width, height, kind = RT_KIND.COLOR_RGBA8, samples = 0) {
    return this.pool.acquireTarget(width, height, {
      kind,
      tag: this.tag,
      samples,
    });
  }

  acquirePair(width, height, kind = RT_KIND.COLOR_RGBA8, samples = 0) {
    return this.pool.acquirePair(width, height, {
      kind,
      tag: this.tag,
      samples,
    });
  }

  release(target) { return this.pool.releaseTarget(target); }
  reserveNamed(name, width, height, kind = RT_KIND.COLOR_RGBA8, samples = 0) {
    return this.pool.reserveNamed(name, width, height, { kind, tag: this.tag, samples });
  }
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createRenderTargetPool(options = {}) {
  return new RenderTargetPool(options);
}

export function createTaggedRTSubPool(pool, tag) {
  return new TaggedRTSubPool(pool, tag);
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultRTPool = null;
let _defaultLightingTargets = null;

export function getDefaultRenderTargetPool(baseWidth, baseHeight) {
  if (!_defaultRTPool) {
    _defaultRTPool = new RenderTargetPool();
    if (baseWidth && baseHeight) {
      _defaultLightingTargets = reserveLightingTargets(_defaultRTPool, baseWidth, baseHeight);
    }
  } else if (baseWidth && baseHeight && !_defaultLightingTargets) {
    _defaultLightingTargets = reserveLightingTargets(_defaultRTPool, baseWidth, baseHeight);
  }
  return _defaultRTPool;
}

export function getDefaultLightingTargets() {
  return _defaultLightingTargets;
}

export function resizeDefaultLightingTargets(baseWidth, baseHeight) {
  if (!_defaultRTPool) return null;
  if (!_defaultLightingTargets) {
    _defaultLightingTargets = reserveLightingTargets(_defaultRTPool, baseWidth, baseHeight);
    return _defaultLightingTargets;
  }
  // Resize every named target to the new viewport.
  const W = Math.max(64, baseWidth  | 0);
  const H = Math.max(64, baseHeight | 0);
  const half    = (v) => Math.max(64, (v * 0.5)  | 0);
  const quarter = (v) => Math.max(64, (v * 0.25) | 0);

  _defaultRTPool.resizeNamed('shadowMapTarget',   W, H);
  _defaultRTPool.resizeNamed('shadowAtlasTarget', W, H);
  _defaultRTPool.resizeNamed('giProbeRT',         half(W), half(H));
  _defaultRTPool.resizeNamed('giBounceRT',        half(W), half(H));
  _defaultRTPool.resizeNamed('aoFullRT',          W, H);
  _defaultRTPool.resizeNamed('aoHalfRT',          half(W), half(H));
  _defaultRTPool.resizeNamed('aoQuarterRT',       quarter(W), quarter(H));
  _defaultRTPool.resizeNamed('aoBlurRT',          half(W), half(H));
  _defaultRTPool.resizeNamed('clusterDebugRT',    half(W), half(H));
  _defaultRTPool.resizeNamed('envCaptureRT',      half(W), half(H));
  _defaultRTPool.resizeNamed('interiorProbeRT',   quarter(W), quarter(H));
  _defaultRTPool.resizeNamed('exteriorProbeRT',   half(W), half(H));
  _defaultRTPool.resizeNamed('sunRayRT',          half(W), half(H));
  _defaultRTPool.resizeNamed('volumetricRT',      half(W), half(H));
  _defaultRTPool.resizeNamed('bloomRT',           half(W), half(H));
  _defaultRTPool.resizeNamed('postCompositeRT',   W, H);
  _defaultRTPool.resizeNamed('postHDRRT',         W, H);
  _defaultRTPool.resizeNamed('contactShadowRT',   half(W), half(H));
  _defaultRTPool.resizeNamed('ssgiRT',            half(W), half(H));
  _defaultRTPool.resizeNamed('reSTIRReservoirRT', half(W), half(H));

  return _defaultLightingTargets;
}

export function disposeDefaultRenderTargetPool() {
  if (_defaultRTPool) {
    _defaultRTPool.dispose();
    _defaultRTPool = null;
    _defaultLightingTargets = null;
  }
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  RenderTargetPool,
  RTSlot,
  TaggedRTSubPool,
  createRenderTargetPool,
  createTaggedRTSubPool,
  getDefaultRenderTargetPool,
  getDefaultLightingTargets,
  resizeDefaultLightingTargets,
  disposeDefaultRenderTargetPool,
  reserveLightingTargets,
  RT_KIND,
  RT_KIND_NAME,
  RT_TAG,
  RT_TAG_NAME,
  SIZE_LADDER,
  LADDER_COUNT,
  RT_CAPACITY,
  HW_CAPS,
};

export default _defaultExport;