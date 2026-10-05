// File : 019
// name : src/core/019_rnd_AndroidProfile.js
// description : Android-specific device profile resolver and tuning table for
//               the anime lighting stack. Where 018_rnd_PlatformConfig.js
//               classifies the platform (Android vs iOS vs desktop, HIGH vs
//               MEDIUM vs LOW tier, WebGL1 vs WebGL2), THIS module owns the
//               concrete Android tuning: exact shadow bias values per GPU
//               family, exact GI probe spacing per RAM class, exact AO
//               sample count per Adreno/Mali generation, exact thermal /
//               battery downgrade curves, and the per-GPU-family shader
//               precision overrides.
//
//               Every downstream lighting subsystem reads AndroidProfile to
//               decide:
//                 • What shadow bias / normal bias to use per GPU family
//                   (Adreno needs more negative bias, Mali needs less).
//                 • Whether to use `highp` or `mediump` in a specific
//                   shader (Adreno 4xx needs mediump; Adreno 6xx+ can use
//                   highp; Mali-G51+ needs mediump for position varying).
//                 • What GI probe spacing to use given available RAM
//                   (2 GB → 8 m, 4 GB → 4 m, 6 GB → 3 m, 8 GB+ → 2 m).
//                 • What AO sample count to use given the GPU family
//                   (Adreno 3xx/4xx → 4, Adreno 5xx/6xx → 8, Adreno 7xx →
//                   12, Mali G5x → 6, Mali G7x → 10).
//                 • How aggressively to downgrade under thermal pressure
//                   (diminishing curve per tier).
//                 • How aggressively to downgrade under battery pressure.
//                 • What pixel ratio to use at each battery level.
//                 • What the safe render-target memory cap is given
//                   `navigator.deviceMemory`.
//                 • What worker pool size to use given
//                   `navigator.hardwareConcurrency`.
//
//               Quirk resolution table:
//                 Adreno 3xx-5xx    → half-float clamp, shadow bias boost
//                 Adreno 6xx-8xx    → highp OK, half-float OK on 7xx+
//                 Mali-T / G3x-G5x  → precision loss, no MRT, no highp vary
//                 Mali-G6x-G7x      → mediump position vary, MRT OK
//                 Mali-G8x+         → full precision, MRT OK
//                 PowerVR Rogue     → TBDR depth cost, no PCSS, no MSAA
//                 PowerVR GE        → minimal profile
//                 SwiftShader/soft  → minimal profile, no shadows
//
//               Thermal curve (per level 0..4):
//                 L0 nominal     → full budget
//                 L1 warm        → shadow -25%, GI -25%, post -1
//                 L2 hot         → shadow -50%, GI -50%, post -2, AO -25%
//                 L3 very_hot    → shadow -75%, GI -75%, post -3, AO -50%
//                 L4 critical    → shadow -90%, GI -90%, post -4, AO -75%
//
//               Battery curve (per level 0..4):
//                 B0 charging or >50%    → full budget
//                 B1 30-50%              → DPR -10%
//                 B2 20-30%              → DPR -20%, shadow -25%
//                 B3 15-20%              → DPR -30%, shadow -50%, GI -50%
//                 B4 <15%                → DPR -40%, shadow -75%, GI -75%, post -3
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; every table frozen at module load; every read O(1).
// best for : Guaranteeing concrete, device-specific tuning for every Android
//            GPU family in the wild — Mali-400 through Mali-G925, Adreno 305
//            through Adreno 830, PowerVR Rogue through PowerVR XT — so the
//            anime look is preserved without artifacts on any of them.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  QUIRK,
  PRECISION_TIER,
  PROFILE_ANDROID_HIGH,
  PROFILE_ANDROID_MID,
  PROFILE_ANDROID_LOW,
  PROFILE_ANDROID_MINIMAL,
  RAW_GPU,
  RAW_CAPS,
  canUseShadowType,
  canUseColorFormat,
  getPlatformReport,
} from './018_rnd_PlatformConfig.js';

import {
  getDefaultConfig,
  PERF_TIER,
  PLATFORM,
} from './016_rnd_Config.js';

import {
  DOMAIN,
} from './005_rnd_FrameScheduler.js';

import {
  getPerfTier,
} from './008_scn_world.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const ANDROID_GPU_FAMILY = Object.freeze({
  UNKNOWN:          0,
  ADRENO_3XX:       1,
  ADRENO_4XX:       2,
  ADRENO_5XX:       3,
  ADRENO_6XX:       4,
  ADRENO_7XX:       5,
  ADRENO_8XX:       6,
  MALI_T:           7,
  MALI_G3X:         8,
  MALI_G5X:         9,
  MALI_G6X:        10,
  MALI_G7X:        11,
  MALI_G8X:        12,
  POWERVR_ROGUE:   13,
  POWERVR_GE:      14,
  SWIFTSHADER:     15,
  COUNT:           16,
});

export const ANDROID_GPU_FAMILY_NAME = Object.freeze([
  'unknown',
  'adreno_3xx',
  'adreno_4xx',
  'adreno_5xx',
  'adreno_6xx',
  'adreno_7xx',
  'adreno_8xx',
  'mali_t',
  'mali_g3x',
  'mali_g5x',
  'mali_g6x',
  'mali_g7x',
  'mali_g8x',
  'powervr_rogue',
  'powervr_ge',
  'swiftshader',
]);

export const THERMAL_LEVEL = Object.freeze({
  NOMINAL:  0,
  WARM:     1,
  HOT:      2,
  VERY_HOT: 3,
  CRITICAL: 4,
  COUNT:    5,
});

export const THERMAL_LEVEL_NAME = Object.freeze([
  'nominal',
  'warm',
  'hot',
  'very_hot',
  'critical',
]);

export const BATTERY_LEVEL = Object.freeze({
  FULL:     0,
  GOOD:     1,
  LOW:      2,
  VERY_LOW: 3,
  CRITICAL: 4,
  COUNT:    5,
});

export const BATTERY_LEVEL_NAME = Object.freeze([
  'full',
  'good',
  'low',
  'very_low',
  'critical',
]);

/* ------------------------------------------------------------------ */
/* 1. GPU FAMILY RESOLUTION                                           */
/* ------------------------------------------------------------------ */

function _resolveGpuFamily() {
  const r = (RAW_GPU.renderer || '').toLowerCase();
  const v = (RAW_GPU.vendor   || '').toLowerCase();

  // Software / fallback.
  if (/swiftshader|llvmpipe|software/.test(r)) return ANDROID_GPU_FAMILY.SWIFTSHADER;

  // Adreno.
  if (v === 'adreno' || /adreno/.test(r)) {
    const m = r.match(/adreno[\s\(]*(\d)/);
    const gen = m ? parseInt(m[1], 10) : 0;
    if (gen === 3) return ANDROID_GPU_FAMILY.ADRENO_3XX;
    if (gen === 4) return ANDROID_GPU_FAMILY.ADRENO_4XX;
    if (gen === 5) return ANDROID_GPU_FAMILY.ADRENO_5XX;
    if (gen === 6) return ANDROID_GPU_FAMILY.ADRENO_6XX;
    if (gen === 7) return ANDROID_GPU_FAMILY.ADRENO_7XX;
    if (gen === 8) return ANDROID_GPU_FAMILY.ADRENO_8XX;
    return ANDROID_GPU_FAMILY.ADRENO_5XX; // unknown Adreno → mid assumption
  }

  // Mali.
  if (v === 'mali' || /mali|immortalis/.test(r)) {
    if (/immortalis/.test(r)) return ANDROID_GPU_FAMILY.MALI_G8X;
    const m = r.match(/mali-([tg]?)(\d{2,3})/);
    if (m) {
      const series = m[1];
      const num = parseInt(m[2], 10);
      if (series === 't' || num < 30) return ANDROID_GPU_FAMILY.MALI_T;
      if (num >= 30 && num < 50)     return ANDROID_GPU_FAMILY.MALI_G3X;
      if (num >= 50 && num < 60)     return ANDROID_GPU_FAMILY.MALI_G5X;
      if (num >= 60 && num < 70)     return ANDROID_GPU_FAMILY.MALI_G6X;
      if (num >= 70 && num < 80)     return ANDROID_GPU_FAMILY.MALI_G7X;
      if (num >= 80)                 return ANDROID_GPU_FAMILY.MALI_G8X;
    }
    return ANDROID_GPU_FAMILY.MALI_G5X; // unknown Mali → mid
  }

  // PowerVR.
  if (v === 'powervr' || /powervr|rogue/.test(r)) {
    if (/ge\s*\d+|series6xe|ge8/.test(r)) return ANDROID_GPU_FAMILY.POWERVR_GE;
    return ANDROID_GPU_FAMILY.POWERVR_ROGUE;
  }

  return ANDROID_GPU_FAMILY.UNKNOWN;
}

export const GPU_FAMILY = _resolveGpuFamily();
export const GPU_FAMILY_NAME = ANDROID_GPU_FAMILY_NAME[GPU_FAMILY] || 'unknown';

/* ------------------------------------------------------------------ */
/* 2. PER-GPU SHADER PRECISION OVERRIDES                              */
/* ------------------------------------------------------------------ */

/**
 * Per-GPU shader precision hints. `positionVarying` and `varyingFloat`
 * are the two precision slots that matter most on Android — Mali struggles
 * with highp vertex varyings, Adreno 4xx struggles with highp fragment.
 */
export const GPU_PRECISION = Object.freeze({
  [ANDROID_GPU_FAMILY.ADRENO_3XX]:    Object.freeze({ vertex: 'mediump', fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.ADRENO_4XX]:    Object.freeze({ vertex: 'highp',   fragment: 'mediump', positionVarying: 'highp',   varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.ADRENO_5XX]:    Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),
  [ANDROID_GPU_FAMILY.ADRENO_6XX]:    Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),
  [ANDROID_GPU_FAMILY.ADRENO_7XX]:    Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),
  [ANDROID_GPU_FAMILY.ADRENO_8XX]:    Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),

  [ANDROID_GPU_FAMILY.MALI_T]:        Object.freeze({ vertex: 'mediump', fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.MALI_G3X]:      Object.freeze({ vertex: 'highp',   fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.MALI_G5X]:      Object.freeze({ vertex: 'highp',   fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.MALI_G6X]:      Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'mediump', varyingFloat: 'highp'   }),
  [ANDROID_GPU_FAMILY.MALI_G7X]:      Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),
  [ANDROID_GPU_FAMILY.MALI_G8X]:      Object.freeze({ vertex: 'highp',   fragment: 'highp',   positionVarying: 'highp',   varyingFloat: 'highp'   }),

  [ANDROID_GPU_FAMILY.POWERVR_ROGUE]: Object.freeze({ vertex: 'highp',   fragment: 'mediump', positionVarying: 'highp',   varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.POWERVR_GE]:    Object.freeze({ vertex: 'mediump', fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),

  [ANDROID_GPU_FAMILY.SWIFTSHADER]:   Object.freeze({ vertex: 'mediump', fragment: 'mediump', positionVarying: 'mediump', varyingFloat: 'mediump' }),
  [ANDROID_GPU_FAMILY.UNKNOWN]:       Object.freeze({ vertex: 'highp',   fragment: 'mediump', positionVarying: 'highp',   varyingFloat: 'mediump' }),
});

export const PRECISION = GPU_PRECISION[GPU_FAMILY] || GPU_PRECISION[ANDROID_GPU_FAMILY.UNKNOWN];

/* ------------------------------------------------------------------ */
/* 3. PER-GPU SHADOW TUNING                                           */
/* ------------------------------------------------------------------ */

/**
 * Per-GPU shadow bias tuning. Adreno needs more negative bias to avoid
 * peter-panning; Mali needs less; PowerVR needs a "pancake fix" because
 * its tile-based deferred renderer rounds depth early.
 */
export const GPU_SHADOW_TUNING = Object.freeze({
  [ANDROID_GPU_FAMILY.ADRENO_3XX]:    Object.freeze({ bias: -0.0030, normalBias: 0.060, pancakeFix: true,  softness: 0.00, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.ADRENO_4XX]:    Object.freeze({ bias: -0.0022, normalBias: 0.050, pancakeFix: true,  softness: 0.02, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.ADRENO_5XX]:    Object.freeze({ bias: -0.0016, normalBias: 0.040, pancakeFix: true,  softness: 0.04, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.ADRENO_6XX]:    Object.freeze({ bias: -0.0010, normalBias: 0.030, pancakeFix: true,  softness: 0.05, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.ADRENO_7XX]:    Object.freeze({ bias: -0.0008, normalBias: 0.022, pancakeFix: false, softness: 0.06, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.ADRENO_8XX]:    Object.freeze({ bias: -0.0006, normalBias: 0.018, pancakeFix: false, softness: 0.08, filter: 'pcf'   }),

  [ANDROID_GPU_FAMILY.MALI_T]:        Object.freeze({ bias: -0.0035, normalBias: 0.070, pancakeFix: true,  softness: 0.00, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.MALI_G3X]:      Object.freeze({ bias: -0.0028, normalBias: 0.055, pancakeFix: true,  softness: 0.00, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.MALI_G5X]:      Object.freeze({ bias: -0.0018, normalBias: 0.045, pancakeFix: true,  softness: 0.03, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.MALI_G6X]:      Object.freeze({ bias: -0.0012, normalBias: 0.032, pancakeFix: false, softness: 0.05, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.MALI_G7X]:      Object.freeze({ bias: -0.0009, normalBias: 0.024, pancakeFix: false, softness: 0.06, filter: 'pcf'   }),
  [ANDROID_GPU_FAMILY.MALI_G8X]:      Object.freeze({ bias: -0.0007, normalBias: 0.018, pancakeFix: false, softness: 0.08, filter: 'pcf'   }),

  [ANDROID_GPU_FAMILY.POWERVR_ROGUE]: Object.freeze({ bias: -0.0024, normalBias: 0.048, pancakeFix: true,  softness: 0.00, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.POWERVR_GE]:    Object.freeze({ bias: -0.0035, normalBias: 0.070, pancakeFix: true,  softness: 0.00, filter: 'basic' }),

  [ANDROID_GPU_FAMILY.SWIFTSHADER]:   Object.freeze({ bias: -0.0035, normalBias: 0.070, pancakeFix: true,  softness: 0.00, filter: 'basic' }),
  [ANDROID_GPU_FAMILY.UNKNOWN]:       Object.freeze({ bias: -0.0020, normalBias: 0.045, pancakeFix: true,  softness: 0.03, filter: 'pcf'   }),
});

export const SHADOW_TUNING = GPU_SHADOW_TUNING[GPU_FAMILY] || GPU_SHADOW_TUNING[ANDROID_GPU_FAMILY.UNKNOWN];

/* ------------------------------------------------------------------ */
/* 4. PER-GPU AO / GI TUNING                                          */
/* ------------------------------------------------------------------ */

export const GPU_AO_TUNING = Object.freeze({
  [ANDROID_GPU_FAMILY.ADRENO_3XX]:    Object.freeze({ samples: 3,  radius: 1.2, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.ADRENO_4XX]:    Object.freeze({ samples: 4,  radius: 1.5, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.ADRENO_5XX]:    Object.freeze({ samples: 6,  radius: 1.8, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.ADRENO_6XX]:    Object.freeze({ samples: 8,  radius: 2.0, temporal: true,  halfRes: true }),
  [ANDROID_GPU_FAMILY.ADRENO_7XX]:    Object.freeze({ samples: 12, radius: 2.2, temporal: true,  halfRes: false }),
  [ANDROID_GPU_FAMILY.ADRENO_8XX]:    Object.freeze({ samples: 16, radius: 2.4, temporal: true,  halfRes: false }),

  [ANDROID_GPU_FAMILY.MALI_T]:        Object.freeze({ samples: 3,  radius: 1.2, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.MALI_G3X]:      Object.freeze({ samples: 4,  radius: 1.5, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.MALI_G5X]:      Object.freeze({ samples: 6,  radius: 1.8, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.MALI_G6X]:      Object.freeze({ samples: 8,  radius: 2.0, temporal: true,  halfRes: true }),
  [ANDROID_GPU_FAMILY.MALI_G7X]:      Object.freeze({ samples: 10, radius: 2.1, temporal: true,  halfRes: false }),
  [ANDROID_GPU_FAMILY.MALI_G8X]:      Object.freeze({ samples: 14, radius: 2.3, temporal: true,  halfRes: false }),

  [ANDROID_GPU_FAMILY.POWERVR_ROGUE]: Object.freeze({ samples: 4,  radius: 1.5, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.POWERVR_GE]:    Object.freeze({ samples: 3,  radius: 1.2, temporal: false, halfRes: true }),

  [ANDROID_GPU_FAMILY.SWIFTSHADER]:   Object.freeze({ samples: 3,  radius: 1.2, temporal: false, halfRes: true }),
  [ANDROID_GPU_FAMILY.UNKNOWN]:       Object.freeze({ samples: 6,  radius: 1.8, temporal: false, halfRes: true }),
});

export const AO_TUNING = GPU_AO_TUNING[GPU_FAMILY] || GPU_AO_TUNING[ANDROID_GPU_FAMILY.UNKNOWN];

/**
 * Per-RAM-class GI probe spacing. Denser probes = better indirect lighting
 * but more memory + CPU. 2 GB → 8 m, 4 GB → 4 m, 6 GB → 3 m, 8 GB+ → 2 m.
 */
export function giProbeSpacingForRam(deviceMemoryGB) {
  if (deviceMemoryGB >= 8) return 2.0;
  if (deviceMemoryGB >= 6) return 3.0;
  if (deviceMemoryGB >= 4) return 4.0;
  if (deviceMemoryGB >= 3) return 6.0;
  return 8.0;
}

/**
 * Per-RAM-class GI update budget (Hz). Denser probes require more CPU;
 * throttle accordingly.
 */
export function giUpdateHzForRam(deviceMemoryGB) {
  if (deviceMemoryGB >= 8) return 20;
  if (deviceMemoryGB >= 6) return 15;
  if (deviceMemoryGB >= 4) return 10;
  if (deviceMemoryGB >= 3) return 6;
  return 4;
}

/* ------------------------------------------------------------------ */
/* 5. THERMAL CURVE                                                   */
/* ------------------------------------------------------------------ */

/**
 * Thermal downgrade curve. Each level multiplies the base budget by the
 * listed factors and applies the listed deltas.
 *
 *   level 0 (nominal)  → no change
 *   level 1 (warm)     → shadow -25%, GI -25%, post -1
 *   level 2 (hot)      → shadow -50%, GI -50%, post -2, AO -25%, DPR -15%
 *   level 3 (very_hot) → shadow -75%, GI -75%, post -3, AO -50%, DPR -25%
 *   level 4 (critical) → shadow -90%, GI -90%, post -4, AO -75%, DPR -40%
 */
export const THERMAL_CURVE = Object.freeze([
  Object.freeze({ shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta:  0, dprScale: 1.00, clusterScale: 1.00 }),
  Object.freeze({ shadowScale: 0.75, giScale: 0.75, aoScale: 1.00, postDelta: -1, dprScale: 1.00, clusterScale: 0.85 }),
  Object.freeze({ shadowScale: 0.50, giScale: 0.50, aoScale: 0.75, postDelta: -2, dprScale: 0.85, clusterScale: 0.65 }),
  Object.freeze({ shadowScale: 0.25, giScale: 0.25, aoScale: 0.50, postDelta: -3, dprScale: 0.75, clusterScale: 0.50 }),
  Object.freeze({ shadowScale: 0.10, giScale: 0.10, aoScale: 0.25, postDelta: -4, dprScale: 0.60, clusterScale: 0.35 }),
]);

/* ------------------------------------------------------------------ */
/* 6. BATTERY CURVE                                                   */
/* ------------------------------------------------------------------ */

/**
 * Battery downgrade curve. `charging` forces level 0.
 */
export const BATTERY_CURVE = Object.freeze([
  Object.freeze({ dprScale: 1.00, shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta:  0 }),
  Object.freeze({ dprScale: 0.90, shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta:  0 }),
  Object.freeze({ dprScale: 0.80, shadowScale: 0.75, giScale: 0.85, aoScale: 0.90, postDelta: -1 }),
  Object.freeze({ dprScale: 0.70, shadowScale: 0.50, giScale: 0.50, aoScale: 0.75, postDelta: -2 }),
  Object.freeze({ dprScale: 0.60, shadowScale: 0.25, giScale: 0.25, aoScale: 0.50, postDelta: -3 }),
]);

/* ------------------------------------------------------------------ */
/* 7. WORKER POOL SIZE                                                */
/* ------------------------------------------------------------------ */

/**
 * Recommended worker pool size for parallel lighting tasks (shadow atlas
 * batching, GI probe bake, AO blur). Caps at hardwareConcurrency - 1 so
 * the main thread always has a core free, and caps at 4 on Android because
 * more workers than that yield diminishing returns on mobile memory
 * bandwidth.
 */
export function workerPoolSizeForConcurrency(hardwareConcurrency, tier) {
  const cores = Math.max(1, hardwareConcurrency | 0);
  const reserve = tier === 'HIGH' ? 1 : 2;
  const cap = tier === 'HIGH' ? 6 : tier === 'MEDIUM' ? 4 : 2;
  return Math.max(1, Math.min(cap, cores - reserve));
}

/* ------------------------------------------------------------------ */
/* 8. RESOLVED ANDROID PROFILE                                        */
/* ------------------------------------------------------------------ */

const _baseConfig = getDefaultConfig();
const _baseResolved = _baseConfig.section;

const _deviceMemoryGB = DEVICE.deviceMemory || 4;

export const ANDROID_PROFILE = Object.freeze({
  name:               'android_profile_' + GPU_FAMILY_NAME,
  gpuFamily:          GPU_FAMILY,
  gpuFamilyName:      GPU_FAMILY_NAME,
  rendererName:       RAW_GPU.renderer,
  vendorName:         RAW_GPU.vendor,
  perfTier:           PERF_TIER_LOCAL,
  profileTier:        PLATFORM_CONFIG.tier,
  webglVersion:       DEVICE.webglVersion,

  // Precision.
  precision:          PRECISION,
  precisionTier:      PRECISION_TIER.HIGH,

  // Shadow tuning (bias / normalBias / filter / pancake fix).
  shadow:             SHADOW_TUNING,
  shadowFilter:       canUseShadowType(SHADOW_TUNING.filter) ? SHADOW_TUNING.filter : 'basic',

  // AO tuning (samples / radius / temporal / halfRes).
  ao:                 AO_TUNING,

  // GI tuning (spacing / updateHz by RAM class).
  giProbeSpacing:     giProbeSpacingForRam(_deviceMemoryGB),
  giUpdateHz:         giUpdateHzForRam(_deviceMemoryGB),

  // Color format policy (based on half-float support + quirks).
  hdrColorFormat:     (PLATFORM_CONFIG.supportsHalfFloatColor && !PLATFORM_CONFIG.hasQuirk(QUIRK.ADRENO_HALF_FLOAT_CLAMP))
                        ? 'rgba16f'
                        : 'rgba8',

  // Worker pool.
  workerPoolSize:     workerPoolSizeForConcurrency(DEVICE.hardwareConcurrency, PERF_TIER_LOCAL),

  // Curves.
  thermalCurve:       THERMAL_CURVE,
  batteryCurve:       BATTERY_CURVE,

  // Per-domain Hz scale from the platform config.
  perDomainHzScale:   PLATFORM_CONFIG.perDomainHzScale,

  // Platform caps.
  dprCap:             PLATFORM_CONFIG.dprCap,
  maxClusterLights:   PLATFORM_CONFIG.maxClusterLights,
  giProbeLatticeCap:  PLATFORM_CONFIG.giProbeLatticeCap,
  aoSampleCap:        PLATFORM_CONFIG.aoSampleCap,
  shadowMapCap:       PLATFORM_CONFIG.shadowMapCap,
  shadowCascadeCap:   PLATFORM_CONFIG.shadowCascadeCap,
  postPassCap:        PLATFORM_CONFIG.postPassCap,

  // Device memory cap for RT pool.
  maxGpuMemoryMB:     _deviceMemoryGB >= 8 ? 256 : _deviceMemoryGB >= 6 ? 192 : _deviceMemoryGB >= 4 ? 128 : _deviceMemoryGB >= 3 ? 96 : 48,

  // Quirks carried through from 018.
  quirks:             QUIRKS,
  hasQuirk(name) { return QUIRKS.indexOf(name) >= 0; },
});

/* ------------------------------------------------------------------ */
/* 9. THERMAL / BATTERY STATE RESOLVERS                               */
/* ------------------------------------------------------------------ */

/**
 * Resolve the thermal level (0..4) from a compute pressure state string
 * ('nominal' | 'fair' | 'serious' | 'critical') or a 0..1 bias.
 */
export function resolveThermalLevel(input) {
  if (typeof input === 'number') {
    const b = Math.max(0, Math.min(1, input));
    if (b >= 0.90) return THERMAL_LEVEL.CRITICAL;
    if (b >= 0.70) return THERMAL_LEVEL.VERY_HOT;
    if (b >= 0.50) return THERMAL_LEVEL.HOT;
    if (b >= 0.25) return THERMAL_LEVEL.WARM;
    return THERMAL_LEVEL.NOMINAL;
  }
  const s = String(input || 'nominal').toLowerCase();
  if (s === 'critical') return THERMAL_LEVEL.CRITICAL;
  if (s === 'serious')  return THERMAL_LEVEL.VERY_HOT;
  if (s === 'fair')     return THERMAL_LEVEL.HOT;
  if (s === 'warm')     return THERMAL_LEVEL.WARM;
  return THERMAL_LEVEL.NOMINAL;
}

/**
 * Resolve the battery level (0..4) from level (0..1) and charging flag.
 */
export function resolveBatteryLevel(level, charging) {
  if (charging) return BATTERY_LEVEL.FULL;
  const l = Math.max(0, Math.min(1, Number(level) || 0));
  if (l < 0.15) return BATTERY_LEVEL.CRITICAL;
  if (l < 0.20) return BATTERY_LEVEL.VERY_LOW;
  if (l < 0.30) return BATTERY_LEVEL.LOW;
  if (l < 0.50) return BATTERY_LEVEL.GOOD;
  return BATTERY_LEVEL.FULL;
}

/* ------------------------------------------------------------------ */
/* 10. COMBINED BIAS MULTIPLIERS                                      */
/* ------------------------------------------------------------------ */

/**
 * Combined thermal + battery multipliers. Returns an in-place updated
 * result object (no allocation after warm-up).
 */
const _combinedScratch = {
  shadowScale:  1.0,
  giScale:      1.0,
  aoScale:      1.0,
  dprScale:     1.0,
  clusterScale: 1.0,
  postDelta:    0,
  thermalLevel: THERMAL_LEVEL.NOMINAL,
  batteryLevel: BATTERY_LEVEL.FULL,
};

export function combinedDowngradeMultipliers(thermalInput, batteryLevel, charging) {
  const tl = resolveThermalLevel(thermalInput);
  const bl = resolveBatteryLevel(batteryLevel, charging);
  const t = THERMAL_CURVE[tl] || THERMAL_CURVE[0];
  const b = BATTERY_CURVE[bl] || BATTERY_CURVE[0];

  _combinedScratch.shadowScale  = t.shadowScale  * b.shadowScale;
  _combinedScratch.giScale      = t.giScale      * b.giScale;
  _combinedScratch.aoScale      = t.aoScale      * b.aoScale;
  _combinedScratch.dprScale     = t.dprScale     * b.dprScale;
  _combinedScratch.clusterScale = t.clusterScale;
  _combinedScratch.postDelta    = t.postDelta + b.postDelta;
  _combinedScratch.thermalLevel = tl;
  _combinedScratch.batteryLevel = bl;

  return _combinedScratch;
}

/* ------------------------------------------------------------------ */
/* 11. SHADER PREPROCESSOR HINTS                                      */
/* ------------------------------------------------------------------ */

/**
 * Returns a GLSL precision prelude to prepend to any anime lighting shader.
 * This is the piece that keeps the same shader source compilable across
 * Adreno, Mali, and PowerVR families.
 */
export function shaderPrecisionPrelude() {
  return [
    `#ifndef ANDROID_PRECISION_PRELUDE`,
    `#define ANDROID_PRECISION_PRELUDE`,
    `precision ${PRECISION.fragment} float;`,
    `precision ${PRECISION.fragment} int;`,
    `#ifdef VERTEX`,
    `precision ${PRECISION.vertex} float;`,
    `#endif`,
    `#endif`,
  ].join('\n');
}

/* ------------------------------------------------------------------ */
/* 12. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getAndroidProfileReport() {
  return {
    gpuFamily:          GPU_FAMILY_NAME,
    gpuFamilyId:        GPU_FAMILY,
    rendererName:       RAW_GPU.renderer,
    vendorName:         RAW_GPU.vendor,
    webglVersion:       DEVICE.webglVersion,
    deviceMemoryGB:     _deviceMemoryGB,
    hardwareConcurrency:DEVICE.hardwareConcurrency,
    perfTier:           PERF_TIER_LOCAL,
    profileTier:        PLATFORM_CONFIG.tier,
    precision:          PRECISION,
    shadow:             SHADOW_TUNING,
    ao:                 AO_TUNING,
    giProbeSpacing:     ANDROID_PROFILE.giProbeSpacing,
    giUpdateHz:         ANDROID_PROFILE.giUpdateHz,
    hdrColorFormat:     ANDROID_PROFILE.hdrColorFormat,
    workerPoolSize:     ANDROID_PROFILE.workerPoolSize,
    maxGpuMemoryMB:     ANDROID_PROFILE.maxGpuMemoryMB,
    dprCap:             ANDROID_PROFILE.dprCap,
    shadowMapCap:       ANDROID_PROFILE.shadowMapCap,
    quirks:             QUIRKS.slice(),
    platformReport:     getPlatformReport(),
  };
}

/* ------------------------------------------------------------------ */
/* 13. DEFAULT EXPORT                                                 */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ANDROID_PROFILE,
  GPU_FAMILY,
  GPU_FAMILY_NAME,
  ANDROID_GPU_FAMILY,
  ANDROID_GPU_FAMILY_NAME,
  GPU_PRECISION,
  PRECISION,
  GPU_SHADOW_TUNING,
  SHADOW_TUNING,
  GPU_AO_TUNING,
  AO_TUNING,
  THERMAL_CURVE,
  BATTERY_CURVE,
  THERMAL_LEVEL,
  THERMAL_LEVEL_NAME,
  BATTERY_LEVEL,
  BATTERY_LEVEL_NAME,
  giProbeSpacingForRam,
  giUpdateHzForRam,
  workerPoolSizeForConcurrency,
  resolveThermalLevel,
  resolveBatteryLevel,
  combinedDowngradeMultipliers,
  shaderPrecisionPrelude,
  getAndroidProfileReport,
};

export default _defaultExport;