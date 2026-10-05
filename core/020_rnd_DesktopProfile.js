// File : 020
// name : src/core/020_rnd_DesktopProfile.js
// description : Desktop fallback profile resolver for the anime lighting
//               stack. The engine targets Android mobile as the primary
//               platform, but every lighting subsystem needs a valid,
//               deterministic profile when running on desktop during
//               development, CI, regression capture, or screenshot
//               comparison. This module provides that profile with the
//               same interface as 019_rnd_AndroidProfile.js so downstream
//               code reads `DesktopProfile.shadow`, `.ao`, `.giProbeSpacing`,
//               `.precision`, etc., without branching on platform.
//
//               Responsibilities:
//                 • Classify the desktop GPU family (NVIDIA / AMD / Intel /
//                   Apple Silicon / Unknown) from the unmasked renderer
//                   string, with explicit handling of swiftshader and
//                   ANGLE-wrapped backends (D3D11 / D3D9 / Metal / OpenGL).
//                 • Provide desktop-grade precision hints (always highp).
//                 • Provide desktop-grade shadow tuning (smaller bias,
//                   smaller normal bias, PCF/PCSS softness).
//                 • Provide desktop-grade AO tuning (16-32 samples,
//                   temporal accumulation, no half-res constraint).
//                 • Provide desktop-grade GI spacing (1.5 m) and update
//                   Hz (30 Hz).
//                 • Provide desktop-grade worker pool sizing
//                   (hardwareConcurrency - 1, capped at 8).
//                 • Provide desktop-grade memory caps (512 MB GPU, 1 GB
//                   CPU heap hint).
//                 • Provide desktop-grade thermal / battery curves (all
//                   passive: level 0 always, since desktops rarely throttle
//                   the way phones do — but the curves are still exposed so
//                   the same quality controller code runs unchanged).
//                 • Detect ANGLE-wrapped backends and expose them as
//                   sub-flags so shader code can opt out of GPU-vendor
//                   specific branches when the real backend is hidden.
//
//               Design constraints:
//                 • Same shape as ANDROID_PROFILE so downstream systems can
//                   swap profiles without touching business logic.
//                 • All tables frozen at module load.
//                 • Every read O(1) and allocation-free.
//                 • Never used on a real Android device — Android picks
//                   its own profile via 019_rnd_AndroidProfile.js; this is
//                   the fallback path for `!isAndroid && !isIOS`.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external detection libs.
// best for : Dev / CI / regression capture on desktop. Every lighting
//            subsystem that already reads ANDROID_PROFILE can also read
//            DESKTOP_PROFILE with zero code changes, so the whole anime
//            pipeline runs on a laptop with the same deterministic output
//            it produces on a phone — just at a higher quality tier.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  QUIRK,
  PRECISION_TIER,
  RAW_GPU,
  RAW_CAPS,
  canUseShadowType,
  getPlatformReport,
} from './018_rnd_PlatformConfig.js';

import {
  getDefaultConfig,
  PERF_TIER,
  PLATFORM,
} from './016_rnd_Config.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

export const DESKTOP_GPU_FAMILY = Object.freeze({
  UNKNOWN:          0,
  NVIDIA_GEFORCE:   1,
  NVIDIA_QUADRO:    2,
  AMD_RADEON:       3,
  AMD_FIREPRO:      4,
  INTEL_HD:         5,
  INTEL_IRIS:       6,
  INTEL_ARC:        7,
  APPLE_M1:         8,
  APPLE_M2:         9,
  APPLE_M3:        10,
  APPLE_M4:        11,
  APPLE_INTEL_IGPU:12,
  SWIFTSHADER:     13,
  LLVMPIPE:        14,
  ANGLE_D3D11:     15,
  ANGLE_D3D9:      16,
  ANGLE_METAL:     17,
  ANGLE_GL:        18,
  ANGLE_VULKAN:    19,
  COUNT:           20,
});

export const DESKTOP_GPU_FAMILY_NAME = Object.freeze([
  'unknown',
  'nvidia_geforce',
  'nvidia_quadro',
  'amd_radeon',
  'amd_firepro',
  'intel_hd',
  'intel_iris',
  'intel_arc',
  'apple_m1',
  'apple_m2',
  'apple_m3',
  'apple_m4',
  'apple_intel_igpu',
  'swiftshader',
  'llvmpipe',
  'angle_d3d11',
  'angle_d3d9',
  'angle_metal',
  'angle_gl',
  'angle_vulkan',
]);

export const DESKTOP_PRECISION = Object.freeze({
  vertex:          'highp',
  fragment:        'highp',
  positionVarying: 'highp',
  varyingFloat:    'highp',
});

export const DESKTOP_SHADOW_TUNING = Object.freeze({
  bias:         -0.0004,
  normalBias:    0.010,
  pancakeFix:   false,
  softness:      0.10,
  filter:       'pcss',
});

export const DESKTOP_AO_TUNING = Object.freeze({
  samples:      32,
  radius:        2.5,
  temporal:     true,
  halfRes:      false,
});

export const DESKTOP_THERMAL_CURVE = Object.freeze([
  Object.freeze({ shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta:  0, dprScale: 1.00, clusterScale: 1.00 }),
  Object.freeze({ shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta:  0, dprScale: 1.00, clusterScale: 1.00 }),
  Object.freeze({ shadowScale: 0.90, giScale: 0.90, aoScale: 1.00, postDelta:  0, dprScale: 1.00, clusterScale: 0.95 }),
  Object.freeze({ shadowScale: 0.75, giScale: 0.75, aoScale: 0.90, postDelta: -1, dprScale: 0.95, clusterScale: 0.85 }),
  Object.freeze({ shadowScale: 0.50, giScale: 0.50, aoScale: 0.75, postDelta: -2, dprScale: 0.90, clusterScale: 0.70 }),
]);

export const DESKTOP_BATTERY_CURVE = Object.freeze([
  Object.freeze({ dprScale: 1.00, shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta: 0 }),
  Object.freeze({ dprScale: 0.95, shadowScale: 1.00, giScale: 1.00, aoScale: 1.00, postDelta: 0 }),
  Object.freeze({ dprScale: 0.90, shadowScale: 0.95, giScale: 0.95, aoScale: 1.00, postDelta: 0 }),
  Object.freeze({ dprScale: 0.85, shadowScale: 0.85, giScale: 0.85, aoScale: 0.90, postDelta: -1 }),
  Object.freeze({ dprScale: 0.75, shadowScale: 0.70, giScale: 0.70, aoScale: 0.80, postDelta: -2 }),
]);

/* ------------------------------------------------------------------ */
/* 1. GPU FAMILY RESOLUTION                                           */
/* ------------------------------------------------------------------ */

function _resolveDesktopGpuFamily() {
  const r = (RAW_GPU.renderer || '').toLowerCase();
  const v = (RAW_GPU.vendor   || '').toLowerCase();

  // Software paths first.
  if (/swiftshader/.test(r))                    return DESKTOP_GPU_FAMILY.SWIFTSHADER;
  if (/llvmpipe|softpipe|software rasterizer/.test(r)) return DESKTOP_GPU_FAMILY.LLVMPIPE;

  // ANGLE wrappers (Chrome / Edge on Windows, Chrome on macOS).
  if (/angle/.test(v) || /angle/.test(r)) {
    if (/d3d11|direct3d11/.test(r)) return DESKTOP_GPU_FAMILY.ANGLE_D3D11;
    if (/d3d9|direct3d9/.test(r))   return DESKTOP_GPU_FAMILY.ANGLE_D3D9;
    if (/metal/.test(r))            return DESKTOP_GPU_FAMILY.ANGLE_METAL;
    if (/vulkan/.test(r))           return DESKTOP_GPU_FAMILY.ANGLE_VULKAN;
    return DESKTOP_GPU_FAMILY.ANGLE_GL;
  }

  // Apple Silicon (M1–M4).
  if (/apple/.test(v) || /apple/.test(r)) {
    if (/m4/.test(r)) return DESKTOP_GPU_FAMILY.APPLE_M4;
    if (/m3/.test(r)) return DESKTOP_GPU_FAMILY.APPLE_M3;
    if (/m2/.test(r)) return DESKTOP_GPU_FAMILY.APPLE_M2;
    if (/m1/.test(r)) return DESKTOP_GPU_FAMILY.APPLE_M1;
    return DESKTOP_GPU_FAMILY.APPLE_INTEL_IGPU;
  }

  // NVIDIA.
  if (/nvidia|geforce/.test(v) || /nvidia|geforce/.test(r)) {
    if (/quadro|rtx a\d|tesla/.test(r)) return DESKTOP_GPU_FAMILY.NVIDIA_QUADRO;
    return DESKTOP_GPU_FAMILY.NVIDIA_GEFORCE;
  }

  // AMD.
  if (/amd|radeon/.test(v) || /amd|radeon/.test(r)) {
    if (/firepro|radeon pro/.test(r)) return DESKTOP_GPU_FAMILY.AMD_FIREPRO;
    return DESKTOP_GPU_FAMILY.AMD_RADEON;
  }

  // Intel.
  if (/intel/.test(v) || /intel/.test(r)) {
    if (/arc/.test(r))          return DESKTOP_GPU_FAMILY.INTEL_ARC;
    if (/iris/.test(r))         return DESKTOP_GPU_FAMILY.INTEL_IRIS;
    if (/hd graphics|uhd/.test(r)) return DESKTOP_GPU_FAMILY.INTEL_HD;
    return DESKTOP_GPU_FAMILY.INTEL_HD;
  }

  return DESKTOP_GPU_FAMILY.UNKNOWN;
}

export const DESKTOP_GPU = _resolveDesktopGpuFamily();
export const DESKTOP_GPU_NAME = DESKTOP_GPU_FAMILY_NAME[DESKTOP_GPU] || 'unknown';

/* ------------------------------------------------------------------ */
/* 2. PER-GPU SHADOW TUNING OVERRIDES                                 */
/* ------------------------------------------------------------------ */

function _resolveDesktopShadowTuning() {
  switch (DESKTOP_GPU) {
    case DESKTOP_GPU_FAMILY.NVIDIA_GEFORCE:
    case DESKTOP_GPU_FAMILY.NVIDIA_QUADRO:
      return Object.freeze({ bias: -0.0003, normalBias: 0.008, pancakeFix: false, softness: 0.10, filter: 'pcss' });

    case DESKTOP_GPU_FAMILY.AMD_RADEON:
    case DESKTOP_GPU_FAMILY.AMD_FIREPRO:
      return Object.freeze({ bias: -0.0005, normalBias: 0.012, pancakeFix: false, softness: 0.10, filter: 'pcss' });

    case DESKTOP_GPU_FAMILY.INTEL_ARC:
      return Object.freeze({ bias: -0.0006, normalBias: 0.014, pancakeFix: false, softness: 0.09, filter: 'pcf' });

    case DESKTOP_GPU_FAMILY.INTEL_IRIS:
    case DESKTOP_GPU_FAMILY.INTEL_HD:
      return Object.freeze({ bias: -0.0010, normalBias: 0.024, pancakeFix: false, softness: 0.07, filter: 'pcf' });

    case DESKTOP_GPU_FAMILY.APPLE_M1:
    case DESKTOP_GPU_FAMILY.APPLE_M2:
    case DESKTOP_GPU_FAMILY.APPLE_M3:
    case DESKTOP_GPU_FAMILY.APPLE_M4:
      return Object.freeze({ bias: -0.0004, normalBias: 0.010, pancakeFix: false, softness: 0.10, filter: 'pcfsoft' });

    case DESKTOP_GPU_FAMILY.APPLE_INTEL_IGPU:
      return Object.freeze({ bias: -0.0012, normalBias: 0.026, pancakeFix: false, softness: 0.06, filter: 'pcf' });

    case DESKTOP_GPU_FAMILY.SWIFTSHADER:
    case DESKTOP_GPU_FAMILY.LLVMPIPE:
      return Object.freeze({ bias: -0.0035, normalBias: 0.070, pancakeFix: true,  softness: 0.00, filter: 'basic' });

    default:
      return DESKTOP_SHADOW_TUNING;
  }
}

export const SHADOW_TUNING = _resolveDesktopShadowTuning();

/* ------------------------------------------------------------------ */
/* 3. PER-GPU AO TUNING OVERRIDES                                     */
/* ------------------------------------------------------------------ */

function _resolveDesktopAOTuning() {
  switch (DESKTOP_GPU) {
    case DESKTOP_GPU_FAMILY.NVIDIA_GEFORCE:
    case DESKTOP_GPU_FAMILY.NVIDIA_QUADRO:
      return Object.freeze({ samples: 32, radius: 2.5, temporal: true, halfRes: false });

    case DESKTOP_GPU_FAMILY.AMD_RADEON:
    case DESKTOP_GPU_FAMILY.AMD_FIREPRO:
      return Object.freeze({ samples: 32, radius: 2.5, temporal: true, halfRes: false });

    case DESKTOP_GPU_FAMILY.INTEL_ARC:
      return Object.freeze({ samples: 32, radius: 2.5, temporal: true, halfRes: false });

    case DESKTOP_GPU_FAMILY.INTEL_IRIS:
      return Object.freeze({ samples: 16, radius: 2.2, temporal: true,  halfRes: false });

    case DESKTOP_GPU_FAMILY.INTEL_HD:
      return Object.freeze({ samples: 8,  radius: 2.0, temporal: true,  halfRes: true  });

    case DESKTOP_GPU_FAMILY.APPLE_M1:
    case DESKTOP_GPU_FAMILY.APPLE_M2:
    case DESKTOP_GPU_FAMILY.APPLE_M3:
    case DESKTOP_GPU_FAMILY.APPLE_M4:
      return Object.freeze({ samples: 24, radius: 2.4, temporal: true, halfRes: false });

    case DESKTOP_GPU_FAMILY.APPLE_INTEL_IGPU:
      return Object.freeze({ samples: 8,  radius: 2.0, temporal: true, halfRes: true });

    case DESKTOP_GPU_FAMILY.SWIFTSHADER:
    case DESKTOP_GPU_FAMILY.LLVMPIPE:
      return Object.freeze({ samples: 4,  radius: 1.5, temporal: false, halfRes: true });

    default:
      return DESKTOP_AO_TUNING;
  }
}

export const AO_TUNING = _resolveDesktopAOTuning();

/* ------------------------------------------------------------------ */
/* 4. GI TUNING                                                       */
/* ------------------------------------------------------------------ */

/**
 * Desktop GI probe spacing — always dense, independent of GPU family,
 * because desktop CPUs have plenty of budget for the probe solve.
 */
export const GI_PROBE_SPACING = 1.5;
export const GI_UPDATE_HZ     = 30;

/* ------------------------------------------------------------------ */
/* 5. WORKER POOL                                                     */
/* ------------------------------------------------------------------ */

export function desktopWorkerPoolSize(hardwareConcurrency) {
  const cores = Math.max(1, hardwareConcurrency | 0);
  return Math.max(1, Math.min(8, cores - 1));
}

/* ------------------------------------------------------------------ */
/* 6. RESOLVED DESKTOP PROFILE                                        */
/* ------------------------------------------------------------------ */

const _baseConfig = getDefaultConfig();
const _baseResolved = _baseConfig.section;

const _desktopMemoryHint =
  (typeof navigator !== 'undefined' && navigator.deviceMemory) ? navigator.deviceMemory : 8;

export const DESKTOP_PROFILE = Object.freeze({
  name:               'desktop_profile_' + DESKTOP_GPU_NAME,
  gpuFamily:          DESKTOP_GPU,
  gpuFamilyName:      DESKTOP_GPU_NAME,
  rendererName:       RAW_GPU.renderer,
  vendorName:         RAW_GPU.vendor,
  perfTier:           PERF_TIER,
  profileTier:        PLATFORM_CONFIG.tier,
  webglVersion:       DEVICE.webglVersion,

  // Precision.
  precision:          DESKTOP_PRECISION,
  precisionTier:      PRECISION_TIER.HIGH,

  // Shadow tuning.
  shadow:             SHADOW_TUNING,
  shadowFilter:       canUseShadowType(SHADOW_TUNING.filter) ? SHADOW_TUNING.filter : 'pcf',

  // AO tuning.
  ao:                 AO_TUNING,

  // GI tuning.
  giProbeSpacing:     GI_PROBE_SPACING,
  giUpdateHz:         GI_UPDATE_HZ,

  // Color format.
  hdrColorFormat:     PLATFORM_CONFIG.supportsHalfFloatColor ? 'rgba16f' : 'rgba8',

  // Worker pool.
  workerPoolSize:     desktopWorkerPoolSize(DEVICE.hardwareConcurrency),

  // Curves.
  thermalCurve:       DESKTOP_THERMAL_CURVE,
  batteryCurve:       DESKTOP_BATTERY_CURVE,

  // Per-domain Hz scale (desktop: 1.0 everywhere).
  perDomainHzScale: Object.freeze({
    shadows: 1.00, gi: 1.00, ao: 1.00, post: 1.00,
    environment: 1.00, interior: 1.00, exterior: 1.00,
  }),

  // Platform caps (desktop: generous).
  dprCap:             2.0,
  maxClusterLights:   256,
  giProbeLatticeCap:  64,
  aoSampleCap:        32,
  shadowMapCap:       4096,
  shadowCascadeCap:   4,
  postPassCap:        8,

  // Device memory cap for RT pool.
  maxGpuMemoryMB:     _desktopMemoryHint >= 16 ? 1024 : _desktopMemoryHint >= 8 ? 512 : 256,

  // Quirks.
  quirks:             QUIRKS,
  hasQuirk(name) { return QUIRKS.indexOf(name) >= 0; },
});

/* ------------------------------------------------------------------ */
/* 7. ANGLE BACKEND DETECTION                                         */
/* ------------------------------------------------------------------ */

function _detectAngleBackend() {
  switch (DESKTOP_GPU) {
    case DESKTOP_GPU_FAMILY.ANGLE_D3D11:  return 'd3d11';
    case DESKTOP_GPU_FAMILY.ANGLE_D3D9:   return 'd3d9';
    case DESKTOP_GPU_FAMILY.ANGLE_METAL:  return 'metal';
    case DESKTOP_GPU_FAMILY.ANGLE_GL:     return 'gl';
    case DESKTOP_GPU_FAMILY.ANGLE_VULKAN: return 'vulkan';
    default:                              return null;
  }
}

export const ANGLE_BACKEND = _detectAngleBackend();
export const IS_ANGLE = ANGLE_BACKEND !== null;

/* ------------------------------------------------------------------ */
/* 8. UNIFIED PROFILE SELECTOR                                        */
/* ------------------------------------------------------------------ */

/**
 * Returns the correct profile for the current platform:
 *   - Android → ANDROID_PROFILE (from 019_rnd_AndroidProfile.js)
 *   - iOS     → handled by 018 profile directly
 *   - Desktop → DESKTOP_PROFILE
 *
 * Because this module cannot import 019 without a circular dependency,
 * the caller is expected to import the Android profile explicitly and
 * use this helper only for the desktop branch. The helper returns
 * `DESKTOP_PROFILE` for any platform the caller flags as desktop.
 */
export function resolveDesktopProfileIfApplicable() {
  if (DEVICE.isAndroid) return null;
  if (DEVICE.isIOS)     return null;
  return DESKTOP_PROFILE;
}

/* ------------------------------------------------------------------ */
/* 9. SHADER PRECISION PRELUDE (desktop)                              */
/* ------------------------------------------------------------------ */

export function desktopShaderPrecisionPrelude() {
  return [
    `#ifndef DESKTOP_PRECISION_PRELUDE`,
    `#define DESKTOP_PRECISION_PRELUDE`,
    `precision highp float;`,
    `precision highp int;`,
    `#endif`,
  ].join('\n');
}

/* ------------------------------------------------------------------ */
/* 10. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getDesktopProfileReport() {
  return {
    gpuFamily:           DESKTOP_GPU_NAME,
    gpuFamilyId:         DESKTOP_GPU,
    rendererName:        RAW_GPU.renderer,
    vendorName:          RAW_GPU.vendor,
    webglVersion:        DEVICE.webglVersion,
    angleBackend:        ANGLE_BACKEND,
    isAngle:             IS_ANGLE,
    hardwareConcurrency: DEVICE.hardwareConcurrency,
    perfTier:            PERF_TIER,
    profileTier:         PLATFORM_CONFIG.tier,
    precision:           DESKTOP_PRECISION,
    shadow:              SHADOW_TUNING,
    ao:                  AO_TUNING,
    giProbeSpacing:      GI_PROBE_SPACING,
    giUpdateHz:          GI_UPDATE_HZ,
    hdrColorFormat:      DESKTOP_PROFILE.hdrColorFormat,
    workerPoolSize:      DESKTOP_PROFILE.workerPoolSize,
    maxGpuMemoryMB:      DESKTOP_PROFILE.maxGpuMemoryMB,
    dprCap:              DESKTOP_PROFILE.dprCap,
    shadowMapCap:        DESKTOP_PROFILE.shadowMapCap,
    quirks:              QUIRKS.slice(),
    platformReport:      getPlatformReport(),
  };
}

/* ------------------------------------------------------------------ */
/* 11. DEFAULT EXPORT                                                 */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  DESKTOP_PROFILE,
  DESKTOP_GPU,
  DESKTOP_GPU_NAME,
  DESKTOP_GPU_FAMILY,
  DESKTOP_GPU_FAMILY_NAME,
  DESKTOP_PRECISION,
  DESKTOP_SHADOW_TUNING,
  DESKTOP_AO_TUNING,
  DESKTOP_THERMAL_CURVE,
  DESKTOP_BATTERY_CURVE,
  SHADOW_TUNING,
  AO_TUNING,
  GI_PROBE_SPACING,
  GI_UPDATE_HZ,
  ANGLE_BACKEND,
  IS_ANGLE,
  desktopWorkerPoolSize,
  resolveDesktopProfileIfApplicable,
  desktopShaderPrecisionPrelude,
  getDesktopProfileReport,
};

export default _defaultExport;