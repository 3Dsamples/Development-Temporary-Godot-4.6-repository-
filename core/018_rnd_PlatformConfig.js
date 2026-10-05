// File : 018
// name : src/core/018_rnd_PlatformConfig.js
// description : Platform configuration and Android device profile resolver for
//               the anime lighting stack. Detects the concrete device class
//               (SoC family, GPU vendor, WebGL capabilities, precision limits),
//               resolves the correct profile from a fixed table, and produces
//               a frozen per-device config object that every lighting
//               subsystem reads to avoid the classic Android rendering bugs:
//
//                 • Mali-4xx/5xx precision loss in GI/AO math → downgrade
//                   GI probe lattice + AO sample count.
//                 • Adreno 3xx/4xx half-float clamp in shadow atlas → use
//                   Unorm16 or Rgba8 with explicit encoding.
//                 • PowerVR Rogue tile-based depth resolve cost → disable
//                   PCSS soft shadows, force BasicShadowMap or PCF.
//                 • Apple GPU (via WebKit) half-float renderability →
//                   enable RGBA16F HDR post chain.
//                 • WebGL1 fallback devices → disable cluster lighting,
//                   disable MSAA targets, force legacy forward path.
//                 • Low-memory Android devices (<3 GB) → cap render target
//                   pool, disable GI multi-bounce, cap shadow atlas.
//                 • High-DPI Android devices → cap DPR, force
//                   `image-rendering: pixelated` on the canvas.
//
//               Profile table (frozen, resolved at module load):
//                 PROFILE_ANDROID_HIGH     — Snapdragon 8xx / Dimensity 9000+
//                 PROFILE_ANDROID_MID      — Snapdragon 7xx / Dimensity 800
//                 PROFILE_ANDROID_LOW      — Snapdragon 6xx / Helio G series
//                 PROFILE_ANDROID_MINIMAL  — Snapdragon 4xx / entry-level
//                 PROFILE_IOS_HIGH         — A14+ / M-series
//                 PROFILE_IOS_MID          — A11-A13
//                 PROFILE_DESKTOP          — fallback for dev work
//
//               Every profile specifies:
//                 • maxTextureUnits, maxVertexAttributes
//                 • supportsHalfFloatColor, supportsFloatColor
//                 • supportsDepthTexture, supportsInstancing
//                 • supportsMultipleRenderTargets, supportsVertexTextures
//                 • precisionTier (highp / mediump)
//                 • preferredShadowType (basic / pcf / pcfsoft / pcss)
//                 • preferredColorFormat (rgba8 / rgba16f)
//                 • dprCap, msaaLimit
//                 • perDomainHzScale (shadows / gi / ao / post)
//                 • clusterGridOverride (or null)
//                 • giProbeLatticeCap
//                 • aoSampleCap
//                 • quirks[] — explicit named quirks so downstream code can
//                   test `profile.hasQuirk('mali_precision_loss')`.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external detection libs; all profiles frozen at
//               module load; every read O(1).
// best for : Guaranteeing that the anime lighting stack adapts to the actual
//            Android device it lands on — not just PERF_TIER. A 4 GB Snapdragon
//            6xx behaves very differently from a 4 GB Snapdragon 8xx; this
//            file makes that distinction explicit so shadow/GI/AO/cluster
//            code can pick the right algorithm without re-detecting.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

/* ------------------------------------------------------------------ */
/* 0. RAW DETECTION (one-shot at module load)                         */
/* ------------------------------------------------------------------ */

function _detectUA() {
  if (typeof navigator === 'undefined') {
    return { ua: '', vendor: '', platform: '' };
  }
  return {
    ua: navigator.userAgent || '',
    vendor: navigator.vendor || '',
    platform: navigator.platform || '',
  };
}

function _detectGPU() {
  if (typeof document === 'undefined') {
    return { vendor: 'unknown', renderer: 'unknown', webglVersion: 0 };
  }
  const out = { vendor: 'unknown', renderer: 'unknown', webglVersion: 0 };
  try {
    const canvas = document.createElement('canvas');
    const gl2 = canvas.getContext('webgl2');
    const gl  = gl2 || canvas.getContext('webgl');
    if (!gl) return out;

    out.webglVersion = gl2 ? 2 : 1;

    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (dbg) {
      out.vendor   = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL)   || 'unknown';
      out.renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || 'unknown';
    } else {
      out.vendor   = gl.getParameter(gl.VENDOR)   || 'unknown';
      out.renderer = gl.getParameter(gl.RENDERER) || 'unknown';
    }

    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch (_) {
    // Leave defaults.
  }
  return out;
}

function _detectCapabilities() {
  const caps = {
    maxTextureUnits:         8,
    maxVertexAttributes:     16,
    maxVertexUniformVectors: 128,
    maxFragmentUniformVectors: 128,
    maxTextureSize:          2048,
    maxRenderbufferSize:     2048,
    maxSamples:              0,
    supportsHalfFloatColor:  false,
    supportsFloatColor:      false,
    supportsDepthTexture:    false,
    supportsInstancing:      false,
    supportsVertexTextures:  false,
    supportsMRT:             false,
    supportsDerivatives:     false,
    highpSupported:          true,
  };

  if (typeof document === 'undefined') return caps;

  try {
    const canvas = document.createElement('canvas');
    const gl2 = canvas.getContext('webgl2');
    const gl  = gl2 || canvas.getContext('webgl');
    if (!gl) return caps;

    caps.maxTextureUnits         = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) || 8;
    caps.maxVertexAttributes     = gl.getParameter(gl.MAX_VERTEX_ATTRIBS) || 16;
    caps.maxTextureSize          = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 2048;
    caps.maxRenderbufferSize     = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) || 2048;
    caps.maxVertexUniformVectors = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) || 128;
    caps.maxFragmentUniformVectors = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS) || 128;

    // Precision probe.
    const hp = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    caps.highpSupported = !!hp && hp.precision > 0;

    if (gl2) {
      caps.supportsHalfFloatColor = !!gl.getExtension('EXT_color_buffer_half_float');
      caps.supportsFloatColor     = !!gl.getExtension('EXT_color_buffer_float');
      caps.supportsDepthTexture   = true;
      caps.supportsInstancing     = true;
      caps.supportsMRT            = true;
      caps.supportsVertexTextures = true;
      caps.supportsDerivatives    = true;
      caps.maxSamples             = gl.getParameter(gl.MAX_SAMPLES) || 0;
    } else {
      caps.supportsHalfFloatColor = !!(
        gl.getExtension('EXT_color_buffer_half_float') ||
        gl.getExtension('OES_texture_half_float')
      );
      caps.supportsFloatColor     = !!gl.getExtension('WEBGL_color_buffer_float');
      caps.supportsDepthTexture   = !!gl.getExtension('WEBGL_depth_texture');
      caps.supportsInstancing     = !!gl.getExtension('ANGLE_instanced_arrays');
      caps.supportsVertexTextures = !!gl.getExtension('OES_texture_float');
      caps.supportsDerivatives    = !!gl.getExtension('OES_standard_derivatives');
      caps.maxSamples             = 0;
    }

    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  } catch (_) {
    // Leave defaults.
  }

  return caps;
}

export const RAW_UA = _detectUA();
export const RAW_GPU = _detectGPU();
export const RAW_CAPS = _detectCapabilities();

/* ------------------------------------------------------------------ */
/* 1. DEVICE CLASSIFICATION                                           */
/* ------------------------------------------------------------------ */

function _classifySoc() {
  const r = (RAW_GPU.renderer || '').toLowerCase();
  const v = (RAW_GPU.vendor   || '').toLowerCase();
  const ua = RAW_UA.ua;

  // GPU-vendor inference.
  let gpuVendor = 'unknown';
  if (/adreno/.test(r))         gpuVendor = 'adreno';
  else if (/mali|immortalis/.test(r)) gpuVendor = 'mali';
  else if (/powervr|rogue/.test(r))   gpuVendor = 'powervr';
  else if (/apple/.test(r))           gpuVendor = 'apple';
  else if (/intel/.test(r))           gpuVendor = 'intel';
  else if (/nvidia|geforce/.test(r))  gpuVendor = 'nvidia';
  else if (/amd|radeon/.test(r))      gpuVendor = 'amd';
  else if (/swiftshader|software/.test(r)) gpuVendor = 'software';

  // Adreno generation inference (Adreno 4xx / 5xx / 6xx / 7xx / 8xx).
  let adrenoGeneration = 0;
  const aMatch = r.match(/adreno[\s\(]*(\d)/);
  if (aMatch) adrenoGeneration = parseInt(aMatch[1], 10);

  // Mali generation.
  let maliGeneration = 0;
  const mMatch = r.match(/mali-?([gt]?)(\d{2,3})/);
  if (mMatch) {
    const series = mMatch[1];
    const num = parseInt(mMatch[2], 10);
    if (series === 'g' || series === '') maliGeneration = num >= 70 ? 5 : num >= 50 ? 4 : num >= 30 ? 3 : 2;
    else if (series === 't') maliGeneration = 1;
  }

  // Apple generation.
  let appleGeneration = 0;
  if (gpuVendor === 'apple') {
    if (/m1|m2|m3|m4/.test(r))    appleGeneration = 4;
    else if (/a1[4-9]/.test(r))   appleGeneration = 3;
    else if (/a1[1-3]/.test(r))   appleGeneration = 2;
    else                          appleGeneration = 1;
  }

  // Device memory + cores for coarse bucketing.
  const deviceMemory = (typeof navigator !== 'undefined' && navigator.deviceMemory) ? navigator.deviceMemory : 4;
  const hardwareConcurrency = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) ? navigator.hardwareConcurrency : 4;

  const isAndroid = /Android/i.test(ua);
  const isIOS     = /iPhone|iPad|iPod/i.test(ua) || (/Macintosh/i.test(ua) && /Apple/.test(RAW_UA.vendor) && 'ontouchend' in (typeof document !== 'undefined' ? document : {}));
  const isMobile  = isAndroid || isIOS;
  const isDesktop = !isMobile;

  return {
    gpuVendor,
    adrenoGeneration,
    maliGeneration,
    appleGeneration,
    deviceMemory,
    hardwareConcurrency,
    isAndroid,
    isIOS,
    isMobile,
    isDesktop,
    webglVersion: RAW_GPU.webglVersion,
  };
}

export const DEVICE = _classifySoc();

/* ------------------------------------------------------------------ */
/* 2. QUIRK FLAGS                                                     */
/* ------------------------------------------------------------------ */

export const QUIRK = Object.freeze({
  MALI_PRECISION_LOSS:       'mali_precision_loss',
  ADRENO_HALF_FLOAT_CLAMP:   'adreno_half_float_clamp',
  ADRENO_SHADOW_BIAS:        'adreno_shadow_bias',
  POWERVR_TBDR_DEPTH_COST:   'powervr_tbdr_depth_cost',
  POWERVR_SHADER_COMPILE:    'powervr_shader_compile',
  APPLE_HALF_FLOAT_OK:       'apple_half_float_ok',
  WEBGL1_ONLY:               'webgl1_only',
  LOW_MEMORY_DEVICE:         'low_memory_device',
  HIGH_DPI_DEVICE:           'high_dpi_device',
  SLOW_UNIFORM_UPDATES:      'slow_uniform_updates',
  MSAA_UNRELIABLE:           'msaa_unreliable',
});

function _resolveQuirks() {
  const quirks = [];
  const v = DEVICE.gpuVendor;
  const r = (RAW_GPU.renderer || '').toLowerCase();

  if (v === 'mali' && DEVICE.maliGeneration >= 1 && DEVICE.maliGeneration <= 3) {
    quirks.push(QUIRK.MALI_PRECISION_LOSS);
  }

  if (v === 'adreno' && DEVICE.adrenoGeneration >= 3 && DEVICE.adrenoGeneration <= 5) {
    quirks.push(QUIRK.ADRENO_HALF_FLOAT_CLAMP);
    quirks.push(QUIRK.ADRENO_SHADOW_BIAS);
  }

  if (v === 'powervr' || /powervr|rogue/.test(r)) {
    quirks.push(QUIRK.POWERVR_TBDR_DEPTH_COST);
    quirks.push(QUIRK.POWERVR_SHADER_COMPILE);
  }

  if (v === 'apple' && DEVICE.appleGeneration >= 2) {
    quirks.push(QUIRK.APPLE_HALF_FLOAT_OK);
  }

  if (DEVICE.webglVersion === 1) {
    quirks.push(QUIRK.WEBGL1_ONLY);
  }

  if (DEVICE.deviceMemory < 4) {
    quirks.push(QUIRK.LOW_MEMORY_DEVICE);
  }

  if (typeof window !== 'undefined' && window.devicePixelRatio > 2.0) {
    quirks.push(QUIRK.HIGH_DPI_DEVICE);
  }

  if (v === 'mali' || v === 'powervr') {
    quirks.push(QUIRK.SLOW_UNIFORM_UPDATES);
  }

  if (DEVICE.webglVersion === 1 || v === 'powervr') {
    quirks.push(QUIRK.MSAA_UNRELIABLE);
  }

  return Object.freeze(quirks);
}

export const QUIRKS = _resolveQuirks();

function _hasQuirk(name) {
  return QUIRKS.indexOf(name) >= 0;
}

/* ------------------------------------------------------------------ */
/* 3. PRECISION TIER                                                  */
/* ------------------------------------------------------------------ */

export const PRECISION_TIER = Object.freeze({
  HIGH:    'highp',
  MEDIUM:  'mediump',
  LOW:     'lowp',
});

function _resolvePrecisionTier() {
  if (!RAW_CAPS.highpSupported) return PRECISION_TIER.MEDIUM;
  if (_hasQuirk(QUIRK.MALI_PRECISION_LOSS)) return PRECISION_TIER.MEDIUM;
  if (DEVICE.deviceMemory < 4 && DEVICE.webglVersion === 1) return PRECISION_TIER.MEDIUM;
  return PRECISION_TIER.HIGH;
}

export const RESOLVED_PRECISION = _resolvePrecisionTier();

/* ------------------------------------------------------------------ */
/* 4. PLATFORM PROFILE TABLE                                          */
/* ------------------------------------------------------------------ */

function _freezeProfile(p) {
  return Object.freeze(Object.assign({}, p, {
    quirks: Object.freeze(p.quirks ? p.quirks.slice() : []),
    hasQuirk(name) { return this.quirks.indexOf(name) >= 0; },
  }));
}

export const PROFILE_ANDROID_HIGH = _freezeProfile({
  name:                    'android_high',
  tier:                    'HIGH',
  dprCap:                  2.0,
  msaaLimit:               4,
  precision:               PRECISION_TIER.HIGH,
  preferredShadowType:     'pcfsoft',
  preferredColorFormat:    'rgba16f',
  supportsHalfFloatColor:  true,
  supportsFloatColor:      true,
  supportsDepthTexture:    true,
  supportsInstancing:      true,
  supportsMRT:             true,
  supportsVertexTextures:  true,
  supportsDerivatives:     true,
  maxClusterLights:        128,
  clusterGridOverride:     null,
  giProbeLatticeCap:       32,
  aoSampleCap:             16,
  shadowMapCap:            2048,
  shadowCascadeCap:        4,
  postPassCap:             6,
  perDomainHzScale: Object.freeze({
    shadows: 1.00, gi: 1.00, ao: 1.00, post: 1.00,
    environment: 1.00, interior: 1.00, exterior: 1.00,
  }),
  quirks: [],
});

export const PROFILE_ANDROID_MID = _freezeProfile({
  name:                    'android_mid',
  tier:                    'MEDIUM',
  dprCap:                  1.75,
  msaaLimit:               0,
  precision:               PRECISION_TIER.HIGH,
  preferredShadowType:     'pcf',
  preferredColorFormat:    'rgba16f',
  supportsHalfFloatColor:  true,
  supportsFloatColor:      false,
  supportsDepthTexture:    true,
  supportsInstancing:      true,
  supportsMRT:             true,
  supportsVertexTextures:  true,
  supportsDerivatives:     true,
  maxClusterLights:        64,
  clusterGridOverride:     null,
  giProbeLatticeCap:       16,
  aoSampleCap:             8,
  shadowMapCap:            1024,
  shadowCascadeCap:        2,
  postPassCap:             4,
  perDomainHzScale: Object.freeze({
    shadows: 0.85, gi: 0.85, ao: 0.85, post: 0.85,
    environment: 1.00, interior: 0.85, exterior: 0.85,
  }),
  quirks: [],
});

export const PROFILE_ANDROID_LOW = _freezeProfile({
  name:                    'android_low',
  tier:                    'LOW',
  dprCap:                  1.5,
  msaaLimit:               0,
  precision:               PRECISION_TIER.MEDIUM,
  preferredShadowType:     'pcf',
  preferredColorFormat:    'rgba8',
  supportsHalfFloatColor:  false,
  supportsFloatColor:      false,
  supportsDepthTexture:    false,
  supportsInstancing:      true,
  supportsMRT:             false,
  supportsVertexTextures:  false,
  supportsDerivatives:     true,
  maxClusterLights:        32,
  clusterGridOverride:     null,
  giProbeLatticeCap:       8,
  aoSampleCap:             4,
  shadowMapCap:            512,
  shadowCascadeCap:        1,
  postPassCap:             2,
  perDomainHzScale: Object.freeze({
    shadows: 0.65, gi: 0.65, ao: 0.65, post: 0.65,
    environment: 0.85, interior: 0.65, exterior: 0.65,
  }),
  quirks: [QUIRK.LOW_MEMORY_DEVICE, QUIRK.SLOW_UNIFORM_UPDATES],
});

export const PROFILE_ANDROID_MINIMAL = _freezeProfile({
  name:                    'android_minimal',
  tier:                    'LOW',
  dprCap:                  1.25,
  msaaLimit:               0,
  precision:               PRECISION_TIER.MEDIUM,
  preferredShadowType:     'basic',
  preferredColorFormat:    'rgba8',
  supportsHalfFloatColor:  false,
  supportsFloatColor:      false,
  supportsDepthTexture:    false,
  supportsInstancing:      false,
  supportsMRT:             false,
  supportsVertexTextures:  false,
  supportsDerivatives:     false,
  maxClusterLights:        16,
  clusterGridOverride:     Object.freeze([16, 9, 16]),
  giProbeLatticeCap:       6,
  aoSampleCap:             3,
  shadowMapCap:            256,
  shadowCascadeCap:        1,
  postPassCap:             1,
  perDomainHzScale: Object.freeze({
    shadows: 0.50, gi: 0.50, ao: 0.50, post: 0.50,
    environment: 0.75, interior: 0.50, exterior: 0.50,
  }),
  quirks: [QUIRK.LOW_MEMORY_DEVICE, QUIRK.WEBGL1_ONLY, QUIRK.MSAA_UNRELIABLE, QUIRK.SLOW_UNIFORM_UPDATES],
});

export const PROFILE_IOS_HIGH = _freezeProfile({
  name:                    'ios_high',
  tier:                    'HIGH',
  dprCap:                  2.0,
  msaaLimit:               4,
  precision:               PRECISION_TIER.HIGH,
  preferredShadowType:     'pcfsoft',
  preferredColorFormat:    'rgba16f',
  supportsHalfFloatColor:  true,
  supportsFloatColor:      true,
  supportsDepthTexture:    true,
  supportsInstancing:      true,
  supportsMRT:             true,
  supportsVertexTextures:  true,
  supportsDerivatives:     true,
  maxClusterLights:        128,
  clusterGridOverride:     null,
  giProbeLatticeCap:       32,
  aoSampleCap:             16,
  shadowMapCap:            2048,
  shadowCascadeCap:        4,
  postPassCap:             6,
  perDomainHzScale: Object.freeze({
    shadows: 1.00, gi: 1.00, ao: 1.00, post: 1.00,
    environment: 1.00, interior: 1.00, exterior: 1.00,
  }),
  quirks: [QUIRK.APPLE_HALF_FLOAT_OK],
});

export const PROFILE_IOS_MID = _freezeProfile({
  name:                    'ios_mid',
  tier:                    'MEDIUM',
  dprCap:                  1.75,
  msaaLimit:               0,
  precision:               PRECISION_TIER.HIGH,
  preferredShadowType:     'pcf',
  preferredColorFormat:    'rgba16f',
  supportsHalfFloatColor:  true,
  supportsFloatColor:      false,
  supportsDepthTexture:    true,
  supportsInstancing:      true,
  supportsMRT:             true,
  supportsVertexTextures:  true,
  supportsDerivatives:     true,
  maxClusterLights:        64,
  clusterGridOverride:     null,
  giProbeLatticeCap:       16,
  aoSampleCap:             8,
  shadowMapCap:            1024,
  shadowCascadeCap:        2,
  postPassCap:             4,
  perDomainHzScale: Object.freeze({
    shadows: 0.85, gi: 0.85, ao: 0.85, post: 0.85,
    environment: 1.00, interior: 0.85, exterior: 0.85,
  }),
  quirks: [QUIRK.APPLE_HALF_FLOAT_OK],
});

export const PROFILE_DESKTOP = _freezeProfile({
  name:                    'desktop',
  tier:                    'HIGH',
  dprCap:                  2.0,
  msaaLimit:               8,
  precision:               PRECISION_TIER.HIGH,
  preferredShadowType:     'pcss',
  preferredColorFormat:    'rgba16f',
  supportsHalfFloatColor:  true,
  supportsFloatColor:      true,
  supportsDepthTexture:    true,
  supportsInstancing:      true,
  supportsMRT:             true,
  supportsVertexTextures:  true,
  supportsDerivatives:     true,
  maxClusterLights:        256,
  clusterGridOverride:     null,
  giProbeLatticeCap:       64,
  aoSampleCap:             32,
  shadowMapCap:            4096,
  shadowCascadeCap:        4,
  postPassCap:             8,
  perDomainHzScale: Object.freeze({
    shadows: 1.00, gi: 1.00, ao: 1.00, post: 1.00,
    environment: 1.00, interior: 1.00, exterior: 1.00,
  }),
  quirks: [],
});

/* ------------------------------------------------------------------ */
/* 5. PROFILE RESOLUTION                                              */
/* ------------------------------------------------------------------ */

function _resolveProfile() {
  // WebGL1 fallback always gets the minimal profile.
  if (DEVICE.webglVersion === 1) {
    if (DEVICE.isAndroid) return PROFILE_ANDROID_MINIMAL;
    return PROFILE_DESKTOP;
  }

  if (DEVICE.isDesktop) return PROFILE_DESKTOP;

  if (DEVICE.isIOS) {
    return DEVICE.appleGeneration >= 3 ? PROFILE_IOS_HIGH : PROFILE_IOS_MID;
  }

  if (DEVICE.isAndroid) {
    const mem = DEVICE.deviceMemory;
    const cores = DEVICE.hardwareConcurrency;
    const gen = DEVICE.adrenoGeneration || DEVICE.maliGeneration;

    if (mem >= 8 && cores >= 8 && gen >= 6) return PROFILE_ANDROID_HIGH;
    if (mem >= 6 && cores >= 6 && gen >= 5) return PROFILE_ANDROID_MID;
    if (mem >= 4 && cores >= 4)             return PROFILE_ANDROID_MID;
    if (mem >= 3 && cores >= 4)             return PROFILE_ANDROID_LOW;
    return PROFILE_ANDROID_MINIMAL;
  }

  // Unknown mobile — safe default.
  return PROFILE_ANDROID_LOW;
}

export const PROFILE = _resolveProfile();

/* ------------------------------------------------------------------ */
/* 6. RESOLVED PLATFORM CONFIG                                        */
/* ------------------------------------------------------------------ */

/**
 * The frozen resolved platform config. Every lighting subsystem reads this
 * once at init and caches the fields it needs — no per-frame detection.
 *
 *   PLATFORM_CONFIG.dprCap
 *   PLATFORM_CONFIG.precision
 *   PLATFORM_CONFIG.preferredShadowType
 *   PLATFORM_CONFIG.supportsHalfFloatColor
 *   PLATFORM_CONFIG.hasQuirk(QUIRK.MALI_PRECISION_LOSS)
 */
export const PLATFORM_CONFIG = Object.freeze({
  profile:              PROFILE,
  name:                 PROFILE.name,
  tier:                 PROFILE.tier,
  perfTier:             getPerfTier(),

  // Rendering capabilities.
  dprCap:               PROFILE.dprCap,
  msaaLimit:            PROFILE.msaaLimit,
  precision:            PROFILE.precision,
  preferredShadowType:  PROFILE.preferredShadowType,
  preferredColorFormat: PROFILE.preferredColorFormat,

  supportsHalfFloatColor: PROFILE.supportsHalfFloatColor,
  supportsFloatColor:     PROFILE.supportsFloatColor,
  supportsDepthTexture:   PROFILE.supportsDepthTexture,
  supportsInstancing:     PROFILE.supportsInstancing,
  supportsMRT:            PROFILE.supportsMRT,
  supportsVertexTextures: PROFILE.supportsVertexTextures,
  supportsDerivatives:    PROFILE.supportsDerivatives,

  // Limits.
  maxClusterLights:     PROFILE.maxClusterLights,
  clusterGridOverride:  PROFILE.clusterGridOverride,
  giProbeLatticeCap:    PROFILE.giProbeLatticeCap,
  aoSampleCap:          PROFILE.aoSampleCap,
  shadowMapCap:         PROFILE.shadowMapCap,
  shadowCascadeCap:     PROFILE.shadowCascadeCap,
  postPassCap:          PROFILE.postPassCap,

  perDomainHzScale:     PROFILE.perDomainHzScale,

  // Raw device info (read-only).
  device: Object.freeze({
    gpuVendor:           DEVICE.gpuVendor,
    adrenoGeneration:    DEVICE.adrenoGeneration,
    maliGeneration:      DEVICE.maliGeneration,
    appleGeneration:     DEVICE.appleGeneration,
    deviceMemory:        DEVICE.deviceMemory,
    hardwareConcurrency: DEVICE.hardwareConcurrency,
    isAndroid:           DEVICE.isAndroid,
    isIOS:               DEVICE.isIOS,
    isMobile:            DEVICE.isMobile,
    isDesktop:           DEVICE.isDesktop,
    webglVersion:        DEVICE.webglVersion,
    rendererName:        RAW_GPU.renderer,
    vendorName:          RAW_GPU.vendor,
  }),

  caps: Object.freeze(Object.assign({}, RAW_CAPS)),
  quirks: QUIRKS,

  hasQuirk(name) {
    return QUIRKS.indexOf(name) >= 0;
  },
});

/* ------------------------------------------------------------------ */
/* 7. ANDROID GUARD HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Returns a recommended pixel ratio after clamping by device DPR and the
 * platform profile.
 */
export function recommendedPixelRatio() {
  const dpr = (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1;
  return Math.min(dpr, PLATFORM_CONFIG.dprCap);
}

/**
 * Returns true when the current platform can use the given shadow filter.
 */
export function canUseShadowType(type) {
  const pref = PLATFORM_CONFIG.preferredShadowType;
  if (type === 'basic')  return true;
  if (type === 'pcf')    return pref === 'pcf' || pref === 'pcfsoft' || pref === 'pcss';
  if (type === 'pcfsoft')return pref === 'pcfsoft' || pref === 'pcss';
  if (type === 'pcss')   return pref === 'pcss';
  if (type === 'vsm')    return !PLATFORM_CONFIG.hasQuirk(QUIRK.MSAA_UNRELIABLE);
  if (type === 'esm')    return false; // never on mobile
  return false;
}

/**
 * Returns true when the current platform can use the given render target
 * color format.
 */
export function canUseColorFormat(format) {
  if (format === 'rgba8')   return true;
  if (format === 'rgb8')    return true;
  if (format === 'rgba16f') return PLATFORM_CONFIG.supportsHalfFloatColor;
  if (format === 'rgba32f') return PLATFORM_CONFIG.supportsFloatColor;
  if (format === 'depth')   return PLATFORM_CONFIG.supportsDepthTexture;
  return false;
}

/**
 * Returns a suggested precision string for GLSL (highp / mediump).
 */
export function suggestedPrecision() {
  return PLATFORM_CONFIG.precision;
}

/**
 * Apply a per-domain Hz scale from the platform profile to a base Hz.
 */
export function scaleDomainHz(domain, baseHz) {
  const scale = PLATFORM_CONFIG.perDomainHzScale[domain];
  if (!Number.isFinite(scale)) return baseHz;
  return Math.max(2, Math.round(baseHz * scale));
}

/**
 * Returns the cluster grid resolution to use (override from profile if any,
 * otherwise the tier default).
 */
export function resolvedClusterGrid(defaultGrid) {
  if (PLATFORM_CONFIG.clusterGridOverride) {
    return PLATFORM_CONFIG.clusterGridOverride;
  }
  return defaultGrid;
}

/* ------------------------------------------------------------------ */
/* 8. DIAGNOSTICS                                                     */
/* ------------------------------------------------------------------ */

export function getPlatformReport() {
  return {
    profile:              PLATFORM_CONFIG.name,
    tier:                 PLATFORM_CONFIG.tier,
    perfTier:             PLATFORM_CONFIG.perfTier,
    gpuVendor:            DEVICE.gpuVendor,
    rendererName:         RAW_GPU.renderer,
    vendorName:           RAW_GPU.vendor,
    webglVersion:         DEVICE.webglVersion,
    deviceMemory:         DEVICE.deviceMemory,
    hardwareConcurrency:  DEVICE.hardwareConcurrency,
    dpr:                  (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
    dprCap:               PLATFORM_CONFIG.dprCap,
    precision:            PLATFORM_CONFIG.precision,
    preferredShadowType:  PLATFORM_CONFIG.preferredShadowType,
    preferredColorFormat: PLATFORM_CONFIG.preferredColorFormat,
    supportsHalfFloatColor: PLATFORM_CONFIG.supportsHalfFloatColor,
    supportsFloatColor:     PLATFORM_CONFIG.supportsFloatColor,
    supportsDepthTexture:   PLATFORM_CONFIG.supportsDepthTexture,
    supportsInstancing:     PLATFORM_CONFIG.supportsInstancing,
    supportsMRT:            PLATFORM_CONFIG.supportsMRT,
    maxClusterLights:       PLATFORM_CONFIG.maxClusterLights,
    giProbeLatticeCap:      PLATFORM_CONFIG.giProbeLatticeCap,
    aoSampleCap:            PLATFORM_CONFIG.aoSampleCap,
    shadowMapCap:           PLATFORM_CONFIG.shadowMapCap,
    shadowCascadeCap:       PLATFORM_CONFIG.shadowCascadeCap,
    postPassCap:            PLATFORM_CONFIG.postPassCap,
    quirks:                 QUIRKS.slice(),
  };
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  PLATFORM_CONFIG,
  PROFILE,
  DEVICE,
  QUIRKS,
  RAW_UA,
  RAW_GPU,
  RAW_CAPS,
  RESOLVED_PRECISION,
  PRECISION_TIER,
  QUIRK,
  PROFILE_ANDROID_HIGH,
  PROFILE_ANDROID_MID,
  PROFILE_ANDROID_LOW,
  PROFILE_ANDROID_MINIMAL,
  PROFILE_IOS_HIGH,
  PROFILE_IOS_MID,
  PROFILE_DESKTOP,
  recommendedPixelRatio,
  canUseShadowType,
  canUseColorFormat,
  suggestedPrecision,
  scaleDomainHz,
  resolvedClusterGrid,
  getPlatformReport,
};

export default _defaultExport;