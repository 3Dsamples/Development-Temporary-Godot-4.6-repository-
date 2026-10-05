// File : 021
// name : src/core/021_rnd_Capabilities.js
// description : Runtime capability probe and feature-flag resolver for the
//               anime lighting stack. Where 018_rnd_PlatformConfig.js owns
//               device CLASSIFICATION (GPU family, tier, precision) and
//               019/020 own per-platform TUNING tables, THIS module owns the
//               canonical RUNTIME CAPABILITY surface: the concrete set of
//               GPU features, extensions, precision limits, texture/render
//               target constraints, and driver quirks that a lighting
//               subsystem must interrogate before enabling a feature.
//
//               It runs ONE capability probe at module init and then exposes
//               a frozen capabilities object + a small set of query helpers
//               so downstream code never touches WebGL context directly.
//
//               Probed capabilities:
//                 • webgl2, webgl1 fallback status
//                 • max texture size, max renderbuffer size, max cube size
//                 • max texture image units (per stage)
//                 • max vertex attributes (usually 16)
//                 • max uniform vectors (vertex + fragment)
//                 • max varying vectors
//                 • max draw buffers (MRT count)
//                 • max samples (WebGL2 MSAA)
//                 • max anisotropy
//                 • shading language version (GLSL 100 vs 300 es)
//                 • precision (highp / mediump / lowp per stage)
//                 • compressed texture formats (ASTC / ETC2 / DXT)
//                 • float renderability (color + depth)
//                 • half-float renderability
//                 • depth texture support
//                 • instanced arrays support
//                 • draw buffers support
//                 • vertex array objects (WebGL2 native vs OES extension)
//                 • standard derivatives
//                 • texture LOD
//                 • shader texture LOD
//                 • element index uint
//                 • frag depth
//                 • shader texture float / half-float linear filtering
//                 • color buffer float / half-float
//                 • multiview (WebXR)
//                 • WEBGL_lose_context (for probes)
//                 • KHR_parallel_shader_compile (async compile)
//                 • EXT_disjoint_timer_query (GPU timers)
//                 • WEBGL_debug_renderer_info (unmasked GPU string)
//
//               All results frozen; every query helper O(1) and allocation-
//               free. Provides a `computeFeatureFlags()` that maps raw
//               capabilities to the boolean decisions the lighting stack
//               actually branches on (canUsePCSS, canUseGIProbeGrid,
//               canUseHDRTargets, canUseMRT, canUseInstancing, etc.).
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external detection libs; context is created, probed,
//               and released at module load — never kept alive.
// best for : Giving every lighting subsystem a single authoritative answer
//            to "does this GPU/driver support X?" so the anime pipeline
//            adapts without a context round-trip inside update loops.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  QUIRK,
  RAW_GPU,
  RAW_CAPS,
} from './018_rnd_PlatformConfig.js';

import {
  ANDROID_PROFILE,
  GPU_FAMILY,
  GPU_FAMILY_NAME,
} from './019_rnd_AndroidProfile.js';

import {
  getPerfTier,
} from './008_scn_world.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const SHADING_LANGUAGE = Object.freeze({
  GLSL_100: 0,   // WebGL1
  GLSL_300: 1,   // WebGL2
});

export const PRECISION_SUPPORT = Object.freeze({
  NONE:    0,
  LOW:     1,
  MEDIUM:  2,
  HIGH:    3,
});

export const COMPRESSED_FORMAT = Object.freeze({
  NONE: 0,
  DXT:  1 << 0,
  ETC1: 1 << 1,
  ETC2: 1 << 2,
  ASTC: 1 << 3,
  PVRTC: 1 << 4,
});

/* ------------------------------------------------------------------ */
/* 1. RAW PROBE (one-shot at module init)                             */
/* ------------------------------------------------------------------ */

const _raw = {
  available: false,
  webgl2: false,
  glslVersion: SHADING_LANGUAGE.GLSL_100,

  maxTextureSize: 2048,
  maxCubeMapSize: 2048,
  maxRenderbufferSize: 2048,
  maxTextureImageUnits: 8,
  maxVertexTextureImageUnits: 0,
  maxCombinedTextureImageUnits: 8,
  maxVertexAttribs: 16,
  maxVertexUniformVectors: 128,
  maxFragmentUniformVectors: 128,
  maxVaryingVectors: 8,
  maxDrawBuffers: 1,
  maxSamples: 0,
  maxAnisotropy: 1,

  precision: {
    vertex:   PRECISION_SUPPORT.HIGH,
    fragment: PRECISION_SUPPORT.HIGH,
  },

  extensions: {
    oesElementIndexUint:       false,
    oesStandardDerivatives:    false,
    oesTextureFloat:           false,
    oesTextureFloatLinear:     false,
    oesTextureHalfFloat:       false,
    oesTextureHalfFloatLinear: false,
    oesVertexArrayObject:      false,
    angleInstancedArrays:      false,
    extBlendMinMax:            false,
    extFragDepth:              false,
    extShaderTextureLod:       false,
    extTextureFilterAnisotropic: false,
    extColorBufferFloat:       false,
    extColorBufferHalfFloat:   false,
    extSrgb:                   false,
    webglDepthTexture:         false,
    webglDrawBuffers:          false,
    khrParallelShaderCompile:  false,
    extDisjointTimerQuery:     false,
    webglMultiDraw:            false,
    webglDebugRendererInfo:    false,
    webglLoseContext:          false,
    extTextureCompressionAstc: false,
  },

  compressedFormats: COMPRESSED_FORMAT.NONE,
};

function _probe() {
  if (typeof document === 'undefined') return;

  let canvas = null;
  let gl = null;

  try {
    canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;

    const gl2 = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });

    gl = gl2 || canvas.getContext('webgl', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });

    if (!gl) return;

    _raw.available = true;
    _raw.webgl2 = !!gl2;
    _raw.glslVersion = gl2 ? SHADING_LANGUAGE.GLSL_300 : SHADING_LANGUAGE.GLSL_100;

    _raw.maxTextureSize           = gl.getParameter(gl.MAX_TEXTURE_SIZE)            || 2048;
    _raw.maxCubeMapSize           = gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE)   || 2048;
    _raw.maxRenderbufferSize      = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)       || 2048;
    _raw.maxTextureImageUnits     = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)     || 8;
    _raw.maxVertexTextureImageUnits = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) || 0;
    _raw.maxCombinedTextureImageUnits = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) || 8;
    _raw.maxVertexAttribs         = gl.getParameter(gl.MAX_VERTEX_ATTRIBS)          || 16;
    _raw.maxVertexUniformVectors  = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS)  || 128;
    _raw.maxFragmentUniformVectors= gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS)|| 128;
    _raw.maxVaryingVectors        = gl.getParameter(gl.MAX_VARYING_VECTORS)         || 8;

    if (gl2) {
      _raw.maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) || 1;
      _raw.maxSamples     = gl.getParameter(gl.MAX_SAMPLES)      || 0;
    } else if (gl.getParameter(gl.MAX_DRAW_BUFFERS)) {
      _raw.maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) || 1;
    }

    // Precision probes per stage.
    const pf = (stage, qual) => {
      const fmt = gl.getShaderPrecisionFormat(stage, qual);
      if (!fmt) return PRECISION_SUPPORT.NONE;
      if (fmt.precision === 0) return PRECISION_SUPPORT.NONE;
      if (fmt.precision >= 23 && fmt.rangeMin >= 127) return PRECISION_SUPPORT.HIGH;
      if (fmt.precision >= 10) return PRECISION_SUPPORT.MEDIUM;
      return PRECISION_SUPPORT.LOW;
    };

    _raw.precision.vertex   = pf(gl.VERTEX_SHADER,   gl.HIGH_FLOAT);
    _raw.precision.fragment = pf(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);

    // Extensions.
    const ext = (name) => {
      try { return gl.getExtension(name) !== null; } catch (_) { return false; }
    };

    _raw.extensions.oesElementIndexUint       = ext('OES_element_index_uint');
    _raw.extensions.oesStandardDerivatives    = ext('OES_standard_derivatives');
    _raw.extensions.oesTextureFloat           = ext('OES_texture_float');
    _raw.extensions.oesTextureFloatLinear     = ext('OES_texture_float_linear');
    _raw.extensions.oesTextureHalfFloat       = ext('OES_texture_half_float');
    _raw.extensions.oesTextureHalfFloatLinear = ext('OES_texture_half_float_linear');
    _raw.extensions.oesVertexArrayObject      = ext('OES_vertex_array_object');
    _raw.extensions.angleInstancedArrays      = ext('ANGLE_instanced_arrays');
    _raw.extensions.extBlendMinMax            = ext('EXT_blend_minmax');
    _raw.extensions.extFragDepth              = ext('EXT_frag_depth');
    _raw.extensions.extShaderTextureLod       = ext('EXT_shader_texture_lod');
    _raw.extensions.extTextureFilterAnisotropic = ext('EXT_texture_filter_anisotropic');
    _raw.extensions.extColorBufferFloat       = ext('EXT_color_buffer_float') || ext('WEBGL_color_buffer_float');
    _raw.extensions.extColorBufferHalfFloat   = ext('EXT_color_buffer_half_float');
    _raw.extensions.extSrgb                   = ext('EXT_sRGB');
    _raw.extensions.webglDepthTexture         = ext('WEBGL_depth_texture');
    _raw.extensions.webglDrawBuffers          = ext('WEBGL_draw_buffers');
    _raw.extensions.khrParallelShaderCompile  = ext('KHR_parallel_shader_compile');
    _raw.extensions.extDisjointTimerQuery     = ext('EXT_disjoint_timer_query') || ext('EXT_disjoint_timer_query_webgl2');
    _raw.extensions.webglMultiDraw            = ext('WEBGL_multi_draw');
    _raw.extensions.webglDebugRendererInfo    = ext('WEBGL_debug_renderer_info');
    _raw.extensions.webglLoseContext          = ext('WEBGL_lose_context');
    _raw.extensions.extTextureCompressionAstc = ext('WEBGL_compressed_texture_astc');

    // Anisotropy.
    if (_raw.extensions.extTextureFilterAnisotropic) {
      try {
        const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
        _raw.maxAnisotropy = gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) || 1;
      } catch (_) {
        _raw.maxAnisotropy = 1;
      }
    }

    // Compressed texture formats.
    let compressedMask = 0;
    if (ext('WEBGL_compressed_texture_s3tc'))        compressedMask |= COMPRESSED_FORMAT.DXT;
    if (ext('WEBGL_compressed_texture_etc1'))        compressedMask |= COMPRESSED_FORMAT.ETC1;
    if (ext('WEBGL_compressed_texture_etc'))         compressedMask |= COMPRESSED_FORMAT.ETC2;
    if (ext('WEBGL_compressed_texture_astc'))        compressedMask |= COMPRESSED_FORMAT.ASTC;
    if (ext('WEBGL_compressed_texture_pvrtc'))       compressedMask |= COMPRESSED_FORMAT.PVRTC;
    _raw.compressedFormats = compressedMask;

    // Release probe context.
    if (_raw.extensions.webglLoseContext) {
      try {
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
      } catch (_) { /* swallow */ }
    }
  } catch (_) {
    // Leave conservative defaults.
  } finally {
    canvas = null;
    gl = null;
  }
}

_probe();

export const RAW_CAPS_DEEP = Object.freeze({
  available: _raw.available,
  webgl2: _raw.webgl2,
  glslVersion: _raw.glslVersion,

  maxTextureSize: _raw.maxTextureSize,
  maxCubeMapSize: _raw.maxCubeMapSize,
  maxRenderbufferSize: _raw.maxRenderbufferSize,
  maxTextureImageUnits: _raw.maxTextureImageUnits,
  maxVertexTextureImageUnits: _raw.maxVertexTextureImageUnits,
  maxCombinedTextureImageUnits: _raw.maxCombinedTextureImageUnits,
  maxVertexAttribs: _raw.maxVertexAttribs,
  maxVertexUniformVectors: _raw.maxVertexUniformVectors,
  maxFragmentUniformVectors: _raw.maxFragmentUniformVectors,
  maxVaryingVectors: _raw.maxVaryingVectors,
  maxDrawBuffers: _raw.maxDrawBuffers,
  maxSamples: _raw.maxSamples,
  maxAnisotropy: _raw.maxAnisotropy,

  precision: Object.freeze(Object.assign({}, _raw.precision)),
  extensions: Object.freeze(Object.assign({}, _raw.extensions)),
  compressedFormats: _raw.compressedFormats,
});

/* ------------------------------------------------------------------ */
/* 2. FEATURE FLAGS (derived decisions the lighting stack branches on) */
/* ------------------------------------------------------------------ */

function _computeFeatureFlags() {
  const caps = RAW_CAPS_DEEP;
  const pconf = PLATFORM_CONFIG;
  const aprofile = ANDROID_PROFILE;

  const isAndroid = DEVICE.isAndroid;
  const isMobile  = DEVICE.isMobile;

  const hasHighpFragment = caps.precision.fragment === PRECISION_SUPPORT.HIGH;
  const hasHighpVertex   = caps.precision.vertex   === PRECISION_SUPPORT.HIGH;

  const supportsHalfFloatColor =
    caps.extensions.extColorBufferHalfFloat ||
    (caps.webgl2 && caps.extensions.extColorBufferFloat);

  const supportsFloatColor   = caps.extensions.extColorBufferFloat;
  const supportsDepthTexture = caps.webgl2 || caps.extensions.webglDepthTexture;
  const supportsMRT          = caps.maxDrawBuffers > 1 || caps.extensions.webglDrawBuffers;
  const supportsInstancing   = caps.webgl2 || caps.extensions.angleInstancedArrays;
  const supportsVAO          = caps.webgl2 || caps.extensions.oesVertexArrayObject;
  const supportsDerivatives  = caps.webgl2 || caps.extensions.oesStandardDerivatives;
  const supportsUintIndex    = caps.webgl2 || caps.extensions.oesElementIndexUint;
  const supportsFragDepth    = caps.webgl2 || caps.extensions.extFragDepth;
  const supportsShaderLod    = caps.webgl2 || caps.extensions.extShaderTextureLod;
  const supportsFloatLinear  = caps.webgl2 || caps.extensions.oesTextureFloatLinear;
  const supportsHalfLinear   = caps.webgl2 || caps.extensions.oesTextureHalfFloatLinear;
  const supportsAsyncCompile = caps.extensions.khrParallelShaderCompile;
  const supportsGpuTimers    = caps.extensions.extDisjointTimerQuery;
  const supportsVertexTex    = caps.maxVertexTextureImageUnits > 0;

  // Feature decisions.
  const canUseHDRTargets = supportsHalfFloatColor && hasHighpFragment;
  const canUseFloatTargets = supportsFloatColor && hasHighpFragment;
  const canUseDepthTexture = supportsDepthTexture;
  const canUseMRT = supportsMRT && caps.maxDrawBuffers >= 2;
  const canUseMRT4 = supportsMRT && caps.maxDrawBuffers >= 4;
  const canUseInstancing = supportsInstancing;
  const canUseVAO = supportsVAO;
  const canUseDerivatives = supportsDerivatives;
  const canUseUint32Indices = supportsUintIndex;
  const canUseFragmentDepth = supportsFragDepth;
  const canUseShaderTextureLOD = supportsShaderLod;
  const canUseVertexTextures = supportsVertexTex;

  // Shadow filter decisions.
  const canUseBasicShadow  = true;
  const canUsePCFShadow    = supportsDerivatives; // PCF needs dFdx/dFdy for cheap offset
  const canUsePCFSoftShadow= canUsePCFShadow && caps.maxTextureSize >= 1024;
  const canUsePCSS         = canUsePCFSoftShadow && !pconf.hasQuirk(QUIRK.POWERVR_TBDR_DEPTH_COST) && !isMobile;
  const canUseVSM          = supportsFloatColor && !isMobile;
  const canUseESM          = false; // never on mobile — expensive

  // GI decisions.
  const canUseGIProbeGrid  = caps.maxTextureSize >= 512 && supportsDerivatives;
  const canUseGIHalfRes    = !hasHighpFragment || caps.maxTextureSize < 2048;
  const canUseGIMultiBounce= !isMobile || PERF_TIER_LOCAL === 'HIGH';

  // AO decisions.
  const canUseSSAO  = supportsDerivatives && caps.maxTextureSize >= 512;
  const canUseHBAO  = supportsDerivatives && caps.maxTextureSize >= 1024;
  const canUseGTAO  = supportsDerivatives && caps.maxTextureSize >= 2048 && !isMobile;
  const canUseTemporalAO = supportsDepthTexture && caps.maxTextureSize >= 1024;

  // Cluster / forward+ decisions.
  const canUseClusterLighting = supportsMRT && caps.maxTextureSize >= 512;
  const canUseForwardPlus     = supportsMRT && caps.maxTextureSize >= 1024 && !pconf.hasQuirk(QUIRK.POWERVR_TBDR_DEPTH_COST);
  const canUseRectAreaLTC     = supportsHalfFloatColor && caps.maxTextureSize >= 512 && !isMobile;

  // Post decisions.
  const canUseBloom           = canUseHDRTargets;
  const canUseVolumetric      = supportsMRT && caps.maxTextureSize >= 512;
  const canUseSSGI            = supportsMRT && supportsFloatColor && !isMobile;
  const canUseTAA             = supportsDepthTexture && !pconf.hasQuirk(QUIRK.MSAA_UNRELIABLE);
  const canUseMotionBlur      = supportsDepthTexture && !isMobile;
  const canUseToneMapping     = true;
  const canUseColorGrading    = true;

  // MSAA decisions.
  const canUseMSAA2 = caps.maxSamples >= 2 && !pconf.hasQuirk(QUIRK.MSAA_UNRELIABLE);
  const canUseMSAA4 = caps.maxSamples >= 4 && !pconf.hasQuirk(QUIRK.MSAA_UNRELIABLE);
  const canUseMSAA8 = caps.maxSamples >= 8 && !pconf.hasQuirk(QUIRK.MSAA_UNRELIABLE);

  // Precision hints for shader generation.
  const shaderHighpFragment  = hasHighpFragment;
  const shaderHighpVertex    = hasHighpVertex;
  const shaderPositionMedium = isAndroid && (GPU_FAMILY >= 7 && GPU_FAMILY <= 9); // Mali-T/G3x/G5x

  return Object.freeze({
    // ---- Raw availability ----
    hasWebGL:        caps.available,
    hasWebGL2:       caps.webgl2,
    hasHighpVertex:  hasHighpVertex,
    hasHighpFragment:hasHighpFragment,

    // ---- Texture / RT formats ----
    supportsHalfFloatColor,
    supportsFloatColor,
    supportsDepthTexture,
    supportsMRT,
    supportsMRT4,
    supportsVertexTextures,

    // ---- Extensions used by lighting ----
    supportsInstancing,
    supportsVAO,
    supportsDerivatives,
    supportsUint32Indices,
    supportsFragmentDepth,
    supportsShaderTextureLOD,
    supportsFloatLinear,
    supportsHalfLinear,
    supportsAsyncCompile,
    supportsGpuTimers,

    // ---- High-level feature decisions ----
    canUseHDRTargets,
    canUseFloatTargets,
    canUseDepthTexture,
    canUseMRT,
    canUseMRT4,
    canUseInstancing,
    canUseVAO,
    canUseDerivatives,
    canUseUint32Indices,
    canUseFragmentDepth,
    canUseShaderTextureLOD,
    canUseVertexTextures,

    // ---- Shadows ----
    canUseBasicShadow,
    canUsePCFShadow,
    canUsePCFSoftShadow,
    canUsePCSS,
    canUseVSM,
    canUseESM,

    // ---- GI ----
    canUseGIProbeGrid,
    canUseGIHalfRes,
    canUseGIMultiBounce,

    // ---- AO ----
    canUseSSAO,
    canUseHBAO,
    canUseGTAO,
    canUseTemporalAO,

    // ---- Cluster / forward+ ----
    canUseClusterLighting,
    canUseForwardPlus,
    canUseRectAreaLTC,

    // ---- Post ----
    canUseBloom,
    canUseVolumetric,
    canUseSSGI,
    canUseTAA,
    canUseMotionBlur,
    canUseToneMapping,
    canUseColorGrading,

    // ---- MSAA ----
    canUseMSAA2,
    canUseMSAA4,
    canUseMSAA8,

    // ---- Shader precision ----
    shaderHighpFragment,
    shaderHighpVertex,
    shaderPositionMedium,
  });
}

export const FEATURES = _computeFeatureFlags();

/* ------------------------------------------------------------------ */
/* 3. CANONICAL CAPABILITIES OBJECT (read-only surface)               */
/* ------------------------------------------------------------------ */

export const CAPABILITIES = Object.freeze({
  name:          'capabilities',
  platform:      Object.freeze({
    isAndroid:   DEVICE.isAndroid,
    isIOS:       DEVICE.isIOS,
    isMobile:    DEVICE.isMobile,
    isDesktop:   DEVICE.isDesktop,
    webglVersion:DEVICE.webglVersion,
    perfTier:    PERF_TIER_LOCAL,
    profileTier: PLATFORM_CONFIG.tier,
    gpuFamily:   GPU_FAMILY_NAME,
  }),

  raw:       RAW_CAPS_DEEP,
  features:  FEATURES,
  quirks:    QUIRKS,

  has(name) {
    return FEATURES[name] === true;
  },

  hasQuirk(name) {
    return QUIRKS.indexOf(name) >= 0;
  },
});

/* ------------------------------------------------------------------ */
/* 4. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

export function hasFeature(name) {
  return FEATURES[name] === true;
}

export function hasQuirk(name) {
  return QUIRKS.indexOf(name) >= 0;
}

export function hasExtension(name) {
  if (!name || typeof name !== 'string') return false;
  return RAW_CAPS_DEEP.extensions[name] === true;
}

export function supportsCompressedFormat(flag) {
  return (RAW_CAPS_DEEP.compressedFormats & flag) !== 0;
}

export function getMaxTextureSize()  { return RAW_CAPS_DEEP.maxTextureSize; }
export function getMaxDrawBuffers()  { return RAW_CAPS_DEEP.maxDrawBuffers; }
export function getMaxSamples()      { return RAW_CAPS_DEEP.maxSamples; }
export function getMaxAnisotropy()   { return RAW_CAPS_DEEP.maxAnisotropy; }
export function getMaxTextureUnits() { return RAW_CAPS_DEEP.maxTextureImageUnits; }
export function getMaxVertexAttribs(){ return RAW_CAPS_DEEP.maxVertexAttribs; }

/* ------------------------------------------------------------------ */
/* 5. SHADER PRECISION HELPERS                                        */
/* ------------------------------------------------------------------ */

/**
 * Returns the recommended GLSL precision keyword for a given stage based
 * on the runtime probe.
 */
export function recommendedPrecisionKeyword(stage) {
  const p = stage === 'vertex'
    ? RAW_CAPS_DEEP.precision.vertex
    : RAW_CAPS_DEEP.precision.fragment;

  if (p === PRECISION_SUPPORT.HIGH)   return 'highp';
  if (p === PRECISION_SUPPORT.MEDIUM) return 'mediump';
  if (p === PRECISION_SUPPORT.LOW)    return 'lowp';
  return 'mediump';
}

/**
 * Returns a full GLSL precision prelude derived from the runtime probe,
 * suitable for prepending to any anime lighting shader.
 */
export function capabilityPrecisionPrelude() {
  const f = recommendedPrecisionKeyword('fragment');
  const v = recommendedPrecisionKeyword('vertex');
  return [
    `#ifndef CAPABILITY_PRECISION_PRELUDE`,
    `#define CAPABILITY_PRECISION_PRELUDE`,
    `precision ${f} float;`,
    `precision ${f} int;`,
    `#ifdef VERTEX`,
    `precision ${v} float;`,
    `#endif`,
    `#endif`,
  ].join('\n');
}

/* ------------------------------------------------------------------ */
/* 6. SHADER DEFINE HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Returns the array of preprocessor defines a lighting shader should use
 * based on the runtime capabilities.
 */
export function capabilityShaderDefines() {
  const defs = [];
  if (FEATURES.hasWebGL2)              defs.push('HAS_WEBGL2');
  if (FEATURES.hasHighpFragment)       defs.push('HAS_HIGHP_FRAGMENT');
  if (FEATURES.hasHighpVertex)         defs.push('HAS_HIGHP_VERTEX');
  if (FEATURES.supportsMRT)            defs.push('HAS_MRT');
  if (FEATURES.supportsMRT4)           defs.push('HAS_MRT4');
  if (FEATURES.supportsHalfFloatColor) defs.push('HAS_HALF_FLOAT_COLOR');
  if (FEATURES.supportsFloatColor)     defs.push('HAS_FLOAT_COLOR');
  if (FEATURES.supportsDepthTexture)   defs.push('HAS_DEPTH_TEXTURE');
  if (FEATURES.supportsInstancing)     defs.push('HAS_INSTANCING');
  if (FEATURES.supportsVAO)            defs.push('HAS_VAO');
  if (FEATURES.supportsDerivatives)    defs.push('HAS_DERIVATIVES');
  if (FEATURES.supportsUint32Indices)  defs.push('HAS_UINT32_INDICES');
  if (FEATURES.supportsFragmentDepth)  defs.push('HAS_FRAGMENT_DEPTH');
  if (FEATURES.supportsShaderTextureLOD) defs.push('HAS_SHADER_TEXTURE_LOD');
  if (FEATURES.canUsePCSS)             defs.push('CAN_USE_PCSS');
  if (FEATURES.canUsePCFSoftShadow)    defs.push('CAN_USE_PCF_SOFT');
  if (FEATURES.canUseHDRTargets)       defs.push('CAN_USE_HDR_TARGETS');
  if (FEATURES.canUseGIProbeGrid)      defs.push('CAN_USE_GI_PROBE_GRID');
  if (FEATURES.canUseSSAO)             defs.push('CAN_USE_SSAO');
  if (FEATURES.canUseHBAO)             defs.push('CAN_USE_HBAO');
  if (FEATURES.canUseGTAO)             defs.push('CAN_USE_GTAO');
  if (FEATURES.canUseForwardPlus)      defs.push('CAN_USE_FORWARD_PLUS');
  if (FEATURES.canUseClusterLighting)  defs.push('CAN_USE_CLUSTER_LIGHTING');
  if (FEATURES.canUseBloom)            defs.push('CAN_USE_BLOOM');
  if (FEATURES.canUseVolumetric)       defs.push('CAN_USE_VOLUMETRIC');
  if (FEATURES.canUseSSGI)             defs.push('CAN_USE_SSGI');
  if (FEATURES.canUseTAA)              defs.push('CAN_USE_TAA');
  if (FEATURES.shaderPositionMedium)   defs.push('POSITION_VARYING_MEDIUMP');
  for (let i = 0; i < QUIRKS.length; i++) {
    defs.push('QUIRK_' + QUIRKS[i].toUpperCase().replace(/[^A-Z0-9]/g, '_'));
  }
  return defs;
}

/**
 * Produces a GLSL string with all capability defines for prepending to a
 * shader source.
 */
export function capabilityDefinesPrelude() {
  const defs = capabilityShaderDefines();
  const lines = ['#ifndef CAPABILITY_DEFINES_PRELUDE', '#define CAPABILITY_DEFINES_PRELUDE'];
  for (let i = 0; i < defs.length; i++) lines.push('#define ' + defs[i]);
  lines.push('#endif');
  return lines.join('\n');
}

/**
 * Full prelude (precision + defines) ready to prepend to a lighting shader.
 */
export function capabilityShaderPrelude() {
  return capabilityPrecisionPrelude() + '\n' + capabilityDefinesPrelude();
}

/* ------------------------------------------------------------------ */
/* 7. PER-FEATURE SAFETY WRAPPERS                                     */
/* ------------------------------------------------------------------ */

/**
 * Returns the safest shadow filter the current device can use.
 */
export function safestShadowFilter() {
  if (FEATURES.canUsePCSS)        return 'pcss';
  if (FEATURES.canUsePCFSoftShadow) return 'pcfsoft';
  if (FEATURES.canUsePCFShadow)   return 'pcf';
  return 'basic';
}

/**
 * Returns the safest HDR color format the current device can render to.
 */
export function safestHDRColorFormat() {
  if (FEATURES.canUseFloatTargets) return 'rgba32f';
  if (FEATURES.canUseHDRTargets)   return 'rgba16f';
  return 'rgba8';
}

/**
 * Returns the highest MSAA sample count the current device can use, capped
 * by a caller-provided limit.
 */
export function safestMSAASamples(limit) {
  const cap = Math.max(0, limit | 0);
  if (cap >= 8 && FEATURES.canUseMSAA8) return 8;
  if (cap >= 4 && FEATURES.canUseMSAA4) return 4;
  if (cap >= 2 && FEATURES.canUseMSAA2) return 2;
  return 0;
}

/* ------------------------------------------------------------------ */
/* 8. DIAGNOSTICS                                                     */
/* ------------------------------------------------------------------ */

export function getCapabilitiesReport() {
  return {
    available:      RAW_CAPS_DEEP.available,
    webgl2:         RAW_CAPS_DEEP.webgl2,
    glslVersion:    RAW_CAPS_DEEP.glslVersion === SHADING_LANGUAGE.GLSL_300 ? '300 es' : '100',
    maxTextureSize: RAW_CAPS_DEEP.maxTextureSize,
    maxDrawBuffers: RAW_CAPS_DEEP.maxDrawBuffers,
    maxSamples:     RAW_CAPS_DEEP.maxSamples,
    maxAnisotropy:  RAW_CAPS_DEEP.maxAnisotropy,
    precision:      RAW_CAPS_DEEP.precision,
    features:       FEATURES,
    extensions:     RAW_CAPS_DEEP.extensions,
    quirks:         QUIRKS.slice(),
    perfTier:       PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  CAPABILITIES,
  RAW_CAPS_DEEP,
  FEATURES,
  SHADING_LANGUAGE,
  PRECISION_SUPPORT,
  COMPRESSED_FORMAT,

  hasFeature,
  hasQuirk,
  hasExtension,
  supportsCompressedFormat,

  getMaxTextureSize,
  getMaxDrawBuffers,
  getMaxSamples,
  getMaxAnisotropy,
  getMaxTextureUnits,
  getMaxVertexAttribs,

  recommendedPrecisionKeyword,
  capabilityPrecisionPrelude,
  capabilityShaderDefines,
  capabilityDefinesPrelude,
  capabilityShaderPrelude,

  safestShadowFilter,
  safestHDRColorFormat,
  safestMSAASamples,

  getCapabilitiesReport,
};

export default _defaultExport;