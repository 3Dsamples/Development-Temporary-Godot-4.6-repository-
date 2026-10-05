API Documentation — src/core/021_rnd_Capabilities.js

File Purpose

This file is the runtime capability probe and feature-flag resolver for the anime lighting stack. Where 018_rnd_PlatformConfig.js owns device classification and 019/020 own per-platform tuning tables, this module owns the canonical runtime capability surface: the concrete set of GPU features, extensions, precision limits, texture and render target constraints, and driver quirks that a lighting subsystem must interrogate before enabling a feature.

The distinction is important:

· 018_rnd_PlatformConfig.js answers "what tier is this device?" — a coarse bucket.
· 019_rnd_AndroidProfile.js and 020_rnd_DesktopProfile.js answer "how should I tune it?" — numeric tables.
· 021_rnd_Capabilities.js answers "does this specific driver support this specific operation?" — the ground truth.

A device might be classified as HIGH tier and have a MID-tier Android profile, but if its specific driver rejects EXT_color_buffer_float, the canUseFloatTargets flag here will be false, and every subsystem that checks it will correctly avoid float render targets. This module is where the engine stops trusting assumptions and starts trusting the actual GPU.

The probe runs exactly once at module load. It creates a temporary WebGL context, queries every relevant capability, then releases the context via WEBGL_lose_context. No context stays alive after this module is imported.

The output has three layers:

1. RAW_CAPS_DEEP — the raw capability values, in their native units. Precision is a three-level enum, extensions are booleans, limits are integers.
2. FEATURES — the derived boolean decisions the lighting stack actually branches on. canUseHDRTargets, canUsePCSS, canUseGIProbeGrid, canUseMRT, and so on.
3. CAPABILITIES — a frozen object that bundles both with convenience methods has(name) and hasQuirk(name).

There are also a set of query helpers (hasFeature, hasExtension, recommendedPrecisionKeyword, capabilityShaderPrelude, safestShadowFilter, safestHDRColorFormat, safestMSAASamples) that downstream code uses to translate capability into concrete settings without re-implementing the checks.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

SHADING_LANGUAGE

Type: frozen enum

Values:

· GLSL_100 = 0 — WebGL1 shader language.
· GLSL_300 = 1 — WebGL2 shader language (#version 300 es).

PRECISION_SUPPORT

Type: frozen enum

Values:

· NONE = 0 — precision query returned zero or failed.
· LOW = 1 — lowp only.
· MEDIUM = 2 — mediump supported.
· HIGH = 3 — highp supported (23-bit mantissa, range 127).

COMPRESSED_FORMAT

Type: frozen enum

Bitmask flags for the supported compressed texture families.

Values:

· NONE = 0
· DXT = 1 << 0 — desktop S3TC.
· ETC1 = 1 << 1 — Android classic.
· ETC2 = 1 << 2 — Android modern.
· ASTC = 1 << 3 — modern Android + Apple.
· PVRTC = 1 << 4 — PowerVR.

RAW_CAPS_DEEP

Type: frozen object

The full raw probe result. Fields:

· available — boolean, whether a WebGL context could be created.
· webgl2 — boolean, whether the context is WebGL2.
· glslVersion — one of SHADING_LANGUAGE.
· maxTextureSize — MAX_TEXTURE_SIZE.
· maxCubeMapSize — MAX_CUBE_MAP_TEXTURE_SIZE.
· maxRenderbufferSize — MAX_RENDERBUFFER_SIZE.
· maxTextureImageUnits — MAX_TEXTURE_IMAGE_UNITS.
· maxVertexTextureImageUnits — MAX_VERTEX_TEXTURE_IMAGE_UNITS.
· maxCombinedTextureImageUnits — MAX_COMBINED_TEXTURE_IMAGE_UNITS.
· maxVertexAttribs — MAX_VERTEX_ATTRIBS.
· maxVertexUniformVectors — MAX_VERTEX_UNIFORM_VECTORS.
· maxFragmentUniformVectors — MAX_FRAGMENT_UNIFORM_VECTORS.
· maxVaryingVectors — MAX_VARYING_VECTORS.
· maxDrawBuffers — MAX_DRAW_BUFFERS on WebGL2, otherwise 1.
· maxSamples — MAX_SAMPLES on WebGL2, otherwise 0.
· maxAnisotropy — from EXT_texture_filter_anisotropic, or 1.
· precision — an object with vertex and fragment fields, both one of PRECISION_SUPPORT.
· extensions — an object with 23 extension-presence booleans.
· compressedFormats — a bitmask of COMPRESSED_FORMAT values.

The extensions fields:

· oesElementIndexUint — OES_element_index_uint.
· oesStandardDerivatives — OES_standard_derivatives.
· oesTextureFloat — OES_texture_float.
· oesTextureFloatLinear — OES_texture_float_linear.
· oesTextureHalfFloat — OES_texture_half_float.
· oesTextureHalfFloatLinear — OES_texture_half_float_linear.
· oesVertexArrayObject — OES_vertex_array_object.
· angleInstancedArrays — ANGLE_instanced_arrays.
· extBlendMinMax — EXT_blend_minmax.
· extFragDepth — EXT_frag_depth.
· extShaderTextureLod — EXT_shader_texture_lod.
· extTextureFilterAnisotropic — EXT_texture_filter_anisotropic.
· extColorBufferFloat — EXT_color_buffer_float or WEBGL_color_buffer_float.
· extColorBufferHalfFloat — EXT_color_buffer_half_float.
· extSrgb — EXT_sRGB.
· webglDepthTexture — WEBGL_depth_texture.
· webglDrawBuffers — WEBGL_draw_buffers.
· khrParallelShaderCompile — KHR_parallel_shader_compile.
· extDisjointTimerQuery — EXT_disjoint_timer_query or EXT_disjoint_timer_query_webgl2.
· webglMultiDraw — WEBGL_multi_draw.
· webglDebugRendererInfo — WEBGL_debug_renderer_info.
· webglLoseContext — WEBGL_lose_context.
· extTextureCompressionAstc — WEBGL_compressed_texture_astc.

FEATURES

Type: frozen object

The derived feature flags. Downstream lighting code reads these booleans.

Raw availability:

· hasWebGL — the context could be created.
· hasWebGL2 — the context is WebGL2.
· hasHighpVertex — vertex highp is supported.
· hasHighpFragment — fragment highp is supported.

Texture and render target support:

· supportsHalfFloatColor — half-float color render targets are available.
· supportsFloatColor — full-float color render targets are available.
· supportsDepthTexture — depth textures are available.
· supportsMRT — multiple render targets are available.
· supportsMRT4 — at least four draw buffers.
· supportsVertexTextures — vertex texture fetch is available.

Extensions used by lighting:

· supportsInstancing — instanced draw calls are available.
· supportsVAO — vertex array objects are available.
· supportsDerivatives — dFdx/dFdy are available (needed for PCF shadow offsets, SSAO, and many others).
· supportsUint32Indices — 32-bit vertex indices are available.
· supportsFragmentDepth — writing to gl_FragDepth is available.
· supportsShaderTextureLOD — textureLod is available in fragment shaders.
· supportsFloatLinear — float textures can be linearly filtered.
· supportsHalfLinear — half-float textures can be linearly filtered.
· supportsAsyncCompile — KHR_parallel_shader_compile is available.
· supportsGpuTimers — EXT_disjoint_timer_query is available.

High-level feature decisions:

· canUseHDRTargets — half-float color plus highp fragment.
· canUseFloatTargets — full float color plus highp fragment.
· canUseDepthTexture — depth textures available.
· canUseMRT — 2+ draw buffers.
· canUseMRT4 — 4+ draw buffers.
· canUseInstancing, canUseVAO, canUseDerivatives, canUseUint32Indices, canUseFragmentDepth, canUseShaderTextureLOD, canUseVertexTextures — direct mappings.

Shadows:

· canUseBasicShadow — always true.
· canUsePCFShadow — requires derivatives.
· canUsePCFSoftShadow — requires PCF plus maxTextureSize >= 1024.
· canUsePCSS — requires PCFSoft plus no PowerVR TBDR quirk plus non-mobile.
· canUseVSM — requires float color plus non-mobile.
· canUseESM — always false. ESM is never enabled on mobile because it requires an extra blur pass.

GI:

· canUseGIProbeGrid — maxTextureSize >= 512 plus derivatives.
· canUseGIHalfRes — true if fragment highp is unavailable, or if maxTextureSize < 2048.
· canUseGIMultiBounce — non-mobile, or HIGH tier.

AO:

· canUseSSAO — derivatives plus maxTextureSize >= 512.
· canUseHBAO — derivatives plus maxTextureSize >= 1024.
· canUseGTAO — derivatives plus maxTextureSize >= 2048 plus non-mobile.
· canUseTemporalAO — depth texture plus maxTextureSize >= 1024.

Cluster and forward+:

· canUseClusterLighting — MRT plus maxTextureSize >= 512.
· canUseForwardPlus — MRT plus maxTextureSize >= 1024 plus no PowerVR TBDR quirk.
· canUseRectAreaLTC — half-float color plus maxTextureSize >= 512 plus non-mobile.

Post:

· canUseBloom — same as HDR targets.
· canUseVolumetric — MRT plus maxTextureSize >= 512.
· canUseSSGI — MRT plus float color plus non-mobile.
· canUseTAA — depth texture plus no MSAA unreliability quirk.
· canUseMotionBlur — depth texture plus non-mobile.
· canUseToneMapping — always true.
· canUseColorGrading — always true.

MSAA:

· canUseMSAA2 — maxSamples >= 2 and no MSAA unreliability quirk.
· canUseMSAA4 — same, threshold 4.
· canUseMSAA8 — same, threshold 8.

Precision hints:

· shaderHighpFragment — highp fragment is available.
· shaderHighpVertex — highp vertex is available.
· shaderPositionMedium — true if the device is Android with a Mali-T, Mali-G3x, or Mali-G5x GPU.

CAPABILITIES

Type: frozen object

The canonical capabilities surface. Fields:

· name — always 'capabilities'.
· platform — a frozen sub-object with isAndroid, isIOS, isMobile, isDesktop, webglVersion, perfTier, profileTier, gpuFamily.
· raw — the RAW_CAPS_DEEP object.
· features — the FEATURES object.
· quirks — the quirks array from 018_rnd_PlatformConfig.js.
· has(name) — a method that returns FEATURES[name] === true.
· hasQuirk(name) — a method that checks the quirks array.

---

Internal State (Not Exported Directly)

_raw

Type: mutable object

The working object the probe fills in. Once _probe() completes, this object's values are frozen into RAW_CAPS_DEEP, and the working object is discarded.

HAS_WINDOW, HAS_DOCUMENT, HAS_NAVIGATOR

Internal booleans used to guard every DOM access.

---

Exported Functions

hasFeature(name)

Parameters: name — the feature name.

Returns: FEATURES[name] === true.

Purpose: the fast query. Every subsystem that needs a capability check calls this.

hasQuirk(name)

Parameters: name — the quirk name.

Returns: boolean.

Purpose: the quirk check. Reads the quirks array from 018_rnd_PlatformConfig.js.

hasExtension(name)

Parameters: name — the extension field name (e.g. 'oesStandardDerivatives').

Returns: RAW_CAPS_DEEP.extensions[name] === true.

Purpose: the raw extension check. Use this only when hasFeature does not provide a derived flag.

supportsCompressedFormat(flag)

Parameters: flag — a COMPRESSED_FORMAT bitmask value.

Returns: boolean.

Purpose: reports whether any of the requested compressed formats are supported.

getMaxTextureSize()

Returns: RAW_CAPS_DEEP.maxTextureSize.

getMaxDrawBuffers()

Returns: RAW_CAPS_DEEP.maxDrawBuffers.

getMaxSamples()

Returns: RAW_CAPS_DEEP.maxSamples.

getMaxAnisotropy()

Returns: RAW_CAPS_DEEP.maxAnisotropy.

getMaxTextureUnits()

Returns: RAW_CAPS_DEEP.maxTextureImageUnits.

getMaxVertexAttribs()

Returns: RAW_CAPS_DEEP.maxVertexAttribs.

Purpose: the six direct limit accessors. Each is O(1) and returns a number.

recommendedPrecisionKeyword(stage)

Parameters: stage — 'vertex' or 'fragment'.

Returns: one of 'highp', 'mediump', 'lowp'.

Purpose: reads the precision enum for the requested stage and maps it to the GLSL keyword. Returns 'mediump' if the enum is NONE.

capabilityPrecisionPrelude()

Returns: a GLSL string with the precision prelude:

```
#ifndef CAPABILITY_PRECISION_PRELUDE
#define CAPABILITY_PRECISION_PRELUDE
precision <fragmentPrecision> float;
precision <fragmentPrecision> int;
#ifdef VERTEX
precision <vertexPrecision> float;
#endif
#endif
```

Purpose: ready to prepend to any shader.

capabilityShaderDefines()

Returns: an array of preprocessor define names that reflect the current capabilities. Includes flags like HAS_WEBGL2, HAS_HIGHP_FRAGMENT, HAS_MRT, HAS_MRT4, HAS_HALF_FLOAT_COLOR, CAN_USE_PCSS, CAN_USE_PCF_SOFT, CAN_USE_HDR_TARGETS, CAN_USE_GI_PROBE_GRID, CAN_USE_SSAO, CAN_USE_HBAO, CAN_USE_GTAO, CAN_USE_FORWARD_PLUS, CAN_USE_CLUSTER_LIGHTING, CAN_USE_BLOOM, CAN_USE_VOLUMETRIC, CAN_USE_SSGI, CAN_USE_TAA, POSITION_VARYING_MEDIUMP, plus one QUIRK_<NAME> define per active quirk.

capabilityDefinesPrelude()

Returns: a GLSL string with all the defines from capabilityShaderDefines() wrapped in a single #ifndef guard.

capabilityShaderPrelude()

Returns: the concatenation of capabilityPrecisionPrelude() and capabilityDefinesPrelude().

Purpose: the single function a shader compilation pipeline calls to get the full prelude.

safestShadowFilter()

Returns: one of 'pcss', 'pcfsoft', 'pcf', 'basic'.

Purpose: the highest-quality shadow filter the device actually supports. Subsystems call this instead of hard-coding a filter choice.

safestHDRColorFormat()

Returns: one of 'rgba32f', 'rgba16f', 'rgba8'.

Purpose: the highest-precision color format the device actually supports.

safestMSAASamples(limit)

Parameters: limit — the maximum sample count the caller would like.

Returns: 8, 4, 2, or 0.

Purpose: the highest MSAA sample count the device actually supports, capped at the caller's requested limit.

getCapabilitiesReport()

Returns: an object with available, webgl2, glslVersion (as a string), maxTextureSize, maxDrawBuffers, maxSamples, maxAnisotropy, precision, features, extensions, quirks, perfTier.

Purpose: the human-readable summary for the debug HUD or CI log.

---

Internal Functions (Not Exported but Documented)

_probe()

No parameters, no return value. The one-shot probe.

Flow:

1. Creates a 1×1 canvas.
2. Tries webgl2, then webgl, with antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'high-performance'.
3. If neither succeeds, sets available = false and returns.
4. Records the WebGL version and shading language version.
5. Queries every MAX_* parameter.
6. Probes precision via getShaderPrecisionFormat for the four precision levels on both stages. Maps the returned { precision, rangeMin, rangeMax } to the PRECISION_SUPPORT enum.
7. Calls getExtension for every extension in the list, guarded by try/catch.
8. If EXT_texture_filter_anisotropic is available, reads MAX_TEXTURE_MAX_ANISOTROPY_EXT.
9. Probes every compressed texture format by calling getExtension.
10. Releases the context via WEBGL_lose_context if available.
11. In the finally block, nulls the canvas and context references so they can be garbage collected.

_computeFeatureFlags()

No parameters. Returns the FEATURES object.

Flow:

1. Reads RAW_CAPS_DEEP, PLATFORM_CONFIG, and ANDROID_PROFILE.
2. Computes the raw flags and the derived flags by applying the rules above.
3. Freezes the result.

---

Default Export

The default export bundles: CAPABILITIES, RAW_CAPS_DEEP, FEATURES, SHADING_LANGUAGE, PRECISION_SUPPORT, COMPRESSED_FORMAT, hasFeature, hasQuirk, hasExtension, supportsCompressedFormat, getMaxTextureSize, getMaxDrawBuffers, getMaxSamples, getMaxAnisotropy, getMaxTextureUnits, getMaxVertexAttribs, recommendedPrecisionKeyword, capabilityPrecisionPrelude, capabilityShaderDefines, capabilityDefinesPrelude, capabilityShaderPrelude, safestShadowFilter, safestHDRColorFormat, safestMSAASamples, getCapabilitiesReport.

---

Usage Pattern

A shader compilation pipeline:

```
import {
  capabilityShaderPrelude,
  hasFeature,
} from './src/core/021_rnd_Capabilities.js';

const prelude = capabilityShaderPrelude();
const defines = capabilityShaderDefines();

// Conditionally include a chunk:
let source = prelude + '\n';
if (hasFeature('canUsePCSS')) {
  source += '#include <shadow_pcss>\n';
} else if (hasFeature('canUsePCFSoftShadow')) {
  source += '#include <shadow_pcf_soft>\n';
} else if (hasFeature('canUsePCFShadow')) {
  source += '#include <shadow_pcf>\n';
} else {
  source += '#include <shadow_basic>\n';
}

source += myMaterialShaderBody;
```

A subsystem that wants to configure itself:

```
if (hasFeature('canUseHDRTargets')) {
  giSystem.enableHDRTargets();
}

if (hasFeature('canUseTemporalAO')) {
  aoSystem.enableTemporalAccumulation();
} else {
  aoSystem.disableTemporalAccumulation();
}

const msaaSamples = safestMSAASamples(4);
renderer.setMSAA(msaaSamples);

const filter = safestShadowFilter();
shadowSystem.setFilter(filter);
```

A debug HUD that dumps the capability report:

```
console.log(JSON.stringify(getCapabilitiesReport(), null, 2));
```

Because every subsystem reads the same FEATURES object, the engine makes a single coherent decision about every capability. There is no risk of one subsystem using HDR targets while another falls back to RGBA8, because they all read canUseHDRTargets from the same place.
