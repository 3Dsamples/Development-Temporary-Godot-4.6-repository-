API Documentation — src/core/018_rnd_PlatformConfig.js

File Purpose

This file is the platform configuration and Android device profile resolver for the anime lighting stack. Where 016_rnd_Config.js produces the tier-level budget tables and 017_rnd_QualityConfig.js produces the runtime quality decisions, this module classifies the concrete platform and provides the frozen device profile that every lighting subsystem reads to avoid the classic Android rendering bugs.

It answers four questions:

1. What is this device, exactly? — Android vs iOS vs desktop, WebGL1 vs WebGL2, GPU vendor, GPU generation, memory class, core count.
2. What WebGL features can I rely on? — half-float color renderability, float color renderability, depth textures, instancing, multiple render targets, vertex textures, derivatives, precision support.
3. What quirks does this driver have? — Mali precision loss, Adreno half-float clamp, Adreno shadow bias needs, PowerVR tile-based deferred depth cost, PowerVR shader compile latency, Apple's high-precision GPU path, WebGL1 fallback, low-memory device, high-DPI device, slow uniform updates, unreliable MSAA.
4. Which concrete profile applies? — one of seven profiles: Android HIGH / MID / LOW / MINIMAL, iOS HIGH / MID, or Desktop.

The output is a single frozen PLATFORM_CONFIG object that downstream code reads without re-detecting. Every subsystem that needs to know "can I use HDR render targets?" or "can I use PCSS shadows?" or "should I use highp in the fragment shader?" reads the answer from here.

The classification uses the following inputs:

· navigator.userAgent for platform detection
· navigator.deviceMemory for memory class
· navigator.hardwareConcurrency for CPU class
· window.devicePixelRatio for DPI class
· WEBGL_debug_renderer_info for GPU vendor and renderer string
· Extension probes for capability detection
· Shader precision probes for highp support

---

Exported Constants

RAW_UA

Type: frozen object

The result of _detectUA(). Fields:

· ua — the full navigator.userAgent string.
· vendor — navigator.vendor.
· platform — navigator.platform.

Returns a stub with empty strings if navigator is undefined.

RAW_GPU

Type: frozen object

The result of _detectGPU(). Fields:

· vendor — the unmasked GPU vendor string, or 'unknown'.
· renderer — the unmasked GPU renderer string, or 'unknown'.
· webglVersion — 2 for WebGL2, 1 for WebGL1, 0 if no context.

The probe creates a temporary canvas, tries webgl2 then webgl, reads the debug renderer extension if available, then releases the context via WEBGL_lose_context.

RAW_CAPS

Type: frozen object

The result of _detectCapabilities(). Fields:

· maxTextureUnits — MAX_TEXTURE_IMAGE_UNITS.
· maxVertexAttributes — MAX_VERTEX_ATTRIBS.
· maxVertexUniformVectors — MAX_VERTEX_UNIFORM_VECTORS.
· maxFragmentUniformVectors — MAX_FRAGMENT_UNIFORM_VECTORS.
· maxTextureSize — MAX_TEXTURE_SIZE.
· maxRenderbufferSize — MAX_RENDERBUFFER_SIZE.
· maxSamples — MAX_SAMPLES for WebGL2, or 0.
· supportsHalfFloatColor — boolean.
· supportsFloatColor — boolean.
· supportsDepthTexture — boolean.
· supportsInstancing — boolean.
· supportsVertexTextures — boolean.
· supportsMRT — boolean.
· supportsDerivatives — boolean.
· highpSupported — boolean. Result of getShaderPrecisionFormat(FRAGMENT_SHADER, HIGH_FLOAT).

DEVICE

Type: frozen object

The result of _classifySoc(). Fields:

· gpuVendor — one of 'adreno', 'mali', 'powervr', 'apple', 'intel', 'nvidia', 'amd', 'software', 'unknown'.
· adrenoGeneration — integer 3–8, or 0.
· maliGeneration — integer 1–5, or 0.
· appleGeneration — integer 1–4, or 0.
· deviceMemory — navigator.deviceMemory or 4.
· hardwareConcurrency — navigator.hardwareConcurrency or 4.
· isAndroid — boolean.
· isIOS — boolean.
· isMobile — boolean.
· isDesktop — boolean.
· webglVersion — 1 or 2.

QUIRK

Type: frozen enum

Named device quirks. Values:

· MALI_PRECISION_LOSS — 'mali_precision_loss'
· ADRENO_HALF_FLOAT_CLAMP — 'adreno_half_float_clamp'
· ADRENO_SHADOW_BIAS — 'adreno_shadow_bias'
· POWERVR_TBDR_DEPTH_COST — 'powervr_tbdr_depth_cost'
· POWERVR_SHADER_COMPILE — 'powervr_shader_compile'
· APPLE_HALF_FLOAT_OK — 'apple_half_float_ok'
· WEBGL1_ONLY — 'webgl1_only'
· LOW_MEMORY_DEVICE — 'low_memory_device'
· HIGH_DPI_DEVICE — 'high_dpi_device'
· SLOW_UNIFORM_UPDATES — 'slow_uniform_updates'
· MSAA_UNRELIABLE — 'msaa_unreliable'

QUIRKS

Type: frozen array of quirk strings

The list of quirks that apply to this device. Computed by _resolveQuirks().

PRECISION_TIER

Type: frozen object

Values:

· HIGH — 'highp'
· MEDIUM — 'mediump'
· LOW — 'lowp'

RESOLVED_PRECISION

Type: string

The recommended shader precision for this device. Computed by _resolvePrecisionTier(). Returns 'mediump' if highp is unsupported, if the device has the Mali precision loss quirk, or if the device is a low-memory WebGL1 device.

Profile Constants

Seven frozen profile objects. Each has the same shape:

· name — the profile's symbolic name.
· tier — 'HIGH' | 'MEDIUM' | 'LOW'.
· dprCap — the maximum device pixel ratio.
· msaaLimit — the maximum MSAA sample count.
· precision — the recommended shader precision.
· preferredShadowType — one of 'basic' | 'pcf' | 'pcfsoft' | 'pcss'.
· preferredColorFormat — 'rgba8' | 'rgba16f'.
· supportsHalfFloatColor, supportsFloatColor, supportsDepthTexture, supportsInstancing, supportsMRT, supportsVertexTextures, supportsDerivatives — booleans.
· maxClusterLights — number.
· clusterGridOverride — a frozen array of three numbers, or null.
· giProbeLatticeCap — number.
· aoSampleCap — number.
· shadowMapCap — number.
· shadowCascadeCap — number.
· postPassCap — number.
· perDomainHzScale — frozen object with shadows, gi, ao, post, environment, interior, exterior multipliers.
· quirks — a frozen array of quirk strings.
· hasQuirk(name) — a method that returns true if the named quirk applies.

PROFILE_ANDROID_HIGH — Snapdragon 8xx, Dimensity 9000+. DPR cap 2.0, MSAA limit 4, highp precision, pcfsoft shadows, RGBA16F, 128 cluster lights, 2048 shadow map, 4 cascades, 6 post passes.

PROFILE_ANDROID_MID — Snapdragon 7xx, Dimensity 800. DPR cap 1.75, no MSAA, highp precision, pcf shadows, RGBA16F, 64 cluster lights, 1024 shadow map, 2 cascades, 4 post passes. Slower shadow/GI/AO/post domains at 0.85×.

PROFILE_ANDROID_LOW — Snapdragon 6xx, Helio G series. DPR cap 1.5, mediump precision, pcf shadows, RGBA8, 32 cluster lights, 512 shadow map, 1 cascade, 2 post passes. Sub-domains scaled to 0.65×. Has LOW_MEMORY_DEVICE and SLOW_UNIFORM_UPDATES quirks.

PROFILE_ANDROID_MINIMAL — Snapdragon 4xx and below, plus all WebGL1 devices. DPR cap 1.25, mediump, basic shadows, RGBA8, 16 cluster lights (16×9×16 grid override), 256 shadow map, 1 cascade, 1 post pass. Almost everything scaled down. Has LOW_MEMORY_DEVICE, WEBGL1_ONLY, MSAA_UNRELIABLE, SLOW_UNIFORM_UPDATES quirks.

PROFILE_IOS_HIGH — A14+ and M-series. DPR cap 2.0, MSAA limit 4, highp, pcfsoft, RGBA16F, 128 cluster lights, 2048 shadow map, 4 cascades, 6 post passes. Has APPLE_HALF_FLOAT_OK quirk.

PROFILE_IOS_MID — A11–A13. DPR cap 1.75, no MSAA, highp, pcf, RGBA16F, 64 cluster lights, 1024 shadow map, 2 cascades, 4 post passes. Has APPLE_HALF_FLOAT_OK.

PROFILE_DESKTOP — dev work, CI, regression capture. DPR cap 2.0, MSAA limit 8, highp, pcss, RGBA16F, 256 cluster lights, 4096 shadow map, 4 cascades, 8 post passes.

PROFILE

Type: frozen profile object

The resolved profile for the current device. Computed by _resolveProfile().

Resolution logic:

1. If webglVersion === 1, returns PROFILE_ANDROID_MINIMAL on Android or PROFILE_DESKTOP on other platforms.
2. If desktop, returns PROFILE_DESKTOP.
3. If iOS, returns PROFILE_IOS_HIGH for apple generation 3+ and PROFILE_IOS_MID otherwise.
4. If Android, picks by memory + cores + GPU generation:
   · 8 GB + 8 cores + gen 6+ → HIGH
   · 6 GB + 6 cores + gen 5+ → MID
   · 4 GB + 4 cores → MID
   · 3 GB + 4 cores → LOW
   · else → MINIMAL
5. Unknown mobile → PROFILE_ANDROID_LOW.

PLATFORM_CONFIG

Type: frozen object

The final resolved platform config. Downstream code reads this exclusively.

Fields:

· profile — the resolved PROFILE.
· name — the profile name.
· tier — the profile tier.
· perfTier — the cached PERF_TIER from 008_scn_world.js.
· dprCap — from the profile.
· msaaLimit — from the profile.
· precision — from the profile.
· preferredShadowType — from the profile.
· preferredColorFormat — from the profile.
· supportsHalfFloatColor, supportsFloatColor, supportsDepthTexture, supportsInstancing, supportsMRT, supportsVertexTextures, supportsDerivatives — booleans.
· maxClusterLights, clusterGridOverride, giProbeLatticeCap, aoSampleCap, shadowMapCap, shadowCascadeCap, postPassCap — numbers or null.
· perDomainHzScale — the profile's frozen object.
· device — a frozen object with gpuVendor, adrenoGeneration, maliGeneration, appleGeneration, deviceMemory, hardwareConcurrency, isAndroid, isIOS, isMobile, isDesktop, webglVersion, rendererName, vendorName.
· caps — a copy of RAW_CAPS.
· quirks — the quirks array.
· hasQuirk(name) — a method.

---

Exported Functions

recommendedPixelRatio()

Returns: min(window.devicePixelRatio, PLATFORM_CONFIG.dprCap).

Purpose: the only function the EngineLoop should call when deciding DPR. It reads the profile cap, not navigator.devicePixelRatio directly.

canUseShadowType(type)

Parameters: type — one of 'basic' | 'pcf' | 'pcfsoft' | 'pcss' | 'vsm' | 'esm'.

Returns: boolean.

Purpose: reports whether the requested shadow filter is safe on this device. The pcss filter is only available on desktops. The pcfsoft filter requires pcfsoft or pcss preferred. The esm filter is never available on mobile. The vsm filter is unavailable when the device has the MSAA_UNRELIABLE quirk.

canUseColorFormat(format)

Parameters: format — one of 'rgba8' | 'rgb8' | 'rgba16f' | 'rgba32f' | 'depth'.

Returns: boolean.

Purpose: reports whether the requested render target color format is safe. rgba8 and rgb8 are always safe. rgba16f requires supportsHalfFloatColor. rgba32f requires supportsFloatColor. depth requires supportsDepthTexture.

suggestedPrecision()

Returns: PLATFORM_CONFIG.precision — 'highp' or 'mediump'.

Purpose: the precision keyword the shader prelude should use.

scaleDomainHz(domain, baseHz)

Parameters:

· domain — one of 'shadows' | 'gi' | 'ao' | 'post' | 'environment' | 'interior' | 'exterior'.
· baseHz — the base frequency to scale.

Returns: the scaled frequency, floored at 2 Hz.

Purpose: the FrameScheduler calls this when initializing per-domain frequencies, so a slower GPU automatically gets lower update rates.

resolvedClusterGrid(defaultGrid)

Parameters: defaultGrid — a three-element array of the default cluster grid resolution.

Returns: the override grid from the profile if any, otherwise the default.

Purpose: LOW-tier profiles may override the cluster grid resolution to a smaller value. This helper applies the override.

getPlatformReport()

Returns: an object with profile, tier, perfTier, gpuVendor, rendererName, vendorName, webglVersion, deviceMemory, hardwareConcurrency, dpr, dprCap, precision, preferredShadowType, preferredColorFormat, the seven capability booleans, maxClusterLights, giProbeLatticeCap, aoSampleCap, shadowMapCap, shadowCascadeCap, postPassCap, quirks.

Purpose: the human-readable summary for the debug HUD, CI log, or regression capture. Do not call per frame.

---

Internal Functions (Not Exported but Documented)

_detectUA()

Returns: { ua, vendor, platform }. Reads navigator.userAgent, navigator.vendor, navigator.platform.

_detectGPU()

Returns: { vendor, renderer, webglVersion }. Creates a probe canvas, tries webgl2 then webgl, reads WEBGL_debug_renderer_info if available, then releases the context.

_detectCapabilities()

Returns: the capability object. Similar probe flow, but reads MAX_* parameters and extension presence.

_classifySoc()

Returns: the DEVICE object. Parses the renderer string to infer the GPU vendor and generation. Reads navigator.deviceMemory and navigator.hardwareConcurrency.

_resolveQuirks()

Returns: the quirks array. Reads DEVICE.gpuVendor, DEVICE.webglVersion, DEVICE.deviceMemory, and window.devicePixelRatio.

_resolvePrecisionTier()

Returns: the precision string. Checks highp support, the Mali precision loss quirk, and the low-memory WebGL1 condition.

_resolveProfile()

Returns: the resolved profile. The resolution logic is described above.

_freezeProfile(p)

Internal. Freezes the profile and its quirks array, and attaches the hasQuirk(name) method.

---

Default Export

The default export bundles: PLATFORM_CONFIG, PROFILE, DEVICE, QUIRKS, RAW_UA, RAW_GPU, RAW_CAPS, RESOLVED_PRECISION, PRECISION_TIER, QUIRK, the seven profile constants, recommendedPixelRatio, canUseShadowType, canUseColorFormat, suggestedPrecision, scaleDomainHz, resolvedClusterGrid, getPlatformReport.

---

Usage Pattern

The EngineLoop reads the platform config at initialization:

```
import {
  PLATFORM_CONFIG,
  recommendedPixelRatio,
  canUseShadowType,
  canUseColorFormat,
} from './src/core/018_rnd_PlatformConfig.js';

renderer.setPixelRatio(recommendedPixelRatio());

const shadowType = canUseShadowType('pcss') ? 'pcss'
                 : canUseShadowType('pcfsoft') ? 'pcfsoft'
                 : canUseShadowType('pcf') ? 'pcf'
                 : 'basic';
shadowSystem.setFilter(shadowType);

if (canUseColorFormat('rgba16f')) {
  giSystem.enableHDRTargets();
} else {
  giSystem.useRGBA8Targets();
}
```

A shader prelude uses the suggested precision:

```
const prelude = [
  `precision ${suggestedPrecision()} float;`,
  `precision ${suggestedPrecision()} int;`,
].join('\n');
```

A subsystem that has quirk-aware code:

```
if (PLATFORM_CONFIG.hasQuirk('mali_precision_loss')) {
  // Use mediump for the position varying, small epsilon for comparisons.
}
```

The profile is the single place where the ambient noise of Android device diversity gets collapsed into a small set of deterministic decisions. Every subsystem reads the same profile, so the entire engine either handles a quirk or does not — no subsystem accidentally uses HDR targets on a device that cannot render to them.

