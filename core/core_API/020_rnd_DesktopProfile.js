API Documentation — src/core/020_rnd_DesktopProfile.js

File Purpose

This file is the desktop fallback profile resolver for the anime lighting stack. The engine targets Android mobile as the primary platform, but every lighting subsystem needs a valid, deterministic profile when running on desktop during development, on CI machines for regression capture, or on a developer's laptop for shader iteration.

This module provides that profile with the same interface as 019_rnd_AndroidProfile.js, so downstream code reads DESKTOP_PROFILE.shadow, .ao, .giProbeSpacing, .precision, .workerPoolSize, and every other field without branching on platform.

The design contract is: any subsystem that already works with ANDROID_PROFILE should also work with DESKTOP_PROFILE with zero code changes. The shape is identical. Only the numeric values differ — desktop gets more samples, larger shadow maps, PCSS instead of PCF, and no RAM-based downgrades.

The module classifies the desktop GPU into twenty sub-families:

· Six NVIDIA (GeForce, Quadro, plus future variants)
· Two AMD (Radeon, FirePro)
· Three Intel (HD, Iris, Arc)
· Five Apple Silicon (M1–M4 and Intel iGPU fallback)
· Two software rasterizers (SwiftShader, LLVMpipe)
· Five ANGLE backends (D3D11, D3D9, Metal, OpenGL, Vulkan)

The ANGLE backend detection is important because Chrome and Edge on Windows report the ANGLE wrapper instead of the real GPU. When that happens, the real vendor is hidden inside the ANGLE renderer string, and this module exposes the backend as a separate flag so shader code can opt out of GPU-vendor-specific branches when the actual driver is not visible.

The module also owns desktop-grade thermal and battery curves. Desktop devices rarely throttle the way phones do, so the curves are much gentler. But the curves exist so the same quality controller code runs unchanged on both platforms.

---

Exported Constants

PERF_TIER

Internal. The cached PERF_TIER string from getPerfTier(). Used for worker pool sizing.

DESKTOP_GPU_FAMILY

Type: frozen enum

Values:

· UNKNOWN = 0
· NVIDIA_GEFORCE = 1
· NVIDIA_QUADRO = 2
· AMD_RADEON = 3
· AMD_FIREPRO = 4
· INTEL_HD = 5
· INTEL_IRIS = 6
· INTEL_ARC = 7
· APPLE_M1 = 8
· APPLE_M2 = 9
· APPLE_M3 = 10
· APPLE_M4 = 11
· APPLE_INTEL_IGPU = 12
· SWIFTSHADER = 13
· LLVMPIPE = 14
· ANGLE_D3D11 = 15
· ANGLE_D3D9 = 16
· ANGLE_METAL = 17
· ANGLE_GL = 18
· ANGLE_VULKAN = 19
· COUNT = 20

DESKTOP_GPU_FAMILY_NAME

Type: frozen array

Values: ['unknown', 'nvidia_geforce', 'nvidia_quadro', 'amd_radeon', 'amd_firepro', 'intel_hd', 'intel_iris', 'intel_arc', 'apple_m1', 'apple_m2', 'apple_m3', 'apple_m4', 'apple_intel_igpu', 'swiftshader', 'llvmpipe', 'angle_d3d11', 'angle_d3d9', 'angle_metal', 'angle_gl', 'angle_vulkan'].

DESKTOP_PRECISION

Type: frozen object

The default desktop precision block. All fields are 'highp':

· vertex — 'highp'
· fragment — 'highp'
· positionVarying — 'highp'
· varyingFloat — 'highp'

Desktop GPUs all support highp in both shader stages, so there is no precision negotiation on this platform.

DESKTOP_SHADOW_TUNING

Type: frozen object

The default desktop shadow tuning block. Fields:

· bias — -0.0004
· normalBias — 0.010
· pancakeFix — false
· softness — 0.10
· filter — 'pcss'

This is the base value used when the GPU is unknown. Specific GPU families override it with tighter biases.

DESKTOP_AO_TUNING

Type: frozen object

The default desktop AO tuning block. Fields:

· samples — 32
· radius — 2.5
· temporal — true
· halfRes — false

Desktop gets maximum AO quality by default.

DESKTOP_THERMAL_CURVE

Type: frozen array of 5 objects

Gentler than the Android thermal curve. One entry per THERMAL_LEVEL:

Level 0 (nominal): all multipliers 1.0.
Level 1 (warm): all multipliers 1.0, no change.
Level 2 (hot): shadow 0.90, GI 0.90, AO 1.0, postDelta 0, DPR 1.0, cluster 0.95.
Level 3 (very_hot): shadow 0.75, GI 0.75, AO 0.90, postDelta -1, DPR 0.95, cluster 0.85.
Level 4 (critical): shadow 0.50, GI 0.50, AO 0.75, postDelta -2, DPR 0.90, cluster 0.70.

Even at CRITICAL, desktop keeps half the shadows and GI. Compare to Android where CRITICAL drops to 10 %.

DESKTOP_BATTERY_CURVE

Type: frozen array of 5 objects

Much gentler than the Android battery curve. Most laptops are plugged in or have large batteries.

Level 0 (full or charging): all multipliers 1.0, postDelta 0.
Level 1 (good): DPR 0.95, others unchanged.
Level 2 (low): DPR 0.90, shadow 0.95, GI 0.95.
Level 3 (very_low): DPR 0.85, shadow 0.85, GI 0.85, AO 0.90, postDelta -1.
Level 4 (critical): DPR 0.75, shadow 0.70, GI 0.70, AO 0.80, postDelta -2.

DESKTOP_GPU

Type: number

The resolved DESKTOP_GPU_FAMILY value for this device. Computed by _resolveDesktopGpuFamily().

DESKTOP_GPU_NAME

Type: string

The string name of the resolved GPU family.

SHADOW_TUNING

Type: frozen object

The shadow tuning for the resolved GPU family. Computed by _resolveDesktopShadowTuning().

The per-family values:

NVIDIA GeForce / Quadro: bias -0.0003, normalBias 0.008, pancake fix false, softness 0.10, PCSS filter.
AMD Radeon / FirePro: bias -0.0005, normalBias 0.012, pancake fix false, softness 0.10, PCSS.
Intel Arc: bias -0.0006, normalBias 0.014, pancake fix false, softness 0.09, PCF.
Intel Iris / HD: bias -0.0010, normalBias 0.024, pancake fix false, softness 0.07, PCF.
Apple M1 / M2 / M3 / M4: bias -0.0004, normalBias 0.010, pancake fix false, softness 0.10, PCFSoft.
Apple Intel iGPU: bias -0.0012, normalBias 0.026, pancake fix false, softness 0.06, PCF.
SwiftShader / LLVMpipe: bias -0.0035, normalBias 0.070, pancake fix true, softness 0.00, basic.
Unknown: the default DESKTOP_SHADOW_TUNING.

AO_TUNING

Type: frozen object

The AO tuning for the resolved GPU family. Computed by _resolveDesktopAOTuning().

The per-family values:

NVIDIA GeForce / Quadro: 32 samples, radius 2.5, temporal, full res.
AMD Radeon / FirePro: 32 samples, radius 2.5, temporal, full res.
Intel Arc: 32 samples, radius 2.5, temporal, full res.
Intel Iris: 16 samples, radius 2.2, temporal, full res.
Intel HD: 8 samples, radius 2.0, temporal, half res.
Apple M1–M4: 24 samples, radius 2.4, temporal, full res.
Apple Intel iGPU: 8 samples, radius 2.0, temporal, half res.
SwiftShader / LLVMpipe: 4 samples, radius 1.5, no temporal, half res.
Unknown: the default DESKTOP_AO_TUNING.

GI_PROBE_SPACING

Type: number

Value: 1.5

Desktop GI probe spacing in meters. Much denser than Android because desktop CPUs have plenty of budget for the probe solve.

GI_UPDATE_HZ

Type: number

Value: 30

Desktop GI update rate in Hertz. Runs at half the frame rate but is still 1.5–7× the Android rate.

ANGLE_BACKEND

Type: string | null

The ANGLE backend name ('d3d11', 'd3d9', 'metal', 'gl', 'vulkan'), or null if the GPU is not ANGLE-wrapped.

IS_ANGLE

Type: boolean

True if the GPU is ANGLE-wrapped.

DESKTOP_PROFILE

Type: frozen object

The final resolved desktop profile. Downstream code reads this exclusively.

Fields:

· name — a composite name like 'desktop_profile_nvidia_geforce'.
· gpuFamily — the DESKTOP_GPU_FAMILY value.
· gpuFamilyName — the family name string.
· rendererName — the raw renderer string.
· vendorName — the raw vendor string.
· perfTier — the PERF_TIER string.
· profileTier — PLATFORM_CONFIG.tier.
· webglVersion — 1 or 2.
· precision — always DESKTOP_PRECISION.
· precisionTier — always PRECISION_TIER.HIGH.
· shadow — the SHADOW_TUNING object.
· shadowFilter — the resolved filter string.
· ao — the AO_TUNING object.
· giProbeSpacing — always GI_PROBE_SPACING.
· giUpdateHz — always GI_UPDATE_HZ.
· hdrColorFormat — 'rgba16f' if PLATFORM_CONFIG.supportsHalfFloatColor, otherwise 'rgba8'.
· workerPoolSize — the recommended worker count.
· thermalCurve — the DESKTOP_THERMAL_CURVE array.
· batteryCurve — the DESKTOP_BATTERY_CURVE array.
· perDomainHzScale — a frozen object with all fields set to 1.0.
· dprCap — always 2.0.
· maxClusterLights — always 256.
· giProbeLatticeCap — always 64.
· aoSampleCap — always 32.
· shadowMapCap — always 4096.
· shadowCascadeCap — always 4.
· postPassCap — always 8.
· maxGpuMemoryMB — 1024 on 16 GB+ devices, 512 on 8 GB+, 256 otherwise.
· quirks — the quirks array from 018_rnd_PlatformConfig.js.
· hasQuirk(name) — a method.

---

Exported Functions

desktopWorkerPoolSize(hardwareConcurrency)

Parameters: hardwareConcurrency — navigator.hardwareConcurrency.

Returns: the recommended worker count.

Purpose: max(1, min(8, cores - 1)). Desktop reserves one core for the main thread and caps at eight workers because more than that yields diminishing returns on desktop.

resolveDesktopProfileIfApplicable()

Returns: DESKTOP_PROFILE if the platform is not Android and not iOS, otherwise null.

Purpose: the platform selector. Callers that already know they are on desktop can use DESKTOP_PROFILE directly. Callers that need to branch use this function.

The function cannot import 019_rnd_AndroidProfile.js without a circular dependency, so it returns null for mobile platforms and expects the caller to have already resolved the Android profile separately.

desktopShaderPrecisionPrelude()

Returns: a GLSL string with the desktop precision prelude:

```
#ifndef DESKTOP_PRECISION_PRELUDE
#define DESKTOP_PRECISION_PRELUDE
precision highp float;
precision highp int;
#endif
```

Purpose: ready to prepend to any desktop shader.

getDesktopProfileReport()

Returns: an object with gpuFamily, gpuFamilyId, rendererName, vendorName, webglVersion, angleBackend, isAngle, hardwareConcurrency, perfTier, profileTier, precision, shadow, ao, giProbeSpacing, giUpdateHz, hdrColorFormat, workerPoolSize, maxGpuMemoryMB, dprCap, shadowMapCap, quirks, and platformReport (the report from 018_rnd_PlatformConfig.js).

Purpose: the human-readable summary for CI logs.

---

Internal Functions (Not Exported but Documented)

_resolveDesktopGpuFamily()

Returns: the DESKTOP_GPU_FAMILY value.

Flow:

1. Checks for SwiftShader and LLVMpipe by renderer string pattern.
2. Checks for ANGLE by vendor or renderer pattern. If ANGLE, examines the renderer string for the specific backend: d3d11, d3d9, metal, vulkan, or gl.
3. Checks for Apple. Distinguishes M1, M2, M3, M4 by pattern. Falls back to APPLE_INTEL_IGPU if the Apple string does not match a known M-series generation.
4. Checks for NVIDIA. Distinguishes GeForce from Quadro by pattern.
5. Checks for AMD. Distinguishes Radeon from FirePro.
6. Checks for Intel. Distinguishes Arc, Iris, and HD Graphics.
7. Returns UNKNOWN if nothing matches.

_resolveDesktopShadowTuning()

Returns: the per-family SHADOW_TUNING block.

_resolveDesktopAOTuning()

Returns: the per-family AO_TUNING block.

_detectAngleBackend()

Returns: the ANGLE backend name string or null.

---

Default Export

The default export bundles: DESKTOP_PROFILE, DESKTOP_GPU, DESKTOP_GPU_NAME, DESKTOP_GPU_FAMILY, DESKTOP_GPU_FAMILY_NAME, DESKTOP_PRECISION, DESKTOP_SHADOW_TUNING, DESKTOP_AO_TUNING, DESKTOP_THERMAL_CURVE, DESKTOP_BATTERY_CURVE, SHADOW_TUNING, AO_TUNING, GI_PROBE_SPACING, GI_UPDATE_HZ, ANGLE_BACKEND, IS_ANGLE, desktopWorkerPoolSize, resolveDesktopProfileIfApplicable, desktopShaderPrecisionPrelude, getDesktopProfileReport.

---

Usage Pattern

A cross-platform subsystem reads the correct profile via the platform selector:

```
import { resolveDesktopProfileIfApplicable } from './src/core/020_rnd_DesktopProfile.js';
import { ANDROID_PROFILE } from './src/core/019_rnd_AndroidProfile.js';
import { DEVICE } from './src/core/018_rnd_PlatformConfig.js';

function getActiveProfile() {
  if (DEVICE.isAndroid) return ANDROID_PROFILE;
  const desktop = resolveDesktopProfileIfApplicable();
  if (desktop) return desktop;
  return null; // iOS handled by 018 directly
}

const profile = getActiveProfile();

// Configure the shadow system with the profile's tuning.
const shadowTuning = profile.shadow;
shadowSystem.setBias(shadowTuning.bias);
shadowSystem.setNormalBias(shadowTuning.normalBias);
shadowSystem.setPancakeFix(shadowTuning.pancakeFix);
shadowSystem.setFilter(profile.shadowFilter);
```

A shader compilation tool that wants a desktop-appropriate prelude:

```
if (IS_ANGLE) {
  // The real GPU is hidden behind ANGLE; use backend-agnostic shader branches.
  define('ANGLE_BACKEND_' + ANGLE_BACKEND.toUpperCase());
}
const prelude = desktopShaderPrecisionPrelude();
```

A CI log that dumps the profile once at startup:

```
console.log(JSON.stringify(getDesktopProfileReport(), null, 2));
```

Because the profile has the same shape as ANDROID_PROFILE, the same subsystem code runs on both platforms without branching. A shader compiled on the desktop with pcss shadows produces the reference output; the same shader compiled on Android with pcf shadows produces the same anime look at lower cost. The two profiles are the two ends of the same quality spectrum.
