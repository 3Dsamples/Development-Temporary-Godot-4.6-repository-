API Documentation — src/core/019_rnd_AndroidProfile.js

File Purpose

This file is the Android-specific device profile resolver and tuning table for the anime lighting stack. Where 018_rnd_PlatformConfig.js classifies the platform at a coarse level — Android vs iOS vs desktop, HIGH vs MEDIUM vs LOW, WebGL1 vs WebGL2 — this module owns the concrete Android tuning at the level of individual GPU families.

The device diversity within Android is enormous. A 4 GB Snapdragon 660 (Adreno 512) behaves completely differently from a 4 GB Snapdragon 778G (Adreno 642L). Shadow bias values that work on one will cause peter-panning on the other. Precision settings that compile on one will fail shader compilation on the other. AO sample counts that are fine on one will spike frame time on the other.

This module collapses that diversity into fifteen named GPU families — six Adreno generations, six Mali generations, two PowerVR families, and SwiftShader — and provides concrete tuning tables for each. Downstream subsystems read the resolved tuning from ANDROID_PROFILE and never re-detect.

The module provides:

1. GPU family classification — parses the unmasked renderer string to identify the exact GPU family.
2. Per-GPU shader precision overrides — highp vs mediump for vertex, fragment, position varying, and varying float. Mali-T and older Mali generations require mediump on the position varying or they produce geometry artifacts.
3. Per-GPU shadow tuning — bias, normal bias, pancake fix flag, softness, preferred filter. Adreno 3xx–5xx need a strongly negative bias and a pancake fix; Adreno 7xx+ need much less. Mali needs slightly more negative bias than Adreno at the same generation.
4. Per-GPU AO tuning — sample count, radius, temporal accumulation flag, half-res flag. Older GPUs get 3–4 samples; newer GPUs get 10–16 samples and temporal accumulation.
5. Per-RAM GI tuning — probe spacing and update rate. A 2 GB device uses 8-meter probe spacing at 4 Hz; an 8 GB device uses 2-meter spacing at 20 Hz.
6. Thermal and battery downgrade curves — five-level curves for each, with per-level multipliers for shadow, GI, AO, DPR, cluster lights, and post passes.
7. Worker pool sizing — a recommended number of workers based on navigator.hardwareConcurrency and the tier.
8. Combined bias resolution — a single function that takes the current thermal input and battery level and returns the combined multipliers that downstream systems apply on top of their quality snapshots.

The final resolved ANDROID_PROFILE object has the same shape as DESKTOP_PROFILE in 020_rnd_DesktopProfile.js, so downstream subsystems can switch profiles without touching their business logic.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string.

ANDROID_GPU_FAMILY

Type: frozen enum

Values:

· UNKNOWN = 0
· ADRENO_3XX = 1
· ADRENO_4XX = 2
· ADRENO_5XX = 3
· ADRENO_6XX = 4
· ADRENO_7XX = 5
· ADRENO_8XX = 6
· MALI_T = 7
· MALI_G3X = 8
· MALI_G5X = 9
· MALI_G6X = 10
· MALI_G7X = 11
· MALI_G8X = 12
· POWERVR_ROGUE = 13
· POWERVR_GE = 14
· SWIFTSHADER = 15
· COUNT = 16

ANDROID_GPU_FAMILY_NAME

Type: frozen array

Values: ['unknown', 'adreno_3xx', 'adreno_4xx', 'adreno_5xx', 'adreno_6xx', 'adreno_7xx', 'adreno_8xx', 'mali_t', 'mali_g3x', 'mali_g5x', 'mali_g6x', 'mali_g7x', 'mali_g8x', 'powervr_rogue', 'powervr_ge', 'swiftshader'].

THERMAL_LEVEL

Type: frozen enum

Values:

· NOMINAL = 0
· WARM = 1
· HOT = 2
· VERY_HOT = 3
· CRITICAL = 4
· COUNT = 5

THERMAL_LEVEL_NAME

Type: frozen array

Values: ['nominal', 'warm', 'hot', 'very_hot', 'critical'].

BATTERY_LEVEL

Type: frozen enum

Values:

· FULL = 0
· GOOD = 1
· LOW = 2
· VERY_LOW = 3
· CRITICAL = 4
· COUNT = 5

BATTERY_LEVEL_NAME

Type: frozen array

Values: ['full', 'good', 'low', 'very_low', 'critical'].

GPU_FAMILY

Type: number

The resolved ANDROID_GPU_FAMILY value for this device. Computed by _resolveGpuFamily().

GPU_FAMILY_NAME

Type: string

The string name of the resolved GPU family.

GPU_PRECISION

Type: frozen object

Maps every ANDROID_GPU_FAMILY value to its precision block. Each block has four fields:

· vertex — 'highp' | 'mediump'
· fragment — 'highp' | 'mediump'
· positionVarying — 'highp' | 'mediump'
· varyingFloat — 'highp' | 'mediump'

The values reflect real driver behavior:

Adreno 3xx: everything mediump.
Adreno 4xx: vertex highp, fragment mediump.
Adreno 5xx–8xx: everything highp.
Mali-T and Mali-G3x: mediump fragment, mediump position varying.
Mali-G5x: mediump fragment, mediump position varying, highp varying float.
Mali-G6x: highp fragment, mediump position varying, highp varying float.
Mali-G7x–G8x: everything highp.
PowerVR Rogue: highp vertex and position, mediump fragment.
PowerVR GE: everything mediump.
SwiftShader: everything mediump.
Unknown: highp vertex and position, mediump fragment.

PRECISION

Type: frozen object

The precision block for the resolved GPU family.

GPU_SHADOW_TUNING

Type: frozen object

Maps every ANDROID_GPU_FAMILY value to its shadow tuning block. Each block has:

· bias — the shadow bias. Negative values.
· normalBias — the normal offset bias.
· pancakeFix — boolean, whether to enable the pancake fix.
· softness — [0, 1] softness.
· filter — 'basic' | 'pcf' | 'pcfsoft'.

The values:

Adreno 3xx: bias -0.0030, normalBias 0.060, pancake fix true, basic filter.
Adreno 4xx: bias -0.0022, normalBias 0.050, pancake fix true, basic filter.
Adreno 5xx: bias -0.0016, normalBias 0.040, pancake fix true, PCF filter.
Adreno 6xx: bias -0.0010, normalBias 0.030, pancake fix true, PCF.
Adreno 7xx: bias -0.0008, normalBias 0.022, pancake fix false, PCF.
Adreno 8xx: bias -0.0006, normalBias 0.018, pancake fix false, PCF.
Mali-T: bias -0.0035, normalBias 0.070, pancake fix true, basic.
Mali-G3x: bias -0.0028, normalBias 0.055, pancake fix true, basic.
Mali-G5x: bias -0.0018, normalBias 0.045, pancake fix true, PCF.
Mali-G6x: bias -0.0012, normalBias 0.032, pancake fix false, PCF.
Mali-G7x: bias -0.0009, normalBias 0.024, pancake fix false, PCF.
Mali-G8x: bias -0.0007, normalBias 0.018, pancake fix false, PCF.
PowerVR Rogue: bias -0.0024, normalBias 0.048, pancake fix true, basic.
PowerVR GE: bias -0.0035, normalBias 0.070, pancake fix true, basic.
SwiftShader: bias -0.0035, normalBias 0.070, pancake fix true, basic.
Unknown: bias -0.0020, normalBias 0.045, pancake fix true, PCF.

SHADOW_TUNING

Type: frozen object

The shadow tuning for the resolved GPU family.

GPU_AO_TUNING

Type: frozen object

Maps every ANDROID_GPU_FAMILY value to its AO tuning block. Each block has:

· samples — the number of AO samples.
· radius — the AO radius in world units.
· temporal — boolean, whether to enable temporal accumulation.
· halfRes — boolean, whether to run at half resolution.

The values:

Adreno 3xx: 3 samples, radius 1.2, no temporal, half res.
Adreno 4xx: 4 samples, radius 1.5, no temporal, half res.
Adreno 5xx: 6 samples, radius 1.8, no temporal, half res.
Adreno 6xx: 8 samples, radius 2.0, temporal, half res.
Adreno 7xx: 12 samples, radius 2.2, temporal, full res.
Adreno 8xx: 16 samples, radius 2.4, temporal, full res.
Mali-T: 3 samples, radius 1.2, no temporal, half res.
Mali-G3x: 4 samples, radius 1.5, no temporal, half res.
Mali-G5x: 6 samples, radius 1.8, no temporal, half res.
Mali-G6x: 8 samples, radius 2.0, temporal, half res.
Mali-G7x: 10 samples, radius 2.1, temporal, full res.
Mali-G8x: 14 samples, radius 2.3, temporal, full res.
PowerVR Rogue: 4 samples, radius 1.5, no temporal, half res.
PowerVR GE: 3 samples, radius 1.2, no temporal, half res.
SwiftShader: 3 samples, radius 1.2, no temporal, half res.
Unknown: 6 samples, radius 1.8, no temporal, half res.

AO_TUNING

Type: frozen object

The AO tuning for the resolved GPU family.

THERMAL_CURVE

Type: frozen array of 5 objects

One entry per THERMAL_LEVEL. Each entry has:

· shadowScale — multiplier for shadow map size.
· giScale — multiplier for GI update rate.
· aoScale — multiplier for AO sample count.
· postDelta — additive delta for post-pass count.
· dprScale — multiplier for DPR.
· clusterScale — multiplier for cluster light count.

Level 0 (nominal): all multipliers 1.0, postDelta 0.
Level 1 (warm): shadow 0.75, GI 0.75, AO 1.0, postDelta -1, DPR 1.0, cluster 0.85.
Level 2 (hot): shadow 0.50, GI 0.50, AO 0.75, postDelta -2, DPR 0.85, cluster 0.65.
Level 3 (very_hot): shadow 0.25, GI 0.25, AO 0.50, postDelta -3, DPR 0.75, cluster 0.50.
Level 4 (critical): shadow 0.10, GI 0.10, AO 0.25, postDelta -4, DPR 0.60, cluster 0.35.

BATTERY_CURVE

Type: frozen array of 5 objects

One entry per BATTERY_LEVEL. Same shape as the thermal curve but different values.

Level 0 (full or charging): all multipliers 1.0, postDelta 0.
Level 1 (good): DPR 0.90, others unchanged.
Level 2 (low): DPR 0.80, shadow 0.75, GI 0.85, AO 0.90, postDelta -1.
Level 3 (very_low): DPR 0.70, shadow 0.50, GI 0.50, AO 0.75, postDelta -2.
Level 4 (critical): DPR 0.60, shadow 0.25, GI 0.25, AO 0.50, postDelta -3.

ANDROID_PROFILE

Type: frozen object

The final resolved profile.

Fields:

· name — a composite name like 'android_profile_adreno_6xx'.
· gpuFamily — the ANDROID_GPU_FAMILY value.
· gpuFamilyName — the family name string.
· rendererName — the raw renderer string.
· vendorName — the raw vendor string.
· perfTier — the PERF_TIER_LOCAL string.
· profileTier — PLATFORM_CONFIG.tier.
· webglVersion — 1 or 2.
· precision — the PRECISION object.
· precisionTier — always PRECISION_TIER.HIGH.
· shadow — the SHADOW_TUNING object.
· shadowFilter — the resolved filter string.
· ao — the AO_TUNING object.
· giProbeSpacing — computed from RAM class.
· giUpdateHz — computed from RAM class.
· hdrColorFormat — 'rgba16f' or 'rgba8'.
· workerPoolSize — the recommended worker count.
· thermalCurve — the THERMAL_CURVE array.
· batteryCurve — the BATTERY_CURVE array.
· perDomainHzScale — from PLATFORM_CONFIG.
· dprCap — from PLATFORM_CONFIG.
· maxClusterLights — from PLATFORM_CONFIG.
· giProbeLatticeCap — from PLATFORM_CONFIG.
· aoSampleCap — from PLATFORM_CONFIG.
· shadowMapCap — from PLATFORM_CONFIG.
· shadowCascadeCap — from PLATFORM_CONFIG.
· postPassCap — from PLATFORM_CONFIG.
· maxGpuMemoryMB — computed from RAM: 256 / 192 / 128 / 96 / 48 MB.
· quirks — the quirks array from 018.
· hasQuirk(name) — a method.

---

Exported Functions

giProbeSpacingForRam(deviceMemoryGB)

Parameters: deviceMemoryGB — the device's RAM in GB.

Returns: the probe spacing in meters.

Values: 8 GB+ → 2.0, 6 GB+ → 3.0, 4 GB+ → 4.0, 3 GB+ → 6.0, else 8.0.

giUpdateHzForRam(deviceMemoryGB)

Parameters: deviceMemoryGB — the device's RAM in GB.

Returns: the GI update rate in Hz.

Values: 8 GB+ → 20, 6 GB+ → 15, 4 GB+ → 10, 3 GB+ → 6, else 4.

workerPoolSizeForConcurrency(hardwareConcurrency, tier)

Parameters:

· hardwareConcurrency — navigator.hardwareConcurrency.
· tier — 'LOW' | 'MEDIUM' | 'HIGH'.

Returns: the recommended worker count.

Values: reserves 1 core on HIGH, 2 on other tiers. Caps at 6 on HIGH, 4 on MEDIUM, 2 on LOW.

resolveThermalLevel(input)

Parameters: input — either a number in [0, 1] or a string.

If a number: 0.90+ → CRITICAL, 0.70+ → VERY_HOT, 0.50+ → HOT, 0.25+ → WARM, else NOMINAL.

If a string: maps 'critical' | 'serious' | 'fair' | 'warm' | 'nominal' to levels.

Returns: one of THERMAL_LEVEL.

resolveBatteryLevel(level, charging)

Parameters:

· level — [0, 1].
· charging — boolean.

Returns: one of BATTERY_LEVEL.

If charging, returns FULL. Otherwise maps level bands to levels: 0.15- → CRITICAL, 0.20- → VERY_LOW, 0.30- → LOW, 0.50- → GOOD, else FULL.

combinedDowngradeMultipliers(thermalInput, batteryLevel, charging)

Parameters:

· thermalInput — passed to resolveThermalLevel.
· batteryLevel — [0, 1].
· charging — boolean.

Returns: a shared scratch object with:

· shadowScale — thermal.shadowScale * battery.shadowScale.
· giScale — thermal.giScale * battery.giScale.
· aoScale — thermal.aoScale * battery.aoScale.
· dprScale — thermal.dprScale * battery.dprScale.
· clusterScale — thermal.clusterScale.
· postDelta — thermal.postDelta + battery.postDelta.
· thermalLevel — the resolved thermal level.
· batteryLevel — the resolved battery level.

The result is written into a module-level scratch object reused across calls, so it is zero-allocation. Callers must not hold a reference across calls.

shaderPrecisionPrelude()

Returns: a GLSL string with the recommended precision keywords. Ready to prepend to any shader.

getAndroidProfileReport()

Returns: an object with gpuFamily, gpuFamilyId, rendererName, vendorName, webglVersion, deviceMemoryGB, hardwareConcurrency, perfTier, profileTier, precision, shadow, ao, giProbeSpacing, giUpdateHz, hdrColorFormat, workerPoolSize, maxGpuMemoryMB, dprCap, shadowMapCap, quirks, and platformReport (the report from 018_rnd_PlatformConfig.js).

---

Internal Functions (Not Exported but Documented)

_resolveGpuFamily()

Returns: the ANDROID_GPU_FAMILY value. Parses RAW_GPU.renderer and RAW_GPU.vendor from 018_rnd_PlatformConfig.js.

Adreno parsing: extracts the generation number from adreno[\s\(]*(\d). Maps to ADRENO_3XX through ADRENO_8XX.

Mali parsing: detects immortalis first, then matches mali-([tg]?)(\d{2,3}) and maps the series and number to the correct generation. The mapping handles Mali-T (older), Mali-G3x (G30–G39), Mali-G5x (G50–G59), Mali-G6x, Mali-G7x, Mali-G8x.

PowerVR parsing: distinguishes Rogue from GE variants.

_resolveDesktopGpuFamily() and _resolveDesktopShadowTuning() and _resolveDesktopAOTuning()

Not present in this file — these are in 020_rnd_DesktopProfile.js.

---

Default Export

The default export bundles: ANDROID_PROFILE, GPU_FAMILY, GPU_FAMILY_NAME, ANDROID_GPU_FAMILY, ANDROID_GPU_FAMILY_NAME, GPU_PRECISION, PRECISION, GPU_SHADOW_TUNING, SHADOW_TUNING, GPU_AO_TUNING, AO_TUNING, THERMAL_CURVE, BATTERY_CURVE, THERMAL_LEVEL, THERMAL_LEVEL_NAME, BATTERY_LEVEL, BATTERY_LEVEL_NAME, giProbeSpacingForRam, giUpdateHzForRam, workerPoolSizeForConcurrency, resolveThermalLevel, resolveBatteryLevel, combinedDowngradeMultipliers, shaderPrecisionPrelude, getAndroidProfileReport.

---

Usage Pattern

The EngineLoop reads the Android profile at initialization:

```
import {
  ANDROID_PROFILE,
  combinedDowngradeMultipliers,
} from './src/core/019_rnd_AndroidProfile.js';

// Configure the shadow system with the profile's tuning.
const shadowTuning = ANDROID_PROFILE.shadow;
shadowSystem.setBias(shadowTuning.bias);
shadowSystem.setNormalBias(shadowTuning.normalBias);
shadowSystem.setPancakeFix(shadowTuning.pancakeFix);
shadowSystem.setFilter(ANDROID_PROFILE.shadowFilter);

// Configure GI from the profile.
giSystem.setProbeSpacing(ANDROID_PROFILE.giProbeSpacing);
giSystem.setUpdateHz(ANDROID_PROFILE.giUpdateHz);

// Configure AO from the profile.
aoSystem.setSampleCount(ANDROID_PROFILE.ao.samples);
aoSystem.setRadius(ANDROID_PROFILE.ao.radius);
aoSystem.setTemporal(ANDROID_PROFILE.ao.temporal);
aoSystem.setHalfRes(ANDROID_PROFILE.ao.halfRes);
```

The App's thermal and battery guards feed the combined downgrade:

```
app.on('thermal', ({ state }) => {
  const m = combinedDowngradeMultipliers(state, batteryLevel, charging);
  // Apply m.shadowScale, m.giScale, m.aoScale, m.dprScale, m.postDelta
  // to the runtime quality snapshot.
});
```

Because every subsystem reads the same profile, a device with an Adreno 505 GPU gets the exact same shadow bias as every other device with an Adreno 505, and the visual output is bit-reproducible across the fleet. This eliminates the "it looks different on my phone" class of bug reports.
