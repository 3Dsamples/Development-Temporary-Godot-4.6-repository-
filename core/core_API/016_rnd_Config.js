API Documentation — src/core/016_rnd_Config.js

File Purpose

This file is the central configuration hub for the anime lighting stack on Android mobile. It is the single source of truth for every tunable knob the engine exposes: platform profile, PERF_TIER-derived quality budgets, light counts, shadow resolutions, GI and AO sample budgets, post-processing presets, GPU and CPU memory caps, draw-call caps, and the runtime-override surface that debug UIs and adaptive-quality controllers write into.

The configuration is layered with seven levels of precedence, from lowest to highest:

1. ENGINE_DEFAULTS — hardcoded safe values baked into the module.
2. PLATFORM_PROFILE — the Android device class (low / mid / high) derived from navigator hints.
3. QUALITY_TIER — LOW / MEDIUM / HIGH / ULTRA, where ULTRA overrides HIGH with maxed-out overrides.
4. ANIME_STYLE_PRESET — reference-image-matched look controls (cel steps, rim power, shadow tint).
5. PERFORMANCE_PRESET — balanced / quality / performance / battery profiles that scale budgets.
6. Scene or biome override (set externally via the world core).
7. Runtime overrides — set at runtime by UI or adaptive quality controller.

The final resolved config is a single frozen plain object with nested sections: platform, renderer, camera, lights, shadows, gi, ao, environment, interior, exterior, post, memory, drawCalls, style, performance. Downstream systems read CONFIG.lights.maxPointLights and similar, never re-derive from PERF_TIER themselves. This is the only place where the tier gets translated into concrete numbers.

The config also owns the reactive surface: set(path, value), applyOverrides(map), setStylePreset, setPerformancePreset, setQualityTier. Every mutation rebuilds the resolved config and emits a changed event so downstream listeners can re-read.

---

Exported Constants

PLATFORM

Type: frozen object

The result of _detectPlatform(). Fields:

· isAndroid — boolean, true if navigator.userAgent matches /Android/.
· isMobile — boolean, true if Android or iPhone/iPad/iPod/Mobile.
· isTouch — boolean, true if ontouchstart is on window or navigator.maxTouchPoints > 0.
· deviceMemory — number, navigator.deviceMemory or 4.
· hardwareConcurrency — number, navigator.hardwareConcurrency or 4.
· devicePixelRatio — number, window.devicePixelRatio or 1.
· tier — one of 'LOW' | 'MEDIUM' | 'HIGH'. Computed from memory, cores, and DPR.
· vendor — string, navigator.vendor or ''.
· language — string, navigator.language or 'en'.

PERF_TIER

Type: string

Value: PLATFORM.tier. The cached performance tier for the current device.

PERF_RANK

Type: number

Value: 3 for HIGH, 2 for MEDIUM, 1 for LOW. A numeric rank used for comparisons.

QUALITY_TIER

Type: frozen enum

Values:

· LOW = 0
· MEDIUM = 1
· HIGH = 2
· ULTRA = 3

QUALITY_TIER_NAME

Type: frozen array

Values: ['low', 'medium', 'high', 'ultra'].

ANIME_STYLE_PRESET

Type: frozen enum

Values:

· REFERENCE_MATCHED = 0 — exact color match to the reference image set (cel steps 4, rim power 3.0, shadow tint [0.12, 0.18, 0.30]).
· SOFT_CEL = 1 — softer bands (cel steps 6, soft shadow edge).
· HARD_CEL = 2 — crisp hard bands (cel steps 3, minimal shadow softness).
· PASTEL_CEL = 3 — pastel tinted (cel steps 5, warm shadow tint).
· NIGHT_CEL = 4 — high contrast night (cel steps 3, deep shadow tint).

PERFORMANCE_PRESET

Type: frozen enum

Values:

· BALANCED = 0 — the default, best visual/battery tradeoff.
· QUALITY = 1 — prefer visuals, higher battery cost.
· PERFORMANCE = 2 — prefer FPS, drop resolution first.
· BATTERY = 3 — prefer battery, drop post and shadows first.

BUDGET_BY_TIER

Type: frozen object

Maps tier names to their budget objects. Three keys: 'LOW', 'MEDIUM', 'HIGH'.

LOW_BUDGET (internal)

Type: frozen object

The base budget for LOW-tier devices. Values:

· dprCap — 1.25
· shadowMapSize — 512
· shadowCascadeCount — 1
· shadowSoftness — 0.0
· shadowFilter — 'pcf'
· shadowDistance — 60.0
· maxPointLights — 8
· maxSpotLights — 4
· maxRectAreaLights — 0
· maxDirectionalLights — 1
· maxHemisphereLights — 1
· maxAmbientLights — 1
· maxShadowCastingLights — 1
· maxClusterLights — 32
· clusterGridRes — [16, 9, 24]
· giProbeGridRes — 8
· giUpdateBudgetHz — 8
· giSampleCount — 4
· giMultiBounce — false
· giHalfRes — true
· aoResolutionScale — 0.5
· aoSampleCount — 4
· aoHalfRes — true
· aoTemporalAccum — false
· postPassBudget — 2
· postBloom — false
· postVolumetric — false
· postSSAO — false
· postSSGI — false
· postMotionBlur — false
· postTAA — false
· maxInstancedDrawCalls — 128
· maxTriangles — 250000
· maxGpuMemoryMB — 48
· maxCpuHeapMB — 128
· targetFps — 30
· minFps — 24
· adaptiveQuality — true

MEDIUM_BUDGET (internal)

Type: frozen object

The base budget for MEDIUM-tier devices. Doubles most of the LOW values:

· dprCap — 1.75
· shadowMapSize — 1024
· shadowCascadeCount — 2
· shadowFilter — 'pcf'
· maxPointLights — 16
· maxSpotLights — 8
· maxRectAreaLights — 2
· maxShadowCastingLights — 2
· maxClusterLights — 64
· clusterGridRes — [24, 14, 32]
· giProbeGridRes — 16
· giUpdateBudgetHz — 15
· giSampleCount — 8
· giMultiBounce — true
· aoSampleCount — 8
· aoTemporalAccum — true
· postPassBudget — 4
· postBloom — true
· postVolumetric — true
· postSSAO — true
· postTAA — true
· maxInstancedDrawCalls — 256
· maxTriangles — 750000
· maxGpuMemoryMB — 128
· maxCpuHeapMB — 256
· targetFps — 45
· minFps — 30

HIGH_BUDGET (internal)

Type: frozen object

The base budget for HIGH-tier devices.

· dprCap — 2.0
· shadowMapSize — 2048
· shadowCascadeCount — 4
· shadowSoftness — 0.08
· shadowFilter — 'pcss'
· maxPointLights — 32
· maxSpotLights — 16
· maxRectAreaLights — 4
· maxShadowCastingLights — 4
· maxClusterLights — 128
· clusterGridRes — [32, 18, 48]
· giProbeGridRes — 32
· giUpdateBudgetHz — 20
· giSampleCount — 16
· giMultiBounce — true
· aoResolutionScale — 1.0
· aoSampleCount — 16
· aoTemporalAccum — true
· postPassBudget — 6
· postBloom — true
· postVolumetric — true
· postSSAO — true
· postSSGI — true
· postMotionBlur — true
· postTAA — true
· maxInstancedDrawCalls — 512
· maxTriangles — 2000000
· maxGpuMemoryMB — 256
· maxCpuHeapMB — 512
· targetFps — 60
· minFps — 30

STYLE_OVERRIDES (internal)

Type: frozen object

Maps each ANIME_STYLE_PRESET to its anime-style numeric overrides. Fields per preset: celSteps, rimPower, rimIntensity, shadowSoftness, shadowTint, ambientStrength.

PERF_MULTIPLIERS (internal)

Type: frozen object

Maps each PERFORMANCE_PRESET to its per-field multipliers applied on top of the tier budget. Fields: dprScale, shadowScale, giHzScale, aoScale, postPassDelta, clusterLightScale, targetFpsDelta.

---

Exported Class — Config

Constructor

```
new Config(options = {})
```

Parameters:

· tier — 'LOW' | 'MEDIUM' | 'HIGH'. Default PERF_TIER.
· qualityTier — one of QUALITY_TIER. Default derived from tier.
· stylePreset — one of ANIME_STYLE_PRESET. Default REFERENCE_MATCHED.
· perfPreset — one of PERFORMANCE_PRESET. Default BALANCED.
· overrides — an initial override map. Default null.
· frozen — reserved.

Constructor work:

1. Stores the options.
2. Copies the override map into _overrides.
3. Calls _buildResolved() to compute the initial resolved object.
4. Sets _version = 1.
5. Allocates _listeners (Map).

Instance Properties (Read-Only Getters)

· section — the full resolved object.
· lights — the lights subsection.
· shadows — the shadows subsection.
· gi — the gi subsection.
· ao — the ao subsection.
· environment — the environment subsection.
· interior — the interior subsection.
· exterior — the exterior subsection.
· post — the post subsection.
· memory — the memory subsection.
· drawCalls — the drawCalls subsection.
· style — the style subsection.
· performance — the performance subsection.
· renderer — the renderer subsection.
· platform — the platform subsection.
· version — the current version integer.

Instance Methods

set(path, value)

Parameters:

· path — a dotted path like 'lights.maxPointLights'.
· value — the value to set.

Returns: boolean.

Purpose: writes into the override map at the given path. Splits the path on ., walks into nested objects, creates empty objects as needed. Increments the version, rebuilds the resolved config, emits changed.

applyOverrides(overrides)

Parameters: overrides — an object with one or more top-level sections. Fields are merged one level deep: { lights: { maxPointLights: 64 } } merges maxPointLights into the existing lights override.

Returns: boolean.

Purpose: bulk override application. Emits changed with path: '*'.

setStylePreset(stylePreset)

Parameters: stylePreset — one of ANIME_STYLE_PRESET.

Returns: boolean.

Purpose: changes the anime style preset. Rebuilds the resolved config. Emits style.

setPerformancePreset(perfPreset)

Parameters: perfPreset — one of PERFORMANCE_PRESET.

Returns: boolean.

Purpose: changes the performance preset. Rebuilds. Emits perf.

setQualityTier(qualityTier)

Parameters: qualityTier — one of QUALITY_TIER.

Returns: boolean.

Purpose: changes the quality tier. Maps the enum to a base tier (HIGH, MEDIUM, LOW, or HIGH for ULTRA). If the tier is ULTRA, applies a set of maximum overrides: 64 point lights, 32 spot lights, 8 rect area lights, 256 cluster lights, 4096 shadow map size, 4 cascades, 48 GI probe resolution, 30 Hz GI, 32 GI samples, full AO resolution, 32 AO samples, 8 post passes, SSGI on.

Rebuilds. Emits quality.

_rebuild()

Internal. Calls _buildResolved() with the current options and overrides.

on(event, fn)

Parameters:

· event — one of 'changed', 'style', 'perf', 'quality'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to every listener registered for event.

toJSON()

Returns: an object with version, options, overrides, resolved. The overrides are deep-cloned so the caller cannot mutate the internal state.

---

Internal Functions (Not Exported but Documented)

_detectPlatform()

Returns: the PLATFORM object. Reads navigator.userAgent, navigator.deviceMemory, navigator.hardwareConcurrency, window.devicePixelRatio. Computes the tier. Called once at module load.

_buildResolved(tier, stylePreset, perfPreset, overrides)

Parameters:

· tier — the budget tier.
· stylePreset — the anime style preset.
· perfPreset — the performance preset.
· overrides — the override map.

Returns: a frozen object with all fifteen sections.

Purpose: the core config builder.

Flow:

1. Fetches the base budget via BUDGET_BY_TIER[tier].
2. Fetches the anime style overrides via STYLE_OVERRIDES[stylePreset].
3. Fetches the performance multipliers via PERF_MULTIPLIERS[perfPreset].
4. Computes derived values:
   · dprCap = min(base.dprCap * mult.dprScale, 2.0)
   · shadowMapSize = nearestPowerOfTwo(max(256, base.shadowMapSize * mult.shadowScale))
   · giHz = max(4, round(base.giUpdateBudgetHz * mult.giHzScale))
   · aoScale = clamp(base.aoResolutionScale * mult.aoScale, 0.25, 1.0)
   · clusterLights = max(8, round(base.maxClusterLights * mult.clusterLightScale))
   · postPass = clamp(base.postPassBudget + mult.postPassDelta, 0, 8)
   · targetFps = clamp(base.targetFps + mult.targetFpsDelta, 24, 120)
5. Builds the fifteen sections: platform, renderer, camera, lights, shadows, gi, ao, environment, interior, exterior, post, memory, drawCalls, style, performance.
6. Applies the override map on top: for each top-level key in overrides, if the value is an object and the corresponding resolved section is also an object, merges the override fields in. Otherwise replaces the section entirely.
7. Freezes the top-level and every nested object.
8. Returns.

_nearestPowerOfTwo(v)

Returns the nearest power of two to v. Used for shadow map sizes because Android GPUs handle power-of-two shadow maps much faster than arbitrary sizes.

_clampInt(v, min, max)

Clamps v to [min, max] and rounds to integer.

---

Exported Functions

getDefaultConfig()

Returns: the module-level singleton Config, creating it on first call.

disposeDefaultConfig()

Returns: nothing.

getResolvedConfig()

Returns: the resolved section of the default config.

Purpose: the fast-access function for per-frame reads. Downstream systems call this once at subsystem init, cache the subsection references they need, and never call it per frame. The returned object is frozen so it cannot be accidentally mutated.

createConfig(options = {})

Returns: a new Config.

---

Default Export

The default export bundles: Config, createConfig, getDefaultConfig, disposeDefaultConfig, getResolvedConfig, PLATFORM, PERF_TIER, PERF_RANK, QUALITY_TIER, QUALITY_TIER_NAME, ANIME_STYLE_PRESET, PERFORMANCE_PRESET, BUDGET_BY_TIER.

---

Usage Pattern

The bootstrap reads the config once:

```
import {
  getDefaultConfig,
  getResolvedConfig,
  PERFORMANCE_PRESET,
} from './src/core/016_rnd_Config.js';

const config = getDefaultConfig();
config.setPerformancePreset(PERFORMANCE_PRESET.BALANCED);

const resolved = getResolvedConfig();
console.log('Shadow map size:', resolved.shadows.mapSize);
console.log('Max point lights:', resolved.lights.maxPointLights);
console.log('Target FPS:', resolved.performance.targetFps);
```

A debug UI or adaptive controller applies an override:

```
config.applyOverrides({
  lights: { maxPointLights: 48 },
  shadows: { mapSize: 1536 },
  post: { passBudget: 5 },
});
```

An adaptive quality controller listening for tier changes:

```
config.on('changed', ({ version }) => {
  const resolved = getResolvedConfig();
  // Re-apply shadow map size, GI update rate, etc.
});
```

Because the resolved config is frozen, downstream systems can safely cache subsection references at init time. If the config changes, they listen for the changed event and re-read. This eliminates the classic Android bug where a subsystem reads the config on every frame and accidentally caches a stale reference.
