API Documentation — src/core/023_rnd_PerfTier.js

File Purpose

This file is the authoritative performance tier resolver for the anime lighting stack on Android mobile. Where 016_rnd_Config.js declared an initial PERF_TIER from a coarse navigator hint (deviceMemory, hardwareConcurrency, devicePixelRatio), this module owns the full, refined, re-evaluable tier classification that consolidates every signal source in the codebase into a single tier value.

The problem it solves is signal fragmentation. By the time the engine is ready to decide quality, it has access to a dozen partial views of "how fast is this device":

· PLATFORM from 016_rnd_Config.js — coarse tier based on RAM, cores, DPR.
· PLATFORM_CONFIG from 018_rnd_PlatformConfig.js — refined profile tier from GPU classification.
· ANDROID_PROFILE from 019_rnd_AndroidProfile.js — concrete GPU family.
· DESKTOP_PROFILE from 020_rnd_DesktopProfile.js — desktop GPU family.
· CAPABILITIES from 021_rnd_Capabilities.js — WebGL features and precision support.
· FEATURE_DETECTOR from 022_rnd_FeatureDetector.js — runtime features and cross-origin isolation.

Without this module, every subsystem would pick a different tier and the engine would drift into inconsistency. With this module, all those signals are aggregated via a weighted scoring function that produces one tier value, with a confidence score, and can be re-evaluated on demand.

The module also owns external bias caps — thermal, battery, saveData, WebGL1 — that act as hard ceilings on the tier. The tier can never rise above what these biases allow, so the engine always respects OS-level throttling and device limitations.

The final tier is a discrete value from a five-level ladder:

· MINIMAL — fallback for WebGL1 or software rasterizers.
· LOW — entry Android.
· MEDIUM — mid-range Android.
· HIGH — flagship Android.
· ULTRA — desktop/emulator/QA.

The discrete ladder matters because shader permutations, render target bucket sizes, and cluster grid resolutions stay cached. If the tier were continuous, every small change would force a shader recompile.

---

Exported Constants

PERF_TIER

Type: frozen enum

Values:

· MINIMAL = 0
· LOW = 1
· MEDIUM = 2
· HIGH = 3
· ULTRA = 4
· COUNT = 5

Lower values mean lower capability. This is the same direction as the enum in 016_rnd_Config.js, with the addition of a MINIMAL level below LOW and an ULTRA level above HIGH.

PERF_TIER_NAME

Type: frozen array

Values: ['minimal', 'low', 'medium', 'high', 'ultra'].

PERF_TIER_RANK

Type: frozen object

Maps the tier names to their numeric values: { MINIMAL: 0, LOW: 1, MEDIUM: 2, HIGH: 3, ULTRA: 4 }.

SIGNAL_WEIGHTS (internal)

Type: frozen object

The weight of each signal in the weighted-average score.

· gpuClass — 0.30
· ramClass — 0.20
· cpuClass — 0.15
· webglClass — 0.10
· featureClass — 0.10
· platformClass — 0.10
· precisionClass — 0.05

The weights sum to 1.0. The GPU is the dominant signal because the lighting stack's computational cost is dominated by fragment shader work. RAM is the second because RAM pressure causes GC pauses that are worse than raw CPU slowness on mobile. CPU, WebGL, features, platform, and precision fill in the rest.

SCORE_THRESHOLDS (internal)

Type: frozen object

The score thresholds for each tier.

· MINIMAL — 0.00
· LOW — 0.20
· MEDIUM — 0.45
· HIGH — 0.70
· ULTRA — 0.90

Any score in [0.00, 0.20) produces MINIMAL. [0.20, 0.45) produces LOW. [0.45, 0.70) produces MEDIUM. [0.70, 0.90) produces HIGH. [0.90, 1.00] produces ULTRA.

UPGRADE_HOLD_EVALS (internal)

Type: number

Value: 3

The number of consecutive evaluations above the upgrade threshold required before the tier actually rises. Prevents a single good spike from triggering an upgrade.

DOWNGRADE_HOLD_EVALS (internal)

Type: number

Value: 1

The number of consecutive evaluations below the downgrade threshold required before the tier actually falls. Downgrades are immediate because the engine prefers to drop quality fast and recover slowly.

BIAS_CAP (internal)

Type: frozen object

The maximum tier each external bias allows.

· thermalNormal — ULTRA
· thermalWarm — HIGH
· thermalHot — MEDIUM
· thermalVeryHot — LOW
· thermalCritical — MINIMAL
· batteryFull — ULTRA
· batteryGood — HIGH
· batteryLow — MEDIUM
· batteryVeryLow — LOW
· batteryCritical — MINIMAL
· saveData — MEDIUM
· webgl1 — LOW
· software — MINIMAL

When any bias is applied, the effective tier is capped at the corresponding value.

BOOT_SNAPSHOT

Type: frozen TierSnapshot

The snapshot taken at module load time. Contains the initial tier computed from the current device signals. Read-only after creation.

BOOT_TIER

Type: number

The effectiveTier value from BOOT_SNAPSHOT.

BOOT_TIER_NAME

Type: string

The effectiveTierName from BOOT_SNAPSHOT.

BOOT_SCORE

Type: number

The score value from BOOT_SNAPSHOT.

BOOT_CONFIDENCE

Type: number

The confidence value from BOOT_SNAPSHOT.

PERF_TIER_STRING

Type: string

The BOOT_TIER_NAME uppercase — 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA'. This is the alias that existing code that imports PERF_TIER as a string continues to work with.

---

Module-Level State (Not Exported Directly)

_bootResolver

Type: PerfTierResolver | null

The resolver instance created at module load to produce the boot snapshot. Retained as the module-level default resolver.

_defaultResolver

Type: PerfTierResolver | null

The lazily-created module-level singleton. Created on first getDefaultPerfTierResolver() call.

---

Exported Class — TierSnapshot

The mutable state container for a single resolver instance. Updated in place on every evaluate() call.

Constructor

```
new TierSnapshot()
```

Instance Properties

· tier — the raw resolved tier before bias caps.
· tierName — the raw tier name string.
· tierRank — the numeric rank of the raw tier.
· score — the weighted average score in [0, 1].
· confidence — a confidence estimate in [0, 1].
· signals — the object containing every individual signal classification.
· thermalCap — the tier cap from the thermal bias.
· batteryCap — the tier cap from the battery bias.
· extraCap — the tier cap from the extra bias (saveData or user override).
· effectiveTier — the final tier after all caps and overrides.
· effectiveTierName — the final tier name.
· frame — the resolver frame counter.
· timestamp — the last evaluation timestamp in milliseconds.
· override — true if a user override is currently active.
· evaluations — the total number of evaluate() calls.

The signals object has seven sub-objects: gpu, ram, cpu, webgl, feature, platform, precision. Each sub-object has a score in [0, 1] and a label string.

---

Exported Class — PerfTierResolver

The main class.

Constructor

```
new PerfTierResolver(options = {})
```

Parameters:

· initialTier — an optional tier enum value or name. If supplied, calls setOverride with it after the first evaluate.
· enableAdaptive — reserved. Default true.
· upgradeHoldEvals — the number of consecutive evaluations required to upgrade. Default UPGRADE_HOLD_EVALS.
· downgradeHoldEvals — the number of consecutive evaluations required to downgrade. Default DOWNGRADE_HOLD_EVALS.
· logDecisions — if true, logs every tier change to the console. Default false.

Constructor work:

1. Initializes _frame, _snapshot, _lastEvaluatedTier, _upgradeAccum, _downgradeAccum, _overrideTier.
2. Allocates _listeners (Map).
3. Calls this.evaluate() to compute the initial tier.
4. If initialTier was supplied, calls setOverride on it.

Instance Properties (Read-Only Getters)

· tier — the raw tier enum value from the snapshot.
· tierName — the raw tier name string.
· effectiveTier — the final tier enum value.
· effectiveTierName — the final tier name string.
· score — the weighted score.
· confidence — the confidence.
· isOverridden — whether a user override is active.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'upgrade', 'downgrade', 'evaluated'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to registered listeners.

_gatherSignals()

Internal. Calls the seven signal classifiers and returns a single object.

evaluate()

Returns: the updated TierSnapshot.

Purpose: the main evaluation entry point.

Flow:

1. Increments _frame.
2. Gathers every signal.
3. Computes the weighted score via _computeWeightedScore.
4. Maps the score to a raw tier via _scoreToTier.
5. Applies asymmetric hysteresis:
   · If the raw tier is higher than the last evaluated tier, increments _upgradeAccum and clears _downgradeAccum. If _upgradeAccum >= upgradeHoldEvals, accepts the new tier and resets the accumulator.
   · If the raw tier is lower, increments _downgradeAccum and clears _upgradeAccum. If _downgradeAccum >= downgradeHoldEvals, accepts the new tier.
   · If the raw tier equals the last, clears both accumulators.
6. Stores the accepted tier in _lastEvaluatedTier.
7. Computes hard caps: MINIMAL if WebGL is unavailable, LOW if WebGL1, MINIMAL if SwiftShader.
8. Applies the thermal, battery, extra, and hard caps to produce the effective tier.
9. If an override is active, replaces the effective tier with the override.
10. Computes confidence by measuring the standard deviation of the seven signal scores and mapping 1 - stddev * 1.6 clamped to [0, 1].
11. Writes every field into the snapshot.
12. Emits evaluated.

setThermalBias(input)

Parameters: input — a number in [0, 1] or a string from 'nominal' | 'warm' | 'fair' | 'serious' | 'critical'.

Returns: this.

Purpose: updates the thermal cap. Number thresholds: 0.90+ → CRITICAL cap, 0.70+ → VERY_HOT, 0.50+ → HOT, 0.25+ → WARM, else NORMAL. String mapping: 'critical' → CRITICAL, 'serious' → VERY_HOT, 'fair' → HOT, 'warm' → WARM, anything else → NORMAL.

Calls evaluate() after the cap is updated.

setBatteryBias(level, charging)

Parameters:

· level — a number in [0, 1].
· charging — boolean.

Returns: this.

Purpose: updates the battery cap. If charging, cap is FULL (no limit). Otherwise, level thresholds: 0.15- → CRITICAL, 0.20- → VERY_LOW, 0.30- → LOW, 0.50- → GOOD, else FULL.

Calls evaluate().

setExtraBias(bias)

Parameters: bias — a number in [0, 1] representing how aggressively to cap.

Returns: this.

Purpose: a generic bias hook for saveData or any other pressure. 0 means no cap. 0.25 or below caps to WARM. 0.50 or below caps to HOT. 0.75 or below caps to VERY_HOT. Otherwise caps to CRITICAL.

Calls evaluate().

setSaveDataBias(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: convenience hook that sets extraCap to MEDIUM if the saveData flag is on, otherwise no cap.

Calls evaluate().

setOverride(tier)

Parameters: tier — a tier enum value, a tier name string, or null to clear.

Returns: this.

Purpose: forces the effective tier to a specific value regardless of the aggregated signals. Used by debug GUIs and QA tools.

Calls evaluate().

clearOverride()

Returns: this.

Purpose: clears the override and re-evaluates.

getSnapshot()

Returns: the TierSnapshot.

getBudget()

Returns: the BUDGET_BY_TIER entry for the effective tier name (uppercase). If the effective tier is ULTRA or MINIMAL and BUDGET_BY_TIER does not have that key, returns the MEDIUM budget as a fallback.

getConfigTierName()

Returns: the effective tier name uppercase — 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'ULTRA'. This is the string format 016_rnd_Config.js expects.

getStats()

Returns: an object with tier, effectiveTier, score, confidence, override, thermalCap, batteryCap, extraCap, evaluations, the full signals object, perfTierLegacy (the coarse tier from 016), and bootstrapTier (the cached tier from 008_scn_world.js).

reset()

Returns: this. Clears all accumulators, caps, and overrides, then re-evaluates.

dispose()

Returns: this. Clears the listener map.

Internal Static Functions

_computeWeightedScore(signals)

Returns: the weighted average in [0, 1].

_scoreToTier(score)

Returns: the tier enum value that corresponds to the score.

_classifyGpuSignal()

Returns: { score, label }.

The classification:

· Desktop with a modern discrete GPU or Apple M-series → { score: 1.00, label: 'high_end_desktop' }.
· Desktop with Iris or Apple Intel iGPU → { score: 0.65, label: 'mid_desktop_igpu' }.
· Desktop with software rasterizer → { score: 0.10, label: 'software_desktop' }.
· Other desktop → { score: 0.45, label: 'low_desktop' }.
· Android Adreno 7xx/8xx or Mali G7x/G8x → { score: 0.95, label: 'flagship_android' }.
· Android Adreno 6xx or Mali G6x → { score: 0.75, label: 'upper_mid_android' }.
· Android Adreno 5xx or Mali G5x → { score: 0.55, label: 'mid_android' }.
· Android Adreno 4xx, Mali G3x, or PowerVR Rogue → { score: 0.35, label: 'low_android' }.
· Android Adreno 3xx, Mali-T, or PowerVR GE → { score: 0.15, label: 'legacy_android' }.
· SwiftShader → { score: 0.05, label: 'software' }.
· Unknown → { score: 0.45, label: 'unknown_gpu' }.

_classifyRamSignal()

Returns: { score, label }.

· 12+ GB → { score: 1.00, label: 'ram_12plus' }.
· 8+ GB → { score: 0.85 }.
· 6+ GB → { score: 0.65 }.
· 4+ GB → { score: 0.50 }.
· 3+ GB → { score: 0.35 }.
· 2+ GB → { score: 0.20 }.
· Else → { score: 0.10 }.

_classifyCpuSignal()

Returns: { score, label }.

· 12+ cores → 1.00.
· 8+ → 0.85.
· 6+ → 0.65.
· 4+ → 0.45.
· 2+ → 0.25.
· Else → 0.10.

_classifyWebglSignal()

Returns: { score, label }.

Starts at 0.30. Adds 0.30 if WebGL2. Adds 0.15 if maxTextureSize >= 4096. Adds 0.10 more if >= 8192. Adds 0.10 if maxDrawBuffers >= 4. Adds 0.05 if maxSamples >= 4. Clamped to [0, 1].

_classifyFeatureSignal()

Returns: { score, label }.

Weighs thirteen runtime features. Each one contributes a fraction of the total. canUseWorkers, canUseWorkerPool, canUseSharedMemory, canUseOffscreenRT, canUsePerfMarks, canUseIdleCallback, canUsePointerEvents, canUseBatteryGuard, canUseThermalGuard, canUseWASM, supportsMRT, canUseHDRTargets, canUseInstancing. The total is normalized to [0, 1].

_classifyPlatformSignal()

Returns: { score, label }.

· 'HIGH' → 0.90.
· 'MEDIUM' → 0.60.
· 'LOW' → 0.30.
· Anything else → 0.50.

_classifyPrecisionSignal()

Returns: { score, label }.

· Both vertex and fragment highp → 1.00.
· Vertex highp, fragment mediump → 0.65.
· Fragment mediump → 0.45.
· Else → 0.20.

---

Exported Functions

getEffectiveTier()

Returns: the current effective tier enum value.

getEffectiveTierName()

Returns: the current effective tier name string.

getConfigTierName()

Returns: the current effective tier name string in uppercase.

getCurrentBudget()

Returns: the frozen budget table for the current tier.

isAtLeast(tierName)

Parameters: tierName — one of 'low' | 'medium' | 'high' | 'ultra' | 'minimal'.

Returns: boolean.

Purpose: convenience comparison. isAtLeast('high') returns true if the effective tier is HIGH or ULTRA.

reevaluateTier()

Returns: the updated snapshot.

Purpose: forces a re-evaluation of the tier. Call this after any external bias change (thermal recovery, battery plugged in, saveData toggled off).

getDefaultPerfTierResolver()

Returns: the module-level singleton PerfTierResolver, creating it on first call.

disposeDefaultPerfTierResolver()

Returns: nothing.

createPerfTierResolver(options = {})

Returns: a new PerfTierResolver.

getPerfTierReport()

Returns: an object with tier, effectiveTier, score, confidence, override, evaluations, thermalCap, batteryCap, extraCap, signals, platform, gpu, hardwareConcurrency, deviceMemoryGB, workerPoolSize, webgl2, quirks, configTier, bootstrapTier.

Purpose: the human-readable summary for the debug HUD or CI log.

---

Default Export

The default export bundles: PerfTierResolver, TierSnapshot, PERF_TIER, PERF_TIER_NAME, PERF_TIER_RANK, PERF_TIER_STRING, BOOT_SNAPSHOT, BOOT_TIER, BOOT_TIER_NAME, BOOT_SCORE, BOOT_CONFIDENCE, createPerfTierResolver, getDefaultPerfTierResolver, disposeDefaultPerfTierResolver, getEffectiveTier, getEffectiveTierName, getConfigTierName, getCurrentBudget, isAtLeast, reevaluateTier, getPerfTierReport.

---

Usage Pattern

The EngineLoop reads the tier once at startup:

```
import {
  getDefaultPerfTierResolver,
  getEffectiveTierName,
  isAtLeast,
  getPerfTierReport,
} from './src/core/023_rnd_PerfTier.js';

const resolver = getDefaultPerfTierResolver();
const tier = getEffectiveTierName();

console.log('Effective tier:', tier);
console.log(JSON.stringify(getPerfTierReport(), null, 2));

if (isAtLeast('high')) {
  // Enable full SSGI, 4 shadow cascades, 32-sample GI.
} else if (isAtLeast('medium')) {
  // Enable PCF shadows, 2 cascades, 16-sample GI.
} else {
  // Enable basic shadows, 1 cascade, 4-sample GI.
}
```

The App's thermal guard updates the resolver in real time:

```
app.on('thermal', ({ state }) => {
  resolver.setThermalBias(state);
});

app.on('battery', ({ level, charging }) => {
  resolver.setBatteryBias(level, charging);
});
```

The resolver re-evaluates on every bias change, applies the caps, and emits upgrade or downgrade events when the effective tier moves. Downstream subsystems listening to those events can reallocate resources accordingly.

Because the resolver aggregates every available signal into one tier, there is no scenario where 016_rnd_Config.js says HIGH but 018_rnd_PlatformConfig.js says MEDIUM but 021_rnd_Capabilities.js says LOW. All three feed into the same weighted score, and the resolver produces one authoritative answer.
