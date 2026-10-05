API Documentation — src/core/017_rnd_QualityConfig.js

File Purpose

This file is the runtime quality controller for the anime lighting stack on Android mobile. Where 016_rnd_Config.js defines the static budget table for each PERF_TIER, this module owns the dynamic quality surface: a per-frame quality state machine that observes frame time, thermal pressure, battery level, and per-domain scheduler pressure, then decides — within strictly bounded steps — how to bias resolution scale, shadow map size, GI update rate, AO resolution, post-pass count, and draw distance so the target FPS is held without ever dropping a frame or changing the anime visual identity.

The controller is designed to avoid the two failure modes of naive adaptive quality:

1. Oscillation — a naive controller upgrades on one good frame, downgrades on one bad frame, and produces visible flicker as the scene quality bounces up and down. The controller here uses hysteresis bands: an upgrade requires 3–4 seconds of sustained good frames; a downgrade requires 0.6–0.8 seconds of sustained bad frames. The asymmetry matches real hardware behavior — Android throttles fast and recovers slow.
2. Cascade collapse — a naive controller reacts to a bad frame by dropping every quality knob at once, turning a momentary blip into a visible visual change. The controller here changes only ONE knob per decision cycle, so a single bad frame cannot cascade into a full visual downgrade.

The controller also reads per-domain pressure from the FrameScheduler (005_rnd_FrameScheduler.js) via getPressure(), giving it per-subsystem visibility. If shadows are over budget but post is idle, the controller biases shadows only, leaving post alone.

The controller is also externally boundable. Thermal bias and battery bias from the App layer (002_rnd_App.js) act as hard ceilings — the controller cannot override an OS-level throttle. If the OS says the device is critical, the controller is forced to LOW or MINIMAL quality regardless of what the frame-time EMA says.

---

Exported Constants

PERF_TIER_LOCAL

Internal constant. The cached PERF_TIER string from getPerfTier(). Used to select threshold defaults.

QUALITY_LEVEL

Type: frozen enum

Values:

· ULTRA = 0
· HIGH = 1
· MEDIUM = 2
· LOW = 3
· MINIMAL = 4
· COUNT = 5

Lower values mean higher quality. This is the inverse of the QUALITY_TIER enum in 016_rnd_Config.js — ULTRA is the highest quality, MINIMAL is the floor.

QUALITY_LEVEL_NAME

Type: frozen array

Values: ['ultra', 'high', 'medium', 'low', 'minimal'].

QUALITY_KNOB

Type: frozen enum

The individual quality parameters the controller may bias.

Values:

· NONE = 0
· RESOLUTION = 1
· SHADOWS = 2
· GI = 3
· AO = 4
· POST = 5
· DRAW_DISTANCE = 6
· CLUSTER = 7
· COUNT = 8

QUALITY_KNOB_NAME

Type: frozen array

Values: ['none', 'resolution', 'shadows', 'gi', 'ao', 'post', 'draw_distance', 'cluster'].

QUALITY_LEVEL_MULTIPLIERS

Type: frozen object

Maps each QUALITY_LEVEL to its multiplier block. Each block has these fields:

· resolutionScale — multiplier applied to DPR.
· shadowScale — multiplier applied to shadow map size.
· giHzScale — multiplier applied to GI update Hz.
· aoScale — multiplier applied to AO resolution and sample count.
· postDelta — additive delta applied to the post-pass budget.
· drawDistance — multiplier applied to fog far distance.
· clusterScale — multiplier applied to max cluster light count.

The values:

ULTRA: { resolutionScale: 1.00, shadowScale: 1.00, giHzScale: 1.00, aoScale: 1.00, postDelta: 0, drawDistance: 1.00, clusterScale: 1.00 }

HIGH: { resolutionScale: 0.90, shadowScale: 0.85, giHzScale: 0.85, aoScale: 0.85, postDelta: -1, drawDistance: 0.90, clusterScale: 0.85 }

MEDIUM: { resolutionScale: 0.75, shadowScale: 0.65, giHzScale: 0.65, aoScale: 0.65, postDelta: -2, drawDistance: 0.75, clusterScale: 0.65 }

LOW: { resolutionScale: 0.60, shadowScale: 0.50, giHzScale: 0.50, aoScale: 0.50, postDelta: -3, drawDistance: 0.60, clusterScale: 0.50 }

MINIMAL: { resolutionScale: 0.45, shadowScale: 0.35, giHzScale: 0.35, aoScale: 0.35, postDelta: -4, drawDistance: 0.45, clusterScale: 0.35 }

THRESHOLDS (internal)

Type: frozen object

Per-tier threshold configuration for the decision logic. Each tier has:

· upgradeBelowMs — frame-time threshold below which the upgrade accumulator advances.
· downgradeAboveMs — frame-time threshold above which the downgrade accumulator advances.
· upgradeHoldMs — how many milliseconds of sustained good frames before an upgrade is allowed.
· downgradeHoldMs — how many milliseconds of sustained bad frames before a downgrade is allowed.

The values:

HIGH: { upgradeBelowMs: 12.0, downgradeAboveMs: 18.0, upgradeHoldMs: 4000, downgradeHoldMs: 800 }

MEDIUM: { upgradeBelowMs: 18.0, downgradeAboveMs: 24.0, upgradeHoldMs: 3500, downgradeHoldMs: 700 }

LOW: { upgradeBelowMs: 30.0, downgradeAboveMs: 40.0, upgradeHoldMs: 3000, downgradeHoldMs: 600 }

The thresholds are in milliseconds. On HIGH, the safe band is [12, 18] ms. Frames inside the band decay both accumulators.

---

Exported Class — QualityState

A plain data structure holding every piece of dynamic quality state.

Constructor

```
new QualityState(tier)
```

Parameters: tier — one of 'LOW' | 'MEDIUM' | 'HIGH'. Used to select the thresholds.

Instance Properties

· tier — the device tier.
· level — the current QUALITY_LEVEL.
· targetLevel — the level the controller is transitioning toward.
· lastChangeFrame — the frame number of the last level change.
· changeCount — the total number of level changes.
· upgradeAccumMs — the accumulator for the upgrade decision.
· downgradeAccumMs — the accumulator for the downgrade decision.
· resolutionBias — [0, 1] bias applied to resolution.
· shadowBias — [0, 1].
· giBias — [0, 1].
· aoBias — [0, 1].
· postBias — [0, 1].
· drawDistanceBias — [0, 1].
· clusterBias — [0, 1].
· thermalBias — [0, 1] external input.
· batteryBias — [0, 1] external input.
· lowPower — 0 or 1.
· lastKnob — the QUALITY_KNOB that was adjusted most recently.
· lastReason — a string describing the last decision.
· lastDecisionMs — the timestamp of the last decision.
· framesObserved — the total frames the controller has seen.
· framesUpgraded — the total frames spent upgrading.
· framesDowngraded — the total frames spent downgrading.
· framesHeld — the total frames where the level was held.
· _thresholds — the threshold object for the tier.

Instance Methods

reset()

Returns: nothing. Resets every field to its initial value.

---

Exported Class — QualitySnapshot

A pre-allocated per-frame read-only view. Downstream systems poll this once per frame without allocating.

Constructor

```
new QualitySnapshot()
```

Instance Properties

· frame — the frame counter.
· level — one of QUALITY_LEVEL.
· levelName — the string name.
· resolutionScale — the resolved DPR multiplier.
· shadowMapSize — the resolved shadow map size in pixels.
· shadowCascadeCount — the resolved cascade count.
· giUpdateBudgetHz — the resolved GI update rate.
· giSampleCount — the resolved GI sample count.
· aoResolutionScale — the resolved AO resolution scale.
· aoSampleCount — the resolved AO sample count.
· postPassBudget — the resolved post-pass count.
· drawDistanceMeters — the resolved fog far distance.
· maxClusterLights — the resolved max cluster light count.
· thermalBias — the current external thermal bias.
· batteryBias — the current external battery bias.
· lowPower — the current low-power flag.
· frameTimeMs — the last frame time in milliseconds.
· lastKnob — the last knob that was adjusted.
· lastReason — the last decision reason.

This snapshot is updated in place every frame. No allocation occurs after construction.

---

Exported Class — QualityController

The main controller.

Constructor

```
new QualityController(options = {})
```

Parameters:

· initialLevel — one of QUALITY_LEVEL. Default HIGH.
· enableAdaptive — if false, the controller never changes level and always applies the initial. Default true.
· enableThermal — reserved. Default true.
· enableBattery — reserved. Default true.
· minLevel — the minimum level the controller may reach. Default MINIMAL.
· maxLevel — the maximum level. Default ULTRA.
· decisionHz — how many times per second the controller runs its decision logic. Default 4.
· logDecisions — if true, logs every level change to the console. Default false.

Constructor work:

1. Stores options.
2. Fetches the default Config singleton from 016_rnd_Config.js.
3. Fetches the default FrameScheduler singleton from 005_rnd_FrameScheduler.js.
4. Allocates state — a QualityState for the current tier.
5. Allocates snapshot — a QualitySnapshot.
6. Caches _baseBudget from getResolvedConfig().
7. Sets the initial level from the options.
8. Allocates _decisionAccumMs, _decisionStepMs, _frame.
9. Allocates _domainPressure — a Float32Array(DOMAIN.COUNT).
10. Allocates _listeners (Map).
11. Calls _applyLevel(true) to populate the snapshot.

Instance Properties

· options — the merged options.
· config — the Config instance.
· scheduler — the FrameScheduler instance.
· state — the QualityState.
· snapshot — the QualitySnapshot.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'level', 'applied'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

setThermalBias(bias)

Parameters: bias — [0, 1].

Returns: this.

Purpose: sets the external thermal bias. High bias values force the controller to LOW or MINIMAL.

setBatteryBias(bias)

Parameters: bias — [0, 1].

Returns: this.

setLowPower(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: forces LOW quality regardless of frame time.

update(dtMs, elapsedSec)

Parameters:

· dtMs — the last frame's delta in milliseconds.
· elapsedSec — total elapsed seconds.

Returns: the QualitySnapshot.

Purpose: the per-frame entry point. Called by the EngineLoop.

Flow:

1. Increments _frame and state.framesObserved.
2. Updates the snapshot's frame and frameTimeMs.
3. If adaptive is off, refreshes the snapshot from the current level and returns.
4. Applies external biases:
   · If lowPower === 1, forces level LOW with reason 'low_power'.
   · If thermalBias >= 0.8, forces LOW. If >= 0.5, forces MEDIUM.
   · If batteryBias >= 0.8, forces LOW. If >= 0.5, forces MEDIUM.
5. Accumulates _decisionAccumMs. If it exceeds the decision step, resets it and calls _runDecision(dtMs).
6. Reads scheduler.getPressure() into _domainPressure.
7. Refreshes the snapshot.
8. Returns the snapshot.

_runDecision(dtMs)

Internal. The core hysteresis logic.

Flow:

1. Reads the tier thresholds.
2. If dtMs > downgradeAboveMs, advances the downgrade accumulator, zeroes the upgrade accumulator.
3. Else if dtMs < upgradeBelowMs, advances the upgrade accumulator, zeroes the downgrade accumulator.
4. Else (inside the safe band), decays both accumulators by half the decision step.
5. If the downgrade accumulator crosses downgradeHoldMs and the level is not MINIMAL, picks the downgrade knob via _pickDowngradeKnob(), biases it by +0.25, and calls _changeLevel(level + 1, knobName, 'downgrade').
6. If the upgrade accumulator crosses upgradeHoldMs and the level is not ULTRA, and thermal/battery biases are low, picks the upgrade knob via _pickUpgradeKnob(), biases it by -0.20, and calls _changeLevel(level - 1, knobName, 'upgrade').
7. Otherwise increments framesHeld.

_pickDowngradeKnob()

Internal. Finds the domain with the highest pressure from _domainPressure and maps it to a QUALITY_KNOB:

· SHADOWS or LIGHTS → SHADOWS
· GI → GI
· AO → AO
· POST → POST
· CLUSTER → CLUSTER
· Default → RESOLUTION

_pickUpgradeKnob()

Internal. Picks the knob with the highest bias value (the most aggressively downgraded one) and reduces its bias first.

The upgrade priority order is: RESOLUTION, SHADOWS, GI, AO, POST, DRAW_DISTANCE, CLUSTER.

_getKnobBias(knob)

Internal. Returns the current bias for the given knob.

_applyKnobBias(knob, delta)

Internal. Adds delta to the knob's bias, clamped to [0, 1].

_changeLevel(newLevel, knobName, reason)

Internal. If the new level differs from the current one:

1. Updates state.level, state.targetLevel, state.lastChangeFrame, state.changeCount.
2. Records the knob name and reason.
3. Increments framesDowngraded or framesUpgraded.
4. Calls _applyLevel(false).
5. If logDecisions, logs the change.
6. Emits level with from/to/knob/reason.

_forceLevel(level, reason)

Internal. Calls _changeLevel only if the target level is lower (higher quality index is worse) than the current. This is the "cap" behavior for external biases.

_applyLevel(initial)

Internal. Computes the resolved snapshot values.

Flow:

1. Fetches the QUALITY_LEVEL_MULTIPLIERS for the current level.
2. Reads base values from _baseBudget.
3. Computes:
   · resolvedDpr = clamp(baseDpr * mult.resolutionScale * (1 - resolutionBias * 0.30), 0.5, baseDpr)
   · resolvedShadow = clamp(baseShadow * mult.shadowScale * (1 - shadowBias * 0.40), 256, baseShadow)
   · cascades = clamp(baseCascades, 1, 4 - floor(shadowBias * 3))
   · resolvedGiHz = clamp(baseGiHz * mult.giHzScale * (1 - giBias * 0.40), 4, baseGiHz)
   · resolvedGiSamples = clamp(baseGiSamples * mult.aoScale, 2, baseGiSamples)
   · resolvedAoScale = clamp(baseAoScale * mult.aoScale * (1 - aoBias * 0.30), 0.25, baseAoScale)
   · resolvedAoSamples = clamp(baseAoSamples * mult.aoScale, 2, baseAoSamples)
   · resolvedPost = clamp(basePost + mult.postDelta - floor(postBias * 2), 1, basePost)
   · resolvedDrawDist = clamp(baseDrawDist * mult.drawDistance * (1 - drawDistanceBias * 0.25), 40, baseDrawDist)
   · resolvedCluster = clamp(baseCluster * mult.clusterScale * (1 - clusterBias * 0.30), 8, baseCluster)
4. Writes every resolved value into the snapshot.
5. If not initial, emits applied.

_updateSnapshotFromLevel()

Internal. Reserved for symmetry. Currently a no-op that returns the snapshot.

getLevel()

Returns: the current level index.

getLevelName()

Returns: the current level string.

getSnapshot()

Returns: the snapshot.

getTargetFps()

Returns: the target FPS from the base budget.

getStats()

Returns: an object with level, levelIndex, targetLevel, lastKnob, lastReason, changes, framesUpgraded, framesDowngraded, framesHeld, framesObserved, upgradeAccumMs, downgradeAccumMs, thermalBias, batteryBias, lowPower, perfTier, and a nested snapshot object with the resolved values.

reset()

Returns: this. Resets the state and re-applies the level.

dispose()

Returns: nothing. Clears listeners.

---

Exported Functions

getDefaultQualityController()

Returns: the module-level singleton QualityController, creating it on first call.

disposeDefaultQualityController()

Returns: nothing.

getQualitySnapshot()

Returns: the current snapshot from the default controller.

Purpose: the fast-path read for downstream systems. Call once per frame. Zero allocations.

createQualityController(options = {})

Returns: a new QualityController.

---

Default Export

The default export bundles: QualityController, QualityState, QualitySnapshot, createQualityController, getDefaultQualityController, disposeDefaultQualityController, getQualitySnapshot, QUALITY_LEVEL, QUALITY_LEVEL_NAME, QUALITY_KNOB, QUALITY_KNOB_NAME, QUALITY_LEVEL_MULTIPLIERS.

---

Usage Pattern

The EngineLoop calls update() once per frame:

```
import {
  getDefaultQualityController,
  getQualitySnapshot,
} from './src/core/017_rnd_QualityConfig.js';

const quality = getDefaultQualityController();

// Per frame, inside the tick:
const snapshot = quality.update(dtMs, elapsedSec);

// Downstream systems read the snapshot:
const shadowSystem = getShadowSystem();
shadowSystem.setShadowMapSize(snapshot.shadowMapSize);
shadowSystem.setCascadeCount(snapshot.shadowCascadeCount);

const giSystem = getGISystem();
giSystem.setUpdateHz(snapshot.giUpdateBudgetHz);
giSystem.setSampleCount(snapshot.giSampleCount);
```

The App's thermal and battery guards feed external biases:

```
app.on('thermal', ({ state }) => {
  const bias = state === 'critical' ? 1.0
             : state === 'serious' ? 0.8
             : state === 'fair' ? 0.5
             : 0.0;
  quality.setThermalBias(bias);
});

app.on('battery', ({ level, charging }) => {
  const bias = charging ? 0.0 : (1 - Math.min(1, level / 0.5));
  quality.setBatteryBias(bias);
});
```

Because every subsystem reads the SAME snapshot, a single quality change keeps them visually coherent. Shadows do not downscale while GI stays at full resolution — everything moves together. This is what makes the adaptive quality invisible on Android.
