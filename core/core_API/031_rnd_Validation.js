API Documentation — src/core/031_rnd_Validation.js

File Purpose

This file provides the runtime validation primitives for the anime lighting stack. Where 030_rnd_ErrorBoundary.js contains the call, this module guards the data. Every uniform, every material, every light descriptor, every biome weight, every color, every matrix, every probe is validated before it reaches the GPU or a worker.

The problem it solves is specific to numerical graphics code. A NaN that sneaks into a shadow bias or a GI probe is invisible to try/catch. It does not throw. The shader compiles fine. The renderer accepts the uniform. But the output is silently wrong — the shadow disappears, the GI leaks, the biome palette fails to match the reference images. By the time the visual artifact is visible, the original bad value is long gone from the call stack.

This module catches those values before they happen. It provides:

1. Scalar guards — finiteness, range, positivity, power-of-two, integral, non-zero.
2. Vector guards — Vector2/3/4 finiteness, length range, normalized, orthogonal, unit quaternions.
3. Color guards — RGB and RGBA in [0, 1], sRGB versus linear, luminance sanity.
4. Matrix guards — Matrix3 and Matrix4 finiteness, invertibility, orthonormality, non-degeneracy.
5. Light guards — intensity, color, distance, decay, angle, penumbra, all clamped to legal physical ranges.
6. Shadow guards — power-of-two map size, bias in range, cascade count, softness.
7. GI guards — probe spacing, lattice resolution, sample count, update rate.
8. AO guards — radius, sample count, resolution scale, intensity.
9. Biome guards — weights in [0, 1] and normalized to 1.0.
10. Uniform guards — every uniform the shader expects exists and matches its JavaScript type.
11. Material guards — color, normal map presence, side, blending, depth-write consistency.
12. Registry guards — the resource handle is live.
13. Snapshot validator — full engine-state sanity pass for the debug HUD or QA.

The design has three parts:

· Rules — a frozen object of pure functions, each returning true or false. Rules are pre-allocated so callers can pass RULES.positiveFinite without creating a closure.
· Validator — a singleton that owns the rules, a failure log, and the mode. Provides check(rule, value, label) for one-off checks and createContext(label) for batched checks.
· ValidationContext — a batching helper that runs many checks and produces one log entry per pass.

Every check has zero allocations on the success path. Detailed diagnostics only happen on failure via check calling the logger. The hot path is a single function call and a comparison.

The module also integrates with the error boundary and profiler: a failed check can optionally trip a boundary, and every failure records a profiler marker so a hitch or visual artifact can be correlated with the validation failure that caused it.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_FAILURE_RECORDS

Type: number

Value: 128 on HIGH, 64 on MEDIUM, 32 on LOW.

The maximum number of failure records retained by the validator's failure log. Older failures are overwritten when the log wraps.

VALIDATION_MODE

Type: frozen enum

The four validation modes.

· STRICT = 0 — a failed check throws.
· WARN = 1 — a failed check logs a warning.
· SILENT = 2 — a failed check is recorded but produces no log output.
· DISABLED = 3 — checks are no-ops.

STRICT is intended for CI and QA runs. WARN is the default for development. SILENT is used in production builds where the failure log is still collected for a debug HUD but the console is quiet. DISABLED is used when every last microsecond matters and validation has been temporarily turned off.

EPSILON_DEFAULT

Type: number

Value: 1e-6

The default numeric epsilon for approximate comparisons.

---

Module-Level State (Not Exported Directly)

_defaultValidator

Type: Validator | null

The module-level singleton, created on first getDefaultValidator() call.

---

Internal Helper Functions (Documented)

_now()

Returns: the current high-resolution timestamp.

_isFinite(v)

Parameters: v — any value.

Returns: true if v is a number and Number.isFinite(v).

_isPowerOfTwo(v)

Parameters: v — any value.

Returns: true if v is a finite positive number that satisfies (v & (v - 1)) === 0.

_isInt(v)

Parameters: v — any value.

Returns: true if v is a finite number with no fractional part.

_isPositiveInt(v)

Parameters: v — any value.

Returns: true if _isInt(v) and v > 0.

---

The RULES Object

Type: frozen object of pure functions.

Every rule takes exactly one value and returns true or false. Rules are grouped by domain.

Scalar rules

· finite(v) — v is a finite number.
· finitePositive(v) — finite and strictly positive.
· finiteNonNegative(v) — finite and non-negative.
· finiteNegative(v) — finite and strictly negative.
· unitScalar(v) — finite and in [0, 1].
· positiveInt(v) — positive integer.
· nonNegativeInt(v) — non-negative integer.
· powerOfTwo(v) — positive power of two.
· nonZero(v) — finite and not zero.
· finiteOrDefault(v) — finite, or null, or undefined.

Vector rules

· finiteVector2(v) — a THREE.Vector2 with finite x and y.
· finiteVector3(v) — a THREE.Vector3 with finite x, y, z.
· finiteVector4(v) — a THREE.Vector4 with finite x, y, z, w.
· finiteQuaternion(v) — a THREE.Quaternion with finite x, y, z, w.
· unitQuaternion(v) — finite quaternion with length within 2 % of 1.

Matrix rules

· finiteMatrix3(m) — a THREE.Matrix3 with all nine elements finite.
· finiteMatrix4(m) — a THREE.Matrix4 with all sixteen elements finite.

Color rules

· unitColor(c) — RGB in [0, 1].
· unitColorWithAlpha(c) — RGBA in [0, 1].

Light rules

· lightIntensity(v) — finite, [0, 1000].
· lightDistance(v) — finite, [0, 10000].
· lightDecay(v) — finite, [0, 8].
· lightAngle(v) — finite, (0, π).
· lightPenumbra(v) — finite, [0, 1].

Shadow rules

· shadowBias(v) — finite, [-0.1, 0.1].
· shadowNormalBias(v) — finite, [0, 1].
· shadowMapSize(v) — power of two, [128, 8192].
· shadowCascades(v) — integer, [1, 8].
· shadowSoftness(v) — finite, [0, 1].

GI rules

· giProbeSpacing(v) — finite, (0, 64].
· giLatticeResolution(v) — positive integer <= 256.
· giSampleCount(v) — positive integer <= 128.
· giUpdateHz(v) — finite, (0, 240].

AO rules

· aoRadius(v) — finite, (0, 32].
· aoSampleCount(v) — positive integer <= 128.
· aoResolutionScale(v) — finite, (0, 1].
· aoIntensity(v) — finite, [0, 8].

Biome rules

· biomeWeight(v) — finite, [0, 1].
· biomeWeightsNormalized(v) — array-like, all weights finite and non-negative, sum within 1 % of 1.

Registry rule

· liveResourceHandle(h) — positive integer that resolves to a live resource in the default registry.

Miscellaneous rules

· isFunction(fn) — typeof fn === 'function'.
· isObject(o) — non-null object.

Closure-producing rules

These return a new rule. They are for cases that need a bound parameter. Each call creates a closure, so use them off the hot path (during config or registration, not per frame).

· inRange(min, max) — returns a rule that checks v ∈ [min, max].
· multipleOf(n) — returns a rule that checks v is a multiple of n within epsilon.
· approximately(target, eps) — returns a rule that checks |v - target| <= eps.
· atLeast(min) — returns a rule that checks v >= min.
· atMost(max) — returns a rule that checks v <= max.

---

Exported Class — ValidationResult

A minimal result object used by ValidationContext. Not exposed directly to callers.

Constructor

```
new ValidationResult()
```

Instance Properties

· ok — true if every check passed.
· failures — the number of failed checks.
· firstLabel — the label of the first failure.
· firstValue — the value of the first failure.
· firstRule — the rule that rejected the first failure.

Instance Methods

reset()

Returns: this. Zeroes every field.

---

Exported Class — ValidationContext

Batches many checks and produces one log entry per pass.

Constructor

```
new ValidationContext(label, options = {})
```

Parameters:

· label — the context's diagnostic label.
· options.channel — the logger channel. Default LOG_CHANNEL.CORE.
· options.mode — one of VALIDATION_MODE. Default WARN.
· options.maxRecords — the maximum failure records to retain. Default MAX_FAILURE_RECORDS.

Instance Properties

· label — the context's label.
· channel — the logger channel.
· mode — the validation mode.
· maxRecords — the failure record cap.
· ok — true if every check passed.
· checkedCount — the total checks attempted.
· failedCount — the total checks failed.
· failures — an array of { label, value, rule } records.
· failureHead — the write pointer into the failure array.

Instance Methods

check(rule, value, label)

Parameters:

· rule — a rule function.
· value — the value to check.
· label — a diagnostic label.

Returns: boolean — true if the check passed.

Purpose: runs the rule. On failure, increments the counters, records the failure, and returns false. Wrapped in try/catch so a throwing rule is treated as a failure.

checkAll(rule, values, labelPrefix)

Parameters:

· rule — a rule function.
· values — an array-like collection.
· labelPrefix — the prefix to append the index to.

Returns: boolean — true if every value passed.

finalize()

Returns: boolean — true if every check passed.

Purpose: completes the batch.

Flow:

1. Records the duration.
2. If everything passed, returns true.
3. If the mode is not SILENT or DISABLED, logs a warning with the label, counts, and first failure.
4. If the mode is STRICT, throws.
5. If the mode is not DISABLED, records a profiler marker.
6. Returns false.

getStats()

Returns: an object with label, ok, checked, failed, durationMs, firstFailure.

reset()

Returns: this. Zeroes every counter and clears the failure array.

---

Exported Class — FailureLog

A ring buffer of recent failures across the entire validator.

Constructor

```
new FailureLog(capacity)
```

Parameters: capacity — the ring size.

Instance Properties

· capacity — the ring size.
· label — an array of labels.
· value — an array of values.
· ruleName — an array of rule names (usually null).
· timeMs — a Float64Array of timestamps.
· frame — a Uint32Array of frame numbers.
· head — the ring write pointer.
· count — the number of valid entries.
· total — the total failures ever recorded.

Instance Methods

record(label, value, ruleName, frame)

Returns: nothing. Writes to the ring and advances the head.

clear()

Returns: nothing. Zeroes every entry.

---

Exported Class — Validator

The main validator.

Constructor

```
new Validator(options = {})
```

Parameters:

· mode — one of VALIDATION_MODE. Default WARN.
· channel — the logger channel. Default LOG_CHANNEL.CORE.
· logFailures — if true, log failures. Default true.
· recordProfilerMark — if true, record a profiler marker on failure. Default true.
· tripBoundary — if true, failures trip a boundary. Default false.
· boundaryName — the name of the boundary to trip. Default 'validation.core'.

Constructor work:

1. Stores options.
2. Allocates failures — a FailureLog(MAX_FAILURE_RECORDS).
3. Initializes totalChecks and totalFailures.
4. If tripBoundary is on, attempts to create a boundary.
5. Allocates _listeners (Map).

Instance Properties

· options — the merged options.
· frame — the current frame number.
· failures — the failure log.
· totalChecks — the total checks across the validator's lifetime.
· totalFailures — the total failures across the validator's lifetime.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'failure'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to listeners.

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

check(rule, value, label, ruleName)

Parameters:

· rule — a rule function.
· value — the value.
· label — a diagnostic label.
· ruleName — an optional rule name.

Returns: boolean — true if the check passed.

Purpose: the main check routine.

Flow:

1. Increments totalChecks.
2. Runs the rule in try/catch. A throwing rule is treated as a failure.
3. If the check passed, returns true.
4. On failure, increments totalFailures and records the failure in the log.
5. If logging is on and the mode is not DISABLED, logs a warning via the logger.
6. If profiler marking is on, records a profiler marker.
7. If a boundary is attached, runs the boundary with a throwing function.
8. Emits failure.
9. If the mode is STRICT, throws.
10. Returns false.

checkOrThrow(rule, value, label, ruleName)

Alias for check. Exists for readability in call sites that intend to always throw on failure (combined with STRICT mode).

createContext(label, options)

Parameters:

· label — the context's label.
· options — overrides for channel, mode.

Returns: a new ValidationContext.

Purpose: batches many checks into a single log entry.

checkUniform(uniform, rule, label)

Parameters:

· uniform — a { value } uniform object.
· rule — a rule function.
· label — the label.

Returns: boolean.

Purpose: convenience for checking a uniform's value. Verifies the uniform has a .value field before running the rule.

checkColorUniform(uniform, label)

Parameters: same shape, but uses RULES.unitColor.

Returns: boolean.

checkVector3Uniform(uniform, label)

Parameters: same shape, but uses RULES.finiteVector3.

Returns: boolean.

checkMatrix4Uniform(uniform, label)

Parameters: same shape, but uses RULES.finiteMatrix4.

Returns: boolean.

validateLightDescriptor(descriptor, label)

Parameters:

· descriptor — a light descriptor with fields intensity, color, and optionally distance, decay, angle, penumbra.
· label — the label prefix.

Returns: boolean — true if every field passes.

Purpose: the canonical light check. Every lighting subsystem that creates a light calls this once at registration time.

validateShadowDescriptor(descriptor, label)

Parameters:

· descriptor — a shadow descriptor with fields mapSize, cascadeCount, bias, normalBias, and optionally softness.
· label — the label prefix.

Returns: boolean.

validateBiomeWeights(weights, label)

Parameters:

· weights — an array-like of biome weights.
· label — the label.

Returns: boolean.

validateMaterial(material, label)

Parameters:

· material — a THREE.Material.
· label — the label.

Returns: boolean.

Purpose: scans the material's uniforms and flags any numeric uniform that is not finite. Catches NaN in uniforms that the renderer would otherwise silently accept.

validateResourceHandle(handle, label)

Parameters:

· handle — a resource handle from 014_rnd_ResourceRegistry.js.
· label — the label.

Returns: boolean.

validateLightingSnapshot(snapshot, label)

Parameters:

· snapshot — the stats snapshot from 025_rnd_StatsCollector.js.
· label — the label.

Returns: boolean.

Purpose: validates the aggregate engine state. Checks that light counts, shadow map sizes, GI probe counts, and quality resolution scales are all within legal ranges.

getStats()

Returns: an object with frame, mode, totalChecks, totalFailures, failureLogSize, failureTotal.

getRecentFailures(max, out)

Parameters:

· max — the maximum number of recent failures.
· out — an array to receive the records.

Returns: the number of records copied.

reset()

Returns: this. Clears the failure log and zeroes the counters.

dispose()

Returns: nothing. Resets and clears listeners.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton.

· validationBeginFrame(frameNumber) — sets the frame on the default validator.
· validate(rule, value, label, ruleName) — calls check.
· validateOrThrow(rule, value, label, ruleName) — runs a check in STRICT mode regardless of the default mode.
· createValidationContext(label, options) — creates a context on the default validator.

---

Exported Functions

getDefaultValidator()

Returns: the module-level singleton Validator, creating it on first call.

disposeDefaultValidator()

Returns: nothing.

createValidator(options = {})

Returns: a new Validator.

---

Default Export

The default export bundles: Validator, ValidationContext, ValidationResult, FailureLog, RULES, VALIDATION_MODE, createValidator, getDefaultValidator, disposeDefaultValidator, validationBeginFrame, validate, validateOrThrow, createValidationContext, MAX_FAILURE_RECORDS, EPSILON_DEFAULT.

---

Usage Pattern

A subsystem that validates its uniforms once at registration:

```
import {
  getDefaultValidator,
  RULES,
} from './src/core/031_rnd_Validation.js';

const validator = getDefaultValidator();

validator.checkUniform(material.uniforms.uShadowBias, RULES.shadowBias, 'shadow.bias');
validator.checkUniform(material.uniforms.uMapSize, RULES.shadowMapSize, 'shadow.mapSize');
validator.checkColorUniform(material.uniforms.uAmbientColor, 'shadow.ambient');
```

A subsystem that validates a batch of values and produces one log entry:

```
const ctx = validator.createContext('lightManager.init');
ctx.check(RULES.lightIntensity, sun.intensity, 'sun.intensity');
ctx.check(RULES.unitColor, sun.color, 'sun.color');
ctx.check(RULES.lightIntensity, moon.intensity, 'moon.intensity');
ctx.check(RULES.unitColor, moon.color, 'moon.color');
ctx.check(RULES.shadowMapSize, shadow.mapSize, 'shadow.mapSize');
ctx.check(RULES.shadowCascades, shadow.cascadeCount, 'shadow.cascadeCount');
if (!ctx.finalize()) {
  // One warning, one line, all failures summarized.
}
```

A subsystem that validates a runtime value before uploading:

```
import { validate, RULES } from './src/core/031_rnd_Validation.js';

const hz = qualitySnapshot.giUpdateBudgetHz;
if (!validate(RULES.giUpdateHz, hz, 'gi.budgetHz')) {
  // The validator logged the failure; fall back to the last known good value.
  hz = lastGoodHz;
}
```

A subsystem that validates the full engine state once per second for QA:

```
setInterval(() => {
  const snap = statsSnapshot();
  validator.validateLightingSnapshot(snap, 'qa');
}, 1000);
```

A debug HUD that lists recent failures:

```
const failures = [];
const n = validator.getRecentFailures(10, failures);
for (let i = 0; i < n; i++) {
  console.warn(`${failures[i].label} = ${failures[i].value}`);
}
```

The validation module is what closes the gap between "the code runs" and "the output is correct." On Android, where numerical bugs in the lighting pipeline produce subtle visual glitches rather than crashes, catching an invalid value at the boundary between the JavaScript and the GPU is the only way to guarantee that the anime look matches the reference images on every device.
