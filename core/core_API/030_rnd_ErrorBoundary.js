API Documentation — src/core/030_rnd_ErrorBoundary.js

File Purpose

This file provides runtime error containment and recovery for the anime lighting stack. Every lighting subsystem wraps its per-frame entry points, worker callbacks, and async continuations in an ErrorBoundary so a single thrown error — a NaN in a shader uniform, a missing render target, a worker desync — never crashes the frame, never leaves the engine in an inconsistent state, and never silently corrupts the visual output.

The problem it solves is specific to real-time graphics on Android. A try/catch around every call site would be too expensive on the hot path. A global window.onerror handler would catch unhandled exceptions but lose the surrounding context — which subsystem, which frame, which payload. A completely unguarded engine would crash hard the first time a shader failed to compile on a strange driver.

The ErrorBoundary approach sits between those extremes. Each subsystem registers one or more named boundaries. Each boundary owns:

1. A state machine — CLOSED, HALF_OPEN, OPEN, DISABLED, DISPOSED.
2. A per-boundary failure count, consecutive failure count, and a threshold at which the boundary trips.
3. A cooldown window, measured in frames, that blocks calls when the boundary is OPEN.
4. An exponential backoff that doubles the cooldown each time the boundary trips, up to a cap.
5. A fallback value returned when the boundary is OPEN.
6. A parent pointer so a child boundary can propagate its trip to a coarse parent.

The circuit-breaker pattern has three benefits over plain try/catch:

Predictability. Once a boundary trips, subsequent calls are cheap — the boundary short-circuits in O(1) without even running the wrapped function. This prevents a broken shader from eating 4 ms per frame trying to compile.

Recoverability. The HALF_OPEN state allows a single probe call after the cooldown expires. If the probe succeeds, the boundary returns to CLOSED and the subsystem resumes. If the probe fails, the boundary returns to OPEN with a longer cooldown.

Observability. Every trip emits an event with the boundary name, the trip count, the last error message, and the frame number. A debug HUD can list every open boundary. A regression tool can correlate a visual glitch with a boundary trip.

The module also provides a BoundaryHandle wrapper so callers can hold a stable handle that survives manager resets. The wrapper exposes run, runAsync, reset, disable, enable, dispose, and setFallback as methods on the handle so calling code is terse.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

MAX_BOUNDARIES

Type: number

Value: 256 on HIGH, 192 on MEDIUM, 128 on LOW.

The fixed capacity of the boundary slot table. Sized to accommodate every named boundary the engine registers — one per lighting subsystem, plus one per sub-pass within each subsystem, plus a handful for infrastructure (asset load, shader compile, GL context).

BOUNDARY_STATE

Type: frozen enum

The five states a boundary can be in.

· CLOSED = 0 — healthy; calls flow through.
· HALF_OPEN = 1 — cooldown expired; exactly one probe call is allowed.
· OPEN = 2 — tripped; all calls are short-circuited and the fallback value is returned.
· DISABLED = 3 — manually silenced; calls flow through, failures are counted, but the boundary never trips.
· DISPOSED = 4 — removed from the registry; no more calls.

The three-way distinction between CLOSED, HALF_OPEN, and OPEN is what makes the circuit breaker work. A single "tripped" boolean would not allow the controlled recovery that HALF_OPEN provides.

BOUNDARY_STATE_NAME

Type: frozen array

Values: ['closed', 'half_open', 'open', 'disabled', 'disposed'].

BOUNDARY_TAG

Type: frozen enum

Purpose tags used for classification and debug grouping. Each lighting subsystem uses its tag.

· GENERIC = 0
· LIGHTS = 1
· SHADOWS = 2
· GI = 3
· AO = 4
· CLUSTER = 5
· ENVIRONMENT = 6
· INTERIOR = 7
· EXTERIOR = 8
· POST = 9
· WORKER = 10
· REGISTRY = 11
· POOL = 12
· FRAME = 13
· COUNT = 14

BOUNDARY_TAG_NAME

Type: frozen array

Values: ['generic', 'lights', 'shadows', 'gi', 'ao', 'cluster', 'environment', 'interior', 'exterior', 'post', 'worker', 'registry', 'pool', 'frame'].

DEFAULT_FAILURE_THRESHOLD

Type: number

Value: 3

How many consecutive failures a boundary tolerates before it trips. Small enough to catch a chronic failure, large enough that a one-off glitch does not trip the boundary.

DEFAULT_COOLDOWN_FRAMES

Type: number

Value: 60

The initial cooldown when a boundary trips. Sixty frames at 60 FPS is one second. At 30 FPS it is two seconds.

MAX_COOLDOWN_FRAMES

Type: number

Value: 3600

The maximum cooldown. Sixty seconds at 60 FPS, or two minutes at 30 FPS. Prevents the backoff from growing unbounded.

BACKOFF_MULTIPLIER

Type: number

Value: 2

Every trip doubles the cooldown. After the first trip, 60 frames. After the second, 120. After the third, 240. And so on, up to 3600.

---

Module-Level State (Not Exported Directly)

_boundaryIdCounter

Type: number

Monotonic counter for boundary ids.

_defaultManager

Type: ErrorBoundaryManager | null

The module-level singleton, created on first getDefaultErrorBoundaries() call.

---

Internal Helper Functions (Documented)

_nextBoundaryId()

Returns: the next monotonic boundary id.

_now()

Returns: the current high-resolution timestamp via performance.now(), or Date.now() as a fallback.

_describeError(e)

Parameters: e — the caught error.

Returns: a string description.

Purpose: normalizes an error to a string. Handles null, undefined, strings, and objects with a message field. If none of those applies, uses String(e) inside a try/catch so a throwing toString cannot itself throw.

_extractStack(e)

Parameters: e — the caught error.

Returns: the error's stack string, or null.

Purpose: extracts the stack trace when available. Some Android browsers do not populate stack on non-Error throws, so the helper returns null and the boundary still records the message.

---

Exported Class — BoundarySlot

One instance per registered boundary.

Constructor

```
new BoundarySlot(index)
```

Parameters: index — the boundary's slot index.

Instance Properties

· index — the slot index.
· id — the boundary's monotonic id.
· name — the boundary's diagnostic name.
· tag — one of BOUNDARY_TAG.
· channel — the logger channel to use for warnings.
· state — one of BOUNDARY_STATE.
· failureThreshold — the number of consecutive failures before the boundary trips.
· cooldownFrames — the base cooldown in frames.
· maxCooldownFrames — the maximum cooldown.
· backoffMultiplier — the multiplier applied on each trip.
· silent — 1 to suppress logging, 0 otherwise.
· propagateToParent — 1 to propagate trips to the parent boundary, 0 otherwise.
· parentIndex — the parent boundary's slot index, or -1.
· failures — the total failure count.
· consecutive — the consecutive failure count.
· successes — the total success count.
· trips — the total number of times the boundary has tripped.
· recoveries — the total number of successful HALF_OPEN probes.
· shortCircuits — the number of times a call was rejected because the boundary was OPEN.
· cooldownStartFrame — the frame when the boundary last tripped.
· cooldownFramesLeft — the remaining cooldown frames.
· currentCooldown — the current exponential-backoff value.
· lastError — the message string of the last error.
· lastStack — the stack trace of the last error, or null.
· lastErrorFrame — the frame of the last error.
· lastErrorMs — the timestamp of the last error.
· fallback — the fallback value returned when the boundary is OPEN.
· hasFallback — 1 if a fallback has been set, 0 otherwise.
· lastCallMs — the duration of the last call in milliseconds.
· totalCallMs — the total duration of all calls in milliseconds.
· callCount — the total number of calls.

Instance Methods

reset()

Returns: nothing.

Purpose: zeroes every field and resets the state to CLOSED. Called when a boundary slot is recycled.

---

Exported Class — ErrorBoundaryManager

The main manager.

Constructor

```
new ErrorBoundaryManager(options = {})
```

Parameters:

· defaultFailureThreshold — the default failure threshold for new boundaries. Default DEFAULT_FAILURE_THRESHOLD.
· defaultCooldownFrames — the default cooldown. Default DEFAULT_COOLDOWN_FRAMES.
· maxCooldownFrames — the maximum cooldown. Default MAX_COOLDOWN_FRAMES.
· autoRecover — if true, boundaries automatically transition to HALF_OPEN when their cooldown expires. Default true.
· logFailures — if true, failures are logged via the logger. Default true.
· recordProfilerMark — if true, a failure marks the profiler. Default true.
· silentInLowTier — reserved. Default true.

Constructor work:

1. Stores options.
2. Allocates slots — an array of MAX_BOUNDARIES entries.
3. Initializes count = 0 and byId (Map).
4. Allocates the free list ring.
5. Initializes frame, globalTrips, globalErrors.
6. Allocates _listeners (Map).

Instance Properties

· options — the merged options.
· capacity — the slot capacity.
· slots — the boundary slot array.
· count — the number of registered boundaries.
· byId — the Map from boundary id to slot index.
· frame — the frame counter.
· globalTrips — the total number of trips across all boundaries.
· globalErrors — the total number of failures across all boundaries.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'registered', 'failure', 'tripped', 'recovered', 'reset', 'disabled', 'enabled', 'disposed'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to listeners.

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

Purpose: sets the frame counter, then ages every OPEN boundary's cooldown. When a cooldown reaches zero, transitions the boundary to HALF_OPEN if autoRecover is true.

register(name, options = {})

Parameters:

· name — the boundary's diagnostic name.
· options.tag — one of BOUNDARY_TAG.
· options.channel — the logger channel.
· options.failureThreshold — override the default threshold.
· options.cooldownFrames — override the default cooldown.
· options.maxCooldownFrames — override the max cooldown.
· options.backoffMultiplier — override the backoff multiplier.
· options.silent — suppress logging.
· options.propagateToParent — whether to propagate trips to the parent.
· options.disabled — start in DISABLED state.
· options.fallback — the fallback value.
· options.parentId — the id of the parent boundary.

Returns: the BoundarySlot, or null if the registry is full or the name is invalid.

Purpose: registers a new boundary. Emits registered.

_findIndexById(id)

Internal. Looks up a slot index from a boundary id.

getById(id)

Returns: the BoundarySlot, or null.

getByName(name)

Returns: the BoundarySlot, or null.

run(slot, fn, ctx)

Parameters:

· slot — the BoundarySlot to run under.
· fn — the function to run.
· ctx — an optional context object.

Returns: whatever fn returned, or the boundary's fallback value on failure.

Purpose: the synchronous protected call. The hot path.

Flow:

1. If the slot is null, returns undefined.
2. Fast path: if the state is CLOSED or HALF_OPEN, times the call, runs fn in try/catch:
   · On success, updates timings. If the state was HALF_OPEN, calls _recover(slot). Otherwise increments successes and resets consecutive.
   · On failure, times the call, calls _recordFailure(slot, e), and returns the fallback value.
3. If the state is OPEN, increments shortCircuits. If propagateToParent is on and the parent is CLOSED, calls _recordFailure on the parent with a synthetic error. Returns the fallback value.
4. If the state is DISABLED, runs fn in try/catch. Counts failures but never trips. Returns the fallback value on failure.
5. If the state is DISPOSED, returns the fallback value.

runAsync(slot, fn, ctx)

Parameters: same as run.

Returns: a Promise that never rejects.

Purpose: the async protected call. Wraps a Promise-returning function.

Flow:

1. If the state is OPEN or DISPOSED, returns a resolved Promise with the fallback.
2. Times the call, runs fn in try/catch.
3. If fn throws synchronously, records the failure and returns a resolved Promise with the fallback.
4. If fn returns a non-Promise, treats it as a synchronous result.
5. If fn returns a Promise, attaches .then(onSuccess, onFailure):
   · On success, updates timings and either recovers from HALF_OPEN or increments successes.
   · On failure, records the failure and resolves with the fallback.

The returned Promise never rejects, so the caller can await without a try/catch.

_recordFailure(slot, e)

Internal. Records a failure.

Flow:

1. Increments failures, consecutive, and the global error counter.
2. Stores the error message, stack, frame, and timestamp.
3. If logFailures is on and the boundary is not silent, logs a warning with the boundary name, failure count, and threshold.
4. Emits failure with the error and counts.
5. If recordProfilerMark is on, records a profiler marker.
6. If the state is not DISABLED and consecutive >= failureThreshold, calls _trip(slot).

_trip(slot)

Internal. Trips the boundary.

Flow:

1. Sets the state to OPEN.
2. Increments trips and globalTrips.
3. Computes the next cooldown by multiplying the current cooldown by backoffMultiplier and clamping to maxCooldownFrames.
4. Sets cooldownStartFrame = frame and cooldownFramesLeft = next.
5. If logging, logs an error with the boundary name, last error, cooldown, and trip count.
6. Emits tripped.

_recover(slot)

Internal. Recovers a boundary from HALF_OPEN.

Flow:

1. Sets the state to CLOSED.
2. Increments recoveries.
3. Resets consecutive to zero.
4. Resets currentCooldown to cooldownFrames (clears the backoff).
5. If logging, logs an info message.
6. Emits recovered.

reset(slot)

Parameters: slot — the boundary to reset.

Returns: boolean.

Purpose: forces the boundary to CLOSED, clears the consecutive counter, and resets the cooldown.

disable(slot)

Parameters: slot — the boundary to disable.

Returns: boolean.

Purpose: sets the boundary to DISABLED. The boundary still runs its function and still counts failures, but never trips.

enable(slot)

Parameters: slot — the boundary to enable.

Returns: boolean.

Purpose: sets the boundary back to CLOSED.

disposeBoundary(slot)

Parameters: slot — the boundary to dispose.

Returns: boolean.

Purpose: sets the boundary to DISPOSED. Subsequent calls return the fallback without running fn.

anyOpen()

Returns: boolean. True if any boundary is OPEN.

countOpen()

Returns: the number of OPEN boundaries.

listOpen()

Returns: an array of stats objects for every OPEN boundary. Each has id, name, tag, trips, cooldownFramesLeft, and lastError.

Purpose: the debug HUD's primary view of failing subsystems.

getStats()

Returns: an object with frame, boundaryCount, capacity, openCount, globalTrips, globalErrors, and a boundaries array with per-boundary stats.

reset()

Returns: this. Resets every boundary and zeroes every counter.

dispose()

Returns: this. Resets and nulls the internal arrays.

---

Exported Class — BoundaryHandle

A lightweight wrapper around a BoundarySlot so callers can hold a stable handle that survives manager resets.

Constructor

```
new BoundaryHandle(manager, slot)
```

Instance Properties (Getters)

· name — the boundary name.
· state — the current BOUNDARY_STATE.
· isOpen — true if the boundary is OPEN.

Instance Methods

run(fn, ctx)

Delegates to manager.run(this.slot, fn, ctx).

runAsync(fn, ctx)

Delegates to manager.runAsync(this.slot, fn, ctx).

reset()

Delegates to manager.reset(this.slot).

disable()

Delegates to manager.disable(this.slot).

enable()

Delegates to manager.enable(this.slot).

dispose()

Delegates to manager.disposeBoundary(this.slot).

setFallback(value)

Parameters: value — the fallback value.

Returns: this.

Purpose: sets the fallback value returned when the boundary is OPEN.

---

Exported Functions

getDefaultErrorBoundaries()

Returns: the module-level singleton ErrorBoundaryManager, creating it on first call.

disposeDefaultErrorBoundaries()

Returns: nothing.

boundariesBeginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

Purpose: convenience wrapper that calls beginFrame on the default manager.

createBoundary(name, options)

Parameters: same as ErrorBoundaryManager.register.

Returns: a BoundaryHandle, or null.

Purpose: convenience wrapper that registers a boundary on the default manager and returns a handle.

trySafe(fn, fallback)

Parameters:

· fn — the function to run.
· fallback — the value to return on failure.

Returns: the function's result on success, or the fallback on failure.

Purpose: the simplest possible error containment. Runs fn() in try/catch. Logs failures via the logger but does not use a boundary. Used for one-off calls that do not warrant a dedicated boundary.

---

Exported Factory

createErrorBoundaryManager(options = {})

Returns: a new ErrorBoundaryManager.

---

Default Export

The default export bundles: ErrorBoundaryManager, BoundarySlot, BoundaryHandle, createErrorBoundaryManager, getDefaultErrorBoundaries, disposeDefaultErrorBoundaries, boundariesBeginFrame, createBoundary, trySafe, BOUNDARY_STATE, BOUNDARY_STATE_NAME, BOUNDARY_TAG, BOUNDARY_TAG_NAME, MAX_BOUNDARIES, DEFAULT_FAILURE_THRESHOLD, DEFAULT_COOLDOWN_FRAMES, MAX_COOLDOWN_FRAMES, BACKOFF_MULTIPLIER.

---

Usage Pattern

A subsystem that registers a boundary and runs its work under it:

```
import { createBoundary, BOUNDARY_TAG } from './src/core/030_rnd_ErrorBoundary.js';

const shadowBoundary = createBoundary('shadowSystem.update', {
  tag: BOUNDARY_TAG.SHADOWS,
  failureThreshold: 5,
  cooldownFrames: 90,
});

// Per frame:
shadowBoundary.run(() => {
  shadowSystem.update(dt, elapsed);
});
```

A subsystem that runs async work under a boundary:

```
const giBakeBoundary = createBoundary('giSystem.bakeAsync', {
  tag: BOUNDARY_TAG.GI,
});

// Fire and forget:
giBakeBoundary.runAsync(async () => {
  const result = await bakeGIProbes();
  commitGIResult(result);
});
```

A subsystem that wants a fallback value when its boundary is OPEN:

```
const envBoundary = createBoundary('envPalette.solve', {
  tag: BOUNDARY_TAG.ENVIRONMENT,
});
envBoundary.setFallback(lastKnownPalette);

const palette = envBoundary.run(() => solvePalette(biome));
// palette is either the fresh result or the last known palette.
```

A debug HUD that lists every open boundary:

```
import { getDefaultErrorBoundaries } from './src/core/030_rnd_ErrorBoundary.js';

const mgr = getDefaultErrorBoundaries();
const open = mgr.listOpen();
for (const info of open) {
  console.warn(`Boundary ${info.name} is OPEN (${info.trips} trips): ${info.lastError}`);
}
```

The App layer's context-lost handler that resets every boundary when the context is restored:

```
app.on('contextrestored', () => {
  const mgr = getDefaultErrorBoundaries();
  for (const slot of mgr.slots) {
    if (slot && slot.state !== BOUNDARY_STATE.DISPOSED) {
      mgr.reset(slot);
    }
  }
});
```

The ErrorBoundary pattern is what keeps the engine running when a single subsystem fails. On Android, where a shader compilation can fail on an unusual driver, where a texture format might not be renderable, where a worker might return a malformed result, the boundary converts a fatal error into a contained one. The lighting subsystem falls back to its last known good state, the engine keeps rendering, and the developer sees a specific, named, frame-tagged warning in the log rather than a black screen.
