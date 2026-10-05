API Documentation — src/core/004_rnd_EngineLoop.js

File Purpose

This file is the single authoritative engine loop that fuses the Bootstrap rAF driver (001_rnd_Bootstrap.js), the Runtime frame scheduler plus job queue plus task graph plus double-buffered barrier (003_rnd_Runtime.js), and the App lifecycle surface (002_rnd_App.js) into one deterministic per-frame pipeline.

Its core responsibilities are:

1. Own the one and only requestAnimationFrame callback for the entire engine. Nothing else in the codebase may call requestAnimationFrame directly — parallel-safe scheduling on Android requires exactly one clock source, otherwise the Runtime's barrier and job queue desynchronize.
2. Run the frame in fixed sub-phases, each timed and charged against the Runtime budget table. The five phases are INPUT, ECS, LIGHTING, RENDER, PRESENT. Each phase has its own per-tier millisecond budget.
3. Enforce a per-frame wall-clock budget. When a phase overruns, the loop downgrades the next frame's phase budget (dynamic quality bias) instead of dropping frames. This keeps the anime visual style stable.
4. Implement low-power mode. rAF is gated to 30 Hz via frame-skip when the runtime EMA crosses a trigger threshold, and released when the EMA drops back below the release threshold.
5. Support render-on-demand. When no lighting state changed, when the biome is still, when the day-cycle is paused, when there is no camera motion, and when no animated emitters exist, the loop skips the render call. This is critical for Android battery life on static scenes.
6. Auto-recover from context loss, visibility changes, and thermal throttling by re-anchoring the scheduler clock so no time-warp spike occurs on resume.

The design constraint is that the loop is the ONLY place that drives rAF after installEngineLoop() runs. The Bootstrap's own rAF driver is disabled when the loop installs itself.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached device tier, used to pick default phase budgets.

PHASE

Type: frozen enum

Values:

· INPUT = 0
· ECS = 1
· LIGHTING = 2
· RENDER = 3
· PRESENT = 4
· COUNT = 5

PHASE_NAME

Type: frozen array

Values: ['input', 'ecs', 'lighting', 'render', 'present'].

FRAME_BUDGET_MS

Type: frozen object

Per-tier phase budgets in milliseconds.

On HIGH: input: 1.0, ecs: 2.0, lighting: 8.0, render: 5.0, present: 0.5, total: 16.6.

On MEDIUM: input: 1.2, ecs: 3.0, lighting: 10.0, render: 5.5, present: 0.5, total: 20.0.

On LOW: input: 1.5, ecs: 4.0, lighting: 12.0, render: 6.0, present: 0.5, total: 24.0.

MAX_DT

Type: number

Value: 0.1

Frame-time hitch cap in seconds.

LOW_POWER_HZ

Type: number

Value: 30

Target frame rate when low-power mode engages.

LOW_POWER_TRIGGER

Type: number

Value: 22.0

Frame-time EMA threshold (milliseconds) above which low-power mode auto-engages.

LOW_POWER_RELEASE

Type: number

Value: 18.0

Frame-time EMA threshold (milliseconds) below which low-power mode can be released, but only if _lowPowerSkip is zero — the release is intentionally conservative so the loop does not oscillate.

ROD_IDLE_FRAMES

Type: number

Value: 30

Number of consecutive frames without render demand after which render-on-demand forces a render anyway. Ensures external DOM changes propagate.

EMA_ALPHA_PHASE

Type: number

Value: 0.15

Smoothing factor for per-phase EMA.

EMA_ALPHA_FRAME

Type: number

Value: 0.10

Smoothing factor for the frame-time EMA.

---

Module-Level State (Not Exported Directly)

_defaultLoop

Type: EngineLoop | null

The module-level singleton, created on first getDefaultLoop() call.

---

Exported Class — EngineLoop

Constructor

```
new EngineLoop(options = {})
```

Parameters:

· targetHz — the target frame rate. Default 60.
· lowPower — if true, low-power mode starts engaged. Default false.
· lowPowerHz — target frame rate in low-power mode. Default LOW_POWER_HZ (30).
· renderOnDemand — if true, the loop skips the render call when no state changed. Default true.
· phaseBudgets — a per-tier phase budget table. If omitted, uses FRAME_BUDGET_MS.
· autoStart — if true, calls start() at the end of initialize(). Default false.

Constructor work:

1. Merges options.
2. Fetches the default Runtime via getDefaultRuntime().
3. Allocates all module state: _rafId, _running, _paused, _frame, _elapsed, _dt, _then.
4. Allocates _phaseMs, _phaseEma, _phaseBudget, _phaseOverrun — all Float64Array or Uint8Array sized to PHASE.COUNT.
5. Allocates _frameEma = 16.67.
6. Initializes low-power state.
7. Initializes render-on-demand state.
8. Allocates _renderer, _scene, _camera as null — they are set in initialize().
9. Allocates _listeners (Map).
10. Pre-binds the internal handlers to instance methods.
11. Calls _applyPhaseBudgets().

Instance Properties (Read-Only)

· running — the _running flag.
· paused — the _paused flag.
· lowPower — the _lowPower flag.
· frame — the frame counter.
· elapsed — total elapsed seconds.
· dt — the last clamped delta.
· frameEma — the frame-time EMA in milliseconds.
· phaseEma — a Float64Array of per-phase EMA milliseconds.
· phaseOverrun — a Uint8Array of per-phase overrun flags (0 or 1).
· perfTier — the cached PERF_TIER.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'initialized', 'started', 'stopped', 'paused', 'resumed', 'lowpower', 'lowpowerauto', 'phase', 'contextlost', 'contextrestored', 'disposed'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Parameters:

· event — event name.
· fn — the callback to remove.

Returns: nothing.

initialize()

Returns: this.

Purpose: one-shot initialization. Idempotent.

1. Calls this.runtime.initialize() — brings up the Runtime.
2. Captures _renderer, _scene, _camera from 008_scn_world.js.
3. Attaches WebGL context lost/restored listeners on renderer.domElement.
4. Attaches visibilitychange on document.
5. Sets _initialized = true.
6. Emits initialized.
7. If autoStart, calls start().

start()

Returns: this.

Purpose: schedules the rAF loop. If not yet initialized, calls initialize() first. Sets _running = true, resets _then to now, schedules _boundTick. Emits started. Idempotent.

stop()

Returns: this.

Purpose: cancels the rAF loop. Sets _running = false. Emits stopped.

pause()

Returns: this.

Purpose: sets _paused = true. The rAF loop keeps running but skips the frame phases. Emits paused.

resume()

Returns: this.

Purpose: clears _paused and re-anchors _then to now so the first frame after resume does not see a huge dt. Emits resumed.

setLowPower(enabled, targetHz = LOW_POWER_HZ)

Parameters:

· enabled — boolean.
· targetHz — target frame rate when engaged. Clamped to [15, 60].

Returns: this.

Purpose: manually engages or disengages low-power mode. Resets the low-power accumulators. Emits lowpower.

requestRender()

Returns: this.

Purpose: sets _renderNeeded = true. Called by external code that changes scene state outside the loop's own knowledge (e.g. debug HUD, capture tool).

dispose()

Returns: this.

Purpose: full teardown.

1. Calls stop().
2. Detaches the renderer context listeners.
3. Detaches the visibility listener.
4. Clears _listeners.
5. Sets _initialized = false.
6. Emits disposed.

getStats()

Returns: an object with frame, elapsed, dt, frameEma, phaseMs (a copy), phaseEma (a copy), phaseOverrun (a copy), lowPower, lowPowerTarget, idleFrames, perfTier. This allocates — call it from a debug HUD or on demand, not per frame.

getPhasePressure()

Returns: a new Float32Array(PHASE.COUNT) where each entry is EMA / budget, clamped to [0, 1]. Consumers (adaptive quality controllers) use this to see which phase is under the most pressure and downgrade accordingly.

Internal Instance Methods (Documented)

_applyPhaseBudgets()

Reads this.options.phaseBudgets (or FRAME_BUDGET_MS) and populates _phaseBudget.

_tick(now)

The main rAF callback. Detailed flow:

1. If not running, returns immediately.
2. Reads now — falls back to performance.now() if not a finite number.
3. Computes dt = (now - _then) * 0.001, clamps to [0, MAX_DT].
4. Updates _dt, _elapsed, _frame, and the frame EMA.
5. If paused, schedules the next frame and returns. No phases run.
6. If low-power is engaged, adds ms to the low-power accumulator. If the accumulator is below the low-power step, schedules the next frame and returns. Otherwise subtracts the step and increments _lowPowerSkip.
7. If low-power is not engaged and the frame EMA exceeds LOW_POWER_TRIGGER, engages low-power auto-mode and emits lowpowerauto.
8. Calls this.runtime.tick(now) — this drains the task graph, drains the job queue, and commits the barrier.
9. Calls _runFramePhases(dt) — the five-phase execution.
10. Emits phase with a snapshot of the phase timings.
11. Schedules the next frame via _scheduleNext().

_scheduleNext()

If running, schedules the next rAF callback.

_runFramePhases(dt)

Runs the five phases in order, timing each and updating _phaseMs, _phaseEma, and _phaseOverrun.

Phase 0 INPUT: calls _phaseInput() (a no-op reserved for App-level input consumption).

Phase 1 ECS: if isWorldReady(), calls stepWorld(dt, _elapsed).

Phase 2 LIGHTING: calls _runLightingPhase(dt, _elapsed), which currently returns 0 because the Runtime's task graph already ran inside runtime.tick(). The phase exists for measuring the residual cost and for future extension.

Phase 3 RENDER: if _shouldRender(), calls renderWorld() and resets _idleFrames. Otherwise increments _idleFrames. The phase measures the render call's cost.

Phase 4 PRESENT: calls _phasePresent(), currently a no-op because the barrier commit already happened inside runtime.tick().

_phaseInput()

No-op. Reserved for future App-level input handling.

_runLightingPhase(dt, elapsed)

Returns 0. Reserved for future per-frame lighting bookkeeping. Downstream lighting systems register their tasks in the Runtime via runtime.registerTask('lights', ...) etc. and the Runtime's task graph runs them during runtime.tick().

_phasePresent()

No-op. Reserved.

_shouldRender()

Returns: boolean.

Purpose: render-on-demand decision.

1. If renderOnDemand is false, always returns true.
2. If _renderNeeded is true, clears the flag and returns true.
3. If _idleFrames >= ROD_IDLE_FRAMES, resets the counter and returns true — forces a render so external DOM changes propagate.
4. If the Runtime has any queued jobs, returns true — jobs mean state changed.
5. If any phase overran this frame, returns true — a quality controller may have swapped resources.

Otherwise returns false.

_computeStateHash()

Returns: an integer hash of the current camera transform. Used to detect camera motion. If the hash does not change between frames and nothing else changed, the loop can safely skip the render. Not used directly by _shouldRender currently but reserved for a future camera-still optimization.

_onVisibility()

Called on document.visibilitychange. Pauses when hidden, resumes when visible.

_onContextLost(e)

Calls e.preventDefault(), stops the loop, emits contextlost. The engine waits for context restore.

_onContextRestored()

Re-anchors _then, calls start(), emits contextrestored.

---

Exported Functions

_now()

Internal helper returning the current high-resolution timestamp.

getDefaultLoop()

Returns: the module-level singleton EngineLoop, creating it on first call.

disposeDefaultLoop()

Returns: nothing.

Purpose: disposes and clears the singleton.

createEngineLoop(options = {})

Parameters: same as the constructor.

Returns: a fresh EngineLoop.

installEngineLoop(options = {})

Parameters: same as the constructor.

Returns: the running EngineLoop instance.

Purpose: the one-call convenience installer.

1. Creates a new EngineLoop.
2. Calls loop.initialize().
3. Calls Bootstrap.stop() to disable the Bootstrap's own rAF driver.
4. Calls loop.start().
5. Returns the loop.

After this call, the EngineLoop is the sole owner of the rAF clock. Any subsequent call to Bootstrap.start() would create a second rAF driver, which is a bug — the loop assumes it is the only clock source.

---

Default Export

The default export bundles: EngineLoop, createEngineLoop, getDefaultLoop, disposeDefaultLoop, installEngineLoop, PHASE, PHASE_NAME, FRAME_BUDGET_MS, JOB_STATE, TASK_STATE.

Note: JOB_STATE and TASK_STATE are re-exported from 003_rnd_Runtime.js for convenience so consumers of the loop do not need to import the runtime separately.

---

Usage Pattern

main.js boots the engine through the App layer, then hands clock ownership to the EngineLoop:

```
import { installEngineLoop } from './src/core/004_rnd_EngineLoop.js';

// After App.boot() has completed:
const loop = installEngineLoop({
  targetHz: 60,
  renderOnDemand: true,
});

loop.on('phase', (info) => {
  if (info.overrun[2]) {
    console.warn('LIGHTING phase over budget');
  }
});

loop.on('lowpowerauto', () => {
  console.info('Entered low-power mode due to slow frames');
});
```

The adaptive quality controller reads loop.getPhasePressure() once per second and adjusts resolution, shadow map size, or GI update rate based on which phase is under the most pressure.

The loop never blocks. It never throws. It never allocates on the hot path. Its only side effects are the phase callbacks, the Runtime tick, and the events it emits.
