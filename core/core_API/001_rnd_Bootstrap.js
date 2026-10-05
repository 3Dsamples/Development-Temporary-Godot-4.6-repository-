API Documentation — src/core/001_rnd_Bootstrap.js

File Purpose

This file is the single boot orchestrator for the entire engine. It owns the one and only requestAnimationFrame driver that the engine uses, drives the frame loop with hitch-capped timestep, and exposes the priority-ordered stage registry that every downstream lighting subsystem (006_lgt_LightManager through 380_lgt_lights) registers itself against.

It does five things:

1. Imports the world singleton from 008_scn_world.js and calls initializeWorld.
2. Runs the one-shot bitECS 0.4.0 audit from 009_scn_BiteCSVersionPolicy.js.
3. Owns a priority-ordered list of stage objects (each with update(dt, elapsed, info) and optional resize/dispose).
4. Drives a single rAF loop that steps every registered stage, then calls stepWorld (ECS) and renderWorld (draw).
5. Bridges browser lifecycle events (visibilitychange, WebGL context loss, window resize, orientationchange) into deterministic pause/resume/resize handling.

It also owns the rolling FPS and frame-time EMA that downstream adaptive-quality modules read.

Its design constraints are: one rAF only, no per-frame allocations on the hot path, no global state that downstream code can mutate, and clean isolation between the ECS step and the render step so a slow renderer never starves the ECS.

---

Exported Constants

MAX_DT

Type: number

Value: 0.1

Hitch cap in seconds. Any delta larger than this is clamped down to this value before it reaches damping math. This protects frame-rate-independent damping (damp(current, target, lambda, dt)) from blowing up when the browser tab stalls or the device sleeps.

FPS_EMA_ALPHA

Type: number

Value: 0.1

Smoothing factor for the rolling FPS EMA. Smaller = smoother, slower to react. 0.1 gives roughly a 10-frame half-life — responsive enough for adaptive-quality decisions but stable enough that one bad frame does not flip the controller.

FRAME_EMA_ALPHA

Type: number

Value: 0.15

Smoothing factor for the rolling frame-time EMA. Slightly more reactive than FPS because frame time is measured in ms and quality decisions read it directly.

SYNC_INTERVAL

Type: number

Value: 0.25

Throttle interval in seconds for the 4 Hz "sync" event. Any subsystem that wants to do heavy bookkeeping (light uniform flush, perceptual-color rebake) subscribes to this event and does the work once every 250 ms instead of every frame.

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH'

Cached performance tier of the current device. Read from getPerfTier() at module load. Used to pick default shadow resolutions, DPR caps, and stage budgets.

MOBILE_DPR_CAP

Type: number

Value: 1.25 | 1.75 | 2.0

Cached device pixel-ratio cap based on PERF_TIER. Any resize call clamps the requested DPR to this value. Prevents 4K phones with devicePixelRatio = 3 from rendering at full resolution and melting.

---

Module-Level State (Not Exported Directly)

_booted

Type: boolean

True after boot() has completed successfully. Guards against double-boot.

_running

Type: boolean

True while the rAF loop is scheduled. Set by start(), cleared by stop().

_paused

Type: boolean

True when the frame loop should skip update and render (visibility hidden, context lost). Distinct from _running — the rAF loop keeps running while paused so it can resume instantly.

_rafId

Type: number

The current requestAnimationFrame handle. Zeroed when not scheduled.

_then

Type: number

The timestamp of the previous frame in milliseconds. Updated at the end of every tick.

_elapsed

Type: number

Total elapsed seconds since start().

_frame

Type: number

Monotonic frame counter. Incremented every tick.

_fps, _fpsSmooth

Type: number

Instantaneous FPS (1 / dt) and its exponential moving average.

_frameTimeMs, _frameTimeSmooth

Type: number

Instantaneous frame time in milliseconds and its EMA.

_dt

Type: number

The clamped delta time for the current frame.

_syncAccum

Type: number

Accumulator for the 4 Hz sync event. Reset to zero every time the sync event fires.

_stages

Type: Array<{ name: string, stage: object, priority: number, enabled: boolean }>

The priority-ordered stage registry. Always kept sorted by priority ascending. The stage object is whatever the registrant supplied; the bootstrap only calls stage.update(dt, elapsed, info) and optionally stage.resize(w, h, dpr) and stage.dispose().

_options

Type: object

The merged boot options. Fields: width, height, pixelRatio, enableShadows, shadowResolution, fov, near, far, cameraX, cameraY, cameraZ, biome, biomeSpeed, autoCycle, cycleTime, timeOfDay, daySpeed, usePaletteLight, preloadAllLayers, sceneShadows, seed, ppu, autostart.

_listeners

Type: Map<string, Array<Function>>

The internal event bus. Events: booted, started, stopped, paused, resumed, disposed, sync, frame, resize, contextlost, contextrestored.

_runtimeInfo

Type: object

A single reusable object returned to every stage's update. Fields: booted, running, paused, dt, elapsed, frame, fps, frameTimeMs, perfTier, mobileDprCap, width, height, pixelRatio, stageCount. Updated in place every frame — never reallocated.

---

Exported Functions

registerStage(name, stage, priority = 100)

Parameters:

· name — unique string identifier for the stage.
· stage — an object that must have an update(dt, elapsed, info) method. Optional: resize(w, h, dpr), dispose().
· priority — integer; lower values run first. Default 100.

Returns: the entry object { name, stage, priority, enabled }.

Purpose: adds a stage to the priority-ordered list. Re-sorts the list. If stage.enabled === false at registration time, the entry starts disabled.

unregisterStage(name)

Parameters: name — the stage name to remove.

Returns: true if found and removed, false otherwise.

Purpose: removes a stage and calls its dispose() if present. Used for hot-reload and chunk unload.

getStage(name)

Parameters: name — the stage name.

Returns: the stage object, or null.

Purpose: look up a registered stage by name.

getStages()

Parameters: none.

Returns: the internal _stages array (do not mutate).

Purpose: expose the registry to debug tools that need to iterate stages.

setStageEnabled(name, enabled)

Parameters:

· name — the stage name.
· enabled — boolean.

Returns: true if found and updated, false otherwise.

Purpose: enable/disable a stage at runtime without unregistering it. Useful for A/B testing or quality downgrades.

on(event, fn)

Parameters:

· event — one of the supported event names.
· fn — a callback (payload) => void.

Returns: an unsubscribe function () => off(event, fn).

Purpose: subscribe to a bootstrap event.

off(event, fn)

Parameters:

· event — event name.
· fn — the callback to remove.

Returns: nothing.

Purpose: unsubscribe.

boot(options = {})

Parameters: options — same shape as _options.

Returns: the runtime info object.

Purpose: the one-shot boot sequence.

1. Merges options into _options.
2. Calls initializeWorld(_options) from 008_scn_world.js.
3. Runs runBiteCSAudit(world.components, world, 100000). Throws if the audit fails.
4. Attaches renderer listeners (context lost / restored).
5. Attaches window listeners (resize, orientationchange).
6. Attaches visibility listener.
7. Calls resize() once with the initial viewport dimensions.
8. Sets _booted = true, emits booted.
9. If _options.autostart is true, calls start().

start()

Parameters: none.

Returns: nothing.

Throws: if called before boot().

Purpose: schedules the rAF loop. Idempotent — calling twice is a no-op.

stop()

Parameters: none.

Returns: nothing.

Purpose: cancels the rAF loop. Does not dispose the world. Emits stopped.

pause()

Parameters: none.

Returns: nothing.

Purpose: sets _paused = true. The rAF loop keeps running but skips update and render. Emits paused.

resume()

Parameters: none.

Returns: nothing.

Purpose: clears _paused and resets _then to the current time so the first frame after resume does not see a huge dt. Emits resumed.

dispose()

Parameters: none.

Returns: nothing.

Purpose: full teardown.

1. Calls stop().
2. Iterates _stages in reverse priority order and calls dispose() on each.
3. Clears _stages.
4. Detaches every listener.
5. Calls disposeWorld() from 008_scn_world.js.
6. Resets all module state.

After dispose() the engine may call boot() again to bring up a fresh instance.

resize(width, height, pixelRatio)

Parameters:

· width — logical CSS pixels.
· height — logical CSS pixels.
· pixelRatio — device pixel ratio, clamped to MOBILE_DPR_CAP.

Returns: nothing.

Purpose: resizes the world renderer and calls resize() on every registered stage that has one. Updates _options with the new dimensions. Emits resize.

getRuntimeInfo()

Parameters: none.

Returns: the reusable _runtimeInfo object.

Purpose: read-only snapshot of the current runtime state. Do not mutate — the object is shared.

isBooted()

Returns: _booted.

isRunning()

Returns: _running.

isPaused()

Returns: _paused.

getFPS()

Returns: _fpsSmooth — the smoothed FPS.

getFrameTimeMs()

Returns: _frameTimeSmooth — the smoothed frame time in milliseconds.

getElapsed()

Returns: _elapsed — total elapsed seconds.

getFrame()

Returns: _frame — the frame counter.

getDeltaTime()

Returns: _dt — the clamped delta time of the last frame.

Re-exported functions from 008_scn_world.js

The bootstrap re-exports getCore, getRenderer, getScene, getCamera, getPerfTier, getMobileDprCap, world, isWorldReady so downstream modules can import everything they need from one place.

Re-exported functions from 009_scn_BiteCSVersionPolicy.js

The bootstrap re-exports isBiteCSReady and getAudit for the same reason.

---

Internal Functions (Not Exported but Documented)

_attachRendererListeners()

No parameters, no return value. Attaches webglcontextlost and webglcontextrestored listeners on the renderer's DOM canvas.

_detachRendererListeners()

No parameters, no return value. Detaches the two context listeners.

_attachWindowListeners()

No parameters, no return value. Attaches resize and orientationchange listeners on window.

_detachWindowListeners()

No parameters, no return value. Detaches the two window listeners.

_attachVisibilityListener()

No parameters, no return value. Attaches visibilitychange on document.

_detachVisibilityListener()

No parameters, no return value. Detaches the visibility listener.

_tick(now)

Parameters: now — the rAF timestamp.

Returns: nothing.

Purpose: the main frame callback.

1. Computes dt = (now - _then) / 1000.
2. Clamps dt to [0, MAX_DT].
3. Updates _elapsed, _frame, _fps, _fpsSmooth, _frameTimeMs, _frameTimeSmooth.
4. If not paused and isWorldReady(), calls _updateStages(dt, _elapsed), then stepWorld(dt, _elapsed), then renderWorld().
5. Increments _syncAccum; if it exceeds SYNC_INTERVAL, resets it and emits sync.
6. Emits frame with { dt, elapsed, frame, fps, fpsSmooth, frameTimeMs }.
7. Re-schedules itself via requestAnimationFrame if _running.

_updateStages(dt, elapsed)

Parameters:

· dt — clamped delta seconds.
· elapsed — total elapsed seconds.

Returns: nothing.

Purpose: iterates _stages and calls stage.update(dt, elapsed, runtimeInfo) on every enabled stage. Wraps each call in try/catch so one stage's error does not stop the rest. A stage that throws is disabled permanently until re-enabled.

_onResize()

No parameters. Debounced resize handler. Calls resize() with the current window dimensions and the clamped DPR.

_onContextLost(e)

Parameters: e — the WebGL context-lost event.

Purpose: calls e.preventDefault() (required by the spec), calls stop(), emits contextlost. The engine waits for the browser to fire contextrestored.

_onContextRestored()

No parameters.

Purpose: calls resize() to rebuild the renderer's viewport state, calls start(), emits contextrestored. The world core is responsible for re-uploading its GPU resources.

_onVisibilityChange()

No parameters.

Purpose: reads document.hidden. If hidden, calls pause(). Otherwise calls resume(). Emits nothing — pause/resume already emit their own events.

_getRuntimeInfo()

No parameters. Returns the _runtimeInfo object, updating every field in place.

_emit(event, payload)

Parameters:

· event — event name.
· payload — arbitrary payload.

Purpose: dispatches to every listener registered for event. Wraps each call in try/catch.

---

Default Export

The default export bundles every named export and every re-export in a single object so downstream modules can import Bootstrap from './001_rnd_Bootstrap.js' and then call Bootstrap.boot(), Bootstrap.registerStage(...), Bootstrap.getFPS(), etc.

Fields on the default export: boot, start, stop, pause, resume, dispose, resize, registerStage, unregisterStage, getStage, getStages, setStageEnabled, on, off, getRuntimeInfo, isBooted, isRunning, isPaused, getFPS, getFrameTimeMs, getElapsed, getFrame, getDeltaTime, getCore, getRenderer, getScene, getCamera, getPerfTier, getMobileDprCap, world, isWorldReady, isBiteCSReady, getAudit.

---

Usage Pattern

main.js (or debug.html) calls:

```
import Bootstrap from './src/core/001_rnd_Bootstrap.js';

Bootstrap.boot({
  width: window.innerWidth,
  height: window.innerHeight,
  pixelRatio: window.devicePixelRatio,
  enableShadows: true,
  autostart: true,
});

// Later, register a lighting subsystem:
Bootstrap.registerStage('LightManager', lightManagerInstance, 10);
Bootstrap.registerStage('ShadowSystem', shadowSystemInstance, 30);
```

The priority argument determines update order. LightManager runs before ShadowSystem so the shadow pass sees the current light list. ShadowSystem runs before GISystem so probes are baked against this frame's shadow state.

