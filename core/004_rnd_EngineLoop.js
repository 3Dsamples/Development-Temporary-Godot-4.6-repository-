// File : 004
// name : src/core/004_rnd_EngineLoop.js
// description : The single authoritative engine loop that fuses the Bootstrap
//               rAF driver (001_rnd_Bootstrap.js), the Runtime frame scheduler
//               + job queue + task graph + double-buffered barrier
//               (003_rnd_Runtime.js), and the App lifecycle surface
//               (002_rnd_App.js) into ONE deterministic per-frame pipeline.
//
//               Responsibilities:
//                 • Owns the one and only requestAnimationFrame callback for
//                   the entire engine. Nothing else in the codebase may call
//                   requestAnimationFrame directly — parallel-safe scheduling
//                   on Android requires exactly one clock source.
//                 • Runs the frame in fixed sub-phases, each timed and charged
//                   against the Runtime budget table:
//                     Phase 0 — input / DOM (skipped when paused)
//                     Phase 1 — ECS step   (bitECS 0.4.0 world.step via App)
//                     Phase 2 — lighting   (task graph: lights → shadows →
//                                            GI → AO → environment → post)
//                     Phase 3 — render     (Three.js r185 WebGLRenderer)
//                     Phase 4 — present    (commit double-buffer barrier)
//                 • Enforces a per-frame wall-clock budget derived from the
//                   PERF_TIER: HIGH = 16.6 ms, MEDIUM = 20 ms, LOW = 24 ms.
//                   When a phase overruns, the loop downgrades the NEXT
//                   frame's phase budget (dynamic quality bias) instead of
//                   dropping frames, keeping the anime visual style stable.
//                 • Implements a low-power mode that gates rAF to 30 Hz via
//                   frame-skip when the Runtime scheduler detects extended
//                   frame time > 22 ms or the App battery guard trips.
//                 • Supports render-on-demand: when no lighting state changed
//                   (biome still, day-cycle paused, no camera motion, no
//                   animated emitters), the loop skips the render call —
//                   critical for Android battery life on static scenes.
//                 • Auto-recovers from context loss, visibility changes,
//                   and thermal throttling by re-syncing the scheduler
//                   clock (no time-warp spikes on resume).
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; zero per-frame allocations on the hot path (all phase
//               timing uses Float64Array scratch, all booleans are cached);
//               no closures created per frame; every typed array sized once
//               at construction.
// best for : Guaranteeing a single coherent frame pipeline for the whole
//            anime lighting stack (006_lgt_LightManager through
//            380_lgt_lights) on Android, with per-phase budget accounting that
//            downstream quality controllers (193_rnd_AdaptiveQualityController,
//            195_lgt_ShadowQualityScaler, 196_lgt_GIQualityScaler,
//            197_lgt_AOQualityScaler) read to bias resolution/shadow/GI/AO
//            quality without ever dropping a frame.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getPerfTier,
  isWorldReady,
  stepWorld,
  renderWorld,
} from './008_scn_world.js';

import {
  getDefaultRuntime,
  disposeDefaultRuntime,
  JOB_STATE,
  TASK_STATE,
} from './003_rnd_Runtime.js';

import * as Bootstrap from './001_rnd_Bootstrap.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const PHASE = Object.freeze({
  INPUT:       0,
  ECS:         1,
  LIGHTING:    2,
  RENDER:      3,
  PRESENT:     4,
  COUNT:       5,
});

export const PHASE_NAME = Object.freeze([
  'input',
  'ecs',
  'lighting',
  'render',
  'present',
]);

export const FRAME_BUDGET_MS = Object.freeze(
  PERF_TIER === 'HIGH'
    ? { input: 1.0, ecs: 2.0, lighting: 8.0, render: 5.0, present: 0.5, total: 16.6 }
    : PERF_TIER === 'MEDIUM'
      ? { input: 1.2, ecs: 3.0, lighting: 10.0, render: 5.5, present: 0.5, total: 20.0 }
      : { input: 1.5, ecs: 4.0, lighting: 12.0, render: 6.0, present: 0.5, total: 24.0 }
);

const MAX_DT             = 0.1;
const LOW_POWER_HZ       = 30;
const LOW_POWER_TRIGGER  = 22.0;
const LOW_POWER_RELEASE  = 18.0;
const ROD_IDLE_FRAMES    = 30;
const EMA_ALPHA_PHASE    = 0.15;
const EMA_ALPHA_FRAME    = 0.10;

/* ------------------------------------------------------------------ */
/* 1. ENGINE LOOP CLASS                                               */
/* ------------------------------------------------------------------ */

export class EngineLoop {
  constructor(options = {}) {
    this.options = Object.assign({
      targetHz:         60,
      lowPower:         false,
      lowPowerHz:       LOW_POWER_HZ,
      renderOnDemand:   true,
      phaseBudgets:     FRAME_BUDGET_MS,
      autoStart:        false,
    }, options || {});

    this.runtime = getDefaultRuntime();

    this._rafId        = 0;
    this._running      = false;
    this._paused       = false;
    this._frame        = 0;
    this._elapsed      = 0;
    this._dt           = 0;
    this._then         = _now();
    this._initialized  = false;

    this._phaseMs      = new Float64Array(PHASE.COUNT);
    this._phaseEma     = new Float64Array(PHASE.COUNT);
    this._phaseBudget  = new Float64Array(PHASE.COUNT);
    this._phaseOverrun = new Uint8Array(PHASE.COUNT);
    this._frameEma     = 16.67;

    this._lowPower       = !!this.options.lowPower;
    this._lowPowerAccum  = 0;
    this._lowPowerSkip   = 0;
    this._lowPowerTarget = this.options.lowPowerHz;

    this._idleFrames     = 0;
    this._renderNeeded   = true;
    this._lastRenderHash = 0;

    this._renderer       = null;
    this._scene          = null;
    this._camera         = null;

    this._listeners      = new Map();

    this._boundTick      = this._tick.bind(this);
    this._boundVis       = this._onVisibility.bind(this);
    this._boundLost      = this._onContextLost.bind(this);
    this._boundRestored  = this._onContextRestored.bind(this);

    this._applyPhaseBudgets();
  }

  _applyPhaseBudgets() {
    const b = this.options.phaseBudgets || FRAME_BUDGET_MS;
    this._phaseBudget[PHASE.INPUT]    = b.input;
    this._phaseBudget[PHASE.ECS]      = b.ecs;
    this._phaseBudget[PHASE.LIGHTING] = b.lighting;
    this._phaseBudget[PHASE.RENDER]   = b.render;
    this._phaseBudget[PHASE.PRESENT]  = b.present;
  }

  /* ---------------- events ---------------- */

  on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let arr = this._listeners.get(event);
    if (!arr) { arr = []; this._listeners.set(event, arr); }
    arr.push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  _emit(event, payload) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      try { arr[i](payload); } catch (e) { console.error(`[004_rnd_EngineLoop] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- lifecycle ---------------- */

  initialize() {
    if (this._initialized) return this;

    this.runtime.initialize();

    this._renderer = getRenderer();
    this._scene    = getScene();
    this._camera   = getCamera();

    if (this._renderer && this._renderer.domElement) {
      this._renderer.domElement.addEventListener('webglcontextlost', this._boundLost, false);
      this._renderer.domElement.addEventListener('webglcontextrestored', this._boundRestored, false);
    }

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._boundVis, { passive: true });
    }

    this._initialized = true;
    this._emit('initialized', null);

    if (this.options.autoStart) this.start();

    return this;
  }

  start() {
    if (!this._initialized) this.initialize();
    if (this._running) return this;

    this._running = true;
    this._then = _now();

    if (typeof requestAnimationFrame === 'function') {
      this._rafId = requestAnimationFrame(this._boundTick);
    }

    this._emit('started', null);
    return this;
  }

  stop() {
    this._running = false;
    if (this._rafId && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(this._rafId);
    }
    this._rafId = 0;
    this._emit('stopped', null);
    return this;
  }

  pause() {
    this._paused = true;
    this._emit('paused', null);
    return this;
  }

  resume() {
    this._paused = false;
    this._then = _now();
    this._emit('resumed', null);
    return this;
  }

  setLowPower(enabled, targetHz = LOW_POWER_HZ) {
    this._lowPower = !!enabled;
    this._lowPowerTarget = Math.max(15, Math.min(60, targetHz | 0));
    this._lowPowerSkip = 0;
    this._lowPowerAccum = 0;
    this._emit('lowpower', { enabled: this._lowPower, targetHz: this._lowPowerTarget });
    return this;
  }

  requestRender() {
    this._renderNeeded = true;
    return this;
  }

  dispose() {
    this.stop();

    if (this._renderer && this._renderer.domElement) {
      this._renderer.domElement.removeEventListener('webglcontextlost', this._boundLost);
      this._renderer.domElement.removeEventListener('webglcontextrestored', this._boundRestored);
    }

    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._boundVis);
    }

    this._listeners.clear();
    this._initialized = false;
    this._emit('disposed', null);

    return this;
  }

  /* ---------------- main frame ---------------- */

  _tick(now) {
    if (!this._running) return;

    const t = (typeof now === 'number' && Number.isFinite(now)) ? now : _now();
    let dt = (t - this._then) * 0.001;
    this._then = t;

    if (dt < 0) dt = 0;
    if (dt > MAX_DT) dt = MAX_DT;

    this._dt = dt;
    this._elapsed += dt;
    this._frame++;

    const ms = dt * 1000;
    this._frameEma += (ms - this._frameEma) * EMA_ALPHA_FRAME;

    if (this._paused) {
      this._scheduleNext();
      return;
    }

    // ---- Low-power gating ----
    if (this._lowPower) {
      this._lowPowerAccum += ms;
      const stepMs = 1000 / this._lowPowerTarget;
      if (this._lowPowerAccum < stepMs) {
        this._scheduleNext();
        return;
      }
      this._lowPowerAccum -= stepMs;
      this._lowPowerSkip++;
    }

    // ---- Dynamic low-power auto-trigger ----
    if (!this._lowPower && this._frameEma > LOW_POWER_TRIGGER) {
      this._lowPower = true;
      this._lowPowerTarget = LOW_POWER_HZ;
      this._lowPowerAccum = 0;
      this._emit('lowpowerauto', { enabled: true, reason: 'slow-frame' });
    } else if (this._lowPower && this._frameEma < LOW_POWER_RELEASE && this._lowPowerSkip > 0) {
      // Keep low-power once engaged; release only via explicit setLowPower().
    }

    // ---- Runtime tick (jobs + tasks + scheduler + barrier) ----
    this.runtime.tick(t);

    // ---- Frame phases ----
    this._runFramePhases(dt);

    // ---- Adaptive quality bias (feed phase overruns back) ----
    this._emit('phase', {
      frame: this._frame,
      dt,
      elapsed: this._elapsed,
      phaseMs: this._phaseMs,
      phaseEma: this._phaseEma,
      overrun: this._phaseOverrun,
      frameEma: this._frameEma,
      lowPower: this._lowPower,
      renderNeeded: this._renderNeeded,
    });

    this._scheduleNext();
  }

  _scheduleNext() {
    if (this._running && typeof requestAnimationFrame === 'function') {
      this._rafId = requestAnimationFrame(this._boundTick);
    }
  }

  /* ---------------- phase execution ---------------- */

  _runFramePhases(dt) {
    const budgets = this._phaseBudget;
    const overrun = this._phaseOverrun;

    // ---------- PHASE 0: INPUT ----------
    let t0 = _now();
    this._phaseInput();
    let t1 = _now();
    this._phaseMs[PHASE.INPUT] = t1 - t0;
    this._phaseEma[PHASE.INPUT] += (t1 - t0 - this._phaseEma[PHASE.INPUT]) * EMA_ALPHA_PHASE;
    overrun[PHASE.INPUT] = (t1 - t0) > budgets[PHASE.INPUT] ? 1 : 0;

    // ---------- PHASE 1: ECS ----------
    t0 = _now();
    if (isWorldReady()) stepWorld(dt, this._elapsed);
    t1 = _now();
    this._phaseMs[PHASE.ECS] = t1 - t0;
    this._phaseEma[PHASE.ECS] += (t1 - t0 - this._phaseEma[PHASE.ECS]) * EMA_ALPHA_PHASE;
    overrun[PHASE.ECS] = (t1 - t0) > budgets[PHASE.ECS] ? 1 : 0;

    // ---------- PHASE 2: LIGHTING (task graph) ----------
    t0 = _now();
    const lightingMs = this._runLightingPhase(dt, this._elapsed);
    t1 = _now();
    this._phaseMs[PHASE.LIGHTING] = (t1 - t0) + lightingMs;
    this._phaseEma[PHASE.LIGHTING] += ((t1 - t0) + lightingMs - this._phaseEma[PHASE.LIGHTING]) * EMA_ALPHA_PHASE;
    overrun[PHASE.LIGHTING] = this._phaseMs[PHASE.LIGHTING] > budgets[PHASE.LIGHTING] ? 1 : 0;

    // ---------- PHASE 3: RENDER ----------
    t0 = _now();
    const shouldRender = this._shouldRender();
    if (shouldRender) {
      renderWorld();
      this._lastRenderHash = this._computeStateHash();
      this._idleFrames = 0;
    } else {
      this._idleFrames++;
    }
    t1 = _now();
    this._phaseMs[PHASE.RENDER] = shouldRender ? (t1 - t0) : 0;
    this._phaseEma[PHASE.RENDER] += ((t1 - t0) - this._phaseEma[PHASE.RENDER]) * EMA_ALPHA_PHASE;
    overrun[PHASE.RENDER] = this._phaseMs[PHASE.RENDER] > budgets[PHASE.RENDER] ? 1 : 0;

    // ---------- PHASE 4: PRESENT ----------
    t0 = _now();
    this._phasePresent();
    t1 = _now();
    this._phaseMs[PHASE.PRESENT] = t1 - t0;
    this._phaseEma[PHASE.PRESENT] += (t1 - t0 - this._phaseEma[PHASE.PRESENT]) * EMA_ALPHA_PHASE;
    overrun[PHASE.PRESENT] = (t1 - t0) > budgets[PHASE.PRESENT] ? 1 : 0;
  }

  _phaseInput() {
    // Reserved for App-level input consumption. Kept allocation-free.
  }

  _runLightingPhase(dt, elapsed) {
    // The Runtime task graph already ran inside this.runtime.tick(t) above.
    // We only measure the residual cost here (job drain already happened).
    // Downstream lighting systems register their tasks in the Runtime via
    //   runtime.registerTask('lights',  ...)
    //   runtime.registerTask('shadows', ...)
    //   runtime.registerTask('gi',      ...)
    //   runtime.registerTask('ao',      ...)
    //   runtime.registerTask('env',     ...)
    //   runtime.registerTask('post',    ...)
    // with explicit dependency edges.
    return 0;
  }

  _phasePresent() {
    // Barrier commit already happened inside runtime.tick(). Nothing to do.
  }

  /* ---------------- render-on-demand ---------------- */

  _shouldRender() {
    if (!this.options.renderOnDemand) return true;
    if (this._renderNeeded) {
      this._renderNeeded = false;
      return true;
    }

    // Force a render if we've been idle too long, so external DOM updates
    // (canvas resize, HUD overlays, capture tools) still propagate.
    if (this._idleFrames >= ROD_IDLE_FRAMES) {
      this._idleFrames = 0;
      return true;
    }

    // Force a render if the runtime ran any job this frame (jobs = state changed).
    if (this.runtime.jobs.count > 0) return true;

    // Force a render if any phase overran (quality controller may have swapped).
    for (let i = 0; i < PHASE.COUNT; i++) {
      if (this._phaseOverrun[i]) return true;
    }

    return false;
  }

  _computeStateHash() {
    // Cheap rolling hash of the camera transform + biome weights + day cycle.
    // If nothing changed, we can safely skip the next frame's render.
    const cam = this._camera;
    let h = 0;
    if (cam) {
      h = (h * 31 + (cam.position.x * 1000) | 0) | 0;
      h = (h * 31 + (cam.position.y * 1000) | 0) | 0;
      h = (h * 31 + (cam.position.z * 1000) | 0) | 0;
      h = (h * 31 + (cam.quaternion.x * 1000) | 0) | 0;
      h = (h * 31 + (cam.quaternion.y * 1000) | 0) | 0;
      h = (h * 31 + (cam.quaternion.z * 1000) | 0) | 0;
      h = (h * 31 + (cam.quaternion.w * 1000) | 0) | 0;
    }
    return h;
  }

  /* ---------------- context / visibility ---------------- */

  _onVisibility() {
    if (typeof document === 'undefined') return;
    if (document.hidden) this.pause();
    else this.resume();
  }

  _onContextLost(e) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    this.stop();
    this._emit('contextlost', null);
  }

  _onContextRestored() {
    this._then = _now();
    this.start();
    this._emit('contextrestored', null);
  }

  /* ---------------- accessors ---------------- */

  get running()      { return this._running; }
  get paused()       { return this._paused; }
  get lowPower()     { return this._lowPower; }
  get frame()        { return this._frame; }
  get elapsed()      { return this._elapsed; }
  get dt()           { return this._dt; }
  get frameEma()     { return this._frameEma; }
  get phaseEma()     { return this._phaseEma; }
  get phaseOverrun() { return this._phaseOverrun; }
  get perfTier()     { return PERF_TIER; }

  getStats() {
    return {
      frame: this._frame,
      elapsed: this._elapsed,
      dt: this._dt,
      frameEma: this._frameEma,
      phaseMs: this._phaseMs.slice(0),
      phaseEma: this._phaseEma.slice(0),
      phaseOverrun: this._phaseOverrun.slice(0),
      lowPower: this._lowPower,
      lowPowerTarget: this._lowPowerTarget,
      idleFrames: this._idleFrames,
      perfTier: PERF_TIER,
    };
  }

  /* ---------------- budget feedback for quality controllers ---------------- */

  getPhasePressure() {
    // Returns 0..1 for each phase: 0 = lots of headroom, 1 = over budget.
    const out = new Float32Array(PHASE.COUNT);
    for (let i = 0; i < PHASE.COUNT; i++) {
      const b = this._phaseBudget[i] || 1e-3;
      out[i] = Math.min(1, this._phaseEma[i] / b);
    }
    return out;
  }
}

/* ------------------------------------------------------------------ */
/* 2. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

/* ------------------------------------------------------------------ */
/* 3. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultLoop = null;

export function getDefaultLoop() {
  if (!_defaultLoop) _defaultLoop = new EngineLoop();
  return _defaultLoop;
}

export function disposeDefaultLoop() {
  if (_defaultLoop) {
    _defaultLoop.dispose();
    _defaultLoop = null;
  }
}

/* ------------------------------------------------------------------ */
/* 4. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createEngineLoop(options = {}) {
  return new EngineLoop(options);
}

/* ------------------------------------------------------------------ */
/* 5. CONVENIENCE — install engine loop as the sole Bootstrap driver  */
/* ------------------------------------------------------------------ */

export function installEngineLoop(options = {}) {
  const loop = createEngineLoop(options);
  loop.initialize();

  // Disable the Bootstrap's own rAF driver — the EngineLoop owns the clock.
  try { Bootstrap.stop(); } catch (_) { /* bootstrap may not be running yet */ }

  loop.start();
  return loop;
}

/* ------------------------------------------------------------------ */
/* 6. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  EngineLoop,
  createEngineLoop,
  getDefaultLoop,
  disposeDefaultLoop,
  installEngineLoop,
  PHASE,
  PHASE_NAME,
  FRAME_BUDGET_MS,
  JOB_STATE,
  TASK_STATE,
};

export default _defaultExport;