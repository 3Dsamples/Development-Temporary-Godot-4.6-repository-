// File : 001
// name : src/core/001_rnd_Bootstrap.js
// description : Application bootstrap and single entry point for the modular
//               anime lighting stack on Android mobile. Owns the full startup
//               sequence and one canonical rAF-driven frame loop that:
//                 (1) imports and initializes the world singleton
//                     (src/core/008_scn_world.js → ProceduralWorldCore in
//                     core/world.js) which owns the sole THREE.WebGLRenderer,
//                     Scene, PerspectiveCamera, and bitECS 0.4.0 world;
//                 (2) runs the one-shot bitECS version audit
//                     (009_scn_BiteCSVersionPolicy) so no legacy 0.3.x surface
//                     can silently slip in via a cached CDN proxy;
//                 (3) registers every lighting subsystem via a priority-ordered
//                     stage list (light, shadow, GI, AO, environment, indoor,
//                     outdoor, director, post) so each module only declares
//                     update(dt, elapsed) and dispose();
//                 (4) drives the frame loop with a hitch-capped timestep
//                     (clamped to 100 ms, frame-rate independent damping),
//                     visibility pause (document.hidden), context-lost /
//                     context-restored recovery, and a rolling FPS + frame
//                     time EMA that downstream adaptive-quality modules read;
//                 (5) exposes a tiny lifecycle API (boot, start, stop, tick,
//                     dispose) plus static getters for handles so any module
//                     can reach world/renderer/scene/camera without importing
//                     the world file directly.
//               Strictly uses only Three.js r185 lights
//               (AmbientLight, HemisphereLight, DirectionalLight, PointLight,
//               SpotLight, RectAreaLight) and the r185 src/ tree; no legacy
//               light types, no external lighting libs. bitECS 0.4.0 API only
//               (createWorld / addEntity / addComponent / query / ... — no
//               Types, no defineComponent, no defineSystem). Zero per-frame
//               allocations on the hot path; fixed MAX_ENTITIES SoA arrays;
//               no dynamic resizing after boot. DPR clamped by PERF_TIER
//               (2.0 / 1.75 / 1.25). powerPreference 'high-performance'.
// best for : Single boot()/start() call site for the whole engine. Every
//            subsequent module in the manifest (006_lgt_LightManager through
//            380_lgt_lights) registers itself against this bootstrap instead
//            of instantiating its own rAF loop, so the frame pipeline stays
//            one graph, one barrier, one timestep, one FPS counter — the
//            only way to guarantee parallel-safe lighting updates on mobile.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  world,
  initializeWorld,
  disposeWorld,
  stepWorld,
  renderWorld,
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getPerfTier,
  getMobileDprCap,
  isWorldReady,
} from './008_scn_world.js';

import {
  runBiteCSAudit,
  assertBiteCSReady,
  isBiteCSReady,
  getAudit,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const MAX_DT            = 0.1;   // hitch cap (100 ms) — protects damping math
const FPS_EMA_ALPHA     = 0.1;   // rolling FPS smoothing
const FRAME_EMA_ALPHA   = 0.15;  // rolling frame-time smoothing
const SYNC_INTERVAL     = 0.25;  // 4 Hz sync throttle (light/perceptual/AO/GI)

const PERF_TIER         = getPerfTier();
const MOBILE_DPR_CAP    = getMobileDprCap();

/* ------------------------------------------------------------------ */
/* 1. BOOTSTRAP SINGLETON STATE                                       */
/* ------------------------------------------------------------------ */

let _booted        = false;
let _running       = false;
let _paused        = false;
let _rafId         = 0;
let _then          = (typeof performance !== 'undefined' ? performance.now() : Date.now());

let _elapsed       = 0;
let _frame         = 0;
let _fps           = 60;
let _fpsSmooth     = 60;
let _frameTimeMs   = 16.67;
let _frameTimeSmooth = 16.67;

let _dt            = 0;
let _syncAccum     = 1.0;

/* Priority-ordered stage list. Lower priority runs first. */
const _stages = [];

const _options = {
  width:              (typeof window !== 'undefined' ? window.innerWidth  : 480),
  height:             (typeof window !== 'undefined' ? window.innerHeight : 720),
  pixelRatio:         Math.min(
                        (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
                        MOBILE_DPR_CAP
                      ),
  enableShadows:      true,
  shadowResolution:   PERF_TIER === 'HIGH' ? 2048 : PERF_TIER === 'MEDIUM' ? 1024 : 512,
  fov:                55,
  near:               0.1,
  far:                1000,
  cameraX:            0,
  cameraY:            8,
  cameraZ:            34,
  biome:              0,
  biomeSpeed:         0.85,
  autoCycle:          false,
  cycleTime:          28,
  timeOfDay:          0.38,
  daySpeed:           0.004,
  usePaletteLight:    true,
  preloadAllLayers:   false,
  sceneShadows:       false,
  seed:               1337,
  ppu:                5.5,
  autostart:          true,
};

/* ------------------------------------------------------------------ */
/* 2. STAGE REGISTRY                                                  */
/* ------------------------------------------------------------------ */

export function registerStage(name, stage, priority = 100) {
  if (!stage || typeof stage !== 'object') {
    throw new Error(`[001_rnd_Bootstrap] registerStage("${name}"): stage must be an object`);
  }
  const entry = {
    name:     String(name || `stage_${_stages.length}`),
    stage,
    priority: priority | 0,
    enabled:  stage.enabled !== false,
  };
  _stages.push(entry);
  _stages.sort((a, b) => a.priority - b.priority);
  return entry;
}

export function unregisterStage(name) {
  for (let i = 0; i < _stages.length; i++) {
    if (_stages[i].name === name) {
      const entry = _stages[i];
      _stages.splice(i, 1);
      if (entry.stage && typeof entry.stage.dispose === 'function') {
        try { entry.stage.dispose(); } catch (_) { /* swallow teardown errors */ }
      }
      return true;
    }
  }
  return false;
}

export function getStage(name) {
  for (let i = 0; i < _stages.length; i++) {
    if (_stages[i].name === name) return _stages[i].stage;
  }
  return null;
}

export function getStages() {
  return _stages;
}

export function setStageEnabled(name, enabled) {
  for (let i = 0; i < _stages.length; i++) {
    if (_stages[i].name === name) {
      _stages[i].enabled = !!enabled;
      if (_stages[i].stage) _stages[i].stage.enabled = !!enabled;
      return true;
    }
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* 3. EVENT BUS (tiny, allocation-free dispatch)                      */
/* ------------------------------------------------------------------ */

const _listeners = new Map();

export function on(event, fn) {
  if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
  let arr = _listeners.get(event);
  if (!arr) { arr = []; _listeners.set(event, arr); }
  arr.push(fn);
  return () => off(event, fn);
}

export function off(event, fn) {
  const arr = _listeners.get(event);
  if (!arr) return;
  const i = arr.indexOf(fn);
  if (i >= 0) arr.splice(i, 1);
}

function _emit(event, payload) {
  const arr = _listeners.get(event);
  if (!arr) return;
  for (let i = 0; i < arr.length; i++) {
    try { arr[i](payload); } catch (e) { console.error(`[001_rnd_Bootstrap] listener error on "${event}"`, e); }
  }
}

/* ------------------------------------------------------------------ */
/* 4. BOOT SEQUENCE                                                   */
/* ------------------------------------------------------------------ */

export function boot(options = {}) {
  if (_booted) return _getRuntimeInfo();

  Object.assign(_options, options || {});

  // ---- 4.1 Initialize the world core (renderer/scene/camera/ECS) ----
  const core = initializeWorld(_options);
  if (!core) throw new Error('[001_rnd_Bootstrap] initializeWorld() returned null');

  // ---- 4.2 Enforce bitECS 0.4.0 policy (one-shot, throws on drift) ----
  const audit = runBiteCSAudit(
    world.components,
    world,
    100000
  );
  if (!audit.passed) {
    throw new Error(`[001_rnd_Bootstrap] bitECS audit failed: ${audit.reason}`);
  }
  assertBiteCSReady();

  // ---- 4.3 Attach renderer/scene/camera listeners ----
  _attachRendererListeners();
  _attachWindowListeners();
  _attachVisibilityListener();

  // ---- 4.4 Resize once at boot ----
  resize(_options.width, _options.height, _options.pixelRatio);

  _booted = true;
  _emit('booted', _getRuntimeInfo());

  if (_options.autostart) start();

  return _getRuntimeInfo();
}

function _attachRendererListeners() {
  const renderer = getRenderer();
  if (!renderer || !renderer.domElement) return;
  const dom = renderer.domElement;

  dom.addEventListener('webglcontextlost', _onContextLost, false);
  dom.addEventListener('webglcontextrestored', _onContextRestored, false);
}

function _detachRendererListeners() {
  const renderer = getRenderer();
  if (!renderer || !renderer.domElement) return;
  const dom = renderer.domElement;
  dom.removeEventListener('webglcontextlost', _onContextLost);
  dom.removeEventListener('webglcontextrestored', _onContextRestored);
}

function _attachWindowListeners() {
  if (typeof window === 'undefined') return;
  window.addEventListener('resize', _onResize, { passive: true });
  window.addEventListener('orientationchange', _onResize, { passive: true });
}

function _detachWindowListeners() {
  if (typeof window === 'undefined') return;
  window.removeEventListener('resize', _onResize);
  window.removeEventListener('orientationchange', _onResize);
}

function _attachVisibilityListener() {
  if (typeof document === 'undefined') return;
  document.addEventListener('visibilitychange', _onVisibilityChange, { passive: true });
}

function _detachVisibilityListener() {
  if (typeof document === 'undefined') return;
  document.removeEventListener('visibilitychange', _onVisibilityChange);
}

/* ------------------------------------------------------------------ */
/* 5. LIFECYCLE                                                       */
/* ------------------------------------------------------------------ */

export function start() {
  if (!_booted) {
    throw new Error('[001_rnd_Bootstrap] start() before boot()');
  }
  if (_running) return;
  _running = true;
  _then = (typeof performance !== 'undefined' ? performance.now() : Date.now());

  if (typeof requestAnimationFrame === 'function') {
    _rafId = requestAnimationFrame(_tick);
  }
  _emit('started', null);
}

export function stop() {
  _running = false;
  if (_rafId && typeof cancelAnimationFrame === 'function') {
    cancelAnimationFrame(_rafId);
  }
  _rafId = 0;
  _emit('stopped', null);
}

export function pause() {
  _paused = true;
  _emit('paused', null);
}

export function resume() {
  _paused = false;
  _then = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  _emit('resumed', null);
}

export function dispose() {
  stop();

  // Teardown stages in reverse priority order.
  for (let i = _stages.length - 1; i >= 0; i--) {
    const entry = _stages[i];
    if (entry.stage && typeof entry.stage.dispose === 'function') {
      try { entry.stage.dispose(); } catch (e) { console.error(`[001_rnd_Bootstrap] dispose error in stage "${entry.name}"`, e); }
    }
  }
  _stages.length = 0;

  _detachRendererListeners();
  _detachWindowListeners();
  _detachVisibilityListener();

  disposeWorld();

  _booted = false;
  _running = false;
  _paused = false;
  _rafId = 0;
  _elapsed = 0;
  _frame = 0;
  _syncAccum = 1.0;

  _emit('disposed', null);
}

/* ------------------------------------------------------------------ */
/* 6. FRAME LOOP (one rAF, hitch-capped, allocation-free hot path)    */
/* ------------------------------------------------------------------ */

function _tick(now) {
  if (!_running) return;

  const t = (typeof now === 'number' && Number.isFinite(now))
    ? now
    : (typeof performance !== 'undefined' ? performance.now() : Date.now());

  let dt = (t - _then) * 0.001;
  _then = t;

  if (dt < 0) dt = 0;
  if (dt > MAX_DT) dt = MAX_DT;

  _dt = dt;
  _elapsed += dt;
  _frame++;

  const instFps = dt > 0 ? 1 / dt : 60;
  _fps = instFps;
  _fpsSmooth += (instFps - _fpsSmooth) * FPS_EMA_ALPHA;

  const ftMs = dt * 1000;
  _frameTimeMs = ftMs;
  _frameTimeSmooth += (ftMs - _frameTimeSmooth) * FRAME_EMA_ALPHA;

  if (!_paused && isWorldReady()) {
    _updateStages(dt, _elapsed);
    stepWorld(dt, _elapsed);
    renderWorld();
  }

  _syncAccum += dt;
  if (_syncAccum >= SYNC_INTERVAL) {
    _syncAccum = 0;
    _emit('sync', null);
  }

  _emit('frame', {
    dt,
    elapsed: _elapsed,
    frame: _frame,
    fps: _fps,
    fpsSmooth: _fpsSmooth,
    frameTimeMs: _frameTimeSmooth,
  });

  if (_running && typeof requestAnimationFrame === 'function') {
    _rafId = requestAnimationFrame(_tick);
  }
}

function _updateStages(dt, elapsed) {
  for (let i = 0; i < _stages.length; i++) {
    const entry = _stages[i];
    if (!entry.enabled) continue;
    const s = entry.stage;
    if (!s || typeof s.update !== 'function') continue;
    if (s.enabled === false) continue;

    try {
      s.update(dt, elapsed, _getRuntimeInfo());
    } catch (e) {
      console.error(`[001_rnd_Bootstrap] update error in stage "${entry.name}"`, e);
      entry.enabled = false;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 7. RESIZE                                                          */
/* ------------------------------------------------------------------ */

function _onResize() {
  resize(
    (typeof window !== 'undefined' ? window.innerWidth  : _options.width),
    (typeof window !== 'undefined' ? window.innerHeight : _options.height),
    Math.min(
      (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
      MOBILE_DPR_CAP
    )
  );
}

export function resize(width, height, pixelRatio) {
  const w = Math.max(1, width  | 0);
  const h = Math.max(1, height | 0);
  const dpr = Math.min(Math.max(0.5, Number(pixelRatio) || 1), MOBILE_DPR_CAP);

  _options.width = w;
  _options.height = h;
  _options.pixelRatio = dpr;

  const core = getCore();
  if (core && typeof core.resize === 'function') {
    core.resize(w, h, dpr);
  }

  for (let i = 0; i < _stages.length; i++) {
    const entry = _stages[i];
    if (entry.stage && typeof entry.stage.resize === 'function') {
      try { entry.stage.resize(w, h, dpr); } catch (e) { console.error(`[001_rnd_Bootstrap] resize error in stage "${entry.name}"`, e); }
    }
  }

  _emit('resize', { width: w, height: h, pixelRatio: dpr });
}

/* ------------------------------------------------------------------ */
/* 8. CONTEXT LOSS / VISIBILITY                                       */
/* ------------------------------------------------------------------ */

function _onContextLost(e) {
  if (e && typeof e.preventDefault === 'function') e.preventDefault();
  stop();
  _emit('contextlost', null);
}

function _onContextRestored() {
  resize(_options.width, _options.height, _options.pixelRatio);
  start();
  _emit('contextrestored', null);
}

function _onVisibilityChange() {
  if (typeof document === 'undefined') return;
  if (document.hidden) {
    pause();
  } else {
    resume();
  }
}

/* ------------------------------------------------------------------ */
/* 9. PUBLIC ACCESSORS (allocation-free, static)                      */
/* ------------------------------------------------------------------ */

let _runtimeInfo = null;
function _getRuntimeInfo() {
  if (!_runtimeInfo) {
    _runtimeInfo = {
      booted: _booted,
      running: _running,
      paused: _paused,
      dt: _dt,
      elapsed: _elapsed,
      frame: _frame,
      fps: _fpsSmooth,
      frameTimeMs: _frameTimeSmooth,
      perfTier: PERF_TIER,
      mobileDprCap: MOBILE_DPR_CAP,
      width: _options.width,
      height: _options.height,
      pixelRatio: _options.pixelRatio,
      stageCount: _stages.length,
    };
  } else {
    _runtimeInfo.booted = _booted;
    _runtimeInfo.running = _running;
    _runtimeInfo.paused = _paused;
    _runtimeInfo.dt = _dt;
    _runtimeInfo.elapsed = _elapsed;
    _runtimeInfo.frame = _frame;
    _runtimeInfo.fps = _fpsSmooth;
    _runtimeInfo.frameTimeMs = _frameTimeSmooth;
    _runtimeInfo.width = _options.width;
    _runtimeInfo.height = _options.height;
    _runtimeInfo.pixelRatio = _options.pixelRatio;
    _runtimeInfo.stageCount = _stages.length;
  }
  return _runtimeInfo;
}

export function getRuntimeInfo()      { return _getRuntimeInfo(); }
export function isBooted()            { return _booted; }
export function isRunning()           { return _running; }
export function isPaused()            { return _paused; }
export function getFPS()              { return _fpsSmooth; }
export function getFrameTimeMs()      { return _frameTimeSmooth; }
export function getElapsed()          { return _elapsed; }
export function getFrame()            { return _frame; }
export function getDeltaTime()        { return _dt; }

export {
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getPerfTier,
  getMobileDprCap,
  world,
  isWorldReady,
  isBiteCSReady,
  getAudit,
};

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                 */
/* ------------------------------------------------------------------ */

export default {
  boot,
  start,
  stop,
  pause,
  resume,
  dispose,
  resize,
  registerStage,
  unregisterStage,
  getStage,
  getStages,
  setStageEnabled,
  on,
  off,
  getRuntimeInfo,
  isBooted,
  isRunning,
  isPaused,
  getFPS,
  getFrameTimeMs,
  getElapsed,
  getFrame,
  getDeltaTime,
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getPerfTier,
  getMobileDprCap,
  world,
  isWorldReady,
  isBiteCSReady,
  getAudit,
};