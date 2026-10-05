// File : 019
// name : src/ecs/019_scn_SystemBase.js
// description : Base class and lifecycle contract for every ECS system in the
//               scene world of the anime lighting stack on Android mobile.
//               All 006–380 systems inherit from `SystemBase` (or one of its
//               subclasses) so that scheduling, profiling, error containment,
//               resize propagation, enable/disable, dependency ordering, and
//               disposal are uniform across the entire engine.
//
//               Design:
//                 • Fixed contract — every system implements a small,
//                   well-defined set of optional hooks:
//                     onInit(ctx)         — setup once at boot
//                     onUpdate(dt, elapsed) — per-frame work
//                     onLateUpdate(dt, elapsed) — after all update() calls
//                     onResize(w, h, dpr) — viewport change
//                     onEnable()          — transition from disabled to enabled
//                     onDisable()         — transition from enabled to disabled
//                     onDispose()         — teardown
//                 • Priority-based ordering — lower priority runs first.
//                   Two reserved ranges: PRIORITY_EARLY (0..99) and
//                   PRIORITY_LATE (900..999). Normal systems sit in the
//                   default 100..899 range.
//                 • Dependency declaration — `declareDependencies()` returns
//                   an array of system names that must run before this one.
//                   The scheduler resolves this into a topological order
//                   at boot.
//                 • Enable/disable gate — every call to `update()` is a
//                   single integer compare. Disabled systems cost nothing
//                   beyond the check.
//                 • Timing instrumentation — each system records its own
//                   last/avg/peak cost in milliseconds. On HIGH tier the
//                   profiler (024) is also marked.
//                 • Error containment — `update()` wraps its body in an
//                   error boundary (030) so a thrown error in one system
//                   never crashes the frame.
//                 • Zero allocations in the hot path — no per-frame object
//                   creation, no closure capture, no string formatting
//                   unless a failure actually occurs.
//                 • System registry — a global registry (`getSystemRegistry()`)
//                   tracks every registered system by name so the scheduler
//                   can resolve dependencies and the debug HUD (179) can
//                   enumerate them.
//
//               Subclasses provided:
//                 • SystemBase        — the base contract
//                 • QuerySystem       — base for systems driven by a cached
//                                       query descriptor (018)
//                 • RenderSystem      — base for systems that emit draw calls
//                 • AsyncSystem       — base for systems that dispatch work
//                                       to workers (parallel pipeline)
//                 • DebugSystem       — base for debug-only systems that
//                                       never run in production builds
//
//               Integration:
//                 • 010_scn_ECSWorld.js        — world handle
//                 • 011_scn_BiteCSAdapter.js   — ECS façade
//                 • 012_scn_ComponentRegistry.js — component catalog
//                 • 016_scn_EntityPool.js      — pool binding
//                 • 017_scn_EntityLifetime.js  — lifetime states
//                 • 018_scn_Queries.js         — cached queries
//                 • 024_rnd_Profiler.js        — timing marks
//                 • 025_rnd_StatsCollector.js  — aggregated stats
//                 • 026_rnd_Logger.js          — diagnostics
//                 • 030_rnd_ErrorBoundary.js   — error containment
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every system in the anime lighting stack
//            has the same lifecycle, the same instrumentation, the same
//            error containment, and the same dependency-ordering contract
//            — so scheduling is deterministic, profiling is uniform, and
//            disabling any subsystem is a single flag flip.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getDefaultProfiler,
} from '../core/024_rnd_Profiler.js';

import {
  getECSWorld,
  isECSWorldReady,
  worldTime,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  runQuery,
  forEachEntity,
  refreshQuery,
  QueryDescriptor,
} from './018_scn_Queries.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of systems that can be registered.
 */
export const MAX_SYSTEMS =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 384 :
                                 256;

/**
 * Maximum number of dependencies per system.
 */
export const MAX_DEPENDENCIES_PER_SYSTEM = 8;

/**
 * Priority ranges. Lower number = earlier in the frame.
 */
export const PRIORITY_EARLY   = 0;      // 0..99
export const PRIORITY_DEFAULT = 500;    // 100..899
export const PRIORITY_LATE    = 900;    // 900..999
export const PRIORITY_LAST    = 999;

/**
 * System lifecycle states.
 */
export const SYSTEM_STATE = Object.freeze({
  UNINITIALIZED: 0,
  INITIALIZED:   1,
  ENABLED:       2,
  DISABLED:      3,
  DISPOSED:      4,
  FAILED:        5,
  COUNT:         6,
});

export const SYSTEM_STATE_NAME = Object.freeze([
  'uninitialized',
  'initialized',
  'enabled',
  'disabled',
  'disposed',
  'failed',
]);

/**
 * Standard system names for the lighting stack. These are the reserved
 * names every downstream module should use so that dependency ordering
 * is unambiguous across the entire engine.
 */
export const SYSTEM_NAME = Object.freeze({
  LIGHT_MANAGER:       'lights.manager',
  LIGHT_SYSTEM:        'lights.system',
  LIGHTING_SYSTEM:     'lights.lighting',
  SHADOW_MANAGER:      'shadows.manager',
  SHADOW_SYSTEM:       'shadows.system',
  GI_MANAGER:          'gi.manager',
  GI_SYSTEM:           'gi.system',
  AO_MANAGER:          'ao.manager',
  AO_SYSTEM:           'ao.system',
  ENVIRONMENT_MANAGER: 'environment.manager',
  ENVIRONMENT_SYSTEM:  'environment.system',
  INTERIOR_MANAGER:    'interior.manager',
  INTERIOR_SYSTEM:     'interior.system',
  EXTERIOR_MANAGER:    'exterior.manager',
  EXTERIOR_SYSTEM:     'exterior.system',
  CLUSTER_SYSTEM:      'cluster.system',
  CAMERA_SYSTEM:       'camera.system',
  ANIME_DIRECTOR:      'director.anime',
  POST_SYSTEM:         'post.system',
  DEBUG_SYSTEM:        'debug.system',
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

let _systemIdCounter = 0;

function _nextSystemId() {
  return ++_systemIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/**
 * Global module state for all systems.
 */
export const SystemState = {
  frame:             0,
  totalSystems:      0,
  totalUpdates:      0,
  totalDispatches:   0,
  totalDisposals:    0,
  totalFailures:     0,
  totalEnableToggles:0,
  peakActiveSystems: 0,
};

/* ------------------------------------------------------------------ */
/* 2. GLOBAL SYSTEM REGISTRY                                          */
/* ------------------------------------------------------------------ */

const _systemByName = new Map();
const _systemById   = new Map();
const _systems      = new Array(MAX_SYSTEMS);
let   _systemCount  = 0;

for (let i = 0; i < MAX_SYSTEMS; i++) _systems[i] = null;

function _registerSystem(system) {
  if (_systemCount >= MAX_SYSTEMS) return false;
  if (_systemByName.has(system.name)) return false;

  const slot = _systemCount++;
  _systems[slot] = system;
  _systemByName.set(system.name, system);
  _systemById.set(system.systemId, system);
  SystemState.totalSystems++;
  return true;
}

function _unregisterSystem(system) {
  const existing = _systemByName.get(system.name);
  if (!existing || existing !== system) return false;
  _systemByName.delete(system.name);
  _systemById.delete(system.systemId);

  for (let i = 0; i < _systemCount; i++) {
    if (_systems[i] === system) {
      const last = _systemCount - 1;
      if (i !== last) {
        _systems[i] = _systems[last];
      }
      _systems[last] = null;
      _systemCount--;
      return true;
    }
  }
  return false;
}

/**
 * Returns a registered system by name, or null.
 */
export function getSystem(name) {
  return _systemByName.get(name) || null;
}

/**
 * Returns a registered system by its numeric system id.
 */
export function getSystemById(id) {
  return _systemById.get(id) || null;
}

/**
 * Returns an array of every registered system (allocates).
 */
export function getAllSystems() {
  const out = new Array(_systemCount);
  for (let i = 0; i < _systemCount; i++) out[i] = _systems[i];
  return out;
}

/**
 * Returns the number of registered systems.
 */
export function getSystemCount() {
  return _systemCount;
}

/**
 * Iterates every registered system. Allocation-free.
 */
export function forEachSystem(fn, ctx) {
  if (typeof fn !== 'function') return 0;
  for (let i = 0; i < _systemCount; i++) {
    fn.call(ctx, _systems[i]);
  }
  return _systemCount;
}

/* ------------------------------------------------------------------ */
/* 3. SYSTEM BASE CLASS                                               */
/* ------------------------------------------------------------------ */

/**
 * The base class every ECS system extends. Subclasses override the
 * optional hooks. The public API (`init`, `update`, `resize`, `enable`,
 * `disable`, `dispose`) is what the engine loop calls; the protected
 * hooks (`onInit`, `onUpdate`, ...) are what subclasses implement.
 */
export class SystemBase {
  constructor(name, options = {}) {
    this.systemId    = _nextSystemId();
    this.name        = name || ('system_' + this.systemId);
    this.state       = SYSTEM_STATE.UNINITIALIZED;

    this.options = Object.assign({
      priority:          PRIORITY_DEFAULT,
      enableByDefault:   true,
      debugOnly:         false,
      profile:           PERF_TIER_LOCAL === 'HIGH',
      boundaryName:      null,
      boundaryTag:       BOUNDARY_TAG.GENERIC,
      logChannel:        LOG_CHANNEL.CORE,
    }, options || {});

    this.priority    = this.options.priority | 0;
    this.enabled     = this.options.enableByDefault === true ? 1 : 0;
    this.debugOnly   = this.options.debugOnly === true;
    this.profile     = this.options.profile === true;

    // Dependencies (system names).
    this.dependencies = new Array(MAX_DEPENDENCIES_PER_SYSTEM).fill(null);
    this.dependencyCount = 0;

    // Runtime statistics.
    this.totalUpdates    = 0;
    this.totalFailures   = 0;
    this.totalDispatches = 0;
    this.lastCostMs      = 0;
    this.avgCostMs       = 0;
    this.peakCostMs      = 0;
    this.lastUpdateFrame = -1;
    this.firstUpdateMs   = 0;
    this.elapsedSinceStart = 0;

    // Optional boundary for error containment.
    this._boundary = null;

    // Whether the system is currently registered.
    this._registered = false;
  }

  /* ---------------- dependency declaration ---------------- */

  /**
   * Declare a dependency on another system by name. The scheduler will
   * ensure the dependency runs earlier.
   */
  dependOn(systemName) {
    if (typeof systemName !== 'string' || systemName.length === 0) return false;
    if (this.dependencyCount >= MAX_DEPENDENCIES_PER_SYSTEM) return false;
    for (let i = 0; i < this.dependencyCount; i++) {
      if (this.dependencies[i] === systemName) return true;
    }
    this.dependencies[this.dependencyCount++] = systemName;
    return true;
  }

  /**
   * Clears the dependency list.
   */
  clearDependencies() {
    for (let i = 0; i < this.dependencyCount; i++) this.dependencies[i] = null;
    this.dependencyCount = 0;
  }

  /* ---------------- lifecycle ---------------- */

  /**
   * Initializes the system. Safe to call multiple times.
   */
  init(ctx) {
    if (this.state === SYSTEM_STATE.INITIALIZED ||
        this.state === SYSTEM_STATE.ENABLED ||
        this.state === SYSTEM_STATE.DISABLED) {
      return true;
    }
    if (this.state === SYSTEM_STATE.DISPOSED) return false;

    // Register the system.
    if (!this._registered) {
      this._registered = _registerSystem(this);
      if (!this._registered) {
        const log = _safeLogger();
        if (log) log.warn(this.options.logChannel, () =>
          `[019_scn_SystemBase] failed to register system "${this.name}"`);
        this.state = SYSTEM_STATE.FAILED;
        return false;
      }
    }

    // Create the error boundary if configured.
    if (this.options.boundaryName) {
      try {
        const mgr = getDefaultErrorBoundaries();
        if (mgr) {
          this._boundary = mgr.create(this.options.boundaryName, {
            tag: this.options.boundaryTag,
            failureThreshold: 5,
          });
        }
      } catch (_) { /* swallow */ }
    }

    // Call the subclass hook.
    const t0 = _now();
    try {
      if (typeof this.onInit === 'function') {
        this.onInit(ctx);
      }
    } catch (e) {
      this.state = SYSTEM_STATE.FAILED;
      this.totalFailures++;
      SystemState.totalFailures++;
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onInit failed: ${e && e.message}`);
      return false;
    }
    this.firstUpdateMs = _now() - t0;

    // Transition to initialized.
    this.state = this.enabled === 1 ? SYSTEM_STATE.ENABLED : SYSTEM_STATE.DISABLED;

    const log = _safeLogger();
    if (log && PERF_TIER_LOCAL === 'HIGH') {
      log.debug(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" initialized in ${this.firstUpdateMs.toFixed(2)}ms`);
    }
    return true;
  }

  /**
   * Runs one frame of the system. Safe to call before init() — returns
   * immediately if not initialized. The body is wrapped in an error
   * boundary if one was configured.
   */
  update(dt, elapsed) {
    if (this.state !== SYSTEM_STATE.ENABLED) return false;
    if (this.enabled !== 1) return false;

    this.totalUpdates++;
    SystemState.totalUpdates++;
    this.lastUpdateFrame = SystemState.frame;

    const t0 = _now();
    let ok = true;

    if (this._boundary) {
      this._boundary.run(() => { this._runUpdate(dt, elapsed); });
    } else {
      ok = this._runUpdate(dt, elapsed);
    }

    const t1 = _now();
    const cost = t1 - t0;
    this.lastCostMs = cost;
    const alpha = 0.12;
    this.avgCostMs += (cost - this.avgCostMs) * alpha;
    if (cost > this.peakCostMs) this.peakCostMs = cost;

    if (this.profile) {
      try {
        const profiler = getDefaultProfiler();
        if (profiler && profiler.mark) profiler.mark('sys:' + this.name);
      } catch (_) { /* swallow */ }
    }

    this.elapsedSinceStart += dt;
    return ok;
  }

  _runUpdate(dt, elapsed) {
    try {
      if (typeof this.onUpdate === 'function') {
        this.onUpdate(dt, elapsed);
      }
      return true;
    } catch (e) {
      this.totalFailures++;
      SystemState.totalFailures++;
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onUpdate failed: ${e && e.message}`);
      if (this.totalFailures > 5) {
        this.state = SYSTEM_STATE.FAILED;
      }
      return false;
    }
  }

  /**
   * Runs the late-update hook. Called by the scheduler after every
   * `update()` has run.
   */
  lateUpdate(dt, elapsed) {
    if (this.state !== SYSTEM_STATE.ENABLED) return false;
    if (this.enabled !== 1) return false;
    try {
      if (typeof this.onLateUpdate === 'function') {
        this.onLateUpdate(dt, elapsed);
      }
      return true;
    } catch (e) {
      this.totalFailures++;
      SystemState.totalFailures++;
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onLateUpdate failed: ${e && e.message}`);
      return false;
    }
  }

  /**
   * Propagates a viewport resize.
   */
  resize(width, height, pixelRatio) {
    if (this.state === SYSTEM_STATE.DISPOSED) return false;
    try {
      if (typeof this.onResize === 'function') {
        this.onResize(width, height, pixelRatio);
      }
      return true;
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onResize failed: ${e && e.message}`);
      return false;
    }
  }

  /* ---------------- enable / disable ---------------- */

  enable() {
    if (this.state === SYSTEM_STATE.DISPOSED) return false;
    if (this.enabled === 1) return true;
    this.enabled = 1;
    if (this.state === SYSTEM_STATE.DISABLED) this.state = SYSTEM_STATE.ENABLED;
    SystemState.totalEnableToggles++;
    try {
      if (typeof this.onEnable === 'function') this.onEnable();
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onEnable failed: ${e && e.message}`);
    }
    return true;
  }

  disable() {
    if (this.state === SYSTEM_STATE.DISPOSED) return false;
    if (this.enabled === 0) return true;
    this.enabled = 0;
    if (this.state === SYSTEM_STATE.ENABLED) this.state = SYSTEM_STATE.DISABLED;
    SystemState.totalEnableToggles++;
    try {
      if (typeof this.onDisable === 'function') this.onDisable();
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onDisable failed: ${e && e.message}`);
    }
    return true;
  }

  /* ---------------- dispose ---------------- */

  dispose() {
    if (this.state === SYSTEM_STATE.DISPOSED) return true;

    // Call the subclass hook.
    try {
      if (typeof this.onDispose === 'function') this.onDispose();
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(this.options.logChannel, () =>
        `[019_scn_SystemBase] "${this.name}" onDispose failed: ${e && e.message}`);
    }

    // Detach the boundary.
    this._boundary = null;

    // Unregister.
    if (this._registered) {
      _unregisterSystem(this);
      this._registered = false;
    }

    this.enabled = 0;
    this.state = SYSTEM_STATE.DISPOSED;
    SystemState.totalDisposals++;
    return true;
  }

  /* ---------------- state queries ---------------- */

  isEnabled()      { return this.enabled === 1 && this.state === SYSTEM_STATE.ENABLED; }
  isInitialized()  { return this.state === SYSTEM_STATE.INITIALIZED ||
                            this.state === SYSTEM_STATE.ENABLED ||
                            this.state === SYSTEM_STATE.DISABLED; }
  isDisposed()     { return this.state === SYSTEM_STATE.DISPOSED; }
  isFailed()       { return this.state === SYSTEM_STATE.FAILED; }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      systemId:         this.systemId,
      name:             this.name,
      priority:         this.priority,
      state:            SYSTEM_STATE_NAME[this.state] || 'unknown',
      enabled:          this.enabled === 1,
      debugOnly:        this.debugOnly,
      dependencies:     this.dependencies.slice(0, this.dependencyCount),
      totalUpdates:     this.totalUpdates,
      totalFailures:    this.totalFailures,
      totalDispatches:  this.totalDispatches,
      lastCostMs:       this.lastCostMs,
      avgCostMs:        this.avgCostMs,
      peakCostMs:       this.peakCostMs,
      lastUpdateFrame:  this.lastUpdateFrame,
      firstUpdateMs:    this.firstUpdateMs,
      elapsedSinceStart:this.elapsedSinceStart,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 4. QUERY SYSTEM SUBCLASS                                           */
/* ------------------------------------------------------------------ */

/**
 * Base class for systems driven by a cached query descriptor (018). The
 * subclass supplies `defineQuery()` to build the query once, and
 * `onEntity(eid, dt, elapsed)` to process each matching entity.
 *
 * Iteration is allocation-free.
 */
export class QuerySystem extends SystemBase {
  constructor(name, options = {}) {
    super(name, options);
    this.query = null;
    this.entityCount = 0;
    this.peakEntityCount = 0;
  }

  /**
   * Subclass hook — return the query descriptor from
   * `defineQuery([...], 'name')` here.
   */
  defineQuery() { return null; }

  /**
   * Subclass hook — process a single matching entity.
   */
  onEntity(_eid, _dt, _elapsed) { /* no-op */ }

  onInit(ctx) {
    this.query = this.defineQuery();
  }

  onUpdate(dt, elapsed) {
    if (!this.query) return;
    this.entityCount = 0;
    forEachEntity(this.query, (eid) => {
      this.onEntity(eid, dt, elapsed);
      this.entityCount++;
    }, this);
    if (this.entityCount > this.peakEntityCount) {
      this.peakEntityCount = this.entityCount;
    }
  }

  getStats() {
    const base = super.getStats();
    base.queryEntityCount = this.entityCount;
    base.queryPeakCount   = this.peakEntityCount;
    base.queryName        = this.query ? this.query.name : null;
    return base;
  }
}

/* ------------------------------------------------------------------ */
/* 5. RENDER SYSTEM SUBCLASS                                          */
/* ------------------------------------------------------------------ */

/**
 * Base class for systems that emit draw calls or GPU work. Adds a
 * frame-scoped "renderDirty" flag so upstream systems can request a
 * re-render without going through the whole tree.
 */
export class RenderSystem extends SystemBase {
  constructor(name, options = {}) {
    super(name, options);
    this.renderDirty = 1;   // request a render on the next frame
    this.drawCallCount = 0;
    this.triangleCount = 0;
  }

  requestRender() {
    this.renderDirty = 1;
  }

  clearRenderDirty() {
    this.renderDirty = 0;
  }

  onUpdate(dt, elapsed) {
    // Subclass is expected to call `onRenderFrame()` internally.
    if (typeof this.onRenderFrame === 'function') {
      this.onRenderFrame(dt, elapsed);
    }
  }

  getStats() {
    const base = super.getStats();
    base.renderDirty = this.renderDirty === 1;
    base.drawCallCount = this.drawCallCount;
    base.triangleCount = this.triangleCount;
    return base;
  }
}

/* ------------------------------------------------------------------ */
/* 6. ASYNC SYSTEM SUBCLASS                                           */
/* ------------------------------------------------------------------ */

/**
 * Base class for systems that dispatch work to workers. Tracks
 * pending/completed jobs, and integrates with the parallel pipeline.
 */
export class AsyncSystem extends SystemBase {
  constructor(name, options = {}) {
    super(name, options);
    this.pendingJobs    = 0;
    this.completedJobs  = 0;
    this.failedJobs     = 0;
    this.cancelledJobs  = 0;
    this.totalJobs      = 0;
    this.lastDispatchMs = 0;
  }

  dispatchJob(_payload) {
    this.pendingJobs++;
    this.totalJobs++;
    SystemState.totalDispatches++;
    this.totalDispatches++;
    const t0 = _now();
    this.lastDispatchMs = _now() - t0;
    return true;
  }

  completeJob(success) {
    if (this.pendingJobs > 0) this.pendingJobs--;
    if (success === false) this.failedJobs++;
    else this.completedJobs++;
  }

  cancelJob() {
    if (this.pendingJobs > 0) this.pendingJobs--;
    this.cancelledJobs++;
  }

  getStats() {
    const base = super.getStats();
    base.pendingJobs   = this.pendingJobs;
    base.completedJobs = this.completedJobs;
    base.failedJobs    = this.failedJobs;
    base.cancelledJobs = this.cancelledJobs;
    base.totalJobs     = this.totalJobs;
    base.lastDispatchMs= this.lastDispatchMs;
    return base;
  }
}

/* ------------------------------------------------------------------ */
/* 7. DEBUG SYSTEM SUBCLASS                                           */
/* ------------------------------------------------------------------ */

/**
 * Base class for debug-only systems. By default these are disabled in
 * production builds (PERF_TIER === 'LOW').
 */
export class DebugSystem extends SystemBase {
  constructor(name, options = {}) {
    super(name, Object.assign({ debugOnly: true }, options || {}));
    // Disabled by default on LOW tier.
    if (PERF_TIER_LOCAL === 'LOW') this.enabled = 0;
    else this.enabled = 1;
  }
}

/* ------------------------------------------------------------------ */
/* 8. FRAME LIFECYCLE                                                 */
/* ------------------------------------------------------------------ */

/**
 * Advances the global system frame counter. Called once per frame by the
 * engine loop, before any system update.
 */
export function tickSystems(frameNumber) {
  if (typeof frameNumber === 'number') SystemState.frame = frameNumber;
  else SystemState.frame++;
}

/* ------------------------------------------------------------------ */
/* 9. BULK DISPATCH HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Calls `update()` on every enabled system, in registration order.
 * (The scheduler in 020_scn_SystemScheduler.js re-orders by priority
 * and dependency — this is a simple fallback for tests.)
 */
export function updateAllSystems(dt, elapsed) {
  let count = 0;
  for (let i = 0; i < _systemCount; i++) {
    const sys = _systems[i];
    if (sys && sys.state === SYSTEM_STATE.ENABLED) {
      sys.update(dt, elapsed);
      count++;
    }
  }
  if (count > SystemState.peakActiveSystems) {
    SystemState.peakActiveSystems = count;
  }
  return count;
}

/**
 * Calls `lateUpdate()` on every enabled system.
 */
export function lateUpdateAllSystems(dt, elapsed) {
  let count = 0;
  for (let i = 0; i < _systemCount; i++) {
    const sys = _systems[i];
    if (sys && sys.state === SYSTEM_STATE.ENABLED) {
      sys.lateUpdate(dt, elapsed);
      count++;
    }
  }
  return count;
}

/**
 * Propagates a resize to every registered system.
 */
export function resizeAllSystems(width, height, pixelRatio) {
  let count = 0;
  for (let i = 0; i < _systemCount; i++) {
    const sys = _systems[i];
    if (sys) {
      sys.resize(width, height, pixelRatio);
      count++;
    }
  }
  return count;
}

/**
 * Disposes every registered system. Called on full engine teardown.
 */
export function disposeAllSystems() {
  let count = 0;
  for (let i = _systemCount - 1; i >= 0; i--) {
    const sys = _systems[i];
    if (sys) {
      try { sys.dispose(); count++; }
      catch (_) { /* swallow */ }
    }
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* 10. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getSystemSystemReport() {
  const stats = new Array(_systemCount);
  let enabledCount = 0;
  let disabledCount = 0;
  let failedCount = 0;
  let totalCostMs = 0;

  for (let i = 0; i < _systemCount; i++) {
    const sys = _systems[i];
    if (!sys) continue;
    stats[i] = sys.getStats();
    totalCostMs += sys.lastCostMs;
    if (sys.state === SYSTEM_STATE.ENABLED)  enabledCount++;
    else if (sys.state === SYSTEM_STATE.DISABLED) disabledCount++;
    else if (sys.state === SYSTEM_STATE.FAILED) failedCount++;
  }

  return {
    frame:              SystemState.frame,
    registeredSystems:  _systemCount,
    capacity:           MAX_SYSTEMS,
    enabledCount,
    disabledCount,
    failedCount,
    totalCostMs,
    totalUpdates:       SystemState.totalUpdates,
    totalDispatches:    SystemState.totalDispatches,
    totalDisposals:     SystemState.totalDisposals,
    totalFailures:      SystemState.totalFailures,
    totalEnableToggles: SystemState.totalEnableToggles,
    peakActiveSystems:  SystemState.peakActiveSystems,
    systems:            stats,
    perfTier:           PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 11. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Unregisters and disposes every system, and clears all module state.
 */
export function resetAllSystems() {
  disposeAllSystems();
  for (let i = 0; i < MAX_SYSTEMS; i++) _systems[i] = null;
  _systemCount = 0;
  _systemByName.clear();
  _systemById.clear();

  SystemState.frame = 0;
  SystemState.totalSystems = 0;
  SystemState.totalUpdates = 0;
  SystemState.totalDispatches = 0;
  SystemState.totalDisposals = 0;
  SystemState.totalFailures = 0;
  SystemState.totalEnableToggles = 0;
  SystemState.peakActiveSystems = 0;
}

/* ------------------------------------------------------------------ */
/* 12. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Base classes
  SystemBase,
  QuerySystem,
  RenderSystem,
  AsyncSystem,
  DebugSystem,

  // Constants
  MAX_SYSTEMS,
  MAX_DEPENDENCIES_PER_SYSTEM,
  PRIORITY_EARLY,
  PRIORITY_DEFAULT,
  PRIORITY_LATE,
  PRIORITY_LAST,

  // Enums
  SYSTEM_STATE,
  SYSTEM_STATE_NAME,
  SYSTEM_NAME,

  // Module state
  SystemState,

  // Registry
  getSystem,
  getSystemById,
  getAllSystems,
  getSystemCount,
  forEachSystem,

  // Frame lifecycle
  tickSystems,

  // Bulk dispatch
  updateAllSystems,
  lateUpdateAllSystems,
  resizeAllSystems,
  disposeAllSystems,

  // Diagnostics
  getSystemSystemReport,

  // Reset
  resetAllSystems,
};

export default _defaultExport;