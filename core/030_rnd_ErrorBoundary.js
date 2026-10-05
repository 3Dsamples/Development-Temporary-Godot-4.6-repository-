// File : 030
// name : src/core/030_rnd_ErrorBoundary.js
// description : Runtime error containment and recovery primitive for the
//               anime lighting stack on Android mobile. Every lighting
//               subsystem (006_lgt_LightManager through 380_lgt_lights)
//               wraps its per-frame entry points, worker callbacks, and
//               async continuations in an ErrorBoundary so a single thrown
//               error — a NaN in a shader uniform, a missing render target,
//               a worker desync — never crashes the frame, never leaves the
//               engine in an inconsistent state, and never silently corrupts
//               the visual output.
//
//               Responsibilities:
//                 • Containment   — catch exceptions from a scoped piece of
//                                   work, mark the boundary DEGRADED or FAILED,
//                                   and return a safe fallback value.
//                 • Circuit breaker — after N consecutive failures a boundary
//                                   trips OPEN and short-circuits subsequent
//                                   calls for a cooldown window (frame-counted
//                                   so Android backgrounding doesn't burn the
//                                   cooldown while paused).
//                 • Backoff      — cooldown window doubles on each trip
//                                   (exponential) up to a cap, so chronic
//                                   failures stop spamming logs and stop
//                                   burning frame time.
//                 • Recovery     — a boundary can be manually reset, or
//                                   automatically probed after the cooldown
//                                   expires. A successful probe returns the
//                                   boundary to CLOSED.
//                 • Telemetry    — per-boundary failure counts, last error,
//                                   last stack, cooldown state, and recovery
//                                   count. Consumed by 024_rnd_Profiler.js
//                                   and 026_rnd_Logger.js.
//                 • Nested boundaries — a boundary can have a parent; a
//                                   child trip propagates up only if the
//                                   child cannot recover, letting fine-
//                                   grained systems (e.g. one GI probe
//                                   batch) fail without taking down the
//                                   coarse system (whole GI pipeline).
//                 • Zero-alloc hot path — the protected call path touches
//                                   only typed arrays and pre-bound
//                                   callbacks; no closures captured per
//                                   call, no Map/Set lookups, no string
//                                   formatting unless logging actually fires.
//                 • Frame-scoped counters so a single bad frame cannot
//                                   permanently lock a boundary.
//
//               States:
//                 CLOSED      — healthy, calls flow through
//                 HALF_OPEN   — cooldown expired, one probe allowed
//                 OPEN        — tripped, all calls short-circuited
//                 DISABLED    — manually silenced (still counts failures
//                               but never trips or logs)
//                 DISPOSED    — removed from the registry, no more calls
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external error-boundary libs; every internal array
//               sized once at construction.
// best for : Guaranteeing frame-level resilience for the anime lighting
//            stack. A blown shadow atlas page, a NaN propagation through
//            GI, a stale render-target handle after context loss, a worker
//            desync during async probe bake — all become contained events
//            the engine can log, telemetry, and recover from without
//            dropping a frame or corrupting the scene.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
  LOG_LEVEL,
} from './026_rnd_Logger.js';

import {
  getDefaultProfiler,
} from './024_rnd_Profiler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_BOUNDARIES =
  PERF_TIER_LOCAL === 'HIGH'   ? 256 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 192 :
                                 128;

export const BOUNDARY_STATE = Object.freeze({
  CLOSED:    0,
  HALF_OPEN: 1,
  OPEN:      2,
  DISABLED:  3,
  DISPOSED:  4,
  COUNT:     5,
});

export const BOUNDARY_STATE_NAME = Object.freeze([
  'closed',
  'half_open',
  'open',
  'disabled',
  'disposed',
]);

export const BOUNDARY_TAG = Object.freeze({
  GENERIC:    0,
  LIGHTS:     1,
  SHADOWS:    2,
  GI:         3,
  AO:         4,
  CLUSTER:    5,
  ENVIRONMENT:6,
  INTERIOR:   7,
  EXTERIOR:   8,
  POST:       9,
  WORKER:    10,
  REGISTRY:  11,
  POOL:      12,
  FRAME:     13,
  COUNT:     14,
});

export const BOUNDARY_TAG_NAME = Object.freeze([
  'generic',
  'lights',
  'shadows',
  'gi',
  'ao',
  'cluster',
  'environment',
  'interior',
  'exterior',
  'post',
  'worker',
  'registry',
  'pool',
  'frame',
]);

export const DEFAULT_FAILURE_THRESHOLD = 3;
export const DEFAULT_COOLDOWN_FRAMES   = 60;
export const MAX_COOLDOWN_FRAMES       = 3600;
export const BACKOFF_MULTIPLIER        = 2;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _boundaryIdCounter = 0;

function _nextBoundaryId() {
  return ++_boundaryIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _describeError(e) {
  if (e === null) return 'null';
  if (e === undefined) return 'undefined';
  if (typeof e === 'string') return e;
  if (typeof e === 'object') {
    if (typeof e.message === 'string' && e.message.length > 0) return e.message;
    if (typeof e.name === 'string') return e.name;
    try { return String(e); } catch (_) { return '<unstringifiable>'; }
  }
  try { return String(e); } catch (_) { return '<unknown>'; }
}

function _extractStack(e) {
  if (!e || typeof e !== 'object') return null;
  if (typeof e.stack === 'string' && e.stack.length > 0) return e.stack;
  return null;
}

/* ------------------------------------------------------------------ */
/* 2. BOUNDARY SLOT                                                   */
/* ------------------------------------------------------------------ */

export class BoundarySlot {
  constructor(index) {
    this.index          = index;
    this.id             = 0;
    this.name           = null;
    this.tag            = BOUNDARY_TAG.GENERIC;
    this.channel        = LOG_CHANNEL.CORE;
    this.state          = BOUNDARY_STATE.CLOSED;

    // Config.
    this.failureThreshold = DEFAULT_FAILURE_THRESHOLD;
    this.cooldownFrames   = DEFAULT_COOLDOWN_FRAMES;
    this.maxCooldownFrames= MAX_COOLDOWN_FRAMES;
    this.backoffMultiplier= BACKOFF_MULTIPLIER;
    this.silent           = 0;
    this.propagateToParent= 1;

    // Parent / child.
    this.parentIndex    = -1;

    // Runtime counters.
    this.failures       = 0;
    this.consecutive    = 0;
    this.successes      = 0;
    this.trips          = 0;
    this.recoveries     = 0;
    this.shortCircuits  = 0;

    // Cooldown.
    this.cooldownStartFrame = 0;
    this.cooldownFramesLeft = 0;
    this.currentCooldown    = this.cooldownFrames;

    // Error metadata (only populated when a failure fires).
    this.lastError      = null;
    this.lastStack      = null;
    this.lastErrorFrame = -1;
    this.lastErrorMs    = 0;

    // Fallback value (returned when tripped).
    this.fallback       = undefined;
    this.hasFallback    = 0;

    // Timing.
    this.lastCallMs     = 0;
    this.totalCallMs    = 0;
    this.callCount      = 0;
  }

  reset() {
    this.name           = null;
    this.tag            = BOUNDARY_TAG.GENERIC;
    this.channel        = LOG_CHANNEL.CORE;
    this.state          = BOUNDARY_STATE.CLOSED;

    this.failureThreshold = DEFAULT_FAILURE_THRESHOLD;
    this.cooldownFrames   = DEFAULT_COOLDOWN_FRAMES;
    this.maxCooldownFrames= MAX_COOLDOWN_FRAMES;
    this.backoffMultiplier= BACKOFF_MULTIPLIER;
    this.silent           = 0;
    this.propagateToParent= 1;

    this.parentIndex    = -1;

    this.failures       = 0;
    this.consecutive    = 0;
    this.successes      = 0;
    this.trips          = 0;
    this.recoveries     = 0;
    this.shortCircuits  = 0;

    this.cooldownStartFrame = 0;
    this.cooldownFramesLeft = 0;
    this.currentCooldown    = this.cooldownFrames;

    this.lastError      = null;
    this.lastStack      = null;
    this.lastErrorFrame = -1;
    this.lastErrorMs    = 0;

    this.fallback       = undefined;
    this.hasFallback    = 0;

    this.lastCallMs     = 0;
    this.totalCallMs    = 0;
    this.callCount      = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 3. ERROR BOUNDARY MANAGER                                          */
/* ------------------------------------------------------------------ */

export class ErrorBoundaryManager {
  constructor(options = {}) {
    this.options = Object.assign({
      defaultFailureThreshold: DEFAULT_FAILURE_THRESHOLD,
      defaultCooldownFrames:   DEFAULT_COOLDOWN_FRAMES,
      maxCooldownFrames:       MAX_COOLDOWN_FRAMES,
      autoRecover:             true,
      logFailures:             true,
      recordProfilerMark:      true,
      silentInLowTier:         true,
    }, options || {});

    this.capacity = MAX_BOUNDARIES;
    this.slots    = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.slots[i] = new BoundarySlot(i);
    this.count        = 0;
    this.byId         = new Map();
    this.freeHead     = 0;
    this.freeList     = new Int32Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;

    this.frame        = 0;
    this.globalTrips  = 0;
    this.globalErrors = 0;

    this._listeners   = new Map();
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
      try { arr[i](payload); } catch (e) { /* swallow */ }
    }
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame(frameNumber) {
    this.frame = (typeof frameNumber === 'number') ? frameNumber : (this.frame + 1);
    // Tick cooldowns.
    for (let i = 0; i < this.count; i++) {
      const slot = this.slots[i];
      if (slot.state !== BOUNDARY_STATE.OPEN) continue;
      if (slot.cooldownFramesLeft > 0) {
        slot.cooldownFramesLeft--;
        if (slot.cooldownFramesLeft === 0 && this.options.autoRecover) {
          slot.state = BOUNDARY_STATE.HALF_OPEN;
        }
      }
    }
  }

  /* ---------------- boundary registration ---------------- */

  register(name, options = {}) {
    if (this.count >= this.capacity) return null;
    if (typeof name !== 'string' || name.length === 0) return null;

    const idx = this.count++;
    const slot = this.slots[idx];
    slot.reset();

    slot.id              = _nextBoundaryId();
    slot.name            = name;
    slot.tag             = options.tag !== undefined ? options.tag : BOUNDARY_TAG.GENERIC;
    slot.channel         = options.channel !== undefined ? options.channel : LOG_CHANNEL.CORE;
    slot.failureThreshold= options.failureThreshold !== undefined ? options.failureThreshold : this.options.defaultFailureThreshold;
    slot.cooldownFrames  = options.cooldownFrames !== undefined ? options.cooldownFrames : this.options.defaultCooldownFrames;
    slot.maxCooldownFrames = options.maxCooldownFrames !== undefined ? options.maxCooldownFrames : this.options.maxCooldownFrames;
    slot.backoffMultiplier = options.backoffMultiplier !== undefined ? options.backoffMultiplier : BACKOFF_MULTIPLIER;
    slot.silent          = options.silent ? 1 : 0;
    slot.propagateToParent = options.propagateToParent === false ? 0 : 1;
    slot.state           = options.disabled ? BOUNDARY_STATE.DISABLED : BOUNDARY_STATE.CLOSED;
    slot.currentCooldown = slot.cooldownFrames;

    if (options.fallback !== undefined) {
      slot.fallback = options.fallback;
      slot.hasFallback = 1;
    }

    if (options.parentId !== undefined) {
      const parentIdx = this._findIndexById(options.parentId);
      if (parentIdx >= 0) slot.parentIndex = parentIdx;
    }

    this.byId.set(slot.id, idx);

    this._emit('registered', { id: slot.id, name, tag: slot.tag });
    return slot;
  }

  _findIndexById(id) {
    const idx = this.byId.get(id);
    return idx === undefined ? -1 : idx;
  }

  getById(id) {
    const idx = this._findIndexById(id);
    if (idx < 0) return null;
    return this.slots[idx];
  }

  getByName(name) {
    for (let i = 0; i < this.count; i++) {
      if (this.slots[i].name === name) return this.slots[i];
    }
    return null;
  }

  /* ---------------- protected call ---------------- */

  /**
   * Runs `fn(ctx)` under the given boundary. On success returns the
   * function's return value. On failure returns the boundary's fallback
   * (or undefined) and transitions the boundary state.
   *
   * The hot path is a single integer compare against `BOUNDARY_STATE.CLOSED`
   * when the boundary is healthy.
   */
  run(slot, fn, ctx) {
    if (!slot) return undefined;

    // Fast path: boundary is closed.
    if (slot.state === BOUNDARY_STATE.CLOSED || slot.state === BOUNDARY_STATE.HALF_OPEN) {
      const t0 = _now();
      try {
        const result = ctx === undefined ? fn() : fn(ctx);
        const t1 = _now();
        slot.lastCallMs = t1 - t0;
        slot.totalCallMs += slot.lastCallMs;
        slot.callCount++;
        if (slot.state === BOUNDARY_STATE.HALF_OPEN) {
          this._recover(slot);
        } else {
          slot.consecutive = 0;
          slot.successes++;
        }
        return result;
      } catch (e) {
        const t1 = _now();
        slot.lastCallMs = t1 - t0;
        slot.totalCallMs += slot.lastCallMs;
        slot.callCount++;
        this._recordFailure(slot, e);
        return slot.hasFallback ? slot.fallback : undefined;
      }
    }

    if (slot.state === BOUNDARY_STATE.OPEN) {
      slot.shortCircuits++;
      // Propagate to parent if configured.
      if (slot.propagateToParent && slot.parentIndex >= 0) {
        const parent = this.slots[slot.parentIndex];
        if (parent && parent.state === BOUNDARY_STATE.CLOSED) {
          this._recordFailure(parent, new Error(`child boundary "${slot.name}" tripped`));
        }
      }
      return slot.hasFallback ? slot.fallback : undefined;
    }

    if (slot.state === BOUNDARY_STATE.DISABLED) {
      // Still run, still count, but never trip.
      try {
        return ctx === undefined ? fn() : fn(ctx);
      } catch (e) {
        slot.failures++;
        this._emit('failure', { id: slot.id, name: slot.name, error: e, silent: true });
        return slot.hasFallback ? slot.fallback : undefined;
      }
    }

    // DISPOSED.
    return slot.hasFallback ? slot.fallback : undefined;
  }

  /**
   * Async wrapper. `fn(ctx)` must return a Promise. Returns a Promise
   * that never rejects — a rejection is contained and treated like a
   * synchronous failure. This is what async lighting tasks (worker probe
   * bakes, shadow atlas rebuilds, GI async updates) run inside.
   */
  runAsync(slot, fn, ctx) {
    if (!slot) return Promise.resolve(undefined);

    if (slot.state === BOUNDARY_STATE.OPEN) {
      slot.shortCircuits++;
      return Promise.resolve(slot.hasFallback ? slot.fallback : undefined);
    }

    if (slot.state === BOUNDARY_STATE.DISPOSED) {
      return Promise.resolve(slot.hasFallback ? slot.fallback : undefined);
    }

    const t0 = _now();
    let p;
    try {
      p = ctx === undefined ? fn() : fn(ctx);
    } catch (e) {
      const t1 = _now();
      slot.lastCallMs = t1 - t0;
      slot.totalCallMs += slot.lastCallMs;
      slot.callCount++;
      this._recordFailure(slot, e);
      return Promise.resolve(slot.hasFallback ? slot.fallback : undefined);
    }

    if (!p || typeof p.then !== 'function') {
      const t1 = _now();
      slot.lastCallMs = t1 - t0;
      slot.totalCallMs += slot.lastCallMs;
      slot.callCount++;
      if (slot.state === BOUNDARY_STATE.HALF_OPEN) this._recover(slot);
      else { slot.consecutive = 0; slot.successes++; }
      return Promise.resolve(p);
    }

    return p.then(
      (result) => {
        const t1 = _now();
        slot.lastCallMs = t1 - t0;
        slot.totalCallMs += slot.lastCallMs;
        slot.callCount++;
        if (slot.state === BOUNDARY_STATE.HALF_OPEN) this._recover(slot);
        else { slot.consecutive = 0; slot.successes++; }
        return result;
      },
      (e) => {
        const t1 = _now();
        slot.lastCallMs = t1 - t0;
        slot.totalCallMs += slot.lastCallMs;
        slot.callCount++;
        this._recordFailure(slot, e);
        return slot.hasFallback ? slot.fallback : undefined;
      }
    );
  }

  /* ---------------- failure / recovery ---------------- */

  _recordFailure(slot, e) {
    slot.failures++;
    slot.consecutive++;
    slot.lastError      = _describeError(e);
    slot.lastStack      = _extractStack(e);
    slot.lastErrorFrame = this.frame;
    slot.lastErrorMs    = _now();

    this.globalErrors++;

    if (this.options.logFailures && !slot.silent) {
      const log = getDefaultLogger();
      if (log) {
        log.warn(slot.channel, () =>
          `[030_rnd_ErrorBoundary] "${slot.name}" failure ${slot.consecutive}/${slot.failureThreshold}: ${slot.lastError}`
        );
      }
    }

    this._emit('failure', {
      id: slot.id,
      name: slot.name,
      error: e,
      consecutive: slot.consecutive,
      threshold: slot.failureThreshold,
    });

    if (this.options.recordProfilerMark) {
      try {
        const p = getDefaultProfiler();
        if (p) p.mark('boundary_failure:' + (slot.name || 'anon'));
      } catch (_) { /* swallow */ }
    }

    // Trip if threshold exceeded.
    if (slot.state !== BOUNDARY_STATE.DISABLED &&
        slot.consecutive >= slot.failureThreshold) {
      this._trip(slot);
    }
  }

  _trip(slot) {
    slot.state = BOUNDARY_STATE.OPEN;
    slot.trips++;
    this.globalTrips++;

    // Exponential backoff on cooldown.
    const next = Math.min(slot.maxCooldownFrames, Math.round(slot.currentCooldown * slot.backoffMultiplier));
    slot.currentCooldown = next;
    slot.cooldownFramesLeft = next;
    slot.cooldownStartFrame = this.frame;

    if (this.options.logFailures && !slot.silent) {
      const log = getDefaultLogger();
      if (log) {
        log.error(slot.channel, () =>
          `[030_rnd_ErrorBoundary] "${slot.name}" tripped OPEN — last error: ${slot.lastError} ` +
          `(cooldown ${next} frames, trip #${slot.trips})`
        );
      }
    }

    this._emit('tripped', {
      id: slot.id,
      name: slot.name,
      trips: slot.trips,
      cooldownFrames: next,
      lastError: slot.lastError,
    });
  }

  _recover(slot) {
    slot.state = BOUNDARY_STATE.CLOSED;
    slot.recoveries++;
    slot.consecutive = 0;
    slot.currentCooldown = slot.cooldownFrames; // reset backoff

    if (this.options.logFailures && !slot.silent) {
      const log = getDefaultLogger();
      if (log) {
        log.info(slot.channel, `[030_rnd_ErrorBoundary] "${slot.name}" recovered (recovery #${slot.recoveries})`);
      }
    }

    this._emit('recovered', {
      id: slot.id,
      name: slot.name,
      recoveries: slot.recoveries,
    });
  }

  /* ---------------- manual controls ---------------- */

  reset(slot) {
    if (!slot) return false;
    if (slot.state === BOUNDARY_STATE.DISPOSED) return false;
    slot.state = BOUNDARY_STATE.CLOSED;
    slot.consecutive = 0;
    slot.currentCooldown = slot.cooldownFrames;
    slot.cooldownFramesLeft = 0;
    this._emit('reset', { id: slot.id, name: slot.name });
    return true;
  }

  disable(slot) {
    if (!slot) return false;
    slot.state = BOUNDARY_STATE.DISABLED;
    this._emit('disabled', { id: slot.id, name: slot.name });
    return true;
  }

  enable(slot) {
    if (!slot) return false;
    if (slot.state === BOUNDARY_STATE.DISABLED) {
      slot.state = BOUNDARY_STATE.CLOSED;
      slot.consecutive = 0;
    }
    this._emit('enabled', { id: slot.id, name: slot.name });
    return true;
  }

  disposeBoundary(slot) {
    if (!slot) return false;
    slot.state = BOUNDARY_STATE.DISPOSED;
    this._emit('disposed', { id: slot.id, name: slot.name });
    return true;
  }

  /* ---------------- diagnostics ---------------- */

  anyOpen() {
    for (let i = 0; i < this.count; i++) {
      if (this.slots[i].state === BOUNDARY_STATE.OPEN) return true;
    }
    return false;
  }

  countOpen() {
    let n = 0;
    for (let i = 0; i < this.count; i++) {
      if (this.slots[i].state === BOUNDARY_STATE.OPEN) n++;
    }
    return n;
  }

  listOpen() {
    const out = [];
    for (let i = 0; i < this.count; i++) {
      const slot = this.slots[i];
      if (slot.state === BOUNDARY_STATE.OPEN) {
        out.push({
          id: slot.id,
          name: slot.name,
          tag: BOUNDARY_TAG_NAME[slot.tag],
          trips: slot.trips,
          cooldownFramesLeft: slot.cooldownFramesLeft,
          lastError: slot.lastError,
        });
      }
    }
    return out;
  }

  getStats() {
    const boundaries = new Array(this.count);
    for (let i = 0; i < this.count; i++) {
      const s = this.slots[i];
      boundaries[i] = {
        id:              s.id,
        name:            s.name,
        tag:             BOUNDARY_TAG_NAME[s.tag],
        state:           BOUNDARY_STATE_NAME[s.state],
        failures:        s.failures,
        consecutive:     s.consecutive,
        successes:       s.successes,
        trips:           s.trips,
        recoveries:      s.recoveries,
        shortCircuits:   s.shortCircuits,
        callCount:       s.callCount,
        avgCallMs:       s.callCount > 0 ? (s.totalCallMs / s.callCount) : 0,
        cooldownFramesLeft: s.cooldownFramesLeft,
        lastError:       s.lastError,
        lastErrorFrame:  s.lastErrorFrame,
      };
    }

    return {
      frame:        this.frame,
      boundaryCount:this.count,
      capacity:     this.capacity,
      openCount:    this.countOpen(),
      globalTrips:  this.globalTrips,
      globalErrors: this.globalErrors,
      boundaries,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    for (let i = 0; i < this.count; i++) this.slots[i].reset();
    this.count = 0;
    this.byId.clear();
    this.freeHead = 0;
    for (let i = 0; i < this.capacity; i++) this.freeList[i] = i;
    this.frame = 0;
    this.globalTrips = 0;
    this.globalErrors = 0;
    return this;
  }

  dispose() {
    this.reset();
    this.slots.length = 0;
    this.slots = null;
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. BOUNDARY HANDLE (ergonomic wrapper)                             */
/* ------------------------------------------------------------------ */

/**
 * A lightweight wrapper around a BoundarySlot so callers can hold a
 * stable handle that survives manager resets. Also exposes bound
 * run/runAsync helpers so calling code is terse:
 *
 *   const b = manager.create('gi.bake', { tag: BOUNDARY_TAG.GI });
 *   b.run(() => doBake());
 *   b.runAsync(() => doAsyncBake());
 */
export class BoundaryHandle {
  constructor(manager, slot) {
    this.manager = manager;
    this.slot    = slot;
  }

  get name()    { return this.slot ? this.slot.name : null; }
  get state()   { return this.slot ? this.slot.state : BOUNDARY_STATE.DISPOSED; }
  get isOpen()  { return this.slot && this.slot.state === BOUNDARY_STATE.OPEN; }

  run(fn, ctx) {
    return this.manager.run(this.slot, fn, ctx);
  }

  runAsync(fn, ctx) {
    return this.manager.runAsync(this.slot, fn, ctx);
  }

  reset()    { return this.manager.reset(this.slot); }
  disable()  { return this.manager.disable(this.slot); }
  enable()   { return this.manager.enable(this.slot); }
  dispose()  { return this.manager.disposeBoundary(this.slot); }

  setFallback(value) {
    if (this.slot) {
      this.slot.fallback = value;
      this.slot.hasFallback = 1;
    }
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. MANAGER EXTENSION — create() returns a BoundaryHandle           */
/* ------------------------------------------------------------------ */

ErrorBoundaryManager.prototype.create = function (name, options) {
  const slot = this.register(name, options);
  if (!slot) return null;
  return new BoundaryHandle(this, slot);
};

/* ------------------------------------------------------------------ */
/* 6. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultManager = null;

export function getDefaultErrorBoundaries() {
  if (!_defaultManager) _defaultManager = new ErrorBoundaryManager();
  return _defaultManager;
}

export function disposeDefaultErrorBoundaries() {
  if (_defaultManager) {
    _defaultManager.dispose();
    _defaultManager = null;
  }
}

/* ------------------------------------------------------------------ */
/* 7. HOT-PATH HELPERS                                                */
/* ------------------------------------------------------------------ */

export function boundariesBeginFrame(frameNumber) {
  getDefaultErrorBoundaries().beginFrame(frameNumber);
}

export function createBoundary(name, options) {
  return getDefaultErrorBoundaries().create(name, options);
}

/**
 * Safe global wrapper for one-off calls that don't have a dedicated
 * boundary. Never throws; always returns `fallback` on error.
 */
export function trySafe(fn, fallback) {
  try {
    return fn();
  } catch (e) {
    const log = getDefaultLogger();
    if (log) log.error(LOG_CHANNEL.CORE, `[030_rnd_ErrorBoundary] trySafe: ${_describeError(e)}`);
    return fallback;
  }
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createErrorBoundaryManager(options = {}) {
  return new ErrorBoundaryManager(options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ErrorBoundaryManager,
  BoundarySlot,
  BoundaryHandle,

  createErrorBoundaryManager,
  getDefaultErrorBoundaries,
  disposeDefaultErrorBoundaries,

  boundariesBeginFrame,
  createBoundary,
  trySafe,

  BOUNDARY_STATE,
  BOUNDARY_STATE_NAME,
  BOUNDARY_TAG,
  BOUNDARY_TAG_NAME,

  MAX_BOUNDARIES,
  DEFAULT_FAILURE_THRESHOLD,
  DEFAULT_COOLDOWN_FRAMES,
  MAX_COOLDOWN_FRAMES,
  BACKOFF_MULTIPLIER,
};

export default _defaultExport;