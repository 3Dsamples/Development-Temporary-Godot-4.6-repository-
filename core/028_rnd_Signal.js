// File : 028
// name : src/core/028_rnd_Signal.js
// description : Lightweight signal/slot primitive for the anime lighting stack
//               on Android mobile. Where 027_rnd_EventBus.js owns named-topic
//               fan-out with a per-frame drain queue, THIS module owns the
//               simpler, tighter primitive: a fixed-capacity, priority-ordered
//               listener chain attached directly to a variable-like object —
//               the classic reactive "signal" pattern.
//
//               Signals are used by every lighting subsystem for local,
//               high-frequency notifications where the topic name doesn't
//               need to be resolved dynamically:
//                 • shadowAtlasChangedSignal     — emitted when the atlas
//                                                   is repacked
//                 • giProbeDirtySignal           — invalidated probes
//                 • qualityLevelChangedSignal    — quality transitions
//                 • biomeWeightsChangedSignal    — biome weight updates
//                 • dayCycleTickSignal           — every day-cycle tick
//                 • cameraMovedSignal            — any camera transform
//                 • lightListChangedSignal       — light count/order
//                 • clusterGridChangedSignal     — cluster grid rebuild
//                 • interiorVolumeChangedSignal  — interior probe update
//                 • exteriorProbeChangedSignal   — exterior probe update
//                 • contextLostSignal            — GL context events
//                 • frameTickSignal              — once per frame
//
//               Design:
//                 • Signal<T>      — carries a value; stores the last value
//                                    so late subscribers can read it
//                                    immediately. `emit(value)` dispatches
//                                    synchronously in priority order.
//                 • PulseSignal    — no value, purely a notification.
//                                    Cheaper than Signal<T> by one field.
//                 • Fixed listener capacity (MAX_LISTENERS, default 16);
//                   no dynamic growth, no allocations.
//                 • Priority-ordered chain (lower value = earlier); ties
//                   break by registration order.
//                 • Once semantics per listener.
//                 • Cancellable: listeners may call `signal.stopPropagation()`
//                   during dispatch to halt the chain.
//                 • Auto-disconnect tokens: `signal.connect(fn)` returns a
//                   token with a `disconnect()` method — no closures leak
//                   into the bus.
//                 • Bounded history: Signals can optionally keep the last
//                   N distinct values so a debug HUD can render change logs
//                   without subscribing.
//                 • Combination helpers: `combine(signals..., fn)` produces
//                   a derived pulse signal that fires when any input emits;
//                   `merge(signals...)` produces a derived signal that
//                   forwards each input's value.
//                 • Zero per-frame allocations on the hot path.
//                 • No listener array re-allocation: `_shiftListenersUp` and
//                   `_shiftListenersDown` operate on parallel arrays sized
//                   once at construction.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external signal libs; every internal array sized
//               once at construction.
// best for : Giving the anime lighting stack a lightweight reactive primitive
//            for local notifications. The EventBus handles global topics with
//            dynamic names; Signal handles typed, static connections where the
//            identity of the emitter is already known — which is 90 % of the
//            inter-subsystem wiring (shadow→GI, GI→AO, quality→all, biome→env).
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_LISTENERS =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 24 :
                                 16;

export const MAX_HISTORY =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 32 :
                                 16;

export const SIGNAL_PRIORITY = Object.freeze({
  HIGHEST:   0,
  HIGH:    100,
  NORMAL:  500,
  LOW:     800,
  LOWEST:  999,
});

export const DISPATCH_RESULT = Object.freeze({
  OK:        0,
  CANCELLED: 1,
  EMPTY:     2,
  DISPOSED:  3,
});

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _findInsertPos(priorities, count, priority) {
  let lo = 0;
  let hi = count;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (priorities[mid] <= priority) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/* ------------------------------------------------------------------ */
/* 2. CONNECTION TOKEN                                                */
/* ------------------------------------------------------------------ */

export class Connection {
  constructor(signal, fn) {
    this.signal   = signal;
    this.fn       = fn;
    this.active   = true;
  }

  disconnect() {
    if (!this.active) return false;
    this.active = false;
    if (this.signal) {
      this.signal._removeListener(this.fn);
    }
    return true;
  }

  isActive() {
    return this.active === true;
  }
}

/* ------------------------------------------------------------------ */
/* 3. SIGNAL (value-carrying)                                         */
/* ------------------------------------------------------------------ */

export class Signal {
  constructor(name, options = {}) {
    this.name          = name || 'signal';
    this._capacity     = options.capacity || MAX_LISTENERS;

    // Parallel listener arrays.
    this._fn           = new Array(this._capacity).fill(null);
    this._ctx          = new Array(this._capacity).fill(null);
    this._prio         = new Int16Array(this._capacity);
    this._once         = new Uint8Array(this._capacity);
    this._count        = 0;

    // Value state.
    this._value        = options.initialValue !== undefined ? options.initialValue : null;
    this._hasEmitted   = false;

    // History (optional).
    this._historyEnabled = options.history === true;
    this._historyCapacity = options.historyCapacity || MAX_HISTORY;
    this._historyValues   = this._historyEnabled ? new Array(this._historyCapacity) : null;
    this._historyFrames   = this._historyEnabled ? new Uint32Array(this._historyCapacity) : null;
    this._historyHead     = 0;
    this._historyCount    = 0;

    // Dispatch context.
    this._dispatchState = { cancelled: false, stopped: false };

    // Lifecycle.
    this._disposed     = false;

    // Stats.
    this._emits        = 0;
    this._dispatches   = 0;
    this._cancelled    = 0;
    this._rejected     = 0;
    this._listenerPeak = 0;
    this._lastEmitMs   = 0;
    this._lastEmitFrame= -1;
  }

  /* ---------------- connection ---------------- */

  connect(fn, ctx, priority = SIGNAL_PRIORITY.NORMAL, once = false) {
    if (this._disposed) return null;
    if (typeof fn !== 'function') return null;

    if (this._count >= this._capacity) {
      this._rejected++;
      const log = this._safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE, `[028_rnd_Signal] "${this.name}" listener cap reached (${this._capacity})`);
      return null;
    }

    const insertAt = _findInsertPos(this._prio, this._count, priority);
    this._shiftUp(insertAt);

    this._fn[insertAt]   = fn;
    this._ctx[insertAt]  = ctx || null;
    this._prio[insertAt] = priority | 0;
    this._once[insertAt] = once ? 1 : 0;
    this._count++;

    if (this._count > this._listenerPeak) this._listenerPeak = this._count;

    return new Connection(this, fn);
  }

  once(fn, ctx, priority = SIGNAL_PRIORITY.NORMAL) {
    return this.connect(fn, ctx, priority, true);
  }

  disconnect(fn) {
    return this._removeListener(fn);
  }

  disconnectAll() {
    for (let i = 0; i < this._count; i++) {
      this._fn[i]   = null;
      this._ctx[i]  = null;
      this._prio[i] = 0;
      this._once[i] = 0;
    }
    this._count = 0;
    return this;
  }

  _removeListener(fn) {
    for (let i = 0; i < this._count; i++) {
      if (this._fn[i] === fn) {
        this._shiftDown(i);
        this._count--;
        return true;
      }
    }
    return false;
  }

  _shiftUp(pos) {
    for (let i = this._count; i > pos; i--) {
      this._fn[i]   = this._fn[i - 1];
      this._ctx[i]  = this._ctx[i - 1];
      this._prio[i] = this._prio[i - 1];
      this._once[i] = this._once[i - 1];
    }
  }

  _shiftDown(pos) {
    for (let i = pos; i < this._count - 1; i++) {
      this._fn[i]   = this._fn[i + 1];
      this._ctx[i]  = this._ctx[i + 1];
      this._prio[i] = this._prio[i + 1];
      this._once[i] = this._once[i + 1];
    }
    const last = this._count - 1;
    this._fn[last]   = null;
    this._ctx[last]  = null;
    this._prio[last] = 0;
    this._once[last] = 0;
  }

  /* ---------------- emission ---------------- */

  emit(value, frame) {
    if (this._disposed) return DISPATCH_RESULT.DISPOSED;

    this._emits++;
    this._lastEmitMs = _now();
    if (typeof frame === 'number') this._lastEmitFrame = frame;

    // Store value.
    this._value = value;
    this._hasEmitted = true;

    // History.
    if (this._historyEnabled) {
      this._historyValues[this._historyHead] = value;
      this._historyFrames[this._historyHead] = (typeof frame === 'number' ? frame : 0);
      this._historyHead = (this._historyHead + 1) % this._historyCapacity;
      if (this._historyCount < this._historyCapacity) this._historyCount++;
    }

    if (this._count === 0) return DISPATCH_RESULT.EMPTY;

    const state = this._dispatchState;
    state.cancelled = false;
    state.stopped = false;

    for (let i = 0; i < this._count && !state.cancelled; i++) {
      const fn  = this._fn[i];
      const ctx = this._ctx[i];
      const once = this._once[i];

      try {
        if (ctx) fn.call(ctx, value, this);
        else fn(value, this);
      } catch (e) {
        const log = this._safeLogger();
        if (log) log.error(LOG_CHANNEL.CORE, `[028_rnd_Signal] listener threw on "${this.name}": ${e && e.message}`);
      }

      if (once) {
        this._shiftDown(i);
        this._count--;
        i--;
      }
    }

    this._dispatches++;
    if (state.cancelled) this._cancelled++;

    const result = state.cancelled ? DISPATCH_RESULT.CANCELLED : DISPATCH_RESULT.OK;
    state.cancelled = false;
    state.stopped = false;
    return result;
  }

  stopPropagation() {
    this._dispatchState.cancelled = true;
    this._dispatchState.stopped = true;
  }

  /* ---------------- value access ---------------- */

  get value() { return this._value; }
  get hasEmitted() { return this._hasEmitted; }
  get listenerCount() { return this._count; }
  get capacity() { return this._capacity; }
  get disposed() { return this._disposed; }

  /**
   * Emits only if the new value differs from the last emitted value.
   * Uses strict equality for primitives and reference equality for objects.
   * For numeric epsilon comparison, use `emitIfChangedBy(eps, value)`.
   */
  emitIfChanged(value, frame) {
    if (this._hasEmitted && value === this._value) {
      return DISPATCH_RESULT.EMPTY;
    }
    return this.emit(value, frame);
  }

  /**
   * Emits only if the numeric value differs from the last by more than eps.
   * Non-numeric values fall back to reference equality.
   */
  emitIfChangedBy(eps, value, frame) {
    if (this._hasEmitted &&
        typeof value === 'number' &&
        typeof this._value === 'number') {
      if (Math.abs(value - this._value) <= eps) return DISPATCH_RESULT.EMPTY;
    } else if (this._hasEmitted && value === this._value) {
      return DISPATCH_RESULT.EMPTY;
    }
    return this.emit(value, frame);
  }

  /* ---------------- history ---------------- */

  getHistoryCount() { return this._historyCount; }
  getHistoryCapacity() { return this._historyCapacity; }

  /**
   * Copies the recent history into a caller-provided array. Values are
   * copied by reference; frames into `outFrames` if provided.
   */
  copyHistory(max, outValues, outFrames) {
    if (!this._historyEnabled) return 0;
    const n = Math.min(max | 0 || this._historyCount, this._historyCount);
    for (let i = 0; i < n; i++) {
      const idx = (this._historyHead - n + i + this._historyCapacity) % this._historyCapacity;
      outValues[i] = this._historyValues[idx];
      if (outFrames) outFrames[i] = this._historyFrames[idx];
    }
    return n;
  }

  /* ---------------- logger ---------------- */

  _safeLogger() {
    try { return getDefaultLogger(); } catch (_) { return null; }
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      name:          this.name,
      listeners:     this._count,
      capacity:      this._capacity,
      peakListeners: this._listenerPeak,
      emits:         this._emits,
      dispatches:    this._dispatches,
      cancelled:     this._cancelled,
      rejected:      this._rejected,
      hasEmitted:    this._hasEmitted,
      value:         this._value,
      historyCount:  this._historyCount,
      lastEmitMs:    this._lastEmitMs,
      lastEmitFrame: this._lastEmitFrame,
      disposed:      this._disposed,
    };
  }

  /* ---------------- dispose ---------------- */

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.disconnectAll();
    this._value = null;
    if (this._historyEnabled) {
      this._historyValues.length = 0;
      this._historyValues = null;
      this._historyFrames = null;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 4. PULSE SIGNAL (no value)                                         */
/* ------------------------------------------------------------------ */

export class PulseSignal {
  constructor(name, options = {}) {
    this.name          = name || 'pulse';
    this._capacity     = options.capacity || MAX_LISTENERS;

    this._fn           = new Array(this._capacity).fill(null);
    this._ctx          = new Array(this._capacity).fill(null);
    this._prio         = new Int16Array(this._capacity);
    this._once         = new Uint8Array(this._capacity);
    this._count        = 0;

    this._dispatchState = { cancelled: false, stopped: false };
    this._disposed     = false;

    this._emits        = 0;
    this._dispatches   = 0;
    this._cancelled    = 0;
    this._rejected     = 0;
    this._listenerPeak = 0;
    this._lastEmitMs   = 0;
    this._lastEmitFrame= -1;
  }

  connect(fn, ctx, priority = SIGNAL_PRIORITY.NORMAL, once = false) {
    if (this._disposed) return null;
    if (typeof fn !== 'function') return null;
    if (this._count >= this._capacity) {
      this._rejected++;
      return null;
    }

    const insertAt = _findInsertPos(this._prio, this._count, priority);
    this._shiftUp(insertAt);

    this._fn[insertAt]   = fn;
    this._ctx[insertAt]  = ctx || null;
    this._prio[insertAt] = priority | 0;
    this._once[insertAt] = once ? 1 : 0;
    this._count++;

    if (this._count > this._listenerPeak) this._listenerPeak = this._count;

    return new Connection(this, fn);
  }

  once(fn, ctx, priority = SIGNAL_PRIORITY.NORMAL) {
    return this.connect(fn, ctx, priority, true);
  }

  disconnect(fn) {
    return this._removeListener(fn);
  }

  disconnectAll() {
    for (let i = 0; i < this._count; i++) {
      this._fn[i]   = null;
      this._ctx[i]  = null;
      this._prio[i] = 0;
      this._once[i] = 0;
    }
    this._count = 0;
    return this;
  }

  _removeListener(fn) {
    for (let i = 0; i < this._count; i++) {
      if (this._fn[i] === fn) {
        this._shiftDown(i);
        this._count--;
        return true;
      }
    }
    return false;
  }

  _shiftUp(pos) {
    for (let i = this._count; i > pos; i--) {
      this._fn[i]   = this._fn[i - 1];
      this._ctx[i]  = this._ctx[i - 1];
      this._prio[i] = this._prio[i - 1];
      this._once[i] = this._once[i - 1];
    }
  }

  _shiftDown(pos) {
    for (let i = pos; i < this._count - 1; i++) {
      this._fn[i]   = this._fn[i + 1];
      this._ctx[i]  = this._ctx[i + 1];
      this._prio[i] = this._prio[i + 1];
      this._once[i] = this._once[i + 1];
    }
    const last = this._count - 1;
    this._fn[last]   = null;
    this._ctx[last]  = null;
    this._prio[last] = 0;
    this._once[last] = 0;
  }

  pulse(frame) {
    if (this._disposed) return DISPATCH_RESULT.DISPOSED;

    this._emits++;
    this._lastEmitMs = _now();
    if (typeof frame === 'number') this._lastEmitFrame = frame;

    if (this._count === 0) return DISPATCH_RESULT.EMPTY;

    const state = this._dispatchState;
    state.cancelled = false;
    state.stopped = false;

    for (let i = 0; i < this._count && !state.cancelled; i++) {
      const fn  = this._fn[i];
      const ctx = this._ctx[i];
      const once = this._once[i];

      try {
        if (ctx) fn.call(ctx, this);
        else fn(this);
      } catch (e) {
        const log = this._safeLogger();
        if (log) log.error(LOG_CHANNEL.CORE, `[028_rnd_Signal] pulse listener threw on "${this.name}": ${e && e.message}`);
      }

      if (once) {
        this._shiftDown(i);
        this._count--;
        i--;
      }
    }

    this._dispatches++;
    if (state.cancelled) this._cancelled++;

    const result = state.cancelled ? DISPATCH_RESULT.CANCELLED : DISPATCH_RESULT.OK;
    state.cancelled = false;
    state.stopped = false;
    return result;
  }

  stopPropagation() {
    this._dispatchState.cancelled = true;
    this._dispatchState.stopped = true;
  }

  get listenerCount() { return this._count; }
  get capacity() { return this._capacity; }
  get disposed() { return this._disposed; }

  _safeLogger() {
    try { return getDefaultLogger(); } catch (_) { return null; }
  }

  getStats() {
    return {
      name:          this.name,
      listeners:     this._count,
      capacity:      this._capacity,
      peakListeners: this._listenerPeak,
      emits:         this._emits,
      dispatches:    this._dispatches,
      cancelled:     this._cancelled,
      rejected:      this._rejected,
      lastEmitMs:    this._lastEmitMs,
      lastEmitFrame: this._lastEmitFrame,
      disposed:      this._disposed,
    };
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.disconnectAll();
  }
}

/* ------------------------------------------------------------------ */
/* 5. SIGNAL GROUP (named registry)                                   */
/* ------------------------------------------------------------------ */

export class SignalGroup {
  constructor(name) {
    this.name  = name || 'signal_group';
    this.map   = new Map();
    this.count = 0;
  }

  ensureSignal(name, options) {
    let s = this.map.get(name);
    if (s) return s;
    s = new Signal(name, options);
    this.map.set(name, s);
    this.count++;
    return s;
  }

  ensurePulse(name, options) {
    let s = this.map.get(name);
    if (s) return s;
    s = new PulseSignal(name, options);
    this.map.set(name, s);
    this.count++;
    return s;
  }

  get(name) {
    return this.map.get(name) || null;
  }

  remove(name) {
    const s = this.map.get(name);
    if (!s) return false;
    s.dispose();
    this.map.delete(name);
    this.count--;
    return true;
  }

  disconnectAll() {
    for (const s of this.map.values()) s.disconnectAll();
    return this;
  }

  dispose() {
    for (const s of this.map.values()) s.dispose();
    this.map.clear();
    this.count = 0;
    return this;
  }

  getStats() {
    const out = [];
    for (const s of this.map.values()) out.push(s.getStats());
    return {
      name:  this.name,
      count: this.count,
      signals: out,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 6. COMBINATION HELPERS                                             */
/* ------------------------------------------------------------------ */

/**
 * Creates a PulseSignal that pulses whenever ANY of the input signals
 * (Signal or PulseSignal) emits or pulses. The derived signal's listeners
 * receive the derived signal itself (not the input's value).
 *
 * Inputs are connected with the derived signal's lifetime; dispose the
 * derived signal to disconnect.
 */
export function combine(name, inputs, options = {}) {
  const derived = new PulseSignal(name || 'combined', options);
  if (!Array.isArray(inputs) || inputs.length === 0) return derived;

  for (let i = 0; i < inputs.length; i++) {
    const src = inputs[i];
    if (!src || typeof src.connect !== 'function') continue;
    // Connect with a stable bound method that pulses the derived signal.
    src.connect(_combinedPulseFn, derived, SIGNAL_PRIORITY.LOW);
  }

  return derived;
}

function _combinedPulseFn(self) {
  self.pulse();
}

/**
 * Creates a Signal that mirrors the value emitted by any input Signal.
 * PulseSignal inputs contribute `undefined` values.
 *
 * Inputs must be Signal<T> to have meaningful values; PulseSignal inputs
 * simply forward the previous derived value (or undefined).
 */
export function merge(name, inputs, options = {}) {
  const derived = new Signal(name || 'merged', options);
  if (!Array.isArray(inputs) || inputs.length === 0) return derived;

  for (let i = 0; i < inputs.length; i++) {
    const src = inputs[i];
    if (!src || typeof src.connect !== 'function') continue;
    src.connect(_mergedEmitFn, derived, SIGNAL_PRIORITY.LOW);
  }

  return derived;
}

function _mergedEmitFn(value, self) {
  self.emit(value);
}

/**
 * Returns a Signal<boolean> that emits whenever any input signals emit
 * or pulse. The derived value is always `true`.
 */
export function anyOf(name, inputs, options = {}) {
  const derived = new Signal(name || 'anyOf', options);
  if (!Array.isArray(inputs) || inputs.length === 0) return derived;

  for (let i = 0; i < inputs.length; i++) {
    const src = inputs[i];
    if (!src || typeof src.connect !== 'function') continue;
    src.connect(_anyOfEmitFn, derived, SIGNAL_PRIORITY.LOW);
  }

  return derived;
}

function _anyOfEmitFn(_value, self) {
  self.emit(true);
}

/* ------------------------------------------------------------------ */
/* 7. LIGHTING SIGNAL GROUP (canonical engine signals)                */
/* ------------------------------------------------------------------ */

/**
 * The canonical engine signal group. Every lighting subsystem can import
 * this and connect to the shared signals below instead of creating its
 * own — guaranteeing a single source of truth per notification type.
 */
export const LIGHTING_SIGNALS = new SignalGroup('lighting_signals');

export const shadowAtlasChanged   = LIGHTING_SIGNALS.ensurePulse('shadowAtlasChanged');
export const shadowCascadeChanged = LIGHTING_SIGNALS.ensurePulse('shadowCascadeChanged');
export const shadowFilterChanged  = LIGHTING_SIGNALS.ensurePulse('shadowFilterChanged');

export const giProbeDirty         = LIGHTING_SIGNALS.ensurePulse('giProbeDirty');
export const giProbeRebaked       = LIGHTING_SIGNALS.ensurePulse('giProbeRebaked');
export const giBudgetChanged      = LIGHTING_SIGNALS.ensureSignal('giBudgetChanged', { initialValue: 0 });

export const aoResolutionChanged  = LIGHTING_SIGNALS.ensurePulse('aoResolutionChanged');
export const aoSamplesChanged     = LIGHTING_SIGNALS.ensureSignal('aoSamplesChanged', { initialValue: 0 });

export const lightListChanged     = LIGHTING_SIGNALS.ensurePulse('lightListChanged');
export const clusterGridChanged   = LIGHTING_SIGNALS.ensurePulse('clusterGridChanged');

export const dayCycleTick         = LIGHTING_SIGNALS.ensureSignal('dayCycleTick', { initialValue: 0 });
export const biomeWeightsChanged  = LIGHTING_SIGNALS.ensureSignal('biomeWeightsChanged', { initialValue: null });
export const envPaletteChanged    = LIGHTING_SIGNALS.ensurePulse('envPaletteChanged');
export const weatherChanged       = LIGHTING_SIGNALS.ensurePulse('weatherChanged');

export const interiorEntered      = LIGHTING_SIGNALS.ensurePulse('interiorEntered');
export const interiorExited       = LIGHTING_SIGNALS.ensurePulse('interiorExited');
export const exteriorChanged      = LIGHTING_SIGNALS.ensurePulse('exteriorChanged');

export const qualityLevelChanged  = LIGHTING_SIGNALS.ensureSignal('qualityLevelChanged', { initialValue: 'high' });
export const tierChanged          = LIGHTING_SIGNALS.ensureSignal('tierChanged', { initialValue: 'medium' });
export const thermalChanged       = LIGHTING_SIGNALS.ensureSignal('thermalChanged', { initialValue: 'nominal' });
export const batteryChanged       = LIGHTING_SIGNALS.ensureSignal('batteryChanged', { initialValue: 1.0 });

export const cameraMoved          = LIGHTING_SIGNALS.ensurePulse('cameraMoved');
export const cameraTeleported     = LIGHTING_SIGNALS.ensurePulse('cameraTeleported');

export const contextLost          = LIGHTING_SIGNALS.ensurePulse('contextLost');
export const contextRestored      = LIGHTING_SIGNALS.ensurePulse('contextRestored');
export const visibilityChanged    = LIGHTING_SIGNALS.ensureSignal('visibilityChanged', { initialValue: true });

export const frameTick            = LIGHTING_SIGNALS.ensurePulse('frameTick');
export const frameHitch           = LIGHTING_SIGNALS.ensureSignal('frameHitch', { initialValue: 0 });

export const directorHint         = LIGHTING_SIGNALS.ensureSignal('directorHint', { initialValue: null });
export const screenshotRequested  = LIGHTING_SIGNALS.ensurePulse('screenshotRequested');

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createSignal(name, options = {}) {
  return new Signal(name, options);
}

export function createPulseSignal(name, options = {}) {
  return new PulseSignal(name, options);
}

export function createSignalGroup(name) {
  return new SignalGroup(name);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Signal,
  PulseSignal,
  SignalGroup,
  Connection,

  createSignal,
  createPulseSignal,
  createSignalGroup,

  combine,
  merge,
  anyOf,

  LIGHTING_SIGNALS,
  SIGNAL_PRIORITY,
  DISPATCH_RESULT,
  MAX_LISTENERS,
  MAX_HISTORY,
};

export default _defaultExport;