// File : 029
// name : src/core/029_rnd_Disposable.js
// description : Deterministic disposal and teardown primitive for the anime
//               lighting stack on Android mobile. Every lighting subsystem
//               (006_lgt_LightManager through 380_lgt_lights) that owns GPU
//               or CPU resources registers them with a Disposable so teardown
//               is guaranteed — no orphan render targets, no leaked worker
//               ports, no dangling event subscriptions, no retained closures.
//
//               This is the counterpart to 014_rnd_ResourceRegistry.js:
//               the Registry tracks live GPU resources and reference counts;
//               Disposable tracks LIFECYCLE ORDER — the sequence in which
//               things must be torn down (listeners first, then pools,
//               then render targets, then renderer itself) so nothing
//               touches a freed handle during teardown.
//
//               Design:
//                 • Disposable              — base class with `dispose()`,
//                                             `isDisposed`, and a linked
//                                             list of child Disposables that
//                                             get disposed in reverse-
//                                             registration order (LIFO).
//                 • DisposableBag           — lightweight collection of
//                                             teardown callbacks executed in
//                                             reverse-registration order;
//                                             used for ad-hoc resources that
//                                             don't warrant a full class.
//                 • DisposableScope         — scoped lifetime (e.g. a chunk,
//                                             a room, a LOD level); child
//                                             resources auto-dispose when the
//                                             scope ends.
//                 • Auto-dispose on GC      — optional FinalizationRegistry
//                                             wrapper to catch missed
//                                             disposals in dev builds.
//                 • Idempotent: calling dispose() twice is safe and cheap.
//                 • Error containment: one child's throw never prevents
//                   siblings from disposing — every teardown is wrapped.
//                 • Per-disposable debug id so leaks can be traced to the
//                   exact subsystem that created them.
//                 • Zero per-frame allocations: dispose is a cold path, but
//                   the bag's `run()` walks a pre-allocated parallel array.
//                 • Integration with 026_rnd_Logger.js for warn/error
//                   surfacing and with 024_rnd_Profiler.js for teardown
//                   timing on HIGH tier.
//                 • Integration with 014_rnd_ResourceRegistry.js: any
//                   registered GPU resource handle can be attached to a
//                   Disposable so it is force-released at teardown.
//                 • Integration with 028_rnd_Signal.js: connection tokens
//                   can be attached so signals disconnect cleanly.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external teardown libs; every internal array sized
//               once at construction.
// best for : Guaranteeing deterministic teardown of the entire anime
//            lighting stack — context loss recovery, chunk unload, room
//            exit, quality downgrade, module hot-reload, and full engine
//            shutdown all flow through the same Disposable tree so nothing
//            is left behind on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

import {
  getDefaultResourceRegistry,
} from './014_rnd_ResourceRegistry.js';

import {
  getDefaultProfiler,
} from './024_rnd_Profiler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_BAG_ENTRIES =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 48 :
                                 32;

export const MAX_SCOPE_CHILDREN =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 24 :
                                 16;

export const DISPOSE_STATE = Object.freeze({
  ALIVE:     0,
  DISPOSING: 1,
  DISPOSED:  2,
  FAILED:    3,
});

export const TEARDOWN_ORDER = Object.freeze({
  FIRST:    0,   // listeners, signal connections
  EARLY:    1,   // pools (release back to free list)
  NORMAL:   2,   // render targets, buffers
  LATE:     3,   // materials, geometries
  LAST:     4,   // GPU resources that others reference
});

const FINALIZER_ENABLED = (typeof FinalizationRegistry !== 'undefined') && PERF_TIER_LOCAL === 'HIGH';

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _disposableIdCounter = 0;

function _nextDisposableId() {
  return ++_disposableIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 2. DISPOSABLE BAG                                                  */
/* ------------------------------------------------------------------ */

/**
 * A fixed-capacity collection of teardown callbacks executed in reverse
 * registration order. Every entry is a small record with:
 *   fn      — callback
 *   ctx     — optional context (bound this)
 *   label   — human-readable name for debugging
 *   order   — teardown order rank
 *
 * Entries are dispatched grouped by `order` (FIRST → LAST), and within a
 * group in reverse registration order (LIFO). This lets a bag mix
 * listener disconnections (FIRST) with render target releases (NORMAL)
 * and geometry disposals (LATE) in a single bag without ambiguity.
 */
export class DisposableBag {
  constructor(label = 'bag') {
    this.label       = label;
    this.capacity    = MAX_BAG_ENTRIES;

    this.fn          = new Array(this.capacity).fill(null);
    this.ctx         = new Array(this.capacity).fill(null);
    this.name        = new Array(this.capacity).fill(null);
    this.order       = new Uint8Array(this.capacity);
    this.count       = 0;

    this._state      = DISPOSE_STATE.ALIVE;
    this._disposedCount = 0;
    this._failedCount   = 0;
  }

  /**
   * Add a teardown callback. Returns true on success, false if the bag
   * is full or disposed.
   */
  add(fn, ctx, name, order = TEARDOWN_ORDER.NORMAL) {
    if (this._state === DISPOSE_STATE.DISPOSED) return false;
    if (this._state === DISPOSE_STATE.DISPOSING) return false;
    if (typeof fn !== 'function') return false;
    if (this.count >= this.capacity) {
      const log = _safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE, `[029_rnd_Disposable] bag "${this.label}" full (${this.capacity})`);
      return false;
    }
    const i = this.count++;
    this.fn[i]    = fn;
    this.ctx[i]   = ctx || null;
    this.name[i]  = name || null;
    this.order[i] = (order | 0);
    return true;
  }

  /**
   * Convenience: add a resource with a `.dispose()` method.
   */
  addDisposable(obj, name, order = TEARDOWN_ORDER.LATE) {
    if (!obj || typeof obj.dispose !== 'function') return false;
    return this.add(obj.dispose, obj, name || obj.constructor.name || 'disposable', order);
  }

  /**
   * Convenience: add a Signal.Connection token so signals disconnect.
   */
  addConnection(connection, name) {
    if (!connection || typeof connection.disconnect !== 'function') return false;
    return this.add(connection.disconnect, connection, name || 'connection', TEARDOWN_ORDER.FIRST);
  }

  /**
   * Convenience: add a resource handle (from 014_rnd_ResourceRegistry) so
   * the registry force-releases it at teardown.
   */
  addResourceHandle(handle, name) {
    if (typeof handle !== 'number' || handle <= 0) return false;
    return this.add(_releaseResourceFn, handle, name || ('resource#' + handle), TEARDOWN_ORDER.NORMAL);
  }

  /**
   * Run all teardown callbacks in the correct order. Idempotent —
   * calling twice on a disposed bag is a no-op.
   */
  run() {
    if (this._state === DISPOSE_STATE.DISPOSED) return true;
    if (this._state === DISPOSE_STATE.DISPOSING) return false;

    this._state = DISPOSE_STATE.DISPOSING;

    // Run order groups in ascending order.
    for (let o = TEARDOWN_ORDER.FIRST; o <= TEARDOWN_ORDER.LAST; o++) {
      // Walk the registered entries in reverse registration order,
      // dispatching only those in the current order group.
      for (let i = this.count - 1; i >= 0; i--) {
        if (this.order[i] !== o) continue;
        const fn  = this.fn[i];
        const ctx = this.ctx[i];
        if (typeof fn !== 'function') continue;

        try {
          if (ctx) fn.call(ctx);
          else fn();
          this._disposedCount++;
        } catch (e) {
          this._failedCount++;
          const log = _safeLogger();
          if (log) log.error(LOG_CHANNEL.CORE, `[029_rnd_Disposable] teardown "${this.name[i] || 'anonymous'}" in bag "${this.label}" threw: ${e && e.message}`);
        }

        // Clear eagerly so a second run can't double-fire.
        this.fn[i]   = null;
        this.ctx[i]  = null;
        this.name[i] = null;
      }
    }

    this.count = 0;
    this._state = DISPOSE_STATE.DISPOSED;
    return true;
  }

  get state()         { return this._state; }
  get isDisposed()    { return this._state === DISPOSE_STATE.DISPOSED; }
  get disposedCount() { return this._disposedCount; }
  get failedCount()   { return this._failedCount; }

  getStats() {
    return {
      label:         this.label,
      capacity:      this.capacity,
      count:         this.count,
      state:         this._state,
      disposedCount: this._disposedCount,
      failedCount:   this._failedCount,
    };
  }

  reset() {
    for (let i = 0; i < this.count; i++) {
      this.fn[i] = null;
      this.ctx[i] = null;
      this.name[i] = null;
      this.order[i] = 0;
    }
    this.count = 0;
    this._disposedCount = 0;
    this._failedCount = 0;
    this._state = DISPOSE_STATE.ALIVE;
    return this;
  }
}

function _releaseResourceFn(handle) {
  const reg = getDefaultResourceRegistry();
  if (reg) reg.forceRelease(handle);
}

/* ------------------------------------------------------------------ */
/* 3. DISPOSABLE (base class)                                         */
/* ------------------------------------------------------------------ */

/**
 * Base class for any object that owns resources and must be torn down
 * deterministically.
 *
 *   class ShadowAtlas extends Disposable {
 *     constructor() {
 *       super('shadow_atlas');
 *       // acquire resources, register them:
 *       this.register(someRenderTarget, 'rt');
 *       this.registerListener(someSignal.connect(...));
 *     }
 *     onDispose() {
 *       // Optional: hook called AFTER bag teardown.
 *     }
 *   }
 */
export class Disposable {
  constructor(label) {
    this.disposableId  = _nextDisposableId();
    this.label         = label || ('disposable_' + this.disposableId);
    this._state        = DISPOSE_STATE.ALIVE;

    // Bag of teardown callbacks.
    this._bag          = new DisposableBag(this.label);

    // Child disposables (LIFO reverse-registration).
    this._children     = new Array(MAX_SCOPE_CHILDREN).fill(null);
    this._childCount   = 0;

    // Timestamps.
    this._createdAtMs  = _now();
    this._disposedAtMs = 0;

    // FinalizationRegistry hookup (HIGH tier, dev-only safety net).
    this._finalizerToken = null;
    if (FINALIZER_ENABLED) {
      _registerWithFinalizer(this);
    }
  }

  /* ---------------- registration ---------------- */

  /**
   * Register an arbitrary teardown callback.
   */
  register(fn, ctx, name, order) {
    return this._bag.add(fn, ctx, name, order);
  }

  /**
   * Register an object with a `.dispose()` method.
   */
  registerDisposable(obj, name, order = TEARDOWN_ORDER.LATE) {
    return this._bag.addDisposable(obj, name, order);
  }

  /**
   * Register a Signal.Connection so it disconnects at teardown.
   */
  registerConnection(connection, name) {
    return this._bag.addConnection(connection, name);
  }

  /**
   * Register a resource handle from the registry.
   */
  registerResource(handle, name) {
    return this._bag.addResourceHandle(handle, name);
  }

  /**
   * Register a named GPU resource (Texture, RT, Geometry, Material,
   * BufferAttribute). The bag will call `.dispose()` on the resource if it
   * exists; if a registry handle was created for it, register the handle
   * separately.
   */
  registerGPU(obj, name) {
    return this._bag.addDisposable(obj, name, TEARDOWN_ORDER.LATE);
  }

  /**
   * Register a child Disposable that will be disposed when this one is.
   * Children are disposed BEFORE the parent's own bag entries so the
   * parent still has valid resources while cleaning up children.
   */
  registerChild(child) {
    if (!child || typeof child.dispose !== 'function') return false;
    if (this._childCount >= MAX_SCOPE_CHILDREN) {
      const log = _safeLogger();
      if (log) log.warn(LOG_CHANNEL.CORE, `[029_rnd_Disposable] "${this.label}" child cap reached (${MAX_SCOPE_CHILDREN})`);
      return false;
    }
    this._children[this._childCount++] = child;
    return true;
  }

  /* ---------------- lifecycle ---------------- */

  dispose() {
    if (this._state === DISPOSE_STATE.DISPOSED) return true;
    if (this._state === DISPOSE_STATE.DISPOSING) return false;

    this._state = DISPOSE_STATE.DISPOSING;

    const t0 = _now();

    // 1. Dispose children FIRST (reverse registration order). This
    //    guarantees the parent is still alive while its children tear down.
    for (let i = this._childCount - 1; i >= 0; i--) {
      const child = this._children[i];
      if (!child) continue;
      try {
        child.dispose();
      } catch (e) {
        const log = _safeLogger();
        if (log) log.error(LOG_CHANNEL.CORE, `[029_rnd_Disposable] child #${i} of "${this.label}" threw: ${e && e.message}`);
      }
      this._children[i] = null;
    }
    this._childCount = 0;

    // 2. Run the bag.
    this._bag.run();

    // 3. Optional subclass hook.
    try {
      if (typeof this.onDispose === 'function') this.onDispose();
    } catch (e) {
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE, `[029_rnd_Disposable] onDispose hook of "${this.label}" threw: ${e && e.message}`);
    }

    // 4. Record timestamps.
    this._disposedAtMs = _now();

    // 5. Finalizer cleanup.
    if (this._finalizerToken && FINALIZER_ENABLED) {
      _unregisterWithFinalizer(this._finalizerToken);
      this._finalizerToken = null;
    }

    this._state = DISPOSE_STATE.DISPOSED;

    // Optionally report teardown cost to the profiler on HIGH tier.
    if (PERF_TIER_LOCAL === 'HIGH') {
      try {
        const p = getDefaultProfiler();
        if (p) p.mark('dispose:' + this.label);
      } catch (_) { /* swallow */ }
    }

    return true;
  }

  /* ---------------- state ---------------- */

  get state()          { return this._state; }
  get isDisposed()     { return this._state === DISPOSE_STATE.DISPOSED; }
  get isDisposing()    { return this._state === DISPOSE_STATE.DISPOSING; }
  get createdAtMs()    { return this._createdAtMs; }
  get disposedAtMs()   { return this._disposedAtMs; }
  get ageMs()          {
    const end = this._disposedAtMs > 0 ? this._disposedAtMs : _now();
    return end - this._createdAtMs;
  }

  getStats() {
    return {
      id:           this.disposableId,
      label:        this.label,
      state:        this._state,
      bag:          this._bag.getStats(),
      children:     this._childCount,
      ageMs:        this.ageMs,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 4. FINALIZATION REGISTRY (dev safety net)                          */
/* ------------------------------------------------------------------ */

let _finalizer = null;
const _finalizerTokens = new Map();

function _ensureFinalizer() {
  if (!FINALIZER_ENABLED) return null;
  if (_finalizer) return _finalizer;
  try {
    _finalizer = new FinalizationRegistry((token) => {
      const info = _finalizerTokens.get(token);
      if (!info) return;
      _finalizerTokens.delete(token);
      if (info.state !== DISPOSE_STATE.DISPOSED) {
        const log = _safeLogger();
        if (log) log.warn(LOG_CHANNEL.CORE, `[029_rnd_Disposable] leaked: "${info.label}" was GC'd without dispose()`);
      }
    });
    return _finalizer;
  } catch (_) {
    return null;
  }
}

function _registerWithFinalizer(disposable) {
  const f = _ensureFinalizer();
  if (!f) return;
  const token = disposable.disposableId;
  _finalizerTokens.set(token, { label: disposable.label, state: disposable._state });
  try {
    f.register(disposable, token, disposable);
  } catch (_) {
    // Some environments reject certain value types — safe to ignore.
  }
}

function _unregisterWithFinalizer(token) {
  const f = _ensureFinalizer();
  if (!f) return;
  _finalizerTokens.delete(token);
  try { f.unregister(token); } catch (_) { /* swallow */ }
}

/* ------------------------------------------------------------------ */
/* 5. DISPOSABLE SCOPE                                                */
/* ------------------------------------------------------------------ */

/**
 * A scoped lifetime container. Used for chunk unload, room exit, LOD swap —
 * anywhere a group of resources must be released together.
 *
 *   const scope = new DisposableScope('chunk_4_2');
 *   scope.own(someGeometry);
 *   scope.own(someMaterial);
 *   scope.track(someConnection);
 *   // ...
 *   scope.end();  // disposes every owned resource in reverse order
 */
export class DisposableScope extends Disposable {
  constructor(label) {
    super(label || 'scope');
    this._ownedCount = 0;
  }

  /** Own a GPU resource; will be disposed in reverse registration order. */
  own(obj, name) {
    if (!obj || typeof obj.dispose !== 'function') return false;
    this.registerDisposable(obj, name, TEARDOWN_ORDER.LATE);
    this._ownedCount++;
    return true;
  }

  /** Track a Signal.Connection. */
  track(connection, name) {
    const ok = this.registerConnection(connection, name);
    if (ok) this._ownedCount++;
    return ok;
  }

  /** Track an arbitrary callback. */
  trackCallback(fn, ctx, name, order) {
    const ok = this.register(fn, ctx, name, order);
    if (ok) this._ownedCount++;
    return ok;
  }

  /** Attach a resource handle from the registry. */
  trackResource(handle, name) {
    const ok = this.registerResource(handle, name);
    if (ok) this._ownedCount++;
    return ok;
  }

  /** Alias for dispose() — reads better in scoped contexts. */
  end() {
    return this.dispose();
  }

  get ownedCount() { return this._ownedCount; }
}

/* ------------------------------------------------------------------ */
/* 6. DISPOSABLE REGISTRY (global tracker for debug)                  */
/* ------------------------------------------------------------------ */

/**
 * Optional global tracker so a debug tool can list all live Disposables.
 * Kept tiny and cold-path; the engine doesn't require it to function.
 */
export class DisposableRegistry {
  constructor() {
    this.map = new Map();
    this.capacity = PERF_TIER_LOCAL === 'HIGH' ? 512 : 256;
  }

  track(disposable) {
    if (!disposable || typeof disposable.disposableId !== 'number') return false;
    if (this.map.size >= this.capacity) return false;
    this.map.set(disposable.disposableId, disposable);
    return true;
  }

  untrack(disposable) {
    if (!disposable) return false;
    return this.map.delete(disposable.disposableId);
  }

  listLive() {
    const out = [];
    for (const d of this.map.values()) {
      if (!d.isDisposed) out.push(d.getStats());
    }
    return out;
  }

  count() {
    let n = 0;
    for (const d of this.map.values()) if (!d.isDisposed) n++;
    return n;
  }

  clear() { this.map.clear(); }
}

/* ------------------------------------------------------------------ */
/* 7. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _defaultRegistry = null;

export function getDisposableRegistry() {
  if (!_defaultRegistry) _defaultRegistry = new DisposableRegistry();
  return _defaultRegistry;
}

export function trackDisposable(disposable) {
  return getDisposableRegistry().track(disposable);
}

export function untrackDisposable(disposable) {
  return getDisposableRegistry().untrack(disposable);
}

export function listLiveDisposables() {
  return getDisposableRegistry().listLive();
}

/* ------------------------------------------------------------------ */
/* 8. CONVENIENCE FUNCTIONS                                           */
/* ------------------------------------------------------------------ */

/**
 * Creates a DisposableBag pre-loaded with the given entries.
 * Each entry is `{ fn, ctx, name, order }`.
 */
export function bagOf(label, entries) {
  const bag = new DisposableBag(label);
  if (Array.isArray(entries)) {
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      if (!e || typeof e.fn !== 'function') continue;
      bag.add(e.fn, e.ctx, e.name, e.order);
    }
  }
  return bag;
}

/**
 * Runs a batch of teardown functions in reverse order, swallowing errors.
 * Useful for ad-hoc cleanups that don't warrant a full bag.
 */
export function runTeardown(fns) {
  if (!Array.isArray(fns)) return 0;
  let n = 0;
  for (let i = fns.length - 1; i >= 0; i--) {
    const fn = fns[i];
    if (typeof fn !== 'function') continue;
    try { fn(); n++; }
    catch (e) {
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE, `[029_rnd_Disposable] teardown #${i} threw: ${e && e.message}`);
    }
  }
  return n;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Disposable,
  DisposableBag,
  DisposableScope,
  DisposableRegistry,

  bagOf,
  runTeardown,

  getDisposableRegistry,
  trackDisposable,
  untrackDisposable,
  listLiveDisposables,

  DISPOSE_STATE,
  TEARDOWN_ORDER,
  MAX_BAG_ENTRIES,
  MAX_SCOPE_CHILDREN,
};

export default _defaultExport;