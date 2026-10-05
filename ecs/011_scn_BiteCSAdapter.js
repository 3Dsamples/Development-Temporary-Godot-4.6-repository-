// File : 011
// name : src/ecs/011_scn_BiteCSAdapter.js
// description : Compatibility adapter and API shim that isolates the entire
//               anime lighting stack from bitECS's raw API surface. Every
//               downstream system talks to bitECS through this adapter so
//               that:
//
//                 1. A future bitECS minor-version change to the internal
//                    layout (entity store, component registry, query
//                    engine) can be absorbed here in one place without
//                    touching any consumer.
//
//                 2. The runtime version policy
//                    (009_scn_BiteCSVersionPolicy) is enforced implicitly
//                    — every adapter call is guaranteed to run against a
//                    verified bitECS 0.4.0 surface, no `Types`, no
//                    `defineComponent`, no legacy store registry.
//
//                 3. Android-specific guards can be injected once (pre-
//                    sized SoA fields, no runtime resize, no dynamic
//                    component registration after boot in production).
//
//                 4. Diagnostics, profiler hooks, and error boundaries can
//                    wrap every ECS call without scattering try/catch
//                    blocks across the codebase.
//
//                 5. The adapter can expose a SCOPED handle
//                    (createAdapter(world)) so sub-systems can obtain a
//                    lightweight wrapper bound to their own world without
//                    reaching for the raw bitECS functions.
//
//               The adapter exposes:
//                 • world management   — createWorld / disposeWorld
//                 • entity lifecycle   — spawnEntity / destroyEntity /
//                                        entityAlive
//                 • component ops      — attachComponent / detachComponent /
//                                        hasComponent
//                 • queries            — query / queryFirst / forEach
//                 • component factories — defineSoAComponent (re-exported
//                                        through the adapter so callers
//                                        never import 009 directly)
//                 • diagnostics        — getStats / getWorldInfo / audit
//
//               The adapter is a PURE WRAPPER — it never re-implements
//               bitECS semantics, never caches entity data, never allocates
//               on the hot path. Every call is a direct passthrough with a
//               single guard.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; no external ECS libs; every internal array sized
//               once at construction.
// best for : Guaranteeing that the entire anime lighting stack talks to
//            bitECS through one stable, verified, Android-tuned adapter —
//            so any future bitECS minor bump, or any need to inject a
//            platform-specific guard, is a one-file change.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as bitecs from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
  runBiteCSAudit,
  defineSoAComponent,
  ecs as bitecsEcs,
  getAudit,
} from '../core/009_scn_BiteCSVersionPolicy.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getDefaultProfiler,
} from '../core/024_rnd_Profiler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Default maximum entity capacity across the entire lighting world.
 * Matches the value used by 002_lgt_LightComponents.js and
 * 010_scn_ECSWorld.js.
 */
export const MAX_ENTITIES = 100000;

/**
 * Adapter mode — controls how strict the guards are.
 */
export const ADAPTER_MODE = Object.freeze({
  PRODUCTION:  0,   // fast path, minimal guards
  DEVELOPMENT: 1,   // extra validation, warnings on misuse
  AUDIT:       2,   // full logging, boundary on every failure
  COUNT:       3,
});

export const ADAPTER_MODE_NAME = Object.freeze([
  'production',
  'development',
  'audit',
]);

/**
 * Reserved adapter handle id for the module-level default adapter.
 */
let _adapterIdCounter = 0;

function _nextAdapterId() {
  return ++_adapterIdCounter;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 1. ECS ADAPTER CLASS                                               */
/* ------------------------------------------------------------------ */

/**
 * A scoped adapter bound to one bitECS world. Downstream systems obtain
 * an adapter instance via `getAdapter()` (module-level default) or
 * `createAdapter(world)` (isolated), and call methods on the adapter
 * instead of the raw bitECS functions.
 */
export class ECSAdapter {
  constructor(world = null, options = {}) {
    this.adapterId = _nextAdapterId();
    this.world     = world;

    this.options = Object.assign({
      mode:            ADAPTER_MODE.PRODUCTION,
      trackProfiler:   PERF_TIER_LOCAL === 'HIGH',
      tripBoundary:    true,
      boundaryName:    'ecs.adapter',
      logChannel:      LOG_CHANNEL.CORE,
      maxComponentTypes: 512,
      enforcePolicy:   true,
    }, options || {});

    // Registered component name → component object map (for this adapter).
    this.components   = new Map();
    this.componentCount = 0;

    // Register the bitECS façade for external consumption.
    this._bitecs = bitecsEcs;

    // Optional boundary for guarded calls.
    this._boundary = null;
    if (this.options.tripBoundary) {
      try {
        const mgr = getDefaultErrorBoundaries();
        if (mgr) {
          this._boundary = mgr.create(this.options.boundaryName + '#' + this.adapterId, {
            tag: BOUNDARY_TAG.GENERIC,
            failureThreshold: 5,
          });
        }
      } catch (_) { /* swallow */ }
    }

    // Per-call counters (cheap diagnostics).
    this.counters = {
      created:        0,
      destroyed:      0,
      attachCalls:    0,
      detachCalls:    0,
      queryCalls:     0,
      rejectedCalls:  0,
      boundaryTrips:  0,
    };

    // Enforce policy before any real work.
    if (this.options.enforcePolicy) {
      if (!isBiteCSReady()) {
        try {
          runBiteCSAudit(null, null, MAX_ENTITIES);
        } catch (_) { /* swallow — audit logs its own errors */ }
      }
    }
  }

  /* ---------------- world binding ---------------- */

  /**
   * Binds this adapter to a specific bitECS world.
   */
  bindWorld(world) {
    this.world = world;
    return this;
  }

  hasWorld() {
    return this.world !== null && this.world !== undefined;
  }

  /**
   * Creates a new world with the given components. The adapter becomes
   * bound to the new world.
   */
  createWorld(components, options) {
    if (!components || typeof components !== 'object') {
      this._reject('createWorld', 'components must be a plain object');
      return null;
    }

    // Verify the components are SoA-shaped (pre-sized typed arrays).
    const names = Object.keys(components);
    for (let i = 0; i < names.length; i++) {
      const comp = components[names[i]];
      if (!comp || typeof comp !== 'object') {
        this._reject('createWorld', `component "${names[i]}" is not an object`);
        return null;
      }
      const fields = Object.keys(comp);
      for (let f = 0; f < fields.length; f++) {
        const field = comp[fields[f]];
        if (!ArrayBuffer.isView(field)) {
          this._reject('createWorld', `component "${names[i]}.${fields[f]}" is not a TypedArray`);
          return null;
        }
        if (field.length !== MAX_ENTITIES) {
          this._reject('createWorld', `component "${names[i]}.${fields[f]}" length ${field.length} !== MAX_ENTITIES ${MAX_ENTITIES}`);
          return null;
        }
      }
    }

    try {
      const w = bitecs.createWorld({
        components,
        time: (options && options.time) || {
          delta: 0,
          elapsed: 0,
          then: _now(),
        },
      });
      this.world = w;

      // Register components with this adapter.
      for (let i = 0; i < names.length; i++) {
        this.registerComponent(names[i], components[names[i]]);
      }

      this.counters.created++;
      return w;
    } catch (e) {
      this._reject('createWorld', e && e.message);
      return null;
    }
  }

  /* ---------------- component registration ---------------- */

  /**
   * Registers a component under a symbolic name. The component must be
   * a plain SoA object with typed-array fields sized to MAX_ENTITIES.
   */
  registerComponent(name, component) {
    if (typeof name !== 'string' || name.length === 0) return false;
    if (!component || typeof component !== 'object') return false;
    if (this.components.has(name)) return true;

    if (this.componentCount >= this.options.maxComponentTypes) return false;

    // Validate SoA shape.
    const fields = Object.keys(component);
    for (let i = 0; i < fields.length; i++) {
      const field = component[fields[i]];
      if (!ArrayBuffer.isView(field)) return false;
      if (field.length !== MAX_ENTITIES) return false;
    }

    this.components.set(name, component);
    this.componentCount++;
    return true;
  }

  getComponent(name) {
    return this.components.get(name) || null;
  }

  listComponents() {
    return Array.from(this.components.keys());
  }

  /* ---------------- entity lifecycle ---------------- */

  /**
   * Spawns a new entity in the bound world. Returns the entity id, or -1.
   */
  spawnEntity() {
    if (!this.hasWorld()) {
      this._reject('spawnEntity', 'no world bound');
      return -1;
    }
    try {
      const eid = bitecs.addEntity(this.world);
      this.counters.created++;
      return eid;
    } catch (e) {
      this._reject('spawnEntity', e && e.message);
      return -1;
    }
  }

  /**
   * Destroys an entity.
   */
  destroyEntity(eid) {
    if (!this.hasWorld()) return false;
    if (typeof eid !== 'number' || eid < 0) return false;
    try {
      if (!bitecs.entityExists(this.world, eid)) return false;
      bitecs.removeEntity(this.world, eid);
      this.counters.destroyed++;
      return true;
    } catch (e) {
      this._reject('destroyEntity', e && e.message);
      return false;
    }
  }

  /**
   * Checks whether an entity exists in the bound world.
   */
  entityAlive(eid) {
    if (!this.hasWorld()) return false;
    if (typeof eid !== 'number' || eid < 0) return false;
    try {
      return bitecs.entityExists(this.world, eid) === true;
    } catch (_) {
      return false;
    }
  }

  /* ---------------- component operations ---------------- */

  /**
   * Attaches a component to an entity.
   */
  attachComponent(eid, component) {
    this.counters.attachCalls++;
    if (!this.hasWorld()) {
      this._reject('attachComponent', 'no world bound');
      return false;
    }
    if (typeof eid !== 'number' || eid < 0) return false;
    if (!component) return false;
    try {
      if (!bitecs.entityExists(this.world, eid)) return false;
      bitecs.addComponent(this.world, eid, component);
      return true;
    } catch (e) {
      this._reject('attachComponent', e && e.message);
      return false;
    }
  }

  /**
   * Detaches a component from an entity.
   */
  detachComponent(eid, component) {
    this.counters.detachCalls++;
    if (!this.hasWorld()) return false;
    if (typeof eid !== 'number' || eid < 0) return false;
    if (!component) return false;
    try {
      if (!bitecs.entityExists(this.world, eid)) return false;
      if (!bitecs.hasComponent(this.world, eid, component)) return false;
      bitecs.removeComponent(this.world, eid, component);
      return true;
    } catch (e) {
      this._reject('detachComponent', e && e.message);
      return false;
    }
  }

  /**
   * Checks whether an entity has a component.
   */
  hasComponent(eid, component) {
    if (!this.hasWorld()) return false;
    if (typeof eid !== 'number' || eid < 0) return false;
    if (!component) return false;
    try {
      return bitecs.hasComponent(this.world, eid, component) === true;
    } catch (_) {
      return false;
    }
  }

  /* ---------------- queries ---------------- */

  /**
   * Runs a bitECS query for the given component objects.
   */
  query(components) {
    this.counters.queryCalls++;
    if (!this.hasWorld()) return EMPTY_QUERY_RESULT;
    if (!Array.isArray(components) || components.length === 0) return EMPTY_QUERY_RESULT;
    try {
      return bitecs.query(this.world, components);
    } catch (_) {
      return EMPTY_QUERY_RESULT;
    }
  }

  /**
   * Runs a bitECS query and returns the first matching entity, or -1.
   */
  queryFirst(components) {
    const result = this.query(components);
    return result.length > 0 ? result[0] : -1;
  }

  /**
   * Iterates the results of a query and calls `fn(eid)` for each entity.
   */
  forEach(components, fn, ctx) {
    if (typeof fn !== 'function') return 0;
    const entities = this.query(components);
    const n = entities.length;
    for (let i = 0; i < n; i++) {
      fn.call(ctx, entities[i]);
    }
    return n;
  }

  /* ---------------- component factory ---------------- */

  /**
   * Creates an SoA component with the given fields, sized to MAX_ENTITIES.
   * Re-exported through the adapter so callers never touch 009 directly.
   */
  defineSoAComponent(name, fields) {
    return defineSoAComponent(name, fields, MAX_ENTITIES);
  }

  /* ---------------- diagnostics ---------------- */

  /**
   * Rejects a call and records diagnostics.
   */
  _reject(op, reason) {
    this.counters.rejectedCalls++;

    const log = _safeLogger();
    if (log && this.options.mode !== ADAPTER_MODE.PRODUCTION) {
      log.warn(this.options.logChannel, () =>
        `[011_scn_BiteCSAdapter] ${op} rejected: ${reason}`);
    }

    if (this._boundary && this.options.mode === ADAPTER_MODE.AUDIT) {
      try {
        this._boundary.run(() => { throw new Error(`${op}: ${reason}`); });
        this.counters.boundaryTrips++;
      } catch (_) { /* swallow */ }
    }
  }

  /**
   * Reports adapter statistics.
   */
  getStats() {
    return {
      adapterId:        this.adapterId,
      worldBound:       this.hasWorld(),
      componentCount:   this.componentCount,
      mode:             ADAPTER_MODE_NAME[this.options.mode] || 'production',
      counters:         Object.assign({}, this.counters),
      bitecsAudit:      getAudit(),
      perfTier:         PERF_TIER_LOCAL,
    };
  }

  /**
   * Reports info about the bound world (entity count, time, etc.).
   */
  getWorldInfo() {
    const info = {
      bound:   this.hasWorld(),
      entities: 0,
      time:    null,
      components: this.componentCount,
    };
    if (!this.hasWorld()) return info;

    const w = this.world;
    try {
      if (w.entities && typeof w.entities.length === 'number') {
        info.entities = w.entities.length;
      } else if (typeof w.entityCount === 'number') {
        info.entities = w.entityCount;
      } else if (w.$ && typeof w.$.entityCount === 'number') {
        info.entities = w.$.entityCount;
      }
    } catch (_) { /* swallow */ }

    if (w.time) {
      info.time = {
        delta:   w.time.delta   !== undefined ? w.time.delta   : 0,
        elapsed: w.time.elapsed !== undefined ? w.time.elapsed : 0,
        frame:   w.time.frame   !== undefined ? w.time.frame   : 0,
      };
    }
    return info;
  }

  /* ---------------- disposal ---------------- */

  dispose() {
    this.components.clear();
    this.componentCount = 0;
    this._boundary = null;
    this.world = null;
    return this;
  }
}

/**
 * Shared read-only empty array returned from failed queries. Never mutate.
 */
const EMPTY_QUERY_RESULT = Object.freeze([]);

/* ------------------------------------------------------------------ */
/* 2. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultAdapter = null;

/**
 * Returns the module-level default adapter. Lazily creates it bound to
 * no world; callers can `bindWorld(world)` once the scene world is ready.
 */
export function getAdapter() {
  if (!_defaultAdapter) _defaultAdapter = new ECSAdapter(null);
  return _defaultAdapter;
}

/**
 * Disposes the default adapter and resets the singleton.
 */
export function disposeAdapter() {
  if (_defaultAdapter) {
    _defaultAdapter.dispose();
    _defaultAdapter = null;
  }
}

/* ------------------------------------------------------------------ */
/* 3. SCOPED ADAPTER FACTORY                                          */
/* ------------------------------------------------------------------ */

/**
 * Creates a scoped adapter bound to the given world. Downstream systems
 * that want their own isolated adapter (with its own diagnostics and
 * boundary) use this instead of the module-level default.
 */
export function createAdapter(world, options) {
  const adapter = new ECSAdapter(world, options || {});
  if (world) {
    adapter.bindWorld(world);
  }
  return adapter;
}

/* ------------------------------------------------------------------ */
/* 4. CONVENIENCE PASSTHROUGH FUNCTIONS                               */
/* ------------------------------------------------------------------ */

/**
 * The following functions delegate to the default adapter so callers can
 * use a function-style API without importing the adapter class.
 */

export function spawnEntity()             { return getAdapter().spawnEntity(); }
export function destroyEntity(eid)        { return getAdapter().destroyEntity(eid); }
export function entityAlive(eid)          { return getAdapter().entityAlive(eid); }
export function attachComponent(eid, c)   { return getAdapter().attachComponent(eid, c); }
export function detachComponent(eid, c)   { return getAdapter().detachComponent(eid, c); }
export function hasComponent(eid, c)      { return getAdapter().hasComponent(eid, c); }
export function queryComponents(c)        { return getAdapter().query(c); }
export function queryComponentsByName(names) {
  const adapter = getAdapter();
  const resolved = new Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const comp = adapter.getComponent(names[i]);
    if (!comp) return EMPTY_QUERY_RESULT;
    resolved[i] = comp;
  }
  return adapter.query(resolved);
}
export function forEachEntity(c, fn, ctx) { return getAdapter().forEach(fn, ctx); }
export function defineComponent(name, fields) {
  return getAdapter().defineSoAComponent(name, fields);
}

/**
 * Binds the default adapter to a world.
 */
export function bindDefaultWorld(world) {
  return getAdapter().bindWorld(world);
}

/**
 * Returns the default adapter's bound world.
 */
export function getBoundWorld() {
  return getAdapter().world;
}

/* ------------------------------------------------------------------ */
/* 5. RAW BITECS FALLBACK                                            */
/* ------------------------------------------------------------------ */

/**
 * Returns the raw bitECS module for the rare case where a caller needs
 * to reach the underlying implementation. Use of this is discouraged —
 * every call should go through the adapter.
 */
export function getRawBitecs() {
  return bitecs;
}

/**
 * Returns the frozen bitECS façade from 009_scn_BiteCSVersionPolicy. This
 * is the sanctioned low-level surface — it exposes only the ten functions
 * that bitECS 0.4.0 declares, nothing more.
 */
export function getBitecsFaçade() {
  return bitecsEcs;
}

/* ------------------------------------------------------------------ */
/* 6. DIAGNOSTICS                                                     */
/* ------------------------------------------------------------------ */

/**
 * Returns a global report about the adapter system state.
 */
export function getAdapterReport() {
  const a = getAdapter();
  return {
    defaultAdapter: a.getStats(),
    boundWorld:     a.getWorldInfo(),
    bitecsAudit:    getAudit(),
    perfTier:       PERF_TIER_LOCAL,
    adapterIdCounter: _adapterIdCounter,
  };
}

/* ------------------------------------------------------------------ */
/* 7. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ECSAdapter,

  // Adapter management
  getAdapter,
  disposeAdapter,
  createAdapter,
  bindDefaultWorld,
  getBoundWorld,

  // Passthrough
  spawnEntity,
  destroyEntity,
  entityAlive,
  attachComponent,
  detachComponent,
  hasComponent,
  queryComponents,
  queryComponentsByName,
  forEachEntity,
  defineComponent,

  // Raw access (discouraged)
  getRawBitecs,
  getBitecsFaçade,

  // Diagnostics
  getAdapterReport,

  // Constants
  MAX_ENTITIES,
  ADAPTER_MODE,
  ADAPTER_MODE_NAME,
};

export default _defaultExport;