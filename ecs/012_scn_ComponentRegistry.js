// File : 012
// name : src/ecs/012_scn_ComponentRegistry.js
// description : Central reflective registry for every component type in the
//               scene ECS world of the anime lighting stack. Where
//               010_scn_ECSWorld.js owns the world handle and
//               011_scn_BiteCSAdapter.js owns the API adapter, THIS module
//               owns the METADATA CATALOG: for every registered component it
//               stores its symbolic name, its category (light / shadow / gi /
//               ao / camera / scene / streaming / animation / weather / biome
//               / interior / exterior / material / post / debug / lod /
//               visibility / spatial), its owning subsystem id, its SoA field
//               list with each field's typed-array kind and element count,
//               its estimated byte cost, its aliases, its dependencies on
//               other components, and a per-registration generation counter.
//
//               The registry is the single source of truth for:
//                 • Discovering components by name at runtime.
//                 • Enumerating all components of a given category (e.g.
//                   "give me every shadow component").
//                 • Estimating GPU/CPU memory cost of a component.
//                 • Reflecting over a component's fields to auto-generate
//                   uniform bindings, worker transfer descriptors, or
//                   save/load serializers.
//                 • Detecting orphan or duplicate registrations.
//                 • Validating that a component has the required SoA shape
//                   before it is added to a world.
//                 • Producing per-subsystem component ownership reports so
//                   that a bulk dispose of a subsystem can enumerate exactly
//                   which components to detach and which arrays to release.
//
//               The registry does NOT own the component arrays themselves —
//               it holds references so callers can obtain the same component
//               object they registered. Every component object is expected
//               to be frozen or at least consistently shaped.
//
//               Structural contract per registered component:
//                 name           — unique string identifier
//                 category       — one of COMPONENT_CATEGORY
//                 subsystem      — symbolic subsystem id (e.g. 'lights',
//                                  'shadows', 'gi', 'ao', 'camera', ...)
//                 fields         — array of { name, kind, length, bytes }
//                 dependencies   — array of names of other components that
//                                  must be present on the same entity for
//                                  this component to be meaningful
//                 aliases        — array of alternative lookup names
//                 generation     — monotonic counter, bumped on re-register
//                 ref            — the component object itself
//
//               All lookups are O(1) via flat maps. Registration is O(fields)
//               because it walks the SoA fields to record metadata. No
//               allocations on the query path.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external ECS libs; every internal array sized once
//               at construction.
// best for : Guaranteeing that every subsystem in the anime lighting stack
//            can discover and introspect every component registered by any
//            other subsystem — without hard-coding imports, without
//            duplicated lists, and without drift between what a system
//            declares and what the rest of the engine believes exists.
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
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of components the registry can hold. Sized once.
 */
export const MAX_REGISTERED_COMPONENTS =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 384 :
                                 256;

/**
 * Maximum number of categories. Matches COMPONENT_CATEGORY.COUNT.
 */
export const MAX_CATEGORIES = 32;

/**
 * Maximum number of subsystems that can own components.
 */
export const MAX_SUBSYSTEMS = 64;

/**
 * Maximum alias chain length per component.
 */
export const MAX_ALIASES_PER_COMPONENT = 4;

/**
 * Maximum dependencies per component.
 */
export const MAX_DEPENDENCIES_PER_COMPONENT = 8;

/* ------------------------------------------------------------------ */
/* 1. ENUMS                                                           */
/* ------------------------------------------------------------------ */

/**
 * Component category — the top-level classification that downstream
 * systems filter by.
 */
export const COMPONENT_CATEGORY = Object.freeze({
  UNKNOWN:      0,
  LIGHT:        1,
  SHADOW:       2,
  GI:           3,
  AO:           4,
  CAMERA:       5,
  SCENE:        6,
  STREAMING:    7,
  ANIMATION:    8,
  WEATHER:      9,
  BIOME:       10,
  INTERIOR:    11,
  EXTERIOR:    12,
  MATERIAL:    13,
  POST:        14,
  DEBUG:       15,
  LOD:         16,
  VISIBILITY:  17,
  SPATIAL:     18,
  TRANSFORM:   19,
  PARTICLE:    20,
  PHYSICS:     21,
  AUDIO:       22,
  NETWORK:     23,
  UI:          24,
  INPUT:       25,
  GRAPH:       26,
  PROBE:       27,
  VOLUME:      28,
  COUNT:       29,
});

export const COMPONENT_CATEGORY_NAME = Object.freeze([
  'unknown',
  'light',
  'shadow',
  'gi',
  'ao',
  'camera',
  'scene',
  'streaming',
  'animation',
  'weather',
  'biome',
  'interior',
  'exterior',
  'material',
  'post',
  'debug',
  'lod',
  'visibility',
  'spatial',
  'transform',
  'particle',
  'physics',
  'audio',
  'network',
  'ui',
  'input',
  'graph',
  'probe',
  'volume',
]);

/**
 * Recognized typed-array kinds. Used to compute per-field byte cost and
 * to auto-generate serializers / worker transfer lists.
 */
export const FIELD_KIND = Object.freeze({
  UNKNOWN:    0,
  F32:        1,
  F64:        2,
  I32:        3,
  U32:        4,
  I16:        5,
  U16:        6,
  I8:         7,
  U8:         8,
  U8_CLAMP:   9,
  COUNT:     10,
});

export const FIELD_KIND_NAME = Object.freeze([
  'unknown',
  'float32',
  'float64',
  'int32',
  'uint32',
  'int16',
  'uint16',
  'int8',
  'uint8',
  'uint8clamped',
]);

export const FIELD_KIND_BYTES = Object.freeze([
  0, 4, 8, 4, 4, 2, 2, 1, 1, 1,
]);

/**
 * Recognized subsystem ids. Used to group components by owner for bulk
 * dispose / introspection.
 */
export const SUBSYSTEM = Object.freeze({
  UNKNOWN:     0,
  CORE:        1,
  LIGHTS:      2,
  SHADOWS:     3,
  GI:          4,
  AO:          5,
  CAMERA:      6,
  ENVIRONMENT: 7,
  INTERIOR:    8,
  EXTERIOR:    9,
  STREAMING:  10,
  MATERIAL:   11,
  POST:       12,
  DEBUG:      13,
  PARALLEL:   14,
  QUALITY:    15,
  PLATFORM:   16,
  COUNT:      17,
});

export const SUBSYSTEM_NAME = Object.freeze([
  'unknown',
  'core',
  'lights',
  'shadows',
  'gi',
  'ao',
  'camera',
  'environment',
  'interior',
  'exterior',
  'streaming',
  'material',
  'post',
  'debug',
  'parallel',
  'quality',
  'platform',
]);

/**
 * Registration result codes.
 */
export const REGISTER_RESULT = Object.freeze({
  OK:              0,
  INVALID_NAME:    1,
  INVALID_SHAPE:   2,
  CAPACITY_FULL:   3,
  DUPLICATE:       4,
  CATEGORY_FULL:   5,
});

export const REGISTER_RESULT_NAME = Object.freeze([
  'ok',
  'invalid_name',
  'invalid_shape',
  'capacity_full',
  'duplicate',
  'category_full',
]);

/* ------------------------------------------------------------------ */
/* 2. COMPONENT ENTRY                                                 */
/* ------------------------------------------------------------------ */

/**
 * Fixed record for a single registered component. Fields are reset on
 * reuse; the registry ring never resizes.
 */
export class ComponentEntry {
  constructor(index) {
    this.index        = index;
    this.name         = null;
    this.category     = COMPONENT_CATEGORY.UNKNOWN;
    this.subsystem    = SUBSYSTEM.UNKNOWN;

    // SoA field metadata (parallel arrays; each slot describes one field).
    this.fieldName    = new Array(32).fill(null);
    this.fieldKind    = new Uint8Array(32);
    this.fieldLength  = new Uint32Array(32);
    this.fieldBytes   = new Uint32Array(32);
    this.fieldCount   = 0;

    // Dependency edges (names of other components).
    this.dependencies = new Array(MAX_DEPENDENCIES_PER_COMPONENT).fill(null);
    this.dependencyCount = 0;

    // Aliases.
    this.aliases      = new Array(MAX_ALIASES_PER_COMPONENT).fill(null);
    this.aliasCount   = 0;

    // Generation counter — bumped every time the same name is re-registered.
    this.generation   = 0;

    // The actual component object.
    this.ref          = null;

    // Precomputed byte cost.
    this.byteCost     = 0;

    // Timestamps.
    this.registeredAt = 0;

    // Free-form tags (bitmask).
    this.tags         = 0;
  }

  reset() {
    this.name         = null;
    this.category     = COMPONENT_CATEGORY.UNKNOWN;
    this.subsystem    = SUBSYSTEM.UNKNOWN;
    for (let i = 0; i < this.fieldCount; i++) {
      this.fieldName[i]   = null;
      this.fieldKind[i]   = FIELD_KIND.UNKNOWN;
      this.fieldLength[i] = 0;
      this.fieldBytes[i]  = 0;
    }
    this.fieldCount = 0;
    for (let i = 0; i < this.dependencyCount; i++) this.dependencies[i] = null;
    this.dependencyCount = 0;
    for (let i = 0; i < this.aliasCount; i++) this.aliases[i] = null;
    this.aliasCount = 0;
    this.generation = 0;
    this.ref = null;
    this.byteCost = 0;
    this.registeredAt = 0;
    this.tags = 0;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _registryIdCounter = 0;

function _nextRegistryId() {
  return ++_registryIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _classifyFieldKind(view) {
  if (view instanceof Float32Array)         return FIELD_KIND.F32;
  if (view instanceof Float64Array)         return FIELD_KIND.F64;
  if (view instanceof Int32Array)           return FIELD_KIND.I32;
  if (view instanceof Uint32Array)          return FIELD_KIND.U32;
  if (view instanceof Int16Array)           return FIELD_KIND.I16;
  if (view instanceof Uint16Array)          return FIELD_KIND.U16;
  if (view instanceof Int8Array)            return FIELD_KIND.I8;
  if (view instanceof Uint8Array)           return FIELD_KIND.U8;
  if (typeof Uint8ClampedArray !== 'undefined' && view instanceof Uint8ClampedArray) return FIELD_KIND.U8_CLAMP;
  return FIELD_KIND.UNKNOWN;
}

/* ------------------------------------------------------------------ */
/* 4. COMPONENT REGISTRY                                              */
/* ------------------------------------------------------------------ */

export class ComponentRegistry {
  constructor(options = {}) {
    this.registryId = _nextRegistryId();

    this.options = Object.assign({
      maxComponents:   MAX_REGISTERED_COMPONENTS,
      enforceSoA:      true,
      strictNames:     true,
      nameMaxLength:   64,
      allowReRegister: true,
      trackProfiler:   PERF_TIER_LOCAL === 'HIGH',
      logChannel:      LOG_CHANNEL.CORE,
    }, options || {});

    // Fixed capacity slots.
    this.capacity = this.options.maxComponents;
    this.entries  = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.entries[i] = new ComponentEntry(i);
    this.count    = 0;

    // Fast lookups.
    this.byName      = new Map();
    this.byAlias     = new Map();
    this.byCategory  = new Array(COMPONENT_CATEGORY.COUNT).fill(null).map(() => []);
    this.bySubsystem = new Array(SUBSYSTEM.COUNT).fill(null).map(() => []);

    // Statistics.
    this.stats = {
      totalRegistrations:  0,
      totalRejections:     0,
      totalReRegistrations:0,
      totalByteCost:       0,
      peakComponentCount:  0,
    };

    // Optional boundary for policy enforcement.
    this._boundary = null;
    try {
      const mgr = getDefaultErrorBoundaries();
      if (mgr) {
        this._boundary = mgr.create('registry.components#' + this.registryId, {
          tag: BOUNDARY_TAG.REGISTRY,
          failureThreshold: 5,
        });
      }
    } catch (_) { /* swallow */ }
  }

  /* ---------------- registration ---------------- */

  /**
   * Registers a component object under a symbolic name. Returns the entry
   * if successful, or null (with `.lastRegisterResult` set to a reason).
   */
  register(name, component, meta = {}) {
    // Validate name.
    if (typeof name !== 'string' || name.length === 0 || name.length > this.options.nameMaxLength) {
      this.lastRegisterResult = REGISTER_RESULT.INVALID_NAME;
      this.stats.totalRejections++;
      return null;
    }

    // Validate shape.
    if (!component || typeof component !== 'object') {
      this.lastRegisterResult = REGISTER_RESULT.INVALID_SHAPE;
      this.stats.totalRejections++;
      return null;
    }

    // Enforce SoA shape (all fields are typed arrays of length MAX_ENTITIES).
    if (this.options.enforceSoA) {
      const fieldNames = Object.keys(component);
      for (let i = 0; i < fieldNames.length; i++) {
        const field = component[fieldNames[i]];
        if (!ArrayBuffer.isView(field)) {
          this.lastRegisterResult = REGISTER_RESULT.INVALID_SHAPE;
          this.stats.totalRejections++;
          return null;
        }
        if (field.length !== MAX_ENTITIES) {
          this.lastRegisterResult = REGISTER_RESULT.INVALID_SHAPE;
          this.stats.totalRejections++;
          return null;
        }
      }
    }

    // Re-registration path.
    const existingIdx = this.byName.get(name);
    if (existingIdx !== undefined) {
      if (!this.options.allowReRegister) {
        this.lastRegisterResult = REGISTER_RESULT.DUPLICATE;
        this.stats.totalRejections++;
        return null;
      }
      const entry = this.entries[existingIdx];
      entry.ref        = component;
      entry.generation++;
      entry.registeredAt = _now();
      this.stats.totalReRegistrations++;
      this.lastRegisterResult = REGISTER_RESULT.OK;
      return entry;
    }

    // Capacity check.
    if (this.count >= this.capacity) {
      this.lastRegisterResult = REGISTER_RESULT.CAPACITY_FULL;
      this.stats.totalRejections++;
      return null;
    }

    // Acquire slot.
    const idx = this.count++;
    const entry = this.entries[idx];
    entry.reset();

    entry.name         = name;
    entry.category     = meta.category  !== undefined ? meta.category  : COMPONENT_CATEGORY.UNKNOWN;
    entry.subsystem    = meta.subsystem !== undefined ? meta.subsystem : SUBSYSTEM.UNKNOWN;
    entry.ref          = component;
    entry.generation   = 1;
    entry.registeredAt = _now();
    entry.tags         = meta.tags || 0;

    // Populate field metadata.
    const fieldNames = Object.keys(component);
    const fieldCount = Math.min(fieldNames.length, entry.fieldName.length);
    let totalBytes = 0;
    for (let i = 0; i < fieldCount; i++) {
      const fieldName = fieldNames[i];
      const view = component[fieldName];
      const kind = _classifyFieldKind(view);
      const bytes = view.length * FIELD_KIND_BYTES[kind];
      entry.fieldName[i]   = fieldName;
      entry.fieldKind[i]   = kind;
      entry.fieldLength[i] = view.length;
      entry.fieldBytes[i]  = bytes;
      totalBytes += bytes;
    }
    entry.fieldCount = fieldCount;
    entry.byteCost = totalBytes;

    // Populate dependencies.
    if (Array.isArray(meta.dependencies)) {
      const n = Math.min(meta.dependencies.length, MAX_DEPENDENCIES_PER_COMPONENT);
      for (let i = 0; i < n; i++) {
        entry.dependencies[i] = String(meta.dependencies[i]);
      }
      entry.dependencyCount = n;
    }

    // Populate aliases.
    if (Array.isArray(meta.aliases)) {
      const n = Math.min(meta.aliases.length, MAX_ALIASES_PER_COMPONENT);
      for (let i = 0; i < n; i++) {
        const alias = String(meta.aliases[i]);
        entry.aliases[i] = alias;
        this.byAlias.set(alias, idx);
      }
      entry.aliasCount = n;
    }

    // Update indexes.
    this.byName.set(name, idx);
    this.byCategory[entry.category].push(idx);
    this.bySubsystem[entry.subsystem].push(idx);

    this.stats.totalRegistrations++;
    this.stats.totalByteCost += totalBytes;
    if (this.count > this.stats.peakComponentCount) {
      this.stats.peakComponentCount = this.count;
    }

    this.lastRegisterResult = REGISTER_RESULT.OK;
    return entry;
  }

  /**
   * Registers many components at once from an array of specs:
   *   [{ name, component, category, subsystem, dependencies, aliases }, ...]
   */
  registerMany(specs) {
    if (!Array.isArray(specs)) return 0;
    let registered = 0;
    for (let i = 0; i < specs.length; i++) {
      const s = specs[i];
      if (!s) continue;
      const entry = this.register(s.name, s.component, {
        category:     s.category,
        subsystem:    s.subsystem,
        dependencies: s.dependencies,
        aliases:      s.aliases,
        tags:         s.tags,
      });
      if (entry) registered++;
    }
    return registered;
  }

  /**
   * Unregisters a component by name. Returns true on success.
   */
  unregister(name) {
    const idx = this.byName.get(name);
    if (idx === undefined) return false;

    const entry = this.entries[idx];

    // Remove from category + subsystem index lists.
    this._removeFromList(this.byCategory[entry.category], idx);
    this._removeFromList(this.bySubsystem[entry.subsystem], idx);

    // Remove aliases.
    for (let i = 0; i < entry.aliasCount; i++) {
      const alias = entry.aliases[i];
      if (alias) this.byAlias.delete(alias);
    }

    // Remove from byName.
    this.byName.delete(name);

    // Rebalance: swap last entry into this slot.
    const last = this.count - 1;
    if (idx !== last) {
      const moved = this.entries[last];
      this.entries[idx] = moved;
      this.entries[idx].index = idx;
      this.entries[last] = new ComponentEntry(last);

      // Update index maps for the moved entry.
      this.byName.set(moved.name, idx);
      this._replaceInList(this.byCategory[moved.category], last, idx);
      this._replaceInList(this.bySubsystem[moved.subsystem], last, idx);
      for (let i = 0; i < moved.aliasCount; i++) {
        if (moved.aliases[i]) this.byAlias.set(moved.aliases[i], idx);
      }
    } else {
      this.entries[last] = new ComponentEntry(last);
    }

    this.count--;
    this.stats.totalByteCost -= entry.byteCost;

    return true;
  }

  _removeFromList(list, idx) {
    const i = list.indexOf(idx);
    if (i >= 0) list.splice(i, 1);
  }

  _replaceInList(list, oldIdx, newIdx) {
    const i = list.indexOf(oldIdx);
    if (i >= 0) list[i] = newIdx;
  }

  /* ---------------- lookup ---------------- */

  /**
   * Returns the entry for a component by name (or alias).
   */
  get(name) {
    let idx = this.byName.get(name);
    if (idx === undefined) idx = this.byAlias.get(name);
    if (idx === undefined) return null;
    return this.entries[idx];
  }

  /**
   * Returns the raw component object by name (or null).
   */
  getComponent(name) {
    const entry = this.get(name);
    return entry ? entry.ref : null;
  }

  /**
   * Returns true if a component is registered under the given name or
   * alias.
   */
  has(name) {
    return this.byName.has(name) || this.byAlias.has(name);
  }

  /**
   * Returns an array of entry names in the given category.
   */
  listCategory(category) {
    if (category < 0 || category >= COMPONENT_CATEGORY.COUNT) return [];
    const list = this.byCategory[category];
    const out = new Array(list.length);
    for (let i = 0; i < list.length; i++) {
      out[i] = this.entries[list[i]].name;
    }
    return out;
  }

  /**
   * Returns an array of entry names owned by the given subsystem.
   */
  listSubsystem(subsystem) {
    if (subsystem < 0 || subsystem >= SUBSYSTEM.COUNT) return [];
    const list = this.bySubsystem[subsystem];
    const out = new Array(list.length);
    for (let i = 0; i < list.length; i++) {
      out[i] = this.entries[list[i]].name;
    }
    return out;
  }

  /**
   * Returns an array of every registered component name.
   */
  listAll() {
    const out = new Array(this.count);
    for (let i = 0; i < this.count; i++) {
      out[i] = this.entries[i].name;
    }
    return out;
  }

  /**
   * Returns an array of component entries whose name matches a predicate.
   */
  filter(fn, ctx) {
    const out = [];
    for (let i = 0; i < this.count; i++) {
      if (fn.call(ctx, this.entries[i])) {
        out.push(this.entries[i]);
      }
    }
    return out;
  }

  /* ---------------- introspection ---------------- */

  /**
   * Returns the SoA field metadata of a component.
   */
  getFields(name) {
    const entry = this.get(name);
    if (!entry) return null;
    const fields = new Array(entry.fieldCount);
    for (let i = 0; i < entry.fieldCount; i++) {
      fields[i] = {
        name:   entry.fieldName[i],
        kind:   FIELD_KIND_NAME[entry.fieldKind[i]],
        length: entry.fieldLength[i],
        bytes:  entry.fieldBytes[i],
      };
    }
    return fields;
  }

  /**
   * Returns the byte cost of a component.
   */
  getByteCost(name) {
    const entry = this.get(name);
    return entry ? entry.byteCost : 0;
  }

  /**
   * Returns the dependencies of a component.
   */
  getDependencies(name) {
    const entry = this.get(name);
    if (!entry) return null;
    const out = new Array(entry.dependencyCount);
    for (let i = 0; i < entry.dependencyCount; i++) out[i] = entry.dependencies[i];
    return out;
  }

  /**
   * Returns the aliases of a component.
   */
  getAliases(name) {
    const entry = this.get(name);
    if (!entry) return null;
    const out = new Array(entry.aliasCount);
    for (let i = 0; i < entry.aliasCount; i++) out[i] = entry.aliases[i];
    return out;
  }

  /**
   * Returns true if a component has a specific dependency.
   */
  hasDependency(name, dependencyName) {
    const entry = this.get(name);
    if (!entry) return false;
    for (let i = 0; i < entry.dependencyCount; i++) {
      if (entry.dependencies[i] === dependencyName) return true;
    }
    return false;
  }

  /* ---------------- bulk operations ---------------- */

  /**
   * Returns a frozen, plain object suitable for JSON serialization of the
   * catalog. Useful for building HUDs or exporting for debugging.
   */
  describe(name) {
    const entry = this.get(name);
    if (!entry) return null;
    const fields = this.getFields(name);
    const deps = this.getDependencies(name);
    const aliases = this.getAliases(name);
    return {
      name:         entry.name,
      category:     COMPONENT_CATEGORY_NAME[entry.category] || 'unknown',
      subsystem:    SUBSYSTEM_NAME[entry.subsystem] || 'unknown',
      fieldCount:   entry.fieldCount,
      fields,
      byteCost:     entry.byteCost,
      dependencies: deps,
      aliases,
      generation:   entry.generation,
      tags:         entry.tags,
      registeredAt: entry.registeredAt,
    };
  }

  /**
   * Returns a complete snapshot of every entry's description.
   */
  describeAll() {
    const out = new Array(this.count);
    for (let i = 0; i < this.count; i++) {
      out[i] = this.describe(this.entries[i].name);
    }
    return out;
  }

  /**
   * Validates that a candidate component object has the required SoA shape.
   */
  validate(name, component) {
    if (!component || typeof component !== 'object') return false;
    const fields = Object.keys(component);
    for (let i = 0; i < fields.length; i++) {
      const view = component[fields[i]];
      if (!ArrayBuffer.isView(view)) return false;
      if (view.length !== MAX_ENTITIES) return false;
    }
    return true;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      registryId:            this.registryId,
      capacity:              this.capacity,
      count:                 this.count,
      peakCount:             this.stats.peakComponentCount,
      totalRegistrations:    this.stats.totalRegistrations,
      totalRejections:       this.stats.totalRejections,
      totalReRegistrations:  this.stats.totalReRegistrations,
      totalByteCost:         this.stats.totalByteCost,
      totalMegabytes:        this.stats.totalByteCost / (1024 * 1024),
      lastRegisterResult:    REGISTER_RESULT_NAME[this.lastRegisterResult] || 'ok',
      categoryCounts:        this._categoryCounts(),
      subsystemCounts:       this._subsystemCounts(),
      perfTier:              PERF_TIER_LOCAL,
    };
  }

  _categoryCounts() {
    const out = [];
    for (let c = 0; c < COMPONENT_CATEGORY.COUNT; c++) {
      const n = this.byCategory[c].length;
      if (n > 0) {
        out.push({
          category: COMPONENT_CATEGORY_NAME[c],
          count:    n,
        });
      }
    }
    return out;
  }

  _subsystemCounts() {
    const out = [];
    for (let s = 0; s < SUBSYSTEM.COUNT; s++) {
      const n = this.bySubsystem[s].length;
      if (n > 0) {
        out.push({
          subsystem: SUBSYSTEM_NAME[s],
          count:     n,
        });
      }
    }
    return out;
  }

  /* ---------------- dispose ---------------- */

  /**
   * Clears every registration but keeps the fixed-capacity slot array.
   */
  reset() {
    for (let i = 0; i < this.capacity; i++) this.entries[i].reset();
    this.count = 0;
    this.byName.clear();
    this.byAlias.clear();
    for (let c = 0; c < COMPONENT_CATEGORY.COUNT; c++) this.byCategory[c].length = 0;
    for (let s = 0; s < SUBSYSTEM.COUNT; s++) this.bySubsystem[s].length = 0;

    this.stats.totalRegistrations = 0;
    this.stats.totalRejections = 0;
    this.stats.totalReRegistrations = 0;
    this.stats.totalByteCost = 0;
    this.stats.peakComponentCount = 0;
    this.lastRegisterResult = REGISTER_RESULT.OK;
    return this;
  }

  /**
   * Full disposal of the registry.
   */
  dispose() {
    this.reset();
    for (let i = 0; i < this.capacity; i++) this.entries[i] = null;
    this.entries.length = 0;
    this.entries = null;
    this._boundary = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultRegistry = null;

export function getDefaultComponentRegistry() {
  if (!_defaultRegistry) _defaultRegistry = new ComponentRegistry();
  return _defaultRegistry;
}

export function disposeDefaultComponentRegistry() {
  if (_defaultRegistry) {
    _defaultRegistry.dispose();
    _defaultRegistry = null;
  }
}

/* ------------------------------------------------------------------ */
/* 6. AUTO-REGISTRATION OF CANONICAL COMPONENTS                       */
/* ------------------------------------------------------------------ */

/**
 * Registers the full canonical component catalog of the anime lighting
 * stack: light, shadow, GI, AO, camera, and support components. Called
 * once at world boot by 010_scn_ECSWorld.js.
 *
 * The `bundle` argument is expected to be the ALL_SCENE_COMPONENTS object
 * exported by 010_scn_ECSWorld.js (or an equivalent merged bundle).
 */
export function registerCanonicalComponents(bundle) {
  if (!bundle || typeof bundle !== 'object') return 0;
  const registry = getDefaultComponentRegistry();

  const specs = [];

  /* ---------------- Light components ---------------- */
  if (bundle.Transform)       specs.push({ name: 'Transform',      component: bundle.Transform,      category: COMPONENT_CATEGORY.TRANSFORM, subsystem: SUBSYSTEM.CORE,    dependencies: [] });
  if (bundle.Target)          specs.push({ name: 'Target',         component: bundle.Target,         category: COMPONENT_CATEGORY.TRANSFORM, subsystem: SUBSYSTEM.CORE,    dependencies: ['Transform'] });
  if (bundle.LightRef)        specs.push({ name: 'LightRef',       component: bundle.LightRef,       category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['Transform'] });
  if (bundle.LightState)      specs.push({ name: 'LightState',     component: bundle.LightState,     category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightShadow)     specs.push({ name: 'LightShadow',    component: bundle.LightShadow,    category: COMPONENT_CATEGORY.SHADOW,    subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightCluster)    specs.push({ name: 'LightCluster',   component: bundle.LightCluster,   category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightBudget)     specs.push({ name: 'LightBudget',    component: bundle.LightBudget,    category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightPriority)   specs.push({ name: 'LightPriority',  component: bundle.LightPriority,  category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightBehavior)   specs.push({ name: 'LightBehavior',  component: bundle.LightBehavior,  category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightComposite)  specs.push({ name: 'LightComposite', component: bundle.LightComposite, category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightIndoor)     specs.push({ name: 'LightIndoor',    component: bundle.LightIndoor,    category: COMPONENT_CATEGORY.INTERIOR,  subsystem: SUBSYSTEM.INTERIOR,dependencies: ['LightRef'] });
  if (bundle.LightIES)        specs.push({ name: 'LightIES',       component: bundle.LightIES,       category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightEmissive)   specs.push({ name: 'LightEmissive',  component: bundle.LightEmissive,  category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightFlicker)    specs.push({ name: 'LightFlicker',   component: bundle.LightFlicker,   category: COMPONENT_CATEGORY.ANIMATION, subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.LightDayCycle)   specs.push({ name: 'LightDayCycle',  component: bundle.LightDayCycle,  category: COMPONENT_CATEGORY.ENVIRONMENT, subsystem: SUBSYSTEM.ENVIRONMENT, dependencies: ['LightRef'] });
  if (bundle.LightTag)        specs.push({ name: 'LightTag',       component: bundle.LightTag,       category: COMPONENT_CATEGORY.LIGHT,     subsystem: SUBSYSTEM.LIGHTS,  dependencies: ['LightRef'] });
  if (bundle.CameraTag)       specs.push({ name: 'CameraTag',      component: bundle.CameraTag,      category: COMPONENT_CATEGORY.CAMERA,    subsystem: SUBSYSTEM.CAMERA,  dependencies: ['Transform'] });

  /* ---------------- Shadow components ---------------- */
  if (bundle.ShadowCasterRef)    specs.push({ name: 'ShadowCasterRef',   component: bundle.ShadowCasterRef,   category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: [] });
  if (bundle.ShadowReceiverRef)  specs.push({ name: 'ShadowReceiverRef', component: bundle.ShadowReceiverRef, category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: [] });
  if (bundle.ShadowAtlas)        specs.push({ name: 'ShadowAtlas',       component: bundle.ShadowAtlas,       category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowCascade)      specs.push({ name: 'ShadowCascade',     component: bundle.ShadowCascade,     category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowBias)         specs.push({ name: 'ShadowBias',        component: bundle.ShadowBias,        category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowFilter)       specs.push({ name: 'ShadowFilter',      component: bundle.ShadowFilter,      category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowSoftness)     specs.push({ name: 'ShadowSoftness',    component: bundle.ShadowSoftness,    category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowFrustum)      specs.push({ name: 'ShadowFrustum',     component: bundle.ShadowFrustum,     category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowCache)        specs.push({ name: 'ShadowCache',       component: bundle.ShadowCache,       category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowBudget)       specs.push({ name: 'ShadowBudget',      component: bundle.ShadowBudget,      category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowContact)      specs.push({ name: 'ShadowContact',     component: bundle.ShadowContact,     category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowVolume)       specs.push({ name: 'ShadowVolume',      component: bundle.ShadowVolume,      category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowDirLight)     specs.push({ name: 'ShadowDirLight',    component: bundle.ShadowDirLight,    category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowPointLight)   specs.push({ name: 'ShadowPointLight',  component: bundle.ShadowPointLight,  category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowSpotLight)    specs.push({ name: 'ShadowSpotLight',   component: bundle.ShadowSpotLight,   category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowAtlasMap)     specs.push({ name: 'ShadowAtlasMap',    component: bundle.ShadowAtlasMap,    category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['ShadowAtlas'] });
  if (bundle.ShadowTint)         specs.push({ name: 'ShadowTint',        component: bundle.ShadowTint,        category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowEdge)         specs.push({ name: 'ShadowEdge',        component: bundle.ShadowEdge,        category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['ShadowSoftness'] });
  if (bundle.ShadowTile)         specs.push({ name: 'ShadowTile',        component: bundle.ShadowTile,        category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: [] });
  if (bundle.ShadowUpdatePolicy) specs.push({ name: 'ShadowUpdatePolicy',component: bundle.ShadowUpdatePolicy,category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });
  if (bundle.ShadowState)        specs.push({ name: 'ShadowState',       component: bundle.ShadowState,       category: COMPONENT_CATEGORY.SHADOW, subsystem: SUBSYSTEM.SHADOWS, dependencies: ['LightShadow'] });

  /* ---------------- GI components ---------------- */
  if (bundle.GIProbeRef)         specs.push({ name: 'GIProbeRef',        component: bundle.GIProbeRef,        category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIIrradiance)       specs.push({ name: 'GIIrradiance',      component: bundle.GIIrradiance,      category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GISH)               specs.push({ name: 'GISH',              component: bundle.GISH,              category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GISHHigh)           specs.push({ name: 'GISHHigh',          component: bundle.GISHHigh,          category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GISH'] });
  if (bundle.GIBouncePath)       specs.push({ name: 'GIBouncePath',      component: bundle.GIBouncePath,      category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GIVoxel)            specs.push({ name: 'GIVoxel',           component: bundle.GIVoxel,           category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIDistanceField)    specs.push({ name: 'GIDistanceField',   component: bundle.GIDistanceField,   category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIVoxel'] });
  if (bundle.GIOcclusion)        specs.push({ name: 'GIOcclusion',       component: bundle.GIOcclusion,       category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GIPortal)           specs.push({ name: 'GIPortal',          component: bundle.GIPortal,          category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIVolume'] });
  if (bundle.GIBudget)           specs.push({ name: 'GIBudget',          component: bundle.GIBudget,          category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GIState)            specs.push({ name: 'GIState',           component: bundle.GIState,           category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIProbeRef'] });
  if (bundle.GIUpdateQueue)      specs.push({ name: 'GIUpdateQueue',     component: bundle.GIUpdateQueue,     category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIRadianceCache)    specs.push({ name: 'GIRadianceCache',   component: bundle.GIRadianceCache,   category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIReflectionProbe)  specs.push({ name: 'GIReflectionProbe', component: bundle.GIReflectionProbe, category: COMPONENT_CATEGORY.PROBE,  subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GILightfield)       specs.push({ name: 'GILightfield',      component: bundle.GILightfield,      category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIVolume)           specs.push({ name: 'GIVolume',          component: bundle.GIVolume,          category: COMPONENT_CATEGORY.VOLUME, subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIIndoor)           specs.push({ name: 'GIIndoor',          component: bundle.GIIndoor,          category: COMPONENT_CATEGORY.INTERIOR, subsystem: SUBSYSTEM.INTERIOR, dependencies: ['GIVolume'] });
  if (bundle.GIOutdoor)          specs.push({ name: 'GIOutdoor',         component: bundle.GIOutdoor,         category: COMPONENT_CATEGORY.EXTERIOR, subsystem: SUBSYSTEM.EXTERIOR, dependencies: ['GIVolume'] });
  if (bundle.GICelBands)         specs.push({ name: 'GICelBands',        component: bundle.GICelBands,        category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIPalette)          specs.push({ name: 'GIPalette',         component: bundle.GIPalette,         category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIAsync)            specs.push({ name: 'GIAsync',           component: bundle.GIAsync,           category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GILeak)             specs.push({ name: 'GILeak',            component: bundle.GILeak,            category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GITemporal)         specs.push({ name: 'GITemporal',        component: bundle.GITemporal,        category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: [] });
  if (bundle.GIVolumeBlend)      specs.push({ name: 'GIVolumeBlend',     component: bundle.GIVolumeBlend,     category: COMPONENT_CATEGORY.GI,     subsystem: SUBSYSTEM.GI, dependencies: ['GIVolume'] });

  /* ---------------- AO components ---------------- */
  if (bundle.AOVolumeRef)           specs.push({ name: 'AOVolumeRef',           component: bundle.AOVolumeRef,           category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOSampling)            specs.push({ name: 'AOSampling',            component: bundle.AOSampling,            category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOKernel)              specs.push({ name: 'AOKernel',              component: bundle.AOKernel,              category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOHistory)             specs.push({ name: 'AOHistory',             component: bundle.AOHistory,             category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOBlur)                specs.push({ name: 'AOBlur',                component: bundle.AOBlur,                category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOQuality)             specs.push({ name: 'AOQuality',             component: bundle.AOQuality,             category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOState)               specs.push({ name: 'AOState',               component: bundle.AOState,               category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOBudget)              specs.push({ name: 'AOBudget',              component: bundle.AOBudget,              category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOScreenSpace)         specs.push({ name: 'AOScreenSpace',         component: bundle.AOScreenSpace,         category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOContactShadow)       specs.push({ name: 'AOContactShadow',       component: bundle.AOContactShadow,       category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AODistanceField)       specs.push({ name: 'AODistanceField',       component: bundle.AODistanceField,       category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOIndoorVolume)        specs.push({ name: 'AOIndoorVolume',        component: bundle.AOIndoorVolume,        category: COMPONENT_CATEGORY.INTERIOR, subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOOutdoorVolume)       specs.push({ name: 'AOOutdoorVolume',       component: bundle.AOOutdoorVolume,       category: COMPONENT_CATEGORY.EXTERIOR, subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AOTemporalAccumulator) specs.push({ name: 'AOTemporalAccumulator', component: bundle.AOTemporalAccumulator, category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });
  if (bundle.AODither)              specs.push({ name: 'AODither',              component: bundle.AODither,              category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOCelBands)            specs.push({ name: 'AOCelBands',            component: bundle.AOCelBands,            category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOInkOutline)          specs.push({ name: 'AOInkOutline',          component: bundle.AOInkOutline,          category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOEdgeFade)            specs.push({ name: 'AOEdgeFade',            component: bundle.AOEdgeFade,            category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOBilateral)           specs.push({ name: 'AOBilateral',           component: bundle.AOBilateral,           category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AODenoiser)            specs.push({ name: 'AODenoiser',            component: bundle.AODenoiser,            category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOAsync)               specs.push({ name: 'AOAsync',               component: bundle.AOAsync,               category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOResidency)           specs.push({ name: 'AOResidency',           component: bundle.AOResidency,           category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOLeak)                specs.push({ name: 'AOLeak',                component: bundle.AOLeak,                category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: [] });
  if (bundle.AOStyle)               specs.push({ name: 'AOStyle',               component: bundle.AOStyle,               category: COMPONENT_CATEGORY.AO,     subsystem: SUBSYSTEM.AO, dependencies: ['AOVolumeRef'] });

  // Register with metadata.
  return registry.registerMany(specs);
}

/* ------------------------------------------------------------------ */
/* 7. CONVENIENCE QUERIES                                             */
/* ------------------------------------------------------------------ */

/**
 * Returns every light-category component name.
 */
export function listLightComponents() {
  return getDefaultComponentRegistry().listCategory(COMPONENT_CATEGORY.LIGHT);
}

/**
 * Returns every shadow-category component name.
 */
export function listShadowComponents() {
  return getDefaultComponentRegistry().listCategory(COMPONENT_CATEGORY.SHADOW);
}

/**
 * Returns every GI-category component name.
 */
export function listGIComponents() {
  return getDefaultComponentRegistry().listCategory(COMPONENT_CATEGORY.GI);
}

/**
 * Returns every AO-category component name.
 */
export function listAOComponents() {
  return getDefaultComponentRegistry().listCategory(COMPONENT_CATEGORY.AO);
}

/**
 * Returns every component owned by a given subsystem.
 */
export function listSubsystemComponents(subsystem) {
  return getDefaultComponentRegistry().listSubsystem(subsystem);
}

/**
 * Returns a full report on the registry state.
 */
export function getRegistryReport() {
  return getDefaultComponentRegistry().getStats();
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createComponentRegistry(options = {}) {
  return new ComponentRegistry(options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                 */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ComponentRegistry,
  ComponentEntry,

  // Singleton
  getDefaultComponentRegistry,
  disposeDefaultComponentRegistry,
  createComponentRegistry,

  // Auto-registration
  registerCanonicalComponents,

  // Convenience queries
  listLightComponents,
  listShadowComponents,
  listGIComponents,
  listAOComponents,
  listSubsystemComponents,
  getRegistryReport,

  // Enums
  COMPONENT_CATEGORY,
  COMPONENT_CATEGORY_NAME,
  FIELD_KIND,
  FIELD_KIND_NAME,
  FIELD_KIND_BYTES,
  SUBSYSTEM,
  SUBSYSTEM_NAME,
  REGISTER_RESULT,
  REGISTER_RESULT_NAME,

  // Constants
  MAX_REGISTERED_COMPONENTS,
  MAX_CATEGORIES,
  MAX_SUBSYSTEMS,
  MAX_ALIASES_PER_COMPONENT,
  MAX_DEPENDENCIES_PER_COMPONENT,
};

export default _defaultExport;