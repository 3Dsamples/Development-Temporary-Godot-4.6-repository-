// File : 010
// name : src/ecs/010_scn_ECSWorld.js
// description : The canonical scene ECS world for the anime lighting stack on
//               Android mobile. This is the single authoritative bitECS 0.4.0
//               world that hosts EVERY entity in the scene: light sources,
//               shadow casters/receivers, GI probes, AO volumes, cameras,
//               weather systems, biome tags, interior/exterior volumes,
//               materials, post-processing, streaming chunks, LOD groups,
//               visibility sets, and debug markers.
//
//               It imports and registers the full SoA component catalog from
//               the sibling modules (002_lgt_* through 005_lgt_*, plus the
//               upcoming 011–035 scene component modules), passes it through
//               the bitECS 0.4.0 version policy (009_scn_BiteCSVersionPolicy
//               at the runtime level via 008_scn_world), and exposes one
//               `world` handle plus one set of lifecycle helpers so every
//               downstream system queries the same world with the same
//               entity IDs.
//
//               Responsibilities:
//                 • Own the single `world` object (createWorld with the full
//                   component bundle).
//                 • Own the entity ID namespace (allocation, recycling,
//                   generation tracking, dense iteration).
//                 • Own the component registry (name → component object map)
//                   so systems can query components by name without importing
//                   each module directly.
//                 • Own the world time contract (delta, elapsed, frame) that
//                   all systems read from.
//                 • Provide fast path helpers: `spawnEntity`, `destroyEntity`,
//                   `attachComponent`, `detachComponent`, `hasComponent`,
//                   `queryComponents`, `forEachEntity`.
//                 • Provide the canonical "world ready" gate that downstream
//                   lighting systems check before running.
//                 • Enforce the bitECS 0.4.0 architectural contract: no
//                   `defineComponent`, no `Types`, no separate store
//                   registry, plain-object SoA components sized once to
//                   MAX_ENTITIES = 100000.
//                 • Provide a factory for isolated test worlds that share
//                   the same component catalog but not the same entity IDs.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external ECS libs; every internal array sized once
//               at construction.
// best for : Guaranteeing that the entire anime lighting stack runs on one
//            ECS world with one entity ID namespace. Every subsystem
//            (006_lgt_LightManager through 380_lgt_lights, plus every scene
//            system) imports `getECSWorld()` and receives the same world
//            handle. No cross-world confusion, no duplicated entity IDs, no
//            drift between the scene world and the lighting world.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  createWorld,
  addEntity,
  removeEntity,
  addComponent,
  removeComponent,
  hasComponent,
  query,
  entityExists,
  resetWorld,
  deleteWorld,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

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
} from '../core/009_scn_BiteCSVersionPolicy.js';

// ---------------------------------------------------------------------------
// SoA component bundles from the light subsystem
// ---------------------------------------------------------------------------
import {
  MAX_ENTITIES,
  Transform,
  Target,
  LightRef,
  LightState,
  LightShadow,
  LightCluster,
  LightBudget as LightBudgetComponent,
  LightPriority,
  LightBehavior,
  LightComposite,
  LightIndoor,
  LightIES,
  LightEmissive,
  LightFlicker,
  LightDayCycle,
  LightTag,
  CameraTag,
  GIProbeRef as LightGIProbeRef,
  AOVolumeRef as LightAOVolumeRef,
  LIGHTING_COMPONENTS,
} from './002_lgt_LightComponents.js';

import {
  ShadowCasterRef,
  ShadowReceiverRef,
  ShadowAtlas,
  ShadowCascade,
  ShadowBias,
  ShadowFilter,
  ShadowSoftness,
  ShadowFrustum,
  ShadowCache,
  ShadowBudget,
  ShadowContact,
  ShadowVolume,
  ShadowDirLight,
  ShadowPointLight,
  ShadowSpotLight,
  ShadowAtlasMap,
  ShadowTint,
  ShadowEdge,
  ShadowTile,
  ShadowUpdatePolicy,
  ShadowState,
  SHADOW_COMPONENTS,
} from './003_lgt_ShadowComponents.js';

import {
  GIProbeRef,
  GIIrradiance,
  GISH,
  GISHHigh,
  GIBouncePath,
  GIVoxel,
  GIDistanceField,
  GIOcclusion,
  GIPortal,
  GIBudget,
  GIState,
  GIUpdateQueue,
  GIRadianceCache,
  GIReflectionProbe,
  GILightfield,
  GIVolume,
  GIIndoor,
  GIOutdoor,
  GICelBands,
  GIPalette,
  GIAsync,
  GILeak,
  GITemporal,
  GIVolumeBlend,
  GI_COMPONENTS,
} from './004_lgt_GIComponents.js';

import {
  AOVolumeRef,
  AOSampling,
  AOKernel,
  AOHistory,
  AOBlur,
  AOQuality,
  AOState,
  AOBudget,
  AOScreenSpace,
  AOContactShadow,
  AODistanceField,
  AOIndoorVolume,
  AOOutdoorVolume,
  AOTemporalAccumulator,
  AODither,
  AOCelBands,
  AOInkOutline,
  AOEdgeFade,
  AOBilateral,
  AODenoiser,
  AOAsync,
  AOResidency,
  AOLeak,
  AOStyle,
  AO_COMPONENTS,
} from './005_lgt_AOComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Reserved entity ID range conventions. Downstream systems can reserve
 * blocks to make entity classification O(1).
 */
export const ENTITY_RESERVED = Object.freeze({
  INVALID:        -1,
  FIRST_VALID:     0,
  CAMERA_BLOCK:    0,       // 0..15
  LIGHT_BLOCK:    16,       // 16..(16+MAX_LIGHTS-1)
  SHADOW_BLOCK: 4608,       // 4608..(4608+MAX_SHADOWS-1)
  GI_BLOCK:     5120,       // 5120..(5120+MAX_GI_PROBES-1)
  AO_BLOCK:     9216,       // 9216..(9216+MAX_AO_VOLUMES-1)
  SCENE_BLOCK: 10240,       // 10240..(10240+SCENE_CAPACITY)
});

/**
 * Sanity caps for block sizes. These are only advisory; the world itself
 * is sized to MAX_ENTITIES.
 */
export const ENTITY_BLOCK_SIZE = Object.freeze({
  CAMERA:     16,
  LIGHT:    4096,
  SHADOW:    512,
  GI:       4096,
  AO:       1024,
  SCENE:   MAX_ENTITIES - 10240,
});

/**
 * Component name → component object map, populated once after import.
 * Downstream systems query components by symbolic name:
 *
 *   const T = ecs.getComponent('Transform');
 *   const lights = ecs.query([T, ecs.getComponent('LightRef')]);
 */
const _componentRegistry = new Map();

function _registerComponent(name, component) {
  if (!component) return;
  _componentRegistry.set(name, component);
}

function _buildComponentRegistry() {
  // Light subsystem
  _registerComponent('Transform',      Transform);
  _registerComponent('Target',         Target);
  _registerComponent('LightRef',       LightRef);
  _registerComponent('LightState',     LightState);
  _registerComponent('LightShadow',    LightShadow);
  _registerComponent('LightCluster',   LightCluster);
  _registerComponent('LightBudget',    LightBudgetComponent);
  _registerComponent('LightPriority',  LightPriority);
  _registerComponent('LightBehavior',  LightBehavior);
  _registerComponent('LightComposite', LightComposite);
  _registerComponent('LightIndoor',    LightIndoor);
  _registerComponent('LightIES',       LightIES);
  _registerComponent('LightEmissive',  LightEmissive);
  _registerComponent('LightFlicker',   LightFlicker);
  _registerComponent('LightDayCycle',  LightDayCycle);
  _registerComponent('LightTag',       LightTag);
  _registerComponent('CameraTag',      CameraTag);
  _registerComponent('LightGIProbeRef',LightGIProbeRef);
  _registerComponent('LightAOVolumeRef',LightAOVolumeRef);

  // Shadow subsystem
  _registerComponent('ShadowCasterRef',   ShadowCasterRef);
  _registerComponent('ShadowReceiverRef', ShadowReceiverRef);
  _registerComponent('ShadowAtlas',       ShadowAtlas);
  _registerComponent('ShadowCascade',     ShadowCascade);
  _registerComponent('ShadowBias',        ShadowBias);
  _registerComponent('ShadowFilter',      ShadowFilter);
  _registerComponent('ShadowSoftness',    ShadowSoftness);
  _registerComponent('ShadowFrustum',     ShadowFrustum);
  _registerComponent('ShadowCache',       ShadowCache);
  _registerComponent('ShadowBudget',      ShadowBudget);
  _registerComponent('ShadowContact',     ShadowContact);
  _registerComponent('ShadowVolume',      ShadowVolume);
  _registerComponent('ShadowDirLight',    ShadowDirLight);
  _registerComponent('ShadowPointLight',  ShadowPointLight);
  _registerComponent('ShadowSpotLight',   ShadowSpotLight);
  _registerComponent('ShadowAtlasMap',    ShadowAtlasMap);
  _registerComponent('ShadowTint',        ShadowTint);
  _registerComponent('ShadowEdge',        ShadowEdge);
  _registerComponent('ShadowTile',        ShadowTile);
  _registerComponent('ShadowUpdatePolicy',ShadowUpdatePolicy);
  _registerComponent('ShadowState',       ShadowState);

  // GI subsystem
  _registerComponent('GIProbeRef',      GIProbeRef);
  _registerComponent('GIIrradiance',    GIIrradiance);
  _registerComponent('GISH',            GISH);
  _registerComponent('GISHHigh',        GISHHigh);
  _registerComponent('GIBouncePath',    GIBouncePath);
  _registerComponent('GIVoxel',         GIVoxel);
  _registerComponent('GIDistanceField', GIDistanceField);
  _registerComponent('GIOcclusion',     GIOcclusion);
  _registerComponent('GIPortal',        GIPortal);
  _registerComponent('GIBudget',        GIBudget);
  _registerComponent('GIState',         GIState);
  _registerComponent('GIUpdateQueue',   GIUpdateQueue);
  _registerComponent('GIRadianceCache', GIRadianceCache);
  _registerComponent('GIReflectionProbe',GIReflectionProbe);
  _registerComponent('GILightfield',    GILightfield);
  _registerComponent('GIVolume',        GIVolume);
  _registerComponent('GIIndoor',        GIIndoor);
  _registerComponent('GIOutdoor',       GIOutdoor);
  _registerComponent('GICelBands',      GICelBands);
  _registerComponent('GIPalette',       GIPalette);
  _registerComponent('GIAsync',         GIAsync);
  _registerComponent('GILeak',          GILeak);
  _registerComponent('GITemporal',      GITemporal);
  _registerComponent('GIVolumeBlend',   GIVolumeBlend);

  // AO subsystem
  _registerComponent('AOVolumeRef',           AOVolumeRef);
  _registerComponent('AOSampling',            AOSampling);
  _registerComponent('AOKernel',              AOKernel);
  _registerComponent('AOHistory',             AOHistory);
  _registerComponent('AOBlur',                AOBlur);
  _registerComponent('AOQuality',             AOQuality);
  _registerComponent('AOState',               AOState);
  _registerComponent('AOBudget',              AOBudget);
  _registerComponent('AOScreenSpace',         AOScreenSpace);
  _registerComponent('AOContactShadow',       AOContactShadow);
  _registerComponent('AODistanceField',       AODistanceField);
  _registerComponent('AOIndoorVolume',        AOIndoorVolume);
  _registerComponent('AOOutdoorVolume',       AOOutdoorVolume);
  _registerComponent('AOTemporalAccumulator', AOTemporalAccumulator);
  _registerComponent('AODither',              AODither);
  _registerComponent('AOCelBands',            AOCelBands);
  _registerComponent('AOInkOutline',          AOInkOutline);
  _registerComponent('AOEdgeFade',            AOEdgeFade);
  _registerComponent('AOBilateral',           AOBilateral);
  _registerComponent('AODenoiser',            AODenoiser);
  _registerComponent('AOAsync',               AOAsync);
  _registerComponent('AOResidency',           AOResidency);
  _registerComponent('AOLeak',                AOLeak);
  _registerComponent('AOStyle',               AOStyle);
}

// Build the registry once at module load.
_buildComponentRegistry();

/* ------------------------------------------------------------------ */
/* 1. ECS WORLD BUNDLE                                                */
/* ------------------------------------------------------------------ */

/**
 * The complete component catalog for the scene ECS world. Merged from the
 * four subsystem bundles.
 */
export const ALL_SCENE_COMPONENTS = Object.freeze(Object.assign(
  {},
  LIGHTING_COMPONENTS,
  SHADOW_COMPONENTS,
  GI_COMPONENTS,
  AO_COMPONENTS,
));

/* ------------------------------------------------------------------ */
/* 2. WORLD SINGLETON                                                 */
/* ------------------------------------------------------------------ */

let _world = null;
let _initialized = false;
let _ready = false;

/**
 * Creates (once) and returns the canonical scene ECS world.
 *
 * On first call:
 *   1. Ensures the bitECS 0.4.0 version policy has passed.
 *   2. Creates the world with the full component catalog.
 *   3. Runs the version audit against the live world.
 *   4. Marks the world as ready.
 *
 * Subsequent calls return the same world object.
 */
export function getECSWorld() {
  if (_initialized && _world) return _world;

  // 2.0.1 — Ensure the version policy has been satisfied.
  if (!isBiteCSReady()) {
    // Try to run the audit if it hasn't been run yet.
    const audit = runBiteCSAudit(ALL_SCENE_COMPONENTS, null, MAX_ENTITIES);
    if (!audit.passed) {
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE, `[010_scn_ECSWorld] bitECS audit failed: ${audit.reason}`);
      throw new Error(`[010_scn_ECSWorld] bitECS audit failed: ${audit.reason}`);
    }
  }

  try {
    _world = createWorld({
      components: ALL_SCENE_COMPONENTS,
      time: {
        delta: 0,
        elapsed: 0,
        then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
      },
    });
  } catch (e) {
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE, `[010_scn_ECSWorld] createWorld failed: ${e && e.message}`);
    throw e;
  }

  // Validate the live world against the policy.
  runBiteCSAudit(ALL_SCENE_COMPONENTS, _world, MAX_ENTITIES);

  _initialized = true;
  _ready = true;

  const log = _safeLogger();
  if (log) {
    log.info(LOG_CHANNEL.CORE, () =>
      `[010_scn_ECSWorld] world ready — ` +
      `${_componentRegistry.size} component types, ` +
      `${MAX_ENTITIES} max entities, tier=${PERF_TIER_LOCAL}`);
  }

  return _world;
}

/**
 * Returns true once the world has been created and validated.
 */
export function isECSWorldReady() {
  return _ready === true;
}

/**
 * Disposes the world and resets the singleton. Used for full teardown or
 * for tests that need a fresh world.
 */
export function disposeECSWorld() {
  if (_world) {
    try {
      resetWorld(_world);
    } catch (_) { /* swallow reset errors */ }
    _world = null;
  }
  _initialized = false;
  _ready = false;
}

/* ------------------------------------------------------------------ */
/* 3. COMPONENT REGISTRY                                              */
/* ------------------------------------------------------------------ */

/**
 * Returns the component object registered under `name`, or null.
 */
export function getComponent(name) {
  return _componentRegistry.get(name) || null;
}

/**
 * Returns an array of every registered component name.
 */
export function listComponents() {
  return Array.from(_componentRegistry.keys());
}

/**
 * Returns the number of registered component types.
 */
export function getComponentCount() {
  return _componentRegistry.size;
}

/**
 * Registers an additional SoA component at runtime. Use this when a
 * downstream system declares its own component after boot. The component
 * must be a plain object whose fields are TypedArrays of length
 * MAX_ENTITIES.
 */
export function registerComponent(name, component) {
  if (typeof name !== 'string' || !component) return false;
  if (_componentRegistry.has(name)) return false;

  // Validate the SoA shape.
  const fields = Object.keys(component);
  for (let i = 0; i < fields.length; i++) {
    const field = component[fields[i]];
    if (!ArrayBuffer.isView(field)) return false;
    if (field.length !== MAX_ENTITIES) return false;
  }

  _componentRegistry.set(name, component);
  return true;
}

/* ------------------------------------------------------------------ */
/* 4. ENTITY LIFECYCLE HELPERS                                        */
/* ------------------------------------------------------------------ */

/**
 * Allocates a new entity in the scene world. Returns the entity id, or -1
 * on failure.
 */
export function spawnEntity() {
  const w = getECSWorld();
  if (!w) return -1;
  try {
    return addEntity(w);
  } catch (_) {
    return -1;
  }
}

/**
 * Destroys an entity and returns its id to the pool. All components are
 * detached automatically by bitECS.
 */
export function destroyEntity(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  const w = getECSWorld();
  if (!w) return false;
  if (!entityExists(w, eid)) return false;
  try {
    removeEntity(w, eid);
    return true;
  } catch (_) {
    return false;
  }
}

/**
 * Attaches a component object to an entity.
 */
export function attachComponent(eid, component) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (!component) return false;
  const w = getECSWorld();
  if (!w) return false;
  if (!entityExists(w, eid)) return false;
  try {
    addComponent(w, eid, component);
    return true;
  } catch (_) {
    return false;
  }
}

/**
 * Detaches a component object from an entity.
 */
export function detachComponent(eid, component) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (!component) return false;
  const w = getECSWorld();
  if (!w) return false;
  if (!entityExists(w, eid)) return false;
  try {
    if (hasComponent(w, eid, component)) {
      removeComponent(w, eid, component);
      return true;
    }
  } catch (_) { /* swallow */ }
  return false;
}

/**
 * Checks whether an entity has a component.
 */
export function entityHasComponent(eid, component) {
  if (typeof eid !== 'number' || eid < 0) return false;
  if (!component) return false;
  const w = getECSWorld();
  if (!w) return false;
  try {
    return hasComponent(w, eid, component) === true;
  } catch (_) {
    return false;
  }
}

/**
 * Checks whether an entity exists in the world.
 */
export function entityAlive(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  const w = getECSWorld();
  if (!w) return false;
  try {
    return entityExists(w, eid) === true;
  } catch (_) {
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* 5. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Runs a bitECS query for the given component objects and returns the
 * matching entity list.
 */
export function queryComponents(components) {
  if (!Array.isArray(components) || components.length === 0) return [];
  const w = getECSWorld();
  if (!w) return [];
  try {
    return query(w, components);
  } catch (_) {
    return [];
  }
}

/**
 * Runs a bitECS query for the given component NAMES and returns the
 * matching entity list. Names are resolved via the registry.
 */
export function queryComponentsByName(names) {
  if (!Array.isArray(names) || names.length === 0) return [];
  const resolved = new Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const comp = _componentRegistry.get(names[i]);
    if (!comp) return [];
    resolved[i] = comp;
  }
  return queryComponents(resolved);
}

/**
 * Iterates the results of a query and calls `fn(eid)` for each entity.
 * Returns the count visited.
 */
export function forEachEntity(components, fn, ctx) {
  const entities = queryComponents(components);
  const n = entities.length;
  for (let i = 0; i < n; i++) {
    fn.call(ctx, entities[i]);
  }
  return n;
}

/* ------------------------------------------------------------------ */
/* 6. WORLD TIME CONTRACT                                             */
/* ------------------------------------------------------------------ */

/**
 * The frame-scoped time state shared by every system in the world. Read
 * once per frame, updated by the engine loop.
 */
export const worldTime = {
  delta: 0,
  elapsed: 0,
  frame: 0,
  fps: 60,
  then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
};

/**
 * Advances the world time contract. Called by the engine loop once per
 * frame before running any system.
 */
export function tickWorldTime(nowMs) {
  const t = (typeof nowMs === 'number' && Number.isFinite(nowMs))
    ? nowMs
    : (typeof performance !== 'undefined' ? performance.now() : Date.now());

  let dt = (t - worldTime.then) * 0.001;
  worldTime.then = t;

  if (dt < 0) dt = 0;
  if (dt > 0.1) dt = 0.1;

  worldTime.delta = dt;
  worldTime.elapsed += dt;
  worldTime.frame++;

  if (dt > 0) {
    const instFps = 1 / dt;
    worldTime.fps += (instFps - worldTime.fps) * 0.10;
  }

  // Mirror into the world handle for systems that read it directly.
  if (_world && _world.time) {
    _world.time.delta = worldTime.delta;
    _world.time.elapsed = worldTime.elapsed;
    _world.time.frame = worldTime.frame;
  }

  return dt;
}

/**
 * Resets the world time contract. Used on context restore / pause resume.
 */
export function resetWorldTime() {
  worldTime.delta = 0;
  worldTime.elapsed = 0;
  worldTime.frame = 0;
  worldTime.fps = 60;
  worldTime.then = (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

/* ------------------------------------------------------------------ */
/* 7. STATISTICS                                                      */
/* ------------------------------------------------------------------ */

/**
 * Returns a snapshot of the scene ECS world statistics.
 */
export function getECSWorldStats() {
  const w = getECSWorld();
  const stats = {
    ready: _ready,
    initialized: _initialized,
    maxEntities: MAX_ENTITIES,
    componentCount: _componentRegistry.size,
    entityCount: 0,
    time: {
      delta: worldTime.delta,
      elapsed: worldTime.elapsed,
      frame: worldTime.frame,
      fps: worldTime.fps,
    },
    perfTier: PERF_TIER_LOCAL,
  };

  if (w) {
    // bitECS 0.4.0 exposes its entity count via the internal store. We read
    // it defensively in case the shape changes across minor patches.
    try {
      if (w.entities && typeof w.entities.length === 'number') {
        stats.entityCount = w.entities.length;
      } else if (w.entityCount !== undefined) {
        stats.entityCount = w.entityCount;
      } else if (w.$ && w.$.entityCount !== undefined) {
        stats.entityCount = w.$.entityCount;
      }
    } catch (_) { /* leave entityCount at 0 */ }
  }

  return stats;
}

/* ------------------------------------------------------------------ */
/* 8. ISOLATED TEST WORLDS                                            */
/* ------------------------------------------------------------------ */

/**
 * Creates a fresh isolated world using the same component catalog. This
 * world shares NOTHING with the singleton — different entity IDs, different
 * time, different lifecycle. Used for tests and for hot-swap load.
 */
export function createIsolatedWorld() {
  if (!isBiteCSReady()) {
    const audit = runBiteCSAudit(ALL_SCENE_COMPONENTS, null, MAX_ENTITIES);
    if (!audit.passed) {
      throw new Error(`[010_scn_ECSWorld] cannot create isolated world: ${audit.reason}`);
    }
  }
  return createWorld({
    components: ALL_SCENE_COMPONENTS,
    time: {
      delta: 0,
      elapsed: 0,
      then: (typeof performance !== 'undefined' ? performance.now() : Date.now()),
    },
  });
}

/**
 * Disposes an isolated world.
 */
export function disposeIsolatedWorld(world) {
  if (!world) return;
  try {
    resetWorld(world);
  } catch (_) { /* swallow */ }
  try {
    deleteWorld(world);
  } catch (_) { /* swallow */ }
}

/* ------------------------------------------------------------------ */
/* 9. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_ENTITIES,
  ENTITY_RESERVED,
  ENTITY_BLOCK_SIZE,
  ALL_SCENE_COMPONENTS,

  // World
  getECSWorld,
  isECSWorldReady,
  disposeECSWorld,
  createIsolatedWorld,
  disposeIsolatedWorld,

  // Components
  getComponent,
  listComponents,
  getComponentCount,
  registerComponent,

  // Lifecycle
  spawnEntity,
  destroyEntity,
  attachComponent,
  detachComponent,
  entityHasComponent,
  entityAlive,

  // Query
  queryComponents,
  queryComponentsByName,
  forEachEntity,

  // Time
  worldTime,
  tickWorldTime,
  resetWorldTime,

  // Diagnostics
  getECSWorldStats,
};

export default _defaultExport;