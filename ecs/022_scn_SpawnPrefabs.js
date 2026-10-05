// File : 022
// name : src/ecs/022_scn_SpawnPrefabs.js
// description : Spawn-prefab module for the scene ECS world of the anime
//               lighting stack on Android mobile. Every pre-configured
//               entity template the lighting stack needs — sun, moon, fire
//               light, neon light, magic glow, interior lamp, window shaft,
//               caustic, aurora, shadow caster, GI probe, GI volume, AO
//               volume, camera, streaming chunk, debug marker — is declared
//               here as a frozen descriptor. `spawnPrefab(world, prefabId,
//               overrides)` acquires an entity from the correct subsystem
//               pool (016), attaches every component the prefab declares
//               (002–005), applies the default tag bits (014), registers a
//               lifetime record (017), and returns the fully initialized
//               entity id.
//
//               Design:
//                 • Fixed-capacity prefab registry — one descriptor per
//                   prefab id, declared once at module load and never
//                   resized.
//                 • Per-prefab defaults — every component field the prefab
//                   cares about is declared as a numeric default or a
//                   frozen tuple. Overrides are applied field-by-field in
//                   a single pass.
//                 • Zero-allocation spawn — every write goes directly into
//                   the SoA arrays; no intermediate objects, no dynamic
//                   prop bag, no closures.
//                 • O(components) per spawn — the prefab's component list
//                   is walked once; each component is attached if the
//                   entity doesn't already have it, and its fields are
//                   written in the same pass.
//                 • Auto-lifetime — every prefab declares a TTL policy
//                   (permanent, transient, scene-scoped) and a default
//                   grace period so the lifetime system (017) can collect
//                   transient entities automatically.
//                 • Auto-pool binding — every prefab declares which
//                   subsystem pool it belongs to, so `releasePrefab(eid)`
//                   always returns the entity to the right place.
//                 • Concurrency-safe — every spawn runs entirely on the
//                   main thread against SoA arrays; no locks, no
//                   cross-thread state.
//                 • Prewarm support — `prewarmPrefab(prefabId, count)` uses
//                   the entity pool (016) to reserve entities in bulk so
//                   later spawns are a strict subset of an already-warm
//                   block.
//
//               Integration:
//                 • 002_lgt_LightComponents.js       — light + shadow components
//                 • 003_lgt_ShadowComponents.js      — shadow components
//                 • 004_lgt_GIComponents.js          — GI components
//                 • 005_lgt_AOComponents.js          — AO components
//                 • 010_scn_ECSWorld.js              — world handle
//                 • 011_scn_BiteCSAdapter.js         — ECS façade
//                 • 014_scn_Tags.js                  — tag bits
//                 • 015_scn_Relations.js             — relationship graph
//                 • 016_scn_EntityPool.js            — subsystem pools
//                 • 017_scn_EntityLifetime.js        — lifetime record
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every entity the anime lighting stack needs
//            is spawned through one canonical, deterministic, allocation-
//            free path — so that every subsystem gets correctly initialized
//            entities with the same default values, the same tag bits, and
//            the same lifetime record.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  entityExists,
  addComponent,
  hasComponent,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

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
  LightTag as LightTagComponent,
  CameraTag as CameraTagComponent,
  LIGHT_TYPE,
  LIGHT_KIND,
  LIGHT_BEHAVIOR,
  LIGHT_PRIORITY,
  LIGHT_STATE_FLAG,
  LIGHT_ENVIRONMENT,
  LIGHT_TAG,
  MAX_BEHAVIORS_PER_LIGHT,
} from './002_lgt_LightComponents.js';

import {
  ShadowCasterRef,
  ShadowAtlas,
  ShadowCascade,
  ShadowBias,
  ShadowFilter,
  ShadowSoftness,
  ShadowFrustum,
  ShadowCache,
  ShadowBudget,
  ShadowContact,
  ShadowDirLight,
  ShadowPointLight,
  ShadowSpotLight,
  ShadowAtlasMap,
  ShadowTint,
  ShadowEdge,
  ShadowUpdatePolicy,
  ShadowState,
  SHADOW_FILTER,
  SHADOW_STATE,
  SHADOW_UPDATE_MODE,
  SHADOW_ATLAS_MODE,
  SHADOW_EDGE,
  SHADOW_SOURCE,
  MAX_SHADOW_CASCADES,
} from './003_lgt_ShadowComponents.js';

import {
  GIProbeRef,
  GIIrradiance,
  GISH,
  GIBouncePath,
  GIOcclusion,
  GIPortal,
  GIBudget,
  GIState,
  GICelBands,
  GIPalette,
  GILeak,
  GITemporal,
  GIVolume,
  GIIndoor,
  GIOutdoor,
  GIReflectionProbe,
  GILightfield,
  GIVolumeBlend,
  GI_STATE,
  GI_QUALITY,
  GI_MODE,
  GI_PROBE_TYPE,
  GI_PORTAL_TYPE,
  GI_LEAK_MODE,
  MAX_SH_COEFFICIENTS,
} from './004_lgt_GIComponents.js';

import {
  AOVolumeRef,
  AOSampling,
  AOBlur,
  AOQuality,
  AOState,
  AOBudget,
  AOContactShadow,
  AOIndoorVolume,
  AOOutdoorVolume,
  AOTemporalAccumulator,
  AODither,
  AOCelBands,
  AOInkOutline,
  AOEdgeFade,
  AOBilateral,
  AODenoiser,
  AOResidency,
  AOLeak,
  AOStyle,
  AO_METHOD,
  AO_QUALITY,
  AO_STATE,
  AO_BLUR_MODE,
  AO_TEMPORAL_MODE,
  AO_DITHER,
  AO_STYLE,
  AO_EDGE,
} from './005_lgt_AOComponents.js';

import {
  getECSWorld,
  attachComponent,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  getTypeId,
} from './013_scn_ComponentTypes.js';

import {
  TAG,
  TAG2,
  tagEntity,
  untagEntity,
  tagEntity2,
  untagEntity2,
  tagAsLight,
  tagAsShadowParticipant,
  tagAsGIProbe,
  tagAsAOVolume,
  tagAsBiome,
  syncTagsToSoA,
} from './014_scn_Tags.js';

import {
  attachChild,
  detachChild,
  setReference,
  clearReference,
  REF,
} from './015_scn_Relations.js';

import {
  POOL,
  acquire,
  release,
  isInUse,
  acquireLightEntity,
  acquireShadowEntity,
  acquireCameraEntity,
  acquireGIEntity,
  acquireAOEntity,
  acquireSceneEntity,
  acquireStreamingEntity,
  acquireDebugEntity,
} from './016_scn_EntityPool.js';

import {
  LIFETIME_STATE,
  LIFETIME_POOL,
  EntityLifetime,
  markSpawning,
  markAlive,
  markDying,
  setTTL,
  setPersistent,
  setTransient,
  DEFAULT_DYING_GRACE_FRAMES,
} from './017_scn_EntityLifetime.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Canonical prefab ids. Every prefab the lighting stack needs is declared
 * here. Downstream systems call `spawnPrefab(world, PREFAB.SUN)` and get
 * back a fully initialized entity.
 */
export const PREFAB = Object.freeze({
  /* ---------------- LIGHTS ---------------- */
  SUN:                    0,
  MOON:                   1,
  AMBIENT:                2,
  HEMISPHERE:             3,
  SKY_LIGHT:              4,

  FIRE_LIGHT:             5,
  NEON_LIGHT:             6,
  MAGIC_GLOW:             7,
  INTERIOR_LAMP:          8,
  WINDOW_SHAFT:           9,
  CAUSTIC:               10,
  AURORA:                11,
  BIO_LUMINESCENT:       12,

  POINT_LIGHT:           13,
  SPOT_LIGHT:            14,
  RECT_AREA_LIGHT:       15,

  /* ---------------- SHADOW ---------------- */
  SHADOW_CASTER:         20,
  SHADOW_RECEIVER:       21,
  SHADOW_ATLAS:          22,
  SHADOW_CASCADE_SET:    23,
  CONTACT_SHADOW:        24,

  /* ---------------- GI ---------------- */
  GI_PROBE:              30,
  GI_HERO_PROBE:         31,
  GI_VOLUME_INDOOR:      32,
  GI_VOLUME_OUTDOOR:     33,
  GI_PORTAL_DOORWAY:     34,
  GI_PORTAL_WINDOW:      35,
  GI_PORTAL_SKYLIGHT:    36,
  GI_PORTAL_CAVE:        37,
  REFLECTION_PROBE:      38,
  LIGHTFIELD:            39,

  /* ---------------- AO ---------------- */
  AO_VOLUME:             50,
  AO_INDOOR_VOLUME:      51,
  AO_OUTDOOR_VOLUME:     52,
  AO_CONTACT_VOLUME:     53,
  AO_CEL_CONTROLLER:     54,
  AO_INK_CONTROLLER:     55,
  AO_DENOISER:           56,
  AO_BILATERAL:          57,

  /* ---------------- CAMERA / SCENE ---------------- */
  CAMERA:                70,
  ORBIT_CAMERA:          71,
  DOF_CAMERA:            72,

  /* ---------------- STREAMING / DEBUG ---------------- */
  STREAMING_CHUNK:       80,
  LOD_GROUP:             81,
  DEBUG_MARKER:          90,
  DEBUG_HUD:             91,
});

export const PREFAB_NAME = Object.freeze([
  'sun', 'moon', 'ambient', 'hemisphere', 'sky_light',
  'fire_light', 'neon_light', 'magic_glow', 'interior_lamp',
  'window_shaft', 'caustic', 'aurora', 'bio_luminescent',
  'point_light', 'spot_light', 'rect_area_light',
  'unused_16', 'unused_17', 'unused_18', 'unused_19',
  'shadow_caster', 'shadow_receiver', 'shadow_atlas', 'shadow_cascade_set',
  'contact_shadow', 'unused_25', 'unused_26', 'unused_27', 'unused_28', 'unused_29',
  'gi_probe', 'gi_hero_probe', 'gi_volume_indoor', 'gi_volume_outdoor',
  'gi_portal_doorway', 'gi_portal_window', 'gi_portal_skylight', 'gi_portal_cave',
  'reflection_probe', 'lightfield',
  'unused_40', 'unused_41', 'unused_42', 'unused_43', 'unused_44',
  'unused_45', 'unused_46', 'unused_47', 'unused_48', 'unused_49',
  'ao_volume', 'ao_indoor_volume', 'ao_outdoor_volume', 'ao_contact_volume',
  'ao_cel_controller', 'ao_ink_controller', 'ao_denoiser', 'ao_bilateral',
  'unused_58', 'unused_59', 'unused_60', 'unused_61', 'unused_62', 'unused_63',
  'unused_64', 'unused_65', 'unused_66', 'unused_67', 'unused_68', 'unused_69',
  'camera', 'orbit_camera', 'dof_camera',
  'unused_73', 'unused_74', 'unused_75', 'unused_76', 'unused_77', 'unused_78', 'unused_79',
  'streaming_chunk', 'lod_group',
  'unused_82', 'unused_83', 'unused_84', 'unused_85', 'unused_86', 'unused_87', 'unused_88', 'unused_89',
  'debug_marker', 'debug_hud',
]);

export const MAX_PREFABS = 128;

/**
 * Prefab lifetime policies. Every prefab declares one; the spawn function
 * applies the corresponding defaults to the lifetime record.
 */
export const PREFAB_LIFETIME = Object.freeze({
  PERMANENT:  0,   // infinite TTL, persistent
  SCENE:      1,   // infinite TTL, not persistent (released with scene)
  TRANSIENT:  2,   // short TTL, transient
  EPHEMERAL:  3,   // very short TTL, transient, immediate dying on release
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const PrefabState = {
  totalSpawns:        0,
  totalReleases:      0,
  totalRejected:      0,
  totalByPrefab:      new Uint32Array(MAX_PREFABS),
  peakLiveByPrefab:   new Uint32Array(MAX_PREFABS),
  liveByPrefab:       new Uint32Array(MAX_PREFABS),
  frame:              0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.spawn_prefabs', {
        tag: BOUNDARY_TAG.GENERIC,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/* ------------------------------------------------------------------ */
/* 2. PREFAB DESCRIPTOR                                               */
/* ------------------------------------------------------------------ */

/**
 * One prefab descriptor. Every field has a numeric default so overrides
 * can be applied in a single pass. All fields are frozen once declared.
 */
export class PrefabDescriptor {
  constructor(spec) {
    this.id          = spec.id;
    this.name        = spec.name || PREFAB_NAME[spec.id] || ('prefab_' + spec.id);
    this.pool        = spec.pool !== undefined ? spec.pool : POOL.SCENE;
    this.lifetime    = spec.lifetime !== undefined ? spec.lifetime : PREFAB_LIFETIME.SCENE;
    this.ttlFrames   = spec.ttlFrames !== undefined ? spec.ttlFrames : 0;

    // Component list (resolved from the world/registry at spawn time).
    this.componentNames = Object.freeze((spec.components || []).slice());

    // Tag bits applied at spawn.
    this.tagMask     = (spec.tagMask !== undefined ? spec.tagMask : 0) >>> 0;
    this.tag2Mask    = (spec.tag2Mask !== undefined ? spec.tag2Mask : 0) >>> 0;

    // Field defaults. Object shape: { componentName: { fieldName: value } }.
    this.defaults = Object.freeze(_freezeDefaults(spec.defaults || {}));

    // Behavior attach list — array of { id, ctx }.
    this.behaviors = Object.freeze((spec.behaviors || []).map((b) => Object.freeze({
      id:  b.id,
      ctx0: b.ctx0 !== undefined ? b.ctx0 : 0,
      ctx1: b.ctx1 !== undefined ? b.ctx1 : 0,
      ctx2: b.ctx2 !== undefined ? b.ctx2 : 0,
    })));

    // Cap on total live entities of this prefab (0 = unlimited).
    this.maxLive     = spec.maxLive !== undefined ? spec.maxLive : 0;

    Object.freeze(this);
  }
}

function _freezeDefaults(d) {
  const out = {};
  for (const compName in d) {
    const fields = d[compName];
    const frozenFields = {};
    for (const fieldName in fields) {
      const v = fields[fieldName];
      frozenFields[fieldName] = Array.isArray(v) ? Object.freeze(v.slice()) : v;
    }
    out[compName] = Object.freeze(frozenFields);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 3. PREFAB REGISTRY                                                 */
/* ------------------------------------------------------------------ */

const _prefabs = new Array(MAX_PREFABS).fill(null);

function _register(prefabId, spec) {
  if (prefabId < 0 || prefabId >= MAX_PREFABS) return false;
  const descriptor = new PrefabDescriptor(Object.assign({ id: prefabId }, spec));
  _prefabs[prefabId] = descriptor;
  return true;
}

/**
 * Returns the descriptor for a prefab id, or null.
 */
export function getPrefab(prefabId) {
  if (prefabId < 0 || prefabId >= MAX_PREFABS) return null;
  return _prefabs[prefabId];
}

/**
 * Returns the descriptor for a prefab name, or null.
 */
export function getPrefabByName(name) {
  if (typeof name !== 'string') return null;
  for (let i = 0; i < MAX_PREFABS; i++) {
    if (_prefabs[i] && _prefabs[i].name === name) return _prefabs[i];
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* 4. FIELD WRITER TABLE                                              */
/* ------------------------------------------------------------------ */

/**
 * Every component field the prefab defaults can write to has a fast
 * writer entry here. This keeps the hot path branch-free and lets the
 * spawn function iterate defaults in declaration order.
 *
 * Field writer signature: (eid, value) => void.
 */
const _fieldWriters = new Map();

function _registerField(componentName, fieldName, writer) {
  const key = componentName + '.' + fieldName;
  _fieldWriters.set(key, writer);
}

(function _registerAllFields() {
  /* ---------------- Transform ---------------- */
  _registerField('Transform', 'x',  (eid, v) => { Transform.x[eid] = v; });
  _registerField('Transform', 'y',  (eid, v) => { Transform.y[eid] = v; });
  _registerField('Transform', 'z',  (eid, v) => { Transform.z[eid] = v; });
  _registerField('Transform', 'qx', (eid, v) => { Transform.qx[eid] = v; });
  _registerField('Transform', 'qy', (eid, v) => { Transform.qy[eid] = v; });
  _registerField('Transform', 'qz', (eid, v) => { Transform.qz[eid] = v; });
  _registerField('Transform', 'qw', (eid, v) => { Transform.qw[eid] = v; });
  _registerField('Transform', 'sx', (eid, v) => { Transform.sx[eid] = v; });
  _registerField('Transform', 'sy', (eid, v) => { Transform.sy[eid] = v; });
  _registerField('Transform', 'sz', (eid, v) => { Transform.sz[eid] = v; });

  /* ---------------- Target ---------------- */
  _registerField('Target', 'x',      (eid, v) => { Target.x[eid] = v; });
  _registerField('Target', 'y',      (eid, v) => { Target.y[eid] = v; });
  _registerField('Target', 'z',      (eid, v) => { Target.z[eid] = v; });
  _registerField('Target', 'active', (eid, v) => { Target.active[eid] = v; });

  /* ---------------- LightRef ---------------- */
  _registerField('LightRef', 'type',         (eid, v) => { LightRef.type[eid] = v; });
  _registerField('LightRef', 'kind',         (eid, v) => { LightRef.kind[eid] = v; });
  _registerField('LightRef', 'colorR',       (eid, v) => { LightRef.colorR[eid] = v; });
  _registerField('LightRef', 'colorG',       (eid, v) => { LightRef.colorG[eid] = v; });
  _registerField('LightRef', 'colorB',       (eid, v) => { LightRef.colorB[eid] = v; });
  _registerField('LightRef', 'intensity',    (eid, v) => { LightRef.intensity[eid] = v; });
  _registerField('LightRef', 'range',        (eid, v) => { LightRef.range[eid] = v; });
  _registerField('LightRef', 'decay',        (eid, v) => { LightRef.decay[eid] = v; });
  _registerField('LightRef', 'angle',        (eid, v) => { LightRef.angle[eid] = v; });
  _registerField('LightRef', 'penumbra',     (eid, v) => { LightRef.penumbra[eid] = v; });
  _registerField('LightRef', 'width',        (eid, v) => { LightRef.width[eid] = v; });
  _registerField('LightRef', 'height',       (eid, v) => { LightRef.height[eid] = v; });
  _registerField('LightRef', 'groundColorR', (eid, v) => { LightRef.groundColorR[eid] = v; });
  _registerField('LightRef', 'groundColorG', (eid, v) => { LightRef.groundColorG[eid] = v; });
  _registerField('LightRef', 'groundColorB', (eid, v) => { LightRef.groundColorB[eid] = v; });

  /* ---------------- LightState ---------------- */
  _registerField('LightState', 'flags',     (eid, v) => { LightState.flags[eid] = v; });
  _registerField('LightState', 'env',       (eid, v) => { LightState.env[eid] = v; });
  _registerField('LightState', 'envBlend',  (eid, v) => { LightState.envBlend[eid] = v; });

  /* ---------------- LightShadow ---------------- */
  _registerField('LightShadow', 'enabled',      (eid, v) => { LightShadow.enabled[eid] = v; });
  _registerField('LightShadow', 'mapSize',      (eid, v) => { LightShadow.mapSize[eid] = v; });
  _registerField('LightShadow', 'cascadeCount', (eid, v) => { LightShadow.cascadeCount[eid] = v; });
  _registerField('LightShadow', 'filter',       (eid, v) => { LightShadow.filter[eid] = v; });
  _registerField('LightShadow', 'bias',         (eid, v) => { LightShadow.bias[eid] = v; });
  _registerField('LightShadow', 'normalBias',   (eid, v) => { LightShadow.normalBias[eid] = v; });
  _registerField('LightShadow', 'softness',     (eid, v) => { LightShadow.softness[eid] = v; });
  _registerField('LightShadow', 'distance',     (eid, v) => { LightShadow.distance[eid] = v; });

  /* ---------------- LightPriority ---------------- */
  _registerField('LightPriority', 'bucket',  (eid, v) => { LightPriority.bucket[eid] = v; });
  _registerField('LightPriority', 'sortKey', (eid, v) => { LightPriority.sortKey[eid] = v; });

  /* ---------------- LightFlicker ---------------- */
  _registerField('LightFlicker', 'baseIntensity', (eid, v) => { LightFlicker.baseIntensity[eid] = v; });
  _registerField('LightFlicker', 'amplitude',     (eid, v) => { LightFlicker.amplitude[eid] = v; });
  _registerField('LightFlicker', 'hz',            (eid, v) => { LightFlicker.hz[eid] = v; });
  _registerField('LightFlicker', 'phase',         (eid, v) => { LightFlicker.phase[eid] = v; });
  _registerField('LightFlicker', 'enabled',       (eid, v) => { LightFlicker.enabled[eid] = v; });

  /* ---------------- LightDayCycle ---------------- */
  _registerField('LightDayCycle', 'dayCycle',     (eid, v) => { LightDayCycle.dayCycle[eid] = v; });
  _registerField('LightDayCycle', 'daySpeed',     (eid, v) => { LightDayCycle.daySpeed[eid] = v; });
  _registerField('LightDayCycle', 'enabled',      (eid, v) => { LightDayCycle.enabled[eid] = v; });
  _registerField('LightDayCycle', 'phaseOffset',  (eid, v) => { LightDayCycle.phaseOffset[eid] = v; });
  _registerField('LightDayCycle', 'maxElevation', (eid, v) => { LightDayCycle.maxElevation[eid] = v; });

  /* ---------------- LightEmissive ---------------- */
  _registerField('LightEmissive', 'emissiveR',    (eid, v) => { LightEmissive.emissiveR[eid] = v; });
  _registerField('LightEmissive', 'emissiveG',    (eid, v) => { LightEmissive.emissiveG[eid] = v; });
  _registerField('LightEmissive', 'emissiveB',    (eid, v) => { LightEmissive.emissiveB[eid] = v; });
  _registerField('LightEmissive', 'emissiveScale',(eid, v) => { LightEmissive.emissiveScale[eid] = v; });
  _registerField('LightEmissive', 'proxyVisible', (eid, v) => { LightEmissive.proxyVisible[eid] = v; });

  /* ---------------- LightIndoor ---------------- */
  _registerField('LightIndoor', 'indoorWeight',   (eid, v) => { LightIndoor.indoorWeight[eid] = v; });
  _registerField('LightIndoor', 'outdoorWeight',  (eid, v) => { LightIndoor.outdoorWeight[eid] = v; });

  /* ---------------- LightCluster ---------------- */
  _registerField('LightCluster', 'cellX', (eid, v) => { LightCluster.cellX[eid] = v; });
  _registerField('LightCluster', 'cellY', (eid, v) => { LightCluster.cellY[eid] = v; });
  _registerField('LightCluster', 'cellZ', (eid, v) => { LightCluster.cellZ[eid] = v; });

  /* ---------------- LightBudget ---------------- */
  _registerField('LightBudget', 'cost', (eid, v) => { LightBudgetComponent.cost[eid] = v; });
  _registerField('LightBudget', 'lod',  (eid, v) => { LightBudgetComponent.lod[eid] = v; });

  /* ---------------- CameraTag ---------------- */
  _registerField('CameraTag', 'active', (eid, v) => { CameraTagComponent.active[eid] = v; });
  _registerField('CameraTag', 'fov',    (eid, v) => { CameraTagComponent.fov[eid] = v; });
  _registerField('CameraTag', 'near',   (eid, v) => { CameraTagComponent.near[eid] = v; });
  _registerField('CameraTag', 'far',    (eid, v) => { CameraTagComponent.far[eid] = v; });
  _registerField('CameraTag', 'aspect', (eid, v) => { CameraTagComponent.aspect[eid] = v; });

  /* ---------------- ShadowCasterRef ---------------- */
  _registerField('ShadowCasterRef', 'casterKind',    (eid, v) => { ShadowCasterRef.casterKind[eid] = v; });
  _registerField('ShadowCasterRef', 'castStrength',  (eid, v) => { ShadowCasterRef.castStrength[eid] = v; });
  _registerField('ShadowCasterRef', 'casterRadius',  (eid, v) => { ShadowCasterRef.casterRadius[eid] = v; });
  _registerField('ShadowCasterRef', 'enabled',       (eid, v) => { ShadowCasterRef.enabled[eid] = v; });

  /* ---------------- ShadowBias ---------------- */
  _registerField('ShadowBias', 'bias',        (eid, v) => { ShadowBias.bias[eid] = v; });
  _registerField('ShadowBias', 'normalBias',  (eid, v) => { ShadowBias.normalBias[eid] = v; });

  /* ---------------- ShadowFilter ---------------- */
  _registerField('ShadowFilter', 'mode',       (eid, v) => { ShadowFilter.mode[eid] = v; });
  _registerField('ShadowFilter', 'kernelSize', (eid, v) => { ShadowFilter.kernelSize[eid] = v; });

  /* ---------------- ShadowSoftness ---------------- */
  _registerField('ShadowSoftness', 'softness',    (eid, v) => { ShadowSoftness.softness[eid] = v; });
  _registerField('ShadowSoftness', 'edgeStyle',   (eid, v) => { ShadowSoftness.edgeStyle[eid] = v; });
  _registerField('ShadowSoftness', 'bandCount',   (eid, v) => { ShadowSoftness.bandCount[eid] = v; });

  /* ---------------- ShadowAtlas ---------------- */
  _registerField('ShadowAtlas', 'mode',       (eid, v) => { ShadowAtlas.mode[eid] = v; });
  _registerField('ShadowAtlas', 'atlasWidth', (eid, v) => { ShadowAtlas.atlasWidth[eid] = v; });
  _registerField('ShadowAtlas', 'atlasHeight',(eid, v) => { ShadowAtlas.atlasHeight[eid] = v; });

  /* ---------------- ShadowTint ---------------- */
  _registerField('ShadowTint', 'tintR',       (eid, v) => { ShadowTint.tintR[eid] = v; });
  _registerField('ShadowTint', 'tintG',       (eid, v) => { ShadowTint.tintG[eid] = v; });
  _registerField('ShadowTint', 'tintB',       (eid, v) => { ShadowTint.tintB[eid] = v; });
  _registerField('ShadowTint', 'tintStrength',(eid, v) => { ShadowTint.tintStrength[eid] = v; });
  _registerField('ShadowTint', 'enabled',     (eid, v) => { ShadowTint.enabled[eid] = v; });

  /* ---------------- ShadowUpdatePolicy ---------------- */
  _registerField('ShadowUpdatePolicy', 'mode',     (eid, v) => { ShadowUpdatePolicy.mode[eid] = v; });
  _registerField('ShadowUpdatePolicy', 'interval', (eid, v) => { ShadowUpdatePolicy.interval[eid] = v; });

  /* ---------------- ShadowContact ---------------- */
  _registerField('ShadowContact', 'enabled',   (eid, v) => { ShadowContact.enabled[eid] = v; });
  _registerField('ShadowContact', 'radius',    (eid, v) => { ShadowContact.radius[eid] = v; });
  _registerField('ShadowContact', 'strength',  (eid, v) => { ShadowContact.strength[eid] = v; });

  /* ---------------- ShadowDirLight ---------------- */
  _registerField('ShadowDirLight', 'cascades',   (eid, v) => { ShadowDirLight.cascades[eid] = v; });
  _registerField('ShadowDirLight', 'stabilize',  (eid, v) => { ShadowDirLight.stabilize[eid] = v; });
  _registerField('ShadowDirLight', 'texelSnap',  (eid, v) => { ShadowDirLight.texelSnap[eid] = v; });

  /* ---------------- GIProbeRef ---------------- */
  _registerField('GIProbeRef', 'type',         (eid, v) => { GIProbeRef.type[eid] = v; });
  _registerField('GIProbeRef', 'quality',      (eid, v) => { GIProbeRef.quality[eid] = v; });
  _registerField('GIProbeRef', 'mode',         (eid, v) => { GIProbeRef.mode[eid] = v; });
  _registerField('GIProbeRef', 'radius',       (eid, v) => { GIProbeRef.radius[eid] = v; });
  _registerField('GIProbeRef', 'enabled',      (eid, v) => { GIProbeRef.enabled[eid] = v; });
  _registerField('GIProbeRef', 'indoorFactor', (eid, v) => { GIProbeRef.indoorFactor[eid] = v; });
  _registerField('GIProbeRef', 'gridX',        (eid, v) => { GIProbeRef.gridX[eid] = v; });
  _registerField('GIProbeRef', 'gridY',        (eid, v) => { GIProbeRef.gridY[eid] = v; });
  _registerField('GIProbeRef', 'gridZ',        (eid, v) => { GIProbeRef.gridZ[eid] = v; });
  _registerField('GIProbeRef', 'worldX',       (eid, v) => { GIProbeRef.worldX[eid] = v; });
  _registerField('GIProbeRef', 'worldY',       (eid, v) => { GIProbeRef.worldY[eid] = v; });
  _registerField('GIProbeRef', 'worldZ',       (eid, v) => { GIProbeRef.worldZ[eid] = v; });

  /* ---------------- GIIrradiance ---------------- */
  _registerField('GIIrradiance', 'r',          (eid, v) => { GIIrradiance.r[eid] = v; });
  _registerField('GIIrradiance', 'g',          (eid, v) => { GIIrradiance.g[eid] = v; });
  _registerField('GIIrradiance', 'b',          (eid, v) => { GIIrradiance.b[eid] = v; });
  _registerField('GIIrradiance', 'confidence', (eid, v) => { GIIrradiance.confidence[eid] = v; });

  /* ---------------- GIOcclusion ---------------- */
  _registerField('GIOcclusion', 'skyOcclusion',      (eid, v) => { GIOcclusion.skyOcclusion[eid] = v; });
  _registerField('GIOcclusion', 'obstacleOcclusion', (eid, v) => { GIOcclusion.obstacleOcclusion[eid] = v; });

  /* ---------------- GIBudget ---------------- */
  _registerField('GIBudget', 'cost',     (eid, v) => { GIBudget.cost[eid] = v; });
  _registerField('GIBudget', 'priority', (eid, v) => { GIBudget.priority[eid] = v; });
  _registerField('GIBudget', 'lod',      (eid, v) => { GIBudget.lod[eid] = v; });

  /* ---------------- GIState ---------------- */
  _registerField('GIState', 'state', (eid, v) => { GIState.state[eid] = v; });

  /* ---------------- GICelBands ---------------- */
  _registerField('GICelBands', 'enabled',        (eid, v) => { GICelBands.enabled[eid] = v; });
  _registerField('GICelBands', 'bandCount',      (eid, v) => { GICelBands.bandCount[eid] = v; });
  _registerField('GICelBands', 'bandSoftness',   (eid, v) => { GICelBands.bandSoftness[eid] = v; });
  _registerField('GICelBands', 'ditherStrength', (eid, v) => { GICelBands.ditherStrength[eid] = v; });

  /* ---------------- GIPalette ---------------- */
  _registerField('GIPalette', 'styleId',          (eid, v) => { GIPalette.styleId[eid] = v; });
  _registerField('GIPalette', 'satBias',          (eid, v) => { GIPalette.satBias[eid] = v; });
  _registerField('GIPalette', 'hueBias',          (eid, v) => { GIPalette.hueBias[eid] = v; });
  _registerField('GIPalette', 'ambientColorR',    (eid, v) => { GIPalette.ambientColorR[eid] = v; });
  _registerField('GIPalette', 'ambientColorG',    (eid, v) => { GIPalette.ambientColorG[eid] = v; });
  _registerField('GIPalette', 'ambientColorB',    (eid, v) => { GIPalette.ambientColorB[eid] = v; });
  _registerField('GIPalette', 'lerpRate',         (eid, v) => { GIPalette.lerpRate[eid] = v; });
  _registerField('GIPalette', 'enabled',          (eid, v) => { GIPalette.enabled[eid] = v; });

  /* ---------------- GILeak ---------------- */
  _registerField('GILeak', 'mode',              (eid, v) => { GILeak.mode[eid] = v; });
  _registerField('GILeak', 'threshold',         (eid, v) => { GILeak.threshold[eid] = v; });
  _registerField('GILeak', 'correctionFactor',  (eid, v) => { GILeak.correctionFactor[eid] = v; });

  /* ---------------- GITemporal ---------------- */
  _registerField('GITemporal', 'historyWeight',      (eid, v) => { GITemporal.historyWeight[eid] = v; });
  _registerField('GITemporal', 'reprojectionBias',   (eid, v) => { GITemporal.reprojectionBias[eid] = v; });
  _registerField('GITemporal', 'rejectionThreshold', (eid, v) => { GITemporal.rejectionThreshold[eid] = v; });
  _registerField('GITemporal', 'valid',              (eid, v) => { GITemporal.valid[eid] = v; });

  /* ---------------- GIVolume ---------------- */
  _registerField('GIVolume', 'kind',          (eid, v) => { GIVolume.kind[eid] = v; });
  _registerField('GIVolume', 'enabled',       (eid, v) => { GIVolume.enabled[eid] = v; });
  _registerField('GIVolume', 'minX',          (eid, v) => { GIVolume.minX[eid] = v; });
  _registerField('GIVolume', 'minY',          (eid, v) => { GIVolume.minY[eid] = v; });
  _registerField('GIVolume', 'minZ',          (eid, v) => { GIVolume.minZ[eid] = v; });
  _registerField('GIVolume', 'maxX',          (eid, v) => { GIVolume.maxX[eid] = v; });
  _registerField('GIVolume', 'maxY',          (eid, v) => { GIVolume.maxY[eid] = v; });
  _registerField('GIVolume', 'maxZ',          (eid, v) => { GIVolume.maxZ[eid] = v; });
  _registerField('GIVolume', 'fillR',         (eid, v) => { GIVolume.fillR[eid] = v; });
  _registerField('GIVolume', 'fillG',         (eid, v) => { GIVolume.fillG[eid] = v; });
  _registerField('GIVolume', 'fillB',         (eid, v) => { GIVolume.fillB[eid] = v; });
  _registerField('GIVolume', 'blendDistance', (eid, v) => { GIVolume.blendDistance[eid] = v; });
  _registerField('GIVolume', 'probeDensity',  (eid, v) => { GIVolume.probeDensity[eid] = v; });
  _registerField('GIVolume', 'leakGate',      (eid, v) => { GIVolume.leakGate[eid] = v; });

  /* ---------------- GIIndoor ---------------- */
  _registerField('GIIndoor', 'wallOcclusion',       (eid, v) => { GIIndoor.wallOcclusion[eid] = v; });
  _registerField('GIIndoor', 'curtainTransmission', (eid, v) => { GIIndoor.curtainTransmission[eid] = v; });
  _registerField('GIIndoor', 'enabled',             (eid, v) => { GIIndoor.enabled[eid] = v; });

  /* ---------------- GIOutdoor ---------------- */
  _registerField('GIOutdoor', 'biomeWeight', (eid, v) => { GIOutdoor.biomeWeight[eid] = v; });
  _registerField('GIOutdoor', 'enabled',     (eid, v) => { GIOutdoor.enabled[eid] = v; });

  /* ---------------- GIPortal ---------------- */
  _registerField('GIPortal', 'type',         (eid, v) => { GIPortal.type[eid] = v; });
  _registerField('GIPortal', 'enabled',      (eid, v) => { GIPortal.enabled[eid] = v; });
  _registerField('GIPortal', 'width',        (eid, v) => { GIPortal.width[eid] = v; });
  _registerField('GIPortal', 'height',       (eid, v) => { GIPortal.height[eid] = v; });
  _registerField('GIPortal', 'transmission', (eid, v) => { GIPortal.transmission[eid] = v; });
  _registerField('GIPortal', 'positionX',    (eid, v) => { GIPortal.positionX[eid] = v; });
  _registerField('GIPortal', 'positionY',    (eid, v) => { GIPortal.positionY[eid] = v; });
  _registerField('GIPortal', 'positionZ',    (eid, v) => { GIPortal.positionZ[eid] = v; });
  _registerField('GIPortal', 'normalX',      (eid, v) => { GIPortal.normalX[eid] = v; });
  _registerField('GIPortal', 'normalY',      (eid, v) => { GIPortal.normalY[eid] = v; });
  _registerField('GIPortal', 'normalZ',      (eid, v) => { GIPortal.normalZ[eid] = v; });

  /* ---------------- GIReflectionProbe ---------------- */
  _registerField('GIReflectionProbe', 'positionX',     (eid, v) => { GIReflectionProbe.positionX[eid] = v; });
  _registerField('GIReflectionProbe', 'positionY',     (eid, v) => { GIReflectionProbe.positionY[eid] = v; });
  _registerField('GIReflectionProbe', 'positionZ',     (eid, v) => { GIReflectionProbe.positionZ[eid] = v; });
  _registerField('GIReflectionProbe', 'radius',        (eid, v) => { GIReflectionProbe.radius[eid] = v; });
  _registerField('GIReflectionProbe', 'resolution',    (eid, v) => { GIReflectionProbe.resolution[eid] = v; });
  _registerField('GIReflectionProbe', 'updateInterval',(eid, v) => { GIReflectionProbe.updateInterval[eid] = v; });
  _registerField('GIReflectionProbe', 'enabled',       (eid, v) => { GIReflectionProbe.enabled[eid] = v; });

  /* ---------------- GILightfield ---------------- */
  _registerField('GILightfield', 'sampleCount', (eid, v) => { GILightfield.sampleCount[0] = v; });

  /* ---------------- AOVolumeRef ---------------- */
  _registerField('AOVolumeRef', 'method',       (eid, v) => { AOVolumeRef.method[eid] = v; });
  _registerField('AOVolumeRef', 'quality',      (eid, v) => { AOVolumeRef.quality[eid] = v; });
  _registerField('AOVolumeRef', 'style',        (eid, v) => { AOVolumeRef.style[eid] = v; });
  _registerField('AOVolumeRef', 'enabled',      (eid, v) => { AOVolumeRef.enabled[eid] = v; });
  _registerField('AOVolumeRef', 'minX',         (eid, v) => { AOVolumeRef.minX[eid] = v; });
  _registerField('AOVolumeRef', 'minY',         (eid, v) => { AOVolumeRef.minY[eid] = v; });
  _registerField('AOVolumeRef', 'minZ',         (eid, v) => { AOVolumeRef.minZ[eid] = v; });
  _registerField('AOVolumeRef', 'maxX',         (eid, v) => { AOVolumeRef.maxX[eid] = v; });
  _registerField('AOVolumeRef', 'maxY',         (eid, v) => { AOVolumeRef.maxY[eid] = v; });
  _registerField('AOVolumeRef', 'maxZ',         (eid, v) => { AOVolumeRef.maxZ[eid] = v; });
  _registerField('AOVolumeRef', 'intensity',    (eid, v) => { AOVolumeRef.intensity[eid] = v; });
  _registerField('AOVolumeRef', 'radius',       (eid, v) => { AOVolumeRef.radius[eid] = v; });
  _registerField('AOVolumeRef', 'bias',         (eid, v) => { AOVolumeRef.bias[eid] = v; });
  _registerField('AOVolumeRef', 'maxDistance',  (eid, v) => { AOVolumeRef.maxDistance[eid] = v; });
  _registerField('AOVolumeRef', 'indoorFactor', (eid, v) => { AOVolumeRef.indoorFactor[eid] = v; });

  /* ---------------- AOSampling ---------------- */
  _registerField('AOSampling', 'sampleCount', (eid, v) => { AOSampling.sampleCount[eid] = v; });
  _registerField('AOSampling', 'stepCount',   (eid, v) => { AOSampling.stepCount[eid] = v; });

  /* ---------------- AOBlur ---------------- */
  _registerField('AOBlur', 'mode',   (eid, v) => { AOBlur.mode[eid] = v; });
  _registerField('AOBlur', 'radius', (eid, v) => { AOBlur.radius[eid] = v; });
  _registerField('AOBlur', 'passes', (eid, v) => { AOBlur.passes[eid] = v; });

  /* ---------------- AOQuality ---------------- */
  _registerField('AOQuality', 'tier',             (eid, v) => { AOQuality.tier[eid] = v; });
  _registerField('AOQuality', 'resolutionScale',  (eid, v) => { AOQuality.resolutionScale[eid] = v; });
  _registerField('AOQuality', 'halfRes',          (eid, v) => { AOQuality.halfRes[eid] = v; });

  /* ---------------- AOState ---------------- */
  _registerField('AOState', 'state', (eid, v) => { AOState.state[eid] = v; });

  /* ---------------- AOContactShadow ---------------- */
  _registerField('AOContactShadow', 'enabled',     (eid, v) => { AOContactShadow.enabled[eid] = v; });
  _registerField('AOContactShadow', 'rayCount',    (eid, v) => { AOContactShadow.rayCount[eid] = v; });
  _registerField('AOContactShadow', 'maxDistance', (eid, v) => { AOContactShadow.maxDistance[eid] = v; });
  _registerField('AOContactShadow', 'strength',    (eid, v) => { AOContactShadow.strength[eid] = v; });

  /* ---------------- AOIndoorVolume ---------------- */
  _registerField('AOIndoorVolume', 'ceilingAO',   (eid, v) => { AOIndoorVolume.ceilingAO[eid] = v; });
  _registerField('AOIndoorVolume', 'floorAO',     (eid, v) => { AOIndoorVolume.floorAO[eid] = v; });
  _registerField('AOIndoorVolume', 'wallAO',      (eid, v) => { AOIndoorVolume.wallAO[eid] = v; });
  _registerField('AOIndoorVolume', 'cornerBoost', (eid, v) => { AOIndoorVolume.cornerBoost[eid] = v; });
  _registerField('AOIndoorVolume', 'enabled',     (eid, v) => { AOIndoorVolume.enabled[eid] = v; });

  /* ---------------- AOOutdoorVolume ---------------- */
  _registerField('AOOutdoorVolume', 'groundAO',    (eid, v) => { AOOutdoorVolume.groundAO[eid] = v; });
  _registerField('AOOutdoorVolume', 'skyAO',       (eid, v) => { AOOutdoorVolume.skyAO[eid] = v; });
  _registerField('AOOutdoorVolume', 'horizonAO',   (eid, v) => { AOOutdoorVolume.horizonAO[eid] = v; });
  _registerField('AOOutdoorVolume', 'enabled',     (eid, v) => { AOOutdoorVolume.enabled[eid] = v; });

  /* ---------------- AOTemporalAccumulator ---------------- */
  _registerField('AOTemporalAccumulator', 'mode',         (eid, v) => { AOTemporalAccumulator.mode[eid] = v; });
  _registerField('AOTemporalAccumulator', 'blendFactor',  (eid, v) => { AOTemporalAccumulator.blendFactor[eid] = v; });
  _registerField('AOTemporalAccumulator', 'enabled',      (eid, v) => { AOTemporalAccumulator.enabled[eid] = v; });

  /* ---------------- AODither ---------------- */
  _registerField('AODither', 'mode',     (eid, v) => { AODither.mode[eid] = v; });
  _registerField('AODither', 'strength', (eid, v) => { AODither.strength[eid] = v; });
  _registerField('AODither', 'enabled',  (eid, v) => { AODither.enabled[eid] = v; });

  /* ---------------- AOCelBands ---------------- */
  _registerField('AOCelBands', 'enabled',      (eid, v) => { AOCelBands.enabled[eid] = v; });
  _registerField('AOCelBands', 'bandCount',    (eid, v) => { AOCelBands.bandCount[eid] = v; });
  _registerField('AOCelBands', 'bandSoftness', (eid, v) => { AOCelBands.bandSoftness[eid] = v; });

  /* ---------------- AOInkOutline ---------------- */
  _registerField('AOInkOutline', 'enabled',   (eid, v) => { AOInkOutline.enabled[eid] = v; });
  _registerField('AOInkOutline', 'thickness', (eid, v) => { AOInkOutline.thickness[eid] = v; });
  _registerField('AOInkOutline', 'strength',  (eid, v) => { AOInkOutline.strength[eid] = v; });

  /* ---------------- AOEdgeFade ---------------- */
  _registerField('AOEdgeFade', 'enabled', (eid, v) => { AOEdgeFade.enabled[eid] = v; });
  _registerField('AOEdgeFade', 'style',   (eid, v) => { AOEdgeFade.style[eid] = v; });

  /* ---------------- AOBilateral ---------------- */
  _registerField('AOBilateral', 'enabled', (eid, v) => { AOBilateral.enabled[eid] = v; });
  _registerField('AOBilateral', 'passes',  (eid, v) => { AOBilateral.passes[eid] = v; });

  /* ---------------- AODenoiser ---------------- */
  _registerField('AODenoiser', 'enabled',        (eid, v) => { AODenoiser.enabled[eid] = v; });
  _registerField('AODenoiser', 'spatialPasses',  (eid, v) => { AODenoiser.spatialPasses[eid] = v; });
  _registerField('AODenoiser', 'temporalPasses', (eid, v) => { AODenoiser.temporalPasses[eid] = v; });
  _registerField('AODenoiser', 'blendStrength',  (eid, v) => { AODenoiser.blendStrength[eid] = v; });

  /* ---------------- AOResidency ---------------- */
  _registerField('AOResidency', 'pinned', (eid, v) => { AOResidency.pinned[eid] = v; });

  /* ---------------- AOLeak ---------------- */
  _registerField('AOLeak', 'enabled',   (eid, v) => { AOLeak.enabled[eid] = v; });
  _registerField('AOLeak', 'threshold', (eid, v) => { AOLeak.threshold[eid] = v; });

  /* ---------------- AOStyle ---------------- */
  _registerField('AOStyle', 'style',      (eid, v) => { AOStyle.style[eid] = v; });
  _registerField('AOStyle', 'enabled',    (eid, v) => { AOStyle.enabled[eid] = v; });
})();

/* ------------------------------------------------------------------ */
/* 5. FIELD WRITER LOOKUP                                             */
/* ------------------------------------------------------------------ */

function _writeField(eid, componentName, fieldName, value) {
  const key = componentName + '.' + fieldName;
  const writer = _fieldWriters.get(key);
  if (!writer) return false;
  try { writer(eid, value); }
  catch (_) { return false; }
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. PREFAB DECLARATIONS                                             */
/* ------------------------------------------------------------------ */

(function _declareAllPrefabs() {
  /* ---------------- SUN ---------------- */
  _register(PREFAB.SUN, {
    name: 'sun',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: [
      'Transform', 'Target',
      'LightRef', 'LightState', 'LightShadow', 'LightPriority',
      'LightDayCycle', 'LightEmissive',
      'ShadowBias', 'ShadowFilter', 'ShadowSoftness', 'ShadowFrustum',
      'ShadowCache', 'ShadowBudget', 'ShadowDirLight', 'ShadowTint',
      'ShadowEdge', 'ShadowAtlas', 'ShadowUpdatePolicy', 'ShadowState',
    ],
    tagMask: TAG.LIGHT | TAG.SUN | TAG.DIRECTIONAL | TAG.ACTIVE | TAG.SHADOW_CASTER,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      Transform: { y: 100, z: -30 },
      Target:    { y: 0, z: 0, active: 1 },
      LightRef:  {
        type: LIGHT_TYPE.DIRECTIONAL,
        kind: LIGHT_KIND.SUN,
        colorR: 1.0, colorG: 0.96, colorB: 0.85,
        intensity: 1.25,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE | LIGHT_STATE_FLAG.CAST_SHADOW },
      LightShadow: { enabled: 1, mapSize: 2048, cascadeCount: 4, filter: SHADOW_FILTER.PCF_SOFT },
      LightPriority: { bucket: LIGHT_PRIORITY.CRITICAL, sortKey: 3.0 },
      LightDayCycle: { dayCycle: 0.38, daySpeed: 0.004, enabled: 1, maxElevation: 75 },
      ShadowBias: { bias: -0.0006, normalBias: 0.018 },
      ShadowFilter: { mode: SHADOW_FILTER.PCF_SOFT, kernelSize: 5 },
      ShadowSoftness: { softness: 0.08, edgeStyle: SHADOW_EDGE.POSTERIZED, bandCount: 4 },
      ShadowTint: { tintR: 0.14, tintG: 0.18, tintB: 0.28, tintStrength: 0.65, enabled: 1 },
      ShadowDirLight: { cascades: 4, stabilize: 1, texelSnap: 1 },
      ShadowAtlas: { mode: SHADOW_ATLAS_MODE.CASCADED, atlasWidth: 2048, atlasHeight: 2048 },
      ShadowUpdatePolicy: { mode: SHADOW_UPDATE_MODE.EVERY_OTHER, interval: 2 },
      ShadowState: { state: SHADOW_STATE.DIRTY },
    },
  });

  /* ---------------- MOON ---------------- */
  _register(PREFAB.MOON, {
    name: 'moon',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: [
      'Transform', 'Target',
      'LightRef', 'LightState', 'LightPriority', 'LightDayCycle',
      'ShadowBias', 'ShadowFilter', 'ShadowSoftness', 'ShadowCache',
      'ShadowState',
    ],
    tagMask: TAG.LIGHT | TAG.MOON | TAG.DIRECTIONAL | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      Transform: { y: 80, z: 30 },
      Target:    { y: 0, z: 0, active: 1 },
      LightRef:  {
        type: LIGHT_TYPE.DIRECTIONAL,
        kind: LIGHT_KIND.MOON,
        colorR: 0.42, colorG: 0.48, colorB: 0.70,
        intensity: 0.35,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 1.8 },
      LightDayCycle: { dayCycle: 0.0, daySpeed: 0.0008, enabled: 1, maxElevation: 65 },
    },
  });

  /* ---------------- AMBIENT ---------------- */
  _register(PREFAB.AMBIENT, {
    name: 'ambient',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.AMBIENT | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      LightRef: { type: LIGHT_TYPE.AMBIENT, kind: LIGHT_KIND.GENERIC,
                  colorR: 0.20, colorG: 0.25, colorB: 0.30, intensity: 0.15 },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.0 },
    },
  });

  /* ---------------- HEMISPHERE ---------------- */
  _register(PREFAB.HEMISPHERE, {
    name: 'hemisphere',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.HEMISPHERE | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      Transform: { y: 50 },
      LightRef: {
        type: LIGHT_TYPE.HEMISPHERE, kind: LIGHT_KIND.GENERIC,
        colorR: 0.45, colorG: 0.62, colorB: 0.85,
        groundColorR: 0.20, groundColorG: 0.20, groundColorB: 0.20,
        intensity: 0.35,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 1.5 },
    },
  });

  /* ---------------- SKY LIGHT ---------------- */
  _register(PREFAB.SKY_LIGHT, {
    name: 'sky_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.HEMISPHERE | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      Transform: { y: 200 },
      LightRef: {
        type: LIGHT_TYPE.HEMISPHERE, kind: LIGHT_KIND.GENERIC,
        colorR: 0.55, colorG: 0.75, colorB: 1.00,
        groundColorR: 0.30, groundColorG: 0.28, groundColorB: 0.25,
        intensity: 0.50,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 1.5 },
    },
  });

  /* ---------------- FIRE LIGHT ---------------- */
  _register(PREFAB.FIRE_LIGHT, {
    name: 'fire_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: [
      'Transform', 'LightRef', 'LightState', 'LightPriority',
      'LightFlicker', 'LightEmissive', 'LightShadow', 'LightBehavior',
      'ShadowBias', 'ShadowFilter', 'ShadowSoftness', 'ShadowPointLight',
      'ShadowState',
    ],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.EMISSIVE | TAG.ACTIVE | TAG.SHADOW_CASTER,
    defaults: {
      Transform: { y: 0.8 },
      LightRef: {
        type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.FIRE,
        colorR: 1.0, colorG: 0.55, colorB: 0.15,
        intensity: 2.5, range: 12, decay: 2.0,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE | LIGHT_STATE_FLAG.CAST_SHADOW },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 2.0 },
      LightFlicker: { baseIntensity: 2.5, amplitude: 0.18, hz: 9.0, phase: 0.0, enabled: 1 },
      LightEmissive: { emissiveR: 1.0, emissiveG: 0.55, emissiveB: 0.15, emissiveScale: 1.0, proxyVisible: 1 },
      LightShadow: { enabled: 1, mapSize: 512, filter: SHADOW_FILTER.PCF },
      ShadowBias: { bias: -0.0020, normalBias: 0.040 },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.FLICKER }],
  });

  /* ---------------- NEON LIGHT ---------------- */
  _register(PREFAB.NEON_LIGHT, {
    name: 'neon_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority',
                 'LightEmissive', 'LightFlicker', 'LightBehavior'],
    tagMask: TAG.LIGHT | TAG.RECT_AREA | TAG.EMISSIVE | TAG.ACTIVE,
    defaults: {
      Transform: { y: 2.0 },
      LightRef: {
        type: LIGHT_TYPE.RECT_AREA, kind: LIGHT_KIND.NEON,
        colorR: 0.9, colorG: 0.4, colorB: 0.7,
        intensity: 3.0, width: 1.2, height: 0.2,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.6 },
      LightEmissive: { emissiveR: 0.9, emissiveG: 0.4, emissiveB: 0.7, emissiveScale: 1.0, proxyVisible: 1 },
      LightFlicker: { baseIntensity: 3.0, amplitude: 0.25, hz: 1.5, phase: 0.0, enabled: 1 },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.PULSE }],
  });

  /* ---------------- MAGIC GLOW ---------------- */
  _register(PREFAB.MAGIC_GLOW, {
    name: 'magic_glow',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority',
                 'LightFlicker', 'LightEmissive', 'LightBehavior'],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.EMISSIVE | TAG.ACTIVE,
    defaults: {
      Transform: { y: 1.2 },
      LightRef: {
        type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.MAGIC,
        colorR: 1.0, colorG: 0.85, colorB: 0.45,
        intensity: 3.5, range: 8, decay: 2.0,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.CRITICAL, sortKey: 3.0 },
      LightFlicker: { baseIntensity: 3.5, amplitude: 0.25, hz: 5.0, phase: 0.0, enabled: 1 },
      LightEmissive: { emissiveR: 1.0, emissiveG: 0.85, emissiveB: 0.45, emissiveScale: 1.0, proxyVisible: 1 },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.FLICKER }],
  });

  /* ---------------- INTERIOR LAMP ---------------- */
  _register(PREFAB.INTERIOR_LAMP, {
    name: 'interior_lamp',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority',
                 'LightIndoor', 'LightShadow', 'LightBehavior',
                 'ShadowBias', 'ShadowFilter', 'ShadowPointLight', 'ShadowState'],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.ACTIVE | TAG.SHADOW_CASTER,
    tag2Mask: TAG2.INDOOR_LIGHT | TAG2.PERSISTENT,
    defaults: {
      Transform: { y: 2.4 },
      LightRef: {
        type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.INTERIOR_LAMP,
        colorR: 1.0, colorG: 0.85, colorB: 0.65,
        intensity: 1.8, range: 6, decay: 2.0,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE | LIGHT_STATE_FLAG.CAST_SHADOW,
                    env: LIGHT_ENVIRONMENT.INDOOR, envBlend: 1.0 },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 1.8 },
      LightIndoor: { indoorWeight: 1.0, outdoorWeight: 0.0 },
      LightShadow: { enabled: 1, mapSize: 512, filter: SHADOW_FILTER.PCF },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.TEMPERATURE_DRIFT }],
  });

  /* ---------------- WINDOW SHAFT ---------------- */
  _register(PREFAB.WINDOW_SHAFT, {
    name: 'window_shaft',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'Target', 'LightRef', 'LightState',
                 'LightShadow', 'ShadowBias', 'ShadowFilter', 'ShadowDirLight',
                 'ShadowState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.DIRECTIONAL | TAG.ACTIVE | TAG.SHADOW_CASTER,
    tag2Mask: TAG2.INDOOR_LIGHT,
    defaults: {
      Transform: { y: 3.0 },
      Target:    { y: 0, active: 1 },
      LightRef: {
        type: LIGHT_TYPE.DIRECTIONAL, kind: LIGHT_KIND.WINDOW_SHAFT,
        colorR: 1.0, colorG: 0.94, colorB: 0.82,
        intensity: 1.2,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE | LIGHT_STATE_FLAG.CAST_SHADOW,
                    env: LIGHT_ENVIRONMENT.INDOOR, envBlend: 1.0 },
      LightShadow: { enabled: 1, mapSize: 1024, filter: SHADOW_FILTER.PCF_SOFT },
      ShadowBias: { bias: -0.0012, normalBias: 0.024 },
      ShadowDirLight: { cascades: 1, stabilize: 1, texelSnap: 1 },
      LightPriority: { bucket: LIGHT_PRIORITY.HIGH, sortKey: 1.9 },
    },
  });

  /* ---------------- CAUSTIC ---------------- */
  _register(PREFAB.CAUSTIC, {
    name: 'caustic',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority',
                 'LightFlicker', 'LightBehavior'],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.ACTIVE,
    defaults: {
      Transform: { y: 0.1 },
      LightRef: {
        type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.CAUSTIC,
        colorR: 0.6, colorG: 0.95, colorB: 1.0,
        intensity: 0.8, range: 4, decay: 2.5,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.4 },
      LightFlicker: { baseIntensity: 0.8, amplitude: 0.20, hz: 2.0, phase: 0.0, enabled: 1 },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.PULSE }],
  });

  /* ---------------- AURORA ---------------- */
  _register(PREFAB.AURORA, {
    name: 'aurora',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.HEMISPHERE | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT | TAG2.SNOW,
    defaults: {
      Transform: { y: 100 },
      LightRef: {
        type: LIGHT_TYPE.HEMISPHERE, kind: LIGHT_KIND.AURORA,
        colorR: 0.35, colorG: 0.95, colorB: 0.75,
        groundColorR: 0.05, groundColorG: 0.10, groundColorB: 0.20,
        intensity: 0.45,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.2 },
    },
  });

  /* ---------------- BIO LUMINESCENT ---------------- */
  _register(PREFAB.BIO_LUMINESCENT, {
    name: 'bio_luminescent',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority',
                 'LightEmissive', 'LightFlicker', 'LightBehavior'],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.EMISSIVE | TAG.ACTIVE,
    tag2Mask: TAG2.FOREST,
    defaults: {
      Transform: { y: 1.5 },
      LightRef: {
        type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.BIO_LUMINESCENT,
        colorR: 0.4, colorG: 1.0, colorB: 0.8,
        intensity: 1.2, range: 5, decay: 2.0,
      },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.3 },
      LightFlicker: { baseIntensity: 1.2, amplitude: 0.35, hz: 0.8, phase: 0.0, enabled: 1 },
      LightEmissive: { emissiveR: 0.4, emissiveG: 1.0, emissiveB: 0.8, emissiveScale: 1.0, proxyVisible: 1 },
    },
    behaviors: [{ id: LIGHT_BEHAVIOR.PULSE }],
  });

  /* ---------------- POINT LIGHT ---------------- */
  _register(PREFAB.POINT_LIGHT, {
    name: 'point_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.POINT | TAG.ACTIVE,
    defaults: {
      Transform: { y: 1.0 },
      LightRef: { type: LIGHT_TYPE.POINT, kind: LIGHT_KIND.GENERIC,
                  colorR: 1.0, colorG: 1.0, colorB: 1.0,
                  intensity: 1.0, range: 8, decay: 2.0 },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.0 },
    },
  });

  /* ---------------- SPOT LIGHT ---------------- */
  _register(PREFAB.SPOT_LIGHT, {
    name: 'spot_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'Target', 'LightRef', 'LightState',
                 'LightPriority', 'LightShadow', 'ShadowSpotLight',
                 'ShadowBias', 'ShadowFilter', 'ShadowState'],
    tagMask: TAG.LIGHT | TAG.SPOT | TAG.ACTIVE | TAG.SHADOW_CASTER,
    defaults: {
      Transform: { y: 3.0 },
      Target:    { y: 0, active: 1 },
      LightRef: { type: LIGHT_TYPE.SPOT, kind: LIGHT_KIND.GENERIC,
                  colorR: 1.0, colorG: 1.0, colorB: 1.0,
                  intensity: 1.5, range: 10, decay: 2.0,
                  angle: Math.PI / 4, penumbra: 0.15 },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE | LIGHT_STATE_FLAG.CAST_SHADOW },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.1 },
      LightShadow: { enabled: 1, mapSize: 1024, filter: SHADOW_FILTER.PCF_SOFT },
    },
  });

  /* ---------------- RECT AREA LIGHT ---------------- */
  _register(PREFAB.RECT_AREA_LIGHT, {
    name: 'rect_area_light',
    pool: POOL.LIGHT,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'LightRef', 'LightState', 'LightPriority'],
    tagMask: TAG.LIGHT | TAG.RECT_AREA | TAG.ACTIVE,
    defaults: {
      Transform: { y: 2.0 },
      LightRef: { type: LIGHT_TYPE.RECT_AREA, kind: LIGHT_KIND.GENERIC,
                  colorR: 1.0, colorG: 1.0, colorB: 1.0,
                  intensity: 2.0, width: 1.0, height: 1.0 },
      LightState: { flags: LIGHT_STATE_FLAG.ACTIVE | LIGHT_STATE_FLAG.VISIBLE },
      LightPriority: { bucket: LIGHT_PRIORITY.NORMAL, sortKey: 1.1 },
    },
  });

  /* ---------------- SHADOW CASTER ---------------- */
  _register(PREFAB.SHADOW_CASTER, {
    name: 'shadow_caster',
    pool: POOL.SHADOW,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'ShadowCasterRef'],
    tagMask: TAG.SHADOW_CASTER,
    defaults: {
      Transform: { sx: 1, sy: 1, sz: 1, qw: 1 },
      ShadowCasterRef: { casterKind: 0, castStrength: 1.0, casterRadius: 1.0, enabled: 1 },
    },
  });

  /* ---------------- SHADOW RECEIVER ---------------- */
  _register(PREFAB.SHADOW_RECEIVER, {
    name: 'shadow_receiver',
    pool: POOL.SHADOW,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform', 'ShadowReceiverRef'],
    tagMask: TAG.SHADOW_RECEIVER,
    defaults: {
      Transform: { sx: 1, sy: 1, sz: 1, qw: 1 },
      ShadowReceiverRef: { receiverKind: 0, sampleQuality: 2, selfShadowBias: 0.02, enabled: 1 },
    },
  });

  /* ---------------- SHADOW ATLAS ---------------- */
  _register(PREFAB.SHADOW_ATLAS, {
    name: 'shadow_atlas',
    pool: POOL.SHADOW,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['ShadowAtlas', 'ShadowBias', 'ShadowFilter', 'ShadowSoftness',
                 'ShadowCache', 'ShadowBudget', 'ShadowTint', 'ShadowEdge',
                 'ShadowUpdatePolicy', 'ShadowState'],
    tagMask: TAG.SHADOW_ATLAS_OWNER,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      ShadowAtlas: { mode: SHADOW_ATLAS_MODE.ATLAS_PACKED, atlasWidth: 2048, atlasHeight: 2048 },
      ShadowBias:  { bias: -0.0008, normalBias: 0.020 },
      ShadowFilter:{ mode: SHADOW_FILTER.PCF, kernelSize: 3 },
      ShadowSoftness: { softness: 0.05, edgeStyle: SHADOW_EDGE.POSTERIZED, bandCount: 4 },
      ShadowUpdatePolicy: { mode: SHADOW_UPDATE_MODE.EVERY_OTHER, interval: 2 },
      ShadowState: { state: SHADOW_STATE.DIRTY },
    },
  });

  /* ---------------- SHADOW CASCADE SET ---------------- */
  _register(PREFAB.SHADOW_CASCADE_SET, {
    name: 'shadow_cascade_set',
    pool: POOL.SHADOW,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['ShadowCascade', 'ShadowDirLight'],
    tagMask: TAG.CASCADE_OWNER,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      ShadowDirLight: { cascades: 4, stabilize: 1, texelSnap: 1, maxDistance: 140 },
    },
  });

  /* ---------------- CONTACT SHADOW ---------------- */
  _register(PREFAB.CONTACT_SHADOW, {
    name: 'contact_shadow',
    pool: POOL.SHADOW,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['ShadowContact', 'AOContactShadow'],
    tagMask: TAG.SHADOW_CASTER,
    tag2Mask: TAG2.CONTACT_SHADOW,
    defaults: {
      ShadowContact:   { enabled: 1, radius: 0.5, thickness: 0.1, strength: 0.6 },
      AOContactShadow: { enabled: 1, rayCount: 8, maxDistance: 2.0, strength: 0.6 },
    },
  });

  /* ---------------- GI PROBE ---------------- */
  _register(PREFAB.GI_PROBE, {
    name: 'gi_probe',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['GIProbeRef', 'GIIrradiance', 'GISH', 'GIOcclusion',
                 'GIBudget', 'GIState', 'GILeak', 'GITemporal',
                 'GIVolumeBlend'],
    tagMask: TAG.GI_PROBE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      GIProbeRef: { type: GI_PROBE_TYPE.OUTDOOR, quality: GI_QUALITY.MEDIUM,
                    mode: GI_MODE.SH_L2, radius: 2.0, enabled: 1, indoorFactor: 0.0,
                    worldY: 3.5 },
      GIIrradiance: { r: 0.15, g: 0.18, b: 0.22, confidence: 0.0 },
      GIOcclusion: { skyOcclusion: 1.0, obstacleOcclusion: 0.0 },
      GIBudget: { cost: 1.0, priority: 100, lod: 0 },
      GIState: { state: GI_STATE.IDLE },
      GILeak: { mode: GI_LEAK_MODE.SMOOTH, threshold: 0.35, correctionFactor: 1.0 },
      GITemporal: { historyWeight: 0.90, reprojectionBias: 0.02, rejectionThreshold: 0.15, valid: 0 },
    },
  });

  /* ---------------- GI HERO PROBE ---------------- */
  _register(PREFAB.GI_HERO_PROBE, {
    name: 'gi_hero_probe',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['GIProbeRef', 'GIIrradiance', 'GISH', 'GIOcclusion',
                 'GIBudget', 'GIState', 'GILeak', 'GITemporal',
                 'GICelBands', 'GIPalette', 'GIVolumeBlend'],
    tagMask: TAG.GI_PROBE | TAG.GI_HERO_PROBE,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      GIProbeRef: { type: GI_PROBE_TYPE.HERO, quality: GI_QUALITY.HIGH,
                    mode: GI_MODE.SH_L3, radius: 3.0, enabled: 1 },
      GICelBands: { enabled: 1, bandCount: 4, bandSoftness: 0.10, ditherStrength: 1.0 / 255.0 },
      GIPalette: { styleId: 0, satBias: 1.0, hueBias: 0.0,
                   ambientColorR: 0.45, ambientColorG: 0.55, ambientColorB: 0.70,
                   lerpRate: 3.2, enabled: 1 },
    },
  });

  /* ---------------- GI VOLUME INDOOR ---------------- */
  _register(PREFAB.GI_VOLUME_INDOOR, {
    name: 'gi_volume_indoor',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIVolume', 'GIIndoor'],
    tagMask: TAG.GI_VOLUME | TAG.AO_INDOOR,
    tag2Mask: TAG2.INDOOR_LIGHT,
    defaults: {
      GIVolume: { kind: 0, enabled: 1, blendDistance: 1.5, probeDensity: 1.5, leakGate: 0.90,
                  fillR: 0.20, fillG: 0.22, fillB: 0.28 },
      GIIndoor: { wallOcclusion: 0.90, curtainTransmission: 0.20, enabled: 1 },
    },
  });

  /* ---------------- GI VOLUME OUTDOOR ---------------- */
  _register(PREFAB.GI_VOLUME_OUTDOOR, {
    name: 'gi_volume_outdoor',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIVolume', 'GIOutdoor'],
    tagMask: TAG.GI_VOLUME | TAG.AO_OUTDOOR,
    tag2Mask: TAG2.OUTDOOR_LIGHT,
    defaults: {
      GIVolume: { kind: 1, enabled: 1, blendDistance: 3.0, probeDensity: 1.0, leakGate: 0.10,
                  fillR: 0.25, fillG: 0.30, fillB: 0.40 },
      GIOutdoor: { biomeWeight: 1.0, enabled: 1 },
    },
  });

  /* ---------------- GI PORTAL DOORWAY ---------------- */
  _register(PREFAB.GI_PORTAL_DOORWAY, {
    name: 'gi_portal_doorway',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIPortal'],
    tagMask: TAG.GI_PORTAL,
    defaults: {
      GIPortal: { type: GI_PORTAL_TYPE.DOORWAY, enabled: 1,
                  positionY: 1.0, normalZ: 1.0,
                  width: 1.0, height: 2.0, transmission: 1.0 },
    },
  });

  /* ---------------- GI PORTAL WINDOW ---------------- */
  _register(PREFAB.GI_PORTAL_WINDOW, {
    name: 'gi_portal_window',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIPortal'],
    tagMask: TAG.GI_PORTAL,
    defaults: {
      GIPortal: { type: GI_PORTAL_TYPE.WINDOW, enabled: 1,
                  positionY: 1.6, normalZ: 1.0,
                  width: 1.2, height: 1.4, transmission: 0.85 },
    },
  });

  /* ---------------- GI PORTAL SKYLIGHT ---------------- */
  _register(PREFAB.GI_PORTAL_SKYLIGHT, {
    name: 'gi_portal_skylight',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIPortal'],
    tagMask: TAG.GI_PORTAL,
    defaults: {
      GIPortal: { type: GI_PORTAL_TYPE.SKYLIGHT, enabled: 1,
                  positionY: 4.0, normalY: -1.0,
                  width: 1.5, height: 1.5, transmission: 0.95 },
    },
  });

  /* ---------------- GI PORTAL CAVE ---------------- */
  _register(PREFAB.GI_PORTAL_CAVE, {
    name: 'gi_portal_cave',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIPortal'],
    tagMask: TAG.GI_PORTAL,
    defaults: {
      GIPortal: { type: GI_PORTAL_TYPE.CAVE_MOUTH, enabled: 1,
                  positionY: 1.5, normalZ: 1.0,
                  width: 3.0, height: 3.0, transmission: 0.70 },
    },
  });

  /* ---------------- REFLECTION PROBE ---------------- */
  _register(PREFAB.REFLECTION_PROBE, {
    name: 'reflection_probe',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GIReflectionProbe'],
    tagMask: TAG.GI_REFLECTION_PROBE,
    defaults: {
      GIReflectionProbe: { positionY: 3.0, radius: 20.0, resolution: 128,
                           updateInterval: 120, enabled: 1 },
    },
  });

  /* ---------------- LIGHTFIELD ---------------- */
  _register(PREFAB.LIGHTFIELD, {
    name: 'lightfield',
    pool: POOL.GI,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['GILightfield'],
    tagMask: TAG.GI_PROBE,
    tag2Mask: TAG2.GI_LIGHTFIELD,
    defaults: {
      GILightfield: { sampleCount: 256 },
    },
  });

  /* ---------------- AO VOLUME ---------------- */
  _register(PREFAB.AO_VOLUME, {
    name: 'ao_volume',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['AOVolumeRef', 'AOSampling', 'AOBlur', 'AOQuality',
                 'AOState', 'AOBudget', 'AOTemporalAccumulator', 'AODither',
                 'AOResidency', 'AOLeak', 'AOStyle'],
    tagMask: TAG.AO_VOLUME,
    defaults: {
      AOVolumeRef: { method: AO_METHOD.HBAO, quality: AO_QUALITY.MEDIUM,
                     style: AO_STYLE.CEL_SOFT, enabled: 1,
                     intensity: 1.0, radius: 2.0, bias: 0.025, maxDistance: 12.0 },
      AOSampling: { sampleCount: 8, stepCount: 8 },
      AOBlur: { mode: AO_BLUR_MODE.ANIME_SOFT, radius: 4.0, passes: 1 },
      AOQuality: { tier: AO_QUALITY.MEDIUM, resolutionScale: 0.5, halfRes: 1 },
      AOState: { state: AO_STATE.IDLE },
      AOTemporalAccumulator: { mode: AO_TEMPORAL_MODE.EXPONENTIAL, blendFactor: 0.10, enabled: 1 },
      AODither: { mode: AO_DITHER.BAYER4, strength: 1.0 / 255.0, enabled: 1 },
      AOLeak: { enabled: 1, threshold: 0.35 },
      AOStyle: { style: AO_STYLE.CEL_SOFT, enabled: 1 },
    },
  });

  /* ---------------- AO INDOOR VOLUME ---------------- */
  _register(PREFAB.AO_INDOOR_VOLUME, {
    name: 'ao_indoor_volume',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['AOVolumeRef', 'AOSampling', 'AOBlur', 'AOQuality',
                 'AOState', 'AOBudget', 'AOTemporalAccumulator', 'AODither',
                 'AOResidency', 'AOLeak', 'AOStyle',
                 'AOIndoorVolume', 'AOCelBands', 'AOInkOutline'],
    tagMask: TAG.AO_VOLUME | TAG.AO_INDOOR | TAG.AO_CEL_BAND,
    tag2Mask: TAG2.INDOOR_LIGHT | TAG2.AO_INK_OUTLINE,
    defaults: {
      AOVolumeRef: { method: AO_METHOD.HBAO, quality: AO_QUALITY.MEDIUM,
                     style: AO_STYLE.CEL_SOFT, enabled: 1, indoorFactor: 1.0,
                     intensity: 1.0, radius: 1.8, bias: 0.022, maxDistance: 8.0 },
      AOIndoorVolume: { ceilingAO: 0.35, floorAO: 0.55, wallAO: 0.75,
                        cornerBoost: 0.35, enabled: 1 },
      AOCelBands: { enabled: 1, bandCount: 4, bandSoftness: 0.10 },
      AOInkOutline: { enabled: 1, thickness: 0.003, strength: 0.35 },
      AOStyle: { style: AO_STYLE.CEL_SOFT, enabled: 1 },
    },
  });

  /* ---------------- AO OUTDOOR VOLUME ---------------- */
  _register(PREFAB.AO_OUTDOOR_VOLUME, {
    name: 'ao_outdoor_volume',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['AOVolumeRef', 'AOSampling', 'AOBlur', 'AOQuality',
                 'AOState', 'AOBudget', 'AOTemporalAccumulator', 'AODither',
                 'AOResidency', 'AOLeak', 'AOStyle', 'AOOutdoorVolume'],
    tagMask: TAG.AO_VOLUME | TAG.AO_OUTDOOR,
    tag2Mask: TAG2.OUTDOOR_LIGHT,
    defaults: {
      AOVolumeRef: { method: AO_METHOD.HBAO, quality: AO_QUALITY.MEDIUM,
                     style: AO_STYLE.CEL_SOFT, enabled: 1, indoorFactor: 0.0,
                     intensity: 1.0, radius: 2.5, bias: 0.030, maxDistance: 16.0 },
      AOOutdoorVolume: { groundAO: 0.65, skyAO: 0.20, horizonAO: 0.35, enabled: 1 },
    },
  });

  /* ---------------- AO CONTACT VOLUME ---------------- */
  _register(PREFAB.AO_CONTACT_VOLUME, {
    name: 'ao_contact_volume',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['AOVolumeRef', 'AOSampling', 'AOQuality', 'AOState',
                 'AOBudget', 'AOContactShadow', 'AOResidency'],
    tagMask: TAG.AO_VOLUME,
    tag2Mask: TAG2.AO_CONTACT,
    defaults: {
      AOVolumeRef: { method: AO_METHOD.CONTACT, quality: AO_QUALITY.HIGH,
                     enabled: 1, intensity: 0.8, radius: 1.0, bias: 0.015, maxDistance: 3.0 },
      AOContactShadow: { enabled: 1, rayCount: 8, maxDistance: 2.0,
                         thickness: 0.10, strength: 0.7 },
    },
  });

  /* ---------------- AO CEL CONTROLLER ---------------- */
  _register(PREFAB.AO_CEL_CONTROLLER, {
    name: 'ao_cel_controller',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['AOCelBands', 'AOStyle'],
    tagMask: TAG.AO_CEL_BAND,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      AOCelBands: { enabled: 1, bandCount: 4, bandSoftness: 0.10 },
      AOStyle: { style: AO_STYLE.CEL_SOFT, enabled: 1 },
    },
  });

  /* ---------------- AO INK CONTROLLER ---------------- */
  _register(PREFAB.AO_INK_CONTROLLER, {
    name: 'ao_ink_controller',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['AOInkOutline', 'AOStyle', 'AOEdgeFade'],
    tagMask: TAG.AO_CEL_BAND,
    tag2Mask: TAG2.PERSISTENT | TAG2.AO_INK_OUTLINE,
    defaults: {
      AOInkOutline: { enabled: 1, thickness: 0.003, strength: 0.35, softness: 0.15 },
      AOStyle: { style: AO_STYLE.INK_LINE, enabled: 1 },
      AOEdgeFade: { enabled: 1, style: AO_EDGE.RADIAL },
    },
  });

  /* ---------------- AO DENOISER ---------------- */
  _register(PREFAB.AO_DENOISER, {
    name: 'ao_denoiser',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['AODenoiser', 'AOTemporalAccumulator'],
    tagMask: 0,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      AODenoiser: { enabled: 1, spatialPasses: 2, temporalPasses: 1, blendStrength: 0.15 },
      AOTemporalAccumulator: { mode: AO_TEMPORAL_MODE.EXPONENTIAL, blendFactor: 0.10, enabled: 1 },
    },
  });

  /* ---------------- AO BILATERAL ---------------- */
  _register(PREFAB.AO_BILATERAL, {
    name: 'ao_bilateral',
    pool: POOL.AO,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['AOBilateral', 'AOBlur'],
    tagMask: 0,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      AOBilateral: { enabled: 1, passes: 1 },
      AOBlur: { mode: AO_BLUR_MODE.BILATERAL, radius: 4.0, passes: 1 },
    },
  });

  /* ---------------- CAMERA ---------------- */
  _register(PREFAB.CAMERA, {
    name: 'camera',
    pool: POOL.CAMERA,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'CameraTag'],
    tagMask: TAG.CAMERA | TAG.ACTIVE | TAG.ACTIVE_CAMERA,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {
      Transform: { x: 0, y: 8, z: 34, qw: 1 },
      CameraTag: { active: 1, fov: 55, near: 0.1, far: 1000, aspect: 0.56 },
    },
  });

  /* ---------------- ORBIT CAMERA ---------------- */
  _register(PREFAB.ORBIT_CAMERA, {
    name: 'orbit_camera',
    pool: POOL.CAMERA,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'CameraTag'],
    tagMask: TAG.CAMERA | TAG.ACTIVE | TAG.ACTIVE_CAMERA,
    tag2Mask: TAG2.PERSISTENT | TAG2.ORBIT_CAMERA,
    defaults: {
      Transform: { x: 0, y: 8, z: 34, qw: 1 },
      CameraTag: { active: 1, fov: 50, near: 0.1, far: 2000, aspect: 0.56 },
    },
  });

  /* ---------------- DOF CAMERA ---------------- */
  _register(PREFAB.DOF_CAMERA, {
    name: 'dof_camera',
    pool: POOL.CAMERA,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: ['Transform', 'CameraTag'],
    tagMask: TAG.CAMERA | TAG.ACTIVE,
    tag2Mask: TAG2.PERSISTENT | TAG2.DOF_CAMERA,
    defaults: {
      Transform: { x: 0, y: 6, z: 24, qw: 1 },
      CameraTag: { active: 0, fov: 40, near: 0.1, far: 1500, aspect: 0.56 },
    },
  });

  /* ---------------- STREAMING CHUNK ---------------- */
  _register(PREFAB.STREAMING_CHUNK, {
    name: 'streaming_chunk',
    pool: POOL.STREAMING,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform'],
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.STREAMING_CHUNK,
    defaults: {
      Transform: { sx: 1, sy: 1, sz: 1, qw: 1 },
    },
  });

  /* ---------------- LOD GROUP ---------------- */
  _register(PREFAB.LOD_GROUP, {
    name: 'lod_group',
    pool: POOL.STREAMING,
    lifetime: PREFAB_LIFETIME.SCENE,
    components: ['Transform'],
    tagMask: TAG.ACTIVE,
    tag2Mask: TAG2.LOD_GROUP,
    defaults: {
      Transform: { qw: 1, sx: 1, sy: 1, sz: 1 },
    },
  });

  /* ---------------- DEBUG MARKER ---------------- */
  _register(PREFAB.DEBUG_MARKER, {
    name: 'debug_marker',
    pool: POOL.DEBUG,
    lifetime: PREFAB_LIFETIME.TRANSIENT,
    ttlFrames: 300,
    components: ['Transform'],
    tagMask: TAG.DEBUG,
    tag2Mask: TAG2.TRANSIENT,
    defaults: {
      Transform: { sx: 0.1, sy: 0.1, sz: 0.1, qw: 1 },
    },
  });

  /* ---------------- DEBUG HUD ---------------- */
  _register(PREFAB.DEBUG_HUD, {
    name: 'debug_hud',
    pool: POOL.DEBUG,
    lifetime: PREFAB_LIFETIME.PERMANENT,
    components: [],
    tagMask: TAG.DEBUG,
    tag2Mask: TAG2.PERSISTENT,
    defaults: {},
  });
})();

/* ------------------------------------------------------------------ */
/* 7. COMPONENT RESOLUTION                                            */
/* ------------------------------------------------------------------ */

/**
 * Per-prefab cache of resolved component objects. Populated on first
 * spawn so subsequent spawns skip the registry lookup.
 */
const _prefabComponentCache = new Array(MAX_PREFABS).fill(null);

function _resolvePrefabComponents(prefab) {
  const cached = _prefabComponentCache[prefab.id];
  if (cached && cached.length === prefab.componentNames.length) return cached;

  const world = getECSWorld();
  const reg = getDefaultComponentRegistry();
  const resolved = new Array(prefab.componentNames.length);

  for (let i = 0; i < prefab.componentNames.length; i++) {
    const name = prefab.componentNames[i];
    // Prefer the world-level component map.
    let comp = null;
    if (world) comp = (typeof getWorldComponent === 'function')
      ? getWorldComponent(name)
      : null;
    if (!comp && reg) {
      const entry = reg.get(name);
      if (entry && entry.ref) comp = entry.ref;
    }
    resolved[i] = comp;
  }

  _prefabComponentCache[prefab.id] = resolved;
  return resolved;
}

/* ------------------------------------------------------------------ */
/* 8. SPAWN                                                           */
/* ------------------------------------------------------------------ */

/**
 * Acquires an entity from the prefab's pool. Applies the prefab's
 * defaults, tag bits, lifetime record, and behaviors. Returns the
 * entity id, or -1 on failure.
 */
export function spawnPrefab(world, prefabId, overrides) {
  const prefab = _prefabs[prefabId];
  if (!prefab) {
    PrefabState.totalRejected++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[022_scn_SpawnPrefabs] unknown prefab id ${prefabId}`);
    return -1;
  }

  // Enforce maxLive if declared.
  if (prefab.maxLive > 0 && PrefabState.liveByPrefab[prefabId] >= prefab.maxLive) {
    PrefabState.totalRejected++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[022_scn_SpawnPrefabs] prefab "${prefab.name}" maxLive reached (${prefab.maxLive})`);
    return -1;
  }

  // Acquire an entity from the correct subsystem pool.
  let eid;
  try {
    eid = acquire(prefab.pool);
  } catch (e) {
    PrefabState.totalRejected++;
    return -1;
  }
  if (eid < 0) {
    PrefabState.totalRejected++;
    return -1;
  }

  // Attach components.
  const resolved = _resolvePrefabComponents(prefab);
  for (let i = 0; i < resolved.length; i++) {
    const comp = resolved[i];
    if (!comp) continue;
    attachComponent(eid, comp);
  }

  // Apply default field values.
  const defaults = prefab.defaults;
  for (const compName in defaults) {
    const fields = defaults[compName];
    for (const fieldName in fields) {
      _writeField(eid, compName, fieldName, fields[fieldName]);
    }
  }

  // Apply overrides. Overrides is a plain object with the same
  // component-name → field-name shape as `defaults`.
  if (overrides && typeof overrides === 'object') {
    for (const compName in overrides) {
      const fields = overrides[compName];
      if (!fields || typeof fields !== 'object') continue;
      for (const fieldName in fields) {
        _writeField(eid, compName, fieldName, fields[fieldName]);
      }
    }
  }

  // Apply tag bits.
  if (prefab.tagMask !== 0)  tagEntity(eid, prefab.tagMask);
  if (prefab.tag2Mask !== 0) tagEntity2(eid, prefab.tag2Mask);

  // Register the lifetime record.
  const lifetimePool = _mapPoolToLifetimePool(prefab.pool);
  markSpawning(eid, lifetimePool, prefab.ttlFrames);

  // Apply lifetime policy.
  switch (prefab.lifetime) {
    case PREFAB_LIFETIME.PERMANENT:
      setPersistent(eid, true);
      break;
    case PREFAB_LIFETIME.SCENE:
      setPersistent(eid, false);
      break;
    case PREFAB_LIFETIME.TRANSIENT:
      setTransient(eid, true);
      setPersistent(eid, false);
      break;
    case PREFAB_LIFETIME.EPHEMERAL:
      setTransient(eid, true);
      setPersistent(eid, false);
      if (prefab.ttlFrames === 0) setTTL(eid, 60);
      break;
    default:
      break;
  }

  // Attach behaviors (light prefabs).
  if (prefab.behaviors.length > 0) {
    for (let i = 0; i < prefab.behaviors.length; i++) {
      const b = prefab.behaviors[i];
      // Behavior IDs are already declared by the light policy; this
      // module records the attach request in the SoA LightBehavior
      // component and lets the policy attach the runtime behavior on
      // its next tick.
      if (LightBehavior.behaviorCount[eid] < MAX_BEHAVIORS_PER_LIGHT) {
        const off = eid * MAX_BEHAVIORS_PER_LIGHT + LightBehavior.behaviorCount[eid];
        LightBehavior.behaviorIds[off]   = b.id;
        LightBehavior.behaviorCtx0[off]  = b.ctx0;
        LightBehavior.behaviorCtx1[off]  = b.ctx1;
        LightBehavior.behaviorCtx2[off]  = b.ctx2;
        LightBehavior.behaviorCount[eid] = LightBehavior.behaviorCount[eid] + 1;
      }
    }
  }

  // Sync the tag bitmask into the SoA marker components (bitECS query).
  syncTagsToSoA(eid);

  // Update counters.
  PrefabState.totalSpawns++;
  PrefabState.totalByPrefab[prefabId]++;
  PrefabState.liveByPrefab[prefabId]++;
  if (PrefabState.liveByPrefab[prefabId] > PrefabState.peakLiveByPrefab[prefabId]) {
    PrefabState.peakLiveByPrefab[prefabId] = PrefabState.liveByPrefab[prefabId];
  }

  // Mark alive.
  markAlive(eid);

  return eid;
}

function _mapPoolToLifetimePool(pool) {
  switch (pool) {
    case POOL.CAMERA:    return LIFETIME_POOL.CAMERA;
    case POOL.LIGHT:     return LIFETIME_POOL.LIGHT;
    case POOL.SHADOW:    return LIFETIME_POOL.SHADOW;
    case POOL.GI:        return LIFETIME_POOL.GI;
    case POOL.AO:        return LIFETIME_POOL.AO;
    case POOL.SCENE:     return LIFETIME_POOL.SCENE;
    case POOL.STREAMING: return LIFETIME_POOL.STREAMING;
    case POOL.DEBUG:     return LIFETIME_POOL.DEBUG;
    case POOL.PARTICLE:  return LIFETIME_POOL.PARTICLE;
    case POOL.UI:        return LIFETIME_POOL.UI;
    case POOL.ANIMATION: return LIFETIME_POOL.ANIMATION;
    case POOL.WEATHER:   return LIFETIME_POOL.WEATHER;
    default:             return LIFETIME_POOL.NONE;
  }
}

/**
 * Releases a prefab entity back to its subsystem pool. Records the
 * release in the PrefabState counters.
 */
export function releasePrefab(eid) {
  if (typeof eid !== 'number' || eid < 0) return false;
  // Determine which prefab this entity came from — use the tag bits and
  // the pool binding recorded in the lifetime record.
  const pool = EntityLifetime.pool[eid];
  if (pool < 0 || pool >= LIFETIME_POOL.COUNT) return false;

  const ok = release(_mapLifetimePoolToPool(pool), eid);
  if (ok) PrefabState.totalReleases++;
  return ok;
}

function _mapLifetimePoolToPool(lifetimePool) {
  switch (lifetimePool) {
    case LIFETIME_POOL.CAMERA:    return POOL.CAMERA;
    case LIFETIME_POOL.LIGHT:     return POOL.LIGHT;
    case LIFETIME_POOL.SHADOW:    return POOL.SHADOW;
    case LIFETIME_POOL.GI:        return POOL.GI;
    case LIFETIME_POOL.AO:        return POOL.AO;
    case LIFETIME_POOL.SCENE:     return POOL.SCENE;
    case LIFETIME_POOL.STREAMING: return POOL.STREAMING;
    case LIFETIME_POOL.DEBUG:     return POOL.DEBUG;
    case LIFETIME_POOL.PARTICLE:  return POOL.PARTICLE;
    case LIFETIME_POOL.UI:        return POOL.UI;
    case LIFETIME_POOL.ANIMATION: return POOL.ANIMATION;
    case LIFETIME_POOL.WEATHER:   return POOL.WEATHER;
    default:                      return POOL.SCENE;
  }
}

/* ------------------------------------------------------------------ */
/* 9. CONVENIENCE SPAWNERS                                            */
/* ------------------------------------------------------------------ */

export function spawnSun(world, overrides)               { return spawnPrefab(world, PREFAB.SUN, overrides); }
export function spawnMoon(world, overrides)              { return spawnPrefab(world, PREFAB.MOON, overrides); }
export function spawnAmbient(world, overrides)           { return spawnPrefab(world, PREFAB.AMBIENT, overrides); }
export function spawnHemisphere(world, overrides)        { return spawnPrefab(world, PREFAB.HEMISPHERE, overrides); }
export function spawnSkyLight(world, overrides)          { return spawnPrefab(world, PREFAB.SKY_LIGHT, overrides); }
export function spawnFireLight(world, overrides)         { return spawnPrefab(world, PREFAB.FIRE_LIGHT, overrides); }
export function spawnNeonLight(world, overrides)         { return spawnPrefab(world, PREFAB.NEON_LIGHT, overrides); }
export function spawnMagicGlow(world, overrides)         { return spawnPrefab(world, PREFAB.MAGIC_GLOW, overrides); }
export function spawnInteriorLamp(world, overrides)      { return spawnPrefab(world, PREFAB.INTERIOR_LAMP, overrides); }
export function spawnWindowShaft(world, overrides)       { return spawnPrefab(world, PREFAB.WINDOW_SHAFT, overrides); }
export function spawnCaustic(world, overrides)           { return spawnPrefab(world, PREFAB.CAUSTIC, overrides); }
export function spawnAurora(world, overrides)            { return spawnPrefab(world, PREFAB.AURORA, overrides); }
export function spawnBioLuminescent(world, overrides)    { return spawnPrefab(world, PREFAB.BIO_LUMINESCENT, overrides); }
export function spawnPointLight(world, overrides)        { return spawnPrefab(world, PREFAB.POINT_LIGHT, overrides); }
export function spawnSpotLight(world, overrides)         { return spawnPrefab(world, PREFAB.SPOT_LIGHT, overrides); }
export function spawnRectAreaLight(world, overrides)     { return spawnPrefab(world, PREFAB.RECT_AREA_LIGHT, overrides); }

export function spawnShadowCaster(world, overrides)      { return spawnPrefab(world, PREFAB.SHADOW_CASTER, overrides); }
export function spawnShadowReceiver(world, overrides)    { return spawnPrefab(world, PREFAB.SHADOW_RECEIVER, overrides); }
export function spawnShadowAtlasEntity(world, overrides) { return spawnPrefab(world, PREFAB.SHADOW_ATLAS, overrides); }
export function spawnShadowCascadeSet(world, overrides)  { return spawnPrefab(world, PREFAB.SHADOW_CASCADE_SET, overrides); }
export function spawnContactShadow(world, overrides)     { return spawnPrefab(world, PREFAB.CONTACT_SHADOW, overrides); }

export function spawnGIProbe(world, overrides)           { return spawnPrefab(world, PREFAB.GI_PROBE, overrides); }
export function spawnGIHeroProbe(world, overrides)       { return spawnPrefab(world, PREFAB.GI_HERO_PROBE, overrides); }
export function spawnGIIndoorVolume(world, overrides)    { return spawnPrefab(world, PREFAB.GI_VOLUME_INDOOR, overrides); }
export function spawnGIOutdoorVolume(world, overrides)   { return spawnPrefab(world, PREFAB.GI_VOLUME_OUTDOOR, overrides); }
export function spawnGIPortalDoorway(world, overrides)   { return spawnPrefab(world, PREFAB.GI_PORTAL_DOORWAY, overrides); }
export function spawnGIPortalWindow(world, overrides)    { return spawnPrefab(world, PREFAB.GI_PORTAL_WINDOW, overrides); }
export function spawnGIPortalSkylight(world, overrides)  { return spawnPrefab(world, PREFAB.GI_PORTAL_SKYLIGHT, overrides); }
export function spawnGIPortalCave(world, overrides)      { return spawnPrefab(world, PREFAB.GI_PORTAL_CAVE, overrides); }
export function spawnReflectionProbe(world, overrides)   { return spawnPrefab(world, PREFAB.REFLECTION_PROBE, overrides); }
export function spawnLightfield(world, overrides)        { return spawnPrefab(world, PREFAB.LIGHTFIELD, overrides); }

export function spawnAOVolume(world, overrides)          { return spawnPrefab(world, PREFAB.AO_VOLUME, overrides); }
export function spawnAOIndoorVolume(world, overrides)    { return spawnPrefab(world, PREFAB.AO_INDOOR_VOLUME, overrides); }
export function spawnAOOutdoorVolume(world, overrides)   { return spawnPrefab(world, PREFAB.AO_OUTDOOR_VOLUME, overrides); }
export function spawnAOContactVolume(world, overrides)   { return spawnPrefab(world, PREFAB.AO_CONTACT_VOLUME, overrides); }
export function spawnAOCelController(world, overrides)   { return spawnPrefab(world, PREFAB.AO_CEL_CONTROLLER, overrides); }
export function spawnAOInkController(world, overrides)   { return spawnPrefab(world, PREFAB.AO_INK_CONTROLLER, overrides); }
export function spawnAODenoiser(world, overrides)        { return spawnPrefab(world, PREFAB.AO_DENOISER, overrides); }
export function spawnAOBilateral(world, overrides)       { return spawnPrefab(world, PREFAB.AO_BILATERAL, overrides); }

export function spawnCamera(world, overrides)            { return spawnPrefab(world, PREFAB.CAMERA, overrides); }
export function spawnOrbitCamera(world, overrides)       { return spawnPrefab(world, PREFAB.ORBIT_CAMERA, overrides); }
export function spawnDOFCamera(world, overrides)         { return spawnPrefab(world, PREFAB.DOF_CAMERA, overrides); }

export function spawnStreamingChunk(world, overrides)    { return spawnPrefab(world, PREFAB.STREAMING_CHUNK, overrides); }
export function spawnLODGroup(world, overrides)          { return spawnPrefab(world, PREFAB.LOD_GROUP, overrides); }
export function spawnDebugMarker(world, overrides)       { return spawnPrefab(world, PREFAB.DEBUG_MARKER, overrides); }
export function spawnDebugHUD(world, overrides)          { return spawnPrefab(world, PREFAB.DEBUG_HUD, overrides); }

/* ------------------------------------------------------------------ */
/* 10. PREWARM                                                        */
/* ------------------------------------------------------------------ */

/**
 * Prewarms a prefab by spawning `count` entities and immediately marking
 * them dying so they return to the pool. The effect is that the pool's
 * free list for that subsystem is fully populated with correctly
 * initialized entities.
 */
export function prewarmPrefab(prefabId, count) {
  const prefab = _prefabs[prefabId];
  if (!prefab) return 0;
  const n = Math.max(0, count | 0);
  let spawned = 0;
  for (let i = 0; i < n; i++) {
    const eid = spawnPrefab(null, prefabId, null);
    if (eid < 0) break;
    // Mark dying with zero grace so the next GC pass returns it.
    markDying(eid, 0);
    spawned++;
  }
  return spawned;
}

/* ------------------------------------------------------------------ */
/* 11. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getPrefabReport() {
  const byPrefab = new Array(MAX_PREFABS);
  for (let i = 0; i < MAX_PREFABS; i++) {
    if (!_prefabs[i]) continue;
    byPrefab[i] = {
      id:           i,
      name:         _prefabs[i].name,
      pool:         POOL[POOL_NAME[_prefabs[i].pool]] !== undefined ? _prefabs[i].pool : _prefabs[i].pool,
      lifetime:     _prefabs[i].lifetime,
      components:   _prefabs[i].componentNames.length,
      behaviors:    _prefabs[i].behaviors.length,
      maxLive:      _prefabs[i].maxLive,
      totalSpawns:  PrefabState.totalByPrefab[i],
      live:         PrefabState.liveByPrefab[i],
      peakLive:     PrefabState.peakLiveByPrefab[i],
    };
  }

  return {
    frame:           PrefabState.frame,
    totalSpawns:     PrefabState.totalSpawns,
    totalReleases:   PrefabState.totalReleases,
    totalRejected:   PrefabState.totalRejected,
    registeredPrefabs: _prefabs.filter((p) => p !== null).length,
    byPrefab,
    perfTier:        PERF_TIER_LOCAL,
  };
}

export function getPrefabStats(prefabId) {
  if (prefabId < 0 || prefabId >= MAX_PREFABS) return null;
  const prefab = _prefabs[prefabId];
  if (!prefab) return null;
  return {
    id:           prefabId,
    name:         prefab.name,
    totalSpawns:  PrefabState.totalByPrefab[prefabId],
    live:         PrefabState.liveByPrefab[prefabId],
    peakLive:     PrefabState.peakLiveByPrefab[prefabId],
  };
}

/* ------------------------------------------------------------------ */
/* 12. REGISTRATION                                                   */
/* ------------------------------------------------------------------ */

/**
 * The prefab module does not declare its own ECS components; it only
 * composes existing ones. No-op, present for API symmetry.
 */
export function registerPrefabComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 13. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Resets runtime counters. The prefab declarations themselves are
 * immutable and remain available.
 */
export function resetPrefabState() {
  PrefabState.totalSpawns = 0;
  PrefabState.totalReleases = 0;
  PrefabState.totalRejected = 0;
  PrefabState.totalByPrefab.fill(0);
  PrefabState.peakLiveByPrefab.fill(0);
  PrefabState.liveByPrefab.fill(0);
  PrefabState.frame = 0;
  for (let i = 0; i < MAX_PREFABS; i++) _prefabComponentCache[i] = null;
}

/**
 * Advances the prefab module frame counter.
 */
export function tickPrefabs(frameNumber) {
  if (typeof frameNumber === 'number') PrefabState.frame = frameNumber;
  else PrefabState.frame++;
}

/* ------------------------------------------------------------------ */
/* 14. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Enums
  PREFAB,
  PREFAB_NAME,
  PREFAB_LIFETIME,
  MAX_PREFABS,

  // Descriptor class
  PrefabDescriptor,

  // State
  PrefabState,

  // Registry
  getPrefab,
  getPrefabByName,

  // Spawn / release
  spawnPrefab,
  releasePrefab,
  prewarmPrefab,

  // Convenience spawners — lights
  spawnSun,
  spawnMoon,
  spawnAmbient,
  spawnHemisphere,
  spawnSkyLight,
  spawnFireLight,
  spawnNeonLight,
  spawnMagicGlow,
  spawnInteriorLamp,
  spawnWindowShaft,
  spawnCaustic,
  spawnAurora,
  spawnBioLuminescent,
  spawnPointLight,
  spawnSpotLight,
  spawnRectAreaLight,

  // Convenience spawners — shadow
  spawnShadowCaster,
  spawnShadowReceiver,
  spawnShadowAtlasEntity,
  spawnShadowCascadeSet,
  spawnContactShadow,

  // Convenience spawners — GI
  spawnGIProbe,
  spawnGIHeroProbe,
  spawnGIIndoorVolume,
  spawnGIOutdoorVolume,
  spawnGIPortalDoorway,
  spawnGIPortalWindow,
  spawnGIPortalSkylight,
  spawnGIPortalCave,
  spawnReflectionProbe,
  spawnLightfield,

  // Convenience spawners — AO
  spawnAOVolume,
  spawnAOIndoorVolume,
  spawnAOOutdoorVolume,
  spawnAOContactVolume,
  spawnAOCelController,
  spawnAOInkController,
  spawnAODenoiser,
  spawnAOBilateral,

  // Convenience spawners — camera / scene / debug
  spawnCamera,
  spawnOrbitCamera,
  spawnDOFCamera,
  spawnStreamingChunk,
  spawnLODGroup,
  spawnDebugMarker,
  spawnDebugHUD,

  // Diagnostics
  getPrefabReport,
  getPrefabStats,

  // Registration
  registerPrefabComponents,

  // Frame
  tickPrefabs,

  // Reset
  resetPrefabState,
};

export default _defaultExport;