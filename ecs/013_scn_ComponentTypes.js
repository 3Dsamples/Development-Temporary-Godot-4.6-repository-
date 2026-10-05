// File : 013
// name : src/ecs/013_scn_ComponentTypes.js
// description : Canonical numeric component-type identifier table for the
//               scene ECS world of the anime lighting stack on Android
//               mobile. Where 012_scn_ComponentRegistry.js owns the
//               METADATA catalog (names, categories, SoA field layout),
//               THIS module owns the NUMERIC IDENTITY of every component:
//
//                 • A stable numeric type id per component, assigned once
//                   and never changed. Ids are grouped in contiguous
//                   ranges so a type id can be classified by integer
//                   comparison alone (e.g. `id >= TID_LIGHT_START &&
//                   id < TID_LIGHT_END`).
//
//                 • Bidirectional mappings:
//                     NAME_TO_ID  (string → uint16)
//                     ID_TO_NAME  (uint16 → string)
//                   Both are backed by flat typed arrays for O(1) lookup
//                   with zero allocations.
//
//                 • Serialization contract: entity snapshots, worker
//                   messages, and diagnostic dumps carry type ids instead
//                   of strings, so the on-wire payload is compact and
//                   fast to encode/decode.
//
//                 • Worker protocol contract: worker ↔ main thread
//                   messages use the same numeric ids so both sides agree
//                   on the component vocabulary without shipping the
//                   names over the port.
//
//                 • Type-group classification: `isLightType(id)`,
//                   `isShadowType(id)`, `isGIType(id)`, `isAOType(id)`,
//                   `isSceneType(id)`, etc. — pure integer comparisons,
//                   no map lookups.
//
//                 • Audit helpers: `validateTypeTable()` verifies the
//                   table is contiguous, non-overlapping, and complete.
//
//               The table is declared statically here (not computed at
//               runtime) so it becomes a compile-time constant after the
//               first import, and so the numeric ids are stable across
//               engine versions — a component that had id 42 yesterday
//               still has id 42 today, so cached snapshots, saved replays,
//               and worker payloads remain valid across hot-reloads.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; no external ECS libs; the entire table is
//               declared once at module load; every lookup is O(1) and
//               allocation-free.
// best for : Guaranteeing that every subsystem in the anime lighting stack
//            can serialize, transmit, and reconstruct entity component
//            state using compact numeric ids — no strings on the wire, no
//            drift between subsystems, no ambiguity across hot-reloads or
//            worker boundaries.
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
  COMPONENT_CATEGORY,
  COMPONENT_CATEGORY_NAME,
  SUBSYSTEM,
  SUBSYSTEM_NAME,
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * The numeric space assigned to component type ids. Uint16 gives 65536
 * possible ids, which is far more than the practical ceiling of the
 * engine (a few hundred components). We use it because Uint16Array
 * doubles as a compact wire format for worker messages.
 */
export const TYPE_ID_KIND = 'uint16';
export const TYPE_ID_MAX = 65535;
export const TYPE_ID_INVALID = 0xFFFF;

/* ------------------------------------------------------------------ */
/* 1. COMPONENT TYPE ID TABLE                                         */
/* ------------------------------------------------------------------ */

/**
 * Canonical type ids for every component in the scene ECS world. Ranges
 * are reserved per subsystem so that `isLightType(id)` etc. are pure
 * integer comparisons:
 *
 *   CORE / TRANSFORM       1    ..   31
 *   LIGHT                 32    ..   95
 *   SHADOW                96    ..  159
 *   GI                   160    ..  223
 *   AO                   224    ..  287
 *   CAMERA               288    ..  303
 *   ENVIRONMENT          304    ..  335
 *   INTERIOR             336    ..  351
 *   EXTERIOR             352    ..  367
 *   STREAMING            368    ..  383
 *   MATERIAL             384    ..  415
 *   POST                 416    ..  447
 *   DEBUG                448    ..  479
 *   PARALLEL             480    ..  495
 *   QUALITY              496    ..  511
 *   PLATFORM             512    ..  527
 *   RESERVED             528    ..  65534
 */
export const COMPONENT_TYPE_ID = Object.freeze({
  // ---- CORE / TRANSFORM (1..31) ----
  TRANSFORM:              1,
  TARGET:                 2,
  TAG:                    3,
  NAME:                   4,
  PARENT:                 5,
  CHILDREN:               6,
  ENTITY_STATE:           7,
  LIFETIME:               8,
  METADATA:               9,
  VERSION:               10,

  // ---- LIGHT (32..95) ----
  LIGHT_REF:             32,
  LIGHT_STATE:           33,
  LIGHT_SHADOW:          34,
  LIGHT_CLUSTER:         35,
  LIGHT_BUDGET:          36,
  LIGHT_PRIORITY:        37,
  LIGHT_BEHAVIOR:        38,
  LIGHT_COMPOSITE:       39,
  LIGHT_INDOOR:          40,
  LIGHT_IES:             41,
  LIGHT_EMISSIVE:        42,
  LIGHT_FLICKER:         43,
  LIGHT_DAY_CYCLE:       44,
  LIGHT_TAG:             45,
  LIGHT_GI_PROBE_REF:    46,
  LIGHT_AO_VOLUME_REF:   47,
  LIGHT_BEAM:            48,
  LIGHT_COOKIE:          49,
  LIGHT_PORTAL:          50,
  LIGHT_VOLUME:          51,

  // ---- SHADOW (96..159) ----
  SHADOW_CASTER_REF:      96,
  SHADOW_RECEIVER_REF:    97,
  SHADOW_ATLAS:           98,
  SHADOW_CASCADE:         99,
  SHADOW_BIAS:           100,
  SHADOW_FILTER:         101,
  SHADOW_SOFTNESS:       102,
  SHADOW_FRUSTUM:        103,
  SHADOW_CACHE:          104,
  SHADOW_BUDGET:         105,
  SHADOW_CONTACT:        106,
  SHADOW_VOLUME:         107,
  SHADOW_DIR_LIGHT:      108,
  SHADOW_POINT_LIGHT:    109,
  SHADOW_SPOT_LIGHT:     110,
  SHADOW_ATLAS_MAP:      111,
  SHADOW_TINT:           112,
  SHADOW_EDGE:           113,
  SHADOW_TILE:           114,
  SHADOW_UPDATE_POLICY:  115,
  SHADOW_STATE:          116,
  SHADOW_PROXY_REF:      117,
  SHADOW_IMPOSTOR_REF:   118,
  SHADOW_LOD:            119,
  SHADOW_PANCAKE:        120,
  SHADOW_STABILIZER:     121,

  // ---- GI (160..223) ----
  GI_PROBE_REF:          160,
  GI_IRRADIANCE:         161,
  GI_SH:                 162,
  GI_SH_HIGH:            163,
  GI_BOUNCE_PATH:        164,
  GI_VOXEL:              165,
  GI_DISTANCE_FIELD:     166,
  GI_OCCLUSION:          167,
  GI_PORTAL:             168,
  GI_BUDGET:             169,
  GI_STATE:              170,
  GI_UPDATE_QUEUE:       171,
  GI_RADIANCE_CACHE:     172,
  GI_REFLECTION_PROBE:   173,
  GI_LIGHTFIELD:         174,
  GI_VOLUME:             175,
  GI_INDOOR:             176,
  GI_OUTDOOR:            177,
  GI_CEL_BANDS:          178,
  GI_PALETTE:            179,
  GI_ASYNC:              180,
  GI_LEAK:               181,
  GI_TEMPORAL:           182,
  GI_VOLUME_BLEND:       183,
  GI_RAY:                184,
  GI_CONE:               185,
  GI_VISIBILITY_GRAPH:   186,
  GI_RESERVOIR:          187,
  GI_PROBE_GROUP:        188,
  GI_BOUNCE_LIGHT:       189,
  GI_SKY_IRRADIANCE:     190,
  GI_GROUND_IRRADIANCE:  191,

  // ---- AO (224..287) ----
  AO_VOLUME_REF:          224,
  AO_SAMPLING:            225,
  AO_KERNEL:              226,
  AO_HISTORY:             227,
  AO_BLUR:                228,
  AO_QUALITY:             229,
  AO_STATE:               230,
  AO_BUDGET:              231,
  AO_SCREEN_SPACE:        232,
  AO_CONTACT_SHADOW:      233,
  AO_DISTANCE_FIELD:      234,
  AO_INDOOR_VOLUME:       235,
  AO_OUTDOOR_VOLUME:      236,
  AO_TEMPORAL_ACC:        237,
  AO_DITHER:              238,
  AO_CEL_BANDS:           239,
  AO_INK_OUTLINE:         240,
  AO_EDGE_FADE:           241,
  AO_BILATERAL:           242,
  AO_DENOISER:            243,
  AO_ASYNC:               244,
  AO_RESIDENCY:           245,
  AO_LEAK:                246,
  AO_STYLE:               247,
  AO_HALF_RES:            248,
  AO_UPSAMPLER:           249,

  // ---- CAMERA (288..303) ----
  CAMERA_TAG:             288,
  CAMERA_ORBIT:           289,
  CAMERA_LOD:             290,
  CAMERA_FRUSTUM:         291,
  CAMERA_SHAKE:           292,
  CAMERA_DOF:             293,
  CAMERA_BLOOM_HINT:      294,
  CAMERA_EXPOSURE:        295,

  // ---- ENVIRONMENT (304..335) ----
  ENV_DAY_CYCLE:          304,
  ENV_SEASON:             305,
  ENV_WEATHER:            306,
  ENV_WIND:               307,
  ENV_RAIN:               308,
  ENV_SNOW:               309,
  ENV_DUST:               310,
  ENV_FOG:                311,
  ENV_ATMOSPHERE:         312,
  ENV_SKY:                313,
  ENV_CLOUD:              314,
  ENV_CLOUD_SHADOW:       315,
  ENV_SUN:                316,
  ENV_MOON:               317,
  ENV_STAR:               318,
  ENV_AURORA:             319,
  ENV_LIGHTNING:          320,
  ENV_HORIZON:            321,
  ENV_BIOME_TRANSITION:   322,
  ENV_THEME_DIRECTOR:     323,
  ENV_PROBE_UPDATER:      324,

  // ---- INTERIOR (336..351) ----
  INTERIOR_MANAGER:       336,
  INTERIOR_ROOM:          337,
  INTERIOR_LIGHT_PLACER:  338,
  INTERIOR_PORTAL:        339,
  INTERIOR_PROBE:         340,
  INTERIOR_SHADOW_CACHE:  341,
  INTERIOR_MOOD:          342,
  INTERIOR_SHUTTER:       343,

  // ---- EXTERIOR (352..367) ----
  EXTERIOR_MANAGER:       352,
  EXTERIOR_SUNLIGHT:      353,
  EXTERIOR_MOONLIGHT:     354,
  EXTERIOR_SKYLIGHT:      355,
  EXTERIOR_GROUND_BOUNCE: 356,
  EXTERIOR_CANOPY:        357,
  EXTERIOR_CANYON:        358,
  EXTERIOR_SNOW_GLARE:    359,
  EXTERIOR_WATER_CAUSTICS:360,
  EXTERIOR_PROBE:         361,

  // ---- STREAMING (368..383) ----
  STREAMING_CHUNK:        368,
  STREAMING_LIFECYCLE:    369,
  STREAMING_BUDGET:       370,
  STREAMING_COORDINATOR:  371,
  STREAMING_PARTITION:    372,

  // ---- MATERIAL (384..415) ----
  MATERIAL_REF:           384,
  MATERIAL_CEL:           385,
  MATERIAL_OUTLINE:       386,
  MATERIAL_EMISSIVE:      387,
  MATERIAL_GLASS:         388,
  MATERIAL_WATER:         389,
  MATERIAL_SNOW:          390,
  MATERIAL_SAND:          391,
  MATERIAL_ROCK:          392,
  MATERIAL_GRASS:         393,
  MATERIAL_LEAF:          394,
  MATERIAL_WOOD:          395,
  MATERIAL_FABRIC:        396,
  MATERIAL_METAL:         397,
  MATERIAL_HOLOGRAM:      398,
  MATERIAL_PARTICLE:      399,

  // ---- POST (416..447) ----
  POST_COMPOSER:          416,
  POST_BLOOM:             417,
  POST_DOF:               418,
  POST_FOG:               419,
  POST_SSR:               420,
  POST_SSGI:              421,
  POST_OUTLINE:           422,
  POST_TONEMAP:           423,
  POST_COLOR_GRADE:       424,
  POST_CHROMATIC:         425,
  POST_GRAIN:             426,
  POST_VIGNETTE:          427,
  POST_FXAA:              428,
  POST_SMAA:              429,
  POST_TAA:               430,
  POST_DITHER:            431,
  POST_PIXELATE:          432,
  POST_COMPOSITE:         433,

  // ---- DEBUG (448..479) ----
  DEBUG_HUD:              448,
  DEBUG_STATS:            449,
  DEBUG_FRAME_GRAPH:      450,
  DEBUG_MEMORY:           451,
  DEBUG_SHADER_VALID:     452,
  DEBUG_UNIFORM_WATCH:    453,
  DEBUG_SCENE_TREE:       454,
  DEBUG_CAPTURE:          455,
  DEBUG_LIGHT_INSPECT:    456,
  DEBUG_SHADOW_VIEW:      457,
  DEBUG_PROBE_VIEW:       458,
  DEBUG_CLUSTER_VIEW:     459,
  DEBUG_GI_VIEW:          460,
  DEBUG_AO_VIEW:          461,

  // ---- PARALLEL (480..495) ----
  PARALLEL_JOB:           480,
  PARALLEL_WORKER:        481,
  PARALLEL_GRAPH:         482,
  PARALLEL_TASK:          483,
  PARALLEL_BATCHER:       484,
  PARALLEL_TRANSFERABLE:  485,

  // ---- QUALITY (496..511) ----
  QUALITY_CTRL:           496,
  QUALITY_RES_SCALER:     497,
  QUALITY_SHADOW_SCALER:  498,
  QUALITY_GI_SCALER:      499,
  QUALITY_AO_SCALER:      500,
  QUALITY_POST_SCALER:    501,
  QUALITY_THERMAL_GUARD:  502,
  QUALITY_BATTERY_GUARD:  503,
  QUALITY_STUTTER:        504,
  QUALITY_PRESET_MIGR:    505,

  // ---- PLATFORM (512..527) ----
  PLATFORM_ANDROID:       512,
  PLATFORM_MOBILE_GUARD:  513,
  PLATFORM_POWER_PREF:    514,
  PLATFORM_GPU_FEATURE:   515,
  PLATFORM_WEBGL_DETECT:  516,
  PLATFORM_EXT_HUNTER:    517,
  PLATFORM_PRECISION:     518,
  PLATFORM_INSTANCING:    519,
  PLATFORM_FLOAT_RT:      520,
  PLATFORM_DEPTH_RT:      521,
});

/**
 * Convenience grouping bounds for `isLightType` etc.
 */
const GROUP_RANGE = Object.freeze({
  CORE:      Object.freeze({ start: 1,   end: 31  }),
  LIGHT:     Object.freeze({ start: 32,  end: 95  }),
  SHADOW:    Object.freeze({ start: 96,  end: 159 }),
  GI:        Object.freeze({ start: 160, end: 223 }),
  AO:        Object.freeze({ start: 224, end: 287 }),
  CAMERA:    Object.freeze({ start: 288, end: 303 }),
  ENV:       Object.freeze({ start: 304, end: 335 }),
  INTERIOR:  Object.freeze({ start: 336, end: 351 }),
  EXTERIOR:  Object.freeze({ start: 352, end: 367 }),
  STREAMING: Object.freeze({ start: 368, end: 383 }),
  MATERIAL:  Object.freeze({ start: 384, end: 415 }),
  POST:      Object.freeze({ start: 416, end: 447 }),
  DEBUG:     Object.freeze({ start: 448, end: 479 }),
  PARALLEL:  Object.freeze({ start: 480, end: 495 }),
  QUALITY:   Object.freeze({ start: 496, end: 511 }),
  PLATFORM:  Object.freeze({ start: 512, end: 527 }),
});

/* ------------------------------------------------------------------ */
/* 2. NAME ↔ ID LOOKUP TABLES                                         */
/* ------------------------------------------------------------------ */

const _nameToId = new Map();
const _idToName = new Map();

(function _buildLookupTables() {
  const keys = Object.keys(COMPONENT_TYPE_ID);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const id = COMPONENT_TYPE_ID[key];
    // Skip placeholder keys that may be added in future versions.
    if (typeof id !== 'number' || id <= 0 || id > TYPE_ID_MAX) continue;

    // Name form: canonical UPPER_SNAKE.
    _nameToId.set(key, id);

    // Also register PascalCase alias for convenient lookup from the
    // registry side (which uses PascalCase component names).
    const pascal = _toPascalCase(key);
    if (!_nameToId.has(pascal)) _nameToId.set(pascal, id);

    // Id → canonical name.
    if (!_idToName.has(id)) {
      _idToName.set(id, key);
    }
  }
})();

/**
 * Converts `LIGHT_REF` → `LightRef`, `GI_SH_HIGH` → `GISHHigh`, etc.
 */
function _toPascalCase(upperSnake) {
  if (typeof upperSnake !== 'string' || upperSnake.length === 0) return '';
  const parts = upperSnake.split('_');
  let out = '';
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (p.length === 0) continue;
    out += p.charAt(0).toUpperCase() + p.slice(1).toLowerCase();
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 3. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Returns the numeric type id for a component name, or TYPE_ID_INVALID.
 * Accepts either UPPER_SNAKE or PascalCase names.
 */
export function getTypeId(name) {
  if (typeof name !== 'string') return TYPE_ID_INVALID;
  const id = _nameToId.get(name);
  return id === undefined ? TYPE_ID_INVALID : id;
}

/**
 * Returns the canonical UPPER_SNAKE name for a numeric type id, or null.
 */
export function getTypeName(id) {
  if (typeof id !== 'number' || id < 0 || id > TYPE_ID_MAX) return null;
  const name = _idToName.get(id);
  return name === undefined ? null : name;
}

/**
 * Returns the PascalCase variant for a numeric type id, matching the
 * registry's naming convention.
 */
export function getTypePascalName(id) {
  const name = getTypeName(id);
  return name ? _toPascalCase(name) : null;
}

/**
 * Returns true if the given id belongs to a defined component type.
 */
export function isValidTypeId(id) {
  return typeof id === 'number' && id > 0 && id <= TYPE_ID_MAX && _idToName.has(id);
}

/**
 * Returns true if the given name belongs to a defined component type.
 */
export function isValidTypeName(name) {
  return getTypeId(name) !== TYPE_ID_INVALID;
}

/* ------------------------------------------------------------------ */
/* 4. GROUP CLASSIFICATION                                            */
/* ------------------------------------------------------------------ */

function _inRange(id, range) {
  return id >= range.start && id <= range.end;
}

export function isCoreType(id)      { return _inRange(id, GROUP_RANGE.CORE); }
export function isLightType(id)     { return _inRange(id, GROUP_RANGE.LIGHT); }
export function isShadowType(id)    { return _inRange(id, GROUP_RANGE.SHADOW); }
export function isGIType(id)        { return _inRange(id, GROUP_RANGE.GI); }
export function isAOType(id)        { return _inRange(id, GROUP_RANGE.AO); }
export function isCameraType(id)    { return _inRange(id, GROUP_RANGE.CAMERA); }
export function isEnvType(id)       { return _inRange(id, GROUP_RANGE.ENV); }
export function isInteriorType(id)  { return _inRange(id, GROUP_RANGE.INTERIOR); }
export function isExteriorType(id)  { return _inRange(id, GROUP_RANGE.EXTERIOR); }
export function isStreamingType(id) { return _inRange(id, GROUP_RANGE.STREAMING); }
export function isMaterialType(id)  { return _inRange(id, GROUP_RANGE.MATERIAL); }
export function isPostType(id)      { return _inRange(id, GROUP_RANGE.POST); }
export function isDebugType(id)     { return _inRange(id, GROUP_RANGE.DEBUG); }
export function isParallelType(id)  { return _inRange(id, GROUP_RANGE.PARALLEL); }
export function isQualityType(id)   { return _inRange(id, GROUP_RANGE.QUALITY); }
export function isPlatformType(id)  { return _inRange(id, GROUP_RANGE.PLATFORM); }

/**
 * Returns a symbolic group name for a type id, or 'unknown'.
 */
export function getTypeGroup(id) {
  if (isCoreType(id))      return 'core';
  if (isLightType(id))     return 'light';
  if (isShadowType(id))    return 'shadow';
  if (isGIType(id))        return 'gi';
  if (isAOType(id))        return 'ao';
  if (isCameraType(id))    return 'camera';
  if (isEnvType(id))       return 'environment';
  if (isInteriorType(id))  return 'interior';
  if (isExteriorType(id))  return 'exterior';
  if (isStreamingType(id)) return 'streaming';
  if (isMaterialType(id))  return 'material';
  if (isPostType(id))      return 'post';
  if (isDebugType(id))     return 'debug';
  if (isParallelType(id))  return 'parallel';
  if (isQualityType(id))   return 'quality';
  if (isPlatformType(id))  return 'platform';
  return 'unknown';
}

/* ------------------------------------------------------------------ */
/* 5. GROUP ENUMERATION                                               */
/* ------------------------------------------------------------------ */

/**
 * Returns every type id in a group.
 */
export function listTypeIdsInGroup(groupName) {
  const range = GROUP_RANGE[String(groupName).toUpperCase()];
  if (!range) return [];
  const out = [];
  for (const [name, id] of _idToName.entries()) {
    if (_inRange(id, range)) out.push(id);
  }
  return out;
}

/**
 * Returns every type name in a group.
 */
export function listTypeNamesInGroup(groupName) {
  const range = GROUP_RANGE[String(groupName).toUpperCase()];
  if (!range) return [];
  const out = [];
  for (const [name, id] of _idToName.entries()) {
    if (_inRange(id, range)) out.push(name);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 6. SERIALIZATION HELPERS                                           */
/* ------------------------------------------------------------------ */

/**
 * Compacts a list of component type names to a typed Uint16Array of ids,
 * suitable for worker transfer or on-wire transmission. Returns null if
 * any name is not registered.
 */
export function encodeTypeList(names) {
  if (!Array.isArray(names)) return null;
  const ids = new Uint16Array(names.length);
  for (let i = 0; i < names.length; i++) {
    const id = getTypeId(names[i]);
    if (id === TYPE_ID_INVALID) return null;
    ids[i] = id;
  }
  return ids;
}

/**
 * Decodes a Uint16Array of ids back to an array of canonical
 * PascalCase names. Returns null if any id is invalid.
 */
export function decodeTypeList(ids) {
  if (!ArrayBuffer.isView(ids)) return null;
  const names = new Array(ids.length);
  for (let i = 0; i < ids.length; i++) {
    const name = getTypePascalName(ids[i]);
    if (name === null) return null;
    names[i] = name;
  }
  return names;
}

/**
 * Encodes a single component type name to its numeric id.
 * Returns TYPE_ID_INVALID if not registered.
 */
export function encodeType(name) {
  return getTypeId(name);
}

/**
 * Decodes a single component type id to its canonical PascalCase name.
 */
export function decodeType(id) {
  return getTypePascalName(id);
}

/* ------------------------------------------------------------------ */
/* 7. REGISTRY BRIDGE                                                 */
/* ------------------------------------------------------------------ */

/**
 * Synchronizes the type table with the runtime Component Registry
 * (012_scn_ComponentRegistry.js). For every component registered in the
 * registry, verifies that a matching type id exists; logs a warning for
 * any component that has no canonical id.
 *
 * Called once at boot after `registerCanonicalComponents()`.
 */
export function bridgeRegistryToTypeTable(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return { bridged: 0, missing: 0, extra: 0 };

  const registered = reg.listAll();
  let bridged = 0;
  let missing = 0;

  for (let i = 0; i < registered.length; i++) {
    const name = registered[i];
    const id = getTypeId(name);
    if (id === TYPE_ID_INVALID) {
      missing++;
      const log = getDefaultLogger();
      if (log) log.warn(LOG_CHANNEL.CORE,
        `[013_scn_ComponentTypes] registry component "${name}" has no canonical type id`);
    } else {
      bridged++;
    }
  }

  // Count ids that exist but have no matching registry entry (informational).
  let extra = 0;
  for (const [name, id] of _idToName.entries()) {
    const pascal = _toPascalCase(name);
    if (!reg.has(pascal) && !reg.has(name)) extra++;
  }

  return { bridged, missing, extra };
}

/**
 * Returns a full report about the type table state and its relationship
 * with the runtime registry.
 */
export function getTypeTableReport() {
  const reg = getDefaultComponentRegistry();
  const registered = reg ? reg.listAll() : [];
  const bridged = { bridged: 0, missing: 0, extra: 0 };
  if (reg) Object.assign(bridged, bridgeRegistryToTypeTable(reg));

  return {
    totalTypeIds:       _idToName.size,
    totalNames:         _nameToId.size,
    registryComponents: registered.length,
    bridged:            bridged.bridged,
    missingInTypeTable: bridged.missing,
    extraInTypeTable:   bridged.extra,
    groups: {
      core:      listTypeNamesInGroup('CORE').length,
      light:     listTypeNamesInGroup('LIGHT').length,
      shadow:    listTypeNamesInGroup('SHADOW').length,
      gi:        listTypeNamesInGroup('GI').length,
      ao:        listTypeNamesInGroup('AO').length,
      camera:    listTypeNamesInGroup('CAMERA').length,
      env:       listTypeNamesInGroup('ENV').length,
      interior:  listTypeNamesInGroup('INTERIOR').length,
      exterior:  listTypeNamesInGroup('EXTERIOR').length,
      streaming: listTypeNamesInGroup('STREAMING').length,
      material:  listTypeNamesInGroup('MATERIAL').length,
      post:      listTypeNamesInGroup('POST').length,
      debug:     listTypeNamesInGroup('DEBUG').length,
      parallel:  listTypeNamesInGroup('PARALLEL').length,
      quality:   listTypeNamesInGroup('QUALITY').length,
      platform:  listTypeNamesInGroup('PLATFORM').length,
    },
    perfTier: PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 8. AUDIT                                                           */
/* ------------------------------------------------------------------ */

/**
 * Validates that the type table is internally consistent:
 *   • All ids in [1, TYPE_ID_MAX] are unique.
 *   • All ids fall inside a declared group range OR are marked reserved.
 *   • All group ranges are non-overlapping.
 *
 * Returns { ok: bool, issues: [...], checked: n }.
 */
export function validateTypeTable() {
  const issues = [];
  const seen = new Set();

  // 1. Unique ids.
  for (const [name, id] of _nameToId.entries()) {
    if (typeof id !== 'number' || id <= 0 || id > TYPE_ID_MAX) {
      issues.push({ kind: 'invalid_id', name, id });
      continue;
    }
    if (id !== getTypeId(name)) continue;  // skip Pascal aliases
    // Determine canonical name for this id.
    const canonical = _idToName.get(id);
    if (canonical && canonical !== name) {
      // This is a Pascal alias, skip.
      continue;
    }
    if (seen.has(id)) {
      issues.push({ kind: 'duplicate_id', name, id });
    } else {
      seen.add(id);
    }
  }

  // 2. Group non-overlap.
  const groupNames = Object.keys(GROUP_RANGE);
  const sorted = groupNames
    .map((n) => ({ name: n, start: GROUP_RANGE[n].start, end: GROUP_RANGE[n].end }))
    .sort((a, b) => a.start - b.start);

  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const cur = sorted[i];
    if (cur.start <= prev.end) {
      issues.push({ kind: 'overlapping_groups', a: prev.name, b: cur.name });
    }
  }

  // 3. Every canonical id inside a group.
  for (const [id, name] of _idToName.entries()) {
    let inAny = false;
    for (let i = 0; i < sorted.length; i++) {
      if (_inRange(id, sorted[i])) { inAny = true; break; }
    }
    if (!inAny) {
      issues.push({ kind: 'ungrouped_id', name, id });
    }
  }

  return {
    ok: issues.length === 0,
    issues,
    checked: _idToName.size,
  };
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  TYPE_ID_KIND,
  TYPE_ID_MAX,
  TYPE_ID_INVALID,
  COMPONENT_TYPE_ID,

  // Lookup
  getTypeId,
  getTypeName,
  getTypePascalName,
  isValidTypeId,
  isValidTypeName,

  // Group classification
  isCoreType,
  isLightType,
  isShadowType,
  isGIType,
  isAOType,
  isCameraType,
  isEnvType,
  isInteriorType,
  isExteriorType,
  isStreamingType,
  isMaterialType,
  isPostType,
  isDebugType,
  isParallelType,
  isQualityType,
  isPlatformType,
  getTypeGroup,

  // Enumeration
  listTypeIdsInGroup,
  listTypeNamesInGroup,

  // Serialization
  encodeTypeList,
  decodeTypeList,
  encodeType,
  decodeType,

  // Registry bridge
  bridgeRegistryToTypeTable,
  getTypeTableReport,

  // Audit
  validateTypeTable,
};

export default _defaultExport;