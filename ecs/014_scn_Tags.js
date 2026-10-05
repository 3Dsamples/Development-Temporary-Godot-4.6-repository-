// File : 014
// name : src/ecs/014_scn_Tags.js
// description : Canonical tag-component module for the scene ECS world of the
//               anime lighting stack on Android mobile. Every pure-marker
//               component in the engine lives here — tags are bitmask flags
//               stored in fixed-capacity SoA typed arrays sized once to
//               MAX_ENTITIES = 100000, so tagging and querying is a single
//               integer operation with zero allocations and zero map lookups.
//
//               Two complementary forms are provided:
//
//                 1. TAG BITMASK (the primary form)
//                    A single Uint32Array EntityTag of length MAX_ENTITIES,
//                    where each bit represents one tag category. Up to 32
//                    primary tags per entity. `tagEntity(eid, TAG)` and
//                    `untagEntity(eid, TAG)` are a single bitwise OR / AND
//                    NOT. `hasTag(eid, TAG)` is one AND + compare. This is
//                    the fastest possible tagging in JavaScript and is what
//                    hot paths use.
//
//                 2. SOA TAG COMPONENTS (for bitECS interop)
//                    Each major tag category also exposes a Uint8Array
//                    marker component so that bitECS `query([TagComponent])`
//                    works exactly as expected. These arrays share the same
//                    semantic meaning as the bitmask but can be used with
//                    bitECS's own query engine.
//
//               The two forms are kept in sync by the helper functions
//               (`tagEntity` / `untagEntity` / `hasTag` update both) so a
//               downstream system can choose either interface without
//               worrying about divergence.
//
//               Tag categories declared:
//                 ─── LIGHTING TAGS ───
//                   LightTag, SunTag, MoonTag, AmbientTag, HemisphereTag,
//                   DirectionalTag, PointTag, SpotTag, RectAreaTag,
//                   EmissiveTag, ShadowCasterTag, ShadowReceiverTag,
//                   ClusterLightTag, IndoorLightTag, OutdoorLightTag
//
//                 ─── CAMERA TAGS ───
//                   CameraTag, ActiveCameraTag, OrbitCameraTag, DOFCameraTag
//
//                 ─── SHADOW TAGS ───
//                   ShadowLightTag, ShadowAtlasOwnerTag, CascadeOwnerTag,
//                   ContactShadowTag, ShadowProxyTag, ShadowImpostorTag
//
//                 ─── GI TAGS ───
//                   GIProbeTag, GIHeroProbeTag, GIVolumeTag, GIPortalTag,
//                   GIReflectionProbeTag, GILightfieldTag, GIDirtyTag,
//                   GIAsyncTag
//
//                 ─── AO TAGS ───
//                   AOVolumeTag, AOIndoorTag, AOOutdoorTag, AOContactTag,
//                   AOCelBandTag, AOInkOutlineTag, AODirtyTag
//
//                 ─── SCENE / STATE TAGS ───
//                   ActiveTag, VisibleTag, EnabledTag, DisabledTag,
//                   FrustumCulledTag, InViewTag, DirtyTag, StaticTag,
//                   DynamicTag, AsyncPendingTag, NeedsUpdateTag,
//                   TransitionTag, HiddenTag, DebugTag
//
//                 ─── LIFECYCLE TAGS ───
//                   SpawnPendingTag, DestroyPendingTag, PersistentTag,
//                   TransientTag, PooledTag
//
//                 ─── STREAMING TAGS ───
//                   StreamingChunkTag, StreamingResidentTag,
//                   StreamingEvictTag, LODGroupTag, ImpostorTag
//
//                 ─── BIOME TAGS ───
//                   DesertTag, SnowTag, SeaTag, ForestTag, CanyonTag,
//                   CoastalTag, WetlandTag, TundraTag, VolcanicTag
//
//                 ─── INTERIOR / EXTERIOR TAGS ───
//                   InteriorTag, ExteriorTag, TransitionZoneTag,
//                   PortalConnectedTag, OccludedTag
//
//                 ─── PLAYER / ENTITY TAGS ───
//                   PlayerTag, EnemyTag, ProjectileTag, InteractiveTag
//
//                 ─── DEBUG TAGS ───
//                   DebugHUDTag, DebugOnlyTag, DebugFreezeTag
//
//               Integration:
//                 • 002_lgt_LightComponents.js — LightTag component
//                 • 010_scn_ECSWorld.js — world handle
//                 • 011_scn_BiteCSAdapter.js — entity lifecycle
//                 • 012_scn_ComponentRegistry.js — catalog
//                 • 013_scn_ComponentTypes.js — numeric type ids
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every typed array sized once at construction.
// best for : Guaranteeing that the entire anime lighting stack can tag,
//            untag, and query entities with a single integer operation —
//            no allocations, no map lookups, no drift between systems —
//            while still exposing bitECS-native tag components for
//            systems that prefer the standard query interface.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  addComponent,
  removeComponent,
  hasComponent,
  query,
  entityExists,
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  COMPONENT_TYPE_ID,
  getTypeId,
  isLightType,
  isShadowType,
  isGIType,
  isAOType,
} from './013_scn_ComponentTypes.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * The primary tag bitmask. Each bit is one of the TAG_* constants below.
 * A single Uint32Array covers every entity.
 */
export const EntityTag = new Uint32Array(MAX_ENTITIES);

/**
 * Secondary tag bitmask, for tags beyond the 32-bit primary set. Kept
 * separate so the primary set (the hottest tags) always fits in a single
 * cache line per group of 16 entities.
 */
export const EntityTag2 = new Uint32Array(MAX_ENTITIES);

/**
 * Frame-scoped dirty tag bitmask. Reset once per frame by `tickTags()`.
 * Used for "needs update this frame" style tags.
 */
export const FrameDirtyTag = new Uint32Array(MAX_ENTITIES);

/* ------------------------------------------------------------------ */
/* 1. PRIMARY TAG BITS (EntityTag)                                    */
/* ------------------------------------------------------------------ */

export const TAG = Object.freeze({
  NONE:                   0,

  // ---- LIGHTING ----
  LIGHT:                  1 << 0,    // any light
  SUN:                    1 << 1,
  MOON:                   1 << 2,
  AMBIENT:                1 << 3,
  HEMISPHERE:             1 << 4,
  DIRECTIONAL:            1 << 5,
  POINT:                  1 << 6,
  SPOT:                   1 << 7,
  RECT_AREA:              1 << 8,
  EMISSIVE:               1 << 9,
  CLUSTER_LIGHT:          1 << 10,

  // ---- SHADOW ----
  SHADOW_CASTER:          1 << 11,
  SHADOW_RECEIVER:        1 << 12,
  SHADOW_ATLAS_OWNER:     1 << 13,
  CASCADE_OWNER:          1 << 14,

  // ---- GI ----
  GI_PROBE:               1 << 15,
  GI_HERO_PROBE:          1 << 16,
  GI_VOLUME:              1 << 17,
  GI_PORTAL:              1 << 18,
  GI_REFLECTION_PROBE:    1 << 19,

  // ---- AO ----
  AO_VOLUME:              1 << 20,
  AO_INDOOR:              1 << 21,
  AO_OUTDOOR:             1 << 22,
  AO_CEL_BAND:            1 << 23,

  // ---- CAMERA ----
  CAMERA:                 1 << 24,
  ACTIVE_CAMERA:          1 << 25,

  // ---- SCENE STATE ----
  ACTIVE:                 1 << 26,
  VISIBLE:                1 << 27,
  DIRTY:                  1 << 28,
  STATIC:                 1 << 29,
  DYNAMIC:                1 << 30,
  DEBUG:                  1 << 31,   // last bit of the primary mask
});

/* ------------------------------------------------------------------ */
/* 2. SECONDARY TAG BITS (EntityTag2)                                 */
/* ------------------------------------------------------------------ */

export const TAG2 = Object.freeze({
  NONE:                   0,

  // ---- LIGHTING (cont.) ----
  INDOOR_LIGHT:           1 << 0,
  OUTDOOR_LIGHT:          1 << 1,

  // ---- SHADOW (cont.) ----
  CONTACT_SHADOW:         1 << 2,
  SHADOW_PROXY:           1 << 3,
  SHADOW_IMPOSTOR:        1 << 4,

  // ---- GI (cont.) ----
  GI_LIGHTFIELD:          1 << 5,
  GI_DIRTY:               1 << 6,
  GI_ASYNC:               1 << 7,

  // ---- AO (cont.) ----
  AO_CONTACT:             1 << 8,
  AO_INK_OUTLINE:         1 << 9,
  AO_DIRTY:               1 << 10,

  // ---- CAMERA (cont.) ----
  ORBIT_CAMERA:           1 << 11,
  DOF_CAMERA:             1 << 12,

  // ---- LIFECYCLE ----
  SPAWN_PENDING:          1 << 13,
  DESTROY_PENDING:        1 << 14,
  PERSISTENT:             1 << 15,
  TRANSIENT:              1 << 16,
  POOLED:                 1 << 17,

  // ---- STREAMING ----
  STREAMING_CHUNK:        1 << 18,
  STREAMING_RESIDENT:     1 << 19,
  STREAMING_EVICT:        1 << 20,
  LOD_GROUP:              1 << 21,
  IMPOSTOR:               1 << 22,

  // ---- BIOME ----
  DESERT:                 1 << 23,
  SNOW:                   1 << 24,
  SEA:                    1 << 25,
  FOREST:                 1 << 26,
  CANYON:                 1 << 27,
  COASTAL:                1 << 28,
  WETLAND:                1 << 29,
  TUNDRA:                 1 << 30,
  VOLCANIC:               1 << 31,   // last bit of the secondary mask
});

/* ------------------------------------------------------------------ */
/* 3. FRAME-DIRTY TAG BITS                                            */
/* ------------------------------------------------------------------ */

export const FDIRTY = Object.freeze({
  NONE:                   0,
  NEEDS_UPDATE:           1 << 0,
  NEEDS_RECOMPUTE:        1 << 1,
  NEEDS_REUPLOAD:         1 << 2,
  NEEDS_RESORT:           1 << 3,
  FRUSTUM_CULLED:         1 << 4,
  IN_VIEW:                1 << 5,
  TRANSITIONING:          1 << 6,
  TRANSITION_ZONE:        1 << 7,
  PORTAL_CONNECTED:       1 << 8,
  OCCLUDED:               1 << 9,
  HIDDEN:                 1 << 10,
  DISABLED:               1 << 11,
  DEBUG_FREEZE:           1 << 12,
});

/* ------------------------------------------------------------------ */
/* 4. SoA TAG COMPONENTS (bitECS interop)                             */
/* ------------------------------------------------------------------ */

/**
 * Light tag component — mirrors TAG.LIGHT in SoA form.
 */
export const LightTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  kind:   new Uint8Array(MAX_ENTITIES),   // LIGHT_KIND (mirrored)
  flags:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Camera tag component.
 */
export const CameraTagMarker = {
  active:  new Uint8Array(MAX_ENTITIES),
  isMain:  new Uint8Array(MAX_ENTITIES),
  isOrbit: new Uint8Array(MAX_ENTITIES),
  hasDOF:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Shadow caster tag component.
 */
export const ShadowCasterTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  castStrength: new Float32Array(MAX_ENTITIES),
};

/**
 * Shadow receiver tag component.
 */
export const ShadowReceiverTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  quality: new Uint8Array(MAX_ENTITIES),
};

/**
 * GI probe tag component.
 */
export const GIProbeTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  isHero: new Uint8Array(MAX_ENTITIES),
  isDirty:new Uint8Array(MAX_ENTITIES),
  isAsync:new Uint8Array(MAX_ENTITIES),
};

/**
 * GI volume tag component.
 */
export const GIVolumeTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  kind:   new Uint8Array(MAX_ENTITIES),   // 0=indoor 1=outdoor 2=transition
};

/**
 * GI portal tag component.
 */
export const GIPortalTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  kind:   new Uint8Array(MAX_ENTITIES),
};

/**
 * AO volume tag component.
 */
export const AOVolumeTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  isIndoor: new Uint8Array(MAX_ENTITIES),
  isOutdoor:new Uint8Array(MAX_ENTITIES),
  isCel:   new Uint8Array(MAX_ENTITIES),
  isInk:   new Uint8Array(MAX_ENTITIES),
  isDirty: new Uint8Array(MAX_ENTITIES),
};

/**
 * Scene state tag component.
 */
export const SceneStateTagMarker = {
  active: new Uint8Array(MAX_ENTITIES),
  visible:new Uint8Array(MAX_ENTITIES),
  isStatic:new Uint8Array(MAX_ENTITIES),
  isDynamic:new Uint8Array(MAX_ENTITIES),
  isDebug:new Uint8Array(MAX_ENTITIES),
};

/**
 * Lifecycle tag component.
 */
export const LifecycleTagMarker = {
  spawnPending:  new Uint8Array(MAX_ENTITIES),
  destroyPending:new Uint8Array(MAX_ENTITIES),
  persistent:    new Uint8Array(MAX_ENTITIES),
  transient:     new Uint8Array(MAX_ENTITIES),
  pooled:        new Uint8Array(MAX_ENTITIES),
};

/**
 * Streaming tag component.
 */
export const StreamingTagMarker = {
  chunk:    new Uint8Array(MAX_ENTITIES),
  resident: new Uint8Array(MAX_ENTITIES),
  evict:    new Uint8Array(MAX_ENTITIES),
  lodGroup: new Uint8Array(MAX_ENTITIES),
  impostor: new Uint8Array(MAX_ENTITIES),
};

/**
 * Biome tag component.
 */
export const BiomeTagMarker = {
  desert:  new Uint8Array(MAX_ENTITIES),
  snow:    new Uint8Array(MAX_ENTITIES),
  sea:     new Uint8Array(MAX_ENTITIES),
  forest:  new Uint8Array(MAX_ENTITIES),
  canyon:  new Uint8Array(MAX_ENTITIES),
  coastal: new Uint8Array(MAX_ENTITIES),
  wetland: new Uint8Array(MAX_ENTITIES),
  tundra:  new Uint8Array(MAX_ENTITIES),
  volcanic:new Uint8Array(MAX_ENTITIES),
};

/**
 * Interior / exterior tag component.
 */
export const EnvironmentTagMarker = {
  interior:  new Uint8Array(MAX_ENTITIES),
  exterior:  new Uint8Array(MAX_ENTITIES),
  transition:new Uint8Array(MAX_ENTITIES),
  portalConnected: new Uint8Array(MAX_ENTITIES),
  occluded:  new Uint8Array(MAX_ENTITIES),
};

/**
 * Gameplay entity tag component (player/enemy/projectile/etc.).
 */
export const GameplayTagMarker = {
  player:     new Uint8Array(MAX_ENTITIES),
  enemy:      new Uint8Array(MAX_ENTITIES),
  projectile: new Uint8Array(MAX_ENTITIES),
  interactive:new Uint8Array(MAX_ENTITIES),
};

/* ------------------------------------------------------------------ */
/* 5. SoA COMPONENT BUNDLE FOR bitECS createWorld                     */
/* ------------------------------------------------------------------ */

export const TAG_COMPONENTS = Object.freeze({
  LightTagMarker,
  CameraTagMarker,
  ShadowCasterTagMarker,
  ShadowReceiverTagMarker,
  GIProbeTagMarker,
  GIVolumeTagMarker,
  GIPortalTagMarker,
  AOVolumeTagMarker,
  SceneStateTagMarker,
  LifecycleTagMarker,
  StreamingTagMarker,
  BiomeTagMarker,
  EnvironmentTagMarker,
  GameplayTagMarker,
});

/* ------------------------------------------------------------------ */
/* 6. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

function _isValidEntity(eid) {
  return typeof eid === 'number' && eid >= 0 && eid < MAX_ENTITIES;
}

/* ------------------------------------------------------------------ */
/* 7. PRIMARY TAG OPERATIONS                                          */
/* ------------------------------------------------------------------ */

/**
 * Tags an entity with a primary tag. Returns the resulting bitmask.
 */
export function tagEntity(eid, tag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof tag !== 'number' || tag === 0) return EntityTag[eid];
  EntityTag[eid] |= (tag >>> 0);
  return EntityTag[eid];
}

/**
 * Removes a primary tag from an entity. Returns the resulting bitmask.
 */
export function untagEntity(eid, tag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof tag !== 'number' || tag === 0) return EntityTag[eid];
  EntityTag[eid] &= ~(tag >>> 0);
  return EntityTag[eid];
}

/**
 * Returns true if an entity has ALL bits in `tag`.
 */
export function hasTag(eid, tag) {
  if (!_isValidEntity(eid)) return false;
  if (typeof tag !== 'number') return false;
  return (EntityTag[eid] & (tag >>> 0)) === (tag >>> 0);
}

/**
 * Returns true if an entity has ANY bit in `tag`.
 */
export function hasAnyTag(eid, tag) {
  if (!_isValidEntity(eid)) return false;
  if (typeof tag !== 'number') return false;
  return (EntityTag[eid] & (tag >>> 0)) !== 0;
}

/**
 * Returns the raw primary tag bitmask of an entity.
 */
export function getTags(eid) {
  if (!_isValidEntity(eid)) return 0;
  return EntityTag[eid];
}

/**
 * Overwrites the primary tag bitmask of an entity.
 */
export function setTags(eid, mask) {
  if (!_isValidEntity(eid)) return 0;
  EntityTag[eid] = mask >>> 0;
  return EntityTag[eid];
}

/**
 * Clears every primary tag from an entity.
 */
export function clearTags(eid) {
  if (!_isValidEntity(eid)) return 0;
  EntityTag[eid] = 0;
  return 0;
}

/* ------------------------------------------------------------------ */
/* 8. SECONDARY TAG OPERATIONS                                        */
/* ------------------------------------------------------------------ */

export function tagEntity2(eid, tag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof tag !== 'number' || tag === 0) return EntityTag2[eid];
  EntityTag2[eid] |= (tag >>> 0);
  return EntityTag2[eid];
}

export function untagEntity2(eid, tag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof tag !== 'number' || tag === 0) return EntityTag2[eid];
  EntityTag2[eid] &= ~(tag >>> 0);
  return EntityTag2[eid];
}

export function hasTag2(eid, tag) {
  if (!_isValidEntity(eid)) return false;
  if (typeof tag !== 'number') return false;
  return (EntityTag2[eid] & (tag >>> 0)) === (tag >>> 0);
}

export function hasAnyTag2(eid, tag) {
  if (!_isValidEntity(eid)) return false;
  if (typeof tag !== 'number') return false;
  return (EntityTag2[eid] & (tag >>> 0)) !== 0;
}

export function getTags2(eid) {
  if (!_isValidEntity(eid)) return 0;
  return EntityTag2[eid];
}

export function setTags2(eid, mask) {
  if (!_isValidEntity(eid)) return 0;
  EntityTag2[eid] = mask >>> 0;
  return EntityTag2[eid];
}

export function clearTags2(eid) {
  if (!_isValidEntity(eid)) return 0;
  EntityTag2[eid] = 0;
  return 0;
}

/* ------------------------------------------------------------------ */
/* 9. FRAME-DIRTY TAG OPERATIONS                                      */
/* ------------------------------------------------------------------ */

export function markFrameDirty(eid, flag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof flag !== 'number' || flag === 0) return FrameDirtyTag[eid];
  FrameDirtyTag[eid] |= (flag >>> 0);
  return FrameDirtyTag[eid];
}

export function clearFrameDirty(eid, flag) {
  if (!_isValidEntity(eid)) return 0;
  if (typeof flag !== 'number' || flag === 0) return FrameDirtyTag[eid];
  FrameDirtyTag[eid] &= ~(flag >>> 0);
  return FrameDirtyTag[eid];
}

export function isFrameDirty(eid, flag) {
  if (!_isValidEntity(eid)) return false;
  if (typeof flag !== 'number') return false;
  return (FrameDirtyTag[eid] & (flag >>> 0)) !== 0;
}

/**
 * Clears the entire frame-dirty mask. Called once per frame.
 */
export function resetFrameDirty() {
  FrameDirtyTag.fill(0);
}

/* ------------------------------------------------------------------ */
/* 10. QUERY HELPERS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Iterates every entity whose primary tag mask contains ALL bits in
 * `tagMask`. Allocation-free — callers get the entity id via callback.
 *
 *   forEachTagged(TAG.LIGHT | TAG.ACTIVE, (eid) => { ... });
 */
export function forEachTagged(tagMask, fn, ctx) {
  if (typeof tagMask !== 'number' || typeof fn !== 'function') return 0;
  let count = 0;
  for (let i = 0; i < MAX_ENTITIES; i++) {
    if ((EntityTag[i] & tagMask) === tagMask) {
      fn.call(ctx, i);
      count++;
    }
  }
  return count;
}

/**
 * Returns an array of entity ids matching ALL bits in `tagMask`.
 * Allocates a fresh array — do not use on the hot path.
 */
export function queryTagged(tagMask) {
  if (typeof tagMask !== 'number') return [];
  const out = [];
  for (let i = 0; i < MAX_ENTITIES; i++) {
    if ((EntityTag[i] & tagMask) === tagMask) out.push(i);
  }
  return out;
}

/**
 * Returns an array of entity ids matching ANY bit in `tagMask`.
 */
export function queryTaggedAny(tagMask) {
  if (typeof tagMask !== 'number') return [];
  const out = [];
  for (let i = 0; i < MAX_ENTITIES; i++) {
    if ((EntityTag[i] & tagMask) !== 0) out.push(i);
  }
  return out;
}

/**
 * Fills `outArray` with entity ids matching ALL bits in `tagMask`.
 * Returns the number written. Zero-alloc.
 */
export function collectTagged(tagMask, outArray) {
  if (typeof tagMask !== 'number' || !outArray) return 0;
  let write = 0;
  const cap = outArray.length;
  for (let i = 0; i < MAX_ENTITIES && write < cap; i++) {
    if ((EntityTag[i] & tagMask) === tagMask) {
      outArray[write++] = i;
    }
  }
  return write;
}

/**
 * Counts entities matching ALL bits in `tagMask`.
 */
export function countTagged(tagMask) {
  if (typeof tagMask !== 'number') return 0;
  let count = 0;
  for (let i = 0; i < MAX_ENTITIES; i++) {
    if ((EntityTag[i] & tagMask) === tagMask) count++;
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* 11. TAG → SoA MARKER SYNC                                          */
/* ------------------------------------------------------------------ */

/**
 * Applies the primary bitmask to the SoA marker components (so bitECS
 * `query` works on tag components). Call this from `syncTagsToSoA(eid)`.
 */
export function syncTagsToSoA(eid) {
  if (!_isValidEntity(eid)) return false;
  const t1 = EntityTag[eid];
  const t2 = EntityTag2[eid];

  // Light
  LightTagMarker.active[eid] = (t1 & TAG.LIGHT) !== 0 ? 1 : 0;

  // Camera
  CameraTagMarker.active[eid]  = (t1 & TAG.CAMERA) !== 0 ? 1 : 0;
  CameraTagMarker.isMain[eid]  = (t1 & TAG.ACTIVE_CAMERA) !== 0 ? 1 : 0;
  CameraTagMarker.isOrbit[eid] = (t2 & TAG2.ORBIT_CAMERA) !== 0 ? 1 : 0;
  CameraTagMarker.hasDOF[eid]  = (t2 & TAG2.DOF_CAMERA) !== 0 ? 1 : 0;

  // Shadow
  ShadowCasterTagMarker.active[eid]   = (t1 & TAG.SHADOW_CASTER) !== 0 ? 1 : 0;
  ShadowReceiverTagMarker.active[eid] = (t1 & TAG.SHADOW_RECEIVER) !== 0 ? 1 : 0;

  // GI
  GIProbeTagMarker.active[eid]   = (t1 & TAG.GI_PROBE) !== 0 ? 1 : 0;
  GIProbeTagMarker.isHero[eid]   = (t1 & TAG.GI_HERO_PROBE) !== 0 ? 1 : 0;
  GIProbeTagMarker.isDirty[eid]  = (t2 & TAG2.GI_DIRTY) !== 0 ? 1 : 0;
  GIProbeTagMarker.isAsync[eid]  = (t2 & TAG2.GI_ASYNC) !== 0 ? 1 : 0;
  GIVolumeTagMarker.active[eid]  = (t1 & TAG.GI_VOLUME) !== 0 ? 1 : 0;
  GIPortalTagMarker.active[eid]  = (t1 & TAG.GI_PORTAL) !== 0 ? 1 : 0;

  // AO
  AOVolumeTagMarker.active[eid]    = (t1 & TAG.AO_VOLUME) !== 0 ? 1 : 0;
  AOVolumeTagMarker.isIndoor[eid]  = (t1 & TAG.AO_INDOOR) !== 0 ? 1 : 0;
  AOVolumeTagMarker.isOutdoor[eid] = (t1 & TAG.AO_OUTDOOR) !== 0 ? 1 : 0;
  AOVolumeTagMarker.isCel[eid]     = (t1 & TAG.AO_CEL_BAND) !== 0 ? 1 : 0;
  AOVolumeTagMarker.isInk[eid]     = (t2 & TAG2.AO_INK_OUTLINE) !== 0 ? 1 : 0;
  AOVolumeTagMarker.isDirty[eid]   = (t2 & TAG2.AO_DIRTY) !== 0 ? 1 : 0;

  // Scene state
  SceneStateTagMarker.active[eid]    = (t1 & TAG.ACTIVE) !== 0 ? 1 : 0;
  SceneStateTagMarker.visible[eid]   = (t1 & TAG.VISIBLE) !== 0 ? 1 : 0;
  SceneStateTagMarker.isStatic[eid]  = (t1 & TAG.STATIC) !== 0 ? 1 : 0;
  SceneStateTagMarker.isDynamic[eid] = (t1 & TAG.DYNAMIC) !== 0 ? 1 : 0;
  SceneStateTagMarker.isDebug[eid]   = (t1 & TAG.DEBUG) !== 0 ? 1 : 0;

  // Lifecycle
  LifecycleTagMarker.spawnPending[eid]   = (t2 & TAG2.SPAWN_PENDING) !== 0 ? 1 : 0;
  LifecycleTagMarker.destroyPending[eid] = (t2 & TAG2.DESTROY_PENDING) !== 0 ? 1 : 0;
  LifecycleTagMarker.persistent[eid]     = (t2 & TAG2.PERSISTENT) !== 0 ? 1 : 0;
  LifecycleTagMarker.transient[eid]      = (t2 & TAG2.TRANSIENT) !== 0 ? 1 : 0;
  LifecycleTagMarker.pooled[eid]         = (t2 & TAG2.POOLED) !== 0 ? 1 : 0;

  // Streaming
  StreamingTagMarker.chunk[eid]    = (t2 & TAG2.STREAMING_CHUNK) !== 0 ? 1 : 0;
  StreamingTagMarker.resident[eid] = (t2 & TAG2.STREAMING_RESIDENT) !== 0 ? 1 : 0;
  StreamingTagMarker.evict[eid]    = (t2 & TAG2.STREAMING_EVICT) !== 0 ? 1 : 0;
  StreamingTagMarker.lodGroup[eid] = (t2 & TAG2.LOD_GROUP) !== 0 ? 1 : 0;
  StreamingTagMarker.impostor[eid] = (t2 & TAG2.IMPOSTOR) !== 0 ? 1 : 0;

  // Biome
  BiomeTagMarker.desert[eid]   = (t2 & TAG2.DESERT)   !== 0 ? 1 : 0;
  BiomeTagMarker.snow[eid]     = (t2 & TAG2.SNOW)     !== 0 ? 1 : 0;
  BiomeTagMarker.sea[eid]      = (t2 & TAG2.SEA)      !== 0 ? 1 : 0;
  BiomeTagMarker.forest[eid]   = (t2 & TAG2.FOREST)   !== 0 ? 1 : 0;
  BiomeTagMarker.canyon[eid]   = (t2 & TAG2.CANYON)   !== 0 ? 1 : 0;
  BiomeTagMarker.coastal[eid]  = (t2 & TAG2.COASTAL)  !== 0 ? 1 : 0;
  BiomeTagMarker.wetland[eid]  = (t2 & TAG2.WETLAND)  !== 0 ? 1 : 0;
  BiomeTagMarker.tundra[eid]   = (t2 & TAG2.TUNDRA)   !== 0 ? 1 : 0;
  BiomeTagMarker.volcanic[eid] = (t2 & TAG2.VOLCANIC) !== 0 ? 1 : 0;

  // Environment
  EnvironmentTagMarker.interior[eid]        = (t2 & TAG2.INDOOR_LIGHT) !== 0 ? 1 : 0;
  EnvironmentTagMarker.exterior[eid]        = (t2 & TAG2.OUTDOOR_LIGHT) !== 0 ? 1 : 0;
  EnvironmentTagMarker.portalConnected[eid] = isFrameDirty(eid, FDIRTY.PORTAL_CONNECTED) ? 1 : 0;
  EnvironmentTagMarker.occluded[eid]        = isFrameDirty(eid, FDIRTY.OCCLUDED) ? 1 : 0;

  return true;
}

/**
 * Synchronizes the SoA marker components back to the primary bitmask.
 * Used when a system writes directly to a marker component.
 */
export function syncSoAToTags(eid) {
  if (!_isValidEntity(eid)) return false;
  let t1 = 0;
  let t2 = 0;

  if (LightTagMarker.active[eid])       t1 |= TAG.LIGHT;
  if (CameraTagMarker.active[eid])      t1 |= TAG.CAMERA;
  if (CameraTagMarker.isMain[eid])      t1 |= TAG.ACTIVE_CAMERA;
  if (CameraTagMarker.isOrbit[eid])     t2 |= TAG2.ORBIT_CAMERA;
  if (CameraTagMarker.hasDOF[eid])      t2 |= TAG2.DOF_CAMERA;
  if (ShadowCasterTagMarker.active[eid])t1 |= TAG.SHADOW_CASTER;
  if (ShadowReceiverTagMarker.active[eid]) t1 |= TAG.SHADOW_RECEIVER;
  if (GIProbeTagMarker.active[eid])     t1 |= TAG.GI_PROBE;
  if (GIProbeTagMarker.isHero[eid])     t1 |= TAG.GI_HERO_PROBE;
  if (GIProbeTagMarker.isDirty[eid])    t2 |= TAG2.GI_DIRTY;
  if (GIProbeTagMarker.isAsync[eid])    t2 |= TAG2.GI_ASYNC;
  if (GIVolumeTagMarker.active[eid])    t1 |= TAG.GI_VOLUME;
  if (GIPortalTagMarker.active[eid])    t1 |= TAG.GI_PORTAL;
  if (AOVolumeTagMarker.active[eid])    t1 |= TAG.AO_VOLUME;
  if (AOVolumeTagMarker.isIndoor[eid])  t1 |= TAG.AO_INDOOR;
  if (AOVolumeTagMarker.isOutdoor[eid]) t1 |= TAG.AO_OUTDOOR;
  if (AOVolumeTagMarker.isCel[eid])     t1 |= TAG.AO_CEL_BAND;
  if (AOVolumeTagMarker.isInk[eid])     t2 |= TAG2.AO_INK_OUTLINE;
  if (AOVolumeTagMarker.isDirty[eid])   t2 |= TAG2.AO_DIRTY;
  if (SceneStateTagMarker.active[eid])  t1 |= TAG.ACTIVE;
  if (SceneStateTagMarker.visible[eid]) t1 |= TAG.VISIBLE;
  if (SceneStateTagMarker.isStatic[eid])t1 |= TAG.STATIC;
  if (SceneStateTagMarker.isDynamic[eid])t1 |= TAG.DYNAMIC;
  if (SceneStateTagMarker.isDebug[eid]) t1 |= TAG.DEBUG;
  if (LifecycleTagMarker.spawnPending[eid])   t2 |= TAG2.SPAWN_PENDING;
  if (LifecycleTagMarker.destroyPending[eid]) t2 |= TAG2.DESTROY_PENDING;
  if (LifecycleTagMarker.persistent[eid])     t2 |= TAG2.PERSISTENT;
  if (LifecycleTagMarker.transient[eid])      t2 |= TAG2.TRANSIENT;
  if (LifecycleTagMarker.pooled[eid])         t2 |= TAG2.POOLED;
  if (StreamingTagMarker.chunk[eid])    t2 |= TAG2.STREAMING_CHUNK;
  if (StreamingTagMarker.resident[eid]) t2 |= TAG2.STREAMING_RESIDENT;
  if (StreamingTagMarker.evict[eid])    t2 |= TAG2.STREAMING_EVICT;
  if (StreamingTagMarker.lodGroup[eid]) t2 |= TAG2.LOD_GROUP;
  if (StreamingTagMarker.impostor[eid]) t2 |= TAG2.IMPOSTOR;
  if (BiomeTagMarker.desert[eid])   t2 |= TAG2.DESERT;
  if (BiomeTagMarker.snow[eid])     t2 |= TAG2.SNOW;
  if (BiomeTagMarker.sea[eid])      t2 |= TAG2.SEA;
  if (BiomeTagMarker.forest[eid])   t2 |= TAG2.FOREST;
  if (BiomeTagMarker.canyon[eid])   t2 |= TAG2.CANYON;
  if (BiomeTagMarker.coastal[eid])  t2 |= TAG2.COASTAL;
  if (BiomeTagMarker.wetland[eid])  t2 |= TAG2.WETLAND;
  if (BiomeTagMarker.tundra[eid])   t2 |= TAG2.TUNDRA;
  if (BiomeTagMarker.volcanic[eid]) t2 |= TAG2.VOLCANIC;

  EntityTag[eid] = t1;
  EntityTag2[eid] = t2;
  return true;
}

/* ------------------------------------------------------------------ */
/* 12. HIGH-LEVEL CONVENIENCE                                         */
/* ------------------------------------------------------------------ */

/**
 * Tags an entity as a light of the given type.
 */
export function tagAsLight(eid, lightType) {
  if (!_isValidEntity(eid)) return false;
  let mask = TAG.LIGHT;
  switch (lightType) {
    case 0: mask |= TAG.AMBIENT; break;
    case 1: mask |= TAG.HEMISPHERE; break;
    case 2: mask |= TAG.DIRECTIONAL; break;
    case 3: mask |= TAG.POINT; break;
    case 4: mask |= TAG.SPOT; break;
    case 5: mask |= TAG.RECT_AREA; break;
    default: break;
  }
  tagEntity(eid, mask);
  syncTagsToSoA(eid);
  return true;
}

/**
 * Tags an entity as a shadow caster and/or receiver.
 */
export function tagAsShadowParticipant(eid, isCaster, isReceiver) {
  if (!_isValidEntity(eid)) return false;
  if (isCaster)   tagEntity(eid, TAG.SHADOW_CASTER);
  if (isReceiver) tagEntity(eid, TAG.SHADOW_RECEIVER);
  syncTagsToSoA(eid);
  return true;
}

/**
 * Tags an entity as a GI probe.
 */
export function tagAsGIProbe(eid, isHero) {
  if (!_isValidEntity(eid)) return false;
  tagEntity(eid, TAG.GI_PROBE);
  if (isHero) tagEntity(eid, TAG.GI_HERO_PROBE);
  syncTagsToSoA(eid);
  return true;
}

/**
 * Tags an entity as an AO volume.
 */
export function tagAsAOVolume(eid, isIndoor, isOutdoor, isCel, isInk) {
  if (!_isValidEntity(eid)) return false;
  tagEntity(eid, TAG.AO_VOLUME);
  if (isIndoor)  tagEntity(eid, TAG.AO_INDOOR);
  if (isOutdoor) tagEntity(eid, TAG.AO_OUTDOOR);
  if (isCel)     tagEntity(eid, TAG.AO_CEL_BAND);
  if (isInk)     tagEntity2(eid, TAG2.AO_INK_OUTLINE);
  syncTagsToSoA(eid);
  return true;
}

/**
 * Tags an entity with a biome.
 */
export function tagAsBiome(eid, biomeId) {
  if (!_isValidEntity(eid)) return false;
  switch (biomeId) {
    case 0: tagEntity2(eid, TAG2.DESERT);   break;
    case 1: tagEntity2(eid, TAG2.SNOW);     break;
    case 2: tagEntity2(eid, TAG2.SEA);      break;
    case 3: tagEntity2(eid, TAG2.FOREST);   break;
    case 4: tagEntity2(eid, TAG2.CANYON);   break;
    case 5: tagEntity2(eid, TAG2.COASTAL);  break;
    case 6: tagEntity2(eid, TAG2.WETLAND);  break;
    case 7: tagEntity2(eid, TAG2.TUNDRA);   break;
    case 8: tagEntity2(eid, TAG2.VOLCANIC); break;
    default: return false;
  }
  syncTagsToSoA(eid);
  return true;
}

/* ------------------------------------------------------------------ */
/* 13. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

let _tagsFrame = 0;

/**
 * Advances the tag system frame counter. Called once per frame by the
 * engine loop before running any tag consumers.
 */
export function tickTags(frameNumber) {
  if (typeof frameNumber === 'number') _tagsFrame = frameNumber;
  else _tagsFrame++;
}

/**
 * Resets the frame-scoped dirty mask. Called once per frame AFTER every
 * consumer has had a chance to read it.
 */
export function endFrameTags() {
  resetFrameDirty();
}

/* ------------------------------------------------------------------ */
/* 14. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getTagStats() {
  let lightCount = 0;
  let cameraCount = 0;
  let shadowCasterCount = 0;
  let shadowReceiverCount = 0;
  let giProbeCount = 0;
  let aoVolumeCount = 0;
  let activeCount = 0;
  let dirtyCount = 0;
  let biomeCount = 0;

  for (let i = 0; i < MAX_ENTITIES; i++) {
    const t1 = EntityTag[i];
    const t2 = EntityTag2[i];
    if ((t1 & TAG.LIGHT) !== 0)           lightCount++;
    if ((t1 & TAG.CAMERA) !== 0)          cameraCount++;
    if ((t1 & TAG.SHADOW_CASTER) !== 0)   shadowCasterCount++;
    if ((t1 & TAG.SHADOW_RECEIVER) !== 0) shadowReceiverCount++;
    if ((t1 & TAG.GI_PROBE) !== 0)        giProbeCount++;
    if ((t1 & TAG.AO_VOLUME) !== 0)       aoVolumeCount++;
    if ((t1 & TAG.ACTIVE) !== 0)          activeCount++;
    if ((t1 & TAG.DIRTY) !== 0)           dirtyCount++;
    if ((t2 & (TAG2.DESERT | TAG2.SNOW | TAG2.SEA | TAG2.FOREST | TAG2.CANYON)) !== 0) biomeCount++;
  }

  return {
    frame:               _tagsFrame,
    maxEntities:         MAX_ENTITIES,
    lightCount,
    cameraCount,
    shadowCasterCount,
    shadowReceiverCount,
    giProbeCount,
    aoVolumeCount,
    activeCount,
    dirtyCount,
    biomeCount,
    perfTier:            PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 15. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Clears every tag on every entity. Use for full teardown / restart.
 */
export function resetAllTags() {
  EntityTag.fill(0);
  EntityTag2.fill(0);
  FrameDirtyTag.fill(0);
  for (let i = 0; i < MAX_ENTITIES; i++) {
    LightTagMarker.active[i] = 0;
    CameraTagMarker.active[i] = 0;
    ShadowCasterTagMarker.active[i] = 0;
    ShadowReceiverTagMarker.active[i] = 0;
    GIProbeTagMarker.active[i] = 0;
    GIVolumeTagMarker.active[i] = 0;
    GIPortalTagMarker.active[i] = 0;
    AOVolumeTagMarker.active[i] = 0;
    SceneStateTagMarker.active[i] = 0;
    LifecycleTagMarker.spawnPending[i] = 0;
    LifecycleTagMarker.destroyPending[i] = 0;
    StreamingTagMarker.chunk[i] = 0;
    BiomeTagMarker.desert[i] = 0;
    EnvironmentTagMarker.interior[i] = 0;
    GameplayTagMarker.player[i] = 0;
  }
  _tagsFrame = 0;
}

/* ------------------------------------------------------------------ */
/* 16. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * Registers every tag component in the runtime component registry so
 * downstream systems can discover them by name.
 */
export function registerTagComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'LightTagMarker',        component: LightTagMarker,        category: 1,  subsystem: 2,  dependencies: [] },
    { name: 'CameraTagMarker',       component: CameraTagMarker,       category: 5,  subsystem: 6,  dependencies: [] },
    { name: 'ShadowCasterTagMarker', component: ShadowCasterTagMarker, category: 2,  subsystem: 3,  dependencies: [] },
    { name: 'ShadowReceiverTagMarker', component: ShadowReceiverTagMarker, category: 2, subsystem: 3, dependencies: [] },
    { name: 'GIProbeTagMarker',      component: GIProbeTagMarker,      category: 3,  subsystem: 4,  dependencies: [] },
    { name: 'GIVolumeTagMarker',     component: GIVolumeTagMarker,     category: 3,  subsystem: 4,  dependencies: [] },
    { name: 'GIPortalTagMarker',     component: GIPortalTagMarker,     category: 3,  subsystem: 4,  dependencies: [] },
    { name: 'AOVolumeTagMarker',     component: AOVolumeTagMarker,     category: 4,  subsystem: 5,  dependencies: [] },
    { name: 'SceneStateTagMarker',   component: SceneStateTagMarker,   category: 6,  subsystem: 1,  dependencies: [] },
    { name: 'LifecycleTagMarker',    component: LifecycleTagMarker,    category: 6,  subsystem: 1,  dependencies: [] },
    { name: 'StreamingTagMarker',    component: StreamingTagMarker,    category: 7,  subsystem: 10, dependencies: [] },
    { name: 'BiomeTagMarker',        component: BiomeTagMarker,        category: 10, subsystem: 7,  dependencies: [] },
    { name: 'EnvironmentTagMarker',  component: EnvironmentTagMarker,  category: 11, subsystem: 8,  dependencies: [] },
    { name: 'GameplayTagMarker',     component: GameplayTagMarker,     category: 6,  subsystem: 1,  dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 17. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Bitmasks
  EntityTag,
  EntityTag2,
  FrameDirtyTag,

  // Enums
  TAG,
  TAG2,
  FDIRTY,

  // SoA marker components
  LightTagMarker,
  CameraTagMarker,
  ShadowCasterTagMarker,
  ShadowReceiverTagMarker,
  GIProbeTagMarker,
  GIVolumeTagMarker,
  GIPortalTagMarker,
  AOVolumeTagMarker,
  SceneStateTagMarker,
  LifecycleTagMarker,
  StreamingTagMarker,
  BiomeTagMarker,
  EnvironmentTagMarker,
  GameplayTagMarker,
  TAG_COMPONENTS,

  // Primary tag operations
  tagEntity,
  untagEntity,
  hasTag,
  hasAnyTag,
  getTags,
  setTags,
  clearTags,

  // Secondary tag operations
  tagEntity2,
  untagEntity2,
  hasTag2,
  hasAnyTag2,
  getTags2,
  setTags2,
  clearTags2,

  // Frame-dirty operations
  markFrameDirty,
  clearFrameDirty,
  isFrameDirty,
  resetFrameDirty,

  // Queries
  forEachTagged,
  queryTagged,
  queryTaggedAny,
  collectTagged,
  countTagged,

  // Sync
  syncTagsToSoA,
  syncSoAToTags,

  // Convenience
  tagAsLight,
  tagAsShadowParticipant,
  tagAsGIProbe,
  tagAsAOVolume,
  tagAsBiome,

  // Lifecycle
  tickTags,
  endFrameTags,

  // Diagnostics
  getTagStats,

  // Reset
  resetAllTags,

  // Registration
  registerTagComponents,
};

export default _defaultExport;