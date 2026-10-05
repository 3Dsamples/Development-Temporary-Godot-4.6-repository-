// File : 024
// name : src/ecs/024_scn_Serialization.js
// description : Entity serialization / deserialization module for the scene
//               ECS world of the anime lighting stack on Android mobile.
//               Captures the full live state of every entity — SoA component
//               field values, tag bitmasks, relation edges, lifetime records,
//               blueprint instance bindings — into a compact, versioned,
//               checksummed payload, and reconstructs it deterministically.
//
//               Two payload forms:
//                 • BINARY  — fixed-capacity Uint8Array buffer with a
//                              versioned header + numeric type ids on the
//                              wire. Smallest and fastest. Used for worker
//                              transfers, save files, and regression
//                              captures.
//                 • JSON    — human-readable object graph produced from the
//                              same internal record model. Used for debug
//                              dumps and diffing.
//
//               Design:
//                 • Schema version header (SCHEMA_VERSION) so old payloads
//                   can be recognized and migrated.
//                 • Numeric type ids from 013_scn_ComponentTypes.js keep
//                   component names off the wire — one Uint16 per component
//                   reference.
//                 • Fixed-capacity writer / reader with byte cursors. No
//                   dynamic array growth, no per-field allocation.
//                 • Streaming record layout — a writer appends records; a
//                   reader decodes them in the same order.
//                 • Checksum (FNV-1a 32-bit) at the tail so truncation and
//                   corruption are detected deterministically.
//                 • Round-trip guarantees — serialize(world) followed by
//                   deserialize(payload) into a fresh world restores every
//                   entity id, every tag bitmask, every relation edge, and
//                   every lifetime record bit-for-bit.
//                 • Blueprint-aware — a payload can optionally carry a
//                   blueprint instance table so a whole scene region can be
//                   restored with the same member ordering.
//                 • Zero allocations on the hot path — the writer and reader
//                   operate on pre-allocated typed arrays; iteration is
//                   index-based, not iterator-based.
//
//               Integration:
//                 • 002_lgt_LightComponents.js      — component SoA arrays
//                 • 003_lgt_ShadowComponents.js     — component SoA arrays
//                 • 004_lgt_GIComponents.js         — component SoA arrays
//                 • 005_lgt_AOComponents.js         — component SoA arrays
//                 • 010_scn_ECSWorld.js             — world handle
//                 • 011_scn_BiteCSAdapter.js        — ECS façade
//                 • 012_scn_ComponentRegistry.js    — component names
//                 • 013_scn_ComponentTypes.js       — numeric type ids
//                 • 014_scn_Tags.js                 — tag bitmasks
//                 • 015_scn_Relations.js            — relation graph
//                 • 016_scn_EntityPool.js           — pool binding
//                 • 017_scn_EntityLifetime.js       — lifetime records
//                 • 022_scn_SpawnPrefabs.js         — prefab ids
//                 • 023_scn_Blueprints.js           — blueprint instances
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that the entire anime lighting stack can capture
//            and restore its live ECS state deterministically — so a scene
//            can be snapshotted, a worker can be seeded from main-thread
//            state, a regression can be replayed bit-for-bit, and a full
//            save/load round-trip preserves every light, shadow, GI probe,
//            AO volume, and relation edge.
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
  MAX_SHADOW_CASCADES,
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
  MAX_SH_COEFFICIENTS,
  MAX_SH_COEFFICIENTS_HIGH,
  MAX_BOUNCE_PATHS,
  MAX_VOXEL_RESOLUTION,
  MAX_RADIANCE_CACHE,
  MAX_LIGHTFIELD_SAMPLES,
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
  MAX_AO_KERNEL_SAMPLES,
} from './005_lgt_AOComponents.js';

import {
  getECSWorld,
  entityAlive,
  ALL_SCENE_COMPONENTS,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  COMPONENT_TYPE_ID,
  TYPE_ID_INVALID,
  TYPE_ID_MAX,
  getTypeId,
  getTypeName,
  getTypePascalName,
  isValidTypeId,
  getTypeGroup,
} from './013_scn_ComponentTypes.js';

import {
  EntityTag,
  EntityTag2,
  FrameDirtyTag,
} from './014_scn_Tags.js';

import {
  Parent,
  HierarchyState,
  Children,
  References,
  ReverseReferences,
  MAX_CHILDREN_PER_ENTITY,
  MAX_REFERENCES_PER_ENTITY,
  MAX_REVERSE_REFERENCES,
  NULL_ENTITY,
  REF,
  REF_NAME,
} from './015_scn_Relations.js';

import {
  POOL,
  POOL_NAME,
  isInUse,
} from './016_scn_EntityPool.js';

import {
  EntityLifetime,
  LIFETIME_STATE,
  LIFETIME_POOL,
  MAX_ALIVE_FRAMES,
} from './017_scn_EntityLifetime.js';

import {
  PREFAB,
  PREFAB_NAME,
} from './022_scn_SpawnPrefabs.js';

import {
  BLUEPRINT,
  BLUEPRINT_NAME,
  getInstance,
  forEachInstance,
} from './023_scn_Blueprints.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Serialization schema version. Bump on any on-wire format change.
 */
export const SCHEMA_VERSION = 1;

/**
 * Magic number at the start of every binary payload.
 * "ASTL" in ASCII = 0x41 0x53 0x54 0x4C.
 */
export const MAGIC = 0x41_53_54_4C;

/**
 * Payload kinds.
 */
export const PAYLOAD_KIND = Object.freeze({
  WORLD_SNAPSHOT:     0,
  BLUEPRINT_INSTANCE: 1,
  ENTITY_RANGE:       2,
  DIFF:               3,
  COUNT:              4,
});

export const PAYLOAD_KIND_NAME = Object.freeze([
  'world_snapshot',
  'blueprint_instance',
  'entity_range',
  'diff',
]);

/**
 * Payload flags (bitmask).
 */
export const PAYLOAD_FLAG = Object.freeze({
  NONE:             0,
  HAS_TAGS:         1 << 0,
  HAS_RELATIONS:    1 << 1,
  HAS_LIFETIMES:    1 << 2,
  HAS_BLUEPRINTS:   1 << 3,
  HAS_PREFABS:      1 << 4,
  HAS_CHECKSUM:     1 << 5,
  COMPRESSED:       1 << 6,
});

/**
 * Maximum serialized payload size per tier.
 */
export const MAX_PAYLOAD_BYTES =
  PERF_TIER_LOCAL === 'HIGH'   ? (64 * 1024 * 1024) :
  PERF_TIER_LOCAL === 'MEDIUM' ? (32 * 1024 * 1024) :
                                 (16 * 1024 * 1024);

/**
 * Maximum number of entities a single payload may contain.
 */
export const MAX_SERIALIZED_ENTITIES =
  PERF_TIER_LOCAL === 'HIGH'   ? 100000 :
  PERF_TIER_LOCAL === 'MEDIUM' ?  60000 :
                                  30000;

/**
 * Header layout offsets (bytes). Fixed 64-byte header.
 *
 *   [ 0..3]   magic (uint32)
 *   [ 4..5]   schema version (uint16)
 *   [ 6..7]   payload kind (uint16)
 *   [ 8..11]  flags (uint32)
 *   [12..15]  entity count (uint32)
 *   [16..19]  component type count (uint32)
 *   [20..23]  tag section byte length (uint32)
 *   [24..27]  relation section byte length (uint32)
 *   [28..31]  lifetime section byte length (uint32)
 *   [32..35]  blueprint section byte length (uint32)
 *   [36..39]  prefab section byte length (uint32)
 *   [40..43]  reserved (uint32)
 *   [44..47]  reserved (uint32)
 *   [48..51]  checksum (uint32)  — FNV-1a of everything after header
 *   [52..63]  reserved (12 bytes)
 */
export const HEADER_SIZE = 64;

/**
 * Section flags for the per-entity component mask.
 */
const ENTITY_HAS_COMPONENT = 1;

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const SerializationState = {
  totalSerializations:   0,
  totalDeserializations: 0,
  totalRoundTrips:       0,
  totalFailures:         0,
  totalBytesWritten:     0,
  totalBytesRead:        0,
  totalEntitiesWritten:  0,
  totalEntitiesRead:     0,
  peakPayloadBytes:      0,
  lastSerializeMs:       0,
  lastDeserializeMs:     0,
  avgSerializeMs:        0,
  avgDeserializeMs:      0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.serialization', {
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

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 2. BINARY WRITER                                                   */
/* ------------------------------------------------------------------ */

/**
 * Fixed-capacity binary writer over a pre-allocated Uint8Array buffer.
 * All writes advance an internal cursor. No dynamic growth.
 */
export class BinaryWriter {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer   = new Uint8Array(capacity);
    this.view     = new DataView(this.buffer.buffer);
    this.cursor   = 0;
    this.overflow = false;
  }

  reset() {
    this.cursor = 0;
    this.overflow = false;
    return this;
  }

  canWrite(n) {
    return !this.overflow && (this.cursor + n) <= this.capacity;
  }

  _advance(n) {
    if (this.overflow) return;
    if (this.cursor + n > this.capacity) {
      this.overflow = true;
      return;
    }
    this.cursor += n;
  }

  writeU8(v) {
    if (!this.canWrite(1)) { this.overflow = true; return; }
    this.buffer[this.cursor] = (v & 0xFF);
    this._advance(1);
  }

  writeU16(v) {
    if (!this.canWrite(2)) { this.overflow = true; return; }
    this.view.setUint16(this.cursor, v & 0xFFFF, true);
    this._advance(2);
  }

  writeU32(v) {
    if (!this.canWrite(4)) { this.overflow = true; return; }
    this.view.setUint32(this.cursor, v >>> 0, true);
    this._advance(4);
  }

  writeI8(v) {
    if (!this.canWrite(1)) { this.overflow = true; return; }
    this.view.setInt8(this.cursor, v | 0);
    this._advance(1);
  }

  writeI16(v) {
    if (!this.canWrite(2)) { this.overflow = true; return; }
    this.view.setInt16(this.cursor, v | 0, true);
    this._advance(2);
  }

  writeI32(v) {
    if (!this.canWrite(4)) { this.overflow = true; return; }
    this.view.setInt32(this.cursor, v | 0, true);
    this._advance(4);
  }

  writeF32(v) {
    if (!this.canWrite(4)) { this.overflow = true; return; }
    this.view.setFloat32(this.cursor, Number.isFinite(v) ? v : 0, true);
    this._advance(4);
  }

  writeF64(v) {
    if (!this.canWrite(8)) { this.overflow = true; return; }
    this.view.setFloat64(this.cursor, Number.isFinite(v) ? v : 0, true);
    this._advance(8);
  }

  writeBytes(src, length) {
    const n = length !== undefined ? length : src.length;
    if (!this.canWrite(n)) { this.overflow = true; return; }
    this.buffer.set(src.subarray(0, n), this.cursor);
    this._advance(n);
  }

  writeFloat32Array(src, count) {
    const n = count !== undefined ? count : src.length;
    for (let i = 0; i < n; i++) this.writeF32(src[i]);
  }

  writeUint32Array(src, count) {
    const n = count !== undefined ? count : src.length;
    for (let i = 0; i < n; i++) this.writeU32(src[i]);
  }

  writeInt32Array(src, count) {
    const n = count !== undefined ? count : src.length;
    for (let i = 0; i < n; i++) this.writeI32(src[i]);
  }

  writeUint16Array(src, count) {
    const n = count !== undefined ? count : src.length;
    for (let i = 0; i < n; i++) this.writeU16(src[i]);
  }

  writeUint8Array(src, count) {
    const n = count !== undefined ? count : src.length;
    for (let i = 0; i < n; i++) this.writeU8(src[i]);
  }

  seek(offset) {
    if (offset < 0 || offset > this.capacity) return false;
    this.cursor = offset;
    return true;
  }

  tell() { return this.cursor; }

  finalize() {
    return this.buffer.subarray(0, this.cursor);
  }
}

/* ------------------------------------------------------------------ */
/* 3. BINARY READER                                                   */
/* ------------------------------------------------------------------ */

/**
 * Fixed-capacity binary reader over a Uint8Array. Symmetric to
 * BinaryWriter.
 */
export class BinaryReader {
  constructor(buffer) {
    this.buffer   = buffer;
    this.view     = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    this.cursor   = 0;
    this.overflow = false;
  }

  reset() {
    this.cursor = 0;
    this.overflow = false;
    return this;
  }

  canRead(n) {
    return !this.overflow && (this.cursor + n) <= this.buffer.byteLength;
  }

  _advance(n) {
    if (this.overflow) return;
    if (this.cursor + n > this.buffer.byteLength) {
      this.overflow = true;
      return;
    }
    this.cursor += n;
  }

  readU8() {
    if (!this.canRead(1)) { this.overflow = true; return 0; }
    const v = this.buffer[this.cursor];
    this._advance(1);
    return v;
  }

  readU16() {
    if (!this.canRead(2)) { this.overflow = true; return 0; }
    const v = this.view.getUint16(this.cursor, true);
    this._advance(2);
    return v;
  }

  readU32() {
    if (!this.canRead(4)) { this.overflow = true; return 0; }
    const v = this.view.getUint32(this.cursor, true);
    this._advance(4);
    return v;
  }

  readI8() {
    if (!this.canRead(1)) { this.overflow = true; return 0; }
    const v = this.view.getInt8(this.cursor);
    this._advance(1);
    return v;
  }

  readI16() {
    if (!this.canRead(2)) { this.overflow = true; return 0; }
    const v = this.view.getInt16(this.cursor, true);
    this._advance(2);
    return v;
  }

  readI32() {
    if (!this.canRead(4)) { this.overflow = true; return 0; }
    const v = this.view.getInt32(this.cursor, true);
    this._advance(4);
    return v;
  }

  readF32() {
    if (!this.canRead(4)) { this.overflow = true; return 0; }
    const v = this.view.getFloat32(this.cursor, true);
    this._advance(4);
    return v;
  }

  readF64() {
    if (!this.canRead(8)) { this.overflow = true; return 0; }
    const v = this.view.getFloat64(this.cursor, true);
    this._advance(8);
    return v;
  }

  readBytes(dst, length) {
    const n = length;
    if (!this.canRead(n)) { this.overflow = true; return 0; }
    dst.set(this.buffer.subarray(this.cursor, this.cursor + n));
    this._advance(n);
    return n;
  }

  seek(offset) {
    if (offset < 0 || offset > this.buffer.byteLength) return false;
    this.cursor = offset;
    return true;
  }

  tell() { return this.cursor; }
}

/* ------------------------------------------------------------------ */
/* 4. CHECKSUM (FNV-1a 32-bit)                                        */
/* ------------------------------------------------------------------ */

const FNV_OFFSET_BASIS = 0x811C9DC5 >>> 0;
const FNV_PRIME = 0x01000193 >>> 0;

function _fnv1a32(bytes, start, end) {
  let hash = FNV_OFFSET_BASIS;
  const s = start !== undefined ? start : 0;
  const e = end !== undefined ? end : bytes.length;
  for (let i = s; i < e; i++) {
    hash ^= bytes[i];
    hash = (hash * FNV_PRIME) >>> 0;
  }
  return hash >>> 0;
}

/* ------------------------------------------------------------------ */
/* 5. COMPONENT SCHEMA TABLE                                          */
/* ------------------------------------------------------------------ */

/**
 * For each component the serializer needs to know:
 *   • its numeric type id (from 013)
 *   • the SoA field list, in a stable order
 *   • per-field kind (f32/u8/u16/...)
 *   • per-field write/read closures
 *
 * We precompute this table once at module load. Every entry is frozen.
 */
class ComponentSchema {
  constructor(typeId, name, component, fields) {
    this.typeId  = typeId;
    this.name    = name;
    this.component = component;
    this.fields  = Object.freeze(fields);
    this.fieldCount = fields.length;
    Object.freeze(this);
  }
}

/**
 * Field kind tags.
 */
const FIELD_F32 = 0;
const FIELD_U8  = 1;
const FIELD_U16 = 2;
const FIELD_U32 = 3;
const FIELD_I8  = 4;
const FIELD_I16 = 5;
const FIELD_I32 = 6;

const _schemas = new Map(); // typeId → ComponentSchema
const _schemaByName = new Map();

function _field(name, kind) { return Object.freeze({ name, kind }); }

/**
 * Build the schema table for every serializable component. Components
 * are declared here in a stable order so the on-wire record layout is
 * deterministic across builds.
 */
function _buildSchemaTable() {
  /* ---------------- Transform ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.TRANSFORM, 'Transform', Transform, [
    _field('x', FIELD_F32),  _field('y', FIELD_F32),  _field('z', FIELD_F32),
    _field('qx', FIELD_F32), _field('qy', FIELD_F32), _field('qz', FIELD_F32), _field('qw', FIELD_F32),
    _field('sx', FIELD_F32), _field('sy', FIELD_F32), _field('sz', FIELD_F32),
  ]);

  /* ---------------- Target ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.TARGET, 'Target', Target, [
    _field('x', FIELD_F32), _field('y', FIELD_F32), _field('z', FIELD_F32),
    _field('active', FIELD_U8),
  ]);

  /* ---------------- LightRef ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_REF, 'LightRef', LightRef, [
    _field('type', FIELD_U8),  _field('kind', FIELD_U8),
    _field('colorR', FIELD_F32), _field('colorG', FIELD_F32), _field('colorB', FIELD_F32),
    _field('intensity', FIELD_F32),
    _field('range', FIELD_F32),
    _field('decay', FIELD_F32),
    _field('angle', FIELD_F32),
    _field('penumbra', FIELD_F32),
    _field('width', FIELD_F32),
    _field('height', FIELD_F32),
    _field('groundColorR', FIELD_F32), _field('groundColorG', FIELD_F32), _field('groundColorB', FIELD_F32),
  ]);

  /* ---------------- LightState ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_STATE, 'LightState', LightState, [
    _field('flags', FIELD_U16), _field('env', FIELD_U8), _field('envBlend', FIELD_F32),
    _field('lastUpdateFrame', FIELD_U32), _field('lastDirtyFrame', FIELD_U32),
    _field('frameActive', FIELD_U8),
  ]);

  /* ---------------- LightShadow ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_SHADOW, 'LightShadow', LightShadow, [
    _field('enabled', FIELD_U8),
    _field('mapSize', FIELD_U16),
    _field('cascadeCount', FIELD_U8),
    _field('filter', FIELD_U8),
    _field('bias', FIELD_F32),
    _field('normalBias', FIELD_F32),
    _field('softness', FIELD_F32),
    _field('distance', FIELD_F32),
    _field('atlasTileX', FIELD_U16), _field('atlasTileY', FIELD_U16),
    _field('atlasTileW', FIELD_U16), _field('atlasTileH', FIELD_U16),
    _field('atlasValid', FIELD_U8),
  ]);

  /* ---------------- LightCluster ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_CLUSTER, 'LightCluster', LightCluster, [
    _field('cellX', FIELD_I16), _field('cellY', FIELD_I16), _field('cellZ', FIELD_I16),
    _field('cellCount', FIELD_U16), _field('clusterDirty', FIELD_U8),
  ]);

  /* ---------------- LightBudget ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_BUDGET, 'LightBudget', LightBudgetComponent, [
    _field('cost', FIELD_F32), _field('costEma', FIELD_F32),
    _field('lod', FIELD_U8), _field('lodTarget', FIELD_U8),
    _field('lastLodFrame', FIELD_U32),
    _field('shadowAllocated', FIELD_U8),
  ]);

  /* ---------------- LightPriority ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_PRIORITY, 'LightPriority', LightPriority, [
    _field('bucket', FIELD_U8), _field('sortKey', FIELD_F32), _field('frameRank', FIELD_U16),
  ]);

  /* ---------------- LightFlicker ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_FLICKER, 'LightFlicker', LightFlicker, [
    _field('baseIntensity', FIELD_F32), _field('amplitude', FIELD_F32),
    _field('hz', FIELD_F32), _field('phase', FIELD_F32), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- LightDayCycle ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_DAY_CYCLE, 'LightDayCycle', LightDayCycle, [
    _field('dayCycle', FIELD_F32), _field('daySpeed', FIELD_F32),
    _field('enabled', FIELD_U8), _field('phaseOffset', FIELD_F32), _field('maxElevation', FIELD_F32),
  ]);

  /* ---------------- LightEmissive ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_EMISSIVE, 'LightEmissive', LightEmissive, [
    _field('emissiveR', FIELD_F32), _field('emissiveG', FIELD_F32), _field('emissiveB', FIELD_F32),
    _field('emissiveScale', FIELD_F32), _field('proxyVisible', FIELD_U8),
  ]);

  /* ---------------- LightIndoor ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.LIGHT_INDOOR, 'LightIndoor', LightIndoor, [
    _field('indoorWeight', FIELD_F32), _field('outdoorWeight', FIELD_F32),
    _field('transitionRate', FIELD_F32),
    _field('portalVisible', FIELD_U8), _field('occludedByWalls', FIELD_U8),
  ]);

  /* ---------------- CameraTag ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.CAMERA_TAG, 'CameraTag', CameraTagComponent, [
    _field('active', FIELD_U8), _field('fov', FIELD_F32), _field('near', FIELD_F32),
    _field('far', FIELD_F32), _field('aspect', FIELD_F32),
  ]);

  /* ---------------- ShadowCasterRef ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_CASTER_REF, 'ShadowCasterRef', ShadowCasterRef, [
    _field('lightEid', FIELD_I32), _field('casterKind', FIELD_U8), _field('castStrength', FIELD_F32),
    _field('casterRadius', FIELD_F32),
    _field('casterCenterX', FIELD_F32), _field('casterCenterY', FIELD_F32), _field('casterCenterZ', FIELD_F32),
    _field('enabled', FIELD_U8), _field('frameActive', FIELD_U8),
  ]);

  /* ---------------- ShadowBias ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_BIAS, 'ShadowBias', ShadowBias, [
    _field('bias', FIELD_F32), _field('normalBias', FIELD_F32),
    _field('depthBias', FIELD_F32), _field('slopeBias', FIELD_F32),
    _field('panCakeFix', FIELD_F32), _field('adaptiveBias', FIELD_U8),
    _field('lastBiasFrame', FIELD_U32),
  ]);

  /* ---------------- ShadowFilter ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_FILTER, 'ShadowFilter', ShadowFilter, [
    _field('mode', FIELD_U8), _field('kernelSize', FIELD_U8),
    _field('blockerSearch', FIELD_U8), _field('penumbraSize', FIELD_F32),
    _field('samples', FIELD_U8), _field('lightBleed', FIELD_F32),
  ]);

  /* ---------------- ShadowSoftness ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_SOFTNESS, 'ShadowSoftness', ShadowSoftness, [
    _field('softness', FIELD_F32), _field('penumbra', FIELD_F32),
    _field('edgeRounding', FIELD_F32), _field('bandCount', FIELD_U8),
    _field('edgeStyle', FIELD_U8), _field('gradientStrength', FIELD_F32),
  ]);

  /* ---------------- ShadowTint ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_TINT, 'ShadowTint', ShadowTint, [
    _field('tintR', FIELD_F32), _field('tintG', FIELD_F32), _field('tintB', FIELD_F32),
    _field('tintStrength', FIELD_F32),
    _field('rimStrength', FIELD_F32),
    _field('rimR', FIELD_F32), _field('rimG', FIELD_F32), _field('rimB', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);

  /* ---------------- ShadowUpdatePolicy ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_UPDATE_POLICY, 'ShadowUpdatePolicy', ShadowUpdatePolicy, [
    _field('mode', FIELD_U8), _field('interval', FIELD_U16),
    _field('lastUpdateFrame', FIELD_U32), _field('frameCounter', FIELD_U16),
    _field('cameraDeltaThreshold', FIELD_F32),
    _field('lastCameraX', FIELD_F32), _field('lastCameraY', FIELD_F32), _field('lastCameraZ', FIELD_F32),
    _field('forceUpdate', FIELD_U8),
  ]);

  /* ---------------- ShadowState ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.SHADOW_STATE, 'ShadowState', ShadowState, [
    _field('state', FIELD_U8), _field('prevState', FIELD_U8),
    _field('lastStateFrame', FIELD_U32),
    _field('failureCount', FIELD_U8), _field('lastError', FIELD_I32),
  ]);

  /* ---------------- GIProbeRef ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_PROBE_REF, 'GIProbeRef', GIProbeRef, [
    _field('type', FIELD_U8), _field('quality', FIELD_U8), _field('mode', FIELD_U8),
    _field('gridX', FIELD_I16), _field('gridY', FIELD_I16), _field('gridZ', FIELD_I16),
    _field('worldX', FIELD_F32), _field('worldY', FIELD_F32), _field('worldZ', FIELD_F32),
    _field('radius', FIELD_F32),
    _field('gridCellId', FIELD_I32),
    _field('enabled', FIELD_U8), _field('indoorFactor', FIELD_F32),
  ]);

  /* ---------------- GIIrradiance ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_IRRADIANCE, 'GIIrradiance', GIIrradiance, [
    _field('r', FIELD_F32), _field('g', FIELD_F32), _field('b', FIELD_F32),
    _field('rPrev', FIELD_F32), _field('gPrev', FIELD_F32), _field('bPrev', FIELD_F32),
    _field('confidence', FIELD_F32), _field('luminance', FIELD_F32),
  ]);

  /* ---------------- GIOcclusion ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_OCCLUSION, 'GIOcclusion', GIOcclusion, [
    _field('skyOcclusion', FIELD_F32), _field('obstacleOcclusion', FIELD_F32),
    _field('portalOcclusion', FIELD_F32), _field('sampleCount', FIELD_U8),
    _field('lastUpdateFrame', FIELD_U32), _field('valid', FIELD_U8),
  ]);

  /* ---------------- GIPortal ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_PORTAL, 'GIPortal', GIPortal, [
    _field('type', FIELD_U8), _field('enabled', FIELD_U8),
    _field('positionX', FIELD_F32), _field('positionY', FIELD_F32), _field('positionZ', FIELD_F32),
    _field('normalX', FIELD_F32), _field('normalY', FIELD_F32), _field('normalZ', FIELD_F32),
    _field('width', FIELD_F32), _field('height', FIELD_F32),
    _field('indoorRoomEid', FIELD_I32), _field('outdoorVolumeEid', FIELD_I32),
    _field('transmission', FIELD_F32),
    _field('tintR', FIELD_F32), _field('tintG', FIELD_F32), _field('tintB', FIELD_F32),
    _field('fluxR', FIELD_F32), _field('fluxG', FIELD_F32), _field('fluxB', FIELD_F32),
    _field('visible', FIELD_U8),
  ]);

  /* ---------------- GIBudget ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_BUDGET, 'GIBudget', GIBudget, [
    _field('cost', FIELD_F32), _field('costEma', FIELD_F32),
    _field('lastCostMs', FIELD_F32),
    _field('lod', FIELD_U8), _field('lodTarget', FIELD_U8),
    _field('lastLodFrame', FIELD_U32), _field('priority', FIELD_U8),
    _field('asyncPending', FIELD_U8),
    _field('asyncWorkerId', FIELD_I16), _field('asyncStartFrame', FIELD_U32),
    _field('asyncTimeout', FIELD_U16),
  ]);

  /* ---------------- GIState ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_STATE, 'GIState', GIState, [
    _field('state', FIELD_U8), _field('prevState', FIELD_U8),
    _field('lastStateFrame', FIELD_U32), _field('lastUpdateFrame', FIELD_U32),
    _field('lastBakeFrame', FIELD_U32),
    _field('failureCount', FIELD_U8), _field('lastError', FIELD_I32),
    _field('leakFlag', FIELD_U8),
  ]);

  /* ---------------- GICelBands ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_CEL_BANDS, 'GICelBands', GICelBands, [
    _field('enabled', FIELD_U8), _field('bandCount', FIELD_U8),
    _field('bandSoftness', FIELD_F32), _field('bandBias', FIELD_F32),
    _field('band0R', FIELD_F32), _field('band0G', FIELD_F32), _field('band0B', FIELD_F32),
    _field('band1R', FIELD_F32), _field('band1G', FIELD_F32), _field('band1B', FIELD_F32),
    _field('band2R', FIELD_F32), _field('band2G', FIELD_F32), _field('band2B', FIELD_F32),
    _field('band3R', FIELD_F32), _field('band3G', FIELD_F32), _field('band3B', FIELD_F32),
    _field('ditherStrength', FIELD_F32), _field('ditherScale', FIELD_F32),
  ]);

  /* ---------------- GIPalette ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_PALETTE, 'GIPalette', GIPalette, [
    _field('styleId', FIELD_U8),
    _field('satBias', FIELD_F32), _field('hueBias', FIELD_F32),
    _field('ambientColorR', FIELD_F32), _field('ambientColorG', FIELD_F32), _field('ambientColorB', FIELD_F32),
    _field('shadowTintR', FIELD_F32), _field('shadowTintG', FIELD_F32), _field('shadowTintB', FIELD_F32),
    _field('bounceWarmth', FIELD_F32), _field('bounceCoolness', FIELD_F32),
    _field('lerpRate', FIELD_F32), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- GILeak ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_LEAK, 'GILeak', GILeak, [
    _field('mode', FIELD_U8),
    _field('leakMagnitude', FIELD_F32), _field('threshold', FIELD_F32),
    _field('correctionFactor', FIELD_F32), _field('portalGateStrength', FIELD_F32),
    _field('detectionCount', FIELD_U32),
    _field('lastDetectionFrame', FIELD_U32), _field('valid', FIELD_U8),
  ]);

  /* ---------------- GITemporal ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_TEMPORAL, 'GITemporal', GITemporal, [
    _field('historyLength', FIELD_U8), _field('historyWeight', FIELD_F32),
    _field('reprojectionBias', FIELD_F32), _field('rejectionThreshold', FIELD_F32),
    _field('lastHistoryFrame', FIELD_U32),
    _field('reprojX', FIELD_F32), _field('reprojY', FIELD_F32),
    _field('variance', FIELD_F32), _field('valid', FIELD_U8),
  ]);

  /* ---------------- GIVolume ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_VOLUME, 'GIVolume', GIVolume, [
    _field('kind', FIELD_U8), _field('enabled', FIELD_U8),
    _field('minX', FIELD_F32), _field('minY', FIELD_F32), _field('minZ', FIELD_F32),
    _field('maxX', FIELD_F32), _field('maxY', FIELD_F32), _field('maxZ', FIELD_F32),
    _field('fillR', FIELD_F32), _field('fillG', FIELD_F32), _field('fillB', FIELD_F32),
    _field('blendDistance', FIELD_F32), _field('cellCount', FIELD_U16),
    _field('probeDensity', FIELD_F32), _field('leakGate', FIELD_F32),
    _field('generation', FIELD_U32),
  ]);

  /* ---------------- GIIndoor ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_INDOOR, 'GIIndoor', GIIndoor, [
    _field('volumeEid', FIELD_I32),
    _field('portalCount', FIELD_U8),
    _field('ceilingBounceR', FIELD_F32), _field('ceilingBounceG', FIELD_F32), _field('ceilingBounceB', FIELD_F32),
    _field('floorBounceR', FIELD_F32), _field('floorBounceG', FIELD_F32), _field('floorBounceB', FIELD_F32),
    _field('emissiveFillR', FIELD_F32), _field('emissiveFillG', FIELD_F32), _field('emissiveFillB', FIELD_F32),
    _field('wallOcclusion', FIELD_F32), _field('curtainTransmission', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);

  /* ---------------- GIOutdoor ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_OUTDOOR, 'GIOutdoor', GIOutdoor, [
    _field('volumeEid', FIELD_I32),
    _field('skyZenithR', FIELD_F32), _field('skyZenithG', FIELD_F32), _field('skyZenithB', FIELD_F32),
    _field('skyHorizonR', FIELD_F32), _field('skyHorizonG', FIELD_F32), _field('skyHorizonB', FIELD_F32),
    _field('groundAlbedoR', FIELD_F32), _field('groundAlbedoG', FIELD_F32), _field('groundAlbedoB', FIELD_F32),
    _field('hazeR', FIELD_F32), _field('hazeG', FIELD_F32), _field('hazeB', FIELD_F32),
    _field('sunDirR', FIELD_F32), _field('sunDirG', FIELD_F32), _field('sunDirB', FIELD_F32),
    _field('biomeWeight', FIELD_F32), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- GIReflectionProbe ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.GI_REFLECTION_PROBE, 'GIReflectionProbe', GIReflectionProbe, [
    _field('positionX', FIELD_F32), _field('positionY', FIELD_F32), _field('positionZ', FIELD_F32),
    _field('radius', FIELD_F32), _field('resolution', FIELD_U16),
    _field('captureFace', FIELD_U8), _field('captureFaceFrame', FIELD_U32),
    _field('lastCaptureFrame', FIELD_U32), _field('updateInterval', FIELD_U16),
    _field('hdrBias', FIELD_F32), _field('parallaxCorrect', FIELD_U8),
    _field('boxMinX', FIELD_F32), _field('boxMinY', FIELD_F32), _field('boxMinZ', FIELD_F32),
    _field('boxMaxX', FIELD_F32), _field('boxMaxY', FIELD_F32), _field('boxMaxZ', FIELD_F32),
    _field('valid', FIELD_U8), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AOVolumeRef ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_VOLUME_REF, 'AOVolumeRef', AOVolumeRef, [
    _field('method', FIELD_U8), _field('quality', FIELD_U8), _field('style', FIELD_U8), _field('enabled', FIELD_U8),
    _field('minX', FIELD_F32), _field('minY', FIELD_F32), _field('minZ', FIELD_F32),
    _field('maxX', FIELD_F32), _field('maxY', FIELD_F32), _field('maxZ', FIELD_F32),
    _field('intensity', FIELD_F32), _field('radius', FIELD_F32), _field('bias', FIELD_F32),
    _field('maxDistance', FIELD_F32), _field('indoorFactor', FIELD_F32),
  ]);

  /* ---------------- AOSampling ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_SAMPLING, 'AOSampling', AOSampling, [
    _field('sampleCount', FIELD_U8), _field('stepCount', FIELD_U8),
    _field('stepScale', FIELD_F32), _field('jitterAmount', FIELD_F32),
    _field('jitterHz', FIELD_F32), _field('hemisphereBias', FIELD_F32),
    _field('useNoise', FIELD_U8), _field('noiseScale', FIELD_F32),
    _field('adaptiveEnabled', FIELD_U8), _field('adaptiveThreshold', FIELD_F32),
    _field('adaptiveMinSteps', FIELD_U8), _field('adaptiveMaxSteps', FIELD_U8),
  ]);

  /* ---------------- AOBlur ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_BLUR, 'AOBlur', AOBlur, [
    _field('mode', FIELD_U8), _field('radius', FIELD_F32),
    _field('kernelSize', FIELD_U8), _field('depthThreshold', FIELD_F32),
    _field('normalThreshold', FIELD_F32), _field('sharpness', FIELD_F32),
    _field('pingPongPhase', FIELD_U8), _field('passes', FIELD_U8),
    _field('lastBlurMs', FIELD_F32), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AOQuality ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_QUALITY, 'AOQuality', AOQuality, [
    _field('tier', FIELD_U8), _field('tierTarget', FIELD_U8),
    _field('resolutionScale', FIELD_F32), _field('halfRes', FIELD_U8), _field('quarterRes', FIELD_U8),
    _field('downscaleFactor', FIELD_F32), _field('upsampleMode', FIELD_U8),
    _field('lastQualityChange', FIELD_U32),
  ]);

  /* ---------------- AOState ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_STATE, 'AOState', AOState, [
    _field('state', FIELD_U8), _field('prevState', FIELD_U8),
    _field('lastStateFrame', FIELD_U32), _field('lastUpdateFrame', FIELD_U32),
    _field('lastBakeFrame', FIELD_U32),
    _field('failureCount', FIELD_U8), _field('lastError', FIELD_I32),
    _field('leakFlag', FIELD_U8),
  ]);

  /* ---------------- AOBudget ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_BUDGET, 'AOBudget', AOBudget, [
    _field('cost', FIELD_F32), _field('costEma', FIELD_F32), _field('lastCostMs', FIELD_F32),
    _field('lod', FIELD_U8), _field('lodTarget', FIELD_U8),
    _field('lastLodFrame', FIELD_U32), _field('priority', FIELD_U8),
    _field('asyncPending', FIELD_U8),
    _field('asyncWorkerId', FIELD_I16), _field('asyncStartFrame', FIELD_U32),
    _field('asyncTimeout', FIELD_U16),
  ]);

  /* ---------------- AOContactShadow ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_CONTACT_SHADOW, 'AOContactShadow', AOContactShadow, [
    _field('enabled', FIELD_U8), _field('rayCount', FIELD_U8),
    _field('maxDistance', FIELD_F32), _field('stepCount', FIELD_U8),
    _field('thickness', FIELD_F32), _field('bias', FIELD_F32),
    _field('jitter', FIELD_F32), _field('fadeNear', FIELD_F32), _field('fadeFar', FIELD_F32),
    _field('strength', FIELD_F32),
    _field('tintR', FIELD_F32), _field('tintG', FIELD_F32), _field('tintB', FIELD_F32),
  ]);

  /* ---------------- AOIndoorVolume ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_INDOOR_VOLUME, 'AOIndoorVolume', AOIndoorVolume, [
    _field('volumeEid', FIELD_I32),
    _field('ceilingAO', FIELD_F32), _field('floorAO', FIELD_F32), _field('wallAO', FIELD_F32),
    _field('cornerBoost', FIELD_F32), _field('contactAO', FIELD_F32), _field('curtainAO', FIELD_F32),
    _field('ceilingAOColorR', FIELD_F32), _field('ceilingAOColorG', FIELD_F32), _field('ceilingAOColorB', FIELD_F32),
    _field('floorAOColorR', FIELD_F32), _field('floorAOColorG', FIELD_F32), _field('floorAOColorB', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AOOutdoorVolume ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_OUTDOOR_VOLUME, 'AOOutdoorVolume', AOOutdoorVolume, [
    _field('volumeEid', FIELD_I32),
    _field('groundAO', FIELD_F32), _field('skyAO', FIELD_F32),
    _field('horizonAO', FIELD_F32), _field('canopyAO', FIELD_F32), _field('waterAO', FIELD_F32),
    _field('biomeDensity', FIELD_F32),
    _field('groundAOColorR', FIELD_F32), _field('groundAOColorG', FIELD_F32), _field('groundAOColorB', FIELD_F32),
    _field('skyAOColorR', FIELD_F32), _field('skyAOColorG', FIELD_F32), _field('skyAOColorB', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AOTemporalAccumulator ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_TEMPORAL_ACC, 'AOTemporalAccumulator', AOTemporalAccumulator, [
    _field('mode', FIELD_U8), _field('frameCount', FIELD_U16),
    _field('blendFactor', FIELD_F32), _field('historyLength', FIELD_U8),
    _field('resetOnDisocclusion', FIELD_U8), _field('disocclusionThreshold', FIELD_F32),
    _field('jitterPhaseX', FIELD_F32), _field('jitterPhaseY', FIELD_F32),
    _field('lastAccumulateFrame', FIELD_U32), _field('persistence', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AODither ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_DITHER, 'AODither', AODither, [
    _field('mode', FIELD_U8), _field('strength', FIELD_F32), _field('scale', FIELD_F32),
    _field('animated', FIELD_U8), _field('animationHz', FIELD_F32), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AOCelBands ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_CEL_BANDS, 'AOCelBands', AOCelBands, [
    _field('enabled', FIELD_U8), _field('bandCount', FIELD_U8),
    _field('bandSoftness', FIELD_F32), _field('bandBias', FIELD_F32),
    _field('band0R', FIELD_F32), _field('band0G', FIELD_F32), _field('band0B', FIELD_F32),
    _field('band1R', FIELD_F32), _field('band1G', FIELD_F32), _field('band1B', FIELD_F32),
    _field('band2R', FIELD_F32), _field('band2G', FIELD_F32), _field('band2B', FIELD_F32),
    _field('band3R', FIELD_F32), _field('band3G', FIELD_F32), _field('band3B', FIELD_F32),
    _field('rimBoost', FIELD_F32),
    _field('shadowTintR', FIELD_F32), _field('shadowTintG', FIELD_F32), _field('shadowTintB', FIELD_F32),
  ]);

  /* ---------------- AOInkOutline ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_INK_OUTLINE, 'AOInkOutline', AOInkOutline, [
    _field('enabled', FIELD_U8), _field('thickness', FIELD_F32), _field('strength', FIELD_F32),
    _field('softness', FIELD_F32),
    _field('depthThreshold', FIELD_F32), _field('normalThreshold', FIELD_F32),
    _field('colorR', FIELD_F32), _field('colorG', FIELD_F32), _field('colorB', FIELD_F32),
    _field('widthLumaBias', FIELD_F32), _field('widthDistanceBias', FIELD_F32),
  ]);

  /* ---------------- AOEdgeFade ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_EDGE_FADE, 'AOEdgeFade', AOEdgeFade, [
    _field('enabled', FIELD_U8), _field('style', FIELD_U8),
    _field('startRadius', FIELD_F32), _field('endRadius', FIELD_F32),
    _field('strength', FIELD_F32), _field('aspectCompensate', FIELD_U8),
  ]);

  /* ---------------- AOBilateral ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_BILATERAL, 'AOBilateral', AOBilateral, [
    _field('depthSigma', FIELD_F32), _field('normalSigma', FIELD_F32), _field('lumaSigma', FIELD_F32),
    _field('spatialSigma', FIELD_F32), _field('kernelRadius', FIELD_U8),
    _field('passes', FIELD_U8), _field('enabled', FIELD_U8),
  ]);

  /* ---------------- AODenoiser ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_DENOISER, 'AODenoiser', AODenoiser, [
    _field('enabled', FIELD_U8), _field('spatialPasses', FIELD_U8), _field('temporalPasses', FIELD_U8),
    _field('blendStrength', FIELD_F32), _field('noiseThreshold', FIELD_F32),
    _field('minVariance', FIELD_F32), _field('preserveEdges', FIELD_U8),
    _field('edgeSharpness', FIELD_F32), _field('lastDenoiseMs', FIELD_F32),
  ]);

  /* ---------------- AOResidency ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_RESIDENCY, 'AOResidency', AOResidency, [
    _field('state', FIELD_U8), _field('bytesAllocated', FIELD_U32), _field('bytesPeak', FIELD_U32),
    _field('lastResidentFrame', FIELD_U32), _field('lastEvictFrame', FIELD_U32),
    _field('evictAfterFrames', FIELD_U16), _field('pinned', FIELD_U8),
  ]);

  /* ---------------- AOLeak ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_LEAK, 'AOLeak', AOLeak, [
    _field('enabled', FIELD_U8), _field('magnitude', FIELD_F32), _field('threshold', FIELD_F32),
    _field('correctionFactor', FIELD_F32), _field('detectionCount', FIELD_U32),
    _field('lastDetectionFrame', FIELD_U32), _field('smoothing', FIELD_F32), _field('valid', FIELD_U8),
  ]);

  /* ---------------- AOStyle ---------------- */
  _registerSchema(COMPONENT_TYPE_ID.AO_STYLE, 'AOStyle', AOStyle, [
    _field('style', FIELD_U8), _field('celBandCount', FIELD_U8), _field('celSoftness', FIELD_F32),
    _field('inkEnabled', FIELD_U8), _field('inkThickness', FIELD_F32), _field('inkStrength', FIELD_F32),
    _field('inkColorR', FIELD_F32), _field('inkColorG', FIELD_F32), _field('inkColorB', FIELD_F32),
    _field('overallIntensity', FIELD_F32), _field('ambientBoost', FIELD_F32),
    _field('tintR', FIELD_F32), _field('tintG', FIELD_F32), _field('tintB', FIELD_F32),
    _field('enabled', FIELD_U8),
  ]);
}

function _registerSchema(typeId, name, component, fields) {
  if (typeId === TYPE_ID_INVALID) return;
  const schema = new ComponentSchema(typeId, name, component, fields);
  _schemas.set(typeId, schema);
  _schemaByName.set(name, schema);
}

// Build once.
_buildSchemaTable();

/* ------------------------------------------------------------------ */
/* 6. FIELD READ / WRITE HELPERS                                      */
/* ------------------------------------------------------------------ */

function _readFieldValue(component, field) {
  return component[field.name];
}

function _writeFieldValue(component, field, index, value) {
  component[field.name][index] = value;
}

function _writeFieldValueToWriter(writer, value, kind) {
  switch (kind) {
    case FIELD_F32: writer.writeF32(value); break;
    case FIELD_U8:  writer.writeU8(value);  break;
    case FIELD_U16: writer.writeU16(value); break;
    case FIELD_U32: writer.writeU32(value); break;
    case FIELD_I8:  writer.writeI8(value);  break;
    case FIELD_I16: writer.writeI16(value); break;
    case FIELD_I32: writer.writeI32(value); break;
    default:        writer.writeF32(value); break;
  }
}

function _readFieldValueFromReader(reader, kind) {
  switch (kind) {
    case FIELD_F32: return reader.readF32();
    case FIELD_U8:  return reader.readU8();
    case FIELD_U16: return reader.readU16();
    case FIELD_U32: return reader.readU32();
    case FIELD_I8:  return reader.readI8();
    case FIELD_I16: return reader.readI16();
    case FIELD_I32: return reader.readI32();
    default:        return reader.readF32();
  }
}

/* ------------------------------------------------------------------ */
/* 7. ENTITY SCAN                                                     */
/* ------------------------------------------------------------------ */

/**
 * Scans the world for live entities and fills the caller-provided
 * Int32Array with their ids. Returns the count written.
 *
 * A "live entity" is any entity that exists in the world AND has at
 * least one component attached. Entities that exist but are bare are
 * skipped.
 */
export function collectLiveEntities(outArray) {
  if (!outArray) return 0;

  const world = getECSWorld();
  if (!world) return 0;

  let write = 0;
  const cap = outArray.length;

  // Scan every registered component's SoA arrays; an entity id is
  // considered live if its slot appears in any of the attached-entity
  // tables. Because bitECS doesn't expose that table publicly, we
  // instead scan the range [0, MAX_ENTITIES) and use `entityExists`
  // from the adapter (which is O(1)).
  const adapter = getAdapter();
  for (let eid = 0; eid < MAX_ENTITIES && write < cap; eid++) {
    if (adapter.entityAlive(eid)) {
      outArray[write++] = eid;
    }
  }
  return write;
}

/* ------------------------------------------------------------------ */
/* 8. WORLD SERIALIZATION                                             */
/* ------------------------------------------------------------------ */

/**
 * Pre-allocated scratch buffer for entity enumeration. Sized once.
 */
const _entityScratch = new Int32Array(MAX_ENTITIES);

/**
 * Pre-allocated writer. Reused across serializations to avoid
 * allocations. Caller receives a subarray view on the result.
 */
const _writer = new BinaryWriter(MAX_PAYLOAD_BYTES);

/**
 * Serializes the world into a binary payload. Returns a Uint8Array
 * subarray of the internal writer buffer — do NOT hold the reference
 * beyond the next call.
 */
export function serializeWorld(world, options) {
  const t0 = _now();

  const w = world || getECSWorld();
  if (!w) {
    SerializationState.totalFailures++;
    return null;
  }

  _writer.reset();

  // 1. Collect live entities.
  const count = collectLiveEntities(_entityScratch);
  const capped = Math.min(count, MAX_SERIALIZED_ENTITIES);

  // 2. Determine payload flags.
  let flags = PAYLOAD_FLAG.HAS_TAGS | PAYLOAD_FLAG.HAS_RELATIONS |
              PAYLOAD_FLAG.HAS_LIFETIMES | PAYLOAD_FLAG.HAS_CHECKSUM;
  if (options && options.includeBlueprints) flags |= PAYLOAD_FLAG.HAS_BLUEPRINTS;

  // 3. Build the header placeholder (we backfill after writing).
  _writer.seek(HEADER_SIZE);

  // 4. Write the entity record section.
  const entitySectionStart = _writer.tell();
  let written = 0;
  for (let i = 0; i < capped; i++) {
    const eid = _entityScratch[i];
    if (_serializeEntity(_writer, eid)) written++;
  }
  const entitySectionEnd = _writer.tell();

  // 5. Write the tag section.
  const tagSectionStart = _writer.tell();
  _writer.writeU32(written);
  for (let i = 0; i < capped; i++) {
    const eid = _entityScratch[i];
    _writer.writeU32(eid);
    _writer.writeU32(EntityTag[eid]);
    _writer.writeU32(EntityTag2[eid]);
    _writer.writeU32(FrameDirtyTag[eid]);
  }
  const tagSectionEnd = _writer.tell();

  // 6. Write the relation section.
  const relationSectionStart = _writer.tell();
  _serializeRelations(_writer);
  const relationSectionEnd = _writer.tell();

  // 7. Write the lifetime section.
  const lifetimeSectionStart = _writer.tell();
  _writer.writeU32(written);
  for (let i = 0; i < capped; i++) {
    const eid = _entityScratch[i];
    _serializeLifetime(_writer, eid);
  }
  const lifetimeSectionEnd = _writer.tell();

  // 8. Blueprint section (optional).
  let blueprintSectionSize = 0;
  if (options && options.includeBlueprints) {
    const blueprintSectionStart = _writer.tell();
    _serializeBlueprintInstances(_writer);
    blueprintSectionSize = _writer.tell() - blueprintSectionStart;
  }

  // 9. Backfill the header.
  const totalSize = _writer.tell();
  const headerEnd = HEADER_SIZE;

  _writer.seek(0);
  _writer.writeU32(MAGIC);
  _writer.writeU16(SCHEMA_VERSION);
  _writer.writeU16(PAYLOAD_KIND.WORLD_SNAPSHOT);
  _writer.writeU32(flags);
  _writer.writeU32(written);
  _writer.writeU32(_schemas.size);
  _writer.writeU32(tagSectionEnd - tagSectionStart);
  _writer.writeU32(relationSectionEnd - relationSectionStart);
  _writer.writeU32(lifetimeSectionEnd - lifetimeSectionStart);
  _writer.writeU32(blueprintSectionSize);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);

  // 10. Compute checksum over everything after the header.
  const bytes = _writer.buffer;
  const checksum = _fnv1a32(bytes, headerEnd, totalSize);
  _writer.seek(48);
  _writer.writeU32(checksum);

  if (_writer.overflow) {
    SerializationState.totalFailures++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[024_scn_Serialization] serializeWorld overflow (payload > ${MAX_PAYLOAD_BYTES} bytes)`);
    return null;
  }

  const out = _writer.buffer.subarray(0, totalSize);

  SerializationState.totalSerializations++;
  SerializationState.totalBytesWritten += totalSize;
  SerializationState.totalEntitiesWritten += written;
  if (totalSize > SerializationState.peakPayloadBytes) {
    SerializationState.peakPayloadBytes = totalSize;
  }

  const t1 = _now();
  const cost = t1 - t0;
  SerializationState.lastSerializeMs = cost;
  SerializationState.avgSerializeMs += (cost - SerializationState.avgSerializeMs) * 0.15;

  try {
    const profiler = getDefaultProfiler();
    if (profiler && profiler.mark) profiler.mark('serialize.world');
  } catch (_) { /* swallow */ }

  return out;
}

/* ------------------------------------------------------------------ */
/* 9. ENTITY SERIALIZATION                                            */
/* ------------------------------------------------------------------ */

function _serializeEntity(writer, eid) {
  // Write the entity id.
  writer.writeU32(eid);

  // Determine which components this entity has.
  // We write a component count followed by (typeId, field-data) pairs.
  const adapter = getAdapter();
  let componentCount = 0;

  // Count first (bitECS doesn't let us enumerate directly; we use the
  // adapter for each known schema).
  for (const [typeId, schema] of _schemas) {
    if (adapter.hasComponent(eid, schema.component)) componentCount++;
  }

  writer.writeU16(componentCount);

  // Write each component record.
  for (const [typeId, schema] of _schemas) {
    if (!adapter.hasComponent(eid, schema.component)) continue;

    writer.writeU16(typeId);
    // Write every field value.
    for (let f = 0; f < schema.fieldCount; f++) {
      const field = schema.fields[f];
      const value = schema.component[field.name][eid];
      _writeFieldValueToWriter(writer, value, field.kind);
    }
  }

  return true;
}

/* ------------------------------------------------------------------ */
/* 10. RELATION SERIALIZATION                                         */
/* ------------------------------------------------------------------ */

function _serializeRelations(writer) {
  // 1. Write parent edges: count, then (childEid, parentEid, refKind).
  let parentCount = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (Parent.eid[eid] !== NULL_ENTITY) parentCount++;
  }
  writer.writeU32(parentCount);
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const parentEid = Parent.eid[eid];
    if (parentEid === NULL_ENTITY) continue;
    writer.writeU32(eid);
    writer.writeI32(parentEid);
    writer.writeU8(Parent.ref[eid]);
  }

  // 2. Write typed reference edges: count, then
  //    (srcEid, kind, dstEid, weight).
  let refCount = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    refCount += References.refCount[eid];
  }
  writer.writeU32(refCount);
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    const base = eid * MAX_REFERENCES_PER_ENTITY;
    const n = References.refCount[eid];
    for (let i = 0; i < n; i++) {
      const kind = References.refKind[base + i];
      const dst  = References.refTarget[base + i];
      const w    = References.refWeight[base + i];
      if (kind === REF.NONE || dst === NULL_ENTITY) continue;
      writer.writeU32(eid);
      writer.writeU8(kind);
      writer.writeI32(dst);
      writer.writeF32(w);
    }
  }
}

/* ------------------------------------------------------------------ */
/* 11. LIFETIME SERIALIZATION                                         */
/* ------------------------------------------------------------------ */

function _serializeLifetime(writer, eid) {
  writer.writeU32(eid);
  writer.writeU8(EntityLifetime.state[eid]);
  writer.writeU8(EntityLifetime.prevState[eid]);
  writer.writeI8(EntityLifetime.pool[eid]);
  writer.writeU32(EntityLifetime.spawnFrame[eid]);
  writer.writeU32(EntityLifetime.aliveFrames[eid]);
  writer.writeU32(EntityLifetime.ttlFrames[eid]);
  writer.writeU16(EntityLifetime.dyingFrames[eid]);
  writer.writeU16(EntityLifetime.dyingGrace[eid]);
  writer.writeU32(EntityLifetime.lastTouchFrame[eid]);
  writer.writeU32(EntityLifetime.lastStateFrame[eid]);
  writer.writeU16(EntityLifetime.stateChangeCount[eid]);
  writer.writeU8(EntityLifetime.persistent[eid]);
  writer.writeU8(EntityLifetime.transient[eid]);
  writer.writeU8(EntityLifetime.failureCount[eid]);
}

/* ------------------------------------------------------------------ */
/* 12. BLUEPRINT SERIALIZATION                                        */
/* ------------------------------------------------------------------ */

function _serializeBlueprintInstances(writer) {
  // First count.
  let count = 0;
  forEachInstance(() => { count++; });
  writer.writeU32(count);

  forEachInstance((instance, publicId) => {
    writer.writeU32(publicId);
    writer.writeI32(instance.blueprintId);
    writer.writeU16(instance.memberCount);
    for (let i = 0; i < instance.memberCount; i++) {
      writer.writeI32(instance.memberEids[i]);
    }
    writer.writeI32(instance.rootEid);
    writer.writeU32(instance.instantiatedAtFrame);
  });
}

/* ------------------------------------------------------------------ */
/* 13. WORLD DESERIALIZATION                                          */
/* ------------------------------------------------------------------ */

/**
 * Pre-allocated reader used for every deserialization.
 * Pointed at the incoming buffer via `reset()`.
 */
let _reader = null;
function _ensureReader(buffer) {
  if (!_reader) _reader = new BinaryReader(buffer);
  else _reader.buffer = buffer, _reader.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return _reader;
}

/**
 * Deserializes a binary payload into the world. Returns a report object
 * with counts, or null on failure.
 *
 *   const report = deserializeWorld(world, payload);
 */
export function deserializeWorld(world, payload) {
  const t0 = _now();

  if (!payload || !ArrayBuffer.isView(payload)) {
    SerializationState.totalFailures++;
    return null;
  }

  const w = world || getECSWorld();
  if (!w) {
    SerializationState.totalFailures++;
    return null;
  }

  const reader = _ensureReader(payload);
  reader.reset();

  // 1. Verify header.
  const magic = reader.readU32();
  if (magic !== MAGIC) {
    SerializationState.totalFailures++;
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[024_scn_Serialization] bad magic: 0x${magic.toString(16)}`);
    return null;
  }

  const version = reader.readU16();
  if (version !== SCHEMA_VERSION) {
    SerializationState.totalFailures++;
    const log = _safeLogger();
    if (log) log.warn(LOG_CHANNEL.CORE,
      `[024_scn_Serialization] schema mismatch: payload v${version} vs current v${SCHEMA_VERSION}`);
    // Continue — the field layout may still be compatible.
  }

  const payloadKind = reader.readU16();
  const flags = reader.readU32();
  const entityCount = reader.readU32();
  const componentTypeCount = reader.readU32();
  const tagSectionSize = reader.readU32();
  const relationSectionSize = reader.readU32();
  const lifetimeSectionSize = reader.readU32();
  const blueprintSectionSize = reader.readU32();
  /* reserved */
  reader.readU32();
  reader.readU32();
  const storedChecksum = reader.readU32();
  /* reserved */
  reader.readU32();
  reader.readU32();
  reader.readU32();

  // 2. Verify checksum.
  if (flags & PAYLOAD_FLAG.HAS_CHECKSUM) {
    const computed = _fnv1a32(payload, HEADER_SIZE, payload.byteLength);
    if (computed !== storedChecksum) {
      SerializationState.totalFailures++;
      const log = _safeLogger();
      if (log) log.error(LOG_CHANNEL.CORE,
        `[024_scn_Serialization] checksum mismatch: expected 0x${storedChecksum.toString(16)}, got 0x${computed.toString(16)}`);
      return null;
    }
  }

  // 3. Read the entity section.
  reader.seek(HEADER_SIZE);
  let entitiesRead = 0;
  for (let i = 0; i < entityCount; i++) {
    if (!_deserializeEntity(reader, w)) break;
    entitiesRead++;
  }

  // 4. Read the tag section.
  if (flags & PAYLOAD_FLAG.HAS_TAGS) {
    const tagCount = reader.readU32();
    for (let i = 0; i < tagCount; i++) {
      const eid = reader.readU32();
      if (eid < 0 || eid >= MAX_ENTITIES) { reader.seek(reader.tell() + 12); continue; }
      EntityTag[eid] = reader.readU32();
      EntityTag2[eid] = reader.readU32();
      FrameDirtyTag[eid] = reader.readU32();
    }
  }

  // 5. Read the relation section.
  if (flags & PAYLOAD_FLAG.HAS_RELATIONS) {
    _deserializeRelations(reader);
  }

  // 6. Read the lifetime section.
  if (flags & PAYLOAD_FLAG.HAS_LIFETIMES) {
    const lifetimeCount = reader.readU32();
    for (let i = 0; i < lifetimeCount; i++) {
      _deserializeLifetime(reader);
    }
  }

  // 7. Blueprint instances.
  if (flags & PAYLOAD_FLAG.HAS_BLUEPRINTS && blueprintSectionSize > 0) {
    _deserializeBlueprintInstances(reader);
  }

  if (reader.overflow) {
    SerializationState.totalFailures++;
    const log = _safeLogger();
    if (log) log.error(LOG_CHANNEL.CORE,
      `[024_scn_Serialization] reader overflow during deserialization`);
    return null;
  }

  SerializationState.totalDeserializations++;
  SerializationState.totalBytesRead += payload.byteLength;
  SerializationState.totalEntitiesRead += entitiesRead;

  const t1 = _now();
  const cost = t1 - t0;
  SerializationState.lastDeserializeMs = cost;
  SerializationState.avgDeserializeMs += (cost - SerializationState.avgDeserializeMs) * 0.15;

  return {
    entityCount:      entitiesRead,
    componentTypes:   componentTypeCount,
    payloadKind:      PAYLOAD_KIND_NAME[payloadKind] || 'unknown',
    payloadBytes:     payload.byteLength,
    schemaVersion:    version,
    flags,
  };
}

function _deserializeEntity(reader, world) {
  const eid = reader.readU32();
  if (eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!entityAlive(eid)) return false;

  const componentCount = reader.readU16();
  const adapter = getAdapter();

  for (let i = 0; i < componentCount; i++) {
    const typeId = reader.readU16();
    const schema = _schemas.get(typeId);
    if (!schema) {
      // Unknown component type — cannot skip since we don't know its
      // field layout. Bail out.
      return false;
    }
    // Ensure the component is attached.
    if (!adapter.hasComponent(eid, schema.component)) {
      adapter.attachComponent(eid, schema.component);
    }
    // Read each field value and write into the SoA.
    for (let f = 0; f < schema.fieldCount; f++) {
      const field = schema.fields[f];
      const value = _readFieldValueFromReader(reader, field.kind);
      schema.component[field.name][eid] = value;
    }
  }

  return true;
}

function _deserializeRelations(reader) {
  // 1. Parent edges.
  const parentCount = reader.readU32();
  for (let i = 0; i < parentCount; i++) {
    const childEid = reader.readU32();
    const parentEid = reader.readI32();
    const refKind = reader.readU8();
    if (childEid < 0 || childEid >= MAX_ENTITIES) continue;
    if (parentEid < -1 || parentEid >= MAX_ENTITIES) continue;
    Parent.eid[childEid] = parentEid;
    Parent.ref[childEid] = refKind;

    // Rebuild the child edge entry on the parent.
    if (parentEid >= 0) {
      const base = parentEid * MAX_CHILDREN_PER_ENTITY;
      for (let s = 0; s < MAX_CHILDREN_PER_ENTITY; s++) {
        if (Children.childEid[base + s] === NULL_ENTITY) {
          Children.childEid[base + s] = childEid;
          Children.childRef[base + s] = refKind;
          if (HierarchyState.childCount[parentEid] < 0xFFFF) {
            HierarchyState.childCount[parentEid]++;
          }
          break;
        }
      }
    }
  }

  // 2. Reference edges.
  const refCount = reader.readU32();
  for (let i = 0; i < refCount; i++) {
    const srcEid = reader.readU32();
    const kind = reader.readU8();
    const dstEid = reader.readI32();
    const weight = reader.readF32();
    if (srcEid < 0 || srcEid >= MAX_ENTITIES) continue;
    if (dstEid < 0 || dstEid >= MAX_ENTITIES) continue;
    if (kind === REF.NONE) continue;

    // Append the reference directly to the SoA arrays.
    const base = srcEid * MAX_REFERENCES_PER_ENTITY;
    const n = References.refCount[srcEid];
    if (n >= MAX_REFERENCES_PER_ENTITY) continue;
    References.refKind[base + n]   = kind;
    References.refTarget[base + n] = dstEid;
    References.refWeight[base + n] = weight;
    References.refCount[srcEid] = n + 1;

    // Reverse reference.
    const rBase = dstEid * MAX_REVERSE_REFERENCES;
    const rN = ReverseReferences.count[dstEid];
    if (rN < MAX_REVERSE_REFERENCES) {
      ReverseReferences.srcEid[rBase + rN] = srcEid;
      ReverseReferences.srcKind[rBase + rN] = kind;
      ReverseReferences.count[dstEid] = rN + 1;
    }
  }
}

function _deserializeLifetime(reader) {
  const eid = reader.readU32();
  if (eid < 0 || eid >= MAX_ENTITIES) return;
  EntityLifetime.state[eid]            = reader.readU8();
  EntityLifetime.prevState[eid]        = reader.readU8();
  EntityLifetime.pool[eid]             = reader.readI8();
  EntityLifetime.spawnFrame[eid]       = reader.readU32();
  EntityLifetime.aliveFrames[eid]      = reader.readU32();
  EntityLifetime.ttlFrames[eid]        = reader.readU32();
  EntityLifetime.dyingFrames[eid]      = reader.readU16();
  EntityLifetime.dyingGrace[eid]       = reader.readU16();
  EntityLifetime.lastTouchFrame[eid]   = reader.readU32();
  EntityLifetime.lastStateFrame[eid]   = reader.readU32();
  EntityLifetime.stateChangeCount[eid] = reader.readU16();
  EntityLifetime.persistent[eid]       = reader.readU8();
  EntityLifetime.transient[eid]        = reader.readU8();
  EntityLifetime.failureCount[eid]     = reader.readU8();
}

function _deserializeBlueprintInstances(reader) {
  const count = reader.readU32();
  for (let i = 0; i < count; i++) {
    const publicId = reader.readU32();
    const blueprintId = reader.readI32();
    const memberCount = reader.readU16();
    const eids = new Array(memberCount);
    for (let m = 0; m < memberCount; m++) {
      eids[m] = reader.readI32();
    }
    const rootEid = reader.readI32();
    const instantiatedAtFrame = reader.readU32();

    // Hand off to 023_scn_Blueprints to register the instance record.
    // We import lazily via the module's public API to avoid a hard
    // circular dependency at module init.
    try {
      const bp = getInstance(publicId);
      if (bp) {
        // Instance already exists — skip.
        continue;
      }
      // Register directly using the blueprint instance slot.
      _registerDeserializedInstance(publicId, blueprintId, eids, rootEid, instantiatedAtFrame);
    } catch (_) {
      // Swallow — instance registration failures are non-fatal.
    }
  }
}

/**
 * Placeholder registration helper. `023_scn_Blueprints.js` exposes the
 * instance registry through its public API; when the deserializer runs,
 * it can opt to restore instances via a callback rather than reaching
 * into the module's internal state.
 *
 * For now, we no-op. Downstream consumers may override this behavior by
 * hooking `setDeserializedInstanceRegistrar(fn)`.
 */
let _instanceRegistrar = null;
export function setDeserializedInstanceRegistrar(fn) {
  _instanceRegistrar = typeof fn === 'function' ? fn : null;
}

function _registerDeserializedInstance(publicId, blueprintId, eids, rootEid, frame) {
  if (_instanceRegistrar) {
    try {
      _instanceRegistrar(publicId, blueprintId, eids, rootEid, frame);
    } catch (_) { /* swallow */ }
  }
}

/* ------------------------------------------------------------------ */
/* 14. BLUEPRINT INSTANCE SERIALIZATION                               */
/* ------------------------------------------------------------------ */

/**
 * Serializes a single blueprint instance into a binary payload.
 */
export function serializeBlueprintInstance(instanceId) {
  const instance = getInstance(instanceId);
  if (!instance) {
    SerializationState.totalFailures++;
    return null;
  }

  _writer.reset();
  _writer.seek(HEADER_SIZE);

  // Body: instance record.
  _writer.writeU32(instanceId);
  _writer.writeI32(instance.blueprintId);
  _writer.writeU16(instance.memberCount);
  for (let i = 0; i < instance.memberCount; i++) {
    _writer.writeI32(instance.memberEids[i]);
  }
  _writer.writeI32(instance.rootEid);
  _writer.writeU32(instance.instantiatedAtFrame);

  const bodySize = _writer.tell() - HEADER_SIZE;

  // Header.
  _writer.seek(0);
  _writer.writeU32(MAGIC);
  _writer.writeU16(SCHEMA_VERSION);
  _writer.writeU16(PAYLOAD_KIND.BLUEPRINT_INSTANCE);
  _writer.writeU32(PAYLOAD_FLAG.HAS_CHECKSUM);
  _writer.writeU32(instance.memberCount);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);
  _writer.writeU32(0);

  const totalSize = HEADER_SIZE + bodySize;
  const checksum = _fnv1a32(_writer.buffer, HEADER_SIZE, totalSize);
  _writer.seek(48);
  _writer.writeU32(checksum);

  SerializationState.totalSerializations++;
  SerializationState.totalBytesWritten += totalSize;

  return _writer.buffer.subarray(0, totalSize);
}

/**
 * Deserializes a blueprint instance payload into a newly spawned
 * instance.
 */
export function deserializeBlueprintInstance(world, payload) {
  if (!payload || !ArrayBuffer.isView(payload)) return -1;

  const reader = _ensureReader(payload);
  reader.reset();

  const magic = reader.readU32();
  if (magic !== MAGIC) return -1;

  const version = reader.readU16();
  const payloadKind = reader.readU16();
  if (payloadKind !== PAYLOAD_KIND.BLUEPRINT_INSTANCE) return -1;

  const flags = reader.readU32();
  const memberCount = reader.readU32();
  /* reserved */
  reader.readU32(); reader.readU32(); reader.readU32(); reader.readU32();
  reader.readU32(); reader.readU32();
  const storedChecksum = reader.readU32();
  reader.readU32(); reader.readU32(); reader.readU32();

  if (flags & PAYLOAD_FLAG.HAS_CHECKSUM) {
    const computed = _fnv1a32(payload, HEADER_SIZE, payload.byteLength);
    if (computed !== storedChecksum) return -1;
  }

  reader.seek(HEADER_SIZE);

  const publicId = reader.readU32();
  const blueprintId = reader.readI32();
  const count = reader.readU16();
  const eids = new Array(count);
  for (let m = 0; m < count; m++) eids[m] = reader.readI32();
  const rootEid = reader.readI32();
  const frame = reader.readU32();

  // Hand off to the blueprint registrar.
  _registerDeserializedInstance(publicId, blueprintId, eids, rootEid, frame);

  SerializationState.totalDeserializations++;
  return publicId;
}

/* ------------------------------------------------------------------ */
/* 15. JSON SERIALIZATION                                             */
/* ------------------------------------------------------------------ */

/**
 * Produces a JSON-friendly object graph from the world. Allocates a
 * fresh object — use only for debug dumps and diffs, not for the hot
 * path.
 */
export function serializeWorldToJSON(world, options) {
  const w = world || getECSWorld();
  if (!w) return null;

  const count = collectLiveEntities(_entityScratch);
  const capped = Math.min(count, MAX_SERIALIZED_ENTITIES);

  const out = {
    schemaVersion: SCHEMA_VERSION,
    payloadKind:   PAYLOAD_KIND_NAME[PAYLOAD_KIND.WORLD_SNAPSHOT],
    entityCount:   capped,
    entities:      [],
  };

  const adapter = getAdapter();
  for (let i = 0; i < capped; i++) {
    const eid = _entityScratch[i];
    const entity = { eid, components: {}, tags: { tag: EntityTag[eid], tag2: EntityTag2[eid] } };

    for (const [typeId, schema] of _schemas) {
      if (!adapter.hasComponent(eid, schema.component)) continue;
      const fields = {};
      for (let f = 0; f < schema.fieldCount; f++) {
        const field = schema.fields[f];
        fields[field.name] = schema.component[field.name][eid];
      }
      entity.components[schema.name] = fields;
    }

    // Relations.
    const parentEid = Parent.eid[eid];
    if (parentEid !== NULL_ENTITY) {
      entity.parent = { eid: parentEid, ref: REF_NAME[Parent.ref[eid]] || 'none' };
    }
    const base = eid * MAX_REFERENCES_PER_ENTITY;
    const n = References.refCount[eid];
    if (n > 0) {
      entity.references = [];
      for (let r = 0; r < n; r++) {
        const kind = References.refKind[base + r];
        const dst  = References.refTarget[base + r];
        if (kind === REF.NONE || dst === NULL_ENTITY) continue;
        entity.references.push({
          kind:   REF_NAME[kind] || 'none',
          target: dst,
          weight: References.refWeight[base + r],
        });
      }
    }

    // Lifetime.
    entity.lifetime = {
      state:    EntityLifetime.state[eid],
      pool:     EntityLifetime.pool[eid],
      spawnFrame: EntityLifetime.spawnFrame[eid],
      aliveFrames: EntityLifetime.aliveFrames[eid],
      ttl:      EntityLifetime.ttlFrames[eid],
    };

    out.entities.push(entity);
  }

  if (options && options.includeBlueprints) {
    const instances = [];
    forEachInstance((inst, id) => {
      instances.push({
        instanceId: id,
        blueprintId: inst.blueprintId,
        memberEids: Array.from(inst.memberEids.subarray(0, inst.memberCount)),
        rootEid:    inst.rootEid,
      });
    });
    out.blueprints = instances;
  }

  return out;
}

/* ------------------------------------------------------------------ */
/* 16. ROUND-TRIP VERIFICATION                                        */
/* ------------------------------------------------------------------ */

/**
 * Verifies that a serialize → deserialize → serialize round trip
 * produces a byte-identical payload. Used as a regression check and
 * during development.
 */
export function verifyRoundTrip(world) {
  const payloadA = serializeWorld(world);
  if (!payloadA) return false;

  // Copy because the internal writer buffer is reused.
  const copyA = payloadA.slice();

  const report = deserializeWorld(world, copyA);
  if (!report) return false;

  const payloadB = serializeWorld(world);
  if (!payloadB) return false;

  if (payloadB.byteLength !== copyA.byteLength) {
    SerializationState.totalFailures++;
    return false;
  }

  for (let i = 0; i < copyA.byteLength; i++) {
    if (copyA[i] !== payloadB[i]) {
      SerializationState.totalFailures++;
      return false;
    }
  }

  SerializationState.totalRoundTrips++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 17. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getSerializationReport() {
  return {
    schemaVersion:          SCHEMA_VERSION,
    registeredSchemas:      _schemas.size,
    totalSerializations:    SerializationState.totalSerializations,
    totalDeserializations:  SerializationState.totalDeserializations,
    totalRoundTrips:        SerializationState.totalRoundTrips,
    totalFailures:          SerializationState.totalFailures,
    totalBytesWritten:      SerializationState.totalBytesWritten,
    totalBytesRead:         SerializationState.totalBytesRead,
    totalEntitiesWritten:   SerializationState.totalEntitiesWritten,
    totalEntitiesRead:      SerializationState.totalEntitiesRead,
    peakPayloadBytes:       SerializationState.peakPayloadBytes,
    lastSerializeMs:        SerializationState.lastSerializeMs,
    lastDeserializeMs:      SerializationState.lastDeserializeMs,
    avgSerializeMs:         SerializationState.avgSerializeMs,
    avgDeserializeMs:       SerializationState.avgDeserializeMs,
    maxPayloadBytes:        MAX_PAYLOAD_BYTES,
    maxSerializedEntities:  MAX_SERIALIZED_ENTITIES,
    perfTier:               PERF_TIER_LOCAL,
  };
}

/**
 * Returns the schema table as a JSON-friendly array. Useful for
 * documentation and regression diffing.
 */
export function describeSchema() {
  const out = [];
  for (const [typeId, schema] of _schemas) {
    out.push({
      typeId,
      name:       schema.name,
      fieldCount: schema.fieldCount,
      fields:     schema.fields.map((f) => ({ name: f.name, kind: f.kind })),
    });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 18. REGISTRATION                                                   */
/* ------------------------------------------------------------------ */

/**
 * The serialization module does not declare its own ECS components; it
 * operates on every registered component via the schema table. No-op,
 * present for API symmetry.
 */
export function registerSerializationComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 19. RESET                                                          */
/* ------------------------------------------------------------------ */

/**
 * Resets runtime counters. The schema table is immutable.
 */
export function resetSerializationState() {
  SerializationState.totalSerializations = 0;
  SerializationState.totalDeserializations = 0;
  SerializationState.totalRoundTrips = 0;
  SerializationState.totalFailures = 0;
  SerializationState.totalBytesWritten = 0;
  SerializationState.totalBytesRead = 0;
  SerializationState.totalEntitiesWritten = 0;
  SerializationState.totalEntitiesRead = 0;
  SerializationState.peakPayloadBytes = 0;
  SerializationState.lastSerializeMs = 0;
  SerializationState.lastDeserializeMs = 0;
  SerializationState.avgSerializeMs = 0;
  SerializationState.avgDeserializeMs = 0;
}

/* ------------------------------------------------------------------ */
/* 20. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  SCHEMA_VERSION,
  MAGIC,
  HEADER_SIZE,
  MAX_PAYLOAD_BYTES,
  MAX_SERIALIZED_ENTITIES,
  PAYLOAD_KIND,
  PAYLOAD_KIND_NAME,
  PAYLOAD_FLAG,

  // Classes
  BinaryWriter,
  BinaryReader,

  // Module state
  SerializationState,

  // World
  serializeWorld,
  deserializeWorld,
  serializeWorldToJSON,
  verifyRoundTrip,

  // Blueprint
  serializeBlueprintInstance,
  deserializeBlueprintInstance,
  setDeserializedInstanceRegistrar,

  // Enumeration
  collectLiveEntities,

  // Diagnostics
  getSerializationReport,
  describeSchema,

  // Registration
  registerSerializationComponents,

  // Reset
  resetSerializationState,
};

export default _defaultExport;