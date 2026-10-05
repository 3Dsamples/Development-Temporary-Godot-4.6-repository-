// File : 032
// name : src/ecs/032_scn_BiomeComponents.js
// description : Biome SoA component module for the scene ECS world of the
//               anime lighting stack on Android mobile, plus the procedural
//               world and dynamic biome transition helpers that let the
//               engine build a complete procedural world at runtime and
//               cross-fade between biomes seamlessly.
//
//               Declares every per-entity biome state, biome weight vector,
//               biome transition timer, biome palette, biome climate
//               modifier, biome material override, biome scatter state,
//               biome scatter budget, biome zone blend, and biome
//               statistics record the lighting stack needs — as
//               fixed-capacity typed arrays sized once to
//               MAX_ENTITIES = 100000 (or a fixed registry for the biome
//               definitions themselves).
//
//               Procedural world generation helpers:
//                 • generateProceduralWorld        — end-to-end terrain +
//                                                    biome + scatter pass
//                 • sampleTerrainHeight            — fBm-driven height
//                 • sampleTerrainNormal            — analytic normal
//                 • sampleBiomeWeights             — biome weights from
//                                                    elevation + moisture +
//                                                    temperature
//                 • classifyBiome                  — dominant biome id
//                 • sampleBiomeColor               — Oklab-interpolated
//                                                    blended color
//                 • generateChunkHeightField       — per-chunk height grid
//                 • generateChunkBiomeField        — per-chunk biome ids
//                 • generateChunkScatterField      — per-chunk scatter
//                                                    placements
//                 • buildTerrainGeometry           — heights → BufferGeometry
//                 • buildBiomeColorAttribute       — biome colors → vertex
//                                                    color attribute
//                 • evaluateChunkRelief            — relief variance estimate
//                 • computeChunkLODFromRelief      — pick LOD from relief
//
//               Dynamic biome transition helpers:
//                 • setBiome                       — install a target biome
//                 • blendBiomeWeights              — cross-fade weights
//                 • tickBiomeTransition            — advance the transition
//                 • cancelBiomeTransition          — abort an in-flight
//                                                    transition
//                 • forceBiomeInstant              — snap to target
//                 • evaluateBiomeTransition        — returns progress [0,1]
//                 • applyBiomeToEntity             — write biome fields on
//                                                    one entity
//                 • applyBiomeToAllEntities        — write biome fields on
//                                                    every registered
//                                                    entity
//                 • registerBiomeEntity            — opt-in a biome-aware
//                                                    entity
//                 • unregisterBiomeEntity          — opt-out
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no runtime resizes.
//                 • Biome definitions live in a frozen registry with Oklab
//                   palettes matching the anime reference image set:
//                     DESERT, SNOW, SEA, FOREST, CANYON, COASTAL,
//                     WETLAND, TUNDRA, VOLCANIC, plus SPRING_SUMMER /
//                     AUTUMN / WINTER season variants.
//                 • Dynamic transition is a smoothstep-eased Oklab lerp
//                   between two palettes and a weighted sum of 3 biome
//                   weights — the same model used by the world core in
//                   world.js so palettes stay consistent across the
//                   engine.
//                 • Procedural world generation is fully deterministic
//                   from a seed, allocation-free per-cell, and works on
//                   any grid resolution.
//                 • Zero allocations on the hot path — every helper works
//                   directly on the SoA arrays; scratch buffers are
//                   module-level.
//
//               Integration:
//                 • 002_lgt_LightComponents.js
//                 • 003_lgt_ShadowComponents.js
//                 • 004_lgt_GIComponents.js
//                 • 005_lgt_AOComponents.js
//                 • 010_scn_ECSWorld.js
//                 • 011_scn_BiteCSAdapter.js
//                 • 012_scn_ComponentRegistry.js
//                 • 013_scn_ComponentTypes.js
//                 • 014_scn_Tags.js
//                 • 015_scn_Relations.js
//                 • 016_scn_EntityPool.js
//                 • 017_scn_EntityLifetime.js
//                 • 025_scn_SpatialComponents.js
//                 • 026_scn_TransformComponents.js
//                 • 027_scn_LODComponents.js
//                 • 028_scn_StreamingComponents.js
//                 • 029_scn_VisibilityComponents.js
//                 • 030_scn_AnimationComponents.js
//                 • 031_scn_WeatherComponents.js
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that the anime lighting stack can build a
//            complete procedural world from a seed — terrain, biome
//            classification, vegetation scatter, vertex colors — and then
//            cross-fade the entire scene between biomes in real time with
//            perceptual Oklab blending, all allocation-free and consistent
//            with the palette model in world.js.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

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
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
  entityAlive,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  COMPONENT_TYPE_ID,
} from './013_scn_ComponentTypes.js';

import {
  TAG,
  TAG2,
  FDIRTY,
  EntityTag,
  EntityTag2,
  markFrameDirty,
  clearFrameDirty,
} from './014_scn_Tags.js';

import {
  Transform,
  TransformWorld,
} from './026_scn_TransformComponents.js';

import {
  Sphere,
  AABB,
} from './025_scn_SpatialComponents.js';

import {
  registerWeatherZone,
  WeatherZone,
  ClimateState,
  applyBiomeClimate,
  BIOME as WEATHER_BIOME,
} from './031_scn_WeatherComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of registered biome definitions.
 */
export const MAX_BIOME_DEFS = 32;

/**
 * Maximum number of simultaneous biome-aware entities. Every registered
 * entity consumes one biome slot. Bounded by MAX_ENTITIES.
 */
export const MAX_BIOME_ENTITIES = MAX_ENTITIES;

/**
 * Maximum number of biome weights blended at once (used by the palette
 * blender).
 */
export const MAX_BIOME_WEIGHTS = 4;

/**
 * Maximum number of scatter kinds a biome can support (trees, grass,
 * flowers, rocks, pebbles, moss, snow tufts, ice shards, coral, etc.).
 */
export const MAX_SCATTER_KINDS = 16;

/**
 * Default biome transition duration in seconds.
 */
export const DEFAULT_BIOME_TRANSITION_DURATION = 2.5;

/**
 * Default procedural terrain size (world units).
 */
export const DEFAULT_WORLD_SIZE = 512.0;

/**
 * Default procedural terrain resolution (grid cells per axis).
 */
export const DEFAULT_WORLD_RESOLUTION =
  PERF_TIER_LOCAL === 'HIGH'   ? 128 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 96 :
                                 64;

/**
 * Default moisture map resolution.
 */
export const DEFAULT_MOISTURE_RESOLUTION = 64;

/**
 * Default biome blend radius (world units) — used by vegetation scatter
 * to determine how strongly a neighboring biome influences a placement.
 */
export const DEFAULT_BIOME_BLEND_RADIUS = 24.0;

/**
 * Canonical biome ids. These match the SLOT enum used by world.js so
 * the palette blend stays consistent across the engine.
 */
export const BIOME = Object.freeze({
  DESERT:   0,
  SNOW:     1,
  SEA:      2,
  FOREST:   3,
  CANYON:   4,
  COASTAL:  5,
  WETLAND:  6,
  TUNDRA:   7,
  VOLCANIC: 8,
  COUNT:    9,
});

export const BIOME_NAME = Object.freeze([
  'desert', 'snow', 'sea', 'forest', 'canyon',
  'coastal', 'wetland', 'tundra', 'volcanic',
]);

/**
 * Biome transition states.
 */
export const BIOME_TRANSITION_STATE = Object.freeze({
  IDLE:       0,
  STARTING:   1,
  TRANSITION: 2,
  SETTLING:   3,
  COMPLETE:   4,
  CANCELLED:  5,
  COUNT:      6,
});

export const BIOME_TRANSITION_STATE_NAME = Object.freeze([
  'idle', 'starting', 'transition', 'settling', 'complete', 'cancelled',
]);

/**
 * Scatter kinds — used by the procedural scatter pass.
 */
export const SCATTER_KIND = Object.freeze({
  NONE:        0,
  TREE_LARGE:  1,
  TREE_MEDIUM: 2,
  TREE_SMALL:  3,
  BUSH:        4,
  GRASS:       5,
  FLOWER:      6,
  ROCK_LARGE:  7,
  ROCK_MEDIUM: 8,
  ROCK_SMALL:  9,
  PEBBLE:     10,
  MOSS:       11,
  SNOW_TUFT:  12,
  ICE_SHARD:  13,
  CORAL:      14,
  SHELL:      15,
  COUNT:      16,
});

export const SCATTER_KIND_NAME = Object.freeze([
  'none',
  'tree_large', 'tree_medium', 'tree_small',
  'bush', 'grass', 'flower',
  'rock_large', 'rock_medium', 'rock_small', 'pebble',
  'moss', 'snow_tuft', 'ice_shard', 'coral', 'shell',
]);

/**
 * Biome flags.
 */
export const BIOME_FLAG = Object.freeze({
  NONE:              0,
  ENABLED:           1 << 0,
  IS_AQUATIC:        1 << 1,
  IS_FROZEN:         1 << 2,
  IS_ARID:           1 << 3,
  IS_TROPICAL:       1 << 4,
  HAS_VEGETATION:    1 << 5,
  HAS_SCATTER:       1 << 6,
  HAS_SNOW:          1 << 7,
  HAS_SAND:          1 << 8,
  HAS_GRASS:         1 << 9,
  HAS_ROCK:          1 << 10,
  SUPPORTS_LOD:      1 << 11,
});

/**
 * World generation progress phases.
 */
export const WORLD_GEN_PHASE = Object.freeze({
  IDLE:        0,
  HEIGHTS:     1,
  BIOMES:      2,
  MOISTURE:    3,
  COLORS:      4,
  SCATTER:     5,
  COMPLETE:    6,
  FAILED:      7,
  COUNT:       8,
});

export const WORLD_GEN_PHASE_NAME = Object.freeze([
  'idle', 'heights', 'biomes', 'moisture', 'colors', 'scatter',
  'complete', 'failed',
]);

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const BiomeState = {
  frame:                    0,
  activeBiomeEntities:      0,
  peakBiomeEntities:        0,
  totalRegistrations:       0,
  totalUnregistrations:     0,
  totalSetBiome:            0,
  totalTransitions:         0,
  totalTransitionTicks:     0,
  totalInstantSnape:        0,
  totalBlends:              0,
  totalApplyToEntity:       0,
  totalApplyToAll:          0,
  totalWorldGenPasses:      0,
  totalChunksGenerated:     0,
  totalScatterPlacements:   0,
  lastTickMs:               0,
  avgTickMs:                0,
  lastBlendMs:              0,
  avgBlendMs:               0,
  lastWorldGenMs:           0,
  avgWorldGenMs:            0,
  globalBiomeWeights:       new Float32Array(BIOME.COUNT),
  globalDominantBiome:      BIOME.DESERT,
  globalTargetBiome:        BIOME.DESERT,
  worldGenPhase:            WORLD_GEN_PHASE.IDLE,
  worldGenProgress:         0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.biome', {
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

function _clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function _clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function _lerp(a, b, t) { return a + (b - a) * t; }
function _smoothstep(e0, e1, x) {
  const t = _clamp01((x - e0) / (e1 - e0 || 1e-6));
  return t * t * (3 - 2 * t);
}
function _smootherstep(e0, e1, x) {
  const t = _clamp01((x - e0) / (e1 - e0 || 1e-6));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * BiomeStateComp — per-entity biome state and dominant biome.
 */
export const BiomeStateComp = {
  enabled:            new Uint8Array(MAX_ENTITIES),
  dominantBiome:      new Uint8Array(MAX_ENTITIES),
  targetBiome:        new Uint8Array(MAX_ENTITIES),
  prevBiome:          new Uint8Array(MAX_ENTITIES),
  transitionState:    new Uint8Array(MAX_ENTITIES),
  flags:              new Uint16Array(MAX_ENTITIES),
  transitionTimer:    new Float32Array(MAX_ENTITIES),
  transitionDuration: new Float32Array(MAX_ENTITIES).fill(DEFAULT_BIOME_TRANSITION_DURATION),
  lastChangeFrame:    new Uint32Array(MAX_ENTITIES),
  changeCount:        new Uint16Array(MAX_ENTITIES),
  priority:           new Uint8Array(MAX_ENTITIES),
};

/**
 * BiomeWeights — one weight per biome id per entity. Flat layout:
 *   weights[eid * BIOME.COUNT + biomeId]
 */
export const BiomeWeights = {
  weights:            new Float32Array(MAX_ENTITIES * BIOME.COUNT),
  targetWeights:      new Float32Array(MAX_ENTITIES * BIOME.COUNT),
  prevWeights:        new Float32Array(MAX_ENTITIES * BIOME.COUNT),
};

/**
 * BiomeTransition — active transition record per entity.
 */
export const BiomeTransition = {
  fromBiome:          new Uint8Array(MAX_ENTITIES),
  toBiome:            new Uint8Array(MAX_ENTITIES),
  elapsed:            new Float32Array(MAX_ENTITIES),
  duration:           new Float32Array(MAX_ENTITIES),
  progress:           new Float32Array(MAX_ENTITIES),
  easing:             new Uint8Array(MAX_ENTITIES),   // 0=linear 1=smooth 2=smoother
  cancelFlag:         new Uint8Array(MAX_ENTITIES),
  active:             new Uint8Array(MAX_ENTITIES),
};

/**
 * BiomePalette — per-entity resolved palette. Contains the blended RGB
 * (linear) for the sky, fog, ground, water, rock, snow, ice, accent,
 * etc. Layout: palette[eid * PALETTE_SLOT_COUNT * 3 + slot * 3 + rgb].
 */
export const PALETTE_SLOT_COUNT = 18;
export const BiomePalette = {
  palette:            new Float32Array(MAX_ENTITIES * PALETTE_SLOT_COUNT * 3),
  dirty:              new Uint8Array(MAX_ENTITIES).fill(1),
  generation:         new Uint32Array(MAX_ENTITIES),
};

/**
 * BiomeClimateBind — per-entity binding to a weather zone, temperature
 * offsets driven by the biome.
 */
export const BiomeClimateBind = {
  weatherZoneEid:     new Int32Array(MAX_ENTITIES).fill(-1),
  tempOffsetC:        new Float32Array(MAX_ENTITIES),
  humidityOffset:     new Float32Array(MAX_ENTITIES),
  windBias:           new Float32Array(MAX_ENTITIES),
  precipBias:         new Float32Array(MAX_ENTITIES),
  fogBias:            new Float32Array(MAX_ENTITIES),
  heatBias:           new Float32Array(MAX_ENTITIES),
  bound:              new Uint8Array(MAX_ENTITIES),
};

/**
 * BiomeMaterialOverride — per-entity material profile driven by the biome.
 * Kinds: 0=none 1=sand 2=snow 3=ice 4=rock 5=grass 6=foliage 7=bark
 *        8=soil 9=volcanic 10=metal 11=water 12=corals.
 */
export const BiomeMaterialOverride = {
  groundKind:         new Uint8Array(MAX_ENTITIES),
  rockKind:           new Uint8Array(MAX_ENTITIES),
  foliageKind:        new Uint8Array(MAX_ENTITIES),
  waterKind:          new Uint8Array(MAX_ENTITIES),
  snowBlend:          new Float32Array(MAX_ENTITIES),
  sandBlend:          new Float32Array(MAX_ENTITIES),
  iceBlend:           new Float32Array(MAX_ENTITIES),
  grassBlend:         new Float32Array(MAX_ENTITIES),
  rockBlend:          new Float32Array(MAX_ENTITIES),
  emissionBoost:      new Float32Array(MAX_ENTITIES),
};

/**
 * BiomeVegetationState — per-entity scatter density and kind weights.
 */
export const BiomeVegetationState = {
  scatterDensity:     new Float32Array(MAX_ENTITIES),
  targetScatterDensity:new Float32Array(MAX_ENTITIES),
  treeWeight:         new Float32Array(MAX_ENTITIES),
  bushWeight:         new Float32Array(MAX_ENTITIES),
  grassWeight:        new Float32Array(MAX_ENTITIES),
  flowerWeight:       new Float32Array(MAX_ENTITIES),
  rockWeight:         new Float32Array(MAX_ENTITIES),
  mossWeight:         new Float32Array(MAX_ENTITIES),
  snowWeight:         new Float32Array(MAX_ENTITIES),
  iceWeight:          new Float32Array(MAX_ENTITIES),
  shellWeight:        new Float32Array(MAX_ENTITIES),
  coralWeight:        new Float32Array(MAX_ENTITIES),
  scatterBudget:      new Float32Array(MAX_ENTITIES),
};

/**
 * BiomeScatterBudget — per-frame global scatter budget.
 */
export const BiomeScatterBudget = {
  totalScatterCost:   new Float32Array(1),
  scatterBudgetCap:   new Float32Array(1).fill(8.0),
  budgetExceeded:     new Uint8Array(1),
  totalPlacements:    new Uint32Array(1),
  totalTrees:         new Uint32Array(1),
  totalGrass:         new Uint32Array(1),
  totalRocks:         new Uint32Array(1),
  totalFoliage:       new Uint32Array(1),
};

/**
 * BiomeZoneBlend — per-entity world-space blend radius so biomes cross
 * fade across chunk boundaries.
 */
export const BiomeZoneBlend = {
  blendRadius:        new Float32Array(MAX_ENTITIES).fill(DEFAULT_BIOME_BLEND_RADIUS),
  blendSoftness:      new Float32Array(MAX_ENTITIES).fill(0.5),
  neighborMask:       new Uint32Array(MAX_ENTITIES),
  dominantBlend:      new Float32Array(MAX_ENTITIES),
};

/**
 * BiomeStats — per-frame aggregate statistics.
 */
export const BiomeStats = {
  activeEntities:     new Uint32Array(1),
  transitioning:      new Uint32Array(1),
  dominantCounts:     new Uint32Array(BIOME.COUNT),
  averageTransitionT: new Float32Array(1),
  lastWorldGenMs:     new Float32Array(1),
  lastScatterCost:    new Float32Array(1),
};

/**
 * Biome component bundle for bitECS createWorld.
 */
export const BIOME_COMPONENTS = Object.freeze({
  BiomeStateComp,
  BiomeWeights,
  BiomeTransition,
  BiomePalette,
  BiomeClimateBind,
  BiomeMaterialOverride,
  BiomeVegetationState,
  BiomeScatterBudget,
  BiomeZoneBlend,
  BiomeStats,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

const _scratchV3 = new Float32Array(3);
const _scratchV3B = new Float32Array(3);
const _scratchRGB = new Float32Array(3);
const _scratchRGB2 = new Float32Array(3);
const _scratchWeights = new Float32Array(BIOME.COUNT);

/* ------------------------------------------------------------------ */
/* 4. COLOR SPACE HELPERS (Linear RGB ↔ Oklab)                        */
/* ------------------------------------------------------------------ */

function _srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function _linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}
function _hexToLinear(hex) {
  const r = ((hex >> 16) & 255) / 255;
  const g = ((hex >>  8) & 255) / 255;
  const b = ( hex        & 255) / 255;
  return [_srgbToLinear(r), _srgbToLinear(g), _srgbToLinear(b)];
}
function _linearRGBToOklab(r, g, b, out) {
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
  const lp = Math.cbrt(l);
  const mp = Math.cbrt(m);
  const sp = Math.cbrt(s);
  out[0] = 0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp;
  out[1] = 1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp;
  out[2] = 0.0259040371 * lp + 0.7827717662 * mp - 0.8086758033 * sp;
  return out;
}
function _oklabToLinearRGB(L, a, b, out) {
  const lp = L + 0.3963377774 * a + 0.2158037573 * b;
  const mp = L - 0.1055613458 * a - 0.0638541728 * b;
  const sp = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = lp * lp * lp;
  const m = mp * mp * mp;
  const s = sp * sp * sp;
  let r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  out[0] = _clamp01(r);
  out[1] = _clamp01(g);
  out[2] = _clamp01(bl);
  return out;
}
function _blendOklab(out, a, b, t) {
  const L = a[0] + (b[0] - a[0]) * t;
  const A = a[1] + (b[1] - a[1]) * t;
  const B = a[2] + (b[2] - a[2]) * t;
  return _oklabToLinearRGB(L, A, B, out);
}

/* ------------------------------------------------------------------ */
/* 5. BIOME DEFINITION REGISTRY                                       */
/* ------------------------------------------------------------------ */

/**
 * A biome definition is a frozen descriptor with:
 *   • palette  — 18 linear-RGB colors matching PALETTE_SLOT_COUNT
 *   • climate  — temperature / humidity / wind biases
 *   • scatter  — per-kind density multipliers
 *   • material — material kind overrides
 *   • flags    — BIOME_FLAG bitmask
 */
class BiomeDefinition {
  constructor(spec) {
    this.id          = spec.id | 0;
    this.name        = spec.name || BIOME_NAME[spec.id] || `biome_${spec.id}`;
    this.flags       = spec.flags !== undefined ? spec.flags | 0 : BIOME_FLAG.ENABLED;

    this.paletteLinear = new Float32Array(PALETTE_SLOT_COUNT * 3);
    if (spec.paletteHex) {
      for (let i = 0; i < PALETTE_SLOT_COUNT; i++) {
        const h = spec.paletteHex[i] !== undefined ? spec.paletteHex[i] : 0x808080;
        const l = _hexToLinear(h);
        this.paletteLinear[i * 3 + 0] = l[0];
        this.paletteLinear[i * 3 + 1] = l[1];
        this.paletteLinear[i * 3 + 2] = l[2];
      }
    }

    this.paletteOklab = new Float32Array(PALETTE_SLOT_COUNT * 3);
    for (let i = 0; i < PALETTE_SLOT_COUNT; i++) {
      _linearRGBToOklab(
        this.paletteLinear[i * 3 + 0],
        this.paletteLinear[i * 3 + 1],
        this.paletteLinear[i * 3 + 2],
        _scratchRGB
      );
      this.paletteOklab[i * 3 + 0] = _scratchRGB[0];
      this.paletteOklab[i * 3 + 1] = _scratchRGB[1];
      this.paletteOklab[i * 3 + 2] = _scratchRGB[2];
    }

    // Climate modifiers.
    this.baseTempC     = spec.baseTempC     !== undefined ? spec.baseTempC : 15;
    this.humidityBias  = spec.humidityBias  !== undefined ? spec.humidityBias : 0;
    this.windBias      = spec.windBias      !== undefined ? spec.windBias : 0;
    this.precipBias    = spec.precipBias    !== undefined ? spec.precipBias : 0;
    this.fogBias       = spec.fogBias       !== undefined ? spec.fogBias : 0;
    this.heatBias      = spec.heatBias      !== undefined ? spec.heatBias : 0;

    // Scatter densities (per SCATTER_KIND).
    this.scatterDensity = new Float32Array(SCATTER_KIND.COUNT);
    if (spec.scatterDensity) {
      for (let k = 0; k < SCATTER_KIND.COUNT; k++) {
        this.scatterDensity[k] = spec.scatterDensity[k] !== undefined ? spec.scatterDensity[k] : 0;
      }
    }
    this.overallScatterDensity = spec.overallScatterDensity !== undefined ? spec.overallScatterDensity : 0.5;

    // Material overrides.
    this.groundKind   = spec.groundKind   !== undefined ? spec.groundKind : 0;
    this.rockKind     = spec.rockKind     !== undefined ? spec.rockKind   : 0;
    this.foliageKind  = spec.foliageKind  !== undefined ? spec.foliageKind : 0;
    this.waterKind    = spec.waterKind    !== undefined ? spec.waterKind   : 0;

    // Material blends.
    this.snowBlend    = spec.snowBlend   !== undefined ? spec.snowBlend : 0;
    this.sandBlend    = spec.sandBlend   !== undefined ? spec.sandBlend : 0;
    this.iceBlend     = spec.iceBlend    !== undefined ? spec.iceBlend : 0;
    this.grassBlend   = spec.grassBlend  !== undefined ? spec.grassBlend : 0;
    this.rockBlend    = spec.rockBlend   !== undefined ? spec.rockBlend : 0;
    this.emissionBoost= spec.emissionBoost !== undefined ? spec.emissionBoost : 0;

    Object.freeze(this);
  }
}

const _biomeRegistry = new Array(MAX_BIOME_DEFS).fill(null);
const _biomeByName = new Map();

/**
 * Registers a biome definition. Returns true on success.
 */
export function registerBiomeDefinition(spec) {
  if (!spec || typeof spec.id !== 'number') return false;
  const id = spec.id | 0;
  if (id < 0 || id >= MAX_BIOME_DEFS) return false;
  if (_biomeRegistry[id]) return true;   // already registered

  const def = new BiomeDefinition(spec);
  _biomeRegistry[id] = def;
  _biomeByName.set(def.name, id);
  return true;
}

export function getBiomeDefinition(id) {
  if (typeof id !== 'number' || id < 0 || id >= MAX_BIOME_DEFS) return null;
  return _biomeRegistry[id];
}

export function getBiomeDefinitionCount() {
  let n = 0;
  for (let i = 0; i < MAX_BIOME_DEFS; i++) if (_biomeRegistry[i]) n++;
  return n;
}

/* ------------------------------------------------------------------ */
/* 6. CANONICAL BIOME PALETTES                                        */
/* ------------------------------------------------------------------ */

(function _registerCanonicalBiomes() {
  // Palette slots are:
  //   0 sky, 1 fog, 2 ground, 3 sun, 4 shadow, 5 ambient,
  //   6 water_deep, 7 water_mid, 8 water_shallow, 9 foam,
  //   10 rock_deep, 11 rock_dark, 12 rock_mid, 13 rock_light,
  //   14 snow, 15 snow_shadow, 16 ice, 17 accent

  registerBiomeDefinition({
    id: BIOME.DESERT,
    name: 'desert',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_ARID | BIOME_FLAG.HAS_SAND | BIOME_FLAG.HAS_ROCK | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x7fb2d9, 0xedcc9e, 0x8b6844, 0xfff0d0, 0x5a4030, 0xd7b98f,
      0x1e5f6b, 0x2f8f8f, 0x58c7b5, 0xf7fbff,
      0x3c2818, 0x6b472b, 0x9e734a, 0xdeb885,
      0xf5ead8, 0xc9b39a, 0x8fc7b8, 0xb89469,
    ],
    baseTempC: 30, humidityBias: -0.30, windBias: 0.15, precipBias: -0.35, fogBias: -0.25, heatBias: 0.45,
    groundKind: 1, rockKind: 4, foliageKind: 0, waterKind: 0,
    snowBlend: 0.0, sandBlend: 0.85, iceBlend: 0.0, grassBlend: 0.02, rockBlend: 0.6, emissionBoost: 0.0,
    overallScatterDensity: 0.35,
    scatterDensity: [
      0, 0, 0, 0, 0.05, 0.02, 0, 0.08, 0.35, 0.9, 0.1, 0, 0, 0, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.SNOW,
    name: 'snow',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_FROZEN | BIOME_FLAG.HAS_SNOW | BIOME_FLAG.HAS_ROCK | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x9fc7ef, 0xc7dff2, 0x2e3e4f, 0xf7fbff, 0x162436, 0x93b7d9,
      0x0e4554, 0x1c8f8f, 0x4ad1bd, 0xf0fbff,
      0x131b25, 0x253242, 0x415468, 0x6e859c,
      0xf5faff, 0xb0c9e6, 0x4ad9b8, 0x8fb3db,
    ],
    baseTempC: -8, humidityBias: 0.05, windBias: 0.20, precipBias: 0.10, fogBias: 0.15, heatBias: -0.35,
    groundKind: 2, rockKind: 4, foliageKind: 0, waterKind: 3,
    snowBlend: 0.85, sandBlend: 0.0, iceBlend: 0.55, grassBlend: 0.0, rockBlend: 0.45, emissionBoost: 0.1,
    overallScatterDensity: 0.30,
    scatterDensity: [
      0, 0.05, 0.15, 0.3, 0.05, 0.05, 0, 0.1, 0.35, 0.85, 0.2, 0.1, 0.8, 0.5, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.SEA,
    name: 'sea',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_AQUATIC | BIOME_FLAG.IS_TROPICAL | BIOME_FLAG.HAS_VEGETATION | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x9fd7e8, 0xbdeef2, 0x063049, 0xfff9e8, 0x04283a, 0x86cfdc,
      0x063049, 0x0e7a8b, 0x3cd1c2, 0xf6feff,
      0x182330, 0x2e3b48, 0x6e6a62, 0xc7a178,
      0xf8fbff, 0xd7e5ef, 0x4fe8cc, 0xc7a178,
    ],
    baseTempC: 22, humidityBias: 0.30, windBias: 0.25, precipBias: 0.15, fogBias: 0.25, heatBias: -0.05,
    groundKind: 11, rockKind: 4, foliageKind: 12, waterKind: 11,
    snowBlend: 0.0, sandBlend: 0.35, iceBlend: 0.0, grassBlend: 0.05, rockBlend: 0.5, emissionBoost: 0.05,
    overallScatterDensity: 0.45,
    scatterDensity: [
      0.05, 0.1, 0.2, 0.3, 0.2, 0.15, 0.1, 0.15, 0.35, 0.5, 0.25, 0.1, 0, 0, 0.6, 0.5,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.FOREST,
    name: 'forest',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.HAS_VEGETATION | BIOME_FLAG.HAS_GRASS | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x88c8e8, 0xc8e4f0, 0x4a7a3a, 0xfff6dc, 0x2a3f2a, 0x8fb878,
      0x0a3040, 0x2a6a4a, 0x3cd0b0, 0xf0fffa,
      0x2a3a28, 0x4a5a42, 0x6a7a58, 0x9ea880,
      0xf5faff, 0xb0c9e6, 0x4ad9b8, 0xa8d878,
    ],
    baseTempC: 15, humidityBias: 0.25, windBias: -0.10, precipBias: 0.20, fogBias: 0.20, heatBias: -0.10,
    groundKind: 5, rockKind: 4, foliageKind: 6, waterKind: 0,
    snowBlend: 0.0, sandBlend: 0.05, iceBlend: 0.0, grassBlend: 0.85, rockBlend: 0.35, emissionBoost: 0.0,
    overallScatterDensity: 0.95,
    scatterDensity: [
      0.8, 0.9, 0.85, 0.7, 0.9, 0.4, 0.15, 0.2, 0.3, 0.35, 0.15, 0.3, 0.1, 0.0, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.CANYON,
    name: 'canyon',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_ARID | BIOME_FLAG.HAS_ROCK | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0xd8a878, 0xe8b890, 0x8a5838, 0xffd870, 0x3a2010, 0xb08858,
      0x1a3040, 0x2a5060, 0x4a9080, 0xf0f8f0,
      0x4a2810, 0x6a4028, 0xa0704a, 0xd8a878,
      0xf5ead8, 0xc9b39a, 0x8fc7b8, 0xb89469,
    ],
    baseTempC: 25, humidityBias: -0.15, windBias: 0.20, precipBias: -0.10, fogBias: 0.00, heatBias: 0.25,
    groundKind: 1, rockKind: 4, foliageKind: 0, waterKind: 0,
    snowBlend: 0.0, sandBlend: 0.6, iceBlend: 0.0, grassBlend: 0.05, rockBlend: 0.9, emissionBoost: 0.0,
    overallScatterDensity: 0.40,
    scatterDensity: [
      0, 0.05, 0.1, 0.15, 0.1, 0.05, 0, 0.3, 0.7, 0.9, 0.3, 0.05, 0, 0, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.COASTAL,
    name: 'coastal',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.HAS_SAND | BIOME_FLAG.HAS_VEGETATION | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0xa8d8e8, 0xd8e8f0, 0xd8b878, 0xfff0c8, 0x4a3828, 0xc8b890,
      0x1a5a6a, 0x2a8a9a, 0x5cd0c0, 0xf8ffff,
      0x3a3028, 0x5a4a3a, 0x8a7a58, 0xb8a878,
      0xf5faff, 0xb0c9e6, 0x4ad9b8, 0xa8d878,
    ],
    baseTempC: 20, humidityBias: 0.35, windBias: 0.25, precipBias: 0.25, fogBias: 0.30, heatBias: 0.0,
    groundKind: 1, rockKind: 4, foliageKind: 6, waterKind: 11,
    snowBlend: 0.0, sandBlend: 0.9, iceBlend: 0.0, grassBlend: 0.15, rockBlend: 0.4, emissionBoost: 0.0,
    overallScatterDensity: 0.55,
    scatterDensity: [
      0.05, 0.1, 0.2, 0.3, 0.2, 0.15, 0.05, 0.15, 0.4, 0.6, 0.35, 0.15, 0, 0, 0.3, 0.6,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.WETLAND,
    name: 'wetland',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_AQUATIC | BIOME_FLAG.HAS_VEGETATION | BIOME_FLAG.HAS_GRASS | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x88b8c8, 0xa8d0d8, 0x3a5a38, 0xf0e8c8, 0x1a3a28, 0x78a878,
      0x0a2840, 0x1a5a5a, 0x3cb0a8, 0xe8f8f0,
      0x2a2820, 0x4a4030, 0x6a6850, 0x9a9088,
      0xf5faff, 0xb0c9e6, 0x4ad9b8, 0x78c878,
    ],
    baseTempC: 22, humidityBias: 0.45, windBias: -0.15, precipBias: 0.35, fogBias: 0.40, heatBias: 0.05,
    groundKind: 8, rockKind: 4, foliageKind: 6, waterKind: 11,
    snowBlend: 0.0, sandBlend: 0.15, iceBlend: 0.0, grassBlend: 0.9, rockBlend: 0.25, emissionBoost: 0.0,
    overallScatterDensity: 0.80,
    scatterDensity: [
      0.3, 0.5, 0.6, 0.7, 0.8, 0.95, 0.15, 0.1, 0.25, 0.4, 0.2, 0.5, 0, 0, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.TUNDRA,
    name: 'tundra',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_FROZEN | BIOME_FLAG.HAS_SNOW | BIOME_FLAG.HAS_GRASS | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0xa8c8e0, 0xc8dce8, 0x5a6a58, 0xf0f4f8, 0x2a3848, 0x98b0c0,
      0x0a3a4a, 0x1a6a6a, 0x4ac8b8, 0xf0ffff,
      0x2a3028, 0x3a4038, 0x5a6050, 0x8a9080,
      0xf8fbff, 0xb8cce0, 0x4ad9b8, 0x8ab8a8,
    ],
    baseTempC: -2, humidityBias: 0.10, windBias: 0.15, precipBias: 0.05, fogBias: 0.10, heatBias: -0.25,
    groundKind: 2, rockKind: 4, foliageKind: 5, waterKind: 3,
    snowBlend: 0.5, sandBlend: 0.0, iceBlend: 0.25, grassBlend: 0.6, rockBlend: 0.5, emissionBoost: 0.05,
    overallScatterDensity: 0.50,
    scatterDensity: [
      0.05, 0.15, 0.25, 0.4, 0.4, 0.7, 0.05, 0.15, 0.35, 0.7, 0.3, 0.4, 0.6, 0.3, 0, 0,
    ],
  });

  registerBiomeDefinition({
    id: BIOME.VOLCANIC,
    name: 'volcanic',
    flags: BIOME_FLAG.ENABLED | BIOME_FLAG.IS_ARID | BIOME_FLAG.HAS_ROCK | BIOME_FLAG.HAS_SCATTER | BIOME_FLAG.SUPPORTS_LOD,
    paletteHex: [
      0x602820, 0x804028, 0x2a1818, 0xff8050, 0x180808, 0x602010,
      0x180810, 0x3a1020, 0x6a2838, 0xe0a080,
      0x1a0a08, 0x2a1010, 0x4a2018, 0x7a3828,
      0xe8d8c8, 0xa89888, 0xff6040, 0xd87848,
    ],
    baseTempC: 42, humidityBias: -0.20, windBias: 0.30, precipBias: -0.15, fogBias: 0.30, heatBias: 0.70,
    groundKind: 9, rockKind: 4, foliageKind: 0, waterKind: 0,
    snowBlend: 0.0, sandBlend: 0.15, iceBlend: 0.0, grassBlend: 0.0, rockBlend: 0.85, emissionBoost: 0.85,
    overallScatterDensity: 0.45,
    scatterDensity: [
      0, 0, 0, 0.05, 0.05, 0.05, 0, 0.4, 0.8, 0.9, 0.35, 0.1, 0, 0, 0, 0,
    ],
  });
})();

/* ------------------------------------------------------------------ */
/* 7. PERLIN-STYLE NOISE (deterministic from seed)                    */
/* ------------------------------------------------------------------ */

function _hash2(x, y, seed) {
  let h = (x * 374761393 + y * 668265263 + seed * 1274126177) | 0;
  h = (h ^ (h >> 13)) | 0;
  h = Math.imul(h, 1274126177) | 0;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

function _valueNoise2D(x, y, seed) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = _hash2(ix,     iy,     seed);
  const b = _hash2(ix + 1, iy,     seed);
  const c = _hash2(ix,     iy + 1, seed);
  const d = _hash2(ix + 1, iy + 1, seed);
  const ab = a + (b - a) * ux;
  const cd = c + (d - c) * ux;
  return ab + (cd - ab) * uy;
}

function _fbm2D(x, y, octaves, lacunarity, gain, seed) {
  let sum = 0;
  let amp = 1;
  let freq = 1;
  let norm = 0;
  const oct = Math.max(1, octaves | 0);
  for (let i = 0; i < oct; i++) {
    sum += amp * _valueNoise2D(x * freq, y * freq, seed + i * 131);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return norm > 0 ? sum / norm : 0;
}

/* ------------------------------------------------------------------ */
/* 8. PROCEDURAL WORLD GENERATION                                     */
/* ------------------------------------------------------------------ */

/**
 * Per-cell scratch buffers for the world generation pass. Pre-allocated
 * so the pipeline is allocation-free.
 */
const _wgenHeight = new Float32Array(DEFAULT_WORLD_RESOLUTION * DEFAULT_WORLD_RESOLUTION);
const _wgenBiomeId = new Uint8Array(DEFAULT_WORLD_RESOLUTION * DEFAULT_WORLD_RESOLUTION);
const _wgenMoisture = new Float32Array(DEFAULT_WORLD_RESOLUTION * DEFAULT_WORLD_RESOLUTION);
const _wgenTemp = new Float32Array(DEFAULT_WORLD_RESOLUTION * DEFAULT_WORLD_RESOLUTION);

/**
 * Samples terrain height at world position (x, z). Deterministic from
 * seed. Uses a layered fBm with continentality, hills, and detail.
 *
 * Returns the height in world units.
 */
export function sampleTerrainHeight(x, z, seed) {
  const s = (seed | 0) >>> 0;
  const continentality = _fbm2D(x * 0.005 + s * 0.0001, z * 0.005 + s * 0.0002, 4, 2.0, 0.5, s + 11);
  const hills = _fbm2D(x * 0.035 + s * 0.0003, z * 0.035 + s * 0.0004, 3, 2.0, 0.5, s + 37);
  const detail = _fbm2D(x * 0.14 + s * 0.0007, z * 0.14 + s * 0.0009, 2, 2.0, 0.5, s + 71);
  return continentality * 24.0 + hills * 6.0 + detail * 1.2 - 6.0;
}

/**
 * Samples the analytic terrain normal at (x, z). Uses central differences.
 * Writes into out[0..2].
 */
export function sampleTerrainNormal(x, z, seed, eps, out) {
  const e = eps !== undefined ? eps : 0.5;
  const hL = sampleTerrainHeight(x - e, z, seed);
  const hR = sampleTerrainHeight(x + e, z, seed);
  const hD = sampleTerrainHeight(x, z - e, seed);
  const hU = sampleTerrainHeight(x, z + e, seed);
  const nx = (hL - hR) / (2 * e);
  const nz = (hD - hU) / (2 * e);
  const ny = 1.0;
  const inv = 1.0 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
  out[0] = nx * inv;
  out[1] = ny * inv;
  out[2] = nz * inv;
  return out;
}

/**
 * Computes biome weights from a world position and the current global
 * biome weights. Writes into out[0..BIOME.COUNT-1] and normalizes.
 */
export function sampleBiomeWeights(x, z, seed, out) {
  const s = (seed | 0) >>> 0;
  const h = sampleTerrainHeight(x, z, seed);
  const elevation = _clamp01((h + 10) / 40);
  const moisture = _fbm2D(x * 0.02 + s * 0.001, z * 0.02 + s * 0.002, 3, 2.0, 0.5, s + 101);
  const temperature = _fbm2D(x * 0.016 + s * 0.003, z * 0.016 + s * 0.004, 2, 2.0, 0.5, s + 137);

  const elevN = elevation;
  const moistN = _clamp01(moisture * 0.5 + 0.5);
  const tempN = _clamp01(temperature * 0.5 + 0.5);

  // Compute per-biome affinity.
  let desert = 0, snow = 0, sea = 0, forest = 0, canyon = 0;
  let coastal = 0, wetland = 0, tundra = 0, volcanic = 0;

  // Elevation classification.
  if (elevN < 0.15) {
    sea     = _smoothstep(0.15, 0.0, elevN) * 0.9;
    coastal = _smoothstep(0.15, 0.05, elevN) * 0.6;
  } else if (elevN < 0.35) {
    coastal = _smoothstep(0.35, 0.15, elevN) * 0.4;
  } else if (elevN > 0.75) {
    snow  = _smoothstep(0.75, 1.0, elevN) * 0.8;
    canyon = _smoothstep(0.75, 0.95, elevN) * 0.4;
  }

  // Temperature classification.
  if (tempN < 0.25) {
    snow   += (1 - tempN / 0.25) * 0.7;
    tundra += (1 - tempN / 0.25) * 0.6;
  } else if (tempN > 0.75) {
    desert += (tempN - 0.75) / 0.25 * 0.6;
    volcanic += (tempN - 0.85) / 0.15 * 0.3;
  }

  // Moisture classification.
  if (moistN < 0.25) {
    desert += (0.25 - moistN) / 0.25 * 0.6;
    canyon += (0.25 - moistN) / 0.25 * 0.4;
  } else if (moistN > 0.75) {
    forest  += (moistN - 0.75) / 0.25 * 0.7;
    wetland += (moistN - 0.85) / 0.15 * 0.5;
  }

  // Mid-range default forest.
  if (elevN > 0.3 && elevN < 0.7 && moistN > 0.4 && moistN < 0.75) {
    forest += 0.5;
  }

  // Blend with global bias from BiomeState.
  desert   = desert   * 0.7 + BiomeState.globalBiomeWeights[BIOME.DESERT] * 0.3;
  snow     = snow     * 0.7 + BiomeState.globalBiomeWeights[BIOME.SNOW] * 0.3;
  sea      = sea      * 0.7 + BiomeState.globalBiomeWeights[BIOME.SEA] * 0.3;
  forest   = forest   * 0.7 + BiomeState.globalBiomeWeights[BIOME.FOREST] * 0.3;
  canyon   = canyon   * 0.7 + BiomeState.globalBiomeWeights[BIOME.CANYON] * 0.3;
  coastal  = coastal  * 0.7 + BiomeState.globalBiomeWeights[BIOME.COASTAL] * 0.3;
  wetland  = wetland  * 0.7 + BiomeState.globalBiomeWeights[BIOME.WETLAND] * 0.3;
  tundra   = tundra   * 0.7 + BiomeState.globalBiomeWeights[BIOME.TUNDRA] * 0.3;
  volcanic = volcanic * 0.7 + BiomeState.globalBiomeWeights[BIOME.VOLCANIC] * 0.3;

  // Sum and normalize.
  const sum = desert + snow + sea + forest + canyon + coastal + wetland + tundra + volcanic;
  if (sum <= 1e-6) {
    // Fallback: forest.
    for (let i = 0; i < BIOME.COUNT; i++) out[i] = 0;
    out[BIOME.FOREST] = 1;
  } else {
    out[BIOME.DESERT]   = desert   / sum;
    out[BIOME.SNOW]     = snow     / sum;
    out[BIOME.SEA]      = sea      / sum;
    out[BIOME.FOREST]   = forest   / sum;
    out[BIOME.CANYON]   = canyon   / sum;
    out[BIOME.COASTAL]  = coastal  / sum;
    out[BIOME.WETLAND]  = wetland  / sum;
    out[BIOME.TUNDRA]   = tundra   / sum;
    out[BIOME.VOLCANIC] = volcanic / sum;
  }
  return out;
}

/**
 * Returns the dominant biome id for the given weights.
 */
export function classifyBiome(weights) {
  if (!weights) return BIOME.FOREST;
  let best = 0;
  let bestW = weights[0];
  for (let i = 1; i < BIOME.COUNT; i++) {
    if (weights[i] > bestW) { bestW = weights[i]; best = i; }
  }
  return best;
}

/**
 * Samples the blended biome color at a world position. Uses Oklab
 * blending across all biome palettes and the given weight vector.
 *
 * Writes into out[0..2] (linear RGB).
 */
export function sampleBiomeColor(x, z, seed, weights, slot, out) {
  if (!weights) return out;
  const s = slot | 0;
  if (s < 0 || s >= PALETTE_SLOT_COUNT) return out;

  let L = 0, A = 0, B = 0;
  for (let i = 0; i < BIOME.COUNT; i++) {
    const w = weights[i];
    if (w <= 0.001) continue;
    const def = _biomeRegistry[i];
    if (!def) continue;
    const base = s * 3;
    L += def.paletteOklab[base + 0] * w;
    A += def.paletteOklab[base + 1] * w;
    B += def.paletteOklab[base + 2] * w;
  }

  return _oklabToLinearRGB(L, A, B, out);
}

/* ------------------------------------------------------------------ */
/* 9. CHUNK GENERATION                                                */
/* ------------------------------------------------------------------ */

/**
 * Generates a per-chunk height field into `outHeights` (Float32Array of
 * length (res+1)^2). Returns the number of cells written.
 */
export function generateChunkHeightField(chunkWorldX, chunkWorldZ, chunkSize, res, seed, outHeights) {
  if (!outHeights) return 0;
  const step = chunkSize / res;
  const half = chunkSize * 0.5;
  let write = 0;

  for (let j = 0; j <= res; j++) {
    const wz = chunkWorldZ - half + j * step;
    for (let i = 0; i <= res; i++) {
      const wx = chunkWorldX - half + i * step;
      outHeights[write++] = sampleTerrainHeight(wx, wz, seed);
    }
  }
  return write;
}

/**
 * Generates a per-chunk biome-id field into `outBiomeIds` (Uint8Array of
 * length (res+1)^2). Returns the number of cells written.
 */
export function generateChunkBiomeField(chunkWorldX, chunkWorldZ, chunkSize, res, seed, outBiomeIds) {
  if (!outBiomeIds) return 0;
  const step = chunkSize / res;
  const half = chunkSize * 0.5;
  let write = 0;

  for (let j = 0; j <= res; j++) {
    const wz = chunkWorldZ - half + j * step;
    for (let i = 0; i <= res; i++) {
      const wx = chunkWorldX - half + i * step;
      sampleBiomeWeights(wx, wz, seed, _scratchWeights);
      outBiomeIds[write++] = classifyBiome(_scratchWeights);
    }
  }
  return write;
}

/**
 * Evaluates the relief variance of a chunk from a height field. Higher
 * variance = more geometric detail needed = higher LOD priority.
 *
 * Returns the variance in world units squared.
 */
export function evaluateChunkRelief(heights, cellCount) {
  if (!heights || cellCount <= 0) return 0;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < cellCount; i++) {
    const h = heights[i];
    sum += h;
    sumSq += h * h;
  }
  const mean = sum / cellCount;
  return Math.max(0, sumSq / cellCount - mean * mean);
}

/**
 * Chooses a LOD level based on relief variance. Higher relief = lower
 * LOD index (higher detail).
 */
export function computeChunkLODFromRelief(reliefVariance, minLOD, maxLOD) {
  const lo = minLOD !== undefined ? minLOD : 0;
  const hi = maxLOD !== undefined ? maxLOD : 3;
  if (reliefVariance > 40) return lo;
  if (reliefVariance > 15) return Math.min(hi, lo + 1);
  if (reliefVariance > 4)  return Math.min(hi, lo + 2);
  return hi;
}

/* ------------------------------------------------------------------ */
/* 10. TERRAIN GEOMETRY BUILDER                                       */
/* ------------------------------------------------------------------ */

/**
 * Builds a Three.js BufferGeometry from a height field + biome color
 * field. The geometry is indexed and uses vertex colors.
 *
 * Returns a BufferGeometry, or null on failure.
 */
export function buildTerrainGeometry(heights, biomeIds, res, chunkSize, chunkWorldX, chunkWorldZ, seed) {
  if (!heights || !biomeIds) return null;
  if (res <= 0) return null;

  const vertexCount = (res + 1) * (res + 1);
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const step = chunkSize / res;
  const half = chunkSize * 0.5;

  for (let j = 0; j <= res; j++) {
    for (let i = 0; i <= res; i++) {
      const idx = j * (res + 1) + i;
      const i3 = idx * 3;
      const i2 = idx * 2;

      const lx = -half + i * step;
      const lz = -half + j * step;

      positions[i3 + 0] = lx;
      positions[i3 + 1] = heights[idx];
      positions[i3 + 2] = lz;

      // Biome color from the biome id.
      const biomeId = biomeIds[idx];
      const def = _biomeRegistry[biomeId];
      if (def) {
        // Ground slot is index 2 in the palette.
        colors[i3 + 0] = def.paletteLinear[2 * 3 + 0];
        colors[i3 + 1] = def.paletteLinear[2 * 3 + 1];
        colors[i3 + 2] = def.paletteLinear[2 * 3 + 2];
      } else {
        colors[i3 + 0] = 0.5;
        colors[i3 + 1] = 0.5;
        colors[i3 + 2] = 0.5;
      }

      uvs[i2 + 0] = i / res;
      uvs[i2 + 1] = j / res;
    }
  }

  const indexCount = res * res * 6;
  const indices = new Uint32Array(indexCount);
  let ptr = 0;
  for (let j = 0; j < res; j++) {
    for (let i = 0; i < res; i++) {
      const a = j * (res + 1) + i;
      const b = j * (res + 1) + i + 1;
      const c = (j + 1) * (res + 1) + i;
      const d = (j + 1) * (res + 1) + i + 1;
      indices[ptr++] = a; indices[ptr++] = c; indices[ptr++] = b;
      indices[ptr++] = b; indices[ptr++] = c; indices[ptr++] = d;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  return geo;
}

/**
 * Builds a per-vertex biome color attribute from a biome id field and a
 * weight field. Writes linear RGB into `outColors` (Float32Array of
 * length vertexCount * 3).
 */
export function buildBiomeColorAttribute(biomeIds, vertexCount, slot, outColors) {
  if (!biomeIds || !outColors) return 0;
  const s = slot | 0;
  const n = Math.min(vertexCount, biomeIds.length);

  for (let i = 0; i < n; i++) {
    const def = _biomeRegistry[biomeIds[i]];
    if (def) {
      outColors[i * 3 + 0] = def.paletteLinear[s * 3 + 0];
      outColors[i * 3 + 1] = def.paletteLinear[s * 3 + 1];
      outColors[i * 3 + 2] = def.paletteLinear[s * 3 + 2];
    } else {
      outColors[i * 3 + 0] = 0.5;
      outColors[i * 3 + 1] = 0.5;
      outColors[i * 3 + 2] = 0.5;
    }
  }
  return n;
}

/* ------------------------------------------------------------------ */
/* 11. FULL WORLD GENERATION PIPELINE                                 */
/* ------------------------------------------------------------------ */

/**
 * Generates a full procedural world in a single pass:
 *   1. Height field
 *   2. Biome id field
 *   3. Vertex color field
 *   4. Geometry
 *
 * Returns a `{ geometry, heights, biomeIds, colors, reliefVariance, lod }`
 * bundle, or null on failure.
 */
export function generateProceduralWorld(spec) {
  const t0 = _now();
  const s = spec || {};

  const size = s.size !== undefined ? Number(s.size) : DEFAULT_WORLD_SIZE;
  const res = s.resolution !== undefined ? (s.resolution | 0) : DEFAULT_WORLD_RESOLUTION;
  const seed = s.seed !== undefined ? s.seed | 0 : 1337;
  const worldX = s.worldX !== undefined ? Number(s.worldX) : 0;
  const worldZ = s.worldZ !== undefined ? Number(s.worldZ) : 0;

  if (res <= 0 || res > 512) return null;

  const vertexCount = (res + 1) * (res + 1);

  // Pre-allocate or reuse scratch. For larger resolutions we allocate
  // locally because the module scratch is sized to DEFAULT_WORLD_RESOLUTION.
  const heights = (res === DEFAULT_WORLD_RESOLUTION)
    ? _wgenHeight
    : new Float32Array(vertexCount);
  const biomeIds = (res === DEFAULT_WORLD_RESOLUTION)
    ? _wgenBiomeId
    : new Uint8Array(vertexCount);
  const colors = new Float32Array(vertexCount * 3);

  // Phase 1: heights.
  BiomeState.worldGenPhase = WORLD_GEN_PHASE.HEIGHTS;
  BiomeState.worldGenProgress = 0.1;
  generateChunkHeightField(worldX, worldZ, size, res, seed, heights);

  // Phase 2: biome ids.
  BiomeState.worldGenPhase = WORLD_GEN_PHASE.BIOMES;
  BiomeState.worldGenProgress = 0.35;
  generateChunkBiomeField(worldX, worldZ, size, res, seed, biomeIds);

  // Phase 3: colors (uses the ground palette slot).
  BiomeState.worldGenPhase = WORLD_GEN_PHASE.COLORS;
  BiomeState.worldGenProgress = 0.6;
  buildBiomeColorAttribute(biomeIds, vertexCount, 2, colors);

  // Phase 4: geometry.
  BiomeState.worldGenPhase = WORLD_GEN_PHASE.SCATTER;
  BiomeState.worldGenProgress = 0.85;
  const geo = buildTerrainGeometry(heights, biomeIds, res, size, worldX, worldZ, seed);

  // Relief + LOD evaluation.
  const reliefVariance = evaluateChunkRelief(heights, vertexCount);
  const lod = computeChunkLODFromRelief(reliefVariance);

  BiomeState.worldGenPhase = WORLD_GEN_PHASE.COMPLETE;
  BiomeState.worldGenProgress = 1.0;
  BiomeState.totalWorldGenPasses++;
  BiomeState.totalChunksGenerated++;

  const t1 = _now();
  BiomeState.lastWorldGenMs = t1 - t0;
  BiomeState.avgWorldGenMs += (BiomeState.lastWorldGenMs - BiomeState.avgWorldGenMs) * 0.15;
  BiomeStats.lastWorldGenMs[0] = BiomeState.lastWorldGenMs;

  return {
    geometry: geo,
    heights,
    biomeIds,
    colors,
    vertexCount,
    reliefVariance,
    lod,
    genMs: BiomeState.lastWorldGenMs,
  };
}

/* ------------------------------------------------------------------ */
/* 12. BIOME ENTITY REGISTRATION                                      */
/* ------------------------------------------------------------------ */

/**
 * Registers an entity as biome-aware. Allocates a biome state record.
 */
export function registerBiomeEntity(eid, spec) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const s = spec || {};
  const initialBiome = s.biome !== undefined ? (s.biome | 0) : BIOME.FOREST;

  BiomeStateComp.enabled[eid] = 1;
  BiomeStateComp.dominantBiome[eid] = initialBiome;
  BiomeStateComp.targetBiome[eid] = initialBiome;
  BiomeStateComp.prevBiome[eid] = initialBiome;
  BiomeStateComp.transitionState[eid] = BIOME_TRANSITION_STATE.IDLE;
  BiomeStateComp.flags[eid] = 0;
  BiomeStateComp.transitionTimer[eid] = 0;
  BiomeStateComp.transitionDuration[eid] = s.duration !== undefined
    ? Number(s.duration)
    : DEFAULT_BIOME_TRANSITION_DURATION;
  BiomeStateComp.lastChangeFrame[eid] = BiomeState.frame;
  BiomeStateComp.changeCount[eid] = 0;
  BiomeStateComp.priority[eid] = s.priority !== undefined ? s.priority | 0 : 2;

  // Zero the weights, then set the initial biome weight to 1.
  const wbase = eid * BIOME.COUNT;
  for (let b = 0; b < BIOME.COUNT; b++) {
    BiomeWeights.weights[wbase + b] = (b === initialBiome) ? 1.0 : 0.0;
    BiomeWeights.targetWeights[wbase + b] = BiomeWeights.weights[wbase + b];
    BiomeWeights.prevWeights[wbase + b] = BiomeWeights.weights[wbase + b];
  }

  BiomeTransition.active[eid] = 0;
  BiomeTransition.fromBiome[eid] = initialBiome;
  BiomeTransition.toBiome[eid] = initialBiome;
  BiomeTransition.elapsed[eid] = 0;
  BiomeTransition.duration[eid] = BiomeStateComp.transitionDuration[eid];
  BiomeTransition.progress[eid] = 1;
  BiomeTransition.easing[eid] = 1;
  BiomeTransition.cancelFlag[eid] = 0;

  BiomePalette.dirty[eid] = 1;
  BiomePalette.generation[eid] = 0;

  BiomeClimateBind.weatherZoneEid[eid] = s.weatherZoneEid !== undefined ? (s.weatherZoneEid | 0) : -1;
  BiomeClimateBind.tempOffsetC[eid] = 0;
  BiomeClimateBind.humidityOffset[eid] = 0;
  BiomeClimateBind.windBias[eid] = 0;
  BiomeClimateBind.precipBias[eid] = 0;
  BiomeClimateBind.fogBias[eid] = 0;
  BiomeClimateBind.heatBias[eid] = 0;
  BiomeClimateBind.bound[eid] = BiomeClimateBind.weatherZoneEid[eid] >= 0 ? 1 : 0;

  BiomeMaterialOverride.groundKind[eid] = 0;
  BiomeMaterialOverride.rockKind[eid] = 0;
  BiomeMaterialOverride.foliageKind[eid] = 0;
  BiomeMaterialOverride.waterKind[eid] = 0;
  BiomeMaterialOverride.snowBlend[eid] = 0;
  BiomeMaterialOverride.sandBlend[eid] = 0;
  BiomeMaterialOverride.iceBlend[eid] = 0;
  BiomeMaterialOverride.grassBlend[eid] = 0;
  BiomeMaterialOverride.rockBlend[eid] = 0;
  BiomeMaterialOverride.emissionBoost[eid] = 0;

  BiomeVegetationState.scatterDensity[eid] = 0;
  BiomeVegetationState.targetScatterDensity[eid] = 0;
  BiomeVegetationState.treeWeight[eid] = 0;
  BiomeVegetationState.bushWeight[eid] = 0;
  BiomeVegetationState.grassWeight[eid] = 0;
  BiomeVegetationState.flowerWeight[eid] = 0;
  BiomeVegetationState.rockWeight[eid] = 0;
  BiomeVegetationState.mossWeight[eid] = 0;
  BiomeVegetationState.snowWeight[eid] = 0;
  BiomeVegetationState.iceWeight[eid] = 0;
  BiomeVegetationState.shellWeight[eid] = 0;
  BiomeVegetationState.coralWeight[eid] = 0;
  BiomeVegetationState.scatterBudget[eid] = 0;

  BiomeZoneBlend.blendRadius[eid] = s.blendRadius !== undefined ? Number(s.blendRadius) : DEFAULT_BIOME_BLEND_RADIUS;
  BiomeZoneBlend.blendSoftness[eid] = s.blendSoftness !== undefined ? Number(s.blendSoftness) : 0.5;
  BiomeZoneBlend.neighborMask[eid] = 0;
  BiomeZoneBlend.dominantBlend[eid] = 0;

  BiomeState.totalRegistrations++;
  BiomeState.activeBiomeEntities++;
  if (BiomeState.activeBiomeEntities > BiomeState.peakBiomeEntities) {
    BiomeState.peakBiomeEntities = BiomeState.activeBiomeEntities;
  }

  return true;
}

/**
 * Unregisters an entity from the biome system.
 */
export function unregisterBiomeEntity(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (BiomeStateComp.enabled[eid] === 0) return false;

  BiomeStateComp.enabled[eid] = 0;
  BiomeState.activeBiomeEntities--;
  BiomeState.totalUnregistrations++;
  return true;
}

export function isBiomeEntity(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  return BiomeStateComp.enabled[eid] === 1;
}

/* ------------------------------------------------------------------ */
/* 13. DYNAMIC BIOME TRANSITIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * Installs a target biome on an entity. Starts a smooth transition.
 */
export function setBiome(eid, biomeId, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (biomeId < 0 || biomeId >= BIOME.COUNT) return false;
  if (BiomeStateComp.enabled[eid] === 0) return false;

  const opts = options || {};
  const duration = opts.duration !== undefined
    ? Math.max(0.0001, Number(opts.duration))
    : BiomeStateComp.transitionDuration[eid];
  const instant = opts.instant === true;

  const wbase = eid * BIOME.COUNT;
  for (let b = 0; b < BIOME.COUNT; b++) {
    BiomeWeights.prevWeights[wbase + b] = BiomeWeights.weights[wbase + b];
    BiomeWeights.targetWeights[wbase + b] = (b === biomeId) ? 1.0 : 0.0;
  }

  if (instant) {
    for (let b = 0; b < BIOME.COUNT; b++) {
      BiomeWeights.weights[wbase + b] = BiomeWeights.targetWeights[wbase + b];
    }
    BiomeStateComp.prevBiome[eid] = BiomeStateComp.dominantBiome[eid];
    BiomeStateComp.dominantBiome[eid] = biomeId;
    BiomeStateComp.targetBiome[eid] = biomeId;
    BiomeStateComp.transitionState[eid] = BIOME_TRANSITION_STATE.COMPLETE;
    BiomeTransition.active[eid] = 0;
    BiomePalette.dirty[eid] = 1;
    BiomeState.totalInstantSnape++;
    applyBiomeToEntity(eid);
    return true;
  }

  // Start a smooth transition.
  BiomeStateComp.prevBiome[eid] = BiomeStateComp.dominantBiome[eid];
  BiomeStateComp.targetBiome[eid] = biomeId;
  BiomeStateComp.transitionState[eid] = BIOME_TRANSITION_STATE.STARTING;
  BiomeStateComp.transitionTimer[eid] = 0;
  BiomeStateComp.transitionDuration[eid] = duration;
  BiomeStateComp.lastChangeFrame[eid] = BiomeState.frame;
  if (BiomeStateComp.changeCount[eid] < 0xFFFF) {
    BiomeStateComp.changeCount[eid]++;
  }

  BiomeTransition.fromBiome[eid] = BiomeStateComp.dominantBiome[eid];
  BiomeTransition.toBiome[eid] = biomeId;
  BiomeTransition.elapsed[eid] = 0;
  BiomeTransition.duration[eid] = duration;
  BiomeTransition.progress[eid] = 0;
  BiomeTransition.easing[eid] = opts.easing !== undefined ? (opts.easing | 0) : 1;
  BiomeTransition.cancelFlag[eid] = 0;
  BiomeTransition.active[eid] = 1;

  BiomeState.totalSetBiome++;
  BiomeState.totalTransitions++;

  return true;
}

/**
 * Cancels an in-flight biome transition.
 */
export function cancelBiomeTransition(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (BiomeTransition.active[eid] === 0) return false;

  BiomeTransition.cancelFlag[eid] = 1;
  BiomeTransition.active[eid] = 0;
  BiomeStateComp.transitionState[eid] = BIOME_TRANSITION_STATE.CANCELLED;

  // Snap back to the from biome.
  const wbase = eid * BIOME.COUNT;
  const fromBiome = BiomeTransition.fromBiome[eid];
  for (let b = 0; b < BIOME.COUNT; b++) {
    BiomeWeights.targetWeights[wbase + b] = (b === fromBiome) ? 1.0 : 0.0;
    BiomeWeights.weights[wbase + b] = BiomeWeights.prevWeights[wbase + b];
  }
  BiomeStateComp.targetBiome[eid] = fromBiome;

  return true;
}

/**
 * Advances an entity's biome transition by dt.
 */
export function tickBiomeTransition(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (BiomeTransition.active[eid] === 0) return false;

  const dur = BiomeTransition.duration[eid];
  let elapsed = BiomeTransition.elapsed[eid] + dt;
  if (elapsed > dur) elapsed = dur;

  BiomeTransition.elapsed[eid] = elapsed;
  const raw = elapsed / dur;
  BiomeTransition.progress[eid] = raw;

  // Easing.
  let eased;
  switch (BiomeTransition.easing[eid]) {
    case 0:  eased = raw; break;
    case 1:  eased = _smoothstep(0, 1, raw); break;
    case 2:  eased = _smootherstep(0, 1, raw); break;
    default: eased = _smoothstep(0, 1, raw); break;
  }

  // Blend weights.
  const wbase = eid * BIOME.COUNT;
  for (let b = 0; b < BIOME.COUNT; b++) {
    const fromW = BiomeWeights.prevWeights[wbase + b];
    const toW = BiomeWeights.targetWeights[wbase + b];
    BiomeWeights.weights[wbase + b] = _lerp(fromW, toW, eased);
  }

  BiomeStateComp.dominantBiome[eid] = classifyBiome(
    BiomeWeights.weights.subarray(wbase, wbase + BIOME.COUNT)
  );
  BiomePalette.dirty[eid] = 1;

  if (elapsed >= dur) {
    BiomeTransition.active[eid] = 0;
    BiomeStateComp.transitionState[eid] = BIOME_TRANSITION_STATE.COMPLETE;

    // Snap to final weights.
    for (let b = 0; b < BIOME.COUNT; b++) {
      BiomeWeights.weights[wbase + b] = BiomeWeights.targetWeights[wbase + b];
      BiomeWeights.prevWeights[wbase + b] = BiomeWeights.targetWeights[wbase + b];
    }
    BiomeStateComp.dominantBiome[eid] = BiomeStateComp.targetBiome[eid];
    applyBiomeToEntity(eid);
  }

  BiomeState.totalTransitionTicks++;
  return true;
}

/**
 * Returns the transition progress for an entity in [0, 1] (1 = settled).
 */
export function evaluateBiomeTransition(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 1;
  if (BiomeTransition.active[eid] === 0) return 1;
  return BiomeTransition.progress[eid];
}

/**
 * Forces an entity to snap to its target biome immediately.
 */
export function forceBiomeInstant(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const target = BiomeStateComp.targetBiome[eid];
  return setBiome(eid, target, { instant: true });
}

/**
 * Blends between two biome weight vectors directly. Used by the
 * compositor to mix a zone's biome with its neighbors.
 */
export function blendBiomeWeights(out, weightsA, weightsB, t) {
  if (!out || !weightsA || !weightsB) return out;
  const k = _clamp01(t);
  const inv = 1 - k;
  let sum = 0;
  for (let i = 0; i < BIOME.COUNT; i++) {
    const w = weightsA[i] * inv + weightsB[i] * k;
    out[i] = w;
    sum += w;
  }
  if (sum > 1e-6) {
    const invSum = 1 / sum;
    for (let i = 0; i < BIOME.COUNT; i++) out[i] *= invSum;
  }
  BiomeState.totalBlends++;
  return out;
}

/* ------------------------------------------------------------------ */
/* 14. APPLY BIOME TO ENTITY                                          */
/* ------------------------------------------------------------------ */

/**
 * Reads the entity's current biome weights and writes every dependent
 * field: palette, climate bind, material overrides, vegetation weights.
 */
export function applyBiomeToEntity(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (BiomeStateComp.enabled[eid] === 0) return false;

  const wbase = eid * BIOME.COUNT;
  const pbase = eid * PALETTE_SLOT_COUNT * 3;

  // Accumulate every palette slot in Oklab space, then convert back.
  for (let slot = 0; slot < PALETTE_SLOT_COUNT; slot++) {
    let L = 0, A = 0, B = 0;
    for (let b = 0; b < BIOME.COUNT; b++) {
      const w = BiomeWeights.weights[wbase + b];
      if (w <= 0.001) continue;
      const def = _biomeRegistry[b];
      if (!def) continue;
      L += def.paletteOklab[slot * 3 + 0] * w;
      A += def.paletteOklab[slot * 3 + 1] * w;
      B += def.paletteOklab[slot * 3 + 2] * w;
    }
    _oklabToLinearRGB(L, A, B, _scratchRGB);
    BiomePalette.palette[pbase + slot * 3 + 0] = _scratchRGB[0];
    BiomePalette.palette[pbase + slot * 3 + 1] = _scratchRGB[1];
    BiomePalette.palette[pbase + slot * 3 + 2] = _scratchRGB[2];
  }

  BiomePalette.dirty[eid] = 0;
  BiomePalette.generation[eid]++;

  // Climate bind — weighted sum of per-biome climate modifiers.
  let tempOffset = 0, humid = 0, windB = 0, precipB = 0, fogB = 0, heatB = 0;
  for (let b = 0; b < BIOME.COUNT; b++) {
    const w = BiomeWeights.weights[wbase + b];
    if (w <= 0.001) continue;
    const def = _biomeRegistry[b];
    if (!def) continue;
    tempOffset += def.baseTempC * w;
    humid      += def.humidityBias * w;
    windB      += def.windBias * w;
    precipB    += def.precipBias * w;
    fogB       += def.fogBias * w;
    heatB      += def.heatBias * w;
  }
  BiomeClimateBind.tempOffsetC[eid] = tempOffset;
  BiomeClimateBind.humidityOffset[eid] = humid;
  BiomeClimateBind.windBias[eid] = windB;
  BiomeClimateBind.precipBias[eid] = precipB;
  BiomeClimateBind.fogBias[eid] = fogB;
  BiomeClimateBind.heatBias[eid] = heatB;

  // If bound to a weather zone, propagate the biome's climate to it.
  if (BiomeClimateBind.bound[eid] === 1) {
    const zoneEid = BiomeClimateBind.weatherZoneEid[eid];
    if (zoneEid >= 0 && zoneEid < MAX_ENTITIES && WeatherZone.enabled[zoneEid] === 1) {
      ClimateState.targetTempC[zoneEid] = tempOffset;
      ClimateState.targetHumidity[zoneEid] = _clamp01(0.5 + humid);
    }
  }

  // Material overrides.
  let groundKind = 0, rockKind = 0, foliageKind = 0, waterKind = 0;
  let snowB = 0, sandB = 0, iceB = 0, grassB = 0, rockB = 0, emission = 0;
  for (let b = 0; b < BIOME.COUNT; b++) {
    const w = BiomeWeights.weights[wbase + b];
    if (w <= 0.001) continue;
    const def = _biomeRegistry[b];
    if (!def) continue;
    groundKind  = Math.max(groundKind, def.groundKind);
    rockKind    = Math.max(rockKind, def.rockKind);
    foliageKind = Math.max(foliageKind, def.foliageKind);
    waterKind   = Math.max(waterKind, def.waterKind);
    snowB    += def.snowBlend * w;
    sandB    += def.sandBlend * w;
    iceB     += def.iceBlend * w;
    grassB   += def.grassBlend * w;
    rockB    += def.rockBlend * w;
    emission += def.emissionBoost * w;
  }
  BiomeMaterialOverride.groundKind[eid] = groundKind;
  BiomeMaterialOverride.rockKind[eid] = rockKind;
  BiomeMaterialOverride.foliageKind[eid] = foliageKind;
  BiomeMaterialOverride.waterKind[eid] = waterKind;
  BiomeMaterialOverride.snowBlend[eid] = _clamp01(snowB);
  BiomeMaterialOverride.sandBlend[eid] = _clamp01(sandB);
  BiomeMaterialOverride.iceBlend[eid] = _clamp01(iceB);
  BiomeMaterialOverride.grassBlend[eid] = _clamp01(grassB);
  BiomeMaterialOverride.rockBlend[eid] = _clamp01(rockB);
  BiomeMaterialOverride.emissionBoost[eid] = emission;

  // Vegetation weights — merge scatter densities across biomes.
  let treeW = 0, bushW = 0, grassW = 0, flowerW = 0, rockW = 0;
  let mossW = 0, snowW = 0, iceW = 0, shellW = 0, coralW = 0;
  let scatterTotal = 0;
  for (let b = 0; b < BIOME.COUNT; b++) {
    const w = BiomeWeights.weights[wbase + b];
    if (w <= 0.001) continue;
    const def = _biomeRegistry[b];
    if (!def) continue;
    treeW    += def.scatterDensity[SCATTER_KIND.TREE_LARGE]  * w;
    bushW    += def.scatterDensity[SCATTER_KIND.BUSH]        * w;
    grassW   += def.scatterDensity[SCATTER_KIND.GRASS]       * w;
    flowerW  += def.scatterDensity[SCATTER_KIND.FLOWER]      * w;
    rockW    += def.scatterDensity[SCATTER_KIND.ROCK_LARGE]  * w;
    mossW    += def.scatterDensity[SCATTER_KIND.MOSS]        * w;
    snowW    += def.scatterDensity[SCATTER_KIND.SNOW_TUFT]   * w;
    iceW     += def.scatterDensity[SCATTER_KIND.ICE_SHARD]   * w;
    shellW   += def.scatterDensity[SCATTER_KIND.SHELL]       * w;
    coralW   += def.scatterDensity[SCATTER_KIND.CORAL]       * w;
    scatterTotal += def.overallScatterDensity * w;
  }
  BiomeVegetationState.treeWeight[eid] = treeW;
  BiomeVegetationState.bushWeight[eid] = bushW;
  BiomeVegetationState.grassWeight[eid] = grassW;
  BiomeVegetationState.flowerWeight[eid] = flowerW;
  BiomeVegetationState.rockWeight[eid] = rockW;
  BiomeVegetationState.mossWeight[eid] = mossW;
  BiomeVegetationState.snowWeight[eid] = snowW;
  BiomeVegetationState.iceWeight[eid] = iceW;
  BiomeVegetationState.shellWeight[eid] = shellW;
  BiomeVegetationState.coralWeight[eid] = coralW;
  BiomeVegetationState.targetScatterDensity[eid] = _clamp01(scatterTotal);

  BiomeState.totalApplyToEntity++;
  markFrameDirty(eid, FDIRTY.NEEDS_UPDATE);
  return true;
}

/**
 * Applies the biome state to every registered biome-aware entity.
 */
export function applyBiomeToAllEntities() {
  let applied = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (BiomeStateComp.enabled[eid] === 0) continue;
    if (applyBiomeToEntity(eid)) applied++;
  }
  BiomeState.totalApplyToAll++;
  return applied;
}

/* ------------------------------------------------------------------ */
/* 15. GLOBAL BIOME MODE                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets the global dominant biome weight vector. This biases every
 * procedural sample toward the given biome. Used for full-scene biome
 * transitions.
 */
export function setGlobalBiome(biomeId, options) {
  if (biomeId < 0 || biomeId >= BIOME.COUNT) return false;
  const opts = options || {};

  for (let b = 0; b < BIOME.COUNT; b++) {
    BiomeState.globalBiomeWeights[b] = (b === biomeId) ? 1.0 : 0.0;
  }
  BiomeState.globalDominantBiome = biomeId;
  BiomeState.globalTargetBiome = biomeId;

  if (opts.instant) return true;

  return true;
}

/**
 * Advances the global biome blend toward the target biome. Called by
 * `tickBiome` each frame.
 */
export function tickGlobalBiomeBlend(dt) {
  const target = BiomeState.globalTargetBiome;
  const targetW = 1.0;
  const k = 1 - Math.exp(-1.4 * dt);
  const weights = BiomeState.globalBiomeWeights;
  for (let b = 0; b < BIOME.COUNT; b++) {
    const tgt = (b === target) ? targetW : 0.0;
    weights[b] += (tgt - weights[b]) * k;
  }
  // Renormalize.
  let sum = 0;
  for (let b = 0; b < BIOME.COUNT; b++) sum += weights[b];
  if (sum > 1e-6) {
    const inv = 1 / sum;
    for (let b = 0; b < BIOME.COUNT; b++) weights[b] *= inv;
  }
}

/* ------------------------------------------------------------------ */
/* 16. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the biome system frame counter.
 */
export function tickBiomeFrame(frameNumber) {
  if (typeof frameNumber === 'number') BiomeState.frame = frameNumber;
  else BiomeState.frame++;
}

/**
 * Full per-frame biome pipeline:
 *   1. tickBiomeFrame
 *   2. tickGlobalBiomeBlend
 *   3. Per-entity transition tick
 *   4. applyBiomeToEntity for dirty entities
 *   5. Update statistics
 */
export function tickBiome(dt) {
  const t0 = _now();

  BiomeState.frame++;

  // Global blend.
  tickGlobalBiomeBlend(dt);

  const adapter = getAdapter();
  let active = 0;
  let transitioning = 0;
  const dominantCounts = BiomeStats.dominantCounts;
  dominantCounts.fill(0);
  let sumTransitions = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (BiomeStateComp.enabled[eid] === 0) continue;
    if (!adapter.entityAlive(eid)) continue;
    active++;

    if (BiomeTransition.active[eid] === 1) {
      tickBiomeTransition(eid, dt);
      sumTransitions += BiomeTransition.progress[eid];
      transitioning++;
    }

    // Apply biome to entity if dirty.
    if (BiomePalette.dirty[eid] === 1) {
      applyBiomeToEntity(eid);
    }

    const dominant = BiomeStateComp.dominantBiome[eid];
    if (dominant >= 0 && dominant < BIOME.COUNT) dominantCounts[dominant]++;
  }

  BiomeStats.activeEntities[0] = active;
  BiomeStats.transitioning[0] = transitioning;
  BiomeStats.averageTransitionT[0] = transitioning > 0 ? sumTransitions / transitioning : 1;

  const t1 = _now();
  BiomeState.lastTickMs = t1 - t0;
  BiomeState.avgTickMs += (BiomeState.lastTickMs - BiomeState.avgTickMs) * 0.15;

  return {
    frame: BiomeState.frame,
    active,
    transitioning,
    globalBiome: BiomeState.globalDominantBiome,
    cost: BiomeState.lastTickMs,
  };
}

/* ------------------------------------------------------------------ */
/* 17. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerBiomeComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'BiomeStateComp',         component: BiomeStateComp,         category: 10, subsystem: 7, dependencies: [] },
    { name: 'BiomeWeights',           component: BiomeWeights,           category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeTransition',        component: BiomeTransition,        category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomePalette',           component: BiomePalette,           category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeClimateBind',       component: BiomeClimateBind,       category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeMaterialOverride',  component: BiomeMaterialOverride,  category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeVegetationState',   component: BiomeVegetationState,   category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeScatterBudget',     component: BiomeScatterBudget,     category: 10, subsystem: 7, dependencies: [] },
    { name: 'BiomeZoneBlend',         component: BiomeZoneBlend,         category: 10, subsystem: 7, dependencies: ['BiomeStateComp'] },
    { name: 'BiomeStats',             component: BiomeStats,             category: 10, subsystem: 7, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 18. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getBiomeEntityStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  if (BiomeStateComp.enabled[eid] === 0) return null;

  const wbase = eid * BIOME.COUNT;
  const weights = new Array(BIOME.COUNT);
  for (let b = 0; b < BIOME.COUNT; b++) {
    weights[b] = BiomeWeights.weights[wbase + b];
  }

  return {
    entity:             eid,
    dominantBiome:      BIOME_NAME[BiomeStateComp.dominantBiome[eid]] || 'unknown',
    targetBiome:        BIOME_NAME[BiomeStateComp.targetBiome[eid]] || 'unknown',
    prevBiome:          BIOME_NAME[BiomeStateComp.prevBiome[eid]] || 'unknown',
    weights,
    transitionActive:   BiomeTransition.active[eid] === 1,
    transitionProgress: BiomeTransition.progress[eid],
    transitionState:    BIOME_TRANSITION_STATE_NAME[BiomeStateComp.transitionState[eid]] || 'idle',
    changeCount:        BiomeStateComp.changeCount[eid],
    tempOffsetC:        BiomeClimateBind.tempOffsetC[eid],
    humidityOffset:     BiomeClimateBind.humidityOffset[eid],
    windBias:           BiomeClimateBind.windBias[eid],
    precipBias:         BiomeClimateBind.precipBias[eid],
    fogBias:            BiomeClimateBind.fogBias[eid],
    heatBias:           BiomeClimateBind.heatBias[eid],
    snowBlend:          BiomeMaterialOverride.snowBlend[eid],
    sandBlend:          BiomeMaterialOverride.sandBlend[eid],
    iceBlend:           BiomeMaterialOverride.iceBlend[eid],
    grassBlend:         BiomeMaterialOverride.grassBlend[eid],
    rockBlend:          BiomeMaterialOverride.rockBlend[eid],
    scatterDensity:     BiomeVegetationState.targetScatterDensity[eid],
    treeWeight:         BiomeVegetationState.treeWeight[eid],
    rockWeight:         BiomeVegetationState.rockWeight[eid],
    grassWeight:        BiomeVegetationState.grassWeight[eid],
    blendRadius:        BiomeZoneBlend.blendRadius[eid],
  };
}

export function getBiomeSystemReport() {
  const dominantHistogram = [];
  for (let b = 0; b < BIOME.COUNT; b++) {
    dominantHistogram.push({
      biome: BIOME_NAME[b],
      count: BiomeStats.dominantCounts[b],
    });
  }

  const globalWeights = [];
  for (let b = 0; b < BIOME.COUNT; b++) {
    globalWeights.push({ biome: BIOME_NAME[b], weight: BiomeState.globalBiomeWeights[b] });
  }

  return {
    frame:                     BiomeState.frame,
    activeBiomeEntities:       BiomeState.activeBiomeEntities,
    peakBiomeEntities:         BiomeState.peakBiomeEntities,
    totalRegistrations:        BiomeState.totalRegistrations,
    totalUnregistrations:      BiomeState.totalUnregistrations,
    totalSetBiome:             BiomeState.totalSetBiome,
    totalTransitions:          BiomeState.totalTransitions,
    totalTransitionTicks:      BiomeState.totalTransitionTicks,
    totalInstantSnape:         BiomeState.totalInstantSnape,
    totalBlends:               BiomeState.totalBlends,
    totalApplyToEntity:        BiomeState.totalApplyToEntity,
    totalApplyToAll:           BiomeState.totalApplyToAll,
    totalWorldGenPasses:       BiomeState.totalWorldGenPasses,
    totalChunksGenerated:      BiomeState.totalChunksGenerated,
    totalScatterPlacements:    BiomeState.totalScatterPlacements,
    lastTickMs:                BiomeState.lastTickMs,
    avgTickMs:                 BiomeState.avgTickMs,
    lastBlendMs:               BiomeState.lastBlendMs,
    avgBlendMs:                BiomeState.avgBlendMs,
    lastWorldGenMs:            BiomeState.lastWorldGenMs,
    avgWorldGenMs:             BiomeState.avgWorldGenMs,
    globalDominantBiome:       BIOME_NAME[BiomeState.globalDominantBiome] || 'unknown',
    globalTargetBiome:         BIOME_NAME[BiomeState.globalTargetBiome] || 'unknown',
    globalWeights,
    worldGenPhase:             WORLD_GEN_PHASE_NAME[BiomeState.worldGenPhase] || 'idle',
    worldGenProgress:          BiomeState.worldGenProgress,
    registeredDefinitions:     getBiomeDefinitionCount(),
    dominantHistogram,
    transitioningEntities:     BiomeStats.transitioning[0],
    perfTier:                  PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 19. RESET                                                          */
/* ------------------------------------------------------------------ */

export function resetBiomeState() {
  BiomeStateComp.enabled.fill(0);
  BiomeStateComp.dominantBiome.fill(0);
  BiomeStateComp.targetBiome.fill(0);
  BiomeStateComp.prevBiome.fill(0);
  BiomeStateComp.transitionState.fill(0);
  BiomeStateComp.flags.fill(0);
  BiomeStateComp.transitionTimer.fill(0);
  BiomeStateComp.transitionDuration.fill(DEFAULT_BIOME_TRANSITION_DURATION);
  BiomeStateComp.lastChangeFrame.fill(0);
  BiomeStateComp.changeCount.fill(0);
  BiomeStateComp.priority.fill(0);

  BiomeWeights.weights.fill(0);
  BiomeWeights.targetWeights.fill(0);
  BiomeWeights.prevWeights.fill(0);

  BiomeTransition.fromBiome.fill(0);
  BiomeTransition.toBiome.fill(0);
  BiomeTransition.elapsed.fill(0);
  BiomeTransition.duration.fill(0);
  BiomeTransition.progress.fill(0);
  BiomeTransition.easing.fill(1);
  BiomeTransition.cancelFlag.fill(0);
  BiomeTransition.active.fill(0);

  BiomePalette.palette.fill(0);
  BiomePalette.dirty.fill(1);
  BiomePalette.generation.fill(0);

  BiomeClimateBind.weatherZoneEid.fill(-1);
  BiomeClimateBind.tempOffsetC.fill(0);
  BiomeClimateBind.humidityOffset.fill(0);
  BiomeClimateBind.windBias.fill(0);
  BiomeClimateBind.precipBias.fill(0);
  BiomeClimateBind.fogBias.fill(0);
  BiomeClimateBind.heatBias.fill(0);
  BiomeClimateBind.bound.fill(0);

  BiomeMaterialOverride.groundKind.fill(0);
  BiomeMaterialOverride.rockKind.fill(0);
  BiomeMaterialOverride.foliageKind.fill(0);
  BiomeMaterialOverride.waterKind.fill(0);
  BiomeMaterialOverride.snowBlend.fill(0);
  BiomeMaterialOverride.sandBlend.fill(0);
  BiomeMaterialOverride.iceBlend.fill(0);
  BiomeMaterialOverride.grassBlend.fill(0);
  BiomeMaterialOverride.rockBlend.fill(0);
  BiomeMaterialOverride.emissionBoost.fill(0);

  BiomeVegetationState.scatterDensity.fill(0);
  BiomeVegetationState.targetScatterDensity.fill(0);
  BiomeVegetationState.treeWeight.fill(0);
  BiomeVegetationState.bushWeight.fill(0);
  BiomeVegetationState.grassWeight.fill(0);
  BiomeVegetationState.flowerWeight.fill(0);
  BiomeVegetationState.rockWeight.fill(0);
  BiomeVegetationState.mossWeight.fill(0);
  BiomeVegetationState.snowWeight.fill(0);
  BiomeVegetationState.iceWeight.fill(0);
  BiomeVegetationState.shellWeight.fill(0);
  BiomeVegetationState.coralWeight.fill(0);
  BiomeVegetationState.scatterBudget.fill(0);

  BiomeScatterBudget.totalScatterCost[0] = 0;
  BiomeScatterBudget.budgetCap[0] = 8.0;
  BiomeScatterBudget.budgetExceeded[0] = 0;
  BiomeScatterBudget.totalPlacements[0] = 0;
  BiomeScatterBudget.totalTrees[0] = 0;
  BiomeScatterBudget.totalGrass[0] = 0;
  BiomeScatterBudget.totalRocks[0] = 0;
  BiomeScatterBudget.totalFoliage[0] = 0;

  BiomeZoneBlend.blendRadius.fill(DEFAULT_BIOME_BLEND_RADIUS);
  BiomeZoneBlend.blendSoftness.fill(0.5);
  BiomeZoneBlend.neighborMask.fill(0);
  BiomeZoneBlend.dominantBlend.fill(0);

  BiomeStats.activeEntities[0] = 0;
  BiomeStats.transitioning[0] = 0;
  BiomeStats.dominantCounts.fill(0);
  BiomeStats.averageTransitionT[0] = 1;
  BiomeStats.lastWorldGenMs[0] = 0;
  BiomeStats.lastScatterCost[0] = 0;

  BiomeState.frame = 0;
  BiomeState.activeBiomeEntities = 0;
  BiomeState.peakBiomeEntities = 0;
  BiomeState.totalRegistrations = 0;
  BiomeState.totalUnregistrations = 0;
  BiomeState.totalSetBiome = 0;
  BiomeState.totalTransitions = 0;
  BiomeState.totalTransitionTicks = 0;
  BiomeState.totalInstantSnape = 0;
  BiomeState.totalBlends = 0;
  BiomeState.totalApplyToEntity = 0;
  BiomeState.totalApplyToAll = 0;
  BiomeState.totalWorldGenPasses = 0;
  BiomeState.totalChunksGenerated = 0;
  BiomeState.totalScatterPlacements = 0;
  BiomeState.lastTickMs = 0;
  BiomeState.avgTickMs = 0;
  BiomeState.lastBlendMs = 0;
  BiomeState.avgBlendMs = 0;
  BiomeState.lastWorldGenMs = 0;
  BiomeState.avgWorldGenMs = 0;
  BiomeState.globalBiomeWeights.fill(0);
  BiomeState.globalBiomeWeights[BIOME.DESERT] = 1;
  BiomeState.globalDominantBiome = BIOME.DESERT;
  BiomeState.globalTargetBiome = BIOME.DESERT;
  BiomeState.worldGenPhase = WORLD_GEN_PHASE.IDLE;
  BiomeState.worldGenProgress = 0;
}

/* ------------------------------------------------------------------ */
/* 20. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_BIOME_DEFS,
  MAX_BIOME_ENTITIES,
  MAX_BIOME_WEIGHTS,
  MAX_SCATTER_KINDS,
  DEFAULT_BIOME_TRANSITION_DURATION,
  DEFAULT_WORLD_SIZE,
  DEFAULT_WORLD_RESOLUTION,
  DEFAULT_MOISTURE_RESOLUTION,
  DEFAULT_BIOME_BLEND_RADIUS,
  PALETTE_SLOT_COUNT,

  // Enums
  BIOME,
  BIOME_NAME,
  BIOME_TRANSITION_STATE,
  BIOME_TRANSITION_STATE_NAME,
  SCATTER_KIND,
  SCATTER_KIND_NAME,
  BIOME_FLAG,
  WORLD_GEN_PHASE,
  WORLD_GEN_PHASE_NAME,

  // Components
  BiomeStateComp,
  BiomeWeights,
  BiomeTransition,
  BiomePalette,
  BiomeClimateBind,
  BiomeMaterialOverride,
  BiomeVegetationState,
  BiomeScatterBudget,
  BiomeZoneBlend,
  BiomeStats,
  BIOME_COMPONENTS,

  // Module state
  BiomeState,

  // Biome definitions
  registerBiomeDefinition,
  getBiomeDefinition,
  getBiomeDefinitionCount,

  // Procedural world
  sampleTerrainHeight,
  sampleTerrainNormal,
  sampleBiomeWeights,
  classifyBiome,
  sampleBiomeColor,
  generateChunkHeightField,
  generateChunkBiomeField,
  evaluateChunkRelief,
  computeChunkLODFromRelief,
  buildTerrainGeometry,
  buildBiomeColorAttribute,
  generateProceduralWorld,

  // Entity registration
  registerBiomeEntity,
  unregisterBiomeEntity,
  isBiomeEntity,

  // Transitions
  setBiome,
  cancelBiomeTransition,
  tickBiomeTransition,
  evaluateBiomeTransition,
  forceBiomeInstant,
  blendBiomeWeights,

  // Application
  applyBiomeToEntity,
  applyBiomeToAllEntities,

  // Global mode
  setGlobalBiome,
  tickGlobalBiomeBlend,

  // Frame
  tickBiomeFrame,
  tickBiome,

  // Diagnostics
  getBiomeEntityStats,
  getBiomeSystemReport,

  // Registration
  registerBiomeComponents,

  // Reset
  resetBiomeState,
};

export default _defaultExport;