// File : 031
// name : src/ecs/031_scn_WeatherComponents.js
// description : Weather, climate, seasonal, and atmospheric SoA component
//               module for the scene ECS world of the anime lighting stack
//               on Android mobile. Declares every per-zone climate state,
//               temperature gradient, season phase, wind field, rain emitter,
//               snow emitter, cloud layer, heat-effect volume, fog volume,
//               lightning strike, precipitation budget, weather blend, and
//               biome climate modifier the lighting stack needs — as
//               fixed-capacity typed arrays sized once to
//               MAX_ENTITIES = 100000 (or a small fixed pool for global
//               weather slots).
//
//               Provides the fast helpers that drive the live weather
//               system:
//                 • setWeatherPreset          — install a preset
//                 • blendWeatherPreset        — cross-fade to a new preset
//                 • setSeason                 — set the current season
//                 • tickSeason                — advance the season phase
//                 • updateClimate             — temperature / humidity /
//                                               pressure integration
//                 • updateBiomeClimate        — biome-specific modifiers
//                 • updateWindField           — turbulence + gust evolution
//                 • sampleWindAt              — wind vector at (x,y,z,t)
//                 • setRainType               — swap rain emitter profile
//                 • emitPrecipitation         — spawn drops / flakes
//                 • tickPrecipitation         — advance all emitters
//                 • setSnowType               — swap snow emitter profile
//                 • setCloudType              — swap cloud layer profile
//                 • updateCloudCoverage       — animate cloud coverage
//                 • updateHeatEffect          — heat shimmer / mirage
//                 • updateFogVolume           — fog density / color / height
//                 • castLightningStrike       — trigger a lightning bolt
//                 • tickLightning             — advance strike animations
//                 • tickWeather               — one-shot per-frame pipeline
//
//               Design:
//                 • Fixed-capacity SoA arrays sized once. No dynamic growth,
//                   no map allocations, no runtime resizes.
//                 • Global weather slots: a small fixed pool (MAX_WEATHER_
//                   ZONES) so multiple coexisting climate zones can be
//                   active at once (e.g. indoor and outdoor transition).
//                 • Temperature is simulated as a leaky integrator with a
//                   season curve, day-cycle modulation, and biome bias.
//                 • Wind is a layered simplex field with gust bursts and a
//                   base direction that follows the biome's prevailing
//                   wind.
//                 • Rain and snow share a unified precipitation emitter
//                   with a profile table (drop size, fall speed, streak
//                   length, tint, alpha).
//                 • Heat effect is a 3-channel volume: shimmer amplitude,
//                   mirage strength, and scorch tint.
//                 • Lightning is a discrete event with position, duration,
//                   intensity, and a bloom tint.
//                 • Every helper is allocation-free — scratch vectors are
//                   module-level and reused.
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
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every biome, indoor room, and outdoor scene
//            in the anime lighting stack has a live, dynamic, season-aware,
//            biome-aware, temperature-aware weather system — with multiple
//            snow types, multiple rain types, layered clouds, layered wind,
//            heat shimmer, fog volumes, and lightning — all driven by a
//            single allocation-free per-frame pipeline.
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
  markFrameDirty,
  clearFrameDirty,
} from './014_scn_Tags.js';

import {
  Transform,
  TransformWorld,
} from './026_scn_TransformComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of simultaneous weather zones. Global weather state
 * (temperature, season, wind) is shared via a small fixed pool so
 * multiple coexisting climate regions (e.g. indoor room + outdoor
 * valley) can each have their own live weather.
 */
export const MAX_WEATHER_ZONES =
  PERF_TIER_LOCAL === 'HIGH'   ? 16 :
  PERF_TIER_LOCAL === 'MEDIUM' ?  8 :
                                  4;

/**
 * Maximum number of active precipitation emitters.
 */
export const MAX_PRECIP_EMITTERS =
  PERF_TIER_LOCAL === 'HIGH'   ? 64 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 32 :
                                 16;

/**
 * Maximum number of active fog volumes.
 */
export const MAX_FOG_VOLUMES =
  PERF_TIER_LOCAL === 'HIGH'   ? 32 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 16 :
                                  8;

/**
 * Maximum number of active lightning strikes at once.
 */
export const MAX_LIGHTNING_STRIKES =
  PERF_TIER_LOCAL === 'HIGH'   ? 8 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 4 :
                                 2;

/**
 * Maximum number of cloud layers per zone.
 */
export const MAX_CLOUD_LAYERS_PER_ZONE = 4;

/**
 * Maximum number of biome climate modifiers registered.
 */
export const MAX_BIOME_CLIMATES = 32;

/**
 * Default frame-rate for weather simulation. Weather updates are cheaper
 * when run at a lower rate than the render loop.
 */
export const WEATHER_UPDATE_HZ =
  PERF_TIER_LOCAL === 'HIGH'   ? 60 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 30 :
                                 15;

/**
 * Weather types.
 */
export const WEATHER_TYPE = Object.freeze({
  CLEAR:          0,
  CLOUDY:         1,
  OVERCAST:       2,
  RAIN_LIGHT:     3,
  RAIN_HEAVY:     4,
  RAIN_STORM:     5,
  SNOW_LIGHT:     6,
  SNOW_HEAVY:     7,
  SNOW_BLIZZARD:  8,
  FOG_LIGHT:      9,
  FOG_DENSE:     10,
  DUST_STORM:    11,
  HAZE:          12,
  THUNDERSTORM:  13,
  SLEET:         14,
  HAIL:          15,
  HEAT_WAVE:     16,
  AURORA_STORM:  17,
  COUNT:         18,
});

export const WEATHER_TYPE_NAME = Object.freeze([
  'clear', 'cloudy', 'overcast', 'rain_light', 'rain_heavy', 'rain_storm',
  'snow_light', 'snow_heavy', 'snow_blizzard', 'fog_light', 'fog_dense',
  'dust_storm', 'haze', 'thunderstorm', 'sleet', 'hail', 'heat_wave',
  'aurora_storm',
]);

/**
 * Seasons.
 */
export const SEASON = Object.freeze({
  SPRING: 0,
  SUMMER: 1,
  AUTUMN: 2,
  WINTER: 3,
  COUNT:  4,
});

export const SEASON_NAME = Object.freeze([
  'spring', 'summer', 'autumn', 'winter',
]);

/**
 * Rain profile types.
 */
export const RAIN_TYPE = Object.freeze({
  DRIZZLE:          0,
  STEADY:           1,
  HEAVY:            2,
  MONSOON:          3,
  THUNDER_SHOWER:   4,
  COUNT:            5,
});

export const RAIN_TYPE_NAME = Object.freeze([
  'drizzle', 'steady', 'heavy', 'monsoon', 'thunder_shower',
]);

/**
 * Snow profile types.
 */
export const SNOW_TYPE = Object.freeze({
  POWDER:     0,
  WET:        1,
  CRUST:      2,
  BLIZZARD:   3,
  SLEET:      4,
  ICE_PELLET: 5,
  COUNT:      6,
});

export const SNOW_TYPE_NAME = Object.freeze([
  'powder', 'wet', 'crust', 'blizzard', 'sleet', 'ice_pellet',
]);

/**
 * Cloud types.
 */
export const CLOUD_TYPE = Object.freeze({
  NONE:           0,
  CIRRUS:         1,
  CUMULUS:        2,
  STRATUS:        3,
  CUMULONIMBUS:   4,
  NIMBOSTRATUS:   5,
  CIRROCUMULUS:   6,
  STRATOCUMULUS:  7,
  ALTOCUMULUS:    8,
  COUNT:          9,
});

export const CLOUD_TYPE_NAME = Object.freeze([
  'none', 'cirrus', 'cumulus', 'stratus', 'cumulonimbus', 'nimbostratus',
  'cirrocumulus', 'stratocumulus', 'altocumulus',
]);

/**
 * Wind profile types.
 */
export const WIND_TYPE = Object.freeze({
  CALM:    0,
  BREEZE:  1,
  GUSTY:   2,
  GALE:    3,
  STORM:   4,
  HURRICANE:5,
  WHIRLWIND:6,
  COUNT:   7,
});

export const WIND_TYPE_NAME = Object.freeze([
  'calm', 'breeze', 'gusty', 'gale', 'storm', 'hurricane', 'whirlwind',
]);

/**
 * Heat effect types.
 */
export const HEAT_EFFECT = Object.freeze({
  NONE:      0,
  SUBTLE:    1,
  MIRAGE:    2,
  SHIMMER:   3,
  SCORCHING: 4,
  COUNT:     5,
});

export const HEAT_EFFECT_NAME = Object.freeze([
  'none', 'subtle', 'mirage', 'shimmer', 'scorching',
]);

/**
 * Climate zones (biome-level classification).
 */
export const CLIMATE_ZONE = Object.freeze({
  POLAR:      0,
  TUNDRA:     1,
  TEMPERATE:  2,
  SUBTROPICAL:3,
  TROPICAL:   4,
  DESERT:     5,
  MOUNTAIN:   6,
  COASTAL:    7,
  VOLCANIC:   8,
  SWAMP:      9,
  SAVANNA:   10,
  COUNT:     11,
});

export const CLIMATE_ZONE_NAME = Object.freeze([
  'polar', 'tundra', 'temperate', 'subtropical', 'tropical', 'desert',
  'mountain', 'coastal', 'volcanic', 'swamp', 'savanna',
]);

/**
 * Precipitation emitter states.
 */
export const PRECIP_STATE = Object.freeze({
  IDLE:     0,
  STARTING: 1,
  ACTIVE:   2,
  FADING:   3,
  STOPPED:  4,
  COUNT:    5,
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const WeatherState = {
  frame:                  0,
  subframeAccum:          0,
  totalSetPresets:        0,
  totalBlends:            0,
  totalSeasonChanges:     0,
  totalLightningStrikes:  0,
  totalPrecipEmits:       0,
  totalWindUpdates:       0,
  totalClimateUpdates:    0,
  totalHeatUpdates:       0,
  totalFogUpdates:        0,
  lastTickMs:             0,
  avgTickMs:              0,
  lastPrecipTickMs:       0,
  avgPrecipTickMs:        0,
  lastWindMs:             0,
  avgWindMs:              0,
  globalTime:             0,
  globalSeason:           SEASON.SPRING,
  globalSeasonPhase:      0,
  globalSeasonSpeed:      1 / 300,   // 300 seconds per season by default
  globalTimeOfDay:        0.38,
  globalDaySpeed:         1 / 180,
  globalYearLength:       300 * 4,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.weather', {
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
function _lerp(a, b, t) { return a + (b - a) * t; }
function _smoothstep(e0, e1, x) {
  const t = _clamp((x - e0) / (e1 - e0 || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
}

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * WeatherZone — one live weather zone per entity that needs it. Zones
 * are indexed by entity id; the per-zone state lives in the arrays
 * below, indexed by entity.
 */
export const WeatherZone = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  weatherType:       new Uint8Array(MAX_ENTITIES),
  weatherTypePrev:   new Uint8Array(MAX_ENTITIES),
  climateZone:       new Uint8Array(MAX_ENTITIES),
  season:            new Uint8Array(MAX_ENTITIES),
  seasonPhase:       new Float32Array(MAX_ENTITIES),
  updateInterval:    new Uint8Array(MAX_ENTITIES),
  updateAccum:       new Float32Array(MAX_ENTITIES),
  priority:          new Uint8Array(MAX_ENTITIES),
  flags:             new Uint16Array(MAX_ENTITIES),
};

/**
 * ClimateState — temperature / humidity / pressure per zone.
 */
export const ClimateState = {
  temperatureC:      new Float32Array(MAX_ENTITIES),
  targetTempC:       new Float32Array(MAX_ENTITIES),
  humidity:          new Float32Array(MAX_ENTITIES),
  targetHumidity:    new Float32Array(MAX_ENTITIES),
  pressureHPa:       new Float32Array(MAX_ENTITIES),
  targetPressureHPa: new Float32Array(MAX_ENTITIES),
  dewPointC:         new Float32Array(MAX_ENTITIES),
  windChillC:        new Float32Array(MAX_ENTITIES),
  heatIndexC:        new Float32Array(MAX_ENTITIES),
  cloudCoverage:     new Float32Array(MAX_ENTITIES),
  targetCloudCoverage:new Float32Array(MAX_ENTITIES),
  integrationRate:   new Float32Array(MAX_ENTITIES).fill(0.4),
  lastUpdateFrame:   new Uint32Array(MAX_ENTITIES),
};

/**
 * SeasonState — season index + phase + year counter per zone.
 */
export const SeasonState = {
  seasonIndex:       new Uint8Array(MAX_ENTITIES),
  seasonPhase:       new Float32Array(MAX_ENTITIES),
  seasonSpeed:       new Float32Array(MAX_ENTITIES).fill(1 / 300),
  yearCounter:       new Uint16Array(MAX_ENTITIES),
  amplitudeC:        new Float32Array(MAX_ENTITIES).fill(15),
  baseTempC:         new Float32Array(MAX_ENTITIES).fill(10),
  springBias:        new Float32Array(MAX_ENTITIES).fill(0),
  summerBias:        new Float32Array(MAX_ENTITIES).fill(1),
  autumnBias:        new Float32Array(MAX_ENTITIES).fill(0),
  winterBias:        new Float32Array(MAX_ENTITIES).fill(-1),
};

/**
 * WindField — wind direction, speed, and turbulence per zone.
 */
export const WindField = {
  dirX:              new Float32Array(MAX_ENTITIES),
  dirY:              new Float32Array(MAX_ENTITIES),
  dirZ:              new Float32Array(MAX_ENTITIES),
  speed:             new Float32Array(MAX_ENTITIES),
  targetSpeed:       new Float32Array(MAX_ENTITIES),
  gustSpeed:         new Float32Array(MAX_ENTITIES),
  gustTimer:         new Float32Array(MAX_ENTITIES),
  gustInterval:      new Float32Array(MAX_ENTITIES),
  turbulence:        new Float32Array(MAX_ENTITIES),
  windType:          new Uint8Array(MAX_ENTITIES),
  prevailingDirX:    new Float32Array(MAX_ENTITIES),
  prevailingDirZ:    new Float32Array(MAX_ENTITIES),
  noiseScale:        new Float32Array(MAX_ENTITIES).fill(0.025),
  noiseSpeed:        new Float32Array(MAX_ENTITIES).fill(0.35),
  noiseAmp:          new Float32Array(MAX_ENTITIES).fill(0.6),
  elevationLift:     new Float32Array(MAX_ENTITIES).fill(0.15),
  swirlStrength:     new Float32Array(MAX_ENTITIES),
};

/**
 * RainEmitter — rain profile per zone.
 */
export const RainEmitter = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  rainType:          new Uint8Array(MAX_ENTITIES),
  intensity:         new Float32Array(MAX_ENTITIES),
  targetIntensity:   new Float32Array(MAX_ENTITIES),
  dropSize:          new Float32Array(MAX_ENTITIES).fill(0.02),
  dropSpeed:         new Float32Array(MAX_ENTITIES).fill(18),
  streakLength:      new Float32Array(MAX_ENTITIES).fill(0.6),
  alpha:             new Float32Array(MAX_ENTITIES).fill(0.6),
  tintR:             new Float32Array(MAX_ENTITIES).fill(0.55),
  tintG:             new Float32Array(MAX_ENTITIES).fill(0.70),
  tintB:             new Float32Array(MAX_ENTITIES).fill(0.95),
  spread:            new Float32Array(MAX_ENTITIES).fill(0.35),
  splashStrength:    new Float32Array(MAX_ENTITIES),
  lightningChance:   new Float32Array(MAX_ENTITIES),
  state:             new Uint8Array(MAX_ENTITIES),
  fadeTimer:         new Float32Array(MAX_ENTITIES),
};

/**
 * SnowEmitter — snow profile per zone.
 */
export const SnowEmitter = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  snowType:          new Uint8Array(MAX_ENTITIES),
  intensity:         new Float32Array(MAX_ENTITIES),
  targetIntensity:   new Float32Array(MAX_ENTITIES),
  flakeSize:         new Float32Array(MAX_ENTITIES).fill(0.05),
  fallSpeed:         new Float32Array(MAX_ENTITIES).fill(2.2),
  driftStrength:     new Float32Array(MAX_ENTITIES).fill(0.8),
  swirlStrength:     new Float32Array(MAX_ENTITIES).fill(0.4),
  alpha:             new Float32Array(MAX_ENTITIES).fill(0.9),
  tintR:             new Float32Array(MAX_ENTITIES).fill(0.95),
  tintG:             new Float32Array(MAX_ENTITIES).fill(0.98),
  tintB:             new Float32Array(MAX_ENTITIES).fill(1.00),
  spread:            new Float32Array(MAX_ENTITIES).fill(0.6),
  accumulation:      new Float32Array(MAX_ENTITIES),
  state:             new Uint8Array(MAX_ENTITIES),
  fadeTimer:         new Float32Array(MAX_ENTITIES),
};

/**
 * CloudLayer — up to MAX_CLOUD_LAYERS_PER_ZONE cloud layers per zone.
 * Layout: layers[(eid * MAX_CLOUD_LAYERS_PER_ZONE + layerIdx)]
 */
export const CloudLayer = {
  cloudType:         new Uint8Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE),
  coverage:          new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE),
  targetCoverage:    new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE),
  altitude:          new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(1.0),
  thickness:         new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(0.4),
  density:           new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(0.7),
  driftX:            new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE),
  driftZ:            new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE),
  tintR:             new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(1.0),
  tintG:             new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(1.0),
  tintB:             new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(1.0),
  shadowStrength:    new Float32Array(MAX_ENTITIES * MAX_CLOUD_LAYERS_PER_ZONE).fill(0.3),
  layerCount:        new Uint8Array(MAX_ENTITIES),
};

/**
 * HeatEffect — heat shimmer / mirage per zone.
 */
export const HeatEffect = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  effectType:        new Uint8Array(MAX_ENTITIES),
  shimmerAmp:        new Float32Array(MAX_ENTITIES),
  targetShimmerAmp:  new Float32Array(MAX_ENTITIES),
  mirageStrength:    new Float32Array(MAX_ENTITIES),
  targetMirage:      new Float32Array(MAX_ENTITIES),
  scorchR:           new Float32Array(MAX_ENTITIES).fill(1.0),
  scorchG:           new Float32Array(MAX_ENTITIES).fill(0.85),
  scorchB:           new Float32Array(MAX_ENTITIES).fill(0.6),
  waveFreq:          new Float32Array(MAX_ENTITIES).fill(6.0),
  waveSpeed:         new Float32Array(MAX_ENTITIES).fill(1.4),
  heightFalloff:     new Float32Array(MAX_ENTITIES).fill(2.0),
  intensity:         new Float32Array(MAX_ENTITIES),
};

/**
 * FogVolume — atmospheric fog per zone.
 */
export const FogVolume = {
  enabled:           new Uint8Array(MAX_ENTITIES),
  density:           new Float32Array(MAX_ENTITIES),
  targetDensity:     new Float32Array(MAX_ENTITIES),
  heightBase:        new Float32Array(MAX_ENTITIES),
  heightFalloff:     new Float32Array(MAX_ENTITIES).fill(0.15),
  colorR:            new Float32Array(MAX_ENTITIES).fill(0.7),
  colorG:            new Float32Array(MAX_ENTITIES).fill(0.8),
  colorB:            new Float32Array(MAX_ENTITIES).fill(0.9),
  near:              new Float32Array(MAX_ENTITIES).fill(20),
  far:               new Float32Array(MAX_ENTITIES).fill(200),
  groundBlend:       new Float32Array(MAX_ENTITIES).fill(0.5),
  horizonBlend:      new Float32Array(MAX_ENTITIES).fill(0.7),
  sunScatter:        new Float32Array(MAX_ENTITIES).fill(0.3),
};

/**
 * LightningStrike — active lightning bolts.
 */
export const LightningStrike = {
  active:            new Uint8Array(MAX_LIGHTNING_STRIKES),
  posX:              new Float32Array(MAX_LIGHTNING_STRIKES),
  posY:              new Float32Array(MAX_LIGHTNING_STRIKES),
  posZ:              new Float32Array(MAX_LIGHTNING_STRIKES),
  intensity:         new Float32Array(MAX_LIGHTNING_STRIKES),
  duration:          new Float32Array(MAX_LIGHTNING_STRIKES),
  elapsed:           new Float32Array(MAX_LIGHTNING_STRIKES),
  branchCount:       new Uint8Array(MAX_LIGHTNING_STRIKES),
  strikeId:          new Uint32Array(MAX_LIGHTNING_STRIKES),
  zoneEid:           new Int32Array(MAX_LIGHTNING_STRIKES).fill(-1),
  tintR:             new Float32Array(MAX_LIGHTNING_STRIKES).fill(0.9),
  tintG:             new Float32Array(MAX_LIGHTNING_STRIKES).fill(0.9),
  tintB:             new Float32Array(MAX_LIGHTNING_STRIKES).fill(1.0),
  strikeCount:       new Uint32Array(1),
  freeHead:          new Uint8Array(1),
};

/**
 * PrecipitationBudget — per-frame cost accounting for weather.
 */
export const PrecipitationBudget = {
  totalCost:         new Float32Array(1),
  budgetCap:         new Float32Array(1).fill(4.0),
  budgetExceeded:    new Uint8Array(1),
  rainEmitterCount:  new Uint32Array(1),
  snowEmitterCount:  new Uint32Array(1),
  cloudLayerCount:   new Uint32Array(1),
  fogVolumeCount:    new Uint32Array(1),
  lightningActive:   new Uint32Array(1),
};

/**
 * WeatherBlend — cross-fade between two weather presets.
 */
export const WeatherBlend = {
  fromPreset:        new Int16Array(MAX_ENTITIES).fill(-1),
  toPreset:          new Int16Array(MAX_ENTITIES).fill(-1),
  duration:          new Float32Array(MAX_ENTITIES),
  elapsed:           new Float32Array(MAX_ENTITIES),
  alpha:             new Float32Array(MAX_ENTITIES),
  active:            new Uint8Array(MAX_ENTITIES),
};

/**
 * WeatherPreset — named preset registry (per zone). Preset data lives in
 * a static registry below; this SoA just tracks the currently-installed
 * preset id per zone.
 */
export const WeatherPreset = {
  currentPresetId:   new Int16Array(MAX_ENTITIES).fill(-1),
  targetPresetId:    new Int16Array(MAX_ENTITIES).fill(-1),
  presetGeneration:  new Uint32Array(MAX_ENTITIES),
};

/**
 * BiomeClimate — per-biome climate modifiers. Fixed-capacity registry.
 */
export const BiomeClimate = {
  enabled:           new Uint8Array(MAX_BIOME_CLIMATES),
  climateZone:       new Uint8Array(MAX_BIOME_CLIMATES),
  baseTempC:         new Float32Array(MAX_BIOME_CLIMATES),
  targetTempOffsetC: new Float32Array(MAX_BIOME_CLIMATES),
  humidityBias:      new Float32Array(MAX_BIOME_CLIMATES),
  windBias:          new Float32Array(MAX_BIOME_CLIMATES),
  precipBias:        new Float32Array(MAX_BIOME_CLIMATES),
  fogBias:           new Float32Array(MAX_BIOME_CLIMATES),
  heatBias:          new Float32Array(MAX_BIOME_CLIMATES),
  biomeId:           new Uint8Array(MAX_BIOME_CLIMATES),
  registeredCount:   new Uint32Array(1),
};

/**
 * WeatherStats — aggregate per-frame statistics.
 */
export const WeatherStats = {
  activeZones:       new Uint32Array(1),
  activeRain:        new Uint32Array(1),
  activeSnow:        new Uint32Array(1),
  activeFog:         new Uint32Array(1),
  activeHeat:        new Uint32Array(1),
  activeClouds:      new Uint32Array(1),
  activeLightning:   new Uint32Array(1),
  totalZones:        new Uint32Array(1),
  totalRains:        new Uint32Array(1),
  totalSnows:        new Uint32Array(1),
  totalFogs:         new Uint32Array(1),
  totalHeats:        new Uint32Array(1),
  totalClouds:       new Uint32Array(1),
  totalLightnings:   new Uint32Array(1),
};

/**
 * Weather component bundle for bitECS createWorld.
 */
export const WEATHER_COMPONENTS = Object.freeze({
  WeatherZone,
  ClimateState,
  SeasonState,
  WindField,
  RainEmitter,
  SnowEmitter,
  CloudLayer,
  HeatEffect,
  FogVolume,
  LightningStrike,
  PrecipitationBudget,
  WeatherBlend,
  WeatherPreset,
  BiomeClimate,
  WeatherStats,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

const _scratchV3 = new Float32Array(3);
const _scratchV3B = new Float32Array(3);

/* ------------------------------------------------------------------ */
/* 4. STATIC PRESET TABLE                                             */
/* ------------------------------------------------------------------ */

/**
 * A weather preset is a frozen descriptor. Presets are installed by id
 * via `setWeatherPreset(zoneEid, presetId)`.
 */
class WeatherPresetDescriptor {
  constructor(spec) {
    this.id                = spec.id;
    this.name              = spec.name || `preset_${spec.id}`;
    this.weatherType       = spec.weatherType | 0;
    this.seasonAffinity    = spec.seasonAffinity !== undefined ? spec.seasonAffinity | 0 : -1;
    this.baseTempC         = spec.baseTempC !== undefined ? Number(spec.baseTempC) : 15;
    this.humidity          = spec.humidity !== undefined ? Number(spec.humidity) : 0.5;
    this.pressureHPa       = spec.pressureHPa !== undefined ? Number(spec.pressureHPa) : 1013;
    this.cloudCoverage     = spec.cloudCoverage !== undefined ? Number(spec.cloudCoverage) : 0.3;
    this.windSpeed         = spec.windSpeed !== undefined ? Number(spec.windSpeed) : 1.5;
    this.windType          = spec.windType !== undefined ? spec.windType | 0 : WIND_TYPE.BREEZE;

    this.rainEnabled       = !!spec.rainEnabled;
    this.rainType          = spec.rainType !== undefined ? spec.rainType | 0 : RAIN_TYPE.STEADY;
    this.rainIntensity     = spec.rainIntensity !== undefined ? Number(spec.rainIntensity) : 0;

    this.snowEnabled       = !!spec.snowEnabled;
    this.snowType          = spec.snowType !== undefined ? spec.snowType | 0 : SNOW_TYPE.POWDER;
    this.snowIntensity     = spec.snowIntensity !== undefined ? Number(spec.snowIntensity) : 0;

    this.fogEnabled        = !!spec.fogEnabled;
    this.fogDensity        = spec.fogDensity !== undefined ? Number(spec.fogDensity) : 0;
    this.fogColor          = spec.fogColor ? Object.freeze(spec.fogColor.slice()) : Object.freeze([0.7, 0.8, 0.9]);

    this.heatEnabled       = !!spec.heatEnabled;
    this.heatEffectType    = spec.heatEffectType !== undefined ? spec.heatEffectType | 0 : HEAT_EFFECT.NONE;
    this.heatIntensity     = spec.heatIntensity !== undefined ? Number(spec.heatIntensity) : 0;

    this.cloudTypes        = Object.freeze((spec.cloudTypes || []).slice());
    this.lightningChance   = spec.lightningChance !== undefined ? Number(spec.lightningChance) : 0;

    Object.freeze(this);
  }
}

/**
 * Static preset registry. Add new presets here or via registerWeatherPreset.
 */
const _presetRegistry = [];
const _presetById = new Map();

export function registerWeatherPreset(spec) {
  if (!spec || typeof spec.id !== 'number') return -1;
  if (_presetById.has(spec.id)) return spec.id;
  const d = new WeatherPresetDescriptor(spec);
  _presetRegistry.push(d);
  _presetById.set(d.id, d);
  return d.id;
}

export function getWeatherPreset(id) {
  return _presetById.get(id) || null;
}

export function getWeatherPresetCount() {
  return _presetRegistry.length;
}

/* ------------------------------------------------------------------ */
/* 5. CANONICAL PRESETS                                               */
/* ------------------------------------------------------------------ */

export const WEATHER_PRESET = Object.freeze({
  CLEAR_DAY:            0,
  CLEAR_NIGHT:          1,
  LIGHT_CLOUDY:         2,
  OVERCAST:             3,
  SPRING_DRIZZLE:       4,
  SUMMER_RAIN:          5,
  AUTUMN_RAIN:          6,
  WINTER_SNOW_LIGHT:    7,
  WINTER_SNOW_HEAVY:    8,
  BLIZZARD:             9,
  SPRING_FOG:          10,
  COASTAL_HAZE:        11,
  DESERT_HEAT:         12,
  DESERT_SANDSTORM:    13,
  TROPICAL_MONSOON:    14,
  THUNDERSTORM:        15,
  HAILSTORM:           16,
  AURORA_NIGHT:        17,
  VOLCANIC_ASH:        18,
  TUNDRA_CLEAR:        19,
  FOREST_MIST:         20,
});

(function _registerCanonicalPresets() {
  registerWeatherPreset({
    id: WEATHER_PRESET.CLEAR_DAY,
    name: 'clear_day',
    weatherType: WEATHER_TYPE.CLEAR,
    baseTempC: 22, humidity: 0.4, pressureHPa: 1018,
    cloudCoverage: 0.1, windSpeed: 1.2, windType: WIND_TYPE.BREEZE,
    cloudTypes: [CLOUD_TYPE.CIRRUS, CLOUD_TYPE.CUMULUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.CLEAR_NIGHT,
    name: 'clear_night',
    weatherType: WEATHER_TYPE.CLEAR,
    baseTempC: 12, humidity: 0.5, pressureHPa: 1016,
    cloudCoverage: 0.05, windSpeed: 0.8, windType: WIND_TYPE.CALM,
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.LIGHT_CLOUDY,
    name: 'light_cloudy',
    weatherType: WEATHER_TYPE.CLOUDY,
    baseTempC: 20, humidity: 0.55, pressureHPa: 1014,
    cloudCoverage: 0.4, windSpeed: 2.0, windType: WIND_TYPE.BREEZE,
    cloudTypes: [CLOUD_TYPE.CUMULUS, CLOUD_TYPE.STRATOCUMULUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.OVERCAST,
    name: 'overcast',
    weatherType: WEATHER_TYPE.OVERCAST,
    baseTempC: 16, humidity: 0.70, pressureHPa: 1010,
    cloudCoverage: 0.85, windSpeed: 3.0, windType: WIND_TYPE.BREEZE,
    cloudTypes: [CLOUD_TYPE.STRATUS, CLOUD_TYPE.NIMBOSTRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.SPRING_DRIZZLE,
    name: 'spring_drizzle',
    weatherType: WEATHER_TYPE.RAIN_LIGHT,
    seasonAffinity: SEASON.SPRING,
    baseTempC: 14, humidity: 0.80, pressureHPa: 1012,
    cloudCoverage: 0.7, windSpeed: 1.8,
    rainEnabled: true, rainType: RAIN_TYPE.DRIZZLE, rainIntensity: 0.35,
    cloudTypes: [CLOUD_TYPE.STRATOCUMULUS, CLOUD_TYPE.NIMBOSTRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.SUMMER_RAIN,
    name: 'summer_rain',
    weatherType: WEATHER_TYPE.RAIN_HEAVY,
    seasonAffinity: SEASON.SUMMER,
    baseTempC: 26, humidity: 0.85, pressureHPa: 1006,
    cloudCoverage: 0.75, windSpeed: 4.5, windType: WIND_TYPE.GUSTY,
    rainEnabled: true, rainType: RAIN_TYPE.HEAVY, rainIntensity: 0.75,
    cloudTypes: [CLOUD_TYPE.CUMULONIMBUS, CLOUD_TYPE.NIMBOSTRATUS],
    lightningChance: 0.15,
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.AUTUMN_RAIN,
    name: 'autumn_rain',
    weatherType: WEATHER_TYPE.RAIN_LIGHT,
    seasonAffinity: SEASON.AUTUMN,
    baseTempC: 12, humidity: 0.78, pressureHPa: 1010,
    cloudCoverage: 0.7, windSpeed: 3.5, windType: WIND_TYPE.GUSTY,
    rainEnabled: true, rainType: RAIN_TYPE.STEADY, rainIntensity: 0.5,
    fogEnabled: true, fogDensity: 0.2,
    cloudTypes: [CLOUD_TYPE.STRATUS, CLOUD_TYPE.NIMBOSTRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.WINTER_SNOW_LIGHT,
    name: 'winter_snow_light',
    weatherType: WEATHER_TYPE.SNOW_LIGHT,
    seasonAffinity: SEASON.WINTER,
    baseTempC: -3, humidity: 0.75, pressureHPa: 1015,
    cloudCoverage: 0.65, windSpeed: 2.0,
    snowEnabled: true, snowType: SNOW_TYPE.POWDER, snowIntensity: 0.4,
    cloudTypes: [CLOUD_TYPE.ALTOCUMULUS, CLOUD_TYPE.STRATOCUMULUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.WINTER_SNOW_HEAVY,
    name: 'winter_snow_heavy',
    weatherType: WEATHER_TYPE.SNOW_HEAVY,
    seasonAffinity: SEASON.WINTER,
    baseTempC: -8, humidity: 0.85, pressureHPa: 1012,
    cloudCoverage: 0.85, windSpeed: 4.0, windType: WIND_TYPE.GUSTY,
    snowEnabled: true, snowType: SNOW_TYPE.WET, snowIntensity: 0.85,
    fogEnabled: true, fogDensity: 0.35,
    cloudTypes: [CLOUD_TYPE.NIMBOSTRATUS, CLOUD_TYPE.STRATOCUMULUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.BLIZZARD,
    name: 'blizzard',
    weatherType: WEATHER_TYPE.SNOW_BLIZZARD,
    seasonAffinity: SEASON.WINTER,
    baseTempC: -15, humidity: 0.90, pressureHPa: 1004,
    cloudCoverage: 0.95, windSpeed: 12.0, windType: WIND_TYPE.GALE,
    snowEnabled: true, snowType: SNOW_TYPE.BLIZZARD, snowIntensity: 1.0,
    fogEnabled: true, fogDensity: 0.6,
    cloudTypes: [CLOUD_TYPE.NIMBOSTRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.SPRING_FOG,
    name: 'spring_fog',
    weatherType: WEATHER_TYPE.FOG_LIGHT,
    seasonAffinity: SEASON.SPRING,
    baseTempC: 10, humidity: 0.92, pressureHPa: 1018,
    cloudCoverage: 0.6, windSpeed: 0.5, windType: WIND_TYPE.CALM,
    fogEnabled: true, fogDensity: 0.5, fogColor: [0.85, 0.88, 0.92],
    cloudTypes: [CLOUD_TYPE.STRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.COASTAL_HAZE,
    name: 'coastal_haze',
    weatherType: WEATHER_TYPE.HAZE,
    baseTempC: 18, humidity: 0.80, pressureHPa: 1014,
    cloudCoverage: 0.4, windSpeed: 3.0,
    fogEnabled: true, fogDensity: 0.3, fogColor: [0.75, 0.85, 0.95],
    cloudTypes: [CLOUD_TYPE.STRATOCUMULUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.DESERT_HEAT,
    name: 'desert_heat',
    weatherType: WEATHER_TYPE.HEAT_WAVE,
    baseTempC: 42, humidity: 0.10, pressureHPa: 1018,
    cloudCoverage: 0.05, windSpeed: 1.5,
    heatEnabled: true, heatEffectType: HEAT_EFFECT.SHIMMER, heatIntensity: 0.8,
    cloudTypes: [CLOUD_TYPE.NONE],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.DESERT_SANDSTORM,
    name: 'desert_sandstorm',
    weatherType: WEATHER_TYPE.DUST_STORM,
    baseTempC: 38, humidity: 0.15, pressureHPa: 1008,
    cloudCoverage: 0.3, windSpeed: 15.0, windType: WIND_TYPE.HURRICANE,
    fogEnabled: true, fogDensity: 0.7, fogColor: [0.9, 0.7, 0.4],
    heatEnabled: true, heatEffectType: HEAT_EFFECT.SCORCHING, heatIntensity: 0.9,
    cloudTypes: [CLOUD_TYPE.NONE],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.TROPICAL_MONSOON,
    name: 'tropical_monsoon',
    weatherType: WEATHER_TYPE.RAIN_STORM,
    seasonAffinity: SEASON.SUMMER,
    baseTempC: 28, humidity: 0.95, pressureHPa: 1002,
    cloudCoverage: 0.9, windSpeed: 8.0, windType: WIND_TYPE.GALE,
    rainEnabled: true, rainType: RAIN_TYPE.MONSOON, rainIntensity: 1.0,
    cloudTypes: [CLOUD_TYPE.CUMULONIMBUS, CLOUD_TYPE.NIMBOSTRATUS],
    lightningChance: 0.35,
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.THUNDERSTORM,
    name: 'thunderstorm',
    weatherType: WEATHER_TYPE.THUNDERSTORM,
    baseTempC: 20, humidity: 0.90, pressureHPa: 1000,
    cloudCoverage: 0.9, windSpeed: 10.0, windType: WIND_TYPE.STORM,
    rainEnabled: true, rainType: RAIN_TYPE.THUNDER_SHOWER, rainIntensity: 0.9,
    fogEnabled: true, fogDensity: 0.35, fogColor: [0.3, 0.35, 0.45],
    cloudTypes: [CLOUD_TYPE.CUMULONIMBUS],
    lightningChance: 0.6,
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.HAILSTORM,
    name: 'hailstorm',
    weatherType: WEATHER_TYPE.HAIL,
    baseTempC: 5, humidity: 0.85, pressureHPa: 1002,
    cloudCoverage: 0.9, windSpeed: 8.0, windType: WIND_TYPE.GALE,
    snowEnabled: true, snowType: SNOW_TYPE.ICE_PELLET, snowIntensity: 0.8,
    rainEnabled: true, rainType: RAIN_TYPE.HEAVY, rainIntensity: 0.4,
    cloudTypes: [CLOUD_TYPE.CUMULONIMBUS],
    lightningChance: 0.25,
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.AURORA_NIGHT,
    name: 'aurora_night',
    weatherType: WEATHER_TYPE.AURORA_STORM,
    seasonAffinity: SEASON.WINTER,
    baseTempC: -12, humidity: 0.4, pressureHPa: 1018,
    cloudCoverage: 0.1, windSpeed: 2.0,
    cloudTypes: [CLOUD_TYPE.CIRRUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.VOLCANIC_ASH,
    name: 'volcanic_ash',
    weatherType: WEATHER_TYPE.HAZE,
    baseTempC: 30, humidity: 0.6, pressureHPa: 1008,
    cloudCoverage: 0.6, windSpeed: 6.0,
    fogEnabled: true, fogDensity: 0.55, fogColor: [0.4, 0.35, 0.32],
    heatEnabled: true, heatEffectType: HEAT_EFFECT.SHIMMER, heatIntensity: 0.5,
    cloudTypes: [CLOUD_TYPE.NIMBOSTRATUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.TUNDRA_CLEAR,
    name: 'tundra_clear',
    weatherType: WEATHER_TYPE.CLEAR,
    seasonAffinity: SEASON.WINTER,
    baseTempC: -5, humidity: 0.5, pressureHPa: 1020,
    cloudCoverage: 0.2, windSpeed: 3.0,
    cloudTypes: [CLOUD_TYPE.CIRRUS],
  });

  registerWeatherPreset({
    id: WEATHER_PRESET.FOREST_MIST,
    name: 'forest_mist',
    weatherType: WEATHER_TYPE.FOG_LIGHT,
    baseTempC: 15, humidity: 0.88, pressureHPa: 1016,
    cloudCoverage: 0.5, windSpeed: 0.8,
    fogEnabled: true, fogDensity: 0.4, fogColor: [0.75, 0.85, 0.78],
    cloudTypes: [CLOUD_TYPE.STRATUS],
  });
})();

/* ------------------------------------------------------------------ */
/* 6. BIOME CLIMATE REGISTRY                                          */
/* ------------------------------------------------------------------ */

export const BIOME = Object.freeze({
  DESERT: 0,
  SNOW: 1,
  SEA: 2,
  FOREST: 3,
  CANYON: 4,
  COASTAL: 5,
  WETLAND: 6,
  TUNDRA: 7,
  VOLCANIC: 8,
  COUNT: 9,
});

(function _registerCanonicalBiomeClimates() {
  const register = (biomeId, climateZone, baseTempC, humidityBias, windBias, precipBias, fogBias, heatBias) => {
    const idx = BiomeClimate.registeredCount[0]++;
    if (idx >= MAX_BIOME_CLIMATES) return;
    BiomeClimate.enabled[idx] = 1;
    BiomeClimate.biomeId[idx] = biomeId;
    BiomeClimate.climateZone[idx] = climateZone;
    BiomeClimate.baseTempC[idx] = baseTempC;
    BiomeClimate.humidityBias[idx] = humidityBias;
    BiomeClimate.windBias[idx] = windBias;
    BiomeClimate.precipBias[idx] = precipBias;
    BiomeClimate.fogBias[idx] = fogBias;
    BiomeClimate.heatBias[idx] = heatBias;
    BiomeClimate.targetTempOffsetC[idx] = 0;
  };

  register(BIOME.DESERT,   CLIMATE_ZONE.DESERT,      28, -0.35,  0.10, -0.40, -0.30,  0.40);
  register(BIOME.SNOW,     CLIMATE_ZONE.POLAR,       -8,  0.05,  0.20,  0.10,  0.15, -0.30);
  register(BIOME.SEA,      CLIMATE_ZONE.COASTAL,     18,  0.30,  0.30,  0.15,  0.25, -0.05);
  register(BIOME.FOREST,   CLIMATE_ZONE.TEMPERATE,   15,  0.25, -0.10,  0.20,  0.20, -0.10);
  register(BIOME.CANYON,   CLIMATE_ZONE.MOUNTAIN,    12, -0.10,  0.20, -0.10,  0.00,  0.10);
  register(BIOME.COASTAL,  CLIMATE_ZONE.COASTAL,     20,  0.35,  0.25,  0.25,  0.30,  0.00);
  register(BIOME.WETLAND,  CLIMATE_ZONE.SWAMP,       22,  0.45, -0.15,  0.35,  0.40,  0.05);
  register(BIOME.TUNDRA,   CLIMATE_ZONE.TUNDRA,      -2,  0.10,  0.15,  0.05,  0.10, -0.20);
  register(BIOME.VOLCANIC, CLIMATE_ZONE.VOLCANIC,    40, -0.20,  0.30, -0.20,  0.30,  0.60);
})();

/**
 * Returns the biome climate index for a biome id, or -1.
 */
export function findBiomeClimateIndex(biomeId) {
  const count = BiomeClimate.registeredCount[0];
  for (let i = 0; i < count; i++) {
    if (BiomeClimate.enabled[i] === 1 && BiomeClimate.biomeId[i] === biomeId) return i;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/* 7. ZONE REGISTRATION                                               */
/* ------------------------------------------------------------------ */

/**
 * Registers a weather zone on an entity.
 */
export function registerWeatherZone(eid, spec) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const s = spec || {};

  WeatherZone.enabled[eid] = 1;
  WeatherZone.weatherType[eid] = s.weatherType !== undefined ? s.weatherType | 0 : WEATHER_TYPE.CLEAR;
  WeatherZone.weatherTypePrev[eid] = WeatherZone.weatherType[eid];
  WeatherZone.climateZone[eid] = s.climateZone !== undefined ? s.climateZone | 0 : CLIMATE_ZONE.TEMPERATE;
  WeatherZone.season[eid] = s.season !== undefined ? s.season | 0 : SEASON.SPRING;
  WeatherZone.seasonPhase[eid] = s.seasonPhase !== undefined ? Number(s.seasonPhase) : 0;
  WeatherZone.updateInterval[eid] = s.updateInterval !== undefined ? s.updateInterval | 0 : WEATHER_UPDATE_HZ;
  WeatherZone.updateAccum[eid] = 0;
  WeatherZone.priority[eid] = s.priority !== undefined ? s.priority | 0 : 2;
  WeatherZone.flags[eid] = 0;

  ClimateState.temperatureC[eid] = s.temperatureC !== undefined ? Number(s.temperatureC) : 15;
  ClimateState.targetTempC[eid] = ClimateState.temperatureC[eid];
  ClimateState.humidity[eid] = s.humidity !== undefined ? Number(s.humidity) : 0.5;
  ClimateState.targetHumidity[eid] = ClimateState.humidity[eid];
  ClimateState.pressureHPa[eid] = s.pressureHPa !== undefined ? Number(s.pressureHPa) : 1013;
  ClimateState.targetPressureHPa[eid] = ClimateState.pressureHPa[eid];
  ClimateState.dewPointC[eid] = 0;
  ClimateState.windChillC[eid] = ClimateState.temperatureC[eid];
  ClimateState.heatIndexC[eid] = ClimateState.temperatureC[eid];
  ClimateState.cloudCoverage[eid] = s.cloudCoverage !== undefined ? Number(s.cloudCoverage) : 0.3;
  ClimateState.targetCloudCoverage[eid] = ClimateState.cloudCoverage[eid];
  ClimateState.integrationRate[eid] = 0.4;
  ClimateState.lastUpdateFrame[eid] = WeatherState.frame;

  SeasonState.seasonIndex[eid] = WeatherZone.season[eid];
  SeasonState.seasonPhase[eid] = WeatherZone.seasonPhase[eid];
  SeasonState.seasonSpeed[eid] = s.seasonSpeed !== undefined ? Number(s.seasonSpeed) : 1 / 300;
  SeasonState.yearCounter[eid] = 0;
  SeasonState.amplitudeC[eid] = s.seasonAmplitudeC !== undefined ? Number(s.seasonAmplitudeC) : 15;
  SeasonState.baseTempC[eid] = s.seasonBaseTempC !== undefined ? Number(s.seasonBaseTempC) : 15;

  // Wind field.
  WindField.dirX[eid] = 1;
  WindField.dirY[eid] = 0;
  WindField.dirZ[eid] = 0;
  WindField.speed[eid] = s.windSpeed !== undefined ? Number(s.windSpeed) : 1.5;
  WindField.targetSpeed[eid] = WindField.speed[eid];
  WindField.gustSpeed[eid] = 0;
  WindField.gustTimer[eid] = 0;
  WindField.gustInterval[eid] = 4.0;
  WindField.turbulence[eid] = 0.4;
  WindField.windType[eid] = s.windType !== undefined ? s.windType | 0 : WIND_TYPE.BREEZE;
  WindField.prevailingDirX[eid] = 1;
  WindField.prevailingDirZ[eid] = 0;
  WindField.noiseScale[eid] = 0.025;
  WindField.noiseSpeed[eid] = 0.35;
  WindField.noiseAmp[eid] = 0.6;
  WindField.elevationLift[eid] = 0.15;
  WindField.swirlStrength[eid] = 0;

  // Rain.
  RainEmitter.enabled[eid] = 0;
  RainEmitter.rainType[eid] = RAIN_TYPE.STEADY;
  RainEmitter.intensity[eid] = 0;
  RainEmitter.targetIntensity[eid] = 0;
  RainEmitter.dropSize[eid] = 0.02;
  RainEmitter.dropSpeed[eid] = 18;
  RainEmitter.streakLength[eid] = 0.6;
  RainEmitter.alpha[eid] = 0.6;
  RainEmitter.tintR[eid] = 0.55;
  RainEmitter.tintG[eid] = 0.70;
  RainEmitter.tintB[eid] = 0.95;
  RainEmitter.spread[eid] = 0.35;
  RainEmitter.splashStrength[eid] = 0;
  RainEmitter.lightningChance[eid] = 0;
  RainEmitter.state[eid] = PRECIP_STATE.IDLE;
  RainEmitter.fadeTimer[eid] = 0;

  // Snow.
  SnowEmitter.enabled[eid] = 0;
  SnowEmitter.snowType[eid] = SNOW_TYPE.POWDER;
  SnowEmitter.intensity[eid] = 0;
  SnowEmitter.targetIntensity[eid] = 0;
  SnowEmitter.flakeSize[eid] = 0.05;
  SnowEmitter.fallSpeed[eid] = 2.2;
  SnowEmitter.driftStrength[eid] = 0.8;
  SnowEmitter.swirlStrength[eid] = 0.4;
  SnowEmitter.alpha[eid] = 0.9;
  SnowEmitter.tintR[eid] = 0.95;
  SnowEmitter.tintG[eid] = 0.98;
  SnowEmitter.tintB[eid] = 1.00;
  SnowEmitter.spread[eid] = 0.6;
  SnowEmitter.accumulation[eid] = 0;
  SnowEmitter.state[eid] = PRECIP_STATE.IDLE;
  SnowEmitter.fadeTimer[eid] = 0;

  // Cloud layers.
  CloudLayer.layerCount[eid] = 0;

  // Heat effect.
  HeatEffect.enabled[eid] = 0;
  HeatEffect.effectType[eid] = HEAT_EFFECT.NONE;
  HeatEffect.shimmerAmp[eid] = 0;
  HeatEffect.targetShimmerAmp[eid] = 0;
  HeatEffect.mirageStrength[eid] = 0;
  HeatEffect.targetMirage[eid] = 0;
  HeatEffect.scorchR[eid] = 1.0;
  HeatEffect.scorchG[eid] = 0.85;
  HeatEffect.scorchB[eid] = 0.60;
  HeatEffect.waveFreq[eid] = 6.0;
  HeatEffect.waveSpeed[eid] = 1.4;
  HeatEffect.heightFalloff[eid] = 2.0;
  HeatEffect.intensity[eid] = 0;

  // Fog.
  FogVolume.enabled[eid] = 0;
  FogVolume.density[eid] = 0;
  FogVolume.targetDensity[eid] = 0;
  FogVolume.heightBase[eid] = 0;
  FogVolume.heightFalloff[eid] = 0.15;
  FogVolume.colorR[eid] = 0.7;
  FogVolume.colorG[eid] = 0.8;
  FogVolume.colorB[eid] = 0.9;
  FogVolume.near[eid] = 20;
  FogVolume.far[eid] = 200;
  FogVolume.groundBlend[eid] = 0.5;
  FogVolume.horizonBlend[eid] = 0.7;
  FogVolume.sunScatter[eid] = 0.3;

  // Blend / preset.
  WeatherBlend.active[eid] = 0;
  WeatherBlend.alpha[eid] = 1;
  WeatherPreset.currentPresetId[eid] = -1;
  WeatherPreset.targetPresetId[eid] = -1;

  WeatherStats.totalZones[0]++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 8. WEATHER PRESET APPLICATION                                      */
/* ------------------------------------------------------------------ */

/**
 * Installs a weather preset on a zone, immediately.
 */
export function setWeatherPreset(eid, presetId) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const preset = _presetById.get(presetId);
  if (!preset) return false;

  WeatherZone.weatherTypePrev[eid] = WeatherZone.weatherType[eid];
  WeatherZone.weatherType[eid] = preset.weatherType;

  ClimateState.targetTempC[eid] = preset.baseTempC;
  ClimateState.targetHumidity[eid] = preset.humidity;
  ClimateState.targetPressureHPa[eid] = preset.pressureHPa;
  ClimateState.targetCloudCoverage[eid] = preset.cloudCoverage;

  WindField.targetSpeed[eid] = preset.windSpeed;
  WindField.windType[eid] = preset.windType;
  WindField.turbulence[eid] = _windTypeToTurbulence(preset.windType);

  // Rain.
  RainEmitter.enabled[eid] = preset.rainEnabled ? 1 : 0;
  RainEmitter.rainType[eid] = preset.rainType;
  RainEmitter.targetIntensity[eid] = preset.rainIntensity;
  RainEmitter.lightningChance[eid] = preset.lightningChance;
  applyRainProfile(eid, preset.rainType);

  // Snow.
  SnowEmitter.enabled[eid] = preset.snowEnabled ? 1 : 0;
  SnowEmitter.snowType[eid] = preset.snowType;
  SnowEmitter.targetIntensity[eid] = preset.snowIntensity;
  applySnowProfile(eid, preset.snowType);

  // Fog.
  FogVolume.enabled[eid] = preset.fogEnabled ? 1 : 0;
  FogVolume.targetDensity[eid] = preset.fogDensity;
  if (preset.fogColor) {
    FogVolume.colorR[eid] = preset.fogColor[0];
    FogVolume.colorG[eid] = preset.fogColor[1];
    FogVolume.colorB[eid] = preset.fogColor[2];
  }

  // Heat.
  HeatEffect.enabled[eid] = preset.heatEnabled ? 1 : 0;
  HeatEffect.effectType[eid] = preset.heatEffectType;
  HeatEffect.targetShimmerAmp[eid] = preset.heatIntensity * 0.04;
  HeatEffect.targetMirage[eid] = preset.heatIntensity;

  // Cloud layers.
  CloudLayer.layerCount[eid] = 0;
  for (let i = 0; i < preset.cloudTypes.length && i < MAX_CLOUD_LAYERS_PER_ZONE; i++) {
    const flat = eid * MAX_CLOUD_LAYERS_PER_ZONE + i;
    const ct = preset.cloudTypes[i];
    CloudLayer.cloudType[flat] = ct;
    CloudLayer.targetCoverage[flat] = preset.cloudCoverage * (0.6 + 0.2 * i);
    CloudLayer.altitude[flat] = 1.0 + i * 0.3;
    CloudLayer.thickness[flat] = 0.3 + 0.15 * i;
    CloudLayer.density[flat] = 0.6 + 0.1 * i;
    CloudLayer.driftX[flat] = preset.windSpeed * 0.05;
    CloudLayer.driftZ[flat] = 0;
    CloudLayer.shadowStrength[flat] = 0.3;
    CloudLayer.layerCount[eid]++;
  }

  WeatherPreset.currentPresetId[eid] = presetId;
  WeatherPreset.presetGeneration[eid]++;

  WeatherState.totalSetPresets++;
  return true;
}

/**
 * Cross-fades from the current weather to a new preset over `duration`
 * seconds.
 */
export function blendWeatherPreset(eid, presetId, duration) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!_presetById.has(presetId)) return false;

  WeatherBlend.fromPreset[eid] = WeatherPreset.currentPresetId[eid];
  WeatherBlend.toPreset[eid] = presetId;
  WeatherBlend.duration[eid] = Math.max(0.0001, Number(duration) || 3);
  WeatherBlend.elapsed[eid] = 0;
  WeatherBlend.alpha[eid] = 0;
  WeatherBlend.active[eid] = 1;
  WeatherState.totalBlends++;

  return true;
}

function _windTypeToTurbulence(windType) {
  switch (windType) {
    case WIND_TYPE.CALM:      return 0.1;
    case WIND_TYPE.BREEZE:    return 0.3;
    case WIND_TYPE.GUSTY:     return 0.6;
    case WIND_TYPE.GALE:      return 0.8;
    case WIND_TYPE.STORM:     return 0.9;
    case WIND_TYPE.HURRICANE: return 1.0;
    case WIND_TYPE.WHIRLWIND: return 1.0;
    default:                  return 0.4;
  }
}

/* ------------------------------------------------------------------ */
/* 9. RAIN / SNOW PROFILE TABLES                                      */
/* ------------------------------------------------------------------ */

export function applyRainProfile(eid, rainType) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  switch (rainType) {
    case RAIN_TYPE.DRIZZLE:
      RainEmitter.dropSize[eid] = 0.010;
      RainEmitter.dropSpeed[eid] = 8;
      RainEmitter.streakLength[eid] = 0.25;
      RainEmitter.alpha[eid] = 0.45;
      RainEmitter.spread[eid] = 0.20;
      RainEmitter.splashStrength[eid] = 0.15;
      break;
    case RAIN_TYPE.STEADY:
      RainEmitter.dropSize[eid] = 0.020;
      RainEmitter.dropSpeed[eid] = 14;
      RainEmitter.streakLength[eid] = 0.55;
      RainEmitter.alpha[eid] = 0.60;
      RainEmitter.spread[eid] = 0.35;
      RainEmitter.splashStrength[eid] = 0.40;
      break;
    case RAIN_TYPE.HEAVY:
      RainEmitter.dropSize[eid] = 0.030;
      RainEmitter.dropSpeed[eid] = 22;
      RainEmitter.streakLength[eid] = 0.90;
      RainEmitter.alpha[eid] = 0.75;
      RainEmitter.spread[eid] = 0.50;
      RainEmitter.splashStrength[eid] = 0.70;
      break;
    case RAIN_TYPE.MONSOON:
      RainEmitter.dropSize[eid] = 0.045;
      RainEmitter.dropSpeed[eid] = 30;
      RainEmitter.streakLength[eid] = 1.40;
      RainEmitter.alpha[eid] = 0.85;
      RainEmitter.spread[eid] = 0.75;
      RainEmitter.splashStrength[eid] = 1.0;
      break;
    case RAIN_TYPE.THUNDER_SHOWER:
      RainEmitter.dropSize[eid] = 0.035;
      RainEmitter.dropSpeed[eid] = 26;
      RainEmitter.streakLength[eid] = 1.10;
      RainEmitter.alpha[eid] = 0.80;
      RainEmitter.spread[eid] = 0.60;
      RainEmitter.splashStrength[eid] = 0.85;
      break;
    default:
      break;
  }
  return true;
}

export function applySnowProfile(eid, snowType) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  switch (snowType) {
    case SNOW_TYPE.POWDER:
      SnowEmitter.flakeSize[eid] = 0.05;
      SnowEmitter.fallSpeed[eid] = 1.8;
      SnowEmitter.driftStrength[eid] = 0.6;
      SnowEmitter.swirlStrength[eid] = 0.3;
      SnowEmitter.alpha[eid] = 0.90;
      break;
    case SNOW_TYPE.WET:
      SnowEmitter.flakeSize[eid] = 0.09;
      SnowEmitter.fallSpeed[eid] = 3.0;
      SnowEmitter.driftStrength[eid] = 0.4;
      SnowEmitter.swirlStrength[eid] = 0.2;
      SnowEmitter.alpha[eid] = 0.95;
      break;
    case SNOW_TYPE.CRUST:
      SnowEmitter.flakeSize[eid] = 0.07;
      SnowEmitter.fallSpeed[eid] = 2.4;
      SnowEmitter.driftStrength[eid] = 0.5;
      SnowEmitter.swirlStrength[eid] = 0.4;
      SnowEmitter.alpha[eid] = 0.85;
      break;
    case SNOW_TYPE.BLIZZARD:
      SnowEmitter.flakeSize[eid] = 0.06;
      SnowEmitter.fallSpeed[eid] = 5.0;
      SnowEmitter.driftStrength[eid] = 2.0;
      SnowEmitter.swirlStrength[eid] = 1.2;
      SnowEmitter.alpha[eid] = 1.0;
      break;
    case SNOW_TYPE.SLEET:
      SnowEmitter.flakeSize[eid] = 0.03;
      SnowEmitter.fallSpeed[eid] = 6.0;
      SnowEmitter.driftStrength[eid] = 0.2;
      SnowEmitter.swirlStrength[eid] = 0.1;
      SnowEmitter.alpha[eid] = 0.75;
      break;
    case SNOW_TYPE.ICE_PELLET:
      SnowEmitter.flakeSize[eid] = 0.02;
      SnowEmitter.fallSpeed[eid] = 12.0;
      SnowEmitter.driftStrength[eid] = 0.1;
      SnowEmitter.swirlStrength[eid] = 0.05;
      SnowEmitter.alpha[eid] = 0.90;
      break;
    default:
      break;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. SEASON MANAGEMENT                                              */
/* ------------------------------------------------------------------ */

/**
 * Sets the season on a zone.
 */
export function setSeason(eid, season, instant) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (season < 0 || season >= SEASON.COUNT) return false;

  SeasonState.seasonIndex[eid] = season;
  WeatherZone.season[eid] = season;
  if (instant) {
    SeasonState.seasonPhase[eid] = 0;
    WeatherZone.seasonPhase[eid] = 0;
  }
  WeatherState.totalSeasonChanges++;
  return true;
}

/**
 * Advances the season phase by dt. Wraps through the 4 seasons.
 */
export function tickSeason(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  let phase = SeasonState.seasonPhase[eid] + dt * SeasonState.seasonSpeed[eid];
  let season = SeasonState.seasonIndex[eid];

  while (phase >= 1) {
    phase -= 1;
    season = (season + 1) % SEASON.COUNT;
    if (season === 0) SeasonState.yearCounter[eid]++;
  }

  SeasonState.seasonPhase[eid] = phase;
  SeasonState.seasonIndex[eid] = season;
  WeatherZone.season[eid] = season;
  WeatherZone.seasonPhase[eid] = phase;

  return season;
}

/**
 * Returns the seasonal temperature offset for a zone (in Celsius).
 */
export function getSeasonalTempOffset(eid) {
  const season = SeasonState.seasonIndex[eid];
  const phase = SeasonState.seasonPhase[eid];
  const amp = SeasonState.amplitudeC[eid];

  // Use a cosine across the seasonal cycle so that transitions are smooth.
  const seasonRadians = (season + phase) * (Math.PI * 0.5);

  // Summer peaks at SEASON.SUMMER (index 1) + phase ~ 0.5.
  const peak = SEASON.SUMMER + 0.5;
  const angle = (season + phase - peak) * (Math.PI * 0.5);

  return Math.cos(angle) * amp * 0.5;
}

/* ------------------------------------------------------------------ */
/* 11. CLIMATE INTEGRATION                                            */
/* ------------------------------------------------------------------ */

/**
 * Integrates the climate state one step. Called once per frame per zone.
 */
export function updateClimate(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const rate = ClimateState.integrationRate[eid];
  const k = 1 - Math.exp(-rate * dt);

  // Season / time-of-day modulation.
  const seasonalOffset = getSeasonalTempOffset(eid);
  const todOffset = Math.sin(WeatherState.globalTimeOfDay * Math.PI * 2) * 4.0;

  const targetTemp = ClimateState.targetTempC[eid] + seasonalOffset + todOffset;

  ClimateState.temperatureC[eid] += (targetTemp - ClimateState.temperatureC[eid]) * k;
  ClimateState.humidity[eid] += (ClimateState.targetHumidity[eid] - ClimateState.humidity[eid]) * k;
  ClimateState.pressureHPa[eid] += (ClimateState.targetPressureHPa[eid] - ClimateState.pressureHPa[eid]) * k;
  ClimateState.cloudCoverage[eid] += (ClimateState.targetCloudCoverage[eid] - ClimateState.cloudCoverage[eid]) * k;

  // Derived values.
  const T = ClimateState.temperatureC[eid];
  const RH = ClimateState.humidity[eid];

  // Dew point (Magnus formula).
  const a = 17.27;
  const b = 237.7;
  const alpha = (a * T) / (b + T) + Math.log(Math.max(0.01, RH));
  ClimateState.dewPointC[eid] = (b * alpha) / (a - alpha);

  // Wind chill (only below 10°C and wind > 4.8 km/h).
  const windMs = WindField.speed[eid];
  if (T <= 10 && windMs > 1.34) {
    const wc = 13.12 + 0.6215 * T - 11.37 * Math.pow(windMs * 3.6, 0.16) + 0.3965 * T * Math.pow(windMs * 3.6, 0.16);
    ClimateState.windChillC[eid] = wc;
  } else {
    ClimateState.windChillC[eid] = T;
  }

  // Heat index (only above 27°C and RH > 40%).
  if (T >= 27 && RH > 0.4) {
    const TF = T * 9 / 5 + 32;
    const RHP = RH * 100;
    const hi = -42.379 + 2.04901523 * TF + 10.14333127 * RHP
      - 0.22475541 * TF * RHP - 0.00683783 * TF * TF
      - 0.05481717 * RHP * RHP + 0.00122874 * TF * TF * RHP
      + 0.00085282 * TF * RHP * RHP - 0.00000199 * TF * TF * RHP * RHP;
    ClimateState.heatIndexC[eid] = (hi - 32) * 5 / 9;
  } else {
    ClimateState.heatIndexC[eid] = T;
  }

  ClimateState.lastUpdateFrame[eid] = WeatherState.frame;
  WeatherState.totalClimateUpdates++;

  return true;
}

/**
 * Applies a biome climate modifier to a zone.
 */
export function applyBiomeClimate(eid, biomeId) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const idx = findBiomeClimateIndex(biomeId);
  if (idx < 0) return false;

  ClimateState.targetTempC[eid] += BiomeClimate.baseTempC[idx];
  ClimateState.targetHumidity[eid] = _clamp(ClimateState.targetHumidity[eid] + BiomeClimate.humidityBias[idx], 0, 1);
  WindField.targetSpeed[eid] = Math.max(0, WindField.targetSpeed[eid] * (1 + BiomeClimate.windBias[idx]));
  FogVolume.targetDensity[eid] = Math.max(0, FogVolume.targetDensity[eid] * (1 + BiomeClimate.fogBias[idx]));
  HeatEffect.targetShimmerAmp[eid] = Math.max(0, HeatEffect.targetShimmerAmp[eid] * (1 + BiomeClimate.heatBias[idx]));

  return true;
}

/* ------------------------------------------------------------------ */
/* 12. WIND FIELD                                                     */
/* ------------------------------------------------------------------ */

/**
 * Updates the wind field for a zone.
 */
export function updateWindField(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  // Base direction drifts slowly with global time.
  const t = WeatherState.globalTime;
  const scale = WindField.noiseScale[eid];
  const speed = WindField.noiseSpeed[eid];
  const amp = WindField.noiseAmp[eid];

  // Base direction — prevailing wind + drift.
  let dirX = WindField.prevailingDirX[eid];
  let dirZ = WindField.prevailingDirZ[eid];

  // Modulate direction by layered sinusoids to simulate turbulence.
  const t1 = t * speed;
  dirX += Math.sin(t1 * 0.31 + 1.7) * amp * 0.3;
  dirZ += Math.cos(t1 * 0.27 + 4.3) * amp * 0.3;

  // Normalize.
  const len = Math.sqrt(dirX * dirX + dirZ * dirZ) || 1;
  dirX /= len;
  dirZ /= len;

  // Gust cycle.
  WindField.gustTimer[eid] += dt;
  if (WindField.gustTimer[eid] >= WindField.gustInterval[eid]) {
    WindField.gustTimer[eid] = 0;
    // Random-ish gust strength based on turbulence.
    const turb = WindField.turbulence[eid];
    WindField.gustSpeed[eid] = turb * (0.5 + Math.random() * 1.5);
    WindField.gustInterval[eid] = 3 + Math.random() * 6;
  } else {
    WindField.gustSpeed[eid] *= 0.98;
  }

  const targetSpeed = WindField.targetSpeed[eid] + WindField.gustSpeed[eid];
  WindField.speed[eid] += (targetSpeed - WindField.speed[eid]) * (1 - Math.exp(-1.5 * dt));

  WindField.dirX[eid] = dirX;
  WindField.dirY[eid] = 0;
  WindField.dirZ[eid] = dirZ;

  WeatherState.totalWindUpdates++;
  return true;
}

/**
 * Samples the wind vector at a world-space position and time.
 */
export function sampleWindAt(eid, x, y, z, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;

  const scale = WindField.noiseScale[eid];
  const t = WeatherState.globalTime * WindField.noiseSpeed[eid];

  // Layered sinusoidal approximation (allocation-free).
  const nx = Math.sin(x * scale + t) * Math.cos(y * scale * 0.5 + t * 0.7);
  const nz = Math.cos(z * scale + t * 1.3) * Math.sin(y * scale * 0.5 + t * 1.1);
  const base = WindField.speed[eid];

  out[0] = WindField.dirX[eid] * base + nx * base * 0.3;
  out[1] = y * WindField.elevationLift[eid] * base * 0.1;
  out[2] = WindField.dirZ[eid] * base + nz * base * 0.3;
  return out;
}

/* ------------------------------------------------------------------ */
/* 13. PRECIPITATION                                                  */
/* ------------------------------------------------------------------ */

/**
 * Advances rain and snow emitters one step.
 */
export function tickPrecipitation(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const t0 = _now();

  // Rain.
  if (RainEmitter.enabled[eid] === 1) {
    RainEmitter.intensity[eid] += (RainEmitter.targetIntensity[eid] - RainEmitter.intensity[eid])
      * (1 - Math.exp(-1.0 * dt));

    if (RainEmitter.intensity[eid] > 0.02) {
      RainEmitter.state[eid] = PRECIP_STATE.ACTIVE;
    } else {
      RainEmitter.state[eid] = PRECIP_STATE.IDLE;
    }
  }

  // Snow.
  if (SnowEmitter.enabled[eid] === 1) {
    SnowEmitter.intensity[eid] += (SnowEmitter.targetIntensity[eid] - SnowEmitter.intensity[eid])
      * (1 - Math.exp(-0.8 * dt));

    // Accumulation grows slowly, evaporates when no snow.
    if (SnowEmitter.intensity[eid] > 0.02) {
      SnowEmitter.state[eid] = PRECIP_STATE.ACTIVE;
      SnowEmitter.accumulation[eid] = _clamp(
        SnowEmitter.accumulation[eid] + dt * 0.001 * SnowEmitter.intensity[eid],
        0, 1);
    } else {
      SnowEmitter.state[eid] = PRECIP_STATE.IDLE;
      SnowEmitter.accumulation[eid] = Math.max(0, SnowEmitter.accumulation[eid] - dt * 0.0005);
    }
  }

  WeatherState.totalPrecipEmits++;
  WeatherState.lastPrecipTickMs = _now() - t0;
  WeatherState.avgPrecipTickMs += (WeatherState.lastPrecipTickMs - WeatherState.avgPrecipTickMs) * 0.15;

  return true;
}

/**
 * Emits a batch of precipitation particles for the zone. Downstream
 * systems call this to spawn actual particles in their pools.
 * Returns the number of particles to emit.
 */
export function emitPrecipitation(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;

  let emitted = 0;

  if (RainEmitter.enabled[eid] === 1 && RainEmitter.intensity[eid] > 0.02) {
    const rate = RainEmitter.intensity[eid] * 400;
    emitted += Math.floor(rate * dt);
  }

  if (SnowEmitter.enabled[eid] === 1 && SnowEmitter.intensity[eid] > 0.02) {
    const rate = SnowEmitter.intensity[eid] * 200;
    emitted += Math.floor(rate * dt);
  }

  return emitted;
}

/**
 * Sets the rain type and updates the profile.
 */
export function setRainType(eid, rainType, intensity) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  RainEmitter.rainType[eid] = rainType | 0;
  applyRainProfile(eid, rainType);
  if (intensity !== undefined) RainEmitter.targetIntensity[eid] = _clamp(Number(intensity), 0, 1);
  return true;
}

/**
 * Sets the snow type and updates the profile.
 */
export function setSnowType(eid, snowType, intensity) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  SnowEmitter.snowType[eid] = snowType | 0;
  applySnowProfile(eid, snowType);
  if (intensity !== undefined) SnowEmitter.targetIntensity[eid] = _clamp(Number(intensity), 0, 1);
  return true;
}

/* ------------------------------------------------------------------ */
/* 14. CLOUDS                                                         */
/* ------------------------------------------------------------------ */

/**
 * Sets the cloud type of a layer.
 */
export function setCloudType(eid, layerIdx, cloudType) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (layerIdx < 0 || layerIdx >= MAX_CLOUD_LAYERS_PER_ZONE) return false;
  const flat = eid * MAX_CLOUD_LAYERS_PER_ZONE + layerIdx;
  CloudLayer.cloudType[flat] = cloudType | 0;
  if (layerIdx >= CloudLayer.layerCount[eid]) CloudLayer.layerCount[eid] = layerIdx + 1;
  return true;
}

/**
 * Updates cloud coverage on all layers of a zone.
 */
export function updateCloudCoverage(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const count = CloudLayer.layerCount[eid];
  const k = 1 - Math.exp(-0.8 * dt);
  const base = ClimateState.cloudCoverage[eid];

  for (let i = 0; i < count; i++) {
    const flat = eid * MAX_CLOUD_LAYERS_PER_ZONE + i;
    const tgt = CloudLayer.targetCoverage[flat] * (base / Math.max(0.001, ClimateState.targetCloudCoverage[eid]));
    CloudLayer.coverage[flat] += (tgt - CloudLayer.coverage[flat]) * k;
    CloudLayer.coverage[flat] = _clamp(CloudLayer.coverage[flat], 0, 1);
  }

  return true;
}

/* ------------------------------------------------------------------ */
/* 15. HEAT EFFECT                                                    */
/* ------------------------------------------------------------------ */

/**
 * Updates heat shimmer / mirage for a zone.
 */
export function updateHeatEffect(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (HeatEffect.enabled[eid] === 0) return false;

  const k = 1 - Math.exp(-1.2 * dt);

  // Shimmer amp grows with temperature above 25°C.
  const T = ClimateState.temperatureC[eid];
  const tempFactor = _clamp((T - 25) / 20, 0, 1);
  const targetAmp = HeatEffect.targetShimmerAmp[eid] * tempFactor;

  HeatEffect.shimmerAmp[eid] += (targetAmp - HeatEffect.shimmerAmp[eid]) * k;
  HeatEffect.mirageStrength[eid] += (HeatEffect.targetMirage[eid] * tempFactor - HeatEffect.mirageStrength[eid]) * k;
  HeatEffect.intensity[eid] = Math.max(HeatEffect.shimmerAmp[eid] * 25, HeatEffect.mirageStrength[eid]);

  WeatherState.totalHeatUpdates++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 16. FOG                                                            */
/* ------------------------------------------------------------------ */

/**
 * Updates fog density, color, and distance.
 */
export function updateFogVolume(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (FogVolume.enabled[eid] === 0) return false;

  const k = 1 - Math.exp(-0.6 * dt);
  FogVolume.density[eid] += (FogVolume.targetDensity[eid] - FogVolume.density[eid]) * k;

  // Near / far follow the density.
  FogVolume.near[eid] = _lerp(40, 5, FogVolume.density[eid]);
  FogVolume.far[eid] = _lerp(400, 60, FogVolume.density[eid]);

  WeatherState.totalFogUpdates++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 17. LIGHTNING                                                      */
/* ------------------------------------------------------------------ */

/**
 * Triggers a lightning strike in the given zone at the given position.
 */
export function castLightningStrike(eid, x, y, z, intensity, duration) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  // Find a free slot.
  let slot = -1;
  for (let i = 0; i < MAX_LIGHTNING_STRIKES; i++) {
    if (LightningStrike.active[i] === 0) {
      slot = i;
      break;
    }
  }
  if (slot < 0) return false;

  LightningStrike.active[slot] = 1;
  LightningStrike.posX[slot] = x;
  LightningStrike.posY[slot] = y;
  LightningStrike.posZ[slot] = z;
  LightningStrike.intensity[slot] = intensity !== undefined ? Number(intensity) : 1.0;
  LightningStrike.duration[slot] = duration !== undefined ? Number(duration) : 0.3;
  LightningStrike.elapsed[slot] = 0;
  LightningStrike.branchCount[slot] = 3 + ((Math.random() * 4) | 0);
  LightningStrike.strikeId[slot] = LightningStrike.strikeCount[0]++;
  LightningStrike.zoneEid[slot] = eid;
  LightningStrike.tintR[slot] = 0.9;
  LightningStrike.tintG[slot] = 0.9;
  LightningStrike.tintB[slot] = 1.0;

  WeatherState.totalLightningStrikes++;
  return true;
}

/**
 * Advances all active lightning strikes.
 */
export function tickLightning(dt) {
  let active = 0;
  for (let i = 0; i < MAX_LIGHTNING_STRIKES; i++) {
    if (LightningStrike.active[i] === 0) continue;
    LightningStrike.elapsed[i] += dt;
    if (LightningStrike.elapsed[i] >= LightningStrike.duration[i]) {
      LightningStrike.active[i] = 0;
    } else {
      active++;
    }
  }
  WeatherStats.activeLightning[0] = active;
  return active;
}

/**
 * Probabilistically fires a lightning strike for a zone based on its
 * rain emitter's lightningChance.
 */
export function maybeCastLightning(eid, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const chance = RainEmitter.lightningChance[eid];
  if (chance <= 0) return false;
  if (RainEmitter.intensity[eid] < 0.3) return false;
  if (Math.random() > chance * 0.02) return false;
  return castLightningStrike(eid, x, y, z, 1.0, 0.35);
}

/* ------------------------------------------------------------------ */
/* 18. WEATHER BLEND                                                  */
/* ------------------------------------------------------------------ */

/**
 * Advances any active preset blend for a zone.
 */
export function tickWeatherBlend(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (WeatherBlend.active[eid] === 0) return false;

  WeatherBlend.elapsed[eid] += dt;
  const dur = WeatherBlend.duration[eid];
  const alpha = _clamp(WeatherBlend.elapsed[eid] / dur, 0, 1);
  WeatherBlend.alpha[eid] = alpha;

  if (alpha >= 1) {
    WeatherBlend.active[eid] = 0;
    const toPreset = WeatherBlend.toPreset[eid];
    if (toPreset >= 0) {
      setWeatherPreset(eid, toPreset);
    }
  }

  return true;
}

/* ------------------------------------------------------------------ */
/* 19. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the weather system frame counter.
 */
export function tickWeatherFrame(frameNumber) {
  if (typeof frameNumber === 'number') WeatherState.frame = frameNumber;
  else WeatherState.frame++;
}

/**
 * Full per-frame weather pipeline. Called once per frame by the engine
 * loop with the frame delta in seconds.
 */
export function tickWeather(dt) {
  const t0 = _now();

  WeatherState.globalTime += dt;
  WeatherState.globalTimeOfDay = (WeatherState.globalTimeOfDay + dt * WeatherState.globalDaySpeed) % 1;
  WeatherState.globalSeasonPhase += dt * WeatherState.globalSeasonSpeed;
  if (WeatherState.globalSeasonPhase >= 1) {
    WeatherState.globalSeasonPhase -= 1;
    WeatherState.globalSeason = (WeatherState.globalSeason + 1) % SEASON.COUNT;
  }

  const adapter = getAdapter();
  let activeZones = 0;
  let activeRain = 0;
  let activeSnow = 0;
  let activeFog = 0;
  let activeHeat = 0;
  let activeClouds = 0;

  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (WeatherZone.enabled[eid] === 0) continue;
    if (!adapter.entityAlive(eid)) continue;
    activeZones++;

    // FPS-gated update interval — weather doesn't need full frame rate.
    WeatherZone.updateAccum[eid] += dt;
    const interval = 1 / Math.max(1, WeatherZone.updateInterval[eid]);
    if (WeatherZone.updateAccum[eid] < interval) continue;
    const stepDt = WeatherZone.updateAccum[eid];
    WeatherZone.updateAccum[eid] = 0;

    tickSeason(eid, stepDt);
    updateClimate(eid, stepDt);
    updateWindField(eid, stepDt);
    tickPrecipitation(eid, stepDt);
    updateCloudCoverage(eid, stepDt);
    updateHeatEffect(eid, stepDt);
    updateFogVolume(eid, stepDt);
    tickWeatherBlend(eid, stepDt);

    if (RainEmitter.state[eid] === PRECIP_STATE.ACTIVE) activeRain++;
    if (SnowEmitter.state[eid] === PRECIP_STATE.ACTIVE) activeSnow++;
    if (FogVolume.enabled[eid] === 1 && FogVolume.density[eid] > 0.02) activeFog++;
    if (HeatEffect.enabled[eid] === 1 && HeatEffect.intensity[eid] > 0.02) activeHeat++;
    if (CloudLayer.layerCount[eid] > 0) activeClouds++;
  }

  const lightning = tickLightning(dt);

  WeatherStats.activeZones[0] = activeZones;
  WeatherStats.activeRain[0] = activeRain;
  WeatherStats.activeSnow[0] = activeSnow;
  WeatherStats.activeFog[0] = activeFog;
  WeatherStats.activeHeat[0] = activeHeat;
  WeatherStats.activeClouds[0] = activeClouds;

  PrecipitationBudget.rainEmitterCount[0] = activeRain;
  PrecipitationBudget.snowEmitterCount[0] = activeSnow;
  PrecipitationBudget.fogVolumeCount[0] = activeFog;
  PrecipitationBudget.lightningActive[0] = lightning;

  // Budget check.
  const cost = activeRain * 0.5 + activeSnow * 0.7 + activeFog * 0.3 + lightning * 0.2;
  PrecipitationBudget.totalCost[0] = cost;
  PrecipitationBudget.budgetExceeded[0] = cost > PrecipitationBudget.budgetCap[0] ? 1 : 0;

  const t1 = _now();
  WeatherState.lastTickMs = t1 - t0;
  WeatherState.avgTickMs += (WeatherState.lastTickMs - WeatherState.avgTickMs) * 0.15;

  return {
    frame: WeatherState.frame,
    activeZones,
    activeRain,
    activeSnow,
    activeFog,
    activeHeat,
    activeClouds,
    activeLightning: lightning,
    globalSeason: WeatherState.globalSeason,
    globalTimeOfDay: WeatherState.globalTimeOfDay,
    cost,
  };
}

/* ------------------------------------------------------------------ */
/* 20. WEATHER PRESET REGISTRATION WITH COMPONENT REGISTRY            */
/* ------------------------------------------------------------------ */

export function registerWeatherComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'WeatherZone',          component: WeatherZone,          category: 9, subsystem: 7, dependencies: [] },
    { name: 'ClimateState',         component: ClimateState,         category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'SeasonState',          component: SeasonState,          category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'WindField',            component: WindField,            category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'RainEmitter',          component: RainEmitter,          category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'SnowEmitter',          component: SnowEmitter,          category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'CloudLayer',           component: CloudLayer,           category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'HeatEffect',           component: HeatEffect,           category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'FogVolume',            component: FogVolume,            category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'LightningStrike',      component: LightningStrike,      category: 9, subsystem: 7, dependencies: [] },
    { name: 'PrecipitationBudget',  component: PrecipitationBudget,  category: 9, subsystem: 7, dependencies: [] },
    { name: 'WeatherBlend',         component: WeatherBlend,         category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'WeatherPreset',        component: WeatherPreset,        category: 9, subsystem: 7, dependencies: ['WeatherZone'] },
    { name: 'BiomeClimate',         component: BiomeClimate,         category: 9, subsystem: 7, dependencies: [] },
    { name: 'WeatherStats',         component: WeatherStats,         category: 9, subsystem: 7, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 21. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getWeatherZoneStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  return {
    entity:            eid,
    enabled:           WeatherZone.enabled[eid] === 1,
    weatherType:       WEATHER_TYPE_NAME[WeatherZone.weatherType[eid]] || 'clear',
    climateZone:       CLIMATE_ZONE_NAME[WeatherZone.climateZone[eid]] || 'temperate',
    season:            SEASON_NAME[SeasonState.seasonIndex[eid]] || 'spring',
    seasonPhase:       SeasonState.seasonPhase[eid],
    temperatureC:      ClimateState.temperatureC[eid],
    targetTempC:       ClimateState.targetTempC[eid],
    humidity:          ClimateState.humidity[eid],
    pressureHPa:       ClimateState.pressureHPa[eid],
    dewPointC:         ClimateState.dewPointC[eid],
    windChillC:        ClimateState.windChillC[eid],
    heatIndexC:        ClimateState.heatIndexC[eid],
    cloudCoverage:     ClimateState.cloudCoverage[eid],
    windSpeed:         WindField.speed[eid],
    windType:          WIND_TYPE_NAME[WindField.windType[eid]] || 'breeze',
    windDir:           [WindField.dirX[eid], WindField.dirY[eid], WindField.dirZ[eid]],
    rainActive:        RainEmitter.state[eid] === PRECIP_STATE.ACTIVE,
    rainType:          RAIN_TYPE_NAME[RainEmitter.rainType[eid]] || 'steady',
    rainIntensity:     RainEmitter.intensity[eid],
    snowActive:        SnowEmitter.state[eid] === PRECIP_STATE.ACTIVE,
    snowType:          SNOW_TYPE_NAME[SnowEmitter.snowType[eid]] || 'powder',
    snowIntensity:     SnowEmitter.intensity[eid],
    snowAccumulation:  SnowEmitter.accumulation[eid],
    cloudLayerCount:   CloudLayer.layerCount[eid],
    fogEnabled:        FogVolume.enabled[eid] === 1,
    fogDensity:        FogVolume.density[eid],
    heatEnabled:       HeatEffect.enabled[eid] === 1,
    heatEffectType:    HEAT_EFFECT_NAME[HeatEffect.effectType[eid]] || 'none',
    heatIntensity:     HeatEffect.intensity[eid],
    blendActive:       WeatherBlend.active[eid] === 1,
    blendAlpha:        WeatherBlend.alpha[eid],
    currentPresetId:   WeatherPreset.currentPresetId[eid],
  };
}

export function getWeatherSystemReport() {
  return {
    frame:                  WeatherState.frame,
    globalTime:             WeatherState.globalTime,
    globalSeason:           SEASON_NAME[WeatherState.globalSeason] || 'spring',
    globalSeasonPhase:      WeatherState.globalSeasonPhase,
    globalTimeOfDay:        WeatherState.globalTimeOfDay,
    totalZones:             WeatherStats.totalZones[0],
    activeZones:            WeatherStats.activeZones[0],
    activeRain:             WeatherStats.activeRain[0],
    activeSnow:             WeatherStats.activeSnow[0],
    activeFog:              WeatherStats.activeFog[0],
    activeHeat:             WeatherStats.activeHeat[0],
    activeClouds:           WeatherStats.activeClouds[0],
    activeLightning:        WeatherStats.activeLightning[0],
    registeredPresets:      getWeatherPresetCount(),
    registeredBiomes:       BiomeClimate.registeredCount[0],
    totalSetPresets:        WeatherState.totalSetPresets,
    totalBlends:            WeatherState.totalBlends,
    totalSeasonChanges:     WeatherState.totalSeasonChanges,
    totalLightningStrikes:  WeatherState.totalLightningStrikes,
    totalPrecipEmits:       WeatherState.totalPrecipEmits,
    totalWindUpdates:       WeatherState.totalWindUpdates,
    totalClimateUpdates:    WeatherState.totalClimateUpdates,
    totalHeatUpdates:       WeatherState.totalHeatUpdates,
    totalFogUpdates:        WeatherState.totalFogUpdates,
    lastTickMs:             WeatherState.lastTickMs,
    avgTickMs:              WeatherState.avgTickMs,
    lastPrecipTickMs:       WeatherState.lastPrecipTickMs,
    avgPrecipTickMs:        WeatherState.avgPrecipTickMs,
    lastWindMs:             WeatherState.lastWindMs,
    avgWindMs:              WeatherState.avgWindMs,
    precipBudget:           PrecipitationBudget.totalCost[0],
    precipBudgetCap:        PrecipitationBudget.budgetCap[0],
    precipBudgetExceeded:   PrecipitationBudget.budgetExceeded[0] === 1,
    perfTier:               PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 22. RESET                                                          */
/* ------------------------------------------------------------------ */

export function resetWeatherState() {
  WeatherZone.enabled.fill(0);
  WeatherZone.weatherType.fill(0);
  WeatherZone.weatherTypePrev.fill(0);
  WeatherZone.climateZone.fill(0);
  WeatherZone.season.fill(0);
  WeatherZone.seasonPhase.fill(0);
  WeatherZone.updateInterval.fill(0);
  WeatherZone.updateAccum.fill(0);
  WeatherZone.priority.fill(0);
  WeatherZone.flags.fill(0);

  ClimateState.temperatureC.fill(0);
  ClimateState.targetTempC.fill(0);
  ClimateState.humidity.fill(0);
  ClimateState.targetHumidity.fill(0);
  ClimateState.pressureHPa.fill(0);
  ClimateState.targetPressureHPa.fill(0);
  ClimateState.dewPointC.fill(0);
  ClimateState.windChillC.fill(0);
  ClimateState.heatIndexC.fill(0);
  ClimateState.cloudCoverage.fill(0);
  ClimateState.targetCloudCoverage.fill(0);
  ClimateState.integrationRate.fill(0.4);
  ClimateState.lastUpdateFrame.fill(0);

  SeasonState.seasonIndex.fill(0);
  SeasonState.seasonPhase.fill(0);
  SeasonState.seasonSpeed.fill(1 / 300);
  SeasonState.yearCounter.fill(0);
  SeasonState.amplitudeC.fill(15);
  SeasonState.baseTempC.fill(15);

  WindField.dirX.fill(0);
  WindField.dirY.fill(0);
  WindField.dirZ.fill(0);
  WindField.speed.fill(0);
  WindField.targetSpeed.fill(0);
  WindField.gustSpeed.fill(0);
  WindField.gustTimer.fill(0);
  WindField.gustInterval.fill(0);
  WindField.turbulence.fill(0);
  WindField.windType.fill(0);
  WindField.prevailingDirX.fill(0);
  WindField.prevailingDirZ.fill(0);
  WindField.noiseScale.fill(0.025);
  WindField.noiseSpeed.fill(0.35);
  WindField.noiseAmp.fill(0.6);
  WindField.elevationLift.fill(0.15);
  WindField.swirlStrength.fill(0);

  RainEmitter.enabled.fill(0);
  RainEmitter.rainType.fill(0);
  RainEmitter.intensity.fill(0);
  RainEmitter.targetIntensity.fill(0);
  RainEmitter.dropSize.fill(0.02);
  RainEmitter.dropSpeed.fill(18);
  RainEmitter.streakLength.fill(0.6);
  RainEmitter.alpha.fill(0.6);
  RainEmitter.tintR.fill(0.55);
  RainEmitter.tintG.fill(0.70);
  RainEmitter.tintB.fill(0.95);
  RainEmitter.spread.fill(0.35);
  RainEmitter.splashStrength.fill(0);
  RainEmitter.lightningChance.fill(0);
  RainEmitter.state.fill(0);
  RainEmitter.fadeTimer.fill(0);

  SnowEmitter.enabled.fill(0);
  SnowEmitter.snowType.fill(0);
  SnowEmitter.intensity.fill(0);
  SnowEmitter.targetIntensity.fill(0);
  SnowEmitter.flakeSize.fill(0.05);
  SnowEmitter.fallSpeed.fill(2.2);
  SnowEmitter.driftStrength.fill(0.8);
  SnowEmitter.swirlStrength.fill(0.4);
  SnowEmitter.alpha.fill(0.9);
  SnowEmitter.tintR.fill(0.95);
  SnowEmitter.tintG.fill(0.98);
  SnowEmitter.tintB.fill(1.0);
  SnowEmitter.spread.fill(0.6);
  SnowEmitter.accumulation.fill(0);
  SnowEmitter.state.fill(0);
  SnowEmitter.fadeTimer.fill(0);

  CloudLayer.cloudType.fill(0);
  CloudLayer.coverage.fill(0);
  CloudLayer.targetCoverage.fill(0);
  CloudLayer.altitude.fill(1.0);
  CloudLayer.thickness.fill(0.4);
  CloudLayer.density.fill(0.7);
  CloudLayer.driftX.fill(0);
  CloudLayer.driftZ.fill(0);
  CloudLayer.tintR.fill(1.0);
  CloudLayer.tintG.fill(1.0);
  CloudLayer.tintB.fill(1.0);
  CloudLayer.shadowStrength.fill(0.3);
  CloudLayer.layerCount.fill(0);

  HeatEffect.enabled.fill(0);
  HeatEffect.effectType.fill(0);
  HeatEffect.shimmerAmp.fill(0);
  HeatEffect.targetShimmerAmp.fill(0);
  HeatEffect.mirageStrength.fill(0);
  HeatEffect.targetMirage.fill(0);
  HeatEffect.scorchR.fill(1.0);
  HeatEffect.scorchG.fill(0.85);
  HeatEffect.scorchB.fill(0.6);
  HeatEffect.waveFreq.fill(6.0);
  HeatEffect.waveSpeed.fill(1.4);
  HeatEffect.heightFalloff.fill(2.0);
  HeatEffect.intensity.fill(0);

  FogVolume.enabled.fill(0);
  FogVolume.density.fill(0);
  FogVolume.targetDensity.fill(0);
  FogVolume.heightBase.fill(0);
  FogVolume.heightFalloff.fill(0.15);
  FogVolume.colorR.fill(0.7);
  FogVolume.colorG.fill(0.8);
  FogVolume.colorB.fill(0.9);
  FogVolume.near.fill(20);
  FogVolume.far.fill(200);
  FogVolume.groundBlend.fill(0.5);
  FogVolume.horizonBlend.fill(0.7);
  FogVolume.sunScatter.fill(0.3);

  LightningStrike.active.fill(0);
  LightningStrike.posX.fill(0);
  LightningStrike.posY.fill(0);
  LightningStrike.posZ.fill(0);
  LightningStrike.intensity.fill(0);
  LightningStrike.duration.fill(0);
  LightningStrike.elapsed.fill(0);
  LightningStrike.branchCount.fill(0);
  LightningStrike.strikeId.fill(0);
  LightningStrike.zoneEid.fill(-1);
  LightningStrike.tintR.fill(0.9);
  LightningStrike.tintG.fill(0.9);
  LightningStrike.tintB.fill(1.0);
  LightningStrike.strikeCount[0] = 0;

  PrecipitationBudget.totalCost[0] = 0;
  PrecipitationBudget.budgetCap[0] = 4.0;
  PrecipitationBudget.budgetExceeded[0] = 0;
  PrecipitationBudget.rainEmitterCount[0] = 0;
  PrecipitationBudget.snowEmitterCount[0] = 0;
  PrecipitationBudget.cloudLayerCount[0] = 0;
  PrecipitationBudget.fogVolumeCount[0] = 0;
  PrecipitationBudget.lightningActive[0] = 0;

  WeatherBlend.fromPreset.fill(-1);
  WeatherBlend.toPreset.fill(-1);
  WeatherBlend.duration.fill(0);
  WeatherBlend.elapsed.fill(0);
  WeatherBlend.alpha.fill(1);
  WeatherBlend.active.fill(0);

  WeatherPreset.currentPresetId.fill(-1);
  WeatherPreset.targetPresetId.fill(-1);
  WeatherPreset.presetGeneration.fill(0);

  BiomeClimate.enabled.fill(0);
  BiomeClimate.climateZone.fill(0);
  BiomeClimate.baseTempC.fill(0);
  BiomeClimate.targetTempOffsetC.fill(0);
  BiomeClimate.humidityBias.fill(0);
  BiomeClimate.windBias.fill(0);
  BiomeClimate.precipBias.fill(0);
  BiomeClimate.fogBias.fill(0);
  BiomeClimate.heatBias.fill(0);
  BiomeClimate.biomeId.fill(0);
  BiomeClimate.registeredCount[0] = 0;

  WeatherStats.activeZones[0] = 0;
  WeatherStats.activeRain[0] = 0;
  WeatherStats.activeSnow[0] = 0;
  WeatherStats.activeFog[0] = 0;
  WeatherStats.activeHeat[0] = 0;
  WeatherStats.activeClouds[0] = 0;
  WeatherStats.activeLightning[0] = 0;
  WeatherStats.totalZones[0] = 0;
  WeatherStats.totalRains[0] = 0;
  WeatherStats.totalSnows[0] = 0;
  WeatherStats.totalFogs[0] = 0;
  WeatherStats.totalHeats[0] = 0;
  WeatherStats.totalClouds[0] = 0;
  WeatherStats.totalLightnings[0] = 0;

  WeatherState.frame = 0;
  WeatherState.subframeAccum = 0;
  WeatherState.totalSetPresets = 0;
  WeatherState.totalBlends = 0;
  WeatherState.totalSeasonChanges = 0;
  WeatherState.totalLightningStrikes = 0;
  WeatherState.totalPrecipEmits = 0;
  WeatherState.totalWindUpdates = 0;
  WeatherState.totalClimateUpdates = 0;
  WeatherState.totalHeatUpdates = 0;
  WeatherState.totalFogUpdates = 0;
  WeatherState.lastTickMs = 0;
  WeatherState.avgTickMs = 0;
  WeatherState.lastPrecipTickMs = 0;
  WeatherState.avgPrecipTickMs = 0;
  WeatherState.lastWindMs = 0;
  WeatherState.avgWindMs = 0;
  WeatherState.globalTime = 0;
  WeatherState.globalSeason = SEASON.SPRING;
  WeatherState.globalSeasonPhase = 0;
  WeatherState.globalTimeOfDay = 0.38;
}

/* ------------------------------------------------------------------ */
/* 23. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_WEATHER_ZONES,
  MAX_PRECIP_EMITTERS,
  MAX_FOG_VOLUMES,
  MAX_LIGHTNING_STRIKES,
  MAX_CLOUD_LAYERS_PER_ZONE,
  MAX_BIOME_CLIMATES,
  WEATHER_UPDATE_HZ,

  // Enums
  WEATHER_TYPE,
  WEATHER_TYPE_NAME,
  SEASON,
  SEASON_NAME,
  RAIN_TYPE,
  RAIN_TYPE_NAME,
  SNOW_TYPE,
  SNOW_TYPE_NAME,
  CLOUD_TYPE,
  CLOUD_TYPE_NAME,
  WIND_TYPE,
  WIND_TYPE_NAME,
  HEAT_EFFECT,
  HEAT_EFFECT_NAME,
  CLIMATE_ZONE,
  CLIMATE_ZONE_NAME,
  PRECIP_STATE,
  BIOME,

  // Preset registry
  WEATHER_PRESET,
  registerWeatherPreset,
  getWeatherPreset,
  getWeatherPresetCount,

  // Components
  WeatherZone,
  ClimateState,
  SeasonState,
  WindField,
  RainEmitter,
  SnowEmitter,
  CloudLayer,
  HeatEffect,
  FogVolume,
  LightningStrike,
  PrecipitationBudget,
  WeatherBlend,
  WeatherPreset,
  BiomeClimate,
  WeatherStats,
  WEATHER_COMPONENTS,

  // Module state
  WeatherState,

  // Zone registration
  registerWeatherZone,

  // Preset control
  setWeatherPreset,
  blendWeatherPreset,

  // Rain / snow profiles
  applyRainProfile,
  applySnowProfile,
  setRainType,
  setSnowType,

  // Season
  setSeason,
  tickSeason,
  getSeasonalTempOffset,

  // Climate
  updateClimate,
  applyBiomeClimate,
  findBiomeClimateIndex,

  // Wind
  updateWindField,
  sampleWindAt,

  // Precipitation
  tickPrecipitation,
  emitPrecipitation,

  // Clouds
  setCloudType,
  updateCloudCoverage,

  // Heat effect
  updateHeatEffect,

  // Fog
  updateFogVolume,

  // Lightning
  castLightningStrike,
  tickLightning,
  maybeCastLightning,

  // Blend
  tickWeatherBlend,

  // Frame
  tickWeatherFrame,
  tickWeather,

  // Diagnostics
  getWeatherZoneStats,
  getWeatherSystemReport,

  // Registration
  registerWeatherComponents,

  // Reset
  resetWeatherState,
};

export default _defaultExport;