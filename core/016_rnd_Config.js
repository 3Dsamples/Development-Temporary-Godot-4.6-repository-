// File : 016
// name : src/core/016_rnd_Config.js
// description : Central configuration hub for the anime lighting stack on
//               Android mobile. Single source of truth for every tunable knob
//               the engine exposes: platform profile, PERF_TIER-derived
//               quality budgets, light counts, shadow resolutions, GI/AO
//               sample budgets, post-processing presets, GPU/CPU memory
//               caps, draw-call caps, and the runtime-override surface that
//               debug UIs and adaptive quality controllers write into.
//
//               Layering (lowest precedence → highest):
//                 1. ENGINE_DEFAULTS      — hardcoded safe values
//                 2. PLATFORM_PROFILE     — Android low/mid/high device class
//                 3. QUALITY_TIER         — LOW / MEDIUM / HIGH / ULTRA
//                 4. ANIME_STYLE_PRESET   — reference-image-matched look
//                 5. PERFORMANCE_PRESET   — balanced / perf / battery
//                 6. Scene / biome override (from world.js)
//                 7. Runtime overrides    — set at runtime by UI / AQ ctrl
//
//               The resolved config is a single frozen plain object with
//               NESTED SECTIONS (lights / shadows / gi / ao / environment /
//               interior / exterior / post / memory / drawCalls / platform).
//               Downstream systems read `CONFIG.lights.maxPointLights` etc.,
//               never re-derive from PERF_TIER themselves — one place to
//               change, one place to audit.
//
//               Optimization rules enforced by this config:
//                 • DPR caps per tier: 1.25 / 1.75 / 2.0 (LOW / MED / HIGH).
//                 • Shadow map size: 512 / 1024 / 2048.
//                 • Shadow cascade count: 1 / 2 / 4.
//                 • Max point lights: 8 / 16 / 32.
//                 • Max spot lights: 4 / 8 / 16.
//                 • Max rect-area lights: 0 / 2 / 4.
//                 • Max shadow-casting lights: 1 / 2 / 4.
//                 • GI probe grid resolution: 8 / 16 / 32.
//                 • GI update budget (Hz): 8 / 15 / 20.
//                 • AO resolution scale: 0.5 / 0.5 / 1.0.
//                 • AO sample count: 4 / 8 / 16.
//                 • Post pass budget: 2 / 4 / 6.
//                 • Max instanced draw calls: 128 / 256 / 512.
//                 • Max triangles: 250K / 750K / 2M.
//                 • GPU memory cap: 48 / 128 / 256 MB.
//                 • CPU heap hint: 128 / 256 / 512 MB.
//                 • Target FPS: 30 / 45 / 60 (with adaptive down to 24 / 30 / 30).
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external config libs; frozen objects for zero-alloc
//               reads; every section allocation-free after boot.
// best for : Guaranteeing one canonical place to read/tune every lighting
//            parameter for the entire stack (006_lgt_LightManager through
//            380_lgt_lights). Any subsystem that asks "how many point lights
//            am I allowed?" or "what's the shadow map resolution?" reads it
//            from here — never re-derives from deviceMemory / hardwareConcurrency.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

/* ------------------------------------------------------------------ */
/* 0. PLATFORM DETECTION (one-shot at module load)                    */
/* ------------------------------------------------------------------ */

function _detectPlatform() {
  const hasWindow = typeof window !== 'undefined';
  const hasNavigator = typeof navigator !== 'undefined';

  const userAgent = hasNavigator ? (navigator.userAgent || '') : '';
  const isAndroid = /Android/i.test(userAgent);
  const isMobile = isAndroid || /iPhone|iPad|iPod|Mobile/i.test(userAgent);
  const isTouch = hasWindow && ('ontouchstart' in window || (hasNavigator && navigator.maxTouchPoints > 0));

  const deviceMemory = hasNavigator && navigator.deviceMemory ? navigator.deviceMemory : 4;
  const hardwareConcurrency = hasNavigator && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 4;
  const devicePixelRatio = hasWindow ? (window.devicePixelRatio || 1) : 1;

  let tier = 'MEDIUM';
  if (deviceMemory >= 8 && hardwareConcurrency >= 8 && devicePixelRatio <= 2.0) {
    tier = 'HIGH';
  } else if (deviceMemory < 4 || hardwareConcurrency < 4) {
    tier = 'LOW';
  }

  return Object.freeze({
    isAndroid,
    isMobile,
    isTouch,
    deviceMemory,
    hardwareConcurrency,
    devicePixelRatio,
    tier,
    vendor: hasNavigator ? (navigator.vendor || '') : '',
    language: hasNavigator ? (navigator.language || 'en') : 'en',
  });
}

export const PLATFORM = _detectPlatform();
export const PERF_TIER = PLATFORM.tier;
export const PERF_RANK = PERF_TIER === 'HIGH' ? 3 : PERF_TIER === 'MEDIUM' ? 2 : 1;

/* ------------------------------------------------------------------ */
/* 1. QUALITY TIER CONSTANTS                                          */
/* ------------------------------------------------------------------ */

export const QUALITY_TIER = Object.freeze({
  LOW:   0,
  MEDIUM:1,
  HIGH:  2,
  ULTRA: 3,
});

export const QUALITY_TIER_NAME = Object.freeze([
  'low',
  'medium',
  'high',
  'ultra',
]);

export const ANIME_STYLE_PRESET = Object.freeze({
  REFERENCE_MATCHED: 0, // exact color match to reference images
  SOFT_CEL:          1, // softer cel bands for cinematic look
  HARD_CEL:          2, // crisp hard cel for classic anime
  PASTEL_CEL:        3, // pastel-tinted cel for hazy scenes
  NIGHT_CEL:         4, // high-contrast night cel
});

export const PERFORMANCE_PRESET = Object.freeze({
  BALANCED: 0, // default — best visual / battery tradeoff
  QUALITY:  1, // prefer visuals, higher battery cost
  PERFORMANCE: 2, // prefer fps, drop res first
  BATTERY:  3, // prefer battery, drop post + shadows first
});

/* ------------------------------------------------------------------ */
/* 2. DEFAULT BUDGET TABLE (all tiers pre-computed, frozen)           */
/* ------------------------------------------------------------------ */

const LOW_BUDGET = Object.freeze({
  dprCap:                 1.25,
  shadowMapSize:          512,
  shadowCascadeCount:     1,
  shadowSoftness:         0.0,
  shadowFilter:           'pcf',
  shadowDistance:         60.0,

  maxPointLights:         8,
  maxSpotLights:          4,
  maxRectAreaLights:      0,
  maxDirectionalLights:   1,
  maxHemisphereLights:    1,
  maxAmbientLights:       1,
  maxShadowCastingLights: 1,
  maxClusterLights:       32,
  clusterGridRes:         [16, 9, 24],

  giProbeGridRes:         8,
  giUpdateBudgetHz:       8,
  giSampleCount:          4,
  giMultiBounce:          false,
  giHalfRes:              true,

  aoResolutionScale:      0.5,
  aoSampleCount:          4,
  aoHalfRes:              true,
  aoTemporalAccum:        false,

  postPassBudget:         2,
  postBloom:              false,
  postVolumetric:         false,
  postSSAO:               false,
  postSSGI:               false,
  postMotionBlur:         false,
  postTAA:                false,

  maxInstancedDrawCalls:  128,
  maxTriangles:           250_000,
  maxGpuMemoryMB:         48,
  maxCpuHeapMB:           128,

  targetFps:              30,
  minFps:                 24,
  adaptiveQuality:        true,
});

const MEDIUM_BUDGET = Object.freeze({
  dprCap:                 1.75,
  shadowMapSize:          1024,
  shadowCascadeCount:     2,
  shadowSoftness:         0.05,
  shadowFilter:           'pcf',
  shadowDistance:         90.0,

  maxPointLights:         16,
  maxSpotLights:          8,
  maxRectAreaLights:      2,
  maxDirectionalLights:   1,
  maxHemisphereLights:    1,
  maxAmbientLights:       1,
  maxShadowCastingLights: 2,
  maxClusterLights:       64,
  clusterGridRes:         [24, 14, 32],

  giProbeGridRes:         16,
  giUpdateBudgetHz:       15,
  giSampleCount:          8,
  giMultiBounce:          true,
  giHalfRes:              true,

  aoResolutionScale:      0.5,
  aoSampleCount:          8,
  aoHalfRes:              true,
  aoTemporalAccum:        true,

  postPassBudget:         4,
  postBloom:              true,
  postVolumetric:         true,
  postSSAO:               true,
  postSSGI:               false,
  postMotionBlur:         false,
  postTAA:                true,

  maxInstancedDrawCalls:  256,
  maxTriangles:           750_000,
  maxGpuMemoryMB:         128,
  maxCpuHeapMB:           256,

  targetFps:              45,
  minFps:                 30,
  adaptiveQuality:        true,
});

const HIGH_BUDGET = Object.freeze({
  dprCap:                 2.0,
  shadowMapSize:          2048,
  shadowCascadeCount:     4,
  shadowSoftness:         0.08,
  shadowFilter:           'pcss',
  shadowDistance:         140.0,

  maxPointLights:         32,
  maxSpotLights:          16,
  maxRectAreaLights:      4,
  maxDirectionalLights:   1,
  maxHemisphereLights:    1,
  maxAmbientLights:       1,
  maxShadowCastingLights: 4,
  maxClusterLights:       128,
  clusterGridRes:         [32, 18, 48],

  giProbeGridRes:         32,
  giUpdateBudgetHz:       20,
  giSampleCount:          16,
  giMultiBounce:          true,
  giHalfRes:              false,

  aoResolutionScale:      1.0,
  aoSampleCount:          16,
  aoHalfRes:              false,
  aoTemporalAccum:        true,

  postPassBudget:         6,
  postBloom:              true,
  postVolumetric:         true,
  postSSAO:               true,
  postSSGI:               true,
  postMotionBlur:         true,
  postTAA:                true,

  maxInstancedDrawCalls:  512,
  maxTriangles:           2_000_000,
  maxGpuMemoryMB:         256,
  maxCpuHeapMB:           512,

  targetFps:              60,
  minFps:                 30,
  adaptiveQuality:        true,
});

export const BUDGET_BY_TIER = Object.freeze({
  LOW:    LOW_BUDGET,
  MEDIUM: MEDIUM_BUDGET,
  HIGH:   HIGH_BUDGET,
});

/* ------------------------------------------------------------------ */
/* 3. ANIME STYLE OVERRIDES (applied on top of tier budget)           */
/* ------------------------------------------------------------------ */

const STYLE_OVERRIDES = Object.freeze({
  REFERENCE_MATCHED: Object.freeze({
    celSteps:        4,
    rimPower:        3.0,
    rimIntensity:    0.45,
    shadowSoftness:  0.05,
    shadowTint:      [0.12, 0.18, 0.30],
    ambientStrength: 0.30,
  }),
  SOFT_CEL: Object.freeze({
    celSteps:        6,
    rimPower:        2.0,
    rimIntensity:    0.35,
    shadowSoftness:  0.15,
    shadowTint:      [0.18, 0.22, 0.34],
    ambientStrength: 0.40,
  }),
  HARD_CEL: Object.freeze({
    celSteps:        3,
    rimPower:        4.0,
    rimIntensity:    0.55,
    shadowSoftness:  0.02,
    shadowTint:      [0.08, 0.12, 0.24],
    ambientStrength: 0.22,
  }),
  PASTEL_CEL: Object.freeze({
    celSteps:        5,
    rimPower:        2.5,
    rimIntensity:    0.50,
    shadowSoftness:  0.10,
    shadowTint:      [0.20, 0.16, 0.28],
    ambientStrength: 0.45,
  }),
  NIGHT_CEL: Object.freeze({
    celSteps:        3,
    rimPower:        3.5,
    rimIntensity:    0.65,
    shadowSoftness:  0.08,
    shadowTint:      [0.05, 0.08, 0.20],
    ambientStrength: 0.18,
  }),
});

/* ------------------------------------------------------------------ */
/* 4. PERFORMANCE PRESET MULTIPLIERS                                  */
/* ------------------------------------------------------------------ */

const PERF_MULTIPLIERS = Object.freeze({
  BALANCED: Object.freeze({
    dprScale:           1.00,
    shadowScale:        1.00,
    giHzScale:          1.00,
    aoScale:            1.00,
    postPassDelta:      0,
    clusterLightScale:  1.00,
    targetFpsDelta:     0,
  }),
  QUALITY: Object.freeze({
    dprScale:           1.00,
    shadowScale:        1.00,
    giHzScale:          1.25,
    aoScale:            1.00,
    postPassDelta:      +1,
    clusterLightScale:  1.00,
    targetFpsDelta:     -5,
  }),
  PERFORMANCE: Object.freeze({
    dprScale:           0.85,
    shadowScale:        0.75,
    giHzScale:          0.75,
    aoScale:            0.75,
    postPassDelta:      -1,
    clusterLightScale:  0.75,
    targetFpsDelta:     +5,
  }),
  BATTERY: Object.freeze({
    dprScale:           0.70,
    shadowScale:        0.50,
    giHzScale:          0.50,
    aoScale:            0.50,
    postPassDelta:      -2,
    clusterLightScale:  0.50,
    targetFpsDelta:     -10,
  }),
});

/* ------------------------------------------------------------------ */
/* 5. RESOLVED CONFIG BUILDER                                         */
/* ------------------------------------------------------------------ */

function _nearestPowerOfTwo(v) {
  let n = 1;
  while (n < v) n <<= 1;
  const lower = n >> 1;
  return (v - lower < n - v) ? lower : n;
}

function _clampInt(v, min, max) {
  return Math.max(min, Math.min(max, v | 0));
}

function _buildResolved(tier, stylePreset, perfPreset, overrides) {
  const base = BUDGET_BY_TIER[tier] || MEDIUM_BUDGET;
  const style = STYLE_OVERRIDES[stylePreset] || STYLE_OVERRIDES.REFERENCE_MATCHED;
  const mult = PERF_MULTIPLIERS[perfPreset] || PERF_MULTIPLIERS.BALANCED;

  const dprCap = Math.min(base.dprCap * mult.dprScale, 2.0);
  const shadowMapSize = _nearestPowerOfTwo(Math.max(256, base.shadowMapSize * mult.shadowScale));
  const giHz = Math.max(4, Math.round(base.giUpdateBudgetHz * mult.giHzScale));
  const aoScale = Math.max(0.25, Math.min(1.0, base.aoResolutionScale * mult.aoScale));
  const clusterLights = Math.max(8, Math.round(base.maxClusterLights * mult.clusterLightScale));
  const postPass = _clampInt(base.postPassBudget + mult.postPassDelta, 0, 8);
  const targetFps = _clampInt(base.targetFps + mult.targetFpsDelta, 24, 120);

  const resolved = {
    // ------------------------------------------------------------
    // Platform block
    // ------------------------------------------------------------
    platform: Object.freeze({
      tier:                tier,
      rank:                PERF_RANK,
      isAndroid:           PLATFORM.isAndroid,
      isMobile:            PLATFORM.isMobile,
      isTouch:             PLATFORM.isTouch,
      deviceMemory:        PLATFORM.deviceMemory,
      hardwareConcurrency: PLATFORM.hardwareConcurrency,
      devicePixelRatio:    PLATFORM.devicePixelRatio,
    }),

    // ------------------------------------------------------------
    // Renderer block (Three.js r185 WebGLRenderer)
    // ------------------------------------------------------------
    renderer: Object.freeze({
      antialias:           false,
      alpha:               false,
      stencil:             false,
      depth:               true,
      powerPreference:     'high-performance',
      preserveDrawingBuffer:false,
      failIfMajorPerformanceCaveat: false,
      dprCap:              dprCap,
      outputColorSpace:    'srgb',
      toneMapping:         'none',
      shadowMapEnabled:    true,
      shadowMapType:       tier === 'HIGH' ? 'pcfsoft' : tier === 'MEDIUM' ? 'pcf' : 'basic',
      sortObjects:         true,
      infoAutoReset:       true,
    }),

    // ------------------------------------------------------------
    // Camera block
    // ------------------------------------------------------------
    camera: Object.freeze({
      fov:                 55,
      near:                0.1,
      far:                 1000,
      logarithmicDepth:    false,
      reverseDepth:        false,
    }),

    // ------------------------------------------------------------
    // Lights block (Three.js r185 only: Ambient / Hemisphere /
    // Directional / Point / Spot / RectArea)
    // ------------------------------------------------------------
    lights: Object.freeze({
      maxAmbientLights:        base.maxAmbientLights,
      maxHemisphereLights:     base.maxHemisphereLights,
      maxDirectionalLights:    base.maxDirectionalLights,
      maxPointLights:          base.maxPointLights,
      maxSpotLights:           base.maxSpotLights,
      maxRectAreaLights:       base.maxRectAreaLights,
      maxShadowCastingLights:  base.maxShadowCastingLights,
      maxClusterLights:        clusterLights,
      clusterGridRes:          Object.freeze(base.clusterGridRes.slice()),
      clusterGridX:            base.clusterGridRes[0],
      clusterGridY:            base.clusterGridRes[1],
      clusterGridZ:            base.clusterGridRes[2],

      physicalUnits:           true,
      defaultIntensity:        1.0,
      defaultColor:            [1.0, 0.96, 0.85],
      defaultDistance:         0,
      defaultDecay:            2,
    }),

    // ------------------------------------------------------------
    // Shadow block
    // ------------------------------------------------------------
    shadows: Object.freeze({
      enabled:                true,
      mapSize:                shadowMapSize,
      cascadeCount:           base.shadowCascadeCount,
      filter:                 base.shadowFilter,
      softness:               style.shadowSoftness,
      distance:               base.shadowDistance,
      bias:                   -0.0008,
      normalBias:             0.020,
      pancakeFix:             tier !== 'LOW',
      animateCasters:         false,
      staticCacheEnabled:     tier !== 'LOW',
      contactShadowEnabled:   tier !== 'LOW',
      tint:                   Object.freeze(style.shadowTint.slice()),
    }),

    // ------------------------------------------------------------
    // GI block
    // ------------------------------------------------------------
    gi: Object.freeze({
      enabled:                true,
      probeGridRes:           base.giProbeGridRes,
      probeSpacing:           4.0,
      probeHeight:            3.5,
      updateBudgetHz:         giHz,
      sampleCount:            base.giSampleCount,
      multiBounce:            base.giMultiBounce,
      halfRes:                base.giHalfRes,
      leakReduction:          0.35,
      skyOcclusionSamples:    tier === 'HIGH' ? 8 : tier === 'MEDIUM' ? 5 : 3,
      bounceStrength:         0.35,
      maxProbeAgeFrames:      120,
    }),

    // ------------------------------------------------------------
    // AO block
    // ------------------------------------------------------------
    ao: Object.freeze({
      enabled:                true,
      resolutionScale:        aoScale,
      sampleCount:            base.aoSampleCount,
      halfRes:                base.aoHalfRes,
      temporalAccum:          base.aoTemporalAccum,
      denoise:                true,
      blurRadius:             4,
      intensity:              1.0,
      radius:                 2.0,
      bias:                   0.025,
      maxDistance:            12.0,
    }),

    // ------------------------------------------------------------
    // Environment block
    // ------------------------------------------------------------
    environment: Object.freeze({
      dayCycleLengthSec:      180,
      autoAdvance:            true,
      windStrength:           0.6,
      cloudCoverage:          0.5,
      fogDensity:             0.0,
      fogNear:                30.0,
      fogFar:                 180.0,
      skyDomeRadius:          220.0,
      starfieldEnabled:       true,
      auroraEnabled:          tier === 'HIGH',
      volumetricEnabled:      base.postVolumetric,
    }),

    // ------------------------------------------------------------
    // Interior block
    // ------------------------------------------------------------
    interior: Object.freeze({
      enabled:                true,
      probeLatticeRes:        tier === 'HIGH' ? 16 : tier === 'MEDIUM' ? 12 : 8,
      updateBudgetHz:         tier === 'HIGH' ? 30 : tier === 'MEDIUM' ? 20 : 10,
      lightPortalCulling:     tier !== 'LOW',
      roomLightPlacer:        true,
      celInteriorShading:     true,
      windowShafts:           tier !== 'LOW',
      doorwayBleed:           true,
      maxInteriorLights:      tier === 'HIGH' ? 24 : tier === 'MEDIUM' ? 12 : 6,
    }),

    // ------------------------------------------------------------
    // Exterior block
    // ------------------------------------------------------------
    exterior: Object.freeze({
      enabled:                true,
      probeLatticeRes:        tier === 'HIGH' ? 24 : tier === 'MEDIUM' ? 16 : 8,
      updateBudgetHz:         tier === 'HIGH' ? 30 : tier === 'MEDIUM' ? 20 : 10,
      groundBounceEnabled:    true,
      cloudBreakEnabled:      tier !== 'LOW',
      heatShimmerEnabled:     tier === 'HIGH',
      snowGlareEnabled:       tier !== 'LOW',
      waterCausticsEnabled:   tier === 'HIGH',
      horizonFadeMeters:      tier === 'HIGH' ? 200 : tier === 'MEDIUM' ? 140 : 80,
    }),

    // ------------------------------------------------------------
    // Post-processing block
    // ------------------------------------------------------------
    post: Object.freeze({
      passBudget:             postPass,
      bloom:                  base.postBloom,
      volumetric:             base.postVolumetric,
      ssao:                   base.postSSAO,
      ssgi:                   base.postSSGI,
      motionBlur:             base.postMotionBlur,
      taa:                    base.postTAA,
      outline:                true,
      dither:                 true,
      pixelQuant:             true,
      chromaticAberration:    tier === 'HIGH',
      filmGrain:              tier !== 'LOW',
      vignette:               true,
      toneMap:                'none',
      colorGrade:             true,
    }),

    // ------------------------------------------------------------
    // Memory / budget block
    // ------------------------------------------------------------
    memory: Object.freeze({
      maxGpuMemoryMB:         base.maxGpuMemoryMB,
      maxCpuHeapMB:           base.maxCpuHeapMB,
      maxRenderTargets:       tier === 'HIGH' ? 256 : tier === 'MEDIUM' ? 128 : 64,
      maxAttributes:          tier === 'HIGH' ? 512 : tier === 'MEDIUM' ? 256 : 128,
      maxInterleaved:         tier === 'HIGH' ? 128 : tier === 'MEDIUM' ? 64 : 32,
      gcIdleFrames:           120,
      leakAuditIntervalFrames:300,
    }),

    // ------------------------------------------------------------
    // Draw-call / geometry budget block
    // ------------------------------------------------------------
    drawCalls: Object.freeze({
      maxInstancedDrawCalls:  base.maxInstancedDrawCalls,
      maxTriangles:           base.maxTriangles,
      maxShadowCasters:       tier === 'HIGH' ? 512 : tier === 'MEDIUM' ? 256 : 128,
      maxLODs:                4,
      forcedLOD:              -1, // -1 = auto
    }),

    // ------------------------------------------------------------
    // Anime style block
    // ------------------------------------------------------------
    style: Object.freeze({
      preset:                 stylePreset,
      presetName:             ANIME_STYLE_PRESET,
      celSteps:               style.celSteps,
      rimPower:               style.rimPower,
      rimIntensity:           style.rimIntensity,
      ambientStrength:        style.ambientStrength,
      shadowTint:             Object.freeze(style.shadowTint.slice()),
      rimColor:               Object.freeze([0.90, 0.95, 1.00]),
      hairSpecular:           true,
      eyeHighlight:           true,
      silhouetteOutline:      true,
    }),

    // ------------------------------------------------------------
    // Performance preset block
    // ------------------------------------------------------------
    performance: Object.freeze({
      preset:                 perfPreset,
      targetFps:              targetFps,
      minFps:                 base.minFps,
      adaptiveQuality:        base.adaptiveQuality,
      lowPowerHz:             30,
      thermalGuard:           true,
      batteryGuard:           true,
      visibilityGuard:        true,
      stutterReducer:         true,
      frameTimeBudgetMs:      Math.round(1000 / targetFps),
    }),
  };

  // Apply runtime overrides (highest precedence, deeply merged one level).
  if (overrides && typeof overrides === 'object') {
    const keys = Object.keys(overrides);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const v = overrides[k];
      if (v && typeof v === 'object' && !Array.isArray(v) && resolved[k] && typeof resolved[k] === 'object') {
        resolved[k] = Object.freeze(Object.assign({}, resolved[k], v));
      } else {
        resolved[k] = v;
      }
    }
  }

  return Object.freeze(resolved);
}

/* ------------------------------------------------------------------ */
/* 6. CONFIG SINGLETON                                                */
/* ------------------------------------------------------------------ */

export class Config {
  constructor(options = {}) {
    this.options = Object.assign({
      tier:            PERF_TIER,
      qualityTier:     PERF_TIER === 'HIGH' ? QUALITY_TIER.HIGH : PERF_TIER === 'MEDIUM' ? QUALITY_TIER.MEDIUM : QUALITY_TIER.LOW,
      stylePreset:     ANIME_STYLE_PRESET.REFERENCE_MATCHED,
      perfPreset:      PERFORMANCE_PRESET.BALANCED,
      overrides:       null,
      frozen:          true,
    }, options || {});

    this._overrides = this.options.overrides ? Object.assign({}, this.options.overrides) : {};
    this._resolved = _buildResolved(
      this.options.tier,
      this.options.stylePreset,
      this.options.perfPreset,
      this._overrides
    );
    this._version = 1;

    this._listeners = new Map();
  }

  /* ---------------- access ---------------- */

  get section() {
    return this._resolved;
  }

  get lights()      { return this._resolved.lights; }
  get shadows()     { return this._resolved.shadows; }
  get gi()          { return this._resolved.gi; }
  get ao()          { return this._resolved.ao; }
  get environment() { return this._resolved.environment; }
  get interior()    { return this._resolved.interior; }
  get exterior()    { return this._resolved.exterior; }
  get post()        { return this._resolved.post; }
  get memory()      { return this._resolved.memory; }
  get drawCalls()   { return this._resolved.drawCalls; }
  get style()       { return this._resolved.style; }
  get performance() { return this._resolved.performance; }
  get renderer()    { return this._resolved.renderer; }
  get platform()    { return this._resolved.platform; }
  get version()     { return this._version; }

  /* ---------------- runtime overrides ---------------- */

  set(path, value) {
    if (!path || typeof path !== 'string') return false;
    const parts = path.split('.');
    if (parts.length === 0) return false;

    let target = this._overrides;
    for (let i = 0; i < parts.length - 1; i++) {
      const k = parts[i];
      if (!target[k] || typeof target[k] !== 'object' || Array.isArray(target[k])) {
        target[k] = {};
      }
      target = target[k];
    }
    target[parts[parts.length - 1]] = value;

    this._version++;
    this._rebuild();
    this._emit('changed', { path, value, version: this._version });
    return true;
  }

  applyOverrides(overrides) {
    if (!overrides || typeof overrides !== 'object') return false;
    const keys = Object.keys(overrides);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const v = overrides[k];
      if (v && typeof v === 'object' && !Array.isArray(v) && this._overrides[k] && typeof this._overrides[k] === 'object') {
        this._overrides[k] = Object.assign({}, this._overrides[k], v);
      } else {
        this._overrides[k] = v;
      }
    }
    this._version++;
    this._rebuild();
    this._emit('changed', { path: '*', overrides, version: this._version });
    return true;
  }

  setStylePreset(stylePreset) {
    this.options.stylePreset = stylePreset;
    this._version++;
    this._rebuild();
    this._emit('style', { stylePreset, version: this._version });
    return true;
  }

  setPerformancePreset(perfPreset) {
    this.options.perfPreset = perfPreset;
    this._version++;
    this._rebuild();
    this._emit('perf', { perfPreset, version: this._version });
    return true;
  }

  setQualityTier(qualityTier) {
    this.options.qualityTier = qualityTier;
    // Quality tier changes the base budget tier as well.
    if (qualityTier === QUALITY_TIER.HIGH) this.options.tier = 'HIGH';
    else if (qualityTier === QUALITY_TIER.MEDIUM) this.options.tier = 'MEDIUM';
    else if (qualityTier === QUALITY_TIER.LOW) this.options.tier = 'LOW';
    else this.options.tier = 'HIGH'; // ULTRA is HIGH base + maxed overrides

    if (qualityTier === QUALITY_TIER.ULTRA) {
      this._overrides = Object.assign({}, this._overrides, {
        lights: { maxPointLights: 64, maxSpotLights: 32, maxRectAreaLights: 8, maxClusterLights: 256 },
        shadows: { mapSize: 4096, cascadeCount: 4 },
        gi: { probeGridRes: 48, updateBudgetHz: 30, sampleCount: 32 },
        ao: { resolutionScale: 1.0, sampleCount: 32 },
        post: { passBudget: 8, ssgi: true },
      });
    }

    this._version++;
    this._rebuild();
    this._emit('quality', { qualityTier, version: this._version });
    return true;
  }

  _rebuild() {
    this._resolved = _buildResolved(
      this.options.tier,
      this.options.stylePreset,
      this.options.perfPreset,
      this._overrides
    );
  }

  /* ---------------- events ---------------- */

  on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let arr = this._listeners.get(event);
    if (!arr) { arr = []; this._listeners.set(event, arr); }
    arr.push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  _emit(event, payload) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      try { arr[i](payload); } catch (e) { console.error(`[016_rnd_Config] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- snapshot ---------------- */

  toJSON() {
    return {
      version: this._version,
      options: Object.assign({}, this.options),
      overrides: JSON.parse(JSON.stringify(this._overrides)),
      resolved: this._resolved,
    };
  }
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultConfig = null;

export function getDefaultConfig() {
  if (!_defaultConfig) _defaultConfig = new Config();
  return _defaultConfig;
}

export function disposeDefaultConfig() {
  _defaultConfig = null;
}

/**
 * Fast access — returns the resolved frozen object for read-only hot paths.
 * Never allocate in per-frame code; call this once at subsystem init and
 * cache the section references.
 */
export function getResolvedConfig() {
  return getDefaultConfig().section;
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createConfig(options = {}) {
  return new Config(options);
}

/* ------------------------------------------------------------------ */
/* 9. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Config,
  createConfig,
  getDefaultConfig,
  disposeDefaultConfig,
  getResolvedConfig,
  PLATFORM,
  PERF_TIER,
  PERF_RANK,
  QUALITY_TIER,
  QUALITY_TIER_NAME,
  ANIME_STYLE_PRESET,
  PERFORMANCE_PRESET,
  BUDGET_BY_TIER,
};

export default _defaultExport;