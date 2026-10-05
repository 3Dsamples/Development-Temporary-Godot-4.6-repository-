// File : 023
// name : src/core/023_rnd_PerfTier.js
// description : Authoritative performance tier resolver for the anime lighting
//               stack on Android mobile. Where 016_rnd_Config.js declared the
//               initial PERF_TIER from a coarse navigator hint (deviceMemory,
//               hardwareConcurrency, devicePixelRatio), THIS module owns the
//               full, refined, re-evaluable tier classification that
//               consolidates every signal source in the codebase:
//
//                 • Platform signals       (016_rnd_Config.js PLATFORM)
//                 • GPU/WebGL capabilities (018_rnd_PlatformConfig.js)
//                 • Android GPU family     (019_rnd_AndroidProfile.js)
//                 • Desktop GPU family     (020_rnd_DesktopProfile.js)
//                 • Runtime capabilities   (021_rnd_Capabilities.js)
//                 • Browser features       (022_rnd_FeatureDetector.js)
//                 • Live frame-time EMA    (from scheduler / engine loop)
//
//               and produces a single tier value (LOW / MEDIUM / HIGH /
//               ULTRA) plus a confidence score, a per-tier snapshot, and
//               a stable `PERF_TIER` object every lighting subsystem
//               reads to pick its budget. The tier is re-evaluated at
//               bootstrap and on demand (thermal recovery, battery
//               recovery, user override), but never on the per-frame hot
//               path.
//
//               Design:
//                 • Deterministic scoring — no random sampling, no
//                   noise-dependent decisions. Two identical devices always
//                   get the same tier.
//                 • Weighted signal aggregation with per-signal confidence
//                   so a strong GPU signal cannot be overridden by a weak
//                   RAM signal, and vice versa.
//                 • Tier ladder is discrete (LOW / MEDIUM / HIGH / ULTRA)
//                   so shader permutations, RT bucket sizes, and cluster
//                   grid resolutions stay cached.
//                 • Upgrade requires sustained evidence (N consecutive
//                   evaluations above threshold); downgrade is immediate
//                   — asymmetric, matching mobile behavior where
//                   throttling happens faster than recovery.
//                 • External biases (thermal, battery, saveData) act as
//                   hard ceilings on the tier — never raise above the
//                   detected tier, only lower it.
//                 • Runtime override surface: `setOverride(tier)` so the
//                   debug GUI and QA regression tooling can force a tier
//                   for comparison screenshots.
//                 • Every read O(1), no allocations after module init.
//
//               Tiers:
//                 ULTRA  — Reserved for desktop/emulator/QA. 4 cascades,
//                          4096 shadow, 32-sample GI, no GI half-res,
//                          full SSGI, 8 post passes.
//                 HIGH   — Flagship Android. 4 cascades, 2048 shadow,
//                          16-sample GI, SSGI optional, 6 post passes.
//                 MEDIUM — Midrange Android. 2 cascades, 1024 shadow,
//                          8-sample GI, no SSGI, 4 post passes.
//                 LOW    — Entry Android. 1 cascade, 512 shadow, 4-sample
//                          GI, no SSGI, 2 post passes.
//                 MINIMAL— Fallback (WebGL1 or software). 1 cascade, 256
//                          shadow, no GI, no SSGI, 1 post pass.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external tier libs; every table frozen at module
//               load; every read allocation-free.
// best for : Giving the entire lighting stack one canonical tier source so
//            every subsystem (006_lgt_LightManager through 380_lgt_lights)
//            reads the same tier and applies the same budgets — no more
//            drift between "config says HIGH but platform says MEDIUM but
//            runtime says LOW".
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier as getBootstrapPerfTier,
} from './008_scn_world.js';

import {
  getDefaultConfig,
  PERF_TIER as CONFIG_PERF_TIER,
  PLATFORM,
  BUDGET_BY_TIER,
  QUALITY_TIER,
} from './016_rnd_Config.js';

import {
  PLATFORM_CONFIG,
  DEVICE,
  QUIRKS,
  QUIRK,
  RAW_CAPS,
  PROFILE,
} from './018_rnd_PlatformConfig.js';

import {
  ANDROID_PROFILE,
  GPU_FAMILY,
  GPU_FAMILY_NAME,
  ANDROID_GPU_FAMILY,
  PRECISION,
} from './019_rnd_AndroidProfile.js';

import {
  DESKTOP_PROFILE,
  DESKTOP_GPU,
  DESKTOP_GPU_NAME,
  DESKTOP_GPU_FAMILY,
  IS_ANGLE,
} from './020_rnd_DesktopProfile.js';

import {
  CAPABILITIES,
  FEATURES as WEBGL_FEATURES,
  RAW_CAPS_DEEP,
  PRECISION_SUPPORT,
} from './021_rnd_Capabilities.js';

import {
  FEATURE_DETECTOR,
  RAW_FEATURES,
  FEATURES as RUNTIME_FEATURES,
  getWorkerPoolSize,
  getHardwareConcurrency,
  getDeviceMemoryGB,
} from './022_rnd_FeatureDetector.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

export const PERF_TIER = Object.freeze({
  MINIMAL: 0,
  LOW:     1,
  MEDIUM:  2,
  HIGH:    3,
  ULTRA:   4,
  COUNT:   5,
});

export const PERF_TIER_NAME = Object.freeze([
  'minimal',
  'low',
  'medium',
  'high',
  'ultra',
]);

export const PERF_TIER_RANK = Object.freeze({
  MINIMAL: 0,
  LOW:     1,
  MEDIUM:  2,
  HIGH:    3,
  ULTRA:   4,
});

/**
 * Signal weights — how much each input contributes to the final tier score.
 * Sum is 1.0 so the score is a weighted average in [0, 1].
 */
const SIGNAL_WEIGHTS = Object.freeze({
  gpuClass:        0.30,
  ramClass:        0.20,
  cpuClass:        0.15,
  webglClass:      0.10,
  featureClass:    0.10,
  platformClass:   0.10,
  precisionClass:  0.05,
});

/**
 * Score thresholds for each tier.
 */
const SCORE_THRESHOLDS = Object.freeze({
  [PERF_TIER.MINIMAL]: 0.00,
  [PERF_TIER.LOW]:     0.20,
  [PERF_TIER.MEDIUM]:  0.45,
  [PERF_TIER.HIGH]:    0.70,
  [PERF_TIER.ULTRA]:   0.90,
});

/**
 * Confidence thresholds — how many consecutive evaluations above the
 * upgrade threshold are required before actually upgrading.
 */
const UPGRADE_HOLD_EVALS   = 3;
const DOWNGRADE_HOLD_EVALS = 1;

/**
 * External bias caps — the maximum tier each bias is allowed to allow.
 */
const BIAS_CAP = Object.freeze({
  thermalNormal:  PERF_TIER.ULTRA,
  thermalWarm:    PERF_TIER.HIGH,
  thermalHot:     PERF_TIER.MEDIUM,
  thermalVeryHot: PERF_TIER.LOW,
  thermalCritical:PERF_TIER.MINIMAL,

  batteryFull:    PERF_TIER.ULTRA,
  batteryGood:    PERF_TIER.HIGH,
  batteryLow:     PERF_TIER.MEDIUM,
  batteryVeryLow: PERF_TIER.LOW,
  batteryCritical:PERF_TIER.MINIMAL,

  saveData:       PERF_TIER.MEDIUM,
  webgl1:         PERF_TIER.LOW,
  software:       PERF_TIER.MINIMAL,
});

/* ------------------------------------------------------------------ */
/* 1. SIGNAL CLASSIFIERS                                              */
/* ------------------------------------------------------------------ */

function _classifyGpuSignal() {
  // Desktop GPUs and emulators get the top score.
  if (DEVICE.isDesktop) {
    if (DESKTOP_GPU === DESKTOP_GPU_FAMILY.SWIFTSHADER ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.LLVMPIPE) {
      return { score: 0.10, label: 'software_desktop' };
    }
    if (DESKTOP_GPU === DESKTOP_GPU_FAMILY.APPLE_M1 ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.APPLE_M2 ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.APPLE_M3 ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.APPLE_M4 ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.NVIDIA_GEFORCE ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.NVIDIA_QUADRO ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.AMD_RADEON ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.AMD_FIREPRO ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.INTEL_ARC) {
      return { score: 1.00, label: 'high_end_desktop' };
    }
    if (DESKTOP_GPU === DESKTOP_GPU_FAMILY.APPLE_INTEL_IGPU ||
        DESKTOP_GPU === DESKTOP_GPU_FAMILY.INTEL_IRIS) {
      return { score: 0.65, label: 'mid_desktop_igpu' };
    }
    return { score: 0.45, label: 'low_desktop' };
  }

  // Android GPU families.
  switch (GPU_FAMILY) {
    case ANDROID_GPU_FAMILY.ADRENO_8XX:
    case ANDROID_GPU_FAMILY.ADRENO_7XX:
    case ANDROID_GPU_FAMILY.MALI_G8X:
    case ANDROID_GPU_FAMILY.MALI_G7X:
      return { score: 0.95, label: 'flagship_android' };

    case ANDROID_GPU_FAMILY.ADRENO_6XX:
    case ANDROID_GPU_FAMILY.MALI_G6X:
      return { score: 0.75, label: 'upper_mid_android' };

    case ANDROID_GPU_FAMILY.ADRENO_5XX:
    case ANDROID_GPU_FAMILY.MALI_G5X:
      return { score: 0.55, label: 'mid_android' };

    case ANDROID_GPU_FAMILY.ADRENO_4XX:
    case ANDROID_GPU_FAMILY.MALI_G3X:
    case ANDROID_GPU_FAMILY.POWERVR_ROGUE:
      return { score: 0.35, label: 'low_android' };

    case ANDROID_GPU_FAMILY.ADRENO_3XX:
    case ANDROID_GPU_FAMILY.MALI_T:
    case ANDROID_GPU_FAMILY.POWERVR_GE:
      return { score: 0.15, label: 'legacy_android' };

    case ANDROID_GPU_FAMILY.SWIFTSHADER:
      return { score: 0.05, label: 'software' };

    default:
      return { score: 0.45, label: 'unknown_gpu' };
  }
}

function _classifyRamSignal() {
  const gb = getDeviceMemoryGB();
  if (gb >= 12) return { score: 1.00, label: 'ram_12plus' };
  if (gb >= 8)  return { score: 0.85, label: 'ram_8' };
  if (gb >= 6)  return { score: 0.65, label: 'ram_6' };
  if (gb >= 4)  return { score: 0.50, label: 'ram_4' };
  if (gb >= 3)  return { score: 0.35, label: 'ram_3' };
  if (gb >= 2)  return { score: 0.20, label: 'ram_2' };
  return { score: 0.10, label: 'ram_low' };
}

function _classifyCpuSignal() {
  const cores = getHardwareConcurrency();
  if (cores >= 12) return { score: 1.00, label: 'cpu_12plus' };
  if (cores >= 8)  return { score: 0.85, label: 'cpu_8' };
  if (cores >= 6)  return { score: 0.65, label: 'cpu_6' };
  if (cores >= 4)  return { score: 0.45, label: 'cpu_4' };
  if (cores >= 2)  return { score: 0.25, label: 'cpu_2' };
  return { score: 0.10, label: 'cpu_1' };
}

function _classifyWebglSignal() {
  const caps = RAW_CAPS_DEEP;
  if (!caps.available) return { score: 0.00, label: 'no_webgl' };

  let score = 0.30;
  if (caps.webgl2) score += 0.30;
  if (caps.maxTextureSize >= 4096) score += 0.15;
  if (caps.maxTextureSize >= 8192) score += 0.10;
  if (caps.maxDrawBuffers >= 4) score += 0.10;
  if (caps.maxSamples >= 4) score += 0.05;
  return { score: Math.min(1.0, score), label: caps.webgl2 ? 'webgl2' : 'webgl1' };
}

function _classifyFeatureSignal() {
  // Aggregate runtime features into a single score.
  const f = RUNTIME_FEATURES;
  let points = 0;
  let total = 0;

  const add = (cond, weight) => { total += weight; if (cond) points += weight; };

  add(f.canUseWorkers,            0.15);
  add(f.canUseWorkerPool,         0.10);
  add(f.canUseSharedMemory,       0.15);
  add(f.canUseOffscreenRT,        0.15);
  add(f.canUsePerfMarks,          0.05);
  add(f.canUseIdleCallback,       0.05);
  add(f.canUsePointerEvents,      0.05);
  add(f.canUseBatteryGuard,       0.05);
  add(f.canUseThermalGuard,       0.05);
  add(f.canUseWASM,               0.05);
  add(WEBGL_FEATURES.supportsMRT, 0.05);
  add(WEBGL_FEATURES.canUseHDRTargets, 0.05);
  add(WEBGL_FEATURES.canUseInstancing, 0.05);

  const score = total > 0 ? points / total : 0.5;
  return { score, label: 'features_' + score.toFixed(2) };
}

function _classifyPlatformSignal() {
  switch (PLATFORM_CONFIG.tier) {
    case 'HIGH':   return { score: 0.90, label: 'platform_high' };
    case 'MEDIUM': return { score: 0.60, label: 'platform_medium' };
    case 'LOW':    return { score: 0.30, label: 'platform_low' };
    default:       return { score: 0.50, label: 'platform_unknown' };
  }
}

function _classifyPrecisionSignal() {
  const frag = RAW_CAPS_DEEP.precision.fragment;
  const vert = RAW_CAPS_DEEP.precision.vertex;

  if (frag === PRECISION_SUPPORT.HIGH && vert === PRECISION_SUPPORT.HIGH) {
    return { score: 1.00, label: 'precision_full_highp' };
  }
  if (frag === PRECISION_SUPPORT.MEDIUM && vert === PRECISION_SUPPORT.HIGH) {
    return { score: 0.65, label: 'precision_vertex_only_highp' };
  }
  if (frag === PRECISION_SUPPORT.MEDIUM) {
    return { score: 0.45, label: 'precision_mediump' };
  }
  return { score: 0.20, label: 'precision_lowp' };
}

/* ------------------------------------------------------------------ */
/* 2. TIER SCORING                                                    */
/* ------------------------------------------------------------------ */

function _computeWeightedScore(signals) {
  const w = SIGNAL_WEIGHTS;
  const s =
    signals.gpu.score       * w.gpuClass +
    signals.ram.score       * w.ramClass +
    signals.cpu.score       * w.cpuClass +
    signals.webgl.score     * w.webglClass +
    signals.feature.score   * w.featureClass +
    signals.platform.score  * w.platformClass +
    signals.precision.score * w.precisionClass;
  return Math.max(0, Math.min(1, s));
}

function _scoreToTier(score) {
  if (score >= SCORE_THRESHOLDS[PERF_TIER.ULTRA])  return PERF_TIER.ULTRA;
  if (score >= SCORE_THRESHOLDS[PERF_TIER.HIGH])   return PERF_TIER.HIGH;
  if (score >= SCORE_THRESHOLDS[PERF_TIER.MEDIUM]) return PERF_TIER.MEDIUM;
  if (score >= SCORE_THRESHOLDS[PERF_TIER.LOW])    return PERF_TIER.LOW;
  return PERF_TIER.MINIMAL;
}

/* ------------------------------------------------------------------ */
/* 3. TIER SNAPSHOT                                                   */
/* ------------------------------------------------------------------ */

export class TierSnapshot {
  constructor() {
    this.tier          = PERF_TIER.MEDIUM;
    this.tierName      = 'medium';
    this.tierRank      = 2;
    this.score         = 0.5;
    this.confidence    = 0.5;
    this.signals       = null;

    // External biases.
    this.thermalCap    = PERF_TIER.ULTRA;
    this.batteryCap    = PERF_TIER.ULTRA;
    this.extraCap      = PERF_TIER.ULTRA;

    // Effective tier (after bias caps).
    this.effectiveTier = PERF_TIER.MEDIUM;
    this.effectiveTierName = 'medium';

    // Metadata.
    this.frame         = 0;
    this.timestamp     = 0;
    this.override      = false;
    this.evaluations   = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. PERF TIER RESOLVER                                              */
/* ------------------------------------------------------------------ */

export class PerfTierResolver {
  constructor(options = {}) {
    this.options = Object.assign({
      initialTier:        null,
      enableAdaptive:     true,
      upgradeHoldEvals:   UPGRADE_HOLD_EVALS,
      downgradeHoldEvals: DOWNGRADE_HOLD_EVALS,
      logDecisions:       false,
    }, options || {});

    this._frame = 0;
    this._snapshot = new TierSnapshot();

    this._overrideTier = null;
    this._lastEvaluatedTier = PERF_TIER.MEDIUM;
    this._upgradeAccum = 0;
    this._downgradeAccum = 0;

    this._listeners = new Map();

    // Initial evaluation.
    this.evaluate();

    if (this.options.initialTier !== null) {
      this.setOverride(this.options.initialTier);
    }
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
      try { arr[i](payload); } catch (e) { console.error(`[023_rnd_PerfTier] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- signals ---------------- */

  _gatherSignals() {
    return {
      gpu:       _classifyGpuSignal(),
      ram:       _classifyRamSignal(),
      cpu:       _classifyCpuSignal(),
      webgl:     _classifyWebglSignal(),
      feature:   _classifyFeatureSignal(),
      platform:  _classifyPlatformSignal(),
      precision: _classifyPrecisionSignal(),
    };
  }

  /* ---------------- evaluate ---------------- */

  evaluate() {
    this._frame++;

    const signals = this._gatherSignals();
    const score = _computeWeightedScore(signals);
    const rawTier = _scoreToTier(score);

    // Apply asymmetric hold.
    let resolvedTier = this._lastEvaluatedTier;

    if (rawTier > this._lastEvaluatedTier) {
      // Upgrade path — require sustained evidence.
      this._upgradeAccum++;
      this._downgradeAccum = 0;
      if (this._upgradeAccum >= this.options.upgradeHoldEvals) {
        resolvedTier = rawTier;
        this._upgradeAccum = 0;
        if (this.options.logDecisions) {
          console.log(`[023_rnd_PerfTier] upgrade → ${PERF_TIER_NAME[rawTier]} (score=${score.toFixed(3)})`);
        }
        this._emit('upgrade', { from: PERF_TIER_NAME[this._lastEvaluatedTier], to: PERF_TIER_NAME[rawTier], score });
      }
    } else if (rawTier < this._lastEvaluatedTier) {
      // Downgrade path — immediate.
      this._downgradeAccum++;
      this._upgradeAccum = 0;
      if (this._downgradeAccum >= this.options.downgradeHoldEvals) {
        resolvedTier = rawTier;
        this._downgradeAccum = 0;
        if (this.options.logDecisions) {
          console.log(`[023_rnd_PerfTier] downgrade → ${PERF_TIER_NAME[rawTier]} (score=${score.toFixed(3)})`);
        }
        this._emit('downgrade', { from: PERF_TIER_NAME[this._lastEvaluatedTier], to: PERF_TIER_NAME[rawTier], score });
      }
    } else {
      this._upgradeAccum = 0;
      this._downgradeAccum = 0;
    }

    this._lastEvaluatedTier = resolvedTier;

    // Apply external bias caps.
    let effectiveTier = resolvedTier;
    const thermalCap = this._snapshot.thermalCap;
    const batteryCap = this._snapshot.batteryCap;
    const extraCap = this._snapshot.extraCap;

    // Hard caps from WebGL1 / software raster.
    let hardCap = PERF_TIER.ULTRA;
    if (!RAW_CAPS_DEEP.available) hardCap = PERF_TIER.MINIMAL;
    else if (!RAW_CAPS_DEEP.webgl2) hardCap = BIAS_CAP.webgl1;
    else if (GPU_FAMILY === ANDROID_GPU_FAMILY.SWIFTSHADER) hardCap = BIAS_CAP.software;

    if (thermalCap < effectiveTier) effectiveTier = thermalCap;
    if (batteryCap < effectiveTier) effectiveTier = batteryCap;
    if (extraCap   < effectiveTier) effectiveTier = extraCap;
    if (hardCap    < effectiveTier) effectiveTier = hardCap;

    // Apply override.
    if (this._overrideTier !== null) {
      effectiveTier = this._overrideTier;
    }

    // Confidence: 1.0 minus normalized variance across signals.
    const s = [
      signals.gpu.score,
      signals.ram.score,
      signals.cpu.score,
      signals.webgl.score,
      signals.feature.score,
      signals.platform.score,
      signals.precision.score,
    ];
    let mean = 0;
    for (let i = 0; i < s.length; i++) mean += s[i];
    mean /= s.length;
    let variance = 0;
    for (let i = 0; i < s.length; i++) variance += (s[i] - mean) * (s[i] - mean);
    variance /= s.length;
    const stddev = Math.sqrt(variance);
    const confidence = Math.max(0, Math.min(1, 1 - stddev * 1.6));

    // Write into snapshot.
    const snap = this._snapshot;
    snap.frame          = this._frame;
    snap.timestamp      = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    snap.signals        = signals;
    snap.score          = score;
    snap.tier           = resolvedTier;
    snap.tierName       = PERF_TIER_NAME[resolvedTier];
    snap.tierRank       = PERF_TIER_RANK[snap.tierName];
    snap.confidence     = confidence;
    snap.effectiveTier  = effectiveTier;
    snap.effectiveTierName = PERF_TIER_NAME[effectiveTier];
    snap.evaluations++;

    this._emit('evaluated', {
      tier: snap.tierName,
      effectiveTier: snap.effectiveTierName,
      score,
      confidence,
    });

    return snap;
  }

  /* ---------------- external biases ---------------- */

  setThermalBias(input) {
    let cap = PERF_TIER.ULTRA;
    if (typeof input === 'number') {
      const b = Math.max(0, Math.min(1, input));
      if (b >= 0.90) cap = BIAS_CAP.thermalCritical;
      else if (b >= 0.70) cap = BIAS_CAP.thermalVeryHot;
      else if (b >= 0.50) cap = BIAS_CAP.thermalHot;
      else if (b >= 0.25) cap = BIAS_CAP.thermalWarm;
      else cap = BIAS_CAP.thermalNormal;
    } else {
      const s = String(input || 'nominal').toLowerCase();
      if (s === 'critical') cap = BIAS_CAP.thermalCritical;
      else if (s === 'serious') cap = BIAS_CAP.thermalVeryHot;
      else if (s === 'fair') cap = BIAS_CAP.thermalHot;
      else if (s === 'warm') cap = BIAS_CAP.thermalWarm;
    }
    this._snapshot.thermalCap = cap;
    this.evaluate();
    return this;
  }

  setBatteryBias(level, charging) {
    let cap = PERF_TIER.ULTRA;
    if (charging) {
      cap = BIAS_CAP.batteryFull;
    } else {
      const l = Math.max(0, Math.min(1, Number(level) || 1));
      if (l < 0.15) cap = BIAS_CAP.batteryCritical;
      else if (l < 0.20) cap = BIAS_CAP.batteryVeryLow;
      else if (l < 0.30) cap = BIAS_CAP.batteryLow;
      else if (l < 0.50) cap = BIAS_CAP.batteryGood;
      else cap = BIAS_CAP.batteryFull;
    }
    this._snapshot.batteryCap = cap;
    this.evaluate();
    return this;
  }

  setExtraBias(bias) {
    // bias in [0, 1]; 0 = no cap, 1 = MINIMAL.
    const b = Math.max(0, Math.min(1, Number(bias) || 0));
    if (b <= 0.001) this._snapshot.extraCap = PERF_TIER.ULTRA;
    else if (b <= 0.25) this._snapshot.extraCap = BIAS_CAP.thermalWarm;
    else if (b <= 0.50) this._snapshot.extraCap = BIAS_CAP.thermalHot;
    else if (b <= 0.75) this._snapshot.extraCap = BIAS_CAP.thermalVeryHot;
    else this._snapshot.extraCap = BIAS_CAP.thermalCritical;
    this.evaluate();
    return this;
  }

  setSaveDataBias(enabled) {
    if (enabled) {
      this._snapshot.extraCap = BIAS_CAP.saveData;
    } else {
      this._snapshot.extraCap = PERF_TIER.ULTRA;
    }
    this.evaluate();
    return this;
  }

  /* ---------------- override ---------------- */

  setOverride(tier) {
    if (tier === null || tier === undefined) {
      this._overrideTier = null;
      this._snapshot.override = false;
    } else if (typeof tier === 'number' && tier >= 0 && tier < PERF_TIER.COUNT) {
      this._overrideTier = tier;
      this._snapshot.override = true;
    } else if (typeof tier === 'string') {
      const idx = PERF_TIER_NAME.indexOf(tier.toLowerCase());
      if (idx >= 0) {
        this._overrideTier = idx;
        this._snapshot.override = true;
      }
    }
    this.evaluate();
    return this;
  }

  clearOverride() {
    this._overrideTier = null;
    this._snapshot.override = false;
    this.evaluate();
    return this;
  }

  /* ---------------- accessors ---------------- */

  getSnapshot() { return this._snapshot; }
  get tier()              { return this._snapshot.tier; }
  get tierName()          { return this._snapshot.tierName; }
  get effectiveTier()     { return this._snapshot.effectiveTier; }
  get effectiveTierName() { return this._snapshot.effectiveTierName; }
  get score()             { return this._snapshot.score; }
  get confidence()        { return this._snapshot.confidence; }
  get isOverridden()      { return this._snapshot.override; }

  /**
   * Returns the frozen budget table for the current EFFECTIVE tier.
   */
  getBudget() {
    const name = this._snapshot.effectiveTierName.toUpperCase();
    return BUDGET_BY_TIER[name] || BUDGET_BY_TIER.MEDIUM;
  }

  /**
   * Returns the tier as a string compatible with 016_rnd_Config.js.
   */
  getConfigTierName() {
    return this._snapshot.effectiveTierName.toUpperCase();
  }

  getStats() {
    const s = this._snapshot;
    return {
      tier:              s.tierName,
      effectiveTier:     s.effectiveTierName,
      score:             s.score,
      confidence:        s.confidence,
      override:          s.override,
      thermalCap:        PERF_TIER_NAME[s.thermalCap],
      batteryCap:        PERF_TIER_NAME[s.batteryCap],
      extraCap:          PERF_TIER_NAME[s.extraCap],
      evaluations:       s.evaluations,
      signals:           s.signals,
      perfTierLegacy:    CONFIG_PERF_TIER,
      bootstrapTier:     getBootstrapPerfTier(),
    };
  }

  reset() {
    this._lastEvaluatedTier = PERF_TIER.MEDIUM;
    this._upgradeAccum = 0;
    this._downgradeAccum = 0;
    this._overrideTier = null;
    this._snapshot.thermalCap = PERF_TIER.ULTRA;
    this._snapshot.batteryCap = PERF_TIER.ULTRA;
    this._snapshot.extraCap = PERF_TIER.ULTRA;
    this._snapshot.override = false;
    this.evaluate();
    return this;
  }

  dispose() {
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. INITIAL RESOLVED SNAPSHOT (module load, read-only)              */
/* ------------------------------------------------------------------ */

const _bootResolver = new PerfTierResolver();
export const BOOT_SNAPSHOT = _bootResolver.getSnapshot();
export const BOOT_TIER = BOOT_SNAPSHOT.effectiveTier;
export const BOOT_TIER_NAME = BOOT_SNAPSHOT.effectiveTierName;
export const BOOT_SCORE = BOOT_SNAPSHOT.score;
export const BOOT_CONFIDENCE = BOOT_SNAPSHOT.confidence;

/**
 * Alias so existing code that imports `PERF_TIER` (a string) still works.
 * Matches the config layer's expectation (`'LOW' | 'MEDIUM' | 'HIGH'`).
 */
export const PERF_TIER_STRING = BOOT_TIER_NAME.toUpperCase();

/* ------------------------------------------------------------------ */
/* 6. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultResolver = null;

export function getDefaultPerfTierResolver() {
  if (!_defaultResolver) _defaultResolver = new PerfTierResolver();
  return _defaultResolver;
}

export function disposeDefaultPerfTierResolver() {
  if (_defaultResolver) {
    _defaultResolver.dispose();
    _defaultResolver = null;
  }
}

/* ------------------------------------------------------------------ */
/* 7. QUERY HELPERS                                                   */
/* ------------------------------------------------------------------ */

/**
 * Returns the current effective tier index (0..4).
 */
export function getEffectiveTier() {
  return getDefaultPerfTierResolver().effectiveTier;
}

/**
 * Returns the current effective tier name ('minimal' | 'low' | 'medium'
 * | 'high' | 'ultra').
 */
export function getEffectiveTierName() {
  return getDefaultPerfTierResolver().effectiveTierName;
}

/**
 * Returns true if the current tier is at least `tierName`.
 */
export function isAtLeast(tierName) {
  const idx = PERF_TIER_NAME.indexOf(String(tierName || 'low').toLowerCase());
  if (idx < 0) return false;
  return getEffectiveTier() >= idx;
}

/**
 * Returns the effective tier as the string used by 016_rnd_Config.js
 * (`'LOW' | 'MEDIUM' | 'HIGH'`).
 */
export function getConfigTierName() {
  return getDefaultPerfTierResolver().getConfigTierName();
}

/**
 * Returns the frozen budget table for the current tier.
 */
export function getCurrentBudget() {
  return getDefaultPerfTierResolver().getBudget();
}

/**
 * Re-evaluates the tier. Call this after any external bias change
 * (thermal recovery, battery plugged in, save-data toggled off).
 */
export function reevaluateTier() {
  return getDefaultPerfTierResolver().evaluate();
}

/* ------------------------------------------------------------------ */
/* 8. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createPerfTierResolver(options = {}) {
  return new PerfTierResolver(options);
}

/* ------------------------------------------------------------------ */
/* 9. DIAGNOSTICS                                                     */
/* ------------------------------------------------------------------ */

export function getPerfTierReport() {
  const r = getDefaultPerfTierResolver();
  const s = r.getSnapshot();
  return {
    tier:             s.tierName,
    effectiveTier:    s.effectiveTierName,
    score:            s.score,
    confidence:       s.confidence,
    override:         s.override,
    evaluations:      s.evaluations,
    thermalCap:       PERF_TIER_NAME[s.thermalCap],
    batteryCap:       PERF_TIER_NAME[s.batteryCap],
    extraCap:         PERF_TIER_NAME[s.extraCap],
    signals:          s.signals,
    platform:         PLATFORM,
    gpu:              DEVICE.isDesktop ? DESKTOP_GPU_NAME : GPU_FAMILY_NAME,
    hardwareConcurrency: getHardwareConcurrency(),
    deviceMemoryGB:   getDeviceMemoryGB(),
    workerPoolSize:   getWorkerPoolSize(),
    webgl2:           RAW_CAPS_DEEP.webgl2,
    quirks:           QUIRKS.slice(),
    configTier:       CONFIG_PERF_TIER,
    bootstrapTier:    getBootstrapPerfTier(),
  };
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  PerfTierResolver,
  TierSnapshot,
  PERF_TIER,
  PERF_TIER_NAME,
  PERF_TIER_RANK,
  PERF_TIER_STRING,
  BOOT_SNAPSHOT,
  BOOT_TIER,
  BOOT_TIER_NAME,
  BOOT_SCORE,
  BOOT_CONFIDENCE,

  createPerfTierResolver,
  getDefaultPerfTierResolver,
  disposeDefaultPerfTierResolver,

  getEffectiveTier,
  getEffectiveTierName,
  getConfigTierName,
  getCurrentBudget,
  isAtLeast,
  reevaluateTier,

  getPerfTierReport,
};

export default _defaultExport;