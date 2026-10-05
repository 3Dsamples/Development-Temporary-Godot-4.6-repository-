// File : 017
// name : src/core/017_rnd_QualityConfig.js
// description : Runtime quality controller for the anime lighting stack on
//               Android mobile. Where 016_rnd_Config.js defines the STATIC
//               budget table for each PERF_TIER, THIS module owns the
//               DYNAMIC quality surface: a per-frame quality state machine
//               that observes frame-time, thermal pressure, battery level,
//               and per-domain scheduler pressure, then decides — within
//               strictly bounded steps — how to bias resolution scale,
//               shadow map size, GI update rate, AO resolution, post-pass
//               count, and draw distance so the target FPS is held without
//               ever dropping a frame or changing the anime visual identity.
//
//               Design:
//                 • Quality levels are a discrete ladder (0..N) — never
//                   continuous — so shader permutations stay cached and
//                   render-target buckets stay stable.
//                 • Every quality knob has a hysteresis band (upgrade
//                   threshold / downgrade threshold / hold frames) so the
//                   controller never oscillates on Android thermal jitter.
//                 • Only ONE knob changes per decision cycle so a single
//                   bad frame cannot cascade into a full visual downgrade.
//                 • Per-domain pressure is read from the FrameScheduler
//                   (005_rnd_FrameScheduler.js) via `getPressure()`, giving
//                   the controller per-subsystem visibility: shadows may
//                   be over budget while post is idle, and we bias only
//                   shadows in that case.
//                 • Thermal / battery biases from the App layer
//                   (002_rnd_App.js) feed in as hard bounds so the
//                   controller cannot override an OS-level throttle.
//                 • Downgrade is fast (1 frame per step), upgrade is slow
//                   (N frames per step) — matches the asymmetric cost of
//                   GPU state changes on mobile.
//                 • All quality state is a flat structure of Int32/Uint8
//                   fields so it never allocates.
//                 • Exposes a stable `QualitySnapshot` object (pre-allocated,
//                   in-place updated) that shader/GI/AO/shadow systems poll
//                   once per frame — zero allocations on the read path.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external controller libs; every internal array
//               sized once at construction.
// best for : Giving the entire lighting stack one authoritative per-frame
//            quality decision surface. Shadow, GI, AO, cluster, environment,
//            interior, exterior, post all read the same snapshot so a
//            single downgrade keeps them visually coherent — critical for
//            the anime look, which breaks instantly if shadows downscale
//            but GI does not.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getDefaultConfig,
  getResolvedConfig,
  QUALITY_TIER,
  PERFORMANCE_PRESET,
  PERF_TIER,
  PLATFORM,
} from './016_rnd_Config.js';

import {
  getDefaultScheduler,
  DOMAIN,
  DOMAIN_NAME,
} from './005_rnd_FrameScheduler.js';

import {
  getPerfTier,
} from './008_scn_world.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const QUALITY_LEVEL = Object.freeze({
  ULTRA:  0,
  HIGH:   1,
  MEDIUM: 2,
  LOW:    3,
  MINIMAL:4,
  COUNT:  5,
});

export const QUALITY_LEVEL_NAME = Object.freeze([
  'ultra',
  'high',
  'medium',
  'low',
  'minimal',
]);

export const QUALITY_KNOB = Object.freeze({
  NONE:           0,
  RESOLUTION:     1,
  SHADOWS:        2,
  GI:             3,
  AO:             4,
  POST:           5,
  DRAW_DISTANCE:  6,
  CLUSTER:        7,
  COUNT:          8,
});

export const QUALITY_KNOB_NAME = Object.freeze([
  'none',
  'resolution',
  'shadows',
  'gi',
  'ao',
  'post',
  'draw_distance',
  'cluster',
]);

/**
 * Per-quality-level multipliers applied ON TOP of the tier budget from
 * 016_rnd_Config.js. ULTRA is 1.0 everywhere; MINIMAL is the safe floor.
 */
export const QUALITY_LEVEL_MULTIPLIERS = Object.freeze({
  [QUALITY_LEVEL.ULTRA]:   Object.freeze({ resolutionScale: 1.00, shadowScale: 1.00, giHzScale: 1.00, aoScale: 1.00, postDelta:  0, drawDistance: 1.00, clusterScale: 1.00 }),
  [QUALITY_LEVEL.HIGH]:    Object.freeze({ resolutionScale: 0.90, shadowScale: 0.85, giHzScale: 0.85, aoScale: 0.85, postDelta: -1, drawDistance: 0.90, clusterScale: 0.85 }),
  [QUALITY_LEVEL.MEDIUM]:  Object.freeze({ resolutionScale: 0.75, shadowScale: 0.65, giHzScale: 0.65, aoScale: 0.65, postDelta: -2, drawDistance: 0.75, clusterScale: 0.65 }),
  [QUALITY_LEVEL.LOW]:     Object.freeze({ resolutionScale: 0.60, shadowScale: 0.50, giHzScale: 0.50, aoScale: 0.50, postDelta: -3, drawDistance: 0.60, clusterScale: 0.50 }),
  [QUALITY_LEVEL.MINIMAL]: Object.freeze({ resolutionScale: 0.45, shadowScale: 0.35, giHzScale: 0.35, aoScale: 0.35, postDelta: -4, drawDistance: 0.45, clusterScale: 0.35 }),
});

/**
 * Frame-time thresholds for upgrade / downgrade decisions.
 * Values are in milliseconds and are scaled per tier on construction.
 */
const THRESHOLDS = Object.freeze({
  HIGH: Object.freeze({
    upgradeBelowMs:  12.0,  // well within 60 fps budget
    downgradeAboveMs:18.0,  // crossing 55 fps boundary
    upgradeHoldMs:   4000,  // 4 seconds of good frames before upgrade
    downgradeHoldMs: 800,   // 0.8 seconds of bad frames before downgrade
  }),
  MEDIUM: Object.freeze({
    upgradeBelowMs:  18.0,
    downgradeAboveMs:24.0,
    upgradeHoldMs:   3500,
    downgradeHoldMs: 700,
  }),
  LOW: Object.freeze({
    upgradeBelowMs:  30.0,
    downgradeAboveMs:40.0,
    upgradeHoldMs:   3000,
    downgradeHoldMs: 600,
  }),
});

/* ------------------------------------------------------------------ */
/* 1. QUALITY STATE                                                   */
/* ------------------------------------------------------------------ */

export class QualityState {
  constructor(tier) {
    this.tier = tier;

    this.level           = QUALITY_LEVEL.HIGH;
    this.targetLevel     = QUALITY_LEVEL.HIGH;
    this.lastChangeFrame = 0;
    this.changeCount     = 0;

    this.upgradeAccumMs   = 0;
    this.downgradeAccumMs = 0;

    // Per-knob biases (each in [0, 1]; 0 = full quality, 1 = maximal bias).
    this.resolutionBias   = 0;
    this.shadowBias       = 0;
    this.giBias           = 0;
    this.aoBias           = 0;
    this.postBias         = 0;
    this.drawDistanceBias = 0;
    this.clusterBias      = 0;

    // Hard bounds from external sources (App thermal / battery).
    this.thermalBias      = 0;
    this.batteryBias      = 0;
    this.lowPower         = 0;

    // Last decision metadata.
    this.lastKnob         = QUALITY_KNOB.NONE;
    this.lastReason       = 'init';
    this.lastDecisionMs   = 0;

    // Frame counters.
    this.framesObserved   = 0;
    this.framesUpgraded   = 0;
    this.framesDowngraded = 0;
    this.framesHeld       = 0;

    this._thresholds      = THRESHOLDS[tier] || THRESHOLDS.MEDIUM;
  }

  reset() {
    this.level           = QUALITY_LEVEL.HIGH;
    this.targetLevel     = QUALITY_LEVEL.HIGH;
    this.upgradeAccumMs   = 0;
    this.downgradeAccumMs = 0;
    this.resolutionBias   = 0;
    this.shadowBias       = 0;
    this.giBias           = 0;
    this.aoBias           = 0;
    this.postBias         = 0;
    this.drawDistanceBias = 0;
    this.clusterBias      = 0;
    this.thermalBias      = 0;
    this.batteryBias      = 0;
    this.lowPower         = 0;
    this.lastKnob         = QUALITY_KNOB.NONE;
    this.lastReason       = 'reset';
    this.framesObserved   = 0;
    this.framesUpgraded   = 0;
    this.framesDowngraded = 0;
    this.framesHeld       = 0;
  }
}

/* ------------------------------------------------------------------ */
/* 2. QUALITY SNAPSHOT (read-only per-frame view)                     */
/* ------------------------------------------------------------------ */

/**
 * Pre-allocated snapshot updated in place every frame. Downstream systems
 * poll this once per frame — never allocate a new one.
 */
export class QualitySnapshot {
  constructor() {
    this.frame             = 0;
    this.level             = QUALITY_LEVEL.HIGH;
    this.levelName         = 'high';
    this.resolutionScale   = 1.0;
    this.shadowMapSize     = 1024;
    this.shadowCascadeCount= 2;
    this.giUpdateBudgetHz  = 15;
    this.giSampleCount     = 8;
    this.aoResolutionScale = 0.5;
    this.aoSampleCount     = 8;
    this.postPassBudget    = 4;
    this.drawDistanceMeters= 140;
    this.maxClusterLights  = 64;
    this.thermalBias       = 0;
    this.batteryBias       = 0;
    this.lowPower          = 0;
    this.frameTimeMs       = 16.67;
    this.lastKnob          = QUALITY_KNOB.NONE;
    this.lastReason        = 'init';
  }
}

/* ------------------------------------------------------------------ */
/* 3. QUALITY CONTROLLER                                              */
/* ------------------------------------------------------------------ */

export class QualityController {
  constructor(options = {}) {
    this.options = Object.assign({
      initialLevel:     QUALITY_LEVEL.HIGH,
      enableAdaptive:   true,
      enableThermal:    true,
      enableBattery:    true,
      minLevel:         QUALITY_LEVEL.MINIMAL,
      maxLevel:         QUALITY_LEVEL.ULTRA,
      decisionHz:       4,          // controller decision frequency
      logDecisions:     false,
    }, options || {});

    this.config    = getDefaultConfig();
    this.scheduler = getDefaultScheduler();
    this.state     = new QualityState(PERF_TIER_LOCAL);
    this.snapshot  = new QualitySnapshot();

    this._baseBudget = getResolvedConfig();

    this.state.level       = this.options.initialLevel;
    this.state.targetLevel = this.options.initialLevel;

    this._decisionAccumMs = 0;
    this._decisionStepMs  = 1000 / Math.max(1, this.options.decisionHz);
    this._frame           = 0;

    this._domainPressure  = new Float32Array(DOMAIN.COUNT);

    this._listeners = new Map();

    this._applyLevel(true);
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
      try { arr[i](payload); } catch (e) { console.error(`[017_rnd_QualityConfig] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- external bias inputs ---------------- */

  setThermalBias(bias) {
    this.state.thermalBias = Math.max(0, Math.min(1, Number(bias) || 0));
    return this;
  }

  setBatteryBias(bias) {
    this.state.batteryBias = Math.max(0, Math.min(1, Number(bias) || 0));
    return this;
  }

  setLowPower(enabled) {
    this.state.lowPower = enabled ? 1 : 0;
    return this;
  }

  /* ---------------- per-frame update ---------------- */

  update(dtMs, elapsedSec) {
    this._frame++;

    const dt = Number.isFinite(dtMs) ? Math.max(0, Math.min(100, dtMs)) : 16.67;
    this.state.framesObserved++;

    // Update snapshot in place (no allocation).
    this.snapshot.frame       = this._frame;
    this.snapshot.frameTimeMs = dt;

    if (!this.options.enableAdaptive) {
      this._updateSnapshotFromLevel();
      return this.snapshot;
    }

    // External biases take effect immediately — they can only lower quality,
    // never raise it above what the controller would choose.
    if (this.state.lowPower === 1) {
      this._forceLevel(QUALITY_LEVEL.LOW, 'low_power');
    }
    if (this.state.thermalBias >= 0.8) {
      this._forceLevel(QUALITY_LEVEL.LOW, 'thermal_high');
    } else if (this.state.thermalBias >= 0.5) {
      this._forceLevel(QUALITY_LEVEL.MEDIUM, 'thermal_mid');
    }
    if (this.state.batteryBias >= 0.8) {
      this._forceLevel(QUALITY_LEVEL.LOW, 'battery_low');
    } else if (this.state.batteryBias >= 0.5) {
      this._forceLevel(QUALITY_LEVEL.MEDIUM, 'battery_mid');
    }

    // Adaptive decision path (throttled to decisionHz).
    this._decisionAccumMs += dt;
    if (this._decisionAccumMs >= this._decisionStepMs) {
      this._decisionAccumMs = 0;
      this._runDecision(dt);
    }

    // Evaluate per-domain pressure for the next decision.
    const pressure = this.scheduler.getPressure();
    for (let i = 0; i < DOMAIN.COUNT && i < pressure.length; i++) {
      this._domainPressure[i] = pressure[i];
    }

    this._updateSnapshotFromLevel();
    return this.snapshot;
  }

  _runDecision(dtMs) {
    const t = this.state._thresholds;

    // Downgrade: bad frame-time accumulated for downgradeHoldMs.
    if (dtMs > t.downgradeAboveMs) {
      this.state.downgradeAccumMs += this._decisionStepMs;
      this.state.upgradeAccumMs = 0;
    } else if (dtMs < t.upgradeBelowMs) {
      this.state.upgradeAccumMs += this._decisionStepMs;
      this.state.downgradeAccumMs = 0;
    } else {
      // In the safe band — decay both accumulators slowly.
      this.state.upgradeAccumMs   = Math.max(0, this.state.upgradeAccumMs   - this._decisionStepMs * 0.5);
      this.state.downgradeAccumMs = Math.max(0, this.state.downgradeAccumMs - this._decisionStepMs * 0.5);
    }

    // Downgrade when accumulator crosses the hold threshold.
    if (this.state.downgradeAccumMs >= t.downgradeHoldMs) {
      this.state.downgradeAccumMs = 0;
      if (this.state.level < QUALITY_LEVEL.MINIMAL) {
        const knob = this._pickDowngradeKnob();
        this._applyKnobBias(knob, +0.25);
        this._changeLevel(this.state.level + 1, QUALITY_KNOB_NAME[knob], 'downgrade');
        return;
      }
    }

    // Upgrade when accumulator crosses the hold threshold.
    if (this.state.upgradeAccumMs >= t.upgradeHoldMs) {
      this.state.upgradeAccumMs = 0;
      if (this.state.level > QUALITY_LEVEL.ULTRA) {
        // Ensure external biases don't block upgrade.
        if (this.state.thermalBias < 0.2 && this.state.batteryBias < 0.2 && this.state.lowPower === 0) {
          const knob = this._pickUpgradeKnob();
          this._applyKnobBias(knob, -0.20);
          this._changeLevel(this.state.level - 1, QUALITY_KNOB_NAME[knob], 'upgrade');
          return;
        }
      }
    }

    // No level change this cycle.
    this.state.framesHeld++;
  }

  _pickDowngradeKnob() {
    // Find the domain with the highest pressure that has a bias < 1.
    let bestDomain = -1;
    let bestPressure = 0;
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      const p = this._domainPressure[i];
      if (p > bestPressure) {
        bestPressure = p;
        bestDomain = i;
      }
    }

    // Map domain → knob.
    switch (bestDomain) {
      case DOMAIN.SHADOWS: return QUALITY_KNOB.SHADOWS;
      case DOMAIN.GI:      return QUALITY_KNOB.GI;
      case DOMAIN.AO:      return QUALITY_KNOB.AO;
      case DOMAIN.POST:    return QUALITY_KNOB.POST;
      case DOMAIN.CLUSTER: return QUALITY_KNOB.CLUSTER;
      case DOMAIN.LIGHTS:  return QUALITY_KNOB.SHADOWS;
      default:             return QUALITY_KNOB.RESOLUTION;
    }
  }

  _pickUpgradeKnob() {
    // Reverse priority: upgrade the knob that was downgraded last, in order.
    const order = [
      QUALITY_KNOB.RESOLUTION,
      QUALITY_KNOB.SHADOWS,
      QUALITY_KNOB.GI,
      QUALITY_KNOB.AO,
      QUALITY_KNOB.POST,
      QUALITY_KNOB.DRAW_DISTANCE,
      QUALITY_KNOB.CLUSTER,
    ];

    let bestKnob = QUALITY_KNOB.RESOLUTION;
    let bestBias = -1;
    for (let i = 0; i < order.length; i++) {
      const knob = order[i];
      const bias = this._getKnobBias(knob);
      if (bias > bestBias) {
        bestBias = bias;
        bestKnob = knob;
      }
    }
    return bestKnob;
  }

  _getKnobBias(knob) {
    switch (knob) {
      case QUALITY_KNOB.RESOLUTION:    return this.state.resolutionBias;
      case QUALITY_KNOB.SHADOWS:       return this.state.shadowBias;
      case QUALITY_KNOB.GI:            return this.state.giBias;
      case QUALITY_KNOB.AO:            return this.state.aoBias;
      case QUALITY_KNOB.POST:          return this.state.postBias;
      case QUALITY_KNOB.DRAW_DISTANCE: return this.state.drawDistanceBias;
      case QUALITY_KNOB.CLUSTER:       return this.state.clusterBias;
      default:                         return 0;
    }
  }

  _applyKnobBias(knob, delta) {
    const clamp = (v) => Math.max(0, Math.min(1, v + delta));
    switch (knob) {
      case QUALITY_KNOB.RESOLUTION:    this.state.resolutionBias   = clamp(this.state.resolutionBias);   break;
      case QUALITY_KNOB.SHADOWS:       this.state.shadowBias       = clamp(this.state.shadowBias);       break;
      case QUALITY_KNOB.GI:            this.state.giBias           = clamp(this.state.giBias);           break;
      case QUALITY_KNOB.AO:            this.state.aoBias           = clamp(this.state.aoBias);           break;
      case QUALITY_KNOB.POST:          this.state.postBias         = clamp(this.state.postBias);         break;
      case QUALITY_KNOB.DRAW_DISTANCE: this.state.drawDistanceBias = clamp(this.state.drawDistanceBias); break;
      case QUALITY_KNOB.CLUSTER:       this.state.clusterBias      = clamp(this.state.clusterBias);      break;
      default: break;
    }
  }

  _changeLevel(newLevel, knobName, reason) {
    if (newLevel === this.state.level) return;
    const from = this.state.level;
    this.state.level = newLevel;
    this.state.targetLevel = newLevel;
    this.state.lastChangeFrame = this._frame;
    this.state.changeCount++;
    this.state.lastKnob = knobName;
    this.state.lastReason = reason;

    if (newLevel > from) this.state.framesDowngraded++;
    else this.state.framesUpgraded++;

    this._applyLevel(false);

    if (this.options.logDecisions) {
      console.log(`[017_rnd_QualityConfig] level ${QUALITY_LEVEL_NAME[from]} → ${QUALITY_LEVEL_NAME[newLevel]} (${reason}, knob=${knobName})`);
    }

    this._emit('level', {
      from: QUALITY_LEVEL_NAME[from],
      to: QUALITY_LEVEL_NAME[newLevel],
      reason,
      knob: knobName,
    });
  }

  _forceLevel(level, reason) {
    if (level <= this.state.level) {
      this._changeLevel(level, 'forced', reason);
    }
  }

  _applyLevel(initial) {
    const mult = QUALITY_LEVEL_MULTIPLIERS[this.state.level];
    if (!mult) return;

    const base = this._baseBudget;

    // Resolution
    const baseDpr = base.renderer ? base.renderer.dprCap : 1.0;
    const resolvedDpr = Math.max(0.5, baseDpr * mult.resolutionScale * (1 - this.state.resolutionBias * 0.30));

    // Shadows
    const baseShadow = base.shadows ? base.shadows.mapSize : 1024;
    const resolvedShadow = Math.max(256, Math.round(baseShadow * mult.shadowScale * (1 - this.state.shadowBias * 0.40)));
    const cascades = Math.max(1, Math.min(base.shadows ? base.shadows.cascadeCount : 2, 4 - Math.floor(this.state.shadowBias * 3)));

    // GI
    const baseGiHz = base.gi ? base.gi.updateBudgetHz : 15;
    const resolvedGiHz = Math.max(4, Math.round(baseGiHz * mult.giHzScale * (1 - this.state.giBias * 0.40)));
    const resolvedGiSamples = Math.max(2, Math.round((base.gi ? base.gi.sampleCount : 8) * mult.aoScale));

    // AO
    const baseAoScale = base.ao ? base.ao.resolutionScale : 0.5;
    const resolvedAoScale = Math.max(0.25, baseAoScale * mult.aoScale * (1 - this.state.aoBias * 0.30));
    const resolvedAoSamples = Math.max(2, Math.round((base.ao ? base.ao.sampleCount : 8) * mult.aoScale));

    // Post
    const basePost = base.post ? base.post.passBudget : 4;
    const resolvedPost = Math.max(1, basePost + mult.postDelta - Math.floor(this.state.postBias * 2));

    // Draw distance
    const baseDrawDist = base.environment ? base.environment.fogFar : 180;
    const resolvedDrawDist = Math.max(40, baseDrawDist * mult.drawDistance * (1 - this.state.drawDistanceBias * 0.25));

    // Cluster
    const baseCluster = base.lights ? base.lights.maxClusterLights : 64;
    const resolvedCluster = Math.max(8, Math.round(baseCluster * mult.clusterScale * (1 - this.state.clusterBias * 0.30)));

    // Write into snapshot (in-place).
    const s = this.snapshot;
    s.level              = this.state.level;
    s.levelName          = QUALITY_LEVEL_NAME[this.state.level];
    s.resolutionScale    = resolvedDpr;
    s.shadowMapSize      = resolvedShadow;
    s.shadowCascadeCount = cascades;
    s.giUpdateBudgetHz   = resolvedGiHz;
    s.giSampleCount      = resolvedGiSamples;
    s.aoResolutionScale  = resolvedAoScale;
    s.aoSampleCount      = resolvedAoSamples;
    s.postPassBudget     = resolvedPost;
    s.drawDistanceMeters = resolvedDrawDist;
    s.maxClusterLights   = resolvedCluster;
    s.thermalBias        = this.state.thermalBias;
    s.batteryBias        = this.state.batteryBias;
    s.lowPower           = this.state.lowPower;
    s.lastKnob           = this.state.lastKnob;
    s.lastReason         = this.state.lastReason;

    if (!initial) {
      this._emit('applied', {
        level: QUALITY_LEVEL_NAME[this.state.level],
        resolutionScale:    resolvedDpr,
        shadowMapSize:      resolvedShadow,
        giUpdateBudgetHz:   resolvedGiHz,
        aoResolutionScale:  resolvedAoScale,
        postPassBudget:     resolvedPost,
        drawDistanceMeters: resolvedDrawDist,
        maxClusterLights:   resolvedCluster,
      });
    }
  }

  _updateSnapshotFromLevel() {
    // The snapshot is refreshed in _applyLevel() on every level change.
    // This method exists for symmetry; do not add per-frame work here.
    return this.snapshot;
  }

  /* ---------------- accessors ---------------- */

  getLevel()          { return this.state.level; }
  getLevelName()      { return QUALITY_LEVEL_NAME[this.state.level]; }
  getSnapshot()       { return this.snapshot; }
  getTargetFps()      { return this._baseBudget.performance ? this._baseBudget.performance.targetFps : 60; }

  getStats() {
    return {
      level:               QUALITY_LEVEL_NAME[this.state.level],
      levelIndex:          this.state.level,
      targetLevel:         QUALITY_LEVEL_NAME[this.state.targetLevel],
      lastKnob:            QUALITY_KNOB_NAME[this.state.lastKnob],
      lastReason:          this.state.lastReason,
      changes:             this.state.changeCount,
      framesUpgraded:      this.state.framesUpgraded,
      framesDowngraded:    this.state.framesDowngraded,
      framesHeld:          this.state.framesHeld,
      framesObserved:      this.state.framesObserved,
      upgradeAccumMs:      this.state.upgradeAccumMs,
      downgradeAccumMs:    this.state.downgradeAccumMs,
      thermalBias:         this.state.thermalBias,
      batteryBias:         this.state.batteryBias,
      lowPower:            this.state.lowPower === 1,
      perfTier:            PERF_TIER_LOCAL,
      snapshot: {
        resolutionScale:    this.snapshot.resolutionScale,
        shadowMapSize:      this.snapshot.shadowMapSize,
        shadowCascadeCount: this.snapshot.shadowCascadeCount,
        giUpdateBudgetHz:   this.snapshot.giUpdateBudgetHz,
        giSampleCount:      this.snapshot.giSampleCount,
        aoResolutionScale:  this.snapshot.aoResolutionScale,
        aoSampleCount:      this.snapshot.aoSampleCount,
        postPassBudget:     this.snapshot.postPassBudget,
        drawDistanceMeters: this.snapshot.drawDistanceMeters,
        maxClusterLights:   this.snapshot.maxClusterLights,
      },
    };
  }

  reset() {
    this.state.reset();
    this._applyLevel(true);
    return this;
  }

  dispose() {
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultQualityController = null;

export function getDefaultQualityController() {
  if (!_defaultQualityController) _defaultQualityController = new QualityController();
  return _defaultQualityController;
}

export function disposeDefaultQualityController() {
  if (_defaultQualityController) {
    _defaultQualityController.dispose();
    _defaultQualityController = null;
  }
}

/**
 * Fast-path read — returns the last-updated snapshot without touching the
 * controller. Downstream systems call this once per frame.
 */
export function getQualitySnapshot() {
  return getDefaultQualityController().snapshot;
}

/* ------------------------------------------------------------------ */
/* 5. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createQualityController(options = {}) {
  return new QualityController(options);
}

/* ------------------------------------------------------------------ */
/* 6. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  QualityController,
  QualityState,
  QualitySnapshot,
  createQualityController,
  getDefaultQualityController,
  disposeDefaultQualityController,
  getQualitySnapshot,
  QUALITY_LEVEL,
  QUALITY_LEVEL_NAME,
  QUALITY_KNOB,
  QUALITY_KNOB_NAME,
  QUALITY_LEVEL_MULTIPLIERS,
};

export default _defaultExport;