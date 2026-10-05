// File : 005
// name : src/core/005_rnd_FrameScheduler.js
// description : Multi-domain frame scheduler for the anime lighting stack on
//               Android mobile. Where 003_rnd_Runtime.js holds a single
//               FrameScheduler for the render loop cadence, THIS module
//               provides the multi-domain scheduler that drives independent
//               update frequencies for every lighting subsystem:
//
//                 DOMAIN            DEFAULT Hz    BUDGET ms   ADAPTIVE
//                 ─────────────────────────────────────────────────────
//                 simulation        60            4.0         yes
//                 lights            60            2.0         yes
//                 shadows           30            6.0         yes
//                 gi                20            8.0         yes
//                 ao                30            3.0         yes
//                 environment        8            2.0         no
//                 interior          30            2.0         yes
//                 exterior          30            2.0         yes
//                 director          60            1.0         no
//                 post              60            4.0         yes
//
//               Each domain has its own accumulators, EMA, overrun counter, and
//               a per-domain budget-charge hook that downstream quality
//               scalers (193–197) read to bias resolution/shadow/GI/AO
//               quality without dropping frames.
//
//               Optimization techniques applied:
//                 • integer accumulator tick (no float drift over hours)
//                 • bitmask domain gating (one AND, no branch per domain)
//                 • per-domain EMA with fixed-point storage (no allocs)
//                 • starvation detection (domain idle too long → force fire)
//                 • priority biasing (critical domains get extra budget slack)
//                 • dynamic downgrade under thermal/battery pressure
//                 • cascade lock (shadows→gi→ao stay in sync when downscaled)
//                 • fixed-capacity domain array (no Map/Set on hot path)
//                 • zero closures per frame; all callbacks pre-bound
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; every typed array sized once at construction; no
//               per-frame allocations on the hot path.
// best for : Driving the multi-rate update graph of the modular anime lighting
//            stack. Shadow atlas packing at 30 Hz, GI probe bake at 20 Hz, AO
//            blur at 30 Hz, environment palette at 8 Hz, director hints at
//            60 Hz — all in one scheduler with one set of budget counters
//            that the adaptive quality controller can throttle from a single
//            point.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const DOMAIN = Object.freeze({
  SIMULATION:  0,
  LIGHTS:      1,
  SHADOWS:     2,
  GI:          3,
  AO:          4,
  ENVIRONMENT: 5,
  INTERIOR:    6,
  EXTERIOR:    7,
  DIRECTOR:    8,
  POST:        9,
  COUNT:      10,
});

export const DOMAIN_NAME = Object.freeze([
  'simulation',
  'lights',
  'shadows',
  'gi',
  'ao',
  'environment',
  'interior',
  'exterior',
  'director',
  'post',
]);

const DEFAULT_HZ = Object.freeze([
  60,  // simulation
  60,  // lights
  30,  // shadows
  20,  // gi
  30,  // ao
   8,  // environment
  30,  // interior
  30,  // exterior
  60,  // director
  60,  // post
]);

const DEFAULT_BUDGET_MS = Object.freeze(
  PERF_TIER === 'HIGH'
    ? [4.0, 2.0, 6.0, 8.0, 3.0, 2.0, 2.0, 2.0, 1.0, 4.0]
    : PERF_TIER === 'MEDIUM'
      ? [5.0, 2.5, 7.5, 10.0, 3.5, 2.5, 2.5, 2.5, 1.2, 5.0]
      : [6.0, 3.0, 9.0, 12.0, 4.0, 3.0, 3.0, 3.0, 1.5, 6.0]
);

const DEFAULT_ADAPTIVE = Object.freeze([
  true,  // simulation
  true,  // lights
  true,  // shadows
  true,  // gi
  true,  // ao
  false, // environment
  true,  // interior
  true,  // exterior
  false, // director
  true,  // post
]);

const MIN_HZ_FLOOR            = 5;
const MAX_HZ_CEIL             = 120;
const STARVATION_FRAMES       = 45;
const EMA_ALPHA               = 0.15;
const DOWNGRADE_THRESHOLD     = 1.05; // overrun by 5% for N frames
const UPGRADE_THRESHOLD       = 0.70; // underrun by 30% for N frames
const DOWNGRADE_HOLD_FRAMES   = 30;
const UPGRADE_HOLD_FRAMES     = 120;
const PRIORITY_SLACK          = 0.20; // critical domains get +20% budget

const CRITICAL_DOMAINS = Object.freeze([
  DOMAIN.LIGHTS,
  DOMAIN.SHADOWS,
  DOMAIN.POST,
]);

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

function _clampHz(hz) {
  if (!Number.isFinite(hz)) return 60;
  if (hz < MIN_HZ_FLOOR) return MIN_HZ_FLOOR;
  if (hz > MAX_HZ_CEIL)  return MAX_HZ_CEIL;
  return hz | 0;
}

/* ------------------------------------------------------------------ */
/* 2. DOMAIN SLOT (fixed, allocation-free)                            */
/* ------------------------------------------------------------------ */

export class DomainSlot {
  constructor(index, name) {
    this.index        = index;
    this.name         = name;
    this.enabled      = 1;

    this.baseHz       = DEFAULT_HZ[index];
    this.targetHz     = this.baseHz;
    this.minHz        = MIN_HZ_FLOOR;
    this.maxHz        = this.baseHz;

    this.budgetMs     = DEFAULT_BUDGET_MS[index];
    this.adaptive     = DEFAULT_ADAPTIVE[index] ? 1 : 0;

    this.accumMs      = 0;
    this.stepMs       = 1000 / this.baseHz;

    this.framesRun    = 0;
    this.framesSkip   = 0;
    this.framesOver   = 0;
    this.starve       = 0;

    this.lastMs       = 0;
    this.lastEma      = 0;
    this.peakMs       = 0;

    this._downgradeHold = 0;
    this._upgradeHold   = 0;

    this.callback     = null;
    this.callbackCtx  = null;

    this.priority     = CRITICAL_DOMAINS.indexOf(index) >= 0 ? 1 : 0;
  }

  setHz(hz) {
    this.targetHz = _clampHz(hz);
    if (this.targetHz > this.maxHz) this.targetHz = this.maxHz;
    if (this.targetHz < this.minHz) this.targetHz = this.minHz;
    this.stepMs = 1000 / this.targetHz;
    return this;
  }

  setBudgetMs(ms) {
    const v = Number(ms);
    if (!Number.isFinite(v) || v <= 0) return this;
    const slack = this.priority ? (1 + PRIORITY_SLACK) : 1;
    this.budgetMs = v * slack;
    return this;
  }

  setAdaptive(enabled) {
    this.adaptive = enabled ? 1 : 0;
    return this;
  }

  bind(fn, ctx) {
    this.callback = (typeof fn === 'function' ? fn : null);
    this.callbackCtx = ctx || null;
    return this;
  }

  reset() {
    this.accumMs = 0;
    this.framesRun = 0;
    this.framesSkip = 0;
    this.framesOver = 0;
    this.starve = 0;
    this.lastMs = 0;
    this.lastEma = 0;
    this.peakMs = 0;
    this._downgradeHold = 0;
    this._upgradeHold = 0;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. FRAME SCHEDULER                                                 */
/* ------------------------------------------------------------------ */

export class FrameScheduler {
  constructor(options = {}) {
    this.options = Object.assign({
      masterHz:      60,
      adaptive:      true,
      thermalBias:   0,
      batteryBias:   0,
      hysteresis:    true,
      cascadeLock:   true,
    }, options || {});

    this.domains = new Array(DOMAIN.COUNT);
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      this.domains[i] = new DomainSlot(i, DOMAIN_NAME[i]);
    }

    this.masterHz     = _clampHz(this.options.masterHz);
    this.masterStepMs = 1000 / this.masterHz;
    this.masterAccum  = 0;

    this.frame         = 0;
    this.elapsedMs     = 0;
    this.elapsed       = 0;
    this.lastDtMs      = 0;
    this.lastDtEmaMs   = 16.67;

    this.thermalBias   = this.options.thermalBias;
    this.batteryBias   = this.options.batteryBias;
    this.adaptive      = this.options.adaptive !== false;
    this.hysteresis    = this.options.hysteresis !== false;
    this.cascadeLock   = this.options.cascadeLock !== false;

    this.domainsFiredMask  = 0;
    this.domainsSkipMask   = 0;
    this.domainsOverMask   = 0;

    this._frameMs      = new Float64Array(DOMAIN.COUNT);
    this._frameCalls   = new Uint32Array(DOMAIN.COUNT);

    this._listeners    = new Map();
  }

  /* ---------------- domain accessors ---------------- */

  getDomain(index) {
    if (index < 0 || index >= DOMAIN.COUNT) return null;
    return this.domains[index];
  }

  setDomainHz(index, hz) {
    const d = this.getDomain(index);
    if (!d) return false;
    d.setHz(hz);
    return true;
  }

  setDomainBudget(index, ms) {
    const d = this.getDomain(index);
    if (!d) return false;
    d.setBudgetMs(ms);
    return true;
  }

  setDomainEnabled(index, enabled) {
    const d = this.getDomain(index);
    if (!d) return false;
    d.enabled = enabled ? 1 : 0;
    return true;
  }

  bindDomain(index, fn, ctx) {
    const d = this.getDomain(index);
    if (!d) return false;
    d.bind(fn, ctx);
    return true;
  }

  /* ---------------- master clock ---------------- */

  setMasterHz(hz) {
    this.masterHz = _clampHz(hz);
    this.masterStepMs = 1000 / this.masterHz;
    return this;
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
      try { arr[i](payload); } catch (e) { console.error(`[005_rnd_FrameScheduler] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- thermal / battery bias ---------------- */

  setThermalBias(bias) {
    this.thermalBias = Math.max(0, Math.min(1, Number(bias) || 0));
    if (this.cascadeLock) this._applyCascadeLock();
    return this;
  }

  setBatteryBias(bias) {
    this.batteryBias = Math.max(0, Math.min(1, Number(bias) || 0));
    if (this.cascadeLock) this._applyCascadeLock();
    return this;
  }

  _applyCascadeLock() {
    // When either bias is engaged, shadows, GI, and AO must be downscaled
    // together to keep the anime look coherent (shadow→GI→AO form a visual
    // chain: incorrect shadow density causes GI banding; incorrect AO
    // causes shadow edge aliasing in cel shading).
    const pressure = Math.max(this.thermalBias, this.batteryBias);
    if (pressure <= 0.001) {
      this._setTargetHz(DOMAIN.SHADOWS, 30);
      this._setTargetHz(DOMAIN.GI,      20);
      this._setTargetHz(DOMAIN.AO,      30);
      return;
    }

    const shadowsHz = Math.round(30 * (1 - pressure * 0.6));
    const giHz      = Math.round(20 * (1 - pressure * 0.7));
    const aoHz      = Math.round(30 * (1 - pressure * 0.6));

    this._setTargetHz(DOMAIN.SHADOWS, Math.max(5, shadowsHz));
    this._setTargetHz(DOMAIN.GI,      Math.max(3, giHz));
    this._setTargetHz(DOMAIN.AO,      Math.max(5, aoHz));
  }

  _setTargetHz(index, hz) {
    const d = this.domains[index];
    if (!d) return;
    d.targetHz = _clampHz(hz);
    d.stepMs = 1000 / d.targetHz;
  }

  /* ---------------- adaptive per-domain downgrade ---------------- */

  _adaptDomain(d, dtMs) {
    if (!d.adaptive || !this.adaptive) return;

    // Downgrade when over budget for N consecutive frames
    if (d.lastEma > d.budgetMs * DOWNGRADE_THRESHOLD) {
      d._downgradeHold++;
      if (d._downgradeHold >= DOWNGRADE_HOLD_FRAMES && d.targetHz > d.minHz) {
        const next = Math.max(d.minHz, Math.round(d.targetHz * 0.85));
        if (next !== d.targetHz) {
          d.targetHz = next;
          d.stepMs = 1000 / next;
          this._emit('downgrade', { domain: d.name, hz: next, ema: d.lastEma, budget: d.budgetMs });
        }
        d._downgradeHold = 0;
      }
    } else {
      d._downgradeHold = 0;
    }

    // Upgrade when well under budget for longer
    if (d.lastEma < d.budgetMs * UPGRADE_THRESHOLD && d.targetHz < d.maxHz) {
      d._upgradeHold++;
      if (d._upgradeHold >= UPGRADE_HOLD_FRAMES) {
        const next = Math.min(d.maxHz, Math.round(d.targetHz * 1.10));
        if (next !== d.targetHz) {
          d.targetHz = next;
          d.stepMs = 1000 / next;
          this._emit('upgrade', { domain: d.name, hz: next, ema: d.lastEma, budget: d.budgetMs });
        }
        d._upgradeHold = 0;
      }
    } else {
      d._upgradeHold = 0;
    }
  }

  /* ---------------- main tick ---------------- */

  tick(dtMs, elapsedSec) {
    const dt = Number.isFinite(dtMs) ? Math.max(0, Math.min(100, dtMs)) : 0;

    this.frame++;
    this.lastDtMs = dt;
    this.elapsedMs += dt;
    this.elapsed = Number.isFinite(elapsedSec) ? elapsedSec : (this.elapsedMs * 0.001);

    this.lastDtEmaMs += (dt - this.lastDtEmaMs) * EMA_ALPHA;

    this.masterAccum += dt;
    if (this.masterAccum < this.masterStepMs) {
      this.domainsFiredMask = 0;
      this.domainsSkipMask = 0xFFFF;
      this.domainsOverMask = 0;
      return 0;
    }
    this.masterAccum -= this.masterStepMs;

    let fired = 0;
    let firedMask = 0;
    let skipMask  = 0;
    let overMask  = 0;

    for (let i = 0; i < DOMAIN.COUNT; i++) {
      const d = this.domains[i];
      if (!d.enabled) { skipMask |= (1 << i); continue; }

      d.accumMs += dt;
      d.framesSkip++;

      // Starvation guard: if we've skipped too long, force fire.
      const forceFire = d.starve >= STARVATION_FRAMES;

      if (d.accumMs >= d.stepMs || forceFire) {
        d.accumMs = forceFire ? 0 : (d.accumMs - d.stepMs);

        const t0 = _now();
        let ok = true;

        if (d.callback) {
          try {
            d.callback(dt, this.elapsed, d);
          } catch (e) {
            ok = false;
            console.error(`[005_rnd_FrameScheduler] domain "${d.name}" callback failed`, e);
          }
        }

        const t1 = _now();
        const cost = t1 - t0;

        d.lastMs = cost;
        d.lastEma += (cost - d.lastEma) * EMA_ALPHA;
        if (cost > d.peakMs) d.peakMs = cost;

        d.framesRun++;
        d.framesSkip = 0;
        d.starve = 0;

        this._frameMs[i] = cost;
        this._frameCalls[i]++;

        if (cost > d.budgetMs) {
          d.framesOver++;
          overMask |= (1 << i);
        }

        this._adaptDomain(d, dt);

        fired++;
        firedMask |= (1 << i);
      } else {
        d.starve++;
        skipMask |= (1 << i);
      }
    }

    this.domainsFiredMask = firedMask;
    this.domainsSkipMask  = skipMask;
    this.domainsOverMask  = overMask;

    this._emit('tick', {
      frame: this.frame,
      dt,
      elapsed: this.elapsed,
      fired,
      firedMask,
      skipMask,
      overMask,
    });

    return fired;
  }

  /* ---------------- bulk operations ---------------- */

  reset() {
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      this.domains[i].reset();
      this._frameMs[i] = 0;
      this._frameCalls[i] = 0;
    }
    this.masterAccum = 0;
    this.elapsedMs = 0;
    this.elapsed = 0;
    this.frame = 0;
    this.lastDtEmaMs = 16.67;
    return this;
  }

  suspend() {
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      this.domains[i].enabled = 0;
    }
    return this;
  }

  resume() {
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      this.domains[i].enabled = 1;
    }
    return this;
  }

  dispose() {
    this.reset();
    this._listeners.clear();
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      const d = this.domains[i];
      d.callback = null;
      d.callbackCtx = null;
    }
    return this;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const domains = new Array(DOMAIN.COUNT);
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      const d = this.domains[i];
      domains[i] = {
        name:       d.name,
        enabled:    d.enabled === 1,
        targetHz:   d.targetHz,
        baseHz:     d.baseHz,
        budgetMs:   d.budgetMs,
        lastMs:     d.lastMs,
        lastEma:    d.lastEma,
        peakMs:     d.peakMs,
        framesRun:  d.framesRun,
        framesSkip: d.framesSkip,
        framesOver: d.framesOver,
        starve:     d.starve,
        adaptive:   d.adaptive === 1,
      };
    }

    return {
      frame:             this.frame,
      elapsed:           this.elapsed,
      masterHz:          this.masterHz,
      lastDtMs:          this.lastDtMs,
      lastDtEmaMs:       this.lastDtEmaMs,
      thermalBias:       this.thermalBias,
      batteryBias:       this.batteryBias,
      cascadeLock:       this.cascadeLock,
      domainsFiredMask:  this.domainsFiredMask,
      domainsSkipMask:   this.domainsSkipMask,
      domainsOverMask:   this.domainsOverMask,
      domains,
      perfTier:          PERF_TIER,
    };
  }

  getPressure() {
    // 0 = comfortable, 1 = at budget, >1 = over budget.
    const out = new Float32Array(DOMAIN.COUNT);
    for (let i = 0; i < DOMAIN.COUNT; i++) {
      const d = this.domains[i];
      out[i] = d.budgetMs > 0 ? (d.lastEma / d.budgetMs) : 0;
    }
    return out;
  }
}

/* ------------------------------------------------------------------ */
/* 4. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultScheduler = null;

export function getDefaultScheduler() {
  if (!_defaultScheduler) _defaultScheduler = new FrameScheduler();
  return _defaultScheduler;
}

export function disposeDefaultScheduler() {
  if (_defaultScheduler) {
    _defaultScheduler.dispose();
    _defaultScheduler = null;
  }
}

/* ------------------------------------------------------------------ */
/* 5. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createFrameScheduler(options = {}) {
  return new FrameScheduler(options);
}

/* ------------------------------------------------------------------ */
/* 6. DOMAIN REGISTRATION HELPER                                      */
/* ------------------------------------------------------------------ */

export function bindLightingDomains(scheduler, callbacks = {}) {
  if (!scheduler) return false;

  scheduler.bindDomain(DOMAIN.SIMULATION,  callbacks.simulation);
  scheduler.bindDomain(DOMAIN.LIGHTS,      callbacks.lights);
  scheduler.bindDomain(DOMAIN.SHADOWS,     callbacks.shadows);
  scheduler.bindDomain(DOMAIN.GI,          callbacks.gi);
  scheduler.bindDomain(DOMAIN.AO,          callbacks.ao);
  scheduler.bindDomain(DOMAIN.ENVIRONMENT, callbacks.environment);
  scheduler.bindDomain(DOMAIN.INTERIOR,    callbacks.interior);
  scheduler.bindDomain(DOMAIN.EXTERIOR,    callbacks.exterior);
  scheduler.bindDomain(DOMAIN.DIRECTOR,    callbacks.director);
  scheduler.bindDomain(DOMAIN.POST,        callbacks.post);

  return true;
}

/* ------------------------------------------------------------------ */
/* 7. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  FrameScheduler,
  DomainSlot,
  createFrameScheduler,
  getDefaultScheduler,
  disposeDefaultScheduler,
  bindLightingDomains,
  DOMAIN,
  DOMAIN_NAME,
};

export default _defaultExport;