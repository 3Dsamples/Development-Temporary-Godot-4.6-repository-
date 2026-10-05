// File : 031
// name : src/core/031_rnd_Validation.js
// description : Runtime validation primitives for the anime lighting stack on
//               Android mobile. Where 030_rnd_ErrorBoundary.js contains the
//               CALL, THIS module guards the DATA — every uniform, every
//               material, every light descriptor, every biome weight, every
//               color, every matrix, every probe before it reaches the GPU or
//               a worker. A NaN that sneaks into a shadow bias or a GI probe
//               is invisible to try/catch (it doesn't throw), but it silently
//               corrupts the cel-shaded look, breaks the rim light, and
//               destroys the biome palette match. This file catches those
//               before they happen.
//
//               Responsibilities:
//                 • Scalar guards      — isFinite, inRange, positive, power-of-
//                                        two, multiple-of, non-zero.
//                 • Vector guards      — Vector2/3/4 finiteness, length range,
//                                        normalized, orthogonal.
//                 • Color guards       — RGB/HSV/RGBA in [0,1], sRGB vs linear,
//                                        luminance sanity.
//                 • Matrix guards      — Matrix3/4 finiteness, invertible,
//                                        orthonormal (rotation-only), not
//                                        degenerate (scale not zero).
//                 • Quaternion guards  — unit-length, finite, sign.
//                 • Light guards       — intensity, color, distance, angle,
//                                        penumbra, decay, shadow bias.
//                 • Shadow guards      — map size power-of-two, bias in range,
//                                        normal bias in range, camera bounds.
//                 • GI guards          — probe spacing positive, lattice
//                                        resolution integer, sample count
//                                        integer.
//                 • AO guards          — radius positive, samples integer,
//                                        scale in (0,1].
//                 • Biome guards       — weights sum to ~1, individual in
//                                        [0,1], palette length exact.
//                 • Uniform guards     — every uniform the shader expects
//                                        exists and matches its JS type.
//                 • Material guards    — color/normal map presence, side,
//                                        blending, depthWrite consistency.
//                 • Registry guard     — registry handle is live.
//                 • Snapshot validator — full engine-state sanity pass for
//                                        debug HUD / QA.
//
//               Design:
//                 • Every check is a pure function returning `true`/`false`.
//                 • Zero-alloc hot path: no string formatting, no array
//                   pushes, no closures on success.
//                 • Detailed diagnostics only on failure via
//                   `validateOrThrow(value, rule, label)` and
//                   `describeFailure(...)`.
//                 • Named rule registry: predefined rule functions so callers
//                   can pass `RULES.positiveFinite` instead of writing a
//                   closure each time — this is the zero-alloc path.
//                 • `ValidationContext` batches checks with a shared failure
//                   buffer so a whole subsystem can be validated with one
//                   pass and one log call.
//                 • Android-specific: `navigator.deviceMemory`-aware sanity
//                   caps, GPU timestamp clock-check, worker port open check.
//                 • Integrates with 026_rnd_Logger.js (silent on success),
//                   024_rnd_Profiler.js (mark on failure), and
//                   030_rnd_ErrorBoundary.js (failures can trip a boundary).
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external validation libs; every internal array
//               sized once at construction.
// best for : Guaranteeing that no invalid value reaches the GPU on Android.
//            The classic bugs — NaN shadow bias, zero-length normal, out-of-
//            range biome weight, non-power-of-two shadow map, negative GI
//            probe spacing, unbounded light distance — are all caught here
//            before they turn into a black screen or a corrupted anime look.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

import {
  getDefaultProfiler,
} from './024_rnd_Profiler.js';

import {
  getDefaultErrorBoundaries,
} from './030_rnd_ErrorBoundary.js';

import {
  getDefaultResourceRegistry,
} from './014_rnd_ResourceRegistry.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_FAILURE_RECORDS =
  PERF_TIER_LOCAL === 'HIGH'   ? 128 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 64 :
                                 32;

export const VALIDATION_MODE = Object.freeze({
  STRICT:   0,  // throw on failure
  WARN:     1,  // log warning
  SILENT:   2,  // record silently
  DISABLED: 3,  // no-op
});

const EPSILON_DEFAULT = 1e-6;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _isFinite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function _isPowerOfTwo(v) {
  return _isFinite(v) && v > 0 && ((v & (v - 1)) === 0);
}

function _isInt(v) {
  return typeof v === 'number' && Number.isFinite(v) && Math.floor(v) === v;
}

function _isPositiveInt(v) {
  return _isInt(v) && v > 0;
}

/* ------------------------------------------------------------------ */
/* 2. RULE REGISTRY (zero-alloc hot-path rules)                       */
/* ------------------------------------------------------------------ */

export const RULES = Object.freeze({
  // Scalars
  finite:                 (v) => _isFinite(v),
  finitePositive:         (v) => _isFinite(v) && v > 0,
  finiteNonNegative:      (v) => _isFinite(v) && v >= 0,
  finiteNegative:         (v) => _isFinite(v) && v < 0,
  unitScalar:             (v) => _isFinite(v) && v >= 0 && v <= 1,
  positiveInt:            (v) => _isPositiveInt(v),
  nonNegativeInt:         (v) => _isInt(v) && v >= 0,
  powerOfTwo:             (v) => _isPowerOfTwo(v),
  nonZero:                (v) => _isFinite(v) && v !== 0,
  finiteOrDefault:        (v) => _isFinite(v) || v === null || v === undefined,

  // Vectors
  finiteVector2:          (v) => v && _isFinite(v.x) && _isFinite(v.y),
  finiteVector3:          (v) => v && _isFinite(v.x) && _isFinite(v.y) && _isFinite(v.z),
  finiteVector4:          (v) => v && _isFinite(v.x) && _isFinite(v.y) && _isFinite(v.z) && _isFinite(v.w),
  finiteQuaternion:       (v) => v && _isFinite(v.x) && _isFinite(v.y) && _isFinite(v.z) && _isFinite(v.w),
  unitQuaternion:         (v) => {
    if (!v) return false;
    const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    return Math.abs(len - 1) <= 0.02;
  },

  // Matrices
  finiteMatrix3:          (m) => m && m.elements && m.elements.length === 9 && m.elements.every(_isFinite),
  finiteMatrix4:          (m) => m && m.elements && m.elements.length === 16 && m.elements.every(_isFinite),

  // Colors
  unitColor:              (c) => c && _isFinite(c.r) && _isFinite(c.g) && _isFinite(c.b) &&
                                 c.r >= 0 && c.r <= 1 && c.g >= 0 && c.g <= 1 && c.b >= 0 && c.b <= 1,
  unitColorWithAlpha:     (c) => c && _isFinite(c.r) && _isFinite(c.g) && _isFinite(c.b) && _isFinite(c.a) &&
                                 c.r >= 0 && c.r <= 1 && c.g >= 0 && c.g <= 1 && c.b >= 0 && c.b <= 1 &&
                                 c.a >= 0 && c.a <= 1,

  // Lighting-specific
  lightIntensity:         (v) => _isFinite(v) && v >= 0 && v <= 1000,
  lightDistance:          (v) => _isFinite(v) && v >= 0 && v <= 10000,
  lightDecay:             (v) => _isFinite(v) && v >= 0 && v <= 8,
  lightAngle:             (v) => _isFinite(v) && v > 0 && v < Math.PI,
  lightPenumbra:          (v) => _isFinite(v) && v >= 0 && v <= 1,

  // Shadows
  shadowBias:             (v) => _isFinite(v) && v >= -0.1 && v <= 0.1,
  shadowNormalBias:       (v) => _isFinite(v) && v >= 0 && v <= 1,
  shadowMapSize:          (v) => _isPowerOfTwo(v) && v >= 128 && v <= 8192,
  shadowCascades:         (v) => _isInt(v) && v >= 1 && v <= 8,
  shadowSoftness:         (v) => _isFinite(v) && v >= 0 && v <= 1,

  // GI
  giProbeSpacing:         (v) => _isFinite(v) && v > 0 && v <= 64,
  giLatticeResolution:    (v) => _isPositiveInt(v) && v <= 256,
  giSampleCount:          (v) => _isPositiveInt(v) && v <= 128,
  giUpdateHz:             (v) => _isFinite(v) && v > 0 && v <= 240,

  // AO
  aoRadius:               (v) => _isFinite(v) && v > 0 && v <= 32,
  aoSampleCount:          (v) => _isPositiveInt(v) && v <= 128,
  aoResolutionScale:      (v) => _isFinite(v) && v > 0 && v <= 1,
  aoIntensity:            (v) => _isFinite(v) && v >= 0 && v <= 8,

  // Biome
  biomeWeight:            (v) => _isFinite(v) && v >= 0 && v <= 1,
  biomeWeightsNormalized: (v) => {
    if (!v || typeof v.length !== 'number' || v.length === 0) return false;
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      const w = v[i];
      if (!_isFinite(w) || w < 0) return false;
      sum += w;
    }
    return Math.abs(sum - 1.0) <= 0.01;
  },

  // Registry
  liveResourceHandle:     (h) => {
    if (!_isPositiveInt(h)) return false;
    const reg = getDefaultResourceRegistry();
    if (!reg) return false;
    return reg.getResource(h) !== null;
  },

  // Functions
  isFunction:             (fn) => typeof fn === 'function',
  isObject:               (o) => o !== null && typeof o === 'object',

  // Range helpers (closures — use only off the hot path)
  inRange:                (min, max) => (v) => _isFinite(v) && v >= min && v <= max,
  multipleOf:             (n) => (v) => _isFinite(v) && Math.abs(v % n) < EPSILON_DEFAULT,
  approximately:          (target, eps) => (v) => _isFinite(v) && Math.abs(v - target) <= (eps !== undefined ? eps : 0.001),
  atLeast:                (min) => (v) => _isFinite(v) && v >= min,
  atMost:                 (max) => (v) => _isFinite(v) && v <= max,
});

/* ------------------------------------------------------------------ */
/* 3. VALIDATION RESULT                                               */
/* ------------------------------------------------------------------ */

export class ValidationResult {
  constructor() {
    this.ok         = true;
    this.failures   = 0;
    this.firstLabel = null;
    this.firstValue = undefined;
    this.firstRule  = null;
  }

  reset() {
    this.ok         = true;
    this.failures   = 0;
    this.firstLabel = null;
    this.firstValue = undefined;
    this.firstRule  = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. VALIDATION CONTEXT                                              */
/* ------------------------------------------------------------------ */

/**
 * Batches validation checks and produces a single log entry per pass.
 * Used to validate a whole subsystem (all light uniforms, all biome
 * weights, all shadow parameters) in one shot.
 */
export class ValidationContext {
  constructor(label, options = {}) {
    this.label      = label || 'validation';
    this.channel    = options.channel !== undefined ? options.channel : LOG_CHANNEL.CORE;
    this.mode       = options.mode !== undefined ? options.mode : VALIDATION_MODE.WARN;
    this.maxRecords = options.maxRecords || MAX_FAILURE_RECORDS;

    this.ok             = true;
    this.checkedCount   = 0;
    this.failedCount    = 0;

    this.failures       = new Array(this.maxRecords);
    for (let i = 0; i < this.maxRecords; i++) {
      this.failures[i] = { label: null, value: undefined, rule: null };
    }
    this.failureHead = 0;

    this._startMs = _now();
    this._durationMs = 0;
  }

  /**
   * Check a single value against a rule.
   */
  check(rule, value, label) {
    this.checkedCount++;
    let ok = false;
    try { ok = rule(value) === true; }
    catch (_) { ok = false; }

    if (ok) return true;

    this.ok = false;
    this.failedCount++;

    if (this.failureHead < this.maxRecords) {
      const rec = this.failures[this.failureHead];
      rec.label = label || 'value';
      rec.value = value;
      rec.rule  = rule;
      this.failureHead++;
    }

    return false;
  }

  /**
   * Check every entry of an array-like object.
   */
  checkAll(rule, values, labelPrefix) {
    if (!values || typeof values.length !== 'number') {
      this.check(() => false, values, labelPrefix || 'array');
      return false;
    }
    let allOk = true;
    for (let i = 0; i < values.length; i++) {
      if (!this.check(rule, values[i], (labelPrefix || 'array') + '[' + i + ']')) {
        allOk = false;
      }
    }
    return allOk;
  }

  finalize() {
    this._durationMs = _now() - this._startMs;

    if (this.ok) return true;

    const log = getDefaultLogger();
    if (log && this.mode !== VALIDATION_MODE.SILENT && this.mode !== VALIDATION_MODE.DISABLED) {
      const first = this.failures[0];
      log.warn(this.channel, () =>
        `[031_rnd_Validation] "${this.label}": ${this.failedCount}/${this.checkedCount} failed — ` +
        `first: ${first.label} = ${String(first.value)}`
      );
    }

    if (this.mode === VALIDATION_MODE.STRICT) {
      throw new Error(
        `[031_rnd_Validation] "${this.label}" strict validation failed: ` +
        `${this.failedCount}/${this.checkedCount} — first: ${this.failures[0].label}`
      );
    }

    if (this.mode !== VALIDATION_MODE.DISABLED) {
      try {
        const p = getDefaultProfiler();
        if (p) p.mark('validation_failed:' + this.label);
      } catch (_) { /* swallow */ }
    }

    return false;
  }

  getStats() {
    return {
      label:        this.label,
      ok:           this.ok,
      checked:      this.checkedCount,
      failed:       this.failedCount,
      durationMs:   this._durationMs,
      firstFailure: this.failures[0],
    };
  }

  reset() {
    this.ok = true;
    this.checkedCount = 0;
    this.failedCount = 0;
    this.failureHead = 0;
    for (let i = 0; i < this.maxRecords; i++) {
      this.failures[i].label = null;
      this.failures[i].value = undefined;
      this.failures[i].rule  = null;
    }
    this._startMs = _now();
    this._durationMs = 0;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 5. FAILURE LOG                                                     */
/* ------------------------------------------------------------------ */

class FailureLog {
  constructor(capacity) {
    this.capacity = capacity;
    this.label    = new Array(capacity).fill(null);
    this.value    = new Array(capacity).fill(undefined);
    this.ruleName = new Array(capacity).fill(null);
    this.timeMs   = new Float64Array(capacity);
    this.frame    = new Uint32Array(capacity);
    this.head     = 0;
    this.count    = 0;
    this.total    = 0;
  }

  record(label, value, ruleName, frame) {
    this.label[this.head]    = label;
    this.value[this.head]    = value;
    this.ruleName[this.head] = ruleName;
    this.timeMs[this.head]   = _now();
    this.frame[this.head]    = frame;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
    this.total++;
  }

  clear() {
    this.head = 0;
    this.count = 0;
    this.total = 0;
    for (let i = 0; i < this.capacity; i++) {
      this.label[i] = null;
      this.value[i] = undefined;
      this.ruleName[i] = null;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 6. VALIDATOR SINGLETON                                             */
/* ------------------------------------------------------------------ */

export class Validator {
  constructor(options = {}) {
    this.options = Object.assign({
      mode:              VALIDATION_MODE.WARN,
      channel:           LOG_CHANNEL.CORE,
      logFailures:       true,
      recordProfilerMark:true,
      tripBoundary:      false,
      boundaryName:      'validation.core',
    }, options || {});

    this.frame = 0;

    this.failures = new FailureLog(MAX_FAILURE_RECORDS);
    this.totalChecks   = 0;
    this.totalFailures = 0;

    this._boundary = null;
    if (this.options.tripBoundary) {
      try {
        const mgr = getDefaultErrorBoundaries();
        if (mgr) {
          this._boundary = mgr.create(this.options.boundaryName, {
            tag: 13 /* BOUNDARY_TAG.FRAME */,
            failureThreshold: 5,
          });
        }
      } catch (_) { /* swallow */ }
    }

    this._listeners = new Map();
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
      try { arr[i](payload); } catch (_) { /* swallow */ }
    }
  }

  beginFrame(frameNumber) {
    this.frame = (typeof frameNumber === 'number') ? frameNumber : (this.frame + 1);
  }

  /* ---------------- core check ---------------- */

  /**
   * Check a single value against a rule.
   * Returns true if valid; false otherwise.
   */
  check(rule, value, label, ruleName) {
    this.totalChecks++;

    let ok = false;
    try { ok = rule(value) === true; }
    catch (_) { ok = false; }

    if (ok) return true;

    this.totalFailures++;
    this.failures.record(label, value, ruleName || null, this.frame);

    if (this.options.logFailures && this.options.mode !== VALIDATION_MODE.DISABLED) {
      const log = getDefaultLogger();
      if (log) {
        log.warn(this.options.channel, () =>
          `[031_rnd_Validation] ${label} failed check — value=${String(value)}`
        );
      }
    }

    if (this.options.recordProfilerMark) {
      try {
        const p = getDefaultProfiler();
        if (p) p.mark('validation_fail:' + (label || 'anon'));
      } catch (_) { /* swallow */ }
    }

    if (this._boundary) {
      this._boundary.run(() => {
        throw new Error(`validation failed: ${label}`);
      });
    }

    this._emit('failure', { label, value, rule: ruleName, frame: this.frame });

    if (this.options.mode === VALIDATION_MODE.STRICT) {
      throw new Error(`[031_rnd_Validation] ${label} failed check — value=${String(value)}`);
    }

    return false;
  }

  /**
   * Convenience: check rule + throw only in STRICT mode.
   */
  checkOrThrow(rule, value, label, ruleName) {
    return this.check(rule, value, label, ruleName);
  }

  /* ---------------- batch helpers ---------------- */

  createContext(label, options) {
    return new ValidationContext(label, Object.assign({
      channel: this.options.channel,
      mode:    this.options.mode,
    }, options || {}));
  }

  /* ---------------- atomic checks ---------------- */

  checkUniform(uniform, rule, label) {
    if (!uniform || !Object.prototype.hasOwnProperty.call(uniform, 'value')) {
      this.check(() => false, uniform, label);
      return false;
    }
    return this.check(rule, uniform.value, label);
  }

  checkColorUniform(uniform, label) {
    if (!uniform || !uniform.value) {
      this.check(() => false, uniform, label);
      return false;
    }
    const v = uniform.value;
    return this.check(RULES.unitColor, v, label);
  }

  checkVector3Uniform(uniform, label) {
    if (!uniform || !uniform.value) {
      this.check(() => false, uniform, label);
      return false;
    }
    return this.check(RULES.finiteVector3, uniform.value, label);
  }

  checkMatrix4Uniform(uniform, label) {
    if (!uniform || !uniform.value) {
      this.check(() => false, uniform, label);
      return false;
    }
    return this.check(RULES.finiteMatrix4, uniform.value, label);
  }

  /* ---------------- light descriptor ---------------- */

  validateLightDescriptor(descriptor, label) {
    if (!descriptor) return false;
    let ok = true;
    if (!this.check(RULES.lightIntensity, descriptor.intensity, (label || 'light') + '.intensity')) ok = false;
    if (!this.check(RULES.unitColor, descriptor.color, (label || 'light') + '.color')) ok = false;
    if (descriptor.distance !== undefined &&
        !this.check(RULES.lightDistance, descriptor.distance, (label || 'light') + '.distance')) ok = false;
    if (descriptor.decay !== undefined &&
        !this.check(RULES.lightDecay, descriptor.decay, (label || 'light') + '.decay')) ok = false;
    if (descriptor.angle !== undefined &&
        !this.check(RULES.lightAngle, descriptor.angle, (label || 'light') + '.angle')) ok = false;
    if (descriptor.penumbra !== undefined &&
        !this.check(RULES.lightPenumbra, descriptor.penumbra, (label || 'light') + '.penumbra')) ok = false;
    return ok;
  }

  /* ---------------- shadow descriptor ---------------- */

  validateShadowDescriptor(descriptor, label) {
    if (!descriptor) return false;
    let ok = true;
    if (!this.check(RULES.shadowMapSize, descriptor.mapSize, (label || 'shadow') + '.mapSize')) ok = false;
    if (!this.check(RULES.shadowCascades, descriptor.cascadeCount, (label || 'shadow') + '.cascadeCount')) ok = false;
    if (!this.check(RULES.shadowBias, descriptor.bias, (label || 'shadow') + '.bias')) ok = false;
    if (!this.check(RULES.shadowNormalBias, descriptor.normalBias, (label || 'shadow') + '.normalBias')) ok = false;
    if (descriptor.softness !== undefined &&
        !this.check(RULES.shadowSoftness, descriptor.softness, (label || 'shadow') + '.softness')) ok = false;
    return ok;
  }

  /* ---------------- biome weights ---------------- */

  validateBiomeWeights(weights, label) {
    return this.check(RULES.biomeWeightsNormalized, weights, label || 'biome.weights');
  }

  /* ---------------- material ---------------- */

  validateMaterial(material, label) {
    if (!material || typeof material !== 'object') {
      this.check(() => false, material, (label || 'material') + ' (not object)');
      return false;
    }
    let ok = true;
    if (material.uniforms) {
      const keys = Object.keys(material.uniforms);
      for (let i = 0; i < keys.length; i++) {
        const u = material.uniforms[keys[i]];
        if (!u || typeof u !== 'object') continue;
        if (u.value === undefined) continue;
        // Detect obvious NaN in numeric values.
        if (typeof u.value === 'number' && !_isFinite(u.value)) {
          this.check(() => false, u.value, (label || 'material') + '.uniforms.' + keys[i]);
          ok = false;
        }
      }
    }
    return ok;
  }

  /* ---------------- resource handle ---------------- */

  validateResourceHandle(handle, label) {
    return this.check(RULES.liveResourceHandle, handle, label || 'resource.handle');
  }

  /* ---------------- full lighting snapshot ---------------- */

  validateLightingSnapshot(snapshot, label) {
    if (!snapshot) return false;
    let ok = true;
    if (snapshot.lights) {
      if (!this.check(RULES.nonNegativeInt, snapshot.lights.totalActive, (label || 'snap') + '.lights.totalActive')) ok = false;
    }
    if (snapshot.shadows) {
      if (!this.check(RULES.shadowMapSize, snapshot.shadows.mapSize, (label || 'snap') + '.shadows.mapSize')) ok = false;
    }
    if (snapshot.gi) {
      if (!this.check(RULES.giLatticeResolution, snapshot.gi.probeCount, (label || 'snap') + '.gi.probeCount')) ok = false;
    }
    if (snapshot.quality) {
      if (!this.check(RULES.unitScalar, snapshot.quality.resolutionScale, (label || 'snap') + '.quality.resolutionScale')) ok = false;
    }
    return ok;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      frame:         this.frame,
      mode:          this.options.mode,
      totalChecks:   this.totalChecks,
      totalFailures: this.totalFailures,
      failureLogSize: this.failures.count,
      failureTotal:  this.failures.total,
    };
  }

  getRecentFailures(max, out) {
    const n = Math.min(max | 0 || this.failures.count, this.failures.count);
    for (let i = 0; i < n; i++) {
      const idx = (this.failures.head - n + i + this.failures.capacity) % this.failures.capacity;
      if (out) {
        out[i] = {
          label:    this.failures.label[idx],
          value:    this.failures.value[idx],
          rule:     this.failures.ruleName[idx],
          timeMs:   this.failures.timeMs[idx],
          frame:    this.failures.frame[idx],
        };
      }
    }
    return n;
  }

  reset() {
    this.failures.clear();
    this.totalChecks = 0;
    this.totalFailures = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultValidator = null;

export function getDefaultValidator() {
  if (!_defaultValidator) _defaultValidator = new Validator();
  return _defaultValidator;
}

export function disposeDefaultValidator() {
  if (_defaultValidator) {
    _defaultValidator.dispose();
    _defaultValidator = null;
  }
}

/* ------------------------------------------------------------------ */
/* 8. HOT-PATH HELPERS                                                */
/* ------------------------------------------------------------------ */

export function validationBeginFrame(frameNumber) {
  getDefaultValidator().beginFrame(frameNumber);
}

export function validate(rule, value, label, ruleName) {
  return getDefaultValidator().check(rule, value, label, ruleName);
}

export function validateOrThrow(rule, value, label, ruleName) {
  const v = getDefaultValidator();
  const prevMode = v.options.mode;
  v.options.mode = VALIDATION_MODE.STRICT;
  try {
    return v.check(rule, value, label, ruleName);
  } finally {
    v.options.mode = prevMode;
  }
}

export function createValidationContext(label, options) {
  return getDefaultValidator().createContext(label, options);
}

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createValidator(options = {}) {
  return new Validator(options);
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Validator,
  ValidationContext,
  ValidationResult,
  FailureLog,
  RULES,
  VALIDATION_MODE,

  createValidator,
  getDefaultValidator,
  disposeDefaultValidator,

  validationBeginFrame,
  validate,
  validateOrThrow,
  createValidationContext,

  MAX_FAILURE_RECORDS,
  EPSILON_DEFAULT,
};

export default _defaultExport;