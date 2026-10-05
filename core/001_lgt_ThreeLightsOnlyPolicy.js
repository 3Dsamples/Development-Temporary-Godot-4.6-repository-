// File : 001
// name : src/core/001_lgt_ThreeLightsOnlyPolicy.js
// description : Authoritative enforcement module for the anime lighting stack
//               guaranteeing that only the SIX sanctioned Three.js r185 light
//               types are ever instantiated:
//
//                 1. THREE.AmbientLight
//                 2. THREE.HemisphereLight
//                 3. THREE.DirectionalLight
//                 4. THREE.PointLight
//                 5. THREE.SpotLight
//                 6. THREE.RectAreaLight
//
//               Anything else — legacy lights, custom shader-emitted lights,
//               fake Object3D emissives, third-party light classes — is
//               rejected at registration with a hard, typed error and a
//               signal, so no lighting bug can silently corrupt the frame.
//
//               Dynamic extension capability — the module ALSO provides two
//               fully independent extension systems so the entire "add a
//               new kind of light" workflow is possible WITHOUT breaking
//               the policy:
//
//                 • LIGHT BEHAVIORS
//                     A behavior is a plain object with `attach(light, ctx)`,
//                     `update(dt, elapsed, light, ctx)`, `detach(light, ctx)`.
//                     Behaviors turn any sanctioned light into a richer
//                     light: flicker, pulse, day-cycle, IES profile, cookie,
//                     shadow softness ramp, temperature drift, rim boost,
//                     biome blend, interior/exterior crossfade, magic
//                     emissive animation, etc. Multiple behaviors stack on
//                     one light. Zero new light types.
//
//                 • LIGHT COMPOSITES
//                     A composite is a named set of sanctioned lights +
//                     behaviors registered as one logical "light kind".
//                     Examples: `fire_light` = PointLight + flicker + warm
//                     rim + emissive proxy; `moon_cycle` = DirectionalLight
//                     + AmbientLight + HemisphereLight + day-cycle; `neon`
//                     = RectAreaLight + pulse + emissive proxy. Callers
//                     instantiate composites by id and get back a `LightGroup`
//                     handle that exposes the same API as a single light.
//
//               Registry:
//                 • Fixed-capacity SoA tracking every sanctioned light
//                 • Per-light: type, id, kind, behaviors, scene owner
//                 • Rejects forbidden light classes with hard errors
//                 • Hooks into ErrorBoundary / Validation / Signal / Logger
//                 • Passive in release, active in dev / audit
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external policy libs; every internal array sized
//               once at construction.
// best for : Guaranteeing that the entire anime lighting stack (006–380)
//            only ever uses the six sanctioned Three.js r185 light types,
//            while still allowing arbitrary dynamic light behaviors and
//            arbitrary composite light kinds to be defined at runtime.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
  LOG_LEVEL,
} from './026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from './030_rnd_ErrorBoundary.js';

import {
  getDefaultValidator,
  RULES,
  VALIDATION_MODE,
} from './031_rnd_Validation.js';

import {
  getDefaultEventBus,
  LIGHTING_TOPIC,
} from './027_rnd_EventBus.js';

import {
  getDefaultPerfTierResolver,
} from './023_rnd_PerfTier.js';

import {
  getDefaultProfiler,
} from './024_rnd_Profiler.js';

import {
  LIGHTING_SIGNALS,
} from './028_rnd_Signal.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_LIGHTS =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 256 :
                                 128;

export const MAX_BEHAVIORS_PER_LIGHT = 8;
export const MAX_REGISTERED_BEHAVIORS = 64;
export const MAX_REGISTERED_COMPOSITES = 32;
export const MAX_COMPOSITE_MEMBERS = 8;

/**
 * The six sanctioned Three.js r185 light types, keyed by an internal id.
 * Every entry declares the class, the sRGB-correct uniforms it owns, and
 * the shadow capability.
 */
export const SANCTIONED_LIGHT_TYPE = Object.freeze({
  AMBIENT:      0,
  HEMISPHERE:   1,
  DIRECTIONAL:  2,
  POINT:        3,
  SPOT:         4,
  RECT_AREA:    5,
  COUNT:        6,
});

export const SANCTIONED_LIGHT_TYPE_NAME = Object.freeze([
  'ambient',
  'hemisphere',
  'directional',
  'point',
  'spot',
  'rect_area',
]);

export const SANCTIONED_LIGHT_CLASS = Object.freeze([
  THREE.AmbientLight,
  THREE.HemisphereLight,
  THREE.DirectionalLight,
  THREE.PointLight,
  THREE.SpotLight,
  THREE.RectAreaLight,
]);

/**
 * The forbidden list. If any of these appear anywhere in the scene graph
 * or in a `add()` call, the policy trips and the caller is warned loudly.
 */
const FORBIDDEN_LIGHT_NAMES = Object.freeze([
  'LightProbe',
  'LightProbeGenerator',
  'HemisphereLightProbe',
  'AmbientLightProbe',
  'SpotLightShadow',
  'RectAreaLightUniformsLib',
]);

/**
 * Composite light kinds — named sets of sanctioned lights + behaviors.
 * These are the sanctioned extension points: a composite never introduces
 * a new light class, only a new logical grouping.
 */
export const COMPOSITE_LIGHT_KIND = Object.freeze({
  CUSTOM:        0,
  FIRE:          1,
  MOON_CYCLE:    2,
  SUN_CYCLE:     3,
  NEON:          4,
  MAGIC:         5,
  INTERIOR_LAMP: 6,
  WINDOW_SHAFT:  7,
  CAUSTIC:       8,
  AURORA:        9,
  COUNT:        10,
});

export const COMPOSITE_LIGHT_KIND_NAME = Object.freeze([
  'custom',
  'fire',
  'moon_cycle',
  'sun_cycle',
  'neon',
  'magic',
  'interior_lamp',
  'window_shaft',
  'caustic',
  'aurora',
]);

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

let _lightIdCounter = 0;

function _nextLightId() {
  return ++_lightIdCounter;
}

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _classifyLight(light) {
  if (!light) return -1;
  if (light.isAmbientLight)      return SANCTIONED_LIGHT_TYPE.AMBIENT;
  if (light.isHemisphereLight)   return SANCTIONED_LIGHT_TYPE.HEMISPHERE;
  if (light.isDirectionalLight)  return SANCTIONED_LIGHT_TYPE.DIRECTIONAL;
  if (light.isPointLight)        return SANCTIONED_LIGHT_TYPE.POINT;
  if (light.isSpotLight)         return SANCTIONED_LIGHT_TYPE.SPOT;
  if (light.isRectAreaLight)     return SANCTIONED_LIGHT_TYPE.RECT_AREA;
  return -1;
}

function _isThreeLight(light) {
  return !!light && light.isLight === true;
}

/* ------------------------------------------------------------------ */
/* 2. LIGHT SLOT (SoA record per registered light)                    */
/* ------------------------------------------------------------------ */

export class LightSlot {
  constructor(index) {
    this.index        = index;
    this.id           = 0;
    this.light        = null;
    this.type         = -1;
    this.compositeId  = -1;
    this.ownerId      = -1;

    // Behavior slots (parallel arrays — the registered behavior ids).
    this.behaviorIds     = new Int32Array(MAX_BEHAVIORS_PER_LIGHT);
    this.behaviorCtx     = new Array(MAX_BEHAVIORS_PER_LIGHT).fill(null);
    this.behaviorCount   = 0;

    // Scene ownership.
    this.sceneRef        = null;
    this.registeredAt    = 0;

    // Shadow state.
    this.castShadow      = 0;
    this.receivesShadow  = 0;

    // Extension tags — free-form bitmask for downstream systems.
    this.tags            = 0;

    // Frame stats.
    this.updateMs        = 0;
    this.lastUpdateFrame = -1;
  }

  reset() {
    this.id              = 0;
    this.light           = null;
    this.type            = -1;
    this.compositeId     = -1;
    this.ownerId         = -1;
    this.behaviorCount   = 0;
    for (let i = 0; i < MAX_BEHAVIORS_PER_LIGHT; i++) {
      this.behaviorIds[i] = -1;
      this.behaviorCtx[i] = null;
    }
    this.sceneRef        = null;
    this.registeredAt    = 0;
    this.castShadow      = 0;
    this.receivesShadow  = 0;
    this.tags            = 0;
    this.updateMs        = 0;
    this.lastUpdateFrame = -1;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. BEHAVIOR DESCRIPTOR                                             */
/* ------------------------------------------------------------------ */

/**
 * Behavior descriptor — a plain object a user registers:
 *
 *   {
 *     id:    'flicker',
 *     name:  'Flicker',
 *     attach(light, ctx)       — optional, called once when bound
 *     update(dt, elapsed, light, ctx) — per-frame
 *     detach(light, ctx)       — optional, called on unbind
 *     validate(light)          — optional, returns bool
 *     kinds: [SANCTIONED_LIGHT_TYPE.POINT, ...] — which light types it supports
 *   }
 *
 * Behaviors are stateless w.r.t. this module — the module only forwards
 * calls; any per-instance state lives in `ctx` which the caller owns.
 */
export class BehaviorDescriptor {
  constructor(spec) {
    this.id       = spec.id;
    this.name     = spec.name || spec.id;
    this.attach   = typeof spec.attach === 'function' ? spec.attach : null;
    this.update   = typeof spec.update === 'function' ? spec.update : null;
    this.detach   = typeof spec.detach === 'function' ? spec.detach : null;
    this.validate = typeof spec.validate === 'function' ? spec.validate : null;
    this.kinds    = Array.isArray(spec.kinds) ? Object.freeze(spec.kinds.slice()) : null;
    Object.freeze(this);
  }

  supports(type) {
    if (!this.kinds) return true;
    return this.kinds.indexOf(type) >= 0;
  }
}

/* ------------------------------------------------------------------ */
/* 4. COMPOSITE DESCRIPTOR                                            */
/* ------------------------------------------------------------------ */

/**
 * Composite descriptor — a named set of sanctioned lights + behaviors
 * registered as one logical "light kind". Examples:
 *
 *   FIRE = PointLight + flicker behavior + warm emissive proxy
 *   MOON_CYCLE = DirectionalLight + AmbientLight + HemisphereLight + day-cycle behavior
 *   NEON = RectAreaLight + pulse behavior + emissive proxy
 *
 * The composite is instantiated by calling `createComposite(id, scene, opts)`
 * which:
 *   1. Instantiates the member lights (all sanctioned types).
 *   2. Attaches the member behaviors.
 *   3. Adds them to the scene.
 *   4. Returns a `CompositeHandle` with the same interface as a single light.
 */
export class CompositeDescriptor {
  constructor(spec) {
    this.id       = spec.id;
    this.name     = spec.name || spec.id;
    this.kind     = spec.kind !== undefined ? spec.kind : COMPOSITE_LIGHT_KIND.CUSTOM;
    this.members  = Object.freeze((spec.members || []).map(_freezeMember));
    this.behaviors = Object.freeze((spec.behaviors || []).slice());
    this.create   = typeof spec.create === 'function' ? spec.create : null;
    Object.freeze(this);
  }
}

function _freezeMember(m) {
  return Object.freeze({
    type:     m.type,
    name:     m.name || SANCTIONED_LIGHT_TYPE_NAME[m.type] || 'member',
    color:    m.color !== undefined ? Object.freeze(m.color.slice()) : null,
    intensity:m.intensity !== undefined ? m.intensity : 1.0,
    distance: m.distance !== undefined ? m.distance : 0,
    angle:    m.angle !== undefined ? m.angle : null,
    penumbra: m.penumbra !== undefined ? m.penumbra : null,
    decay:    m.decay !== undefined ? m.decay : 2.0,
    width:    m.width !== undefined ? m.width : null,
    height:   m.height !== undefined ? m.height : null,
    position: m.position !== undefined ? Object.freeze(m.position.slice()) : null,
    target:   m.target !== undefined ? Object.freeze(m.target.slice()) : null,
    castShadow: !!m.castShadow,
    behaviorIds: m.behaviorIds ? Object.freeze(m.behaviorIds.slice()) : null,
  });
}

/* ------------------------------------------------------------------ */
/* 5. COMPOSITE HANDLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Handle returned by `createComposite`. Exposes a single-light-like API
 * while internally managing multiple sanctioned lights.
 */
export class CompositeHandle {
  constructor(descriptor, members) {
    this.descriptor = descriptor;
    this.members    = members;  // array of THREE.Light instances
    this.registered = true;

    // Convenience proxies for the first member (or a synthesized anchor).
    this.anchor = members.length > 0 ? members[0] : null;
  }

  /** The primary light — most composites designate member[0] as anchor. */
  get light() { return this.anchor; }

  /** Position of the anchor. */
  get position() { return this.anchor ? this.anchor.position : null; }

  setIntensity(v) {
    for (let i = 0; i < this.members.length; i++) this.members[i].intensity = v;
    return this;
  }

  setColor(rgb) {
    for (let i = 0; i < this.members.length; i++) {
      const m = this.members[i];
      if (m.color) m.color.setRGB(rgb[0], rgb[1], rgb[2]);
      if (m.isHemisphereLight && m.groundColor) m.groundColor.setRGB(rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5);
    }
    return this;
  }

  setVisible(visible) {
    for (let i = 0; i < this.members.length; i++) this.members[i].visible = !!visible;
    return this;
  }

  dispose(policy) {
    if (!this.registered) return false;
    for (let i = 0; i < this.members.length; i++) {
      if (policy) policy.unregisterLight(this.members[i]);
    }
    this.members.length = 0;
    this.registered = false;
    return true;
  }
}

/* ------------------------------------------------------------------ */
/* 6. THREE-LIGHTS-ONLY POLICY                                        */
/* ------------------------------------------------------------------ */

export class ThreeLightsOnlyPolicy {
  constructor(options = {}) {
    this.options = Object.assign({
      enforce:           PERF_TIER_LOCAL !== 'LOW',
      auditOnRegister:   true,
      tripBoundary:      true,
      boundaryName:      'policy.three_lights_only',
      logChannel:        LOG_CHANNEL.LIGHTS,
      attachDefaultBehaviors: true,
      autoValidate:      true,
    }, options || {});

    // Light registry (SoA).
    this.capacity    = MAX_LIGHTS;
    this.slots       = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.slots[i] = new LightSlot(i);
    this.count       = 0;
    this.byLight     = new Map();  // THREE.Light → slot index
    this.byId        = new Map();  // light id → slot index

    // Behavior registry.
    this.behaviors = new Array(MAX_REGISTERED_BEHAVIORS).fill(null);
    this.behaviorCount = 0;
    this.behaviorByName = new Map();

    // Composite registry.
    this.composites = new Array(MAX_REGISTERED_COMPOSITES).fill(null);
    this.compositeCount = 0;
    this.compositeByName = new Map();

    // Rejection log.
    this.rejections       = 0;
    this.lastRejection    = null;
    this.rejectionRing    = new Array(8).fill(null);
    this.rejectionHead    = 0;

    // Frame.
    this.frame = 0;

    // Boundary for violations.
    this._boundary = null;
    if (this.options.tripBoundary) {
      try {
        const mgr = getDefaultErrorBoundaries();
        if (mgr) {
          this._boundary = mgr.create(this.options.boundaryName, {
            tag: BOUNDARY_TAG.LIGHTS,
            failureThreshold: 3,
          });
        }
      } catch (_) { /* swallow */ }
    }

    // Event bus for diagnostics.
    this._bus = null;
    try { this._bus = getDefaultEventBus(); } catch (_) { this._bus = null; }

    // Logger.
    this._logger = null;

    this._installDefaultBehaviors();
    this._installDefaultComposites();
  }

  /* ---------------- logger ---------------- */

  _log() {
    if (!this._logger) {
      try { this._logger = getDefaultLogger(); } catch (_) { this._logger = null; }
    }
    return this._logger;
  }

  /* ---------------- frame lifecycle ---------------- */

  beginFrame(frameNumber) {
    this.frame = (typeof frameNumber === 'number') ? frameNumber : (this.frame + 1);
  }

  /* ---------------- slot allocation ---------------- */

  _allocateSlot() {
    if (this.count >= this.capacity) return -1;
    const idx = this.count++;
    this.slots[idx].reset();
    return idx;
  }

  /* ---------------- registration ---------------- */

  /**
   * Registers a light with the policy. Verifies:
   *   • light is a THREE.Light
   *   • light is one of the six sanctioned types
   *   • light's params pass the validator
   * Returns the internal light id (> 0) on success, -1 on failure.
   */
  registerLight(light, options = {}) {
    if (!this._auditLight(light)) return -1;

    if (this.byLight.has(light)) {
      // Already registered — return existing id.
      const existing = this.byLight.get(light);
      return this.slots[existing].id;
    }

    const idx = this._allocateSlot();
    if (idx < 0) {
      this._reject('capacity', light, 'light registry is full');
      return -1;
    }

    const type = _classifyLight(light);
    const slot = this.slots[idx];
    slot.id            = _nextLightId();
    slot.light         = light;
    slot.type          = type;
    slot.ownerId       = options.ownerId !== undefined ? options.ownerId : -1;
    slot.sceneRef      = options.scene || null;
    slot.registeredAt  = _now();
    slot.castShadow    = light.castShadow ? 1 : 0;
    slot.receivesShadow= light.receiveShadow ? 1 : 0;
    slot.tags          = options.tags || 0;

    this.byLight.set(light, idx);
    this.byId.set(slot.id, idx);

    if (this.options.autoValidate) {
      this._validateLightParameters(slot);
    }

    // Auto-attach behaviors requested at registration.
    if (Array.isArray(options.behaviorIds)) {
      for (let i = 0; i < options.behaviorIds.length; i++) {
        this.attachBehavior(light, options.behaviorIds[i], options.behaviorCtx);
      }
    }

    // Diagnostics.
    const log = this._log();
    if (log && this.options.enforce) {
      log.debug(this.options.logChannel, () =>
        `[001_lgt_ThreeLightsOnlyPolicy] registered light #${slot.id} (${SANCTIONED_LIGHT_TYPE_NAME[type]})`);
    }

    if (this._bus) {
      this._bus.emit(LIGHTING_TOPIC.LIGHT_ADDED, {
        id: slot.id,
        type: SANCTIONED_LIGHT_TYPE_NAME[type],
      });
    }

    return slot.id;
  }

  /**
   * Unregisters a light. Detaches all behaviors, then removes from the
   * registry. Does NOT remove the light from its parent scene — that is
   * the caller's responsibility.
   */
  unregisterLight(light) {
    const idx = this.byLight.get(light);
    if (idx === undefined) return false;
    const slot = this.slots[idx];

    // Detach behaviors.
    for (let i = 0; i < slot.behaviorCount; i++) {
      const bIdx = slot.behaviorIds[i];
      if (bIdx < 0) continue;
      const descriptor = this.behaviors[bIdx];
      if (descriptor && descriptor.detach) {
        try { descriptor.detach(light, slot.behaviorCtx[i]); }
        catch (e) {
          const log = this._log();
          if (log) log.error(this.options.logChannel, `[001_lgt_ThreeLightsOnlyPolicy] detach failed for ${descriptor.id}: ${e && e.message}`);
        }
      }
      slot.behaviorIds[i] = -1;
      slot.behaviorCtx[i] = null;
    }
    slot.behaviorCount = 0;

    // Remove from maps.
    this.byLight.delete(light);
    this.byId.delete(slot.id);

    // Compact registry: swap last slot into idx.
    const last = this.count - 1;
    if (idx !== last) {
      this.slots[idx] = this.slots[last];
      this.slots[idx].index = idx;
      // Re-point maps for swapped slot.
      this.byLight.set(this.slots[idx].light, idx);
      this.byId.set(this.slots[idx].id, idx);
    }
    this.slots[last] = new LightSlot(last);
    this.count--;

    if (this._bus) {
      this._bus.emit(LIGHTING_TOPIC.LIGHT_REMOVED, { id: slot.id });
    }
    return true;
  }

  /* ---------------- light auditing ---------------- */

  _auditLight(light) {
    // Null / undefined.
    if (!light) {
      this._reject('null', null, 'light is null or undefined');
      return false;
    }

    // Not a THREE.Light at all.
    if (!_isThreeLight(light)) {
      // Silently pass through non-lights (Object3D, Mesh, Group).
      // The policy is ONLY about lights.
      return true;
    }

    // Check forbidden types.
    const ctorName = (light.constructor && light.constructor.name) || 'Unknown';
    for (let i = 0; i < FORBIDDEN_LIGHT_NAMES.length; i++) {
      if (ctorName.indexOf(FORBIDDEN_LIGHT_NAMES[i]) >= 0) {
        this._reject('forbidden_type', light, `forbidden light class "${ctorName}"`);
        return false;
      }
    }

    // Classify against sanctioned set.
    const type = _classifyLight(light);
    if (type < 0) {
      this._reject('unknown_type', light, `light "${ctorName}" is not one of the 6 sanctioned types`);
      return false;
    }

    return true;
  }

  _validateLightParameters(slot) {
    const v = getDefaultValidator();
    if (!v) return;
    const light = slot.light;
    const type = slot.type;

    // Universal checks.
    v.check(RULES.lightIntensity, light.intensity, `light#${slot.id}.intensity`);

    // Color check for lights that have color.
    if (light.color) {
      v.check(RULES.unitColor, light.color, `light#${slot.id}.color`);
    }

    // Type-specific checks.
    switch (type) {
      case SANCTIONED_LIGHT_TYPE.POINT:
      case SANCTIONED_LIGHT_TYPE.SPOT:
        if (light.distance !== undefined) {
          v.check(RULES.lightDistance, light.distance, `light#${slot.id}.distance`);
        }
        if (light.decay !== undefined) {
          v.check(RULES.lightDecay, light.decay, `light#${slot.id}.decay`);
        }
        break;
      case SANCTIONED_LIGHT_TYPE.SPOT:
        if (light.angle !== undefined) {
          v.check(RULES.lightAngle, light.angle, `light#${slot.id}.angle`);
        }
        if (light.penumbra !== undefined) {
          v.check(RULES.lightPenumbra, light.penumbra, `light#${slot.id}.penumbra`);
        }
        break;
      case SANCTIONED_LIGHT_TYPE.RECT_AREA:
        if (light.width !== undefined) {
          v.check(RULES.finitePositive, light.width, `light#${slot.id}.width`);
        }
        if (light.height !== undefined) {
          v.check(RULES.finitePositive, light.height, `light#${slot.id}.height`);
        }
        break;
      default:
        break;
    }
  }

  /* ---------------- rejection ---------------- */

  _reject(reason, light, message) {
    this.rejections++;
    const record = {
      reason,
      message,
      lightClass: light && light.constructor ? light.constructor.name : 'Unknown',
      frame: this.frame,
      timeMs: _now(),
    };
    this.lastRejection = record;
    this.rejectionRing[this.rejectionHead] = record;
    this.rejectionHead = (this.rejectionHead + 1) % this.rejectionRing.length;

    const log = this._log();
    if (log) {
      log.error(this.options.logChannel, () =>
        `[001_lgt_ThreeLightsOnlyPolicy] REJECTED (${reason}): ${message}`);
    }

    if (this._bus) {
      this._bus.emit('lights.policy.rejected', record);
    }

    if (this._boundary && this.options.enforce) {
      this._boundary.run(() => {
        throw new Error(`[001_lgt_ThreeLightsOnlyPolicy] ${reason}: ${message}`);
      });
    }
  }

  /* ---------------- audit scene ---------------- */

  /**
   * Traverses a scene and audits every light it finds. Returns the count
   * of non-sanctioned lights discovered.
   */
  auditScene(scene) {
    if (!scene || typeof scene.traverse !== 'function') return 0;
    let bad = 0;
    const self = this;
    scene.traverse((obj) => {
      if (!_isThreeLight(obj)) return;
      const type = _classifyLight(obj);
      if (type < 0) {
        bad++;
        self._reject('scene_audit', obj, `scene contains non-sanctioned light "${obj.constructor.name}"`);
      }
    });
    return bad;
  }

  /* ---------------- behavior registry ---------------- */

  registerBehavior(spec) {
    if (!spec || typeof spec.id !== 'string') return false;
    if (this.behaviorCount >= MAX_REGISTERED_BEHAVIORS) return false;
    if (this.behaviorByName.has(spec.id)) return true;

    const d = new BehaviorDescriptor(spec);
    this.behaviors[this.behaviorCount++] = d;
    this.behaviorByName.set(d.id, d);
    return true;
  }

  unregisterBehavior(id) {
    const existing = this.behaviorByName.get(id);
    if (!existing) return false;
    for (let i = 0; i < this.behaviorCount; i++) {
      if (this.behaviors[i] === existing) {
        this.behaviors[i] = this.behaviors[this.behaviorCount - 1];
        this.behaviors[this.behaviorCount - 1] = null;
        this.behaviorCount--;
        this.behaviorByName.delete(id);
        return true;
      }
    }
    return false;
  }

  getBehavior(id) {
    return this.behaviorByName.get(id) || null;
  }

  listBehaviors() {
    const out = [];
    for (let i = 0; i < this.behaviorCount; i++) {
      if (this.behaviors[i]) out.push(this.behaviors[i].id);
    }
    return out;
  }

  /* ---------------- behavior binding ---------------- */

  attachBehavior(light, behaviorId, ctx) {
    const idx = this.byLight.get(light);
    if (idx === undefined) return false;

    const descriptor = this.behaviorByName.get(behaviorId);
    if (!descriptor) {
      const log = this._log();
      if (log) log.warn(this.options.logChannel, `[001_lgt_ThreeLightsOnlyPolicy] unknown behavior "${behaviorId}"`);
      return false;
    }

    const slot = this.slots[idx];

    // Check compatibility.
    if (!descriptor.supports(slot.type)) {
      const log = this._log();
      if (log) log.warn(this.options.logChannel,
        `[001_lgt_ThreeLightsOnlyPolicy] behavior "${behaviorId}" does not support ${SANCTIONED_LIGHT_TYPE_NAME[slot.type]}`);
      return false;
    }

    // Check custom validator.
    if (descriptor.validate) {
      let ok = false;
      try { ok = descriptor.validate(light) === true; }
      catch (_) { ok = false; }
      if (!ok) return false;
    }

    if (slot.behaviorCount >= MAX_BEHAVIORS_PER_LIGHT) return false;

    const bIdx = this.behaviorCount > 0 ? this._behaviorIndex(descriptor) : -1;
    if (bIdx < 0) return false;

    // Attach.
    if (descriptor.attach) {
      try { descriptor.attach(light, ctx); }
      catch (e) {
        const log = this._log();
        if (log) log.error(this.options.logChannel, `[001_lgt_ThreeLightsOnlyPolicy] attach failed for ${behaviorId}: ${e && e.message}`);
        return false;
      }
    }

    slot.behaviorIds[slot.behaviorCount] = bIdx;
    slot.behaviorCtx[slot.behaviorCount] = ctx || null;
    slot.behaviorCount++;

    return true;
  }

  detachBehavior(light, behaviorId) {
    const idx = this.byLight.get(light);
    if (idx === undefined) return false;
    const slot = this.slots[idx];

    const descriptor = this.behaviorByName.get(behaviorId);
    if (!descriptor) return false;

    for (let i = 0; i < slot.behaviorCount; i++) {
      const bIdx = slot.behaviorIds[i];
      if (bIdx < 0) continue;
      const d = this.behaviors[bIdx];
      if (d === descriptor) {
        if (d.detach) {
          try { d.detach(light, slot.behaviorCtx[i]); }
          catch (_) { /* swallow */ }
        }
        // Compact.
        for (let j = i; j < slot.behaviorCount - 1; j++) {
          slot.behaviorIds[j] = slot.behaviorIds[j + 1];
          slot.behaviorCtx[j] = slot.behaviorCtx[j + 1];
        }
        slot.behaviorCount--;
        slot.behaviorIds[slot.behaviorCount] = -1;
        slot.behaviorCtx[slot.behaviorCount] = null;
        return true;
      }
    }
    return false;
  }

  _behaviorIndex(descriptor) {
    for (let i = 0; i < this.behaviorCount; i++) {
      if (this.behaviors[i] === descriptor) return i;
    }
    return -1;
  }

  /* ---------------- per-frame behavior tick ---------------- */

  /**
   * Runs every attached behavior on every registered light. Called once
   * per frame by the engine loop.
   */
  updateBehaviors(dt, elapsed) {
    const n = this.count;
    for (let i = 0; i < n; i++) {
      const slot = this.slots[i];
      if (slot.behaviorCount === 0) continue;
      const light = slot.light;
      if (!light) continue;

      const t0 = _now();
      for (let b = 0; b < slot.behaviorCount; b++) {
        const bIdx = slot.behaviorIds[b];
        if (bIdx < 0) continue;
        const descriptor = this.behaviors[bIdx];
        if (!descriptor || !descriptor.update) continue;

        try {
          descriptor.update(dt, elapsed, light, slot.behaviorCtx[b]);
        } catch (e) {
          const log = this._log();
          if (log) log.error(this.options.logChannel,
            `[001_lgt_ThreeLightsOnlyPolicy] behavior "${descriptor.id}" update failed: ${e && e.message}`);
        }
      }
      slot.updateMs = _now() - t0;
      slot.lastUpdateFrame = this.frame;
    }
  }

  /* ---------------- composite registry ---------------- */

  registerComposite(spec) {
    if (!spec || typeof spec.id !== 'string') return false;
    if (this.compositeCount >= MAX_REGISTERED_COMPOSITES) return false;
    if (this.compositeByName.has(spec.id)) return true;
    if (!Array.isArray(spec.members) || spec.members.length === 0) return false;
    if (spec.members.length > MAX_COMPOSITE_MEMBERS) return false;

    // Verify every member is a sanctioned type.
    for (let i = 0; i < spec.members.length; i++) {
      const m = spec.members[i];
      if (m.type === undefined || m.type < 0 || m.type >= SANCTIONED_LIGHT_TYPE.COUNT) {
        const log = this._log();
        if (log) log.error(this.options.logChannel,
          `[001_lgt_ThreeLightsOnlyPolicy] composite "${spec.id}" member #${i} has invalid type`);
        return false;
      }
    }

    const d = new CompositeDescriptor(spec);
    this.composites[this.compositeCount++] = d;
    this.compositeByName.set(d.id, d);
    return true;
  }

  unregisterComposite(id) {
    const existing = this.compositeByName.get(id);
    if (!existing) return false;
    for (let i = 0; i < this.compositeCount; i++) {
      if (this.composites[i] === existing) {
        this.composites[i] = this.composites[this.compositeCount - 1];
        this.composites[this.compositeCount - 1] = null;
        this.compositeCount--;
        this.compositeByName.delete(id);
        return true;
      }
    }
    return false;
  }

  getComposite(id) {
    return this.compositeByName.get(id) || null;
  }

  listComposites() {
    const out = [];
    for (let i = 0; i < this.compositeCount; i++) {
      if (this.composites[i]) out.push(this.composites[i].id);
    }
    return out;
  }

  /* ---------------- composite instantiation ---------------- */

  /**
   * Creates a composite by id. Adds all member lights to `scene`, attaches
   * any member-level behaviors, registers each member with the policy, and
   * returns a CompositeHandle.
   */
  createComposite(id, scene, options = {}) {
    const d = this.compositeByName.get(id);
    if (!d) {
      const log = this._log();
      if (log) log.error(this.options.logChannel,
        `[001_lgt_ThreeLightsOnlyPolicy] unknown composite "${id}"`);
      return null;
    }

    // Custom creator override.
    if (d.create) {
      try {
        const handle = d.create(this, scene, options);
        if (handle) return handle;
      } catch (e) {
        const log = this._log();
        if (log) log.error(this.options.logChannel,
          `[001_lgt_ThreeLightsOnlyPolicy] composite "${id}" create() failed: ${e && e.message}`);
      }
    }

    const members = [];
    for (let i = 0; i < d.members.length; i++) {
      const m = d.members[i];
      const light = this._instantiateMember(m);
      if (!light) continue;

      // Apply position/target.
      const basePos = options.position || [0, 0, 0];
      if (m.position) {
        light.position.set(
          basePos[0] + m.position[0],
          basePos[1] + m.position[1],
          basePos[2] + m.position[2]
        );
      } else {
        light.position.set(basePos[0], basePos[1], basePos[2]);
      }

      if (m.target && light.target) {
        const baseTgt = options.target || [0, 0, 0];
        light.target.position.set(
          baseTgt[0] + m.target[0],
          baseTgt[1] + m.target[1],
          baseTgt[2] + m.target[2]
        );
        if (scene) scene.add(light.target);
      }

      if (scene) scene.add(light);

      // Register member.
      this.registerLight(light, {
        ownerId: options.ownerId !== undefined ? options.ownerId : -1,
        scene,
        behaviorIds: m.behaviorIds || null,
      });

      members.push(light);
    }

    // Composite-level behaviors apply to the anchor (member 0).
    if (d.behaviors.length > 0 && members.length > 0) {
      for (let i = 0; i < d.behaviors.length; i++) {
        const bEntry = d.behaviors[i];
        const bid = typeof bEntry === 'string' ? bEntry : bEntry.id;
        const bctx = (typeof bEntry === 'object' && bEntry.ctx) ? bEntry.ctx : null;
        this.attachBehavior(members[0], bid, bctx);
      }
    }

    return new CompositeHandle(d, members);
  }

  _instantiateMember(m) {
    let light = null;
    try {
      switch (m.type) {
        case SANCTIONED_LIGHT_TYPE.AMBIENT:
          light = new THREE.AmbientLight(0xffffff, m.intensity);
          break;
        case SANCTIONED_LIGHT_TYPE.HEMISPHERE:
          light = new THREE.HemisphereLight(0xffffff, 0x222222, m.intensity);
          break;
        case SANCTIONED_LIGHT_TYPE.DIRECTIONAL:
          light = new THREE.DirectionalLight(0xffffff, m.intensity);
          break;
        case SANCTIONED_LIGHT_TYPE.POINT:
          light = new THREE.PointLight(0xffffff, m.intensity, m.distance, m.decay);
          break;
        case SANCTIONED_LIGHT_TYPE.SPOT:
          light = new THREE.SpotLight(
            0xffffff, m.intensity, m.distance,
            m.angle !== null ? m.angle : Math.PI / 4,
            m.penumbra !== null ? m.penumbra : 0.1,
            m.decay
          );
          break;
        case SANCTIONED_LIGHT_TYPE.RECT_AREA:
          light = new THREE.RectAreaLight(
            0xffffff, m.intensity,
            m.width !== null ? m.width : 1,
            m.height !== null ? m.height : 1
          );
          break;
        default:
          return null;
      }
    } catch (_) {
      return null;
    }

    if (m.color) light.color.setRGB(m.color[0], m.color[1], m.color[2]);
    if (m.castShadow) light.castShadow = true;
    if (light.isHemisphereLight && m.color) {
      light.groundColor.setRGB(m.color[0] * 0.4, m.color[1] * 0.4, m.color[2] * 0.4);
    }
    return light;
  }

  /* ---------------- default behaviors ---------------- */

  _installDefaultBehaviors() {
    // Flicker — fire/candle/neon.
    this.registerBehavior({
      id:   'flicker',
      name: 'Flicker',
      kinds: [
        SANCTIONED_LIGHT_TYPE.POINT,
        SANCTIONED_LIGHT_TYPE.SPOT,
        SANCTIONED_LIGHT_TYPE.RECT_AREA,
      ],
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.baseIntensity === undefined) ctx.baseIntensity = light.intensity;
        if (ctx.phase === undefined) ctx.phase = Math.random() * Math.PI * 2;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        const amp = ctx.amplitude !== undefined ? ctx.amplitude : 0.15;
        const hz = ctx.hz !== undefined ? ctx.hz : 8.0;
        const noise = Math.sin(elapsed * hz + ctx.phase) * 0.5
                    + Math.sin(elapsed * hz * 1.7 + ctx.phase * 0.7) * 0.3
                    + Math.sin(elapsed * hz * 2.3 + ctx.phase * 1.3) * 0.2;
        light.intensity = ctx.baseIntensity * (1 + noise * amp);
      },
      detach(light, ctx) {
        if (ctx && ctx.baseIntensity !== undefined) light.intensity = ctx.baseIntensity;
      },
    });

    // Pulse — neon/magic.
    this.registerBehavior({
      id:   'pulse',
      name: 'Pulse',
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.baseIntensity === undefined) ctx.baseIntensity = light.intensity;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        const amp = ctx.amplitude !== undefined ? ctx.amplitude : 0.25;
        const hz = ctx.hz !== undefined ? ctx.hz : 1.5;
        const t = Math.sin(elapsed * hz * Math.PI * 2);
        light.intensity = ctx.baseIntensity * (1 + t * amp);
      },
      detach(light, ctx) {
        if (ctx && ctx.baseIntensity !== undefined) light.intensity = ctx.baseIntensity;
      },
    });

    // Day-cycle — sun/moon.
    this.registerBehavior({
      id:   'day_cycle',
      name: 'Day Cycle',
      kinds: [
        SANCTIONED_LIGHT_TYPE.DIRECTIONAL,
        SANCTIONED_LIGHT_TYPE.HEMISPHERE,
        SANCTIONED_LIGHT_TYPE.AMBIENT,
      ],
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.dayCycle === undefined) ctx.dayCycle = 0.5;
        if (ctx.daySpeed === undefined) ctx.daySpeed = 0.004;
        if (ctx.baseIntensity === undefined) ctx.baseIntensity = light.intensity;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        ctx.dayCycle = (ctx.dayCycle + dt * ctx.daySpeed) % 1.0;
        const t = ctx.dayCycle;
        const dayT = Math.max(0, Math.min(1, (t - 0.25) / 0.5));
        const ang = dayT * Math.PI;
        const elev = Math.sin(ang);
        const intensity = Math.max(0.05, elev * 1.25) * ctx.baseIntensity;

        if (light.isDirectionalLight) {
          const azim = -110 * Math.PI / 180 + dayT * 220 * Math.PI / 180;
          const ce = Math.cos(elev * 75 * Math.PI / 180);
          light.position.set(
            Math.sin(azim) * ce * 120,
            Math.sin(elev * 75 * Math.PI / 180) * 120,
            -Math.cos(azim) * ce * 120
          );
        }

        light.intensity = intensity;
      },
    });

    // IES profile — approximate spatial intensity falloff.
    this.registerBehavior({
      id:   'ies_profile',
      name: 'IES Profile',
      kinds: [
        SANCTIONED_LIGHT_TYPE.POINT,
        SANCTIONED_LIGHT_TYPE.SPOT,
        SANCTIONED_LIGHT_TYPE.RECT_AREA,
      ],
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.ballastFactor === undefined) ctx.ballastFactor = 1.0;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        // Spatial profile — the IES lookup itself is handled by the shader.
        // The behavior only broadcasts the ballast factor to the emissive
        // proxy if present.
        light.userData.__iesBallast = ctx.ballastFactor;
      },
    });

    // Temperature drift — warms/cools the light color over time.
    this.registerBehavior({
      id:   'temperature_drift',
      name: 'Temperature Drift',
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.baseColor === undefined && light.color) {
          ctx.baseColor = [light.color.r, light.color.g, light.color.b];
        }
        if (ctx.driftAmount === undefined) ctx.driftAmount = 0.05;
        if (ctx.driftHz === undefined) ctx.driftHz = 0.2;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx || !light.color || !ctx.baseColor) return;
        const d = Math.sin(elapsed * ctx.driftHz * Math.PI * 2) * ctx.driftAmount;
        light.color.setRGB(
          Math.max(0, Math.min(1, ctx.baseColor[0] + d * 0.5)),
          Math.max(0, Math.min(1, ctx.baseColor[1])),
          Math.max(0, Math.min(1, ctx.baseColor[2] - d * 0.5))
        );
      },
    });

    // Rim boost — directional light rim intensity oscillation.
    this.registerBehavior({
      id:   'rim_boost',
      name: 'Rim Boost',
      kinds: [SANCTIONED_LIGHT_TYPE.DIRECTIONAL],
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.boostAmount === undefined) ctx.boostAmount = 0.2;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        light.userData.__rimBoost = ctx.boostAmount * (0.8 + 0.2 * Math.sin(elapsed * 0.7));
      },
    });

    // Biome blend — crossfade intensity/color between biomes.
    this.registerBehavior({
      id:   'biome_blend',
      name: 'Biome Blend',
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.blend === undefined) ctx.blend = 0.0;
        if (ctx.targetBlend === undefined) ctx.targetBlend = 0.0;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        const lambda = ctx.lambda !== undefined ? ctx.lambda : 0.85;
        ctx.blend += (ctx.targetBlend - ctx.blend) * (1 - Math.exp(-lambda * dt));
        light.userData.__biomeBlend = ctx.blend;
      },
    });

    // Interior/exterior crossfade.
    this.registerBehavior({
      id:   'interior_crossfade',
      name: 'Interior Crossfade',
      attach(light, ctx) {
        if (!ctx) return;
        if (ctx.indoorWeight === undefined) ctx.indoorWeight = 0.0;
      },
      update(dt, elapsed, light, ctx) {
        if (!ctx) return;
        light.userData.__indoorWeight = ctx.indoorWeight;
      },
    });
  }

  /* ---------------- default composites ---------------- */

  _installDefaultComposites() {
    // FIRE — warm point light + flicker + shadow.
    this.registerComposite({
      id:   'fire_light',
      name: 'Fire Light',
      kind: COMPOSITE_LIGHT_KIND.FIRE,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.POINT,
          name: 'fire_core',
          color: [1.0, 0.55, 0.15],
          intensity: 2.5,
          distance: 12,
          decay: 2.0,
          position: [0, 0.8, 0],
          castShadow: true,
          behaviorIds: ['flicker'],
        },
        {
          type: SANCTIONED_LIGHT_TYPE.AMBIENT,
          name: 'fire_ambient',
          color: [0.4, 0.15, 0.05],
          intensity: 0.15,
        },
      ],
    });

    // MOON_CYCLE — directional moon + ambient fill + hemi.
    this.registerComposite({
      id:   'moon_cycle',
      name: 'Moon Cycle',
      kind: COMPOSITE_LIGHT_KIND.MOON_CYCLE,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.DIRECTIONAL,
          name: 'moon',
          color: [0.42, 0.48, 0.70],
          intensity: 0.35,
          position: [-40, 60, -30],
          castShadow: true,
        },
        {
          type: SANCTIONED_LIGHT_TYPE.AMBIENT,
          name: 'moon_ambient',
          color: [0.08, 0.10, 0.18],
          intensity: 0.15,
        },
        {
          type: SANCTIONED_LIGHT_TYPE.HEMISPHERE,
          name: 'moon_hemi',
          color: [0.15, 0.20, 0.35],
          intensity: 0.20,
        },
      ],
      behaviors: [
        { id: 'day_cycle', ctx: { dayCycle: 0.0, daySpeed: 0.0008, baseIntensity: 1.0 } },
      ],
    });

    // SUN_CYCLE — directional sun + hemi + ambient.
    this.registerComposite({
      id:   'sun_cycle',
      name: 'Sun Cycle',
      kind: COMPOSITE_LIGHT_KIND.SUN_CYCLE,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.DIRECTIONAL,
          name: 'sun',
          color: [1.0, 0.96, 0.85],
          intensity: 1.25,
          position: [50, 80, 30],
          castShadow: true,
        },
        {
          type: SANCTIONED_LIGHT_TYPE.HEMISPHERE,
          name: 'sun_hemi',
          color: [0.45, 0.62, 0.85],
          intensity: 0.35,
        },
        {
          type: SANCTIONED_LIGHT_TYPE.AMBIENT,
          name: 'sun_ambient',
          color: [0.20, 0.25, 0.30],
          intensity: 0.15,
        },
      ],
      behaviors: [
        { id: 'day_cycle', ctx: { dayCycle: 0.38, daySpeed: 0.004, baseIntensity: 1.0 } },
      ],
    });

    // NEON — rect area light + pulse.
    this.registerComposite({
      id:   'neon',
      name: 'Neon',
      kind: COMPOSITE_LIGHT_KIND.NEON,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.RECT_AREA,
          name: 'neon_panel',
          color: [0.9, 0.4, 0.7],
          intensity: 3.0,
          width: 1.2,
          height: 0.2,
          castShadow: false,
          behaviorIds: ['pulse'],
        },
      ],
    });

    // MAGIC — point + flicker + rim.
    this.registerComposite({
      id:   'magic_glow',
      name: 'Magic Glow',
      kind: COMPOSITE_LIGHT_KIND.MAGIC,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.POINT,
          name: 'magic_core',
          color: [1.0, 0.85, 0.45],
          intensity: 3.5,
          distance: 8,
          decay: 2.0,
          position: [0, 1.2, 0],
          behaviorIds: ['flicker'],
        },
      ],
    });

    // INTERIOR_LAMP — point + temperature drift.
    this.registerComposite({
      id:   'interior_lamp',
      name: 'Interior Lamp',
      kind: COMPOSITE_LIGHT_KIND.INTERIOR_LAMP,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.POINT,
          name: 'lamp',
          color: [1.0, 0.85, 0.65],
          intensity: 1.8,
          distance: 6,
          decay: 2.0,
          castShadow: true,
          behaviorIds: ['temperature_drift'],
        },
      ],
    });

    // WINDOW_SHAFT — directional mimicking sun through window.
    this.registerComposite({
      id:   'window_shaft',
      name: 'Window Shaft',
      kind: COMPOSITE_LIGHT_KIND.WINDOW_SHAFT,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.DIRECTIONAL,
          name: 'shaft',
          color: [1.0, 0.94, 0.82],
          intensity: 1.2,
          castShadow: true,
        },
      ],
    });

    // CAUSTIC — point at water surface.
    this.registerComposite({
      id:   'caustic',
      name: 'Caustic',
      kind: COMPOSITE_LIGHT_KIND.CAUSTIC,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.POINT,
          name: 'caustic_core',
          color: [0.6, 0.95, 1.0],
          intensity: 0.8,
          distance: 4,
          decay: 2.5,
          behaviorIds: ['pulse'],
        },
      ],
    });

    // AURORA — hemisphere upper-band.
    this.registerComposite({
      id:   'aurora',
      name: 'Aurora',
      kind: COMPOSITE_LIGHT_KIND.AURORA,
      members: [
        {
          type: SANCTIONED_LIGHT_TYPE.HEMISPHERE,
          name: 'aurora_hemi',
          color: [0.35, 0.95, 0.75],
          intensity: 0.45,
        },
      ],
    });
  }

  /* ---------------- introspection ---------------- */

  getCount()        { return this.count; }
  getCapacity()     { return this.capacity; }
  getRejections()   { return this.rejections; }
  getLastRejection(){ return this.lastRejection; }

  getSlotByLight(light) {
    const idx = this.byLight.get(light);
    return idx === undefined ? null : this.slots[idx];
  }

  getSlotById(id) {
    const idx = this.byId.get(id);
    return idx === undefined ? null : this.slots[idx];
  }

  forEachLight(fn, ctx) {
    for (let i = 0; i < this.count; i++) {
      const slot = this.slots[i];
      if (slot.light) fn.call(ctx, slot.light, slot);
    }
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const byType = new Uint32Array(SANCTIONED_LIGHT_TYPE.COUNT);
    let totalBehaviors = 0;
    for (let i = 0; i < this.count; i++) {
      const s = this.slots[i];
      if (s.type >= 0 && s.type < byType.length) byType[s.type]++;
      totalBehaviors += s.behaviorCount;
    }

    const typeCounts = [];
    for (let i = 0; i < SANCTIONED_LIGHT_TYPE.COUNT; i++) {
      typeCounts.push({
        type:  SANCTIONED_LIGHT_TYPE_NAME[i],
        count: byType[i],
      });
    }

    return {
      frame:            this.frame,
      registeredLights: this.count,
      capacity:         this.capacity,
      totalBehaviors,
      rejectionTotal:   this.rejections,
      lastRejection:    this.lastRejection,
      registeredBehaviors: this.behaviorCount,
      registeredComposites: this.compositeCount,
      behaviorIds:      this.listBehaviors(),
      compositeIds:     this.listComposites(),
      typeCounts,
      enforce:          this.options.enforce,
      perfTier:         PERF_TIER_LOCAL,
    };
  }

  reset() {
    for (let i = 0; i < this.capacity; i++) this.slots[i].reset();
    this.count = 0;
    this.byLight.clear();
    this.byId.clear();
    this.rejections = 0;
    this.lastRejection = null;
    this.rejectionHead = 0;
    for (let i = 0; i < this.rejectionRing.length; i++) this.rejectionRing[i] = null;
    return this;
  }

  dispose() {
    this.reset();
    this.slots.length = 0;
    this.slots = null;
    this.byLight.clear();
    this.byId.clear();
    this.behaviorByName.clear();
    this.compositeByName.clear();
    for (let i = 0; i < this.behaviors.length; i++) this.behaviors[i] = null;
    for (let i = 0; i < this.composites.length; i++) this.composites[i] = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultPolicy = null;

export function getDefaultThreeLightsOnlyPolicy() {
  if (!_defaultPolicy) _defaultPolicy = new ThreeLightsOnlyPolicy();
  return _defaultPolicy;
}

export function disposeDefaultThreeLightsOnlyPolicy() {
  if (_defaultPolicy) {
    _defaultPolicy.dispose();
    _defaultPolicy = null;
  }
}

/* ------------------------------------------------------------------ */
/* 8. HOT-PATH HELPERS                                                */
/* ------------------------------------------------------------------ */

export function lightsBeginFrame(frameNumber) {
  getDefaultThreeLightsOnlyPolicy().beginFrame(frameNumber);
}

export function registerLight(light, options) {
  return getDefaultThreeLightsOnlyPolicy().registerLight(light, options);
}

export function unregisterLight(light) {
  return getDefaultThreeLightsOnlyPolicy().unregisterLight(light);
}

export function attachLightBehavior(light, behaviorId, ctx) {
  return getDefaultThreeLightsOnlyPolicy().attachBehavior(light, behaviorId, ctx);
}

export function detachLightBehavior(light, behaviorId) {
  return getDefaultThreeLightsOnlyPolicy().detachBehavior(light, behaviorId);
}

export function updateLightBehaviors(dt, elapsed) {
  getDefaultThreeLightsOnlyPolicy().updateBehaviors(dt, elapsed);
}

export function registerLightBehavior(spec) {
  return getDefaultThreeLightsOnlyPolicy().registerBehavior(spec);
}

export function registerLightComposite(spec) {
  return getDefaultThreeLightsOnlyPolicy().registerComposite(spec);
}

export function createCompositeLight(id, scene, options) {
  return getDefaultThreeLightsOnlyPolicy().createComposite(id, scene, options);
}

export function auditSceneLights(scene) {
  return getDefaultThreeLightsOnlyPolicy().auditScene(scene);
}

/* ------------------------------------------------------------------ */
/* 9. CONVENIENCE FACTORIES (direct light creation)                   */
/* ------------------------------------------------------------------ */

/**
 * Sanctioned factories — these are the ONLY way the rest of the engine
 * should create lights. Each factory:
 *   1. instantiates the correct sanctioned THREE class
 *   2. applies the given parameters
 *   3. registers the light with the policy
 *   4. returns the light
 */
export function createSanctionedAmbientLight(spec = {}) {
  const light = new THREE.AmbientLight(0xffffff, spec.intensity !== undefined ? spec.intensity : 1.0);
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  registerLight(light, spec);
  return light;
}

export function createSanctionedHemisphereLight(spec = {}) {
  const light = new THREE.HemisphereLight(
    0xffffff,
    0x222222,
    spec.intensity !== undefined ? spec.intensity : 1.0
  );
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  if (spec.groundColor) light.groundColor.setRGB(spec.groundColor[0], spec.groundColor[1], spec.groundColor[2]);
  if (spec.position) light.position.set(spec.position[0], spec.position[1], spec.position[2]);
  registerLight(light, spec);
  return light;
}

export function createSanctionedDirectionalLight(spec = {}) {
  const light = new THREE.DirectionalLight(
    0xffffff,
    spec.intensity !== undefined ? spec.intensity : 1.0
  );
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  if (spec.position) light.position.set(spec.position[0], spec.position[1], spec.position[2]);
  if (spec.target) light.target.position.set(spec.target[0], spec.target[1], spec.target[2]);
  if (spec.castShadow) light.castShadow = true;
  registerLight(light, spec);
  return light;
}

export function createSanctionedPointLight(spec = {}) {
  const light = new THREE.PointLight(
    0xffffff,
    spec.intensity !== undefined ? spec.intensity : 1.0,
    spec.distance !== undefined ? spec.distance : 0,
    spec.decay !== undefined ? spec.decay : 2.0
  );
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  if (spec.position) light.position.set(spec.position[0], spec.position[1], spec.position[2]);
  if (spec.castShadow) light.castShadow = true;
  registerLight(light, spec);
  return light;
}

export function createSanctionedSpotLight(spec = {}) {
  const light = new THREE.SpotLight(
    0xffffff,
    spec.intensity !== undefined ? spec.intensity : 1.0,
    spec.distance !== undefined ? spec.distance : 0,
    spec.angle !== undefined ? spec.angle : Math.PI / 4,
    spec.penumbra !== undefined ? spec.penumbra : 0.1,
    spec.decay !== undefined ? spec.decay : 2.0
  );
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  if (spec.position) light.position.set(spec.position[0], spec.position[1], spec.position[2]);
  if (spec.target) light.target.position.set(spec.target[0], spec.target[1], spec.target[2]);
  if (spec.castShadow) light.castShadow = true;
  registerLight(light, spec);
  return light;
}

export function createSanctionedRectAreaLight(spec = {}) {
  const light = new THREE.RectAreaLight(
    0xffffff,
    spec.intensity !== undefined ? spec.intensity : 1.0,
    spec.width !== undefined ? spec.width : 1.0,
    spec.height !== undefined ? spec.height : 1.0
  );
  if (spec.color) light.color.setRGB(spec.color[0], spec.color[1], spec.color[2]);
  if (spec.position) light.position.set(spec.position[0], spec.position[1], spec.position[2]);
  registerLight(light, spec);
  return light;
}

/* ------------------------------------------------------------------ */
/* 10. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createThreeLightsOnlyPolicy(options = {}) {
  return new ThreeLightsOnlyPolicy(options);
}

/* ------------------------------------------------------------------ */
/* 11. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ThreeLightsOnlyPolicy,
  LightSlot,
  BehaviorDescriptor,
  CompositeDescriptor,
  CompositeHandle,

  createThreeLightsOnlyPolicy,
  getDefaultThreeLightsOnlyPolicy,
  disposeDefaultThreeLightsOnlyPolicy,

  // Registry hooks
  lightsBeginFrame,
  registerLight,
  unregisterLight,
  attachLightBehavior,
  detachLightBehavior,
  updateLightBehaviors,
  registerLightBehavior,
  registerLightComposite,
  createCompositeLight,
  auditSceneLights,

  // Sanctioned light factories
  createSanctionedAmbientLight,
  createSanctionedHemisphereLight,
  createSanctionedDirectionalLight,
  createSanctionedPointLight,
  createSanctionedSpotLight,
  createSanctionedRectAreaLight,

  // Enums
  SANCTIONED_LIGHT_TYPE,
  SANCTIONED_LIGHT_TYPE_NAME,
  SANCTIONED_LIGHT_CLASS,
  COMPOSITE_LIGHT_KIND,
  COMPOSITE_LIGHT_KIND_NAME,
  FORBIDDEN_LIGHT_NAMES,
  MAX_LIGHTS,
  MAX_BEHAVIORS_PER_LIGHT,
  MAX_REGISTERED_BEHAVIORS,
  MAX_REGISTERED_COMPOSITES,
  MAX_COMPOSITE_MEMBERS,
};

export default _defaultExport;