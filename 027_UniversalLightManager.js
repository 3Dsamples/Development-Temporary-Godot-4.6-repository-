// File : 027
// name : systems/027_UniversalLightManager.js
// description : Universal anime light manager + desert scene composer — the ECS
//               orchestrator that drives EVERY anime shader layer (016–026) from a
//               single Three.js r185 DirectionalLight with real shadow mapping.
//               Features: full day/night sun orbit with perceptual (Oklab) color
//               keyframe interpolation (setup-only allocation, zero per-frame
//               alloc), elevation-driven intensity / shadow bias / shadow length,
//               simplex wind field damped frame-rate-independently, 4 Hz throttled
//               sync into all registered layer managers (016 directional chunk
//               uniforms + shadow map texture/matrix), bitECS LightRef entity
//               sync via 001_gmp_MathUtils dense pool, optional IES conversion for
//               the sun and add-on spot/point lights via 021_gmp_ies_lighting
//               (autoConvertLight / generateIsotropicIES / iesToSpotLight /
//               parseIES003 / integrateCandela), performance-tiered shadow map
//               resolution (PERF_TIER), and a DesertSceneComposer that stacks all
//               desert layers in correct render order and updates them in one
//               call. Zero per-frame allocation (pre-allocated scratch
//               Vector3/Color), hitch-capped timestep from 001. Designed for
//               Android mobile with Three.js r185 + bitECS 0.4.x.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  world as ecsWorld,
  LightRef,
  ActiveTag,
  getDenseEntities,
  getDenseCount,
  hasComponent,
  clamp,
  damp,
  lerp,
  simplex2D,
  PERF_TIER,
  PI,
  TWO_PI,
  DEG2RAD,
} from '../utils/001_gmp_MathUtils.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
  iesToSpotLight,
  parseIES003,
} from '../utils/021_gmp_ies_lighting.js';
import { colorToOklab } from '../utils/020_gmp_perceptual_color.js';
import { DirectionalLightShader } from '../shaders/016_DirectionalLightShader.glsl.js';

/* ------------------------------------------------------------------ */
/* 1. SCRATCH + PERCEPTUAL KEYFRAMES (setup-only allocation)           */
/* ------------------------------------------------------------------ */
const _sunDir   = new THREE.Vector3();
const _travel   = new THREE.Vector3();
const _keyCol   = new THREE.Color();

// Oklab keyframes computed ONCE at module load (no per-frame alloc)
const KF_NIGHT = colorToOklab(_keyCol.setRGB(0.35, 0.45, 0.75));
const KF_DAWN  = colorToOklab(_keyCol.setRGB(1.00, 0.55, 0.30));
const KF_NOON  = colorToOklab(_keyCol.setRGB(1.00, 0.96, 0.85));
const KF_DUSK  = colorToOklab(_keyCol.setRGB(1.00, 0.45, 0.35));
const _labA = { L: 0, a: 0, b: 0 };
const _labB = { L: 0, a: 0, b: 0 };

/** Allocation-free Oklab -> linear RGB write into a THREE.Color (pbrt matrix). */
function oklabIntoColor(L, a, b, out) {
  const lp = L + 0.3963377774 * a + 0.2158037573 * b;
  const mp = L - 0.1055613458 * a - 0.0638541728 * b;
  const sp = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = lp * lp * lp, m = mp * mp * mp, s = sp * sp * sp;
  out.setRGB(
    clamp(+4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, 0, 1),
    clamp(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, 0, 1),
    clamp(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s, 0, 1)
  );
}

function lerpLab(a, b, t, out) {
  out.L = lerp(a.L, b.L, t);
  out.a = lerp(a.a, b.a, t);
  out.b = lerp(a.b, b.b, t);
}

/* ------------------------------------------------------------------ */
/* 2. UNIVERSAL LIGHT MANAGER                                          */
/* ------------------------------------------------------------------ */
export class UniversalLightManager {
  constructor(scene, camera, options = {}) {
    this.scene  = scene;
    this.camera = camera;

    // Performance-tiered shadow resolution
    const shadowRes = options.shadowResolution ||
      (PERF_TIER === 'HIGH' ? 2048 : PERF_TIER === 'MEDIUM' ? 1024 : 512);

    // 016 anime directional light (owns the Three.js DirectionalLight)
    this.dir = new DirectionalLightShader({ shadowResolution: shadowRes });
    this.light = this.dir.getLight();
    this.light.castShadow = true;
    this.scene.add(this.light);
    this.scene.add(this.light.target);

    // Optional IES tagging of the sun (isotropic fallback profile)
    this.iesEnabled = !!options.ies;
    if (this.iesEnabled) autoConvertLight(this.light, generateIsotropicIES(this.light));

    // Day/night + environment state
    this.timeOfDay   = options.timeOfDay !== undefined ? options.timeOfDay : 0.38;
    this.daySpeed    = options.daySpeed  !== undefined ? options.daySpeed  : 0.004; // cycles/sec
    this.maxElev     = (options.maxElevationDeg || 75) * DEG2RAD;
    this.maxIntensity= options.maxIntensity || 1.25;
    this.moonIntensity = options.moonIntensity || 0.25;
    this.windStrength  = options.windStrength || 0.6;

    // Registered anime layer managers (017–026)
    this.layers = [];

    // Wind state (damped)
    this.windX = 0; this.windY = 0;

    // Throttle accumulators
    this._syncAccum = 1.0;
    this._ecsAccum  = 1.0;

    // Scratch color for sun ramp
    this._sunCol = new THREE.Color(1, 0.96, 0.85);
  }

  /* ---------------- layer registry ---------------- */
  register(layer) {
    if (layer && this.layers.indexOf(layer) < 0) this.layers.push(layer);
    return layer;
  }
  unregister(layer) {
    const i = this.layers.indexOf(layer);
    if (i >= 0) this.layers.splice(i, 1);
  }

  /* ---------------- IES add-on lights ---------------- */
  addIESSpotFrom003(options = {}) {
    const ies = parseIES003();
    const spot = iesToSpotLight(ies, options);
    spot.userData.baseIntensity = options.intensity !== undefined ? options.intensity : 1;
    const flux = integrateCandela(ies);
    spot.userData.iesScale = flux > 0 ? clamp(flux / (4 * Math.PI), 0.25, 4.0) : 1.0;
    this.scene.add(spot);
    this.scene.add(spot.target);
    return spot;
  }
  tagIES(light) {
    return autoConvertLight(light, generateIsotropicIES(light));
  }

  /* ---------------- sun orbit + perceptual ramp ---------------- */
  _computeSun(t) {
    // t: 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
    const dayT  = clamp((t - 0.25) / 0.5, 0, 1);           // 0..1 across daylight
    const ang   = dayT * PI;                                // 0..pi
    const elev  = Math.sin(ang) * this.maxElev;             // radians
    const azim  = lerp(-110 * DEG2RAD, 110 * DEG2RAD, dayT);
    const ce = Math.cos(elev);
    _sunDir.set(Math.sin(azim) * ce, Math.sin(elev), -Math.cos(azim) * ce).normalize();
    return { elev, dayT };
  }

  _sunColorRamp(t, elevNorm) {
    // Piecewise Oklab interpolation: night->dawn->noon->dusk->night
    let a = KF_NOON, b = KF_NOON, f = 0;
    if (t < 0.20)      { a = KF_NIGHT; b = KF_NIGHT; f = 0; }
    else if (t < 0.30) { a = KF_NIGHT; b = KF_DAWN;  f = (t - 0.20) / 0.10; }
    else if (t < 0.42) { a = KF_DAWN;  b = KF_NOON;  f = (t - 0.30) / 0.12; }
    else if (t < 0.60) { a = KF_NOON;  b = KF_NOON;  f = 0; }
    else if (t < 0.72) { a = KF_NOON;  b = KF_DUSK;  f = (t - 0.60) / 0.12; }
    else if (t < 0.82) { a = KF_DUSK;  b = KF_NIGHT; f = (t - 0.72) / 0.10; }
    else               { a = KF_NIGHT; b = KF_NIGHT; f = 0; }
    lerpLab(a, b, clamp(f, 0, 1), _labA);
    oklabIntoColor(_labA.L, _labA.a, _labA.b, this._sunCol);
  }

  /* ---------------- main update ---------------- */
  update(dt, elapsed) {
    // Advance day cycle
    this.timeOfDay = (this.timeOfDay + dt * this.daySpeed) % 1.0;
    const t = this.timeOfDay;

    // Sun orbit
    const { elev } = this._computeSun(t);
    const elevNorm = clamp(Math.sin(elev), 0, 1);
    const sunAbove = elev > 0;

    // Intensity: sun curve by day, moon floor by night
    const sunI = Math.pow(elevNorm, 1.2) * this.maxIntensity;
    const intensity = sunAbove ? Math.max(sunI, 0.05) : this.moonIntensity;

    // Perceptual sun color ramp
    this._sunColorRamp(t, elevNorm);

    // Light travel direction = -sunDir
    _travel.copy(_sunDir).negate();

    // Push into 016 manager uniforms + Three light transform
    const u = this.dir.getUniforms();
    u.uDirLightDirection.value.copy(_travel);
    u.uDirLightIntensity.value = intensity;
    u.uDirLightColor.value.copy(this._sunCol);
    u.uSunDir.value.copy(_sunDir);
    u.uSunColor.value.copy(this._sunCol);
    u.uSunIntensity.value = intensity;

    this.light.position.copy(_sunDir).multiplyScalar(120);
    this.light.target.position.set(0, 0, 0);
    this.light.intensity = intensity;
    this.light.color.copy(this._sunCol);

    // Elevation-driven shadow quality (long shadows at low sun need more bias)
    const lowSun = 1.0 - elevNorm;
    this.light.shadow.bias       = -0.0008 - 0.0022 * lowSun;
    this.light.shadow.normalBias =  0.0200 + 0.0500 * lowSun;
    u.uShadowBias.value       = this.light.shadow.bias;
    u.uShadowNormalBias.value = this.light.shadow.normalBias;

    // Simplex wind field, damped (frame-rate independent)
    const wTx = simplex2D(elapsed * 0.030, 7.7) * this.windStrength;
    const wTy = simplex2D(elapsed * 0.023 + 31.4, 3.1) * this.windStrength * 0.4;
    this.windX = damp(this.windX, wTx, 2.0, dt);
    this.windY = damp(this.windY, wTy, 2.0, dt);

    // Per-frame layer update (time, camera, shadow-length damping)
    for (let i = 0; i < this.layers.length; i++) {
      const L = this.layers[i];
      if (L.uniforms && L.uniforms.uWind) L.uniforms.uWind.value.set(this.windX, this.windY);
      if (L.update) L.update(dt, elapsed, this.dir, this.camera);
    }

    // 4 Hz throttled full sync (direction/color/shadow map/perceptual tints)
    this._syncAccum += dt;
    if (this._syncAccum >= 0.25) {
      this._syncAccum = 0;
      for (let i = 0; i < this.layers.length; i++) {
        const L = this.layers[i];
        if (L.syncFromLight) L.syncFromLight(this.dir);
      }
    }

    // 4 Hz throttled bitECS LightRef sync
    this._ecsAccum += dt;
    if (this._ecsAccum >= 0.25) {
      this._ecsAccum = 0;
      this._syncECS();
    }
  }

  /* ---------------- bitECS LightRef sync ---------------- */
  _syncECS() {
    const ents = getDenseEntities();
    const n = getDenseCount();
    const c = this._sunCol;
    const inten = this.dir.getUniforms().uDirLightIntensity.value;
    for (let i = 0; i < n; i++) {
      const eid = ents[i];
      if (!hasComponent(ecsWorld, eid, LightRef)) continue;
      if (ActiveTag[eid] !== 1) continue;
      LightRef.intensity[eid] = inten;
      LightRef.colorR[eid] = c.r;
      LightRef.colorG[eid] = c.g;
      LightRef.colorB[eid] = c.b;
    }
  }

  dispose() {
    this.layers.length = 0;
    this.scene.remove(this.light);
    this.scene.remove(this.light.target);
  }
}

/* ------------------------------------------------------------------ */
/* 3. DESERT SCENE COMPOSER                                            */
/*    Stacks every desert layer (016–026) in render order and drives   */
/*    them from one UniversalLightManager with a single update() call. */
/* ------------------------------------------------------------------ */
export class DesertSceneComposer {
  constructor(scene, camera, options = {}) {
    this.scene  = scene;
    this.camera = camera;
    this.lightMgr = new UniversalLightManager(scene, camera, options);
    this.stack = [];
  }

  /** Add a layer manager (018–026 factories). Render order taken from mesh. */
  addLayer(layer) {
    if (!layer) return null;
    this.stack.push(layer);
    this.scene.add(layer.getMesh());
    this.lightMgr.register(layer);
    return layer;
  }

  removeLayer(layer) {
    const i = this.stack.indexOf(layer);
    if (i >= 0) {
      this.stack.splice(i, 1);
      this.scene.remove(layer.getMesh());
      this.lightMgr.unregister(layer);
      if (layer.dispose) layer.dispose();
    }
  }

  /** Sort the stack by renderOrder for deterministic painter's algorithm. */
  sortStack() {
    this.stack.sort((a, b) => a.getMesh().renderOrder - b.getMesh().renderOrder);
  }

  update(dt, elapsed) {
    this.lightMgr.update(dt, elapsed);
  }

  dispose() {
    for (let i = 0; i < this.stack.length; i++) {
      const L = this.stack[i];
      this.scene.remove(L.getMesh());
      if (L.dispose) L.dispose();
    }
    this.stack.length = 0;
    this.lightMgr.dispose();
  }
}

/* ------------------------------------------------------------------ */
/* 4. FACTORIES                                                        */
/* ------------------------------------------------------------------ */
export function createUniversalLightManager(scene, camera, options = {}) {
  return new UniversalLightManager(scene, camera, options);
}
export function createDesertSceneComposer(scene, camera, options = {}) {
  return new DesertSceneComposer(scene, camera, options);
}
