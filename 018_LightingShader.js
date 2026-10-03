// File : 18
// name : shaders/LightingShader.js
// description : Cel-shaded anime lighting shader chunks and light entity management. Provides GLSL injection blocks for toon diffuse, specular, rim, hemisphere ambient, atmospheric god rays, and sky gradient lighting. Uses bitECS SoA components for light entities with fixed capacity, drives sun/moon direction and color from day cycle, exposes reusable GLSL_CHUNK strings for other materials. Mobile-optimized with branchless toon bands and single directional light plus hemisphere ambient.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  addComponent,
  removeComponent,
  hasComponent,
  entityExists
} from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

import {
  world,
  acquireEntity,
  releaseEntity,
  clamp,
  mix,
  smoothstep,
  MAX_ENTITIES,
  PERF_TIER
} from '../utils/001_MathUtils.js';

// ---------------------------------------------------------------------------
// LIGHT TYPE IDS
// ---------------------------------------------------------------------------
export const LIGHT_TYPE_SUN = 0;
export const LIGHT_TYPE_MOON = 1;
export const LIGHT_TYPE_SKY = 2;
export const LIGHT_TYPE_RIM = 3;
export const LIGHT_TYPE_BOUNCE = 4;
export const LIGHT_TYPE_POINT = 5;

// ---------------------------------------------------------------------------
// LIGHT COMPONENT CAPACITY
// ---------------------------------------------------------------------------
const MAX_LIGHTS = 64;
const TWO_PI_LOCAL = Math.PI * 2;
const HALF_PI_LOCAL = Math.PI * 0.5;

// ---------------------------------------------------------------------------
// GLSL — TOON DIFFUSE
// ---------------------------------------------------------------------------
export const GLSL_TOON_DIFFUSE = `
float toonDiffuse(float ndotl, float b1, float b2, float b3) {
  float n = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
  float band1 = step(b1, n);
  float band2 = step(b2, n);
  float band3 = step(b3, n);
  return 0.18 + band1 * 0.27 + band2 * 0.30 + band3 * 0.25;
}

float toonDiffuseSoft(float ndotl, float b1, float b2, float b3, float softness) {
  float n = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
  float band1 = smoothstep(b1 - softness, b1 + softness, n);
  float band2 = smoothstep(b2 - softness, b2 + softness, n);
  float band3 = smoothstep(b3 - softness, b3 + softness, n);
  return 0.18 + band1 * 0.27 + band2 * 0.30 + band3 * 0.25;
}
`;

// ---------------------------------------------------------------------------
// GLSL — TOON SPECULAR
// ---------------------------------------------------------------------------
export const GLSL_TOON_SPECULAR = `
float toonSpecular(vec3 N, vec3 V, vec3 L, float power, float strength, float cutoff) {
  vec3 H = normalize(L + V);
  float ndoth = max(dot(N, H), 0.0);
  float spec = pow(ndoth, power);
  spec = smoothstep(cutoff - 0.05, cutoff + 0.02, spec);
  return spec * strength;
}

float toonSpecularAniso(vec3 N, vec3 V, vec3 L, vec3 T, float power, float strength, float aniso) {
  vec3 H = normalize(L + V);
  float ndoth = max(dot(N, H), 0.0);
  float tdh = dot(T, H);
  float anisoFactor = 1.0 - aniso * tdh * tdh;
  float spec = pow(ndoth * anisoFactor, power);
  spec = smoothstep(0.5, 0.55, spec);
  return spec * strength;
}
`;

// ---------------------------------------------------------------------------
// GLSL — RIM LIGHT
// ---------------------------------------------------------------------------
export const GLSL_RIM_LIGHT = `
float rimFactor(vec3 N, vec3 V, float power) {
  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, power);
  return smoothstep(0.3, 0.75, rim);
}

vec3 applyRim(vec3 baseColor, float rimFactor, vec3 rimColor, float strength) {
  return baseColor + rimColor * rimFactor * strength;
}
`;

// ---------------------------------------------------------------------------
// GLSL — HEMISPHERE AMBIENT
// ---------------------------------------------------------------------------
export const GLSL_HEMI_AMBIENT = `
vec3 hemiAmbient(vec3 N, vec3 skyColor, vec3 groundColor, float skyStrength, float groundStrength) {
  float upFactor = N.y * 0.5 + 0.5;
  vec3 sky = skyColor * skyStrength * upFactor;
  vec3 ground = groundColor * groundStrength * (1.0 - upFactor);
  return sky + ground;
}
`;

// ---------------------------------------------------------------------------
// GLSL — ATMOSPHERIC FOG
// ---------------------------------------------------------------------------
export const GLSL_ATMOSPHERIC = `
vec3 atmosphericFog(vec3 color, float depth, vec3 fogColor, float near, float far, float density) {
  float f = smoothstep(near, far, depth);
  f = 1.0 - exp(-f * density);
  return mix(color, fogColor, f);
}

float godRayFactor(vec3 worldPos, vec3 sunDir, vec3 viewDir, float intensity) {
  vec3 sunScreen = normalize(sunDir);
  vec3 viewDirNorm = normalize(viewDir);
  float align = max(dot(viewDirNorm, -sunScreen), 0.0);
  align = pow(align, 8.0);
  return align * intensity;
}

vec3 godRayColor(vec3 baseColor, float ray, vec3 sunColor, float strength) {
  return baseColor + sunColor * ray * strength;
}
`;

// ---------------------------------------------------------------------------
// GLSL — SHADOW TINT
// ---------------------------------------------------------------------------
export const GLSL_SHADOW_TINT = `
vec3 shadowTint(vec3 litColor, float shadowMask, vec3 tint, float strength) {
  vec3 tinted = litColor * tint * 1.4;
  return mix(litColor, tinted, shadowMask * strength);
}
`;

// ---------------------------------------------------------------------------
// GLSL — FULL LIGHTING PIPELINE
// ---------------------------------------------------------------------------
export const GLSL_FULL_LIGHTING = `
struct ToonLightResult {
  vec3 color;
  float luminance;
  float shadowMask;
};

ToonLightResult computeToonLighting(
  vec3 N,
  vec3 V,
  vec3 L,
  vec3 baseColor,
  vec3 sunColor,
  vec3 skyColor,
  vec3 groundColor,
  vec3 rimColor,
  float band1,
  float band2,
  float band3,
  float skyStrength,
  float groundStrength,
  float rimPower,
  float rimStrength,
  float shadowStrength,
  vec3 shadowTintColor
) {
  ToonLightResult result;

  float ndotl = dot(N, L);
  float toon = toonDiffuse(ndotl, band1, band2, band3);

  vec3 ambient = hemiAmbient(N, skyColor, groundColor, skyStrength, groundStrength);

  vec3 lit = baseColor * sunColor * toon + baseColor * ambient;

  float shadowMask = 1.0 - toon;
  lit = shadowTint(lit, shadowMask, shadowTintColor, shadowStrength);

  float rim = rimFactor(N, V, rimPower);
  lit = applyRim(lit, rim, rimColor, rimStrength);

  result.color = lit;
  result.luminance = dot(lit, vec3(0.299, 0.587, 0.114));
  result.shadowMask = shadowMask;

  return result;
}
`;

// ---------------------------------------------------------------------------
// GLSL — LIGHT UNIFORMS DECLARATION
// ---------------------------------------------------------------------------
export const GLSL_LIGHT_UNIFORMS = `
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uMoonDir;
uniform vec3 uMoonColor;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform vec3 uBounceColor;
uniform vec3 uRimColor;
uniform vec3 uFogColor;
uniform vec3 uShadowTintColor;

uniform float uSunIntensity;
uniform float uMoonIntensity;
uniform float uSkyStrength;
uniform float uGroundStrength;
uniform float uBounceStrength;
uniform float uRimStrength;
uniform float uRimPower;
uniform float uFogNear;
uniform float uFogFar;
uniform float uFogDensity;
uniform float uShadowStrength;
uniform float uBand1;
uniform float uBand2;
uniform float uBand3;
uniform float uSpecPower;
uniform float uSpecStrength;
uniform vec3 uSpecColor;
`;

// ---------------------------------------------------------------------------
// LIGHTING CONFIGURATION
// ---------------------------------------------------------------------------
export class LightingConfiguration {
  constructor() {
    this.sunDir = new THREE.Vector3(0.5, 0.8, 0.3).normalize();
    this.sunColor = new THREE.Color(1.0, 0.96, 0.85);
    this.sunIntensity = 1.0;

    this.moonDir = new THREE.Vector3(-0.4, 0.6, -0.7).normalize();
    this.moonColor = new THREE.Color(0.42, 0.48, 0.70);
    this.moonIntensity = 0.35;

    this.skyColor = new THREE.Color(0.45, 0.62, 0.85);
    this.groundColor = new THREE.Color(0.18, 0.22, 0.26);
    this.bounceColor = new THREE.Color(0.15, 0.20, 0.25);
    this.rimColor = new THREE.Color(0.90, 0.95, 1.00);
    this.fogColor = new THREE.Color(0.35, 0.55, 0.70);
    this.shadowTintColor = new THREE.Color(0.12, 0.18, 0.30);

    this.skyStrength = 0.35;
    this.groundStrength = 0.15;
    this.bounceStrength = 0.3;
    this.rimStrength = 0.55;
    this.rimPower = 3.0;

    this.fogNear = 8.0;
    this.fogFar = 60.0;
    this.fogDensity = 1.0;

    this.shadowStrength = 0.65;

    this.band1 = 0.35;
    this.band2 = 0.55;
    this.band3 = 0.75;

    this.specPower = 32.0;
    this.specStrength = 0.4;
    this.specColor = new THREE.Color(1.0, 1.0, 1.0);

    this.ambientOcclusionStrength = 0.6;

    this.atmosphereTop = new THREE.Color(0.10, 0.30, 0.50);
    this.atmosphereBottom = new THREE.Color(0.02, 0.12, 0.25);
  }

  clone() {
    const c = new LightingConfiguration();
    c.sunDir.copy(this.sunDir);
    c.sunColor.copy(this.sunColor);
    c.sunIntensity = this.sunIntensity;
    c.moonDir.copy(this.moonDir);
    c.moonColor.copy(this.moonColor);
    c.moonIntensity = this.moonIntensity;
    c.skyColor.copy(this.skyColor);
    c.groundColor.copy(this.groundColor);
    c.bounceColor.copy(this.bounceColor);
    c.rimColor.copy(this.rimColor);
    c.fogColor.copy(this.fogColor);
    c.shadowTintColor.copy(this.shadowTintColor);
    c.skyStrength = this.skyStrength;
    c.groundStrength = this.groundStrength;
    c.bounceStrength = this.bounceStrength;
    c.rimStrength = this.rimStrength;
    c.rimPower = this.rimPower;
    c.fogNear = this.fogNear;
    c.fogFar = this.fogFar;
    c.fogDensity = this.fogDensity;
    c.shadowStrength = this.shadowStrength;
    c.band1 = this.band1;
    c.band2 = this.band2;
    c.band3 = this.band3;
    c.specPower = this.specPower;
    c.specStrength = this.specStrength;
    c.specColor.copy(this.specColor);
    c.ambientOcclusionStrength = this.ambientOcclusionStrength;
    c.atmosphereTop.copy(this.atmosphereTop);
    c.atmosphereBottom.copy(this.atmosphereBottom);
    return c;
  }
}

// ---------------------------------------------------------------------------
// LIGHTING SHADER — light entity manager + uniform writer
// ---------------------------------------------------------------------------
export class LightingShader {
  constructor(options = {}) {
    this.config = options.config || new LightingConfiguration();
    this.capacity = options.capacity || MAX_LIGHTS;

    this.lightEntities = new Int32Array(this.capacity).fill(-1);
    this.lightTypes = new Uint8Array(this.capacity);
    this.lightCount = 0;
    this.freeSlots = [];

    this.elapsed = 0.0;
    this.dayCycle = 0.5;

    this._sunVec = new THREE.Vector3();
    this._moonVec = new THREE.Vector3();

    this.registeredMaterials = [];
  }

  // -------------------------------------------------------------------------
  // SPAWN LIGHT ENTITY
  // -------------------------------------------------------------------------
  spawnLight(type, r, g, b, intensity) {
    let slot;
    if (this.freeSlots.length > 0) {
      slot = this.freeSlots.pop();
    } else if (this.lightCount < this.capacity) {
      slot = this.lightCount++;
    } else {
      return -1;
    }

    const eid = acquireEntity();

    world.components.LightRef.intensity[eid] = intensity;
    world.components.LightRef.colorR[eid] = r;
    world.components.LightRef.colorG[eid] = g;
    world.components.LightRef.colorB[eid] = b;

    world.components.Position.x[eid] = 0;
    world.components.Position.y[eid] = 0;
    world.components.Position.z[eid] = 0;

    addComponent(world, eid, world.components.LightRef);
    addComponent(world, eid, world.components.Position);

    this.lightEntities[slot] = eid;
    this.lightTypes[slot] = type;

    return eid;
  }

  removeLight(eid) {
    if (!entityExists(world, eid)) return;

    if (hasComponent(world, eid, world.components.LightRef)) {
      removeComponent(world, eid, world.components.LightRef);
    }
    if (hasComponent(world, eid, world.components.Position)) {
      removeComponent(world, eid, world.components.Position);
    }

    releaseEntity(eid);

    for (let i = 0; i < this.lightCount; i++) {
      if (this.lightEntities[i] === eid) {
        this.lightEntities[i] = -1;
        this.freeSlots.push(i);
        break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // REGISTER MATERIAL FOR AUTO-UPDATE
  // -------------------------------------------------------------------------
  registerMaterial(material) {
    if (!material || !material.uniforms) return;
    if (this.registeredMaterials.indexOf(material) >= 0) return;
    this.registeredMaterials.push(material);
    this.writeUniforms(material.uniforms);
  }

  unregisterMaterial(material) {
    const idx = this.registeredMaterials.indexOf(material);
    if (idx >= 0) this.registeredMaterials.splice(idx, 1);
  }

  // -------------------------------------------------------------------------
  // DAY CYCLE — update sun/moon direction and colors
  // -------------------------------------------------------------------------
  applyDayCycle(cycle) {
    this.dayCycle = cycle - Math.floor(cycle);

    const angle = this.dayCycle * TWO_PI_LOCAL - HALF_PI_LOCAL;
    this._sunVec.set(Math.cos(angle), Math.sin(angle), 0.35).normalize();
    this._moonVec.copy(this._sunVec).negate();

    this.config.sunDir.copy(this._sunVec);
    this.config.moonDir.copy(this._moonVec);

    const elevation = Math.sin(angle);
    const dayFactor = smoothstep(-0.2, 0.3, elevation);

    this.config.sunIntensity = mix(0.15, 1.0, dayFactor);
    this.config.moonIntensity = mix(0.45, 0.15, dayFactor);

    const sunR = mix(1.00, 1.00, dayFactor);
    const sunG = mix(0.65, 0.96, dayFactor);
    const sunB = mix(0.35, 0.85, dayFactor);
    this.config.sunColor.setRGB(sunR, sunG, sunB);

    const moonR = mix(0.55, 0.40, dayFactor);
    const moonG = mix(0.60, 0.45, dayFactor);
    const moonB = mix(0.85, 0.70, dayFactor);
    this.config.moonColor.setRGB(moonR, moonG, moonB);

    this.config.skyColor.setRGB(
      mix(0.08, 0.45, dayFactor),
      mix(0.10, 0.62, dayFactor),
      mix(0.22, 0.85, dayFactor)
    );

    this.config.fogColor.setRGB(
      mix(0.05, 0.35, dayFactor),
      mix(0.08, 0.55, dayFactor),
      mix(0.15, 0.70, dayFactor)
    );

    this.config.rimColor.setRGB(
      mix(0.55, 0.90, dayFactor),
      mix(0.65, 0.95, dayFactor),
      mix(0.90, 1.00, dayFactor)
    );

    return this;
  }

  // -------------------------------------------------------------------------
  // WRITE UNIFORMS INTO A SHADER MATERIAL
  // -------------------------------------------------------------------------
  writeUniforms(uniforms) {
    if (!uniforms) return;

    const c = this.config;

    if (uniforms.uSunDir) uniforms.uSunDir.value.copy(c.sunDir);
    if (uniforms.uSunColor) uniforms.uSunColor.value.copy(c.sunColor);
    if (uniforms.uSunIntensity) uniforms.uSunIntensity.value = c.sunIntensity;

    if (uniforms.uMoonDir) uniforms.uMoonDir.value.copy(c.moonDir);
    if (uniforms.uMoonColor) uniforms.uMoonColor.value.copy(c.moonColor);
    if (uniforms.uMoonIntensity) uniforms.uMoonIntensity.value = c.moonIntensity;

    if (uniforms.uSkyColor) uniforms.uSkyColor.value.copy(c.skyColor);
    if (uniforms.uGroundColor) uniforms.uGroundColor.value.copy(c.groundColor);
    if (uniforms.uBounceColor) uniforms.uBounceColor.value.copy(c.bounceColor);
    if (uniforms.uRimColor) uniforms.uRimColor.value.copy(c.rimColor);
    if (uniforms.uFogColor) uniforms.uFogColor.value.copy(c.fogColor);
    if (uniforms.uShadowTintColor) uniforms.uShadowTintColor.value.copy(c.shadowTintColor);

    if (uniforms.uSkyStrength) uniforms.uSkyStrength.value = c.skyStrength;
    if (uniforms.uGroundStrength) uniforms.uGroundStrength.value = c.groundStrength;
    if (uniforms.uBounceStrength) uniforms.uBounceStrength.value = c.bounceStrength;
    if (uniforms.uRimStrength) uniforms.uRimStrength.value = c.rimStrength;
    if (uniforms.uRimPower) uniforms.uRimPower.value = c.rimPower;

    if (uniforms.uFogNear) uniforms.uFogNear.value = c.fogNear;
    if (uniforms.uFogFar) uniforms.uFogFar.value = c.fogFar;
    if (uniforms.uFogDensity) uniforms.uFogDensity.value = c.fogDensity;

    if (uniforms.uShadowStrength) uniforms.uShadowStrength.value = c.shadowStrength;

    if (uniforms.uBand1) uniforms.uBand1.value = c.band1;
    if (uniforms.uBand2) uniforms.uBand2.value = c.band2;
    if (uniforms.uBand3) uniforms.uBand3.value = c.band3;

    if (uniforms.uSpecPower) uniforms.uSpecPower.value = c.specPower;
    if (uniforms.uSpecStrength) uniforms.uSpecStrength.value = c.specStrength;
    if (uniforms.uSpecColor) uniforms.uSpecColor.value.copy(c.specColor);

    if (uniforms.uAmbientOcclusionStrength) uniforms.uAmbientOcclusionStrength.value = c.ambientOcclusionStrength;
  }

  // -------------------------------------------------------------------------
  // UNIFORM FACTORY
  // -------------------------------------------------------------------------
  createUniforms() {
    const c = this.config;
    return {
      uSunDir: { value: c.sunDir.clone() },
      uSunColor: { value: c.sunColor.clone() },
      uSunIntensity: { value: c.sunIntensity },
      uMoonDir: { value: c.moonDir.clone() },
      uMoonColor: { value: c.moonColor.clone() },
      uMoonIntensity: { value: c.moonIntensity },
      uSkyColor: { value: c.skyColor.clone() },
      uGroundColor: { value: c.groundColor.clone() },
      uBounceColor: { value: c.bounceColor.clone() },
      uRimColor: { value: c.rimColor.clone() },
      uFogColor: { value: c.fogColor.clone() },
      uShadowTintColor: { value: c.shadowTintColor.clone() },
      uSkyStrength: { value: c.skyStrength },
      uGroundStrength: { value: c.groundStrength },
      uBounceStrength: { value: c.bounceStrength },
      uRimStrength: { value: c.rimStrength },
      uRimPower: { value: c.rimPower },
      uFogNear: { value: c.fogNear },
      uFogFar: { value: c.fogFar },
      uFogDensity: { value: c.fogDensity },
      uShadowStrength: { value: c.shadowStrength },
      uBand1: { value: c.band1 },
      uBand2: { value: c.band2 },
      uBand3: { value: c.band3 },
      uSpecPower: { value: c.specPower },
      uSpecStrength: { value: c.specStrength },
      uSpecColor: { value: c.specColor.clone() },
      uAmbientOcclusionStrength: { value: c.ambientOcclusionStrength }
    };
  }

  // -------------------------------------------------------------------------
  // FRAME UPDATE
  // -------------------------------------------------------------------------
  update(delta, elapsed) {
    this.elapsed = elapsed;

    for (let i = 0; i < this.registeredMaterials.length; i++) {
      const mat = this.registeredMaterials[i];
      if (mat && mat.uniforms) {
        this.writeUniforms(mat.uniforms);
      }
    }
  }

  // -------------------------------------------------------------------------
  // PRESET PHASE
  // -------------------------------------------------------------------------
  applyPhase(phase) {
    const c = this.config;
    switch (phase) {
      case 'dawn':
        c.skyColor.setRGB(0.60, 0.50, 0.70);
        c.rimColor.setRGB(1.00, 0.85, 0.70);
        c.fogColor.setRGB(0.55, 0.45, 0.55);
        c.rimStrength = 0.7;
        break;
      case 'noon':
        c.skyColor.setRGB(0.45, 0.62, 0.85);
        c.rimColor.setRGB(0.90, 0.95, 1.00);
        c.fogColor.setRGB(0.35, 0.55, 0.70);
        c.rimStrength = 0.45;
        break;
      case 'dusk':
        c.skyColor.setRGB(0.55, 0.35, 0.45);
        c.rimColor.setRGB(1.00, 0.70, 0.45);
        c.fogColor.setRGB(0.45, 0.35, 0.40);
        c.rimStrength = 0.65;
        break;
      case 'night':
        c.skyColor.setRGB(0.08, 0.10, 0.22);
        c.rimColor.setRGB(0.55, 0.65, 0.90);
        c.fogColor.setRGB(0.05, 0.08, 0.15);
        c.rimStrength = 0.35;
        break;
    }
    return this;
  }

  // -------------------------------------------------------------------------
  // STATS
  // -------------------------------------------------------------------------
  getStats() {
    return {
      lightCount: this.lightCount,
      capacity: this.capacity,
      freeSlots: this.freeSlots.length,
      sunIntensity: this.config.sunIntensity,
      moonIntensity: this.config.moonIntensity,
      dayCycle: this.dayCycle,
      registeredMaterials: this.registeredMaterials.length
    };
  }

  // -------------------------------------------------------------------------
  // DISPOSE
  // -------------------------------------------------------------------------
  dispose() {
    for (let i = 0; i < this.lightCount; i++) {
      const eid = this.lightEntities[i];
      if (eid >= 0 && entityExists(world, eid)) {
        releaseEntity(eid);
      }
    }
    this.lightEntities.fill(-1);
    this.lightCount = 0;
    this.freeSlots = [];
    this.registeredMaterials = [];
  }
}

// ---------------------------------------------------------------------------
// SYSTEM FUNCTION
// ---------------------------------------------------------------------------
export function lightingShaderSystem(worldRef, lighting, delta, elapsed) {
  lighting.update(delta, elapsed);
}

// ---------------------------------------------------------------------------
// HELPER — APPLY LIGHTING UNIFORMS TO A MATERIAL
// ---------------------------------------------------------------------------
export function applyLightingToMaterial(material, lighting) {
  if (!material || !material.uniforms || !lighting) return;
  lighting.writeUniforms(material.uniforms);
}

export default {
  LightingShader,
  LightingConfiguration,
  lightingShaderSystem,
  applyLightingToMaterial,
  GLSL_TOON_DIFFUSE,
  GLSL_TOON_SPECULAR,
  GLSL_RIM_LIGHT,
  GLSL_HEMI_AMBIENT,
  GLSL_ATMOSPHERIC,
  GLSL_SHADOW_TINT,
  GLSL_FULL_LIGHTING,
  GLSL_LIGHT_UNIFORMS,
  LIGHT_TYPE_SUN,
  LIGHT_TYPE_MOON,
  LIGHT_TYPE_SKY,
  LIGHT_TYPE_RIM,
  LIGHT_TYPE_BOUNCE,
  LIGHT_TYPE_POINT
};