// File : 5
// name : shaders/UniversalAnimeShaderMaterial.js
// description : Master cel-shaded anime material system. Produces hard-edged toon shading with multi-band lighting, rim light, specular highlights, ambient occlusion, and shadow tinting. Single ShaderMaterial class usable across rocks, water, vegetation, and environment. Mobile-optimized with minimal uniforms and early-out branches.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import { GLSL_SIMPLEX_2D, GLSL_FBM_2D } from '../utils/003_ProceduralNoise.js';
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';

// ---------------------------------------------------------------------------
// VERTEX SHADER
// Passes world position, normal, view direction, and custom attributes to
// the fragment shader. Supports vertex color modulation and wind sway.
// ---------------------------------------------------------------------------
export const ANIME_VERTEX_SHADER = `
precision highp float;

attribute vec3 color;
#ifdef USE_SWAY
attribute float aSwayPhase;
attribute float aSwayStrength;
#endif

uniform float uTime;
uniform float uWindStrength;
uniform vec2 uWindDir;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vSwayMask;

void main() {
  vColor = color;

  vec3 pos = position;
  vSwayMask = 0.0;

#ifdef USE_SWAY
  if (aSwayStrength > 0.0) {
    float sway = sin(uTime * 1.8 + aSwayPhase) * aSwayStrength * uWindStrength;
    float bendMask = max(position.y, 0.0);
    pos.x += uWindDir.x * sway * bendMask;
    pos.z += uWindDir.y * sway * bendMask;
    vSwayMask = bendMask;
  }
#endif

  vec4 worldPos = modelMatrix * vec4(pos, 1.0);
  vWorldPos = worldPos.xyz;

  vec4 viewPos = viewMatrix * worldPos;
  vViewDir = normalize(cameraPosition - worldPos.xyz);

  vNormal = normalize(normalMatrix * normal);

  vDepth = -viewPos.z;

  gl_Position = projectionMatrix * viewPos;
}
`;

// ---------------------------------------------------------------------------
// FRAGMENT SHADER — Cel-shaded anime lighting
// ---------------------------------------------------------------------------
export const ANIME_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vSwayMask;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uBounceColor;
uniform vec3 uRimColor;

uniform float uBand1;
uniform float uBand2;
uniform float uBand3;
uniform float uShadowStrength;
uniform vec3 uShadowTint;

uniform float uSpecPower;
uniform float uSpecStrength;
uniform vec3 uSpecColor;

uniform float uRimPower;
uniform float uRimStrength;

uniform float uAOStrength;

uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogFar;

uniform float uTime;

uniform float uSurfaceNoiseScale;
uniform float uSurfaceNoiseStrength;
uniform vec3 uBaseColor;
uniform float uUseVertexColor;
uniform float uAlphaTest;
uniform float uOpacity;
uniform float uSwayTintStrength;

${GLSL_SIMPLEX_2D}
${GLSL_FBM_2D}
${GLSL_COLOR_UTILS}

float toonDiffuse(float ndotl, float b1, float b2, float b3) {
  if (ndotl > b3) return 1.0;
  if (ndotl > b2) return 0.75;
  if (ndotl > b1) return 0.45;
  return 0.18;
}

float toonSpecular(vec3 normal, vec3 viewDir, vec3 lightDir, float power, float strength) {
  vec3 halfVec = normalize(lightDir + viewDir);
  float ndoth = max(dot(normal, halfVec), 0.0);
  float spec = pow(ndoth, power);
  spec = smoothstep(0.5, 0.55, spec);
  return spec * strength;
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  vec3 L = normalize(uSunDir);

  if (dot(N, V) < 0.0) N = -N;

  vec3 baseColor = mix(uBaseColor, vColor, uUseVertexColor);

  float surfaceNoise = fbm(vWorldPos.xz * uSurfaceNoiseScale, 3, 2.0, 0.5);
  baseColor *= 1.0 + surfaceNoise * uSurfaceNoiseStrength;

  baseColor = mix(baseColor, baseColor * vec3(0.85, 1.0, 0.85), vSwayMask * uSwayTintStrength);

  float ndotl = dot(N, L) * 0.5 + 0.5;
  float toon = toonDiffuse(ndotl, uBand1, uBand2, uBand3);

  vec3 ambient = uSkyColor * 0.35;

  float bounceFactor = max(-N.y, 0.0) * 0.3;
  vec3 bounce = uBounceColor * bounceFactor;

  vec3 lit = baseColor * uSunColor * toon + baseColor * ambient + baseColor * bounce;

  float shadowMask = 1.0 - toon;
  lit = applyShadowTint(lit, shadowMask * uShadowStrength, uShadowTint);

  float spec = toonSpecular(N, V, L, uSpecPower, uSpecStrength);
  lit += uSpecColor * spec;

  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, uRimPower);
  rim = smoothstep(0.3, 0.7, rim);
  lit += uRimColor * rim * uRimStrength;

  float vLum = dot(vColor, vec3(0.299, 0.587, 0.114));
  lit *= mix(1.0, vLum, uAOStrength);

  float fogFactor = smoothstep(uFogNear, uFogFar, vDepth);
  lit = mix(lit, uFogColor, fogFactor);

  float alpha = uOpacity;
  if (alpha < uAlphaTest) discard;

  gl_FragColor = vec4(lit, alpha);
}
`;

// ---------------------------------------------------------------------------
// UNIVERSAL ANIME MATERIAL CLASS
// ---------------------------------------------------------------------------
export class UniversalAnimeMaterial extends THREE.ShaderMaterial {
  constructor(params = {}) {
    const {
      baseColor = new THREE.Color(1, 1, 1),
      useVertexColor = 1.0,
      opacity = 1.0,
      alphaTest = 0.01,
      side = THREE.FrontSide,
      transparent = false,
      depthWrite = true,
      windStrength = 0.0,
      useSway = false
    } = params;

    const defines = {};
    if (useSway) defines.USE_SWAY = '';

    super({
      vertexShader: ANIME_VERTEX_SHADER,
      fragmentShader: ANIME_FRAGMENT_SHADER,
      defines,
      vertexColors: true,
      side,
      transparent,
      depthWrite,
      lights: false,
      fog: false,
      uniforms: {
        uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
        uSunColor: { value: new THREE.Color(1.0, 0.96, 0.85) },
        uSkyColor: { value: new THREE.Color(0.45, 0.62, 0.85) },
        uBounceColor: { value: new THREE.Color(0.15, 0.20, 0.25) },
        uRimColor: { value: new THREE.Color(0.90, 0.95, 1.0) },

        uBand1: { value: 0.35 },
        uBand2: { value: 0.55 },
        uBand3: { value: 0.75 },
        uShadowStrength: { value: 0.65 },
        uShadowTint: { value: new THREE.Color(0.12, 0.18, 0.30) },

        uSpecPower: { value: 32.0 },
        uSpecStrength: { value: 0.4 },
        uSpecColor: { value: new THREE.Color(1.0, 1.0, 1.0) },

        uRimPower: { value: 3.0 },
        uRimStrength: { value: 0.6 },

        uAOStrength: { value: 0.6 },

        uFogColor: { value: new THREE.Color(0.35, 0.55, 0.70) },
        uFogNear: { value: 5.0 },
        uFogFar: { value: 40.0 },

        uTime: { value: 0.0 },

        uWindStrength: { value: windStrength },
        uWindDir: { value: new THREE.Vector2(1.0, 0.3).normalize() },

        uSurfaceNoiseScale: { value: 0.5 },
        uSurfaceNoiseStrength: { value: 0.1 },
        uBaseColor: { value: baseColor },
        uUseVertexColor: { value: useVertexColor },
        uAlphaTest: { value: alphaTest },
        uOpacity: { value: opacity },
        uSwayTintStrength: { value: 0.15 }
      }
    });

    this.userData.isAnimeMaterial = true;
  }

  setSun(dir, color) {
    if (dir) this.uniforms.uSunDir.value.copy(dir).normalize();
    if (color) this.uniforms.uSunColor.value.copy(color);
    return this;
  }

  setSky(color) {
    if (color) this.uniforms.uSkyColor.value.copy(color);
    return this;
  }

  setBounce(color) {
    if (color) this.uniforms.uBounceColor.value.copy(color);
    return this;
  }

  setRim(color, power, strength) {
    if (color) this.uniforms.uRimColor.value.copy(color);
    if (power !== undefined) this.uniforms.uRimPower.value = power;
    if (strength !== undefined) this.uniforms.uRimStrength.value = strength;
    return this;
  }

  setToonBands(b1, b2, b3) {
    if (b1 !== undefined) this.uniforms.uBand1.value = b1;
    if (b2 !== undefined) this.uniforms.uBand2.value = b2;
    if (b3 !== undefined) this.uniforms.uBand3.value = b3;
    return this;
  }

  setSpecular(power, strength, color) {
    if (power !== undefined) this.uniforms.uSpecPower.value = power;
    if (strength !== undefined) this.uniforms.uSpecStrength.value = strength;
    if (color) this.uniforms.uSpecColor.value.copy(color);
    return this;
  }

  setShadow(strength, tint) {
    if (strength !== undefined) this.uniforms.uShadowStrength.value = strength;
    if (tint) this.uniforms.uShadowTint.value.copy(tint);
    return this;
  }

  setFog(color, near, far) {
    if (color) this.uniforms.uFogColor.value.copy(color);
    if (near !== undefined) this.uniforms.uFogNear.value = near;
    if (far !== undefined) this.uniforms.uFogFar.value = far;
    return this;
  }

  setWind(strength, dirX, dirY) {
    if (strength !== undefined) this.uniforms.uWindStrength.value = strength;
    if (dirX !== undefined && dirY !== undefined) {
      this.uniforms.uWindDir.value.set(dirX, dirY).normalize();
    }
    return this;
  }

  setSurface(noiseScale, noiseStrength) {
    if (noiseScale !== undefined) this.uniforms.uSurfaceNoiseScale.value = noiseScale;
    if (noiseStrength !== undefined) this.uniforms.uSurfaceNoiseStrength.value = noiseStrength;
    return this;
  }

  setBaseColor(color) {
    if (color) this.uniforms.uBaseColor.value.copy(color);
    return this;
  }

  setOpacity(opacity, transparent) {
    this.uniforms.uOpacity.value = opacity;
    if (transparent !== undefined) this.transparent = transparent;
    return this;
  }

  setSwayTint(strength) {
    if (strength !== undefined) this.uniforms.uSwayTintStrength.value = strength;
    return this;
  }

  updateTime(elapsed) {
    this.uniforms.uTime.value = elapsed;
    return this;
  }
}

// ---------------------------------------------------------------------------
// MATERIAL PRESETS
// ---------------------------------------------------------------------------

export function createRockMaterial() {
  const mat = new UniversalAnimeMaterial({
    baseColor: new THREE.Color(0.72, 0.66, 0.55),
    useVertexColor: 1.0,
    side: THREE.FrontSide,
    useSway: false
  });
  mat.setToonBands(0.32, 0.52, 0.72);
  mat.setSpecular(16.0, 0.15, new THREE.Color(0.8, 0.8, 0.8));
  mat.setRim(new THREE.Color(0.9, 0.95, 1.0), 2.5, 0.35);
  mat.setShadow(0.7, new THREE.Color(0.14, 0.18, 0.28));
  mat.setSurface(0.8, 0.15);
  mat.setFog(new THREE.Color(0.35, 0.55, 0.70), 8.0, 45.0);
  return mat;
}

export function createVegetationMaterial() {
  const mat = new UniversalAnimeMaterial({
    baseColor: new THREE.Color(0.25, 0.55, 0.18),
    useVertexColor: 1.0,
    side: THREE.DoubleSide,
    transparent: false,
    alphaTest: 0.01,
    windStrength: 1.0,
    useSway: true
  });
  mat.setToonBands(0.40, 0.58, 0.78);
  mat.setSpecular(24.0, 0.1, new THREE.Color(0.9, 1.0, 0.8));
  mat.setRim(new THREE.Color(0.7, 1.0, 0.6), 3.0, 0.5);
  mat.setShadow(0.5, new THREE.Color(0.10, 0.22, 0.10));
  mat.setSurface(1.5, 0.2);
  mat.setWind(1.0, 1.0, 0.3);
  mat.setFog(new THREE.Color(0.35, 0.55, 0.70), 8.0, 45.0);
  return mat;
}

export function createWaterMaterial() {
  const mat = new UniversalAnimeMaterial({
    baseColor: new THREE.Color(0.06, 0.48, 0.65),
    useVertexColor: 1.0,
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    useSway: false
  });
  mat.setToonBands(0.38, 0.60, 0.80);
  mat.setSpecular(64.0, 0.6, new THREE.Color(1.0, 1.0, 1.0));
  mat.setRim(new THREE.Color(0.6, 0.95, 1.0), 2.0, 0.4);
  mat.setShadow(0.4, new THREE.Color(0.05, 0.15, 0.30));
  mat.setSurface(2.0, 0.08);
  mat.setFog(new THREE.Color(0.20, 0.45, 0.65), 10.0, 60.0);
  return mat;
}

export function createEnvironmentMaterial() {
  const mat = new UniversalAnimeMaterial({
    baseColor: new THREE.Color(0.55, 0.65, 0.75),
    useVertexColor: 1.0,
    side: THREE.BackSide,
    useSway: false
  });
  mat.setToonBands(0.30, 0.50, 0.70);
  mat.setSpecular(8.0, 0.05, new THREE.Color(0.5, 0.6, 0.8));
  mat.setRim(new THREE.Color(0.8, 0.9, 1.0), 1.5, 0.2);
  mat.setShadow(0.3, new THREE.Color(0.10, 0.15, 0.25));
  mat.setSurface(0.3, 0.05);
  return mat;
}

// ---------------------------------------------------------------------------
// MATERIAL CACHE
// ---------------------------------------------------------------------------
const _materialCache = new Map();

export function getCachedMaterial(key, factoryFn) {
  if (_materialCache.has(key)) return _materialCache.get(key);
  const mat = factoryFn();
  _materialCache.set(key, mat);
  return mat;
}

export function disposeMaterialCache() {
  _materialCache.forEach(m => m.dispose());
  _materialCache.clear();
}

// ---------------------------------------------------------------------------
// GLOBAL MATERIAL REGISTRY
// ---------------------------------------------------------------------------
export const MaterialRegistry = {
  rock: null,
  vegetation: null,
  water: null,
  environment: null,

  init() {
    this.rock = createRockMaterial();
    this.vegetation = createVegetationMaterial();
    this.water = createWaterMaterial();
    this.environment = createEnvironmentMaterial();
  },

  updateTime(elapsed) {
    if (this.rock) this.rock.updateTime(elapsed);
    if (this.vegetation) this.vegetation.updateTime(elapsed);
    if (this.water) this.water.updateTime(elapsed);
    if (this.environment) this.environment.updateTime(elapsed);
  },

  updateSun(dir, color) {
    if (this.rock) this.rock.setSun(dir, color);
    if (this.vegetation) this.vegetation.setSun(dir, color);
    if (this.water) this.water.setSun(dir, color);
    if (this.environment) this.environment.setSun(dir, color);
  },

  updateSky(color) {
    if (this.rock) this.rock.setSky(color);
    if (this.vegetation) this.vegetation.setSky(color);
    if (this.water) this.water.setSky(color);
    if (this.environment) this.environment.setSky(color);
  },

  dispose() {
    if (this.rock) this.rock.dispose();
    if (this.vegetation) this.vegetation.dispose();
    if (this.water) this.water.dispose();
    if (this.environment) this.environment.dispose();
    this.rock = null;
    this.vegetation = null;
    this.water = null;
    this.environment = null;
  }
};

export default {
  UniversalAnimeMaterial,
  ANIME_VERTEX_SHADER,
  ANIME_FRAGMENT_SHADER,
  createRockMaterial,
  createVegetationMaterial,
  createWaterMaterial,
  createEnvironmentMaterial,
  getCachedMaterial,
  disposeMaterialCache,
  MaterialRegistry
};