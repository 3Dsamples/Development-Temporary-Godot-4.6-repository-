// File : 6
// name : shaders/VegetationShader.js
// description : Specialized vegetation shader extending UniversalAnimeShaderMaterial. Adds multi-layer grass wind simulation, per-blade sway, tip curvature, color gradient by height, and alpha-tested billboard support. Integrates with bitecs Vegetation component for real-time wind response on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import { GLSL_SIMPLEX_2D, GLSL_FBM_2D } from '../utils/003_ProceduralNoise.js';

// ---------------------------------------------------------------------------
// VEGETATION VERTEX SHADER
// Multi-frequency wind simulation, per-blade bending, root anchoring.
// ---------------------------------------------------------------------------
export const VEGETATION_VERTEX_SHADER = `
precision highp float;

attribute vec3 color;
attribute float aSwayPhase;
attribute float aSwayStrength;
attribute float aHeightFactor;
attribute float aBladeId;

uniform float uTime;
uniform float uWindStrength;
uniform vec2 uWindDir;
uniform float uWindFrequency;
uniform float uWindGustStrength;
uniform float uBendCurve;
uniform float uRootAnchor;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vSwayMask;
varying float vHeightFactor;
varying float vBladeId;

${GLSL_SIMPLEX_2D}

void main() {
  vColor = color;
  vHeightFactor = aHeightFactor;
  vBladeId = aBladeId;

  vec3 pos = position;

  float bendMask = max(pos.y - uRootAnchor, 0.0);
  bendMask = pow(bendMask, uBendCurve);

  float baseSway = sin(uTime * uWindFrequency + aSwayPhase) * aSwayStrength;
  float gust = sin(uTime * 0.6 + aSwayPhase * 0.3) * uWindGustStrength;
  float highFreq = sin(uTime * 3.1 + aSwayPhase * 2.0) * 0.15;

  float totalSway = (baseSway + gust + highFreq) * uWindStrength;

  float curvature = bendMask * bendMask * 0.3;

  pos.x += uWindDir.x * totalSway * bendMask + curvature * uWindDir.x;
  pos.z += uWindDir.y * totalSway * bendMask + curvature * uWindDir.y;
  pos.y -= bendMask * abs(totalSway) * 0.2;

  float rustle = snoise(pos.xz * 4.0 + uTime * 0.5) * 0.02;
  pos.x += rustle * bendMask;
  pos.z += rustle * bendMask;

  vSwayMask = bendMask;

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
// VEGETATION FRAGMENT SHADER
// Height-based color gradient, tip highlights, translucent backlight.
// ---------------------------------------------------------------------------
export const VEGETATION_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vSwayMask;
varying float vHeightFactor;
varying float vBladeId;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uBounceColor;
uniform vec3 uRimColor;

uniform vec3 uVegRootColor;
uniform vec3 uVegMidColor;
uniform vec3 uVegTipColor;

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

uniform float uTranslucency;
uniform float uTipGlow;
uniform float uAlphaTest;
uniform float uOpacity;

${GLSL_FBM_2D}

float toonDiffuse(float ndotl, float b1, float b2, float b3) {
  if (ndotl > b3) return 1.0;
  if (ndotl > b2) return 0.75;
  if (ndotl > b1) return 0.45;
  return 0.18;
}

vec3 applyShadowTint(vec3 color, float shadowMask, vec3 tint) {
  return mix(color, color * tint * 1.4, clamp(shadowMask, 0.0, 1.0));
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  vec3 L = normalize(uSunDir);

  if (dot(N, V) < 0.0) N = -N;

  vec3 vegGradient;
  float h = clamp(vHeightFactor, 0.0, 1.0);
  if (h < 0.5) {
    vegGradient = mix(uVegRootColor, uVegMidColor, h * 2.0);
  } else {
    vegGradient = mix(uVegMidColor, uVegTipColor, (h - 0.5) * 2.0);
  }

  vec3 baseColor = mix(vegGradient, vColor, 0.35);

  float microNoise = fbm(vWorldPos.xz * 3.0 + vBladeId * 0.1, 2, 2.0, 0.5);
  baseColor *= 0.9 + microNoise * 0.2;

  float ndotl = dot(N, L) * 0.5 + 0.5;
  float toon = toonDiffuse(ndotl, uBand1, uBand2, uBand3);

  vec3 ambient = uSkyColor * 0.4;

  float bounceFactor = max(-N.y, 0.0) * 0.25;
  vec3 bounce = uBounceColor * bounceFactor;

  float trans = max(dot(-L, V), 0.0);
  trans = pow(trans, 2.0) * uTranslucency;
  vec3 translucent = uSunColor * trans * baseColor * 0.5;

  vec3 lit = baseColor * uSunColor * toon + baseColor * ambient + baseColor * bounce + translucent;

  float shadowMask = 1.0 - toon;
  lit = applyShadowTint(lit, shadowMask * uShadowStrength, uShadowTint);

  vec3 halfVec = normalize(L + V);
  float ndoth = max(dot(N, halfVec), 0.0);
  float spec = pow(ndoth, uSpecPower);
  spec = smoothstep(0.6, 0.65, spec) * uSpecStrength;
  lit += uSpecColor * spec;

  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, uRimPower);
  rim = smoothstep(0.3, 0.75, rim);
  lit += uRimColor * rim * uRimStrength;

  float tipGlow = smoothstep(0.6, 1.0, vHeightFactor) * uTipGlow;
  lit += uSunColor * tipGlow * 0.4;

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
// VEGETATION MATERIAL CLASS
// ---------------------------------------------------------------------------
export class VegetationMaterial extends THREE.ShaderMaterial {
  constructor(params = {}) {
    const {
      opacity = 1.0,
      alphaTest = 0.01,
      side = THREE.DoubleSide,
      transparent = false,
      depthWrite = true,
      windStrength = 1.0
    } = params;

    super({
      vertexShader: VEGETATION_VERTEX_SHADER,
      fragmentShader: VEGETATION_FRAGMENT_SHADER,
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
        uRimColor: { value: new THREE.Color(0.70, 1.00, 0.60) },

        uVegRootColor: { value: new THREE.Color(0.04, 0.14, 0.03) },
        uVegMidColor: { value: new THREE.Color(0.18, 0.44, 0.10) },
        uVegTipColor: { value: new THREE.Color(0.30, 0.62, 0.18) },

        uBand1: { value: 0.40 },
        uBand2: { value: 0.58 },
        uBand3: { value: 0.78 },
        uShadowStrength: { value: 0.5 },
        uShadowTint: { value: new THREE.Color(0.10, 0.22, 0.10) },

        uSpecPower: { value: 24.0 },
        uSpecStrength: { value: 0.15 },
        uSpecColor: { value: new THREE.Color(0.9, 1.0, 0.8) },

        uRimPower: { value: 3.0 },
        uRimStrength: { value: 0.5 },

        uAOStrength: { value: 0.4 },

        uFogColor: { value: new THREE.Color(0.35, 0.55, 0.70) },
        uFogNear: { value: 8.0 },
        uFogFar: { value: 45.0 },

        uTime: { value: 0.0 },

        uWindStrength: { value: windStrength },
        uWindDir: { value: new THREE.Vector2(1.0, 0.3).normalize() },
        uWindFrequency: { value: 1.8 },
        uWindGustStrength: { value: 0.4 },
        uBendCurve: { value: 1.4 },
        uRootAnchor: { value: 0.0 },

        uTranslucency: { value: 0.35 },
        uTipGlow: { value: 0.6 },
        uAlphaTest: { value: alphaTest },
        uOpacity: { value: opacity }
      }
    });

    this.userData.isVegetationMaterial = true;
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

  setGradient(root, mid, tip) {
    if (root) this.uniforms.uVegRootColor.value.copy(root);
    if (mid) this.uniforms.uVegMidColor.value.copy(mid);
    if (tip) this.uniforms.uVegTipColor.value.copy(tip);
    return this;
  }

  setToonBands(b1, b2, b3) {
    if (b1 !== undefined) this.uniforms.uBand1.value = b1;
    if (b2 !== undefined) this.uniforms.uBand2.value = b2;
    if (b3 !== undefined) this.uniforms.uBand3.value = b3;
    return this;
  }

  setWind(strength, dirX, dirY, frequency) {
    if (strength !== undefined) this.uniforms.uWindStrength.value = strength;
    if (dirX !== undefined && dirY !== undefined) {
      this.uniforms.uWindDir.value.set(dirX, dirY).normalize();
    }
    if (frequency !== undefined) this.uniforms.uWindFrequency.value = frequency;
    return this;
  }

  setWindGust(strength) {
    if (strength !== undefined) this.uniforms.uWindGustStrength.value = strength;
    return this;
  }

  setBend(curve, anchor) {
    if (curve !== undefined) this.uniforms.uBendCurve.value = curve;
    if (anchor !== undefined) this.uniforms.uRootAnchor.value = anchor;
    return this;
  }

  setRim(color, power, strength) {
    if (color) this.uniforms.uRimColor.value.copy(color);
    if (power !== undefined) this.uniforms.uRimPower.value = power;
    if (strength !== undefined) this.uniforms.uRimStrength.value = strength;
    return this;
  }

  setTranslucency(value) {
    if (value !== undefined) this.uniforms.uTranslucency.value = value;
    return this;
  }

  setTipGlow(value) {
    if (value !== undefined) this.uniforms.uTipGlow.value = value;
    return this;
  }

  setFog(color, near, far) {
    if (color) this.uniforms.uFogColor.value.copy(color);
    if (near !== undefined) this.uniforms.uFogNear.value = near;
    if (far !== undefined) this.uniforms.uFogFar.value = far;
    return this;
  }

  updateTime(elapsed) {
    this.uniforms.uTime.value = elapsed;
    return this;
  }
}

// ---------------------------------------------------------------------------
// VEGETATION MATERIAL PRESETS
// ---------------------------------------------------------------------------
export function createGrassMaterial() {
  const mat = new VegetationMaterial({ windStrength: 1.2 });
  mat.setGradient(
    new THREE.Color(0.04, 0.14, 0.03),
    new THREE.Color(0.18, 0.44, 0.10),
    new THREE.Color(0.30, 0.62, 0.18)
  );
  mat.setToonBands(0.40, 0.58, 0.78);
  mat.setWind(1.2, 1.0, 0.3, 1.8);
  mat.setWindGust(0.45);
  mat.setBend(1.4, 0.0);
  mat.setTranslucency(0.40);
  mat.setTipGlow(0.65);
  mat.setRim(new THREE.Color(0.70, 1.00, 0.60), 3.0, 0.5);
  return mat;
}

export function createBushMaterial() {
  const mat = new VegetationMaterial({ windStrength: 0.6 });
  mat.setGradient(
    new THREE.Color(0.05, 0.16, 0.04),
    new THREE.Color(0.20, 0.42, 0.12),
    new THREE.Color(0.32, 0.58, 0.20)
  );
  mat.setToonBands(0.38, 0.56, 0.76);
  mat.setWind(0.6, 1.0, 0.3, 1.2);
  mat.setWindGust(0.25);
  mat.setBend(1.8, 0.05);
  mat.setTranslucency(0.25);
  mat.setTipGlow(0.40);
  mat.setRim(new THREE.Color(0.65, 0.95, 0.55), 3.5, 0.4);
  return mat;
}

export function createMossMaterial() {
  const mat = new VegetationMaterial({ windStrength: 0.1 });
  mat.setGradient(
    new THREE.Color(0.06, 0.20, 0.05),
    new THREE.Color(0.14, 0.36, 0.10),
    new THREE.Color(0.22, 0.48, 0.16)
  );
  mat.setToonBands(0.42, 0.60, 0.80);
  mat.setWind(0.1, 1.0, 0.3, 0.8);
  mat.setWindGust(0.05);
  mat.setBend(2.0, 0.0);
  mat.setTranslucency(0.15);
  mat.setTipGlow(0.20);
  mat.setRim(new THREE.Color(0.55, 0.85, 0.45), 4.0, 0.3);
  return mat;
}

// ---------------------------------------------------------------------------
// VEGETATION MATERIAL REGISTRY
// ---------------------------------------------------------------------------
export const VegetationMaterialRegistry = {
  grass: null,
  bush: null,
  moss: null,

  init() {
    this.grass = createGrassMaterial();
    this.bush = createBushMaterial();
    this.moss = createMossMaterial();
  },

  updateTime(elapsed) {
    if (this.grass) this.grass.updateTime(elapsed);
    if (this.bush) this.bush.updateTime(elapsed);
    if (this.moss) this.moss.updateTime(elapsed);
  },

  updateSun(dir, color) {
    if (this.grass) this.grass.setSun(dir, color);
    if (this.bush) this.bush.setSun(dir, color);
    if (this.moss) this.moss.setSun(dir, color);
  },

  updateSky(color) {
    if (this.grass) this.grass.setSky(color);
    if (this.bush) this.bush.setSky(color);
    if (this.moss) this.moss.setSky(color);
  },

  updateWind(strength, dirX, dirY) {
    if (this.grass) this.grass.setWind(strength, dirX, dirY);
    if (this.bush) this.bush.setWind(strength * 0.5, dirX, dirY);
    if (this.moss) this.moss.setWind(strength * 0.1, dirX, dirY);
  },

  dispose() {
    if (this.grass) this.grass.dispose();
    if (this.bush) this.bush.dispose();
    if (this.moss) this.moss.dispose();
    this.grass = null;
    this.bush = null;
    this.moss = null;
  }
};

// ---------------------------------------------------------------------------
// VERTEX ATTRIBUTE FACTORY
// ---------------------------------------------------------------------------
export function applyVegetationAttributes(geometry, instanceCount, instanceData) {
  const swayPhase = new Float32Array(instanceCount);
  const swayStrength = new Float32Array(instanceCount);
  const heightFactor = new Float32Array(instanceCount);
  const bladeId = new Float32Array(instanceCount);

  for (let i = 0; i < instanceCount; i++) {
    const d = instanceData[i];
    swayPhase[i] = d.swayPhase || 0;
    swayStrength[i] = d.swayStrength || 1.0;
    heightFactor[i] = d.heightFactor || 0;
    bladeId[i] = d.bladeId || i;
  }

  geometry.setAttribute('aSwayPhase', new THREE.InstancedBufferAttribute(swayPhase, 1));
  geometry.setAttribute('aSwayStrength', new THREE.InstancedBufferAttribute(swayStrength, 1));
  geometry.setAttribute('aHeightFactor', new THREE.InstancedBufferAttribute(heightFactor, 1));
  geometry.setAttribute('aBladeId', new THREE.InstancedBufferAttribute(bladeId, 1));

  return geometry;
}

export default {
  VegetationMaterial,
  VEGETATION_VERTEX_SHADER,
  VEGETATION_FRAGMENT_SHADER,
  createGrassMaterial,
  createBushMaterial,
  createMossMaterial,
  VegetationMaterialRegistry,
  applyVegetationAttributes
};