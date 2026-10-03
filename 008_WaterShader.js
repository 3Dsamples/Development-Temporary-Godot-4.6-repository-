// File : 8
// name : shaders/WaterShader.js
// description : Specialized water shader for cel-shaded anime ocean. Supports Gerstner wave displacement, depth-based color gradient, foam generation near rocks, animated caustics, and specular highlights. Extends the UniversalAnimeShaderMaterial approach with dynamic wave simulation driven by bitecs WaterTile components.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import { GLSL_WORLEY_2D } from '../utils/003_ProceduralNoise.js';

// ---------------------------------------------------------------------------
// WATER VERTEX SHADER
// ---------------------------------------------------------------------------
export const WATER_VERTEX_SHADER = `
precision highp float;

attribute vec2 aTileCoord;
attribute float aWavePhase;
attribute float aWaveAmp;
attribute float aFoamIntensity;

uniform float uTime;
uniform vec2 uWaveDir1;
uniform vec2 uWaveDir2;
uniform vec2 uWaveDir3;
uniform float uWaveFreq1;
uniform float uWaveFreq2;
uniform float uWaveFreq3;
uniform float uWaveAmp1;
uniform float uWaveAmp2;
uniform float uWaveAmp3;
uniform float uWaveSpeed;
uniform float uGlobalAmp;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vWaveHeight;
varying float vFoam;
varying vec2 vUv;
varying vec2 vTileCoord;

vec3 gerstnerWave(vec2 dir, float freq, float amp, float speed, vec2 p, float t, out vec2 grad) {
  float phase = dot(dir, p) * freq + t * speed;
  float s = sin(phase);
  float c = cos(phase);
  float steepness = clamp(amp * freq, 0.0, 1.0);

  grad.x = dir.x * freq * amp * c;
  grad.y = dir.y * freq * amp * c;

  float dispY = amp * s;
  float dispX = dir.x * amp * c * steepness;
  float dispZ = dir.y * amp * c * steepness;

  return vec3(dispX, dispY, dispZ);
}

void main() {
  vColor = color;
  vUv = uv;
  vTileCoord = aTileCoord;

  vec3 pos = position;

  vec2 p = pos.xz;
  vec2 grad1;
  vec2 grad2;
  vec2 grad3;

  float t = uTime * uWaveSpeed + aWavePhase;
  float ampScale = uGlobalAmp * aWaveAmp;

  vec3 w1 = gerstnerWave(uWaveDir1, uWaveFreq1, uWaveAmp1 * ampScale, 1.0, p, t, grad1);
  vec3 w2 = gerstnerWave(uWaveDir2, uWaveFreq2, uWaveAmp2 * ampScale, 1.3, p, t * 1.2, grad2);
  vec3 w3 = gerstnerWave(uWaveDir3, uWaveFreq3, uWaveAmp3 * ampScale, 0.7, p, t * 0.8, grad3);

  vec3 waveOffset = w1 + w2 + w3;
  pos += waveOffset;

  vWaveHeight = waveOffset.y;

  vec2 grad = grad1 + grad2 + grad3;
  vec3 waveNormal = normalize(vec3(-grad.x, 1.0, -grad.y));

  vec4 worldPos = modelMatrix * vec4(pos, 1.0);
  vWorldPos = worldPos.xyz;

  vec4 viewPos = viewMatrix * worldPos;
  vViewDir = normalize(cameraPosition - worldPos.xyz);

  vec3 baseNormal = normalize(normalMatrix * normal);
  vec3 perturbedNormal = normalize(normalMatrix * waveNormal);
  vNormal = normalize(mix(baseNormal, perturbedNormal, 0.85));

  vDepth = -viewPos.z;

  float crestFoam = smoothstep(0.35, 0.75, waveOffset.y * 2.0 + 0.3);
  vFoam = clamp(crestFoam * aFoamIntensity + aFoamIntensity * 0.3, 0.0, 1.0);

  gl_Position = projectionMatrix * viewPos;
}
`;

// ---------------------------------------------------------------------------
// WATER FRAGMENT SHADER
// ---------------------------------------------------------------------------
export const WATER_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;
varying float vWaveHeight;
varying float vFoam;
varying vec2 vUv;
varying vec2 vTileCoord;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uRimColor;

uniform vec3 uWaterDeep;
uniform vec3 uWaterMid;
uniform vec3 uWaterShallow;
uniform vec3 uWaterCrest;
uniform vec3 uWaterHighlight;

uniform vec3 uFoamColor;
uniform vec3 uFoamSoftColor;

uniform float uBand1;
uniform float uBand2;
uniform float uBand3;

uniform float uSpecPower;
uniform float uSpecStrength;
uniform vec3 uSpecColor;

uniform float uRimPower;
uniform float uRimStrength;

uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogFar;

uniform float uTime;
uniform float uCausticStrength;
uniform float uCausticScale;
uniform float uCausticSpeed;

uniform float uOpacity;

${GLSL_WORLEY_2D}

float toonDiffuse(float ndotl, float b1, float b2, float b3) {
  if (ndotl > b3) return 1.0;
  if (ndotl > b2) return 0.75;
  if (ndotl > b1) return 0.45;
  return 0.18;
}

vec3 waterDepthColor(float depth) {
  float d = clamp(depth, 0.0, 1.0);
  if (d < 0.25) {
    return mix(uWaterHighlight, uWaterCrest, d * 4.0);
  }
  if (d < 0.5) {
    return mix(uWaterCrest, uWaterShallow, (d - 0.25) * 4.0);
  }
  if (d < 0.75) {
    return mix(uWaterShallow, uWaterMid, (d - 0.5) * 4.0);
  }
  return mix(uWaterMid, uWaterDeep, (d - 0.75) * 4.0);
}

float causticPattern(vec2 p, float t) {
  vec2 p1 = p * uCausticScale + vec2(t * uCausticSpeed, t * uCausticSpeed * 0.7);
  vec2 p2 = p * uCausticScale * 1.5 + vec2(-t * uCausticSpeed * 0.8, t * uCausticSpeed * 0.4);

  float w1 = worley(p1, 42.0);
  float w2 = worley(p2, 17.0);

  float c = 1.0 - abs(w1 - w2);
  c = pow(c, 3.0);
  return c;
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  vec3 L = normalize(uSunDir);

  if (dot(N, V) < 0.0) N = -N;

  float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

  float shallowFactor = smoothstep(0.0, 12.0, vDepth);
  vec3 baseWater = waterDepthColor(shallowFactor);

  float crestBrightness = smoothstep(0.0, 0.4, vWaveHeight + 0.2);
  baseWater = mix(baseWater, uWaterCrest, crestBrightness * 0.3);

  float caustic = causticPattern(vWorldPos.xz, uTime);
  float causticMask = (1.0 - shallowFactor) * uCausticStrength;
  baseWater += uWaterHighlight * caustic * causticMask * 0.4;

  float ndotl = dot(N, L) * 0.5 + 0.5;
  float toon = toonDiffuse(ndotl, uBand1, uBand2, uBand3);

  vec3 ambient = uSkyColor * 0.5;
  vec3 skyReflection = uSkyColor * fresnel * 0.6;

  vec3 lit = baseWater * uSunColor * toon + baseWater * ambient + skyReflection;

  vec3 halfVec = normalize(L + V);
  float ndoth = max(dot(N, halfVec), 0.0);
  float spec = pow(ndoth, uSpecPower);
  spec = smoothstep(0.85, 0.92, spec) * uSpecStrength;
  lit += uSpecColor * spec;

  float foamEdge = smoothstep(0.4, 0.65, vFoam + vColor.r * 0.3);
  vec3 foamColor = mix(uFoamSoftColor, uFoamColor, foamEdge);
  lit = mix(lit, foamColor, foamEdge * 0.95);

  float sparkle = smoothstep(0.55, 0.85, vWaveHeight * 2.0);
  lit += uFoamColor * sparkle * 0.3;

  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, uRimPower);
  rim = smoothstep(0.35, 0.75, rim);
  lit += uRimColor * rim * uRimStrength;

  float fogFactor = smoothstep(uFogNear, uFogFar, vDepth);
  lit = mix(lit, uFogColor, fogFactor);

  gl_FragColor = vec4(lit, uOpacity);
}
`;

// ---------------------------------------------------------------------------
// WATER MATERIAL CLASS
// ---------------------------------------------------------------------------
export class WaterMaterial extends THREE.ShaderMaterial {
  constructor(params = {}) {
    const {
      opacity = 1.0,
      transparent = false,
      side = THREE.FrontSide,
      depthWrite = true
    } = params;

    super({
      vertexShader: WATER_VERTEX_SHADER,
      fragmentShader: WATER_FRAGMENT_SHADER,
      vertexColors: true,
      side,
      transparent,
      depthWrite,
      lights: false,
      fog: false,
      uniforms: {
        uTime: { value: 0.0 },

        uWaveDir1: { value: new THREE.Vector2(1.0, 0.3).normalize() },
        uWaveDir2: { value: new THREE.Vector2(-0.6, 1.0).normalize() },
        uWaveDir3: { value: new THREE.Vector2(0.4, -0.9).normalize() },
        uWaveFreq1: { value: 1.2 },
        uWaveFreq2: { value: 2.4 },
        uWaveFreq3: { value: 4.8 },
        uWaveAmp1: { value: 0.18 },
        uWaveAmp2: { value: 0.08 },
        uWaveAmp3: { value: 0.04 },
        uWaveSpeed: { value: 1.0 },
        uGlobalAmp: { value: 1.0 },

        uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
        uSunColor: { value: new THREE.Color(1.0, 0.96, 0.85) },
        uSkyColor: { value: new THREE.Color(0.45, 0.62, 0.85) },
        uRimColor: { value: new THREE.Color(0.60, 0.95, 1.00) },

        uWaterDeep: { value: new THREE.Color(0.015, 0.075, 0.180) },
        uWaterMid: { value: new THREE.Color(0.030, 0.280, 0.440) },
        uWaterShallow: { value: new THREE.Color(0.060, 0.540, 0.680) },
        uWaterCrest: { value: new THREE.Color(0.100, 0.760, 0.860) },
        uWaterHighlight: { value: new THREE.Color(0.580, 0.940, 0.980) },

        uFoamColor: { value: new THREE.Color(1.0, 1.0, 1.0) },
        uFoamSoftColor: { value: new THREE.Color(0.86, 0.94, 1.0) },

        uBand1: { value: 0.38 },
        uBand2: { value: 0.60 },
        uBand3: { value: 0.80 },

        uSpecPower: { value: 64.0 },
        uSpecStrength: { value: 0.7 },
        uSpecColor: { value: new THREE.Color(1.0, 1.0, 1.0) },

        uRimPower: { value: 2.0 },
        uRimStrength: { value: 0.45 },

        uFogColor: { value: new THREE.Color(0.20, 0.45, 0.65) },
        uFogNear: { value: 10.0 },
        uFogFar: { value: 60.0 },

        uCausticStrength: { value: 0.6 },
        uCausticScale: { value: 0.8 },
        uCausticSpeed: { value: 0.3 },

        uOpacity: { value: opacity }
      }
    });

    this.userData.isWaterMaterial = true;
  }

  setWaveParams(dir1, dir2, dir3, freq1, freq2, freq3, amp1, amp2, amp3) {
    if (dir1) this.uniforms.uWaveDir1.value.copy(dir1).normalize();
    if (dir2) this.uniforms.uWaveDir2.value.copy(dir2).normalize();
    if (dir3) this.uniforms.uWaveDir3.value.copy(dir3).normalize();
    if (freq1 !== undefined) this.uniforms.uWaveFreq1.value = freq1;
    if (freq2 !== undefined) this.uniforms.uWaveFreq2.value = freq2;
    if (freq3 !== undefined) this.uniforms.uWaveFreq3.value = freq3;
    if (amp1 !== undefined) this.uniforms.uWaveAmp1.value = amp1;
    if (amp2 !== undefined) this.uniforms.uWaveAmp2.value = amp2;
    if (amp3 !== undefined) this.uniforms.uWaveAmp3.value = amp3;
    return this;
  }

  setWaveSpeed(speed, globalAmp) {
    if (speed !== undefined) this.uniforms.uWaveSpeed.value = speed;
    if (globalAmp !== undefined) this.uniforms.uGlobalAmp.value = globalAmp;
    return this;
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

  setWaterColors(deep, mid, shallow, crest, highlight) {
    if (deep) this.uniforms.uWaterDeep.value.copy(deep);
    if (mid) this.uniforms.uWaterMid.value.copy(mid);
    if (shallow) this.uniforms.uWaterShallow.value.copy(shallow);
    if (crest) this.uniforms.uWaterCrest.value.copy(crest);
    if (highlight) this.uniforms.uWaterHighlight.value.copy(highlight);
    return this;
  }

  setCaustics(strength, scale, speed) {
    if (strength !== undefined) this.uniforms.uCausticStrength.value = strength;
    if (scale !== undefined) this.uniforms.uCausticScale.value = scale;
    if (speed !== undefined) this.uniforms.uCausticSpeed.value = speed;
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
// WATER MATERIAL PRESETS
// ---------------------------------------------------------------------------
export function createWaterMaterial() {
  const mat = new WaterMaterial({ opacity: 1.0 });
  mat.setWaveParams(
    new THREE.Vector2(1.0, 0.3),
    new THREE.Vector2(-0.6, 1.0),
    new THREE.Vector2(0.4, -0.9),
    1.2, 2.4, 4.8,
    0.18, 0.08, 0.04
  );
  mat.setWaveSpeed(1.0, 1.0);
  mat.setWaterColors(
    new THREE.Color(0.015, 0.075, 0.180),
    new THREE.Color(0.030, 0.280, 0.440),
    new THREE.Color(0.060, 0.540, 0.680),
    new THREE.Color(0.100, 0.760, 0.860),
    new THREE.Color(0.580, 0.940, 0.980)
  );
  mat.setCaustics(0.6, 0.8, 0.3);
  mat.setFog(new THREE.Color(0.20, 0.45, 0.65), 10.0, 60.0);
  return mat;
}

export function createCalmWaterMaterial() {
  const mat = new WaterMaterial({ opacity: 1.0 });
  mat.setWaveParams(
    new THREE.Vector2(1.0, 0.2),
    new THREE.Vector2(-0.5, 0.9),
    new THREE.Vector2(0.3, -0.8),
    0.8, 1.6, 3.2,
    0.08, 0.04, 0.02
  );
  mat.setWaveSpeed(0.6, 0.7);
  mat.setCaustics(0.8, 1.0, 0.2);
  return mat;
}

export function createStormyWaterMaterial() {
  const mat = new WaterMaterial({ opacity: 1.0 });
  mat.setWaveParams(
    new THREE.Vector2(1.0, 0.4),
    new THREE.Vector2(-0.7, 1.0),
    new THREE.Vector2(0.5, -0.9),
    1.8, 3.6, 7.2,
    0.32, 0.16, 0.08
  );
  mat.setWaveSpeed(1.6, 1.6);
  mat.setSun(new THREE.Vector3(0.3, 0.9, 0.2), new THREE.Color(0.7, 0.7, 0.75));
  mat.setSky(new THREE.Color(0.35, 0.42, 0.55));
  mat.setCaustics(0.3, 0.6, 0.5);
  mat.setFog(new THREE.Color(0.15, 0.25, 0.35), 8.0, 40.0);
  return mat;
}

// ---------------------------------------------------------------------------
// WATER MATERIAL REGISTRY
// ---------------------------------------------------------------------------
export const WaterMaterialRegistry = {
  default: null,
  calm: null,
  stormy: null,

  init() {
    this.default = createWaterMaterial();
    this.calm = createCalmWaterMaterial();
    this.stormy = createStormyWaterMaterial();
  },

  updateTime(elapsed) {
    if (this.default) this.default.updateTime(elapsed);
    if (this.calm) this.calm.updateTime(elapsed);
    if (this.stormy) this.stormy.updateTime(elapsed);
  },

  updateSun(dir, color) {
    if (this.default) this.default.setSun(dir, color);
    if (this.calm) this.calm.setSun(dir, color);
    if (this.stormy) this.stormy.setSun(dir, color);
  },

  updateSky(color) {
    if (this.default) this.default.setSky(color);
    if (this.calm) this.calm.setSky(color);
    if (this.stormy) this.stormy.setSky(color);
  },

  updateWaterColors(deep, mid, shallow, crest, highlight) {
    if (this.default) this.default.setWaterColors(deep, mid, shallow, crest, highlight);
    if (this.calm) this.calm.setWaterColors(deep, mid, shallow, crest, highlight);
    if (this.stormy) this.stormy.setWaterColors(deep, mid, shallow, crest, highlight);
  },

  dispose() {
    if (this.default) this.default.dispose();
    if (this.calm) this.calm.dispose();
    if (this.stormy) this.stormy.dispose();
    this.default = null;
    this.calm = null;
    this.stormy = null;
  }
};

// ---------------------------------------------------------------------------
// TILE ATTRIBUTE FACTORY
// Adds per-instance (or per-vertex) tile data to a water tile geometry.
// ---------------------------------------------------------------------------
export function applyWaterTileAttributes(geometry, tileCoordX, tileCoordY, wavePhase, waveAmp, foamIntensity) {
  const posAttr = geometry.getAttribute('position');
  const count = posAttr ? posAttr.count : 0;
  if (count === 0) return geometry;

  const tileCoords = new Float32Array(count * 2);
  const wavePhases = new Float32Array(count);
  const waveAmps = new Float32Array(count);
  const foamIntensities = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    tileCoords[i * 2] = tileCoordX;
    tileCoords[i * 2 + 1] = tileCoordY;
    wavePhases[i] = wavePhase;
    waveAmps[i] = waveAmp;
    foamIntensities[i] = foamIntensity;
  }

  if (geometry.isInstancedBufferGeometry) {
    geometry.setAttribute('aTileCoord', new THREE.InstancedBufferAttribute(tileCoords, 2));
    geometry.setAttribute('aWavePhase', new THREE.InstancedBufferAttribute(wavePhases, 1));
    geometry.setAttribute('aWaveAmp', new THREE.InstancedBufferAttribute(waveAmps, 1));
    geometry.setAttribute('aFoamIntensity', new THREE.InstancedBufferAttribute(foamIntensities, 1));
  } else {
    geometry.setAttribute('aTileCoord', new THREE.BufferAttribute(tileCoords, 2));
    geometry.setAttribute('aWavePhase', new THREE.BufferAttribute(wavePhases, 1));
    geometry.setAttribute('aWaveAmp', new THREE.BufferAttribute(waveAmps, 1));
    geometry.setAttribute('aFoamIntensity', new THREE.BufferAttribute(foamIntensities, 1));
  }

  return geometry;
}

export default {
  WaterMaterial,
  WATER_VERTEX_SHADER,
  WATER_FRAGMENT_SHADER,
  createWaterMaterial,
  createCalmWaterMaterial,
  createStormyWaterMaterial,
  WaterMaterialRegistry,
  applyWaterTileAttributes
};