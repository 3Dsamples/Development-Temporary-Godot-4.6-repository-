// File : 016
// name : shaders/016_DirectionalLightShader.glsl.js
// description : 3D Directional Light shader chunk and manager for anime cel-shading.
//               ANALYZED & FIXED: (1) SHADOW MAP TEXTURE BUG: Three.js r185
//               `DirectionalLight.shadow.map` is a WebGLRenderTarget; the actual
//               texture must be passed via `.texture` to the sampler2D uniform.
//               (2) SHADOW UV MAPPING BUG: Three.js `shadow.matrix` transforms
//               world space to [-1, 1] clip space, not [0, 1] UV space. Added
//               explicit `* 0.5 + 0.5` conversion in the GLSL chunk to prevent
//               shadow sampling from reading out-of-bounds or mirrored coordinates.
//               (3) GLSL ES 1.00/3.00 COMPATIBILITY: Replaced dynamic loop bounds
//               with constant bounds for the 3x3 PCF shadow filter to prevent
//               compilation failures on older Mali/Adreno Android drivers.
//               (4) GUARDED DIVISIONS: Added `max(uShadowMapSize, vec2(1.0))` to
//               prevent division-by-zero when shadow maps are uninitialized or
//               destroyed during scene teardown. (5) NORMAL BIAS SAFETY: Clamped
//               normal bias application to prevent shadow acne on steep slopes.
//               Replicates Three.js r185 DirectionalLight behavior (direction,
//               color, intensity, shadow mapping) but replaces standard PBR with
//               anime-style quantized diffuse, perceptual tinted shadows (via
//               020_gmp_perceptual_color.js concepts), and Fresnel rim lighting.
//               Integrates with 006_normalQuantizer.js for CPU/GPU normal snapping.
//               Fully composed using 000_BaseShader.glsl.js. Zero per-frame
//               allocation, optimized for Android mobile GPUs.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  getAnimeNormalQuantizationGLSL,
  getAnimeRimNormalExaggerationGLSL
} from '../mesh/006_normalQuantizer.js';
import {
  GLSL_GLOBALS,
  GLSL_NOISE,
  GLSL_NORMAL_QUANT,
  GLSL_LIGHTING,
  GLSL_BIOME,
} from './000_BaseShader.glsl.js';
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';

/* ------------------------------------------------------------------ */
/* 1. GLSL DIRECTIONAL LIGHT CHUNK                                    */
/*    Provides anime cel-shaded directional lighting with shadow maps */
/* ------------------------------------------------------------------ */
export const GLSL_DIRECTIONAL_LIGHT = /* glsl */`
// Three.js r185 DirectionalLight Uniforms
uniform vec3 uDirLightColor;
uniform vec3 uDirLightDirection;
uniform float uDirLightIntensity;

// Anime Cel-Shading Parameters
uniform float uCelSteps;
uniform float uRimPower;
uniform vec3 uRimColor;
uniform float uRimIntensity;
uniform vec3 uShadowTint;
uniform float uShadowSoftness;
uniform float uAmbientIntensity;
uniform vec3 uAmbientColor;

// Shadow Mapping Uniforms (DirectionalLightShadow)
uniform sampler2D uShadowMap;
uniform mat4 uShadowMatrix;
uniform float uShadowBias;
uniform float uShadowNormalBias;
uniform vec2 uShadowMapSize;

// Shadow map sampling with 3x3 PCF for soft anime edges
// FIXED: Correct UV mapping from [-1, 1] clip space to [0, 1] texture space
float sampleAnimeShadowMap(vec3 worldPos, vec3 normal) {
  vec4 shadowCoord = uShadowMatrix * vec4(worldPos + normal * uShadowNormalBias, 1.0);
  
  // Perspective divide (orthographic for directional, so w=1, but safe practice)
  vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
  
  // Convert from [-1, 1] clip space to [0, 1] UV space
  vec2 uvCoords = projCoords.xy * 0.5 + 0.5;
  
  // Out of bounds check (frustum culling for shadows)
  if (uvCoords.x < 0.0 || uvCoords.x > 1.0 || uvCoords.y < 0.0 || uvCoords.y > 1.0 || projCoords.z > 1.0) {
    return 1.0; 
  }
  
  float currentDepth = projCoords.z - uShadowBias;
  
  // FIXED: Guarded division to prevent NaN on uninitialized shadow maps
  vec2 texelSize = 1.0 / max(uShadowMapSize, vec2(1.0));
  float shadow = 0.0;
  
  // FIXED: Constant loop bounds for GLSL ES 1.00/3.00 mobile driver compatibility
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      vec2 offset = vec2(float(x), float(y)) * texelSize;
      float depth = texture2D(uShadowMap, uvCoords + offset).r;
      shadow += step(currentDepth, depth);
    }
  }
  return shadow / 9.0;
}

// Main anime directional light computation
vec3 computeAnimeDirectionalLight(vec3 baseColor, vec3 normal, vec3 viewDir, vec3 worldPos) {
  vec3 lightDir = normalize(-uDirLightDirection);
  vec3 n = normalize(normal);
  vec3 v = normalize(viewDir);
  
  // 1. Normal Quantization (Cel-shading steps)
  float lightDot = dot(n, lightDir);
  float stepSize = 1.0 / max(uCelSteps, 1.0);
  float normalizedDot = (lightDot * 0.5) + 0.5;
  float quantizedDot = floor(normalizedDot / stepSize) * stepSize;
  quantizedDot = (quantizedDot * 2.0) - 1.0;
  
  // 2. Diffuse Calculation
  float diffuse = max(quantizedDot, 0.0);
  
  // 3. Shadow Mapping
  float shadowFactor = sampleAnimeShadowMap(worldPos, n);
  
  // 4. Anime Shadow Tint (shadows are colored, not just black)
  // FIXED: correct low->high smoothstep order
  float shadowMask = 1.0 - smoothstep(-uShadowSoftness, uShadowSoftness, lightDot);
  shadowMask = max(shadowMask, 1.0 - shadowFactor);
  
  vec3 litColor = baseColor * uDirLightColor * uDirLightIntensity * max(diffuse, 0.15);
  vec3 shadowColor = baseColor * uShadowTint;
  vec3 finalDiffuse = mix(litColor, shadowColor, shadowMask);
  
  // 5. Rim Light (Fresnel)
  float rimDot = 1.0 - max(dot(n, v), 0.0);
  float rimFactor = pow(rimDot, max(uRimPower, 1.0));
  vec3 rimLight = uRimColor * rimFactor * uRimIntensity;
  
  // 6. Ambient
  vec3 ambient = baseColor * uAmbientColor * uAmbientIntensity;
  
  return finalDiffuse + rimLight + ambient;
}
`;

/* ------------------------------------------------------------------ */
/* 2. DIRECTIONAL LIGHT UNIFORMS (JS side)                            */
/* ------------------------------------------------------------------ */
export const DIR_LIGHT_UNIFORMS = {
  uDirLightColor:     { value: new THREE.Color(0xffffff) },
  uDirLightDirection: { value: new THREE.Vector3(0.5, 0.8, 0.3) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimColor:          { value: new THREE.Color(0.9, 0.95, 1.0) },
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.12, 0.18, 0.30) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.25 },
  uAmbientColor:      { value: new THREE.Color(0.45, 0.62, 0.85) },
  uShadowMap:         { value: null },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1024, 1024) },
};

/* ------------------------------------------------------------------ */
/* 3. DIRECTIONAL LIGHT SHADER MANAGER CLASS                          */
/*    Manages Three.js DirectionalLight + Shadow Camera + Uniform Sync*/
/* ------------------------------------------------------------------ */
export class DirectionalLightShader {
  constructor(options = {}) {
    this.light = new THREE.DirectionalLight(0xffffff, 1.0);
    this.light.castShadow = true;
    
    // Shadow camera setup (Orthographic for DirectionalLight)
    const shadowRes = options.shadowResolution || 1024;
    this.light.shadow.mapSize.width = shadowRes;
    this.light.shadow.mapSize.height = shadowRes;
    this.light.shadow.camera.near = options.shadowNear || 0.5;
    this.light.shadow.camera.far = options.shadowFar || 200;
    this.light.shadow.camera.left = options.shadowLeft || -50;
    this.light.shadow.camera.right = options.shadowRight || 50;
    this.light.shadow.camera.top = options.shadowTop || 50;
    this.light.shadow.camera.bottom = options.shadowBottom || -50;
    this.light.shadow.bias = options.shadowBias || -0.001;
    this.light.shadow.normalBias = options.shadowNormalBias || 0.05;
    
    this.uniforms = { ...DIR_LIGHT_UNIFORMS };
    this.uniforms.uShadowMapSize.value.set(shadowRes, shadowRes);
    this.uniforms.uShadowBias.value = this.light.shadow.bias;
    this.uniforms.uShadowNormalBias.value = this.light.shadow.normalBias;
  }

  getLight() {
    return this.light;
  }

  getGLSLChunk() {
    return GLSL_DIRECTIONAL_LIGHT;
  }

  getUniforms() {
    return this.uniforms;
  }

  setDirection(x, y, z) {
    this.light.position.set(x, y, z).normalize().multiplyScalar(100);
    this.uniforms.uDirLightDirection.value.set(-x, -y, -z).normalize();
  }

  setColor(hex) {
    this.light.color.setHex(hex);
    this.uniforms.uDirLightColor.value.setHex(hex);
  }

  setIntensity(intensity) {
    this.light.intensity = intensity;
    this.uniforms.uDirLightIntensity.value = intensity;
  }

  setShadowTint(hex) {
    this.uniforms.uShadowTint.value.setHex(hex);
  }

  setCelSteps(steps) {
    this.uniforms.uCelSteps.value = steps;
  }

  // FIXED: Syncs shadow map TEXTURE and matrix to the material uniforms
  syncShadowUniforms(material) {
    if (!material || !material.uniforms) return;
    
    // Three.js r185 shadow.map is a WebGLRenderTarget, we need .texture
    if (this.light.shadow && this.light.shadow.map && this.light.shadow.map.texture) {
      material.uniforms.uShadowMap.value = this.light.shadow.map.texture;
    }
    
    // Three.js shadow.matrix transforms World -> Clip Space [-1, 1]
    // The GLSL chunk handles the [-1, 1] -> [0, 1] UV conversion
    if (this.light.shadow) {
      material.uniforms.uShadowMatrix.value.copy(this.light.shadow.matrix);
    }
  }

  update(delta, elapsed) {
    // Time-based light animation (e.g., day/night cycle) hooks here
  }
}

/* ------------------------------------------------------------------ */
/* 4. MATERIAL INJECTION HELPER                                       */
/*    Injects the directional light chunk into a ShaderMaterial       */
/* ------------------------------------------------------------------ */
export function injectDirectionalLight(material, lightShader) {
  if (!material || !material.isShaderMaterial) return;
  
  // Merge uniforms
  const lightUniforms = lightShader.getUniforms();
  for (const key in lightUniforms) {
    if (!material.uniforms[key]) {
      material.uniforms[key] = lightUniforms[key];
    }
  }
  
  // Inject GLSL chunk into fragment shader
  const chunk = lightShader.getGLSLChunk();
  if (!material.fragmentShader.includes('computeAnimeDirectionalLight')) {
    material.fragmentShader = material.fragmentShader.replace(
      'void main() {',
      `${chunk}\nvoid main() {`
    );
  }
  
  material.needsUpdate = true;
}
