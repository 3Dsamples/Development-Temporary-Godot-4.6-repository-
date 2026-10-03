// File : 003
// name : shaders/003_ShadowPoolsShader.glsl.js
// description : Procedural shadow pools shader (desaturated teal ground blobs).
//               Features organic metaball-like shapes with noise-distorted edges,
//               subtle breathing (pulse) animation, and wind-reactive softening.
//               Fully composed using 000_BaseShader.glsl.js and GLSL_COLOR_UTILS
//               from 004_ColorPalette.js to prevent redefinition errors. Optimized
//               for Android mobile with low-octave noise and zero-texture GPU instancing.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

/* ------------------------------------------------------------------ */
/* 1. BASE SHADER CHUNKS (Injected from 000_BaseShader.glsl.js)        */
/* ------------------------------------------------------------------ */
import {
  GLSL_GLOBALS,
  GLSL_NOISE,
  GLSL_NORMAL_QUANT,
  GLSL_LIGHTING,
  GLSL_BIOME,
} from './000_BaseShader.glsl.js';

/* ------------------------------------------------------------------ */
/* 2. COLOR UTILITIES (From 004_ColorPalette.js)                       */
/*    Required for computeCelLighting (applyShadowTint, applyRimLight) */
/* ------------------------------------------------------------------ */
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';

/* ------------------------------------------------------------------ */
/* 3. SHADOW POOLS UNIFORMS (JS side)                                  */
/* ------------------------------------------------------------------ */
export const SHADOW_POOLS_UNIFORMS = {
  // Base globals
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  
  // Lighting (synced by LightingSystem.js)
  uSunDir:           { value: new THREE.Vector3(0.5, 0.8, 0.3) },
  uMoonDir:          { value: new THREE.Vector3(-0.5, -0.8, -0.3) },
  uSunColor:         { value: new THREE.Vector3(1.0, 0.96, 0.85) },
  uMoonColor:        { value: new THREE.Vector3(0.42, 0.48, 0.70) },
  uSkyColor:         { value: new THREE.Vector3(0.45, 0.62, 0.85) },
  uGroundColor:      { value: new THREE.Vector3(0.02, 0.12, 0.25) },
  uFogColor:         { value: new THREE.Vector3(0.10, 0.30, 0.50) },
  uShadowTintColor:  { value: new THREE.Vector3(0.12, 0.18, 0.30) },
  uRimColor:         { value: new THREE.Vector3(0.90, 0.95, 1.00) },
  uSunIntensity:     { value: 1.0 },
  uMoonIntensity:    { value: 0.35 },
  uSkyStrength:      { value: 0.35 },
  uGroundStrength:   { value: 0.15 },
  uBounceStrength:   { value: 0.30 },
  uRimStrength:      { value: 0.20 }, // Reduced rim for shadows
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Shadow specific colors (exact image palette: desaturated teal)
  uColShadowBase: { value: new THREE.Vector3(0.278, 0.408, 0.369) }, // 0x47685e
  uColShadowDeep: { value: new THREE.Vector3(0.180, 0.294, 0.267) }, // 0x2e4b44
  uPulseSpeed:    { value: 0.8 },
  uEdgeSoftness:  { value: 0.65 },
};

/* ------------------------------------------------------------------ */
/* 4. SHADOW POOLS VERTEX SHADER                                       */
/* ------------------------------------------------------------------ */
export const SHADOW_POOLS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = small/fast pulse, 1.0 = large/slow pulse

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;
varying float vPulsePhase;

void main() {
  vUv   = position.xy + 0.5;
  vSeed = aSeed;
  vTint = aTint;
  vPulsePhase = aSeed * 6.2831;

  float c = cos(aRot);
  float s = sin(aRot);
  vec2 p = vec2(position.x * c - position.y * s, position.x * s + position.y * c) * aSize;
  
  // Breathing animation (subtle size pulsing)
  float pulseSpeed = mix(0.6, 1.2, vTint);
  float pulse = sin(uTime * uPulseSpeed * pulseSpeed + vPulsePhase) * 0.06;
  vec2 animSize = aSize * (1.0 + pulse);
  p = vec2(position.x * c - position.y * s, position.x * s + position.y * c) * animSize;

  // Pixel-art quantization on world position
  vec2 w = pxq(aOff + p, uPPU);
  
  // Wind softening (slight positional jitter for edges)
  float windStrength = length(uWind) * 0.15;
  vec2 windOffset = uWind * windStrength * sin(uTime * 1.2 + aSeed * 15.0);
  w += windOffset;
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. SHADOW POOLS FRAGMENT SHADER                                     */
/* ------------------------------------------------------------------ */
export const SHADOW_POOLS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColShadowBase;
uniform vec3  uColShadowDeep;
uniform float uEdgeSoftness;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;
varying float vPulsePhase;

// Metaball-like organic shape function
float metaball(vec2 uv, vec2 center, float radius, float seed) {
  float d = length(uv - center);
  // Noise distortion for organic edges
  float noise = fbm(uv * 3.0 + seed, 2) * 0.12;
  return smoothstep(radius + noise + uEdgeSoftness, radius + noise - uEdgeSoftness, d);
}

void main() {
  // 1. Create organic blob shape using multiple overlapping metaballs
  vec2 c1 = vec2(0.5 + sin(vSeed * 10.0) * 0.12, 0.5 + cos(vSeed * 13.0) * 0.10);
  vec2 c2 = vec2(0.4 + cos(vSeed * 7.0) * 0.15, 0.6 + sin(vSeed * 11.0) * 0.12);
  vec2 c3 = vec2(0.6 + sin(vSeed * 9.0) * 0.12, 0.4 + cos(vSeed * 8.0) * 0.10);
  
  float r1 = 0.35 + fbm(vUv * 2.0 + vSeed, 2) * 0.08;
  float r2 = 0.28 + fbm(vUv * 2.5 + vSeed + 10.0, 2) * 0.06;
  float r3 = 0.25 + fbm(vUv * 3.0 + vSeed + 20.0, 2) * 0.07;
  
  float shape = metaball(vUv, c1, r1, vSeed);
  shape = max(shape, metaball(vUv, c2, r2, vSeed + 5.0));
  shape = max(shape, metaball(vUv, c3, r3, vSeed + 10.0));
  
  // 2. Edge softening and fade
  shape = smoothstep(0.1, 0.5, shape);
  if (shape < 0.02) discard;

  // 3. Base color mixing (exact image palette)
  float depthNoise = fbm(vUv * 4.0 + vSeed * 20.0, 3);
  vec3 col = mix(uColShadowBase, uColShadowDeep, depthNoise * 0.65);
  
  // 4. Cel-shaded lighting (subtle, shadows are mostly ambient)
  vec3 normal = vec3(0.0, 0.0, 1.0);
  col = computeCelLighting(col, normal, uViewDir);

  // 5. Darken center, lighter edges (typical shadow gradient)
  float gradient = 1.0 - smoothstep(0.0, 0.7, length(vUv - 0.5));
  col *= 0.75 + gradient * 0.25;

  // 6. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 12.0, 6.0);
  float pxNoise = h21(pUv + vSeed);
  col *= 0.94 + pxNoise * 0.12;

  // 7. Final alpha with breathing pulse modulation
  float pulse = sin(uTime * uPulseSpeed + vPulsePhase) * 0.08;
  float alpha = shape * (0.85 + pulse);

  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createShadowPoolsMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: SHADOW_POOLS_UNIFORMS,
    vertexShader: SHADOW_POOLS_VERTEX_SHADER,
    fragmentShader: SHADOW_POOLS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false, // Orthographic painter's algorithm
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending, // Shadows blend normally over ground
  });
}
