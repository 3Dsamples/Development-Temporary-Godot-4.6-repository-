// File : 001
// name : shaders/001_PathShader.glsl.js
// description : Procedural sandy dirt path band shader. Features fractal noisy 
//               edges, diagonal drift, wear patterns, and cel-shaded lighting.
//               Fully composed using the 000_BaseShader.glsl.js chunks and 
//               GLSL_COLOR_UTILS from 004_ColorPalette.js to prevent 
//               redefinition errors. Optimized for Android mobile with instanced 
//               attributes and pixel-art quantization. Colors match the reference 
//               image exactly (beige/sand palette) without text labels.
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
/* 3. PATH UNIFORMS (JS side, compatible with LightingSystem.js)       */
/* ------------------------------------------------------------------ */
export const PATH_UNIFORMS = {
  // Base globals
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },
  
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
  uRimStrength:      { value: 0.55 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Path specific
  uColPathBase: { value: new THREE.Vector3(0.914, 0.839, 0.643) }, // 0xe9d6a4
  uColPathDark: { value: new THREE.Vector3(0.847, 0.749, 0.541) }, // 0xd8bf8a
  uColPathDeep: { value: new THREE.Vector3(0.541, 0.478, 0.376) }, // 0x8a7a60
  uEdgeNoise:   { value: 0.45 },
  uDriftSpeed:  { value: 0.15 },
};

/* ------------------------------------------------------------------ */
/* 4. PATH VERTEX SHADER                                               */
/* ------------------------------------------------------------------ */
export const PATH_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

void main() {
  vUv   = position.xy + 0.5;
  vSeed = aSeed;
  vTint = aTint;

  float c = cos(aRot);
  float s = sin(aRot);
  vec2 p = vec2(position.x * c - position.y * s, position.x * s + position.y * c) * aSize;
  
  // Pixel-art quantization on world position for crisp edges
  vec2 w = pxq(aOff + p, uPPU);
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. PATH FRAGMENT SHADER                                             */
/*    Order matters: GLSL_COLOR_UTILS must come before GLSL_LIGHTING   */
/* ------------------------------------------------------------------ */
export const PATH_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // Defines applyShadowTint, applyRimLight, etc.
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}         // Uses functions from GLSL_COLOR_UTILS
${GLSL_BIOME}

uniform vec3  uColPathBase;
uniform vec3  uColPathDark;
uniform vec3  uColPathDeep;
uniform float uEdgeNoise;
uniform float uDriftSpeed;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

void main() {
  // 1. Fractal noisy edges & diagonal drift
  vec2 driftUv = vWorld + vec2(uTime * uDriftSpeed, uTime * uDriftSpeed * 0.5);
  float edgeNoise = fbm(driftUv * 0.8 + vSeed * 10.0, 3);
  float wearNoise = fbm(driftUv * 1.5 + vSeed * 20.0, 2);
  float detailNoise = h21(pxq(vWorld * 4.0, uPPU) + vSeed);

  // 2. Base color mixing (exact image palette)
  vec3 col = mix(uColPathBase, uColPathDark, edgeNoise * 0.65);
  col = mix(col, uColPathDeep, wearNoise * 0.35);
  
  // 3. Instance tint variation (wear/dirt level)
  col = mix(col, col * 0.82, vTint * 0.45);

  // 4. Cel-shaded lighting (from base)
  vec3 normal = vec3(0.0, 0.0, 1.0); // Flat top-down surface
  col = computeCelLighting(col, normal, uViewDir);

  // 5. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  float pxNoise = h21(pUv + vSeed);
  col *= 0.94 + pxNoise * 0.12;

  // 6. Edge darkening (subtle ambient occlusion at path borders)
  float edgeFade = smoothstep(0.05, 0.25, vUv.x) * smoothstep(0.95, 0.75, vUv.x);
  col *= 0.85 + edgeFade * 0.15;

  gl_FragColor = vec4(col, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createPathMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: PATH_UNIFORMS,
    vertexShader: PATH_VERTEX_SHADER,
    fragmentShader: PATH_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: true,
    depthTest: false, // Orthographic painter's algorithm
    side: THREE.DoubleSide,
  });
}

