// File : 004
// name : shaders/004_CanopyDarkShader.glsl.js
// description : Procedural deep-green fractal foliage mass shader. Features organic 
//               metaball-like leaf clusters, cel-shaded lighting with strong top-down 
//               highlights, dynamic wind-reactive swaying in the vertex shader, and 
//               pixel-art quantization. Fully composed using 000_BaseShader.glsl.js 
//               and GLSL_COLOR_UTILS from 004_ColorPalette.js to prevent redefinition 
//               errors. Optimized for Android mobile with low-octave noise and zero-
//               texture GPU instancing. Colors match the reference image exactly 
//               (deep forest green palette) without text labels.
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
/* ------------------------------------------------------------------ */
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';

/* ------------------------------------------------------------------ */
/* 3. CANOPY DARK UNIFORMS (JS side)                                   */
/* ------------------------------------------------------------------ */
export const CANOPY_DARK_UNIFORMS = {
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
  uRimStrength:      { value: 0.45 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Canopy dark specific colors (exact image palette: deep forest green)
  uColCanopyDark1: { value: new THREE.Vector3(0.118, 0.275, 0.204) }, // 0x1e4634
  uColCanopyDark2: { value: new THREE.Vector3(0.165, 0.361, 0.251) }, // 0x2a5c40
  uColCanopyDark3: { value: new THREE.Vector3(0.078, 0.196, 0.137) }, // 0x143223
  uWindStrength:   { value: 0.6 },
  uDensity:        { value: 0.85 },
};

/* ------------------------------------------------------------------ */
/* 4. CANOPY DARK VERTEX SHADER                                        */
/* ------------------------------------------------------------------ */
export const CANOPY_DARK_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = background layer, 1.0 = foreground layer

uniform vec2 uWind;
uniform float uWindStrength;

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
  
  // Pixel-art quantization on world position
  vec2 w = pxq(aOff + p, uPPU);
  
  // Wind sway (stronger for foreground layer where aTint > 0.5)
  float isForeground = step(0.5, vTint);
  float swayAmount = (0.4 + isForeground * 0.6) * uWindStrength;
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.0 + aSeed * 27.3 + vUv.y * 3.0) * vUv.y;
  w += windOffset;
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. CANOPY DARK FRAGMENT SHADER                                      */
/* ------------------------------------------------------------------ */
export const CANOPY_DARK_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColCanopyDark1;
uniform vec3  uColCanopyDark2;
uniform vec3  uColCanopyDark3;
uniform float uWindStrength;
uniform float uDensity;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

// Procedural organic foliage blob shape
float foliageBlob(vec2 uv, vec2 center, float radius, float seed) {
  float d = length(uv - center);
  float angle = atan(uv.y - center.y, uv.x - center.x);
  // Fluffy bumps around the edge
  float bumps = sin(angle * 7.0 + seed * 9.0) * 0.12;
  bumps += sin(angle * 11.0 + seed * 13.0) * 0.08;
  float noisyRadius = radius + bumps;
  float noise = fbm(uv * 8.0 + seed, 3) * 0.10;
  return smoothstep(noisyRadius + 0.08, noisyRadius - 0.12, d + noise);
}

void main() {
  // 1. Create organic foliage mass using multiple overlapping blobs
  vec2 center1 = vec2(0.5 + sin(vSeed * 11.0) * 0.12, 0.5 + cos(vSeed * 13.0) * 0.10);
  vec2 center2 = vec2(0.38 + cos(vSeed * 9.0) * 0.15, 0.62 + sin(vSeed * 7.0) * 0.12);
  vec2 center3 = vec2(0.62 + sin(vSeed * 5.0) * 0.14, 0.38 + cos(vSeed * 17.0) * 0.11);
  vec2 center4 = vec2(0.45 + cos(vSeed * 19.0) * 0.10, 0.55 + sin(vSeed * 3.0) * 0.13);
  
  float radius1 = 0.40 + fbm(vUv * 3.0 + vSeed, 2) * 0.06;
  float radius2 = 0.36 + fbm(vUv * 4.0 + vSeed + 10.0, 2) * 0.05;
  float radius3 = 0.34 + fbm(vUv * 5.0 + vSeed + 20.0, 2) * 0.07;
  float radius4 = 0.32 + fbm(vUv * 6.0 + vSeed + 30.0, 2) * 0.04;
  
  float shape = foliageBlob(vUv, center1, radius1, vSeed);
  shape = max(shape, foliageBlob(vUv, center2, radius2, vSeed + 5.0));
  shape = max(shape, foliageBlob(vUv, center3, radius3, vSeed + 10.0));
  shape = max(shape, foliageBlob(vUv, center4, radius4, vSeed + 15.0));
  
  // Add fine leaf detail noise
  float leafDetail = fbm(vUv * 16.0 + vSeed * 22.0, 4);
  float detailMask = smoothstep(0.35, 0.65, leafDetail);
  shape *= (0.75 + detailMask * 0.25) * uDensity;
  
  if (shape < 0.06) discard;

  // 2. Base color mixing (exact image palette)
  float colorNoise = fbm(vUv * 5.0 + vSeed + uTime * 0.02, 3);
  float heightGradient = smoothstep(0.0, 1.0, vUv.y);
  
  vec3 baseColor = mix(uColCanopyDark1, uColCanopyDark2, colorNoise);
  baseColor = mix(baseColor, uColCanopyDark3, heightGradient * 0.4);
  
  // Foreground layer is slightly darker/richer
  baseColor = mix(baseColor, baseColor * 0.9, step(0.5, vTint) * 0.2);

  // 3. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);

  // 4. Top-down highlight (sun hitting the top of the canopy)
  float topHighlight = smoothstep(0.6, 0.9, vUv.y) * smoothstep(0.3, 0.7, dot(normalize(vUv - 0.5), uSunDir.xy));
  litColor += uColCanopyDark2 * topHighlight * 0.15;

  // 5. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 14.0, 7.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.94 + pxNoise * 0.12;

  // 6. Edge softening for foliage
  float alpha = shape;
  alpha *= smoothstep(0.0, 0.15, shape);

  gl_FragColor = vec4(litColor, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createCanopyDarkMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: CANOPY_DARK_UNIFORMS,
    vertexShader: CANOPY_DARK_VERTEX_SHADER,
    fragmentShader: CANOPY_DARK_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false, // Orthographic painter's algorithm
    side: THREE.DoubleSide,
  });
}
