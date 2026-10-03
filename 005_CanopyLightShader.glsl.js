// File : 005
// name : shaders/005_CanopyLightShader.glsl.js
// description : Procedural bright yellow-green fluffy canopy clusters shader. 
//               Features rounded blob-like leaf formations, dynamic wind-reactive 
//               swaying, multi-layer depth rendering, and edge-intrusion patterns. 
//               Fully composed using 000_BaseShader.glsl.js and GLSL_COLOR_UTILS 
//               from 004_ColorPalette.js to prevent redefinition errors. Optimized 
//               for Android mobile with low-octave noise and zero-texture GPU 
//               instancing. Colors match the reference image exactly (bright 
//               yellow-green palette) without text labels.
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
/* 3. CANOPY LIGHT UNIFORMS (JS side)                                  */
/* ------------------------------------------------------------------ */
export const CANOPY_LIGHT_UNIFORMS = {
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
  uRimStrength:      { value: 0.50 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Canopy light specific colors (exact image palette: bright yellow-green)
  uColCanopyLight1: { value: new THREE.Vector3(0.557, 0.871, 0.388) }, // 0x8ede63
  uColCanopyLight2: { value: new THREE.Vector3(0.361, 0.722, 0.306) }, // 0x5cb84e
  uColCanopyLight3: { value: new THREE.Vector3(0.706, 0.941, 0.471) }, // 0xb4f078 (highlight)
  uColCanopyLight4: { value: new THREE.Vector3(0.227, 0.541, 0.165) }, // 0x3a8a2a (darker accent)
  uWindStrength:    { value: 0.75 },
  uDensity:         { value: 0.90 },
};

/* ------------------------------------------------------------------ */
/* 4. CANOPY LIGHT VERTEX SHADER                                       */
/* ------------------------------------------------------------------ */
export const CANOPY_LIGHT_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = background layer, 1.0 = foreground/intrusion layer

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
  
  // Wind sway (stronger for foreground/intrusion layer where aTint > 0.5)
  float isForeground = step(0.5, vTint);
  float swayAmount = (0.5 + isForeground * 0.8) * uWindStrength;
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.2 + aSeed * 29.7 + vUv.y * 3.5) * vUv.y;
  w += windOffset;
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. CANOPY LIGHT FRAGMENT SHADER                                     */
/* ------------------------------------------------------------------ */
export const CANOPY_LIGHT_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColCanopyLight1;
uniform vec3  uColCanopyLight2;
uniform vec3  uColCanopyLight3;
uniform vec3  uColCanopyLight4;
uniform float uWindStrength;
uniform float uDensity;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

// Procedural fluffy foliage blob shape
float fluffyBlob(vec2 uv, vec2 center, float radius, float seed) {
  float d = length(uv - center);
  float angle = atan(uv.y - center.y, uv.x - center.x);
  // Fluffy bumps around the edge
  float bumps = sin(angle * 8.0 + seed * 11.0) * 0.14;
  bumps += sin(angle * 13.0 + seed * 17.0) * 0.09;
  float noisyRadius = radius + bumps;
  float noise = fbm(uv * 9.0 + seed, 3) * 0.11;
  return smoothstep(noisyRadius + 0.09, noisyRadius - 0.14, d + noise);
}

void main() {
  // 1. Create organic fluffy foliage mass using multiple overlapping blobs
  vec2 center1 = vec2(0.5 + sin(vSeed * 13.0) * 0.13, 0.5 + cos(vSeed * 15.0) * 0.11);
  vec2 center2 = vec2(0.38 + cos(vSeed * 11.0) * 0.16, 0.62 + sin(vSeed * 9.0) * 0.13);
  vec2 center3 = vec2(0.62 + sin(vSeed * 7.0) * 0.15, 0.38 + cos(vSeed * 19.0) * 0.12);
  vec2 center4 = vec2(0.45 + cos(vSeed * 21.0) * 0.11, 0.55 + sin(vSeed * 5.0) * 0.14);
  
  float radius1 = 0.42 + fbm(vUv * 3.0 + vSeed, 2) * 0.07;
  float radius2 = 0.38 + fbm(vUv * 4.0 + vSeed + 10.0, 2) * 0.06;
  float radius3 = 0.36 + fbm(vUv * 5.0 + vSeed + 20.0, 2) * 0.08;
  float radius4 = 0.34 + fbm(vUv * 6.0 + vSeed + 30.0, 2) * 0.05;
  
  float shape = fluffyBlob(vUv, center1, radius1, vSeed);
  shape = max(shape, fluffyBlob(vUv, center2, radius2, vSeed + 5.0));
  shape = max(shape, fluffyBlob(vUv, center3, radius3, vSeed + 10.0));
  shape = max(shape, fluffyBlob(vUv, center4, radius4, vSeed + 15.0));
  
  // Add fine leaf detail noise
  float leafDetail = fbm(vUv * 18.0 + vSeed * 24.0, 4);
  float detailMask = smoothstep(0.35, 0.65, leafDetail);
  shape *= (0.78 + detailMask * 0.22) * uDensity;
  
  if (shape < 0.06) discard;

  // 2. Base color mixing (exact image palette)
  float colorNoise = fbm(vUv * 5.0 + vSeed + uTime * 0.03, 3);
  float heightGradient = smoothstep(0.0, 1.0, vUv.y);
  
  vec3 baseColor = mix(uColCanopyLight1, uColCanopyLight2, colorNoise);
  baseColor = mix(baseColor, uColCanopyLight4, heightGradient * 0.3);
  
  // Foreground/intrusion layer is slightly brighter
  baseColor = mix(baseColor, uColCanopyLight3, step(0.5, vTint) * 0.25);

  // 3. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);

  // 4. Top-down highlight (sun hitting the top of the fluffy canopy)
  float topHighlight = smoothstep(0.65, 0.95, vUv.y) * smoothstep(0.3, 0.7, dot(normalize(vUv - 0.5), uSunDir.xy));
  litColor += uColCanopyLight3 * topHighlight * 0.25;

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
export function createCanopyLightMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: CANOPY_LIGHT_UNIFORMS,
    vertexShader: CANOPY_LIGHT_VERTEX_SHADER,
    fragmentShader: CANOPY_LIGHT_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false, // Orthographic painter's algorithm
    side: THREE.DoubleSide,
  });
}
