// File : 002
// name : shaders/002_SpecklesShader.glsl.js
// description : Procedural path speckles shader (pebbles, small rocks, and grass tufts).
//               Features distinct procedural shapes for pebbles (noise-distorted circles)
//               and grass tufts (blended blade smoothsteps). Integrates dynamic wind sway
//               in the vertex shader, cel-shaded lighting, and pixel-art quantization.
//               Fully composed using 000_BaseShader.glsl.js and GLSL_COLOR_UTILS from
//               004_ColorPalette.js to prevent redefinition errors. Optimized for Android
//               mobile with low-octave noise and zero-texture GPU instancing.
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
/* 3. SPECKLES UNIFORMS (JS side)                                      */
/* ------------------------------------------------------------------ */
export const SPECKLES_UNIFORMS = {
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
  uRimStrength:      { value: 0.55 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Speckle specific colors (exact image palette)
  uColPebbleDark:  { value: new THREE.Vector3(0.353, 0.478, 0.290) }, // 0x5a7a4a
  uColPebbleLight: { value: new THREE.Vector3(0.561, 0.820, 0.310) }, // 0x8fd14f
  uColGrassDark:   { value: new THREE.Vector3(0.220, 0.360, 0.160) }, // 0x385c29
  uColGrassLight:  { value: new THREE.Vector3(0.450, 0.720, 0.280) }, // 0x73b847
  uWindStrength:   { value: 0.8 },
};

/* ------------------------------------------------------------------ */
/* 4. SPECKLES VERTEX SHADER                                           */
/* ------------------------------------------------------------------ */
export const SPECKLES_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = pebble, 1.0 = grass

uniform vec2 uWind;
uniform float uWindStrength;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;
varying float vWindSway;

void main() {
  vUv   = position.xy + 0.5;
  vSeed = aSeed;
  vTint = aTint;

  float c = cos(aRot);
  float s = sin(aRot);
  vec2 p = vec2(position.x * c - position.y * s, position.x * s + position.y * c) * aSize;
  
  // Pixel-art quantization on world position
  vec2 w = pxq(aOff + p, uPPU);
  
  // Wind sway (only affects grass tufts where aTint > 0.5)
  float isGrass = step(0.5, vTint);
  float swayAmount = isGrass * uWindStrength * 0.35;
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.5 + aSeed * 31.4 + vUv.y * 4.0) * vUv.y;
  w += windOffset;
  
  vWindSway = swayAmount;
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. SPECKLES FRAGMENT SHADER                                         */
/* ------------------------------------------------------------------ */
export const SPECKLES_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColPebbleDark;
uniform vec3  uColPebbleLight;
uniform vec3  uColGrassDark;
uniform vec3  uColGrassLight;
uniform float uWindStrength;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;
varying float vWindSway;

// Procedural pebble shape (noise-distorted circle)
float pebbleShape(vec2 uv, float seed) {
  vec2 c = uv - 0.5;
  float angle = atan(c.y, c.x);
  float radius = 0.38 + sin(angle * 5.0 + seed * 7.0) * 0.06 + sin(angle * 9.0 + seed * 11.0) * 0.03;
  float d = length(c);
  return smoothstep(radius + 0.05, radius - 0.05, d);
}

// Procedural grass tuft shape (multiple blended blades)
float grassShape(vec2 uv, float seed) {
  float blade1 = smoothstep(0.05, 0.0, abs(uv.x - 0.30) - uv.y * 0.15);
  float blade2 = smoothstep(0.04, 0.0, abs(uv.x - 0.60) - uv.y * 0.12);
  float blade3 = smoothstep(0.04, 0.0, abs(uv.x - 0.50) - uv.y * 0.18);
  float blades = max(max(blade1, blade2), blade3);
  blades *= smoothstep(0.95, 0.40, uv.y); // Taper at top
  return blades;
}

void main() {
  // 1. Determine shape based on instance tint (pebble vs grass)
  float isGrass = step(0.5, vTint);
  float shape = mix(pebbleShape(vUv, vSeed), grassShape(vUv, vSeed), isGrass);
  
  if (shape < 0.05) discard;

  // 2. Base color mixing
  vec3 colPebble = mix(uColPebbleDark, uColPebbleLight, vSeed);
  vec3 colGrass  = mix(uColGrassDark, uColGrassLight, vSeed);
  vec3 baseColor = mix(colPebble, colGrass, isGrass);

  // 3. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);

  // 4. Grass-specific lighting tweaks (brighter tips)
  float tipHighlight = smoothstep(0.6, 0.9, vUv.y) * isGrass;
  litColor += vec3(0.15, 0.25, 0.10) * tipHighlight;

  // 5. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.94 + pxNoise * 0.12;

  // 6. Edge softening for grass
  float alpha = shape;
  if (isGrass > 0.5) {
    alpha *= smoothstep(0.0, 0.15, shape);
  }

  gl_FragColor = vec4(litColor, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createSpecklesMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: SPECKLES_UNIFORMS,
    vertexShader: SPECKLES_VERTEX_SHADER,
    fragmentShader: SPECKLES_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false, // Orthographic painter's algorithm
    side: THREE.DoubleSide,
  });
}
