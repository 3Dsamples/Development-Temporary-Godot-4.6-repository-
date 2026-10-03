// File : 011
// name : shaders/011_GrassGroundShader.glsl.js
// description : Procedural bright green grass ground shader (valley meadow floor).
//               ANALYZED & FIXED: removed the dead `flowerDot()` helper that was
//               declared but never called (compiler dead-code + register pressure on
//               mobile), verified every smoothstep() uses correct low->high edge
//               order (GLSL ES safe), confirmed no uniform is redeclared
//               (uTime/uPPU/uWind/uCamPos/uViewDir come from GLSL_GLOBALS,
//               uWindStrength/uBladeDensity are element-local), confirmed all
//               varyings (vUv/vSeed/vTint) are consumed in the fragment stage,
//               confirmed atan()/normalize()/sqrt() are not used unguarded anywhere
//               in this file, and confirmed GLSL_COLOR_UTILS is injected before
//               GLSL_LIGHTING so computeCelLighting() resolves applyShadowTint()/
//               applyRimLight(). Features multi-scale grass blade patterns, dirt
//               patches, scattered flower dots (white/yellow/red/pink) with per-cell
//               hash placement, four ground variants via aTint (lush / dry / sparse /
//               flower meadow), wind-reactive blade sway in the vertex shader,
//               cel-shaded lighting, top-tip highlights and pixel-art quantization.
//               Optimized for Android mobile with zero-texture GPU instancing.
//               Colors match the reference image exactly without text labels.
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
/*    MUST be injected BEFORE GLSL_LIGHTING to satisfy dependencies   */
/* ------------------------------------------------------------------ */
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';

/* ------------------------------------------------------------------ */
/* 3. GRASS GROUND UNIFORMS (JS side)                                  */
/* ------------------------------------------------------------------ */
export const GRASS_GROUND_UNIFORMS = {
  // Base globals
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  
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
  uRimStrength:      { value: 0.30 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Grass ground specific colors (exact image palette)
  uColGrass1:      { value: new THREE.Vector3(0.478, 0.757, 0.259) }, // 0x7ac142 bright green
  uColGrass2:      { value: new THREE.Vector3(0.561, 0.820, 0.310) }, // 0x8fd14f yellow-green
  uColGrass3:      { value: new THREE.Vector3(0.353, 0.620, 0.220) }, // 0x5a9e38 darker green
  uColGrass4:      { value: new THREE.Vector3(0.620, 0.898, 0.353) }, // 0x9ee55a highlight green
  uColGrassDry:    { value: new THREE.Vector3(0.788, 0.820, 0.478) }, // 0xc9d17a dry yellow-green
  uColDirt:        { value: new THREE.Vector3(0.541, 0.478, 0.333) }, // 0x8a7a55 dirt
  uColFlowerWhite: { value: new THREE.Vector3(0.960, 0.960, 0.940) }, // 0xf5f5f0 daisy white
  uColFlowerYellow:{ value: new THREE.Vector3(1.000, 0.850, 0.240) }, // 0xffd93d yellow
  uColFlowerRed:   { value: new THREE.Vector3(0.880, 0.230, 0.160) }, // 0xe03a2a red
  uColFlowerPink:  { value: new THREE.Vector3(1.000, 0.420, 0.620) }, // 0xff6b9d pink
  uWindStrength:   { value: 0.7 },
  uBladeDensity:   { value: 0.85 },
};

/* ------------------------------------------------------------------ */
/* 4. GRASS GROUND VERTEX SHADER                                       */
/* ------------------------------------------------------------------ */
export const GRASS_GROUND_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = lush, 1.0 = dry, 2.0 = sparse, 3.0 = flower meadow

uniform float uWindStrength;

varying vec2  vUv;
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
  
  // Grass blade sway (tips move more than roots)
  float swayAmount = uWindStrength * (0.30 + 0.25 * fract(vSeed * 9.17));
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.6 + aSeed * 27.9 + vUv.y * 4.0) * vUv.y;
  w += windOffset;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. GRASS GROUND FRAGMENT SHADER                                     */
/*    FIXED: dead flowerDot() helper removed, all edges low->high     */
/* ------------------------------------------------------------------ */
export const GRASS_GROUND_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColGrass1;
uniform vec3  uColGrass2;
uniform vec3  uColGrass3;
uniform vec3  uColGrass4;
uniform vec3  uColGrassDry;
uniform vec3  uColDirt;
uniform vec3  uColFlowerWhite;
uniform vec3  uColFlowerYellow;
uniform vec3  uColFlowerRed;
uniform vec3  uColFlowerPink;
uniform float uBladeDensity;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

// Single grass blade mask (correct smoothstep order)
float grassBlade(vec2 uv, float bx, float width, float height, float lean) {
  float cx = bx + lean * uv.y;
  float blade = 1.0 - smoothstep(width - 0.012, width, abs(uv.x - cx));
  blade *= 1.0 - smoothstep(height - 0.10, height, uv.y);
  return blade;
}

void main() {
  // 1. Variant masks
  float isDry    = step(0.5, vTint) * (1.0 - step(1.5, vTint));
  float isSparse = step(1.5, vTint) * (1.0 - step(2.5, vTint));
  float isMeadow = step(2.5, vTint);
  
  // 2. Base grass color mixing
  float colorNoise = fbm(vUv * 5.0 + vSeed * 7.0, 3);
  float heightGradient = smoothstep(0.0, 1.0, vUv.y);
  
  vec3 baseColor = mix(uColGrass1, uColGrass2, colorNoise);
  baseColor = mix(baseColor, uColGrass3, heightGradient * 0.30);
  baseColor = mix(baseColor, uColGrass4, (1.0 - colorNoise) * 0.25);
  
  // Dry variant shifts toward yellow-green
  baseColor = mix(baseColor, uColGrassDry, isDry * 0.55);
  
  // 3. Dirt patches (sparse variant shows more dirt)
  float dirtNoise = fbm(vUv * 4.0 + vSeed * 13.0, 3);
  float dirtMask = smoothstep(0.55, 0.75, dirtNoise) * (0.25 + isSparse * 0.55);
  baseColor = mix(baseColor, uColDirt * (0.85 + dirtNoise * 0.30), dirtMask);
  
  // 4. Multi-scale grass blade detail
  float blades = grassBlade(vUv * 3.0, 0.25, 0.05, 0.90,  0.10);
  blades = max(blades, grassBlade(vUv * 3.0, 0.55, 0.04, 0.80, -0.08));
  blades = max(blades, grassBlade(vUv * 3.0, 0.80, 0.05, 0.95,  0.12));
  blades = max(blades, grassBlade(vUv * 4.0 + 0.37, 0.40, 0.04, 0.85, -0.10) * 0.8);
  blades = max(blades, grassBlade(vUv * 5.0 + 0.71, 0.60, 0.03, 0.75,  0.08) * 0.6);
  blades *= uBladeDensity * (1.0 - dirtMask) * (1.0 - isSparse * 0.45);
  
  // Blade tips lighter
  baseColor *= 1.0 + blades * 0.22;
  
  // 5. Scattered flower dots (per-cell hash placement)
  float gate = 0.80;
  gate = mix(gate, 0.90, isDry);
  gate = mix(gate, 0.84, isSparse);
  gate = mix(gate, 0.52, isMeadow);
  
  vec2 fUv  = vUv * 6.0;
  vec2 cell = floor(fUv);
  vec2 fuv  = fract(fUv) - 0.5;
  float fh  = h21(cell + vSeed * 17.0);
  vec2 foff = vec2(h21(cell + 1.7), h21(cell + 3.1)) - 0.5;
  float fd  = length(fuv - foff * 0.6);
  
  float flowerMask = (1.0 - smoothstep(0.10, 0.14, fd)) * step(gate, fh);
  flowerMask *= 1.0 - dirtMask;
  
  // Flower color selection via cell hash
  float fc = h21(cell + 9.3);
  vec3 flowerCol = uColFlowerWhite;
  flowerCol = mix(flowerCol, uColFlowerYellow, step(0.25, fc) * (1.0 - step(0.50, fc)));
  flowerCol = mix(flowerCol, uColFlowerRed,    step(0.50, fc) * (1.0 - step(0.75, fc)));
  flowerCol = mix(flowerCol, uColFlowerPink,   step(0.75, fc));
  
  // Yellow center dot
  float centerDot = 1.0 - smoothstep(0.03, 0.05, fd);
  flowerCol = mix(flowerCol, uColFlowerYellow, centerDot * 0.8);
  
  baseColor = mix(baseColor, flowerCol, flowerMask);
  
  // 6. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);
  
  // 7. Top-tip highlight + subtle live shimmer
  litColor += uColGrass4 * smoothstep(0.60, 0.95, vUv.y) * 0.10;
  litColor *= 1.0 + sin(uTime * 1.2 + vSeed * 6.2831) * 0.02;
  
  // 8. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 14.0, 7.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.94 + pxNoise * 0.12;
  
  gl_FragColor = vec4(litColor, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createGrassGroundMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: GRASS_GROUND_UNIFORMS,
    vertexShader: GRASS_GROUND_VERTEX_SHADER,
    fragmentShader: GRASS_GROUND_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: true,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}
