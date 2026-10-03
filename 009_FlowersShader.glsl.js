// File : 009
// name : shaders/009_FlowersShader.glsl.js
// description : Procedural meadow flowers shader (white daisies, yellow, red and pink
//               blossoms). ANALYZED & FIXED: removed the duplicate `uniform vec2 uWind`
//               redeclaration in the vertex stage (already declared by GLSL_GLOBALS,
//               which can fault strict Mali/Adreno compilers), guarded the atan()
//               zero-length singularity at the petal centroid (atan(0,0) is undefined
//               per GLSL ES spec and can NaN on some mobile GPUs), removed unused
//               fragment uniform declarations, and verified every smoothstep() uses
//               correct low->high edge order (GLSL ES safe). Features radial
//               cosine-lobe petal shapes with per-type petal counts, distinct center
//               discs, subtle bloom pulsing, wind-reactive stem sway in the vertex
//               shader, cel-shaded lighting and pixel-art quantization. Fully composed
//               using 000_BaseShader.glsl.js and GLSL_COLOR_UTILS from
//               004_ColorPalette.js with correct injection order (COLOR_UTILS before
//               LIGHTING). All varyings are consumed in the fragment stage and no
//               uniform is redeclared (uTime/uPPU/uWind come from GLSL_GLOBALS).
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
/* 3. FLOWERS UNIFORMS (JS side)                                       */
/* ------------------------------------------------------------------ */
export const FLOWERS_UNIFORMS = {
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
  uRimStrength:      { value: 0.45 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Flowers specific colors (exact image palette)
  uColPetalWhite:  { value: new THREE.Vector3(0.960, 0.960, 0.940) }, // 0xf5f5f0 daisy petals
  uColPetalYellow: { value: new THREE.Vector3(1.000, 0.850, 0.240) }, // 0xffd93d yellow blossom
  uColPetalRed:    { value: new THREE.Vector3(0.880, 0.230, 0.160) }, // 0xe03a2a red blossom
  uColPetalPink:   { value: new THREE.Vector3(1.000, 0.420, 0.620) }, // 0xff6b9d pink blossom
  uColCenterYellow:{ value: new THREE.Vector3(0.900, 0.660, 0.090) }, // 0xe6a817 daisy center
  uColCenterOrange:{ value: new THREE.Vector3(0.850, 0.470, 0.080) }, // 0xd97814 yellow-flower center
  uColCenterDark:  { value: new THREE.Vector3(0.420, 0.100, 0.080) }, // 0x6b1a14 red-flower center
  uColLeaf:        { value: new THREE.Vector3(0.280, 0.550, 0.200) }, // 0x478c33 leaf hint
  uWindStrength:   { value: 0.8 },
  uBloomPulse:     { value: 0.08 },
};

/* ------------------------------------------------------------------ */
/* 4. FLOWERS VERTEX SHADER                                            */
/*    FIXED: no duplicate uWind declaration (comes from GLSL_GLOBALS)  */
/* ------------------------------------------------------------------ */
export const FLOWERS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = daisy, 1.0 = yellow, 2.0 = red, 3.0 = pink

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
  
  // Stem-anchored wind sway (base fixed, head sways with height)
  float swayAmount = uWindStrength * (0.35 + 0.30 * fract(vSeed * 7.31));
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.4 + aSeed * 33.7 + vUv.y * 3.0) * vUv.y;
  w += windOffset;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. FLOWERS FRAGMENT SHADER                                          */
/*    FIXED: atan() zero-length singularity guarded at petal centroid  */
/* ------------------------------------------------------------------ */
export const FLOWERS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColPetalWhite;
uniform vec3  uColPetalYellow;
uniform vec3  uColPetalRed;
uniform vec3  uColPetalPink;
uniform vec3  uColCenterYellow;
uniform vec3  uColCenterOrange;
uniform vec3  uColCenterDark;
uniform vec3  uColLeaf;
uniform float uBloomPulse;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

// Radial cosine-lobe petal mask (correct smoothstep order, guarded atan)
float petalShape(vec2 uv, float petals, float radius, float seed) {
  vec2 c = uv - 0.5;
  float d = length(c) * 2.0;
  float angle = atan(c.y, c.x + 1e-6); // guard atan(0,0) undefined behavior
  float lobes = 0.78 + 0.22 * cos(angle * petals + seed * 2.0);
  float shape = 1.0 - smoothstep(radius * lobes - 0.14, radius * lobes, d);
  return shape;
}

// Center disc mask (correct smoothstep order)
float centerDisc(vec2 uv, float radius) {
  float d = length(uv - 0.5) * 2.0;
  return 1.0 - smoothstep(radius - 0.08, radius, d);
}

// Small leaf pair at the base (correct smoothstep order)
float leafPair(vec2 uv) {
  vec2 c = uv - vec2(0.5, 0.10);
  float leafL = 1.0 - smoothstep(0.10, 0.16, length(vec2(c.x + 0.16, c.y * 1.8)));
  float leafR = 1.0 - smoothstep(0.10, 0.16, length(vec2(c.x - 0.16, c.y * 1.8)));
  return max(leafL, leafR);
}

void main() {
  // 1. Per-type petal counts and colors
  float petalCount = 12.0;
  vec3 petalCol = uColPetalWhite;
  vec3 centerCol = uColCenterYellow;
  
  if (vTint > 0.5 && vTint < 1.5) {
    petalCount = 8.0;
    petalCol = uColPetalYellow;
    centerCol = uColCenterOrange;
  } else if (vTint > 1.5 && vTint < 2.5) {
    petalCount = 10.0;
    petalCol = uColPetalRed;
    centerCol = uColCenterDark;
  } else if (vTint > 2.5) {
    petalCount = 9.0;
    petalCol = uColPetalPink;
    centerCol = uColCenterYellow;
  }
  
  // 2. Bloom pulsing (live breathing of the blossom)
  float bloom = 1.0 - uBloomPulse + uBloomPulse * sin(uTime * 1.4 + vSeed * 6.2831);
  
  // 3. Shapes
  float petals = petalShape(vUv, petalCount, 0.88 * bloom, vSeed);
  float center = centerDisc(vUv, 0.34);
  float leaves = leafPair(vUv) * (1.0 - step(0.5, vTint)); // leaves mostly on daisies
  
  float shape = max(petals, max(center, leaves));
  if (shape < 0.05) discard;
  
  // 4. Base color assembly
  vec3 baseColor = mix(petalCol * 0.82, petalCol, smoothstep(0.0, 0.55, length(vUv - 0.5) * 2.0));
  baseColor = mix(baseColor, centerCol, center);
  baseColor = mix(baseColor, uColLeaf, leaves * (1.0 - center));
  
  // Per-petal noise variation
  float petalNoise = fbm(vUv * 10.0 + vSeed * 17.0, 2);
  baseColor *= 0.92 + petalNoise * 0.16;
  
  // 5. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);
  
  // 6. Center disc highlight (pollen glint)
  float glint = center * smoothstep(0.55, 0.85, fract(vSeed * 13.7 + uTime * 0.15));
  litColor += centerCol * glint * 0.25;
  
  // 7. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.94 + pxNoise * 0.12;
  
  // 8. Edge softening
  float alpha = shape * smoothstep(0.0, 0.12, shape);
  
  gl_FragColor = vec4(litColor, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createFlowersMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: FLOWERS_UNIFORMS,
    vertexShader: FLOWERS_VERTEX_SHADER,
    fragmentShader: FLOWERS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}
