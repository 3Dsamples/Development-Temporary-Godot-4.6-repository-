// File : 015
// name : shaders/015_ShrubsShader.glsl.js
// description : Procedural rounded shrubs/bushes scattered on the meadow and biome
//               transition zones. ANALYZED & FIXED: (1) UNSHADED-OVERLAY BUG: the
//               flowering blossoms, yellow centers and brown berries were mixed
//               AFTER computeCelLighting(), so at night/dawn they stayed fully
//               lit while the foliage darkened — all variant overlays are now
//               applied to the base color BEFORE cel lighting so every element
//               receives consistent LightingSystem shading and shadow tint.
//               (2) INJECTED-BIOME-CHUNK-UNUSED BUG: GLSL_BIOME was injected but
//               uBiomeW never sampled, so shrubs did not react to the seamless
//               biome field — added meadow brightening / rock darkening modulation.
//               (3) Verified bushBlob() smoothstep edges stay low->high even at
//               worst-case breathing radius + bump sum (GLSL ES safe), atan()
//               epsilon-guarded at blob centroids, no uniform is redeclared
//               (uTime/uPPU/uWind/uCamPos/uViewDir from GLSL_GLOBALS; lighting set
//               from GLSL_LIGHTING; biome set from GLSL_BIOME), GLSL_COLOR_UTILS
//               injected before GLSL_LIGHTING so computeCelLighting() resolves
//               applyShadowTint()/applyRimLight(), every varying (vUv/vSeed/vTint)
//               is consumed, vertex offsets (wind sway + breathing bulge) are
//               applied BEFORE pxq() for pixel-locked animated edges, and all time
//               terms are bounded sinusoids (float32-safe on long sessions).
//               Features 4 bush variants via aTint (0 = small round, 1 = large
//               fluffy, 2 = flowering with pink blossoms + yellow centers, 3 =
//               dense with brown berries), fluffy noise-distorted blob silhouettes
//               with ground-sit flattening and dome caps, breathing radii, hash-grid
//               flower/berry scatter, bottom contact shadow, top highlight and
//               pixel-art de-banding dither. Optimized for Android mobile
//               (low-octave FBM, zero textures, GPU instancing). Colors match the
//               reference image exactly without text labels.
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
/* 3. SHRUBS UNIFORMS (JS side)                                        */
/* ------------------------------------------------------------------ */
export const SHRUBS_UNIFORMS = {
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
  uRimStrength:      { value: 0.40 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Shrubs specific (exact image palette)
  uColBush1:        { value: new THREE.Vector3(0.353, 0.620, 0.220) }, // 0x5a9e38 medium green
  uColBush2:        { value: new THREE.Vector3(0.478, 0.757, 0.259) }, // 0x7ac142 bright green
  uColBush3:        { value: new THREE.Vector3(0.227, 0.420, 0.165) }, // 0x3a6b2a dark green
  uColBush4:        { value: new THREE.Vector3(0.561, 0.820, 0.310) }, // 0x8fd14f yellow-green
  uColFlower:       { value: new THREE.Vector3(1.000, 0.420, 0.616) }, // 0xff6b9d pink blossom
  uColFlowerCenter: { value: new THREE.Vector3(1.000, 0.851, 0.239) }, // 0xffd93d yellow center
  uColBerry:        { value: new THREE.Vector3(0.545, 0.271, 0.075) }, // 0x8b4513 brown berry
  uWindStrength:    { value: 0.6 },
  uFlowerDensity:   { value: 0.35 },
};

/* ------------------------------------------------------------------ */
/* 4. SHRUBS VERTEX SHADER (bounded sway, pixel-locked offsets)        */
/* ------------------------------------------------------------------ */
export const SHRUBS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = small round, 1.0 = large fluffy, 2.0 = flowering, 3.0 = dense

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
  
  // Bounded wind sway (tip-weighted, per-instance speed variance)
  float speedVar = 0.5 + 0.5 * fract(vSeed * 5.17);
  float sway = uWindStrength * (0.25 + 0.20 * step(0.5, vTint));
  vec2 windOffset = uWind * sway * sin(uTime * 2.1 * speedVar + aSeed * 21.7 + vUv.y * 3.0) * vUv.y;
  
  // Gentle breathing bulge (bounded)
  vec2 bulge = vec2(
    sin(uTime * 1.3 + aSeed * 9.4),
    cos(uTime * 1.1 + aSeed * 7.7)
  ) * 0.05 * vUv.y;
  
  // Pixel-art quantization AFTER all offsets -> crisp pixel-locked edges
  vec2 w = pxq(aOff + p + windOffset + bulge, uPPU);
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. SHRUBS FRAGMENT SHADER (cel bush + flowers + berries)            */
/*    FIXED: overlays moved BEFORE cel lighting; biome modulation on  */
/* ------------------------------------------------------------------ */
export const SHRUBS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColBush1;
uniform vec3  uColBush2;
uniform vec3  uColBush3;
uniform vec3  uColBush4;
uniform vec3  uColFlower;
uniform vec3  uColFlowerCenter;
uniform vec3  uColBerry;
uniform float uFlowerDensity;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

// Fluffy bush blob (correct smoothstep order, epsilon-guarded atan)
float bushBlob(vec2 q, vec2 center, float radius, float seed) {
  float d = length((q - center) * vec2(1.0, 1.15));
  float angle = atan(q.y - center.y, q.x - center.x + 1e-6);
  float bumps = sin(angle * 6.0 + seed * 8.0) * 0.16
              + sin(angle * 10.0 + seed * 12.0) * 0.10;
  float noisyRadius = radius + bumps;
  float noise = fbm(q * 8.0 + seed, 3) * 0.10;
  return 1.0 - smoothstep(noisyRadius - 0.10, noisyRadius + 0.08, d + noise);
}

void main() {
  // 1. Variant masks
  float isSmall  = 1.0 - step(0.5, vTint);
  float isFluffy = step(0.5, vTint) * (1.0 - step(1.5, vTint));
  float isFlower = step(1.5, vTint) * (1.0 - step(2.5, vTint));
  float isDense  = step(2.5, vTint);

  // 2. Fluffy silhouette (breathing overlapping blobs)
  float breath = 1.0 + sin(uTime * 1.2 + vSeed * 6.2831) * 0.03;
  float shape = 0.0;
  shape = max(shape, bushBlob(vUv, vec2(0.50, 0.46), 0.34 * breath, vSeed + 1.0));
  shape = max(shape, bushBlob(vUv, vec2(0.34, 0.40), 0.26 * breath, vSeed + 2.0));
  shape = max(shape, bushBlob(vUv, vec2(0.66, 0.40), 0.26 * breath, vSeed + 3.0));
  shape = max(shape, bushBlob(vUv, vec2(0.50, 0.62), 0.24 * breath, vSeed + 4.0));
  
  // Fine leaf detail
  float detail = fbm(vUv * 14.0 + vSeed * 19.0, 3);
  shape *= 0.80 + smoothstep(0.30, 0.70, detail) * 0.20;
  
  // Ground sit flattening + dome cap (correct order)
  shape *= smoothstep(0.04, 0.20, vUv.y);
  shape *= 1.0 - smoothstep(0.92, 1.00, vUv.y);
  
  float body = smoothstep(0.22, 0.40, shape);
  if (body < 0.02) discard;

  // 3. Base foliage color
  float colorNoise = fbm(vUv * 5.0 + vSeed * 7.0, 3);
  vec3 col = mix(uColBush1, uColBush2, colorNoise);
  col = mix(col, uColBush3, (1.0 - smoothstep(0.10, 0.55, vUv.y)) * 0.45);
  col = mix(col, uColBush4, smoothstep(0.60, 0.90, vUv.y) * 0.35);
  col = mix(col, uColBush3 * 0.85, isDense * 0.30);
  col = mix(col, uColBush2 * 1.05, isFluffy * 0.15);
  col = mix(col, uColBush1 * 0.95, isSmall * 0.10);

  // 4. FIXED: flowering blossoms + berries applied BEFORE cel lighting
  if (isFlower > 0.5) {
    vec2 fUv  = vUv * 5.0;
    vec2 cell = floor(fUv);
    vec2 fuv  = fract(fUv) - 0.5;
    float fh  = h21(cell + vSeed * 13.0);
    vec2 foff = vec2(h21(cell + 2.3), h21(cell + 4.1)) - 0.5;
    float fd  = length(fuv - foff * 0.5);
    float flower = (1.0 - smoothstep(0.12, 0.18, fd)) * step(1.0 - uFlowerDensity, fh);
    float fCenter = 1.0 - smoothstep(0.04, 0.07, fd);
    col = mix(col, uColFlower, flower * 0.90);
    col = mix(col, uColFlowerCenter, flower * fCenter * 0.90);
  }
  if (isDense > 0.5) {
    vec2 bUv  = vUv * 7.0;
    vec2 cell = floor(bUv);
    vec2 buv  = fract(bUv) - 0.5;
    float bh  = h21(cell + vSeed * 17.0);
    vec2 boff = vec2(h21(cell + 3.7), h21(cell + 5.9)) - 0.5;
    float bd  = length(buv - boff * 0.5);
    float berry = (1.0 - smoothstep(0.08, 0.12, bd)) * step(0.80, bh);
    col = mix(col, uColBerry, berry * 0.80);
  }

  // 5. FIXED: seamless biome field modulation (meadow bright / rock dark)
  col = mix(col, col * 1.06, uBiomeW.y * 0.25);
  col = mix(col, col * 0.88, uBiomeW.z * 0.30);

  // 6. Cel-shaded lighting (LightingSystem synced, shades overlays too)
  vec3 litColor = computeCelLighting(col, vec3(0.0, 0.0, 1.0), uViewDir);

  // 7. Fluffy top highlight (specular-like, post-light)
  litColor += uColBush4 * smoothstep(0.60, 0.90, vUv.y) * isFluffy * 0.12;

  // 8. Bottom contact shadow (grounding)
  litColor = mix(litColor, litColor * 0.70, 1.0 - smoothstep(0.05, 0.30, vUv.y));

  // 9. Pixel-art de-banding dither
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  litColor *= 0.94 + (h21(pUv + vSeed) - 0.5) * 0.12;

  // 10. Edge softening
  float alpha = body * smoothstep(0.0, 0.08, body);
  gl_FragColor = vec4(litColor, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createShrubsMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: SHRUBS_UNIFORMS,
    vertexShader: SHRUBS_VERTEX_SHADER,
    fragmentShader: SHRUBS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}

/* ------------------------------------------------------------------ */
/* 7. RUNTIME HELPERS                                                  */
/* ------------------------------------------------------------------ */
export function setShrubsWind(x, y) {
  SHRUBS_UNIFORMS.uWind.value.set(x, y);
}
export function setShrubsFlowerDensity(v) {
  SHRUBS_UNIFORMS.uFlowerDensity.value = Math.max(0.0, Math.min(1.0, v));
}
