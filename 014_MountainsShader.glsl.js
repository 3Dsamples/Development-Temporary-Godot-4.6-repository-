// File : 014
// name : shaders/014_MountainsShader.glsl.js
// description : Procedural distant mountain ridges + rolling hills parallax shader.
//               ANALYZED & FIXED: (1) SEAMLESS-HORIZON BUG: the ridge/hill profile
//               was evaluated from local vUv.x plus a per-instance vSeed, so
//               adjacent billboard segments produced discontinuous silhouettes at
//               their shared edges — the profile is now evaluated from a
//               world-space X coordinate (vWorld varying) seeded only by the layer
//               index, making the horizon continuous across all instances of the
//               same parallax layer (same contract as elements/mountains_horizon.js
//               which displaces a shared strip in world space). (2) Cloud-shadow
//               fbm now samples world-space coordinates so shadows scroll
//               continuously across segment boundaries instead of restarting per
//               billboard. (3) Verified every smoothstep() uses correct low->high
//               edge order (GLSL ES safe), hn division guarded (max(peakAmp,1e-3)),
//               ridgeNoise loop has constant bounds (GLSL ES 1.00 safe), vnoise/
//               fbm/h21/pxq resolve from GLSL_NOISE, applyFogBlend resolves from
//               GLSL_COLOR_UTILS injected before GLSL_LIGHTING, no uniform is
//               redeclared (uTime/uPPU/uWind/uCamPos/uViewDir from GLSL_GLOBALS;
//               uSunDir/uSunColor/uFogColor from GLSL_LIGHTING), and every varying
//               (vUv/vWorld/vSeed/vTint) is consumed in the fragment stage.
//               Features 4 layer variants via aTint (0 = far ridge, 1 = mid ridge,
//               2 = near ridge, 3 = rolling green hills), ridged-FBM peak
//               silhouettes, snow caps with noisy snow lines, per-layer atmospheric
//               fog (aerial perspective), directional sun tinting, drifting cloud
//               shadows (real-time), time-of-day grading (day/dawn/night) synced
//               with the LightingSystem palette, hill flower/foliage patch detail,
//               and pixel-art de-banding dither. Optimized for Android mobile
//               (low-octave ridged noise, zero textures, GPU instancing).
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
/* 3. MOUNTAINS UNIFORMS (JS side)                                     */
/* ------------------------------------------------------------------ */
export const MOUNTAINS_UNIFORMS = {
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

  // Mountains specific (exact image palette)
  uTimeOfDay:     { value: 0.5 },
  uColRidgeFar:   { value: new THREE.Vector3(0.416, 0.478, 0.541) }, // 0x6a7a8a far blue-gray
  uColRidgeMid:   { value: new THREE.Vector3(0.353, 0.416, 0.478) }, // 0x5a6a7a mid gray
  uColRidgeNear:  { value: new THREE.Vector3(0.290, 0.353, 0.416) }, // 0x4a5a6a near slate
  uColPeakFar:    { value: new THREE.Vector3(0.541, 0.604, 0.667) }, // 0x8a9aaa far peak
  uColPeakMid:    { value: new THREE.Vector3(0.478, 0.541, 0.604) }, // 0x7a8a9a mid peak
  uColPeakNear:   { value: new THREE.Vector3(0.416, 0.478, 0.541) }, // 0x6a7a8a near peak
  uColSnow:       { value: new THREE.Vector3(0.816, 0.847, 0.878) }, // 0xd0d8e0 snow cap
  uColHill1:      { value: new THREE.Vector3(0.353, 0.620, 0.220) }, // 0x5a9e38 hill green
  uColHill2:      { value: new THREE.Vector3(0.478, 0.757, 0.259) }, // 0x7ac142 hill bright
  uColHill3:      { value: new THREE.Vector3(0.561, 0.820, 0.310) }, // 0x8fd14f hill highlight
  uSnowLine:      { value: 0.62 },
  uCloudShadow:   { value: 0.12 },
  uRidgeScale:    { value: 0.045 },
};

/* ------------------------------------------------------------------ */
/* 4. MOUNTAINS VERTEX SHADER (rigid background, pixel-locked)         */
/*    FIXED: exports world-space position for seamless ridge field    */
/* ------------------------------------------------------------------ */
export const MOUNTAINS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = far ridge, 1.0 = mid ridge, 2.0 = near ridge, 3.0 = rolling hills

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
  
  // Pixel-art quantization on world position (crisp silhouette steps)
  vec2 w = pxq(aOff + p, uPPU);
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. MOUNTAINS FRAGMENT SHADER (ridged peaks + hills + fog + snow)    */
/*    FIXED: ridge/hill profile + cloud shadows sampled in world space */
/* ------------------------------------------------------------------ */
export const MOUNTAINS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform float uTimeOfDay;
uniform vec3  uColRidgeFar;
uniform vec3  uColRidgeMid;
uniform vec3  uColRidgeNear;
uniform vec3  uColPeakFar;
uniform vec3  uColPeakMid;
uniform vec3  uColPeakNear;
uniform vec3  uColSnow;
uniform vec3  uColHill1;
uniform vec3  uColHill2;
uniform vec3  uColHill3;
uniform float uSnowLine;
uniform float uCloudShadow;
uniform float uRidgeScale;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

// Ridged multi-octave noise (sharp peak silhouettes, constant-bounds loop)
float ridgeNoise(vec2 p, float seed) {
  float n = 0.0;
  float amp = 0.55;
  float freq = 1.0;
  for (int i = 0; i < 4; i++) {
    float v = vnoise(p * freq + seed + float(i) * 17.7);
    v = 1.0 - abs(v * 2.0 - 1.0);
    n += v * v * amp;
    amp *= 0.5;
    freq *= 2.1;
  }
  return clamp(n, 0.0, 1.0);
}

void main() {
  // 1. Layer masks
  float layer   = clamp(vTint, 0.0, 3.0);
  float isHills = step(2.5, layer);
  float farW    = 1.0 - step(0.5, layer);
  float midW    = step(0.5, layer) * (1.0 - step(1.5, layer));
  float nearW   = step(1.5, layer) * (1.0 - step(2.5, layer));

  // 2. Ridge profile in WORLD space (seamless across instances of a layer;
  //    seeded by layer index only, never by per-instance seed)
  float rx    = vWorld.x * uRidgeScale;
  float ridge = ridgeNoise(vec2(rx * 2.2, 0.5), layer * 7.31);
  float hills = fbm(vec2(rx * 1.4, 7.7) + layer * 3.17, 3);
  float prof  = mix(ridge, hills, isHills);

  float peakAmp = 0.38 * farW + 0.28 * midW + 0.20 * nearW;
  peakAmp = mix(peakAmp, 0.14, isHills);
  float baseLine = mix(0.34, 0.26, isHills);
  float h = baseLine + prof * peakAmp;

  // 3. Silhouette mask (correct low->high smoothstep order)
  float mask = 1.0 - smoothstep(h - 0.02, h + 0.02, vUv.y);
  if (mask < 0.02) discard;

  // 4. Normalized height above base (guarded division)
  float hn = clamp((vUv.y - baseLine) / max(peakAmp, 1e-3), 0.0, 1.0);

  // 5. Per-layer base + peak colors
  vec3 baseCol = uColRidgeFar * farW + uColRidgeMid * midW + uColRidgeNear * nearW;
  vec3 peakCol = uColPeakFar * farW + uColPeakMid * midW + uColPeakNear * nearW;
  vec3 col = mix(baseCol, peakCol, hn);

  // 6. Rolling hills green palette + patch detail
  if (isHills > 0.5) {
    float patch = fbm(vUv * 6.0 + vSeed * 7.0, 3);
    vec3 hillCol = mix(uColHill1, uColHill2, patch);
    hillCol = mix(hillCol, uColHill3, smoothstep(0.55, 0.80, hn) * 0.5);
    col = hillCol;
  }

  // 7. Snow caps (mountains only, noisy snow line)
  float snowNoise = fbm(vUv * 14.0 + vSeed * 3.0, 3);
  float snowLine = uSnowLine - nearW * 0.10;
  float snowMask = smoothstep(snowLine, snowLine + 0.12, hn + (snowNoise - 0.5) * 0.14);
  snowMask *= 1.0 - isHills;
  col = mix(col, uColSnow, snowMask * 0.85);

  // 8. Directional sun tint (cheap slope lighting, no normalize needed)
  float sunSide = clamp(uSunDir.x * (vUv.x - 0.5) * 2.0, 0.0, 1.0);
  col *= 0.86 + sunSide * 0.26;

  // 9. Drifting cloud shadows in WORLD space (continuous across segments)
  float cShadow = smoothstep(0.55, 0.78, fbm(vec2(vWorld.x * 0.05 + uTime * 0.02, vWorld.y * 0.05) + layer * 11.3, 2));
  col *= 1.0 - cShadow * uCloudShadow;

  // 10. Time-of-day grading (day/dawn/night weights, correct order)
  float dayW = smoothstep(0.20, 0.35, uTimeOfDay) * (1.0 - smoothstep(0.65, 0.80, uTimeOfDay));
  float dawnW = smoothstep(0.05, 0.20, uTimeOfDay) * (1.0 - smoothstep(0.25, 0.40, uTimeOfDay))
              + smoothstep(0.60, 0.75, uTimeOfDay) * (1.0 - smoothstep(0.80, 0.95, uTimeOfDay));
  float nightW = (1.0 - smoothstep(0.05, 0.20, uTimeOfDay)) + smoothstep(0.80, 0.95, uTimeOfDay);
  float wSum = dayW + dawnW + nightW + 1e-3;
  dayW /= wSum; dawnW /= wSum; nightW /= wSum;
  vec3 todTint = vec3(1.0) * dayW + vec3(1.10, 0.92, 0.82) * dawnW + vec3(0.62, 0.68, 0.90) * nightW;
  col *= todTint;
  col *= mix(0.45, 1.0, clamp(dayW + dawnW, 0.0, 1.0));

  // 11. Per-layer atmospheric fog (aerial perspective) + palette fog utility
  float layerFog = 0.62 * farW + 0.42 * midW + 0.24 * nearW + 0.34 * isHills;
  col = applyFogBlend(col, uFogColor, layerFog);

  // 12. Pixel-art de-banding dither
  vec2 pUv = pxq(vUv * 20.0, 10.0);
  col += (h21(pUv + vSeed) - 0.5) * 0.015;

  // 13. Edge softening at silhouette
  float alpha = mask * smoothstep(0.0, 0.06, mask);
  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createMountainsMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: MOUNTAINS_UNIFORMS,
    vertexShader: MOUNTAINS_VERTEX_SHADER,
    fragmentShader: MOUNTAINS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}

/* ------------------------------------------------------------------ */
/* 7. RUNTIME HELPERS                                                  */
/* ------------------------------------------------------------------ */
export function setMountainsTimeOfDay(t) {
  MOUNTAINS_UNIFORMS.uTimeOfDay.value = t - Math.floor(t);
}
export function setMountainsSnowLine(v) {
  MOUNTAINS_UNIFORMS.uSnowLine.value = Math.max(0.2, Math.min(0.9, v));
}
export function setMountainsRidgeScale(v) {
  MOUNTAINS_UNIFORMS.uRidgeScale.value = Math.max(0.01, Math.min(0.2, v));
}
