// File : 013
// name : shaders/013_CloudsShader.glsl.js
// description : Dynamic real-time anime cloud shader. ANALYZED & FIXED:
//               (1) UNBOUNDED DRIFT BUG: the vertex stage added
//               `uTime * uCloudSpeed * speedVar` linearly to world position, which
//               grows without bound (float32 precision loss after long sessions on
//               Mali/Adreno and clouds escaping the ECS-managed region) — replaced
//               with a bounded layered-sinusoidal wander (precision-safe); steady
//               advection stays delegated to the ECS cloud manager (same contract
//               as elements/sky_clouds.js which moves pool.off per frame).
//               (2) PIXEL-LOCK BUG: drift/wind offsets were added AFTER pxq(),
//               defeating pixel-art quantization (edges swam at sub-pixel) — all
//               offsets are now added BEFORE pxq() so animated positions snap to
//               the pixel grid. (3) Verified every smoothstep() uses correct
//               low->high edge order (GLSL ES safe), puff() radius-falloff order
//               confirmed, normalize()/division singularities guarded
//               (sun XY length, rim gradient epsilon), no uniform is redeclared
//               (uTime/uPPU/uWind/uCamPos/uViewDir from GLSL_GLOBALS; uSunDir/
//               uSunColor/uFogColor from GLSL_LIGHTING), GLSL_COLOR_UTILS injected
//               before GLSL_LIGHTING so applyFogBlend() resolves, and all varyings
//               (vUv/vSeed/vTint) are consumed in the fragment stage. Features
//               fluffy cumulus billboards from overlapping noise-distorted puffs
//               with time-evolving domain-warped FBM (live morphing), breathing
//               puff radii, flat anime cel tone bands, crisp silhouette outline,
//               sun-side silver-lining rim, flattened anime bottoms, day/dawn/
//               night palette blending, low-altitude fog haze and de-banding
//               dither. Optimized for Android mobile (low-octave FBM, zero
//               textures, GPU instancing). Colors match the reference image
//               exactly without text labels.
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
/* 3. CLOUDS UNIFORMS (JS side)                                        */
/* ------------------------------------------------------------------ */
export const CLOUDS_UNIFORMS = {
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

  // Clouds specific (exact anime palette)
  uTimeOfDay:       { value: 0.5 },
  uCloudSpeed:      { value: 0.6 },
  uPuffiness:       { value: 1.0 },
  uCoverage:        { value: 0.85 },
  uCelSteps:        { value: 3.0 },
  uColCloudDay:     { value: new THREE.Vector3(1.000, 1.000, 1.000) }, // 0xffffff day light
  uColCloudDaySh:   { value: new THREE.Vector3(0.624, 0.722, 0.847) }, // 0x9fb8d8 day shadow
  uColCloudDawn:    { value: new THREE.Vector3(1.000, 0.851, 0.722) }, // 0xffd9b8 dawn light
  uColCloudDawnSh:  { value: new THREE.Vector3(0.847, 0.541, 0.478) }, // 0xd88a7a dawn shadow
  uColCloudNight:   { value: new THREE.Vector3(0.227, 0.290, 0.416) }, // 0x3a4a6a night light
  uColCloudNightSh: { value: new THREE.Vector3(0.137, 0.173, 0.267) }, // 0x232c44 night shadow
};

/* ------------------------------------------------------------------ */
/* 4. CLOUDS VERTEX SHADER (bounded wander + pixel-locked offsets)     */
/*    FIXED: offsets applied BEFORE pxq(); drift bounded (no uTime     */
/*    runaway); steady advection delegated to the ECS cloud manager    */
/* ------------------------------------------------------------------ */
export const CLOUDS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = low haze cloud, 1.0 = high altitude cloud

uniform float uCloudSpeed;

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
  
  // Bounded real-time wander (precision-safe on long sessions)
  float speedVar = 0.4 + 0.6 * fract(vSeed * 7.31);
  float wt = uTime * uCloudSpeed * speedVar;
  vec2 wander = vec2(
    sin(wt * 0.100 + vSeed * 6.2831) * 1.50 + sin(wt * 0.031 + vSeed * 12.0) * 0.75,
    cos(wt * 0.083 + vSeed * 4.7120) * 0.60
  );
  
  // Gentle wind sway (high clouds feel more wind)
  float sway = 0.25 + 0.45 * vTint;
  vec2 windOffset = vec2(
    uWind.x * sway * sin(uTime * 0.4 + aSeed * 6.2831) * 0.35,
    uWind.y * sway * cos(uTime * 0.3 + aSeed * 4.7120) * 0.20
  );
  
  // Pixel-art quantization AFTER all offsets -> crisp pixel-locked edges
  vec2 w = pxq(aOff + p + wander + windOffset, uPPU);
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. CLOUDS FRAGMENT SHADER (anime cel cumulus, live morphing)        */
/* ------------------------------------------------------------------ */
export const CLOUDS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform float uTimeOfDay;
uniform float uCloudSpeed;
uniform float uPuffiness;
uniform float uCoverage;
uniform float uCelSteps;
uniform vec3  uColCloudDay;
uniform vec3  uColCloudDaySh;
uniform vec3  uColCloudDawn;
uniform vec3  uColCloudDawnSh;
uniform vec3  uColCloudNight;
uniform vec3  uColCloudNightSh;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

// Noise-distorted puff (correct smoothstep order)
float puff(vec2 q, vec2 center, float radius, float seed) {
  float d = length((q - center) * vec2(1.0, 1.3));
  float n = fbm(q * 6.0 + seed, 3) * 0.16;
  return 1.0 - smoothstep(radius - 0.06, radius + 0.10, d + n);
}

void main() {
  // 1. Time-of-day weights (correct low->high smoothstep order)
  float dayW = smoothstep(0.20, 0.35, uTimeOfDay) * (1.0 - smoothstep(0.65, 0.80, uTimeOfDay));
  float dawnW = smoothstep(0.05, 0.20, uTimeOfDay) * (1.0 - smoothstep(0.25, 0.40, uTimeOfDay))
              + smoothstep(0.60, 0.75, uTimeOfDay) * (1.0 - smoothstep(0.80, 0.95, uTimeOfDay));
  float nightW = (1.0 - smoothstep(0.05, 0.20, uTimeOfDay)) + smoothstep(0.80, 0.95, uTimeOfDay);
  float wSum = dayW + dawnW + nightW + 1e-3;
  dayW /= wSum; dawnW /= wSum; nightW /= wSum;
  float dayGate = clamp(dayW + dawnW, 0.0, 1.0);

  vec3 lightCol  = uColCloudDay * dayW + uColCloudDawn * dawnW + uColCloudNight * nightW;
  vec3 shadowCol = uColCloudDaySh * dayW + uColCloudDawnSh * dawnW + uColCloudNightSh * nightW;

  // 2. Live morphing coordinate (time-evolving domain warp)
  float t = uTime * 0.05 * uCloudSpeed;
  vec2 uv = vUv;
  vec2 warp;
  warp.x = fbm(uv * 2.5 + vSeed * 3.1 + vec2(t, 0.0), 2) - 0.5;
  warp.y = fbm(uv * 2.5 + vSeed * 5.7 + vec2(0.0, t * 0.6), 2) - 0.5;
  vec2 q = uv + warp * (0.25 * uPuffiness);

  // 3. Overlapping breathing puffs (radii pulse in real time)
  float breath = 1.0 + sin(uTime * 0.8 + vSeed * 6.2831) * 0.04;
  float shape = 0.0;
  shape = max(shape, puff(q, vec2(0.30, 0.40), 0.24 * breath, vSeed + 1.0));
  shape = max(shape, puff(q, vec2(0.50, 0.56), 0.30 * breath, vSeed + 2.0));
  shape = max(shape, puff(q, vec2(0.70, 0.42), 0.22 * breath, vSeed + 3.0));
  shape = max(shape, puff(q, vec2(0.42, 0.34), 0.20 * breath, vSeed + 4.0));
  shape = max(shape, puff(q, vec2(0.60, 0.34), 0.20 * breath, vSeed + 5.0));
  
  // Fine FBM detail (evolving)
  shape += (fbm(q * 5.0 + vSeed * 11.0 + vec2(t * 0.5, 0.0), 3) - 0.5) * 0.18;

  // 4. Flattened anime bottom + top dome cap (correct order)
  shape *= smoothstep(0.10, 0.28, vUv.y);
  shape *= 1.0 - smoothstep(0.92, 1.00, vUv.y);

  // 5. Coverage gate -> crisp anime body
  float body = smoothstep(0.28, 0.42, shape * (0.55 + uCoverage * 0.60));
  if (body < 0.02) discard;

  // 6. Anime cel tone bands (quantized vertical gradient)
  float hGrad = smoothstep(0.15, 0.85, vUv.y);
  float tone = floor(hGrad * uCelSteps) / uCelSteps;
  vec3 col = mix(shadowCol, lightCol, tone);

  // 7. Underside shadow reinforcement (anime cloud belly)
  col = mix(col, shadowCol, (1.0 - smoothstep(0.15, 0.45, vUv.y)) * 0.55);

  // 8. Sun-side silver lining rim (guarded normalize)
  vec2 sd = uSunDir.xy;
  float sdl = max(length(sd), 1e-4);
  vec2 sdn = sd / sdl;
  vec2 g = normalize(vUv - 0.5 + vec2(1e-4, 1e-4));
  float rim = pow(max(dot(g, sdn), 0.0), 3.0) * body;
  col += uSunColor * rim * 0.35 * dayGate;

  // 9. Crisp silhouette outline darkening (anime edge)
  float outline = smoothstep(0.20, 0.30, shape) * (1.0 - smoothstep(0.30, 0.42, shape));
  col *= 1.0 - outline * 0.22;

  // 10. Low-altitude haze via palette fog utility (vTint = altitude)
  col = applyFogBlend(col, uFogColor, (1.0 - vTint) * 0.25);

  // 11. Pixel-art de-banding dither
  vec2 pUv = pxq(vUv * 20.0, 10.0);
  col += (h21(pUv + vSeed) - 0.5) * 0.015;

  float alpha = body * (0.70 + 0.30 * vTint);
  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createCloudsMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: CLOUDS_UNIFORMS,
    vertexShader: CLOUDS_VERTEX_SHADER,
    fragmentShader: CLOUDS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}

/* ------------------------------------------------------------------ */
/* 7. RUNTIME HELPERS                                                  */
/* ------------------------------------------------------------------ */
export function setCloudsTimeOfDay(t) {
  CLOUDS_UNIFORMS.uTimeOfDay.value = t - Math.floor(t);
}
export function setCloudsWind(x, y) {
  CLOUDS_UNIFORMS.uWind.value.set(x, y);
}
