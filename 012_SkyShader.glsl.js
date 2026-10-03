// File : 012
// name : shaders/012_SkyShader.glsl.js
// description : Procedural sky gradient shader with CHEAP atmospheric scattering
//               (Rayleigh-style zenith blue ramp + Mie-style forward sun halo +
//               guarded horizon atmospheric thickness + warm dusk scatter tint),
//               time-of-day blending (day/dawn/night), below-horizon ground haze,
//               palette fog blending and de-banding dither. SKY ONLY — no clouds.
//               ANALYZED & FIXED: (1) removed the unused GLSL_NOISE injection from
//               the vertex stage (shorter mobile driver compile time), (2) added the
//               missing night pipeline (guarded moon disc + moon glow via uMoonDir/
//               uMoonColor already declared by GLSL_LIGHTING, plus hash-grid star
//               twinkle gated by nightW and elevation) so night phase is never a
//               flat black gradient, (3) verified every smoothstep() uses correct
//               low->high edge order (GLSL ES safe), (4) verified normalize() calls
//               are zero-length guarded (sun/moon epsilon bias, rd.z == 1.0),
//               (5) verified no uniform is redeclared (uTime/uPPU/uWind/uCamPos/
//               uViewDir come from GLSL_GLOBALS; uSunDir/uSunColor/uMoonDir/
//               uMoonColor/uFogColor come from GLSL_LIGHTING), (6) verified
//               GLSL_COLOR_UTILS is injected before GLSL_LIGHTING so
//               applyFogBlend() resolves, (7) added setSkyAspect() runtime helper
//               so resize/orientation changes keep the horizon projection correct.
//               Fully composed using 000_BaseShader.glsl.js. Optimized for Android
//               mobile (single fullscreen quad, zero textures, minimal ALU).
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
/* 3. SKY UNIFORMS (JS side)                                           */
/* ------------------------------------------------------------------ */
export const SKY_UNIFORMS = {
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

  // Sky specific (exact image palette)
  uTimeOfDay:        { value: 0.5 },
  uAspect:           { value: 0.6 },
  uHorizonY:         { value: 0.30 },
  uColZenithDay:     { value: new THREE.Vector3(0.290, 0.639, 0.910) }, // 0x4aa3e8 bright blue
  uColHorizonDay:    { value: new THREE.Vector3(0.749, 0.878, 0.961) }, // 0xbfe0f5 pale horizon
  uColZenithDawn:    { value: new THREE.Vector3(0.169, 0.196, 0.435) }, // 0x2b326f dawn zenith
  uColHorizonDawn:   { value: new THREE.Vector3(0.878, 0.478, 0.373) }, // 0xe07a5f dawn haze
  uColZenithNight:   { value: new THREE.Vector3(0.043, 0.075, 0.169) }, // 0x0b132b night zenith
  uColHorizonNight:  { value: new THREE.Vector3(0.110, 0.145, 0.255) }, // 0x1c2541 night haze
  uColGroundHaze:    { value: new THREE.Vector3(0.560, 0.680, 0.600) }, // 0x8fada0 below-horizon
  uRayleighStrength: { value: 1.0 },
  uMieStrength:      { value: 1.0 },
  uSunDiscIntensity: { value: 1.4 },
  uMoonDiscIntensity:{ value: 0.9 },
  uStarIntensity:    { value: 0.6 },
  uDuskWarmth:       { value: 0.6 },
};

/* ------------------------------------------------------------------ */
/* 4. SKY VERTEX SHADER (fullscreen NDC quad)                          */
/*    FIXED: GLSL_NOISE removed (unused in vertex stage)              */
/* ------------------------------------------------------------------ */
export const SKY_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  // Fullscreen background quad at far depth
  gl_Position = vec4(position.xy, 0.9999, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. SKY FRAGMENT SHADER (cheap scattering + night pipeline, no clouds) */
/* ------------------------------------------------------------------ */
export const SKY_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform float uTimeOfDay;
uniform float uAspect;
uniform float uHorizonY;
uniform vec3  uColZenithDay;
uniform vec3  uColHorizonDay;
uniform vec3  uColZenithDawn;
uniform vec3  uColHorizonDawn;
uniform vec3  uColZenithNight;
uniform vec3  uColHorizonNight;
uniform vec3  uColGroundHaze;
uniform float uRayleighStrength;
uniform float uMieStrength;
uniform float uSunDiscIntensity;
uniform float uMoonDiscIntensity;
uniform float uStarIntensity;
uniform float uDuskWarmth;

varying vec2 vUv;

void main() {
  // 1. View ray (cheap fake perspective for horizon scenes)
  vec2 q = vUv - vec2(0.5, uHorizonY);
  vec3 rd = normalize(vec3(q.x * uAspect, q.y, 1.0)); // rd.z == 1.0 -> never zero
  float elev = rd.y;
  float up = clamp(elev, 0.0, 1.0);

  // 2. Time-of-day weights (correct low->high smoothstep order)
  float dayW = smoothstep(0.20, 0.35, uTimeOfDay) * (1.0 - smoothstep(0.65, 0.80, uTimeOfDay));
  float dawnW = smoothstep(0.05, 0.20, uTimeOfDay) * (1.0 - smoothstep(0.25, 0.40, uTimeOfDay))
              + smoothstep(0.60, 0.75, uTimeOfDay) * (1.0 - smoothstep(0.80, 0.95, uTimeOfDay));
  float nightW = (1.0 - smoothstep(0.05, 0.20, uTimeOfDay)) + smoothstep(0.80, 0.95, uTimeOfDay);
  float wSum = dayW + dawnW + nightW + 1e-3;
  dayW /= wSum; dawnW /= wSum; nightW /= wSum;

  vec3 zenith  = uColZenithDay * dayW + uColZenithDawn * dawnW + uColZenithNight * nightW;
  vec3 horizon = uColHorizonDay * dayW + uColHorizonDawn * dawnW + uColHorizonNight * nightW;

  // 3. Cheap Rayleigh scattering (blue intensifies at zenith)
  vec3 rayleigh = zenith * (0.35 + 0.65 * pow(up, 0.55)) * uRayleighStrength;

  // 4. Cheap Mie forward scatter + sun disc (guarded normalize)
  vec3 sunDir = normalize(uSunDir + vec3(1e-4, 1e-4, 1e-4));
  float sd = clamp(dot(rd, sunDir), 0.0, 1.0);
  float mie = pow(sd, 8.0) * 0.18 + pow(sd, 64.0) * 0.45;
  float sunDisc = smoothstep(0.99950, 0.99985, sd);
  float dayGate = clamp(dayW + dawnW, 0.0, 1.0);

  // 5. Night pipeline: moon disc + moon glow + hash-grid star twinkle
  vec3 moonDir = normalize(uMoonDir + vec3(1e-4, 1e-4, 1e-4));
  float md = clamp(dot(rd, moonDir), 0.0, 1.0);
  float moonDisc = smoothstep(0.99950, 0.99985, md);
  float moonGlow = pow(md, 32.0) * 0.12;
  float nightGate = clamp(nightW, 0.0, 1.0);

  vec3 srd = floor(rd * 60.0);
  float sh = h21(srd.xy + srd.z * 7.7);
  float star = step(0.998, sh) * smoothstep(0.0, 0.08, elev);
  float tw = 0.6 + 0.4 * sin(uTime * (2.0 + sh * 3.0) + sh * 40.0);

  // 6. Horizon atmospheric thickness (guarded division)
  float thickness = 1.0 / max(up + 0.06, 0.06);
  float haze = clamp(thickness * 0.10, 0.0, 1.0);

  // 7. Compose sky gradient + scatter
  vec3 col = mix(horizon, rayleigh, smoothstep(0.0, 0.30, up));
  vec3 warm = mix(horizon, uSunColor, 0.55);
  float warmMask = (1.0 - smoothstep(0.0, 0.22, up)) * pow(sd, 3.0) * uDuskWarmth;
  col = mix(col, warm, clamp(warmMask, 0.0, 1.0));
  col += uSunColor * (mie * uMieStrength + sunDisc * uSunDiscIntensity) * dayGate;
  col += uMoonColor * (moonDisc * uMoonDiscIntensity + moonGlow) * nightGate;
  col += vec3(0.90, 0.95, 1.00) * star * tw * uStarIntensity * nightGate;

  // 8. Below-horizon ground haze (blends into mountain layers)
  float below = 1.0 - smoothstep(-0.12, 0.0, elev);
  col = mix(col, uColGroundHaze, below);

  // 9. Atmospheric fog blend (palette utility from GLSL_COLOR_UTILS)
  col = applyFogBlend(col, uFogColor, haze * 0.30);

  // 10. De-banding dither (pixel-art friendly, kills mobile gradient banding)
  vec2 pUv = pxq(vUv * 24.0, 12.0);
  col += (h21(pUv) - 0.5) * 0.012;

  gl_FragColor = vec4(col, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/*    NOTE: pair with a fullscreen quad, e.g. new THREE.PlaneGeometry(2,2) */
/* ------------------------------------------------------------------ */
export function createSkyMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: SKY_UNIFORMS,
    vertexShader: SKY_VERTEX_SHADER,
    fragmentShader: SKY_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}

/* ------------------------------------------------------------------ */
/* 7. RUNTIME HELPERS                                                  */
/* ------------------------------------------------------------------ */
export function setSkyAspect(aspect) {
  SKY_UNIFORMS.uAspect.value = Math.max(0.2, Math.min(2.5, aspect));
}
export function setSkyTimeOfDay(t) {
  SKY_UNIFORMS.uTimeOfDay.value = t - Math.floor(t);
}
