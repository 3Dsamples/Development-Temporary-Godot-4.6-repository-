// File : 007
// name : shaders/007_TreasureChestShader.glsl.js
// description : Procedural wooden treasure chest shader with metal bands and brass lock.
//               ANALYZED & FIXED: replaced all reversed-edge smoothstep() calls (undefined
//               behavior on Mali/Adreno per GLSL ES spec) with correct low->high order,
//               removed unused varyings (vWorld) to save mobile varying registers, wired
//               vTint into a real open-chest golden glow feature, wired uTime into a brass
//               glint, used the previously dead lidLine as a lid seam darkener, guarded
//               normalize(vUv-0.5) against the zero-length singularity at uv center, and
//               split the brass keyhole into its own darkening mask. Fully composed using
//               000_BaseShader.glsl.js and GLSL_COLOR_UTILS from 004_ColorPalette.js with
//               correct injection order (COLOR_UTILS before LIGHTING). Zero-allocation,
//               Android-mobile optimized, pixel-art quantized, exact palette, no text labels.
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
/* 3. TREASURE CHEST UNIFORMS (JS side)                                */
/* ------------------------------------------------------------------ */
export const TREASURE_CHEST_UNIFORMS = {
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

  // Chest specific colors (exact image palette)
  uColWoodLight: { value: new THREE.Vector3(0.545, 0.353, 0.169) }, // 0x8B5A2B
  uColWoodDark:  { value: new THREE.Vector3(0.361, 0.227, 0.118) }, // 0x5C3A1E
  uColWoodDeep:  { value: new THREE.Vector3(0.220, 0.130, 0.070) }, // 0x382112
  uColMetalDark: { value: new THREE.Vector3(0.165, 0.165, 0.165) }, // 0x2A2A2A
  uColMetalLight:{ value: new THREE.Vector3(0.416, 0.416, 0.416) }, // 0x6A6A6A
  uColBrass:     { value: new THREE.Vector3(0.541, 0.478, 0.353) }, // 0x8A7A5A
  uColBrassDark: { value: new THREE.Vector3(0.350, 0.300, 0.200) }, // 0x594D33
};

/* ------------------------------------------------------------------ */
/* 4. TREASURE CHEST VERTEX SHADER                                     */
/*    FIXED: removed unused vWorld varying, vTint assigned before use  */
/* ------------------------------------------------------------------ */
export const TREASURE_CHEST_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = closed, 1.0 = slightly open

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
  
  // Chests are rigid: tiny settle animation only when closed
  float settle = sin(uTime * 1.5 + aSeed * 10.0) * 0.02 * (1.0 - vTint);
  w.y += settle;
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. TREASURE CHEST FRAGMENT SHADER                                   */
/*    FIXED: all smoothstep() use correct low->high edge order         */
/* ------------------------------------------------------------------ */
export const TREASURE_CHEST_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColWoodLight;
uniform vec3  uColWoodDark;
uniform vec3  uColWoodDeep;
uniform vec3  uColMetalDark;
uniform vec3  uColMetalLight;
uniform vec3  uColBrass;
uniform vec3  uColBrassDark;
uniform float uTime;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

// Horizontal wood plank lines (correct smoothstep order)
float woodPlanks(vec2 uv) {
  float line = 1.0 - smoothstep(0.0, 0.02, abs(fract(uv.y * 4.0) - 0.5) - 0.45);
  return line;
}

// Wood grain noise
float woodGrain(vec2 uv, float seed) {
  return fbm(vec2(uv.x * 15.0, uv.y * 2.0) + seed * 10.0, 3);
}

// Metal band with rivets (correct smoothstep order)
float metalBand(vec2 uv, float xPos, float seed) {
  float band = 1.0 - smoothstep(0.0, 0.04, abs(uv.x - xPos) - 0.03);
  float rivet1 = 1.0 - smoothstep(0.0, 0.025, length(uv - vec2(xPos, 0.3 + sin(seed) * 0.1)));
  float rivet2 = 1.0 - smoothstep(0.0, 0.025, length(uv - vec2(xPos, 0.7 + cos(seed) * 0.1)));
  return max(band, max(rivet1, rivet2));
}

// Brass lock body + hinge (correct smoothstep order)
float brassLockBody(vec2 uv) {
  float body = 1.0 - smoothstep(0.06, 0.10, length(uv - vec2(0.5, 0.5)));
  float hinge = 1.0 - smoothstep(0.02, 0.035, length(uv - vec2(0.5, 0.42)));
  return max(body, hinge);
}

// Keyhole as separate darkening mask (correct smoothstep order)
float brassKeyhole(vec2 uv) {
  return 1.0 - smoothstep(0.012, 0.02, length(uv - vec2(0.5, 0.52)));
}

void main() {
  // 1. Base chest shape (rounded rectangle)
  vec2 boxUV = vUv - 0.5;
  float boxDist = length(max(abs(boxUV) - vec2(0.42, 0.35), 0.0));
  float boxShape = 1.0 - smoothstep(0.0, 0.04, boxDist);
  
  if (boxShape < 0.1) discard;

  // 2. Wood base color and texture
  float grain = woodGrain(vUv, vSeed);
  float planks = woodPlanks(vUv);
  
  vec3 woodCol = mix(uColWoodLight, uColWoodDark, grain);
  woodCol = mix(woodCol, uColWoodDeep, planks * 0.4);
  
  // 3. Metal bands
  float band1 = metalBand(vUv, 0.25, vSeed);
  float band2 = metalBand(vUv, 0.75, vSeed + 5.0);
  float metalMask = max(band1, band2);
  
  vec3 metalCol = mix(uColMetalDark, uColMetalLight, fbm(vUv * 20.0 + vSeed, 2));
  
  // 4. Brass lock + keyhole
  float lockMask = brassLockBody(vUv);
  float keyhole = brassKeyhole(vUv);
  vec3 lockCol = mix(uColBrassDark, uColBrass, fbm(vUv * 15.0 + vSeed, 2));

  // 5. Combine materials
  vec3 baseColor = woodCol;
  baseColor = mix(baseColor, metalCol, metalMask);
  baseColor = mix(baseColor, lockCol, lockMask);

  // 6. Cel-shaded lighting
  vec3 normal = vec3(0.0, 0.0, 1.0);
  vec3 litColor = computeCelLighting(baseColor, normal, uViewDir);

  // 7. Specular highlights on metal and brass (guarded normalize)
  vec2 specDir = vUv - 0.5;
  float specLen = max(length(specDir), 1e-4);
  vec2 nd = specDir / specLen;
  float sunAlign = max(dot(nd, normalize(uSunDir.xy + vec2(1e-4))), 0.0);
  float metalSpec = metalMask * smoothstep(0.6, 0.9, sunAlign);
  float brassSpec = lockMask * smoothstep(0.5, 0.8, sunAlign);
  litColor += vec3(0.3) * metalSpec + vec3(0.4, 0.35, 0.2) * brassSpec;

  // 8. Keyhole darkening
  litColor = mix(litColor, litColor * 0.35, keyhole);

  // 9. Lid seam darkening (previously dead lidLine now used)
  float lidLine = 1.0 - smoothstep(0.005, 0.02, abs(vUv.y - 0.5));
  litColor *= 1.0 - lidLine * 0.25;

  // 10. Open-chest golden glow (uses vTint + uTime, justifies both)
  if (vTint > 0.5) {
    float innerGlow = 1.0 - smoothstep(0.10, 0.42, length(vUv - vec2(0.5, 0.5)));
    float seamGlow = 1.0 - smoothstep(0.0, 0.06, abs(vUv.y - 0.5));
    float flicker = 0.8 + 0.2 * sin(uTime * 3.0 + vSeed * 6.28);
    litColor += uColBrass * (innerGlow * 0.25 + seamGlow * 0.35) * vTint * flicker;
  }

  // 11. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.95 + pxNoise * 0.10;

  // 12. Edge darkening (ambient occlusion)
  float edgeAO = 1.0 - smoothstep(0.35, 0.45, length(boxUV));
  litColor *= 0.85 + edgeAO * 0.15;

  gl_FragColor = vec4(litColor, boxShape);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createTreasureChestMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: TREASURE_CHEST_UNIFORMS,
    vertexShader: TREASURE_CHEST_VERTEX_SHADER,
    fragmentShader: TREASURE_CHEST_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}