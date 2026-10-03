// File : 010
// name : shaders/010_RocksShader.glsl.js
// description : Procedural gray boulder cluster shader (rounded striated boulders,
//               tan and dark variants, moss patches). ANALYZED & FIXED: replaced the
//               reversed-edge smoothstep(0.25, 0.02, vUv.y) in the bottom contact
//               shadow with correct low->high order 1.0 - smoothstep(0.02, 0.25, vUv.y)
//               (reversed edges are undefined behavior per GLSL ES spec and can fault
//               Mali/Adreno drivers), verified every other smoothstep() uses correct
//               low->high edge order, verified atan()/sqrt()/normalize() singularities
//               are guarded (facet normalize is provably non-zero because n.z >= 0.25),
//               confirmed no uniform is redeclared (uTime/uPPU/uWind/uCamPos/uViewDir
//               come from GLSL_GLOBALS), confirmed all varyings (vUv/vSeed/vTint) are
//               consumed in the fragment stage, and confirmed GLSL_COLOR_UTILS is
//               injected before GLSL_LIGHTING so computeCelLighting() resolves
//               applyShadowTint()/applyRimLight(). Features noise-distorted blob shapes
//               with flattened contact base, quantized facet normals for flat cel-shaded
//               rock faces, per-variant palettes (gray / tan / dark / mossy), top
//               highlights, bottom contact shadow, and pixel-art quantization.
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
/* 3. ROCKS UNIFORMS (JS side)                                         */
/* ------------------------------------------------------------------ */
export const ROCKS_UNIFORMS = {
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
  uRimStrength:      { value: 0.35 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Rocks specific colors (exact image palette)
  uColRockLight:   { value: new THREE.Vector3(0.604, 0.627, 0.651) }, // 0x9aa0a6 gray light
  uColRockMid:     { value: new THREE.Vector3(0.420, 0.439, 0.463) }, // 0x6b7076 gray mid
  uColRockDark:    { value: new THREE.Vector3(0.263, 0.282, 0.302) }, // 0x43484d gray dark
  uColRockTan:     { value: new THREE.Vector3(0.659, 0.596, 0.502) }, // 0xa89880 tan light
  uColRockTanDark: { value: new THREE.Vector3(0.478, 0.416, 0.333) }, // 0x7a6a55 tan dark
  uColMoss:        { value: new THREE.Vector3(0.290, 0.478, 0.227) }, // 0x4a7a3a moss green
  uColContact:     { value: new THREE.Vector3(0.150, 0.170, 0.200) }, // 0x262b33 contact shadow
  uFacetStrength:  { value: 0.55 },
};

/* ------------------------------------------------------------------ */
/* 4. ROCKS VERTEX SHADER                                              */
/*    Rigid element: no wind sway, minimal varyings for mobile GPUs   */
/* ------------------------------------------------------------------ */
export const ROCKS_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = gray, 1.0 = tan, 2.0 = dark, 3.0 = mossy

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
  
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. ROCKS FRAGMENT SHADER                                            */
/*    FIXED: contact shadow smoothstep now correct low->high order    */
/* ------------------------------------------------------------------ */
export const ROCKS_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColRockLight;
uniform vec3  uColRockMid;
uniform vec3  uColRockDark;
uniform vec3  uColRockTan;
uniform vec3  uColRockTanDark;
uniform vec3  uColMoss;
uniform vec3  uColContact;
uniform float uFacetStrength;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

void main() {
  // 1. Boulder shape: noise-distorted blob with flattened contact base
  vec2 q = vUv - 0.5;
  float aspect = mix(1.15, 0.90, fract(vSeed * 3.7));
  q.x /= max(aspect, 1e-4);
  float r = length(q) * 2.0;
  
  float ang = atan(q.y, q.x + 1e-6); // guarded atan
  float wob = sin(ang * 5.0 + vSeed * 12.0) * 0.10
            + sin(ang * 9.0 + vSeed * 21.0) * 0.06;
  float edge = 0.92 + wob;
  
  float shape = 1.0 - smoothstep(edge - 0.10, edge, r);      // soft rim (low->high)
  shape *= 1.0 - smoothstep(0.40, 0.48, -q.y);               // flatten base (low->high)
  if (shape < 0.05) discard;
  
  // 2. Pseudo sphere normal (guarded sqrt), flattened for boulder look
  float nz = max(sqrt(max(1.0 - r * r * 0.6, 0.0)), 0.25);
  vec3 n = normalize(vec3(q * 1.6, nz));
  
  // 3. Quantized facet normal -> flat cel rock faces (provably non-zero: n.z >= 0.25)
  vec3 nf = normalize(floor(n * 4.0) + 0.5);
  vec3 fn = normalize(mix(n, nf, uFacetStrength));
  
  // 4. Per-variant palette selection
  vec3 lightC = uColRockLight;
  vec3 midC   = uColRockMid;
  vec3 darkC  = uColRockDark;
  if (vTint > 0.5 && vTint < 1.5) {
    lightC = uColRockTan;
    midC   = uColRockTanDark;
    darkC  = uColRockTanDark * 0.80;
  } else if (vTint > 1.5 && vTint < 2.5) {
    lightC = uColRockMid;
    midC   = uColRockDark;
    darkC  = uColRockDark * 0.80;
  }
  
  // 5. Facet-driven base color (flat per-face tones)
  float facetNoise = h21(floor(n * 4.0).xy + vSeed);
  vec3 baseColor = mix(midC, lightC, facetNoise * 0.60);
  baseColor = mix(baseColor, darkC, smoothstep(0.55, 0.85, fbm(vUv * 6.0 + vSeed * 13.0, 2)) * 0.45);
  
  // 6. Cel-shaded lighting on facet normal
  vec3 litColor = computeCelLighting(baseColor, fn, uViewDir);
  
  // 7. Moss patches (mossy variant, upper faces only)
  if (vTint > 2.5) {
    float mossNoise = fbm(vUv * 7.0 + vSeed * 31.0, 3);
    float mossMask = smoothstep(0.55, 0.75, mossNoise) * smoothstep(0.0, 0.40, n.y);
    litColor = mix(litColor, uColMoss * (0.85 + facetNoise * 0.30), mossMask * 0.70);
  }
  
  // 8. Top highlight + bottom contact shadow (grounding)
  //    FIXED: smoothstep edges corrected to low->high (GLSL ES safe)
  litColor += lightC * smoothstep(0.55, 0.90, n.y) * 0.18;
  litColor = mix(litColor, uColContact, (1.0 - smoothstep(0.02, 0.25, vUv.y)) * 0.45);
  
  // 9. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 14.0, 7.0);
  float pxNoise = h21(pUv + vSeed);
  litColor *= 0.94 + pxNoise * 0.12;
  
  // 10. Edge softening
  float alpha = shape * smoothstep(0.0, 0.12, shape);
  
  gl_FragColor = vec4(litColor, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createRocksMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: ROCKS_UNIFORMS,
    vertexShader: ROCKS_VERTEX_SHADER,
    fragmentShader: ROCKS_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}
