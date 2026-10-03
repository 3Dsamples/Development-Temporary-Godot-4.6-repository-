// File : 008
// name : shaders/008_ChromeSphereShader.glsl.js
// description : Procedural reflective chrome/mirror sphere shader. ANALYZED & FIXED:
//               removed the duplicate `uniform float uTime` redeclaration in the
//               fragment stage (already declared by GLSL_GLOBALS, which can fault
//               strict Mali/Adreno compilers), wired the previously dead vTint
//               varying into real features (spec power + environment distortion
//               variation per instance size), replaced the stylized normal-based
//               environment lookup with a mathematically correct reflect() vector
//               for the sky/grass/sun-glow sampling, and guarded the sun-direction
//               normalize() against zero-length XY projections. Fake environment
//               reflection (sky top / grass bottom / sun glow spot), fresnel rim
//               darkening, cel-quantized specular highlight, distorted noise
//               reflections, and pixel-art quantization. Fully composed using
//               000_BaseShader.glsl.js and GLSL_COLOR_UTILS from 004_ColorPalette.js
//               with correct injection order (COLOR_UTILS before LIGHTING). All
//               smoothstep() calls use correct low->high edge order (GLSL ES safe).
//               Optimized for Android mobile with zero-texture GPU instancing.
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
/* 3. CHROME SPHERE UNIFORMS (JS side)                                 */
/* ------------------------------------------------------------------ */
export const CHROME_SPHERE_UNIFORMS = {
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
  uRimStrength:      { value: 0.55 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Chrome sphere specific colors (exact image palette)
  uColEnvSky:   { value: new THREE.Vector3(0.450, 0.620, 0.850) }, // 0x739ed9 sky reflection
  uColEnvGrass: { value: new THREE.Vector3(0.450, 0.720, 0.300) }, // 0x73b84d grass reflection
  uColSunGlow:  { value: new THREE.Vector3(1.000, 0.950, 0.750) }, // 0xfff2bf sun glint
  uColRimDark:  { value: new THREE.Vector3(0.250, 0.280, 0.330) }, // 0x404754 dark rim
  uChromeTint:  { value: new THREE.Vector3(0.900, 0.930, 0.970) }, // 0xe6edf7 chrome base
  uReflectivity:{ value: 0.85 },
  uSpecPower:   { value: 60.0 },
};

/* ------------------------------------------------------------------ */
/* 4. CHROME SPHERE VERTEX SHADER                                      */
/* ------------------------------------------------------------------ */
export const CHROME_SPHERE_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = small orb, 1.0 = large mirror sphere

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
/* 5. CHROME SPHERE FRAGMENT SHADER                                    */
/*    FIXED: no duplicate uTime declaration, vTint now used, correct   */
/*    reflect() environment sampling, guarded sun normalize            */
/* ------------------------------------------------------------------ */
export const CHROME_SPHERE_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColEnvSky;
uniform vec3  uColEnvGrass;
uniform vec3  uColSunGlow;
uniform vec3  uColRimDark;
uniform vec3  uChromeTint;
uniform float uReflectivity;
uniform float uSpecPower;

varying vec2  vUv;
varying float vSeed;
varying float vTint;

void main() {
  // 1. Sphere mask (correct smoothstep order, guarded sqrt)
  vec2 q = vUv - 0.5;
  float r = length(q) * 2.0;
  if (r > 1.0) discard;
  float alpha = 1.0 - smoothstep(0.96, 1.0, r);
  
  // 2. Reconstruct sphere normal (zero-length guarded)
  float z = sqrt(max(1.0 - r * r, 0.0));
  vec3 n = normalize(vec3(q * 2.0, max(z, 1e-4)));
  
  // 3. Correct reflection vector for environment sampling
  vec3 refl = reflect(-uViewDir, n);
  
  // Fake environment reflection (sky top / grass bottom via reflected Y)
  float horizon = smoothstep(-0.25, 0.25, refl.y);
  vec3 envCol = mix(uColEnvGrass, uColEnvSky, horizon);
  
  // Distorted noise reflections (per-instance variation, modulated by vTint)
  float envScale = mix(5.0, 4.0, vTint);
  float envNoise = fbm(vec2(refl.x * envScale + vSeed * 7.0, refl.y * envScale + vSeed * 3.0), 3);
  envCol = mix(envCol, envCol * 0.78, smoothstep(0.40, 0.70, envNoise) * 0.35);
  
  // Sun glow spot in reflection (guarded normalize of sun XY)
  vec2 sunXY = uSunDir.xy;
  float sunXYLen = max(length(sunXY), 1e-4);
  vec3 sunDir3 = normalize(vec3(sunXY / sunXYLen, 0.6));
  float sunDot = max(dot(refl, sunDir3), 0.0);
  envCol += uColSunGlow * pow(sunDot, 12.0) * 0.8;
  
  // Warm horizon band
  float band = 1.0 - smoothstep(0.0, 0.30, abs(refl.y - 0.05));
  envCol = mix(envCol, envCol * vec3(1.10, 1.02, 0.90), band * 0.25);
  
  // 4. Cel-shaded chrome base
  vec3 lit = computeCelLighting(uChromeTint, n, uViewDir);
  
  // 5. Combine reflection + cel base by reflectivity
  vec3 col = mix(lit, envCol * uChromeTint, uReflectivity);
  
  // 6. Cel-quantized specular highlight (vTint scales sharpness, guarded half-vector)
  vec3 hVec = uViewDir + sunDir3;
  float hLen = max(length(hVec), 1e-4);
  vec3 h = hVec / hLen;
  float liveSpecPower = mix(uSpecPower * 0.5, uSpecPower, vTint);
  float spec = pow(max(dot(n, h), 0.0), liveSpecPower);
  spec = floor(spec * uShadingSteps) / uShadingSteps;
  col += uColSunGlow * spec * 0.9;
  
  // 7. Fresnel rim darkening
  float fres = pow(1.0 - max(dot(n, uViewDir), 0.0), 3.0);
  col = mix(col, uColRimDark, fres * 0.55);
  
  // 8. Bottom contact shadow (grounding)
  float contact = 1.0 - smoothstep(0.55, 0.95, vUv.y);
  col *= 1.0 - contact * 0.18;
  
  // 9. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 16.0, 8.0);
  float pxNoise = h21(pUv + vSeed);
  col *= 0.95 + pxNoise * 0.10;
  
  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createChromeSphereMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: CHROME_SPHERE_UNIFORMS,
    vertexShader: CHROME_SPHERE_VERTEX_SHADER,
    fragmentShader: CHROME_SPHERE_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}
