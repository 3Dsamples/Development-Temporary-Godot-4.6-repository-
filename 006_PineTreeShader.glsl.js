// File : 006
// name : shaders/006_PineTreeShader.glsl.js
// description : Procedural pine tree shader with glowing light effects and floating
//               particles/bubbles. Features layered conifer foliage with sun rays
//               shining through, animated floating orbs, and detailed needle clusters.
//               Fully composed using 000_BaseShader.glsl.js and GLSL_COLOR_UTILS
//               from 004_ColorPalette.js. Optimized for Android mobile with GPU
//               instancing and zero-texture procedural shaders.
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
/* 3. PINE TREE UNIFORMS (JS side)                                     */
/* ------------------------------------------------------------------ */
export const PINE_TREE_UNIFORMS = {
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
  uRimStrength:      { value: 0.50 },
  uShadingSteps:     { value: 4.0 },

  // Biome blending
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // Pine tree specific colors
  uColNeedleDark:  { value: new THREE.Vector3(0.102, 0.290, 0.161) }, // 0x1a4a29
  uColNeedleMid:   { value: new THREE.Vector3(0.165, 0.400, 0.227) }, // 0x2a663a
  uColNeedleLight: { value: new THREE.Vector3(0.227, 0.510, 0.290) }, // 0x3a824a
  uColNeedleTip:   { value: new THREE.Vector3(0.302, 0.620, 0.353) }, // 0x4d9e5a
  uColTrunk:       { value: new THREE.Vector3(0.290, 0.196, 0.114) }, // 0x4a321d
  uColGlow:        { value: new THREE.Vector3(1.000, 0.922, 0.502) }, // 0xffeb80
  uColParticle:    { value: new THREE.Vector3(1.000, 1.000, 0.902) }, // 0xffffe6
  uWindStrength:   { value: 0.5 },
  uGlowIntensity:  { value: 1.0 },
};

/* ------------------------------------------------------------------ */
/* 4. PINE TREE VERTEX SHADER                                          */
/* ------------------------------------------------------------------ */
export const PINE_TREE_VERTEX_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

attribute vec2  aOff;
attribute vec2  aSize;
attribute float aRot;
attribute float aSeed;
attribute float aTint; // 0.0 = trunk, 1.0 = foliage, 2.0 = particle

uniform vec2 uWind;
uniform float uWindStrength;

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
  
  // Pixel-art quantization on world position
  vec2 w = pxq(aOff + p, uPPU);
  
  // Wind sway (stronger for foliage, minimal for trunk/particles)
  float isFoliage = step(0.5, vTint) * (1.0 - step(1.5, vTint));
  float swayAmount = isFoliage * uWindStrength * 0.6;
  vec2 windOffset = uWind * swayAmount * sin(uTime * 2.0 + aSeed * 25.0 + vUv.y * 2.0) * vUv.y;
  w += windOffset;
  
  vWorld = w;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(w, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. PINE TREE FRAGMENT SHADER                                        */
/* ------------------------------------------------------------------ */
export const PINE_TREE_FRAGMENT_SHADER = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}      // MUST be before GLSL_LIGHTING
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}

uniform vec3  uColNeedleDark;
uniform vec3  uColNeedleMid;
uniform vec3  uColNeedleLight;
uniform vec3  uColNeedleTip;
uniform vec3  uColTrunk;
uniform vec3  uColGlow;
uniform vec3  uColParticle;
uniform float uWindStrength;
uniform float uGlowIntensity;

varying vec2  vUv;
varying vec2  vWorld;
varying float vSeed;
varying float vTint;

// Pine needle cluster shape
float needleCluster(vec2 uv, vec2 center, float size, float seed) {
  float d = length(uv - center);
  float angle = atan(uv.y - center.y, uv.x - center.x);
  // Pine needle spikes
  float spikes = sin(angle * 16.0 + seed * 7.0) * 0.08;
  spikes += sin(angle * 24.0 + seed * 11.0) * 0.05;
  float noisyRadius = size + spikes;
  float noise = fbm(uv * 10.0 + seed, 3) * 0.08;
  return smoothstep(noisyRadius + 0.06, noisyRadius - 0.08, d + noise);
}

// Glowing orb/particle shape
float glowOrb(vec2 uv, vec2 center, float radius, float seed) {
  float d = length(uv - center);
  float glow = exp(-d * d / (radius * radius * 0.5));
  float core = smoothstep(radius * 0.3, 0.0, d);
  return glow * 0.6 + core * 0.4;
}

void main() {
  // 1. Determine element type based on tint
  float isTrunk = 1.0 - step(0.5, vTint);
  float isFoliage = step(0.5, vTint) * (1.0 - step(1.5, vTint));
  float isParticle = step(1.5, vTint);

  vec3 col = vec3(0.0);
  float alpha = 1.0;

  // 2. Trunk rendering
  if (isTrunk > 0.5) {
    // Trunk shape with bark texture
    float edgeNoise = fbm(vUv * 8.0 + vSeed, 3) * 0.1;
    float shape = 1.0 - smoothstep(0.42 - edgeNoise, 0.50 + edgeNoise, abs(vUv.x - 0.5));
    shape *= smoothstep(0.0, 0.05, vUv.y);
    shape *= smoothstep(1.0, 0.95, vUv.y);
    
    if (shape < 0.1) discard;
    
    // Bark texture
    float barkNoise = fbm(vUv * 12.0 + vSeed * 15.0, 4);
    float barkStreaks = smoothstep(0.3, 0.7, sin(vUv.y * 40.0 + barkNoise * 10.0));
    col = mix(uColTrunk, uColTrunk * 0.8, barkNoise);
    col *= 0.7 + barkStreaks * 0.3;
    alpha = shape;
  }
  
  // 3. Foliage rendering (pine needle clusters)
  else if (isFoliage > 0.5) {
    // Create layered pine foliage using multiple needle clusters
    vec2 center1 = vec2(0.5 + sin(vSeed * 11.0) * 0.12, 0.5 + cos(vSeed * 13.0) * 0.10);
    vec2 center2 = vec2(0.38 + cos(vSeed * 9.0) * 0.15, 0.62 + sin(vSeed * 7.0) * 0.12);
    vec2 center3 = vec2(0.62 + sin(vSeed * 5.0) * 0.14, 0.38 + cos(vSeed * 17.0) * 0.11);
    vec2 center4 = vec2(0.45 + cos(vSeed * 19.0) * 0.10, 0.55 + sin(vSeed * 3.0) * 0.13);
    
    float radius1 = 0.38 + fbm(vUv * 3.0 + vSeed, 2) * 0.06;
    float radius2 = 0.34 + fbm(vUv * 4.0 + vSeed + 10.0, 2) * 0.05;
    float radius3 = 0.32 + fbm(vUv * 5.0 + vSeed + 20.0, 2) * 0.07;
    float radius4 = 0.30 + fbm(vUv * 6.0 + vSeed + 30.0, 2) * 0.04;
    
    float shape = needleCluster(vUv, center1, radius1, vSeed);
    shape = max(shape, needleCluster(vUv, center2, radius2, vSeed + 5.0));
    shape = max(shape, needleCluster(vUv, center3, radius3, vSeed + 10.0));
    shape = max(shape, needleCluster(vUv, center4, radius4, vSeed + 15.0));
    
    // Add fine needle detail
    float needleDetail = fbm(vUv * 18.0 + vSeed * 22.0, 4);
    float detailMask = smoothstep(0.35, 0.65, needleDetail);
    shape *= (0.75 + detailMask * 0.25);
    
    if (shape < 0.06) discard;
    
    // Color variation based on height and noise
    float colorNoise = fbm(vUv * 5.0 + vSeed + uTime * 0.02, 3);
    float heightGradient = smoothstep(0.0, 1.0, vUv.y);
    
    col = mix(uColNeedleDark, uColNeedleMid, colorNoise);
    col = mix(col, uColNeedleLight, heightGradient * 0.4);
    col = mix(col, uColNeedleTip, step(0.7, heightGradient) * 0.3);
    
    // Cel-shaded lighting
    vec3 normal = vec3(0.0, 0.0, 1.0);
    col = computeCelLighting(col, normal, uViewDir);
    
    // Glowing light effect (sun rays through tree)
    float glowNoise = fbm(vUv * 4.0 + vSeed * 10.0, 3);
    float glowMask = smoothstep(0.5, 0.8, glowNoise) * smoothstep(0.3, 0.7, vUv.y);
    col = mix(col, uColGlow, glowMask * 0.4 * uGlowIntensity);
    
    alpha = shape;
  }
  
  // 4. Particle/glowing orb rendering
  else if (isParticle > 0.5) {
    // Animated floating orb
    float orbPhase = uTime * 0.5 + vSeed * 6.28;
    vec2 orbCenter = vec2(0.5 + sin(orbPhase) * 0.1, 0.5 + cos(orbPhase * 0.7) * 0.1);
    float orbRadius = 0.15 + sin(uTime * 2.0 + vSeed * 10.0) * 0.03;
    
    float orbShape = glowOrb(vUv, orbCenter, orbRadius, vSeed);
    
    if (orbShape < 0.05) discard;
    
    col = uColParticle;
    col *= 0.8 + orbShape * 0.4;
    alpha = orbShape * 0.8;
  }

  // 5. Pixel-art dithering / quantization
  vec2 pUv = pxq(vUv * 14.0, 7.0);
  float pxNoise = h21(pUv + vSeed);
  col *= 0.94 + pxNoise * 0.12;

  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. MATERIAL FACTORY (Zero-allocation, ready for Three.js r185)      */
/* ------------------------------------------------------------------ */
export function createPineTreeMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: PINE_TREE_UNIFORMS,
    vertexShader: PINE_TREE_VERTEX_SHADER,
    fragmentShader: PINE_TREE_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  });
}
