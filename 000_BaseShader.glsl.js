// File : 000
// name : shaders/000_BaseShader.glsl.js
// description : Foundational GLSL shader base for the infinite procedural anime 
//               pixel-art forest/valley/winter scene. Integrates deterministic noise, 
//               pixel-art quantization, palette mixing, shadow tinting, rim lighting, 
//               fog blending, anime normal quantization, and seamless biome cross-fading. 
//               Fully compatible with 001_MathUtils.js, 004_ColorPalette.js, 
//               006_normalQuantizer.js, and 019_LightingSystem.js. 
//               Uses #ifndef guards to prevent redefinition errors when concatenated 
//               into element shaders. Zero-allocation GPU execution, maximum FPS on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

/* ------------------------------------------------------------------ */
/* 1. PRECISION & GLOBALS                                              */
/* ------------------------------------------------------------------ */
export const GLSL_GLOBALS = /* glsl */`
#ifndef GLSL_GLOBALS_INCLUDED
#define GLSL_GLOBALS_INCLUDED
precision highp float;
precision highp int;

uniform float uTime;
uniform float uPPU;
uniform vec2  uWind;
uniform vec3  uCamPos;
uniform vec3  uViewDir; // Fixed for orthographic top-down (0,0,1)
#endif
`;

/* ------------------------------------------------------------------ */
/* 2. DETERMINISTIC NOISE + PIXEL QUANTIZATION                         */
/* ------------------------------------------------------------------ */
export const GLSL_NOISE = /* glsl */`
#ifndef GLSL_NOISE_INCLUDED
#define GLSL_NOISE_INCLUDED

float h21(vec2 p){ 
  p = fract(p * vec2(123.34, 456.21)); 
  p += dot(p, p + 45.32); 
  return fract(p.x * p.y); 
}

float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); 
  f = f * f * (3.0 - 2.0 * f);
  float a = h21(i), b = h21(i + vec2(1.0, 0.0)), 
        c = h21(i + vec2(0.0, 1.0)), d = h21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int oct){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 6; i++) { 
    if (i >= oct) break; 
    s += a * vnoise(p); 
    p = p * 2.03 + 17.17; 
    a *= 0.5; 
  }
  return s;
}

vec2 pxq(vec2 w, float ppu){ 
  return floor(w * ppu) / ppu; 
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 3. PALETTE MIXING + COLOR UTILITIES (from 004_ColorPalette.js)      */
/*    Wrapped in #ifndef to prevent redefinition errors in element shaders */
/* ------------------------------------------------------------------ */
export const GLSL_PALETTE = /* glsl */`
#ifndef GLSL_PALETTE_INCLUDED
#define GLSL_PALETTE_INCLUDED

vec3 paletteMix5(vec3 c0, vec3 c1, vec3 c2, vec3 c3, vec3 c4, float t) {
  float s = clamp(t, 0.0, 1.0) * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

vec3 paletteMix4(vec3 c0, vec3 c1, vec3 c2, vec3 c3, float t) {
  float s = clamp(t, 0.0, 1.0) * 3.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  return mix(c2, c3, s - 2.0);
}

vec3 applyShadowTint(vec3 color, float shadowMask, vec3 tint) {
  return mix(color, color * tint * 1.4, clamp(shadowMask, 0.0, 1.0));
}

vec3 applyRimLight(vec3 color, float rimFactor, vec3 rimColor, float strength) {
  return color + rimColor * rimFactor * strength;
}

vec3 applyFogBlend(vec3 color, vec3 fogColor, float factor) {
  return mix(color, fogColor, clamp(factor, 0.0, 1.0));
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 4. ANIME NORMAL QUANTIZATION + RIM EXAGGERATION                     */
/*    (from 006_normalQuantizer.js) - Fixed for Orthographic Camera    */
/* ------------------------------------------------------------------ */
export const GLSL_NORMAL_QUANT = /* glsl */`
#ifndef GLSL_NORMAL_QUANT_INCLUDED
#define GLSL_NORMAL_QUANT_INCLUDED

vec3 getAnimeNormal(vec3 normal, vec3 lightDir, float steps, vec3 viewDir, float rimPower) {
  float lightDot = dot(normalize(normal), normalize(lightDir));
  float stepSize = 1.0 / steps;
  float normalizedDot = (lightDot * 0.5) + 0.5;
  float quantizedDot = floor(normalizedDot / stepSize) * stepSize;
  quantizedDot = (quantizedDot * 2.0) - 1.0;
  
  vec3 right = normalize(cross(lightDir, viewDir));
  float rightLen = length(right);
  right = right / max(rightLen, 0.001);
  
  float angle = acos(clamp(quantizedDot, -1.0, 1.0));
  vec3 quantizedNormal = (lightDir * cos(angle)) + (right * sin(angle));
  
  float silhouetteMask = 1.0 - abs(dot(normalize(normal), viewDir));
  vec3 finalNormal = normalize(mix(normal, quantizedNormal, smoothstep(0.1, 0.3, silhouetteMask)));
  
  float rimFactor = 1.0 - max(dot(finalNormal, viewDir), 0.0);
  rimFactor = pow(rimFactor, rimPower);
  return normalize(mix(finalNormal, viewDir, rimFactor * 0.3));
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 5. LIGHTING UNIFORMS + CALCULATIONS (from 019_LightingSystem.js)    */
/*    Exact uniform names matched to LightingSystem.js                 */
/* ------------------------------------------------------------------ */
export const GLSL_LIGHTING = /* glsl */`
#ifndef GLSL_LIGHTING_INCLUDED
#define GLSL_LIGHTING_INCLUDED

uniform vec3  uSunDir;
uniform vec3  uMoonDir;
uniform vec3  uSunColor;
uniform vec3  uMoonColor;
uniform vec3  uSkyColor;
uniform vec3  uGroundColor;
uniform vec3  uFogColor;
uniform vec3  uShadowTintColor;
uniform vec3  uRimColor;
uniform float uSunIntensity;
uniform float uMoonIntensity;
uniform float uSkyStrength;
uniform float uGroundStrength;
uniform float uBounceStrength;
uniform float uRimStrength;
uniform float uShadingSteps;

vec3 computeCelLighting(vec3 baseColor, vec3 normal, vec3 viewDir) {
  float sunDot = dot(normal, uSunDir);
  float moonDot = dot(normal, uMoonDir);
  
  float sunShadow = step(0.0, sunDot);
  float sunLight = (sunDot * 0.5 + 0.5);
  float sunQuant = floor(sunLight * uShadingSteps) / uShadingSteps;
  sunQuant = sunQuant * 2.0 - 1.0;
  
  float moonShadow = step(0.0, moonDot);
  float moonLight = (moonDot * 0.5 + 0.5);
  float moonQuant = floor(moonLight * uShadingSteps) / uShadingSteps;
  moonQuant = moonQuant * 2.0 - 1.0;
  
  float hemiDot = dot(normal, vec3(0.0, 1.0, 0.0));
  vec3 hemiColor = mix(uGroundColor, uSkyColor, hemiDot * 0.5 + 0.5);
  
  vec3 col = baseColor * hemiColor * uSkyStrength;
  col += baseColor * uSunColor * max(sunQuant, 0.0) * uSunIntensity * sunShadow;
  col += baseColor * uMoonColor * max(moonQuant, 0.0) * uMoonIntensity * moonShadow;
  
  float shadowMask = 1.0 - (sunShadow * 0.7 + moonShadow * 0.3);
  col = applyShadowTint(col, shadowMask, uShadowTintColor);
  
  float rimFactor = 1.0 - max(dot(normal, viewDir), 0.0);
  rimFactor = pow(rimFactor, 3.0);
  col = applyRimLight(col, rimFactor, uRimColor, uRimStrength);
  
  return col;
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 6. BIOME BLENDING (Fixed logic, WebGL1/2 safe)                      */
/* ------------------------------------------------------------------ */
export const GLSL_BIOME = /* glsl */`
#ifndef GLSL_BIOME_INCLUDED
#define GLSL_BIOME_INCLUDED

uniform vec4  uBiomeW;       // forest, meadow, rock, alpine weights
uniform vec3  uValley;       // centerX, halfWidth, slopeAtCam
uniform float uTransMix;
uniform float uTransFrom;
uniform float uTransTo;

vec3 getBiomeColor(int idx, vec3 cF, vec3 cM, vec3 cR, vec3 cA) {
  if (idx == 0) return cF;
  if (idx == 1) return cM;
  if (idx == 2) return cR;
  return cA;
}

vec3 biomeBlend(vec3 colForest, vec3 colMeadow, vec3 colRock, vec3 colAlpine) {
  vec3 blended = colForest * uBiomeW.x + colMeadow * uBiomeW.y + colRock * uBiomeW.z + colAlpine * uBiomeW.w;
  
  if (uTransMix < 1.0) {
    vec3 fromCol = getBiomeColor(int(uTransFrom), colForest, colMeadow, colRock, colAlpine);
    vec3 toCol = getBiomeColor(int(uTransTo), colForest, colMeadow, colRock, colAlpine);
    blended = mix(fromCol, toCol, uTransMix);
  }
  return blended;
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 7. FULL BASE SHADER COMPOSITION                                     */
/* ------------------------------------------------------------------ */
export const GLSL_BASE = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_PALETTE}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
`;

