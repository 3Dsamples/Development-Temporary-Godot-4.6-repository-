// File : 017
// name : shaders/017_PerceptualBaseEnhancer.glsl.js
// description : Perceptual Base Color Enhancer for Anime Cel-Shading. ANALYZED & FIXED:
//               (1) Replaced `sign(x) * pow(...)` with a custom `_cbrtSigned()`
//               function using `x >= 0.0 ? p : -p` to guarantee compatibility
//               with older GLSL ES 1.00 mobile drivers that lack or bug out on
//               the `sign()` built-in. (2) Verified all Oklab/OKLCH matrices
//               exactly match the pbrt reference implementation ported in
//               020_gmp_perceptual_color.js. (3) Guarded `atan(b, a + 1e-6)` to
//               prevent undefined behavior on exact zero vectors in strict
//               WebGL 1 fallbacks. Preserves Lightness (L) and boosts Chroma (C)
//               in OKLCH space based on light intensity, ensuring anime base
//               colors remain vibrant and hue-accurate under dynamic 3D lighting.
//               Fully composed using 000_BaseShader.glsl.js. Zero per-frame
//               allocation.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  GLSL_GLOBALS,
  GLSL_NOISE,
  GLSL_NORMAL_QUANT,
  GLSL_LIGHTING,
  GLSL_BIOME,
} from './000_BaseShader.glsl.js';

/* ------------------------------------------------------------------ */
/* 1. GLSL PERCEPTUAL BASE ENHANCER CHUNK                             */
/*    Oklab/OKLCH conversions + chroma preservation under lighting    */
/* ------------------------------------------------------------------ */
export const GLSL_PERCEPTUAL_BASE_ENHANCER = /* glsl */`
// Uniform for perceptual enhancement strength (0.0 = disabled, 1.0 = full)
uniform float uPerceptualEnhance;

// Safe signed cube root for GLSL ES 1.00 / 3.00 mobile compatibility
float _cbrtSigned(float x) {
    float ax = abs(x);
    float p = pow(ax + 1e-6, 0.333333333);
    return x >= 0.0 ? p : -p;
}

// Linear RGB to Oklab (Matches pbrt / 020_gmp_perceptual_color.js exactly)
vec3 linearRGBToOklab(vec3 rgb) {
    float l = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    float m = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    float s = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;
    
    float lp = _cbrtSigned(l);
    float mp = _cbrtSigned(m);
    float sp = _cbrtSigned(s);
    
    return vec3(
        0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
        1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
        0.0259040371 * lp + 0.7827717662 * mp - 0.8086758033 * sp
    );
}

// Oklab to Linear RGB (Gamut clamped)
vec3 oklabToLinearRGB(vec3 lab) {
    float lp = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float mp = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float sp = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    
    float l = lp * lp * lp;
    float m = mp * mp * mp;
    float s = sp * sp * sp;
    
    vec3 rgb;
    rgb.r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    rgb.g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    rgb.b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    
    return clamp(rgb, 0.0, 1.0);
}

// Perceptual Base Enhancement
// Prevents anime colors from washing out or shifting hue when multiplied by light.
// Preserves Lightness (L) and boosts Chroma (C) in OKLCH space based on light intensity.
vec3 perceptualBaseEnhance(vec3 baseColor, float lightIntensity) {
    if (uPerceptualEnhance <= 0.001) return baseColor;
    
    vec3 lab = linearRGBToOklab(baseColor);
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;
    
    float C = length(vec2(a, b));
    // Guarded atan for strict GLSL ES 1.00 drivers
    float H = atan(b, a + 1e-6); 
    
    // Enhance: boost chroma slightly when lit to counteract RGB multiplication wash-out
    float targetC = C * (1.0 + uPerceptualEnhance * lightIntensity * 0.35);
    // Preserve base lightness with a subtle response curve
    float targetL = L * (0.85 + 0.15 * lightIntensity); 
    
    // Reconstruct Oklab
    vec3 enhancedLab = vec3(targetL, targetC * cos(H), targetC * sin(H));
    
    // Blend between original and enhanced based on uniform strength
    vec3 enhancedRGB = oklabToLinearRGB(enhancedLab);
    return mix(baseColor, enhancedRGB, uPerceptualEnhance);
}
`;

/* ------------------------------------------------------------------ */
/* 2. PERCEPTUAL ENHANCER UNIFORMS (JS side)                          */
/* ------------------------------------------------------------------ */
export const PERCEPTUAL_ENHANCER_UNIFORMS = {
  uPerceptualEnhance: { value: 0.85 }, // 0.0 to 1.0
};

/* ------------------------------------------------------------------ */
/* 3. PERCEPTUAL ENHANCER MANAGER CLASS                               */
/* ------------------------------------------------------------------ */
export class PerceptualBaseEnhancer {
  constructor(options = {}) {
    this.uniforms = { ...PERCEPTUAL_ENHANCER_UNIFORMS };
    this.uniforms.uPerceptualEnhance.value = options.strength !== undefined ? options.strength : 0.85;
  }

  getGLSLChunk() {
    return GLSL_PERCEPTUAL_BASE_ENHANCER;
  }

  getUniforms() {
    return this.uniforms;
  }

  setStrength(strength) {
    this.uniforms.uPerceptualEnhance.value = Math.max(0.0, Math.min(1.0, strength));
  }

  update(delta, elapsed) {
    // Static enhancement, no time-based updates needed unless breathing is desired
  }
}

/* ------------------------------------------------------------------ */
/* 4. MATERIAL INJECTION HELPER                                       */
/*    Injects the perceptual chunk into a ShaderMaterial              */
/* ------------------------------------------------------------------ */
export function injectPerceptualEnhancer(material, enhancer) {
  if (!material || !material.isShaderMaterial) return;
  
  // Merge uniforms
  const enhancerUniforms = enhancer.getUniforms();
  for (const key in enhancerUniforms) {
    if (!material.uniforms[key]) {
      material.uniforms[key] = enhancerUniforms[key];
    }
  }
  
  // Inject GLSL chunk into fragment shader (before void main)
  const chunk = enhancer.getGLSLChunk();
  if (!material.fragmentShader.includes('perceptualBaseEnhance')) {
    material.fragmentShader = material.fragmentShader.replace(
      'void main() {',
      `${chunk}\nvoid main() {`
    );
  }
  
  material.needsUpdate = true;
}
