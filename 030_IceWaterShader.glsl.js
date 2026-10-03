// File : 030
// name : shaders/030_IceWaterShader.glsl.js
// description : Anime turquoise ice-water / meltstream shader for the frozen canyon
//               scene. ANALYZED & FIXED: (1) ALLOCATION BUG: perceptual color
//               helpers previously allocated stores/scratch per sync tick. Now uses
//               module-level pre-allocated SoA scratch from 020_gmp_perceptual_color.js,
//               keeping steady-state allocation at zero. (2) PHASE-WRAP POP BUG:
//               all animated phases are integrated in JS at their exact angular
//               frequency and wrapped by 2pi. Shader sine terms use only these
//               continuous phases; caustic double-phase is still 2pi-periodic.
//               (3) FALSE-DEPTH SHADOW BUG: fullscreen parallax water uses a neutral
//               1x1 white shadow texture and identity shadow matrix by default.
//               Live scene shadow-map binding is opt-in via options.sceneShadows.
//               (4) COLOR COPY BUG: Color/Vector3 sources are sanitized and copied
//               explicitly; no Vector3.uniform.copy(Color) path is used.
//               (5) NOISE RANGE BUG: every fbm value driving palette/mask/normal
//               detail is normalized and clamped to 0..1.
//               (6) OVERBRIGHT BUG: foam, caustics, subsurface, specular, sparkle,
//               GI and dither are clamped or bounded; final color is clamped for
//               mobile framebuffers.
//               (7) FILL-RATE BUG: sky pixels and non-water pixels discard before
//               expensive noise work.
//               (8) INSTANCE SHARING BUG: uniforms are cloned per manager instance.
//               (9) CULLING BUG: fullscreen quad uses DoubleSide and frustumCulled=false.
//               (10) SMOOTHSTEP AUDIT: every smoothstep() edge is low->high.
//               (11) GUARDED MATH: aspect, width, band count, rim power, normalize
//               inputs, divisions, phase wrapping, dt, elapsed, wind and camera
//               values are all guarded against NaN/Infinity/degenerate ranges.
//               (12) IMPORT HYGIENE: GLSL_LIGHTING and GLSL_NORMAL_QUANT are not
//               injected with the 016 directional chunk; all local GI uniforms are
//               uniquely prefixed uIW* to prevent duplicate uniform collisions.
//               Renders a winding turquoise meltstream with perspective-compressed
//               flow, cel-quantized wave bands, animated caustics, foam shorelines,
//               surface ice patches, subsurface cyan glow, sky reflection, sun
//               glints, sparkle twinkle, aerial haze and pixel-art de-banding dither.
//               Fully interacts with 016_DirectionalLightShader.glsl.js and the
//               global illumination system created in this chat. Composed on
//               shaders/000_BaseShader.glsl.js with all required imports:
//               GLSL_GLOBALS, GLSL_NOISE, GLSL_BIOME, GLSL_COLOR_UTILS,
//               GLSL_PERCEPTUAL_BASE_ENHANCER, GLSL_DIRECTIONAL_LIGHT,
//               006_normalQuantizer rim-exaggeration snippet + CPU normal bake,
//               020_gmp_perceptual_color SoA perceptual helpers,
//               021_gmp_ies_lighting IES auto-conversion hooks, and
//               001_gmp_MathUtils scalar helpers.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  GLSL_GLOBALS,
  GLSL_NOISE,
  GLSL_BIOME,
} from './000_BaseShader.glsl.js';
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';
import { GLSL_PERCEPTUAL_BASE_ENHANCER } from './017_PerceptualBaseEnhancer.glsl.js';
import { GLSL_DIRECTIONAL_LIGHT } from './016_DirectionalLightShader.glsl.js';
import {
  getAnimeRimNormalExaggerationGLSL,
  quantizeGeometryNormalsCPU,
} from '../mesh/006_normalQuantizer.js';
import {
  PerceptualSoA,
  createScratch,
  rgbStore,
  tintStore,
} from '../utils/020_gmp_perceptual_color.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
} from '../utils/021_gmp_ies_lighting.js';
import { TWO_PI } from '../utils/001_gmp_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. NEUTRAL SHADOW TEXTURE                                           */
/*    Keeps 016 shadow sampling defined without false-depth artifacts  */
/*    on fullscreen parallax geometry unless sceneShadows is enabled.  */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(
  new Uint8Array([255, 255, 255, 255]),
  1,
  1
);
_whiteShadow.minFilter = THREE.NearestFilter;
_whiteShadow.magFilter = THREE.NearestFilter;
_whiteShadow.wrapS = THREE.ClampToEdgeWrapping;
_whiteShadow.wrapT = THREE.ClampToEdgeWrapping;
_whiteShadow.generateMipmaps = false;
_whiteShadow.needsUpdate = true;

/* ------------------------------------------------------------------ */
/* 2. PRE-ALLOCATED PERCEPTUAL SCRATCH (zero steady-state allocation)  */
/* ------------------------------------------------------------------ */
const _pcScratch = createScratch(1);
const _pcBase = rgbStore(1);
const _pcTint = tintStore(1);
const _pcOut = rgbStore(1);

/* ------------------------------------------------------------------ */
/* 3. SAFE SCALAR / COLOR / VECTOR HELPERS                             */
/* ------------------------------------------------------------------ */
function _finite(v, fallback) {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function _clampFinite(v, min, max, fallback) {
  const x = _finite(v, fallback);
  return x < min ? min : x > max ? max : x;
}

function _copyColorFrom(dst, src, fr, fg, fb) {
  let r = fr;
  let g = fg;
  let b = fb;

  if (src) {
    if (src.isColor) {
      r = src.r;
      g = src.g;
      b = src.b;
    } else if (src.isVector3) {
      r = src.x;
      g = src.y;
      b = src.z;
    } else if (typeof src === 'object') {
      if (typeof src.r === 'number') r = src.r;
      if (typeof src.g === 'number') g = src.g;
      if (typeof src.b === 'number') b = src.b;
    }
  }

  dst.setRGB(
    _clampFinite(r, 0.0, 1.0, fr),
    _clampFinite(g, 0.0, 1.0, fg),
    _clampFinite(b, 0.0, 1.0, fb)
  );
}

function _copyVec3From(dst, src, fr, fg, fb) {
  let x = fr;
  let y = fg;
  let z = fb;

  if (src) {
    if (src.isVector3) {
      x = src.x;
      y = src.y;
      z = src.z;
    } else if (src.isColor) {
      x = src.r;
      y = src.g;
      z = src.b;
    } else if (typeof src === 'object') {
      if (typeof src.x === 'number') x = src.x;
      if (typeof src.y === 'number') y = src.y;
      if (typeof src.z === 'number') z = src.z;
    }
  }

  dst.set(
    _clampFinite(x, -1000.0, 1000.0, fr),
    _clampFinite(y, -1000.0, 1000.0, fg),
    _clampFinite(z, -1000.0, 1000.0, fb)
  );
}

function _setRgbStoreFromSource(store, src, fr, fg, fb) {
  let r = fr;
  let g = fg;
  let b = fb;

  if (src) {
    if (src.isVector3) {
      r = src.x;
      g = src.y;
      b = src.z;
    } else if (src.isColor) {
      r = src.r;
      g = src.g;
      b = src.b;
    } else if (typeof src === 'object') {
      if (typeof src.r === 'number') r = src.r;
      if (typeof src.g === 'number') g = src.g;
      if (typeof src.b === 'number') b = src.b;
      if (typeof src.x === 'number' && typeof src.r !== 'number') r = src.x;
      if (typeof src.y === 'number' && typeof src.g !== 'number') g = src.y;
      if (typeof src.z === 'number' && typeof src.b !== 'number') b = src.z;
    }
  }

  store.r[0] = _clampFinite(r, 0.0, 1.0, fr);
  store.g[0] = _clampFinite(g, 0.0, 1.0, fg);
  store.b[0] = _clampFinite(b, 0.0, 1.0, fb);
  if (store.a) store.a[0] = 1.0;
}

function _setTintStoreFromSource(tint, src, factor, fr, fg, fb) {
  let r = fr;
  let g = fg;
  let b = fb;

  if (src) {
    if (src.isColor) {
      r = src.r;
      g = src.g;
      b = src.b;
    } else if (src.isVector3) {
      r = src.x;
      g = src.y;
      b = src.z;
    } else if (typeof src === 'object') {
      if (typeof src.r === 'number') r = src.r;
      if (typeof src.g === 'number') g = src.g;
      if (typeof src.b === 'number') b = src.b;
    }
  }

  tint.r[0] = _clampFinite(r, 0.0, 1.0, fr);
  tint.g[0] = _clampFinite(g, 0.0, 1.0, fg);
  tint.b[0] = _clampFinite(b, 0.0, 1.0, fb);
  tint.factor[0] = _clampFinite(factor, 0.0, 1.0, 1.0);
}

function _applyChromaToVec3(src, scale, dst) {
  _setRgbStoreFromSource(_pcBase, src, 0.5, 0.5, 0.5);
  const s = _clampFinite(scale, 0.0, 4.0, 1.0);
  PerceptualSoA.applyChromatization(_pcBase, _pcOut, 0, s, _pcScratch);
  dst.set(
    _clampFinite(_pcOut.r[0], 0.0, 1.0, 0.5),
    _clampFinite(_pcOut.g[0], 0.0, 1.0, 0.5),
    _clampFinite(_pcOut.b[0], 0.0, 1.0, 0.5)
  );
}

function _applyTintToVec3(base, tint, factor, dst) {
  _setRgbStoreFromSource(_pcBase, base, 0.5, 0.5, 0.5);
  _setTintStoreFromSource(_pcTint, tint, factor, 0.5, 0.5, 0.5);
  PerceptualSoA.applyPerfectTint(_pcBase, _pcTint, _pcOut, 0, _pcScratch);
  dst.set(
    _clampFinite(_pcOut.r[0], 0.0, 1.0, 0.5),
    _clampFinite(_pcOut.g[0], 0.0, 1.0, 0.5),
    _clampFinite(_pcOut.b[0], 0.0, 1.0, 0.5)
  );
}

/* ------------------------------------------------------------------ */
/* 4. ICE WATER DEFAULT UNIFORMS (JS side)                             */
/* ------------------------------------------------------------------ */
export const ICE_WATER_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // Unique local global-illumination providers (no chunk collisions)
  uIWSunDir:          { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uIWSunIntensity:    { value: 1.0 },
  uIWSkyColor:        { value: new THREE.Vector3(0.58, 0.76, 0.94) },
  uIWGroundColor:     { value: new THREE.Vector3(0.07, 0.12, 0.21) },
  uIWGroundStrength:  { value: 0.28 },
  uIWFogColor:        { value: new THREE.Vector3(0.78, 0.88, 0.96) },

  // Pre-integrated animation phases (already multiplied by their frequencies)
  uIWFlowPhase:    { value: 0.0 },
  uIWCausticPhase: { value: 0.0 },
  uIWSparkPhase:   { value: 0.0 },

  // GLSL_BIOME providers
  uBiomeW:    { value: new THREE.Vector4(0.0, 0.0, 0.0, 1.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // 017 perceptual enhancer provider
  uPerceptualEnhance: { value: 0.85 },

  // 016 directional light providers (safe defaults; synced at runtime)
  uDirLightColor:     { value: new THREE.Color(0.96, 0.98, 1.00) },
  uDirLightDirection: { value: new THREE.Vector3(-0.35, -0.78, -0.52) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimColor:          { value: new THREE.Color(0.88, 0.96, 1.00) },
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.16, 0.24, 0.36) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.28 },
  uAmbientColor:      { value: new THREE.Color(0.58, 0.76, 0.94) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: 0.0 },
  uShadowNormalBias:  { value: 0.0 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Ice water specific (exact image palette)
  uColWaterDeep:    { value: new THREE.Vector3(0.055, 0.270, 0.330) }, // 0x0e4554 deep teal
  uColWaterMid:     { value: new THREE.Vector3(0.110, 0.560, 0.560) }, // 0x1c8f8f mid turquoise
  uColWaterShallow: { value: new THREE.Vector3(0.290, 0.820, 0.740) }, // 0x4ad1bd shallow cyan
  uColFoam:         { value: new THREE.Vector3(0.940, 0.985, 1.000) }, // 0xf0fbff white foam
  uColIceGlow:      { value: new THREE.Vector3(0.330, 0.910, 0.790) }, // 0x54e8ca ice glow
  uColSparkle:      { value: new THREE.Vector3(1.000, 1.000, 0.980) }, // 0xfffffa sparkle

  // Channel / surface parameters
  uHorizonY:        { value: 0.50 },
  uAspect:          { value: 0.56 },
  uWidthNear:       { value: 0.18 },
  uWidthFar:        { value: 0.055 },
  uFlowStrength:    { value: 0.85 },
  uCausticStrength: { value: 0.90 },
  uFoamStrength:    { value: 1.00 },
  uSubsurfaceStrength: { value: 0.85 },
  uReflectStrength: { value: 0.55 },
  uIceAmount:       { value: 0.45 },
  uSparkleDensity:  { value: 0.35 },
  uGlowStrength:    { value: 0.75 },
};

/* ------------------------------------------------------------------ */
/* 5. PER-INSTANCE UNIFORM CLONE (setup-only allocation)               */
/* ------------------------------------------------------------------ */
function _cloneUniformValue(v) {
  if (!v) return v;
  if (v.isVector2) return new THREE.Vector2().copy(v);
  if (v.isVector3) return new THREE.Vector3().copy(v);
  if (v.isVector4) return new THREE.Vector4().copy(v);
  if (v.isColor) return new THREE.Color().copy(v);
  if (v.isMatrix4) return new THREE.Matrix4().copy(v);
  return v;
}

function _cloneIceWaterUniforms(src) {
  const out = {};
  for (const key in src) {
    if (Object.prototype.hasOwnProperty.call(src, key)) {
      out[key] = { value: _cloneUniformValue(src[key].value) };
    }
  }
  return out;
}

function _assignUniformOption(uniform, val) {
  if (!uniform) return;
  const target = uniform.value;

  if (target && target.isVector3 && val && val.isColor) {
    target.set(val.r, val.g, val.b);
    return;
  }
  if (target && target.isVector3 && val && val.isVector3) {
    target.copy(val);
    return;
  }
  if (target && target.isColor && val && val.isColor) {
    target.copy(val);
    return;
  }
  if (target && target.isVector2 && val && val.isVector2) {
    target.copy(val);
    return;
  }
  if (target && target.isVector4 && val && val.isVector4) {
    target.copy(val);
    return;
  }
  if (target && target.isMatrix4 && val && val.isMatrix4) {
    target.copy(val);
    return;
  }
  if (typeof val === 'number') {
    uniform.value = val;
    return;
  }
  if (target && typeof target.set === 'function' && val && typeof val === 'object') {
    target.set(val);
    return;
  }
  uniform.value = val;
}

/* ------------------------------------------------------------------ */
/* 6. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const ICE_WATER_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 7. FRAGMENT SHADER (turquoise ice water + caustics + foam + GI)     */
/*    Injection order:                                                 */
/*    GLOBALS -> NOISE -> COLOR_UTILS -> BIOME -> 017 -> 016           */
/*    GLSL_NORMAL_QUANT and GLSL_LIGHTING are intentionally not        */
/*    injected here to avoid duplicate uniform declarations.           */
/* ------------------------------------------------------------------ */
export const ICE_WATER_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

// Unique local global-illumination uniforms
uniform vec3  uIWSunDir;
uniform float uIWSunIntensity;
uniform vec3  uIWSkyColor;
uniform vec3  uIWGroundColor;
uniform float uIWGroundStrength;
uniform vec3  uIWFogColor;

// Pre-integrated animation phases
uniform float uIWFlowPhase;
uniform float uIWCausticPhase;
uniform float uIWSparkPhase;

// Ice water palette
uniform vec3  uColWaterDeep;
uniform vec3  uColWaterMid;
uniform vec3  uColWaterShallow;
uniform vec3  uColFoam;
uniform vec3  uColIceGlow;
uniform vec3  uColSparkle;

// Channel / surface parameters
uniform float uHorizonY;
uniform float uAspect;
uniform float uWidthNear;
uniform float uWidthFar;
uniform float uFlowStrength;
uniform float uCausticStrength;
uniform float uFoamStrength;
uniform float uSubsurfaceStrength;
uniform float uReflectStrength;
uniform float uIceAmount;
uniform float uSparkleDensity;
uniform float uGlowStrength;

varying vec2 vUv;

// Local pixel quantization (avoids GLSL_NORMAL_QUANT collisions)
vec2 iwPxq(vec2 uv, float q) {
  float safeQ = max(q, 1.0);
  return floor(uv * safeQ + 0.5) / safeQ;
}

// Local hash (avoids depending on a specific h21 symbol name)
float iwHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + vec2(45.32));
  return fract(p.x * p.y);
}

void main() {
  float aspect = max(uAspect, 0.2);
  float ax = (vUv.x - 0.5) * aspect;

  // Water lives below the horizon line (low->high smoothstep)
  float belowMask = 1.0 - smoothstep(uHorizonY - 0.004, uHorizonY + 0.004, vUv.y);
  if (belowMask <= 0.001) discard;

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.58, uHorizonY, vUv.y);

  // Perspective compression toward the horizon
  float persp = mix(1.0, 5.5, hp * hp);
  vec2 noiseUv = vec2(vUv.x * aspect * 12.0, vUv.y * 38.0) * persp;

  // Continuous, pre-integrated 2pi phases (no multiply-after-wrap pop)
  float flowPh    = mod(uIWFlowPhase, 6.2831853);
  float causticPh = mod(uIWCausticPhase, 6.2831853);
  float sparkPh   = mod(uIWSparkPhase, 6.2831853);

  // Winding channel center (static spatial bend; no phase-fed fbm)
  float bendRaw = fbm(vec2(vUv.y * 1.8, 3.7), 2);
  float bend = (clamp(bendRaw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 0.14;
  float center = bend;

  // Channel width: wider near camera, narrower at horizon
  float widthNoiseRaw = fbm(vec2(vUv.y * 2.7, 9.1), 2);
  float widthNoise = (clamp(widthNoiseRaw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 0.035;
  float width = mix(max(uWidthNear, 0.02), max(uWidthFar, 0.02), hp) + widthNoise;
  width = clamp(width, 0.02, 0.45);

  float edge = abs(ax - center) / width;
  float waterMask = 1.0 - smoothstep(0.88, 1.0, edge);
  if (waterMask < 0.02) discard;

  // Analytic wave field (phase added only to sine arguments)
  float flowStr = clamp(uFlowStrength, 0.0, 2.0);
  float w1Arg = noiseUv.y * 3.0 + noiseUv.x * 0.7 + flowPh;
  float w2Arg = noiseUv.x * 2.3 - noiseUv.y * 1.7 - flowPh + 1.7;

  float w1 = sin(w1Arg) * 0.5 + 0.5;
  float w2 = sin(w2Arg) * 0.5 + 0.5;

  float detailRaw = fbm(noiseUv * 2.2 + vec2(13.7), 2);
  float detail = clamp(detailRaw * 0.5 + 0.5, 0.0, 1.0);

  float height = clamp(w1 * 0.34 + w2 * 0.24 + detail * 0.42, 0.0, 1.0);

  // Approximate normal from analytic wave derivatives + detail slope
  float dw1dy = cos(w1Arg) * 1.5;
  float dw1dx = cos(w1Arg) * 0.35;
  float dw2dx = cos(w2Arg) * 1.15;
  float dw2dy = cos(w2Arg) * (-0.85);

  float gradX = (dw1dx + dw2dx) * 0.35 + (detail - 0.5) * 1.10;
  float gradY = (dw1dy + dw2dy) * 0.35 + (detail - 0.5) * 0.55;

  vec3 nrm = normalize(vec3(
    -gradX * flowStr,
    -gradY * flowStr,
    1.0
  ));

  // Cel-quantized water tone
  float bands = max(uCelSteps, 1.0);
  float tone = floor(height * bands + 0.5) / bands;

  vec3 waterCol = paletteMix4(
    uColWaterDeep,
    uColWaterMid,
    uColWaterShallow,
    uColIceGlow,
    tone
  );

  // Shallow cyan near banks, deeper teal in center
  waterCol = mix(waterCol, uColWaterShallow, smoothstep(0.35, 0.92, edge) * 0.55);
  waterCol = mix(waterCol, uColWaterDeep, smoothstep(0.00, 0.35, edge) * 0.25);

  // Surface ice patches near banks
  float iceAmountSafe = clamp(uIceAmount, 0.0, 1.0);
  float iceRaw = fbm(noiseUv * 1.4 + vec2(29.3), 2);
  float iceNoise = clamp(iceRaw * 0.5 + 0.5, 0.0, 1.0);
  float iceMask = smoothstep(0.62, 0.84, iceNoise)
                * (0.25 + 0.75 * smoothstep(0.50, 0.95, edge))
                * iceAmountSafe;
  iceMask = max(iceMask, smoothstep(0.72, 0.88, edge) * 0.18 * iceAmountSafe);
  iceMask = clamp(iceMask, 0.0, 1.0);

  vec3 iceCol = mix(uColIceGlow, uColFoam, smoothstep(0.45, 0.90, edge));
  waterCol = mix(waterCol, iceCol, iceMask * 0.65);
  nrm = mix(nrm, vec3(0.0, 0.0, 1.0), iceMask * 0.55);

  // Foam shoreline + hash bubbles
  float foamBand = smoothstep(0.70, 0.90, edge)
                 * (1.0 - smoothstep(0.95, 1.06, edge));

  float foamNoiseRaw = fbm(noiseUv * 5.0 + vec2(7.1), 2);
  float foamNoise = clamp(foamNoiseRaw * 0.5 + 0.5, 0.0, 1.0);

  vec2 bp = noiseUv * 16.0;
  vec2 bcell = floor(bp);
  vec2 bf = fract(bp) - 0.5;
  float bh = iwHash21(bcell);
  vec2 bj = vec2(iwHash21(bcell + vec2(3.1)), iwHash21(bcell + vec2(5.7))) - 0.5;
  float bdist = length(bf - bj * 0.60);
  float bubble = (1.0 - smoothstep(0.0, 0.18, bdist)) * step(0.62, bh);

  float foamStrength = clamp(uFoamStrength, 0.0, 1.5);
  float foam = clamp(
    foamBand * (0.45 + 0.55 * foamNoise) +
    bubble * foamBand * 0.65,
    0.0,
    1.0
  ) * foamStrength;
  foam = clamp(foam, 0.0, 1.0);

  waterCol = mix(waterCol, uColFoam, foam * 0.85);

  // Animated caustics (integer phase multipliers only)
  float causticStrength = clamp(uCausticStrength, 0.0, 1.5);
  float glowStr = clamp(uGlowStrength, 0.0, 1.5);

  float c1 = sin(noiseUv.x * 7.0 + causticPh) * sin(noiseUv.y * 9.0 - causticPh);
  float c2 = sin((noiseUv.x + noiseUv.y) * 6.0 + causticPh * 2.0);
  float causticRaw = clamp(0.5 + 0.5 * (c1 * 0.6 + c2 * 0.4), 0.0, 1.0);
  float caustic = pow(causticRaw, 2.0);

  caustic *= (1.0 - smoothstep(0.55, 1.0, edge));
  caustic *= (1.0 - foam * 0.65);
  caustic *= (1.0 - iceMask * 0.85);
  caustic *= causticStrength;
  caustic = clamp(caustic, 0.0, 1.0);

  waterCol += uColIceGlow * caustic * 0.35 * glowStr;

  // Subsurface cyan glow driven by live sun direction
  float subsurfaceStrength = clamp(uSubsurfaceStrength, 0.0, 1.5);
  vec3 sunVec = normalize(uIWSunDir + vec3(0.0001));
  float sunDot = max(dot(nrm, sunVec), 0.0);

  float sss = pow(sunDot, 1.4);
  sss *= (1.0 - smoothstep(0.45, 1.0, edge));
  sss *= (1.0 - iceMask * 0.55);
  sss *= subsurfaceStrength;
  sss = clamp(sss, 0.0, 1.0);

  waterCol += uColIceGlow * sss * 0.40 * glowStr;
  waterCol += uDirLightColor * sss * 0.18;
  waterCol = clamp(waterCol, 0.0, 1.0);

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
  float dirIntensitySafe = clamp(uDirLightIntensity, 0.0, 1.5);
  vec3 baseCol = perceptualBaseEnhance(waterCol, dirIntensitySafe);
  baseCol = clamp(baseCol, 0.0, 1.0);

  // 006 rim-normal exaggeration on dedicated locals (no GI clobber)
  vec3 nrmF = nrm;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  float rimPowSafe = max(uRimPower, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'rimPowSafe')}

  // 016 anime directional cel light. Scene shadow map is neutral unless opted in.
  vec3 worldPos = vec3(ax * 70.0, (1.0 - hp) * 45.0, -0.5);
  vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

  // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + cyan water/ice bounce
  float hemi = nrmF.y * 0.5 + 0.5;
  vec3 gi = mix(uIWGroundColor, uIWSkyColor, hemi) * clamp(uIWGroundStrength, 0.0, 1.0);
  gi += uColIceGlow * (1.0 - hemi) * 0.18 * subsurfaceStrength * glowStr;
  gi = clamp(gi, 0.0, 1.0);
  lit += baseCol * gi;

  // Sky reflection / Fresnel
  float reflectStrength = clamp(uReflectStrength, 0.0, 1.0);
  float fres = pow(1.0 - max(dot(nrmF, viewF), 0.0), 3.0);
  lit = mix(lit, uIWSkyColor, fres * reflectStrength * 0.35);

  // Sun glint
  float sunIntensitySafe = clamp(uIWSunIntensity, 0.0, 1.5);
  float spec = pow(sunDot, 70.0);
  spec *= (1.0 - foam * 0.70);
  spec *= (1.0 - iceMask * 0.35);
  spec = clamp(spec, 0.0, 1.0);
  lit += uDirLightColor * spec * 0.45 * sunIntensitySafe;

  // Sparkle twinkle (integer phase multiplier only)
  float sparkleDensity = clamp(uSparkleDensity, 0.0, 1.0);
  vec2 sp = noiseUv * 55.0;
  vec2 scell = floor(sp);
  float sh = iwHash21(scell);
  vec2 sf = fract(sp) - 0.5;
  vec2 sj = vec2(iwHash21(scell + vec2(2.3)), iwHash21(scell + vec2(4.9))) - 0.5;
  float sdist = length(sf - sj * 0.55);

  float spark = (1.0 - smoothstep(0.0, 0.10, sdist)) * step(1.0 - sparkleDensity, sh);
  float tw = 0.35 + 0.65 * (0.5 + 0.5 * sin(sparkPh + sh * 43.0));

  spark *= tw;
  spark *= sunIntensitySafe;
  spark *= (1.0 - hp * 0.55);
  spark *= (1.0 - foam * 0.25);
  spark = clamp(spark, 0.0, 1.0);

  lit += uColSparkle * spark * 0.45;

  // Distance haze toward horizon (aerial perspective)
  lit = applyFogBlend(lit, uIWFogColor, hp * hp * 0.30);

  // Pixel-art de-banding dither + final mobile-safe clamp
  vec2 pq = iwPxq(vUv * 112.0, 56.0);
  lit += (iwHash21(pq) - 0.5) * 0.010;
  lit = clamp(lit, 0.0, 1.0);

  float alpha = clamp(
    waterMask * mix(0.88, 1.0, max(foam, iceMask)),
    0.0,
    1.0
  );

  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 8. ICE WATER SHADER MANAGER                                         */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();

export class IceWaterShader {
  constructor(options = {}) {
    this.uniforms = _cloneIceWaterUniforms(ICE_WATER_UNIFORMS);
    this.sceneShadows = options.sceneShadows === true;

    for (const k in options) {
      if (
        Object.prototype.hasOwnProperty.call(options, k) &&
        this.uniforms[k] &&
        options[k] !== undefined
      ) {
        _assignUniformOption(this.uniforms[k], options[k]);
      }
    }

    this.geometry = new THREE.PlaneGeometry(2, 2);
    quantizeGeometryNormalsCPU(this.geometry, 4); // 006 CPU bake (harmless on quad)

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: ICE_WATER_VERTEX,
      fragmentShader: ICE_WATER_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 580; // above snow ground, below floating ice/rocks/foam overlays

    this._syncAccum = 1.0;
    this._chroma = _clampFinite(options.chroma, 1.0, 2.0, 1.10);
    this._iesScale = 1.0;

    // Angular speeds (rad/s). Phases are integrated and wrapped by 2pi.
    this._flowSpeed = _clampFinite(options.flowSpeed, 0.0, 5.0, 0.85);
    this._causticSpeed = _clampFinite(options.causticSpeed, 0.0, 5.0, 0.55);
    this._sparkSpeed = _clampFinite(options.sparkSpeed, 0.0, 8.0, 2.10);

    this._flowPhase = 0.0;
    this._causticPhase = 0.0;
    this._sparkPhase = 0.0;
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uAspect.value = _clampFinite(a, 0.2, 2.5, 0.56);
  }

  setHorizon(y) {
    this.uniforms.uHorizonY.value = _clampFinite(y, 0.1, 0.9, 0.50);
  }

  setChannelWidth(nearV, farV) {
    this.uniforms.uWidthNear.value = _clampFinite(nearV, 0.02, 0.45, 0.18);
    this.uniforms.uWidthFar.value = _clampFinite(farV, 0.02, 0.45, 0.055);
  }

  setFlowStrength(v) {
    this.uniforms.uFlowStrength.value = _clampFinite(v, 0.0, 2.0, 0.85);
  }

  setCausticStrength(v) {
    this.uniforms.uCausticStrength.value = _clampFinite(v, 0.0, 1.5, 0.90);
  }

  setFoamStrength(v) {
    this.uniforms.uFoamStrength.value = _clampFinite(v, 0.0, 1.5, 1.00);
  }

  setSubsurfaceStrength(v) {
    this.uniforms.uSubsurfaceStrength.value = _clampFinite(v, 0.0, 1.5, 0.85);
  }

  setReflectStrength(v) {
    this.uniforms.uReflectStrength.value = _clampFinite(v, 0.0, 1.0, 0.55);
  }

  setIceAmount(v) {
    this.uniforms.uIceAmount.value = _clampFinite(v, 0.0, 1.0, 0.45);
  }

  setSparkleDensity(v) {
    this.uniforms.uSparkleDensity.value = _clampFinite(v, 0.0, 1.0, 0.35);
  }

  setGlowStrength(v) {
    this.uniforms.uGlowStrength.value = _clampFinite(v, 0.0, 1.5, 0.75);
  }

  /* Optional IES hook (021_gmp_ies_lighting): tag the directional light
     with an isotropic IES profile; ballast-scaled intensity feeds uDirLightIntensity. */
  enableIES(light) {
    if (!light) return;
    try {
      autoConvertLight(light, generateIsotropicIES(light));
      const ies = light.userData && light.userData.ies;
      if (ies) {
        const flux = integrateCandela(ies);
        this._iesScale = _clampFinite(
          (ies.ballastFactor || 1.0) *
          (flux > 0.0 ? flux / (4.0 * Math.PI) : 1.0),
          0.25,
          4.0,
          1.0
        );
      }
    } catch (_) {
      this._iesScale = 1.0;
    }
  }

  /* Full interaction with 016_DirectionalLightShader: direction, color,
     intensity, rim, ambient, shadow tint + optional real shadow map.       */
  syncFromLight(lightShader) {
    if (!lightShader || typeof lightShader.getUniforms !== 'function') return;

    const lu = lightShader.getUniforms();
    const u = this.uniforms;

    // Direction (travel direction from sun into scene)
    _copyVec3From(
      u.uDirLightDirection.value,
      lu.uDirLightDirection && lu.uDirLightDirection.value,
      -0.35,
      -0.78,
      -0.52
    );
    if (u.uDirLightDirection.value.lengthSq() < 1e-12) {
      u.uDirLightDirection.value.set(-0.35, -0.78, -0.52);
    }
    u.uDirLightDirection.value.normalize();
    u.uIWSunDir.value.copy(u.uDirLightDirection.value).negate();

    // Color
    _copyColorFrom(
      u.uDirLightColor.value,
      lu.uDirLightColor && lu.uDirLightColor.value,
      0.96,
      0.98,
      1.00
    );

    // Intensity
    const rawIntensity = lu.uDirLightIntensity ? lu.uDirLightIntensity.value : 1.0;
    const intensity = _clampFinite(rawIntensity, 0.0, 4.0, 1.0) *
      _clampFinite(this._iesScale, 0.25, 4.0, 1.0);
    u.uDirLightIntensity.value = intensity;
    u.uIWSunIntensity.value = intensity;

    // Shadow tint -> ground bounce color
    _copyColorFrom(
      u.uShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.16,
      0.24,
      0.36
    );
    u.uIWGroundColor.value.set(
      _clampFinite(u.uShadowTint.value.r * 0.45, 0.0, 1.0, 0.07),
      _clampFinite(u.uShadowTint.value.g * 0.50, 0.0, 1.0, 0.12),
      _clampFinite(u.uShadowTint.value.b * 0.58, 0.0, 1.0, 0.21)
    );

    const rawAmbientIntensity = lu.uAmbientIntensity ? lu.uAmbientIntensity.value : 0.28;
    u.uIWGroundStrength.value = _clampFinite(rawAmbientIntensity * 0.75, 0.0, 1.0, 0.28);

    // Rim color
    _copyColorFrom(
      u.uRimColor.value,
      lu.uRimColor && lu.uRimColor.value,
      0.88,
      0.96,
      1.00
    );

    // Ambient color -> sky / fog
    _copyColorFrom(
      u.uAmbientColor.value,
      lu.uAmbientColor && lu.uAmbientColor.value,
      0.58,
      0.76,
      0.94
    );
    u.uIWSkyColor.value.set(
      _clampFinite(u.uAmbientColor.value.r, 0.0, 1.0, 0.58),
      _clampFinite(u.uAmbientColor.value.g, 0.0, 1.0, 0.76),
      _clampFinite(u.uAmbientColor.value.b, 0.0, 1.0, 0.94)
    );
    u.uIWFogColor.value.set(
      _clampFinite(u.uAmbientColor.value.r * 0.88 + 0.12, 0.0, 1.0, 0.78),
      _clampFinite(u.uAmbientColor.value.g * 0.88 + 0.12, 0.0, 1.0, 0.88),
      _clampFinite(u.uAmbientColor.value.b * 0.88 + 0.12, 0.0, 1.0, 0.96)
    );

    // Cel / rim / shadow scalars
    u.uCelSteps.value = _clampFinite(lu.uCelSteps ? lu.uCelSteps.value : 4.0, 1.0, 8.0, 4.0);
    u.uRimPower.value = _clampFinite(lu.uRimPower ? lu.uRimPower.value : 3.0, 1.0, 8.0, 3.0);
    u.uRimIntensity.value = _clampFinite(lu.uRimIntensity ? lu.uRimIntensity.value : 0.45, 0.0, 1.0, 0.45);
    u.uShadowSoftness.value = _clampFinite(lu.uShadowSoftness ? lu.uShadowSoftness.value : 0.05, 0.0, 0.5, 0.05);
    u.uAmbientIntensity.value = _clampFinite(rawAmbientIntensity, 0.0, 1.0, 0.28);
    u.uPerceptualEnhance.value = _clampFinite(u.uPerceptualEnhance.value, 0.0, 1.0, 0.85);

    // Shadow map: neutral by default, opt-in for real 3D proxies only
    if (this.sceneShadows) {
      const light = typeof lightShader.getLight === 'function' ? lightShader.getLight() : null;
      if (light && light.shadow && light.shadow.map && light.shadow.map.texture) {
        u.uShadowMap.value = light.shadow.map.texture;
        u.uShadowMatrix.value.copy(light.shadow.matrix);
        u.uShadowMapSize.value.set(
          _clampFinite(light.shadow.mapSize.x, 1.0, 8192.0, 1024.0),
          _clampFinite(light.shadow.mapSize.y, 1.0, 8192.0, 1024.0)
        );
        u.uShadowBias.value = _clampFinite(light.shadow.bias, -0.05, 0.05, -0.001);
        u.uShadowNormalBias.value = _clampFinite(light.shadow.normalBias, 0.0, 0.5, 0.05);
      } else {
        u.uShadowMap.value = _whiteShadow;
        u.uShadowMatrix.value.identity();
        u.uShadowMapSize.value.set(1, 1);
        u.uShadowBias.value = 0.0;
        u.uShadowNormalBias.value = 0.0;
      }
    } else {
      u.uShadowMap.value = _whiteShadow;
      u.uShadowMatrix.value.identity();
      u.uShadowMapSize.value.set(1, 1);
      u.uShadowBias.value = 0.0;
      u.uShadowNormalBias.value = 0.0;
    }

    // Perceptual vibrancy / tint updates (zero allocation in steady state)
    _applyChromaToVec3(
      u.uColWaterShallow.value,
      this._chroma,
      u.uColWaterShallow.value
    );

    _applyTintToVec3(
      u.uColIceGlow.value,
      u.uDirLightColor.value,
      0.24,
      u.uColIceGlow.value
    );

    _applyTintToVec3(
      u.uColWaterDeep.value,
      u.uShadowTint.value,
      0.28,
      u.uColWaterDeep.value
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;

    dt = _clampFinite(dt, 0.0, 0.1, 0.0);
    elapsed = _finite(elapsed, 0.0);
    u.uTime.value = elapsed;

    const windVec = u.uWind && u.uWind.value;
    const windX = windVec && Number.isFinite(windVec.x) ? windVec.x : 0.0;

    // Integrated, 2pi-wrapped phases at exact angular frequencies.
    this._flowPhase = (this._flowPhase + dt * (this._flowSpeed + Math.abs(windX) * 0.12)) % TWO_PI;
    if (this._flowPhase < 0.0) this._flowPhase += TWO_PI;

    this._causticPhase = (this._causticPhase + dt * this._causticSpeed) % TWO_PI;
    if (this._causticPhase < 0.0) this._causticPhase += TWO_PI;

    this._sparkPhase = (this._sparkPhase + dt * this._sparkSpeed) % TWO_PI;
    if (this._sparkPhase < 0.0) this._sparkPhase += TWO_PI;

    u.uIWFlowPhase.value = _finite(this._flowPhase, 0.0);
    u.uIWCausticPhase.value = _finite(this._causticPhase, 0.0);
    u.uIWSparkPhase.value = _finite(this._sparkPhase, 0.0);

    if (camera && camera.position) {
      u.uCamPos.value.set(
        _finite(camera.position.x, 0.0),
        _finite(camera.position.y, 0.0),
        _finite(camera.position.z, 0.0)
      );
    }

    if (lightShader) {
      this._syncAccum += dt;
      if (this._syncAccum >= 0.25) { // throttled light+GI+perceptual sync (4 Hz)
        this._syncAccum = 0.0;
        this.syncFromLight(lightShader);
      }
    }
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                          */
/* ------------------------------------------------------------------ */
export function createIceWaterShader(options = {}) {
  return new IceWaterShader(options);
}

export default {
  IceWaterShader,
  createIceWaterShader,
  ICE_WATER_UNIFORMS,
  ICE_WATER_VERTEX,
  ICE_WATER_FRAGMENT,
};
