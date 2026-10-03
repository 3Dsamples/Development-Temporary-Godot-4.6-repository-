// File : 032
// name : shaders/032_SeaRocksShader.glsl.js
// description : Anime coastal sea-rock shader for the turquoise ocean scene. Renders
//               dark faceted rock bases, sunlit tan tops, wet black-green bands,
//               foam rings, splash flecks, projected water shadows, cyan sea bounce
//               GI, sun glitter and aerial haze. Fully interacts with
//               016_DirectionalLightShader.glsl.js and the global illumination
//               system created in this chat. Composed on shaders/000_BaseShader.glsl.js
//               with all required imports. Local uniforms are prefixed uSR* to avoid
//               duplicate collisions with the 016 directional chunk. All phases are
//               integrated and wrapped by 2pi, all smoothstep edges are low->high,
//               all normalize/division inputs are guarded, all noise palette drivers
//               are normalized to 0..1, final color is clamped, fullscreen quads are
//               DoubleSide, uniforms are cloned per instance, and perceptual sync is
//               zero-allocation in steady state.
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
import { clamp, damp, TWO_PI } from '../utils/001_gmp_MathUtils.js';

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

const _pcScratch = createScratch(1);
const _pcBase = rgbStore(1);
const _pcTint = tintStore(1);
const _pcOut = rgbStore(1);

function _finite(v, fallback) {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function _clampFinite(v, min, max, fallback) {
  const x = _finite(v, fallback);
  return clamp(x, min, max);
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

export const SEA_ROCKS_UNIFORMS = {
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  uSRSunDir:          { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uSRSunIntensity:    { value: 1.0 },
  uSRSkyColor:        { value: new THREE.Vector3(0.58, 0.76, 0.94) },
  uSRGroundColor:     { value: new THREE.Vector3(0.04, 0.16, 0.22) },
  uSRGroundStrength:  { value: 0.30 },
  uSRFogColor:        { value: new THREE.Vector3(0.72, 0.88, 0.92) },
  uSRShadowTint:      { value: new THREE.Vector3(0.05, 0.18, 0.25) },

  uSRFoamPhase:  { value: 0.0 },
  uSRWindPhase:  { value: 0.0 },
  uSRSparkPhase: { value: 0.0 },

  uBiomeW:    { value: new THREE.Vector4(0.0, 0.0, 1.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  uPerceptualEnhance: { value: 0.85 },

  uDirLightColor:     { value: new THREE.Color(1.00, 0.98, 0.92) },
  uDirLightDirection: { value: new THREE.Vector3(-0.35, -0.78, -0.52) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimColor:          { value: new THREE.Color(0.92, 0.98, 1.00) },
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.05, 0.18, 0.25) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.30 },
  uAmbientColor:      { value: new THREE.Color(0.58, 0.76, 0.94) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: 0.0 },
  uShadowNormalBias:  { value: 0.0 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  uSRColRockTan:   { value: new THREE.Vector3(0.780, 0.630, 0.470) },
  uSRColRockMid:   { value: new THREE.Vector3(0.430, 0.410, 0.390) },
  uSRColRockDark:  { value: new THREE.Vector3(0.180, 0.230, 0.285) },
  uSRColRockDeep:  { value: new THREE.Vector3(0.075, 0.115, 0.150) },
  uSRColWet:       { value: new THREE.Vector3(0.085, 0.145, 0.170) },
  uSRColFoam:      { value: new THREE.Vector3(0.960, 0.995, 1.000) },
  uSRColSeaShallow:{ value: new THREE.Vector3(0.250, 0.820, 0.760) },
  uSRColSeaMid:    { value: new THREE.Vector3(0.075, 0.520, 0.560) },
  uSRColSparkle:   { value: new THREE.Vector3(1.000, 1.000, 0.985) },
  uSRGIColor:      { value: new THREE.Vector3(0.180, 0.760, 0.700) },

  uSRHorizonY:       { value: 0.48 },
  uSRAspect:         { value: 0.56 },
  uSRRockDensity:    { value: 0.58 },
  uSRFoamStrength:   { value: 0.95 },
  uSRWetStrength:    { value: 0.82 },
  uSRShadowLen:      { value: 0.72 },
  uSRShadowStrength: { value: 0.48 },
  uSRSparkleDensity: { value: 0.30 },
  uSRGIStrength:     { value: 0.55 },
};

function _cloneUniformValue(v) {
  if (!v) return v;
  if (v.isVector2) return new THREE.Vector2().copy(v);
  if (v.isVector3) return new THREE.Vector3().copy(v);
  if (v.isVector4) return new THREE.Vector4().copy(v);
  if (v.isColor) return new THREE.Color().copy(v);
  if (v.isMatrix4) return new THREE.Matrix4().copy(v);
  return v;
}

function _cloneSeaRocksUniforms(src) {
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

export const SEA_ROCKS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

export const SEA_ROCKS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uSRSunDir;
uniform float uSRSunIntensity;
uniform vec3  uSRSkyColor;
uniform vec3  uSRGroundColor;
uniform float uSRGroundStrength;
uniform vec3  uSRFogColor;
uniform vec3  uSRShadowTint;

uniform float uSRFoamPhase;
uniform float uSRWindPhase;
uniform float uSRSparkPhase;

uniform vec3  uSRColRockTan;
uniform vec3  uSRColRockMid;
uniform vec3  uSRColRockDark;
uniform vec3  uSRColRockDeep;
uniform vec3  uSRColWet;
uniform vec3  uSRColFoam;
uniform vec3  uSRColSeaShallow;
uniform vec3  uSRColSeaMid;
uniform vec3  uSRColSparkle;
uniform vec3  uSRGIColor;

uniform float uSRHorizonY;
uniform float uSRAspect;
uniform float uSRRockDensity;
uniform float uSRFoamStrength;
uniform float uSRWetStrength;
uniform float uSRShadowLen;
uniform float uSRShadowStrength;
uniform float uSRSparkleDensity;
uniform float uSRGIStrength;

varying vec2 vUv;

vec2 srPxq(vec2 uv, float q) {
  float safeQ = max(q, 1.0);
  return floor(uv * safeQ + 0.5) / safeQ;
}

float srHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + vec2(45.32));
  return fract(p.x * p.y);
}

void main() {
  float aspect = max(uSRAspect, 0.2);
  float ax = (vUv.x - 0.5) * aspect;

  float seaMask = 1.0 - smoothstep(uSRHorizonY - 0.004, uSRHorizonY + 0.004, vUv.y);
  if (seaMask <= 0.001) discard;

  float hp = smoothstep(uSRHorizonY - 0.58, uSRHorizonY, vUv.y);
  float persp = mix(1.0, 6.5, hp * hp);
  vec2 sp = vec2(vUv.x * aspect * 22.0, vUv.y * 64.0) * persp;
  vec2 cell = floor(sp);
  vec2 f = fract(sp) - 0.5;

  float h = srHash21(cell);
  float h2 = srHash21(cell + vec2(7.7));

  float density = clamp(uSRRockDensity, 0.0, 1.0);
  float gate = step(1.0 - density * (1.0 - hp * 0.50), h);

  vec2 sdirRaw = uDirLightDirection.xy;
  float slen = max(length(sdirRaw), 1e-4);
  vec2 sdir = sdirRaw / slen;
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0);
  float shLen = max(uSRShadowLen, 0.05);

  vec2 jitter = vec2(srHash21(cell + vec2(3.1)), srHash21(cell + vec2(5.7))) - 0.5;
  vec2 q = f - jitter * 0.72;
  float radius = max(0.11 + h2 * 0.19, 0.06);
  float dist = length(q);
  float dirLen = max(dist, 1e-4);
  vec2 dir = q / dirLen;

  float wob = 1.0
    + sin(dir.x * 5.0 + dir.y * 3.0 + h * 31.0) * 0.14
    + sin(dir.x * 9.0 - dir.y * 7.0 + h * 19.0) * 0.08;
  float rw = radius * max(wob, 0.5);

  float rockMask = (1.0 - smoothstep(rw - 0.03, rw + 0.03, dist)) * gate;
  rockMask *= 1.0 - smoothstep(0.08, 0.22, -q.y - radius * 0.65);
  rockMask = clamp(rockMask, 0.0, 1.0);

  float foamPh = mod(uSRFoamPhase, 6.2831853);
  float windPh = mod(uSRWindPhase, 6.2831853);
  float sparkPh = mod(uSRSparkPhase, 6.2831853);

  float ringInner = radius * wob * 1.02;
  float ringOuter = radius * wob * (1.28 + 0.12 * sin(foamPh + h * 24.0));
  float foamNoiseRaw = fbm(vec2(dist * 8.0 + h * 13.0, dir.x * 4.0 + dir.y * 3.0) + vec2(7.1), 2);
  float foamNoise = clamp(foamNoiseRaw * 0.5 + 0.5, 0.0, 1.0);
  float foamBreath = 0.72 + 0.28 * sin(foamPh * 2.0 + h * 17.0);

  float foamRing =
    smoothstep(ringInner - 0.025, ringInner + 0.025, dist) *
    (1.0 - smoothstep(ringOuter - 0.035, ringOuter + 0.035, dist));
  foamRing *= gate * (0.55 + 0.45 * foamNoise) * foamBreath;
  float foamMask = clamp(foamRing * clamp(uSRFoamStrength, 0.0, 1.5), 0.0, 1.0);

  vec2 fp = sp * 1.8 + q * 3.0;
  vec2 fcell = floor(fp);
  float fh = srHash21(fcell);
  vec2 ff = fract(fp) - 0.5;
  vec2 fj = vec2(srHash21(fcell + vec2(2.3)), srHash21(fcell + vec2(4.9))) - 0.5;
  float fdist = length(ff - fj * 0.55);

  float splashRing =
    smoothstep(radius * 1.25, radius * 1.45, dist) *
    (1.0 - smoothstep(radius * 1.75, radius * 2.05, dist));
  float splash =
    (1.0 - smoothstep(0.0, 0.12, fdist)) *
    step(0.78, fh) *
    splashRing *
    gate;
  splash *= 0.55 + 0.45 * sin(sparkPh + fh * 31.0);
  splash *= clamp(uSRFoamStrength, 0.0, 1.5);
  splash = clamp(splash, 0.0, 1.0);

  float foamAll = clamp(max(foamMask, splash * 0.55), 0.0, 1.0);

  vec2 spScale = vec2(aspect * 22.0, 64.0) * persp;
  vec2 sdirCellRaw = sdir / max(spScale, vec2(1e-4));
  float scLen = max(length(sdirCellRaw), 1e-6);
  vec2 sdirCell = sdirCellRaw / scLen;

  float len = radius * (1.0 + shLen * 1.9);
  float alongC = clamp(dot(q, sdirCell), 0.0, len);
  float dSeg = length(q - sdirCell * alongC);

  float shadowMask = (1.0 - smoothstep(radius * 0.82, radius * 1.18, dSeg)) * gate * sunAbove;
  shadowMask *= 1.0 - smoothstep(len * 0.62, len, alongC);
  shadowMask *= 0.84 + 0.16 * srHash21(cell + vec2(11.3));
  shadowMask = clamp(shadowMask * (1.0 - rockMask) * (1.0 - foamAll * 0.35), 0.0, 1.0);

  if (rockMask < 0.02 && foamAll < 0.02 && shadowMask < 0.02) discard;

  if (rockMask >= foamAll && rockMask >= shadowMask) {
    vec3 nrm = normalize(vec3(q.x * 2.1, q.y * 2.1, 1.0));
    vec3 fn = floor(nrm * 3.0 + 0.5) / 3.0;
    nrm = normalize(mix(nrm, fn, 0.48));

    float qyn = q.y / max(radius, 1e-3);
    float topFac = smoothstep(-0.15, 0.55, qyn);

    float cvar = clamp(srHash21(cell + vec2(13.7)), 0.0, 1.0);
    vec3 rockCol = paletteMix4(
      uSRColRockDeep,
      uSRColRockDark,
      uSRColRockMid,
      uSRColRockTan,
      cvar
    );
    rockCol = mix(rockCol, uSRColRockTan, topFac * 0.62);
    rockCol = mix(rockCol, uSRColRockDeep, (1.0 - topFac) * 0.25);

    float strRaw = fbm(vec2(q.x * 8.0 + q.y * 5.0, h * 11.0) + vec2(4.4), 2);
    float strNoise = clamp(strRaw * 0.5 + 0.5, 0.0, 1.0);
    rockCol *= 0.90 + strNoise * 0.14;

    float crackRaw = fbm(vec2(q.x * 5.0 + q.y * 7.0, h * 17.0) + vec2(9.1), 2);
    float crackNoise = clamp(crackRaw * 0.5 + 0.5, 0.0, 1.0);
    float crack = smoothstep(0.84, 0.95, crackNoise);
    rockCol *= 1.0 - crack * 0.28;

    float edgeNorm = clamp(dist / max(rw, 1e-3), 0.0, 1.0);
    float rimWet = smoothstep(0.45, 0.95, edgeNorm) * smoothstep(-0.65, 0.10, qyn);

    float wetRaw = fbm(vec2(q.x * 5.0 + q.y * 7.0, h * 15.0) + vec2(5.1), 2);
    float wetNoise = clamp(wetRaw * 0.5 + 0.5, 0.0, 1.0);
    float wetBreath = 0.88 + 0.12 * sin(windPh + h * 21.0);
    float wetMask = smoothstep(0.38, 0.78, wetNoise)
                  * clamp(uSRWetStrength, 0.0, 1.0)
                  * wetBreath
                  * rimWet;
    wetMask = clamp(wetMask, 0.0, 1.0);

    rockCol = mix(rockCol, uSRColWet, wetMask * 0.65);

    float contactFoam = smoothstep(0.75, 1.0, edgeNorm) * foamAll;
    rockCol *= 1.0 - contactFoam * 0.12;

    vec3 baseCol = perceptualBaseEnhance(rockCol, clamp(uDirLightIntensity, 0.0, 1.5));
    baseCol = clamp(baseCol, 0.0, 1.0);

    vec3 nrmF = nrm;
    vec3 viewF = vec3(0.0, 0.0, 1.0);
    float rimPowSafe = max(uRimPower, 1.0);
    ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'rimPowSafe')}

    vec3 worldPos = vec3(ax * 75.0, (1.0 - hp) * 48.0, 0.0);
    vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

    float hemi = nrmF.y * 0.5 + 0.5;
    float seaBounce = (1.0 - hp) * 0.28 + wetMask * 0.36 + foamAll * 0.22;
    seaBounce = clamp(seaBounce, 0.0, 1.0);

    vec3 gi = mix(uSRGroundColor, uSRSkyColor, hemi) * clamp(uSRGroundStrength, 0.0, 1.0);
    gi += uSRGIColor * seaBounce * clamp(uSRGIStrength, 0.0, 1.5);
    gi = clamp(gi, 0.0, 1.0);
    lit += baseCol * gi;

    vec3 sunVec = normalize(uSRSunDir + vec3(0.0001));
    float sunDot = max(dot(nrmF, sunVec), 0.0);
    float spec = pow(sunDot, 56.0)
               * (wetMask * 0.65 + topFac * 0.22)
               * clamp(uSRSunIntensity, 0.0, 1.5);
    spec = clamp(spec, 0.0, 1.0);
    lit += uDirLightColor * spec * 0.38;

    float sparkGate = (wetMask * 0.65 + foamAll * 0.25) * (1.0 - topFac * 0.25);
    vec2 sg = sp * 4.0 + q * 10.0;
    vec2 scell = floor(sg);
    float sh = srHash21(scell);
    vec2 sf = fract(sg) - 0.5;
    vec2 sj = vec2(srHash21(scell + vec2(2.7)), srHash21(scell + vec2(4.3))) - 0.5;
    float sdist = length(sf - sj * 0.52);

    float spark = (1.0 - smoothstep(0.0, 0.11, sdist))
                * step(1.0 - clamp(uSRSparkleDensity, 0.0, 1.0), sh)
                * sparkGate;
    float tw = 0.35 + 0.65 * (0.5 + 0.5 * sin(sparkPh + sh * 37.0));
    spark *= tw;
    spark *= clamp(uSRSunIntensity, 0.0, 1.5);
    spark *= (1.0 - hp * 0.45);
    spark = clamp(spark, 0.0, 1.0);

    lit += uSRColSparkle * spark * 0.40;

    float contact = 1.0 - smoothstep(0.0, 0.28, q.y + radius * 0.70);
    lit *= 1.0 - contact * 0.22;

    lit = applyFogBlend(lit, uSRFogColor, hp * hp * 0.28);

    vec2 pq = srPxq(vUv * 112.0, 56.0);
    lit += (srHash21(pq) - 0.5) * 0.010;
    lit = clamp(lit, 0.0, 1.0);

    float alpha = clamp(rockMask * seaMask, 0.0, 1.0);
    gl_FragColor = vec4(lit, alpha);
    return;
  }

  if (foamAll >= shadowMask) {
    vec3 foamCol = mix(uSRColFoam, uSRColSeaShallow, 0.18 + 0.12 * sin(foamPh + h * 13.0));
    foamCol += uDirLightColor * 0.08 * clamp(uSRSunIntensity, 0.0, 1.5);
    foamCol = applyFogBlend(foamCol, uSRFogColor, hp * hp * 0.22);

    vec2 pq = srPxq(vUv * 112.0, 56.0);
    foamCol += (srHash21(pq) - 0.5) * 0.009;
    foamCol = clamp(foamCol, 0.0, 1.0);

    float alpha = clamp(foamAll * seaMask * 0.82, 0.0, 0.92);
    gl_FragColor = vec4(foamCol, alpha);
    return;
  }

  vec3 shCol = mix(uSRColSeaMid, uSRShadowTint, 0.55) * 0.52;
  shCol += uSRGIColor * 0.05;
  shCol = applyFogBlend(shCol, uSRFogColor, hp * hp * 0.35);

  vec2 pq2 = srPxq(vUv * 112.0, 56.0);
  shCol += (srHash21(pq2) - 0.5) * 0.009;
  shCol = clamp(shCol, 0.0, 1.0);

  float shAlpha = clamp(
    shadowMask * seaMask * uSRShadowStrength * sunAbove,
    0.0,
    0.72
  );

  gl_FragColor = vec4(shCol, shAlpha);
}
`;

const _sunDirTmp = new THREE.Vector3();

export class SeaRocksShader {
  constructor(options = {}) {
    this.uniforms = _cloneSeaRocksUniforms(SEA_ROCKS_UNIFORMS);
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
    quantizeGeometryNormalsCPU(this.geometry, 4);

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: SEA_ROCKS_VERTEX,
      fragmentShader: SEA_ROCKS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 590;

    this._syncAccum = 1.0;
    this._chroma = _clampFinite(options.chroma, 1.0, 2.0, 1.10);
    this._iesScale = 1.0;

    this._windSmooth = 0.0;
    this._foamPhase = 0.0;
    this._windPhase = 0.0;
    this._sparkPhase = 0.0;
    this._shadowLenCur = _clampFinite(this.uniforms.uSRShadowLen.value, 0.05, 2.0, 0.72);
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uSRAspect.value = _clampFinite(a, 0.2, 2.5, 0.56);
  }

  setHorizon(y) {
    this.uniforms.uSRHorizonY.value = _clampFinite(y, 0.1, 0.9, 0.48);
  }

  setRockDensity(v) {
    this.uniforms.uSRRockDensity.value = _clampFinite(v, 0.0, 1.0, 0.58);
  }

  setFoamStrength(v) {
    this.uniforms.uSRFoamStrength.value = _clampFinite(v, 0.0, 1.5, 0.95);
  }

  setWetStrength(v) {
    this.uniforms.uSRWetStrength.value = _clampFinite(v, 0.0, 1.0, 0.82);
  }

  setShadowLength(v) {
    this.uniforms.uSRShadowLen.value = _clampFinite(v, 0.05, 2.0, 0.72);
    this._shadowLenCur = this.uniforms.uSRShadowLen.value;
  }

  setShadowStrength(v) {
    this.uniforms.uSRShadowStrength.value = _clampFinite(v, 0.0, 1.0, 0.48);
  }

  setSparkleDensity(v) {
    this.uniforms.uSRSparkleDensity.value = _clampFinite(v, 0.0, 1.0, 0.30);
  }

  setGIStrength(v) {
    this.uniforms.uSRGIStrength.value = _clampFinite(v, 0.0, 1.5, 0.55);
  }

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

  syncFromLight(lightShader) {
    if (!lightShader || typeof lightShader.getUniforms !== 'function') return;

    const lu = lightShader.getUniforms();
    const u = this.uniforms;

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
    u.uSRSunDir.value.copy(u.uDirLightDirection.value).negate();

    _copyColorFrom(
      u.uDirLightColor.value,
      lu.uDirLightColor && lu.uDirLightColor.value,
      1.00,
      0.98,
      0.92
    );

    const rawIntensity = lu.uDirLightIntensity ? lu.uDirLightIntensity.value : 1.0;
    const intensity = _clampFinite(rawIntensity, 0.0, 4.0, 1.0) *
      _clampFinite(this._iesScale, 0.25, 4.0, 1.0);
    u.uDirLightIntensity.value = intensity;
    u.uSRSunIntensity.value = intensity;

    _copyColorFrom(
      u.uShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.05,
      0.18,
      0.25
    );
    _copyVec3From(
      u.uSRShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.05,
      0.18,
      0.25
    );

    u.uSRGroundColor.value.set(
      _clampFinite(u.uSRShadowTint.value.x * 0.55, 0.0, 1.0, 0.04),
      _clampFinite(u.uSRShadowTint.value.y * 0.72, 0.0, 1.0, 0.16),
      _clampFinite(u.uSRShadowTint.value.z * 0.82, 0.0, 1.0, 0.22)
    );

    const rawAmbientIntensity = lu.uAmbientIntensity ? lu.uAmbientIntensity.value : 0.30;
    u.uSRGroundStrength.value = _clampFinite(rawAmbientIntensity * 0.82, 0.0, 1.0, 0.30);

    _copyColorFrom(
      u.uRimColor.value,
      lu.uRimColor && lu.uRimColor.value,
      0.92,
      0.98,
      1.00
    );

    _copyColorFrom(
      u.uAmbientColor.value,
      lu.uAmbientColor && lu.uAmbientColor.value,
      0.58,
      0.76,
      0.94
    );
    u.uSRSkyColor.value.set(
      _clampFinite(u.uAmbientColor.value.r, 0.0, 1.0, 0.58),
      _clampFinite(u.uAmbientColor.value.g, 0.0, 1.0, 0.76),
      _clampFinite(u.uAmbientColor.value.b, 0.0, 1.0, 0.94)
    );
    u.uSRFogColor.value.set(
      _clampFinite(u.uAmbientColor.value.r * 0.78 + 0.18, 0.0, 1.0, 0.72),
      _clampFinite(u.uAmbientColor.value.g * 0.86 + 0.14, 0.0, 1.0, 0.88),
      _clampFinite(u.uAmbientColor.value.b * 0.88 + 0.10, 0.0, 1.0, 0.92)
    );

    u.uCelSteps.value = _clampFinite(lu.uCelSteps ? lu.uCelSteps.value : 4.0, 1.0, 8.0, 4.0);
    u.uRimPower.value = _clampFinite(lu.uRimPower ? lu.uRimPower.value : 3.0, 1.0, 8.0, 3.0);
    u.uRimIntensity.value = _clampFinite(lu.uRimIntensity ? lu.uRimIntensity.value : 0.45, 0.0, 1.0, 0.45);
    u.uShadowSoftness.value = _clampFinite(lu.uShadowSoftness ? lu.uShadowSoftness.value : 0.05, 0.0, 0.5, 0.05);
    u.uAmbientIntensity.value = _clampFinite(rawAmbientIntensity, 0.0, 1.0, 0.30);
    u.uPerceptualEnhance.value = _clampFinite(u.uPerceptualEnhance.value, 0.0, 1.0, 0.85);

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

    _applyChromaToVec3(
      u.uSRColFoam.value,
      this._chroma,
      u.uSRColFoam.value
    );

    _applyTintToVec3(
      u.uSRColRockTan.value,
      u.uDirLightColor.value,
      0.22,
      u.uSRColRockTan.value
    );

    _applyTintToVec3(
      u.uSRColSeaMid.value,
      u.uSRShadowTint.value,
      0.28,
      u.uSRColSeaMid.value
    );

    u.uSRGIColor.value.set(
      _clampFinite(u.uSRColSeaShallow.value.x * 0.72, 0.0, 1.0, 0.18),
      _clampFinite(u.uSRColSeaShallow.value.y * 0.94, 0.0, 1.0, 0.76),
      _clampFinite(u.uSRColSeaShallow.value.z * 0.92, 0.0, 1.0, 0.70)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;

    dt = _clampFinite(dt, 0.0, 0.1, 0.0);
    elapsed = _finite(elapsed, 0.0);
    u.uTime.value = elapsed;

    const windVec = u.uWind && u.uWind.value;
    const rawWindX = windVec && Number.isFinite(windVec.x) ? windVec.x : 0.0;
    const windX = _clampFinite(rawWindX, -8.0, 8.0, 0.0);

    this._windSmooth = damp(this._windSmooth, windX, 2.0, dt);
    this._windSmooth = _finite(this._windSmooth, 0.0);

    this._foamPhase = (this._foamPhase + dt * (0.85 + Math.abs(this._windSmooth) * 0.15)) % TWO_PI;
    if (this._foamPhase < 0.0) this._foamPhase += TWO_PI;

    this._windPhase = (this._windPhase + dt * (0.22 + Math.abs(this._windSmooth) * 0.12)) % TWO_PI;
    if (this._windPhase < 0.0) this._windPhase += TWO_PI;

    this._sparkPhase = (this._sparkPhase + dt * (1.55 + Math.abs(this._windSmooth) * 0.20)) % TWO_PI;
    if (this._sparkPhase < 0.0) this._sparkPhase += TWO_PI;

    u.uSRFoamPhase.value = _finite(this._foamPhase, 0.0);
    u.uSRWindPhase.value = _finite(this._windPhase, 0.0);
    u.uSRSparkPhase.value = _finite(this._sparkPhase, 0.0);

    const elev = _clampFinite(-u.uDirLightDirection.value.y, 0.05, 1.0, 0.5);
    const targetLen = _clampFinite(0.34 / elev, 0.20, 1.35, 0.72);
    this._shadowLenCur = damp(this._shadowLenCur, targetLen, 2.5, dt);
    u.uSRShadowLen.value = _finite(this._shadowLenCur, 0.72);

    if (camera && camera.position) {
      u.uCamPos.value.set(
        _finite(camera.position.x, 0.0),
        _finite(camera.position.y, 0.0),
        _finite(camera.position.z, 0.0)
      );
    }

    if (lightShader) {
      this._syncAccum += dt;
      if (this._syncAccum >= 0.25) {
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

export function createSeaRocksShader(options = {}) {
  return new SeaRocksShader(options);
}

export default {
  SeaRocksShader,
  createSeaRocksShader,
  SEA_ROCKS_UNIFORMS,
  SEA_ROCKS_VERTEX,
  SEA_ROCKS_FRAGMENT,
};
