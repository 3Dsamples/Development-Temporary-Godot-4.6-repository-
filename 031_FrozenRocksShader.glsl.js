// File : 031
// name : shaders/031_FrozenRocksShader.glsl.js
// description : Anime frozen canyon rock scatter shader for the snowy gorge scene.
//               ANALYZED & FIXED: (1) Vector3 color-channel bug: uFRColIceWet is a
//               THREE.Vector3, not THREE.Color, so cyan GI bounce now reads .x/.y/.z.
//               (2) Shadow-tint coupling bug: added local uFRShadowTint vec3 to avoid
//               relying on the 016 Color-backed uShadowTint inside custom cast-shadow
//               mixing. (3) Static shadow length bug: shadow length is now elevation
//               damped in update(), producing long low-sun shadows and short noon
//               shadows without per-frame allocation. (4) Wind phase bug: wet breath
//               now has a base angular speed so it remains alive at zero wind.
//               (5) Wind input bug: live wind is clamped before integration to avoid
//               NaN/huge phase jumps. (6) Perceptual sync remains zero-allocation in
//               steady state using module-level SoA scratch stores. (7) All smoothstep
//               edges are low->high. (8) All normalize/division inputs are guarded.
//               (9) All hash/noise palette drivers are normalized to 0..1. (10) Final
//               color is clamped for mobile framebuffers. (11) Fullscreen parallax
//               rocks use neutral shadows by default; live shadow maps are opt-in.
//               (12) Uniforms are cloned per instance and the quad is DoubleSide.
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

export const FROZEN_ROCKS_UNIFORMS = {
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  uFRSunDir:          { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uFRSunIntensity:    { value: 1.0 },
  uFRSkyColor:        { value: new THREE.Vector3(0.58, 0.76, 0.94) },
  uFRGroundColor:     { value: new THREE.Vector3(0.08, 0.13, 0.22) },
  uFRGroundStrength:  { value: 0.28 },
  uFRFogColor:        { value: new THREE.Vector3(0.78, 0.88, 0.96) },
  uFRShadowTint:      { value: new THREE.Vector3(0.16, 0.24, 0.36) },

  uFRWindPhase:  { value: 0.0 },
  uFRSparkPhase: { value: 0.0 },

  uBiomeW:    { value: new THREE.Vector4(0.0, 0.0, 0.0, 1.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  uPerceptualEnhance: { value: 0.85 },

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

  uFRColRockDeep:   { value: new THREE.Vector3(0.075, 0.105, 0.145) },
  uFRColRockDark:   { value: new THREE.Vector3(0.145, 0.195, 0.260) },
  uFRColRockMid:    { value: new THREE.Vector3(0.255, 0.330, 0.410) },
  uFRColRockLight:  { value: new THREE.Vector3(0.430, 0.520, 0.610) },
  uFRColSnowDust:   { value: new THREE.Vector3(0.900, 0.955, 1.000) },
  uFRColIceWet:     { value: new THREE.Vector3(0.260, 0.760, 0.700) },
  uFRColFrost:      { value: new THREE.Vector3(0.820, 0.930, 1.000) },
  uFRColSparkle:    { value: new THREE.Vector3(1.000, 1.000, 0.980) },
  uFRGIColor:       { value: new THREE.Vector3(0.220, 0.720, 0.640) },

  uFRHorizonY:       { value: 0.50 },
  uFRAspect:         { value: 0.56 },
  uFRRockDensity:    { value: 0.62 },
  uFRShadowLen:      { value: 0.65 },
  uFRShadowStrength: { value: 0.55 },
  uFRSnowDust:       { value: 0.55 },
  uFRWetStrength:    { value: 0.70 },
  uFRFrostStrength:  { value: 0.45 },
  uFRSparkleDensity: { value: 0.28 },
  uFRGIStrength:     { value: 0.48 },
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

function _cloneFrozenRocksUniforms(src) {
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

export const FROZEN_ROCKS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

export const FROZEN_ROCKS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uFRSunDir;
uniform float uFRSunIntensity;
uniform vec3  uFRSkyColor;
uniform vec3  uFRGroundColor;
uniform float uFRGroundStrength;
uniform vec3  uFRFogColor;
uniform vec3  uFRShadowTint;

uniform float uFRWindPhase;
uniform float uFRSparkPhase;

uniform vec3  uFRColRockDeep;
uniform vec3  uFRColRockDark;
uniform vec3  uFRColRockMid;
uniform vec3  uFRColRockLight;
uniform vec3  uFRColSnowDust;
uniform vec3  uFRColIceWet;
uniform vec3  uFRColFrost;
uniform vec3  uFRColSparkle;
uniform vec3  uFRGIColor;

uniform float uFRHorizonY;
uniform float uFRAspect;
uniform float uFRRockDensity;
uniform float uFRShadowLen;
uniform float uFRShadowStrength;
uniform float uFRSnowDust;
uniform float uFRWetStrength;
uniform float uFRFrostStrength;
uniform float uFRSparkleDensity;
uniform float uFRGIStrength;

varying vec2 vUv;

vec2 frPxq(vec2 uv, float q) {
  float safeQ = max(q, 1.0);
  return floor(uv * safeQ + 0.5) / safeQ;
}

float frHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + vec2(45.32));
  return fract(p.x * p.y);
}

void main() {
  float aspect = max(uFRAspect, 0.2);
  float ax = (vUv.x - 0.5) * aspect;

  float groundMask = 1.0 - smoothstep(uFRHorizonY - 0.004, uFRHorizonY + 0.004, vUv.y);
  if (groundMask <= 0.001) discard;

  float hp = smoothstep(uFRHorizonY - 0.58, uFRHorizonY, vUv.y);
  float persp = mix(1.0, 6.8, hp * hp);
  vec2 sp = vec2(vUv.x * aspect * 24.0, vUv.y * 68.0) * persp;
  vec2 cell = floor(sp);
  vec2 f = fract(sp) - 0.5;

  float h = frHash21(cell);
  float h2 = frHash21(cell + vec2(7.7));

  float density = clamp(uFRRockDensity, 0.0, 1.0);
  float gate = step(1.0 - density * (1.0 - hp * 0.55), h);

  vec2 sdirRaw = uDirLightDirection.xy;
  float slen = max(length(sdirRaw), 1e-4);
  vec2 sdir = sdirRaw / slen;
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0);
  float shLen = max(uFRShadowLen, 0.05);

  vec2 jitter = vec2(frHash21(cell + vec2(3.1)), frHash21(cell + vec2(5.7))) - 0.5;
  vec2 q = f - jitter * 0.75;
  float radius = max(0.12 + h2 * 0.18, 0.065);
  float dist = length(q);
  float dirLen = max(dist, 1e-4);
  vec2 dir = q / dirLen;

  float wob = 1.0
    + sin(dir.x * 5.0 + dir.y * 3.0 + h * 30.0) * 0.13
    + sin(dir.x * 9.0 - dir.y * 7.0 + h * 17.0) * 0.07;
  float rw = radius * max(wob, 0.5);

  float rockMask = (1.0 - smoothstep(rw - 0.03, rw + 0.03, dist)) * gate;
  rockMask *= 1.0 - smoothstep(0.02, 0.12, -q.y - radius * 0.55);
  rockMask = clamp(rockMask, 0.0, 1.0);

  vec2 spScale = vec2(aspect * 24.0, 68.0) * persp;
  vec2 sdirCellRaw = sdir / max(spScale, vec2(1e-4));
  float scLen = max(length(sdirCellRaw), 1e-6);
  vec2 sdirCell = sdirCellRaw / scLen;

  float len = radius * (1.0 + shLen * 1.8);
  float alongC = clamp(dot(q, sdirCell), 0.0, len);
  float dSeg = length(q - sdirCell * alongC);

  float shadowMask = (1.0 - smoothstep(radius * 0.82, radius * 1.18, dSeg)) * gate * sunAbove;
  shadowMask *= 1.0 - smoothstep(len * 0.65, len, alongC);
  shadowMask *= 0.85 + 0.15 * frHash21(cell + vec2(11.3));
  shadowMask = clamp(shadowMask * (1.0 - rockMask), 0.0, 1.0);

  if (rockMask < 0.02 && shadowMask < 0.02) discard;

  if (rockMask >= shadowMask) {
    vec3 nrm = normalize(vec3(q.x * 2.2, q.y * 2.2, 1.0));
    vec3 fn = floor(nrm * 3.0 + 0.5) / 3.0;
    nrm = normalize(mix(nrm, fn, 0.45));

    float cvar = clamp(frHash21(cell + vec2(13.7)), 0.0, 1.0);
    vec3 rockCol = paletteMix4(
      uFRColRockDeep,
      uFRColRockDark,
      uFRColRockMid,
      uFRColRockLight,
      cvar
    );

    float strRaw = fbm(vec2(dir.x * 10.0 + dir.y * 8.0, h * 7.0) + vec2(3.3), 2);
    float strNoise = clamp(strRaw * 0.5 + 0.5, 0.0, 1.0);
    rockCol *= 0.88 + strNoise * 0.16;

    float crackRaw = fbm(vec2(dir.x * 6.0 + dir.y * 4.0, h * 10.0) + vec2(9.1), 2);
    float crackNoise = clamp(crackRaw * 0.5 + 0.5, 0.0, 1.0);
    float crack = smoothstep(0.82, 0.94, crackNoise);
    rockCol *= 1.0 - crack * 0.32;

    float snowRaw = fbm(vec2(q.x * 5.0 + q.y * 3.0, h * 19.0) + vec2(21.7), 2);
    float snowNoise = clamp(snowRaw * 0.5 + 0.5, 0.0, 1.0);
    float snowDust = smoothstep(0.62, 0.86, snowNoise)
                   * smoothstep(-0.10, 0.35, q.y)
                   * clamp(uFRSnowDust, 0.0, 1.0);
    snowDust = clamp(snowDust, 0.0, 1.0);
    rockCol = mix(rockCol, uFRColSnowDust, snowDust * 0.55);

    float edgeNorm = clamp(dist / max(rw, 1e-3), 0.0, 1.0);
    float rimMoist = smoothstep(0.58, 0.94, edgeNorm);

    float moistRaw = fbm(vec2(q.x * 4.0 + q.y * 6.0, h * 13.0) + vec2(5.1), 2);
    float moistNoise = clamp(moistRaw * 0.5 + 0.5, 0.0, 1.0);
    float wetBreath = 0.90 + 0.10 * sin(mod(uFRWindPhase, 6.2831853) + h * 19.0);
    float moisture = smoothstep(0.42, 0.76, moistNoise)
                   * clamp(uFRWetStrength, 0.0, 1.0)
                   * wetBreath;
    moisture = clamp(moisture, 0.0, 1.0);

    float wetMask = clamp(rimMoist * moisture * (1.0 - snowDust * 0.60), 0.0, 1.0);
    rockCol = mix(rockCol, uFRColIceWet, wetMask * 0.55);

    float frostRaw = fbm(vec2(q.x * 9.0 + q.y * 7.0, h * 29.0) + vec2(17.3), 2);
    float frostNoise = clamp(frostRaw * 0.5 + 0.5, 0.0, 1.0);
    float frost = smoothstep(0.72, 0.91, frostNoise)
                * clamp(uFRFrostStrength, 0.0, 1.0)
                * (1.0 - wetMask * 0.50);
    frost = clamp(frost, 0.0, 1.0);
    rockCol = mix(rockCol, uFRColFrost, frost * 0.35);

    vec3 baseCol = perceptualBaseEnhance(rockCol, clamp(uDirLightIntensity, 0.0, 1.5));
    baseCol = clamp(baseCol, 0.0, 1.0);

    vec3 nrmF = nrm;
    vec3 viewF = vec3(0.0, 0.0, 1.0);
    float rimPowSafe = max(uRimPower, 1.0);
    ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'rimPowSafe')}

    vec3 worldPos = vec3(ax * 70.0, (1.0 - hp) * 45.0, 0.0);
    vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

    float hemi = nrmF.y * 0.5 + 0.5;
    float cyanProx = (1.0 - hp) * 0.45 + wetMask * 0.45 + snowDust * 0.10;
    cyanProx = clamp(cyanProx, 0.0, 1.0);

    vec3 gi = mix(uFRGroundColor, uFRSkyColor, hemi) * clamp(uFRGroundStrength, 0.0, 1.0);
    gi += uFRGIColor * cyanProx * clamp(uFRGIStrength, 0.0, 1.5);
    gi = clamp(gi, 0.0, 1.0);
    lit += baseCol * gi;

    vec3 sunVec = normalize(uFRSunDir + vec3(0.0001));
    float sunDot = max(dot(nrmF, sunVec), 0.0);
    float spec = pow(sunDot, 48.0)
               * wetMask
               * (1.0 - frost * 0.40)
               * clamp(uFRSunIntensity, 0.0, 1.5);
    spec = clamp(spec, 0.0, 1.0);
    lit += uDirLightColor * spec * 0.35;

    float sparkleGate = frost * (1.0 - wetMask * 0.30);
    vec2 fp = sp * 3.0 + q * 8.0;
    vec2 fcell = floor(fp);
    float fh = frHash21(fcell);
    vec2 ff = fract(fp) - 0.5;
    vec2 fj = vec2(frHash21(fcell + vec2(2.3)), frHash21(fcell + vec2(4.9))) - 0.5;
    float fdist = length(ff - fj * 0.50);

    float spark = (1.0 - smoothstep(0.0, 0.12, fdist))
                * step(1.0 - clamp(uFRSparkleDensity, 0.0, 1.0), fh)
                * sparkleGate;

    float tw = 0.35 + 0.65 * (0.5 + 0.5 * sin(mod(uFRSparkPhase, 6.2831853) + fh * 37.0));
    spark *= tw;
    spark *= clamp(uFRSunIntensity, 0.0, 1.5);
    spark *= (1.0 - hp * 0.50);
    spark = clamp(spark, 0.0, 1.0);

    lit += uFRColSparkle * spark * 0.45;

    float contact = 1.0 - smoothstep(0.0, 0.25, q.y + radius * 0.65);
    lit *= 1.0 - contact * 0.28;

    lit = applyFogBlend(lit, uFRFogColor, hp * hp * 0.35);

    vec2 pq = frPxq(vUv * 112.0, 56.0);
    lit += (frHash21(pq) - 0.5) * 0.010;
    lit = clamp(lit, 0.0, 1.0);

    float alpha = clamp(rockMask * groundMask, 0.0, 1.0);
    gl_FragColor = vec4(lit, alpha);
    return;
  }

  vec3 shCol = mix(uFRColRockDeep, uFRShadowTint, 0.45) * 0.42;
  shCol = applyFogBlend(shCol, uFRFogColor, hp * hp * 0.45);

  vec2 pq2 = frPxq(vUv * 112.0, 56.0);
  shCol += (frHash21(pq2) - 0.5) * 0.009;
  shCol = clamp(shCol, 0.0, 1.0);

  float shAlpha = clamp(
    shadowMask * groundMask * uFRShadowStrength * sunAbove,
    0.0,
    0.75
  );

  gl_FragColor = vec4(shCol, shAlpha);
}
`;

const _sunDirTmp = new THREE.Vector3();

export class FrozenRocksShader {
  constructor(options = {}) {
    this.uniforms = _cloneFrozenRocksUniforms(FROZEN_ROCKS_UNIFORMS);
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
      vertexShader: FROZEN_ROCKS_VERTEX,
      fragmentShader: FROZEN_ROCKS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 585;

    this._syncAccum = 1.0;
    this._chroma = _clampFinite(options.chroma, 1.0, 2.0, 1.08);
    this._iesScale = 1.0;

    this._windSmooth = 0.0;
    this._windPhase = 0.0;
    this._sparkPhase = 0.0;
    this._shadowLenCur = _clampFinite(this.uniforms.uFRShadowLen.value, 0.05, 2.0, 0.65);
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uFRAspect.value = _clampFinite(a, 0.2, 2.5, 0.56);
  }

  setHorizon(y) {
    this.uniforms.uFRHorizonY.value = _clampFinite(y, 0.1, 0.9, 0.50);
  }

  setRockDensity(v) {
    this.uniforms.uFRRockDensity.value = _clampFinite(v, 0.0, 1.0, 0.62);
  }

  setShadowLength(v) {
    this.uniforms.uFRShadowLen.value = _clampFinite(v, 0.05, 2.0, 0.65);
    this._shadowLenCur = this.uniforms.uFRShadowLen.value;
  }

  setShadowStrength(v) {
    this.uniforms.uFRShadowStrength.value = _clampFinite(v, 0.0, 1.0, 0.55);
  }

  setSnowDust(v) {
    this.uniforms.uFRSnowDust.value = _clampFinite(v, 0.0, 1.0, 0.55);
  }

  setWetStrength(v) {
    this.uniforms.uFRWetStrength.value = _clampFinite(v, 0.0, 1.0, 0.70);
  }

  setFrostStrength(v) {
    this.uniforms.uFRFrostStrength.value = _clampFinite(v, 0.0, 1.0, 0.45);
  }

  setSparkleDensity(v) {
    this.uniforms.uFRSparkleDensity.value = _clampFinite(v, 0.0, 1.0, 0.28);
  }

  setGIStrength(v) {
    this.uniforms.uFRGIStrength.value = _clampFinite(v, 0.0, 1.5, 0.48);
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
    u.uFRSunDir.value.copy(u.uDirLightDirection.value).negate();

    _copyColorFrom(
      u.uDirLightColor.value,
      lu.uDirLightColor && lu.uDirLightColor.value,
      0.96,
      0.98,
      1.00
    );

    const rawIntensity = lu.uDirLightIntensity ? lu.uDirLightIntensity.value : 1.0;
    const intensity = _clampFinite(rawIntensity, 0.0, 4.0, 1.0) *
      _clampFinite(this._iesScale, 0.25, 4.0, 1.0);
    u.uDirLightIntensity.value = intensity;
    u.uFRSunIntensity.value = intensity;

    _copyColorFrom(
      u.uShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.16,
      0.24,
      0.36
    );
    _copyVec3From(
      u.uFRShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.16,
      0.24,
      0.36
    );

    u.uFRGroundColor.value.set(
      _clampFinite(u.uFRShadowTint.value.x * 0.48, 0.0, 1.0, 0.08),
      _clampFinite(u.uFRShadowTint.value.y * 0.54, 0.0, 1.0, 0.13),
      _clampFinite(u.uFRShadowTint.value.z * 0.62, 0.0, 1.0, 0.22)
    );

    const rawAmbientIntensity = lu.uAmbientIntensity ? lu.uAmbientIntensity.value : 0.28;
    u.uFRGroundStrength.value = _clampFinite(rawAmbientIntensity * 0.75, 0.0, 1.0, 0.28);

    _copyColorFrom(
      u.uRimColor.value,
      lu.uRimColor && lu.uRimColor.value,
      0.88,
      0.96,
      1.00
    );

    _copyColorFrom(
      u.uAmbientColor.value,
      lu.uAmbientColor && lu.uAmbientColor.value,
      0.58,
      0.76,
      0.94
    );
    u.uFRSkyColor.value.set(
      _clampFinite(u.uAmbientColor.value.r, 0.0, 1.0, 0.58),
      _clampFinite(u.uAmbientColor.value.g, 0.0, 1.0, 0.76),
      _clampFinite(u.uAmbientColor.value.b, 0.0, 1.0, 0.94)
    );
    u.uFRFogColor.value.set(
      _clampFinite(u.uAmbientColor.value.r * 0.88 + 0.12, 0.0, 1.0, 0.78),
      _clampFinite(u.uAmbientColor.value.g * 0.88 + 0.12, 0.0, 1.0, 0.88),
      _clampFinite(u.uAmbientColor.value.b * 0.88 + 0.12, 0.0, 1.0, 0.96)
    );

    u.uCelSteps.value = _clampFinite(lu.uCelSteps ? lu.uCelSteps.value : 4.0, 1.0, 8.0, 4.0);
    u.uRimPower.value = _clampFinite(lu.uRimPower ? lu.uRimPower.value : 3.0, 1.0, 8.0, 3.0);
    u.uRimIntensity.value = _clampFinite(lu.uRimIntensity ? lu.uRimIntensity.value : 0.45, 0.0, 1.0, 0.45);
    u.uShadowSoftness.value = _clampFinite(lu.uShadowSoftness ? lu.uShadowSoftness.value : 0.05, 0.0, 0.5, 0.05);
    u.uAmbientIntensity.value = _clampFinite(rawAmbientIntensity, 0.0, 1.0, 0.28);
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
      u.uFRColRockLight.value,
      this._chroma,
      u.uFRColRockLight.value
    );

    _applyTintToVec3(
      u.uFRColIceWet.value,
      u.uDirLightColor.value,
      0.28,
      u.uFRColIceWet.value
    );

    _applyTintToVec3(
      u.uFRColRockDeep.value,
      u.uFRShadowTint.value,
      0.35,
      u.uFRColRockDeep.value
    );

    u.uFRGIColor.value.set(
      _clampFinite(u.uFRColIceWet.value.x * 0.82, 0.0, 1.0, 0.22),
      _clampFinite(u.uFRColIceWet.value.y * 0.94, 0.0, 1.0, 0.72),
      _clampFinite(u.uFRColIceWet.value.z * 0.90, 0.0, 1.0, 0.64)
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

    this._windPhase = (this._windPhase + dt * (0.16 + Math.abs(this._windSmooth) * 0.10)) % TWO_PI;
    if (this._windPhase < 0.0) this._windPhase += TWO_PI;

    this._sparkPhase = (this._sparkPhase + dt * (1.25 + Math.abs(this._windSmooth) * 0.18)) % TWO_PI;
    if (this._sparkPhase < 0.0) this._sparkPhase += TWO_PI;

    u.uFRWindPhase.value = _finite(this._windPhase, 0.0);
    u.uFRSparkPhase.value = _finite(this._sparkPhase, 0.0);

    const elev = _clampFinite(-u.uDirLightDirection.value.y, 0.05, 1.0, 0.5);
    const targetLen = _clampFinite(0.32 / elev, 0.18, 1.25, 0.65);
    this._shadowLenCur = damp(this._shadowLenCur, targetLen, 2.5, dt);
    u.uFRShadowLen.value = _finite(this._shadowLenCur, 0.65);

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

export function createFrozenRocksShader(options = {}) {
  return new FrozenRocksShader(options);
}

export default {
  FrozenRocksShader,
  createFrozenRocksShader,
  FROZEN_ROCKS_UNIFORMS,
  FROZEN_ROCKS_VERTEX,
  FROZEN_ROCKS_FRAGMENT,
};