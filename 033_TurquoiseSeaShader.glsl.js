// File : 033
// name : shaders/033_TurquoiseSeaShader.glsl.js
// description : Anime turquoise sea surface shader for the coastal ocean scene.
//               ANALYZED & FIXED: (1) PERCEPTUAL DRIFT BUG: shallow/glow/deep water
//               palette uniforms were chromatized/tinted from their current values on
//               every 4 Hz sync, causing cumulative saturation/hue drift. A immutable
//               per-instance base palette snapshot is now restored before every
//               perceptual update. (2) MANUAL CHOP OVERRIDE BUG: wind-driven chop
//               smoothing overwrote user-set chop permanently. Chop now has a stored
//               base value and wind adds a bounded modulation on top of it. (3) FOAM
//               SUN INTERACTION BUG: foam color was static. Foam now mixes locally
//               with the live 016 directional color in-shader, avoiding uniform drift.
//               (4) ENHANCEMENT STRENGTH BUG: perceptualBaseEnhance received intensity
//               up to 1.5. It is now clamped to the intended 0..1 strength range.
//               (5) TONE/FOG BUG: ShaderMaterial now disables scene fog and tone
//               mapping to preserve exact anime palette and reduce unnecessary uniform
//               injection / post-transform cost. (6) CUMULATIVE COLOR COPY BUG: Color
//               -> Vector3 and Vector3 -> Color synchronization uses explicit sanitized
//               component copies, never ambiguous .copy() across incompatible types.
//               (7) OPTION ASSIGNMENT BUG: uniform options now accept Color, Vector3,
//               hex numbers, arrays, and plain {r,g,b}/{x,y,z} objects safely.
//               (8) PHASE-WRAP POP BUG: all animated phases are integrated in JS at
//               exact angular frequencies and wrapped by 2pi. Shader sine terms use
//               only these continuous phases. (9) NOISE DISCONTINUITY BUG: FBM inputs
//               remain static spatial coordinates; phases are only added to analytic
//               sine arguments. (10) SMOOTHSTEP AUDIT: every smoothstep() edge is
//               low->high. (11) GUARDED MATH: aspect, wave scale/amp, chop, foam
//               threshold, normalize inputs, divisions, lengths, dt, elapsed, wind,
//               camera, and all scalar uniforms are guarded against NaN/Infinity.
//               (12) NOISE RANGE BUG: every fbm/hash value driving palette/mask/normal
//               detail is normalized and clamped to 0..1. (13) FALSE-DEPTH SHADOW BUG:
//               fullscreen parallax sea uses a neutral 1x1 white shadow texture and
//               identity shadow matrix by default. Live scene shadow-map binding is
//               opt-in via options.sceneShadows. (14) OVERBRIGHT BUG: foam, caustics,
//               subsurface, specular, sparkle, GI and dither are bounded; final color
//               is clamped for mobile framebuffers. (15) FILL-RATE BUG: sky pixels
//               discard before expensive noise work. (16) INSTANCE SHARING BUG:
//               uniforms are cloned per manager instance. (17) CULLING BUG: fullscreen
//               quad uses DoubleSide and frustumCulled=false. (18) ZERO STEADY-STATE
//               ALLOCATION: perceptual SoA scratch buffers are module-level and reused.
//               Renders a perspective-compressed animated water plane with cel
//               quantized wave bands, directional swell/chop, foam crests, foam
//               flecks, caustic interference, subsurface cyan glow, sun glitter,
//               sparkle twinkle, sky reflection Fresnel, horizon haze and pixel-art
//               de-banding dither. Fully interacts with
//               016_DirectionalLightShader.glsl.js through
//               computeAnimeDirectionalLight(), live directional color/intensity/
//               direction synchronization, rim light, ambient light, shadow tint and
//               optional live shadow-map sampling. Global illumination is provided
//               by unique local uTS* uniforms (hemisphere sky/ground bounce + cyan
//               shallow-water bounce) to avoid duplicate uniform collisions with the
//               016 directional chunk. Composed on shaders/000_BaseShader.glsl.js
//               with all required imports: GLSL_GLOBALS, GLSL_NOISE, GLSL_BIOME,
//               GLSL_COLOR_UTILS, GLSL_PERCEPTUAL_BASE_ENHANCER,
//               GLSL_DIRECTIONAL_LIGHT, 006_normalQuantizer rim-exaggeration snippet
//               + CPU normal bake, 020_gmp_perceptual_color SoA perceptual helpers,
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
import { clamp, damp, TWO_PI } from '../utils/001_gmp_MathUtils.js';

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
_whiteShadow.format = THREE.RGBAFormat;
_whiteShadow.type = THREE.UnsignedByteType;
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
  return clamp(x, min, max);
}

function _isPlainColorLike(v) {
  return !!v && typeof v === 'object' && (
    typeof v.r === 'number' ||
    typeof v.x === 'number'
  );
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
      if (typeof src.r === 'number' && typeof src.x !== 'number') x = src.r;
      if (typeof src.g === 'number' && typeof src.y !== 'number') y = src.g;
      if (typeof src.b === 'number' && typeof src.z !== 'number') z = src.b;
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
/* 4. TURQUOISE SEA DEFAULT UNIFORMS (JS side)                         */
/* ------------------------------------------------------------------ */
export const TURQUOISE_SEA_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // Unique local global-illumination providers (no chunk collisions)
  uTSSunDir:          { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uTSSunIntensity:    { value: 1.0 },
  uTSSkyColor:        { value: new THREE.Vector3(0.58, 0.76, 0.94) },
  uTSGroundColor:     { value: new THREE.Vector3(0.035, 0.145, 0.205) },
  uTSGroundStrength:  { value: 0.30 },
  uTSFogColor:        { value: new THREE.Vector3(0.74, 0.90, 0.94) },

  // Pre-integrated animation phases
  uTSFlowPhase:   { value: 0.0 },
  uTSSwellPhase:  { value: 0.0 },
  uTSChopPhase:   { value: 0.0 },
  uTSFoamPhase:   { value: 0.0 },
  uTSSparkPhase:  { value: 0.0 },
  uTSSlowPhase:   { value: 0.0 },

  // GLSL_BIOME providers
  uBiomeW:    { value: new THREE.Vector4(0.0, 0.0, 1.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // 017 perceptual enhancer provider
  uPerceptualEnhance: { value: 0.85 },

  // 016 directional light providers (safe defaults; synced at runtime)
  uDirLightColor:     { value: new THREE.Color(1.00, 0.98, 0.92) },
  uDirLightDirection: { value: new THREE.Vector3(-0.35, -0.78, -0.52) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimColor:          { value: new THREE.Color(0.92, 0.98, 1.00) },
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.04, 0.16, 0.24) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.30 },
  uAmbientColor:      { value: new THREE.Color(0.58, 0.76, 0.94) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: 0.0 },
  uShadowNormalBias:  { value: 0.0 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Turquoise sea palette (exact coastal image colors)
  uTSDeep:        { value: new THREE.Vector3(0.025, 0.190, 0.285) }, // 0x063049 deep ocean teal
  uTSMid:         { value: new THREE.Vector3(0.055, 0.480, 0.545) }, // 0x0e7a8b mid turquoise
  uTSShallow:     { value: new THREE.Vector3(0.235, 0.820, 0.760) }, // 0x3cd1c2 shallow cyan
  uTSFoam:        { value: new THREE.Vector3(0.965, 0.995, 1.000) }, // 0xf6feff white foam
  uTSGlow:        { value: new THREE.Vector3(0.310, 0.910, 0.800) }, // 0x4fe8cc subsurface glow
  uTSSparkle:     { value: new THREE.Vector3(1.000, 1.000, 0.985) }, // 0xfffffb sun sparkle
  uTSGIColor:     { value: new THREE.Vector3(0.170, 0.760, 0.700) }, // cyan shallow bounce

  // Sea surface parameters
  uTSHorizonY:         { value: 0.48 },
  uTSAspect:           { value: 0.56 },
  uTSWaveScale:        { value: 1.00 },
  uTSWaveAmp:          { value: 0.72 },
  uTSChop:             { value: 0.78 },
  uTSFoamThreshold:    { value: 0.74 },
  uTSFoamAmount:       { value: 0.88 },
  uTSCausticStrength:  { value: 0.82 },
  uTSSSSStrength:      { value: 0.90 },
  uTSReflectStrength:  { value: 0.58 },
  uTSSunGlintStrength: { value: 0.85 },
  uTSSparkleDensity:   { value: 0.32 },
  uTSGIStrength:       { value: 0.55 },
  uTSHazeStrength:     { value: 0.70 },
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

function _cloneTurquoiseSeaUniforms(src) {
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

  if (target && target.isColor) {
    if (val && val.isColor) {
      target.copy(val);
    } else if (val && val.isVector3) {
      target.setRGB(val.x, val.y, val.z);
    } else if (typeof val === 'number') {
      target.set(val);
    } else if (Array.isArray(val) && val.length >= 3) {
      target.setRGB(val[0], val[1], val[2]);
    } else if (_isPlainColorLike(val)) {
      if (typeof val.r === 'number') {
        target.setRGB(val.r, val.g, val.b);
      } else if (typeof val.x === 'number') {
        target.setRGB(val.x, val.y, val.z);
      }
    }
    return;
  }

  if (target && target.isVector3) {
    if (val && val.isVector3) {
      target.copy(val);
    } else if (val && val.isColor) {
      target.set(val.r, val.g, val.b);
    } else if (typeof val === 'number') {
      target.setScalar(val);
    } else if (Array.isArray(val) && val.length >= 3) {
      target.set(val[0], val[1], val[2]);
    } else if (_isPlainColorLike(val)) {
      if (typeof val.x === 'number') {
        target.set(val.x, val.y, val.z);
      } else if (typeof val.r === 'number') {
        target.set(val.r, val.g, val.b);
      }
    }
    return;
  }

  if (target && target.isVector2) {
    if (val && val.isVector2) {
      target.copy(val);
    } else if (typeof val === 'number') {
      target.setScalar(val);
    } else if (Array.isArray(val) && val.length >= 2) {
      target.set(val[0], val[1]);
    } else if (val && typeof val === 'object' && typeof val.x === 'number') {
      target.set(val.x, val.y);
    }
    return;
  }

  if (target && target.isVector4) {
    if (val && val.isVector4) {
      target.copy(val);
    } else if (typeof val === 'number') {
      target.setScalar(val);
    } else if (Array.isArray(val) && val.length >= 4) {
      target.set(val[0], val[1], val[2], val[3]);
    } else if (val && typeof val === 'object' && typeof val.x === 'number') {
      target.set(val.x, val.y, val.z, val.w);
    }
    return;
  }

  if (target && target.isMatrix4) {
    if (val && val.isMatrix4) {
      target.copy(val);
    } else if (Array.isArray(val) && val.length >= 16) {
      target.fromArray(val);
    }
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
export const TURQUOISE_SEA_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 7. FRAGMENT SHADER (turquoise sea + foam + caustics + GI + 016)     */
/*    Injection order:                                                 */
/*    GLOBALS -> NOISE -> COLOR_UTILS -> BIOME -> 017 -> 016           */
/*    GLSL_NORMAL_QUANT and GLSL_LIGHTING are intentionally not        */
/*    injected here to avoid duplicate uniform declarations.           */
/* ------------------------------------------------------------------ */
export const TURQUOISE_SEA_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

const float TS_TAU = 6.2831853;

// Unique local global-illumination uniforms
uniform vec3  uTSSunDir;
uniform float uTSSunIntensity;
uniform vec3  uTSSkyColor;
uniform vec3  uTSGroundColor;
uniform float uTSGroundStrength;
uniform vec3  uTSFogColor;

// Pre-integrated animation phases
uniform float uTSFlowPhase;
uniform float uTSSwellPhase;
uniform float uTSChopPhase;
uniform float uTSFoamPhase;
uniform float uTSSparkPhase;
uniform float uTSSlowPhase;

// Turquoise sea palette
uniform vec3  uTSDeep;
uniform vec3  uTSMid;
uniform vec3  uTSShallow;
uniform vec3  uTSFoam;
uniform vec3  uTSGlow;
uniform vec3  uTSSparkle;
uniform vec3  uTSGIColor;

// Sea surface parameters
uniform float uTSHorizonY;
uniform float uTSAspect;
uniform float uTSWaveScale;
uniform float uTSWaveAmp;
uniform float uTSChop;
uniform float uTSFoamThreshold;
uniform float uTSFoamAmount;
uniform float uTSCausticStrength;
uniform float uTSSSSStrength;
uniform float uTSReflectStrength;
uniform float uTSSunGlintStrength;
uniform float uTSSparkleDensity;
uniform float uTSGIStrength;
uniform float uTSHazeStrength;

varying vec2 vUv;

// Local pixel quantization (avoids GLSL_NORMAL_QUANT collisions)
vec2 tsPxq(vec2 uv, float q) {
  float safeQ = max(q, 1.0);
  return floor(uv * safeQ + 0.5) / safeQ;
}

// Local hash (avoids depending on a specific h21 symbol name)
float tsHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + vec2(45.32));
  return fract(p.x * p.y);
}

void main() {
  float aspect = max(uTSAspect, 0.2);
  float ax = (vUv.x - 0.5) * aspect;

  // Sea lives below the horizon line (low->high smoothstep)
  float seaMask = 1.0 - smoothstep(uTSHorizonY - 0.004, uTSHorizonY + 0.004, vUv.y);
  if (seaMask <= 0.001) discard;

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uTSHorizonY - 0.58, uTSHorizonY, vUv.y);

  // Perspective compression toward the horizon
  float persp = mix(1.0, 7.0, hp * hp);
  vec2 sp = vec2(vUv.x * aspect * 18.0, vUv.y * 50.0) * persp;

  // Continuous, pre-integrated 2pi phases (no multiply-after-wrap pop)
  float flowPh   = mod(uTSFlowPhase, TS_TAU);
  float swellPh  = mod(uTSSwellPhase, TS_TAU);
  float chopPh   = mod(uTSChopPhase, TS_TAU);
  float foamPh   = mod(uTSFoamPhase, TS_TAU);
  float sparkPh  = mod(uTSSparkPhase, TS_TAU);
  float slowPh   = mod(uTSSlowPhase, TS_TAU);

  // Guarded wave directions
  vec2 d1 = normalize(vec2(1.0001, 0.3501));
  vec2 d2 = normalize(vec2(-0.5501, 1.0001));
  vec2 d3 = normalize(vec2(0.2501, -0.8501));

  // Directional swell / secondary swell / chop
  float a1 = dot(sp, d1) * 2.2 + flowPh;
  float a2 = dot(sp, d2) * 3.1 - swellPh;
  float a3 = dot(sp, d3) * 5.7 + chopPh;

  float w1 = sin(a1) * 0.5 + 0.5;
  float w2 = sin(a2) * 0.5 + 0.5;
  float w3 = sin(a3) * 0.5 + 0.5;

  // Static spatial detail (phase is NOT fed into fbm)
  float detailRaw = fbm(sp * 1.25 + vec2(17.3), 2);
  float detail = clamp(detailRaw * 0.5 + 0.5, 0.0, 1.0);

  float waveScale = max(uTSWaveScale, 0.1);
  float waveAmp = max(uTSWaveAmp, 0.05);
  float chop = clamp(uTSChop, 0.0, 2.0);

  float height = clamp(
    w1 * 0.38 +
    w2 * 0.28 +
    w3 * 0.18 * chop +
    detail * 0.16,
    0.0,
    1.0
  );

  // Analytic normal from wave derivatives + cheap detail slope
  float c1 = cos(a1);
  float c2 = cos(a2);
  float c3 = cos(a3);

  vec2 grad =
    d1 * (c1 * 2.2 * 0.5 * 0.38) +
    d2 * (c2 * 3.1 * 0.5 * 0.28) +
    d3 * (c3 * 5.7 * 0.5 * 0.18 * chop);

  grad += vec2(detail - 0.5) * 0.22 * chop;

  vec3 nrm = normalize(vec3(
    -grad.x * waveAmp * waveScale,
    -grad.y * waveAmp * waveScale + (detail - 0.5) * 0.10 * chop,
    1.0
  ));

  // Depth / shallowness tone
  float depthT = clamp(
    height * 0.64 +
    (1.0 - hp) * 0.22 +
    detail * 0.14,
    0.0,
    1.0
  );

  vec3 sea = paletteMix4(
    uTSDeep,
    uTSMid,
    uTSShallow,
    uTSGlow,
    depthT
  );

  // Tropical biome nudges water toward bright shallow cyan
  float biomeTropical = clamp(uBiomeW.z, 0.0, 1.0);
  sea = mix(sea, uTSShallow, biomeTropical * 0.08);

  // Slow perceptual breathing (continuous 2pi phase)
  float breathe = 0.5 + 0.5 * sin(slowPh + detail * 5.0);
  sea = mix(sea, uTSMid, breathe * 0.035);

  // Near-horizon lightening
  sea = mix(sea, uTSShallow, smoothstep(0.58, 1.0, hp) * 0.16);

  // Foam crests
  float foamThreshold = clamp(uTSFoamThreshold, 0.0, 0.95);
  float crest = smoothstep(
    foamThreshold,
    foamThreshold + 0.18,
    height
  );

  float foamNoiseRaw = fbm(sp * 2.4 + vec2(31.7), 2);
  float foamNoise = clamp(foamNoiseRaw * 0.5 + 0.5, 0.0, 1.0);

  // foamPh * 2.0 remains 2pi-periodic because foamPh is wrapped by 2pi
  float foamBreath = 0.62 + 0.38 * sin(foamPh * 2.0 + detail * 12.0);
  float foamAmount = clamp(uTSFoamAmount, 0.0, 1.5);

  float foam = clamp(
    crest *
    foamBreath *
    (0.55 + 0.45 * foamNoise) *
    foamAmount,
    0.0,
    1.0
  );

  // Foam flecks / micro-bubbles
  vec2 fp = sp * 9.0;
  vec2 fcell = floor(fp);
  float fh = tsHash21(fcell);
  vec2 ff = fract(fp) - 0.5;
  vec2 fj = vec2(tsHash21(fcell + vec2(2.3)), tsHash21(fcell + vec2(4.9))) - 0.5;
  float fdist = length(ff - fj * 0.55);

  float fleck =
    (1.0 - smoothstep(0.0, 0.12, fdist)) *
    step(0.72, fh) *
    foam;

  foam = clamp(max(foam, fleck), 0.0, 1.0);

  // Foam takes a small live sun tint without mutating uniforms
  float sunIntensitySafe = clamp(uTSSunIntensity, 0.0, 1.5);
  vec3 foamCol = mix(uTSFoam, uDirLightColor, 0.08 * sunIntensitySafe);
  sea = mix(sea, foamCol, foam * 0.82);

  // Subsurface cyan glow driven by live sun direction
  vec3 sunVec = normalize(uTSSunDir + vec3(0.0001));
  float sunDot = max(dot(nrm, sunVec), 0.0);

  float sss = pow(sunDot, 1.45)
            * (1.0 - foam * 0.55)
            * clamp(uTSSSSStrength, 0.0, 1.5);
  sss = clamp(sss, 0.0, 1.0);

  sea += uTSGlow * sss * 0.34;
  sea += uDirLightColor * sss * 0.12;

  // Caustic interference (integer/continuous phase multipliers only)
  float ca1 = sin(sp.x * 7.0 + flowPh) * sin(sp.y * 9.0 - swellPh);
  float ca2 = sin((sp.x + sp.y) * 6.0 + chopPh);
  float causticRaw = clamp(0.5 + 0.5 * (ca1 * 0.6 + ca2 * 0.4), 0.0, 1.0);

  float caustic = pow(causticRaw, 2.35)
                * (1.0 - foam * 0.70)
                * (1.0 - smoothstep(0.72, 1.0, hp))
                * clamp(uTSCausticStrength, 0.0, 1.5);
  caustic = clamp(caustic, 0.0, 1.0);

  sea += uTSGlow * caustic * 0.22;
  sea = clamp(sea, 0.0, 1.0);

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
  float dirIntensitySafe = clamp(uDirLightIntensity, 0.0, 1.5);
  float enhanceStrength = clamp(dirIntensitySafe, 0.0, 1.0);
  vec3 baseCol = perceptualBaseEnhance(sea, enhanceStrength);
  baseCol = clamp(baseCol, 0.0, 1.0);

  // 006 rim-normal exaggeration on dedicated locals (no GI clobber)
  vec3 nrmF = nrm;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  float rimPowSafe = max(uRimPower, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'rimPowSafe')}

  // 016 anime directional cel light. Scene shadow map is neutral unless opted in.
  vec3 worldPos = vec3(ax * 90.0, (1.0 - hp) * 55.0, -1.0);
  vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

  // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + cyan shallow bounce
  float hemi = nrmF.y * 0.5 + 0.5;
  vec3 gi = mix(uTSGroundColor, uTSSkyColor, hemi) * clamp(uTSGroundStrength, 0.0, 1.0);

  float shallowBounce = smoothstep(0.34, 0.96, depthT) * (1.0 - foam * 0.35);
  shallowBounce = clamp(shallowBounce, 0.0, 1.0);

  gi += uTSGIColor * shallowBounce * clamp(uTSGIStrength, 0.0, 1.5);
  gi = clamp(gi, 0.0, 1.0);
  lit += baseCol * gi;

  // Sky reflection / Fresnel
  float reflectStrength = clamp(uTSReflectStrength, 0.0, 1.0);
  float fres = pow(1.0 - max(dot(nrmF, viewF), 0.0), 3.0);
  lit = mix(lit, uTSSkyColor, fres * reflectStrength * 0.35);

  // Sun glint / specular sparkle on wave faces
  float spec = pow(sunDot, 90.0)
             * (1.0 - foam * 0.45)
             * clamp(uTSSunGlintStrength, 0.0, 1.5)
             * sunIntensitySafe;
  spec = clamp(spec, 0.0, 1.0);
  lit += uDirLightColor * spec * 0.55;

  // High-frequency water sparkle twinkle
  float sparkleDensity = clamp(uTSSparkleDensity, 0.0, 1.0);
  float sparkGate = (1.0 - foam * 0.25) * (1.0 - hp * 0.45);

  vec2 sg = sp * 22.0;
  vec2 scell = floor(sg);
  float sh = tsHash21(scell);
  vec2 sf = fract(sg) - 0.5;
  vec2 sj = vec2(tsHash21(scell + vec2(2.7)), tsHash21(scell + vec2(4.3))) - 0.5;
  float sdist = length(sf - sj * 0.52);

  float spark =
    (1.0 - smoothstep(0.0, 0.09, sdist)) *
    step(1.0 - sparkleDensity, sh) *
    sparkGate;

  float tw = 0.35 + 0.65 * (0.5 + 0.5 * sin(sparkPh + sh * 37.0));
  spark *= tw;
  spark *= sunIntensitySafe;
  spark = clamp(spark, 0.0, 1.0);

  lit += uTSSparkle * spark * 0.45;

  // Horizon haze / aerial perspective
  float haze = smoothstep(0.56, 1.0, hp) * clamp(uTSHazeStrength, 0.0, 1.0);
  lit = applyFogBlend(lit, uTSFogColor, haze);

  // Pixel-art de-banding dither + final mobile-safe clamp
  vec2 pq = tsPxq(vUv * 112.0, 56.0);
  lit += (tsHash21(pq) - 0.5) * 0.010;
  lit = clamp(lit, 0.0, 1.0);

  float alpha = clamp(seaMask, 0.0, 1.0);
  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 8. TURQUOISE SEA SHADER MANAGER                                     */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();

export class TurquoiseSeaShader {
  constructor(options = {}) {
    this.uniforms = _cloneTurquoiseSeaUniforms(TURQUOISE_SEA_UNIFORMS);
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

    // Immutable base palette snapshot prevents cumulative perceptual drift.
    this._basePalette = {
      shallow: new THREE.Vector3().copy(this.uniforms.uTSShallow.value),
      glow: new THREE.Vector3().copy(this.uniforms.uTSGlow.value),
      deep: new THREE.Vector3().copy(this.uniforms.uTSDeep.value),
    };

    this._chopBase = _clampFinite(this.uniforms.uTSChop.value, 0.0, 2.0, 0.78);
    this._chopSmooth = this._chopBase;

    this.geometry = new THREE.PlaneGeometry(2, 2);
    quantizeGeometryNormalsCPU(this.geometry, 4); // 006 CPU bake (harmless on quad)

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: TURQUOISE_SEA_VERTEX,
      fragmentShader: TURQUOISE_SEA_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
      fog: false,
      toneMapped: false,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 588; // below sea rocks (032), above base land/water layers

    this._syncAccum = 1.0;
    this._chroma = _clampFinite(options.chroma, 1.0, 2.0, 1.10);
    this._iesScale = 1.0;

    this._windSmooth = 0.0;

    this._flowPhase = 0.0;
    this._swellPhase = 0.0;
    this._chopPhase = 0.0;
    this._foamPhase = 0.0;
    this._sparkPhase = 0.0;
    this._slowPhase = 0.0;
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uTSAspect.value = _clampFinite(a, 0.2, 2.5, 0.56);
  }

  setHorizon(y) {
    this.uniforms.uTSHorizonY.value = _clampFinite(y, 0.1, 0.9, 0.48);
  }

  setWaveScale(v) {
    this.uniforms.uTSWaveScale.value = _clampFinite(v, 0.1, 3.0, 1.0);
  }

  setWaveAmp(v) {
    this.uniforms.uTSWaveAmp.value = _clampFinite(v, 0.05, 2.0, 0.72);
  }

  setChop(v) {
    this._chopBase = _clampFinite(v, 0.0, 2.0, 0.78);
    this._chopSmooth = this._chopBase;
    this.uniforms.uTSChop.value = this._chopBase;
  }

  setFoamThreshold(v) {
    this.uniforms.uTSFoamThreshold.value = _clampFinite(v, 0.0, 0.95, 0.74);
  }

  setFoamAmount(v) {
    this.uniforms.uTSFoamAmount.value = _clampFinite(v, 0.0, 1.5, 0.88);
  }

  setCausticStrength(v) {
    this.uniforms.uTSCausticStrength.value = _clampFinite(v, 0.0, 1.5, 0.82);
  }

  setSubsurfaceStrength(v) {
    this.uniforms.uTSSSSStrength.value = _clampFinite(v, 0.0, 1.5, 0.90);
  }

  setReflectStrength(v) {
    this.uniforms.uTSReflectStrength.value = _clampFinite(v, 0.0, 1.0, 0.58);
  }

  setSunGlintStrength(v) {
    this.uniforms.uTSSunGlintStrength.value = _clampFinite(v, 0.0, 1.5, 0.85);
  }

  setSparkleDensity(v) {
    this.uniforms.uTSSparkleDensity.value = _clampFinite(v, 0.0, 1.0, 0.32);
  }

  setGIStrength(v) {
    this.uniforms.uTSGIStrength.value = _clampFinite(v, 0.0, 1.5, 0.55);
  }

  setHazeStrength(v) {
    this.uniforms.uTSHazeStrength.value = _clampFinite(v, 0.0, 1.0, 0.70);
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
    } catch (err) {
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
    u.uTSSunDir.value.copy(u.uDirLightDirection.value).negate();

    // Color
    _copyColorFrom(
      u.uDirLightColor.value,
      lu.uDirLightColor && lu.uDirLightColor.value,
      1.00,
      0.98,
      0.92
    );

    // Intensity
    const rawIntensity = lu.uDirLightIntensity ? lu.uDirLightIntensity.value : 1.0;
    const intensity = _clampFinite(rawIntensity, 0.0, 4.0, 1.0) *
      _clampFinite(this._iesScale, 0.25, 4.0, 1.0);
    u.uDirLightIntensity.value = intensity;
    u.uTSSunIntensity.value = intensity;

    // Shadow tint -> ground bounce color
    _copyColorFrom(
      u.uShadowTint.value,
      lu.uShadowTint && lu.uShadowTint.value,
      0.04,
      0.16,
      0.24
    );
    u.uTSGroundColor.value.set(
      _clampFinite(u.uShadowTint.value.r * 0.58, 0.0, 1.0, 0.035),
      _clampFinite(u.uShadowTint.value.g * 0.74, 0.0, 1.0, 0.145),
      _clampFinite(u.uShadowTint.value.b * 0.82, 0.0, 1.0, 0.205)
    );

    const rawAmbientIntensity = lu.uAmbientIntensity ? lu.uAmbientIntensity.value : 0.30;
    u.uTSGroundStrength.value = _clampFinite(rawAmbientIntensity * 0.82, 0.0, 1.0, 0.30);

    // Rim color
    _copyColorFrom(
      u.uRimColor.value,
      lu.uRimColor && lu.uRimColor.value,
      0.92,
      0.98,
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
    u.uTSSkyColor.value.set(
      _clampFinite(u.uAmbientColor.value.r, 0.0, 1.0, 0.58),
      _clampFinite(u.uAmbientColor.value.g, 0.0, 1.0, 0.76),
      _clampFinite(u.uAmbientColor.value.b, 0.0, 1.0, 0.94)
    );
    u.uTSFogColor.value.set(
      _clampFinite(u.uAmbientColor.value.r * 0.78 + 0.18, 0.0, 1.0, 0.74),
      _clampFinite(u.uAmbientColor.value.g * 0.88 + 0.14, 0.0, 1.0, 0.90),
      _clampFinite(u.uAmbientColor.value.b * 0.90 + 0.10, 0.0, 1.0, 0.94)
    );

    // Cel / rim / shadow scalars
    u.uCelSteps.value = _clampFinite(lu.uCelSteps ? lu.uCelSteps.value : 4.0, 1.0, 8.0, 4.0);
    u.uRimPower.value = _clampFinite(lu.uRimPower ? lu.uRimPower.value : 3.0, 1.0, 8.0, 3.0);
    u.uRimIntensity.value = _clampFinite(lu.uRimIntensity ? lu.uRimIntensity.value : 0.45, 0.0, 1.0, 0.45);
    u.uShadowSoftness.value = _clampFinite(lu.uShadowSoftness ? lu.uShadowSoftness.value : 0.05, 0.0, 0.5, 0.05);
    u.uAmbientIntensity.value = _clampFinite(rawAmbientIntensity, 0.0, 1.0, 0.30);
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

    // Restore immutable base palette before perceptual updates.
    u.uTSShallow.value.copy(this._basePalette.shallow);
    u.uTSGlow.value.copy(this._basePalette.glow);
    u.uTSDeep.value.copy(this._basePalette.deep);

    // Perceptual vibrancy / tint updates (zero allocation in steady state)
    _applyChromaToVec3(
      u.uTSShallow.value,
      this._chroma,
      u.uTSShallow.value
    );

    _applyTintToVec3(
      u.uTSGlow.value,
      u.uDirLightColor.value,
      0.24,
      u.uTSGlow.value
    );

    _applyTintToVec3(
      u.uTSDeep.value,
      u.uShadowTint.value,
      0.28,
      u.uTSDeep.value
    );

    // Cyan bounce color follows shallow water for coherent GI
    u.uTSGIColor.value.set(
      _clampFinite(u.uTSShallow.value.x * 0.72, 0.0, 1.0, 0.17),
      _clampFinite(u.uTSShallow.value.y * 0.94, 0.0, 1.0, 0.76),
      _clampFinite(u.uTSShallow.value.z * 0.92, 0.0, 1.0, 0.70)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;

    dt = _clampFinite(dt, 0.0, 0.1, 0.0);
    elapsed = _finite(elapsed, 0.0);
    u.uTime.value = elapsed;

    const windVec = u.uWind && u.uWind.value;
    const rawWindX = windVec && Number.isFinite(windVec.x) ? windVec.x : 0.0;
    const rawWindY = windVec && Number.isFinite(windVec.y) ? windVec.y : 0.0;
    const windX = _clampFinite(rawWindX, -8.0, 8.0, 0.0);
    const windY = _clampFinite(rawWindY, -8.0, 8.0, 0.0);
    const windMag = Math.sqrt(windX * windX + windY * windY);

    // Smooth wind and chop (frame-rate independent, no per-frame allocation)
    this._windSmooth = damp(this._windSmooth, windX, 2.0, dt);
    this._windSmooth = _finite(this._windSmooth, 0.0);

    const targetChop = _clampFinite(this._chopBase + windMag * 0.38, 0.0, 2.0, this._chopBase);
    this._chopSmooth = damp(this._chopSmooth, targetChop, 1.8, dt);
    this._chopSmooth = _clampFinite(this._chopSmooth, 0.0, 2.0, this._chopBase);
    u.uTSChop.value = this._chopSmooth;

    // Integrated, 2pi-wrapped phases at exact angular frequencies.
    this._flowPhase = (this._flowPhase + dt * (0.72 + Math.abs(this._windSmooth) * 0.12)) % TWO_PI;
    if (this._flowPhase < 0.0) this._flowPhase += TWO_PI;

    this._swellPhase = (this._swellPhase + dt * 0.34) % TWO_PI;
    if (this._swellPhase < 0.0) this._swellPhase += TWO_PI;

    this._chopPhase = (this._chopPhase + dt * (1.15 + Math.abs(this._windSmooth) * 0.18)) % TWO_PI;
    if (this._chopPhase < 0.0) this._chopPhase += TWO_PI;

    this._foamPhase = (this._foamPhase + dt * (0.58 + Math.abs(this._windSmooth) * 0.10)) % TWO_PI;
    if (this._foamPhase < 0.0) this._foamPhase += TWO_PI;

    this._sparkPhase = (this._sparkPhase + dt * (1.85 + Math.abs(this._windSmooth) * 0.22)) % TWO_PI;
    if (this._sparkPhase < 0.0) this._sparkPhase += TWO_PI;

    this._slowPhase = (this._slowPhase + dt * 0.07) % TWO_PI;
    if (this._slowPhase < 0.0) this._slowPhase += TWO_PI;

    u.uTSFlowPhase.value = _finite(this._flowPhase, 0.0);
    u.uTSSwellPhase.value = _finite(this._swellPhase, 0.0);
    u.uTSChopPhase.value = _finite(this._chopPhase, 0.0);
    u.uTSFoamPhase.value = _finite(this._foamPhase, 0.0);
    u.uTSSparkPhase.value = _finite(this._sparkPhase, 0.0);
    u.uTSSlowPhase.value = _finite(this._slowPhase, 0.0);

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
export function createTurquoiseSeaShader(options = {}) {
  return new TurquoiseSeaShader(options);
}

export default {
  TurquoiseSeaShader,
  createTurquoiseSeaShader,
  TURQUOISE_SEA_UNIFORMS,
  TURQUOISE_SEA_VERTEX,
  TURQUOISE_SEA_FRAGMENT,
};
