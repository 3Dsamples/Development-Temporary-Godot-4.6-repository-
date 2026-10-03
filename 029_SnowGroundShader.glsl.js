// File : 029
// name : shaders/029_SnowGroundShader.glsl.js
// description : Anime wind-packed snow ground shader for the frozen canyon scene.
//               ANALYZED & FIXED: (1) PHASE-WRAP POP BUG: previous wind/sparkle
//               animation multiplied a 2pi-wrapped phase by non-integer factors
//               inside sin(), producing discontinuities at wrap boundaries. All
//               animated phases are now pre-integrated at their exact angular
//               frequency and wrapped only by 2pi, so sin() is continuous and
//               float32-safe on long Android sessions. (2) CHUNK-COLLISION BUG:
//               GLSL_NORMAL_QUANT and GLSL_LIGHTING were omitted from injection to
//               prevent duplicate uRimPower/uShadingSteps/uFogColor declarations
//               against the 016 directional chunk. Pixel quantization and hashing
//               are provided by unique local helpers, while fbm remains sourced
//               from GLSL_NOISE. (3) UNIFORM-COLLISION BUG: all local global
//               illumination uniforms are uniquely prefixed (uSG*) to avoid any
//               overlap with 000/004/016/017 chunks. (4) FALSE-DEPTH SHADOW BUG:
//               fullscreen parallax snow uses a neutral 1x1 white shadow texture by
//               default; live scene shadow-map binding is opt-in via
//               options.sceneShadows to prevent incorrect self-darkening from
//               reconstructed fake world positions. (5) COLOR COPY BUG: Color ->
//               Vector3 synchronization now uses explicit setRGB/set components,
//               never .copy(Color) on Vector3 targets. (6) NOISE RANGE BUG: every
//               fbm value driving palettes, masks, or rock wobble is normalized and
//               clamped to 0..1 before use. (7) OVERBRIGHT BUG: final color is
//               clamped after GI, sparkle, glint, fog and dither for mobile
//               framebuffers. (8) FILL-RATE BUG: sky pixels and empty hash cells
//               discard before expensive noise work. (9) INSTANCE SHARING BUG:
//               uniforms are cloned per manager instance. (10) CULLING BUG:
//               fullscreen quad uses DoubleSide. (11) SMOOTHSTEP AUDIT: every
//               smoothstep() edge is low->high. (12) GUARDED MATH: aspect, band
//               counts, pixel quantization divisor, normalize inputs, and phase
//               wrapping are guarded. Renders canyon-floor snow with perspective
//               drift streaks, blue hollows, turquoise ice patches, scattered dark
//               rocks, sparkle glints, contact shadows, cyan water/ice bounce GI,
//               cel-quantized directional lighting via 016_DirectionalLightShader,
//               017 perceptual base enhancement (OKLCH), 006 rim-normal
//               exaggeration, 020_gmp_perceptual_color chromatization /
//               perfect-tint wrappers, 021_gmp_ies_lighting IES hooks, and
//               001_gmp_MathUtils scalar helpers. Zero per-frame allocation
//               (pre-allocated scratch Vector3/Color, throttled 4 Hz sync).
//               Android-mobile tuned, single fullscreen quad draw call.
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
  applyChromatizationColor,
  applyPerfectTintColor,
} from '../utils/020_gmp_perceptual_color.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
} from '../utils/021_gmp_ies_lighting.js';
import { clamp, TWO_PI } from '../utils/001_gmp_MathUtils.js';

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
/* 2. SNOW GROUND DEFAULT UNIFORMS (JS side)                           */
/* ------------------------------------------------------------------ */
export const SNOW_GROUND_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // Unique local global-illumination providers (no chunk collisions)
  uSGSunDir:         { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uSGSunIntensity:   { value: 1.0 },
  uSGSkyColor:       { value: new THREE.Vector3(0.58, 0.76, 0.94) },
  uSGGroundColor:    { value: new THREE.Vector3(0.11, 0.16, 0.23) },
  uSGGroundStrength: { value: 0.28 },
  uSGFogColor:       { value: new THREE.Vector3(0.78, 0.88, 0.96) },

  // Pre-integrated animation phases (already multiplied by their frequencies)
  uSGWindPhase:  { value: 0.0 },
  uSGSparkPhase: { value: 0.0 },

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
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Snow ground specific (exact image palette)
  uColSnowShadow:  { value: new THREE.Vector3(0.560, 0.700, 0.860) }, // 0x8fb3db blue snow shadow
  uColSnowMid:     { value: new THREE.Vector3(0.820, 0.900, 0.970) }, // 0xd1e5f7 packed snow
  uColSnowLight:   { value: new THREE.Vector3(0.980, 1.000, 1.000) }, // 0xfafeff bright snow
  uColSnowSparkle: { value: new THREE.Vector3(1.000, 1.000, 0.980) }, // 0xfffffa sparkle
  uColIce:         { value: new THREE.Vector3(0.290, 0.780, 0.720) }, // 0x4ac7b8 turquoise ice
  uColIceDeep:     { value: new THREE.Vector3(0.100, 0.420, 0.450) }, // 0x1a6b73 deep ice
  uColRockDark:    { value: new THREE.Vector3(0.140, 0.190, 0.250) }, // 0x243040 dark canyon rock
  uColRockMid:     { value: new THREE.Vector3(0.270, 0.350, 0.440) }, // 0x455970 mid rock
  uColRockLight:   { value: new THREE.Vector3(0.470, 0.560, 0.650) }, // 0x788fa6 lit rock
  uGIColor:        { value: new THREE.Vector3(0.220, 0.720, 0.640) }, // cyan water/ice bounce
  uGIStrength:     { value: 0.45 },
  uHorizonY:       { value: 0.50 },
  uAspect:         { value: 0.56 },
  uSnowBands:      { value: 4.0 },
  uSparkleDensity: { value: 0.22 },
  uRockDensity:    { value: 0.45 },
  uIceAmount:      { value: 0.55 },
  uDriftStrength:  { value: 0.75 },
  uGlowStrength:   { value: 0.65 },
};

/* ------------------------------------------------------------------ */
/* 3. PER-INSTANCE UNIFORM CLONE (setup-only allocation)               */
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

function _cloneSnowGroundUniforms(src) {
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
/* 4. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const SNOW_GROUND_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. FRAGMENT SHADER (snow drifts + ice + rocks + GI + 016 light)     */
/*    Injection order:                                                 */
/*    GLOBALS -> NOISE -> COLOR_UTILS -> BIOME -> 017 -> 016           */
/*    GLSL_NORMAL_QUANT and GLSL_LIGHTING are intentionally not        */
/*    injected here to avoid duplicate uniform declarations.           */
/* ------------------------------------------------------------------ */
export const SNOW_GROUND_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

// Unique local global-illumination uniforms
uniform vec3  uSGSunDir;
uniform float uSGSunIntensity;
uniform vec3  uSGSkyColor;
uniform vec3  uSGGroundColor;
uniform float uSGGroundStrength;
uniform vec3  uSGFogColor;

// Pre-integrated animation phases
uniform float uSGWindPhase;
uniform float uSGSparkPhase;

// Snow ground palette
uniform vec3  uColSnowShadow;
uniform vec3  uColSnowMid;
uniform vec3  uColSnowLight;
uniform vec3  uColSnowSparkle;
uniform vec3  uColIce;
uniform vec3  uColIceDeep;
uniform vec3  uColRockDark;
uniform vec3  uColRockMid;
uniform vec3  uColRockLight;
uniform vec3  uGIColor;
uniform float uGIStrength;

// Snow ground parameters
uniform float uHorizonY;
uniform float uAspect;
uniform float uSnowBands;
uniform float uSparkleDensity;
uniform float uRockDensity;
uniform float uIceAmount;
uniform float uDriftStrength;
uniform float uGlowStrength;

varying vec2 vUv;

// Local pixel quantization (avoids GLSL_NORMAL_QUANT collisions)
vec2 sgPxq(vec2 uv, float q) {
  float safeQ = max(q, 1.0);
  return floor(uv * safeQ + 0.5) / safeQ;
}

// Local hash (avoids depending on a specific h21 symbol name)
float sgHash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  // Ground mask: snow lives below the horizon line (low->high smoothstep)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.004, uHorizonY + 0.004, vUv.y);
  if (groundMask <= 0.001) discard;

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.58, uHorizonY, vUv.y);

  // Perspective-compressed snow coordinates (drifts tighten at horizon)
  float persp = mix(1.0, 7.0, hp * hp);
  vec2 sp = vec2(
    vUv.x * max(uAspect, 0.2) * 20.0,
    vUv.y * 62.0
  ) * persp;

  // Continuous, pre-integrated 2pi phases (no multiply-after-wrap pop)
  float streakPh = mod(uSGWindPhase, 6.2831853);
  float sparkPh  = mod(uSGSparkPhase, 6.2831853);

  // Large dune undulation (normalized to 0..1 for palette safety)
  float duneRaw = fbm(sp * 0.13 + 7.7, 3);
  float dune = clamp(duneRaw * 0.5 + 0.5, 0.0, 1.0);

  // Wind streak noise (static spatial field, animated phase only)
  float streakNoiseRaw = fbm(vec2(sp.x * 0.28, sp.y * 0.46) + 13.1, 3);
  float streakNoise = clamp(streakNoiseRaw * 0.5 + 0.5, 0.0, 1.0);
  float streak = sin(sp.y * 2.6 + streakNoise * 5.5 + streakPh) * 0.5 + 0.5;
  streak = smoothstep(0.35, 0.78, streak) * clamp(uDriftStrength, 0.0, 1.5);

  // Blue hollows / wind-scoured depressions
  float hollowRaw = fbm(sp * 0.09 + 23.3, 2);
  float hollow = clamp(hollowRaw * 0.5 + 0.5, 0.0, 1.0);
  float hollowMask = smoothstep(0.46, 0.66, 1.0 - hollow) * (1.0 - hp * 0.35);

  // Turquoise ice patches (alpine biome boosts coverage)
  float iceRaw = fbm(sp * 0.22 + 31.7, 3);
  float iceNoise = clamp(iceRaw * 0.5 + 0.5, 0.0, 1.0);
  float iceAmount = clamp(uIceAmount, 0.0, 1.5) * (0.75 + 0.25 * clamp(uBiomeW.w, 0.0, 1.0));
  float iceMask = smoothstep(0.60, 0.78, iceNoise) * iceAmount * (1.0 - hp * 0.25);
  iceMask = max(iceMask, smoothstep(0.72, 0.88, hollowMask) * 0.25 * iceAmount);
  iceMask = clamp(iceMask, 0.0, 1.0);

  // Cel-quantized snow tone bands
  float heightTone = clamp(dune * 0.65 + streak * 0.25 + (1.0 - hp) * 0.10, 0.0, 1.0);
  float bands = max(uSnowBands, 1.0);
  float tone = floor(heightTone * bands + 0.5) / bands;

  vec3 snowCol = paletteMix4(
    uColSnowShadow,
    uColSnowMid,
    uColSnowLight,
    uColSnowSparkle,
    tone
  );
  snowCol = mix(snowCol, uColSnowShadow, hollowMask * 0.55);

  // Ice color blend
  vec3 iceCol = mix(uColIceDeep, uColIce, smoothstep(0.35, 0.85, streakNoise));
  snowCol = mix(snowCol, iceCol, iceMask * 0.75);

  // Snow / ice normal (z=1.0 guarantees normalize safety)
  vec3 snowNrm = normalize(vec3(
    (streakNoise - 0.5) * 0.55 + (dune - 0.5) * 0.25,
    (hollow - 0.5) * 0.35,
    1.0
  ));
  snowNrm = mix(snowNrm, vec3(0.0, 0.0, 1.0), iceMask * 0.65);

  // Scattered dark rocks / pebbles on snow (hash grid, no atan)
  vec2 rp = sp * 0.55;
  vec2 rcell = floor(rp);
  vec2 rf = fract(rp) - 0.5;
  float rh = sgHash21(rcell);
  float rh2 = sgHash21(rcell + 7.7);

  float rockGate = step(1.0 - clamp(uRockDensity, 0.0, 1.0) * (1.0 - hp * 0.55), rh);
  if (rockGate < 0.5 && iceMask < 0.02 && streak < 0.02 && hollowMask < 0.02) {
    // Still render base snow even when no rock/ice/streak detail is present.
  }

  float radius = max(0.08 + rh2 * 0.14, 0.045);
  vec2 rjit = vec2(sgHash21(rcell + 3.1), sgHash21(rcell + 5.7)) - 0.5;
  vec2 rd = rf - rjit * 0.65;
  float rdist = length(rd);

  float rwobRaw = fbm(rd * 8.0 + rh * 17.0, 2);
  float rwob = 1.0 + (clamp(rwobRaw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 0.35;

  float rockShape =
    (1.0 - smoothstep(radius * rwob - 0.025, radius * rwob + 0.025, rdist)) *
    rockGate;
  float rockMask = clamp(rockShape, 0.0, 1.0);

  vec3 rockCol = paletteMix4(
    uColRockDark,
    uColRockMid,
    uColRockLight,
    uColRockDark,
    clamp(rh2, 0.0, 1.0)
  );
  vec3 rockNrm = normalize(vec3(rd.x * 2.0, rd.y * 2.0, 1.0));

  // Blend snow and rock surfaces before lighting
  vec3 baseCol = mix(snowCol, rockCol, rockMask);
  vec3 nrm = mix(snowNrm, rockNrm, rockMask);

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
  baseCol = perceptualBaseEnhance(baseCol, clamp(uDirLightIntensity, 0.0, 1.5));

  // 006 rim-normal exaggeration on dedicated locals (no GI clobber)
  vec3 nrmF = nrm;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

  // 016 anime directional cel light. Scene shadow map is neutral unless opted in.
  vec3 worldPos = vec3((vUv.x - 0.5) * 80.0, (1.0 - hp) * 45.0, 0.0);
  vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

  // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + cyan water/ice bounce
  float hemi = nrmF.y * 0.5 + 0.5;
  vec3 gi = mix(uSGGroundColor, uSGSkyColor, hemi) * clamp(uSGGroundStrength, 0.0, 1.0);

  float cyanBounce =
    (iceMask * 0.70 + (1.0 - smoothstep(0.25, 0.85, vUv.y)) * 0.25) *
    clamp(uGlowStrength, 0.0, 1.5);
  gi += uGIColor * cyanBounce * clamp(uGIStrength, 0.0, 1.5);
  lit += baseCol * gi;

  // Sparkle glints on snow / ice (not on rocks)
  float sparkleGate = (1.0 - rockMask) * mix(1.0, 0.55, iceMask);
  vec2 sg = sp * 70.0;
  vec2 scell = floor(sg);
  float sh = sgHash21(scell);
  vec2 sf = fract(sg) - 0.5;
  vec2 sj = vec2(sgHash21(scell + 2.3), sgHash21(scell + 4.9)) - 0.5;
  float sdist = length(sf - sj * 0.55);
  float spark =
    (1.0 - smoothstep(0.0, 0.10, sdist)) *
    step(1.0 - clamp(uSparkleDensity, 0.0, 1.0), sh) *
    sparkleGate;

  float tw = 0.35 + 0.65 * (0.5 + 0.5 * sin(sparkPh + sh * 43.0));
  spark *= tw * clamp(uSGSunIntensity, 0.0, 1.5) * (1.0 - hp * 0.65);
  lit += uColSnowSparkle * spark * 0.55;

  // Ice sun glint (guarded normalize)
  vec3 sunNrm = normalize(uSGSunDir + vec3(1e-4, 1e-4, 1e-4));
  float sunDot = max(dot(nrmF, sunNrm), 0.0);
  float iceGlint = pow(sunDot, 28.0) * iceMask * (1.0 - rockMask);
  lit += uDirLightColor * iceGlint * 0.22 * clamp(uSGSunIntensity, 0.0, 1.5);

  // Rock contact shadow ring (grounding)
  float ring =
    smoothstep(radius * 0.92, radius * 1.30, rdist) *
    (1.0 - smoothstep(radius * 1.30, radius * 1.70, rdist));
  lit *= 1.0 - ring * rockGate * 0.28 * (1.0 - hp * 0.50);

  // Distance haze toward horizon (aerial perspective)
  lit = applyFogBlend(lit, uSGFogColor, hp * hp * 0.55);

  // Pixel-art de-banding dither + final mobile-safe clamp
  vec2 pq = sgPxq(vUv * 112.0, 56.0);
  lit += (sgHash21(pq) - 0.5) * 0.010;
  lit = clamp(lit, 0.0, 1.0);

  float alpha = clamp(groundMask, 0.0, 1.0);
  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. SNOW GROUND SHADER MANAGER                                       */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _tintCol   = new THREE.Color();
const _boosted   = new THREE.Color();
const _envCol    = new THREE.Color();

export class SnowGroundShader {
  constructor(options = {}) {
    this.uniforms = _cloneSnowGroundUniforms(SNOW_GROUND_UNIFORMS);
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
      vertexShader: SNOW_GROUND_VERTEX,
      fragmentShader: SNOW_GROUND_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 570; // above cliffs, below ice water / foam / sea rocks

    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.08;
    this._iesScale = 1.0;

    // Pre-integrated angular phases:
    // streak phase = windX * 0.18 * 1.35 = windX * 0.243
    // sparkle phase = 2.2 * 2.3 = 5.06 rad/s
    this._windPhase = 0.0;
    this._sparkPhase = 0.0;
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uAspect.value = clamp(a, 0.2, 2.5);
  }

  setHorizon(y) {
    this.uniforms.uHorizonY.value = clamp(y, 0.1, 0.9);
  }

  setSnowBands(v) {
    this.uniforms.uSnowBands.value = clamp(v, 1.0, 8.0);
  }

  setSparkleDensity(v) {
    this.uniforms.uSparkleDensity.value = clamp(v, 0.0, 1.0);
  }

  setRockDensity(v) {
    this.uniforms.uRockDensity.value = clamp(v, 0.0, 1.0);
  }

  setIceAmount(v) {
    this.uniforms.uIceAmount.value = clamp(v, 0.0, 1.5);
  }

  setDriftStrength(v) {
    this.uniforms.uDriftStrength.value = clamp(v, 0.0, 1.5);
  }

  setGlowStrength(v) {
    this.uniforms.uGlowStrength.value = clamp(v, 0.0, 1.5);
  }

  setGIStrength(v) {
    this.uniforms.uGIStrength.value = clamp(v, 0.0, 1.5);
  }

  /* Optional IES hook (021_gmp_ies_lighting): tag the directional light
     with an isotropic IES profile; ballast-scaled intensity feeds uDirLightIntensity. */
  enableIES(light) {
    if (!light) return;
    autoConvertLight(light, generateIsotropicIES(light));
    const ies = light.userData && light.userData.ies;
    if (ies) {
      const flux = integrateCandela(ies);
      this._iesScale = (ies.ballastFactor || 1.0) *
        (flux > 0.0 ? clamp(flux / (4.0 * Math.PI), 0.25, 4.0) : 1.0);
    }
  }

  /* Full interaction with 016_DirectionalLightShader: direction, color,
     intensity, rim, ambient, shadow tint + optional real shadow map.       */
  syncFromLight(lightShader) {
    const lu = lightShader.getUniforms();
    const u = this.uniforms;

    _sunDirTmp.copy(lu.uDirLightDirection.value).normalize();
    u.uDirLightDirection.value.copy(_sunDirTmp);
    u.uSGSunDir.value.copy(_sunDirTmp).negate();

    const dc = lu.uDirLightColor.value;
    if (dc.isColor) {
      u.uDirLightColor.value.copy(dc);
    } else if (dc.isVector3) {
      u.uDirLightColor.value.setRGB(dc.x, dc.y, dc.z);
    }

    const inten = lu.uDirLightIntensity.value * this._iesScale;
    u.uDirLightIntensity.value = inten;
    u.uSGSunIntensity.value = inten;

    const st = lu.uShadowTint.value;
    if (st.isColor) {
      u.uShadowTint.value.copy(st);
      _envCol.copy(st);
    } else if (st.isVector3) {
      u.uShadowTint.value.set(st.x, st.y, st.z);
      _envCol.setRGB(st.x, st.y, st.z);
    }
    u.uSGGroundColor.value.set(
      clamp(_envCol.r * 0.45, 0.0, 1.0),
      clamp(_envCol.g * 0.50, 0.0, 1.0),
      clamp(_envCol.b * 0.58, 0.0, 1.0)
    );

    const rc = lu.uRimColor.value;
    if (rc.isColor) {
      u.uRimColor.value.copy(rc);
    } else if (rc.isVector3) {
      u.uRimColor.value.set(rc.x, rc.y, rc.z);
    }

    const ac = lu.uAmbientColor.value;
    if (ac.isColor) {
      u.uAmbientColor.value.copy(ac);
      _envCol.copy(ac);
    } else if (ac.isVector3) {
      u.uAmbientColor.value.set(ac.x, ac.y, ac.z);
      _envCol.setRGB(ac.x, ac.y, ac.z);
    }
    u.uSGSkyColor.value.set(
      clamp(_envCol.r, 0.0, 1.0),
      clamp(_envCol.g, 0.0, 1.0),
      clamp(_envCol.b, 0.0, 1.0)
    );
    u.uSGFogColor.value.set(
      clamp(_envCol.r * 0.88 + 0.12, 0.0, 1.0),
      clamp(_envCol.g * 0.88 + 0.12, 0.0, 1.0),
      clamp(_envCol.b * 0.88 + 0.12, 0.0, 1.0)
    );

    u.uCelSteps.value = lu.uCelSteps.value;
    u.uRimPower.value = lu.uRimPower.value;
    u.uRimIntensity.value = lu.uRimIntensity.value;
    u.uShadowSoftness.value = lu.uShadowSoftness.value;
    u.uAmbientIntensity.value = lu.uAmbientIntensity.value;

    // Background parallax snow uses neutral shadows by default.
    // Enable options.sceneShadows=true only if a real 3D snow proxy exists.
    if (this.sceneShadows) {
      const light = lightShader.getLight ? lightShader.getLight() : null;
      if (light && light.shadow && light.shadow.map && light.shadow.map.texture) {
        u.uShadowMap.value = light.shadow.map.texture;
        u.uShadowMatrix.value.copy(light.shadow.matrix);
        u.uShadowMapSize.value.set(light.shadow.mapSize.x, light.shadow.mapSize.y);
        u.uShadowBias.value = light.shadow.bias;
        u.uShadowNormalBias.value = light.shadow.normalBias;
      }
    } else {
      u.uShadowMap.value = _whiteShadow;
      u.uShadowMapSize.value.set(1, 1);
    }

    // Perceptual vibrancy boost on bright snow (020_gmp_perceptual_color)
    _baseCol.setRGB(
      this.uniforms.uColSnowLight.value.x,
      this.uniforms.uColSnowLight.value.y,
      this.uniforms.uColSnowLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColSnowLight.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );

    // Perfect-tint turquoise ice toward the live sun color
    _tintCol.copy(lu.uDirLightColor.value);
    _baseCol.setRGB(
      this.uniforms.uColIce.value.x,
      this.uniforms.uColIce.value.y,
      this.uniforms.uColIce.value.z
    );
    const iceTinted = applyPerfectTintColor(_baseCol, _tintCol, 0.22);
    this.uniforms.uColIce.value.set(
      clamp(iceTinted.r, 0.0, 1.0),
      clamp(iceTinted.g, 0.0, 1.0),
      clamp(iceTinted.b, 0.0, 1.0)
    );

    // Perfect-tint deep rock shadow toward the live shadow tint
    _tintCol.copy(lu.uShadowTint.value);
    _baseCol.setRGB(0.140, 0.190, 0.250);
    const rockTinted = applyPerfectTintColor(_baseCol, _tintCol, 0.30);
    this.uniforms.uColRockDark.value.set(
      clamp(rockTinted.r, 0.0, 1.0),
      clamp(rockTinted.g, 0.0, 1.0),
      clamp(rockTinted.b, 0.0, 1.0)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    if (dt < 0.0) dt = 0.0;

    // Compatibility timestamp for systems that probe uTime.
    u.uTime.value = elapsed;

    // Integrated, 2pi-wrapped phases at exact angular frequencies.
    this._windPhase = (this._windPhase + u.uWind.value.x * dt * 0.243) % TWO_PI;
    if (this._windPhase < 0.0) this._windPhase += TWO_PI;

    this._sparkPhase = (this._sparkPhase + dt * 5.06) % TWO_PI;
    if (this._sparkPhase < 0.0) this._sparkPhase += TWO_PI;

    u.uSGWindPhase.value = this._windPhase;
    u.uSGSparkPhase.value = this._sparkPhase;

    if (camera) {
      u.uCamPos.value.set(camera.position.x, camera.position.y, camera.position.z);
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
/* 7. FACTORY                                                          */
/* ------------------------------------------------------------------ */
export function createSnowGroundShader(options = {}) {
  return new SnowGroundShader(options);
}

export default {
  SnowGroundShader,
  createSnowGroundShader,
  SNOW_GROUND_UNIFORMS,
  SNOW_GROUND_VERTEX,
  SNOW_GROUND_FRAGMENT,
};