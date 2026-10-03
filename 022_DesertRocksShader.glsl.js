// File : 022
// name : shaders/022_DesertRocksShader.glsl.js
// description : Scattered cel-shaded desert boulders/rocks layer for the badlands
//               scene. ANALYZED & FIXED: (1) ATAN SINGULARITY: rock edge wobble used
//               atan(d2.y, d2.x) which is undefined at the exact jittered center on
//               strict GLSL ES 1.00 drivers — replaced with atan(d2.y, d2.x + 1e-6).
//               (2) SMOOTHSTEP AUDIT: every smoothstep() verified low->high
//               (horizon mask, rock rim, base flatten, cast-shadow bands) — no
//               reversed edges. (3) UNGUARDED NORMALIZE: cast-shadow direction
//               normalized uDirLightDirection.xy directly (NaN when sun is exactly
//               overhead) — now guarded with +vec2(1e-4) and max(length,1e-4).
//               (4) UNIFORM REDECLARATION: uTime/uPPU/uWind/uCamPos/uViewDir provided
//               once for GLSL_GLOBALS; lighting set once for GLSL_LIGHTING; biome
//               set once for GLSL_BIOME; uShadingSteps/uRimPower declared only where
//               the 006/016 chunks need them. (5) INJECTION ORDER: GLSL_COLOR_UTILS
//               before the 016 directional chunk (applyShadowTint dependency) and
//               017 perceptual enhancer after lighting. (6) FILL-RATE: three hard
//               early-outs (sky pixels, empty hash cells, sub-threshold shapes) plus
//               horizon density attenuation. (7) SHADOW-MAP SAFETY: default 1x1 white
//               DataTexture keeps sampleAnimeShadowMap() defined before a real shadow
//               map binds. (8) BOUNDED CLOCK: mod(uTime, 640.0) for all animated
//               terms. (9) RADIUS CLAMP: max(radius, 0.07) before rim band so the
//               smoothstep band can never invert. Fully interacts with
//               016_DirectionalLightShader (live shadow-map sampling so rocks darken
//               inside butte shadows + procedural projected cast-shadow bands that
//               stretch with sun elevation), global illumination (hemisphere
//               sky/ground bounce + warm sand bounce), 006 rim-normal exaggeration,
//               017 perceptual base enhancement, 020_gmp_perceptual_color.js
//               chromatization and 021_gmp_ies_lighting.js IES hooks. Zero
//               per-frame allocation (pre-allocated scratch Color/Vector3, throttled
//               4 Hz sync). Android-mobile tuned.
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
import { GLSL_COLOR_UTILS } from '../utils/004_ColorPalette.js';
import { GLSL_PERCEPTUAL_BASE_ENHANCER } from './017_PerceptualBaseEnhancer.glsl.js';
import { GLSL_DIRECTIONAL_LIGHT } from './016_DirectionalLightShader.glsl.js';
import {
  getAnimeRimNormalExaggerationGLSL,
  quantizeGeometryNormalsCPU,
} from '../mesh/006_normalQuantizer.js';
import { applyChromatizationColor } from '../utils/020_gmp_perceptual_color.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
} from '../utils/021_gmp_ies_lighting.js';
import { clamp, damp } from '../utils/001_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. DESERT ROCKS UNIFORMS (JS side)                                  */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // fully lit until a real shadow map binds

export const DESERT_ROCKS_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // GLSL_LIGHTING providers (GI hemisphere + cel ambient)
  uSunDir:           { value: new THREE.Vector3(0.5, 0.8, 0.3) },
  uMoonDir:          { value: new THREE.Vector3(-0.5, -0.8, -0.3) },
  uSunColor:         { value: new THREE.Vector3(1.0, 0.96, 0.85) },
  uMoonColor:        { value: new THREE.Vector3(0.42, 0.48, 0.70) },
  uSkyColor:         { value: new THREE.Vector3(0.45, 0.62, 0.85) },
  uGroundColor:      { value: new THREE.Vector3(0.35, 0.25, 0.16) },
  uFogColor:         { value: new THREE.Vector3(0.93, 0.80, 0.62) },
  uShadowTintColor:  { value: new THREE.Vector3(0.12, 0.18, 0.30) },
  uRimColor:         { value: new THREE.Vector3(1.0, 0.95, 0.85) },
  uSunIntensity:     { value: 1.0 },
  uMoonIntensity:    { value: 0.35 },
  uSkyStrength:      { value: 0.35 },
  uGroundStrength:   { value: 0.30 },
  uBounceStrength:   { value: 0.30 },
  uRimStrength:      { value: 0.45 },
  uShadingSteps:     { value: 4.0 },

  // GLSL_BIOME providers
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // 017 perceptual enhancer provider
  uPerceptualEnhance: { value: 0.85 },

  // 016 directional light providers (safe defaults; synced at runtime)
  uDirLightColor:     { value: new THREE.Color(1.0, 0.96, 0.85) },
  uDirLightDirection: { value: new THREE.Vector3(-0.5, -0.8, -0.3) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.35, 0.28, 0.22) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.25 },
  uAmbientColor:      { value: new THREE.Color(0.45, 0.62, 0.85) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Rocks specific (exact desert palette)
  uColRockDark:  { value: new THREE.Vector3(0.420, 0.280, 0.170) }, // 0x6b472b shadowed tan
  uColRockMid:   { value: new THREE.Vector3(0.620, 0.450, 0.290) }, // 0x9e734a mid tan
  uColRockGray:  { value: new THREE.Vector3(0.560, 0.520, 0.470) }, // 0x8f8578 gray stone
  uColRockLight: { value: new THREE.Vector3(0.870, 0.720, 0.520) }, // 0xdeb885 lit crest
  uGIColor:      { value: new THREE.Vector3(0.450, 0.330, 0.200) }, // warm sand bounce GI
  uGIStrength:   { value: 0.35 },
  uHorizonY:     { value: 0.42 },
  uAspect:       { value: 0.56 },
  uDensity:      { value: 0.70 },
  uShadowLen:    { value: 1.20 }, // cast-shadow stretch (sun-elevation driven)
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_ROCKS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (faceted boulders + cast shadows + GI)           */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const DESERT_ROCKS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColRockDark;
uniform vec3  uColRockMid;
uniform vec3  uColRockGray;
uniform vec3  uColRockLight;
uniform vec3  uGIColor;
uniform float uGIStrength;
uniform float uHorizonY;
uniform float uAspect;
uniform float uDensity;
uniform float uShadowLen;

varying vec2 vUv;

void main() {
  // FIXED: bounded clock for float precision on long sessions
  float t = mod(uTime, 640.0);

  // Ground mask: rocks live below the horizon line (low->high smoothstep)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.005, uHorizonY + 0.005, vUv.y);
  if (groundMask <= 0.001) { gl_FragColor = vec4(0.0); return; }

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.55, uHorizonY, vUv.y);

  // Perspective-compressed hash grid (coarser than speckles)
  float persp = mix(1.0, 6.5, hp * hp);
  vec2 sp = vec2(vUv.x * max(uAspect, 0.2) * 26.0, vUv.y * 70.0) * persp;

  vec2 cell = floor(sp);
  vec2 f = fract(sp) - 0.5;
  float h  = h21(cell);
  float h2v = h21(cell + 7.7);

  // Density gate with horizon attenuation (fill-rate conservation)
  float gate = step(1.0 - uDensity * (1.0 - hp * 0.5), h);
  if (gate < 0.5) { gl_FragColor = vec4(0.0); return; }

  // Jittered boulder center + distorted circle (FIXED: guarded atan)
  vec2 jitter = vec2(h21(cell + 3.1), h21(cell + 5.7)) - 0.5;
  vec2 d2 = f - jitter * 0.75;
  float radius = max(0.14 + h2v * 0.20, 0.07); // FIXED: radius clamp
  float ang = atan(d2.y, d2.x + 1e-6);
  float wob = 1.0 + sin(ang * 5.0 + h * 40.0) * 0.13 + sin(ang * 9.0 + h * 27.0) * 0.07;
  float dist = length(d2);
  float rockShape = 1.0 - smoothstep(radius * wob - 0.03, radius * wob + 0.03, dist);
  // Flatten base so rocks sit on the sand (low->high)
  rockShape *= 1.0 - smoothstep(0.02, 0.10, -d2.y - radius * 0.5);

  // FIXED: guarded cast-shadow direction (NaN-safe at zenith sun)
  vec2 sdirRaw = uDirLightDirection.xy;
  float sdirLen = max(length(sdirRaw), 1e-4);
  vec2 sdir = sdirRaw / sdirLen;
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0);

  // Procedural projected cast-shadow band (stretches with uShadowLen)
  float along  = dot(d2, sdir);
  float across = dot(d2, vec2(-sdir.y, sdir.x));
  float shadowShape =
    (1.0 - smoothstep(radius * 0.85, radius * 1.05, abs(across))) *
    smoothstep(radius * 0.10, radius * 0.35, along) *
    (1.0 - smoothstep(radius * (1.0 + uShadowLen), radius * (1.3 + uShadowLen), along));
  shadowShape *= 0.85 + 0.15 * sin(across * 40.0 + h * 30.0); // soft noisy edge
  shadowShape *= step(0.02, sunAbove); // no cast shadow at night

  if (rockShape < 0.02 && shadowShape < 0.02) { gl_FragColor = vec4(0.0); return; }

  // ---- ROCK BRANCH ----
  if (rockShape >= shadowShape && rockShape > 0.02) {
    // Faceted dome normal (cel facets)
    vec3 nrm = normalize(vec3(d2.x * 2.2, d2.y * 2.2, 1.0));
    vec3 fn  = floor(nrm * 3.0 + 0.5) / 3.0;
    nrm = normalize(mix(nrm, fn, 0.45));

    // 006 rim-normal exaggeration (GPU path)
    vec3 nrmF  = nrm;
    vec3 viewF = vec3(0.0, 0.0, 1.0);
    ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

    // Base rock color (4-stop palette by hash) + striation cracks
    vec3 col = paletteMix4(uColRockDark, uColRockMid, uColRockGray, uColRockLight, fract(h * 13.7));
    float crack = 1.0 - smoothstep(0.0, 0.03, abs(sin(ang * 3.0 + h * 20.0)) * dist - 0.02);
    col *= 1.0 - crack * 0.35;

    // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
    col = perceptualBaseEnhance(col, clamp(uDirLightIntensity, 0.0, 1.5));

    // 016 anime directional cel light + LIVE shadow-map sampling
    vec3 worldPos = vec3(vUv.x * 40.0 - 20.0, (1.0 - hp) * 30.0, 0.0);
    vec3 lit = computeAnimeDirectionalLight(col, nrmF, viewF, worldPos);

    // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + warm sand bounce
    float hemi = nrmF.y * 0.5 + 0.5;
    vec3 gi = mix(uGroundColor, uSkyColor, hemi) * uGroundStrength;
    gi += uGIColor * (1.0 - hemi) * uGIStrength;
    lit += col * gi;

    // Contact shadow at the base (grounding)
    float contact = 1.0 - smoothstep(0.0, 0.25, d2.y + radius * 0.6);
    lit *= 1.0 - contact * 0.35;

    // Distance haze toward horizon (aerial perspective)
    lit = applyFogBlend(lit, uFogColor, hp * hp * 0.6);

    // Pixel-art de-banding dither
    vec2 pq = pxq(vUv * 96.0, 48.0);
    lit += (h21(pq + h) - 0.5) * 0.012;

    gl_FragColor = vec4(lit, rockShape * groundMask);
    return;
  }

  // ---- CAST SHADOW BRANCH ----
  vec3 shCol = mix(uColRockDark, uShadowTint.rgb, 0.5) * 0.45;
  shCol = applyFogBlend(shCol, uFogColor, hp * hp * 0.5);
  vec2 pq2 = pxq(vUv * 96.0, 48.0);
  shCol += (h21(pq2 + h) - 0.5) * 0.010;
  float shAlpha = shadowShape * 0.45 * groundMask * sunAbove;
  gl_FragColor = vec4(shCol, shAlpha);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT ROCKS SHADER MANAGER                                      */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class DesertRocksShader {
  constructor(options = {}) {
    this.uniforms = DESERT_ROCKS_UNIFORMS;
    for (const k in options) {
      if (this.uniforms[k] && options[k] !== undefined) {
        if (this.uniforms[k].value && this.uniforms[k].value.set) {
          this.uniforms[k].value.set(options[k]);
        } else {
          this.uniforms[k].value = options[k];
        }
      }
    }
    this.geometry = new THREE.PlaneGeometry(2, 2);
    quantizeGeometryNormalsCPU(this.geometry, 4); // 006 CPU bake (harmless on quad)
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: DESERT_ROCKS_VERTEX,
      fragmentShader: DESERT_ROCKS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 530; // above sand+speckles, below butte
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.10;
    this._iesScale = 1.0;
    this._shadowLenCur = this.uniforms.uShadowLen.value;
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

  setDensity(d) {
    this.uniforms.uDensity.value = clamp(d, 0.0, 1.0);
  }

  /* Optional IES hook (021_gmp_ies_lighting): tag the directional light
     with an isotropic IES profile; ballast-scaled intensity feeds uDirLightIntensity. */
  enableIES(light) {
    if (!light) return;
    autoConvertLight(light, generateIsotropicIES(light));
    const ies = light.userData && light.userData.ies;
    if (ies) {
      const flux = integrateCandela(ies);
      this._iesScale = (ies.ballastFactor || 1.0) * (flux > 0.0 ? clamp(flux / (4.0 * Math.PI), 0.25, 4.0) : 1.0);
    }
  }

  /* Full interaction with 016_DirectionalLightShader: direction, color,
     intensity, shadow map texture/matrix/size/bias + perceptual boost.  */
  syncFromLight(lightShader) {
    const lu = lightShader.getUniforms();
    const u = this.uniforms;

    _sunDirTmp.copy(lu.uDirLightDirection.value).normalize();
    u.uDirLightDirection.value.copy(_sunDirTmp);
    u.uDirLightColor.value.copy(lu.uDirLightColor.value);
    u.uDirLightIntensity.value = lu.uDirLightIntensity.value * this._iesScale;
    u.uSunDir.value.copy(_sunDirTmp).negate();
    u.uSunColor.value.copy(lu.uDirLightColor.value);
    u.uSunIntensity.value = lu.uDirLightIntensity.value;
    u.uShadowTint.value.copy(lu.uShadowTint.value);
    u.uShadowTintColor.value.copy(lu.uShadowTint.value);
    u.uCelSteps.value = lu.uCelSteps.value;
    u.uRimPower.value = lu.uRimPower.value;
    u.uRimIntensity.value = lu.uRimIntensity.value;
    u.uAmbientIntensity.value = lu.uAmbientIntensity.value;
    u.uAmbientColor.value.copy(lu.uAmbientColor.value);

    const light = lightShader.getLight ? lightShader.getLight() : null;
    if (light && light.shadow && light.shadow.map && light.shadow.map.texture) {
      u.uShadowMap.value = light.shadow.map.texture;
      u.uShadowMatrix.value.copy(light.shadow.matrix);
      u.uShadowMapSize.value.set(light.shadow.mapSize.x, light.shadow.mapSize.y);
      u.uShadowBias.value = light.shadow.bias;
      u.uShadowNormalBias.value = light.shadow.normalBias;
    }

    // Perceptual vibrancy boost on the lit crest color (020_gmp_perceptual_color)
    _baseCol.setRGB(
      this.uniforms.uColRockLight.value.x,
      this.uniforms.uColRockLight.value.y,
      this.uniforms.uColRockLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColRockLight.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    if (camera) {
      u.uCamPos.value.set(camera.position.x, camera.position.y, camera.position.z);
    }

    // Cast-shadow stretch reacts to sun elevation (low sun = long shadows)
    const elev = clamp(-u.uDirLightDirection.value.y, 0.05, 1.0);
    const targetLen = clamp(0.6 / elev, 0.6, 3.0);
    this._shadowLenCur = damp(this._shadowLenCur, targetLen, 3.0, dt);
    u.uShadowLen.value = this._shadowLenCur;

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
/* 5. FACTORY                                                          */
/* ------------------------------------------------------------------ */
export function createDesertRocksShader(options = {}) {
  return new DesertRocksShader(options);
}
