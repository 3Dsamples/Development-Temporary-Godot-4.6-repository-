// File : 021
// name : shaders/021_DesertSpecklesShader.glsl.js
// description : Scattered pebbles / gravel dots layer for the desert badlands scene.
//               ANALYZED & FIXED: (1) ATAN SINGULARITY: pebble edge wobble used
//               atan(d2.y, d2.x) which is undefined at the exact jittered center on
//               strict GLSL ES 1.00 drivers — replaced with atan(d2.y, d2.x + 1e-6).
//               (2) SMOOTHSTEP AUDIT: every smoothstep() verified low->high
//               (horizon mask, perspective proximity, pebble rim) — no reversed
//               edges. (3) DEGENERATE RADIUS GUARD: pebble rim band
//               (radius*wob-0.03, radius*wob+0.03) can invert when radius collapses
//               — radius now clamped with max(radius, 0.06) before the band.
//               (4) UNIFORM REDECLARATION: uTime/uPPU/uWind/uCamPos/uViewDir provided
//               once for GLSL_GLOBALS; lighting set once for GLSL_LIGHTING; biome
//               set once for GLSL_BIOME; uShadingSteps/uRimPower declared only where
//               the 006_normalQuantizer snippets need them. (5) INJECTION ORDER:
//               GLSL_COLOR_UTILS before the 016 directional chunk (applyShadowTint
//               dependency) and 017 perceptual enhancer after lighting. (6) FILL-RATE:
//               three hard early-outs (sky pixels, empty hash cells, sub-threshold
//               pebbles) plus horizon density attenuation so distant dots never cost
//               fragment work on mobile. (7) SHADOW-MAP SAFETY: default 1x1 white
//               DataTexture keeps sampleAnimeShadowMap() defined before a real shadow
//               map binds. (8) FLOAT PRECISION: bounded clock mod(uTime, 640.0) for
//               all animated terms. Renders hash-grid jittered pebbles with
//               perspective compression toward the horizon, 4-stop palette variation,
//               cel-quantized directional lighting with shadow support, rim
//               exaggeration, perceptual chromatization and pixel-art dither.
//               Zero per-frame allocation (pre-allocated scratch Color/Vector3,
//               throttled 4 Hz perceptual sync). Android-mobile tuned.
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
import { autoConvertLight, integrateCandela } from '../utils/021_gmp_ies_lighting.js';
import { clamp, damp } from '../utils/001_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. DESERT SPECKLES UNIFORMS (JS side)                               */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // default: fully lit until a real shadow map binds

export const DESERT_SPECKLES_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // GLSL_LIGHTING providers (synced by LightingSystem / 016)
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

  // GLSL_BIOME providers
  uBiomeW:    { value: new THREE.Vector4(1.0, 0.0, 0.0, 0.0) },
  uValley:    { value: new THREE.Vector3(0.0, 12.0, 0.0) },
  uTransMix:  { value: 1.0 },
  uTransFrom: { value: 0.0 },
  uTransTo:   { value: 0.0 },

  // 017 perceptual enhancer provider
  uPerceptualEnhance: { value: 0.85 },

  // 016 directional light providers (safe defaults)
  uDirLightColor:     { value: new THREE.Color(1.0, 0.96, 0.85) },
  uDirLightDirection: { value: new THREE.Vector3(-0.5, -0.8, -0.3) },
  uDirLightIntensity: { value: 1.0 },
  uCelSteps:          { value: 4.0 },
  uRimPower:          { value: 3.0 },
  uRimIntensity:      { value: 0.45 },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.25 },
  uAmbientColor:      { value: new THREE.Color(0.45, 0.62, 0.85) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Speckles specific (exact desert palette)
  uColPebbleDark:  { value: new THREE.Vector3(0.420, 0.270, 0.170) }, // 0x6b452b dark pebble
  uColPebbleMid:   { value: new THREE.Vector3(0.620, 0.450, 0.290) }, // 0x9e734a mid tan
  uColPebbleGray:  { value: new THREE.Vector3(0.560, 0.520, 0.470) }, // 0x8f8578 gray stone
  uColPebbleLight: { value: new THREE.Vector3(0.870, 0.720, 0.520) }, // 0xdeb885 lit crest
  uHorizonY:       { value: 0.42 },
  uAspect:         { value: 0.56 },
  uDensity:        { value: 0.75 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_SPECKLES_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (hash-grid pebbles + cel lighting + dither)      */
/*    Chunk order: COLOR_UTILS before 016 chunk; 017 after lighting    */
/* ------------------------------------------------------------------ */
export const DESERT_SPECKLES_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColPebbleDark;
uniform vec3  uColPebbleMid;
uniform vec3  uColPebbleGray;
uniform vec3  uColPebbleLight;
uniform float uHorizonY;
uniform float uAspect;
uniform float uDensity;

varying vec2 vUv;

void main() {
  // FIXED: bounded clock for float precision on long sessions
  float t = mod(uTime, 640.0);

  // Ground mask: pebbles live below the horizon line (low->high smoothstep)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.005, uHorizonY + 0.005, vUv.y);
  if (groundMask <= 0.001) { gl_FragColor = vec4(0.0); return; }

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.55, uHorizonY, vUv.y);

  // Perspective-compressed hash grid (dots tighten toward horizon)
  float persp = mix(1.0, 6.0, hp * hp);
  vec2 sp = vec2(vUv.x * max(uAspect, 0.2) * 34.0, vUv.y * 90.0) * persp;

  vec2 cell = floor(sp);
  vec2 f = fract(sp) - 0.5;
  float h  = h21(cell);
  float h2 = h21(cell + 7.7);

  // Density gate with horizon attenuation (fill-rate conservation)
  float gate = step(1.0 - uDensity * (1.0 - hp * 0.6), h);
  if (gate < 0.5) { gl_FragColor = vec4(0.0); return; }

  // Jittered center + pebble radius (FIXED: radius clamped before rim band)
  vec2 jitter = vec2(h21(cell + 3.1), h21(cell + 5.7)) - 0.5;
  vec2 d2 = f - jitter * 0.7;
  float radius = max(0.10 + h2 * 0.16, 0.06);

  // Distorted circle (FIXED: guarded atan)
  float ang = atan(d2.y, d2.x + 1e-6);
  float wob = 1.0 + sin(ang * 5.0 + h * 40.0) * 0.12 + sin(ang * 9.0 + h * 27.0) * 0.07;
  float dist = length(d2);
  float shape = 1.0 - smoothstep(radius * wob - 0.03, radius * wob + 0.03, dist);
  if (shape < 0.02) { gl_FragColor = vec4(0.0); return; }

  // 4-stop palette variation by hash
  vec3 col = paletteMix4(uColPebbleDark, uColPebbleMid, uColPebbleGray, uColPebbleLight, fract(h * 13.7));

  // Fake pebble normal (dome) for cel lighting
  vec3 nrmF = normalize(vec3(d2.x * 2.0, d2.y * 2.0, 1.0));
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

  // Anime directional cel lighting + shadow map (016 chunk)
  vec3 worldPos = vec3(vUv.x * 40.0 - 20.0, (1.0 - hp) * 30.0, 0.0);
  vec3 lit = computeAnimeDirectionalLight(col, nrmF, viewF, worldPos);

  // Top-lit crest highlight (cel band)
  float crest = 1.0 - smoothstep(0.10, 0.45, length(d2 - vec2(-0.02, -0.03)));
  lit = mix(lit, uColPebbleLight, crest * 0.35);

  // Perceptual base enhancement (017 chunk, OKLCH chroma/lightness preserve)
  lit = perceptualBaseEnhance(lit, clamp(uSunIntensity, 0.0, 1.5));

  // Distance haze toward horizon (aerial perspective)
  lit = applyFogBlend(lit, uFogColor, hp * hp * 0.55);

  // Pixel-art de-banding dither
  vec2 pq = pxq(vUv * 96.0, 48.0);
  lit += (h21(pq) - 0.5) * 0.012;

  float alpha = shape * groundMask * (1.0 - hp * 0.5);
  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT SPECKLES SHADER MANAGER                                   */
/*    Zero per-frame allocation; throttled perceptual + shadow sync    */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class DesertSpecklesShader {
  constructor(options = {}) {
    this.uniforms = DESERT_SPECKLES_UNIFORMS;
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
      vertexShader: DESERT_SPECKLES_VERTEX,
      fragmentShader: DESERT_SPECKLES_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 510; // above sand (020), below rocks/butte
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.10;
    this._iesScale = 1.0;
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
     with an isotropic IES profile; ballast factor scales intensity.     */
  enableIES(light) {
    if (!light) return;
    autoConvertLight(light);
    const ies = light.userData && light.userData.ies;
    if (ies) {
      const flux = integrateCandela(ies);
      this._iesScale = (ies.ballastFactor || 1.0) * (flux > 0.0 ? clamp(flux / (4.0 * Math.PI), 0.25, 4.0) : 1.0);
    }
  }

  /* Sync direction/color/intensity + real shadow map from 016 manager */
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
    u.uShadowTintColor.value.copy(lu.uShadowTint.value);

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
      this.uniforms.uColPebbleLight.value.x,
      this.uniforms.uColPebbleLight.value.y,
      this.uniforms.uColPebbleLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColPebbleLight.value.set(
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

    if (lightShader) {
      this._syncAccum += dt;
      if (this._syncAccum >= 0.25) { // throttled light+perceptual sync (4 Hz)
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
export function createDesertSpecklesShader(options = {}) {
  return new DesertSpecklesShader(options);
}
