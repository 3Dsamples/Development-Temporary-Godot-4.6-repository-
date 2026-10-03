// File : 020
// name : shaders/020_DesertSandShader.glsl.js
// description : Procedural desert sand ground shader for the badlands scene.
//               Renders the tan/orange sand floor below the horizon line with
//               perspective-compressed wind ripples, large dune undulation,
//               hash-grid pebble speckles, real-time heat shimmer near the
//               horizon, anime cel-quantized directional lighting with shadow-map
//               support (016_DirectionalLightShader chunk), perceptual base color
//               enhancement (017_PerceptualBaseEnhancer chunk), rim-exaggerated
//               ripple normals (006_normalQuantizer snippet), distance haze toward
//               the horizon (004_ColorPalette fog utility) and pixel-art de-banding
//               dither. Composed on shaders/000_BaseShader.glsl.js with ALL imports
//               wired: GLSL chunks (GLOBALS/NOISE/COLOR_UTILS/NORMAL_QUANT/LIGHTING/
//               BIOME), 016 directional light, 017 perceptual enhancer, 006 normal
//               quantizer, 020_gmp_perceptual_color.js chromatization wrappers,
//               021_gmp_ies_lighting.js IES auto-conversion hook, and
//               001_MathUtils.js scalar helpers. Injection order guarantees
//               GLSL_COLOR_UTILS before the 016 chunk (applyShadowTint dependency).
//               Zero per-frame allocation (pre-allocated scratch Color/Vector3,
//               throttled 4 Hz perceptual sync), bounded time clock, guarded
//               divisions, low->high smoothstep edges everywhere, single fullscreen
//               quad draw call. Android-mobile tuned.
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
/* 1. DESERT SAND UNIFORMS (JS side)                                   */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // default: fully lit until a real shadow map binds

export const DESERT_SAND_UNIFORMS = {
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

  // 016 directional light providers (safe defaults: white 1x1 shadow = lit)
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

  // Sand specific (exact desert palette)
  uColSandBase:  { value: new THREE.Vector3(0.850, 0.630, 0.390) }, // 0xd9a163 tan sand
  uColSandLight: { value: new THREE.Vector3(0.940, 0.760, 0.520) }, // 0xf0c285 ripple crest
  uColSandDark:  { value: new THREE.Vector3(0.720, 0.500, 0.300) }, // 0xb8804d ripple trough
  uColSpeckle:   { value: new THREE.Vector3(0.450, 0.300, 0.200) }, // 0x734d33 pebble dots
  uColHaze:      { value: new THREE.Vector3(0.930, 0.800, 0.620) }, // 0xedcc9e horizon haze
  uHorizonY:     { value: 0.42 },
  uAspect:       { value: 0.56 },
  uRippleStrength: { value: 0.75 },
  uShimmer:      { value: 1.0 },
  uSpeckleDensity: { value: 0.35 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_SAND_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9994, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (sand ripples + speckles + shimmer + cel light)  */
/*    Chunk order: COLOR_UTILS before 016 chunk (applyShadowTint dep)  */
/* ------------------------------------------------------------------ */
export const DESERT_SAND_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColSandBase;
uniform vec3  uColSandLight;
uniform vec3  uColSandDark;
uniform vec3  uColSpeckle;
uniform vec3  uColHaze;
uniform float uHorizonY;
uniform float uAspect;
uniform float uRippleStrength;
uniform float uShimmer;
uniform float uSpeckleDensity;

varying vec2 vUv;

void main() {
  // FIXED-style bounded clock for float precision on long sessions
  float t = mod(uTime, 640.0);

  // Ground mask: sand lives below the horizon line (low->high smoothstep)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.005, uHorizonY + 0.005, vUv.y);
  if (groundMask <= 0.001) { gl_FragColor = vec4(0.0); return; }

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.55, uHorizonY, vUv.y);

  // Real-time heat shimmer (stronger near horizon, two octave wobble)
  vec2 uv = vUv;
  uv.x += sin(vUv.y * 220.0 + t * 2.6) * 0.0016 * uShimmer * hp;
  uv.x += sin(vUv.y *  90.0 - t * 1.7) * 0.0011 * uShimmer * hp;

  // Perspective-compressed sand coordinates (ripples tighten at horizon)
  float persp = mix(1.0, 5.5, hp * hp);
  vec2 sp = vec2(uv.x * max(uAspect, 0.2) * 4.0, uv.y * 14.0) * persp;

  // Wind-drifted ripple field
  vec2 drift = vec2(t * 0.010, t * 0.004);
  float rippleWarp = fbm(sp * 0.35 + drift + 31.7, 3);
  float ripple = sin(sp.y * 3.0 + rippleWarp * 4.5);
  float rippleMask = smoothstep(0.15, 0.85, ripple * 0.5 + 0.5);

  // Large dune undulation
  float dune = fbm(vec2(sp.x * 0.10, sp.y * 0.22) + 7.7, 3);

  // Base sand color (trough -> base -> crest)
  vec3 col = mix(uColSandDark, uColSandBase, rippleMask * 0.7 + dune * 0.3);
  col = mix(col, uColSandLight, smoothstep(0.55, 0.95, rippleMask) * uRippleStrength);

  // Hash-grid pebble speckles (denser near camera)
  vec2 gp = sp * 6.0;
  vec2 cell = floor(gp);
  float h = h21(cell);
  vec2 f = fract(gp) - 0.5;
  vec2 jitter = vec2(h21(cell + 3.1), h21(cell + 5.7)) - 0.5;
  float dotM = 1.0 - smoothstep(0.05, 0.16, length(f - jitter * 0.6));
  float speck = dotM * step(1.0 - uSpeckleDensity, h) * (1.0 - hp * 0.7);
  col = mix(col, uColSpeckle, speck * 0.6);

  // Fake ripple normal (cheap analytic-ish gradient) for cel lighting
  float slopeX = (fbm((sp + vec2(0.02, 0.0)) * 0.35 + drift + 31.7, 2) - rippleWarp) * 6.0;
  float slopeY = ripple * 0.35;
  vec3 nrmF = normalize(vec3(-slopeX * uRippleStrength, -slopeY * uRippleStrength, 1.0));
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

  // Anime directional cel lighting + shadow map (016 chunk)
  vec3 worldPos = vec3(uv.x * 40.0 - 20.0, (1.0 - hp) * 30.0, 0.0);
  vec3 lit = computeAnimeDirectionalLight(col, nrmF, viewF, worldPos);

  // Perceptual base enhancement (017 chunk, OKLCH chroma/lightness preserve)
  lit = perceptualBaseEnhance(lit, clamp(uSunIntensity, 0.0, 1.5));

  // Distance haze toward horizon (aerial perspective)
  lit = applyFogBlend(lit, uColHaze, hp * hp * 0.75);

  // Pixel-art de-banding dither
  vec2 pq = pxq(vUv * 96.0, 48.0);
  lit += (h21(pq) - 0.5) * 0.012;

  gl_FragColor = vec4(lit, groundMask);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT SAND SHADER MANAGER                                       */
/*    Zero per-frame allocation; throttled perceptual + shadow sync    */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class DesertSandShader {
  constructor(options = {}) {
    this.uniforms = DESERT_SAND_UNIFORMS;
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
      vertexShader: DESERT_SAND_VERTEX,
      fragmentShader: DESERT_SAND_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 500; // above sky/clouds/sun-rays, below rocks/butte
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.10;
    this._iesScale = 1.0;
    this._shimmerCur = this.uniforms.uShimmer.value;
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

    // Perceptual vibrancy boost on sand crest color (020_gmp_perceptual_color)
    _baseCol.setRGB(
      this.uniforms.uColSandLight.value.x,
      this.uniforms.uColSandLight.value.y,
      this.uniforms.uColSandLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColSandLight.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    // Heat shimmer reacts to wind + time of day (damped, frame-rate independent)
    const windMag = u.uWind.value.length();
    const targetShimmer = clamp(0.6 + windMag * 0.8, 0.0, 1.6);
    this._shimmerCur = damp(this._shimmerCur, targetShimmer, 3.0, dt);
    u.uShimmer.value = this._shimmerCur;

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
export function createDesertSandShader(options = {}) {
  return new DesertSandShader(options);
}