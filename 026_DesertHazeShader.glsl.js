// File : 026
// name : shaders/026_DesertHazeShader.glsl.js
// description : Horizon heat-haze / dust atmosphere veil for the desert badlands
//               scene — final compositing layer of the desert set. ANALYZED (ROUND
//               2) & FIXED: (1) WIND-MULTIPLY POP BUG: drift used
//               uWind.x * uTime * 0.004 — because uWind is a damped live value,
//               any wind change rescaled the whole accumulated time product and
//               the dust bands jumped. Replaced with an INTEGRATED uniform
//               uWindDrift (this._windDrift += windX * dt * k) written each frame
//               by the manager, so dust advection is continuous under changing
//               wind. (2) MANAGER OSCILLATION WRAP POP: haze-strength target used
//               Math.sin((elapsed % 640) * 0.04) whose 640 s wrap is not sin-
//               periodic — now mod-by-2pi phase (continuous + precision-safe).
//               (3) RE-VERIFIED round-1 fixes: shimmer phases mod-by-2pi, unwrapped
//               small-coefficient base drift, all smoothstep() low->high, guarded
//               aspect/haze-height/denom/length divisions, night-gated Mie halo,
//               hard alpha early-out, chunk injection order (COLOR_UTILS before
//               016; 017 after lighting), single-provider uniform sets. Renders a
//               vertical veil peaking at the horizon with wider upper falloff,
//               wind-advected dust bands (anisotropic fbm), dual-octave heat
//               shimmer wobble, Mie-like sun scatter halo locked to the live sun
//               screen position, hemisphere + warm sand bounce GI into the veil,
//               perceptual base enhancement and pixel-art dither. Fully interacts
//               with 016_DirectionalLightShader (direction/intensity/sun color
//               synced at 4 Hz, sun screen projected from the live camera each
//               frame). Zero per-frame allocation (pre-allocated scratch
//               Vector3/Color, throttled 4 Hz sync). Android-mobile tuned, single
//               draw call.
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
import { quantizeGeometryNormalsCPU } from '../mesh/006_normalQuantizer.js';
import {
  applyChromatizationColor,
  applyPerfectTintColor,
} from '../utils/020_gmp_perceptual_color.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
} from '../utils/021_gmp_ies_lighting.js';
import { clamp, damp } from '../utils/001_gmp_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. DESERT HAZE UNIFORMS (JS side)                                   */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // fully lit until a real shadow map binds

export const DESERT_HAZE_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // GLSL_LIGHTING providers (GI palette + fog)
  uSunDir:           { value: new THREE.Vector3(0.5, 0.8, 0.3) },
  uMoonDir:          { value: new THREE.Vector3(-0.5, -0.8, -0.3) },
  uSunColor:         { value: new THREE.Vector3(1.0, 0.96, 0.85) },
  uMoonColor:        { value: new THREE.Vector3(0.42, 0.48, 0.70) },
  uSkyColor:         { value: new THREE.Vector3(0.45, 0.62, 0.85) },
  uGroundColor:      { value: new THREE.Vector3(0.35, 0.25, 0.16) },
  uFogColor:         { value: new THREE.Vector3(0.93, 0.80, 0.62) },
  uShadowTintColor:  { value: new THREE.Vector3(0.35, 0.28, 0.22) },
  uRimColor:         { value: new THREE.Vector3(1.0, 0.95, 1.00) },
  uSunIntensity:     { value: 1.0 },
  uMoonIntensity:    { value: 0.35 },
  uSkyStrength:      { value: 0.35 },
  uGroundStrength:   { value: 0.30 },
  uBounceStrength:   { value: 0.30 },
  uRimStrength:      { value: 0.40 },
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
  uRimIntensity:      { value: 0.40 },
  uShadowTint:        { value: new THREE.Color(0.35, 0.28, 0.22) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.25 },
  uAmbientColor:      { value: new THREE.Color(0.45, 0.62, 0.85) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Haze specific (exact desert palette)
  uColHaze:      { value: new THREE.Vector3(0.880, 0.800, 0.680) }, // 0xe0ccae warm horizon haze
  uColDust:      { value: new THREE.Vector3(0.780, 0.660, 0.470) }, // 0xc7a878 drifting dust
  uGIColor:      { value: new THREE.Vector3(0.450, 0.330, 0.200) }, // warm sand bounce
  uGIStrength:   { value: 0.30 },
  uHazeHeight:   { value: 0.22 },
  uHazeStrength: { value: 0.80 },
  uDustDensity:  { value: 0.60 },
  uShimmer:      { value: 1.00 },
  uHorizonY:     { value: 0.42 },
  uAspect:       { value: 0.56 },
  uSunScreen:    { value: new THREE.Vector2(0.18, 0.86) },
  uWindDrift:    { value: 0.0 }, // FIXED: integrated wind advection phase
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_HAZE_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (horizon veil + dust bands + shimmer + halo)     */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const DESERT_HAZE_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColHaze;
uniform vec3  uColDust;
uniform vec3  uGIColor;
uniform float uGIStrength;
uniform float uHazeHeight;
uniform float uHazeStrength;
uniform float uDustDensity;
uniform float uShimmer;
uniform float uHorizonY;
uniform float uAspect;
uniform vec2  uSunScreen;
uniform float uWindDrift;

varying vec2 vUv;

void main() {
  // FIXED (round 1): sin-phase clocks wrapped by 2pi (continuous + precise)
  float phA = mod(uTime * 2.0, 6.2831853);
  float phB = mod(uTime * 1.3, 6.2831853);

  float ax = (vUv.x - 0.5) * max(uAspect, 0.2);
  float up = vUv.y - uHorizonY;

  // Vertical veil profile: peaks at horizon, wider falloff above (low->high)
  float hH = max(uHazeHeight, 0.05);
  float veil = 1.0 - smoothstep(0.0, hH, abs(up));
  veil = max(veil * 0.90, (1.0 - smoothstep(0.0, hH * 2.2, up)) * 0.55);

  // Dual-octave heat shimmer wobble (real-time, pop-free)
  float wob = sin(phA + vUv.x * 30.0) * 0.020 * uShimmer
            + sin(phB + vUv.x * 17.0) * 0.013 * uShimmer;

  // FIXED (round 2): drift = unwrapped base + INTEGRATED wind phase
  // (continuous under changing wind; no rescale pop)
  float drift = uTime * 0.02 + uWindDrift;

  // Wind-advected dust bands (anisotropic fbm)
  vec2 dp = vec2(ax * 3.0 + drift, (vUv.y + wob) * 9.0);
  float dust = fbm(dp, 3);
  float bands = smoothstep(0.35, 0.75, dust) * uDustDensity;

  // Mie-like sun scatter halo (guarded length, night-gated)
  vec2 dSun = (vUv - uSunScreen) * vec2(max(uAspect, 0.2), 1.0);
  float sd = max(length(dSun), 1e-4);
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0);
  float halo = (exp(-sd * 4.0) * 0.6 + exp(-sd * 12.0) * 0.4) * sunAbove;

  float alpha = (veil * (0.50 + bands * 0.30) + halo * 0.35) * uHazeStrength;
  if (alpha < 0.01) { gl_FragColor = vec4(0.0); return; }

  // Atmosphere color: horizon dust -> sky blend with altitude (guarded denom)
  float skyMix = clamp(up / max(hH * 2.2, 1e-3), 0.0, 1.0);
  vec3 col = mix(uColHaze, uSkyColor, skyMix * 0.6);
  col = mix(col, uColDust, bands * 0.4);

  // GLOBAL ILLUMINATION into the veil: hemisphere + warm sand bounce
  float hemi = clamp(0.5 + up * 2.0, 0.0, 1.0);
  vec3 giCol = mix(uGroundColor, uSkyColor, hemi) * uGroundStrength;
  giCol += uGIColor * (1.0 - hemi) * uGIStrength;
  col += giCol * 0.35;

  // Sun tint into the halo
  col = mix(col, uSunColor, clamp(halo * 0.6, 0.0, 1.0));

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
  col = perceptualBaseEnhance(col, clamp(uDirLightIntensity, 0.0, 1.5) * 0.5);

  // Pixel-art de-banding dither
  vec2 pq = pxq(vUv * 96.0, 48.0);
  col += (h21(pq) - 0.5) * 0.012;

  gl_FragColor = vec4(col, clamp(alpha, 0.0, 0.90));
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT HAZE SHADER MANAGER                                       */
/*    Zero per-frame allocation; throttled 4 Hz light sync             */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _sunPos    = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _tintCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class DesertHazeShader {
  constructor(options = {}) {
    this.uniforms = DESERT_HAZE_UNIFORMS;
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
      vertexShader: DESERT_HAZE_VERTEX,
      fragmentShader: DESERT_HAZE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 900; // top of the desert stack (below HUD)
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.06;
    this._iesScale = 1.0;
    this._strengthCur = this.uniforms.uHazeStrength.value;
    this._windDrift = 0.0; // FIXED (round 2): integrated wind advection
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uAspect.value = clamp(a, 0.2, 0.2, 2.5) !== undefined ? clamp(a, 0.2, 2.5) : a;
  }

  setHorizon(y) {
    this.uniforms.uHorizonY.value = clamp(y, 0.1, 0.9);
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

  /* Full interaction with 016_DirectionalLightShader: direction, intensity,
     sun color + perceptual perfect-tint of the haze hue, chromatized dust. */
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

    // Perceptual perfect-tint: haze hue leans toward the live sun tint
    _baseCol.setRGB(
      this.uniforms.uColHaze.value.x,
      this.uniforms.uColHaze.value.y,
      this.uniforms.uColHaze.value.z
    );
    _tintCol.copy(lu.uDirLightColor.value);
    const tinted = applyPerfectTintColor(_baseCol, _tintCol, 0.25);
    this.uniforms.uColHaze.value.set(tinted.r, tinted.g, tinted.b);

    // Chromatized dust bands keep vibrancy under strong sun
    _baseCol.setRGB(
      this.uniforms.uColDust.value.x,
      this.uniforms.uColDust.value.y,
      this.uniforms.uColDust.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColDust.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    // FIXED (round 2): integrate wind into a drift phase (no rescale pop)
    this._windDrift += u.uWind.value.x * dt * 0.25;
    u.uWindDrift.value = this._windDrift;

    if (camera) {
      u.uCamPos.value.set(camera.position.x, camera.position.y, camera.position.z);

      // Live sun screen position for the scatter halo
      _sunDirTmp.copy(u.uDirLightDirection.value).negate().normalize();
      _sunPos.copy(camera.position).addScaledVector(_sunDirTmp, 500.0).project(camera);
      u.uSunScreen.value.set(_sunPos.x * 0.5 + 0.5, _sunPos.y * 0.5 + 0.5);
    }

    // FIXED (round 2): mod-by-2pi phase for the heat oscillation (no wrap pop)
    const ph = (elapsed * 0.04) % 6.2831853;
    const windMag = Math.sqrt(u.uWind.value.x * u.uWind.value.x + u.uWind.value.y * u.uWind.value.y);
    const target = clamp(0.7 + windMag * 0.5 + 0.15 * Math.sin(ph), 0.3, 1.3);
    this._strengthCur = damp(this._strengthCur, target, 1.5, dt);
    u.uHazeStrength.value = this._strengthCur;

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
export function createDesertHazeShader(options = {}) {
  return new DesertHazeShader(options);
}
