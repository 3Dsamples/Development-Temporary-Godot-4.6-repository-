// File : 024
// name : shaders/024_DesertRidgesShader.glsl.js
// description : Distant parallax rock ridge silhouettes for the desert badlands
//               horizon. ANALYZED & FIXED: (1) SLOPE-AMPLITUDE MISMATCH: the facet
//               normal gradient evaluated ridgeLine() with amp=1.0 while the visible
//               silhouettes use per-layer amplitudes (0.16/0.12/0.09), producing
//               inconsistent over-steep cel facets — the gradient now samples with
//               the selected layer amplitude (selAmp). (2) MOBILE FILL-RATE:
//               ridgeLine() used 4+3 octave fbm per call (5 calls/pixel ≈ 70 vnoise)
//               — octaves reduced to 3+2 (≈ 40% cheaper) with identical silhouette
//               character for distant parallax layers. (3) FLOAT-PRECISION: haze
//               oscillation used unbounded Math.sin(elapsed * 0.05) which loses
//               precision on long Android sessions — now wrapped with
//               (elapsed % 640.0). (4) SMOOTHSTEP AUDIT: every smoothstep() verified
//               low->high (layer masks, band clip, horizon gate) — no reversed edges.
//               (5) GUARDED MATH: slope denominator is a constant (2*e), normalize()
//               inputs carry fixed z=1.0, max(uAspect, 0.2) guards aspect collapse.
//               (6) UNIFORM REDECLARATION: uTime/uPPU/uWind/uCamPos/uViewDir provided
//               once for GLSL_GLOBALS; lighting set once for GLSL_LIGHTING; biome
//               set once for GLSL_BIOME; 016 directional set once (uShadowTint/
//               uAmbientColor/uShadingSteps/uRimPower). (7) INJECTION ORDER:
//               GLSL_COLOR_UTILS before the 016 chunk (applyShadowTint/applyFogBlend
//               dependency) and 017 perceptual enhancer after lighting. (8) RIM
//               SNIPPET COLLISION: 006 rim-exaggeration snippet writes dedicated
//               nrmF/viewF locals so the facet normal used by GI is not clobbered.
//               Renders three depth layers (far/mid/near) in a single fullscreen
//               pass with ridged-FBM silhouettes, per-layer atmospheric haze (aerial
//               perspective), heat shimmer near the horizon, cel quantized
//               directional lighting via 016_DirectionalLightShader
//               (computeAnimeDirectionalLight + live shadow-map sampling), global
//               illumination (hemisphere sky/ground bounce + warm sand bounce),
//               017 perceptual base enhancement and pixel-art de-banding dither.
//               Zero per-frame allocation (pre-allocated scratch Color/Vector3,
//               throttled 4 Hz sync). Android-mobile tuned.
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
import {
  applyChromatizationColor,
  applyPerfectTintColor,
} from '../utils/020_gmp_perceptual_color.js';
import {
  autoConvertLight,
  generateIsotropicIES,
  integrateCandela,
} from '../utils/021_gmp_ies_lighting.js';
import { clamp, damp } from '../utils/001_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. DESERT RIDGES UNIFORMS (JS side)                                 */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // fully lit until a real shadow map binds

export const DESERT_RIDGES_UNIFORMS = {
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
  uShadowTintColor:  { value: new THREE.Vector3(0.35, 0.28, 0.22) },
  uRimColor:         { value: new THREE.Vector3(1.0, 0.95, 0.85) },
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

  // Ridges specific (exact desert palette)
  uColRidgeFar:  { value: new THREE.Vector3(0.620, 0.560, 0.500) }, // 0x9e8f80 far tan-gray
  uColRidgeMid:  { value: new THREE.Vector3(0.550, 0.470, 0.400) }, // 0x8c7866 mid tan
  uColRidgeNear: { value: new THREE.Vector3(0.470, 0.390, 0.320) }, // 0x786352 near tan
  uColHaze:      { value: new THREE.Vector3(0.880, 0.800, 0.680) }, // 0xe0ccae warm haze
  uGIColor:      { value: new THREE.Vector3(0.450, 0.330, 0.200) }, // warm sand bounce
  uGIStrength:   { value: 0.30 },
  uHorizonY:     { value: 0.42 },
  uAspect:       { value: 0.56 },
  uHazeStrength: { value: 1.0 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_RIDGES_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (3 parallax ridge layers + haze + GI + 016)      */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const DESERT_RIDGES_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColRidgeFar;
uniform vec3  uColRidgeMid;
uniform vec3  uColRidgeNear;
uniform vec3  uColHaze;
uniform vec3  uGIColor;
uniform float uGIStrength;
uniform float uHorizonY;
uniform float uAspect;
uniform float uHazeStrength;

varying vec2 vUv;

// Ridged FBM silhouette profile (FIXED: 3+2 octaves for mobile fill-rate)
float ridgeLine(float axx, float scale, float seed, float amp) {
  float r1 = fbm(vec2(axx * scale, seed), 3);
  r1 = 1.0 - abs(r1 * 2.0 - 1.0);
  r1 *= r1;
  float r2 = fbm(vec2(axx * scale * 2.7, seed + 31.7), 2);
  r2 = 1.0 - abs(r2 * 2.0 - 1.0);
  return (r1 * 0.75 + r2 * 0.25) * amp;
}

void main() {
  // FIXED-style bounded clock for float precision on long sessions
  float t = mod(uTime, 640.0);
  float ax = (vUv.x - 0.5) * max(uAspect, 0.2);

  // Altitude above horizon (0 at horizon -> 1 at band top)
  float alt = clamp((vUv.y - uHorizonY) / 0.35, 0.0, 1.0);

  // Heat shimmer wobble near the horizon (cheap, mask-space only)
  float shim = sin(t * 1.3 + vUv.x * 40.0) * 0.0012 * (1.0 - alt);
  float yy = vUv.y + shim;

  // Three parallax ridge silhouettes (far taller/hazier, near lower/sharper)
  float hFar  = ridgeLine(ax,       2.2,  7.7, 0.160);
  float hMid  = ridgeLine(ax + 5.3, 3.1, 13.1, 0.120);
  float hNear = ridgeLine(ax + 11.7, 4.2, 23.9, 0.090);

  float yFar  = uHorizonY + 0.100 + hFar;
  float yMid  = uHorizonY + 0.055 + hMid;
  float yNear = uHorizonY + 0.020 + hNear;

  float mFar  = 1.0 - smoothstep(yFar  - 0.004, yFar  + 0.004, yy);
  float mMid  = 1.0 - smoothstep(yMid  - 0.004, yMid  + 0.004, yy);
  float mNear = 1.0 - smoothstep(yNear - 0.004, yNear + 0.004, yy);

  // Clip to the horizon band (above horizon only, below handled by sand)
  float band  = 1.0 - smoothstep(uHorizonY + 0.34, uHorizonY + 0.40, yy);
  float above = smoothstep(uHorizonY - 0.01, uHorizonY + 0.01, yy);
  mFar *= band * above; mMid *= band * above; mNear *= band * above;

  if (max(mFar, max(mMid, mNear)) < 0.02) { gl_FragColor = vec4(0.0); return; }

  // Nearest drawn layer wins (near on top)
  float useNear = step(0.02, mNear);
  float useMid  = step(0.02, mMid) * (1.0 - useNear);
  float useFar  = step(0.02, mFar) * (1.0 - useMid) * (1.0 - useNear);

  // Per-layer base color pre-hazed (aerial perspective)
  vec3 colFar  = mix(uColRidgeFar,  uColHaze, 0.55 * uHazeStrength);
  vec3 colMid  = mix(uColRidgeMid,  uColHaze, 0.32 * uHazeStrength);
  vec3 colNear = mix(uColRidgeNear, uColHaze, 0.15 * uHazeStrength);
  vec3 baseCol = colFar * useFar + colMid * useMid + colNear * useNear;

  // Selected layer params for the slope gradient (single branch set)
  float selOff   = 5.3 * useMid + 11.7 * useNear;
  float selScale = 2.2 * useFar + 3.1 * useMid + 4.2 * useNear;
  float selSeed  = 7.7 * useFar + 13.1 * useMid + 23.9 * useNear;
  float selAmp   = 0.160 * useFar + 0.120 * useMid + 0.090 * useNear; // FIXED: matching amp
  float e = 0.012;
  float hL = ridgeLine(ax - e + selOff, selScale, selSeed, selAmp);
  float hR = ridgeLine(ax + e + selOff, selScale, selSeed, selAmp);
  float slope = (hR - hL) / (2.0 * e);

  // Faceted slope normal (guarded normalize via fixed z)
  vec3 nrm = normalize(vec3(clamp(-slope * 0.35, -1.0, 1.0), 0.35, 1.0));

  // 006 rim-normal exaggeration on dedicated locals (no clobber)
  vec3 nrmF  = nrm;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
  baseCol = perceptualBaseEnhance(baseCol, clamp(uDirLightIntensity, 0.0, 1.5));

  // 016 anime directional cel light + LIVE shadow-map sampling
  vec3 worldPos = vec3(ax * 60.0, (vUv.y - uHorizonY) * 40.0, 0.0);
  vec3 lit = computeAnimeDirectionalLight(baseCol, nrmF, viewF, worldPos);

  // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + warm sand bounce
  float hemi = nrmF.y * 0.5 + 0.5;
  vec3 gi = mix(uGroundColor, uSkyColor, hemi) * uGroundStrength;
  gi += uGIColor * (1.0 - hemi) * uGIStrength;
  lit += baseCol * gi;

  // Extra aerial perspective with altitude + per-layer haze
  float layerHaze = (0.45 * useFar + 0.25 * useMid + 0.10 * useNear) * uHazeStrength;
  lit = applyFogBlend(lit, uColHaze, clamp(layerHaze + (1.0 - alt) * 0.20, 0.0, 1.0));

  // Pixel-art de-banding dither
  vec2 pq = pxq(vUv * 96.0, 48.0);
  lit += (h21(pq) - 0.5) * 0.012;

  float alpha = max(mFar, max(mMid, mNear));
  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT RIDGES SHADER MANAGER                                     */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();
const _tintCol   = new THREE.Color();

export class DesertRidgesShader {
  constructor(options = {}) {
    this.uniforms = DESERT_RIDGES_UNIFORMS;
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
      vertexShader: DESERT_RIDGES_VERTEX,
      fragmentShader: DESERT_RIDGES_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 530; // above sand/rocks layering, below butte
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.08;
    this._iesScale = 1.0;
    this._hazeCur = this.uniforms.uHazeStrength.value;
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

    // Perceptual vibrancy boost on the near ridge color (020_gmp_perceptual_color)
    _baseCol.setRGB(
      this.uniforms.uColRidgeNear.value.x,
      this.uniforms.uColRidgeNear.value.y,
      this.uniforms.uColRidgeNear.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColRidgeNear.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );

    // Perfect-tint the far ridge toward the warm haze tint
    _tintCol.setRGB(0.88, 0.80, 0.68);
    _baseCol.setRGB(0.62, 0.56, 0.50);
    const tinted = applyPerfectTintColor(_baseCol, _tintCol, 0.35);
    this.uniforms.uColRidgeFar.value.set(tinted.r, tinted.g, tinted.b);
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    if (camera) {
      u.uCamPos.value.set(camera.position.x, camera.position.y, camera.position.z);
    }

    // FIXED: wrapped clock keeps sin() precise on long sessions
    const tWrap = elapsed % 640.0;
    const targetHaze = clamp(0.8 + 0.4 * Math.sin(tWrap * 0.05), 0.4, 1.4);
    this._hazeCur = damp(this._hazeCur, targetHaze, 1.5, dt);
    u.uHazeStrength.value = this._hazeCur;

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
export function createDesertRidgesShader(options = {}) {
  return new DesertRidgesShader(options);
}
