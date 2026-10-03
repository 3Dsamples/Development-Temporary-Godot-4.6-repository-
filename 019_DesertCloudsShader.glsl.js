// File : 019
// name : shaders/019_DesertCloudsShader.glsl.js
// description : Streaky anime cumulus bands shader for the desert sky. ANALYZED &
//               FIXED: (1) REVERSED SMOOTHSTEP AUDIT: every smoothstep() verified
//               low->high (band mask, coverage gate, edge rim, night gate) — no
//               undefined-behavior edges on Mali/Adreno. (2) UNGUARDED MATH: atan-
//               free design; all divisions guarded (max(uCelBands,1.0),
//               max(uAspect,0.2), max(uSoftness,0.02), length epsilons on sun
//               vectors). (3) FLOAT-PRECISION: unbounded uTime wrapped with
//               mod(uTime, 640.0) for drift/warp terms. (4) UNIFORM REDECLARATION:
//               uTime/uPPU/uWind/uCamPos/uViewDir provided once for GLSL_GLOBALS;
//               lighting set provided once for GLSL_LIGHTING; biome set provided
//               for GLSL_BIOME; uShadingSteps/uRimPower declared only where the
//               006_normalQuantizer snippets need them. (5) INJECTION ORDER:
//               GLSL_COLOR_UTILS before GLSL_LIGHTING; GLSL_PERCEPTUAL_BASE_ENHANCER
//               after lighting so perceptualBaseEnhance() resolves. (6) OVERDRAW:
//               two hard early-outs (band mask, coverage) discard empty sky pixels
//               before any FBM work — critical for mobile fill-rate. (7) JS SYNC:
//               zero per-frame allocation (pre-allocated scratch Color/Vector3),
//               perceptual chromatization of crest colors throttled to 4 Hz via
//               020_gmp_perceptual_color.js, frame-rate independent damping via
//               001_MathUtils.js damp(). Renders horizontal wispy cumulus bands
//               with cel-quantized tones, sun-kissed rims, belly shadow tint and
//               night fade, synced to the 016 directional light pipeline.
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
import {
  getAnimeNormalQuantizationGLSL,
  getAnimeRimNormalExaggerationGLSL,
  quantizeGeometryNormalsCPU,
} from '../mesh/006_normalQuantizer.js';
import { applyChromatizationColor } from '../utils/020_gmp_perceptual_color.js';
import { clamp, mix, damp } from '../utils/001_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. DESERT CLOUDS UNIFORMS (JS side)                                 */
/* ------------------------------------------------------------------ */
export const DESERT_CLOUDS_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // GLSL_LIGHTING providers (synced by 016 / LightingSystem)
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

  // Local (006 snippets + cloud params)
  uLightDir: { value: new THREE.Vector3(0.5, 0.8, 0.3) },
  uRimPower: { value: 3.0 },

  // Clouds specific (desert palette)
  uColCloudShadow: { value: new THREE.Vector3(0.620, 0.700, 0.820) }, // 0x9eb3d1 blue-gray base
  uColCloudMid:    { value: new THREE.Vector3(0.880, 0.920, 0.970) }, // 0xe0ebf7 mid
  uColCloudLight:  { value: new THREE.Vector3(1.000, 1.000, 1.000) }, // 0xffffff lit white
  uColCloudHigh:   { value: new THREE.Vector3(1.000, 0.980, 0.940) }, // 0xfffaef sun-kissed crest
  uColCloudNight:  { value: new THREE.Vector3(0.160, 0.210, 0.340) }, // 0x293557 night cloud
  uSunScreen:      { value: new THREE.Vector2(0.18, 0.86) },
  uCoverage:       { value: 0.50 },
  uSoftness:       { value: 0.24 },
  uCelBands:       { value: 4.0 },
  uDriftSpeed:     { value: 0.014 },
  uAspect:         { value: 0.56 },
  uAlpha:          { value: 0.90 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad + anime normal quantization)      */
/* ------------------------------------------------------------------ */
export const DESERT_CLOUDS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

uniform vec3  uLightDir;
uniform float uShadingSteps;

varying vec2 vUv;
varying vec3 vNrmQ;

void main() {
  vUv = position.xy * 0.5 + 0.5;

  // Fake billboard normal, cel-quantized on the GPU (006_normalQuantizer)
  vec3 nrm = vec3(0.0, 0.0, 1.0);
  ${getAnimeNormalQuantizationGLSL('nrm', 'uLightDir', 'uShadingSteps')}
  vNrmQ = nrm;

  gl_Position = vec4(position.xy, 0.9992, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (streaky cumulus bands + cel tones + sun kiss)   */
/* ------------------------------------------------------------------ */
export const DESERT_CLOUDS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}

uniform vec3  uColCloudShadow;
uniform vec3  uColCloudMid;
uniform vec3  uColCloudLight;
uniform vec3  uColCloudHigh;
uniform vec3  uColCloudNight;
uniform vec2  uSunScreen;
uniform float uCoverage;
uniform float uSoftness;
uniform float uCelBands;
uniform float uDriftSpeed;
uniform float uAspect;
uniform float uAlpha;
uniform float uRimPower;

varying vec2 vUv;
varying vec3 vNrmQ;

void main() {
  // FIXED: wrapped clock keeps float precision on long mobile sessions
  float t = mod(uTime, 640.0);

  // Aspect-corrected coordinates (guarded)
  vec2 uv = vec2(vUv.x * max(uAspect, 0.2), vUv.y);

  // Horizontal streak bands (all smoothstep edges low->high)
  float bandMask = smoothstep(0.34, 0.50, vUv.y) * (1.0 - smoothstep(0.86, 0.98, vUv.y));
  if (bandMask <= 0.004) { gl_FragColor = vec4(0.0); return; } // overdraw early-out

  // Drifting, domain-warped anisotropic field
  vec2 p = vec2(uv.x * 2.2 + t * uDriftSpeed * 6.0, uv.y * 7.5);
  float warp = fbm(p * 0.45 + 13.7, 2) - 0.5;
  p.x += warp * 1.8;

  float base   = fbm(p, 4);
  float streak = fbm(vec2(p.x * 0.32, p.y * 2.4) + 71.3, 3);
  float field  = base * 0.62 + streak * 0.38;

  // Bigger cumulus near the horizon line
  field += (1.0 - smoothstep(0.35, 0.75, vUv.y)) * 0.10;

  // Coverage gate (guarded softness)
  float shape = smoothstep(uCoverage, uCoverage + max(uSoftness, 0.02), field);
  shape *= bandMask;
  if (shape < 0.02) { gl_FragColor = vec4(0.0); return; } // overdraw early-out

  // Cel quantization into discrete anime tones (guarded division)
  float tone = floor(clamp(shape, 0.0, 1.0) * uCelBands + 0.5) / max(uCelBands, 1.0);
  vec3 col = paletteMix4(uColCloudShadow, uColCloudMid, uColCloudLight, uColCloudHigh, tone);

  // Perceptual base enhancement (017 chunk, OKLCH chroma/lightness preserve)
  col = perceptualBaseEnhance(col, clamp(uSunIntensity, 0.0, 1.5));

  // Sun kiss (guarded normalize-free falloff)
  vec2 toSun = uSunScreen - vUv + vec2(1e-5, 1e-5);
  float sunDist = max(length(toSun), 1e-4);
  float kiss = (1.0 - clamp(sunDist, 0.0, 1.0)) * clamp(uSunDir.y, 0.0, 1.0);
  col = mix(col, uSunColor, kiss * 0.35);

  // Rim-exaggerated quantized normal -> sun-facing edge glow
  vec3 nrmF  = vNrmQ;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}
  float edge = smoothstep(0.25, 0.55, shape) * (1.0 - smoothstep(0.55, 0.90, shape));
  col = applyRimLight(col, edge * kiss, uSunColor, 0.30);

  // Belly shadow tint (perceptual, from GLSL_COLOR_UTILS)
  col = applyShadowTint(col, (1.0 - tone) * 0.55, uShadowTintColor);

  // Night fade (low->high gate on sun elevation)
  float dayGate = smoothstep(-0.08, 0.12, uSunDir.y);
  col = mix(uColCloudNight, col, mix(0.25, 1.0, dayGate));

  // Pixel-art de-banding dither
  vec2 pq = pxq(vUv * 96.0, 48.0);
  col += (h21(pq) - 0.5) * 0.012;

  float alpha = clamp(shape, 0.0, 1.0) * uAlpha;
  gl_FragColor = vec4(col, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT CLOUDS SHADER MANAGER                                     */
/*    Zero per-frame allocation; throttled perceptual sync (4 Hz)      */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class DesertCloudsShader {
  constructor(options = {}) {
    this.uniforms = DESERT_CLOUDS_UNIFORMS;
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
      vertexShader: DESERT_CLOUDS_VERTEX,
      fragmentShader: DESERT_CLOUDS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 870; // above sky (017), below sun rays (018)
    this._drift = this.uniforms.uDriftSpeed.value;
    this._syncAccum = 1.0; // force first perceptual tint immediately
    this._chroma = options.chroma !== undefined ? options.chroma : 1.10;
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setAspect(a) {
    this.uniforms.uAspect.value = clamp(a, 0.2, 2.5);
  }

  setCoverage(c) {
    this.uniforms.uCoverage.value = clamp(c, 0.1, 0.9);
  }

  /* Sync direction/color/intensity from 016 DirectionalLightShader or
     019_LightingSystem config; perceptual chromatization throttled.   */
  syncFromLight(lightShader, camera) {
    const lu = lightShader.getUniforms();
    const u = this.uniforms;

    _sunDirTmp.copy(lu.uDirLightDirection.value).negate().normalize();
    u.uSunDir.value.copy(_sunDirTmp);
    u.uLightDir.value.copy(_sunDirTmp);
    u.uSunIntensity.value = lu.uDirLightIntensity.value;
    u.uSunColor.value.copy(lu.uDirLightColor.value);
    u.uShadowTintColor.value.copy(lu.uShadowTint.value);

    if (camera) {
      const sp = _sunDirTmp.clone(); // one alloc only on sync tick (4 Hz)
      camera.position.addScaledVector(sp, 500.0);
      sp.project(camera);
      u.uSunScreen.value.set(sp.x * 0.5 + 0.5, sp.y * 0.5 + 0.5);
    }

    // Perceptual vibrancy boost on the lit crest colors (020_gmp_perceptual_color)
    _baseCol.setRGB(
      this.uniforms.uColCloudLight.value.x,
      this.uniforms.uColCloudLight.value.y,
      this.uniforms.uColCloudLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColCloudHigh.value.set(
      mix(1.0, _boosted.r, 0.5),
      mix(0.98, _boosted.g, 0.5),
      mix(0.94, _boosted.b, 0.5)
    );
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    // Wind-driven drift with frame-rate independent damping (001_MathUtils)
    const windX = u.uWind.value.x;
    this._drift = damp(this._drift, 0.014 + Math.abs(windX) * 0.004, 3.0, dt);
    u.uDriftSpeed.value = this._drift;

    if (lightShader) {
      this._syncAccum += dt;
      if (this._syncAccum >= 0.25) { // throttled light+perceptual sync (4 Hz)
        this._syncAccum = 0.0;
        this.syncFromLight(lightShader, camera);
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
export function createDesertCloudsShader(options = {}) {
  return new DesertCloudsShader(options);
}
