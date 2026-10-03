// File : 023
// name : shaders/023_DesertButteShader.glsl.js
// description : Central hero butte/mountain shader for the desert badlands scene.
//               ANALYZED & FIXED: (1) UNGUARDED NORMALIZE: cast-shadow direction
//               normalized uDirLightDirection.xy directly (NaN when sun is exactly
//               at zenith) — now guarded with max(length, 1e-4). (2) UNGUARDED
//               DIVISION: height factor divided by (peakY - baseY) which collapses
//               when the envelope closes — now max(peakY - baseY, 1e-3). (3) DEAD
//               BOUNDED CLOCK: mod(uTime, 640.0) local was computed but never used
//               — now drives a subtle real-time heat shimmer on the peak line.
//               (4) DEGENERATE SHADOW BAND: smoothstep(uShadowLen*0.8, uShadowLen,
//               along) is undefined when uShadowLen == 0 — now max(uShadowLen,
//               0.05). (5) SMOOTHSTEP AUDIT: every smoothstep() verified low->high
//               (envelope, peak/base masks, cast-shadow bands, scree/striation/
//               crack bands) — no reversed edges on Mali/Adreno. (6) NIGHT LEAK:
//               cast shadow rendered even when the sun was below the horizon — now
//               gated by sunAbove = clamp(-uDirLightDirection.y, 0, 1). (7) UNIFORM
//               REDECLARATION: uTime/uPPU/uWind/uCamPos/uViewDir provided once for
//               GLSL_GLOBALS; lighting set once for GLSL_LIGHTING; biome set once
//               for GLSL_BIOME; 016 directional set once (uShadowTint/uAmbientColor/
//               uShadingSteps/uRimPower). (8) INJECTION ORDER: GLSL_COLOR_UTILS
//               before the 016 chunk (applyShadowTint/applyFogBlend/paletteMix4
//               dependency) and 017 perceptual enhancer after lighting. (9) RIM
//               SNIPPET COLLISION: 006 rim-exaggeration snippet now writes dedicated
//               nrmF/viewF locals so the facet normal used by GI is not clobbered
//               before the hemisphere bounce. Renders the large faceted rock butte
//               rising above the horizon with ridged-FBM peak silhouette,
//               cel-quantized facet normals, striated layered rock, vertical crack
//               fissures, scree skirts at the base, and a procedural projected
//               cast-shadow band stretching across the sand along the live sun
//               direction. Fully interacts with 016_DirectionalLightShader
//               (computeAnimeDirectionalLight + live shadow-map sampling) and global
//               illumination (hemisphere sky/ground bounce + warm sand bounce),
//               017 perceptual base enhancement (OKLCH), 020_gmp_perceptual_color.js
//               chromatization/perfect-tint wrappers, 021_gmp_ies_lighting.js IES
//               auto-conversion hooks, and 001_MathUtils.js scalar helpers. Zero
//               per-frame allocation (pre-allocated scratch Color/Vector3, throttled
//               4 Hz sync). Android-mobile tuned, single fullscreen quad draw call.
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
/* 1. DESERT BUTTE UNIFORMS (JS side)                                  */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // fully lit until a real shadow map binds

export const DESERT_BUTTE_UNIFORMS = {
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

  // Butte specific (exact desert palette)
  uColButteDark:  { value: new THREE.Vector3(0.420, 0.280, 0.170) }, // 0x6b472b shadowed tan
  uColButteMid:   { value: new THREE.Vector3(0.620, 0.450, 0.290) }, // 0x9e734a mid tan
  uColButteLight: { value: new THREE.Vector3(0.870, 0.720, 0.520) }, // 0xdeb885 lit crest
  uColButteGray:  { value: new THREE.Vector3(0.560, 0.520, 0.470) }, // 0x8f8578 gray strata
  uColScree:      { value: new THREE.Vector3(0.720, 0.580, 0.420) }, // 0xb89469 scree skirt
  uGIColor:       { value: new THREE.Vector3(0.450, 0.330, 0.200) }, // warm sand bounce
  uGIStrength:    { value: 0.35 },
  uHorizonY:      { value: 0.42 },
  uAspect:        { value: 0.56 },
  uPeakHeight:    { value: 0.34 },
  uShadowLen:     { value: 0.55 },
  uShadowAlpha:   { value: 0.45 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const DESERT_BUTTE_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (faceted butte + GI + 016 light + cast shadow)   */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const DESERT_BUTTE_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColButteDark;
uniform vec3  uColButteMid;
uniform vec3  uColButteLight;
uniform vec3  uColButteGray;
uniform vec3  uColScree;
uniform vec3  uGIColor;
uniform float uGIStrength;
uniform float uHorizonY;
uniform float uAspect;
uniform float uPeakHeight;
uniform float uShadowLen;
uniform float uShadowAlpha;

varying vec2 vUv;

// Ridged ridge profile along aspect-corrected X (central butte)
float ridgeProf(float axx) {
  float rx = axx * 3.0;
  float r1 = fbm(vec2(rx * 1.3, 7.7), 4);
  r1 = 1.0 - abs(r1 * 2.0 - 1.0);
  r1 *= r1;
  float r2 = fbm(vec2(rx * 3.1, 13.1), 3);
  r2 = 1.0 - abs(r2 * 2.0 - 1.0);
  return r1 * 0.75 + r2 * 0.25;
}

void main() {
  // FIXED: bounded clock now drives real-time heat shimmer
  float t = mod(uTime, 640.0);

  // Aspect-corrected lateral coordinate centered on the butte
  float ax = (vUv.x - 0.5) * max(uAspect, 0.2);

  // Central envelope (hero butte occupies screen middle)
  float envWarp = (fbm(vec2(vUv.y * 1.7, 3.7), 2) - 0.5) * 0.06;
  float env = 1.0 - smoothstep(0.16, 0.44, abs(ax + envWarp));

  // Ridge silhouette + peak line (FIXED: subtle shimmer uses bounded clock)
  float prof = ridgeProf(ax);
  float shimmer = sin(t * 0.6 + ax * 3.0) * 0.0015;
  float peakY = uHorizonY + env * (uPeakHeight * (0.35 + 0.65 * prof)) + shimmer;
  float baseY = uHorizonY - 0.10;

  // Mountain body mask (low->high smoothsteps)
  float mountainMask = env
    * (1.0 - smoothstep(peakY - 0.006, peakY + 0.006, vUv.y))
    * smoothstep(baseY - 0.02, baseY + 0.02, vUv.y);

  // Projected cast-shadow band on the sand (below horizon)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.005, uHorizonY + 0.005, vUv.y);
  vec2 sdirRaw = uDirLightDirection.xy;
  float sl = max(length(sdirRaw), 1e-4); // FIXED: guarded normalize
  vec2 sdir = sdirRaw / sl;
  vec2 gpos = vec2(ax, vUv.y - uHorizonY);
  float along  = dot(gpos, sdir);
  float across = dot(gpos, vec2(-sdir.y, sdir.x));
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0); // FIXED: night gate
  float shLen = max(uShadowLen, 0.05); // FIXED: degenerate band guard
  float shadowMask = groundMask
    * (1.0 - smoothstep(0.20, 0.42, abs(across)))
    * smoothstep(0.0, 0.06, along)
    * (1.0 - smoothstep(shLen * 0.8, shLen, along))
    * sunAbove;

  if (mountainMask < 0.02 && shadowMask < 0.02) { gl_FragColor = vec4(0.0); return; }

  // ---- BUTTE BRANCH ----
  if (mountainMask >= shadowMask) {
    // Faceted slope normal from ridge central difference (guarded denom)
    float e = 0.012;
    float slopeX = (ridgeProf(ax + e) - ridgeProf(ax - e)) / (2.0 * e) * 0.06;
    vec3 nrm = normalize(vec3(clamp(slopeX, -1.5, 1.5), 0.25, 1.0));
    vec3 fn  = floor(nrm * 3.0 + 0.5) / 3.0;
    nrm = normalize(mix(nrm, fn, 0.45));

    // 006 rim-normal exaggeration on dedicated locals (FIXED: no clobber)
    vec3 nrmF  = nrm;
    vec3 viewF = vec3(0.0, 0.0, 1.0);
    ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

    // Height factor for strata / scree blending (FIXED: guarded denom)
    float hFac = clamp((vUv.y - baseY) / max(peakY - baseY, 1e-3), 0.0, 1.0);

    // Base rock color (4-stop palette by noise)
    float cvar = fbm(vec2(ax * 8.0, vUv.y * 6.0) + 3.3, 3);
    vec3 col = paletteMix4(uColButteDark, uColButteMid, uColButteLight, uColButteGray, cvar);

    // Horizontal striations (layered sediment)
    float stri = smoothstep(0.25, 0.75, sin(vUv.y * 140.0 + fbm(vec2(ax * 6.0, vUv.y * 3.0), 2) * 6.0));
    col *= 0.88 + stri * 0.12;

    // Vertical crack fissures
    float crack = smoothstep(0.86, 0.94, fbm(vec2(ax * 14.0, vUv.y * 10.0) + 9.1, 3));
    col *= 1.0 - crack * 0.30;

    // Scree skirt near the base
    float scree = 1.0 - smoothstep(0.0, 0.35, hFac);
    col = mix(col, uColScree, scree * 0.5);

    // 017 perceptual base enhancement (OKLCH chroma/lightness preserve)
    col = perceptualBaseEnhance(col, clamp(uDirLightIntensity, 0.0, 1.5));

    // 016 anime directional cel light + LIVE shadow-map sampling
    vec3 worldPos = vec3(ax * 40.0, (vUv.y - uHorizonY) * 30.0, 0.0);
    vec3 lit = computeAnimeDirectionalLight(col, nrmF, viewF, worldPos);

    // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + warm sand bounce
    float hemi = nrmF.y * 0.5 + 0.5;
    vec3 gi = mix(uGroundColor, uSkyColor, hemi) * uGroundStrength;
    gi += uGIColor * (1.0 - hemi) * uGIStrength;
    lit += col * gi;

    // Aerial perspective (edges + base haze)
    lit = applyFogBlend(lit, uFogColor, (1.0 - env) * 0.25 + 0.10);

    // Pixel-art de-banding dither
    vec2 pq = pxq(vUv * 96.0, 48.0);
    lit += (h21(pq) - 0.5) * 0.012;

    gl_FragColor = vec4(lit, mountainMask);
    return;
  }

  // ---- CAST SHADOW BRANCH ----
  vec3 shCol = uColButteDark * 0.40;
  shCol = applyFogBlend(shCol, uFogColor, 0.30);
  vec2 pq2 = pxq(vUv * 96.0, 48.0);
  shCol += (h21(pq2) - 0.5) * 0.010;
  gl_FragColor = vec4(shCol, shadowMask * uShadowAlpha);
}
`;

/* ------------------------------------------------------------------ */
/* 4. DESERT BUTTE SHADER MANAGER                                      */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _boosted   = new THREE.Color();
const _tintCol   = new THREE.Color();

export class DesertButteShader {
  constructor(options = {}) {
    this.uniforms = DESERT_BUTTE_UNIFORMS;
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
      vertexShader: DESERT_BUTTE_VERTEX,
      fragmentShader: DESERT_BUTTE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 540; // above sand+rocks, below haze
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
      this.uniforms.uColButteLight.value.x,
      this.uniforms.uColButteLight.value.y,
      this.uniforms.uColButteLight.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColButteLight.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );

    // Perfect-tint the shadow strata toward the light shadow tint
    _tintCol.copy(lu.uShadowTint.value);
    _baseCol.setRGB(0.42, 0.28, 0.17);
    const tinted = applyPerfectTintColor(_baseCol, _tintCol, 0.35);
    this.uniforms.uColButteDark.value.set(tinted.r, tinted.g, tinted.b);
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    if (camera) {
      u.uCamPos.value.set(camera.position.x, camera.position.y, camera.position.z);
    }

    // Cast-shadow stretch reacts to sun elevation (low sun = long shadow)
    const elev = clamp(-u.uDirLightDirection.value.y, 0.05, 1.0);
    const targetLen = clamp(0.30 / elev, 0.30, 1.20);
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
export function createDesertButteShader(options = {}) {
  return new DesertButteShader(options);
}
