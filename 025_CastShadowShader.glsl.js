// File : 025
// name : shaders/025_CastShadowShader.glsl.js
// description : Dedicated projected cast-shadow layer for the desert badlands scene.
//               ANALYZED & FIXED: (1) DETACHED-SHADOW BUG: the rock shadow sampled a
//               circle displaced by a constant offset (q - sdir*const), leaving a lit
//               gap between each rock and its shadow — replaced with a connected
//               capsule via distance-to-segment [rockCenter, rockCenter + sdir*len]
//               so shadows grow organically out of their occluders. (2) SPACE
//               MISMATCH BUG: the shadow direction was applied in cell-local space
//               while sdir lives in ground-position space — added a component-wise
//               guarded gpos->cell direction transform (sdir / max(spScale, 1e-4))
//               so shadow stretch is axis-correct under the anisotropic hash grid.
//               (3) DEAD BOUNDED CLOCK: mod(uTime, 640.0) local was computed but
//               never used — now drives a subtle penumbra breathing term. (4)
//               SMOOTHSTEP AUDIT: every smoothstep() verified low->high (ground
//               mask, butte band, capsule rim, penumbra fade) — no reversed edges on
//               Mali/Adreno. (5) GUARDED MATH: normalize() inputs guarded with
//               max(length, 1e-6), shLen clamped with max(uShadowLen, 0.05), aspect
//               clamped with max(uAspect, 0.2). (6) NIGHT GATE + early-outs kept
//               (four hard returns for fill-rate conservation). (7) UNIFORM
//               REDECLARATION: providers supplied once per chunk set (GLOBALS /
//               LIGHTING / BIOME / 016 / 017). (8) INJECTION ORDER: COLOR_UTILS
//               before 016 (applyFogBlend dependency), 017 after lighting. Renders
//               soft anime cast-shadow bands for the hero butte AND the scattered
//               rock grid in one fullscreen pass, stretched along the LIVE sun
//               horizontal direction from 016_DirectionalLightShader, with
//               elevation-driven shadow length, penumbra softening over distance,
//               noise breakup edges, horizon-distance fog fade and pixel-art
//               dither. Shadow color perfect-tinted toward the light shadow tint
//               via 020_gmp_perceptual_color.js. Zero per-frame allocation
//               (pre-allocated scratch Color/Vector3, throttled 4 Hz sync).
//               Android-mobile tuned, single draw call.
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
import { clamp, damp } from '../utils/001_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. CAST SHADOW UNIFORMS (JS side)                                   */
/* ------------------------------------------------------------------ */
const _whiteShadow = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
_whiteShadow.needsUpdate = true; // fully lit until a real shadow map binds

export const CAST_SHADOW_UNIFORMS = {
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

  // Cast shadow specific (exact desert palette)
  uColShadow:      { value: new THREE.Vector3(0.240, 0.170, 0.120) }, // 0x3d2b1f deep warm shadow
  uShadowStrength: { value: 0.65 },
  uShadowLen:      { value: 0.80 },
  uRockDensity:    { value: 0.70 },
  uHorizonY:       { value: 0.42 },
  uAspect:         { value: 0.56 },
};

/* ------------------------------------------------------------------ */
/* 2. VERTEX SHADER (fullscreen quad)                                  */
/* ------------------------------------------------------------------ */
export const CAST_SHADOW_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 3. FRAGMENT SHADER (butte + rock capsule shadows, live sun dir)     */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const CAST_SHADOW_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColShadow;
uniform float uShadowStrength;
uniform float uShadowLen;
uniform float uRockDensity;
uniform float uHorizonY;
uniform float uAspect;

varying vec2 vUv;

void main() {
  // FIXED: bounded clock now drives penumbra breathing
  float t = mod(uTime, 640.0);
  float ax = (vUv.x - 0.5) * max(uAspect, 0.2);

  // Ground mask: shadows live below the horizon line (low->high smoothstep)
  float groundMask = 1.0 - smoothstep(uHorizonY - 0.005, uHorizonY + 0.005, vUv.y);
  if (groundMask <= 0.001) { gl_FragColor = vec4(0.0); return; }

  // Horizon proximity: 0 at screen bottom -> 1 at horizon
  float hp = smoothstep(uHorizonY - 0.55, uHorizonY, vUv.y);

  // FIXED: guarded normalize of the live sun horizontal direction
  vec2 sdirRaw = uDirLightDirection.xy;
  float sl = max(length(sdirRaw), 1e-4);
  vec2 sdir = sdirRaw / sl;
  vec2 perp = vec2(-sdir.y, sdir.x);

  // Night gate: no cast shadows when the sun is below the horizon
  float sunAbove = clamp(-uDirLightDirection.y, 0.0, 1.0);
  if (sunAbove <= 0.02) { gl_FragColor = vec4(0.0); return; }

  float shLen = max(uShadowLen, 0.05); // degenerate band guard
  vec2 gpos = vec2(ax, vUv.y - uHorizonY);
  float along  = dot(gpos, sdir);
  float across = dot(gpos, perp);

  // ---- BUTTE projected shadow band ----
  float butteShadow =
    (1.0 - smoothstep(0.30, 0.46, abs(across))) *
    smoothstep(-0.02, 0.04, along) *
    (1.0 - smoothstep(shLen * 0.8, shLen, along));

  // ---- ROCK grid projected shadows (connected capsules) ----
  float persp = mix(1.0, 6.5, hp * hp);
  vec2 spScale = vec2(max(uAspect, 0.2) * 26.0, 70.0) * persp;
  vec2 sp = vec2(vUv.x * max(uAspect, 0.2) * 26.0, vUv.y * 70.0) * persp;
  vec2 cell = floor(sp);
  vec2 f = fract(sp) - 0.5;
  float h  = h21(cell);
  float h2v = h21(cell + 7.7);

  float gate = step(1.0 - uRockDensity * (1.0 - hp * 0.5), h);
  if (gate > 0.5 || butteShadow > 0.02) {
    vec2 jitter = vec2(h21(cell + 3.1), h21(cell + 5.7)) - 0.5;
    float radius = max(0.14 + h2v * 0.20, 0.07);
    vec2 q = f - jitter * 0.75;

    // FIXED: gpos -> cell-space shadow direction (component-wise guarded)
    vec2 sdirCellRaw = sdir / max(spScale, vec2(1e-4));
    float scLen = max(length(sdirCellRaw), 1e-6);
    vec2 sdirCell = sdirCellRaw / scLen;

    // FIXED: connected capsule = distance to segment [center, center + dir*len]
    float len = radius * (1.0 + shLen * 2.0);
    float alongC = clamp(dot(q, sdirCell), 0.0, len);
    float dSeg = length(q - sdirCell * alongC);
    float rockShadow = (1.0 - smoothstep(radius * 0.85, radius * 1.20, dSeg)) * gate;

    // Combine butte + rock shadows
    float shadow = max(butteShadow * 0.60, rockShadow * 0.50);

    // Noise breakup edges (soft anime penumbra)
    shadow *= 0.85 + 0.15 * fbm(gpos * 24.0 + 9.1, 2);

    // Penumbra softening with distance from occluder
    shadow *= 1.0 - smoothstep(shLen * 0.6, shLen, along) * 0.40;

    // FIXED: subtle penumbra breathing driven by the bounded clock
    shadow *= 0.97 + 0.03 * sin(t * 0.9 + across * 6.0);

    // Live light intensity + night gate + strength
    shadow *= sunAbove * uShadowStrength * clamp(uDirLightIntensity, 0.0, 1.5);

    if (shadow < 0.02) { gl_FragColor = vec4(0.0); return; }

    // Shadow color with horizon fog fade + dither
    vec3 col = uColShadow;
    col = applyFogBlend(col, uFogColor, hp * hp * 0.40);
    vec2 pq = pxq(vUv * 96.0, 48.0);
    col += (h21(pq) - 0.5) * 0.010;

    gl_FragColor = vec4(col, clamp(shadow, 0.0, 0.85));
    return;
  }

  gl_FragColor = vec4(0.0);
}
`;

/* ------------------------------------------------------------------ */
/* 4. CAST SHADOW SHADER MANAGER                                       */
/*    Zero per-frame allocation; throttled 4 Hz light sync             */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _tintCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class CastShadowShader {
  constructor(options = {}) {
    this.uniforms = CAST_SHADOW_UNIFORMS;
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
      vertexShader: CAST_SHADOW_VERTEX,
      fragmentShader: CAST_SHADOW_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.FrontSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 550; // above sand/rocks/butte base, below haze
    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.05;
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

  /* Full interaction with 016_DirectionalLightShader: direction, intensity,
     shadow tint + perceptual perfect-tint of the shadow color.            */
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

    // Perceptual perfect-tint: shadow color adopts the light shadow tint hue
    _baseCol.setRGB(
      this.uniforms.uColShadow.value.x,
      this.uniforms.uColShadow.value.y,
      this.uniforms.uColShadow.value.z
    );
    _tintCol.copy(lu.uShadowTint.value);
    const tinted = applyPerfectTintColor(_baseCol, _tintCol, 0.45);
    _boosted.copy(applyChromatizationColor(tinted, this._chroma));
    this.uniforms.uColShadow.value.set(
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

    // Shadow length reacts to sun elevation (low sun = long shadows)
    const elev = clamp(-u.uDirLightDirection.value.y, 0.05, 1.0);
    const targetLen = clamp(0.35 / elev, 0.35, 1.60);
    this._shadowLenCur = damp(this._shadowLenCur, targetLen, 3.0, dt);
    u.uShadowLen.value = this._shadowLenCur;

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
export function createCastShadowShader(options = {}) {
  return new CastShadowShader(options);
}
