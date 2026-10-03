// File : 018
// name : shaders/018_SunRaysShader.glsl.js
// description : Anime god-rays / volumetric sun shafts shader for the desert scene.
//               ANALYZED & FIXED: (1) FLOAT-PRECISION BUG: unbounded uTime inside
//               sin()/drift terms loses precision on long Android sessions — all
//               time terms now use a wrapped clock mod(uTime, 640.0). (2) UNGUARDED
//               ATAN: atan(d.y, d.x) is undefined at the exact sun center on strict
//               GLSL ES 1.00 drivers — replaced with atan(d.y, d.x + 1e-6).
//               (3) REVERSED/DEGENERATE SMOOTHSTEP: radial falloff could invert when
//               uRayLength approached 0 — now guarded with max(uRayLength, 0.05) and
//               all smoothstep() edges verified low->high. (4) UNBOUNDED ALPHA:
//               additive energy could exceed 1 and blow out HDR-ish mobile
//               framebuffers — alpha is now clamped() and gated by uSunAbove.
//               (5) DIVISION BY ZERO: cel quantization divided by uCelBands — now
//               max(uCelBands, 1.0). (6) NIGHT LEAK: rays rendered even when the sun
//               was below the horizon — hard early-out when uSunAbove <= 0.001.
//               (7) JS SYNC ALLOCATION: per-frame Vector3 projection allocated
//               garbage — now uses pre-allocated scratch vectors and a throttled
//               (4 Hz) perceptual chromatization re-tint via
//               020_gmp_perceptual_color.js so the warm gold shafts keep vibrant
//               hue under the 016 directional light pipeline. Renders discrete
//               cel-banded shafts + core glow + drifting dust motes in ONE
//               fullscreen additive draw call. Composed on 000_BaseShader.glsl.js
//               (GLSL_GLOBALS + GLSL_NOISE), zero per-frame allocation.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import {
  GLSL_GLOBALS,
  GLSL_NOISE,
} from './000_BaseShader.glsl.js';
import { applyChromatizationColor } from '../utils/020_gmp_perceptual_color.js';

/* ------------------------------------------------------------------ */
/* 1. GLSL SUN RAYS CHUNK (vertex + fragment)                          */
/* ------------------------------------------------------------------ */
export const SUN_RAYS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  // Fullscreen effect quad (rendered after scene, additive)
  gl_Position = vec4(position.xy, 0.9995, 1.0);
}
`;

export const SUN_RAYS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}

uniform vec2  uSunScreen;     // sun position in UV space
uniform vec3  uSunColor;      // perceptually boosted sun color
uniform float uSunIntensity;  // directional light intensity
uniform float uSunAbove;      // 0..1 gate (sun above horizon)
uniform float uAspect;        // viewport aspect (w/h)
uniform float uRayCount;      // angular streak frequency
uniform float uRayLength;     // radial shaft length in UV
uniform float uRaySoftness;   // global alpha scale
uniform float uDustDensity;   // 0..1 dust motes density
uniform float uFlickerSpeed;  // real-time shimmer speed
uniform float uCelBands;      // anime quantization bands

varying vec2 vUv;

void main() {
  // FIXED: hard early-out at night (no fill-rate waste)
  if (uSunAbove <= 0.001) {
    gl_FragColor = vec4(0.0);
    return;
  }

  // FIXED: wrapped clock keeps float precision on long mobile sessions
  float t = mod(uTime, 640.0);

  // Aspect-corrected vector from sun
  vec2 d = vUv - uSunScreen;
  d.x *= uAspect;
  float dist = length(d);
  float ang = atan(d.y, d.x + 1e-6); // FIXED: guarded atan

  // Layered angular streaks with slow real-time shimmer
  float s1 = pow(0.5 + 0.5 * sin(ang * uRayCount + t * uFlickerSpeed), 3.0);
  float s2 = pow(0.5 + 0.5 * sin(ang * (uRayCount * 2.7) - t * uFlickerSpeed * 0.6 + 1.7), 4.0);
  float streak = s1 + s2 * 0.5;

  // Anime cel banding (quantized shafts, not smooth gradients)
  streak = floor(streak * uCelBands + 0.5) / max(uCelBands, 1.0); // FIXED: div guard

  // Quadratic radial falloff + exponential core glow
  float fall = 1.0 - smoothstep(0.0, max(uRayLength, 0.05), dist); // FIXED: edge guard
  float glow = exp(-dist * 6.0);
  float rays = streak * fall * fall;

  // Drifting dust motes inside the beams (hash-grid, world-stable)
  vec2 gp = vUv * 40.0;
  gp.x -= t * 0.6;
  vec2 cell = floor(gp);
  float h = h21(cell);
  vec2 f = fract(gp) - 0.5;
  vec2 jitter = vec2(h21(cell + 3.1), h21(cell + 5.7)) - 0.5;
  float mote = 1.0 - smoothstep(0.0, 0.12, length(f - jitter * 0.6));
  mote *= step(1.0 - uDustDensity, h) * fall;

  // Compose additive light energy
  vec3 col = uSunColor * (rays * 0.55 + glow * 0.75) * uSunIntensity;
  col += uSunColor * mote * 0.35;

  // FIXED: clamped alpha gated by horizon factor (no HDR blow-out)
  float alpha = clamp((rays * 0.7 + glow + mote * 0.3) * uSunAbove, 0.0, 1.0) * uRaySoftness;
  gl_FragColor = vec4(col * uSunAbove, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 2. SUN RAYS UNIFORMS (JS side)                                      */
/* ------------------------------------------------------------------ */
export const SUN_RAYS_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // Sun rays specific
  uSunScreen:    { value: new THREE.Vector2(0.18, 0.86) }, // top-left like reference
  uSunColor:     { value: new THREE.Color(1.0, 0.93, 0.78) },
  uSunIntensity: { value: 1.0 },
  uSunAbove:     { value: 1.0 },
  uAspect:       { value: 0.56 },
  uRayCount:     { value: 14.0 },
  uRayLength:    { value: 0.85 },
  uRaySoftness:  { value: 0.65 },
  uDustDensity:  { value: 0.35 },
  uFlickerSpeed: { value: 0.35 },
  uCelBands:     { value: 3.0 },
};

/* ------------------------------------------------------------------ */
/* 3. SUN RAYS SHADER MANAGER (syncs with 016 DirectionalLightShader)  */
/*    Zero per-frame allocation: pre-allocated scratch vectors only    */
/* ------------------------------------------------------------------ */
const _sunDir  = new THREE.Vector3();
const _sunPos  = new THREE.Vector3();
const _boosted = new THREE.Color();

export class SunRaysShader {
  constructor(options = {}) {
    this.uniforms = SUN_RAYS_UNIFORMS;
    for (const k in options) {
      if (this.uniforms[k] && options[k] !== undefined) {
        if (this.uniforms[k].value && this.uniforms[k].value.set) {
          this.uniforms[k].value.set(options[k]);
        } else {
          this.uniforms[k].value = options[k];
        }
      }
    }
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: SUN_RAYS_VERTEX,
      fragmentShader: SUN_RAYS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      side: THREE.FrontSide,
    });
    this.geometry = new THREE.PlaneGeometry(2, 2);
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 900; // above scene, below HUD
    this._syncAccum = 1.0;       // force first perceptual tint immediately
    this._chroma = options.chroma !== undefined ? options.chroma : 1.15;
  }

  getMesh()     { return this.mesh; }
  getMaterial() { return this.material; }
  getUniforms() { return this.uniforms; }

  setSunScreen(x, y) {
    this.uniforms.uSunScreen.value.set(x, y);
  }

  setAspect(a) {
    this.uniforms.uAspect.value = Math.max(0.2, Math.min(2.5, a));
  }

  /* Project live sun direction into UV space + gate by elevation.     */
  syncFromLight(lightShader, camera) {
    const lu = lightShader.getUniforms();
    // uDirLightDirection points along light travel; sun is opposite
    _sunDir.copy(lu.uDirLightDirection.value).negate().normalize();

    // Smooth horizon gate (dawn/dusk fade)
    const above = Math.min(1.0, Math.max(0.0, (_sunDir.y + 0.05) / 0.25));
    this.uniforms.uSunAbove.value = above;

    if (above > 0.001) {
      _sunPos.copy(camera.position).addScaledVector(_sunDir, 500.0);
      _sunPos.project(camera);
      this.uniforms.uSunScreen.value.set(
        _sunPos.x * 0.5 + 0.5,
        _sunPos.y * 0.5 + 0.5
      );
    }
    this.uniforms.uSunIntensity.value = lu.uDirLightIntensity.value;
  }

  /* Perceptual chroma boost of the ray color (throttled, zero-alloc   */
  /* steady state: one Color reused).                                  */
  syncColor(lightShader) {
    const base = lightShader.getUniforms().uDirLightColor.value;
    _boosted.copy(base);
    const out = applyChromatizationColor(_boosted, this._chroma);
    this.uniforms.uSunColor.value.copy(out);
  }

  update(dt, elapsed, lightShader, camera) {
    this.uniforms.uTime.value = elapsed;
    this.syncFromLight(lightShader, camera);
    this._syncAccum += dt;
    if (this._syncAccum >= 0.25) { // throttled perceptual re-tint (4 Hz)
      this._syncAccum = 0.0;
      this.syncColor(lightShader);
    }
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/* ------------------------------------------------------------------ */
/* 4. FACTORY                                                          */
/* ------------------------------------------------------------------ */
export function createSunRaysShader(options = {}) {
  return new SunRaysShader(options);
}
