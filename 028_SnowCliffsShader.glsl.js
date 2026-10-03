// File : 028
// name : shaders/028_SnowCliffsShader.glsl.js
// description : Anime snowy canyon cliff shader for the frozen gorge scene. ANALYZED
//               & FIXED: (1) FALSE-DEPTH SHADOW BUG: fullscreen parallax cliffs have
//               no real scene geometry, so binding the live directional shadow map to
//               reconstructed fake world positions can produce incorrect self-darkening.
//               Scene shadow-map binding is now opt-in via options.sceneShadows; by
//               default the layer uses a neutral 1x1 white shadow texture while still
//               fully interacting with 016 directional cel lighting, shadow tint, rim
//               light, ambient, and global illumination. (2) COLOR COPY BUG: Color ->
//               Vector3 copies previously used .copy(Color), which reads undefined
//               x/y/z on some Three.js paths. All Color sources are now explicitly
//               written into Vector3 uniforms with set(r,g,b). (3) WIND POP BUG:
//               snow ledge drift now uses an integrated 2pi wind phase uniform instead
//               of multiplying live wind by raw time, preventing rescale jumps when
//               wind changes. (4) NOISE RANGE BUG: palette/crack/ledge noise values
//               were assumed 0..1 but fbm can be signed; all palette-driving noise is
//               now normalized/clamped to 0..1. (5) OVERBRIGHT BUG: GI + ice glow +
//               rim can exceed 1 on mobile framebuffers; final color is clamped after
//               dither. (6) CULLING BUG: fullscreen quad used FrontSide and could be
//               culled depending on PlaneGeometry winding; material is now DoubleSide.
//               (7) FILL-RATE BUG: empty canyon interior now uses discard instead of
//               writing transparent black. (8) INSTANCE SHARING BUG: uniforms are
//               cloned per manager instance so multiple scenes cannot mutate the shared
//               default uniform object. (9) SMOOTHSTEP AUDIT: every smoothstep() is
//               low->high. (10) GUARDED MATH: aspect, edge separation, normalize inputs,
//               divisions, and wind phase wrapping are guarded. Renders left/right dark
//               blue-gray faceted cliff walls with wind-phase snow ledges, inner-rim
//               snow caps, vertical striations, crack fissures, cyan ice-bounce GI from
//               the canyon water, cel-quantized directional lighting via
//               016_DirectionalLightShader.glsl.js, 017 perceptual base enhancement
//               (OKLCH), 006 rim-normal exaggeration, 020_gmp_perceptual_color.js
//               chromatization / perfect-tint wrappers, 021_gmp_ies_lighting.js IES
//               auto-conversion hooks, and 001_gmp_MathUtils.js scalar helpers. Composed
//               on shaders/000_BaseShader.glsl.js with correct chunk injection order
//               (COLOR_UTILS before 016; 017 perceptual enhancer after lighting). Zero
//               per-frame allocation (pre-allocated scratch Vector3/Color, throttled
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
import { clamp, damp, TWO_PI } from '../utils/001_gmp_MathUtils.js';

/* ------------------------------------------------------------------ */
/* 1. NEUTRAL SHADOW TEXTURE                                           */
/*    Used when sceneShadows=false to keep 016 shadow sampling defined */
/*    without false-depth artifacts on fullscreen parallax geometry.   */
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
/* 2. SNOW CLIFFS DEFAULT UNIFORMS (JS side)                           */
/* ------------------------------------------------------------------ */
export const SNOW_CLIFFS_UNIFORMS = {
  // GLSL_GLOBALS providers
  uTime:     { value: 0.0 },
  uPPU:      { value: 5.5 },
  uWind:     { value: new THREE.Vector2(0.0, 0.0) },
  uCamPos:   { value: new THREE.Vector3(0.0, 0.0, 0.0) },
  uViewDir:  { value: new THREE.Vector3(0.0, 0.0, 1.0) },

  // GLSL_LIGHTING providers (GI hemisphere + cel ambient)
  uSunDir:           { value: new THREE.Vector3(0.35, 0.78, 0.52) },
  uMoonDir:          { value: new THREE.Vector3(-0.35, -0.78, -0.52) },
  uSunColor:         { value: new THREE.Vector3(0.96, 0.98, 1.00) },
  uMoonColor:        { value: new THREE.Vector3(0.38, 0.48, 0.72) },
  uSkyColor:         { value: new THREE.Vector3(0.52, 0.70, 0.92) },
  uGroundColor:      { value: new THREE.Vector3(0.18, 0.24, 0.32) },
  uFogColor:         { value: new THREE.Vector3(0.76, 0.86, 0.94) },
  uShadowTintColor:  { value: new THREE.Vector3(0.16, 0.24, 0.36) },
  uRimColor:         { value: new THREE.Vector3(0.88, 0.96, 1.00) },
  uSunIntensity:     { value: 1.0 },
  uMoonIntensity:    { value: 0.35 },
  uSkyStrength:      { value: 0.40 },
  uGroundStrength:   { value: 0.25 },
  uBounceStrength:   { value: 0.35 },
  uRimStrength:      { value: 0.45 },
  uShadingSteps:     { value: 4.0 },

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
  uRimIntensity:      { value: 0.45 },
  uShadowTint:        { value: new THREE.Color(0.16, 0.24, 0.36) },
  uShadowSoftness:    { value: 0.05 },
  uAmbientIntensity:  { value: 0.28 },
  uAmbientColor:      { value: new THREE.Color(0.52, 0.70, 0.92) },
  uShadowMap:         { value: _whiteShadow },
  uShadowMatrix:      { value: new THREE.Matrix4() },
  uShadowBias:        { value: -0.001 },
  uShadowNormalBias:  { value: 0.05 },
  uShadowMapSize:     { value: new THREE.Vector2(1, 1) },

  // Snow cliffs specific (exact image palette)
  uColCliffDeep:   { value: new THREE.Vector3(0.105, 0.145, 0.190) }, // 0x1b2530 deep shadow rock
  uColCliffDark:   { value: new THREE.Vector3(0.180, 0.245, 0.310) }, // 0x2e3e4f dark blue-gray
  uColCliffMid:    { value: new THREE.Vector3(0.290, 0.370, 0.450) }, // 0x4a5e73 mid slate
  uColCliffLight:  { value: new THREE.Vector3(0.430, 0.520, 0.610) }, // 0x6e859c lit edge
  uColSnow:        { value: new THREE.Vector3(0.920, 0.960, 1.000) }, // 0xebf5ff bright snow
  uColSnowShadow:  { value: new THREE.Vector3(0.690, 0.790, 0.900) }, // 0xb0c9e6 blue snow shadow
  uColIceGlow:     { value: new THREE.Vector3(0.290, 0.850, 0.720) }, // 0x4ad9b8 turquoise ice glow
  uGIColor:        { value: new THREE.Vector3(0.220, 0.620, 0.560) }, // cyan water bounce
  uGIStrength:     { value: 0.42 },
  uHorizonY:       { value: 0.50 },
  uAspect:         { value: 0.56 },
  uSnowLine:       { value: 0.58 },
  uGlowStrength:   { value: 0.55 },
  uWindPhase:      { value: 0.0 },
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

function _cloneSnowCliffsUniforms(src) {
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
export const SNOW_CLIFFS_VERTEX = /* glsl */`
${GLSL_GLOBALS}

varying vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.9993, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. FRAGMENT SHADER (faceted snowy canyon cliffs + GI + 016 light)   */
/*    Chunk order: COLOR_UTILS before 016; 017 after lighting          */
/* ------------------------------------------------------------------ */
export const SNOW_CLIFFS_FRAGMENT = /* glsl */`
${GLSL_GLOBALS}
${GLSL_NOISE}
${GLSL_COLOR_UTILS}
${GLSL_NORMAL_QUANT}
${GLSL_LIGHTING}
${GLSL_BIOME}
${GLSL_PERCEPTUAL_BASE_ENHANCER}
${GLSL_DIRECTIONAL_LIGHT}

uniform vec3  uColCliffDeep;
uniform vec3  uColCliffDark;
uniform vec3  uColCliffMid;
uniform vec3  uColCliffLight;
uniform vec3  uColSnow;
uniform vec3  uColSnowShadow;
uniform vec3  uColIceGlow;
uniform vec3  uGIColor;
uniform float uGIStrength;
uniform float uHorizonY;
uniform float uAspect;
uniform float uSnowLine;
uniform float uGlowStrength;
uniform float uWindPhase;

varying vec2 vUv;

void main() {
  // Integrated 2pi wind phase: continuous under changing wind, no rescale pop.
  float driftPh = mod(uWindPhase, 6.2831853);

  float ax = (vUv.x - 0.5) * max(uAspect, 0.2);
  float depth = clamp(1.0 - vUv.y, 0.0, 1.0);

  // Canyon opening: narrower at the top, wider near the camera.
  float halfOpen = mix(0.10, 0.30, depth);
  float openNoise = clamp(fbm(vec2(vUv.y * 2.5, 5.1), 2) * 0.5 + 0.5, 0.0, 1.0);
  halfOpen += (openNoise - 0.5) * 0.05;
  halfOpen = clamp(halfOpen, 0.06, 0.38);

  // Jagged inner cliff edges (static noise, no unbounded fbm drift).
  float nLraw = fbm(vec2(vUv.y * 6.0, 17.0), 3);
  float nRraw = fbm(vec2(vUv.y * 6.0, 29.0), 3);
  float nL = (clamp(nLraw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 0.18;
  float nR = (clamp(nRraw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 0.18;

  float leftEdge = -halfOpen + nL;
  float rightEdge = halfOpen + nR;
  if (rightEdge <= leftEdge + 0.02) {
    rightEdge = leftEdge + 0.02;
  }

  // Left/right wall masks (all smoothstep edges low->high).
  float leftMask = 1.0 - smoothstep(leftEdge - 0.012, leftEdge + 0.012, vUv.x);
  float rightMask = smoothstep(rightEdge - 0.012, rightEdge + 0.012, vUv.x);
  float wallMask = max(leftMask, rightMask);
  if (wallMask < 0.02) discard;

  // Branchless side sign: +1 left wall faces right, -1 right wall faces left.
  float side = clamp((leftMask - rightMask) * 8.0, -1.0, 1.0);
  float edgeDist = min(abs(vUv.x - leftEdge), abs(vUv.x - rightEdge));

  // Faceted slope normal (z=1.0 guarantees normalize safety).
  float slopeRaw = fbm(vec2(vUv.x * 10.0, vUv.y * 4.0) + 3.3, 3);
  float slopeNoise = clamp(slopeRaw * 0.8, -0.7, 0.7);
  vec3 nrm = normalize(vec3(
    side * 0.42 + slopeNoise * 0.25,
    0.18,
    1.0
  ));

  // Vertical striations + crack fissures + roughness (normalized to 0..1).
  float strRaw = fbm(vec2(vUv.x * 5.0, vUv.y * 2.0) + 31.7, 2);
  float strWarp = (clamp(strRaw * 0.5 + 0.5, 0.0, 1.0) - 0.5) * 4.0;
  float str = smoothstep(0.25, 0.75, sin(vUv.y * 160.0 + strWarp));

  float crackRaw = fbm(vec2(vUv.x * 16.0, vUv.y * 9.0) + 9.1, 3);
  float crack = smoothstep(0.82, 0.94, clamp(crackRaw * 0.5 + 0.5, 0.0, 1.0));

  float roughRaw = fbm(vec2(vUv.x * 8.0, vUv.y * 6.0) + 21.7, 3);
  float rough = clamp(roughRaw * 0.5 + 0.5, 0.0, 1.0);

  vec3 rock = paletteMix4(
    uColCliffDeep,
    uColCliffDark,
    uColCliffMid,
    uColCliffLight,
    rough
  );
  rock *= 0.88 + str * 0.18;
  rock *= 1.0 - crack * 0.35;

  // Wind-phase snow ledges + inner-rim snow caps.
  float ledgeRaw = fbm(vec2(vUv.x * 4.0, vUv.y * 10.0) + 31.0, 3);
  float ledgeNoise = clamp(ledgeRaw * 0.5 + 0.5, 0.0, 1.0);
  float ledgeBand = sin(vUv.y * 110.0 + ledgeNoise * 8.0 + driftPh) * 0.5 + 0.5;
  float ledge = smoothstep(uSnowLine, uSnowLine + 0.14, ledgeBand);
  ledge *= smoothstep(0.08, 0.35, vUv.y);

  float rimSnow = 1.0 - smoothstep(0.0, 0.05, edgeDist);
  float snowMask = clamp(max(ledge * 0.80, rimSnow * 0.55) * wallMask, 0.0, 1.0);

  // Alpine biome boosts snow coverage.
  snowMask *= 0.65 + 0.35 * clamp(uBiomeW.w, 0.0, 1.0);

  vec3 snowCol = mix(
    uColSnowShadow,
    uColSnow,
    smoothstep(0.0, 0.75, nrm.y + ledgeNoise * 0.35)
  );

  vec3 col = mix(rock, snowCol, snowMask);

  // 017 perceptual base enhancement (OKLCH chroma/lightness preserve).
  col = perceptualBaseEnhance(col, clamp(uDirLightIntensity, 0.0, 1.5));

  // 006 rim-normal exaggeration on dedicated locals (no GI clobber).
  vec3 nrmF = nrm;
  vec3 viewF = vec3(0.0, 0.0, 1.0);
  ${getAnimeRimNormalExaggerationGLSL('nrmF', 'viewF', 'uRimPower')}

  // 016 anime directional cel light. Scene shadow map is neutral unless opted in.
  vec3 worldPos = vec3(ax * 70.0, (vUv.y - uHorizonY) * 55.0, side * 12.0);
  vec3 lit = computeAnimeDirectionalLight(col, nrmF, viewF, worldPos);

  // GLOBAL ILLUMINATION: hemisphere sky/ground bounce + cyan water bounce.
  float hemi = nrmF.y * 0.5 + 0.5;
  vec3 gi = mix(uGroundColor, uSkyColor, hemi) * uGroundStrength;
  float nearWater = 1.0 - smoothstep(0.0, 0.34, abs(ax));
  gi += uGIColor * nearWater * uGIStrength;
  lit += col * gi;

  // Turquoise ice glow reflected onto lower inner walls.
  float glow = nearWater * (1.0 - smoothstep(0.25, 0.75, vUv.y)) * uGlowStrength;
  lit += uColIceGlow * glow * 0.22 * clamp(uDirLightIntensity, 0.0, 1.5);

  // Bottom contact darkening + top aerial haze.
  lit *= 1.0 - (1.0 - smoothstep(0.0, 0.22, vUv.y)) * 0.25;
  float haze = smoothstep(0.55, 1.0, vUv.y) * 0.25;
  lit = applyFogBlend(lit, uFogColor, haze);

  // Pixel-art de-banding dither + final mobile-safe clamp.
  vec2 pq = pxq(vUv * 112.0, 56.0);
  lit += (h21(pq) - 0.5) * 0.010;
  lit = clamp(lit, 0.0, 1.0);

  float alpha = clamp(wallMask * smoothstep(0.0, 0.08, wallMask), 0.0, 1.0);
  gl_FragColor = vec4(lit, alpha);
}
`;

/* ------------------------------------------------------------------ */
/* 6. SNOW CLIFFS SHADER MANAGER                                       */
/*    Zero per-frame allocation; throttled 4 Hz light/GI sync          */
/* ------------------------------------------------------------------ */
const _sunDirTmp = new THREE.Vector3();
const _baseCol   = new THREE.Color();
const _tintCol   = new THREE.Color();
const _boosted   = new THREE.Color();

export class SnowCliffsShader {
  constructor(options = {}) {
    this.uniforms = _cloneSnowCliffsUniforms(SNOW_CLIFFS_UNIFORMS);
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
      vertexShader: SNOW_CLIFFS_VERTEX,
      fragmentShader: SNOW_CLIFFS_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 560; // above desert stack, below snow ground/ice water

    this._syncAccum = 1.0;
    this._chroma = options.chroma !== undefined ? options.chroma : 1.08;
    this._iesScale = 1.0;
    this._windPhase = 0.0;
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

  setSnowLine(v) {
    this.uniforms.uSnowLine.value = clamp(v, 0.35, 0.85);
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
    u.uSunDir.value.copy(_sunDirTmp).negate();

    const dc = lu.uDirLightColor.value;
    u.uDirLightColor.value.copy(dc);
    u.uSunColor.value.set(dc.r, dc.g, dc.b);

    u.uDirLightIntensity.value = lu.uDirLightIntensity.value * this._iesScale;
    u.uSunIntensity.value = lu.uDirLightIntensity.value;

    const st = lu.uShadowTint.value;
    u.uShadowTint.value.copy(st);
    u.uShadowTintColor.value.set(st.r, st.g, st.b);

    const ac = lu.uAmbientColor.value;
    u.uAmbientColor.value.copy(ac);

    if (lu.uRimColor && lu.uRimColor.value) {
      const rc = lu.uRimColor.value;
      if (rc.isColor) {
        u.uRimColor.value.set(rc.r, rc.g, rc.b);
      } else if (rc.isVector3) {
        u.uRimColor.value.copy(rc);
      }
    }

    u.uCelSteps.value = lu.uCelSteps.value;
    u.uRimPower.value = lu.uRimPower.value;
    u.uRimIntensity.value = lu.uRimIntensity.value;
    u.uShadowSoftness.value = lu.uShadowSoftness.value;
    u.uAmbientIntensity.value = lu.uAmbientIntensity.value;

    // Background parallax cliffs use neutral shadows by default.
    // Enable options.sceneShadows=true only if a real 3D cliff proxy exists.
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

    // Perceptual vibrancy boost on snow highlight (020_gmp_perceptual_color).
    _baseCol.setRGB(
      this.uniforms.uColSnow.value.x,
      this.uniforms.uColSnow.value.y,
      this.uniforms.uColSnow.value.z
    );
    _boosted.copy(applyChromatizationColor(_baseCol, this._chroma));
    this.uniforms.uColSnow.value.set(
      clamp(_boosted.r, 0.0, 1.0),
      clamp(_boosted.g, 0.0, 1.0),
      clamp(_boosted.b, 0.0, 1.0)
    );

    // Perfect-tint the deep cliff shadow toward the live shadow tint.
    _tintCol.copy(lu.uShadowTint.value);
    _baseCol.setRGB(0.105, 0.145, 0.190);
    const tinted = applyPerfectTintColor(_baseCol, _tintCol, 0.35);
    this.uniforms.uColCliffDeep.value.set(tinted.r, tinted.g, tinted.b);
  }

  update(dt, elapsed, lightShader, camera) {
    const u = this.uniforms;
    u.uTime.value = elapsed;

    // Integrated 2pi wind phase (continuous under changing wind, no fbm drift pop).
    this._windPhase = (this._windPhase + u.uWind.value.x * dt * 0.25) % TWO_PI;
    if (this._windPhase < 0.0) this._windPhase += TWO_PI;
    u.uWindPhase.value = this._windPhase;

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
export function createSnowCliffsShader(options = {}) {
  return new SnowCliffsShader(options);
}

export default {
  SnowCliffsShader,
  createSnowCliffsShader,
  SNOW_CLIFFS_UNIFORMS,
  SNOW_CLIFFS_VERTEX,
  SNOW_CLIFFS_FRAGMENT,
};
