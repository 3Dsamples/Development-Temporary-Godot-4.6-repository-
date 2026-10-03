// File : 14
// name : shaders/TriplanarRockShader.js
// description : Triplanar rock material for cel-shaded anime scenes. Projects procedural rock albedo and roughness across world XZ/Y/Z planes, blending by surface normal. Works without UVs, ideal for procedurally generated rock meshes. Includes cel-shaded lighting bands, rim light, ambient occlusion from vertex color, and distance fog. Optimized for Android mobile with precomputed triplanar weights.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';
import { GLSL_SIMPLEX_2D, GLSL_FBM_2D } from '../utils/003_ProceduralNoise.js';

// ---------------------------------------------------------------------------
// TRIPLANAR VERTEX SHADER
// ---------------------------------------------------------------------------
export const TRIPLANAR_VERTEX_SHADER = `
precision highp float;

attribute vec3 color;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;

void main() {
  vColor = color;

  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPos = worldPos.xyz;

  vec4 viewPos = viewMatrix * worldPos;
  vViewDir = normalize(cameraPosition - worldPos.xyz);

  vNormal = normalize(normalMatrix * normal);
  vDepth = -viewPos.z;

  gl_Position = projectionMatrix * viewPos;
}
`;

// ---------------------------------------------------------------------------
// TRIPLANAR FRAGMENT SHADER
// ---------------------------------------------------------------------------
export const TRIPLANAR_FRAGMENT_SHADER = `
precision highp float;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vColor;
varying float vDepth;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
uniform vec3 uBounceColor;
uniform vec3 uRimColor;

uniform vec3 uRockLight;
uniform vec3 uRockMid;
uniform vec3 uRockDark;
uniform vec3 uRockShadow;

uniform float uBand1;
uniform float uBand2;
uniform float uBand3;
uniform float uShadowStrength;
uniform vec3 uShadowTint;

uniform float uSpecPower;
uniform float uSpecStrength;
uniform vec3 uSpecColor;

uniform float uRimPower;
uniform float uRimStrength;

uniform float uAOStrength;

uniform vec3 uFogColor;
uniform float uFogNear;
uniform float uFogFar;

uniform float uTextureScale;
uniform float uRoughnessBias;
uniform float uMacroScale;
uniform float uDetailScale;
uniform float uBlendSharpness;

${GLSL_SIMPLEX_2D}
${GLSL_FBM_2D}

float toonDiffuse(float ndotl, float b1, float b2, float b3) {
  if (ndotl > b3) return 1.0;
  if (ndotl > b2) return 0.75;
  if (ndotl > b1) return 0.45;
  return 0.18;
}

vec3 triplanarBlend(vec3 n) {
  vec3 w = pow(abs(n), vec3(uBlendSharpness));
  float sum = w.x + w.y + w.z;
  w /= max(sum, 0.0001);
  return w;
}

float rockPattern(vec2 uv, float seed) {
  float macro = fbm(uv * uMacroScale + seed, 3, 2.0, 0.5);
  float detail = fbm(uv * uDetailScale + seed * 7.3, 2, 2.0, 0.5);
  float veins = abs(snoise(uv * uDetailScale * 0.7 + seed * 3.1));
  return macro * 0.5 + detail * 0.3 + veins * 0.2;
}

vec3 rockColorFromPattern(float p) {
  float t = clamp(p * 0.5 + 0.5, 0.0, 1.0);
  if (t < 0.33) return mix(uRockShadow, uRockDark, t * 3.0);
  if (t < 0.66) return mix(uRockDark, uRockMid, (t - 0.33) * 3.0);
  return mix(uRockMid, uRockLight, (t - 0.66) * 3.0);
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  vec3 L = normalize(uSunDir);

  if (dot(N, V) < 0.0) N = -N;

  vec3 blend = triplanarBlend(N);

  vec2 uvXZ = vWorldPos.xz * uTextureScale;
  vec2 uvXY = vWorldPos.xy * uTextureScale;
  vec2 uvZY = vWorldPos.zy * uTextureScale;

  float pXZ = rockPattern(uvXZ, 0.0);
  float pXY = rockPattern(uvXY, 3.7);
  float pZY = rockPattern(uvZY, 7.1);

  float pattern = pXZ * blend.y + pXY * blend.z + pZY * blend.x;

  vec3 baseColor = rockColorFromPattern(pattern);

  vec3 vColMod = mix(vec3(1.0), vColor * 1.2, 0.5);
  baseColor *= vColMod;

  float ndotl = dot(N, L) * 0.5 + 0.5;
  float toon = toonDiffuse(ndotl, uBand1, uBand2, uBand3);

  vec3 ambient = uSkyColor * 0.35;
  float bounceFactor = max(-N.y, 0.0) * 0.3;
  vec3 bounce = uBounceColor * bounceFactor;

  vec3 lit = baseColor * uSunColor * toon + baseColor * ambient + baseColor * bounce;

  float shadowMask = 1.0 - toon;
  lit = mix(lit, lit * uShadowTint * 1.4, shadowMask * uShadowStrength);

  float roughness = clamp(0.5 + uRoughnessBias + pattern * 0.3, 0.2, 1.0);
  float specPower = uSpecPower / roughness;

  vec3 halfVec = normalize(L + V);
  float ndoth = max(dot(N, halfVec), 0.0);
  float spec = pow(ndoth, specPower);
  spec = smoothstep(0.5, 0.55, spec) * uSpecStrength;
  lit += uSpecColor * spec;

  float rim = 1.0 - max(dot(N, V), 0.0);
  rim = pow(rim, uRimPower);
  rim = smoothstep(0.3, 0.7, rim);
  lit += uRimColor * rim * uRimStrength;

  float vLum = dot(vColor, vec3(0.299, 0.587, 0.114));
  lit *= mix(1.0, vLum, uAOStrength);

  float fogFactor = smoothstep(uFogNear, uFogFar, vDepth);
  lit = mix(lit, uFogColor, fogFactor);

  gl_FragColor = vec4(lit, 1.0);
}
`;

// ---------------------------------------------------------------------------
// UNIFORM FACTORY — used by both the material and its LOD clones
// ---------------------------------------------------------------------------
function createBaseUniforms() {
  return {
    uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
    uSunColor: { value: new THREE.Color(1.0, 0.96, 0.85) },
    uSkyColor: { value: new THREE.Color(0.45, 0.62, 0.85) },
    uBounceColor: { value: new THREE.Color(0.15, 0.20, 0.25) },
    uRimColor: { value: new THREE.Color(0.90, 0.95, 1.0) },

    uRockLight: { value: new THREE.Color(0.78, 0.72, 0.60) },
    uRockMid: { value: new THREE.Color(0.62, 0.56, 0.45) },
    uRockDark: { value: new THREE.Color(0.34, 0.30, 0.24) },
    uRockShadow: { value: new THREE.Color(0.15, 0.13, 0.115) },

    uBand1: { value: 0.32 },
    uBand2: { value: 0.52 },
    uBand3: { value: 0.72 },
    uShadowStrength: { value: 0.7 },
    uShadowTint: { value: new THREE.Color(0.14, 0.18, 0.28) },

    uSpecPower: { value: 16.0 },
    uSpecStrength: { value: 0.15 },
    uSpecColor: { value: new THREE.Color(0.8, 0.8, 0.8) },

    uRimPower: { value: 2.5 },
    uRimStrength: { value: 0.35 },

    uAOStrength: { value: 0.6 },

    uFogColor: { value: new THREE.Color(0.35, 0.55, 0.70) },
    uFogNear: { value: 8.0 },
    uFogFar: { value: 45.0 },

    uTextureScale: { value: 0.35 },
    uRoughnessBias: { value: 0.0 },
    uMacroScale: { value: 2.5 },
    uDetailScale: { value: 12.0 },
    uBlendSharpness: { value: 4.0 }
  };
}

// ---------------------------------------------------------------------------
// TRIPLANAR ROCK MATERIAL
// ---------------------------------------------------------------------------
export class TriplanarRockMaterial extends THREE.ShaderMaterial {
  constructor(params = {}) {
    const {
      side = THREE.FrontSide,
      transparent = false,
      depthWrite = true,
      textureScale,
      roughnessBias,
      macroScale,
      detailScale,
      blendSharpness
    } = params;

    const uniforms = createBaseUniforms();

    if (textureScale !== undefined) uniforms.uTextureScale.value = textureScale;
    if (roughnessBias !== undefined) uniforms.uRoughnessBias.value = roughnessBias;
    if (macroScale !== undefined) uniforms.uMacroScale.value = macroScale;
    if (detailScale !== undefined) uniforms.uDetailScale.value = detailScale;
    if (blendSharpness !== undefined) uniforms.uBlendSharpness.value = blendSharpness;

    super({
      vertexShader: TRIPLANAR_VERTEX_SHADER,
      fragmentShader: TRIPLANAR_FRAGMENT_SHADER,
      uniforms,
      vertexColors: true,
      side,
      transparent,
      depthWrite,
      lights: false,
      fog: false
    });

    this.userData.isTriplanarRockMaterial = true;
  }

  setSun(dir, color) {
    if (dir) this.uniforms.uSunDir.value.copy(dir).normalize();
    if (color) this.uniforms.uSunColor.value.copy(color);
    return this;
  }

  setSky(color) {
    if (color) this.uniforms.uSkyColor.value.copy(color);
    return this;
  }

  setBounce(color) {
    if (color) this.uniforms.uBounceColor.value.copy(color);
    return this;
  }

  setRockColors(light, mid, dark, shadow) {
    if (light) this.uniforms.uRockLight.value.copy(light);
    if (mid) this.uniforms.uRockMid.value.copy(mid);
    if (dark) this.uniforms.uRockDark.value.copy(dark);
    if (shadow) this.uniforms.uRockShadow.value.copy(shadow);
    return this;
  }

  setToonBands(b1, b2, b3) {
    if (b1 !== undefined) this.uniforms.uBand1.value = b1;
    if (b2 !== undefined) this.uniforms.uBand2.value = b2;
    if (b3 !== undefined) this.uniforms.uBand3.value = b3;
    return this;
  }

  setShadow(strength, tint) {
    if (strength !== undefined) this.uniforms.uShadowStrength.value = strength;
    if (tint) this.uniforms.uShadowTint.value.copy(tint);
    return this;
  }

  setRim(color, power, strength) {
    if (color) this.uniforms.uRimColor.value.copy(color);
    if (power !== undefined) this.uniforms.uRimPower.value = power;
    if (strength !== undefined) this.uniforms.uRimStrength.value = strength;
    return this;
  }

  setFog(color, near, far) {
    if (color) this.uniforms.uFogColor.value.copy(color);
    if (near !== undefined) this.uniforms.uFogNear.value = near;
    if (far !== undefined) this.uniforms.uFogFar.value = far;
    return this;
  }

  setTextureScale(scale) {
    if (scale !== undefined) this.uniforms.uTextureScale.value = scale;
    return this;
  }

  setBlendSharpness(sharpness) {
    if (sharpness !== undefined) this.uniforms.uBlendSharpness.value = sharpness;
    return this;
  }

  cloneTriplanar() {
    const mat = new TriplanarRockMaterial();
    const src = this.uniforms;
    const dst = mat.uniforms;

    dst.uSunDir.value.copy(src.uSunDir.value);
    dst.uSunColor.value.copy(src.uSunColor.value);
    dst.uSkyColor.value.copy(src.uSkyColor.value);
    dst.uBounceColor.value.copy(src.uBounceColor.value);
    dst.uRimColor.value.copy(src.uRimColor.value);

    dst.uRockLight.value.copy(src.uRockLight.value);
    dst.uRockMid.value.copy(src.uRockMid.value);
    dst.uRockDark.value.copy(src.uRockDark.value);
    dst.uRockShadow.value.copy(src.uRockShadow.value);

    dst.uBand1.value = src.uBand1.value;
    dst.uBand2.value = src.uBand2.value;
    dst.uBand3.value = src.uBand3.value;
    dst.uShadowStrength.value = src.uShadowStrength.value;
    dst.uShadowTint.value.copy(src.uShadowTint.value);

    dst.uSpecPower.value = src.uSpecPower.value;
    dst.uSpecStrength.value = src.uSpecStrength.value;
    dst.uSpecColor.value.copy(src.uSpecColor.value);

    dst.uRimPower.value = src.uRimPower.value;
    dst.uRimStrength.value = src.uRimStrength.value;

    dst.uAOStrength.value = src.uAOStrength.value;

    dst.uFogColor.value.copy(src.uFogColor.value);
    dst.uFogNear.value = src.uFogNear.value;
    dst.uFogFar.value = src.uFogFar.value;

    dst.uTextureScale.value = src.uTextureScale.value;
    dst.uRoughnessBias.value = src.uRoughnessBias.value;
    dst.uMacroScale.value = src.uMacroScale.value;
    dst.uDetailScale.value = src.uDetailScale.value;
    dst.uBlendSharpness.value = src.uBlendSharpness.value;

    mat.side = this.side;
    mat.transparent = this.transparent;
    mat.depthWrite = this.depthWrite;

    return mat;
  }
}

// ---------------------------------------------------------------------------
// PRESETS
// ---------------------------------------------------------------------------
export function createGraniteMaterial() {
  const mat = new TriplanarRockMaterial({ textureScale: 0.28 });
  mat.setRockColors(
    new THREE.Color(0.78, 0.72, 0.60),
    new THREE.Color(0.62, 0.56, 0.45),
    new THREE.Color(0.34, 0.30, 0.24),
    new THREE.Color(0.15, 0.13, 0.115)
  );
  mat.setToonBands(0.32, 0.52, 0.72);
  mat.setShadow(0.7, new THREE.Color(0.14, 0.18, 0.28));
  mat.setRim(new THREE.Color(0.9, 0.95, 1.0), 2.5, 0.35);
  mat.setFog(new THREE.Color(0.35, 0.55, 0.70), 8.0, 45.0);
  return mat;
}

export function createSandstoneMaterial() {
  const mat = new TriplanarRockMaterial({ textureScale: 0.22, roughnessBias: 0.15 });
  mat.setRockColors(
    new THREE.Color(0.82, 0.68, 0.48),
    new THREE.Color(0.68, 0.54, 0.36),
    new THREE.Color(0.42, 0.32, 0.22),
    new THREE.Color(0.20, 0.15, 0.11)
  );
  mat.setToonBands(0.34, 0.54, 0.74);
  mat.setShadow(0.65, new THREE.Color(0.18, 0.14, 0.20));
  mat.setRim(new THREE.Color(0.95, 0.90, 0.80), 2.8, 0.30);
  mat.setFog(new THREE.Color(0.38, 0.50, 0.62), 8.0, 45.0);
  return mat;
}

export function createLimestoneMaterial() {
  const mat = new TriplanarRockMaterial({ textureScale: 0.32, roughnessBias: -0.05 });
  mat.setRockColors(
    new THREE.Color(0.84, 0.82, 0.74),
    new THREE.Color(0.66, 0.64, 0.56),
    new THREE.Color(0.38, 0.36, 0.30),
    new THREE.Color(0.18, 0.17, 0.14)
  );
  mat.setToonBands(0.30, 0.50, 0.70);
  mat.setShadow(0.72, new THREE.Color(0.12, 0.16, 0.24));
  mat.setRim(new THREE.Color(0.92, 0.94, 1.0), 2.2, 0.38);
  mat.setFog(new THREE.Color(0.35, 0.55, 0.70), 8.0, 45.0);
  return mat;
}

export function createDarkBasaltMaterial() {
  const mat = new TriplanarRockMaterial({ textureScale: 0.40, roughnessBias: 0.25 });
  mat.setRockColors(
    new THREE.Color(0.45, 0.42, 0.40),
    new THREE.Color(0.30, 0.28, 0.26),
    new THREE.Color(0.16, 0.15, 0.14),
    new THREE.Color(0.08, 0.07, 0.07)
  );
  mat.setToonBands(0.36, 0.56, 0.76);
  mat.setShadow(0.75, new THREE.Color(0.10, 0.12, 0.20));
  mat.setRim(new THREE.Color(0.70, 0.80, 0.95), 3.0, 0.40);
  mat.setFog(new THREE.Color(0.30, 0.45, 0.60), 8.0, 45.0);
  return mat;
}

// ---------------------------------------------------------------------------
// MATERIAL REGISTRY
// ---------------------------------------------------------------------------
export const TriplanarRockMaterialRegistry = {
  granite: null,
  sandstone: null,
  limestone: null,
  basalt: null,

  init() {
    this.granite = createGraniteMaterial();
    this.sandstone = createSandstoneMaterial();
    this.limestone = createLimestoneMaterial();
    this.basalt = createDarkBasaltMaterial();
  },

  updateSun(dir, color) {
    if (this.granite) this.granite.setSun(dir, color);
    if (this.sandstone) this.sandstone.setSun(dir, color);
    if (this.limestone) this.limestone.setSun(dir, color);
    if (this.basalt) this.basalt.setSun(dir, color);
  },

  updateSky(color) {
    if (this.granite) this.granite.setSky(color);
    if (this.sandstone) this.sandstone.setSky(color);
    if (this.limestone) this.limestone.setSky(color);
    if (this.basalt) this.basalt.setSky(color);
  },

  updateFog(color, near, far) {
    if (this.granite) this.granite.setFog(color, near, far);
    if (this.sandstone) this.sandstone.setFog(color, near, far);
    if (this.limestone) this.limestone.setFog(color, near, far);
    if (this.basalt) this.basalt.setFog(color, near, far);
  },

  dispose() {
    if (this.granite) this.granite.dispose();
    if (this.sandstone) this.sandstone.dispose();
    if (this.limestone) this.limestone.dispose();
    if (this.basalt) this.basalt.dispose();
    this.granite = null;
    this.sandstone = null;
    this.limestone = null;
    this.basalt = null;
  }
};

// ---------------------------------------------------------------------------
// LOD MATERIAL FACTORY
// ---------------------------------------------------------------------------
export function createTriplanarLODMaterial(baseMaterial, lodLevel) {
  const mat = baseMaterial.cloneTriplanar();

  if (lodLevel >= 2) {
    mat.setToonBands(0.35, 0.55, 0.75);
    mat.setRim(null, 2.0, 0.2);
    mat.uniforms.uMacroScale.value = 1.5;
    mat.uniforms.uDetailScale.value = 6.0;
    mat.uniforms.uBlendSharpness.value = 2.0;
  } else if (lodLevel === 1) {
    mat.uniforms.uDetailScale.value = 8.0;
    mat.uniforms.uBlendSharpness.value = 3.0;
  }

  return mat;
}

export default {
  TriplanarRockMaterial,
  TRIPLANAR_VERTEX_SHADER,
  TRIPLANAR_FRAGMENT_SHADER,
  createGraniteMaterial,
  createSandstoneMaterial,
  createLimestoneMaterial,
  createDarkBasaltMaterial,
  createTriplanarLODMaterial,
  TriplanarRockMaterialRegistry
};