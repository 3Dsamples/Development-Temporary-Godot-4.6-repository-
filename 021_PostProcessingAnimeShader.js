// File : 21
// name : shaders/PostProcessingAnimeShader.js
// description : Anime-style post-processing shader chain for Three.js r185. Provides cel-shaded edge detection, bloom, color grading, vignette, chromatic aberration, halftone, and anime outline passes. All effects exposed as individual ShaderMaterial instances compatible with EffectComposer. Mobile-optimized for Android with PERF_TIER-driven pass enable flags and low-resolution bloom downsampling. No external dependencies, uses only Three.js r185 ShaderMaterial and WebGLRenderTarget.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  clamp,
  mix,
  smoothstep,
  PERF_TIER
} from '../utils/001_MathUtils.js';

// ---------------------------------------------------------------------------
// FULLSCREEN QUAD VERTEX SHADER
// ---------------------------------------------------------------------------
export const FULLSCREEN_VERTEX_SHADER = `
precision highp float;

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ---------------------------------------------------------------------------
// SOBEL EDGE DETECTION — anime outline
// ---------------------------------------------------------------------------
export const SOBEL_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform vec2 uResolution;
uniform float uEdgeStrength;
uniform float uEdgeThreshold;
uniform vec3 uOutlineColor;
uniform float uDepthEdgeStrength;
uniform float uNormalEdgeStrength;
uniform float uThickness;

float sobel(sampler2D tex, vec2 uv, vec2 texel, float scale) {
  float tl = texture2D(tex, uv + vec2(-texel.x, -texel.y) * scale).r;
  float t  = texture2D(tex, uv + vec2( 0.0,    -texel.y) * scale).r;
  float tr = texture2D(tex, uv + vec2( texel.x, -texel.y) * scale).r;
  float l  = texture2D(tex, uv + vec2(-texel.x,  0.0)   * scale).r;
  float r  = texture2D(tex, uv + vec2( texel.x,  0.0)   * scale).r;
  float bl = texture2D(tex, uv + vec2(-texel.x,  texel.y) * scale).r;
  float b  = texture2D(tex, uv + vec2( 0.0,     texel.y) * scale).r;
  float br = texture2D(tex, uv + vec2( texel.x,  texel.y) * scale).r;

  float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
  float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

  return sqrt(gx * gx + gy * gy);
}

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  vec2 texel = 1.0 / uResolution;
  float scale = uThickness;

  float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

  float edgeLuma = sobel(tDiffuse, vUv, texel, scale);
  float edgeDepth = sobel(tDepth, vUv, texel, scale * 0.9);
  float edgeNormal = sobel(tNormal, vUv, texel, scale * 0.85);

  float edge = edgeLuma * uEdgeStrength;
  edge += edgeDepth * uDepthEdgeStrength;
  edge += edgeNormal * uNormalEdgeStrength;

  edge = smoothstep(uEdgeThreshold, uEdgeThreshold + 0.08, edge);

  vec3 outlineColor = mix(color.rgb, uOutlineColor, edge);

  gl_FragColor = vec4(outlineColor, color.a);
}
`;

// ---------------------------------------------------------------------------
// BLOOM — bright pass + blur + composite
// ---------------------------------------------------------------------------
export const BLOOM_EXTRACT_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uThreshold;
uniform float uSoftKnee;
uniform float uIntensity;

void main() {
  vec4 color = texture2D(tDiffuse, vUv);
  float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

  float knee = uThreshold * uSoftKnee + 0.0001;
  float soft = clamp(lum - uThreshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);

  float contribution = max(soft, lum - uThreshold) / max(lum, 0.0001);

  vec3 bloomColor = color.rgb * contribution * uIntensity;
  gl_FragColor = vec4(bloomColor, 1.0);
}
`;

export const BLOOM_BLUR_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform vec2 uDirection;
uniform float uRadius;

void main() {
  vec2 texel = uDirection * uRadius;

  vec3 sum = vec3(0.0);
  sum += texture2D(tDiffuse, vUv - texel * 4.0).rgb * 0.0162;
  sum += texture2D(tDiffuse, vUv - texel * 3.0).rgb * 0.0540;
  sum += texture2D(tDiffuse, vUv - texel * 2.0).rgb * 0.1216;
  sum += texture2D(tDiffuse, vUv - texel * 1.0).rgb * 0.1946;
  sum += texture2D(tDiffuse, vUv).rgb * 0.2270;
  sum += texture2D(tDiffuse, vUv + texel * 1.0).rgb * 0.1946;
  sum += texture2D(tDiffuse, vUv + texel * 2.0).rgb * 0.1216;
  sum += texture2D(tDiffuse, vUv + texel * 3.0).rgb * 0.0540;
  sum += texture2D(tDiffuse, vUv + texel * 4.0).rgb * 0.0162;

  gl_FragColor = vec4(sum, 1.0);
}
`;

export const BLOOM_COMPOSITE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform sampler2D tBloom;
uniform float uBloomStrength;
uniform vec3 uBloomTint;

void main() {
  vec4 base = texture2D(tDiffuse, vUv);
  vec3 bloom = texture2D(tBloom, vUv).rgb * uBloomTint;

  vec3 result = base.rgb + bloom * uBloomStrength;

  gl_FragColor = vec4(result, base.a);
}
`;

// ---------------------------------------------------------------------------
// COLOR GRADING — anime saturation, contrast, temperature
// ---------------------------------------------------------------------------
export const COLOR_GRADE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uSaturation;
uniform float uContrast;
uniform float uBrightness;
uniform float uTemperature;
uniform float uTint;
uniform vec3 uLiftColor;
uniform vec3 uGainColor;
uniform float uLiftAmount;
uniform float uGainAmount;

vec3 applyLift(vec3 color, vec3 lift, float amount) {
  return color + lift * amount;
}

vec3 applyGain(vec3 color, vec3 gain, float amount) {
  return color * mix(vec3(1.0), gain, amount);
}

vec3 applyTemperature(vec3 color, float temp, float tint) {
  color.r += temp * 0.1;
  color.b -= temp * 0.1;
  color.g += tint * 0.1;
  color.r -= tint * 0.05;
  color.b -= tint * 0.05;
  return color;
}

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  vec3 c = color.rgb;

  c = applyTemperature(c, uTemperature, uTint);

  float lum = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(lum), c, uSaturation);

  c = (c - 0.5) * uContrast + 0.5;
  c = c + uBrightness;

  c = applyLift(c, uLiftColor, uLiftAmount);
  c = applyGain(c, uGainColor, uGainAmount);

  c = clamp(c, 0.0, 1.0);

  gl_FragColor = vec4(c, color.a);
}
`;

// ---------------------------------------------------------------------------
// VIGNETTE — anime soft frame darkening
// ---------------------------------------------------------------------------
export const VIGNETTE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uVignetteStrength;
uniform float uVignetteSoftness;
uniform vec3 uVignetteColor;

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  vec2 center = vUv - 0.5;
  float dist = length(center) * 1.414;

  float vignette = smoothstep(0.85 - uVignetteSoftness, 0.85, dist);
  vignette *= uVignetteStrength;

  vec3 graded = mix(color.rgb, uVignetteColor, vignette);

  gl_FragColor = vec4(graded, color.a);
}
`;

// ---------------------------------------------------------------------------
// CHROMATIC ABERRATION — anime-style subtle color fringing
// ---------------------------------------------------------------------------
export const CHROMATIC_ABERRATION_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uAberrationStrength;
uniform vec2 uAberrationCenter;

void main() {
  vec2 dir = vUv - uAberrationCenter;

  float r = texture2D(tDiffuse, vUv + dir * uAberrationStrength).r;
  float g = texture2D(tDiffuse, vUv).g;
  float b = texture2D(tDiffuse, vUv - dir * uAberrationStrength).b;

  vec4 color = texture2D(tDiffuse, vUv);

  gl_FragColor = vec4(r, g, b, color.a);
}
`;

// ---------------------------------------------------------------------------
// FILM GRAIN — subtle noise for anime texture
// ---------------------------------------------------------------------------
export const FILM_GRAIN_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uTime;
uniform float uGrainStrength;
uniform float uGrainScale;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  vec2 grainUV = vUv * uGrainScale + uTime;
  float grain = hash(grainUV) - 0.5;

  vec3 grained = color.rgb + grain * uGrainStrength;

  gl_FragColor = vec4(grained, color.a);
}
`;

// ---------------------------------------------------------------------------
// HALFTONE — anime print dot pattern
// ---------------------------------------------------------------------------
export const HALFTONE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uHalftoneScale;
uniform float uHalftoneStrength;
uniform float uHalftoneAngle;
uniform float uHalftoneCutoff;

vec2 rotate(vec2 uv, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
}

void main() {
  vec4 color = texture2D(tDiffuse, vUv);
  float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

  vec2 rotated = rotate(vUv * uHalftoneScale, uHalftoneAngle);

  vec2 cell = fract(rotated) - 0.5;
  float dist = length(cell);

  float radius = (1.0 - lum) * 0.7 * uHalftoneCutoff;
  float dotMask = smoothstep(radius + 0.05, radius - 0.05, dist);

  vec3 halftoneColor = color.rgb * mix(1.0, dotMask * 1.6, uHalftoneStrength);

  gl_FragColor = vec4(halftoneColor, color.a);
}
`;

// ---------------------------------------------------------------------------
// SCANLINES — retro anime CRT effect (optional)
// ---------------------------------------------------------------------------
export const SCANLINE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform vec2 uResolution;
uniform float uScanlineIntensity;
uniform float uScanlineCount;

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  float scan = sin(vUv.y * uScanlineCount * 3.14159) * 0.5 + 0.5;
  scan = mix(1.0, scan, uScanlineIntensity);

  gl_FragColor = vec4(color.rgb * scan, color.a);
}
`;

// ---------------------------------------------------------------------------
// TOON COLOR QUANTIZATION — reduce color bands
// ---------------------------------------------------------------------------
export const TOON_QUANTIZE_FRAGMENT_SHADER = `
precision highp float;

varying vec2 vUv;

uniform sampler2D tDiffuse;
uniform float uBands;
uniform float uStrength;

void main() {
  vec4 color = texture2D(tDiffuse, vUv);

  vec3 quantized = floor(color.rgb * uBands + 0.5) / uBands;
  vec3 result = mix(color.rgb, quantized, uStrength);

  gl_FragColor = vec4(result, color.a);
}
`;

// ---------------------------------------------------------------------------
// ANIME MATERIAL PRESETS
// ---------------------------------------------------------------------------
export class PostProcessingAnimeShader {
  constructor(options = {}) {
    this.width = options.width || window.innerWidth;
    this.height = options.height || window.innerHeight;
    this.perfTier = PERF_TIER;

    this.passes = {
      edge: null,
      bloomExtract: null,
      bloomBlurH: null,
      bloomBlurV: null,
      bloomComposite: null,
      colorGrade: null,
      vignette: null,
      chromatic: null,
      grain: null,
      halftone: null,
      scanline: null,
      toonQuantize: null
    };

    this.edgeStrength = 0.65;
    this.edgeThreshold = 0.22;
    this.outlineColor = new THREE.Color(0.05, 0.06, 0.10);
    this.depthEdgeStrength = 0.55;
    this.normalEdgeStrength = 0.35;
    this.thickness = 1.0;

    this.bloomThreshold = 0.78;
    this.bloomSoftKnee = 0.55;
    this.bloomIntensity = 1.15;
    this.bloomStrength = 0.55;
    this.bloomTint = new THREE.Color(1.0, 0.96, 0.92);

    this.saturation = 1.15;
    this.contrast = 1.06;
    this.brightness = 0.0;
    this.temperature = 0.05;
    this.tint = 0.0;
    this.liftColor = new THREE.Color(0.02, 0.03, 0.06);
    this.gainColor = new THREE.Color(1.02, 1.00, 0.98);
    this.liftAmount = 0.5;
    this.gainAmount = 0.4;

    this.vignetteStrength = 0.35;
    this.vignetteSoftness = 0.55;
    this.vignetteColor = new THREE.Color(0.0, 0.02, 0.06);

    this.aberrationStrength = 0.0022;
    this.aberrationCenter = new THREE.Vector2(0.5, 0.5);

    this.grainStrength = 0.04;
    this.grainScale = 480.0;

    this.halftoneScale = 180.0;
    this.halftoneStrength = 0.18;
    this.halftoneAngle = 0.45;
    this.halftoneCutoff = 0.85;

    this.scanlineIntensity = 0.06;
    this.scanlineCount = 1080.0;

    this.bands = 6.0;
    this.quantizeStrength = 0.22;

    this.enableEdge = options.enableEdge !== undefined ? options.enableEdge : true;
    this.enableBloom = options.enableBloom !== undefined ? options.enableBloom : true;
    this.enableColorGrade = options.enableColorGrade !== undefined ? options.enableColorGrade : true;
    this.enableVignette = options.enableVignette !== undefined ? options.enableVignette : true;
    this.enableChromatic = options.enableChromatic !== undefined ? options.enableChromatic : (PERF_TIER !== 'LOW');
    this.enableGrain = options.enableGrain !== undefined ? options.enableGrain : false;
    this.enableHalftone = options.enableHalftone !== undefined ? options.enableHalftone : false;
    this.enableScanline = options.enableScanline !== undefined ? options.enableScanline : false;
    this.enableQuantize = options.enableQuantize !== undefined ? options.enableQuantize : (PERF_TIER === 'HIGH');

    this.enabled = true;

    this._time = 0.0;
    this._resolution = new THREE.Vector2(this.width, this.height);
  }

  // -------------------------------------------------------------------------
  // BUILD ALL PASSES
  // -------------------------------------------------------------------------
  build() {
    this.passes.edge = this._createEdgePass();
    this.passes.bloomExtract = this._createBloomExtractPass();
    this.passes.bloomBlurH = this._createBloomBlurPass(1, 0);
    this.passes.bloomBlurV = this._createBloomBlurPass(0, 1);
    this.passes.bloomComposite = this._createBloomCompositePass();
    this.passes.colorGrade = this._createColorGradePass();
    this.passes.vignette = this._createVignettePass();
    this.passes.chromatic = this._createChromaticPass();
    this.passes.grain = this._createGrainPass();
    this.passes.halftone = this._createHalftonePass();
    this.passes.scanline = this._createScanlinePass();
    this.passes.toonQuantize = this._createQuantizePass();

    return this;
  }

  // -------------------------------------------------------------------------
  // PASS FACTORIES
  // -------------------------------------------------------------------------
  _createEdgePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: SOBEL_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        tDepth: { value: null },
        tNormal: { value: null },
        uResolution: { value: this._resolution },
        uEdgeStrength: { value: this.edgeStrength },
        uEdgeThreshold: { value: this.edgeThreshold },
        uOutlineColor: { value: this.outlineColor.clone() },
        uDepthEdgeStrength: { value: this.depthEdgeStrength },
        uNormalEdgeStrength: { value: this.normalEdgeStrength },
        uThickness: { value: this.thickness }
      }
    });
  }

  _createBloomExtractPass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: BLOOM_EXTRACT_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uThreshold: { value: this.bloomThreshold },
        uSoftKnee: { value: this.bloomSoftKnee },
        uIntensity: { value: this.bloomIntensity }
      }
    });
  }

  _createBloomBlurPass(dx, dy) {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: BLOOM_BLUR_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uDirection: { value: new THREE.Vector2(dx, dy) },
        uRadius: { value: 1.0 }
      }
    });
  }

  _createBloomCompositePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: BLOOM_COMPOSITE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        tBloom: { value: null },
        uBloomStrength: { value: this.bloomStrength },
        uBloomTint: { value: this.bloomTint.clone() }
      }
    });
  }

  _createColorGradePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: COLOR_GRADE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uSaturation: { value: this.saturation },
        uContrast: { value: this.contrast },
        uBrightness: { value: this.brightness },
        uTemperature: { value: this.temperature },
        uTint: { value: this.tint },
        uLiftColor: { value: this.liftColor.clone() },
        uGainColor: { value: this.gainColor.clone() },
        uLiftAmount: { value: this.liftAmount },
        uGainAmount: { value: this.gainAmount }
      }
    });
  }

  _createVignettePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: VIGNETTE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uVignetteStrength: { value: this.vignetteStrength },
        uVignetteSoftness: { value: this.vignetteSoftness },
        uVignetteColor: { value: this.vignetteColor.clone() }
      }
    });
  }

  _createChromaticPass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: CHROMATIC_ABERRATION_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uAberrationStrength: { value: this.aberrationStrength },
        uAberrationCenter: { value: this.aberrationCenter.clone() }
      }
    });
  }

  _createGrainPass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: FILM_GRAIN_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uTime: { value: 0.0 },
        uGrainStrength: { value: this.grainStrength },
        uGrainScale: { value: this.grainScale }
      }
    });
  }

  _createHalftonePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: HALFTONE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uHalftoneScale: { value: this.halftoneScale },
        uHalftoneStrength: { value: this.halftoneStrength },
        uHalftoneAngle: { value: this.halftoneAngle },
        uHalftoneCutoff: { value: this.halftoneCutoff }
      }
    });
  }

  _createScanlinePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: SCANLINE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uResolution: { value: this._resolution },
        uScanlineIntensity: { value: this.scanlineIntensity },
        uScanlineCount: { value: this.scanlineCount }
      }
    });
  }

  _createQuantizePass() {
    return new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERTEX_SHADER,
      fragmentShader: TOON_QUANTIZE_FRAGMENT_SHADER,
      uniforms: {
        tDiffuse: { value: null },
        uBands: { value: this.bands },
        uStrength: { value: this.quantizeStrength }
      }
    });
  }

  // -------------------------------------------------------------------------
  // UPDATE UNIFORMS
  // -------------------------------------------------------------------------
  updateTime(elapsed) {
    this._time = elapsed;
    if (this.passes.grain) {
      this.passes.grain.uniforms.uTime.value = elapsed * 0.1;
    }
  }

  onResize(width, height) {
    this.width = width;
    this.height = height;
    this._resolution.set(width, height);
    if (this.passes.scanline) {
      this.passes.scanline.uniforms.uScanlineCount.value = height * 1.0;
    }
  }

  // -------------------------------------------------------------------------
  // SETTERS
  // -------------------------------------------------------------------------
  setEdgeStrength(strength, threshold) {
    if (strength !== undefined) this.edgeStrength = strength;
    if (threshold !== undefined) this.edgeThreshold = threshold;
    if (this.passes.edge) {
      this.passes.edge.uniforms.uEdgeStrength.value = this.edgeStrength;
      this.passes.edge.uniforms.uEdgeThreshold.value = this.edgeThreshold;
    }
    return this;
  }

  setOutlineColor(color) {
    if (color) {
      this.outlineColor.copy(color);
      if (this.passes.edge) this.passes.edge.uniforms.uOutlineColor.value.copy(color);
    }
    return this;
  }

  setBloomIntensity(intensity, threshold, softKnee) {
    if (intensity !== undefined) this.bloomIntensity = intensity;
    if (threshold !== undefined) this.bloomThreshold = threshold;
    if (softKnee !== undefined) this.bloomSoftKnee = softKnee;
    if (this.passes.bloomExtract) {
      this.passes.bloomExtract.uniforms.uIntensity.value = this.bloomIntensity;
      this.passes.bloomExtract.uniforms.uThreshold.value = this.bloomThreshold;
      this.passes.bloomExtract.uniforms.uSoftKnee.value = this.bloomSoftKnee;
    }
    return this;
  }

  setBloomStrength(strength, tint) {
    if (strength !== undefined) this.bloomStrength = strength;
    if (tint) this.bloomTint.copy(tint);
    if (this.passes.bloomComposite) {
      this.passes.bloomComposite.uniforms.uBloomStrength.value = this.bloomStrength;
      this.passes.bloomComposite.uniforms.uBloomTint.value.copy(this.bloomTint);
    }
    return this;
  }

  setColorGrade(saturation, contrast, brightness, temperature, tint) {
    if (saturation !== undefined) this.saturation = saturation;
    if (contrast !== undefined) this.contrast = contrast;
    if (brightness !== undefined) this.brightness = brightness;
    if (temperature !== undefined) this.temperature = temperature;
    if (tint !== undefined) this.tint = tint;
    if (this.passes.colorGrade) {
      const u = this.passes.colorGrade.uniforms;
      u.uSaturation.value = this.saturation;
      u.uContrast.value = this.contrast;
      u.uBrightness.value = this.brightness;
      u.uTemperature.value = this.temperature;
      u.uTint.value = this.tint;
    }
    return this;
  }

  setVignette(strength, softness, color) {
    if (strength !== undefined) this.vignetteStrength = strength;
    if (softness !== undefined) this.vignetteSoftness = softness;
    if (color) this.vignetteColor.copy(color);
    if (this.passes.vignette) {
      const u = this.passes.vignette.uniforms;
      u.uVignetteStrength.value = this.vignetteStrength;
      u.uVignetteSoftness.value = this.vignetteSoftness;
      u.uVignetteColor.value.copy(this.vignetteColor);
    }
    return this;
  }

  setChromaticAberration(strength) {
    if (strength !== undefined) this.aberrationStrength = strength;
    if (this.passes.chromatic) {
      this.passes.chromatic.uniforms.uAberrationStrength.value = this.aberrationStrength;
    }
    return this;
  }

  setGrain(strength, scale) {
    if (strength !== undefined) this.grainStrength = strength;
    if (scale !== undefined) this.grainScale = scale;
    if (this.passes.grain) {
      this.passes.grain.uniforms.uGrainStrength.value = this.grainStrength;
      this.passes.grain.uniforms.uGrainScale.value = this.grainScale;
    }
    return this;
  }

  setHalftone(scale, strength, angle, cutoff) {
    if (scale !== undefined) this.halftoneScale = scale;
    if (strength !== undefined) this.halftoneStrength = strength;
    if (angle !== undefined) this.halftoneAngle = angle;
    if (cutoff !== undefined) this.halftoneCutoff = cutoff;
    if (this.passes.halftone) {
      const u = this.passes.halftone.uniforms;
      u.uHalftoneScale.value = this.halftoneScale;
      u.uHalftoneStrength.value = this.halftoneStrength;
      u.uHalftoneAngle.value = this.halftoneAngle;
      u.uHalftoneCutoff.value = this.halftoneCutoff;
    }
    return this;
  }

  setQuantize(bands, strength) {
    if (bands !== undefined) this.bands = bands;
    if (strength !== undefined) this.quantizeStrength = strength;
    if (this.passes.toonQuantize) {
      this.passes.toonQuantize.uniforms.uBands.value = this.bands;
      this.passes.toonQuantize.uniforms.uStrength.value = this.quantizeStrength;
    }
    return this;
  }

  setTimeOfDay(cycle) {
    const elevation = Math.sin(cycle * Math.PI * 2 - Math.PI * 0.5);
    const dayFactor = smoothstep(-0.2, 0.3, elevation);

    this.temperature = mix(-0.08, 0.10, dayFactor);
    this.saturation = mix(0.85, 1.15, dayFactor);
    this.contrast = mix(1.02, 1.08, dayFactor);
    this.bloomThreshold = mix(0.55, 0.78, dayFactor);
    this.bloomStrength = mix(0.35, 0.55, dayFactor);

    this.setColorGrade(this.saturation, this.contrast, this.brightness, this.temperature, this.tint);
    this.setBloomStrength(this.bloomStrength);
    this.setBloomIntensity(this.bloomIntensity, this.bloomThreshold, this.bloomSoftKnee);

    return this;
  }

  // -------------------------------------------------------------------------
  // ENABLE / DISABLE
  // -------------------------------------------------------------------------
  setEnabled(enabled) {
    this.enabled = enabled;
    return this;
  }

  setPassEnabled(name, enabled) {
    if (name === 'edge') this.enableEdge = enabled;
    else if (name === 'bloom') this.enableBloom = enabled;
    else if (name === 'colorGrade') this.enableColorGrade = enabled;
    else if (name === 'vignette') this.enableVignette = enabled;
    else if (name === 'chromatic') this.enableChromatic = enabled;
    else if (name === 'grain') this.enableGrain = enabled;
    else if (name === 'halftone') this.enableHalftone = enabled;
    else if (name === 'scanline') this.enableScanline = enabled;
    else if (name === 'quantize') this.enableQuantize = enabled;
    return this;
  }

  // -------------------------------------------------------------------------
  // ACTIVE PASS LIST
  // -------------------------------------------------------------------------
  getActivePasses() {
    const list = [];
    if (this.enableEdge && this.passes.edge) list.push(this.passes.edge);
    if (this.enableBloom && this.passes.bloomExtract && this.passes.bloomComposite) {
      list.push(this.passes.bloomExtract);
      list.push(this.passes.bloomBlurH);
      list.push(this.passes.bloomBlurV);
      list.push(this.passes.bloomComposite);
    }
    if (this.enableColorGrade && this.passes.colorGrade) list.push(this.passes.colorGrade);
    if (this.enableQuantize && this.passes.toonQuantize) list.push(this.passes.toonQuantize);
    if (this.enableChromatic && this.passes.chromatic) list.push(this.passes.chromatic);
    if (this.enableHalftone && this.passes.halftone) list.push(this.passes.halftone);
    if (this.enableScanline && this.passes.scanline) list.push(this.passes.scanline);
    if (this.enableVignette && this.passes.vignette) list.push(this.passes.vignette);
    if (this.enableGrain && this.passes.grain) list.push(this.passes.grain);
    return list;
  }

  // -------------------------------------------------------------------------
  // PRESETS
  // -------------------------------------------------------------------------
  applyPreset(preset) {
    switch (preset) {
      case 'cinematic':
        this.setEdgeStrength(0.55, 0.24);
        this.setBloomStrength(0.45, new THREE.Color(1.0, 0.95, 0.88));
        this.setColorGrade(1.10, 1.08, -0.01, 0.06, 0.0);
        this.setVignette(0.45, 0.55, new THREE.Color(0.0, 0.02, 0.06));
        this.setChromaticAberration(0.0025);
        this.setQuantize(8.0, 0.10);
        break;
      case 'anime':
        this.setEdgeStrength(0.75, 0.20);
        this.setBloomStrength(0.65, new THREE.Color(1.0, 0.98, 0.95));
        this.setColorGrade(1.22, 1.06, 0.01, 0.04, 0.0);
        this.setVignette(0.30, 0.60, new THREE.Color(0.0, 0.02, 0.06));
        this.setChromaticAberration(0.0020);
        this.setQuantize(6.0, 0.22);
        break;
      case 'soft':
        this.setEdgeStrength(0.40, 0.30);
        this.setBloomStrength(0.35, new THREE.Color(1.0, 0.96, 0.94));
        this.setColorGrade(1.08, 1.02, 0.02, 0.02, 0.0);
        this.setVignette(0.22, 0.70, new THREE.Color(0.0, 0.03, 0.08));
        this.setChromaticAberration(0.0012);
        this.setQuantize(10.0, 0.08);
        break;
      case 'graphic':
        this.setEdgeStrength(0.90, 0.16);
        this.setBloomStrength(0.30, new THREE.Color(1.0, 1.0, 1.0));
        this.setColorGrade(1.30, 1.15, 0.0, 0.03, 0.0);
        this.setVignette(0.38, 0.50, new THREE.Color(0.0, 0.01, 0.04));
        this.setChromaticAberration(0.0008);
        this.setQuantize(5.0, 0.30);
        break;
      case 'retro':
        this.setEdgeStrength(0.60, 0.22);
        this.setBloomStrength(0.40, new THREE.Color(1.0, 0.92, 0.78));
        this.setColorGrade(1.14, 1.04, -0.02, 0.10, 0.05);
        this.setVignette(0.55, 0.45, new THREE.Color(0.05, 0.02, 0.0));
        this.setChromaticAberration(0.0035);
        this.setGrain(0.06, 512.0);
        this.setScanline(0.08, this.height);
        this.enableGrain = true;
        this.enableScanline = true;
        break;
    }
    return this;
  }

  setScanline(intensity, count) {
    if (intensity !== undefined) this.scanlineIntensity = intensity;
    if (count !== undefined) this.scanlineCount = count;
    if (this.passes.scanline) {
      this.passes.scanline.uniforms.uScanlineIntensity.value = this.scanlineIntensity;
      this.passes.scanline.uniforms.uScanlineCount.value = this.scanlineCount;
    }
    return this;
  }

  // -------------------------------------------------------------------------
  // DISPOSE
  // -------------------------------------------------------------------------
  dispose() {
    for (const key in this.passes) {
      const pass = this.passes[key];
      if (pass && pass.dispose) pass.dispose();
    }
    this.passes = {
      edge: null,
      bloomExtract: null,
      bloomBlurH: null,
      bloomBlurV: null,
      bloomComposite: null,
      colorGrade: null,
      vignette: null,
      chromatic: null,
      grain: null,
      halftone: null,
      scanline: null,
      toonQuantize: null
    };
  }
}

// ---------------------------------------------------------------------------
// FACTORY
// ---------------------------------------------------------------------------
export function createPostProcessingAnimeShader(options = {}) {
  const shader = new PostProcessingAnimeShader(options);
  shader.build();
  return shader;
}

export default {
  PostProcessingAnimeShader,
  createPostProcessingAnimeShader,
  FULLSCREEN_VERTEX_SHADER,
  SOBEL_FRAGMENT_SHADER,
  BLOOM_EXTRACT_FRAGMENT_SHADER,
  BLOOM_BLUR_FRAGMENT_SHADER,
  BLOOM_COMPOSITE_FRAGMENT_SHADER,
  COLOR_GRADE_FRAGMENT_SHADER,
  VIGNETTE_FRAGMENT_SHADER,
  CHROMATIC_ABERRATION_FRAGMENT_SHADER,
  FILM_GRAIN_FRAGMENT_SHADER,
  HALFTONE_FRAGMENT_SHADER,
  SCANLINE_FRAGMENT_SHADER,
  TOON_QUANTIZE_FRAGMENT_SHADER
};