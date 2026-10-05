// File : 034
// name : src/core/034_rnd_GradientGenerators.js
// description : Procedural gradient generator library — the continuous-blend
//               counterpart to 033_rnd_ProceduralColorComposer.js. Where the
//               composer maps GRAYSCALE → PALETTE in discrete or banded
//               steps, THIS module produces the SMOOTH, continuous, multi-
//               stop gradients that underlie the anime look: sky zenith-to-
//               horizon ramps, water depth gradients, sunset banding, haze
//               layers, aurora sweeps, atmospheric scattering ramps, and
//               magic glow falloffs. Every gradient is defined purely by
//               numbers — no image textures, no LUTs unless procedural.
//
//               What it provides:
//                 1. GRADIENT KINDS
//                    • Linear            — 1D along an axis
//                    • Radial            — 1D from a center
//                    • Conic             — angular around a center
//                    • Diagonal          — 1D along a rotated axis
//                    • DomainWarped      — linear/radial warped by noise
//                    • MultiStop         — 2..8 stops with easing between
//                    • Bilinear          — 2D blend of two ramps
//                    • Spherical         — sky / atmosphere ramp by elevation
//                    • DepthFade         — exponential depth→fog curve
//
//                 2. EASING FUNCTIONS
//                    • linear / smoothstep / smootherstep / smootherstep5
//                    • quadIn / quadOut / quadInOut
//                    • cubicIn / cubicOut / cubicInOut
//                    • expIn / expOut / expInOut
//                    • anime-specific: softBand / hardBand / posterizeBand
//                    • physically-inspired: rayleigh / mie / fresnel
//
//                 3. ANIME PRESETS (matching the reference image set)
//                    • SKY_DAY_BLUE       (image 1 top-left + image 9)
//                    • SKY_SUNSET_PURPLE  (image 6)
//                    • SKY_SPACE_NAVY     (image 4)
//                    • WATER_DEPTH_TURQ   (image 1 + image 2)
//                    • SNOW_WHITE_BLUE    (image 3)
//                    • DESERT_SAND_WARM   (image 5)
//                    • CANYON_ROCK_WARM   (image 1)
//                    • FOLIAGE_GREEN_MIX  (image 2 + image 9)
//                    • MAGIC_GOLD_GLOW    (image 8)
//                    • PASTEL_PINK_LAV    (image 7)
//                    • ATMOSPHERIC_HAZE   (all images)
//
//                 4. OUTPUT FORM
//                    • Each preset is a GradientDescriptor: array of stops
//                      (each { t, color:[r,g,b], ease }), a kind, and meta.
//                    • JS samplers: `sampleGradient1D`, `sampleGradient2D`.
//                    • GLSL chunks: linear, radial, conic, sky, water,
//                      depth-fade, bilinear, domain-warp, multi-stop.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external gradient libs; every preset frozen at
//               module load; every sampler allocation-free.
// best for : Giving the whole lighting stack a shared vocabulary of smooth
//            color transitions. Sky dome, water surface, ground haze,
//            atmospheric fog, aurora, magic glow, and character rim all
//            reference gradients from this module so their color progression
//            matches the reference anime look without image textures.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  markProcedural,
} from './032_rnd_NoImageTexturePolicy.js';

import {
  STYLE_PALETTES,
  STYLE_ID,
} from './033_rnd_ProceduralColorComposer.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_STOPS        = 8;
export const MAX_GRADIENTS    = 32;
export const MAX_PRESETS      = 64;

export const GRADIENT_KIND = Object.freeze({
  LINEAR:         0,
  RADIAL:         1,
  CONIC:          2,
  DIAGONAL:       3,
  DOMAIN_WARPED:  4,
  MULTI_STOP:     5,
  BILINEAR:       6,
  SPHERICAL:      7,
  DEPTH_FADE:     8,
  COUNT:          9,
});

export const GRADIENT_KIND_NAME = Object.freeze([
  'linear',
  'radial',
  'conic',
  'diagonal',
  'domain_warped',
  'multi_stop',
  'bilinear',
  'spherical',
  'depth_fade',
]);

export const EASE_KIND = Object.freeze({
  LINEAR:        0,
  SMOOTHSTEP:    1,
  SMOOTHERSTEP:  2,
  SMOOTHERSTEP5: 3,
  QUAD_IN:       4,
  QUAD_OUT:      5,
  QUAD_IN_OUT:   6,
  CUBIC_IN:      7,
  CUBIC_OUT:     8,
  CUBIC_IN_OUT:  9,
  EXP_IN:        10,
  EXP_OUT:       11,
  EXP_IN_OUT:    12,
  SOFT_BAND:     13,
  HARD_BAND:     14,
  POSTERIZE_BAND:15,
  RAYLEIGH:      16,
  MIE:           17,
  FRESNEL:       18,
  COUNT:         19,
});

export const EASE_KIND_NAME = Object.freeze([
  'linear',
  'smoothstep',
  'smootherstep',
  'smootherstep5',
  'quad_in',
  'quad_out',
  'quad_in_out',
  'cubic_in',
  'cubic_out',
  'cubic_in_out',
  'exp_in',
  'exp_out',
  'exp_in_out',
  'soft_band',
  'hard_band',
  'posterize_band',
  'rayleigh',
  'mie',
  'fresnel',
]);

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _lin(hex) {
  const r = ((hex >> 16) & 255) / 255;
  const g = ((hex >>  8) & 255) / 255;
  const b = ( hex        & 255) / 255;
  return [
    r <= 0.04045 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4),
    g <= 0.04045 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4),
    b <= 0.04045 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4),
  ];
}

function _clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function _mix3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/* ------------------------------------------------------------------ */
/* 2. EASING FUNCTIONS (JS)                                           */
/* ------------------------------------------------------------------ */

export const EASE_JS = Object.freeze({
  [EASE_KIND.LINEAR]:        (t) => _clamp01(t),
  [EASE_KIND.SMOOTHSTEP]:    (t) => { const x = _clamp01(t); return x * x * (3 - 2 * x); },
  [EASE_KIND.SMOOTHERSTEP]:  (t) => { const x = _clamp01(t); return x * x * x * (x * (x * 6 - 15) + 10); },
  [EASE_KIND.SMOOTHERSTEP5]: (t) => {
    const x = _clamp01(t);
    return x * x * x * x * x * (x * (x * 6 - 15) + 10);
  },
  [EASE_KIND.QUAD_IN]:       (t) => { const x = _clamp01(t); return x * x; },
  [EASE_KIND.QUAD_OUT]:      (t) => { const x = _clamp01(t); return 1 - (1 - x) * (1 - x); },
  [EASE_KIND.QUAD_IN_OUT]:   (t) => {
    const x = _clamp01(t);
    return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
  },
  [EASE_KIND.CUBIC_IN]:      (t) => { const x = _clamp01(t); return x * x * x; },
  [EASE_KIND.CUBIC_OUT]:     (t) => { const x = _clamp01(t); return 1 - Math.pow(1 - x, 3); },
  [EASE_KIND.CUBIC_IN_OUT]:  (t) => {
    const x = _clamp01(t);
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  },
  [EASE_KIND.EXP_IN]:        (t) => { const x = _clamp01(t); return x === 0 ? 0 : Math.pow(2, 10 * x - 10); },
  [EASE_KIND.EXP_OUT]:       (t) => { const x = _clamp01(t); return x === 1 ? 1 : 1 - Math.pow(2, -10 * x); },
  [EASE_KIND.EXP_IN_OUT]:    (t) => {
    const x = _clamp01(t);
    if (x === 0) return 0;
    if (x === 1) return 1;
    return x < 0.5
      ? Math.pow(2, 20 * x - 10) / 2
      : (2 - Math.pow(2, -20 * x + 10)) / 2;
  },
  [EASE_KIND.SOFT_BAND]:     (t) => {
    const x = _clamp01(t);
    // Smooth band that eases at both ends
    return x * x * (3 - 2 * x) * 0.5 + x * 0.5;
  },
  [EASE_KIND.HARD_BAND]:     (t) => (t < 0.5 ? 0 : 1),
  [EASE_KIND.POSTERIZE_BAND]:(t) => {
    const x = _clamp01(t);
    return Math.floor(x * 6) / 6;
  },
  [EASE_KIND.RAYLEIGH]:      (t) => {
    // Rayleigh-ish forward-scatter ramp: low at horizon, high at zenith
    const x = _clamp01(t);
    return Math.pow(x, 0.55);
  },
  [EASE_KIND.MIE]:           (t) => {
    // Mie forward scatter: sharp peak near sun direction
    const x = _clamp01(t);
    return Math.pow(x, 8);
  },
  [EASE_KIND.FRESNEL]:       (t) => {
    // Fresnel-ish: high at edges, low at center
    const x = _clamp01(t);
    return Math.pow(1 - x, 5);
  },
});

/* ------------------------------------------------------------------ */
/* 3. EASING FUNCTIONS (GLSL)                                         */
/* ------------------------------------------------------------------ */

export const GLSL_EASING = /* glsl */`
#ifndef GLSL_EASING_INCLUDED
#define GLSL_EASING_INCLUDED

float easeLinear(float t)       { return clamp(t, 0.0, 1.0); }
float easeSmoothstep(float t)   { float x = clamp(t, 0.0, 1.0); return x * x * (3.0 - 2.0 * x); }
float easeSmootherstep(float t) { float x = clamp(t, 0.0, 1.0); return x * x * x * (x * (x * 6.0 - 15.0) + 10.0); }
float easeSmootherstep5(float t){
  float x = clamp(t, 0.0, 1.0);
  return x * x * x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}
float easeQuadIn(float t)       { float x = clamp(t, 0.0, 1.0); return x * x; }
float easeQuadOut(float t)      { float x = clamp(t, 0.0, 1.0); return 1.0 - (1.0 - x) * (1.0 - x); }
float easeQuadInOut(float t) {
  float x = clamp(t, 0.0, 1.0);
  return x < 0.5 ? 2.0 * x * x : 1.0 - pow(-2.0 * x + 2.0, 2.0) * 0.5;
}
float easeCubicIn(float t)      { float x = clamp(t, 0.0, 1.0); return x * x * x; }
float easeCubicOut(float t)     { float x = clamp(t, 0.0, 1.0); return 1.0 - pow(1.0 - x, 3.0); }
float easeCubicInOut(float t) {
  float x = clamp(t, 0.0, 1.0);
  return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) * 0.5;
}
float easeExpIn(float t)        { float x = clamp(t, 0.0, 1.0); return x <= 0.0 ? 0.0 : pow(2.0, 10.0 * x - 10.0); }
float easeExpOut(float t)       { float x = clamp(t, 0.0, 1.0); return x >= 1.0 ? 1.0 : 1.0 - pow(2.0, -10.0 * x); }
float easeExpInOut(float t) {
  float x = clamp(t, 0.0, 1.0);
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return x < 0.5 ? pow(2.0, 20.0 * x - 10.0) * 0.5 : (2.0 - pow(2.0, -20.0 * x + 10.0)) * 0.5;
}
float easeSoftBand(float t) {
  float x = clamp(t, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x) * 0.5 + x * 0.5;
}
float easeHardBand(float t)     { return t < 0.5 ? 0.0 : 1.0; }
float easePosterizeBand(float t){ return floor(clamp(t, 0.0, 1.0) * 6.0) / 6.0; }
float easeRayleigh(float t)     { return pow(clamp(t, 0.0, 1.0), 0.55); }
float easeMie(float t)          { return pow(clamp(t, 0.0, 1.0), 8.0); }
float easeFresnel(float t)      { return pow(1.0 - clamp(t, 0.0, 1.0), 5.0); }

float applyEase(float t, float mode) {
  if (mode < 0.5)  return easeLinear(t);
  if (mode < 1.5)  return easeSmoothstep(t);
  if (mode < 2.5)  return easeSmootherstep(t);
  if (mode < 3.5)  return easeSmootherstep5(t);
  if (mode < 4.5)  return easeQuadIn(t);
  if (mode < 5.5)  return easeQuadOut(t);
  if (mode < 6.5)  return easeQuadInOut(t);
  if (mode < 7.5)  return easeCubicIn(t);
  if (mode < 8.5)  return easeCubicOut(t);
  if (mode < 9.5)  return easeCubicInOut(t);
  if (mode < 10.5) return easeExpIn(t);
  if (mode < 11.5) return easeExpOut(t);
  if (mode < 12.5) return easeExpInOut(t);
  if (mode < 13.5) return easeSoftBand(t);
  if (mode < 14.5) return easeHardBand(t);
  if (mode < 15.5) return easePosterizeBand(t);
  if (mode < 16.5) return easeRayleigh(t);
  if (mode < 17.5) return easeMie(t);
  return easeFresnel(t);
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 4. GRADIENT DESCRIPTOR                                             */
/* ------------------------------------------------------------------ */

/**
 * A gradient is a compact description:
 *   kind      — GRADIENT_KIND
 *   stops     — Array<{ t:[0..1], color:[r,g,b], ease:EASE_KIND }>
 *               (2..MAX_STOPS, sorted by t ascending)
 *   origin    — optional [x,y] in normalized [0,1] space (for radial/conic)
 *   angle     — optional radians (for linear/diagonal/conic)
 *   warpStrength — optional (for DOMAIN_WARPED)
 *   warpScale    — optional (for DOMAIN_WARPED)
 *   name, id   — metadata
 */
export class GradientDescriptor {
  constructor(spec) {
    this.id     = spec.id || 'gradient';
    this.name   = spec.name || this.id;
    this.kind   = spec.kind !== undefined ? spec.kind : GRADIENT_KIND.LINEAR;
    this.stops  = Object.freeze((spec.stops || []).map(_freezeStop));
    this.origin = spec.origin ? Object.freeze(spec.origin.slice()) : null;
    this.angle  = spec.angle || 0;
    this.warpStrength = spec.warpStrength || 0;
    this.warpScale    = spec.warpScale || 1.0;
    this.stopCount    = this.stops.length;
  }
}

function _freezeStop(s) {
  return Object.freeze({
    t:     _clamp01(s.t),
    color: Object.freeze([s.color[0], s.color[1], s.color[2]]),
    ease:  s.ease !== undefined ? s.ease : EASE_KIND.LINEAR,
  });
}

/* ------------------------------------------------------------------ */
/* 5. JS GRADIENT SAMPLING                                            */
/* ------------------------------------------------------------------ */

/**
 * Sample a gradient descriptor at scalar t ∈ [0,1]. Returns a fresh
 * [r,g,b] array — use `sampleGradient1DInto(out, g, t)` on hot paths.
 */
export function sampleGradient1D(gradient, t) {
  const out = [0, 0, 0];
  sampleGradient1DInto(out, gradient, t);
  return out;
}

/**
 * Zero-alloc version: writes into `out` (length ≥ 3).
 */
export function sampleGradient1DInto(out, gradient, t) {
  const stops = gradient.stops;
  const n = stops.length;
  if (n === 0) { out[0] = out[1] = out[2] = 0; return out; }
  if (n === 1) { out[0] = stops[0].color[0]; out[1] = stops[0].color[1]; out[2] = stops[0].color[2]; return out; }

  const x = _clamp01(t);

  // Below first stop.
  if (x <= stops[0].t) {
    out[0] = stops[0].color[0]; out[1] = stops[0].color[1]; out[2] = stops[0].color[2];
    return out;
  }
  // Above last stop.
  if (x >= stops[n - 1].t) {
    const c = stops[n - 1].color;
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
    return out;
  }

  // Find segment.
  for (let i = 0; i < n - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (x >= a.t && x <= b.t) {
      const span = b.t - a.t;
      let local = span > 1e-6 ? (x - a.t) / span : 0;
      const easeFn = EASE_JS[b.ease] || EASE_JS[EASE_KIND.LINEAR];
      local = easeFn(local);
      out[0] = a.color[0] + (b.color[0] - a.color[0]) * local;
      out[1] = a.color[1] + (b.color[1] - a.color[1]) * local;
      out[2] = a.color[2] + (b.color[2] - a.color[2]) * local;
      return out;
    }
  }

  // Fallback — should not reach.
  const c = stops[n - 1].color;
  out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
  return out;
}

/**
 * Sample a 2D gradient (bilinear or warped) at (u,v) ∈ [0,1]². Writes
 * into `out` (length ≥ 3).
 */
export function sampleGradient2DInto(out, gradient, u, v) {
  switch (gradient.kind) {
    case GRADIENT_KIND.LINEAR: {
      // Linear along angle.
      const c = Math.cos(gradient.angle);
      const s = Math.sin(gradient.angle);
      const t = (u * c + v * s);
      return sampleGradient1DInto(out, gradient, t);
    }
    case GRADIENT_KIND.DIAGONAL: {
      return sampleGradient1DInto(out, gradient, (u + v) * 0.5);
    }
    case GRADIENT_KIND.RADIAL: {
      const ox = gradient.origin ? gradient.origin[0] : 0.5;
      const oy = gradient.origin ? gradient.origin[1] : 0.5;
      const dx = u - ox;
      const dy = v - oy;
      const d = Math.sqrt(dx * dx + dy * dy);
      return sampleGradient1DInto(out, gradient, d);
    }
    case GRADIENT_KIND.CONIC: {
      const ox = gradient.origin ? gradient.origin[0] : 0.5;
      const oy = gradient.origin ? gradient.origin[1] : 0.5;
      const dx = u - ox;
      const dy = v - oy;
      let a = Math.atan2(dy, dx) / (Math.PI * 2);
      if (a < 0) a += 1;
      return sampleGradient1DInto(out, gradient, a);
    }
    case GRADIENT_KIND.DOMAIN_WARPED: {
      const w = gradient.warpStrength;
      const s = gradient.warpScale;
      const nx = Math.sin(u * 12.9 * s + v * 78.2 * s) * 0.5 + 0.5;
      const ny = Math.sin(u * 45.7 * s + v * 12.3 * s) * 0.5 + 0.5;
      const warp = (nx * 0.5 + ny * 0.5 - 0.5) * w;
      return sampleGradient1DInto(out, gradient, _clamp01(u + warp));
    }
    case GRADIENT_KIND.BILINEAR: {
      // Treat u as blend of two 1D ramps along v.
      return sampleGradient1DInto(out, gradient, v);
    }
    case GRADIENT_KIND.SPHERICAL: {
      // Spherical: v is elevation from 0 (down) to 1 (up).
      return sampleGradient1DInto(out, gradient, v);
    }
    case GRADIENT_KIND.DEPTH_FADE: {
      // Depth fade: exponential falloff.
      const d = Math.exp(-v * 4.0);
      return sampleGradient1DInto(out, gradient, 1 - d);
    }
    default:
      return sampleGradient1DInto(out, gradient, (u + v) * 0.5);
  }
}

/* ------------------------------------------------------------------ */
/* 6. GLSL GRADIENT CHUNKS                                            */
/* ------------------------------------------------------------------ */

/**
 * Uniform array of gradient stops (position + color). Up to MAX_STOPS
 * stops. This is the low-cost form that avoids per-pixel JS bridging.
 */
export const GLSL_GRADIENT_UNIFORMS = /* glsl */`
#define MAX_GRADIENT_STOPS 8

uniform int   uGradientStopCount;
uniform float uGradientStopT[MAX_GRADIENT_STOPS];
uniform vec3  uGradientStopColor[MAX_GRADIENT_STOPS];
uniform int   uGradientStopEase[MAX_GRADIENT_STOPS];

uniform int   uGradientKind;
uniform vec2  uGradientOrigin;
uniform float uGradientAngle;
uniform float uGradientWarpStrength;
uniform float uGradientWarpScale;
`;

/**
 * Full GLSL gradient sampler. Requires GLSL_EASING to be included first.
 *
 *   vec3 sampleGradient1D(float t)
 *   vec3 sampleGradient2D(vec2 uv)
 *
 * Zero-texture, allocation-free, branches are unrolled by the compiler
 * because uGradientStopCount is a uniform int (≤ 8).
 */
export const GLSL_GRADIENT_SAMPLE = /* glsl */`
vec3 sampleGradient1D(float t) {
  float x = clamp(t, 0.0, 1.0);
  if (uGradientStopCount <= 0) return vec3(0.0);
  if (uGradientStopCount == 1) return uGradientStopColor[0];

  if (x <= uGradientStopT[0]) return uGradientStopColor[0];

  for (int i = 0; i < MAX_GRADIENT_STOPS - 1; i++) {
    if (i + 1 >= uGradientStopCount) break;
    float tA = uGradientStopT[i];
    float tB = uGradientStopT[i + 1];
    if (x >= tA && x <= tB) {
      float span = max(tB - tA, 1e-6);
      float local = (x - tA) / span;
      local = applyEase(local, float(uGradientStopEase[i + 1]));
      vec3 ca = uGradientStopColor[i];
      vec3 cb = uGradientStopColor[i + 1];
      return mix(ca, cb, local);
    }
  }
  return uGradientStopColor[uGradientStopCount - 1];
}

vec3 sampleGradient2D(vec2 uv) {
  // Kind 0: linear along uGradientAngle
  if (uGradientKind == 0) {
    vec2 dir = vec2(cos(uGradientAngle), sin(uGradientAngle));
    float t = dot(uv - 0.5, dir) + 0.5;
    return sampleGradient1D(t);
  }

  // Kind 1: radial from uGradientOrigin
  if (uGradientKind == 1) {
    float d = length(uv - uGradientOrigin);
    return sampleGradient1D(d);
  }

  // Kind 2: conic around uGradientOrigin
  if (uGradientKind == 2) {
    vec2 d = uv - uGradientOrigin;
    float a = atan(d.y, d.x) / 6.2831853 + 0.5;
    return sampleGradient1D(fract(a));
  }

  // Kind 3: diagonal
  if (uGradientKind == 3) {
    return sampleGradient1D((uv.x + uv.y) * 0.5);
  }

  // Kind 4: domain-warped
  if (uGradientKind == 4) {
    float nx = sin(uv.x * 12.9 * uGradientWarpScale + uv.y * 78.2 * uGradientWarpScale) * 0.5 + 0.5;
    float ny = sin(uv.x * 45.7 * uGradientWarpScale + uv.y * 12.3 * uGradientWarpScale) * 0.5 + 0.5;
    float warp = (nx * 0.5 + ny * 0.5 - 0.5) * uGradientWarpStrength;
    return sampleGradient1D(clamp(uv.x + warp, 0.0, 1.0));
  }

  // Kind 5..8: default to linear vertical / spherical / depth-fade
  // Kind 5 multi-stop uses 1D
  if (uGradientKind == 5) return sampleGradient1D(uv.x);
  if (uGradientKind == 6) return sampleGradient1D(uv.y);
  if (uGradientKind == 7) return sampleGradient1D(uv.y);
  if (uGradientKind == 8) {
    float d = exp(-uv.y * 4.0);
    return sampleGradient1D(1.0 - d);
  }

  return sampleGradient1D((uv.x + uv.y) * 0.5);
}
`;

/* ------------------------------------------------------------------ */
/* 7. ANIME GRADIENT PRESETS (matching reference images)              */
/* ------------------------------------------------------------------ */

function _preset(id, name, kind, stops, opts) {
  return Object.freeze(new GradientDescriptor(Object.assign({
    id, name, kind, stops,
  }, opts || {})));
}

/* ---- SKY DAY BLUE (image 1 top, image 9) ------------------------ */
export const GRADIENT_SKY_DAY_BLUE = _preset(
  'sky_day_blue',
  'Sky Day Blue',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0xd8ecf8), ease: EASE_KIND.SMOOTHSTEP },  // horizon warm pale
    { t: 0.25, color: _lin(0xa0c8e8), ease: EASE_KIND.RAYLEIGH },    // mid horizon
    { t: 0.55, color: _lin(0x4a90d8), ease: EASE_KIND.RAYLEIGH },    // upper sky
    { t: 0.85, color: _lin(0x1e5fa8), ease: EASE_KIND.RAYLEIGH },    // zenith blue
    { t: 1.00, color: _lin(0x0e3a70), ease: EASE_KIND.LINEAR },      // deep zenith
  ],
  { angle: 0 }
);

/* ---- SKY SUNSET PURPLE (image 6) -------------------------------- */
export const GRADIENT_SKY_SUNSET_PURPLE = _preset(
  'sky_sunset_purple',
  'Sky Sunset Purple',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0xffb878), ease: EASE_KIND.SMOOTHSTEP },  // sun-warm horizon
    { t: 0.18, color: _lin(0xe89888), ease: EASE_KIND.SMOOTHSTEP },  // peach
    { t: 0.40, color: _lin(0xc088c0), ease: EASE_KIND.SMOOTHSTEP },  // violet
    { t: 0.65, color: _lin(0x8878c8), ease: EASE_KIND.SMOOTHSTEP },  // deep purple
    { t: 0.88, color: _lin(0x5a5898), ease: EASE_KIND.SMOOTHSTEP },  // dusk violet
    { t: 1.00, color: _lin(0x3a3868), ease: EASE_KIND.LINEAR },      // zenith
  ],
  { angle: 0 }
);

/* ---- SKY SPACE NAVY (image 4) ----------------------------------- */
export const GRADIENT_SKY_SPACE_NAVY = _preset(
  'sky_space_navy',
  'Sky Space Navy',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0x1a3060), ease: EASE_KIND.RAYLEIGH },    // horizon glow
    { t: 0.20, color: _lin(0x0a1a3a), ease: EASE_KIND.SMOOTHSTEP },  // atmosphere band
    { t: 0.55, color: _lin(0x050a18), ease: EASE_KIND.LINEAR },      // deep navy
    { t: 1.00, color: _lin(0x020308), ease: EASE_KIND.LINEAR },      // space black
  ],
  { angle: 0 }
);

/* ---- WATER DEPTH TURQUOISE (image 1 + 2) ------------------------ */
export const GRADIENT_WATER_DEPTH_TURQ = _preset(
  'water_depth_turq',
  'Water Depth Turquoise',
  GRADIENT_KIND.DEPTH_FADE,
  [
    { t: 0.00, color: _lin(0xf0fbff), ease: EASE_KIND.SMOOTHSTEP },  // foam highlight
    { t: 0.15, color: _lin(0x8adce0), ease: EASE_KIND.SMOOTHSTEP },  // shallow cyan
    { t: 0.40, color: _lin(0x3ab8c8), ease: EASE_KIND.SMOOTHSTEP },  // mid turquoise
    { t: 0.70, color: _lin(0x1a6a7a), ease: EASE_KIND.SMOOTHSTEP },  // deep turquoise
    { t: 1.00, color: _lin(0x0a3a4a), ease: EASE_KIND.LINEAR },      // abyss
  ],
  { angle: 0 }
);

/* ---- SNOW WHITE BLUE (image 3) ---------------------------------- */
export const GRADIENT_SNOW_WHITE_BLUE = _preset(
  'snow_white_blue',
  'Snow White Blue',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0x3a6a9a), ease: EASE_KIND.SMOOTHSTEP },  // shadow blue
    { t: 0.25, color: _lin(0x7aa8d8), ease: EASE_KIND.SMOOTHSTEP },  // mid blue
    { t: 0.55, color: _lin(0xb8d8f0), ease: EASE_KIND.SMOOTHSTEP },  // light blue
    { t: 0.85, color: _lin(0xe8f4ff), ease: EASE_KIND.SMOOTHSTEP },  // pale snow
    { t: 1.00, color: _lin(0xffffff), ease: EASE_KIND.LINEAR },      // bright snow
  ],
  { angle: 0 }
);

/* ---- DESERT SAND WARM (image 5) --------------------------------- */
export const GRADIENT_DESERT_SAND_WARM = _preset(
  'desert_sand_warm',
  'Desert Sand Warm',
  GRADIENT_KIND.LINEAR,
  [
    { t: 0.00, color: _lin(0x3a2818), ease: EASE_KIND.SMOOTHSTEP },  // shadow
    { t: 0.30, color: _lin(0x8a5838), ease: EASE_KIND.SMOOTHSTEP },  // mid brown
    { t: 0.65, color: _lin(0xc89868), ease: EASE_KIND.SMOOTHSTEP },  // lit sand
    { t: 0.90, color: _lin(0xe8c898), ease: EASE_KIND.SMOOTHSTEP },  // bright sand
    { t: 1.00, color: _lin(0xfff0d0), ease: EASE_KIND.LINEAR },      // hot highlight
  ],
  { angle: 0 }
);

/* ---- CANYON ROCK WARM (image 1) --------------------------------- */
export const GRADIENT_CANYON_ROCK_WARM = _preset(
  'canyon_rock_warm',
  'Canyon Rock Warm',
  GRADIENT_KIND.LINEAR,
  [
    { t: 0.00, color: _lin(0x2a1a10), ease: EASE_KIND.SMOOTHSTEP },  // dark crevice
    { t: 0.25, color: _lin(0x6a4028), ease: EASE_KIND.SMOOTHSTEP },  // shadow side
    { t: 0.55, color: _lin(0xb08050), ease: EASE_KIND.SMOOTHSTEP },  // lit side
    { t: 0.82, color: _lin(0xdeb885), ease: EASE_KIND.SMOOTHSTEP },  // sunlit
    { t: 1.00, color: _lin(0xf5e0b8), ease: EASE_KIND.LINEAR },      // top highlight
  ],
  { angle: 0 }
);

/* ---- FOLIAGE GREEN MIX (image 2 + 9) ---------------------------- */
export const GRADIENT_FOLIAGE_GREEN_MIX = _preset(
  'foliage_green_mix',
  'Foliage Green Mix',
  GRADIENT_KIND.MULTI_STOP,
  [
    { t: 0.00, color: _lin(0x0a2010), ease: EASE_KIND.SMOOTHSTEP },  // deep shadow green
    { t: 0.20, color: _lin(0x1a4a20), ease: EASE_KIND.SMOOTHSTEP },  // dark green
    { t: 0.45, color: _lin(0x3a8a30), ease: EASE_KIND.SMOOTHSTEP },  // mid green
    { t: 0.70, color: _lin(0x6ac050), ease: EASE_KIND.SMOOTHSTEP },  // bright green
    { t: 0.90, color: _lin(0xa8e870), ease: EASE_KIND.SMOOTHSTEP },  // yellow-green tip
    { t: 1.00, color: _lin(0xd8f8a0), ease: EASE_KIND.LINEAR },      // sunlit tip
  ],
  { angle: 0 }
);

/* ---- MAGIC GOLD GLOW (image 8) ---------------------------------- */
export const GRADIENT_MAGIC_GOLD_GLOW = _preset(
  'magic_gold_glow',
  'Magic Gold Glow',
  GRADIENT_KIND.RADIAL,
  [
    { t: 0.00, color: _lin(0xfff8c8), ease: EASE_KIND.SMOOTHSTEP },  // core white-gold
    { t: 0.25, color: _lin(0xffd868), ease: EASE_KIND.SMOOTHSTEP },  // bright gold
    { t: 0.55, color: _lin(0xffa030), ease: EASE_KIND.SMOOTHSTEP },  // orange
    { t: 0.85, color: _lin(0x6a3010), ease: EASE_KIND.SMOOTHSTEP },  // dark warm
    { t: 1.00, color: _lin(0x0a0808), ease: EASE_KIND.LINEAR },      // black
  ],
  { origin: [0.5, 0.5] }
);

/* ---- PASTEL PINK LAVENDER (image 7) ----------------------------- */
export const GRADIENT_PASTEL_PINK_LAV = _preset(
  'pastel_pink_lav',
  'Pastel Pink Lavender',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0x8a80c0), ease: EASE_KIND.SMOOTHSTEP },  // lavender shadow
    { t: 0.25, color: _lin(0xb8a8d8), ease: EASE_KIND.SMOOTHSTEP },  // mid lavender
    { t: 0.50, color: _lin(0xe8c0e8), ease: EASE_KIND.SMOOTHSTEP },  // pink-lavender
    { t: 0.75, color: _lin(0xf8d8e8), ease: EASE_KIND.SMOOTHSTEP },  // light pink
    { t: 1.00, color: _lin(0xfff0f8), ease: EASE_KIND.LINEAR },      // pale highlight
  ],
  { angle: 0 }
);

/* ---- ATMOSPHERIC HAZE (all images) ------------------------------ */
export const GRADIENT_ATMOSPHERIC_HAZE = _preset(
  'atmospheric_haze',
  'Atmospheric Haze',
  GRADIENT_KIND.DEPTH_FADE,
  [
    { t: 0.00, color: _lin(0x0a0a10), ease: EASE_KIND.SMOOTHSTEP },  // near black
    { t: 0.40, color: _lin(0x506a80), ease: EASE_KIND.SMOOTHSTEP },  // mid haze
    { t: 0.75, color: _lin(0xb0c0d0), ease: EASE_KIND.SMOOTHSTEP },  // light haze
    { t: 1.00, color: _lin(0xe8eef4), ease: EASE_KIND.LINEAR },      // far haze
  ],
  { angle: 0 }
);

/* ---- AURORA GREEN CYAN (bonus for snow biome) ------------------- */
export const GRADIENT_AURORA_GREEN_CYAN = _preset(
  'aurora_green_cyan',
  'Aurora Green Cyan',
  GRADIENT_KIND.SPHERICAL,
  [
    { t: 0.00, color: _lin(0x0a1a2a), ease: EASE_KIND.SMOOTHSTEP },  // night sky
    { t: 0.30, color: _lin(0x2a8a8a), ease: EASE_KIND.SMOOTHSTEP },  // teal band
    { t: 0.60, color: _lin(0x4ae8c0), ease: EASE_KIND.SMOOTHSTEP },  // bright cyan
    { t: 0.85, color: _lin(0x8affd8), ease: EASE_KIND.SMOOTHSTEP },  // pale aurora
    { t: 1.00, color: _lin(0xd8ffe8), ease: EASE_KIND.LINEAR },      // white crown
  ],
  { angle: 0 }
);

/* ---- FIRE EMBER (bonus for volcanic) ---------------------------- */
export const GRADIENT_FIRE_EMBER = _preset(
  'fire_ember',
  'Fire Ember',
  GRADIENT_KIND.RADIAL,
  [
    { t: 0.00, color: _lin(0xfff8e0), ease: EASE_KIND.SMOOTHSTEP },  // white core
    { t: 0.20, color: _lin(0xffd050), ease: EASE_KIND.SMOOTHSTEP },  // yellow
    { t: 0.45, color: _lin(0xff7020), ease: EASE_KIND.SMOOTHSTEP },  // orange
    { t: 0.75, color: _lin(0xa02810), ease: EASE_KIND.SMOOTHSTEP },  // deep red
    { t: 1.00, color: _lin(0x180808), ease: EASE_KIND.LINEAR },      // black
  ],
  { origin: [0.5, 0.5] }
);

/* ------------------------------------------------------------------ */
/* 8. PRESET REGISTRY                                                 */
/* ------------------------------------------------------------------ */

export const GRADIENT_PRESETS = Object.freeze({
  sky_day_blue:           GRADIENT_SKY_DAY_BLUE,
  sky_sunset_purple:      GRADIENT_SKY_SUNSET_PURPLE,
  sky_space_navy:         GRADIENT_SKY_SPACE_NAVY,
  water_depth_turq:       GRADIENT_WATER_DEPTH_TURQ,
  snow_white_blue:        GRADIENT_SNOW_WHITE_BLUE,
  desert_sand_warm:       GRADIENT_DESERT_SAND_WARM,
  canyon_rock_warm:       GRADIENT_CANYON_ROCK_WARM,
  foliage_green_mix:      GRADIENT_FOLIAGE_GREEN_MIX,
  magic_gold_glow:        GRADIENT_MAGIC_GOLD_GLOW,
  pastel_pink_lav:        GRADIENT_PASTEL_PINK_LAV,
  atmospheric_haze:       GRADIENT_ATMOSPHERIC_HAZE,
  aurora_green_cyan:      GRADIENT_AURORA_GREEN_CYAN,
  fire_ember:             GRADIENT_FIRE_EMBER,
});

export const GRADIENT_PRESET_ID = Object.freeze({
  SKY_DAY_BLUE:          'sky_day_blue',
  SKY_SUNSET_PURPLE:     'sky_sunset_purple',
  SKY_SPACE_NAVY:        'sky_space_navy',
  WATER_DEPTH_TURQ:      'water_depth_turq',
  SNOW_WHITE_BLUE:       'snow_white_blue',
  DESERT_SAND_WARM:      'desert_sand_warm',
  CANYON_ROCK_WARM:      'canyon_rock_warm',
  FOLIAGE_GREEN_MIX:     'foliage_green_mix',
  MAGIC_GOLD_GLOW:       'magic_gold_glow',
  PASTEL_PINK_LAV:       'pastel_pink_lav',
  ATMOSPHERIC_HAZE:      'atmospheric_haze',
  AURORA_GREEN_CYAN:     'aurora_green_cyan',
  FIRE_EMBER:            'fire_ember',
});

/* ------------------------------------------------------------------ */
/* 9. PRESET → UNIFORM BLOCK                                          */
/* ------------------------------------------------------------------ */

/**
 * Converts a GradientDescriptor into a set of uniforms consumable by
 * GLSL_GRADIENT_SAMPLE:
 *
 *   uGradientStopCount   int
 *   uGradientStopT[]     float[]
 *   uGradientStopColor[] vec3[]
 *   uGradientStopEase[]  int[]
 *   uGradientKind        int
 *   uGradientOrigin      vec2
 *   uGradientAngle       float
 *   uGradientWarpStrength float
 *   uGradientWarpScale   float
 *
 * Every value is a plain number, typed array element, or Vector — no
 * texture, no bitmap, no image.
 */
export function createGradientUniforms(gradient) {
  if (!gradient) return null;

  const stopCount = Math.min(gradient.stopCount, MAX_STOPS);
  const tArr     = new Float32Array(MAX_STOPS);
  const colorArr = [];
  for (let i = 0; i < MAX_STOPS; i++) {
    if (i < stopCount) {
      tArr[i] = gradient.stops[i].t;
      const c = gradient.stops[i].color;
      colorArr.push(new THREE.Vector3(c[0], c[1], c[2]));
    } else {
      tArr[i] = i < 1 ? 1 : 1;
      colorArr.push(new THREE.Vector3(0, 0, 0));
    }
  }
  const easeArr = new Int32Array(MAX_STOPS);
  for (let i = 0; i < stopCount; i++) easeArr[i] = gradient.stops[i].ease;

  return {
    uGradientStopCount:    { value: stopCount | 0 },
    uGradientStopT:        { value: tArr },
    uGradientStopColor:    { value: colorArr },
    uGradientStopEase:     { value: easeArr },
    uGradientKind:         { value: gradient.kind | 0 },
    uGradientOrigin:       { value: new THREE.Vector2(
      gradient.origin ? gradient.origin[0] : 0.5,
      gradient.origin ? gradient.origin[1] : 0.5) },
    uGradientAngle:        { value: gradient.angle || 0 },
    uGradientWarpStrength: { value: gradient.warpStrength || 0 },
    uGradientWarpScale:    { value: gradient.warpScale || 1.0 },
  };
}

/**
 * Apply a gradient to an existing uniform set in place. Useful for
 * hot-swapping gradients without reallocating Vector3 arrays.
 */
export function applyGradientToUniforms(uniforms, gradient) {
  if (!uniforms || !gradient) return false;
  const stopCount = Math.min(gradient.stopCount, MAX_STOPS);
  const tArr = uniforms.uGradientStopT.value;
  const cArr = uniforms.uGradientStopColor.value;
  const eArr = uniforms.uGradientStopEase.value;

  for (let i = 0; i < MAX_STOPS; i++) {
    if (i < stopCount) {
      const s = gradient.stops[i];
      tArr[i] = s.t;
      cArr[i].set(s.color[0], s.color[1], s.color[2]);
      eArr[i] = s.ease;
    } else {
      tArr[i] = 1;
      cArr[i].set(0, 0, 0);
      eArr[i] = 0;
    }
  }

  uniforms.uGradientStopCount.value = stopCount | 0;
  uniforms.uGradientKind.value = gradient.kind | 0;
  uniforms.uGradientOrigin.value.set(
    gradient.origin ? gradient.origin[0] : 0.5,
    gradient.origin ? gradient.origin[1] : 0.5
  );
  uniforms.uGradientAngle.value = gradient.angle || 0;
  uniforms.uGradientWarpStrength.value = gradient.warpStrength || 0;
  uniforms.uGradientWarpScale.value = gradient.warpScale || 1.0;
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. GRADIENT BAKER (JS → DataTexture, procedural)                  */
/* ------------------------------------------------------------------ */

/**
 * Bakes a 1D gradient into a DataTexture LUT of `size` samples. This is
 * the fast-path alternative to the per-pixel multi-stop evaluation: a
 * single `texture2D(uGradientLUT, vec2(t, 0.5))` call replaces the
 * whole loop.
 *
 * The LUT is procedurally filled (satisfies 032 no-image-texture policy).
 */
export function bakeGradientLUT(gradient, size = 256) {
  if (!gradient) return null;
  const s = Math.max(4, size | 0);
  const data = new Uint8Array(s * 4);
  const rgb = [0, 0, 0];

  for (let i = 0; i < s; i++) {
    const t = i / (s - 1);
    sampleGradient1DInto(rgb, gradient, t);
    const i4 = i * 4;
    data[i4]     = _linearToSRGBByte(rgb[0]);
    data[i4 + 1] = _linearToSRGBByte(rgb[1]);
    data[i4 + 2] = _linearToSRGBByte(rgb[2]);
    data[i4 + 3] = 255;
  }

  const tex = new THREE.DataTexture(data, s, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  markProcedural(tex, 'bakeGradientLUT:' + gradient.id);
  return tex;
}

/**
 * Bakes a 2D gradient (bilinear kind or domain-warped) into a DataTexture
 * of size×size.
 */
export function bakeGradientLUT2D(gradient, size = 64) {
  if (!gradient) return null;
  const s = Math.max(4, size | 0);
  const data = new Uint8Array(s * s * 4);
  const rgb = [0, 0, 0];

  for (let y = 0; y < s; y++) {
    const v = y / (s - 1);
    for (let x = 0; x < s; x++) {
      const u = x / (s - 1);
      sampleGradient2DInto(rgb, gradient, u, v);
      const i4 = (y * s + x) * 4;
      data[i4]     = _linearToSRGBByte(rgb[0]);
      data[i4 + 1] = _linearToSRGBByte(rgb[1]);
      data[i4 + 2] = _linearToSRGBByte(rgb[2]);
      data[i4 + 3] = 255;
    }
  }

  const tex = new THREE.DataTexture(data, s, s, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  markProcedural(tex, 'bakeGradientLUT2D:' + gradient.id);
  return tex;
}

function _linearToSRGBByte(c) {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/* ------------------------------------------------------------------ */
/* 11. GRADIENT BUILDER (runtime custom gradients)                    */
/* ------------------------------------------------------------------ */

/**
 * Builds a custom GradientDescriptor from a compact spec. Use this when
 * a subsystem needs a gradient that isn't in the preset registry.
 *
 *   buildGradient({
 *     id:    'my_custom',
 *     name:  'My Custom',
 *     kind:  GRADIENT_KIND.LINEAR,
 *     stops: [
 *       { t: 0, color: [0,0,0], ease: EASE_KIND.LINEAR },
 *       { t: 1, color: [1,1,1], ease: EASE_KIND.SMOOTHSTEP },
 *     ],
 *   })
 */
export function buildGradient(spec) {
  if (!spec || !Array.isArray(spec.stops) || spec.stops.length < 2) return null;
  if (spec.stops.length > MAX_STOPS) return null;

  // Verify ascending t.
  for (let i = 1; i < spec.stops.length; i++) {
    if (spec.stops[i].t < spec.stops[i - 1].t) return null;
  }
  return new GradientDescriptor(spec);
}

/* ------------------------------------------------------------------ */
/* 12. MODULE-LEVEL COMPOSER REGISTRY                                 */
/* ------------------------------------------------------------------ */

const _customRegistry = new Map();

export function registerGradient(gradient) {
  if (!gradient || !gradient.id) return false;
  _customRegistry.set(gradient.id, gradient);
  return true;
}

export function unregisterGradient(id) {
  return _customRegistry.delete(id);
}

export function getGradient(id) {
  return _customRegistry.get(id) || GRADIENT_PRESETS[id] || null;
}

export function listGradients() {
  const out = [];
  for (const k in GRADIENT_PRESETS) out.push(k);
  for (const k of _customRegistry.keys()) out.push(k);
  return out;
}

/* ------------------------------------------------------------------ */
/* 13. STYLE → GRADIENT MAPPING (convenience for the composer)        */
/* ------------------------------------------------------------------ */

/**
 * Maps a style palette id from 033_rnd_ProceduralColorComposer.js to a
 * companion gradient. This is what material factories use to pair a
 * palette with its default smooth ramp.
 */
export const STYLE_TO_GRADIENT = Object.freeze({
  [STYLE_ID.CANYON_TURQUOISE]:   GRADIENT_PRESET_ID.CANYON_ROCK_WARM,
  [STYLE_ID.RIVER_TOP_DOWN]:     GRADIENT_PRESET_ID.WATER_DEPTH_TURQ,
  [STYLE_ID.SNOW_BLUE_TOP_DOWN]: GRADIENT_PRESET_ID.SNOW_WHITE_BLUE,
  [STYLE_ID.ORBITAL_SPACE]:      GRADIENT_PRESET_ID.SKY_SPACE_NAVY,
  [STYLE_ID.DESERT_RUINS]:       GRADIENT_PRESET_ID.DESERT_SAND_WARM,
  [STYLE_ID.SUNSET_TOMBSTONE]:   GRADIENT_PRESET_ID.SKY_SUNSET_PURPLE,
  [STYLE_ID.PASTEL_PORTRAIT]:    GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  [STYLE_ID.MAGIC_CASTER]:       GRADIENT_PRESET_ID.MAGIC_GOLD_GLOW,
  [STYLE_ID.FLOWER_FIELD]:       GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
});

export function getCompanionGradientForStyle(styleId) {
  const gid = STYLE_TO_GRADIENT[styleId];
  if (!gid) return GRADIENT_ATMOSPHERIC_HAZE;
  return getGradient(gid);
}

/* ------------------------------------------------------------------ */
/* 14. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createGradient(spec) {
  return buildGradient(spec);
}

/* ------------------------------------------------------------------ */
/* 15. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  GradientDescriptor,

  // Presets
  GRADIENT_PRESETS,
  GRADIENT_PRESET_ID,
  GRADIENT_SKY_DAY_BLUE,
  GRADIENT_SKY_SUNSET_PURPLE,
  GRADIENT_SKY_SPACE_NAVY,
  GRADIENT_WATER_DEPTH_TURQ,
  GRADIENT_SNOW_WHITE_BLUE,
  GRADIENT_DESERT_SAND_WARM,
  GRADIENT_CANYON_ROCK_WARM,
  GRADIENT_FOLIAGE_GREEN_MIX,
  GRADIENT_MAGIC_GOLD_GLOW,
  GRADIENT_PASTEL_PINK_LAV,
  GRADIENT_ATMOSPHERIC_HAZE,
  GRADIENT_AURORA_GREEN_CYAN,
  GRADIENT_FIRE_EMBER,

  // JS sampling
  sampleGradient1D,
  sampleGradient1DInto,
  sampleGradient2DInto,

  // GLSL
  GLSL_EASING,
  GLSL_GRADIENT_UNIFORMS,
  GLSL_GRADIENT_SAMPLE,

  // Uniforms
  createGradientUniforms,
  applyGradientToUniforms,

  // Baking
  bakeGradientLUT,
  bakeGradientLUT2D,

  // Building & registry
  buildGradient,
  createGradient,
  registerGradient,
  unregisterGradient,
  getGradient,
  listGradients,

  // Style pairing
  STYLE_TO_GRADIENT,
  getCompanionGradientForStyle,

  // Enums
  GRADIENT_KIND,
  GRADIENT_KIND_NAME,
  EASE_KIND,
  EASE_KIND_NAME,
  EASE_JS,

  MAX_STOPS,
  MAX_GRADIENTS,
  MAX_PRESETS,
};

export default _defaultExport;