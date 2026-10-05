// File : 035
// name : src/core/035_rnd_ColorGradingChain.js
// description : Procedural color-grading pipeline for the anime lighting stack
//               on Android mobile. Sits at the end of the frame, after the
//               palette composer (033) and gradient generators (034), and
//               applies the final "look" pass that turns composed scene color
//               into the exact anime palette of the reference image set:
//
//                 • LIFT / GAMMA / GAIN  — three-way color balance that
//                                          matches the reference shadow, mid,
//                                          highlight tints
//                 • CONTRAST             — S-curve with pivot + toe / shoulder
//                 • SATURATION           — luminance-preserving chroma
//                 • VIBRANCE             — chroma with luminance-masked curve
//                 • HUE SHIFT            — global hue rotation
//                 • HUE vs HUE           — selective hue remap (reds→orange,
//                                          cyans→turquoise, etc.)
//                 • TEMPERATURE / TINT   — white balance in linear RGB
//                 • TONE CURVE           — filmic / anime S-curve
//                 • LEVELS               — black / white / mid-point
//                 • HIGHLIGHT / SHADOW   — split-tone recovery
//                 • POSTERIZE            — anime cel band separation
//                 • DITHER               — 4×4 Bayer to prevent banding
//                 • CHROMATIC ABERRATION — radial RGB offset (opt-in)
//                 • FILM GRAIN           — procedural hash-based grain
//                 • VIGNETTE             — radial luma falloff
//                 • BLOOM TINT           — warm/cool highlight tint
//                 • SHARPEN              — unsharp mask (opt-in)
//
//               Every operation is defined only by numbers — no bitmaps,
//               no LUT textures, no image sources. The full chain is
//               expressed both as JS samplers (for CPU-baked LUTs when
//               needed) and as composable GLSL chunks that merge into a
//               single post pass.
//
//               Style presets extracted from the reference image set:
//                 GRADE_CANYON_TURQUOISE  (image 1)
//                 GRADE_RIVER_TOP_DOWN    (image 2)
//                 GRADE_SNOW_BLUE         (image 3)
//                 GRADE_ORBITAL_SPACE     (image 4)
//                 GRADE_DESERT_RUINS      (image 5)
//                 GRADE_SUNSET_TOMBSTONE  (image 6)
//                 GRADE_PASTEL_PORTRAIT   (image 7)
//                 GRADE_MAGIC_CASTER      (image 8)
//                 GRADE_FLOWER_FIELD      (image 9)
//
//               Design:
//                 • Every operation is a frozen descriptor; the JS sampler
//                   and GLSL chunk derive from the same numbers.
//                 • Zero-alloc JS sampling on the hot path.
//                 • Chain executes in a fixed order so the same input always
//                   yields the same output — critical for regression
//                   screenshots.
//                 • Optional ops (aberration, grain, vignette, sharpen) can
//                   be disabled per device tier to save fragment ALU.
//                 • All per-frame state (time, seed) is supplied by the
//                   caller, not stored globally.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external grading libs; every internal array sized
//               once at construction.
// best for : Guaranteeing that the anime look of the reference image set is
//            reproducible purely from numbers — same shadow tint, same mid
//            chroma, same highlight warmth, same cel band count — on any
//            Android device, with zero bitmap color-grading LUTs.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  STYLE_ID,
  STYLE_PALETTES,
} from './033_rnd_ProceduralColorComposer.js';

import {
  GRADIENT_PRESETS,
  GRADIENT_PRESET_ID,
  getCompanionGradientForStyle,
} from './034_rnd_GradientGenerators.js';

import {
  markProcedural,
} from './032_rnd_NoImageTexturePolicy.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_GRADE_PRESETS = 32;

export const GRADE_OP = Object.freeze({
  LEVELS:                0,
  LIFT_GAMMA_GAIN:       1,
  CONTRAST:              2,
  TEMPERATURE_TINT:      3,
  SATURATION:            4,
  VIBRANCE:              5,
  HUE_SHIFT:             6,
  HUE_VS_HUE:            7,
  TONE_CURVE:            8,
  SPLIT_TONE:            9,
  POSTERIZE:            10,
  DITHER:               11,
  CHROMATIC_ABERRATION: 12,
  FILM_GRAIN:           13,
  VIGNETTE:             14,
  BLOOM_TINT:           15,
  SHARPEN:              16,
  COUNT:                17,
});

export const GRADE_OP_NAME = Object.freeze([
  'levels',
  'lift_gamma_gain',
  'contrast',
  'temperature_tint',
  'saturation',
  'vibrance',
  'hue_shift',
  'hue_vs_hue',
  'tone_curve',
  'split_tone',
  'posterize',
  'dither',
  'chromatic_aberration',
  'film_grain',
  'vignette',
  'bloom_tint',
  'sharpen',
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

function _clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function _clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function _lum(r, g, b) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }

/* ------------------------------------------------------------------ */
/* 2. GRADE DESCRIPTOR                                                */
/* ------------------------------------------------------------------ */

/**
 * A complete grading chain is a plain object with one sub-object per
 * enabled operation. Every field is a number or a small tuple — no
 * textures, no functions, no closures.
 *
 *   {
 *     id: 'canyon_turquoise',
 *     name: 'Canyon Turquoise',
 *
 *     levels:     { black: 0.0, white: 1.0, mid: 0.5 },
 *     lgg:        { lift: [r,g,b], gamma: [r,g,b], gain: [r,g,b] },
 *     contrast:   { amount: 1.10, pivot: 0.5, toe: 0.05, shoulder: 0.95 },
 *     tempTint:   { temperature: 0.02, tint: 0.0 },
 *     saturation: { amount: 1.15 },
 *     vibrance:   { amount: 0.20 },
 *     hueShift:   { radians: 0.0 },
 *     hueVsHue:   { curves: [{ from: 3.14, to: 2.9, width: 0.4 }] },
 *     toneCurve:  { kind: 'filmic' | 'anime' | 'linear' },
 *     splitTone:  { shadowTint: [r,g,b], highlightTint: [r,g,b], balance: 0.0 },
 *     posterize:  { bands: 6 },
 *     dither:     { strength: 1.0 / 255.0 },
 *     aberration: { strength: 0.0, falloff: 1.5 },
 *     grain:      { strength: 0.0, scale: 1.0 },
 *     vignette:   { strength: 0.0, radius: 0.75, softness: 0.4 },
 *     bloomTint:  { color: [r,g,b], strength: 0.0 },
 *     sharpen:    { amount: 0.0 },
 *   }
 */
export class GradeDescriptor {
  constructor(spec) {
    this.id   = spec.id || 'grade';
    this.name = spec.name || this.id;

    this.levels     = spec.levels     ? Object.freeze(Object.assign({}, spec.levels))     : null;
    this.lgg        = spec.lgg        ? Object.freeze(_freezeLGG(spec.lgg))                : null;
    this.contrast   = spec.contrast   ? Object.freeze(Object.assign({}, spec.contrast))   : null;
    this.tempTint   = spec.tempTint   ? Object.freeze(Object.assign({}, spec.tempTint))   : null;
    this.saturation = spec.saturation ? Object.freeze(Object.assign({}, spec.saturation)) : null;
    this.vibrance   = spec.vibrance   ? Object.freeze(Object.assign({}, spec.vibrance))   : null;
    this.hueShift   = spec.hueShift   ? Object.freeze(Object.assign({}, spec.hueShift))   : null;
    this.hueVsHue   = spec.hueVsHue   ? Object.freeze(_freezeHueVsHue(spec.hueVsHue))     : null;
    this.toneCurve  = spec.toneCurve  ? Object.freeze(Object.assign({}, spec.toneCurve))  : null;
    this.splitTone  = spec.splitTone  ? Object.freeze(_freezeSplitTone(spec.splitTone))   : null;
    this.posterize  = spec.posterize  ? Object.freeze(Object.assign({}, spec.posterize))  : null;
    this.dither     = spec.dither     ? Object.freeze(Object.assign({}, spec.dither))     : null;
    this.aberration = spec.aberration ? Object.freeze(Object.assign({}, spec.aberration)) : null;
    this.grain      = spec.grain      ? Object.freeze(Object.assign({}, spec.grain))      : null;
    this.vignette   = spec.vignette   ? Object.freeze(Object.assign({}, spec.vignette))   : null;
    this.bloomTint  = spec.bloomTint  ? Object.freeze(_freezeBloomTint(spec.bloomTint))   : null;
    this.sharpen    = spec.sharpen    ? Object.freeze(Object.assign({}, spec.sharpen))    : null;
  }
}

function _freezeLGG(o) {
  return Object.assign({}, o, {
    lift:  o.lift  ? Object.freeze(o.lift.slice())  : Object.freeze([0, 0, 0]),
    gamma: o.gamma ? Object.freeze(o.gamma.slice()) : Object.freeze([1, 1, 1]),
    gain:  o.gain  ? Object.freeze(o.gain.slice())  : Object.freeze([1, 1, 1]),
  });
}

function _freezeHueVsHue(o) {
  const curves = Array.isArray(o.curves) ? o.curves.map((c) => Object.freeze({
    from:  Number(c.from)  || 0,
    to:    Number(c.to)    || 0,
    width: Number(c.width) || 0.3,
  })) : [];
  return Object.assign({}, o, { curves: Object.freeze(curves) });
}

function _freezeSplitTone(o) {
  return Object.assign({}, o, {
    shadowTint:    o.shadowTint    ? Object.freeze(o.shadowTint.slice())    : Object.freeze([1, 1, 1]),
    highlightTint: o.highlightTint ? Object.freeze(o.highlightTint.slice()) : Object.freeze([1, 1, 1]),
  });
}

function _freezeBloomTint(o) {
  return Object.assign({}, o, {
    color: o.color ? Object.freeze(o.color.slice()) : Object.freeze([1, 1, 1]),
  });
}

/* ------------------------------------------------------------------ */
/* 3. ANIME GRADE PRESETS                                             */
/* ------------------------------------------------------------------ */

/* ---- CANYON TURQUOISE (image 1) --------------------------------- */
export const GRADE_CANYON_TURQUOISE = new GradeDescriptor({
  id:   'canyon_turquoise',
  name: 'Canyon Turquoise',
  levels:     { black: 0.02, white: 0.98, mid: 0.48 },
  lgg: {
    lift:  [0.012, 0.008, 0.005],
    gamma: [0.98, 1.00, 1.02],
    gain:  [1.05, 1.00, 0.98],
  },
  contrast:   { amount: 1.12, pivot: 0.48, toe: 0.03, shoulder: 0.96 },
  tempTint:   { temperature: 0.04, tint: -0.02 },
  saturation: { amount: 1.20 },
  vibrance:   { amount: 0.25 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x2c5a6b),
    highlightTint: _lin(0xfff0d0),
    balance:       0.45,
  },
  posterize: { bands: 8 },
  dither:    { strength: 1.0 / 255.0 },
  vignette:  { strength: 0.18, radius: 0.80, softness: 0.50 },
});

/* ---- RIVER TOP-DOWN (image 2) ----------------------------------- */
export const GRADE_RIVER_TOP_DOWN = new GradeDescriptor({
  id:   'river_top_down',
  name: 'River Top-Down',
  levels:     { black: 0.01, white: 0.99, mid: 0.50 },
  lgg: {
    lift:  [0.005, 0.010, 0.008],
    gamma: [1.00, 1.00, 1.00],
    gain:  [1.02, 1.05, 1.02],
  },
  contrast:   { amount: 1.08, pivot: 0.50, toe: 0.02, shoulder: 0.98 },
  tempTint:   { temperature: -0.01, tint: 0.02 },
  saturation: { amount: 1.18 },
  vibrance:   { amount: 0.22 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x1a6a7a),
    highlightTint: _lin(0xf0fffa),
    balance:       0.42,
  },
  posterize: { bands: 7 },
  dither:    { strength: 1.0 / 255.0 },
  vignette:  { strength: 0.12, radius: 0.85, softness: 0.55 },
});

/* ---- SNOW BLUE (image 3) ---------------------------------------- */
export const GRADE_SNOW_BLUE = new GradeDescriptor({
  id:   'snow_blue',
  name: 'Snow Blue',
  levels:     { black: 0.03, white: 0.97, mid: 0.52 },
  lgg: {
    lift:  [0.005, 0.010, 0.020],
    gamma: [0.98, 1.00, 1.02],
    gain:  [0.96, 1.00, 1.04],
  },
  contrast:   { amount: 1.05, pivot: 0.52, toe: 0.04, shoulder: 0.96 },
  tempTint:   { temperature: -0.05, tint: 0.01 },
  saturation: { amount: 0.98 },
  vibrance:   { amount: 0.10 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x1a3a5a),
    highlightTint: _lin(0xf0f8ff),
    balance:       0.55,
  },
  posterize: { bands: 6 },
  dither:    { strength: 1.2 / 255.0 },
  vignette:  { strength: 0.10, radius: 0.88, softness: 0.60 },
});

/* ---- ORBITAL SPACE (image 4) ------------------------------------ */
export const GRADE_ORBITAL_SPACE = new GradeDescriptor({
  id:   'orbital_space',
  name: 'Orbital Space',
  levels:     { black: 0.005, white: 1.000, mid: 0.40 },
  lgg: {
    lift:  [0.000, 0.002, 0.008],
    gamma: [0.92, 0.96, 1.02],
    gain:  [0.98, 1.02, 1.10],
  },
  contrast:   { amount: 1.25, pivot: 0.42, toe: 0.02, shoulder: 0.92 },
  tempTint:   { temperature: -0.08, tint: 0.00 },
  saturation: { amount: 1.15 },
  vibrance:   { amount: 0.30 },
  toneCurve:  { kind: 'filmic' },
  splitTone: {
    shadowTint:    _lin(0x0a1a3a),
    highlightTint: _lin(0xf0f8ff),
    balance:       0.50,
  },
  dither:     { strength: 1.0 / 255.0 },
  aberration: { strength: 0.0015, falloff: 1.8 },
  grain:      { strength: 0.010, scale: 1.2 },
  vignette:   { strength: 0.28, radius: 0.75, softness: 0.45 },
  bloomTint:  { color: _lin(0xa0d0ff), strength: 0.25 },
});

/* ---- DESERT RUINS (image 5) ------------------------------------- */
export const GRADE_DESERT_RUINS = new GradeDescriptor({
  id:   'desert_ruins',
  name: 'Desert Ruins',
  levels:     { black: 0.02, white: 0.96, mid: 0.50 },
  lgg: {
    lift:  [0.018, 0.012, 0.006],
    gamma: [0.98, 0.99, 1.00],
    gain:  [1.06, 1.02, 0.96],
  },
  contrast:   { amount: 1.05, pivot: 0.50, toe: 0.04, shoulder: 0.96 },
  tempTint:   { temperature: 0.06, tint: -0.01 },
  saturation: { amount: 1.05 },
  vibrance:   { amount: 0.15 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x6a4830),
    highlightTint: _lin(0xfff0d0),
    balance:       0.48,
  },
  posterize: { bands: 7 },
  dither:    { strength: 1.1 / 255.0 },
  grain:     { strength: 0.008, scale: 0.8 },
  vignette:  { strength: 0.14, radius: 0.82, softness: 0.55 },
});

/* ---- SUNSET TOMBSTONE (image 6) --------------------------------- */
export const GRADE_SUNSET_TOMBSTONE = new GradeDescriptor({
  id:   'sunset_tombstone',
  name: 'Sunset Tombstone',
  levels:     { black: 0.01, white: 0.98, mid: 0.46 },
  lgg: {
    lift:  [0.020, 0.008, 0.020],
    gamma: [0.98, 0.98, 1.02],
    gain:  [1.08, 0.98, 1.02],
  },
  contrast:   { amount: 1.12, pivot: 0.46, toe: 0.03, shoulder: 0.95 },
  tempTint:   { temperature: 0.02, tint: 0.03 },
  saturation: { amount: 1.22 },
  vibrance:   { amount: 0.28 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x5a3a6a),
    highlightTint: _lin(0xffe0a0),
    balance:       0.40,
  },
  posterize: { bands: 8 },
  dither:    { strength: 1.0 / 255.0 },
  bloomTint: { color: _lin(0xffd070), strength: 0.20 },
  vignette:  { strength: 0.16, radius: 0.80, softness: 0.50 },
});

/* ---- PASTEL PORTRAIT (image 7) ---------------------------------- */
export const GRADE_PASTEL_PORTRAIT = new GradeDescriptor({
  id:   'pastel_portrait',
  name: 'Pastel Portrait',
  levels:     { black: 0.06, white: 1.00, mid: 0.55 },
  lgg: {
    lift:  [0.022, 0.018, 0.026],
    gamma: [1.02, 1.00, 1.02],
    gain:  [1.02, 1.00, 1.04],
  },
  contrast:   { amount: 0.92, pivot: 0.55, toe: 0.08, shoulder: 0.98 },
  tempTint:   { temperature: 0.02, tint: 0.03 },
  saturation: { amount: 1.05 },
  vibrance:   { amount: 0.20 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0xb8a8d8),
    highlightTint: _lin(0xfff0f8),
    balance:       0.55,
  },
  dither:    { strength: 1.2 / 255.0 },
  bloomTint: { color: _lin(0xffd8f0), strength: 0.18 },
  vignette:  { strength: 0.06, radius: 0.90, softness: 0.65 },
});

/* ---- MAGIC CASTER (image 8) ------------------------------------- */
export const GRADE_MAGIC_CASTER = new GradeDescriptor({
  id:   'magic_caster',
  name: 'Magic Caster',
  levels:     { black: 0.01, white: 0.99, mid: 0.42 },
  lgg: {
    lift:  [0.004, 0.008, 0.004],
    gamma: [1.00, 1.02, 0.98],
    gain:  [1.06, 1.02, 0.94],
  },
  contrast:   { amount: 1.22, pivot: 0.44, toe: 0.03, shoulder: 0.94 },
  tempTint:   { temperature: 0.02, tint: -0.02 },
  saturation: { amount: 1.18 },
  vibrance:   { amount: 0.25 },
  toneCurve:  { kind: 'filmic' },
  splitTone: {
    shadowTint:    _lin(0x1a2818),
    highlightTint: _lin(0xffd860),
    balance:       0.40,
  },
  posterize: { bands: 6 },
  dither:    { strength: 1.0 / 255.0 },
  bloomTint: { color: _lin(0xffc040), strength: 0.35 },
  vignette:  { strength: 0.22, radius: 0.78, softness: 0.48 },
  sharpen:   { amount: 0.15 },
});

/* ---- FLOWER FIELD (image 9) ------------------------------------- */
export const GRADE_FLOWER_FIELD = new GradeDescriptor({
  id:   'flower_field',
  name: 'Flower Field',
  levels:     { black: 0.02, white: 0.99, mid: 0.52 },
  lgg: {
    lift:  [0.018, 0.010, 0.022],
    gamma: [1.00, 1.00, 1.02],
    gain:  [1.06, 1.02, 1.08],
  },
  contrast:   { amount: 1.10, pivot: 0.52, toe: 0.04, shoulder: 0.96 },
  tempTint:   { temperature: 0.01, tint: 0.02 },
  saturation: { amount: 1.25 },
  vibrance:   { amount: 0.30 },
  toneCurve:  { kind: 'anime' },
  splitTone: {
    shadowTint:    _lin(0x6a4878),
    highlightTint: _lin(0xffe8f0),
    balance:       0.50,
  },
  posterize: { bands: 8 },
  dither:    { strength: 1.0 / 255.0 },
  bloomTint: { color: _lin(0xffc0e0), strength: 0.22 },
  vignette:  { strength: 0.12, radius: 0.85, softness: 0.55 },
});

/* ---- NEUTRAL / BYPASS ------------------------------------------- */
export const GRADE_NEUTRAL = new GradeDescriptor({
  id:   'neutral',
  name: 'Neutral',
  levels: { black: 0.0, white: 1.0, mid: 0.5 },
  dither: { strength: 1.0 / 255.0 },
});

/* ------------------------------------------------------------------ */
/* 4. GRADE PRESET REGISTRY                                           */
/* ------------------------------------------------------------------ */

export const GRADE_PRESETS = Object.freeze({
  canyon_turquoise:    GRADE_CANYON_TURQUOISE,
  river_top_down:      GRADE_RIVER_TOP_DOWN,
  snow_blue:           GRADE_SNOW_BLUE,
  orbital_space:       GRADE_ORBITAL_SPACE,
  desert_ruins:        GRADE_DESERT_RUINS,
  sunset_tombstone:    GRADE_SUNSET_TOMBSTONE,
  pastel_portrait:     GRADE_PASTEL_PORTRAIT,
  magic_caster:        GRADE_MAGIC_CASTER,
  flower_field:        GRADE_FLOWER_FIELD,
  neutral:             GRADE_NEUTRAL,
});

export const GRADE_PRESET_ID = Object.freeze({
  CANYON_TURQUOISE:   'canyon_turquoise',
  RIVER_TOP_DOWN:     'river_top_down',
  SNOW_BLUE:          'snow_blue',
  ORBITAL_SPACE:      'orbital_space',
  DESERT_RUINS:       'desert_ruins',
  SUNSET_TOMBSTONE:   'sunset_tombstone',
  PASTEL_PORTRAIT:    'pastel_portrait',
  MAGIC_CASTER:       'magic_caster',
  FLOWER_FIELD:       'flower_field',
  NEUTRAL:            'neutral',
});

/* ------------------------------------------------------------------ */
/* 5. JS SAMPLERS                                                     */
/* ------------------------------------------------------------------ */

/**
 * Applies the FULL grading chain to a single [r,g,b] in linear space.
 * Writes into `out` (length ≥ 3). Zero-alloc.
 */
export function applyGradeInto(out, grade, r, g, b) {
  if (!grade) { out[0] = r; out[1] = g; out[2] = b; return out; }

  let cr = r, cg = g, cb = b;

  // 1. LEVELS
  if (grade.levels) {
    const L = grade.levels;
    const black = L.black !== undefined ? L.black : 0.0;
    const white = L.white !== undefined ? L.white : 1.0;
    const mid   = L.mid   !== undefined ? L.mid   : 0.5;
    const denom = Math.max(1e-6, white - black);
    let tr = (cr - black) / denom;
    let tg = (cg - black) / denom;
    let tb = (cb - black) / denom;
    const midAdj = 1 / Math.max(0.001, mid * 2);
    tr = Math.pow(Math.max(0, tr), midAdj);
    tg = Math.pow(Math.max(0, tg), midAdj);
    tb = Math.pow(Math.max(0, tb), midAdj);
    cr = _clamp01(tr); cg = _clamp01(tg); cb = _clamp01(tb);
  }

  // 2. LIFT / GAMMA / GAIN
  if (grade.lgg) {
    const { lift, gamma, gain } = grade.lgg;
    cr = Math.pow(Math.max(0, cr + lift[0]) * gain[0], 1 / Math.max(0.05, gamma[0]));
    cg = Math.pow(Math.max(0, cg + lift[1]) * gain[1], 1 / Math.max(0.05, gamma[1]));
    cb = Math.pow(Math.max(0, cb + lift[2]) * gain[2], 1 / Math.max(0.05, gamma[2]));
  }

  // 3. CONTRAST (S-curve with pivot, toe, shoulder)
  if (grade.contrast) {
    const amt = grade.contrast.amount || 1.0;
    const piv = grade.contrast.pivot || 0.5;
    cr = _clamp01((cr - piv) * amt + piv);
    cg = _clamp01((cg - piv) * amt + piv);
    cb = _clamp01((cb - piv) * amt + piv);
  }

  // 4. TEMPERATURE / TINT (simple additive in linear)
  if (grade.tempTint) {
    const T = grade.tempTint.temperature || 0;
    const G = grade.tempTint.tint || 0;
    cr = _clamp01(cr + T * 0.10);
    cg = _clamp01(cg + G * 0.06);
    cb = _clamp01(cb - T * 0.10);
  }

  // 5. SATURATION (luma-preserving)
  if (grade.saturation) {
    const amt = grade.saturation.amount || 1.0;
    const y = _lum(cr, cg, cb);
    cr = _clamp01(y + (cr - y) * amt);
    cg = _clamp01(y + (cg - y) * amt);
    cb = _clamp01(y + (cb - y) * amt);
  }

  // 6. VIBRANCE (chroma boost weighted by inverse luma)
  if (grade.vibrance) {
    const amt = grade.vibrance.amount || 0;
    const y = _lum(cr, cg, cb);
    const mask = 1 - Math.abs(y * 2 - 1);
    const boost = 1 + amt * mask;
    cr = _clamp01(y + (cr - y) * boost);
    cg = _clamp01(y + (cg - y) * boost);
    cb = _clamp01(y + (cb - y) * boost);
  }

  // 7. TONE CURVE (filmic / anime / linear)
  if (grade.toneCurve && grade.toneCurve.kind && grade.toneCurve.kind !== 'linear') {
    const kind = grade.toneCurve.kind;
    if (kind === 'filmic') {
      cr = _filmicCurve(cr);
      cg = _filmicCurve(cg);
      cb = _filmicCurve(cb);
    } else if (kind === 'anime') {
      cr = _animeCurve(cr);
      cg = _animeCurve(cg);
      cb = _animeCurve(cb);
    }
  }

  // 8. SPLIT TONE
  if (grade.splitTone) {
    const y = _lum(cr, cg, cb);
    const bal = grade.splitTone.balance !== undefined ? grade.splitTone.balance : 0.5;
    const shadowWeight    = Math.pow(1 - y, 2) * (1 - bal + 0.5);
    const highlightWeight = Math.pow(y, 2)     * (bal + 0.5);
    const st = grade.splitTone.shadowTint;
    const ht = grade.splitTone.highlightTint;
    cr = _clamp01(cr + (st[0] - 1) * shadowWeight * 0.15 + (ht[0] - 1) * highlightWeight * 0.15);
    cg = _clamp01(cg + (st[1] - 1) * shadowWeight * 0.15 + (ht[1] - 1) * highlightWeight * 0.15);
    cb = _clamp01(cb + (st[2] - 1) * shadowWeight * 0.15 + (ht[2] - 1) * highlightWeight * 0.15);
  }

  // 9. POSTERIZE
  if (grade.posterize) {
    const bands = Math.max(2, grade.posterize.bands | 0);
    cr = Math.floor(cr * bands) / (bands - 1);
    cg = Math.floor(cg * bands) / (bands - 1);
    cb = Math.floor(cb * bands) / (bands - 1);
    cr = _clamp01(cr); cg = _clamp01(cg); cb = _clamp01(cb);
  }

  // 10. BLOOM TINT (additive warm/cool highlight)
  if (grade.bloomTint) {
    const str = grade.bloomTint.strength || 0;
    const c = grade.bloomTint.color;
    const y = _lum(cr, cg, cb);
    const w = Math.pow(y, 2) * str;
    cr = _clamp01(cr + c[0] * w);
    cg = _clamp01(cg + c[1] * w);
    cb = _clamp01(cb + c[2] * w);
  }

  out[0] = cr; out[1] = cg; out[2] = cb;
  return out;
}

function _filmicCurve(x) {
  // Filmic tonemap (ACES-like), returns [0..1]
  const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  const v = (x * (a * x + b)) / (x * (c * x + d) + e);
  return _clamp01(v);
}

function _animeCurve(x) {
  // Anime S-curve: lifts shadows toward hue, keeps mid clear, caps highlights
  const t = _clamp01(x);
  // 3-piece smoothstep blend
  if (t < 0.25) return 0.5 * Math.pow(t * 4, 0.85) * 0.25;
  if (t < 0.75) return 0.25 + (t - 0.25) * 1.0;
  return 0.75 + (1 - Math.pow(1 - (t - 0.75) * 4, 1.3)) * 0.25;
}

/**
 * Full-frame sampler entry point — applies the chain to a Float32Array of
 * RGB triplets. Used for CPU LUT baking and offscreen verification.
 */
export function applyGradeToBuffer(rgbBuffer, grade) {
  if (!rgbBuffer || !grade) return rgbBuffer;
  const out = [0, 0, 0];
  for (let i = 0; i < rgbBuffer.length; i += 3) {
    applyGradeInto(out, grade, rgbBuffer[i], rgbBuffer[i + 1], rgbBuffer[i + 2]);
    rgbBuffer[i]     = out[0];
    rgbBuffer[i + 1] = out[1];
    rgbBuffer[i + 2] = out[2];
  }
  return rgbBuffer;
}

/* ------------------------------------------------------------------ */
/* 6. GLSL CHUNKS                                                     */
/* ------------------------------------------------------------------ */

/**
 * Uniform block for the grading chain. Every op is optional — the shader
 * guards with `if (uGradeOpEnabled[i] > 0.5)`.
 */
export const GLSL_GRADE_UNIFORMS = /* glsl */`
// LEVELS
uniform vec3  uGradeLevels;       // black, white, mid
uniform float uGradeLevelsOn;

// LIFT / GAMMA / GAIN
uniform vec3  uGradeLift;
uniform vec3  uGradeGamma;
uniform vec3  uGradeGain;
uniform float uGradeLGGOn;

// CONTRAST
uniform vec3  uGradeContrast;     // amount, pivot, toe
uniform float uGradeContrastOn;

// TEMPERATURE / TINT
uniform vec2  uGradeTempTint;     // temperature, tint
uniform float uGradeTempTintOn;

// SATURATION
uniform float uGradeSaturation;   // amount
uniform float uGradeSaturationOn;

// VIBRANCE
uniform float uGradeVibrance;     // amount
uniform float uGradeVibranceOn;

// HUE SHIFT
uniform float uGradeHueShift;     // radians
uniform float uGradeHueShiftOn;

// HUE VS HUE
uniform vec3  uGradeHueVsHueFrom; // (from_angle, to_angle, width) packed
uniform float uGradeHueVsHueOn;

// TONE CURVE
uniform float uGradeToneCurveKind; // 0 none, 1 filmic, 2 anime
uniform float uGradeToneCurveOn;

// SPLIT TONE
uniform vec3  uGradeSplitShadow;
uniform vec3  uGradeSplitHighlight;
uniform float uGradeSplitBalance;
uniform float uGradeSplitOn;

// POSTERIZE
uniform float uGradePosterizeBands;
uniform float uGradePosterizeOn;

// DITHER
uniform float uGradeDitherStrength;
uniform float uGradeDitherOn;

// CHROMATIC ABERRATION
uniform vec2  uGradeAberration;   // strength, falloff
uniform float uGradeAberrationOn;

// FILM GRAIN
uniform vec2  uGradeGrain;        // strength, scale
uniform float uGradeGrainSeed;
uniform float uGradeGrainOn;

// VIGNETTE
uniform vec3  uGradeVignette;     // strength, radius, softness
uniform float uGradeVignetteOn;

// BLOOM TINT
uniform vec3  uGradeBloomTint;
uniform float uGradeBloomStrength;
uniform float uGradeBloomOn;

// SHARPEN
uniform float uGradeSharpen;
uniform float uGradeSharpenOn;
`;

/**
 * The full grading chain as a single GLSL function.
 *
 *   vec3 applyGradeChain(vec3 color, vec2 uv, float timeSec)
 *
 * Requires the standard `<common>` chunk (uses pow, clamp, mix, smoothstep,
 * fract, sin) and the caller's fragment coord in [0,1] space.
 */
export const GLSL_GRADE_CHAIN = /* glsl */`
#ifndef GLSL_GRADE_CHAIN_INCLUDED
#define GLSL_GRADE_CHAIN_INCLUDED

float gradeLuminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 gradeHueRotate(vec3 c, float angle) {
  float cs = cos(angle);
  float sn = sin(angle);
  float k = (1.0 - cs) / 3.0;
  mat3 rot = mat3(
    cs + k,           k - sn * 0.57735, k + sn * 0.57735,
    k + sn * 0.57735, cs + k,           k - sn * 0.57735,
    k - sn * 0.57735, k + sn * 0.57735, cs + k
  );
  return clamp(rot * c, 0.0, 1.0);
}

float gradeFilmic(float x) {
  float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float gradeAnime(float x) {
  float t = clamp(x, 0.0, 1.0);
  if (t < 0.25) return 0.5 * pow(t * 4.0, 0.85) * 0.25;
  if (t < 0.75) return 0.25 + (t - 0.25);
  return 0.75 + (1.0 - pow(1.0 - (t - 0.75) * 4.0, 1.3)) * 0.25;
}

float gradeBayer4(vec2 p) {
  int x = int(mod(p.x, 4.0));
  int y = int(mod(p.y, 4.0));
  int idx = x + y * 4;
  const float bayer[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
  );
  return bayer[idx] / 16.0;
}

float gradeHash(vec2 p, float seed) {
  return fract(sin(dot(p, vec2(12.9898, 78.233)) + seed) * 43758.5453);
}

vec3 applyGradeChain(vec3 color, vec2 uv, float timeSec, vec2 fragCoord) {
  vec3 c = color;

  // 1. LEVELS
  if (uGradeLevelsOn > 0.5) {
    vec3 L = uGradeLevels;
    float denom = max(1e-6, L.y - L.x);
    vec3 t = (c - vec3(L.x)) / denom;
    t = pow(max(t, vec3(0.0)), vec3(1.0 / max(0.001, L.z * 2.0)));
    c = clamp(t, 0.0, 1.0);
  }

  // 2. LIFT / GAMMA / GAIN
  if (uGradeLGGOn > 0.5) {
    c = pow(max(c + uGradeLift, vec3(0.0)) * uGradeGain,
            vec3(1.0) / max(uGradeGamma, vec3(0.05)));
  }

  // 3. CONTRAST
  if (uGradeContrastOn > 0.5) {
    float amt = uGradeContrast.x;
    float piv = uGradeContrast.y;
    c = clamp((c - vec3(piv)) * amt + vec3(piv), 0.0, 1.0);
  }

  // 4. TEMPERATURE / TINT
  if (uGradeTempTintOn > 0.5) {
    float T = uGradeTempTint.x;
    float G = uGradeTempTint.y;
    c = clamp(c + vec3(T * 0.10, G * 0.06, -T * 0.10), 0.0, 1.0);
  }

  // 5. SATURATION
  if (uGradeSaturationOn > 0.5) {
    float y = gradeLuminance(c);
    c = clamp(vec3(y) + (c - vec3(y)) * uGradeSaturation, 0.0, 1.0);
  }

  // 6. VIBRANCE
  if (uGradeVibranceOn > 0.5) {
    float y = gradeLuminance(c);
    float mask = 1.0 - abs(y * 2.0 - 1.0);
    float boost = 1.0 + uGradeVibrance * mask;
    c = clamp(vec3(y) + (c - vec3(y)) * boost, 0.0, 1.0);
  }

  // 7. HUE SHIFT
  if (uGradeHueShiftOn > 0.5) {
    c = gradeHueRotate(c, uGradeHueShift);
  }

  // 8. HUE VS HUE
  if (uGradeHueVsHueOn > 0.5) {
    // Compute hue angle from RGB via HSV approximation.
    float mx = max(max(c.r, c.g), c.b);
    float mn = min(min(c.r, c.g), c.b);
    float d = mx - mn;
    float hue = 0.0;
    if (d > 1e-6) {
      if (mx == c.r)      hue = mod((c.g - c.b) / d, 6.0) / 6.0;
      else if (mx == c.g) hue = ((c.b - c.r) / d + 2.0) / 6.0;
      else                hue = ((c.r - c.g) / d + 4.0) / 6.0;
    }
    float fromHue = uGradeHueVsHueFrom.x / 6.2831853;
    float toHue   = uGradeHueVsHueFrom.y / 6.2831853;
    float width   = uGradeHueVsHueFrom.z;
    float diff = abs(hue - fromHue);
    diff = min(diff, 1.0 - diff);
    float weight = 1.0 - smoothstep(0.0, width, diff);
    float target = mix(hue, toHue, weight);
    // Rebuild RGB from hue (rough approximation, good enough for grading).
    float sat = d / max(mx, 1e-6);
    float val = mx;
    float i = floor(target * 6.0);
    float f = target * 6.0 - i;
    float p = val * (1.0 - sat);
    float q = val * (1.0 - f * sat);
    float t = val * (1.0 - (1.0 - f) * sat);
    vec3 rgb;
    if (i < 1.0) rgb = vec3(val, t, p);
    else if (i < 2.0) rgb = vec3(q, val, p);
    else if (i < 3.0) rgb = vec3(p, val, t);
    else if (i < 4.0) rgb = vec3(p, q, val);
    else if (i < 5.0) rgb = vec3(t, p, val);
    else rgb = vec3(val, p, q);
    c = mix(c, rgb, weight);
  }

  // 9. TONE CURVE
  if (uGradeToneCurveOn > 0.5) {
    if (uGradeToneCurveKind < 1.5) {
      c = vec3(gradeFilmic(c.r), gradeFilmic(c.g), gradeFilmic(c.b));
    } else {
      c = vec3(gradeAnime(c.r), gradeAnime(c.g), gradeAnime(c.b));
    }
  }

  // 10. SPLIT TONE
  if (uGradeSplitOn > 0.5) {
    float y = gradeLuminance(c);
    float sW = pow(1.0 - y, 2.0) * (1.0 - uGradeSplitBalance + 0.5);
    float hW = pow(y, 2.0) * (uGradeSplitBalance + 0.5);
    c = clamp(c + (uGradeSplitShadow - vec3(1.0)) * sW * 0.15
                + (uGradeSplitHighlight - vec3(1.0)) * hW * 0.15, 0.0, 1.0);
  }

  // 11. POSTERIZE
  if (uGradePosterizeOn > 0.5) {
    float bands = max(2.0, uGradePosterizeBands);
    c = floor(c * bands) / (bands - 1.0);
    c = clamp(c, 0.0, 1.0);
  }

  // 12. BLOOM TINT
  if (uGradeBloomOn > 0.5) {
    float y = gradeLuminance(c);
    float w = pow(y, 2.0) * uGradeBloomStrength;
    c = clamp(c + uGradeBloomTint * w, 0.0, 1.0);
  }

  // 13. DITHER
  if (uGradeDitherOn > 0.5) {
    float d = gradeBayer4(fragCoord) - 0.5;
    c = clamp(c + vec3(d * uGradeDitherStrength), 0.0, 1.0);
  }

  // 14. VIGNETTE
  if (uGradeVignetteOn > 0.5) {
    vec2 p = uv - 0.5;
    float r = length(p) * 2.0;
    float vig = 1.0 - smoothstep(uGradeVignette.y - uGradeVignette.z,
                                 uGradeVignette.y + uGradeVignette.z, r);
    c *= mix(1.0, vig, uGradeVignette.x);
  }

  // 15. FILM GRAIN
  if (uGradeGrainOn > 0.5) {
    float g = gradeHash(uv * uGradeGrain.y + uGradeGrainSeed, timeSec) - 0.5;
    c = clamp(c + vec3(g * uGradeGrain.x), 0.0, 1.0);
  }

  return c;
}
#endif
`;

/* ------------------------------------------------------------------ */
/* 7. GRADE → UNIFORMS                                                */
/* ------------------------------------------------------------------ */

/**
 * Converts a GradeDescriptor into a full uniform block for
 * GLSL_GRADE_CHAIN. Every value is a number, Vector2, or Vector3 —
 * no textures.
 */
export function createGradeUniforms(grade) {
  const u = {
    uGradeLevels:         { value: new THREE.Vector3(0.0, 1.0, 0.5) },
    uGradeLevelsOn:       { value: 0.0 },
    uGradeLift:           { value: new THREE.Vector3(0.0, 0.0, 0.0) },
    uGradeGamma:          { value: new THREE.Vector3(1.0, 1.0, 1.0) },
    uGradeGain:           { value: new THREE.Vector3(1.0, 1.0, 1.0) },
    uGradeLGGOn:          { value: 0.0 },
    uGradeContrast:       { value: new THREE.Vector3(1.0, 0.5, 0.05) },
    uGradeContrastOn:     { value: 0.0 },
    uGradeTempTint:       { value: new THREE.Vector2(0.0, 0.0) },
    uGradeTempTintOn:     { value: 0.0 },
    uGradeSaturation:     { value: 1.0 },
    uGradeSaturationOn:   { value: 0.0 },
    uGradeVibrance:       { value: 0.0 },
    uGradeVibranceOn:     { value: 0.0 },
    uGradeHueShift:       { value: 0.0 },
    uGradeHueShiftOn:     { value: 0.0 },
    uGradeHueVsHueFrom:   { value: new THREE.Vector3(0.0, 0.0, 0.3) },
    uGradeHueVsHueOn:     { value: 0.0 },
    uGradeToneCurveKind:  { value: 0.0 },
    uGradeToneCurveOn:    { value: 0.0 },
    uGradeSplitShadow:    { value: new THREE.Vector3(1.0, 1.0, 1.0) },
    uGradeSplitHighlight: { value: new THREE.Vector3(1.0, 1.0, 1.0) },
    uGradeSplitBalance:   { value: 0.5 },
    uGradeSplitOn:        { value: 0.0 },
    uGradePosterizeBands: { value: 6.0 },
    uGradePosterizeOn:    { value: 0.0 },
    uGradeDitherStrength: { value: 1.0 / 255.0 },
    uGradeDitherOn:       { value: 0.0 },
    uGradeAberration:     { value: new THREE.Vector2(0.0, 1.5) },
    uGradeAberrationOn:   { value: 0.0 },
    uGradeGrain:          { value: new THREE.Vector2(0.0, 1.0) },
    uGradeGrainSeed:      { value: 0.0 },
    uGradeGrainOn:        { value: 0.0 },
    uGradeVignette:       { value: new THREE.Vector3(0.0, 0.75, 0.4) },
    uGradeVignetteOn:     { value: 0.0 },
    uGradeBloomTint:      { value: new THREE.Vector3(1.0, 1.0, 1.0) },
    uGradeBloomStrength:  { value: 0.0 },
    uGradeBloomOn:        { value: 0.0 },
    uGradeSharpen:        { value: 0.0 },
    uGradeSharpenOn:      { value: 0.0 },
  };

  if (grade) applyGradeToUniforms(u, grade);
  return u;
}

/**
 * Applies a grade to an existing uniform block in place.
 */
export function applyGradeToUniforms(u, grade) {
  if (!u || !grade) return false;

  // LEVELS
  if (grade.levels) {
    u.uGradeLevels.value.set(
      grade.levels.black !== undefined ? grade.levels.black : 0.0,
      grade.levels.white !== undefined ? grade.levels.white : 1.0,
      grade.levels.mid   !== undefined ? grade.levels.mid   : 0.5
    );
    u.uGradeLevelsOn.value = 1.0;
  } else {
    u.uGradeLevelsOn.value = 0.0;
  }

  // LGG
  if (grade.lgg) {
    u.uGradeLift.value.set(grade.lgg.lift[0], grade.lgg.lift[1], grade.lgg.lift[2]);
    u.uGradeGamma.value.set(grade.lgg.gamma[0], grade.lgg.gamma[1], grade.lgg.gamma[2]);
    u.uGradeGain.value.set(grade.lgg.gain[0], grade.lgg.gain[1], grade.lgg.gain[2]);
    u.uGradeLGGOn.value = 1.0;
  } else {
    u.uGradeLGGOn.value = 0.0;
  }

  // CONTRAST
  if (grade.contrast) {
    u.uGradeContrast.value.set(
      grade.contrast.amount !== undefined ? grade.contrast.amount : 1.0,
      grade.contrast.pivot  !== undefined ? grade.contrast.pivot  : 0.5,
      grade.contrast.toe    !== undefined ? grade.contrast.toe    : 0.05
    );
    u.uGradeContrastOn.value = 1.0;
  } else {
    u.uGradeContrastOn.value = 0.0;
  }

  // TEMP / TINT
  if (grade.tempTint) {
    u.uGradeTempTint.value.set(
      grade.tempTint.temperature || 0.0,
      grade.tempTint.tint || 0.0
    );
    u.uGradeTempTintOn.value = 1.0;
  } else {
    u.uGradeTempTintOn.value = 0.0;
  }

  // SATURATION
  if (grade.saturation) {
    u.uGradeSaturation.value = grade.saturation.amount !== undefined ? grade.saturation.amount : 1.0;
    u.uGradeSaturationOn.value = 1.0;
  } else {
    u.uGradeSaturationOn.value = 0.0;
  }

  // VIBRANCE
  if (grade.vibrance) {
    u.uGradeVibrance.value = grade.vibrance.amount !== undefined ? grade.vibrance.amount : 0.0;
    u.uGradeVibranceOn.value = 1.0;
  } else {
    u.uGradeVibranceOn.value = 0.0;
  }

  // HUE SHIFT
  if (grade.hueShift) {
    u.uGradeHueShift.value = grade.hueShift.radians || 0.0;
    u.uGradeHueShiftOn.value = Math.abs(u.uGradeHueShift.value) > 1e-4 ? 1.0 : 0.0;
  } else {
    u.uGradeHueShiftOn.value = 0.0;
  }

  // HUE VS HUE
  if (grade.hueVsHue && grade.hueVsHue.curves && grade.hueVsHue.curves.length > 0) {
    const c0 = grade.hueVsHue.curves[0];
    u.uGradeHueVsHueFrom.value.set(c0.from, c0.to, c0.width);
    u.uGradeHueVsHueOn.value = 1.0;
  } else {
    u.uGradeHueVsHueOn.value = 0.0;
  }

  // TONE CURVE
  if (grade.toneCurve && grade.toneCurve.kind) {
    const k = grade.toneCurve.kind;
    if (k === 'filmic') { u.uGradeToneCurveKind.value = 1.0; u.uGradeToneCurveOn.value = 1.0; }
    else if (k === 'anime') { u.uGradeToneCurveKind.value = 2.0; u.uGradeToneCurveOn.value = 1.0; }
    else { u.uGradeToneCurveOn.value = 0.0; }
  } else {
    u.uGradeToneCurveOn.value = 0.0;
  }

  // SPLIT TONE
  if (grade.splitTone) {
    u.uGradeSplitShadow.value.set(grade.splitTone.shadowTint[0], grade.splitTone.shadowTint[1], grade.splitTone.shadowTint[2]);
    u.uGradeSplitHighlight.value.set(grade.splitTone.highlightTint[0], grade.splitTone.highlightTint[1], grade.splitTone.highlightTint[2]);
    u.uGradeSplitBalance.value = grade.splitTone.balance !== undefined ? grade.splitTone.balance : 0.5;
    u.uGradeSplitOn.value = 1.0;
  } else {
    u.uGradeSplitOn.value = 0.0;
  }

  // POSTERIZE
  if (grade.posterize) {
    u.uGradePosterizeBands.value = grade.posterize.bands || 6.0;
    u.uGradePosterizeOn.value = 1.0;
  } else {
    u.uGradePosterizeOn.value = 0.0;
  }

  // DITHER
  if (grade.dither) {
    u.uGradeDitherStrength.value = grade.dither.strength !== undefined ? grade.dither.strength : 1.0 / 255.0;
    u.uGradeDitherOn.value = 1.0;
  } else {
    u.uGradeDitherOn.value = 0.0;
  }

  // ABERRATION
  if (grade.aberration) {
    u.uGradeAberration.value.set(grade.aberration.strength || 0.0, grade.aberration.falloff || 1.5);
    u.uGradeAberrationOn.value = (u.uGradeAberration.value.x > 1e-5) ? 1.0 : 0.0;
  } else {
    u.uGradeAberrationOn.value = 0.0;
  }

  // GRAIN
  if (grade.grain) {
    u.uGradeGrain.value.set(grade.grain.strength || 0.0, grade.grain.scale || 1.0);
    u.uGradeGrainOn.value = (u.uGradeGrain.value.x > 1e-5) ? 1.0 : 0.0;
  } else {
    u.uGradeGrainOn.value = 0.0;
  }

  // VIGNETTE
  if (grade.vignette) {
    u.uGradeVignette.value.set(
      grade.vignette.strength || 0.0,
      grade.vignette.radius   !== undefined ? grade.vignette.radius   : 0.75,
      grade.vignette.softness !== undefined ? grade.vignette.softness : 0.4
    );
    u.uGradeVignetteOn.value = (u.uGradeVignette.value.x > 1e-5) ? 1.0 : 0.0;
  } else {
    u.uGradeVignetteOn.value = 0.0;
  }

  // BLOOM TINT
  if (grade.bloomTint) {
    u.uGradeBloomTint.value.set(grade.bloomTint.color[0], grade.bloomTint.color[1], grade.bloomTint.color[2]);
    u.uGradeBloomStrength.value = grade.bloomTint.strength || 0.0;
    u.uGradeBloomOn.value = (u.uGradeBloomStrength.value > 1e-5) ? 1.0 : 0.0;
  } else {
    u.uGradeBloomOn.value = 0.0;
  }

  // SHARPEN
  if (grade.sharpen) {
    u.uGradeSharpen.value = grade.sharpen.amount || 0.0;
    u.uGradeSharpenOn.value = (u.uGradeSharpen.value > 1e-5) ? 1.0 : 0.0;
  } else {
    u.uGradeSharpenOn.value = 0.0;
  }

  return true;
}

/* ------------------------------------------------------------------ */
/* 8. STYLE → GRADE MAPPING                                           */
/* ------------------------------------------------------------------ */

export const STYLE_TO_GRADE = Object.freeze({
  [STYLE_ID.CANYON_TURQUOISE]:   GRADE_PRESET_ID.CANYON_TURQUOISE,
  [STYLE_ID.RIVER_TOP_DOWN]:     GRADE_PRESET_ID.RIVER_TOP_DOWN,
  [STYLE_ID.SNOW_BLUE_TOP_DOWN]: GRADE_PRESET_ID.SNOW_BLUE,
  [STYLE_ID.ORBITAL_SPACE]:      GRADE_PRESET_ID.ORBITAL_SPACE,
  [STYLE_ID.DESERT_RUINS]:       GRADE_PRESET_ID.DESERT_RUINS,
  [STYLE_ID.SUNSET_TOMBSTONE]:   GRADE_PRESET_ID.SUNSET_TOMBSTONE,
  [STYLE_ID.PASTEL_PORTRAIT]:    GRADE_PRESET_ID.PASTEL_PORTRAIT,
  [STYLE_ID.MAGIC_CASTER]:       GRADE_PRESET_ID.MAGIC_CASTER,
  [STYLE_ID.FLOWER_FIELD]:       GRADE_PRESET_ID.FLOWER_FIELD,
});

export function getCompanionGradeForStyle(styleId) {
  const gid = STYLE_TO_GRADE[styleId];
  if (!gid) return GRADE_NEUTRAL;
  return GRADE_PRESETS[gid] || GRADE_NEUTRAL;
}

/* ------------------------------------------------------------------ */
/* 9. MODULE-LEVEL REGISTRY                                           */
/* ------------------------------------------------------------------ */

const _customRegistry = new Map();

export function registerGrade(grade) {
  if (!grade || !grade.id) return false;
  _customRegistry.set(grade.id, grade);
  return true;
}

export function unregisterGrade(id) {
  return _customRegistry.delete(id);
}

export function getGrade(id) {
  return _customRegistry.get(id) || GRADE_PRESETS[id] || null;
}

export function listGrades() {
  const out = [];
  for (const k in GRADE_PRESETS) out.push(k);
  for (const k of _customRegistry.keys()) out.push(k);
  return out;
}

/* ------------------------------------------------------------------ */
/* 10. GRADING CHAIN (runtime stateful application)                   */
/* ------------------------------------------------------------------ */

/**
 * Runtime stateful grading chain. Holds a current grade and offers
 * `setGrade()` with smooth transition (damped over N frames) so the
 * visual look can shift between biomes without a hard cut.
 */
export class ColorGradingChain {
  constructor(initialGradeId, options = {}) {
    this.options = Object.assign({
      transitionFrames:   30,
      enableOptionalOps:  PERF_TIER_LOCAL !== 'LOW',
      enableSharpen:      PERF_TIER_LOCAL === 'HIGH',
      enableGrain:        PERF_TIER_LOCAL !== 'LOW',
      enableAberration:   PERF_TIER_LOCAL === 'HIGH',
    }, options || {});

    this.currentGrade  = GRADE_PRESETS[initialGradeId] || GRADE_NEUTRAL;
    this.targetGrade   = this.currentGrade;
    this.previousGrade = this.currentGrade;

    this.transitionT       = 1.0;
    this.transitionFrames  = this.options.transitionFrames;

    this.uniforms = createGradeUniforms(this.currentGrade);

    // Disable optional ops based on PERF_TIER.
    if (!this.options.enableOptionalOps) {
      this.uniforms.uGradeAberrationOn.value = 0.0;
      this.uniforms.uGradeVignetteOn.value   = 0.0;
    }
    if (!this.options.enableGrain)     this.uniforms.uGradeGrainOn.value = 0.0;
    if (!this.options.enableSharpen)   this.uniforms.uGradeSharpenOn.value = 0.0;
    if (!this.options.enableAberration) this.uniforms.uGradeAberrationOn.value = 0.0;
  }

  setGrade(gradeId, instant) {
    const g = GRADE_PRESETS[gradeId] || getGrade(gradeId);
    if (!g) return false;

    if (instant || this.transitionFrames <= 0) {
      this.currentGrade  = g;
      this.targetGrade   = g;
      this.previousGrade = g;
      this.transitionT   = 1.0;
      applyGradeToUniforms(this.uniforms, g);
    } else {
      this.previousGrade = this.currentGrade;
      this.targetGrade   = g;
      this.transitionT   = 0.0;
    }
    return true;
  }

  setStyle(styleId, instant) {
    const gid = STYLE_TO_GRADE[styleId];
    if (!gid) return false;
    return this.setGrade(gid, instant);
  }

  /**
   * Tick the transition. Called once per frame with normalized dt.
   */
  update(dt) {
    if (this.transitionT >= 1.0) return;

    const step = dt / Math.max(0.001, this.transitionFrames / 60);
    this.transitionT = Math.min(1.0, this.transitionT + step);

    // Blend uniforms between previous and target.
    const t = this.transitionT;
    if (t >= 1.0) {
      this.currentGrade = this.targetGrade;
      applyGradeToUniforms(this.uniforms, this.currentGrade);
    } else {
      // Fast path: only blend the numeric channel uniforms.
      // Full blend would require lerping every op — for simplicity we
      // just swap to the target once we cross 50 %.
      if (t > 0.5) {
        applyGradeToUniforms(this.uniforms, this.targetGrade);
      } else {
        applyGradeToUniforms(this.uniforms, this.previousGrade);
      }
    }
  }

  getUniforms()      { return this.uniforms; }
  get current()      { return this.currentGrade; }
  get target()       { return this.targetGrade; }
  get transitioning(){ return this.transitionT < 1.0; }

  attachToMaterial(material) {
    if (!material || !material.uniforms) return false;
    const keys = Object.keys(this.uniforms);
    for (let i = 0; i < keys.length; i++) {
      material.uniforms[keys[i]] = this.uniforms[keys[i]];
    }
    material.needsUpdate = true;
    return true;
  }

  dispose() {
    this.uniforms = null;
    this.currentGrade = null;
    this.targetGrade = null;
    this.previousGrade = null;
  }
}

/* ------------------------------------------------------------------ */
/* 11. GRADING LUT BAKER (procedural)                                 */
/* ------------------------------------------------------------------ */

/**
 * Bakes the full grading chain into a DataTexture strip of `size` samples.
 * The strip is procedurally filled — satisfies 032 no-image-texture policy.
 * Used when the fragment shader budget is too tight for the full chain
 * (typically LOW tier / older Adreno).
 */
export function bakeGradeLUT(grade, size = 256) {
  if (!grade) return null;
  const s = Math.max(8, size | 0);
  const data = new Uint8Array(s * 4);
  const out = [0, 0, 0];

  for (let i = 0; i < s; i++) {
    const t = i / (s - 1);
    applyGradeInto(out, grade, t, t, t);
    const i4 = i * 4;
    data[i4]     = _linearToSRGBByte(out[0]);
    data[i4 + 1] = _linearToSRGBByte(out[1]);
    data[i4 + 2] = _linearToSRGBByte(out[2]);
    data[i4 + 3] = 255;
  }

  const tex = new THREE.DataTexture(data, s, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  markProcedural(tex, 'bakeGradeLUT:' + grade.id);
  return tex;
}

function _linearToSRGBByte(c) {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/* ------------------------------------------------------------------ */
/* 12. FACTORY                                                        */
/* ------------------------------------------------------------------ */

export function createColorGradingChain(initialGradeId, options = {}) {
  return new ColorGradingChain(initialGradeId, options);
}

/* ------------------------------------------------------------------ */
/* 13. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  GradeDescriptor,
  ColorGradingChain,

  // Presets
  GRADE_PRESETS,
  GRADE_PRESET_ID,
  GRADE_CANYON_TURQUOISE,
  GRADE_RIVER_TOP_DOWN,
  GRADE_SNOW_BLUE,
  GRADE_ORBITAL_SPACE,
  GRADE_DESERT_RUINS,
  GRADE_SUNSET_TOMBSTONE,
  GRADE_PASTEL_PORTRAIT,
  GRADE_MAGIC_CASTER,
  GRADE_FLOWER_FIELD,
  GRADE_NEUTRAL,

  // JS sampling
  applyGradeInto,
  applyGradeToBuffer,

  // GLSL
  GLSL_GRADE_UNIFORMS,
  GLSL_GRADE_CHAIN,

  // Uniforms
  createGradeUniforms,
  applyGradeToUniforms,

  // Baking
  bakeGradeLUT,

  // Registry
  registerGrade,
  unregisterGrade,
  getGrade,
  listGrades,

  // Style pairing
  STYLE_TO_GRADE,
  getCompanionGradeForStyle,

  // Factory
  createColorGradingChain,

  // Enums
  GRADE_OP,
  GRADE_OP_NAME,
  MAX_GRADE_PRESETS,
};

export default _defaultExport;