// File : 033
// name : src/core/033_rnd_ProceduralColorComposer.js
// description : Procedural color composition engine — the continuation of
//               032_rnd_NoImageTexturePolicy.js that turns GRAYSCALE
//               patterns (noise, cells, gradients, distance fields, SDF)
//               into fully colored anime compositions WITHOUT any bitmap
//               source. Every pixel's hue and chroma is derived at runtime
//               from:
//                 1. a scalar luminance field   (the "grayscale pattern"),
//                 2. a multi-stop palette        (JS data, uploaded as
//                                                 individual vec3 uniforms),
//                 3. a tone-mapping curve        (cel / posterize / smooth),
//                 4. a chromatic mixer           (multiply / screen /
//                                                 overlay / soft-light),
//                 5. atmospheric blend           (depth / fog / haze),
//                 6. backlight + rim              (per-style presets).
//
//               This module is the JS + GLSL side of the composition: it
//               exposes the palette data, the uniform blocks, and the GLSL
//               chunks that any anime material (water, snow, rock, foliage,
//               canyon, ruins, magic glow, pastel portrait, flower field)
//               uses to colorize its procedural pattern.
//
//               Style presets extracted from the reference image set:
//
//                 STYLE_CANYON_TURQUOISE   (image 1) — warm desert rock
//                                            edges + turquoise water +
//                                            white foam; strong chromatic
//                                            contrast, hard cel bands.
//                 STYLE_RIVER_TOP_DOWN     (image 2) — deep turquoise
//                                            gradient + green foliage
//                                            clusters + orange flower
//                                            accents; smooth + cel mix.
//                 STYLE_SNOW_BLUE_TOP_DOWN (image 3) — cold blue base +
//                                            white snow + dark navy
//                                            shadows; posterized with
//                                            soft edge dither.
//                 STYLE_ORBITAL_SPACE      (image 4) — deep navy + cyan
//                                            atmosphere band + white
//                                            starfield + cloud fractals.
//                 STYLE_DESERT_RUINS       (image 5) — warm sand + brick
//                                            red + moss green + dusty
//                                            haze; soft-light blend.
//                 STYLE_SUNSET_TOMBSTONE   (image 6) — sunset purple/orange
//                                            gradient + glowing mint
//                                            green cracks + pale stone.
//                 STYLE_PASTEL_PORTRAIT    (image 7) — pink + lavender +
//                                            white, soft glow, no hard
//                                            cel bands.
//                 STYLE_MAGIC_CASTER       (image 8) — dark olive + black +
//                                            golden magical glow;
//                                            rim-lit cel bands.
//                 STYLE_FLOWER_FIELD       (image 9) — magenta / violet /
//                                            white flower mix + deep blue
//                                            sky + backlit rim.
//
//               All palettes are stored as Float32Array (linear RGB) so
//               they can be uploaded as uniform arrays. All GLSL chunks
//               are string-exported so they compose into any ShaderMaterial
//               without image textures.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; no external palette libs; every internal array
//               sized once at construction.
// best for : Guaranteeing that the entire anime look from the reference
//            image set is produced procedurally — the same style, the same
//            chromatic composition, the same grayscale-to-color pipeline —
//            with zero bitmap dependency. Provides the shared color
//            vocabulary every downstream material (water, snow, rock,
//            foliage, sky, magic glow, portrait, flower) speaks.
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
  getDefaultLogger,
  LOG_CHANNEL,
} from './026_rnd_Logger.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const MAX_PALETTE_STOPS = 8;
export const MAX_PALETTES      = 32;

export const COMPOSE_MODE = Object.freeze({
  CEL_HARD:     0,
  CEL_SOFT:     1,
  POSTERIZE:    2,
  SMOOTH:       3,
  DITHERED:     4,
  BANDED:       5,
  COUNT:        6,
});

export const COMPOSE_MODE_NAME = Object.freeze([
  'cel_hard',
  'cel_soft',
  'posterize',
  'smooth',
  'dithered',
  'banded',
]);

export const BLEND_MODE = Object.freeze({
  REPLACE:    0,
  MULTIPLY:   1,
  SCREEN:     2,
  OVERLAY:    3,
  SOFT_LIGHT: 4,
  HARD_LIGHT: 5,
  COLOR_BURN: 6,
  COLOR_DODGE:7,
  ADD:        8,
  COUNT:      9,
});

export const BLEND_MODE_NAME = Object.freeze([
  'replace',
  'multiply',
  'screen',
  'overlay',
  'soft_light',
  'hard_light',
  'color_burn',
  'color_dodge',
  'add',
]);

/* ------------------------------------------------------------------ */
/* 1. STYLE PALETTES (linear-RGB Float32Array, 6-8 stops each)        */
/* ------------------------------------------------------------------ */

/**
 * Each style palette holds:
 *   name          — symbolic style id
 *   shadow        — [r,g,b] deep shadow
 *   dark          — [r,g,b] shadow
 *   mid           — [r,g,b] mid-tone
 *   light         — [r,g,b] lit tone
 *   highlight     — [r,g,b] highlight / rim
 *   accent        — [r,g,b] chromatic accent (flowers, glow, foam)
 *   ambient       — [r,g,b] ambient bounce
 *   fog           — [r,g,b] atmospheric blend
 *   glowColor     — [r,g,b] emissive / rim colour
 *   sky           — [r,g,b] sky / background
 *   satBias       — chromatic saturation bias (JS only)
 *   hueBias       — hue rotation in radians (JS only)
 *   mode          — COMPOSE_MODE
 *
 * All colours are in LINEAR RGB [0,1].
 * Every entry is frozen once constructed.
 */

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

function _freezePalette(p) {
  return Object.freeze(Object.assign({}, p, {
    shadow:    Object.freeze(p.shadow),
    dark:      Object.freeze(p.dark),
    mid:       Object.freeze(p.mid),
    light:     Object.freeze(p.light),
    highlight: Object.freeze(p.highlight),
    accent:    Object.freeze(p.accent),
    ambient:   Object.freeze(p.ambient),
    fog:       Object.freeze(p.fog),
    glowColor: Object.freeze(p.glowColor),
    sky:       Object.freeze(p.sky),
  }));
}

/* ---- STYLE 1 — CANYON TURQUOISE (image 1) ---------------------- */
export const STYLE_CANYON_TURQUOISE = _freezePalette({
  id:          'canyon_turquoise',
  name:        'Canyon Turquoise',
  shadow:      _lin(0x1a2a3a),
  dark:        _lin(0x2c5a6b),
  mid:         _lin(0x3ab8c8),
  light:       _lin(0x7ad8e0),
  highlight:   _lin(0xf0fbff),
  accent:      _lin(0xffffff),
  ambient:     _lin(0x8a7055),
  fog:         _lin(0xb0a080),
  glowColor:   _lin(0xfff4d8),
  sky:         _lin(0x7fb2d9),
  satBias:     1.15,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.CEL_HARD,
  bandCount:   4,
});

/* ---- STYLE 2 — RIVER TOP-DOWN (image 2) ------------------------ */
export const STYLE_RIVER_TOP_DOWN = _freezePalette({
  id:          'river_top_down',
  name:        'River Top-Down',
  shadow:      _lin(0x0a3a4a),
  dark:        _lin(0x1a6a7a),
  mid:         _lin(0x3ab0b8),
  light:       _lin(0x6dd0c0),
  highlight:   _lin(0xf0fffa),
  accent:      _lin(0xff8a40),
  ambient:     _lin(0x4a8a5a),
  fog:         _lin(0x3a7a8a),
  glowColor:   _lin(0xc8ffe8),
  sky:         _lin(0x4a9ac8),
  satBias:     1.10,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.CEL_SOFT,
  bandCount:   5,
});

/* ---- STYLE 3 — SNOW BLUE TOP-DOWN (image 3) -------------------- */
export const STYLE_SNOW_BLUE_TOP_DOWN = _freezePalette({
  id:          'snow_blue_top_down',
  name:        'Snow Blue Top-Down',
  shadow:      _lin(0x1a3a5a),
  dark:        _lin(0x3a6a9a),
  mid:         _lin(0x7aa8d8),
  light:       _lin(0xb8d8f0),
  highlight:   _lin(0xffffff),
  accent:      _lin(0xe8f4ff),
  ambient:     _lin(0x5a7a9a),
  fog:         _lin(0x9fc0e0),
  glowColor:   _lin(0xe0f0ff),
  sky:         _lin(0x9fc7ef),
  satBias:     0.85,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.POSTERIZE,
  bandCount:   5,
});

/* ---- STYLE 4 — ORBITAL SPACE (image 4) ------------------------- */
export const STYLE_ORBITAL_SPACE = _freezePalette({
  id:          'orbital_space',
  name:        'Orbital Space',
  shadow:      _lin(0x050a18),
  dark:        _lin(0x0a1a3a),
  mid:         _lin(0x1a3a6a),
  light:       _lin(0x4a8ac8),
  highlight:   _lin(0xf0f8ff),
  accent:      _lin(0xffd070),
  ambient:     _lin(0x0a1830),
  fog:         _lin(0x0a2040),
  glowColor:   _lin(0xa0d0ff),
  sky:         _lin(0x020408),
  satBias:     1.05,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.SMOOTH,
  bandCount:   6,
});

/* ---- STYLE 5 — DESERT RUINS (image 5) -------------------------- */
export const STYLE_DESERT_RUINS = _freezePalette({
  id:          'desert_ruins',
  name:        'Desert Ruins',
  shadow:      _lin(0x3a2818),
  dark:        _lin(0x6a4830),
  mid:         _lin(0xb88858),
  light:       _lin(0xe0b878),
  highlight:   _lin(0xfff0d0),
  accent:      _lin(0x8a5a30),
  ambient:     _lin(0xa08058),
  fog:         _lin(0xe0c8a0),
  glowColor:   _lin(0xffe8b0),
  sky:         _lin(0xd8e8f0),
  satBias:     0.95,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.SOFT ? COMPOSE_MODE.CEL_SOFT : COMPOSE_MODE.CEL_SOFT,
  bandCount:   4,
});

/* ---- STYLE 6 — SUNSET TOMBSTONE (image 6) ---------------------- */
export const STYLE_SUNSET_TOMBSTONE = _freezePalette({
  id:          'sunset_tombstone',
  name:        'Sunset Tombstone',
  shadow:      _lin(0x2a1a3a),
  dark:        _lin(0x5a3a6a),
  mid:         _lin(0xa070a8),
  light:       _lin(0xe8a870),
  highlight:   _lin(0xffe0a0),
  accent:      _lin(0x60ff90),
  ambient:     _lin(0x6a4a5a),
  fog:         _lin(0xd8a878),
  glowColor:   _lin(0x80ffa0),
  sky:         _lin(0xe8a078),
  satBias:     1.20,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.CEL_SOFT,
  bandCount:   5,
});

/* ---- STYLE 7 — PASTEL PORTRAIT (image 7) ----------------------- */
export const STYLE_PASTEL_PORTRAIT = _freezePalette({
  id:          'pastel_portrait',
  name:        'Pastel Portrait',
  shadow:      _lin(0x8a7090),
  dark:        _lin(0xc0a0c8),
  mid:         _lin(0xe8c8e0),
  light:       _lin(0xf8e0f0),
  highlight:   _lin(0xffffff),
  accent:      _lin(0xf0a0c8),
  ambient:     _lin(0xd8c8e8),
  fog:         _lin(0xe8d0f0),
  glowColor:   _lin(0xffe0f0),
  sky:         _lin(0xc8d0f0),
  satBias:     0.75,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.SMOOTH,
  bandCount:   6,
});

/* ---- STYLE 8 — MAGIC CASTER (image 8) -------------------------- */
export const STYLE_MAGIC_CASTER = _freezePalette({
  id:          'magic_caster',
  name:        'Magic Caster',
  shadow:      _lin(0x0a1008),
  dark:        _lin(0x1a2818),
  mid:         _lin(0x3a4a30),
  light:       _lin(0x6a7a50),
  highlight:   _lin(0xffd860),
  accent:      _lin(0xffb040),
  ambient:     _lin(0x2a3020),
  fog:         _lin(0x1a2018),
  glowColor:   _lin(0xffd870),
  sky:         _lin(0x1a2818),
  satBias:     1.25,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.CEL_HARD,
  bandCount:   4,
});

/* ---- STYLE 9 — FLOWER FIELD (image 9) -------------------------- */
export const STYLE_FLOWER_FIELD = _freezePalette({
  id:          'flower_field',
  name:        'Flower Field',
  shadow:      _lin(0x3a2858),
  dark:        _lin(0x6a4878),
  mid:         _lin(0xc868a0),
  light:       _lin(0xe898c8),
  highlight:   _lin(0xffe8f0),
  accent:      _lin(0xffe060),
  ambient:     _lin(0x8a6898),
  fog:         _lin(0xe0b8d8),
  glowColor:   _lin(0xffd8e8),
  sky:         _lin(0x3a78c8),
  satBias:     1.20,
  hueBias:     0.0,
  mode:        COMPOSE_MODE.CEL_SOFT,
  bandCount:   5,
});

/* ------------------------------------------------------------------ */
/* 2. STYLE REGISTRY                                                  */
/* ------------------------------------------------------------------ */

export const STYLE_PALETTES = Object.freeze({
  canyon_turquoise:    STYLE_CANYON_TURQUOISE,
  river_top_down:      STYLE_RIVER_TOP_DOWN,
  snow_blue_top_down:  STYLE_SNOW_BLUE_TOP_DOWN,
  orbital_space:       STYLE_ORBITAL_SPACE,
  desert_ruins:        STYLE_DESERT_RUINS,
  sunset_tombstone:    STYLE_SUNSET_TOMBSTONE,
  pastel_portrait:     STYLE_PASTEL_PORTRAIT,
  magic_caster:        STYLE_MAGIC_CASTER,
  flower_field:        STYLE_FLOWER_FIELD,
});

export const STYLE_ID = Object.freeze({
  CANYON_TURQUOISE:    'canyon_turquoise',
  RIVER_TOP_DOWN:      'river_top_down',
  SNOW_BLUE_TOP_DOWN:  'snow_blue_top_down',
  ORBITAL_SPACE:       'orbital_space',
  DESERT_RUINS:        'desert_ruins',
  SUNSET_TOMBSTONE:    'sunset_tombstone',
  PASTEL_PORTRAIT:     'pastel_portrait',
  MAGIC_CASTER:        'magic_caster',
  FLOWER_FIELD:        'flower_field',
});

/* ------------------------------------------------------------------ */
/* 3. PALETTE → UNIFORM BLOCK                                         */
/* ------------------------------------------------------------------ */

/**
 * Converts a style palette into a flat uniform block that any
 * ShaderMaterial can consume:
 *
 *   {
 *     uPaletteShadow:    { value: new THREE.Vector3(...) },
 *     uPaletteDark:      { value: new THREE.Vector3(...) },
 *     uPaletteMid:       { value: new THREE.Vector3(...) },
 *     uPaletteLight:     { value: new THREE.Vector3(...) },
 *     uPaletteHighlight: { value: new THREE.Vector3(...) },
 *     uPaletteAccent:    { value: new THREE.Vector3(...) },
 *     uPaletteAmbient:   { value: new THREE.Vector3(...) },
 *     uPaletteFog:       { value: new THREE.Vector3(...) },
 *     uPaletteGlow:      { value: new THREE.Vector3(...) },
 *     uPaletteSky:       { value: new THREE.Vector3(...) },
 *     uPaletteBandCount: { value: 4.0 },
 *     uPaletteMode:      { value: 0.0 },
 *     uPaletteSatBias:   { value: 1.0 },
 *     uPaletteHueBias:   { value: 0.0 },
 *   }
 *
 * Every value is a THREE.Vector3 / number — never a texture.
 */
export function createPaletteUniforms(palette) {
  if (!palette) return null;
  return {
    uPaletteShadow:    { value: new THREE.Vector3(palette.shadow[0],    palette.shadow[1],    palette.shadow[2]) },
    uPaletteDark:      { value: new THREE.Vector3(palette.dark[0],      palette.dark[1],      palette.dark[2]) },
    uPaletteMid:       { value: new THREE.Vector3(palette.mid[0],       palette.mid[1],       palette.mid[2]) },
    uPaletteLight:     { value: new THREE.Vector3(palette.light[0],     palette.light[1],     palette.light[2]) },
    uPaletteHighlight: { value: new THREE.Vector3(palette.highlight[0], palette.highlight[1], palette.highlight[2]) },
    uPaletteAccent:    { value: new THREE.Vector3(palette.accent[0],    palette.accent[1],    palette.accent[2]) },
    uPaletteAmbient:   { value: new THREE.Vector3(palette.ambient[0],   palette.ambient[1],   palette.ambient[2]) },
    uPaletteFog:       { value: new THREE.Vector3(palette.fog[0],       palette.fog[1],       palette.fog[2]) },
    uPaletteGlow:      { value: new THREE.Vector3(palette.glowColor[0], palette.glowColor[1], palette.glowColor[2]) },
    uPaletteSky:       { value: new THREE.Vector3(palette.sky[0],       palette.sky[1],       palette.sky[2]) },
    uPaletteBandCount: { value: palette.bandCount || 4.0 },
    uPaletteMode:      { value: palette.mode | 0 },
    uPaletteSatBias:   { value: palette.satBias || 1.0 },
    uPaletteHueBias:   { value: palette.hueBias || 0.0 },
  };
}

/**
 * Applies a palette to an existing uniform object in place. Use this
 * to swap styles without reallocating uniforms.
 */
export function applyPaletteToUniforms(uniforms, palette) {
  if (!uniforms || !palette) return false;
  uniforms.uPaletteShadow.value.set(palette.shadow[0], palette.shadow[1], palette.shadow[2]);
  uniforms.uPaletteDark.value.set(palette.dark[0], palette.dark[1], palette.dark[2]);
  uniforms.uPaletteMid.value.set(palette.mid[0], palette.mid[1], palette.mid[2]);
  uniforms.uPaletteLight.value.set(palette.light[0], palette.light[1], palette.light[2]);
  uniforms.uPaletteHighlight.value.set(palette.highlight[0], palette.highlight[1], palette.highlight[2]);
  uniforms.uPaletteAccent.value.set(palette.accent[0], palette.accent[1], palette.accent[2]);
  uniforms.uPaletteAmbient.value.set(palette.ambient[0], palette.ambient[1], palette.ambient[2]);
  uniforms.uPaletteFog.value.set(palette.fog[0], palette.fog[1], palette.fog[2]);
  uniforms.uPaletteGlow.value.set(palette.glowColor[0], palette.glowColor[1], palette.glowColor[2]);
  uniforms.uPaletteSky.value.set(palette.sky[0], palette.sky[1], palette.sky[2]);
  uniforms.uPaletteBandCount.value = palette.bandCount || 4.0;
  uniforms.uPaletteMode.value = palette.mode | 0;
  uniforms.uPaletteSatBias.value = palette.satBias || 1.0;
  uniforms.uPaletteHueBias.value = palette.hueBias || 0.0;
  return true;
}

/* ------------------------------------------------------------------ */
/* 4. GLSL CHUNKS                                                     */
/* ------------------------------------------------------------------ */

/**
 * Palettes as GLSL uniforms.
 */
export const GLSL_PALETTE_UNIFORMS = /* glsl */`
uniform vec3  uPaletteShadow;
uniform vec3  uPaletteDark;
uniform vec3  uPaletteMid;
uniform vec3  uPaletteLight;
uniform vec3  uPaletteHighlight;
uniform vec3  uPaletteAccent;
uniform vec3  uPaletteAmbient;
uniform vec3  uPaletteFog;
uniform vec3  uPaletteGlow;
uniform vec3  uPaletteSky;
uniform float uPaletteBandCount;
uniform float uPaletteMode;
uniform float uPaletteSatBias;
uniform float uPaletteHueBias;
`;

/**
 * Core colorization: takes a scalar luminance [0,1] and returns the
 * palette color that matches, using the mode:
 *   0 cel_hard     — hard quantized bands
 *   1 cel_soft     — quantized with smoothstep transitions
 *   2 posterize    — geometric quantization
 *   3 smooth       — continuous 6-stop interpolation
 *   4 dithered     — smooth with ordered dither
 *   5 banded       — post-smooth color band separation
 */
export const GLSL_PALETTE_SAMPLE = /* glsl */`
vec3 paletteCelHard(float t, float bands) {
  float b = max(bands, 2.0);
  float q = floor(clamp(t, 0.0, 1.0) * b) / (b - 1.0);
  q = clamp(q, 0.0, 1.0);

  if (q < 0.20) return uPaletteShadow;
  if (q < 0.35) return uPaletteDark;
  if (q < 0.55) return uPaletteMid;
  if (q < 0.75) return uPaletteLight;
  return uPaletteHighlight;
}

vec3 paletteCelSoft(float t, float bands) {
  float b = max(bands, 2.0);
  float q = floor(clamp(t, 0.0, 1.0) * b) / (b - 1.0);

  // Smooth blend between adjacent bands using a narrow transition.
  vec3 a, bb;
  float t01 = clamp(t, 0.0, 1.0);
  if (t01 < 0.20) {
    a = uPaletteShadow; bb = uPaletteDark;
    return mix(a, bb, smoothstep(0.10, 0.25, t01));
  }
  if (t01 < 0.40) {
    a = uPaletteDark; bb = uPaletteMid;
    return mix(a, bb, smoothstep(0.30, 0.45, t01));
  }
  if (t01 < 0.60) {
    a = uPaletteMid; bb = uPaletteLight;
    return mix(a, bb, smoothstep(0.50, 0.65, t01));
  }
  if (t01 < 0.80) {
    a = uPaletteLight; bb = uPaletteHighlight;
    return mix(a, bb, smoothstep(0.70, 0.85, t01));
  }
  return uPaletteHighlight;
}

vec3 palettePosterize(float t, float bands) {
  float b = max(bands, 2.0);
  float q = floor(clamp(t, 0.0, 1.0) * b) / (b - 1.0);
  return mix(uPaletteShadow, uPaletteHighlight, q);
}

vec3 paletteSmooth(float t) {
  float q = clamp(t, 0.0, 1.0);
  if (q < 0.20) return mix(uPaletteShadow, uPaletteDark, q / 0.20);
  if (q < 0.40) return mix(uPaletteDark, uPaletteMid, (q - 0.20) / 0.20);
  if (q < 0.60) return mix(uPaletteMid, uPaletteLight, (q - 0.40) / 0.20);
  if (q < 0.80) return mix(uPaletteLight, uPaletteHighlight, (q - 0.60) / 0.20);
  return uPaletteHighlight;
}

float paletteBayer4(vec2 p) {
  // 4x4 ordered dither matrix normalized to [0,1)
  int x = int(mod(p.x, 4.0));
  int y = int(mod(p.y, 4.0));
  int idx = x + y * 4;

  // Row-major Bayer 4x4 / 16.0
  const float bayer[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0
  );
  return bayer[idx] / 16.0;
}

vec3 paletteDithered(float t, float bands, vec2 fragCoord) {
  float b = max(bands, 2.0);
  float d = paletteBayer4(fragCoord) - 0.5;
  float biased = clamp(t + d / b, 0.0, 1.0);
  return paletteCelSoft(biased, b);
}

vec3 paletteBanded(float t, float bands) {
  float b = max(bands, 2.0);
  float q = floor(clamp(t, 0.0, 1.0) * b) / b;
  float q2 = clamp(q + 0.5 / b, 0.0, 1.0);
  return paletteSmooth(q2);
}

vec3 samplePalette(float t, float bands, float mode, vec2 fragCoord) {
  if (mode < 0.5) return paletteCelHard(t, bands);
  if (mode < 1.5) return paletteCelSoft(t, bands);
  if (mode < 2.5) return palettePosterize(t, bands);
  if (mode < 3.5) return paletteSmooth(t);
  if (mode < 4.5) return paletteDithered(t, bands, fragCoord);
  return paletteBanded(t, bands);
}
`;

/**
 * Chromatic helpers: hue/saturation bias, gamma-correct blending, and the
 * blend modes used to mix grayscale patterns with palette colors.
 */
export const GLSL_CHROMATIC = /* glsl */`
vec3 linearToSrgbApprox(vec3 c) {
  return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}

vec3 srgbToLinearApprox(vec3 c) {
  return pow(clamp(c, 0.0, 1.0), vec3(2.2));
}

float luminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 applyChroma(vec3 c, float satBias) {
  float y = luminance(c);
  return clamp(mix(vec3(y), c, satBias), 0.0, 1.0);
}

vec3 hueRotate(vec3 c, float angle) {
  // Rodrigues rotation of the RGB cube around the (1,1,1)/sqrt(3) axis.
  float cs = cos(angle);
  float sn = sin(angle);
  float k = (1.0 - cs) / 3.0;
  mat3 rot = mat3(
    cs + k,        k - sn * 0.57735, k + sn * 0.57735,
    k + sn * 0.57735, cs + k,        k - sn * 0.57735,
    k - sn * 0.57735, k + sn * 0.57735, cs + k
  );
  return clamp(rot * c, 0.0, 1.0);
}

vec3 blendMultiply(vec3 a, vec3 b) { return a * b; }
vec3 blendScreen(vec3 a, vec3 b)   { return 1.0 - (1.0 - a) * (1.0 - b); }
vec3 blendOverlay(vec3 a, vec3 b)  {
  return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(0.5, a));
}
vec3 blendSoftLight(vec3 a, vec3 b) {
  vec3 lo = 2.0 * a * b + a * a * (1.0 - 2.0 * b);
  vec3 hi = sqrt(a) * (2.0 * b - 1.0) + 2.0 * a * (1.0 - b);
  return mix(lo, hi, step(0.5, b));
}
vec3 blendHardLight(vec3 a, vec3 b) { return blendOverlay(b, a); }
vec3 blendColorBurn(vec3 a, vec3 b) { return 1.0 - min(vec3(1.0), (1.0 - a) / max(b, vec3(1e-4))); }
vec3 blendColorDodge(vec3 a, vec3 b){ return min(vec3(1.0), a / max(1.0 - b, vec3(1e-4))); }
vec3 blendAdd(vec3 a, vec3 b)       { return min(vec3(1.0), a + b); }

vec3 applyBlend(vec3 a, vec3 b, float mode) {
  if (mode < 0.5) return b;
  if (mode < 1.5) return blendMultiply(a, b);
  if (mode < 2.5) return blendScreen(a, b);
  if (mode < 3.5) return blendOverlay(a, b);
  if (mode < 4.5) return blendSoftLight(a, b);
  if (mode < 5.5) return blendHardLight(a, b);
  if (mode < 6.5) return blendColorBurn(a, b);
  if (mode < 7.5) return blendColorDodge(a, b);
  return blendAdd(a, b);
}
`;

/**
 * Atmosphere / depth / haze — the layer that gives every reference image
 * its characteristic "distance fade" toward the fog color.
 */
export const GLSL_ATMOSPHERE_BLEND = /* glsl */`
vec3 applyAtmosphere(vec3 c, float depth01, float density) {
  float d = clamp(depth01, 0.0, 1.0);
  float k = 1.0 - exp(-d * d * density);
  return mix(c, uPaletteFog, clamp(k, 0.0, 1.0));
}

vec3 applyAmbientBounce(vec3 c, float upDot, float strength) {
  float up = clamp(upDot * 0.5 + 0.5, 0.0, 1.0);
  return mix(c, c * uPaletteAmbient * 2.0, up * strength);
}

vec3 applyBacklight(vec3 c, float rimFactor, float strength) {
  float r = pow(clamp(rimFactor, 0.0, 1.0), 2.5);
  return c + uPaletteGlow * r * strength;
}

vec3 applyGlow(vec3 c, float glow01, float strength) {
  return c + uPaletteGlow * clamp(glow01, 0.0, 1.0) * strength;
}
`;

/**
 * Full compositor chunk: grayscale → palette → chroma → blend → atmosphere.
 */
export const GLSL_PROCEDURAL_COMPOSITOR = /* glsl */`
${GLSL_PALETTE_UNIFORMS}
${GLSL_PALETTE_SAMPLE}
${GLSL_CHROMATIC}
${GLSL_ATMOSPHERE_BLEND}

// Final composer. Inputs:
//   grayPattern  — scalar grayscale value in [0,1]
//   accentMask   — scalar accent weight in [0,1] (flowers, foam, glow)
//   depth01      — linear depth [0,1] for atmospheric fade
//   rimFactor    — backlight rim factor [0,1]
//   glow01       — emissive glow [0,1]
//   fragCoord    — gl_FragCoord.xy for dithering
//
// Output: fully composed palette color.
vec3 composeFromGrayscale(
  float grayPattern,
  float accentMask,
  float depth01,
  float rimFactor,
  float glow01,
  vec2 fragCoord
) {
  // 1. Sample palette at the grayscale value.
  vec3 base = samplePalette(grayPattern, uPaletteBandCount, uPaletteMode, fragCoord);

  // 2. Chromatic bias (saturation + hue).
  base = applyChroma(base, uPaletteSatBias);
  if (abs(uPaletteHueBias) > 1e-4) base = hueRotate(base, uPaletteHueBias);

  // 3. Accent blend (soft light) toward the accent color.
  base = applyBlend(base, uPaletteAccent, 4.0);
  base = mix(base, base * (1.0 + accentMask), accentMask);

  // 4. Ambient bounce using palette ambient.
  base = applyAmbientBounce(base, 0.5, 0.25);

  // 5. Atmospheric fade.
  base = applyAtmosphere(base, depth01, 1.2);

  // 6. Rim light.
  base = applyBacklight(base, rimFactor, 0.35);

  // 7. Emissive glow.
  base = applyGlow(base, glow01, 0.6);

  return clamp(base, 0.0, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 5. PRESET DRIVEN COMPOSER (JS API)                                 */
/* ------------------------------------------------------------------ */

export class ProceduralColorComposer {
  constructor(styleId, options = {}) {
    this.styleId = styleId || STYLE_ID.CANYON_TURQUOISE;
    this.palette = STYLE_PALETTES[this.styleId] || STYLE_CANYON_TURQUOISE;

    this.options = Object.assign({
      bandCount:  null,
      satBias:    null,
      hueBias:    null,
      mode:       null,
      atmosphereDensity: 1.2,
      ambientStrength:   0.25,
      rimStrength:       0.35,
      glowStrength:      0.60,
    }, options || {});

    // Per-instance overrides.
    this._bandCount = this.options.bandCount !== null ? this.options.bandCount : this.palette.bandCount;
    this._satBias   = this.options.satBias   !== null ? this.options.satBias   : this.palette.satBias;
    this._hueBias   = this.options.hueBias   !== null ? this.options.hueBias   : this.palette.hueBias;
    this._mode      = this.options.mode      !== null ? this.options.mode      : this.palette.mode;

    this.uniforms = createPaletteUniforms(this.palette);
    this._pushOverrides();
  }

  _pushOverrides() {
    if (!this.uniforms) return;
    this.uniforms.uPaletteBandCount.value = this._bandCount;
    this.uniforms.uPaletteSatBias.value   = this._satBias;
    this.uniforms.uPaletteHueBias.value   = this._hueBias;
    this.uniforms.uPaletteMode.value      = this._mode;
  }

  setStyle(styleId) {
    const p = STYLE_PALETTES[styleId];
    if (!p) return false;
    this.styleId = styleId;
    this.palette = p;
    applyPaletteToUniforms(this.uniforms, p);
    this._bandCount = p.bandCount;
    this._satBias   = p.satBias;
    this._hueBias   = p.hueBias;
    this._mode      = p.mode;
    this._pushOverrides();
    return true;
  }

  setMode(mode) {
    this._mode = mode | 0;
    this.uniforms.uPaletteMode.value = this._mode;
    return this;
  }

  setBandCount(n) {
    this._bandCount = Math.max(2, Math.min(16, n | 0));
    this.uniforms.uPaletteBandCount.value = this._bandCount;
    return this;
  }

  setSaturation(s) {
    this._satBias = Math.max(0, Math.min(3, Number(s)));
    this.uniforms.uPaletteSatBias.value = this._satBias;
    return this;
  }

  setHueRotation(radians) {
    this._hueBias = Number(radians) || 0;
    this.uniforms.uPaletteHueBias.value = this._hueBias;
    return this;
  }

  getUniforms()   { return this.uniforms; }
  getPalette()    { return this.palette; }
  get style()     { return this.styleId; }
  get mode()      { return this._mode; }
  get bandCount() { return this._bandCount; }

  /**
   * Attaches the composer's uniforms to a ShaderMaterial so the GLSL
   * chunks can access them. Material.fragmentShader must include
   * GLSL_PROCEDURAL_COMPOSITOR.
   */
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
    this.palette  = null;
  }
}

/* ------------------------------------------------------------------ */
/* 6. PROCEDURAL GRAYSCALE PATTERN GENERATORS                         */
/* ------------------------------------------------------------------ */

/**
 * JS-side helpers to compute a grayscale luminance field per pixel
 * (for CPU fallback / LUT baking). All accept (x, y, w, h) → [0,1].
 */
export const GRAYSCALE_GENERATORS = Object.freeze({
  /**
   * Linear vertical gradient.
   */
  linearVertical: (x, y, w, h) => (h > 1 ? y / (h - 1) : 0),

  /**
   * Linear horizontal gradient.
   */
  linearHorizontal: (x, y, w, h) => (w > 1 ? x / (w - 1) : 0),

  /**
   * Radial gradient from center.
   */
  radialCenter: (x, y, w, h) => {
    const cx = w * 0.5, cy = h * 0.5;
    const dx = (x - cx) / cx, dy = (y - cy) / cy;
    return Math.min(1, Math.sqrt(dx * dx + dy * dy));
  },

  /**
   * Diagonal gradient.
   */
  diagonal: (x, y, w, h) => {
    const a = w > 1 ? x / (w - 1) : 0;
    const b = h > 1 ? y / (h - 1) : 0;
    return (a + b) * 0.5;
  },

  /**
   * Smooth 2D Perlin-ish fbm using hash noise (matches GLSL FBM shape).
   */
  fbm2D: (x, y, w, h, octaves = 4, seed = 0) => {
    const fx = x / Math.max(1, w);
    const fy = y / Math.max(1, h);
    let sum = 0, amp = 1, freq = 1, norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * _hashNoise(fx * freq * 8 + seed, fy * freq * 8 + seed * 0.7);
      norm += amp;
      amp *= 0.5;
      freq *= 2;
    }
    return norm > 0 ? sum / norm : 0;
  },

  /**
   * Worley / cellular pattern (F1 distance), grayscale inverted.
   */
  worleyCells: (x, y, w, h, cellsX = 6, cellsY = 6, seed = 0) => {
    const fx = x / Math.max(1, w) * cellsX;
    const fy = y / Math.max(1, h) * cellsY;
    const ix = Math.floor(fx), iy = Math.floor(fy);
    let minD = 1e9;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = ix + dx, cy = iy + dy;
        const hx = _hashNoise(cx, cy, seed);
        const hy = _hashNoise(cx + 91.7, cy + 13.4, seed);
        const px = cx + hx, py = cy + hy;
        const ddx = px - fx, ddy = py - fy;
        const d = ddx * ddx + ddy * ddy;
        if (d < minD) minD = d;
      }
    }
    return Math.min(1, Math.sqrt(minD));
  },

  /**
   * Cracked stone pattern (approximate image 5 / 6).
   */
  crackedStone: (x, y, w, h, seed = 0) => {
    const w1 = GRAYSCALE_GENERATORS.worleyCells(x, y, w, h, 8, 8, seed);
    const w2 = GRAYSCALE_GENERATORS.worleyCells(x, y, w, h, 14, 14, seed + 3.7);
    const crack = Math.max(w1, w2);
    return Math.pow(crack, 0.6);
  },

  /**
   * Snow cluster pattern (approximate image 3).
   */
  snowClusters: (x, y, w, h, seed = 0) => {
    const fbm = GRAYSCALE_GENERATORS.fbm2D(x, y, w, h, 3, seed);
    const cells = GRAYSCALE_GENERATORS.worleyCells(x, y, w, h, 12, 12, seed + 11);
    const cluster = Math.max(0, fbm - cells * 0.6 + 0.3);
    return Math.min(1, cluster);
  },

  /**
   * Star field density (approximate image 4).
   */
  starField: (x, y, w, h, seed = 0) => {
    const h1 = _hashNoise(Math.floor(x / 2), Math.floor(y / 2), seed);
    if (h1 > 0.985) return 1.0;
    if (h1 > 0.97)  return 0.6;
    if (h1 > 0.95)  return 0.3;
    return 0;
  },

  /**
   * Flower cluster pattern (approximate image 9).
   */
  flowerClusters: (x, y, w, h, seed = 0) => {
    const cells = GRAYSCALE_GENERATORS.worleyCells(x, y, w, h, 10, 10, seed);
    const edges = Math.max(0, 1 - cells * 3);
    const mask = GRAYSCALE_GENERATORS.fbm2D(x, y, w, h, 2, seed + 7);
    return Math.min(1, Math.max(0, edges * mask * 1.6));
  },

  /**
   * Magic glow radial (approximate image 8).
   */
  magicGlow: (x, y, w, h) => {
    const cx = w * 0.5, cy = h * 0.75;
    const dx = (x - cx) / (w * 0.5);
    const dy = (y - cy) / (h * 0.5);
    const d = Math.sqrt(dx * dx + dy * dy);
    return Math.max(0, 1 - d * 1.2);
  },
});

function _hashNoise(x, y, seed) {
  const sx = Math.sin(x * 12.9898 + y * 78.233 + seed * 4.1414) * 43758.5453;
  return sx - Math.floor(sx);
}

/* ------------------------------------------------------------------ */
/* 7. PROCEDURAL GRAYSCALE-TO-PALETTE LUT                             */
/* ------------------------------------------------------------------ */

/**
 * Bakes a 256×1 palette LUT into a DataTexture that the shader samples
 * with a single `texture2D(uPaletteLUT, vec2(gray, 0.5))` call. This is
 * the "grayscale → color" fast path for cases where the per-pixel
 * palette math is too expensive on the vertex/fragment budget.
 *
 * The LUT is procedurally generated (no image), so it satisfies the
 * 032 policy.
 */
export function bakePaletteLUT(palette, size = 256) {
  if (!palette) return null;
  const s = Math.max(8, size | 0);
  const data = new Uint8Array(s * 4);

  for (let i = 0; i < s; i++) {
    const t = i / (s - 1);
    const rgb = _samplePaletteJS(palette, t);

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

  markProcedural(tex, 'bakePaletteLUT:' + palette.id);
  return tex;
}

function _samplePaletteJS(palette, t) {
  const q = Math.max(0, Math.min(1, t));
  if (q < 0.20) return _mix3(palette.shadow, palette.dark, q / 0.20);
  if (q < 0.40) return _mix3(palette.dark,   palette.mid,  (q - 0.20) / 0.20);
  if (q < 0.60) return _mix3(palette.mid,    palette.light, (q - 0.40) / 0.20);
  if (q < 0.80) return _mix3(palette.light,  palette.highlight, (q - 0.60) / 0.20);
  return palette.highlight;
}

function _mix3(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function _linearToSRGBByte(c) {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/* ------------------------------------------------------------------ */
/* 8. MODULE-LEVEL COMPOSER REGISTRY                                  */
/* ------------------------------------------------------------------ */

const _composerRegistry = new Map();

export function getComposer(styleId, options) {
  if (_composerRegistry.has(styleId)) return _composerRegistry.get(styleId);
  const composer = new ProceduralColorComposer(styleId, options);
  _composerRegistry.set(styleId, composer);
  return composer;
}

export function disposeComposer(styleId) {
  const c = _composerRegistry.get(styleId);
  if (!c) return false;
  c.dispose();
  _composerRegistry.delete(styleId);
  return true;
}

export function disposeAllComposers() {
  for (const c of _composerRegistry.values()) c.dispose();
  _composerRegistry.clear();
}

export function listComposerStyles() {
  return Array.from(_composerRegistry.keys());
}

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createColorComposer(styleId, options = {}) {
  return new ProceduralColorComposer(styleId, options);
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  ProceduralColorComposer,
  createColorComposer,

  // Style palettes
  STYLE_PALETTES,
  STYLE_ID,
  STYLE_CANYON_TURQUOISE,
  STYLE_RIVER_TOP_DOWN,
  STYLE_SNOW_BLUE_TOP_DOWN,
  STYLE_ORBITAL_SPACE,
  STYLE_DESERT_RUINS,
  STYLE_SUNSET_TOMBSTONE,
  STYLE_PASTEL_PORTRAIT,
  STYLE_MAGIC_CASTER,
  STYLE_FLOWER_FIELD,

  // Uniforms
  createPaletteUniforms,
  applyPaletteToUniforms,

  // GLSL chunks
  GLSL_PALETTE_UNIFORMS,
  GLSL_PALETTE_SAMPLE,
  GLSL_CHROMATIC,
  GLSL_ATMOSPHERE_BLEND,
  GLSL_PROCEDURAL_COMPOSITOR,

  // JS generators
  GRAYSCALE_GENERATORS,
  bakePaletteLUT,

  // Registry
  getComposer,
  disposeComposer,
  disposeAllComposers,
  listComposerStyles,

  COMPOSE_MODE,
  COMPOSE_MODE_NAME,
  BLEND_MODE,
  BLEND_MODE_NAME,
  MAX_PALETTE_STOPS,
  MAX_PALETTES,
};

export default _defaultExport;