// File : 036
// name : src/core/036_rnd_MaterialColorPipeline.js
// description : Material-level color pipeline that fuses the three
//               procedural color layers into one call:
//
//                 • 033_rnd_ProceduralColorComposer.js   (grayscale → palette)
//                 • 034_rnd_GradientGenerators.js         (smooth ramps)
//                 • 035_rnd_ColorGradingChain.js          (final look)
//
//               Every anime material (water, snow, rock, foliage, canyon,
//               character skin, hair, cloth, magic glow, sky, clouds, foam,
//               flower, moss, brick, mossy stone, tombstone) acquires a
//               MaterialColorPipeline instead of hand-wiring the three
//               modules. The pipeline exposes:
//
//                 composeRGB(input, out)      — full JS pipeline
//                 composeGLSL()               — GLSL chunk string
//                 uniforms()                  — merged uniform block
//                 bakeLUTs()                  — one-shot procedural bake for
//                                               LOW tier devices
//                 attachToMaterial(material)  — merge uniforms + GLSL into
//                                               a THREE.ShaderMaterial
//
//               The GLSL chunk output is a single `vec3
//               applyMaterialColor(vec2 uv, float grayPattern, float
//               accentMask, float depth01, float rimFactor, float glow01)`
//               function that composes palette + gradient + grading in one
//               call — so each material's fragment shader only has ONE
//               include and ONE function call site.
//
//               Material presets (matching the reference image set):
//                 MAT_WATER_TURQUOISE     (image 1, 2)
//                 MAT_WATER_DEEP          (image 1 abyss)
//                 MAT_FOAM_WHITE          (image 1, 2)
//                 MAT_SNOW_BRIGHT         (image 3)
//                 MAT_SNOW_SHADOW         (image 3)
//                 MAT_ICE_TURQUOISE       (image 3 creek)
//                 MAT_ROCK_WARM           (image 1, 5, 6)
//                 MAT_ROCK_COOL           (image 3 cliff)
//                 MAT_SAND_DESERT         (image 5)
//                 MAT_BRICK_WEATHERED     (image 5)
//                 MAT_MOSS_GREEN          (image 5)
//                 MAT_STONE_CRACKED       (image 6)
//                 MAT_CRYSTAL_MINT_GLOW   (image 6)
//                 MAT_FOLIAGE_BRIGHT      (image 2)
//                 MAT_FOLIAGE_DARK        (image 2 shadow)
//                 MAT_FLOWER_MAGENTA      (image 9)
//                 MAT_FLOWER_VIOLET       (image 9)
//                 MAT_FLOWER_WHITE        (image 9)
//                 MAT_SKY_DAY             (image 1, 9)
//                 MAT_SKY_SUNSET          (image 6)
//                 MAT_SKY_SPACE           (image 4)
//                 MAT_CLOUD_WHITE         (image 1, 9)
//                 MAT_CHARACTER_SKIN      (image 7)
//                 MAT_CHARACTER_HAIR_PINK (image 7)
//                 MAT_CHARACTER_CLOTH_LAV (image 7)
//                 MAT_MAGIC_GLOW_GOLD     (image 8)
//                 MAT_MAGIC_CLOAK_GREEN   (image 8)
//
//               Per-tier behavior:
//                 HIGH    — full per-pixel composition, no LUT
//                 MEDIUM  — full per-pixel composition, optionally LUT
//                 LOW     — bake palette + gradient + grade LUTs, sample
//                           three textures per fragment
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external color libs; every internal array sized once
//               at construction.
// best for : Guaranteeing every anime material in the lighting stack speaks
//            the same color language — same palette system, same gradient
//            system, same grading system — and can be tuned per-device-tier
//            without touching shader code, so the reference-image look is
//            reproduced on any Android GPU.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js';

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  STYLE_ID,
  STYLE_PALETTES,
  ProceduralColorComposer,
  createPaletteUniforms,
  applyPaletteToUniforms,
  bakePaletteLUT,
  GLSL_PALETTE_UNIFORMS,
  GLSL_PALETTE_SAMPLE,
  GLSL_CHROMATIC,
  GLSL_ATMOSPHERE_BLEND,
  GLSL_PROCEDURAL_COMPOSITOR,
  COMPOSE_MODE,
} from './033_rnd_ProceduralColorComposer.js';

import {
  GRADIENT_PRESETS,
  GRADIENT_PRESET_ID,
  GradientDescriptor,
  sampleGradient1DInto,
  sampleGradient2DInto,
  createGradientUniforms,
  applyGradientToUniforms,
  bakeGradientLUT,
  bakeGradientLUT2D,
  GLSL_EASING,
  GLSL_GRADIENT_UNIFORMS,
  GLSL_GRADIENT_SAMPLE,
  GRADIENT_KIND,
  EASE_KIND,
  getCompanionGradientForStyle,
} from './034_rnd_GradientGenerators.js';

import {
  GRADE_PRESETS,
  GRADE_PRESET_ID,
  GradeDescriptor,
  applyGradeInto,
  createGradeUniforms,
  applyGradeToUniforms,
  bakeGradeLUT,
  ColorGradingChain,
  GLSL_GRADE_UNIFORMS,
  GLSL_GRADE_CHAIN,
  getCompanionGradeForStyle,
} from './035_rnd_ColorGradingChain.js';

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

export const MATERIAL_KIND = Object.freeze({
  WATER:       0,
  SNOW:        1,
  ICE:         2,
  ROCK:        3,
  SAND:        4,
  BRICK:       5,
  MOSS:        6,
  STONE:       7,
  CRYSTAL:     8,
  FOLIAGE:     9,
  FLOWER:     10,
  SKY:        11,
  CLOUD:      12,
  CHARACTER:  13,
  MAGIC:      14,
  COUNT:      15,
});

export const MATERIAL_KIND_NAME = Object.freeze([
  'water',
  'snow',
  'ice',
  'rock',
  'sand',
  'brick',
  'moss',
  'stone',
  'crystal',
  'foliage',
  'flower',
  'sky',
  'cloud',
  'character',
  'magic',
]);

/**
 * LUT baking strategy per PERF_TIER.
 *   HIGH / MEDIUM: per-pixel composition (no LUT textures)
 *   LOW:            three LUT textures (palette 1D, gradient 1D, grade 1D)
 */
export const USE_LUT_ON_TIER = Object.freeze({
  HIGH:   false,
  MEDIUM: false,
  LOW:    true,
});

/* ------------------------------------------------------------------ */
/* 1. MATERIAL PRESETS (descriptor)                                   */
/* ------------------------------------------------------------------ */

/**
 * A material color preset binds:
 *   styleId      — palette from 033
 *   gradientId   — gradient from 034
 *   gradeId      — grading from 035
 *   kind         — MATERIAL_KIND
 *   smoothBlend  — 0..1 blend weight between palette (0) and gradient (1)
 *   accentBlend  — 0..1 blend weight for accent color (foam, flower tip, glow)
 *   atmosphere   — 0..1 atmosphere density multiplier
 *   rimStrength  — 0..1 rim light strength
 *   glowStrength — 0..1 emissive glow strength
 *   bandOverride — override palette band count (or null)
 *   satOverride  — override palette saturation (or null)
 */
export class MaterialColorPreset {
  constructor(spec) {
    this.id           = spec.id;
    this.name         = spec.name || spec.id;
    this.kind         = spec.kind !== undefined ? spec.kind : MATERIAL_KIND.ROCK;
    this.styleId      = spec.styleId;
    this.gradientId   = spec.gradientId;
    this.gradeId      = spec.gradeId;

    this.smoothBlend  = spec.smoothBlend  !== undefined ? spec.smoothBlend  : 0.35;
    this.accentBlend  = spec.accentBlend  !== undefined ? spec.accentBlend  : 0.0;
    this.atmosphere   = spec.atmosphere   !== undefined ? spec.atmosphere   : 1.0;
    this.rimStrength  = spec.rimStrength  !== undefined ? spec.rimStrength  : 0.35;
    this.glowStrength = spec.glowStrength !== undefined ? spec.glowStrength : 0.0;
    this.bandOverride = spec.bandOverride !== undefined ? spec.bandOverride : null;
    this.satOverride  = spec.satOverride  !== undefined ? spec.satOverride  : null;

    Object.freeze(this);
  }
}

/* ---- WATER ------------------------------------------------------ */
export const MAT_WATER_TURQUOISE = new MaterialColorPreset({
  id: 'water_turquoise', name: 'Water Turquoise', kind: MATERIAL_KIND.WATER,
  styleId:      STYLE_ID.CANYON_TURQUOISE,
  gradientId:   GRADIENT_PRESET_ID.WATER_DEPTH_TURQ,
  gradeId:      GRADE_PRESET_ID.CANYON_TURQUOISE,
  smoothBlend: 0.45, accentBlend: 0.55, atmosphere: 1.20, rimStrength: 0.45, glowStrength: 0.15,
  bandOverride: 8,
});

export const MAT_WATER_DEEP = new MaterialColorPreset({
  id: 'water_deep', name: 'Water Deep', kind: MATERIAL_KIND.WATER,
  styleId:      STYLE_ID.RIVER_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.WATER_DEPTH_TURQ,
  gradeId:      GRADE_PRESET_ID.RIVER_TOP_DOWN,
  smoothBlend: 0.55, accentBlend: 0.15, atmosphere: 1.40, rimStrength: 0.30, glowStrength: 0.10,
  bandOverride: 7,
});

export const MAT_FOAM_WHITE = new MaterialColorPreset({
  id: 'foam_white', name: 'Foam White', kind: MATERIAL_KIND.WATER,
  styleId:      STYLE_ID.CANYON_TURQUOISE,
  gradientId:   GRADIENT_PRESET_ID.WATER_DEPTH_TURQ,
  gradeId:      GRADE_PRESET_ID.CANYON_TURQUOISE,
  smoothBlend: 0.25, accentBlend: 0.85, atmosphere: 0.55, rimStrength: 0.55, glowStrength: 0.35,
  bandOverride: 4,
});

/* ---- SNOW / ICE ------------------------------------------------- */
export const MAT_SNOW_BRIGHT = new MaterialColorPreset({
  id: 'snow_bright', name: 'Snow Bright', kind: MATERIAL_KIND.SNOW,
  styleId:      STYLE_ID.SNOW_BLUE_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.SNOW_WHITE_BLUE,
  gradeId:      GRADE_PRESET_ID.SNOW_BLUE,
  smoothBlend: 0.40, accentBlend: 0.20, atmosphere: 0.80, rimStrength: 0.40, glowStrength: 0.20,
  bandOverride: 6,
});

export const MAT_SNOW_SHADOW = new MaterialColorPreset({
  id: 'snow_shadow', name: 'Snow Shadow', kind: MATERIAL_KIND.SNOW,
  styleId:      STYLE_ID.SNOW_BLUE_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.SNOW_WHITE_BLUE,
  gradeId:      GRADE_PRESET_ID.SNOW_BLUE,
  smoothBlend: 0.60, accentBlend: 0.05, atmosphere: 1.00, rimStrength: 0.25, glowStrength: 0.05,
  bandOverride: 5,
});

export const MAT_ICE_TURQUOISE = new MaterialColorPreset({
  id: 'ice_turquoise', name: 'Ice Turquoise', kind: MATERIAL_KIND.ICE,
  styleId:      STYLE_ID.SNOW_BLUE_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.WATER_DEPTH_TURQ,
  gradeId:      GRADE_PRESET_ID.SNOW_BLUE,
  smoothBlend: 0.50, accentBlend: 0.30, atmosphere: 1.10, rimStrength: 0.55, glowStrength: 0.35,
  bandOverride: 6,
});

/* ---- ROCK / SAND / BRICK / STONE -------------------------------- */
export const MAT_ROCK_WARM = new MaterialColorPreset({
  id: 'rock_warm', name: 'Rock Warm', kind: MATERIAL_KIND.ROCK,
  styleId:      STYLE_ID.CANYON_TURQUOISE,
  gradientId:   GRADIENT_PRESET_ID.CANYON_ROCK_WARM,
  gradeId:      GRADE_PRESET_ID.CANYON_TURQUOISE,
  smoothBlend: 0.55, accentBlend: 0.10, atmosphere: 1.00, rimStrength: 0.40, glowStrength: 0.10,
  bandOverride: 6,
});

export const MAT_ROCK_COOL = new MaterialColorPreset({
  id: 'rock_cool', name: 'Rock Cool', kind: MATERIAL_KIND.ROCK,
  styleId:      STYLE_ID.SNOW_BLUE_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.SNOW_WHITE_BLUE,
  gradeId:      GRADE_PRESET_ID.SNOW_BLUE,
  smoothBlend: 0.55, accentBlend: 0.05, atmosphere: 1.00, rimStrength: 0.35, glowStrength: 0.05,
  bandOverride: 6,
});

export const MAT_SAND_DESERT = new MaterialColorPreset({
  id: 'sand_desert', name: 'Sand Desert', kind: MATERIAL_KIND.SAND,
  styleId:      STYLE_ID.DESERT_RUINS,
  gradientId:   GRADIENT_PRESET_ID.DESERT_SAND_WARM,
  gradeId:      GRADE_PRESET_ID.DESERT_RUINS,
  smoothBlend: 0.35, accentBlend: 0.10, atmosphere: 1.30, rimStrength: 0.30, glowStrength: 0.15,
  bandOverride: 7,
});

export const MAT_BRICK_WEATHERED = new MaterialColorPreset({
  id: 'brick_weathered', name: 'Brick Weathered', kind: MATERIAL_KIND.BRICK,
  styleId:      STYLE_ID.DESERT_RUINS,
  gradientId:   GRADIENT_PRESET_ID.DESERT_SAND_WARM,
  gradeId:      GRADE_PRESET_ID.DESERT_RUINS,
  smoothBlend: 0.45, accentBlend: 0.15, atmosphere: 1.10, rimStrength: 0.35, glowStrength: 0.10,
  bandOverride: 6,
});

export const MAT_MOSS_GREEN = new MaterialColorPreset({
  id: 'moss_green', name: 'Moss Green', kind: MATERIAL_KIND.MOSS,
  styleId:      STYLE_ID.FLOWER_FIELD,
  gradientId:   GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
  gradeId:      GRADE_PRESET_ID.FLOWER_FIELD,
  smoothBlend: 0.45, accentBlend: 0.20, atmosphere: 1.00, rimStrength: 0.35, glowStrength: 0.05,
  bandOverride: 6,
});

export const MAT_STONE_CRACKED = new MaterialColorPreset({
  id: 'stone_cracked', name: 'Stone Cracked', kind: MATERIAL_KIND.STONE,
  styleId:      STYLE_ID.SUNSET_TOMBSTONE,
  gradientId:   GRADIENT_PRESET_ID.SKY_SUNSET_PURPLE,
  gradeId:      GRADE_PRESET_ID.SUNSET_TOMBSTONE,
  smoothBlend: 0.40, accentBlend: 0.35, atmosphere: 0.95, rimStrength: 0.50, glowStrength: 0.55,
  bandOverride: 6,
});

export const MAT_CRYSTAL_MINT_GLOW = new MaterialColorPreset({
  id: 'crystal_mint_glow', name: 'Crystal Mint Glow', kind: MATERIAL_KIND.CRYSTAL,
  styleId:      STYLE_ID.SUNSET_TOMBSTONE,
  gradientId:   GRADIENT_PRESET_ID.MAGIC_GOLD_GLOW,
  gradeId:      GRADE_PRESET_ID.SUNSET_TOMBSTONE,
  smoothBlend: 0.50, accentBlend: 0.85, atmosphere: 0.65, rimStrength: 0.85, glowStrength: 1.00,
  bandOverride: 5,
});

/* ---- FOLIAGE / FLOWER ------------------------------------------- */
export const MAT_FOLIAGE_BRIGHT = new MaterialColorPreset({
  id: 'foliage_bright', name: 'Foliage Bright', kind: MATERIAL_KIND.FOLIAGE,
  styleId:      STYLE_ID.RIVER_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
  gradeId:      GRADE_PRESET_ID.RIVER_TOP_DOWN,
  smoothBlend: 0.35, accentBlend: 0.15, atmosphere: 0.85, rimStrength: 0.55, glowStrength: 0.10,
  bandOverride: 6,
});

export const MAT_FOLIAGE_DARK = new MaterialColorPreset({
  id: 'foliage_dark', name: 'Foliage Dark', kind: MATERIAL_KIND.FOLIAGE,
  styleId:      STYLE_ID.RIVER_TOP_DOWN,
  gradientId:   GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
  gradeId:      GRADE_PRESET_ID.RIVER_TOP_DOWN,
  smoothBlend: 0.55, accentBlend: 0.05, atmosphere: 1.10, rimStrength: 0.30, glowStrength: 0.03,
  bandOverride: 5,
});

export const MAT_FLOWER_MAGENTA = new MaterialColorPreset({
  id: 'flower_magenta', name: 'Flower Magenta', kind: MATERIAL_KIND.FLOWER,
  styleId:      STYLE_ID.FLOWER_FIELD,
  gradientId:   GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
  gradeId:      GRADE_PRESET_ID.FLOWER_FIELD,
  smoothBlend: 0.30, accentBlend: 0.65, atmosphere: 0.85, rimStrength: 0.60, glowStrength: 0.25,
  bandOverride: 6,
});

export const MAT_FLOWER_VIOLET = new MaterialColorPreset({
  id: 'flower_violet', name: 'Flower Violet', kind: MATERIAL_KIND.FLOWER,
  styleId:      STYLE_ID.FLOWER_FIELD,
  gradientId:   GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  gradeId:      GRADE_PRESET_ID.FLOWER_FIELD,
  smoothBlend: 0.35, accentBlend: 0.55, atmosphere: 0.85, rimStrength: 0.60, glowStrength: 0.25,
  bandOverride: 6,
});

export const MAT_FLOWER_WHITE = new MaterialColorPreset({
  id: 'flower_white', name: 'Flower White', kind: MATERIAL_KIND.FLOWER,
  styleId:      STYLE_ID.FLOWER_FIELD,
  gradientId:   GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  gradeId:      GRADE_PRESET_ID.FLOWER_FIELD,
  smoothBlend: 0.25, accentBlend: 0.75, atmosphere: 0.75, rimStrength: 0.55, glowStrength: 0.30,
  bandOverride: 6,
});

/* ---- SKY / CLOUD ------------------------------------------------ */
export const MAT_SKY_DAY = new MaterialColorPreset({
  id: 'sky_day', name: 'Sky Day', kind: MATERIAL_KIND.SKY,
  styleId:      STYLE_ID.CANYON_TURQUOISE,
  gradientId:   GRADIENT_PRESET_ID.SKY_DAY_BLUE,
  gradeId:      GRADE_PRESET_ID.CANYON_TURQUOISE,
  smoothBlend: 0.85, accentBlend: 0.10, atmosphere: 0.40, rimStrength: 0.15, glowStrength: 0.20,
  bandOverride: 8,
});

export const MAT_SKY_SUNSET = new MaterialColorPreset({
  id: 'sky_sunset', name: 'Sky Sunset', kind: MATERIAL_KIND.SKY,
  styleId:      STYLE_ID.SUNSET_TOMBSTONE,
  gradientId:   GRADIENT_PRESET_ID.SKY_SUNSET_PURPLE,
  gradeId:      GRADE_PRESET_ID.SUNSET_TOMBSTONE,
  smoothBlend: 0.85, accentBlend: 0.15, atmosphere: 0.35, rimStrength: 0.20, glowStrength: 0.35,
  bandOverride: 8,
});

export const MAT_SKY_SPACE = new MaterialColorPreset({
  id: 'sky_space', name: 'Sky Space', kind: MATERIAL_KIND.SKY,
  styleId:      STYLE_ID.ORBITAL_SPACE,
  gradientId:   GRADIENT_PRESET_ID.SKY_SPACE_NAVY,
  gradeId:      GRADE_PRESET_ID.ORBITAL_SPACE,
  smoothBlend: 0.90, accentBlend: 0.05, atmosphere: 0.25, rimStrength: 0.10, glowStrength: 0.15,
  bandOverride: 8,
});

export const MAT_CLOUD_WHITE = new MaterialColorPreset({
  id: 'cloud_white', name: 'Cloud White', kind: MATERIAL_KIND.CLOUD,
  styleId:      STYLE_ID.CANYON_TURQUOISE,
  gradientId:   GRADIENT_PRESET_ID.SKY_DAY_BLUE,
  gradeId:      GRADE_PRESET_ID.CANYON_TURQUOISE,
  smoothBlend: 0.35, accentBlend: 0.40, atmosphere: 0.55, rimStrength: 0.55, glowStrength: 0.25,
  bandOverride: 5,
});

/* ---- CHARACTER -------------------------------------------------- */
export const MAT_CHARACTER_SKIN = new MaterialColorPreset({
  id: 'character_skin', name: 'Character Skin', kind: MATERIAL_KIND.CHARACTER,
  styleId:      STYLE_ID.PASTEL_PORTRAIT,
  gradientId:   GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  gradeId:      GRADE_PRESET_ID.PASTEL_PORTRAIT,
  smoothBlend: 0.55, accentBlend: 0.15, atmosphere: 0.50, rimStrength: 0.45, glowStrength: 0.15,
  bandOverride: 7,
});

export const MAT_CHARACTER_HAIR_PINK = new MaterialColorPreset({
  id: 'character_hair_pink', name: 'Character Hair Pink', kind: MATERIAL_KIND.CHARACTER,
  styleId:      STYLE_ID.PASTEL_PORTRAIT,
  gradientId:   GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  gradeId:      GRADE_PRESET_ID.PASTEL_PORTRAIT,
  smoothBlend: 0.45, accentBlend: 0.35, atmosphere: 0.50, rimStrength: 0.75, glowStrength: 0.35,
  bandOverride: 8,
});

export const MAT_CHARACTER_CLOTH_LAV = new MaterialColorPreset({
  id: 'character_cloth_lav', name: 'Character Cloth Lavender', kind: MATERIAL_KIND.CHARACTER,
  styleId:      STYLE_ID.PASTEL_PORTRAIT,
  gradientId:   GRADIENT_PRESET_ID.PASTEL_PINK_LAV,
  gradeId:      GRADE_PRESET_ID.PASTEL_PORTRAIT,
  smoothBlend: 0.50, accentBlend: 0.20, atmosphere: 0.55, rimStrength: 0.50, glowStrength: 0.15,
  bandOverride: 6,
});

/* ---- MAGIC ------------------------------------------------------ */
export const MAT_MAGIC_GLOW_GOLD = new MaterialColorPreset({
  id: 'magic_glow_gold', name: 'Magic Glow Gold', kind: MATERIAL_KIND.MAGIC,
  styleId:      STYLE_ID.MAGIC_CASTER,
  gradientId:   GRADIENT_PRESET_ID.MAGIC_GOLD_GLOW,
  gradeId:      GRADE_PRESET_ID.MAGIC_CASTER,
  smoothBlend: 0.55, accentBlend: 0.90, atmosphere: 0.45, rimStrength: 0.85, glowStrength: 1.00,
  bandOverride: 5,
});

export const MAT_MAGIC_CLOAK_GREEN = new MaterialColorPreset({
  id: 'magic_cloak_green', name: 'Magic Cloak Green', kind: MATERIAL_KIND.MAGIC,
  styleId:      STYLE_ID.MAGIC_CASTER,
  gradientId:   GRADIENT_PRESET_ID.FOLIAGE_GREEN_MIX,
  gradeId:      GRADE_PRESET_ID.MAGIC_CASTER,
  smoothBlend: 0.40, accentBlend: 0.15, atmosphere: 0.55, rimStrength: 0.55, glowStrength: 0.15,
  bandOverride: 6,
});

/* ------------------------------------------------------------------ */
/* 2. MATERIAL PRESET REGISTRY                                        */
/* ------------------------------------------------------------------ */

export const MATERIAL_PRESETS = Object.freeze({
  water_turquoise:      MAT_WATER_TURQUOISE,
  water_deep:           MAT_WATER_DEEP,
  foam_white:           MAT_FOAM_WHITE,
  snow_bright:          MAT_SNOW_BRIGHT,
  snow_shadow:          MAT_SNOW_SHADOW,
  ice_turquoise:        MAT_ICE_TURQUOISE,
  rock_warm:            MAT_ROCK_WARM,
  rock_cool:            MAT_ROCK_COOL,
  sand_desert:          MAT_SAND_DESERT,
  brick_weathered:      MAT_BRICK_WEATHERED,
  moss_green:           MAT_MOSS_GREEN,
  stone_cracked:        MAT_STONE_CRACKED,
  crystal_mint_glow:    MAT_CRYSTAL_MINT_GLOW,
  foliage_bright:       MAT_FOLIAGE_BRIGHT,
  foliage_dark:         MAT_FOLIAGE_DARK,
  flower_magenta:       MAT_FLOWER_MAGENTA,
  flower_violet:        MAT_FLOWER_VIOLET,
  flower_white:         MAT_FLOWER_WHITE,
  sky_day:              MAT_SKY_DAY,
  sky_sunset:           MAT_SKY_SUNSET,
  sky_space:            MAT_SKY_SPACE,
  cloud_white:          MAT_CLOUD_WHITE,
  character_skin:       MAT_CHARACTER_SKIN,
  character_hair_pink:  MAT_CHARACTER_HAIR_PINK,
  character_cloth_lav:  MAT_CHARACTER_CLOTH_LAV,
  magic_glow_gold:      MAT_MAGIC_GLOW_GOLD,
  magic_cloak_green:    MAT_MAGIC_CLOAK_GREEN,
});

export const MATERIAL_PRESET_ID = Object.freeze({
  WATER_TURQUOISE:     'water_turquoise',
  WATER_DEEP:          'water_deep',
  FOAM_WHITE:          'foam_white',
  SNOW_BRIGHT:         'snow_bright',
  SNOW_SHADOW:         'snow_shadow',
  ICE_TURQUOISE:       'ice_turquoise',
  ROCK_WARM:           'rock_warm',
  ROCK_COOL:           'rock_cool',
  SAND_DESERT:         'sand_desert',
  BRICK_WEATHERED:     'brick_weathered',
  MOSS_GREEN:          'moss_green',
  STONE_CRACKED:       'stone_cracked',
  CRYSTAL_MINT_GLOW:   'crystal_mint_glow',
  FOLIAGE_BRIGHT:      'foliage_bright',
  FOLIAGE_DARK:        'foliage_dark',
  FLOWER_MAGENTA:      'flower_magenta',
  FLOWER_VIOLET:       'flower_violet',
  FLOWER_WHITE:        'flower_white',
  SKY_DAY:             'sky_day',
  SKY_SUNSET:          'sky_sunset',
  SKY_SPACE:           'sky_space',
  CLOUD_WHITE:         'cloud_white',
  CHARACTER_SKIN:      'character_skin',
  CHARACTER_HAIR_PINK: 'character_hair_pink',
  CHARACTER_CLOTH_LAV: 'character_cloth_lav',
  MAGIC_GLOW_GOLD:     'magic_glow_gold',
  MAGIC_CLOAK_GREEN:   'magic_cloak_green',
});

/* ------------------------------------------------------------------ */
/* 3. GLSL COMPOSITE CHUNK                                            */
/* ------------------------------------------------------------------ */

/**
 * The single GLSL entry point every anime material calls.
 * Requires:
 *   - GLSL_PALETTE_UNIFORMS + GLSL_PALETTE_SAMPLE + GLSL_CHROMATIC +
 *     GLSL_ATMOSPHERE_BLEND (from 033)
 *   - GLSL_EASING + GLSL_GRADIENT_UNIFORMS + GLSL_GRADIENT_SAMPLE (from 034)
 *   - GLSL_GRADE_UNIFORMS + GLSL_GRADE_CHAIN (from 035)
 *   - the material-level uniforms declared in GLSL_MATERIAL_PIPELINE_UNIFORMS
 */
export const GLSL_MATERIAL_PIPELINE_UNIFORMS = /* glsl */`
uniform float uMatSmoothBlend;    // [0,1] palette (0) ← → gradient (1)
uniform float uMatAccentBlend;    // [0,1] accent weight
uniform float uMatAtmosphere;     // [0,1] atmosphere density multiplier
uniform float uMatRimStrength;    // [0,1]
uniform float uMatGlowStrength;   // [0,1]
uniform float uMatBandOverride;   // palette band count override
uniform float uMatSatOverride;    // palette saturation override
uniform float uMatUseLUT;         // 1 if LUT path active
`;

/**
 * Full composite function:
 *
 *   vec3 applyMaterialColor(
 *     vec2  uv,          // fragment uv in [0,1]
 *     float grayPattern, // the grayscale field [0,1]
 *     float accentMask,  // accent weight [0,1]
 *     float depth01,     // linear depth [0,1]
 *     float rimFactor,   // rim [0,1]
 *     float glow01       // emissive glow [0,1]
 *   );
 *
 * It runs:
 *   1. Palette sample at grayPattern (per-pixel)
 *   2. Gradient sample at grayPattern (per-pixel)
 *   3. Mix palette & gradient by uMatSmoothBlend
 *   4. Accent blend toward uPaletteAccent by uMatAccentBlend * accentMask
 *   5. Atmosphere fade
 *   6. Rim light
 *   7. Emissive glow
 *   8. Full grading chain
 */
export const GLSL_MATERIAL_PIPELINE = /* glsl */`
vec3 applyMaterialColor(
  vec2  uv,
  float grayPattern,
  float accentMask,
  float depth01,
  float rimFactor,
  float glow01
) {
  float bands = uMatBandOverride > 0.5 ? uMatBandOverride : uPaletteBandCount;

  // 1. Palette sample.
  vec3 palColor = samplePalette(grayPattern, bands, uPaletteMode, gl_FragCoord.xy);
  palColor = applyChroma(palColor, uMatSatOverride > 0.0 ? uMatSatOverride : uPaletteSatBias);
  if (abs(uPaletteHueBias) > 1e-4) palColor = hueRotate(palColor, uPaletteHueBias);

  // 2. Gradient sample.
  vec3 gradColor = sampleGradient2D(vec2(grayPattern, uv.y));

  // 3. Mix palette & gradient.
  vec3 base = mix(palColor, gradColor, clamp(uMatSmoothBlend, 0.0, 1.0));

  // 4. Accent blend (soft light toward accent).
  float accentW = clamp(uMatAccentBlend * accentMask, 0.0, 1.0);
  base = applyBlend(base, uPaletteAccent, 4.0);
  base = mix(base, base * (1.0 + accentW), accentW);

  // 5. Atmosphere fade.
  base = applyAtmosphere(base, depth01, uMatAtmosphere * 1.2);

  // 6. Ambient bounce.
  base = applyAmbientBounce(base, 0.5, 0.25);

  // 7. Rim light.
  base = applyBacklight(base, rimFactor, uMatRimStrength);

  // 8. Emissive glow.
  base = applyGlow(base, glow01, uMatGlowStrength);

  // 9. Full grading chain.
  base = applyGradeChain(base, uv, 0.0, gl_FragCoord.xy);

  return clamp(base, 0.0, 1.0);
}
`;

/**
 * A convenience "assemble" function that produces the complete fragment
 * prelude a material needs. Callers concatenate this before their own
 * `void main()`.
 *
 *   const prelude = assembleMaterialPipelineGLSL();
 *   material.fragmentShader = prelude + myMainFn;
 */
export function assembleMaterialPipelineGLSL() {
  return (
    GLSL_PALETTE_UNIFORMS + '\n' +
    GLSL_EASING + '\n' +
    GLSL_GRADIENT_UNIFORMS + '\n' +
    GLSL_GRADE_UNIFORMS + '\n' +
    GLSL_MATERIAL_PIPELINE_UNIFORMS + '\n' +
    GLSL_PALETTE_SAMPLE + '\n' +
    GLSL_CHROMATIC + '\n' +
    GLSL_ATMOSPHERE_BLEND + '\n' +
    GLSL_GRADIENT_SAMPLE + '\n' +
    GLSL_GRADE_CHAIN + '\n' +
    GLSL_MATERIAL_PIPELINE + '\n'
  );
}

/* ------------------------------------------------------------------ */
/* 4. MATERIAL COLOR PIPELINE (runtime)                               */
/* ------------------------------------------------------------------ */

/**
 * The runtime object a material acquires. Owns:
 *   - a palette composer (033)
 *   - a gradient uniform set (034)
 *   - a grading chain (035)
 *   - merged uniform block
 *   - optional LUT textures (LOW tier)
 */
export class MaterialColorPipeline {
  constructor(presetOrId, options = {}) {
    this.preset = _resolvePreset(presetOrId);
    if (!this.preset) {
      throw new Error('[036_rnd_MaterialColorPipeline] unknown preset: ' + String(presetOrId));
    }

    this.options = Object.assign({
      forceLUT:        false,
      lutSize:         256,
      transitionFrames:30,
    }, options || {});

    this.useLUT = this.options.forceLUT || USE_LUT_ON_TIER[PERF_TIER_LOCAL] === true;

    // Composer (palette)
    this.composer = new ProceduralColorComposer(this.preset.styleId, {
      bandCount: this.preset.bandOverride !== null ? this.preset.bandOverride : undefined,
      satBias:   this.preset.satOverride  !== null ? this.preset.satOverride  : undefined,
    });

    // Gradient uniforms (from preset's companion gradient)
    const gradient = GRADIENT_PRESETS[this.preset.gradientId] || GRADIENT_PRESETS.atmospheric_haze;
    this.gradientDescriptor = gradient;
    this.gradientUniforms   = createGradientUniforms(gradient);

    // Grading chain (stateful, with transitions)
    this.gradingChain = new ColorGradingChain(this.preset.gradeId, {
      transitionFrames: this.options.transitionFrames,
    });

    // Material-level uniforms.
    this.materialUniforms = {
      uMatSmoothBlend:  { value: this.preset.smoothBlend },
      uMatAccentBlend:  { value: this.preset.accentBlend },
      uMatAtmosphere:   { value: this.preset.atmosphere },
      uMatRimStrength:  { value: this.preset.rimStrength },
      uMatGlowStrength: { value: this.preset.glowStrength },
      uMatBandOverride: { value: this.preset.bandOverride !== null ? this.preset.bandOverride : 0.0 },
      uMatSatOverride:  { value: this.preset.satOverride  !== null ? this.preset.satOverride  : 0.0 },
      uMatUseLUT:       { value: this.useLUT ? 1.0 : 0.0 },
    };

    // LUTs (only on LOW tier or when forced).
    this.paletteLUT  = null;
    this.gradientLUT = null;
    this.gradeLUT    = null;
    if (this.useLUT) {
      this.bakeLUTs(this.options.lutSize);
    }

    // Merged uniforms block (references, not copies).
    this._mergedUniforms = Object.assign(
      {},
      this.composer.getUniforms(),
      this.gradientUniforms,
      this.gradingChain.getUniforms(),
      this.materialUniforms
    );

    // JS sample buffers (zero-alloc).
    this._gradRGB  = [0, 0, 0];
    this._gradeRGB = [0, 0, 0];
  }

  /* ---------------- JS sampling ---------------- */

  /**
   * Full JS pipeline sample. Writes into `out` (length ≥ 3).
   * Inputs:
   *   grayPattern  — [0,1]
   *   accentMask   — [0,1]
   *   depth01      — [0,1]
   *   rimFactor    — [0,1]
   *   glow01       — [0,1]
   *   u, v         — fragment uv (only used by gradients)
   */
  composeRGB(out, grayPattern, accentMask, depth01, rimFactor, glow01, u, v) {
    const p = this.preset;

    // 1. Palette sample (via JS palette mixer).
    const palRGB = _samplePaletteJS(this.composer.getPalette(), grayPattern);

    // 2. Gradient sample.
    const gradRGB = this._gradRGB;
    if (this.gradientDescriptor) {
      sampleGradient2DInto(gradRGB, this.gradientDescriptor, u !== undefined ? u : grayPattern, v !== undefined ? v : 0.5);
    } else {
      gradRGB[0] = palRGB[0]; gradRGB[1] = palRGB[1]; gradRGB[2] = palRGB[2];
    }

    // 3. Mix.
    const sb = p.smoothBlend;
    let r = palRGB[0] * (1 - sb) + gradRGB[0] * sb;
    let g = palRGB[1] * (1 - sb) + gradRGB[1] * sb;
    let b = palRGB[2] * (1 - sb) + gradRGB[2] * sb;

    // 4. Accent blend.
    const accentW = _clamp01(p.accentBlend * accentMask);
    if (accentW > 0) {
      const acc = this.composer.getPalette().accent;
      // Simple lerp toward accent (soft-light approximated by lerp).
      r = _clamp01(r + (acc[0] - r) * accentW * 0.55);
      g = _clamp01(g + (acc[1] - g) * accentW * 0.55);
      b = _clamp01(b + (acc[2] - b) * accentW * 0.55);
    }

    // 5. Atmosphere fade.
    const pal = this.composer.getPalette();
    const fog = pal.fog;
    const atm = _clamp01(depth01) * p.atmosphere;
    const k = 1 - Math.exp(-atm * atm * 1.2);
    r = r * (1 - k) + fog[0] * k;
    g = g * (1 - k) + fog[1] * k;
    b = b * (1 - k) + fog[2] * k;

    // 6. Rim light.
    if (rimFactor > 0 && p.rimStrength > 0) {
      const glow = pal.glowColor;
      const rf = Math.pow(_clamp01(rimFactor), 2.5) * p.rimStrength;
      r = _clamp01(r + glow[0] * rf);
      g = _clamp01(g + glow[1] * rf);
      b = _clamp01(b + glow[2] * rf);
    }

    // 7. Emissive glow.
    if (glow01 > 0 && p.glowStrength > 0) {
      const glow = pal.glowColor;
      const gf = _clamp01(glow01) * p.glowStrength;
      r = _clamp01(r + glow[0] * gf);
      g = _clamp01(g + glow[1] * gf);
      b = _clamp01(b + glow[2] * gf);
    }

    // 8. Grading chain.
    const gr = this._gradeRGB;
    applyGradeInto(gr, this.gradingChain.current, r, g, b);

    out[0] = gr[0];
    out[1] = gr[1];
    out[2] = gr[2];
    return out;
  }

  /* ---------------- GLSL ---------------- */

  /**
   * Returns the complete GLSL prelude for this material. Callers
   * concatenate this before their own `void main()`.
   */
  composeGLSL() {
    return assembleMaterialPipelineGLSL();
  }

  /* ---------------- uniforms ---------------- */

  uniforms() {
    return this._mergedUniforms;
  }

  /** Merges every uniform block into an existing material. */
  attachToMaterial(material) {
    if (!material || !material.uniforms) return false;
    const keys = Object.keys(this._mergedUniforms);
    for (let i = 0; i < keys.length; i++) {
      material.uniforms[keys[i]] = this._mergedUniforms[keys[i]];
    }

    // If we have LUTs, attach them as texture uniforms.
    if (this.useLUT) {
      if (this.paletteLUT)  material.uniforms.uPaletteLUT  = { value: this.paletteLUT };
      if (this.gradientLUT) material.uniforms.uGradientLUT = { value: this.gradientLUT };
      if (this.gradeLUT)    material.uniforms.uGradeLUT    = { value: this.gradeLUT };
    }

    material.needsUpdate = true;
    return true;
  }

  /* ---------------- LUT baking ---------------- */

  bakeLUTs(size) {
    const s = size || this.options.lutSize || 256;

    this.paletteLUT  = bakePaletteLUT(this.composer.getPalette(), s);
    this.gradientLUT = bakeGradientLUT(this.gradientDescriptor, s);
    this.gradeLUT    = bakeGradeLUT(this.gradingChain.current, s);

    return this;
  }

  disposeLUTs() {
    if (this.paletteLUT)  { this.paletteLUT.dispose();  this.paletteLUT = null; }
    if (this.gradientLUT) { this.gradientLUT.dispose(); this.gradientLUT = null; }
    if (this.gradeLUT)    { this.gradeLUT.dispose();    this.gradeLUT = null; }
    return this;
  }

  /* ---------------- preset swap ---------------- */

  setPreset(presetOrId, instant) {
    const p = _resolvePreset(presetOrId);
    if (!p) return false;
    this.preset = p;

    this.composer.setStyle(p.styleId);

    const gradient = GRADIENT_PRESETS[p.gradientId];
    if (gradient) {
      this.gradientDescriptor = gradient;
      applyGradientToUniforms(this.gradientUniforms, gradient);
    }

    this.gradingChain.setGrade(p.gradeId, instant === true);

    this.materialUniforms.uMatSmoothBlend.value  = p.smoothBlend;
    this.materialUniforms.uMatAccentBlend.value  = p.accentBlend;
    this.materialUniforms.uMatAtmosphere.value   = p.atmosphere;
    this.materialUniforms.uMatRimStrength.value  = p.rimStrength;
    this.materialUniforms.uMatGlowStrength.value = p.glowStrength;
    this.materialUniforms.uMatBandOverride.value = p.bandOverride !== null ? p.bandOverride : 0.0;
    this.materialUniforms.uMatSatOverride.value  = p.satOverride  !== null ? p.satOverride  : 0.0;

    if (this.useLUT) this.bakeLUTs(this.options.lutSize);
    return true;
  }

  /* ---------------- per-frame tick ---------------- */

  /** Advances any stateful internal animators (grading transition). */
  update(dt) {
    this.gradingChain.update(dt);
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      presetId:        this.preset.id,
      kind:            MATERIAL_KIND_NAME[this.preset.kind],
      styleId:         this.preset.styleId,
      gradientId:      this.preset.gradientId,
      gradeId:         this.preset.gradeId,
      useLUT:          this.useLUT,
      hasPaletteLUT:   !!this.paletteLUT,
      hasGradientLUT:  !!this.gradientLUT,
      hasGradeLUT:     !!this.gradeLUT,
      perfTier:        PERF_TIER_LOCAL,
    };
  }

  dispose() {
    this.disposeLUTs();
    this.composer.dispose();
    this.gradingChain.dispose();
    this._mergedUniforms = null;
  }
}

/* ------------------------------------------------------------------ */
/* 5. PRESET RESOLUTION                                               */
/* ------------------------------------------------------------------ */

function _resolvePreset(presetOrId) {
  if (!presetOrId) return null;
  if (presetOrId instanceof MaterialColorPreset) return presetOrId;
  if (typeof presetOrId === 'string') {
    return MATERIAL_PRESETS[presetOrId] || null;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* 6. JS PALETTE SAMPLER (mirrors the GLSL one in 033)                */
/* ------------------------------------------------------------------ */

function _samplePaletteJS(palette, t) {
  const q = _clamp01(t);
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

function _clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

/* ------------------------------------------------------------------ */
/* 7. MODULE-LEVEL PIPELINE CACHE                                     */
/* ------------------------------------------------------------------ */

const _pipelineCache = new Map();

/**
 * Returns a shared pipeline for the given preset id. Multiple materials
 * that share the same look (e.g. all water tiles) share a pipeline, so
 * uniforms are only uploaded once.
 */
export function getMaterialPipeline(presetId, options) {
  const key = presetId + (options && options.forceLUT ? ':lut' : '');
  if (_pipelineCache.has(key)) return _pipelineCache.get(key);
  const p = new MaterialColorPipeline(presetId, options);
  _pipelineCache.set(key, p);
  return p;
}

export function disposeMaterialPipeline(presetId) {
  for (const key of Array.from(_pipelineCache.keys())) {
    if (key === presetId || key.startsWith(presetId + ':')) {
      const p = _pipelineCache.get(key);
      if (p) p.dispose();
      _pipelineCache.delete(key);
    }
  }
  return true;
}

export function disposeAllMaterialPipelines() {
  for (const p of _pipelineCache.values()) p.dispose();
  _pipelineCache.clear();
}

export function listMaterialPipelines() {
  return Array.from(_pipelineCache.keys());
}

/* ------------------------------------------------------------------ */
/* 8. CONVENIENCE BUILDERS                                            */
/* ------------------------------------------------------------------ */

/**
 * Given a preset id and an existing material, wires the full color
 * pipeline into it: uniforms + GLSL prelude + LUT textures (if enabled).
 *
 *   const mat = new THREE.ShaderMaterial({ ... });
 *   const pipeline = wireMaterialPipeline(mat, 'water_turquoise');
 *   mat.fragmentShader = pipeline.composeGLSL() + myMain;
 */
export function wireMaterialPipeline(material, presetId, options) {
  if (!material || !material.isShaderMaterial) return null;
  const pipeline = getMaterialPipeline(presetId, options);
  pipeline.attachToMaterial(material);
  return pipeline;
}

/**
 * Produces a new THREE.ShaderMaterial with the full color pipeline
 * pre-wired. Caller supplies vertex shader and the fragment main body.
 */
export function createAnimeMaterial(spec) {
  if (!spec || typeof spec.fragmentMain !== 'function') return null;
  const presetId = spec.presetId || MATERIAL_PRESET_ID.ROCK_WARM;
  const pipeline = new MaterialColorPipeline(presetId, spec.options);

  const prelude = pipeline.composeGLSL();
  const mainSrc = spec.fragmentMain(prelude);

  const uniforms = Object.assign(
    {},
    pipeline.uniforms(),
    spec.extraUniforms || {}
  );

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader:   spec.vertexShader   || _defaultVertexShader,
    fragmentShader: mainSrc,
    transparent:    spec.transparent === true,
    depthWrite:     spec.depthWrite !== false,
    depthTest:      spec.depthTest !== false,
    side:           spec.side !== undefined ? spec.side : THREE.FrontSide,
    blending:       spec.blending !== undefined ? spec.blending : THREE.NormalBlending,
    vertexColors:   spec.vertexColors === true,
    lights:         false,
    fog:            false,
  });

  material.userData.__pipeline = pipeline;
  material.userData.__pipelinePresetId = presetId;
  return material;
}

const _defaultVertexShader = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/* ------------------------------------------------------------------ */
/* 9. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createMaterialColorPipeline(presetOrId, options = {}) {
  return new MaterialColorPipeline(presetOrId, options);
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  MaterialColorPipeline,
  MaterialColorPreset,

  // Preset registry
  MATERIAL_PRESETS,
  MATERIAL_PRESET_ID,

  // Individual presets (for direct import)
  MAT_WATER_TURQUOISE,
  MAT_WATER_DEEP,
  MAT_FOAM_WHITE,
  MAT_SNOW_BRIGHT,
  MAT_SNOW_SHADOW,
  MAT_ICE_TURQUOISE,
  MAT_ROCK_WARM,
  MAT_ROCK_COOL,
  MAT_SAND_DESERT,
  MAT_BRICK_WEATHERED,
  MAT_MOSS_GREEN,
  MAT_STONE_CRACKED,
  MAT_CRYSTAL_MINT_GLOW,
  MAT_FOLIAGE_BRIGHT,
  MAT_FOLIAGE_DARK,
  MAT_FLOWER_MAGENTA,
  MAT_FLOWER_VIOLET,
  MAT_FLOWER_WHITE,
  MAT_SKY_DAY,
  MAT_SKY_SUNSET,
  MAT_SKY_SPACE,
  MAT_CLOUD_WHITE,
  MAT_CHARACTER_SKIN,
  MAT_CHARACTER_HAIR_PINK,
  MAT_CHARACTER_CLOTH_LAV,
  MAT_MAGIC_GLOW_GOLD,
  MAT_MAGIC_CLOAK_GREEN,

  // GLSL
  GLSL_MATERIAL_PIPELINE_UNIFORMS,
  GLSL_MATERIAL_PIPELINE,
  assembleMaterialPipelineGLSL,

  // Pipeline cache
  getMaterialPipeline,
  disposeMaterialPipeline,
  disposeAllMaterialPipelines,
  listMaterialPipelines,

  // Material builders
  wireMaterialPipeline,
  createAnimeMaterial,
  createMaterialColorPipeline,

  // Enums
  MATERIAL_KIND,
  MATERIAL_KIND_NAME,
  USE_LUT_ON_TIER,
};

export default _defaultExport;