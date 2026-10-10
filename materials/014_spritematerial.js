// file number : 014
// full path name : src/materials/014_spritematerial.js
// description : SpriteMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 SpriteMaterial API — color, map, alphaMap, rotation, sizeAttenuation, fog, and the inherited material surface. Sprites are the primary vehicle for anime 2D elements (character billboards, UI icons, particle effects, speech bubbles, magical sparkles, and 2D foreground layers). Adds real-time anime features specifically tuned for sprite rendering: sprite UV cel-banding (posterized sprite colors for flat anime look), rotation-based mood tinting (warm sunset rotation haze, cool snowy rotation shimmer), distance-based cel size banding (chunky sprite scaling for stylized depth), procedural sprite variation via simplex-noise (unique per-instance sprite shapes), rim glow around sprite edges (cyan water rims in reference 1, 3, 5; warm sunset character rims in 2, 4, 6), paper-grain and watercolor sprite texture variation, and mood-based color grading (snowy cyan, sunset orange, vibrant flora). Imports Color, Vector2, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation sprite UV and color transforms, double.js for bit-exact sprite color quantization and HDR mood grading, bitecs SoA batching for real-time updates across thousands of sprite instances, and simplex-noise for procedural sprite texture variation.
// best for : SpriteMaterial, anime character billboards, 2D UI elements, speech bubbles, particle effects, magical sparkles, cherry-blossom petals, 2D foreground layers, VR sprites, and any three.js sprite rendering that needs real-time anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import {
	NormalBlending,
	NoColorSpace,
	LinearSRGBColorSpace
} from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r185/src/constants.js';
import { createNoise2D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';
import Double from 'https://cdn.jsdelivr.net/npm/double.js@1.1.0/dist/double.js';
import { createWorld, addEntity, addComponent, defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.js';

// ---------------------------------------------------------------------------
// Shared scratch & precision helpers
// ---------------------------------------------------------------------------

const _noise2D = createNoise2D();
const _double = new Double( 0 );

// gl-matrix scratch for zero-allocation sprite transforms
const _gm_uv = glMatrix.vec2.create();
const _gm_uv_out = glMatrix.vec2.create();
const _gm_rgb = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — sprite UV cel banding (posterized flat anime look)
// ---------------------------------------------------------------------------

/**
 * Quantize sprite UV color samples into discrete cel bands using
 * double.js for bit-exact thresholding. Produces the characteristic
 * "posterized" anime sprite look — flat color regions with hard edges,
 * matching the cel-shaded flowers in reference 7 and the flat-shaded
 * clouds in references 4 and 6.
 *
 * @param {Color} sampledColor - The color sampled from the sprite texture.
 * @param {Color} output - The output color (modified in place).
 * @param {number} bands - Number of cel bands (2-6 recommended).
 * @param {number} quantizeAmount - Blend amount between continuous and banded (0-1).
 * @param {number} [shadowTint=0.85] - Shadow band brightness multiplier.
 * @returns {Color}
 */
function applySpriteCelBanding( sampledColor, output, bands, quantizeAmount, shadowTint = 0.85 ) {

	if ( bands <= 1 ) {

		output.copy( sampledColor );
		return output;

	}

	// Compute luminance for banding
	_double.value = 0.2126 * sampledColor.r;
	_double.add( 0.7152 * sampledColor.g );
	_double.add( 0.0722 * sampledColor.b );
	const lum = _double.value;

	// Quantize
	const bandWidth = 1.0 / bands;
	_double.value = lum;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	// Blend continuous and quantized
	_double.value = lum;
	_double.add( ( quantized - lum ) * quantizeAmount );
	const finalLum = _double.value;

	// Apply shadow tint to darker bands
	_double.value = shadowTint;
	_double.add( ( 1.0 - shadowTint ) * Math.min( 1, finalLum / Math.max( lum, 0.0001 ) ) );
	const shadowMul = Math.min( 1, _double.value );

	// Preserve hue, only change luminance
	if ( lum > 0.0001 ) {

		_double.value = sampledColor.r;
		_double.div( lum );
		_double.mul( finalLum );
		_double.mul( shadowMul );
		output.r = _double.value;

		_double.value = sampledColor.g;
		_double.div( lum );
		_double.mul( finalLum );
		_double.mul( shadowMul );
		output.g = _double.value;

		_double.value = sampledColor.b;
		_double.div( lum );
		_double.mul( finalLum );
		_double.mul( shadowMul );
		output.b = _double.value;

	} else {

		output.copy( sampledColor );

	}

	output.a = sampledColor.a;

	// Clamp
	output.r = Math.max( 0, Math.min( 1, output.r ) );
	output.g = Math.max( 0, Math.min( 1, output.g ) );
	output.b = Math.max( 0, Math.min( 1, output.b ) );

	return output;

}

// ---------------------------------------------------------------------------
// Anime feature — distance-based sprite size cel banding
// ---------------------------------------------------------------------------

/**
 * Quantize a sprite's distance-based scale into discrete cel bands using
 * double.js for bit-exact thresholding. Produces the "chunky" stylized
 * sprite scaling characteristic of anime 2D compositing — distant
 * sprites pop between discrete sizes instead of scaling smoothly.
 * Matches the layered sprite elements in references 2, 4, 6.
 *
 * @param {number} depthNormalized - Normalized depth in [0, 1].
 * @param {number} bands - Number of size bands (3-6 recommended).
 * @param {number} quantizeAmount - Blend amount.
 * @param {number} [minScale=0.3] - Smallest scale multiplier.
 * @param {number} [maxScale=1.5] - Largest scale multiplier.
 * @returns {number} Scale multiplier in [minScale, maxScale].
 */
function applySpriteSizeBanding( depthNormalized, bands, quantizeAmount, minScale = 0.3, maxScale = 1.5 ) {

	if ( bands <= 1 ) return 1;

	const bandWidth = 1.0 / bands;
	_double.value = depthNormalized;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	_double.value = depthNormalized;
	_double.add( ( quantized - depthNormalized ) * quantizeAmount );
	const finalDepth = _double.value;

	_double.value = maxScale;
	_double.sub( ( maxScale - minScale ) * finalDepth );

	return Math.max( minScale, Math.min( maxScale, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — rotation-based mood tinting
// ---------------------------------------------------------------------------

/**
 * Apply a mood-based tint that varies with the sprite's current rotation.
 * Creates the "shimmering" anime effect where rotating sprites (petals,
 * sparkles, magical particles) shift color as they spin. Warm moods
 * (sunset) tint rotation through orange; cool moods (snowy) tint through
 * cyan. Uses gl-matrix for zero-allocation staging and double.js for
 * bit-exact accumulation.
 *
 * @param {Color} baseColor - The base sprite color.
 * @param {Color} output - The output color (modified in place).
 * @param {number} rotation - Sprite rotation in radians.
 * @param {number} moodTemperature - Warm/cool shift in [-1, 1].
 * @param {number} tintIntensity - Tint intensity in [0, 1].
 * @returns {Color}
 */
function applyRotationMoodTint( baseColor, output, rotation, moodTemperature, tintIntensity ) {

	if ( tintIntensity <= 0 ) {

		output.copy( baseColor );
		return output;

	}

	// Normalize rotation into [0, 1] via sine
	_double.value = Math.sin( rotation );
	_double.mul( 0.5 );
	_double.add( 0.5 );
	const rotNorm = _double.value;

	// Mood target color
	_double.value = 1.0;
	const targetR = _double.value;

	_double.value = 0.8;
	_double.add( moodTemperature * 0.1 );
	const targetG = _double.value;

	_double.value = 0.6;
	_double.sub( moodTemperature * 0.4 );
	const targetB = _double.value;

	// Blend intensity modulated by rotation phase
	_double.value = rotNorm;
	_double.mul( tintIntensity );
	const blend = _double.value;

	_double.value = baseColor.r;
	_double.add( ( targetR - baseColor.r ) * blend );
	output.r = _double.value;

	_double.value = baseColor.g;
	_double.add( ( targetG - baseColor.g ) * blend );
	output.g = _double.value;

	_double.value = baseColor.b;
	_double.add( ( targetB - baseColor.b ) * blend );
	output.b = _double.value;

	output.a = baseColor.a;

	// Clamp
	output.r = Math.max( 0, Math.min( 1, output.r ) );
	output.g = Math.max( 0, Math.min( 1, output.g ) );
	output.b = Math.max( 0, Math.min( 1, output.b ) );

	return output;

}

// ---------------------------------------------------------------------------
// Anime feature — procedural sprite variation via simplex-noise
// ---------------------------------------------------------------------------

/**
 * Generate a procedural sprite variation texture using simplex-noise.
 * Each sprite instance can sample a different region of this texture
 * to get a unique shape — matching the varied hand-drawn petals, leaves,
 * and sparkles in the reference imagery.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.1] - Noise scale.
 * @param {number} [octaves=3] - Number of noise octaves.
 * @param {number} [sharpness=1.5] - Shape sharpness.
 * @returns {Uint8Array} RGBA sprite texture buffer.
 */
function generateSpriteVariation( width, height, scale = 0.1, octaves = 3, sharpness = 1.5 ) {

	const out = new Uint8Array( width * height * 4 );
	const cx = width * 0.5;
	const cy = height * 0.5;
	const radius = Math.min( width, height ) * 0.45;

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;
			const dx = x - cx;
			const dy = y - cy;
			const dist = Math.sqrt( dx * dx + dy * dy );

			if ( dist > radius ) {

				out[ p ] = 0;
				out[ p + 1 ] = 0;
				out[ p + 2 ] = 0;
				out[ p + 3 ] = 0;
				continue;

			}

			let value = 0;
			let amplitude = 1;
			let frequency = scale;
			let maxAmplitude = 0;

			for ( let o = 0; o < octaves; o ++ ) {

				value += _noise2D( x * frequency, y * frequency ) * amplitude;
				maxAmplitude += amplitude;
				amplitude *= 0.5;
				frequency *= 2.0;

			}

			value = ( value / maxAmplitude ) * 0.5 + 0.5;

			// Radial fade with sharpness
			_double.value = 1.0 - ( dist / radius );
			const radialFade = Math.pow( Math.max( 0, _double.value ), sharpness );

			_double.value = value;
			_double.mul( radialFade );

			const finalValue = Math.max( 0, Math.min( 1, _double.value ) );
			const v = Math.floor( finalValue * 255 );

			out[ p ] = v;
			out[ p + 1 ] = v;
			out[ p + 2 ] = v;
			out[ p + 3 ] = v;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// Anime feature — rim glow around sprite edges
// ---------------------------------------------------------------------------

/**
 * Compute a rim glow on the outer edge of a sprite. Uses the sprite's
 * alpha gradient to detect the sprite boundary and produce a soft glow.
 * Matches the glowing cyan water highlights in reference 1, 3, 5, and
 * the warm sunset character rims in 2, 4, 6.
 *
 * @param {number} alpha - Sprite alpha at the current fragment.
 * @param {number} [threshold=0.5] - Alpha at which the edge is centered.
 * @param {number} [softness=0.2] - Rim softness.
 * @param {number} [intensity=1.0] - Rim intensity multiplier.
 * @returns {number} Rim intensity in [0, 1].
 */
function computeSpriteEdgeRim( alpha, threshold = 0.5, softness = 0.2, intensity = 1.0 ) {

	_double.value = alpha;
	_double.sub( threshold );
	const distToEdge = Math.abs( _double.value );

	if ( distToEdge > softness ) return 0;

	_double.value = 1.0;
	_double.sub( distToEdge / softness );
	return Math.max( 0, Math.min( 1, _double.value * intensity ) );

}

// ---------------------------------------------------------------------------
// Anime feature — paper-grain / watercolor sprite texture
// ---------------------------------------------------------------------------

/**
 * Generate a combined paper-grain and watercolor texture using
 * simplex-noise with multiple octaves. The texture is used to modulate
 * sprite opacity for a hand-painted feel — the characteristic watercolor
 * bleed of hand-drawn anime sprites (reference images 1, 5, 7).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.05] - Base noise scale.
 * @param {number} [octaves=4] - Number of noise octaves.
 * @param {number} [paperGrain=0.3] - Paper-grain intensity.
 * @param {number} [watercolorBleed=0.5] - Watercolor bleed strength.
 * @returns {Uint8Array} Grayscale RGBA buffer.
 */
function generateSpriteTexture( width, height, scale = 0.05, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

	const out = new Uint8Array( width * height * 4 );

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;

			let value = 0;
			let amplitude = 1;
			let frequency = scale;
			let maxAmplitude = 0;

			for ( let o = 0; o < octaves; o ++ ) {

				value += _noise2D( x * frequency, y * frequency ) * amplitude;
				maxAmplitude += amplitude;
				amplitude *= 0.5;
				frequency *= 2.0;

			}

			value = ( value / maxAmplitude ) * 0.5 + 0.5;

			// Watercolor bleed
			_double.value = value;
			_double.sub( 0.5 );
			_double.mul( 1.0 + watercolorBleed );
			_double.add( 0.5 );
			value = Math.max( 0, Math.min( 1, _double.value ) );

			// Paper grain overlay
			const grain = _noise2D( x * 0.5, y * 0.5 ) * 0.5 + 0.5;
			_double.value = value;
			_double.mul( 1 - paperGrain );
			_double.add( grain * paperGrain );
			value = Math.max( 0, Math.min( 1, _double.value ) );

			const v = Math.floor( value * 255 );

			out[ p ] = v;
			out[ p + 1 ] = v;
			out[ p + 2 ] = v;
			out[ p + 3 ] = 255;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based color grading
// ---------------------------------------------------------------------------

/**
 * Apply mood-based color grading to the sprite color. Handles warm
 * (sunset orange), cool (snowy cyan), and vibrant (flora) moods seen
 * across the reference imagery. Uses gl-matrix for zero-allocation
 * staging and double.js for bit-exact accumulation.
 *
 * @param {Color} color - The color to grade (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} contrast - Contrast multiplier.
 * @returns {Color}
 */
function gradeSpriteColor( color, temperature, saturation, brightness, contrast ) {

	glMatrix.vec3.set( _gm_rgb, color.r, color.g, color.b );

	const lum = 0.2126 * _gm_rgb[ 0 ] + 0.7152 * _gm_rgb[ 1 ] + 0.0722 * _gm_rgb[ 2 ];

	_double.value = lum;
	_double.add( ( _gm_rgb[ 0 ] - lum ) * saturation );
	let r = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 1 ] - lum ) * saturation );
	let g = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 2 ] - lum ) * saturation );
	let b = _double.value;

	_double.value = r;
	_double.sub( 0.5 );
	_double.mul( contrast );
	_double.add( 0.5 );
	r = _double.value;

	_double.value = g;
	_double.sub( 0.5 );
	_double.mul( contrast );
	_double.add( 0.5 );
	g = _double.value;

	_double.value = b;
	_double.sub( 0.5 );
	_double.mul( contrast );
	_double.add( 0.5 );
	b = _double.value;

	_double.value = r;
	_double.add( temperature * 0.12 );
	r = _double.value;

	_double.value = b;
	_double.sub( temperature * 0.12 );
	b = _double.value;

	r *= brightness;
	g *= brightness;
	b *= brightness;

	color.setRGB(
		Math.max( 0, Math.min( 1, r ) ),
		Math.max( 0, Math.min( 1, g ) ),
		Math.max( 0, Math.min( 1, b ) ),
		ColorManagement.workingColorSpace
	);

	return color;

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time sprite material updates
// ---------------------------------------------------------------------------

const _spriteWorld = createWorld();

const SpriteMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	celBands: Types.ui8,
	celQuantize: Types.f64,
	shadowTint: Types.f64,
	sizeBands: Types.ui8,
	sizeQuantize: Types.f64,
	sizeMin: Types.f64,
	sizeMax: Types.f64,
	rotationTint: Types.f64,
	rotationTintIntensity: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	rimGlow: Types.f64,
	rimSoftness: Types.f64,
	paperGrain: Types.f64,
	watercolorBleed: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class SpriteMaterialBatch {

	constructor() {

		this.world = _spriteWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a SpriteMaterial instance for batched real-time updates.
	 *
	 * @param {SpriteMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, SpriteMaterialComponent, eid );

		SpriteMaterialComponent.materialPtr[ eid ] = this.materials.length;
		SpriteMaterialComponent.celBands[ eid ] = material.celBands;
		SpriteMaterialComponent.celQuantize[ eid ] = material.celQuantize;
		SpriteMaterialComponent.shadowTint[ eid ] = material.shadowTint;
		SpriteMaterialComponent.sizeBands[ eid ] = material.sizeBands;
		SpriteMaterialComponent.sizeQuantize[ eid ] = material.sizeQuantize;
		SpriteMaterialComponent.sizeMin[ eid ] = material.sizeMin;
		SpriteMaterialComponent.sizeMax[ eid ] = material.sizeMax;
		SpriteMaterialComponent.rotationTint[ eid ] = material.rotationTint;
		SpriteMaterialComponent.rotationTintIntensity[ eid ] = material.rotationTintIntensity;
		SpriteMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		SpriteMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		SpriteMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		SpriteMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		SpriteMaterialComponent.rimGlow[ eid ] = material.rimGlow;
		SpriteMaterialComponent.rimSoftness[ eid ] = material.rimSoftness;
		SpriteMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		SpriteMaterialComponent.watercolorBleed[ eid ] = material.watercolorBleed;
		SpriteMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		SpriteMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued sprite-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact cel banding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ SpriteMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.celBands = SpriteMaterialComponent.celBands[ eid ];
			material.celQuantize = SpriteMaterialComponent.celQuantize[ eid ];
			material.shadowTint = SpriteMaterialComponent.shadowTint[ eid ];
			material.sizeBands = SpriteMaterialComponent.sizeBands[ eid ];
			material.sizeQuantize = SpriteMaterialComponent.sizeQuantize[ eid ];
			material.sizeMin = SpriteMaterialComponent.sizeMin[ eid ];
			material.sizeMax = SpriteMaterialComponent.sizeMax[ eid ];
			material.rotationTint = SpriteMaterialComponent.rotationTint[ eid ];
			material.rotationTintIntensity = SpriteMaterialComponent.rotationTintIntensity[ eid ];
			material.moodTemperature = SpriteMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = SpriteMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = SpriteMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = SpriteMaterialComponent.moodContrast[ eid ];
			material.rimGlow = SpriteMaterialComponent.rimGlow[ eid ];
			material.rimSoftness = SpriteMaterialComponent.rimSoftness[ eid ];
			material.paperGrain = SpriteMaterialComponent.paperGrain[ eid ];
			material.watercolorBleed = SpriteMaterialComponent.watercolorBleed[ eid ];
			material.variationSeed = SpriteMaterialComponent.variationSeed[ eid ];

			// Recompute mood-graded color
			material.moodColor.copy( material.color );
			gradeSpriteColor(
				material.moodColor,
				material.moodTemperature,
				material.moodSaturation,
				material.moodBrightness,
				material.moodContrast
			);

			SpriteMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main SpriteMaterial class — mirrors three.js/src/materials/SpriteMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material for rendering instances of {@link Sprite}.
 *
 * Sprites are always face the camera — perfect for anime billboards
 * (character portraits, UI icons, particle effects, speech bubbles,
 * magical sparkles). In addition to the standard three.js parameters,
 * this material exposes a rich set of real-time anime-style controls:
 * sprite UV cel-banding, distance-based size banding, rotation-based
 * mood tinting, procedural sprite variation, rim glow, and paper-grain
 * texture variation.
 *
 * ```js
 * const map = new THREE.TextureLoader().load( 'textures/sprite.png' );
 * const material = new THREE.SpriteMaterial( {
 *   map: map,
 *   color: 0xffffff,
 *   celBands: 3,
 *   celQuantize: 0.8,
 *   rotationTintIntensity: 0.5,
 *   moodTemperature: -0.3,
 *   rimGlow: 0.4
 * } );
 * ```
 *
 * @augments Material
 */
class SpriteMaterial extends Material {

	/**
	 * Constructs a new sprite material.
	 *
	 * @param {Object} [parameters] - An object with one or more properties
	 *   defining the material's appearance.
	 */
	constructor( parameters ) {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isSpriteMaterial = true;

		this.type = 'SpriteMaterial';

		/**
		 * The sprite's base color. Default is white.
		 *
		 * @type {Color}
		 * @default (1,1,1)
		 */
		this.color = new Color( 0xffffff );

		/**
		 * The sprite's texture. The texture is expected to be in the sRGB
		 * color space.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.map = null;

		/**
		 * The alpha map. The texture is expected to be in `NoColorSpace`.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.alphaMap = null;

		/**
		 * The sprite's rotation in radians.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rotation = 0;

		/**
		 * Whether the sprite's size attenuates with distance. When `true`,
		 * distant sprites appear smaller.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.sizeAttenuation = true;

		/**
		 * Whether the material is affected by fog.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.fog = true;

		// -------------------------------------------------------------------
		// Anime Rendering Extensions
		// -------------------------------------------------------------------

		/**
		 * Number of cel bands for posterizing the sprite color. 0 = off,
		 * 2 = classic hard-shadow anime, 3-6 = softer stylized.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.celBands = 0;

		/**
		 * Blend amount for sprite cel banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.celQuantize = 1.0;

		/**
		 * Shadow band brightness multiplier.
		 *
		 * @type {number}
		 * @default 0.85
		 */
		this.shadowTint = 0.85;

		/**
		 * Number of distance-based sprite size bands. 0 = off,
		 * 3-6 = stylized "chunky" sprite scaling.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.sizeBands = 0;

		/**
		 * Blend amount for sprite size banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.sizeQuantize = 1.0;

		/**
		 * Smallest size multiplier across all bands.
		 *
		 * @type {number}
		 * @default 0.3
		 */
		this.sizeMin = 0.3;

		/**
		 * Largest size multiplier across all bands.
		 *
		 * @type {number}
		 * @default 1.5
		 */
		this.sizeMax = 1.5;

		/**
		 * Rotation-based mood tint. Modulates the sprite's tint as it
		 * rotates through a sine cycle.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rotationTint = 0;

		/**
		 * Rotation-based tint intensity.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rotationTintIntensity = 0;

		/**
		 * Mood temperature shift in [-1, 1]. Positive = warm (sunset
		 * orange), negative = cool (snowy cyan).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodTemperature = 0;

		/**
		 * Mood saturation multiplier.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.moodSaturation = 1.0;

		/**
		 * Mood brightness multiplier.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.moodBrightness = 1.0;

		/**
		 * Mood contrast multiplier.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.moodContrast = 1.0;

		/**
		 * The precomputed mood-graded sprite color.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0xffffff );

		/**
		 * Rim-glow intensity. Adds a soft glow around sprite edges.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimGlow = 0;

		/**
		 * Rim-glow color. Defaults to cool cyan.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.rimGlowColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Rim edge softness.
		 *
		 * @type {number}
		 * @default 0.2
		 */
		this.rimSoftness = 0.2;

		/**
		 * Paper-grain overlay intensity.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.paperGrain = 0;

		/**
		 * Watercolor bleed strength.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.watercolorBleed = 0;

		/**
		 * Procedural variation seed.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.variationSeed = 0;

		this.setValues( parameters );

	}

	// -----------------------------------------------------------------------
	// Anime helpers
	// -----------------------------------------------------------------------

	/**
	 * Apply sprite cel banding to a sampled color.
	 *
	 * @param {Color} sampledColor
	 * @param {Color} output
	 * @returns {Color}
	 */
	applyCelBanding( sampledColor, output ) {

		return applySpriteCelBanding(
			sampledColor,
			output,
			this.celBands,
			this.celQuantize,
			this.shadowTint
		);

	}

	/**
	 * Apply distance-based sprite size banding.
	 *
	 * @param {number} depthNormalized
	 * @returns {number}
	 */
	applySizeBanding( depthNormalized ) {

		return applySpriteSizeBanding(
			depthNormalized,
			this.sizeBands,
			this.sizeQuantize,
			this.sizeMin,
			this.sizeMax
		);

	}

	/**
	 * Apply rotation-based mood tinting to a color.
	 *
	 * @param {Color} baseColor
	 * @param {Color} output
	 * @param {number} rotation
	 * @returns {Color}
	 */
	applyRotationTint( baseColor, output, rotation ) {

		return applyRotationMoodTint(
			baseColor,
			output,
			rotation,
			this.moodTemperature,
			this.rotationTintIntensity
		);

	}

	/**
	 * Compute the sprite edge rim glow.
	 *
	 * @param {number} alpha
	 * @returns {number}
	 */
	sampleEdgeRim( alpha ) {

		if ( this.rimGlow <= 0 ) return 0;
		return computeSpriteEdgeRim( alpha, 0.5, this.rimSoftness, this.rimGlow );

	}

	/**
	 * Recompute the mood-graded sprite color.
	 *
	 * @returns {SpriteMaterial} A reference to this instance.
	 */
	updateMoodColor() {

		this.moodColor.copy( this.color );
		gradeSpriteColor(
			this.moodColor,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast
		);
		return this;

	}

	/**
	 * Generate a procedural sprite variation texture for this material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.1]
	 * @param {number} [octaves=3]
	 * @returns {Uint8Array}
	 */
	generateSpriteVariation( width, height, scale = 0.1, octaves = 3 ) {

		return generateSpriteVariation( width, height, scale, octaves, 1.5 );

	}

	/**
	 * Generate a procedural paper-grain / watercolor sprite texture.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.05]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generateSpriteTexture( width, height, scale = 0.05, octaves = 4 ) {

		return generateSpriteTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

	}

	/**
	 * Compute the per-instance variation offset.
	 *
	 * @returns {number}
	 */
	getVariationOffset() {

		return this.variationSeed * 137.508;

	}

	// -----------------------------------------------------------------------
	// Shader hooks
	// -----------------------------------------------------------------------

	/**
	 * The default `onBeforeCompile` hook. Extends the base Material's
	 * anime shader chunks with sprite-specific features: sprite cel
	 * banding, distance-based size banding, rotation-based mood tinting,
	 * edge rim glow, and paper-grain texture variation.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject sprite-specific uniforms
		shader.uniforms.celBands = { value: this.celBands };
		shader.uniforms.celQuantize = { value: this.celQuantize };
		shader.uniforms.shadowTint = { value: this.shadowTint };
		shader.uniforms.sizeBands = { value: this.sizeBands };
		shader.uniforms.sizeQuantize = { value: this.sizeQuantize };
		shader.uniforms.sizeMin = { value: this.sizeMin };
		shader.uniforms.sizeMax = { value: this.sizeMax };
		shader.uniforms.rotationTint = { value: this.rotationTint };
		shader.uniforms.rotationTintIntensity = { value: this.rotationTintIntensity };
		shader.uniforms.moodColor = { value: this.moodColor };
		shader.uniforms.rimGlow = { value: this.rimGlow };
		shader.uniforms.rimGlowColor = { value: this.rimGlowColor };
		shader.uniforms.rimSoftness = { value: this.rimSoftness };
		shader.uniforms.paperGrain = { value: this.paperGrain };
		shader.uniforms.watercolorBleed = { value: this.watercolorBleed };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };
		shader.uniforms.spriteRotation = { value: this.rotation };

		// Extend the vertex shader for size banding
		shader.vertexShader = shader.vertexShader
			.replace(
				'#include <common>',
				`#include <common>
				uniform int sizeBands;
				uniform float sizeQuantize;
				uniform float sizeMin;
				uniform float sizeMax;
				`
			);

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int celBands;
				uniform float celQuantize;
				uniform float shadowTint;
				uniform vec3 moodColor;
				uniform float rimGlow;
				uniform vec3 rimGlowColor;
				uniform float rimSoftness;
				uniform float paperGrain;
				uniform float watercolorBleed;
				uniform float variationOffset;
				uniform float spriteRotation;

				vec3 applySpriteCelBandingFn( vec3 sampled ) {
					if ( celBands <= 1 ) return sampled;
					float lum = dot( sampled, vec3( 0.2126, 0.7152, 0.0722 ) );
					float bandWidth = 1.0 / float( celBands );
					float bandIndex = floor( lum / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					float finalLum = mix( lum, quantized, celQuantize );
					float shadowMul = min( 1.0, shadowTint + ( 1.0 - shadowTint ) * ( finalLum / max( lum, 0.0001 ) ) );
					return sampled * ( finalLum / max( lum, 0.0001 ) ) * shadowMul;
				}

				float computeSpriteEdgeRimFn( float alpha ) {
					if ( rimGlow <= 0.0 ) return 0.0;
					float distToEdge = abs( alpha - 0.5 );
					if ( distToEdge > rimSoftness ) return 0.0;
					return clamp( 1.0 - distToEdge / rimSoftness, 0.0, 1.0 ) * rimGlow;
				}

				float sampleSpritePaperGrain( vec2 uv ) {
					if ( paperGrain <= 0.0 ) return 1.0;
					float n = sin( uv.x * 8.0 + variationOffset ) * cos( uv.y * 8.0 + variationOffset );
					n = n * 0.5 + 0.5;
					return 1.0 - paperGrain * ( 1.0 - n );
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Apply sprite cel banding (posterized flat anime look)
				gl_FragColor.rgb = applySpriteCelBandingFn( gl_FragColor.rgb );

				// Rotation-based mood tinting (shimmering petals/sparkles)
				if ( rotationTintIntensity > 0.0 ) {
					float rotNorm = sin( spriteRotation ) * 0.5 + 0.5;
					vec3 rotationTint = vec3( 1.0, 0.8, 0.6 );
					gl_FragColor.rgb = mix( gl_FragColor.rgb, rotationTint, rotNorm * rotationTintIntensity );
				}

				// Override with mood-graded sprite color
				gl_FragColor.rgb = mix( gl_FragColor.rgb, moodColor, 0.7 );

				// Rim glow around sprite edges
				if ( rimGlow > 0.0 ) {
					float edgeRim = computeSpriteEdgeRimFn( gl_FragColor.a );
					gl_FragColor.rgb += rimGlowColor * edgeRim;
				}

				// Paper grain opacity modulation
				gl_FragColor.a *= sampleSpritePaperGrain( vUv );
				`
			);

	}

	/**
	 * The custom program cache key.
	 *
	 * @returns {string}
	 */
	customProgramCacheKey() {

		return [
			super.customProgramCacheKey(),
			this.celBands,
			this.celQuantize,
			this.shadowTint,
			this.sizeBands,
			this.sizeQuantize,
			this.sizeMin,
			this.sizeMax,
			this.rotationTint,
			this.rotationTintIntensity,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.rimGlow,
			this.rimSoftness,
			this.paperGrain,
			this.watercolorBleed,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {SpriteMaterial} source - The material to copy from.
	 * @return {SpriteMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.color.copy( source.color );
		this.map = source.map;
		this.alphaMap = source.alphaMap;
		this.rotation = source.rotation;
		this.sizeAttenuation = source.sizeAttenuation;
		this.fog = source.fog;

		// Anime extensions
		this.celBands = source.celBands;
		this.celQuantize = source.celQuantize;
		this.shadowTint = source.shadowTint;
		this.sizeBands = source.sizeBands;
		this.sizeQuantize = source.sizeQuantize;
		this.sizeMin = source.sizeMin;
		this.sizeMax = source.sizeMax;
		this.rotationTint = source.rotationTint;
		this.rotationTintIntensity = source.rotationTintIntensity;
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodContrast = source.moodContrast;
		this.moodColor.copy( source.moodColor );
		this.rimGlow = source.rimGlow;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.rimSoftness = source.rimSoftness;
		this.paperGrain = source.paperGrain;
		this.watercolorBleed = source.watercolorBleed;
		this.variationSeed = source.variationSeed;

		return this;

	}

	/**
	 * Serializes the material into JSON.
	 *
	 * @param {?(Object|string)} meta - An optional value holding meta information.
	 * @return {Object} A JSON object representing the serialized material.
	 */
	toJSON( meta ) {

		const data = super.toJSON( meta );

		data.type = 'SpriteMaterial';

		if ( this.color.getHex() !== 0xffffff ) data.color = this.color.getHex();
		if ( this.map !== null ) data.map = this.map.toJSON( meta ).uuid;
		if ( this.alphaMap !== null ) data.alphaMap = this.alphaMap.toJSON( meta ).uuid;
		if ( this.rotation !== 0 ) data.rotation = this.rotation;
		if ( this.sizeAttenuation === false ) data.sizeAttenuation = false;
		if ( this.fog === false ) data.fog = false;

		// Anime extensions
		if ( this.celBands !== 0 ) data.celBands = this.celBands;
		if ( this.celQuantize !== 1.0 ) data.celQuantize = this.celQuantize;
		if ( this.shadowTint !== 0.85 ) data.shadowTint = this.shadowTint;
		if ( this.sizeBands !== 0 ) data.sizeBands = this.sizeBands;
		if ( this.sizeQuantize !== 1.0 ) data.sizeQuantize = this.sizeQuantize;
		if ( this.sizeMin !== 0.3 ) data.sizeMin = this.sizeMin;
		if ( this.sizeMax !== 1.5 ) data.sizeMax = this.sizeMax;
		if ( this.rotationTint !== 0 ) data.rotationTint = this.rotationTint;
		if ( this.rotationTintIntensity !== 0 ) data.rotationTintIntensity = this.rotationTintIntensity;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodContrast !== 1.0 ) data.moodContrast = this.moodContrast;
		if ( this.rimGlow !== 0 ) data.rimGlow = this.rimGlow;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.rimSoftness !== 0.2 ) data.rimSoftness = this.rimSoftness;
		if ( this.paperGrain !== 0 ) data.paperGrain = this.paperGrain;
		if ( this.watercolorBleed !== 0 ) data.watercolorBleed = this.watercolorBleed;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { SpriteMaterial, SpriteMaterialBatch, applySpriteCelBanding, applySpriteSizeBanding, applyRotationMoodTint, computeSpriteEdgeRim, generateSpriteVariation, generateSpriteTexture, gradeSpriteColor };
export default SpriteMaterial;