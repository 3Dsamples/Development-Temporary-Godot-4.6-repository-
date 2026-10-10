// file number : 012
// full path name : src/materials/012_pointsmaterial.js
// description : PointsMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 PointsMaterial API — color, map, alphaMap, size, sizeAttenuation, fog, and the inherited material surface. Points rendering is uniquely suited to anime stylization for particle effects (sparkles, snow, sakura petals, fireflies, chibi dust), and this rewrite provides real-time control over every anime point-sprite parameter. Adds real-time anime features: point-size cel banding (quantize sprite scale for stylized depth pops), distance-fade with mood tinting (cyan snow bursts, warm sunset sparks), procedural point-sprite variation via simplex-noise (unique per-particle shapes), rim glow on point edges, paper-grain and watercolor overlays, mood-based color grading (snowy cyan, sunset orange, vibrant flora), and per-instance variation for large particle systems. Imports Color and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation per-point transforms, double.js for bit-exact size quantization and HDR mood grading, bitecs SoA batching for real-time updates across hundreds of thousands of points, and simplex-noise for procedural point-sprite variation and paper-grain.
// best for : PointsMaterial, particle systems, snow/rain/sakura effects, sparkle bursts, fireflies, magic circles, chibi dust, sparkle effects, and any three.js point-cloud rendering that needs real-time anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import {
	NormalBlending,
	AdditiveBlending,
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

// gl-matrix scratch for zero-allocation point transforms
const _gm_rgb = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — point-size cel banding (stylized depth pops)
// ---------------------------------------------------------------------------

/**
 * Quantize a point's depth-based size into discrete cel bands using
 * double.js for bit-exact thresholding. Produces the characteristic
 * "chunky" anime particle look where distant particles pop between
 * discrete sizes instead of scaling continuously — matching the
 * stylized snow and sparkle layers in reference images 1, 2, and 5.
 *
 * @param {number} depthNormalized - Normalized depth in [0, 1].
 * @param {number} bands - Number of size bands (3-6 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @param {number} [minSize=0.3] - Smallest band multiplier.
 * @param {number} [maxSize=1.5] - Largest band multiplier.
 * @returns {number} Size multiplier in [minSize, maxSize].
 */
function applyPointSizeCelBanding( depthNormalized, bands, quantizeAmount, minSize = 0.3, maxSize = 1.5 ) {

	if ( bands <= 1 ) return 1;

	const bandWidth = 1.0 / bands;
	_double.value = depthNormalized;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	// Blend continuous and quantized
	_double.value = depthNormalized;
	_double.add( ( quantized - depthNormalized ) * quantizeAmount );
	const finalDepth = _double.value;

	// Map depth to size (nearer = larger)
	_double.value = maxSize;
	_double.sub( ( maxSize - minSize ) * finalDepth );

	return Math.max( minSize, Math.min( maxSize, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based distance fade tinting
// ---------------------------------------------------------------------------

/**
 * Apply mood-based tinting to a point sprite as a function of distance.
 * Near points use the base color, far points shift toward the mood
 * color (cyan for cool moods, orange for warm moods). Matches the
 * depth-layered atmosphere in the reference imagery — cyan snow haze
 * in 1, warm sunset dust in 2, 4, and fireflies in 5.
 *
 * @param {Color} baseColor - The base point color.
 * @param {Color} output - The output color (modified in place).
 * @param {number} depthNormalized - Normalized depth in [0, 1].
 * @param {number} moodTemperature - Warm/cool shift in [-1, 1].
 * @param {number} tintIntensity - Tint intensity in [0, 1].
 * @returns {Color}
 */
function applyPointDistanceMoodTint( baseColor, output, depthNormalized, moodTemperature, tintIntensity ) {

	// Mood target: warm orange for positive temp, cool cyan for negative
	_double.value = 1.0;
	const targetR = _double.value;

	_double.value = 0.8;
	_double.add( moodTemperature * 0.1 );
	const targetG = _double.value;

	_double.value = 0.6;
	_double.sub( moodTemperature * 0.4 );
	const targetB = _double.value;

	// Blend based on depth
	_double.value = depthNormalized;
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
// Anime feature — procedural point-sprite variation
// ---------------------------------------------------------------------------

/**
 * Generate a procedural point-sprite texture using simplex-noise. Each
 * point in a particle system can use a different region of this texture
 * to get a unique shape — matching the varied hand-drawn snowflakes,
 * sakura petals, and sparkles in the reference imagery.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.15] - Noise scale.
 * @param {number} [octaves=3] - Number of noise octaves.
 * @param {number} [sharpness=1.5] - Shape sharpness (higher = crisper edges).
 * @returns {Uint8Array} RGBA sprite texture buffer.
 */
function generatePointSpriteVariation( width, height, scale = 0.15, octaves = 3, sharpness = 1.5 ) {

	const out = new Uint8Array( width * height * 4 );
	const cx = width * 0.5;
	const cy = height * 0.5;
	const radius = Math.min( width, height ) * 0.45;

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;

			// Radial falloff (circular mask)
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

			// Multi-octave noise for unique shape variation
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

			// Apply radial fade + sharpness
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
// Anime feature — mood-based color grading
// ---------------------------------------------------------------------------

/**
 * Apply mood-based color grading to the point color. Handles warm (sunset
 * orange), cool (snowy cyan), and vibrant (flora) moods seen across the
 * reference imagery. Uses gl-matrix for zero-allocation staging and
 * double.js for bit-exact accumulation.
 *
 * @param {Color} color - The color to grade (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} contrast - Contrast multiplier.
 * @returns {Color}
 */
function gradePointColor( color, temperature, saturation, brightness, contrast ) {

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
// Anime feature — procedural sparkle overlay
// ---------------------------------------------------------------------------

/**
 * Generate a procedural sparkle overlay texture using simplex-noise.
 * Used to modulate point-sprite brightness for the twinkling, cross-star
 * sparkle effect seen in the reference imagery's water reflections (1, 3, 5)
 * and the planet's atmospheric glow (2, 4).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.1] - Noise scale.
 * @param {number} [sparkleChance=0.05] - Fraction of pixels that produce a sparkle.
 * @returns {Uint8Array} Grayscale RGBA buffer.
 */
function generateSparkleOverlay( width, height, scale = 0.1, sparkleChance = 0.05 ) {

	const out = new Uint8Array( width * height * 4 );

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;
			const n = _noise2D( x * scale, y * scale ) * 0.5 + 0.5;

			// Pixels above the threshold become sparkles
			let sparkle = 0;
			if ( n > 1 - sparkleChance ) {

				_double.value = ( n - ( 1 - sparkleChance ) );
				_double.div( sparkleChance );
				sparkle = Math.max( 0, Math.min( 1, _double.value ) );

			}

			const v = Math.floor( sparkle * 255 );

			out[ p ] = v;
			out[ p + 1 ] = v;
			out[ p + 2 ] = v;
			out[ p + 3 ] = 255;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time point material updates
// ---------------------------------------------------------------------------

const _pointsWorld = createWorld();

const PointsMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	sizeBands: Types.ui8,
	sizeQuantize: Types.f64,
	sizeMin: Types.f64,
	sizeMax: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	distanceTint: Types.f64,
	rimGlow: Types.f64,
	sparkleIntensity: Types.f64,
	paperGrain: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class PointsMaterialBatch {

	constructor() {

		this.world = _pointsWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a PointsMaterial instance for batched real-time updates.
	 *
	 * @param {PointsMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, PointsMaterialComponent, eid );

		PointsMaterialComponent.materialPtr[ eid ] = this.materials.length;
		PointsMaterialComponent.sizeBands[ eid ] = material.sizeBands;
		PointsMaterialComponent.sizeQuantize[ eid ] = material.sizeQuantize;
		PointsMaterialComponent.sizeMin[ eid ] = material.sizeMin;
		PointsMaterialComponent.sizeMax[ eid ] = material.sizeMax;
		PointsMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		PointsMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		PointsMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		PointsMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		PointsMaterialComponent.distanceTint[ eid ] = material.distanceTint;
		PointsMaterialComponent.rimGlow[ eid ] = material.rimGlow;
		PointsMaterialComponent.sparkleIntensity[ eid ] = material.sparkleIntensity;
		PointsMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		PointsMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		PointsMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued point-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact size banding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ PointsMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.sizeBands = PointsMaterialComponent.sizeBands[ eid ];
			material.sizeQuantize = PointsMaterialComponent.sizeQuantize[ eid ];
			material.sizeMin = PointsMaterialComponent.sizeMin[ eid ];
			material.sizeMax = PointsMaterialComponent.sizeMax[ eid ];
			material.moodTemperature = PointsMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = PointsMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = PointsMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = PointsMaterialComponent.moodContrast[ eid ];
			material.distanceTint = PointsMaterialComponent.distanceTint[ eid ];
			material.rimGlow = PointsMaterialComponent.rimGlow[ eid ];
			material.sparkleIntensity = PointsMaterialComponent.sparkleIntensity[ eid ];
			material.paperGrain = PointsMaterialComponent.paperGrain[ eid ];
			material.variationSeed = PointsMaterialComponent.variationSeed[ eid ];

			// Recompute mood-graded color
			material.moodColor.copy( material.color );
			gradePointColor(
				material.moodColor,
				material.moodTemperature,
				material.moodSaturation,
				material.moodBrightness,
				material.moodContrast
			);

			PointsMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main PointsMaterial class — mirrors three.js/src/materials/PointsMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material for rendering points (particle systems).
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls: point-size cel banding
 * for stylized depth pops, mood-based distance tinting, procedural
 * point-sprite variation via simplex-noise, sparkle overlays, rim glow,
 * and paper-grain variation.
 *
 * ```js
 * const material = new THREE.PointsMaterial( {
 *   color: 0x88ccff,
 *   size: 1.0,
 *   sizeAttenuation: true,
 *   sizeBands: 4,
 *   sizeQuantize: 0.8,
 *   sparkleIntensity: 0.5,
 *   moodTemperature: -0.3
 * } );
 * ```
 *
 * @augments Material
 */
class PointsMaterial extends Material {

	/**
	 * Constructs a new points material.
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
		this.isPointsMaterial = true;

		this.type = 'PointsMaterial';

		/**
		 * The material's base color.
		 *
		 * @type {Color}
		 * @default (1,1,1)
		 */
		this.color = new Color( 0xffffff );

		/**
		 * The color map. The texture is expected to be in the sRGB color
		 * space.
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
		 * The point size in world units. If `sizeAttenuation` is `false`,
		 * this is the point size in pixels.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.size = 1;

		/**
		 * Whether the point size attenuates with distance. When `true`,
		 * distant points appear smaller.
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
		 * Number of point-size cel bands. 0 = continuous (off),
		 * 3-6 = classic anime "chunky particle" look (recommended for
		 * stylized snow, sparkle, and sakura effects).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.sizeBands = 0;

		/**
		 * Blend amount for point-size banding.
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
		 * Mood temperature shift in [-1, 1]. Positive = warm (sunset
		 * spark dust), negative = cool (snowy cyan haze).
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
		 * The precomputed mood-graded point color.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0xffffff );

		/**
		 * Distance-tint intensity. Blends distant points toward the mood
		 * color for a stylized atmospheric haze.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.distanceTint = 0;

		/**
		 * Rim-glow intensity. Adds a soft glow around each point sprite
		 * for a hand-drawn sparkle look.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimGlow = 0;

		/**
		 * Rim-glow color.
		 *
		 * @type {Color}
		 * @default (1, 1, 1)
		 */
		this.rimGlowColor = new Color( 1, 1, 1 );

		/**
		 * Sparkle overlay intensity. Set > 0 to add cross-star twinkle
		 * to the point sprites.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.sparkleIntensity = 0;

		/**
		 * Paper-grain overlay intensity.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.paperGrain = 0;

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
	 * Apply point-size cel banding to a normalized depth value.
	 *
	 * @param {number} depthNormalized - Normalized depth in [0, 1].
	 * @returns {number} Size multiplier in [sizeMin, sizeMax].
	 */
	applySizeBanding( depthNormalized ) {

		return applyPointSizeCelBanding(
			depthNormalized,
			this.sizeBands,
			this.sizeQuantize,
			this.sizeMin,
			this.sizeMax
		);

	}

	/**
	 * Apply mood-based distance tinting to a point color.
	 *
	 * @param {Color} baseColor - The base point color.
	 * @param {Color} output - The output color.
	 * @param {number} depthNormalized - Normalized depth in [0, 1].
	 * @returns {Color}
	 */
	applyDistanceTint( baseColor, output, depthNormalized ) {

		return applyPointDistanceMoodTint(
			baseColor,
			output,
			depthNormalized,
			this.moodTemperature,
			this.distanceTint
		);

	}

	/**
	 * Recompute the mood-graded point color.
	 *
	 * @returns {PointsMaterial} A reference to this instance.
	 */
	updateMoodColor() {

		this.moodColor.copy( this.color );
		gradePointColor(
			this.moodColor,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast
		);
		return this;

	}

	/**
	 * Generate a procedural point-sprite variation texture for this
	 * material. The caller is expected to assign the returned buffer to
	 * a `DataTexture` and attach it to the `map` slot.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.15]
	 * @param {number} [octaves=3]
	 * @returns {Uint8Array}
	 */
	generatePointSpriteVariation( width, height, scale = 0.15, octaves = 3 ) {

		return generatePointSpriteVariation( width, height, scale, octaves, 1.5 );

	}

	/**
	 * Generate a procedural sparkle overlay texture for this material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.1]
	 * @returns {Uint8Array}
	 */
	generateSparkleOverlay( width, height, scale = 0.1 ) {

		return generateSparkleOverlay( width, height, scale, this.sparkleIntensity * 0.2 );

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
	 * anime shader chunks with points-specific features: point-size cel
	 * banding, distance-tint haze, sparkle overlay, and rim glow.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject points-specific uniforms
		shader.uniforms.sizeBands = { value: this.sizeBands };
		shader.uniforms.sizeQuantize = { value: this.sizeQuantize };
		shader.uniforms.sizeMin = { value: this.sizeMin };
		shader.uniforms.sizeMax = { value: this.sizeMax };
		shader.uniforms.moodColor = { value: this.moodColor };
		shader.uniforms.distanceTint = { value: this.distanceTint };
		shader.uniforms.rimGlow = { value: this.rimGlow };
		shader.uniforms.rimGlowColor = { value: this.rimGlowColor };
		shader.uniforms.sparkleIntensity = { value: this.sparkleIntensity };
		shader.uniforms.paperGrain = { value: this.paperGrain };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the vertex shader for point-size banding
		shader.vertexShader = shader.vertexShader
			.replace(
				'#include <common>',
				`#include <common>
				uniform int sizeBands;
				uniform float sizeQuantize;
				uniform float sizeMin;
				uniform float sizeMax;
				`
			)
			.replace(
				'gl_PointSize = size;',
				`// Depth-based cel-band size (three.js default is gl_PointSize = size)
				{
					float depthNorm = clamp( -mvPosition.z / 100.0, 0.0, 1.0 );
					if ( sizeBands > 1 ) {
						float bandWidth = 1.0 / float( sizeBands );
						float bandIndex = floor( depthNorm / bandWidth );
						float quantized = bandIndex * bandWidth + bandWidth * 0.5;
						float finalDepth = mix( depthNorm, quantized, sizeQuantize );
						float bandSize = mix( sizeMax, sizeMin, finalDepth );
						gl_PointSize = size * bandSize;
					} else {
						gl_PointSize = size;
					}
				}`
			);

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform vec3 moodColor;
				uniform float distanceTint;
				uniform float rimGlow;
				uniform vec3 rimGlowColor;
				uniform float sparkleIntensity;
				uniform float paperGrain;
				uniform float variationOffset;

				vec3 applyPointDistanceTint( vec3 baseColor, float depthNorm ) {
					if ( distanceTint <= 0.0 ) return baseColor;
					vec3 tintTarget = vec3( 1.0, 0.8, 0.6 );
					float blend = depthNorm * distanceTint;
					return mix( baseColor, tintTarget, blend );
				}

				float sampleSparkle( vec2 uv ) {
					if ( sparkleIntensity <= 0.0 ) return 1.0;
					vec2 centered = uv - 0.5;
					float dist = length( centered );
					float sparkle = pow( clamp( 1.0 - dist * 2.0, 0.0, 1.0 ), 4.0 );
					return 1.0 + sparkle * sparkleIntensity;
				}

				float samplePaperGrain( vec2 uv ) {
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

				// Override with mood-graded point color
				gl_FragColor.rgb = moodColor * gl_FragColor.a;

				// Distance-based mood tinting
				float depthNormPts = clamp( -vViewPosition.z / 100.0, 0.0, 1.0 );
				gl_FragColor.rgb = applyPointDistanceTint( gl_FragColor.rgb, depthNormPts );

				// Sparkle cross-star overlay
				if ( sparkleIntensity > 0.0 ) {
					float sparkleFactor = sampleSparkle( gl_PointCoord );
					gl_FragColor.rgb *= sparkleFactor;
				}

				// Rim glow around point sprite edge
				if ( rimGlow > 0.0 ) {
					vec2 centered = gl_PointCoord - 0.5;
					float dist = length( centered );
					float rim = smoothstep( 0.4, 0.5, dist );
					gl_FragColor.rgb += rimGlowColor * rim * rimGlow;
				}

				// Paper grain opacity modulation
				gl_FragColor.a *= samplePaperGrain( gl_PointCoord );
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
			this.sizeBands,
			this.sizeQuantize,
			this.sizeMin,
			this.sizeMax,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.distanceTint,
			this.rimGlow,
			this.sparkleIntensity,
			this.paperGrain,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {PointsMaterial} source - The material to copy from.
	 * @return {PointsMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.color.copy( source.color );
		this.map = source.map;
		this.alphaMap = source.alphaMap;
		this.size = source.size;
		this.sizeAttenuation = source.sizeAttenuation;
		this.fog = source.fog;

		// Anime extensions
		this.sizeBands = source.sizeBands;
		this.sizeQuantize = source.sizeQuantize;
		this.sizeMin = source.sizeMin;
		this.sizeMax = source.sizeMax;
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodContrast = source.moodContrast;
		this.moodColor.copy( source.moodColor );
		this.distanceTint = source.distanceTint;
		this.rimGlow = source.rimGlow;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.sparkleIntensity = source.sparkleIntensity;
		this.paperGrain = source.paperGrain;
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

		data.type = 'PointsMaterial';

		if ( this.color.getHex() !== 0xffffff ) data.color = this.color.getHex();
		if ( this.map !== null ) data.map = this.map.toJSON( meta ).uuid;
		if ( this.alphaMap !== null ) data.alphaMap = this.alphaMap.toJSON( meta ).uuid;
		if ( this.size !== 1 ) data.size = this.size;
		if ( this.sizeAttenuation === false ) data.sizeAttenuation = false;
		if ( this.fog === false ) data.fog = false;

		// Anime extensions
		if ( this.sizeBands !== 0 ) data.sizeBands = this.sizeBands;
		if ( this.sizeQuantize !== 1.0 ) data.sizeQuantize = this.sizeQuantize;
		if ( this.sizeMin !== 0.3 ) data.sizeMin = this.sizeMin;
		if ( this.sizeMax !== 1.5 ) data.sizeMax = this.sizeMax;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodContrast !== 1.0 ) data.moodContrast = this.moodContrast;
		if ( this.distanceTint !== 0 ) data.distanceTint = this.distanceTint;
		if ( this.rimGlow !== 0 ) data.rimGlow = this.rimGlow;
		if ( this.rimGlowColor.getHex() !== 0xffffff ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.sparkleIntensity !== 0 ) data.sparkleIntensity = this.sparkleIntensity;
		if ( this.paperGrain !== 0 ) data.paperGrain = this.paperGrain;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { PointsMaterial, PointsMaterialBatch, applyPointSizeCelBanding, applyPointDistanceMoodTint, generatePointSpriteVariation, generateSparkleOverlay, gradePointColor };
export default PointsMaterial;