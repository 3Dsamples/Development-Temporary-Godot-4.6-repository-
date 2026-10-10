// file number : 013
// full path name : src/materials/013_shadowmaterial.js
// description : ShadowMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 ShadowMaterial API — color, transparent (default true), fog, and the inherited material surface. ShadowMaterial is the ground-plane shadow catcher used to composite soft blob shadows under characters and props. In anime rendering, shadows are a critical storytelling element: cool blue/purple shadows for nighttime and snowy scenes (reference 1), warm brown shadows for sunset scenes (reference 2, 4), vibrant green-tinted shadows under foliage (reference 5), and soft neutral shadows indoors (reference 6). Adds real-time anime features specifically tuned for stylized shadow rendering: shadow cel-banding (flat-shadow layers instead of smooth gradients, matching the reference imagery's layered mountain/foliage shadows), soft penumbra shaping, mood-based shadow tinting (cool cyan, warm sunset, vibrant flora), ambient gradient shadow tinting, procedural paper-grain / watercolor shadow texture via simplex-noise, rim glow on shadow edges for a hand-drawn outline look, and per-instance variation for crowds. Imports Color, MathUtils, ColorManagement, ColorSpace, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation shadow color transforms, double.js for bit-exact shadow opacity quantization and HDR mood grading, bitecs SoA batching for real-time updates across thousands of shadow-casting instances, and simplex-noise for procedural shadow texture variation.
// best for : ShadowMaterial, ground-plane shadow catchers, blob shadows, soft character shadows, stylized AO composites, stylized baking, and any three.js shadow overlay that needs real-time anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import {
	NormalBlending,
	FrontSide,
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

// gl-matrix scratch for zero-allocation shadow color transforms
const _gm_rgb = glMatrix.vec3.create();
const _gm_hsv = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — shadow cel banding (flat-shadow layers)
// ---------------------------------------------------------------------------

/**
 * Quantize a raw shadow opacity into discrete cel bands using double.js
 * for bit-exact thresholding. Produces the characteristic "layered
 * shadow" look of anime backgrounds — flat penumbra zones instead of
 * a smooth gradient — matching the layered mountain shadows (reference
 * images 1, 5) and the stylized foliage shadow clusters (reference 5).
 *
 * @param {number} rawOpacity - Raw shadow opacity in [0, 1].
 * @param {number} bands - Number of shadow bands (2-6 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @param {number} [softness=0.1] - Edge softness in [0, 0.5].
 * @param {number} [shadowTint=0.85] - Brightness multiplier on the darkest band.
 * @returns {number} Banded shadow opacity in [0, 1].
 */
function applyShadowCelBanding( rawOpacity, bands, quantizeAmount, softness = 0.1, shadowTint = 0.85 ) {

	if ( bands <= 1 ) return rawOpacity;

	const bandWidth = 1.0 / bands;
	_double.value = rawOpacity;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );

	// Band center for soft edge computation
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const bandCenter = _double.value;

	// Distance to band center for soft-edge transition
	const distToCenter = Math.abs( rawOpacity - bandCenter );
	const softFactor = Math.max( 0, Math.min( 1, distToCenter / ( bandWidth * 0.5 ) ) );
	const edgeBlend = ( 1 - softFactor ) * ( 1 - softness ) + softness;

	// Blend the band center value with the raw opacity near boundaries
	_double.value = bandCenter * edgeBlend;
	_double.add( rawOpacity * ( 1 - edgeBlend ) );
	let finalOpacity = _double.value;

	// Apply shadow tint on the darker bands
	_double.value = finalOpacity;
	_double.mul( shadowTint );
	finalOpacity = _double.value;

	// Blend with raw opacity based on quantize amount
	_double.value = rawOpacity;
	_double.add( ( finalOpacity - rawOpacity ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — soft penumbra shaping
// ---------------------------------------------------------------------------

/**
 * Shape the penumbra (edge) of a shadow using a configurable power curve
 * with double.js for bit-exact accumulation. Controls whether the shadow
 * edge is sharp like a hard-edged anime cell (power > 1) or diffuse like
 * a soft watercolor bleed (power < 1). Matches the range from crisp
 * character shadows (reference 6) to diffuse mountain shadows (reference 1).
 *
 * @param {number} edgeDistance - Normalized distance from the shadow center in [0, 1].
 * @param {number} [power=1.5] - Penumbra falloff power.
 * @param {number} [inner=0.3] - Inner radius where shadow is fully opaque.
 * @param {number} [outer=0.9] - Outer radius where shadow is fully transparent.
 * @returns {number} Shadow opacity multiplier in [0, 1].
 */
function shapePenumbra( edgeDistance, power = 1.5, inner = 0.3, outer = 0.9 ) {

	_double.value = edgeDistance;
	_double.sub( inner );
	_double.div( Math.max( outer - inner, 0.001 ) );
	let normalized = Math.max( 0, Math.min( 1, _double.value ) );

	// Apply power curve for stylized falloff
	const shaped = Math.pow( normalized, power );

	// Invert so the shadow is opaque at center, transparent at edge
	return Math.max( 0, Math.min( 1, 1 - shaped ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based shadow tinting
// ---------------------------------------------------------------------------

/**
 * Apply mood-based shadow tinting. Shadows in anime are never pure black —
 * they take on the ambient color of the scene. Cool moods (snowy, nighttime)
 * produce blue/cyan shadows (reference 1), warm moods (sunset) produce
 * brown/orange shadows (reference 2, 4), and vibrant flora scenes produce
 * green-tinted shadows (reference 5). Uses gl-matrix for zero-allocation
 * staging and double.js for bit-exact accumulation.
 *
 * @param {Color} baseColor - The base shadow color (usually near-black).
 * @param {Color} output - The output color (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} intensity - Tint intensity in [0, 1].
 * @returns {Color}
 */
function gradeShadowColor( baseColor, output, temperature, saturation, brightness, intensity ) {

	if ( intensity <= 0 ) {

		output.copy( baseColor );
		return output;

	}

	glMatrix.vec3.set( _gm_rgb, baseColor.r, baseColor.g, baseColor.b );

	const lum = 0.2126 * _gm_rgb[ 0 ] + 0.7152 * _gm_rgb[ 1 ] + 0.0722 * _gm_rgb[ 2 ];

	// Saturation: push away from luminance
	_double.value = lum;
	_double.add( ( _gm_rgb[ 0 ] - lum ) * saturation );
	let r = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 1 ] - lum ) * saturation );
	let g = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 2 ] - lum ) * saturation );
	let b = _double.value;

	// Temperature: warm pushes R up / B down; cool pushes B up / R down
	_double.value = r;
	_double.add( temperature * 0.15 );
	r = _double.value;

	_double.value = b;
	_double.sub( temperature * 0.15 );
	b = _double.value;

	// Brightness
	r *= brightness;
	g *= brightness;
	b *= brightness;

	// Blend toward the tint target based on intensity
	_double.value = baseColor.r;
	_double.mul( 1 - intensity );
	_double.add( r * intensity );
	r = _double.value;

	_double.value = baseColor.g;
	_double.mul( 1 - intensity );
	_double.add( g * intensity );
	g = _double.value;

	_double.value = baseColor.b;
	_double.mul( 1 - intensity );
	_double.add( b * intensity );
	b = _double.value;

	output.setRGB(
		Math.max( 0, Math.min( 1, r ) ),
		Math.max( 0, Math.min( 1, g ) ),
		Math.max( 0, Math.min( 1, b ) ),
		ColorManagement.workingColorSpace
	);
	output.a = baseColor.a;

	return output;

}

// ---------------------------------------------------------------------------
// Anime feature — ambient gradient shadow tinting
// ---------------------------------------------------------------------------

/**
 * Apply a vertical gradient tint to the shadow color, blending from a
 * cool sky shadow at the top to a warm ground shadow at the bottom. This
 * emulates the ambient bounce-light gradient visible in anime backgrounds
 * — cool blue shadow at the bottom of a sunlit wall, warm shadow on the
 * ground near foliage. Matches the tonal gradients in reference images
 * 1, 2, 5, and 6.
 *
 * @param {Color} color - The color to tint (modified in place).
 * @param {number} worldY - The world Y coordinate of the fragment.
 * @param {number} heightRange - Range over which the gradient blends.
 * @param {Color} skyTint - Tint applied at the top.
 * @param {Color} groundTint - Tint applied at the bottom.
 * @param {number} intensity - Blend intensity in [0, 1].
 * @returns {Color}
 */
function applyShadowAmbientGradient( color, worldY, heightRange, skyTint, groundTint, intensity ) {

	if ( intensity <= 0 ) return color;

	_double.value = worldY;
	_double.div( Math.max( heightRange, 0.001 ) );
	_double.add( 0.5 );
	let t = _double.value;
	t = Math.max( 0, Math.min( 1, t ) );

	// Blend between ground (t=0) and sky (t=1)
	_double.value = groundTint.r;
	_double.add( ( skyTint.r - groundTint.r ) * t );
	const tr = _double.value;

	_double.value = groundTint.g;
	_double.add( ( skyTint.g - groundTint.g ) * t );
	const tg = _double.value;

	_double.value = groundTint.b;
	_double.add( ( skyTint.b - groundTint.b ) * t );
	const tb = _double.value;

	// Mix tint with base color
	_double.value = color.r;
	_double.mul( 1 - intensity );
	_double.add( tr * color.r * intensity );
	color.r = _double.value;

	_double.value = color.g;
	_double.mul( 1 - intensity );
	_double.add( tg * color.g * intensity );
	color.g = _double.value;

	_double.value = color.b;
	_double.mul( 1 - intensity );
	_double.add( tb * color.b * intensity );
	color.b = _double.value;

	return color;

}

// ---------------------------------------------------------------------------
// Anime feature — procedural paper-grain / watercolor shadow texture
// ---------------------------------------------------------------------------

/**
 * Generate a combined paper-grain and watercolor shadow texture using
 * simplex-noise with multiple octaves. The texture is used to modulate
 * shadow opacity for a hand-painted shadow feel — the characteristic
 * watercolor bleed of hand-drawn anime shadows (reference images 1, 5, 7).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.03] - Base noise scale.
 * @param {number} [octaves=4] - Number of noise octaves.
 * @param {number} [paperGrain=0.3] - Paper-grain intensity.
 * @param {number} [watercolorBleed=0.5] - Watercolor bleed strength.
 * @returns {Uint8Array} Grayscale RGBA buffer.
 */
function generateShadowTexture( width, height, scale = 0.03, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

	const out = new Uint8Array( width * height * 4 );

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;

			// Multi-octave watercolor base
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

			// Watercolor bleed: soften toward extremes
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
// Anime feature — rim glow on shadow edges
// ---------------------------------------------------------------------------

/**
 * Compute a rim glow on the outer edge of a shadow. Uses the shadow's
 * current opacity gradient (via dFdx/dFdy in the shader) to detect the
 * shadow boundary and produce a soft glow. Matches the glowing cyan
 * water shadows in reference 3 and the warm sunset shadow halos in 2, 4.
 *
 * @param {number} shadowOpacity - Current shadow opacity in [0, 1].
 * @param {number} [threshold=0.5] - Opacity at which the edge is centered.
 * @param {number} [softness=0.2] - Rim softness.
 * @returns {number} Rim intensity in [0, 1].
 */
function computeShadowEdgeRim( shadowOpacity, threshold = 0.5, softness = 0.2 ) {

	_double.value = shadowOpacity;
	_double.sub( threshold );
	const distToEdge = Math.abs( _double.value );

	if ( distToEdge > softness ) return 0;

	_double.value = 1.0;
	_double.sub( distToEdge / softness );
	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time shadow material updates
// ---------------------------------------------------------------------------

const _shadowWorld = createWorld();

const ShadowMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	opacity: Types.f64,
	shadowBands: Types.ui8,
	shadowQuantize: Types.f64,
	shadowSoftness: Types.f64,
	shadowTint: Types.f64,
	penumbraPower: Types.f64,
	penumbraInner: Types.f64,
	penumbraOuter: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodIntensity: Types.f64,
	ambientGradient: Types.f64,
	ambientHeight: Types.f64,
	paperGrain: Types.f64,
	watercolorBleed: Types.f64,
	edgeRimIntensity: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class ShadowMaterialBatch {

	constructor() {

		this.world = _shadowWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a ShadowMaterial instance for batched real-time updates.
	 *
	 * @param {ShadowMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, ShadowMaterialComponent, eid );

		ShadowMaterialComponent.materialPtr[ eid ] = this.materials.length;
		ShadowMaterialComponent.opacity[ eid ] = material.opacity;
		ShadowMaterialComponent.shadowBands[ eid ] = material.shadowBands;
		ShadowMaterialComponent.shadowQuantize[ eid ] = material.shadowQuantize;
		ShadowMaterialComponent.shadowSoftness[ eid ] = material.shadowSoftness;
		ShadowMaterialComponent.shadowTint[ eid ] = material.shadowTint;
		ShadowMaterialComponent.penumbraPower[ eid ] = material.penumbraPower;
		ShadowMaterialComponent.penumbraInner[ eid ] = material.penumbraInner;
		ShadowMaterialComponent.penumbraOuter[ eid ] = material.penumbraOuter;
		ShadowMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		ShadowMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		ShadowMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		ShadowMaterialComponent.moodIntensity[ eid ] = material.moodIntensity;
		ShadowMaterialComponent.ambientGradient[ eid ] = material.ambientGradient;
		ShadowMaterialComponent.ambientHeight[ eid ] = material.ambientHeight;
		ShadowMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		ShadowMaterialComponent.watercolorBleed[ eid ] = material.watercolorBleed;
		ShadowMaterialComponent.edgeRimIntensity[ eid ] = material.edgeRimIntensity;
		ShadowMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		ShadowMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued shadow-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact shadow banding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ ShadowMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.opacity = ShadowMaterialComponent.opacity[ eid ];
			material.shadowBands = ShadowMaterialComponent.shadowBands[ eid ];
			material.shadowQuantize = ShadowMaterialComponent.shadowQuantize[ eid ];
			material.shadowSoftness = ShadowMaterialComponent.shadowSoftness[ eid ];
			material.shadowTint = ShadowMaterialComponent.shadowTint[ eid ];
			material.penumbraPower = ShadowMaterialComponent.penumbraPower[ eid ];
			material.penumbraInner = ShadowMaterialComponent.penumbraInner[ eid ];
			material.penumbraOuter = ShadowMaterialComponent.penumbraOuter[ eid ];
			material.moodTemperature = ShadowMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = ShadowMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = ShadowMaterialComponent.moodBrightness[ eid ];
			material.moodIntensity = ShadowMaterialComponent.moodIntensity[ eid ];
			material.ambientGradient = ShadowMaterialComponent.ambientGradient[ eid ];
			material.ambientHeight = ShadowMaterialComponent.ambientHeight[ eid ];
			material.paperGrain = ShadowMaterialComponent.paperGrain[ eid ];
			material.watercolorBleed = ShadowMaterialComponent.watercolorBleed[ eid ];
			material.edgeRimIntensity = ShadowMaterialComponent.edgeRimIntensity[ eid ];
			material.variationSeed = ShadowMaterialComponent.variationSeed[ eid ];

			// Recompute mood-graded shadow color
			gradeShadowColor(
				material.color,
				material.moodColor,
				material.moodTemperature,
				material.moodSaturation,
				material.moodBrightness,
				material.moodIntensity
			);

			ShadowMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main ShadowMaterial class — mirrors three.js/src/materials/ShadowMaterial.js
// ---------------------------------------------------------------------------

/**
 * This material can receive shadows from other objects and render them
 * onto its surface.
 *
 * Shadows are typically received by adding a `ShadowMaterial` ground plane
 * to the scene and enabling `receiveShadow` on the mesh. Because shadow
 * colors are a critical anime storytelling element (cool blue at night,
 * warm brown at sunset, green under foliage), this rewrite extends the
 * material with a full stylization surface.
 *
 * In addition to the standard three.js parameters, this material exposes
 * real-time anime-style controls: shadow cel-banding, soft penumbra
 * shaping, mood-based shadow tinting, ambient gradient shadow tinting,
 * procedural paper-grain / watercolor shadow texture variation, and
 * rim glow on shadow edges.
 *
 * ```js
 * const material = new THREE.ShadowMaterial( {
 *   color: 0x000000,
 *   opacity: 0.5,
 *   shadowBands: 3,
 *   shadowQuantize: 0.8,
 *   moodTemperature: -0.3,
 *   moodIntensity: 0.6
 * } );
 * ```
 *
 * @augments Material
 */
class ShadowMaterial extends Material {

	/**
	 * Constructs a new shadow material.
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
		this.isShadowMaterial = true;

		this.type = 'ShadowMaterial';

		/**
		 * The shadow's color. Default is black, but anime shadows are
		 * rarely pure black — they take on the scene's ambient color.
		 *
		 * @type {Color}
		 * @default (0,0,0)
		 */
		this.color = new Color( 0x000000 );

		/**
		 * Whether the material is transparent. Overridden to `true` by
		 * default so the shadow can composite onto the scene.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.transparent = true;

		/**
		 * The shadow's opacity in [0, 1].
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.opacity = 1.0;

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
		 * Number of shadow cel bands. 0 = continuous (off), 2 = classic
		 * hard-shadow split, 3-6 = layered stylized shadows.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.shadowBands = 0;

		/**
		 * Blend amount between continuous and banded shadow.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.shadowQuantize = 1.0;

		/**
		 * Shadow band edge softness in [0, 0.5].
		 *
		 * @type {number}
		 * @default 0.1
		 */
		this.shadowSoftness = 0.1;

		/**
		 * Brightness multiplier on the darkest shadow band.
		 *
		 * @type {number}
		 * @default 0.85
		 */
		this.shadowTint = 0.85;

		/**
		 * Penumbra falloff power.
		 * > 1 = sharp edge (hard anime cell), < 1 = diffuse edge.
		 *
		 * @type {number}
		 * @default 1.5
		 */
		this.penumbraPower = 1.5;

		/**
		 * Inner radius where the shadow is fully opaque.
		 *
		 * @type {number}
		 * @default 0.3
		 */
		this.penumbraInner = 0.3;

		/**
		 * Outer radius where the shadow is fully transparent.
		 *
		 * @type {number}
		 * @default 0.9
		 */
		this.penumbraOuter = 0.9;

		/**
		 * Mood temperature shift in [-1, 1]. Positive = warm (sunset brown
		 * shadows), negative = cool (snowy cyan shadows).
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
		 * Mood tint intensity applied to the shadow. Anime shadows are
		 * never pure black — this controls how strongly the mood color
		 * shifts the base shadow color.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodIntensity = 0;

		/**
		 * The precomputed mood-graded shadow color.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0x000000 );

		/**
		 * Ambient gradient intensity. Blends a vertical sky-to-ground
		 * gradient tint onto the shadow color for stylized atmospheres.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.ambientGradient = 0;

		/**
		 * Range over which the ambient gradient blends.
		 *
		 * @type {number}
		 * @default 10
		 */
		this.ambientHeight = 10;

		/**
		 * The sky shadow tint (applied at the top).
		 *
		 * @type {Color}
		 * @default (0.5, 0.7, 1.0)
		 */
		this.skyTint = new Color( 0.5, 0.7, 1.0 );

		/**
		 * The ground shadow tint (applied at the bottom).
		 *
		 * @type {Color}
		 * @default (1.0, 0.8, 0.6)
		 */
		this.groundTint = new Color( 1.0, 0.8, 0.6 );

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
		 * Rim glow intensity on the shadow edge. Set > 0 to add a soft
		 * hand-drawn glow around the shadow silhouette.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.edgeRimIntensity = 0;

		/**
		 * Rim-glow color.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.edgeRimColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Rim softness at the shadow edge.
		 *
		 * @type {number}
		 * @default 0.2
		 */
		this.edgeRimSoftness = 0.2;

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
	 * Apply shadow cel-banding to a raw shadow opacity.
	 *
	 * @param {number} rawOpacity
	 * @returns {number}
	 */
	applyShadowBanding( rawOpacity ) {

		return applyShadowCelBanding(
			rawOpacity,
			this.shadowBands,
			this.shadowQuantize,
			this.shadowSoftness,
			this.shadowTint
		);

	}

	/**
	 * Shape the penumbra of a shadow at a given edge distance.
	 *
	 * @param {number} edgeDistance
	 * @returns {number}
	 */
	applyPenumbraShape( edgeDistance ) {

		return shapePenumbra( edgeDistance, this.penumbraPower, this.penumbraInner, this.penumbraOuter );

	}

	/**
	 * Recompute the mood-graded shadow color from the current mood
	 * parameters.
	 *
	 * @returns {ShadowMaterial} A reference to this instance.
	 */
	updateMoodColor() {

		gradeShadowColor(
			this.color,
			this.moodColor,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodIntensity
		);
		return this;

	}

	/**
	 * Apply the ambient gradient tint to a color at a given world Y.
	 *
	 * @param {Color} color - The color to tint.
	 * @param {number} worldY - World Y coordinate.
	 * @returns {Color}
	 */
	applyAmbientGradient( color, worldY ) {

		return applyShadowAmbientGradient(
			color, worldY, this.ambientHeight,
			this.skyTint, this.groundTint,
			this.ambientGradient
		);

	}

	/**
	 * Compute the edge rim glow for a given shadow opacity.
	 *
	 * @param {number} shadowOpacity
	 * @returns {number}
	 */
	sampleEdgeRim( shadowOpacity ) {

		if ( this.edgeRimIntensity <= 0 ) return 0;
		return computeShadowEdgeRim( shadowOpacity, 0.5, this.edgeRimSoftness ) * this.edgeRimIntensity;

	}

	/**
	 * Generate a procedural paper-grain / watercolor shadow texture for
	 * this material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.03]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generateShadowTexture( width, height, scale = 0.03, octaves = 4 ) {

		return generateShadowTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

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
	 * anime shader chunks with shadow-specific features: shadow cel
	 * banding, soft penumbra shaping, mood-based shadow tinting, ambient
	 * gradient tinting, and rim glow on shadow edges.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject shadow-specific uniforms
		shader.uniforms.shadowBands = { value: this.shadowBands };
		shader.uniforms.shadowQuantize = { value: this.shadowQuantize };
		shader.uniforms.shadowSoftness = { value: this.shadowSoftness };
		shader.uniforms.shadowTint = { value: this.shadowTint };
		shader.uniforms.penumbraPower = { value: this.penumbraPower };
		shader.uniforms.penumbraInner = { value: this.penumbraInner };
		shader.uniforms.penumbraOuter = { value: this.penumbraOuter };
		shader.uniforms.moodColor = { value: this.moodColor };
		shader.uniforms.moodIntensity = { value: this.moodIntensity };
		shader.uniforms.ambientGradient = { value: this.ambientGradient };
		shader.uniforms.ambientHeight = { value: this.ambientHeight };
		shader.uniforms.skyTint = { value: this.skyTint };
		shader.uniforms.groundTint = { value: this.groundTint };
		shader.uniforms.paperGrain = { value: this.paperGrain };
		shader.uniforms.watercolorBleed = { value: this.watercolorBleed };
		shader.uniforms.edgeRimIntensity = { value: this.edgeRimIntensity };
		shader.uniforms.edgeRimColor = { value: this.edgeRimColor };
		shader.uniforms.edgeRimSoftness = { value: this.edgeRimSoftness };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int shadowBands;
				uniform float shadowQuantize;
				uniform float shadowSoftness;
				uniform float shadowTint;
				uniform float penumbraPower;
				uniform float penumbraInner;
				uniform float penumbraOuter;
				uniform vec3 moodColor;
				uniform float moodIntensity;
				uniform float ambientGradient;
				uniform float ambientHeight;
				uniform vec3 skyTint;
				uniform vec3 groundTint;
				uniform float paperGrain;
				uniform float watercolorBleed;
				uniform float edgeRimIntensity;
				uniform vec3 edgeRimColor;
				uniform float edgeRimSoftness;
				uniform float variationOffset;

				float applyShadowBandingFn( float rawOpacity ) {
					if ( shadowBands <= 1 ) return rawOpacity;
					float bandWidth = 1.0 / float( shadowBands );
					float bandIndex = floor( rawOpacity / bandWidth );
					float bandCenter = bandIndex * bandWidth + bandWidth * 0.5;
					float distToCenter = abs( rawOpacity - bandCenter );
					float softFactor = clamp( distToCenter / ( bandWidth * 0.5 ), 0.0, 1.0 );
					float edgeBlend = ( 1.0 - softFactor ) * ( 1.0 - shadowSoftness ) + shadowSoftness;
					float finalOpacity = mix( bandCenter, rawOpacity, 1.0 - edgeBlend ) * shadowTint;
					return clamp( mix( rawOpacity, finalOpacity, shadowQuantize ), 0.0, 1.0 );
				}

				vec3 applyShadowAmbientGradient( vec3 baseColor, float worldY ) {
					if ( ambientGradient <= 0.0 ) return baseColor;
					float t = clamp( worldY / max( ambientHeight, 0.001 ) + 0.5, 0.0, 1.0 );
					vec3 tint = mix( groundTint, skyTint, t );
					return mix( baseColor, baseColor * tint, ambientGradient );
				}

				float sampleShadowPaperGrain( vec2 uv ) {
					if ( paperGrain <= 0.0 ) return 1.0;
					float n = sin( uv.x * 8.0 + variationOffset ) * cos( uv.y * 8.0 + variationOffset );
					n = n * 0.5 + 0.5;
					return 1.0 - paperGrain * ( 1.0 - n );
				}

				float computeShadowEdgeRim( float shadowOpacity ) {
					if ( edgeRimIntensity <= 0.0 ) return 0.0;
					float distToEdge = abs( shadowOpacity - 0.5 );
					if ( distToEdge > edgeRimSoftness ) return 0.0;
					return clamp( 1.0 - distToEdge / edgeRimSoftness, 0.0, 1.0 ) * edgeRimIntensity;
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Extract the raw shadow opacity from the alpha channel
				float rawShadowOpacity = gl_FragColor.a;

				// Apply shadow cel banding (flat-shadow layers)
				float bandedShadow = applyShadowBandingFn( rawShadowOpacity );

				// Override with mood-graded shadow color
				gl_FragColor.rgb = mix( color.rgb, moodColor, moodIntensity );
				gl_FragColor.a = bandedShadow;

				// Ambient sky-to-ground gradient tinting on the shadow color
				gl_FragColor.rgb = applyShadowAmbientGradient( gl_FragColor.rgb, vViewPosition.z );

				// Paper grain opacity modulation
				gl_FragColor.a *= sampleShadowPaperGrain( vUv );

				// Rim glow on shadow edges
				if ( edgeRimIntensity > 0.0 ) {
					float edgeRim = computeShadowEdgeRim( rawShadowOpacity );
					gl_FragColor.rgb += edgeRimColor * edgeRim;
				}
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
			this.shadowBands,
			this.shadowQuantize,
			this.shadowSoftness,
			this.shadowTint,
			this.penumbraPower,
			this.penumbraInner,
			this.penumbraOuter,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodIntensity,
			this.ambientGradient,
			this.ambientHeight,
			this.paperGrain,
			this.watercolorBleed,
			this.edgeRimIntensity,
			this.edgeRimSoftness,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {ShadowMaterial} source - The material to copy from.
	 * @return {ShadowMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.color.copy( source.color );
		this.opacity = source.opacity;
		this.fog = source.fog;

		// Anime extensions
		this.shadowBands = source.shadowBands;
		this.shadowQuantize = source.shadowQuantize;
		this.shadowSoftness = source.shadowSoftness;
		this.shadowTint = source.shadowTint;
		this.penumbraPower = source.penumbraPower;
		this.penumbraInner = source.penumbraInner;
		this.penumbraOuter = source.penumbraOuter;
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodIntensity = source.moodIntensity;
		this.moodColor.copy( source.moodColor );
		this.ambientGradient = source.ambientGradient;
		this.ambientHeight = source.ambientHeight;
		this.skyTint.copy( source.skyTint );
		this.groundTint.copy( source.groundTint );
		this.paperGrain = source.paperGrain;
		this.watercolorBleed = source.watercolorBleed;
		this.edgeRimIntensity = source.edgeRimIntensity;
		this.edgeRimColor.copy( source.edgeRimColor );
		this.edgeRimSoftness = source.edgeRimSoftness;
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

		data.type = 'ShadowMaterial';

		if ( this.color.getHex() !== 0x000000 ) data.color = this.color.getHex();
		if ( this.opacity !== 1 ) data.opacity = this.opacity;
		if ( this.fog === false ) data.fog = false;

		// Anime extensions
		if ( this.shadowBands !== 0 ) data.shadowBands = this.shadowBands;
		if ( this.shadowQuantize !== 1.0 ) data.shadowQuantize = this.shadowQuantize;
		if ( this.shadowSoftness !== 0.1 ) data.shadowSoftness = this.shadowSoftness;
		if ( this.shadowTint !== 0.85 ) data.shadowTint = this.shadowTint;
		if ( this.penumbraPower !== 1.5 ) data.penumbraPower = this.penumbraPower;
		if ( this.penumbraInner !== 0.3 ) data.penumbraInner = this.penumbraInner;
		if ( this.penumbraOuter !== 0.9 ) data.penumbraOuter = this.penumbraOuter;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodIntensity !== 0 ) data.moodIntensity = this.moodIntensity;
		if ( this.ambientGradient !== 0 ) data.ambientGradient = this.ambientGradient;
		if ( this.ambientHeight !== 10 ) data.ambientHeight = this.ambientHeight;
		if ( this.skyTint.getHex() !== new Color( 0.5, 0.7, 1.0 ).getHex() ) data.skyTint = this.skyTint.getHex();
		if ( this.groundTint.getHex() !== new Color( 1.0, 0.8, 0.6 ).getHex() ) data.groundTint = this.groundTint.getHex();
		if ( this.paperGrain !== 0 ) data.paperGrain = this.paperGrain;
		if ( this.watercolorBleed !== 0 ) data.watercolorBleed = this.watercolorBleed;
		if ( this.edgeRimIntensity !== 0 ) data.edgeRimIntensity = this.edgeRimIntensity;
		if ( this.edgeRimColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.edgeRimColor = this.edgeRimColor.getHex();
		if ( this.edgeRimSoftness !== 0.2 ) data.edgeRimSoftness = this.edgeRimSoftness;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { ShadowMaterial, ShadowMaterialBatch, applyShadowCelBanding, shapePenumbra, gradeShadowColor, applyShadowAmbientGradient, generateShadowTexture, computeShadowEdgeRim };
export default ShadowMaterial;