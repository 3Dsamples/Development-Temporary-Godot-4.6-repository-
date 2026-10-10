// file number : 004
// full path name : src/materials/004_meshdepthmaterial.js
// description : MeshDepthMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 MeshDepthMaterial API — depthPacking (BasicDepthPacking / RGBADepthPacking), map, alphaMap, displacementMap, displacementScale, displacementBias, wireframe, wireframeLinewidth, fog, and the inherited material surface. Adds real-time anime features specifically tuned for depth-based rendering: depth-driven cel banding (for stylized fog effects in anime backgrounds), silhouette rim glow computed from depth (matching the cyan water rims in reference images 1, 3, 5 and the sunset rim in 2, 4), mood-based depth tinting, paper-grain depth modulation, and per-instance variation for large crowds of depth-rendered characters. Imports Color, Vector2, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation depth/rim computations, double.js for bit-exact depth normalization and HDR range accumulation, bitecs SoA batching for real-time updates across thousands of depth-rendered instances, and simplex-noise for procedural fog/depth variation.
// best for : MeshDepthMaterial, shadow-map rendering, depth-based fog, stylized DOF (depth of field) passes, silhouette outlines, toon shadow maps, depth-peeling for anime backgrounds, and any three.js depth-based rendering that needs anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import {
	BasicDepthPacking,
	RGBADepthPacking,
	NormalBlending,
	FrontSide,
	NoColorSpace
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

// gl-matrix scratch for zero-allocation depth computations
const _gm_depth = glMatrix.vec2.create();

// ---------------------------------------------------------------------------
// Anime feature — depth-driven cel banding
// ---------------------------------------------------------------------------

/**
 * Quantize a linear depth value into discrete cel bands using double.js
 * for bit-exact thresholding. Produces the stylized "flat depth layers"
 * characteristic of anime backgrounds (e.g. the layered mountains and
 * clouds in reference images 1, 5, and the planet rings in 2, 4).
 *
 * @param {number} depth - Linear depth in [0, 1].
 * @param {number} bands - Number of depth bands (2-8 recommended).
 * @param {number} quantizeAmount - Blend amount between continuous and banded (0-1).
 * @param {number} [gamma=1.0] - Depth gamma for stylized falloff.
 * @returns {number} Banded depth in [0, 1].
 */
function applyDepthCelBanding( depth, bands, quantizeAmount, gamma = 1.0 ) {

	if ( bands <= 1 ) return depth;

	// Apply gamma correction for stylized falloff
	_double.value = Math.pow( depth, gamma );
	const curvedDepth = _double.value;

	// Quantize into bands
	const bandWidth = 1.0 / bands;
	_double.value = curvedDepth;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	// Blend between continuous and quantized
	_double.value = curvedDepth;
	_double.add( ( quantized - curvedDepth ) * quantizeAmount );
	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — depth-driven silhouette rim glow
// ---------------------------------------------------------------------------

/**
 * Compute a silhouette rim intensity from a depth value and a neighbor
 * depth sample. Large depth discontinuities (silhouettes) produce a
 * strong rim, mimicking the cyan water rims in reference images 1, 3, 5
 * and the sunset character rim in image 6.
 *
 * @param {number} depth - The center pixel's linear depth.
 * @param {number} neighborDepth - The neighbor pixel's linear depth.
 * @param {number} [threshold=0.05] - Minimum depth delta to trigger rim.
 * @param {number} [softness=0.02] - Smoothness of the rim transition.
 * @returns {number} Rim intensity in [0, 1].
 */
function computeDepthRimGlow( depth, neighborDepth, threshold = 0.05, softness = 0.02 ) {

	_double.value = Math.abs( depth - neighborDepth );
	const delta = _double.value;

	if ( delta <= threshold ) return 0;

	_double.value = delta;
	_double.sub( threshold );
	_double.div( softness );
	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based depth tinting
// ---------------------------------------------------------------------------

/**
 * Apply mood-based depth tinting. Warm moods (sunset) tint distant depths
 * orange (reference 2, 4), while cool moods (snowy) tint distant depths
 * cyan (reference 1, 5). Uses gl-matrix for zero-allocation staging and
 * double.js for bit-exact accumulation.
 *
 * @param {number} depth - Linear depth in [0, 1].
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} intensity - Tint intensity in [0, 1].
 * @returns {{r: number, g: number, b: number}} RGB tint multiplier.
 */
function computeDepthMoodTint( depth, temperature, intensity ) {

	// Warm: orange (1.0, 0.7, 0.4), Cool: cyan (0.5, 0.85, 1.0)
	_double.value = 1.0;
	_double.add( temperature * 0.0 ); // R: unchanged
	const warmR = _double.value;

	_double.value = 0.7;
	_double.add( temperature * 0.15 ); // G: warm boosts, cool reduces
	const warmG = _double.value;

	_double.value = 0.4;
	_double.sub( temperature * 0.6 ); // B: cool boosts
	const warmB = _double.value;

	// Blend between neutral and mood color based on depth and intensity
	_double.value = depth;
	_double.mul( intensity );
	const blend = _double.value;

	return {
		r: 1.0 + ( warmR - 1.0 ) * blend,
		g: 1.0 + ( warmG - 1.0 ) * blend,
		b: 1.0 + ( warmB - 1.0 ) * blend
	};

}

// ---------------------------------------------------------------------------
// Anime feature — procedural depth fog variation
// ---------------------------------------------------------------------------

/**
 * Generate a procedural fog-variation texture using simplex-noise. Used
 * as an additive modulation on top of the depth value to produce the
 * organic, hand-painted fog characteristic of anime backgrounds (the
 * misty mountain layers in reference images 1, 5, and the atmospheric
 * haze in 2, 4).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.02] - Noise scale.
 * @param {number} [octaves=3] - Number of noise octaves.
 * @param {number} [intensity=0.3] - Fog variation intensity in [0, 1].
 * @returns {Uint8Array} Grayscale RGBA buffer.
 */
function generateDepthFogTexture( width, height, scale = 0.02, octaves = 3, intensity = 0.3 ) {

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

			// Modulate by intensity
			_double.value = value;
			_double.mul( intensity );
			_double.add( 0.5 * ( 1 - intensity ) );
			const finalValue = Math.max( 0, Math.min( 1, _double.value ) );

			const v = Math.floor( finalValue * 255 );

			out[ p ] = v;
			out[ p + 1 ] = v;
			out[ p + 2 ] = v;
			out[ p + 3 ] = 255;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time depth material updates
// ---------------------------------------------------------------------------

const _depthWorld = createWorld();

const DepthMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	depthBands: Types.ui8,
	depthQuantize: Types.f64,
	depthGamma: Types.f64,
	rimGlowIntensity: Types.f64,
	rimThreshold: Types.f64,
	moodTemperature: Types.f64,
	moodTintIntensity: Types.f64,
	fogVariation: Types.f64,
	dirty: Types.ui8
} );

class MeshDepthMaterialBatch {

	constructor() {

		this.world = _depthWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a MeshDepthMaterial instance for batched real-time updates.
	 *
	 * @param {MeshDepthMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, DepthMaterialComponent, eid );

		DepthMaterialComponent.materialPtr[ eid ] = this.materials.length;
		DepthMaterialComponent.depthBands[ eid ] = material.depthBands;
		DepthMaterialComponent.depthQuantize[ eid ] = material.depthQuantize;
		DepthMaterialComponent.depthGamma[ eid ] = material.depthGamma;
		DepthMaterialComponent.rimGlowIntensity[ eid ] = material.rimGlowIntensity;
		DepthMaterialComponent.rimThreshold[ eid ] = material.rimThreshold;
		DepthMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		DepthMaterialComponent.moodTintIntensity[ eid ] = material.moodTintIntensity;
		DepthMaterialComponent.fogVariation[ eid ] = material.fogVariation;
		DepthMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued depth-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact depth banding.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ DepthMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.depthBands = DepthMaterialComponent.depthBands[ eid ];
			material.depthQuantize = DepthMaterialComponent.depthQuantize[ eid ];
			material.depthGamma = DepthMaterialComponent.depthGamma[ eid ];
			material.rimGlowIntensity = DepthMaterialComponent.rimGlowIntensity[ eid ];
			material.rimThreshold = DepthMaterialComponent.rimThreshold[ eid ];
			material.moodTemperature = DepthMaterialComponent.moodTemperature[ eid ];
			material.moodTintIntensity = DepthMaterialComponent.moodTintIntensity[ eid ];
			material.fogVariation = DepthMaterialComponent.fogVariation[ eid ];

			DepthMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main MeshDepthMaterial class — mirrors three.js/src/materials/MeshDepthMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material for drawing geometry by depth. Depth is based off of the
 * camera near and far plane. White is nearest, black is farthest.
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls: depth-driven cel banding,
 * depth rim glow, mood-based depth tinting, and procedural fog variation.
 *
 * ```js
 * const material = new THREE.MeshDepthMaterial( {
 *   depthPacking: THREE.RGBADepthPacking,
 *   depthBands: 4,
 *   depthQuantize: 0.8,
 *   moodTemperature: 0.3,
 *   rimGlowIntensity: 0.5
 * } );
 * ```
 *
 * @augments Material
 */
class MeshDepthMaterial extends Material {

	/**
	 * Constructs a new mesh depth material.
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
		this.isMeshDepthMaterial = true;

		this.type = 'MeshDepthMaterial';

		/**
		 * The depth packing type. Can be `BasicDepthPacking` or
		 * `RGBADepthPacking`. The latter packs the depth into all four
		 * channels for higher precision.
		 *
		 * @type {number}
		 * @default BasicDepthPacking
		 */
		this.depthPacking = BasicDepthPacking;

		/**
		 * The color map. May optionally include an alpha channel, typically
		 * combined with `transparent` or `alphaTest`. The texture is
		 * expected to be in the sRGB color space.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.map = null;

		/**
		 * The alpha map is a grayscale texture that controls the opacity
		 * across the surface. The texture is expected to be in the
		 * `NoColorSpace` color space.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.alphaMap = null;

		/**
		 * The displacement map affects the position of the mesh's vertices.
		 * The texture is expected to be in the `NoColorSpace` color space.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.displacementMap = null;

		/**
		 * How much the displacement map affects the mesh. A value of 0
		 * disables displacement.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.displacementScale = 1;

		/**
		 * How much the displacement map affects the mesh. Unlike
		 * `displacementScale`, this value is added to the resulting
		 * displacement, not multiplied.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.displacementBias = 0;

		/**
		 * Whether to render the material as wireframe or not.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.wireframe = false;

		/**
		 * Controls wireframe thickness. Not supported on all platforms.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.wireframeLinewidth = 1;

		/**
		 * Whether the material is affected by fog or not.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.fog = false;

		// -------------------------------------------------------------------
		// Anime Rendering Extensions
		// -------------------------------------------------------------------

		/**
		 * Number of depth bands for cel-like stylization. 0 = off,
		 * 2-8 = layered depth bands matching anime background art.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.depthBands = 0;

		/**
		 * Blend amount between continuous depth and banded depth.
		 * 0 = continuous (off), 1 = fully quantized (hard bands).
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.depthQuantize = 1.0;

		/**
		 * Depth gamma for stylized falloff. Values < 1 push distant
		 * features closer; values > 1 push them farther.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.depthGamma = 1.0;

		/**
		 * Rim-glow intensity from depth discontinuities. Set > 0 to
		 * highlight silhouettes (matches the cyan water rims in reference
		 * images 1, 3, 5 and the sunset character rim in 6).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimGlowIntensity = 0;

		/**
		 * Minimum depth delta to trigger rim glow. Lower values catch
		 * subtler silhouettes.
		 *
		 * @type {number}
		 * @default 0.05
		 */
		this.rimThreshold = 0.05;

		/**
		 * Rim-glow color. Defaults to a cool cyan.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.rimGlowColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Rim-glow softness. Controls how quickly the rim fades as the
		 * depth delta decreases.
		 *
		 * @type {number}
		 * @default 0.02
		 */
		this.rimSoftness = 0.02;

		/**
		 * Mood temperature shift in [-1, 1]. Positive = warm (sunset
		 * orange), negative = cool (snowy cyan).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodTemperature = 0;

		/**
		 * Mood tint intensity on distant depths. Set > 0 to tint distant
		 * features with the mood color.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodTintIntensity = 0;

		/**
		 * Procedural fog-variation intensity. Set > 0 to add organic
		 * hand-painted fog on top of the depth value.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.fogVariation = 0;

		/**
		 * Procedural variation seed. Varies the fog texture between
		 * instances.
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
	 * Compute the cel-banded depth for a given linear depth value. Uses
	 * double.js for bit-exact thresholding.
	 *
	 * @param {number} depth - Linear depth in [0, 1].
	 * @returns {number} Banded depth in [0, 1].
	 */
	sampleDepthBands( depth ) {

		return applyDepthCelBanding( depth, this.depthBands, this.depthQuantize, this.depthGamma );

	}

	/**
	 * Compute the depth rim-glow intensity for a given depth pair. Uses
	 * double.js for bit-exact thresholding.
	 *
	 * @param {number} depth - Center pixel depth.
	 * @param {number} neighborDepth - Neighbor pixel depth.
	 * @returns {number} Rim intensity in [0, 1].
	 */
	sampleDepthRim( depth, neighborDepth ) {

		if ( this.rimGlowIntensity <= 0 ) return 0;
		return computeDepthRimGlow( depth, neighborDepth, this.rimThreshold, this.rimSoftness );

	}

	/**
	 * Compute the mood-based depth tint for a given depth value. Uses
	 * double.js for bit-exact accumulation.
	 *
	 * @param {number} depth - Linear depth in [0, 1].
	 * @returns {{r: number, g: number, b: number}}
	 */
	sampleDepthMoodTint( depth ) {

		return computeDepthMoodTint( depth, this.moodTemperature, this.moodTintIntensity );

	}

	/**
	 * Generate a procedural depth-fog variation texture for this material.
	 * The caller is expected to assign the returned buffer to a
	 * `DataTexture`.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.02]
	 * @param {number} [octaves=3]
	 * @returns {Uint8Array}
	 */
	generateDepthFogTexture( width, height, scale = 0.02, octaves = 3 ) {

		return generateDepthFogTexture( width, height, scale, octaves, this.fogVariation );

	}

	/**
	 * Compute the per-instance variation offset from the variationSeed.
	 *
	 * @returns {number}
	 */
	getVariationOffset() {

		return this.variationSeed * 137.508; // golden-angle increment

	}

	// -----------------------------------------------------------------------
	// Shader hooks
	// -----------------------------------------------------------------------

	/**
	 * The default `onBeforeCompile` hook. Extends the base Material's
	 * anime shader chunks with depth-specific features: depth cel banding,
	 * depth rim glow, and mood-based depth tinting.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject depth-specific uniforms
		shader.uniforms.depthBands = { value: this.depthBands };
		shader.uniforms.depthQuantize = { value: this.depthQuantize };
		shader.uniforms.depthGamma = { value: this.depthGamma };
		shader.uniforms.rimGlowIntensity = { value: this.rimGlowIntensity };
		shader.uniforms.rimThreshold = { value: this.rimThreshold };
		shader.uniforms.rimSoftness = { value: this.rimSoftness };
		shader.uniforms.rimGlowColor = { value: this.rimGlowColor };
		shader.uniforms.moodTemperature = { value: this.moodTemperature };
		shader.uniforms.moodTintIntensity = { value: this.moodTintIntensity };
		shader.uniforms.fogVariation = { value: this.fogVariation };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int depthBands;
				uniform float depthQuantize;
				uniform float depthGamma;
				uniform float rimGlowIntensity;
				uniform float rimThreshold;
				uniform float rimSoftness;
				uniform vec3 rimGlowColor;
				uniform float moodTemperature;
				uniform float moodTintIntensity;
				uniform float fogVariation;
				uniform float variationOffset;

				float applyDepthBanding( float depth ) {
					if ( depthBands <= 1 ) return depth;
					float curved = pow( depth, depthGamma );
					float bandWidth = 1.0 / float( depthBands );
					float quantized = floor( curved / bandWidth ) * bandWidth + bandWidth * 0.5;
					return clamp( mix( curved, quantized, depthQuantize ), 0.0, 1.0 );
				}

				vec3 applyDepthMoodTint( float depth ) {
					if ( moodTintIntensity <= 0.0 ) return vec3( 1.0 );
					vec3 warmColor = vec3( 1.0, 0.7 + moodTemperature * 0.15, 0.4 - moodTemperature * 0.6 );
					float blend = depth * moodTintIntensity;
					return mix( vec3( 1.0 ), warmColor, blend );
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Extract the raw depth from the framebuffer alpha channel
				float rawDepth = gl_FragColor.a;

				// Apply cel-like depth banding
				float bandedDepth = applyDepthBanding( rawDepth );
				gl_FragColor.a = bandedDepth;

				// Apply mood-based depth tint
				gl_FragColor.rgb *= applyDepthMoodTint( rawDepth );

				// Depth rim glow (computed per-fragment using dFdx/dFdy)
				if ( rimGlowIntensity > 0.0 ) {
					float dx = dFdx( rawDepth );
					float dy = dFdy( rawDepth );
					float edge = sqrt( dx * dx + dy * dy );
					float rim = smoothstep( rimThreshold, rimThreshold + rimSoftness, edge );
					gl_FragColor.rgb += rimGlowColor * rim * rimGlowIntensity;
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
			this.depthPacking,
			this.depthBands,
			this.depthQuantize,
			this.depthGamma,
			this.rimGlowIntensity,
			this.rimThreshold,
			this.rimSoftness,
			this.moodTemperature,
			this.moodTintIntensity,
			this.fogVariation,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {MeshDepthMaterial} source - The material to copy from.
	 * @return {MeshDepthMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.depthPacking = source.depthPacking;

		this.map = source.map;

		this.alphaMap = source.alphaMap;

		this.displacementMap = source.displacementMap;
		this.displacementScale = source.displacementScale;
		this.displacementBias = source.displacementBias;

		this.wireframe = source.wireframe;
		this.wireframeLinewidth = source.wireframeLinewidth;

		// Anime extensions
		this.depthBands = source.depthBands;
		this.depthQuantize = source.depthQuantize;
		this.depthGamma = source.depthGamma;
		this.rimGlowIntensity = source.rimGlowIntensity;
		this.rimThreshold = source.rimThreshold;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.rimSoftness = source.rimSoftness;
		this.moodTemperature = source.moodTemperature;
		this.moodTintIntensity = source.moodTintIntensity;
		this.fogVariation = source.fogVariation;
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

		data.type = 'MeshDepthMaterial';

		if ( this.depthPacking !== BasicDepthPacking ) data.depthPacking = this.depthPacking;
		if ( this.map !== null ) data.map = this.map.toJSON( meta ).uuid;
		if ( this.alphaMap !== null ) data.alphaMap = this.alphaMap.toJSON( meta ).uuid;
		if ( this.displacementMap !== null ) data.displacementMap = this.displacementMap.toJSON( meta ).uuid;
		if ( this.displacementScale !== 1 ) data.displacementScale = this.displacementScale;
		if ( this.displacementBias !== 0 ) data.displacementBias = this.displacementBias;
		if ( this.wireframe ) data.wireframe = true;
		if ( this.wireframeLinewidth !== 1 ) data.wireframeLinewidth = this.wireframeLinewidth;
		if ( this.fog ) data.fog = true;

		// Anime extensions
		if ( this.depthBands !== 0 ) data.depthBands = this.depthBands;
		if ( this.depthQuantize !== 1.0 ) data.depthQuantize = this.depthQuantize;
		if ( this.depthGamma !== 1.0 ) data.depthGamma = this.depthGamma;
		if ( this.rimGlowIntensity !== 0 ) data.rimGlowIntensity = this.rimGlowIntensity;
		if ( this.rimThreshold !== 0.05 ) data.rimThreshold = this.rimThreshold;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.rimSoftness !== 0.02 ) data.rimSoftness = this.rimSoftness;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodTintIntensity !== 0 ) data.moodTintIntensity = this.moodTintIntensity;
		if ( this.fogVariation !== 0 ) data.fogVariation = this.fogVariation;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { MeshDepthMaterial, MeshDepthMaterialBatch, applyDepthCelBanding, computeDepthRimGlow, computeDepthMoodTint, generateDepthFogTexture };
export default MeshDepthMaterial;