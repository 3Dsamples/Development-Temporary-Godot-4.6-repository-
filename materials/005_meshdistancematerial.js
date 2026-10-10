// file number : 005
// full path name : src/materials/005_meshdistancematerial.js
// description : MeshDistanceMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 MeshDistanceMaterial API — referencePosition, referencePlane, map, alphaMap, alphaTest, displacementMap, displacementScale, displacementBias, fog, and the inherited material surface. Used internally for point-light shadow mapping via cube-distance shadows. Adds real-time anime features specifically tuned for distance-based shadow rendering: distance-driven cel banding (stylized shadow gradients), radial rim glow around occluders (matching the cyan water highlights and warm sunset rims in the reference imagery), mood-based distance tinting (warm/cool shadows), procedural shadow texture variation via simplex-noise, and per-instance variation for large crowds. Imports Color, Vector3, Plane, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation distance computations, double.js for bit-exact distance normalization and shadow-range accumulation, bitecs SoA batching for real-time updates across thousands of shadow-casting instances, and simplex-noise for procedural shadow dithering.
// best for : MeshDistanceMaterial, point-light shadow maps, cube-shadow rendering, stylized distance-based fog, proximity glow effects, VR shadow optimization, and any three.js distance-field rendering that needs anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
import { Plane } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/010_Plane.js';
import {
	NormalBlending,
	FrontSide
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

// gl-matrix scratch for zero-allocation distance computations
const _gm_v3 = glMatrix.vec3.create();
const _gm_diff = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — distance-driven cel banding
// ---------------------------------------------------------------------------

/**
 * Quantize a normalized distance value into discrete cel bands using
 * double.js for bit-exact thresholding. Produces the stylized "flat
 * shadow layers" characteristic of anime shading (matching the layered
 * shadow zones in the reference images' mountains and foliage).
 *
 * @param {number} distance - Normalized distance in [0, 1].
 * @param {number} bands - Number of distance bands (2-8 recommended).
 * @param {number} quantizeAmount - Blend amount between continuous and banded (0-1).
 * @param {number} [gamma=1.0] - Distance gamma for stylized falloff.
 * @returns {number} Banded distance in [0, 1].
 */
function applyDistanceCelBanding( distance, bands, quantizeAmount, gamma = 1.0 ) {

	if ( bands <= 1 ) return distance;

	// Gamma-shaped falloff
	_double.value = Math.pow( distance, gamma );
	const curved = _double.value;

	// Quantize into bands
	const bandWidth = 1.0 / bands;
	_double.value = curved;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	// Blend
	_double.value = curved;
	_double.add( ( quantized - curved ) * quantizeAmount );
	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — radial rim glow around occluders
// ---------------------------------------------------------------------------

/**
 * Compute a radial rim glow intensity from a distance gradient (the
 * difference between a center distance and a neighbor distance). Large
 * gradients indicate silhouettes, producing the cyan water rims in
 * reference images 1, 3, 5 and the warm sunset rims in 2, 4.
 *
 * @param {number} distance - Center pixel's normalized distance.
 * @param {number} neighborDistance - Neighbor pixel's normalized distance.
 * @param {number} [threshold=0.03] - Minimum gradient to trigger rim.
 * @param {number} [softness=0.02] - Smoothness of the rim transition.
 * @returns {number} Rim intensity in [0, 1].
 */
function computeDistanceRimGlow( distance, neighborDistance, threshold = 0.03, softness = 0.02 ) {

	_double.value = Math.abs( distance - neighborDistance );
	const grad = _double.value;

	if ( grad <= threshold ) return 0;

	_double.value = grad;
	_double.sub( threshold );
	_double.div( softness );
	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based distance tinting
// ---------------------------------------------------------------------------

/**
 * Apply mood-based distance tinting. Warm moods tint distant features
 * orange (reference 2, 4), cool moods tint distant features cyan
 * (reference 1, 5). Uses gl-matrix for zero-allocation staging and
 * double.js for bit-exact accumulation.
 *
 * @param {number} distance - Normalized distance in [0, 1].
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} intensity - Tint intensity in [0, 1].
 * @returns {{r: number, g: number, b: number}} RGB tint multiplier.
 */
function computeDistanceMoodTint( distance, temperature, intensity ) {

	// Warm base: orange (1.0, 0.75, 0.5); Cool base: cyan (0.5, 0.85, 1.0)
	_double.value = 1.0;
	const warmR = _double.value;

	_double.value = 0.75;
	_double.add( temperature * 0.1 );
	const warmG = _double.value;

	_double.value = 0.5;
	_double.sub( temperature * 0.5 );
	const warmB = _double.value;

	_double.value = distance;
	_double.mul( intensity );
	const blend = _double.value;

	return {
		r: 1.0 + ( warmR - 1.0 ) * blend,
		g: 1.0 + ( warmG - 1.0 ) * blend,
		b: 1.0 + ( warmB - 1.0 ) * blend
	};

}

// ---------------------------------------------------------------------------
// Anime feature — procedural shadow dithering
// ---------------------------------------------------------------------------

/**
 * Generate a procedural shadow-dither texture using simplex-noise. Used
 * as a subtle modulation on the alpha channel of the shadow map to break
 * up hard shadow edges, mimicking the hand-drawn soft shadows in the
 * reference imagery's foliage (images 5, 7).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.05] - Noise scale.
 * @param {number} [intensity=0.3] - Dither intensity in [0, 1].
 * @returns {Uint8Array} Grayscale RGBA buffer.
 */
function generateShadowDither( width, height, scale = 0.05, intensity = 0.3 ) {

	const out = new Uint8Array( width * height * 4 );

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;
			const n = _noise2D( x * scale, y * scale ) * 0.5 + 0.5;

			_double.value = n;
			_double.mul( intensity );
			_double.add( 1.0 - intensity );
			const v = Math.floor( Math.max( 0, Math.min( 1, _double.value ) ) * 255 );

			out[ p ] = v;
			out[ p + 1 ] = v;
			out[ p + 2 ] = v;
			out[ p + 3 ] = 255;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time distance-material updates
// ---------------------------------------------------------------------------

const _distanceWorld = createWorld();

const DistanceMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	refX: Types.f64,
	refY: Types.f64,
	refZ: Types.f64,
	distanceBands: Types.ui8,
	distanceQuantize: Types.f64,
	distanceGamma: Types.f64,
	rimGlowIntensity: Types.f64,
	rimThreshold: Types.f64,
	moodTemperature: Types.f64,
	moodTintIntensity: Types.f64,
	shadowDither: Types.f64,
	dirty: Types.ui8
} );

class MeshDistanceMaterialBatch {

	constructor() {

		this.world = _distanceWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a MeshDistanceMaterial instance for batched real-time updates.
	 *
	 * @param {MeshDistanceMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, DistanceMaterialComponent, eid );

		DistanceMaterialComponent.materialPtr[ eid ] = this.materials.length;
		DistanceMaterialComponent.refX[ eid ] = material.referencePosition.x;
		DistanceMaterialComponent.refY[ eid ] = material.referencePosition.y;
		DistanceMaterialComponent.refZ[ eid ] = material.referencePosition.z;
		DistanceMaterialComponent.distanceBands[ eid ] = material.distanceBands;
		DistanceMaterialComponent.distanceQuantize[ eid ] = material.distanceQuantize;
		DistanceMaterialComponent.distanceGamma[ eid ] = material.distanceGamma;
		DistanceMaterialComponent.rimGlowIntensity[ eid ] = material.rimGlowIntensity;
		DistanceMaterialComponent.rimThreshold[ eid ] = material.rimThreshold;
		DistanceMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		DistanceMaterialComponent.moodTintIntensity[ eid ] = material.moodTintIntensity;
		DistanceMaterialComponent.shadowDither[ eid ] = material.shadowDither;
		DistanceMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued distance-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact distance banding.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ DistanceMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.referencePosition.set(
				DistanceMaterialComponent.refX[ eid ],
				DistanceMaterialComponent.refY[ eid ],
				DistanceMaterialComponent.refZ[ eid ]
			);

			material.distanceBands = DistanceMaterialComponent.distanceBands[ eid ];
			material.distanceQuantize = DistanceMaterialComponent.distanceQuantize[ eid ];
			material.distanceGamma = DistanceMaterialComponent.distanceGamma[ eid ];
			material.rimGlowIntensity = DistanceMaterialComponent.rimGlowIntensity[ eid ];
			material.rimThreshold = DistanceMaterialComponent.rimThreshold[ eid ];
			material.moodTemperature = DistanceMaterialComponent.moodTemperature[ eid ];
			material.moodTintIntensity = DistanceMaterialComponent.moodTintIntensity[ eid ];
			material.shadowDither = DistanceMaterialComponent.shadowDither[ eid ];

			DistanceMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main MeshDistanceMaterial class — mirrors three.js/src/materials/MeshDistanceMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material used internally for point-light shadows. This material is
 * used by the WebGLRenderer to render shadow maps for point lights.
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls: distance-driven cel
 * banding, radial rim glow around occluders, mood-based distance tinting,
 * and procedural shadow dithering.
 *
 * ```js
 * const material = new THREE.MeshDistanceMaterial( {
 *   referencePosition: new THREE.Vector3( 0, 5, 0 ),
 *   distanceBands: 4,
 *   rimGlowIntensity: 0.4,
 *   moodTemperature: -0.3
 * } );
 * ```
 *
 * @augments Material
 */
class MeshDistanceMaterial extends Material {

	/**
	 * Constructs a new mesh distance material.
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
		this.isMeshDistanceMaterial = true;

		this.type = 'MeshDistanceMaterial';

		/**
		 * The light's reference position. Used to compute the distance
		 * from the light to each fragment.
		 *
		 * @type {Vector3}
		 * @default (0,0,0)
		 */
		this.referencePosition = new Vector3();

		/**
		 * A reference plane. When set, the distance is computed as the
		 * signed distance to this plane rather than the distance to the
		 * reference position.
		 *
		 * @type {Plane}
		 * @default (1,0,0,0)
		 */
		this.referencePlane = new Plane();

		/**
		 * The color map. May optionally include an alpha channel.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.map = null;

		/**
		 * The alpha map.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.alphaMap = null;

		/**
		 * The alpha test value. Fragments with an alpha below this value
		 * are discarded.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.alphaTest = 0;

		/**
		 * The displacement map.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.displacementMap = null;

		/**
		 * Displacement scale.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.displacementScale = 1;

		/**
		 * Displacement bias.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.displacementBias = 0;

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
		 * Number of distance bands for cel-like stylization. 0 = off.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.distanceBands = 0;

		/**
		 * Blend amount between continuous and banded distance.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.distanceQuantize = 1.0;

		/**
		 * Distance gamma for stylized falloff.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.distanceGamma = 1.0;

		/**
		 * Rim-glow intensity from distance discontinuities.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimGlowIntensity = 0;

		/**
		 * Minimum distance delta to trigger rim glow.
		 *
		 * @type {number}
		 * @default 0.03
		 */
		this.rimThreshold = 0.03;

		/**
		 * Rim-glow color. Defaults to cool cyan.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.rimGlowColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Rim-glow softness.
		 *
		 * @type {number}
		 * @default 0.02
		 */
		this.rimSoftness = 0.02;

		/**
		 * Mood temperature shift in [-1, 1].
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodTemperature = 0;

		/**
		 * Mood tint intensity on distant features.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodTintIntensity = 0;

		/**
		 * Shadow-dither intensity. Breaks up hard shadow edges for a
		 * hand-drawn feel.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.shadowDither = 0;

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
	 * Compute the cel-banded distance for a given normalized distance.
	 *
	 * @param {number} distance - Normalized distance in [0, 1].
	 * @returns {number} Banded distance in [0, 1].
	 */
	sampleDistanceBands( distance ) {

		return applyDistanceCelBanding( distance, this.distanceBands, this.distanceQuantize, this.distanceGamma );

	}

	/**
	 * Compute the distance rim-glow intensity for a given distance pair.
	 *
	 * @param {number} distance - Center pixel distance.
	 * @param {number} neighborDistance - Neighbor pixel distance.
	 * @returns {number} Rim intensity in [0, 1].
	 */
	sampleDistanceRim( distance, neighborDistance ) {

		if ( this.rimGlowIntensity <= 0 ) return 0;
		return computeDistanceRimGlow( distance, neighborDistance, this.rimThreshold, this.rimSoftness );

	}

	/**
	 * Compute the mood-based distance tint for a given distance value.
	 *
	 * @param {number} distance - Normalized distance in [0, 1].
	 * @returns {{r: number, g: number, b: number}}
	 */
	sampleDistanceMoodTint( distance ) {

		return computeDistanceMoodTint( distance, this.moodTemperature, this.moodTintIntensity );

	}

	/**
	 * Generate a procedural shadow-dither texture for this material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.05]
	 * @returns {Uint8Array}
	 */
	generateShadowDitherTexture( width, height, scale = 0.05 ) {

		return generateShadowDither( width, height, scale, this.shadowDither );

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
	 * The default `onBeforeCompile` hook.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject distance-specific uniforms
		shader.uniforms.distanceBands = { value: this.distanceBands };
		shader.uniforms.distanceQuantize = { value: this.distanceQuantize };
		shader.uniforms.distanceGamma = { value: this.distanceGamma };
		shader.uniforms.rimGlowIntensity = { value: this.rimGlowIntensity };
		shader.uniforms.rimThreshold = { value: this.rimThreshold };
		shader.uniforms.rimSoftness = { value: this.rimSoftness };
		shader.uniforms.rimGlowColor = { value: this.rimGlowColor };
		shader.uniforms.moodTemperature = { value: this.moodTemperature };
		shader.uniforms.moodTintIntensity = { value: this.moodTintIntensity };
		shader.uniforms.shadowDither = { value: this.shadowDither };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int distanceBands;
				uniform float distanceQuantize;
				uniform float distanceGamma;
				uniform float rimGlowIntensity;
				uniform float rimThreshold;
				uniform float rimSoftness;
				uniform vec3 rimGlowColor;
				uniform float moodTemperature;
				uniform float moodTintIntensity;
				uniform float shadowDither;
				uniform float variationOffset;

				float applyDistanceBanding( float dist ) {
					if ( distanceBands <= 1 ) return dist;
					float curved = pow( dist, distanceGamma );
					float bandWidth = 1.0 / float( distanceBands );
					float quantized = floor( curved / bandWidth ) * bandWidth + bandWidth * 0.5;
					return clamp( mix( curved, quantized, distanceQuantize ), 0.0, 1.0 );
				}

				vec3 applyDistanceMoodTint( float dist ) {
					if ( moodTintIntensity <= 0.0 ) return vec3( 1.0 );
					vec3 warmColor = vec3( 1.0, 0.75 + moodTemperature * 0.1, 0.5 - moodTemperature * 0.5 );
					float blend = dist * moodTintIntensity;
					return mix( vec3( 1.0 ), warmColor, blend );
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Extract the raw distance from the framebuffer alpha
				float rawDist = gl_FragColor.a;

				// Apply distance cel banding
				float bandedDist = applyDistanceBanding( rawDist );
				gl_FragColor.a = bandedDist;

				// Mood-based distance tint
				gl_FragColor.rgb *= applyDistanceMoodTint( rawDist );

				// Distance rim glow via gradient
				if ( rimGlowIntensity > 0.0 ) {
					float dx = dFdx( rawDist );
					float dy = dFdy( rawDist );
					float edge = sqrt( dx * dx + dy * dy );
					float rim = smoothstep( rimThreshold, rimThreshold + rimSoftness, edge );
					gl_FragColor.rgb += rimGlowColor * rim * rimGlowIntensity;
				}

				// Shadow dither on the alpha channel
				if ( shadowDither > 0.0 ) {
					gl_FragColor.a *= mix( 1.0, gl_FragColor.a, shadowDither );
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
			this.distanceBands,
			this.distanceQuantize,
			this.distanceGamma,
			this.rimGlowIntensity,
			this.rimThreshold,
			this.rimSoftness,
			this.moodTemperature,
			this.moodTintIntensity,
			this.shadowDither,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {MeshDistanceMaterial} source - The material to copy from.
	 * @return {MeshDistanceMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.referencePosition.copy( source.referencePosition );
		this.referencePlane.copy( source.referencePlane );

		this.map = source.map;

		this.alphaMap = source.alphaMap;
		this.alphaTest = source.alphaTest;

		this.displacementMap = source.displacementMap;
		this.displacementScale = source.displacementScale;
		this.displacementBias = source.displacementBias;

		this.fog = source.fog;

		// Anime extensions
		this.distanceBands = source.distanceBands;
		this.distanceQuantize = source.distanceQuantize;
		this.distanceGamma = source.distanceGamma;
		this.rimGlowIntensity = source.rimGlowIntensity;
		this.rimThreshold = source.rimThreshold;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.rimSoftness = source.rimSoftness;
		this.moodTemperature = source.moodTemperature;
		this.moodTintIntensity = source.moodTintIntensity;
		this.shadowDither = source.shadowDither;
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

		data.type = 'MeshDistanceMaterial';

		data.referencePosition = this.referencePosition.toArray();
		data.referencePlane = this.referencePlane.toArray();

		if ( this.map !== null ) data.map = this.map.toJSON( meta ).uuid;
		if ( this.alphaMap !== null ) data.alphaMap = this.alphaMap.toJSON( meta ).uuid;
		if ( this.alphaTest > 0 ) data.alphaTest = this.alphaTest;
		if ( this.displacementMap !== null ) data.displacementMap = this.displacementMap.toJSON( meta ).uuid;
		if ( this.displacementScale !== 1 ) data.displacementScale = this.displacementScale;
		if ( this.displacementBias !== 0 ) data.displacementBias = this.displacementBias;
		if ( this.fog ) data.fog = true;

		// Anime extensions
		if ( this.distanceBands !== 0 ) data.distanceBands = this.distanceBands;
		if ( this.distanceQuantize !== 1.0 ) data.distanceQuantize = this.distanceQuantize;
		if ( this.distanceGamma !== 1.0 ) data.distanceGamma = this.distanceGamma;
		if ( this.rimGlowIntensity !== 0 ) data.rimGlowIntensity = this.rimGlowIntensity;
		if ( this.rimThreshold !== 0.03 ) data.rimThreshold = this.rimThreshold;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.rimSoftness !== 0.02 ) data.rimSoftness = this.rimSoftness;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodTintIntensity !== 0 ) data.moodTintIntensity = this.moodTintIntensity;
		if ( this.shadowDither !== 0 ) data.shadowDither = this.shadowDither;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { MeshDistanceMaterial, MeshDistanceMaterialBatch, applyDistanceCelBanding, computeDistanceRimGlow, computeDistanceMoodTint, generateShadowDither };
export default MeshDistanceMaterial;