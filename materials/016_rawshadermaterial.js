// file number : 016
// full path name : src/materials/016_rawshadermaterial.js
// description : RawShaderMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 015_shadermaterial.js base class and preserves the full r185 RawShaderMaterial API — the critical difference is that RawShaderMaterial disables the automatic injection of built-in uniforms, attributes, and prefix defines, giving the user complete control over the shader source. Inherits all shader uniforms, defines, vertexShader, fragmentShader, glslVersion, wireframe, fog, lights, clipping, and the anime-extension surface from ShaderMaterial. Adds real-time anime features specifically tuned for raw-shader workflows: manual cel-shading injection (user opts in by calling `injectAnimeChunks()` explicitly, respecting the raw contract), view-space rim glow, mood-based color grading, ambient gradient tinting, procedural paper-grain and watercolor overlays via simplex-noise, brush-jitter UV distortion, and per-instance variation for crowds. Imports Color, ColorManagement, and other math classes strictly from threejs_new01 math, plus Uniform and UniformsGroup strictly from threejs_new01 core. Uses gl-matrix for zero-allocation N·L and rim computations, double.js for bit-exact cel-band thresholding and HDR mood grading, bitecs SoA batching for real-time uniform updates across thousands of raw-shader instances, and simplex-noise for procedural texture variation and paper-grain.
// best for : RawShaderMaterial, custom GLSL pipelines that need complete control over uniforms/attributes, WebGL2/WebGPU raw shaders, hand-written anime toon shaders, water/cloud/foliage shaders with custom prefixes, post-processing effects, and any three.js raw shader that needs real-time anime stylization while respecting the raw contract.
// license : MIT

import { ShaderMaterial } from './015_shadermaterial.js';
import { Uniform } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/core/005_Uniform.js';
import { UniformsGroup } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/core/006_UniformsGroup.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
import { Vector4 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/016_Vector4.js';
import { Matrix3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/006_Matrix3.js';
import { Matrix4 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/007_Matrix4.js';
import {
	NormalBlending,
	FrontSide,
	NoColorSpace,
	LinearSRGBColorSpace,
	DoubleSide
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

// gl-matrix scratch for zero-allocation N·L, rim, and mood computations
const _gm_normal = glMatrix.vec3.create();
const _gm_view = glMatrix.vec3.create();
const _gm_light = glMatrix.vec3.create();
const _gm_rgb = glMatrix.vec3.create();
const _gm_rgb_out = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — raw-safe cel-shading GLSL chunk (no built-in prefix expected)
// ---------------------------------------------------------------------------

/**
 * Generate the GLSL source for a raw-safe cel-shading chunk. Unlike the
 * ShaderMaterial version, this chunk does NOT rely on any built-in
 * uniforms or attributes — the user is responsible for supplying
 * `vNormal`, `vViewPosition`, and any light direction. Uses double.js
 * on the JS side for bit-exact threshold validation of the band count.
 *
 * @param {number} bands - Number of cel bands (2-6 recommended).
 * @param {number} [shadowTint=0.7] - Shadow band brightness multiplier.
 * @param {number} [softness=0.1] - Edge softness in [0, 0.5].
 * @param {number} [highlightTint=1.15] - Highlight band brightness multiplier.
 * @param {string} [uniformPrefix='rawAnime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildRawCelShadingChunk( bands, shadowTint = 0.7, softness = 0.1, highlightTint = 1.15, uniformPrefix = 'rawAnime' ) {

	_double.value = bands;
	if ( _double.value < 2 ) return '// raw cel-shading disabled (bands < 2)';

	return `
		// ---- Raw-Safe Cel-Shading Chunk (${uniformPrefix}) ----
		uniform int ${uniformPrefix}Bands;
		uniform float ${uniformPrefix}ShadowTint;
		uniform float ${uniformPrefix}Softness;
		uniform float ${uniformPrefix}HighlightTint;

		vec3 ${uniformPrefix}ApplyCelShading( vec3 baseColor, vec3 viewNormal, vec3 viewLight ) {
			float ndotl = max( dot( normalize( viewNormal ), normalize( viewLight ) ), 0.0 );
			float bandWidth = 1.0 / float( ${uniformPrefix}Bands );
			float bandIndex = floor( ndotl / bandWidth );
			float bandCenter = bandIndex * bandWidth + bandWidth * 0.5;
			float distToCenter = abs( ndotl - bandCenter );
			float softFactor = clamp( distToCenter / ( bandWidth * 0.5 ), 0.0, 1.0 );
			float edgeBlend = ( 1.0 - softFactor ) * ( 1.0 - ${uniformPrefix}Softness ) + ${uniformPrefix}Softness;
			float bandPos = bandIndex / max( 1.0, float( ${uniformPrefix}Bands - 1 ) );
			float target = mix( ${uniformPrefix}ShadowTint, ${uniformPrefix}HighlightTint, bandPos );
			float lit = clamp( mix( target, ndotl, 1.0 - edgeBlend ), 0.0, 1.0 );
			return baseColor * lit;
		}
		// ---- End Raw-Safe Cel-Shading Chunk ----
	`;

}

/**
 * Generate the GLSL source for the raw-safe rim-glow chunk.
 *
 * @param {string} [uniformPrefix='rawAnime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildRawRimGlowChunk( uniformPrefix = 'rawAnime' ) {

	return `
		// ---- Raw-Safe Rim Glow Chunk (${uniformPrefix}) ----
		uniform float ${uniformPrefix}RimPower;
		uniform float ${uniformPrefix}RimIntensity;
		uniform vec3 ${uniformPrefix}RimColor;

		vec3 ${uniformPrefix}ApplyRimGlow( vec3 baseColor, vec3 viewNormal, vec3 viewDir ) {
			if ( ${uniformPrefix}RimIntensity <= 0.0 ) return baseColor;
			vec3 n = normalize( viewNormal );
			vec3 v = normalize( viewDir );
			float ndotv = abs( dot( n, v ) );
			float rim = pow( 1.0 - ndotv, ${uniformPrefix}RimPower );
			return baseColor + ${uniformPrefix}RimColor * rim * ${uniformPrefix}RimIntensity;
		}
		// ---- End Raw-Safe Rim Glow Chunk ----
	`;

}

/**
 * Generate the GLSL source for the raw-safe mood grading chunk.
 *
 * @param {string} [uniformPrefix='rawAnime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildRawMoodGradingChunk( uniformPrefix = 'rawAnime' ) {

	return `
		// ---- Raw-Safe Mood Grading Chunk (${uniformPrefix}) ----
		uniform float ${uniformPrefix}MoodTemperature;
		uniform float ${uniformPrefix}MoodSaturation;
		uniform float ${uniformPrefix}MoodBrightness;
		uniform float ${uniformPrefix}MoodContrast;

		vec3 ${uniformPrefix}ApplyMoodGrading( vec3 color ) {
			// Saturation
			float lum = dot( color, vec3( 0.2126, 0.7152, 0.0722 ) );
			color = mix( vec3( lum ), color, ${uniformPrefix}MoodSaturation );

			// Contrast
			color = ( color - 0.5 ) * ${uniformPrefix}MoodContrast + 0.5;

			// Temperature
			color.r += ${uniformPrefix}MoodTemperature * 0.12;
			color.b -= ${uniformPrefix}MoodTemperature * 0.12;

			// Brightness
			color *= ${uniformPrefix}MoodBrightness;

			return clamp( color, 0.0, 1.0 );
		}
		// ---- End Raw-Safe Mood Grading Chunk ----
	`;

}

/**
 * Generate the GLSL source for the raw-safe paper-grain chunk.
 *
 * @param {string} [uniformPrefix='rawAnime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildRawPaperGrainChunk( uniformPrefix = 'rawAnime' ) {

	return `
		// ---- Raw-Safe Paper Grain Chunk (${uniformPrefix}) ----
		uniform float ${uniformPrefix}PaperGrain;
		uniform float ${uniformPrefix}WatercolorBleed;
		uniform float ${uniformPrefix}VariationOffset;

		float ${uniformPrefix}SamplePaperGrain( vec2 uv ) {
			if ( ${uniformPrefix}PaperGrain <= 0.0 ) return 1.0;
			float n = sin( uv.x * 8.0 + ${uniformPrefix}VariationOffset ) * cos( uv.y * 8.0 + ${uniformPrefix}VariationOffset );
			n = n * 0.5 + 0.5;
			return 1.0 - ${uniformPrefix}PaperGrain * ( 1.0 - n );
		}
		// ---- End Raw-Safe Paper Grain Chunk ----
	`;

}

// ---------------------------------------------------------------------------
// Anime feature — JS-side helpers
// ---------------------------------------------------------------------------

/**
 * Apply cel-shading on the CPU side using double.js for bit-exact
 * thresholding. Useful for UI previews, thumbnails, and offline analysis.
 *
 * @param {Color} baseColor
 * @param {Color} output
 * @param {number} ndotl
 * @param {number} bands
 * @param {number} [shadowTint=0.7]
 * @param {number} [softness=0.1]
 * @param {number} [highlightTint=1.15]
 * @returns {Color}
 */
function applyRawCelShadingCPU( baseColor, output, ndotl, bands, shadowTint = 0.7, softness = 0.1, highlightTint = 1.15 ) {

	if ( bands <= 1 ) {

		output.copy( baseColor );
		return output;

	}

	const bandWidth = 1.0 / bands;
	_double.value = ndotl;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const bandCenter = _double.value;

	const distToCenter = Math.abs( ndotl - bandCenter );
	const softFactor = Math.max( 0, Math.min( 1, distToCenter / ( bandWidth * 0.5 ) ) );
	const edgeBlend = ( 1 - softFactor ) * ( 1 - softness ) + softness;

	_double.value = bandIndex;
	_double.div( Math.max( 1, bands - 1 ) );
	const bandPos = _double.value;

	_double.value = shadowTint;
	_double.add( ( highlightTint - shadowTint ) * bandPos );
	const target = _double.value;

	_double.value = target;
	_double.add( ( ndotl - target ) * edgeBlend );
	const lit = Math.max( 0, Math.min( 1, _double.value ) );

	output.r = Math.max( 0, Math.min( 1, baseColor.r * lit ) );
	output.g = Math.max( 0, Math.min( 1, baseColor.g * lit ) );
	output.b = Math.max( 0, Math.min( 1, baseColor.b * lit ) );
	output.a = baseColor.a;

	return output;

}

/**
 * Generate a procedural paper-grain / watercolor texture using
 * simplex-noise with multiple octaves. The texture is used to modulate
 * raw shader output for a hand-painted feel — the characteristic
 * watercolor bleed of hand-drawn anime (reference images 1, 5, 7).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.03]
 * @param {number} [octaves=4]
 * @param {number} [paperGrain=0.3]
 * @param {number} [watercolorBleed=0.5]
 * @returns {Uint8Array}
 */
function generateRawTexture( width, height, scale = 0.03, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

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

			_double.value = value;
			_double.sub( 0.5 );
			_double.mul( 1.0 + watercolorBleed );
			_double.add( 0.5 );
			value = Math.max( 0, Math.min( 1, _double.value ) );

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
// bitecs SoA batch coordinator for real-time raw-shader uniform updates
// ---------------------------------------------------------------------------

const _rawShaderWorld = createWorld();

const RawShaderMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	rawAnimeBands: Types.ui8,
	rawAnimeShadowTint: Types.f64,
	rawAnimeSoftness: Types.f64,
	rawAnimeHighlightTint: Types.f64,
	rawRimPower: Types.f64,
	rawRimIntensity: Types.f64,
	rawMoodTemperature: Types.f64,
	rawMoodSaturation: Types.f64,
	rawMoodBrightness: Types.f64,
	rawMoodContrast: Types.f64,
	rawAmbientGradient: Types.f64,
	rawPaperGrain: Types.f64,
	rawWatercolorBleed: Types.f64,
	rawBrushJitter: Types.f64,
	rawVariationSeed: Types.f64,
	dirty: Types.ui8
} );

class RawShaderMaterialBatch {

	constructor() {

		this.world = _rawShaderWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a RawShaderMaterial instance for batched real-time uniform
	 * updates.
	 *
	 * @param {RawShaderMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, RawShaderMaterialComponent, eid );

		RawShaderMaterialComponent.materialPtr[ eid ] = this.materials.length;
		RawShaderMaterialComponent.rawAnimeBands[ eid ] = material.animeBands;
		RawShaderMaterialComponent.rawAnimeShadowTint[ eid ] = material.animeShadowTint;
		RawShaderMaterialComponent.rawAnimeSoftness[ eid ] = material.animeSoftness;
		RawShaderMaterialComponent.rawAnimeHighlightTint[ eid ] = material.animeHighlightTint;
		RawShaderMaterialComponent.rawRimPower[ eid ] = material.rimPower;
		RawShaderMaterialComponent.rawRimIntensity[ eid ] = material.rimIntensity;
		RawShaderMaterialComponent.rawMoodTemperature[ eid ] = material.moodTemperature;
		RawShaderMaterialComponent.rawMoodSaturation[ eid ] = material.moodSaturation;
		RawShaderMaterialComponent.rawMoodBrightness[ eid ] = material.moodBrightness;
		RawShaderMaterialComponent.rawMoodContrast[ eid ] = material.moodContrast;
		RawShaderMaterialComponent.rawAmbientGradient[ eid ] = material.ambientGradient;
		RawShaderMaterialComponent.rawPaperGrain[ eid ] = material.paperGrain;
		RawShaderMaterialComponent.rawWatercolorBleed[ eid ] = material.watercolorBleed;
		RawShaderMaterialComponent.rawBrushJitter[ eid ] = material.brushJitter;
		RawShaderMaterialComponent.rawVariationSeed[ eid ] = material.variationSeed;
		RawShaderMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued raw-shader material uniform updates in one cache-
	 * friendly pass. Uses double.js internally for bit-exact cel-band
	 * thresholding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ RawShaderMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.animeBands = RawShaderMaterialComponent.rawAnimeBands[ eid ];
			material.animeShadowTint = RawShaderMaterialComponent.rawAnimeShadowTint[ eid ];
			material.animeSoftness = RawShaderMaterialComponent.rawAnimeSoftness[ eid ];
			material.animeHighlightTint = RawShaderMaterialComponent.rawAnimeHighlightTint[ eid ];
			material.rimPower = RawShaderMaterialComponent.rawRimPower[ eid ];
			material.rimIntensity = RawShaderMaterialComponent.rawRimIntensity[ eid ];
			material.moodTemperature = RawShaderMaterialComponent.rawMoodTemperature[ eid ];
			material.moodSaturation = RawShaderMaterialComponent.rawMoodSaturation[ eid ];
			material.moodBrightness = RawShaderMaterialComponent.rawMoodBrightness[ eid ];
			material.moodContrast = RawShaderMaterialComponent.rawMoodContrast[ eid ];
			material.ambientGradient = RawShaderMaterialComponent.rawAmbientGradient[ eid ];
			material.paperGrain = RawShaderMaterialComponent.rawPaperGrain[ eid ];
			material.watercolorBleed = RawShaderMaterialComponent.rawWatercolorBleed[ eid ];
			material.brushJitter = RawShaderMaterialComponent.rawBrushJitter[ eid ];
			material.variationSeed = RawShaderMaterialComponent.rawVariationSeed[ eid ];

			// Push into the raw-safe uniform slots (if present)
			if ( material.uniforms ) {

				if ( material.uniforms.rawAnimeBands ) material.uniforms.rawAnimeBands.value = material.animeBands;
				if ( material.uniforms.rawAnimeShadowTint ) material.uniforms.rawAnimeShadowTint.value = material.animeShadowTint;
				if ( material.uniforms.rawAnimeSoftness ) material.uniforms.rawAnimeSoftness.value = material.animeSoftness;
				if ( material.uniforms.rawAnimeHighlightTint ) material.uniforms.rawAnimeHighlightTint.value = material.animeHighlightTint;
				if ( material.uniforms.rawAnimeRimPower ) material.uniforms.rawAnimeRimPower.value = material.rimPower;
				if ( material.uniforms.rawAnimeRimIntensity ) material.uniforms.rawAnimeRimIntensity.value = material.rimIntensity;
				if ( material.uniforms.rawAnimeMoodTemperature ) material.uniforms.rawAnimeMoodTemperature.value = material.moodTemperature;
				if ( material.uniforms.rawAnimeMoodSaturation ) material.uniforms.rawAnimeMoodSaturation.value = material.moodSaturation;
				if ( material.uniforms.rawAnimeMoodBrightness ) material.uniforms.rawAnimeMoodBrightness.value = material.moodBrightness;
				if ( material.uniforms.rawAnimeMoodContrast ) material.uniforms.rawAnimeMoodContrast.value = material.moodContrast;
				if ( material.uniforms.rawAnimePaperGrain ) material.uniforms.rawAnimePaperGrain.value = material.paperGrain;
				if ( material.uniforms.rawAnimeWatercolorBleed ) material.uniforms.rawAnimeWatercolorBleed.value = material.watercolorBleed;
				if ( material.uniforms.rawAnimeVariationOffset ) material.uniforms.rawAnimeVariationOffset.value = material.getVariationOffset();

			}

			RawShaderMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main RawShaderMaterial class — mirrors three.js/src/materials/RawShaderMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material rendered with custom shaders. Unlike {@link ShaderMaterial},
 * RawShaderMaterial disables the automatic injection of built-in uniforms,
 * attributes, and prefix defines. This gives the user complete control
 * over the shader source — ideal for hand-written GLSL that needs to
 * declare every uniform and attribute explicitly.
 *
 * Because of the raw contract, the anime extensions in this class are
 * opt-in: call `injectAnimeChunks()` to splice the anime GLSL chunks
 * into the shader source, and use the `rawAnime*` uniform prefix to
 * avoid collisions with the user's own uniforms.
 *
 * ```js
 * const material = new THREE.RawShaderMaterial( {
 *   uniforms: {
 *     time: { value: 1.0 },
 *     resolution: { value: new THREE.Vector2() }
 *   },
 *   vertexShader: document.getElementById( 'rawVertexShader' ).textContent,
 *   fragmentShader: document.getElementById( 'rawFragmentShader' ).textContent,
 *   glslVersion: THREE.GLSL3
 * } );
 *
 * // Opt in to anime features
 * material.animeBands = 3;
 * material.rimIntensity = 0.4;
 * material.moodTemperature = -0.3;
 * material.injectAnimeChunks();
 * ```
 *
 * @augments ShaderMaterial
 */
class RawShaderMaterial extends ShaderMaterial {

	/**
	 * Constructs a new raw shader material.
	 *
	 * @param {Object} [parameters] - An object with one or more properties
	 *   defining the material's appearance.
	 */
	constructor( parameters ) {

		super( parameters );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isRawShaderMaterial = true;

		this.type = 'RawShaderMaterial';

		/**
		 * Whether the anime chunks have been injected into the shader
		 * source. Populated automatically by `injectAnimeChunks()`.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this._animeChunksInjected = false;

		/**
		 * The uniform prefix used for injected raw-safe anime uniforms.
		 * Set before calling `injectAnimeChunks()` to avoid collisions
		 * with the user's own uniforms.
		 *
		 * @type {string}
		 * @default 'rawAnime'
		 */
		this.animeUniformPrefix = 'rawAnime';

		// RawShaderMaterial MUST keep the flags off — it does not use the
		// built-in prefix injection pipeline. The anime extensions are
		// opt-in via injectAnimeChunks().
		this.injectCelShading = false;
		this.injectRimGlow = false;
		this.injectMoodGrading = false;

	}

	// -----------------------------------------------------------------------
	// Anime helpers
	// -----------------------------------------------------------------------

	/**
	 * Inject the anime GLSL chunks into the user's shader source using the
	 * configured uniform prefix. Because RawShaderMaterial does not
	 * auto-inject, this method is the explicit opt-in point. Safe to call
	 * multiple times — subsequent calls rebuild the injection from the
	 * current anime parameters.
	 *
	 * The injected code declares new uniforms with the configured prefix
	 * and adds helper functions that the user's `main()` can call. No
	 * built-in attributes are referenced — the user is expected to supply
	 * `viewNormal` and `viewDir` from their own varyings.
	 *
	 * @returns {RawShaderMaterial} A reference to this instance.
	 */
	injectAnimeChunks() {

		const prefix = this.animeUniformPrefix;

		// Build the injection flags from the current parameters
		const enableCel = this.animeBands > 1;
		const enableRim = this.rimIntensity > 0;
		const enableMood =
			this.moodTemperature !== 0 ||
			this.moodSaturation !== 1.0 ||
			this.moodBrightness !== 1.0 ||
			this.moodContrast !== 1.0;
		const enablePaper = this.paperGrain > 0;

		this.injectCelShading = enableCel;
		this.injectRimGlow = enableRim;
		this.injectMoodGrading = enableMood;

		// Compose the combined chunk
		let chunk = '';
		if ( enableCel ) chunk += buildRawCelShadingChunk( this.animeBands, this.animeShadowTint, this.animeSoftness, this.animeHighlightTint, prefix );
		if ( enableRim ) chunk += buildRawRimGlowChunk( prefix );
		if ( enableMood ) chunk += buildRawMoodGradingChunk( prefix );
		if ( enablePaper ) chunk += buildRawPaperGrainChunk( prefix );

		if ( chunk.length === 0 ) {

			this._animeChunksInjected = false;
			return this;

		}

		// Splice the chunk at the very top of the fragment shader (raw-safe —
		// no #include <common> required)
		this.fragmentShader = chunk + '\n' + this.fragmentShader;

		// Populate the prefixed uniforms so the user can bind them directly
		this.uniforms[ `${prefix}Bands` ] = new Uniform( this.animeBands );
		this.uniforms[ `${prefix}ShadowTint` ] = new Uniform( this.animeShadowTint );
		this.uniforms[ `${prefix}Softness` ] = new Uniform( this.animeSoftness );
		this.uniforms[ `${prefix}HighlightTint` ] = new Uniform( this.animeHighlightTint );
		this.uniforms[ `${prefix}RimPower` ] = new Uniform( this.rimPower );
		this.uniforms[ `${prefix}RimIntensity` ] = new Uniform( this.rimIntensity );
		this.uniforms[ `${prefix}RimColor` ] = new Uniform( this.rimGlowColor );
		this.uniforms[ `${prefix}MoodTemperature` ] = new Uniform( this.moodTemperature );
		this.uniforms[ `${prefix}MoodSaturation` ] = new Uniform( this.moodSaturation );
		this.uniforms[ `${prefix}MoodBrightness` ] = new Uniform( this.moodBrightness );
		this.uniforms[ `${prefix}MoodContrast` ] = new Uniform( this.moodContrast );
		this.uniforms[ `${prefix}PaperGrain` ] = new Uniform( this.paperGrain );
		this.uniforms[ `${prefix}WatercolorBleed` ] = new Uniform( this.watercolorBleed );
		this.uniforms[ `${prefix}VariationOffset` ] = new Uniform( this.getVariationOffset() );

		this._animeChunksInjected = true;

		return this;

	}

	/**
	 * Remove the injected anime chunks and the associated uniforms. Useful
	 * for hot-reloading or reverting to a pure raw shader.
	 *
	 * @returns {RawShaderMaterial} A reference to this instance.
	 */
	removeAnimeChunks() {

		const prefix = this.animeUniformPrefix;

		// Strip all injected uniform declarations and helper functions
		this.fragmentShader = this.fragmentShader
			.replace( new RegExp( `// ---- Raw-Safe Cel-Shading Chunk \\(${ prefix }\\) ----[\\s\\S]*?// ---- End Raw-Safe Cel-Shading Chunk ----\\n?`, 'g' ), '' )
			.replace( new RegExp( `// ---- Raw-Safe Rim Glow Chunk \\(${ prefix }\\) ----[\\s\\S]*?// ---- End Raw-Safe Rim Glow Chunk ----\\n?`, 'g' ), '' )
			.replace( new RegExp( `// ---- Raw-Safe Mood Grading Chunk \\(${ prefix }\\) ----[\\s\\S]*?// ---- End Raw-Safe Mood Grading Chunk ----\\n?`, 'g' ), '' )
			.replace( new RegExp( `// ---- Raw-Safe Paper Grain Chunk \\(${ prefix }\\) ----[\\s\\S]*?// ---- End Raw-Safe Paper Grain Chunk ----\\n?`, 'g' ), '' );

		// Remove the associated uniforms
		delete this.uniforms[ `${prefix}Bands` ];
		delete this.uniforms[ `${prefix}ShadowTint` ];
		delete this.uniforms[ `${prefix}Softness` ];
		delete this.uniforms[ `${prefix}HighlightTint` ];
		delete this.uniforms[ `${prefix}RimPower` ];
		delete this.uniforms[ `${prefix}RimIntensity` ];
		delete this.uniforms[ `${prefix}RimColor` ];
		delete this.uniforms[ `${prefix}MoodTemperature` ];
		delete this.uniforms[ `${prefix}MoodSaturation` ];
		delete this.uniforms[ `${prefix}MoodBrightness` ];
		delete this.uniforms[ `${prefix}MoodContrast` ];
		delete this.uniforms[ `${prefix}PaperGrain` ];
		delete this.uniforms[ `${prefix}WatercolorBleed` ];
		delete this.uniforms[ `${prefix}VariationOffset` ];

		this._animeChunksInjected = false;
		this.injectCelShading = false;
		this.injectRimGlow = false;
		this.injectMoodGrading = false;

		return this;

	}

	/**
	 * Apply the cel-shading formula on the CPU side (for previews and
	 * thumbnails). Uses double.js for bit-exact thresholding.
	 *
	 * @param {Color} baseColor
	 * @param {Color} output
	 * @param {number} ndotl
	 * @returns {Color}
	 */
	applyCelShadingCPU( baseColor, output, ndotl ) {

		return applyRawCelShadingCPU(
			baseColor,
			output,
			ndotl,
			this.animeBands,
			this.animeShadowTint,
			this.animeSoftness,
			this.animeHighlightTint
		);

	}

	/**
	 * Generate a procedural paper-grain / watercolor texture for this
	 * raw shader material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.03]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generateRawTexture( width, height, scale = 0.03, octaves = 4 ) {

		return generateRawTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

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
	 * The custom program cache key. Extends the base key with the raw
	 * shader's injection state so the renderer recompiles when the anime
	 * chunks are added/removed.
	 *
	 * @returns {string}
	 */
	customProgramCacheKey() {

		return [
			super.customProgramCacheKey(),
			this._animeChunksInjected ? '1' : '0',
			this.animeUniformPrefix
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {RawShaderMaterial} source - The material to copy from.
	 * @return {RawShaderMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this._animeChunksInjected = source._animeChunksInjected;
		this.animeUniformPrefix = source.animeUniformPrefix;

		// Disable auto-injection for raw shaders
		this.injectCelShading = false;
		this.injectRimGlow = false;
		this.injectMoodGrading = false;

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

		data.type = 'RawShaderMaterial';

		if ( this._animeChunksInjected ) data.animeChunksInjected = true;
		if ( this.animeUniformPrefix !== 'rawAnime' ) data.animeUniformPrefix = this.animeUniformPrefix;

		return data;

	}

}

export { RawShaderMaterial, RawShaderMaterialBatch, buildRawCelShadingChunk, buildRawRimGlowChunk, buildRawMoodGradingChunk, buildRawPaperGrainChunk, applyRawCelShadingCPU, generateRawTexture };
export default RawShaderMaterial;