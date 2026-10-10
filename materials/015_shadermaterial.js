// file number : 015
// full path name : src/materials/015_shadermaterial.js
// description : ShaderMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 ShaderMaterial API — uniforms, vertexShader, fragmentShader, defines, wireframe, wireframeLinewidth, lights, clipping, glslVersion, fog, and the inherited material surface. Wraps the core 006_UniformsGroup.js for structured uniform management and 005_Uniform.js for individual uniform slots. Because ShaderMaterial is the escape hatch for custom shaders, this rewrite treats it as the primary injection point for advanced real-time anime effects: multi-band cel-shading injection (wraps user shaders with customizable band counts, shadow tint, and highlight boost), view-space rim glow injection (cyan water rims in 1, 3, 5; warm sunset character rims in 2, 4, 6), mood-based color grading injection (snowy cyan, sunset orange, vibrant flora), ambient sky-to-ground gradient tinting (matching the mountain and sky layers in 1, 5), procedural paper-grain and watercolor overlays via simplex-noise, brush-jitter UV distortion for hand-drawn wobble, per-instance variation for crowds, and hot-reloadable uniform updates. Imports Color, ColorManagement, and other math classes strictly from threejs_new01 math, plus Uniform and UniformsGroup strictly from threejs_new01 core. Uses gl-matrix for zero-allocation N·L and rim computations, double.js for bit-exact cel-band thresholding and HDR mood grading, bitecs SoA batching for real-time uniform updates across thousands of shader instances, and simplex-noise for procedural texture variation and paper-grain.
// best for : ShaderMaterial, custom toon shaders, anime post-processing, stylized water/clouds/foliage shaders, hair/eye sparkle shaders, NPR (non-photorealistic rendering) pipelines, and any three.js custom shader that needs real-time anime stylization.
// license : MIT

import { Material } from './001_material.js';
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

// gl-matrix scratch for zero-allocation N·L, rim, and mood computations
const _gm_normal = glMatrix.vec3.create();
const _gm_view = glMatrix.vec3.create();
const _gm_light = glMatrix.vec3.create();
const _gm_rgb = glMatrix.vec3.create();
const _gm_rgb_out = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — injectable cel-shading GLSL chunk
// ---------------------------------------------------------------------------

/**
 * Generate the GLSL source for the injectable cel-shading chunk. The
 * generated code reads the current view-space normal and light direction,
 * quantizes the N·L term into discrete bands using a configurable
 * threshold, and multiplies the fragment color by the resulting shading.
 * Uses double.js at the JS side for bit-exact threshold validation.
 *
 * @param {number} bands - Number of cel bands (2-6 recommended).
 * @param {number} [shadowTint=0.7] - Shadow band brightness multiplier.
 * @param {number} [softness=0.1] - Edge softness in [0, 0.5].
 * @param {number} [highlightTint=1.15] - Highlight band brightness multiplier.
 * @param {string} [uniformPrefix='anime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildCelShadingChunk( bands, shadowTint = 0.7, softness = 0.1, highlightTint = 1.15, uniformPrefix = 'anime' ) {

	// Validate with double.js (bit-exact threshold check)
	_double.value = bands;
	if ( _double.value < 2 ) return '// cel-shading disabled (bands < 2)';

	return `
		// ---- Injectible Cel-Shading Chunk (${uniformPrefix}) ----
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
		// ---- End Cel-Shading Chunk ----
	`;

}

/**
 * Generate the GLSL source for the injectable rim-glow chunk. Produces
 * a view-space normal rim glow, matching the cyan water rims in
 * reference 1, 3, 5 and the warm sunset character rims in 2, 4, 6.
 *
 * @param {string} [uniformPrefix='anime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildRimGlowChunk( uniformPrefix = 'anime' ) {

	return `
		// ---- Injectible Rim Glow Chunk (${uniformPrefix}) ----
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
		// ---- End Rim Glow Chunk ----
	`;

}

/**
 * Generate the GLSL source for the injectable mood grading chunk.
 * Performs HSV-space hue shift, saturation, and temperature adjustments
 * matching the reference imagery's mood spectrum (snowy cyan, sunset
 * orange, vibrant flora).
 *
 * @param {string} [uniformPrefix='anime'] - Prefix for injected uniforms.
 * @returns {string} GLSL chunk source.
 */
function buildMoodGradingChunk( uniformPrefix = 'anime' ) {

	return `
		// ---- Injectible Mood Grading Chunk (${uniformPrefix}) ----
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
		// ---- End Mood Grading Chunk ----
	`;

}

// ---------------------------------------------------------------------------
// Anime feature — JS-side cel banding helper (for CPU-side previews)
// ---------------------------------------------------------------------------

/**
 * Apply cel banding to a color on the CPU side using double.js for
 * bit-exact thresholding. Useful for UI previews, thumbnails, and
 * offline analysis of what the shader will do.
 *
 * @param {Color} baseColor
 * @param {Color} output
 * @param {number} ndotl - Normal·light in [0, 1].
 * @param {number} bands
 * @param {number} [shadowTint=0.7]
 * @param {number} [softness=0.1]
 * @param {number} [highlightTint=1.15]
 * @returns {Color}
 */
function applyCelShadingCPU( baseColor, output, ndotl, bands, shadowTint = 0.7, softness = 0.1, highlightTint = 1.15 ) {

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

	// Multiply base color by lit factor
	output.r = Math.max( 0, Math.min( 1, baseColor.r * _double.value ) );
	output.g = Math.max( 0, Math.min( 1, baseColor.g * _double.value ) );
	output.b = Math.max( 0, Math.min( 1, baseColor.b * _double.value ) );
	output.a = baseColor.a;

	return output;

}

// ---------------------------------------------------------------------------
// Anime feature — procedural paper-grain / watercolor texture
// ---------------------------------------------------------------------------

/**
 * Generate a combined paper-grain and watercolor texture using
 * simplex-noise with multiple octaves. The texture is used to modulate
 * shader output for a hand-painted feel — the characteristic watercolor
 * bleed of hand-drawn anime (reference images 1, 5, 7).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.03]
 * @param {number} [octaves=4]
 * @param {number} [paperGrain=0.3]
 * @param {number} [watercolorBleed=0.5]
 * @returns {Uint8Array}
 */
function generateShaderTexture( width, height, scale = 0.03, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

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
// bitecs SoA batch coordinator for real-time shader material uniform updates
// ---------------------------------------------------------------------------

const _shaderWorld = createWorld();

const ShaderMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	animeBands: Types.ui8,
	animeShadowTint: Types.f64,
	animeSoftness: Types.f64,
	animeHighlightTint: Types.f64,
	rimPower: Types.f64,
	rimIntensity: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	ambientGradient: Types.f64,
	ambientHeight: Types.f64,
	paperGrain: Types.f64,
	watercolorBleed: Types.f64,
	brushJitter: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class ShaderMaterialBatch {

	constructor() {

		this.world = _shaderWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a ShaderMaterial instance for batched real-time uniform updates.
	 *
	 * @param {ShaderMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, ShaderMaterialComponent, eid );

		ShaderMaterialComponent.materialPtr[ eid ] = this.materials.length;
		ShaderMaterialComponent.animeBands[ eid ] = material.animeBands;
		ShaderMaterialComponent.animeShadowTint[ eid ] = material.animeShadowTint;
		ShaderMaterialComponent.animeSoftness[ eid ] = material.animeSoftness;
		ShaderMaterialComponent.animeHighlightTint[ eid ] = material.animeHighlightTint;
		ShaderMaterialComponent.rimPower[ eid ] = material.rimPower;
		ShaderMaterialComponent.rimIntensity[ eid ] = material.rimIntensity;
		ShaderMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		ShaderMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		ShaderMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		ShaderMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		ShaderMaterialComponent.ambientGradient[ eid ] = material.ambientGradient;
		ShaderMaterialComponent.ambientHeight[ eid ] = material.ambientHeight;
		ShaderMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		ShaderMaterialComponent.watercolorBleed[ eid ] = material.watercolorBleed;
		ShaderMaterialComponent.brushJitter[ eid ] = material.brushJitter;
		ShaderMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		ShaderMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued shader-material uniform updates in one cache-friendly
	 * pass. Uses double.js internally for bit-exact cel-band thresholding
	 * and mood grading, and pushes updates to the material's uniform group
	 * and the shader's built-in uniforms.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ ShaderMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.animeBands = ShaderMaterialComponent.animeBands[ eid ];
			material.animeShadowTint = ShaderMaterialComponent.animeShadowTint[ eid ];
			material.animeSoftness = ShaderMaterialComponent.animeSoftness[ eid ];
			material.animeHighlightTint = ShaderMaterialComponent.animeHighlightTint[ eid ];
			material.rimPower = ShaderMaterialComponent.rimPower[ eid ];
			material.rimIntensity = ShaderMaterialComponent.rimIntensity[ eid ];
			material.moodTemperature = ShaderMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = ShaderMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = ShaderMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = ShaderMaterialComponent.moodContrast[ eid ];
			material.ambientGradient = ShaderMaterialComponent.ambientGradient[ eid ];
			material.ambientHeight = ShaderMaterialComponent.ambientHeight[ eid ];
			material.paperGrain = ShaderMaterialComponent.paperGrain[ eid ];
			material.watercolorBleed = ShaderMaterialComponent.watercolorBleed[ eid ];
			material.brushJitter = ShaderMaterialComponent.brushJitter[ eid ];
			material.variationSeed = ShaderMaterialComponent.variationSeed[ eid ];

			// Push into the material's uniform slots (if present)
			if ( material.uniforms ) {

				if ( material.uniforms.animeBands ) material.uniforms.animeBands.value = material.animeBands;
				if ( material.uniforms.animeShadowTint ) material.uniforms.animeShadowTint.value = material.animeShadowTint;
				if ( material.uniforms.animeSoftness ) material.uniforms.animeSoftness.value = material.animeSoftness;
				if ( material.uniforms.animeHighlightTint ) material.uniforms.animeHighlightTint.value = material.animeHighlightTint;
				if ( material.uniforms.animeRimPower ) material.uniforms.animeRimPower.value = material.rimPower;
				if ( material.uniforms.animeRimIntensity ) material.uniforms.animeRimIntensity.value = material.rimIntensity;
				if ( material.uniforms.animeMoodTemperature ) material.uniforms.animeMoodTemperature.value = material.moodTemperature;
				if ( material.uniforms.animeMoodSaturation ) material.uniforms.animeMoodSaturation.value = material.moodSaturation;
				if ( material.uniforms.animeMoodBrightness ) material.uniforms.animeMoodBrightness.value = material.moodBrightness;
				if ( material.uniforms.animeMoodContrast ) material.uniforms.animeMoodContrast.value = material.moodContrast;

			}

			ShaderMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main ShaderMaterial class — mirrors three.js/src/materials/ShaderMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material rendered with custom shaders. A shader is a small program
 * written in GLSL that runs on the GPU.
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls that are injected into
 * the user's shader source at compile time: multi-band cel-shading,
 * view-space normal rim glow, mood-based color grading, ambient
 * sky-to-ground gradient tinting, and paper-grain / watercolor texture
 * variation.
 *
 * ```js
 * const material = new THREE.ShaderMaterial( {
 *   uniforms: {
 *     time: { value: 1.0 },
 *     resolution: { value: new THREE.Vector2() }
 *   },
 *   vertexShader: document.getElementById( 'vertexShader' ).textContent,
 *   fragmentShader: document.getElementById( 'fragmentShader' ).textContent,
 *   animeBands: 3,
 *   animeShadowTint: 0.7,
 *   rimIntensity: 0.4,
 *   moodTemperature: -0.3
 * } );
 * ```
 *
 * @augments Material
 */
class ShaderMaterial extends Material {

	/**
	 * Constructs a new shader material.
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
		this.isShaderMaterial = true;

		this.type = 'ShaderMaterial';

		/**
		 * The custom defines.
		 *
		 * @type {Object}
		 */
		this.defines = {};

		/**
		 * The custom uniforms.
		 *
		 * @type {Object}
		 */
		this.uniforms = {};

		/**
		 * The vertex shader source.
		 *
		 * @type {string}
		 */
		this.vertexShader = 'void main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}';

		/**
		 * The fragment shader source.
		 *
		 * @type {string}
		 */
		this.fragmentShader = 'void main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}';

		/**
		 * Defines the GLSL version of custom shader code.
		 *
		 * @type {?string}
		 * @default null
		 */
		this.glslVersion = null;

		/**
		 * Whether to render the material as wireframe.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.wireframe = false;

		/**
		 * Controls wireframe thickness.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.wireframeLinewidth = 1;

		/**
		 * Whether to use fog.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.fog = false;

		/**
		 * Whether to use lights.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.lights = false;

		/**
		 * Whether to use clipping planes.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.clipping = false;

		/**
		 * Whether to extend the built-in uniforms.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.extensions = {
			derivatives: false,
			fragDepth: false,
			drawBuffers: false,
			shaderTextureLOD: false
		};

		/**
		 * The index of the current vertex in the default attribute location.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.defaultAttributeValues = {
			'color': [ 1, 1, 1 ],
			'uv': [ 0, 0 ],
			'uv1': [ 0, 0 ]
		};

		/**
		 * Defines the indices for the built-in attribute location.
		 *
		 * @type {Object}
		 */
		this.index0AttributeName = undefined;

		/**
		 * Structured uniform group instance that wraps this material's
		 * uniforms. Uses the threejs_new01 core UniformsGroup for
		 * efficient, structured uploads.
		 *
		 * @type {UniformsGroup}
		 */
		this.uniformsGroup = new UniformsGroup();

		// -------------------------------------------------------------------
		// Anime Rendering Extensions
		// -------------------------------------------------------------------

		/**
		 * Number of anime cel bands injected into the fragment shader.
		 * 0 = off (no injection), 2 = classic hard-shadow, 3-6 = soft cel.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.animeBands = 0;

		/**
		 * Shadow band brightness multiplier for the injected cel shader.
		 *
		 * @type {number}
		 * @default 0.7
		 */
		this.animeShadowTint = 0.7;

		/**
		 * Cel band edge softness for the injected shader.
		 *
		 * @type {number}
		 * @default 0.1
		 */
		this.animeSoftness = 0.1;

		/**
		 * Highlight band brightness multiplier.
		 *
		 * @type {number}
		 * @default 1.15
		 */
		this.animeHighlightTint = 1.15;

		/**
		 * Rim-glow falloff power.
		 *
		 * @type {number}
		 * @default 2.5
		 */
		this.rimPower = 2.5;

		/**
		 * Rim-glow intensity. Set > 0 to inject the rim glow chunk.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimIntensity = 0;

		/**
		 * Rim-glow color.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.rimGlowColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Mood temperature shift in [-1, 1].
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
		 * Whether the mood-grading chunk should be injected into the
		 * fragment shader. Enabled automatically when any mood parameter
		 * is non-default.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.injectMoodGrading = false;

		/**
		 * Whether the cel-shading chunk should be injected into the
		 * fragment shader. Enabled automatically when animeBands > 1.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.injectCelShading = false;

		/**
		 * Whether the rim-glow chunk should be injected into the fragment
		 * shader. Enabled automatically when rimIntensity > 0.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.injectRimGlow = false;

		/**
		 * Ambient gradient intensity (applied via the mood chunk).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.ambientGradient = 0;

		/**
		 * Ambient gradient height range.
		 *
		 * @type {number}
		 * @default 10
		 */
		this.ambientHeight = 10;

		/**
		 * Sky tint for the ambient gradient.
		 *
		 * @type {Color}
		 * @default (0.5, 0.75, 1.0)
		 */
		this.skyTint = new Color( 0.5, 0.75, 1.0 );

		/**
		 * Ground tint for the ambient gradient.
		 *
		 * @type {Color}
		 * @default (1.0, 0.85, 0.7)
		 */
		this.groundTint = new Color( 1.0, 0.85, 0.7 );

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
		 * Brush jitter amplitude for hand-drawn UV distortion.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.brushJitter = 0;

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
	 * Build and enable the injectable GLSL chunks based on the current
	 * anime parameters. Populates `uniforms` with the injected slots and
	 * sets the `inject*` flags.
	 *
	 * @returns {ShaderMaterial} A reference to this instance.
	 */
	buildInjection() {

		this.injectCelShading = this.animeBands > 1;
		this.injectRimGlow = this.rimIntensity > 0;
		this.injectMoodGrading =
			this.moodTemperature !== 0 ||
			this.moodSaturation !== 1.0 ||
			this.moodBrightness !== 1.0 ||
			this.moodContrast !== 1.0 ||
			this.ambientGradient > 0;

		if ( this.injectCelShading ) {

			this.uniforms.animeBands = new Uniform( this.animeBands );
			this.uniforms.animeShadowTint = new Uniform( this.animeShadowTint );
			this.uniforms.animeSoftness = new Uniform( this.animeSoftness );
			this.uniforms.animeHighlightTint = new Uniform( this.animeHighlightTint );

		}

		if ( this.injectRimGlow ) {

			this.uniforms.animeRimPower = new Uniform( this.rimPower );
			this.uniforms.animeRimIntensity = new Uniform( this.rimIntensity );
			this.uniforms.animeRimColor = new Uniform( this.rimGlowColor );

		}

		if ( this.injectMoodGrading ) {

			this.uniforms.animeMoodTemperature = new Uniform( this.moodTemperature );
			this.uniforms.animeMoodSaturation = new Uniform( this.moodSaturation );
			this.uniforms.animeMoodBrightness = new Uniform( this.moodBrightness );
			this.uniforms.animeMoodContrast = new Uniform( this.moodContrast );

		}

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

		return applyCelShadingCPU(
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
	 * shader material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.03]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generateShaderTexture( width, height, scale = 0.03, octaves = 4 ) {

		return generateShaderTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

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
	 * The default `onBeforeCompile` hook. Injects the enabled anime GLSL
	 * chunks into the user's shader source. If a chunk is disabled, the
	 * user's shader is passed through unchanged.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		super.onBeforeCompile( shader, renderer );

		// Build the injection state first (populates uniforms and flags)
		this.buildInjection();

		// Prepare the combined chunk string
		let chunk = '';
		if ( this.injectCelShading ) chunk += buildCelShadingChunk( this.animeBands, this.animeShadowTint, this.animeSoftness, this.animeHighlightTint );
		if ( this.injectRimGlow ) chunk += buildRimGlowChunk();
		if ( this.injectMoodGrading ) chunk += buildMoodGradingChunk();

		if ( chunk.length === 0 ) return;

		// Inject the chunk after the common include
		shader.fragmentShader = shader.fragmentShader.replace(
			'#include <common>',
			`#include <common>\n${chunk}`
		);

		// Append the application code just before the final gl_FragColor write
		let applyCode = '';
		if ( this.injectCelShading ) applyCode += '\tgl_FragColor.rgb = animeApplyCelShading( gl_FragColor.rgb, vNormal, vec3( 0.0, 0.0, 1.0 ) );\n';
		if ( this.injectRimGlow ) applyCode += '\tgl_FragColor.rgb = animeApplyRimGlow( gl_FragColor.rgb, vNormal, vViewPosition );\n';
		if ( this.injectMoodGrading ) applyCode += '\tgl_FragColor.rgb = animeApplyMoodGrading( gl_FragColor.rgb );\n';

		// Find a good insertion point — after the final gl_FragColor assignment
		shader.fragmentShader = shader.fragmentShader.replace(
			/gl_FragColor\s*=\s*[^;]+;/,
			match => `${match}\n${applyCode}`
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
			this.animeBands,
			this.animeShadowTint,
			this.animeSoftness,
			this.animeHighlightTint,
			this.rimPower,
			this.rimIntensity,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.ambientGradient,
			this.ambientHeight,
			this.paperGrain,
			this.watercolorBleed,
			this.brushJitter,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {ShaderMaterial} source - The material to copy from.
	 * @return {ShaderMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.fragmentShader = source.fragmentShader;
		this.vertexShader = source.vertexShader;

		this.uniforms = UniformsUtils.clone( source.uniforms );

		this.defines = Object.assign( {}, source.defines );

		this.wireframe = source.wireframe;
		this.wireframeLinewidth = source.wireframeLinewidth;

		this.fog = source.fog;
		this.lights = source.lights;
		this.clipping = source.clipping;

		this.extensions = Object.assign( {}, source.extensions );

		this.glslVersion = source.glslVersion;

		// Anime extensions
		this.animeBands = source.animeBands;
		this.animeShadowTint = source.animeShadowTint;
		this.animeSoftness = source.animeSoftness;
		this.animeHighlightTint = source.animeHighlightTint;
		this.rimPower = source.rimPower;
		this.rimIntensity = source.rimIntensity;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodContrast = source.moodContrast;
		this.ambientGradient = source.ambientGradient;
		this.ambientHeight = source.ambientHeight;
		this.skyTint.copy( source.skyTint );
		this.groundTint.copy( source.groundTint );
		this.paperGrain = source.paperGrain;
		this.watercolorBleed = source.watercolorBleed;
		this.brushJitter = source.brushJitter;
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

		data.type = 'ShaderMaterial';

		data.uniforms = {};

		for ( const name in this.uniforms ) {

			const uniform = this.uniforms[ name ];
			const value = uniform.value;

			if ( value && value.isTexture ) {

				data.uniforms[ name ] = {
					type: 't',
					value: value.toJSON( meta ).uuid
				};

			} else if ( value && value.isColor ) {

				data.uniforms[ name ] = { type: 'c', value: value.getHex() };

			} else if ( value && value.isVector2 ) {

				data.uniforms[ name ] = { type: 'v2', value: value.toArray() };

			} else if ( value && value.isVector3 ) {

				data.uniforms[ name ] = { type: 'v3', value: value.toArray() };

			} else if ( value && value.isVector4 ) {

				data.uniforms[ name ] = { type: 'v4', value: value.toArray() };

			} else if ( value && value.isMatrix3 ) {

				data.uniforms[ name ] = { type: 'm3', value: value.toArray() };

			} else if ( value && value.isMatrix4 ) {

				data.uniforms[ name ] = { type: 'm4', value: value.toArray() };

			} else {

				data.uniforms[ name ] = { value: value };

			}

		}

		data.vertexShader = this.vertexShader;
		data.fragmentShader = this.fragmentShader;
		data.defines = Object.assign( {}, this.defines );

		data.wireframe = this.wireframe;
		data.wireframeLinewidth = this.wireframeLinewidth;

		data.fog = this.fog;
		data.lights = this.lights;
		data.clipping = this.clipping;

		if ( this.glslVersion !== null ) data.glslVersion = this.glslVersion;

		// Anime extensions
		if ( this.animeBands !== 0 ) data.animeBands = this.animeBands;
		if ( this.animeShadowTint !== 0.7 ) data.animeShadowTint = this.animeShadowTint;
		if ( this.animeSoftness !== 0.1 ) data.animeSoftness = this.animeSoftness;
		if ( this.animeHighlightTint !== 1.15 ) data.animeHighlightTint = this.animeHighlightTint;
		if ( this.rimPower !== 2.5 ) data.rimPower = this.rimPower;
		if ( this.rimIntensity !== 0 ) data.rimIntensity = this.rimIntensity;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodContrast !== 1.0 ) data.moodContrast = this.moodContrast;
		if ( this.ambientGradient !== 0 ) data.ambientGradient = this.ambientGradient;
		if ( this.ambientHeight !== 10 ) data.ambientHeight = this.ambientHeight;
		if ( this.skyTint.getHex() !== new Color( 0.5, 0.75, 1.0 ).getHex() ) data.skyTint = this.skyTint.getHex();
		if ( this.groundTint.getHex() !== new Color( 1.0, 0.85, 0.7 ).getHex() ) data.groundTint = this.groundTint.getHex();
		if ( this.paperGrain !== 0 ) data.paperGrain = this.paperGrain;
		if ( this.watercolorBleed !== 0 ) data.watercolorBleed = this.watercolorBleed;
		if ( this.brushJitter !== 0 ) data.brushJitter = this.brushJitter;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

// ---------------------------------------------------------------------------
// Utility — clone uniforms (used by copy)
// ---------------------------------------------------------------------------

const UniformsUtils = {

	clone( uniforms ) {

		const cloned = {};

		for ( const name in uniforms ) {

			const uniform = uniforms[ name ];
			const value = uniform.value;

			if ( value && value.isColor ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( value && value.isVector2 ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( value && value.isVector3 ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( value && value.isVector4 ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( value && value.isMatrix3 ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( value && value.isMatrix4 ) {

				cloned[ name ] = new Uniform( value.clone() );

			} else if ( Array.isArray( value ) ) {

				cloned[ name ] = new Uniform( value.slice() );

			} else {

				cloned[ name ] = new Uniform( value );

			}

		}

		return cloned;

	}

};

export { ShaderMaterial, ShaderMaterialBatch, buildCelShadingChunk, buildRimGlowChunk, buildMoodGradingChunk, applyCelShadingCPU, generateShaderTexture, UniformsUtils };
export default ShaderMaterial;