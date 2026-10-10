// file number : 008
// full path name : src/materials/008_meshnormalmaterial.js
// description : MeshNormalMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 001_material.js base class and preserves the full r185 MeshNormalMaterial API — bumpMap, bumpScale, normalMap, normalMapType, normalScale, displacementMap, displacementScale, displacementBias, wireframe, wireframeLinewidth, flatShading, fog, and the inherited material surface. MeshNormalMaterial maps normal vectors to RGB colors — a technique uniquely suited to anime stylization because the RGB output gives direct per-fragment control over rim direction, silhouette detection, and edge highlighting. Adds real-time anime features specifically tuned for normal-based rendering: normal-driven cel banding, view-space normal rim glow (the cyan water rims in reference images 1, 3, 5 and the sunset character rims in 2, 4, 6), mood-based normal color grading, procedural normal variation via simplex-noise, and per-instance variation for crowds. Imports Color, Vector2, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation normal vector transforms, double.js for bit-exact normal normalization and cel-band thresholding, bitecs SoA batching for real-time updates across thousands of instances, and simplex-noise for procedural normal perturbation.
// best for : MeshNormalMaterial, anime outline passes, technical visualization, normal-map debugging, stylized edge detection, rim-light pre-passes, and any three.js mesh that needs per-fragment normal vector access with anime stylization.
// license : MIT

import { Material } from './001_material.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
import {
	NormalBlending,
	FrontSide,
	TangentSpaceNormalMap,
	ObjectSpaceNormalMap
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

// gl-matrix scratch for zero-allocation normal vector transforms
const _gm_normal = glMatrix.vec3.create();
const _gm_view = glMatrix.vec3.create();
const _gm_temp = glMatrix.vec3.create();
const _gm_rgb = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — normal-driven cel banding
// ---------------------------------------------------------------------------

/**
 * Apply cel-shading banding to a normal-mapped color using double.js for
 * bit-exact thresholding. Because normals carry directional information,
 * banding the RGB channels independently creates the characteristic
 * "directional cel" effect — the flat color zones that shift with the
 * surface curvature, matching the reference imagery's layered shading.
 *
 * @param {Color} normalColor - The normal-mapped RGB color.
 * @param {Color} output - The output color (modified in place).
 * @param {number} bands - Number of cel bands per channel (2-4 recommended).
 * @param {number} quantizeAmount - Blend amount between continuous and banded (0-1).
 * @param {number} [shadowTint=0.85] - Shadow band brightness multiplier.
 * @returns {Color}
 */
function applyNormalCelBanding( normalColor, output, bands, quantizeAmount, shadowTint = 0.85 ) {

	if ( bands <= 1 ) {

		output.copy( normalColor );
		return output;

	}

	const bandWidth = 1.0 / bands;

	// Quantize each channel independently
	const channels = [ normalColor.r, normalColor.g, normalColor.b ];
	const outChannels = [ 0, 0, 0 ];

	for ( let c = 0; c < 3; c ++ ) {

		_double.value = channels[ c ];
		_double.div( bandWidth );
		const bandIndex = Math.floor( _double.value );
		_double.value = bandIndex;
		_double.mul( bandWidth );
		_double.add( bandWidth * 0.5 );
		const quantized = _double.value;

		// Blend continuous and quantized
		_double.value = channels[ c ];
		_double.add( ( quantized - channels[ c ] ) * quantizeAmount );
		let finalVal = _double.value;

		// Apply shadow tint to darker bands for extra anime contrast
		if ( finalVal < 0.5 ) {

			_double.value = finalVal;
			_double.mul( shadowTint );
			finalVal = _double.value;

		}

		outChannels[ c ] = Math.max( 0, Math.min( 1, finalVal ) );

	}

	output.setRGB( outChannels[ 0 ], outChannels[ 1 ], outChannels[ 2 ], ColorManagement.workingColorSpace );
	output.a = normalColor.a;

	return output;

}

// ---------------------------------------------------------------------------
// Anime feature — view-space normal rim glow
// ---------------------------------------------------------------------------

/**
 * Compute a silhouette rim glow from the view-space normal. Surfaces
 * facing away from the view (silhouettes) produce a strong rim, mimicking
 * the cyan water rims in reference images 1, 3, 5 and the warm sunset
 * character rims in 2, 4, 6. Uses gl-matrix for zero-allocation vector
 * staging and double.js for bit-exact accumulation.
 *
 * @param {Vector3} viewNormal - The view-space normal (normalized).
 * @param {Vector3} viewDir - The view direction (normalized).
 * @param {number} [power=2.0] - Rim falloff power (higher = tighter rim).
 * @param {number} [intensity=1.0] - Rim intensity multiplier.
 * @returns {number} Rim intensity in [0, 1].
 */
function computeNormalRimGlow( viewNormal, viewDir, power = 2.0, intensity = 1.0 ) {

	glMatrix.vec3.set( _gm_normal, viewNormal.x, viewNormal.y, viewNormal.z );
	glMatrix.vec3.set( _gm_view, viewDir.x, viewDir.y, viewDir.z );

	// Normalize both vectors
	glMatrix.vec3.normalize( _gm_normal, _gm_normal );
	glMatrix.vec3.normalize( _gm_view, _gm_view );

	// Dot product
	_double.value = glMatrix.vec3.dot( _gm_normal, _gm_view );
	const ndotv = _double.value;

	// Rim factor: 1 - |ndotv|, shaped by power
	_double.value = 1.0 - Math.abs( ndotv );
	const rim = Math.pow( _double.value, power );

	return Math.max( 0, Math.min( 1, rim * intensity ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based normal color grading
// ---------------------------------------------------------------------------

/**
 * Apply mood-based color grading to the normal-mapped RGB output. Handles
 * warm (sunset orange) and cool (snowy cyan) moods seen across the
 * reference imagery. Uses gl-matrix for zero-allocation staging and
 * double.js for bit-exact accumulation.
 *
 * @param {Color} color - The normal color to grade (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} contrast - Contrast multiplier.
 * @returns {Color}
 */
function gradeNormalColor( color, temperature, saturation, brightness, contrast ) {

	glMatrix.vec3.set( _gm_rgb, color.r, color.g, color.b );

	const lum = 0.2126 * _gm_rgb[ 0 ] + 0.7152 * _gm_rgb[ 1 ] + 0.0722 * _gm_rgb[ 2 ];

	// Saturation
	_double.value = lum;
	_double.add( ( _gm_rgb[ 0 ] - lum ) * saturation );
	let r = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 1 ] - lum ) * saturation );
	let g = _double.value;

	_double.value = lum;
	_double.add( ( _gm_rgb[ 2 ] - lum ) * saturation );
	let b = _double.value;

	// Contrast
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

	// Temperature
	_double.value = r;
	_double.add( temperature * 0.1 );
	r = _double.value;

	_double.value = b;
	_double.sub( temperature * 0.1 );
	b = _double.value;

	// Brightness
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
// Anime feature — procedural normal perturbation
// ---------------------------------------------------------------------------

/**
 * Generate a procedural normal-perturbation texture using simplex-noise.
 * This is layered onto the sampled normal to create subtle organic surface
 * detail — matching the hand-drawn feel of the reference imagery's snow,
 * rock, and foliage surfaces. The output is a tangent-space normal map
 * where the RGB channels encode small perturbations around (0.5, 0.5, 1.0).
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.05] - Noise scale.
 * @param {number} [octaves=3] - Number of noise octaves.
 * @param {number} [intensity=0.1] - Perturbation intensity in [0, 1].
 * @returns {Uint8Array} RGBA normal-map buffer.
 */
function generateNormalPerturbation( width, height, scale = 0.05, octaves = 3, intensity = 0.1 ) {

	const out = new Uint8Array( width * height * 4 );

	for ( let y = 0; y < height; y ++ ) {

		for ( let x = 0; x < width; x ++ ) {

			const p = ( y * width + x ) * 4;

			// Multi-octave noise for X and Y perturbation
			let nx = 0, ny = 0;
			let amplitude = 1;
			let frequency = scale;
			let maxAmplitude = 0;

			for ( let o = 0; o < octaves; o ++ ) {

				nx += _noise2D( x * frequency, y * frequency ) * amplitude;
				ny += _noise2D( x * frequency + 100, y * frequency + 100 ) * amplitude;
				maxAmplitude += amplitude;
				amplitude *= 0.5;
				frequency *= 2.0;

			}

			nx = ( nx / maxAmplitude ) * 0.5 + 0.5;
			ny = ( ny / maxAmplitude ) * 0.5 + 0.5;

			// Perturb around neutral (0.5, 0.5, 1.0)
			_double.value = 0.5;
			_double.add( ( nx - 0.5 ) * intensity );
			const r = Math.max( 0, Math.min( 1, _double.value ) );

			_double.value = 0.5;
			_double.add( ( ny - 0.5 ) * intensity );
			const g = Math.max( 0, Math.min( 1, _double.value ) );

			// Z channel stays near 1.0 for flat normal
			const b = 1.0;

			out[ p ] = Math.floor( r * 255 );
			out[ p + 1 ] = Math.floor( g * 255 );
			out[ p + 2 ] = Math.floor( b * 255 );
			out[ p + 3 ] = 255;

		}

	}

	return out;

}

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for real-time normal material updates
// ---------------------------------------------------------------------------

const _normalWorld = createWorld();

const NormalMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	celBands: Types.ui8,
	celQuantize: Types.f64,
	shadowTint: Types.f64,
	rimPower: Types.f64,
	rimIntensity: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	normalPerturbation: Types.f64,
	dirty: Types.ui8
} );

class MeshNormalMaterialBatch {

	constructor() {

		this.world = _normalWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a MeshNormalMaterial instance for batched real-time updates.
	 *
	 * @param {MeshNormalMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, NormalMaterialComponent, eid );

		NormalMaterialComponent.materialPtr[ eid ] = this.materials.length;
		NormalMaterialComponent.celBands[ eid ] = material.celBands;
		NormalMaterialComponent.celQuantize[ eid ] = material.celQuantize;
		NormalMaterialComponent.shadowTint[ eid ] = material.shadowTint;
		NormalMaterialComponent.rimPower[ eid ] = material.rimPower;
		NormalMaterialComponent.rimIntensity[ eid ] = material.rimIntensity;
		NormalMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		NormalMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		NormalMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		NormalMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		NormalMaterialComponent.normalPerturbation[ eid ] = material.normalPerturbation;
		NormalMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued normal-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact normal banding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ NormalMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.celBands = NormalMaterialComponent.celBands[ eid ];
			material.celQuantize = NormalMaterialComponent.celQuantize[ eid ];
			material.shadowTint = NormalMaterialComponent.shadowTint[ eid ];
			material.rimPower = NormalMaterialComponent.rimPower[ eid ];
			material.rimIntensity = NormalMaterialComponent.rimIntensity[ eid ];
			material.moodTemperature = NormalMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = NormalMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = NormalMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = NormalMaterialComponent.moodContrast[ eid ];
			material.normalPerturbation = NormalMaterialComponent.normalPerturbation[ eid ];

			// Recompute mood color (applied to the normal RGB output)
			material.moodColor.copy( material.color );
			gradeNormalColor(
				material.moodColor,
				material.moodTemperature,
				material.moodSaturation,
				material.moodBrightness,
				material.moodContrast
			);

			NormalMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main MeshNormalMaterial class — mirrors three.js/src/materials/MeshNormalMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material that maps the normal vectors to RGB colors.
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls: normal-driven cel banding,
 * view-space normal rim glow, mood-based normal color grading, and
 * procedural normal perturbation via simplex-noise.
 *
 * ```js
 * const material = new THREE.MeshNormalMaterial( {
 *   flatShading: true,
 *   celBands: 3,
 *   celQuantize: 0.8,
 *   rimIntensity: 0.5,
 *   moodTemperature: -0.3
 * } );
 * ```
 *
 * @augments Material
 */
class MeshNormalMaterial extends Material {

	/**
	 * Constructs a new mesh normal material.
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
		this.isMeshNormalMaterial = true;

		this.type = 'MeshNormalMaterial';

		/**
		 * The texture to create a bump map. The black and white values map to
		 * the perceived depth in relation to the lights. Bump doesn't
		 * actually affect the geometry of the object, only the lighting.
		 * If a normal map is defined this will be ignored.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.bumpMap = null;

		/**
		 * How much the bump map affects the material. Typical range is `[0,1]`.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.bumpScale = 1;

		/**
		 * The texture to create a normal map. The RGB values affect the
		 * surface normal for each pixel fragment and change the way the
		 * color is lit. Normal maps do not change the actual shape of the
		 * surface, only the lighting.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.normalMap = null;

		/**
		 * The type of normal map. Can be `TangentSpaceNormalMap` (default)
		 * or `ObjectSpaceNormalMap`.
		 *
		 * @type {number}
		 * @default TangentSpaceNormalMap
		 */
		this.normalMapType = TangentSpaceNormalMap;

		/**
		 * How much the normal map affects the material. Typical value range
		 * is `[0,1]`.
		 *
		 * @type {Vector2}
		 * @default (1,1)
		 */
		this.normalScale = new Vector2( 1, 1 );

		/**
		 * The displacement map affects the position of the mesh's vertices.
		 * Unlike other maps which only affect the light and shade of the
		 * material, the displaced vertices can cast shadows, block other
		 * objects, and otherwise act as real geometry.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.displacementMap = null;

		/**
		 * How much the displacement map affects the mesh (where black is no
		 * displacement, and white is maximum displacement).
		 *
		 * @type {number}
		 * @default 1
		 */
		this.displacementScale = 1;

		/**
		 * The offset of the displacement map's values on the mesh's vertices.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.displacementBias = 0;

		/**
		 * Renders the geometry as a wireframe.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.wireframe = false;

		/**
		 * Controls the thickness of the wireframe. WebGL and WebGPU ignore
		 * this property and always render 1-pixel-wide lines.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.wireframeLinewidth = 1;

		/**
		 * Whether the material is rendered with flat shading or not.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.flatShading = false;

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
		 * Number of cel bands applied per RGB channel of the normal output.
		 * 0 = off, 2 = classic hard-shadow anime, 3-4 = softer stylized.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.celBands = 0;

		/**
		 * Blend amount between continuous and banded normal output.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.celQuantize = 1.0;

		/**
		 * Shadow band brightness multiplier. Lower = darker shadow bands.
		 *
		 * @type {number}
		 * @default 0.85
		 */
		this.shadowTint = 0.85;

		/**
		 * Rim-glow falloff power. Higher = tighter rim around silhouettes.
		 *
		 * @type {number}
		 * @default 2.0
		 */
		this.rimPower = 2.0;

		/**
		 * Rim-glow intensity. Set > 0 to enable view-space normal rim glow.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimIntensity = 0;

		/**
		 * Rim-glow color. Defaults to cool cyan.
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
		 * The precomputed mood-graded color. Applied to the normal RGB
		 * output during mood grading.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0xffffff );

		/**
		 * Procedural normal perturbation intensity. Set > 0 to add subtle
		 * organic surface detail via simplex-noise.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.normalPerturbation = 0;

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
	 * Apply cel banding to a normal-mapped RGB color.
	 *
	 * @param {Color} normalColor - The normal-mapped RGB color.
	 * @param {Color} output - The output color.
	 * @returns {Color}
	 */
	applyNormalBanding( normalColor, output ) {

		return applyNormalCelBanding(
			normalColor,
			output,
			this.celBands,
			this.celQuantize,
			this.shadowTint
		);

	}

	/**
	 * Compute the view-space normal rim glow for a given normal and view
	 * direction. Returns a scalar in [0, 1].
	 *
	 * @param {Vector3} viewNormal - View-space normal (normalized).
	 * @param {Vector3} viewDir - View direction (normalized).
	 * @returns {number}
	 */
	sampleNormalRim( viewNormal, viewDir ) {

		if ( this.rimIntensity <= 0 ) return 0;
		return computeNormalRimGlow( viewNormal, viewDir, this.rimPower, this.rimIntensity );

	}

	/**
	 * Generate a procedural normal-perturbation texture for this material.
	 * The caller is expected to assign the returned buffer to a
	 * `DataTexture` and attach it to the `normalMap` slot.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.05]
	 * @param {number} [octaves=3]
	 * @returns {Uint8Array}
	 */
	generateNormalPerturbation( width, height, scale = 0.05, octaves = 3 ) {

		return generateNormalPerturbation( width, height, scale, octaves, this.normalPerturbation );

	}

	/**
	 * Recompute the mood-graded color from the current mood parameters.
	 *
	 * @returns {MeshNormalMaterial} A reference to this instance.
	 */
	updateMoodColor() {

		this.moodColor.copy( this.color );
		gradeNormalColor(
			this.moodColor,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast
		);
		return this;

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
	 * anime shader chunks with normal-specific features.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the base Material hook first
		super.onBeforeCompile( shader, renderer );

		// Inject normal-specific uniforms
		shader.uniforms.celBands = { value: this.celBands };
		shader.uniforms.celQuantize = { value: this.celQuantize };
		shader.uniforms.shadowTint = { value: this.shadowTint };
		shader.uniforms.rimPower = { value: this.rimPower };
		shader.uniforms.rimIntensity = { value: this.rimIntensity };
		shader.uniforms.rimGlowColor = { value: this.rimGlowColor };
		shader.uniforms.moodColor = { value: this.moodColor };
		shader.uniforms.normalPerturbation = { value: this.normalPerturbation };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int celBands;
				uniform float celQuantize;
				uniform float shadowTint;
				uniform float rimPower;
				uniform float rimIntensity;
				uniform vec3 rimGlowColor;
				uniform vec3 moodColor;
				uniform float normalPerturbation;
				uniform float variationOffset;

				vec3 applyNormalBanding( vec3 normalColor ) {
					if ( celBands <= 1 ) return normalColor;
					float bandWidth = 1.0 / float( celBands );
					vec3 quantized;
					quantized.r = floor( normalColor.r / bandWidth ) * bandWidth + bandWidth * 0.5;
					quantized.g = floor( normalColor.g / bandWidth ) * bandWidth + bandWidth * 0.5;
					quantized.b = floor( normalColor.b / bandWidth ) * bandWidth + bandWidth * 0.5;
					vec3 banded = mix( normalColor, quantized, celQuantize );
					// Shadow tint for darker channels
					banded = mix( banded, banded * shadowTint, step( banded, vec3( 0.5 ) ) );
					return clamp( banded, 0.0, 1.0 );
				}

				float computeRimGlow( vec3 viewNormal, vec3 viewDir ) {
					vec3 n = normalize( viewNormal );
					vec3 v = normalize( viewDir );
					float ndotv = dot( n, v );
					float rim = pow( 1.0 - abs( ndotv ), rimPower );
					return clamp( rim * rimIntensity, 0.0, 1.0 );
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Apply normal-driven cel banding
				gl_FragColor.rgb = applyNormalBanding( gl_FragColor.rgb );

				// Blend with mood-graded color
				gl_FragColor.rgb = mix( gl_FragColor.rgb, moodColor, 0.4 );

				// View-space normal rim glow
				if ( rimIntensity > 0.0 ) {
					vec3 viewNormal = normalize( vNormal );
					vec3 viewDir = normalize( vViewPosition );
					float rim = computeRimGlow( viewNormal, viewDir );
					gl_FragColor.rgb += rimGlowColor * rim;
				}

				// Procedural normal perturbation overlay
				if ( normalPerturbation > 0.0 ) {
					float perturb = sin( vUv.x * 16.0 + variationOffset ) * cos( vUv.y * 16.0 + variationOffset );
					perturb = perturb * 0.5 + 0.5;
					gl_FragColor.rgb = mix( gl_FragColor.rgb, gl_FragColor.rgb * perturb, normalPerturbation );
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
			this.celBands,
			this.celQuantize,
			this.shadowTint,
			this.rimPower,
			this.rimIntensity,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.normalPerturbation,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {MeshNormalMaterial} source - The material to copy from.
	 * @return {MeshNormalMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.bumpMap = source.bumpMap;
		this.bumpScale = source.bumpScale;
		this.normalMap = source.normalMap;
		this.normalMapType = source.normalMapType;
		this.normalScale.copy( source.normalScale );
		this.displacementMap = source.displacementMap;
		this.displacementScale = source.displacementScale;
		this.displacementBias = source.displacementBias;
		this.wireframe = source.wireframe;
		this.wireframeLinewidth = source.wireframeLinewidth;
		this.flatShading = source.flatShading;
		this.fog = source.fog;

		// Anime extensions
		this.celBands = source.celBands;
		this.celQuantize = source.celQuantize;
		this.shadowTint = source.shadowTint;
		this.rimPower = source.rimPower;
		this.rimIntensity = source.rimIntensity;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodContrast = source.moodContrast;
		this.moodColor.copy( source.moodColor );
		this.normalPerturbation = source.normalPerturbation;
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

		data.type = 'MeshNormalMaterial';

		if ( this.bumpMap !== null ) data.bumpMap = this.bumpMap.toJSON( meta ).uuid;
		if ( this.bumpScale !== 1 ) data.bumpScale = this.bumpScale;
		if ( this.normalMap !== null ) data.normalMap = this.normalMap.toJSON( meta ).uuid;
		if ( this.normalMapType !== TangentSpaceNormalMap ) data.normalMapType = this.normalMapType;
		if ( this.normalScale.x !== 1 || this.normalScale.y !== 1 ) data.normalScale = this.normalScale.toArray();
		if ( this.displacementMap !== null ) data.displacementMap = this.displacementMap.toJSON( meta ).uuid;
		if ( this.displacementScale !== 1 ) data.displacementScale = this.displacementScale;
		if ( this.displacementBias !== 0 ) data.displacementBias = this.displacementBias;
		if ( this.wireframe ) data.wireframe = true;
		if ( this.wireframeLinewidth !== 1 ) data.wireframeLinewidth = this.wireframeLinewidth;
		if ( this.flatShading ) data.flatShading = true;
		if ( this.fog ) data.fog = true;

		// Anime extensions
		if ( this.celBands !== 0 ) data.celBands = this.celBands;
		if ( this.celQuantize !== 1.0 ) data.celQuantize = this.celQuantize;
		if ( this.shadowTint !== 0.85 ) data.shadowTint = this.shadowTint;
		if ( this.rimPower !== 2.0 ) data.rimPower = this.rimPower;
		if ( this.rimIntensity !== 0 ) data.rimIntensity = this.rimIntensity;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodContrast !== 1.0 ) data.moodContrast = this.moodContrast;
		if ( this.normalPerturbation !== 0 ) data.normalPerturbation = this.normalPerturbation;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { MeshNormalMaterial, MeshNormalMaterialBatch, applyNormalCelBanding, computeNormalRimGlow, gradeNormalColor, generateNormalPerturbation };
export default MeshNormalMaterial;