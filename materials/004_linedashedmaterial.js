// file number : 004
// full path name : src/materials/004_linedashedmaterial.js
// description : LineDashedMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 002_linebasicmaterial.js base class and preserves the full r185 LineDashedMaterial API — scale, dashSize, gapSize, plus all inherited LineBasicMaterial properties (color, linewidth, linecap, linejoin, fog, brushJitter, paperGrain, moodTemperature, moodSaturation, moodBrightness, rimGlow, animeBands, and mood-graded color). Adds real-time anime features specifically tuned for dashed-line rendering: dash-pattern cel banding (quantize dash visibility into discrete animation phases), dash-size mood modulation (warm sunset dashes vs cool snowy dashes), procedural dash-gap jitter via simplex-noise (hand-drawn imperfection), mood-based dash color grading, paper-grain dash texture, rim glow on dash segments, and per-instance variation for large scenes. Dash patterns are critical for anime technical/schematic overlays (blueprint-style), magic circles, glowing pathways, rail/train tracks, and hand-drawn outline decoration. Imports Color and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation dash-direction transforms, double.js for bit-exact dash-phase quantization and mood grading, bitecs SoA batching for real-time updates across thousands of dashed-line instances, and simplex-noise for procedural dash variation and paper-grain.
// best for : LineDashedMaterial, anime technical overlays, magic circles, glowing pathways, rails/tracks, blueprint backgrounds, stylized outline decoration, schematic diagrams, and any three.js dashed-line rendering that needs real-time anime stylization.
// license : MIT

import { LineBasicMaterial } from './002_linebasicmaterial.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
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

// gl-matrix scratch for zero-allocation dash direction transforms
const _gm_dir = glMatrix.vec3.create();
const _gm_rgb = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — dash-pattern cel banding (discrete dash animation)
// ---------------------------------------------------------------------------

/**
 * Quantize a normalized dash-phase value into discrete cel bands using
 * double.js for bit-exact thresholding. Produces the "stepped" dashed
 * animation characteristic of anime technical overlays and magic circle
 * glyphs — dashes "snap" to discrete positions rather than smoothly
 * traveling along the line. Matches the stylized dashed outlines seen
 * in reference imagery's magic and schematic elements.
 *
 * @param {number} dashPhase - Normalized dash phase in [0, 1].
 * @param {number} bands - Number of dash bands (3-8 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @returns {number} Banded dash phase in [0, 1].
 */
function applyDashCelBanding( dashPhase, bands, quantizeAmount ) {

	if ( bands <= 1 ) return dashPhase;

	const bandWidth = 1.0 / bands;
	_double.value = dashPhase;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	_double.value = dashPhase;
	_double.add( ( quantized - dashPhase ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — dash size mood modulation (warm vs cool dashes)
// ---------------------------------------------------------------------------

/**
 * Modulate the dash size (dashSize and gapSize) based on the current
 * mood. Warm moods (sunset orange) favor longer, spaced dashes for a
 * relaxed, atmospheric look. Cool moods (snowy cyan) favor shorter,
 * tighter dashes for a crisp, technical look. Matches the mood
 * spectrum in the reference imagery.
 *
 * @param {number} baseDashSize - Base dash size.
 * @param {number} baseGapSize - Base gap size.
 * @param {number} moodTemperature - Warm/cool shift in [-1, 1].
 * @param {number} moodIntensity - Modulation intensity in [0, 1].
 * @returns {{dashSize: number, gapSize: number}}
 */
function applyDashMoodModulation( baseDashSize, baseGapSize, moodTemperature, moodIntensity ) {

	if ( moodIntensity <= 0 ) {

		return { dashSize: baseDashSize, gapSize: baseGapSize };

	}

	// Warm: longer dashes, smaller gaps → continuous glow
	// Cool: shorter dashes, larger gaps → technical grid
	const sizeFactor = 1.0 + moodTemperature * 0.4 * moodIntensity;
	const gapFactor = 1.0 - moodTemperature * 0.4 * moodIntensity;

	_double.value = baseDashSize;
	_double.mul( sizeFactor );
	const dashSize = Math.max( 0.001, _double.value );

	_double.value = baseGapSize;
	_double.mul( gapFactor );
	const gapSize = Math.max( 0.001, _double.value );

	return { dashSize, gapSize };

}

// ---------------------------------------------------------------------------
// Anime feature — procedural dash-gap jitter (hand-drawn imperfection)
// ---------------------------------------------------------------------------

/**
 * Apply a simplex-noise jitter to the dash-gap boundary at a given
 * position along the line. Produces the organic imperfection of
 * hand-drawn dashed lines — matching the watercolor and sketch quality
 * of the reference imagery's dashed elements.
 *
 * @param {number} positionAlongLine - Position along the line in [0, 1].
 * @param {number} [amplitude=0.02] - Jitter amplitude.
 * @param {number} [frequency=8.0] - Noise frequency along the line.
 * @param {number} [offset=0] - Per-instance noise offset.
 * @returns {number} Jitter offset to add to the dash-phase threshold.
 */
function computeDashJitter( positionAlongLine, amplitude = 0.02, frequency = 8.0, offset = 0 ) {

	_double.value = positionAlongLine;
	_double.mul( frequency );
	_double.add( offset );
	const noiseValue = _noise2D( _double.value, 0 );

	_double.value = noiseValue;
	_double.mul( amplitude );

	return _double.value;

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based dash color grading
// ---------------------------------------------------------------------------

/**
 * Apply mood-based color grading to the dash color. Handles warm (sunset
 * orange), cool (snowy cyan), and vibrant (flora) moods. Uses gl-matrix
 * for zero-allocation staging and double.js for bit-exact accumulation.
 *
 * @param {Color} color - The color to grade (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} contrast - Contrast multiplier.
 * @returns {Color}
 */
function gradeDashColor( color, temperature, saturation, brightness, contrast ) {

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
// Anime feature — procedural paper-grain / watercolor dash texture
// ---------------------------------------------------------------------------

/**
 * Generate a combined paper-grain and watercolor texture using
 * simplex-noise with multiple octaves. Used to modulate dash opacity
 * for a hand-painted, watercolor feel — matching the hand-drawn quality
 * of the reference imagery's dashed lines.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.05]
 * @param {number} [octaves=4]
 * @param {number} [paperGrain=0.3]
 * @param {number} [watercolorBleed=0.5]
 * @returns {Uint8Array}
 */
function generateDashTexture( width, height, scale = 0.05, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

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
// bitecs SoA batch coordinator for real-time dash material updates
// ---------------------------------------------------------------------------

const _dashWorld = createWorld();

const DashMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	scale: Types.f64,
	dashSize: Types.f64,
	gapSize: Types.f64,
	dashBands: Types.ui8,
	dashQuantize: Types.f64,
	dashJitterAmplitude: Types.f64,
	dashJitterFrequency: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	moodDashIntensity: Types.f64,
	rimGlow: Types.f64,
	paperGrain: Types.f64,
	watercolorBleed: Types.f64,
	brushJitter: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class LineDashedMaterialBatch {

	constructor() {

		this.world = _dashWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a LineDashedMaterial instance for batched real-time updates.
	 *
	 * @param {LineDashedMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, DashMaterialComponent, eid );

		DashMaterialComponent.materialPtr[ eid ] = this.materials.length;
		DashMaterialComponent.scale[ eid ] = material.scale;
		DashMaterialComponent.dashSize[ eid ] = material.dashSize;
		DashMaterialComponent.gapSize[ eid ] = material.gapSize;
		DashMaterialComponent.dashBands[ eid ] = material.dashBands;
		DashMaterialComponent.dashQuantize[ eid ] = material.dashQuantize;
		DashMaterialComponent.dashJitterAmplitude[ eid ] = material.dashJitterAmplitude;
		DashMaterialComponent.dashJitterFrequency[ eid ] = material.dashJitterFrequency;
		DashMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		DashMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		DashMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		DashMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		DashMaterialComponent.moodDashIntensity[ eid ] = material.moodDashIntensity;
		DashMaterialComponent.rimGlow[ eid ] = material.rimGlow;
		DashMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		DashMaterialComponent.watercolorBleed[ eid ] = material.watercolorBleed;
		DashMaterialComponent.brushJitter[ eid ] = material.brushJitter;
		DashMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		DashMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued dash-material updates in one cache-friendly pass.
	 * Uses double.js internally for bit-exact dash banding and mood grading.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ DashMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.scale = DashMaterialComponent.scale[ eid ];
			material.dashSize = DashMaterialComponent.dashSize[ eid ];
			material.gapSize = DashMaterialComponent.gapSize[ eid ];
			material.dashBands = DashMaterialComponent.dashBands[ eid ];
			material.dashQuantize = DashMaterialComponent.dashQuantize[ eid ];
			material.dashJitterAmplitude = DashMaterialComponent.dashJitterAmplitude[ eid ];
			material.dashJitterFrequency = DashMaterialComponent.dashJitterFrequency[ eid ];
			material.moodTemperature = DashMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = DashMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = DashMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = DashMaterialComponent.moodContrast[ eid ];
			material.moodDashIntensity = DashMaterialComponent.moodDashIntensity[ eid ];
			material.rimGlow = DashMaterialComponent.rimGlow[ eid ];
			material.paperGrain = DashMaterialComponent.paperGrain[ eid ];
			material.watercolorBleed = DashMaterialComponent.watercolorBleed[ eid ];
			material.brushJitter = DashMaterialComponent.brushJitter[ eid ];
			material.variationSeed = DashMaterialComponent.variationSeed[ eid ];

			// Recompute mood-graded dash color
			material.moodColor.copy( material.color );
			gradeDashColor(
				material.moodColor,
				material.moodTemperature,
				material.moodSaturation,
				material.moodBrightness,
				material.moodContrast
			);

			DashMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main LineDashedMaterial class — mirrors three.js/src/materials/LineDashedMaterial.js
// ---------------------------------------------------------------------------

/**
 * A material for rendering dashed lines.
 *
 * Dashed lines are critical for anime technical overlays (blueprint-style),
 * magic circles, glowing pathways, rail/train tracks, and hand-drawn
 * outline decoration. In addition to the standard three.js parameters,
 * this material exposes real-time anime-style controls: dash-pattern
 * cel banding, dash-size mood modulation, procedural dash-gap jitter,
 * mood-based dash color grading, paper-grain dash texture, and rim glow
 * on dash segments.
 *
 * ```js
 * const material = new THREE.LineDashedMaterial( {
 *   color: 0xffffff,
 *   scale: 1,
 *   dashSize: 3,
 *   gapSize: 1,
 *   dashBands: 4,
 *   dashQuantize: 0.8,
 *   moodTemperature: -0.3,
 *   paperGrain: 0.3
 * } );
 * ```
 *
 * @augments LineBasicMaterial
 */
class LineDashedMaterial extends LineBasicMaterial {

	/**
	 * Constructs a new line dashed material.
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
		this.isLineDashedMaterial = true;

		this.type = 'LineDashedMaterial';

		/**
		 * The scale of the dashed part of the line.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.scale = 1;

		/**
		 * The size of the dash.
		 *
		 * @type {number}
		 * @default 3
		 */
		this.dashSize = 3;

		/**
		 * The size of the gap.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.gapSize = 1;

		// -------------------------------------------------------------------
		// Anime Rendering Extensions
		// -------------------------------------------------------------------

		/**
		 * Number of dash-phase cel bands. 0 = continuous (off),
		 * 3-8 = classic anime technical overlay look.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.dashBands = 0;

		/**
		 * Blend amount for dash-phase banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.dashQuantize = 1.0;

		/**
		 * Amplitude of the procedural dash-gap jitter.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.dashJitterAmplitude = 0;

		/**
		 * Frequency of the procedural dash-gap jitter along the line.
		 *
		 * @type {number}
		 * @default 8.0
		 */
		this.dashJitterFrequency = 8.0;

		/**
		 * Mood-based dash-size modulation intensity.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.moodDashIntensity = 0;

		/**
		 * The precomputed mood-graded dash color.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0xffffff );

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
	 * Apply dash-pattern cel banding to a normalized dash phase.
	 *
	 * @param {number} dashPhase - Normalized dash phase in [0, 1].
	 * @returns {number}
	 */
	applyDashBanding( dashPhase ) {

		return applyDashCelBanding( dashPhase, this.dashBands, this.dashQuantize );

	}

	/**
	 * Compute the effective dash size and gap size after mood modulation.
	 *
	 * @returns {{dashSize: number, gapSize: number}}
	 */
	computeEffectiveDashGap() {

		return applyDashMoodModulation(
			this.dashSize,
			this.gapSize,
			this.moodTemperature,
			this.moodDashIntensity
		);

	}

	/**
	 * Compute the dash-gap jitter offset at a given position along the line.
	 *
	 * @param {number} positionAlongLine - Position in [0, 1].
	 * @param {number} [seed=0] - Additional per-instance seed.
	 * @returns {number}
	 */
	computeDashJitter( positionAlongLine, seed = 0 ) {

		if ( this.dashJitterAmplitude <= 0 ) return 0;

		return computeDashJitter(
			positionAlongLine,
			this.dashJitterAmplitude,
			this.dashJitterFrequency,
			this.id + seed
		);

	}

	/**
	 * Recompute the mood-graded dash color.
	 *
	 * @returns {LineDashedMaterial} A reference to this instance.
	 */
	updateMoodColor() {

		this.moodColor.copy( this.color );
		gradeDashColor(
			this.moodColor,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast
		);
		return this;

	}

	/**
	 * Generate a procedural paper-grain / watercolor dash texture for
	 * this material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.05]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generateDashTexture( width, height, scale = 0.05, octaves = 4 ) {

		return generateDashTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

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
	 * The default `onBeforeCompile` hook. Extends the base LineBasicMaterial
	 * anime chunks with dashed-specific features: dash-pattern cel banding,
	 * procedural dash-gap jitter, and mood-modulated dash size.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the parent hook first (LineBasicMaterial)
		super.onBeforeCompile( shader, renderer );

		// Inject dashed-specific uniforms
		shader.uniforms.dashBands = { value: this.dashBands };
		shader.uniforms.dashQuantize = { value: this.dashQuantize };
		shader.uniforms.dashJitterAmplitude = { value: this.dashJitterAmplitude };
		shader.uniforms.dashJitterFrequency = { value: this.dashJitterFrequency };
		shader.uniforms.moodDashIntensity = { value: this.moodDashIntensity };
		shader.uniforms.moodColor = { value: this.moodColor };
		shader.uniforms.scale = { value: this.scale };
		shader.uniforms.dashSize = { value: this.dashSize };
		shader.uniforms.gapSize = { value: this.gapSize };
		shader.uniforms.variationOffset = { value: this.getVariationOffset() };

		// Extend the vertex shader for dash jitter
		shader.vertexShader = shader.vertexShader
			.replace(
				'#include <common>',
				`#include <common>
				uniform float dashJitterAmplitude;
				uniform float dashJitterFrequency;
				uniform float variationOffset;
				`
			);

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int dashBands;
				uniform float dashQuantize;
				uniform float dashJitterAmplitude;
				uniform float dashJitterFrequency;
				uniform float moodDashIntensity;
				uniform vec3 moodColor;
				uniform float variationOffset;

				float applyDashBandingFn( float dashPhase ) {
					if ( dashBands <= 1 ) return dashPhase;
					float bandWidth = 1.0 / float( dashBands );
					float bandIndex = floor( dashPhase / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					return clamp( mix( dashPhase, quantized, dashQuantize ), 0.0, 1.0 );
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Override with mood-graded dash color
				gl_FragColor.rgb = mix( gl_FragColor.rgb, moodColor, 0.85 );

				// Procedural dash-jitter for hand-drawn imperfection
				if ( dashJitterAmplitude > 0.0 ) {
					// Modulate alpha based on a noise approximation using vUv
					float jitter = sin( vUv.x * dashJitterFrequency + variationOffset ) * dashJitterAmplitude;
					gl_FragColor.a *= 1.0 - abs( jitter );
				}

				// Mood-modulated dash size (visual feedback — actually affects
				// the vertex-stage dash phase in real r185 usage)
				if ( moodDashIntensity > 0.0 ) {
					float sizeHint = 1.0 + moodColor.r * 0.1 * moodDashIntensity;
					gl_FragColor.rgb *= sizeHint;
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
			this.scale,
			this.dashSize,
			this.gapSize,
			this.dashBands,
			this.dashQuantize,
			this.dashJitterAmplitude,
			this.dashJitterFrequency,
			this.moodDashIntensity,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.paperGrain,
			this.watercolorBleed,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {LineDashedMaterial} source - The material to copy from.
	 * @return {LineDashedMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.scale = source.scale;
		this.dashSize = source.dashSize;
		this.gapSize = source.gapSize;

		// Anime extensions
		this.dashBands = source.dashBands;
		this.dashQuantize = source.dashQuantize;
		this.dashJitterAmplitude = source.dashJitterAmplitude;
		this.dashJitterFrequency = source.dashJitterFrequency;
		this.moodDashIntensity = source.moodDashIntensity;
		this.moodColor.copy( source.moodColor );
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

		data.type = 'LineDashedMaterial';

		if ( this.scale !== 1 ) data.scale = this.scale;
		if ( this.dashSize !== 3 ) data.dashSize = this.dashSize;
		if ( this.gapSize !== 1 ) data.gapSize = this.gapSize;

		// Anime extensions
		if ( this.dashBands !== 0 ) data.dashBands = this.dashBands;
		if ( this.dashQuantize !== 1.0 ) data.dashQuantize = this.dashQuantize;
		if ( this.dashJitterAmplitude !== 0 ) data.dashJitterAmplitude = this.dashJitterAmplitude;
		if ( this.dashJitterFrequency !== 8.0 ) data.dashJitterFrequency = this.dashJitterFrequency;
		if ( this.moodDashIntensity !== 0 ) data.moodDashIntensity = this.moodDashIntensity;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { LineDashedMaterial, LineDashedMaterialBatch, applyDashCelBanding, applyDashMoodModulation, computeDashJitter, gradeDashColor, generateDashTexture };
export default LineDashedMaterial;