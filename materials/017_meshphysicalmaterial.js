// file number : 017
// full path name : src/materials/017_meshphysicalmaterial.js
// description : MeshPhysicalMaterial (three.js r185) rewritten as a high-performance ES module with deep anime-stylization integration. Extends the anime-enabled 010_meshstandardmaterial.js base class and preserves the full r185 MeshPhysicalMaterial API — clearcoat, clearcoatMap, clearcoatRoughness, clearcoatRoughnessMap, clearcoatNormalMap, clearcoatNormalScale, iridescence, iridescenceMap, iridescenceIOR, iridescenceThicknessRange, iridescenceThicknessMap, sheen, sheenColor, sheenColorMap, sheenRoughness, sheenRoughnessMap, transmission, transmissionMap, thickness, thicknessMap, attenuationDistance, attenuationColor, specularIntensity, specularIntensityMap, specularColor, specularColorMap, anisotropy, anisotropyRotation, anisotropyMap, dispersion, ior, reflectivity, plus all inherited MeshStandardMaterial and Material properties. MeshPhysicalMaterial's extended parameter surface (clearcoat, sheen, iridescence, transmission, anisotropy) maps beautifully to the classic anime material taxonomy: clearcoat becomes glossy anime lips/hair/nail polish, sheen becomes velvet/fur fabric for anime clothing, iridescence becomes magical rainbow shimmer for spells and wings, transmission becomes stylized glass and gemstone, and anisotropy becomes brushed metal and shiny anime hair streaks. Adds real-time anime features specifically tuned for the extended PBR surface: clearcoat cel banding (hard-edged glossy anime highlights), sheen cel banding (flat velvet shading for fabric), iridescence cel banding (rainbow anime shimmer with configurable band separation), transmission cel banding (stylized colored glass), anisotropy toon shaping (elongated hair streaks), mood-based per-channel grading, procedural paper-grain and watercolor texture variation via simplex-noise, rim glow on silhouettes, and per-instance variation for crowds. Imports Color, Euler, Vector2, Vector3, and other math classes strictly from threejs_new01 math, and uses gl-matrix for zero-allocation clearcoat/sheen/iridescence vector transforms, double.js for bit-exact cel-band thresholding and HDR mood grading across all extended PBR channels, bitecs SoA batching for real-time updates across thousands of anime characters and advanced PBR props, and simplex-noise for procedural texture variation.
// best for : MeshPhysicalMaterial, anime character hair/lips/nails (clearcoat), anime fabric/fur (sheen), anime magical effects and fairy wings (iridescence), anime glass and gemstones (transmission), brushed anime metal (anisotropy), and any three.js advanced PBR mesh that needs real-time anime stylization.
// license : MIT

import { MeshStandardMaterial } from './010_meshstandardmaterial.js';
import { Color } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/018_Color.js';
import { ColorManagement } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/017_ColorManagement.js';
import { Euler } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/008_Euler.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
import {
	NormalBlending,
	FrontSide,
	TangentSpaceNormalMap,
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

// gl-matrix scratch for zero-allocation PBR vector transforms
const _gm_rgb = glMatrix.vec3.create();
const _gm_normal = glMatrix.vec3.create();
const _gm_view = glMatrix.vec3.create();
const _gm_half = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Anime feature — clearcoat cel banding (glossy anime highlights)
// ---------------------------------------------------------------------------

/**
 * Quantize the clearcoat intensity into discrete cel bands using
 * double.js for bit-exact thresholding. Clearcoat produces the sharp,
 * specular "top layer" shine used on anime hair, lips, nails, and glass
 * (reference images 2, 3, 6, 7). Cel-banding produces the classic
 * hard-edged glossy highlight of traditional anime ink.
 *
 * @param {number} clearcoatIntensity - Raw clearcoat reflection in [0, 1].
 * @param {number} bands - Number of clearcoat bands (2-3 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @param {number} [threshold=0.5] - Minimum intensity to trigger the first band.
 * @param {number} [softness=0.05] - Edge softness in [0, 0.3].
 * @returns {number} Banded clearcoat intensity in [0, 1].
 */
function applyClearcoatCelBanding( clearcoatIntensity, bands, quantizeAmount, threshold = 0.5, softness = 0.05 ) {

	if ( bands <= 1 || clearcoatIntensity < threshold ) return 0;

	_double.value = clearcoatIntensity;
	_double.sub( threshold );
	_double.div( 1.0 - threshold );
	const normalized = Math.max( 0, Math.min( 1, _double.value ) );

	const bandWidth = 1.0 / bands;
	_double.value = normalized;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	const distToBoundary = Math.abs( normalized - bandIndex * bandWidth - bandWidth * 0.5 );
	const softFactor = Math.max( 0, Math.min( 1, distToBoundary / ( bandWidth * 0.5 ) ) );
	const softQuantized = quantized + ( normalized - quantized ) * ( 1 - softFactor ) * softness;

	_double.value = normalized;
	_double.add( ( softQuantized - normalized ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — sheen cel banding (flat velvet shading for fabric)
// ---------------------------------------------------------------------------

/**
 * Quantize the sheen intensity into discrete cel bands using double.js
 * for bit-exact thresholding. Sheen is the retroreflective fabric shading
 * used on anime velvet, fur, and embroidered costumes — quantizing it
 * produces the flat velvet shading of hand-painted anime character art.
 *
 * @param {number} sheenIntensity - Raw sheen intensity in [0, 1].
 * @param {number} bands - Number of sheen bands (2-4 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @param {number} [softness=0.15] - Edge softness.
 * @returns {number} Banded sheen intensity in [0, 1].
 */
function applySheenCelBanding( sheenIntensity, bands, quantizeAmount, softness = 0.15 ) {

	if ( bands <= 1 ) return sheenIntensity;

	const bandWidth = 1.0 / bands;
	_double.value = sheenIntensity;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	const distToBoundary = Math.abs( sheenIntensity - bandIndex * bandWidth - bandWidth * 0.5 );
	const softFactor = Math.max( 0, Math.min( 1, distToBoundary / ( bandWidth * 0.5 ) ) );
	const softQuantized = quantized + ( sheenIntensity - quantized ) * ( 1 - softFactor ) * softness;

	_double.value = sheenIntensity;
	_double.add( ( softQuantized - sheenIntensity ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — iridescence cel banding (rainbow anime shimmer)
// ---------------------------------------------------------------------------

/**
 * Quantize the iridescence intensity into discrete cel bands using
 * double.js for bit-exact thresholding. Iridescence is the rainbow
 * thin-film shimmer used on anime magical effects, fairy wings, and
 * butterfly wings — quantizing it separates the rainbow into discrete
 * color bands for a stylized sprite-sheet-like look.
 *
 * @param {number} iridescenceIntensity - Raw iridescence in [0, 1].
 * @param {number} bands - Number of iridescence bands (3-6 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @returns {number} Banded iridescence intensity in [0, 1].
 */
function applyIridescenceCelBanding( iridescenceIntensity, bands, quantizeAmount ) {

	if ( bands <= 1 ) return iridescenceIntensity;

	const bandWidth = 1.0 / bands;
	_double.value = iridescenceIntensity;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	_double.value = iridescenceIntensity;
	_double.add( ( quantized - iridescenceIntensity ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — transmission cel banding (stylized colored glass)
// ---------------------------------------------------------------------------

/**
 * Quantize the transmission (light passing through a transparent surface)
 * into discrete cel bands using double.js for bit-exact thresholding.
 * Produces the "flat glass" look of anime windows, bottles, and stained
 * glass — reference 1's cyan ice windows and reference 3's deep cyan
 * water both benefit from this stylized approach.
 *
 * @param {number} transmissionIntensity - Raw transmission in [0, 1].
 * @param {number} bands - Number of transmission bands (2-4 recommended).
 * @param {number} quantizeAmount - Blend between continuous and banded (0-1).
 * @returns {number} Banded transmission intensity in [0, 1].
 */
function applyTransmissionCelBanding( transmissionIntensity, bands, quantizeAmount ) {

	if ( bands <= 1 ) return transmissionIntensity;

	const bandWidth = 1.0 / bands;
	_double.value = transmissionIntensity;
	_double.div( bandWidth );
	const bandIndex = Math.floor( _double.value );
	_double.value = bandIndex;
	_double.mul( bandWidth );
	_double.add( bandWidth * 0.5 );
	const quantized = _double.value;

	_double.value = transmissionIntensity;
	_double.add( ( quantized - transmissionIntensity ) * quantizeAmount );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — anisotropy toon shaping (elongated hair streaks)
// ---------------------------------------------------------------------------

/**
 * Shape the anisotropic specular highlight into an elongated toon form.
 * Classic anime hair shine is an anisotropic "streak" that follows the
 * strand direction rather than a circular dot. This function remaps the
 * specular intensity based on the direction of the half-vector.
 *
 * @param {number} specularIntensity - Raw specular intensity in [0, 1].
 * @param {number} anisotropy - Elongation factor in [0, 1].
 * @param {number} angle - Anisotropy axis angle in radians.
 * @param {number} halfVecX - X component of the half vector (view-space).
 * @param {number} halfVecY - Y component of the half vector (view-space).
 * @returns {number} Shaped specular intensity in [0, 1].
 */
function applyAnisotropyToonShaping( specularIntensity, anisotropy, angle, halfVecX, halfVecY ) {

	if ( anisotropy <= 0 ) return specularIntensity;

	const cosA = Math.cos( - angle );
	const sinA = Math.sin( - angle );

	_double.value = halfVecX * cosA - halfVecY * sinA;
	const localX = _double.value;

	_double.value = halfVecX * sinA + halfVecY * cosA;
	const localY = _double.value;

	_double.value = localX * localX;
	_double.add( localY * localY / Math.max( 1e-4, 1 - anisotropy ) );
	const distSquared = _double.value;

	_double.value = 1.0 - Math.min( 1, distSquared );
	const shape = Math.pow( Math.max( 0, _double.value ), 0.5 );

	_double.value = specularIntensity;
	_double.mul( shape );

	return Math.max( 0, Math.min( 1, _double.value ) );

}

// ---------------------------------------------------------------------------
// Anime feature — mood-based color grading (extended PBR channels)
// ---------------------------------------------------------------------------

/**
 * Apply mood-based color grading to any extended PBR color channel using
 * gl-matrix for zero-allocation staging and double.js for bit-exact
 * accumulation.
 *
 * @param {Color} color - The color to grade (modified in place).
 * @param {number} temperature - Warm/cool shift in [-1, 1].
 * @param {number} saturation - Saturation multiplier.
 * @param {number} brightness - Brightness multiplier.
 * @param {number} contrast - Contrast multiplier.
 * @returns {Color}
 */
function gradePbrExtColor( color, temperature, saturation, brightness, contrast ) {

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
// Anime feature — procedural texture (paper grain + watercolor)
// ---------------------------------------------------------------------------

/**
 * Generate a combined paper-grain and watercolor texture using
 * simplex-noise with multiple octaves. Produces the hand-painted
 * texture characteristic of the reference imagery's snow, foliage,
 * and water surfaces.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} [scale=0.03]
 * @param {number} [octaves=4]
 * @param {number} [paperGrain=0.3]
 * @param {number} [watercolorBleed=0.5]
 * @returns {Uint8Array}
 */
function generatePbrExtTexture( width, height, scale = 0.03, octaves = 4, paperGrain = 0.3, watercolorBleed = 0.5 ) {

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
// bitecs SoA batch coordinator for real-time physical material updates
// ---------------------------------------------------------------------------

const _physicalWorld = createWorld();

const PhysicalMaterialComponent = defineComponent( {
	materialPtr: Types.ui32,
	clearcoatBands: Types.ui8,
	clearcoatQuantize: Types.f64,
	clearcoatThreshold: Types.f64,
	clearcoatSoftness: Types.f64,
	sheenBands: Types.ui8,
	sheenQuantize: Types.f64,
	sheenSoftness: Types.f64,
	iridescenceBands: Types.ui8,
	iridescenceQuantize: Types.f64,
	transmissionBands: Types.ui8,
	transmissionQuantize: Types.f64,
	anisotropy: Types.f64,
	anisotropyAngle: Types.f64,
	moodTemperature: Types.f64,
	moodSaturation: Types.f64,
	moodBrightness: Types.f64,
	moodContrast: Types.f64,
	rimGlow: Types.f64,
	paperGrain: Types.f64,
	watercolorBleed: Types.f64,
	variationSeed: Types.f64,
	dirty: Types.ui8
} );

class MeshPhysicalMaterialBatch {

	constructor() {

		this.world = _physicalWorld;
		this.materials = [];
		this.entities = [];

	}

	/**
	 * Register a MeshPhysicalMaterial instance for batched real-time updates.
	 *
	 * @param {MeshPhysicalMaterial} material
	 * @returns {number} entity id
	 */
	add( material ) {

		const eid = addEntity( this.world );
		addComponent( this.world, PhysicalMaterialComponent, eid );

		PhysicalMaterialComponent.materialPtr[ eid ] = this.materials.length;
		PhysicalMaterialComponent.clearcoatBands[ eid ] = material.clearcoatBands;
		PhysicalMaterialComponent.clearcoatQuantize[ eid ] = material.clearcoatQuantize;
		PhysicalMaterialComponent.clearcoatThreshold[ eid ] = material.clearcoatThreshold;
		PhysicalMaterialComponent.clearcoatSoftness[ eid ] = material.clearcoatSoftness;
		PhysicalMaterialComponent.sheenBands[ eid ] = material.sheenBands;
		PhysicalMaterialComponent.sheenQuantize[ eid ] = material.sheenQuantize;
		PhysicalMaterialComponent.sheenSoftness[ eid ] = material.sheenSoftness;
		PhysicalMaterialComponent.iridescenceBands[ eid ] = material.iridescenceBands;
		PhysicalMaterialComponent.iridescenceQuantize[ eid ] = material.iridescenceQuantize;
		PhysicalMaterialComponent.transmissionBands[ eid ] = material.transmissionBands;
		PhysicalMaterialComponent.transmissionQuantize[ eid ] = material.transmissionQuantize;
		PhysicalMaterialComponent.anisotropy[ eid ] = material.anisotropy;
		PhysicalMaterialComponent.anisotropyAngle[ eid ] = material.anisotropyAngle;
		PhysicalMaterialComponent.moodTemperature[ eid ] = material.moodTemperature;
		PhysicalMaterialComponent.moodSaturation[ eid ] = material.moodSaturation;
		PhysicalMaterialComponent.moodBrightness[ eid ] = material.moodBrightness;
		PhysicalMaterialComponent.moodContrast[ eid ] = material.moodContrast;
		PhysicalMaterialComponent.rimGlow[ eid ] = material.rimGlow;
		PhysicalMaterialComponent.paperGrain[ eid ] = material.paperGrain;
		PhysicalMaterialComponent.watercolorBleed[ eid ] = material.watercolorBleed;
		PhysicalMaterialComponent.variationSeed[ eid ] = material.variationSeed;
		PhysicalMaterialComponent.dirty[ eid ] = 0;

		this.materials.push( material );
		this.entities.push( eid );

		return eid;

	}

	/**
	 * Apply all queued physical-material updates in one cache-friendly
	 * pass. Uses double.js internally for bit-exact cel-band thresholding
	 * and mood grading across all extended PBR channels.
	 */
	process() {

		const entities = this.entities;

		for ( let i = 0, l = entities.length; i < l; i ++ ) {

			const eid = entities[ i ];
			const material = this.materials[ PhysicalMaterialComponent.materialPtr[ eid ] ];
			if ( ! material ) continue;

			material.clearcoatBands = PhysicalMaterialComponent.clearcoatBands[ eid ];
			material.clearcoatQuantize = PhysicalMaterialComponent.clearcoatQuantize[ eid ];
			material.clearcoatThreshold = PhysicalMaterialComponent.clearcoatThreshold[ eid ];
			material.clearcoatSoftness = PhysicalMaterialComponent.clearcoatSoftness[ eid ];
			material.sheenBands = PhysicalMaterialComponent.sheenBands[ eid ];
			material.sheenQuantize = PhysicalMaterialComponent.sheenQuantize[ eid ];
			material.sheenSoftness = PhysicalMaterialComponent.sheenSoftness[ eid ];
			material.iridescenceBands = PhysicalMaterialComponent.iridescenceBands[ eid ];
			material.iridescenceQuantize = PhysicalMaterialComponent.iridescenceQuantize[ eid ];
			material.transmissionBands = PhysicalMaterialComponent.transmissionBands[ eid ];
			material.transmissionQuantize = PhysicalMaterialComponent.transmissionQuantize[ eid ];
			material.anisotropy = PhysicalMaterialComponent.anisotropy[ eid ];
			material.anisotropyAngle = PhysicalMaterialComponent.anisotropyAngle[ eid ];
			material.moodTemperature = PhysicalMaterialComponent.moodTemperature[ eid ];
			material.moodSaturation = PhysicalMaterialComponent.moodSaturation[ eid ];
			material.moodBrightness = PhysicalMaterialComponent.moodBrightness[ eid ];
			material.moodContrast = PhysicalMaterialComponent.moodContrast[ eid ];
			material.rimGlow = PhysicalMaterialComponent.rimGlow[ eid ];
			material.paperGrain = PhysicalMaterialComponent.paperGrain[ eid ];
			material.watercolorBleed = PhysicalMaterialComponent.watercolorBleed[ eid ];
			material.variationSeed = PhysicalMaterialComponent.variationSeed[ eid ];

			// Recompute mood-graded colors for all extended channels
			material.moodColor.copy( material.color );
			gradePbrExtColor( material.moodColor, material.moodTemperature, material.moodSaturation, material.moodBrightness, material.moodContrast );

			material.moodEmissive.copy( material.emissive );
			gradePbrExtColor( material.moodEmissive, material.moodTemperature, material.moodSaturation, material.moodBrightness, material.moodContrast );

			material.moodSheenColor.copy( material.sheenColor );
			gradePbrExtColor( material.moodSheenColor, material.moodTemperature, material.moodSaturation, material.moodBrightness, material.moodContrast );

			material.moodAttenuationColor.copy( material.attenuationColor );
			gradePbrExtColor( material.moodAttenuationColor, material.moodTemperature, material.moodSaturation, material.moodBrightness, material.moodContrast );

			material.moodSpecularColor.copy( material.specularColor );
			gradePbrExtColor( material.moodSpecularColor, material.moodTemperature, material.moodSaturation, material.moodBrightness, material.moodContrast );

			PhysicalMaterialComponent.dirty[ eid ] = 1;

		}

	}

}

// ---------------------------------------------------------------------------
// Main MeshPhysicalMaterial class — mirrors three.js/src/materials/MeshPhysicalMaterial.js
// ---------------------------------------------------------------------------

/**
 * A standard physically based material, using Metallic-Roughness workflow.
 * Extends {@link MeshStandardMaterial} with advanced PBR properties:
 * clearcoat, iridescence, sheen, transmission, thickness, attenuation,
 * specular intensity/color, and anisotropy.
 *
 * In addition to the standard three.js parameters, this material exposes
 * a rich set of real-time anime-style controls: clearcoat cel banding,
 * sheen cel banding, iridescence cel banding, transmission cel banding,
 * anisotropy toon shaping, mood-based per-channel grading, rim glow,
 * and procedural paper-grain / watercolor texture variation.
 *
 * ```js
 * const material = new THREE.MeshPhysicalMaterial( {
 *   color: 0xff88cc,
 *   roughness: 0.3,
 *   metalness: 0.0,
 *   clearcoat: 1.0,
 *   clearcoatRoughness: 0.1,
 *   clearcoatBands: 2,
 *   sheen: 1.0,
 *   sheenColor: 0xffccdd,
 *   sheenBands: 3,
 *   iridescence: 0.8,
 *   iridescenceBands: 5,
 *   anisotropy: 0.6,
 *   anisotropyAngle: Math.PI / 4,
 *   moodTemperature: 0.2
 * } );
 * ```
 *
 * @augments MeshStandardMaterial
 */
class MeshPhysicalMaterial extends MeshStandardMaterial {

	/**
	 * Constructs a new mesh physical material.
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
		this.isMeshPhysicalMaterial = true;

		this.type = 'MeshPhysicalMaterial';

		/**
		 * The clearcoat of the material. A value of 0.0 means no clearcoat,
		 * 1.0 means full clearcoat. Represents the reflective lacquer layer
		 * on top of the base PBR response — classic anime hair/nails/lips.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.clearcoat = 0.0;

		/**
		 * The clearcoat intensity map (red channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.clearcoatMap = null;

		/**
		 * The roughness of the clearcoat layer.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.clearcoatRoughness = 0.0;

		/**
		 * The clearcoat roughness map (green channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.clearcoatRoughnessMap = null;

		/**
		 * The clearcoat normal map.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.clearcoatNormalMap = null;

		/**
		 * The clearcoat normal scale.
		 *
		 * @type {Vector2}
		 * @default (1,1)
		 */
		this.clearcoatNormalScale = new Vector2( 1, 1 );

		/**
		 * The sheen of the material. Represents the retroreflective fabric
		 * shading on velvet, fur, and embroidered anime costumes.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.sheen = 0.0;

		/**
		 * The sheen color.
		 *
		 * @type {Color}
		 * @default (0,0,0)
		 */
		this.sheenColor = new Color( 0x000000 );

		/**
		 * The sheen color map.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.sheenColorMap = null;

		/**
		 * The sheen roughness.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.sheenRoughness = 1.0;

		/**
		 * The sheen roughness map (alpha channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.sheenRoughnessMap = null;

		/**
		 * The transmission of the material. Enables refraction of light
		 * through transparent surfaces — glass, water, gemstones.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.transmission = 0.0;

		/**
		 * The transmission map (red channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.transmissionMap = null;

		/**
		 * The thickness of the volume beneath the surface.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.thickness = 0.0;

		/**
		 * The thickness map (green channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.thicknessMap = null;

		/**
		 * The attenuation distance of the material.
		 *
		 * @type {number}
		 * @default Infinity
		 */
		this.attenuationDistance = Infinity;

		/**
		 * The attenuation color.
		 *
		 * @type {Color}
		 * @default (1,1,1)
		 */
		this.attenuationColor = new Color( 0xffffff );

		/**
		 * The intensity of the specular reflection.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.specularIntensity = 1.0;

		/**
		 * The specular intensity map (alpha channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.specularIntensityMap = null;

		/**
		 * The color of the specular reflection.
		 *
		 * @type {Color}
		 * @default (1,1,1)
		 */
		this.specularColor = new Color( 0xffffff );

		/**
		 * The specular color map (RGB channels).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.specularColorMap = null;

		/**
		 * The anisotropy of the material. A value of 1.0 is fully
		 * anisotropic — brushed metal, shiny anime hair streaks.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.anisotropy = 0.0;

		/**
		 * The rotation of the anisotropy axis in radians.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.anisotropyRotation = 0.0;

		/**
		 * The anisotropy map (RG channels).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.anisotropyMap = null;

		/**
		 * The index of refraction (IOR) of the material.
		 *
		 * @type {number}
		 * @default 1.5
		 */
		this.ior = 1.5;

		/**
		 * The dispersion of the material.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.dispersion = 0.0;

		/**
		 * The reflectivity of the material.
		 *
		 * @type {number}
		 * @default 0.5
		 */
		this.reflectivity = 0.5;

		/**
		 * The iridescence of the material. Represents the rainbow
		 * thin-film interference used on anime magical effects, fairy
		 * wings, and butterfly wings.
		 *
		 * @type {number}
		 * @default 0.0
		 */
		this.iridescence = 0.0;

		/**
		 * The iridescence map (red channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.iridescenceMap = null;

		/**
		 * The index of refraction of the iridescence layer.
		 *
		 * @type {number}
		 * @default 1.3
		 */
		this.iridescenceIOR = 1.3;

		/**
		 * The minimum and maximum thickness of the iridescence layer.
		 *
		 * @type {Array<number>}
		 * @default [100, 400]
		 */
		this.iridescenceThicknessRange = [ 100, 400 ];

		/**
		 * The iridescence thickness map (green channel).
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.iridescenceThicknessMap = null;

		// -------------------------------------------------------------------
		// Anime Rendering Extensions
		// -------------------------------------------------------------------

		/**
		 * Number of clearcoat cel bands. 2 = classic hard-edged glossy
		 * anime highlight, 3 = softer stylized, 0 = continuous (off).
		 *
		 * @type {number}
		 * @default 0
		 */
		this.clearcoatBands = 0;

		/**
		 * Blend amount for clearcoat banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.clearcoatQuantize = 1.0;

		/**
		 * Minimum clearcoat intensity to trigger the first band.
		 *
		 * @type {number}
		 * @default 0.5
		 */
		this.clearcoatThreshold = 0.5;

		/**
		 * Clearcoat band edge softness.
		 *
		 * @type {number}
		 * @default 0.05
		 */
		this.clearcoatSoftness = 0.05;

		/**
		 * Number of sheen cel bands. 2-4 = classic flat velvet shading.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.sheenBands = 0;

		/**
		 * Blend amount for sheen banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.sheenQuantize = 1.0;

		/**
		 * Sheen band edge softness.
		 *
		 * @type {number}
		 * @default 0.15
		 */
		this.sheenSoftness = 0.15;

		/**
		 * Number of iridescence cel bands. 3-6 = classic rainbow anime
		 * magical shimmer with discrete color bands.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.iridescenceBands = 0;

		/**
		 * Blend amount for iridescence banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.iridescenceQuantize = 1.0;

		/**
		 * Number of transmission cel bands. 2-4 = classic flat anime glass.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.transmissionBands = 0;

		/**
		 * Blend amount for transmission banding.
		 *
		 * @type {number}
		 * @default 1.0
		 */
		this.transmissionQuantize = 1.0;

		/**
		 * Anisotropy toon shaping. Set > 0 to elongate specular highlights
		 * for anime hair streaks and brushed metal.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.anisotropyToonAmount = 0;

		/**
		 * Anisotropy toon axis angle in radians.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.anisotropyAngle = 0;

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
		 * The precomputed mood-graded base color.
		 *
		 * @type {Color}
		 */
		this.moodColor = new Color( 0xffffff );

		/**
		 * The precomputed mood-graded emissive color.
		 *
		 * @type {Color}
		 */
		this.moodEmissive = new Color( 0x000000 );

		/**
		 * The precomputed mood-graded sheen color.
		 *
		 * @type {Color}
		 */
		this.moodSheenColor = new Color( 0x000000 );

		/**
		 * The precomputed mood-graded attenuation color.
		 *
		 * @type {Color}
		 */
		this.moodAttenuationColor = new Color( 0xffffff );

		/**
		 * The precomputed mood-graded specular color.
		 *
		 * @type {Color}
		 */
		this.moodSpecularColor = new Color( 0xffffff );

		/**
		 * Rim-glow intensity from fresnel.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.rimGlow = 0;

		/**
		 * Rim-glow color.
		 *
		 * @type {Color}
		 * @default (0.5, 0.9, 1.0)
		 */
		this.rimGlowColor = new Color( 0.5, 0.9, 1.0 );

		/**
		 * Fresnel rim falloff power.
		 *
		 * @type {number}
		 * @default 3.0
		 */
		this.rimPower = 3.0;

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
	 * Apply clearcoat cel banding to a raw clearcoat intensity.
	 *
	 * @param {number} clearcoatIntensity
	 * @returns {number}
	 */
	applyClearcoatBanding( clearcoatIntensity ) {

		return applyClearcoatCelBanding(
			clearcoatIntensity,
			this.clearcoatBands,
			this.clearcoatQuantize,
			this.clearcoatThreshold,
			this.clearcoatSoftness
		);

	}

	/**
	 * Apply sheen cel banding to a raw sheen intensity.
	 *
	 * @param {number} sheenIntensity
	 * @returns {number}
	 */
	applySheenBanding( sheenIntensity ) {

		return applySheenCelBanding(
			sheenIntensity,
			this.sheenBands,
			this.sheenQuantize,
			this.sheenSoftness
		);

	}

	/**
	 * Apply iridescence cel banding to a raw iridescence intensity.
	 *
	 * @param {number} iridescenceIntensity
	 * @returns {number}
	 */
	applyIridescenceBanding( iridescenceIntensity ) {

		return applyIridescenceCelBanding(
			iridescenceIntensity,
			this.iridescenceBands,
			this.iridescenceQuantize
		);

	}

	/**
	 * Apply transmission cel banding to a raw transmission intensity.
	 *
	 * @param {number} transmissionIntensity
	 * @returns {number}
	 */
	applyTransmissionBanding( transmissionIntensity ) {

		return applyTransmissionCelBanding(
			transmissionIntensity,
			this.transmissionBands,
			this.transmissionQuantize
		);

	}

	/**
	 * Apply anisotropy toon shaping to a specular intensity.
	 *
	 * @param {number} specularIntensity
	 * @param {number} halfVecX
	 * @param {number} halfVecY
	 * @returns {number}
	 */
	applyAnisotropyShaping( specularIntensity, halfVecX, halfVecY ) {

		return applyAnisotropyToonShaping(
			specularIntensity,
			this.anisotropyToonAmount,
			this.anisotropyAngle,
			halfVecX,
			halfVecY
		);

	}

	/**
	 * Recompute all mood-graded colors.
	 *
	 * @returns {MeshPhysicalMaterial} A reference to this instance.
	 */
	updateMoodColors() {

		const t = this.moodTemperature;
		const s = this.moodSaturation;
		const b = this.moodBrightness;
		const c = this.moodContrast;

		this.moodColor.copy( this.color );
		gradePbrExtColor( this.moodColor, t, s, b, c );

		this.moodEmissive.copy( this.emissive );
		gradePbrExtColor( this.moodEmissive, t, s, b, c );

		this.moodSheenColor.copy( this.sheenColor );
		gradePbrExtColor( this.moodSheenColor, t, s, b, c );

		this.moodAttenuationColor.copy( this.attenuationColor );
		gradePbrExtColor( this.moodAttenuationColor, t, s, b, c );

		this.moodSpecularColor.copy( this.specularColor );
		gradePbrExtColor( this.moodSpecularColor, t, s, b, c );

		return this;

	}

	/**
	 * Generate a procedural paper-grain / watercolor texture for this
	 * material.
	 *
	 * @param {number} width
	 * @param {number} height
	 * @param {number} [scale=0.03]
	 * @param {number} [octaves=4]
	 * @returns {Uint8Array}
	 */
	generatePbrExtTexture( width, height, scale = 0.03, octaves = 4 ) {

		return generatePbrExtTexture( width, height, scale, octaves, this.paperGrain, this.watercolorBleed );

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
	 * The default `onBeforeCompile` hook. Extends the base MeshStandardMaterial
	 * anime chunks with physical-material-specific features: clearcoat,
	 * sheen, iridescence, transmission cel banding, and anisotropy toon
	 * shaping.
	 *
	 * @param {Object} shader - The shader object.
	 * @param {WebGLRenderer} renderer - The renderer.
	 */
	onBeforeCompile( shader, renderer ) {

		// Call the parent hook first (MeshStandardMaterial)
		super.onBeforeCompile( shader, renderer );

		// Inject physical-material-specific uniforms
		shader.uniforms.clearcoatBands = { value: this.clearcoatBands };
		shader.uniforms.clearcoatQuantize = { value: this.clearcoatQuantize };
		shader.uniforms.clearcoatThreshold = { value: this.clearcoatThreshold };
		shader.uniforms.clearcoatSoftness = { value: this.clearcoatSoftness };
		shader.uniforms.sheenBands = { value: this.sheenBands };
		shader.uniforms.sheenQuantize = { value: this.sheenQuantize };
		shader.uniforms.sheenSoftness = { value: this.sheenSoftness };
		shader.uniforms.iridescenceBands = { value: this.iridescenceBands };
		shader.uniforms.iridescenceQuantize = { value: this.iridescenceQuantize };
		shader.uniforms.transmissionBands = { value: this.transmissionBands };
		shader.uniforms.transmissionQuantize = { value: this.transmissionQuantize };
		shader.uniforms.anisotropyToonAmount = { value: this.anisotropyToonAmount };
		shader.uniforms.anisotropyAngle = { value: this.anisotropyAngle };
		shader.uniforms.moodSheenColor = { value: this.moodSheenColor };
		shader.uniforms.moodAttenuationColor = { value: this.moodAttenuationColor };
		shader.uniforms.moodSpecularColor = { value: this.moodSpecularColor };

		// Extend the fragment shader
		shader.fragmentShader = shader.fragmentShader
			.replace(
				'uniform float brushJitter;',
				`uniform float brushJitter;
				uniform int clearcoatBands;
				uniform float clearcoatQuantize;
				uniform float clearcoatThreshold;
				uniform float clearcoatSoftness;
				uniform int sheenBands;
				uniform float sheenQuantize;
				uniform float sheenSoftness;
				uniform int iridescenceBands;
				uniform float iridescenceQuantize;
				uniform int transmissionBands;
				uniform float transmissionQuantize;
				uniform float anisotropyToonAmount;
				uniform float anisotropyAngle;
				uniform vec3 moodSheenColor;
				uniform vec3 moodAttenuationColor;
				uniform vec3 moodSpecularColor;

				float applyClearcoatBandingFn( float spec ) {
					if ( clearcoatBands <= 1 || spec < clearcoatThreshold ) return 0.0;
					float normalized = clamp( ( spec - clearcoatThreshold ) / max( 1.0 - clearcoatThreshold, 0.0001 ), 0.0, 1.0 );
					float bandWidth = 1.0 / float( clearcoatBands );
					float bandIndex = floor( normalized / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					return clamp( mix( normalized, quantized, clearcoatQuantize ), 0.0, 1.0 );
				}

				float applySheenBandingFn( float sheen ) {
					if ( sheenBands <= 1 ) return sheen;
					float bandWidth = 1.0 / float( sheenBands );
					float bandIndex = floor( sheen / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					return clamp( mix( sheen, quantized, sheenQuantize ), 0.0, 1.0 );
				}

				float applyIridescenceBandingFn( float irid ) {
					if ( iridescenceBands <= 1 ) return irid;
					float bandWidth = 1.0 / float( iridescenceBands );
					float bandIndex = floor( irid / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					return clamp( mix( irid, quantized, iridescenceQuantize ), 0.0, 1.0 );
				}

				float applyTransmissionBandingFn( float trans ) {
					if ( transmissionBands <= 1 ) return trans;
					float bandWidth = 1.0 / float( transmissionBands );
					float bandIndex = floor( trans / bandWidth );
					float quantized = bandIndex * bandWidth + bandWidth * 0.5;
					return clamp( mix( trans, quantized, transmissionQuantize ), 0.0, 1.0 );
				}

				float applyAnisotropyToonFn( float spec, vec3 halfVec ) {
					if ( anisotropyToonAmount <= 0.0 ) return spec;
					float cosA = cos( - anisotropyAngle );
					float sinA = sin( - anisotropyAngle );
					vec2 hv2 = halfVec.xy;
					vec2 rotated = vec2( hv2.x * cosA - hv2.y * sinA, hv2.x * sinA + hv2.y * cosA );
					float distSquared = rotated.x * rotated.x + rotated.y * rotated.y / max( 1e-4, 1.0 - anisotropyToonAmount );
					float shape = pow( clamp( 1.0 - distSquared, 0.0, 1.0 ), 0.5 );
					return spec * shape;
				}
				`
			)
			.replace(
				'gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );',
				`gl_FragColor.rgb = applyMoodGrading( gl_FragColor.rgb );

				// Compute view-space half-vector
				vec3 viewDirPhys = normalize( vViewPosition );
				vec3 lightDirPhys = normalize( vec3( 0.0, 0.0, 1.0 ) );
				vec3 halfVecPhys = normalize( viewDirPhys + lightDirPhys );

				// Extract raw specular response for clearcoat and anisotropy
				float rawSpecPhys = max( max( gl_FragColor.r - moodColor.r, gl_FragColor.g - moodColor.g ), gl_FragColor.b - moodColor.b );
				rawSpecPhys = max( 0.0, rawSpecPhys );

				// Apply anisotropy toon shaping to the specular
				float shapedSpecPhys = applyAnisotropyToonFn( rawSpecPhys, halfVecPhys );

				// Apply clearcoat cel banding
				float bandedClearcoat = applyClearcoatBandingFn( shapedSpecPhys );

				// Apply sheen cel banding
				float bandedSheen = applySheenBandingFn( rawSpecPhys );

				// Rebuild the fragment color from mood-graded base + extensions
				vec3 basePhys = moodColor;
				basePhys += moodEmissive * 0.5;
				basePhys += moodSheenColor * bandedSheen * 0.3;
				basePhys += vec3( bandedClearcoat ) * 0.5;
				basePhys = mix( basePhys, moodSpecularColor * bandedClearcoat, 0.4 );

				// Transmission tinting (stylized colored glass)
				if ( transmissionBands > 0 ) {
					float transNorm = applyTransmissionBandingFn( 0.5 );
					basePhys = mix( basePhys, moodAttenuationColor, transNorm * 0.3 );
				}

				gl_FragColor.rgb = basePhys;

				// Fresnel rim glow
				if ( rimGlow > 0.0 ) {
					float fresnelRim = 1.0 - abs( dot( normalize( vNormal ), viewDirPhys ) );
					fresnelRim = pow( fresnelRim, rimPower ) * rimGlow;
					gl_FragColor.rgb += rimGlowColor * fresnelRim;
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
			this.clearcoatBands,
			this.clearcoatQuantize,
			this.clearcoatThreshold,
			this.clearcoatSoftness,
			this.sheenBands,
			this.sheenQuantize,
			this.sheenSoftness,
			this.iridescenceBands,
			this.iridescenceQuantize,
			this.transmissionBands,
			this.transmissionQuantize,
			this.anisotropyToonAmount,
			this.anisotropyAngle,
			this.moodTemperature,
			this.moodSaturation,
			this.moodBrightness,
			this.moodContrast,
			this.rimGlow,
			this.paperGrain,
			this.watercolorBleed,
			this.variationSeed
		].join( '|' );

	}

	/**
	 * Copy the given material's properties into this one.
	 *
	 * @param {MeshPhysicalMaterial} source - The material to copy from.
	 * @return {MeshPhysicalMaterial} A reference to this instance.
	 */
	copy( source ) {

		super.copy( source );

		this.defines = Object.assign( {}, source.defines );

		this.clearcoat = source.clearcoat;
		this.clearcoatMap = source.clearcoatMap;
		this.clearcoatRoughness = source.clearcoatRoughness;
		this.clearcoatRoughnessMap = source.clearcoatRoughnessMap;
		this.clearcoatNormalMap = source.clearcoatNormalMap;
		this.clearcoatNormalScale.copy( source.clearcoatNormalScale );

		this.sheen = source.sheen;
		this.sheenColor.copy( source.sheenColor );
		this.sheenColorMap = source.sheenColorMap;
		this.sheenRoughness = source.sheenRoughness;
		this.sheenRoughnessMap = source.sheenRoughnessMap;

		this.transmission = source.transmission;
		this.transmissionMap = source.transmissionMap;
		this.thickness = source.thickness;
		this.thicknessMap = source.thicknessMap;
		this.attenuationDistance = source.attenuationDistance;
		this.attenuationColor.copy( source.attenuationColor );

		this.specularIntensity = source.specularIntensity;
		this.specularIntensityMap = source.specularIntensityMap;
		this.specularColor.copy( source.specularColor );
		this.specularColorMap = source.specularColorMap;

		this.anisotropy = source.anisotropy;
		this.anisotropyRotation = source.anisotropyRotation;
		this.anisotropyMap = source.anisotropyMap;

		this.ior = source.ior;
		this.dispersion = source.dispersion;
		this.reflectivity = source.reflectivity;

		this.iridescence = source.iridescence;
		this.iridescenceMap = source.iridescenceMap;
		this.iridescenceIOR = source.iridescenceIOR;
		this.iridescenceThicknessRange = source.iridescenceThicknessRange.slice();
		this.iridescenceThicknessMap = source.iridescenceThicknessMap;

		// Anime extensions
		this.clearcoatBands = source.clearcoatBands;
		this.clearcoatQuantize = source.clearcoatQuantize;
		this.clearcoatThreshold = source.clearcoatThreshold;
		this.clearcoatSoftness = source.clearcoatSoftness;
		this.sheenBands = source.sheenBands;
		this.sheenQuantize = source.sheenQuantize;
		this.sheenSoftness = source.sheenSoftness;
		this.iridescenceBands = source.iridescenceBands;
		this.iridescenceQuantize = source.iridescenceQuantize;
		this.transmissionBands = source.transmissionBands;
		this.transmissionQuantize = source.transmissionQuantize;
		this.anisotropyToonAmount = source.anisotropyToonAmount;
		this.anisotropyAngle = source.anisotropyAngle;
		this.moodTemperature = source.moodTemperature;
		this.moodSaturation = source.moodSaturation;
		this.moodBrightness = source.moodBrightness;
		this.moodContrast = source.moodContrast;
		this.moodColor.copy( source.moodColor );
		this.moodEmissive.copy( source.moodEmissive );
		this.moodSheenColor.copy( source.moodSheenColor );
		this.moodAttenuationColor.copy( source.moodAttenuationColor );
		this.moodSpecularColor.copy( source.moodSpecularColor );
		this.rimGlow = source.rimGlow;
		this.rimGlowColor.copy( source.rimGlowColor );
		this.rimPower = source.rimPower;
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

		data.type = 'MeshPhysicalMaterial';

		if ( this.clearcoat > 0 ) data.clearcoat = this.clearcoat;
		if ( this.clearcoatMap !== null ) data.clearcoatMap = this.clearcoatMap.toJSON( meta ).uuid;
		if ( this.clearcoatRoughness > 0 ) data.clearcoatRoughness = this.clearcoatRoughness;
		if ( this.clearcoatRoughnessMap !== null ) data.clearcoatRoughnessMap = this.clearcoatRoughnessMap.toJSON( meta ).uuid;
		if ( this.clearcoatNormalMap !== null ) data.clearcoatNormalMap = this.clearcoatNormalMap.toJSON( meta ).uuid;
		if ( this.clearcoatNormalScale.x !== 1 || this.clearcoatNormalScale.y !== 1 ) data.clearcoatNormalScale = this.clearcoatNormalScale.toArray();

		if ( this.sheen > 0 ) data.sheen = this.sheen;
		if ( this.sheenColor.getHex() !== 0x000000 ) data.sheenColor = this.sheenColor.getHex();
		if ( this.sheenColorMap !== null ) data.sheenColorMap = this.sheenColorMap.toJSON( meta ).uuid;
		if ( this.sheenRoughness !== 1 ) data.sheenRoughness = this.sheenRoughness;
		if ( this.sheenRoughnessMap !== null ) data.sheenRoughnessMap = this.sheenRoughnessMap.toJSON( meta ).uuid;

		if ( this.transmission > 0 ) data.transmission = this.transmission;
		if ( this.transmissionMap !== null ) data.transmissionMap = this.transmissionMap.toJSON( meta ).uuid;
		if ( this.thickness > 0 ) data.thickness = this.thickness;
		if ( this.thicknessMap !== null ) data.thicknessMap = this.thicknessMap.toJSON( meta ).uuid;
		if ( this.attenuationDistance !== Infinity ) data.attenuationDistance = this.attenuationDistance;
		if ( this.attenuationColor.getHex() !== 0xffffff ) data.attenuationColor = this.attenuationColor.getHex();

		if ( this.specularIntensity !== 1 ) data.specularIntensity = this.specularIntensity;
		if ( this.specularIntensityMap !== null ) data.specularIntensityMap = this.specularIntensityMap.toJSON( meta ).uuid;
		if ( this.specularColor.getHex() !== 0xffffff ) data.specularColor = this.specularColor.getHex();
		if ( this.specularColorMap !== null ) data.specularColorMap = this.specularColorMap.toJSON( meta ).uuid;

		if ( this.anisotropy > 0 ) data.anisotropy = this.anisotropy;
		if ( this.anisotropyRotation !== 0 ) data.anisotropyRotation = this.anisotropyRotation;
		if ( this.anisotropyMap !== null ) data.anisotropyMap = this.anisotropyMap.toJSON( meta ).uuid;

		if ( this.ior !== 1.5 ) data.ior = this.ior;
		if ( this.dispersion !== 0 ) data.dispersion = this.dispersion;
		if ( this.reflectivity !== 0.5 ) data.reflectivity = this.reflectivity;

		if ( this.iridescence > 0 ) data.iridescence = this.iridescence;
		if ( this.iridescenceMap !== null ) data.iridescenceMap = this.iridescenceMap.toJSON( meta ).uuid;
		if ( this.iridescenceIOR !== 1.3 ) data.iridescenceIOR = this.iridescenceIOR;
		if ( this.iridescenceThicknessRange[ 0 ] !== 100 || this.iridescenceThicknessRange[ 1 ] !== 400 ) data.iridescenceThicknessRange = this.iridescenceThicknessRange.slice();
		if ( this.iridescenceThicknessMap !== null ) data.iridescenceThicknessMap = this.iridescenceThicknessMap.toJSON( meta ).uuid;

		// Anime extensions
		if ( this.clearcoatBands !== 0 ) data.clearcoatBands = this.clearcoatBands;
		if ( this.clearcoatQuantize !== 1.0 ) data.clearcoatQuantize = this.clearcoatQuantize;
		if ( this.clearcoatThreshold !== 0.5 ) data.clearcoatThreshold = this.clearcoatThreshold;
		if ( this.clearcoatSoftness !== 0.05 ) data.clearcoatSoftness = this.clearcoatSoftness;
		if ( this.sheenBands !== 0 ) data.sheenBands = this.sheenBands;
		if ( this.sheenQuantize !== 1.0 ) data.sheenQuantize = this.sheenQuantize;
		if ( this.sheenSoftness !== 0.15 ) data.sheenSoftness = this.sheenSoftness;
		if ( this.iridescenceBands !== 0 ) data.iridescenceBands = this.iridescenceBands;
		if ( this.iridescenceQuantize !== 1.0 ) data.iridescenceQuantize = this.iridescenceQuantize;
		if ( this.transmissionBands !== 0 ) data.transmissionBands = this.transmissionBands;
		if ( this.transmissionQuantize !== 1.0 ) data.transmissionQuantize = this.transmissionQuantize;
		if ( this.anisotropyToonAmount !== 0 ) data.anisotropyToonAmount = this.anisotropyToonAmount;
		if ( this.anisotropyAngle !== 0 ) data.anisotropyAngle = this.anisotropyAngle;
		if ( this.moodTemperature !== 0 ) data.moodTemperature = this.moodTemperature;
		if ( this.moodSaturation !== 1.0 ) data.moodSaturation = this.moodSaturation;
		if ( this.moodBrightness !== 1.0 ) data.moodBrightness = this.moodBrightness;
		if ( this.moodContrast !== 1.0 ) data.moodContrast = this.moodContrast;
		if ( this.rimGlow !== 0 ) data.rimGlow = this.rimGlow;
		if ( this.rimGlowColor.getHex() !== new Color( 0.5, 0.9, 1.0 ).getHex() ) data.rimGlowColor = this.rimGlowColor.getHex();
		if ( this.rimPower !== 3.0 ) data.rimPower = this.rimPower;
		if ( this.paperGrain !== 0 ) data.paperGrain = this.paperGrain;
		if ( this.watercolorBleed !== 0 ) data.watercolorBleed = this.watercolorBleed;
		if ( this.variationSeed !== 0 ) data.variationSeed = this.variationSeed;

		return data;

	}

}

export { MeshPhysicalMaterial, MeshPhysicalMaterialBatch, applyClearcoatCelBanding, applySheenCelBanding, applyIridescenceCelBanding, applyTransmissionCelBanding, applyAnisotropyToonShaping, gradePbrExtColor, generatePbrExtTexture };
export default MeshPhysicalMaterial;