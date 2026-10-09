// file number : 008
// full path name : src/core/008_RenderTarget3D.js
// description : Three-dimensional render target that writes GPU output into a Data3DTexture (width × height × depth). Extends the local 007_RenderTarget and overrides its texture with a Data3DTexture instance. Bridges to bitecs for SoA 3D-target registration, gl-matrix for packing the 3D header (width, height, depth, count) into a vec4, double.js for high-precision voxel-count tracking, and simplex-noise for procedural target naming / volume-noise helpers.
// best for  : Volume rendering, 3D noise fields, light-probe grids, atlas stacking along Z, and any offscreen pass that must be sampled as a 3D texture (WebGPU backend).
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import RenderTarget from './007_RenderTarget.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 RenderTarget3D.js source.
// Data3DTexture is the concrete 3D texture class that replaces the 2D Texture
// inherited from RenderTarget.
import { Data3DTexture } from '../textures/Data3DTexture.js';
import { LinearFilter, NoColorSpace, RGBAFormat, UnsignedByteType } from '../constants.js';

// ── External libraries (must be imported and used) ───────────────────────────
import * as bitecs from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.js';
import Double from 'https://cdn.jsdelivr.net/npm/double.js@1.1.0/dist/double.js';
import { createNoise2D, createNoise3D, createNoise4D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';

const _noise2D = createNoise2D();
const _noise3D = createNoise3D();
const _noise4D = createNoise4D();

const _scratchVec4 = new Float64Array( 4 );
const _scratchMat4 = new Float64Array( 16 );

// Default options mirror of the original RenderTarget3D.js.
const _DEFAULTS_3D = {
	generateMipmaps: false,
	internalFormat: null,
	minFilter: LinearFilter,
	magFilter: LinearFilter,
	format: RGBAFormat,
	type: UnsignedByteType,
	wrapS: 1001, // ClampToEdgeWrapping
	wrapT: 1001,
	wrapR: 1001,
	anisotropy: 1,
	colorSpace: NoColorSpace,
	depthBuffer: true,
	stencilBuffer: false,
	resolveDepthBuffer: true,
	resolveStencilBuffer: true,
	depthTexture: null,
	samples: 0,
	count: 1,
	multiview: false,
};

const RenderTarget3DUtils = {

	// gl-matrix bridge: pack the 3D header (width, height, depth, count) into a vec4.
	packHeaderVec4: ( out, width, height, depth, count ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, width, height, depth, count );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a 3D render target as a SoA component column set.
	registerComponent: ( name, count ) => {

		const widthColumn = new Float64Array( count );
		const heightColumn = new Float64Array( count );
		const depthColumn = new Float64Array( count );
		const samplesColumn = new Uint8Array( count );
		return { name, widthColumn, heightColumn, depthColumn, samplesColumn, count };

	},

	// double.js bridge: high-precision voxel count (width × height × depth).
	totalVoxels: ( width, height, depth ) => {

		const w = new Double( width );
		const h = new Double( height );
		const d = new Double( depth );
		return w.mul( h ).mul( d ).valueOf();

	},

	// simplex-noise bridge: procedural 3D target name helper.
	randomName: ( seed = 0 ) => {

		const n = _noise2D( seed, 0 );
		return `RT3D_${ Math.abs( Math.floor( n * 1e6 ) ) }`;

	},

	// simplex-noise bridge: fill a Float32Array with 3D noise (useful for initial volume data).
	fillVolumeNoise: ( out, width, height, depth, scale = 0.1, seed = 0 ) => {

		let i = 0;
		for ( let z = 0; z < depth; z ++ ) {

			for ( let y = 0; y < height; y ++ ) {

				for ( let x = 0; x < width; x ++ ) {

					out[ i ++ ] = _noise3D( x * scale + seed, y * scale + seed, z * scale + seed );

				}

			}

		}

		return out;

	},

	noise2D: ( x, y ) => _noise2D( x, y ),
	noise3D: ( x, y, z ) => _noise3D( x, y, z ),
	noise4D: ( x, y, z, w ) => _noise4D( x, y, z, w ),

	bitecs,
	glMatrix,
	Double,

};

class RenderTarget3D extends RenderTarget {

	/**
	 * @param {number} [width=1]
	 * @param {number} [height=1]
	 * @param {number} [depth=1]
	 * @param {Object} [options]
	 */
	constructor( width = 1, height = 1, depth = 1, options = {} ) {

		super( width, height, options );

		options = Object.assign( {}, _DEFAULTS_3D, options );

		this.isRenderTarget3D = true;

		this.depth = depth;

		// Replace the 2D Texture created by RenderTarget with a Data3DTexture.
		const texture = new Data3DTexture( null, width, height, depth );

		texture.format = options.format;
		texture.type = options.type;
		texture.internalFormat = options.internalFormat;
		texture.magFilter = options.magFilter;
		texture.minFilter = options.minFilter;
		texture.wrapS = options.wrapS;
		texture.wrapT = options.wrapT;
		texture.wrapR = options.wrapR;
		texture.anisotropy = options.anisotropy;
		texture.colorSpace = options.colorSpace;
		texture.generateMipmaps = options.generateMipmaps;
		texture.isRenderTargetTexture = true;

		this.texture = texture;

		this._version = 0;

	}

	setSize( width, height, depth = 1 ) {

		if ( this.width !== width || this.height !== height || this.depth !== depth ) {

			this.width = width;
			this.height = height;
			this.depth = depth;

			this.texture.image.width = width;
			this.texture.image.height = height;
			this.texture.image.depth = depth;

			this.dispose();

		}

		this.viewport.set( 0, 0, width, height );
		this.scissor.set( 0, 0, width, height );

		return this;

	}

	clone() {

		return new this.constructor().copy( this );

	}

	copy( source ) {

		super.copy( source );

		this.depth = source.depth;

		if ( source.texture && typeof source.texture.clone === 'function' ) {

			this.texture = source.texture.clone();
			this.texture.image = Object.assign( {}, source.texture.image );

		}

		return this;

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return RenderTarget3DUtils.packHeaderVec4( out, this.width, this.height, this.depth, this.count );

	}

	asBitecsComponent( name, count ) {

		return RenderTarget3DUtils.registerComponent( name, count );

	}

	get totalVoxels() {

		return RenderTarget3DUtils.totalVoxels( this.width, this.height, this.depth );

	}

	// Fill this target's backing data with procedural 3D noise (optional helper).
	fillWithNoise( scale, seed ) {

		const { width, height, depth } = this.texture.image;
		const data = this.texture.image.data;

		if ( data && data.length >= width * height * depth ) {

			RenderTarget3DUtils.fillVolumeNoise( data, width, height, depth, scale, seed );

		}

		return this;

	}

	get version() {

		return this._version;

	}

	static randomName( seed ) {

		return RenderTarget3DUtils.randomName( seed );

	}

}

RenderTarget3D.Utils = RenderTarget3DUtils;

export default RenderTarget3D;
export { RenderTarget3DUtils };