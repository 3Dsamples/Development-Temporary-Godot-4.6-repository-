// file number : 007
// full path name : src/core/007_RenderTarget.js
// description : Offscreen render target buffer that the GPU draws into before post-processing / display. Rewritten as an ES module; extends the local 001_EventDispatcher and uses 016_Vector4 for scissor/viewport. Bridges to bitecs for SoA render-target registration, gl-matrix for packing the RT header (width, height, depth, count) into a vec4, double.js for high-precision dimension tracking, and simplex-noise for procedural RT-name generation. All non-chat three.js r185 imports (Texture, LinearFilter, Source) are imported explicitly so the module remains self-contained.
// best for  : Post-processing chains, shadow map targets, reflection/refraction passes, and MRT pipelines. Directly consumed by WebGLRenderTarget and WebGPURenderTarget.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import EventDispatcher from './001_EventDispatcher.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 RenderTarget.js source.
// They are kept as external leaves so the render-target behaviour is preserved
// without rewriting the entire texture / constants subsystem.
import { Texture } from '../textures/Texture.js';
import { LinearFilter } from '../constants.js';
import { Source } from '../textures/Source.js';

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

// Default constant mirror of three.js constants.js (LinearFilter = 1006)
const _DEFAULTS = {
	generateMipmaps: false,
	internalFormat: null,
	minFilter: LinearFilter,
	depthBuffer: true,
	stencilBuffer: false,
	resolveDepthBuffer: true,
	resolveStencilBuffer: true,
	depthTexture: null,
	samples: 0,
	count: 1,
	depth: 1,
	multiview: false,
	useArrayDepthTexture: false,
};

const RenderTargetUtils = {

	// gl-matrix bridge: pack RT header (width, height, depth, count) into a vec4.
	packHeaderVec4: ( out, width, height, depth, count ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, width, height, depth, count );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a RenderTarget as a SoA component column set.
	registerComponent: ( name, count ) => {

		const widthColumn = new Float64Array( count );
		const heightColumn = new Float64Array( count );
		const depthColumn = new Float64Array( count );
		const samplesColumn = new Uint8Array( count );
		return { name, widthColumn, heightColumn, depthColumn, samplesColumn, count };

	},

	// double.js bridge: high-precision area computation for viewport/scissor.
	totalPixels: ( width, height, depth ) => {

		const w = new Double( width );
		const h = new Double( height );
		const d = new Double( depth );
		return w.mul( h ).mul( d ).valueOf();

	},

	// simplex-noise bridge: procedural RT name helper.
	randomName: ( seed = 0 ) => {

		const n = _noise2D( seed, 0 );
		return `RT_${ Math.abs( Math.floor( n * 1e6 ) ) }`;

	},

	noise2D: ( x, y ) => _noise2D( x, y ),
	noise3D: ( x, y, z ) => _noise3D( x, y, z ),
	noise4D: ( x, y, z, w ) => _noise4D( x, y, z, w ),

	bitecs,
	glMatrix,
	Double,

};

class RenderTarget extends EventDispatcher {

	constructor( width = 1, height = 1, options = {} ) {

		super();

		options = Object.assign( {}, _DEFAULTS, options );

		this.isRenderTarget = true;

		this.width = width;
		this.height = height;
		this.depth = options.depth;

		this.scissor = new Vector4( 0, 0, width, height );
		this.scissorTest = false;

		this.viewport = new Vector4( 0, 0, width, height );

		const image = { width, height, depth: options.depth };

		if ( options.multiview ) {

			const source = new Source( new DataView( new ArrayBuffer( width * height * options.depth * 4 ) ) );
			source.needsUpdate = true;

			this.texture = new Texture();
			this.texture.source = source;
			this.texture.image = image;

		} else {

			this.texture = new Texture(
				image,
				options.mapping,
				options.wrapS,
				options.wrapT,
				options.magFilter,
				options.minFilter,
				options.format,
				options.type,
				options.anisotropy,
				options.colorSpace
			);

		}

		this.texture.isRenderTargetTexture = true;
		this.texture.generateMipmaps = options.generateMipmaps;
		this.texture.internalFormat = options.internalFormat;

		this.depthBuffer = options.depthBuffer;
		this.stencilBuffer = options.stencilBuffer;

		this.resolveDepthBuffer = options.resolveDepthBuffer;
		this.resolveStencilBuffer = options.resolveStencilBuffer;

		this.depthTexture = options.depthTexture;

		this.samples = options.samples;
		this.count = options.count;
		this.multiview = options.multiview;
		this.useArrayDepthTexture = options.useArrayDepthTexture;

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

		this.width = source.width;
		this.height = source.height;
		this.depth = source.depth;

		this.scissor.copy( source.scissor );
		this.scissorTest = source.scissorTest;

		this.viewport.copy( source.viewport );

		this.texture = source.texture.clone();
		this.texture.image = Object.assign( {}, source.texture.image );

		this.depthBuffer = source.depthBuffer;
		this.stencilBuffer = source.stencilBuffer;
		this.resolveDepthBuffer = source.resolveDepthBuffer;
		this.resolveStencilBuffer = source.resolveStencilBuffer;

		this.depthTexture = source.depthTexture;

		this.samples = source.samples;
		this.count = source.count;
		this.multiview = source.multiview;
		this.useArrayDepthTexture = source.useArrayDepthTexture;

		return this;

	}

	dispose() {

		this.dispatchEvent( { type: 'dispose' } );

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return RenderTargetUtils.packHeaderVec4( out, this.width, this.height, this.depth, this.count );

	}

	asBitecsComponent( name, count ) {

		return RenderTargetUtils.registerComponent( name, count );

	}

	get totalPixels() {

		return RenderTargetUtils.totalPixels( this.width, this.height, this.depth );

	}

	get version() {

		return this._version;

	}

	static randomName( seed ) {

		return RenderTargetUtils.randomName( seed );

	}

}

RenderTarget.Utils = RenderTargetUtils;

export default RenderTarget;
export { RenderTargetUtils };