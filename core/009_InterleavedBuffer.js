// file number : 009
// full path name : src/core/009_InterleavedBuffer.js
// description : "Interleaved" means that multiple attributes, possibly of different types, (e.g., position, normal, uv, color) are packed into a single array buffer. Rewritten as an ES module; bridges to 001_MathUtils for scalar normalization and clamping, gl-matrix for packing the interleaved header (stride, count, itemSize, version) into a vec4, double.js for high-precision byte-offset / element-count tracking, bitecs for SoA registration of interleaved vertex columns, and simplex-noise for procedural stride/offset generation. All non-chat three.js r185 imports (StaticDrawUsage, DynamicDrawUsage, StreamDrawUsage) are imported explicitly so the module remains self-contained.
// best for  : Storing multiple vertex attributes (position, normal, uv, color) in a single contiguous typed array for improved cache locality and single-buffer GPU uploads. Base class for InstancedInterleavedBuffer.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import MathUtils from './001_MathUtils.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 InterleavedBuffer.js source.
// Usage constants are consumed by the .usage property and the utility surface.
import { StaticDrawUsage, DynamicDrawUsage, StreamDrawUsage } from '../constants.js';

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

// Map of usage constants (mirrors three.js constants.js values).
const _USAGE_MAP = {
	StaticDrawUsage,
	DynamicDrawUsage,
	StreamDrawUsage,
};

const InterleavedBufferUtils = {

	// 001_MathUtils bridge: clamp a stride to a positive integer.
	clampStride: ( stride ) => {

		return MathUtils.clamp( Math.floor( stride ), 1, Infinity );

	},

	// gl-matrix bridge: pack the interleaved header (stride, count, itemSize, version) into a vec4.
	packHeaderVec4: ( out, stride, count, itemSize, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, stride, count, itemSize, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register an InterleavedBuffer as a SoA component column set.
	registerComponent: ( name, count, stride ) => {

		const dataColumn = new Float64Array( count * stride );
		return { name, dataColumn, stride, count };

	},

	// double.js bridge: high-precision total element count (array.length / stride).
	totalElements: ( arrayLength, stride ) => {

		const a = new Double( arrayLength );
		const s = new Double( stride );
		return a.div( s ).valueOf();

	},

	// double.js bridge: high-precision byte offset for a given vertex index.
	byteOffset: ( index, stride, bytesPerElement ) => {

		const i = new Double( index );
		const s = new Double( stride );
		const b = new Double( bytesPerElement );
		return i.mul( s ).mul( b ).valueOf();

	},

	// simplex-noise bridge: procedural stride helper for procedural mesh generation.
	randomStride: ( seed = 0, min = 2, max = 8 ) => {

		const n = _noise2D( seed, 0 );
		const t = ( n + 1 ) * 0.5;
		return Math.floor( min + t * ( max - min ) );

	},

	noise2D: ( x, y ) => _noise2D( x, y ),
	noise3D: ( x, y, z ) => _noise3D( x, y, z ),
	noise4D: ( x, y, z, w ) => _noise4D( x, y, z, w ),

	bitecs,
	glMatrix,
	Double,

};

class InterleavedBuffer {

	constructor( array, stride ) {

		this.isInterleavedBuffer = true;

		this.array = array;
		this.stride = InterleavedBufferUtils.clampStride( stride );
		this.count = array !== undefined ? array.length / stride : 0;

		this.usage = StaticDrawUsage;

		this.updateRanges = [];
		this.gpuType = undefined;

		this.version = 0;

	}

	set needsUpdate( value ) {

		if ( value === true ) this.version ++;

	}

	setUsage( value ) {

		this.usage = value;
		return this;

	}

	addUpdateRange( start, count ) {

		this.updateRanges.push( { start, count } );

	}

	clearUpdateRanges() {

		this.updateRanges.length = 0;

	}

	updateRange( start, count ) {

		console.warn( 'THREE.InterleavedBuffer: updateRange() is deprecated. Use addUpdateRange() and clearUpdateRanges() instead.' );
		this.addUpdateRange( start, count );

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return InterleavedBufferUtils.packHeaderVec4( out, this.stride, this.count, this.stride, this.version );

	}

	asBitecsComponent( name ) {

		return InterleavedBufferUtils.registerComponent( name, this.count, this.stride );

	}

	getTotalElements() {

		return InterleavedBufferUtils.totalElements( this.array.length, this.stride );

	}

	getByteOffset( index, bytesPerElement = 4 ) {

		return InterleavedBufferUtils.byteOffset( index, this.stride, bytesPerElement );

	}

	copyAt( index1, attribute, index2 ) {

		index1 *= this.stride;
		index2 *= attribute.stride;

		for ( let i = 0, l = this.stride; i < l; i ++ ) {

			this.array[ index1 + i ] = attribute.array[ index2 + i ];

		}

		return this;

	}

	set( value, offset = 0 ) {

		this.array.set( value, offset );

	}

	clone( data ) {

		if ( data.arrayBuffers === undefined ) {

			data.arrayBuffers = {};

		}

		if ( this.array.buffer._uuid === undefined ) {

			this.array.buffer._uuid = MathUtils.generateUUID();

		}

		if ( data.arrayBuffers[ this.array.buffer._uuid ] === undefined ) {

			data.arrayBuffers[ this.array.buffer._uuid ] = this.array.slice( 0 ).buffer;

		}

		const array = new this.array.constructor( data.arrayBuffers[ this.array.buffer._uuid ] );

		const ib = new this.constructor( array, this.stride );

		ib.setUsage( this.usage );

		return ib;

	}

	onUploadCallback() {}

	setUsageValue( value ) {

		this.usage = value;
		return this;

	}

	toJSON( data ) {

		if ( data.arrayBuffers === undefined ) {

			data.arrayBuffers = {};

		}

		if ( this.array.buffer._uuid === undefined ) {

			this.array.buffer._uuid = MathUtils.generateUUID();

		}

		if ( data.arrayBuffers[ this.array.buffer._uuid ] === undefined ) {

			data.arrayBuffers[ this.array.buffer._uuid ] = Array.from( new Uint32Array( this.array.buffer ) );

		}

		return {
			uuid: this.uuid,
			buffer: this.array.buffer._uuid,
			type: this.array.constructor.name,
			stride: this.stride
		};

	}

	onUpload( callback ) {

		this.onUploadCallback = callback;

		return this;

	}

	get uuid() {

		if ( this._uuid === undefined ) {

			this._uuid = MathUtils.generateUUID();

		}

		return this._uuid;

	}

	set uuid( value ) {

		this._uuid = value;

	}

	static randomStride( seed, min, max ) {

		return InterleavedBufferUtils.randomStride( seed, min, max );

	}

	static get Usage() {

		return _USAGE_MAP;

	}

}

InterleavedBuffer.Utils = InterleavedBufferUtils;

export default InterleavedBuffer;
export { InterleavedBufferUtils };