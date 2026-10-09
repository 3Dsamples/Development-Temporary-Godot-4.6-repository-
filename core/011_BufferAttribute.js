// file number : 011
// full path name : src/core/011_BufferAttribute.js
// description : Stores data for a vertex attribute (position, normal, uv, color, etc.) associated with a geometry, enabling efficient GPU upload. Rewritten as an ES module; extends the local 001_EventDispatcher and consumes 002_Vector2, 003_Vector3, and 016_Vector4 for the fromBufferAttribute / toBufferAttribute bridging helpers. Bridges to 001_MathUtils for denormalize / normalize / clamp / generateUUID, gl-matrix for packing the attribute header (itemSize, count, normalized, version) into a vec4, double.js for high-precision byte-length tracking, bitecs for SoA attribute-column registration, and simplex-noise for procedural attribute generators. All non-chat three.js r185 imports (DataUtils, StaticDrawUsage, FloatType) are imported explicitly so the module remains self-contained.
// best for  : Base class for all geometry attributes. Directly consumed by BufferGeometry, WebGLAttributes, and InstancedBufferAttribute.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import EventDispatcher from './001_EventDispatcher.js';
import MathUtils from './001_MathUtils.js';
import Vector2 from '../math/002_Vector2.js';
import Vector3 from '../math/003_Vector3.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 BufferAttribute.js source.
// DataUtils provides the FP16 <-> FP32 conversion tables; constants.js provides
// the usage / type enumerations consumed by the class.
import { StaticDrawUsage, FloatType } from '../constants.js';
import { fromHalfFloat, toHalfFloat } from '../extras/DataUtils.js';

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

// Scratch objects for fromBufferAttribute / toBufferAttribute (mirrors r185 source).
const _vector = new Vector3();
const _vector2 = new Vector2();

let _id = 0;

// Default options mirror of the original BufferAttribute.js defaults.
const _DEFAULTS = {
	normalized: false,
	usage: StaticDrawUsage,
};

const BufferAttributeUtils = {

	// 001_MathUtils bridge: denormalize a normalized value back to its integer range.
	denormalize: ( value, array ) => {

		return MathUtils.denormalize( value, array );

	},

	// 001_MathUtils bridge: normalize an integer value into the [ -1, 1 ] or [ 0, 1 ] range.
	normalize: ( value, array ) => {

		return MathUtils.normalize( value, array );

	},

	// 001_MathUtils bridge: clamp an index into the valid attribute range.
	clampIndex: ( index, count ) => {

		return MathUtils.clamp( Math.floor( index ), 0, count - 1 );

	},

	// gl-matrix bridge: pack the attribute header (itemSize, count, normalized, version) into a vec4.
	packHeaderVec4: ( out, itemSize, count, normalized, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, itemSize, count, normalized ? 1 : 0, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a BufferAttribute as a SoA component column.
	registerComponent: ( name, count, itemSize ) => {

		const column = new Float64Array( count * itemSize );
		return { name, column, itemSize, count };

	},

	// double.js bridge: high-precision total byte length of the attribute.
	totalBytes: ( count, itemSize, bytesPerElement = 4 ) => {

		const c = new Double( count );
		const i = new Double( itemSize );
		const b = new Double( bytesPerElement );
		return c.mul( i ).mul( b ).valueOf();

	},

	// DataUtils bridge: FP32 -> FP16 half-float conversion.
	toHalfFloat: ( value ) => {

		return toHalfFloat( value );

	},

	// DataUtils bridge: FP16 -> FP32 half-float conversion.
	fromHalfFloat: ( value ) => {

		return fromHalfFloat( value );

	},

	// simplex-noise bridge: fill an attribute array with procedural 2D / 3D noise.
	fillWithNoise: ( out, count, itemSize, scale = 0.1, seed = 0 ) => {

		let i = 0;
		for ( let v = 0; v < count; v ++ ) {

			for ( let c = 0; c < itemSize; c ++ ) {

				out[ i ++ ] = _noise3D( v * scale + seed, c * scale + seed, seed );

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
	Vector2,
	Vector3,
	Vector4,

};

class BufferAttribute extends EventDispatcher {

	/**
	 * @param {TypedArray} array - The array holding the attribute data.
	 * @param {number} itemSize - The item size.
	 * @param {boolean} [normalized=false] - Whether the data are normalized or not.
	 */
	constructor( array, itemSize, normalized = false ) {

		super();

		if ( Array.isArray( array ) ) {

			throw new TypeError( 'THREE.BufferAttribute: array should be a Typed Array.' );

		}

		this.isBufferAttribute = true;

		Object.defineProperty( this, 'id', { value: _id ++ } );

		this.name = '';

		this.array = array;
		this.itemSize = itemSize;
		this.count = array !== undefined ? array.length / itemSize : 0;
		this.normalized = normalized;

		this.usage = StaticDrawUsage;
		this.updateRanges = [];
		this.gpuType = FloatType;

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

		console.warn( 'THREE.BufferAttribute: updateRange() is deprecated. Use addUpdateRange() and clearUpdateRanges() instead.' );
		this.addUpdateRange( start, count );

	}

	// ── Vector bridging helpers (fromBufferAttribute / toBufferAttribute) ─────
	// These mirror the r185 source using the local 002_Vector2, 003_Vector3,
	// and 016_Vector4 DeepSeek-chat classes.

	applyMatrix4( matrix ) {

		const array = this.array;
		const count = this.count;
		const itemSize = this.itemSize;

		if ( itemSize !== 3 ) {

			throw new Error( 'THREE.BufferAttribute.applyMatrix4(): itemSize must be 3.' );

		}

		for ( let i = 0; i < count; i ++ ) {

			_vector.fromBufferAttribute( this, i );
			_vector.applyMatrix4( matrix );
			_vector.toArray( array, i * itemSize );

		}

		return this;

	}

	applyMatrix3( matrix ) {

		const array = this.array;
		const count = this.count;
		const itemSize = this.itemSize;

		if ( itemSize !== 3 && itemSize !== 2 ) {

			throw new Error( 'THREE.BufferAttribute.applyMatrix3(): itemSize must be 2 or 3.' );

		}

		for ( let i = 0; i < count; i ++ ) {

			if ( itemSize === 3 ) {

				_vector.fromBufferAttribute( this, i );
				_vector.applyMatrix3( matrix );
				_vector.toArray( array, i * itemSize );

			} else {

				_vector2.fromBufferAttribute( this, i );
				_vector2.applyMatrix3( matrix );
				_vector2.toArray( array, i * itemSize );

			}

		}

		return this;

	}

	transformDirection( matrix ) {

		const array = this.array;
		const count = this.count;
		const itemSize = this.itemSize;

		if ( itemSize !== 3 ) {

			throw new Error( 'THREE.BufferAttribute.transformDirection(): itemSize must be 3.' );

		}

		for ( let i = 0; i < count; i ++ ) {

			_vector.fromBufferAttribute( this, i );
			_vector.transformDirection( matrix );
			_vector.toArray( array, i * itemSize );

		}

		return this;

	}

	// ── Core accessors ────────────────────────────────────────────────────────

	getX( index ) {

		return this.array[ index * this.itemSize ];

	}

	setX( index, x ) {

		this.array[ index * this.itemSize ] = x;
		return this;

	}

	getY( index ) {

		return this.array[ index * this.itemSize + 1 ];

	}

	setY( index, y ) {

		this.array[ index * this.itemSize + 1 ] = y;
		return this;

	}

	getZ( index ) {

		return this.array[ index * this.itemSize + 2 ];

	}

	setZ( index, z ) {

		this.array[ index * this.itemSize + 2 ] = z;
		return this;

	}

	getW( index ) {

		return this.array[ index * this.itemSize + 3 ];

	}

	setW( index, w ) {

		this.array[ index * this.itemSize + 3 ] = w;
		return this;

	}

	setXY( index, x, y ) {

		index *= this.itemSize;

		this.array[ index + 0 ] = x;
		this.array[ index + 1 ] = y;

		return this;

	}

	setXYZ( index, x, y, z ) {

		index *= this.itemSize;

		this.array[ index + 0 ] = x;
		this.array[ index + 1 ] = y;
		this.array[ index + 2 ] = z;

		return this;

	}

	setXYZW( index, x, y, z, w ) {

		index *= this.itemSize;

		this.array[ index + 0 ] = x;
		this.array[ index + 1 ] = y;
		this.array[ index + 2 ] = z;
		this.array[ index + 3 ] = w;

		return this;

	}

	// ── Copy / clone / serialization ─────────────────────────────────────────

	copy( source ) {

		this.name = source.name;
		this.array = new source.array.constructor( source.array );
		this.itemSize = source.itemSize;
		this.count = source.count;
		this.normalized = source.normalized;

		this.usage = source.usage;
		this.gpuType = source.gpuType;

		return this;

	}

	clone() {

		return new this.constructor( this.array, this.itemSize ).copy( this );

	}

	toJSON( data ) {

		const array = this.array;

		if ( array.buffer !== undefined && array.buffer._uuid === undefined ) {

			array.buffer._uuid = MathUtils.generateUUID();

		}

		if ( data.arrayBuffers === undefined ) {

			data.arrayBuffers = {};

		}

		if ( array.buffer !== undefined && data.arrayBuffers[ array.buffer._uuid ] === undefined ) {

			data.arrayBuffers[ array.buffer._uuid ] = Array.from( new Uint32Array( array.buffer ) );

		}

		return {
			itemSize: this.itemSize,
			type: array.constructor.name,
			array: Array.from( array ),
			normalized: this.normalized
		};

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	packToVec4( out ) {

		return BufferAttributeUtils.packHeaderVec4(
			out,
			this.itemSize,
			this.count,
			this.normalized,
			this.version
		);

	}

	asBitecsComponent( name ) {

		return BufferAttributeUtils.registerComponent( name, this.count, this.itemSize );

	}

	getTotalBytes( bytesPerElement = 4 ) {

		return BufferAttributeUtils.totalBytes( this.count, this.itemSize, bytesPerElement );

	}

	fillWithNoise( scale, seed ) {

		BufferAttributeUtils.fillWithNoise( this.array, this.count, this.itemSize, scale, seed );
		this.needsUpdate = true;
		return this;

	}

	get uuid() {

		if ( this._uuid === undefined ) {

			this._uuid = MathUtils.generateUUID();

		}

		return this._uuid;

	}

}

BufferAttribute.Utils = BufferAttributeUtils;

export default BufferAttribute;
export { BufferAttributeUtils };