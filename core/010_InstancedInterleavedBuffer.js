// file number : 010
// full path name : src/core/010_InstancedInterleavedBuffer.js
// description : Instanced version of InterleavedBuffer. Adds a meshPerAttribute field that controls how many instances share each interleaved vertex record. Rewritten as an ES module; extends the local 009_InterleavedBuffer and reuses its entire stride/count/usage surface. Bridges to 001_MathUtils for clamping meshPerAttribute, gl-matrix for packing the instance header (stride, count, meshPerAttribute, version) into a vec4, double.js for high-precision instance-byte tracking, bitecs for SoA instance-column registration, and simplex-noise for procedural instance-count helpers.
// best for  : InstancedMesh / InstancedBufferGeometry attributes that must repeat a vertex record N times per instance (e.g. per-instance matrices stored in an interleaved layout).
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import InterleavedBuffer from './009_InterleavedBuffer.js';
import MathUtils from './001_MathUtils.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// The original r185 InstancedInterleavedBuffer.js only imports InterleavedBuffer,
// which is already provided by the DeepSeek chat link as 009_InterleavedBuffer.js.
// No additional three.js r185 src/ file is required for this module.

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

const InstancedInterleavedBufferUtils = {

	// 001_MathUtils bridge: clamp meshPerAttribute to a positive integer >= 1.
	clampMeshPerAttribute: ( value ) => {

		return MathUtils.clamp( Math.floor( value ), 1, Infinity );

	},

	// gl-matrix bridge: pack the instance header (stride, count, meshPerAttribute, version) into a vec4.
	packInstanceVec4: ( out, stride, count, meshPerAttribute, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, stride, count, meshPerAttribute, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register an InstancedInterleavedBuffer as a SoA instance column set.
	registerComponent: ( name, count, stride ) => {

		const dataColumn = new Float64Array( count * stride );
		const meshPerAttributeColumn = new Uint32Array( count );
		return { name, dataColumn, meshPerAttributeColumn, stride, count };

	},

	// double.js bridge: high-precision total instance bytes (count * stride * bytesPerElement).
	totalInstanceBytes: ( count, stride, bytesPerElement = 4 ) => {

		const c = new Double( count );
		const s = new Double( stride );
		const b = new Double( bytesPerElement );
		return c.mul( s ).mul( b ).valueOf();

	},

	// simplex-noise bridge: procedural meshPerAttribute helper.
	randomMeshPerAttribute: ( seed = 0, min = 1, max = 4 ) => {

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

class InstancedInterleavedBuffer extends InterleavedBuffer {

	/**
	 * @param {TypedArray} array - A typed array with a shared buffer storing attribute data.
	 * @param {number} stride - The number of typed-array elements per vertex.
	 * @param {number} [meshPerAttribute=1] - Defines how often a value of this interleaved buffer should be repeated.
	 */
	constructor( array, stride, meshPerAttribute = 1 ) {

		super( array, stride );

		this.isInstancedInterleavedBuffer = true;

		this.meshPerAttribute = InstancedInterleavedBufferUtils.clampMeshPerAttribute( meshPerAttribute );

	}

	copy( source ) {

		super.copy( source );

		this.meshPerAttribute = source.meshPerAttribute;

		return this;

	}

	clone( data ) {

		const ib = super.clone( data );

		ib.meshPerAttribute = this.meshPerAttribute;

		return ib;

	}

	toJSON( data ) {

		const json = super.toJSON( data );

		json.isInstancedInterleavedBuffer = true;
		json.meshPerAttribute = this.meshPerAttribute;

		return json;

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return InstancedInterleavedBufferUtils.packInstanceVec4(
			out,
			this.stride,
			this.count,
			this.meshPerAttribute,
			this.version
		);

	}

	asBitecsComponent( name ) {

		return InstancedInterleavedBufferUtils.registerComponent( name, this.count, this.stride );

	}

	getTotalInstanceBytes( bytesPerElement = 4 ) {

		return InstancedInterleavedBufferUtils.totalInstanceBytes(
			this.count,
			this.stride,
			bytesPerElement
		);

	}

	static randomMeshPerAttribute( seed, min, max ) {

		return InstancedInterleavedBufferUtils.randomMeshPerAttribute( seed, min, max );

	}

}

InstancedInterleavedBuffer.Utils = InstancedInterleavedBufferUtils;

export default InstancedInterleavedBuffer;
export { InstancedInterleavedBufferUtils };