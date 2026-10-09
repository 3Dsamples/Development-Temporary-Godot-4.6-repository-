// file number : 012
// full path name : src/core/012_InstancedBufferAttribute.js
// description : Instanced version of BufferAttribute. Adds a meshPerAttribute field that defines how often a value of the buffer attribute should be repeated across consecutive instances. Rewritten as an ES module; extends the local 011_BufferAttribute and reuses its entire array/itemSize/normalized/usage surface. Bridges to 001_MathUtils for clamping meshPerAttribute, gl-matrix for packing the instance-attribute header (itemSize, count, meshPerAttribute, version) into a vec4, double.js for high-precision instance-byte tracking, bitecs for SoA instance-attribute registration, and simplex-noise for procedural per-instance data generation. All non-chat three.js r185 imports (StaticDrawUsage, FloatType, DataUtils) are inherited through BufferAttribute; no additional three.js r185 src/ file is required for this module.
// best for  : Per-instance vertex attributes (offset matrices, colors, scales) consumed by InstancedMesh and InstancedBufferGeometry. Enables GPU instancing where each instance reads its own attribute record.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import BufferAttribute from './011_BufferAttribute.js';
import MathUtils from './001_MathUtils.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// The original r185 InstancedBufferAttribute.js only imports BufferAttribute,
// which is already provided by the DeepSeek chat link as 011_BufferAttribute.js.
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

const InstancedBufferAttributeUtils = {

	// 001_MathUtils bridge: clamp meshPerAttribute to a positive integer >= 1.
	clampMeshPerAttribute: ( value ) => {

		return MathUtils.clamp( Math.floor( value ), 1, Infinity );

	},

	// gl-matrix bridge: pack the instanced-attribute header (itemSize, count, meshPerAttribute, version) into a vec4.
	packInstanceVec4: ( out, itemSize, count, meshPerAttribute, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, itemSize, count, meshPerAttribute, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register an InstancedBufferAttribute as a SoA instance column.
	registerComponent: ( name, count, itemSize ) => {

		const column = new Float64Array( count * itemSize );
		const meshPerAttributeColumn = new Uint32Array( count );
		return { name, column, meshPerAttributeColumn, itemSize, count };

	},

	// double.js bridge: high-precision total instance bytes (count * itemSize * bytesPerElement).
	totalInstanceBytes: ( count, itemSize, bytesPerElement = 4 ) => {

		const c = new Double( count );
		const i = new Double( itemSize );
		const b = new Double( bytesPerElement );
		return c.mul( i ).mul( b ).valueOf();

	},

	// simplex-noise bridge: fill an instanced attribute array with procedural noise.
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

};

class InstancedBufferAttribute extends BufferAttribute {

	/**
	 * @param {TypedArray} array - The array holding the attribute data.
	 * @param {number} itemSize - The item size.
	 * @param {boolean} [normalized=false] - Whether the data are normalized or not.
	 * @param {number} [meshPerAttribute=1] - How often a value of this buffer attribute should be repeated.
	 */
	constructor( array, itemSize, normalized = false, meshPerAttribute = 1 ) {

		super( array, itemSize, normalized );

		this.isInstancedBufferAttribute = true;

		this.meshPerAttribute = InstancedBufferAttributeUtils.clampMeshPerAttribute( meshPerAttribute );

	}

	copy( source ) {

		super.copy( source );

		this.meshPerAttribute = source.meshPerAttribute;

		return this;

	}

	clone() {

		return new this.constructor( this.array, this.itemSize ).copy( this );

	}

	toJSON() {

		const data = super.toJSON();

		data.meshPerAttribute = this.meshPerAttribute;
		data.isInstancedBufferAttribute = true;

		return data;

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	packToVec4( out ) {

		return InstancedBufferAttributeUtils.packInstanceVec4(
			out,
			this.itemSize,
			this.count,
			this.meshPerAttribute,
			this.version
		);

	}

	asBitecsComponent( name ) {

		return InstancedBufferAttributeUtils.registerComponent( name, this.count, this.itemSize );

	}

	getTotalInstanceBytes( bytesPerElement = 4 ) {

		return InstancedBufferAttributeUtils.totalInstanceBytes(
			this.count,
			this.itemSize,
			bytesPerElement
		);

	}

	fillWithNoise( scale, seed ) {

		InstancedBufferAttributeUtils.fillWithNoise(
			this.array,
			this.count,
			this.itemSize,
			scale,
			seed
		);
		this.needsUpdate = true;
		return this;

	}

}

InstancedBufferAttribute.Utils = InstancedBufferAttributeUtils;

export default InstancedBufferAttribute;
export { InstancedBufferAttributeUtils };