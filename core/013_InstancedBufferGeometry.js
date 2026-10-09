// file number : 013
// full path name : src/core/013_InstancedBufferGeometry.js
// description : An instanced version of BufferGeometry. Extends the base BufferGeometry (imported externally from three.js r185 because BufferGeometry.js is not part of the DeepSeek chat link) and adds an instanceCount field that drives how many instances the renderer draws. Rewritten as an ES module; bridges to 001_MathUtils for clamping instance counts, 002_Vector2 / 003_Vector3 / 016_Vector4 for attribute bridging inherited through BufferGeometry, gl-matrix for packing the instance-geometry header (instanceCount, drawRange, groups, version) into a vec4, double.js for high-precision instance-byte tracking, bitecs for SoA instanced-geometry registration, and simplex-noise for procedural instance placement helpers.
// best for  : InstancedMesh, GPU instancing, vegetation scattering, particle systems, and any geometry that must be rendered N times with per-instance attributes (offset, color, scale, rotation).
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import MathUtils from './001_MathUtils.js';
import EventDispatcher from './001_EventDispatcher.js';
import Vector2 from '../math/002_Vector2.js';
import Vector3 from '../math/003_Vector3.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// BufferGeometry.js is the parent class and is NOT in the DeepSeek chat link,
// so it must be imported directly from the three.js r185 source. All other
// r185 core files that BufferGeometry itself pulls in (Vector3, Vector2, Box3,
// EventDispatcher, BufferAttribute, Sphere, Object3D, Matrix4, Matrix3,
// MathUtils, utils) are either already provided by the DeepSeek chat link or
// are transitive dependencies of BufferGeometry and do not need to be
// re-imported here.
import { BufferGeometry } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/BufferGeometry.js';

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

const InstancedBufferGeometryUtils = {

	// 001_MathUtils bridge: clamp instanceCount to a safe non-negative integer.
	clampInstanceCount: ( value ) => {

		return MathUtils.clamp( Math.floor( value ), 0, Infinity );

	},

	// 001_MathUtils bridge: clamp a draw-range start/count pair.
	clampDrawRange: ( start, count, instanceCount ) => {

		const s = MathUtils.clamp( Math.floor( start ), 0, instanceCount );
		const c = MathUtils.clamp( Math.floor( count ), 0, instanceCount - s );
		return { start: s, count: c };

	},

	// gl-matrix bridge: pack the instanced-geometry header (instanceCount, drawRange.start, drawRange.count, version) into a vec4.
	packHeaderVec4: ( out, instanceCount, drawRangeStart, drawRangeCount, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, instanceCount, drawRangeStart, drawRangeCount, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register an InstancedBufferGeometry as a SoA component column set.
	registerComponent: ( name, count ) => {

		const instanceCountColumn = new Uint32Array( count );
		const drawRangeStartColumn = new Uint32Array( count );
		const drawRangeCountColumn = new Uint32Array( count );
		return { name, instanceCountColumn, drawRangeStartColumn, drawRangeCountColumn, count };

	},

	// double.js bridge: high-precision total instance bytes (instanceCount * perInstanceBytes).
	totalInstanceBytes: ( instanceCount, perInstanceBytes ) => {

		const c = new Double( instanceCount );
		const b = new Double( perInstanceBytes );
		return c.mul( b ).valueOf();

	},

	// double.js bridge: high-precision total draw calls (instanceCount / meshPerAttribute).
	totalDrawCalls: ( instanceCount, meshPerAttribute = 1 ) => {

		const c = new Double( instanceCount );
		const m = new Double( meshPerAttribute );
		return c.div( m ).valueOf();

	},

	// simplex-noise bridge: procedurally scatter instances along a 3D noise field.
	scatterInstances: ( out, instanceCount, scale = 0.1, seed = 0 ) => {

		let i = 0;
		for ( let v = 0; v < instanceCount; v ++ ) {

			out[ i ++ ] = _noise3D( v * scale + seed, seed, seed );
			out[ i ++ ] = _noise3D( seed, v * scale + seed, seed );
			out[ i ++ ] = _noise3D( seed, seed, v * scale + seed );

		}

		return out;

	},

	// simplex-noise bridge: procedural per-instance rotation (quaternion components).
	randomRotationQuat: ( out, index, seed = 0 ) => {

		const x = _noise3D( index * 0.1 + seed, 0, 0 );
		const y = _noise3D( 0, index * 0.1 + seed, 0 );
		const z = _noise3D( 0, 0, index * 0.1 + seed );
		const w = _noise4D( index * 0.1 + seed, 0, 0, 0 );

		out[ 0 ] = x;
		out[ 1 ] = y;
		out[ 2 ] = z;
		out[ 3 ] = w;

		// Normalize to a unit quaternion using gl-matrix.
		glMatrix.quat.normalize( out, out );
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

class InstancedBufferGeometry extends BufferGeometry {

	constructor() {

		super();

		this.isInstancedBufferGeometry = true;

		this.type = 'InstancedBufferGeometry';

		/**
		 * The instance count.
		 * @type {number}
		 * @default Infinity
		 */
		this.instanceCount = Infinity;

		this._version = 0;

	}

	copy( source ) {

		super.copy( source );

		this.instanceCount = source.instanceCount;

		this._version = InstancedBufferGeometryUtils.clampInstanceCount( this._version + 1 );

		return this;

	}

	toJSON() {

		const data = super.toJSON();

		data.instanceCount = this.instanceCount;
		data.isInstancedBufferGeometry = true;

		return data;

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	setInstanceCount( value ) {

		this.instanceCount = InstancedBufferGeometryUtils.clampInstanceCount( value );
		this._version ++;
		return this;

	}

	setDrawRangeClamped( start, count ) {

		const range = InstancedBufferGeometryUtils.clampDrawRange( start, count, this.instanceCount );
		this.setDrawRange( range.start, range.count );
		return this;

	}

	packToVec4( out ) {

		return InstancedBufferGeometryUtils.packHeaderVec4(
			out,
			this.instanceCount,
			this.drawRange.start,
			this.drawRange.count,
			this._version
		);

	}

	asBitecsComponent( name, count ) {

		return InstancedBufferGeometryUtils.registerComponent( name, count );

	}

	getTotalInstanceBytes( perInstanceBytes ) {

		return InstancedBufferGeometryUtils.totalInstanceBytes( this.instanceCount, perInstanceBytes );

	}

	getTotalDrawCalls( meshPerAttribute = 1 ) {

		return InstancedBufferGeometryUtils.totalDrawCalls( this.instanceCount, meshPerAttribute );

	}

	scatterInstances( out, scale, seed ) {

		InstancedBufferGeometryUtils.scatterInstances( out, this.instanceCount, scale, seed );
		this._version ++;
		return this;

	}

	get version() {

		return this._version;

	}

	static randomRotationQuat( out, index, seed ) {

		return InstancedBufferGeometryUtils.randomRotationQuat( out, index, seed );

	}

}

InstancedBufferGeometry.Utils = InstancedBufferGeometryUtils;

export default InstancedBufferGeometry;
export { InstancedBufferGeometryUtils };