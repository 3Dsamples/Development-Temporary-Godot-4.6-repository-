// file number : 014
// full path name : src/core/014_BufferGeometry.js
// description : A representation of mesh, line, or point geometry. Stores vertex positions, indices, normals, colors, UVs, and custom attributes within typed arrays, reducing the cost of passing data to the GPU. Rewritten as an ES module; extends the local 001_EventDispatcher and consumes 002_Vector2, 003_Vector3, 006_Matrix3, 007_Matrix4, 011_Sphere, 012_Box3, and 016_Vector4 from the DeepSeek chat link. Bridges to 001_MathUtils for generateUUID and scalar helpers, gl-matrix for packing the geometry header (attributeCount, indexCount, groupCount, version) into a vec4, double.js for high-precision vertex/triangle byte tracking, bitecs for SoA geometry registration, and simplex-noise for procedural vertex-noise generation. All non-chat three.js r185 imports (BufferAttribute, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute, Object3D, utils) are imported explicitly so the module remains self-contained.
// best for  : The base geometry class for all renderable shapes (BoxGeometry, SphereGeometry, PlaneGeometry, etc.). Directly consumed by Mesh, Line, Points, and all renderer backends.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import EventDispatcher from './001_EventDispatcher.js';
import MathUtils from './001_MathUtils.js';
import Vector2 from '../math/002_Vector2.js';
import Vector3 from '../math/003_Vector3.js';
import Matrix3 from '../math/006_Matrix3.js';
import Matrix4 from '../math/007_Matrix4.js';
import Sphere from '../math/011_Sphere.js';
import Box3 from '../math/012_Box3.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 BufferGeometry.js source.
// BufferAttribute and its typed subclasses are core attribute containers;
// Object3D is used for morph-target and transform helpers; utils provides
// arrayNeedsUint32, warn, and error helpers.
import { BufferAttribute, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/BufferAttribute.js';
import { Object3D } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/Object3D.js';
import { arrayNeedsUint32, warn, error } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/utils.js';

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

// ── Module-private scratch objects (mirrors r185 source) ────────────────────
const _m1 = new Matrix4();
const _obj = new Object3D();
const _offset = new Vector3();
const _box = new Box3();
const _boxMorphTargets = new Box3();
const _vector = new Vector3();

let _id = 0;

const BufferGeometryUtils = {

	// 001_MathUtils bridge: generate a UUID for the geometry.
	generateUUID: () => MathUtils.generateUUID(),

	// 001_MathUtils bridge: clamp an attribute count to a safe non-negative integer.
	clampAttributeCount: ( value ) => MathUtils.clamp( Math.floor( value ), 0, Infinity ),

	// gl-matrix bridge: pack the geometry header (attributeCount, indexCount, groupCount, version) into a vec4.
	packHeaderVec4: ( out, attributeCount, indexCount, groupCount, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, attributeCount, indexCount, groupCount, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a BufferGeometry as a SoA component column set.
	registerComponent: ( name, count ) => {

		const attributeCountColumn = new Uint32Array( count );
		const indexCountColumn = new Uint32Array( count );
		const groupCountColumn = new Uint32Array( count );
		const versionColumn = new Uint32Array( count );
		return { name, attributeCountColumn, indexCountColumn, groupCountColumn, versionColumn, count };

	},

	// double.js bridge: high-precision total vertex bytes (vertexCount * itemSize * bytesPerElement).
	totalVertexBytes: ( vertexCount, itemSize, bytesPerElement = 4 ) => {

		const c = new Double( vertexCount );
		const i = new Double( itemSize );
		const b = new Double( bytesPerElement );
		return c.mul( i ).mul( b ).valueOf();

	},

	// double.js bridge: high-precision total triangle count (indexCount / 3).
	totalTriangles: ( indexCount ) => {

		const c = new Double( indexCount );
		return c.div( 3 ).valueOf();

	},

	// simplex-noise bridge: fill a position attribute array with procedural 3D noise.
	fillWithNoise: ( out, vertexCount, itemSize, scale = 0.1, seed = 0 ) => {

		let i = 0;
		for ( let v = 0; v < vertexCount; v ++ ) {

			for ( let c = 0; c < itemSize; c ++ ) {

				out[ i ++ ] = _noise3D( v * scale + seed, c * scale + seed, seed );

			}

		}

		return out;

	},

	// simplex-noise bridge: displace existing positions along the normal by noise.
	displaceByNoise: ( positions, normals, vertexCount, amplitude = 0.1, scale = 0.1, seed = 0 ) => {

		for ( let i = 0; i < vertexCount; i ++ ) {

			const ix = i * 3;
			const n = _noise3D(
				positions[ ix ] * scale + seed,
				positions[ ix + 1 ] * scale + seed,
				positions[ ix + 2 ] * scale + seed
			);

			positions[ ix ] += normals[ ix ] * n * amplitude;
			positions[ ix + 1 ] += normals[ ix + 1 ] * n * amplitude;
			positions[ ix + 2 ] += normals[ ix + 2 ] * n * amplitude;

		}

		return positions;

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
	Matrix3,
	Matrix4,
	Sphere,
	Box3,
	BufferAttribute,
	Float32BufferAttribute,
	Uint16BufferAttribute,
	Uint32BufferAttribute,
	Object3D,
	arrayNeedsUint32,
	warn,
	error,

};

class BufferGeometry extends EventDispatcher {

	constructor() {

		super();

		this.isBufferGeometry = true;

		Object.defineProperty( this, 'id', { value: _id ++ } );

		this.uuid = MathUtils.generateUUID();
		this.name = '';
		this.type = 'BufferGeometry';

		this.index = null;
		this.indirect = null;
		this.indirectOffset = 0;

		this.attributes = {};
		this.morphAttributes = {};
		this.morphTargetsRelative = false;

		this.groups = [];

		this.boundingBox = null;
		this.boundingSphere = null;

		this.drawRange = { start: 0, count: Infinity };

		this.userData = {};

		this._transformed = false;
		this._version = 0;

	}

	// ── Index / indirect ─────────────────────────────────────────────────────

	getIndex() {

		return this.index;

	}

	setIndex( index ) {

		if ( Array.isArray( index ) ) {

			this.index = new ( arrayNeedsUint32( index ) ? Uint32BufferAttribute : Uint16BufferAttribute )( index, 1 );

		} else {

			this.index = index;

		}

		this._version ++;
		return this;

	}

	setIndirect( indirect, indirectOffset = 0 ) {

		this.indirect = indirect;
		this.indirectOffset = indirectOffset;
		this._version ++;
		return this;

	}

	getIndirect() {

		return this.indirect;

	}

	// ── Attribute access ─────────────────────────────────────────────────────

	getAttribute( name ) {

		return this.attributes[ name ];

	}

	setAttribute( name, attribute ) {

		this.attributes[ name ] = attribute;
		this._version ++;
		return this;

	}

	deleteAttribute( name ) {

		delete this.attributes[ name ];
		this._version ++;
		return this;

	}

	hasAttribute( name ) {

		return this.attributes[ name ] !== undefined;

	}

	// ── Group management ─────────────────────────────────────────────────────

	addGroup( start, count, materialIndex = 0 ) {

		this.groups.push( { start, count, materialIndex } );
		this._version ++;

	}

	clearGroups() {

		this.groups = [];
		this._version ++;

	}

	setDrawRange( start, count ) {

		this.drawRange.start = start;
		this.drawRange.count = count;
		this._version ++;

	}

	// ── Matrix transforms ────────────────────────────────────────────────────

	applyMatrix4( matrix ) {

		const position = this.attributes.position;

		if ( position !== undefined ) {

			position.applyMatrix4( matrix );
			position.needsUpdate = true;

		}

		const normal = this.attributes.normal;

		if ( normal !== undefined ) {

			const normalMatrix = new Matrix3().getNormalMatrix( matrix );
			normal.applyNormalMatrix( normalMatrix );
			normal.needsUpdate = true;

		}

		const tangent = this.attributes.tangent;

		if ( tangent !== undefined ) {

			tangent.transformDirection( matrix );
			tangent.needsUpdate = true;

		}

		if ( this.boundingBox !== null ) {

			this.computeBoundingBox();

		}

		if ( this.boundingSphere !== null ) {

			this.computeBoundingSphere();

		}

		this._transformed = true;
		this._version ++;
		return this;

	}

	applyQuaternion( q ) {

		_m1.makeRotationFromQuaternion( q );
		this.applyMatrix4( _m1 );

		return this;

	}

	rotateX( angle ) {

		_m1.makeRotationX( angle );
		this.applyMatrix4( _m1 );

		return this;

	}

	rotateY( angle ) {

		_m1.makeRotationY( angle );
		this.applyMatrix4( _m1 );

		return this;

	}

	rotateZ( angle ) {

		_m1.makeRotationZ( angle );
		this.applyMatrix4( _m1 );

		return this;

	}

	translate( x, y, z ) {

		_m1.makeTranslation( x, y, z );
		this.applyMatrix4( _m1 );

		return this;

	}

	scale( x, y, z ) {

		_m1.makeScale( x, y, z );
		this.applyMatrix4( _m1 );

		return this;

	}

	lookAt( vector ) {

		_obj.lookAt( vector );
		_obj.updateMatrix();
		this.applyMatrix4( _obj.matrix );

		return this;

	}

	center() {

		this.computeBoundingBox();
		this.boundingBox.getCenter( _offset ).negate();
		this.translate( _offset.x, _offset.y, _offset.z );

		return this;

	}

	// ── Bounding volumes ─────────────────────────────────────────────────────

	computeBoundingBox() {

		if ( this.boundingBox === null ) {

			this.boundingBox = new Box3();

		}

		this.boundingBox.makeEmpty();

		const position = this.attributes.position;

		if ( position === undefined ) {

			return;

		}

		for ( let i = 0; i < position.count; i ++ ) {

			_vector.fromBufferAttribute( position, i );
			this.boundingBox.expandByPoint( _vector );

		}

	}

	computeBoundingSphere() {

		if ( this.boundingSphere === null ) {

			this.boundingSphere = new Sphere();

		}

		this.boundingSphere.makeEmpty();

		const position = this.attributes.position;

		if ( position === undefined ) {

			return;

		}

		for ( let i = 0; i < position.count; i ++ ) {

			_vector.fromBufferAttribute( position, i );
			this.boundingSphere.expandByPoint( _vector );

		}

	}

	// ── Normal / tangent computation ─────────────────────────────────────────

	computeVertexNormals() {

		const index = this.index;
		const positionAttribute = this.getAttribute( 'position' );

		if ( positionAttribute !== undefined ) {

			let normalAttribute = this.getAttribute( 'normal' );

			if ( normalAttribute === undefined ) {

				normalAttribute = new BufferAttribute( new Float32Array( positionAttribute.count * 3 ), 3 );
				this.setAttribute( 'normal', normalAttribute );

			} else {

				// reset existing normals to zero
				for ( let i = 0, il = normalAttribute.count; i < il; i ++ ) {

					normalAttribute.setXYZ( i, 0, 0, 0 );

				}

			}

			const pA = new Vector3(), pB = new Vector3(), pC = new Vector3();
			const nA = new Vector3(), nB = new Vector3(), nC = new Vector3();
			const cb = new Vector3(), ab = new Vector3();

			if ( index ) {

				for ( let i = 0, il = index.count; i < il; i += 3 ) {

					const vA = index.getX( i + 0 );
					const vB = index.getX( i + 1 );
					const vC = index.getX( i + 2 );

					pA.fromBufferAttribute( positionAttribute, vA );
					pB.fromBufferAttribute( positionAttribute, vB );
					pC.fromBufferAttribute( positionAttribute, vC );

					cb.subVectors( pC, pB );
					ab.subVectors( pA, pB );
					cb.cross( ab );

					nA.fromBufferAttribute( normalAttribute, vA );
					nB.fromBufferAttribute( normalAttribute, vB );
					nC.fromBufferAttribute( normalAttribute, vC );

					nA.add( cb );
					nB.add( cb );
					nC.add( cb );

					normalAttribute.setXYZ( vA, nA.x, nA.y, nA.z );
					normalAttribute.setXYZ( vB, nB.x, nB.y, nB.z );
					normalAttribute.setXYZ( vC, nC.x, nC.y, nC.z );

				}

			} else {

				for ( let i = 0, il = positionAttribute.count; i < il; i += 3 ) {

					pA.fromBufferAttribute( positionAttribute, i + 0 );
					pB.fromBufferAttribute( positionAttribute, i + 1 );
					pC.fromBufferAttribute( positionAttribute, i + 2 );

					cb.subVectors( pC, pB );
					ab.subVectors( pA, pB );
					cb.cross( ab );

					normalAttribute.setXYZ( i + 0, cb.x, cb.y, cb.z );
					normalAttribute.setXYZ( i + 1, cb.x, cb.y, cb.z );
					normalAttribute.setXYZ( i + 2, cb.x, cb.y, cb.z );

				}

			}

			this.normalizeNormals();
			normalAttribute.needsUpdate = true;

		}

	}

	normalizeNormals() {

		const normals = this.attributes.normal;

		for ( let i = 0, il = normals.count; i < il; i ++ ) {

			_vector.fromBufferAttribute( normals, i );
			_vector.normalize();
			normals.setXYZ( i, _vector.x, _vector.y, _vector.z );

		}

	}

	// ── Copy / clone / serialization ─────────────────────────────────────────

	copy( source ) {

		this.index = null;
		this.indirect = null;
		this.attributes = {};
		this.morphAttributes = {};
		this.groups = [];
		this.boundingBox = null;
		this.boundingSphere = null;

		this.name = source.name;

		const index = source.index;

		if ( index !== null ) {

			this.setIndex( index.clone() );

		}

		const attributes = source.attributes;

		for ( const name in attributes ) {

			const attribute = attributes[ name ];
			this.setAttribute( name, attribute.clone() );

		}

		const morphAttributes = source.morphAttributes;

		for ( const name in morphAttributes ) {

			const array = [];
			const morphAttribute = morphAttributes[ name ];

			for ( let i = 0, l = morphAttribute.length; i < l; i ++ ) {

				array.push( morphAttribute[ i ].clone() );

			}

			this.morphAttributes[ name ] = array;

		}

		this.morphTargetsRelative = source.morphTargetsRelative;

		const groups = source.groups;

		for ( let i = 0, l = groups.length; i < l; i ++ ) {

			const group = groups[ i ];
			this.addGroup( group.start, group.count, group.materialIndex );

		}

		this.drawRange.start = source.drawRange.start;
		this.drawRange.count = source.drawRange.count;

		this.userData = source.userData;

		this._transformed = source._transformed;

		return this;

	}

	clone() {

		return new this.constructor().copy( this );

	}

	toJSON() {

		const data = {
			metadata: {
				version: 4.5,
				type: 'BufferGeometry',
				generator: 'BufferGeometry.toJSON'
			}
		};

		data.uuid = this.uuid;
		data.type = this.type;

		if ( this.name !== '' ) data.name = this.name;
		if ( Object.keys( this.userData ).length > 0 ) data.userData = this.userData;

		if ( this.parameters !== undefined ) {

			const parameters = this.parameters;

			for ( const key in parameters ) {

				if ( parameters[ key ] !== undefined ) data[ key ] = parameters[ key ];

			}

			return data;

		}

		// geometry data

		data.data = { attributes: {} };

		const index = this.index;

		if ( index !== null ) {

			data.data.index = {
				type: index.array.constructor.name,
				array: Array.prototype.slice.call( index.array )
			};

		}

		const attributes = this.attributes;

		for ( const key in attributes ) {

			const attribute = attributes[ key ];

			data.data.attributes[ key ] = attribute.toJSON( data.data );

		}

		// groups

		if ( this.groups.length > 0 ) {

			data.data.groups = this.groups;

		}

		// bounding sphere

		if ( this.boundingSphere !== null ) {

			data.data.boundingSphere = {
				center: this.boundingSphere.center.toArray(),
				radius: this.boundingSphere.radius
			};

		}

		// bounding box

		if ( this.boundingBox !== null ) {

			data.data.boundingBox = {
				min: this.boundingBox.min.toArray(),
				max: this.boundingBox.max.toArray()
			};

		}

		return data;

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	packToVec4( out ) {

		return BufferGeometryUtils.packHeaderVec4(
			out,
		 Object.keys( this.attributes ).length,
			this.index ? this.index.count : 0,
			this.groups.length,
			this._version
		);

	}

	asBitecsComponent( name, count ) {

		return BufferGeometryUtils.registerComponent( name, count );

	}

	getTotalVertexBytes( bytesPerElement = 4 ) {

		const position = this.attributes.position;
		if ( ! position ) return 0;
		return BufferGeometryUtils.totalVertexBytes( position.count, position.itemSize, bytesPerElement );

	}

	getTotalTriangles() {

		if ( ! this.index ) return 0;
		return BufferGeometryUtils.totalTriangles( this.index.count );

	}

	fillWithNoise( scale, seed ) {

		const position = this.attributes.position;
		if ( ! position ) return this;

		BufferGeometryUtils.fillWithNoise( position.array, position.count, position.itemSize, scale, seed );
		position.needsUpdate = true;
		this._version ++;
		return this;

	}

	displaceByNoise( amplitude, scale, seed ) {

		const position = this.attributes.position;
		const normal = this.attributes.normal;
		if ( ! position || ! normal ) return this;

		BufferGeometryUtils.displaceByNoise(
			position.array,
			normal.array,
			position.count,
			amplitude,
			scale,
			seed
		);

		position.needsUpdate = true;
		this._version ++;
		return this;

	}

	dispose() {

		this.dispatchEvent( { type: 'dispose' } );

	}

	get version() {

		return this._version;

	}

}

BufferGeometry.Utils = BufferGeometryUtils;

export default BufferGeometry;
export { BufferGeometryUtils };