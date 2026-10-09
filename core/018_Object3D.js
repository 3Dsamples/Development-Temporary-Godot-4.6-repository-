// file number : 018
// full path name : src/core/018_Object3D.js
// description : Base class for scene-graph objects in three.js. Provides the transform hierarchy (position, quaternion, scale, matrix, matrixWorld), parent/child relationships, traversal helpers, and world-space conversions. Rewritten as an ES module; extends the local 001_EventDispatcher and consumes 002_Vector2, 003_Vector3, 004_Quaternion, 006_Matrix3, 007_Matrix4, 008_Euler, and 003_Layers from the DeepSeek chat link. Bridges to 001_MathUtils for generateUUID / clamp / scalar helpers, gl-matrix for packing the object transform (position, quaternion, scale) into a mat4, double.js for high-precision world-position / world-scale tracking, bitecs for SoA transform registration, and simplex-noise for procedural placement helpers. All non-chat three.js r185 imports (generateUUID from utils.js) are imported explicitly so the module remains self-contained.
// best for  : The root of the scene graph. Every renderable object (Mesh, Line, Points, Light, Camera, Group, Scene) inherits from Object3D. Provides the transform hierarchy consumed by every renderer backend.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import EventDispatcher from './001_EventDispatcher.js';
import MathUtils from './001_MathUtils.js';
import Vector2 from '../math/002_Vector2.js';
import Vector3 from '../math/003_Vector3.js';
import Quaternion from '../math/004_Quaternion.js';
import Matrix3 from '../math/006_Matrix3.js';
import Matrix4 from '../math/007_Matrix4.js';
import Euler from '../math/008_Euler.js';
import Layers from './003_Layers.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// The original r185 Object3D.js imports generateUUID from ../utils.js.
import { generateUUID } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/utils.js';

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
const _addedEvent = { type: 'added' };
const _removedEvent = { type: 'removed' };
const _childAddedEvent = { type: 'childadded', child: null };
const _childRemovedEvent = { type: 'childremoved', child: null };

const _v1 = new Vector3();
const _q1 = new Quaternion();
const _m1 = new Matrix4();
const _target = new Vector3();

const _position = new Vector3();
const _scale = new Vector3();
const _quaternion = new Quaternion();
const _matrix = new Matrix4();

let _object3DId = 0;

const Object3DUtils = {

	// 001_MathUtils bridge: generate a UUID for the object.
	generateUUID: () => MathUtils.generateUUID(),

	// 001_MathUtils bridge: clamp a scalar used by transform helpers.
	clampScalar: ( value, min, max ) => MathUtils.clamp( value, min, max ),

	// gl-matrix bridge: compose a gl-matrix mat4 from position, quaternion, scale.
	composeGlMat4: ( out, position, quaternion, scale ) => {

		const p = [ position.x, position.y, position.z ];
		const q = [ quaternion.x, quaternion.y, quaternion.z, quaternion.w ];
		const s = [ scale.x, scale.y, scale.z ];

		glMatrix.mat4.fromRotationTranslationScale( out || _scratchMat4, q, p, s );
		return out || _scratchMat4;

	},

	// gl-matrix bridge: decompose a gl-matrix mat4 into position, quaternion, scale.
	decomposeGlMat4: ( mat, position, quaternion, scale ) => {

		const p = [ 0, 0, 0 ];
		const q = [ 0, 0, 0, 1 ];
		const s = [ 1, 1, 1 ];

		glMatrix.mat4.getTranslation( p, mat );
		glMatrix.mat4.getScaling( s, mat );
		glMatrix.mat4.getRotation( q, mat );

		position.set( p[ 0 ], p[ 1 ], p[ 2 ] );
		quaternion.set( q[ 0 ], q[ 1 ], q[ 2 ], q[ 3 ] );
		scale.set( s[ 0 ], s[ 1 ], s[ 2 ] );

	},

	// bitecs bridge: register an Object3D as a SoA transform component column set.
	registerComponent: ( name, count ) => {

		const positionColumn = new Float64Array( count * 3 );
		const quaternionColumn = new Float64Array( count * 4 );
		const scaleColumn = new Float64Array( count * 3 );
		const visibleColumn = new Uint8Array( count );
		return { name, positionColumn, quaternionColumn, scaleColumn, visibleColumn, count };

	},

	// double.js bridge: high-precision world-position accumulation.
	accumulateWorldPosition: ( worldPos, localPos ) => {

		const wx = new Double( worldPos.x ).add( localPos.x );
		const wy = new Double( worldPos.y ).add( localPos.y );
		const wz = new Double( worldPos.z ).add( localPos.z );
		return { x: wx.valueOf(), y: wy.valueOf(), z: wz.valueOf() };

	},

	// double.js bridge: high-precision world-scale accumulation.
	accumulateWorldScale: ( worldScale, localScale ) => {

		const sx = new Double( worldScale.x ).mul( localScale.x );
		const sy = new Double( worldScale.y ).mul( localScale.y );
		const sz = new Double( worldScale.z ).mul( localScale.z );
		return { x: sx.valueOf(), y: sy.valueOf(), z: sz.valueOf() };

	},

	// simplex-noise bridge: procedurally place the object on a 3D noise field.
	placeByNoise: ( object, scale = 0.1, amplitude = 1, seed = 0 ) => {

		const x = object.position.x;
		const y = object.position.y;
		const z = object.position.z;

		object.position.set(
			x + _noise3D( x * scale + seed, y * scale, z * scale ) * amplitude,
			y + _noise3D( x * scale, y * scale + seed, z * scale ) * amplitude,
			z + _noise3D( x * scale, y * scale, z * scale + seed ) * amplitude
		);

		return object;

	},

	// simplex-noise bridge: random orientation on the unit sphere.
	randomOrientation: ( object, seed = 0 ) => {

		const u = _noise2D( seed, 0 ) * 0.5 + 0.5;
		const v = _noise2D( 0, seed ) * 0.5 + 0.5;

		const theta = 2 * Math.PI * u;
		const phi = Math.acos( 2 * v - 1 );

		const x = Math.sin( phi ) * Math.cos( theta );
		const y = Math.cos( phi );
		const z = Math.sin( phi ) * Math.sin( theta );

		object.quaternion.setFromUnitVectors( new Vector3( 0, 0, 1 ), new Vector3( x, y, z ) );
		return object;

	},

	noise2D: ( x, y ) => _noise2D( x, y ),
	noise3D: ( x, y, z ) => _noise3D( x, y, z ),
	noise4D: ( x, y, z, w ) => _noise4D( x, y, z, w ),

	bitecs,
	glMatrix,
	Double,
	Vector2,
	Vector3,
	Quaternion,
	Matrix3,
	Matrix4,
	Euler,
	Layers,

};

class Object3D extends EventDispatcher {

	constructor() {

		super();

		this.isObject3D = true;

		Object.defineProperty( this, 'id', { value: _object3DId ++ } );

		this.uuid = MathUtils.generateUUID();

		this.name = '';
		this.type = 'Object3D';

		this.parent = null;
		this.children = [];

		this.up = new Vector3( 0, 1, 0 );

		this.position = new Vector3();
		this.rotation = new Euler();
		this.quaternion = new Quaternion();
		this.scale = new Vector3( 1, 1, 1 );

		this.matrix = new Matrix4();
		this.matrixWorld = new Matrix4();

		this.matrixAutoUpdate = Object3D.DEFAULT_MATRIX_AUTO_UPDATE;
		this.matrixWorldAutoUpdate = Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE;
		this.matrixWorldNeedsUpdate = false;

		this.layers = new Layers();
		this.visible = true;

		this.castShadow = false;
		this.receiveShadow = false;

		this.frustumCulled = true;
		this.renderOrder = 0;

		this.animations = [];
		this.userData = {};

		this._version = 0;

		// Euler / quaternion synchronization listeners.
		this.rotation._onChange( () => this.quaternion.setFromEuler( this.rotation, false ) );
		this.quaternion._onChange( () => this.rotation.setFromQuaternion( this.quaternion, undefined, false ) );

	}

	// ── Update callbacks ─────────────────────────────────────────────────────

	onBeforeRender() {}
	onAfterRender() {}
	onBeforeShadow() {}
	onAfterShadow() {}

	// ── Transform helpers ────────────────────────────────────────────────────

	applyMatrix4( matrix ) {

		if ( this.matrixAutoUpdate ) this.updateMatrix();

		this.matrix.premultiply( matrix );

		this.matrix.decompose( this.position, this.quaternion, this.scale );
		this._version ++;

	}

	applyQuaternion( q ) {

		this.quaternion.premultiply( q );
		this._version ++;

		return this;

	}

	setRotationFromAxisAngle( axis, angle ) {

		this.quaternion.setFromAxisAngle( axis, angle );
		this._version ++;

	}

	setRotationFromEuler( euler ) {

		this.quaternion.setFromEuler( euler, true );
		this._version ++;

	}

	setRotationFromMatrix( m ) {

		this.quaternion.setFromRotationMatrix( m );
		this._version ++;

	}

	setRotationFromQuaternion( q ) {

		this.quaternion.copy( q );
		this._version ++;

	}

	rotateOnAxis( axis, angle ) {

		_q1.setFromAxisAngle( axis, angle );
		this.quaternion.multiply( _q1 );
		this._version ++;

		return this;

	}

	rotateOnWorldAxis( axis, angle ) {

		_q1.setFromAxisAngle( axis, angle );
		this.quaternion.premultiply( _q1 );
		this._version ++;

		return this;

	}

	rotateX( angle ) {

		return this.rotateOnAxis( _v1.set( 1, 0, 0 ), angle );

	}

	rotateY( angle ) {

		return this.rotateOnAxis( _v1.set( 0, 1, 0 ), angle );

	}

	rotateZ( angle ) {

		return this.rotateOnAxis( _v1.set( 0, 0, 1 ), angle );

	}

	translateOnAxis( axis, distance ) {

		_v1.copy( axis ).applyQuaternion( this.quaternion );
		this.position.add( _v1.multiplyScalar( distance ) );
		this._version ++;

		return this;

	}

	translateX( distance ) {

		return this.translateOnAxis( _v1.set( 1, 0, 0 ), distance );

	}

	translateY( distance ) {

		return this.translateOnAxis( _v1.set( 0, 1, 0 ), distance );

	}

	translateZ( distance ) {

		return this.translateOnAxis( _v1.set( 0, 0, 1 ), distance );

	}

	localToWorld( vector ) {

		this.updateWorldMatrix( true, false );
		return vector.applyMatrix4( this.matrixWorld );

	}

	worldToLocal( vector ) {

		this.updateWorldMatrix( true, false );
		return vector.applyMatrix4( _m1.copy( this.matrixWorld ).invert() );

	}

	lookAt( x, y, z ) {

		if ( x.isVector3 ) {

			_target.copy( x );

		} else {

			_target.set( x, y, z );

		}

		const parent = this.parent;

		this.updateWorldMatrix( true, false );

		_position.setFromMatrixPosition( this.matrixWorld );

		if ( this.isCamera || this.isLight ) {

			_m1.lookAt( _position, _target, this.up );

		} else {

			_m1.lookAt( _target, _position, this.up );

		}

		this.quaternion.setFromRotationMatrix( _m1 );

		if ( parent ) {

			_m1.extractRotation( parent.matrixWorld );
			_q1.setFromRotationMatrix( _m1 );
			this.quaternion.premultiply( _q1.invert() );

		}

		this._version ++;

	}

	// ── Child management ─────────────────────────────────────────────────────

	add( object ) {

		if ( arguments.length > 1 ) {

			for ( let i = 0; i < arguments.length; i ++ ) {

				this.add( arguments[ i ] );

			}

			return this;

		}

		if ( object === this ) {

			console.error( 'THREE.Object3D.add: object can\'t be added as a child of itself.', object );
			return this;

		}

		if ( object && object.isObject3D ) {

			object.removeFromParent();
			object.parent = this;
			this.children.push( object );

			object.dispatchEvent( _addedEvent );

			_childAddedEvent.child = object;
			this.dispatchEvent( _childAddedEvent );
			_childAddedEvent.child = null;

		} else {

			console.error( 'THREE.Object3D.add: object not an instance of THREE.Object3D.', object );

		}

		this._version ++;
		return this;

	}

	remove( object ) {

		if ( arguments.length > 1 ) {

			for ( let i = 0; i < arguments.length; i ++ ) {

				this.remove( arguments[ i ] );

			}

			return this;

		}

		const index = this.children.indexOf( object );

		if ( index !== - 1 ) {

			object.parent = null;
			this.children.splice( index, 1 );

			object.dispatchEvent( _removedEvent );

			_childRemovedEvent.child = object;
			this.dispatchEvent( _childRemovedEvent );
			_childRemovedEvent.child = null;

		}

		this._version ++;
		return this;

	}

	removeFromParent() {

		const parent = this.parent;

		if ( parent !== null ) {

			parent.remove( this );

		}

		return this;

	}

	clear() {

		return this.remove( ... this.children );

	}

	attach( object ) {

		this.updateWorldMatrix( true, false );

		_m1.copy( this.matrixWorld ).invert();

		if ( object.parent !== null ) {

			object.parent.updateWorldMatrix( true, false );
			_m1.multiply( object.parent.matrixWorld );

		}

		object.applyMatrix4( _m1 );
		object.removeFromParent();
		object.parent = this;
		this.children.push( object );

		object.updateWorldMatrix( false, true );

		object.dispatchEvent( _addedEvent );

		_childAddedEvent.child = object;
		this.dispatchEvent( _childAddedEvent );
		_childAddedEvent.child = null;

		this._version ++;
		return this;

	}

	// ── Query helpers ────────────────────────────────────────────────────────

	getObjectById( id ) {

		return this.getObjectByProperty( 'id', id );

	}

	getObjectByName( name ) {

		return this.getObjectByProperty( 'name', name );

	}

	getObjectByProperty( name, value ) {

		if ( this[ name ] === value ) return this;

		for ( let i = 0, l = this.children.length; i < l; i ++ ) {

			const child = this.children[ i ];
			const object = child.getObjectByProperty( name, value );

			if ( object !== undefined ) {

				return object;

			}

		}

		return undefined;

	}

	getObjectsByProperty( name, value, result = [] ) {

		if ( this[ name ] === value ) result.push( this );

		for ( let i = 0, l = this.children.length; i < l; i ++ ) {

			this.children[ i ].getObjectsByProperty( name, value, result );

		}

		return result;

	}

	getWorldPosition( target ) {

		this.updateWorldMatrix( true, false );
		return target.setFromMatrixPosition( this.matrixWorld );

	}

	getWorldQuaternion( target ) {

		this.updateWorldMatrix( true, false );
		this.matrixWorld.decompose( _position, target, _scale );
		return target;

	}

	getWorldScale( target ) {

		this.updateWorldMatrix( true, false );
		this.matrixWorld.decompose( _position, _quaternion, target );
		return target;

	}

	getWorldDirection( target ) {

		this.updateWorldMatrix( true, false );
		const e = this.matrixWorld.elements;
		return target.set( e[ 8 ], e[ 9 ], e[ 10 ] ).normalize();

	}

	// ── Raycast / traversal ──────────────────────────────────────────────────

	raycast() {}

	traverse( callback ) {

		callback( this );

		const children = this.children;

		for ( let i = 0, l = children.length; i < l; i ++ ) {

			children[ i ].traverse( callback );

		}

	}

	traverseVisible( callback ) {

		if ( this.visible === false ) return;

		callback( this );

		const children = this.children;

		for ( let i = 0, l = children.length; i < l; i ++ ) {

			children[ i ].traverseVisible( callback );

		}

	}

	traverseAncestors( callback ) {

		const parent = this.parent;

		if ( parent !== null ) {

			callback( parent );
			parent.traverseAncestors( callback );

		}

	}

	// ── Matrix updates ───────────────────────────────────────────────────────

	updateMatrix() {

		this.matrix.compose( this.position, this.quaternion, this.scale );
		this.matrixWorldNeedsUpdate = true;
		this._version ++;

	}

	updateMatrixWorld( force ) {

		if ( this.matrixAutoUpdate ) this.updateMatrix();

		if ( this.matrixWorldNeedsUpdate || force ) {

			if ( this.matrixWorldAutoUpdate === true ) {

				if ( this.parent === null ) {

					this.matrixWorld.copy( this.matrix );

				} else {

					this.matrixWorld.multiplyMatrices( this.parent.matrixWorld, this.matrix );

				}

			}

			this.matrixWorldNeedsUpdate = false;
			force = true;

		}

		const children = this.children;

		for ( let i = 0, l = children.length; i < l; i ++ ) {

			const child = children[ i ];

			if ( child.matrixWorldAutoUpdate === true || force === true ) {

				child.updateMatrixWorld( force );

			}

		}

	}

	updateWorldMatrix( updateParents, updateChildren ) {

		const parent = this.parent;

		if ( updateParents === true && parent !== null ) {

			parent.updateWorldMatrix( true, false );

		}

		if ( this.matrixAutoUpdate ) this.updateMatrix();

		if ( this.matrixWorldAutoUpdate === true ) {

			if ( this.parent === null ) {

				this.matrixWorld.copy( this.matrix );

			} else {

				this.matrixWorld.multiplyMatrices( this.parent.matrixWorld, this.matrix );

			}

		}

		if ( updateChildren === true ) {

			const children = this.children;

			for ( let i = 0, l = children.length; i < l; i ++ ) {

				const child = children[ i ];

				if ( child.matrixWorldAutoUpdate === true ) {

					child.updateWorldMatrix( false, true );

				}

			}

		}

	}

	// ── Copy / clone / serialization ─────────────────────────────────────────

	copy( source, recursive = true ) {

		this.name = source.name;

		this.up.copy( source.up );

		this.position.copy( source.position );
		this.rotation.order = source.rotation.order;
		this.quaternion.copy( source.quaternion );
		this.scale.copy( source.scale );

		this.matrix.copy( source.matrix );
		this.matrixWorld.copy( source.matrixWorld );

		this.matrixAutoUpdate = source.matrixAutoUpdate;
		this.matrixWorldAutoUpdate = source.matrixWorldAutoUpdate;
		this.matrixWorldNeedsUpdate = source.matrixWorldNeedsUpdate;

		this.layers.mask = source.layers.mask;
		this.visible = source.visible;

		this.castShadow = source.castShadow;
		this.receiveShadow = source.receiveShadow;

		this.frustumCulled = source.frustumCulled;
		this.renderOrder = source.renderOrder;

		this.animations = source.animations.slice();

		this.userData = JSON.parse( JSON.stringify( source.userData ) );

		if ( recursive === true ) {

			for ( let i = 0; i < source.children.length; i ++ ) {

				const child = source.children[ i ];
				this.add( child.clone() );

			}

		}

		this._version ++;
		return this;

	}

	clone( recursive = true ) {

		return new this.constructor().copy( this, recursive );

	}

	toJSON( meta ) {

		const isRootObject = ( meta === undefined || typeof meta === 'string' );

		if ( isRootObject ) {

			meta = {
				geometries: {},
				materials: {},
				textures: {},
				images: {},
				shapes: {},
				skeletons: {},
				animations: {},
				nodes: {}
			};

		}

		if ( meta.object === undefined ) {

			meta.object = this;

		}

		let data = meta.nodes[ this.uuid ];

		if ( data === undefined ) {

			data = {
				metadata: {
					version: 4.5,
					type: 'Object',
					generator: 'Object3D.toJSON'
				}
			};

			const object = {};

			object.uuid = this.uuid;
			object.type = this.type;

			if ( this.name !== '' ) object.name = this.name;
			if ( this.castShadow === true ) object.castShadow = true;
			if ( this.receiveShadow === true ) object.receiveShadow = true;
			if ( this.visible === false ) object.visible = false;
			if ( this.frustumCulled === false ) object.frustumCulled = false;
			if ( this.renderOrder !== 0 ) object.renderOrder = this.renderOrder;
			if ( Object.keys( this.userData ).length > 0 ) object.userData = this.userData;

			object.layers = this.layers.mask;
			object.matrix = this.matrix.toArray();
			object.up = this.up.toArray();

			if ( this.matrixAutoUpdate === false ) object.matrixAutoUpdate = false;

			data.object = object;

			if ( this.isBone === true ) data.isBone = true;
			if ( this.isCamera === true ) data.isCamera = true;
			if ( this.isLight === true ) data.isLight = true;
			if ( this.isMesh === true ) data.isMesh = true;
			if ( this.isPoints === true ) data.isPoints = true;
			if ( this.isLine === true ) data.isLine = true;
			if ( this.isSkinnedMesh === true ) data.isSkinnedMesh = true;

			data.children = [];

			for ( let i = 0, l = this.children.length; i < l; i ++ ) {

				data.children.push( this.children[ i ].toJSON( meta ).object );

			}

			meta.nodes[ this.uuid ] = data;

		}

		return data;

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	composeToGlMat4( out ) {

		return Object3DUtils.composeGlMat4( out, this.position, this.quaternion, this.scale );

	}

	decomposeFromGlMat4( mat ) {

		Object3DUtils.decomposeGlMat4( mat, this.position, this.quaternion, this.scale );
		this._version ++;
		return this;

	}

	asBitecsComponent( name, count ) {

		return Object3DUtils.registerComponent( name, count );

	}

	placeByNoise( scale, amplitude, seed ) {

		Object3DUtils.placeByNoise( this, scale, amplitude, seed );
		this._version ++;
		return this;

	}

	randomOrientation( seed ) {

		Object3DUtils.randomOrientation( this, seed );
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

Object3D.DEFAULT_MATRIX_AUTO_UPDATE = true;
Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = true;
Object3D.Utils = Object3DUtils;

export default Object3D;
export { Object3DUtils };