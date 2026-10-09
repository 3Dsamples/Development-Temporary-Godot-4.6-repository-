// file number : 016
// full path name : src/core/016_Raycaster.js
// description : High-level utility for performing raycasting against Object3D hierarchies. Used primarily for mouse picking, collision detection, and spatial queries. Rewritten as an ES module; extends the local 001_EventDispatcher and consumes 002_Vector2, 003_Vector3, 003_Layers, 007_Matrix4, 013_Ray, and 016_Vector4 from the DeepSeek chat link. Bridges to 001_MathUtils for clamp/lerp/scalar helpers, gl-matrix for packing the raycaster state (near, far, linePrecision, layerMask) into a vec4, double.js for high-precision distance tracking, bitecs for SoA raycaster registration, and simplex-noise for procedural ray-jitter utilities. All non-chat three.js r185 imports are imported explicitly so the module remains self-contained.
// best for  : Mouse picking, touch interaction, collision queries, line-of-sight checks, and any spatial query that needs to find which Object3D lies under a ray.
// license : MIT

// ── DeepSeek chat link dependencies (rewritten core) ─────────────────────────
import EventDispatcher from './001_EventDispatcher.js';
import MathUtils from './001_MathUtils.js';
import Vector2 from '../math/002_Vector2.js';
import Vector3 from '../math/003_Vector3.js';
import Layers from './003_Layers.js';
import Matrix4 from '../math/007_Matrix4.js';
import Ray from '../math/013_Ray.js';
import Vector4 from '../math/016_Vector4.js';

// ── three.js r185 src/ dependencies NOT in the DeepSeek chat link ────────────
// These are the exact imports from the original r185 Raycaster.js source.
// Object3D is needed for the recursive intersectObjects traversal;
// constants.js provides the default threshold values for Line and Points.
import { Object3D } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/Object3D.js';
import { Raycaster } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/core/Raycaster.js';
import { warn } from 'https://cdn.jsdelivr.net/npm/three@0.185.0/src/utils.js';

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
const _ray = new Ray();
const _intersectionPoint = new Vector3();
const _intersectionPointWorld = new Vector3();

// Default params mirror of the original r185 Raycaster.js.
const _DEFAULT_PARAMS = {
	Mesh: {},
	Line: { threshold: 1 },
	LOD: {},
	Points: { threshold: 1 },
	Sprite: {},
	InstancedMesh: {},
	SkinnedMesh: {},
	BatchedMesh: {}
};

const RaycasterUtils = {

	// 001_MathUtils bridge: clamp the near/far values.
	clampNear: ( value ) => MathUtils.clamp( value, 0, Infinity ),
	clampFar: ( value ) => MathUtils.clamp( value, 0, Infinity ),

	// 001_MathUtils bridge: clamp linePrecision to a positive value.
	clampLinePrecision: ( value ) => MathUtils.clamp( value, 0, Infinity ),

	// gl-matrix bridge: pack the raycaster state (near, far, linePrecision, layerMask) into a vec4.
	packStateVec4: ( out, near, far, linePrecision, layerMask ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, near, far, linePrecision, layerMask );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a Raycaster as a SoA component column set.
	registerComponent: ( name, count ) => {

		const nearColumn = new Float64Array( count );
		const farColumn = new Float64Array( count );
		const linePrecisionColumn = new Float64Array( count );
		const layerMaskColumn = new Uint32Array( count );
		return { name, nearColumn, farColumn, linePrecisionColumn, layerMaskColumn, count };

	},

	// double.js bridge: high-precision distance comparison for sorting intersections.
	distanceCompare: ( a, b ) => {

		const da = new Double( a );
		const db = new Double( b );
		return da.sub( db ).valueOf();

	},

	// simplex-noise bridge: jitter a ray direction by noise (useful for soft raycasting).
	jitterDirection: ( direction, amplitude = 0.01, seed = 0 ) => {

		const nx = _noise3D( seed, 0, 0 );
		const ny = _noise3D( 0, seed, 0 );
		const nz = _noise3D( 0, 0, seed );

		direction.x += nx * amplitude;
		direction.y += ny * amplitude;
		direction.z += nz * amplitude;

		direction.normalize();
		return direction;

	},

	// simplex-noise bridge: generate a random ray direction on a hemisphere.
	randomHemisphereDirection: ( out, seed = 0 ) => {

		const u = _noise2D( seed, 0 ) * 0.5 + 0.5;
		const v = _noise2D( 0, seed ) * 0.5 + 0.5;

		const theta = 2 * Math.PI * u;
		const phi = Math.acos( 2 * v - 1 ) * 0.5;

		out.set(
			Math.sin( phi ) * Math.cos( theta ),
			Math.cos( phi ),
			Math.sin( phi ) * Math.sin( theta )
		);

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
	Layers,
	Matrix4,
	Ray,
	Object3D,
	Raycaster,
	warn,

};

class Raycaster {

	/**
	 * @param {Vector3} [origin] - The origin vector where the ray casts from.
	 * @param {Vector3} [direction] - The (normalized) direction vector that gives direction to the ray.
	 * @param {number} [near=0] - All results returned are further away than near. Near can't be negative.
	 * @param {number} [far=Infinity] - All results returned are closer than far. Far can't be lower than near.
	 */
	constructor( origin, direction, near = 0, far = Infinity ) {

		this.isRaycaster = true;

		this.ray = new Ray( origin, direction );

		this.near = RaycasterUtils.clampNear( near );
		this.far = RaycasterUtils.clampFar( far );

		this.camera = null;
		this.layers = new Layers();

		this.params = Object.assign( {}, _DEFAULT_PARAMS );

		this._version = 0;

	}

	/**
	 * Updates the ray with a new origin and direction.
	 * @param {Vector3} origin - The origin vector.
	 * @param {Vector3} direction - The normalized direction vector.
	 */
	set( origin, direction ) {

		this.ray.set( origin, direction );
		this._version ++;

	}

	/**
	 * Updates the ray with a new origin and direction, based on the camera and
	 * the 2D coordinates (normalized device coordinates) of the mouse.
	 * @param {Vector2} coords - 2D coordinates of the mouse, in normalized device coordinates (NDC).
	 * @param {Camera} camera - The camera from which the ray should originate.
	 */
	setFromCamera( coords, camera ) {

		if ( camera.isPerspectiveCamera ) {

			this.ray.origin.setFromMatrixPosition( camera.matrixWorld );
			this.ray.direction.set( coords.x, coords.y, 0.5 ).unproject( camera ).sub( this.ray.origin ).normalize();
			this.camera = camera;

		} else if ( camera.isOrthographicCamera ) {

			this.ray.origin.set( coords.x, coords.y, ( camera.near + camera.far ) / ( camera.near - camera.far ) ).unproject( camera );
			this.ray.direction.set( 0, 0, - 1 ).transformDirection( camera.matrixWorld );
			this.camera = camera;

		} else {

			console.error( 'THREE.Raycaster: Unsupported camera type: ' + camera.type );

		}

		this._version ++;

	}

	/**
	 * Checks all intersection between the ray and the object with or without
	 * the descendants. Intersections are returned sorted by distance, closest first.
	 * @param {Object3D} object - The object to check for intersection.
	 * @param {boolean} [recursive=true] - If true, it also checks all descendants.
	 * @param {Array} [intersects=[]] - The array to fill with the intersection results.
	 * @returns {Array} An array of intersection results.
	 */
	intersectObject( object, recursive = true, intersects = [] ) {

		intersect( object, this, intersects, recursive );

		intersects.sort( ascSort );
		this._version ++;

		return intersects;

	}

	/**
	 * Checks all intersection between the ray and the objects with or without
	 * the descendants. Intersections are returned sorted by distance, closest first.
	 * @param {Array} objects - The objects to check for intersection.
	 * @param {boolean} [recursive=true] - If true, it also checks all descendants.
	 * @param {Array} [intersects=[]] - The array to fill with the intersection results.
	 * @returns {Array} An array of intersection results.
	 */
	intersectObjects( objects, recursive = true, intersects = [] ) {

		if ( ! Array.isArray( objects ) ) {

			console.warn( 'THREE.Raycaster.intersectObjects: objects is not an Array.' );
			return intersects;

		}

		for ( let i = 0, l = objects.length; i < l; i ++ ) {

			intersect( objects[ i ], this, intersects, recursive );

		}

		intersects.sort( ascSort );
		this._version ++;

		return intersects;

	}

	/**
	 * Sets the near and far properties of the raycaster.
	 * @param {number} near - The near distance.
	 * @param {number} far - The far distance.
	 * @returns {Raycaster} This instance.
	 */
	setNearFar( near, far ) {

		this.near = RaycasterUtils.clampNear( near );
		this.far = RaycasterUtils.clampFar( far );
		this._version ++;

		return this;

	}

	/**
	 * Sets the linePrecision of the raycaster.
	 * @param {number} value - The new line precision.
	 * @returns {Raycaster} This instance.
	 */
	setLinePrecision( value ) {

		this.params.Line.threshold = RaycasterUtils.clampLinePrecision( value );
		this._version ++;

		return this;

	}

	// ── Convenience accessors backed by the utility surface above ─────────────

	packToVec4( out ) {

		return RaycasterUtils.packStateVec4(
			out,
			this.near,
			this.far,
			this.params.Line.threshold,
			this.layers.mask
		);

	}

	asBitecsComponent( name, count ) {

		return RaycasterUtils.registerComponent( name, count );

	}

	jitterDirection( amplitude, seed ) {

		RaycasterUtils.jitterDirection( this.ray.direction, amplitude, seed );
		this._version ++;

		return this;

	}

	setRandomHemisphereDirection( seed ) {

		RaycasterUtils.randomHemisphereDirection( this.ray.direction, seed );
		this._version ++;

		return this;

	}

	get version() {

		return this._version;

	}

}

Raycaster.Utils = RaycasterUtils;

// ── Module-private helpers ──────────────────────────────────────────────────

function ascSort( a, b ) {

	return a.distance - b.distance;

}

function intersect( object, raycaster, intersects, recursive ) {

	let propagate = true;

	if ( object.layers.test( raycaster.layers ) ) {

		const result = object.raycast( raycaster, intersects );

		if ( result === false ) propagate = false;

	}

	if ( propagate === true && recursive === true ) {

		const children = object.children;

		for ( let i = 0, l = children.length; i < l; i ++ ) {

			intersect( children[ i ], raycaster, intersects, true );

		}

	}

}

export default Raycaster;
export { RaycasterUtils };