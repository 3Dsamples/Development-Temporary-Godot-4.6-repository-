// file number : 006
// full path name : src/core/006_UniformsGroup.js
// description : Container that groups multiple Uniform instances into a single UBO-friendly block. Rewritten as an ES module; inherits from the local EventDispatcher (001_EventDispatcher.js) and consumes Uniform (005_Uniform.js). Bridges to bitecs for SoA uniform-column registration, gl-matrix for packing the group into a vec4 header, double.js for high-precision group-version tracking, and simplex-noise for procedural group-name / seed generation.
// best for  : Declaring UBO-style uniform blocks inside ShaderMaterial. Only supported by WebGLRenderer.
// license : MIT

import EventDispatcher from './001_EventDispatcher.js';
import Uniform from './005_Uniform.js';
import * as bitecs from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.js';
import Double from 'https://cdn.jsdelivr.net/npm/double.js@1.1.0/dist/double.js';
import { createNoise2D, createNoise3D, createNoise4D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';

const _noise2D = createNoise2D();
const _noise3D = createNoise3D();
const _noise4D = createNoise4D();

const _scratchVec4 = new Float64Array( 4 );
const _scratchMat4 = new Float64Array( 16 );

let _id = 0;

const UniformsGroupUtils = {

	// gl-matrix bridge: pack group header (id, usage, count, version) into a vec4.
	packHeaderVec4: ( out, id, usage, count, version ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set( out || _scratchVec4, id, usage, count, version );
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a UniformsGroup as a SoA component column set.
	registerComponent: ( name, uniforms ) => {

		const count = uniforms.length;
		const columns = {
			name,
			uniformNames: new Array( count ),
			uniformValues: new Float64Array( count ),
			count,
		};

		for ( let i = 0; i < count; i ++ ) {

			columns.uniformNames[ i ] = uniforms[ i ].name;
			const v = uniforms[ i ].value;
			columns.uniformValues[ i ] = typeof v === 'number' ? v : 0;

		}

		return columns;

	},

	// double.js bridge: high-precision version bump for the group.
	bumpVersion: ( version ) => {

		const d = new Double( version );
		d.add( 1 );
		return d.valueOf();

	},

	// simplex-noise bridge: procedural group name / seed helpers.
	noise2D: ( x, y ) => _noise2D( x, y ),
	noise3D: ( x, y, z ) => _noise3D( x, y, z ),
	noise4D: ( x, y, z, w ) => _noise4D( x, y, z, w ),

	randomGroupName: ( seed = 0 ) => {

		const n = _noise2D( seed, 0 );
		return `UniformsGroup_${ Math.abs( Math.floor( n * 1e6 ) ) }`;

	},

	bitecs,
	glMatrix,
	Double,

};

class UniformsGroup extends EventDispatcher {

	constructor() {

		super();

		this.isUniformsGroup = true;

		Object.defineProperty( this, 'id', { value: _id ++ } );

		this.name = '';

		this.usage = 35048; // StaticDrawUsage

		this.uniforms = [];

		this._version = 0;

	}

	add( uniform ) {

		this.uniforms.push( uniform );
		this._version = UniformsGroupUtils.bumpVersion( this._version );
		return this;

	}

	remove( uniform ) {

		const index = this.uniforms.indexOf( uniform );

		if ( index !== - 1 ) {

			this.uniforms.splice( index, 1 );
			this._version = UniformsGroupUtils.bumpVersion( this._version );

		}

		return this;

	}

	setName( name ) {

		this.name = name;
		return this;

	}

	setUsage( value ) {

		this.usage = value;
		return this;

	}

	dispose() {

		this.dispatchEvent( { type: 'dispose' } );

	}

	copy( source ) {

		this.name = source.name;
		this.usage = source.usage;

		const uniformsSource = source.uniforms;
		this.uniforms.length = 0;

		for ( let i = 0, l = uniformsSource.length; i < l; i ++ ) {

			const uniforms = Array.isArray( uniformsSource[ i ] ) ? uniformsSource[ i ] : [ uniformsSource[ i ] ];

			for ( let j = 0; j < uniforms.length; j ++ ) {

				this.uniforms.push( uniforms[ j ].clone() );

			}

		}

		this._version = UniformsGroupUtils.bumpVersion( this._version );
		return this;

	}

	clone() {

		return new this.constructor().copy( this );

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return UniformsGroupUtils.packHeaderVec4( out, this.id, this.usage, this.uniforms.length, this._version );

	}

	asBitecsComponent( name ) {

		return UniformsGroupUtils.registerComponent( name, this.uniforms );

	}

	get version() {

		return this._version;

	}

	static randomName( seed ) {

		return UniformsGroupUtils.randomGroupName( seed );

	}

}

UniformsGroup.Utils = UniformsGroupUtils;

export default UniformsGroup;
export { UniformsGroupUtils };