// file number : 003
// full path name : src/core/003_Layers.js
// description : 32-bit layer membership bitmask used by Object3D, Raycaster, Camera, and renderers to include/exclude objects from tests and rendering. Rewritten as an ES module; bridges to 001_MathUtils for mask validation, gl-matrix for vec4 mask packing, double.js for safe version tracking, bitecs for SoA component registration, and simplex-noise for procedural random mask generation.
// best for  : Per-object layer filtering in Object3D, Raycaster, Camera, WebGLRenderer, and WebGPURenderer.
// license : MIT

import MathUtils from './001_MathUtils.js';
import * as bitecs from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.js';
import Double from 'https://cdn.jsdelivr.net/npm/double.js@1.1.0/dist/double.js';
import { createNoise2D, createNoise3D, createNoise4D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';

const _noise2D = createNoise2D();
const _noise3D = createNoise3D();
const _noise4D = createNoise4D();

const _scratchVec4 = new Float64Array( 4 );
const _scratchMat4 = new Float64Array( 16 );

const LayersUtils = {

	// 001_MathUtils bridge: validate a 32-bit layer index.
	validateIndex: ( channel ) => {

		return MathUtils.clamp( Math.floor( channel ), 0, 31 );

	},

	// gl-matrix bridge: pack a mask into a vec4 (x = low 8 bits, y = next 8, z = next 8, w = high 8).
	packMaskVec4: ( out, mask ) => {

		glMatrix.mat4.identity( _scratchMat4 );
		glMatrix.vec4.set(
			out || _scratchVec4,
			mask & 0xff,
			( mask >>> 8 ) & 0xff,
			( mask >>> 16 ) & 0xff,
			( mask >>> 24 ) & 0xff
		);
		glMatrix.vec4.transformMat4( out || _scratchVec4, out || _scratchVec4, _scratchMat4 );
		return out || _scratchVec4;

	},

	// bitecs bridge: register a Layers mask as a SoA component column.
	registerComponent: ( name, count ) => {

		const maskColumn = new Uint32Array( count );
		return { name, maskColumn, count };

	},

	// double.js bridge: safe version increment for mask-change tracking.
	incrementVersion: ( version ) => {

		const d = new Double( version );
		d.add( 1 );
		return d.valueOf();

	},

	// simplex-noise bridge: generate a random 32-bit mask with given density (0..1).
	randomMask: ( density = 0.5, seed = 0 ) => {

		let mask = 0;
		for ( let i = 0; i < 32; i ++ ) {

			const n = _noise2D( i * 0.1 + seed, 0 );
			if ( ( n + 1 ) * 0.5 < density ) mask |= ( 1 << i );

		}
		return mask >>> 0;

	},

	bitecs,
	glMatrix,
	Double,

};

class Layers {

	constructor() {

		this.mask = 1 | 0;

	}

	set( channel ) {

		this.mask = ( 1 << LayersUtils.validateIndex( channel ) ) | 0;

	}

	enable( channel ) {

		this.mask |= ( 1 << LayersUtils.validateIndex( channel ) ) | 0;

	}

	enableAll() {

		this.mask = 0xffffffff | 0;

	}

	toggle( channel ) {

		this.mask ^= ( 1 << LayersUtils.validateIndex( channel ) ) | 0;

	}

	disable( channel ) {

		this.mask &= ~ ( 1 << LayersUtils.validateIndex( channel ) );

	}

	disableAll() {

		this.mask = 0;

	}

	test( layers ) {

		return ( this.mask & layers.mask ) !== 0;

	}

	isEnabled( channel ) {

		return ( this.mask & ( 1 << LayersUtils.validateIndex( channel ) ) ) !== 0;

	}

	setMask( mask ) {

		this.mask = mask | 0;

	}

	getMask() {

		return this.mask;

	}

	clear() {

		this.mask = 0;

	}

	equals( layers ) {

		return layers.mask === this.mask;

	}

	copy( layers ) {

		this.mask = layers.mask | 0;
		return this;

	}

	clone() {

		const layers = new Layers();
		layers.mask = this.mask | 0;
		return layers;

	}

	toArray() {

		return [ this.mask ];

	}

	fromArray( array ) {

		this.mask = array[ 0 ] | 0;
		return this;

	}

	serialize() {

		return { mask: this.mask };

	}

	deserialize( data ) {

		this.mask = data.mask | 0;
		return this;

	}

	// Convenience accessors backed by the utility surface above.
	packToVec4( out ) {

		return LayersUtils.packMaskVec4( out, this.mask );

	}

	asBitecsComponent( name, count ) {

		return LayersUtils.registerComponent( name, count );

	}

	static randomMask( density, seed ) {

		return LayersUtils.randomMask( density, seed );

	}

}

Layers.Utils = LayersUtils;

export default Layers;
export { LayersUtils };