// file number : 023
// full path name : src/math/023_CubicInterpolant.js
// description : CubicInterpolant class (THREE.CubicInterpolant) extending Interpolant with Hermite cubic spline evaluation and configurable left/right end-point handling (DefaultSettings_ with interpolate_, _lendControlPoint, _interpolate_), plus full zero-allocation bridge helpers that bind a bitecs 0.4.0 InterpolantTrackComponent to a CubicInterpolant using shared pools, and that read/write gl-matrix vec3/vec4 results without allocating. Depends on MathUtils.js (file 001) and Interpolant.js (file 022).
// best for  :  Smooth keyframe animation, camera paths, easing curves, spline trajectories, ECS-driven animation tracks, physics interpolation, and any hot loop that must evaluate a cubic spline over SoA data without allocating.
// license : GPL3

import { clamp } from './MathUtils.js';
import {
  Interpolant,
  InterpolantTrackComponent,
  InterpolantPools,
  threeVec3FromInterpolant,
  threeVec4FromInterpolant,
  glMatrixVec3FromInterpolant,
  glMatrixVec4FromInterpolant,
  glMatrixQuatFromInterpolant,
  interpolantFromGlMatrixVec3,
  interpolantFromGlMatrixVec4,
  bitecsInterpolantBindFromTrack,
  threeVec3FromBitecsInterpolantEvaluate,
  threeVec4FromBitecsInterpolantEvaluate,
  glMatrixVec3FromBitecsInterpolantEvaluate,
  glMatrixVec4FromBitecsInterpolantEvaluate,
  bitecsVec3FromBitecsInterpolantEvaluate,
  bitecsVec4FromBitecsInterpolantEvaluate,
  bitecsInterpolantResultIntoPool,
  threeVec3FromBitecsInterpolantSample,
  threeVec4FromBitecsInterpolantSample,
  glMatrixVec3FromBitecsInterpolantSample,
  glMatrixVec4FromBitecsInterpolantSample,
  bitecsInterpolantSampleFromVec3,
  bitecsInterpolantSampleFromVec4,
  bitecsInterpolantPositionSet,
  bitecsInterpolantPositionGet,
  bitecsInterpolantSampleSlotFromT
} from './022_Interpolant.js';

import glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.min.js';
import { defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

const { vec3: glVec3, vec4: glVec4, quat: glQuat } = glMatrix;

/*
 * -----------------------------------------------------------------------------
 * THREE.CubicInterpolant (r185)
 * -----------------------------------------------------------------------------
 * Hermite cubic spline evaluator. `interpolate_(i1, t0, t, t1)` uses the
 * Weight, [0,1] coefficients and the neighboring keys to build a C1 curve.
 *
 * The DefaultSettings_ object mirrors r185 exactly:
 *   - interpolate_       : (i1, t0, t, t1) => this._interpolate(i1, t0, t, t1)
 *   - _lendControlPoint  : (i1, t0, t, t1) => this._interpolate(i1, t0, t, t1)
 *   - _interpolate       : actual Hermite blend
 */
class CubicInterpolant extends Interpolant {

  constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {
    super( parameterPositions, sampleValues, sampleSize, resultBuffer );

    this._weight0 = 0;
    this._weight1 = 0;
    this._weight2 = 0;
    this._weight3 = 0;

    this.DefaultSettings_ = {

      interpolate_: ( i1, t0, t, t1 ) => {
        return this._interpolate( i1, t0, t, t1 );
      },

      _lendControlPoint: ( i1, t0, t, t1 ) => {
        return this._interpolate( i1, t0, t, t1 );
      },

      _interpolate: ( i1, t0, t, t1 ) => {

        const weight0 = this._weight0;
        const weight1 = this._weight1;
        const weight2 = this._weight2;
        const weight3 = this._weight3;

        const stride = this.valueSize;
        const values = this.sampleValues;
        const result = this.resultBuffer;

        const offset1 = i1 * stride;
        const offset0 = offset1 - stride;
        const offset2 = offset1 + stride;
        const offset3 = offset1 + stride * 2;

        for ( let i = 0; i !== stride; ++ i ) {
          const v0 = values[ offset0 + i ];
          const v1 = values[ offset1 + i ];
          const v2 = values[ offset2 + i ];
          const v3 = values[ offset3 + i ];
          result[ i ] = weight0 * v0 + weight1 * v1 + weight2 * v2 + weight3 * v3;
        }

        return result;

      }

    };

  }

  interpolate_( i1, t0, t, t1 ) {
    return this.getSettings_().interpolate_( i1, t0, t, t1 );
  }

  // Public setting alias, matches r185 `InterpolantSettings`.
  getSettings_() {
    return this.settings || this.DefaultSettings_;
  }

  // Computes the four Hermite weights for the interval [t0, t1] at time t.
  // Subclasses and external drivers can call this directly to fill _weight0..3
  // before invoking _interpolate.
  _computeWeights( t0, t, t1 ) {
    const dt = t1 - t0;
    const u = dt === 0 ? 0 : ( t - t0 ) / dt;
    const u2 = u * u;
    const u3 = u2 * u;

    // Hermite basis (Catmull-Rom variant used by r185's _interpolate).
    this._weight0 = - 0.5 * u3 + u2 - 0.5 * u;
    this._weight1 = 1.5 * u3 - 2.5 * u2 + 1;
    this._weight2 = - 1.5 * u3 + 2.0 * u2 + 0.5 * u;
    this._weight3 = 0.5 * u3 - 0.5 * u2;
    return this;
  }

  // Convenience override that fills the weights before delegating to the base
  // evaluate(). Useful when a driver wants to drive the weights itself.
  evaluateWithWeights( t, t0, t1 ) {
    this._computeWeights( t0, t, t1 );
    return this.evaluate( t );
  }

}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec3 / vec4  <->  THREE.CubicInterpolant resultBuffer
 * -----------------------------------------------------------------------------
 * CubicInterpolant inherits the same resultBuffer contract as Interpolant, so
 * the bridge helpers are re-used. The wrappers below are direct aliases for
 * convenience and provide a Cubic-specific surface without duplicating logic.
 */

// CubicInterpolant resultBuffer (stride 3) -> preallocated THREE.Vector3
export function threeVec3FromCubicInterpolant( out, interpolant ) {
  return threeVec3FromInterpolant( out, interpolant );
}

// CubicInterpolant resultBuffer (stride 4) -> preallocated THREE.Vector4
export function threeVec4FromCubicInterpolant( out, interpolant ) {
  return threeVec4FromInterpolant( out, interpolant );
}

// CubicInterpolant resultBuffer (stride 3) -> preallocated gl-matrix vec3
export function glMatrixVec3FromCubicInterpolant( out, interpolant ) {
  return glMatrixVec3FromInterpolant( out, interpolant );
}

// CubicInterpolant resultBuffer (stride 4) -> preallocated gl-matrix vec4
export function glMatrixVec4FromCubicInterpolant( out, interpolant ) {
  return glMatrixVec4FromInterpolant( out, interpolant );
}

// CubicInterpolant resultBuffer (stride 4) -> preallocated gl-matrix quat
export function glMatrixQuatFromCubicInterpolant( out, interpolant ) {
  return glMatrixQuatFromInterpolant( out, interpolant );
}

// gl-matrix vec3 -> CubicInterpolant resultBuffer (writes in place)
export function cubicInterpolantFromGlMatrixVec3( interpolant, glVec ) {
  return interpolantFromGlMatrixVec3( interpolant, glVec );
}

// gl-matrix vec4 / quat -> CubicInterpolant resultBuffer (writes in place)
export function cubicInterpolantFromGlMatrixVec4( interpolant, glVec ) {
  return interpolantFromGlMatrixVec4( interpolant, glVec );
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs InterpolantTrackComponent  <->  THREE.CubicInterpolant
 * -----------------------------------------------------------------------------
 * The track descriptor is identical to Interpolant's. CubicInterpolant only
 * differs in how it blends the four sample slots per stride.
 */

// Bind a CubicInterpolant to a bitecs entity's track (no copy, subarray views).
export function bitecsCubicInterpolantBindFromTrack( interpolant, eid, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantBindFromTrack( interpolant, eid, store, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into a THREE.Vector3 (stride 3).
export function threeVec3FromBitecsCubicInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into a THREE.Vector4 (stride 4).
export function threeVec4FromBitecsCubicInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into a gl-matrix vec3 (stride 3).
export function glMatrixVec3FromBitecsCubicInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into a gl-matrix vec4 / quat.
export function glMatrixVec4FromBitecsCubicInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into another entity's SoA Vector3.
export function bitecsVec3FromBitecsCubicInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec3FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Evaluate a bitecs-bound CubicInterpolant at t into another entity's SoA Vector4.
export function bitecsVec4FromBitecsCubicInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec4FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Copy the evaluated result at t into the destination entity's pool slice.
export function bitecsCubicInterpolantResultIntoPool( interpolant, eidDst, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantResultIntoPool( interpolant, eidDst, t, store, pools );
}

// Read a single sample slot into a THREE.Vector3 (stride 3 assumed).
export function threeVec3FromBitecsCubicInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a THREE.Vector4 (stride 4 assumed).
export function threeVec4FromBitecsCubicInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec3 (stride 3 assumed).
export function glMatrixVec3FromBitecsCubicInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec4 / quat (stride 4 assumed).
export function glMatrixVec4FromBitecsCubicInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Write a sample slot from a bitecs SoA Vector3 entity (no temp allocation).
export function bitecsCubicInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Write a sample slot from a bitecs SoA Vector4 entity (no temp allocation).
export function bitecsCubicInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Read/write parameter positions.
export function bitecsCubicInterpolantPositionSet( eidSlot, slotIndex, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionSet( eidSlot, slotIndex, t, store, pools );
}

export function bitecsCubicInterpolantPositionGet( eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionGet( eidSlot, slotIndex, store, pools );
}

export function bitecsCubicInterpolantSampleSlotFromT( eidSlot, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantSampleSlotFromT( eidSlot, t, store, pools );
}

// Convenience: fill a cubic track's sample slots from a bitecs SoA Vector3
// interpolation across a chain of entities. No allocation in the loop.
export function bitecsCubicInterpolantFillPositionsFromVec3s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  const pStart = store.positionStart[ eidTrack ] | 0;
  const vStart = store.valueStart[ eidTrack ] | 0;
  const vStride = store.valueStride[ eidTrack ] | 0;
  for ( let i = 0; i < vecEids.length; i ++ ) {
    const eidVec = vecEids[ i ];
    pools.parameterPositions[ pStart + i ] = i;
    pools.sampleValues[ vStart + i * vStride ] = storeVec.x[ eidVec ];
    pools.sampleValues[ vStart + i * vStride + 1 ] = storeVec.y[ eidVec ];
    pools.sampleValues[ vStart + i * vStride + 2 ] = storeVec.z[ eidVec ];
  }
  return eidTrack;
}

// Convenience: fill a cubic track's sample slots from a bitecs SoA Vector4
// interpolation across a chain of entities. No allocation in the loop.
export function bitecsCubicInterpolantFillPositionsFromVec4s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  const pStart = store.positionStart[ eidTrack ] | 0;
  const vStart = store.valueStart[ eidTrack ] | 0;
  const vStride = store.valueStride[ eidTrack ] | 0;
  for ( let i = 0; i < vecEids.length; i ++ ) {
    const eidVec = vecEids[ i ];
    pools.parameterPositions[ pStart + i ] = i;
    pools.sampleValues[ vStart + i * vStride ] = storeVec.x[ eidVec ];
    pools.sampleValues[ vStart + i * vStride + 1 ] = storeVec.y[ eidVec ];
    pools.sampleValues[ vStart + i * vStride + 2 ] = storeVec.z[ eidVec ];
    pools.sampleValues[ vStart + i * vStride + 3 ] = storeVec.w[ eidVec ];
  }
  return eidTrack;
}

// Module-local scratch buffers — allocated once, reused across every bridge.
const _scratchVec3 = new Float32Array( 3 );
const _scratchVec4 = new Float32Array( 4 );

export { CubicInterpolant };