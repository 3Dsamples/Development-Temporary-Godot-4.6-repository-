// file number : 024
// full path name : src/math/024_LinearInterpolant.js
// description : LinearInterpolant class (THREE.LinearInterpolant) extending Interpolant with simple two-key linear blending, plus full zero-allocation bridge helpers that bind a bitecs 0.4.0 InterpolantTrackComponent to a LinearInterpolant using shared pools, and that read/write gl-matrix vec3/vec4 results without allocating. Depends on MathUtils.js (file 001) and Interpolant.js (file 022).
// best for  :  Keyframe animation between adjacent samples, color gradient sampling, path following, ECS-driven linear tracks, physics interpolation, and any hot loop that must evaluate a linear segment over SoA data without allocating.
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
 * THREE.LinearInterpolant (r185)
 * -----------------------------------------------------------------------------
 * Simple two-key linear blend. `interpolate_(i1, t0, t, t1)` uses
 * `weight1 = (t - t0) / (t1 - t0)` and `weight0 = 1 - weight1`.
 */
class LinearInterpolant extends Interpolant {

  constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {
    super( parameterPositions, sampleValues, sampleSize, resultBuffer );
  }

  interpolate_( i1, t0, t, t1 ) {

    const result = this.resultBuffer;
    const values = this.sampleValues;
    const stride = this.valueSize;

    const offset1 = i1 * stride;
    const offset0 = offset1 - stride;

    const weight1 = ( t - t0 ) / ( t1 - t0 );
    const weight0 = 1 - weight1;

    for ( let i = 0; i !== stride; ++ i ) {
      result[ i ] = values[ offset0 + i ] * weight0 + values[ offset1 + i ] * weight1;
    }

    return result;

  }

}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec3 / vec4  <->  THREE.LinearInterpolant resultBuffer
 * -----------------------------------------------------------------------------
 * LinearInterpolant inherits the same resultBuffer contract as Interpolant,
 * so the bridge helpers are re-used. The wrappers below are direct aliases.
 */

// LinearInterpolant resultBuffer (stride 3) -> preallocated THREE.Vector3
export function threeVec3FromLinearInterpolant( out, interpolant ) {
  return threeVec3FromInterpolant( out, interpolant );
}

// LinearInterpolant resultBuffer (stride 4) -> preallocated THREE.Vector4
export function threeVec4FromLinearInterpolant( out, interpolant ) {
  return threeVec4FromInterpolant( out, interpolant );
}

// LinearInterpolant resultBuffer (stride 3) -> preallocated gl-matrix vec3
export function glMatrixVec3FromLinearInterpolant( out, interpolant ) {
  return glMatrixVec3FromInterpolant( out, interpolant );
}

// LinearInterpolant resultBuffer (stride 4) -> preallocated gl-matrix vec4
export function glMatrixVec4FromLinearInterpolant( out, interpolant ) {
  return glMatrixVec4FromInterpolant( out, interpolant );
}

// LinearInterpolant resultBuffer (stride 4) -> preallocated gl-matrix quat
export function glMatrixQuatFromLinearInterpolant( out, interpolant ) {
  return glMatrixQuatFromInterpolant( out, interpolant );
}

// gl-matrix vec3 -> LinearInterpolant resultBuffer (writes in place)
export function linearInterpolantFromGlMatrixVec3( interpolant, glVec ) {
  return interpolantFromGlMatrixVec3( interpolant, glVec );
}

// gl-matrix vec4 / quat -> LinearInterpolant resultBuffer (writes in place)
export function linearInterpolantFromGlMatrixVec4( interpolant, glVec ) {
  return interpolantFromGlMatrixVec4( interpolant, glVec );
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs InterpolantTrackComponent  <->  THREE.LinearInterpolant
 * -----------------------------------------------------------------------------
 * The track descriptor is identical to Interpolant's. LinearInterpolant only
 * differs in how it blends the two adjacent sample slots per stride.
 */

// Bind a LinearInterpolant to a bitecs entity's track (no copy, subarray views).
export function bitecsLinearInterpolantBindFromTrack( interpolant, eid, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantBindFromTrack( interpolant, eid, store, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into a THREE.Vector3 (stride 3).
export function threeVec3FromBitecsLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into a THREE.Vector4 (stride 4).
export function threeVec4FromBitecsLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into a gl-matrix vec3 (stride 3).
export function glMatrixVec3FromBitecsLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into a gl-matrix vec4 / quat.
export function glMatrixVec4FromBitecsLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into another entity's SoA Vector3.
export function bitecsVec3FromBitecsLinearInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec3FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Evaluate a bitecs-bound LinearInterpolant at t into another entity's SoA Vector4.
export function bitecsVec4FromBitecsLinearInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec4FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Copy the evaluated result at t into the destination entity's pool slice.
export function bitecsLinearInterpolantResultIntoPool( interpolant, eidDst, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantResultIntoPool( interpolant, eidDst, t, store, pools );
}

// Read a single sample slot into a THREE.Vector3 (stride 3 assumed).
export function threeVec3FromBitecsLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a THREE.Vector4 (stride 4 assumed).
export function threeVec4FromBitecsLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec3 (stride 3 assumed).
export function glMatrixVec3FromBitecsLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec4 / quat (stride 4 assumed).
export function glMatrixVec4FromBitecsLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Write a sample slot from a bitecs SoA Vector3 entity (no temp allocation).
export function bitecsLinearInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Write a sample slot from a bitecs SoA Vector4 entity (no temp allocation).
export function bitecsLinearInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Read/write parameter positions.
export function bitecsLinearInterpolantPositionSet( eidSlot, slotIndex, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionSet( eidSlot, slotIndex, t, store, pools );
}

export function bitecsLinearInterpolantPositionGet( eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionGet( eidSlot, slotIndex, store, pools );
}

export function bitecsLinearInterpolantSampleSlotFromT( eidSlot, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantSampleSlotFromT( eidSlot, t, store, pools );
}

// Convenience: fill a linear track's sample slots from a bitecs SoA Vector3
// chain. No allocation in the loop.
export function bitecsLinearInterpolantFillPositionsFromVec3s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
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

// Convenience: fill a linear track's sample slots from a bitecs SoA Vector4
// chain. No allocation in the loop.
export function bitecsLinearInterpolantFillPositionsFromVec4s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
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

export { LinearInterpolant };