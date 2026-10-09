// file number : 026
// full path name : src/math/026_QuaternionLinearInterpolant.js
// description : QuaternionLinearInterpolant class (THREE.QuaternionLinearInterpolant) extending Interpolant with spherical linear interpolation (slerp) between adjacent quaternion samples, plus full zero-allocation bridge helpers that bind a bitecs 0.4.0 InterpolantTrackComponent to a QuaternionLinearInterpolant using shared pools, and that read/write gl-matrix vec4/quat results without allocating. Depends on MathUtils.js (file 001), Interpolant.js (file 022), and Quaternion.js (file 004) for Quaternion.slerpFlat.
// best for  :  Rotational keyframe animation, camera orientation tracks, skeletal bone animation, quaternion-based easing, ECS-driven rotation tracks, and any hot loop that must evaluate slerp over SoA data without allocating.
// license : GPL3

import { clamp } from './MathUtils.js';
import { Quaternion } from './004_Quaternion.js';
import {
  Interpolant,
  InterpolantTrackComponent,
  InterpolantPools,
  threeVec4FromInterpolant,
  glMatrixVec4FromInterpolant,
  glMatrixQuatFromInterpolant,
  interpolantFromGlMatrixVec4,
  bitecsInterpolantBindFromTrack,
  threeVec4FromBitecsInterpolantEvaluate,
  glMatrixVec4FromBitecsInterpolantEvaluate,
  bitecsVec4FromBitecsInterpolantEvaluate,
  bitecsInterpolantResultIntoPool,
  threeVec4FromBitecsInterpolantSample,
  glMatrixVec4FromBitecsInterpolantSample,
  bitecsInterpolantSampleFromVec4,
  bitecsInterpolantPositionSet,
  bitecsInterpolantPositionGet,
  bitecsInterpolantSampleSlotFromT
} from './022_Interpolant.js';

import glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.min.js';
import { defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

const { vec4: glVec4, quat: glQuat } = glMatrix;

/*
 * -----------------------------------------------------------------------------
 * BITECS 0.4.0 QUATERNION TRACK COMPONENT (SoA, archetype-friendly)
 * -----------------------------------------------------------------------------
 * Quaternion tracks reuse the same InterpolantTrackComponent descriptor, but
 * the pool layout expects stride = 4 (x, y, z, w). This component is provided
 * as a typed alias so systems can declare intent without duplicating fields.
 */
export const QuaternionTrackComponent = InterpolantTrackComponent;

/*
 * -----------------------------------------------------------------------------
 * THREE.QuaternionLinearInterpolant (r185)
 * -----------------------------------------------------------------------------
 * Slerp-based quaternion interpolant. `interpolate_` calls Quaternion.slerpFlat
 * with the two adjacent samples and the computed blend factor.
 */
class QuaternionLinearInterpolant extends Interpolant {

  constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {
    super( parameterPositions, sampleValues, sampleSize, resultBuffer );
  }

  interpolate_( i1, t0, t, t1 ) {

    const result = this.resultBuffer;
    const values = this.sampleValues;
    const stride = this.valueSize;

    const alpha = ( t - t0 ) / ( t1 - t0 );

    let offset = i1 * stride;

    for ( let end = offset + stride; offset !== end; offset += 4 ) {
      Quaternion.slerpFlat( result, 0, values, offset - stride, values, offset, alpha );
    }

    return result;

  }

}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec4 / quat  <->  THREE.QuaternionLinearInterpolant
 * -----------------------------------------------------------------------------
 * QuaternionLinearInterpolant inherits the same resultBuffer contract as
 * Interpolant. The wrappers below are direct aliases for convenience and
 * provide a quaternion-specific surface without duplicating logic.
 */

// QuaternionLinearInterpolant resultBuffer -> preallocated THREE.Vector4
export function threeVec4FromQuaternionLinearInterpolant( out, interpolant ) {
  return threeVec4FromInterpolant( out, interpolant );
}

// QuaternionLinearInterpolant resultBuffer -> preallocated gl-matrix vec4
export function glMatrixVec4FromQuaternionLinearInterpolant( out, interpolant ) {
  return glMatrixVec4FromInterpolant( out, interpolant );
}

// QuaternionLinearInterpolant resultBuffer -> preallocated gl-matrix quat
export function glMatrixQuatFromQuaternionLinearInterpolant( out, interpolant ) {
  return glMatrixQuatFromInterpolant( out, interpolant );
}

// gl-matrix vec4 / quat -> QuaternionLinearInterpolant resultBuffer (writes in place)
export function quaternionLinearInterpolantFromGlMatrixVec4( interpolant, glVec ) {
  return interpolantFromGlMatrixVec4( interpolant, glVec );
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs InterpolantTrackComponent  <->  THREE.QuaternionLinearInterpolant
 * -----------------------------------------------------------------------------
 * The track descriptor is identical to Interpolant's. QuaternionLinearInterpolant
 * only differs in that it slerps adjacent quaternion samples instead of
 * blending floats.
 */

// Bind a QuaternionLinearInterpolant to a bitecs entity's track (no copy).
export function bitecsQuaternionLinearInterpolantBindFromTrack( interpolant, eid, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantBindFromTrack( interpolant, eid, store, pools );
}

// Evaluate a bitecs-bound QuaternionLinearInterpolant at t into a THREE.Vector4.
export function threeVec4FromBitecsQuaternionLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound QuaternionLinearInterpolant at t into a gl-matrix vec4 / quat.
export function glMatrixVec4FromBitecsQuaternionLinearInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound QuaternionLinearInterpolant at t into another entity's SoA Vector4.
export function bitecsVec4FromBitecsQuaternionLinearInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec4FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Copy the evaluated result at t into the destination entity's pool slice.
export function bitecsQuaternionLinearInterpolantResultIntoPool( interpolant, eidDst, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantResultIntoPool( interpolant, eidDst, t, store, pools );
}

// Read a single sample slot into a THREE.Vector4 (stride 4 assumed).
export function threeVec4FromBitecsQuaternionLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec4 / quat (stride 4 assumed).
export function glMatrixVec4FromBitecsQuaternionLinearInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Write a sample slot from a bitecs SoA Vector4 entity (no temp allocation).
export function bitecsQuaternionLinearInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Read/write parameter positions.
export function bitecsQuaternionLinearInterpolantPositionSet( eidSlot, slotIndex, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionSet( eidSlot, slotIndex, t, store, pools );
}

export function bitecsQuaternionLinearInterpolantPositionGet( eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionGet( eidSlot, slotIndex, store, pools );
}

export function bitecsQuaternionLinearInterpolantSampleSlotFromT( eidSlot, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantSampleSlotFromT( eidSlot, t, store, pools );
}

// Convenience: fill a quaternion track's sample slots from a bitecs SoA
// Vector4 chain. No allocation in the loop.
export function bitecsQuaternionLinearInterpolantFillPositionsFromVec4s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
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
const _scratchVec4 = new Float32Array( 4 );

export { QuaternionLinearInterpolant };