// file number : 025
// full path name : src/math/025_DiscreteInterpolant.js
// description : DiscreteInterpolant class (THREE.DiscreteInterpolant) extending Interpolant with nearest-preceding-sample evaluation (zero-order hold / step), plus full zero-allocation bridge helpers that bind a bitecs 0.4.0 InterpolantTrackComponent to a DiscreteInterpolant using shared pools, and that read/write gl-matrix vec3/vec4 results without allocating. Depends on MathUtils.js (file 001) and Interpolant.js (file 022).
// best for  :  Boolean/step keyframe animation, discrete state tracks, nearest-neighbour color snapping, retro/pixel-art animation, ECS-driven discrete state machines, and any hot loop that must evaluate a discrete track over SoA data without allocating.
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
 * THREE.DiscreteInterpolant (r185)
 * -----------------------------------------------------------------------------
 * Zero-order hold / nearest-preceding-sample interpolant. `interpolate_` is
 * overridden to return the sample at index i1 - 1 (the key at or immediately
 * before the parameter t). This matches r185 exactly.
 */
class DiscreteInterpolant extends Interpolant {

  constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {
    super( parameterPositions, sampleValues, sampleSize, resultBuffer );
  }

  interpolate_( i1 /*, t0, t, t1 */ ) {
    return this.copySampleValue_( i1 - 1 );
  }

}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec3 / vec4  <->  THREE.DiscreteInterpolant resultBuffer
 * -----------------------------------------------------------------------------
 * DiscreteInterpolant inherits the same resultBuffer contract as Interpolant,
 * so the bridge helpers are re-used. The wrappers below are direct aliases for
 * convenience and provide a Discrete-specific surface without duplicating logic.
 */

// DiscreteInterpolant resultBuffer (stride 3) -> preallocated THREE.Vector3
export function threeVec3FromDiscreteInterpolant( out, interpolant ) {
  return threeVec3FromInterpolant( out, interpolant );
}

// DiscreteInterpolant resultBuffer (stride 4) -> preallocated THREE.Vector4
export function threeVec4FromDiscreteInterpolant( out, interpolant ) {
  return threeVec4FromInterpolant( out, interpolant );
}

// DiscreteInterpolant resultBuffer (stride 3) -> preallocated gl-matrix vec3
export function glMatrixVec3FromDiscreteInterpolant( out, interpolant ) {
  return glMatrixVec3FromInterpolant( out, interpolant );
}

// DiscreteInterpolant resultBuffer (stride 4) -> preallocated gl-matrix vec4
export function glMatrixVec4FromDiscreteInterpolant( out, interpolant ) {
  return glMatrixVec4FromInterpolant( out, interpolant );
}

// DiscreteInterpolant resultBuffer (stride 4) -> preallocated gl-matrix quat
export function glMatrixQuatFromDiscreteInterpolant( out, interpolant ) {
  return glMatrixQuatFromInterpolant( out, interpolant );
}

// gl-matrix vec3 -> DiscreteInterpolant resultBuffer (writes in place)
export function discreteInterpolantFromGlMatrixVec3( interpolant, glVec ) {
  return interpolantFromGlMatrixVec3( interpolant, glVec );
}

// gl-matrix vec4 / quat -> DiscreteInterpolant resultBuffer (writes in place)
export function discreteInterpolantFromGlMatrixVec4( interpolant, glVec ) {
  return interpolantFromGlMatrixVec4( interpolant, glVec );
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs InterpolantTrackComponent  <->  THREE.DiscreteInterpolant
 * -----------------------------------------------------------------------------
 * The track descriptor is identical to Interpolant's. DiscreteInterpolant only
 * differs in that it returns the preceding sample slot instead of blending.
 */

// Bind a DiscreteInterpolant to a bitecs entity's track (no copy, subarray views).
export function bitecsDiscreteInterpolantBindFromTrack( interpolant, eid, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantBindFromTrack( interpolant, eid, store, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into a THREE.Vector3 (stride 3).
export function threeVec3FromBitecsDiscreteInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into a THREE.Vector4 (stride 4).
export function threeVec4FromBitecsDiscreteInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into a gl-matrix vec3 (stride 3).
export function glMatrixVec3FromBitecsDiscreteInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into a gl-matrix vec4 / quat.
export function glMatrixVec4FromBitecsDiscreteInterpolantEvaluate( out, interpolant, eid, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantEvaluate( out, interpolant, eid, t, store, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into another entity's SoA Vector3.
export function bitecsVec3FromBitecsDiscreteInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec3FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Evaluate a bitecs-bound DiscreteInterpolant at t into another entity's SoA Vector4.
export function bitecsVec4FromBitecsDiscreteInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsVec4FromBitecsInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack, storeVec, pools );
}

// Copy the evaluated result at t into the destination entity's pool slice.
export function bitecsDiscreteInterpolantResultIntoPool( interpolant, eidDst, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantResultIntoPool( interpolant, eidDst, t, store, pools );
}

// Read a single sample slot into a THREE.Vector3 (stride 3 assumed).
export function threeVec3FromBitecsDiscreteInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a THREE.Vector4 (stride 4 assumed).
export function threeVec4FromBitecsDiscreteInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return threeVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec3 (stride 3 assumed).
export function glMatrixVec3FromBitecsDiscreteInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec3FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Read a single sample slot into a gl-matrix vec4 / quat (stride 4 assumed).
export function glMatrixVec4FromBitecsDiscreteInterpolantSample( out, eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return glMatrixVec4FromBitecsInterpolantSample( out, eidSlot, slotIndex, store, pools );
}

// Write a sample slot from a bitecs SoA Vector3 entity (no temp allocation).
export function bitecsDiscreteInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Write a sample slot from a bitecs SoA Vector4 entity (no temp allocation).
export function bitecsDiscreteInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
  return bitecsInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store, storeVec, pools );
}

// Read/write parameter positions.
export function bitecsDiscreteInterpolantPositionSet( eidSlot, slotIndex, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionSet( eidSlot, slotIndex, t, store, pools );
}

export function bitecsDiscreteInterpolantPositionGet( eidSlot, slotIndex, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantPositionGet( eidSlot, slotIndex, store, pools );
}

export function bitecsDiscreteInterpolantSampleSlotFromT( eidSlot, t, store = InterpolantTrackComponent, pools = InterpolantPools ) {
  return bitecsInterpolantSampleSlotFromT( eidSlot, t, store, pools );
}

// Convenience: fill a discrete track's sample slots from a bitecs SoA Vector3
// chain. No allocation in the loop.
export function bitecsDiscreteInterpolantFillPositionsFromVec3s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
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

// Convenience: fill a discrete track's sample slots from a bitecs SoA Vector4
// chain. No allocation in the loop.
export function bitecsDiscreteInterpolantFillPositionsFromVec4s( eidTrack, vecEids, store = InterpolantTrackComponent, storeVec, pools = InterpolantPools ) {
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

export { DiscreteInterpolant };