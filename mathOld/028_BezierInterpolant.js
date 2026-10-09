// file number : 028
// full path name : src/math/028_BezierInterpolant.js
// description : BezierInterpolant class (THREE.BezierInterpolant) extending Interpolant with cubic Bezier curves using 2D control points (time, value) for COLLADA/Maya-style animation, plus full zero-allocation bridge helpers that bind a bitecs 0.4.0 InterpolantTrackComponent to a BezierInterpolant using shared pools (with optional tangents pools), and that read/write gl-matrix vec3/vec4 results without allocating. Depends on MathUtils.js (file 001) and Interpolant.js (file 022).
// best for  :  COLLADA/Maya keyframe animation, animation editors with tangent handles, custom ease curves, cinematic camera paths, and any hot loop that must evaluate Bezier interpolants over SoA data without allocating.
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
 * BITECS 0.4.0 BEZIER TRACK COMPONENT (SoA, archetype-friendly)
 * -----------------------------------------------------------------------------
 * A Bezier track is described by:
 *   - positionStart   : f32 offset into shared parameterPositions pool
 *   - positionCount   : u32 number of keys
 *   - valueStart      : f32 offset into shared sampleValues pool
 *   - valueStride     : u32 components per sample
 *   - inTangentStart  : f32 offset into shared inTangents pool
 *   - inTangentStride : u32 components per in-tangent (time, value)
 *   - outTangentStart : f32 offset into shared outTangents pool
 *   - outTangentStride: u32 components per out-tangent (time, value)
 *
 * The in/out tangents pools are also caller-owned Float32Arrays, exactly like
 * InterpolantPools. This keeps the ECS side fully SoA-friendly.
 */
export const BezierTrackComponent = defineComponent( {
  positionStart: Types.f32,
  positionCount: Types.u32,
  valueStart: Types.f32,
  valueStride: Types.u32,
  inTangentStart: Types.f32,
  inTangentStride: Types.u32,
  outTangentStart: Types.f32,
  outTangentStride: Types.u32
} );

/*
 * -----------------------------------------------------------------------------
 * SHARED TANGENT POOLS (caller-managed, zero-allocation)
 * -----------------------------------------------------------------------------
 * Two additional Float32Array pools for the in/out tangents. Each tangent is
 * stored as (time, value) pairs — stride 2 per key per component.
 */
export const BezierPools = {
  inTangents: new Float32Array( 0 ),
  outTangents: new Float32Array( 0 ),

  set( inT, outT ) {
    BezierPools.inTangents = inT;
    BezierPools.outTangents = outT;
  },

  ensureInTangents( size ) {
    if ( BezierPools.inTangents.length < size ) {
      const next = new Float32Array( size );
      next.set( BezierPools.inTangents );
      BezierPools.inTangents = next;
    }
    return BezierPools.inTangents;
  },

  ensureOutTangents( size ) {
    if ( BezierPools.outTangents.length < size ) {
      const next = new Float32Array( size );
      next.set( BezierPools.outTangents );
      BezierPools.outTangents = next;
    }
    return BezierPools.outTangents;
  }
};

/*
 * -----------------------------------------------------------------------------
 * THREE.BezierInterpolant (r185)
 * -----------------------------------------------------------------------------
 * Cubic Bezier interpolant with explicit in/out tangents as 2D control points
 * (time, value). Each keyframe has an in-tangent and an out-tangent, both of
 * which are 2D points relative to the keyframe.
 */
class BezierInterpolant extends Interpolant {

  constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {
    super( parameterPositions, sampleValues, sampleSize, resultBuffer );

    this.inTangents = null;
    this.outTangents = null;
    this.tangentStride = 2;

  }

  interpolate_( i1, t0, t, t1 ) {

    const result = this.resultBuffer;
    const values = this.sampleValues;
    const stride = this.valueSize;

    const offset1 = i1 * stride;
    const offset0 = offset1 - stride;

    const inT = this.inTangents;
    const outT = this.outTangents;
    const tStride = this.tangentStride;

    // If no tangents are bound, fall back to linear interpolation (matches
    // r185's robust behavior when tangents are missing).
    if ( inT === null || outT === null ) {
      const weight1 = ( t - t0 ) / ( t1 - t0 );
      const weight0 = 1 - weight1;
      for ( let i = 0; i !== stride; ++ i ) {
        result[ i ] = values[ offset0 + i ] * weight0 + values[ offset1 + i ] * weight1;
      }
      return result;
    }

    // Bezier control points for the interval [t0, t1]:
    //   p0 = (t0, values[offset0])
    //   p1 = (t0, values[offset0]) + outTangent[offset0]
    //   p2 = (t1, values[offset1]) + inTangent[offset1]
    //   p3 = (t1, values[offset1])
    // We solve for the parameter u along the Bezier curve such that the time
    // component matches t, then evaluate the value component at u.
    const tSpan = t1 - t0;

    for ( let i = 0; i !== stride; ++ i ) {

      const v0 = values[ offset0 + i ];
      const v1 = values[ offset1 + i ];

      const ot = offset0 * tStride + i * 2;
      const it = offset1 * tStride + i * 2;

      const p0t = t0;
      const p0v = v0;
      const p1t = t0 + outT[ ot ];
      const p1v = v0 + outT[ ot + 1 ];
      const p2t = t1 + inT[ it ];
      const p2v = v1 + inT[ it + 1 ];
      const p3t = t1;
      const p3v = v1;

      // Solve for u such that bezierTime(u) = t using Newton-Raphson.
      // bezierTime(u) = (1-u)^3 p0t + 3(1-u)^2 u p1t + 3(1-u)u^2 p2t + u^3 p3t
      // We use the closed-form cubic coefficients and iterate.
      const a = - p0t + 3 * p1t - 3 * p2t + p3t;
      const b = 3 * p0t - 6 * p1t + 3 * p2t;
      const c = - 3 * p0t + 3 * p1t;
      const d = p0t - t;

      let u = tSpan === 0 ? 0 : ( t - t0 ) / tSpan; // initial guess (linear)
      for ( let iter = 0; iter < 8; iter ++ ) {
        const u2 = u * u;
        const u3 = u2 * u;
        const f = a * u3 + b * u2 + c * u + d;
        const df = 3 * a * u2 + 2 * b * u + c;
        if ( Math.abs( df ) < 1e-9 ) break;
        const delta = f / df;
        u -= delta;
        if ( Math.abs( delta ) < 1e-7 ) break;
        if ( u < 0 ) u = 0;
        if ( u > 1 ) u = 1;
      }

      // Evaluate value component at u.
      const u2 = u * u;
      const u3 = u2 * u;
      const oneMinusU = 1 - u;
      const om2 = oneMinusU * oneMinusU;
      const om3 = om2 * oneMinusU;
      const bu0 = om3;
      const bu1 = 3 * om2 * u;
      const bu2 = 3 * oneMinusU * u2;
      const bu3 = u3;

      result[ i ] = bu0 * p0v + bu1 * p1v + bu2 * p2v + bu3 * p3v;

    }

    return result;

  }

}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec3 / vec4  <->  THREE.BezierInterpolant resultBuffer
 * -----------------------------------------------------------------------------
 * BezierInterpolant inherits the same resultBuffer contract as Interpolant.
 * The wrappers below are direct aliases for convenience.
 */

// BezierInterpolant resultBuffer (stride 3) -> preallocated THREE.Vector3
export function threeVec3FromBezierInterpolant( out, interpolant ) {
  return threeVec3FromInterpolant( out, interpolant );
}

// BezierInterpolant resultBuffer (stride 4) -> preallocated THREE.Vector4
export function threeVec4FromBezierInterpolant( out, interpolant ) {
  return threeVec4FromInterpolant( out, interpolant );
}

// BezierInterpolant resultBuffer (stride 3) -> preallocated gl-matrix vec3
export function glMatrixVec3FromBezierInterpolant( out, interpolant ) {
  return glMatrixVec3FromInterpolant( out, interpolant );
}

// BezierInterpolant resultBuffer (stride 4) -> preallocated gl-matrix vec4
export function glMatrixVec4FromBezierInterpolant( out, interpolant ) {
  return glMatrixVec4FromInterpolant( out, interpolant );
}

// BezierInterpolant resultBuffer (stride 4) -> preallocated gl-matrix quat
export function glMatrixQuatFromBezierInterpolant( out, interpolant ) {
  return glMatrixQuatFromInterpolant( out, interpolant );
}

// gl-matrix vec3 -> BezierInterpolant resultBuffer (writes in place)
export function bezierInterpolantFromGlMatrixVec3( interpolant, glVec ) {
  return interpolantFromGlMatrixVec3( interpolant, glVec );
}

// gl-matrix vec4 / quat -> BezierInterpolant resultBuffer (writes in place)
export function bezierInterpolantFromGlMatrixVec4( interpolant, glVec ) {
  return interpolantFromGlMatrixVec4( interpolant, glVec );
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs BezierTrackComponent  <->  THREE.BezierInterpolant
 * -----------------------------------------------------------------------------
 * The Bezier track descriptor binds not only parameterPositions and sampleValues
 * but also the in/out tangent subarrays from BezierPools.
 */

// Bind a BezierInterpolant to a bitecs entity's track (no copy, subarray views).
export function bitecsBezierInterpolantBindFromTrack( interpolant, eid, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  const pStart = store.positionStart[ eid ] | 0;
  const pCount = store.positionCount[ eid ] | 0;
  const vStart = store.valueStart[ eid ] | 0;
  const vStride = store.valueStride[ eid ] | 0;
  const vCount = pCount * vStride;

  interpolant.parameterPositions = pools.parameterPositions.subarray( pStart, pStart + pCount );
  interpolant.sampleValues = pools.sampleValues.subarray( vStart, vStart + vCount );
  interpolant.valueSize = vStride;

  const inStart = store.inTangentStart[ eid ] | 0;
  const inStride = store.inTangentStride[ eid ] | 0;
  const outStart = store.outTangentStart[ eid ] | 0;
  const outStride = store.outTangentStride[ eid ] | 0;

  interpolant.inTangents = bezPools.inTangents.subarray( inStart, inStart + pCount * inStride );
  interpolant.outTangents = bezPools.outTangents.subarray( outStart, outStart + pCount * outStride );
  interpolant.tangentStride = inStride;

  interpolant._cachedIndex = 0;
  return interpolant;
}

// Evaluate a bitecs-bound BezierInterpolant at t into a THREE.Vector3 (stride 3).
export function threeVec3FromBitecsBezierInterpolantEvaluate( out, interpolant, eid, t, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eid, store, pools, bezPools );
  const r = interpolant.evaluate( t );
  out.x = r[ 0 ]; out.y = r[ 1 ]; out.z = r[ 2 ];
  return out;
}

// Evaluate a bitecs-bound BezierInterpolant at t into a THREE.Vector4 (stride 4).
export function threeVec4FromBitecsBezierInterpolantEvaluate( out, interpolant, eid, t, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eid, store, pools, bezPools );
  const r = interpolant.evaluate( t );
  out.x = r[ 0 ]; out.y = r[ 1 ]; out.z = r[ 2 ]; out.w = r[ 3 ];
  return out;
}

// Evaluate a bitecs-bound BezierInterpolant at t into a gl-matrix vec3 (stride 3).
export function glMatrixVec3FromBitecsBezierInterpolantEvaluate( out, interpolant, eid, t, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eid, store, pools, bezPools );
  const r = interpolant.evaluate( t );
  out[ 0 ] = r[ 0 ]; out[ 1 ] = r[ 1 ]; out[ 2 ] = r[ 2 ];
  return out;
}

// Evaluate a bitecs-bound BezierInterpolant at t into a gl-matrix vec4 / quat.
export function glMatrixVec4FromBitecsBezierInterpolantEvaluate( out, interpolant, eid, t, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eid, store, pools, bezPools );
  const r = interpolant.evaluate( t );
  out[ 0 ] = r[ 0 ]; out[ 1 ] = r[ 1 ]; out[ 2 ] = r[ 2 ]; out[ 3 ] = r[ 3 ];
  return out;
}

// Evaluate a bitecs-bound BezierInterpolant at t into another entity's SoA Vector3.
export function bitecsVec3FromBitecsBezierInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = BezierTrackComponent, storeVec, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eidTrack, storeTrack, pools, bezPools );
  const r = interpolant.evaluate( t );
  storeVec.x[ eidOut ] = r[ 0 ];
  storeVec.y[ eidOut ] = r[ 1 ];
  storeVec.z[ eidOut ] = r[ 2 ];
  return eidOut;
}

// Evaluate a bitecs-bound BezierInterpolant at t into another entity's SoA Vector4.
export function bitecsVec4FromBitecsBezierInterpolantEvaluate( eidOut, interpolant, eidTrack, t, storeTrack = BezierTrackComponent, storeVec, pools = InterpolantPools, bezPools = BezierPools ) {
  bitecsBezierInterpolantBindFromTrack( interpolant, eidTrack, storeTrack, pools, bezPools );
  const r = interpolant.evaluate( t );
  storeVec.x[ eidOut ] = r[ 0 ];
  storeVec.y[ eidOut ] = r[ 1 ];
  storeVec.z[ eidOut ] = r[ 2 ];
  storeVec.w[ eidOut ] = r[ 3 ];
  return eidOut;
}

// Copy the evaluated result at t into the destination entity's pool slice.
export function bitecsBezierInterpolantResultIntoPool( interpolant, eidDst, t, store = BezierTrackComponent, pools = InterpolantPools, bezPools = BezierPools ) {
  const vStart = store.valueStart[ eidDst ] | 0;
  const vStride = store.valueStride[ eidDst ] | 0;
  const r = interpolant.evaluate( t );
  for ( let i = 0; i < vStride; i ++ ) {
    pools.sampleValues[ vStart + i ] = r[ i ];
  }
  return eidDst;
}

// Read a single sample slot into a THREE.Vector3 (stride 3 assumed).
export function threeVec3FromBitecsBezierInterpolantSample( out, eidSlot, slotIndex, store = BezierTrackComponent, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  out.x = pools.sampleValues[ o ];
  out.y = pools.sampleValues[ o + 1 ];
  out.z = pools.sampleValues[ o + 2 ];
  return out;
}

// Read a single sample slot into a THREE.Vector4 (stride 4 assumed).
export function threeVec4FromBitecsBezierInterpolantSample( out, eidSlot, slotIndex, store = BezierTrackComponent, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  out.x = pools.sampleValues[ o ];
  out.y = pools.sampleValues[ o + 1 ];
  out.z = pools.sampleValues[ o + 2 ];
  out.w = pools.sampleValues[ o + 3 ];
  return out;
}

// Read a single sample slot into a gl-matrix vec3 (stride 3 assumed).
export function glMatrixVec3FromBitecsBezierInterpolantSample( out, eidSlot, slotIndex, store = BezierTrackComponent, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  out[ 0 ] = pools.sampleValues[ o ];
  out[ 1 ] = pools.sampleValues[ o + 1 ];
  out[ 2 ] = pools.sampleValues[ o + 2 ];
  return out;
}

// Read a single sample slot into a gl-matrix vec4 / quat (stride 4 assumed).
export function glMatrixVec4FromBitecsBezierInterpolantSample( out, eidSlot, slotIndex, store = BezierTrackComponent, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  out[ 0 ] = pools.sampleValues[ o ];
  out[ 1 ] = pools.sampleValues[ o + 1 ];
  out[ 2 ] = pools.sampleValues[ o + 2 ];
  out[ 3 ] = pools.sampleValues[ o + 3 ];
  return out;
}

// Write a sample slot from a bitecs SoA Vector3 entity (no temp allocation).
export function bitecsBezierInterpolantSampleFromVec3( eidSlot, slotIndex, eidVec, store = BezierTrackComponent, storeVec, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  pools.sampleValues[ o ] = storeVec.x[ eidVec ];
  pools.sampleValues[ o + 1 ] = storeVec.y[ eidVec ];
  pools.sampleValues[ o + 2 ] = storeVec.z[ eidVec ];
  return eidSlot;
}

// Write a sample slot from a bitecs SoA Vector4 entity (no temp allocation).
export function bitecsBezierInterpolantSampleFromVec4( eidSlot, slotIndex, eidVec, store = BezierTrackComponent, storeVec, pools = InterpolantPools ) {
  const vStart = store.valueStart[ eidSlot ] | 0;
  const vStride = store.valueStride[ eidSlot ] | 0;
  const o = vStart + slotIndex * vStride;
  pools.sampleValues[ o ] = storeVec.x[ eidVec ];
  pools.sampleValues[ o + 1 ] = storeVec.y[ eidVec ];
  pools.sampleValues[ o + 2 ] = storeVec.z[ eidVec ];
  pools.sampleValues[ o + 3 ] = storeVec.w[ eidVec ];
  return eidSlot;
}

// Write an in-tangent slot (time, value) for a specific component.
export function bitecsBezierInterpolantInTangentSet( eidSlot, slotIndex, componentIndex, time, value, store = BezierTrackComponent, pools = BezierPools ) {
  const inStart = store.inTangentStart[ eidSlot ] | 0;
  const inStride = store.inTangentStride[ eidSlot ] | 0;
  const o = inStart + ( slotIndex * inStride + componentIndex ) * 2;
  pools.inTangents[ o ] = time;
  pools.inTangents[ o + 1 ] = value;
  return eidSlot;
}

// Write an out-tangent slot (time, value) for a specific component.
export function bitecsBezierInterpolantOutTangentSet( eidSlot, slotIndex, componentIndex, time, value, store = BezierTrackComponent, pools = BezierPools ) {
  const outStart = store.outTangentStart[ eidSlot ] | 0;
  const outStride = store.outTangentStride[ eidSlot ] | 0;
  const o = outStart + ( slotIndex * outStride + componentIndex ) * 2;
  pools.outTangents[ o ] = time;
  pools.outTangents[ o + 1 ] = value;
  return eidSlot;
}

// Read an in-tangent slot (time, value) into a caller-owned 2-element Float32Array.
export function bitecsBezierInterpolantInTangentGet( out2, eidSlot, slotIndex, componentIndex, store = BezierTrackComponent, pools = BezierPools ) {
  const inStart = store.inTangentStart[ eidSlot ] | 0;
  const inStride = store.inTangentStride[ eidSlot ] | 0;
  const o = inStart + ( slotIndex * inStride + componentIndex ) * 2;
  out2[ 0 ] = pools.inTangents[ o ];
  out2[ 1 ] = pools.inTangents[ o + 1 ];
  return out2;
}

// Read an out-tangent slot (time, value) into a caller-owned 2-element Float32Array.
export function bitecsBezierInterpolantOutTangentGet( out2, eidSlot, slotIndex, componentIndex, store = BezierTrackComponent, pools = BezierPools ) {
  const outStart = store.outTangentStart[ eidSlot ] | 0;
  const outStride = store.outTangentStride[ eidSlot ] | 0;
  const o = outStart + ( slotIndex * outStride + componentIndex ) * 2;
  out2[ 0 ] = pools.outTangents[ o ];
  out2[ 1 ] = pools.outTangents[ o + 1 ];
  return out2;
}

// Read/write parameter positions.
export function bitecsBezierInterpolantPositionSet( eidSlot, slotIndex, t, store = BezierTrackComponent, pools = InterpolantPools ) {
  const pStart = store.positionStart[ eidSlot ] | 0;
  pools.parameterPositions[ pStart + slotIndex ] = t;
  return eidSlot;
}

export function bitecsBezierInterpolantPositionGet( eidSlot, slotIndex, store = BezierTrackComponent, pools = InterpolantPools ) {
  const pStart = store.positionStart[ eidSlot ] | 0;
  return pools.parameterPositions[ pStart + slotIndex ];
}

export function bitecsBezierInterpolantSampleSlotFromT( eidSlot, t, store = BezierTrackComponent, pools = InterpolantPools ) {
  const pStart = store.positionStart[ eidSlot ] | 0;
  const pCount = store.positionCount[ eidSlot ] | 0;
  if ( pCount <= 1 ) return 0;
  const t0 = pools.parameterPositions[ pStart ];
  const t1 = pools.parameterPositions[ pStart + pCount - 1 ];
  if ( t1 === t0 ) return 0;
  const u = clamp( ( t - t0 ) / ( t1 - t0 ), 0, 1 );
  return u * ( pCount - 1 );
}

// Convenience: fill a Bezier track's sample slots from a bitecs SoA Vector3
// chain. No allocation in the loop.
export function bitecsBezierInterpolantFillPositionsFromVec3s( eidTrack, vecEids, store = BezierTrackComponent, storeVec, pools = InterpolantPools ) {
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

// Convenience: fill a Bezier track's sample slots from a bitecs SoA Vector4
// chain. No allocation in the loop.
export function bitecsBezierInterpolantFillPositionsFromVec4s( eidTrack, vecEids, store = BezierTrackComponent, storeVec, pools = InterpolantPools ) {
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

export { BezierInterpolant };