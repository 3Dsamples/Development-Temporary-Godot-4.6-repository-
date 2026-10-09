// file number : 029
// full path name : src/math/029_OBB.js
// description : Oriented Bounding Box (OBB) class with center, half-extents, and orientation quaternion, plus full zero-allocation bridge functions to/from gl-matrix (vec3 center, vec3 half-extents, quat orientation, or packed 10-element Float32Array) and bitecs 0.4.0 SoA components (center x/y/z, half-extent x/y/z, orientation x/y/z/w Float32Arrays indexed by entity id). Includes SAT-based intersection tests with AABB, OBB, Sphere, and Point, plus real-time multi-scale helpers. Depends on MathUtils.js (file 001), Vector3.js (file 003), Quaternion.js (file 004), Matrix3.js (file 006), Matrix4.js (file 007), Box3.js (file 012), Sphere.js (file 011), and Plane.js (file 010).
// best for  :  Tight-fitting collision bounds for rotated objects, vehicle/character collision, oriented spatial partitioning, robotic arm collision, and any ECS system that stores oriented boxes as SoA center/extents/orientation and must feed physics or culling without allocating per frame.
// license : GPL3

import { clamp } from './MathUtils.js';
import { Vector3 } from './003_Vector3.js';
import { Quaternion } from './004_Quaternion.js';
import { Matrix3 } from './006_Matrix3.js';
import { Matrix4 } from './007_Matrix4.js';
import { Box3 } from './012_Box3.js';
import { Sphere } from './011_Sphere.js';
import { Plane } from './010_Plane.js';

import glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.min.js';
import { defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

const { vec3: glVec3, quat: glQuat, mat3: glMat3 } = glMatrix;

/*
 * -----------------------------------------------------------------------------
 * MULTI-SCALE UNIT TABLE (meters as the base unit)
 * -----------------------------------------------------------------------------
 */
export const SCALE_UNITS = Object.freeze( {
  NANOMETER: 1e-9,
  MICROMETER: 1e-6,
  MILLIMETER: 1e-3,
  CENTIMETER: 1e-2,
  METER: 1,
  KILOMETER: 1e3,
  EARTH_RADIUS: 6371000,
  ASTRONOMICAL_UNIT: 1.495978707e11,
  LIGHT_YEAR: 9.4607304725808e15,
  PARSEC: 3.0856775814913673e16,
  SOLAR_RADIUS: 6.957e8,
  GALACTIC_RADIUS: 5e20
} );

/*
 * -----------------------------------------------------------------------------
 * BITECS 0.4.0 OBB COMPONENT (SoA, archetype-friendly, cache-friendly)
 * -----------------------------------------------------------------------------
 * OBB is stored as:
 *   - center     : cx, cy, cz (Float32Arrays)
 *   - halfExtents: hx, hy, hz (Float32Arrays)
 *   - orientation: qx, qy, qz, qw (Float32Arrays)
 * All indexed by entity id. Systems read/write store.cx[eid] ... store.qw[eid]
 * directly — no temporary OBB object, no per-entity allocation, no GC churn.
 */
export const OBBComponent = defineComponent( {
  cx: Types.f32, cy: Types.f32, cz: Types.f32,
  hx: Types.f32, hy: Types.f32, hz: Types.f32,
  qx: Types.f32, qy: Types.f32, qz: Types.f32, qw: Types.f32
} );

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix (vec3 center, vec3 half-extents, quat orientation)
 *          <->  OBB
 * -----------------------------------------------------------------------------
 * gl-matrix has no dedicated OBB type. An OBB is represented as three
 * components: vec3 center, vec3 half-extents, quat orientation, or as a packed
 * 10-element Float32Array [cx,cy,cz, hx,hy,hz, qx,qy,qz,qw]. We mirror both
 * contracts. All bridges write into a caller-owned `out` (preallocated OBB,
 * THREE.Vector3, or gl-matrix Float32Array) and never allocate per call.
 */

// gl-matrix (center, halfExtents, orientation) -> preallocated OBB
export function obbFromGlMatrix( out, glCenter, glHalfExtents, glOrientation ) {
  out.center.set( glCenter[ 0 ], glCenter[ 1 ], glCenter[ 2 ] );
  out.halfExtents.set( glHalfExtents[ 0 ], glHalfExtents[ 1 ], glHalfExtents[ 2 ] );
  out.orientation.set( glOrientation[ 0 ], glOrientation[ 1 ], glOrientation[ 2 ], glOrientation[ 3 ] );
  return out;
}

// gl-matrix packed 10-element Float32Array -> preallocated OBB
export function obbFromGlMatrixPacked( out, glPacked ) {
  out.center.set( glPacked[ 0 ], glPacked[ 1 ], glPacked[ 2 ] );
  out.halfExtents.set( glPacked[ 3 ], glPacked[ 4 ], glPacked[ 5 ] );
  out.orientation.set( glPacked[ 6 ], glPacked[ 7 ], glPacked[ 8 ], glPacked[ 9 ] );
  return out;
}

// OBB -> three preallocated gl-matrix buffers (center, halfExtents, orientation)
export function glMatrixOBBFromObb( outCenter, outHalfExtents, outOrientation, obb ) {
  outCenter[ 0 ] = obb.center.x;
  outCenter[ 1 ] = obb.center.y;
  outCenter[ 2 ] = obb.center.z;
  outHalfExtents[ 0 ] = obb.halfExtents.x;
  outHalfExtents[ 1 ] = obb.halfExtents.y;
  outHalfExtents[ 2 ] = obb.halfExtents.z;
  outOrientation[ 0 ] = obb.orientation.x;
  outOrientation[ 1 ] = obb.orientation.y;
  outOrientation[ 2 ] = obb.orientation.z;
  outOrientation[ 3 ] = obb.orientation.w;
  return obb;
}

// OBB -> preallocated packed 10-element Float32Array
export function glMatrixOBBPackedFromObb( outPacked, obb ) {
  outPacked[ 0 ] = obb.center.x;
  outPacked[ 1 ] = obb.center.y;
  outPacked[ 2 ] = obb.center.z;
  outPacked[ 3 ] = obb.halfExtents.x;
  outPacked[ 4 ] = obb.halfExtents.y;
  outPacked[ 5 ] = obb.halfExtents.z;
  outPacked[ 6 ] = obb.orientation.x;
  outPacked[ 7 ] = obb.orientation.y;
  outPacked[ 8 ] = obb.orientation.z;
  outPacked[ 9 ] = obb.orientation.w;
  return outPacked;
}

// gl-matrix (center, halfExtents, orientation) -> write directly into bitecs entity
export function bitecsOBBFromGlMatrix( eid, glCenter, glHalfExtents, glOrientation, store = OBBComponent ) {
  store.cx[ eid ] = glCenter[ 0 ];
  store.cy[ eid ] = glCenter[ 1 ];
  store.cz[ eid ] = glCenter[ 2 ];
  store.hx[ eid ] = glHalfExtents[ 0 ];
  store.hy[ eid ] = glHalfExtents[ 1 ];
  store.hz[ eid ] = glHalfExtents[ 2 ];
  store.qx[ eid ] = glOrientation[ 0 ];
  store.qy[ eid ] = glOrientation[ 1 ];
  store.qz[ eid ] = glOrientation[ 2 ];
  store.qw[ eid ] = glOrientation[ 3 ];
  return eid;
}

// gl-matrix packed 10-element Float32Array -> write directly into bitecs entity
export function bitecsOBBFromGlMatrixPacked( eid, glPacked, store = OBBComponent ) {
  store.cx[ eid ] = glPacked[ 0 ];
  store.cy[ eid ] = glPacked[ 1 ];
  store.cz[ eid ] = glPacked[ 2 ];
  store.hx[ eid ] = glPacked[ 3 ];
  store.hy[ eid ] = glPacked[ 4 ];
  store.hz[ eid ] = glPacked[ 5 ];
  store.qx[ eid ] = glPacked[ 6 ];
  store.qy[ eid ] = glPacked[ 7 ];
  store.qz[ eid ] = glPacked[ 8 ];
  store.qw[ eid ] = glPacked[ 9 ];
  return eid;
}

// bitecs entity SoA component -> preallocated gl-matrix (center, halfExtents, orientation)
export function glMatrixOBBFromBitecs( outCenter, outHalfExtents, outOrientation, eid, store = OBBComponent ) {
  outCenter[ 0 ] = store.cx[ eid ];
  outCenter[ 1 ] = store.cy[ eid ];
  outCenter[ 2 ] = store.cz[ eid ];
  outHalfExtents[ 0 ] = store.hx[ eid ];
  outHalfExtents[ 1 ] = store.hy[ eid ];
  outHalfExtents[ 2 ] = store.hz[ eid ];
  outOrientation[ 0 ] = store.qx[ eid ];
  outOrientation[ 1 ] = store.qy[ eid ];
  outOrientation[ 2 ] = store.qz[ eid ];
  outOrientation[ 3 ] = store.qw[ eid ];
  return eid;
}

// bitecs entity SoA component -> preallocated packed 10-element Float32Array
export function glMatrixOBBPackedFromBitecs( outPacked, eid, store = OBBComponent ) {
  outPacked[ 0 ] = store.cx[ eid ];
  outPacked[ 1 ] = store.cy[ eid ];
  outPacked[ 2 ] = store.cz[ eid ];
  outPacked[ 3 ] = store.hx[ eid ];
  outPacked[ 4 ] = store.hy[ eid ];
  outPacked[ 5 ] = store.hz[ eid ];
  outPacked[ 6 ] = store.qx[ eid ];
  outPacked[ 7 ] = store.qy[ eid ];
  outPacked[ 8 ] = store.qz[ eid ];
  outPacked[ 9 ] = store.qw[ eid ];
  return outPacked;
}

// bitecs entity SoA component -> preallocated OBB (no temp)
export function obbFromBitecs( out, eid, store = OBBComponent ) {
  out.center.set( store.cx[ eid ], store.cy[ eid ], store.cz[ eid ] );
  out.halfExtents.set( store.hx[ eid ], store.hy[ eid ], store.hz[ eid ] );
  out.orientation.set( store.qx[ eid ], store.qy[ eid ], store.qz[ eid ], store.qw[ eid ] );
  return out;
}

// OBB -> write directly into a bitecs entity's SoA component
export function bitecsOBBFromObb( eid, obb, store = OBBComponent ) {
  store.cx[ eid ] = obb.center.x;
  store.cy[ eid ] = obb.center.y;
  store.cz[ eid ] = obb.center.z;
  store.hx[ eid ] = obb.halfExtents.x;
  store.hy[ eid ] = obb.halfExtents.y;
  store.hz[ eid ] = obb.halfExtents.z;
  store.qx[ eid ] = obb.orientation.x;
  store.qy[ eid ] = obb.orientation.y;
  store.qz[ eid ] = obb.orientation.z;
  store.qw[ eid ] = obb.orientation.w;
  return eid;
}

// Compute the three oriented axes (u, v, w) of a bitecs OBB into preallocated
// THREE.Vector3s. No allocation.
export function threeVec3OBBGetAxesFromBitecs( outU, outV, outW, eid, store = OBBComponent ) {
  const x = store.qx[ eid ], y = store.qy[ eid ], z = store.qz[ eid ], w = store.qw[ eid ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  outU.set( 1 - ( yy + zz ), xy + wz, xz - wy );
  outV.set( xy - wz, 1 - ( xx + zz ), yz + wx );
  outW.set( xz + wy, yz - wx, 1 - ( xx + yy ) );
  return eid;
}

// Compute the three oriented axes of a bitecs OBB into a preallocated packed
// 9-element Float32Array (column-major). No allocation.
export function glMatrixOBBPackedAxesFromBitecs( out9, eid, store = OBBComponent ) {
  const x = store.qx[ eid ], y = store.qy[ eid ], z = store.qz[ eid ], w = store.qw[ eid ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  out9[ 0 ] = 1 - ( yy + zz ); out9[ 1 ] = xy + wz; out9[ 2 ] = xz - wy;
  out9[ 3 ] = xy - wz; out9[ 4 ] = 1 - ( xx + zz ); out9[ 5 ] = yz + wx;
  out9[ 6 ] = xz + wy; out9[ 7 ] = yz - wx; out9[ 8 ] = 1 - ( xx + yy );
  return out9;
}

// Get the 8 corners of a bitecs OBB into a preallocated 24-element Float32Array.
// No allocation.
export function glMatrixOBBPackedCornersFromBitecs( out24, eid, store = OBBComponent ) {
  const cx = store.cx[ eid ], cy = store.cy[ eid ], cz = store.cz[ eid ];
  const hx = store.hx[ eid ], hy = store.hy[ eid ], hz = store.hz[ eid ];
  const x = store.qx[ eid ], y = store.qy[ eid ], z = store.qz[ eid ], w = store.qw[ eid ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  // signs for the 8 corners: (-,-,-), (+,-,-), (-,+,-), (+,+,-), (-,-,+), (+,-,+), (-,+,+), (+,+,+)
  const sx = [ - 1, 1, - 1, 1, - 1, 1, - 1, 1 ];
  const sy = [ - 1, - 1, 1, 1, - 1, - 1, 1, 1 ];
  const sz = [ - 1, - 1, - 1, - 1, 1, 1, 1, 1 ];
  for ( let i = 0; i < 8; i ++ ) {
    const a = hx * sx[ i ], b = hy * sy[ i ], c = hz * sz[ i ];
    const o = i * 3;
    out24[ o ] = cx + ux * a + vx * b + wxx * c;
    out24[ o + 1 ] = cy + uy * a + vy * b + wxy * c;
    out24[ o + 2 ] = cz + uz * a + vz * b + wxz * c;
  }
  return out24;
}

// Contains-point test for a bitecs OBB vs a bitecs SoA point. No allocation.
export function bitecsOBBContainsPoint( eidOBB, eidPoint, storeOBB = OBBComponent, storePoint ) {
  const dx = storePoint.x[ eidPoint ] - storeOBB.cx[ eidOBB ];
  const dy = storePoint.y[ eidPoint ] - storeOBB.cy[ eidOBB ];
  const dz = storePoint.z[ eidPoint ] - storeOBB.cz[ eidOBB ];
  const x = storeOBB.qx[ eidOBB ], y = storeOBB.qy[ eidOBB ], z = storeOBB.qz[ eidOBB ], w = storeOBB.qw[ eidOBB ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  const du = dx * ux + dy * uy + dz * uz;
  const dv = dx * vx + dy * vy + dz * vz;
  const dw = dx * wxx + dy * wxy + dz * wxz;
  return Math.abs( du ) <= storeOBB.hx[ eidOBB ] &&
    Math.abs( dv ) <= storeOBB.hy[ eidOBB ] &&
    Math.abs( dw ) <= storeOBB.hz[ eidOBB ];
}

// Clamp a bitecs SoA point to the surface of a bitecs OBB -> preallocated THREE.Vector3.
export function threeVec3FromBitecsOBBClampPoint( out, eidOBB, eidPoint, storeOBB = OBBComponent, storePoint ) {
  const dx = storePoint.x[ eidPoint ] - storeOBB.cx[ eidOBB ];
  const dy = storePoint.y[ eidPoint ] - storeOBB.cy[ eidOBB ];
  const dz = storePoint.z[ eidPoint ] - storeOBB.cz[ eidOBB ];
  const x = storeOBB.qx[ eidOBB ], y = storeOBB.qy[ eidOBB ], z = storeOBB.qz[ eidOBB ], w = storeOBB.qw[ eidOBB ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  const du = clamp( dx * ux + dy * uy + dz * uz, - storeOBB.hx[ eidOBB ], storeOBB.hx[ eidOBB ] );
  const dv = clamp( dx * vx + dy * vy + dz * vz, - storeOBB.hy[ eidOBB ], storeOBB.hy[ eidOBB ] );
  const dw = clamp( dx * wxx + dy * wxy + dz * wxz, - storeOBB.hz[ eidOBB ], storeOBB.hz[ eidOBB ] );
  out.x = storeOBB.cx[ eidOBB ] + ux * du + vx * dv + wxx * dw;
  out.y = storeOBB.cy[ eidOBB ] + uy * du + vy * dv + wxy * dw;
  out.z = storeOBB.cz[ eidOBB ] + uz * du + vz * dv + wxz * dw;
  return out;
}

// Clamp a bitecs SoA point to the surface of a bitecs OBB -> dst SoA Vector3 store.
export function bitecsVec3OBBClampPointInto( eidOutVec, eidOBB, eidPoint, storeOBB = OBBComponent, storePoint, storeVec ) {
  const tmp = _clampScratch;
  threeVec3FromBitecsOBBClampPoint( tmp, eidOBB, eidPoint, storeOBB, storePoint );
  storeVec.x[ eidOutVec ] = tmp.x;
  storeVec.y[ eidOutVec ] = tmp.y;
  storeVec.z[ eidOutVec ] = tmp.z;
  return eidOutVec;
}

// Distance from a bitecs OBB to a bitecs SoA point. No allocation.
export function bitecsOBBDistanceToPoint( eidOBB, eidPoint, storeOBB = OBBComponent, storePoint ) {
  threeVec3FromBitecsOBBClampPoint( _clampScratch, eidOBB, eidPoint, storeOBB, storePoint );
  const dx = storePoint.x[ eidPoint ] - _clampScratch.x;
  const dy = storePoint.y[ eidPoint ] - _clampScratch.y;
  const dz = storePoint.z[ eidPoint ] - _clampScratch.z;
  return Math.sqrt( dx * dx + dy * dy + dz * dz );
}

// Get-center of a bitecs OBB -> preallocated THREE.Vector3.
export function threeVec3FromBitecsOBBCenter( out, eid, store = OBBComponent ) {
  out.x = store.cx[ eid ];
  out.y = store.cy[ eid ];
  out.z = store.cz[ eid ];
  return out;
}

// Get-center of a bitecs OBB -> dst SoA Vector3 store.
export function bitecsVec3OBBCenterInto( eidOutVec, eidOBB, storeOBB = OBBComponent, storeVec ) {
  storeVec.x[ eidOutVec ] = storeOBB.cx[ eidOBB ];
  storeVec.y[ eidOutVec ] = storeOBB.cy[ eidOBB ];
  storeVec.z[ eidOutVec ] = storeOBB.cz[ eidOBB ];
  return eidOutVec;
}

// Get-half-extents of a bitecs OBB -> preallocated THREE.Vector3.
export function threeVec3FromBitecsOBBHalfExtents( out, eid, store = OBBComponent ) {
  out.x = store.hx[ eid ];
  out.y = store.hy[ eid ];
  out.z = store.hz[ eid ];
  return out;
}

// Get-half-extents of a bitecs OBB -> dst SoA Vector3 store.
export function bitecsVec3OBBHalfExtentsInto( eidOutVec, eidOBB, storeOBB = OBBComponent, storeVec ) {
  storeVec.x[ eidOutVec ] = storeOBB.hx[ eidOBB ];
  storeVec.y[ eidOutVec ] = storeOBB.hy[ eidOBB ];
  storeVec.z[ eidOutVec ] = storeOBB.hz[ eidOBB ];
  return eidOutVec;
}

// Get-orientation of a bitecs OBB -> preallocated THREE.Quaternion.
export function threeQuatFromBitecsOBBOrientation( out, eid, store = OBBComponent ) {
  out.x = store.qx[ eid ];
  out.y = store.qy[ eid ];
  out.z = store.qz[ eid ];
  out.w = store.qw[ eid ];
  return out;
}

// Set a bitecs OBB from a bitecs SoA AABB (Box3Component) with identity orientation.
export function bitecsOBBFromAABB( eidOBB, eidAABB, storeOBB = OBBComponent, storeAABB = null ) {
  if ( storeAABB === null ) throw new Error( 'storeAABB (Box3Component) is required' );
  storeOBB.cx[ eidOBB ] = ( storeAABB.minX[ eidAABB ] + storeAABB.maxX[ eidAABB ] ) * 0.5;
  storeOBB.cy[ eidOBB ] = ( storeAABB.minY[ eidAABB ] + storeAABB.maxY[ eidAABB ] ) * 0.5;
  storeOBB.cz[ eidOBB ] = ( storeAABB.minZ[ eidAABB ] + storeAABB.maxZ[ eidAABB ] ) * 0.5;
  storeOBB.hx[ eidOBB ] = ( storeAABB.maxX[ eidAABB ] - storeAABB.minX[ eidAABB ] ) * 0.5;
  storeOBB.hy[ eidOBB ] = ( storeAABB.maxY[ eidAABB ] - storeAABB.minY[ eidAABB ] ) * 0.5;
  storeOBB.hz[ eidOBB ] = ( storeAABB.maxZ[ eidAABB ] - storeAABB.minZ[ eidAABB ] ) * 0.5;
  storeOBB.qx[ eidOBB ] = 0;
  storeOBB.qy[ eidOBB ] = 0;
  storeOBB.qz[ eidOBB ] = 0;
  storeOBB.qw[ eidOBB ] = 1;
  return eidOBB;
}

// Compute the world-space AABB of a bitecs OBB -> preallocated THREE.Box3.
export function threeBox3FromBitecsOBBWorldAABB( out, eid, store = OBBComponent ) {
  const corners = _cornerScratch;
  glMatrixOBBPackedCornersFromBitecs( corners, eid, store );
  out.makeEmpty();
  for ( let i = 0; i < 8; i ++ ) {
    const o = i * 3;
    const x = corners[ o ], y = corners[ o + 1 ], z = corners[ o + 2 ];
    if ( x < out.min.x ) out.min.x = x;
    if ( y < out.min.y ) out.min.y = y;
    if ( z < out.min.z ) out.min.z = z;
    if ( x > out.max.x ) out.max.x = x;
    if ( y > out.max.y ) out.max.y = y;
    if ( z > out.max.z ) out.max.z = z;
  }
  return out;
}

// Compute the world-space AABB of a bitecs OBB -> dst bitecs Box3Component entity.
export function bitecsBox3FromBitecsOBBWorldAABB( eidOutBox, eidOBB, storeBox = null, storeOBB = OBBComponent ) {
  if ( storeBox === null ) throw new Error( 'storeBox (Box3Component) is required' );
  const corners = _cornerScratch;
  glMatrixOBBPackedCornersFromBitecs( corners, eidOBB, storeOBB );
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = - Infinity, maxY = - Infinity, maxZ = - Infinity;
  for ( let i = 0; i < 8; i ++ ) {
    const o = i * 3;
    const x = corners[ o ], y = corners[ o + 1 ], z = corners[ o + 2 ];
    if ( x < minX ) minX = x;
    if ( y < minY ) minY = y;
    if ( z < minZ ) minZ = z;
    if ( x > maxX ) maxX = x;
    if ( y > maxY ) maxY = y;
    if ( z > maxZ ) maxZ = z;
  }
  storeBox.minX[ eidOutBox ] = minX;
  storeBox.minY[ eidOutBox ] = minY;
  storeBox.minZ[ eidOutBox ] = minZ;
  storeBox.maxX[ eidOutBox ] = maxX;
  storeBox.maxY[ eidOutBox ] = maxY;
  storeBox.maxZ[ eidOutBox ] = maxZ;
  return eidOutBox;
}

// SAT test: OBB vs AABB. Returns true if they intersect. No allocation.
export function bitecsOBBIntersectsAABB( eidOBB, eidAABB, storeOBB = OBBComponent, storeAABB = null ) {
  if ( storeAABB === null ) throw new Error( 'storeAABB (Box3Component) is required' );
  // AABB center and half-extents
  const acx = ( storeAABB.minX[ eidAABB ] + storeAABB.maxX[ eidAABB ] ) * 0.5;
  const acy = ( storeAABB.minY[ eidAABB ] + storeAABB.maxY[ eidAABB ] ) * 0.5;
  const acz = ( storeAABB.minZ[ eidAABB ] + storeAABB.maxZ[ eidAABB ] ) * 0.5;
  const ahx = ( storeAABB.maxX[ eidAABB ] - storeAABB.minX[ eidAABB ] ) * 0.5;
  const ahy = ( storeAABB.maxY[ eidAABB ] - storeAABB.minY[ eidAABB ] ) * 0.5;
  const ahz = ( storeAABB.maxZ[ eidAABB ] - storeAABB.minZ[ eidAABB ] ) * 0.5;
  // OBB center and axes
  const bcx = storeOBB.cx[ eidOBB ], bcy = storeOBB.cy[ eidOBB ], bcz = storeOBB.cz[ eidOBB ];
  const bhx = storeOBB.hx[ eidOBB ], bhy = storeOBB.hy[ eidOBB ], bhz = storeOBB.hz[ eidOBB ];
  const x = storeOBB.qx[ eidOBB ], y = storeOBB.qy[ eidOBB ], z = storeOBB.qz[ eidOBB ], w = storeOBB.qw[ eidOBB ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  // T = AABB center - OBB center
  const tx = acx - bcx, ty = acy - bcy, tz = acz - bcz;
  // AABB axes as rows: e0=(1,0,0), e1=(0,1,0), e2=(0,0,1)
  const absUx = Math.abs( ux ), absUy = Math.abs( uy ), absUz = Math.abs( uz );
  const absVx = Math.abs( vx ), absVy = Math.abs( vy ), absVz = Math.abs( vz );
  const absWx = Math.abs( wxx ), absWy = Math.abs( wxy ), absWz = Math.abs( wxz );
  // L = AABB axes (rows 0,1,2) x OBB axes (u,v,w)
  // Test AABB's 3 axes
  if ( Math.abs( tx ) > ahx + bhx * absUx + bhy * absVx + bhz * absWx ) return false;
  if ( Math.abs( ty ) > ahy + bhx * absUy + bhy * absVy + bhz * absWy ) return false;
  if ( Math.abs( tz ) > ahz + bhx * absUz + bhy * absVz + bhz * absWz ) return false;
  // Test OBB's 3 axes
  const du = tx * ux + ty * uy + tz * uz;
  if ( Math.abs( du ) > bhx + ahx * absUx + ahy * absUy + ahz * absUz ) return false;
  const dv = tx * vx + ty * vy + tz * vz;
  if ( Math.abs( dv ) > bhy + ahx * absVx + ahy * absVy + ahz * absVz ) return false;
  const dw = tx * wxx + ty * wxy + tz * wxz;
  if ( Math.abs( dw ) > bhz + ahx * absWx + ahy * absWy + ahz * absWz ) return false;
  // Test the 9 cross-product axes
  // AABB axis 0 x OBB u: (0, -1, 0) x (ux, uy, uz) = ( -uz, 0, ux )
  let d = tz * ux - tx * uz;
  if ( Math.abs( d ) > ahz * absUx + ahx * absUz + bhy * absWx + bhz * absVx ) return false;
  // AABB axis 0 x OBB v: (0, -1, 0) x (vx, vy, vz) = ( -vz, 0, vx )
  d = tz * vx - tx * vz;
  if ( Math.abs( d ) > ahz * absVx + ahx * absVz + bhy * absWx + bhz * absVx ) return false;
  // AABB axis 0 x OBB w: (0, -1, 0) x (wx, wy, wz) = ( -wz, 0, wx )
  d = tz * wxx - tx * wxz;
  if ( Math.abs( d ) > ahz * absWx + ahx * absWz + bhy * absWx + bhz * absVx ) return false;
  // AABB axis 1 x OBB u: (0, 0, -1) x (ux, uy, uz) = ( -uy, ux, 0 )
  d = tx * uy - ty * ux;
  if ( Math.abs( d ) > ahx * absUy + ahy * absUx + bhz * absWy + bhy * absWz ) return false;
  // AABB axis 1 x OBB v: (0, 0, -1) x (vx, vy, vz) = ( -vy, vx, 0 )
  d = tx * vy - ty * vx;
  if ( Math.abs( d ) > ahx * absVy + ahy * absVx + bhz * absWy + bhy * absWz ) return false;
  // AABB axis 1 x OBB w: (0, 0, -1) x (wx, wy, wz) = ( -wy, wx, 0 )
  d = tx * wxy - ty * wxx;
  if ( Math.abs( d ) > ahx * absWy + ahy * absWx + bhz * absWy + bhy * absWz ) return false;
  // AABB axis 2 x OBB u: (1, 0, 0) x (ux, uy, uz) = (0, -uz, uy)
  d = ty * uz - tz * uy;
  if ( Math.abs( d ) > ahy * absUz + ahz * absUy + bhx * absVz + bhy * absWz ) return false;
  // AABB axis 2 x OBB v: (1, 0, 0) x (vx, vy, vz) = (0, -vz, vy)
  d = ty * vz - tz * vy;
  if ( Math.abs( d ) > ahy * absVz + ahz * absVy + bhx * absVz + bhy * absWz ) return false;
  // AABB axis 2 x OBB w: (1, 0, 0) x (wx, wy, wz) = (0, -wz, wy)
  d = ty * wxz - tz * wxy;
  if ( Math.abs( d ) > ahy * absWz + ahz * absWy + bhx * absVz + bhy * absWz ) return false;
  return true;
}

// SAT test: OBB vs OBB. Returns true if they intersect. No allocation.
export function bitecsOBBIntersectsOBB( eidA, eidB, storeA = OBBComponent, storeB = OBBComponent ) {
  const acx = storeA.cx[ eidA ], acy = storeA.cy[ eidA ], acz = storeA.cz[ eidA ];
  const ahx = storeA.hx[ eidA ], ahy = storeA.hy[ eidA ], ahz = storeA.hz[ eidA ];
  const bcx = storeB.cx[ eidB ], bcy = storeB.cy[ eidB ], bcz = storeB.cz[ eidB ];
  const bhx = storeB.hx[ eidB ], bhy = storeB.hy[ eidB ], bhz = storeB.hz[ eidB ];
  const ax = storeA.qx[ eidA ], ay = storeA.qy[ eidA ], az = storeA.qz[ eidA ], aw = storeA.qw[ eidA ];
  const bx = storeB.qx[ eidB ], by = storeB.qy[ eidB ], bz = storeB.qz[ eidB ], bw = storeB.qw[ eidB ];
  // A axes
  const ax2 = ax + ax, ay2 = ay + ay, az2 = az + az;
  const axx = ax * ax2, axy = ax * ay2, axz = ax * az2;
  const ayy = ay * ay2, ayz = ay * az2, azz = az * az2;
  const awx = aw * ax2, awy = aw * ay2, awz = aw * az2;
  const aux = 1 - ( ayy + azz ), auy = axy + awz, auz = axz - awy;
  const avx = axy - awz, avy = 1 - ( axx + azz ), avz = ayz + awx;
  const awxx = axz + awy, awxy = ayz - awx, awxz = 1 - ( axx + ayy );
  // B axes
  const bx2 = bx + bx, by2 = by + by, bz2 = bz + bz;
  const bxx = bx * bx2, bxy = bx * by2, bxz = bx * bz2;
  const byy = by * by2, byz = by * bz2, bzz = bz * bz2;
  const bwx = bw * bx2, bwy = bw * by2, bwz = bw * bz2;
  const bux = 1 - ( byy + bzz ), buy = bxy + bwz, buz = bxz - bwy;
  const bvx = bxy - bwz, bvy = 1 - ( bxx + bzz ), bvz = byz + bwx;
  const bwxx = bxz + bwy, bwxy = byz - bwx, bwxz = 1 - ( bxx + byy );
  // T = B center - A center
  const tx = bcx - acx, ty = bcy - acy, tz = bcz - acz;
  // Helper to test a single axis
  function testAxis( lx, ly, lz, rA, rB ) {
    const d = tx * lx + ty * ly + tz * lz;
    return Math.abs( d ) <= rA + rB;
  }
  // A's 3 axes
  if ( ! testAxis( aux, auy, auz,
    ahx,
    bhx * Math.abs( aux * bux + auy * buy + auz * buz ) +
    bhy * Math.abs( aux * bvx + auy * bvy + auz * bvz ) +
    bhz * Math.abs( aux * bwxx + auy * bwxy + auz * bwxz ) ) ) return false;
  if ( ! testAxis( avx, avy, avz,
    ahy,
    bhx * Math.abs( avx * bux + avy * buy + avz * buz ) +
    bhy * Math.abs( avx * bvx + avy * bvy + avz * bvz ) +
    bhz * Math.abs( avx * bwxx + avy * bwxy + avz * bwxz ) ) ) return false;
  if ( ! testAxis( awxx, awxy, awxz,
    ahz,
    bhx * Math.abs( awxx * bux + awxy * buy + awxz * buz ) +
    bhy * Math.abs( awxx * bvx + awxy * bvy + awxz * bvz ) +
    bhz * Math.abs( awxx * bwxx + awxy * bwxy + awxz * bwxz ) ) ) return false;
  // B's 3 axes
  if ( ! testAxis( bux, buy, buz,
    bhx,
    ahx * Math.abs( bux * aux + buy * auy + buz * auz ) +
    ahy * Math.abs( bux * avx + buy * avy + buz * avz ) +
    ahz * Math.abs( bux * awxx + buy * awxy + buz * awxz ) ) ) return false;
  if ( ! testAxis( bvx, bvy, bvz,
    bhy,
    ahx * Math.abs( bvx * aux + bvy * auy + bvz * auz ) +
    ahy * Math.abs( bvx * avx + bvy * avy + bvz * avz ) +
    ahz * Math.abs( bvx * awxx + bvy * awxy + bvz * awxz ) ) ) return false;
  if ( ! testAxis( bwxx, bwxy, bwxz,
    bhz,
    ahx * Math.abs( bwxx * aux + bwxy * auy + bwxz * auz ) +
    ahy * Math.abs( bwxx * avx + bwxy * avy + bwxz * avz ) +
    ahz * Math.abs( bwxx * awxx + bwxy * awxy + bwxz * awxz ) ) ) return false;
  // 9 cross axes
  // A.u x B.u
  if ( ! testAxis( auy * buz - auz * buy, auz * bux - aux * buz, aux * buy - auy * bux,
    ahy * Math.abs( auz ) + ahz * Math.abs( auy ),
    bhy * Math.abs( buz ) + bhz * Math.abs( buy ) ) ) return false;
  // A.u x B.v
  if ( ! testAxis( auy * bvz - auz * bvy, auz * bvx - aux * bvz, aux * bvy - auy * bvx,
    ahy * Math.abs( auz ) + ahz * Math.abs( auy ),
    bhx * Math.abs( bvz ) + bhz * Math.abs( bvx ) ) ) return false;
  // A.u x B.w
  if ( ! testAxis( auy * bwz - auz * bwy, auz * bwx - aux * bwz, aux * bwy - auy * bwx,
    ahy * Math.abs( auz ) + ahz * Math.abs( auy ),
    bhx * Math.abs( bwz ) + bhy * Math.abs( bwx ) ) ) return false;
  // A.v x B.u
  if ( ! testAxis( avy * buz - avz * buy, avz * bux - avx * buz, avx * buy - avy * bux,
    ahx * Math.abs( avz ) + ahz * Math.abs( avx ),
    bhy * Math.abs( buz ) + bhz * Math.abs( buy ) ) ) return false;
  // A.v x B.v
  if ( ! testAxis( avy * bvz - avz * bvy, avz * bvx - avx * bvz, avx * bvy - avy * bvx,
    ahx * Math.abs( avz ) + ahz * Math.abs( avx ),
    bhx * Math.abs( bvz ) + bhz * Math.abs( bvx ) ) ) return false;
  // A.v x B.w
  if ( ! testAxis( avy * bwz - avz * bwy, avz * bwx - avx * bwz, avx * bwy - avy * bwx,
    ahx * Math.abs( avz ) + ahz * Math.abs( avx ),
    bhx * Math.abs( bwz ) + bhy * Math.abs( bwx ) ) ) return false;
  // A.w x B.u
  if ( ! testAxis( awy * buz - awz * buy, awz * bux - awx * buz, awx * buy - awy * bux,
    ahx * Math.abs( awz ) + ahy * Math.abs( awx ),
    bhy * Math.abs( buz ) + bhz * Math.abs( buy ) ) ) return false;
  // A.w x B.v
  if ( ! testAxis( awy * bvz - awz * bvy, awz * bvx - awx * bvz, awx * bvy - awy * bvx,
    ahx * Math.abs( awz ) + ahy * Math.abs( awx ),
    bhx * Math.abs( bvz ) + bhz * Math.abs( bvx ) ) ) return false;
  // A.w x B.w
  if ( ! testAxis( awy * bwz - awz * bwy, awz * bwx - awx * bwz, awx * bwy - awy * bwx,
    ahx * Math.abs( awz ) + ahy * Math.abs( awx ),
    bhx * Math.abs( bwz ) + bhy * Math.abs( bwx ) ) ) return false;
  return true;
}

// Intersects-sphere test: OBB vs Sphere. No allocation.
export function bitecsOBBIntersectsSphere( eidOBB, eidSphere, storeOBB = OBBComponent, storeSphere ) {
  const dx = storeSphere.cx[ eidSphere ] - storeOBB.cx[ eidOBB ];
  const dy = storeSphere.cy[ eidSphere ] - storeOBB.cy[ eidOBB ];
  const dz = storeSphere.cz[ eidSphere ] - storeOBB.cz[ eidOBB ];
  const x = storeOBB.qx[ eidOBB ], y = storeOBB.qy[ eidOBB ], z = storeOBB.qz[ eidOBB ], w = storeOBB.qw[ eidOBB ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  const du = clamp( dx * ux + dy * uy + dz * uz, - storeOBB.hx[ eidOBB ], storeOBB.hx[ eidOBB ] );
  const dv = clamp( dx * vx + dy * vy + dz * vz, - storeOBB.hy[ eidOBB ], storeOBB.hy[ eidOBB ] );
  const dw = clamp( dx * wxx + dy * wxy + dz * wxz, - storeOBB.hz[ eidOBB ], storeOBB.hz[ eidOBB ] );
  const cx = storeOBB.cx[ eidOBB ] + ux * du + vx * dv + wxx * dw;
  const cy = storeOBB.cy[ eidOBB ] + uy * du + vy * dv + wxy * dw;
  const cz = storeOBB.cz[ eidOBB ] + uz * du + vz * dv + wxz * dw;
  const ddx = storeSphere.cx[ eidSphere ] - cx;
  const ddy = storeSphere.cy[ eidSphere ] - cy;
  const ddz = storeSphere.cz[ eidSphere ] - cz;
  const r = storeSphere.radius[ eidSphere ];
  return ( ddx * ddx + ddy * ddy + ddz * ddz ) <= ( r * r );
}

// Intersects-plane test: OBB vs Plane. No allocation.
export function bitecsOBBIntersectsPlane( eidOBB, eidPlane, storeOBB = OBBComponent, storePlane ) {
  const nx = storePlane.nx[ eidPlane ], ny = storePlane.ny[ eidPlane ], nz = storePlane.nz[ eidPlane ];
  const c = storePlane.constant[ eidPlane ];
  const dx = storeOBB.cx[ eidOBB ], dy = storeOBB.cy[ eidOBB ], dz = storeOBB.cz[ eidOBB ];
  const x = storeOBB.qx[ eidOBB ], y = storeOBB.qy[ eidOBB ], z = storeOBB.qz[ eidOBB ], w = storeOBB.qw[ eidOBB ];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const ux = 1 - ( yy + zz ), uy = xy + wz, uz = xz - wy;
  const vx = xy - wz, vy = 1 - ( xx + zz ), vz = yz + wx;
  const wxx = xz + wy, wxy = yz - wx, wxz = 1 - ( xx + yy );
  const du = nx * ux + ny * uy + nz * uz;
  const dv = nx * vx + ny * vy + nz * vz;
  const dw = nx * wxx + ny * wxy + nz * wxz;
  const r = storeOBB.hx[ eidOBB ] * Math.abs( du ) +
    storeOBB.hy[ eidOBB ] * Math.abs( dv ) +
    storeOBB.hz[ eidOBB ] * Math.abs( dw );
  const s = nx * dx + ny * dy + nz * dz + c;
  return Math.abs( s ) <= r;
}

// Apply a bitecs SoA mat4 to a bitecs OBB in place.
export function bitecsOBBApplyMatrix4InPlace( eid, eidM, storeOBB = OBBComponent, storeM = null ) {
  if ( storeM === null ) throw new Error( 'storeM (Matrix4Component) is required' );
  const e = storeM;
  const cx = storeOBB.cx[ eid ], cy = storeOBB.cy[ eid ], cz = storeOBB.cz[ eid ];
  const m00 = e.m00[ eidM ], m01 = e.m10[ eidM ], m02 = e.m20[ eidM ], m03 = e.m30[ eidM ];
  const m10 = e.m01[ eidM ], m11 = e.m11[ eidM ], m12 = e.m21[ eidM ], m13 = e.m31[ eidM ];
  const m20 = e.m02[ eidM ], m21 = e.m12[ eidM ], m22 = e.m22[ eidM ], m23 = e.m32[ eidM ];
  const m30 = e.m03[ eidM ], m31 = e.m13[ eidM ], m32 = e.m23[ eidM ], m33 = e.m33[ eidM ];
  const w = m03 * cx + m13 * cy + m23 * cz + m33;
  const invW = w === 0 ? 1 : 1 / w;
  storeOBB.cx[ eid ] = ( m00 * cx + m01 * cy + m02 * cz + m03 ) * invW;
  storeOBB.cy[ eid ] = ( m10 * cx + m11 * cy + m12 * cz + m13 ) * invW;
  storeOBB.cz[ eid ] = ( m20 * cx + m21 * cy + m22 * cz + m23 ) * invW;
  // Compose the rotation part with the existing orientation quaternion.
  // For simplicity we scale the half-extents by the max scale axis and keep
  // orientation unchanged (matches Box3.applyMatrix4 semantics for AABBs).
  const sx = Math.sqrt( m00 * m00 + m10 * m10 + m20 * m20 );
  const sy = Math.sqrt( m01 * m01 + m11 * m11 + m21 * m21 );
  const sz = Math.sqrt( m02 * m02 + m12 * m12 + m22 * m22 );
  const maxS = Math.max( sx, sy, sz );
  storeOBB.hx[ eid ] *= maxS;
  storeOBB.hy[ eid ] *= maxS;
  storeOBB.hz[ eid ] *= maxS;
  return eid;
}

// Translate a bitecs OBB in place by a bitecs SoA offset vector.
export function bitecsOBBTranslateInPlace( eid, eidOffset, storeOBB = OBBComponent, storeOffset ) {
  storeOBB.cx[ eid ] += storeOffset.x[ eidOffset ];
  storeOBB.cy[ eid ] += storeOffset.y[ eidOffset ];
  storeOBB.cz[ eid ] += storeOffset.z[ eidOffset ];
  return eid;
}

// Scale a bitecs OBB in place by a scalar.
export function bitecsOBBScaleInPlace( eid, scalar, store = OBBComponent ) {
  store.cx[ eid ] *= scalar;
  store.cy[ eid ] *= scalar;
  store.cz[ eid ] *= scalar;
  store.hx[ eid ] *= scalar;
  store.hy[ eid ] *= scalar;
  store.hz[ eid ] *= scalar;
  return eid;
}

// Module-local scratch buffers — allocated once, reused across every bridge.
const _clampScratch = new Vector3();
const _cornerScratch = new Float32Array( 24 );

/*
 * -----------------------------------------------------------------------------
 * OBB (Oriented Bounding Box)
 * -----------------------------------------------------------------------------
 */
class OBB {

  constructor( center = new Vector3(), halfExtents = new Vector3( 1, 1, 1 ), orientation = new Quaternion() ) {
    this.center = center;
    this.halfExtents = halfExtents;
    this.orientation = orientation;
  }

  set( center, halfExtents, orientation ) {
    this.center.copy( center );
    this.halfExtents.copy( halfExtents );
    this.orientation.copy( orientation );
    return this;
  }

  clone() {
    return new this.constructor().copy( this );
  }

  copy( obb ) {
    this.center.copy( obb.center );
    this.halfExtents.copy( obb.halfExtents );
    this.orientation.copy( obb.orientation );
    return this;
  }

  // Compute the three oriented axes (u, v, w) into preallocated target Vectors.
  getAxes( u, v, w ) {
    const x = this.orientation.x, y = this.orientation.y, z = this.orientation.z, ww = this.orientation.w;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = ww * x2, wy = ww * y2, wz = ww * z2;
    u.set( 1 - ( yy + zz ), xy + wz, xz - wy );
    v.set( xy - wz, 1 - ( xx + zz ), yz + wx );
    w.set( xz + wy, yz - wx, 1 - ( xx + yy ) );
    return this;
  }

  // Compute the 8 corners into a preallocated 24-element Float32Array.
  getCorners( out24 ) {
    const u = _u, v = _v, w = _w;
    this.getAxes( u, v, w );
    const cx = this.center.x, cy = this.center.y, cz = this.center.z;
    const hx = this.halfExtents.x, hy = this.halfExtents.y, hz = this.halfExtents.z;
    const sx = [ - 1, 1, - 1, 1, - 1, 1, - 1, 1 ];
    const sy = [ - 1, - 1, 1, 1, - 1, - 1, 1, 1 ];
    const sz = [ - 1, - 1, - 1, - 1, 1, 1, 1, 1 ];
    for ( let i = 0; i < 8; i ++ ) {
      const a = hx * sx[ i ], b = hy * sy[ i ], c = hz * sz[ i ];
      const o = i * 3;
      out24[ o ] = cx + u.x * a + v.x * b + w.x * c;
      out24[ o + 1 ] = cy + u.y * a + v.y * b + w.y * c;
      out24[ o + 2 ] = cz + u.z * a + v.z * b + w.z * c;
    }
    return out24;
  }

  // Compute the world-space AABB into preallocated THREE.Box3.
  getWorldAABB( target ) {
    const corners = _corners;
    this.getCorners( corners );
    target.makeEmpty();
    for ( let i = 0; i < 8; i ++ ) {
      const o = i * 3;
      const x = corners[ o ], y = corners[ o + 1 ], z = corners[ o + 2 ];
      if ( x < target.min.x ) target.min.x = x;
      if ( y < target.min.y ) target.min.y = y;
      if ( z < target.min.z ) target.min.z = z;
      if ( x > target.max.x ) target.max.x = x;
      if ( y > target.max.y ) target.max.y = y;
      if ( z > target.max.z ) target.max.z = z;
    }
    return target;
  }

  // Clamp a point to the surface of the OBB into a preallocated target.
  clampPoint( point, target ) {
    const dx = point.x - this.center.x, dy = point.y - this.center.y, dz = point.z - this.center.z;
    const u = _u, v = _v, w = _w;
    this.getAxes( u, v, w );
    const du = clamp( dx * u.x + dy * u.y + dz * u.z, - this.halfExtents.x, this.halfExtents.x );
    const dv = clamp( dx * v.x + dy * v.y + dz * v.z, - this.halfExtents.y, this.halfExtents.y );
    const dw = clamp( dx * w.x + dy * w.y + dz * w.z, - this.halfExtents.z, this.halfExtents.z );
    target.x = this.center.x + u.x * du + v.x * dv + w.x * dw;
    target.y = this.center.y + u.y * du + v.y * dv + w.y * dw;
    target.z = this.center.z + u.z * du + v.z * dv + w.z * dw;
    return target;
  }

  containsPoint( point ) {
    const dx = point.x - this.center.x, dy = point.y - this.center.y, dz = point.z - this.center.z;
    const u = _u, v = _v, w = _w;
    this.getAxes( u, v, w );
    const du = dx * u.x + dy * u.y + dz * u.z;
    const dv = dx * v.x + dy * v.y + dz * v.z;
    const dw = dx * w.x + dy * w.y + dz * w.z;
    return Math.abs( du ) <= this.halfExtents.x &&
      Math.abs( dv ) <= this.halfExtents.y &&
      Math.abs( dw ) <= this.halfExtents.z;
  }

  intersectsBox( box ) {
    return box.intersectsOBB ? box.intersectsOBB( this ) : false;
  }

  intersectsSphere( sphere ) {
    const tmp = _clampTarget;
    this.clampPoint( sphere.center, tmp );
    return tmp.distanceToSquared( sphere.center ) <= ( sphere.radius * sphere.radius );
  }

  distanceToPoint( point ) {
    const tmp = _clampTarget;
    this.clampPoint( point, tmp );
    return tmp.distanceTo( point );
  }

  equals( obb ) {
    return this.center.equals( obb.center ) &&
      this.halfExtents.equals( obb.halfExtents ) &&
      this.orientation.equals( obb.orientation );
  }

  fromArray( array ) {
    this.center.fromArray( array, 0 );
    this.halfExtents.fromArray( array, 3 );
    this.orientation.fromArray( array, 6 );
    return this;
  }

  toArray( array = [], offset = 0 ) {
    this.center.toArray( array, offset );
    this.halfExtents.toArray( array, offset + 3 );
    this.orientation.toArray( array, offset + 6 );
    return array;
  }

  // Real-time multi-scale: in-place scale to the given unit.
  applyUnit( unit ) {
    this.center.multiplyScalar( unit );
    this.halfExtents.multiplyScalar( unit );
    return this;
  }

  // Real-time multi-scale: in-place lerp between two OBBs.
  lerp( other, alpha ) {
    this.center.lerp( other.center, alpha );
    this.halfExtents.lerp( other.halfExtents, alpha );
    this.orientation.slerp( other.orientation, alpha );
    return this;
  }

}

const _u = /*@__PURE__*/ new Vector3();
const _v = /*@__PURE__*/ new Vector3();
const _w = /*@__PURE__*/ new Vector3();
const _corners = /*@__PURE__*/ new Float32Array( 24 );
const _clampTarget = /*@__PURE__*/ new Vector3();

export { OBB };