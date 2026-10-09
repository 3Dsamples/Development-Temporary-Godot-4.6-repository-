// file number : 001
// full path name : src/math/MathUtils.js
// description : Core math utility functions for three.js (clamp, lerp, smoothstep, degToRad, randInt, etc.). Rewritten with gl-matrix, double.js, simplex-noise, and bitecs bridge imports and no hot-loop conversion.
// best for  :  Foundational scalar math used by all vector, matrix, and geometry classes. Serves as the base utility layer for the entire math module.
// license : MIT

import glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.js';
import { defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
import double from 'https://cdn.jsdelivr.net/npm/double.js/dist/double.js';
import { createNoise2D, createNoise3D, createNoise4D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';

const { vec2, vec3, vec4, mat2, mat3, mat4, quat } = glMatrix;

/*
 * BRIDGE NOTE:
 * MathUtils is not a vector/matrix type, so it has no direct gl-matrix or bitecs
 * counterpart. The bridge functions that convert between gl-matrix outputs,
 * bitecs component arrays, and THREE.Vector2/3/4/Matrix3/4/Quaternion are
 * implemented in their respective files (Vector2.js, Vector3.js, Vector4.js,
 * Matrix2.js, Matrix3.js, Matrix4.js, Quaternion.js) to avoid duplication and
 * to keep the bridge layer close to the type it serves.
 *
 * The imports above are kept here so that every rewritten math file shares the
 * same external dependency surface, and so that consumers of MathUtils can
 * rely on glMatrix, bitecs, double.js, and simplex-noise being available in
 * the same module graph.
 */

// Inlined from ../utils.js to keep this file self-contained.
function warn( message ) {
  if ( typeof console !== 'undefined' && console.warn ) {
    console.warn( message );
  }
}

const _lut = [
  '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '0a', '0b', '0c', '0d', '0e', '0f',
  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '1a', '1b', '1c', '1d', '1e', '1f',
  '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '2a', '2b', '2c', '2d', '2e', '2f',
  '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '3a', '3b', '3c', '3d', '3e', '3f',
  '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '4a', '4b', '4c', '4d', '4e', '4f',
  '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '5a', '5b', '5c', '5d', '5e', '5f',
  '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '6a', '6b', '6c', '6d', '6e', '6f',
  '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '7a', '7b', '7c', '7d', '7e', '7f',
  '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '8a', '8b', '8c', '8d', '8e', '8f',
  '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '9a', '9b', '9c', '9d', '9e', '9f',
  'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'aa', 'ab', 'ac', 'ad', 'ae', 'af',
  'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'ba', 'bb', 'bc', 'bd', 'be', 'bf',
  'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf',
  'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'da', 'db', 'dc', 'dd', 'de', 'df',
  'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef',
  'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'fa', 'fb', 'fc', 'fd', 'fe', 'ff'
];

let _seed = 1234567;

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function generateUUID() {
  const d0 = Math.random() * 0xffffffff | 0;
  const d1 = Math.random() * 0xffffffff | 0;
  const d2 = Math.random() * 0xffffffff | 0;
  const d3 = Math.random() * 0xffffffff | 0;
  const uuid = _lut[ d0 & 0xff ] + _lut[ d0 >> 8 & 0xff ] + _lut[ d0 >> 16 & 0xff ] + _lut[ d0 >> 24 & 0xff ] + '-' +
    _lut[ d1 & 0xff ] + _lut[ d1 >> 8 & 0xff ] + '-' + _lut[ d1 >> 16 & 0x0f | 0x40 ] + _lut[ d1 >> 24 & 0xff ] + '-' +
    _lut[ d2 & 0x3f | 0x80 ] + _lut[ d2 >> 8 & 0xff ] + '-' + _lut[ d2 >> 16 & 0xff ] + _lut[ d2 >> 24 & 0xff ] +
    _lut[ d3 & 0xff ] + _lut[ d3 >> 8 & 0xff ] + _lut[ d3 >> 16 & 0xff ] + _lut[ d3 >> 24 & 0xff ];
  return uuid.toLowerCase();
}

function clamp( value, min, max ) {
  return Math.max( min, Math.min( max, value ) );
}

function euclideanModulo( n, m ) {
  return ( ( n % m ) + m ) % m;
}

function mapLinear( x, a1, a2, b1, b2 ) {
  return b1 + ( x - a1 ) * ( b2 - b1 ) / ( a2 - a1 );
}

function inverseLerp( x, y, value ) {
  if ( x !== y ) {
    return ( value - x ) / ( y - x );
  } else {
    return 0;
  }
}

function lerp( x, y, t ) {
  return ( 1 - t ) * x + t * y;
}

function damp( x, y, lambda, dt ) {
  return lerp( x, y, 1 - Math.exp( - lambda * dt ) );
}

function pingpong( x, length = 1 ) {
  return length - Math.abs( euclideanModulo( x, length * 2 ) - length );
}

function smoothstep( x, min, max ) {
  if ( x <= min ) return 0;
  if ( x >= max ) return 1;
  x = ( x - min ) / ( max - min );
  return x * x * ( 3 - 2 * x );
}

function smootherstep( x, min, max ) {
  if ( x <= min ) return 0;
  if ( x >= max ) return 1;
  x = ( x - min ) / ( max - min );
  return x * x * x * ( x * ( x * 6 - 15 ) + 10 );
}

function randInt( low, high ) {
  return low + Math.floor( Math.random() * ( high - low + 1 ) );
}

function randFloat( low, high ) {
  return low + Math.random() * ( high - low );
}

function randFloatSpread( range ) {
  return range * ( 0.5 - Math.random() );
}

function seededRandom( s ) {
  if ( s !== undefined ) _seed = s;
  let t = _seed += 0x6D2B79F5;
  t = Math.imul( t ^ t >>> 15, t | 1 );
  t ^= t + Math.imul( t ^ t >>> 7, t | 61 );
  return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;
}

function degToRad( degrees ) {
  return degrees * DEG2RAD;
}

function radToDeg( radians ) {
  return radians * RAD2DEG;
}

function isPowerOfTwo( value ) {
  return ( value & ( value - 1 ) ) === 0 && value !== 0;
}

function ceilPowerOfTwo( value ) {
  return Math.pow( 2, Math.ceil( Math.log( value ) / Math.LN2 ) );
}

function floorPowerOfTwo( value ) {
  return Math.pow( 2, Math.floor( Math.log( value ) / Math.LN2 ) );
}

function setQuaternionFromProperEuler( q, a, b, c, order ) {
  const cos = Math.cos;
  const sin = Math.sin;
  const c2 = cos( b / 2 );
  const s2 = sin( b / 2 );
  const c13 = cos( ( a + c ) / 2 );
  const s13 = sin( ( a + c ) / 2 );
  const c1_3 = cos( ( a - c ) / 2 );
  const s1_3 = sin( ( a - c ) / 2 );
  const c3_1 = cos( ( c - a ) / 2 );
  const s3_1 = sin( ( c - a ) / 2 );
  switch ( order ) {
    case 'XYX': q.set( c2 * s13, s2 * c1_3, s2 * s1_3, c2 * c13 ); break;
    case 'YZY': q.set( s2 * s1_3, c2 * s13, s2 * c1_3, c2 * c13 ); break;
    case 'ZXZ': q.set( s2 * c1_3, s2 * s1_3, c2 * s13, c2 * c13 ); break;
    case 'XZX': q.set( c2 * s13, s2 * s3_1, s2 * c3_1, c2 * c13 ); break;
    case 'YXY': q.set( s2 * c3_1, c2 * s13, s2 * s3_1, c2 * c13 ); break;
    case 'ZYZ': q.set( s2 * s3_1, s2 * c3_1, c2 * s13, c2 * c13 ); break;
    default:
      warn( 'MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: ' + order );
  }
}

function denormalize( value, array ) {
  switch ( array.constructor ) {
    case Float32Array: return value;
    case Uint32Array: return value / 4294967295.0;
    case Uint16Array: return value / 65535.0;
    case Uint8Array: return value / 255.0;
    case Int32Array: return Math.max( value / 2147483647.0, - 1.0 );
    case Int16Array: return Math.max( value / 32767.0, - 1.0 );
    case Int8Array: return Math.max( value / 127.0, - 1.0 );
    default: throw new Error( 'THREE.MathUtils: Invalid component type.' );
  }
}

function normalize( value, array ) {
  switch ( array.constructor ) {
    case Float32Array: return value;
    case Uint32Array: return Math.round( value * 4294967295.0 );
    case Uint16Array: return Math.round( value * 65535.0 );
    case Uint8Array: return Math.round( value * 255.0 );
    case Int32Array: return Math.round( value * 2147483647.0 );
    case Int16Array: return Math.round( value * 32767.0 );
    case Int8Array: return Math.round( value * 127.0 );
    default: throw new Error( 'THREE.MathUtils: Invalid component type.' );
  }
}

/*
 * -----------------------------------------------------------------------------
 * HIGH-PRECISION HELPERS (double.js)
 * -----------------------------------------------------------------------------
 * These wrap double.js's double-double type for callers who need more than
 * IEEE-754 f64 precision (e.g. large-world camera math, Kahan-style accumulation,
 * or converting between scene scales without catastrophic cancellation).
 */

// Adds two numbers using double-double precision and returns a plain number.
function preciseAdd( a, b ) {
  const da = new double.Double( a );
  const db = new double.Double( b );
  return da.add( db ).valueOf();
}

// Subtracts two numbers using double-double precision and returns a plain number.
function preciseSub( a, b ) {
  const da = new double.Double( a );
  const db = new double.Double( b );
  return da.subtract( db ).valueOf();
}

// Multiplies two numbers using double-double precision and returns a plain number.
function preciseMul( a, b ) {
  const da = new double.Double( a );
  const db = new double.Double( b );
  return da.multiply( db ).valueOf();
}

// Divides two numbers using double-double precision and returns a plain number.
function preciseDiv( a, b ) {
  const da = new double.Double( a );
  const db = new double.Double( b );
  return da.divide( db ).valueOf();
}

// High-precision linear interpolation (avoids the cancellation in (1-t)*x + t*y).
function preciseLerp( x, y, t ) {
  const dx = new double.Double( x );
  const dy = new double.Double( y );
  const dt = new double.Double( t );
  const oneMinusT = new double.Double( 1 ).subtract( dt );
  return dx.multiply( oneMinusT ).add( dy.multiply( dt ) ).valueOf();
}

// High-precision inverseLerp (avoids the cancellation in (v - x) / (y - x)).
function preciseInverseLerp( x, y, value ) {
  if ( x === y ) return 0;
  const dv = new double.Double( value );
  const dx = new double.Double( x );
  const dy = new double.Double( y );
  return dv.subtract( dx ).divide( dy.subtract( dx ) ).valueOf();
}

/*
 * -----------------------------------------------------------------------------
 * PROCEDURAL NOISE (simplex-noise)
 * -----------------------------------------------------------------------------
 * Deterministic seeded 2D/3D/4D simplex noise generators. Kept here so the
 * math module has a single canonical noise surface that vector/geometry code
 * can call without importing a second noise library.
 */

function mulberry32( seed ) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul( a ^ a >>> 15, 1 | a );
    t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
    return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;
  };
}

function createNoise2DSeeded( seed = 0 ) {
  return createNoise2D( mulberry32( seed ) );
}

function createNoise3DSeeded( seed = 0 ) {
  return createNoise3D( mulberry32( seed ) );
}

function createNoise4DSeeded( seed = 0 ) {
  return createNoise4D( mulberry32( seed ) );
}

// Convenience wrappers that sample the seeded generators directly.
function noise2D( x, y, seed = 0 ) {
  return createNoise2DSeeded( seed )( x, y );
}

function noise3D( x, y, z, seed = 0 ) {
  return createNoise3DSeeded( seed )( x, y, z );
}

function noise4D( x, y, z, w, seed = 0 ) {
  return createNoise4DSeeded( seed )( x, y, z, w );
}

/*
 * -----------------------------------------------------------------------------
 * BITECS 0.4.0 SOA MATH HELPERS
 * -----------------------------------------------------------------------------
 * Lightweight SoA helpers that other math files use as a common baseline.
 * These do not belong to a specific vector type; they are scalar-in/SoA-out
 * operations shared by Vector2/3/4, matrices, and scalar fields.
 */
export const ScalarComponent = defineComponent( {
  value: Types.f32
} );

// Write a scalar into a bitecs entity's SoA component.
export function bitecsScalarFromValue( eid, value, store = ScalarComponent ) {
  store.value[ eid ] = value;
  return eid;
}

// Read a scalar from a bitecs entity's SoA component.
export function bitecsScalarValue( eid, store = ScalarComponent ) {
  return store.value[ eid ];
}

// Add two bitecs SoA scalars into a destination entity.
export function bitecsScalarAddInto( eidOut, eidA, eidB, storeA = ScalarComponent, storeB = ScalarComponent, storeOut = storeA ) {
  storeOut.value[ eidOut ] = storeA.value[ eidA ] + storeB.value[ eidB ];
  return eidOut;
}

// Multiply two bitecs SoA scalars into a destination entity.
export function bitecsScalarMultiplyInto( eidOut, eidA, eidB, storeA = ScalarComponent, storeB = ScalarComponent, storeOut = storeA ) {
  storeOut.value[ eidOut ] = storeA.value[ eidA ] * storeB.value[ eidB ];
  return eidOut;
}

// Clamp a bitecs SoA scalar in place.
export function bitecsScalarClampInPlace( eid, min, max, store = ScalarComponent ) {
  store.value[ eid ] = clamp( store.value[ eid ], min, max );
  return eid;
}

// Linear interpolate two bitecs SoA scalars into a destination entity.
export function bitecsScalarLerpInto( eidOut, eidA, eidB, alpha, storeA = ScalarComponent, storeB = ScalarComponent, storeOut = storeA ) {
  storeOut.value[ eidOut ] = storeA.value[ eidA ] + ( storeB.value[ eidB ] - storeA.value[ eidA ] ) * alpha;
  return eidOut;
}

// Apply smoothstep to a bitecs SoA scalar in place.
export function bitecsScalarSmoothstepInPlace( eid, min, max, store = ScalarComponent ) {
  store.value[ eid ] = smoothstep( store.value[ eid ], min, max );
  return eid;
}

// Apply a seeded simplex noise field to a bitecs SoA scalar (1D input = x).
export function bitecsScalarNoise1DInPlace( eid, x, frequency, amplitude, seed = 0, store = ScalarComponent ) {
  store.value[ eid ] = noise2D( x * frequency, 0, seed ) * amplitude;
  return eid;
}

// Apply a seeded simplex noise field to a bitecs SoA scalar (2D input = x, y).
export function bitecsScalarNoise2DInPlace( eid, x, y, frequency, amplitude, seed = 0, store = ScalarComponent ) {
  store.value[ eid ] = noise2D( x * frequency, y * frequency, seed ) * amplitude;
  return eid;
}

// Apply a seeded simplex noise field to a bitecs SoA scalar (3D input = x, y, z).
export function bitecsScalarNoise3DInPlace( eid, x, y, z, frequency, amplitude, seed = 0, store = ScalarComponent ) {
  store.value[ eid ] = noise3D( x * frequency, y * frequency, z * frequency, seed ) * amplitude;
  return eid;
}

const MathUtils = {
  DEG2RAD,
  RAD2DEG,
  generateUUID,
  clamp,
  euclideanModulo,
  mapLinear,
  inverseLerp,
  lerp,
  damp,
  pingpong,
  smoothstep,
  smootherstep,
  randInt,
  randFloat,
  randFloatSpread,
  seededRandom,
  degToRad,
  radToDeg,
  isPowerOfTwo,
  ceilPowerOfTwo,
  floorPowerOfTwo,
  setQuaternionFromProperEuler,
  normalize,
  denormalize,
  preciseAdd,
  preciseSub,
  preciseMul,
  preciseDiv,
  preciseLerp,
  preciseInverseLerp,
  createNoise2DSeeded,
  createNoise3DSeeded,
  createNoise4DSeeded,
  noise2D,
  noise3D,
  noise4D
};

export {
  DEG2RAD,
  RAD2DEG,
  generateUUID,
  clamp,
  euclideanModulo,
  mapLinear,
  inverseLerp,
  lerp,
  damp,
  pingpong,
  smoothstep,
  smootherstep,
  randInt,
  randFloat,
  randFloatSpread,
  seededRandom,
  degToRad,
  radToDeg,
  isPowerOfTwo,
  ceilPowerOfTwo,
  floorPowerOfTwo,
  setQuaternionFromProperEuler,
  normalize,
  denormalize,
  preciseAdd,
  preciseSub,
  preciseMul,
  preciseDiv,
  preciseLerp,
  preciseInverseLerp,
  createNoise2DSeeded,
  createNoise3DSeeded,
  createNoise4DSeeded,
  noise2D,
  noise3D,
  noise4D,
  ScalarComponent,
  bitecsScalarFromValue,
  bitecsScalarValue,
  bitecsScalarAddInto,
  bitecsScalarMultiplyInto,
  bitecsScalarClampInPlace,
  bitecsScalarLerpInto,
  bitecsScalarSmoothstepInPlace,
  bitecsScalarNoise1DInPlace,
  bitecsScalarNoise2DInPlace,
  bitecsScalarNoise3DInPlace,
  MathUtils
};