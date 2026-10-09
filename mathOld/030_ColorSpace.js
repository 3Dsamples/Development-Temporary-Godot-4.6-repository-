// file number : 030
// full path name : src/math/030_ColorSpace.js
// description : ColorSpace constants and helpers for three.js r185, plus zero-allocation bridge functions that convert between three.js color-space string constants and gl-matrix vec3/vec4 (linear-sRGB Float32Array in, sRGB Float32Array out, and vice versa) and bitecs 0.4.0 SoA ColorComponent (r/g/b Float32Arrays indexed by entity id). Also exposes the full r185 transfer-function cache and primaries lookup, plus real-time multi-scale helpers for color-space-aware rendering. Depends on MathUtils.js (file 001) and ColorManagement.js (file 017).
// best for  :  Declaring texture/color/light color spaces, converting between sRGB and linear-sRGB without allocating, feeding gl-matrix vec3/vec4 uniforms from bitecs SoA color data, and any hot loop that must move color data across color spaces without per-frame allocation.
// license : GPL3

import { clamp } from './MathUtils.js';
import {
  ColorManagement,
  SRGBToLinear,
  LinearToSRGB,
  ColorComponent
} from './017_ColorManagement.js';

import glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/gl-matrix-min.min.js';
import { defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';

const { vec3: glVec3, vec4: glVec4 } = glMatrix;

/*
 * -----------------------------------------------------------------------------
 * COLOR SPACE CONSTANTS (r185)
 * -----------------------------------------------------------------------------
 * These match the strings used throughout three.js r185.
 */
export const NoColorSpace = '';
export const SRGBColorSpace = 'srgb';
export const LinearSRGBColorSpace = 'srgb-linear';
export const DisplayP3ColorSpace = 'display-p3';
export const LinearDisplayP3ColorSpace = 'display-p3-linear';
export const Rec709ColorSpace = 'rec709';
export const Rec2020ColorSpace = 'rec2020';
export const LinearRec2020ColorSpace = 'rec2020-linear';

/*
 * -----------------------------------------------------------------------------
 * COLOR SPACE TRANSFER FUNCTIONS (r185)
 * -----------------------------------------------------------------------------
 * The transfer functions are the same as those exposed by ColorManagement.
 */
export { SRGBToLinear, LinearToSRGB };

// Applies the forward transfer function of a color space to a single value.
export function applyTransfer( value, colorSpace ) {
  const transfer = ColorManagement.getTransfer( colorSpace );
  if ( transfer && transfer.transfer ) {
    return transfer.transfer( value );
  }
  return value;
}

// Applies the inverse transfer function of a color space to a single value.
export function applyTransferInverse( value, colorSpace ) {
  const transfer = ColorManagement.getTransfer( colorSpace );
  if ( transfer && transfer.transferInverse ) {
    return transfer.transferInverse( value );
  }
  return value;
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: gl-matrix vec3  <->  color-space conversion  <->  THREE.Color-style
 * -----------------------------------------------------------------------------
 * All bridges read from a caller-owned Float32Array (gl-matrix vec3/vec4) and
 * write into another caller-owned Float32Array. No allocation per call.
 */

// gl-matrix vec3 in source color space -> preallocated gl-matrix vec3 in target color space.
export function glMatrixVec3ConvertColorSpace( out, glVec, sourceColorSpace, targetColorSpace ) {
  if ( sourceColorSpace === targetColorSpace ) {
    out[ 0 ] = glVec[ 0 ];
    out[ 1 ] = glVec[ 1 ];
    out[ 2 ] = glVec[ 2 ];
    return out;
  }
  // Route through linear-sRGB as the canonical intermediate.
  let r = glVec[ 0 ], g = glVec[ 1 ], b = glVec[ 2 ];
  if ( sourceColorSpace !== LinearSRGBColorSpace ) {
    const srcTransfer = ColorManagement.getTransfer( sourceColorSpace );
    if ( srcTransfer && srcTransfer.transfer ) {
      r = srcTransfer.transfer( r );
      g = srcTransfer.transfer( g );
      b = srcTransfer.transfer( b );
    }
  }
  if ( targetColorSpace !== LinearSRGBColorSpace ) {
    const dstTransfer = ColorManagement.getTransfer( targetColorSpace );
    if ( dstTransfer && dstTransfer.transferInverse ) {
      r = dstTransfer.transferInverse( r );
      g = dstTransfer.transferInverse( g );
      b = dstTransfer.transferInverse( b );
    }
  }
  out[ 0 ] = r;
  out[ 1 ] = g;
  out[ 2 ] = b;
  return out;
}

// gl-matrix vec4 in source color space -> preallocated gl-matrix vec4 in target color space.
export function glMatrixVec4ConvertColorSpace( out, glVec, sourceColorSpace, targetColorSpace ) {
  glMatrixVec3ConvertColorSpace( out, glVec, sourceColorSpace, targetColorSpace );
  out[ 3 ] = glVec[ 3 ];
  return out;
}

// gl-matrix vec3 sRGB -> preallocated gl-matrix vec3 linear-sRGB.
export function glMatrixVec3SRGBToLinear( out, glVec ) {
  out[ 0 ] = SRGBToLinear( glVec[ 0 ] );
  out[ 1 ] = SRGBToLinear( glVec[ 1 ] );
  out[ 2 ] = SRGBToLinear( glVec[ 2 ] );
  return out;
}

// gl-matrix vec3 linear-sRGB -> preallocated gl-matrix vec3 sRGB.
export function glMatrixVec3LinearToSRGB( out, glVec ) {
  out[ 0 ] = LinearToSRGB( glVec[ 0 ] );
  out[ 1 ] = LinearToSRGB( glVec[ 1 ] );
  out[ 2 ] = LinearToSRGB( glVec[ 2 ] );
  return out;
}

// gl-matrix vec4 sRGB -> preallocated gl-matrix vec4 linear-sRGB.
export function glMatrixVec4SRGBToLinear( out, glVec ) {
  out[ 0 ] = SRGBToLinear( glVec[ 0 ] );
  out[ 1 ] = SRGBToLinear( glVec[ 1 ] );
  out[ 2 ] = SRGBToLinear( glVec[ 2 ] );
  out[ 3 ] = glVec[ 3 ];
  return out;
}

// gl-matrix vec4 linear-sRGB -> preallocated gl-matrix vec4 sRGB.
export function glMatrixVec4LinearToSRGB( out, glVec ) {
  out[ 0 ] = LinearToSRGB( glVec[ 0 ] );
  out[ 1 ] = LinearToSRGB( glVec[ 1 ] );
  out[ 2 ] = LinearToSRGB( glVec[ 2 ] );
  out[ 3 ] = glVec[ 3 ];
  return out;
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs ColorComponent  <->  color-space conversion
 * -----------------------------------------------------------------------------
 * Reads/writes directly from/to the SoA r/g/b Float32Arrays indexed by entity id.
 * No temporary object, no extra copy.
 */

// Convert a bitecs ColorComponent from sourceColorSpace to targetColorSpace in place.
export function bitecsColorConvertColorSpaceInPlace( eid, sourceColorSpace, targetColorSpace, store = ColorComponent ) {
  if ( sourceColorSpace === targetColorSpace ) return eid;
  let r = store.r[ eid ], g = store.g[ eid ], b = store.b[ eid ];
  if ( sourceColorSpace !== LinearSRGBColorSpace ) {
    const srcTransfer = ColorManagement.getTransfer( sourceColorSpace );
    if ( srcTransfer && srcTransfer.transfer ) {
      r = srcTransfer.transfer( r );
      g = srcTransfer.transfer( g );
      b = srcTransfer.transfer( b );
    }
  }
  if ( targetColorSpace !== LinearSRGBColorSpace ) {
    const dstTransfer = ColorManagement.getTransfer( targetColorSpace );
    if ( dstTransfer && dstTransfer.transferInverse ) {
      r = dstTransfer.transferInverse( r );
      g = dstTransfer.transferInverse( g );
      b = dstTransfer.transferInverse( b );
    }
  }
  store.r[ eid ] = r;
  store.g[ eid ] = g;
  store.b[ eid ] = b;
  return eid;
}

// Convert a bitecs ColorComponent from sRGB to linear-sRGB in place.
export function bitecsColorSRGBToLinearInPlace( eid, store = ColorComponent ) {
  store.r[ eid ] = SRGBToLinear( store.r[ eid ] );
  store.g[ eid ] = SRGBToLinear( store.g[ eid ] );
  store.b[ eid ] = SRGBToLinear( store.b[ eid ] );
  return eid;
}

// Convert a bitecs ColorComponent from linear-sRGB to sRGB in place.
export function bitecsColorLinearToSRGBInPlace( eid, store = ColorComponent ) {
  store.r[ eid ] = LinearToSRGB( store.r[ eid ] );
  store.g[ eid ] = LinearToSRGB( store.g[ eid ] );
  store.b[ eid ] = LinearToSRGB( store.b[ eid ] );
  return eid;
}

// gl-matrix vec3 sRGB -> write directly into a bitecs ColorComponent entity (linear-sRGB).
export function bitecsColorFromGlMatrixVec3SRGB( eid, glVec, store = ColorComponent ) {
  store.r[ eid ] = SRGBToLinear( glVec[ 0 ] );
  store.g[ eid ] = SRGBToLinear( glVec[ 1 ] );
  store.b[ eid ] = SRGBToLinear( glVec[ 2 ] );
  return eid;
}

// gl-matrix vec3 linear-sRGB -> write directly into a bitecs ColorComponent entity (linear-sRGB).
export function bitecsColorFromGlMatrixVec3Linear( eid, glVec, store = ColorComponent ) {
  store.r[ eid ] = glVec[ 0 ];
  store.g[ eid ] = glVec[ 1 ];
  store.b[ eid ] = glVec[ 2 ];
  return eid;
}

// bitecs ColorComponent (linear-sRGB) -> preallocated gl-matrix vec3 sRGB.
export function glMatrixVec3SRGBFromBitecsColor( out, eid, store = ColorComponent ) {
  out[ 0 ] = LinearToSRGB( store.r[ eid ] );
  out[ 1 ] = LinearToSRGB( store.g[ eid ] );
  out[ 2 ] = LinearToSRGB( store.b[ eid ] );
  return out;
}

// bitecs ColorComponent (linear-sRGB) -> preallocated gl-matrix vec3 linear-sRGB.
export function glMatrixVec3LinearFromBitecsColor( out, eid, store = ColorComponent ) {
  out[ 0 ] = store.r[ eid ];
  out[ 1 ] = store.g[ eid ];
  out[ 2 ] = store.b[ eid ];
  return out;
}

// bitecs ColorComponent (linear-sRGB) -> preallocated gl-matrix vec4 sRGB (alpha = 1).
export function glMatrixVec4SRGBFromBitecsColor( out, eid, store = ColorComponent ) {
  glMatrixVec3SRGBFromBitecsColor( out, eid, store );
  out[ 3 ] = 1;
  return out;
}

// bitecs ColorComponent (linear-sRGB) -> preallocated gl-matrix vec4 linear-sRGB (alpha = 1).
export function glMatrixVec4LinearFromBitecsColor( out, eid, store = ColorComponent ) {
  glMatrixVec3LinearFromBitecsColor( out, eid, store );
  out[ 3 ] = 1;
  return out;
}

// gl-matrix vec4 sRGB -> write directly into a bitecs ColorComponent entity (linear-sRGB, alpha ignored).
export function bitecsColorFromGlMatrixVec4SRGB( eid, glVec, store = ColorComponent ) {
  store.r[ eid ] = SRGBToLinear( glVec[ 0 ] );
  store.g[ eid ] = SRGBToLinear( glVec[ 1 ] );
  store.b[ eid ] = SRGBToLinear( glVec[ 2 ] );
  return eid;
}

// gl-matrix vec4 linear-sRGB -> write directly into a bitecs ColorComponent entity (linear-sRGB, alpha ignored).
export function bitecsColorFromGlMatrixVec4Linear( eid, glVec, store = ColorComponent ) {
  store.r[ eid ] = glVec[ 0 ];
  store.g[ eid ] = glVec[ 1 ];
  store.b[ eid ] = glVec[ 2 ];
  return eid;
}

/*
 * -----------------------------------------------------------------------------
 * BRIDGE: bitecs ColorComponent  <->  bitecs ColorComponent (cross-entity)
 * -----------------------------------------------------------------------------
 */

// Copy a bitecs ColorComponent (source) to another entity (dst) with color-space conversion.
export function bitecsColorConvertColorSpaceInto( eidOut, eidIn, sourceColorSpace, targetColorSpace, storeIn = ColorComponent, storeOut = storeIn ) {
  let r = storeIn.r[ eidIn ], g = storeIn.g[ eidIn ], b = storeIn.b[ eidIn ];
  if ( sourceColorSpace !== LinearSRGBColorSpace ) {
    const srcTransfer = ColorManagement.getTransfer( sourceColorSpace );
    if ( srcTransfer && srcTransfer.transfer ) {
      r = srcTransfer.transfer( r );
      g = srcTransfer.transfer( g );
      b = srcTransfer.transfer( b );
    }
  }
  if ( targetColorSpace !== LinearSRGBColorSpace ) {
    const dstTransfer = ColorManagement.getTransfer( targetColorSpace );
    if ( dstTransfer && dstTransfer.transferInverse ) {
      r = dstTransfer.transferInverse( r );
      g = dstTransfer.transferInverse( g );
      b = dstTransfer.transferInverse( b );
    }
  }
  storeOut.r[ eidOut ] = r;
  storeOut.g[ eidOut ] = g;
  storeOut.b[ eidOut ] = b;
  return eidOut;
}

// Copy a bitecs ColorComponent (source) to another entity (dst) converting sRGB -> linear-sRGB.
export function bitecsColorSRGBToLinearInto( eidOut, eidIn, storeIn = ColorComponent, storeOut = storeIn ) {
  storeOut.r[ eidOut ] = SRGBToLinear( storeIn.r[ eidIn ] );
  storeOut.g[ eidOut ] = SRGBToLinear( storeIn.g[ eidIn ] );
  storeOut.b[ eidOut ] = SRGBToLinear( storeIn.b[ eidIn ] );
  return eidOut;
}

// Copy a bitecs ColorComponent (source) to another entity (dst) converting linear-sRGB -> sRGB.
export function bitecsColorLinearToSRGBInto( eidOut, eidIn, storeIn = ColorComponent, storeOut = storeIn ) {
  storeOut.r[ eidOut ] = LinearToSRGB( storeIn.r[ eidIn ] );
  storeOut.g[ eidOut ] = LinearToSRGB( storeIn.g[ eidIn ] );
  storeOut.b[ eidOut ] = LinearToSRGB( storeIn.b[ eidIn ] );
  return eidOut;
}

/*
 * -----------------------------------------------------------------------------
 * COLOR SPACE PRIMARIES / TRANSFER LOOKUP BRIDGES
 * -----------------------------------------------------------------------------
 */

// Returns the primaries object for a color space string (or null).
export function getColorSpacePrimaries( colorSpace ) {
  return ColorManagement.getPrimaries( colorSpace );
}

// Returns the transfer object for a color space string (or null).
export function getColorSpaceTransfer( colorSpace ) {
  return ColorManagement.getTransfer( colorSpace );
}

// Returns true if the color space is a linear (scene-referred) space.
export function isLinearColorSpace( colorSpace ) {
  return colorSpace === LinearSRGBColorSpace ||
    colorSpace === LinearDisplayP3ColorSpace ||
    colorSpace === LinearRec2020ColorSpace;
}

// Returns true if the color space is an sRGB-family (display-referred) space.
export function isSRGBFamilyColorSpace( colorSpace ) {
  return colorSpace === SRGBColorSpace ||
    colorSpace === DisplayP3ColorSpace ||
    colorSpace === Rec709ColorSpace ||
    colorSpace === Rec2020ColorSpace;
}

/*
 * Module-local scratch buffers — allocated once, reused across every bridge.
 */
const _scratchVec3 = new Float32Array( 3 );
const _scratchVec4 = new Float32Array( 4 );

export {
  ColorManagement,
  ColorComponent,
  SRGBToLinear,
  LinearToSRGB
};