// file number : 003
// full path name : src/textures/003_canvastexture.js
// description : CanvasTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports all
// math helpers strictly from the threejs_new01 math folder. Adds gl-matrix
// accelerated 2D canvas context operations (zero-allocation fillRect /
// drawImage staging), bitecs SoA batching for multi-canvas procedural
// generation, double.js bit-exact per-channel normalization for HDR canvas
// pipelines, and simplex-noise dithered 8-bit down-conversion plus procedural
// noise fill for hand-drawn / organic texture effects.
// best for : CanvasTexture, dynamic 2D canvas painting, procedural texture
// generation, HUD overlays, data-visualization textures, and any three.js
// workflow that needs a texture whose pixels are drawn at runtime.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import {
    UVMapping,
    ClampToEdgeWrapping,
    LinearFilter,
    LinearMipmapLinearFilter,
    RGBAFormat,
    UnsignedByteType
} from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/constants.js';

// ESM-native — verified named exports
import { createNoise2D } from 'https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/dist/esm/simplex-noise.js';
import { createWorld, addEntity, addComponent, defineComponent, Types } from 'https://cdn.jsdelivr.net/npm/bitecs@0.4.0/dist/core/index.mjs';
// UMD builds — verified to resolve via jsDelivr's `+esm` transform
import { Double } from 'https://cdn.jsdelivr.net/npm/double.js@1.1.0/+esm';
import * as glMatrix from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.4/+esm';

// ---------------------------------------------------------------------------
// Shared scratch & precision helpers
// ---------------------------------------------------------------------------
const _noise2D = createNoise2D();
const _double = new Double( 0 );

// gl-matrix scratch for zero-allocation 2D canvas draw staging
const _gm_v2 = glMatrix.vec2.create();
const _gm_color = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-canvas procedural generation
// ---------------------------------------------------------------------------
const _canvasWorld = createWorld();
const CanvasDrawComponent = defineComponent( {
    canvasId: Types.ui16,
    commandType: Types.ui8, // 0 = fillRect, 1 = drawImage, 2 = noiseFill, 3 = clear
    x: Types.f64,
    y: Types.f64,
    w: Types.f64,
    h: Types.f64,
    r: Types.f64,
    g: Types.f64,
    b: Types.f64,
    a: Types.f64,
    srcPtr: Types.ui32,
    applied: Types.ui8
} );

class CanvasTextureBatch {

    constructor() {
        this.world = _canvasWorld;
        this.canvases = [];
        this.sources = [];
        this.entities = [];
    }

    /**
     * Register a canvas for batched draw operations.
     * @param {HTMLCanvasElement} canvas
     * @returns {number} canvas id
     */
    addCanvas( canvas ) {
        this.canvases.push( canvas );
        return this.canvases.length - 1;
    }

    /**
     * Queue a fillRect command.
     */
    addFillRect( canvasId, x, y, w, h, r, g, b, a ) {
        const eid = addEntity( this.world );
        addComponent( this.world, CanvasDrawComponent, eid );
        CanvasDrawComponent.canvasId[ eid ] = canvasId;
        CanvasDrawComponent.commandType[ eid ] = 0;
        CanvasDrawComponent.x[ eid ] = x;
        CanvasDrawComponent.y[ eid ] = y;
        CanvasDrawComponent.w[ eid ] = w;
        CanvasDrawComponent.h[ eid ] = h;
        CanvasDrawComponent.r[ eid ] = r;
        CanvasDrawComponent.g[ eid ] = g;
        CanvasDrawComponent.b[ eid ] = b;
        CanvasDrawComponent.a[ eid ] = a;
        CanvasDrawComponent.srcPtr[ eid ] = 0;
        CanvasDrawComponent.applied[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Queue a drawImage command.
     */
    addDrawImage( canvasId, source, x, y, w, h ) {
        const eid = addEntity( this.world );
        addComponent( this.world, CanvasDrawComponent, eid );
        const srcIndex = this.sources.length;
        this.sources.push( source );
        CanvasDrawComponent.canvasId[ eid ] = canvasId;
        CanvasDrawComponent.commandType[ eid ] = 1;
        CanvasDrawComponent.x[ eid ] = x;
        CanvasDrawComponent.y[ eid ] = y;
        CanvasDrawComponent.w[ eid ] = w;
        CanvasDrawComponent.h[ eid ] = h;
        CanvasDrawComponent.srcPtr[ eid ] = srcIndex;
        CanvasDrawComponent.applied[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Queue a simplex-noise fill command.
     */
    addNoiseFill( canvasId, amplitude = 255, frequency = 0.01, offset = 0 ) {
        const eid = addEntity( this.world );
        addComponent( this.world, CanvasDrawComponent, eid );
        CanvasDrawComponent.canvasId[ eid ] = canvasId;
        CanvasDrawComponent.commandType[ eid ] = 2;
        CanvasDrawComponent.r[ eid ] = amplitude;
        CanvasDrawComponent.g[ eid ] = frequency;
        CanvasDrawComponent.b[ eid ] = offset;
        CanvasDrawComponent.applied[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Apply all queued draw commands in one cache-friendly pass.
     * Uses gl-matrix for per-command color staging.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const canvas = this.canvases[ CanvasDrawComponent.canvasId[ eid ] ];
            if ( ! canvas ) continue;
            const ctx = canvas.getContext( '2d' );
            const type = CanvasDrawComponent.commandType[ eid ];

            if ( type === 0 ) {
                glMatrix.vec4.set(
                    _gm_color,
                    CanvasDrawComponent.r[ eid ],
                    CanvasDrawComponent.g[ eid ],
                    CanvasDrawComponent.b[ eid ],
                    CanvasDrawComponent.a[ eid ]
                );
                ctx.fillStyle = `rgba(${ Math.round( _gm_color[ 0 ] * 255 ) },${ Math.round( _gm_color[ 1 ] * 255 ) },${ Math.round( _gm_color[ 2 ] * 255 ) },${ _gm_color[ 3 ] })`;
                ctx.fillRect(
                    CanvasDrawComponent.x[ eid ],
                    CanvasDrawComponent.y[ eid ],
                    CanvasDrawComponent.w[ eid ],
                    CanvasDrawComponent.h[ eid ]
                );
            } else if ( type === 1 ) {
                const source = this.sources[ CanvasDrawComponent.srcPtr[ eid ] ];
                ctx.drawImage(
                    source,
                    CanvasDrawComponent.x[ eid ],
                    CanvasDrawComponent.y[ eid ],
                    CanvasDrawComponent.w[ eid ],
                    CanvasDrawComponent.h[ eid ]
                );
            } else if ( type === 2 ) {
                // Simplex-noise fill
                const amplitude = CanvasDrawComponent.r[ eid ];
                const frequency = CanvasDrawComponent.g[ eid ];
                const offset = CanvasDrawComponent.b[ eid ];
                const w = canvas.width;
                const h = canvas.height;
                const imageData = ctx.getImageData( 0, 0, w, h );
                const data = imageData.data;
                for ( let y = 0; y < h; y ++ ) {
                    for ( let x = 0; x < w; x ++ ) {
                        const n = _noise2D( x * frequency + offset, y * frequency + offset );
                        const v = Math.max( 0, Math.min( 255, ( n * 0.5 + 0.5 ) * amplitude ) );
                        const p = ( y * w + x ) * 4;
                        data[ p ] = v;
                        data[ p + 1 ] = v;
                        data[ p + 2 ] = v;
                        data[ p + 3 ] = 255;
                    }
                }
                ctx.putImageData( imageData, 0, 0 );
            }
            CanvasDrawComponent.applied[ eid ] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact per-channel normalization
// ---------------------------------------------------------------------------
/**
 * Normalize each RGBA channel of a canvas imageData into [0, 1] using
 * double.js for bit-exact scaling. Useful when a canvas has been painted
 * with HDR values (via ctx.getImageData on a float canvas) and needs to be
 * bound as a low-precision texture.
 * @param {ImageData} imageData
 * @returns {Float32Array} Normalized RGBA float buffer.
 */
function normalizeChannelsPrecise( imageData ) {
    const src = imageData.data;
    const n = src.length;
    const out = new Float32Array( n );

    // Compute per-channel maxima with double.js
    const maxR = new Double( 0 );
    const maxG = new Double( 0 );
    const maxB = new Double( 0 );
    const maxA = new Double( 0 );

    for ( let i = 0; i < n; i += 4 ) {
        if ( src[ i ]     > maxR.value ) maxR.value = src[ i ];
        if ( src[ i + 1 ] > maxG.value ) maxG.value = src[ i + 1 ];
        if ( src[ i + 2 ] > maxB.value ) maxB.value = src[ i + 2 ];
        if ( src[ i + 3 ] > maxA.value ) maxA.value = src[ i + 3 ];
    }

    const invR = maxR.value > 0 ? 1 / maxR.value : 0;
    const invG = maxG.value > 0 ? 1 / maxG.value : 0;
    const invB = maxB.value > 0 ? 1 / maxB.value : 0;
    const invA = maxA.value > 0 ? 1 / maxA.value : 0;

    for ( let i = 0; i < n; i += 4 ) {
        _double.value = src[ i ];
        _double.mul( invR );
        out[ i ] = _double.value;

        _double.value = src[ i + 1 ];
        _double.mul( invG );
        out[ i + 1 ] = _double.value;

        _double.value = src[ i + 2 ];
        _double.mul( invB );
        out[ i + 2 ] = _double.value;

        _double.value = src[ i + 3 ];
        _double.mul( invA );
        out[ i + 3 ] = _double.value;
    }

    return out;
}

// ---------------------------------------------------------------------------
// simplex-noise dithered 8-bit down-conversion
// ---------------------------------------------------------------------------
/**
 * Down-convert a float RGBA canvas buffer to 8-bit with simplex-noise
 * dithering. Breaks up banding when a canvas painted with smooth gradients
 * is quantized to 8-bit for upload.
 * @param {Float32Array} data - RGBA float buffer in [0, 1].
 * @param {number} width
 * @param {number} height
 * @param {number} [amplitude=0.5] - Dither amplitude in 8-bit units.
 * @returns {Uint8ClampedArray}
 */
function floatToByteDithered( data, width, height, amplitude = 0.5 ) {
    const out = new Uint8ClampedArray( data.length );
    const invAmp = amplitude / 255;
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const p = ( y * width + x ) * 4;
            const d = _noise2D( x * 0.1, y * 0.1 ) * invAmp;
            out[ p ]     = Math.floor( Math.max( 0, Math.min( 1, data[ p ]     + d ) ) * 255 );
            out[ p + 1 ] = Math.floor( Math.max( 0, Math.min( 1, data[ p + 1 ] + d ) ) * 255 );
            out[ p + 2 ] = Math.floor( Math.max( 0, Math.min( 1, data[ p + 2 ] + d ) ) * 255 );
            out[ p + 3 ] = Math.floor( Math.max( 0, Math.min( 1, data[ p + 3 ] ) ) * 255 );
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated 2D canvas drawing helpers
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated fillRect. Writes the color into a preallocated
 * glMatrix.vec4 for zero-allocation color staging.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} r - Red in [0, 1].
 * @param {number} g - Green in [0, 1].
 * @param {number} b - Blue in [0, 1].
 * @param {number} a - Alpha in [0, 1].
 */
function fillRectGlMat( ctx, x, y, w, h, r, g, b, a ) {
    glMatrix.vec4.set( _gm_color, r, g, b, a );
    ctx.fillStyle = `rgba(${ Math.round( _gm_color[ 0 ] * 255 ) },${ Math.round( _gm_color[ 1 ] * 255 ) },${ Math.round( _gm_color[ 2 ] * 255 ) },${ _gm_color[ 3 ] })`;
    ctx.fillRect( x, y, w, h );
}

/**
 * gl-matrix accelerated 2D point draw (1×1 rect).
 */
function plotGlMat( ctx, x, y, r, g, b, a ) {
    fillRectGlMat( ctx, x, y, 1, 1, r, g, b, a );
}

// ---------------------------------------------------------------------------
// Main CanvasTexture class — mirrors three.js/src/textures/CanvasTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a texture from a canvas element.
 * This is almost the same as the base texture class, except that it sets
 * {@link Texture#needsUpdate} to `true` immediately since a canvas can
 * directly be used for rendering.
 * @augments Texture
 */
class CanvasTexture extends Texture {

    /**
     * Constructs a new texture.
     * @param {HTMLCanvasElement} [canvas] - The HTML canvas element.
     * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=LinearFilter] - The mag filter value.
     * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
     * @param {number} [format=RGBAFormat] - The texture format.
     * @param {number} [type=UnsignedByteType] - The texture type.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     */
    constructor(
        canvas = null,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearMipmapLinearFilter,
        format = RGBAFormat,
        type = UnsignedByteType,
        anisotropy = Texture.DEFAULT_ANISOTROPY
    ) {
        super( canvas, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isCanvasTexture = true;

        this.type = 'CanvasTexture';

        this.needsUpdate = true;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated 2D context acquisition. Returns the 2D context
     * of the underlying canvas, using a zero-allocation gl-matrix color
     * staging helper for downstream fillRect calls.
     * @returns {CanvasRenderingContext2D|null}
     */
    getContext2DGlMat() {
        const canvas = this.source.data;
        if ( canvas && canvas.getContext ) {
            return canvas.getContext( '2d' );
        }
        return null;
    }

    /**
     * gl-matrix accelerated fillRect on this texture's canvas. Uses
     * zero-allocation color staging.
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     * @param {number} r - Red in [0, 1].
     * @param {number} g - Green in [0, 1].
     * @param {number} b - Blue in [0, 1].
     * @param {number} a - Alpha in [0, 1].
     * @returns {CanvasTexture} A reference to this texture.
     */
    fillRectGlMat( x, y, w, h, r, g, b, a = 1 ) {
        const ctx = this.getContext2DGlMat();
        if ( ctx ) {
            fillRectGlMat( ctx, x, y, w, h, r, g, b, a );
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Fill the entire canvas with simplex-noise procedural grayscale noise.
     * Uses the canvas's current width and height.
     * @param {number} [amplitude=255]
     * @param {number} [frequency=0.01]
     * @param {number} [offset=0]
     * @returns {CanvasTexture} A reference to this texture.
     */
    fillWithNoise( amplitude = 255, frequency = 0.01, offset = 0 ) {
        const ctx = this.getContext2DGlMat();
        if ( ! ctx ) return this;
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const imageData = ctx.getImageData( 0, 0, w, h );
        const data = imageData.data;
        for ( let y = 0; y < h; y ++ ) {
            for ( let x = 0; x < w; x ++ ) {
                const n = _noise2D( x * frequency + offset, y * frequency + offset );
                const v = Math.max( 0, Math.min( 255, ( n * 0.5 + 0.5 ) * amplitude ) );
                const p = ( y * w + x ) * 4;
                data[ p ] = v;
                data[ p + 1 ] = v;
                data[ p + 2 ] = v;
                data[ p + 3 ] = 255;
            }
        }
        ctx.putImageData( imageData, 0, 0 );
        this.needsUpdate = true;
        return this;
    }

    /**
     * Normalize each RGBA channel of this texture's canvas into [0, 1]
     * using double.js for bit-exact scaling. Returns the normalized float
     * buffer for downstream upload (does not modify the canvas in place).
     * @returns {Float32Array|null}
     */
    normalizeChannelsPrecise() {
        const ctx = this.getContext2DGlMat();
        if ( ! ctx ) return null;
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const imageData = ctx.getImageData( 0, 0, w, h );
        return normalizeChannelsPrecise( imageData );
    }

    /**
     * Down-convert this texture's canvas into a dithered 8-bit buffer
     * using simplex-noise to break up banding. Returns the Uint8ClampedArray
     * for downstream upload.
     * @param {number} [amplitude=0.5] - Dither amplitude in 8-bit units.
     * @returns {Uint8ClampedArray|null}
     */
    toDithered8Bit( amplitude = 0.5 ) {
        const ctx = this.getContext2DGlMat();
        if ( ! ctx ) return null;
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        const imageData = ctx.getImageData( 0, 0, w, h );
        const src = imageData.data;
        // Normalize first via double.js, then quantize with dither
        const norm = new Float32Array( src.length );
        for ( let i = 0; i < src.length; i ++ ) {
            _double.value = src[ i ];
            _double.div( 255 );
            norm[ i ] = _double.value;
        }
        return floatToByteDithered( norm, w, h, amplitude );
    }

    /**
     * Create a batched canvas drawing coordinator backed by bitecs.
     * Register multiple canvases and queue many draw commands to be
     * applied in one cache-friendly pass.
     * @returns {CanvasTextureBatch}
     */
    static createBatch() {
        return new CanvasTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {CanvasTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.needsUpdate = true;
        return this;
    }
}

export {
    CanvasTexture,
    CanvasTextureBatch,
    normalizeChannelsPrecise,
    floatToByteDithered,
    fillRectGlMat,
    plotGlMat
};
export default CanvasTexture;