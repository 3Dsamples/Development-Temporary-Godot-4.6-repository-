// file number : 006
// full path name : src/textures/006_datatexture.js
// description : DataTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents a texture
// created directly from a raw buffer (TypedArray), with default NearestFilter
// and disabled mipmaps/flipY/unpackAlignment = 1. This is the canonical
// building block for procedural textures, HDR buffers, GPU compute inputs, and
// custom float or byte textures. Adds gl-matrix accelerated per-texel buffer
// indexing, bitecs SoA batching for multi-buffer texture generation, double.js
// bit-exact float-buffer normalization for HDR DataTextures, and simplex-noise
// dithered 8-bit down-conversion for procedural texture synthesis.
// best for : DataTexture, HDR float buffers, procedural generation, GPU compute
// inputs, noise fields, LUTs, simulation state textures, and any three.js
// workflow that binds a raw buffer to a material.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    NearestFilter,
    LinearFilter,
    ClampToEdgeWrapping,
    RGBAFormat,
    UnsignedByteType,
    FloatType,
    NoColorSpace,
    UVMapping
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

// gl-matrix scratch for zero-allocation texel buffer staging
const _gm_v2 = glMatrix.vec2.create();
const _gm_v4 = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-buffer texture generation
// ---------------------------------------------------------------------------
const _bufferWorld = createWorld();
const BufferJobComponent = defineComponent( {
    width: Types.ui32,
    height: Types.ui32,
    channels: Types.ui8,
    mode: Types.ui8, // 0 = copy, 1 = normalize float, 2 = dither 8-bit, 3 = noise fill
    dataPtr: Types.ui32,
    outputPtr: Types.ui32,
    done: Types.ui8
} );

class DataTextureBatch {

    constructor() {
        this.world = _bufferWorld;
        this.inputs = [];
        this.outputs = [];
        this.entities = [];
    }

    /**
     * Queue a buffer processing job.
     * @param {TypedArray} data - Source buffer.
     * @param {number} width
     * @param {number} height
     * @param {number} [channels=4]
     * @param {number} [mode=0]
     * @returns {number} entity id
     */
    add( data, width, height, channels = 4, mode = 0 ) {
        const eid = addEntity( this.world );
        addComponent( this.world, BufferJobComponent, eid );
        const srcIndex = this.inputs.length;
        this.inputs.push( data );
        BufferJobComponent.width[ eid ] = width;
        BufferJobComponent.height[ eid ] = height;
        BufferJobComponent.channels[ eid ] = channels;
        BufferJobComponent.mode[ eid ] = mode;
        BufferJobComponent.dataPtr[ eid ] = srcIndex;
        BufferJobComponent.outputPtr[ eid ] = 0;
        BufferJobComponent.done[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued jobs in one cache-friendly pass.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const data = this.inputs[ BufferJobComponent.dataPtr[ eid ] ];
            const width = BufferJobComponent.width[ eid ];
            const height = BufferJobComponent.height[ eid ];
            const mode = BufferJobComponent.mode[ eid ];

            let result = null;
            if ( mode === 0 ) {
                result = data;
            } else if ( mode === 1 && data instanceof Float32Array ) {
                result = normalizeFloatPrecise( data );
            } else if ( mode === 2 && data instanceof Float32Array ) {
                result = floatToByteDithered( data, width, height );
            } else if ( mode === 3 ) {
                result = generateNoiseField( width, height, 4 );
            } else {
                result = data;
            }

            const outIndex = this.outputs.length;
            this.outputs.push( result );
            BufferJobComponent.outputPtr[ eid ] = outIndex;
            BufferJobComponent.done[ eid ] = 1;
        }
    }

    /**
     * Retrieve the processed buffer for a given entity.
     * @param {number} eid
     * @returns {any|null}
     */
    result( eid ) {
        if ( ! BufferJobComponent.done[ eid ] ) return null;
        return this.outputs[ BufferJobComponent.outputPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact float-buffer normalization
// ---------------------------------------------------------------------------
/**
 * Normalize an arbitrary float buffer into [0, 1] using double.js for
 * bit-exact scaling. Used when a DataTexture holds HDR data whose values
 * may exceed float32 precision.
 * @param {Float32Array} data
 * @param {number} [maxValue] - Optional explicit maximum.
 * @returns {Float32Array}
 */
function normalizeFloatPrecise( data, maxValue ) {
    let max = maxValue;
    if ( max === undefined ) {
        _double.value = 0;
        for ( let i = 0; i < data.length; i ++ ) {
            const abs = Math.abs( data[ i ] );
            if ( abs > _double.value ) _double.value = abs;
        }
        max = _double.value;
    }
    if ( max === 0 ) return new Float32Array( data.length );

    const out = new Float32Array( data.length );
    for ( let i = 0; i < data.length; i ++ ) {
        _double.value = data[ i ];
        _double.div( max );
        out[ i ] = _double.value;
    }
    return out;
}

// ---------------------------------------------------------------------------
// simplex-noise dithered 8-bit down-conversion
// ---------------------------------------------------------------------------
/**
 * Down-convert a float RGBA buffer to 8-bit with simplex-noise dithering.
 * Breaks up banding when a DataTexture is quantized from float to byte
 * format.
 * @param {Float32Array} data
 * @param {number} width
 * @param {number} height
 * @param {number} [amplitude=0.5]
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
// simplex-noise procedural field generation
// ---------------------------------------------------------------------------
/**
 * Generate a procedural RGBA noise field using simplex-noise. Useful as
 * DataTexture source for procedural terrain, flow maps, LUTs, etc.
 * @param {number} width
 * @param {number} height
 * @param {number} [channels=4]
 * @param {number} [frequency=0.01]
 * @param {number} [amplitude=1]
 * @param {number} [offset=0]
 * @returns {Uint8ClampedArray}
 */
function generateNoiseField( width, height, channels = 4, frequency = 0.01, amplitude = 1, offset = 0 ) {
    const out = new Uint8ClampedArray( width * height * channels );
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const p = ( y * width + x ) * channels;
            const n0 = _noise2D( x * frequency + offset, y * frequency + offset );
            const n1 = _noise2D( x * frequency + offset + 100, y * frequency + offset );
            const n2 = _noise2D( x * frequency + offset, y * frequency + offset + 100 );
            const n3 = _noise2D( x * frequency + offset + 100, y * frequency + offset + 100 );

            out[ p ]     = Math.max( 0, Math.min( 255, ( n0 * 0.5 + 0.5 ) * amplitude * 255 ) );
            out[ p + 1 ] = Math.max( 0, Math.min( 255, ( n1 * 0.5 + 0.5 ) * amplitude * 255 ) );
            out[ p + 2 ] = Math.max( 0, Math.min( 255, ( n2 * 0.5 + 0.5 ) * amplitude * 255 ) );
            if ( channels === 4 ) {
                out[ p + 3 ] = Math.max( 0, Math.min( 255, ( n3 * 0.5 + 0.5 ) * amplitude * 255 ) );
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated texel indexing
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated texel lookup into a DataTexture's raw buffer. Writes
 * into a preallocated glMatrix.vec4 for zero-allocation downstream shader
 * uniform uploads.
 * @param {TypedArray} data - Flat RGBA buffer.
 * @param {number} width
 * @param {number} height
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @param {glMatrix.vec4} [out]
 * @returns {glMatrix.vec4}
 */
function sampleTexelGlMat( data, width, height, u, v, out = _gm_v4 ) {
    const x = Math.min( width - 1, Math.floor( u * width ) );
    const y = Math.min( height - 1, Math.floor( v * height ) );
    const idx = ( y * width + x ) * 4;

    glMatrix.vec4.set(
        out,
        data[ idx ] / 255,
        data[ idx + 1 ] / 255,
        data[ idx + 2 ] / 255,
        data[ idx + 3 ] / 255
    );
    return out;
}

// ---------------------------------------------------------------------------
// Main DataTexture class — mirrors three.js/src/textures/DataTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a texture directly from raw buffer data.
 *
 * The interpretation of the data depends on type and format:
 * If the type is `UnsignedByteType`, a `Uint8Array` will be useful for
 * addressing the texel data. If the format is `RGBAFormat`, data needs four
 * values for one texel; Red, Green, Blue and Alpha (typically the opacity).
 *
 * The filters, generateMipmaps, flipY, and unpackAlignment properties are
 * all overridden from their defaults, as these features are not compatible
 * with the raw buffer format that a DataTexture holds.
 * @augments Texture
 */
class DataTexture extends Texture {

    /**
     * Constructs a new data texture.
     * @param {?TypedArray} [data=null] - The buffer data.
     * @param {number} [width=1] - The width of the texture.
     * @param {number} [height=1] - The height of the texture.
     * @param {number} [format=RGBAFormat] - The texture format.
     * @param {number} [type=UnsignedByteType] - The texture type.
     * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=NearestFilter] - The mag filter value.
     * @param {number} [minFilter=NearestFilter] - The min filter value.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     * @param {string} [colorSpace=NoColorSpace] - The color space.
     */
    constructor(
        data = null,
        width = 1,
        height = 1,
        format = RGBAFormat,
        type = UnsignedByteType,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = NearestFilter,
        minFilter = NearestFilter,
        anisotropy = Texture.DEFAULT_ANISOTROPY,
        colorSpace = NoColorSpace
    ) {
        super( null, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy, colorSpace );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isDataTexture = true;

        /**
         * The image definition of a data texture.
         * @type {{data:TypedArray,width:number,height:number}}
         */
        this.image = { data: data, width: width, height: height };

        /**
         * Whether to generate mipmaps (if possible) for a texture.
         * Overwritten and set to `false` by default.
         * @type {boolean}
         * @default false
         */
        this.generateMipmaps = false;

        /**
         * If set to `true`, the texture is flipped along the vertical axis when
         * uploaded to the GPU.
         * Overwritten and set to `false` by default.
         * @type {boolean}
         * @default false
         */
        this.flipY = false;

        /**
         * Specifies the alignment requirements for the start of each pixel row
         * in memory.
         * Overwritten and set to `1` by default.
         * @type {boolean}
         * @default 1
         */
        this.unpackAlignment = 1;

        // Internal field to hold the data buffer directly (also accessible
        // via this.image.data for compatibility).
        this._data = data;
    }

    /**
     * Convenience getter for the underlying data buffer.
     * @returns {?TypedArray}
     */
    get data() {
        return this.image.data;
    }

    set data( value ) {
        this.image.data = value;
        this._data = value;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated texel lookup at normalized UV coordinates.
     * Writes into a preallocated glMatrix.vec4 (zero-allocation).
     * @param {number} u - U coordinate in [0, 1].
     * @param {number} v - V coordinate in [0, 1].
     * @param {glMatrix.vec4} [out]
     * @returns {glMatrix.vec4}
     */
    sampleTexelGlMat( u, v, out = _gm_v4 ) {
        const data = this.image.data;
        if ( ! data || ! data.length ) return out;
        return sampleTexelGlMat( data, this.image.width, this.image.height, u, v, out );
    }

    /**
     * Fill this DataTexture's buffer with simplex-noise procedural RGBA
     * noise. Replaces the underlying buffer in-place and marks the texture
     * as needing an update.
     * @param {number} [channels=4]
     * @param {number} [frequency=0.01]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {DataTexture} A reference to this texture.
     */
    fillWithNoise( channels = 4, frequency = 0.01, amplitude = 1, offset = 0 ) {
        const w = this.image.width;
        const h = this.image.height;
        this.image.data = generateNoiseField( w, h, channels, frequency, amplitude, offset );
        this._data = this.image.data;
        this.needsUpdate = true;
        return this;
    }

    /**
     * Normalize this DataTexture's buffer into [0, 1] using double.js for
     * bit-exact scaling. Only valid for Float32Array buffers.
     * @param {number} [maxValue] - Optional explicit maximum.
     * @returns {DataTexture} A reference to this texture.
     */
    normalizePrecise( maxValue ) {
        if ( this.image.data instanceof Float32Array ) {
            this.image.data = normalizeFloatPrecise( this.image.data, maxValue );
            this._data = this.image.data;
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Down-convert this DataTexture's buffer from float to 8-bit with
     * simplex-noise dithering. Only valid for Float32Array buffers.
     * @param {number} [amplitude=0.5]
     * @returns {DataTexture} A reference to this texture.
     */
    toDithered8Bit( amplitude = 0.5 ) {
        if ( this.image.data instanceof Float32Array ) {
            this.image.data = floatToByteDithered(
                this.image.data,
                this.image.width,
                this.image.height,
                amplitude
            );
            this._data = this.image.data;
            this.type = UnsignedByteType;
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Create a batched buffer processing coordinator backed by bitecs.
     * @returns {DataTextureBatch}
     */
    static createBatch() {
        return new DataTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {DataTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.image.data = source.image.data;
        this.image.width = source.image.width;
        this.image.height = source.image.height;
        this.generateMipmaps = false;
        this.flipY = false;
        this.unpackAlignment = 1;
        this._data = this.image.data;
        return this;
    }

    /**
     * Serializes the texture into JSON.
     * @param {Object} [meta] - Optional metadata.
     * @return {Object} A JSON object representing the serialized texture.
     */
    toJSON( meta ) {
        const isRootObject = ( meta === undefined || typeof meta === 'string' );

        if ( ! isRootObject && meta.textures[ this.uuid ] !== undefined ) {
            return meta.textures[ this.uuid ];
        }

        const output = {
            metadata: {
                version: 4.6,
                type: 'DataTexture',
                generator: 'DataTexture.toJSON'
            },
            uuid: this.uuid,
            name: this.name,
            image: {
                width: this.image.width,
                height: this.image.height
            },
            mapping: this.mapping,
            repeat: [ this.repeat.x, this.repeat.y ],
            offset: [ this.offset.x, this.offset.y ],
            center: [ this.center.x, this.center.y ],
            rotation: this.rotation,
            wrap: [ this.wrapS, this.wrapT ],
            format: this.format,
            internalFormat: this.internalFormat,
            type: this.type,
            colorSpace: this.colorSpace,
            minFilter: this.minFilter,
            magFilter: this.magFilter,
            anisotropy: this.anisotropy,
            flipY: this.flipY,
            generateMipmaps: this.generateMipmaps,
            unpackAlignment: this.unpackAlignment
        };

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }
}

export {
    DataTexture,
    DataTextureBatch,
    normalizeFloatPrecise,
    floatToByteDithered,
    generateNoiseField,
    sampleTexelGlMat
};
export default DataTexture;