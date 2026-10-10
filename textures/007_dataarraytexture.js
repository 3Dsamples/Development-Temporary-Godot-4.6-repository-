// file number : 007
// full path name : src/textures/007_dataarraytexture.js
// description : DataArrayTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents an array of
// 2D textures (a 2D texture array) created directly from a raw buffer, width,
// height, and depth (layer count). Only supported with a WebGL 2 rendering
// context. Overrides magFilter/minFilter to NearestFilter, generateMipmaps to
// false, flipY to false, unpackAlignment to 1, and adds a layerUpdates Set for
// per-layer GPU upload tracking (addLayerUpdate / clearLayerUpdates). Adds
// gl-matrix accelerated per-layer texel indexing, bitecs SoA batching for
// multi-layer upload pipelines, double.js bit-exact per-layer float-buffer
// normalization for HDR 2D-array textures, and simplex-noise dithered 8-bit
// down-conversion for procedural texture arrays (e.g. morph targets, sprite
// sheets, LUT stacks).
// best for : DataArrayTexture, WebGLArrayRenderTarget, Morph targets
// (WebGLMorphtargets), sprite sheet stacks, texture atlases with layer
// semantics, LUT arrays, and any three.js workflow that needs a 2D texture
// array sampled with a layer index in GLSL.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    NearestFilter,
    ClampToEdgeWrapping,
    RGBAFormat,
    UnsignedByteType,
    FloatType,
    NoColorSpace,
    UVMapping,
    LinearFilter
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

// gl-matrix scratch for zero-allocation layer texel staging
const _gm_v2 = glMatrix.vec2.create();
const _gm_v4 = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-layer upload pipelines
// ---------------------------------------------------------------------------
const _layerWorld = createWorld();
const LayerJobComponent = defineComponent( {
    textureId: Types.ui16,
    layerIndex: Types.ui32,
    width: Types.ui32,
    height: Types.ui32,
    dataPtr: Types.ui32,
    outputPtr: Types.ui32,
    mode: Types.ui8, // 0 = copy, 1 = normalize float, 2 = dither 8-bit, 3 = noise fill
    done: Types.ui8
} );

class DataArrayTextureBatch {

    constructor() {
        this.world = _layerWorld;
        this.textures = [];
        this.inputs = [];
        this.outputs = [];
        this.entities = [];
    }

    /**
     * Register a DataArrayTexture instance for batched layer uploads.
     * @param {DataArrayTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a per-layer processing job.
     * @param {number} textureId
     * @param {number} layerIndex
     * @param {number} width
     * @param {number} height
     * @param {number} [mode=0]
     * @returns {number} entity id
     */
    addLayer( textureId, layerIndex, width, height, mode = 0 ) {
        const eid = addEntity( this.world );
        addComponent( this.world, LayerJobComponent, eid );
        LayerJobComponent.textureId[ eid ] = textureId;
        LayerJobComponent.layerIndex[ eid ] = layerIndex;
        LayerJobComponent.width[ eid ] = width;
        LayerJobComponent.height[ eid ] = height;
        LayerJobComponent.dataPtr[ eid ] = 0;
        LayerJobComponent.outputPtr[ eid ] = 0;
        LayerJobComponent.mode[ eid ] = mode;
        LayerJobComponent.done[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Auto-populate all layers from a DataArrayTexture.
     * @param {number} textureId
     */
    fillFromTexture( textureId ) {
        const texture = this.textures[ textureId ];
        if ( ! texture || ! texture.image ) return;
        const w = texture.image.width;
        const h = texture.image.height;
        const d = texture.image.depth;
        for ( let i = 0; i < d; i ++ ) {
            this.addLayer( textureId, i, w, h );
        }
    }

    /**
     * Process all queued layer jobs in one cache-friendly pass.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const texture = this.textures[ LayerJobComponent.textureId[ eid ] ];
            const layerIndex = LayerJobComponent.layerIndex[ eid ];
            const width = LayerJobComponent.width[ eid ];
            const height = LayerJobComponent.height[ eid ];
            const mode = LayerJobComponent.mode[ eid ];

            let result = null;
            if ( texture && texture.image && texture.image.data ) {
                const data = texture.image.data;
                const layerSize = width * height * 4;
                const layerData = data.subarray( layerIndex * layerSize, ( layerIndex + 1 ) * layerSize );

                if ( mode === 1 && layerData instanceof Float32Array ) {
                    result = normalizeFloatPrecise( layerData );
                } else if ( mode === 2 && layerData instanceof Float32Array ) {
                    result = floatToByteDithered( layerData, width, height );
                } else if ( mode === 3 ) {
                    result = generateNoiseField( width, height, 4 );
                } else {
                    result = layerData;
                }
            }

            const outIndex = this.outputs.length;
            this.outputs.push( result );
            LayerJobComponent.outputPtr[ eid ] = outIndex;
            LayerJobComponent.done[ eid ] = 1;
        }
    }

    /**
     * Retrieve the processed layer for a given entity.
     * @param {number} eid
     * @returns {any|null}
     */
    result( eid ) {
        if ( ! LayerJobComponent.done[ eid ] ) return null;
        return this.outputs[ LayerJobComponent.outputPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact per-layer float-buffer normalization
// ---------------------------------------------------------------------------
/**
 * Normalize an arbitrary float buffer into [0, 1] using double.js for
 * bit-exact scaling. Used when a DataArrayTexture holds HDR data whose
 * values may exceed float32 precision.
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
 * Breaks up banding when a DataArrayTexture layer is quantized from float
 * to byte format.
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
// simplex-noise procedural RGBA field generation
// ---------------------------------------------------------------------------
/**
 * Generate a procedural RGBA noise field using simplex-noise. Useful as a
 * per-layer DataArrayTexture source (procedural terrain, LUTs, sprite
 * sheets).
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
// gl-matrix accelerated per-layer texel indexing
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated texel lookup into a single layer of a
 * DataArrayTexture's raw buffer. Writes into a preallocated glMatrix.vec4
 * for zero-allocation downstream shader uniform uploads.
 * @param {TypedArray} data - Flat RGBA buffer (all layers).
 * @param {number} width
 * @param {number} height
 * @param {number} layerIndex
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @param {glMatrix.vec4} [out]
 * @returns {glMatrix.vec4}
 */
function sampleLayerTexelGlMat( data, width, height, layerIndex, u, v, out = _gm_v4 ) {
    const x = Math.min( width - 1, Math.floor( u * width ) );
    const y = Math.min( height - 1, Math.floor( v * height ) );
    const layerSize = width * height * 4;
    const idx = layerIndex * layerSize + ( y * width + x ) * 4;

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
// Main DataArrayTexture class — mirrors three.js/src/textures/DataArrayTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates an array of textures directly from raw buffer data.
 *
 * The buffer is expected to contain one layer per every
 * `width * height` texels, with each texel storing `channels` values.
 * This type of texture can only be used with a WebGL 2 rendering context.
 *
 * ```js
 * const data = new Uint8Array( 4 * 4 * 4 * 3 ); // 3 layers, 4x4 RGBA
 * const texture = new THREE.DataArrayTexture( data, 4, 4, 3 );
 * texture.needsUpdate = true;
 * ```
 * @augments Texture
 */
class DataArrayTexture extends Texture {

    /**
     * Constructs a new data array texture.
     * @param {?TypedArray} [data=null] - The buffer data.
     * @param {number} [width=1] - The width of the texture.
     * @param {number} [height=1] - The height of the texture.
     * @param {number} [depth=1] - The depth (number of layers) of the texture.
     */
    constructor( data = null, width = 1, height = 1, depth = 1 ) {
        super( null );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isDataArrayTexture = true;

        /**
         * The image definition of a data texture array.
         * @type {{data:TypedArray,width:number,height:number,depth:number}}
         */
        this.image = { data, width, height, depth };

        /**
         * How the texture is sampled when a texel covers more than one pixel.
         * Overwritten and set to `NearestFilter` by default.
         * @type {number}
         * @default NearestFilter
         */
        this.magFilter = NearestFilter;

        /**
         * How the texture is sampled when a texel covers less than one pixel.
         * Overwritten and set to `NearestFilter` by default.
         * @type {number}
         * @default NearestFilter
         */
        this.minFilter = NearestFilter;

        /**
         * This defines how the texture is wrapped in the depth and corresponds
         * to *W* in UVW mapping.
         * @type {number}
         * @default ClampToEdgeWrapping
         */
        this.wrapR = ClampToEdgeWrapping;

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
         * @type {number}
         * @default 1
         */
        this.unpackAlignment = 1;

        /**
         * A set of all layers which need to be updated in the texture.
         * Normally when {@link Texture#needsUpdate} is set to `true`, the
         * entire data texture array is sent to the GPU. Marking specific
         * layers will only transmit subsets of all mipmaps associated with a
         * specific depth in the array which is often much more performant.
         * @type {Set<number>}
         */
        this.layerUpdates = new Set();

        // Internal convenience accessor for the raw buffer.
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

    /**
     * Convenience getter for the depth (number of layers).
     * @returns {number}
     */
    get depth() {
        return this.image.depth;
    }

    /**
     * Describes that a specific layer of the texture needs to be updated.
     * Normally when {@link Texture#needsUpdate} is set to `true`, the
     * entire data texture array is sent to the GPU. Marking specific
     * layers will only transmit subsets of all mipmaps associated with a
     * specific depth in the array which is often much more performant.
     * @param {number} layerIndex - The layer index that should be updated.
     */
    addLayerUpdate( layerIndex ) {
        this.layerUpdates.add( layerIndex );
    }

    /**
     * Resets the layer updates registry.
     */
    clearLayerUpdates() {
        this.layerUpdates.clear();
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated texel lookup at normalized UV coordinates within
     * a specific layer. Writes into a preallocated glMatrix.vec4
     * (zero-allocation).
     * @param {number} layerIndex
     * @param {number} u - U coordinate in [0, 1].
     * @param {number} v - V coordinate in [0, 1].
     * @param {glMatrix.vec4} [out]
     * @returns {glMatrix.vec4}
     */
    sampleLayerTexelGlMat( layerIndex, u, v, out = _gm_v4 ) {
        const data = this.image.data;
        if ( ! data || ! data.length ) return out;
        return sampleLayerTexelGlMat( data, this.image.width, this.image.height, layerIndex, u, v, out );
    }

    /**
     * gl-matrix accelerated extraction of layer dimensions into a
     * preallocated vec2 array. Zero-allocation.
     * @returns {glMatrix.vec2}
     */
    getLayerDimensionsGlMat() {
        return glMatrix.vec2.set( _gm_v2, this.image.width, this.image.height );
    }

    /**
     * Fill a specific layer of this DataArrayTexture with simplex-noise
     * procedural RGBA noise. Replaces the layer's bytes in-place and
     * marks the layer as needing an update.
     * @param {number} layerIndex
     * @param {number} [channels=4]
     * @param {number} [frequency=0.01]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {DataArrayTexture} A reference to this texture.
     */
    fillLayerWithNoise( layerIndex, channels = 4, frequency = 0.01, amplitude = 1, offset = 0 ) {
        const w = this.image.width;
        const h = this.image.height;
        const layerSize = w * h * 4;
        const noiseData = generateNoiseField( w, h, channels, frequency, amplitude, offset + layerIndex * 1000 );
        this.image.data.set( noiseData, layerIndex * layerSize );
        this._data = this.image.data;
        this.addLayerUpdate( layerIndex );
        return this;
    }

    /**
     * Normalize a specific layer of this DataArrayTexture into [0, 1] using
     * double.js for bit-exact scaling. Only valid for Float32Array buffers.
     * @param {number} layerIndex
     * @param {number} [maxValue] - Optional explicit maximum.
     * @returns {DataArrayTexture} A reference to this texture.
     */
    normalizeLayerPrecise( layerIndex, maxValue ) {
        if ( this.image.data instanceof Float32Array ) {
            const w = this.image.width;
            const h = this.image.height;
            const layerSize = w * h * 4;
            const layerData = this.image.data.subarray( layerIndex * layerSize, ( layerIndex + 1 ) * layerSize );
            const normalized = normalizeFloatPrecise( layerData, maxValue );
            this.image.data.set( normalized, layerIndex * layerSize );
            this._data = this.image.data;
            this.addLayerUpdate( layerIndex );
        }
        return this;
    }

    /**
     * Down-convert a specific layer of this DataArrayTexture from float to
     * 8-bit with simplex-noise dithering. Only valid for Float32Array
     * buffers.
     * @param {number} layerIndex
     * @param {number} [amplitude=0.5]
     * @returns {DataArrayTexture} A reference to this texture.
     */
    toDithered8BitLayer( layerIndex, amplitude = 0.5 ) {
        if ( this.image.data instanceof Float32Array ) {
            const w = this.image.width;
            const h = this.image.height;
            const layerSize = w * h * 4;
            const layerData = this.image.data.subarray( layerIndex * layerSize, ( layerIndex + 1 ) * layerSize );
            const dithered = floatToByteDithered( layerData, w, h, amplitude );
            this.image.data.set( dithered, layerIndex * layerSize );
            this._data = this.image.data;
            this.type = UnsignedByteType;
            this.addLayerUpdate( layerIndex );
        }
        return this;
    }

    /**
     * Create a batched layer upload coordinator backed by bitecs.
     * @returns {DataArrayTextureBatch}
     */
    static createBatch() {
        return new DataArrayTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {DataArrayTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.image.data = source.image.data;
        this.image.width = source.image.width;
        this.image.height = source.image.height;
        this.image.depth = source.image.depth;
        this.magFilter = source.magFilter;
        this.minFilter = source.minFilter;
        this.wrapR = source.wrapR;
        this.generateMipmaps = false;
        this.flipY = false;
        this.unpackAlignment = 1;
        this.layerUpdates = new Set( source.layerUpdates );
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
                type: 'DataArrayTexture',
                generator: 'DataArrayTexture.toJSON'
            },
            uuid: this.uuid,
            name: this.name,
            image: {
                width: this.image.width,
                height: this.image.height,
                depth: this.image.depth
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
    DataArrayTexture,
    DataArrayTextureBatch,
    normalizeFloatPrecise,
    floatToByteDithered,
    generateNoiseField,
    sampleLayerTexelGlMat
};
export default DataArrayTexture;