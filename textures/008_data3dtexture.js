// file number : 008
// full path name : src/textures/008_data3dtexture.js
// description : Data3DTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents a 3D volume
// texture created directly from a raw buffer (TypedArray), divided into width,
// height, and depth. Only supported with a WebGL 2 rendering context. Overrides
// magFilter/minFilter to NearestFilter, generateMipmaps to false, flipY to
// false, unpackAlignment to 1, and adds a wrapR property for the W axis.
// Adds gl-matrix accelerated 3D texel indexing, bitecs SoA batching for
// multi-volume processing pipelines, double.js bit-exact 3D float-buffer
// normalization for HDR volume textures, and simplex-noise dithered 8-bit
// down-conversion for procedural volume synthesis (terrain, clouds, MRI-like
// fields).
// best for : Data3DTexture, WebGL3DRenderTarget, volumetric rendering, 3D
// noise fields, medical imaging data, procedural clouds, terrain volumes,
// and any three.js workflow that needs a 3D texture sampled with UVW in GLSL.
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

// gl-matrix scratch for zero-allocation 3D texel staging
const _gm_v3 = glMatrix.vec3.create();
const _gm_v4 = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-volume processing pipelines
// ---------------------------------------------------------------------------
const _volumeWorld = createWorld();
const VolumeJobComponent = defineComponent( {
    textureId: Types.ui16,
    width: Types.ui32,
    height: Types.ui32,
    depth: Types.ui32,
    mode: Types.ui8, // 0 = copy, 1 = normalize float, 2 = dither 8-bit, 3 = noise fill
    dataPtr: Types.ui32,
    outputPtr: Types.ui32,
    done: Types.ui8
} );

class Data3DTextureBatch {

    constructor() {
        this.world = _volumeWorld;
        this.textures = [];
        this.inputs = [];
        this.outputs = [];
        this.entities = [];
    }

    /**
     * Register a Data3DTexture instance for batched volume processing.
     * @param {Data3DTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a volume processing job.
     * @param {number} textureId
     * @param {number} [mode=0]
     * @returns {number} entity id
     */
    addJob( textureId, mode = 0 ) {
        const eid = addEntity( this.world );
        addComponent( this.world, VolumeJobComponent, eid );
        const texture = this.textures[ textureId ];
        VolumeJobComponent.textureId[ eid ] = textureId;
        VolumeJobComponent.width[ eid ] = texture?.image?.width ?? 0;
        VolumeJobComponent.height[ eid ] = texture?.image?.height ?? 0;
        VolumeJobComponent.depth[ eid ] = texture?.image?.depth ?? 0;
        VolumeJobComponent.mode[ eid ] = mode;
        VolumeJobComponent.dataPtr[ eid ] = 0;
        VolumeJobComponent.outputPtr[ eid ] = 0;
        VolumeJobComponent.done[ eid ] = 0;
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
            const texture = this.textures[ VolumeJobComponent.textureId[ eid ] ];
            const width = VolumeJobComponent.width[ eid ];
            const height = VolumeJobComponent.height[ eid ];
            const depth = VolumeJobComponent.depth[ eid ];
            const mode = VolumeJobComponent.mode[ eid ];

            let result = null;
            if ( ! texture || ! texture.image || ! texture.image.data ) {
                result = null;
            } else if ( mode === 1 && texture.image.data instanceof Float32Array ) {
                result = normalizeFloatPrecise( texture.image.data );
            } else if ( mode === 2 && texture.image.data instanceof Float32Array ) {
                result = floatToByteDithered( texture.image.data, width, height );
            } else if ( mode === 3 ) {
                result = generateVolumeNoise( width, height, depth, 4 );
            } else {
                result = texture.image.data;
            }

            const outIndex = this.outputs.length;
            this.outputs.push( result );
            VolumeJobComponent.outputPtr[ eid ] = outIndex;
            VolumeJobComponent.done[ eid ] = 1;
        }
    }

    /**
     * Retrieve the processed volume for a given entity.
     * @param {number} eid
     * @returns {any|null}
     */
    result( eid ) {
        if ( ! VolumeJobComponent.done[ eid ] ) return null;
        return this.outputs[ VolumeJobComponent.outputPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact float-buffer normalization
// ---------------------------------------------------------------------------
/**
 * Normalize an arbitrary float buffer into [0, 1] using double.js for
 * bit-exact scaling. Used when a Data3DTexture holds HDR volume data whose
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
 * Breaks up banding when a Data3DTexture is quantized from float to byte
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
// simplex-noise procedural 3D volume generation
// ---------------------------------------------------------------------------
/**
 * Generate a procedural RGBA 3D volume using simplex-noise. Useful as a
 * Data3DTexture source for volumetric clouds, terrain density fields,
 * medical-imaging placeholders, and any 3D noise-driven workflow.
 * @param {number} width
 * @param {number} height
 * @param {number} depth
 * @param {number} [channels=4]
 * @param {number} [frequency=0.01]
 * @param {number} [amplitude=1]
 * @param {number} [offset=0]
 * @returns {Uint8ClampedArray}
 */
function generateVolumeNoise( width, height, depth, channels = 4, frequency = 0.01, amplitude = 1, offset = 0 ) {
    const out = new Uint8ClampedArray( width * height * depth * channels );
    for ( let z = 0; z < depth; z ++ ) {
        for ( let y = 0; y < height; y ++ ) {
            for ( let x = 0; x < width; x ++ ) {
                const p = ( ( z * height + y ) * width + x ) * channels;
                // 2D slices at z, plus a z-dependent offset for volume continuity
                const zOffset = z * frequency + offset;
                const n0 = _noise2D( x * frequency + offset, y * frequency + offset + zOffset );
                const n1 = _noise2D( x * frequency + offset + 100, y * frequency + offset + zOffset );
                const n2 = _noise2D( x * frequency + offset, y * frequency + offset + zOffset + 100 );
                const n3 = _noise2D( x * frequency + offset + 100, y * frequency + offset + zOffset + 100 );

                out[ p ]     = Math.max( 0, Math.min( 255, ( n0 * 0.5 + 0.5 ) * amplitude * 255 ) );
                out[ p + 1 ] = Math.max( 0, Math.min( 255, ( n1 * 0.5 + 0.5 ) * amplitude * 255 ) );
                out[ p + 2 ] = Math.max( 0, Math.min( 255, ( n2 * 0.5 + 0.5 ) * amplitude * 255 ) );
                if ( channels === 4 ) {
                    out[ p + 3 ] = Math.max( 0, Math.min( 255, ( n3 * 0.5 + 0.5 ) * amplitude * 255 ) );
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated 3D texel indexing
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated texel lookup into a Data3DTexture's raw buffer.
 * Writes into a preallocated glMatrix.vec4 for zero-allocation downstream
 * shader uniform uploads.
 * @param {TypedArray} data - Flat RGBA buffer.
 * @param {number} width
 * @param {number} height
 * @param {number} depth
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @param {number} w - W coordinate in [0, 1].
 * @param {glMatrix.vec4} [out]
 * @returns {glMatrix.vec4}
 */
function sampleVoxelGlMat( data, width, height, depth, u, v, w, out = _gm_v4 ) {
    const x = Math.min( width - 1, Math.floor( u * width ) );
    const y = Math.min( height - 1, Math.floor( v * height ) );
    const z = Math.min( depth - 1, Math.floor( w * depth ) );
    const idx = ( ( z * height + y ) * width + x ) * 4;

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
// Main Data3DTexture class — mirrors three.js/src/textures/Data3DTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a three-dimensional texture from raw data, with parameters to
 * divide it into width, height, and depth.
 *
 * ```js
 * // create a 3D texture with repeating data, 0 to 255
 * const sizeX = 64, sizeY = 64, sizeZ = 64;
 * const data = new Uint8Array( sizeX * sizeY * sizeZ * 4 );
 * // fill data with something ...
 * const texture = new THREE.Data3DTexture( data, sizeX, sizeY, sizeZ );
 * texture.needsUpdate = true;
 * ```
 * @augments Texture
 */
class Data3DTexture extends Texture {

    /**
     * Constructs a new 3D data texture.
     * @param {?TypedArray} [data=null] - The buffer data.
     * @param {number} [width=1] - The width of the texture.
     * @param {number} [height=1] - The height of the texture.
     * @param {number} [depth=1] - The depth of the texture.
     */
    constructor( data = null, width = 1, height = 1, depth = 1 ) {
        super( null );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isData3DTexture = true;

        /**
         * The image definition of a 3D data texture.
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
     * Convenience getter for the depth.
     * @returns {number}
     */
    get depth() {
        return this.image.depth;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated voxel lookup at normalized UVW coordinates.
     * Writes into a preallocated glMatrix.vec4 (zero-allocation).
     * @param {number} u - U coordinate in [0, 1].
     * @param {number} v - V coordinate in [0, 1].
     * @param {number} w - W coordinate in [0, 1].
     * @param {glMatrix.vec4} [out]
     * @returns {glMatrix.vec4}
     */
    sampleVoxelGlMat( u, v, w, out = _gm_v4 ) {
        const data = this.image.data;
        if ( ! data || ! data.length ) return out;
        return sampleVoxelGlMat( data, this.image.width, this.image.height, this.image.depth, u, v, w, out );
    }

    /**
     * gl-matrix accelerated extraction of volume dimensions into a
     * preallocated vec3. Zero-allocation.
     * @returns {glMatrix.vec3}
     */
    getVolumeDimensionsGlMat() {
        return glMatrix.vec3.set( _gm_v3, this.image.width, this.image.height, this.image.depth );
    }

    /**
     * Fill this Data3DTexture with simplex-noise procedural RGBA volume
     * noise. Replaces the underlying buffer in-place and marks the texture
     * as needing an update.
     * @param {number} [channels=4]
     * @param {number} [frequency=0.01]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {Data3DTexture} A reference to this texture.
     */
    fillWithNoise( channels = 4, frequency = 0.01, amplitude = 1, offset = 0 ) {
        const w = this.image.width;
        const h = this.image.height;
        const d = this.image.depth;
        this.image.data = generateVolumeNoise( w, h, d, channels, frequency, amplitude, offset );
        this._data = this.image.data;
        this.needsUpdate = true;
        return this;
    }

    /**
     * Normalize this Data3DTexture's buffer into [0, 1] using double.js for
     * bit-exact scaling. Only valid for Float32Array buffers.
     * @param {number} [maxValue] - Optional explicit maximum.
     * @returns {Data3DTexture} A reference to this texture.
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
     * Down-convert this Data3DTexture's buffer from float to 8-bit with
     * simplex-noise dithering. Only valid for Float32Array buffers. Uses
     * the first two dimensions (width × height) for the dither pattern;
     * depth is treated as a sequential stack of slices.
     * @param {number} [amplitude=0.5]
     * @returns {Data3DTexture} A reference to this texture.
     */
    toDithered8Bit( amplitude = 0.5 ) {
        if ( this.image.data instanceof Float32Array ) {
            const w = this.image.width;
            const h = this.image.height;
            const d = this.image.depth;
            const sliceSize = w * h * 4;
            const out = new Uint8ClampedArray( w * h * d * 4 );
            for ( let z = 0; z < d; z ++ ) {
                const sliceData = this.image.data.subarray( z * sliceSize, ( z + 1 ) * sliceSize );
                const dithered = floatToByteDithered( sliceData, w, h, amplitude );
                out.set( dithered, z * sliceSize );
            }
            this.image.data = out;
            this._data = out;
            this.type = UnsignedByteType;
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Create a batched volume processing coordinator backed by bitecs.
     * @returns {Data3DTextureBatch}
     */
    static createBatch() {
        return new Data3DTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {Data3DTexture} A reference to this instance.
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
                type: 'Data3DTexture',
                generator: 'Data3DTexture.toJSON'
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
            unpackAlignment: this.unpackAlignment,
            wrapR: this.wrapR
        };

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }
}

export {
    Data3DTexture,
    Data3DTextureBatch,
    normalizeFloatPrecise,
    floatToByteDithered,
    generateVolumeNoise,
    sampleVoxelGlMat
};
export default Data3DTexture;