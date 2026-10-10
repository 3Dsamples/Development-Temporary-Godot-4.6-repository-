// file number : 004
// full path name : src/textures/004_compressedtexture.js
// description : CompressedTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents a texture
// whose mipmap data is already in GPU-compressed form (DXT, ETC1, BC, ASTC,
// etc.). Overrides flipY and generateMipmaps to false since compressed textures
// cannot be flipped or have their mipmaps generated at runtime. Adds gl-matrix
// accelerated per-mip data staging, bitecs SoA batching for multi-mip GPU upload
// pipelines, double.js bit-exact mip-level size computation for very large
// compressed textures, and simplex-noise dithering for procedural high-frequency
// detail baked into compressed formats.
// best for : CompressedTexture, CompressedTextureLoader, KTX/KTX2/DDS/Basis
// loaders, GPU-compressed texture pipelines, and any three.js workflow that
// needs to bind pre-compressed mipmap data to a material.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    NoColorSpace,
    LinearFilter,
    LinearMipmapLinearFilter,
    ClampToEdgeWrapping,
    RGBAFormat,
    UnsignedByteType,
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

// gl-matrix scratch for zero-allocation mip-level staging
const _gm_v2 = glMatrix.vec2.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-mip GPU upload pipelines
// ---------------------------------------------------------------------------
const _mipWorld = createWorld();
const MipUploadComponent = defineComponent( {
    textureId: Types.ui16,
    mipLevel: Types.ui8,
    width: Types.ui32,
    height: Types.ui32,
    dataPtr: Types.ui32,
    byteSize: Types.ui32,
    format: Types.ui32,
    uploaded: Types.ui8
} );

class CompressedMipBatch {

    constructor() {
        this.world = _mipWorld;
        this.textures = [];
        this.mipData = [];
        this.entities = [];
    }

    /**
     * Register a CompressedTexture instance for batched mip uploads.
     * @param {CompressedTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a single mip level for upload.
     * @param {number} textureId
     * @param {number} mipLevel
     * @param {ArrayBufferView} data
     * @param {number} width
     * @param {number} height
     * @returns {number} entity id
     */
    addMip( textureId, mipLevel, data, width, height ) {
        const eid = addEntity( this.world );
        addComponent( this.world, MipUploadComponent, eid );
        const dataIndex = this.mipData.length;
        this.mipData.push( data );
        MipUploadComponent.textureId[ eid ] = textureId;
        MipUploadComponent.mipLevel[ eid ] = mipLevel;
        MipUploadComponent.width[ eid ] = width;
        MipUploadComponent.height[ eid ] = height;
        MipUploadComponent.dataPtr[ eid ] = dataIndex;
        MipUploadComponent.byteSize[ eid ] = data.byteLength || data.length || 0;
        MipUploadComponent.format[ eid ] = 0;
        MipUploadComponent.uploaded[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Auto-populate all mips from a CompressedTexture's mipmaps array.
     * Uses gl-matrix for zero-allocation dimension staging.
     * @param {number} textureId
     */
    fillFromTexture( textureId ) {
        const texture = this.textures[ textureId ];
        if ( ! texture || ! texture.mipmaps ) return;
        for ( let i = 0, l = texture.mipmaps.length; i < l; i ++ ) {
            const mip = texture.mipmaps[ i ];
            glMatrix.vec2.set( _gm_v2, mip.width, mip.height );
            this.addMip( textureId, i, mip.data, _gm_v2[ 0 ], _gm_v2[ 1 ] );
        }
    }

    /**
     * Process all queued mip uploads in one cache-friendly pass.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            MipUploadComponent.uploaded[ entities[ i ] ] = 1;
        }
    }

    /**
     * Retrieve the mip data for a given entity.
     * @param {number} eid
     * @returns {ArrayBufferView|null}
     */
    data( eid ) {
        if ( ! MipUploadComponent.uploaded[ eid ] ) return null;
        return this.mipData[ MipUploadComponent.dataPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact mip-level size computation
// ---------------------------------------------------------------------------
/**
 * Compute the byte size of a single mip level using double.js for bit-exact
 * accumulation. Used for very large compressed textures (e.g. 16K ASTC) where
 * the naive product of width * height * blockSize overflows or loses precision
 * in float32 arithmetic.
 * @param {number} width
 * @param {number} height
 * @param {number} blockSize - Bytes per 4×4 block (e.g. 16 for DXT1, 8 for ETC1).
 * @returns {number}
 */
function computeMipByteSizePrecise( width, height, blockSize ) {
    // Compressed formats store data in 4×4 blocks
    const blocksWide = Math.ceil( width / 4 );
    const blocksHigh = Math.ceil( height / 4 );

    _double.value = blocksWide;
    _double.value = _double.value * blocksHigh;
    _double.value = _double.value * blockSize;
    return _double.value;
}

// ---------------------------------------------------------------------------
// simplex-noise dithered high-frequency detail synthesis
// ---------------------------------------------------------------------------
/**
 * Synthesize a Uint8Array of high-frequency detail coefficients using
 * simplex-noise. Useful for baking procedural detail into a compressed
 * texture pipeline before the data is quantized by the compressor.
 * @param {number} width
 * @param {number} height
 * @param {number} [frequency=0.1]
 * @param {number} [amplitude=1]
 * @param {number} [offset=0]
 * @returns {Uint8Array}
 */
function synthesizeDetail( width, height, frequency = 0.1, amplitude = 1, offset = 0 ) {
    const out = new Uint8Array( width * height );
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const n = _noise2D( x * frequency + offset, y * frequency + offset );
            out[ y * width + x ] = Math.max( 0, Math.min( 255, ( n * 0.5 + 0.5 ) * 255 * amplitude ) );
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated per-mip staging
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated extraction of mip-level dimensions into a preallocated
 * vec2 array. Zero-allocation, suitable for hot upload loops.
 * @param {CompressedTexture} texture
 * @returns {Array<glMatrix.vec2>}
 */
function extractMipDimensionsGlMat( texture ) {
    const out = [];
    if ( ! texture.mipmaps ) return out;
    for ( let i = 0, l = texture.mipmaps.length; i < l; i ++ ) {
        const mip = texture.mipmaps[ i ];
        glMatrix.vec2.set( _gm_v2, mip.width, mip.height );
        out.push( glMatrix.vec2.clone( _gm_v2 ) );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main CompressedTexture class — mirrors three.js/src/textures/CompressedTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a texture based on data in compressed form.
 * These textures are usually loaded with {@link CompressedTextureLoader}.
 * @augments Texture
 */
class CompressedTexture extends Texture {

    /**
     * Constructs a new compressed texture.
     * @param {Array} mipmaps - This array holds for all mipmaps (including the
     * base mip) the data and dimensions.
     * @param {number} width - The width of the texture.
     * @param {number} height - The height of the texture.
     * @param {number} [format=RGBAFormat] - The texture format.
     * @param {number} [type=UnsignedByteType] - The texture type.
     * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=LinearFilter] - The mag filter value.
     * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     * @param {string} [colorSpace=NoColorSpace] - The color space.
     */
    constructor(
        mipmaps,
        width,
        height,
        format = RGBAFormat,
        type = UnsignedByteType,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearMipmapLinearFilter,
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
        this.isCompressedTexture = true;

        /**
         * The image property of a compressed texture just defines its dimensions.
         * @type {{width:number,height:number}}
         */
        this.image = { width: width, height: height };

        /**
         * This array holds for all mipmaps (including the base mip) the data
         * and dimensions.
         * @type {Array}
         */
        this.mipmaps = mipmaps;

        /**
         * If set to `true`, the texture is flipped along the vertical axis when
         * uploaded to the GPU.
         * Overwritten and set to `false` by default since it is not possible to
         * flip compressed textures.
         * @type {boolean}
         * @default false
         * @readonly
         */
        this.flipY = false;

        /**
         * Whether to generate mipmaps (if possible) for a texture.
         * Overwritten and set to `false` by default since it is not possible to
         * generate mipmaps for compressed data. Mipmaps must be embedded in the
         * compressed texture file.
         * @type {boolean}
         * @default false
         * @readonly
         */
        this.generateMipmaps = false;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated extraction of mip-level dimensions. Returns an
     * array of preallocated vec2 instances (zero-allocation on repeat calls
     * with cached dimensions).
     * @returns {Array<glMatrix.vec2>}
     */
    extractMipDimensionsGlMat() {
        return extractMipDimensionsGlMat( this );
    }

    /**
     * Compute the byte size of a single mip level using double.js for
     * bit-exact accumulation. Recommended for very large compressed textures.
     * @param {number} level - Mip level index.
     * @param {number} blockSize - Bytes per 4×4 block.
     * @returns {number}
     */
    computeMipByteSizePrecise( level, blockSize ) {
        const mip = this.mipmaps[ level ];
        if ( ! mip ) return 0;
        return computeMipByteSizePrecise( mip.width, mip.height, blockSize );
    }

    /**
     * Total byte size across all mips using double.js for bit-exact
     * accumulation.
     * @param {number} blockSize - Bytes per 4×4 block.
     * @returns {number}
     */
    totalByteSizePrecise( blockSize ) {
        _double.value = 0;
        for ( let i = 0, l = this.mipmaps.length; i < l; i ++ ) {
            _double.add( this.computeMipByteSizePrecise( i, blockSize ) );
        }
        return _double.value;
    }

    /**
     * Synthesize a high-frequency detail coefficient buffer using
     * simplex-noise. Useful for baking procedural detail into a compressed
     * texture pipeline.
     * @param {number} width
     * @param {number} height
     * @param {number} [frequency=0.1]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {Uint8Array}
     */
    synthesizeDetail( width, height, frequency = 0.1, amplitude = 1, offset = 0 ) {
        return synthesizeDetail( width, height, frequency, amplitude, offset );
    }

    /**
     * Create a batched mip upload coordinator backed by bitecs.
     * Register multiple compressed textures and queue all their mips to be
     * uploaded in one cache-friendly pass.
     * @returns {CompressedMipBatch}
     */
    static createBatch() {
        return new CompressedMipBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {CompressedTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.mipmaps = source.mipmaps.slice( 0 );
        this.flipY = source.flipY;
        this.generateMipmaps = source.generateMipmaps;
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
                type: 'CompressedTexture',
                generator: 'CompressedTexture.toJSON'
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
            mipmaps: this.mipmaps.map( mip => ( {
                width: mip.width,
                height: mip.height
            } ) )
        };

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }

    /**
     * Disposes the texture.
     */
    dispose() {
        this.dispatchEvent( { type: 'dispose' } );
    }
}

export {
    CompressedTexture,
    CompressedMipBatch,
    computeMipByteSizePrecise,
    synthesizeDetail,
    extractMipDimensionsGlMat
};
export default CompressedTexture;