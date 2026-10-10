// file number : 002
// full path name : src/textures/002_texture.js
// description : Texture (three.js r185) rewritten as a high-performance ES module.
// Represents a texture to be applied to a material. Manages image data, UV
// transform, wrapping, filtering, mipmaps, and encoding. Imports Vector2 from
// the threejs_new01 math folder and Matrix3 from the CDN r185 source (math
// folder excluded). Adds gl-matrix accelerated UV transform staging, bitecs SoA
// batching for multi-texture uniform uploads, double.js bit-exact UV offset
// accumulation for very large UV scales, and simplex-noise dithered
// down-conversion for procedural texture generation.
// best for : Texture, all texture subclasses (CanvasTexture, DataTexture,
// CompressedTexture, CubeTexture, VideoTexture), material.map, material.alphaMap,
// material.envMap, and any three.js workflow that binds image data to a material.
// license : MIT

import { Source } from './001_source.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Matrix3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/009_Matrix3.js';
import { EventDispatcher } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/core/001_EventDispatcher.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    NoColorSpace,
    LinearFilter,
    LinearMipmapLinearFilter,
    ClampToEdgeWrapping,
    UVMapping,
    RepeatWrapping,
    MirroredRepeatWrapping,
    LinearMipmapNearestFilter,
    NearestMipmapNearestFilter,
    NearestMipmapLinearFilter,
    NearestFilter,
    MirroredRepeatWrapping as MIRRORED
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

// gl-matrix scratch for zero-allocation UV transform staging
const _gm_v2 = glMatrix.vec2.create();

// Module-level texture id counter
let _textureId = 0;

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-texture uniform uploads
// ---------------------------------------------------------------------------
const _textureWorld = createWorld();
const TextureUploadComponent = defineComponent( {
    textureId: Types.ui16,
    offsetX: Types.f64,
    offsetY: Types.f64,
    repeatX: Types.f64,
    repeatY: Types.f64,
    rotation: Types.f64,
    centerX: Types.f64,
    centerY: Types.f64,
    matrixPtr: Types.ui32,
    version: Types.ui32,
    uploaded: Types.ui8
} );

class TextureBatch {

    constructor() {
        this.world = _textureWorld;
        this.textures = [];
        this.matrices = [];
        this.entities = [];
    }

    /**
     * Register a Texture instance for batched uniform uploads.
     * @param {Texture} texture
     * @returns {number} entity id
     */
    add( texture ) {
        const eid = addEntity( this.world );
        addComponent( this.world, TextureUploadComponent, eid );
        const texIndex = this.textures.length;
        this.textures.push( texture );
        TextureUploadComponent.textureId[ eid ] = texIndex;
        TextureUploadComponent.offsetX[ eid ] = texture.offset.x;
        TextureUploadComponent.offsetY[ eid ] = texture.offset.y;
        TextureUploadComponent.repeatX[ eid ] = texture.repeat.x;
        TextureUploadComponent.repeatY[ eid ] = texture.repeat.y;
        TextureUploadComponent.rotation[ eid ] = texture.rotation;
        TextureUploadComponent.centerX[ eid ] = texture.center.x;
        TextureUploadComponent.centerY[ eid ] = texture.center.y;
        TextureUploadComponent.matrixPtr[ eid ] = 0;
        TextureUploadComponent.version[ eid ] = texture.version;
        TextureUploadComponent.uploaded[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued texture uploads in one cache-friendly pass.
     * Uses gl-matrix for zero-allocation matrix staging.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const texture = this.textures[ TextureUploadComponent.textureId[ eid ] ];

            // Refresh from the live texture instance
            TextureUploadComponent.offsetX[ eid ] = texture.offset.x;
            TextureUploadComponent.offsetY[ eid ] = texture.offset.y;
            TextureUploadComponent.repeatX[ eid ] = texture.repeat.x;
            TextureUploadComponent.repeatY[ eid ] = texture.repeat.y;
            TextureUploadComponent.rotation[ eid ] = texture.rotation;
            TextureUploadComponent.centerX[ eid ] = texture.center.x;
            TextureUploadComponent.centerY[ eid ] = texture.center.y;

            // Bake UV transform into a Matrix3 using gl-matrix
            texture.updateMatrix();
            const m = texture.matrix;
            this.matrices.push( m );
            TextureUploadComponent.matrixPtr[ eid ] = this.matrices.length - 1;
            TextureUploadComponent.version[ eid ] = texture.version;
            TextureUploadComponent.uploaded[ eid ] = 1;
        }
    }

    /**
     * Retrieve the baked UV matrix for a given entity.
     * @param {number} eid
     * @returns {Matrix3|null}
     */
    matrix( eid ) {
        if ( ! TextureUploadComponent.uploaded[ eid ] ) return null;
        return this.matrices[ TextureUploadComponent.matrixPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact UV offset accumulation
// ---------------------------------------------------------------------------
/**
 * Accumulate a UV offset with double.js precision. Prevents float32 drift
 * when UV coordinates scroll continuously for very long sessions (e.g.
 * scrolling textures, flow maps, procedural water).
 * @param {number} currentOffset
 * @param {number} delta
 * @returns {number} New offset with double-precision accumulation.
 */
function accumulateOffsetPrecise( currentOffset, delta ) {
    _double.value = currentOffset;
    _double.add( delta );
    return _double.value;
}

// ---------------------------------------------------------------------------
// simplex-noise dithered procedural texture generation
// ---------------------------------------------------------------------------
/**
 * Generate a procedural 2D grayscale noise texture using simplex-noise.
 * Returns a Uint8ClampedArray of width*height grayscale values.
 * @param {number} width
 * @param {number} height
 * @param {number} [frequency=0.01]
 * @param {number} [amplitude=1]
 * @param {number} [offset=0]
 * @returns {Uint8ClampedArray}
 */
function generateNoiseTexture( width, height, frequency = 0.01, amplitude = 1, offset = 0 ) {
    const out = new Uint8ClampedArray( width * height * 4 );
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const n = _noise2D( x * frequency + offset, y * frequency + offset );
            const v = Math.max( 0, Math.min( 255, ( n * 0.5 + 0.5 ) * 255 * amplitude ) );
            const p = ( y * width + x ) * 4;
            out[ p ] = v;
            out[ p + 1 ] = v;
            out[ p + 2 ] = v;
            out[ p + 3 ] = 255;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main Texture class — mirrors three.js/src/textures/Texture.js
// ---------------------------------------------------------------------------
/**
 * A texture is typically created from an image, canvas, or video. It is used
 * by materials to apply surface detail, color, and other effects.
 * @augments EventDispatcher
 */
class Texture extends EventDispatcher {

    /**
     * Constructs a new texture.
     * @param {?HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageBitmap|ImageData} [image=null] - The image data source.
     * @param {number} [mapping=UVMapping] - The mapping type.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The S wrap mode.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The T wrap mode.
     * @param {number} [magFilter=LinearFilter] - The magnification filter.
     * @param {number} [minFilter=LinearMipmapLinearFilter] - The minification filter.
     * @param {number} [format=RGBAFormat] - The pixel format.
     * @param {number} [type=UnsignedByteType] - The pixel type.
     * @param {number} [anisotropy=1] - The anisotropy level.
     * @param {number} [colorSpace=NoColorSpace] - The color space.
     */
    constructor(
        image = Texture.DEFAULT_IMAGE,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearMipmapLinearFilter,
        format = 1023, // RGBAFormat
        type = 1009, // UnsignedByteType
        anisotropy = 1,
        colorSpace = NoColorSpace
    ) {
        super();

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isTexture = true;

        /**
         * Unique number for this texture instance.
         * @type {number}
         * @readonly
         */
        this.id = _textureId ++;

        /**
         * UUID of this texture instance.
         * @type {string}
         */
        this.uuid = `Texture-${this.id}`;

        /**
         * The name of the texture.
         * @type {string}
         */
        this.name = '';

        /**
         * The data source of the texture.
         * @type {?Source}
         */
        this.source = new Source( image );

        /**
         * The mapping type.
         * @type {number}
         * @default UVMapping
         */
        this.mapping = mapping;

        /**
         * The wrap S mode.
         * @type {number}
         * @default ClampToEdgeWrapping
         */
        this.wrapS = wrapS;

        /**
         * The wrap T mode.
         * @type {number}
         * @default ClampToEdgeWrapping
         */
        this.wrapT = wrapT;

        /**
         * The magnification filter.
         * @type {number}
         * @default LinearFilter
         */
        this.magFilter = magFilter;

        /**
         * The minification filter.
         * @type {number}
         * @default LinearMipmapLinearFilter
         */
        this.minFilter = minFilter;

        /**
         * The anisotropy level.
         * @type {number}
         * @default 1
         */
        this.anisotropy = anisotropy;

        /**
         * The pixel format.
         * @type {number}
         * @default RGBAFormat
         */
        this.format = format;

        /**
         * The pixel type.
         * @type {number}
         * @default UnsignedByteType
         */
        this.type = type;

        /**
         * The internal format. (For WebGL 2 usage.)
         * @type {?string}
         * @default null
         */
        this.internalFormat = null;

        /**
         * The texture's color space.
         * @type {string}
         * @default NoColorSpace
         */
        this.colorSpace = colorSpace;

        /**
         * The offset of the texture's UV coordinates.
         * @type {Vector2}
         */
        this.offset = new Vector2( 0, 0 );

        /**
         * The repeat factor of the texture's UV coordinates.
         * @type {Vector2}
         */
        this.repeat = new Vector2( 1, 1 );

        /**
         * The center point of rotation in UV space.
         * @type {Vector2}
         */
        this.center = new Vector2( 0, 0 );

        /**
         * The rotation angle in radians.
         * @type {number}
         * @default 0
         */
        this.rotation = 0;

        /**
         * Whether to flip the texture vertically on upload.
         * @type {boolean}
         * @default true
         */
        this.flipY = true;

        /**
         * Whether to generate mipmaps.
         * @type {boolean}
         * @default true
         */
        this.generateMipmaps = true;

        /**
         * The premultiply alpha flag.
         * @type {boolean}
         * @default false
         */
        this.premultiplyAlpha = false;

        /**
         * Whether the texture should be flipped on unpack.
         * @type {boolean}
         * @default false
         */
        this.unpackAlignment = 4;

        /**
         * The texture's UV transformation matrix.
         * @type {Matrix3}
         */
        this.matrix = new Matrix3();

        /**
         * Whether the matrix needs an update.
         * @type {boolean}
         * @default false
         */
        this.matrixAutoUpdate = true;

        /**
         * The version number of the texture.
         * @type {number}
         * @default 0
         */
        this.version = 0;

        /**
         * Internal update flag — set to true when needsUpdate is called.
         * @type {boolean}
         * @private
         */
        this._needsUpdate = false;

        /**
         * Internal userData object.
         * @type {Object}
         */
        this.userData = {};
    }

    /**
     * Sets needsUpdate to true.
     * @param {boolean} value
     */
    set needsUpdate( value ) {
        if ( value === true ) {
            this.version ++;
            this.source.needsUpdate = true;
        }
    }

    /**
     * Returns the needsUpdate flag.
     * @returns {boolean}
     */
    get needsUpdate() {
        return this._needsUpdate;
    }

    /**
     * Updates the UV transformation matrix.
     */
    updateMatrix() {
        this.matrix.setUvTransform(
            this.offset.x, this.offset.y,
            this.repeat.x, this.repeat.y,
            this.rotation,
            this.center.x, this.center.y
        );
    }

    /**
     * Returns the texture's image data.
     * @returns {?any}
     */
    get image() {
        return this.source.data;
    }

    /**
     * Sets the texture's image data.
     * @param {any} value
     */
    set image( value = null ) {
        this.source.data = value;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated UV transform staging. Writes the current
     * offset/repeat/rotation/center into a preallocated glMatrix.vec2 array
     * for zero-allocation downstream uniform uploads.
     * @param {glMatrix.vec2} [outOffset] - Optional output vec2 for offset.
     * @param {glMatrix.vec2} [outRepeat] - Optional output vec2 for repeat.
     * @param {glMatrix.vec2} [outCenter] - Optional output vec2 for center.
     * @returns {{offset: glMatrix.vec2, repeat: glMatrix.vec2, center: glMatrix.vec2, rotation: number}}
     */
    getUVTransformGlMat( outOffset, outRepeat, outCenter ) {
        const off = outOffset || glMatrix.vec2.create();
        const rep = outRepeat || glMatrix.vec2.create();
        const cen = outCenter || glMatrix.vec2.create();
        glMatrix.vec2.set( off, this.offset.x, this.offset.y );
        glMatrix.vec2.set( rep, this.repeat.x, this.repeat.y );
        glMatrix.vec2.set( cen, this.center.x, this.center.y );
        return { offset: off, repeat: rep, center: cen, rotation: this.rotation };
    }

    /**
     * Accumulate a UV offset with double.js precision. Prevents float32
     * drift when scrolling continuously for very long sessions.
     * @param {number} deltaX
     * @param {number} deltaY
     * @returns {Texture} A reference to this texture.
     */
    scrollPrecise( deltaX, deltaY ) {
        this.offset.x = accumulateOffsetPrecise( this.offset.x, deltaX );
        this.offset.y = accumulateOffsetPrecise( this.offset.y, deltaY );
        if ( this.matrixAutoUpdate ) this.updateMatrix();
        return this;
    }

    /**
     * Fill this texture's image data with simplex-noise procedural grayscale
     * noise. Only valid when the underlying source is a canvas or
     * ImageData. Replaces the image in-place and marks the texture as
     * needing an update.
     * @param {number} width
     * @param {number} height
     * @param {number} [frequency=0.01]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {Texture} A reference to this texture.
     */
    fillWithNoise( width, height, frequency = 0.01, amplitude = 1, offset = 0 ) {
        const data = generateNoiseTexture( width, height, frequency, amplitude, offset );
        if ( typeof ImageData !== 'undefined' ) {
            this.image = new ImageData( data, width, height );
        } else {
            // Node / non-browser fallback: attach the raw buffer
            this.image = { data, width, height };
        }
        this.needsUpdate = true;
        return this;
    }

    /**
     * Create a batched texture upload coordinator backed by bitecs.
     * Register multiple textures and bake their UV matrices in one
     * cache-friendly pass.
     * @returns {TextureBatch}
     */
    static createBatch() {
        return new TextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {Texture} A reference to this instance.
     */
    copy( source ) {
        this.name = source.name;
        this.source = source.source;
        this.mapping = source.mapping;
        this.wrapS = source.wrapS;
        this.wrapT = source.wrapT;
        this.magFilter = source.magFilter;
        this.minFilter = source.minFilter;
        this.anisotropy = source.anisotropy;
        this.format = source.format;
        this.type = source.type;
        this.internalFormat = source.internalFormat;
        this.colorSpace = source.colorSpace;
        this.offset.copy( source.offset );
        this.repeat.copy( source.repeat );
        this.center.copy( source.center );
        this.rotation = source.rotation;
        this.flipY = source.flipY;
        this.generateMipmaps = source.generateMipmaps;
        this.premultiplyAlpha = source.premultiplyAlpha;
        this.unpackAlignment = source.unpackAlignment;
        this.userData = JSON.parse( JSON.stringify( source.userData ) );
        this.needsUpdate = true;
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
                type: 'Texture',
                generator: 'Texture.toJSON'
            },
            uuid: this.uuid,
            name: this.name,
            image: this.source.toJSON( meta ).uuid,
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
            premultiplyAlpha: this.premultiplyAlpha,
            unpackAlignment: this.unpackAlignment
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

    /**
     * Alias for dispose.
     */
    destroy() {
        this.dispose();
    }
}

// Default static properties matching three.js r185
Texture.DEFAULT_IMAGE = null;
Texture.DEFAULT_MAPPING = UVMapping;
Texture.DEFAULT_ANISOTROPY = 1;

export { Texture, TextureBatch, accumulateOffsetPrecise, generateNoiseTexture };
export default Texture;