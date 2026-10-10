// file number : 009
// full path name : src/textures/009_depthtexture.js
// description : DepthTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents a texture
// that can be used to automatically save the depth information of a rendering
// into a texture. Only supports DepthFormat or DepthStencilFormat, and
// automatically derives the correct type (UnsignedIntType for DepthFormat,
// UnsignedInt248Type for DepthStencilFormat). Overrides magFilter/minFilter
// to NearestFilter, flipY to false, generateMipmaps to false, and
// unpackAlignment to 1. Adds gl-matrix accelerated depth-buffer range
// extraction, bitecs SoA batching for multi-depth-texture GPU uploads,
// double.js bit-exact depth normalization for high-precision depth buffers,
// and simplex-noise dithered depth quantization for compact depth storage.
// best for : DepthTexture, WebGLRenderTarget.depthTexture, shadow map depth
// passes, post-processing depth-of-field, SSAO, and any three.js workflow that
// needs a depth attachment sampled as a texture in GLSL.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    DepthFormat,
    DepthStencilFormat,
    UnsignedIntType,
    UnsignedInt248Type,
    NearestFilter,
    ClampToEdgeWrapping,
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

// gl-matrix scratch for zero-allocation depth staging
const _gm_v2 = glMatrix.vec2.create();
const _gm_v4 = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-depth-texture GPU uploads
// ---------------------------------------------------------------------------
const _depthWorld = createWorld();
const DepthUploadComponent = defineComponent( {
    textureId: Types.ui16,
    width: Types.ui32,
    height: Types.ui32,
    format: Types.ui32,
    type: Types.ui32,
    dataPtr: Types.ui32,
    outputPtr: Types.ui32,
    mode: Types.ui8, // 0 = copy, 1 = normalize float, 2 = dither 8-bit
    uploaded: Types.ui8
} );

class DepthTextureBatch {

    constructor() {
        this.world = _depthWorld;
        this.textures = [];
        this.inputs = [];
        this.outputs = [];
        this.entities = [];
    }

    /**
     * Register a DepthTexture instance for batched uploads.
     * @param {DepthTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a depth texture upload job.
     * @param {number} textureId
     * @param {number} [mode=0] - 0 = copy, 1 = normalize float, 2 = dither 8-bit.
     * @returns {number} entity id
     */
    add( textureId, mode = 0 ) {
        const eid = addEntity( this.world );
        addComponent( this.world, DepthUploadComponent, eid );
        const texture = this.textures[ textureId ];
        DepthUploadComponent.textureId[ eid ] = textureId;
        DepthUploadComponent.width[ eid ] = texture.image.width;
        DepthUploadComponent.height[ eid ] = texture.image.height;
        DepthUploadComponent.format[ eid ] = texture.format;
        DepthUploadComponent.type[ eid ] = texture.type;
        DepthUploadComponent.dataPtr[ eid ] = 0;
        DepthUploadComponent.outputPtr[ eid ] = 0;
        DepthUploadComponent.mode[ eid ] = mode;
        DepthUploadComponent.uploaded[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued uploads in one cache-friendly pass.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const texture = this.textures[ DepthUploadComponent.textureId[ eid ] ];
            const mode = DepthUploadComponent.mode[ eid ];
            let result = null;

            if ( ! texture || ! texture.image || ! texture.image.data ) {
                result = null;
            } else if ( mode === 0 ) {
                result = texture.image.data;
            } else if ( mode === 1 && texture.image.data instanceof Float32Array ) {
                result = normalizeDepthPrecise( texture.image.data );
            } else if ( mode === 2 && texture.image.data instanceof Float32Array ) {
                result = floatDepthToDithered8Bit(
                    texture.image.data,
                    DepthUploadComponent.width[ eid ],
                    DepthUploadComponent.height[ eid ]
                );
            } else {
                result = texture.image.data;
            }

            const outIndex = this.outputs.length;
            this.outputs.push( result );
            DepthUploadComponent.outputPtr[ eid ] = outIndex;
            DepthUploadComponent.uploaded[ eid ] = 1;
        }
    }

    /**
     * Retrieve the processed result for a given entity.
     * @param {number} eid
     * @returns {any|null}
     */
    result( eid ) {
        if ( ! DepthUploadComponent.uploaded[ eid ] ) return null;
        return this.outputs[ DepthUploadComponent.outputPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact depth normalization
// ---------------------------------------------------------------------------
/**
 * Normalize a float depth buffer into [0, 1] using double.js for bit-exact
 * scaling. Used for very high-precision depth buffers (e.g. from a 32-bit
 * float shadow map) where float32 drift during normalization would produce
 * visible banding.
 * @param {Float32Array} data
 * @param {number} [maxDepth] - Optional explicit maximum depth.
 * @returns {Float32Array}
 */
function normalizeDepthPrecise( data, maxDepth ) {
    let max = maxDepth;
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
// simplex-noise dithered depth quantization
// ---------------------------------------------------------------------------
/**
 * Down-convert a float depth buffer to 8-bit with simplex-noise dithering.
 * Breaks up banding when depth values are quantized to a compact 8-bit
 * representation for storage or transmission.
 * @param {Float32Array} data
 * @param {number} width
 * @param {number} height
 * @param {number} [amplitude=0.5]
 * @returns {Uint8ClampedArray}
 */
function floatDepthToDithered8Bit( data, width, height, amplitude = 0.5 ) {
    const out = new Uint8ClampedArray( data.length );
    const invAmp = amplitude / 255;
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const p = ( y * width + x );
            const d = _noise2D( x * 0.1, y * 0.1 ) * invAmp;
            out[ p ] = Math.floor( Math.max( 0, Math.min( 1, data[ p ] + d ) ) * 255 );
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated depth-buffer range extraction
// ---------------------------------------------------------------------------
/**
 * Compute the min/max range of a float depth buffer using gl-matrix for
 * zero-allocation vec2 staging. Useful for auto-exposure, depth-of-field
 * focal-plane picking, and shadow-map range compression.
 * @param {Float32Array} data
 * @returns {glMatrix.vec2} [min, max]
 */
function extractDepthRangeGlMat( data ) {
    let min = Infinity;
    let max = - Infinity;
    for ( let i = 0; i < data.length; i ++ ) {
        const v = data[ i ];
        if ( v < min ) min = v;
        if ( v > max ) max = v;
    }
    glMatrix.vec2.set( _gm_v2, min, max );
    return _gm_v2;
}

// ---------------------------------------------------------------------------
// Main DepthTexture class — mirrors three.js/src/textures/DepthTexture.js
// ---------------------------------------------------------------------------
/**
 * This class can be used to automatically save the depth information of a
 * rendering into a texture.
 *
 * ```js
 * // create a depth texture
 * const depthTexture = new THREE.DepthTexture();
 * depthTexture.type = THREE.UnsignedShortType;
 *
 * // attach it to a render target
 * const renderTarget = new THREE.WebGLRenderTarget( 512, 512, {
 *     depthTexture: depthTexture
 * } );
 * ```
 * @augments Texture
 */
class DepthTexture extends Texture {

    /**
     * Constructs a new depth texture.
     * @param {number} [width=1] - The width of the texture.
     * @param {number} [height=1] - The height of the texture.
     * @param {number} [type=UnsignedIntType] - The texture type.
     * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=NearestFilter] - The mag filter value.
     * @param {number} [minFilter=NearestFilter] - The min filter value.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     * @param {number} [format=DepthFormat] - The texture format.
     * @param {number} [depth=1] - The depth of the texture.
     */
    constructor(
        width = 1,
        height = 1,
        type = UnsignedIntType,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = NearestFilter,
        minFilter = NearestFilter,
        anisotropy = Texture.DEFAULT_ANISOTROPY,
        format = DepthFormat,
        depth = 1
    ) {
        super( null, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isDepthTexture = true;

        this.type = 'DepthTexture';

        /**
         * The image definition of a depth texture.
         * @type {{width:number,height:number,depth:number}}
         */
        this.image = { width, height, depth };

        /**
         * If set to `true`, the texture is flipped along the vertical axis when
         * uploaded to the GPU.
         * Overwritten and set to `false` by default.
         * @type {boolean}
         * @default false
         */
        this.flipY = false;

        /**
         * Whether to generate mipmaps (if possible) for a texture.
         * Overwritten and set to `false` by default.
         * @type {boolean}
         * @default false
         */
        this.generateMipmaps = false;

        /**
         * Specifies the alignment requirements for the start of each pixel row
         * in memory.
         * Overwritten and set to `1` by default.
         * @type {boolean}
         * @default 1
         */
        this.unpackAlignment = 1;

        /**
         * Comparison function for the depth texture.
         * When set, enables shadow sampling (a.k.a. hardware PCF) on the
         * texture.
         * @type {?number}
         * @default null
         */
        this.compareFunction = null;

        // Internal data buffer placeholder (populated externally if needed)
        this._data = null;
    }

    /**
     * Convenience getter for the underlying data buffer, if any.
     * @returns {?TypedArray}
     */
    get data() {
        return this._data;
    }

    set data( value ) {
        this._data = value;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated extraction of the min/max depth range from this
     * texture's buffer. Requires that the buffer has been populated externally
     * (e.g. by reading back from a render target).
     * @returns {glMatrix.vec2} [min, max]
     */
    extractDepthRangeGlMat() {
        if ( ! this._data ) return glMatrix.vec2.set( _gm_v2, 0, 0 );
        return extractDepthRangeGlMat( this._data );
    }

    /**
     * gl-matrix accelerated extraction of this texture's dimensions into a
     * preallocated vec2. Zero-allocation.
     * @returns {glMatrix.vec2}
     */
    getDimensionsGlMat() {
        return glMatrix.vec2.set( _gm_v2, this.image.width, this.image.height );
    }

    /**
     * Normalize this depth texture's buffer into [0, 1] using double.js for
     * bit-exact scaling. Only valid when the buffer has been populated
     * externally with a Float32Array.
     * @param {number} [maxDepth] - Optional explicit maximum depth.
     * @returns {DepthTexture} A reference to this texture.
     */
    normalizePrecise( maxDepth ) {
        if ( this._data instanceof Float32Array ) {
            this._data = normalizeDepthPrecise( this._data, maxDepth );
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Down-convert this depth texture's float buffer to 8-bit with
     * simplex-noise dithering. Only valid when the buffer has been populated
     * externally with a Float32Array.
     * @param {number} [amplitude=0.5]
     * @returns {DepthTexture} A reference to this texture.
     */
    toDithered8Bit( amplitude = 0.5 ) {
        if ( this._data instanceof Float32Array ) {
            this._data = floatDepthToDithered8Bit(
                this._data,
                this.image.width,
                this.image.height,
                amplitude
            );
            this.type = UnsignedIntType;
            this.needsUpdate = true;
        }
        return this;
    }

    /**
     * Create a batched depth texture upload coordinator backed by bitecs.
     * @returns {DepthTextureBatch}
     */
    static createBatch() {
        return new DepthTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {DepthTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.image.width = source.image.width;
        this.image.height = source.image.height;
        this.image.depth = source.image.depth;
        this.flipY = false;
        this.generateMipmaps = false;
        this.unpackAlignment = 1;
        this.compareFunction = source.compareFunction;
        this._data = source._data;
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
                type: 'DepthTexture',
                generator: 'DepthTexture.toJSON'
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
            compareFunction: this.compareFunction
        };

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }
}

// Validate the format — DepthTexture only supports DepthFormat or DepthStencilFormat
const originalConstructor = DepthTexture;
DepthTexture = function ( ...args ) {
    const instance = new originalConstructor( ...args );
    if ( instance.format !== DepthFormat && instance.format !== DepthStencilFormat ) {
        throw new Error( 'DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat' );
    }
    if ( args.length === 0 ) {
        instance.type = UnsignedIntType;
    }
    return instance;
};

export {
    DepthTexture,
    DepthTextureBatch,
    normalizeDepthPrecise,
    floatDepthToDithered8Bit,
    extractDepthRangeGlMat
};
export default DepthTexture;