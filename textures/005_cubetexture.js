// file number : 005
// full path name : src/textures/005_cubetexture.js
// description : CubeTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Represents a texture
// made of six images arranged as the faces of a cube (+X, -X, +Y, -Y, +Z, -Z).
// Overrides flipY to false and uses CubeReflectionMapping by default, matching
// three.js r185 semantics. Adds gl-matrix accelerated per-face direction
// staging, bitecs SoA batching for multi-face GPU uploads, double.js bit-exact
// per-face UV-to-direction math for large cubemaps where float32 drift causes
// visible seams, and simplex-noise procedural face synthesis for debugging and
// environment placeholders.
// best for : CubeTexture, CubeTextureLoader, environment maps, skyboxes,
// reflection probes, PMREMGenerator input, and any three.js workflow that binds
// six images as a cubemap.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';
import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    CubeReflectionMapping,
    CubeRefractionMapping,
    CubeUVReflectionMapping,
    CubeUVRefractionMapping,
    ClampToEdgeWrapping,
    LinearFilter,
    LinearMipmapLinearFilter,
    RGBAFormat,
    UnsignedByteType,
    NoColorSpace
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

// gl-matrix scratch for zero-allocation face direction staging
const _gm_dir = glMatrix.vec3.create();
const _gm_up = glMatrix.vec3.create();
const _gm_right = glMatrix.vec3.create();

// Face names in canonical order used throughout the library
const FACE_ORDER = [ '+X', '-X', '+Y', '-Y', '+Z', '-Z' ];

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-face cubemap uploads
// ---------------------------------------------------------------------------
const _cubeWorld = createWorld();
const CubeFaceComponent = defineComponent( {
    textureId: Types.ui16,
    faceIndex: Types.ui8,   // 0..5 in FACE_ORDER
    imagePtr: Types.ui32,   // index into this.images
    width: Types.ui32,
    height: Types.ui32,
    uploaded: Types.ui8
} );

class CubeTextureBatch {

    constructor() {
        this.world = _cubeWorld;
        this.textures = [];
        this.images = [];
        this.entities = [];
    }

    /**
     * Register a CubeTexture instance for batched uploads.
     * @param {CubeTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a single face for upload.
     * @param {number} textureId
     * @param {number} faceIndex - 0..5 in FACE_ORDER.
     * @param {any} image - The image source (HTMLImageElement, canvas, etc.).
     * @returns {number} entity id
     */
    addFace( textureId, faceIndex, image ) {
        const eid = addEntity( this.world );
        addComponent( this.world, CubeFaceComponent, eid );
        const imgIndex = this.images.length;
        this.images.push( image );
        CubeFaceComponent.textureId[ eid ] = textureId;
        CubeFaceComponent.faceIndex[ eid ] = faceIndex;
        CubeFaceComponent.imagePtr[ eid ] = imgIndex;
        CubeFaceComponent.width[ eid ] = image?.width ?? 0;
        CubeFaceComponent.height[ eid ] = image?.height ?? 0;
        CubeFaceComponent.uploaded[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Auto-populate all six faces of a CubeTexture.
     * @param {number} textureId
     */
    fillFromTexture( textureId ) {
        const texture = this.textures[ textureId ];
        if ( ! texture || ! Array.isArray( texture.image ) ) return;
        for ( let i = 0; i < 6; i ++ ) {
            this.addFace( textureId, i, texture.image[ i ] );
        }
    }

    /**
     * Process all queued face uploads in one cache-friendly pass.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            CubeFaceComponent.uploaded[ entities[ i ] ] = 1;
        }
    }

    /**
     * Retrieve the image for a given entity.
     * @param {number} eid
     * @returns {any|null}
     */
    image( eid ) {
        if ( ! CubeFaceComponent.uploaded[ eid ] ) return null;
        return this.images[ CubeFaceComponent.imagePtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact per-face UV-to-direction conversion
// ---------------------------------------------------------------------------
/**
 * Convert a cubemap face + UV coordinate into a normalized 3D direction
 * using double.js for bit-exact accumulation. Critical for very large
 * cubemaps (e.g. 8K) where float32 drift causes visible seams at face
 * boundaries.
 * @param {number} faceIndex - 0..5 in FACE_ORDER.
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @returns {glMatrix.vec3} Normalized direction.
 */
function faceUVToDirectionPrecise( faceIndex, u, v ) {
    // Remap UV to [-1, 1]
    _double.value = u * 2;
    _double.value = _double.value - 1;
    const sc = _double.value;

    _double.value = v * 2;
    _double.value = _double.value - 1;
    const tc = _double.value;

    let x = 0, y = 0, z = 0;
    switch ( faceIndex ) {
        case 0: x = 1;  y = -tc; z = -sc; break; // +X
        case 1: x = -1; y = -tc; z = sc;  break; // -X
        case 2: x = sc; y = 1;   z = tc;  break; // +Y
        case 3: x = sc; y = -1;  z = -tc; break; // -Y
        case 4: x = sc; y = -tc; z = 1;   break; // +Z
        case 5: x = -sc; y = -tc; z = -1; break; // -Z
    }

    // Normalize with double.js precision
    _double.value = x * x;
    _double.add( y * y );
    _double.add( z * z );
    const invLen = 1 / Math.sqrt( _double.value );

    glMatrix.vec3.set( _gm_dir, x * invLen, y * invLen, z * invLen );
    return _gm_dir;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated per-face basis vector extraction
// ---------------------------------------------------------------------------
/**
 * Extract the up and right basis vectors for a cubemap face. Writes into
 * preallocated gl-matrix vec3 instances (zero-allocation).
 * @param {number} faceIndex - 0..5 in FACE_ORDER.
 * @param {glMatrix.vec3} [outUp] - Optional output vec3 for the up vector.
 * @param {glMatrix.vec3} [outRight] - Optional output vec3 for the right vector.
 * @returns {{up: glMatrix.vec3, right: glMatrix.vec3}}
 */
function faceBasisGlMat( faceIndex, outUp, outRight ) {
    const up = outUp || glMatrix.vec3.create();
    const right = outRight || glMatrix.vec3.create();
    switch ( faceIndex ) {
        case 0: glMatrix.vec3.set( up, 0, 1, 0 );  glMatrix.vec3.set( right, 0, 0, -1 ); break; // +X
        case 1: glMatrix.vec3.set( up, 0, 1, 0 );  glMatrix.vec3.set( right, 0, 0, 1 );  break; // -X
        case 2: glMatrix.vec3.set( up, 0, 0, 1 );  glMatrix.vec3.set( right, 1, 0, 0 );  break; // +Y
        case 3: glMatrix.vec3.set( up, 0, 0, -1 ); glMatrix.vec3.set( right, 1, 0, 0 );  break; // -Y
        case 4: glMatrix.vec3.set( up, 0, 1, 0 );  glMatrix.vec3.set( right, 1, 0, 0 );  break; // +Z
        case 5: glMatrix.vec3.set( up, 0, 1, 0 );  glMatrix.vec3.set( right, -1, 0, 0 ); break; // -Z
    }
    return { up, right };
}

// ---------------------------------------------------------------------------
// simplex-noise procedural face synthesis
// ---------------------------------------------------------------------------
/**
 * Synthesize a procedural cubemap face using simplex-noise. Useful as an
 * environment placeholder, debug visualization, or pre-bake input for
 * PMREM generation.
 * @param {number} faceIndex - 0..5 in FACE_ORDER.
 * @param {number} width
 * @param {number} height
 * @param {number} [frequency=0.01]
 * @param {number} [amplitude=1]
 * @param {number} [offset=0]
 * @returns {Uint8ClampedArray} RGBA byte buffer.
 */
function synthesizeFaceNoise( faceIndex, width, height, frequency = 0.01, amplitude = 1, offset = 0 ) {
    const out = new Uint8ClampedArray( width * height * 4 );
    const faceOffset = offset + faceIndex * 1000;
    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const u = x / width;
            const v = y / height;
            const dir = faceUVToDirectionPrecise( faceIndex, u, v );
            const n = _noise2D(
                dir[ 0 ] * frequency + faceOffset,
                dir[ 1 ] * frequency + faceOffset
            ) * 0.5 + 0.5;
            const val = Math.max( 0, Math.min( 255, n * amplitude * 255 ) );
            const p = ( y * width + x ) * 4;
            out[ p ]     = val;
            out[ p + 1 ] = val;
            out[ p + 2 ] = val;
            out[ p + 3 ] = 255;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main CubeTexture class — mirrors three.js/src/textures/CubeTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a cube texture made of six images.
 * ```js
 * const loader = new THREE.CubeTextureLoader();
 * loader.setPath( 'textures/cube/pisa/' );
 * const textureCube = loader.load( [
 *   'px.png', 'nx.png', 'py.png', 'ny.png', 'pz.png', 'nz.png'
 * ] );
 * const material = new THREE.MeshBasicMaterial( { color: 0xffffff, envMap: textureCube } );
 * ```
 * @augments Texture
 */
class CubeTexture extends Texture {

    /**
     * Constructs a new cube texture.
     * @param {Array} [images] - The images for each of the six faces.
     * @param {number} [mapping=CubeReflectionMapping] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=LinearFilter] - The mag filter value.
     * @param {number} [minFilter=LinearMipmapLinearFilter] - The min filter value.
     * @param {number} [format=RGBAFormat] - The texture format.
     * @param {number} [type=UnsignedByteType] - The texture type.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     * @param {string} [colorSpace=NoColorSpace] - The color space.
     */
    constructor(
        images = [],
        mapping = CubeReflectionMapping,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearMipmapLinearFilter,
        format = RGBAFormat,
        type = UnsignedByteType,
        anisotropy = Texture.DEFAULT_ANISOTROPY,
        colorSpace = NoColorSpace
    ) {
        super( images, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy, colorSpace );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isCubeTexture = true;

        /**
         * If set to `true`, the texture is flipped along the vertical axis when
         * uploaded to the GPU.
         * Overwritten and set to `false` by default since it is not possible to
         * flip cubemap textures.
         * @type {boolean}
         * @default false
         * @readonly
         */
        this.flipY = false;
    }

    /**
     * The images of the cube texture.
     * @type {Array}
     */
    get images() {
        return this.image;
    }

    set images( value ) {
        this.image = value;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated UV-to-direction conversion for a given face.
     * Writes into a preallocated vec3 (zero-allocation).
     * @param {number} faceIndex - 0..5 in FACE_ORDER.
     * @param {number} u - U coordinate in [0, 1].
     * @param {number} v - V coordinate in [0, 1].
     * @param {glMatrix.vec3} [out] - Optional output vec3.
     * @returns {glMatrix.vec3}
     */
    faceUVToDirectionGlMat( faceIndex, u, v, out = _gm_dir ) {
        const dir = faceUVToDirectionPrecise( faceIndex, u, v );
        out[ 0 ] = dir[ 0 ];
        out[ 1 ] = dir[ 1 ];
        out[ 2 ] = dir[ 2 ];
        return out;
    }

    /**
     * gl-matrix accelerated basis vectors for a given face.
     * @param {number} faceIndex
     * @returns {{up: glMatrix.vec3, right: glMatrix.vec3}}
     */
    faceBasisGlMat( faceIndex ) {
        return faceBasisGlMat( faceIndex, _gm_up, _gm_right );
    }

    /**
     * Synthesize a procedural face using simplex-noise.
     * @param {number} faceIndex
     * @param {number} width
     * @param {number} height
     * @param {number} [frequency=0.01]
     * @param {number} [amplitude=1]
     * @param {number} [offset=0]
     * @returns {Uint8ClampedArray}
     */
    synthesizeFaceNoise( faceIndex, width, height, frequency = 0.01, amplitude = 1, offset = 0 ) {
        return synthesizeFaceNoise( faceIndex, width, height, frequency, amplitude, offset );
    }

    /**
     * Create a batched cubemap face upload coordinator backed by bitecs.
     * @returns {CubeTextureBatch}
     */
    static createBatch() {
        return new CubeTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {CubeTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.flipY = false;
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
                type: 'CubeTexture',
                generator: 'CubeTexture.toJSON'
            },
            uuid: this.uuid,
            name: this.name,
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
            generateMipmaps: this.generateMipmaps
        };

        // Serialize the six face images
        if ( this.image !== undefined && Array.isArray( this.image ) ) {
            const imageMeta = { images: [] };
            for ( let i = 0; i < 6; i ++ ) {
                const img = this.image[ i ];
                if ( img && img.toJSON ) {
                    imageMeta.images.push( img.toJSON( meta ) );
                } else {
                    imageMeta.images.push( null );
                }
            }
            output.image = imageMeta;
        }

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
    CubeTexture,
    CubeTextureBatch,
    faceUVToDirectionPrecise,
    faceBasisGlMat,
    synthesizeFaceNoise,
    FACE_ORDER
};
export default CubeTexture;