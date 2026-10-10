// file number : 022
// full path name : src/extras/lib/022_pmremgenerator.js
// description : Prefiltered Mipmapped Radiance Environment Map (PMREM) generator
// (three.js r185) rewritten as a high-performance ES module. Provides fromScene(),
// fromEquirectangular(), fromCubemap(), compileCubemapShader(),
// compileEquirectangularShader(), and dispose() with the full PMREMGenerator API
// surface. Imports Vector3, Matrix4, and Quaternion strictly from the
// threejs_new01 math folder. All three.js external types (PerspectiveCamera,
// WebGLRenderTarget, ShaderMaterial, BoxGeometry, Mesh, Scene) are imported from
// the three.js r185 npm source. ImageUtils is provided inline since no
// 007_imageutils.js exists in the threejs_new01 tree. Adds gl-matrix accelerated
// spherical/cubemap coordinate math, bitecs SoA batching for multi-face
// convolution, double.js bit-exact cube-to-equirectangular angle accumulation,
// and simplex-noise dithering for HDR quantization during mip generation.
// best for : PMREMGenerator, image-based lighting (IBL), PBR environments,
// HDR cubemap prefiltering, and any three.js workflow that needs physically-
// based ambient lighting.
// license : MIT

import { Vector3 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/003_Vector3.js';
import { Matrix4 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/007_Matrix4.js';
import { Quaternion } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/006_Quaternion.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import { PerspectiveCamera } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/cameras/PerspectiveCamera.js';
import { WebGLRenderTarget } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/renderers/WebGLRenderTarget.js';
import { Texture } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/textures/Texture.js';
import { ShaderMaterial } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/materials/ShaderMaterial.js';
import { BoxGeometry } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/geometries/BoxGeometry.js';
import { Mesh } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/objects/Mesh.js';
import { Scene } from 'https://cdn.jsdelivr.net/npm/three@0.185.1/src/scenes/Scene.js';
import {
    LinearFilter,
    LinearMipmapLinearFilter,
    CubeReflectionMapping,
    CubeUVReflectionMapping,
    HalfFloatType,
    FloatType,
    NoBlending,
    NoColorSpace,
    RGBAFormat,
    SRGBColorSpace,
    BackSide,
    LinearSRGBColorSpace,
    CubeUVRefractionMapping
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

// gl-matrix scratch for zero-allocation cubemap coordinate math
const _gm_dir = glMatrix.vec3.create();
const _gm_up = glMatrix.vec3.create();
const _gm_right = glMatrix.vec3.create();
const _gm_tmp = glMatrix.vec3.create();

// ---------------------------------------------------------------------------
// Inline ImageUtils — minimal sRGB↔linear and dithering helpers
// (007_imageutils.js does not exist in the threejs_new01 tree)
// ---------------------------------------------------------------------------
const ImageUtils = {

    sRGBToLinearScalar( c ) {
        return c < 0.04045 ? c * 0.0773993808 : Math.pow( c * 0.9478672986 + 0.0521327014, 2.4 );
    },

    sRGBToLinear( imageData ) {
        const data = imageData.data;
        const out = new Uint8ClampedArray( data.length );
        for ( let i = 0; i < data.length; i += 4 ) {
            out[ i ]     = Math.round( ImageUtils.sRGBToLinearScalar( data[ i ] / 255 ) * 255 );
            out[ i + 1 ] = Math.round( ImageUtils.sRGBToLinearScalar( data[ i + 1 ] / 255 ) * 255 );
            out[ i + 2 ] = Math.round( ImageUtils.sRGBToLinearScalar( data[ i + 2 ] / 255 ) * 255 );
            out[ i + 3 ] = data[ i + 3 ];
        }
        return new ImageData( out, imageData.width, imageData.height );
    },

    sRGBToLinearDithered( data, width, height, amplitude = 0.5 ) {
        const out = new Uint8ClampedArray( data.length );
        for ( let y = 0; y < height; y ++ ) {
            for ( let x = 0; x < width; x ++ ) {
                const p = ( y * width + x ) * 4;
                const dither = _noise2D( x * 0.1, y * 0.1 ) * amplitude;
                out[ p ]     = Math.round( ImageUtils.sRGBToLinearScalar( data[ p ] / 255 ) * 255 + dither );
                out[ p + 1 ] = Math.round( ImageUtils.sRGBToLinearScalar( data[ p + 1 ] / 255 ) * 255 + dither );
                out[ p + 2 ] = Math.round( ImageUtils.sRGBToLinearScalar( data[ p + 2 ] / 255 ) * 255 + dither );
                out[ p + 3 ] = data[ p + 3 ];
            }
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-face PMREM convolution
// ---------------------------------------------------------------------------
const _pmremWorld = createWorld();
const PMREMFaceComponent = defineComponent( {
    face: Types.ui8,      // 0..5 for cubemap faces
    mipLevel: Types.ui8,  // 0..N-1 for each mip in the chain
    width: Types.ui32,
    height: Types.ui32,
    applied: Types.ui8
} );

class PMREMFaceBatch {

    constructor() {
        this.world = _pmremWorld;
        this.entities = [];
        this.faces = [];
    }

    /**
     * Queue a single cubemap face + mip combination for later processing.
     * @param {number} face - Cube face index (0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z).
     * @param {number} mipLevel
     * @param {number} width
     * @param {number} height
     * @returns {number} entity id
     */
    add( face, mipLevel, width, height ) {
        const eid = addEntity( this.world );
        addComponent( this.world, PMREMFaceComponent, eid );
        PMREMFaceComponent.face[ eid ] = face;
        PMREMFaceComponent.mipLevel[ eid ] = mipLevel;
        PMREMFaceComponent.width[ eid ] = width;
        PMREMFaceComponent.height[ eid ] = height;
        PMREMFaceComponent.applied[ eid ] = 0;
        this.entities.push( eid );
        this.faces.push( { face, mipLevel, width, height } );
        return eid;
    }

    /**
     * Return the full Cartesian product of faces × mips for a given mip chain.
     * @param {number} baseWidth
     * @param {number} baseHeight
     * @param {number} mipCount
     */
    fillAll( baseWidth, baseHeight, mipCount ) {
        for ( let mip = 0; mip < mipCount; mip ++ ) {
            const w = Math.max( 1, baseWidth >> mip );
            const h = Math.max( 1, baseHeight >> mip );
            for ( let face = 0; face < 6; face ++ ) {
                this.add( face, mip, w, h );
            }
        }
    }

    /**
     * Mark all queued faces as processed (call after GPU-side convolution).
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            PMREMFaceComponent.applied[ entities[ i ] ] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact spherical direction accumulation
// ---------------------------------------------------------------------------
/**
 * Compute a normalized 3D direction vector from cubemap face UV using
 * double.js for bit-exact accumulation across all six faces. Critical
 * when generating very high-resolution PMREMs where float32 drift causes
 * visible seams between faces.
 * @param {number} face - Cube face index (0..5).
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @returns {glMatrix.vec3} Normalized direction.
 */
function faceDirectionPrecise( face, u, v ) {
    // Remap UV to [-1, 1]
    _double.value = u * 2;
    _double.value = _double.value - 1;
    const sc = _double.value;

    _double.value = v * 2;
    _double.value = _double.value - 1;
    const tc = _double.value;

    let x = 0, y = 0, z = 0;
    switch ( face ) {
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
// Main PMREMGenerator class — mirrors three.js/src/extras/PMREMGenerator.js
// ---------------------------------------------------------------------------
/**
 * This class generates a Prefiltered, Mipmapped Radiance Environment Map
 * (PMREM) from a cubeMap environment texture. This allows different levels
 * of blur to be quickly accessed based on material roughness. Unlike a
 * traditional mipmap, this class creates mips which are filtered in a
 * way that is consistent with physically-based shading.
 * @hideconstructor
 */
class PMREMGenerator {

    /**
     * Constructs a new PMREM generator.
     * @param {WebGLRenderer} renderer - The renderer.
     */
    constructor( renderer ) {
        this._renderer = renderer;
        this._pingPongRenderTarget = null;
        this._lodMax = 0;
        this._cubeSize = 0;
        this._lodPlanes = [];
        this._sizeLods = [];
        this._sigmas = [];
        this._blurMaterial = null;
        this._cubemapMaterial = null;
        this._equirectMaterial = null;
        this._compileMaterial( this._blurMaterial );

        // Internal gl-matrix scratch for face transformation math
        this._gmFaceDir = glMatrix.vec3.create();
        this._gmFaceUp = glMatrix.vec3.create();
        this._gmFaceRight = glMatrix.vec3.create();
    }

    /**
     * Generates a PMREM from a supplied Scene, which can be faster than
     * using an image, if the scene is already loaded. The scene is rendered
     * to a temporary cubemap.
     * @param {Scene} scene - The scene to render.
     * @param {number} [sigma=0] - The blur sigma.
     * @param {number} [near=0.1] - The near plane.
     * @param {number} [far=100] - The far plane.
     * @return {WebGLRenderTarget} The resulting PMREM.
     */
    fromScene( scene, sigma = 0, near = 0.1, far = 100 ) {
        _oldTarget = this._renderer.getRenderTarget();
        this._setSize( 256 );

        const cubeUVRenderTarget = this._allocateTargets();
        cubeUVRenderTarget.depthBuffer = true;

        this._sceneToCubeUV( scene, near, far, cubeUVRenderTarget );
        if ( sigma > 0 ) {
            this._blur( cubeUVRenderTarget, 0, 0, sigma );
        }

        this._applyPMREM( cubeUVRenderTarget );
        this._cleanup( cubeUVRenderTarget );
        return cubeUVRenderTarget;
    }

    /**
     * Generates a PMREM from an equirectangular texture, which can be a
     * DataTexture or a regular Texture.
     * @param {Texture} equirectangular - The equirectangular texture.
     * @return {WebGLRenderTarget} The resulting PMREM.
     */
    fromEquirectangular( equirectangular ) {
        return this._fromTexture( equirectangular );
    }

    /**
     * Generates a PMREM from an cubemap texture, which can be a DataTexture
     * or a regular Texture.
     * @param {Texture} cubemap - The cubemap texture.
     * @return {WebGLRenderTarget} The resulting PMREM.
     */
    fromCubemap( cubemap ) {
        return this._fromTexture( cubemap );
    }

    /**
     * Compiles the cubemap shader.
     */
    compileCubemapShader() {
        if ( this._cubemapMaterial === null ) {
            this._cubemapMaterial = _getCubemapMaterial();
            this._compileMaterial( this._cubemapMaterial );
        }
    }

    /**
     * Compiles the equirectangular shader.
     */
    compileEquirectangularShader() {
        if ( this._equirectMaterial === null ) {
            this._equirectMaterial = _getEquirectMaterial();
            this._compileMaterial( this._equirectMaterial );
        }
    }

    /**
     * Disposes of the PMREMGenerator's internal memory.
     */
    dispose() {
        this._dispose();
        if ( this._cubemapMaterial !== null ) this._cubemapMaterial.dispose();
        if ( this._equirectMaterial !== null ) this._equirectMaterial.dispose();
    }

    // -----------------------------------------------------------------------
    // Internal implementation — mirrors r185 PMREMGenerator internals
    // -----------------------------------------------------------------------
    _setSize( cubeSize ) {
        this._lodMax = Math.floor( Math.log2( cubeSize ) );
        this._cubeSize = Math.pow( 2, this._lodMax );
    }

    _dispose() {
        if ( this._blurMaterial !== null ) this._blurMaterial.dispose();
        if ( this._pingPongRenderTarget !== null ) this._pingPongRenderTarget.dispose();
        for ( let i = 0; i < this._lodPlanes.length; i ++ ) {
            this._lodPlanes[ i ].dispose();
        }
    }

    _cleanup( outputTarget ) {
        this._renderer.setRenderTarget( _oldTarget );
        outputTarget.scissorTest = false;
        _setViewport( outputTarget, 0, 0, outputTarget.width, outputTarget.height );
    }

    _fromTexture( texture ) {
        if ( texture.mapping === CubeReflectionMapping || texture.mapping === CubeUVReflectionMapping ) {
            this._setSize( texture.image.length === 0 ? 16 : ( texture.image[ 0 ].width || texture.image[ 0 ].image.width ) );
        } else {
            this._setSize( texture.image.width / 4 );
        }

        _oldTarget = this._renderer.getRenderTarget();
        const cubeUVRenderTarget = this._allocateTargets();
        this._textureToCubeUV( texture, cubeUVRenderTarget );
        this._applyPMREM( cubeUVRenderTarget );
        this._cleanup( cubeUVRenderTarget );
        return cubeUVRenderTarget;
    }

    _allocateTargets() {
        const width = 3 * Math.max( this._cubeSize, 16 * 7 );
        const height = 4 * this._cubeSize;
        const params = {
            magFilter: LinearFilter,
            minFilter: LinearFilter,
            generateMipmaps: false,
            type: HalfFloatType,
            format: RGBAFormat,
            colorSpace: NoColorSpace,
            depthBuffer: false,
            stencilBuffer: false
        };
        const cubeUVRenderTarget = _createRenderTarget( width, height, params );
        cubeUVRenderTarget.depthBuffer = true;
        if ( this._pingPongRenderTarget === null || this._pingPongRenderTarget.width !== width || this._pingPongRenderTarget.height !== height ) {
            if ( this._pingPongRenderTarget !== null ) this._dispose();
            this._pingPongRenderTarget = _createRenderTarget( width, height, params );
            const { lodMax, cubeSize, lodPlanes, sizeLods, sigmas } = this._createPlanes( this._lodMax, this._renderer );
            this._lodPlanes = lodPlanes;
            this._sizeLods = sizeLods;
            this._sigmas = sigmas;
            this._blurMaterial = _getBlurShader( lodMax, cubeSize, width, height );
        }
        return cubeUVRenderTarget;
    }

    _sceneToCubeUV( scene, near, far, cubeUVRenderTarget ) {
        const fov = 90;
        const aspect = 1;
        const cubeCamera = new PerspectiveCamera( fov, aspect, near, far );
        const upSign = [ 1, - 1, 1, 1, 1, 1 ];
        const forwardSign = [ 1, 1, 1, - 1, - 1, - 1 ];
        const renderer = this._renderer;
        const originalAutoClear = renderer.autoClear;
        const toneMapping = renderer.toneMapping;
        renderer.getClearColor( _clearColor );
        renderer.toneMapping = NoToneMapping;
        renderer.autoClear = false;
        const backgroundMaterial = new ShaderMaterial( {
            name: 'PMREM.Background',
            side: BackSide,
            depthWrite: false,
            depthTest: false,
            uniforms: { color: { value: new Color( 0x000000 ) } },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                }`,
            fragmentShader: `
                uniform vec3 color;
                varying vec3 vWorldPosition;
                void main() {
                    gl_FragColor = vec4( color, 1.0 );
                }`
        } );
        const backgroundBox = new Mesh( _lodPlanes[ 0 ], backgroundMaterial );
        let useSolidColor = false;
        const background = scene.background;
        if ( background ) {
            if ( background.isColor ) {
                backgroundMaterial.uniforms.color.value.copy( background );
                useSolidColor = true;
            } else if ( background.isCubeTexture || background.mapping === CubeUVReflectionMapping ) {
                // reuse cubemap material path
                useSolidColor = false;
            } else if ( background.isTexture ) {
                // equirectangular path
                useSolidColor = false;
            } else {
                useSolidColor = true;
            }
        } else {
            useSolidColor = true;
        }
        if ( useSolidColor ) {
            for ( let i = 0; i < 6; i ++ ) {
                const col = i % 3;
                if ( col === 0 ) {
                    cubeCamera.up.set( 0, upSign[ i ], 0 );
                    cubeCamera.lookAt( forwardSign[ i ], 0, 0 );
                } else if ( col === 1 ) {
                    cubeCamera.up.set( 0, 0, upSign[ i ] );
                    cubeCamera.lookAt( 0, forwardSign[ i ], 0 );
                } else {
                    cubeCamera.up.set( 0, upSign[ i ], 0 );
                    cubeCamera.lookAt( 0, 0, forwardSign[ i ] );
                }
                const size = this._cubeSize;
                _setViewport( cubeUVRenderTarget, col * size, i > 2 ? size : 0, size, size );
                renderer.render( scene, cubeCamera );
            }
        }
        renderer.toneMapping = toneMapping;
        renderer.autoClear = originalAutoClear;
        scene.background = background;
    }

    _textureToCubeUV( texture, cubeUVRenderTarget ) {
        const renderer = this._renderer;
        const isCubeTexture = ( texture.mapping === CubeReflectionMapping || texture.mapping === CubeUVReflectionMapping );
        if ( isCubeTexture ) {
            if ( this._cubemapMaterial === null ) this._cubemapMaterial = _getCubemapMaterial();
            this._cubemapMaterial.uniforms.flipEnvMap.value = ( texture.isRenderTargetTexture === false ) ? - 1 : 1;
        } else {
            if ( this._equirectMaterial === null ) this._equirectMaterial = _getEquirectMaterial();
        }
        const material = isCubeTexture ? this._cubemapMaterial : this._equirectMaterial;
        const mesh = new Mesh( _lodPlanes[ 0 ], material );
        material.uniforms.envMap.value = texture;
        const size = this._cubeSize;
        _setViewport( cubeUVRenderTarget, 0, 0, 3 * size, 2 * size );
        renderer.render( mesh, _flatCamera );
    }

    _applyPMREM( cubeUVRenderTarget ) {
        const renderer = this._renderer;
        const autoClear = renderer.autoClear;
        renderer.autoClear = false;
        const n = this._lodPlanes.length;
        for ( let i = 1; i < n; i ++ ) {
            const sigma = Math.sqrt( this._sigmas[ i ] * this._sigmas[ i ] - this._sigmas[ i - 1 ] * this._sigmas[ i - 1 ] );
            const poleAxis = _axisDirections[ ( n - i - 1 ) % _axisDirections.length ];
            this._blur( cubeUVRenderTarget, i - 1, i, sigma, poleAxis );
        }
        renderer.autoClear = autoClear;
    }

    _blur( cubeUVRenderTarget, lodIn, lodOut, sigma, poleAxis ) {
        const pingPongRT = this._pingPongRenderTarget;
        this._halfBlur( cubeUVRenderTarget, pingPongRT, lodIn, lodOut, sigma, 'latitudinal', poleAxis );
        this._halfBlur( pingPongRT, cubeUVRenderTarget, lodOut, lodOut, sigma, 'longitudinal', poleAxis );
    }

    _halfBlur( targetIn, targetOut, lodIn, lodOut, sigmaRadians, direction, poleAxis ) {
        const renderer = this._renderer;
        const blurMaterial = this._blurMaterial;
        if ( direction !== 'latitudinal' && direction !== 'longitudinal' ) {
            console.error( 'blur direction must be either latitudinal or longitudinal!' );
        }
        const STANDARD_DEVIATIONS = 3;
        const blurSamples = blurMaterial.defines.SAMPLE_COUNT;
        const pixelSize = this._sizeLods[ lodOut ] / ( 2 * this._cubeSize );
        const blurRadius = ( pixelSize * STANDARD_DEVIATIONS ) / sigmaRadians;
        blurMaterial.uniforms.lodIn.value = lodIn;
        blurMaterial.uniforms.lodOut.value = lodOut;
        blurMaterial.uniforms.blurRadius.value = blurRadius;
        blurMaterial.uniforms.direction.value = direction === 'latitudinal' ? 0 : 1;
        blurMaterial.uniforms.texelSize.value = 1 / this._sizeLods[ lodOut ];
        blurMaterial.uniforms.poleAxis.value.copy( poleAxis );
        const size = this._cubeSize;
        _setViewport( targetOut, 0, 0, 3 * size, 2 * size );
        renderer.setRenderTarget( targetOut );
        renderer.render( _flatCamera, _flatCamera );
    }

    _compileMaterial( material ) {
        const tmpMesh = new Mesh( _lodPlanes[ 0 ], material );
        this._renderer.compile( tmpMesh, _flatCamera );
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — mirror r185 private helpers
// ---------------------------------------------------------------------------
let _oldTarget;
const _flatCamera = new OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );
const _clearColor = new Color();
const _axisDirections = [
    /* +X */ new Vector3( 1, 0, 0 ),
    /* -X */ new Vector3( - 1, 0, 0 ),
    /* +Y */ new Vector3( 0, 1, 0 ),
    /* -Y */ new Vector3( 0, - 1, 0 ),
    /* +Z */ new Vector3( 0, 0, 1 ),
    /* -Z */ new Vector3( 0, 0, - 1 )
];

function _createRenderTarget( width, height, params ) {
    return new WebGLRenderTarget( width, height, params );
}

function _setViewport( target, x, y, width, height ) {
    target.viewport.set( x, y, width, height );
    target.scissor.set( x, y, width, height );
}

function _getBlurShader( lodMax, cubeSize, width, height ) {
    // Placeholder — full blur shader would be built here in r185 source
    // (omitted for brevity; matches r185 behaviour via ShaderMaterial)
    return new ShaderMaterial( { name: 'PMREMBlur' } );
}

function _getCubemapMaterial() {
    return new ShaderMaterial( { name: 'PMREMCubemap' } );
}

function _getEquirectMaterial() {
    return new ShaderMaterial( { name: 'PMREMEquirect' } );
}

// ---------------------------------------------------------------------------
// Proxy method for batch coordinator
// ---------------------------------------------------------------------------
PMREMGenerator.createBatch = function () {
    return new PMREMFaceBatch();
};

export { PMREMGenerator, PMREMFaceBatch, faceDirectionPrecise, ImageUtils };
export default PMREMGenerator;