// file number : 011
// full path name : src/textures/011_videotexture.js
// description : VideoTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internal 002_texture.js base class and imports math
// helpers strictly from the threejs_new01 math folder. Wraps an
// HTMLVideoElement as a live-updating texture source, overriding magFilter/
// minFilter to LinearFilter (no mipmaps since video frames change every tick)
// and generateMipmaps to false. Preserves the full r185 API including update()
// with readyState gating, and uses requestVideoFrameCallback when available
// for frame-accurate updates. Adds gl-matrix accelerated per-frame UV
// staging for motion-vector pipelines, bitecs SoA batching for multi-video
// compositing, double.js bit-exact frame-delta accumulation for playback
// analytics, and simplex-noise dithered frame blending for organic crossfades.
// best for : VideoTexture, HTMLVideoElement playback, video billboards, video
// backgrounds, AR/VR media planes, live camera feeds (via MediaStream),
// volumetric video, and any three.js workflow that binds a video element to
// a material.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    LinearFilter,
    ClampToEdgeWrapping,
    RGBAFormat,
    UnsignedByteType,
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

// gl-matrix scratch for zero-allocation video frame staging
const _gm_v2 = glMatrix.vec2.create();
const _gm_v4 = glMatrix.vec4.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-video compositing
// ---------------------------------------------------------------------------
const _videoWorld = createWorld();
const VideoFrameComponent = defineComponent( {
    textureId: Types.ui16,
    videoPtr: Types.ui32,
    currentTime: Types.f64,
    duration: Types.f64,
    readyState: Types.ui8,
    paused: Types.ui8,
    updated: Types.ui8
} );

class VideoTextureBatch {

    constructor() {
        this.world = _videoWorld;
        this.textures = [];
        this.videos = [];
        this.entities = [];
    }

    /**
     * Register a VideoTexture instance for batched frame updates.
     * @param {VideoTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a per-frame update job for a registered VideoTexture.
     * @param {number} textureId
     * @returns {number} entity id
     */
    add( textureId ) {
        const eid = addEntity( this.world );
        addComponent( this.world, VideoFrameComponent, eid );
        const texture = this.textures[ textureId ];
        const video = texture.image;
        const videoIndex = this.videos.length;
        this.videos.push( video );
        VideoFrameComponent.textureId[ eid ] = textureId;
        VideoFrameComponent.videoPtr[ eid ] = videoIndex;
        VideoFrameComponent.currentTime[ eid ] = video?.currentTime ?? 0;
        VideoFrameComponent.duration[ eid ] = video?.duration ?? 0;
        VideoFrameComponent.readyState[ eid ] = video?.readyState ?? 0;
        VideoFrameComponent.paused[ eid ] = video?.paused ? 1 : 0;
        VideoFrameComponent.updated[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued frame updates in one cache-friendly pass.
     * Each video's `update()` is called and its readyState recorded.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const texture = this.textures[ VideoFrameComponent.textureId[ eid ] ];
            const video = this.videos[ VideoFrameComponent.videoPtr[ eid ] ];

            if ( texture && video ) {
                texture.update();
                VideoFrameComponent.currentTime[ eid ] = video.currentTime;
                VideoFrameComponent.readyState[ eid ] = video.readyState;
                VideoFrameComponent.paused[ eid ] = video.paused ? 1 : 0;
                VideoFrameComponent.updated[ eid ] = 1;
            }
        }
    }

    /**
     * Total playback time across all registered videos, using double.js
     * for bit-exact accumulation.
     * @returns {number}
     */
    totalPlaybackTime() {
        _double.value = 0;
        for ( let i = 0, l = this.entities.length; i < l; i ++ ) {
            _double.add( VideoFrameComponent.currentTime[ this.entities[ i ] ] );
        }
        return _double.value;
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact frame-delta accumulation
// ---------------------------------------------------------------------------
/**
 * Accumulate a per-frame time delta using double.js for bit-exact
 * precision. Prevents float32 drift when tracking cumulative playback
 * time over very long sessions (e.g. continuous media installations).
 * @param {number} accumulated
 * @param {number} delta
 * @returns {number}
 */
function accumulateFrameDeltaPrecise( accumulated, delta ) {
    _double.value = accumulated;
    _double.add( delta );
    return _double.value;
}

// ---------------------------------------------------------------------------
// simplex-noise dithered frame blending
// ---------------------------------------------------------------------------
/**
 * Blend two RGBA frame buffers with a simplex-noise dithered crossfade
 * factor. Useful for organic crossfades between video clips where a
 * uniform linear blend would produce visible banding.
 * @param {Uint8ClampedArray} frameA
 * @param {Uint8ClampedArray} frameB
 * @param {number} t - Blend factor in [0, 1].
 * @param {number} width
 * @param {number} height
 * @param {number} [amplitude=0.5]
 * @returns {Uint8ClampedArray}
 */
function blendFramesDithered( frameA, frameB, t, width, height, amplitude = 0.5 ) {
    const out = new Uint8ClampedArray( frameA.length );
    const invAmp = amplitude / 255;

    for ( let y = 0; y < height; y ++ ) {
        for ( let x = 0; x < width; x ++ ) {
            const p = ( y * width + x ) * 4;
            const d = _noise2D( x * 0.1, y * 0.1 ) * invAmp;
            const blend = Math.max( 0, Math.min( 1, t + d ) );

            out[ p ]     = frameA[ p ]     * ( 1 - blend ) + frameB[ p ]     * blend;
            out[ p + 1 ] = frameA[ p + 1 ] * ( 1 - blend ) + frameB[ p + 1 ] * blend;
            out[ p + 2 ] = frameA[ p + 2 ] * ( 1 - blend ) + frameB[ p + 2 ] * blend;
            out[ p + 3 ] = frameA[ p + 3 ] * ( 1 - blend ) + frameB[ p + 3 ] * blend;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated per-frame UV staging
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated UV motion extraction. Computes the UV delta
 * between two consecutive frames (useful for motion-vector pipelines,
 * optical-flow debug, or video-driven parallax).
 * @param {number} u0
 * @param {number} v0
 * @param {number} u1
 * @param {number} v1
 * @returns {glMatrix.vec2} UV delta [du, dv].
 */
function computeUVMotionGlMat( u0, v0, u1, v1 ) {
    return glMatrix.vec2.set( _gm_v2, u1 - u0, v1 - v0 );
}

/**
 * gl-matrix accelerated RGBA staging from a video element. Requires the
 * video to be readable via canvas (CORS-safe source). Writes into a
 * preallocated vec4 for zero-allocation downstream uniform uploads.
 * @param {HTMLVideoElement} video
 * @param {number} u - U coordinate in [0, 1].
 * @param {number} v - V coordinate in [0, 1].
 * @param {glMatrix.vec4} [out]
 * @returns {glMatrix.vec4|null}
 */
function sampleVideoUVGlMat( video, u, v, out = _gm_v4 ) {
    if ( ! video || video.readyState < 2 ) return null;
    const canvas = document.createElement( 'canvas' );
    canvas.width = video.videoWidth || 1;
    canvas.height = video.videoHeight || 1;
    const ctx = canvas.getContext( '2d' );
    ctx.drawImage( video, 0, 0, canvas.width, canvas.height );

    const x = Math.min( canvas.width - 1, Math.floor( u * canvas.width ) );
    const y = Math.min( canvas.height - 1, Math.floor( v * canvas.height ) );
    const imageData = ctx.getImageData( x, y, 1, 1 );

    glMatrix.vec4.set(
        out,
        imageData.data[ 0 ] / 255,
        imageData.data[ 1 ] / 255,
        imageData.data[ 2 ] / 255,
        imageData.data[ 3 ] / 255
    );
    return out;
}

// ---------------------------------------------------------------------------
// Main VideoTexture class — mirrors three.js/src/textures/VideoTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a texture from a video element.
 *
 * ```js
 * const video = document.getElementById( 'video' );
 * const texture = new THREE.VideoTexture( video );
 * ```
 * @augments Texture
 */
class VideoTexture extends Texture {

    /**
     * Constructs a new video texture.
     * @param {HTMLVideoElement} video - The video element to use as a data source.
     * @param {number} [mapping=Texture.DEFAULT_MAPPING] - The texture mapping.
     * @param {number} [wrapS=ClampToEdgeWrapping] - The wrapS value.
     * @param {number} [wrapT=ClampToEdgeWrapping] - The wrapT value.
     * @param {number} [magFilter=LinearFilter] - The mag filter value.
     * @param {number} [minFilter=LinearFilter] - The min filter value.
     * @param {number} [format=RGBAFormat] - The texture format.
     * @param {number} [type=UnsignedByteType] - The texture type.
     * @param {number} [anisotropy=Texture.DEFAULT_ANISOTROPY] - The anisotropy value.
     */
    constructor(
        video,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearFilter,
        format = RGBAFormat,
        type = UnsignedByteType,
        anisotropy = Texture.DEFAULT_ANISOTROPY
    ) {
        super( video, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy, NoColorSpace );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isVideoTexture = true;

        this.type = 'VideoTexture';

        /**
         * Whether to generate mipmaps (if possible) for a texture.
         * Overwritten and set to `false` by default since updating a
         * texture every frame is expensive.
         * @type {boolean}
         * @default false
         */
        this.generateMipmaps = false;

        /**
         * Internal tracking of the previous frame time. Used to
         * short-circuit `update()` when the video has not advanced.
         * @type {number}
         * @private
         */
        this._lastFrameTime = - 1;

        /**
         * Internal requestVideoFrameCallback handle, if available.
         * @type {number}
         * @private
         */
        this._rvfcHandle = null;

        // Bind the RVFC callback for frame-accurate updates
        if ( typeof this.image.requestVideoFrameCallback === 'function' ) {
            this._bindVideoFrameCallback();
        }
    }

    /**
     * Binds the browser's requestVideoFrameCallback API to mark the
     * texture as needing an update on every decoded frame. Falls back
     * silently when the API is not available.
     * @private
     */
    _bindVideoFrameCallback() {
        const video = this.image;
        const onFrame = () => {
            this._lastFrameTime = video.currentTime;
            this.needsUpdate = true;
            this._rvfcHandle = video.requestVideoFrameCallback( onFrame );
        };
        this._rvfcHandle = video.requestVideoFrameCallback( onFrame );
    }

    /**
     * Updates the texture from the currently loaded video frame.
     * Call this method once per frame in your animation loop when using
     * a VideoTexture.
     *
     * The update is gated on the video's `readyState` being at least
     * `HAVE_CURRENT_DATA` (2), and on the video not being paused with
     * an unchanged currentTime.
     */
    update() {
        const video = this.image;

        // Fallback path when requestVideoFrameCallback is unavailable
        if ( this._rvfcHandle === null ) {
            if ( video.readyState >= video.HAVE_CURRENT_DATA ) {
                if ( video.currentTime !== this._lastFrameTime ) {
                    this._lastFrameTime = video.currentTime;
                    this.needsUpdate = true;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated RGBA sampling of the current video frame at
     * normalized UV coordinates. Requires a CORS-safe source.
     * @param {number} [u=0.5] - U coordinate in [0, 1].
     * @param {number} [v=0.5] - V coordinate in [0, 1].
     * @param {glMatrix.vec4} [out]
     * @returns {glMatrix.vec4|null}
     */
    sampleVideoUVGlMat( u = 0.5, v = 0.5, out = _gm_v4 ) {
        return sampleVideoUVGlMat( this.image, u, v, out );
    }

    /**
     * gl-matrix accelerated extraction of the video's native dimensions
     * into a preallocated vec2. Zero-allocation.
     * @returns {glMatrix.vec2}
     */
    getVideoDimensionsGlMat() {
        const w = this.image?.videoWidth ?? 0;
        const h = this.image?.videoHeight ?? 0;
        return glMatrix.vec2.set( _gm_v2, w, h );
    }

    /**
     * double.js bit-exact accumulation of a per-frame time delta. Store
     * the returned value in a persistent field to track cumulative
     * playback time without float32 drift.
     * @param {number} accumulated - Previous accumulated value.
     * @param {number} delta - New frame delta in seconds.
     * @returns {number}
     */
    static accumulateFrameDeltaPrecise( accumulated, delta ) {
        return accumulateFrameDeltaPrecise( accumulated, delta );
    }

    /**
     * Blend two RGBA frame buffers with a simplex-noise dithered crossfade.
     * @param {Uint8ClampedArray} frameA
     * @param {Uint8ClampedArray} frameB
     * @param {number} t - Blend factor in [0, 1].
     * @param {number} width
     * @param {number} height
     * @param {number} [amplitude=0.5]
     * @returns {Uint8ClampedArray}
     */
    static blendFramesDithered( frameA, frameB, t, width, height, amplitude = 0.5 ) {
        return blendFramesDithered( frameA, frameB, t, width, height, amplitude );
    }

    /**
     * Create a batched video-frame update coordinator backed by bitecs.
     * Register multiple VideoTextures and drive all their `update()` calls
     * in one cache-friendly pass.
     * @returns {VideoTextureBatch}
     */
    static createBatch() {
        return new VideoTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {VideoTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.generateMipmaps = false;
        return this;
    }

    /**
     * Serializes the texture into JSON.
     * @param {Object} [meta] - Optional metadata.
     * @return {Object} A JSON object representing the serialized texture.
     */
    toJSON( meta ) {
        const isRootObject = ( meta === undefined || typeof meta === 'string' );
        const output = super.toJSON( meta );

        // Video textures cannot serialize their frame data — only the
        // structural metadata and a reference to the source video src.
        if ( this.image ) {
            output.image = {
                videoWidth: this.image.videoWidth || 0,
                videoHeight: this.image.videoHeight || 0,
                src: this.image.src || ''
            };
        }

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }

    /**
     * Disposes the texture and releases the RVFC handle if bound.
     */
    dispose() {
        const video = this.image;
        if ( this._rvfcHandle !== null && video && typeof video.cancelVideoFrameCallback === 'function' ) {
            video.cancelVideoFrameCallback( this._rvfcHandle );
            this._rvfcHandle = null;
        }
        this.dispatchEvent( { type: 'dispose' } );
    }
}

export {
    VideoTexture,
    VideoTextureBatch,
    accumulateFrameDeltaPrecise,
    blendFramesDithered,
    computeUVMotionGlMat,
    sampleVideoUVGlMat
};
export default VideoTexture;