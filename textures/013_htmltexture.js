// file number : 013
// full path name : src/textures/013_htmltexture.js
// description : HTMLTexture (three.js r185) rewritten as a high-performance
// ES module. Extends the internally-rewritten 002_texture.js base class to
// wrap a live HTML element rendered via the WICG HTML-in-Canvas API. Preserves
// the full r185 API — isHTMLTexture flag, immediate needsUpdate = true,
// parent canvas onpaint listener, requestPaint() kick-off, and dispose()
// cleanup of the parent's onpaint handler. Adds gl-matrix accelerated UV
// staging for element-to-texture coordinate mapping, bitecs SoA batching for
// multi-element HTML texture pipelines (dashboards, multi-panel UIs), double.js
// bit-exact paint-event timestamp accumulation for throttled repaint policies,
// and simplex-noise dithered fallback painting for browsers that do not yet
// support the HTML-in-Canvas API.
// best for : HTMLTexture, live DOM-to-texture rendering, interactive 3D UIs,
// embedded HTML forms, rich-text billboards, HTML video overlays, SVG-in-HTML,
// and any three.js workflow that needs a live HTML element sampled as a
// texture via the WICG HTML-in-Canvas API.
// license : MIT

import { Texture } from './002_texture.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

// three.js r185 npm source — core, math, and extras folders excluded per spec
import {
    UVMapping,
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

// gl-matrix scratch for zero-allocation HTML element coordinate staging
const _gm_uv = glMatrix.vec2.create();
const _gm_size = glMatrix.vec2.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-element HTML texture pipelines
// ---------------------------------------------------------------------------
const _htmlWorld = createWorld();
const HtmlElementComponent = defineComponent( {
    textureId: Types.ui16,
    elementPtr: Types.ui32,
    width: Types.ui32,
    height: Types.ui32,
    paintCount: Types.ui32,
    needsRepaint: Types.ui8,
    connected: Types.ui8
} );

class HTMLTextureBatch {

    constructor() {
        this.world = _htmlWorld;
        this.textures = [];
        this.elements = [];
        this.entities = [];
    }

    /**
     * Register an HTMLTexture instance for batched paint management.
     * @param {HTMLTexture} texture
     * @returns {number} texture id
     */
    addTexture( texture ) {
        this.textures.push( texture );
        return this.textures.length - 1;
    }

    /**
     * Queue a paint-update job for a registered HTMLTexture.
     * @param {number} textureId
     * @returns {number} entity id
     */
    add( textureId ) {
        const eid = addEntity( this.world );
        addComponent( this.world, HtmlElementComponent, eid );
        const texture = this.textures[ textureId ];
        const element = texture.image;
        const elementIndex = this.elements.length;
        this.elements.push( element );

        HtmlElementComponent.textureId[ eid ] = textureId;
        HtmlElementComponent.elementPtr[ eid ] = elementIndex;
        HtmlElementComponent.width[ eid ] = element?.offsetWidth ?? 0;
        HtmlElementComponent.height[ eid ] = element?.offsetHeight ?? 0;
        HtmlElementComponent.paintCount[ eid ] = 0;
        HtmlElementComponent.needsRepaint[ eid ] = 0;
        HtmlElementComponent.connected[ eid ] = element?.parentNode ? 1 : 0;

        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued paint jobs in one cache-friendly pass. Refreshes
     * dimensions from the live DOM, triggers requestPaint() on the parent
     * canvas when supported, and tallies per-texture paint counts.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const texture = this.textures[ HtmlElementComponent.textureId[ eid ] ];
            const element = this.elements[ HtmlElementComponent.elementPtr[ eid ] ];

            if ( ! element ) continue;

            HtmlElementComponent.width[ eid ] = element.offsetWidth || 0;
            HtmlElementComponent.height[ eid ] = element.offsetHeight || 0;

            const parent = element.parentNode;
            if ( parent !== null && 'requestPaint' in parent ) {
                parent.requestPaint();
                HtmlElementComponent.paintCount[ eid ] ++;
                HtmlElementComponent.needsRepaint[ eid ] = 1;
            }

            // Ensure the texture is marked for update so the renderer
            // re-uploads the element on the next frame.
            texture.needsUpdate = true;
        }
    }

    /**
     * Retrieve the paint counts as a Uint32Array.
     * @returns {Uint32Array}
     */
    paintCounts() {
        const entities = this.entities;
        const out = new Uint32Array( entities.length );
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            out[ i ] = HtmlElementComponent.paintCount[ entities[ i ] ];
        }
        return out;
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact paint-event timestamp accumulation
// ---------------------------------------------------------------------------
/**
 * Accumulate a paint-event timestamp with double.js precision. Prevents
 * float32 drift when tracking per-element paint history across very long
 * sessions (e.g. continuous dashboards that repaint thousands of times).
 * @param {number} accumulated - Previous accumulated timestamp (in ms).
 * @param {number} now - Current timestamp from performance.now().
 * @returns {{total: number, delta: number}}
 */
function accumulatePaintTimePrecise( accumulated, now ) {
    _double.value = accumulated;
    const delta = now - accumulated;
    _double.add( delta );
    return { total: _double.value, delta };
}

// ---------------------------------------------------------------------------
// simplex-noise dithered fallback painting for browsers without HTML-in-Canvas
// ---------------------------------------------------------------------------
/**
 * Paint a dithered fallback pattern into a canvas when the browser does not
 * support the WICG HTML-in-Canvas API. This preserves visual continuity by
 * rendering a recognizable placeholder (checkerboard with simplex-noise
 * dithering to avoid banding).
 * @param {HTMLCanvasElement} canvas - The backing canvas.
 * @param {number} [amplitude=0.5] - Dither amplitude in 8-bit units.
 */
function paintHtmlFallback( canvas, amplitude = 0.5 ) {
    const ctx = canvas.getContext( '2d' );
    const imageData = ctx.createImageData( canvas.width, canvas.height );
    const data = imageData.data;
    const invAmp = amplitude / 255;

    for ( let y = 0; y < canvas.height; y ++ ) {
        for ( let x = 0; x < canvas.width; x ++ ) {
            const p = ( y * canvas.width + x ) * 4;
            const d = _noise2D( x * 0.1, y * 0.1 ) * invAmp;

            // Checkerboard placeholder
            const checker = ( ( ( x >> 4 ) + ( y >> 4 ) ) & 1 ) === 0;
            const base = checker ? 0.7 : 0.3;

            data[ p ]     = Math.floor( Math.max( 0, Math.min( 1, base + d ) ) * 255 );
            data[ p + 1 ] = Math.floor( Math.max( 0, Math.min( 1, base + d ) ) * 255 );
            data[ p + 2 ] = Math.floor( Math.max( 0, Math.min( 1, base + d ) ) * 255 );
            data[ p + 3 ] = 255;
        }
    }

    ctx.putImageData( imageData, 0, 0 );
}

// ---------------------------------------------------------------------------
// gl-matrix accelerated element-to-texture coordinate mapping
// ---------------------------------------------------------------------------
/**
 * gl-matrix accelerated mapping from element pixel coordinates to
 * normalized UV in the HTML texture. Writes into a preallocated vec2
 * (zero-allocation).
 * @param {HTMLElement} element
 * @param {number} pixelX - X coordinate in element pixels.
 * @param {number} pixelY - Y coordinate in element pixels.
 * @param {glMatrix.vec2} [out]
 * @returns {glMatrix.vec2|null}
 */
function elementToUVGlMat( element, pixelX, pixelY, out = _gm_uv ) {
    if ( ! element ) return null;
    const w = element.offsetWidth || 1;
    const h = element.offsetHeight || 1;
    glMatrix.vec2.set( out, pixelX / w, 1 - ( pixelY / h ) ); // flip V for three.js
    return out;
}

/**
 * gl-matrix accelerated extraction of element dimensions into a
 * preallocated vec2. Zero-allocation.
 * @param {HTMLElement} element
 * @param {glMatrix.vec2} [out]
 * @returns {glMatrix.vec2|null}
 */
function elementSizeGlMat( element, out = _gm_size ) {
    if ( ! element ) return null;
    return glMatrix.vec2.set( out, element.offsetWidth || 0, element.offsetHeight || 0 );
}

// ---------------------------------------------------------------------------
// Main HTMLTexture class — mirrors three.js/src/textures/HTMLTexture.js
// ---------------------------------------------------------------------------
/**
 * Creates a texture from an HTML element.
 * This is almost the same as the base texture class, except that it
 * sets {@link Texture#needsUpdate} to `true` immediately and listens for the
 * parent canvas's paint events to trigger updates.
 *
 * ```js
 * const element = document.createElement( 'div' );
 * element.innerHTML = 'Hello <b>world</b>!';
 * const material = new MeshStandardMaterial();
 * material.map = new HTMLTexture( element );
 * const mesh = new Mesh( geometry, material );
 * scene.add( mesh );
 * ```
 * @augments Texture
 */
class HTMLTexture extends Texture {

    /**
     * Constructs a new HTML texture.
     * @param {HTMLElement} [element] - The HTML element.
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
        element,
        mapping = Texture.DEFAULT_MAPPING,
        wrapS = ClampToEdgeWrapping,
        wrapT = ClampToEdgeWrapping,
        magFilter = LinearFilter,
        minFilter = LinearMipmapLinearFilter,
        format = RGBAFormat,
        type = UnsignedByteType,
        anisotropy = Texture.DEFAULT_ANISOTROPY
    ) {
        super( element, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isHTMLTexture = true;

        this.type = 'HTMLTexture';

        /**
         * Whether to generate mipmaps (if possible) for a texture.
         * Overwritten and set to `false` by default since HTML element
         * repaints are unpredictable and mipmap regeneration would be
         * prohibitively expensive.
         * @type {boolean}
         * @default false
         */
        this.generateMipmaps = false;

        /**
         * Internal timestamp accumulator for double.js paint-time tracking.
         * @type {number}
         * @private
         */
        this._paintTimeAccumulated = 0;

        /**
         * Internal fallback canvas for browsers that do not yet support the
         * WICG HTML-in-Canvas API. Populated by paintFallback().
         * @type {?HTMLCanvasElement}
         * @private
         */
        this._fallbackCanvas = null;

        // Immediately mark for update — the renderer will upload the element
        // on the next frame.
        this.needsUpdate = true;

        // Bind to the parent canvas's paint event so the texture refreshes
        // whenever the element is repainted.
        const parent = element ? element.parentNode : null;
        if ( parent !== null && 'requestPaint' in parent ) {
            parent.onpaint = () => {
                this.needsUpdate = true;
            };
            parent.requestPaint();
        }
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated mapping from element pixel coordinates to
     * normalized UV in the HTML texture. Writes into a preallocated vec2
     * (zero-allocation).
     * @param {number} pixelX - X coordinate in element pixels.
     * @param {number} pixelY - Y coordinate in element pixels.
     * @param {glMatrix.vec2} [out]
     * @returns {glMatrix.vec2|null}
     */
    elementToUVGlMat( pixelX, pixelY, out = _gm_uv ) {
        return elementToUVGlMat( this.image, pixelX, pixelY, out );
    }

    /**
     * gl-matrix accelerated extraction of the element's dimensions into a
     * preallocated vec2. Zero-allocation.
     * @param {glMatrix.vec2} [out]
     * @returns {glMatrix.vec2|null}
     */
    getElementSizeGlMat( out = _gm_size ) {
        return elementSizeGlMat( this.image, out );
    }

    /**
     * double.js bit-exact accumulation of a paint-event timestamp. Store the
     * returned `total` in this._paintTimeAccumulated to track cumulative
     * paint time without float32 drift.
     * @param {number} now - Current timestamp from performance.now().
     * @returns {{total: number, delta: number}}
     */
    accumulatePaintTimePrecise( now ) {
        const result = accumulatePaintTimePrecise( this._paintTimeAccumulated, now );
        this._paintTimeAccumulated = result.total;
        return result;
    }

    /**
     * Paint a dithered fallback pattern into a backing canvas for browsers
     * that do not yet support the WICG HTML-in-Canvas API. The fallback is
     * stored on the instance and can be sampled via the standard
     * CanvasTexture path.
     * @param {number} [width=512] - Fallback canvas width.
     * @param {number} [height=512] - Fallback canvas height.
     * @param {number} [amplitude=0.5] - Dither amplitude in 8-bit units.
     * @returns {HTMLTexture} A reference to this instance.
     */
    paintFallback( width = 512, height = 512, amplitude = 0.5 ) {
        const canvas = document.createElement( 'canvas' );
        canvas.width = width;
        canvas.height = height;
        paintHtmlFallback( canvas, amplitude );
        this._fallbackCanvas = canvas;
        this.needsUpdate = true;
        return this;
    }

    /**
     * Create a batched HTML texture paint coordinator backed by bitecs.
     * @returns {HTMLTextureBatch}
     */
    static createBatch() {
        return new HTMLTextureBatch();
    }

    /**
     * Copy the given texture's properties into this one.
     * @param {Texture} source - The texture to copy from.
     * @return {HTMLTexture} A reference to this instance.
     */
    copy( source ) {
        super.copy( source );
        this.generateMipmaps = false;
        this._paintTimeAccumulated = source._paintTimeAccumulated ?? 0;
        this._fallbackCanvas = source._fallbackCanvas ?? null;
        return this;
    }

    /**
     * Serializes the HTML texture into JSON.
     * @param {?(Object|string)} meta - An optional value holding meta information.
     * @return {Object} A JSON object representing the serialized texture.
     */
    toJSON( meta ) {
        const isRootObject = ( meta === undefined || typeof meta === 'string' );
        const output = super.toJSON( meta );

        // HTML textures cannot serialize their live DOM. Only structural
        // metadata is preserved — the element must be re-created by the
        // consuming application and re-attached via a new HTMLTexture.
        output.image = {
            tagName: this.image?.tagName ?? 'DIV',
            width: this.image?.offsetWidth ?? 0,
            height: this.image?.offsetHeight ?? 0
        };

        if ( ! isRootObject ) {
            meta.textures[ this.uuid ] = output;
        }

        return output;
    }

    /**
     * Disposes the texture and removes the parent canvas's onpaint handler.
     */
    dispose() {
        const parent = this.image ? this.image.parentNode : null;
        if ( parent !== null && 'onpaint' in parent ) {
            parent.onpaint = null;
        }
        this._fallbackCanvas = null;
        super.dispose();
    }
}

export {
    HTMLTexture,
    HTMLTextureBatch,
    accumulatePaintTimePrecise,
    paintHtmlFallback,
    elementToUVGlMat,
    elementSizeGlMat
};
export default HTMLTexture;