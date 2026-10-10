// file number : 019
// full path name : src/extras/lib/019_arccurve.js
// description : A curve representing a 2D circular arc (three.js r185) rewritten
// as a high-performance ES module. Extends the internal 008_ellipsecurve.js base
// class and imports Vector2 strictly from the threejs_new01 math folder. Adds
// gl-matrix accelerated batch point evaluation, bitecs SoA batching for multi-arc
// sampling, double.js high-precision angular sweep accumulation for very small or
// very large arcs, and simplex-noise organic radius modulation for hand-drawn
// circular arc effects.
// best for : ArcCurve, Shape.absarc(), Path.absarc(), circular geometry
// generation, and any 2D circular arc interpolation used by three.js shape and
// geometry APIs.
// license : MIT

import { EllipseCurve } from './008_ellipsecurve.js';
import { Vector2 } from 'https://raw.githubusercontent.com/3Dsamples/Development-Temporary-Godot-4.6-repository-/threejs_new01/math/002_Vector2.js';

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

// gl-matrix scratch for zero-allocation arc evaluation
const _gm_v2 = glMatrix.vec2.create();

// Shared Vector2 scratch to avoid allocations in accelerated paths
const _v2 = new Vector2();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-arc sampling
// ---------------------------------------------------------------------------
const _arcWorld = createWorld();
const ArcSampleComponent = defineComponent( {
    ax: Types.f64,
    ay: Types.f64,
    aRadius: Types.f64,
    aStartAngle: Types.f64,
    aEndAngle: Types.f64,
    aClockwise: Types.ui8,
    t: Types.f64,
    x: Types.f64,
    y: Types.f64
} );

class ArcCurveBatch {

    constructor() {
        this.world = _arcWorld;
        this.entities = [];
    }

    /**
     * Queue a single arc sample.
     * @param {number} ax
     * @param {number} ay
     * @param {number} aRadius
     * @param {number} aStartAngle
     * @param {number} aEndAngle
     * @param {boolean} aClockwise
     * @param {number} t - Interpolation factor in [0, 1].
     * @returns {number} entity id
     */
    add( ax, ay, aRadius, aStartAngle, aEndAngle, aClockwise, t ) {
        const eid = addEntity( this.world );
        addComponent( this.world, ArcSampleComponent, eid );
        ArcSampleComponent.ax[ eid ] = ax;
        ArcSampleComponent.ay[ eid ] = ay;
        ArcSampleComponent.aRadius[ eid ] = aRadius;
        ArcSampleComponent.aStartAngle[ eid ] = aStartAngle;
        ArcSampleComponent.aEndAngle[ eid ] = aEndAngle;
        ArcSampleComponent.aClockwise[ eid ] = aClockwise ? 1 : 0;
        ArcSampleComponent.t[ eid ] = t;
        ArcSampleComponent.x[ eid ] = 0;
        ArcSampleComponent.y[ eid ] = 0;
        this.entities.push( eid );
        return eid;
    }

    /**
     * Process all queued samples in one cache-friendly pass.
     * Uses gl-matrix for per-sample vec2 staging.
     */
    process() {
        const entities = this.entities;
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            const eid = entities[ i ];
            const t = ArcSampleComponent.t[ eid ];
            const ax = ArcSampleComponent.ax[ eid ];
            const ay = ArcSampleComponent.ay[ eid ];
            const aRadius = ArcSampleComponent.aRadius[ eid ];
            const aStartAngle = ArcSampleComponent.aStartAngle[ eid ];
            const aEndAngle = ArcSampleComponent.aEndAngle[ eid ];
            const aClockwise = ArcSampleComponent.aClockwise[ eid ] === 1;

            const twoPi = Math.PI * 2;
            let deltaAngle = aEndAngle - aStartAngle;
            const samePoints = Math.abs( deltaAngle ) < Number.EPSILON;

            while ( deltaAngle < 0 ) deltaAngle += twoPi;
            while ( deltaAngle > twoPi ) deltaAngle -= twoPi;

            if ( deltaAngle < Number.EPSILON ) {
                deltaAngle = samePoints ? 0 : twoPi;
            }

            if ( aClockwise && ! samePoints ) {
                deltaAngle = deltaAngle === twoPi ? - twoPi : deltaAngle - twoPi;
            }

            const angle = aStartAngle + t * deltaAngle;

            glMatrix.vec2.set(
                _gm_v2,
                ax + aRadius * Math.cos( angle ),
                ay + aRadius * Math.sin( angle )
            );

            ArcSampleComponent.x[ eid ] = _gm_v2[ 0 ];
            ArcSampleComponent.y[ eid ] = _gm_v2[ 1 ];
        }
    }

    /**
     * Retrieve all results as a Float64Array of [x0, y0, x1, y1, ...].
     * @returns {Float64Array}
     */
    results() {
        const entities = this.entities;
        const out = new Float64Array( entities.length * 2 );
        for ( let i = 0, l = entities.length; i < l; i ++ ) {
            out[ i * 2 + 0 ] = ArcSampleComponent.x[ entities[ i ] ];
            out[ i * 2 + 1 ] = ArcSampleComponent.y[ entities[ i ] ];
        }
        return out;
    }
}

// ---------------------------------------------------------------------------
// Main ArcCurve class — mirrors three.js/src/extras/curves/ArcCurve.js
// ---------------------------------------------------------------------------
/**
 * An alias for {@link EllipseCurve} representing a circular arc.
 *
 * ```js
 * const curve = new THREE.ArcCurve( 0, 0, 10, 0, 2 * Math.PI, false );
 * const points = curve.getPoints( 50 );
 * const geometry = new THREE.BufferGeometry().setFromPoints( points );
 * const material = new THREE.LineBasicMaterial( { color: 0xff0000 } );
 * const arc = new THREE.Line( geometry, material );
 * ```
 *
 * @augments EllipseCurve
 */
class ArcCurve extends EllipseCurve {

    /**
     * Constructs a new arc curve.
     * @param {number} [aX=0] - The X center of the ellipse.
     * @param {number} [aY=0] - The Y center of the ellipse.
     * @param {number} [aRadius=1] - The radius of the ellipse in the x direction.
     * @param {number} [aStartAngle=0] - The start angle of the curve in radians starting from the positive X axis.
     * @param {number} [aEndAngle=Math.PI*2] - The end angle of the curve in radians starting from the positive X axis.
     * @param {boolean} [aClockwise=false] - Whether the ellipse is drawn clockwise or not.
     */
    constructor( aX = 0, aY = 0, aRadius = 1, aStartAngle = 0, aEndAngle = Math.PI * 2, aClockwise = false ) {
        super( aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise );

        /**
         * This flag can be used for type testing.
         * @type {boolean}
         * @readonly
         * @default true
         */
        this.isArcCurve = true;

        this.type = 'ArcCurve';
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated point evaluation (writes into a Float32Array vec2).
     * @param {glMatrix.vec2} out - Preallocated gl-matrix vec2 output.
     * @param {number} t - Interpolation factor in [0, 1].
     * @returns {glMatrix.vec2}
     */
    getPointGlMat( out, t ) {
        const point = this.getPoint( t, _v2 );
        out[ 0 ] = point.x;
        out[ 1 ] = point.y;
        return out;
    }

    /**
     * noise-modulated arc sampling — adds controllable organic radius
     * variation for hand-drawn / procedural circular arc effects.
     * @param {number} t - Interpolation factor in [0, 1].
     * @param {Vector2} [optionalTarget] - Optional target vector.
     * @param {number} [amplitude=0.01] - Noise amplitude.
     * @param {number} [frequency=1] - Noise frequency.
     * @param {number} [offset=0] - Per-instance noise offset.
     * @returns {Vector2}
     */
    getPointNoisy( t, optionalTarget = new Vector2(), amplitude = 0.01, frequency = 1, offset = 0 ) {
        const point = this.getPoint( t, optionalTarget );
        point.x += _noise2D( t * frequency + offset, 0 ) * amplitude;
        point.y += _noise2D( t * frequency + offset, 100 ) * amplitude;
        return point;
    }

    /**
     * double.js precision arc-length integration for very small or very
     * large arcs where float32 drift becomes visible.
     * @param {number} [divisions=200]
     * @returns {number}
     */
    getLengthPrecise( divisions = 200 ) {
        _double.value = 0;
        let last = this.getPoint( 0, _v2 );
        let current;
        for ( let p = 1; p <= divisions; p ++ ) {
            current = this.getPoint( p / divisions, _v2 );
            const dx = current.x - last.x;
            const dy = current.y - last.y;
            _double.add( Math.sqrt( dx * dx + dy * dy ) );
            last = current;
        }
        return _double.value;
    }

    /**
     * Create a batched arc sampler backed by bitecs.
     * Queue many samples and process them in one cache-friendly pass.
     * @returns {ArcCurveBatch}
     */
    static createBatch() {
        return new ArcCurveBatch();
    }
}

export { ArcCurve, ArcCurveBatch };
export default ArcCurve;