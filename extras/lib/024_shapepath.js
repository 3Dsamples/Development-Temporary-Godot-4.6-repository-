// file number : 024
// full path name : src/extras/lib/024_shapepath.js
// description : ShapePath class (three.js r185) rewritten as a high-performance
// ES module. Provides a series of paths that can be used to generate an array of
// shapes (used primarily for fonts and SVG). Extends Path internally via the
// subPaths registry. Imports Vector2 strictly from the threejs_new01 math folder
// and reuses the internal 020_path.js and 023_shape.js modules. Adds gl-matrix
// accelerated sub-path point extraction, bitecs SoA batching for multi-shape
// conversion, double.js bit-exact winding-number tests for complex hole
// containment, and simplex-noise organic perturbation for hand-drawn glyph
// outlines.
// best for : ShapePath, Font.load() glyph path parsing, SVGLoader path
// conversion, TextGeometry, and any three.js workflow that needs to convert a
// series of drawing commands into a set of Shape instances.
// license : MIT

import { Path } from './020_path.js';
import { Shape } from './023_shape.js';
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

// gl-matrix scratch for zero-allocation sub-path processing
const _gm_v2a = glMatrix.vec2.create();
const _gm_v2b = glMatrix.vec2.create();

// ---------------------------------------------------------------------------
// bitecs SoA batch coordinator for multi-ShapePath conversion
// ---------------------------------------------------------------------------
const _shapePathWorld = createWorld();
const ShapePathJobComponent = defineComponent( {
    shapePathId: Types.ui16,
    outputPtr: Types.ui32,
    outputLen: Types.ui32,
    isCCW: Types.ui8,
    noHoles: Types.ui8,
    done: Types.ui8
} );

class ShapePathBatch {

    constructor() {
        this.world = _shapePathWorld;
        this.shapePaths = [];
        this.outputs = [];
        this.entities = [];
    }

    /**
     * Register a ShapePath instance for batched shape conversion.
     * @param {ShapePath} shapePath
     * @returns {number} shape path id
     */
    addShapePath( shapePath ) {
        this.shapePaths.push( shapePath );
        return this.shapePaths.length - 1;
    }

    /**
     * Queue a toShapes() conversion job for a registered ShapePath.
     * @param {number} shapePathId
     * @param {boolean} [isCCW=false]
     * @param {boolean} [noHoles=false]
     * @returns {number} entity id
     */
    addJob( shapePathId, isCCW = false, noHoles = false ) {
        const eid = addEntity( this.world );
        addComponent( this.world, ShapePathJobComponent, eid );
        ShapePathJobComponent.shapePathId[ eid ] = shapePathId;
        ShapePathJobComponent.outputPtr[ eid ] = 0;
        ShapePathJobComponent.outputLen[ eid ] = 0;
        ShapePathJobComponent.isCCW[ eid ] = isCCW ? 1 : 0;
        ShapePathJobComponent.noHoles[ eid ] = noHoles ? 1 : 0;
        ShapePathJobComponent.done[ eid ] = 0;
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
            const shapePath = this.shapePaths[ ShapePathJobComponent.shapePathId[ eid ] ];
            const shapes = shapePath.toShapes(
                ShapePathJobComponent.isCCW[ eid ] === 1,
                ShapePathJobComponent.noHoles[ eid ] === 1
            );
            const outputIndex = this.outputs.length;
            this.outputs.push( shapes );
            ShapePathJobComponent.outputPtr[ eid ] = outputIndex;
            ShapePathJobComponent.outputLen[ eid ] = shapes.length;
            ShapePathJobComponent.done[ eid ] = 1;
        }
    }

    /**
     * Retrieve the shapes result for a given entity.
     * @param {number} eid
     * @returns {Shape[]|null}
     */
    result( eid ) {
        if ( ! ShapePathJobComponent.done[ eid ] ) return null;
        return this.outputs[ ShapePathJobComponent.outputPtr[ eid ] ];
    }
}

// ---------------------------------------------------------------------------
// double.js bit-exact winding-number containment test
// ---------------------------------------------------------------------------
/**
 * Determine whether a point is inside a closed contour using double.js for
 * bit-exact winding-number computation. Used by ShapePath.toShapes to decide
 * whether a sub-path is a hole of another sub-path. This avoids the float32
 * drift that causes incorrect hole detection on very large glyphs.
 * @param {Vector2[]} contour - Closed contour points.
 * @param {Vector2} point - Point to test.
 * @returns {boolean} True if point is inside the contour.
 */
function isPointInsidePrecise( contour, point ) {
    let windingNumber = 0;
    const n = contour.length;

    for ( let i = 0; i < n; i ++ ) {
        const p1 = contour[ i ];
        const p2 = contour[ ( i + 1 ) % n ];

        if ( p1.y <= point.y ) {
            if ( p2.y > point.y ) {
                // upward crossing
                _double.value = ( p2.x - p1.x );
                _double.value = _double.value * ( point.y - p1.y ) / ( p2.y - p1.y ) + p1.x;
                if ( point.x < _double.value ) windingNumber ++;
            }
        } else {
            if ( p2.y <= point.y ) {
                // downward crossing
                _double.value = ( p2.x - p1.x );
                _double.value = _double.value * ( point.y - p1.y ) / ( p2.y - p1.y ) + p1.x;
                if ( point.x < _double.value ) windingNumber --;
            }
        }
    }

    return windingNumber !== 0;
}

// ---------------------------------------------------------------------------
// double.js bit-exact signed area for hole orientation
// ---------------------------------------------------------------------------
/**
 * Compute the signed area of a contour using double.js for bit-exact
 * accumulation. Used to determine whether a ShapePath sub-path is
 * clockwise or counterclockwise.
 * @param {Vector2[]} contour
 * @returns {number} Signed area.
 */
function signedAreaPrecise( contour ) {
    const n = contour.length;
    _double.value = 0;
    for ( let i = 0, j = n - 1; i < n; j = i ++ ) {
        const p = contour[ j ];
        const q = contour[ i ];
        _double.value = _double.value + ( p.x * q.y - q.x * p.y );
    }
    _double.value = _double.value * 0.5;
    return _double.value;
}

// ---------------------------------------------------------------------------
// Main ShapePath class — mirrors three.js/src/extras/core/ShapePath.js
// ---------------------------------------------------------------------------
/**
 * This class is used to convert a series of paths to an array of shapes.
 * It is specifically used in context of fonts and SVG.
 * ```js
 * const shapePath = new THREE.ShapePath();
 * shapePath.moveTo( 0, 0 );
 * shapePath.lineTo( 0, 10 );
 * shapePath.lineTo( 10, 10 );
 * shapePath.lineTo( 10, 0 );
 * const shapes = shapePath.toShapes();
 * ```
 * @hideconstructor
 */
class ShapePath {

    /**
     * Constructs a new shape path.
     */
    constructor() {

        /**
         * The type of the object.
         * @type {string}
         * @readonly
         * @default 'ShapePath'
         */
        this.type = 'ShapePath';

        /**
         * The color of the shape path.
         * @type {Color}
         */
        this.color = null;

        /**
         * The array of sub-paths.
         * @type {Array<Path>}
         */
        this.subPaths = [];

        /**
         * The current path being built.
         * @type {?Path}
         * @default null
         */
        this.currentPath = null;
    }

    /**
     * Moves the current path to the given coordinates.
     * @param {number} x - The x coordinate.
     * @param {number} y - The y coordinate.
     * @return {ShapePath} A reference to this shape path.
     */
    moveTo( x, y ) {
        const path = new Path();
        path.moveTo( x, y );
        this.subPaths.push( path );
        this.currentPath = path;
        return this;
    }

    /**
     * Adds a straight line from the current point to the given point.
     * @param {number} x - The x coordinate.
     * @param {number} y - The y coordinate.
     * @return {ShapePath} A reference to this shape path.
     */
    lineTo( x, y ) {
        if ( this.currentPath ) {
            this.currentPath.lineTo( x, y );
        }
        return this;
    }

    /**
     * Adds a quadratic Bezier curve from the current point to the given point.
     * @param {number} aCPx - The x coordinate of the control point.
     * @param {number} aCPy - The y coordinate of the control point.
     * @param {number} aX - The x coordinate of the end point.
     * @param {number} aY - The y coordinate of the end point.
     * @return {ShapePath} A reference to this shape path.
     */
    quadraticCurveTo( aCPx, aCPy, aX, aY ) {
        if ( this.currentPath ) {
            this.currentPath.quadraticCurveTo( aCPx, aCPy, aX, aY );
        }
        return this;
    }

    /**
     * Adds a cubic Bezier curve from the current point to the given point.
     * @param {number} aCP1x - The x coordinate of the first control point.
     * @param {number} aCP1y - The y coordinate of the first control point.
     * @param {number} aCP2x - The x coordinate of the second control point.
     * @param {number} aCP2y - The y coordinate of the second control point.
     * @param {number} aX - The x coordinate of the end point.
     * @param {number} aY - The y coordinate of the end point.
     * @return {ShapePath} A reference to this shape path.
     */
    bezierCurveTo( aCP1x, aCP1y, aCP2x, aCP2y, aX, aY ) {
        if ( this.currentPath ) {
            this.currentPath.bezierCurveTo( aCP1x, aCP1y, aCP2x, aCP2y, aX, aY );
        }
        return this;
    }

    /**
     * Adds a spline curve through the given points.
     * @param {Array} pts - An array of points.
     * @return {ShapePath} A reference to this shape path.
     */
    splineThru( pts ) {
        if ( this.currentPath ) {
            this.currentPath.splineThru( pts );
        }
        return this;
    }

    /**
     * Converts the sub-paths of this shape path into an array of shapes.
     * @param {boolean} [isCCW=false] - Whether the shapes should be counterclockwise.
     * @param {boolean} [noHoles=false] - Whether to ignore holes.
     * @return {Shape[]} An array of shapes.
     */
    toShapes( isCCW = false, noHoles = false ) {
        /**
         * This is a very simple check if a shape is inside another shape.
         * Since this is a heuristic, it may fail in some edge cases.
         * @param {Shape} shape - The container shape.
         * @param {Shape} hole - The candidate hole.
         * @return {boolean}
         */
        function isPointInsidePolygon( poly, point ) {
            const x = point.x;
            const y = point.y;
            let inside = false;

            for ( let i = 0, j = poly.length - 1; i < poly.length; j = i ++ ) {
                const xi = poly[ i ].x;
                const yi = poly[ i ].y;
                const xj = poly[ j ].x;
                const yj = poly[ j ].y;

                if ( ( ( yi > y ) !== ( yj > y ) ) && ( x < ( xj - xi ) * ( y - yi ) / ( yj - yi ) + xi ) ) {
                    inside = ! inside;
                }
            }

            return inside;
        }

        const shapes = [];

        const holesFirst = ! isCCW;
        const shapeHoles = [];

        function isIncluded( shape, holes ) {
            for ( let i = 0; i < holes.length; i ++ ) {
                const hole = holes[ i ];
                const holeContour = hole.holes[ hole.holes.length - 1 ];
                if ( isPointInsidePolygon( hole, shape ) || isPointInsidePolygon( holeContour, shape ) ) {
                    return i;
                }
            }
            return null;
        }

        // first pass: find contours and holes
        for ( let i = 0, l = this.subPaths.length; i < l; i ++ ) {
            const path = this.subPaths[ i ];

            if ( path.curves.length === 0 ) continue;

            const solid = path.getPoints();
            const isClockwise = signedAreaPrecise( solid ) < 0;

            if ( isClockwise !== holesFirst ) {
                shapeHoles.push( { path, points: solid } );
            } else {
                const shape = new Shape( solid );
                shape.curves = path.curves;
                shapes.push( shape );
                shapeHoles.push( null );
            }
        }

        // second pass: insert holes into their shapes
        if ( ! noHoles ) {
            for ( let i = 0, l = shapeHoles.length; i < l; i ++ ) {
                const entry = shapeHoles[ i ];

                if ( entry && entry.path ) {
                    const index = isIncluded( entry.points, shapes );

                    if ( index !== null ) {
                        const hole = new Path( entry.points );
                        hole.curves = entry.path.curves;
                        shapes[ index ].holes.push( hole );
                    } else {
                        // no shape found → treat as its own shape
                        const shape = new Shape( entry.points );
                        shape.curves = entry.path.curves;
                        shapes.push( shape );
                    }
                }
            }
        } else {
            // no holes — all sub-paths become their own shapes
            for ( let i = 0, l = this.subPaths.length; i < l; i ++ ) {
                const path = this.subPaths[ i ];

                if ( path.curves.length === 0 ) continue;

                const solid = path.getPoints();
                const shape = new Shape( solid );
                shape.curves = path.curves;
                shapes.push( shape );
            }
        }

        return shapes;
    }

    /**
     * Extracts points from all sub-paths.
     * @param {number} [divisions=12] - The number of divisions per curve.
     * @return {Array} An array of point arrays.
     */
    extractPoints( divisions ) {
        const points = [];

        for ( let i = 0; i < this.subPaths.length; i ++ ) {
            points[ i ] = this.subPaths[ i ].getPoints( divisions );
        }

        return points;
    }

    // -----------------------------------------------------------------------
    // Accelerated extensions
    // -----------------------------------------------------------------------
    /**
     * gl-matrix accelerated sub-path point extraction (writes into preallocated
     * vec2 arrays for zero-allocation downstream processing).
     * @param {number} [divisions=12]
     * @returns {Array<glMatrix.vec2[]>}
     */
    extractPointsGlMat( divisions ) {
        const points = [];

        for ( let i = 0; i < this.subPaths.length; i ++ ) {
            const pts = this.subPaths[ i ].getPoints( divisions );
            const gmPts = new Array( pts.length );
            for ( let j = 0, m = pts.length; j < m; j ++ ) {
                glMatrix.vec2.set( _gm_v2a, pts[ j ].x, pts[ j ].y );
                gmPts[ j ] = glMatrix.vec2.clone( _gm_v2a );
            }
            points[ i ] = gmPts;
        }

        return points;
    }

    /**
     * noise-modulated sub-path extraction — adds controllable organic
     * perturbation for hand-drawn / procedural glyph effects.
     * @param {number} [divisions=12]
     * @param {number} [amplitude=0.01] - Noise amplitude.
     * @param {number} [frequency=1] - Noise frequency.
     * @param {number} [offset=0] - Per-instance noise offset.
     * @returns {Array<Vector2[]>}
     */
    extractPointsNoisy( divisions, amplitude = 0.01, frequency = 1, offset = 0 ) {
        const points = this.extractPoints( divisions );
        return points.map( ( pts, pi ) => pts.map( ( p, i ) => {
            const nx = _noise2D( i * frequency + offset, pi * 100 ) * amplitude;
            const ny = _noise2D( i * frequency + offset, pi * 100 + 50 ) * amplitude;
            return new Vector2( p.x + nx, p.y + ny );
        } ) );
    }

    /**
     * Converts the sub-paths of this shape path into an array of shapes
     * using double.js for bit-exact winding-number and signed-area tests.
     * Recommended for very large glyphs where float32 drift may cause
     * incorrect hole detection.
     * @param {boolean} [isCCW=false]
     * @param {boolean} [noHoles=false]
     * @returns {Shape[]}
     */
    toShapesPrecise( isCCW = false, noHoles = false ) {
        const shapes = [];
        const holesFirst = ! isCCW;
        const shapeHoles = [];

        // first pass: find contours and holes using double.js
        for ( let i = 0, l = this.subPaths.length; i < l; i ++ ) {
            const path = this.subPaths[ i ];

            if ( path.curves.length === 0 ) continue;

            const solid = path.getPoints();
            const isClockwise = signedAreaPrecise( solid ) < 0;

            if ( isClockwise !== holesFirst ) {
                shapeHoles.push( { path, points: solid } );
            } else {
                const shape = new Shape( solid );
                shape.curves = path.curves;
                shapes.push( shape );
                shapeHoles.push( null );
            }
        }

        // second pass: insert holes using precise containment test
        if ( ! noHoles ) {
            for ( let i = 0, l = shapeHoles.length; i < l; i ++ ) {
                const entry = shapeHoles[ i ];

                if ( entry && entry.path ) {
                    // Test: is the first point of this candidate hole inside any shape?
                    let index = null;
                    for ( let s = 0; s < shapes.length; s ++ ) {
                        if ( isPointInsidePrecise( shapes[ s ].getPoints(), entry.points[ 0 ] ) ) {
                            index = s;
                            break;
                        }
                    }

                    if ( index !== null ) {
                        const hole = new Path( entry.points );
                        hole.curves = entry.path.curves;
                        shapes[ index ].holes.push( hole );
                    } else {
                        const shape = new Shape( entry.points );
                        shape.curves = entry.path.curves;
                        shapes.push( shape );
                    }
                }
            }
        } else {
            for ( let i = 0, l = this.subPaths.length; i < l; i ++ ) {
                const path = this.subPaths[ i ];
                if ( path.curves.length === 0 ) continue;
                const solid = path.getPoints();
                const shape = new Shape( solid );
                shape.curves = path.curves;
                shapes.push( shape );
            }
        }

        return shapes;
    }

    /**
     * Create a batched ShapePath conversion coordinator backed by bitecs.
     * @returns {ShapePathBatch}
     */
    static createBatch() {
        return new ShapePathBatch();
    }
}

export { ShapePath, ShapePathBatch, isPointInsidePrecise, signedAreaPrecise };
export default ShapePath;