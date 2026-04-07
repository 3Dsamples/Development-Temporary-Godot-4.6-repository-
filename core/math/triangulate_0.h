--- START OF FILE core/math/triangulate.h ---

#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/math/vector2.h"
#include "src/fixed_math_core.h"

/**
 * Triangulate Class
 * 
 * Provides deterministic 2D polygon triangulation.
 * Optimized for real-time mesh reconstruction in destructible environments.
 * Uses FixedMathCore to guarantee consistent ear-clipping results across all hardware.
 */
class Triangulate {
public:
	/**
	 * triangulate()
	 * 
	 * Performs ear-clipping on a 2D polygon defined by FixedMath coordinates.
	 * Returns a vector of indices representing the resulting triangles.
	 * Designed for high-frequency execution within Warp kernels.
	 */
	static bool triangulate(const Vector<Vector2f> &p_polygon, Vector<int> &r_triangles);

private:
	// Internal helper to determine if a triangle "ear" can be snipped deterministically.
	static ET_SIMD_INLINE bool _is_snip(const Vector<Vector2f> &p_polygon, int u, int v, int w, int n, const Vector<int> &V);
};

#endif // TRIANGULATE_H

--- END OF FILE core/math/triangulate.h ---
