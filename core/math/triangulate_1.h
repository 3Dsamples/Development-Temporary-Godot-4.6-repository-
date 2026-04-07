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
 * Features 32-byte alignment for SIMD-accelerated geometric processing.
 */
class ET_ALIGN_32 Triangulate {
public:
	/**
	 * triangulate()
	 * 
	 * Performs ear-clipping on a 2D polygon defined by FixedMath coordinates.
	 * Returns a vector of indices representing the resulting triangles.
	 * Designed for high-frequency execution within Warp kernels at 120 FPS.
	 */
	static bool triangulate(const Vector<Vector2f> &p_polygon, Vector<int> &r_triangles);

private:
	/**
	 * _is_snip()
	 * 
	 * Internal deterministic helper to check if a triangle formed by indices
	 * (u, v, w) is a valid "ear" to be snipped.
	 */
	static _FORCE_INLINE_ bool _is_snip(const Vector<Vector2f> &p_polygon, int u, int v, int w, int n, const Vector<int> &V);
};

#endif // TRIANGULATE_H

--- END OF FILE core/math/triangulate.h ---
