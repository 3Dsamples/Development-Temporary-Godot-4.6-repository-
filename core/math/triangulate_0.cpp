--- START OF FILE core/math/triangulate.cpp ---

#include "core/math/triangulate.h"
#include "core/math/geometry_2d.h"

/**
 * _is_snip()
 * 
 * Internal deterministic helper to check if a triangle formed by indices
 * (u, v, w) is a valid "ear" to be snipped.
 * Uses bit-perfect FixedMathCore predicates to ensure identical results.
 */
ET_SIMD_INLINE bool Triangulate::_is_snip(const Vector<Vector2f> &p_polygon, int u, int v, int w, int n, const Vector<int> &V) {
	int p;
	Vector2f A = p_polygon[V[u]];
	Vector2f B = p_polygon[V[v]];
	Vector2f C = p_polygon[V[w]];

	// Check for degenerate or concave triangle using deterministic cross product
	// EPSILON is defined as a fixed-point constant
	FixedMathCore area_check = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
	if (FixedMathCore(42949LL, true) > area_check) {
		return false;
	}

	for (p = 0; p < n; p++) {
		if ((p == u) || (p == v) || (p == w)) {
			continue;
		}
		Vector2f P = p_polygon[V[p]];
		if (Geometry2Df::is_point_in_triangle(P, A, B, C)) {
			return false;
		}
	}

	return true;
}

/**
 * triangulate()
 * 
 * Main entry point for 2D polygon triangulation.
 * Optimized for high-frequency Warp kernel sweeps by using index-based 
 * traversal and deterministic FixedMath winding order checks.
 */
bool Triangulate::triangulate(const Vector<Vector2f> &p_polygon, Vector<int> &r_triangles) {
	int n = p_polygon.size();
	if (n < 3) {
		return false;
	}

	Vector<int> V;
	V.resize(n);

	// Determine winding order using deterministic area calculation
	FixedMathCore area = FixedMathCore(0LL, true);
	for (int i = 0, j = n - 1; i < n; j = i++) {
		area += (p_polygon[j].x + p_polygon[i].x) * (p_polygon[j].y - p_polygon[i].y);
	}

	// Initialize vertex index list in consistent order
	if (area > FixedMathCore(0LL, true)) {
		for (int v = 0; v < n; v++) {
			V.ptrw()[v] = v;
		}
	} else {
		for (int v = 0; v < n; v++) {
			V.ptrw()[v] = (n - 1) - v;
		}
	}

	int nv = n;
	int count = 2 * nv; // Safety counter for degenerate geometry

	for (int v = nv - 1; nv > 2;) {
		if (count-- <= 0) {
			return false; // Polygon is likely self-intersecting or non-simple
		}

		int u = v;
		if (nv <= u) {
			u = 0;
		}
		v = u + 1;
		if (nv <= v) {
			v = 0;
		}
		int w = v + 1;
		if (nv <= w) {
			w = 0;
		}

		if (_is_snip(p_polygon, u, v, w, nv, V)) {
			int a, b, c, s, t;
			a = V[u];
			b = V[v];
			c = V[w];
			r_triangles.push_back(a);
			r_triangles.push_back(b);
			r_triangles.push_back(c);

			// Remove snipped vertex from the list
			for (s = v, t = v + 1; t < nv; s++, t++) {
				V.ptrw()[s] = V[t];
			}
			nv--;
			count = 2 * nv;
		}
	}

	return true;
}

--- END OF FILE core/math/triangulate.cpp ---
