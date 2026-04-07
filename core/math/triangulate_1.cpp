--- START OF FILE core/math/triangulate.cpp ---

#include "core/math/triangulate.h"
#include "core/math/geometry_2d.h"
#include "core/math/math_funcs.h"

/**
 * _is_snip()
 * 
 * Internal deterministic helper to check if a triangle formed by indices
 * (u, v, w) is a valid "ear" to be snipped.
 * Strictly uses FixedMathCore predicates to ensure identical results across nodes.
 */
bool Triangulate::_is_snip(const Vector<Vector2f> &p_polygon, int u, int v, int w, int n, const Vector<int> &V) {
	Vector2f A = p_polygon[V[u]];
	Vector2f B = p_polygon[V[v]];
	Vector2f C = p_polygon[V[w]];

	// 1. Convexity check using deterministic cross product.
	// We check if the triangle is counter-clockwise oriented relative to the indices.
	FixedMathCore val = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
	if (val <= MathConstants<FixedMathCore>::zero()) {
		return false;
	}

	// 2. Point-in-triangle check for all other remaining vertices.
	// If any other vertex is inside the ABC triangle, it's not an ear.
	for (int p = 0; p < n; p++) {
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
 * The master triangulation kernel.
 * 1. Computes polygon winding order via bit-perfect Shoelace area.
 * 2. Iteratively clips "ears" from the polygon until only one triangle remains.
 * 3. Handles degenerate or self-intersecting hulls via iteration safety caps.
 */
bool Triangulate::triangulate(const Vector<Vector2f> &p_polygon, Vector<int> &r_triangles) {
	int n = p_polygon.size();
	if (n < 3) {
		return false;
	}

	Vector<int> V;
	V.resize(n);

	// Determine winding order using the bit-perfect shoelace formula
	FixedMathCore area = MathConstants<FixedMathCore>::zero();
	for (int i = 0, j = n - 1; i < n; j = i++) {
		// (x2 + x1) * (y2 - y1)
		area += (p_polygon[j].x + p_polygon[i].x) * (p_polygon[j].y - p_polygon[i].y);
	}

	// Initialize vertex index list in a consistent orientation
	if (area > MathConstants<FixedMathCore>::zero()) {
		for (int v = 0; v < n; v++) {
			V.ptrw()[v] = v;
		}
	} else {
		for (int v = 0; v < n; v++) {
			V.ptrw()[v] = (n - 1) - v;
		}
	}

	int nv = n;
	// Iteration limit: prevent infinite loop on impossible/degenerate geometry
	int count = 2 * nv; 

	for (int v = nv - 1; nv > 2;) {
		if (count-- <= 0) {
			// Polygon is likely non-simple or self-intersecting
			return false; 
		}

		// Sliding window of three indices
		int u = v; if (nv <= u) u = 0;
		v = u + 1; if (nv <= v) v = 0;
		int w = v + 1; if (nv <= w) w = 0;

		if (_is_snip(p_polygon, u, v, w, nv, V)) {
			// We found an ear: record the triangle
			r_triangles.push_back(V[u]);
			r_triangles.push_back(V[v]);
			r_triangles.push_back(V[w]);

			// Remove the ear vertex (v) from the working list
			for (int s = v, t = v + 1; t < nv; s++, t++) {
				V.ptrw()[s] = V[t];
			}
			nv--;
			
			// Reset iteration counter for the next pass
			count = 2 * nv;
		}
	}

	return true;
}

--- END OF FILE core/math/triangulate.cpp ---
