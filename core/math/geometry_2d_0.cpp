--- START OF FILE core/math/geometry_2d.cpp ---

#include "core/math/geometry_2d.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 2D geometric suite for the Universal Solver backend.
 * These symbols enable EnTT to manage 2D component data while Warp kernels 
 * invoke these routines for batch-oriented 2D mesh reconstruction.
 */
template class Geometry2D<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect 2D Physics
template class Geometry2D<BigIntCore>;    // TIER_MACRO: Discrete 2D Region Logic

// ============================================================================
// Ear-Clipping Triangulation (Deterministic Reconstruction)
// ============================================================================

template <typename T>
Vector<int> Geometry2D<T>::triangulate_ear_clipping(const Vector<Vector2<T>> &p_polygon) {
	Vector<int> indices;
	int n = p_polygon.size();
	if (n < 3) return indices;

	Vector<int> v_indices(n);
	T area = MathConstants<T>::zero();
	for (int i = 0, j = n - 1; i < n; j = i++) {
		area += (p_polygon[j].x + p_polygon[i].x) * (p_polygon[j].y - p_polygon[i].y);
	}

	if (area > MathConstants<T>::zero()) {
		for (int v = 0; v < n; v++) v_indices.ptrw()[v] = v;
	} else {
		for (int v = 0; v < n; v++) v_indices.ptrw()[v] = (n - 1) - v;
	}

	int nv = n;
	int count = 2 * nv;
	for (int v = nv - 1; nv > 2;) {
		if (count-- <= 0) return indices;

		int u = v; if (nv <= u) u = 0;
		v = u + 1; if (nv <= v) v = 0;
		int w = v + 1; if (nv <= w) w = 0;

		bool snip = true;
		Vector2<T> A = p_polygon[v_indices[u]];
		Vector2<T> B = p_polygon[v_indices[v]];
		Vector2<T> C = p_polygon[v_indices[w]];

		// FixedMath convexity check
		if (T(42949LL, true) > (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) {
			snip = false;
		} else {
			for (int p = 0; p < nv; p++) {
				if ((p == u) || (p == v) || (p == w)) continue;
				if (is_point_in_triangle(p_polygon[v_indices[p]], A, B, C)) {
					snip = false;
					break;
				}
			}
		}

		if (snip) {
			indices.push_back(v_indices[u]);
			indices.push_back(v_indices[v]);
			indices.push_back(v_indices[w]);
			for (int s = v, t = v + 1; t < nv; s++, t++) v_indices.ptrw()[s] = v_indices[t];
			nv--;
			count = 2 * nv;
		}
	}
	return indices;
}

// ============================================================================
// Douglas-Peucker Simplification (2D Automatic LOD)
// ============================================================================

template <typename T>
static void _simplify_recursive(const Vector<Vector2<T>> &p_points, int p_first, int p_last, T p_epsilon_sq, Vector<bool> &r_keep) {
	T dmax_sq = MathConstants<T>::zero();
	int index = p_first;

	for (int i = p_first + 1; i < p_last; i++) {
		Vector2<T> p = p_points[i];
		Vector2<T> a = p_points[p_first];
		Vector2<T> b = p_points[p_last];
		T d_sq = Geometry2D<T>::get_closest_point_on_segment(p, &a).distance_squared_to(p);
		if (d_sq > dmax_sq) {
			index = i;
			dmax_sq = d_sq;
		}
	}

	if (dmax_sq > p_epsilon_sq) {
		r_keep.ptrw()[index] = true;
		_simplify_recursive(p_points, p_first, index, p_epsilon_sq, r_keep);
		_simplify_recursive(p_points, index, p_last, p_epsilon_sq, r_keep);
	}
}

template <typename T>
Vector<Vector2<T>> Geometry2D<T>::simplify_douglas_peucker(const Vector<Vector2<T>> &p_polygon, T p_epsilon) {
	int n = p_polygon.size();
	if (n < 3) return p_polygon;

	Vector<bool> keep(n);
	for (int i = 0; i < n; i++) keep.ptrw()[i] = false;
	keep.ptrw()[0] = true;
	keep.ptrw()[n - 1] = true;

	_simplify_recursive(p_polygon, 0, n - 1, p_epsilon * p_epsilon, keep);

	Vector<Vector2<T>> res;
	for (int i = 0; i < n; i++) {
		if (keep[i]) res.push_back(p_polygon[i]);
	}
	return res;
}

// ============================================================================
// Sutherland-Hodgman Clipping (Procedural Holes)
// ============================================================================

template <typename T>
Vector<Vector2<T>> Geometry2D<T>::clip_polygons(const Vector<Vector2<T>> &p_subject, const Vector<Vector2<T>> &p_clip) {
	Vector<Vector2<T>> output = p_subject;

	for (uint32_t j = 0; j < p_clip.size(); j++) {
		Vector<Vector2<T>> input = output;
		output.clear();
		if (input.is_empty()) break;

		Vector2<T> s = input[input.size() - 1];
		Vector2<T> edge_start = p_clip[j];
		Vector2<T> edge_end = p_clip[(j + 1) % p_clip.size()];

		for (uint32_t i = 0; i < input.size(); i++) {
			Vector2<T> e = input[i];
			// Using cross product for half-plane side test
			T side_e = (edge_end.x - edge_start.x) * (e.y - edge_start.y) - (edge_end.y - edge_start.y) * (e.x - edge_start.x);
			T side_s = (edge_end.x - edge_start.x) * (s.y - edge_start.y) - (edge_end.y - edge_start.y) * (s.x - edge_start.x);

			if (side_e >= MathConstants<T>::zero()) {
				if (side_s < MathConstants<T>::zero()) {
					T t = side_s / (side_s - side_e);
					output.push_back(s + (e - s) * t);
				}
				output.push_back(e);
			} else if (side_s >= MathConstants<T>::zero()) {
				T t = side_s / (side_s - side_e);
				output.push_back(s + (e - s) * t);
			}
			s = e;
		}
	}
	return output;
}

--- END OF FILE core/math/geometry_2d.cpp ---
