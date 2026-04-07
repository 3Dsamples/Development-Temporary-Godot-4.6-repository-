--- START OF FILE core/math/geometry_2d.cpp ---

#include "core/math/geometry_2d.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"

/**
 * Explicit Template Instantiation
 * 
 * Compiles the 2D geometric analyzer for the deterministic tiers.
 */
template class Geometry2D<FixedMathCore>;
template class Geometry2D<BigIntCore>;

// ============================================================================
// Ear-Clipping Triangulation (Deterministic)
// ============================================================================

template <typename T>
Vector<int> Geometry2D<T>::triangulate_ear_clipping(const Vector<Vector2<T>> &p_polygon) {
	Vector<int> indices;
	int n = p_polygon.size();
	if (n < 3) return indices;

	Vector<int> v_indices;
	v_indices.resize(n);

	// 1. Determine winding order via Trapezoid Area calculation
	T area = MathConstants<T>::zero();
	for (int i = 0, j = n - 1; i < n; j = i++) {
		area += (p_polygon[j].x + p_polygon[i].x) * (p_polygon[j].y - p_polygon[i].y);
	}

	// 2. Initialize vertex indices in consistent clockwise/counter-clockwise order
	if (area > MathConstants<T>::zero()) {
		for (int v = 0; v < n; v++) v_indices.ptrw()[v] = v;
	} else {
		for (int v = 0; v < n; v++) v_indices.ptrw()[v] = (n - 1) - v;
	}

	// 3. Ear Clipping Loop
	int nv = n;
	int count = 2 * nv; // Safety guard for degenerate polygons

	for (int v = nv - 1; nv > 2;) {
		if (count-- <= 0) return indices; // Avoid infinite loop in non-simple polygons

		int u = v; if (nv <= u) u = 0;
		v = u + 1; if (nv <= v) v = 0;
		int w = v + 1; if (nv <= w) w = 0;

		bool is_ear = true;
		Vector2<T> A = p_polygon[v_indices[u]];
		Vector2<T> B = p_polygon[v_indices[v]];
		Vector2<T> C = p_polygon[v_indices[w]];

		// Check for convexity via cross product
		if (T(CMP_EPSILON_RAW, true) > (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) {
			is_ear = false;
		} else {
			// Ensure no other point is inside the candidate triangle
			for (int p = 0; p < nv; p++) {
				if ((p == u) || (p == v) || (p == w)) continue;
				if (is_point_in_triangle(p_polygon[v_indices[p]], A, B, C)) {
					is_ear = false;
					break;
				}
			}
		}

		if (is_ear) {
			indices.push_back(v_indices[u]);
			indices.push_back(v_indices[v]);
			indices.push_back(v_indices[w]);
			// Remove snipped vertex from list
			v_indices.remove_at(v);
			nv--;
			count = 2 * nv;
		}
	}
	return indices;
}

// ============================================================================
// Douglas-Peucker Simplification (Recursive Bit-Perfect LOD)
// ============================================================================

template <typename T>
static void _simplify_recursive(const Vector<Vector2<T>> &p_points, int p_first, int p_last, T p_epsilon_sq, Vector<bool> &r_keep) {
	T dmax_sq = MathConstants<T>::zero();
	int index = p_first;

	for (int i = p_first + 1; i < p_last; i++) {
		Vector2<T> p = p_points[i];
		Vector2<T> segment[2] = { p_points[p_first], p_points[p_last] };
		T d_sq = Geometry2D<T>::get_closest_point_on_segment(p, segment).distance_squared_to(p);
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

	Vector<bool> keep;
	keep.resize(n);
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
// Sutherland-Hodgman Clipping (Deterministic Booleans)
// ============================================================================

template <typename T>
Vector<Vector2<T>> Geometry2D<T>::clip_polygons(const Vector<Vector2<T>> &p_subject, const Vector<Vector2<T>> &p_clip) {
	Vector<Vector2<T>> output = p_subject;

	for (int j = 0; j < p_clip.size(); j++) {
		Vector<Vector2<T>> input = output;
		output.clear();
		if (input.is_empty()) break;

		Vector2<T> edge_start = p_clip[j];
		Vector2<T> edge_end = p_clip[(j + 1) % p_clip.size()];
		Vector2<T> s = input[input.size() - 1];

		for (int i = 0; i < input.size(); i++) {
			Vector2<T> e = input[i];

			// Deterministic Side-of-Line Check
			auto is_inside = [&](const Vector2<T> &p) {
				return (edge_end.x - edge_start.x) * (p.y - edge_start.y) - (edge_end.y - edge_start.y) * (p.x - edge_start.x) >= MathConstants<T>::zero();
			};

			if (is_inside(e)) {
				if (!is_inside(s)) {
					// Compute intersection using bit-perfect FixedMath
					T dx = s.x - e.x;
					T dy = s.y - e.y;
					T ex = edge_start.x - edge_end.x;
					T ey = edge_start.y - edge_end.y;
					T den = (s.x - e.x) * (edge_start.y - edge_end.y) - (s.y - e.y) * (edge_start.x - edge_end.x);
					T t = ((s.x - edge_start.x) * (edge_start.y - edge_end.y) - (s.y - edge_start.y) * (edge_start.x - edge_end.x)) / den;
					output.push_back(s + (e - s) * t);
				}
				output.push_back(e);
			} else if (is_inside(s)) {
				T den = (s.x - e.x) * (edge_start.y - edge_end.y) - (s.y - e.y) * (edge_start.x - edge_end.x);
				T t = ((s.x - edge_start.x) * (edge_start.y - edge_end.y) - (s.y - edge_start.y) * (edge_start.x - edge_end.x)) / den;
				output.push_back(s + (e - s) * t);
			}
			s = e;
		}
	}
	return output;
}

// ============================================================================
// Procedural Physics Actions
// ============================================================================

template <typename T>
void Geometry2D<T>::apply_procedural_tear(Vector<Vector2<T>> &r_polygon, uint32_t p_edge_idx, T p_jaggedness) {
	if (r_polygon.size() < 3 || p_edge_idx >= static_cast<uint32_t>(r_polygon.size())) return;

	uint32_t next = (p_edge_idx + 1) % r_polygon.size();
	Vector2<T> v1 = r_polygon[p_edge_idx];
	Vector2<T> v2 = r_polygon[next];

	Vector2<T> mid = (v1 + v2) * MathConstants<T>::half();
	Vector2<T> norm = (v2 - v1).perpendicular().normalized();

	// Deterministic entropy for jagged fracture edge
	RandomPCG pcg;
	pcg.seed(static_cast<uint64_t>(v1.x.get_raw() ^ v2.y.get_raw()));
	T noise = (pcg.randf() - MathConstants<T>::half()) * p_jaggedness;

	r_polygon.insert(next, mid + norm * noise);
}

--- END OF FILE core/math/geometry_2d.cpp ---
