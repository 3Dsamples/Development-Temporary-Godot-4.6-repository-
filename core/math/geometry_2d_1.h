--- START OF FILE core/math/geometry_2d.h ---

#ifndef GEOMETRY_2D_H
#define GEOMETRY_2D_H

#include "core/math/vector2.h"
#include "core/math/rect2.h"
#include "core/templates/vector.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Geometry2D
 * 
 * Static geometric analyzer for deterministic 2D simulations.
 * Engineered for EnTT SoA vertex streams and Warp kernel execution.
 * strictly uses Software-Defined Arithmetic for absolute coherence.
 */
template <typename T>
class Geometry2D {
public:
	// ------------------------------------------------------------------------
	// Deterministic Geometric Predicates
	// ------------------------------------------------------------------------

	/**
	 * is_point_in_triangle()
	 * Uses bit-perfect cross product checks to determine containment.
	 */
	static _FORCE_INLINE_ bool is_point_in_triangle(const Vector2<T> &p_point, const Vector2<T> &p_a, const Vector2<T> &p_b, const Vector2<T> &p_c) {
		T det = (p_b.x - p_a.x) * (p_c.y - p_a.y) - (p_b.y - p_a.y) * (p_c.x - p_a.x);
		T s = (p_a.x - p_point.x) * (p_b.y - p_a.y) - (p_a.y - p_point.y) * (p_b.x - p_a.x);
		T t = (p_b.x - p_point.x) * (p_c.y - p_b.y) - (p_b.y - p_point.y) * (p_c.x - p_b.x);
		T u = (p_c.x - p_point.x) * (p_a.y - p_c.y) - (p_c.y - p_point.y) * (p_a.x - p_c.x);

		if (det > MathConstants<T>::zero()) {
			return s >= MathConstants<T>::zero() && t >= MathConstants<T>::zero() && u >= MathConstants<T>::zero();
		} else {
			return s <= MathConstants<T>::zero() && t <= MathConstants<T>::zero() && u <= MathConstants<T>::zero();
		}
	}

	static _FORCE_INLINE_ Vector2<T> get_closest_point_on_segment(const Vector2<T> &p_point, const Vector2<T> *p_segment) {
		Vector2<T> ss = p_segment[1] - p_segment[0];
		T l2 = ss.length_squared();
		if (unlikely(l2 == MathConstants<T>::zero())) return p_segment[0];
		
		T t = (p_point - p_segment[0]).dot(ss) / l2;
		if (t <= MathConstants<T>::zero()) return p_segment[0];
		if (t >= MathConstants<T>::one()) return p_segment[1];
		return p_segment[0] + ss * t;
	}

	// ------------------------------------------------------------------------
	// Hyper-Simulation physical Surface actions
	// ------------------------------------------------------------------------

	/**
	 * apply_impact_crater()
	 * Simulates 2D plastic surface displacement from a point impact.
	 * Optimized for Warp-style parallel sweeps.
	 */
	static void apply_impact_crater(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_point, T p_force, T p_radius) {
		uint32_t count = r_vertices.size();
		Vector2<T> *ptr = r_vertices.ptrw();
		T r2 = p_radius * p_radius;

		for (uint32_t i = 0; i < count; i++) {
			Vector2<T> diff = ptr[i] - p_point;
			T d2 = diff.length_squared();
			if (d2 < r2) {
				T dist = Math::sqrt(d2);
				T weight = (p_radius - dist) / p_radius;
				T disp = p_force * (weight * weight); // Quadratic falloff
				if (dist > MathConstants<T>::zero()) {
					ptr[i] += diff.normalized() * disp;
				}
			}
		}
	}

	/**
	 * apply_thermal_buckling()
	 * Simulates surface warping due to heat conduction (e.g. atmospheric entry).
	 */
	static void apply_thermal_buckling(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_origin, T p_temp_delta) {
		uint32_t count = r_vertices.size();
		Vector2<T> *ptr = r_vertices.ptrw();
		T k_expansion(42949LL, true); // 0.00001 expansion coeff

		for (uint32_t i = 0; i < count; i++) {
			Vector2<T> diff = ptr[i] - p_origin;
			T dist = diff.length();
			T expansion = (p_temp_delta / (dist + MathConstants<T>::one())) * k_expansion;
			ptr[i] += diff.normalized() * expansion;
		}
	}

	/**
	 * apply_procedural_tear()
	 * Inserts a jagged fracture into a 2D mesh based on torsional stress.
	 */
	static void apply_procedural_tear(Vector<Vector2<T>> &r_polygon, uint32_t p_edge_idx, T p_jaggedness);

	// ------------------------------------------------------------------------
	// Reconstruction & Simplification API
	// ------------------------------------------------------------------------

	/**
	 * triangulate_ear_clipping()
	 * Deterministic ear-clipping. Critical for reconstructing meshes after destruction.
	 */
	static Vector<int> triangulate_ear_clipping(const Vector<Vector2<T>> &p_polygon);

	/**
	 * simplify_douglas_peucker()
	 * Automatic LOD reduction for 2D structures. 
	 * Uses bit-perfect distance thresholds to ensure same LOD on all nodes.
	 */
	static Vector<Vector2<T>> simplify_douglas_peucker(const Vector<Vector2<T>> &p_polygon, T p_epsilon);

	/**
	 * clip_polygons()
	 * Sutherland-Hodgman implementation for bit-perfect 2D boolean ops.
	 */
	static Vector<Vector2<T>> clip_polygons(const Vector<Vector2<T>> &p_subject, const Vector<Vector2<T>> &p_clip);
};

typedef Geometry2D<FixedMathCore> Geometry2Df;
typedef Geometry2D<BigIntCore> Geometry2Db;

#endif // GEOMETRY_2D_H

--- END OF FILE core/math/geometry_2d.h ---
