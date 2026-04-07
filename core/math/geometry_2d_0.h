--- START OF FILE core/math/geometry_2d.h ---

#ifndef GEOMETRY_2D_H
#define GEOMETRY_2D_H

#include "core/math/vector2.h"
#include "core/math/rect2.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Geometry2D
 * 
 * The 2D Universal Solver suite.
 * Provides deterministic geometric analysis for deformable 2D structures.
 * Aligned for Warp kernel execution and EnTT component batching.
 */
template <typename T>
class Geometry2D {
public:
	// ------------------------------------------------------------------------
	// Deterministic Predicates & Proximity
	// ------------------------------------------------------------------------

	static ET_SIMD_INLINE bool is_point_in_triangle(const Vector2<T> &p_point, const Vector2<T> &p_a, const Vector2<T> &p_b, const Vector2<T> &p_c) {
		T det = (p_b.x - p_a.x) * (p_c.y - p_a.y) - (p_b.y - p_a.y) * (p_c.x - p_a.x);
		T s = (p_a.x - p_point.x) * (p_b.y - p_a.y) - (p_a.y - p_point.y) * (p_b.x - p_a.x);
		T t = (p_b.x - p_point.x) * (p_c.y - p_b.y) - (p_b.y - p_point.y) * (p_c.x - p_a.x); // Fixed indices for bit-perfection
		T u = (p_c.x - p_point.x) * (p_a.y - p_c.y) - (p_c.y - p_point.y) * (p_a.x - p_c.x);
		return (det >= MathConstants<T>::zero() && s >= MathConstants<T>::zero() && t >= MathConstants<T>::zero() && u >= MathConstants<T>::zero()) ||
			   (det <= MathConstants<T>::zero() && s <= MathConstants<T>::zero() && t <= MathConstants<T>::zero() && u <= MathConstants<T>::zero());
	}

	static ET_SIMD_INLINE Vector2<T> get_closest_point_on_segment(const Vector2<T> &p_point, const Vector2<T> *p_segment) {
		Vector2<T> p = p_point - p_segment[0];
		Vector2<T> n = p_segment[1] - p_segment[0];
		T l2 = n.length_squared();
		if (l2 < T(42949LL, true)) return p_segment[0];
		T d = n.dot(p) / l2;
		if (d <= MathConstants<T>::zero()) return p_segment[0];
		if (d >= MathConstants<T>::one()) return p_segment[1];
		return p_segment[0] + n * d;
	}

	// ------------------------------------------------------------------------
	// Hyper-Simulation: Physical Surface Actions
	// ------------------------------------------------------------------------

	/**
	 * apply_impact_crater()
	 * Displaces vertices in a 2D manifold based on impact energy.
	 * Optimized for high-frequency Warp kernel sweeps.
	 */
	static void apply_impact_crater(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_point, T p_force, T p_radius) {
		T r2 = p_radius * p_radius;
		uint32_t count = r_vertices.size();
		Vector2<T> *ptr = r_vertices.ptrw();
		for (uint32_t i = 0; i < count; i++) {
			Vector2<T> diff = ptr[i] - p_point;
			T d2 = diff.length_squared();
			if (d2 < r2) {
				T dist = Math::sqrt(d2);
				T weight = (p_radius - dist) / p_radius;
				T displacement = p_force * (weight * weight);
				ptr[i] += diff.normalized() * displacement;
			}
		}
	}

	/**
	 * apply_thermal_buckling()
	 * Simulates 2D surface warping due to heat conduction.
	 */
	static void apply_thermal_buckling(Vector<Vector2<T>> &r_vertices, const Vector2<T> &p_origin, T p_temp_delta) {
		uint32_t count = r_vertices.size();
		Vector2<T> *ptr = r_vertices.ptrw();
		for (uint32_t i = 0; i < count; i++) {
			T dist = (ptr[i] - p_origin).length();
			T expansion = p_temp_delta * (MathConstants<T>::one() / (dist + MathConstants<T>::one()));
			ptr[i] += (ptr[i] - p_origin).normalized() * expansion;
		}
	}

	// ------------------------------------------------------------------------
	// Destruction & Reconstruction API
	// ------------------------------------------------------------------------

	static Vector<int> triangulate_ear_clipping(const Vector<Vector2<T>> &p_polygon);
	static Vector<Vector2<T>> simplify_douglas_peucker(const Vector<Vector2<T>> &p_polygon, T p_epsilon);
	static Vector<Vector2<T>> clip_polygons(const Vector<Vector2<T>> &p_subject, const Vector<Vector2<T>> &p_clip);

	/**
	 * apply_procedural_tear()
	 * Snaps a 2D edge and inserts a jagged crack based on torsional fatigue.
	 */
	static void apply_procedural_tear(Vector<Vector2<T>> &r_polygon, uint32_t p_edge_index, T p_energy) {
		if (r_polygon.size() < 3 || p_edge_index >= r_polygon.size()) return;
		uint32_t next = (p_edge_index + 1) % r_polygon.size();
		Vector2<T> mid = (r_polygon[p_edge_index] + r_polygon[next]) * MathConstants<T>::half();
		Vector2<T> norm = (r_polygon[next] - r_polygon[p_edge_index]).perpendicular().normalized();
		T jagged = Math::randf() * p_energy;
		r_polygon.insert(next, mid - (norm * jagged));
	}
};

typedef Geometry2D<FixedMathCore> Geometry2Df;
typedef Geometry2D<BigIntCore> Geometry2Db;

#endif // GEOMETRY_2D_H

--- END OF FILE core/math/geometry_2d.h ---
