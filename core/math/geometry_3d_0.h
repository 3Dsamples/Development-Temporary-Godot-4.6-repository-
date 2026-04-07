--- START OF FILE core/math/geometry_3d.h ---

#ifndef GEOMETRY_3D_H
#define GEOMETRY_3D_H

#include "core/math/vector3.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Geometry3D
 * 
 * The 3D Universal Solver suite.
 * Orchestrates deterministic geometric analysis and physical structural actions.
 * Aligned for Warp kernel execution and EnTT SoA batch processing.
 */
template <typename T>
class Geometry3D {
public:
	// ------------------------------------------------------------------------
	// Deterministic Proximity & Intersection
	// ------------------------------------------------------------------------

	static ET_SIMD_INLINE Vector3<T> get_closest_point_on_segment(const Vector3<T> &p_point, const Vector3<T> *p_segment) {
		Vector3<T> p = p_point - p_segment[0];
		Vector3<T> n = p_segment[1] - p_segment[0];
		T l2 = n.length_squared();
		if (l2 < T(42949LL, true)) return p_segment[0];
		T d = n.dot(p) / l2;
		if (d <= MathConstants<T>::zero()) return p_segment[0];
		if (d >= MathConstants<T>::one()) return p_segment[1];
		return p_segment[0] + n * d;
	}

	static ET_SIMD_INLINE void get_closest_points_between_segments(const Vector3<T> &p1, const Vector3<T> &q1, const Vector3<T> &p2, const Vector3<T> &q2, Vector3<T> &c1, Vector3<T> &c2) {
		Vector3<T> d1 = q1 - p1;
		Vector3<T> d2 = q2 - p2;
		Vector3<T> r = p1 - p2;
		T a = d1.length_squared();
		T e = d2.length_squared();
		T f = d2.dot(r);
		T zero = MathConstants<T>::zero();
		T one = MathConstants<T>::one();

		if (a <= T(42949LL, true) && e <= T(42949LL, true)) {
			c1 = p1; c2 = p2; return;
		}
		T s, t;
		if (a <= T(42949LL, true)) {
			s = zero;
			t = CLAMP(f / e, zero, one);
		} else {
			T c = d1.dot(r);
			if (e <= T(42949LL, true)) {
				t = zero;
				s = CLAMP(-c / a, zero, one);
			} else {
				T b = d1.dot(d2);
				T denom = a * e - b * b;
				if (denom != zero) {
					s = CLAMP((b * f - c * e) / denom, zero, one);
				} else {
					s = zero;
				}
				t = (b * s + f) / e;
				if (t < zero) {
					t = zero;
					s = CLAMP(-c / a, zero, one);
				} else if (t > one) {
					t = one;
					s = CLAMP((b - c) / a, zero, one);
				}
			}
		}
		c1 = p1 + d1 * s;
		c2 = p2 + d2 * t;
	}

	// ------------------------------------------------------------------------
	// Hyper-Simulation: Physical Structural Actions
	// ------------------------------------------------------------------------

	/**
	 * apply_impact_deformation()
	 * Displaces vertices in a 3D mesh based on kinetic energy tensors.
	 * Used for real-time cratering and plastic deformation.
	 */
	static void apply_impact_deformation(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_point, const Vector3<T> &p_direction, T p_force, T p_radius) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		T r2 = p_radius * p_radius;
		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> diff = ptr[i] - p_point;
			T d2 = diff.length_squared();
			if (d2 < r2) {
				T dist = Math::sqrt(d2);
				T weight = (p_radius - dist) / p_radius;
				T displacement = p_force * (weight * weight);
				ptr[i] += p_direction * displacement;
			}
		}
	}

	/**
	 * apply_structural_bend()
	 * Simulates structural folding along a hinge axis. 
	 * Essential for buckling simulations in micro and galactic scales.
	 */
	static void apply_structural_bend(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_pivot, const Vector3<T> &p_axis, T p_angle) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		Vector3<T> axis_n = p_axis.normalized();
		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> rel = ptr[i] - p_pivot;
			if (rel.dot(axis_n.perpendicular()) > MathConstants<T>::zero()) {
				ptr[i] = p_pivot + rel.rotated(axis_n, p_angle);
			}
		}
	}

	/**
	 * apply_torsional_screw()
	 * Rotates vertices around an axis with a torque-based gradient.
	 * Foundation for "Screwing" mechanics in the Universal Solver.
	 */
	static void apply_torsional_screw(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_origin, const Vector3<T> &p_axis, T p_torque) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		Vector3<T> axis_n = p_axis.normalized();
		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> rel = ptr[i] - p_origin;
			T dist = rel.dot(axis_n);
			ptr[i] = p_origin + rel.rotated(axis_n, p_torque * dist);
		}
	}

	// ------------------------------------------------------------------------
	// Volumetric Destruction API
	// ------------------------------------------------------------------------

	/**
	 * generate_fracture_shards()
	 * Slices a 3D volume into shards based on stochastic planes.
	 * Bit-perfect synchronization for multiplayer destruction.
	 */
	static Vector<Vector<Face3<T>>> generate_fracture_shards(const Vector<Face3<T>> &p_mesh, const Vector3<T> &p_epicenter, int p_shard_count);

	/**
	 * apply_mesh_perforation()
	 * Procedurally punches a hole in a 3D geometry by removing faces
	 * and re-triangulating edge loops.
	 */
	static void apply_mesh_perforation(Vector<Face3<T>> &r_mesh, const Vector3<T> &p_center, T p_radius);
};

typedef Geometry3D<FixedMathCore> Geometry3Df;
typedef Geometry3D<BigIntCore> Geometry3Db;

#endif // GEOMETRY_3D_H

--- END OF FILE core/math/geometry_3d.h ---
