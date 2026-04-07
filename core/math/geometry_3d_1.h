--- START OF FILE core/math/geometry_3d.h ---

#ifndef GEOMETRY_3D_H
#define GEOMETRY_3D_H

#include "core/math/vector3.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Geometry3D
 * 
 * Master static analyzer for deterministic 3D physical actions.
 * Engineered for EnTT component streams and high-frequency Warp execution.
 * strictly uses Software-Defined Arithmetic to prevent floating-point drift.
 */
template <typename T>
class Geometry3D {
public:
	// ------------------------------------------------------------------------
	// Deterministic Proximity API
	// ------------------------------------------------------------------------

	static _FORCE_INLINE_ Vector3<T> get_closest_point_on_segment(const Vector3<T> &p_point, const Vector3<T> *p_segment) {
		Vector3<T> ss = p_segment[1] - p_segment[0];
		T l2 = ss.length_squared();
		if (unlikely(l2 == MathConstants<T>::zero())) return p_segment[0];
		
		T t = (p_point - p_segment[0]).dot(ss) / l2;
		if (t <= MathConstants<T>::zero()) return p_segment[0];
		if (t >= MathConstants<T>::one()) return p_segment[1];
		return p_segment[0] + ss * t;
	}

	/**
	 * get_closest_points_between_segments()
	 * Bit-perfect resolve for two 3D lines. Used for high-speed cable and 
	 * beam-particle interactions.
	 */
	static void get_closest_points_between_segments(const Vector3<T> &p1, const Vector3<T> &q1, const Vector3<T> &p2, const Vector3<T> &q2, Vector3<T> &c1, Vector3<T> &c2) {
		Vector3<T> d1 = q1 - p1;
		Vector3<T> d2 = q2 - p2;
		Vector3<T> r = p1 - p2;
		T a = d1.length_squared();
		T e = d2.length_squared();
		T f = d2.dot(r);
		T zero = MathConstants<T>::zero();
		T one = MathConstants<T>::one();

		if (a <= T(CMP_EPSILON_RAW, true) && e <= T(CMP_EPSILON_RAW, true)) {
			c1 = p1; c2 = p2; return;
		}
		T s, t;
		if (a <= T(CMP_EPSILON_RAW, true)) {
			s = zero;
			t = CLAMP(f / e, zero, one);
		} else {
			T c = d1.dot(r);
			if (e <= T(CMP_EPSILON_RAW, true)) {
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
	// Hyper-Simulation: Structural Deformation Kernels
	// ------------------------------------------------------------------------

	/**
	 * apply_impact_deformation()
	 * Simulates plastic denting. Displaces vertices along impact direction.
	 * k_elasticity determines if form is permanent or returns (Balloon effect).
	 */
	static void apply_impact_deformation(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_point, const Vector3<T> &p_dir, T p_force, T p_radius, T p_elasticity) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		T r2 = p_radius * p_radius;

		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> diff = ptr[i] - p_point;
			T d2 = diff.length_squared();
			if (d2 < r2) {
				T dist = Math::sqrt(d2);
				T weight = (p_radius - dist) / p_radius;
				T disp = p_force * (weight * weight) * (MathConstants<T>::one() - p_elasticity);
				ptr[i] += p_dir * disp;
			}
		}
	}

	/**
	 * apply_structural_bend()
	 * Fold geometry along a deterministic hinge axis. 
	 * Essential for buckling simulations in ship hulls.
	 */
	static void apply_structural_bend(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_pivot, const Vector3<T> &p_axis, T p_angle) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		Vector3<T> ax = p_axis.normalized();
		Vector3<T> side_norm = ax.cross(Vector3<T>(0LL, 1LL, 0LL)).normalized();

		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> rel = ptr[i] - p_pivot;
			if (rel.dot(side_norm) > MathConstants<T>::zero()) {
				ptr[i] = p_pivot + rel.rotated(ax, p_angle);
			}
		}
	}

	/**
	 * apply_torsional_screw()
	 * Advanced Behavior: Physically twists the vertex stream around an axis.
	 */
	static void apply_torsional_screw(Vector<Vector3<T>> &r_vertices, const Vector3<T> &p_origin, const Vector3<T> &p_axis, T p_torque) {
		uint32_t count = r_vertices.size();
		Vector3<T> *ptr = r_vertices.ptrw();
		Vector3<T> ax = p_axis.normalized();

		for (uint32_t i = 0; i < count; i++) {
			Vector3<T> rel = ptr[i] - p_origin;
			T dist = rel.dot(ax);
			ptr[i] = p_origin + rel.rotated(ax, p_torque * dist);
		}
	}

	// ------------------------------------------------------------------------
	// Destruction & Perforation API
	// ------------------------------------------------------------------------

	/**
	 * generate_fracture_shards()
	 * Uses stochastic planes to slice a volume into fragments.
	 * Returns bit-perfect shard lists for EnTT entity spawning.
	 */
	static Vector<Vector<Face3<T>>> generate_fracture_shards(const Vector<Face3<T>> &p_mesh, const Vector3<T> &p_epicenter, int p_shard_count, const BigIntCore &p_seed);

	/**
	 * apply_mesh_perforation()
	 * Procedural hole punching. Removes faces and re-triangulates edge loops.
	 */
	static void apply_mesh_perforation(Vector<Face3<T>> &r_mesh, const Vector3<T> &p_center, T p_radius);
};

typedef Geometry3D<FixedMathCore> Geometry3Df;
typedef Geometry3D<BigIntCore> Geometry3Db;

#endif // GEOMETRY_3D_H

--- END OF FILE core/math/geometry_3d.h ---
