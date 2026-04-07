--- START OF FILE core/math/collision_solver.h ---

#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "core/math/vector3.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/math/transform_3d.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * CollisionSolver
 * 
 * Master template for deterministic collision detection.
 * Provides Narrow-phase and Swept-Volume (CCD) routines.
 * Aligned to 32 bytes for Warp kernel performance.
 */
template <typename T>
class ET_ALIGN_32 CollisionSolver {
public:
	/**
	 * CollisionResult
	 * Bit-perfect manifold data returned by the solver.
	 */
	struct ET_ALIGN_32 CollisionResult {
		bool collided = false;
		T time_of_impact;    // Range [0.0, 1.0]
		Vector3<T> contact_point;
		Vector3<T> contact_normal;
		T penetration_depth;

		_FORCE_INLINE_ CollisionResult() : 
			time_of_impact(MathConstants<T>::one()),
			penetration_depth(MathConstants<T>::zero()) {}
	};

	// ------------------------------------------------------------------------
	// Continuous Collision Detection (CCD) Kernels
	// ------------------------------------------------------------------------

	/**
	 * solve_swept_sphere_vs_face()
	 * 
	 * Calculates the exact bit-perfect moment a moving sphere intersects a static triangle.
	 * Solves for TOI in the interval [0, 1].
	 */
	static bool solve_swept_sphere_vs_face(
			const Vector3<T> &p_from,
			const Vector3<T> &p_to,
			T p_radius,
			const Face3<T> &p_face,
			CollisionResult &r_result);

	/**
	 * solve_swept_aabb_vs_aabb()
	 * 
	 * Uses the Slab Method on the Minkowski Difference to resolve TOI for two 
	 * moving volumes. Essential for high-speed spaceship broadphase.
	 */
	static bool solve_swept_aabb_vs_aabb(
			const AABB<T> &p_box_a, const Vector3<T> &p_vel_a,
			const AABB<T> &p_box_b, const Vector3<T> &p_vel_b,
			CollisionResult &r_result);

	// ------------------------------------------------------------------------
	// Narrow-phase API (Convex Hulls)
	// ------------------------------------------------------------------------

	/**
	 * gjk_epa_solve()
	 * 
	 * Complete implementation of GJK (Gilbert-Johnson-Keerthi) and 
	 * EPA (Expanding Polytope Algorithm) using strictly Software-Defined Arithmetic.
	 * Returns bit-identical penetration depths and normals on all simulation nodes.
	 */
	static bool gjk_epa_solve(
			const Vector<Vector3<T>> &p_hull_a, const Transform3D<T> &p_xform_a,
			const Vector<Vector3<T>> &p_hull_b, const Transform3D<T> &p_xform_b,
			CollisionResult &r_result);

	// ------------------------------------------------------------------------
	// Deterministic Static Tests
	// ------------------------------------------------------------------------

	static _FORCE_INLINE_ bool test_aabb_overlap(const AABB<T> &p_a, const AABB<T> &p_b) {
		return p_a.intersects(p_b);
	}

private:
	/**
	 * _get_support()
	 * Minkowski support mapping for convex hulls.
	 */
	static Vector3<T> _get_support(const Vector<Vector3<T>> &p_hull, const Vector3<T> &p_direction);
};

// Typedef for the deterministic 120 FPS physics tier
typedef CollisionSolver<FixedMathCore> CollisionSolverf;

#endif // COLLISION_SOLVER_H

--- END OF FILE core/math/collision_solver.h ---
