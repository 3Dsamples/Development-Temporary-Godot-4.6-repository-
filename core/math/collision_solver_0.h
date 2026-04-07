--- START OF FILE core/math/collision_solver.h ---

#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "core/math/vector3.h"
#include "core/math/face3.h"
#include "core/math/aabb.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"

/**
 * CollisionSolver
 * 
 * High-performance deterministic collision engine.
 * Provides Continuous Collision Detection (CCD) to prevent tunneling 
 * at any velocity scale (Microscopic to Galactic).
 */
class ET_ALIGN_32 CollisionSolver {
public:
	struct ET_ALIGN_32 CollisionResult {
		bool collided = false;
		FixedMathCore time_of_impact; // Range [0, 1]
		Vector3f contact_point;
		Vector3f contact_normal;
		FixedMathCore penetration_depth;
	};

	// ------------------------------------------------------------------------
	// Swept-Volume Time-of-Impact (TOI) Solver
	// ------------------------------------------------------------------------

	/**
	 * solve_swept_sphere_vs_face()
	 * Calculates the exact bit-perfect moment a moving sphere hits a static triangle.
	 * Critical for 120 FPS stability of high-velocity projectiles.
	 */
	static bool solve_swept_sphere_vs_face(
			const Vector3f &p_from,
			const Vector3f &p_to,
			FixedMathCore p_radius,
			const Face3f &p_face,
			CollisionResult &r_result);

	/**
	 * solve_swept_aabb_vs_aabb()
	 * Resolves intersections between two moving bounding volumes.
	 * Foundation for deterministic broadphase-to-narrowphase transitions.
	 */
	static bool solve_swept_aabb_vs_aabb(
			const AABBf &p_box_a, const Vector3f &p_vel_a,
			const AABBf &p_box_b, const Vector3f &p_vel_b,
			CollisionResult &r_result);

	// ------------------------------------------------------------------------
	// Deterministic Static Tests
	// ------------------------------------------------------------------------

	static ET_SIMD_INLINE bool test_aabb_overlap(const AABBf &p_a, const AABBf &p_b) {
		return p_a.intersects(p_b);
	}

	/**
	 * gjk_epa_calculate_distance()
	 * Ported GJK-EPA algorithm using FixedMathCore.
	 * Returns the minimum separation or penetration between two convex hulls.
	 */
	static bool gjk_epa_solve(
			const Vector<Vector3f> &p_hull_a, const Transformf &p_xform_a,
			const Vector<Vector3f> &p_hull_b, const Transformf &p_xform_b,
			CollisionResult &r_result);

private:
	// Internal helper for Minkowski Difference calculations in Warp kernels
	static ET_SIMD_INLINE Vector3f _get_support(const Vector<Vector3f> &p_hull, const Vector3f &p_direction);
};

#endif // COLLISION_SOLVER_H

--- END OF FILE core/math/collision_solver.h ---
