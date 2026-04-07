--- START OF FILE core/math/collision_solver.cpp ---

#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/transform.h"

/**
 * solve_swept_sphere_vs_face()
 * 
 * Determines the exact bit-perfect moment a sphere moving from p_from to p_to 
 * intersects the triangle p_face. This prevents tunneling by solving for TOI.
 */
bool CollisionSolver::solve_swept_sphere_vs_face(
		const Vector3f &p_from,
		const Vector3f &p_to,
		FixedMathCore p_radius,
		const Face3f &p_face,
		CollisionResult &r_result) {

	Vector3f velocity = p_to - p_from;
	Vector3f normal = p_face.get_normal();

	// Calculate distance from sphere center to the plane of the face
	FixedMathCore dist_from = p_face.get_plane().distance_to(p_from);
	FixedMathCore dist_to = p_face.get_plane().distance_to(p_to);

	// If the sphere is already embedded or moving away, handle as static
	if (Math::abs(dist_from) < p_radius) {
		Vector3f closest = p_face.get_closest_point(p_from);
		FixedMathCore d2 = (p_from - closest).length_squared();
		if (d2 <= p_radius * p_radius) {
			r_result.collided = true;
			r_result.time_of_impact = MathConstants<FixedMathCore>::zero();
			r_result.contact_normal = (p_from - closest).normalized();
			r_result.contact_point = closest;
			return true;
		}
	}

	// Check if the sphere crosses the plane within this frame
	if (dist_from > p_radius && dist_to < p_radius) {
		// TOI calculation: (Radius - StartDistance) / (EndDistance - StartDistance)
		FixedMathCore toi = (p_radius - dist_from) / (dist_to - dist_from);
		
		if (toi >= MathConstants<FixedMathCore>::zero() && toi <= MathConstants<FixedMathCore>::one()) {
			Vector3f pos_at_toi = p_from + (velocity * toi);
			Vector3f contact_on_plane = pos_at_toi - (normal * p_radius);

			// Verify if the contact point at TOI lies within the triangle boundaries
			if (p_face.intersects_ray(pos_at_toi, -normal, &r_result.contact_point)) {
				r_result.collided = true;
				r_result.time_of_impact = toi;
				r_result.contact_normal = normal;
				return true;
			}
		}
	}

	return false;
}

/**
 * solve_swept_aabb_vs_aabb()
 * 
 * Uses the Minkowski Difference and AABB ray-casting to find the TOI between 
 * two moving volumes. Essential for broadphase synchronization in the 
 * Universal Solver.
 */
bool CollisionSolver::solve_swept_aabb_vs_aabb(
		const AABBf &p_box_a, const Vector3f &p_vel_a,
		const AABBf &p_box_b, const Vector3f &p_vel_b,
		CollisionResult &r_result) {

	// Relative velocity of A in the frame of B
	Vector3f rel_v = p_vel_a - p_vel_b;
	if (rel_v.length_squared() == MathConstants<FixedMathCore>::zero()) {
		if (p_box_a.intersects(p_box_b)) {
			r_result.collided = true;
			r_result.time_of_impact = MathConstants<FixedMathCore>::zero();
			return true;
		}
		return false;
	}

	// Expand Box B by the dimensions of Box A (Minkowski Sum)
	AABBf expanded_b;
	expanded_b.position = p_box_b.position - p_box_a.size;
	expanded_b.size = p_box_b.size + p_box_a.size;

	// Ray-cast relative velocity against expanded AABB
	FixedMathCore t_min = -FixedMathCore(2147483647LL, false); // Large negative
	FixedMathCore t_max = FixedMathCore(2147483647LL, false);

	for (int i = 0; i < 3; i++) {
		if (rel_v[i] == MathConstants<FixedMathCore>::zero()) {
			if (p_box_a.position[i] < expanded_b.position[i] || p_box_a.position[i] > expanded_b.position[i] + expanded_b.size[i]) {
				return false;
			}
		} else {
			FixedMathCore inv_v = MathConstants<FixedMathCore>::one() / rel_v[i];
			FixedMathCore t1 = (expanded_b.position[i] - p_box_a.position[i]) * inv_v;
			FixedMathCore t2 = (expanded_b.position[i] + expanded_b.size[i] - p_box_a.position[i]) * inv_v;

			if (t1 > t2) {
				FixedMathCore tmp = t1; t1 = t2; t2 = tmp;
			}
			if (t1 > t_min) t_min = t1;
			if (t2 < t_max) t_max = t2;
		}
	}

	if (t_min < t_max && t_min >= MathConstants<FixedMathCore>::zero() && t_min <= MathConstants<FixedMathCore>::one()) {
		r_result.collided = true;
		r_result.time_of_impact = t_min;
		return true;
	}

	return false;
}

/**
 * _get_support()
 * 
 * Minkowski support function for GJK. Returns the vertex of a convex hull 
 * furthest in the given direction. Strictly deterministic using FixedMathCore.
 */
ET_SIMD_INLINE Vector3f CollisionSolver::_get_support(const Vector<Vector3f> &p_hull, const Vector3f &p_direction) {
	FixedMathCore max_dot = -FixedMathCore(2147483647LL, false);
	int best_idx = 0;
	for (int i = 0; i < p_hull.size(); i++) {
		FixedMathCore d = p_hull[i].dot(p_direction);
		if (d > max_dot) {
			max_dot = d;
			best_idx = i;
		}
	}
	return p_hull[best_idx];
}

--- END OF FILE core/math/collision_solver.cpp ---
