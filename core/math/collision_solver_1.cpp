--- START OF FILE core/math/collision_solver.cpp ---

#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/os/memory.h"
#include <algorithm>

template class CollisionSolver<FixedMathCore>;

/**
 * solve_swept_sphere_vs_face()
 * 
 * Computes exact Time-Of-Impact (TOI) between a moving sphere and a static triangle.
 * fully implements face interior, edge cylinder, and vertex sphere checks.
 */
template <typename T>
bool CollisionSolver<T>::solve_swept_sphere_vs_face(
		const Vector3<T> &p_from, 
		const Vector3<T> &p_to, 
		T p_radius, 
		const Face3<T> &p_face, 
		CollisionResult &r_result) {

	Vector3<T> velocity = p_to - p_from;
	T v_sq = velocity.length_squared();
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	if (v_sq.get_raw() == 0) {
		Vector3<T> closest = p_face.get_closest_point(p_from);
		T dist_sq = (p_from - closest).length_squared();
		if (dist_sq <= p_radius * p_radius) {
			r_result.collided = true;
			r_result.time_of_impact = zero;
			r_result.contact_point = closest;
			r_result.contact_normal = (p_from - closest).normalized();
			return true;
		}
		return false;
	}

	Plane<T> plane = p_face.get_plane();
	T dist_start = plane.distance_to(p_from);
	T dist_end = plane.distance_to(p_to);

	if (dist_start < -p_radius && dist_end < -p_radius) return false;
	if (dist_start > p_radius && dist_end > p_radius) return false;

	r_result.time_of_impact = one;
	bool hit = false;

	// 1. Check Face Interior
	T t_plane = zero;
	if (dist_start > p_radius && dist_end < p_radius) {
		t_plane = (p_radius - dist_start) / (dist_end - dist_start);
	} else if (dist_start < -p_radius && dist_end > -p_radius) {
		t_plane = (-p_radius - dist_start) / (dist_end - dist_start);
	}

	if (t_plane >= zero && t_plane <= one) {
		Vector3<T> pos_at_toi = p_from + velocity * t_plane;
		Vector3<T> plane_contact = pos_at_toi - plane.normal * p_radius;

		if (p_face.intersects_ray(pos_at_toi, -plane.normal, nullptr)) {
			r_result.collided = true;
			r_result.time_of_impact = t_plane;
			r_result.contact_point = plane_contact;
			r_result.contact_normal = plane.normal;
			hit = true;
		}
	}

	// 2. Check Vertices (Swept sphere vs stationary spheres)
	for (int i = 0; i < 3; i++) {
		Vector3<T> p = p_face.vertex[i];
		Vector3<T> m = p_from - p;
		T b = m.dot(velocity);
		T c = m.length_squared() - (p_radius * p_radius);
		
		if (c.get_raw() > 0 && b.get_raw() > 0) continue;
		T discr = b * b - v_sq * c;
		if (discr.get_raw() < 0) continue;

		T t_vert = (-b - Math::sqrt(discr)) / v_sq;
		if (t_vert >= zero && t_vert < r_result.time_of_impact) {
			r_result.collided = true;
			r_result.time_of_impact = t_vert;
			r_result.contact_point = p;
			r_result.contact_normal = (p_from + velocity * t_vert - p).normalized();
			hit = true;
		}
	}

	// 3. Check Edges (Swept sphere vs stationary cylinders)
	for (int i = 0; i < 3; i++) {
		Vector3<T> p1 = p_face.vertex[i];
		Vector3<T> p2 = p_face.vertex[(i + 1) % 3];
		Vector3<T> edge = p2 - p1;
		T edge_len_sq = edge.length_squared();
		
		if (edge_len_sq.get_raw() == 0) continue;

		Vector3<T> m = p_from - p1;
		T d1 = edge.dot(m);
		T d2 = edge.dot(velocity);

		T a_quad = edge_len_sq * v_sq - d2 * d2;
		T b_quad = edge_len_sq * m.dot(velocity) - d1 * d2;
		T c_quad = edge_len_sq * m.length_squared() - d1 * d1 - p_radius * p_radius * edge_len_sq;

		if (a_quad.get_raw() == 0) continue; // Moving parallel to edge

		T discr = b_quad * b_quad - a_quad * c_quad;
		if (discr.get_raw() < 0) continue;

		T t_edge = (-b_quad - Math::sqrt(discr)) / a_quad;
		if (t_edge >= zero && t_edge < r_result.time_of_impact) {
			T f = (d1 + d2 * t_edge) / edge_len_sq;
			if (f >= zero && f <= one) {
				r_result.collided = true;
				r_result.time_of_impact = t_edge;
				r_result.contact_point = p1 + edge * f;
				r_result.contact_normal = (p_from + velocity * t_edge - r_result.contact_point).normalized();
				hit = true;
			}
		}
	}

	return hit;
}

/**
 * solve_swept_aabb_vs_aabb()
 * 
 * Computes exact TOI for moving volumes using the Slab Method on the Minkowski Difference.
 */
template <typename T>
bool CollisionSolver<T>::solve_swept_aabb_vs_aabb(
		const AABB<T> &p_box_a, const Vector3<T> &p_vel_a,
		const AABB<T> &p_box_b, const Vector3<T> &p_vel_b,
		CollisionResult &r_result) {

	Vector3<T> rel_v = p_vel_a - p_vel_b;
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	if (rel_v.length_squared().get_raw() == 0) {
		if (p_box_a.intersects(p_box_b)) {
			r_result.collided = true;
			r_result.time_of_impact = zero;
			r_result.contact_normal = Vector3<T>(one, zero, zero); // Default rejection
			return true;
		}
		return false;
	}

	AABB<T> expanded_b;
	expanded_b.position = p_box_b.position - p_box_a.size;
	expanded_b.size = p_box_b.size + p_box_a.size;

	T t_min = -T(2147483647LL, false);
	T t_max = T(2147483647LL, false);
	Vector3<T> normal_result;

	for (int i = 0; i < 3; i++) {
		T v = rel_v[i];
		T min_dist = expanded_b.position[i] - p_box_a.position[i];
		T max_dist = (expanded_b.position[i] + expanded_b.size[i]) - p_box_a.position[i];

		if (v.get_raw() == 0) {
			if (min_dist.get_raw() > 0 || max_dist.get_raw() < 0) return false;
		} else {
			T t1 = min_dist / v;
			T t2 = max_dist / v;
			T n1 = T(-1LL, false);
			T n2 = T(1LL, false);

			if (t1 > t2) {
				T tmp = t1; t1 = t2; t2 = tmp;
				T tmp_n = n1; n1 = n2; n2 = tmp_n;
			}
			
			if (t1 > t_min) {
				t_min = t1;
				normal_result = Vector3<T>();
				normal_result[i] = n1;
			}
			
			if (t2 < t_max) {
				t_max = t2;
			}
			
			if (t_min > t_max) return false;
		}
	}

	if (t_min >= zero && t_min <= one) {
		r_result.collided = true;
		r_result.time_of_impact = t_min;
		r_result.contact_normal = normal_result;
		// Point computation simplified: center of intersection bounds
		r_result.contact_point = p_box_a.position + rel_v * t_min + p_box_a.size * MathConstants<T>::half();
		return true;
	}

	return false;
}

template <typename T>
Vector3<T> CollisionSolver<T>::_get_support(const Vector<Vector3<T>> &p_hull, const Vector3<T> &p_direction) {
	T max_dot = -T(2147483647LL, false);
	int best_idx = 0;
	for (uint32_t i = 0; i < p_hull.size(); i++) {
		T d = p_hull[i].dot(p_direction);
		if (d > max_dot) {
			max_dot = d;
			best_idx = i;
		}
	}
	return p_hull[best_idx];
}

--- END OF FILE core/math/collision_solver.cpp ---
