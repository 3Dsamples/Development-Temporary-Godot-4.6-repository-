--- START OF FILE core/math/collision_solver.cpp ---

#include "core/math/collision_solver.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"

/**
 * Explicit Template Instantiation
 * 
 * Compiles the machine code for the deterministic physics tier.
 */
template class CollisionSolver<FixedMathCore>;

// ============================================================================
// Narrow-phase implementation (GJK-EPA)
// ============================================================================

/**
 * _get_support()
 * 
 * Computes the Minkowski support point in a specific direction.
 * S = Support(HullA, Dir) - Support(HullB, -Dir)
 */
template <typename T>
Vector3<T> CollisionSolver<T>::_get_support(const Vector<Vector3<T>> &p_hull, const Vector3<T> &p_direction) {
	T max_dot = -T(2147483647LL, false); // Negative Infinity
	int best_idx = 0;
	for (int i = 0; i < p_hull.size(); i++) {
		T dot_val = p_hull[i].dot(p_direction);
		if (dot_val > max_dot) {
			max_dot = dot_val;
			best_idx = i;
		}
	}
	return p_hull[best_idx];
}

template <typename T>
bool CollisionSolver<T>::gjk_epa_solve(
		const Vector<Vector3<T>> &p_hull_a, const Transform3D<T> &p_xform_a,
		const Vector<Vector3<T>> &p_hull_b, const Transform3D<T> &p_xform_b,
		CollisionResult &r_result) {

	// 1. GJK Phase: Determine if shapes overlap
	Vector3<T> simplex[4];
	uint32_t simplex_count = 0;

	auto get_minkowski_support = [&](const Vector3<T> &p_dir) {
		Vector3<T> dir_a = p_xform_a.basis.inverse().xform(p_dir);
		Vector3<T> dir_b = p_xform_b.basis.inverse().xform(-p_dir);
		return p_xform_a.xform(CollisionSolver<T>::_get_support(p_hull_a, dir_a)) - 
		       p_xform_b.xform(CollisionSolver<T>::_get_support(p_hull_b, dir_b));
	};

	Vector3<T> d = p_xform_b.origin - p_xform_a.origin;
	if (d.length_squared().get_raw() == 0) d = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero());

	simplex[0] = get_minkowski_support(d);
	simplex_count = 1;
	d = -simplex[0];

	for (int iter = 0; iter < 32; iter++) {
		Vector3<T> a = get_minkowski_support(d);
		if (a.dot(d) < MathConstants<T>::zero()) return false; // No intersection

		simplex[simplex_count++] = a;

		// Sub-simplex processing
		if (simplex_count == 2) {
			Vector3<T> AB = simplex[0] - simplex[1];
			Vector3<T> AO = -simplex[1];
			d = AB.cross(AO).cross(AB);
		} else if (simplex_count == 3) {
			Vector3<T> A = simplex[2];
			Vector3<T> B = simplex[1];
			Vector3<T> C = simplex[0];
			Vector3<T> AB = B - A;
			Vector3<T> AC = C - A;
			Vector3<T> ABC = AB.cross(AC);
			Vector3<T> AO = -A;
			if (ABC.cross(AC).dot(AO) > MathConstants<T>::zero()) {
				d = AC.cross(AO).cross(AC);
				simplex[0] = C; simplex[1] = A; simplex_count = 2;
			} else if (AB.cross(ABC).dot(AO) > MathConstants<T>::zero()) {
				d = AB.cross(AO).cross(AB);
				simplex[0] = B; simplex[1] = A; simplex_count = 2;
			} else {
				if (ABC.dot(AO) > MathConstants<T>::zero()) d = ABC;
				else { d = -ABC; simplex[0] = B; simplex[1] = C; }
			}
		} else if (simplex_count == 4) {
			// 2. EPA Phase: Resolve penetration depth and normal
			r_result.collided = true;
			// (EPA implementation continues here with Face and Edge refinement)
			r_result.contact_normal = d.normalized();
			r_result.penetration_depth = MathConstants<T>::one(); // Placeholder for refined EPA depth
			return true;
		}
	}
	return false;
}

// ============================================================================
// Continuous Collision Detection (Bit-Perfect TOI)
// ============================================================================

template <typename T>
bool CollisionSolver<T>::solve_swept_sphere_vs_face(
		const Vector3<T> &p_from, 
		const Vector3<T> &p_to, 
		T p_radius, 
		const Face3<T> &p_face, 
		CollisionResult &r_result) {

	Vector3<T> v = p_to - p_from;
	T v_sq = v.length_squared();
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	// Static Check
	Vector3<T> closest = p_face.get_closest_point(p_from);
	if ((p_from - closest).length_squared() <= p_radius * p_radius) {
		r_result.collided = true;
		r_result.time_of_impact = zero;
		r_result.contact_point = closest;
		r_result.contact_normal = (p_from - closest).normalized();
		return true;
	}

	Plane<T> plane = p_face.get_plane();
	T dist_start = plane.distance_to(p_from);
	T dist_end = plane.distance_to(p_to);

	// Case 1: Sphere crosses plane
	if (Math::abs(dist_start) > p_radius && (dist_start * dist_end) < zero) {
		T t = (dist_start > zero ? dist_start - p_radius : dist_start + p_radius) / (dist_start - dist_end);
		Vector3<T> pos_at_toi = p_from + v * t;
		if (p_face.has_point(pos_at_toi - plane.normal * (dist_start > zero ? p_radius : -p_radius), T(CMP_EPSILON_RAW, true))) {
			r_result.collided = true;
			r_result.time_of_impact = t;
			r_result.contact_point = pos_at_toi - plane.normal * p_radius;
			r_result.contact_normal = plane.normal;
			return true;
		}
	}

	// Case 2: Vertices (Swept sphere vs Points)
	T best_t = one;
	bool hit = false;
	for (int i = 0; i < 3; i++) {
		Vector3<T> m = p_from - p_face.vertex[i];
		T b = m.dot(v);
		T c = m.length_squared() - p_radius * p_radius;
		T discr = b * b - v_sq * c;
		if (discr.get_raw() >= 0) {
			T t = (-b - Math::sqrt(discr)) / v_sq;
			if (t >= zero && t < best_t) {
				best_t = t;
				r_result.contact_normal = (p_from + v * t - p_face.vertex[i]).normalized();
				r_result.contact_point = p_face.vertex[i];
				hit = true;
			}
		}
	}

	// Case 3: Edges (Swept sphere vs Segments)
	for (int i = 0; i < 3; i++) {
		Vector3<T> p1 = p_face.vertex[i];
		Vector3<T> p2 = p_face.vertex[(i + 1) % 3];
		Vector3<T> edge = p2 - p1;
		Vector3<T> m = p_from - p1;
		T e_sq = edge.length_squared();
		T ev = edge.dot(v);
		T em = edge.dot(m);
		T a = e_sq * v_sq - ev * ev;
		T b = e_sq * m.dot(v) - em * ev;
		T c = e_sq * m.length_squared() - em * em - p_radius * p_radius * e_sq;
		T discr = b * b - a * c;
		if (discr.get_raw() >= 0) {
			T t = (-b - Math::sqrt(discr)) / a;
			T s = (em + ev * t) / e_sq;
			if (t >= zero && t < best_t && s >= zero && s <= one) {
				best_t = t;
				r_result.contact_point = p1 + edge * s;
				r_result.contact_normal = (p_from + v * t - r_result.contact_point).normalized();
				hit = true;
			}
		}
	}

	if (hit) {
		r_result.collided = true;
		r_result.time_of_impact = best_t;
		return true;
	}
	return false;
}

template <typename T>
bool CollisionSolver<T>::solve_swept_aabb_vs_aabb(
		const AABB<T> &p_box_a, const Vector3<T> &p_vel_a,
		const AABB<T> &p_box_b, const Vector3<T> &p_vel_b,
		CollisionResult &r_result) {

	Vector3<T> v = p_vel_a - p_vel_b;
	AABB<T> target;
	target.position = p_box_b.position - p_box_a.size;
	target.size = p_box_b.size + p_box_a.size;

	T t_min = -T(2147483647LL, false);
	T t_max = T(2147483647LL, false);
	T zero = MathConstants<T>::zero();

	for (int i = 0; i < 3; i++) {
		if (v[i].get_raw() == 0) {
			if (p_box_a.position[i] < target.position[i] || p_box_a.position[i] > target.position[i] + target.size[i]) return false;
		} else {
			T t1 = (target.position[i] - p_box_a.position[i]) / v[i];
			T t2 = (target.position[i] + target.size[i] - p_box_a.position[i]) / v[i];
			if (t1 > t2) std::swap(t1, t2);
			if (t1 > t_min) t_min = t1;
			if (t2 < t_max) t_max = t2;
			if (t_min > t_max) return false;
		}
	}

	if (t_min >= zero && t_min <= MathConstants<T>::one()) {
		r_result.collided = true;
		r_result.time_of_impact = t_min;
		return true;
	}
	return false;
}

--- END OF FILE core/math/collision_solver.cpp ---
