--- START OF FILE core/math/face3.cpp ---

#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Face3 material logic for the Universal Solver.
 * These symbols enable Warp kernels to perform zero-copy batch-processing
 * on EnTT-managed geometric components with bit-perfect accuracy.
 */
template struct Face3<FixedMathCore>;

/**
 * apply_impact_deformation()
 * 
 * Simulates localized cratering and plastic displacement.
 * Moves vertices based on impact proximity and material elasticity.
 * Returns true if structural fatigue exceeds the yield limit.
 */
template <typename T>
bool Face3<T>::apply_impact_deformation(const Vector3<T> &p_impact_point, const Vector3<T> &p_force_vec, T p_radius) {
	T dist = (get_median() - p_impact_point).length();
	if (dist >= p_radius) return false;

	T falloff = (p_radius - dist) / p_radius;
	T impact_intensity = p_force_vec.length() * falloff;

	// Update material fatigue tensor
	T damage = impact_intensity / (yield_strength * FixedMathCore(100LL, false));
	structural_fatigue += damage;

	// Calculate displacement based on surface tension resistance
	T displacement_mag = (impact_intensity * falloff) / (surface_tension + MathConstants<T>::one());
	Vector3<T> displacement_vec = p_force_vec.normalized() * displacement_mag;

	vertex[0] += displacement_vec;
	vertex[1] += displacement_vec;
	vertex[2] += displacement_vec;

	_update_geometric_caches();
	return structural_fatigue >= yield_strength;
}

/**
 * apply_torsional_screw()
 * 
 * Physically twists the vertices around the face normal.
 * Generates thermal energy based on torsional friction.
 */
template <typename T>
void Face3<T>::apply_torsional_screw(T p_torque_angle) {
	Vector3<T> center = get_median();
	Vector3<T> axis = get_normal();

	T resistance = surface_tension * MathConstants<T>::half();
	T actual_rotation = p_torque_angle / (MathConstants<T>::one() + resistance);

	for (int i = 0; i < 3; i++) {
		Vector3<T> local_pos = vertex[i] - center;
		vertex[i] = center + local_pos.rotated(axis, actual_rotation);
	}

	torsional_energy += Math::abs(actual_rotation);
	// Convert friction to thermal energy in Kelvin
	thermal_state += Math::abs(p_torque_angle) * FixedMathCore(15LL, false); 
	_update_geometric_caches();
}

/**
 * apply_bending_moment()
 * 
 * Folds the face along a world-space hinge axis.
 * Only vertices on the 'positive' side of the hinge are affected.
 */
template <typename T>
void Face3<T>::apply_bending_moment(const Vector3<T> &p_pivot_origin, const Vector3<T> &p_axis, T p_angle) {
	Vector3<T> n_axis = p_axis.normalized();
	Vector3<T> fold_side_vec = n_axis.cross(normal_cache).normalized();

	for (int i = 0; i < 3; i++) {
		Vector3<T> rel = vertex[i] - p_pivot_origin;
		if (rel.dot(fold_side_vec) > MathConstants<T>::zero()) {
			vertex[i] = p_pivot_origin + rel.rotated(n_axis, p_angle);
		}
	}

	structural_fatigue += Math::abs(p_angle) * FixedMathCore(429496729LL, true); // 0.1 scale
	_update_geometric_caches();
}

/**
 * split()
 * 
 * Performs 4-way deterministic tessellation.
 * Used for procedural mesh destruction and fragment generation.
 */
template <typename T>
Vector<Face3<T>> Face3<T>::split() const {
	Vector<Face3<T>> fragments;
	Vector3<T> m01 = (vertex[0] + vertex[1]) * MathConstants<T>::half();
	Vector3<T> m12 = (vertex[1] + vertex[2]) * MathConstants<T>::half();
	Vector3<T> m20 = (vertex[2] + vertex[0]) * MathConstants<T>::half();

	fragments.push_back(Face3<T>(vertex[0], m01, m20));
	fragments.push_back(Face3<T>(vertex[1], m12, m01));
	fragments.push_back(Face3<T>(vertex[2], m20, m12));
	fragments.push_back(Face3<T>(m01, m12, m20));

	for (int i = 0; i < 4; i++) {
		fragments.ptrw()[i].surface_tension = surface_tension;
		fragments.ptrw()[i].thermal_state = thermal_state;
		fragments.ptrw()[i].structural_fatigue = structural_fatigue * MathConstants<T>::half();
	}
	return fragments;
}

/**
 * intersects_ray()
 * 
 * Deterministic Moller-Trumbore intersection algorithm.
 */
template <typename T>
bool Face3<T>::intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection) const {
	Vector3<T> edge1 = vertex[1] - vertex[0];
	Vector3<T> edge2 = vertex[2] - vertex[0];
	Vector3<T> h = p_dir.cross(edge2);
	T a = edge1.dot(h);

	if (Math::abs(a) < FixedMathCore(42949LL, true)) return false;

	T f = MathConstants<T>::one() / a;
	Vector3<T> s = p_from - vertex[0];
	T u = f * s.dot(h);
	if (u < MathConstants<T>::zero() || u > MathConstants<T>::one()) return false;

	Vector3<T> q = s.cross(edge1);
	T v = f * p_dir.dot(q);
	if (v < MathConstants<T>::zero() || u + v > MathConstants<T>::one()) return false;

	T t = f * edge2.dot(q);
	if (t > FixedMathCore(42949LL, true)) {
		if (p_intersection) *p_intersection = p_from + p_dir * t;
		return true;
	}
	return false;
}

/**
 * get_closest_point()
 * 
 * Bit-perfect proximity solver using Voronoi regions.
 */
template <typename T>
Vector3<T> Face3<T>::get_closest_point(const Vector3<T> &p_point) const {
	Vector3<T> edge0 = vertex[1] - vertex[0];
	Vector3<T> edge1 = vertex[2] - vertex[0];
	Vector3<T> v0 = vertex[0] - p_point;

	T a = edge0.dot(edge0);
	T b = edge0.dot(edge1);
	T c = edge1.dot(edge1);
	T d = edge0.dot(v0);
	T e = edge1.dot(v0);

	T det = a * c - b * b;
	T s = b * e - c * d;
	T t = b * d - a * e;

	if (s + t <= det) {
		if (s < MathConstants<T>::zero()) {
			if (t < MathConstants<T>::zero()) {
				if (d < MathConstants<T>::zero()) {
					s = CLAMP(-d / a, MathConstants<T>::zero(), MathConstants<T>::one());
					t = MathConstants<T>::zero();
				} else {
					s = MathConstants<T>::zero();
					t = CLAMP(-e / c, MathConstants<T>::zero(), MathConstants<T>::one());
				}
			} else {
				s = MathConstants<T>::zero();
				t = CLAMP(-e / c, MathConstants<T>::zero(), MathConstants<T>::one());
			}
		} else if (t < MathConstants<T>::zero()) {
			s = CLAMP(-d / a, MathConstants<T>::zero(), MathConstants<T>::one());
			t = MathConstants<T>::zero();
		} else {
			T invDet = MathConstants<T>::one() / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < MathConstants<T>::zero()) {
			T tmp0 = b + d;
			T tmp1 = c + e;
			if (tmp1 > tmp0) {
				T numer = tmp1 - tmp0;
				T denom = a - FixedMathCore(2LL, false) * b + c;
				s = CLAMP(numer / denom, MathConstants<T>::zero(), MathConstants<T>::one());
				t = MathConstants<T>::one() - s;
			} else {
				t = CLAMP(-e / c, MathConstants<T>::zero(), MathConstants<T>::one());
				s = MathConstants<T>::zero();
			}
		} else if (t < MathConstants<T>::zero()) {
			if (a + d > b + e) {
				T numer = c + e - b - d;
				T denom = a - FixedMathCore(2LL, false) * b + c;
				s = CLAMP(numer / denom, MathConstants<T>::zero(), MathConstants<T>::one());
				t = MathConstants<T>::one() - s;
			} else {
				s = CLAMP(-e / c, MathConstants<T>::zero(), MathConstants<T>::one());
				t = MathConstants<T>::zero();
			}
		} else {
			T numer = c + e - b - d;
			T denom = a - FixedMathCore(2LL, false) * b + c;
			s = CLAMP(numer / denom, MathConstants<T>::zero(), MathConstants<T>::one());
			t = MathConstants<T>::one() - s;
		}
	}

	return vertex[0] + edge0 * s + edge1 * t;
}

--- END OF FILE core/math/face3.cpp ---
