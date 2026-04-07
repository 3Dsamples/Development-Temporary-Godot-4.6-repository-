--- START OF FILE core/math/face3.cpp ---

#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiation
 * 
 * Compiles the Face3 logic for the deterministic tiers.
 * - Face3f: Used for 120 FPS physics and material interaction.
 */
template struct Face3<FixedMathCore>;

// ============================================================================
// Physical Interaction Kernels (Deterministic)
// ============================================================================

/**
 * apply_impact_deformation()
 * 
 * Simulates localized plastic displacement.
 * Displaces the entire face based on proximity to the impact epicenter.
 * Updates the structural fatigue tensor using bit-perfect FixedMath.
 */
template <typename T>
bool Face3<T>::apply_impact_deformation(const Vector3<T> &p_impact_point, const Vector3<T> &p_force_vec, T p_radius) {
	Vector3<T> median = get_median();
	T dist = (median - p_impact_point).length();
	
	if (dist >= p_radius) return false;

	// Deterministic Falloff: 1.0 at center, 0.0 at radius
	T falloff = (p_radius - dist) / p_radius;
	T impact_intensity = p_force_vec.length() * falloff;

	// Update material fatigue based on force vs yield strength
	T damage = impact_intensity / (yield_strength * T(100LL));
	structural_fatigue += damage;

	// Displacement calculation relative to surface tension
	T disp_mag = (impact_intensity * falloff) / (surface_tension + MathConstants<T>::one());
	Vector3<T> disp_vec = p_force_vec.normalized() * disp_mag;

	vertex[0] += disp_vec;
	vertex[1] += disp_vec;
	vertex[2] += disp_vec;

	_update_geometric_caches();
	
	// Returns true if the material has "snapped" (Exceeded yield)
	return structural_fatigue >= yield_strength;
}

/**
 * apply_torsional_screw()
 * 
 * Simulates mechanical twisting. Rotates vertices around the triangle's 
 * median in the plane of the face. Generates heat via internal friction.
 */
template <typename T>
void Face3<T>::apply_torsional_screw(T p_torque_angle) {
	Vector3<T> center = get_median();
	Vector3<T> axis = get_normal();

	// Resistance reduces the effective twist
	T actual_rot = p_torque_angle / (MathConstants<T>::one() + surface_tension);

	for (int i = 0; i < 3; i++) {
		Vector3<T> local_v = vertex[i] - center;
		vertex[i] = center + local_v.rotated(axis, actual_rot);
	}

	// Friction increases thermal energy (Kelvin)
	thermal_state += Math::abs(p_torque_angle) * T(15LL); 
	_update_geometric_caches();
}

/**
 * apply_bending_moment()
 * 
 * Folds the face along an arbitrary world-space hinge axis.
 */
template <typename T>
void Face3<T>::apply_bending_moment(const Vector3<T> &p_pivot_origin, const Vector3<T> &p_axis, T p_angle) {
	Vector3<T> n_axis = p_axis.normalized();
	
	// Only affect vertices on the "active" side of the bending plane
	Vector3<T> bend_plane_normal = n_axis.cross(normal_cache).normalized();

	for (int i = 0; i < 3; i++) {
		Vector3<T> rel = vertex[i] - p_pivot_origin;
		if (rel.dot(bend_plane_normal) > MathConstants<T>::zero()) {
			vertex[i] = p_pivot_origin + rel.rotated(n_axis, p_angle);
		}
	}

	// Accumulate fatigue from the stress of the fold
	structural_fatigue += Math::abs(p_angle) * T(429496729LL, true); // 0.1 scale
	_update_geometric_caches();
}

/**
 * split()
 * 
 * Deterministic 4-way subdivision (Tessellation).
 * Used for real-time mesh shattering and procedural detail enhancement.
 */
template <typename T>
Vector<Face3<T>> Face3<T>::split() const {
	Vector<Face3<T>> shards;
	Vector3<T> m01 = (vertex[0] + vertex[1]) * MathConstants<T>::half();
	Vector3<T> m12 = (vertex[1] + vertex[2]) * MathConstants<T>::half();
	Vector3<T> m20 = (vertex[2] + vertex[0]) * MathConstants<T>::half();

	shards.push_back(Face3<T>(vertex[0], m01, m20));
	shards.push_back(Face3<T>(vertex[1], m12, m01));
	shards.push_back(Face3<T>(vertex[2], m20, m12));
	shards.push_back(Face3<T>(m01, m12, m20));

	// Heritage: Inherit material properties with bit-perfect precision
	for (int i = 0; i < 4; i++) {
		shards.ptrw()[i].surface_tension = surface_tension;
		shards.ptrw()[i].thermal_state = thermal_state;
		shards.ptrw()[i].yield_strength = yield_strength;
		shards.ptrw()[i].structural_fatigue = structural_fatigue;
	}
	return shards;
}

// ============================================================================
// Geometric Solvers (Bit-Perfect)
// ============================================================================

/**
 * intersects_ray()
 * 
 * Implementation of the Möller–Trumbore intersection algorithm.
 * strictly uses FixedMathCore to ensure no FP-rounding errors during raycasts.
 */
template <typename T>
bool Face3<T>::intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection) const {
	Vector3<T> e1 = vertex[1] - vertex[0];
	Vector3<T> e2 = vertex[2] - vertex[0];
	Vector3<T> h = p_dir.cross(e2);
	T a = e1.dot(h);

	// Deterministic epsilon check
	if (Math::abs(a) < T(CMP_EPSILON_RAW, true)) return false;

	T f = MathConstants<T>::one() / a;
	Vector3<T> s = p_from - vertex[0];
	T u = f * s.dot(h);
	if (u < MathConstants<T>::zero() || u > MathConstants<T>::one()) return false;

	Vector3<T> q = s.cross(e1);
	T v = f * p_dir.dot(q);
	if (v < MathConstants<T>::zero() || u + v > MathConstants<T>::one()) return false;

	T t = f * e2.dot(q);
	if (t > T(CMP_EPSILON_RAW, true)) {
		if (p_intersection) *p_intersection = p_from + p_dir * t;
		return true;
	}
	return false;
}

/**
 * get_closest_point()
 * 
 * Determines the point on the triangle nearest to p_point using Voronoi regions.
 * Bit-perfect implementation for robotic sensors and precision collisions.
 */
template <typename T>
Vector3<T> Face3<T>::get_closest_point(const Vector3<T> &p_point) const {
	Vector3<T> ab = vertex[1] - vertex[0];
	Vector3<T> ac = vertex[2] - vertex[0];
	Vector3<T> ap = p_point - vertex[0];

	T d1 = ab.dot(ap);
	T d2 = ac.dot(ap);
	if (d1 <= T(0LL) && d2 <= T(0LL)) return vertex[0];

	Vector3<T> bp = p_point - vertex[1];
	T d3 = ab.dot(bp);
	T d4 = ac.dot(bp);
	if (d3 >= T(0LL) && d4 <= d3) return vertex[1];

	T vc = d1 * d4 - d3 * d2;
	if (vc <= T(0LL) && d1 >= T(0LL) && d3 <= T(0LL)) {
		T v = d1 / (d1 - d3);
		return vertex[0] + ab * v;
	}

	Vector3<T> cp = p_point - vertex[2];
	T d5 = ab.dot(cp);
	T d6 = ac.dot(cp);
	if (d6 >= T(0LL) && d5 <= d6) return vertex[2];

	T vb = d5 * d2 - d1 * d6;
	if (vb <= T(0LL) && d2 >= T(0LL) && d6 <= T(0LL)) {
		T w = d2 / (d2 - d6);
		return vertex[0] + ac * w;
	}

	T va = d3 * d6 - d5 * d4;
	if (va <= T(0LL) && (d4 - d3) >= T(0LL) && (d5 - d6) >= T(0LL)) {
		T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return vertex[1] + (vertex[2] - vertex[1]) * w;
	}

	T den = MathConstants<T>::one() / (va + vb + vc);
	T v = vb * den;
	T w = vc * den;
	return vertex[0] + ab * v + ac * w;
}

--- END OF FILE core/math/face3.cpp ---
