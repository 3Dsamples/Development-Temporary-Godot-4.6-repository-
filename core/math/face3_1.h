--- START OF FILE core/math/face3.h ---

#ifndef FACE3_H
#define FACE3_H

#include "core/math/vector3.h"
#include "core/math/plane.h"
#include "core/math/math_defs.h"
#include "core/templates/vector.h"
#include "src/fixed_math_core.h"

/**
 * Face3
 * 
 * The foundational physical and geometric unit of the Universal Solver.
 * Beyond simple vertices, this structure simulates a solid material surface
 * capable of reacting to complex physical stressors in real-time.
 * Strictly uses FixedMathCore for all calculations to maintain bit-perfection.
 */
template <typename T>
struct ET_ALIGN_32 Face3 {
	// --- Fundamental Geometry (SIMD Aligned for Warp Kernels) ---
	Vector3<T> vertex[3];
	Vector3<T> normal_cache;
	T area_cache;

	// --- Material Physics Tensors (Universal Solver Extensions) ---
	T surface_tension;    // Material elasticity / Resistance to stretching
	T yield_strength;     // Force required for permanent plastic deformation
	T structural_fatigue; // Accumulated micro-cracks and damage [0..1]
	T thermal_state;      // Local temperature in Kelvin
	T torsional_energy;   // Internal energy stored from twisting forces
	T material_density;   // Mass per surface area unit

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------

	ET_SIMD_INLINE Face3() {
		vertex[0] = Vector3<T>();
		vertex[1] = Vector3<T>();
		vertex[2] = Vector3<T>();
		surface_tension = MathConstants<T>::one();
		yield_strength = MathConstants<T>::one();
		structural_fatigue = MathConstants<T>::zero();
		thermal_state = FixedMathCore(293LL << 32, true); // 293.15 K
		torsional_energy = MathConstants<T>::zero();
		material_density = MathConstants<T>::one();
		_update_geometric_caches();
	}

	ET_SIMD_INLINE Face3(const Vector3<T> &p_v1, const Vector3<T> &p_v2, const Vector3<T> &p_v3) {
		vertex[0] = p_v1;
		vertex[1] = p_v2;
		vertex[2] = p_v3;
		surface_tension = MathConstants<T>::one();
		yield_strength = MathConstants<T>::one();
		structural_fatigue = MathConstants<T>::zero();
		thermal_state = FixedMathCore(293LL << 32, true);
		torsional_energy = MathConstants<T>::zero();
		material_density = MathConstants<T>::one();
		_update_geometric_caches();
	}

	// ------------------------------------------------------------------------
	// Internal Logic
	// ------------------------------------------------------------------------

	ET_SIMD_INLINE void _update_geometric_caches() {
		Vector3<T> edge1 = vertex[1] - vertex[0];
		Vector3<T> edge2 = vertex[2] - vertex[0];
		Vector3<T> cross = edge1.cross(edge2);
		area_cache = cross.length() * MathConstants<T>::half();
		if (area_cache > MathConstants<T>::zero()) {
			normal_cache = cross / (area_cache * FixedMathCore(2LL, false));
		} else {
			normal_cache = Vector3<T>();
		}
	}

	ET_SIMD_INLINE Vector3<T> get_normal() const { return normal_cache; }
	ET_SIMD_INLINE T get_area() const { return area_cache; }
	ET_SIMD_INLINE Vector3<T> get_median() const { 
		T third = MathConstants<T>::one() / FixedMathCore(3LL, false);
		return (vertex[0] + vertex[1] + vertex[2]) * third; 
	}

	// ------------------------------------------------------------------------
	// Deterministic Physical Actions
	// ------------------------------------------------------------------------

	/**
	 * apply_impact_deformation()
	 * Simulates plastic/elastic cratering. Displaces vertices along the force vector.
	 * Returns true if the face exceeds yield strength and should fracture.
	 */
	bool apply_impact_deformation(const Vector3<T> &p_impact_point, const Vector3<T> &p_force_vec, T p_radius);

	/**
	 * apply_torsional_screw()
	 * Swizzles the vertices around the face normal.
	 */
	void apply_torsional_screw(T p_torque_angle);

	/**
	 * apply_bending_moment()
	 * Pivots vertices on the positive side of a hinge axis.
	 */
	void apply_bending_moment(const Vector3<T> &p_pivot_origin, const Vector3<T> &p_axis, T p_angle);

	/**
	 * split()
	 * Tessellates the face into 4 sub-triangles for procedural destruction.
	 */
	Vector<Face3<T>> split() const;

	// ------------------------------------------------------------------------
	// Intersection & Geometry
	// ------------------------------------------------------------------------

	bool intersects_ray(const Vector3<T> &p_from, const Vector3<T> &p_dir, Vector3<T> *p_intersection = nullptr) const;
	Vector3<T> get_closest_point(const Vector3<T> &p_point) const;

	// Godot UI Conversion
	operator String() const {
		return "(" + (String)vertex[0] + "), (" + (String)vertex[1] + "), (" + (String)vertex[2] + ")";
	}
};

typedef Face3<FixedMathCore> Face3f;

#endif // FACE3_H

--- END OF FILE core/math/face3.h ---
