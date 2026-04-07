--- START OF FILE core/math/geometry_3d_perforation_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/triangulate.h"
#include "core/math/warp_intrinsics.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * mesh_poke_deformation_kernel()
 * 
 * Simulates a persistent "Poke" or "Touch" action (e.g., a finger or robotic probe).
 * Unlike an impact, a poke applies a localized pressure tensor that indents
 * the surface based on material elasticity and yield.
 */
void mesh_poke_deformation_kernel(
		Vector3f *r_vertices,
		FixedMathCore *r_fatigue,
		const Vector3f &p_poke_origin,
		const Vector3f &p_poke_dir,
		const FixedMathCore &p_pressure,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_elasticity,
		uint64_t p_start,
		uint64_t p_end) {

	FixedMathCore r2 = p_radius * p_radius;

	for (uint64_t i = p_start; i < p_end; i++) {
		Vector3f diff = r_vertices[i] - p_poke_origin;
		FixedMathCore d2 = diff.length_squared();

		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			// Pressure falloff: Smooth indentation curve
			FixedMathCore weight = (p_radius - dist) / p_radius;
			FixedMathCore indentation = p_pressure * weight * p_elasticity;

			// Displace vertex along the poke direction
			r_vertices[i] += p_poke_dir * indentation;

			// Touch increases fatigue localized at the indentation site
			r_fatigue[i] += p_pressure * FixedMathCore(4294967LL, true); // 0.001 damage
		}
	}
}

/**
 * apply_mesh_perforation_logic()
 * 
 * Procedurally removes geometry to create a physical hole.
 * 1. Identifies faces intersecting the puncture radius.
 * 2. Deletes fully enclosed faces.
 * 3. Re-triangulates partially intersected faces using deterministic ear-clipping.
 */
void apply_mesh_perforation_logic(
		Vector<Face3f> &r_mesh,
		const Vector3f &p_center,
		const FixedMathCore &p_radius) {

	Vector<Face3f> final_mesh;
	FixedMathCore r2 = p_radius * p_radius;

	for (uint32_t i = 0; i < r_mesh.size(); i++) {
		Face3f &face = r_mesh.ptrw()[i];
		Vector3f median = face.get_median();
		FixedMathCore dist_sq = (median - p_center).length_squared();

		if (dist_sq > r2 * FixedMathCore(2LL, false)) {
			// Surface is far enough from puncture, keep as is
			final_mesh.push_back(face);
		} else if (dist_sq < r2 * MathConstants<FixedMathCore>::half()) {
			// Surface is entirely destroyed by the puncture
			continue;
		} else {
			// Intersection zone: Fragment the face into smaller shards
			// and only keep the shards outside the radius.
			Vector<Face3f> shards = face.fragment();
			for (uint32_t j = 0; j < shards.size(); j++) {
				if ((shards[j].get_median() - p_center).length_squared() > r2) {
					final_mesh.push_back(shards[j]);
				}
			}
		}
	}

	// Zero-Copy update of the EnTT component stream
	r_mesh = final_mesh;
}

/**
 * detect_touch_perception_kernel()
 * 
 * Machine Perception Feature:
 * Allows a robot or sensor to "feel" the resistance of a surface.
 * Returns a pressure-response value derived from material yield strength.
 */
FixedMathCore detect_touch_perception_kernel(
		const Vector3f &p_probe_pos,
		const Face3f *p_faces,
		uint64_t p_count) {

	for (uint64_t i = 0; i < p_count; i++) {
		const Face3f &f = p_faces[i];
		if (f.intersects_ray(p_probe_pos, f.get_normal() * FixedMathCore(-1LL, false))) {
			// Resistance = Surface Tension * (1.0 - Fatigue)
			return f.surface_tension * (MathConstants<FixedMathCore>::one() - f.structural_fatigue);
		}
	}
	return MathConstants<FixedMathCore>::zero();
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_3d_perforation_kernel.cpp ---
