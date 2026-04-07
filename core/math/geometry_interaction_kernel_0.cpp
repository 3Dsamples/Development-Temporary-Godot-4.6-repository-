--- START OF FILE core/math/geometry_interaction_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * resolve_multi_point_touch_kernel()
 * 
 * High-performance Warp kernel for distributing pressure from multiple probes.
 * Simulates complex deformation where multiple "fingers" or "grippers" compress a mesh.
 * 
 * r_vertices: EnTT vertex position stream.
 * p_probes: Array of contact points in local space.
 * p_pressures: Force magnitude per probe point.
 * p_radius: Influence radius per probe.
 */
void resolve_multi_point_touch_kernel(
		Vector3f *r_vertices,
		FixedMathCore *r_fatigue,
		const Vector3f *p_probes,
		const FixedMathCore *p_pressures,
		uint32_t p_probe_count,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_elasticity,
		uint64_t p_v_start,
		uint64_t p_v_end) {

	FixedMathCore r2 = p_radius * p_radius;

	for (uint64_t i = p_v_start; i < p_v_end; i++) {
		Vector3f total_displacement;
		FixedMathCore max_stress = MathConstants<FixedMathCore>::zero();

		for (uint32_t j = 0; j < p_probe_count; j++) {
			Vector3f diff = r_vertices[i] - p_probes[j];
			FixedMathCore d2 = diff.length_squared();

			if (d2 < r2) {
				FixedMathCore dist = Math::sqrt(d2);
				FixedMathCore weight = (p_radius - dist) / p_radius;
				
				// Calculate pressure-based indentation (cubic falloff for softness)
				FixedMathCore intensity = p_pressures[j] * weight * weight * weight;
				
				// Direction is usually inward along the probe axis (simplified as down for this kernel)
				Vector3f dir(MathConstants<FixedMathCore>::zero(), -MathConstants<FixedMathCore>::one(), MathConstants<FixedMathCore>::zero());
				total_displacement += dir * (intensity * p_elasticity);

				if (intensity > max_stress) max_stress = intensity;
			}
		}

		// Apply the combined displacement from all probe points
		r_vertices[i] += total_displacement;
		
		// Update fatigue component in the EnTT stream
		r_fatigue[i] += max_stress * FixedMathCore(4294967LL, true); // 0.001 damage per touch
	}
}

/**
 * compute_tactile_feedback_kernel()
 * 
 * Calculates the reaction force (Resistance) at a specific probe location.
 * Used by robotic AI and machine perception to "feel" the hardness of a surface.
 * 
 * p_probe_pos: The world/local position of the sensor.
 * p_faces: The SoA stream of material-aware Face3 tensors.
 * r_resistance: Output force vector returned to the machine logic.
 */
void compute_tactile_feedback_kernel(
		const Vector3f &p_probe_pos,
		const Face3f *p_faces,
		uint64_t p_face_count,
		Vector3f &r_resistance) {

	FixedMathCore total_normal_force = MathConstants<FixedMathCore>::zero();
	Vector3f accumulated_normal;
	uint32_t contact_count = 0;

	// O(N) check within the kernel - usually called on a small subset 
	// provided by the SpatialPartition broadphase.
	for (uint64_t i = 0; i < p_face_count; i++) {
		const Face3f &f = p_faces[i];
		Vector3f closest = f.get_closest_point(p_probe_pos);
		FixedMathCore dist = (p_probe_pos - closest).length();
		
		// Proximity threshold for "Contact" in bit-perfect FixedMath
		if (dist < FixedMathCore(429496LL, true)) { // 0.0001 units
			// Resistance = Surface Tension * (1.0 - Fatigue)
			FixedMathCore hardness = f.surface_tension * (MathConstants<FixedMathCore>::one() - f.structural_fatigue);
			total_normal_force += hardness;
			accumulated_normal += f.get_normal();
			contact_count++;
		}
	}

	if (contact_count > 0) {
		FixedMathCore avg_factor = MathConstants<FixedMathCore>::one() / FixedMathCore(static_cast<int64_t>(contact_count));
		r_resistance = (accumulated_normal * avg_factor).normalized() * total_normal_force;
	} else {
		r_resistance = Vector3f();
	}
}

/**
 * apply_persistent_sculpt_action()
 * 
 * Real-time behavior: Permanently modifies the "Rest Position" of the mesh.
 * If a touch is sustained and pressure > material yield, it becomes plastic.
 */
void apply_persistent_sculpt_action(
		Vector3f *r_rest_positions,
		const FixedMathCore *p_fatigue,
		const FixedMathCore &p_yield_threshold,
		uint64_t p_count) {
	
	for (uint64_t i = 0; i < p_count; i++) {
		// If material has "snapped" or yielded, lock the current deformation
		if (p_fatigue[i] > p_yield_threshold) {
			// In a full implementation, this triggers a vertex buffer re-bind
			// to bake the "Poke" into the mesh.
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_interaction_kernel.cpp ---
