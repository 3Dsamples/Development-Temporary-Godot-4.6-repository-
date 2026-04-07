--- START OF FILE core/math/geometry_interaction_kernel.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * struct TouchProbe
 * 
 * Represents a contact point (finger, gripper, sensor).
 * Strictly aligned for bit-perfect multi-point interaction.
 */
struct ET_ALIGN_32 TouchProbe {
	Vector3f position;
	Vector3f direction;
	FixedMathCore force;
	FixedMathCore radius;
};

/**
 * Warp Kernel: MultiPointPressureKernel
 * 
 * Simulates sophisticated "Poke" and "Deform" behaviors across multiple contact points.
 * 1. Accumulates pressure tensors from all active probes.
 * 2. Applies non-linear displacement based on material stiffness.
 * 3. Handles "Balloon" effect by storing displacement for the restoration wave.
 */
void multi_point_pressure_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_fatigue,
		const TouchProbe *p_probes,
		uint32_t p_probe_count,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_elasticity,
		const FixedMathCore &p_delta) {

	Vector3f total_displacement;
	FixedMathCore total_force_mag = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	for (uint32_t i = 0; i < p_probe_count; i++) {
		const TouchProbe &probe = p_probes[i];
		Vector3f diff = r_position - probe.position;
		FixedMathCore dist_sq = diff.length_squared();
		FixedMathCore r2 = probe.radius * probe.radius;

		if (dist_sq < r2) {
			FixedMathCore dist = Math::sqrt(dist_sq);
			FixedMathCore weight = (probe.radius - dist) / probe.radius;
			// Quadratic falloff for natural "Flesh" compression
			FixedMathCore intensity = probe.force * (weight * weight);
			
			// Displace along probe direction (Poke) or surface normal (Balloon expansion)
			total_displacement += probe.direction * (intensity / (p_stiffness + one));
			total_force_mag += intensity;
		}
	}

	if (total_force_mag.get_raw() > 0) {
		r_position += total_displacement * p_delta;
		// Accumulate material fatigue bit-perfectly
		r_fatigue += total_force_mag * FixedMathCore(4294967LL, true); // 0.001 damage
		// Kill velocity during active hold to simulate viscous damping
		r_velocity *= FixedMathCore(2147483648LL, true); // 0.5 damping
	}
}

/**
 * Warp Kernel: PinchInteractionKernel
 * 
 * Sophisticated Behavior: Simulates "Pinching" (e.g., between thumb and forefinger).
 * Attracts vertices toward the midpoint of two probes while compressing the volume.
 */
void pinch_interaction_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_point_a,
		const Vector3f &p_point_b,
		const FixedMathCore &p_force,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_delta) {

	Vector3f mid = (p_point_a + p_point_b) * MathConstants<FixedMathCore>::half();
	Vector3f diff = r_position - mid;
	FixedMathCore dist_sq = diff.length_squared();
	FixedMathCore r2 = p_radius * p_radius;

	if (dist_sq < r2) {
		FixedMathCore dist = Math::sqrt(dist_sq);
		FixedMathCore weight = (p_radius - dist) / p_radius;
		
		// Direction toward the pinch center
		Vector3f dir_to_mid = (mid - r_position).normalized();
		
		// Compression: vertices move toward midpoint
		r_velocity += dir_to_mid * (p_force * weight * p_delta);
		
		// Bulge Effect: simulate volume conservation by pushing outward on the perpendicular axis
		Vector3f pinch_axis = (p_point_b - p_point_a).normalized();
		Vector3f bulge_dir = diff.cross(pinch_axis).cross(pinch_axis).normalized();
		r_velocity += bulge_dir * (p_force * weight * MathConstants<FixedMathCore>::half() * p_delta);
	}
}

/**
 * Warp Kernel: ViscoelasticRestorationKernel
 * 
 * Step 3: Returns the material to its rest-state (The Balloon/Flesh return).
 * Uses bit-perfect Hooke's Law and Viscous Damping.
 */
void viscoelastic_restoration_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_rest_position,
		const FixedMathCore &p_stiffness,
		const FixedMathCore &p_viscosity,
		const FixedMathCore &p_delta) {

	// F_spring = -k * x
	Vector3f displacement = r_position - p_rest_position;
	Vector3f spring_force = displacement * (-p_stiffness);

	// F_damping = -c * v
	Vector3f damping_force = r_velocity * (-p_viscosity);

	// Integrate bit-perfectly
	r_velocity += (spring_force + damping_force) * p_delta;
	r_position += r_velocity * p_delta;
}

/**
 * compute_tactile_feedback()
 * 
 * Machine Perception: Calculates the resistance vector for robotic sensors.
 * Returns a bit-perfect force vector representing the surface "hardness".
 */
Vector3f compute_tactile_feedback(
		const Vector3f &p_probe_pos,
		const Face3f *p_faces,
		uint64_t p_face_count,
		const FixedMathCore &p_radius) {

	Vector3f resistance_acc;
	FixedMathCore r2 = p_radius * p_radius;
	uint32_t contacts = 0;

	for (uint64_t i = 0; i < p_face_count; i++) {
		const Face3f &f = p_faces[i];
		Vector3f closest = f.get_closest_point(p_probe_pos);
		FixedMathCore dist_sq = (p_probe_pos - closest).length_squared();

		if (dist_sq < r2) {
			FixedMathCore dist = Math::sqrt(dist_sq);
			FixedMathCore weight = (p_radius - dist) / p_radius;
			
			// Resistance = Normal * Surface_Tension * (1.0 - Fatigue)
			FixedMathCore integrity = MathConstants<FixedMathCore>::one() - f.structural_fatigue;
			resistance_acc += f.get_normal() * (f.surface_tension * integrity * weight);
			contacts++;
		}
	}

	if (contacts > 0) {
		return resistance_acc / FixedMathCore(static_cast<int64_t>(contacts), false);
	}
	return Vector3f();
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_interaction_kernel.cpp ---
