--- START OF FILE core/math/geometry_instance_deformation.cpp ---

#include "core/math/geometry_instance.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: SoftBodyElasticKernel
 * 
 * Simulates the "Balloon Effect" (Elasticity).
 * Processes a batch of vertices: applies restoration forces toward the rest position
 * and integrates velocities with damping to simulate flesh-like behavior.
 */
void soft_body_elastic_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_rest_position,
		const FixedMathCore &p_stiffness, // k
		const FixedMathCore &p_damping,   // damping factor
		const FixedMathCore &p_delta) {

	// 1. Hooke's Law: F = -k * x (Restoration toward rest position)
	Vector3f displacement = r_position - p_rest_position;
	Vector3f restoration_force = displacement * (-p_stiffness);

	// 2. Apply Force and Damping
	r_velocity += restoration_force * p_delta;
	r_velocity *= (MathConstants<FixedMathCore>::one() - (p_damping * p_delta));

	// 3. Update Position
	r_position += r_velocity * p_delta;
}

/**
 * apply_pinch_interaction()
 * 
 * Advanced Feature: Simulates "Pinching" (Opposing forces between two points).
 * Used for realistic flesh manipulation in real-time.
 */
void apply_pinch_interaction(
		Vector3f *r_vertices,
		Vector3f *r_velocities,
		uint64_t p_count,
		const Vector3f &p_point_a,
		const Vector3f &p_point_b,
		const FixedMathCore &p_force,
		const FixedMathCore &p_radius) {

	Vector3f midpoint = (p_point_a + p_point_b) * MathConstants<FixedMathCore>::half();
	FixedMathCore r2 = p_radius * p_radius;

	for (uint64_t i = 0; i < p_count; i++) {
		Vector3f diff = r_vertices[i] - midpoint;
		if (diff.length_squared() < r2) {
			// Compress vertices toward the midpoint between fingers
			FixedMathCore dist = diff.length();
			FixedMathCore weight = (p_radius - dist) / p_radius;
			r_velocities[i] += (midpoint - r_vertices[i]).normalized() * (p_force * weight);
		}
	}
}

/**
 * apply_pull_interaction()
 * 
 * Advanced Feature: "Pulling" or "Dragging" a surface area.
 */
void apply_pull_interaction(
		Vector3f *r_vertices,
		Vector3f *r_velocities,
		uint64_t p_count,
		const Vector3f &p_grab_point,
		const Vector3f &p_pull_target,
		const FixedMathCore &p_radius) {

	FixedMathCore r2 = p_radius * p_radius;
	Vector3f pull_vec = p_pull_target - p_grab_point;

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore dist_sq = (r_vertices[i] - p_grab_point).length_squared();
		if (dist_sq < r2) {
			FixedMathCore weight = MathConstants<FixedMathCore>::one() - (Math::sqrt(dist_sq) / p_radius);
			// Directly influence velocity for a reactive "Tug" feel
			r_velocities[i] += pull_vec * (weight * weight);
		}
	}
}

/**
 * resolve_flesh_tensors()
 * 
 * Specialized logic for Flesh/Buttock/Breast behavior.
 * Adjusts stiffness and damping parameters based on "Fat" or "Muscle" descriptors.
 */
void resolve_flesh_tensors(
		FixedMathCore &r_stiffness,
		FixedMathCore &r_damping,
		const StringName &p_flesh_type) {
	
	if (p_flesh_type == SNAME("breast")) {
		r_stiffness = FixedMathCore(2147483648LL, true); // Low stiffness (Soft)
		r_damping = FixedMathCore(429496729LL, true);   // 0.1 Damping
	} else if (p_flesh_type == SNAME("buttock")) {
		r_stiffness = FixedMathCore(4294967296LL, false); // Medium stiffness
		r_damping = FixedMathCore(858993459LL, true);    // 0.2 Damping
	} else {
		r_stiffness = FixedMathCore(10LL, false);        // High stiffness (Muscular)
		r_damping = FixedMathCore(2147483648LL, true);   // 0.5 Damping
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_instance_deformation.cpp ---
