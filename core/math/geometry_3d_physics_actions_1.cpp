--- START OF FILE core/math/geometry_3d_physics_actions.cpp ---

#include "core/math/geometry_3d.h"
#include "core/math/face3.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ImpactDeformationKernel
 * 
 * Simulates high-speed plastic flow during a collision epicenter.
 * Vertices are displaced along the impact normal based on energy density.
 * Uses a deterministic yield-check to determine if deformation is permanent.
 */
void impact_deformation_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_fatigue,
		const Vector3f &p_impact_point,
		const Vector3f &p_impact_normal,
		const FixedMathCore &p_impact_force,
		const FixedMathCore &p_radius,
		const FixedMathCore &p_yield_strength) {

	Vector3f diff = r_position - p_impact_point;
	FixedMathCore dist_sq = diff.length_squared();
	FixedMathCore r2 = p_radius * p_radius;

	if (dist_sq < r2) {
		FixedMathCore dist = Math::sqrt(dist_sq);
		FixedMathCore falloff = (p_radius - dist) / p_radius;
		FixedMathCore effective_force = p_impact_force * (falloff * falloff);

		// Material Physics: Accumulate Fatigue and resolve Plasticity
		r_fatigue += effective_force * FixedMathCore(4294967LL, true); // 0.001 scale

		if (effective_force > p_yield_strength) {
			// Permanent displacement (Plastic Flow)
			FixedMathCore displacement = (effective_force - p_yield_strength) * FixedMathCore(2147483648LL, true); // 0.5 factor
			r_position += p_impact_normal * displacement;
			// Reset velocity to prevent elastic bounce-back in the plastic zone
			r_velocity = Vector3f_ZERO;
		} else {
			// Elastic jiggle (Balloon Effect)
			r_velocity += p_impact_normal * (effective_force * FixedMathCore(429496730LL, true)); // 0.1 factor
		}
	}
}

/**
 * Warp Kernel: TorsionalStressKernel
 * 
 * Simulates the "Screwing" or twisting of a physical structure.
 * Rotates vertices around an axis based on torque magnitude and distance.
 */
void torsional_stress_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const Vector3f &p_axis_origin,
		const Vector3f &p_axis_dir,
		const FixedMathCore &p_torque,
		const FixedMathCore &p_radius) {

	Vector3f rel = r_position - p_axis_origin;
	FixedMathCore dist_proj = rel.dot(p_axis_dir);
	Vector3f projection = p_axis_dir * dist_proj;
	Vector3f rejection = rel - projection;
	
	FixedMathCore r_dist = rejection.length();
	if (r_dist > p_radius || r_dist.get_raw() == 0) return;

	FixedMathCore weight = (p_radius - r_dist) / p_radius;
	FixedMathCore angle = p_torque * weight;

	// Perform bit-perfect rotation using FixedMathCore sin/cos
	r_position = p_axis_origin + rel.rotated(p_axis_dir, angle);
	
	// Add rotational velocity (Angular Momentum Transfer)
	r_velocity += p_axis_dir.cross(rejection).normalized() * (p_torque * r_dist);
}

/**
 * Warp Kernel: VolumetricFragmentationKernel
 * 
 * Partitions mesh triangles into shards based on stochastic fracture planes.
 * Strictly uses FixedMathCore for side-of-plane checks to ensure shard 
 * boundaries are identical across all clients.
 */
void volumetric_fragmentation_kernel(
		const Face3f *p_faces,
		const Plane<FixedMathCore> *p_fracture_planes,
		uint32_t p_plane_count,
		uint32_t *r_shard_ids,
		uint64_t p_face_start,
		uint64_t p_face_end) {

	for (uint64_t i = p_face_start; i < p_face_end; i++) {
		const Face3f &face = p_faces[i];
		Vector3f median = face.get_median();
		
		uint32_t shard_id = 0;
		for (uint32_t p = 0; p < p_plane_count; p++) {
			// Deterministic Bit-Mask generation: each plane divides space in half
			if (p_fracture_planes[p].is_point_over(median)) {
				shard_id |= (1u << p);
			}
		}
		r_shard_ids[i] = shard_id;
	}
}

/**
 * execute_physical_action_wave()
 * 
 * Master orchestrator for high-frequency structural physics.
 * Batches physical interventions into Warp kernels to sustain 120 FPS.
 */
void execute_physical_action_wave(
		KernelRegistry &p_registry,
		const Vector3f &p_epicenter,
		const FixedMathCore &p_energy_mag,
		const StringName &p_action_type) {

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &fatigue_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_FATIGUE);
	
	uint64_t count = pos_stream.size();
	uint64_t chunk = count / workers;

	if (p_action_type == SNAME("impact")) {
		for (uint32_t w = 0; w < workers; w++) {
			uint64_t start = w * chunk;
			uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

			SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &fatigue_stream]() {
				for (uint64_t i = start; i < end; i++) {
					impact_deformation_kernel(
						BigIntCore(static_cast<int64_t>(i)),
						pos_stream[i],
						vel_stream[i],
						fatigue_stream[i],
						p_epicenter,
						Vector3f(0, -1, 0), // Simplified Normal
						p_energy_mag,
						FixedMathCore(5LL, false), // 5.0 unit radius
						FixedMathCore(10LL, false)  // 10.0 yield threshold
					);
				}
			}, SimulationThreadPool::PRIORITY_HIGH);
		}
	} else if (p_action_type == SNAME("torsion")) {
		// Similar parallel sweep for torsional_stress_kernel...
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_3d_physics_actions.cpp ---
