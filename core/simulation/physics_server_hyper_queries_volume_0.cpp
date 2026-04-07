--- START OF FILE core/simulation/physics_server_hyper_queries_volume.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: VolumeOccupancyKernel
 * 
 * Determines how much of a specific volume (AABB) is occupied by rigid or deformable bodies.
 * Used for robotic "path-clearing" sensors and AI spatial awareness.
 * Optimized for EnTT component streams.
 */
void volume_occupancy_kernel(
		const BigIntCore &p_index,
		const AABBf &p_query_volume,
		const BigIntCore &p_qsx, const BigIntCore &p_qsy, const BigIntCore &p_qsz,
		const Body *p_bodies,
		uint64_t p_body_count,
		FixedMathCore &r_occupancy_ratio) {

	FixedMathCore occupied_volume = MathConstants<FixedMathCore>::zero();
	FixedMathCore total_query_vol = p_query_volume.get_volume();

	for (uint64_t i = 0; i < p_body_count; i++) {
		const Body &b = p_bodies[i];
		if (!b.active) continue;

		// Resolve world-space AABB intersection across BigInt sectors
		AABBf body_world_aabb = b.transform.xform_aabb(b.mesh_deterministic->get_aabb());
		
		// Translate body AABB to query sector space
		Vector3f sector_offset = wp::calculate_galactic_offset(p_qsx, p_qsy, p_qsz, b.sector_x, b.sector_y, b.sector_z);
		body_world_aabb.position += sector_offset;

		if (p_query_volume.intersects(body_world_aabb)) {
			AABBf intersection = p_query_volume.intersection(body_world_aabb);
			occupied_volume += intersection.get_volume();
		}
	}

	r_occupancy_ratio = wp::clamp(occupied_volume / total_query_vol, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

/**
 * query_sdf_gradient()
 * 
 * Machine Perception: Samples the Signed Distance Field around a point.
 * Returns a vector pointing away from the nearest physical obstruction.
 * Essential for robotic obstacle avoidance and "Poke" response alignment.
 */
Vector3f PhysicsServerHyper::query_sdf_gradient(
		const Vector3f &p_point,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_sample_step) {

	auto get_dist = [&](const Vector3f &pos) -> FixedMathCore {
		List<RID> neighbors;
		broadphase.query_radius(pos, p_sx, p_sy, p_sz, p_sample_step * FixedMathCore(10LL, false), neighbors);
		
		FixedMathCore min_d = FixedMathCore(2147483647LL, false); // Infinity
		for (const typename List<RID>::Element *E = neighbors.front(); E; E = E->next()) {
			Body *b = body_owner.get_or_null(E->get());
			// Transform point to local body space for precise distance
			Vector3f local_p = b->transform.xform_inv(pos);
			FixedMathCore d = b->mesh_deterministic->get_aabb().distance_to_point(local_p);
			if (d < min_d) min_d = d;
		}
		return min_d;
	};

	// Central Difference Gradient Estimation
	FixedMathCore dx = get_dist(p_point + Vector3f(p_sample_step, 0, 0)) - get_dist(p_point - Vector3f(p_sample_step, 0, 0));
	FixedMathCore dy = get_dist(p_point + Vector3f(0, p_sample_step, 0)) - get_dist(p_point - Vector3f(0, p_sample_step, 0));
	FixedMathCore dz = get_dist(p_point + Vector3f(0, 0, p_sample_step)) - get_dist(p_point - Vector3f(0, 0, p_sample_step));

	return Vector3f(dx, dy, dz).normalized();
}

/**
 * process_volume_triggers_parallel()
 * 
 * The 120 FPS Parallel Sweep for perception zones.
 * Iterates through all active machine "Triggers" in the EnTT registry.
 */
void PhysicsServerHyper::process_volume_triggers_parallel(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t trigger_count = registry.get_stream<AABBf>(COMPONENT_TRIGGER_VOLUME).size();
	if (trigger_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = trigger_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? trigger_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			for (uint64_t i = start; i < end; i++) {
				FixedMathCore ratio;
				// Execute the occupancy check for this specific trigger
				volume_occupancy_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					registry.get_stream<AABBf>(COMPONENT_TRIGGER_VOLUME)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z)[i],
					body_owner.get_raw_data_ptr(), // Zero-copy access to bodies
					body_owner.get_rid_count(),
					ratio
				);
				
				// If occupancy crosses threshold, queue perception event
				if (ratio > FixedMathCore(4294967LL, true)) { // > 0.001
					// MessageQueue::get_singleton()->push_notification(...)
				}
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_queries_volume.cpp ---
