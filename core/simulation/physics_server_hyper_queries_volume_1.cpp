--- START OF FILE core/simulation/physics_server_hyper_queries_volume.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * volume_occupancy_kernel()
 * 
 * Computes the ratio of a volume occupied by physical bodies.
 * 1. Resolves world-space bounds of bodies across BigInt sectors.
 * 2. Accumulates overlapping volumes using bit-perfect AABB intersections.
 * 3. Returns a normalized ratio in FixedMathCore [0..1].
 */
void volume_occupancy_kernel(
		const AABBf &p_query_volume,
		const BigIntCore &p_qsx, const BigIntCore &p_qsy, const BigIntCore &p_qsz,
		const PhysicsServerHyper::Body **p_bodies,
		uint32_t p_body_count,
		FixedMathCore &r_occupancy_ratio) {

	FixedMathCore occupied_acc = MathConstants<FixedMathCore>::zero();
	FixedMathCore total_vol = p_query_volume.get_volume();

	if (total_vol.get_raw() == 0) {
		r_occupancy_ratio = MathConstants<FixedMathCore>::zero();
		return;
	}

	for (uint32_t i = 0; i < p_body_count; i++) {
		const PhysicsServerHyper::Body *b = p_bodies[i];
		if (!b->active) continue;

		// Calculate the offset from the body's sector to the query's sector
		Vector3f sector_offset = wp::calculate_galactic_relative_pos(
			Vector3f_ZERO, p_qsx, p_qsy, p_qsz,
			Vector3f_ZERO, b->sector_x, b->sector_y, b->sector_z,
			FixedMathCore(10000LL, false) // 10k sector size
		);

		// Transform the body's AABB into the query's sector space
		AABBf body_world_aabb = b->transform.xform_aabb(b->mesh_data->get_aabb());
		body_world_aabb.position += sector_offset;

		if (p_query_volume.intersects(body_world_aabb)) {
			AABBf overlap = p_query_volume.intersection(body_world_aabb);
			occupied_acc += overlap.get_volume();
		}
	}

	r_occupancy_ratio = wp::clamp(occupied_acc / total_vol, MathConstants<FixedMathCore>::zero(), MathConstants<FixedMathCore>::one());
}

/**
 * query_sdf_gradient()
 * 
 * Calculates the direction of the nearest surface for machine perception.
 * Uses a deterministic central-difference approximation of the distance field gradient.
 */
Vector3f PhysicsServerHyper::query_sdf_gradient(
		const Vector3f &p_point,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_epsilon) {

	auto sample_sdf = [&](const Vector3f &p_pos) -> FixedMathCore {
		List<RID> neighbors;
		// Query galactic broadphase for local neighborhood
		broadphase.query_radius(p_pos, p_sx, p_sy, p_sz, p_epsilon * FixedMathCore(50LL, false), neighbors);
		
		FixedMathCore min_dist = FixedMathCore(2147483647LL, false); // Infinity

		for (const typename List<RID>::Element *E = neighbors.front(); E; E = E->next()) {
			Body *b = body_owner.get_or_null(E->get());
			if (!b || !b->active) continue;

			// Resolve relative position in body's local space
			Vector3f local_p = b->transform.xform_inv(p_pos);
			FixedMathCore d = b->mesh_data->get_aabb().distance_to_point(local_p);
			if (d < min_dist) min_dist = d;
		}
		return min_dist;
	};

	// 6-point bit-perfect symmetric gradient estimation
	FixedMathCore dx = sample_sdf(p_point + Vector3f(p_epsilon, 0, 0)) - sample_sdf(p_point - Vector3f(p_epsilon, 0, 0));
	FixedMathCore dy = sample_sdf(p_point + Vector3f(0, p_epsilon, 0)) - sample_sdf(p_point - Vector3f(0, p_epsilon, 0));
	FixedMathCore dz = sample_sdf(p_point + Vector3f(0, 0, p_epsilon)) - sample_sdf(p_point - Vector3f(0, 0, p_epsilon));

	Vector3f gradient(dx, dy, dz);
	if (gradient.length_squared().get_raw() == 0) {
		return Vector3f(0, 1, 0); // Default for empty space
	}

	return gradient.normalized();
}

/**
 * process_volume_triggers_parallel()
 * 
 * Master parallel orchestrator for machine triggers.
 * 1. Iterates through all EnTT entities with a VolumeTrigger component.
 * 2. Launches Warp kernels to calculate occupancy ratios in parallel.
 * 3. Dispatches bit-perfect results to the Machine Perception Queue at 120 FPS.
 */
void PhysicsServerHyper::process_volume_triggers_parallel(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &trigger_stream = registry.get_stream<AABBf>(COMPONENT_TRIGGER_VOLUME);
	auto &results_stream = registry.get_stream<FixedMathCore>(COMPONENT_OCCUPANCY_RESULT);
	
	uint64_t trigger_count = trigger_stream.size();
	if (trigger_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = trigger_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? trigger_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			// Zero-copy access to the body array
			const Body **body_array = body_owner.get_raw_data_ptr();
			uint32_t total_bodies = body_owner.get_rid_count();

			for (uint64_t i = start; i < end; i++) {
				const BigIntCore &sx = registry.get_component<BigIntCore>(BigIntCore(static_cast<int64_t>(i)), COMPONENT_SECTOR_X);
				const BigIntCore &sy = registry.get_component<BigIntCore>(BigIntCore(static_cast<int64_t>(i)), COMPONENT_SECTOR_Y);
				const BigIntCore &sz = registry.get_component<BigIntCore>(BigIntCore(static_cast<int64_t>(i)), COMPONENT_SECTOR_Z);

				volume_occupancy_kernel(
					registry.get_stream<AABBf>(COMPONENT_TRIGGER_VOLUME)[i],
					sx, sy, sz,
					body_array,
					total_bodies,
					registry.get_stream<FixedMathCore>(COMPONENT_OCCUPANCY_RESULT)[i]
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_queries_volume.cpp ---
