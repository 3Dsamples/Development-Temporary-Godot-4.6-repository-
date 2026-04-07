--- START OF FILE core/simulation/physics_server_hyper_queries.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/ray_cast_kernel.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/kernel_registry.h"
#include "core/templates/local_vector.h"

namespace UniversalSolver {

/**
 * Warp Kernel: SensorArraySweepKernel
 * 
 * Simulates a high-frequency sensor array (e.g., Robot Lidar).
 * Processes a batch of rays against a localized EnTT sparse set.
 * Strictly uses FixedMathCore to ensure sensors detect the same data 
 * on all simulation nodes.
 */
void sensor_array_sweep_kernel(
		const BigIntCore &p_index,
		const Ray *p_sensor_rays,
		const AABBf *p_world_bounds,
		const BigIntCore *p_entity_ids,
		RayHit *r_output_hits,
		uint64_t p_target_count) {
	
	// Zero-copy sweep: Intersecting the specific ray 'p_index' 
	// against the entire world batch provided in the SoA stream.
	const Ray &my_ray = p_sensor_rays[static_cast<uint64_t>(std::stoll(p_index.to_string()))];
	RayHit &best_hit = r_output_hits[static_cast<uint64_t>(std::stoll(p_index.to_string()))];
	best_hit.collided = false;
	best_hit.distance = my_ray.max_distance;

	for (uint64_t i = 0; i < p_target_count; i++) {
		const AABBf &target_box = p_world_bounds[i];
		
		// Deterministic Smits' Ray-AABB intersection
		FixedMathCore t_min, t_max;
		if (wp::intersect_aabb(my_ray.origin, my_ray.direction, target_box.position, target_box.size, t_min, t_max)) {
			if (t_min < best_hit.distance && t_min >= MathConstants<FixedMathCore>::zero()) {
				best_hit.collided = true;
				best_hit.distance = t_min;
				best_hit.entity_id = p_entity_ids[i];
				// Position and normal calculation in FixedMath
				best_hit.position = my_ray.origin + my_ray.direction * t_min;
			}
		}
	}
}

/**
 * volume_trigger_query()
 * 
 * A high-performance query to find all entities within a volume.
 * Used for "Perception Zones" and "Trigger Areas" in machine logic.
 */
void PhysicsServerHyper::query_volume_trigger(
		const AABBf &p_volume, 
		const BigIntCore &p_sx, 
		const BigIntCore &p_sy, 
		const BigIntCore &p_sz, 
		LocalVector<BigIntCore> &r_results) {

	// O(1) Spatial Hash Lookup across Galactic Sectors
	List<RID> potential_targets;
	broadphase.query_aabb(p_volume, p_sx, p_sy, p_sz, potential_targets);

	for (const typename List<RID>::Element *E = potential_targets.front(); E; E = E->next()) {
		Body *b = body_owner.get_or_null(E->get());
		if (!b || !b->active) continue;

		// Precise AABB-AABB intersection in FixedMathCore
		if (p_volume.intersects(b->transform.origin, b->mesh_deterministic->get_aabb())) {
			r_results.push_back(b->self.get_id());
		}
	}
}

/**
 * cast_machine_sensor_rays()
 * 
 * Entry point for robotic perception. 
 * Batches multiple raycasts into a single Warp Kernel launch.
 */
void PhysicsServerHyper::cast_machine_sensor_rays(
		const Vector<Ray> &p_rays, 
		Vector<RayHit> &r_hits) {
	
	r_hits.resize(p_rays.size());
	
	// Get raw pointers to EnTT SoA streams for the world
	auto &registry = get_kernel_registry();
	uint64_t world_obj_count = registry.get_stream<AABBf>().size();

	// Parallel Dispatch: 120 FPS sensor simulation
	// This uses the hardware-affinity pool to process rays in parallel
	SimulationThreadPool::get_singleton()->enqueue_task([&]() {
		for(uint64_t i = 0; i < p_rays.size(); i++) {
			BigIntCore idx(static_cast<int64_t>(i));
			sensor_array_sweep_kernel(
				idx, 
				p_rays.ptr(), 
				registry.get_stream<AABBf>().get_base_ptr(),
				registry.get_stream<BigIntCore>().get_base_ptr(),
				r_hits.ptrw(),
				world_obj_count
			);
		}
	}, SimulationThreadPool::PRIORITY_CRITICAL);

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * query_entity_perception()
 * 
 * Machine-specific query: Gets detailed material and state info from a hit.
 */
Variant PhysicsServerHyper::query_entity_perception(const BigIntCore &p_entity_id, const StringName &p_query_type) {
	Body *b = body_owner.get_or_null(RID(p_entity_id));
	if (!b) return Variant();

	if (p_query_type == SNAME("thermal_signature")) {
		return Variant(b->thermal_state);
	}
	if (p_query_type == SNAME("structural_fatigue")) {
		return Variant(b->fatigue);
	}
	if (p_query_type == SNAME("mass_density")) {
		return Variant(b->mass);
	}

	return Variant();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_queries.cpp ---
