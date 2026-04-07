--- START OF FILE core/simulation/physics_server_hyper_queries_percept.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * calculate_local_sdf_distance()
 * 
 * Internal helper to compute the bit-perfect distance from a point to the nearest 
 * physical boundary in local body space.
 */
static _FORCE_INLINE_ FixedMathCore calculate_local_sdf_distance(
		const Body *p_body, 
		const Vector3f &p_world_point,
		const BigIntCore &p_wsx, const BigIntCore &p_wsy, const BigIntCore &p_wsz) {

	// 1. Resolve galactic offset between point and body
	Vector3f rel_offset = wp::calculate_galactic_relative_pos(
		p_world_point, p_wsx, p_wsy, p_wsz,
		p_body->transform.origin, p_body->sector_x, p_body->sector_y, p_body->sector_z,
		FixedMathCore(10000LL, false)
	);

	// 2. Transform to local body space
	Vector3f local_p = p_body->transform.xform_inv(p_world_point + rel_offset);

	// 3. Query local AABB/Mesh distance
	// Returns negative if inside (SDF property)
	return p_body->mesh_data->get_aabb().distance_to_point(local_p);
}

/**
 * query_sdf_gradient()
 * 
 * Machine Perception: Calculates the normal vector pointing away from the nearest 
 * obstruction. Essential for robotic path-correction and "Poke" vector alignment.
 * Uses a deterministic central-difference approximation.
 */
Vector3f PhysicsServerHyper::query_sdf_gradient(
		const Vector3f &p_point,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_sample_step) {

	auto get_dist = [&](const Vector3f &pos) -> FixedMathCore {
		List<RID> neighbors;
		// Query broadphase for local neighbors within 10x sample step
		broadphase.query_radius(pos, p_sx, p_sy, p_sz, p_sample_step * FixedMathCore(10LL, false), neighbors);
		
		FixedMathCore min_d = FixedMathCore(2147483647LL, false); // Infinity
		for (const typename List<RID>::Element *E = neighbors.front(); E; E = E->next()) {
			Body *b = body_owner.get_or_null(E->get());
			if (!b || !b->active) continue;

			Fixed_Store d = calculate_local_sdf_distance(b, pos, p_sx, p_sy, p_sz);
			if (d < min_d) min_d = d;
		}
		return min_d;
	};

	// 6-point bit-perfect symmetric gradient estimation
	FixedMathCore dx = get_dist(p_point + Vector3f(p_sample_step, 0, 0)) - get_dist(p_point - Vector3f(p_sample_step, 0, 0));
	FixedMathCore dy = get_dist(p_point + Vector3f(0, p_sample_step, 0)) - get_dist(p_point - Vector3f(0, p_sample_step, 0));
	FixedMathCore dz = get_dist(p_point + Vector3f(0, 0, p_sample_step)) - get_dist(p_point - Vector3f(0, 0, p_sample_step));

	Vector3f gradient(dx, dy, dz);
	if (gradient.length_squared().get_raw() == 0) {
		return Vector3f(0, 1, 0); // Default for void
	}

	return gradient.normalized();
}

/**
 * Warp Kernel: VolumetricPerceptionKernel
 * 
 * Parallel reduction to aggregate sensor data for a machine's perception volume.
 * Gathers: Total Mass (BigInt), Average Temperature (Fixed), and Occupancy Ratio.
 */
void volumetric_perception_kernel(
		const BigIntCore &p_index,
		const AABBf &p_sensor_volume,
		const BigIntCore &p_ssx, const BigIntCore &p_ssy, const BigIntCore &p_ssz,
		const Body **p_bodies,
		uint32_t p_body_count,
		BigIntCore &r_total_mass,
		FixedMathCore &r_avg_temp,
		FixedMathCore &r_occupancy) {

	BigIntCore mass_acc(0LL);
	BigIntCore temp_acc_raw(0LL);
	FixedMathCore occupied_vol = MathConstants<FixedMathCore>::zero();
	uint32_t count = 0;

	for (uint32_t i = 0; i < p_body_count; i++) {
		const Body *b = p_bodies[i];
		if (!b || !b->active) continue;

		// 1. Resolve Sector-Aware Proximity
		Vector3f offset = wp::calculate_galactic_relative_pos(
			Vector3f_ZERO, p_ssx, p_ssy, p_ssz,
			b->transform.origin, b->sector_x, b->sector_y, b->sector_z,
			FixedMathCore(10000LL, false)
		);

		AABBf world_bounds = b->transform.xform_aabb(b->mesh_data->get_aabb());
		world_bounds.position += offset;

		// 2. Accumulate Tensors
		if (p_sensor_volume.intersects(world_bounds)) {
			mass_acc += b->mass;
			temp_acc_raw += BigIntCore(b->thermal_state.get_raw());
			
			AABBf overlap = p_sensor_volume.intersection(world_bounds);
			occupied_vol += overlap.get_volume();
			count++;
		}
	}

	r_total_mass = mass_acc;
	r_occupancy = occupied_vol / (p_sensor_volume.get_volume() + MathConstants<FixedMathCore>::unit_epsilon());
	
	if (count > 0) {
		r_avg_temp = FixedMathCore(static_cast<int64_t>((temp_acc_raw / BigIntCore(static_cast<int64_t>(count))).operator int64_t()), true);
	}
}

/**
 * execute_sensor_resolve_wave()
 * 
 * Orchestrates the parallel 120 FPS perception wave for all robots and machines.
 */
void PhysicsServerHyper::execute_sensor_resolve_wave(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &sensor_stream = registry.get_stream<AABBf>(COMPONENT_SENSOR_VOLUME);
	uint64_t sensor_count = sensor_stream.size();
	if (sensor_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = sensor_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? sensor_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			const Body **bodies = body_owner.get_raw_data_ptr();
			uint32_t n_bodies = body_owner.get_rid_count();

			for (uint64_t i = start; i < end; i++) {
				BigIntCore h = BigIntCore(static_cast<int64_t>(i));
				
				volumetric_perception_kernel(
					h,
					registry.get_stream<AABBf>(COMPONENT_SENSOR_VOLUME)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y)[i],
					registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z)[i],
					bodies,
					n_bodies,
					registry.get_stream<BigIntCore>(COMPONENT_PERCEIVED_MASS)[i],
					registry.get_stream<FixedMathCore>(COMPONENT_PERCEIVED_TEMP)[i],
					registry.get_stream<FixedMathCore>(COMPONENT_OCCUPANCY_RATIO)[i]
				);
			}
		}, SimulationThreadPool::PRIORITY_NORMAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_queries_percept.cpp ---
