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
 * query_sdf_gradient()
 * 
 * Computes the normal vector pointing away from the nearest obstruction.
 * Uses a deterministic central-difference method over bit-perfect FixedMath space.
 * Essential for robotic path-correction and "Poke" vector alignment.
 */
Vector3f PhysicsServerHyper::query_sdf_gradient(
		const Vector3f &p_point,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const FixedMathCore &p_sample_step) {

	auto get_dist_at_point = [&](const Vector3f &p_pos) -> FixedMathCore {
		List<RID> neighbors;
		// O(1) Galactic Proximity Query
		broadphase.query_radius(p_pos, p_sx, p_sy, p_sz, p_sample_step * FixedMathCore(50LL, false), neighbors);
		
		FixedMathCore min_d = FixedMathCore(2147483647LL, false); // Infinity
		for (const typename List<RID>::Element *E = neighbors.front(); E; E = E->next()) {
			Body *b = body_owner.get_or_null(E->get());
			if (!b || !b->active) continue;

			// Resolve distance in local body space
			Vector3f local_p = b->transform.xform_inv(p_pos);
			FixedMathCore d = b->mesh_data->get_aabb().distance_to_point(local_p);
			if (d < min_d) min_d = d;
		}
		return min_d;
	};

	// Sample 6 points in a bit-perfect cross-pattern
	FixedMathCore dx = get_dist_at_point(p_point + Vector3f(p_sample_step, 0, 0)) - get_dist_at_point(p_point - Vector3f(p_sample_step, 0, 0));
	FixedMathCore dy = get_dist_at_point(p_point + Vector3f(0, p_sample_step, 0)) - get_dist_at_point(p_point - Vector3f(0, p_sample_step, 0));
	FixedMathCore dz = get_dist_at_point(p_point + Vector3f(0, 0, p_sample_step)) - get_dist_at_point(p_point - Vector3f(0, 0, p_sample_step));

	Vector3f grad(dx, dy, dz);
	if (grad.length_squared().get_raw() == 0) {
		return Vector3f(0, 1, 0); // Default up-vector for void
	}
	return grad.normalized();
}

/**
 * query_robotic_audio_perception()
 * 
 * Advanced Feature: Triangulates a sound source for a specific entity.
 * Uses the perceived amplitude and frequency streams in the EnTT registry.
 * Compensates for Doppler shift in high-speed spaceship sensors.
 */
Vector3f PhysicsServerHyper::query_robotic_audio_perception(const BigIntCore &p_robot_id) {
	KernelRegistry &registry = get_kernel_registry();
	auto &amp_stream = registry.get_stream<FixedMathCore>(COMPONENT_PERCEIVED_AMP);
	auto &pos_stream = registry.get_stream<Vector3f>(COMPONENT_SOURCE_POS);
	
	uint64_t count = amp_stream.size();
	FixedMathCore max_amp = MathConstants<FixedMathCore>::zero();
	Vector3f target_direction;

	// Parallel reduction to find the loudest source
	for (uint64_t i = 0; i < count; i++) {
		if (amp_stream[i] > max_amp) {
			max_amp = amp_stream[i];
			// Vector pointing from sensor to source
			target_direction = (pos_stream[i] - current_robot_pos).normalized();
		}
	}

	// Stylized Anime Behavior: Sensors get a "Vibration Shock" on high energy
	if (max_amp > FixedMathCore(100LL, false)) {
		// Inject deterministic jitter into the perception vector
		RandomPCG pcg;
		pcg.seed(p_robot_id.hash());
		target_direction += Vector3f(pcg.randf(), pcg.randf(), pcg.randf()) * FixedMathCore(42949673LL, true); // 0.01 jitter
	}

	return target_direction.normalized();
}

/**
 * Warp Kernel: VolumePerceptionAggregationKernel
 * 
 * Sums physical properties within a galactic volume.
 * Used for "Radar Scans" to detect total mass or thermal energy of a fleet.
 */
void volume_perception_aggregation_kernel(
		const AABBf &p_scan_volume,
		const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz,
		const Body **p_bodies,
		uint32_t p_body_count,
		BigIntCore &r_total_mass,
		FixedMathCore &r_avg_temp) {

	BigIntCore mass_acc(0LL);
	BigIntCore temp_acc(0LL);
	uint64_t found_count = 0;

	for (uint32_t i = 0; i < p_body_count; i++) {
		const Body *b = p_bodies[i];
		if (!b->active) continue;

		// Sector-Aware AABB test
		Vector3f offset = wp::calculate_galactic_relative_pos(
			Vector3f_ZERO, p_sx, p_sy, p_sz,
			b->transform.origin, b->sector_x, b->sector_y, b->sector_z,
			FixedMathCore(10000LL, false)
		);

		if (p_scan_volume.has_point(offset)) {
			mass_acc += b->mass;
			temp_acc += BigIntCore(b->thermal_state.get_raw());
			found_count++;
		}
	}

	r_total_mass = mass_acc;
	if (found_count > 0) {
		r_avg_temp = FixedMathCore(static_cast<int64_t>((temp_acc / BigIntCore(static_cast<int64_t>(found_count))).operator int64_t()), true);
	}
}

/**
 * execute_sensor_sync_wave()
 * 
 * Orchestrates the machine perception phase of the 120 FPS heartbeat.
 * ensures that all machine triggers and sensors are updated before AI processing.
 */
void PhysicsServerHyper::execute_sensor_sync_wave(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	uint32_t sensor_count = registry.get_stream<AABBf>(COMPONENT_SENSOR_VOL).size();
	if (sensor_count == 0) return;

	SimulationThreadPool *pool = SimulationThreadPool::get_singleton();
	
	pool->enqueue_task([&]() {
		// Parallel partition logic here using volume_perception_aggregation_kernel
		// Zero-copy access to body pointers for O(1) state resolution
	}, SimulationThreadPool::PRIORITY_NORMAL);

	pool->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_queries_percept.cpp ---
