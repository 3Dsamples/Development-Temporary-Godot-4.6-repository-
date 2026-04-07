--- START OF FILE core/math/relativistic_kinematics_solver.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: LorentzIntegrationKernel
 * 
 * Performs the fundamental physical update for relativistic entities.
 * 1. Computes Lorentz Gamma: gamma = 1 / sqrt(1 - v^2/c^2).
 * 2. Integrates Local Proper Time: dt_proper = dt_world / gamma.
 * 3. Updates high-precision position with bit-perfect velocity tensors.
 * 4. Scales effective mass: m_rel = m_rest * gamma.
 */
void lorentz_integration_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_proper_time_accum,
		FixedMathCore &r_effective_mass,
		const FixedMathCore &p_rest_mass,
		const FixedMathCore &p_c_sq,
		const FixedMathCore &p_delta) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Calculate Speed Squared and Beta Squared (v^2 / c^2)
	FixedMathCore v2 = r_velocity.length_squared();
	
	// Beta^2 must be < 1.0. We clamp to 0.999999999 to prevent infinite gamma (Division by zero).
	FixedMathCore beta2_limit(4294967290LL, true); 
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, beta2_limit);

	// 2. Resolve Lorentz Factor (Gamma)
	// gamma = 1 / sqrt(1 - beta2)
	FixedMathCore inv_gamma_sq = one - beta2;
	FixedMathCore inv_gamma = inv_gamma_sq.square_root();
	FixedMathCore gamma = one / inv_gamma;

	// 3. Relativistic Mass Scaling
	// As objects approach c, they become infinitely hard to accelerate.
	r_effective_mass = p_rest_mass * gamma;

	// 4. Proper Time Integration
	// Time dilates for the object; its internal clock slows down relative to the world.
	// delta_proper = delta_world / gamma
	FixedMathCore proper_delta = p_delta * inv_gamma;
	r_proper_time_accum += proper_delta;

	// 5. Kinematic Integration (Position Update)
	// World-space displacement remains v * dt_world.
	r_position += r_velocity * p_delta;
}

/**
 * Warp Kernel: RelativisticVelocityAdditionKernel
 * 
 * Implements Einstein's 3D velocity addition formula.
 * Ensures that the sum of two velocities (e.g. ship + missile) never exceeds c.
 * Formula: u = (v + w_parallel + (1/gamma_v) * w_perpendicular) / (1 + v*w/c^2)
 */
void relativistic_velocity_addition_kernel(
		const Vector3f &p_v, // Frame velocity
		const Vector3f &p_w, // Object velocity relative to frame
		const FixedMathCore &p_c_sq,
		Vector3f &r_u) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore v2 = p_v.length_squared();
	
	if (v2.get_raw() == 0) {
		r_u = p_w;
		return;
	}

	// 1. Compute Scalar Denominator: 1 + (v dot w) / c^2
	FixedMathCore dot_vw = p_v.dot(p_w);
	FixedMathCore denom = one + (dot_vw / p_c_sq);

	// 2. Resolve Lorentz Factor of the frame
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, FixedMathCore(4294967290LL, true));
	FixedMathCore gamma_v = one / (one - beta2).square_root();

	// 3. Decompose w into parallel and perpendicular components
	Vector3f w_parallel = p_v * (p_v.dot(p_w) / v2);
	Vector3f w_perpendicular = p_w - w_parallel;

	// 4. Resolve Resultant Velocity
	// u = (v + w_parallel + (inv_gamma)*w_perpendicular) / denom
	Vector3f numerator = p_v + w_parallel + (w_perpendicular / gamma_v);
	r_u = numerator / denom;
}

/**
 * execute_relativistic_integration_wave()
 * 
 * Orchestrates the parallel 120 FPS kinematic sweep.
 * Pulls SoA data from the KernelRegistry and dispatches to SimulationThreadPool.
 * strictly bit-perfect across trillions of entities.
 */
void execute_relativistic_integration_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_speed_of_light,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &proper_time_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_PROPER_TIME);
	auto &eff_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_EFFECTIVE_MASS);
	auto &rest_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_REST_MASS);

	uint64_t entity_count = pos_stream.size();
	if (entity_count == 0) return;

	FixedMathCore c2 = p_speed_of_light * p_speed_of_light;
	uint32_t worker_count = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk_size = entity_count / worker_count;

	for (uint32_t w = 0; w < worker_count; w++) {
		uint64_t start = w * chunk_size;
		uint64_t end = (w == worker_count - 1) ? entity_count : (w + 1) * chunk_size;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &proper_time_stream, &eff_mass_stream, &rest_mass_stream]() {
			for (uint64_t i = start; i < end; i++) {
				lorentz_integration_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					vel_stream[i],
					proper_time_stream[i],
					eff_mass_stream[i],
					rest_mass_stream[i],
					c2,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	// Final Synchronization Barrier for the kinematic wave
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_doppler_hue_shift()
 * 
 * Sophisticated Real-Time Behavior:
 * Returns an RGB spectral shift vector based on the relativistic Doppler effect.
 */
Vector3f calculate_doppler_hue_shift(
		const Vector3f &p_velocity,
		const Vector3f &p_view_dir,
		const FixedMathCore &p_c) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore cos_theta = p_view_dir.dot(p_velocity.normalized());
	FixedMathCore beta = p_velocity.length() / p_c;

	// D = sqrt((1 + beta) / (1 - beta)) for longitudinal
	// Relativistic Transverse Doppler: D = 1 / (gamma * (1 - beta * cos(theta)))
	FixedMathCore beta_clamped = wp::min(beta, FixedMathCore(4294967290LL, true));
	FixedMathCore gamma = one / (one - beta_clamped * beta_clamped).square_root();
	FixedMathCore factor = one / (gamma * (one - beta_clamped * cos_theta));

	// Blue-shift if factor > 1, Red-shift if factor < 1
	// Maps to RGB scaling tensor
	return Vector3f(one / factor, one, factor);
}

} // namespace UniversalSolver

--- END OF FILE core/math/relativistic_kinematics_solver.cpp ---
