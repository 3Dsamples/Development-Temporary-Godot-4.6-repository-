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

	// 1. Calculate Speed Squared and Beta Squared
	FixedMathCore v2 = r_velocity.length_squared();
	// beta2 = v^2 / c^2. Clamp to 0.9999999999 to avoid division by zero near c.
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, FixedMathCore(4294967290LL, true));

	// 2. Resolve Lorentz Factor (Gamma)
	// gamma = 1 / sqrt(1 - beta2)
	FixedMathCore inv_gamma_sq = one - beta2;
	FixedMathCore inv_gamma = inv_gamma_sq.square_root();
	FixedMathCore gamma = one / inv_gamma;

	// 3. Relativistic Mass Scaling: m_eff = m_rest * gamma
	r_effective_mass = p_rest_mass * gamma;

	// 4. Proper Time Integration
	// Time dilates for the object; its internal clock (proper time) slows down.
	// delta_proper = delta_world * (1 / gamma)
	FixedMathCore proper_delta = p_delta * inv_gamma;
	r_proper_time_accum += proper_delta;

	// 5. Kinematic Integration (Position Update)
	// Even at relativistic speeds, world-space position is integrated using world-delta.
	r_position += r_velocity * p_delta;
}

/**
 * Warp Kernel: RelativisticVelocityAdditionKernel
 * 
 * Implements Einstein's 3D velocity addition formula.
 * u = (v + w) / (1 + (v.w)/c^2) [parallel component simplification]
 * Essential for ships launching projectiles at high speeds.
 */
void relativistic_velocity_addition_kernel(
		const Vector3f &p_v, // velocity of frame
		const Vector3f &p_w, // velocity of object in frame
		const FixedMathCore &p_c_sq,
		Vector3f &r_u) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// Scalar denominator: 1 + (v dot w) / c^2
	FixedMathCore dot_vw = p_v.dot(p_w);
	FixedMathCore denom = one + (dot_vw / p_c_sq);

	// Lorentz factor of frame velocity
	FixedMathCore v2 = p_v.length_squared();
	FixedMathCore gamma_v = one / (one - (v2 / p_c_sq)).square_root();

	// Transverse component correction
	FixedMathCore factor = one / (denom * gamma_v);
	
	// Full 3D Relativistic Addition
	// u = (1/denom) * [ v + w_parallel + (1/gamma_v)*w_perpendicular ]
	Vector3f w_parallel = p_v * (p_v.dot(p_w) / (v2 + MathConstants<FixedMathCore>::unit_epsilon()));
	Vector3f w_perpendicular = p_w - w_parallel;

	r_u = (p_v + w_parallel + w_perpendicular * (one / gamma_v)) / denom;
}

/**
 * execute_relativistic_wave()
 * 
 * Orchestrates the parallel 120 FPS integration sweep for high-speed entities.
 * Manages the transition between EnTT SoA components and Warp execution lanes.
 */
void execute_relativistic_wave(
		KernelRegistry &p_registry,
		const FixedMathCore &p_c,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &proper_time_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_PROPER_TIME);
	auto &eff_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_EFFECTIVE_MASS);
	auto &rest_mass_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_REST_MASS);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	FixedMathCore c2 = p_c * p_c;
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

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

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_time_dilation_visual_shift()
 * 
 * Advanced Behavior: Computes the spectral shift tensor for visuals.
 * As time dilates, the perceived frequency of light shifts (Doppler + Relativistic).
 */
Vector3f calculate_time_dilation_visual_shift(
		const Vector3f &p_velocity,
		const Vector3f &p_view_dir,
		const FixedMathCore &p_c) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore cos_theta = p_view_dir.dot(p_velocity.normalized());
	FixedMathCore beta = p_velocity.length() / p_c;

	// Doppler Factor: D = 1 / (gamma * (1 - beta*cos(theta)))
	FixedMathCore gamma = one / (one - beta * beta).square_root();
	FixedMathCore doppler = one / (gamma * (one - beta * cos_theta));

	// Convert shift to RGB scale tensor
	FixedMathCore shift = wp::clamp(doppler, FixedMathCore(2147483648LL, true), FixedMathCore(8589934592LL, true)); // [0.5, 2.0]
	return Vector3f(one / shift, one, shift); // Red shift / Blue shift approximation
}

} // namespace UniversalSolver

--- END OF FILE core/math/relativistic_kinematics_solver.cpp ---
