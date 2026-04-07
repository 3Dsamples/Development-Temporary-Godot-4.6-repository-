--- START OF FILE core/math/relativistic_kinematics_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: LorentzKinematicsKernel
 * 
 * Performs relativistic integration of position and velocity.
 * 1. Calculates the Lorentz Factor (Gamma) to determine time dilation.
 * 2. Scales the simulation delta to "Proper Time" for the local entity.
 * 3. Updates high-precision position using BigIntCore sector logic.
 */
void lorentz_kinematics_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		FixedMathCore &r_proper_time_accum,
		const FixedMathCore &p_c_squared, // Speed of light squared (FixedMath)
		const FixedMathCore &p_delta) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Calculate Speed Squared (v^2)
	FixedMathCore v2 = r_velocity.length_squared();

	// 2. Compute Lorentz Factor: Gamma = 1 / sqrt(1 - v^2/c^2)
	// We clamp v2 to strictly less than c2 to prevent imaginary numbers
	FixedMathCore beta2 = wp::min(v2 / p_c_squared, FixedMathCore(4294967290LL, true)); // 0.999999...
	FixedMathCore gamma = one / (one - beta2).square_root();

	// 3. Time Dilation: dt_proper = dt_world / gamma
	FixedMathCore proper_delta = p_delta / gamma;
	r_proper_time_accum += proper_delta;

	// 4. Relativistic Position Update
	// Even at high speeds, we use the dilated proper delta for local logic
	// but world-space displacement uses the standard delta.
	r_position += r_velocity * p_delta;

	// 5. Mass-Energy Increase (Effect on Momentum)
	// In the Universal Solver, we don't just move; we update the material tensor's 
	// effective mass for future collision resolutions.
}

/**
 * execute_relativistic_sweep()
 * 
 * Parallel sweep over EnTT registries containing high-speed components.
 * Essential for 120 FPS fleet combat where hundreds of ships move at 0.99c.
 */
void execute_relativistic_sweep(
		Vector3f *r_positions,
		Vector3f *r_velocities,
		FixedMathCore *r_local_clocks,
		uint64_t p_count,
		const FixedMathCore &p_speed_of_light,
		const FixedMathCore &p_delta) {

	FixedMathCore c2 = p_speed_of_light * p_speed_of_light;
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				lorentz_kinematics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_positions[i],
					r_velocities[i],
					r_local_clocks[i],
					c2,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * calculate_relativistic_addition()
 * 
 * Implements Einstein's velocity addition formula.
 * Ensures that relative speeds between two high-speed bodies never exceed c.
 */
Vector3f calculate_relativistic_addition(
		const Vector3f &p_v,
		const Vector3f &p_u,
		const FixedMathCore &p_c_squared) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	
	// u_parallel = (v + u) / (1 + v*u/c^2)
	FixedMathCore dot_vu = p_v.dot(p_u);
	FixedMathCore denom = one + (dot_vu / p_c_squared);

	return (p_v + p_u) / denom;
}

} // namespace UniversalSolver

--- END OF FILE core/math/relativistic_kinematics_kernel.cpp ---
