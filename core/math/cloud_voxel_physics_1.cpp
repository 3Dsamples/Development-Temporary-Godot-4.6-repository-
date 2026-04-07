--- START OF FILE core/math/cloud_voxel_physics.cpp ---

#include "core/math/cloud_voxel_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: CloudPhysicsDynamicsKernel
 * 
 * Simulates the physical state change of a cloud voxel.
 * 1. Advection: Moves density and moisture according to the velocity field.
 * 2. Condensation: Vapor turns to density when exceeding saturation pressure.
 * 3. Thermodynamics: Released latent heat increases local temperature.
 * 4. Buoyancy: Density-driven vertical acceleration.
 */
void cloud_physics_dynamics_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_density,
		FixedMathCore &r_moisture,
		FixedMathCore &r_temperature,
		Vector3f &r_velocity,
		const Vector3f &p_global_wind,
		const FixedMathCore &p_lapse_rate,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Local Saturation Vapor Pressure (Deterministic Approx)
	// Simplified Magnus formula for FixedMath: Psat = 6.112 * exp(17.67 * T / (T + 243.5))
	// T is in Celsius, we convert from Kelvin (273.15 offset)
	FixedMathCore temp_c = r_temperature - FixedMathCore(11731631500LL, true); // 273.15 K
	FixedMathCore sat_threshold = FixedMathCore(1000LL, false) - (temp_c * FixedMathCore(20LL, false)); // Simplified linear inverse for 120 FPS

	// 2. Phase-Transition: Condensation / Evaporation
	// If moisture (vapor) > sat_threshold, it converts to density (droplets/ice)
	if (r_moisture > sat_threshold) {
		FixedMathCore condensation_rate = (r_moisture - sat_threshold) * FixedMathCore(429496730LL, true); // 0.1 delta
		r_moisture -= condensation_rate;
		r_density += condensation_rate;
		
		// Latent Heat Release: Warming effect
		FixedMathCore latent_heat_coeff(10737418240LL, true); // 2.5 factor
		r_temperature += condensation_rate * latent_heat_coeff;
	} else if (r_density > zero) {
		// Evaporative Cooling
		FixedMathCore evaporation = (sat_threshold - r_moisture) * FixedMathCore(214748364LL, true); // 0.05 delta
		evaporation = wp::min(evaporation, r_density);
		r_density -= evaporation;
		r_moisture += evaporation;
		
		r_temperature -= evaporation * FixedMathCore(8589934592LL, true); // 2.0 cooling factor
	}

	// 3. Buoyancy and Advection
	// B = g * (T_v - T_env) / T_env
	FixedMathCore env_temp = FixedMathCore(12376175411LL, true) - (p_lapse_rate * FixedMathCore(p_index.operator int64_t() / 10)); 
	FixedMathCore buoyancy = (r_temperature - env_temp) * FixedMathCore(429496729LL, true); // 0.1 scale
	
	r_velocity.y += buoyancy * p_delta;
	
	// Drag against global wind field
	Vector3f wind_diff = p_global_wind - r_velocity;
	r_velocity += wind_diff * FixedMathCore(85899346LL, true) * p_delta; // 0.02 drag

	// 4. Position Shift (Logic for grid-based advection)
	// In a full implementation, density/moisture are shifted between neighbors
	// here using a bit-perfect Semi-Lagrangian back-trace.
}

/**
 * execute_weather_physics_sweep()
 * 
 * Orchestrates the parallel update of the planetary cloud-layer.
 * Partitions the EnTT voxel stream into SIMD-friendly worker chunks.
 */
void execute_weather_physics_sweep(
		KernelRegistry &p_registry,
		const Vector3f &p_global_wind,
		const FixedMathCore &p_delta) {

	auto &density_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_DENSITY);
	auto &moisture_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_MOISTURE);
	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);

	uint64_t count = density_stream.size();
	if (count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	FixedMathCore lapse_rate(27917287LL, true); // 0.0065 K/m

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &density_stream, &moisture_stream, &temp_stream, &vel_stream]() {
			for (uint64_t i = start; i < end; i++) {
				cloud_physics_dynamics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					density_stream[i],
					moisture_stream[i],
					temp_stream[i],
					vel_stream[i],
					p_global_wind,
					lapse_rate,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_moisture_source_kernel()
 * 
 * Simulates real-time evaporation from planetary surfaces (oceans/lakes).
 * Injects moisture into the bottom layer of the cloud voxel grid.
 */
void apply_moisture_source_kernel(
		FixedMathCore *r_moisture_stream,
		const FixedMathCore *p_surface_temp,
		const BigIntCore &p_count,
		const FixedMathCore &p_delta) {

	FixedMathCore evaporation_k(42949673LL, true); // 0.01 base

	uint64_t total = static_cast<uint64_t>(std::stoll(p_count.to_string()));
	for (uint64_t i = 0; i < total; i++) {
		// Vapor generation = surface_heat * constant
		r_moisture_stream[i] += p_surface_temp[i] * evaporation_k * p_delta;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/cloud_voxel_physics.cpp ---
