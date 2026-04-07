--- START OF FILE core/math/cloud_voxel_physics.cpp ---

#include "core/math/cloud_voxel_kernel.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: CloudDynamicsPhysicsKernel
 * 
 * Performs the physical update for a single cloud voxel.
 * 1. Buoyancy: Hotter voxels rise relative to the lapse rate.
 * 2. Advection: Voxels shift based on global and local wind tensors.
 * 3. Phase Change: Moisture condenses or evaporates based on pressure/temp.
 */
void cloud_dynamics_physics_kernel(
		const BigIntCore &p_index,
		FixedMathCore &r_density,
		Vector3f &r_velocity,
		FixedMathCore &r_temperature,
		FixedMathCore &r_moisture,
		const Vector3f &p_wind_vector,
		const FixedMathCore &p_gravity,
		const FixedMathCore &p_lapse_rate,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Thermal Buoyancy Logic
	// B = g * (T_voxel - T_ambient) / T_ambient
	// Ambient temperature assumed at sea level (288.15K) minus lapse rate * altitude
	FixedMathCore altitude = p_index.operator int64_t() / 100; // Simplified altitude mapping
	FixedMathCore ambient_temp = FixedMathCore(12376175411LL, true) - (p_lapse_rate * altitude);
	
	FixedMathCore temp_diff = r_temperature - ambient_temp;
	FixedMathCore buoyancy_accel = (p_gravity * temp_diff) / ambient_temp;
	
	// Apply buoyancy to vertical velocity
	r_velocity.y += buoyancy_accel * p_delta;

	// 2. Wind Advection & Drag
	// Clouds are largely carried by wind but have internal inertia
	Vector3f wind_force = (p_wind_vector - r_velocity) * FixedMathCore(214748364LL, true); // 0.05 drag coeff
	r_velocity += wind_force * p_delta;

	// 3. Phase Change (Vapor to Liquid/Ice)
	// Saturation vapor pressure approx via deterministic polynomial
	FixedMathCore saturation_point = FixedMathCore(4294967296LL, false) - (r_temperature * FixedMathCore(42949672LL, true));
	
	if (r_moisture > saturation_point) {
		FixedMathCore condensation = (r_moisture - saturation_point) * FixedMathCore(429496730LL, true); // 0.1 rate
		r_moisture -= condensation;
		r_density += condensation; // Voxel becomes more opaque/dense
		// Latent heat release: condensation warms the voxel
		r_temperature += condensation * FixedMathCore(2147483648LL, true); // 0.5 heat boost
	} else if (r_density > zero && r_temperature > ambient_temp) {
		// Evaporation
		FixedMathCore evaporation = (ambient_temp / r_temperature) * FixedMathCore(42949673LL, true);
		r_density = wp::max(zero, r_density - evaporation);
		r_moisture += evaporation;
		r_temperature -= evaporation * FixedMathCore(1073741824LL, true); // 0.25 cooling
	}

	// 4. Kinetic Integration
	// Note: Voxel positions in EnTT are usually grid-aligned, so velocity 
	// often represents the 'flow' between cells (Eularian approach).
}

/**
 * execute_cloud_physics_sweep()
 * 
 * Orchestrates the parallel simulation of all active cloud volumes.
 * Partitions the EnTT voxel registry into SIMD-friendly worker batches.
 */
void execute_cloud_physics_sweep(
		const BigIntCore &p_total_voxels,
		FixedMathCore *r_densities,
		Vector3f *r_velocities,
		FixedMathCore *r_temperatures,
		FixedMathCore *r_moisture_levels,
		const Vector3f &p_global_wind,
		const FixedMathCore &p_delta) {

	uint64_t total = static_cast<uint64_t>(std::stoll(p_total_voxels.to_string()));
	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = total / workers;

	FixedMathCore gravity(42122340352LL, true);   // 9.80665
	FixedMathCore lapse(27917287LL, true);      // 0.0065 K/m

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? total : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=]() {
			for (uint64_t i = start; i < end; i++) {
				cloud_dynamics_physics_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					r_densities[i],
					r_velocities[i],
					r_temperatures[i],
					r_moisture_levels[i],
					p_global_wind,
					gravity,
					lapse,
					p_delta
				);
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	// Barrier sync for 120 FPS consistency
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_localized_explosion_to_clouds()
 * 
 * Advanced Feature: Physical interaction between explosions and clouds.
 * High-energy events displace densities and trigger instant evaporation.
 */
void apply_localized_explosion_to_clouds(
		FixedMathCore *r_densities,
		FixedMathCore *r_temperatures,
		const Vector3f *p_voxel_positions,
		uint64_t p_count,
		const Vector3f &p_epicenter,
		const FixedMathCore &p_energy,
		const FixedMathCore &p_radius) {

	FixedMathCore r2 = p_radius * p_radius;

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore d2 = (p_voxel_positions[i] - p_epicenter).length_squared();
		if (d2 < r2) {
			FixedMathCore dist = Math::sqrt(d2);
			FixedMathCore falloff = (p_radius - dist) / p_radius;
			
			// Disperse cloud density
			r_densities[i] *= (MathConstants<FixedMathCore>::one() - falloff);
			// Massive thermal spike
			r_temperatures[i] += p_energy * falloff;
		}
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/cloud_voxel_physics.cpp ---
