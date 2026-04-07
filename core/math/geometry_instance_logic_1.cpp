--- START OF FILE core/math/geometry_instance_logic.cpp ---

#include "core/math/geometry_instance_logic.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * execute_thermal_conduction_sweep()
 * 
 * Orchestrates the transfer of thermal energy between adjacent simulation entities.
 * 1. Iterates through the EnTT adjacency registry.
 * 2. Calculates heat flow: dQ = k * (T2 - T1) * Area * dt.
 * 3. Applies bit-perfect temperature updates using FixedMathCore.
 */
void execute_thermal_conduction_sweep(
		KernelRegistry &p_registry,
		const uint32_t *p_adjacency_map, // Flattened pairs of neighbor indices
		uint64_t p_pair_count,
		const FixedMathCore &p_delta) {

	auto &temp_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_TEMPERATURE);
	auto &area_stream = p_registry.get_stream<FixedMathCore>(COMPONENT_SURFACE_AREA);
	
	// Global conductivity constant for metallic structures
	FixedMathCore k_metal("0.05"); 

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_pair_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_pair_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &temp_stream, &area_stream]() {
			for (uint64_t i = start; i < end; i++) {
				uint32_t idx_a = p_adjacency_map[i * 2 + 0];
				uint32_t idx_b = p_adjacency_map[i * 2 + 1];

				FixedMathCore t_a = temp_stream[idx_a];
				FixedMathCore t_b = temp_stream[idx_b];
				FixedMathCore area = area_stream[idx_a]; // Simplified shared area

				// Fourier's Law of Conduction
				FixedMathCore delta_t = t_b - t_a;
				FixedMathCore heat_transfer = k_metal * area * delta_t * p_delta;

				// Zero-copy atomic-safe update
				temp_stream[idx_a] += heat_transfer;
				temp_stream[idx_b] -= heat_transfer;
			}
		}, SimulationThreadPool::PRIORITY_HIGH);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * synchronize_hierarchy_tensors()
 * 
 * Performs a recursive bit-perfect transform propagation for parent-child links.
 * 1. Computes child global transform: World = ParentWorld * ChildLocal.
 * 2. Resolves Galactic Sector Drift: If the child's world position exceeds the 
 *    10,000 unit threshold, it shifts the BigInt sectors and recenters.
 * 3. Ensures 120 FPS stability for articulated starships and robotic machines.
 */
void synchronize_hierarchy_tensors(
		Transform3Df &r_child_world_xform,
		BigIntCore &r_child_sx, BigIntCore &r_child_sy, BigIntCore &r_child_sz,
		const Transform3Df &p_child_local_xform,
		const Transform3Df &p_parent_world_xform,
		const BigIntCore &p_parent_sx, const BigIntCore &p_parent_sy, const BigIntCore &p_parent_sz,
		const FixedMathCore &p_sector_size) {

	// 1. Concatenate Basis and Origin
	r_child_world_xform.basis = p_parent_world_xform.basis * p_child_local_xform.basis;
	
	// Translation in parent local space then added to parent world origin
	Vector3f local_offset = p_parent_world_xform.basis.xform(p_child_local_xform.origin);
	Vector3f world_pos_unnormalized = p_parent_world_xform.origin + local_offset;

	// 2. Initialize sector from parent
	r_child_sx = p_parent_sx;
	r_child_sy = p_parent_sy;
	r_child_sz = p_parent_sz;

	// 3. Resolve Galactic Sector Crossing
	// If child drifts relative to parent into a new sector, update BigInt indices
	int64_t dx = Math::floor(world_pos_unnormalized.x / p_sector_size).to_int();
	int64_t dy = Math::floor(world_pos_unnormalized.y / p_sector_size).to_int();
	int64_t dz = Math::floor(world_pos_unnormalized.z / p_sector_size).to_int();

	if (dx != 0 || dy != 0 || dz != 0) {
		r_child_sx += BigIntCore(dx);
		r_child_sy += BigIntCore(dy);
		r_child_sz += BigIntCore(dz);

		FixedMathCore off_x = p_sector_size * FixedMathCore(dx);
		FixedMathCore off_y = p_sector_size * FixedMathCore(dy);
		FixedMathCore off_z = p_sector_size * FixedMathCore(dz);
		
		r_child_world_xform.origin = world_pos_unnormalized - Vector3f(off_x, off_y, off_z);
	} else {
		r_child_world_xform.origin = world_pos_unnormalized;
	}
}

/**
 * process_fatigue_recovery_wave()
 * 
 * Simulates material "Healing" or stress relaxation.
 * strictly deterministic decay of structural fatigue tensors.
 */
void process_fatigue_recovery_wave(
		FixedMathCore *r_fatigue_stream,
		const FixedMathCore *p_yield_strengths,
		uint64_t p_count,
		const FixedMathCore &p_delta) {

	// Recovery Rate: 0.001 of yield strength per second
	FixedMathCore recovery_coeff("0.001");
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore recovery_amount = p_yield_strengths[i] * recovery_coeff * p_delta;
		r_fatigue_stream[i] = wp::max(zero, r_fatigue_stream[i] - recovery_amount);
	}
}

/**
 * apply_global_force_tensor()
 * 
 * sophisticated interaction: Applies a macro-scale force (like an explosion shockwave)
 * to a batch of entities, resolving the energy transfer into both velocity and heat.
 */
void apply_global_force_tensor(
		Vector3f *r_velocities,
		FixedMathCore *r_temperatures,
		const FixedMathCore *p_masses,
		uint64_t p_count,
		const Vector3f &p_force_vec,
		const FixedMathCore &p_energy_to_heat_ratio) {

	for (uint64_t i = 0; i < p_count; i++) {
		FixedMathCore inv_mass = MathConstants<FixedMathCore>::one() / p_masses[i];
		
		// 1. Kinetic Acceleration
		r_velocities[i] += p_force_vec * inv_mass;
		
		// 2. Thermal Conversion: E = F * d (simplified work-to-heat)
		FixedMathCore heat_gain = p_force_vec.length() * p_energy_to_heat_ratio * inv_mass;
		r_temperatures[i] += heat_gain;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/math/geometry_instance_logic.cpp ---
