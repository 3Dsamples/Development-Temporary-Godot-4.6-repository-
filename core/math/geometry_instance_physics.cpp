--- START OF FILE core/math/geometry_instance_physics.cpp ---

#include "core/math/geometry_instance.h"
#include "core/math/math_funcs.h"
#include "core/simulation/simulation_manager.h"
#include "core/math/collision_solver.h"
#include "core/os/memory.h"

/**
 * _update_internal_physics()
 * 
 * Orchestrates the macroscopic physical evolution of the instance.
 * Called during the TIER_DETERMINISTIC sweep. It processes material fatigue 
 * recovery and global thermal radiation.
 */
template <typename T>
void GeometryInstance<T>::_update_internal_physics(T p_delta) {
	if (mesh_data.is_null() || current_state == STATE_DESTROYED) {
		return;
	}

	// 1. Structural Integrity & Fatigue Relaxation
	// In the Universal Solver, we simulate the 'healing' or 'settling' of material.
	if (current_state == STATE_DEFORMING) {
		T recovery_rate = params.elastic_recovery_rate * p_delta;
		current_integrity += recovery_rate;
		
		if (current_integrity >= MathConstants<T>::one()) {
			current_integrity = MathConstants<T>::one();
			current_state = STATE_STABLE;
		}
	}

	// 2. Global Thermal Radiation
	// Simulates the cooling of the entire object into the environment.
	// Ambient temperature in Kelvin (293.15) represented in FixedMathCore.
	T ambient_temp(12591030272LL, true); 
	T thermal_loss_coeff(42949673LL, true); // 0.01 conductivity
	
	T temp_diff = internal_temperature - ambient_temp;
	internal_temperature -= temp_diff * thermal_loss_coeff * p_delta;

	// 3. Resolve Galactic Sector Transition
	// Ensures the object is always anchored to the nearest BigIntCore sector
	// to prevent precision loss in the FixedMathCore local coordinates.
	_handle_sector_transition();
}

/**
 * apply_impact()
 * 
 * Receives a physical impulse from the world space. 
 * It performs a zero-copy coordinate transformation into the local 
 * mesh tensor space and triggers localized plastic deformation.
 */
template <typename T>
void GeometryInstance<T>::apply_impact(const Vector3<T> &p_world_point, const Vector3<T> &p_force_vec, T p_radius) {
	if (mesh_data.is_null()) return;

	// 1. Sector-Aware Transformation
	// Project the world-space impact into the local mesh space using bit-perfect inverse transforms.
	Vector3<T> local_point = world_transform.xform_inv(p_world_point);
	Vector3<T> local_force = world_transform.basis.inverse().xform(p_force_vec);

	// 2. Mesh-Level Impact Processing
	// Directly modifies the SoA vertex streams in the DynamicMesh.
	mesh_data->apply_impact(local_point, local_force.normalized(), local_force.length(), p_radius);

	// 3. Energy-to-Heat Conversion
	// Kinetic energy absorbed results in localized thermal spikes.
	T kinetic_energy = local_force.length() * p_radius;
	T heating_factor(4294967LL, true); // 0.001 efficiency
	internal_temperature += (kinetic_energy * heating_factor) / (params.mass + MathConstants<T>::one());

	// 4. Update Macroscopic State
	T yield_check = local_force.length() / (params.yield_strength + MathConstants<T>::one());
	if (yield_check > T(10LL, false)) { // 10.0x yield triggers fracture
		current_state = STATE_FRACTURING;
	} else {
		current_state = STATE_DEFORMING;
		current_integrity -= yield_check * T(42949673LL, true); // Minor integrity loss
	}

	aabb_dirty = true;
}

/**
 * apply_screw_torsion()
 * 
 * Simulates mechanical twisting applied to the object. 
 * Used for structural damage simulation in large-scale vehicles or stations.
 */
template <typename T>
void GeometryInstance<T>::apply_screw_torsion(const Vector3<T> &p_axis_origin, const Vector3<T> &p_axis_dir, T p_torque) {
	if (mesh_data.is_null()) return;

	Vector3<T> local_origin = world_transform.xform_inv(p_axis_origin);
	Vector3<T> local_dir = world_transform.basis.inverse().xform(p_axis_dir).normalized();

	// Pass torque into the deterministic mesh kernel
	mesh_data->apply_torsional_screw(local_origin, local_dir, p_torque, (mesh_data->get_bounds().size.length() * MathConstants<T>::half()));

	current_state = STATE_DEFORMING;
	aabb_dirty = true;
}

/**
 * apply_bend()
 * 
 * Simulates the physical folding of the geometry around a hinge axis.
 */
template <typename T>
void GeometryInstance<T>::apply_bend(const Vector3<T> &p_pivot_origin, const Vector3<T> &p_axis_dir, T p_angle) {
	if (mesh_data.is_null()) return;

	Vector3<T> local_pivot = world_transform.xform_inv(p_pivot_origin);
	Vector3<T> local_axis = world_transform.basis.inverse().xform(p_axis_dir).normalized();

	mesh_data->apply_structural_bend(local_pivot, local_axis, p_angle, (mesh_data->get_bounds().size.length()));

	current_state = STATE_DEFORMING;
	aabb_dirty = true;
}

// Explicit template instantiations for the linker
template class GeometryInstance<FixedMathCore>;

--- END OF FILE core/math/geometry_instance_physics.cpp ---
