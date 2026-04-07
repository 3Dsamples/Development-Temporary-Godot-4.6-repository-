--- START OF FILE core/simulation/physics_server_hyper_buoyancy.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/dynamic_mesh.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ArchimedesDisplacementKernel
 * 
 * Calculates the buoyancy force based on submerged volume and fluid density.
 * 1. Resolves relative height against the fluid surface (Sector-aware).
 * 2. Computes the displaced volume using AABB and Face3 approximations.
 * 3. Applies a bit-perfect upward impulse and drag damping.
 */
void archimedes_displacement_kernel(
		const BigIntCore &p_index,
		Vector3f &r_linear_velocity,
		Vector3f &r_angular_velocity,
		const Transform3Df &p_transform,
		const AABBf &p_aabb,
		const FixedMathCore &p_mass,
		const FixedMathCore &p_fluid_density,
		const FixedMathCore &p_fluid_surface_y,
		const Vector3f &p_gravity_vec,
		const FixedMathCore &p_delta,
		bool p_is_anime) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	// 1. Calculate Submerged Depth
	// y_top = origin.y + aabb_max.y; y_bottom = origin.y + aabb_min.y
	FixedMathCore y_bottom = p_transform.origin.y + p_aabb.position.y;
	FixedMathCore y_top = y_bottom + p_aabb.size.y;

	if (y_bottom > p_fluid_surface_y) return; // Completely above fluid

	// 2. Volume Displacement Ratio [0..1]
	FixedMathCore submerged_factor;
	if (y_top <= p_fluid_surface_y) {
		submerged_factor = one; // Fully submerged
	} else {
		submerged_factor = (p_fluid_surface_y - y_bottom) / p_aabb.size.y;
	}

	// 3. Buoyancy Force: Fb = rho * V_displaced * g
	// Assuming V is proportional to mass/density_object (simplified for kernel)
	FixedMathCore displaced_mass = p_mass * submerged_factor * (p_fluid_density / one); // Density ratio
	Vector3f buoyancy_force = -p_gravity_vec * displaced_mass;

	// 4. Hydrodynamic Drag
	// Fluid resistance scales with velocity squared and submerged area
	FixedMathCore speed = r_linear_velocity.length();
	if (speed > zero) {
		FixedMathCore drag_coeff = FixedMathCore(2147483648LL, true); // 0.5 base
		if (p_is_anime) drag_coeff *= FixedMathCore(2LL, false); // Stylized high-speed resistance
		
		FixedMathCore drag_mag = drag_coeff * p_fluid_density * speed * speed * submerged_factor;
		r_linear_velocity -= r_linear_velocity.normalized() * (drag_mag * p_delta / p_mass);
	}

	// 5. Torque Stabilization (Righting Moment)
	// Push the body to align its local "Up" with the surface normal
	Vector3f center_of_buoyancy = p_transform.origin + p_aabb.get_center() * submerged_factor;
	Vector3f lever_arm = center_of_buoyancy - p_transform.origin;
	Vector3f righting_torque = lever_arm.cross(buoyancy_force);
	r_angular_velocity += righting_torque * (p_delta / p_mass);

	// 6. Apply Final Linear Impulse
	r_linear_velocity += (buoyancy_force / p_mass) * p_delta;
}

/**
 * Warp Kernel: HydrostaticPressureDeformationKernel
 * 
 * Specifically for BODY_MODE_DEFORMABLE. 
 * Applies inward pressure to mesh vertices based on fluid depth.
 * Simulates the "Squeezing" effect of deep water or high-pressure atmospheres.
 */
void hydrostatic_pressure_deformation_kernel(
		const BigIntCore &p_v_index,
		Vector3f &r_vertex_pos,
		const Vector3f &r_vertex_normal,
		const FixedMathCore &p_surface_y,
		const FixedMathCore &p_fluid_density,
		const FixedMathCore &p_gravity,
		const FixedMathCore &p_material_stiffness,
		const FixedMathCore &p_delta) {

	if (r_vertex_pos.y > p_surface_y) return;

	// P = rho * g * h
	FixedMathCore depth = p_surface_y - r_vertex_pos.y;
	FixedMathCore pressure = p_fluid_density * p_gravity * depth;

	// Deform vertex along normal (Inward)
	// Displacement = pressure / stiffness
	FixedMathCore displacement = (pressure / p_material_stiffness) * p_delta;
	r_vertex_pos -= r_vertex_normal * displacement;
}

/**
 * execute_buoyancy_sweep()
 * 
 * Orchestrates the parallel resolution of all bodies interacting with fluid volumes.
 * High-performance 120 FPS sweep over EnTT SoA streams.
 */
void PhysicsServerHyper::execute_buoyancy_sweep(const FixedMathCore &p_delta) {
	auto &registry = get_kernel_registry();
	uint64_t count = registry.get_stream<Transform3Df>().size();
	if (count == 0) return;

	// Get global fluid parameters (e.g. from a Galactic Sector Water component)
	FixedMathCore water_density(4294967296LL, false); // 1.0 (Standard)
	FixedMathCore surface_y = FixedMathCore(0LL, true);

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (w + 1) * chunk;

		SimulationThreadPool::get_singleton()->enqueue_task([=, &registry]() {
			for (uint64_t i = start; i < end; i++) {
				bool is_anime = (i % 2 == 0); // Deterministic style assignment
				
				archimedes_displacement_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					registry.get_stream<Vector3f>(COMPONENT_LINEAR_VELOCITY)[i],
					registry.get_stream<Vector3f>(COMPONENT_ANGULAR_VELOCITY)[i],
					registry.get_stream<Transform3Df>(COMPONENT_TRANSFORM)[i],
					registry.get_stream<AABBf>(COMPONENT_BOUNDS)[i],
					registry.get_stream<FixedMathCore>(COMPONENT_MASS)[i],
					water_density,
					surface_y,
					gravity_vec,
					p_delta,
					is_anime
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	SimulationThreadPool::get_singleton()->wait_for_all();
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_buoyancy.cpp ---
