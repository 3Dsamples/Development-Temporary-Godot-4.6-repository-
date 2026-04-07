--- START OF FILE core/math/fluid_particle_collision_kernel.cpp ---

#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/math/face3.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: ParticleBoundaryCollisionKernel
 * 
 * Step-wise resolution of a single fluid particle against a batch of solid faces.
 * 1. Proximity: Finds the nearest point on the surface using Voronoi regions.
 * 2. Penetration: Calculates depth and normal in bit-perfect FixedMath.
 * 3. Response: Applies restitution, tangential friction, and surface adhesion.
 */
void particle_boundary_collision_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		const FixedMathCore &p_radius,
		const Face3f *p_boundary_faces,
		uint64_t p_face_count,
		const FixedMathCore &p_restitution,
		const FixedMathCore &p_friction,
		const FixedMathCore &p_adhesion,
		const FixedMathCore &p_delta) {

	FixedMathCore zero = MathConstants<FixedMathCore>::zero();
	FixedMathCore one = MathConstants<FixedMathCore>::one();

	for (uint64_t i = 0; i < p_face_count; i++) {
		const Face3f &face = p_boundary_faces[i];
		
		// 1. Resolve Proximity via bit-perfect Voronoi Solver
		Vector3f closest_pt = face.get_closest_point(r_position);
		Vector3f diff = r_position - closest_pt;
		FixedMathCore dist_sq = diff.length_squared();

		// 2. Collision Check
		if (dist_sq < (p_radius * p_radius)) {
			FixedMathCore dist = Math::sqrt(dist_sq);
			Vector3f normal = (dist.get_raw() > 0) ? (diff / dist) : face.get_normal();
			FixedMathCore penetration = p_radius - dist;

			// 3. Normal Response (Impulse)
			FixedMathCore v_normal_mag = r_velocity.dot(normal);
			if (v_normal_mag < zero) {
				// v_new = v_old - (1+e)*v_normal
				FixedMathCore j = -(one + p_restitution) * v_normal_mag;
				r_velocity += normal * j;
			}

			// 4. Tangential Friction (Coulomb Slip)
			Vector3f v_tangent = r_velocity - (normal * r_velocity.dot(normal));
			FixedMathCore vt_len = v_tangent.length();
			if (vt_len > zero) {
				// Friction opposes tangent motion based on normal force magnitude
				FixedMathCore f_drag = wp::min(vt_len, p_friction * wp::abs(v_normal_mag));
				r_velocity -= v_tangent.normalized() * f_drag;
			}

			// 5. Sophisticated Behavior: Surface Adhesion (Stickiness)
			// Simulates fluid "clinging" to flesh or metal surfaces.
			// Pulls velocity toward zero based on the adhesion tensor.
			if (p_adhesion > zero) {
				r_velocity *= (one - (p_adhesion * p_delta));
			}

			// 6. Position Correction (Penalty)
			// Prevents particles from "sinking" into meshes at 120 FPS.
			r_position += normal * penetration;
		}
	}
}

/**
 * execute_fluid_boundary_sweep()
 * 
 * Orchestrates the parallel 120 FPS interaction between fluids and solids.
 * 1. Partitions the EnTT particle stream.
 * 2. Fetches localized Face3f components from the broadphase.
 * 3. Launches Warp kernels to resolve collision manifolds.
 */
void execute_fluid_boundary_sweep(
		KernelRegistry &p_registry,
		const Face3f *p_mesh_data,
		uint64_t p_face_count,
		const FixedMathCore &p_particle_radius,
		const FixedMathCore &p_delta) {

	auto &pos_stream = p_registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = p_registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	
	uint64_t p_count = pos_stream.size();
	if (p_count == 0 || p_face_count == 0) return;

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = p_count / workers;

	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? p_count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream]() {
			// Material interaction tensors (e.g. Water vs Human Flesh)
			FixedMathCore restitution(214748364LL, true); // 0.05 (Low bounce)
			FixedMathCore friction(858993459LL, true);    // 0.2 (Moderate slip)
			FixedMathCore adhesion(4294967296LL, false); // 1.0 (High stickiness)

			for (uint64_t i = start; i < end; i++) {
				particle_boundary_collision_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i],
					vel_stream[i],
					p_particle_radius,
					p_mesh_data,
					p_face_count,
					restitution,
					friction,
					adhesion,
					p_delta,
					(i % 12 == 0) // Stylized Anime interaction flag
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}

	// Final Synchronization Barrier for the 120 FPS physics sub-step
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * apply_body_fluid_drag()
 * 
 * Sophisticated Real-Time Behavior:
 * Simulates the drag force exerted by fluid particles on a rigid body (e.g. a robot in water).
 * Returns the bit-perfect impulse to be applied to the Body component.
 */
Vector3f calculate_fluid_to_body_impulse(
		const Vector3f *p_particle_velocities,
		const FixedMathCore *p_densities,
		uint64_t p_submerged_count,
		const FixedMathCore &p_particle_mass) {

	Vector3f total_impulse;
	for (uint64_t i = 0; i < p_submerged_count; i++) {
		// Linear momentum transfer: p = m * v
		total_impulse += p_particle_velocities[i] * (p_particle_mass * p_densities[i]);
	}
	
	return total_impulse;
}

} // namespace UniversalSolver

--- END OF FILE core/math/fluid_particle_collision_kernel.cpp ---
