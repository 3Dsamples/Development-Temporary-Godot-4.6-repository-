--- START OF FILE core/simulation/physics_server_hyper_integrator.cpp ---

#include "core/simulation/physics_server_hyper.h"
#include "core/math/math_funcs.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * Warp Kernel: DeterministicKinematicKernel
 * 
 * Performs the fundamental physical update for a 120 FPS heartbeat.
 * 1. Force Integration: v = v + (F/m) * dt
 * 2. Relativistic Correction: Applies Lorentz factor for high-speed ship frames.
 * 3. Angular Resolve: Updates Basis orientation using angular velocity tensors.
 * 4. Temporal Dilation: Accumulates proper time based on velocity.
 */
void deterministic_kinematic_kernel(
		const BigIntCore &p_index,
		Vector3f &r_position,
		Vector3f &r_velocity,
		Basis<FixedMathCore> &r_orientation,
		Vector3f &r_angular_velocity,
		FixedMathCore &r_proper_time,
		const Vector3f &p_force,
		const Vector3f &p_torque,
		const FixedMathCore &p_inv_mass,
		const FixedMathCore &p_inv_inertia,
		const FixedMathCore &p_delta,
		const FixedMathCore &p_c_sq) {

	FixedMathCore one = MathConstants<FixedMathCore>::one();
	FixedMathCore zero = MathConstants<FixedMathCore>::zero();

	// 1. Resolve Linear Velocity with External Force Tensors
	r_velocity += p_force * p_inv_mass * p_delta;

	// 2. Relativistic Correction (Lorentz Transform)
	// gamma = 1 / sqrt(1 - v^2/c^2)
	FixedMathCore v2 = r_velocity.length_squared();
	FixedMathCore beta2 = wp::min(v2 / p_c_sq, FixedMathCore(4294967290LL, true)); // 0.999c cap
	FixedMathCore inv_gamma = (one - beta2).square_root();
	
	// Proper Time Integration: local clock slows down at high speeds
	r_proper_time += p_delta * inv_gamma;

	// 3. Resolve Angular Velocity
	r_angular_velocity += p_torque * p_inv_inertia * p_delta;

	// 4. Update Spatial Position (Bit-Perfect Linear Translation)
	// Even at warp speeds, world-space displacement is v * dt_world
	r_position += r_velocity * p_delta;

	// 5. Update Orientation Basis
	// Uses bit-perfect Rodrigues' rotation formula via Basis::rotate
	FixedMathCore ang_speed = r_angular_velocity.length();
	if (ang_speed > MathConstants<FixedMathCore>::unit_epsilon()) {
		Vector3f axis = r_angular_velocity / ang_speed;
		FixedMathCore angle = ang_speed * p_delta;
		r_orientation.rotate(axis, angle);
		
		// Deterministic Re-Orthonormalization to prevent matrix skew
		r_orientation.orthonormalize();
	}
}

/**
 * Warp Kernel: GalacticSectorSyncKernel
 * 
 * Sophisticated Feature: Handles the hand-off between FixedMathCore and BigIntCore.
 * If an entity (e.g. high-speed ship) moves beyond the 10,000 unit threshold,
 * it increments the sector coordinate and resets the local origin.
 */
void galactic_sector_sync_kernel(
		const BigIntCore &p_index,
		Vector3f &r_local_pos,
		BigIntCore &r_sx, BigIntCore &r_sy, BigIntCore &r_sz,
		const FixedMathCore &p_sector_size) {

	// Threshold-based drift correction (Zero-Jitter Logic)
	int64_t move_x = Math::floor(r_local_pos.x / p_sector_size).to_int();
	int64_t move_y = Math::floor(r_local_pos.y / p_sector_size).to_int();
	int64_t move_z = Math::floor(r_local_pos.z / p_sector_size).to_int();

	if (move_x != 0 || move_y != 0 || move_z != 0) {
		// Advance Galactic Sectors via arbitrary-precision BigInt
		r_sx += BigIntCore(move_x);
		r_sy += BigIntCore(move_y);
		r_sz += BigIntCore(move_z);

		// Recenter local FixedMath position bit-perfectly
		r_local_pos.x -= p_sector_size * FixedMathCore(move_x);
		r_local_pos.y -= p_sector_size * FixedMathCore(move_y);
		r_local_pos.z -= p_sector_size * FixedMathCore(move_z);
	}
}

/**
 * execute_kinematic_integration_wave()
 * 
 * Orchestrates the master 120 FPS integration sweep across the EnTT registry.
 * Zero-copy: Operates directly on the aligned SoA memory buffers.
 */
void PhysicsServerHyper::execute_kinematic_integration_wave(const FixedMathCore &p_delta) {
	KernelRegistry &registry = get_kernel_registry();
	auto &pos_stream = registry.get_stream<Vector3f>(COMPONENT_POSITION);
	auto &vel_stream = registry.get_stream<Vector3f>(COMPONENT_VELOCITY);
	auto &basis_stream = registry.get_stream<Basis<FixedMathCore>>(COMPONENT_ORIENTATION);
	auto &ang_vel_stream = registry.get_stream<Vector3f>(COMPONENT_ANGULAR_VELOCITY);
	auto &force_stream = registry.get_stream<Vector3f>(COMPONENT_FORCE_ACCUM);
	auto &torque_stream = registry.get_stream<Vector3f>(COMPONENT_TORQUE_ACCUM);
	auto &inv_mass_stream = registry.get_stream<FixedMathCore>(COMPONENT_INV_MASS);
	auto &inv_inertia_stream = registry.get_stream<FixedMathCore>(COMPONENT_INV_INERTIA);
	auto &time_stream = registry.get_stream<FixedMathCore>(COMPONENT_PROPER_TIME);
	
	auto &sx_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_X);
	auto &sy_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Y);
	auto &sz_stream = registry.get_stream<BigIntCore>(COMPONENT_SECTOR_Z);

	uint64_t count = pos_stream.size();
	if (count == 0) return;

	FixedMathCore c_sq = PHYSICS_C * PHYSICS_C;
	FixedMathCore sector_size(10000LL, false);

	uint32_t workers = SimulationThreadPool::get_singleton()->get_worker_count();
	uint64_t chunk = count / workers;

	// PASS 1: Parallel Deterministic Integration
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &vel_stream, &basis_stream, &ang_vel_stream, &force_stream, &torque_stream, &inv_mass_stream, &inv_inertia_stream, &time_stream]() {
			for (uint64_t i = start; i < end; i++) {
				deterministic_kinematic_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i], vel_stream[i], basis_stream[i], ang_vel_stream[i],
					time_stream[i], force_stream[i], torque_stream[i],
					inv_mass_stream[i], inv_inertia_stream[i],
					p_delta, c_sq
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// PASS 2: Parallel Galactic Sector Sync
	for (uint32_t w = 0; w < workers; w++) {
		uint64_t start = w * chunk;
		uint64_t end = (w == workers - 1) ? count : (start + chunk);

		SimulationThreadPool::get_singleton()->enqueue_task([=, &pos_stream, &sx_stream, &sy_stream, &sz_stream]() {
			for (uint64_t i = start; i < end; i++) {
				galactic_sector_sync_kernel(
					BigIntCore(static_cast<int64_t>(i)),
					pos_stream[i], sx_stream[i], sy_stream[i], sz_stream[i],
					sector_size
				);
			}
		}, SimulationThreadPool::PRIORITY_CRITICAL);
	}
	SimulationThreadPool::get_singleton()->wait_for_all();

	// Reset force accumulators for the next wave
	for (uint64_t i = 0; i < count; i++) {
		force_stream[i] = Vector3f_ZERO;
		torque_stream[i] = Vector3f_ZERO;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/physics_server_hyper_integrator.cpp ---
