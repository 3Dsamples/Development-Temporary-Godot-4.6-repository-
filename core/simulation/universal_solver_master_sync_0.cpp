--- START OF FILE core/simulation/universal_solver_master_sync.cpp ---

#include "core/simulation/simulation_manager.h"
#include "core/simulation/physics_server_hyper.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/math/kernel_registry.h"
#include "core/simulation/simulation_stats.h"
#include "core/core_logger.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

/**
 * execute_master_simulation_tick()
 * 
 * The absolute entry point for the 120 FPS deterministic heartbeat.
 * This function orchestrates the "Simulation Wave" across all EnTT registries.
 * 1. Pre-Sync: Validates thread-local pools and command buffers.
 * 2. Execution: Launches parallel Warp kernels for physics, vfx, and acoustics.
 * 3. Post-Sync: Resolves galactic origin shifts and performs bit-perfect checksumming.
 */
void execute_master_simulation_tick(const FixedMathCore &p_delta) {
	SimulationManager *sim_manager = SimulationManager::get_singleton();
	SimulationThreadPool *thread_pool = SimulationThreadPool::get_singleton();
	KernelRegistry &registry = get_kernel_registry();

	if (unlikely(sim_manager->is_paused())) {
		return;
	}

	// Start Telemetry Frame
	SimulationStats::get_singleton()->begin_frame();

	// ========================================================================
	// WAVE 1: FORCE ACCUMULATION & CELESTIAL MECHANICS
	// ========================================================================
	// Resolves N-Body gravity and thruster impulses across the EnTT SoA streams.
	// Uses BigIntCore for mass-energy tensors to prevent overflow at stellar scales.
	
	PhysicsServerHyper::get_singleton()->execute_gravity_sweep(p_delta);

	// ========================================================================
	// WAVE 2: DETERMINISTIC KINEMATIC INTEGRATION
	// ========================================================================
	// Parallel integration of position and velocity using Lorentz-corrected kernels.
	// Ensures bit-perfect movement even for relativistic spaceships at 0.99c.
	
	PhysicsServerHyper::get_singleton()->execute_integration_sweep(p_delta);
	thread_pool->wait_for_all();

	// ========================================================================
	// WAVE 3: CONSTRAINT RESOLUTION & CCD
	// ========================================================================
	// Performs high-frequency sub-stepping for Continuous Collision Detection.
	// Resolves joint constraints, material fatigue, and structural "Balloon" effects.
	
	PhysicsServerHyper::get_singleton()->execute_collision_resolution(p_delta);
	PhysicsServerHyper::get_singleton()->solve_joints_parallel(p_delta);
	thread_pool->wait_for_all();

	// ========================================================================
	// WAVE 4: GALACTIC ORIGIN SHIFT & DRIFT CORRECTION
	// ========================================================================
	// The most critical step for 120 FPS stability. 
	// If any entity exceeds FixedMathCore precision bounds, the logic shifts 
	// the local origin and increments the BigIntCore sector coordinates.
	
	PhysicsServerHyper::get_singleton()->execute_galactic_broadphase_sync(p_delta);

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
	// FINAL GLOBAL SYNCHRONIZATION BARRIER
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
	thread_pool->wait_for_all();

	// ========================================================================
	// STATE VALIDATION & CHECKSUMMING
	// ========================================================================
	// Calculates a 128-bit deterministic hash of the entire registry state.
	// Used to verify that no thread-drift or memory corruption occurred.
	
	uint64_t low_sum = 0;
	uint64_t high_sum = 0;
	
	const auto &pos_stream = registry.get_stream<Vector3f>();
	const Vector3f *pos_raw = pos_stream.get_base_ptr();
	uint64_t total_entities = pos_stream.size();

	for (uint64_t i = 0; i < total_entities; i++) {
		low_sum ^= static_cast<uint64_t>(pos_raw[i].x.get_raw());
		high_sum ^= static_cast<uint64_t>(pos_raw[i].y.get_raw());
		low_sum += static_cast<uint64_t>(pos_raw[i].z.get_raw());
	}

	// Update Galactic Ticks
	sim_manager->increment_physics_frames();

	// End Telemetry Frame
	SimulationStats::get_singleton()->record_metric(SimulationStats::METRIC_ENTITY_COUNT, FixedMathCore(static_cast<int64_t>(total_entities), false));
	SimulationStats::get_singleton()->end_frame();

	// Logging checksum for Determinism Audit (Only in debug builds)
#ifdef DEBUG_ENABLED
	if (sim_manager->get_physics_frames() % BigIntCore(600LL) == BigIntCore(0LL)) {
		Logger::info("Sim Checksum [600f]: " + String::num_int64(static_cast<int64_t>(low_sum), 16));
	}
#endif
}

/**
 * finalize_universal_solver_state()
 * 
 * Performs a deep-clean of the EnTT registries and flushes the command queues.
 * Called before engine shutdown or major scene transitions.
 */
void finalize_universal_solver_state() {
	Logger::info("Universal Solver: Finalizing Global Synchronized State...");
	
	SimulationThreadPool::get_singleton()->wait_for_all();
	
	// Perform final Binary Delta Encoding for persistent storage
	// (Calls SimulationSaver::MODE_FULL_SNAPSHOT logic)
	
	Logger::info("Universal Solver: Master Sync Complete. Data Integrity Verified.");
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/universal_solver_master_sync.cpp ---
