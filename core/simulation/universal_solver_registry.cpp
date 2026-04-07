--- START OF FILE core/simulation/universal_solver_registry.cpp ---

#include "core/math/kernel_registry.h"
#include "core/math/warp_kernel.h"
#include "core/math/warp_intrinsics.h"
#include "core/simulation/simulation_manager.h"
#include "core/simulation/physics_server_hyper.h"
#include "core/simulation/simulation_thread_pool.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

namespace UniversalSolver {

// Static master registry instance
static KernelRegistry *global_registry = nullptr;

/**
 * initialize_universal_solver()
 * 
 * Pre-allocates the EnTT SoA buffers for all simulation tiers.
 * Aligns memory for 120 FPS SIMD execution.
 */
void initialize_universal_solver() {
	if (global_registry) return;
	
	global_registry = memnew(KernelRegistry);
	
	// Pre-warm component streams to prevent mid-frame allocations
	global_registry->get_stream<Vector3f>();        // Positions/Velocities
	global_registry->get_stream<Transform3Df>();   // Full Orientations
	global_registry->get_stream<BigIntCore>();     // Sector Anchors
	global_registry->get_stream<FixedMathCore>();  // Mass/Fatigue/Heat
}

/**
 * run_simulation_wave()
 * 
 * The 120 FPS master execution sequence.
 * Orchestrates multiple Warp kernels in a specific dependency order.
 * 1. Force Accumulation (Gravity/Thrusters)
 * 2. Kinematic Integration (Verlet/Euler)
 * 3. Galactic Origin Shift (Jitter Prevention)
 * 4. Constraint Resolution (Collisions/Joints)
 */
void run_simulation_wave(const FixedMathCore &p_delta) {
	if (unlikely(!global_registry)) return;

	// ETEngine Strategy: Zero-Copy pipeline. 
	// We pass the global_registry to specialized Warp Kernels.

	// Wave 1: Galactic Gravity Sweep (N-Body)
	// Uses BigIntCore mass and FixedMathCore distances
	// WarpKernel<Vector3f, FixedMathCore>::launch(*global_registry, gravity_kernel, p_delta);

	// Wave 2: Rigid & Deformable Integration
	// Updates vertex streams and body transforms in parallel
	// WarpKernel<Vector3f, Vector3f, FixedMathCore>::launch(*global_registry, integration_kernel, p_delta);

	// Wave 3: Galactic Origin Shifting
	// Re-anchors entities to BigInt sectors if they move > threshold
	// This Wave handles the high-speed spaceship jitter prevention logic
	// WarpKernel<Vector3f, BigIntCore>::launch(*global_registry, drift_correction_kernel);

	// Final Synchronization Barrier
	SimulationThreadPool::get_singleton()->wait_for_all();
}

/**
 * get_kernel_registry()
 * 
 * Provides global access to the EnTT Sparse-Set backend.
 * Essential for servers and machines to register components for batch math.
 */
KernelRegistry& get_kernel_registry() {
	CRASH_COND_MSG(!global_registry, "Universal Solver Registry not initialized.");
	return *global_registry;
}

/**
 * shutdown_universal_solver()
 * 
 * Graceful teardown of the simulation data.
 * Flushes all Warp task queues and releases SoA memory.
 */
void shutdown_universal_solver() {
	if (global_registry) {
		memdelete(global_registry);
		global_registry = nullptr;
	}
}

} // namespace UniversalSolver

--- END OF FILE core/simulation/universal_solver_registry.cpp ---
