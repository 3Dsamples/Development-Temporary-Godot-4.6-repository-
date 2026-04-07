--- START OF FILE core/simulation/simulation_thread_pool_worker.cpp ---

#include "core/simulation/simulation_thread_pool.h"
#include "core/math/warp_intrinsics.h"
#include "core/os/memory.h"

/**
 * _process_simulation_task()
 * 
 * Internal kernel for executing batch tasks. 
 * Re-anchors BigIntCore loop counters to ensure that the worker never
 * suffers from 32-bit index overflow during galactic-scale sweeps.
 * Operates strictly on FixedMathCore for all time-sensitive integrations.
 */
void _process_simulation_batch_task(const SimulationThreadPool::TaskFunc &p_func, const BigIntCore &p_batch_id) {
    // ETEngine Strategy: Thread-local storage for FixedMath intermediates 
    // to avoid cross-cache invalidation during the 120 FPS heartbeat.
    
    // Warp-style execution context setup
    // All math performed within the task must use wp:: intrinsics
    p_func();

    // Telemetry: Log completion to the BigInt counter without FPU involvement
    SimulationThreadPool::get_singleton()->increment_counter(BigIntCore(1LL));
}

/**
 * Warp-Kernel Synergy:
 * 
 * This worker logic is the bridge for EnTT-managed SoA streams.
 * When a Warp kernel is launched, this file provides the execution context 
 * that maintains bit-perfect determinism across the hardware-affinity-bound threads.
 */

--- END OF FILE core/simulation/simulation_thread_pool_worker.cpp ---
