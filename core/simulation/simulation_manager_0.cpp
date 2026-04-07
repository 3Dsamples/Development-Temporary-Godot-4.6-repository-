--- START OF FILE core/simulation/simulation_manager.cpp ---

#include "core/simulation/simulation_manager.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/config/engine.h"

SimulationManager *SimulationManager::singleton = nullptr;

// ============================================================================
// Method Bindings (Godot Native Integration)
// ============================================================================

void SimulationManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("advance_simulation", "delta"), &SimulationManager::advance_simulation);
	ClassDB::bind_method(D_METHOD("set_fixed_fps", "fps"), &SimulationManager::set_fixed_fps);
	ClassDB::bind_method(D_METHOD("set_time_scale", "scale"), &SimulationManager::set_time_scale);
	ClassDB::bind_method(D_METHOD("get_time_scale"), &SimulationManager::get_time_scale);
	ClassDB::bind_method(D_METHOD("set_paused", "paused"), &SimulationManager::set_paused);
	ClassDB::bind_method(D_METHOD("is_paused"), &SimulationManager::is_paused);
	ClassDB::bind_method(D_METHOD("get_total_frames"), &SimulationManager::get_total_frames);
	ClassDB::bind_method(D_METHOD("get_simulation_time"), &SimulationManager::get_simulation_time);

	BIND_ENUM_CONSTANT(TIER_DETERMINISTIC);
	BIND_ENUM_CONSTANT(TIER_MACRO_ECONOMY);
}

// ============================================================================
// Lifecycle Management
// ============================================================================

SimulationManager::SimulationManager() {
	singleton = this;
	
	// Default to 120 FPS step (1 / 120 seconds) in bit-perfect FixedMath
	// 120Hz = 0.0083333333333333 seconds
	fixed_step_time = FixedMathCore(1LL, false) / FixedMathCore(120LL, false);
	deterministic_accumulator = FixedMathCore(0LL, true);
	time_scale = FixedMathCore(1LL, false);
	
	total_frames = BigIntCore(0LL);
	physics_frames = BigIntCore(0LL);
}

SimulationManager::~SimulationManager() {
	singleton = nullptr;
}

// ============================================================================
// Heartbeat API (120 FPS Execution)
// ============================================================================

/**
 * advance_simulation()
 * 
 * The master loop entry. Converts the variable OS delta into a 
 * deterministic accumulation buffer. This handles "Catch-up" logic 
 * if the hardware lags, while maintaining the 120Hz simulation speed.
 */
void SimulationManager::advance_simulation(const FixedMathCore &p_delta) {
	if (unlikely(paused)) {
		return;
	}

	total_frames += BigIntCore(1LL);
	deterministic_accumulator += p_delta * time_scale;

	// Loop to satisfy the fixed-step requirement (120 FPS)
	// Prevents "Spiral of Death" by limiting catch-up steps per frame
	int safety_limit = 0;
	while (deterministic_accumulator >= fixed_step_time && safety_limit < 8) {
		step_deterministic(fixed_step_time);
		deterministic_accumulator -= fixed_step_time;
		safety_limit++;
	}
}

/**
 * step_deterministic()
 * 
 * Performs a single bit-perfect simulation tick.
 * This is where EnTT SoA buffers are fed into Warp Kernels for
 * zero-copy physics resolution and logic updates.
 */
void SimulationManager::step_deterministic(const FixedMathCore &p_fixed_delta) {
	physics_frames += BigIntCore(1LL);

	// 1. TIER_DETERMINISTIC: Warp-Style Parallel Physics Kernels
	// (Call to PhysicsServerHyper or Warp Kernel Dispatcher)

	// 2. TIER_MACRO_ECONOMY: Infinite Scale Discrete Logic
	// (Logic involving BigIntCore ledgers and galactic distances)
	
	// ETEngine Strategy: Flush CommandQueueMT after each deterministic step
	// to synchronize cross-thread simulation events.
}

// ============================================================================
// Configuration & Telemetry
// ============================================================================

void SimulationManager::set_fixed_fps(int p_fps) {
	if (p_fps <= 0) return;
	fixed_step_time = FixedMathCore(1LL, false) / FixedMathCore(static_cast<int64_t>(p_fps));
}

void SimulationManager::set_time_scale(const FixedMathCore &p_scale) {
	time_scale = p_scale;
}

void SimulationManager::set_paused(bool p_paused) {
	paused = p_paused;
}

FixedMathCore SimulationManager::get_simulation_time() const {
	// time = physics_frames * fixed_step_time
	// Calculated using BigInt to Fixed conversion to avoid overflow
	FixedMathCore frames_f(static_cast<int64_t>(std::stoll(physics_frames.to_string())));
	return frames_f * fixed_step_time;
}

--- END OF FILE core/simulation/simulation_manager.cpp ---
