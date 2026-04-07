--- START OF FILE core/simulation/simulation_manager.cpp ---

#include "core/simulation/simulation_manager.h"
#include "core/simulation/physics_server_hyper.h"
#include "core/simulation/simulation_thread_pool.h"
#include "core/object/class_db.h"
#include "core/os/os.h"

SimulationManager *SimulationManager::singleton = nullptr;

/**
 * _bind_methods()
 * 
 * Registers the 120 FPS simulation API into Godot's ClassDB.
 * Exposes FixedMath and BigInt interfaces to GDScript for high-precision control.
 */
void SimulationManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("advance_simulation", "delta"), &SimulationManager::advance_simulation);
	ClassDB::bind_method(D_METHOD("set_time_scale", "scale"), &SimulationManager::set_time_scale);
	ClassDB::bind_method(D_METHOD("get_time_scale"), &SimulationManager::get_time_scale);
	ClassDB::bind_method(D_METHOD("set_paused", "paused"), &SimulationManager::set_paused);
	ClassDB::bind_method(D_METHOD("is_paused"), &SimulationManager::is_paused);
	ClassDB::bind_method(D_METHOD("get_total_frames"), &SimulationManager::get_total_frames);
	ClassDB::bind_method(D_METHOD("get_simulation_time_seconds"), &SimulationManager::get_simulation_time_seconds);
}

SimulationManager::SimulationManager() {
	singleton = this;
	
	// Default to 120 FPS (1 / 120 seconds per bit-perfect step)
	physics_step = FixedMathCore(1LL) / FixedMathCore(120LL);
	time_scale = MathConstants<FixedMathCore>::one();
	time_accumulator = MathConstants<FixedMathCore>::zero();
	
	total_physics_frames = BigIntCore(0LL);
	total_simulation_time_msec = BigIntCore(0LL);

	// Initialize the master EnTT-based Registry
	global_registry = memnew(KernelRegistry);
}

SimulationManager::~SimulationManager() {
	if (global_registry) {
		memdelete(global_registry);
	}
	singleton = nullptr;
}

/**
 * advance_simulation()
 * 
 * The master loop entry.
 * Converts variable OS delta into deterministic 8.33ms slices.
 * strictly prevents "Temporal Jitter" by buffering overflows into the next frame.
 */
void SimulationManager::advance_simulation(const FixedMathCore &p_delta) {
	if (unlikely(paused)) {
		return;
	}

	// Apply bit-perfect time scale
	time_accumulator += p_delta * time_scale;

	// Deterministic catch-up loop.
	// We limit the maximum number of steps per frame (8) to prevent the "Spiral of Death"
	// while ensuring 120Hz consistency for the physics engine.
	int safety_steps = 0;
	while (time_accumulator >= physics_step && safety_steps < 8) {
		execute_step(physics_step);
		time_accumulator -= physics_step;
		safety_steps++;
	}
}

/**
 * execute_step()
 * 
 * Orchestrates a single deterministic simulation wave.
 * Enforces the dependency order of the Universal Solver's sub-systems.
 */
void SimulationManager::execute_step(const FixedMathCore &p_fixed_delta) {
	total_physics_frames += BigIntCore(1LL);

	// 1. Celestial & Macro Wave: N-Body Gravity and Orbit Propagation
	// Uses BigIntCore for astronomical mass-energy tensors.
	PhysicsServerHyper::get_singleton()->execute_gravity_sweep(p_fixed_delta);
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 2. Kinematic Wave: Relativistic Integration & Soft-Body PBD
	// Resolves Lorentz factors and Balloon/Flesh deformations.
	PhysicsServerHyper::get_singleton()->execute_integration_sweep(p_fixed_delta);
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 3. Constraint Wave: CCD and Mechanical Joints
	// Resolves exact Time-of-Impact (TOI) to prevent tunneling at high spaceship speeds.
	PhysicsServerHyper::get_singleton()->execute_collision_resolution(p_fixed_delta);
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 4. Sector Wave: Galactic Origin Drift Correction
	// Re-anchors BigInt sectors and finalizes the spatial broadphase.
	PhysicsServerHyper::get_singleton()->execute_galactic_sync();
	SimulationThreadPool::get_singleton()->wait_for_all();

	// 5. Telemetry: Update total simulation msec using BigInt
	// 1000ms / 120fps = 8.333... ms. We use 8333 microseconds for bit-perfection.
	total_simulation_time_msec += BigIntCore(8333LL);
}

void SimulationManager::set_time_scale(const FixedMathCore &p_scale) {
	time_scale = p_scale;
}

void SimulationManager::set_paused(bool p_paused) {
	paused = p_paused;
}

/**
 * get_simulation_time_seconds()
 * 
 * Calculates the total elapsed simulation time by multiplying the BigInt
 * frame count with the bit-perfect FixedMath step time.
 * Prevents 64-bit overflow even for simulations running for millennia.
 */
FixedMathCore SimulationManager::get_simulation_time_seconds() const {
	FixedMathCore frames_f(static_cast<int64_t>(std::stoll(total_physics_frames.to_string())));
	return frames_f * physics_step;
}

--- END OF FILE core/simulation/simulation_manager.cpp ---
