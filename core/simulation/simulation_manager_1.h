--- START OF FILE core/simulation/simulation_manager.h ---

#ifndef SIMULATION_MANAGER_H
#define SIMULATION_MANAGER_H

#include "core/object/object.h"
#include "core/math/math_defs.h"
#include "core/math/kernel_registry.h"
#include "core/templates/rid_owner.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimulationManager
 * 
 * The master heartbeat controller for the Universal Solver.
 * Coordinates 120 FPS execution waves across tiered simulation scales.
 * ensures bit-perfect synchronization and zero temporal drift.
 */
class SimulationManager : public Object {
	GDCLASS(SimulationManager, Object);

	static SimulationManager *singleton;

	// Temporal State (Strictly FixedMathCore/BigIntCore)
	FixedMathCore time_scale;
	FixedMathCore physics_step;        // Default: 1/120s
	FixedMathCore time_accumulator;
	
	BigIntCore total_physics_frames;   // Galactic-scale clock (never overflows)
	BigIntCore total_simulation_time_msec;

	// Registry & Execution context
	KernelRegistry *global_registry = nullptr;

	bool paused = false;
	bool interpolation_enabled = true;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ SimulationManager *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Simulation Heartbeat API
	// ------------------------------------------------------------------------

	/**
	 * advance_simulation()
	 * The master entry point called every engine frame.
	 * Converts variable frame time into deterministic 120Hz sub-steps.
	 */
	void advance_simulation(const FixedMathCore &p_delta);

	/**
	 * execute_step()
	 * Triggers a single simulation wave across all EnTT SoA streams.
	 * Order: Celestial -> Kinematic -> Constraint -> Drift Correction.
	 */
	void execute_step(const FixedMathCore &p_fixed_delta);

	// ------------------------------------------------------------------------
	// Configuration & State
	// ------------------------------------------------------------------------

	void set_time_scale(const FixedMathCore &p_scale);
	_FORCE_INLINE_ FixedMathCore get_time_scale() const { return time_scale; }

	void set_paused(bool p_paused);
	_FORCE_INLINE_ bool is_paused() const { return paused; }

	_FORCE_INLINE_ BigIntCore get_total_frames() const { return total_physics_frames; }
	
	/**
	 * get_simulation_time_seconds()
	 * Returns total active simulation time in bit-perfect FixedMath.
	 */
	FixedMathCore get_simulation_time_seconds() const;

	// ------------------------------------------------------------------------
	// Registry Access (Zero-Copy Entry)
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ KernelRegistry* get_registry() { return global_registry; }

	SimulationManager();
	~SimulationManager();
};

#endif // SIMULATION_MANAGER_H

--- END OF FILE core/simulation/simulation_manager.h ---
