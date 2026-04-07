--- START OF FILE core/simulation/simulation_manager.h ---

#ifndef SIMULATION_MANAGER_H
#define SIMULATION_MANAGER_H

#include "core/object/object.h"
#include "core/templates/vector.h"
#include "core/templates/list.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimulationManager
 * 
 * The master orchestrator for the Scale-Aware pipeline.
 * Coordinates 120 FPS heartbeats across TIER_DETERMINISTIC and TIER_MACRO.
 * Manages the transition of data from EnTT SoA pools to Warp execution kernels.
 */
class ET_ALIGN_32 SimulationManager : public Object {
	GDCLASS(SimulationManager, Object);

	static SimulationManager *singleton;

public:
	enum SimulationTier {
		TIER_DETERMINISTIC, // 120 FPS Fixed-Point Physics/Logic (FixedMathCore)
		TIER_MACRO_ECONOMY  // Infinite-scale Discrete Logic (BigIntCore)
	};

private:
	// Timing state using bit-perfect FixedMathCore
	FixedMathCore deterministic_accumulator;
	FixedMathCore fixed_step_time; // Default to 1/120 seconds
	FixedMathCore time_scale;

	// Global frame counters using BigIntCore to prevent overflow in galactic eras
	BigIntCore total_frames;
	BigIntCore physics_frames;

	bool paused = false;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ SimulationManager *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Frame & Heartbeat API
	// ------------------------------------------------------------------------

	/**
	 * advance_simulation()
	 * The 120 FPS entry point. Accumulates FixedMath delta and triggers
	 * bit-perfect simulation steps for all EnTT registries.
	 */
	void advance_simulation(const FixedMathCore &p_delta);

	/**
	 * step_deterministic()
	 * Triggers a single Warp-kernel sweep across the EnTT registry.
	 */
	void step_deterministic(const FixedMathCore &p_fixed_delta);

	// ------------------------------------------------------------------------
	// Configuration
	// ------------------------------------------------------------------------

	void set_fixed_fps(int p_fps);
	_FORCE_INLINE_ FixedMathCore get_fixed_step_time() const { return fixed_step_time; }

	void set_time_scale(const FixedMathCore &p_scale);
	_FORCE_INLINE_ FixedMathCore get_time_scale() const { return time_scale; }

	void set_paused(bool p_paused);
	_FORCE_INLINE_ bool is_paused() const { return paused; }

	// ------------------------------------------------------------------------
	// High-Precision Telemetry
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ BigIntCore get_total_frames() const { return total_frames; }
	_FORCE_INLINE_ BigIntCore get_physics_frames() const { return physics_frames; }

	/**
	 * get_simulation_time()
	 * Returns total active simulation time in FixedMathCore (seconds).
	 */
	FixedMathCore get_simulation_time() const;

	SimulationManager();
	~SimulationManager();
};

#endif // SIMULATION_MANAGER_H

--- END OF FILE core/simulation/simulation_manager.h ---
