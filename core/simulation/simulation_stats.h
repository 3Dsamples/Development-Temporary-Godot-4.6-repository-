--- START OF FILE core/simulation/simulation_stats.h ---

#ifndef SIMULATION_STATS_H
#define SIMULATION_STATS_H

#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimulationStats
 * 
 * High-performance telemetry singleton.
 * Tracks frame timings, task throughput, and memory pressure.
 * Strictly uses FixedMathCore for averages and BigIntCore for lifetime counters.
 */
class SimulationStats : public Object {
	GDCLASS(SimulationStats, Object);

	static SimulationStats *singleton;

public:
	enum StatMetric {
		METRIC_FRAME_TIME,
		METRIC_PHYSICS_STEP_TIME,
		METRIC_TASK_LATENCY,
		METRIC_MEMORY_USAGE,
		METRIC_ENTITY_COUNT,
		METRIC_MAX
	};

private:
	// Performance buffers using bit-perfect FixedMath
	FixedMathCore current_metrics[METRIC_MAX];
	FixedMathCore rolling_averages[METRIC_MAX];
	
	// Cumulative lifetime stats using BigIntCore to prevent galactic-scale overflow
	BigIntCore total_simulation_ticks;
	BigIntCore total_warp_kernels_launched;
	
	uint64_t last_frame_ticks;

protected:
	static void _bind_methods();

public:
	static _FORCE_INLINE_ SimulationStats *get_singleton() { return singleton; }

	// ------------------------------------------------------------------------
	// Telemetry API
	// ------------------------------------------------------------------------

	/**
	 * begin_frame()
	 * Captures high-resolution timestamp for the 120 FPS heartbeat start.
	 */
	void begin_frame();

	/**
	 * end_frame()
	 * Finalizes metrics for the current frame and updates rolling averages.
	 */
	void end_frame();

	/**
	 * record_metric()
	 * Sets a specific simulation metric value using deterministic FixedMath.
	 */
	_FORCE_INLINE_ void record_metric(StatMetric p_metric, const FixedMathCore &p_value) {
		current_metrics[p_metric] = p_value;
		// Exponential Moving Average: Alpha = 0.1
		FixedMathCore alpha(429496730LL, true); 
		rolling_averages[p_metric] = (p_value * alpha) + (rolling_averages[p_metric] * (MathConstants<FixedMathCore>::one() - alpha));
	}

	/**
	 * increment_counter()
	 * Adds to lifetime BigInt counters (e.g., total entities processed).
	 */
	_FORCE_INLINE_ void increment_counter(const BigIntCore &p_val) {
		total_simulation_ticks += p_val;
	}

	// ------------------------------------------------------------------------
	// Data Accessors
	// ------------------------------------------------------------------------

	FixedMathCore get_metric(StatMetric p_metric) const;
	FixedMathCore get_average(StatMetric p_metric) const;
	BigIntCore get_total_ticks() const;

	String get_performance_report() const;

	SimulationStats();
	~SimulationStats();
};

#endif // SIMULATION_STATS_H

--- END OF FILE core/simulation/simulation_stats.h ---
