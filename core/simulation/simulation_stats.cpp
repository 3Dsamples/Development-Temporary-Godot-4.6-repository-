--- START OF FILE core/simulation/simulation_stats.cpp ---

#include "core/simulation/simulation_stats.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/string/ustring.h"

SimulationStats *SimulationStats::singleton = nullptr;

void SimulationStats::_bind_methods() {
	ClassDB::bind_method(D_METHOD("begin_frame"), &SimulationStats::begin_frame);
	ClassDB::bind_method(D_METHOD("end_frame"), &SimulationStats::end_frame);
	ClassDB::bind_method(D_METHOD("get_metric", "metric"), &SimulationStats::get_metric);
	ClassDB::bind_method(D_METHOD("get_average", "metric"), &SimulationStats::get_average);
	ClassDB::bind_method(D_METHOD("get_performance_report"), &SimulationStats::get_performance_report);

	BIND_ENUM_CONSTANT(METRIC_FRAME_TIME);
	BIND_ENUM_CONSTANT(METRIC_PHYSICS_STEP_TIME);
	BIND_ENUM_CONSTANT(METRIC_TASK_LATENCY);
	BIND_ENUM_CONSTANT(METRIC_MEMORY_USAGE);
	BIND_ENUM_CONSTANT(METRIC_ENTITY_COUNT);
}

SimulationStats::SimulationStats() {
	singleton = this;
	last_frame_ticks = 0;
	total_simulation_ticks = BigIntCore(0LL);
	total_warp_kernels_launched = BigIntCore(0LL);

	for (int i = 0; i < METRIC_MAX; i++) {
		current_metrics[i] = FixedMathCore(0LL, true);
		rolling_averages[i] = FixedMathCore(0LL, true);
	}
}

SimulationStats::~SimulationStats() {
	singleton = nullptr;
}

void SimulationStats::begin_frame() {
	last_frame_ticks = OS::get_singleton()->get_ticks_usec();
}

void SimulationStats::end_frame() {
	uint64_t current_ticks = OS::get_singleton()->get_ticks_usec();
	uint64_t delta = current_ticks - last_frame_ticks;

	// Convert microseconds to FixedMathCore seconds (delta / 1,000,000)
	FixedMathCore frame_time = FixedMathCore(static_cast<int64_t>(delta)) / FixedMathCore(1000000LL, false);
	record_metric(METRIC_FRAME_TIME, frame_time);

	total_simulation_ticks += BigIntCore(1LL);
}

FixedMathCore SimulationStats::get_metric(StatMetric p_metric) const {
	ERR_FAIL_INDEX_V(p_metric, METRIC_MAX, FixedMathCore(0LL, true));
	return current_metrics[p_metric];
}

FixedMathCore SimulationStats::get_average(StatMetric p_metric) const {
	ERR_FAIL_INDEX_V(p_metric, METRIC_MAX, FixedMathCore(0LL, true));
	return rolling_averages[p_metric];
}

BigIntCore SimulationStats::get_total_ticks() const {
	return total_simulation_ticks;
}

String SimulationStats::get_performance_report() const {
	String report = "--- UNIVERSAL SOLVER PERFORMANCE ---\n";
	report += "Frame Time (AVG): " + String(get_average(METRIC_FRAME_TIME).to_string().c_str()) + "s\n";
	report += "Physics Step (AVG): " + String(get_average(METRIC_PHYSICS_STEP_TIME).to_string().c_str()) + "s\n";
	report += "Active Entities: " + String(get_metric(METRIC_ENTITY_COUNT).to_int()) + "\n";
	report += "Total Simulation Ticks: " + String(total_simulation_ticks.to_string().c_str()) + "\n";
	report += "Memory Pressure: " + String(get_metric(METRIC_MEMORY_USAGE).to_int()) + " bytes\n";
	report += "------------------------------------";
	return report;
}

--- END OF FILE core/simulation/simulation_stats.cpp ---
