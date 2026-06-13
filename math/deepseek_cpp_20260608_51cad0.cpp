// File 324: modules/vienna/src/debug/vienna_profiler.h
// High‑performance non‑intrusive profiler for ViennaPhysicsEngine.
// Uses ring‑buffers and Godot's microsecond timestamp to record elapsed
// time of each physics stage (broad‑phase, narrow‑phase, solver, etc.)
// without mutex locks in the hot path. Supports averaging, per‑frame
// spikes, and exporting data to Godot's console.

#ifndef VIENNA_DEBUG_PROFILER_H
#define VIENNA_DEBUG_PROFILER_H

#include "core/os/os.h"
#include "core/print_string.h"
#include "core/typedefs.h"

namespace vienna {

class ViennaProfiler {
public:
	// Number of frames to keep in the ring buffer for averaging.
	static constexpr int RING_SIZE = 120; // 2 seconds at 60 fps

	// Stage identifiers
	enum Stage {
		STAGE_BROAD_PHASE = 0,
		STAGE_NARROW_PHASE,
		STAGE_ISLAND_BUILD,
		STAGE_SOLVER_CONTACTS,
		STAGE_SOLVER_JOINTS,
		STAGE_INTEGRATION,
		STAGE_CLOTH,
		STAGE_PARTICLES,
		STAGE_TOTAL,
		STAGE_COUNT
	};

private:
	// Ring buffer for each stage
	real_t ring[STAGE_COUNT][RING_SIZE];
	int write_index;
	int frame_count;            // number of frames recorded (up to RING_SIZE)
	uint64_t start_times[STAGE_COUNT]; // temporary during frame

public:
	ViennaProfiler() : write_index(0), frame_count(0) {
		for (int s = 0; s < STAGE_COUNT; ++s) {
			for (int i = 0; i < RING_SIZE; ++i) ring[s][i] = 0.0;
		}
	}

	// Begin a stage (call before the code block).
	inline void begin_stage(Stage p_stage) {
		start_times[p_stage] = OS::get_singleton()->get_ticks_usec();
	}

	// End a stage and record the elapsed time in milliseconds.
	inline void end_stage(Stage p_stage) {
		uint64_t end_time = OS::get_singleton()->get_ticks_usec();
		ring[p_stage][write_index] = (end_time - start_times[p_stage]) * 0.001; // ms
	}

	// Advance to the next frame (call once per physics step at the end).
	void next_frame() {
		write_index = (write_index + 1) % RING_SIZE;
		if (frame_count < RING_SIZE) frame_count++;
		// Reset ring for the new frame (optional, begin_stage overwrites).
	}

	// Get the average time for a stage over the recorded frames.
	real_t get_average(Stage p_stage) const {
		if (frame_count == 0) return 0.0;
		real_t sum = 0.0;
		for (int i = 0; i < frame_count; ++i) {
			sum += ring[p_stage][i];
		}
		return sum / frame_count;
	}

	// Get the maximum time for a stage in the last ring buffer.
	real_t get_max(Stage p_stage) const {
		if (frame_count == 0) return 0.0;
		real_t mx = ring[p_stage][0];
		for (int i = 1; i < frame_count; ++i) {
			if (ring[p_stage][i] > mx) mx = ring[p_stage][i];
		}
		return mx;
	}

	// Get the last frame's stage time.
	real_t get_last(Stage p_stage) const {
		if (frame_count == 0) return 0.0;
		int idx = (write_index - 1 + RING_SIZE) % RING_SIZE;
		return ring[p_stage][idx];
	}

	// Print a summary to Godot's output.
	void print_summary() const {
		print_line("--- Vienna Physics Profiler (last 120 frames) ---");
		print_line(vformat("Broad‑Phase:     avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_BROAD_PHASE), get_max(STAGE_BROAD_PHASE), get_last(STAGE_BROAD_PHASE)));
		print_line(vformat("Narrow‑Phase:    avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_NARROW_PHASE), get_max(STAGE_NARROW_PHASE), get_last(STAGE_NARROW_PHASE)));
		print_line(vformat("Island Build:    avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_ISLAND_BUILD), get_max(STAGE_ISLAND_BUILD), get_last(STAGE_ISLAND_BUILD)));
		print_line(vformat("Solver Contacts: avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_SOLVER_CONTACTS), get_max(STAGE_SOLVER_CONTACTS), get_last(STAGE_SOLVER_CONTACTS)));
		print_line(vformat("Solver Joints:   avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_SOLVER_JOINTS), get_max(STAGE_SOLVER_JOINTS), get_last(STAGE_SOLVER_JOINTS)));
		print_line(vformat("Integration:     avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_INTEGRATION), get_max(STAGE_INTEGRATION), get_last(STAGE_INTEGRATION)));
		print_line(vformat("Cloth:           avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_CLOTH), get_max(STAGE_CLOTH), get_last(STAGE_CLOTH)));
		print_line(vformat("Particles:       avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_PARTICLES), get_max(STAGE_PARTICLES), get_last(STAGE_PARTICLES)));
		print_line(vformat("TOTAL:           avg %5.2f ms   max %5.2f ms   last %5.2f ms",
			get_average(STAGE_TOTAL), get_max(STAGE_TOTAL), get_last(STAGE_TOTAL)));
		print_line("---------------------------------------------------");
	}
};

} // namespace vienna

#endif // VIENNA_DEBUG_PROFILER_H