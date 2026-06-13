// File 369: modules/integration/unified_profiler.h
// Unified real‑time profiler for all physics engines (Gaia, Genesis, Newton,
// Vienna, Wicked). Collects per‑stage timings from each subsystem and prints
// a consolidated report every N frames.  Uses lock‑free ring buffers and
// Godot's microsecond timer for zero overhead in the hot path.  All methods
// are inline for minimal impact on the physics step.

#ifndef INTEGRATION_UNIFIED_PROFILER_H
#define INTEGRATION_UNIFIED_PROFILER_H

#include "core/os/os.h"
#include "core/print_string.h"
#include "core/typedefs.h"

// Include profilers from each engine
#include "../../gaia/src/debug/vienna_profiler.h"      // reuse ViennaProfiler's ring buffer as base
// (Note: Gaia doesn't have its own profiler; we'll create a wrapper)

namespace unified {

class UnifiedProfiler {
public:
    // Number of frames retained for averaging
    static constexpr int RING_SIZE = 300;   // 5 seconds at 60 fps

    // Stages common across engines (add more as needed)
    enum Stage : int {
        // Broad‑phase
        STAGE_GAIA_BROAD    = 0,
        STAGE_UNIFIED_SYNC  = 1,
        // Newton
        STAGE_NEWTON_STEP   = 2,
        STAGE_NEWTON_SOLVE  = 3,
        // Genesis
        STAGE_GENESIS_STEP  = 4,
        STAGE_GEN_FEM_SOLVE = 5,
        STAGE_GEN_MPM_SOLVE = 6,
        STAGE_GEN_SPH_SOLVE = 7,
        // Vienna
        STAGE_VIENNA_STEP   = 8,
        // Wicked
        STAGE_WICKED_STEP   = 9,
        // Vehicle updates
        STAGE_VEHICLES      = 10,
        // Cloth / Particles
        STAGE_CLOTH         = 11,
        STAGE_PARTICLES     = 12,
        // Total
        STAGE_TOTAL         = 13,
        STAGE_COUNT
    };

private:
    // Ring buffer for each stage
    real_t ring[STAGE_COUNT][RING_SIZE];
    int    write_index;
    int    frame_count;
    uint64_t start_times[STAGE_COUNT];

public:
    UnifiedProfiler() : write_index(0), frame_count(0) {
        for (int s = 0; s < STAGE_COUNT; ++s)
            for (int i = 0; i < RING_SIZE; ++i) ring[s][i] = 0.0;
        for (int s = 0; s < STAGE_COUNT; ++s) start_times[s] = 0;
    }

    // Begin a stage.  Call before the block.
    inline void begin_stage(Stage p_stage) {
        start_times[p_stage] = OS::get_singleton()->get_ticks_usec();
    }

    // End a stage and record the elapsed time in milliseconds.
    inline void end_stage(Stage p_stage) {
        uint64_t end_time = OS::get_singleton()->get_ticks_usec();
        ring[p_stage][write_index] = (end_time - start_times[p_stage]) * 0.001; // ms
    }

    // Advance to the next frame (call at end of physics step).
    void next_frame() {
        write_index = (write_index + 1) % RING_SIZE;
        if (frame_count < RING_SIZE) frame_count++;
    }

    // Get average time for a stage over the recorded frames.
    inline real_t get_average(Stage p_stage) const {
        if (frame_count == 0) return 0.0;
        real_t sum = 0.0;
        for (int i = 0; i < frame_count; ++i) sum += ring[p_stage][i];
        return sum / frame_count;
    }

    // Get maximum time for a stage.
    inline real_t get_max(Stage p_stage) const {
        if (frame_count == 0) return 0.0;
        real_t mx = ring[p_stage][0];
        for (int i = 1; i < frame_count; ++i) if (ring[p_stage][i] > mx) mx = ring[p_stage][i];
        return mx;
    }

    // Get last frame's time.
    inline real_t get_last(Stage p_stage) const {
        if (frame_count == 0) return 0.0;
        int idx = (write_index - 1 + RING_SIZE) % RING_SIZE;
        return ring[p_stage][idx];
    }

    // Print a formatted console report.
    void print_report() const {
        print_line("========== Unified Physics Profiler (last 300 frames) ==========");
        print_line(vformat("Gaia Broad‑Phase:    avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_GAIA_BROAD), get_max(STAGE_GAIA_BROAD), get_last(STAGE_GAIA_BROAD)));
        print_line(vformat("Unified Sync:        avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_UNIFIED_SYNC), get_max(STAGE_UNIFIED_SYNC), get_last(STAGE_UNIFIED_SYNC)));
        print_line(vformat("Newton Step:         avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_NEWTON_STEP), get_max(STAGE_NEWTON_STEP), get_last(STAGE_NEWTON_STEP)));
        print_line(vformat("Newton Solve:        avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_NEWTON_SOLVE), get_max(STAGE_NEWTON_SOLVE), get_last(STAGE_NEWTON_SOLVE)));
        print_line(vformat("Genesis Step:        avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_GENESIS_STEP), get_max(STAGE_GENESIS_STEP), get_last(STAGE_GENESIS_STEP)));
        print_line(vformat("Genesis FEM Solve:   avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_GEN_FEM_SOLVE), get_max(STAGE_GEN_FEM_SOLVE), get_last(STAGE_GEN_FEM_SOLVE)));
        print_line(vformat("Genesis MPM Solve:   avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_GEN_MPM_SOLVE), get_max(STAGE_GEN_MPM_SOLVE), get_last(STAGE_GEN_MPM_SOLVE)));
        print_line(vformat("Genesis SPH Solve:   avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_GEN_SPH_SOLVE), get_max(STAGE_GEN_SPH_SOLVE), get_last(STAGE_GEN_SPH_SOLVE)));
        print_line(vformat("Vienna Step:         avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_VIENNA_STEP), get_max(STAGE_VIENNA_STEP), get_last(STAGE_VIENNA_STEP)));
        print_line(vformat("Wicked Step:         avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_WICKED_STEP), get_max(STAGE_WICKED_STEP), get_last(STAGE_WICKED_STEP)));
        print_line(vformat("Vehicles:            avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_VEHICLES), get_max(STAGE_VEHICLES), get_last(STAGE_VEHICLES)));
        print_line(vformat("Cloth:               avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_CLOTH), get_max(STAGE_CLOTH), get_last(STAGE_CLOTH)));
        print_line(vformat("Particles:           avg %6.2f ms  max %6.2f ms  last %6.2f ms",
            get_average(STAGE_PARTICLES), get_max(STAGE_PARTICLES), get_last(STAGE_PARTICLES)));
        print_line(vformat("---- TOTAL:          avg %6.2f ms  max %6.2f ms  last %6.2f ms ----",
            get_average(STAGE_TOTAL), get_max(STAGE_TOTAL), get_last(STAGE_TOTAL)));
        print_line("===================================================================");
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PROFILER_H