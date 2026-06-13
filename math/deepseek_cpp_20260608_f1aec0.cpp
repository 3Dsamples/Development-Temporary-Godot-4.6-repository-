// File 42: modules/gaia/src/utility/timer.h

#ifndef GAIA_UTILITY_TIMER_H
#define GAIA_UTILITY_TIMER_H

#include "core/os/os.h"
#include "core/typedefs.h"

namespace gaia::utility {

/**
 * High-resolution CPU timer.
 *
 * Replaces Gaia's original timer with a wrapper around Godot's OS
 * microsecond tick counter.
 */
class Timer {
public:
	Timer() : start_time(0), elapsed(0.0), running(false) {}

	// Start or restart the timer.
	void start() {
		start_time = OS::get_singleton()->get_ticks_usec();
		elapsed = 0.0;
		running = true;
	}

	// Stop the timer and accumulate elapsed time.
	void stop() {
		if (!running) return;
		uint64_t now = OS::get_singleton()->get_ticks_usec();
		elapsed += (now - start_time) * 1e-6;
		running = false;
	}

	// Resume a stopped timer without resetting accumulated time.
	void resume() {
		if (running) return;
		start_time = OS::get_singleton()->get_ticks_usec();
		running = true;
	}

	// Return elapsed time in seconds (accumulated + current run).
	real_t get_elapsed() const {
		if (!running) return elapsed;
		uint64_t now = OS::get_singleton()->get_ticks_usec();
		return elapsed + (now - start_time) * 1e-6;
	}

	// Reset accumulated time (and stop if running).
	void reset() {
		elapsed = 0.0;
		if (running) {
			start_time = OS::get_singleton()->get_ticks_usec();
		}
	}

private:
	uint64_t start_time;
	real_t elapsed;
	bool running;
};

} // namespace gaia::utility

#endif // GAIA_UTILITY_TIMER_H