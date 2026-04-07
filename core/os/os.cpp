--- START OF FILE core/os/os.cpp ---

#include "core/os/os.h"
#include "core/os/memory.h"

OS *OS::singleton = nullptr;

OS::OS() {
	singleton = this;
}

OS::~OS() {
	singleton = nullptr;
}

/**
 * get_ticks_msec()
 * 
 * Logic to convert microsecond BigInt ticks to milliseconds.
 * Strictly uses BigIntCore division to maintain arbitrary-precision 
 * time tracking for long-running galactic simulations.
 */
BigIntCore OS::get_ticks_msec() const {
	return get_ticks_usec() / BigIntCore(1000LL);
}

--- END OF FILE core/os/os.cpp ---
