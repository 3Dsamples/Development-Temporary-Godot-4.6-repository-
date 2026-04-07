--- START OF FILE core/math/math_funcs.cpp ---

#include "core/math/math_funcs.h"
#include "core/os/os.h"

// ============================================================================
// Deterministic Random Number Generation (PCG32)
// ============================================================================

// Internal state for the global math generator.
// These are bit-aligned to prevent cache-line contention in multi-threaded sweeps.
static uint64_t pcg32_state = 0x853c49e6748fea9bULL;
static uint64_t pcg32_inc = 0xda3e39cb94b95bdbULL;

/**
 * Math::seed
 * 
 * Initializes the deterministic random state.
 * Mandatory for synchronized simulation start across network peers.
 */
void Math::seed(uint64_t p_seed) {
	pcg32_state = 0U;
	pcg32_inc = (p_seed << 1u) | 1u;
	Math::rand();
	pcg32_state += 0x853c49e6748fea9bULL;
	Math::rand();
}

/**
 * Math::rand
 * 
 * Returns a 32-bit unsigned random integer using the PCG algorithm.
 * High-performance path for procedural generation and entropy.
 */
uint32_t Math::rand() {
	uint64_t oldstate = pcg32_state;
	pcg32_state = oldstate * 6364136223846793005ULL + pcg32_inc;
	uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = (uint32_t)(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * Math::randf
 * 
 * Returns a deterministic FixedMathCore in the range [0.0, 1.0].
 * It maps the 32-bit random entropy directly to the fractional bits
 * of the Q32.32 format, bypassing all FPU operations.
 */
FixedMathCore Math::randf() {
	uint32_t r = Math::rand();
	// Treat the random bits as the 32-bit fraction, with 0 as the integer part.
	// This ensures a bit-perfect distribution between 0 and 0.999999999...
	return FixedMathCore(static_cast<int64_t>(r), true);
}

// ============================================================================
// Tiered Math Utility Implementation
// ============================================================================

/**
 * Math::fposmod
 * 
 * Floating-point style modulo that always returns a positive result.
 * Essential for wrapping galactic coordinates in toroidal space.
 */
FixedMathCore fposmod(FixedMathCore p_x, FixedMathCore p_y) {
	if (unlikely(p_y.get_raw() == 0)) return FixedMathCore(0LL, true);
	FixedMathCore value = p_x % p_y;
	if ((value.get_raw() < 0 && p_y.get_raw() > 0) || (value.get_raw() > 0 && p_y.get_raw() < 0)) {
		value += p_y;
	}
	return value;
}

/**
 * Math::step_decimals
 * 
 * Returns the number of decimal places in a fixed-point step.
 * Used for UI alignment and precision-aware snapping.
 */
int step_decimals(FixedMathCore p_step) {
	static const int64_t factors[10] = {
		1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
	};

	int64_t abs_v = ABS(p_step.get_raw());
	int64_t frac = abs_v & (FixedMathCore::ONE - 1);
	if (frac == 0) return 0;

	for (int i = 1; i < 10; i++) {
		if (((frac * factors[i]) >> FixedMathCore::FRACTIONAL_BITS) != 0) {
			return i;
		}
	}
	return 9;
}

--- END OF FILE core/math/math_funcs.cpp ---
