--- START OF FILE core/math/math_funcs.cpp ---

#include "core/math/math_funcs.h"

/**
 * Global Deterministic Random State
 * 
 * ETEngine Strategy: Use PCG32 for superior statistical properties 
 * and absolute determinism across different CPU architectures.
 */
static uint64_t pcg32_state = 0x853c49e6748fea9bULL;
static uint64_t pcg32_inc = 0xda3e39cb94b95bdbULL;

/**
 * seed()
 * 
 * Initializes the random generator. For 120 FPS multiplayer sync, 
 * all clients must call this with the same BigIntCore-derived seed 
 * at simulation start.
 */
void Math::seed(uint64_t p_seed) {
	pcg32_state = 0U;
	pcg32_inc = (p_seed << 1u) | 1u;
	Math::rand();
	pcg32_state += 0x853c49e6748fea9bULL;
	Math::rand();
}

/**
 * rand()
 * 
 * Returns a 32-bit unsigned random integer.
 * Optimized for Warp-style kernel usage where high-entropy entropy is required 
 * for procedural geometry and fracturing.
 */
uint32_t Math::rand() {
	uint64_t oldstate = pcg32_state;
	// Advance internal state
	pcg32_state = oldstate * 6364136223846793005ULL + pcg32_inc;
	// Calculate output function (XSH RR)
	uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = (uint32_t)(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * randf()
 * 
 * Returns a deterministic FixedMathCore in the range [0.0, 1.0].
 * It maps the 32 random bits directly into the fractional part of the Q32.32 format.
 * This ensures there is NO floating-point conversion, maintaining bit-perfection.
 */
FixedMathCore Math::randf() {
	uint32_t r = Math::rand();
	// We use the raw bit constructor (friend access assumed from fixed_math_core.h)
	// This sets the integer part to 0 and the fractional part to the random bits.
	return FixedMathCore(static_cast<int64_t>(r), true);
}

// ============================================================================
// Advanced Behavioral Kernels
// ============================================================================

/**
 * fposmod()
 * 
 * Deterministic positive-only modulo. 
 * Critical for wrapping galactic coordinates in toroidal space loops.
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
 * cubic_interpolate_f()
 * 
 * Heavy implementation for smooth animation and deformation curves.
 * strictly uses FixedMathCore for 120 FPS consistency.
 */
FixedMathCore Math::cubic_interpolate(FixedMathCore p_pre, FixedMathCore p_from, FixedMathCore p_to, FixedMathCore p_post, FixedMathCore p_weight) {
	FixedMathCore p2 = p_weight * p_weight;
	FixedMathCore p3 = p2 * p_weight;
	FixedMathCore half = MathConstants<FixedMathCore>::half();

	// Result = f + 0.5 * w * ( (t - p) + w * ( (2p - 5f + 4t - o) + w * ( 3(f - t) + o - p ) ) )
	// f: from, t: to, p: pre, o: post
	FixedMathCore term1 = p_to - p_pre;
	FixedMathCore term2 = (FixedMathCore(2LL) * p_pre) - (FixedMathCore(5LL) * p_from) + (FixedMathCore(4LL) * p_to) - p_post;
	FixedMathCore term3 = (FixedMathCore(3LL) * (p_from - p_to)) + p_post - p_pre;

	return p_from + half * p_weight * (term1 + p_weight * (term2 + p_weight * term3));
}

/**
 * step_decimals()
 * 
 * Utility to determine precision requirements for UI display and data-delta encoding.
 */
int step_decimals(FixedMathCore p_step) {
	static const int64_t factors[10] = {
		1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
	};

	int64_t abs_v = (p_step.get_raw() < 0) ? -p_step.get_raw() : p_step.get_raw();
	int64_t frac = abs_v & 0xFFFFFFFFLL;
	if (frac == 0) return 0;

	for (int i = 1; i < 10; i++) {
		if (((frac * factors[i]) >> 32) != 0) {
			return i;
		}
	}
	return 9;
}

--- END OF FILE core/math/math_funcs.cpp ---
