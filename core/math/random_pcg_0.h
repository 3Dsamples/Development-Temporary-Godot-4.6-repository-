--- START OF FILE core/math/random_pcg.h ---

#ifndef RANDOM_PCG_H
#define RANDOM_PCG_H

#include "core/typedefs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * RandomPCG
 * 
 * Deterministic random number generator (PCG32).
 * Aligned to 32 bytes for SIMD-safe state management within Warp kernels.
 * Replaces all floating-point randomness with FixedMathCore [0, 1] ranges.
 */
struct ET_ALIGN_32 RandomPCG {
	uint64_t state = 0x853c49e6748fea9bULL;
	uint64_t inc = 0xda3e39cb94b95bdbULL;

	// ------------------------------------------------------------------------
	// Seeding API
	// ------------------------------------------------------------------------

	ET_SIMD_INLINE void seed(uint64_t p_seed) {
		state = 0U;
		inc = (p_seed << 1u) | 1u;
		rand();
		state += 0x853c49e6748fea9bULL;
		rand();
	}

	/**
	 * seed_big()
	 * Uses BigIntCore to initialize the state, allowing for galactic-scale 
	 * procedural seeds without precision loss.
	 */
	inline void seed_big(const BigIntCore &p_seed) {
		seed(p_seed.hash());
	}

	// ------------------------------------------------------------------------
	// Discrete Randomness
	// ------------------------------------------------------------------------

	ET_SIMD_INLINE uint32_t rand() {
		uint64_t oldstate = state;
		state = oldstate * 6364136223846793005ULL + inc;
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}

	ET_SIMD_INLINE int64_t rand_range(int64_t p_from, int64_t p_to) {
		if (p_from == p_to) return p_from;
		uint64_t r = ((uint64_t)rand() << 32) | rand();
		if (p_from < p_to) {
			return static_cast<int64_t>(r % static_cast<uint64_t>(p_to - p_from + 1)) + p_from;
		} else {
			return static_cast<int64_t>(r % static_cast<uint64_t>(p_from - p_to + 1)) + p_to;
		}
	}

	// ------------------------------------------------------------------------
	// Deterministic Simulation Randomness (FixedMathCore)
	// ------------------------------------------------------------------------

	/**
	 * randf()
	 * Returns a bit-perfect FixedMathCore in the range [0.0, 1.0].
	 * Maps the 32-bit entropy directly to the Q32.32 fractional part.
	 */
	ET_SIMD_INLINE FixedMathCore randf() {
		uint32_t r = rand();
		// Set integer part to 0, fractional part to raw 32-bit random bits.
		return FixedMathCore(static_cast<int64_t>(r), true);
	}

	ET_SIMD_INLINE FixedMathCore rand_range_fixed(FixedMathCore p_from, FixedMathCore p_to) {
		return p_from + (p_to - p_from) * randf();
	}

	// ------------------------------------------------------------------------
	// Macro-Scale Randomness (BigIntCore)
	// ------------------------------------------------------------------------

	/**
	 * rand_big()
	 * Generates a random BigIntCore within a massive range. 
	 * Essential for procedural star placement and economic fluctuations.
	 */
	BigIntCore rand_big(const BigIntCore &p_from, const BigIntCore &p_to);

	RandomPCG() {}
};

#endif // RANDOM_PCG_H

--- END OF FILE core/math/random_pcg.h ---
