--- START OF FILE core/math/random_pcg.h ---

#ifndef RANDOM_PCG_H
#define RANDOM_PCG_H

#include "core/typedefs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * RandomPCG
 * 
 * High-performance deterministic random number generator.
 * Strictly uses PCG32/XSH-RR to ensure zero-drift stochasticity.
 * Aligned for Warp kernel execution and EnTT component streams.
 */
struct ET_ALIGN_32 RandomPCG {
private:
	uint64_t state;
	uint64_t inc;

public:
	// ------------------------------------------------------------------------
	// Seeding API (BigInt Compatible)
	// ------------------------------------------------------------------------

	/**
	 * seed()
	 * Standard 64-bit initialization.
	 */
	_FORCE_INLINE_ void seed(uint64_t p_seed, uint64_t p_inc = 1442695040888963407ULL) {
		state = 0U;
		inc = (p_inc << 1u) | 1u;
		rand();
		state += p_seed;
		rand();
	}

	/**
	 * seed_big()
	 * Initializes entropy using a BigIntCore handle, allowing for seeds 
	 * derived from galactic coordinates or trillion-unit economy ledgers.
	 */
	void seed_big(const BigIntCore &p_seed);

	// ------------------------------------------------------------------------
	// Discrete Randomness API
	// ------------------------------------------------------------------------

	/**
	 * rand()
	 * Returns a deterministic 32-bit unsigned integer.
	 */
	_FORCE_INLINE_ uint32_t rand() {
		uint64_t oldstate = state;
		state = oldstate * 6364136223846793005ULL + inc;
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}

	_FORCE_INLINE_ uint64_t rand64() {
		return (static_cast<uint64_t>(rand()) << 32) | rand();
	}

	int64_t rand_range(int64_t p_from, int64_t p_to);

	// ------------------------------------------------------------------------
	// Deterministic Simulation API (FixedMathCore)
	// ------------------------------------------------------------------------

	/**
	 * randf()
	 * Returns a bit-perfect FixedMathCore in the range [0.0, 1.0].
	 * Replaces float-based randf() to eliminate rounding variance.
	 */
	_FORCE_INLINE_ FixedMathCore randf() {
		uint32_t r = rand();
		// Set integer part to 0, fractional part to the random bits
		return FixedMathCore(static_cast<int64_t>(r), true);
	}

	_FORCE_INLINE_ FixedMathCore rand_range_fixed(FixedMathCore p_from, FixedMathCore p_to) {
		if (p_from == p_to) return p_from;
		return p_from + (p_to - p_from) * randf();
	}

	// ------------------------------------------------------------------------
	// Macro-Scale Randomness (BigIntCore)
	// ------------------------------------------------------------------------

	/**
	 * rand_big()
	 * Generates an arbitrary-precision random integer in the range [p_from, p_to].
	 * Essential for procedural star placement and galactic-scale state resets.
	 */
	BigIntCore rand_big(const BigIntCore &p_from, const BigIntCore &p_to);

	// ------------------------------------------------------------------------
	// Advanced Sophisticated Behaviors
	// ------------------------------------------------------------------------

	/**
	 * rand_unit_vector()
	 * Deterministic direction generation for shattering and light scattering.
	 */
	Vector3f rand_unit_vector();

	/**
	 * rand_gaussian()
	 * Box-Muller transform for bit-perfect normal distributions in FixedMath.
	 */
	FixedMathCore rand_gaussian(FixedMathCore p_mean, FixedMathCore p_stddev);

	RandomPCG() : state(0x853c49e6748fea9bULL), inc(0xda3e39cb94b95bdbULL) {}
};

#endif // RANDOM_PCG_H

--- END OF FILE core/math/random_pcg.h ---
