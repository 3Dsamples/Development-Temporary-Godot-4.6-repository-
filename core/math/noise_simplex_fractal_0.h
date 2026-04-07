--- START OF FILE core/math/noise_simplex_fractal.h ---

#ifndef NOISE_SIMPLEX_FRACTAL_H
#define NOISE_SIMPLEX_FRACTAL_H

#include "core/math/noise_simplex.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimplexNoiseFractal
 * 
 * High-performance Fractal Brownian Motion (fBm) implementation.
 * Layers multiple octaves of Simplex Noise to create natural complexity.
 * Strictly uses FixedMathCore to ensure identical procedural results 
 * across all simulation scales (Micro to Galactic).
 */
template <typename T>
struct ET_ALIGN_32 SimplexNoiseFractal {
private:
	SimplexNoise<T> noise_kernel;
	
	int octaves = 4;
	T persistence; // Decay factor for amplitude
	T lacunarity;  // Increase factor for frequency

	// Precomputed buffers for Warp-kernel performance
	T amplitudes[16];
	T frequencies[16];

public:
	// ------------------------------------------------------------------------
	// Configuration
	// ------------------------------------------------------------------------

	void set_seed(const BigIntCore &p_seed) {
		noise_kernel.seed(p_seed);
	}

	void set_parameters(int p_octaves, T p_persistence, T p_lacunarity) {
		octaves = CLAMP(p_octaves, 1, 16);
		persistence = p_persistence;
		lacunarity = p_lacunarity;

		// Precompute weights to eliminate hot-loop divisions/powers
		T cur_amp = MathConstants<T>::one();
		T cur_freq = MathConstants<T>::one();
		for (int i = 0; i < 16; i++) {
			amplitudes[i] = cur_amp;
			frequencies[i] = cur_freq;
			cur_amp *= persistence;
			cur_freq *= lacunarity;
		}
	}

	// ------------------------------------------------------------------------
	// Sampling API (Zero-Copy Warp Ready)
	// ------------------------------------------------------------------------

	/**
	 * sample_3d()
	 * Sums multiple octaves of bit-perfect noise. 
	 * Optimized for high-throughput batching in EnTT registries.
	 */
	ET_SIMD_INLINE T sample_3d(T x, T y, T z) const {
		T total = MathConstants<T>::zero();
		T max_amp = MathConstants<T>::zero();

		for (int i = 0; i < octaves; i++) {
			T f = frequencies[i];
			total += noise_kernel.sample_3d(x * f, y * f, z * f) * amplitudes[i];
			max_amp += amplitudes[i];
		}

		// Normalize to [-1, 1] range
		return total / max_amp;
	}

	/**
	 * sample_galactic()
	 * Samples noise across infinite coordinates using BigIntCore anchoring.
	 */
	ET_SIMD_INLINE T sample_galactic(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const {
		T total = MathConstants<T>::zero();
		T max_amp = MathConstants<T>::zero();

		for (int i = 0; i < octaves; i++) {
			T f = frequencies[i];
			// Use sector-aware kernel sampling to prevent precision jitter
			total += noise_kernel.sample_galactic(p_sx, p_sy, p_sz, p_lx * f, p_ly * f, p_lz * f) * amplitudes[i];
			max_amp += amplitudes[i];
		}

		return total / max_amp;
	}

	SimplexNoiseFractal() {
		set_parameters(4, FixedMathCore(2147483648LL, true), FixedMathCore(2LL, false)); // Default: Persistence 0.5, Lacunarity 2.0
	}
};

typedef SimplexNoiseFractal<FixedMathCore> SimplexNoiseFractalf;

#endif // NOISE_SIMPLEX_FRACTAL_H

--- END OF FILE core/math/noise_simplex_fractal.h ---
