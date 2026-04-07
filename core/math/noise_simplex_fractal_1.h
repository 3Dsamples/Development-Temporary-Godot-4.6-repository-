--- START OF FILE core/math/noise_simplex_fractal.h ---

#ifndef NOISE_SIMPLEX_FRACTAL_H
#define NOISE_SIMPLEX_FRACTAL_H

#include "core/math/noise_simplex.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimplexNoiseFractal
 * 
 * Advanced fractal noise aggregator.
 * Supports fBm, Rigid-Multifractal, and Domain Warping.
 * Aligned for Warp-style parallel batch processing at 120 FPS.
 */
template <typename T>
struct ET_ALIGN_32 SimplexNoiseFractal {
private:
	SimplexNoise<T> noise_kernel;
	
	int octaves;
	T persistence;
	T lacunarity;

	// Precomputed tensors to eliminate hot-loop calculation overhead
	T amplitudes[16];
	T frequencies[16];

public:
	// ------------------------------------------------------------------------
	// Configuration (Deterministic)
	// ------------------------------------------------------------------------

	void set_seed(const BigIntCore &p_seed) {
		noise_kernel.seed(p_seed);
	}

	/**
	 * set_parameters()
	 * Pre-calculates fractal weights to maintain 120 FPS throughput.
	 */
	void set_parameters(int p_octaves, T p_persistence, T p_lacunarity) {
		octaves = (p_octaves < 1) ? 1 : (p_octaves > 16 ? 16 : p_octaves);
		persistence = p_persistence;
		lacunarity = p_lacunarity;

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
	// Fractal Sampling API (Zero-Copy Warp Ready)
	// ------------------------------------------------------------------------

	/**
	 * sample_fbm()
	 * standard Fractional Brownian Motion. Sums octaves for smooth clouds/terrain.
	 */
	T sample_fbm(T x, T y, T z) const {
		T total = MathConstants<T>::zero();
		T max_amp = MathConstants<T>::zero();

		for (int i = 0; i < octaves; i++) {
			T f = frequencies[i];
			total += noise_kernel.sample_3d(x * f, y * f, z * f) * amplitudes[i];
			max_amp += amplitudes[i];
		}
		return total / max_amp;
	}

	/**
	 * sample_rigid()
	 * Rigid-Multifractal noise. Produces sharp peaks for mountains and cracks.
	 */
	T sample_rigid(T x, T y, T z) const {
		T total = MathConstants<T>::zero();
		T max_amp = MathConstants<T>::zero();
		T one = MathConstants<T>::one();

		for (int i = 0; i < octaves; i++) {
			T f = frequencies[i];
			T sample = noise_kernel.sample_3d(x * f, y * f, z * f);
			// 1.0 - abs(noise) creates the sharp "ridge"
			T v = one - Math::abs(sample);
			total += v * v * amplitudes[i];
			max_amp += amplitudes[i];
		}
		return total / max_amp;
	}

	/**
	 * sample_galactic_fbm()
	 * Scale-Aware Fractal sampling. Uses BigIntCore sectors to prevent 
	 * precision loss in distant star clusters.
	 */
	T sample_galactic_fbm(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const {
		T total = MathConstants<T>::zero();
		T max_amp = MathConstants<T>::zero();

		for (int i = 0; i < octaves; i++) {
			T f = frequencies[i];
			// Internal kernel handles the BigInt sector-offset logic
			total += noise_kernel.sample_galactic(p_sx, p_sy, p_sz, p_lx * f, p_ly * f, p_lz * f) * amplitudes[i];
			max_amp += amplitudes[i];
		}
		return total / max_amp;
	}

	/**
	 * sample_warped()
	 * Sophisticated Behavior: Domain Warping.
	 * Displaces the sampling coordinates by a previous noise pass to 
	 * simulate fluid turbulence or organic tissue patterns.
	 */
	T sample_warped(T x, T y, T z, T p_intensity) const {
		T dx = sample_fbm(x + T(13LL), y + T(17LL), z + T(19LL));
		T dy = sample_fbm(x + T(23LL), y + T(29LL), z + T(31LL));
		T dz = sample_fbm(x + T(37LL), y + T(41LL), z + T(43LL));
		
		return sample_fbm(x + dx * p_intensity, y + dy * p_intensity, z + dz * p_intensity);
	}

	SimplexNoiseFractal() {
		// Default: 4 octaves, 0.5 persistence, 2.0 lacunarity
		set_parameters(4, MathConstants<T>::half(), T(2LL));
	}
};

// Simulation Tier Typedefs
typedef SimplexNoiseFractal<FixedMathCore> SimplexNoiseFractalf;

#endif // NOISE_SIMPLEX_FRACTAL_H

--- END OF FILE core/math/noise_simplex_fractal.h ---
