--- START OF FILE core/math/noise_simplex.h ---

#ifndef NOISE_SIMPLEX_H
#define NOISE_SIMPLEX_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimplexNoise Template
 * 
 * High-performance deterministic noise kernel for the Universal Solver.
 * Replaces standard floating-point skewing and gradients with FixedMathCore.
 * Optimized for 120 FPS parallel sampling in EnTT-based procedural systems.
 */
template <typename T>
struct ET_ALIGN_32 SimplexNoise {
private:
	uint8_t perm[512];
	static const int8_t grad3[12][3];

	// Skewing and unskewing factors for 3D (Q32.32 Fixed-Point constants)
	// F3 = 1/3, G3 = 1/6
	static inline T F3() { return FixedMathCore(1431655765LL, true); }
	static inline T G3() { return FixedMathCore(715827882LL, true); }

	ET_SIMD_INLINE int _fast_floor(T p_val) const {
		int i = p_val.to_int();
		return (p_val.get_raw() < 0 && (p_val.get_raw() & 0xFFFFFFFFLL) != 0) ? i - 1 : i;
	}

	ET_SIMD_INLINE T _dot(const int8_t *g, T x, T y, T z) const {
		return T(g[0]) * x + T(g[1]) * y + T(g[2]) * z;
	}

public:
	/**
	 * seed()
	 * Initializes the permutation table using a deterministic BigIntCore hash.
	 * Guarantees identical noise patterns across all Warp kernel instances.
	 */
	void seed(const BigIntCore &p_seed);

	/**
	 * sample_3d()
	 * The core Simplex Noise kernel. Uses bit-perfect Q32.32 math for skewing.
	 * Hardware-agnostic: designed for zero-copy execution on CPU SIMD or GPU.
	 */
	T sample_3d(T x, T y, T z) const;

	/**
	 * sample_galactic()
	 * Scale-aware sampling. Uses BigIntCore for sector coordinates and 
	 * FixedMathCore for sub-sector detail, preventing precision loss at infinite scale.
	 */
	T sample_galactic(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const;

	SimplexNoise() {
		seed(BigIntCore(12345LL));
	}
};

typedef SimplexNoise<FixedMathCore> SimplexNoisef;

#endif // NOISE_SIMPLEX_H

--- END OF FILE core/math/noise_simplex.h ---
