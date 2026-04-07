--- START OF FILE core/math/noise_simplex.h ---

#ifndef NOISE_SIMPLEX_H
#define NOISE_SIMPLEX_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * SimplexNoise Template
 * 
 * High-performance 3D noise kernel.
 * Aligned for Warp kernel execution and EnTT SoA streams.
 * Strictly avoids FPU usage to ensure bit-perfect procedural generation.
 */
template <typename T>
struct ET_ALIGN_32 SimplexNoise {
private:
	uint8_t perm[512];
	
	// Deterministic Gradients for 3D (12 directions to midpoints of cube edges)
	static const int8_t grad3[12][3];

	// Skewing factors for 3D (Q32.32 Constants)
	// F3 = 1/3, G3 = 1/6
	static _FORCE_INLINE_ T F3() { return T(1431655765LL, true); }
	static _FORCE_INLINE_ T G3() { return T(715827882LL, true); }

	_FORCE_INLINE_ int _fast_floor(T p_val) const {
		int64_t raw = p_val.get_raw();
		int64_t i = raw >> 32;
		return (raw < 0 && (raw & 0xFFFFFFFFLL) != 0) ? (int)i - 1 : (int)i;
	}

	_FORCE_INLINE_ T _dot(const int8_t *g, T x, T y, T z) const {
		return T(static_cast<int64_t>(g[0])) * x + T(static_cast<int64_t>(g[1])) * y + T(static_cast<int64_t>(g[2])) * z;
	}

public:
	/**
	 * seed()
	 * 
	 * Initializes the permutation table using a deterministic BigIntCore hash.
	 * Guarantees that procedural patterns are identical on all clients.
	 */
	void seed(const BigIntCore &p_seed);

	/**
	 * sample_3d()
	 * 
	 * The core noise logic. Performs skewing, simplex identification, 
	 * and contribution summation using bit-perfect FixedMath.
	 */
	T sample_3d(T x, T y, T z) const;

	/**
	 * sample_galactic()
	 * 
	 * Scale-Aware Sampling: Combines BigInt sector coordinates with 
	 * FixedMath local offsets. This allows for noise sampling at 
	 * astronomical distances without the "pixelation" caused by float precision loss.
	 */
	T sample_galactic(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const;

	SimplexNoise() {
		seed(BigIntCore(12345LL));
	}
};

// Typedef for the deterministic physics/visual tier
typedef SimplexNoise<FixedMathCore> SimplexNoisef;

#endif // NOISE_SIMPLEX_H

--- END OF FILE core/math/noise_simplex.h ---
