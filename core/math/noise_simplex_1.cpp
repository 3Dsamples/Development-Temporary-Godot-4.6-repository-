--- START OF FILE core/math/noise_simplex.cpp ---

#include "core/math/noise_simplex.h"
#include "core/math/random_pcg.h"

/**
 * Deterministic 3D Gradients
 * Midpoints of the 12 edges of a cube.
 */
template <typename T>
const int8_t SimplexNoise<T>::grad3[12][3] = {
	{ 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 },
	{ 1, 0, 1 }, { -1, 0, 1 }, { 1, 0, -1 }, { -1, 0, -1 },
	{ 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 }
};

/**
 * seed()
 * 
 * Generates a deterministic permutation table.
 * Uses a RandomPCG engine seeded by the BigIntCore handle to shuffle
 * the 256-byte lookup table. This ensures procedural stability across the network.
 */
template <typename T>
void SimplexNoise<T>::seed(const BigIntCore &p_seed) {
	uint8_t source[256];
	for (int i = 0; i < 256; i++) {
		source[i] = (uint8_t)i;
	}

	RandomPCG pcg;
	pcg.seed(p_seed.hash());

	for (int i = 255; i > 0; i--) {
		uint32_t r = pcg.rand() % (i + 1);
		uint8_t tmp = source[i];
		source[i] = source[r];
		source[r] = tmp;
	}

	for (int i = 0; i < 256; i++) {
		perm[i] = source[i];
		perm[i + 256] = source[i];
	}
}

/**
 * sample_3d()
 * 
 * Heavy implementation of the Simplex 3D kernel.
 * Ported to Software-Defined Arithmetic to maintain 120 FPS synchronization.
 */
template <typename T>
T SimplexNoise<T>::sample_3d(T x, T y, T z) const {
	T n0, n1, n2, n3; // Noise contributions from the four corners

	// 1. Skew the input space to determine which simplex cell we're in
	T s = (x + y + z) * F3(); 
	int i = _fast_floor(x + s);
	int j = _fast_floor(y + s);
	int k = _fast_floor(z + s);

	T t = T(static_cast<int64_t>(i + j + k)) * G3();
	T X0 = T(static_cast<int64_t>(i)) - t;
	T Y0 = T(static_cast<int64_t>(j)) - t;
	T Z0 = T(static_cast<int64_t>(k)) - t;
	T x0 = x - X0;
	T y0 = y - Y0;
	T z0 = z - Z0;

	// 2. Determine which simplex we are in
	int i1, j1, k1; // Offsets for second corner
	int i2, j2, k2; // Offsets for third corner

	if (x0 >= y0) {
		if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // X Y Z order
		else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; } // X Z Y order
		else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; } // Z X Y order
	} else { // x0 < y0
		if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; } // Z Y X order
		else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; } // Y Z X order
		else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; } // Y X Z order
	}

	// 3. Coordinate offsets for remaining corners
	T x1 = x0 - T(static_cast<int64_t>(i1)) + G3();
	T y1 = y0 - T(static_cast<int64_t>(j1)) + G3();
	T z1 = z0 - T(static_cast<int64_t>(k1)) + G3();
	
	T g3_2 = G3() * T(2LL);
	T x2 = x0 - T(static_cast<int64_t>(i2)) + g3_2;
	T y2 = y0 - T(static_cast<int64_t>(j2)) + g3_2;
	T z2 = z0 - T(static_cast<int64_t>(k2)) + g3_2;
	
	T g3_3 = G3() * T(3LL);
	T x3 = x0 - T(1LL) + g3_3;
	T y3 = y0 - T(1LL) + g3_3;
	T z3 = z0 - T(1LL) + g3_3;

	// 4. Wrap indices and fetch gradients
	int ii = i & 255;
	int jj = j & 255;
	int kk = k & 255;

	// 5. Calculate contributions from each corner
	auto calc_contribution = [&](T tx, T ty, T tz, int gi, int gj, int gk) -> T {
		// Falloff: (0.6 - x^2 - y^2 - z^2)
		T t_val = T(2576980377LL, true) - tx * tx - ty * ty - tz * tz;
		if (t_val.get_raw() < 0) return MathConstants<T>::zero();
		
		t_val *= t_val;
		uint8_t g_idx = perm[gi + perm[gj + perm[gk]]] % 12;
		return t_val * t_val * _dot(grad3[g_idx], tx, ty, tz);
	};

	n0 = calc_contribution(x0, y0, z0, ii, jj, kk);
	n1 = calc_contribution(x1, y1, z1, ii + i1, jj + j1, kk + k1);
	n2 = calc_contribution(x2, y2, z2, ii + i2, jj + j2, kk + k2);
	n3 = calc_contribution(x3, y3, z3, ii + 1, jj + 1, kk + 1);

	// 6. Scale and return [-1, 1] range. Multiplier 32.0 in FixedMath
	return (n0 + n1 + n2 + n3) * T(32LL);
}

/**
 * sample_galactic()
 * 
 * Re-centers the sampling grid based on the BigIntCore sector coordinates.
 * This effectively offsets the noise seed per sector, ensuring infinite
 * variety and zero precision loss in astronomical volumes.
 */
template <typename T>
T SimplexNoise<T>::sample_galactic(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const {
	// Derive local noise offsets from the unique sector hashes
	T ox = T(static_cast<int64_t>(p_sx.hash()));
	T oy = T(static_cast<int64_t>(p_sy.hash()));
	T oz = T(static_cast<int64_t>(p_sz.hash()));
	
	return sample_3d(p_lx + ox, p_ly + oy, p_lz + oz);
}

// Explicit Instantiation for the deterministic tier
template struct SimplexNoise<FixedMathCore>;

--- END OF FILE core/math/noise_simplex.cpp ---
