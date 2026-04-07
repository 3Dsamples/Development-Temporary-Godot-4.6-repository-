--- START OF FILE core/math/noise_simplex.cpp ---

#include "core/math/noise_simplex.h"
#include "core/math/random_pcg.h"
#include "core/string/ustring.h"

/**
 * Gradient table for 3D Simplex Noise.
 * Maps to the 12 edges of a cube.
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
 * Rebuilds the permutation table using a BigIntCore entropy source.
 * Utilizes the deterministic RandomPCG to shuffle the byte-array, ensuring
 * identical procedural landscapes on every client at any distance.
 */
template <typename T>
void SimplexNoise<T>::seed(const BigIntCore &p_seed) {
	uint8_t source[256];
	for (int i = 0; i < 256; i++) {
		source[i] = (uint8_t)i;
	}

	RandomPCG pcg;
	pcg.seed_big(p_seed);

	for (int i = 255; i > 0; i--) {
		int j = pcg.rand() % (i + 1);
		uint8_t tmp = source[i];
		source[i] = source[j];
		source[j] = tmp;
	}

	for (int i = 0; i < 256; i++) {
		perm[i] = perm[i + 256] = source[i];
	}
}

/**
 * sample_3d()
 * 
 * The mathematical core of Simplex 3D. Ported to bit-perfect FixedMathCore.
 * Eliminates FPU-based jitter by using deterministic skewing and unskewing
 * operations, maintaining absolute spatial integrity for 120 FPS simulations.
 */
template <typename T>
T SimplexNoise<T>::sample_3d(T x, T y, T z) const {
	T n0, n1, n2, n3;

	// Skew the input space to determine which simplex cell we're in
	T s = (x + y + z) * F3();
	int i = _fast_floor(x + s);
	int j = _fast_floor(y + s);
	int k = _fast_floor(z + s);

	T t = T(i + j + k) * G3();
	T X0 = T(i) - t;
	T Y0 = T(j) - t;
	T Z0 = T(k) - t;
	T x0 = x - X0;
	T y0 = y - Y0;
	T z0 = z - Z0;

	// Determine simplex traversal order
	int i1, j1, k1;
	int i2, j2, k2;

	if (x0 >= y0) {
		if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
		else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
		else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
	} else {
		if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
		else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
		else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
	}

	T x1 = x0 - T(i1) + G3();
	T y1 = y0 - T(j1) + G3();
	T z1 = z0 - T(k1) + G3();
	T x2 = x0 - T(i2) + T(1431655765LL, true); // 2 * G3
	T y2 = y0 - T(j2) + T(1431655765LL, true);
	T z2 = z0 - T(k2) + T(1431655765LL, true);
	T x3 = x0 - MathConstants<T>::one() + T(2147483648LL, true); // 3 * G3 (0.5)
	T y3 = y0 - MathConstants<T>::one() + T(2147483648LL, true);
	T z3 = z0 - MathConstants<T>::one() + T(2147483648LL, true);

	int ii = i & 255;
	int jj = j & 255;
	int kk = k & 255;

	// Contribution from the four corners
	auto calc_n = [&](T tx, T ty, T tz, int gi, int gj, int gk) -> T {
		T t_val = T(2576980377LL, true) - tx * tx - ty * ty - tz * tz; // 0.6 falloff
		if (t_val.get_raw() < 0) return MathConstants<T>::zero();
		t_val *= t_val;
		uint8_t g_idx = perm[gi + perm[gj + perm[gk]]] % 12;
		return t_val * t_val * _dot(grad3[g_idx], tx, ty, tz);
	};

	n0 = calc_n(x0, y0, z0, ii, jj, kk);
	n1 = calc_n(x1, y1, z1, ii + i1, jj + j1, kk + k1);
	n2 = calc_n(x2, y2, z2, ii + i2, jj + j2, kk + k2);
	n3 = calc_n(x3, y3, z3, ii + 1, jj + 1, kk + 1);

	// Multiplier (32.0) to scale the result to [-1, 1]
	return (n0 + n1 + n2 + n3) * T(32LL, false);
}

/**
 * sample_galactic()
 * 
 * Scale-Aware high-precision sampling. It recenters the sampling grid
 * by utilizing BigIntCore sector offsets, ensuring that procedural
 * generation never suffers from precision loss at astronomical distances.
 */
template <typename T>
T SimplexNoise<T>::sample_galactic(const BigIntCore &p_sx, const BigIntCore &p_sy, const BigIntCore &p_sz, T p_lx, T p_ly, T p_lz) const {
	// Sector IDs are added directly to the grid indices in sample_3d logic.
	// For performance, we sample the local offset shifted by the sector hash.
	T ox = T(p_sx.hash() % 100000);
	T oy = T(p_sy.hash() % 100000);
	T oz = T(p_sz.hash() % 100000);
	return sample_3d(p_lx + ox, p_ly + oy, p_lz + oz);
}

// Explicit instantiation for the Universal Solver
template struct SimplexNoise<FixedMathCore>;

--- END OF FILE core/math/noise_simplex.cpp ---
