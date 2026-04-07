--- START OF FILE core/math/quaternion.cpp ---

#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Generates the compiled machine code for the deterministic and macro-scale tiers.
 * This allows EnTT to manage rotation components as raw data streams while 
 * Warp kernels invoke these robust mathematical routines during batch processing.
 */
template struct Quaternion<FixedMathCore>;
template struct Quaternion<BigIntCore>;

/**
 * Quaternion(const Basis<T> &p_basis)
 * 
 * Robust Matrix-to-Quaternion conversion.
 * Ported to Software-Defined Arithmetic to eliminate rounding drift.
 */
template <typename T>
Quaternion<T>::Quaternion(const Basis<T> &p_basis) {
	T trace = p_basis[0][0] + p_basis[1][1] + p_basis[2][2];
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	if (trace > zero) {
		T s = Math::sqrt(trace + one) * FixedMathCore(2LL, false);
		T inv_s = one / s;
		w = FixedMathCore(2147483648LL, true) * s; // 0.25 * s
		x = (p_basis[2][1] - p_basis[1][2]) * inv_s;
		y = (p_basis[0][2] - p_basis[2][0]) * inv_s;
		z = (p_basis[1][0] - p_basis[0][1]) * inv_s;
	} else {
		if (p_basis[0][0] > p_basis[1][1] && p_basis[0][0] > p_basis[2][2]) {
			T s = Math::sqrt(one + p_basis[0][0] - p_basis[1][1] - p_basis[2][2]) * FixedMathCore(2LL, false);
			T inv_s = one / s;
			w = (p_basis[2][1] - p_basis[1][2]) * inv_s;
			x = FixedMathCore(2147483648LL, true) * s; // 0.25 * s
			y = (p_basis[0][1] + p_basis[1][0]) * inv_s;
			z = (p_basis[0][2] + p_basis[2][0]) * inv_s;
		} else if (p_basis[1][1] > p_basis[2][2]) {
			T s = Math::sqrt(one + p_basis[1][1] - p_basis[0][0] - p_basis[2][2]) * FixedMathCore(2LL, false);
			T inv_s = one / s;
			w = (p_basis[0][2] - p_basis[2][0]) * inv_s;
			x = (p_basis[0][1] + p_basis[1][0]) * inv_s;
			y = FixedMathCore(2147483648LL, true) * s;
			z = (p_basis[1][2] + p_basis[2][1]) * inv_s;
		} else {
			T s = Math::sqrt(one + p_basis[2][2] - p_basis[0][0] - p_basis[1][1]) * FixedMathCore(2LL, false);
			T inv_s = one / s;
			w = (p_basis[1][0] - p_basis[0][1]) * inv_s;
			x = (p_basis[0][2] + p_basis[2][0]) * inv_s;
			y = (p_basis[1][2] + p_basis[2][1]) * inv_s;
			z = FixedMathCore(2147483648LL, true) * s;
		}
	}
}

// Instantiate specific constructor implementations
template Quaternion<FixedMathCore>::Quaternion(const Basis<FixedMathCore> &p_basis);
template Quaternion<BigIntCore>::Quaternion(const Basis<BigIntCore> &p_basis);

/**
 * Deterministic Identity Constants
 */
const Quaternionf Quaternionf_IDENTITY = Quaternionf(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(1LL, false));
const Quaternionb Quaternionb_IDENTITY = Quaternionb(BigIntCore(0), BigIntCore(0), BigIntCore(0), BigIntCore(1));

--- END OF FILE core/math/quaternion.cpp ---
