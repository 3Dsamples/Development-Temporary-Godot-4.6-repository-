--- START OF FILE core/math/quaternion.cpp ---

#include "core/math/quaternion.h"
#include "core/math/basis.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the simulation tiers:
 * - Quaternionf: Bit-perfect 4D rotations for physics and kinematics (FixedMathCore).
 * - Quaternionb: Discrete macro-orientations for galactic sector mapping (BigIntCore).
 */
template struct Quaternion<FixedMathCore>;
template struct Quaternion<BigIntCore>;

// ============================================================================
// Basis-to-Quaternion Conversion (Deterministic Logic)
// ============================================================================

/**
 * Quaternion(const Basis<T> &p_basis)
 * 
 * High-fidelity extraction of rotation data from a 3x3 matrix.
 * Ported to Software-Defined Arithmetic to eliminate rounding drift.
 */
template <typename T>
Quaternion<T>::Quaternion(const Basis<T> &p_basis) {
	T trace = p_basis[0][0] + p_basis[1][1] + p_basis[2][2];
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	if (trace > zero) {
		T s = Math::sqrt(trace + one) * T(2LL);
		T inv_s = one / s;
		w = T(2147483648LL, true) * s; // 0.25 * s
		x = (p_basis[2][1] - p_basis[1][2]) * inv_s;
		y = (p_basis[0][2] - p_basis[2][0]) * inv_s;
		z = (p_basis[1][0] - p_basis[0][1]) * inv_s;
	} else {
		if (p_basis[0][0] > p_basis[1][1] && p_basis[0][0] > p_basis[2][2]) {
			T s = Math::sqrt(one + p_basis[0][0] - p_basis[1][1] - p_basis[2][2]) * T(2LL);
			T inv_s = one / s;
			w = (p_basis[2][1] - p_basis[1][2]) * inv_s;
			x = T(2147483648LL, true) * s;
			y = (p_basis[0][1] + p_basis[1][0]) * inv_s;
			z = (p_basis[0][2] + p_basis[2][0]) * inv_s;
		} else if (p_basis[1][1] > p_basis[2][2]) {
			T s = Math::sqrt(one + p_basis[1][1] - p_basis[0][0] - p_basis[2][2]) * T(2LL);
			T inv_s = one / s;
			w = (p_basis[0][2] - p_basis[2][0]) * inv_s;
			x = (p_basis[0][1] + p_basis[1][0]) * inv_s;
			y = T(2147483648LL, true) * s;
			z = (p_basis[1][2] + p_basis[2][1]) * inv_s;
		} else {
			T s = Math::sqrt(one + p_basis[2][2] - p_basis[0][0] - p_basis[1][1]) * T(2LL);
			T inv_s = one / s;
			w = (p_basis[1][0] - p_basis[0][1]) * inv_s;
			x = (p_basis[0][2] + p_basis[2][0]) * inv_s;
			y = (p_basis[1][2] + p_basis[2][1]) * inv_s;
			z = T(2147483648LL, true) * s;
		}
	}
}

// ============================================================================
// Sophisticated Interpolation (Bit-Perfect SLERP)
// ============================================================================

/**
 * slerp()
 * 
 * Spherical Linear Interpolation for orientations.
 * Optimized for high-frequency 120 FPS camera and robotic stabilization.
 */
template <typename T>
Quaternion<T> Quaternion<T>::slerp(const Quaternion<T> &p_to, T p_weight) const {
	T cos_half_theta = this->dot(p_to);
	Quaternion<T> to_final = p_to;

	// Shortest path resolve: flip destination if negative dot product
	if (cos_half_theta < MathConstants<T>::zero()) {
		to_final = -p_to;
		cos_half_theta = -cos_half_theta;
	}

	// Precision-Aware Fallback: If angles are nearly identical, use LERP
	// Threshold: 0.9999 (raw bits: 4294924294LL)
	if (Math::abs(cos_half_theta) >= T(4294924294LL, true)) {
		return Quaternion<T>(
			Math::lerp(x, to_final.x, p_weight),
			Math::lerp(y, to_final.y, p_weight),
			Math::lerp(z, to_final.z, p_weight),
			Math::lerp(w, to_final.w, p_weight)
		).normalized();
	}

	// Standard Deterministic SLERP
	T half_theta = Math::acos(cos_half_theta);
	T sin_half_theta = Math::sqrt(MathConstants<T>::one() - cos_half_theta * cos_half_theta);

	T ratio_a = Math::sin((MathConstants<T>::one() - p_weight) * half_theta) / sin_half_theta;
	T ratio_b = Math::sin(p_weight * half_theta) / sin_half_theta;

	return (*this * ratio_a) + (to_final * ratio_b);
}

/**
 * Deterministic Identity Constants
 */
const Quaternionf Quaternionf_IDENTITY = Quaternionf(
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(0LL, true), 
	FixedMathCore(FixedMathCore::ONE_RAW, true)
);

const Quaternionb Quaternionb_IDENTITY = Quaternionb(
	BigIntCore(0LL), 
	BigIntCore(0LL), 
	BigIntCore(0LL), 
	BigIntCore(1LL)
);

/**
 * Performance Synergy Note:
 * 
 * This CPP file provides the compiled symbols that allow the PhysicsServerHyper
 * to perform bit-perfect angular velocity integration across EnTT registries.
 * By maintaining strict bit-perfection in the Hamilton products and SLERP, 
 * network clients will never desync on rotational state even during 
 * high-velocity spaceship combat at 120 FPS.
 */

--- END OF FILE core/math/quaternion.cpp ---
