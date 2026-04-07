--- START OF FILE core/math/quaternion.h ---

#ifndef QUATERNION_H
#define QUATERNION_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

template <typename T>
struct Basis;

/**
 * Quaternion Template
 * 
 * Optimized 4D rotation logic for the Universal Solver.
 * Aligned to 32 bytes for zero-copy EnTT integration and high-speed Warp sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Quaternion {
	T x, y, z, w;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Quaternion() : x(T()), y(T()), z(T()), w(MathConstants<T>::one()) {}
	ET_SIMD_INLINE Quaternion(T p_x, T p_y, T p_z, T p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}

	/**
	 * Axis-Angle Constructor
	 * Deterministic conversion using FixedMathCore sin/cos.
	 */
	inline Quaternion(const Vector3<T> &p_axis, T p_angle) {
		T d = p_axis.length();
		if (d == MathConstants<T>::zero()) {
			x = y = z = MathConstants<T>::zero();
			w = MathConstants<T>::one();
		} else {
			T half_angle = p_angle * MathConstants<T>::half();
			T s = Math::sin(half_angle) / d;
			x = p_axis.x * s;
			y = p_axis.y * s;
			z = p_axis.z * s;
			w = Math::cos(half_angle);
		}
	}

	Quaternion(const Basis<T> &p_basis);

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T &operator[](int p_idx) { return (&x)[p_idx]; }
	ET_SIMD_INLINE const T &operator[](int p_idx) const { return (&x)[p_idx]; }

	// ------------------------------------------------------------------------
	// Operators (Hamiltonian Batch Math)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Quaternion<T> operator+(const Quaternion<T> &p_q) const { return Quaternion<T>(x + p_q.x, y + p_q.y, z + p_q.z, w + p_q.w); }
	ET_SIMD_INLINE Quaternion<T> operator-(const Quaternion<T> &p_q) const { return Quaternion<T>(x - p_q.x, y - p_q.y, z - p_q.z, w - p_q.w); }
	ET_SIMD_INLINE Quaternion<T> operator-() const { return Quaternion<T>(-x, -y, -z, -w); }

	ET_SIMD_INLINE Quaternion<T> operator*(const Quaternion<T> &p_q) const {
		return Quaternion<T>(
				w * p_q.x + x * p_q.w + y * p_q.z - z * p_q.y,
				w * p_q.y + y * p_q.w + z * p_q.x - x * p_q.z,
				w * p_q.z + z * p_q.w + x * p_q.y - y * p_q.x,
				w * p_q.w - x * p_q.x - y * p_q.y - z * p_q.z);
	}

	ET_SIMD_INLINE void operator*=(const Quaternion<T> &p_q) { *this = *this * p_q; }
	ET_SIMD_INLINE void operator*=(T p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; w *= p_scalar; }
	ET_SIMD_INLINE Quaternion<T> operator*(T p_scalar) const { return Quaternion<T>(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar); }

	ET_SIMD_INLINE bool operator==(const Quaternion<T> &p_q) const { return x == p_q.x && y == p_q.y && z == p_q.z && w == p_q.w; }
	ET_SIMD_INLINE bool operator!=(const Quaternion<T> &p_q) const { return x != p_q.x || y != p_q.y || z != p_q.z || w != p_q.w; }

	// ------------------------------------------------------------------------
	// Deterministic Quaternion Math
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T dot(const Quaternion<T> &p_q) const { return x * p_q.x + y * p_q.y + z * p_q.z + w * p_q.w; }
	ET_SIMD_INLINE T length_squared() const { return x * x + y * y + z * z + w * w; }
	ET_SIMD_INLINE T length() const { return Math::sqrt(length_squared()); }

	ET_SIMD_INLINE void normalize() {
		T l = length();
		if (l == MathConstants<T>::zero()) {
			x = y = z = MathConstants<T>::zero();
			w = MathConstants<T>::one();
		} else {
			T inv_l = MathConstants<T>::one() / l;
			x *= inv_l; y *= inv_l; z *= inv_l; w *= inv_l;
		}
	}

	ET_SIMD_INLINE Quaternion<T> normalized() const {
		Quaternion<T> q = *this;
		q.normalize();
		return q;
	}

	ET_SIMD_INLINE Quaternion<T> inverse() const { return Quaternion<T>(-x, -y, -z, w); }

	/**
	 * SLERP (Spherical Linear Interpolation)
	 * Bit-perfect orientation blending for high-frequency simulation frames.
	 */
	inline Quaternion<T> slerp(const Quaternion<T> &p_to, T p_weight) const {
		T cosom = dot(p_to);
		Quaternion<T> to_final = p_to;

		if (cosom < MathConstants<T>::zero()) {
			cosom = -cosom;
			to_final = -p_to;
		}

		T scale0, scale1;
		if ((MathConstants<T>::one() - cosom) > T(CMP_EPSILON)) {
			T omega = Math::acos(cosom);
			T sinom = Math::sin(omega);
			scale0 = Math::sin((MathConstants<T>::one() - p_weight) * omega) / sinom;
			scale1 = Math::sin(p_weight * omega) / sinom;
		} else {
			scale0 = MathConstants<T>::one() - p_weight;
			scale1 = p_weight;
		}

		return (*this * scale0) + (to_final * scale1);
	}

	/**
	 * xform()
	 * Rotates a Vector3 by this quaternion.
	 * Zero-copy execution path for Warp kernels.
	 */
	ET_SIMD_INLINE Vector3<T> xform(const Vector3<T> &p_v) const {
		Vector3<T> u(x, y, z);
		Vector3<T> uv = u.cross(p_v);
		Vector3<T> uuv = u.cross(uv);
		T two(2147483648LL, true); // 2.0 in FixedPoint bits if T is FixedMathCore
		if constexpr (std::is_same<T, BigIntCore>::value) {
			return p_v + ((uv * w) + uuv) * BigIntCore(2);
		} else {
			// Optimized v' = v + 2w(u x v) + 2(u x (u x v))
			return p_v + ((uv * w) + uuv) * FixedMathCore(2LL, false);
		}
	}

	// Godot UI Conversion
	operator String() const {
		return "(" + String(x.to_string().c_str()) + ", " + 
		             String(y.to_string().c_str()) + ", " + 
		             String(z.to_string().c_str()) + ", " + 
		             String(w.to_string().c_str()) + ")";
	}
};

typedef Quaternion<FixedMathCore> Quaternionf;
typedef Quaternion<BigIntCore> Quaternionb;

#endif // QUATERNION_H

--- END OF FILE core/math/quaternion.h ---
