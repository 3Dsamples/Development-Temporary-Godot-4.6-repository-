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
 * 32-byte aligned 4D rotation component.
 * Strictly deterministic math for bit-perfect synchronization.
 */
template <typename T>
struct ET_ALIGN_32 Quaternion {
	T x, y, z, w;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Quaternion() : x(MathConstants<T>::zero()), y(MathConstants<T>::zero()), z(MathConstants<T>::zero()), w(MathConstants<T>::one()) {}
	_FORCE_INLINE_ Quaternion(T p_x, T p_y, T p_z, T p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}

	/**
	 * Axis-Angle Constructor
	 * Uses deterministic sin/cos for bit-perfect orientation creation.
	 */
	Quaternion(const Vector3<T> &p_axis, T p_angle) {
		T d = p_axis.length();
		if (unlikely(d == MathConstants<T>::zero())) {
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
	// Operators (Hamilton Product)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T &operator[](int p_idx) { return (&x)[p_idx]; }
	_FORCE_INLINE_ const T &operator[](int p_idx) const { return (&x)[p_idx]; }

	/**
	 * operator*
	 * Hamilton product implemented in Software-Defined Arithmetic.
	 * 120 FPS Optimized.
	 */
	_FORCE_INLINE_ Quaternion<T> operator*(const Quaternion<T> &p_q) const {
		return Quaternion<T>(
				w * p_q.x + x * p_q.w + y * p_q.z - z * p_q.y,
				w * p_q.y + y * p_q.w + z * p_q.x - x * p_q.z,
				w * p_q.z + z * p_q.w + x * p_q.y - y * p_q.x,
				w * p_q.w - x * p_q.x - y * p_q.y - z * p_q.z);
	}

	_FORCE_INLINE_ void operator*=(const Quaternion<T> &p_q) { *this = *this * p_q; }

	_FORCE_INLINE_ Quaternion<T> operator+(const Quaternion<T> &p_q) const { return Quaternion<T>(x + p_q.x, y + p_q.y, z + p_q.z, w + p_q.w); }
	_FORCE_INLINE_ Quaternion<T> operator-(const Quaternion<T> &p_q) const { return Quaternion<T>(x - p_q.x, y - p_q.y, z - p_q.z, w - p_q.w); }
	_FORCE_INLINE_ Quaternion<T> operator-() const { return Quaternion<T>(-x, -y, -z, -w); }
	_FORCE_INLINE_ Quaternion<T> operator*(T p_scalar) const { return Quaternion<T>(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar); }
	_FORCE_INLINE_ void operator*=(T p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; w *= p_scalar; }

	_FORCE_INLINE_ bool operator==(const Quaternion<T> &p_q) const { return x == p_q.x && y == p_q.y && z == p_q.z && w == p_q.w; }
	_FORCE_INLINE_ bool operator!=(const Quaternion<T> &p_q) const { return !(*this == p_q); }

	// ------------------------------------------------------------------------
	// Geometric API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T dot(const Quaternion<T> &p_q) const { return x * p_q.x + y * p_q.y + z * p_q.z + w * p_q.w; }
	_FORCE_INLINE_ T length_squared() const { return x * x + y * y + z * z + w * w; }
	_FORCE_INLINE_ T length() const { return Math::sqrt(length_squared()); }

	void normalize() {
		T l = length();
		if (unlikely(l == MathConstants<T>::zero())) {
			x = y = z = MathConstants<T>::zero();
			w = MathConstants<T>::one();
		} else {
			T inv_l = MathConstants<T>::one() / l;
			x *= inv_l; y *= inv_l; z *= inv_l; w *= inv_l;
		}
	}

	_FORCE_INLINE_ Quaternion<T> normalized() const {
		Quaternion<T> q = *this;
		q.normalize();
		return q;
	}

	_FORCE_INLINE_ Quaternion<T> inverse() const { return Quaternion<T>(-x, -y, -z, w); }

	/**
	 * xform()
	 * High-speed rotation of a vector by this quaternion.
	 * v' = v + 2 * q_xyz.cross(q_xyz.cross(v) + q_w * v)
	 */
	_FORCE_INLINE_ Vector3<T> xform(const Vector3<T> &p_v) const {
		Vector3<T> q_v(x, y, z);
		Vector3<T> uv = q_v.cross(p_v);
		Vector3<T> uuv = q_v.cross(uv);
		return p_v + ((uv * w) + uuv) * T(2LL);
	}

	// ------------------------------------------------------------------------
	// Sophisticated Interpolation (120 FPS Stability)
	// ------------------------------------------------------------------------

	/**
	 * slerp()
	 * Bit-perfect Spherical Linear Interpolation.
	 * Ensures the shortest path is always taken to prevent orientation flipping.
	 */
	Quaternion<T> slerp(const Quaternion<T> &p_to, T p_weight) const;

	/**
	 * integrate_angular_velocity()
	 * Advanced Behavior: Updates orientation based on angular velocity tensor.
	 * dq/dt = 0.5 * omega * q
	 */
	_FORCE_INLINE_ Quaternion<T> integrate_angular_velocity(const Vector3<T> &p_omega, T p_delta) const {
		Quaternion<T> delta_q(p_omega.x, p_omega.y, p_omega.z, MathConstants<T>::zero());
		Quaternion<T> res = *this + (delta_q * (*this)) * (p_delta * MathConstants<T>::half());
		return res.normalized();
	}

	operator String() const { 
		return "(" + String(x.to_string().c_str()) + ", " + 
		             String(y.to_string().c_str()) + ", " + 
		             String(z.to_string().c_str()) + ", " + 
		             String(w.to_string().c_str()) + ")"; 
	}
};

// Simulation Tier Typedefs
typedef Quaternion<FixedMathCore> Quaternionf; // Bit-perfect 3D Rotations
typedef Quaternion<BigIntCore> Quaternionb;    // Discrete Macro-Orientations

#endif // QUATERNION_H

--- END OF FILE core/math/quaternion.h ---
