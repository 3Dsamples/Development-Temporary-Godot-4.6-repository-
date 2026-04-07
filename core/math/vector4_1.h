--- START OF FILE core/math/vector4.h ---

#ifndef VECTOR4_H
#define VECTOR4_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector4 Template
 * 
 * 32-byte aligned 4D vector for deterministic high-dimensional logic.
 * Essential for projection matrices, 4D physics tensors, and spectral energy.
 */
template <typename T>
struct ET_ALIGN_32 Vector4 {
	T x, y, z, w;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector4() : x(MathConstants<T>::zero()), y(MathConstants<T>::zero()), z(MathConstants<T>::zero()), w(MathConstants<T>::zero()) {}
	_FORCE_INLINE_ Vector4(T p_x, T p_y, T p_z, T p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}
	_FORCE_INLINE_ Vector4(const Vector3<T> &p_v, T p_w) : x(p_v.x), y(p_v.y), z(p_v.z), w(p_w) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector4<T> operator+(const Vector4<T> &p_v) const { return Vector4<T>(x + p_v.x, y + p_v.y, z + p_v.z, w + p_v.w); }
	_FORCE_INLINE_ void operator+=(const Vector4<T> &p_v) { x += p_v.x; y += p_v.y; z += p_v.z; w += p_v.w; }
	_FORCE_INLINE_ Vector4<T> operator-(const Vector4<T> &p_v) const { return Vector4<T>(x - p_v.x, y - p_v.y, z - p_v.z, w - p_v.w); }
	_FORCE_INLINE_ void operator-=(const Vector4<T> &p_v) { x -= p_v.x; y -= p_v.y; z -= p_v.z; w -= p_v.w; }

	_FORCE_INLINE_ Vector4<T> operator*(const Vector4<T> &p_v) const { return Vector4<T>(x * p_v.x, y * p_v.y, z * p_v.z, w * p_v.w); }
	_FORCE_INLINE_ Vector4<T> operator*(const T &p_scalar) const { return Vector4<T>(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar); }
	_FORCE_INLINE_ void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; w *= p_scalar; }

	_FORCE_INLINE_ Vector4<T> operator/(const Vector4<T> &p_v) const { return Vector4<T>(x / p_v.x, y / p_v.y, z / p_v.z, w / p_v.w); }
	_FORCE_INLINE_ Vector4<T> operator/(const T &p_scalar) const { return Vector4<T>(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar); }
	_FORCE_INLINE_ void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; z /= p_scalar; w /= p_scalar; }

	_FORCE_INLINE_ Vector4<T> operator-() const { return Vector4<T>(-x, -y, -z, -w); }

	_FORCE_INLINE_ bool operator==(const Vector4<T> &p_v) const { return x == p_v.x && y == p_v.y && z == p_v.z && w == p_v.w; }
	_FORCE_INLINE_ bool operator!=(const Vector4<T> &p_v) const { return x != p_v.x || y != p_v.y || z != p_v.z || w != p_v.w; }

	_FORCE_INLINE_ T &operator[](int p_idx) { return (&x)[p_idx]; }
	_FORCE_INLINE_ const T &operator[](int p_idx) const { return (&x)[p_idx]; }

	// ------------------------------------------------------------------------
	// Geometric & Tensor API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T dot(const Vector4<T> &p_v) const { return x * p_v.x + y * p_v.y + z * p_v.z + w * p_v.w; }
	_FORCE_INLINE_ T length_squared() const { return x * x + y * y + z * z + w * w; }
	_FORCE_INLINE_ T length() const { return Math::sqrt(length_squared()); }

	void normalize() {
		T l_sq = length_squared();
		if (l_sq != MathConstants<T>::zero()) {
			T l = Math::sqrt(l_sq);
			x /= l; y /= l; z /= l; w /= l;
		}
	}

	_FORCE_INLINE_ Vector4<T> normalized() const {
		Vector4<T> v = *this;
		v.normalize();
		return v;
	}

	/**
	 * perspective_divide()
	 * Converts homogeneous coordinates to 3D. 
	 * Essential for bit-perfect camera projection in galactic rendering.
	 */
	_FORCE_INLINE_ Vector3<T> perspective_divide() const {
		if (w == MathConstants<T>::zero()) return Vector3<T>(x, y, z);
		T inv_w = MathConstants<T>::one() / w;
		return Vector3<T>(x * inv_w, y * inv_w, z * inv_w);
	}

	/**
	 * spectral_mix()
	 * Specialized for Spectral Energy Tensors. Performs a bit-perfect 
	 * weighted blend between 4D color/energy components.
	 */
	_FORCE_INLINE_ Vector4<T> spectral_mix(const Vector4<T> &p_other, T p_weight) const {
		return lerp(p_other, p_weight);
	}

	_FORCE_INLINE_ Vector4<T> lerp(const Vector4<T> &p_to, T p_weight) const {
		return Vector4<T>(
				Math::lerp(x, p_to.x, p_weight),
				Math::lerp(y, p_to.y, p_weight),
				Math::lerp(z, p_to.z, p_weight),
				Math::lerp(w, p_to.w, p_weight));
	}

	_FORCE_INLINE_ Vector4<T> abs() const { return Vector4<T>(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w)); }

	_FORCE_INLINE_ Vector4<T> snapped(const Vector4<T> &p_step) const {
		return Vector4<T>(Math::snapped(x, p_step.x), Math::snapped(y, p_step.y), Math::snapped(z, p_step.z), Math::snapped(w, p_step.w));
	}

	operator String() const { 
		return "(" + String(x.to_string().c_str()) + ", " + 
		             String(y.to_string().c_str()) + ", " + 
		             String(z.to_string().c_str()) + ", " + 
		             String(w.to_string().c_str()) + ")"; 
	}
};

// Simulation Tier Typedefs
typedef Vector4<FixedMathCore> Vector4f; // Bit-perfect 4D Tensors / Homogeneous Coords
typedef Vector4<BigIntCore> Vector4b;    // Discrete 4D Metadata / State Bitsets

#endif // VECTOR4_H

--- END OF FILE core/math/vector4.h ---
