--- START OF FILE core/math/vector4.h ---

#ifndef VECTOR4_H
#define VECTOR4_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector4 Template
 * 
 * High-performance 4D vector for homogeneous transformations and 
 * relativistic physics tensors. Aligned for NVIDIA Warp SoA sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Vector4 {
	T x, y, z, w;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector4() : x(T()), y(T()), z(T()), w(T()) {}
	ET_SIMD_INLINE Vector4(T p_x, T p_y, T p_z, T p_w) : x(p_x), y(p_y), z(p_z), w(p_w) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector4<T> operator+(const Vector4<T> &p_v) const { return Vector4<T>(x + p_v.x, y + p_v.y, z + p_v.z, w + p_v.w); }
	ET_SIMD_INLINE void operator+=(const Vector4<T> &p_v) { x += p_v.x; y += p_v.y; z += p_v.z; w += p_v.w; }
	ET_SIMD_INLINE Vector4<T> operator-(const Vector4<T> &p_v) const { return Vector4<T>(x - p_v.x, y - p_v.y, z - p_v.z, w - p_v.w); }
	ET_SIMD_INLINE void operator-=(const Vector4<T> &p_v) { x -= p_v.x; y -= p_v.y; z -= p_v.z; w -= p_v.w; }
	ET_SIMD_INLINE Vector4<T> operator*(const Vector4<T> &p_v) const { return Vector4<T>(x * p_v.x, y * p_v.y, z * p_v.z, w * p_v.w); }
	ET_SIMD_INLINE Vector4<T> operator*(const T &p_scalar) const { return Vector4<T>(x * p_scalar, y * p_scalar, z * p_scalar, w * p_scalar); }
	ET_SIMD_INLINE void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; w *= p_scalar; }
	ET_SIMD_INLINE Vector4<T> operator/(const Vector4<T> &p_v) const { return Vector4<T>(x / p_v.x, y / p_v.y, z / p_v.z, w / p_v.w); }
	ET_SIMD_INLINE Vector4<T> operator/(const T &p_scalar) const { return Vector4<T>(x / p_scalar, y / p_scalar, z / p_scalar, w / p_scalar); }
	ET_SIMD_INLINE void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; z /= p_scalar; w /= p_scalar; }
	ET_SIMD_INLINE Vector4<T> operator-() const { return Vector4<T>(-x, -y, -z, -w); }

	ET_SIMD_INLINE bool operator==(const Vector4<T> &p_v) const { return x == p_v.x && y == p_v.y && z == p_v.z && w == p_v.w; }
	ET_SIMD_INLINE bool operator!=(const Vector4<T> &p_v) const { return x != p_v.x || y != p_v.y || z != p_v.z || w != p_v.w; }

	// ------------------------------------------------------------------------
	// Geometric & Warp Kernel Math
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T dot(const Vector4<T> &p_v) const { return x * p_v.x + y * p_v.y + z * p_v.z + w * p_v.w; }
	ET_SIMD_INLINE T length_squared() const { return x * x + y * y + z * z + w * w; }
	ET_SIMD_INLINE T length() const { return Math::sqrt(length_squared()); }

	ET_SIMD_INLINE void normalize() {
		T l_sq = length_squared();
		if (l_sq != T(0)) {
			T l = Math::sqrt(l_sq);
			x /= l; y /= l; z /= l; w /= l;
		}
	}

	ET_SIMD_INLINE Vector4<T> normalized() const {
		Vector4<T> v = *this;
		v.normalize();
		return v;
	}

	ET_SIMD_INLINE Vector4<T> abs() const { return Vector4<T>(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w)); }

	ET_SIMD_INLINE Vector4<T> lerp(const Vector4<T> &p_to, T p_weight) const {
		return Vector4<T>(
				Math::lerp(x, p_to.x, p_weight),
				Math::lerp(y, p_to.y, p_weight),
				Math::lerp(z, p_to.z, p_weight),
				Math::lerp(w, p_to.w, p_weight));
	}

	ET_SIMD_INLINE Vector4<T> snapped(const Vector4<T> &p_step) const {
		return Vector4<T>(
				Math::snapped(x, p_step.x),
				Math::snapped(y, p_step.y),
				Math::snapped(z, p_step.z),
				Math::snapped(w, p_step.w));
	}

	// Internal zero-copy Warp indexing
	ET_SIMD_INLINE T& operator[](int p_idx) { return (&x)[p_idx]; }
	ET_SIMD_INLINE const T& operator[](int p_idx) const { return (&x)[p_idx]; }

	// Godot UI Conversion
	operator String() const {
		return "(" + String(x.to_string().c_str()) + ", " + 
		             String(y.to_string().c_str()) + ", " + 
		             String(z.to_string().c_str()) + ", " + 
		             String(w.to_string().c_str()) + ")";
	}
};

// Simulation Tier Aliases
typedef Vector4<FixedMathCore> Vector4f; // Deterministic 4D Projection / Tensors
typedef Vector4<BigIntCore> Vector4b;    // Macro-Scale 4D Discrete Metadata

#endif // VECTOR4_H

--- END OF FILE core/math/vector4.h ---
