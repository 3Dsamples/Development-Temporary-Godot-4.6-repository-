--- START OF FILE core/math/vector3.h ---

#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector3 Template
 * 
 * The primary 3D spatial component for the Universal Solver.
 * Aligned for 120 FPS Warp-kernel execution and EnTT SoA batching.
 */
template <typename T>
struct ET_ALIGN_32 Vector3 {
	T x, y, z;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3() : x(T()), y(T()), z(T()) {}
	ET_SIMD_INLINE Vector3(T p_x, T p_y, T p_z) : x(p_x), y(p_y), z(p_z) {}

	// ------------------------------------------------------------------------
	// Operators (Hardware-Agnostic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> operator+(const Vector3<T> &p_v) const { return Vector3<T>(x + p_v.x, y + p_v.y, z + p_v.z); }
	ET_SIMD_INLINE void operator+=(const Vector3<T> &p_v) { x += p_v.x; y += p_v.y; z += p_v.z; }
	ET_SIMD_INLINE Vector3<T> operator-(const Vector3<T> &p_v) const { return Vector3<T>(x - p_v.x, y - p_v.y, z - p_v.z); }
	ET_SIMD_INLINE void operator-=(const Vector3<T> &p_v) { x -= p_v.x; y -= p_v.y; z -= p_v.z; }
	ET_SIMD_INLINE Vector3<T> operator*(const Vector3<T> &p_v) const { return Vector3<T>(x * p_v.x, y * p_v.y, z * p_v.z); }
	ET_SIMD_INLINE Vector3<T> operator*(const T &p_scalar) const { return Vector3<T>(x * p_scalar, y * p_scalar, z * p_scalar); }
	ET_SIMD_INLINE void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; }
	ET_SIMD_INLINE Vector3<T> operator/(const Vector3<T> &p_v) const { return Vector3<T>(x / p_v.x, y / p_v.y, z / p_v.z); }
	ET_SIMD_INLINE Vector3<T> operator/(const T &p_scalar) const { return Vector3<T>(x / p_scalar, y / p_scalar, z / p_scalar); }
	ET_SIMD_INLINE void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; z /= p_scalar; }
	ET_SIMD_INLINE Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

	ET_SIMD_INLINE bool operator==(const Vector3<T> &p_v) const { return x == p_v.x && y == p_v.y && z == p_v.z; }
	ET_SIMD_INLINE bool operator!=(const Vector3<T> &p_v) const { return x != p_v.x || y != p_v.y || z != p_v.z; }
	ET_SIMD_INLINE bool operator<(const Vector3<T> &p_v) const {
		if (x != p_v.x) return x < p_v.x;
		if (y != p_v.y) return y < p_v.y;
		return z < p_v.z;
	}

	// ------------------------------------------------------------------------
	// Geometric Math (SIMD Optimized)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T length() const { return Math::sqrt(x * x + y * y + z * z); }
	ET_SIMD_INLINE T length_squared() const { return x * x + y * y + z * z; }

	ET_SIMD_INLINE void normalize() {
		T l_sq = length_squared();
		if (l_sq != T(0)) {
			T l = Math::sqrt(l_sq);
			x /= l; y /= l; z /= l;
		}
	}

	ET_SIMD_INLINE Vector3<T> normalized() const {
		Vector3<T> v = *this;
		v.normalize();
		return v;
	}

	ET_SIMD_INLINE T dot(const Vector3<T> &p_v) const { return x * p_v.x + y * p_v.y + z * p_v.z; }
	
	ET_SIMD_INLINE Vector3<T> cross(const Vector3<T> &p_v) const {
		return Vector3<T>(
				(y * p_v.z) - (z * p_v.y),
				(z * p_v.x) - (x * p_v.z),
				(x * p_v.y) - (y * p_v.x));
	}

	ET_SIMD_INLINE T distance_to(const Vector3<T> &p_v) const { return (p_v - *this).length(); }
	ET_SIMD_INLINE T distance_squared_to(const Vector3<T> &p_v) const { return (p_v - *this).length_squared(); }

	ET_SIMD_INLINE Vector3<T> abs() const { return Vector3<T>(Math::abs(x), Math::abs(y), Math::abs(z)); }

	ET_SIMD_INLINE Vector3<T> lerp(const Vector3<T> &p_to, T p_weight) const {
		return Vector3<T>(
				Math::lerp(x, p_to.x, p_weight),
				Math::lerp(y, p_to.y, p_weight),
				Math::lerp(z, p_to.z, p_weight));
	}

	ET_SIMD_INLINE Vector3<T> snapped(const Vector3<T> &p_step) const {
		return Vector3<T>(Math::snapped(x, p_step.x), Math::snapped(y, p_step.y), Math::snapped(z, p_step.z));
	}

	/**
	 * rotated()
	 * Deterministic rotation around an arbitrary axis.
	 */
	ET_SIMD_INLINE Vector3<T> rotated(const Vector3<T> &p_axis, T p_angle) const {
		T s = Math::sin(p_angle);
		T c = Math::cos(p_angle);
		return p_axis * dot(p_axis) + (*this - p_axis * dot(p_axis)) * c + p_axis.cross(*this) * s;
	}

	// ------------------------------------------------------------------------
	// Helpers for Zero-Copy Warp Kernels
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T& operator[](int p_idx) { return (&x)[p_idx]; }
	ET_SIMD_INLINE const T& operator[](int p_idx) const { return (&x)[p_idx]; }

	// Godot Native String for debugging/UI
	operator String() const { 
		return "(" + String(x.to_string().c_str()) + ", " + String(y.to_string().c_str()) + ", " + String(z.to_string().c_str()) + ")"; 
	}
};

// Simulation Type Definitions
typedef Vector3<FixedMathCore> Vector3f; // Deterministic Physics/Trajectories
typedef Vector3<BigIntCore> Vector3b;    // Massive Discrete Space/Voxel Grids

#endif // VECTOR3_H

--- END OF FILE core/math/vector3.h ---
