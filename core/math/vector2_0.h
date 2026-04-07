--- START OF FILE core/math/vector2.h ---

#ifndef VECTOR2_H
#define VECTOR2_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector2 Template
 * 
 * High-performance 2D vector for the Universal Solver.
 * Aligned for Warp kernel batching and EnTT SoA storage.
 */
template <typename T>
struct ET_ALIGN_32 Vector2 {
	T x, y;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector2() : x(T()), y(T()) {}
	ET_SIMD_INLINE Vector2(T p_x, T p_y) : x(p_x), y(p_y) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic & Batch-Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector2<T> operator+(const Vector2<T> &p_v) const { return Vector2<T>(x + p_v.x, y + p_v.y); }
	ET_SIMD_INLINE void operator+=(const Vector2<T> &p_v) { x += p_v.x; y += p_v.y; }
	ET_SIMD_INLINE Vector2<T> operator-(const Vector2<T> &p_v) const { return Vector2<T>(x - p_v.x, y - p_v.y); }
	ET_SIMD_INLINE void operator-=(const Vector2<T> &p_v) { x -= p_v.x; y -= p_v.y; }
	ET_SIMD_INLINE Vector2<T> operator*(const Vector2<T> &p_v1) const { return Vector2<T>(x * p_v1.x, y * p_v1.y); }
	ET_SIMD_INLINE Vector2<T> operator*(const T &p_scalar) const { return Vector2<T>(x * p_scalar, y * p_scalar); }
	ET_SIMD_INLINE void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; }
	ET_SIMD_INLINE Vector2<T> operator/(const Vector2<T> &p_v1) const { return Vector2<T>(x / p_v1.x, y / p_v1.y); }
	ET_SIMD_INLINE Vector2<T> operator/(const T &p_scalar) const { return Vector2<T>(x / p_scalar, y / p_scalar); }
	ET_SIMD_INLINE void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; }
	ET_SIMD_INLINE Vector2<T> operator-() const { return Vector2<T>(-x, -y); }

	ET_SIMD_INLINE bool operator==(const Vector2<T> &p_v) const { return x == p_v.x && y == p_v.y; }
	ET_SIMD_INLINE bool operator!=(const Vector2<T> &p_v) const { return x != p_v.x || y != p_v.y; }
	ET_SIMD_INLINE bool operator<(const Vector2<T> &p_v) const { return (x == p_v.x) ? (y < p_v.y) : (x < p_v.x); }

	// ------------------------------------------------------------------------
	// Geometric Math (Warp-Optimized)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T length() const { return Math::sqrt(x * x + y * y); }
	ET_SIMD_INLINE T length_squared() const { return x * x + y * y; }

	ET_SIMD_INLINE void normalize() {
		T l = length();
		if (l != T(0)) {
			x /= l;
			y /= l;
		}
	}

	ET_SIMD_INLINE Vector2<T> normalized() const {
		Vector2<T> v = *this;
		v.normalize();
		return v;
	}

	ET_SIMD_INLINE T dot(const Vector2<T> &p_other) const { return x * p_other.x + y * p_other.y; }
	ET_SIMD_INLINE T cross(const Vector2<T> &p_other) const { return x * p_other.y - y * p_other.x; }

	ET_SIMD_INLINE T distance_to(const Vector2<T> &p_vector2) const { return (p_vector2 - *this).length(); }
	ET_SIMD_INLINE T distance_squared_to(const Vector2<T> &p_vector2) const { return (p_vector2 - *this).length_squared(); }

	ET_SIMD_INLINE T angle() const { return Math::atan2(y, x); }

	ET_SIMD_INLINE Vector2<T> abs() const { return Vector2<T>(Math::abs(x), Math::abs(y)); }

	ET_SIMD_INLINE Vector2<T> rotated(T p_by) const {
		T sine = Math::sin(p_by);
		T cosi = Math::cos(p_by);
		return Vector2<T>(x * cosi - y * sine, x * sine + y * cosi);
	}

	ET_SIMD_INLINE Vector2<T> lerp(const Vector2<T> &p_to, T p_weight) const {
		return Vector2<T>(Math::lerp(x, p_to.x, p_weight), Math::lerp(y, p_to.y, p_weight));
	}

	ET_SIMD_INLINE Vector2<T> snapped(const Vector2<T> &p_step) const {
		return Vector2<T>(Math::snapped(x, p_step.x), Math::snapped(y, p_step.y));
	}

	// Conversion to Godot Native String for UI
	operator String() const { return "(" + String(x.to_string().c_str()) + ", " + String(y.to_string().c_str()) + ")"; }
};

// Unified Simulation Aliases
typedef Vector2<FixedMathCore> Vector2f; // Deterministic Physics/Rendering
typedef Vector2<BigIntCore> Vector2b;    // Macro-Scale/Infinite Grids

#endif // VECTOR2_H

--- END OF FILE core/math/vector2.h ---
