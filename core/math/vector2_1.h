--- START OF FILE core/math/vector2.h ---

#ifndef VECTOR2_H
#define VECTOR2_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector2 Template
 * 
 * 32-byte aligned 2D vector for bit-perfect spatial logic.
 * Replaces standard floating-point vectors to eliminate rounding drift.
 */
template <typename T>
struct ET_ALIGN_32 Vector2 {
	T x, y;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2() : x(MathConstants<T>::zero()), y(MathConstants<T>::zero()) {}
	_FORCE_INLINE_ Vector2(T p_x, T p_y) : x(p_x), y(p_y) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic & SIMD Friendly)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2<T> operator+(const Vector2<T> &p_v) const { return Vector2<T>(x + p_v.x, y + p_v.y); }
	_FORCE_INLINE_ void operator+=(const Vector2<T> &p_v) { x += p_v.x; y += p_v.y; }
	_FORCE_INLINE_ Vector2<T> operator-(const Vector2<T> &p_v) const { return Vector2<T>(x - p_v.x, y - p_v.y); }
	_FORCE_INLINE_ void operator-=(const Vector2<T> &p_v) { x -= p_v.x; y -= p_v.y; }

	_FORCE_INLINE_ Vector2<T> operator*(const Vector2<T> &p_v1) const { return Vector2<T>(x * p_v1.x, y * p_v1.y); }
	_FORCE_INLINE_ Vector2<T> operator*(const T &p_scalar) const { return Vector2<T>(x * p_scalar, y * p_scalar); }
	_FORCE_INLINE_ void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; }

	_FORCE_INLINE_ Vector2<T> operator/(const Vector2<T> &p_v1) const { return Vector2<T>(x / p_v1.x, y / p_v1.y); }
	_FORCE_INLINE_ Vector2<T> operator/(const T &p_scalar) const { return Vector2<T>(x / p_scalar, y / p_scalar); }
	_FORCE_INLINE_ void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; }

	_FORCE_INLINE_ Vector2<T> operator-() const { return Vector2<T>(-x, -y); }

	_FORCE_INLINE_ bool operator==(const Vector2<T> &p_v) const { return x == p_v.x && y == p_v.y; }
	_FORCE_INLINE_ bool operator!=(const Vector2<T> &p_v) const { return x != p_v.x || y != p_v.y; }
	_FORCE_INLINE_ bool operator<(const Vector2<T> &p_v) const { return (x == p_v.x) ? (y < p_v.y) : (x < p_v.x); }

	_FORCE_INLINE_ T &operator[](int p_idx) { return p_idx ? y : x; }
	_FORCE_INLINE_ const T &operator[](int p_idx) const { return p_idx ? y : x; }

	// ------------------------------------------------------------------------
	// Geometric API (Bit-Perfect)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T length() const { return Math::sqrt(x * x + y * y); }
	_FORCE_INLINE_ T length_squared() const { return x * x + y * y; }

	void normalize() {
		T l = length();
		if (l != MathConstants<T>::zero()) {
			x /= l;
			y /= l;
		}
	}

	_FORCE_INLINE_ Vector2<T> normalized() const {
		Vector2<T> v = *this;
		v.normalize();
		return v;
	}

	_FORCE_INLINE_ T dot(const Vector2<T> &p_other) const { return x * p_other.x + y * p_other.y; }
	_FORCE_INLINE_ T cross(const Vector2<T> &p_other) const { return x * p_other.y - y * p_other.x; }

	_FORCE_INLINE_ T angle() const { return Math::atan2(y, x); }
	_FORCE_INLINE_ T distance_to(const Vector2<T> &p_vector2) const { return (p_vector2 - *this).length(); }
	_FORCE_INLINE_ T distance_squared_to(const Vector2<T> &p_vector2) const { return (p_vector2 - *this).length_squared(); }

	// ------------------------------------------------------------------------
	// Sophisticated Physics Behaviors
	// ------------------------------------------------------------------------

	/**
	 * reflect()
	 * Returns the vector reflected across a plane defined by the normal.
	 * Essential for bit-perfect collision response in 120 FPS simulations.
	 */
	_FORCE_INLINE_ Vector2<T> reflect(const Vector2<T> &p_normal) const {
		return *this - p_normal * (this->dot(p_normal) * T(2LL));
	}

	/**
	 * bounce()
	 * Physics response for bouncing off a surface.
	 */
	_FORCE_INLINE_ Vector2<T> bounce(const Vector2<T> &p_normal) const {
		return -reflect(p_normal);
	}

	/**
	 * project()
	 * Returns the component of this vector projected onto another.
	 */
	_FORCE_INLINE_ Vector2<T> project(const Vector2<T> &p_to) const {
		T den = p_to.length_squared();
		if (den == MathConstants<T>::zero()) return Vector2<T>();
		return p_to * (this->dot(p_to) / den);
	}

	/**
	 * slide()
	 * Returns the component of this vector sliding along a surface normal.
	 * Used for smooth robotic movement and flesh deformation sliding.
	 */
	_FORCE_INLINE_ Vector2<T> slide(const Vector2<T> &p_normal) const {
		return *this - p_normal * this->dot(p_normal);
	}

	// ------------------------------------------------------------------------
	// Transformation & Interpolation
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2<T> lerp(const Vector2<T> &p_to, T p_weight) const {
		return Vector2<T>(Math::lerp(x, p_to.x, p_weight), Math::lerp(y, p_to.y, p_weight));
	}

	_FORCE_INLINE_ Vector2<T> slerp(const Vector2<T> &p_to, T p_weight) const {
		T start_l = length();
		T end_l = p_to.length();
		T start_a = angle();
		T end_a = p_to.angle();
		T res_l = Math::lerp(start_l, end_l, p_weight);
		T res_a = Math::lerp(start_a, end_a, p_weight);
		return Vector2<T>(Math::cos(res_a) * res_l, Math::sin(res_a) * res_l);
	}

	_FORCE_INLINE_ Vector2<T> rotated(T p_by) const {
		T sine = Math::sin(p_by);
		T cosi = Math::cos(p_by);
		return Vector2<T>(x * cosi - y * sine, x * sine + y * cosi);
	}

	_FORCE_INLINE_ Vector2<T> abs() const { return Vector2<T>(Math::abs(x), Math::abs(y)); }

	_FORCE_INLINE_ Vector2<T> snapped(const Vector2<T> &p_step) const {
		return Vector2<T>(Math::snapped(x, p_step.x), Math::snapped(y, p_step.y));
	}

	_FORCE_INLINE_ T aspect() const {
		if (y == MathConstants<T>::zero()) return MathConstants<T>::zero();
		return x / y;
	}

	operator String() const { return "(" + String(x.to_string().c_str()) + ", " + String(y.to_string().c_str()) + ")"; }
};

// Simulation Tier Typedefs
typedef Vector2<FixedMathCore> Vector2f; // Bit-perfect 2D Physics
typedef Vector2<BigIntCore> Vector2b;    // Discrete Macro-Grid Logic

#endif // VECTOR2_H

--- END OF FILE core/math/vector2.h ---
