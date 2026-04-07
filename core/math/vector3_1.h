--- START OF FILE core/math/vector3.h ---

#ifndef VECTOR3_H
#define VECTOR3_H

#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Vector3 Template
 * 
 * 32-byte aligned 3D vector for deterministic spatial logic.
 * Engineered for zero-copy data flow in the Scale-Aware pipeline.
 */
template <typename T>
struct ET_ALIGN_32 Vector3 {
	T x, y, z;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3() : x(MathConstants<T>::zero()), y(MathConstants<T>::zero()), z(MathConstants<T>::zero()) {}
	_FORCE_INLINE_ Vector3(T p_x, T p_y, T p_z) : x(p_x), y(p_y), z(p_z) {}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3<T> operator+(const Vector3<T> &p_v) const { return Vector3<T>(x + p_v.x, y + p_v.y, z + p_v.z); }
	_FORCE_INLINE_ void operator+=(const Vector3<T> &p_v) { x += p_v.x; y += p_v.y; z += p_v.z; }
	_FORCE_INLINE_ Vector3<T> operator-(const Vector3<T> &p_v) const { return Vector3<T>(x - p_v.x, y - p_v.y, z - p_v.z); }
	_FORCE_INLINE_ void operator-=(const Vector3<T> &p_v) { x -= p_v.x; y -= p_v.y; z -= p_v.z; }

	_FORCE_INLINE_ Vector3<T> operator*(const Vector3<T> &p_v) const { return Vector3<T>(x * p_v.x, y * p_v.y, z * p_v.z); }
	_FORCE_INLINE_ Vector3<T> operator*(const T &p_scalar) const { return Vector3<T>(x * p_scalar, y * p_scalar, z * p_scalar); }
	_FORCE_INLINE_ void operator*=(const T &p_scalar) { x *= p_scalar; y *= p_scalar; z *= p_scalar; }

	_FORCE_INLINE_ Vector3<T> operator/(const Vector3<T> &p_v) const { return Vector3<T>(x / p_v.x, y / p_v.y, z / p_v.z); }
	_FORCE_INLINE_ Vector3<T> operator/(const T &p_scalar) const { return Vector3<T>(x / p_scalar, y / p_scalar, z / p_scalar); }
	_FORCE_INLINE_ void operator/=(const T &p_scalar) { x /= p_scalar; y /= p_scalar; z /= p_scalar; }

	_FORCE_INLINE_ Vector3<T> operator-() const { return Vector3<T>(-x, -y, -z); }

	_FORCE_INLINE_ bool operator==(const Vector3<T> &p_v) const { return x == p_v.x && y == p_v.y && z == p_v.z; }
	_FORCE_INLINE_ bool operator!=(const Vector3<T> &p_v) const { return x != p_v.x || y != p_v.y || z != p_v.z; }
	_FORCE_INLINE_ bool operator<(const Vector3<T> &p_v) const {
		if (x != p_v.x) return x < p_v.x;
		if (y != p_v.y) return y < p_v.y;
		return z < p_v.z;
	}

	_FORCE_INLINE_ T &operator[](int p_idx) { return (&x)[p_idx]; }
	_FORCE_INLINE_ const T &operator[](int p_idx) const { return (&x)[p_idx]; }

	// ------------------------------------------------------------------------
	// Geometric API (Warp-Kernel Ready)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T length() const { return Math::sqrt(x * x + y * y + z * z); }
	_FORCE_INLINE_ T length_squared() const { return x * x + y * y + z * z; }

	void normalize() {
		T l_sq = length_squared();
		if (l_sq != MathConstants<T>::zero()) {
			T l = Math::sqrt(l_sq);
			x /= l; y /= l; z /= l;
		}
	}

	_FORCE_INLINE_ Vector3<T> normalized() const {
		Vector3<T> v = *this;
		v.normalize();
		return v;
	}

	_FORCE_INLINE_ T dot(const Vector3<T> &p_v) const { return x * p_v.x + y * p_v.y + z * p_v.z; }
	
	_FORCE_INLINE_ Vector3<T> cross(const Vector3<T> &p_v) const {
		return Vector3<T>(
				(y * p_v.z) - (z * p_v.y),
				(z * p_v.x) - (x * p_v.z),
				(x * p_v.y) - (y * p_v.x));
	}

	_FORCE_INLINE_ T distance_to(const Vector3<T> &p_v) const { return (p_v - *this).length(); }
	_FORCE_INLINE_ T distance_squared_to(const Vector3<T> &p_v) const { return (p_v - *this).length_squared(); }

	// ------------------------------------------------------------------------
	// Sophisticated Physics Behaviors
	// ------------------------------------------------------------------------

	/**
	 * reflect()
	 * Returns the vector reflected across a plane defined by the normal.
	 * Used for bit-perfect collision and light-ray bounces.
	 */
	_FORCE_INLINE_ Vector3<T> reflect(const Vector3<T> &p_normal) const {
		return *this - p_normal * (this->dot(p_normal) * T(2LL));
	}

	/**
	 * bounce()
	 * Essential for Continuous Collision Detection (CCD) responses.
	 */
	_FORCE_INLINE_ Vector3<T> bounce(const Vector3<T> &p_normal) const {
		return -reflect(p_normal);
	}

	/**
	 * project()
	 * Projects this vector onto another. Used for mechanical link constraints.
	 */
	_FORCE_INLINE_ Vector3<T> project(const Vector3<T> &p_to) const {
		T den = p_to.length_squared();
		if (den == MathConstants<T>::zero()) return Vector3<T>();
		return p_to * (this->dot(p_to) / den);
	}

	/**
	 * slide()
	 * Used for smooth motion against walls and flesh deformation sliding.
	 */
	_FORCE_INLINE_ Vector3<T> slide(const Vector3<T> &p_normal) const {
		return *this - p_normal * this->dot(p_normal);
	}

	// ------------------------------------------------------------------------
	// Deterministic Interpolation & Transforms
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3<T> lerp(const Vector3<T> &p_to, T p_weight) const {
		return Vector3<T>(
				Math::lerp(x, p_to.x, p_weight),
				Math::lerp(y, p_to.y, p_weight),
				Math::lerp(z, p_to.z, p_weight));
	}

	_FORCE_INLINE_ Vector3<T> slerp(const Vector3<T> &p_to, T p_weight) const {
		T theta = Math::acos(CLAMP(this->normalized().dot(p_to.normalized()), -MathConstants<T>::one(), MathConstants<T>::one()));
		if (theta.get_raw() == 0) return *this;
		T sin_theta = Math::sin(theta);
		return (*this * (Math::sin((MathConstants<T>::one() - p_weight) * theta) / sin_theta) +
				p_to * (Math::sin(p_weight * theta) / sin_theta));
	}

	/**
	 * rotated()
	 * Bit-perfect rotation around an arbitrary axis.
	 * Optimized for 120 FPS robotic arm and spaceship stabilization.
	 */
	_FORCE_INLINE_ Vector3<T> rotated(const Vector3<T> &p_axis, T p_angle) const {
		Vector3<T> axis = p_axis.normalized();
		T s = Math::sin(p_angle);
		T c = Math::cos(p_angle);
		return axis * dot(axis) + (*this - axis * dot(axis)) * c + axis.cross(*this) * s;
	}

	_FORCE_INLINE_ Vector3<T> abs() const { return Vector3<T>(Math::abs(x), Math::abs(y), Math::abs(z)); }

	_FORCE_INLINE_ Vector3<T> snapped(const Vector3<T> &p_step) const {
		return Vector3<T>(Math::snapped(x, p_step.x), Math::snapped(y, p_step.y), Math::snapped(z, p_step.z));
	}

	/**
	 * is_relativistic()
	 * Checks if the magnitude of the vector (as velocity) requires 
	 * Lorentz-Correction within the Universal Solver.
	 */
	_FORCE_INLINE_ bool is_relativistic() const {
		if constexpr (std::is_same<T, FixedMathCore>::value) {
			// Threshold: 10% of Speed of Light
			return length_squared() > (PHYSICS_C * PHYSICS_C * FixedMathCore("0.01"));
		}
		return false;
	}

	operator String() const { 
		return "(" + String(x.to_string().c_str()) + ", " + 
		             String(y.to_string().c_str()) + ", " + 
		             String(z.to_string().c_str()) + ")"; 
	}
};

// Simulation Tier Typedefs
typedef Vector3<FixedMathCore> Vector3f; // Bit-perfect 3D Physics
typedef Vector3<BigIntCore> Vector3b;    // Discrete 3D Macro-Sector mapping

#endif // VECTOR3_H

--- END OF FILE core/math/vector3.h ---
