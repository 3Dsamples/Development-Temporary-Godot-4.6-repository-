--- START OF FILE core/math/transform_2d.h ---

#ifndef TRANSFORM_2D_H
#define TRANSFORM_2D_H

#include "core/math/vector2.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Transform2D Template
 * 
 * A 2x3 matrix for 2D transformations.
 * Aligned to 32 bytes for zero-copy EnTT integration and high-speed Warp sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Transform2D {
	// Column 0: X Basis
	// Column 1: Y Basis
	// Column 2: Origin
	Vector2<T> columns[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Transform2D() {
		columns[0] = Vector2<T>(MathConstants<T>::one(), MathConstants<T>::zero());
		columns[1] = Vector2<T>(MathConstants<T>::zero(), MathConstants<T>::one());
		columns[2] = Vector2<T>(MathConstants<T>::zero(), MathConstants<T>::zero());
	}

	ET_SIMD_INLINE Transform2D(T p_xx, T p_xy, T p_yx, T p_yy, T p_ox, T p_oy) {
		columns[0] = Vector2<T>(p_xx, p_xy);
		columns[1] = Vector2<T>(p_yx, p_yy);
		columns[2] = Vector2<T>(p_ox, p_oy);
	}

	ET_SIMD_INLINE Transform2D(const Vector2<T> &p_x, const Vector2<T> &p_y, const Vector2<T> &p_origin) {
		columns[0] = p_x;
		columns[1] = p_y;
		columns[2] = p_origin;
	}

	// ------------------------------------------------------------------------
	// Deterministic Transformation API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector2<T> basis_xform(const Vector2<T> &p_v) const {
		return Vector2<T>(
				columns[0].x * p_v.x + columns[1].x * p_v.y,
				columns[0].y * p_v.x + columns[1].y * p_v.y);
	}

	ET_SIMD_INLINE Vector2<T> basis_xform_inv(const Vector2<T> &p_v) const {
		return Vector2<T>(columns[0].dot(p_v), columns[1].dot(p_v));
	}

	ET_SIMD_INLINE Vector2<T> xform(const Vector2<T> &p_v) const {
		return Vector2<T>(
				columns[0].x * p_v.x + columns[1].x * p_v.y + columns[2].x,
				columns[0].y * p_v.x + columns[1].y * p_v.y + columns[2].y);
	}

	ET_SIMD_INLINE Vector2<T> xform_inv(const Vector2<T> &p_v) const {
		Vector2<T> v = p_v - columns[2];
		return Vector2<T>(columns[0].dot(v), columns[1].dot(v));
	}

	// ------------------------------------------------------------------------
	// Operators (Batch Math Optimized)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE void operator*=(const Transform2D<T> &p_other) {
		columns[2] = xform(p_other.columns[2]);
		T x0 = columns[0].x * p_other.columns[0].x + columns[1].x * p_other.columns[0].y;
		T y0 = columns[0].y * p_other.columns[0].x + columns[1].y * p_other.columns[0].y;
		T x1 = columns[0].x * p_other.columns[1].x + columns[1].x * p_other.columns[1].y;
		T y1 = columns[0].y * p_other.columns[1].x + columns[1].y * p_other.columns[1].y;
		columns[0].x = x0;
		columns[0].y = y0;
		columns[1].x = x1;
		columns[1].y = y1;
	}

	ET_SIMD_INLINE Transform2D<T> operator*(const Transform2D<T> &p_other) const {
		Transform2D<T> res = *this;
		res *= p_other;
		return res;
	}

	ET_SIMD_INLINE bool operator==(const Transform2D<T> &p_other) const {
		return columns[0] == p_other.columns[0] && columns[1] == p_other.columns[1] && columns[2] == p_other.columns[2];
	}

	// ------------------------------------------------------------------------
	// Specialized Logic (60 FPS Performance)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T determinant() const {
		return columns[0].x * columns[1].y - columns[0].y * columns[1].x;
	}

	void affine_invert() {
		T det = determinant();
		if (det == MathConstants<T>::zero()) {
			*this = Transform2D<T>();
			return;
		}
		T idet = MathConstants<T>::one() / det;
		T tmp = columns[0].x;
		columns[0].x = columns[1].y * idet;
		columns[1].y = tmp * idet;
		columns[0].y *= -idet;
		columns[1].x *= -idet;
		columns[2] = basis_xform(-columns[2]);
	}

	ET_SIMD_INLINE Transform2D<T> affine_inverted() const {
		Transform2D<T> res = *this;
		res.affine_invert();
		return res;
	}

	void rotate(T p_angle) {
		T s = Math::sin(p_angle);
		T c = Math::cos(p_angle);
		Transform2D<T> rot(c, s, -s, c, MathConstants<T>::zero(), MathConstants<T>::zero());
		*this = (*this) * rot;
	}

	void scale(const Vector2<T> &p_scale) {
		columns[0] *= p_scale.x;
		columns[1] *= p_scale.y;
	}

	void orthonormalize() {
		columns[0].normalize();
		columns[1] = (columns[1] - columns[0] * columns[0].dot(columns[1]));
		columns[1].normalize();
	}

	ET_SIMD_INLINE T get_rotation() const {
		return Math::atan2(columns[0].y, columns[0].x);
	}

	// Accessors for Warp Kernels
	ET_SIMD_INLINE Vector2<T> &operator[](int p_idx) { return columns[p_idx]; }
	ET_SIMD_INLINE const Vector2<T> &operator[](int p_idx) const { return columns[p_idx]; }

	// Godot UI Conversion
	operator String() const {
		return "[X: " + (String)columns[0] + ", Y: " + (String)columns[1] + ", O: " + (String)columns[2] + "]";
	}
};

typedef Transform2D<FixedMathCore> Transform2Df;
typedef Transform2D<BigIntCore> Transform2Db;

#endif // TRANSFORM_2D_H

--- END OF FILE core/math/transform_2d.h ---
