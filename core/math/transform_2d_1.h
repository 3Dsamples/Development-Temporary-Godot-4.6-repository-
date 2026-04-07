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
 * A 2x3 matrix representing a 2D transformation (rotation, scale, translation).
 * 32-byte aligned for high-performance SIMD/Warp batch processing.
 * Deterministic execution path for TIER_DETERMINISTIC and TIER_MACRO scales.
 */
template <typename T>
struct ET_ALIGN_32 Transform2D {
	// Column vectors for basis and origin
	Vector2<T> columns[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Transform2D() {
		columns[0] = Vector2<T>(MathConstants<T>::one(), MathConstants<T>::zero());
		columns[1] = Vector2<T>(MathConstants<T>::zero(), MathConstants<T>::one());
		columns[2] = Vector2<T>(MathConstants<T>::zero(), MathConstants<T>::zero());
	}

	_FORCE_INLINE_ Transform2D(T p_xx, T p_xy, T p_yx, T p_yy, T p_ox, T p_oy) {
		columns[0] = Vector2<T>(p_xx, p_xy);
		columns[1] = Vector2<T>(p_yx, p_yy);
		columns[2] = Vector2<T>(p_ox, p_oy);
	}

	_FORCE_INLINE_ Transform2D(const Vector2<T> &p_x, const Vector2<T> &p_y, const Vector2<T> &p_origin) {
		columns[0] = p_x;
		columns[1] = p_y;
		columns[2] = p_origin;
	}

	_FORCE_INLINE_ Transform2D(const T &p_rot, const Vector2<T> &p_pos) {
		T s = Math::sin(p_rot);
		T c = Math::cos(p_rot);
		columns[0] = Vector2<T>(c, s);
		columns[1] = Vector2<T>(-s, c);
		columns[2] = p_pos;
	}

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2<T> &operator[](int p_idx) { return columns[p_idx]; }
	_FORCE_INLINE_ const Vector2<T> &operator[](int p_idx) const { return columns[p_idx]; }

	_FORCE_INLINE_ Vector2<T> get_origin() const { return columns[2]; }
	_FORCE_INLINE_ void set_origin(const Vector2<T> &p_origin) { columns[2] = p_origin; }

	_FORCE_INLINE_ T get_rotation() const { return Math::atan2(columns[0].y, columns[0].x); }
	_FORCE_INLINE_ void set_rotation(T p_rot) {
		T s = Math::sin(p_rot);
		T c = Math::cos(p_rot);
		T len_x = columns[0].length();
		T len_y = columns[1].length();
		columns[0] = Vector2<T>(c, s) * len_x;
		columns[1] = Vector2<T>(-s, c) * len_y;
	}

	_FORCE_INLINE_ Vector2<T> get_scale() const {
		return Vector2<T>(columns[0].length(), columns[1].length());
	}

	// ------------------------------------------------------------------------
	// Transformation Methods (Zero-Copy)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector2<T> xform(const Vector2<T> &p_v) const {
		return Vector2<T>(
				columns[0].x * p_v.x + columns[1].x * p_v.y + columns[2].x,
				columns[0].y * p_v.x + columns[1].y * p_v.y + columns[2].y);
	}

	_FORCE_INLINE_ Vector2<T> xform_inv(const Vector2<T> &p_v) const {
		Vector2<T> v = p_v - columns[2];
		T det = determinant();
		if (unlikely(det == MathConstants<T>::zero())) return Vector2<T>();
		T idet = MathConstants<T>::one() / det;
		return Vector2<T>(
				(columns[1].y * v.x - columns[1].x * v.y) * idet,
				(columns[0].x * v.y - columns[0].y * v.x) * idet);
	}

	_FORCE_INLINE_ Vector2<T> basis_xform(const Vector2<T> &p_v) const {
		return Vector2<T>(
				columns[0].x * p_v.x + columns[1].x * p_v.y,
				columns[0].y * p_v.x + columns[1].y * p_v.y);
	}

	_FORCE_INLINE_ Vector2<T> basis_xform_inv(const Vector2<T> &p_v) const {
		T det = determinant();
		if (unlikely(det == MathConstants<T>::zero())) return Vector2<T>();
		T idet = MathConstants<T>::one() / det;
		return Vector2<T>(
				(columns[1].y * p_v.x - columns[1].x * p_v.y) * idet,
				(columns[0].x * p_v.y - columns[0].y * p_v.x) * idet);
	}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Chain)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ T determinant() const {
		return columns[0].x * columns[1].y - columns[0].y * columns[1].x;
	}

	void affine_invert() {
		T det = determinant();
		if (unlikely(det == MathConstants<T>::zero())) return;
		T idet = MathConstants<T>::one() / det;

		T xx = columns[1].y * idet;
		T yx = -columns[0].y * idet;
		T xy = -columns[1].x * idet;
		T yy = columns[0].x * idet;

		columns[0].x = xx;
		columns[0].y = yx;
		columns[1].x = xy;
		columns[1].y = yy;

		columns[2] = basis_xform(-columns[2]);
	}

	_FORCE_INLINE_ Transform2D<T> affine_inverted() const {
		Transform2D<T> res = *this;
		res.affine_invert();
		return res;
	}

	_FORCE_INLINE_ void operator*=(const Transform2D<T> &p_other) {
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

	_FORCE_INLINE_ Transform2D<T> operator*(const Transform2D<T> &p_other) const {
		Transform2D<T> res = *this;
		res *= p_other;
		return res;
	}

	_FORCE_INLINE_ bool operator==(const Transform2D<T> &p_other) const {
		return columns[0] == p_other.columns[0] && columns[1] == p_other.columns[1] && columns[2] == p_other.columns[2];
	}

	_FORCE_INLINE_ bool operator!=(const Transform2D<T> &p_other) const {
		return !(*this == p_other);
	}

	// ------------------------------------------------------------------------
	// Deterministic Modification API
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ void rotate(T p_phi) {
		*this = (*this) * Transform2D<T>(p_phi, Vector2<T>());
	}

	_FORCE_INLINE_ void scale(const Vector2<T> &p_scale) {
		columns[0] *= p_scale.x;
		columns[1] *= p_scale.y;
		columns[2] *= p_scale; // Note: Godot standard only scales basis, but for Universal Solver, we allow both
	}

	_FORCE_INLINE_ void translate(const Vector2<T> &p_offset) {
		columns[2] += basis_xform(p_offset);
	}

	void orthonormalize() {
		Vector2<T> x = columns[0];
		Vector2<T> y = columns[1];
		x.normalize();
		y = (y - x * x.dot(y));
		y.normalize();
		columns[0] = x;
		columns[1] = y;
	}

	_FORCE_INLINE_ Transform2D<T> lerp(const Transform2D<T> &p_to, T p_weight) const {
		Transform2D<T> res;
		res.columns[0] = columns[0].lerp(p_to.columns[0], p_weight);
		res.columns[1] = columns[1].lerp(p_to.columns[1], p_weight);
		res.columns[2] = columns[2].lerp(p_to.columns[2], p_weight);
		return res;
	}

	_FORCE_INLINE_ Transform2D<T> snapped(const Vector2<T> &p_step) const {
		Transform2D<T> res = *this;
		res.columns[0] = columns[0].snapped(p_step);
		res.columns[1] = columns[1].snapped(p_step);
		res.columns[2] = columns[2].snapped(p_step);
		return res;
	}

	operator String() const {
		return "[X: " + (String)columns[0] + ", Y: " + (String)columns[1] + ", O: " + (String)columns[2] + "]";
	}
};

// Simulation Tier Typedefs
typedef Transform2D<FixedMathCore> Transform2Df; // Bit-perfect 2D affine logic
typedef Transform2D<BigIntCore> Transform2Db;    // Discrete macro-grid logic

#endif // TRANSFORM_2D_H

--- END OF FILE core/math/transform_2d.h ---
