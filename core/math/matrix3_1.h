--- START OF FILE core/math/matrix3.h ---

#ifndef MATRIX3_H
#define MATRIX3_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Matrix3 Template
 * 
 * 32-byte aligned 3x3 matrix for deterministic rotations and scaling.
 * Replaces standard floating-point matrices to guarantee architectural coherence.
 */
template <typename T>
struct ET_ALIGN_32 Matrix3 {
	Vector3<T> rows[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Matrix3() {
		rows[0] = Vector3<T>(MathConstants<T>::one(), MathConstants<T>::zero(), MathConstants<T>::zero());
		rows[1] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero());
		rows[2] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());
	}

	_FORCE_INLINE_ Matrix3(const Vector3<T> &p_row0, const Vector3<T> &p_row1, const Vector3<T> &p_row2) {
		rows[0] = p_row0;
		rows[1] = p_row1;
		rows[2] = p_row2;
	}

	_FORCE_INLINE_ Matrix3(T p_xx, T p_xy, T p_xz, T p_yx, T p_yy, T p_yz, T p_zx, T p_zy, T p_zz) {
		rows[0] = Vector3<T>(p_xx, p_xy, p_xz);
		rows[1] = Vector3<T>(p_yx, p_yy, p_yz);
		rows[2] = Vector3<T>(p_zx, p_zy, p_zz);
	}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3<T> &operator[](int p_idx) { return rows[p_idx]; }
	_FORCE_INLINE_ const Vector3<T> &operator[](int p_idx) const { return rows[p_idx]; }

	_FORCE_INLINE_ Matrix3<T> operator*(const Matrix3<T> &p_m) const {
		return Matrix3<T>(
				p_m.get_column(0).dot(rows[0]), p_m.get_column(1).dot(rows[0]), p_m.get_column(2).dot(rows[0]),
				p_m.get_column(0).dot(rows[1]), p_m.get_column(1).dot(rows[1]), p_m.get_column(2).dot(rows[1]),
				p_m.get_column(0).dot(rows[2]), p_m.get_column(1).dot(rows[2]), p_m.get_column(2).dot(rows[2]));
	}

	_FORCE_INLINE_ void operator*=(const Matrix3<T> &p_m) { *this = *this * p_m; }

	_FORCE_INLINE_ Vector3<T> operator*(const Vector3<T> &p_v) const {
		return Vector3<T>(rows[0].dot(p_v), rows[1].dot(p_v), rows[2].dot(p_v));
	}

	_FORCE_INLINE_ bool operator==(const Matrix3<T> &p_m) const {
		return rows[0] == p_m.rows[0] && rows[1] == p_m.rows[1] && rows[2] == p_m.rows[2];
	}

	_FORCE_INLINE_ bool operator!=(const Matrix3<T> &p_m) const { return !(*this == p_m); }

	// ------------------------------------------------------------------------
	// Matrix API (Warp-Kernel Ready)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector3<T> get_column(int p_idx) const {
		return Vector3<T>(rows[0][p_idx], rows[1][p_idx], rows[2][p_idx]);
	}

	_FORCE_INLINE_ void set_column(int p_idx, const Vector3<T> &p_v) {
		rows[0][p_idx] = p_v.x;
		rows[1][p_idx] = p_v.y;
		rows[2][p_idx] = p_v.z;
	}

	_FORCE_INLINE_ Matrix3<T> transpose() const {
		return Matrix3<T>(
				rows[0].x, rows[1].x, rows[2].x,
				rows[0].y, rows[1].y, rows[2].y,
				rows[0].z, rows[1].z, rows[2].z);
	}

	_FORCE_INLINE_ T determinant() const {
		return rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y) -
				rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x) +
				rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x);
	}

	Matrix3<T> inverse() const;

	/**
	 * orthonormalize()
	 * Bit-perfect Gram-Schmidt process.
	 * Essential for maintaining stable rotations in 120 FPS rigid-body loops.
	 */
	void orthonormalize() {
		Vector3<T> x = get_column(0);
		Vector3<T> y = get_column(1);
		Vector3<T> z = get_column(2);

		x.normalize();
		y = (y - x * x.dot(y));
		y.normalize();
		z = (z - x * x.dot(z) - y * y.dot(z));
		z.normalize();

		set_column(0, x);
		set_column(1, y);
		set_column(2, z);
	}

	// ------------------------------------------------------------------------
	// Rotation API (FixedMath Only)
	// ------------------------------------------------------------------------
	void set_euler(const Vector3<T> &p_euler);
	Vector3<T> get_euler() const;

	/**
	 * rotate()
	 * In-place rotation around an arbitrary axis using deterministic sin/cos.
	 */
	_FORCE_INLINE_ void rotate(const Vector3<T> &p_axis, T p_angle) {
		*this = Matrix3<T>(p_axis, p_angle) * (*this);
	}

	Matrix3(const Vector3<T> &p_axis, T p_angle);

	operator String() const {
		return "(" + String(rows[0]) + ", " + String(rows[1]) + ", " + String(rows[2]) + ")";
	}
};

// Simulation Tier Typedefs
typedef Matrix3<FixedMathCore> Matrix3f; // Bit-perfect 3D Rotations/Scaling
typedef Matrix3<BigIntCore> Matrix3b;    // Discrete Macro-Basis Transformations

#endif // MATRIX3_H

--- END OF FILE core/math/matrix3.h ---
