--- START OF FILE core/math/matrix3.h ---

#ifndef MATRIX3_H
#define MATRIX3_H

#include "core/math/vector3.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Matrix3 Template
 * 
 * A 3x3 matrix optimized for deterministic rotations and basis transformations.
 * Aligned to 32 bytes for zero-copy EnTT integration and SIMD Warp kernel sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Matrix3 {
	Vector3<T> rows[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Matrix3() {
		rows[0] = Vector3<T>(MathConstants<T>::one(), MathConstants<T>::zero(), MathConstants<T>::zero());
		rows[1] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero());
		rows[2] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());
	}

	ET_SIMD_INLINE Matrix3(const Vector3<T> &p_row0, const Vector3<T> &p_row1, const Vector3<T> &p_row2) {
		rows[0] = p_row0;
		rows[1] = p_row1;
		rows[2] = p_row2;
	}

	// ------------------------------------------------------------------------
	// Accessors (Warp Kernel Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE const Vector3<T> &operator[](int p_row) const { return rows[p_row]; }
	ET_SIMD_INLINE Vector3<T> &operator[](int p_row) { return rows[p_row]; }

	ET_SIMD_INLINE Vector3<T> get_column(int p_idx) const {
		return Vector3<T>(rows[0][p_idx], rows[1][p_idx], rows[2][p_idx]);
	}

	ET_SIMD_INLINE void set_column(int p_idx, const Vector3<T> &p_v) {
		rows[0][p_idx] = p_v.x;
		rows[1][p_idx] = p_v.y;
		rows[2][p_idx] = p_v.z;
	}

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Matrix3<T> operator*(const Matrix3<T> &p_m) const {
		return Matrix3<T>(
				p_m.get_column(0).dot(rows[0]), p_m.get_column(1).dot(rows[0]), p_m.get_column(2).dot(rows[0]),
				p_m.get_column(0).dot(rows[1]), p_m.get_column(1).dot(rows[1]), p_m.get_column(2).dot(rows[1]),
				p_m.get_column(0).dot(rows[2]), p_m.get_column(1).dot(rows[2]), p_m.get_column(2).dot(rows[2]));
	}

	ET_SIMD_INLINE Vector3<T> operator*(const Vector3<T> &p_v) const {
		return Vector3<T>(rows[0].dot(p_v), rows[1].dot(p_v), rows[2].dot(p_v));
	}

	ET_SIMD_INLINE void operator*=(const Matrix3<T> &p_m) { *this = *this * p_m; }

	// ------------------------------------------------------------------------
	// Matrix Math Functions
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T determinant() const {
		return rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) -
				rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0]) +
				rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
	}

	ET_SIMD_INLINE Matrix3<T> transpose() const {
		return Matrix3<T>(
				rows[0][0], rows[1][0], rows[2][0],
				rows[0][1], rows[1][1], rows[2][1],
				rows[0][2], rows[1][2], rows[2][2]);
	}

	Matrix3<T> inverse() const {
		T det = determinant();
		if (det == MathConstants<T>::zero()) return Matrix3<T>();
		T inv_det = MathConstants<T>::one() / det;
		Matrix3<T> m;
		m[0][0] = (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) * inv_det;
		m[0][1] = (rows[0][2] * rows[2][1] - rows[0][1] * rows[2][2]) * inv_det;
		m[0][2] = (rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1]) * inv_det;
		m[1][0] = (rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2]) * inv_det;
		m[1][1] = (rows[0][0] * rows[2][2] - rows[0][2] * rows[2][0]) * inv_det;
		m[1][2] = (rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2]) * inv_det;
		m[2][0] = (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]) * inv_det;
		m[2][1] = (rows[0][1] * rows[2][0] - rows[0][0] * rows[2][1]) * inv_det;
		m[2][2] = (rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) * inv_det;
		return m;
	}

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

	// UI/Debug Conversion
	operator String() const {
		return String(rows[0]) + ", " + String(rows[1]) + ", " + String(rows[2]);
	}
};

// Simulation Aliases
typedef Matrix3<FixedMathCore> Matrix3f; // Deterministic Physics
typedef Matrix3<BigIntCore> Matrix3b;    // Macro Transformations

#endif // MATRIX3_H

--- END OF FILE core/math/matrix3.h ---
