--- START OF FILE core/math/basis.h ---

#ifndef BASIS_H
#define BASIS_H

#include "core/math/vector3.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Basis Template
 * 
 * Represents a 3x3 matrix for rotation and scale. 
 * Integrated with the Universal Solver for bit-perfect spatial integrity.
 * Aligned to 32 bytes for high-performance EnTT SoA sweeps and Warp kernels.
 */
template <typename T>
struct ET_ALIGN_32 Basis {
	Vector3<T> rows[3];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Basis() {
		rows[0] = Vector3<T>(MathConstants<T>::one(), MathConstants<T>::zero(), MathConstants<T>::zero());
		rows[1] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero());
		rows[2] = Vector3<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());
	}

	ET_SIMD_INLINE Basis(const Vector3<T> &p_x, const Vector3<T> &p_y, const Vector3<T> &p_z) {
		rows[0] = p_x;
		rows[1] = p_y;
		rows[2] = p_z;
	}

	ET_SIMD_INLINE Basis(T p_xx, T p_xy, T p_xz, T p_yx, T p_yy, T p_yz, T p_zx, T p_zy, T p_zz) {
		rows[0] = Vector3<T>(p_xx, p_xy, p_xz);
		rows[1] = Vector3<T>(p_yx, p_yy, p_yz);
		rows[2] = Vector3<T>(p_zx, p_zy, p_zz);
	}

	// ------------------------------------------------------------------------
	// Accessors
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
	// Deterministic Euler Logic (Scale-Aware)
	// ------------------------------------------------------------------------
	void set_euler(const Vector3<T> &p_euler) {
		T c, s;

		c = Math::cos(p_euler.x);
		s = Math::sin(p_euler.x);
		Basis<T> xmat(MathConstants<T>::one(), MathConstants<T>::zero(), MathConstants<T>::zero(), 
		              MathConstants<T>::zero(), c, -s, 
		              MathConstants<T>::zero(), s, c);

		c = Math::cos(p_euler.y);
		s = Math::sin(p_euler.y);
		Basis<T> ymat(c, MathConstants<T>::zero(), s, 
		              MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero(), 
		              -s, MathConstants<T>::zero(), c);

		c = Math::cos(p_euler.z);
		s = Math::sin(p_euler.z);
		Basis<T> zmat(c, -s, MathConstants<T>::zero(), 
		              s, c, MathConstants<T>::zero(), 
		              MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());

		*this = ymat * xmat * zmat;
	}

	Vector3<T> get_euler() const {
		Basis<T> m = *this;
		m.orthonormalize();
		Vector3<T> euler;
		euler.y = Math::asin(CLAMP(m[0][2], -MathConstants<T>::one(), MathConstants<T>::one()));
		if (Math::abs(m[0][2]) < T(4294924294LL, true)) { // 0.9999
			euler.x = Math::atan2(-m[1][2], m[2][2]);
			euler.z = Math::atan2(-m[0][1], m[0][0]);
		} else {
			euler.x = Math::atan2(m[2][1], m[1][1]);
			euler.z = MathConstants<T>::zero();
		}
		return euler;
	}

	// ------------------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Basis<T> operator*(const Basis<T> &p_matrix) const {
		return Basis<T>(
			p_matrix.get_column(0).dot(rows[0]), p_matrix.get_column(1).dot(rows[0]), p_matrix.get_column(2).dot(rows[0]),
			p_matrix.get_column(0).dot(rows[1]), p_matrix.get_column(1).dot(rows[1]), p_matrix.get_column(2).dot(rows[1]),
			p_matrix.get_column(0).dot(rows[2]), p_matrix.get_column(1).dot(rows[2]), p_matrix.get_column(2).dot(rows[2])
		);
	}

	ET_SIMD_INLINE Vector3<T> xform(const Vector3<T> &p_v) const {
		return Vector3<T>(rows[0].dot(p_v), rows[1].dot(p_v), rows[2].dot(p_v));
	}

	ET_SIMD_INLINE void operator*=(const Basis<T> &p_matrix) { *this = *this * p_matrix; }

	// ------------------------------------------------------------------------
	// Transformation API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE T determinant() const {
		return rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) -
		       rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0]) +
		       rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
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

	ET_SIMD_INLINE Basis<T> inverse() const {
		T det = determinant();
		if (det == MathConstants<T>::zero()) return Basis<T>();
		T inv_det = MathConstants<T>::one() / det;
		return Basis<T>(
			(rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) * inv_det,
			(rows[0][2] * rows[2][1] - rows[0][1] * rows[2][2]) * inv_det,
			(rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1]) * inv_det,
			(rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2]) * inv_det,
			(rows[0][0] * rows[2][2] - rows[0][2] * rows[2][0]) * inv_det,
			(rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2]) * inv_det,
			(rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]) * inv_det,
			(rows[0][1] * rows[2][0] - rows[0][0] * rows[2][1]) * inv_det,
			(rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) * inv_det
		);
	}

	ET_SIMD_INLINE Basis<T> scaled(const Vector3<T> &p_scale) const {
		Basis<T> b = *this;
		b.rows[0] *= p_scale.x;
		b.rows[1] *= p_scale.y;
		b.rows[2] *= p_scale.z;
		return b;
	}

	ET_SIMD_INLINE bool is_equal_approx(const Basis<T> &p_basis) const {
		return rows[0].is_equal_approx(p_basis.rows[0]) &&
		       rows[1].is_equal_approx(p_basis.rows[1]) &&
		       rows[2].is_equal_approx(p_basis.rows[2]);
	}

	operator String() const {
		return String(rows[0]) + ", " + String(rows[1]) + ", " + String(rows[2]);
	}
};

typedef Basis<FixedMathCore> Basisf;
typedef Basis<BigIntCore> Basisb;

#endif // BASIS_H

--- END OF FILE core/math/basis.h ---
