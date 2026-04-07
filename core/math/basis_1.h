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
 * A 3x3 matrix used for 3D rotation and scale.
 * Strictly deterministic and aligned to 32 bytes for SIMD-accelerated 
 * Warp kernels. Part of the Universal Solver's Scale-Aware pipeline.
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
	// Operators (Deterministic & Batch-Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> &operator[](int p_idx) { return rows[p_idx]; }
	ET_SIMD_INLINE const Vector3<T> &operator[](int p_idx) const { return rows[p_idx]; }

	ET_SIMD_INLINE Basis<T> operator*(const Basis<T> &p_matrix) const {
		return Basis<T>(
				p_matrix.get_column(0).dot(rows[0]), p_matrix.get_column(1).dot(rows[0]), p_matrix.get_column(2).dot(rows[0]),
				p_matrix.get_column(0).dot(rows[1]), p_matrix.get_column(1).dot(rows[1]), p_matrix.get_column(2).dot(rows[1]),
				p_matrix.get_column(0).dot(rows[2]), p_matrix.get_column(1).dot(rows[2]), p_matrix.get_column(2).dot(rows[2]));
	}

	ET_SIMD_INLINE void operator*=(const Basis<T> &p_matrix) { *this = *this * p_matrix; }

	ET_SIMD_INLINE Vector3<T> xform(const Vector3<T> &p_v) const {
		return Vector3<T>(rows[0].dot(p_v), rows[1].dot(p_v), rows[2].dot(p_v));
	}

	ET_SIMD_INLINE Vector3<T> xform_inv(const Vector3<T> &p_v) const {
		return Vector3<T>(
				rows[0][0] * p_v.x + rows[1][0] * p_v.y + rows[2][0] * p_v.z,
				rows[0][1] * p_v.x + rows[1][1] * p_v.y + rows[2][1] * p_v.z,
				rows[0][2] * p_v.x + rows[1][2] * p_v.y + rows[2][2] * p_v.z);
	}

	ET_SIMD_INLINE bool operator==(const Basis<T> &p_matrix) const {
		return rows[0] == p_matrix.rows[0] && rows[1] == p_matrix.rows[1] && rows[2] == p_matrix.rows[2];
	}

	ET_SIMD_INLINE bool operator!=(const Basis<T> &p_matrix) const { return !(*this == p_matrix); }

	// ------------------------------------------------------------------------
	// Deterministic Transformation API
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> get_column(int p_idx) const {
		return Vector3<T>(rows[0][p_idx], rows[1][p_idx], rows[2][p_idx]);
	}

	ET_SIMD_INLINE void set_column(int p_idx, const Vector3<T> &p_v) {
		rows[0][p_idx] = p_v.x;
		rows[1][p_idx] = p_v.y;
		rows[2][p_idx] = p_v.z;
	}

	ET_SIMD_INLINE Vector3<T> get_main_diagonal() const {
		return Vector3<T>(rows[0][0], rows[1][1], rows[2][2]);
	}

	ET_SIMD_INLINE T determinant() const {
		return rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y) -
				rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x) +
				rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x);
	}

	ET_SIMD_INLINE Basis<T> transpose() const {
		return Basis<T>(
				rows[0].x, rows[1].x, rows[2].x,
				rows[0].y, rows[1].y, rows[2].y,
				rows[0].z, rows[1].z, rows[2].z);
	}

	Basis<T> inverse() const;

	/**
	 * orthonormalize()
	 * Bit-perfect Gram-Schmidt re-projection.
	 * Essential for preventing rotational drift in long-running robotic or 
	 * orbital simulations at 120 FPS.
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
	// Sophisticated Scaling & Euler behaviors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector3<T> get_scale() const {
		return Vector3<T>(
				get_column(0).length(),
				get_column(1).length(),
				get_column(2).length());
	}

	void set_euler(const Vector3<T> &p_euler);
	Vector3<T> get_euler() const;

	/**
	 * scaled()
	 * Returns a basis with local scale applied deterministically.
	 */
	ET_SIMD_INLINE Basis<T> scaled(const Vector3<T> &p_scale) const {
		Basis<T> b = *this;
		b.rows[0] *= p_scale.x;
		b.rows[1] *= p_scale.y;
		b.rows[2] *= p_scale.z;
		return b;
	}

	// ------------------------------------------------------------------------
	// Interpolation (Warp-Kernel Friendly)
	// ------------------------------------------------------------------------
	
	/**
	 * lerp()
	 * Linear interpolation of basis components.
	 */
	ET_SIMD_INLINE Basis<T> lerp(const Basis<T> &p_to, T p_weight) const {
		Basis<T> res;
		res.rows[0] = rows[0].lerp(p_to.rows[0], p_weight);
		res.rows[1] = rows[1].lerp(p_to.rows[1], p_weight);
		res.rows[2] = rows[2].lerp(p_to.rows[2], p_weight);
		return res;
	}

	/**
	 * slerp()
	 * Spherical linear interpolation of the rotation component.
	 * Handled via internal conversion to bit-perfect Quaternions.
	 */
	Basis<T> slerp(const Basis<T> &p_to, T p_weight) const;

	operator String() const {
		return "[" + String(rows[0]) + ", " + String(rows[1]) + ", " + String(rows[2]) + "]";
	}
};

// Simulation Tier Typedefs
typedef Basis<FixedMathCore> Basisf; // Bit-perfect 3D Physical Basis
typedef Basis<BigIntCore> Basisb;    // Macro-Grid Orientation

#endif // BASIS_H

--- END OF FILE core/math/basis.h ---
