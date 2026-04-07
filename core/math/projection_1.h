--- START OF FILE core/math/projection.h ---

#ifndef PROJECTION_H
#define PROJECTION_H

#include "core/math/vector4.h"
#include "core/math/math_funcs.h"
#include "src/fixed_math_core.h"
#include "src/big_int_core.h"

/**
 * Projection Template
 * 
 * 32-byte aligned 4x4 matrix specifically for camera space and projection.
 * Replaces floating-point depth and frustum logic with bit-perfect Software-Defined Arithmetic.
 * Optimized for EnTT component streams and Warp kernel sweeps.
 */
template <typename T>
struct ET_ALIGN_32 Projection {
	Vector4<T> columns[4];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Projection() {
		columns[0] = Vector4<T>(MathConstants<T>::one(),  MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero());
		columns[1] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::one(),  MathConstants<T>::zero(), MathConstants<T>::zero());
		columns[2] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one(),  MathConstants<T>::zero());
		columns[3] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());
	}

	_FORCE_INLINE_ Projection(const Vector4<T> &p_c0, const Vector4<T> &p_c1, const Vector4<T> &p_c2, const Vector4<T> &p_c3) {
		columns[0] = p_c0;
		columns[1] = p_c1;
		columns[2] = p_c2;
		columns[3] = p_c3;
	}

	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ Vector4<T> &operator[](int p_idx) { return columns[p_idx]; }
	_FORCE_INLINE_ const Vector4<T> &operator[](int p_idx) const { return columns[p_idx]; }

	_FORCE_INLINE_ T operator()(int p_row, int p_col) const { return columns[p_col][p_row]; }
	_FORCE_INLINE_ T &operator()(int p_row, int p_col) { return columns[p_col][p_row]; }

	// ------------------------------------------------------------------------
	// Operators (Deterministic Batch Logic)
	// ------------------------------------------------------------------------

	/**
	 * operator*
	 * Bit-perfect 4x4 matrix multiplication. 
	 * Essential for camera-to-world and projection concatenations.
	 */
	_FORCE_INLINE_ Projection<T> operator*(const Projection<T> &p_matrix) const {
		Projection<T> res;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				res.columns[j][i] = columns[0][i] * p_matrix.columns[j][0] +
									columns[1][i] * p_matrix.columns[j][1] +
									columns[2][i] * p_matrix.columns[j][2] +
									columns[3][i] * p_matrix.columns[j][3];
			}
		}
		return res;
	}

	_FORCE_INLINE_ Vector4<T> xform(const Vector4<T> &p_vec) const {
		return Vector4<T>(
				columns[0][0] * p_vec.x + columns[1][0] * p_vec.y + columns[2][0] * p_vec.z + columns[3][0] * p_vec.w,
				columns[0][1] * p_vec.x + columns[1][1] * p_vec.y + columns[2][1] * p_vec.z + columns[3][1] * p_vec.w,
				columns[0][2] * p_vec.x + columns[1][2] * p_vec.y + columns[2][2] * p_vec.z + columns[3][2] * p_vec.w,
				columns[0][3] * p_vec.x + columns[1][3] * p_vec.y + columns[2][3] * p_vec.z + columns[3][3] * p_vec.w);
	}

	// ------------------------------------------------------------------------
	// Sophisticated Projection API
	// ------------------------------------------------------------------------

	/**
	 * create_perspective()
	 * Deterministic camera frustum generation using FixedMath trigonometry.
	 */
	static Projection<T> create_perspective(T p_fovy_degrees, T p_aspect, T p_z_near, T p_z_far) {
		T radians = Math::deg_to_rad(p_fovy_degrees) * MathConstants<T>::half();
		T delta_z = p_z_far - p_z_near;
		T sine = Math::sin(radians);
		if (unlikely(delta_z == MathConstants<T>::zero() || sine == MathConstants<T>::zero() || p_aspect == MathConstants<T>::zero())) {
			return Projection<T>();
		}
		T cotangent = Math::cos(radians) / sine;

		Projection<T> m;
		m.columns[0][0] = cotangent / p_aspect;
		m.columns[1][1] = cotangent;
		m.columns[2][2] = -(p_z_far + p_z_near) / delta_z;
		m.columns[2][3] = -MathConstants<T>::one();
		m.columns[3][2] = -(T(2LL) * p_z_far * p_z_near) / delta_z;
		m.columns[3][3] = MathConstants<T>::zero();
		return m;
	}

	/**
	 * create_orthogonal()
	 * Zero-parallax projection for deterministic UI and sensor maps.
	 */
	static Projection<T> create_orthogonal(T p_left, T p_right, T p_bottom, T p_top, T p_znear, T p_zfar) {
		Projection<T> m;
		T two = T(2LL);
		m.columns[0][0] = two / (p_right - p_left);
		m.columns[1][1] = two / (p_top - p_bottom);
		m.columns[2][2] = -two / (p_zfar - p_znear);
		m.columns[3][0] = -(p_right + p_left) / (p_right - p_left);
		m.columns[3][1] = -(p_top + p_bottom) / (p_top - p_bottom);
		m.columns[3][2] = -(p_zfar + p_znear) / (p_zfar - p_znear);
		m.columns[3][3] = MathConstants<T>::one();
		return m;
	}

	/**
	 * invert()
	 * Heavy implementation of Gauss-Jordan elimination for 4x4 matrices.
	 * Essential for unprojecting 2D robotic clicks into 3D galactic world-space.
	 */
	void invert();

	_FORCE_INLINE_ Projection<T> inverted() const {
		Projection<T> m = *this;
		m.invert();
		return m;
	}

	operator String() const {
		return "(" + (String)columns[0] + "), (" + (String)columns[1] + "), (" + (String)columns[2] + "), (" + (String)columns[3] + ")";
	}
};

// Simulation Tier Typedefs
typedef Projection<FixedMathCore> Projectionf; // Bit-perfect perspective/frustum
typedef Projection<BigIntCore> Projectionb;    // Macro-depth transformations

#endif // PROJECTION_H

--- END OF FILE core/math/projection.h ---
