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
 * A 4x4 matrix specifically optimized for camera projection and 
 * homogeneous coordinates. Aligned to 32 bytes to ensure that EnTT 
 * component streams are SIMD-ready for high-speed Warp kernel 
 * frustum culling and vertex projection.
 */
template <typename T>
struct ET_ALIGN_32 Projection {
	Vector4<T> columns[4];

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Projection() {
		columns[0] = Vector4<T>(MathConstants<T>::one(),  MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero());
		columns[1] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::one(),  MathConstants<T>::zero(), MathConstants<T>::zero());
		columns[2] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one(),  MathConstants<T>::zero());
		columns[3] = Vector4<T>(MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());
	}

	ET_SIMD_INLINE Projection(const Vector4<T> &p_c0, const Vector4<T> &p_c1, const Vector4<T> &p_c2, const Vector4<T> &p_c3) {
		columns[0] = p_c0;
		columns[1] = p_c1;
		columns[2] = p_c2;
		columns[3] = p_c3;
	}

	// ------------------------------------------------------------------------
	// Accessors (Warp Kernel Friendly)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Vector4<T> &operator[](int p_idx) { return columns[p_idx]; }
	ET_SIMD_INLINE const Vector4<T> &operator[](int p_idx) const { return columns[p_idx]; }

	ET_SIMD_INLINE T operator()(int p_row, int p_col) const { return columns[p_col][p_row]; }
	ET_SIMD_INLINE T &operator()(int p_row, int p_col) { return columns[p_col][p_row]; }

	// ------------------------------------------------------------------------
	// Operators (Batch Transformation Logic)
	// ------------------------------------------------------------------------
	ET_SIMD_INLINE Projection<T> operator*(const Projection<T> &p_matrix) const {
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

	ET_SIMD_INLINE Vector4<T> xform(const Vector4<T> &p_vec) const {
		return Vector4<T>(
				columns[0][0] * p_vec.x + columns[1][0] * p_vec.y + columns[2][0] * p_vec.z + columns[3][0] * p_vec.w,
				columns[0][1] * p_vec.x + columns[1][1] * p_vec.y + columns[2][1] * p_vec.z + columns[3][1] * p_vec.w,
				columns[0][2] * p_vec.x + columns[1][2] * p_vec.y + columns[2][2] * p_vec.z + columns[3][2] * p_vec.w,
				columns[0][3] * p_vec.x + columns[1][3] * p_vec.y + columns[2][3] * p_vec.z + columns[3][3] * p_vec.w);
	}

	// ------------------------------------------------------------------------
	// Deterministic Matrix Generation
	// ------------------------------------------------------------------------
	static Projection<T> create_perspective(T p_fovy_degrees, T p_aspect, T p_z_near, T p_z_far) {
		T radians = Math::deg_to_rad(p_fovy_degrees) * MathConstants<T>::half();
		T delta_z = p_z_far - p_z_near;
		T sine = Math::sin(radians);
		if (delta_z == MathConstants<T>::zero() || sine == MathConstants<T>::zero() || p_aspect == MathConstants<T>::zero()) {
			return Projection<T>();
		}
		T cotangent = Math::cos(radians) / sine;

		Projection<T> m;
		m.columns[0][0] = cotangent / p_aspect;
		m.columns[1][1] = cotangent;
		m.columns[2][2] = -(p_z_far + p_z_near) / delta_z;
		m.columns[2][3] = -MathConstants<T>::one();
		m.columns[3][2] = -(FixedMathCore(2LL, false) * p_z_far * p_z_near) / delta_z;
		m.columns[3][3] = MathConstants<T>::zero();
		return m;
	}

	static Projection<T> create_orthogonal(T p_left, T p_right, T p_bottom, T p_top, T p_znear, T p_zfar) {
		Projection<T> m;
		T two = FixedMathCore(2LL, false);
		m.columns[0][0] = two / (p_right - p_left);
		m.columns[1][1] = two / (p_top - p_bottom);
		m.columns[2][2] = -two / (p_zfar - p_znear);
		m.columns[3][0] = -(p_right + p_left) / (p_right - p_left);
		m.columns[3][1] = -(p_top + p_bottom) / (p_top - p_bottom);
		m.columns[3][2] = -(p_zfar + p_znear) / (p_zfar - p_znear);
		m.columns[3][3] = MathConstants<T>::one();
		return m;
	}

	// ------------------------------------------------------------------------
	// Transformation API
	// ------------------------------------------------------------------------
	void invert() {
		T m[4][8];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				m[i][j] = columns[j][i];
				m[i][j + 4] = (i == j) ? MathConstants<T>::one() : MathConstants<T>::zero();
			}
		}

		for (int i = 0; i < 4; i++) {
			int pivot = i;
			for (int j = i + 1; j < 4; j++) {
				if (Math::abs(m[j][i]) > Math::abs(m[pivot][i])) pivot = j;
			}
			for (int k = 0; k < 8; k++) {
				T temp = m[i][k];
				m[i][k] = m[pivot][k];
				m[pivot][k] = temp;
			}
			
			T div = m[i][i];
			if (div == MathConstants<T>::zero()) continue; 

			for (int k = 0; k < 8; k++) m[i][k] /= div;
			for (int j = 0; j < 4; j++) {
				if (j != i) {
					T prev = m[j][i];
					for (int k = 0; k < 8; k++) m[j][k] -= prev * m[i][k];
				}
			}
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				columns[j][i] = m[i][j + 4];
			}
		}
	}

	ET_SIMD_INLINE Projection<T> inverted() const {
		Projection<T> m = *this;
		m.invert();
		return m;
	}

	// Godot UI Conversion
	operator String() const {
		return "(" + (String)columns[0] + "), (" + (String)columns[1] + "), (" + (String)columns[2] + "), (" + (String)columns[3] + ")";
	}
};

typedef Projection<FixedMathCore> Projectionf;
typedef Projection<BigIntCore> Projectionb;

#endif // PROJECTION_H

--- END OF FILE core/math/projection.h ---
