--- START OF FILE core/math/matrix3.cpp ---

#include "core/math/matrix3.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for the two core simulation tiers:
 * - Matrix3f: Bit-perfect 3D transformations for physics and robotics.
 * - Matrix3b: Discrete macro-transformations for galactic sector mapping.
 */
template struct Matrix3<FixedMathCore>;
template struct Matrix3<BigIntCore>;

// ============================================================================
// Matrix Inversion (Deterministic Cramer's Rule)
// ============================================================================

template <typename T>
Matrix3<T> Matrix3<T>::inverse() const {
	T det = determinant();
	if (unlikely(det == MathConstants<T>::zero())) {
		return Matrix3<T>(); // Return Identity if non-invertible
	}

	T inv_det = MathConstants<T>::one() / det;
	Matrix3<T> res;

	// Transpose of the Cofactor Matrix (Adjugate)
	res[0][0] = (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) * inv_det;
	res[0][1] = (rows[0][2] * rows[2][1] - rows[0][1] * rows[2][2]) * inv_det;
	res[0][2] = (rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1]) * inv_det;
	
	res[1][0] = (rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2]) * inv_det;
	res[1][1] = (rows[0][0] * rows[2][2] - rows[0][2] * rows[2][0]) * inv_det;
	res[1][2] = (rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2]) * inv_det;
	
	res[2][0] = (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]) * inv_det;
	res[2][1] = (rows[0][1] * rows[2][0] - rows[0][0] * rows[2][1]) * inv_det;
	res[2][2] = (rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]) * inv_det;

	return res;
}

// ============================================================================
// Rotation API (Deterministic Logic)
// ============================================================================

/**
 * set_euler()
 * 
 * Converts XYZ Euler angles to a rotation matrix.
 * Uses bit-perfect FixedMathCore sin/cos to eliminate precision jitter.
 */
template <typename T>
void Matrix3<T>::set_euler(const Vector3<T> &p_euler) {
	T c, s;

	c = Math::cos(p_euler.x);
	s = Math::sin(p_euler.x);
	Matrix3<T> xmat(MathConstants<T>::one(), MathConstants<T>::zero(), MathConstants<T>::zero(), 
	                MathConstants<T>::zero(), c, -s, 
	                MathConstants<T>::zero(), s, c);

	c = Math::cos(p_euler.y);
	s = Math::sin(p_euler.y);
	Matrix3<T> ymat(c, MathConstants<T>::zero(), s, 
	                MathConstants<T>::zero(), MathConstants<T>::one(), MathConstants<T>::zero(), 
	                -s, MathConstants<T>::zero(), c);

	c = Math::cos(p_euler.z);
	s = Math::sin(p_euler.z);
	Matrix3<T> zmat(c, -s, MathConstants<T>::zero(), 
	                s, c, MathConstants<T>::zero(), 
	                MathConstants<T>::zero(), MathConstants<T>::zero(), MathConstants<T>::one());

	// Order: YXZ for standard engine camera/entity orientation
	*this = ymat * xmat * zmat;
}

/**
 * get_euler()
 * 
 * Extracts XYZ Euler angles using bit-perfect trigonometry.
 */
template <typename T>
Vector3<T> Matrix3<T>::get_euler() const {
	Vector3<T> euler;
	// Use bit-perfect asin check
	euler.x = Math::asin(-CLAMP(rows[1][2], -MathConstants<T>::one(), MathConstants<T>::one()));
	
	if (Math::abs(rows[1][2]) < T(4294924294LL, true)) { // ~0.999999
		euler.y = Math::atan2(rows[0][2], rows[2][2]);
		euler.z = Math::atan2(rows[1][0], rows[1][1]);
	} else {
		euler.y = Math::atan2(-rows[2][0], rows[0][0]);
		euler.z = MathConstants<T>::zero();
	}
	return euler;
}

/**
 * Axis-Angle Constructor
 * 
 * Implementation of Rodrigues' Rotation Formula.
 * R = I + sin(theta)K + (1 - cos(theta))K^2
 */
template <typename T>
Matrix3<T>::Matrix3(const Vector3<T> &p_axis, T p_angle) {
	Vector3<T> axis = p_axis.normalized();
	T s = Math::sin(p_angle);
	T c = Math::cos(p_angle);
	T omc = MathConstants<T>::one() - c;

	T x = axis.x;
	T y = axis.y;
	T z = axis.z;

	rows[0][0] = x * x * omc + c;
	rows[0][1] = x * y * omc - z * s;
	rows[0][2] = x * z * omc + y * s;
	
	rows[1][0] = y * x * omc + z * s;
	rows[1][1] = y * y * omc + c;
	rows[1][2] = y * z * omc - x * s;
	
	rows[2][0] = z * x * omc - y * s;
	rows[2][1] = z * y * omc + x * s;
	rows[2][2] = z * z * omc + c;
}

// Instantiate specific non-inline constructor logic
template Matrix3<FixedMathCore>::Matrix3(const Vector3<FixedMathCore> &p_axis, FixedMathCore p_angle);
template Matrix3<BigIntCore>::Matrix3(const Vector3<BigIntCore> &p_axis, BigIntCore p_angle);

// ============================================================================
// Deterministic Basis Constants
// ============================================================================

const Matrix3f Matrix3f_IDENTITY = Matrix3f();
const Matrix3b Matrix3b_IDENTITY = Matrix3b();

--- END OF FILE core/math/matrix3.cpp ---
