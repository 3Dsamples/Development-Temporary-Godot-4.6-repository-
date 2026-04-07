--- START OF FILE core/math/face3.cpp ---

#include "core/math/face3.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Face3 logic for the Universal Solver backend.
 * By instantiating for FixedMathCore, we provide the linker with 
 * high-performance symbols that EnTT Sparse Sets use to process 
 * geometric primitives during batch-oriented physics sweeps.
 */

template struct Face3<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect Geometry

/**
 * get_closest_point()
 * 
 * Determines the point on the triangle surface closest to p_point.
 * Ported to deterministic software-arithmetic to ensure identical 
 * contact resolution across all simulation instances.
 */
template <typename T>
Vector3<T> Face3<T>::get_closest_point(const Vector3<T> &p_point) const {
	Vector3<T> edge0 = vertex[1] - vertex[0];
	Vector3<T> edge1 = vertex[2] - vertex[0];
	Vector3<T> v0 = vertex[0] - p_point;

	T a = edge0.dot(edge0);
	T b = edge0.dot(edge1);
	T c = edge1.dot(edge1);
	T d = edge0.dot(v0);
	T e = edge1.dot(v0);

	T det = a * c - b * b;
	T s = b * e - c * d;
	T t = b * d - a * e;

	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	if (s + t <= det) {
		if (s < zero) {
			if (t < zero) {
				if (d < zero) {
					s = CLAMP(-d / a, zero, one);
					t = zero;
				} else {
					s = zero;
					t = CLAMP(-e / c, zero, one);
				}
			} else {
				s = zero;
				t = CLAMP(-e / c, zero, one);
			}
		} else if (t < zero) {
			s = CLAMP(-d / a, zero, one);
			t = zero;
		} else {
			T invDet = one / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < zero) {
			T tmp0 = b + d;
			T tmp1 = c + e;
			if (tmp1 > tmp0) {
				T numer = tmp1 - tmp0;
				T denom = a - FixedMathCore(2LL, false) * b + c;
				s = CLAMP(numer / denom, zero, one);
				t = one - s;
			} else {
				t = CLAMP(-e / c, zero, one);
				s = zero;
			}
		} else if (t < zero) {
			if (a + d > b + e) {
				T numer = c + e - b - d;
				T denom = a - FixedMathCore(2LL, false) * b + c;
				s = CLAMP(numer / denom, zero, one);
				t = one - s;
			} else {
				s = CLAMP(-e / c, zero, one);
				t = zero;
			}
		} else {
			T numer = c + e - b - d;
			T denom = a - FixedMathCore(2LL, false) * b + c;
			s = CLAMP(numer / denom, zero, one);
			t = one - s;
		}
	}

	return vertex[0] + edge0 * s + edge1 * t;
}

--- END OF FILE core/math/face3.cpp ---
