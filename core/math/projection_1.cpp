--- START OF FILE core/math/projection.cpp ---

#include "core/math/projection.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the machine code for:
 * - Projectionf: Bit-perfect camera math and homogeneous culling (FixedMathCore).
 * - Projectionb: Discrete macro-depth transformations (BigIntCore).
 */
template struct Projection<FixedMathCore>;
template struct Projection<BigIntCore>;

// ============================================================================
// Matrix Inversion (Deterministic Gauss-Jordan Elimination)
// ============================================================================

template <typename T>
void Projection<T>::invert() {
	// Create an augmented 4x8 matrix [Matrix | Identity]
	T m[4][8];
	T zero = MathConstants<T>::zero();
	T one = MathConstants<T>::one();

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m[i][j] = columns[j][i];
			m[i][j + 4] = (i == j) ? one : zero;
		}
	}

	// Partial Pivoting and Elimination
	for (int i = 0; i < 4; i++) {
		int pivot = i;
		T max_val = Math::abs(m[i][i]);

		// Find best pivot for numerical stability
		for (int j = i + 1; j < 4; j++) {
			T current_abs = Math::abs(m[j][i]);
			if (current_abs > max_val) {
				max_val = current_abs;
				pivot = j;
			}
		}

		// Swap rows in the augmented matrix
		if (pivot != i) {
			for (int k = 0; k < 8; k++) {
				T temp = m[i][k];
				m[i][k] = m[pivot][k];
				m[pivot][k] = temp;
			}
		}

		T div = m[i][i];
		// If divisor is near zero, the matrix is singular (degenerate frustum)
		if (unlikely(Math::abs(div) < CMP_EPSILON)) {
			*this = Projection<T>(); // Reset to Identity
			return;
		}

		// Scale pivot row
		T inv_div = one / div;
		for (int k = i; k < 8; k++) {
			m[i][k] *= inv_div;
		}

		// Eliminate other rows
		for (int j = 0; j < 4; j++) {
			if (j != i) {
				T factor = m[j][i];
				for (int k = i; k < 8; k++) {
					m[j][k] -= factor * m[i][k];
				}
			}
		}
	}

	// Extract the inverted matrix from the right side of the augmented matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			columns[j][i] = m[i][j + 4];
		}
	}
}

// ============================================================================
// Global Deterministic Constants (FixedMathCore Q32.32)
// ============================================================================

/**
 * Static Constants for Projectionf
 * 
 * Defined via raw bit patterns to ensure zero-cost loading in parallel 
 * Warp-style simulation sweeps.
 */

const Projectionf Projectionf_IDENTITY = Projectionf(
	Vector4f(FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true), FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(FixedMathCore::ONE_RAW, true))
);

// ============================================================================
// Global Deterministic Constants (BigIntCore)
// ============================================================================

const Projectionb Projectionb_IDENTITY = Projectionb(
	Vector4b(BigIntCore(1LL), BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL)),
	Vector4b(BigIntCore(0LL), BigIntCore(1LL), BigIntCore(0LL), BigIntCore(0LL)),
	Vector4b(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(1LL), BigIntCore(0LL)),
	Vector4b(BigIntCore(0LL), BigIntCore(0LL), BigIntCore(0LL), BigIntCore(1LL))
);

/**
 * Implementation Note:
 * 
 * The Gauss-Jordan algorithm implemented here is bit-perfect. In a 
 * 120 FPS high-speed spaceship sequence, a robotic sensor might need 
 * to unproject millions of depth samples into 3D space to reconstruct 
 * a local sector map. Because this code avoids the FPU entirely, 
 * every unprojected point is identical on all clients, enabling 
 * zero-copy collision detection against procedurally reconstructed geometry.
 */

--- END OF FILE core/math/projection.cpp ---
