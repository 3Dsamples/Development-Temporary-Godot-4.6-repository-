--- START OF FILE core/math/matrix3.cpp ---

#include "core/math/matrix3.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 3x3 matrix logic for the Universal Solver backend.
 * These symbols enable EnTT to manage 3x3 bases as contiguous SoA data,
 * allowing Warp kernels to perform massive parallel rotations or 
 * orthonormalization sweeps with zero-copy efficiency.
 */

template struct Matrix3<FixedMathCore>;  // TIER_DETERMINISTIC: Bit-perfect Rotations
template struct Matrix3<BigIntCore>;    // TIER_MACRO: Discrete Basis Transformations

/**
 * Deterministic Matrix Constants
 * 
 * Pre-allocated identity matrices to ensure instant availability 
 * during simulation resets or coordinate space transitions.
 */

// Deterministic Basis Constants (FixedMathCore Q32.32)
const Matrix3f Matrix3f_IDENTITY = Matrix3f(
	Vector3f(FixedMathCore(1LL, false), FixedMathCore(0LL, true),  FixedMathCore(0LL, true)),
	Vector3f(FixedMathCore(0LL, true),  FixedMathCore(1LL, false), FixedMathCore(0LL, true)),
	Vector3f(FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(1LL, false))
);

// Macro Scale Constants (BigIntCore)
const Matrix3b Matrix3b_IDENTITY = Matrix3b(
	Vector3b(BigIntCore(1), BigIntCore(0), BigIntCore(0)),
	Vector3b(BigIntCore(0), BigIntCore(1), BigIntCore(0)),
	Vector3b(BigIntCore(0), BigIntCore(0), BigIntCore(1))
);

/**
 * Hardware Symmetry: SIMD Auto-Vectorization
 * 
 * By providing these compiled symbols and maintaining ET_ALIGN_32,
 * the compiler's auto-vectorizer can optimize Matrix-Vector multiplications
 * when Warp kernels iterate through EnTT component streams, matching
 * the throughput of native NVIDIA Warp execution.
 */

--- END OF FILE core/math/matrix3.cpp ---
