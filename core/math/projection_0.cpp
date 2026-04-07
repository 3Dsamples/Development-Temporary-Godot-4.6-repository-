--- START OF FILE core/math/projection.cpp ---

#include "core/math/projection.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 4x4 Projection logic for the Universal Solver backend.
 * By instantiating for FixedMathCore and BigIntCore, we provide the 
 * linker with high-performance symbols that EnTT registries use to 
 * process homogeneous coordinate batches with zero-copy efficiency.
 */

template struct Projection<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect Frustums
template struct Projection<BigIntCore>;    // TIER_MACRO: Discrete Scale Transformations

/**
 * Deterministic Projection Constants
 * 
 * Pre-allocated identity matrices using raw bit-assignment.
 * Used for initializing camera and transform buffers without 
 * triggering runtime parsing or floating-point anomalies.
 */

// Identity Projection (FixedMathCore Q32.32)
const Projectionf Projectionf_IDENTITY = Projectionf(
	Vector4f(FixedMathCore(1LL, false), FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true),  FixedMathCore(1LL, false), FixedMathCore(0LL, true),  FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(1LL, false), FixedMathCore(0LL, true)),
	Vector4f(FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(0LL, true),  FixedMathCore(1LL, false))
);

// Macro Scale Identity (BigIntCore)
const Projectionb Projectionb_IDENTITY = Projectionb(
	Vector4b(BigIntCore(1), BigIntCore(0), BigIntCore(0), BigIntCore(0)),
	Vector4b(BigIntCore(0), BigIntCore(1), BigIntCore(0), BigIntCore(0)),
	Vector4b(BigIntCore(0), BigIntCore(0), BigIntCore(1), BigIntCore(0)),
	Vector4b(BigIntCore(0), BigIntCore(0), BigIntCore(0), BigIntCore(1))
);

/**
 * Warp Optimization: SIMD Transform Sweeps
 * 
 * Because Projection is ET_ALIGN_32, EnTT streams of these matrices 
 * are perfectly aligned for AVX-512 or CUDA memory access. 
 * Warp kernels can batch-transform entire entity sectors, 
 * maintaining absolute spatial integrity in galactic environments.
 */

--- END OF FILE core/math/projection.cpp ---
