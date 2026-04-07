--- START OF FILE core/math/vector4.cpp ---

#include "core/math/vector4.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the 4D vector logic for the Universal Solver backend.
 * These symbols are mapped directly into the EnTT sparse sets to allow
 * SIMD-accelerated Warp kernels to process 4D tensors and colors.
 */

template struct Vector4<FixedMathCore>;  // TIER_DETERMINISTIC: Homogeneous Coordinates
template struct Vector4<BigIntCore>;    // TIER_MACRO: Discrete 4D Metadata

/**
 * Deterministic 4D Constants
 * 
 * Pre-defined bit-perfect vectors. These are used to initialize simulation
 * buffers without the overhead of runtime parsing or FPU conversion.
 */

// Deterministic Physics Constants (FixedMathCore Q32.32)
const Vector4f Vector4f_ZERO = Vector4f(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Vector4f Vector4f_ONE  = Vector4f(FixedMathCore(1LL, false), FixedMathCore(1LL, false), FixedMathCore(1LL, false), FixedMathCore(1LL, false));

// Macro Scale Constants (BigIntCore)
const Vector4b Vector4b_ZERO = Vector4b(BigIntCore(0), BigIntCore(0), BigIntCore(0), BigIntCore(0));
const Vector4b Vector4b_ONE  = Vector4b(BigIntCore(1), BigIntCore(1), BigIntCore(1), BigIntCore(1));

/**
 * Warp Optimization: Linear Memory Mapping
 * 
 * Because Vector4 is ET_ALIGN_32, EnTT streams of Vector4f are perfectly
 * cache-aligned. A Warp kernel can process 4 entities per AVX-512 lane,
 * achieving massive throughput for high-dimensional physics solver steps.
 */

--- END OF FILE core/math/vector4.cpp ---
