--- START OF FILE core/math/color.cpp ---

#include "core/math/color.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Color logic for the Universal Solver backend.
 * By instantiating for FixedMathCore and BigIntCore, we enable EnTT 
 * to store color tensors in dense SoA buffers, allowing Warp kernels 
 * to perform massive parallel color correction or light energy 
 * integration with bit-perfect accuracy.
 */

template struct Color<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect HDR/Sim
template struct Color<BigIntCore>;    // TIER_MACRO: Discrete Spectral Analysis

/**
 * Deterministic Color Constants
 * 
 * Pre-allocated bit-perfect colors. These utilize raw bit-assignment 
 * to bypass runtime calculation, ensuring that standard palettes are 
 * available instantly for Warp-based rendering kernels.
 */

// Identity/Standard Colors (FixedMathCore Q32.32)
const Colorf Colorf_BLACK       = Colorf(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(1LL, false));
const Colorf Colorf_WHITE       = Colorf(FixedMathCore(1LL, false), FixedMathCore(1LL, false), FixedMathCore(1LL, false), FixedMathCore(1LL, false));
const Colorf Colorf_TRANSPARENT = Colorf(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(0LL, true));
const Colorf Colorf_RED         = Colorf(FixedMathCore(1LL, false), FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(1LL, false));
const Colorf Colorf_GREEN       = Colorf(FixedMathCore(0LL, true), FixedMathCore(1LL, false), FixedMathCore(0LL, true), FixedMathCore(1LL, false));
const Colorf Colorf_BLUE        = Colorf(FixedMathCore(0LL, true), FixedMathCore(0LL, true), FixedMathCore(1LL, false), FixedMathCore(1LL, false));
const Colorf Colorf_GRAY        = Colorf(FixedMathCore(2147483648LL, true), FixedMathCore(2147483648LL, true), FixedMathCore(2147483648LL, true), FixedMathCore(1LL, false));

// Macro Scale Discrete Colors (BigIntCore)
const Colorb Colorb_BLACK = Colorb(BigIntCore(0), BigIntCore(0), BigIntCore(0), BigIntCore(1));
const Colorb Colorb_WHITE = Colorb(BigIntCore(1), BigIntCore(1), BigIntCore(1), BigIntCore(1));

/**
 * Warp Symmetry: Parallel Spectral Processing
 * 
 * Because Color is ET_ALIGN_32, it fits the SIMD register profile of 
 * modern CPUs and GPUs. Warp kernels can treat EnTT streams of Colorf 
 * as raw energy tensors, performing light attenuation or albedo 
 * blending across millions of entities simultaneously without any 
 * floating-point rounding divergence.
 */

--- END OF FILE core/math/color.cpp ---
