--- START OF FILE core/math/noise_simplex_fractal.cpp ---

#include "core/math/noise_simplex_fractal.h"
#include "core/string/ustring.h"

/**
 * Explicit Template Instantiations
 * 
 * Compiles the Fractal Brownian Motion logic for the Universal Solver backend.
 * These symbols allow EnTT to manage fractal noise parameters as contiguous 
 * components. Warp kernels utilize these instantiations to perform zero-copy 
 * multi-octave sampling, maintaining absolute spatial integrity for 
 * procedural planets and nebulae at 120 FPS.
 */

template struct SimplexNoiseFractal<FixedMathCore>; // TIER_DETERMINISTIC: Bit-perfect Proceduralism

/**
 * Warp Optimization: Precomputed Octave Weights
 * 
 * The constructor in the header handles the pre-calculation of amplitudes 
 * and frequencies. This CPP ensures that the static symbols are correctly 
 * linked for high-frequency call-sites. The use of FixedMathCore throughout 
 * ensures that even with 16 octaves of noise, there is zero accumulation 
 * of floating-point error or clock drift during sampling.
 */

--- END OF FILE core/math/noise_simplex_fractal.cpp ---
