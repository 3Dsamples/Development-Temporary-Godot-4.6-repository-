/**************************************************************************/
/*  fixed_math_core.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef FIXED_MATH_CORE_H
#define FIXED_MATH_CORE_H

#include "core/typedefs.h"
#include "src/big_int_core.h"

/**
 * @class FixedMathCore
 * @brief Provides deterministic fixed-point arithmetic to eliminate floating-point non-determinism.
 * 
 * Uses a 64-bit integer with 32 bits for the fractional part (Q32.32 format).
 * This provides a range of +/- 2 billion units with sub-nanometer precision.
 * For galactic scales, these are used as local offsets relative to BigIntCore coordinates.
 */

class FixedMathCore {
public:
	typedef int64_t fixed_t;
	typedef __int128_t intermediate_t;

	static const int32_t FRACTIONAL_BITS = 32;
	static const fixed_t ONE = (fixed_t)1 << FRACTIONAL_BITS;
	static const fixed_t HALF = ONE >> 1;
	static const fixed_t PI = 13491395137LL; // 3.1415926535... * 2^32
	static const fixed_t TWO_PI = 26982790274LL;
	static const fixed_t HALF_PI = 6745697568LL;
	static const fixed_t EPSILON = 1;

	// Basic Conversion
	static _FORCE_INLINE_ fixed_t from_int(int64_t p_val) { return p_val << FRACTIONAL_BITS; }
	static _FORCE_INLINE_ fixed_t from_double(double p_val) { return (fixed_t)(p_val * (double)ONE + (p_val >= 0 ? 0.5 : -0.5)); }
	static _FORCE_INLINE_ double to_double(fixed_t p_val) { return (double)p_val / (double)ONE; }
	static _FORCE_INLINE_ int64_t to_int(fixed_t p_val) { return p_val >> FRACTIONAL_BITS; }

	// Arithmetic
	static _FORCE_INLINE_ fixed_t mul(fixed_t p_a, fixed_t p_b) {
		return (fixed_t)(((intermediate_t)p_a * p_b) >> FRACTIONAL_BITS);
	}

	static _FORCE_INLINE_ fixed_t div(fixed_t p_a, fixed_t p_b) {
		if (p_b == 0) return 0; // Logic for division by zero
		return (fixed_t)(((intermediate_t)p_a << FRACTIONAL_BITS) / p_b);
	}

	// Advanced Math Functions (Full Logic Implementation)
	static fixed_t sqrt(fixed_t p_val);
	static fixed_t sin(fixed_t p_val);
	static fixed_t cos(fixed_t p_val);
	static fixed_t tan(fixed_t p_val);
	static fixed_t atan2(fixed_t p_y, fixed_t p_x);
	static fixed_t pow(fixed_t p_base, fixed_t p_exp);
	static fixed_t exp(fixed_t p_val);
	static fixed_t log(fixed_t p_val);

	// Interpolation and Clamp
	static _FORCE_INLINE_ fixed_t lerp(fixed_t p_a, fixed_t p_b, fixed_t p_t) {
		return p_a + mul(p_b - p_a, p_t);
	}

	static _FORCE_INLINE_ fixed_t clamp(fixed_t p_val, fixed_t p_min, fixed_t p_max) {
		if (p_val < p_min) return p_min;
		if (p_val > p_max) return p_max;
		return p_val;
	}

	static _FORCE_INLINE_ fixed_t abs(fixed_t p_val) {
		return p_val < 0 ? -p_val : p_val;
	}

	static _FORCE_INLINE_ fixed_t floor(fixed_t p_val) {
		return p_val & ~(ONE - 1);
	}

	static _FORCE_INLINE_ fixed_t ceil(fixed_t p_val) {
		return (p_val + ONE - 1) & ~(ONE - 1);
	}

	// Bit-perfect comparisons
	static _FORCE_INLINE_ bool is_equal_approx(fixed_t p_a, fixed_t p_b) {
		return abs(p_a - p_b) <= EPSILON;
	}
};

#endif // FIXED_MATH_CORE_H