/**************************************************************************/
/*  fixed_math_core.cpp                                                   */
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

#include "src/fixed_math_core.h"

FixedMathCore::fixed_t FixedMathCore::sqrt(fixed_t p_val) {
	if (p_val < 0) return 0;
	if (p_val == 0) return 0;

	intermediate_t res = 0;
	intermediate_t bit = (intermediate_t)1 << 62; // Second-to-top bit
	intermediate_t num = (intermediate_t)p_val << FRACTIONAL_BITS;

	while (bit > num) {
		bit >>= 2;
	}

	while (bit != 0) {
		if (num >= res + bit) {
			num -= res + bit;
			res = (res >> 1) + bit;
		} else {
			res >>= 1;
		}
		bit >>= 2;
	}
	return (fixed_t)res;
}

FixedMathCore::fixed_t FixedMathCore::sin(fixed_t p_val) {
	// Range reduction to [0, 2PI)
	fixed_t x = p_val % TWO_PI;
	if (x < 0) x += TWO_PI;

	// Further reduction to [0, PI/2]
	bool neg = false;
	if (x > PI) {
		x -= PI;
		neg = true;
	}
	if (x > HALF_PI) {
		x = PI - x;
	}

	// 13th degree Taylor series for high precision
	// sin(x) = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13!
	intermediate_t x1 = x;
	intermediate_t x2 = mul(x, x);
	intermediate_t x3 = mul(x1, x2);
	intermediate_t x5 = mul(x3, x2);
	intermediate_t x7 = mul(x5, x2);
	intermediate_t x9 = mul(x7, x2);
	intermediate_t x11 = mul(x9, x2);
	intermediate_t x13 = mul(x11, x2);

	fixed_t res = (fixed_t)(x1 
		- x3 / 6 
		+ x5 / 120 
		- x7 / 5040 
		+ x9 / 362880 
		- x11 / 39916800 
		+ x13 / 6227020800LL);

	return neg ? -res : res;
}

FixedMathCore::fixed_t FixedMathCore::cos(fixed_t p_val) {
	return sin(p_val + HALF_PI);
}

FixedMathCore::fixed_t FixedMathCore::tan(fixed_t p_val) {
	fixed_t c = cos(p_val);
	if (abs(c) < EPSILON) return (p_val >= 0) ? (fixed_t)0x7FFFFFFFFFFFFFFFLL : (fixed_t)0x8000000000000000LL;
	return div(sin(p_val), c);
}

FixedMathCore::fixed_t FixedMathCore::atan2(fixed_t p_y, fixed_t p_x) {
	if (p_x == 0) {
		if (p_y > 0) return HALF_PI;
		if (p_y < 0) return -HALF_PI;
		return 0;
	}

	// Rational approximation for atan(z) where z = y/x
	fixed_t z = div(p_y, p_x);
	fixed_t res;
	
	if (abs(z) < ONE) {
		// atan(z) = z / (1 + 0.28z^2)
		res = div(z, ONE + mul(1202590842LL, mul(z, z))); // 0.28 * 2^32
	} else {
		// atan(z) = PI/2 - z / (z^2 + 0.28)
		res = (z > 0 ? HALF_PI : -HALF_PI) - div(z, mul(z, z) + 1202590842LL);
	}

	if (p_x < 0) {
		res += (p_y >= 0) ? PI : -PI;
	}

	return res;
}

FixedMathCore::fixed_t FixedMathCore::exp(fixed_t p_val) {
	if (p_val == 0) return ONE;
	if (p_val < -20 * ONE) return 0; // Practical zero

	// Taylor series for e^x
	// 1 + x + x^2/2! + x^3/3! + x^4/4! ...
	intermediate_t res = ONE;
	intermediate_t term = ONE;
	for (int i = 1; i < 20; i++) {
		term = mul((fixed_t)term, p_val) / i;
		res += term;
		if (term == 0) break;
	}
	return (fixed_t)res;
}

FixedMathCore::fixed_t FixedMathCore::log(fixed_t p_val) {
	if (p_val <= 0) return 0;

	// Newton's method for ln(x): y_{n+1} = y_n + 2 * (x - e^{y_n}) / (x + e^{y_n})
	fixed_t res = from_int(1); // Initial guess
	for (int i = 0; i < 10; i++) {
		fixed_t ex = exp(res);
		res = res + div(mul(from_int(2), p_val - ex), p_val + ex);
	}
	return res;
}

FixedMathCore::fixed_t FixedMathCore::pow(fixed_t p_base, fixed_t p_exp) {
	if (p_base <= 0) return 0;
	// x^y = exp(y * ln(x))
	return exp(mul(p_exp, log(p_base)));
}