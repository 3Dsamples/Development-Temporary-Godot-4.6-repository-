/**************************************************************************/
/*  big_number.h                                                          */
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

#ifndef BIG_NUMBER_H
#define BIG_NUMBER_H

#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * @class BigNumber
 * @brief Unified coordinate system for microscopic to galactic scales.
 * 
 * Uses BigIntCore for the "Sector" or "Integer" part and FixedMathCore::fixed_t
 * for the high-precision local offset. This ensures that even at a distance of
 * billions of light-years, local physics remain bit-perfect and jitter-free.
 */

class BigNumber {
	BigIntCore integer;
	FixedMathCore::fixed_t fractional; // Range [0, FixedMathCore::ONE)

	void _normalize();

public:
	BigNumber();
	BigNumber(int64_t p_val);
	BigNumber(double p_val);
	BigNumber(const BigIntCore &p_int, FixedMathCore::fixed_t p_frac = 0);
	BigNumber(const String &p_str);

	// Comparison
	bool operator==(const BigNumber &p_other) const;
	bool operator!=(const BigNumber &p_other) const;
	bool operator<(const BigNumber &p_other) const;
	bool operator<=(const BigNumber &p_other) const;
	bool operator>(const BigNumber &p_other) const;
	bool operator>=(const BigNumber &p_other) const;

	// Arithmetic
	BigNumber operator+(const BigNumber &p_other) const;
	BigNumber operator-(const BigNumber &p_other) const;
	BigNumber operator*(const BigNumber &p_other) const;
	BigNumber operator/(const BigNumber &p_other) const;
	BigNumber operator-() const;

	BigNumber &operator+=(const BigNumber &p_other);
	BigNumber &operator-=(const BigNumber &p_other);
	BigNumber &operator*=(const BigNumber &p_other);
	BigNumber &operator/=(const BigNumber &p_other);

	// Utilities
	String to_string() const;
	double to_double() const;
	float to_float() const;
	
	_FORCE_INLINE_ const BigIntCore& get_integer() const { return integer; }
	_FORCE_INLINE_ FixedMathCore::fixed_t get_fractional() const { return fractional; }

	static BigNumber floor(const BigNumber &p_val);
	static BigNumber ceil(const BigNumber &p_val);
	static BigNumber round(const BigNumber &p_val);
	static BigNumber abs(const BigNumber &p_val);
};

#endif // BIG_NUMBER_H