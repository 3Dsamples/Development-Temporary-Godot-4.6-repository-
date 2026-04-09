/**************************************************************************/
/*  big_number.cpp                                                        */
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

#include "src/big_number.h"

void BigNumber::_normalize() {
	// Ensure fractional is in range [0, ONE)
	if (fractional >= FixedMathCore::ONE) {
		int64_t carry = fractional >> FixedMathCore::FRACTIONAL_BITS;
		integer += BigIntCore(carry);
		fractional &= (FixedMathCore::ONE - 1);
	} else if (fractional < 0) {
		int64_t borrow = ((-fractional + FixedMathCore::ONE - 1) >> FixedMathCore::FRACTIONAL_BITS);
		integer -= BigIntCore(borrow);
		fractional += (borrow << FixedMathCore::FRACTIONAL_BITS);
	}
	
	// Handle negative zero edge cases
	if (integer.is_zero() && fractional == 0) {
		// BigIntCore handles its own sign, but we ensure consistency
	}
}

BigNumber::BigNumber() : fractional(0) {}

BigNumber::BigNumber(int64_t p_val) : integer(p_val), fractional(0) {}

BigNumber::BigNumber(double p_val) {
	double int_part;
	double frac_part = modf(p_val, &int_part);
	integer = BigIntCore((int64_t)int_part);
	fractional = FixedMathCore::from_double(frac_part);
	_normalize();
}

BigNumber::BigNumber(const BigIntCore &p_int, FixedMathCore::fixed_t p_frac) : integer(p_int), fractional(p_frac) {
	_normalize();
}

BigNumber::BigNumber(const String &p_str) {
	int dot_pos = p_str.find(".");
	if (dot_pos == -1) {
		integer = BigIntCore(p_str);
		fractional = 0;
	} else {
		integer = BigIntCore(p_str.substr(0, dot_pos));
		String frac_str = p_str.substr(dot_pos + 1);
		// Parse fractional part manually to maintain precision
		double f_val = 0.0;
		double divisor = 10.0;
		for (int i = 0; i < frac_str.length(); i++) {
			f_val += (frac_str[i] - '0') / divisor;
			divisor *= 10.0;
		}
		fractional = FixedMathCore::from_double(f_val);
	}
	_normalize();
}

bool BigNumber::operator==(const BigNumber &p_other) const {
	return integer == p_other.integer && fractional == p_other.fractional;
}

bool BigNumber::operator<(const BigNumber &p_other) const {
	if (integer != p_other.integer) return integer < p_other.integer;
	return fractional < p_other.fractional;
}

BigNumber BigNumber::operator+(const BigNumber &p_other) const {
	BigNumber res;
	res.integer = integer + p_other.integer;
	res.fractional = fractional + p_other.fractional;
	res._normalize();
	return res;
}

BigNumber BigNumber::operator-(const BigNumber &p_other) const {
	BigNumber res;
	res.integer = integer - p_other.integer;
	res.fractional = fractional - p_other.fractional;
	res._normalize();
	return res;
}

BigNumber BigNumber::operator*(const BigNumber &p_other) const {
	// (I1 + F1) * (I2 + F2) = I1*I2 + I1*F2 + I2*F1 + F1*F2
	BigNumber res;
	res.integer = integer * p_other.integer;
	
	// Cross terms
	BigIntCore i1f2 = integer * BigIntCore(p_other.fractional);
	BigIntCore i2f1 = p_other.integer * BigIntCore(fractional);
	
	// Add cross terms shifted by fractional bits
	res.integer += (i1f2 + i2f1) >> FixedMathCore::FRACTIONAL_BITS;
	
	// Fractional parts product
	FixedMathCore::fixed_t f1f2 = FixedMathCore::mul(fractional, p_other.fractional);
	res.fractional = f1f2;
	
	// Add remainder of cross terms
	res.fractional += (FixedMathCore::fixed_t)((i1f2.to_int64() + i2f1.to_int64()) & (FixedMathCore::ONE - 1));
	
	res._normalize();
	return res;
}

BigNumber BigNumber::operator/(const BigNumber &p_other) const {
	if (p_other == BigNumber(0)) return BigNumber(0);
	
	// Basic implementation: convert to scaled BigInt and divide
	BigIntCore a = (integer << FixedMathCore::FRACTIONAL_BITS) + BigIntCore(fractional);
	BigIntCore b = (p_other.integer << FixedMathCore::FRACTIONAL_BITS) + BigIntCore(p_other.fractional);
	
	BigIntCore q, r;
	// We want q to have FRACTIONAL_BITS of precision
	BigIntCore::div_mod(a << FixedMathCore::FRACTIONAL_BITS, b, q, r);
	
	BigNumber res;
	res.integer = q >> FixedMathCore::FRACTIONAL_BITS;
	res.fractional = (FixedMathCore::fixed_t)(q.to_int64() & (FixedMathCore::ONE - 1));
	res._normalize();
	return res;
}

String BigNumber::to_string() const {
	String s = integer.to_string();
	if (fractional != 0) {
		s += ".";
		double f = FixedMathCore::to_double(fractional);
		String fs = String::num(f, 10);
		if (fs.find(".") != -1) {
			s += fs.substr(fs.find(".") + 1);
		}
	}
	return s;
}

double BigNumber::to_double() const {
	return integer.to_double() + FixedMathCore::to_double(fractional);
}

BigNumber BigNumber::abs(const BigNumber &p_val) {
	BigNumber res = p_val;
	if (res.integer.is_negative()) {
		res.integer = -res.integer;
		// Fractional is already normalized to be positive if integer is handled correctly
	}
	return res;
}

BigNumber &BigNumber::operator+=(const BigNumber &p_other) { *this = *this + p_other; return *this; }
BigNumber &BigNumber::operator-=(const BigNumber &p_other) { *this = *this - p_other; return *this; }