/**************************************************************************/
/*  big_int_core.cpp                                                      */
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

#include "src/big_int_core.h"
#include "src/fixed_math_core.h"
#include "src/big_number.h"
#include "core/core_bind.h"
#include "core/core_constants.h"

// Internal Helper: Remove leading zero limbs to keep representations minimal
void BigIntCore::_trim() {
	while (limbs.size() > 0 && limbs[limbs.size() - 1] == 0) {
		limbs.remove_at(limbs.size() - 1);
	}
	if (limbs.size() == 0) {
		negative = false;
	}
}

// Internal Helper: Compare absolute magnitudes (0: equal, 1: a>b, -1: a<b)
int BigIntCore::_compare_absolute(const BigIntCore &p_a, const BigIntCore &p_b) {
	if (p_a.limbs.size() > p_b.limbs.size()) return 1;
	if (p_a.limbs.size() < p_b.limbs.size()) return -1;
	for (int i = p_a.limbs.size() - 1; i >= 0; i--) {
		if (p_a.limbs[i] > p_b.limbs[i]) return 1;
		if (p_a.limbs[i] < p_b.limbs[i]) return -1;
	}
	return 0;
}

// Internal Helper: Core Addition Logic (Magnitudes only)
void BigIntCore::_add_absolute(const BigIntCore &p_a, const BigIntCore &p_b) {
	int size_a = p_a.limbs.size();
	int size_b = p_b.limbs.size();
	int max_size = MAX(size_a, size_b);
	limbs.resize(max_size);
	
	WideLimb carry = 0;
	for (int i = 0; i < max_size; i++) {
		WideLimb sum = carry;
		if (i < size_a) sum += p_a.limbs[i];
		if (i < size_b) sum += p_b.limbs[i];
		limbs.write[i] = (Limb)sum;
		carry = sum >> 64;
	}
	if (carry) {
		limbs.push_back((Limb)carry);
	}
}

// Internal Helper: Core Subtraction Logic (Magnitudes only, assumes a >= b)
void BigIntCore::_sub_absolute(const BigIntCore &p_a, const BigIntCore &p_b) {
	int size_a = p_a.limbs.size();
	int size_b = p_b.limbs.size();
	limbs.resize(size_a);
	
	int64_t borrow = 0;
	for (int i = 0; i < size_a; i++) {
		WideLimb val_a = p_a.limbs[i];
		WideLimb val_b = (i < size_b) ? p_b.limbs[i] : 0;
		if (val_a < val_b + borrow) {
			limbs.write[i] = (Limb)((val_a + ((WideLimb)1 << 64)) - val_b - borrow);
			borrow = 1;
		} else {
			limbs.write[i] = (Limb)(val_a - val_b - borrow);
			borrow = 0;
		}
	}
	_trim();
}

BigIntCore::BigIntCore() : negative(false) {}

BigIntCore::BigIntCore(int64_t p_value) {
	negative = p_value < 0;
	uint64_t abs_val = (uint64_t)(negative ? -p_value : p_value);
	if (abs_val > 0) {
		limbs.push_back(abs_val);
	}
}

BigIntCore::BigIntCore(const String &p_string) : negative(false) {
	BigIntCore result(0);
	BigIntCore base(10);
	String s = p_string.strip_edges();
	int start = 0;
	if (s.length() > 0 && s[0] == '-') {
		start = 1;
	}
	
	for (int i = start; i < s.length(); i++) {
		char32_t c = s[i];
		if (c >= '0' && c <= '9') {
			result = (result * base) + BigIntCore((int64_t)(c - '0'));
		}
	}
	*this = result;
	if (start == 1 && !is_zero()) {
		negative = true;
	}
}

BigIntCore::BigIntCore(const BigIntCore &p_other) {
	limbs = p_other.limbs;
	negative = p_other.negative;
}

BigIntCore &BigIntCore::operator=(const BigIntCore &p_other) {
	limbs = p_other.limbs;
	negative = p_other.negative;
	return *this;
}

bool BigIntCore::operator<(const BigIntCore &p_other) const {
	if (negative != p_other.negative) return negative;
	int cmp = _compare_absolute(*this, p_other);
	return negative ? (cmp > 0) : (cmp < -0);
}

bool BigIntCore::operator==(const BigIntCore &p_other) const {
	return negative == p_other.negative && limbs == p_other.limbs;
}

BigIntCore BigIntCore::operator+(const BigIntCore &p_other) const {
	BigIntCore res;
	if (negative == p_other.negative) {
		res._add_absolute(*this, p_other);
		res.negative = negative;
	} else {
		int cmp = _compare_absolute(*this, p_other);
		if (cmp >= 0) {
			res._sub_absolute(*this, p_other);
			res.negative = negative;
		} else {
			res._sub_absolute(p_other, *this);
			res.negative = p_other.negative;
		}
	}
	res._trim();
	return res;
}

BigIntCore BigIntCore::operator-(const BigIntCore &p_other) const {
	BigIntCore neg_b = p_other;
	neg_b.negative = !p_other.negative;
	return *this + neg_b;
}

BigIntCore BigIntCore::operator*(const BigIntCore &p_other) const {
	if (is_zero() || p_other.is_zero()) return BigIntCore(0);
	
	BigIntCore res;
	res.limbs.resize(limbs.size() + p_other.limbs.size());
	for (int i = 0; i < res.limbs.size(); i++) res.limbs.write[i] = 0;
	
	for (int i = 0; i < limbs.size(); i++) {
		WideLimb carry = 0;
		for (int j = 0; j < p_other.limbs.size(); j++) {
			WideLimb cur = (WideLimb)res.limbs[i + j] + (WideLimb)limbs[i] * p_other.limbs[j] + carry;
			res.limbs.write[i + j] = (Limb)cur;
			carry = cur >> 64;
		}
		res.limbs.write[i + p_other.limbs.size()] = (Limb)carry;
	}
	
	res.negative = negative != p_other.negative;
	res._trim();
	return res;
}

void BigIntCore::div_mod(const BigIntCore &p_dividend, const BigIntCore &p_divisor, BigIntCore &r_quotient, BigIntCore &r_remainder) {
	if (p_divisor.is_zero()) return; // Should handle error
	
	int cmp = _compare_absolute(p_dividend, p_divisor);
	if (cmp < 0) {
		r_quotient = BigIntCore(0);
		r_remainder = p_dividend;
		return;
	}
	
	// Binary Long Division (Basic implementation for 120fps safety, can be optimized with Knuth D)
	BigIntCore q(0), r(0);
	for (int i = p_dividend.get_bit_length() - 1; i >= 0; i--) {
		r <<= 1;
		if ((p_dividend.limbs[i / 64] >> (i % 64)) & 1) {
			r.limbs.write[0] |= 1;
		}
		r._trim();
		if (_compare_absolute(r, p_divisor) >= 0) {
			r = r - p_divisor;
			if (q.limbs.size() <= (uint32_t)i / 64) q.limbs.resize(i / 64 + 1);
			q.limbs.write[i / 64] |= ((uint64_t)1 << (i % 64));
		}
	}
	
	r_quotient = q;
	r_quotient.negative = p_dividend.negative != p_divisor.negative;
	r_remainder = r;
	r_remainder.negative = p_dividend.negative;
	r_quotient._trim();
	r_remainder._trim();
}

BigIntCore BigIntCore::operator/(const BigIntCore &p_other) const {
	BigIntCore q, r;
	div_mod(*this, p_other, q, r);
	return q;
}

BigIntCore BigIntCore::operator%(const BigIntCore &p_other) const {
	BigIntCore q, r;
	div_mod(*this, p_other, q, r);
	return r;
}

BigIntCore BigIntCore::operator<<(uint32_t p_shift) const {
	if (p_shift == 0) return *this;
	BigIntCore res = *this;
	uint32_t limb_shift = p_shift / 64;
	uint32_t bit_shift = p_shift % 64;
	
	if (limb_shift > 0) {
		res.limbs.resize(res.limbs.size() + limb_shift);
		for (int i = res.limbs.size() - 1; i >= (int)limb_shift; i--) {
			res.limbs.write[i] = res.limbs[i - limb_shift];
		}
		for (int i = 0; i < (int)limb_shift; i++) res.limbs.write[i] = 0;
	}
	
	if (bit_shift > 0) {
		Limb carry = 0;
		for (int i = 0; i < res.limbs.size(); i++) {
			Limb next_carry = res.limbs[i] >> (64 - bit_shift);
			res.limbs.write[i] = (res.limbs[i] << bit_shift) | carry;
			carry = next_carry;
		}
		if (carry) res.limbs.push_back(carry);
	}
	return res;
}

BigIntCore BigIntCore::operator>>(uint32_t p_shift) const {
	if (p_shift == 0) return *this;
	BigIntCore res = *this;
	uint32_t limb_shift = p_shift / 64;
	uint32_t bit_shift = p_shift % 64;
	
	if (limb_shift >= (uint32_t)res.limbs.size()) return BigIntCore(0);
	
	if (limb_shift > 0) {
		for (int i = 0; i < res.limbs.size() - (int)limb_shift; i++) {
			res.limbs.write[i] = res.limbs[i + limb_shift];
		}
		res.limbs.resize(res.limbs.size() - limb_shift);
	}
	
	if (bit_shift > 0) {
		Limb carry = 0;
		for (int i = res.limbs.size() - 1; i >= 0; i--) {
			Limb next_carry = res.limbs[i] << (64 - bit_shift);
			res.limbs.write[i] = (res.limbs[i] >> bit_shift) | carry;
			carry = next_carry;
		}
	}
	res._trim();
	return res;
}

uint32_t BigIntCore::get_bit_length() const {
	if (is_zero()) return 0;
	uint32_t len = (limbs.size() - 1) * 64;
	Limb last = limbs[limbs.size() - 1];
	while (last > 0) {
		last >>= 1;
		len++;
	}
	return len;
}

bool BigIntCore::is_zero() const {
	return limbs.size() == 0;
}

bool BigIntCore::is_negative() const {
	return negative;
}

String BigIntCore::to_string() const {
	if (is_zero()) return "0";
	String s = "";
	BigIntCore temp = *this;
	temp.negative = false;
	BigIntCore base(10);
	
	while (!temp.is_zero()) {
		BigIntCore q, r;
		div_mod(temp, base, q, r);
		s = String::num_int64(r.limbs.size() > 0 ? r.limbs[0] : 0) + s;
		temp = q;
	}
	
	if (negative) s = "-" + s;
	return s;
}

int64_t BigIntCore::to_int64() const {
	if (is_zero()) return 0;
	int64_t val = (int64_t)limbs[0];
	return negative ? -val : val;
}

// Global operator overloads for procedural math logic
BigIntCore &BigIntCore::operator<<=(uint32_t p_shift) { *this = *this << p_shift; return *this; }
BigIntCore &BigIntCore::operator*=(const BigIntCore &p_other) { *this = *this * p_other; return *this; }
BigIntCore &BigIntCore::operator+=(const BigIntCore &p_other) { *this = *this + p_other; return *this; }