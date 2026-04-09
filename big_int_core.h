/**************************************************************************/
/*  big_int_core.h                                                        */
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

#ifndef BIG_INT_CORE_H
#define BIG_INT_CORE_H

#include "core/typedefs.h"
#include "core/templates/vector.h"
#include "core/string/ustring.h"

/**
 * @class BigIntCore
 * @brief A heavy-duty, high-performance arbitrary-precision integer implementation.
 * 
 * Designed for 120fps real-time simulations ranging from microscopic to galactic scales.
 * This class ensures bit-perfect determinism across different architectures (32/64-bit)
 * by avoiding floating-point math and managing overflows via a dynamic limb-based system.
 */

class BigIntCore {
	// Limb type: 64-bit unsigned integer provides the best performance on modern CPUs
	typedef uint64_t Limb;
	typedef __int128_t WideLimb; // Used for intermediate multiplication/division steps

	Vector<Limb> limbs;
	bool negative;

	// Internal helper functions for heavy math logic
	void _trim();
	static int _compare_absolute(const BigIntCore &p_a, const BigIntCore &p_b);
	void _add_absolute(const BigIntCore &p_a, const BigIntCore &p_b);
	void _sub_absolute(const BigIntCore &p_a, const BigIntCore &p_b);
	static void _multiply_limb(const BigIntCore &p_a, Limb p_limb, BigIntCore &r_dest);

public:
	// Constructors
	BigIntCore();
	BigIntCore(int64_t p_value);
	BigIntCore(const String &p_string);
	BigIntCore(const BigIntCore &p_other);

	// Assignment
	BigIntCore &operator=(const BigIntCore &p_other);
	BigIntCore &operator=(int64_t p_value);

	// Comparison Operators
	bool operator==(const BigIntCore &p_other) const;
	bool operator!=(const BigIntCore &p_other) const;
	bool operator<(const BigIntCore &p_other) const;
	bool operator<=(const BigIntCore &p_other) const;
	bool operator>(const BigIntCore &p_other) const;
	bool operator>=(const BigIntCore &p_other) const;

	// Arithmetic Operators
	BigIntCore operator+(const BigIntCore &p_other) const;
	BigIntCore operator-(const BigIntCore &p_other) const;
	BigIntCore operator*(const BigIntCore &p_other) const;
	BigIntCore operator/(const BigIntCore &p_other) const;
	BigIntCore operator%(const BigIntCore &p_other) const;
	BigIntCore operator-() const;

	BigIntCore &operator+=(const BigIntCore &p_other);
	BigIntCore &operator-=(const BigIntCore &p_other);
	BigIntCore &operator*=(const BigIntCore &p_other);
	BigIntCore &operator/=(const BigIntCore &p_other);
	BigIntCore &operator%=(const BigIntCore &p_other);

	// Bitwise Operators
	BigIntCore operator<<(uint32_t p_shift) const;
	BigIntCore operator>>(uint32_t p_shift) const;
	BigIntCore &operator<<=(uint32_t p_shift);
	BigIntCore &operator>>=(uint32_t p_shift);
	BigIntCore operator&(const BigIntCore &p_other) const;
	BigIntCore operator|(const BigIntCore &p_other) const;
	BigIntCore operator^(const BigIntCore &p_other) const;
	BigIntCore operator~() const;

	// Conversion and Utilities
	String to_string() const;
	int64_t to_int64() const;
	double to_double() const; // For visualization only, not for simulation logic
	bool is_zero() const;
	bool is_negative() const;
	uint32_t get_bit_length() const;
	
	// Specialized math
	static BigIntCore pow(const BigIntCore &p_base, uint32_t p_exp);
	static BigIntCore sqrt(const BigIntCore &p_val);
	static void div_mod(const BigIntCore &p_dividend, const BigIntCore &p_divisor, BigIntCore &r_quotient, BigIntCore &r_remainder);

	// Memory Management for 120fps
	void reserve(uint32_t p_limbs);
	void clear();
};

#endif // BIG_INT_CORE_H