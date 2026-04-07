--- START OF FILE src/fixed_math_core.h ---

#ifndef FIXED_MATH_CORE_H
#define FIXED_MATH_CORE_H

#include "core/typedefs.h"
#include <string>

/**
 * FixedMathCore (Q32.32)
 * 
 * The central deterministic authority for continuous physical values.
 * Replaces 'float' and 'double' to guarantee bit-perfect 120 FPS simulation.
 * Aligned for SIMD/Warp performance and EnTT SoA packing.
 */
struct ET_ALIGN_32 FixedMathCore {
private:
	int64_t raw_value;

	// Internal constructor for raw bit-injection
	_FORCE_INLINE_ explicit FixedMathCore(int64_t p_raw, bool p_is_raw) : raw_value(p_raw) {}

public:
	// ------------------------------------------------------------------------
	// Constants (Bit-Perfect Q32.32)
	// ------------------------------------------------------------------------
	static const int64_t FRACTIONAL_BITS = 32;
	static const int64_t ONE_RAW = 1LL << 32;
	static const int64_t HALF_RAW = 1LL << 31;
	
	// Pi = 3.1415926535... * 2^32
	static const int64_t PI_RAW = 13493037704LL;
	static const int64_t TWO_PI_RAW = 26986075409LL;
	static const int64_t HALF_PI_RAW = 6746518852LL;
	static const int64_t E_RAW = 11670868155LL;

	// ------------------------------------------------------------------------
	// Constructors
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ FixedMathCore() : raw_value(0) {}
	_FORCE_INLINE_ FixedMathCore(int32_t p_val) : raw_value(static_cast<int64_t>(p_val) << 32) {}
	_FORCE_INLINE_ FixedMathCore(int64_t p_val) : raw_value(p_val << 32) {}
	FixedMathCore(const std::string &p_val);
	_FORCE_INLINE_ FixedMathCore(const FixedMathCore &p_other) : raw_value(p_other.raw_value) {}

	// ------------------------------------------------------------------------
	// Arithmetic (Warp-Kernel Optimized)
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ FixedMathCore operator+(const FixedMathCore &p_other) const { return FixedMathCore(raw_value + p_other.raw_value, true); }
	_FORCE_INLINE_ FixedMathCore operator-(const FixedMathCore &p_other) const { return FixedMathCore(raw_value - p_other.raw_value, true); }
	_FORCE_INLINE_ FixedMathCore operator-() const { return FixedMathCore(-raw_value, true); }

	/**
	 * operator*
	 * Uses 128-bit intermediate storage to prevent 64-bit overflow during 
	 * high-impulse collisions or relativistic velocity calculations.
	 */
	_FORCE_INLINE_ FixedMathCore operator*(const FixedMathCore &p_other) const {
		__int128_t res = static_cast<__int128_t>(raw_value) * static_cast<__int128_t>(p_other.raw_value);
		return FixedMathCore(static_cast<int64_t>(res >> 32), true);
	}

	_FORCE_INLINE_ FixedMathCore operator/(const FixedMathCore &p_other) const {
		CRASH_COND_MSG(p_other.raw_value == 0, "FixedMathCore: Division by zero.");
		__int128_t res = (static_cast<__int128_t>(raw_value) << 32) / p_other.raw_value;
		return FixedMathCore(static_cast<int64_t>(res), true);
	}

	_FORCE_INLINE_ FixedMathCore operator%(const FixedMathCore &p_other) const {
		return FixedMathCore(raw_value % p_other.raw_value, true);
	}

	_FORCE_INLINE_ FixedMathCore& operator+=(const FixedMathCore &p_other) { raw_value += p_other.raw_value; return *this; }
	_FORCE_INLINE_ FixedMathCore& operator-=(const FixedMathCore &p_other) { raw_value -= p_other.raw_value; return *this; }
	_FORCE_INLINE_ FixedMathCore& operator*=(const FixedMathCore &p_other) { *this = *this * p_other; return *this; }
	_FORCE_INLINE_ FixedMathCore& operator/=(const FixedMathCore &p_other) { *this = *this / p_other; return *this; }

	// ------------------------------------------------------------------------
	// Comparison
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ bool operator==(const FixedMathCore &p_other) const { return raw_value == p_other.raw_value; }
	_FORCE_INLINE_ bool operator!=(const FixedMathCore &p_other) const { return raw_value != p_other.raw_value; }
	_FORCE_INLINE_ bool operator<(const FixedMathCore &p_other) const { return raw_value < p_other.raw_value; }
	_FORCE_INLINE_ bool operator<=(const FixedMathCore &p_other) const { return raw_value <= p_other.raw_value; }
	_FORCE_INLINE_ bool operator>(const FixedMathCore &p_other) const { return raw_value > p_other.raw_value; }
	_FORCE_INLINE_ bool operator>=(const FixedMathCore &p_other) const { return raw_value >= p_other.raw_value; }

	// ------------------------------------------------------------------------
	// Advanced Sophisticated Math (Non-FPU Transcendental)
	// ------------------------------------------------------------------------
	FixedMathCore sin() const;
	FixedMathCore cos() const;
	FixedMathCore tan() const;
	FixedMathCore atan2(const FixedMathCore &p_x) const;
	FixedMathCore square_root() const;
	FixedMathCore power(int32_t p_exp) const;
	FixedMathCore exp() const; // For Atmospheric Scattering and Damping
	FixedMathCore log() const; // For Shannon Entropy

	_FORCE_INLINE_ FixedMathCore absolute() const { return FixedMathCore(raw_value < 0 ? -raw_value : raw_value, true); }
	
	// ------------------------------------------------------------------------
	// Utility & Hashing
	// ------------------------------------------------------------------------
	_FORCE_INLINE_ int64_t get_raw() const { return raw_value; }
	_FORCE_INLINE_ int64_t to_int() const { return raw_value >> 32; }
	std::string to_string() const;

	/**
	 * hash()
	 * High-entropy Murmur-style hash for EnTT Component lookup.
	 */
	_FORCE_INLINE_ uint32_t hash() const {
		uint64_t v = static_cast<uint64_t>(raw_value);
		v ^= v >> 33;
		v *= 0xff51afd7ed558ccdULL;
		v ^= v >> 33;
		v *= 0xc4ceb9fe1a85ec53ULL;
		v ^= v >> 33;
		return static_cast<uint32_t>(v);
	}

	static _FORCE_INLINE_ FixedMathCore pi() { return FixedMathCore(PI_RAW, true); }
	static _FORCE_INLINE_ FixedMathCore e() { return FixedMathCore(E_RAW, true); }
};

/**
 * MathConstants Template
 * Provides type-agnostic access to zero, one, and half for math templates.
 */
template <typename T>
struct MathConstants {
	static _FORCE_INLINE_ T zero() { return T(0LL); }
	static _FORCE_INLINE_ T one() { return T(1LL); }
	static _FORCE_INLINE_ T half() { return T(0LL); } // Overridden in specialization
};

template <>
struct MathConstants<FixedMathCore> {
	static _FORCE_INLINE_ FixedMathCore zero() { return FixedMathCore(0LL, true); }
	static _FORCE_INLINE_ FixedMathCore one() { return FixedMathCore(FixedMathCore::ONE_RAW, true); }
	static _FORCE_INLINE_ FixedMathCore half() { return FixedMathCore(FixedMathCore::HALF_RAW, true); }
};

#endif // FIXED_MATH_CORE_H

--- END OF FILE src/fixed_math_core.h ---
