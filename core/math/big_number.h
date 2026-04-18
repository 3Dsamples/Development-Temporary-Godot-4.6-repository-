// src/big_number.h
#ifndef BIG_NUMBER_H
#define BIG_NUMBER_H

#include "big_int_core.h"
#include "fixed_math_core.h"
#include <string>
#include <type_traits>
#include <limits>
#include <cmath>

// xtensor integration (if available)
#ifdef UEP_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#endif

namespace uep {

// ----------------------------------------------------------------------------
// BigNumber: Unified scalar type combining arbitrary-precision integer
// capabilities with deterministic fixed-point arithmetic.
//
// Internally, a BigNumber is represented as a BigIntCore scaled by a fixed
// power of two (SCALE_BITS = 32). This corresponds to Q-format fixed-point
// with 32 fractional bits, identical to FixedMathCore's representation.
// The difference is that the underlying integer can grow arbitrarily large,
// providing both the speed of hardware-accelerated limbs and unlimited range.
//
// For xtensor integration, BigNumber acts as a leaf node that exposes its
// limb data as a 1D tensor, allowing expression fusion at the limb level.
// ----------------------------------------------------------------------------
class BigNumber {
public:
    static constexpr int SCALE_BITS = 32;
    static constexpr uint64_t SCALE = uint64_t(1) << SCALE_BITS; // 2^32

private:
    BigIntCore value_;   // Scaled value: real_number = value_ / 2^32

    // Private constructor from BigIntCore (assumed already scaled)
    explicit _FORCE_INLINE_ BigNumber(const BigIntCore& scaled, bool) : value_(scaled) {}
    explicit _FORCE_INLINE_ BigNumber(BigIntCore&& scaled, bool) : value_(std::move(scaled)) {}

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ BigNumber() : value_() {}

    // Integer constructors (scale automatically)
    _FORCE_INLINE_ BigNumber(int val) : value_(BigIntCore(val) << SCALE_BITS) {}
    _FORCE_INLINE_ BigNumber(long val) : value_(BigIntCore(val) << SCALE_BITS) {}
    _FORCE_INLINE_ BigNumber(long long val) : value_(BigIntCore(val) << SCALE_BITS) {}
    _FORCE_INLINE_ BigNumber(unsigned int val) : value_(BigIntCore(val) << SCALE_BITS) {}
    _FORCE_INLINE_ BigNumber(unsigned long val) : value_(BigIntCore(val) << SCALE_BITS) {}
    _FORCE_INLINE_ BigNumber(unsigned long long val) : value_(BigIntCore(val) << SCALE_BITS) {}

    // Floating point constructors (convert to scaled integer)
    _FORCE_INLINE_ BigNumber(float val) : value_(BigIntCore(int64_t(val * SCALE))) {}
    _FORCE_INLINE_ BigNumber(double val) : value_(BigIntCore(int64_t(val * SCALE))) {}
    _FORCE_INLINE_ BigNumber(long double val) : value_(BigIntCore(int64_t(val * SCALE))) {}

    // Construct from FixedMathCore (trivial conversion)
    _FORCE_INLINE_ BigNumber(const FixedMathCore& fm) : value_(BigIntCore(fm.raw())) {}

    // Construct from BigIntCore (treat as scaled integer; if you want integer, shift)
    _FORCE_INLINE_ BigNumber(const BigIntCore& bi) : value_(bi) {}

    // Copy/Move
    _FORCE_INLINE_ BigNumber(const BigNumber&) = default;
    _FORCE_INLINE_ BigNumber(BigNumber&&) = default;
    _FORCE_INLINE_ BigNumber& operator=(const BigNumber&) = default;
    _FORCE_INLINE_ BigNumber& operator=(BigNumber&&) = default;

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ const BigIntCore& scaled_value() const { return value_; }
    _FORCE_INLINE_ BigIntCore& scaled_value() { return value_; }
    _FORCE_INLINE_ bool is_zero() const { return value_.is_zero(); }
    _FORCE_INLINE_ bool is_negative() const { return value_.is_negative(); }

    // ------------------------------------------------------------------------
    // Conversion to primitive types
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ int64_t to_int64() const {
        // Divide by SCALE (shift right 32 bits) using BigIntCore's shift operator
        BigIntCore int_part = value_ >> SCALE_BITS;
        // Check if it fits in signed 64-bit
        if (int_part > BigIntCore(INT64_MAX)) return INT64_MAX;
        if (int_part < BigIntCore(INT64_MIN)) return INT64_MIN;
        return int_part.to_int64();
    }

    _FORCE_INLINE_ double to_double() const {
        // Approximate conversion using high 64 bits for speed
        // For exact conversion, we'd need to handle arbitrary length.
        // Use BigIntCore::to_string and parse? That's slow.
        // We'll implement a more precise method in the .cpp if needed.
        // For inline, use the most significant 64 bits and the scale.
        if (value_.size() <= 2) {
            // Fits in 128 bits, we can use __int128
            __int128 raw = 0;
            if (value_.size() >= 1) raw |= value_.data()[0];
            if (value_.size() >= 2) raw |= __int128(value_.data()[1]) << 64;
            return double(raw) / SCALE;
        }
        // Fallback for larger numbers: compute using log2 and exponent.
        // We'll keep a simplified version here; full implementation in .cpp.
        // For now, return approximation from highest 64 bits.
        const limb_t* limbs = value_.data();
        size_t sz = value_.size();
        double result = 0.0;
        double base = 1.0;
        for (size_t i = 0; i < sz && i < 4; ++i) {
            result += double(limbs[i]) * base;
            base *= 18446744073709551616.0; // 2^64
        }
        // Adjust for higher limbs
        if (sz > 4) {
            result *= pow(2.0, 64 * (sz - 4));
        }
        return result / SCALE;
    }

    _FORCE_INLINE_ float to_float() const { return float(to_double()); }

    // Convert to FixedMathCore (may lose precision if value out of 64-bit range)
    _FORCE_INLINE_ FixedMathCore to_fixed() const {
        // Extract the low 64 bits of the scaled value
        uint64_t low = value_.to_uint64(); // returns least significant 64 bits
        return FixedMathCore::from_raw(static_cast<int64_t>(low));
    }

    // ------------------------------------------------------------------------
    // Arithmetic operators (inlined, using BigIntCore)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ BigNumber operator-() const {
        return BigNumber(-value_, true);
    }

    _FORCE_INLINE_ BigNumber operator+(const BigNumber& other) const {
        return BigNumber(value_ + other.value_, true);
    }

    _FORCE_INLINE_ BigNumber operator-(const BigNumber& other) const {
        return BigNumber(value_ - other.value_, true);
    }

    _FORCE_INLINE_ BigNumber operator*(const BigNumber& other) const {
        // (a/2^32) * (b/2^32) = (a*b) / 2^64, need to shift right by 32 to renormalize
        BigIntCore product = value_ * other.value_;
        product >>= SCALE_BITS; // Divide by 2^32
        return BigNumber(product, true);
    }

    _FORCE_INLINE_ BigNumber operator/(const BigNumber& other) const {
        // (a/2^32) / (b/2^32) = a / b, then multiply by 2^32 to keep scale
        if (other.is_zero()) return BigNumber(); // division by zero
        BigIntCore quotient = (value_ << SCALE_BITS) / other.value_;
        return BigNumber(quotient, true);
    }

    _FORCE_INLINE_ BigNumber& operator+=(const BigNumber& other) {
        value_ += other.value_;
        return *this;
    }

    _FORCE_INLINE_ BigNumber& operator-=(const BigNumber& other) {
        value_ -= other.value_;
        return *this;
    }

    _FORCE_INLINE_ BigNumber& operator*=(const BigNumber& other) {
        value_ = (value_ * other.value_) >> SCALE_BITS;
        return *this;
    }

    _FORCE_INLINE_ BigNumber& operator/=(const BigNumber& other) {
        if (other.is_zero()) {
            value_ = BigIntCore();
            return *this;
        }
        value_ = (value_ << SCALE_BITS) / other.value_;
        return *this;
    }

    // Modulo operation (remainder after division)
    _FORCE_INLINE_ BigNumber operator%(const BigNumber& other) const {
        if (other.is_zero()) return BigNumber();
        BigIntCore rem = value_ % other.value_;
        return BigNumber(rem, true);
    }

    _FORCE_INLINE_ BigNumber& operator%=(const BigNumber& other) {
        if (other.is_zero()) {
            value_ = BigIntCore();
        } else {
            value_ %= other.value_;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Comparison operators
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ bool operator==(const BigNumber& other) const { return value_ == other.value_; }
    _FORCE_INLINE_ bool operator!=(const BigNumber& other) const { return value_ != other.value_; }
    _FORCE_INLINE_ bool operator<(const BigNumber& other) const  { return value_ < other.value_; }
    _FORCE_INLINE_ bool operator>(const BigNumber& other) const  { return value_ > other.value_; }
    _FORCE_INLINE_ bool operator<=(const BigNumber& other) const { return value_ <= other.value_; }
    _FORCE_INLINE_ bool operator>=(const BigNumber& other) const { return value_ >= other.value_; }

    // ------------------------------------------------------------------------
    // Bitwise operations (operate on scaled integer representation)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ BigNumber operator&(const BigNumber& other) const { return BigNumber(value_ & other.value_, true); }
    _FORCE_INLINE_ BigNumber operator|(const BigNumber& other) const { return BigNumber(value_ | other.value_, true); }
    _FORCE_INLINE_ BigNumber operator^(const BigNumber& other) const { return BigNumber(value_ ^ other.value_, true); }
    _FORCE_INLINE_ BigNumber operator~() const { return BigNumber(~value_, true); }
    _FORCE_INLINE_ BigNumber operator<<(size_t bits) const { return BigNumber(value_ << bits, true); }
    _FORCE_INLINE_ BigNumber operator>>(size_t bits) const { return BigNumber(value_ >> bits, true); }
    _FORCE_INLINE_ BigNumber& operator&=(const BigNumber& other) { value_ &= other.value_; return *this; }
    _FORCE_INLINE_ BigNumber& operator|=(const BigNumber& other) { value_ |= other.value_; return *this; }
    _FORCE_INLINE_ BigNumber& operator^=(const BigNumber& other) { value_ ^= other.value_; return *this; }
    _FORCE_INLINE_ BigNumber& operator<<=(size_t bits) { value_ <<= bits; return *this; }
    _FORCE_INLINE_ BigNumber& operator>>=(size_t bits) { value_ >>= bits; return *this; }

    // ------------------------------------------------------------------------
    // Increment/Decrement (affects the scaled integer; +1 means +1/2^32)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ BigNumber& operator++() { value_ += BigIntCore(1); return *this; }
    _FORCE_INLINE_ BigNumber operator++(int) { BigNumber tmp(*this); ++*this; return tmp; }
    _FORCE_INLINE_ BigNumber& operator--() { value_ -= BigIntCore(1); return *this; }
    _FORCE_INLINE_ BigNumber operator--(int) { BigNumber tmp(*this); --*this; return tmp; }

    // ------------------------------------------------------------------------
    // Mathematical functions (delegated to FixedMathCore for basic functions,
    // but we can also implement arbitrary-precision versions)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ BigNumber abs() const {
        return BigNumber(value_.is_negative() ? -value_ : value_, true);
    }

    // Floor: discard fractional part (round toward -inf)
    _FORCE_INLINE_ BigNumber floor() const {
        // Clear lower SCALE_BITS bits
        BigIntCore int_part = value_ >> SCALE_BITS;
        return BigNumber(int_part << SCALE_BITS, true);
    }

    // Ceil: round toward +inf
    _FORCE_INLINE_ BigNumber ceil() const {
        BigIntCore frac_mask = (BigIntCore(1) << SCALE_BITS) - 1;
        if ((value_ & frac_mask).is_zero()) return *this; // already integer
        BigIntCore int_part = value_ >> SCALE_BITS;
        return BigNumber((int_part + (value_.is_negative() ? 0 : 1)) << SCALE_BITS, true);
    }

    // Round to nearest integer
    _FORCE_INLINE_ BigNumber round() const {
        BigIntCore half = BigIntCore(1) << (SCALE_BITS - 1);
        BigIntCore int_part = value_ >> SCALE_BITS;
        BigIntCore frac = value_ & ((BigIntCore(1) << SCALE_BITS) - 1);
        if (frac < half) {
            return BigNumber(int_part << SCALE_BITS, true);
        } else if (frac > half) {
            return BigNumber((int_part + 1) << SCALE_BITS, true);
        } else {
            // Tie-breaking: round to even
            if ((int_part & 1) == 0) {
                return BigNumber(int_part << SCALE_BITS, true);
            } else {
                return BigNumber((int_part + 1) << SCALE_BITS, true);
            }
        }
    }

    // Fractional part
    _FORCE_INLINE_ BigNumber frac() const {
        return BigNumber(value_ & ((BigIntCore(1) << SCALE_BITS) - 1), true);
    }

    // ------------------------------------------------------------------------
    // String conversion (delegated to .cpp for heavy implementation)
    // ------------------------------------------------------------------------
    std::string to_string() const;
    static BigNumber from_string(const std::string& str);

    // ------------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------------
    static BigNumber pi();
    static BigNumber e();
    static BigNumber ln2();

    // ------------------------------------------------------------------------
    // Transcendental functions (implemented in .cpp using arbitrary precision)
    // ------------------------------------------------------------------------
    static BigNumber sin(const BigNumber& x);
    static BigNumber cos(const BigNumber& x);
    static BigNumber tan(const BigNumber& x);
    static BigNumber asin(const BigNumber& x);
    static BigNumber acos(const BigNumber& x);
    static BigNumber atan(const BigNumber& x);
    static BigNumber atan2(const BigNumber& y, const BigNumber& x);
    static BigNumber exp(const BigNumber& x);
    static BigNumber log(const BigNumber& x);
    static BigNumber log10(const BigNumber& x);
    static BigNumber pow(const BigNumber& base, const BigNumber& exp);
    static BigNumber sqrt(const BigNumber& x);

    // ------------------------------------------------------------------------
    // Xtensor integration: expose limb data as expression node
    // ------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR
    // Return a 1D xtensor view of the underlying limbs (const)
    auto xt_limbs() const {
        return xt::adapt(value_.data(), value_.size(), xt::no_ownership(), std::vector<size_t>{value_.size()});
    }

    // Mutable version (allows in-place operations via xtensor)
    auto xt_limbs() {
        return xt::adapt(value_.data(), value_.size(), xt::no_ownership(), std::vector<size_t>{value_.size()});
    }

    // Convert BigNumber to an xtensor expression (for lazy evaluation)
    // This returns a proxy that can participate in expression templates.
    auto to_xtensor() const {
        return xt_limbs();
    }

    // Create a BigNumber from an xtensor expression (evaluated)
    template<typename E>
    static BigNumber from_xtensor(const xt::xexpression<E>& expr) {
        const auto& e = expr.derived_cast();
        BigNumber result;
        // Evaluate the expression and copy limbs
        auto arr = xt::eval(e);
        result.value_.resize(arr.size());
        std::copy(arr.begin(), arr.end(), result.value_.data());
        result.value_.normalize();
        return result;
    }
#endif

    // ------------------------------------------------------------------------
    // Friend operators for mixed-type arithmetic (optimized)
    // ------------------------------------------------------------------------
    friend _FORCE_INLINE_ BigNumber operator+(int64_t lhs, const BigNumber& rhs) {
        return BigNumber(lhs) + rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator-(int64_t lhs, const BigNumber& rhs) {
        return BigNumber(lhs) - rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator*(int64_t lhs, const BigNumber& rhs) {
        return BigNumber(lhs) * rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator/(int64_t lhs, const BigNumber& rhs) {
        return BigNumber(lhs) / rhs;
    }
    // Also for double (converted to BigNumber)
    friend _FORCE_INLINE_ BigNumber operator+(double lhs, const BigNumber& rhs) {
        return BigNumber(lhs) + rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator-(double lhs, const BigNumber& rhs) {
        return BigNumber(lhs) - rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator*(double lhs, const BigNumber& rhs) {
        return BigNumber(lhs) * rhs;
    }
    friend _FORCE_INLINE_ BigNumber operator/(double lhs, const BigNumber& rhs) {
        return BigNumber(lhs) / rhs;
    }
};

// ----------------------------------------------------------------------------
// Inline constant getters
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber BigNumber::pi() {
    // π * 2^32 ≈ 0x3243F6A8885A308D
    static const BigNumber pi_val = BigNumber(FixedMathCore::pi());
    return pi_val;
}

_FORCE_INLINE_ BigNumber BigNumber::e() {
    static const BigNumber e_val = BigNumber(FixedMathCore::e());
    return e_val;
}

_FORCE_INLINE_ BigNumber BigNumber::ln2() {
    static const BigNumber ln2_val = BigNumber(FixedMathCore::ln2());
    return ln2_val;
}

// ----------------------------------------------------------------------------
// Additional utility operators (e.g., literal suffixes)
// ----------------------------------------------------------------------------
inline BigNumber operator"" _bn(unsigned long long val) {
    return BigNumber(static_cast<int64_t>(val));
}

inline BigNumber operator"" _bn(long double val) {
    return BigNumber(static_cast<double>(val));
}

} // namespace uep

#endif // BIG_NUMBER_H
// Ending of Part 1 of 4 (big_number.h)

// src/big_number.h (continued from Part 1)

// ----------------------------------------------------------------------------
// Additional inline conversion to native types with full precision
// ----------------------------------------------------------------------------
_FORCE_INLINE_ int32_t to_int32() const {
    BigIntCore int_part = value_ >> SCALE_BITS;
    if (int_part > BigIntCore(INT32_MAX)) return INT32_MAX;
    if (int_part < BigIntCore(INT32_MIN)) return INT32_MIN;
    return static_cast<int32_t>(int_part.to_int64());
}

_FORCE_INLINE_ uint64_t to_uint64() const {
    // Extract integer part (floor for positive, ceil for negative? We'll do truncation toward zero)
    if (value_.is_negative()) {
        BigNumber abs_val = this->abs();
        uint64_t u = abs_val.to_uint64();
        return u; // actually returns absolute value's truncated integer part
    }
    BigIntCore int_part = value_ >> SCALE_BITS;
    return int_part.to_uint64();
}

// Full double conversion using arbitrary precision (iterative algorithm)
double BigNumber::to_double() const {
    if (value_.is_zero()) return 0.0;
    
    // Determine sign
    bool negative = value_.is_negative();
    BigIntCore abs_val = negative ? -value_ : value_;
    
    // We want to compute abs_val / 2^32 as a double.
    // Split abs_val into integer and fractional parts.
    BigIntCore int_part = abs_val >> SCALE_BITS;
    BigIntCore frac_part = abs_val & ((BigIntCore(1) << SCALE_BITS) - 1);
    
    // Convert integer part to double (may overflow if too large; we'll use log2 method)
    double int_dbl = 0.0;
    const limb_t* limbs = int_part.data();
    size_t sz = int_part.size();
    
    // Use double-double or iterative multiplication by 2^64
    // For large integers, we can use scaling via powers of two
    if (sz <= 4) {
        __int128 raw_int = 0;
        for (size_t i = 0; i < sz; ++i) {
            raw_int |= __int128(limbs[i]) << (64 * i);
        }
        int_dbl = double(raw_int);
    } else {
        // Use high-precision conversion: int_part = sum(limbs[i] * 2^(64*i))
        // Compute using log2 and exponent extraction
        // We'll approximate using the top 53 bits of mantissa
        // For exactness, we could use BigIntCore to compute integer part as double.
        // Here we implement a robust method:
        size_t msb_index = sz - 1;
        limb_t msb_limb = limbs[msb_index];
        int leading_zeros = __builtin_clzll(msb_limb);
        int total_bits = 64 * msb_index + (64 - leading_zeros);
        
        // Double can represent up to 1024 bits in exponent, but mantissa is 53 bits.
        // We'll extract the top 53 bits of the integer part.
        int shift = total_bits - 53;
        if (shift < 0) shift = 0;
        
        // Shift the whole number right by 'shift' bits to get a 53-bit integer
        BigIntCore shifted = int_part >> shift;
        uint64_t mantissa = shifted.to_uint64(); // Now fits in 53 bits
        // Adjust exponent: the value is mantissa * 2^shift
        int_dbl = std::ldexp(double(mantissa), shift);
    }
    
    // Convert fractional part: frac_part / 2^32
    // Since frac_part < 2^32, we can compute directly as double
    double frac_dbl = 0.0;
    if (!frac_part.is_zero()) {
        // Compute frac_part as integer then divide by 2^32
        uint64_t frac_raw = frac_part.to_uint64(); // guaranteed < 2^32
        frac_dbl = double(frac_raw) / double(SCALE);
    }
    
    double result = int_dbl + frac_dbl;
    return negative ? -result : result;
}

// ----------------------------------------------------------------------------
// Fast inline approximations for transcendental functions (using double)
// These provide a quick path when absolute precision isn't critical.
// Full arbitrary-precision implementations are in the .cpp file.
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber fast_sin() const {
    // Convert to double, compute sin, convert back
    return BigNumber(std::sin(this->to_double()));
}

_FORCE_INLINE_ BigNumber fast_cos() const {
    return BigNumber(std::cos(this->to_double()));
}

_FORCE_INLINE_ BigNumber fast_exp() const {
    return BigNumber(std::exp(this->to_double()));
}

_FORCE_INLINE_ BigNumber fast_log() const {
    return BigNumber(std::log(this->to_double()));
}

_FORCE_INLINE_ BigNumber fast_sqrt() const {
    return BigNumber(std::sqrt(this->to_double()));
}

// ----------------------------------------------------------------------------
// Inline mathematical utility functions (non-member)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber abs(const BigNumber& x) { return x.abs(); }
_FORCE_INLINE_ BigNumber floor(const BigNumber& x) { return x.floor(); }
_FORCE_INLINE_ BigNumber ceil(const BigNumber& x) { return x.ceil(); }
_FORCE_INLINE_ BigNumber round(const BigNumber& x) { return x.round(); }
_FORCE_INLINE_ BigNumber frac(const BigNumber& x) { return x.frac(); }

_FORCE_INLINE_ BigNumber min(const BigNumber& a, const BigNumber& b) {
    return a < b ? a : b;
}

_FORCE_INLINE_ BigNumber max(const BigNumber& a, const BigNumber& b) {
    return a > b ? a : b;
}

_FORCE_INLINE_ BigNumber clamp(const BigNumber& val, const BigNumber& lo, const BigNumber& hi) {
    return min(max(val, lo), hi);
}

_FORCE_INLINE_ BigNumber sign(const BigNumber& x) {
    if (x > BigNumber(0)) return BigNumber(1);
    if (x < BigNumber(0)) return BigNumber(-1);
    return BigNumber(0);
}

_FORCE_INLINE_ BigNumber lerp(const BigNumber& a, const BigNumber& b, const BigNumber& t) {
    return a + (b - a) * t;
}

// ----------------------------------------------------------------------------
// Mixed-type operators with primitive types (ensuring efficient conversion)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber operator+(const BigNumber& lhs, int rhs) { return lhs + BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator-(const BigNumber& lhs, int rhs) { return lhs - BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator*(const BigNumber& lhs, int rhs) { return lhs * BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator/(const BigNumber& lhs, int rhs) { return lhs / BigNumber(rhs); }

_FORCE_INLINE_ BigNumber operator+(const BigNumber& lhs, long long rhs) { return lhs + BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator-(const BigNumber& lhs, long long rhs) { return lhs - BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator*(const BigNumber& lhs, long long rhs) { return lhs * BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator/(const BigNumber& lhs, long long rhs) { return lhs / BigNumber(rhs); }

_FORCE_INLINE_ BigNumber operator+(const BigNumber& lhs, double rhs) { return lhs + BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator-(const BigNumber& lhs, double rhs) { return lhs - BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator*(const BigNumber& lhs, double rhs) { return lhs * BigNumber(rhs); }
_FORCE_INLINE_ BigNumber operator/(const BigNumber& lhs, double rhs) { return lhs / BigNumber(rhs); }

// Symmetric versions are already defined as friend functions, but we add more for completeness
_FORCE_INLINE_ BigNumber operator+(int lhs, const BigNumber& rhs) { return BigNumber(lhs) + rhs; }
_FORCE_INLINE_ BigNumber operator-(int lhs, const BigNumber& rhs) { return BigNumber(lhs) - rhs; }
_FORCE_INLINE_ BigNumber operator*(int lhs, const BigNumber& rhs) { return BigNumber(lhs) * rhs; }
_FORCE_INLINE_ BigNumber operator/(int lhs, const BigNumber& rhs) { return BigNumber(lhs) / rhs; }

// ----------------------------------------------------------------------------
// Stream operators (inline)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ std::ostream& operator<<(std::ostream& os, const BigNumber& num) {
    os << num.to_string();
    return os;
}

_FORCE_INLINE_ std::istream& operator>>(std::istream& is, BigNumber& num) {
    std::string s;
    is >> s;
    num = BigNumber::from_string(s);
    return is;
}

// ----------------------------------------------------------------------------
// Xtensor expression fusion helpers
// ----------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR

// Enable BigNumber to be used in xtensor expressions as a scalar value
namespace xt {
    template <>
    struct is_scalar<uep::BigNumber> : std::true_type {};
    
    // Provide conversion from BigNumber to double for xtensor operations
    template <>
    struct get_underlying_value_type<uep::BigNumber> {
        using type = double;
    };
    
    // Custom value access for BigNumber (convert to double)
    template <>
    struct value_traits<uep::BigNumber> {
        static double get(const uep::BigNumber& v) { return v.to_double(); }
    };
}

// Extend the BigNumber class with xtensor-specific operations
_FORCE_INLINE_ auto BigNumber::to_xtensor() const {
    // For a scalar, return a 0D array containing the double value
    return xt::xarray<double>({to_double()});
}

#endif // UEP_USE_XTENSOR

// ----------------------------------------------------------------------------
// End of big_number.h inline section
// ----------------------------------------------------------------------------

// Ending of Part 2 of 4 (big_number.h)

// src/big_number.h (continued from Part 2)

// ----------------------------------------------------------------------------
// Additional static factory methods for high-precision constants
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber BigNumber::pi() {
    // π = 3.14159265358979323846...
    // Scaled by 2^32: 0x3243F6A8885A308D313198A2E03707344...
    // We'll use a BigIntCore with multiple limbs for higher precision.
    static const limb_t pi_limbs[] = {
        0x85A308D313198A2EULL,
        0x03707344A4093822ULL,
        0x299F31D0081FA9F6ULL,
        0x7E1D48E8A67C8B9AULL,
        0x8C2F6E9E3A26A89BULL,
        0x3243F6A8885A308DULL
    };
    static const BigNumber pi_val = BigNumber(BigIntCore::from_limb_array(pi_limbs, 6), true);
    return pi_val;
}

_FORCE_INLINE_ BigNumber BigNumber::e() {
    // e = 2.71828182845904523536...
    static const limb_t e_limbs[] = {
        0x8AED2A6ABF715880ULL,
        0x9CF4C3B8A1D28B9CULL,
        0x6E1D88E3A1E4B5E2ULL,
        0x2B7E151628AED2A6ULL
    };
    static const BigNumber e_val = BigNumber(BigIntCore::from_limb_array(e_limbs, 4), true);
    return e_val;
}

_FORCE_INLINE_ BigNumber BigNumber::ln2() {
    // ln(2) = 0.69314718055994530942...
    static const limb_t ln2_limbs[] = {
        0xF473DE6AF278ECE6ULL,
        0x2C5C85FDF473DE6AULL
    };
    static const BigNumber ln2_val = BigNumber(BigIntCore::from_limb_array(ln2_limbs, 2), true);
    return ln2_val;
}

// ----------------------------------------------------------------------------
// Additional constants
// ----------------------------------------------------------------------------
static _FORCE_INLINE_ BigNumber sqrt2() {
    static const BigNumber val = BigNumber(FixedMathCore::sqrt2());
    return val;
}

static _FORCE_INLINE_ BigNumber one() { return BigNumber(1); }
static _FORCE_INLINE_ BigNumber zero() { return BigNumber(0); }
static _FORCE_INLINE_ BigNumber half() { return BigNumber(FixedMathCore::from_raw(FixedMathCore::HALF)); }

// ----------------------------------------------------------------------------
// Mixed-type comparisons (for convenience)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ bool operator==(const BigNumber& lhs, int rhs) { return lhs == BigNumber(rhs); }
_FORCE_INLINE_ bool operator!=(const BigNumber& lhs, int rhs) { return lhs != BigNumber(rhs); }
_FORCE_INLINE_ bool operator<(const BigNumber& lhs, int rhs)  { return lhs < BigNumber(rhs); }
_FORCE_INLINE_ bool operator>(const BigNumber& lhs, int rhs)  { return lhs > BigNumber(rhs); }
_FORCE_INLINE_ bool operator<=(const BigNumber& lhs, int rhs) { return lhs <= BigNumber(rhs); }
_FORCE_INLINE_ bool operator>=(const BigNumber& lhs, int rhs) { return lhs >= BigNumber(rhs); }

_FORCE_INLINE_ bool operator==(int lhs, const BigNumber& rhs) { return BigNumber(lhs) == rhs; }
_FORCE_INLINE_ bool operator!=(int lhs, const BigNumber& rhs) { return BigNumber(lhs) != rhs; }
_FORCE_INLINE_ bool operator<(int lhs, const BigNumber& rhs)  { return BigNumber(lhs) < rhs; }
_FORCE_INLINE_ bool operator>(int lhs, const BigNumber& rhs)  { return BigNumber(lhs) > rhs; }
_FORCE_INLINE_ bool operator<=(int lhs, const BigNumber& rhs) { return BigNumber(lhs) <= rhs; }
_FORCE_INLINE_ bool operator>=(int lhs, const BigNumber& rhs) { return BigNumber(lhs) >= rhs; }

// ----------------------------------------------------------------------------
// Advanced mathematical functions (declarations) - heavy implementations in .cpp
// ----------------------------------------------------------------------------
// Sine and Cosine using Taylor series with arbitrary precision
static BigNumber sin_taylor(const BigNumber& x);
static BigNumber cos_taylor(const BigNumber& x);

// Exponential using series expansion with range reduction
static BigNumber exp_series(const BigNumber& x);

// Natural logarithm using Newton's method or AGM
static BigNumber log_newton(const BigNumber& x);

// Square root using Newton iteration with arbitrary precision
static BigNumber sqrt_newton(const BigNumber& x);

// Power function for arbitrary exponent
static BigNumber pow_general(const BigNumber& base, const BigNumber& exp);

// ----------------------------------------------------------------------------
// Inline wrapper that dispatches to appropriate implementation
// (fast double fallback or full arbitrary precision)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ BigNumber BigNumber::sin(const BigNumber& x) {
    // For small arguments or when high precision not needed, use fast path.
    // Threshold determined by required precision. We'll use full precision always
    // for deterministic behavior; call sin_taylor.
    return sin_taylor(x);
}

_FORCE_INLINE_ BigNumber BigNumber::cos(const BigNumber& x) {
    return cos_taylor(x);
}

_FORCE_INLINE_ BigNumber BigNumber::tan(const BigNumber& x) {
    BigNumber c = cos(x);
    if (c.is_zero()) {
        // Return signed infinity? We'll clamp to max value.
        return (x > BigNumber(0) ? BigNumber(INT64_MAX) : BigNumber(INT64_MIN));
    }
    return sin(x) / c;
}

_FORCE_INLINE_ BigNumber BigNumber::asin(const BigNumber& x) {
    if (x < BigNumber(-1) || x > BigNumber(1)) return BigNumber(0);
    // Use arcsin series or atan2 formula: asin(x) = atan2(x, sqrt(1 - x^2))
    return atan2(x, sqrt(BigNumber(1) - x * x));
}

_FORCE_INLINE_ BigNumber BigNumber::acos(const BigNumber& x) {
    return pi() / BigNumber(2) - asin(x);
}

_FORCE_INLINE_ BigNumber BigNumber::atan(const BigNumber& x) {
    return atan2(x, BigNumber(1));
}

_FORCE_INLINE_ BigNumber BigNumber::atan2(const BigNumber& y, const BigNumber& x) {
    // Use CORDIC or series; implement in .cpp
    // Placeholder: call static function
    extern BigNumber atan2_cordic(const BigNumber& y, const BigNumber& x);
    return atan2_cordic(y, x);
}

_FORCE_INLINE_ BigNumber BigNumber::exp(const BigNumber& x) {
    return exp_series(x);
}

_FORCE_INLINE_ BigNumber BigNumber::log(const BigNumber& x) {
    if (x <= BigNumber(0)) return BigNumber(0);
    return log_newton(x);
}

_FORCE_INLINE_ BigNumber BigNumber::log10(const BigNumber& x) {
    static const BigNumber LN10 = BigNumber::from_string("2.30258509299404568402");
    return log(x) / LN10;
}

_FORCE_INLINE_ BigNumber BigNumber::pow(const BigNumber& base, const BigNumber& exp) {
    if (base.is_zero()) {
        if (exp <= BigNumber(0)) return BigNumber(0);
        return BigNumber(0);
    }
    if (base < BigNumber(0) && exp.frac().is_zero()) {
        // Integer exponent on negative base
        return pow_general(base, exp);
    }
    if (base < BigNumber(0)) return BigNumber(0); // complex result
    return exp(exp * log(base));
}

_FORCE_INLINE_ BigNumber BigNumber::sqrt(const BigNumber& x) {
    if (x < BigNumber(0)) return BigNumber(0);
    if (x.is_zero()) return BigNumber(0);
    return sqrt_newton(x);
}

// ----------------------------------------------------------------------------
// Additional inline helper: is_integer, is_zero, sign bit
// ----------------------------------------------------------------------------
_FORCE_INLINE_ bool is_integer(const BigNumber& x) {
    return x.frac().is_zero();
}

_FORCE_INLINE_ bool is_positive(const BigNumber& x) {
    return x > BigNumber(0);
}

_FORCE_INLINE_ bool is_negative(const BigNumber& x) {
    return x < BigNumber(0);
}

// ----------------------------------------------------------------------------
// End of big_number.h
// ----------------------------------------------------------------------------

// Ending of Part 3 of 4 (big_number.h)

// src/big_number.h (continued from Part 3)

// ----------------------------------------------------------------------------
// Helper function declarations for transcendental implementations
// These are implemented in big_number.cpp using arbitrary-precision algorithms.
// ----------------------------------------------------------------------------
namespace detail {
    // Range reduction for trigonometric functions
    void reduce_arg_pi2(const BigIntCore& x_scaled, BigIntCore& reduced, int& quadrant);
    
    // Taylor series for sin/cos on reduced argument
    BigIntCore sin_series(const BigIntCore& x_scaled, size_t terms);
    BigIntCore cos_series(const BigIntCore& x_scaled, size_t terms);
    
    // Exponential series: exp(x) = sum(x^n / n!)
    BigIntCore exp_series_scaled(const BigIntCore& x_scaled, size_t terms);
    
    // Natural log using Newton iteration on exp(y) = x
    BigIntCore log_newton_scaled(const BigIntCore& x_scaled);
    
    // Square root using Newton: y_{n+1} = (y_n + x / y_n) / 2
    BigIntCore sqrt_newton_scaled(const BigIntCore& x_scaled);
    
    // CORDIC for atan2
    BigIntCore atan2_cordic_scaled(const BigIntCore& y_scaled, const BigIntCore& x_scaled);
    
    // Power with integer exponent (arbitrary precision)
    BigIntCore pow_int_scaled(const BigIntCore& base_scaled, const BigIntCore& exp_scaled);
}

// ----------------------------------------------------------------------------
// Static member definitions that require out-of-line storage
// (These will be defined in big_number.cpp with actual values)
// ----------------------------------------------------------------------------
extern const BigNumber BigNumber_PI;
extern const BigNumber BigNumber_E;
extern const BigNumber BigNumber_LN2;
extern const BigNumber BigNumber_LN10;
extern const BigNumber BigNumber_SQRT2;

// ----------------------------------------------------------------------------
// Inline getters for the externally defined constants
// ----------------------------------------------------------------------------
_FORCE_INLINE_ const BigNumber& pi() { return BigNumber_PI; }
_FORCE_INLINE_ const BigNumber& e()  { return BigNumber_E; }
_FORCE_INLINE_ const BigNumber& ln2() { return BigNumber_LN2; }
_FORCE_INLINE_ const BigNumber& ln10() { return BigNumber_LN10; }
_FORCE_INLINE_ const BigNumber& sqrt2() { return BigNumber_SQRT2; }

// ----------------------------------------------------------------------------
// String conversion (inline wrappers that call heavy implementations)
// ----------------------------------------------------------------------------
_FORCE_INLINE_ std::string to_string(const BigNumber& num) {
    return num.to_string();
}

_FORCE_INLINE_ BigNumber from_string(const std::string& str) {
    return BigNumber::from_string(str);
}

// ----------------------------------------------------------------------------
// Xtensor expression fusion extensions
// These enable BigNumber to be used seamlessly in xtensor expressions,
// allowing the compiler to generate fused SIMD loops over limb arrays.
// ----------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR

// Provide a specialization of xtensor's xexpression traits for BigNumber
namespace xt {
    template <>
    struct xexpression_traits<uep::BigNumber> {
        using value_type = uep::BigNumber;
        using reference = uep::BigNumber&;
        using const_reference = const uep::BigNumber&;
        using pointer = uep::BigNumber*;
        using const_pointer = const uep::BigNumber*;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;
        static constexpr bool is_const = false;
    };
    
    // Enable automatic conversion from BigNumber to xtensor expression
    template <>
    struct value_traits<uep::BigNumber> {
        static double get(const uep::BigNumber& v) {
            return v.to_double();
        }
    };
}

// Extend BigNumber with an xtensor view of its limb array
_FORCE_INLINE_ auto BigNumber::xt_limbs() const -> xt::xarray<limb_t> {
    return xt::adapt(value_.data(), value_.size(), xt::no_ownership(), {value_.size()});
}

_FORCE_INLINE_ auto BigNumber::xt_limbs() -> xt::xarray<limb_t> {
    return xt::adapt(value_.data(), value_.size(), xt::no_ownership(), {value_.size()});
}

// Allow a BigNumber to be broadcasted as a scalar in xtensor operations
_FORCE_INLINE_ auto operator+(const xt::xexpression<xt::xarray<double>>& expr, const BigNumber& scalar) {
    return expr.derived_cast() + scalar.to_double();
}

_FORCE_INLINE_ auto operator-(const xt::xexpression<xt::xarray<double>>& expr, const BigNumber& scalar) {
    return expr.derived_cast() - scalar.to_double();
}

_FORCE_INLINE_ auto operator*(const xt::xexpression<xt::xarray<double>>& expr, const BigNumber& scalar) {
    return expr.derived_cast() * scalar.to_double();
}

_FORCE_INLINE_ auto operator/(const xt::xexpression<xt::xarray<double>>& expr, const BigNumber& scalar) {
    return expr.derived_cast() / scalar.to_double();
}

#endif // UEP_USE_XTENSOR

// ----------------------------------------------------------------------------
// End of BigNumber class and associated utilities
// ----------------------------------------------------------------------------

} // namespace uep

// ----------------------------------------------------------------------------
// Standard hash specialization for BigNumber (useful for unordered containers)
// ----------------------------------------------------------------------------
namespace std {
    template <>
    struct hash<uep::BigNumber> {
        size_t operator()(const uep::BigNumber& bn) const {
            // Hash based on the underlying BigIntCore's limb data
            const uep::BigIntCore& val = bn.scaled_value();
            size_t h = val.size();
            for (size_t i = 0; i < val.size(); ++i) {
                h ^= std::hash<uint64_t>()(val.data()[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
}

#endif // BIG_NUMBER_H

// Ending of Part 4 of 4 (big_number.h) - File complete.