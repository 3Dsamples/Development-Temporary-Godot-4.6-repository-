// src/fixed_math_core.h
#ifndef FIXED_MATH_CORE_H
#define FIXED_MATH_CORE_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>

#if defined(__GNUC__) || defined(__clang__)
#define _FORCE_INLINE_ __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define _FORCE_INLINE_ __forceinline
#else
#define _FORCE_INLINE_ inline
#endif

// SIMD includes
#ifdef __AVX512F__
#include <immintrin.h>
#define FIXED_SIMD_AVX512 1
#endif
#ifdef __AVX2__
#include <immintrin.h>
#define FIXED_SIMD_AVX2 1
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#define FIXED_SIMD_NEON 1
#endif

// Forward declaration for BigIntCore (used for intermediate precision)
namespace uep {
    class BigIntCore;
}

// xtensor integration (if available)
#ifdef UEP_USE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#endif

namespace uep {

// ----------------------------------------------------------------------------
// FixedMathCore: Q32.32 deterministic fixed-point arithmetic
// Representation: int64_t with 32 fractional bits.
// Range: approximately [-2^31, 2^31 - 1] with 1/2^32 precision (~2.3e-10).
// ----------------------------------------------------------------------------
class FixedMathCore {
private:
    int64_t value_;  // raw fixed-point value (signed 64-bit)

    // Private constructor from raw value (for internal use)
    explicit _FORCE_INLINE_ FixedMathCore(int64_t raw, bool) : value_(raw) {}

public:
    // Constants
    static constexpr int FRACTIONAL_BITS = 32;
    static constexpr int64_t ONE = 1LL << FRACTIONAL_BITS;
    static constexpr int64_t HALF = 1LL << (FRACTIONAL_BITS - 1);
    static constexpr int64_t PI_RAW = 0x3243F6A8885A308DLL; // π * 2^32 (high-precision)
    static constexpr int64_t E_RAW  = 0x2B7E151628AED2A6LL; // e * 2^32
    static constexpr int64_t LN2_RAW = 0x2C5C85FDF473DE6ALL; // ln(2) * 2^32
    static constexpr int64_t SQRT2_RAW = 0x16A09E667F3BCC90LL; // sqrt(2) * 2^32

    // Constructors
    _FORCE_INLINE_ FixedMathCore() : value_(0) {}
    _FORCE_INLINE_ FixedMathCore(int val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(long val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(long long val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(unsigned int val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(unsigned long val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(unsigned long long val) : value_(int64_t(val) << FRACTIONAL_BITS) {}
    _FORCE_INLINE_ FixedMathCore(float val) : value_(int64_t(val * ONE)) {}
    _FORCE_INLINE_ FixedMathCore(double val) : value_(int64_t(val * ONE)) {}
    _FORCE_INLINE_ FixedMathCore(long double val) : value_(int64_t(val * ONE)) {}

    // Copy/move constructors
    _FORCE_INLINE_ FixedMathCore(const FixedMathCore&) = default;
    _FORCE_INLINE_ FixedMathCore(FixedMathCore&&) = default;
    _FORCE_INLINE_ FixedMathCore& operator=(const FixedMathCore&) = default;
    _FORCE_INLINE_ FixedMathCore& operator=(FixedMathCore&&) = default;

    // Access to raw value
    _FORCE_INLINE_ int64_t raw() const { return value_; }
    _FORCE_INLINE_ static FixedMathCore from_raw(int64_t raw) { return FixedMathCore(raw, true); }

    // Conversion to primitive types
    _FORCE_INLINE_ int to_int() const { return int(value_ >> FRACTIONAL_BITS); }
    _FORCE_INLINE_ long to_long() const { return long(value_ >> FRACTIONAL_BITS); }
    _FORCE_INLINE_ long long to_llong() const { return value_ >> FRACTIONAL_BITS; }
    _FORCE_INLINE_ float to_float() const { return float(value_) / ONE; }
    _FORCE_INLINE_ double to_double() const { return double(value_) / ONE; }

    // ------------------------------------------------------------------------
    // Inlined arithmetic operators (using 128-bit intermediates)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore operator-() const {
        return FixedMathCore(-value_, true);
    }

    _FORCE_INLINE_ FixedMathCore operator+(const FixedMathCore& other) const {
        return FixedMathCore(value_ + other.value_, true);
    }

    _FORCE_INLINE_ FixedMathCore operator-(const FixedMathCore& other) const {
        return FixedMathCore(value_ - other.value_, true);
    }

    _FORCE_INLINE_ FixedMathCore operator*(const FixedMathCore& other) const {
        // Multiply two Q32.32 numbers using 128-bit intermediate
        __int128 prod = __int128(value_) * other.value_;
        // Shift right by 32 to renormalize, with rounding
        return FixedMathCore(int64_t((prod + (1LL << 31)) >> FRACTIONAL_BITS), true);
    }

    _FORCE_INLINE_ FixedMathCore operator/(const FixedMathCore& other) const {
        // Divide: (value * 2^32) / other.value_
        __int128 num = __int128(value_) << FRACTIONAL_BITS;
        return FixedMathCore(int64_t(num / other.value_), true);
    }

    _FORCE_INLINE_ FixedMathCore& operator+=(const FixedMathCore& other) {
        value_ += other.value_;
        return *this;
    }

    _FORCE_INLINE_ FixedMathCore& operator-=(const FixedMathCore& other) {
        value_ -= other.value_;
        return *this;
    }

    _FORCE_INLINE_ FixedMathCore& operator*=(const FixedMathCore& other) {
        *this = *this * other;
        return *this;
    }

    _FORCE_INLINE_ FixedMathCore& operator/=(const FixedMathCore& other) {
        *this = *this / other;
        return *this;
    }

    // ------------------------------------------------------------------------
    // Comparison operators
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ bool operator==(const FixedMathCore& other) const { return value_ == other.value_; }
    _FORCE_INLINE_ bool operator!=(const FixedMathCore& other) const { return value_ != other.value_; }
    _FORCE_INLINE_ bool operator<(const FixedMathCore& other) const  { return value_ < other.value_; }
    _FORCE_INLINE_ bool operator>(const FixedMathCore& other) const  { return value_ > other.value_; }
    _FORCE_INLINE_ bool operator<=(const FixedMathCore& other) const { return value_ <= other.value_; }
    _FORCE_INLINE_ bool operator>=(const FixedMathCore& other) const { return value_ >= other.value_; }

    // ------------------------------------------------------------------------
    // Bitwise operations (treat as raw bits, not mathematical)
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore operator&(const FixedMathCore& other) const { return from_raw(value_ & other.value_); }
    _FORCE_INLINE_ FixedMathCore operator|(const FixedMathCore& other) const { return from_raw(value_ | other.value_); }
    _FORCE_INLINE_ FixedMathCore operator^(const FixedMathCore& other) const { return from_raw(value_ ^ other.value_); }
    _FORCE_INLINE_ FixedMathCore operator~() const { return from_raw(~value_); }

    // ------------------------------------------------------------------------
    // Absolute value
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore abs() const {
        return FixedMathCore(value_ < 0 ? -value_ : value_, true);
    }

    // ------------------------------------------------------------------------
    // Floor and ceil
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore floor() const {
        return FixedMathCore((value_ >> FRACTIONAL_BITS) << FRACTIONAL_BITS, true);
    }

    _FORCE_INLINE_ FixedMathCore ceil() const {
        int64_t int_part = value_ >> FRACTIONAL_BITS;
        if ((value_ & (ONE - 1)) == 0) return *this;
        return FixedMathCore((int_part + (value_ > 0 ? 1 : 0)) << FRACTIONAL_BITS, true);
    }

    // ------------------------------------------------------------------------
    // Fractional part
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore frac() const {
        return FixedMathCore(value_ & (ONE - 1), true);
    }

    // ------------------------------------------------------------------------
    // Round to nearest integer
    // ------------------------------------------------------------------------
    _FORCE_INLINE_ FixedMathCore round() const {
        int64_t frac = value_ & (ONE - 1);
        int64_t int_part = value_ - frac;
        if (frac >= HALF) int_part += ONE;
        return FixedMathCore(int_part, true);
    }

    // ------------------------------------------------------------------------
    // Static constants
    // ------------------------------------------------------------------------
    static _FORCE_INLINE_ FixedMathCore pi()    { return from_raw(PI_RAW); }
    static _FORCE_INLINE_ FixedMathCore e()     { return from_raw(E_RAW); }
    static _FORCE_INLINE_ FixedMathCore ln2()   { return from_raw(LN2_RAW); }
    static _FORCE_INLINE_ FixedMathCore sqrt2() { return from_raw(SQRT2_RAW); }

    // ------------------------------------------------------------------------
    // Trigonometric functions (declarations, implemented in .cpp)
    // ------------------------------------------------------------------------
    static FixedMathCore sin(FixedMathCore x);   // using Taylor series
    static FixedMathCore cos(FixedMathCore x);
    static FixedMathCore tan(FixedMathCore x);
    static FixedMathCore asin(FixedMathCore x);
    static FixedMathCore acos(FixedMathCore x);
    static FixedMathCore atan(FixedMathCore x);
    static FixedMathCore atan2(FixedMathCore y, FixedMathCore x); // CORDIC

    // ------------------------------------------------------------------------
    // Hyperbolic functions
    // ------------------------------------------------------------------------
    static FixedMathCore sinh(FixedMathCore x);
    static FixedMathCore cosh(FixedMathCore x);
    static FixedMathCore tanh(FixedMathCore x);

    // ------------------------------------------------------------------------
    // Exponential and logarithmic
    // ------------------------------------------------------------------------
    static FixedMathCore exp(FixedMathCore x);
    static FixedMathCore log(FixedMathCore x);
    static FixedMathCore log10(FixedMathCore x);
    static FixedMathCore pow(FixedMathCore base, FixedMathCore exp);

    // ------------------------------------------------------------------------
    // Square root (Newton-Raphson)
    // ------------------------------------------------------------------------
    static FixedMathCore sqrt(FixedMathCore x);

    // ------------------------------------------------------------------------
    // SIMD-accelerated vector operations (for arrays of FixedMathCore)
    // ------------------------------------------------------------------------
#ifdef FIXED_SIMD_AVX512
    static void add_array(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        size_t i = 0;
        for (; i + 7 < count; i += 8) {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const int64_t*>(a + i));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const int64_t*>(b + i));
            __m512i sum = _mm512_add_epi64(va, vb);
            _mm512_storeu_si512(reinterpret_cast<int64_t*>(dst + i), sum);
        }
        for (; i < count; ++i) dst[i] = a[i] + b[i];
    }

    static void mul_array(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        size_t i = 0;
        // AVX-512 has _mm512_mullo_epi64 for 64-bit low multiply, but we need 128-bit intermediate.
        // Use scalar for simplicity (can be optimized with assembly).
        for (; i < count; ++i) dst[i] = a[i] * b[i];
    }
#endif

#ifdef FIXED_SIMD_AVX2
    static void add_array(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        size_t i = 0;
        for (; i + 3 < count; i += 4) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            __m256i sum = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), sum);
        }
        for (; i < count; ++i) dst[i] = a[i] + b[i];
    }
#endif

#ifdef FIXED_SIMD_NEON
    static void add_array(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        size_t i = 0;
        for (; i + 1 < count; i += 2) {
            int64x2_t va = vld1q_s64(reinterpret_cast<const int64_t*>(a + i));
            int64x2_t vb = vld1q_s64(reinterpret_cast<const int64_t*>(b + i));
            int64x2_t sum = vaddq_s64(va, vb);
            vst1q_s64(reinterpret_cast<int64_t*>(dst + i), sum);
        }
        for (; i < count; ++i) dst[i] = a[i] + b[i];
    }
#endif

    // Fallback scalar array operations
    static void add_array_scalar(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        for (size_t i = 0; i < count; ++i) dst[i] = a[i] + b[i];
    }

    static void mul_array_scalar(FixedMathCore* dst, const FixedMathCore* a, const FixedMathCore* b, size_t count) {
        for (size_t i = 0; i < count; ++i) dst[i] = a[i] * b[i];
    }

    // ------------------------------------------------------------------------
    // Xtensor integration
    // ------------------------------------------------------------------------
#ifdef UEP_USE_XTENSOR
    // Adapt a FixedMathCore to xtensor expression
    template<typename T = int64_t>
    auto to_xtensor() const {
        // Represent as a scalar tensor
        return xt::xarray<int64_t>({value_});
    }

    static FixedMathCore from_xtensor(const xt::xarray<int64_t>& arr) {
        if (arr.size() == 0) return FixedMathCore();
        return from_raw(arr(0));
    }
#endif

    // ------------------------------------------------------------------------
    // String conversion
    // ------------------------------------------------------------------------
    std::string to_string() const;
    static FixedMathCore from_string(const std::string& str);

    // ------------------------------------------------------------------------
    // Conversion to/from BigIntCore (high precision)
    // ------------------------------------------------------------------------
    // These are declared here but defined in big_number.cpp (to avoid circular dependency)
    static FixedMathCore from_bigint(const BigIntCore& bi);
    BigIntCore to_bigint() const;
};

// ----------------------------------------------------------------------------
// Additional inline utility functions
// ----------------------------------------------------------------------------
_FORCE_INLINE_ FixedMathCore operator"" _fx(unsigned long long val) {
    return FixedMathCore(static_cast<int64_t>(val));
}

_FORCE_INLINE_ FixedMathCore operator"" _fx(long double val) {
    return FixedMathCore(static_cast<double>(val));
}

// ----------------------------------------------------------------------------
// Stream operators
// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const FixedMathCore& num);
std::istream& operator>>(std::istream& is, FixedMathCore& num);

} // namespace uep

#endif // FIXED_MATH_CORE_H
// Ending of Part 1 of 3 (fixed_math_core.h)

// ----------------------------------------------------------------------------
// Additional inlined functions that require BigIntCore (if included after)
// These will be defined later when both types are known.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Constants definitions (static members need out-of-class definitions in .cpp)
// We provide them here as constexpr for inline usage.
// ----------------------------------------------------------------------------
constexpr int FixedMathCore::FRACTIONAL_BITS;
constexpr int64_t FixedMathCore::ONE;
constexpr int64_t FixedMathCore::HALF;
constexpr int64_t FixedMathCore::PI_RAW;
constexpr int64_t FixedMathCore::E_RAW;
constexpr int64_t FixedMathCore::LN2_RAW;
constexpr int64_t FixedMathCore::SQRT2_RAW;

// ----------------------------------------------------------------------------
// End of fixed_math_core.h
// ----------------------------------------------------------------------------

// Ending of Part 2 of 3 (fixed_math_core.h)
