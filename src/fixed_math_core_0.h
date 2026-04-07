--- START OF FILE src/fixed_math_core.h ---

#ifndef FIXED_MATH_CORE_H
#define FIXED_MATH_CORE_H

#include <string>
#include <cstdint>

// Enforce SIMD cache-line alignment for Warp kernel processing and EnTT SoA packing
#if defined(__GNUC__) || defined(__clang__)
    #define ET_ALIGN_32 __attribute__((aligned(32)))
    #define ET_SIMD_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define ET_ALIGN_32 __declspec(align(32))
    #define ET_SIMD_INLINE __forceinline
#else
    #define ET_ALIGN_32 alignas(32)
    #define ET_SIMD_INLINE inline
#endif

/**
 * FixedMathCore
 * 
 * The deterministic Q32.32 fixed-point math backend.
 * Replaces 'double' and 'float' across the entire engine to guarantee 
 * bit-perfect physics, rendering, and logic synchronization.
 */
struct ET_ALIGN_32 FixedMathCore {
private:
    // Q32.32 fixed-point representation.
    // 32 bits for the integer part, 32 bits for the fractional part.
    // Direct 64-bit primitive storage ensures zero-copy Warp kernel mapping.
    int64_t raw_value;

    // Internal constructor for direct raw value assignment (Zero-cost)
    ET_SIMD_INLINE explicit FixedMathCore(int64_t p_raw, bool p_is_raw);

public:
    // Constants shifted by 32 bits
    static const int64_t FRACTIONAL_BITS = 32;
    static const int64_t ONE = 1LL << FRACTIONAL_BITS;
    static const int64_t PI_RAW = 13493037704LL;      // 3.14159265358979323846 * 2^32
    static const int64_t HALF_PI_RAW = 6746518852LL;  // 1.57079632679489661923 * 2^32
    static const int64_t TWO_PI_RAW = 26986075409LL;  // 6.28318530717958647692 * 2^32
    static const int64_t E_RAW = 11670868155LL;       // 2.71828182845904523536 * 2^32

    // Constructors
    ET_SIMD_INLINE FixedMathCore();
    ET_SIMD_INLINE FixedMathCore(int32_t p_value);
    ET_SIMD_INLINE FixedMathCore(int64_t p_value);
    ET_SIMD_INLINE FixedMathCore(double p_value); // Non-deterministic, initialization only
    FixedMathCore(const std::string& p_value);
    ET_SIMD_INLINE FixedMathCore(const FixedMathCore& p_other);

    // Assignment
    ET_SIMD_INLINE FixedMathCore& operator=(const FixedMathCore& p_other);

    // Arithmetic Operators (Inline for ECS batch loops)
    ET_SIMD_INLINE FixedMathCore operator+(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE FixedMathCore operator-(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE FixedMathCore operator*(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE FixedMathCore operator/(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE FixedMathCore operator%(const FixedMathCore& p_other) const;

    // Compound Assignment Operators
    ET_SIMD_INLINE FixedMathCore& operator+=(const FixedMathCore& p_other);
    ET_SIMD_INLINE FixedMathCore& operator-=(const FixedMathCore& p_other);
    ET_SIMD_INLINE FixedMathCore& operator*=(const FixedMathCore& p_other);
    ET_SIMD_INLINE FixedMathCore& operator/=(const FixedMathCore& p_other);

    // Comparison Operators (Branchless potential)
    ET_SIMD_INLINE bool operator==(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE bool operator!=(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE bool operator<(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE bool operator<=(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE bool operator>(const FixedMathCore& p_other) const;
    ET_SIMD_INLINE bool operator>=(const FixedMathCore& p_other) const;

    // Deterministic Scientific / Physics Math
    ET_SIMD_INLINE FixedMathCore absolute() const;
    FixedMathCore square_root() const;
    FixedMathCore power(int32_t exponent) const;
    FixedMathCore sin() const;
    FixedMathCore cos() const;
    FixedMathCore tan() const;
    FixedMathCore atan2(const FixedMathCore& x) const;

    // Constants Getters
    static ET_SIMD_INLINE FixedMathCore pi();
    static ET_SIMD_INLINE FixedMathCore e();

    // Utilities, Conversions & Warp Hashing
    ET_SIMD_INLINE int64_t get_raw() const;
    ET_SIMD_INLINE double to_double() const;
    ET_SIMD_INLINE int64_t to_int() const;
    std::string to_string() const;
    
    // Hash function crucial for EnTT Sparse Set indexing and dictionaries
    ET_SIMD_INLINE uint32_t hash() const;
};

// ============================================================================
// SIMD/Warp Inline Implementations
// ============================================================================
// To prevent function call overhead in physics sweeps, core math is strictly inlined.

ET_SIMD_INLINE FixedMathCore::FixedMathCore(int64_t p_raw, bool p_is_raw) : raw_value(p_raw) {}
ET_SIMD_INLINE FixedMathCore::FixedMathCore() : raw_value(0) {}
ET_SIMD_INLINE FixedMathCore::FixedMathCore(int32_t p_value) : raw_value(static_cast<int64_t>(p_value) << FRACTIONAL_BITS) {}
ET_SIMD_INLINE FixedMathCore::FixedMathCore(int64_t p_value) : raw_value(p_value << FRACTIONAL_BITS) {}
ET_SIMD_INLINE FixedMathCore::FixedMathCore(double p_value) : raw_value(static_cast<int64_t>(p_value * static_cast<double>(ONE))) {}
ET_SIMD_INLINE FixedMathCore::FixedMathCore(const FixedMathCore& p_other) : raw_value(p_other.raw_value) {}

ET_SIMD_INLINE FixedMathCore& FixedMathCore::operator=(const FixedMathCore& p_other) {
    raw_value = p_other.raw_value;
    return *this;
}

ET_SIMD_INLINE FixedMathCore FixedMathCore::operator+(const FixedMathCore& p_other) const {
    return FixedMathCore(raw_value + p_other.raw_value, true);
}

ET_SIMD_INLINE FixedMathCore FixedMathCore::operator-(const FixedMathCore& p_other) const {
    return FixedMathCore(raw_value - p_other.raw_value, true);
}

// 128-bit multiplication for perfect precision without clipping overflow
ET_SIMD_INLINE FixedMathCore FixedMathCore::operator*(const FixedMathCore& p_other) const {
    __int128_t temp = static_cast<__int128_t>(raw_value) * static_cast<__int128_t>(p_other.raw_value);
    temp >>= FRACTIONAL_BITS;
    return FixedMathCore(static_cast<int64_t>(temp), true);
}

ET_SIMD_INLINE FixedMathCore FixedMathCore::operator/(const FixedMathCore& p_other) const {
    __int128_t temp = static_cast<__int128_t>(raw_value) << FRACTIONAL_BITS;
    temp /= static_cast<__int128_t>(p_other.raw_value); // Note: Divide by zero protection handled at higher logic
    return FixedMathCore(static_cast<int64_t>(temp), true);
}

ET_SIMD_INLINE FixedMathCore FixedMathCore::operator%(const FixedMathCore& p_other) const {
    return FixedMathCore(raw_value % p_other.raw_value, true);
}

ET_SIMD_INLINE FixedMathCore& FixedMathCore::operator+=(const FixedMathCore& p_other) { raw_value += p_other.raw_value; return *this; }
ET_SIMD_INLINE FixedMathCore& FixedMathCore::operator-=(const FixedMathCore& p_other) { raw_value -= p_other.raw_value; return *this; }
ET_SIMD_INLINE FixedMathCore& FixedMathCore::operator*=(const FixedMathCore& p_other) { *this = *this * p_other; return *this; }
ET_SIMD_INLINE FixedMathCore& FixedMathCore::operator/=(const FixedMathCore& p_other) { *this = *this / p_other; return *this; }

ET_SIMD_INLINE bool FixedMathCore::operator==(const FixedMathCore& p_other) const { return raw_value == p_other.raw_value; }
ET_SIMD_INLINE bool FixedMathCore::operator!=(const FixedMathCore& p_other) const { return raw_value != p_other.raw_value; }
ET_SIMD_INLINE bool FixedMathCore::operator<(const FixedMathCore& p_other) const { return raw_value < p_other.raw_value; }
ET_SIMD_INLINE bool FixedMathCore::operator<=(const FixedMathCore& p_other) const { return raw_value <= p_other.raw_value; }
ET_SIMD_INLINE bool FixedMathCore::operator>(const FixedMathCore& p_other) const { return raw_value > p_other.raw_value; }
ET_SIMD_INLINE bool FixedMathCore::operator>=(const FixedMathCore& p_other) const { return raw_value >= p_other.raw_value; }

ET_SIMD_INLINE FixedMathCore FixedMathCore::absolute() const {
    return FixedMathCore(raw_value < 0 ? -raw_value : raw_value, true);
}

ET_SIMD_INLINE int64_t FixedMathCore::get_raw() const { return raw_value; }
ET_SIMD_INLINE double FixedMathCore::to_double() const { return static_cast<double>(raw_value) / static_cast<double>(ONE); }
ET_SIMD_INLINE int64_t FixedMathCore::to_int() const { return raw_value >> FRACTIONAL_BITS; }

ET_SIMD_INLINE FixedMathCore FixedMathCore::pi() { return FixedMathCore(PI_RAW, true); }
ET_SIMD_INLINE FixedMathCore FixedMathCore::e() { return FixedMathCore(E_RAW, true); }

// High-speed Murmur-style hash for O(1) ECS registry lookup
ET_SIMD_INLINE uint32_t FixedMathCore::hash() const {
    uint64_t v = (uint64_t)raw_value;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return (uint32_t)(v ^ (v >> 31));
}

#endif // FIXED_MATH_CORE_H

--- END OF FILE src/fixed_math_core.h ---
