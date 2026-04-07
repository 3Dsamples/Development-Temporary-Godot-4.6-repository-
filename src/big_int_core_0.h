--- START OF FILE src/big_int_core.h ---

#ifndef BIG_INT_CORE_H
#define BIG_INT_CORE_H

#include <vector>
#include <string>
#include <iostream>
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

// Forward declaration to allow FixedMathCore to interface cleanly without floats
class FixedMathCore;

/**
 * BigIntCore
 * 
 * The ultimate arbitrary-precision engine component.
 * Forms the "Where" and "Who" in the Universal Solver architecture.
 */
struct ET_ALIGN_32 BigIntCore {
private:
    // Base 10^9 chunking. Uses 32-bit integers for storage and 64-bit for calculation.
    // Stored directly to support EnTT contiguous registries where possible.
    std::vector<uint32_t> chunks; 
    bool is_negative;
    static const uint32_t BASE = 1000000000;

    ET_SIMD_INLINE void trim();
    ET_SIMD_INLINE void divide_by_2();

public:
    // Constructors (Warp-Kernel friendly defaults)
    ET_SIMD_INLINE BigIntCore();
    ET_SIMD_INLINE BigIntCore(int64_t value);
    BigIntCore(const std::string& value);
    ET_SIMD_INLINE BigIntCore(const BigIntCore& other);

    // Assignment
    ET_SIMD_INLINE BigIntCore& operator=(const BigIntCore& other);
    ET_SIMD_INLINE BigIntCore& operator=(int64_t value);
    BigIntCore& operator=(const std::string& value);

    // Arithmetic Operators (Zero-copy parameter paths)
    BigIntCore operator+(const BigIntCore& other) const;
    BigIntCore operator-(const BigIntCore& other) const;
    BigIntCore operator*(const BigIntCore& other) const;
    BigIntCore operator/(const BigIntCore& other) const;
    BigIntCore operator%(const BigIntCore& other) const;

    // Compound Assignment Operators
    ET_SIMD_INLINE BigIntCore& operator+=(const BigIntCore& other);
    ET_SIMD_INLINE BigIntCore& operator-=(const BigIntCore& other);
    ET_SIMD_INLINE BigIntCore& operator*=(const BigIntCore& other);
    ET_SIMD_INLINE BigIntCore& operator/=(const BigIntCore& other);
    ET_SIMD_INLINE BigIntCore& operator%=(const BigIntCore& other);

    // Comparison Operators (Constant time logical checks for EnTT view filtering)
    bool operator==(const BigIntCore& other) const;
    bool operator!=(const BigIntCore& other) const;
    bool operator<(const BigIntCore& other) const;
    bool operator<=(const BigIntCore& other) const;
    bool operator>(const BigIntCore& other) const;
    bool operator>=(const BigIntCore& other) const;

    // Advanced Math Features (Hardware-agnostic precision)
    BigIntCore power(const BigIntCore& exponent) const;
    BigIntCore square_root() const;
    BigIntCore absolute() const;

    // String Formatting specifically built for Idle / Incremental Games
    std::string to_string() const;
    std::string to_scientific() const;
    std::string to_aa_notation() const;
    std::string to_metric_symbol() const;
    std::string to_metric_name() const;

    // Warp-Kernel Data Extractors (For SoA transformation mapping)
    ET_SIMD_INLINE const std::vector<uint32_t>& get_chunks() const { return chunks; }
    
    // Hashing (For EnTT Entity generation / Sparse Set indexing)
    uint32_t hash() const;

    // Utility
    ET_SIMD_INLINE bool is_zero() const;
    ET_SIMD_INLINE int sign() const;
};

#endif // BIG_INT_CORE_H

--- END OF FILE src/big_int_core.h ---
