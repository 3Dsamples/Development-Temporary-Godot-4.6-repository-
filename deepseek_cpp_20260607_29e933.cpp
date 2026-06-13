/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file bitset_arithmetic.h
 * @brief Low‑level bitwise operations for Morton codes, interleaving, and spatial hashing.
 *
 * This file provides high‑performance, branchless functions for:
 * - Morton code encoding/decoding (2D, 3D) with 32‑bit and 64‑bit outputs
 * - Bit spreading (interleaving) and compacting (deinterleaving)
 * - Fast computation of child octant/quadrant indices from Morton codes
 * - Prefix sums and bit‑scan operations useful for tree construction
 *
 * All functions are constexpr where possible and use SIMD‑friendly bitwise
 * arithmetic. They are central to the linear BVH and dynamic octree's spatial
 * ordering.
 */

#ifndef ORTHOTREE_DETAIL_BITSET_ARITHMETIC_H_INCLUDED
#define ORTHOTREE_DETAIL_BITSET_ARITHMETIC_H_INCLUDED

#include "../core/types.h"
#include "../core/build_config.h"
#include <cstdint>
#include <type_traits>

namespace OrthoTree {
namespace detail {

// ============================================================================
//  Bit‑spreading (interleaving) functions – magic number method
// ============================================================================

/**
 * @brief Spread bits of a 10‑bit integer to 30 bits (for 2D Morton).
 *        Used internally for 32‑bit Morton codes in 2D.
 */
constexpr uint32_t spreadBits2D(uint32_t x) noexcept {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    return x;
}

/**
 * @brief Spread bits of a 21‑bit integer to 63 bits (for 3D Morton).
 *        Used for 64‑bit Morton codes in 3D.
 */
constexpr uint64_t spreadBits3D(uint64_t x) noexcept {
    x = (x | (x << 32)) & 0x001F00000000FFFF;
    x = (x | (x << 16)) & 0x001F0000FF0000FF;
    x = (x | (x << 8))  & 0x100F00F00F00F00F;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

/**
 * @brief Compact bits (inverse of spreadBits2D) – extract interleaved coordinates.
 */
constexpr uint32_t compactBits2D(uint32_t x) noexcept {
    x &= 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x = (x | (x >> 16)) & 0x000003FF;
    return x;
}

/**
 * @brief Compact bits (inverse of spreadBits3D) for 64‑bit Morton.
 */
constexpr uint64_t compactBits3D(uint64_t x) noexcept {
    x &= 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3;
    x = (x | (x >> 4)) & 0x100F00F00F00F00F;
    x = (x | (x >> 8)) & 0x001F0000FF0000FF;
    x = (x | (x >> 16)) & 0x001F00000000FFFF;
    x = (x | (x >> 32)) & 0x00000000001FFFFF;
    return x;
}

// ============================================================================
//  Morton code encoding (2D and 3D, 32/64‑bit)
// ============================================================================

/**
 * @brief Encode 2D coordinates (x, y) into a 32‑bit Morton code.
 * @param x, y Coordinates in range [0, 1023] (10 bits each).
 * @return Interleaved 32‑bit code.
 */
constexpr uint32_t mortonEncode2D_32(uint32_t x, uint32_t y) noexcept {
    return (spreadBits2D(x) << 1) | spreadBits2D(y);
}

/**
 * @brief Encode 3D coordinates (x, y, z) into a 64‑bit Morton code.
 * @param x, y, z Coordinates in range [0, 2097151] (21 bits each).
 * @return Interleaved 64‑bit code.
 */
constexpr uint64_t mortonEncode3D_64(uint64_t x, uint64_t y, uint64_t z) noexcept {
    return (spreadBits3D(x) << 2) | (spreadBits3D(y) << 1) | spreadBits3D(z);
}

/**
 * @brief Decode a 32‑bit Morton code back to 2D coordinates.
 */
constexpr void mortonDecode2D_32(uint32_t code, uint32_t& x, uint32_t& y) noexcept {
    x = compactBits2D(code >> 1);
    y = compactBits2D(code);
}

/**
 * @brief Decode a 64‑bit Morton code back to 3D coordinates.
 */
constexpr void mortonDecode3D_64(uint64_t code, uint64_t& x, uint64_t& y, uint64_t& z) noexcept {
    x = compactBits3D(code >> 2);
    y = compactBits3D(code >> 1);
    z = compactBits3D(code);
}

// ============================================================================
//  Generic template for dimension‑aware Morton encoding
// ============================================================================

/**
 * @brief Compute Morton code from normalized coordinates [0,1] in 2D/3D.
 * @tparam Dim Dimension (2 or 3).
 * @tparam T Floating‑point type.
 * @param coords Array of length Dim, each in [0,1].
 * @return Morton code (32‑bit for 2D, 64‑bit for 3D).
 */
template <Dimension Dim, typename T>
constexpr auto mortonEncode(const T* coords) noexcept {
    if constexpr (Dim == Dim2) {
        uint32_t ix = static_cast<uint32_t>(coords[0] * 1023.0f);
        uint32_t iy = static_cast<uint32_t>(coords[1] * 1023.0f);
        return mortonEncode2D_32(ix, iy);
    } else if constexpr (Dim == Dim3) {
        uint64_t ix = static_cast<uint64_t>(coords[0] * 2097151.0);
        uint64_t iy = static_cast<uint64_t>(coords[1] * 2097151.0);
        uint64_t iz = static_cast<uint64_t>(coords[2] * 2097151.0);
        return mortonEncode3D_64(ix, iy, iz);
    }
}

// Overload for Vector types from core/math
template <Dimension Dim, typename T>
constexpr auto mortonEncode(const Math::Vector<T, static_cast<int>(Dim)>& v) noexcept {
    T normCoords[Dim == Dim2 ? 2 : 3];
    for (std::size_t i = 0; i < static_cast<std::size_t>(Dim); ++i) {
        normCoords[i] = v[i];
    }
    return mortonEncode<Dim>(normCoords);
}

// ============================================================================
//  Child index computation from Morton code (for octree/quadtree navigation)
// ============================================================================

/**
 * @brief Given a parent Morton code and the child's relative position (0..7 for 3D),
 *        compute the child's Morton code.
 * @param parentCode Morton code of parent cell.
 * @param childIdx 0..(2^Dim -1) index (0=min, 1=max in each dimension).
 * @return Child's Morton code.
 */
constexpr uint64_t mortonChildCode(uint64_t parentCode, uint32_t childIdx, Dimension dim) noexcept {
    if (dim == Dim2) {
        // For 2D, childIdx 0..3: bits: bit0=X low? Actually we shift in two bits
        uint32_t childBits = static_cast<uint32_t>(childIdx);
        // shift parent left by 2 bits and insert child bits at LSB
        return (parentCode << 2) | childBits;
    } else {
        uint64_t childBits = static_cast<uint64_t>(childIdx & 0x7);
        return (parentCode << 3) | childBits;
    }
}

/**
 * @brief Get the child index (0..7) from a Morton code at a given depth.
 * @param code Full Morton code.
 * @param depth Depth (0 = root). Returns the 3 bits for that depth.
 * @param dim Dimension.
 */
constexpr uint32_t mortonChildIndex(uint64_t code, uint32_t depth, Dimension dim) noexcept {
    if (dim == Dim2) {
        return (code >> (2 * depth)) & 0x3;
    } else {
        return (code >> (3 * depth)) & 0x7;
    }
}

/**
 * @brief Compute the parent Morton code by shifting right.
 */
constexpr uint64_t mortonParentCode(uint64_t code, Dimension dim) noexcept {
    if (dim == Dim2) return code >> 2;
    else return code >> 3;
}

// ============================================================================
//  Prefix sum of bits (for LBVH construction)
// ============================================================================

/**
 * @brief Count leading zeros in a 32‑bit integer (compiler builtin wrapper).
 */
inline uint32_t countLeadingZeros(uint32_t x) noexcept {
    if (x == 0) return 32;
    return static_cast<uint32_t>(__builtin_clz(x));
}

/**
 * @brief Count leading zeros in 64‑bit integer.
 */
inline uint64_t countLeadingZeros(uint64_t x) noexcept {
    if (x == 0) return 64;
    return static_cast<uint64_t>(__builtin_clzll(x));
}

/**
 * @brief Count trailing zeros.
 */
inline uint32_t countTrailingZeros(uint32_t x) noexcept {
    if (x == 0) return 32;
    return static_cast<uint32_t>(__builtin_ctz(x));
}

inline uint64_t countTrailingZeros(uint64_t x) noexcept {
    if (x == 0) return 64;
    return static_cast<uint64_t>(__builtin_ctzll(x));
}

/**
 * @brief Find the highest set bit position (0‑based).
 */
inline uint32_t highestBitPos(uint32_t x) noexcept {
    return 31 - countLeadingZeros(x);
}

inline uint64_t highestBitPos(uint64_t x) noexcept {
    return 63 - countLeadingZeros(x);
}

/**
 * @brief Compute the length of common prefix of two Morton codes (number of shared bits from MSB).
 * @return Number of equal most significant bits.
 */
template <typename T>
constexpr uint32_t commonPrefixLength(T a, T b) noexcept {
    static_assert(std::is_integral_v<T>);
    T diff = a ^ b;
    if (diff == 0) return sizeof(T) * 8;
    return static_cast<uint32_t>(highestBitPos(diff)) + 1;
}

// ============================================================================
//  Bit arithmetic helpers for morton grid cells
// ============================================================================

/**
 * @brief Given a morton code, compute the morton code of the next cell along X axis.
 *        Uses bitwise addition without branching.
 */
constexpr uint64_t mortonNextX(uint64_t code) noexcept {
    return code + 0x1249249249249249ULL; // pattern for X increment in 3D Morton
}

constexpr uint32_t mortonNextX(uint32_t code) noexcept {
    return code + 0x09249249U;
}

/**
 * @brief Given a morton code, compute the morton code of the next cell along Y axis.
 */
constexpr uint64_t mortonNextY(uint64_t code) noexcept {
    return code + 0x2492492492492492ULL;
}

constexpr uint32_t mortonNextY(uint32_t code) noexcept {
    return code + 0x12492492U;
}

/**
 * @brief Given a morton code, compute the morton code of the next cell along Z axis.
 */
constexpr uint64_t mortonNextZ(uint64_t code) noexcept {
    return code + 0x4924924924924924ULL;
}

// ============================================================================
//  Transformation between floating point and morton coordinates
// ============================================================================

/**
 * @brief Convert floating point position to integer grid coordinates (0..(2^bits-1)).
 * @param pos Position in world space (range [min, max]).
 * @param min, max Bounds of the grid.
 * @param bits Number of bits for each coordinate (≤21 for 64‑bit morton).
 * @return Grid coordinate integer.
 */
template <typename T>
constexpr uint64_t floatToGrid(T pos, T min, T max, uint32_t bits) noexcept {
    T t = (pos - min) / (max - min);
    if (t < T(0)) t = T(0);
    if (t > T(1)) t = T(1);
    uint64_t maxVal = (uint64_t(1) << bits) - 1;
    return static_cast<uint64_t>(t * static_cast<T>(maxVal));
}

/**
 * @brief Convert grid coordinate back to floating point position (center of cell).
 */
template <typename T>
constexpr T gridToFloat(uint64_t grid, T min, T max, uint32_t bits) noexcept {
    T t = static_cast<T>(grid) / static_cast<T>((uint64_t(1) << bits) - 1);
    return min + t * (max - min);
}

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_BITSET_ARITHMETIC_H_INCLUDED