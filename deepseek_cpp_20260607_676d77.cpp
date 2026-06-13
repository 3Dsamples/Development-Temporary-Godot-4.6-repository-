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
 * @file si_morton.h
 * @brief Morton code (Z-order curve) utilities for spatial indexing.
 *
 * This file provides high‑performance functions for encoding/decoding
 * 2D and 3D coordinates into Morton codes (interleaved bits). It includes:
 * - 32‑bit encoding for 2D (10 bits per component)
 * - 64‑bit encoding for 2D (21 bits per component) and 3D (21 bits per component)
 * - Fast bit spreading and compacting using magic numbers and shifts
 * - Coordinate quantization from floating point to integer grid
 * - Neighbour computation: next/previous cell in X/Y/Z direction via bit addition
 * - Hierarchical properties: parent/child codes, common prefix length
 *
 * All functions are constexpr where possible and use branchless arithmetic.
 * They are essential for linear BVH (LBVH) construction and fast spatial hashing.
 */

#ifndef ORTHOTREE_DETAIL_SI_MORTON_H_INCLUDED
#define ORTHOTREE_DETAIL_SI_MORTON_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "common.h"
#include <cstdint>
#include <limits>
#include <type_traits>

namespace OrthoTree {
namespace detail {

// ============================================================================
//  Bit spreading (interleaving) – 2D, 32‑bit
// ============================================================================

/**
 * @brief Spread bits of a 10‑bit integer to occupy every other bit (positions 0,2,4,...).
 *        Used for 2D Morton coding (32‑bit result).
 */
constexpr uint32_t spreadBits2D_32(uint32_t x) noexcept {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    return x;
}

/**
 * @brief Compact bits (inverse of spreadBits2D_32). Extracts original 10‑bit value.
 */
constexpr uint32_t compactBits2D_32(uint32_t x) noexcept {
    x &= 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x = (x | (x >> 16)) & 0x000003FF;
    return x;
}

/**
 * @brief Encode 2D coordinates (0..1023) into 32‑bit Morton code.
 */
constexpr uint32_t mortonEncode2D_32(uint32_t x, uint32_t y) noexcept {
    return (spreadBits2D_32(x) << 1) | spreadBits2D_32(y);
}

/**
 * @brief Decode 32‑bit Morton code back to 2D coordinates.
 */
constexpr void mortonDecode2D_32(uint32_t code, uint32_t& x, uint32_t& y) noexcept {
    x = compactBits2D_32(code >> 1);
    y = compactBits2D_32(code);
}

// ============================================================================
//  Bit spreading – 2D, 64‑bit (21 bits per component)
// ============================================================================

/**
 * @brief Spread bits of a 21‑bit integer to occupy even bits (0,2,4,...,62).
 *        Used for 2D Morton code in 64‑bit (42 bits used).
 */
constexpr uint64_t spreadBits2D_64(uint64_t x) noexcept {
    x = (x | (x << 32)) & 0x0000001F000000FFULL;
    x = (x | (x << 16)) & 0x0001000F000F00FFULL;
    x = (x | (x << 8))  & 0x010001000F00F00FULL;
    x = (x | (x << 4))  & 0x100100100100F0F0ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

/**
 * @brief Compact 2D bits from 64‑bit Morton code.
 */
constexpr uint64_t compactBits2D_64(uint64_t x) noexcept {
    x &= 0x1249249249249249ULL;
    x = (x | (x >> 2)) & 0x100100100100F0F0ULL;
    x = (x | (x >> 4)) & 0x010001000F00F00FULL;
    x = (x | (x >> 8)) & 0x0001000F000F00FFULL;
    x = (x | (x >> 16)) & 0x0000001F000000FFULL;
    x = (x | (x >> 32)) & 0x00000000001FFFFFULL;
    return x;
}

/**
 * @brief Encode 2D coordinates (0..2097151) into 64‑bit Morton code.
 */
constexpr uint64_t mortonEncode2D_64(uint64_t x, uint64_t y) noexcept {
    return (spreadBits2D_64(x) << 1) | spreadBits2D_64(y);
}

/**
 * @brief Decode 64‑bit 2D Morton code.
 */
constexpr void mortonDecode2D_64(uint64_t code, uint64_t& x, uint64_t& y) noexcept {
    x = compactBits2D_64(code >> 1);
    y = compactBits2D_64(code);
}

// ============================================================================
//  3D Morton coding (64‑bit, 21 bits per component)
// ============================================================================

/**
 * @brief Spread bits of a 21‑bit integer to occupy every third bit (0,3,6,...).
 */
constexpr uint64_t spreadBits3D(uint64_t x) noexcept {
    x = (x | (x << 32)) & 0x001F00000000FFFFULL;
    x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
    x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

/**
 * @brief Compact 3D bits from 64‑bit Morton code.
 */
constexpr uint64_t compactBits3D(uint64_t x) noexcept {
    x &= 0x1249249249249249ULL;
    x = (x | (x >> 2)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x >> 4)) & 0x100F00F00F00F00FULL;
    x = (x | (x >> 8)) & 0x001F0000FF0000FFULL;
    x = (x | (x >> 16)) & 0x001F00000000FFFFULL;
    x = (x | (x >> 32)) & 0x00000000001FFFFFULL;
    return x;
}

/**
 * @brief Encode 3D coordinates into 64‑bit Morton code.
 *        Each coordinate must be in [0, 2097151].
 */
constexpr uint64_t mortonEncode3D(uint64_t x, uint64_t y, uint64_t z) noexcept {
    return (spreadBits3D(x) << 2) | (spreadBits3D(y) << 1) | spreadBits3D(z);
}

/**
 * @brief Decode 64‑bit 3D Morton code.
 */
constexpr void mortonDecode3D(uint64_t code, uint64_t& x, uint64_t& y, uint64_t& z) noexcept {
    x = compactBits3D(code >> 2);
    y = compactBits3D(code >> 1);
    z = compactBits3D(code);
}

// ============================================================================
//  Generic dimension‑aware encoding (using floating point quantization)
// ============================================================================

/**
 * @brief Quantize a floating point value to integer grid with given bits.
 * @param value Normalised coordinate in [0,1].
 * @param bits Number of bits (<= 21 for 64‑bit morton).
 * @return Integer in [0, 2^bits -1].
 */
constexpr uint64_t quantize(T value, uint32_t bits) noexcept {
    uint64_t maxVal = (uint64_t(1) << bits) - 1;
    if (value <= T(0)) return 0;
    if (value >= T(1)) return maxVal;
    return static_cast<uint64_t>(value * static_cast<T>(maxVal));
}

/**
 * @brief Encode 2D/3D point from world coordinates to Morton code.
 * @tparam Dim Dimension (2 or 3).
 * @param point Point in world space (not yet normalised).
 * @param min, max Bounds of the grid.
 * @param bits Number of bits per coordinate.
 */
template <Dimension Dim, typename T>
uint64_t mortonEncodeWorld(const Math::Vector<T, Dim>& point,
                           const Math::Vector<T, Dim>& min,
                           const Math::Vector<T, Dim>& max,
                           uint32_t bits = 21) {
    Math::Vector<T, Dim> t = (point - min) / (max - min);
    if constexpr (Dim == Dim2) {
        uint64_t x = quantize(t[0], bits);
        uint64_t y = quantize(t[1], bits);
        return mortonEncode2D_64(x, y);
    } else if constexpr (Dim == Dim3) {
        uint64_t x = quantize(t[0], bits);
        uint64_t y = quantize(t[1], bits);
        uint64_t z = quantize(t[2], bits);
        return mortonEncode3D(x, y, z);
    }
}

// ============================================================================
//  Morton arithmetic (neighbour cells)
// ============================================================================

/**
 * @brief Given a Morton code, compute code of the cell one step in +X direction.
 *        Uses bitwise addition without branching.
 */
constexpr uint64_t mortonNextX(uint64_t code) noexcept {
    return code + 0x1249249249249249ULL; // pattern for 3D, also works for 2D with masking?
}
constexpr uint32_t mortonNextX(uint32_t code) noexcept {
    return code + 0x09249249U;
}

/**
 * @brief Step in +Y direction.
 */
constexpr uint64_t mortonNextY(uint64_t code) noexcept {
    return code + 0x2492492492492492ULL;
}
constexpr uint32_t mortonNextY(uint32_t code) noexcept {
    return code + 0x12492492U;
}

/**
 * @brief Step in +Z direction (3D only).
 */
constexpr uint64_t mortonNextZ(uint64_t code) noexcept {
    return code + 0x4924924924924924ULL;
}

/**
 * @brief Step in -X direction.
 */
constexpr uint64_t mortonPrevX(uint64_t code) noexcept {
    return code - 0x1249249249249249ULL;
}
constexpr uint32_t mortonPrevX(uint32_t code) noexcept {
    return code - 0x09249249U;
}

/**
 * @brief Compute the parent Morton code (shift right by dim bits).
 */
template <Dimension Dim>
constexpr uint64_t mortonParent(uint64_t code) noexcept {
    if constexpr (Dim == Dim2) return code >> 2;
    else return code >> 3;
}

/**
 * @brief Compute child code for given child index (0..2^Dim-1).
 */
template <Dimension Dim>
constexpr uint64_t mortonChild(uint64_t parentCode, uint8_t childIdx) noexcept {
    if constexpr (Dim == Dim2) return (parentCode << 2) | (childIdx & 0x3);
    else return (parentCode << 3) | (childIdx & 0x7);
}

/**
 * @brief Extract child index from a code at a given depth (0 = root).
 */
template <Dimension Dim>
constexpr uint8_t mortonChildIndex(uint64_t code, uint32_t depth) noexcept {
    if constexpr (Dim == Dim2) return (code >> (2 * depth)) & 0x3;
    else return (code >> (3 * depth)) & 0x7;
}

/**
 * @brief Number of common leading bits (depth) of two Morton codes.
 */
template <typename T>
constexpr uint32_t commonPrefixBits(T a, T b) noexcept {
    T diff = a ^ b;
    if (diff == 0) return sizeof(T) * 8;
    return static_cast<uint32_t>(std::numeric_limits<T>::digits - countLeadingZeros(diff));
}

/**
 * @brief Get the depth (number of levels) from a Morton code's most significant non‑zero bit.
 *        Assumes bits per level = 2 (2D) or 3 (3D).
 */
template <Dimension Dim>
constexpr uint32_t mortonDepth(uint64_t code) noexcept {
    if (code == 0) return 0;
    uint32_t bits = static_cast<uint32_t>(highestBitPos(code));
    return bits / (Dim == Dim2 ? 2 : 3);
}

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_SI_MORTON_H_INCLUDED