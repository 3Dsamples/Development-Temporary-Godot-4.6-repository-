// system name : onetbb-warp
// File 0047 : core/math/morton.h
// Description : Morton (Z‑order) encoding/decoding, Hilbert curve index, cache‑oblivious iteration, octree navigation.

#ifndef __TBB_WARP_CORE_MATH_MORTON_H
#define __TBB_WARP_CORE_MATH_MORTON_H

#include "core/math/constants.h"
#include "core/math/scalar.h"
#include <cstdint>
#include <array>
#include <algorithm>
#include <type_traits>

namespace tbb {
namespace core {
namespace math {

// ============================================================
// Spread bits: interleave zeros between bits of 32‑bit integer
// ============================================================

inline std::uint64_t spread_bits_2d(std::uint32_t v) noexcept {
    std::uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

// ============================================================
// Compact bits: undo spread_bits_2d
// ============================================================

inline std::uint32_t compact_bits_2d(std::uint64_t x) noexcept {
    x &= 0x5555555555555555ULL;
    x = (x ^ (x >> 1))  & 0x3333333333333333ULL;
    x = (x ^ (x >> 2))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x ^ (x >> 4))  & 0x00FF00FF00FF00FFULL;
    x = (x ^ (x >> 8))  & 0x0000FFFF0000FFFFULL;
    x = (x ^ (x >> 16)) & 0x00000000FFFFFFFFULL;
    return static_cast<std::uint32_t>(x);
}

// ============================================================
// 2D Morton encode / decode
// ============================================================

inline std::uint64_t morton_encode_2d(std::uint32_t x, std::uint32_t y) noexcept {
    return spread_bits_2d(x) | (spread_bits_2d(y) << 1);
}

inline void morton_decode_2d(std::uint64_t code, std::uint32_t& x, std::uint32_t& y) noexcept {
    x = compact_bits_2d(code);
    y = compact_bits_2d(code >> 1);
}

// ============================================================
// 3D Morton encode / decode
// ============================================================

inline std::uint64_t spread_bits_3d(std::uint32_t v) noexcept {
    std::uint64_t x = v & 0x1FFFFF; // 21 bits
    x = (x | (x << 32)) & 0x001F00000000FFFFULL;
    x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
    x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

inline std::uint32_t compact_bits_3d(std::uint64_t x) noexcept {
    x &= 0x1249249249249249ULL;
    x = (x ^ (x >> 2))  & 0x10C30C30C30C30C3ULL;
    x = (x ^ (x >> 4))  & 0x100F00F00F00F00FULL;
    x = (x ^ (x >> 8))  & 0x001F0000FF0000FFULL;
    x = (x ^ (x >> 16)) & 0x001F00000000FFFFULL;
    x = (x ^ (x >> 32)) & 0x00000000001FFFFFULL;
    return static_cast<std::uint32_t>(x);
}

inline std::uint64_t morton_encode_3d(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2);
}

inline void morton_decode_3d(std::uint64_t code, std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) noexcept {
    x = compact_bits_3d(code);
    y = compact_bits_3d(code >> 1);
    z = compact_bits_3d(code >> 2);
}

// ============================================================
// Octree node addressing at a given depth
// ============================================================

inline std::uint64_t morton_at_depth(std::uint64_t full_morton, std::uint32_t depth) noexcept {
    std::uint32_t shift = 3 * (21 - depth);
    std::uint64_t mask = (std::uint64_t{1} << (3 * depth)) - 1;
    return (full_morton >> shift) & mask;
}

inline std::uint64_t parent_morton(std::uint64_t child_morton) noexcept {
    return child_morton >> 3;
}

inline std::uint64_t child_morton(std::uint64_t parent_morton, std::uint8_t octant) noexcept {
    return (parent_morton << 3) | (octant & 0x7);
}

// ============================================================
// Hilbert curve (2D) – encode / decode
// ============================================================

inline std::uint64_t hilbert_encode_2d(std::uint32_t x, std::uint32_t y, std::uint32_t order) noexcept {
    std::uint64_t index = 0;
    for (std::int32_t s = static_cast<std::int32_t>(order) - 1; s >= 0; --s) {
        std::uint64_t rx = (x >> s) & 1;
        std::uint64_t ry = (y >> s) & 1;
        index = (index << 2) | ((rx << 1) ^ ry);
        if (ry == 0) {
            if (rx == 1) { x ^= (1u << s) - 1; y ^= (1u << s) - 1; }
        } else {
            if (rx == 0) { x ^= (1u << s) - 1; y ^= (1u << s) - 1; }
        }
    }
    return index;
}

inline void hilbert_decode_2d(std::uint64_t index, std::uint32_t order, std::uint32_t& x, std::uint32_t& y) noexcept {
    x = 0; y = 0;
    for (std::int32_t s = 0; s < static_cast<std::int32_t>(order); ++s) {
        std::uint64_t quad = (index >> (2 * (order - 1 - s))) & 3;
        std::uint64_t rx = (quad >> 1) ^ (quad & 1);
        std::uint64_t ry = quad ^ rx;
        x = (x << 1) | static_cast<std::uint32_t>(rx);
        y = (y << 1) | static_cast<std::uint32_t>(ry);
        if (ry == 0) {
            if (rx == 1) { x ^= (1u << (s+1)) - 1; y ^= (1u << (s+1)) - 1; }
        } else {
            if (rx == 0) { x ^= (1u << (s+1)) - 1; y ^= (1u << (s+1)) - 1; }
        }
    }
}

// ============================================================
// Cache‑oblivious iteration over a 2D range in Morton order
// ============================================================

template<typename Func>
void for_each_morton_2d(std::uint32_t width, std::uint32_t height, Func&& func) noexcept(noexcept(func(0u,0u))) {
    for (std::uint32_t i = 0; i < width * height; ++i) {
        std::uint32_t x = 0, y = 0;
        std::uint32_t z = i;
        int bit = 0;
        while (z) {
            x |= (z & 1) << bit;
            z >>= 1;
            y |= (z & 1) << bit;
            z >>= 1;
            ++bit;
        }
        if (x < width && y < height) func(x, y);
    }
}

// ============================================================
// Morton order sorting of indices
// ============================================================

inline std::vector<std::uint64_t> sort_by_morton_2d(const std::vector<std::array<std::uint32_t, 2>>& points) noexcept {
    std::vector<std::pair<std::uint64_t, std::size_t>> morton_pairs;
    morton_pairs.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        morton_pairs.emplace_back(morton_encode_2d(points[i][0], points[i][1]), i);
    }
    std::sort(morton_pairs.begin(), morton_pairs.end());
    std::vector<std::uint64_t> indices(points.size());
    for (std::size_t i = 0; i < morton_pairs.size(); ++i) {
        indices[i] = morton_pairs[i].second;
    }
    return indices;
}

// ============================================================
// Octree cell coordinates from Morton code at given depth
// ============================================================

inline std::array<std::uint32_t, 3> morton_to_cell(std::uint64_t code, std::uint32_t depth) noexcept {
    std::uint32_t x = 0, y = 0, z = 0;
    for (std::uint32_t i = 0; i < depth; ++i) {
        x |= ((code >> (3*i)) & 1) << i;
        y |= ((code >> (3*i + 1)) & 1) << i;
        z |= ((code >> (3*i + 2)) & 1) << i;
    }
    return {x, y, z};
}

// ============================================================
// Morton order traversal of octree nodes at a given level
// ============================================================

template<typename Func>
void traverse_octree_level(std::uint32_t depth, Func&& func) noexcept {
    std::uint64_t total = std::uint64_t{1} << (3 * depth);
    for (std::uint64_t i = 0; i < total; ++i) {
        auto cell = morton_to_cell(i, depth);
        func(cell[0], cell[1], cell[2], i);
    }
}

// ============================================================
// Compute the Morton code range for a bounding box [min, max]
// ============================================================

inline std::pair<std::uint64_t, std::uint64_t> morton_range_2d(
    std::uint32_t min_x, std::uint32_t min_y,
    std::uint32_t max_x, std::uint32_t max_y) noexcept {
    return {morton_encode_2d(min_x, min_y), morton_encode_2d(max_x, max_y)};
}

inline std::pair<std::uint64_t, std::uint64_t> morton_range_3d(
    std::uint32_t min_x, std::uint32_t min_y, std::uint32_t min_z,
    std::uint32_t max_x, std::uint32_t max_y, std::uint32_t max_z) noexcept {
    return {morton_encode_3d(min_x, min_y, min_z), morton_encode_3d(max_x, max_y, max_z)};
}

// ============================================================
// Check if a Morton code lies within a range
// ============================================================

inline bool morton_in_range(std::uint64_t code, std::uint64_t min_code, std::uint64_t max_code) noexcept {
    return code >= min_code && code <= max_code;
}

// ============================================================
// Nearest common ancestor depth of two Morton codes
// ============================================================

inline std::uint32_t common_ancestor_depth(std::uint64_t a, std::uint64_t b, std::uint32_t max_depth = 21) noexcept {
    std::uint32_t depth = max_depth;
    while (depth > 0 && morton_at_depth(a, depth) != morton_at_depth(b, depth)) {
        --depth;
    }
    return depth;
}

} // namespace math
} // namespace core
} // namespace tbb

#endif // __TBB_WARP_CORE_MATH_MORTON_H