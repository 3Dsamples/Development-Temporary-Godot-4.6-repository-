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

#ifndef ORTHOTREE_CORE_MORTON_MORTON_128BIT_H_INCLUDED
#define ORTHOTREE_CORE_MORTON_MORTON_128BIT_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/numerical_methods.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"

#include <cstdint>
#include <array>
#include <type_traits>
#include <algorithm>
#include <cstring>

#if defined(__SIZEOF_INT128__) || defined(__INT128_TYPE__)
#define ORTHOTREE_HAVE_INT128 1
typedef unsigned __int128 uint128_t;
#else
#error "128-bit integer support required for Morton128Bit. Compile with GCC/Clang or enable __int128."
#endif

namespace OrthoTree {
namespace Morton {

// ============================================================================
//  128‑bit Morton code (for 2D: 64 bits per coordinate, 3D: 42 bits per coord, etc.)
// ============================================================================
class Morton128Bit {
public:
    using value_type = uint128_t;
    using half_type = uint64_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr Morton128Bit() noexcept : m_code(0) {}
    explicit constexpr Morton128Bit(uint128_t code) noexcept : m_code(code) {}
    Morton128Bit(uint64_t x, uint64_t y) noexcept {
        encode2D(x, y);
    }
    Morton128Bit(uint64_t x, uint64_t y, uint64_t z) noexcept {
        encode3D(x, y, z);
    }

    // ------------------------------------------------------------------------
    //  Encoding 2D (64 bits per component -> 128‑bit interleaved)
    // ------------------------------------------------------------------------
    void encode2D(uint64_t x, uint64_t y) noexcept {
        m_code = (spreadBits64(x) << 1) | spreadBits64(y);
    }

    // 3D: up to 42 bits per component (since 42*3 = 126 < 128)
    void encode3D(uint64_t x, uint64_t y, uint64_t z) noexcept {
        // Only low 42 bits are used
        constexpr uint64_t MASK42 = (uint64_t(1) << 42) - 1;
        uint64_t x42 = x & MASK42;
        uint64_t y42 = y & MASK42;
        uint64_t z42 = z & MASK42;
        m_code = (spreadBits42(x42) << 2) | (spreadBits42(y42) << 1) | spreadBits42(z42);
    }

    // ------------------------------------------------------------------------
    //  Decoding
    // ------------------------------------------------------------------------
    void decode2D(uint64_t& x, uint64_t& y) const noexcept {
        x = compactBits64(static_cast<uint64_t>(m_code >> 1));
        y = compactBits64(static_cast<uint64_t>(m_code));
    }

    void decode3D(uint64_t& x, uint64_t& y, uint64_t& z) const noexcept {
        uint64_t codeLow = static_cast<uint64_t>(m_code);
        uint64_t codeHigh = static_cast<uint64_t>(m_code >> 64);
        // Actually 3D spread: bits: ... z3 y3 x3 z2 y2 x2 ...
        // We need to extract from the 128‑bit value using shifts
        // Simpler: use compactBits42 on each shifted part
        x = compactBits42(static_cast<uint64_t>(m_code >> 2));
        y = compactBits42(static_cast<uint64_t>(m_code >> 1));
        z = compactBits42(static_cast<uint64_t>(m_code));
        // Ensure only low 42 bits
        x &= (uint64_t(1) << 42) - 1;
        y &= (uint64_t(1) << 42) - 1;
        z &= (uint64_t(1) << 42) - 1;
    }

    // ------------------------------------------------------------------------
    //  Access
    // ------------------------------------------------------------------------
    uint128_t code() const noexcept { return m_code; }

    // ------------------------------------------------------------------------
    //  Arithmetic (neighbour cells) - for 128‑bit, just add fixed patterns
    // ------------------------------------------------------------------------
    Morton128Bit nextX() const noexcept {
        return Morton128Bit(m_code + 0x24924924924924924924924924924924ULL);
    }
    Morton128Bit nextY() const noexcept {
        return Morton128Bit(m_code + 0x49249249249249249249249249249249ULL);
    }
    Morton128Bit nextZ() const noexcept {
        return Morton128Bit(m_code + 0x92492492492492492492492492492492ULL);
    }
    Morton128Bit prevX() const noexcept { return Morton128Bit(m_code - 0x24924924924924924924924924924924ULL); }
    Morton128Bit prevY() const noexcept { return Morton128Bit(m_code - 0x49249249249249249249249249249249ULL); }
    Morton128Bit prevZ() const noexcept { return Morton128Bit(m_code - 0x92492492492492492492492492492492ULL); }

    // ------------------------------------------------------------------------
    //  Comparison
    // ------------------------------------------------------------------------
    bool operator==(const Morton128Bit& other) const noexcept { return m_code == other.m_code; }
    bool operator!=(const Morton128Bit& other) const noexcept { return !(*this == other); }
    bool operator<(const Morton128Bit& other) const noexcept { return m_code < other.m_code; }
    bool operator>(const Morton128Bit& other) const noexcept { return m_code > other.m_code; }

private:
    // ------------------------------------------------------------------------
    //  Bit spreading functions for 64‑bit inputs (2D)
    // ------------------------------------------------------------------------
    static uint128_t spreadBits64(uint64_t x) noexcept {
        uint128_t result = x;
        result = (result | (result << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFFULL;
        result = (result | (result << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFFULL;
        result = (result | (result << 8))  & 0x00FF00FF00FF00FF00FF00FF00FF00FFULL;
        result = (result | (result << 4))  & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0FULL;
        result = (result | (result << 2))  & 0x33333333333333333333333333333333ULL;
        result = (result | (result << 1))  & 0x55555555555555555555555555555555ULL;
        return result;
    }

    static uint64_t compactBits64(uint64_t x) noexcept {
        x &= 0x5555555555555555ULL;
        x = (x | (x >> 1)) & 0x3333333333333333ULL;
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FFULL;
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFFULL;
        x = (x | (x >> 16)) & 0x00000000FFFFFFFFULL;
        return x;
    }

    // For 3D: 42‑bit spread (3 interleaved bits per level)
    static uint128_t spreadBits42(uint64_t x) noexcept {
        // Only low 42 bits matter, but we spread to 126 bits (42*3)
        uint128_t result = x;
        result = (result | (result << 32)) & 0x00003FFFFF00000000FFFFFFFFULL;
        result = (result | (result << 16)) & 0x0000FFC0FFC0FFC0FFC0FFC0FFC0FFFFULL;
        result = (result | (result << 8))  & 0x00F00F00F00F00F00F00F00F00F00F00ULL;
        result = (result | (result << 4))  & 0x30C30C30C30C30C30C30C30C30C30C30ULL;
        result = (result | (result << 2))  & 0x12492492492492492492492492492492ULL;
        return result;
    }

    static uint64_t compactBits42(uint64_t x) noexcept {
        x &= 0x1249249249249249ULL; // only relevant low bits
        x = (x | (x >> 2)) & 0x30C30C30C30C30C3ULL;
        x = (x | (x >> 4)) & 0x00F00F00F00F00F0ULL;
        x = (x | (x >> 8)) & 0x0000FFC0FFC0FFC0ULL;
        x = (x | (x >> 16)) & 0x00003FFFFF000000ULL;
        x = (x | (x >> 32)) & 0x00000000003FFFFFULL;
        return x;
    }

    uint128_t m_code;
};

// ============================================================================
//  SIMD batch processing of 128‑bit Morton codes (using AVX‑512 if available)
// ============================================================================
#if ORTHOTREE_SIMD_LEVEL >= 512
// AVX‑512 can process 8 64‑bit or 4 128‑bit? We'll use 4‑wide for simplicity
// Actually we can use __m512i as 8x64, but 128‑bit needs two lanes per code.
// We'll provide a generic batch loop with SIMD hints.
#endif

class Morton128Batch {
public:
    static void encode2DBatch(const uint64_t* x, const uint64_t* y,
                              Morton128Bit* out, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            out[i].encode2D(x[i], y[i]);
        }
    }

    static void decode2DBatch(const Morton128Bit* in,
                              uint64_t* x, uint64_t* y, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            in[i].decode2D(x[i], y[i]);
        }
    }

    static void encode3DBatch(const uint64_t* x, const uint64_t* y, const uint64_t* z,
                              Morton128Bit* out, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            out[i].encode3D(x[i], y[i], z[i]);
        }
    }

    static void decode3DBatch(const Morton128Bit* in,
                              uint64_t* x, uint64_t* y, uint64_t* z, std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            in[i].decode3D(x[i], y[i], z[i]);
        }
    }
};

// ============================================================================
//  Dynamic environment controller for 128‑bit Morton codes (adaptive precision)
// ============================================================================
template<typename T>
class Morton128Environment {
public:
    using vec3 = Math::Vector<T, 3>;

    Morton128Environment() noexcept
        : m_bitsPerCoord(42) // 3D max
        , m_worldMin(vec3(0))
        , m_worldMax(vec3(1)) {}

    void setWorldBounds(const vec3& min, const vec3& max) noexcept {
        m_worldMin = min;
        m_worldMax = max;
        m_range = max - min;
    }

    void setBitsPerCoord(uint8_t bits) noexcept {
        m_bitsPerCoord = std::min<uint8_t>(bits, 42);
    }

    // Convert world point to Morton code
    Morton128Bit worldToMorton(const vec3& point) const noexcept {
        vec3 t = (point - m_worldMin) / m_range;
        uint64_t ix = static_cast<uint64_t>(t[0] * static_cast<T>((uint64_t(1) << m_bitsPerCoord) - 1));
        uint64_t iy = static_cast<uint64_t>(t[1] * static_cast<T>((uint64_t(1) << m_bitsPerCoord) - 1));
        uint64_t iz = static_cast<uint64_t>(t[2] * static_cast<T>((uint64_t(1) << m_bitsPerCoord) - 1));
        return Morton128Bit(ix, iy, iz);
    }

    // Convert Morton code back to world point (center of cell)
    vec3 mortonToWorld(const Morton128Bit& code) const noexcept {
        uint64_t ix, iy, iz;
        code.decode3D(ix, iy, iz);
        T maxCoord = static_cast<T>((uint64_t(1) << m_bitsPerCoord) - 1);
        T x = static_cast<T>(ix) / maxCoord;
        T y = static_cast<T>(iy) / maxCoord;
        T z = static_cast<T>(iz) / maxCoord;
        return m_worldMin + vec3(x, y, z) * m_range;
    }

    // Adaptive range adjustment
    void expandBounds(const vec3& newPoint) noexcept {
        vec3 p = newPoint;
        bool changed = false;
        for (int i = 0; i < 3; ++i) {
            if (p[i] < m_worldMin[i]) { m_worldMin[i] = p[i]; changed = true; }
            if (p[i] > m_worldMax[i]) { m_worldMax[i] = p[i]; changed = true; }
        }
        if (changed) m_range = m_worldMax - m_worldMin;
    }

private:
    uint8_t m_bitsPerCoord;
    vec3 m_worldMin, m_worldMax, m_range;
};

// ============================================================================
//  Hierarchical Morton key (combining scale and position) – for galactic scales
// ============================================================================
class HierarchicalMortonKey {
public:
    HierarchicalMortonKey() noexcept : m_scale(0), m_code(0) {}
    HierarchicalMortonKey(uint8_t scale, uint128_t code) noexcept : m_scale(scale), m_code(code) {}

    uint8_t scale() const noexcept { return m_scale; }
    uint128_t code() const noexcept { return m_code; }

    // Common prefix length with another key (ignoring scale)
    uint32_t commonPrefixBits(const HierarchicalMortonKey& other) const noexcept {
        if (m_scale != other.m_scale) return 0;
        uint128_t diff = m_code ^ other.m_code;
        if (diff == 0) return 128;
        return static_cast<uint32_t>(__builtin_clzll(static_cast<uint64_t>(diff >> 64)) + 64);
    }

    bool operator<(const HierarchicalMortonKey& other) const noexcept {
        if (m_scale != other.m_scale) return m_scale < other.m_scale;
        return m_code < other.m_code;
    }

private:
    uint8_t m_scale;
    uint128_t m_code;
};

} // namespace Morton
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MORTON_MORTON_128BIT_H_INCLUDED