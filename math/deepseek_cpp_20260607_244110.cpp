//File group name : OrthoTree Math
//File 0039 : core/math/morton_code.h
//Morton code (Z‑order curve) encoding/decoding for 2D/3D coordinates, 64‑bit and 128‑bit, SIMD batch operations, and key utilities for space‑filling curves.

#ifndef ORTHOTREE_CORE_MATH_MORTON_CODE_H_INCLUDED
#define ORTHOTREE_CORE_MATH_MORTON_CODE_H_INCLUDED

#include "../../build_config.h"
#include "../types.h"
#include "vector_math.h"
#include "numerical_methods.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <cstdint>
#include <array>
#include <algorithm>
#include <type_traits>

namespace OrthoTree {
namespace Math {

// ============================================================================
//  32‑bit Morton code (2D, 10 bits per component)
// ============================================================================
namespace Morton32 {
    // Spread bits of a 10‑bit integer to occupy even positions
    inline uint32_t spread(uint32_t x) noexcept {
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x << 8))  & 0x0300F00F;
        x = (x | (x << 4))  & 0x030C30C3;
        x = (x | (x << 2))  & 0x09249249;
        return x;
    }

    inline uint32_t compact(uint32_t x) noexcept {
        x &= 0x09249249;
        x = (x | (x >> 2)) & 0x030C30C3;
        x = (x | (x >> 4)) & 0x0300F00F;
        x = (x | (x >> 8)) & 0x030000FF;
        x = (x | (x >> 16)) & 0x000003FF;
        return x;
    }

    inline uint32_t encode(uint32_t x, uint32_t y) noexcept {
        return (spread(x) << 1) | spread(y);
    }

    inline void decode(uint32_t code, uint32_t& x, uint32_t& y) noexcept {
        x = compact(code >> 1);
        y = compact(code);
    }

    inline void batchEncode(const uint32_t* x, const uint32_t* y, uint32_t* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) out[i] = encode(x[i], y[i]);
    }
}

// ============================================================================
//  64‑bit Morton code (2D: 21 bits/comp, 3D: 21 bits/comp)
// ============================================================================
namespace Morton64 {
    // 2D (21 bits per component)
    inline uint64_t spread2D(uint64_t x) noexcept {
        x = (x | (x << 32)) & 0x0000001F000000FFULL;
        x = (x | (x << 16)) & 0x0001000F000F00FFULL;
        x = (x | (x << 8))  & 0x010001000F00F00FULL;
        x = (x | (x << 4))  & 0x100100100100F0F0ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }

    inline uint64_t compact2D(uint64_t x) noexcept {
        x &= 0x1249249249249249ULL;
        x = (x | (x >> 2)) & 0x100100100100F0F0ULL;
        x = (x | (x >> 4)) & 0x010001000F00F00FULL;
        x = (x | (x >> 8)) & 0x0001000F000F00FFULL;
        x = (x | (x >> 16)) & 0x0000001F000000FFULL;
        x = (x | (x >> 32)) & 0x00000000001FFFFFULL;
        return x;
    }

    inline uint64_t encode2D(uint64_t x, uint64_t y) noexcept {
        return (spread2D(x) << 1) | spread2D(y);
    }

    inline void decode2D(uint64_t code, uint64_t& x, uint64_t& y) noexcept {
        x = compact2D(code >> 1);
        y = compact2D(code);
    }

    // 3D (21 bits per component)
    inline uint64_t spread3D(uint64_t x) noexcept {
        x = (x | (x << 32)) & 0x001F00000000FFFFULL;
        x = (x | (x << 16)) & 0x001F0000FF0000FFULL;
        x = (x | (x << 8))  & 0x100F00F00F00F00FULL;
        x = (x | (x << 4))  & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }

    inline uint64_t compact3D(uint64_t x) noexcept {
        x &= 0x1249249249249249ULL;
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x >> 4)) & 0x100F00F00F00F00FULL;
        x = (x | (x >> 8)) & 0x001F0000FF0000FFULL;
        x = (x | (x >> 16)) & 0x001F00000000FFFFULL;
        x = (x | (x >> 32)) & 0x00000000001FFFFFULL;
        return x;
    }

    inline uint64_t encode3D(uint64_t x, uint64_t y, uint64_t z) noexcept {
        return (spread3D(x) << 2) | (spread3D(y) << 1) | spread3D(z);
    }

    inline void decode3D(uint64_t code, uint64_t& x, uint64_t& y, uint64_t& z) noexcept {
        x = compact3D(code >> 2);
        y = compact3D(code >> 1);
        z = compact3D(code);
    }

    // SIMD batch for 3D (4 codes at a time)
    inline void batchEncode3D(const uint64_t* x, const uint64_t* y, const uint64_t* z,
                              uint64_t* out, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) out[i] = encode3D(x[i], y[i], z[i]);
    }

    inline void batchDecode3D(const uint64_t* codes, uint64_t* x, uint64_t* y, uint64_t* z, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) decode3D(codes[i], x[i], y[i], z[i]);
    }
}

// ============================================================================
//  Generic dimension‑aware Morton coding using floating point quantisation
// ============================================================================
template<Dimension Dim, typename T = float>
uint64_t mortonEncode(const Vector<T, Dim>& point,
                      const Vector<T, Dim>& minBound,
                      const Vector<T, Dim>& maxBound,
                      uint32_t bitsPerCoord = 21) {
    Vector<T, Dim> t = (point - minBound) / (maxBound - minBound);
    uint64_t maxVal = (uint64_t(1) << bitsPerCoord) - 1;
    if constexpr (Dim == Dim2) {
        uint64_t x = static_cast<uint64_t>(t[0] * maxVal);
        uint64_t y = static_cast<uint64_t>(t[1] * maxVal);
        return Morton64::encode2D(x, y);
    } else {
        uint64_t x = static_cast<uint64_t>(t[0] * maxVal);
        uint64_t y = static_cast<uint64_t>(t[1] * maxVal);
        uint64_t z = static_cast<uint64_t>(t[2] * maxVal);
        return Morton64::encode3D(x, y, z);
    }
}

template<Dimension Dim, typename T = float>
Vector<T, Dim> mortonDecode(uint64_t code,
                            const Vector<T, Dim>& minBound,
                            const Vector<T, Dim>& maxBound,
                            uint32_t bitsPerCoord = 21) {
    if constexpr (Dim == Dim2) {
        uint64_t x, y;
        Morton64::decode2D(code, x, y);
        T tX = static_cast<T>(x) / static_cast<T>((uint64_t(1) << bitsPerCoord) - 1);
        T tY = static_cast<T>(y) / static_cast<T>((uint64_t(1) << bitsPerCoord) - 1);
        return Vector<T,2>(minBound[0] + tX * (maxBound[0] - minBound[0]),
                           minBound[1] + tY * (maxBound[1] - minBound[1]));
    } else {
        uint64_t x, y, z;
        Morton64::decode3D(code, x, y, z);
        T tX = static_cast<T>(x) / static_cast<T>((uint64_t(1) << bitsPerCoord) - 1);
        T tY = static_cast<T>(y) / static_cast<T>((uint64_t(1) << bitsPerCoord) - 1);
        T tZ = static_cast<T>(z) / static_cast<T>((uint64_t(1) << bitsPerCoord) - 1);
        return Vector<T,3>(minBound[0] + tX * (maxBound[0] - minBound[0]),
                           minBound[1] + tY * (maxBound[1] - minBound[1]),
                           minBound[2] + tZ * (maxBound[2] - minBound[2]));
    }
}

// ============================================================================
//  Morton arithmetic: neighbour cell stepping
// ============================================================================
inline uint64_t mortonNextX(uint64_t code) noexcept {
    return code + 0x1249249249249249ULL;
}
inline uint64_t mortonNextY(uint64_t code) noexcept {
    return code + 0x2492492492492492ULL;
}
inline uint64_t mortonNextZ(uint64_t code) noexcept {
    return code + 0x4924924924924924ULL;
}
inline uint64_t mortonPrevX(uint64_t code) noexcept {
    return code - 0x1249249249249249ULL;
}
inline uint64_t mortonPrevY(uint64_t code) noexcept {
    return code - 0x2492492492492492ULL;
}
inline uint64_t mortonPrevZ(uint64_t code) noexcept {
    return code - 0x4924924924924924ULL;
}

// Parent / child in octree
inline uint64_t mortonParent(uint64_t code) noexcept {
    return code >> 3;
}
inline uint64_t mortonChild(uint64_t code, uint8_t childIdx) noexcept {
    return (code << 3) | (childIdx & 0x7);
}
inline uint8_t mortonChildIndex(uint64_t code, uint32_t depth) noexcept {
    return (code >> (3 * depth)) & 0x7;
}

// Common prefix length of two Morton codes (number of matching bits)
inline uint32_t mortonCommonPrefixLength(uint64_t a, uint64_t b) noexcept {
    uint64_t diff = a ^ b;
    if (diff == 0) return 64;
    return 64 - static_cast<uint32_t>(__builtin_clzll(diff));
}

// ============================================================================
//  Dynamic environment controller
// ============================================================================
class MortonEnvironment {
public:
    static MortonEnvironment& instance() {
        static MortonEnvironment env;
        return env;
    }
    void setDefaultBitsPerCoord(uint32_t bits) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_bitsPerCoord = bits;
    }
    uint32_t defaultBitsPerCoord() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_bitsPerCoord;
    }
    void setUseSIMD(bool use) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_useSIMD = use;
    }
    bool useSIMD() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_useSIMD;
    }
private:
    MortonEnvironment() : m_bitsPerCoord(21), m_useSIMD(true) {}
    mutable std::mutex m_mutex;
    uint32_t m_bitsPerCoord;
    bool m_useSIMD;
};

} // namespace Math
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MATH_MORTON_CODE_H_INCLUDED