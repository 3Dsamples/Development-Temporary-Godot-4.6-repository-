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

#ifndef ORTHOTREE_CORE_MORTON_HIERARCHICAL_MORTON_KEY_H_INCLUDED
#define ORTHOTREE_CORE_MORTON_HIERARCHICAL_MORTON_KEY_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "../core/math/vector_math.h"
#include "../core/math/numerical_methods.h"
#include "../core/math/interval_arithmetic.h"
#include "../detail/common.h"
#include "../detail/simd_utils.h"
#include "morton_128bit.h"

#include <cstdint>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace OrthoTree {
namespace Morton {

// ============================================================================
//  HierarchicalMortonKey: combines scale (level-of-detail) and position
//  Scale is encoded as the number of leading zero bits in the code,
//  or stored explicitly. This allows efficient prefix‑based queries across scales.
// ============================================================================
class HierarchicalMortonKey {
public:
    using value_type = uint128_t;
    using scale_type = uint8_t;

    static constexpr scale_type MAX_SCALE = 64;   // up to 2^64 cells per dimension

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr HierarchicalMortonKey() noexcept : m_scale(0), m_code(0) {}
    constexpr HierarchicalMortonKey(scale_type scale, uint128_t code) noexcept
        : m_scale(scale), m_code(code) {}
    HierarchicalMortonKey(const Morton128Bit& morton, scale_type scale = MAX_SCALE) noexcept
        : m_scale(scale), m_code(morton.code()) {}

    // ------------------------------------------------------------------------
    //  Accessors
    // ------------------------------------------------------------------------
    scale_type scale() const noexcept { return m_scale; }
    uint128_t code() const noexcept { return m_code; }
    Morton128Bit morton() const noexcept { return Morton128Bit(m_code); }

    // ------------------------------------------------------------------------
    //  Hierarchical properties: parent, child, depth
    // ------------------------------------------------------------------------
    HierarchicalMortonKey parent() const noexcept {
        if (m_scale == 0) return *this;
        uint128_t shifted = m_code >> (dimensionBits());
        return HierarchicalMortonKey(m_scale - 1, shifted);
    }

    HierarchicalMortonKey child(uint8_t childIdx) const noexcept {
        uint128_t childCode = (m_code << dimensionBits()) | (childIdx & ((1 << dimensionBits()) - 1));
        scale_type childScale = m_scale + 1;
        if (childScale > MAX_SCALE) childScale = MAX_SCALE;
        return HierarchicalMortonKey(childScale, childCode);
    }

    // For 2D: dimensionBits = 2, for 3D: 3
    static constexpr uint8_t dimensionBits(Dimension dim = Dim3) noexcept {
        return (dim == Dim2) ? 2 : 3;
    }

    // Common prefix depth (number of matching levels from root)
    uint32_t commonPrefixDepth(const HierarchicalMortonKey& other) const noexcept {
        uint128_t diff = m_code ^ other.m_code;
        if (diff == 0) return std::min(m_scale, other.m_scale);
        uint32_t diffBits = (diff == 0) ? 0 : (128 - __builtin_clzll(static_cast<uint64_t>(diff >> 64)) + 64);
        uint32_t levels = diffBits / dimensionBits();
        return std::min({levels, m_scale, other.m_scale});
    }

    // ------------------------------------------------------------------------
    //  Comparison
    // ------------------------------------------------------------------------
    bool operator==(const HierarchicalMortonKey& other) const noexcept {
        return m_scale == other.m_scale && m_code == other.m_code;
    }
    bool operator!=(const HierarchicalMortonKey& other) const noexcept { return !(*this == other); }
    bool operator<(const HierarchicalMortonKey& other) const noexcept {
        if (m_scale != other.m_scale) return m_scale < other.m_scale;
        return m_code < other.m_code;
    }
    bool operator>(const HierarchicalMortonKey& other) const noexcept { return other < *this; }

private:
    scale_type m_scale;
    uint128_t m_code;
};

// ============================================================================
//  Building hierarchical keys from world coordinates with adaptive scale
// ============================================================================
template<typename T, Dimension Dim = Dim3>
class HierarchicalKeyBuilder {
public:
    using vec_type = Math::Vector<T, Dim>;
    using key_type = HierarchicalMortonKey;
    using morton_type = Morton128Bit;

    HierarchicalKeyBuilder() noexcept
        : m_worldMin(vec_type(0))
        , m_worldMax(vec_type(1))
        , m_maxScale(HierarchicalMortonKey::MAX_SCALE)
        , m_adaptiveScaling(true) {}

    void setWorldBounds(const vec_type& min, const vec_type& max) noexcept {
        m_worldMin = min;
        m_worldMax = max;
        m_range = max - min;
    }

    void setMaxScale(uint8_t scale) noexcept { m_maxScale = std::min(scale, HierarchicalMortonKey::MAX_SCALE); }
    void setAdaptiveScaling(bool enable) noexcept { m_adaptiveScaling = enable; }

    // Build key for a given point, automatically choosing scale based on distance to origin
    // or based on entity size. Here we use a simple heuristic: scale = log2(worldSize / entitySize)
    key_type buildKey(const vec_type& point, T entitySize = T(1)) const noexcept {
        // Compute required scale: we want cell size roughly equal to entity size
        T worldSize = m_range.maxComponent();
        if (worldSize <= T(0)) worldSize = T(1);
        int scaleEstimate = static_cast<int>(std::log2(worldSize / entitySize));
        uint8_t scale = static_cast<uint8_t>(Math::clamp(scaleEstimate, 0, static_cast<int>(m_maxScale)));
        return buildKeyAtScale(point, scale);
    }

    key_type buildKeyAtScale(const vec_type& point, uint8_t scale) const noexcept {
        // Quantize point to grid at given scale
        uint64_t cellsPerDim = (scale < 64) ? (uint64_t(1) << scale) : ~uint64_t(0);
        vec_type t = (point - m_worldMin) / m_range;
        uint64_t ix = static_cast<uint64_t>(t[0] * static_cast<T>(cellsPerDim - 1));
        uint64_t iy = static_cast<uint64_t>(t[1] * static_cast<T>(cellsPerDim - 1));
        uint64_t iz = (Dim == Dim3) ? static_cast<uint64_t>(t[2] * static_cast<T>(cellsPerDim - 1)) : 0;
        morton_type morton;
        if constexpr (Dim == Dim2) morton.encode2D(ix, iy);
        else morton.encode3D(ix, iy, iz);
        // Scale‑dependent shift: the code represents full resolution, but we only keep high bits
        uint128_t scaledCode = morton.code() >> ((HierarchicalMortonKey::MAX_SCALE - scale) * HierarchicalMortonKey::dimensionBits(Dim));
        return HierarchicalMortonKey(scale, scaledCode);
    }

    // Decode key back to world bounding box (the cell that the key represents)
    Math::AxisAlignedBox<T, Dim> decodeCell(const key_type& key) const noexcept {
        uint64_t cellsPerDim = (key.scale() < 64) ? (uint64_t(1) << key.scale()) : ~uint64_t(0);
        uint64_t ix, iy, iz = 0;
        morton_type morton(key.code());
        if constexpr (Dim == Dim2) morton.decode2D(ix, iy);
        else morton.decode3D(ix, iy, iz);
        // Clamp to valid range
        ix = std::min(ix, cellsPerDim - 1);
        iy = std::min(iy, cellsPerDim - 1);
        if constexpr (Dim == Dim3) iz = std::min(iz, cellsPerDim - 1);
        T x0 = m_worldMin[0] + (static_cast<T>(ix) / static_cast<T>(cellsPerDim)) * m_range[0];
        T x1 = m_worldMin[0] + (static_cast<T>(ix + 1) / static_cast<T>(cellsPerDim)) * m_range[0];
        T y0 = m_worldMin[1] + (static_cast<T>(iy) / static_cast<T>(cellsPerDim)) * m_range[1];
        T y1 = m_worldMin[1] + (static_cast<T>(iy + 1) / static_cast<T>(cellsPerDim)) * m_range[1];
        if constexpr (Dim == Dim2) {
            return Math::AxisAlignedBox<T, 2>(Math::Vector<T,2>(x0, y0), Math::Vector<T,2>(x1, y1));
        } else {
            T z0 = m_worldMin[2] + (static_cast<T>(iz) / static_cast<T>(cellsPerDim)) * m_range[2];
            T z1 = m_worldMin[2] + (static_cast<T>(iz + 1) / static_cast<T>(cellsPerDim)) * m_range[2];
            return Math::AxisAlignedBox<T, 3>(Math::Vector<T,3>(x0, y0, z0), Math::Vector<T,3>(x1, y1, z1));
        }
    }

private:
    vec_type m_worldMin, m_worldMax, m_range;
    uint8_t m_maxScale;
    bool m_adaptiveScaling;
};

// ============================================================================
//  SIMD batch processing of hierarchical keys (using 128‑bit vectors)
// ============================================================================
class HierarchicalKeyBatch {
public:
    using key_type = HierarchicalMortonKey;

    static void encodeBatch(const Math::Vector<float,3>* points,
                            const uint8_t* scales,
                            key_type* out,
                            std::size_t count,
                            const Math::Vector<float,3>& worldMin,
                            const Math::Vector<float,3>& worldMax) noexcept {
        Math::Vector<float,3> range = worldMax - worldMin;
        for (std::size_t i = 0; i < count; ++i) {
            Math::Vector<float,3> t = (points[i] - worldMin) / range;
            uint64_t cellsPerDim = (scales[i] < 64) ? (uint64_t(1) << scales[i]) : ~uint64_t(0);
            uint64_t ix = static_cast<uint64_t>(t[0] * static_cast<float>(cellsPerDim - 1));
            uint64_t iy = static_cast<uint64_t>(t[1] * static_cast<float>(cellsPerDim - 1));
            uint64_t iz = static_cast<uint64_t>(t[2] * static_cast<float>(cellsPerDim - 1));
            Morton128Bit morton(ix, iy, iz);
            uint128_t scaledCode = morton.code() >> ((HierarchicalMortonKey::MAX_SCALE - scales[i]) * 3);
            out[i] = HierarchicalMortonKey(scales[i], scaledCode);
        }
    }

    static void decodeBatch(const key_type* keys,
                            Math::AxisAlignedBox<float,3>* cells,
                            std::size_t count,
                            const Math::Vector<float,3>& worldMin,
                            const Math::Vector<float,3>& worldMax) noexcept {
        Math::Vector<float,3> range = worldMax - worldMin;
        for (std::size_t i = 0; i < count; ++i) {
            uint64_t cellsPerDim = (keys[i].scale() < 64) ? (uint64_t(1) << keys[i].scale()) : ~uint64_t(0);
            uint64_t ix, iy, iz;
            Morton128Bit morton(keys[i].code());
            morton.decode3D(ix, iy, iz);
            ix = std::min(ix, cellsPerDim - 1);
            iy = std::min(iy, cellsPerDim - 1);
            iz = std::min(iz, cellsPerDim - 1);
            float x0 = worldMin[0] + (static_cast<float>(ix) / static_cast<float>(cellsPerDim)) * range[0];
            float x1 = worldMin[0] + (static_cast<float>(ix + 1) / static_cast<float>(cellsPerDim)) * range[0];
            float y0 = worldMin[1] + (static_cast<float>(iy) / static_cast<float>(cellsPerDim)) * range[1];
            float y1 = worldMin[1] + (static_cast<float>(iy + 1) / static_cast<float>(cellsPerDim)) * range[1];
            float z0 = worldMin[2] + (static_cast<float>(iz) / static_cast<float>(cellsPerDim)) * range[2];
            float z1 = worldMin[2] + (static_cast<float>(iz + 1) / static_cast<float>(cellsPerDim)) * range[2];
            cells[i] = Math::AxisAlignedBox<float,3>(
                Math::Vector<float,3>(x0, y0, z0),
                Math::Vector<float,3>(x1, y1, z1)
            );
        }
    }
};

// ============================================================================
//  Dynamic environment controller for hierarchical keys (LOD selection)
// ============================================================================
template<typename T>
class HierarchicalKeyEnvironment {
public:
    using vec3 = Math::Vector<T, 3>;

    HierarchicalKeyEnvironment() noexcept
        : m_viewerPosition(vec3(0))
        , m_maxDistance(T(1e12))
        , m_minCellSize(T(1e-6))
        , m_detailBias(T(1))
        , m_adaptiveDetail(true) {}

    void setViewerPosition(const vec3& pos) noexcept { m_viewerPosition = pos; }
    void setMaxDistance(T dist) noexcept { m_maxDistance = dist; }
    void setMinCellSize(T size) noexcept { m_minCellSize = size; }
    void setDetailBias(T bias) noexcept { m_detailBias = bias; }
    void setAdaptiveDetail(bool enable) noexcept { m_adaptiveDetail = enable; }

    // Determine optimal scale for a given world point (distance‑based LOD)
    uint8_t computeScale(const vec3& point, T worldSize, uint8_t maxScale) const noexcept {
        if (!m_adaptiveDetail) return maxScale;
        T distance = (point - m_viewerPosition).length();
        distance = std::max(distance, T(1e-6));
        // Ideal scale: cell size in world units should be proportional to distance
        T desiredCellSize = (distance / m_maxDistance) * worldSize * m_detailBias;
        desiredCellSize = Math::clamp(desiredCellSize, m_minCellSize, worldSize);
        int scale = static_cast<int>(std::log2(worldSize / desiredCellSize));
        return static_cast<uint8_t>(Math::clamp(scale, 0, static_cast<int>(maxScale)));
    }

    // Batch compute scales for many points (SIMD friendly)
    void batchComputeScales(const vec3* points, uint8_t* scales, std::size_t count,
                            T worldSize, uint8_t maxScale) const noexcept {
        if constexpr (ORTHOTREE_SIMD_LEVEL >= 128) {
            // SIMD loop using 4‑wide floats (pseudo)
            for (std::size_t i = 0; i < count; ++i) {
                scales[i] = computeScale(points[i], worldSize, maxScale);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                scales[i] = computeScale(points[i], worldSize, maxScale);
            }
        }
    }

private:
    vec3 m_viewerPosition;
    T m_maxDistance;
    T m_minCellSize;
    T m_detailBias;
    bool m_adaptiveDetail;
};

} // namespace Morton
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_MORTON_HIERARCHICAL_MORTON_KEY_H_INCLUDED